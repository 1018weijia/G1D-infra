#!/usr/bin/env python3
"""Convert Unitree JSON episodes to a local LeRobot v3.0 dataset.

Writes with the official LeRobot Dataset API (add_frame / save_episode / finalize).
Camera names follow the physical G1 layout, not unitree_lerobot's software labels.

Numeric JSON fields under states / actions / tactiles are stored as separate
LeRobot columns that keep the original grouping, for example:
  - observation.state.left_arm.qpos
  - observation.state.left_arm_pose.qpos
  - action.right_arm.qpos
  - action.torso.qvel
Empty lists (typical unused qvel/torque) are skipped because they have no
values; every non-empty numeric leaf is kept. RGB images are encoded as MP4.

Bad source episodes (folder numbers such as episode_0007 -> 7) are kept in the
dataset and tagged on every frame as:
  - complementary_info.is_bad   True if the source episode is in the bad list
  - next.success                False if the source episode is in the bad list
  - complementary_info.source_episode_index   original episode_XXXX number
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import logging
import re
import shutil
import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CONSTANTS_PATH = SCRIPT_DIR / "constants.py"
DEFAULT_OFFICIAL_LEROBOT_SRC = REPO_ROOT / "3rd" / "lerobot" / "src"

STATE_PREFIX = "observation.state"
ACTION_PREFIX = "action"
TACTILE_PREFIX = "observation.tactile"
VECTOR_GROUPS = (
    ("states", STATE_PREFIX),
    ("actions", ACTION_PREFIX),
    ("tactiles", TACTILE_PREFIX),
)

# Physical cameras on this binocular-head G1.
# teleop_hand_and_arm.py writes color_2 from left_wrist_img and color_3 from
# right_wrist_img, and unitree_lerobot.constants copies that software naming.
# The recorded images show those two wrist streams are swapped vs the real
# hands, so this converter maps by what the cameras actually see:
#   color_0 left eye, color_1 right eye, color_2 right wrist, color_3 left wrist.
CAMERA_TO_IMAGE_KEY = {
    "color_0": "cam_left_high",
    "color_1": "cam_right_high",
    "color_2": "cam_right_wrist",
    "color_3": "cam_left_wrist",
}


def load_json(path: Path):
    raw = path.read_bytes()
    try:
        import orjson

        return orjson.loads(raw)
    except ImportError:
        return json.loads(raw)


def load_rgb_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_camera_images(
    episode_dir: Path,
    colors: dict,
    features: dict,
    camera_to_image_key: dict[str, str],
    pool: ThreadPoolExecutor | None,
) -> dict[str, np.ndarray]:
    jobs: list[tuple[str, Path]] = []
    for camera_name, rel_path in (colors or {}).items():
        if not rel_path:
            continue
        image_key = camera_feature_name(camera_name, camera_to_image_key)
        feature_name = f"observation.images.{image_key}"
        if feature_name not in features:
            continue
        jobs.append((feature_name, episode_dir / rel_path))
    if not jobs:
        return {}
    if pool is None or len(jobs) == 1:
        return {name: load_rgb_image(path) for name, path in jobs}
    futures = {name: pool.submit(load_rgb_image, path) for name, path in jobs}
    return {name: future.result() for name, future in futures.items()}


def _call_with_supported_kwargs(fn, **kwargs):
    params = inspect.signature(fn).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)
    filtered = {key: value for key, value in kwargs.items() if key in params}
    ignored = sorted(key for key in kwargs if key not in params)
    if ignored:
        print(f"Note: LeRobotDataset.create ignored unsupported args: {ignored}")
    return fn(**filtered)


def _patch_streaming_encoder_no_drop() -> None:
    """Conversion must not drop frames when the encoder queue fills up."""
    try:
        from lerobot.datasets.video_utils import StreamingVideoEncoder
    except ImportError:
        return

    def feed_frame(self, video_key: str, image: np.ndarray) -> None:
        if not self._episode_active:
            raise RuntimeError("No active episode. Call start_episode() first.")

        thread = self._threads[video_key]
        if not thread.is_alive():
            raise RuntimeError(f"Encoder thread for {video_key} is not alive")

        # Block instead of dropping. Source arrays are unique per imread, so skip copy.
        self._frame_queues[video_key].put(image)

    StreamingVideoEncoder.feed_frame = feed_frame  # type: ignore[method-assign]


def _patch_fast_software_codec_preset(preset_name: str) -> None:
    try:
        from lerobot.datasets import video_utils as video_utils
    except ImportError:
        return

    original = video_utils._get_codec_options

    def _get_codec_options(vcodec, g=2, crf=30, preset=None):
        options = original(vcodec, g, crf, preset)
        if vcodec in ("h264", "hevc") and "preset" not in options:
            options["preset"] = preset_name
        return options

    video_utils._get_codec_options = _get_codec_options


def _ensure_import_paths() -> None:
    official_src = str(DEFAULT_OFFICIAL_LEROBOT_SRC)
    if official_src not in sys.path and Path(official_src).exists():
        sys.path.insert(0, official_src)


def load_robot_configs(constants_path: Path | None = None):
    constants_path = Path(constants_path) if constants_path else DEFAULT_CONSTANTS_PATH
    if constants_path.is_dir():
        nested = constants_path / "unitree_lerobot" / "utils" / "constants.py"
        constants_path = nested if nested.exists() else constants_path / "constants.py"
    try:
        from unitree_lerobot.utils.constants import ROBOT_CONFIGS  # type: ignore

        return ROBOT_CONFIGS
    except ImportError:
        if not constants_path.exists():
            raise ImportError(
                "Cannot import robot constants. "
                f"Checked {constants_path}. Set --constants."
            ) from None
        spec = importlib.util.spec_from_file_location("unitree_robot_constants", constants_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module.ROBOT_CONFIGS


def parse_episode_ids(values: Iterable[str] | None) -> set[int]:
    ids: set[int] = set()
    if not values:
        return ids
    for raw in values:
        if raw is None:
            continue
        for token in re.split(r"[,\s]+", str(raw).strip()):
            if not token or token.startswith("#"):
                continue
            match = re.search(r"(\d+)$", token.replace("episode_", ""))
            if match is None:
                raise ValueError(f"Cannot parse episode id from '{token}'")
            ids.add(int(match.group(1)))
    return ids


def load_bad_ids_from_file(path: Path | None) -> set[int]:
    if path is None or not path.exists():
        return set()
    ids: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        ids |= parse_episode_ids([stripped])
    return ids


def source_episode_id(episode_dir: Path) -> int | None:
    match = re.search(r"(\d+)$", episode_dir.name)
    if match is None:
        return None
    return int(match.group(1))


def discover_episode_dirs(raw_dir: Path) -> list[Path]:
    """Accept either a task dir (episode_*) or a parent of task dirs."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {raw_dir}")

    direct = sorted(
        p
        for p in raw_dir.iterdir()
        if p.is_dir() and (p / "data.json").is_file()
    )
    if direct:
        return direct

    nested: list[Path] = []
    for task_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        nested.extend(
            sorted(
                p
                for p in task_dir.iterdir()
                if p.is_dir() and (p / "data.json").is_file()
            )
        )
    if not nested:
        raise FileNotFoundError(
            f"No episode folders with data.json found under {raw_dir}. "
            "Pass a path like ~/unitree_eai_environment/data/pick_place_100"
        )
    return nested


def as_float_vector(value) -> np.ndarray | None:
    """Return a 1-D float32 vector, or None when the JSON value has no numbers."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return np.asarray([value], dtype=np.float32)
    if isinstance(value, list):
        if not value:
            return None
        try:
            array = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None
        if array.size == 0:
            return None
        return array
    return None


def collect_numeric_leaves(node, prefix: str) -> dict[str, np.ndarray]:
    leaves: dict[str, np.ndarray] = {}
    if not isinstance(node, dict):
        return leaves
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            leaves.update(collect_numeric_leaves(value, path))
            continue
        vector = as_float_vector(value)
        if vector is not None:
            leaves[path] = vector
    return leaves


def collect_empty_paths(node, prefix: str) -> list[str]:
    paths: list[str] = []
    if not isinstance(node, dict):
        return paths
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            paths.extend(collect_empty_paths(value, path))
        elif isinstance(value, list) and len(value) == 0:
            paths.append(path)
    return paths


def pad_vector(vector: np.ndarray | None, dim: int) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    if vector is None or vector.size == 0:
        return out
    size = min(dim, int(vector.size))
    out[:size] = vector.reshape(-1)[:size]
    return out


def camera_feature_name(camera_name: str, camera_to_image_key: dict[str, str]) -> str:
    return camera_to_image_key.get(camera_name, camera_name)


def vector_feature_spec(path: str, dim: int) -> dict:
    short = path.split(".", 1)[-1]
    names = [f"{short}_{i}" for i in range(dim)] if dim > 1 else [short]
    return {
        "dtype": "float32",
        "shape": (dim,),
        "names": [names],
    }


def bool_feature(value: bool) -> np.ndarray:
    return np.asarray([bool(value)], dtype=np.bool_)


def int64_feature(value: int) -> np.ndarray:
    return np.asarray([int(value)], dtype=np.int64)


def build_features(
    vector_dims: dict[str, int],
    cameras: dict[str, tuple[int, int, int]],
    mode: str,
) -> dict:
    features = {
        "next.success": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        },
        "complementary_info.is_bad": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        },
        "complementary_info.source_episode_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
    }
    for path, dim in sorted(vector_dims.items()):
        features[path] = vector_feature_spec(path, dim)
    for cam, (height, width, channels) in cameras.items():
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (height, width, channels),
            "names": ["height", "width", "channel"],
        }
    return features


def scan_schema(
    episode_dirs: list[Path],
    camera_to_image_key: dict[str, str],
    fps_fallback: int,
    max_schema_episodes: int = 1,
) -> tuple[dict[str, int], dict[str, tuple[int, int, int]], int, list[str]]:
    vector_dims: dict[str, int] = {}
    cameras: dict[str, tuple[int, int, int]] = {}
    empty_always: set[str] | None = None
    fps: int | None = None

    if max_schema_episodes <= 0:
        schema_dirs = episode_dirs
    else:
        schema_dirs = episode_dirs[:max_schema_episodes]

    for episode_dir in tqdm(schema_dirs, desc="Scan schema"):
        episode_data = load_json(episode_dir / "data.json")
        if fps is None:
            fps = int(round(float(episode_data.get("info", {}).get("image", {}).get("fps", fps_fallback))))
        samples = episode_data.get("data") or []
        if not samples:
            continue

        episode_empty: set[str] = set()
        for sample in samples:
            for json_key, prefix in VECTOR_GROUPS:
                leaves = collect_numeric_leaves(sample.get(json_key), prefix)
                for path, vector in leaves.items():
                    vector_dims[path] = max(vector_dims.get(path, 0), int(vector.size))
                episode_empty.update(collect_empty_paths(sample.get(json_key), prefix))

            for camera_name, rel_path in (sample.get("colors") or {}).items():
                if not rel_path:
                    continue
                out_name = camera_feature_name(camera_name, camera_to_image_key)
                if out_name in cameras:
                    continue
                image = cv2.imread(str(episode_dir / rel_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Failed to read image: {episode_dir / rel_path}")
                height, width = image.shape[:2]
                cameras[out_name] = (height, width, 3)

        present = set(vector_dims)
        skipped = {path for path in episode_empty if path not in present}
        empty_always = skipped if empty_always is None else empty_always & skipped

    if not vector_dims:
        raise RuntimeError("No numeric state/action fields found in source JSON.")
    if not cameras:
        raise RuntimeError("Could not read any camera image while scanning schema.")

    return vector_dims, cameras, fps or fps_fallback, sorted(empty_always or [])


def extract_frame_vectors(sample: dict, vector_dims: dict[str, int]) -> dict[str, np.ndarray]:
    leaves: dict[str, np.ndarray] = {}
    for json_key, prefix in VECTOR_GROUPS:
        leaves.update(collect_numeric_leaves(sample.get(json_key), prefix))
    return {path: pad_vector(leaves.get(path), dim) for path, dim in vector_dims.items()}


def convert(args: argparse.Namespace) -> None:
    _ensure_import_paths()
    robot_configs = load_robot_configs(args.constants)
    if args.robot_type not in robot_configs:
        available = ", ".join(sorted(robot_configs))
        raise KeyError(f"Unknown robot_type '{args.robot_type}'. Available: {available}")

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        try:
            from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
        except ImportError:
            from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
    except ImportError as exc:
        raise ImportError(
            "Cannot import official lerobot. Use the unitree_lerobot conda env "
            f"(it already has huggingface_hub/datasets/av), keep {DEFAULT_OFFICIAL_LEROBOT_SRC.parent} "
            "on PYTHONPATH, and use a Python 3.10-compatible checkout such as v0.4.4. "
            "Current official main requires Python >= 3.12."
        ) from exc

    robot_cfg = robot_configs[args.robot_type]
    raw_dir = args.raw_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    episode_dirs = discover_episode_dirs(raw_dir)

    bad_ids = parse_episode_ids(args.bad_episodes)
    bad_ids |= load_bad_ids_from_file(args.bad_episodes_file)
    source_ids = {source_episode_id(p) for p in episode_dirs}
    unknown_bad = sorted(i for i in bad_ids if i not in source_ids)
    if unknown_bad:
        raise ValueError(
            "These bad episode ids were not found in the input folder: "
            f"{unknown_bad}. Use numbers from episode_XXXX, e.g. 7 for episode_0007."
        )

    if args.max_episodes is not None:
        episode_dirs = episode_dirs[: args.max_episodes]

    cv2.setNumThreads(1)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    vector_dims, cameras, fps, empty_fields = scan_schema(
        episode_dirs,
        CAMERA_TO_IMAGE_KEY,
        args.fps,
        max_schema_episodes=args.schema_episodes,
    )
    features = build_features(vector_dims, cameras, args.mode)

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    repo_id = args.repo_id or f"local/{output_dir.name}"
    create_params = inspect.signature(LeRobotDataset.create).parameters
    supports_streaming = "streaming_encoding" in create_params
    streaming_encoding = bool(args.streaming_encoding and args.mode == "video" and supports_streaming)
    if args.streaming_encoding and args.mode == "video" and not supports_streaming:
        print("Note: this LeRobot build has no streaming_encoding; falling back to PNG then encode.")

    # LeRobot v0.4.x crashes when batch_encoding_size > 1: save_episode looks up
    # self.meta.episodes[i]["data/chunk_index"] while episode parquet is still
    # buffered (meta.episodes is None). Encode each episode immediately instead.
    # https://github.com/huggingface/lerobot/issues/2509
    if args.batch_encoding_size != 1:
        print(
            f"Warning: --batch-encoding-size {args.batch_encoding_size} is unsafe "
            "(meta.episodes lookup). Using 1."
        )
        args.batch_encoding_size = 1

    if args.image_writer_threads is None:
        image_writer_threads = 0 if streaming_encoding else 16
    else:
        image_writer_threads = args.image_writer_threads

    if streaming_encoding:
        _patch_streaming_encoder_no_drop()
    if args.vcodec in ("h264", "hevc"):
        _patch_fast_software_codec_preset(args.preset or "veryfast")

    print(
        "Encode settings: "
        f"vcodec={args.vcodec}, streaming={streaming_encoding}, "
        f"encoder_threads={args.encoder_threads}, "
        f"image_writer_threads={image_writer_threads}, "
        f"image_load_workers={args.image_load_workers}"
    )
    print(
        "Note: torchcodec is a video decoder for training/loading, not encoding. "
        "It will not speed up this conversion, and LeRobot does not install it on aarch64/Jetson."
    )

    dataset = _call_with_supported_kwargs(
        LeRobotDataset.create,
        repo_id=repo_id,
        fps=fps,
        robot_type=args.robot_type,
        features=features,
        root=output_dir,
        use_videos=args.mode == "video",
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=image_writer_threads,
        batch_encoding_size=args.batch_encoding_size,
        vcodec=args.vcodec,
        streaming_encoding=streaming_encoding,
        encoder_queue_maxsize=args.encoder_queue_maxsize,
        encoder_threads=args.encoder_threads,
    )

    quality_rows = []
    converted = 0
    tagged_bad = 0
    skipped_bad = 0
    image_workers = max(0, int(args.image_load_workers))
    image_pool = ThreadPoolExecutor(max_workers=image_workers) if image_workers > 1 else None

    try:
        for episode_dir in tqdm(episode_dirs, desc="Episodes"):
            source_id = source_episode_id(episode_dir)
            is_bad = source_id in bad_ids
            if is_bad and args.skip_bad:
                skipped_bad += 1
                quality_rows.append(
                    {
                        "source_episode": source_id,
                        "source_dir": episode_dir.name,
                        "lerobot_episode_index": None,
                        "is_bad": True,
                        "skipped": True,
                    }
                )
                continue

            episode_data = load_json(episode_dir / "data.json")
            samples = episode_data["data"]
            task = args.task or episode_data.get("text", {}).get("goal") or raw_dir.name
            src_index = source_id if source_id is not None else converted

            for sample in tqdm(samples, desc=episode_dir.name, leave=False):
                frame = extract_frame_vectors(sample, vector_dims)
                frame.update(
                    {
                        "next.success": bool_feature(not is_bad),
                        "complementary_info.is_bad": bool_feature(is_bad),
                        "complementary_info.source_episode_index": int64_feature(src_index),
                        "task": task,
                    }
                )
                frame.update(
                    load_camera_images(
                        episode_dir,
                        sample.get("colors") or {},
                        features,
                        CAMERA_TO_IMAGE_KEY,
                        image_pool,
                    )
                )
                dataset.add_frame(frame)

            dataset.save_episode()
            quality_rows.append(
                {
                    "source_episode": source_id,
                    "source_dir": episode_dir.name,
                    "lerobot_episode_index": converted,
                    "is_bad": is_bad,
                    "skipped": False,
                    "num_frames": len(samples),
                }
            )
            if is_bad:
                tagged_bad += 1
            converted += 1
    finally:
        if image_pool is not None:
            image_pool.shutdown(wait=True)

    dataset.finalize()

    quality_path = output_dir / "meta" / "episode_quality.json"
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality_payload = {
        "codebase_version": CODEBASE_VERSION,
        "index_convention": "source_episode is the number in episode_XXXX (episode_0007 -> 7)",
        "robot_type": args.robot_type,
        "raw_dir": str(raw_dir),
        "vector_features": {path: dim for path, dim in sorted(vector_dims.items())},
        "empty_json_fields_skipped": empty_fields,
        "cameras": sorted(cameras),
        "encode": {
            "vcodec": args.vcodec,
            "streaming_encoding": streaming_encoding,
            "encoder_threads": args.encoder_threads,
            "image_writer_threads": image_writer_threads,
            "image_load_workers": args.image_load_workers,
        },
        "bad_source_episodes": sorted(bad_ids),
        "converted_episodes": converted,
        "tagged_bad_episodes": tagged_bad,
        "skipped_bad_episodes": skipped_bad,
        "episodes": quality_rows,
    }
    quality_path.write_text(json.dumps(quality_payload, indent=2), encoding="utf-8")

    print(f"Wrote LeRobot {CODEBASE_VERSION} dataset to {output_dir}")
    print(f"Converted {converted} episodes, tagged bad={tagged_bad}, skipped bad={skipped_bad}")
    print(f"Vector features: {len(vector_dims)} columns (JSON grouping preserved)")
    print(f"Quality map: {quality_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True, help="Unitree JSON task dir, e.g. pick_place_100")
    parser.add_argument("--output-dir", type=Path, required=True, help="Local LeRobot dataset output directory")
    parser.add_argument("--repo-id", type=str, default=None, help="Optional local repo id, default local/<output name>")
    parser.add_argument(
        "--robot-type",
        type=str,
        default="Unitree_G1_MoveibleLift_Dex1_NoUseWaist",
        help="Stored in dataset metadata; vector fields are taken from JSON.",
    )
    parser.add_argument(
        "--bad-episodes",
        action="append",
        default=None,
        help="Bad source episode numbers, e.g. --bad-episodes '7,15,23' or --bad-episodes 7 --bad-episodes 15",
    )
    parser.add_argument("--bad-episodes-file", type=Path, default=None, help="Optional file with one id per line")
    parser.add_argument("--skip-bad", action="store_true", help="Drop bad episodes instead of converting+tagging them")
    parser.add_argument("--task", type=str, default=None, help="Override language task; default uses JSON text.goal")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--mode", choices=["video", "image"], default="video")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None, help="Debug: convert only the first N episodes")
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=None,
        help="PNG writer threads. Default 0 with streaming encoding, 16 otherwise.",
    )
    parser.add_argument(
        "--batch-encoding-size",
        "--batch_encoding_size",
        dest="batch_encoding_size",
        type=int,
        default=1,
        help="Episodes to accumulate before video encode. Values >1 are ignored.",
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default="h264",
        help="Video codec: h264 (fast, default), libsvtav1 (smaller/slower), auto, h264_nvenc, ...",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Software encoder preset. Default veryfast for h264/hevc.",
    )
    parser.add_argument(
        "--streaming-encoding",
        dest="streaming_encoding",
        action="store_true",
        default=True,
        help="Encode MP4 while reading frames, skip the PNG round-trip (default).",
    )
    parser.add_argument(
        "--no-streaming-encoding",
        dest="streaming_encoding",
        action="store_false",
        help="Write PNG files first, then encode at episode end.",
    )
    parser.add_argument(
        "--encoder-threads",
        type=int,
        default=1,
        help="Threads per camera encoder. Keep 1 when encoding 4 cameras on an 8-core Jetson.",
    )
    parser.add_argument(
        "--encoder-queue-maxsize",
        type=int,
        default=120,
        help="Per-camera encoder queue. Conversion blocks instead of dropping frames.",
    )
    parser.add_argument(
        "--image-load-workers",
        type=int,
        default=4,
        help="Parallel JPEG readers. 4 matches the G1 camera count.",
    )
    parser.add_argument(
        "--schema-episodes",
        type=int,
        default=1,
        help="How many episodes to scan for vector/camera schema. 0 means all.",
    )
    parser.add_argument("--constants", type=Path, default=DEFAULT_CONSTANTS_PATH,
                        help="Path to constants.py (or a unitree_lerobot checkout).")
    return parser


if __name__ == "__main__":
    convert(build_parser().parse_args())
