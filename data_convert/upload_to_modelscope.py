#!/usr/bin/env python3
"""Upload a local LeRobot dataset folder to a ModelScope dataset repository.

Usage:
  python upload_to_modelscope.py --token ms-xxx --repo user/pick_place_100 \\
      --local-dir /home/unitree/data2lerobot/datasets/pick_place_100

  python upload_to_modelscope.py ms-xxx user/pick_place_100 \\
      /home/unitree/data2lerobot/datasets/pick_place_100

Token can also come from MODELSCOPE_API_TOKEN / MODELSCOPE_TOKEN.
If --repo is only a name (no owner/), the script prepends the token's username.
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path

DEFAULT_IGNORE = [
    "**/.git/**",
    "**/.git",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.DS_Store",
    "**/.cache/**",
    "**/Thumbs.db",
]
EMPTY_IMAGE_IGNORE = ["images/**", "images"]
LEROBOT_MARKERS = ("meta/info.json", "meta", "data", "videos")


def _die(msg: str, code: int = 1) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local LeRobot dataset to ModelScope Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --token ms-xxxx --repo alice/pick_place_100 \\
      --local-dir /home/unitree/data2lerobot/datasets/pick_place_100

  %(prog)s ms-xxxx alice/pick_place_100 \\
      /home/unitree/data2lerobot/datasets/pick_place_100

  MODELSCOPE_API_TOKEN=ms-xxxx %(prog)s --repo alice/pick_place_100 \\
      --local-dir /home/unitree/data2lerobot/datasets/pick_place_100
""",
    )
    parser.add_argument("positional", nargs="*", help=argparse.SUPPRESS)
    parser.add_argument(
        "--token",
        default=os.environ.get("MODELSCOPE_API_TOKEN")
        or os.environ.get("MODELSCOPE_TOKEN")
        or "",
        help="ModelScope access token (or MODELSCOPE_API_TOKEN / MODELSCOPE_TOKEN)",
    )
    parser.add_argument(
        "--repo",
        default="",
        help="Dataset repo id, e.g. username/pick_place_100",
    )
    parser.add_argument(
        "--local-dir",
        "--path",
        dest="local_dir",
        default="",
        help="Local LeRobot dataset folder",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private (default: public)",
    )
    parser.add_argument(
        "--commit-message",
        default="",
        help="Commit message (default: auto)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel upload workers (default: 4)",
    )
    parser.add_argument(
        "--include-images",
        action="store_true",
        help="Also upload the images/ staging folder (skipped by default)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files and size, do not upload",
    )
    parser.add_argument(
        "--no-create",
        action="store_true",
        help="Do not create the remote repo if it is missing",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("MODELSCOPE_ENDPOINT", "https://www.modelscope.cn"),
        help="ModelScope endpoint (default: https://www.modelscope.cn)",
    )
    args = parser.parse_args()

    pos = list(args.positional)
    if not args.token and pos:
        args.token = pos.pop(0)
    if not args.repo and pos:
        args.repo = pos.pop(0)
    if not args.local_dir and pos:
        args.local_dir = pos.pop(0)
    if pos:
        _die(f"Unexpected extra arguments: {' '.join(pos)}")
    return args


def human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{n} B"


def iter_upload_files(root: Path, ignore_images: bool) -> list[Path]:
    files: list[Path] = []
    skip_dirs = {".git", "__pycache__", ".cache"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        rel_dir = Path(dirpath).relative_to(root)
        if ignore_images and (rel_dir == Path("images") or rel_dir.parts[:1] == ("images",)):
            dirnames[:] = []
            continue
        for name in filenames:
            if name in {".DS_Store", "Thumbs.db"} or name.endswith(".pyc"):
                continue
            files.append(Path(dirpath) / name)
    files.sort()
    return files


def looks_like_lerobot(root: Path) -> bool:
    return (root / "meta" / "info.json").is_file()


def import_hub_api():
    try:
        from modelscope_hub import HubApi  # type: ignore

        return HubApi, "modelscope_hub"
    except ImportError:
        pass
    try:
        from modelscope.hub.api import HubApi  # type: ignore

        return HubApi, "modelscope"
    except ImportError:
        _die(
            "Neither modelscope-hub nor modelscope is installed.\n"
            "Install one of:\n"
            "  pip install modelscope-hub\n"
            "  pip install modelscope"
        )


def make_api(hub_cls, source: str, token: str, endpoint: str):
    kwargs = {}
    sig = inspect.signature(hub_cls.__init__)
    if "token" in sig.parameters:
        kwargs["token"] = token
    if "endpoint" in sig.parameters:
        kwargs["endpoint"] = endpoint
    api = hub_cls(**kwargs)
    if source == "modelscope" and hasattr(api, "login"):
        api.login(token)
    return api


def whoami_username(api) -> str:
    if hasattr(api, "whoami"):
        info = api.whoami()
        if isinstance(info, dict):
            return str(info.get("Name") or info.get("name") or info.get("username") or "")
        for attr in ("username", "name", "Name"):
            value = getattr(info, attr, None)
            if value:
                return str(value)
    if hasattr(api, "get_current_username"):
        value = api.get_current_username()
        if value:
            return str(value)
    return ""


def normalize_repo_id(repo: str, username: str) -> str:
    repo = repo.strip().strip("/")
    if not repo:
        _die("Repo name is empty")
    if "/" not in repo:
        if not username:
            _die("Repo has no owner/. Pass username/repo, or use a token that can resolve whoami")
        repo = f"{username}/{repo}"
    parts = repo.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        _die(f"Invalid repo id {repo!r}. Expected owner/name")
    return repo


def call_filtered(fn, **kwargs):
    """Call fn with only kwargs it actually accepts."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(**kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return fn(**filtered)


def ensure_dataset_repo(api, repo_id: str, private: bool, no_create: bool) -> None:
    visibility = "private" if private else "public"
    if no_create:
        return
    if hasattr(api, "create_repo"):
        try:
            call_filtered(
                api.create_repo,
                repo_id=repo_id,
                repo_type="dataset",
                visibility=visibility,
                license="apache-2.0",
                exist_ok=True,
                chinese_name=repo_id.split("/")[-1],
            )
            print(f"Remote dataset repo ready: {repo_id} ({visibility})")
            return
        except TypeError:
            try:
                api.create_repo(repo_id, "dataset", visibility=visibility, exist_ok=True)
                print(f"Remote dataset repo ready: {repo_id} ({visibility})")
                return
            except Exception as exc:
                print(f"create_repo warning: {exc}")
        except Exception as exc:
            name = type(exc).__name__
            if "AlreadyExists" in name or "exist" in str(exc).lower():
                print(f"Remote dataset repo already exists: {repo_id}")
                return
            print(f"create_repo warning: {exc}")

    if hasattr(api, "create_dataset"):
        owner, name = repo_id.split("/", 1)
        vis = 5 if private else 1
        try:
            from modelscope.hub.constants import DatasetVisibility, Licenses  # type: ignore

            vis = DatasetVisibility.PRIVATE if private else DatasetVisibility.PUBLIC
            license_id = Licenses.APACHE_V2
        except Exception:
            license_id = "Apache License 2.0"
        try:
            call_filtered(
                api.create_dataset,
                dataset_name=name,
                namespace=owner,
                chinese_name=name,
                license=license_id,
                visibility=vis,
            )
            print(f"Created dataset repo: {repo_id} ({visibility})")
        except Exception as exc:
            if "exist" in str(exc).lower() or "Already" in type(exc).__name__:
                print(f"Remote dataset repo already exists: {repo_id}")
            else:
                print(f"create_dataset warning: {exc}")


def upload_folder(api, repo_id: str, local_dir: Path, args: argparse.Namespace) -> None:
    ignore = list(DEFAULT_IGNORE)
    if not args.include_images:
        ignore.extend(EMPTY_IMAGE_IGNORE)
    commit = args.commit_message or f"Upload LeRobot dataset from {local_dir.name}"
    kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "folder_path": str(local_dir),
        "path_in_repo": "",
        "commit_message": commit,
        "ignore_patterns": ignore,
        "allow_patterns": None,
        "max_workers": args.max_workers,
        "revision": "master",
    }

    if not hasattr(api, "upload_folder"):
        _die("Installed ModelScope SDK has no upload_folder(). Upgrade modelscope-hub or modelscope.")

    try:
        result = call_filtered(api.upload_folder, **kwargs)
        return result
    except TypeError:
        pass

    # modelscope_hub positional style: upload_folder(repo_id, repo_type, folder, path_in_repo=...)
    try:
        return api.upload_folder(
            repo_id,
            "dataset",
            str(local_dir),
            path_in_repo="",
            commit_message=commit,
            ignore_patterns=ignore,
            max_workers=args.max_workers,
        )
    except TypeError:
        return api.upload_folder(
            repo_id=repo_id,
            folder_path=str(local_dir),
            repo_type="dataset",
            commit_message=commit,
        )


def main() -> None:
    args = parse_args()
    if not args.token:
        _die("Missing token. Pass --token, a positional token, or MODELSCOPE_API_TOKEN")
    if not args.repo:
        _die("Missing repo. Pass --repo username/name or a positional repo id")
    if not args.local_dir:
        _die("Missing local folder. Pass --local-dir PATH")

    local_dir = Path(args.local_dir).expanduser().resolve()
    if not local_dir.is_dir():
        _die(f"Local directory not found: {local_dir}")

    ignore_images = not args.include_images
    files = iter_upload_files(local_dir, ignore_images=ignore_images)
    if not files:
        _die(f"No files to upload under {local_dir}")
    total = sum(p.stat().st_size for p in files)

    print(f"Local dir : {local_dir}")
    print(f"Files     : {len(files)}")
    print(f"Size      : {human_size(total)}")
    if looks_like_lerobot(local_dir):
        print("Format    : LeRobot dataset (meta/info.json found)")
    else:
        print("Warning   : meta/info.json not found; uploading the folder as-is")
    if ignore_images and (local_dir / "images").exists():
        print("Skip      : images/ (pass --include-images to upload it)")
    print()

    if args.dry_run:
        shown = files[:30]
        for path in shown:
            rel = path.relative_to(local_dir)
            print(f"  {rel}  ({human_size(path.stat().st_size)})")
        if len(files) > len(shown):
            print(f"  ... and {len(files) - len(shown)} more files")
        print("\nDry run only. No upload.")
        return

    hub_cls, source = import_hub_api()
    print(f"SDK       : {source}")
    api = make_api(hub_cls, source, args.token, args.endpoint)
    username = whoami_username(api)
    if username:
        print(f"User      : {username}")
    repo_id = normalize_repo_id(args.repo, username)
    print(f"Repo      : {repo_id}")
    print()

    ensure_dataset_repo(api, repo_id, private=args.private, no_create=args.no_create)
    print("Uploading... interrupted runs can be resumed by running the same command.")
    upload_folder(api, repo_id, local_dir, args)

    page = args.endpoint.rstrip("/")
    if "modelscope.ai" in page:
        url = f"{page}/datasets/{repo_id}"
    else:
        url = f"https://www.modelscope.cn/datasets/{repo_id}"
    print()
    print("Upload finished.")
    print(f"Dataset URL: {url}")


if __name__ == "__main__":
    main()
