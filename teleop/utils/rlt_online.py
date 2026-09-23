"""Stage-2 online RL messages for the G1-D policy client.

Field names follow RealWorld-RLinf ``rlt-online-rl/v1`` (``act``,
``transition``, ``discard``). The robot stays in raw AbsQpos. Images are a
stitched RGB frame; ZMQ carries the array, WebSocket carries a JPEG.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

REQUEST_KEY = "rlt/request"
REQUEST_ACT = "act"
REQUEST_TRANSITION = "transition"
REQUEST_DISCARD = "discard"
REQUEST_REWIND_EXIT = "rewind_exit_correction"
REQUEST_REWIND_CREDIT = "rewind_credit_correction"
REQUEST_EPISODE_END = "episode_end"
ACTION_SPACE_ROBOT = "robot"
# remote-franka stage2.server.franka: rewind_physical_exit_reward / credit.
REWIND_TERMINAL_REWARD = -0.2
REWIND_PREFIX_REWARD = 0.1
PROGRESS_REWARD = 0.5
SUCCESS_REWARD = 1.0
# The robot uplink is about 1 Mbit/s. A raw 384x320 frame is 369 KB, JPEG q90
# is about 20 KB, so a 64-step transition drops from ~24 MB to ~1.3 MB.
JPEG_QUALITY = 90
# Same reorder as PolicyAdapter: raw [L7, R7, LG, RG] -> [L7, LG, R7, RG].
_REORDER_FROM_RAW = [0, 1, 2, 3, 4, 5, 6, 14, 7, 8, 9, 10, 11, 12, 13, 15]
# Residual actor can push a gripper slightly past the SFT 5.5 safety cap.
# Joints stay inside the existing arm limit.
RL_MAX_ABS_ARM_Q = 3.5
RL_MAX_ABS_GRIPPER_Q = 6.5


def transition_fields(
    num_actions: int,
    outcome: Optional[str],
    intervention: bool = False,
    rewards: Optional[np.ndarray] = None,
) -> dict:
    """Reward and bootstrap flags for one executed chunk.

    A success or failure ends the episode and cuts bootstrap. When the caller
    already stamped per-step scores, those values are kept. Otherwise a success
    still writes +1 on the last step.
    """
    done = (not intervention) and outcome in ("success", "failure")
    if rewards is None:
        reward_row = chunk_rewards(num_actions, outcome if done else None)
    else:
        reward_row = np.asarray(rewards, dtype=np.float32).reshape(-1)[: int(num_actions)].copy()
    return {
        "rewards": reward_row,
        "done": done,
        "bootstrap_mask": 0.0 if done else 1.0,
        "intervention": bool(intervention),
    }


def outcome_ends_without_chunk(outcome, chunk_open: bool, queued_steps: int) -> bool:
    """Y/N with nothing executing ends now, instead of riding the next chunk."""
    return outcome in ("success", "failure") and not chunk_open and int(queued_steps) <= 0


def resolve_observation(holder) -> Optional[dict]:
    """Observation from a stitch holder, or None if it was skipped or failed."""
    if not isinstance(holder, dict):
        return None
    return holder.get("obs")


def resolve_step_observations(entries, length: int) -> list:
    """Exactly ``length`` step observations; steps without one stay None."""
    resolved = [resolve_observation(entry) for entry in list(entries or [])[: int(length)]]
    return resolved + [None] * (int(length) - len(resolved))


def chunk_rewards(length: int, outcome: Optional[str]) -> np.ndarray:
    """Per-step rewards. Only a success writes +1 on the last step."""
    rewards = np.zeros(int(length), dtype=np.float32)
    if outcome == "success" and rewards.size:
        rewards[-1] = 1.0
    return rewards


def encode_jpeg_rgb(frame_rgb: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    import cv2

    image = np.asarray(frame_rgb)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    )
    if not ok:
        raise RuntimeError("failed to encode an RL frame as JPEG")
    return buf.tobytes()


def observation_zmq(frame_rgb: np.ndarray, state: np.ndarray, prompt: str) -> dict:
    return {
        "observation/state": np.asarray(state, dtype=np.float32).reshape(-1),
        "observation/image_jpeg": encode_jpeg_rgb(frame_rgb),
        "prompt": str(prompt),
    }


def observation_ws(frame_rgb: np.ndarray, state: np.ndarray, prompt: str, jpeg_b64: str) -> dict:
    return {
        "observation/state": np.asarray(state, dtype=np.float32).reshape(-1).tolist(),
        "observation/image_jpeg": jpeg_b64,
        "prompt": str(prompt),
    }


class RLTRollout:
    """Tracks the one action chunk currently being executed."""

    def __init__(self):
        self.episode_id = 0
        self.next_chunk_id = 0
        self.stored_chunks = 0
        self.open: Optional[dict] = None
        self.outcome: Optional[str] = None
        # The server reports its window stride with every act reply.
        self.step_obs_stride = 4
        self.stitch_failures = 0
        self._stitch_queue: queue.Queue = queue.Queue()
        self._stitch_thread: Optional[threading.Thread] = None

    def begin_episode(self) -> int:
        self.episode_id += 1
        self.next_chunk_id = 0
        self.stored_chunks = 0
        self.open = None
        self.outcome = None
        return self.episode_id

    def note_stored(self) -> None:
        self.stored_chunks += 1

    def note_outcome(self, outcome: str) -> None:
        if outcome not in ("success", "failure"):
            raise ValueError(f"outcome must be success or failure, got {outcome!r}")
        if self.outcome is None:
            self.outcome = outcome

    def take_outcome(self) -> Optional[str]:
        outcome = self.outcome
        self.outcome = None
        return outcome

    def accept_chunk(self, transition_id: str, actions: np.ndarray, queue_len: int) -> dict:
        chunk_id = self.next_chunk_id
        self.next_chunk_id += 1
        self.open = {
            "transition_id": str(transition_id),
            "actions": np.asarray(actions, dtype=np.float32).copy(),
            "episode_id": int(self.episode_id),
            "chunk_id": int(chunk_id),
            "queue_len": int(queue_len),
            "remaining": int(queue_len),
            "rewards": np.zeros(int(queue_len), dtype=np.float32),
            "step_observations": [],
        }
        return self.open

    def _ensure_stitcher(self) -> None:
        if self._stitch_thread is not None:
            return
        self._stitch_thread = threading.Thread(target=self._stitch_loop, daemon=True)
        self._stitch_thread.start()

    def _stitch_loop(self) -> None:
        while True:
            item = self._stitch_queue.get()
            try:
                if item is None:
                    return
                holder, stitch, head, left, right, state = item
                holder["obs"] = {
                    "observation/image_jpeg": encode_jpeg_rgb(stitch(head, left, right)),
                    "observation/state": state,
                    "prompt": "",
                }
            except Exception:
                self.stitch_failures += 1
                if self.stitch_failures == 1:
                    logger.exception("RL step observation stitch failed; later failures are counted only")
            finally:
                self._stitch_queue.task_done()

    def queue_observation(self, cameras, stitch, state) -> dict:
        """Stitch one observation off the control thread. ``holder["obs"]`` fills in later."""
        holder: dict = {"obs": None}
        head, left, right = cameras
        self._ensure_stitcher()
        self._stitch_queue.put((
            holder, stitch, head, left, right,
            np.asarray(state, dtype=np.float32).reshape(-1).copy(),
        ))
        return holder

    def step_obs_wanted(self, index: int) -> bool:
        """The server builds windows at offsets stride, 2*stride, ... < chunk."""
        stride = int(self.step_obs_stride)
        return stride > 0 and index > 0 and index % stride == 0

    def on_step(self, frame_rgb=None, state=None, cameras=None, stitch=None) -> None:
        if self.open is None or self.open["remaining"] <= 0:
            return
        chunk = self.open
        index = int(chunk["queue_len"]) - int(chunk["remaining"])
        chunk["remaining"] -= 1
        entry = None
        if self.step_obs_wanted(index) and state is not None:
            if cameras is not None and stitch is not None:
                entry = self.queue_observation(cameras, stitch, state)
            elif frame_rgb is not None:
                entry = {"obs": {
                    "observation/image_jpeg": encode_jpeg_rgb(frame_rgb),
                    "observation/state": np.asarray(state, dtype=np.float32).reshape(-1).copy(),
                    "prompt": "",
                }}
        chunk["step_observations"].append(entry)

    def _flush_step_images(self) -> None:
        if self._stitch_thread is None:
            return
        self._stitch_queue.join()

    def add_step_reward(self, value: float) -> None:
        """Add ``value`` onto the step that just ran."""
        if self.open is None:
            return
        index = int(self.open["queue_len"]) - int(self.open["remaining"]) - 1
        if index < 0 or index >= int(self.open["rewards"].shape[0]):
            return
        self.open["rewards"][index] += float(value)

    def chunk_finished(self) -> bool:
        return self.open is not None and self.open["remaining"] <= 0

    def wait_step_images(self) -> None:
        """Block until background stitching started before this call has finished."""
        self._flush_step_images()

    def detach_open(self) -> Optional[dict]:
        chunk = self.open
        self.open = None
        return chunk

    def take_open(self) -> Optional[dict]:
        self._flush_step_images()
        return self.detach_open()

    def interrupt(self):
        """Drop a chunk that rollback cut off.

        Returns ``("discard", chunk)`` when no step ran, or
        ``("transition", chunk)`` when the human cut in after some steps.
        """
        chunk = self.detach_open()
        if chunk is None:
            return None
        executed = int(chunk["queue_len"]) - int(chunk["remaining"])
        if executed <= 0:
            return ("discard", chunk)
        chunk["intervention"] = True
        chunk["executed_steps"] = executed
        return ("transition", chunk)


def rewind_frame_count(executed_steps: int, chunk_len: int) -> int:
    """How many recorded commands to play backward for a one-chunk rewind.

    The newest command is the pose already being held, so it is not replayed.
    A chunk that has started rewinds only the steps it executed. Otherwise the
    previous full chunk is rewound.
    """
    chunk_len = max(1, int(chunk_len))
    steps = int(executed_steps) if int(executed_steps) > 0 else chunk_len
    return max(0, steps - 1)


def rewind_plan(frames: int, chunk_len: int, stored_chunks: int, include_current: bool) -> Optional[dict]:
    """Map one physical rollback onto a one-chunk RLinf rewind correction.

    The bad branch is always the single latest chunk. ``chunk_len`` is accepted
    so callers can keep passing it; the playback length is chosen separately by
    ``rewind_frame_count``. No motion and no open chunk falls back to
    ``rewind_credit``.
    """
    del chunk_len
    available = int(stored_chunks) + (1 if include_current else 0)
    if available <= 0:
        return None
    if int(frames) <= 0 and not include_current:
        return {
            "mode": "credit",
            "chunks": 1,
            "terminal_reward": REWIND_TERMINAL_REWARD,
            "prefix_reward": REWIND_PREFIX_REWARD,
        }
    return {
        "mode": "exit",
        "chunks": 1,
        "terminal_reward": REWIND_TERMINAL_REWARD,
        "prefix_reward": 0.0,
    }


def qpos_command(arm_q, left_grip: float, right_grip: float) -> np.ndarray:
    """One commanded AbsQpos step, in the same layout the policy executes."""
    raw = np.zeros(16, dtype=np.float32)
    arm = np.asarray(arm_q, dtype=np.float32).reshape(-1)
    raw[0:7] = arm[0:7]
    raw[7:14] = arm[7:14]
    raw[14] = float(left_grip)
    raw[15] = float(right_grip)
    return raw[_REORDER_FROM_RAW]


def _rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    relative = np.asarray(a, dtype=float)[:3, :3].T @ np.asarray(b, dtype=float)[:3, :3]
    return float(np.arccos(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)))


class TeleopMotionGate:
    """remote-franka ``GelloTakeover.should_count`` for both G1-D arms.

    A teleop tick becomes a takeover chunk step only if either arm moved past
    the deadband since the last counted step. Holding still is still executed,
    but it is never stored as an intervention action.
    """

    POS_DEADBAND_M = 0.0015
    ROT_DEADBAND_RAD = 0.01
    JOINT_DEADBAND_RAD = 0.02
    # Franka compares a 0/1 gripper against 0.25. G1-D grippers span about 5.5.
    GRIP_DEADBAND = 0.25 * 5.5

    def __init__(self):
        self._last: Optional[tuple] = None

    @property
    def seeded(self) -> bool:
        return self._last is not None

    def clear(self) -> None:
        self._last = None

    def reset(self, command, left_pose, right_pose) -> None:
        """Use the pose held at handoff as the baseline; it is not a step."""
        self._last = (
            np.asarray(command, dtype=np.float32).reshape(-1).copy(),
            np.asarray(left_pose, dtype=float).copy(),
            np.asarray(right_pose, dtype=float).copy(),
        )

    def should_count(self, command, left_pose, right_pose, *, commit: bool = True) -> bool:
        cmd = np.asarray(command, dtype=np.float32).reshape(-1)
        if self._last is None:
            if commit:
                self.reset(cmd, left_pose, right_pose)
            return False
        last_cmd, last_left, last_right = self._last
        moved = False
        for joints, grip, pose, last_pose in (
            (slice(0, 7), 7, left_pose, last_left),
            (slice(8, 15), 15, right_pose, last_right),
        ):
            pose = np.asarray(pose, dtype=float)
            if (np.linalg.norm(pose[:3, 3] - last_pose[:3, 3]) >= self.POS_DEADBAND_M
                    or _rotation_angle(last_pose, pose) >= self.ROT_DEADBAND_RAD
                    or np.linalg.norm(cmd[joints] - last_cmd[joints]) >= self.JOINT_DEADBAND_RAD
                    or abs(float(cmd[grip]) - float(last_cmd[grip])) > self.GRIP_DEADBAND):
                moved = True
                break
        if moved and commit:
            self.reset(cmd, left_pose, right_pose)
        return moved


class TeleopChunker:
    """Cut counted teleop steps into chunks locally, without waiting on the server.

    Each chunk keeps the observation at its first step. The next chunk's first
    observation is this chunk's next observation, as in remote-franka's
    continuous Gello takeover. Chunks are uploaded later as act + transition.
    """

    def __init__(self, chunk_len: int = 64):
        self.chunk_len = int(chunk_len)
        self._chunk: Optional[dict] = None

    @property
    def active(self) -> bool:
        return self._chunk is not None

    @property
    def steps(self) -> int:
        return 0 if self._chunk is None else len(self._chunk["actions"])

    @property
    def full(self) -> bool:
        return self._chunk is not None and self.steps >= self.chunk_len

    def open(self, start_obs, episode_id: int, chunk_id: int) -> None:
        self._chunk = {
            "start": start_obs,
            "episode_id": int(episode_id),
            "chunk_id": int(chunk_id),
            "actions": [],
            "step_observations": [],
        }

    def push(self, action, step_obs=None) -> bool:
        """Add one counted step. Returns True once the chunk holds ``chunk_len`` steps."""
        if self._chunk is None:
            raise RuntimeError("teleop chunk is not open")
        if self.full:
            raise RuntimeError("teleop chunk is full; close it first")
        self._chunk["actions"].append(np.asarray(action, dtype=np.float32).reshape(-1).copy())
        self._chunk["step_observations"].append(step_obs)
        return self.full

    def close(self, next_obs) -> Optional[dict]:
        """Finish the chunk. Returns None when it has no steps."""
        chunk, self._chunk = self._chunk, None
        if chunk is None or not chunk["actions"]:
            return None
        chunk["actions"] = np.stack(chunk["actions"], axis=0).astype(np.float32)
        chunk["next"] = next_obs
        return chunk
