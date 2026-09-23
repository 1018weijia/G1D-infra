"""Stage-2 online RL messages for the G1-D policy client.

Field names follow RealWorld-RLinf ``rlt-online-rl/v1`` (``act``,
``transition``, ``discard``). The robot stays in raw AbsQpos. Images are a
stitched RGB frame; ZMQ carries the array, WebSocket carries a JPEG.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

REQUEST_KEY = "rlt/request"
REQUEST_ACT = "act"
REQUEST_TRANSITION = "transition"
REQUEST_DISCARD = "discard"
ACTION_SPACE_ROBOT = "robot"
# Same reorder as PolicyAdapter: raw [L7, R7, LG, RG] -> [L7, LG, R7, RG].
_REORDER_FROM_RAW = [0, 1, 2, 3, 4, 5, 6, 14, 7, 8, 9, 10, 11, 12, 13, 15]
# Residual actor can push a gripper slightly past the SFT 5.5 safety cap.
# Joints stay inside the existing arm limit.
RL_MAX_ABS_ARM_Q = 3.5
RL_MAX_ABS_GRIPPER_Q = 6.5


def transition_fields(num_actions: int, outcome: Optional[str], intervention: bool = False) -> dict:
    """Reward and bootstrap flags for one executed chunk.

    Success writes +1 on the last step and ends the episode. Failure ends
    the episode with zeros. A rollback after some steps is an intervention:
    reward stays 0 and the critic may still bootstrap.
    """
    done = (not intervention) and outcome in ("success", "failure")
    return {
        "rewards": chunk_rewards(num_actions, outcome if done else None),
        "done": done,
        "bootstrap_mask": 0.0 if done else 1.0,
        "intervention": bool(intervention),
    }


def chunk_rewards(length: int, outcome: Optional[str]) -> np.ndarray:
    """Per-step rewards. Only a success writes +1 on the last step."""
    rewards = np.zeros(int(length), dtype=np.float32)
    if outcome == "success" and rewards.size:
        rewards[-1] = 1.0
    return rewards


def observation_zmq(frame_rgb: np.ndarray, state: np.ndarray, prompt: str) -> dict:
    image = np.asarray(frame_rgb)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return {
        "observation/state": np.asarray(state, dtype=np.float32).reshape(-1),
        "observation/image": image,
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
        self.open: Optional[dict] = None
        self.outcome: Optional[str] = None

    def begin_episode(self) -> int:
        self.episode_id += 1
        self.next_chunk_id = 0
        self.open = None
        self.outcome = None
        return self.episode_id

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
        }
        return self.open

    def on_step(self) -> None:
        if self.open is not None and self.open["remaining"] > 0:
            self.open["remaining"] -= 1

    def chunk_finished(self) -> bool:
        return self.open is not None and self.open["remaining"] <= 0

    def take_open(self) -> Optional[dict]:
        chunk = self.open
        self.open = None
        return chunk

    def interrupt(self):
        """Drop a chunk that rollback cut off.

        Returns ``("discard", chunk)`` when no step ran, or
        ``("transition", chunk)`` when the human cut in after some steps.
        """
        chunk = self.take_open()
        if chunk is None:
            return None
        executed = int(chunk["queue_len"]) - int(chunk["remaining"])
        if executed <= 0:
            return ("discard", chunk)
        chunk["intervention"] = True
        chunk["executed_steps"] = executed
        return ("transition", chunk)


def qpos_command(arm_q, left_grip: float, right_grip: float) -> np.ndarray:
    """One commanded AbsQpos step, in the same layout the policy executes."""
    raw = np.zeros(16, dtype=np.float32)
    arm = np.asarray(arm_q, dtype=np.float32).reshape(-1)
    raw[0:7] = arm[0:7]
    raw[7:14] = arm[7:14]
    raw[14] = float(left_grip)
    raw[15] = float(right_grip)
    return raw[_REORDER_FROM_RAW]


class TakeoverChunk:
    """Human joint commands recorded against one pending ``act``.

    The server already stored the observation at ``act`` time. These rows are
    the actions that observation should be paired with, so behavior cloning
    follows the operator instead of the action the policy had proposed.
    """

    def __init__(self):
        self.transition_id: Optional[str] = None
        self.episode_id = 0
        self.chunk_id = 0
        self.chunk_len = 0
        self.rows: list = []

    @property
    def active(self) -> bool:
        return self.transition_id is not None

    @property
    def steps(self) -> int:
        return len(self.rows)

    def open(self, transition_id: str, episode_id: int, chunk_id: int, chunk_len: int) -> None:
        self.transition_id = str(transition_id)
        self.episode_id = int(episode_id)
        self.chunk_id = int(chunk_id)
        self.chunk_len = int(chunk_len)
        self.rows = []

    def push(self, action) -> bool:
        if not self.active:
            raise RuntimeError("takeover chunk is not open")
        self.rows.append(np.asarray(action, dtype=np.float32).reshape(-1).copy())
        return self.steps >= self.chunk_len

    def close(self):
        """Return ``(identity, actions [T, 16], executed)`` and forget the act.

        ``executed`` is 0 when the operator left before any command was recorded.
        The caller discards that pending act.
        """
        if not self.active:
            return None
        identity = {
            "transition_id": self.transition_id,
            "episode_id": int(self.episode_id),
            "chunk_id": int(self.chunk_id),
        }
        if self.rows:
            limit = self.chunk_len if self.chunk_len > 0 else len(self.rows)
            actions = np.stack(self.rows[:limit], axis=0).astype(np.float32)
        else:
            actions = np.zeros((0, 16), dtype=np.float32)
        executed = int(actions.shape[0])
        self.transition_id = None
        self.rows = []
        self.chunk_len = 0
        return identity, actions, executed
