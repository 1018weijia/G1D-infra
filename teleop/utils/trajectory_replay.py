"""Read a recorded episode back as a joint-space trajectory. No hardware here.

collect.py stores, per frame, the commanded arm joints and gripper openings.
Replaying those numbers reproduces the take open loop. This module does the
loading, range checking, and the ramp that carries the robot from wherever it
is standing to the first frame; teleop.replay owns everything that talks to DDS.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from teleop.utils.ready_pose import smoothstep

ARM_DIM = 14
GRIPPER_DIM = 2

# Matches the checks policy_client applies to a policy chunk, so a replayed
# trajectory cannot command anything a policy would be refused for.
MAX_ABS_ARM_Q = 3.5
GRIPPER_LIMITS = (-0.1, 5.5)
# The controller clips to 30 rad/s, which at 30 Hz still allows ~1 rad between
# two commands. Recorded teleoperation stays far below this, so anything larger
# means a damaged file rather than a fast motion.
MAX_JOINT_STEP = 0.25


class TrajectoryError(ValueError):
    """Raised when an episode cannot be turned into a safe trajectory."""


@dataclass
class Trajectory:
    arm_q: np.ndarray        # (T, 14) left 7 then right 7
    grippers: np.ndarray     # (T, 2) left, right
    fps: float
    source: str              # "actions" or "states"
    path: str
    frame_indices: np.ndarray
    success: Optional[bool]

    def __len__(self):
        return int(self.arm_q.shape[0])

    @property
    def duration(self):
        return len(self) / self.fps if self.fps > 0 else float("nan")


def _leaf(frame, block, group, key, index):
    try:
        node = frame[block][group]
    except (KeyError, TypeError):
        raise TrajectoryError(
            f"frame {index} has no '{block}.{group}'. "
            f"Is this a G1-D episode recorded by collect.py?"
        ) from None
    value = node.get(key)
    if value is None:
        raise TrajectoryError(f"frame {index} has no '{block}.{group}.{key}'")
    vector = np.atleast_1d(np.asarray(value, dtype=float)).reshape(-1)
    if vector.size == 0:
        raise TrajectoryError(f"frame {index}: '{block}.{group}.{key}' is empty")
    return vector


def load_trajectory(data_json, source="actions", start=0, end=None, stride=1):
    """Build a Trajectory from an episode data.json.

    source="actions" replays what was commanded, which is what reproduces the
    take. source="states" replays what the joints measured, which lags the
    command by the tracking error and is mostly useful for comparison.
    """
    if source not in ("actions", "states"):
        raise TrajectoryError(f"source must be 'actions' or 'states', got {source!r}")
    if stride < 1:
        raise TrajectoryError(f"stride must be >= 1, got {stride}")

    path = os.path.abspath(os.path.expanduser(data_json))
    if os.path.isdir(path):
        path = os.path.join(path, "data.json")
    if not os.path.isfile(path):
        raise TrajectoryError(f"no such episode file: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise TrajectoryError(
                f"{path} is not valid JSON ({exc}). An episode killed mid-write "
                f"stays truncated; check for a FAILED marker or re-record."
            ) from None

    frames = payload.get("data")
    if not frames:
        raise TrajectoryError(f"{path} has no frames under 'data'")

    total = len(frames)
    stop = total if end is None else min(int(end), total)
    begin = max(0, int(start))
    if begin >= stop:
        raise TrajectoryError(f"empty frame range [{begin}, {stop}) over {total} frames")

    indices = list(range(begin, stop, stride))
    arm_q = np.empty((len(indices), ARM_DIM), dtype=float)
    grippers = np.empty((len(indices), GRIPPER_DIM), dtype=float)
    for row, index in enumerate(indices):
        frame = frames[index]
        left = _leaf(frame, source, "left_arm", "qpos", index)
        right = _leaf(frame, source, "right_arm", "qpos", index)
        if left.size != 7 or right.size != 7:
            raise TrajectoryError(
                f"frame {index}: expected 7 joints per arm, got {left.size} and {right.size}"
            )
        arm_q[row, :7] = left
        arm_q[row, 7:] = right
        grippers[row, 0] = _leaf(frame, source, "left_ee", "qpos", index)[0]
        grippers[row, 1] = _leaf(frame, source, "right_ee", "qpos", index)[0]

    fps = float(payload.get("info", {}).get("image", {}).get("fps") or 30.0)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    # Taking every Nth frame shortens wall-clock time unless the rate follows.
    fps = fps / stride

    return Trajectory(
        arm_q=arm_q,
        grippers=grippers,
        fps=fps,
        source=source,
        path=path,
        frame_indices=np.asarray(indices, dtype=int),
        success=payload.get("success"),
    )


def check_trajectory(traj, max_abs_arm_q=MAX_ABS_ARM_Q, gripper_limits=GRIPPER_LIMITS,
                     max_joint_step=MAX_JOINT_STEP):
    """Return a list of reasons this trajectory is unsafe to send. Empty is good."""
    problems = []

    if not np.all(np.isfinite(traj.arm_q)) or not np.all(np.isfinite(traj.grippers)):
        problems.append("trajectory contains NaN or inf")
        return problems

    worst = float(np.max(np.abs(traj.arm_q)))
    if worst > max_abs_arm_q:
        row, col = np.unravel_index(int(np.argmax(np.abs(traj.arm_q))), traj.arm_q.shape)
        problems.append(
            f"arm joint {col} reaches {traj.arm_q[row, col]:+.3f} rad at frame "
            f"{traj.frame_indices[row]}, over the {max_abs_arm_q} rad limit"
        )

    low, high = gripper_limits
    if float(traj.grippers.min()) < low or float(traj.grippers.max()) > high:
        problems.append(
            f"gripper command spans [{traj.grippers.min():.3f}, {traj.grippers.max():.3f}], "
            f"outside [{low}, {high}]"
        )

    if len(traj) > 1:
        steps = np.abs(np.diff(traj.arm_q, axis=0))
        worst_step = float(steps.max())
        if worst_step > max_joint_step:
            row, col = np.unravel_index(int(np.argmax(steps)), steps.shape)
            problems.append(
                f"arm joint {col} jumps {worst_step:.3f} rad between frames "
                f"{traj.frame_indices[row]} and {traj.frame_indices[row + 1]}, "
                f"over the {max_joint_step} rad limit"
            )

    return problems


def approach_profile(current_q, current_grippers, target_q, target_grippers, seconds, fps):
    """Ramp from the robot's pose to the first frame, easing in and out.

    The robot is almost never parked where the take started, so jumping
    straight to frame 0 would be the largest motion of the whole replay.
    """
    current_q = np.asarray(current_q, dtype=float).reshape(-1)
    target_q = np.asarray(target_q, dtype=float).reshape(-1)
    if current_q.size != ARM_DIM or target_q.size != ARM_DIM:
        raise TrajectoryError(f"approach needs {ARM_DIM} arm joints on both ends")
    current_grippers = np.asarray(current_grippers, dtype=float).reshape(-1)
    target_grippers = np.asarray(target_grippers, dtype=float).reshape(-1)
    if seconds <= 0 or fps <= 0:
        return [(target_q.copy(), target_grippers.copy())]

    steps = max(1, int(round(seconds * fps)))
    out = []
    for i in range(1, steps + 1):
        gain = smoothstep(i / steps)
        out.append((
            current_q + (target_q - current_q) * gain,
            current_grippers + (target_grippers - current_grippers) * gain,
        ))
    return out


def summarize(traj, current_q=None):
    """Human-readable report shown before anything moves."""
    lines = [
        f"episode : {traj.path}",
        f"frames  : {len(traj)} from source index "
        f"{traj.frame_indices[0]}..{traj.frame_indices[-1]}  (block: {traj.source})",
        f"playback: {traj.fps:.1f} Hz, {traj.duration:.1f} s",
    ]
    if traj.success is False:
        lines.append("label   : this episode is marked FAILED")
    elif traj.success is True:
        lines.append("label   : success")

    left, right = traj.arm_q[:, :7], traj.arm_q[:, 7:]
    lines.append(f"left arm : [{left.min():+.3f}, {left.max():+.3f}] rad, "
                 f"max step {np.abs(np.diff(left, axis=0)).max() if len(traj) > 1 else 0.0:.3f}")
    lines.append(f"right arm: [{right.min():+.3f}, {right.max():+.3f}] rad, "
                 f"max step {np.abs(np.diff(right, axis=0)).max() if len(traj) > 1 else 0.0:.3f}")
    lines.append(f"grippers : left [{traj.grippers[:, 0].min():.2f}, {traj.grippers[:, 0].max():.2f}], "
                 f"right [{traj.grippers[:, 1].min():.2f}, {traj.grippers[:, 1].max():.2f}]")

    if current_q is not None:
        gap = np.abs(np.asarray(current_q, dtype=float).reshape(-1) - traj.arm_q[0])
        lines.append(f"approach : largest joint gap to frame {traj.frame_indices[0]} "
                     f"is {gap.max():.3f} rad (joint {int(np.argmax(gap))})")
    return "\n".join(lines)
