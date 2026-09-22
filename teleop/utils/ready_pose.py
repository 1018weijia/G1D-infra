"""Shared startup pose and smooth reset helpers for G1-D arm workflows."""
import json
import time
from dataclasses import dataclass

import numpy as np


ARM_DIM = 14
MAX_ABS_ARM_Q = 3.5
DEFAULT_READY_FREQUENCY = 30.0


class ReadyPoseError(ValueError):
    pass


@dataclass
class ReadyPoseResult:
    completed: bool
    arm_q: np.ndarray
    tau: np.ndarray


def _arm_q(value, label):
    q = np.asarray(value, dtype=float).reshape(-1)
    if q.size != ARM_DIM:
        raise ReadyPoseError(f"{label} must contain {ARM_DIM} arm joints, got {q.size}")
    if not np.all(np.isfinite(q)):
        raise ReadyPoseError(f"{label} contains NaN or inf")
    if float(np.max(np.abs(q))) > MAX_ABS_ARM_Q:
        raise ReadyPoseError(
            f"{label} exceeds the {MAX_ABS_ARM_Q} rad startup safety limit"
        )
    return q


def load_ready_pose(path):
    """Load and validate a left-then-right 14-joint ready pose."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise ReadyPoseError(f"cannot read ready pose config {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ReadyPoseError(f"ready pose config is not valid JSON: {path}: {exc}") from exc

    if not isinstance(payload, dict) or "arm_q" not in payload:
        raise ReadyPoseError("ready pose config must contain an arm_q array")
    return _arm_q(payload["arm_q"], "ready pose")


def smoothstep(progress):
    p = float(np.clip(progress, 0.0, 1.0))
    return p * p * (3.0 - 2.0 * p)


def ready_pose_profile(current_q, target_q, seconds, frequency):
    """Create an eased command profile ending exactly at the ready pose."""
    current = _arm_q(current_q, "current pose")
    target = _arm_q(target_q, "ready pose")
    if seconds <= 0.0:
        raise ReadyPoseError("ready pose duration must be positive")
    if frequency <= 0.0:
        raise ReadyPoseError("ready pose frequency must be positive")

    steps = max(1, int(round(float(seconds) * float(frequency))))
    return np.stack([
        current + (target - current) * smoothstep(i / steps)
        for i in range(1, steps + 1)
    ])


def move_to_ready_pose(
        arm_ctrl, arm_ik, target_q, seconds, frequency, grippers=None,
        settle_seconds=0.5, stop_requested=None):
    """Move both arms to the shared ready pose while preserving the grippers."""
    if settle_seconds < 0.0:
        raise ReadyPoseError("ready pose settle duration must be non-negative")
    should_stop = stop_requested or (lambda: False)
    start_q = _arm_q(arm_ctrl.get_current_dual_arm_q()[:ARM_DIM], "current pose")
    target = _arm_q(target_q, "ready pose")
    profile = ready_pose_profile(start_q, target, seconds, frequency)

    grip = None
    if grippers is not None:
        grip = np.asarray(grippers, dtype=float).reshape(-1)
        if grip.size != 2 or not np.all(np.isfinite(grip)):
            raise ReadyPoseError("grippers must contain two finite values")

    arm_ik.reset_solution_state(start_q)
    last_q = start_q.copy()
    last_tau = arm_ik.solve_tau(last_q)
    period = 1.0 / float(frequency)
    deadline = time.monotonic()

    def send(q):
        nonlocal last_q, last_tau, deadline
        last_q = np.asarray(q, dtype=float).copy()
        last_tau = arm_ik.solve_tau(last_q)
        arm_ctrl.ctrl_dual_arm(last_q, last_tau)
        if grip is not None:
            arm_ctrl.set_policy_gripper_q(float(grip[0]), float(grip[1]))
        deadline += period
        time.sleep(max(0.0, deadline - time.monotonic()))

    for q in profile:
        if should_stop():
            arm_ik.reset_solution_state(last_q)
            return ReadyPoseResult(False, last_q, last_tau)
        send(q)

    settle_steps = int(round(float(settle_seconds) * float(frequency)))
    for _ in range(settle_steps):
        if should_stop():
            arm_ik.reset_solution_state(last_q)
            return ReadyPoseResult(False, last_q, last_tau)
        send(target)

    arm_ik.reset_solution_state(target)
    return ReadyPoseResult(True, target.copy(), last_tau.copy())
