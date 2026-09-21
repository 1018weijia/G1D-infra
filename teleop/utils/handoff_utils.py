"""Shared helpers for rollback alignment and no-jump handoff."""
import numpy as np


def smoothstep_handoff_gain(elapsed, duration):
    """Return a smooth 0-to-1 handoff gain."""
    if duration <= 0.0:
        return 1.0
    progress = float(np.clip(elapsed / duration, 0.0, 1.0))
    return progress * progress * (3.0 - 2.0 * progress)


def limit_joint_step(target, previous, max_delta):
    """Limit each arm-joint command relative to the last sent command."""
    target = np.asarray(target, dtype=float).reshape(-1)
    previous = np.asarray(previous, dtype=float).reshape(-1)
    if target.shape != previous.shape:
        raise ValueError("target and previous joint commands must have the same shape")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(previous)):
        raise ValueError("joint commands must contain only finite values")
    if not np.isfinite(max_delta) or max_delta < 0.0:
        raise ValueError("max_delta must be finite and non-negative")
    return previous + np.clip(target - previous, -max_delta, max_delta)


def rollback_endpoint_to_xr_targets(arm_ik, tv_wrapper, endpoint_q, head_pose, forward_offset=0.0):
    """Convert rollback endpoint WAIST EEF poses to OpenXR controller targets."""
    endpoint_q = np.asarray(endpoint_q, dtype=float).reshape(-1)
    if endpoint_q.size != 14 or not np.all(np.isfinite(endpoint_q)):
        raise ValueError("rollback endpoint must contain 14 finite arm joints")
    left_waist, right_waist = arm_ik.solve_fk_matrix(endpoint_q)
    targets = {
        "left": tv_wrapper.waist_pose_to_openxr_controller(left_waist, head_pose, "left"),
        "right": tv_wrapper.waist_pose_to_openxr_controller(right_waist, head_pose, "right"),
    }
    if forward_offset:
        targets = {
            side: tv_wrapper.apply_openxr_forward_offset(target, head_pose, forward_offset)
            for side, target in targets.items()
        }
    for side, target in targets.items():
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError(f"invalid XR rollback target for {side}")
        if not np.allclose(target[3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError(f"non-homogeneous XR rollback target for {side}")
    return targets
