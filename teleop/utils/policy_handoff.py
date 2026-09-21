"""Policy/teleop handoff state names and hardware-free control helpers."""
from dataclasses import dataclass

import numpy as np

from teleop.utils.handoff_utils import limit_joint_step, smoothstep_handoff_gain

POLICY_IDLE = "POLICY_IDLE"
POLICY_LIVE = "POLICY_LIVE"
POLICY_ROLLBACK = "POLICY_ROLLBACK"
ALIGNING = "ALIGNING"
TELEOP_LIVE = "TELEOP_LIVE"

TELEOP_PHASES = (TELEOP_LIVE,)
RESUME_FROM_PHASES = (ALIGNING,) + TELEOP_PHASES
ROLLBACK_FROM_PHASES = (POLICY_LIVE, TELEOP_LIVE)

A_HANDOFF_DEBOUNCE_S = 0.5
A_TELEOP_READY_DEBOUNCE_S = 0.2
A_GAP_S = 0.25

START_POLICY = "start_policy"
RESUME_POLICY = "resume_policy"
QUIT = "quit"
ROLLBACK = "rollback"
TAKEOVER = "takeover"
IGNORE_A = "ignore_a"
DEBOUNCE_A = "debounce_a"
REPEAT_A = "repeat_a"
NONE = "none"

ALIGNMENT_CODE = {"ALIGNING": 0, "ALIGNED": 1, "ACTIVE": 2}


def is_tracking_valid(tele_data):
    return bool(
        getattr(tele_data, "motion_data_ready", False)
        and getattr(tele_data, "left_wrist_valid", False)
        and getattr(tele_data, "right_wrist_valid", False)
    )


def alignment_state_code(state):
    return ALIGNMENT_CODE.get(state, 2)


def hold_arm_cmd(hold_q, last_arm_q):
    source = last_arm_q if hold_q is None else hold_q
    return np.asarray(source, dtype=float)[:14].copy()


def hold_tau_cmd(hold_tau, last_tau):
    source = last_tau if hold_tau is None else hold_tau
    return np.asarray(source, dtype=float)[:14].copy()


def hold_gripper_cmd(hold_grip, last_left, last_right):
    """Use the frozen rollback grippers when present; otherwise keep the last command."""
    if hold_grip is None:
        return float(last_left), float(last_right)
    return float(hold_grip[0]), float(hold_grip[1])


def rollback_hold_from_last_command(last_arm_q, last_tau, last_left_grip, last_right_grip):
    """Freeze the last commanded rollback pose, not lagged motor state."""
    hold_q = np.asarray(last_arm_q, dtype=float).reshape(-1)[:14].copy()
    hold_tau = np.asarray(last_tau, dtype=float).reshape(-1)[:14].copy()
    if hold_q.size != 14 or hold_tau.size != 14:
        raise ValueError("rollback hold requires 14-dof arm command and tau")
    if not np.all(np.isfinite(hold_q)) or not np.all(np.isfinite(hold_tau)):
        raise ValueError("rollback hold command must be finite")
    return hold_q, hold_tau, float(last_left_grip), float(last_right_grip)


def new_handoff_runtime():
    return {
        "hold_q": None,
        "hold_tau": None,
        "hold_grip": None,
        "previous_q": None,
        "last_command_q": None,
        "blend_started": None,
        "max_delta": 0.0,
        "tracking_lost": False,
        "overlay_left": None,
        "overlay_right": None,
        "warning_time": 0.0,
    }


def reset_handoff_runtime(handoff):
    handoff.update(new_handoff_runtime())
    return handoff


def is_blending(handoff):
    return handoff.get("blend_started") is not None


def interpret_key(key, phase, now, debounce_until, last_a_at, a_gap_s=A_GAP_S):
    """Map one stdin character to a control intent.

    Returns (action, last_a_at). last_a_at is updated for every 'a', including
    ignored repeats, so a held key cannot both take over and immediately resume.
    """
    if key == "s":
        if phase in TELEOP_PHASES:
            return RESUME_POLICY, last_a_at
        if phase in (POLICY_IDLE, ALIGNING):
            return START_POLICY, last_a_at
        return NONE, last_a_at
    if key == "q":
        return QUIT, last_a_at
    if key == "b":
        if phase in ROLLBACK_FROM_PHASES:
            return ROLLBACK, last_a_at
        return NONE, last_a_at
    if key != "a":
        return NONE, last_a_at

    gap = now - last_a_at if last_a_at else 1.0
    last_a_at = now
    if phase == ALIGNING:
        return TAKEOVER, last_a_at
    if phase not in TELEOP_PHASES:
        return IGNORE_A, last_a_at
    if now < debounce_until:
        return DEBOUNCE_A, last_a_at
    if gap < a_gap_s:
        return REPEAT_A, last_a_at
    return RESUME_POLICY, last_a_at


def stale_key_flags(phase, start_policy, resume_policy, align_confirm, rollback_request):
    """Drop latched keys that are not valid in the current phase."""
    if start_policy and phase not in (POLICY_IDLE, ALIGNING):
        start_policy = False
    if resume_policy and phase not in TELEOP_PHASES:
        resume_policy = False
    if align_confirm and phase != ALIGNING:
        align_confirm = False
    if rollback_request and phase not in ROLLBACK_FROM_PHASES:
        rollback_request = False
    return start_policy, resume_policy, align_confirm, rollback_request


def blend_should_finish(elapsed, duration, tracking_valid, ik_ok):
    return float(elapsed) >= float(duration) and bool(tracking_valid) and bool(ik_ok)


@dataclass
class RelativeArmCommand:
    sol_q: np.ndarray
    sol_tauff: np.ndarray
    tracking_lost: bool
    blend_started: float | None
    max_delta: float
    reanchored: bool


def compute_relative_arm_command(
        alignment,
        arm_ik,
        left_waist,
        right_waist,
        tracking_valid,
        blending,
        tracking_lost,
        now,
        blend_started,
        blend_seconds,
        max_joint_speed,
        frequency,
        previous_q,
        last_command_q,
        last_arm_q,
        current_lr_arm_q):
    """Relative XR-to-robot arm command used during teleop, including blend-in."""
    hold_source = previous_q if blending else last_command_q
    if not tracking_valid:
        sol_q = hold_arm_cmd(hold_source, last_arm_q)
        sol_tauff = np.zeros_like(sol_q)
        arm_ik.reset_solution_state(sol_q)
        return RelativeArmCommand(
            sol_q=sol_q,
            sol_tauff=sol_tauff,
            tracking_lost=True,
            blend_started=blend_started,
            max_delta=0.0,
            reanchored=False,
        )

    reanchored = False
    if tracking_lost:
        anchor_q = hold_arm_cmd(hold_source, last_arm_q)
        robot_left, robot_right = arm_ik.solve_fk_matrix(np.asarray(anchor_q, dtype=float))
        alignment.reanchor(left_waist, right_waist, robot_left, robot_right)
        arm_ik.reset_solution_state(anchor_q)
        if blending:
            blend_started = now
        reanchored = True

    if blending and blend_started is None:
        blend_started = now

    blend_seed = hold_arm_cmd(
        previous_q if blending else current_lr_arm_q,
        current_lr_arm_q,
    )
    gain = (
        smoothstep_handoff_gain(now - blend_started, blend_seconds)
        if blending else 1.0
    )
    poses = alignment.targets_from_hand(left_waist, right_waist, gain)
    if poses is None:
        sol_q = hold_arm_cmd(hold_source, last_arm_q)
        sol_tauff = np.zeros_like(sol_q)
        arm_ik.reset_solution_state(sol_q)
        return RelativeArmCommand(
            sol_q=sol_q,
            sol_tauff=sol_tauff,
            tracking_lost=False,
            blend_started=blend_started,
            max_delta=0.0,
            reanchored=reanchored,
        )
    sol_q, sol_tauff = arm_ik.solve_ik(poses[0], poses[1], blend_seed)
    max_delta = 0.0
    if blending:
        step = max_joint_speed / frequency
        limited = limit_joint_step(sol_q[:14], blend_seed, step)
        max_delta = float(np.max(np.abs(limited - blend_seed)))
        sol_q = limited
        arm_ik.reset_solution_state(limited)
    return RelativeArmCommand(
        sol_q=sol_q,
        sol_tauff=sol_tauff,
        tracking_lost=False,
        blend_started=blend_started,
        max_delta=max_delta,
        reanchored=reanchored,
    )
