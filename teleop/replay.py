#!/usr/bin/env python3
"""Open-loop replay of a recorded episode on the G1-D.

Point it at an episode's data.json and the arms and grippers repeat what was
recorded. No XR, no cameras, no policy server.

    ./scripts/run_replay.sh ~/unitree_eai_environment/data/pick_place/episode_0000
    python -m teleop.replay --data-json .../episode_0000/data.json --dry-run

Open loop means the robot follows the recorded joint angles regardless of what
is actually on the table. Clear the workspace and keep a hand on the e-stop.
"""
import argparse
import logging_mp
import os
import sys
import threading
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

logging_mp.basicConfig(
    level=logging_mp.INFO,
    file=True,
    file_path="/home/unitree/unitree_eai_environment/logs",
    backup_count=100,
    max_file_size=50 * 1024 * 1024,
    file_name_format="{prog_name}_%Y%m%d.log",
)
logger_mp = logging_mp.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
from teleop.utils.dex1_arm_bundle import create_dex1_arm_controller
from teleop.utils.trajectory_replay import (
    MAX_JOINT_STEP, TrajectoryError, approach_profile, check_trajectory,
    load_trajectory, summarize,
)
from teleop.utils.ready_pose import (
    DEFAULT_READY_FREQUENCY, ReadyPoseError, load_ready_pose, move_to_ready_pose,
)

from sshkeyboard import listen_keyboard, stop_listening

ABORT = False


def on_press(key):
    global ABORT
    if key in ("q", "esc"):
        ABORT = True
        logger_mp.warning("[replay] abort requested; will go home after hold")
        stop_listening()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-json', required=True,
                        help="episode data.json, or the episode_XXXX directory holding it")
    parser.add_argument('--source', choices=['actions', 'states'], default='actions',
                        help="replay commanded joints (default) or measured ones")
    parser.add_argument('--start', type=int, default=0, help="first frame to replay")
    parser.add_argument('--end', type=int, default=None, help="stop before this frame")
    parser.add_argument('--stride', type=int, default=1, help="replay every Nth frame")
    parser.add_argument('--speed', type=float, default=1.0,
                        help="playback rate multiplier; 0.5 is half speed")
    parser.add_argument('--fps', type=float, default=None,
                        help="override the rate recorded in info.image.fps")
    parser.add_argument('--loop', type=int, default=1, help="how many times to replay")
    parser.add_argument('--approach-seconds', type=float, default=3.0,
                        help="ramp from the current pose to the first frame")
    parser.add_argument('--ready-pose-config',
                        default=os.path.join(REPO_ROOT, 'configs', 'ready_pose.json'),
                        help="14-joint raised startup pose shared with policy deployment")
    parser.add_argument('--ready-pose-seconds', type=float, default=3.0,
                        help="seconds to move to the raised startup pose before frame 0")
    parser.add_argument('--max-joint-step', type=float, default=MAX_JOINT_STEP,
                        help="reject the episode if consecutive frames jump more than this (rad)")
    parser.add_argument('--dry-run', action='store_true',
                        help="load, check and summarize without touching the robot")
    parser.add_argument('--yes', action='store_true', help="skip the confirmation prompt")
    parser.add_argument('--no-go-home', action='store_true',
                        help="hold the last pose instead of returning to home")
    parser.add_argument('--network-interface', type=str, default=os.environ.get("UNITREE_DDSINTERFACE"),
                        help="DDS interface, e.g. eth0")
    return parser


def send(arm_ctrl, arm_ik, arm_q, grippers):
    arm_ctrl.ctrl_dual_arm(arm_q, arm_ik.solve_tau(arm_q))
    arm_ctrl.set_policy_gripper_q(float(grippers[0]), float(grippers[1]))


def run_sequence(arm_ctrl, arm_ik, steps, dt, label, log_every=1.0, last_command=None):
    """Send one command per tick, pacing to dt. Returns False if aborted."""
    started = time.time()
    next_log = started + log_every
    total = len(steps)
    for i, (arm_q, grippers) in enumerate(steps):
        if ABORT:
            logger_mp.warning(f"[replay] {label} aborted at step {i + 1}/{total}")
            return False
        tick = time.time()
        send(arm_ctrl, arm_ik, arm_q, grippers)
        if last_command is not None:
            last_command["arm_q"] = np.asarray(arm_q, dtype=float).copy()
            last_command["grippers"] = np.asarray(grippers, dtype=float).copy()
        if time.time() >= next_log:
            logger_mp.info(f"[replay] {label} {i + 1}/{total} "
                           f"({(i + 1) / total * 100:.0f}%, {time.time() - started:.1f}s)")
            next_log = time.time() + log_every
        time.sleep(max(0.0, dt - (time.time() - tick)))
    logger_mp.info(f"[replay] {label} done, {total} steps in {time.time() - started:.1f}s")
    return True


def main():
    args = build_parser().parse_args()

    if args.speed <= 0:
        print("[replay] --speed must be positive", file=sys.stderr)
        return 2
    if args.ready_pose_seconds <= 0:
        print("[replay] --ready-pose-seconds must be positive", file=sys.stderr)
        return 2

    try:
        ready_pose_q = load_ready_pose(args.ready_pose_config)
    except ReadyPoseError as exc:
        print(f"[replay] {exc}", file=sys.stderr)
        return 2

    try:
        traj = load_trajectory(args.data_json, source=args.source, start=args.start,
                               end=args.end, stride=args.stride)
    except TrajectoryError as exc:
        # print, not logger_mp: the reason a replay was refused has to reach the
        # operator even when the log listener failed to start.
        print(f"[replay] {exc}", file=sys.stderr)
        return 2

    if args.fps is not None:
        traj.fps = float(args.fps)
    dt = 1.0 / (traj.fps * args.speed)

    problems = check_trajectory(traj, max_joint_step=args.max_joint_step)
    print(summarize(traj))
    print(f"speed   : x{args.speed:g} -> {1.0 / dt:.1f} Hz, "
          f"{traj.duration / args.speed:.1f} s per pass, {args.loop} pass(es)")
    if problems:
        # Part of the same report, so keep it on stdout. Splitting it across
        # stderr reorders the output the moment anything pipes or tees it.
        print()
        for problem in problems:
            print(f"UNSAFE  : {problem}")
        print("refusing to replay this episode. Narrow the range with --start/--end, "
              "or raise --max-joint-step if the jump is real.")
        return 2
    print("checks  : ok")

    if args.dry_run:
        print("\ndry run, robot untouched")
        return 0

    ChannelFactoryInitialize(0, networkInterface=args.network_interface)

    # xr_motion_data_ready must start False. With no XR process to set it, a
    # True value sends the controller down the hand-tracking branch, which maps
    # an unset trigger to 0 and snaps both grippers shut before we take over.
    bundle = create_dex1_arm_controller(simulation_mode=False, use_waist=False)
    arm_ctrl = bundle.arm_ctrl
    arm_ik = G1_29_ArmIK()

    hold_q = arm_ctrl.get_current_dual_arm_q()[:14].copy()
    hold_grippers = arm_ctrl.get_current_dual_gripper_q().copy()
    arm_ik.reset_solution_state(hold_q)
    send(arm_ctrl, arm_ik, hold_q, hold_grippers)
    time.sleep(1.0)

    print()
    ready_gap = np.abs(hold_q - ready_pose_q)
    print(f"ready   : largest joint gap from current pose is {ready_gap.max():.3f} rad "
          f"(joint {int(np.argmax(ready_gap))})")
    print(summarize(traj, current_q=ready_pose_q))
    if not args.yes:
        try:
            answer = input("\nClear the workspace, hand on the e-stop. Replay? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            print("cancelled")
            return 1

    keyboard = threading.Thread(
        target=listen_keyboard,
        kwargs={"on_press": on_press, "until": None, "sequential": False},
        daemon=True,
    )
    keyboard.start()
    logger_mp.info("[replay] press Q (or Ctrl+C) in this terminal to abort and go home")

    completed = 0
    last_command = {
        "arm_q": hold_q.copy(),
        "grippers": hold_grippers.copy(),
    }
    try:
        logger_mp.info(
            "[replay] moving both arms to the raised ready pose (%.1fs)",
            args.ready_pose_seconds,
        )
        ready_result = move_to_ready_pose(
            arm_ctrl,
            arm_ik,
            ready_pose_q,
            args.ready_pose_seconds,
            DEFAULT_READY_FREQUENCY,
            grippers=hold_grippers,
            stop_requested=lambda: ABORT,
        )
        hold_q = ready_result.arm_q.copy()
        last_command["arm_q"] = hold_q.copy()
        if not ready_result.completed:
            logger_mp.warning("[replay] ready-pose reset interrupted; trajectory not started")
            return 1
        logger_mp.info("[replay] raised ready pose reached; approaching frame 0")

        approach = approach_profile(hold_q, hold_grippers, traj.arm_q[0], traj.grippers[0],
                                    args.approach_seconds, traj.fps)
        if not run_sequence(
                arm_ctrl, arm_ik, approach, 1.0 / traj.fps, "approach",
                last_command=last_command):
            return 1

        frames = list(zip(traj.arm_q, traj.grippers))
        for lap in range(args.loop):
            label = "replay" if args.loop == 1 else f"replay pass {lap + 1}/{args.loop}"
            if not run_sequence(
                    arm_ctrl, arm_ik, frames, dt, label, last_command=last_command):
                return 1
            completed += 1
            if lap + 1 < args.loop and not ABORT:
                back = approach_profile(traj.arm_q[-1], traj.grippers[-1],
                                        traj.arm_q[0], traj.grippers[0],
                                        args.approach_seconds, traj.fps)
                if not run_sequence(
                        arm_ctrl, arm_ik, back, 1.0 / traj.fps, "return to start",
                        last_command=last_command):
                    return 1
    except KeyboardInterrupt:
        logger_mp.warning("[replay] interrupted; will go home after hold")
    finally:
        stop_listening()
        # Hold the last command so the arms do not sag once we stop publishing.
        last_q = last_command["arm_q"]
        last_grippers = last_command["grippers"]
        deadline = time.time() + 1.0
        while time.time() < deadline:
            send(arm_ctrl, arm_ik, last_q, last_grippers)
            time.sleep(1.0 / 30.0)
        if not args.no_go_home:
            try:
                logger_mp.info("[replay] returning to home (arms down)")
                arm_ctrl.ctrl_dual_arm_go_home()
            except Exception as exc:
                logger_mp.error(f"[replay] go home failed: {exc}")

    logger_mp.info(f"[replay] finished {completed}/{args.loop} pass(es)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
