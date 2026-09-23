"""Policy rollout with rollback and XR controller handoff on a single robot owner."""
import argparse
import fcntl
import logging_mp
import os
import sys
import termios
import threading
import time
import tty
from multiprocessing import Array

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

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
from teleop.teleimager.src.teleimager.image_client import ImageClient
from teleop.utils.alignment import ACTIVE, ALIGNED, DualArmAlignment
from teleop.utils.alignment_config import load_targets
from teleop.utils.dex1_arm_bundle import create_dex1_arm_controller
from teleop.utils.ego_projection import EgoPixelOverlay
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.handoff_utils import rollback_endpoint_to_xr_targets
from teleop.utils.policy_client import PolicyAdapter, PolicyRemoteClient
from teleop.utils.rlt_online import (
    RL_MAX_ABS_ARM_Q,
    RL_MAX_ABS_GRIPPER_Q,
    RLTRollout,
    TakeoverChunk,
    qpos_command,
    rewind_plan,
    transition_fields,
)
from teleop.utils.policy_handoff import (
    A_GAP_S,
    A_HANDOFF_DEBOUNCE_S,
    A_TELEOP_READY_DEBOUNCE_S,
    ALIGNING,
    DEBOUNCE_A,
    IGNORE_A,
    NONE,
    POLICY_IDLE,
    POLICY_LIVE,
    POLICY_ROLLBACK,
    QUIT,
    REPEAT_A,
    RESUME_POLICY as RESUME_POLICY_ACTION,
    ROLLBACK,
    ROLLBACK_FROM_PHASES,
    START_POLICY as START_POLICY_ACTION,
    TAKEOVER,
    TELEOP_LIVE,
    alignment_state_code,
    blend_should_finish,
    ButtonRisingEdge,
    compute_relative_arm_command,
    hold_arm_cmd,
    hold_gripper_cmd,
    hold_tau_cmd,
    interpret_key,
    is_blending,
    is_tracking_valid,
    new_handoff_runtime,
    reset_handoff_runtime,
    rollback_hold_from_last_command,
    stale_key_flags,
)
from teleop.utils.rollback import (
    ROLLBACK_EASE_SECONDS,
    PolicyRollbackBuffer,
    ease_out_playback,
)
from teleop.utils.ready_pose import ReadyPoseError, load_ready_pose, move_to_ready_pose

STOP = False
START_POLICY = False
ROLLBACK_REQUEST = False
ALIGN_CONFIRM = False
RESUME_POLICY = False
RUN_PHASE = POLICY_IDLE
ALIGNMENT_STATE = None
A_LAST_CHAR_AT = 0.0
A_DEBOUNCE_UNTIL = 0.0
KEY_LISTENER_STOP = threading.Event()
RL_ONLINE = False
RL_OUTCOME_REQUEST = None


def on_press(key):
    global STOP, START_POLICY, ROLLBACK_REQUEST, ALIGN_CONFIRM, RESUME_POLICY
    global RUN_PHASE, ALIGNMENT_STATE, A_LAST_CHAR_AT, RL_OUTCOME_REQUEST
    if RL_ONLINE and key in ("y", "n"):
        if RUN_PHASE != POLICY_LIVE:
            logger_mp.warning("[on_press] %s ignored outside POLICY_LIVE (phase=%s).", key.upper(), RUN_PHASE)
            return
        RL_OUTCOME_REQUEST = "success" if key == "y" else "failure"
        logger_mp.info(
            "[on_press] %s: this chunk will end the episode as %s.",
            key.upper(), RL_OUTCOME_REQUEST,
        )
        return
    action, A_LAST_CHAR_AT = interpret_key(
        key, RUN_PHASE, time.monotonic(), A_DEBOUNCE_UNTIL, A_LAST_CHAR_AT, A_GAP_S,
    )
    if action == START_POLICY_ACTION:
        START_POLICY = True
    elif action == RESUME_POLICY_ACTION:
        RESUME_POLICY = True
        logger_mp.info("[on_press] %s: return control to policy (phase=%s).", key.upper(), RUN_PHASE)
    elif action == QUIT:
        STOP = True
    elif action == ROLLBACK:
        ROLLBACK_REQUEST = True
    elif action == TAKEOVER:
        ALIGN_CONFIRM = True
        if ALIGNMENT_STATE != ALIGNED:
            logger_mp.warning(
                "[on_press] A requested before ALIGNED (state=%s); forcing relative handoff.",
                ALIGNMENT_STATE,
            )
        else:
            logger_mp.info("[on_press] A: take over from alignment.")
    elif action == DEBOUNCE_A:
        logger_mp.info("[on_press] A ignored until handoff settles.")
    elif action == IGNORE_A:
        logger_mp.warning(
            "[on_press] A ignored in %s. Press keyboard B to rollback, then gamepad A to take over.",
            RUN_PHASE,
        )
    elif action == REPEAT_A:
        return
    elif action == NONE:
        if key not in ("s", "b"):
            logger_mp.warning("[on_press] %s was pressed, but no action is defined for this key.", key)


def _stdin_key_loop(on_press_cb, stop_event):
    """Deliver every stdin character as a key press. sshkeyboard coalesces a
    second 'a' into a hold and never fires on_press again until another key."""
    if not sys.stdin.isatty():
        logger_mp.error("stdin is not a TTY; keyboard control is unavailable.")
        return
    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    try:
        tty.setcbreak(fd)
        fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)
        while not stop_event.is_set():
            try:
                ch = sys.stdin.read(1)
            except (BlockingIOError, InterruptedError, TypeError):
                ch = ""
            if not ch:
                time.sleep(0.01)
                continue
            if ch == "\x03":
                on_press_cb("q")
                continue
            if ch == "\x1b":
                time.sleep(0.005)
                while True:
                    try:
                        extra = sys.stdin.read(1)
                    except (BlockingIOError, InterruptedError, TypeError):
                        extra = ""
                    if not extra:
                        break
                continue
            if ch.isalpha():
                on_press_cb(ch.lower())
    finally:
        try:
            fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)
            termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        except Exception:
            pass


def _read_grippers(arm_ctrl):
    grips = arm_ctrl.get_current_dual_gripper_q()
    return float(grips[0]), float(grips[1])


def _start_policy_from_current_pose(
        adapter, img_client, arm_ctrl, arm_ik, inference_lock, inference_state, start_inference,
        request_fn=None):
    """Capture the current robot pose and request the next policy chunk."""
    resume_q = arm_ctrl.get_current_dual_arm_q()[:14].copy()
    resume_left_grip, resume_right_grip = _read_grippers(arm_ctrl)
    resume_request = _prepare_policy_request(adapter, img_client, arm_ctrl)
    if request_fn is not None:
        resume_request = request_fn(resume_request)
    with inference_lock:
        inference_state["result"] = None
        inference_state["error"] = None
        inference_state["accept_result"] = True
    if not start_inference(resume_request):
        raise RuntimeError("policy inference is already busy")
    arm_ik.reset_solution_state(resume_q)
    arm_ctrl.set_policy_gripper_q(resume_left_grip, resume_right_grip)
    return resume_q, arm_ik.solve_tau(resume_q), resume_left_grip, resume_right_grip


def _write_controller_grippers(left_gripper_value, right_gripper_value, tele_data):
    with left_gripper_value.get_lock():
        left_gripper_value.value = tele_data.left_ctrl_triggerValue
    with right_gripper_value.get_lock():
        right_gripper_value.value = tele_data.right_ctrl_triggerValue


def _write_xr_grippers(left_gripper_value, right_gripper_value, tele_data, input_mode: str):
    source = getattr(tele_data, "tracking_source", "") or input_mode
    if source == "hand":
        with left_gripper_value.get_lock():
            left_gripper_value.value = tele_data.left_hand_pinchValue
        with right_gripper_value.get_lock():
            right_gripper_value.value = tele_data.right_hand_pinchValue
    else:
        _write_controller_grippers(left_gripper_value, right_gripper_value, tele_data)


def _prepare_policy_request(adapter, img_client, arm_ctrl):
    """Capture one observation in the main loop without contacting the cloud."""
    current_arm_q = arm_ctrl.get_current_dual_arm_q()[:14].copy()
    left_grip, right_grip = _read_grippers(arm_ctrl)
    head_img = img_client.get_head_frame()
    left_wrist_img = img_client.get_left_wrist_frame()
    right_wrist_img = img_client.get_right_wrist_frame()
    if head_img is None or left_wrist_img is None or right_wrist_img is None:
        raise RuntimeError("missing camera frame for policy inference")
    first_frame_np, state_np = adapter.build_model_input(
        head_img.bgr, left_wrist_img.bgr, right_wrist_img.bgr,
        current_arm_q, left_grip, right_grip,
    )
    return first_frame_np, state_np, current_arm_q


def _predict_policy_queue(adapter, remote, request, instruction):
    """Run the blocking cloud request; intended for the inference worker only."""
    first_frame_np, state_np, current_arm_q = request
    actions, predict_ms = remote.predict(first_frame_np, state_np, instruction)
    queue = adapter.build_exec_queue(actions, current_arm_q)
    logger_mp.info("Policy chunk received: predict=%.1fms queue=%d", predict_ms, len(queue))
    return queue


def _truncate_model_actions(adapter, actions):
    actions = np.asarray(actions, dtype=np.float32)
    limit = int(adapter.exec_chunk_steps)
    if 0 < limit < actions.shape[0]:
        actions = actions[:limit]
    return actions


def _run_rlt_job(adapter, remote, job, instruction):
    """Send one Stage-2 report, then optionally request the next chunk."""
    discard_id = job.get("discard_id")
    if discard_id:
        remote.discard(discard_id)
        logger_mp.info("RL discard sent for %s", discard_id)
    spec = job.get("transition")
    if spec is not None:
        remote.report_transition(
            spec["transition_id"],
            spec["next_frame"],
            spec["next_state"],
            instruction,
            spec["rewards"],
            done=spec["done"],
            bootstrap_mask=spec["bootstrap_mask"],
            action_chunk=spec["action_chunk"],
            intervention=spec["intervention"],
            episode_id=spec["episode_id"],
            chunk_id=spec["chunk_id"],
        )
        logger_mp.info(
            "RL transition sent id=%s done=%s intervention=%s reward_sum=%.1f",
            spec["transition_id"], spec["done"], spec["intervention"],
            float(np.sum(spec["rewards"])),
        )
    rewind = job.get("rewind")
    if rewind:
        remote.rewind(
            rewind,
            episode_id=int(rewind.get("episode_id", 0)),
            chunk_id=int(rewind.get("chunk_id", 0)),
        )
        logger_mp.info(
            "RL rewind %s chunks=%d terminal=%.2f",
            rewind.get("mode"), int(rewind.get("chunks", 0)),
            float(rewind.get("terminal_reward", 0.0)),
        )
    if discard_id and spec is None and not job.get("act"):
        return {"kind": "discard"}
    act = job.get("act")
    if act is None:
        # A finished episode has no next act. A takeover report doesn't either,
        # and must not be treated as the episode ending.
        if spec is not None and not spec.get("done"):
            return {"kind": "reported"}
        return {"kind": "closed"}
    frame, state, arm_q, episode_id, chunk_id = act
    reply = remote.act(
        frame, state, instruction, episode_id=episode_id, chunk_id=chunk_id,
    )
    model_actions = _truncate_model_actions(adapter, reply["actions"])
    queue = adapter.build_exec_queue(
        model_actions, arm_q,
        max_abs_arm_q=RL_MAX_ABS_ARM_Q,
        max_abs_gripper_q=RL_MAX_ABS_GRIPPER_Q,
    )
    logger_mp.info(
        "RL act chunk=%s id=%s queue=%d",
        chunk_id, reply["transition_id"], len(queue),
    )
    return {
        "kind": "act",
        "queue": queue,
        "transition_id": reply["transition_id"],
        "model_actions": model_actions,
        "hold_actions": bool(job.get("hold_actions")),
        "issued_chunk_id": None if act is None else int(act[4]),
    }


def _split_head_colors(head_img, camera_config):
    colors = {}
    if head_img is None or getattr(head_img, "bgr", None) is None:
        return colors
    if camera_config["head_camera"].get("binocular"):
        half = camera_config["head_camera"]["image_shape"][1] // 2
        colors["color_0"] = head_img.bgr[:, :half]
        colors["color_1"] = head_img.bgr[:, half:]
    else:
        colors["color_0"] = head_img.bgr
    return colors


def _start_record_episode(args, recorder, recording):
    if recorder is None or recording:
        return recording
    if recorder.create_episode():
        logger_mp.info("Recording policy episode to %s", recorder.episode_dir)
        return True
    logger_mp.error("Failed to create record episode.")
    return False


def _record_policy_frame(recorder, recording, camera_config, head_img, left_wrist_img, right_wrist_img,
                         current_q, command_q, left_grip, right_grip, phase):
    if recorder is None or not recording:
        return
    colors = _split_head_colors(head_img, camera_config)
    if left_wrist_img is not None and getattr(left_wrist_img, "bgr", None) is not None:
        colors["color_2"] = left_wrist_img.bgr
    if right_wrist_img is not None and getattr(right_wrist_img, "bgr", None) is not None:
        colors["color_3"] = right_wrist_img.bgr
    current_q = np.asarray(current_q, dtype=float).reshape(-1)
    command_q = np.asarray(command_q, dtype=float).reshape(-1)
    recorder.add_item(
        colors=colors,
        depths={},
        states={
            "left_arm": {"qpos": current_q[:7].tolist()},
            "right_arm": {"qpos": current_q[7:14].tolist()},
            "left_ee": {"qpos": [float(left_grip)]},
            "right_ee": {"qpos": [float(right_grip)]},
        },
        actions={
            "left_arm": {"qpos": command_q[:7].tolist()},
            "right_arm": {"qpos": command_q[7:14].tolist()},
            "left_ee": {"qpos": [float(left_grip)]},
            "right_ee": {"qpos": [float(right_grip)]},
            "phase": phase,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Policy rollout with rollback and XR handoff")
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--display-mode", type=str, choices=["immersive", "ego", "pass-through"],
                        default="ego")
    parser.add_argument("--img-server-ip", type=str, default="192.168.123.164")
    parser.add_argument("--network-interface", type=str, default=None)
    parser.add_argument("--server-host", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=15555)
    parser.add_argument("--protocol", type=str, choices=["ws", "zmq", "websocket"], default="zmq")
    parser.add_argument("--action-interp-factor", type=int, default=1)
    parser.add_argument("--exec-chunk-steps", type=int, default=0)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--rollback-seconds", type=float, default=3.0)
    parser.add_argument("--alignment-target-config",
                        default=os.path.join(REPO_ROOT, "configs", "alignment_targets.json"))
    parser.add_argument("--alignment-position-tolerance", type=float, default=0.04)
    parser.add_argument("--alignment-rotation-tolerance", type=float, default=0.20)
    parser.add_argument("--alignment-stable-seconds", type=float, default=0.5)
    parser.add_argument("--alignment-forward-offset", type=float, default=0.0)
    parser.add_argument("--alignment-handoff-seconds", type=float, default=0.3)
    parser.add_argument("--alignment-handoff-max-joint-speed", type=float, default=0.5)
    parser.add_argument("--ego-pixel-overlay", action="store_true")
    parser.add_argument("--policy-prefetch-steps", type=int, default=4,
                        help="Request the next chunk when this many actions remain")
    parser.add_argument(
        "--rl-online", action="store_true",
        help="Stage-2 online RL. Sends act, then transition after the chunk runs. "
             "Disables prefetch. Keyboard Y marks success, N marks failure.",
    )
    parser.add_argument(
        "--ready-pose-config",
        default=os.path.join(REPO_ROOT, "configs", "ready_pose.json"),
        help="14-joint raised startup pose shared with replay",
    )
    parser.add_argument(
        "--ready-pose-seconds", type=float, default=3.0,
        help="seconds to move from the current pose to the raised startup pose",
    )
    parser.add_argument("--swap-wrists", action="store_true", default=True)
    parser.add_argument("--no-swap-wrists", action="store_false", dest="swap_wrists")
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=["hand", "controller"],
        default="controller",
        help="XR tracking source (controller by default).",
    )
    parser.add_argument("--record", action="store_true", help="Save policy/teleop frames as episodes")
    parser.add_argument("--task-dir", type=str, default="/home/unitree/unitree_eai_environment/data/")
    parser.add_argument("--task-name", type=str, default="policy_rollout")
    parser.add_argument("--task-goal", type=str, default="")
    parser.add_argument("--task-desc", type=str, default="policy rollout with rollback handoff")
    parser.add_argument("--task-steps", type=str, default="")
    args = parser.parse_args()
    if args.record and not args.task_goal:
        args.task_goal = args.instruction
    if args.alignment_target_config and not os.path.isfile(args.alignment_target_config):
        args.alignment_target_config = None

    if args.alignment_handoff_seconds < 0.0:
        parser.error("--alignment-handoff-seconds must be non-negative")
    if args.alignment_handoff_max_joint_speed <= 0.0:
        parser.error("--alignment-handoff-max-joint-speed must be positive")
    if args.policy_prefetch_steps < 0:
        parser.error("--policy-prefetch-steps must be non-negative")
    if args.rl_online and args.policy_prefetch_steps != 0:
        logger_mp.warning(
            "RL online disables prefetch (was %d). The next act waits for this chunk's transition.",
            args.policy_prefetch_steps,
        )
        args.policy_prefetch_steps = 0
    RL_ONLINE = bool(args.rl_online)
    if args.frequency <= 0.0:
        parser.error("--frequency must be positive")
    if args.ready_pose_seconds <= 0.0:
        parser.error("--ready-pose-seconds must be positive")
    try:
        ready_pose_q = load_ready_pose(args.ready_pose_config)
    except ReadyPoseError as error:
        parser.error(str(error))

    from teleop.televuer.tv_wrapper import TeleVuerWrapper

    alignment_targets = load_targets(args.alignment_target_config)
    alignment = DualArmAlignment(
        alignment_targets,
        args.alignment_position_tolerance,
        args.alignment_rotation_tolerance,
        args.alignment_stable_seconds,
    )
    alignment.state = ACTIVE
    alignment_state_shared = Array("i", 1, lock=True)
    alignment_state_shared[0] = 2

    KEY_LISTENER_STOP.clear()
    listen_keyboard_thread = threading.Thread(
        target=_stdin_key_loop,
        args=(on_press, KEY_LISTENER_STOP),
        daemon=True,
    )
    listen_keyboard_thread.start()

    img_client = None
    remote = None
    tv_wrapper = None
    arm_ctrl = None
    recorder = None
    recording = False
    inference_thread = None
    inference_stop = False
    try:
        ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
        camera_config = img_client.get_cam_config()
        if args.record:
            recorder = EpisodeWriter(
                task_dir=os.path.join(args.task_dir, args.task_name),
                task_goal=args.task_goal,
                task_desc=args.task_desc,
                task_steps=args.task_steps,
                frequency=args.frequency,
                rerun_log=False,
            )
        use_webrtc = bool(camera_config["head_camera"]["enable_webrtc"]) and not args.ego_pixel_overlay
        use_zmq_display = args.ego_pixel_overlay or (
            not use_webrtc and bool(camera_config["head_camera"].get("enable_zmq", False))
        )
        if args.ego_pixel_overlay and not camera_config["head_camera"].get("enable_zmq", False):
            raise RuntimeError("--ego-pixel-overlay requires head_camera.enable_zmq=true")

        tv_wrapper = TeleVuerWrapper(
            use_hand_tracking=(args.input_mode == "hand"),
            binocular=camera_config["head_camera"]["binocular"],
            img_shape=camera_config["head_camera"]["image_shape"],
            display_mode=args.display_mode,
            zmq=use_zmq_display,
            webrtc=use_webrtc and not args.ego_pixel_overlay,
            webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
            arm_reference_mode="head_yaw",
            alignment_targets=np.stack([alignment_targets["left"], alignment_targets["right"]]),
            alignment_current=np.stack([alignment_targets["left"], alignment_targets["right"]]),
            alignment_state_shared=alignment_state_shared,
        )

        bundle = create_dex1_arm_controller(simulation_mode=False, use_waist=False)
        arm_ctrl = bundle.arm_ctrl
        xr_motion_data_ready = bundle.xr_motion_data_ready
        left_gripper_value = bundle.left_gripper_value
        right_gripper_value = bundle.right_gripper_value
        arm_ik = G1_29_ArmIK()

        hold_q = arm_ctrl.get_current_dual_arm_q()[:14].copy()
        hold_tau = arm_ik.solve_tau(hold_q)
        arm_ctrl.ctrl_dual_arm(hold_q, hold_tau)
        left_hold_grip, right_hold_grip = _read_grippers(arm_ctrl)
        arm_ctrl.set_policy_gripper_q(left_hold_grip, right_hold_grip)
        time.sleep(1.0)

        pad_joint_values = os.environ.get("PAD_JOINT_VALUES")
        adapter = PolicyAdapter(
            args.config_path,
            args.instruction,
            swap_wrists=args.swap_wrists,
            action_interp_factor=args.action_interp_factor,
            exec_chunk_steps=args.exec_chunk_steps,
            pad_joint_values=pad_joint_values,
        )
        # Connect lazily in the inference worker so startup and Q remain responsive.
        remote = PolicyRemoteClient(
            args.server_host,
            args.server_port,
            timeout_ms=180000 if args.rl_online else 10000,
            auto_connect=False,
            protocol=args.protocol,
        )
        rollback_buffer = PolicyRollbackBuffer(args.rollback_seconds, args.frequency)
        gamepad_a = ButtonRisingEdge()

        ego_overlay = None
        if args.ego_pixel_overlay:
            img_h, img_w = camera_config["head_camera"]["image_shape"][:2]
            half_w = img_w // 2 if camera_config["head_camera"].get("binocular", False) else img_w
            ego_overlay = EgoPixelOverlay(half_width=half_w, half_height=img_h)
            logger_mp.warning(ego_overlay.warning)

        prefetched_queue = []
        tracking_hint_logged = False
        logger_mp.info(
            "Connected (XR input-mode=%s). Raising both arms to the ready pose before inference.",
            args.input_mode,
        )

        action_queue = []
        rollback_sequence = []
        policy_inference_enabled = False
        rollout = RLTRollout() if args.rl_online else None
        takeover = TakeoverChunk()
        # arming: an act is in flight whose actions must not move the arm.
        # after: what to do when that act returns ("discard", "resume", or None).
        # rows: human commands captured while that act is still in flight.
        takeover_gate = {"arming": False, "after": None, "rows": [], "chunk_len": 64, "issued_chunk_id": 0}
        rl_report_due = False
        rl_inflight = None
        last_arm_q = hold_q.copy()
        last_tau = hold_tau.copy()
        last_left_grip = left_hold_grip
        last_right_grip = right_hold_grip
        handoff = new_handoff_runtime()
        resume_wait_log_time = 0.0
        alignment_log_time = 0.0
        inference_lock = threading.Lock()
        inference_state = {
            "thread": None,
            "result": None,
            "error": None,
            "generation": 0,
            "accept_result": True,
            "stop": False,
        }

        def inference_worker(request, generation):
            try:
                if isinstance(request, dict) and request.get("rlt"):
                    result = _run_rlt_job(adapter, remote, request, args.instruction)
                    if request.get("transition") is not None and rollout is not None:
                        rollout.note_stored()
                    discard_id = None
                    with inference_lock:
                        fresh = (
                            generation == inference_state["generation"]
                            and inference_state["accept_result"]
                        )
                        if fresh:
                            inference_state["result"] = result
                            inference_state["error"] = None
                        elif result.get("kind") == "act":
                            discard_id = result.get("transition_id")
                    if discard_id:
                        try:
                            remote.discard(discard_id)
                            logger_mp.info("Discarded stale RL act %s", discard_id)
                        except Exception as discard_error:
                            logger_mp.error(
                                "Failed to discard stale RL act %s: %s",
                                discard_id, discard_error,
                            )
                    return
                result = _predict_policy_queue(adapter, remote, request, args.instruction)
                with inference_lock:
                    if (generation == inference_state["generation"] and
                            inference_state["accept_result"]):
                        inference_state["result"] = result
                        inference_state["error"] = None
            except Exception as error:
                with inference_lock:
                    if (generation == inference_state["generation"] and
                            inference_state["accept_result"]):
                        inference_state["result"] = None
                        inference_state["error"] = error
            finally:
                with inference_lock:
                    if inference_state["thread"] is threading.current_thread():
                        inference_state["thread"] = None

        def start_inference(request):
            with inference_lock:
                if inference_state["thread"] is not None or inference_state["stop"]:
                    return False
                generation = inference_state["generation"]
                inference_state["accept_result"] = True
                worker = threading.Thread(
                    target=inference_worker, args=(request, generation), daemon=True
                )
                inference_state["thread"] = worker
                worker.start()
                return True

        def invalidate_inference():
            with inference_lock:
                inference_state["generation"] += 1
                inference_state["accept_result"] = False
                inference_state["result"] = None
                inference_state["error"] = None

        def inference_busy():
            with inference_lock:
                return inference_state["thread"] is not None

        def clear_handoff_runtime():
            global ALIGNMENT_STATE
            alignment.state = ACTIVE
            ALIGNMENT_STATE = alignment.state
            alignment_state_shared[0] = 2
            reset_handoff_runtime(handoff)

        def _rlt_act_request(request):
            frame, state, arm_q = request
            return {
                "rlt": True,
                "act": (frame, state, arm_q, rollout.episode_id, rollout.next_chunk_id),
            }

        def _rlt_transition_job(chunk, frame, state, outcome, intervention):
            actions = np.asarray(chunk["actions"], dtype=np.float32)
            executed = int(chunk.get("executed_steps", actions.shape[0]))
            executed = max(0, min(executed, int(actions.shape[0])))
            actions = actions[:executed]
            fields = transition_fields(int(actions.shape[0]), outcome, intervention)
            return {
                "rlt": True,
                "transition": {
                    "transition_id": chunk["transition_id"],
                    "next_frame": frame,
                    "next_state": state,
                    "action_chunk": actions,
                    "episode_id": chunk["episode_id"],
                    "chunk_id": chunk["chunk_id"],
                    **fields,
                },
            }

        def _arm_takeover():
            if (not args.rl_online or takeover.active or takeover_gate["arming"]
                    or inference_busy()):
                return False
            try:
                request = _rlt_act_request(
                    _prepare_policy_request(adapter, img_client, arm_ctrl)
                )
            except Exception as error:
                logger_mp.error("Takeover observation failed: %s", error)
                return False
            request["hold_actions"] = True
            takeover_gate["issued_chunk_id"] = int(request["act"][4])
            takeover_gate["rows"] = []
            if not start_inference(request):
                return False
            takeover_gate["arming"] = True
            return True

        def _takeover_job(identity, actions, next_act):
            frame, state, arm_q = _prepare_policy_request(adapter, img_client, arm_ctrl)
            if actions.shape[0] == 0:
                job = {"rlt": True, "discard_id": identity["transition_id"]}
            else:
                job = _rlt_transition_job(
                    {
                        "transition_id": identity["transition_id"],
                        "episode_id": identity["episode_id"],
                        "chunk_id": identity["chunk_id"],
                        "actions": actions,
                        "executed_steps": int(actions.shape[0]),
                    },
                    frame, state, None, True,
                )
            if next_act:
                job["act"] = (
                    frame, state, arm_q, rollout.episode_id, rollout.next_chunk_id,
                )
                job["hold_actions"] = next_act == "hold"
                if next_act == "hold":
                    takeover_gate["issued_chunk_id"] = int(job["act"][4])
                    takeover_gate["rows"] = []
                    takeover_gate["arming"] = True
            return job

        def _commit_takeover(next_act):
            if not takeover.active:
                return False
            if inference_busy():
                logger_mp.error(
                    "Takeover report is busy; chunk %s was not sent",
                    takeover.transition_id,
                )
                return False
            closed = takeover.close()
            if closed is None:
                return False
            identity, actions, _executed = closed
            try:
                job = _takeover_job(identity, actions, next_act)
            except Exception as error:
                logger_mp.error("Takeover observation failed: %s", error)
                takeover_gate["arming"] = False
                return False
            if not start_inference(job):
                takeover_gate["arming"] = False
                logger_mp.error(
                    "Could not report takeover chunk %s", identity["transition_id"]
                )
                return False
            logger_mp.info(
                "Takeover chunk id=%s steps=%d next=%s",
                identity["transition_id"], int(actions.shape[0]), next_act or "none",
            )
            return True

        def _enter_policy_from_takeover():
            global RUN_PHASE, action_queue, policy_inference_enabled, recording
            global last_arm_q, last_tau, last_left_grip, last_right_grip
            if takeover_gate["arming"] or inference_busy():
                takeover_gate["after"] = "resume"
                return False
            if takeover.active:
                if not _commit_takeover("execute"):
                    return False
            else:
                try:
                    request = _rlt_act_request(
                        _prepare_policy_request(adapter, img_client, arm_ctrl)
                    )
                except Exception as error:
                    logger_mp.error("Policy resume observation failed: %s", error)
                    return False
                if not start_inference(request):
                    return False
            resume_q = arm_ctrl.get_current_dual_arm_q()[:14].copy()
            resume_left, resume_right = _read_grippers(arm_ctrl)
            arm_ik.reset_solution_state(resume_q)
            arm_ctrl.set_policy_gripper_q(resume_left, resume_right)
            action_queue = []
            policy_inference_enabled = True
            last_arm_q = resume_q.copy()
            last_tau = arm_ik.solve_tau(resume_q).copy()
            last_left_grip, last_right_grip = resume_left, resume_right
            clear_handoff_runtime()
            RUN_PHASE = POLICY_LIVE
            recording = _start_record_episode(args, recorder, recording)
            logger_mp.info("Policy resumed from teleop; waiting for chunk.")
            return True

        def _accept_hold(result):
            global RESUME_POLICY
            takeover_gate["arming"] = False
            follow = takeover_gate["after"]
            takeover_gate["after"] = None
            if follow == "discard" or RUN_PHASE != TELEOP_LIVE:
                discard = {"rlt": True, "discard_id": result["transition_id"]}
                takeover_gate["rows"] = []
                if not start_inference(discard):
                    logger_mp.error(
                        "Could not discard unused takeover act %s",
                        result["transition_id"],
                    )
                return
            chunk_len = int(np.asarray(result["model_actions"]).shape[0])
            takeover_gate["chunk_len"] = chunk_len
            issued = result.get("issued_chunk_id")
            if issued is None:
                issued = takeover_gate["issued_chunk_id"]
            takeover.open(
                result["transition_id"], rollout.episode_id, int(issued), chunk_len,
            )
            rollout.next_chunk_id = max(rollout.next_chunk_id, int(issued) + 1)
            for row in takeover_gate["rows"][:chunk_len]:
                takeover.push(row)
            takeover_gate["rows"] = []
            logger_mp.info(
                "Takeover recording id=%s steps_already=%d",
                result["transition_id"], takeover.steps,
            )
            if follow == "resume":
                if _enter_policy_from_takeover():
                    RESUME_POLICY = False
                return
            if takeover.steps >= takeover.chunk_len:
                _commit_takeover("hold")

        def _pop_hold_result():
            with inference_lock:
                ready = inference_state["result"]
                if not (isinstance(ready, dict) and ready.get("hold_actions")):
                    return None
                inference_state["result"] = None
                inference_state["error"] = None
            return ready

        def try_resume_policy(from_label):
            global RUN_PHASE
            global action_queue, policy_inference_enabled
            global last_arm_q, last_tau, last_left_grip, last_right_grip
            global resume_wait_log_time, recording
            if inference_busy():
                now = time.monotonic()
                if now - resume_wait_log_time >= 1.0:
                    logger_mp.info(
                        "Policy resume queued; waiting for previous inference request to finish."
                    )
                    resume_wait_log_time = now
                return False
            resume_q, resume_tau, resume_left_grip, resume_right_grip = (
                _start_policy_from_current_pose(
                    adapter, img_client, arm_ctrl, arm_ik,
                    inference_lock, inference_state, start_inference,
                    request_fn=_rlt_act_request if args.rl_online else None,
                )
            )
            action_queue = []
            policy_inference_enabled = True
            last_arm_q = resume_q.copy()
            last_tau = resume_tau.copy()
            last_left_grip = resume_left_grip
            last_right_grip = resume_right_grip
            clear_handoff_runtime()
            RUN_PHASE = POLICY_LIVE
            recording = _start_record_episode(args, recorder, recording)
            logger_mp.info("Policy resumed from %s; waiting for chunk.", from_label)
            return True

        logger_mp.info(
            "Moving both arms to the raised ready pose (%.1fs). Press Q to abort.",
            args.ready_pose_seconds,
        )
        ready_result = move_to_ready_pose(
            arm_ctrl,
            arm_ik,
            ready_pose_q,
            args.ready_pose_seconds,
            args.frequency,
            grippers=(last_left_grip, last_right_grip),
            stop_requested=lambda: STOP,
        )
        last_arm_q = ready_result.arm_q.copy()
        last_tau = ready_result.tau.copy()
        if not ready_result.completed:
            logger_mp.warning("Ready-pose motion interrupted; policy was not started.")
            STOP = True
        else:
            if START_POLICY:
                logger_mp.info(
                    "S during the ready-pose motion was ignored. Press S again to start inference."
                )
            START_POLICY = False
            logger_mp.info(
                "Ready pose reached and held. Press keyboard S to start inference. "
                "B=rollback, gamepad A=take over, gamepad A again (or S)=return policy, Q=quit.%s",
                " Y=success, N=failure." if args.rl_online else "",
            )

        while not STOP:
            loop_start = time.time()
            head_img = img_client.get_head_frame() if (use_zmq_display or args.record) else None
            left_wrist_img = img_client.get_left_wrist_frame() if args.record else None
            right_wrist_img = img_client.get_right_wrist_frame() if args.record else None

            tele_data = tv_wrapper.get_tele_data()
            if gamepad_a.update(getattr(tele_data, "right_ctrl_aButton", False)):
                on_press("a")
            aligned_left_pose = tele_data.left_wrist_pose_openxr
            aligned_right_pose = tele_data.right_wrist_pose_openxr
            if args.alignment_forward_offset:
                aligned_left_pose = tv_wrapper.apply_openxr_forward_offset(
                    aligned_left_pose, tele_data.head_pose, args.alignment_forward_offset
                )
                aligned_right_pose = tv_wrapper.apply_openxr_forward_offset(
                    aligned_right_pose, tele_data.head_pose, args.alignment_forward_offset
                )
            aligned_left_waist_pose = tv_wrapper.openxr_controller_to_waist(
                aligned_left_pose, tele_data.head_pose
            )
            aligned_right_waist_pose = tv_wrapper.openxr_controller_to_waist(
                aligned_right_pose, tele_data.head_pose
            )
            with xr_motion_data_ready.get_lock():
                xr_motion_data_ready.value = tele_data.motion_data_ready
            tracking_valid = is_tracking_valid(tele_data)
            blending = RUN_PHASE == TELEOP_LIVE and is_blending(handoff)
            if RUN_PHASE == TELEOP_LIVE and not blending:
                _write_xr_grippers(
                    left_gripper_value, right_gripper_value, tele_data, args.input_mode
                )
            START_POLICY, RESUME_POLICY, ALIGN_CONFIRM, ROLLBACK_REQUEST = stale_key_flags(
                RUN_PHASE, START_POLICY, RESUME_POLICY, ALIGN_CONFIRM, ROLLBACK_REQUEST,
            )

            if START_POLICY and RUN_PHASE == POLICY_IDLE:
                START_POLICY = False
                action_queue = []
                prefetched_queue = []
                policy_inference_enabled = True
                RUN_PHASE = POLICY_LIVE
                recording = _start_record_episode(args, recorder, recording)
                if args.rl_online:
                    RL_OUTCOME_REQUEST = None
                    rl_report_due = False
                    episode_id = rollout.begin_episode()
                    logger_mp.info(
                        "RL episode %d started from the ready pose. Y=success, N=failure.",
                        episode_id,
                    )
                else:
                    logger_mp.info(
                        "Policy rollout started from the ready pose and is requesting the first chunk."
                    )

            if args.rl_online:
                held = _pop_hold_result()
                if held is not None:
                    _accept_hold(held)

            if args.rl_online and RESUME_POLICY and RUN_PHASE == TELEOP_LIVE and (
                    takeover_gate["arming"] or takeover.active):
                try:
                    if _enter_policy_from_takeover():
                        RESUME_POLICY = False
                except Exception as resume_error:
                    logger_mp.error(
                        "Failed to resume policy from teleop; will retry: %s",
                        resume_error,
                    )
            elif ((START_POLICY and RUN_PHASE == ALIGNING) or
                    (RESUME_POLICY and RUN_PHASE == TELEOP_LIVE)):
                from_label = "alignment pose" if RUN_PHASE == ALIGNING else "teleop pose"
                try:
                    if try_resume_policy(from_label):
                        START_POLICY = False
                        RESUME_POLICY = False
                except Exception as resume_error:
                    logger_mp.error(
                        "Failed to resume policy from %s; will retry: %s",
                        from_label, resume_error,
                    )

            if ROLLBACK_REQUEST and RUN_PHASE in ROLLBACK_FROM_PHASES:
                ROLLBACK_REQUEST = False
                rl_event = None
                close_takeover = False
                if args.rl_online and RUN_PHASE == POLICY_LIVE:
                    rl_report_due = False
                    rl_inflight = None
                    RL_OUTCOME_REQUEST = None
                    rl_event = rollout.interrupt()
                elif args.rl_online and RUN_PHASE == TELEOP_LIVE:
                    close_takeover = takeover.active
                    if takeover_gate["arming"]:
                        takeover_gate["arming"] = False
                        takeover_gate["after"] = None
                        takeover_gate["rows"] = []
                policy_inference_enabled = False
                invalidate_inference()
                action_queue.clear()
                prefetched_queue.clear()
                if close_takeover:
                    _commit_takeover(None)
                raw_rollback = rollback_buffer.reverse_playback(exclude_latest=True)
                if args.rl_online and RUN_PHASE == POLICY_LIVE:
                    kind, chunk = rl_event if rl_event is not None else (None, None)
                    include_current = kind == "transition"
                    plan = rewind_plan(
                        len(raw_rollback),
                        takeover_gate["chunk_len"],
                        0 if rollout is None else rollout.stored_chunks,
                        include_current,
                    )
                    try:
                        job = {"rlt": True}
                        if kind == "discard":
                            job["discard_id"] = chunk["transition_id"]
                        elif kind == "transition":
                            frame, state, _arm_q = _prepare_policy_request(
                                adapter, img_client, arm_ctrl
                            )
                            # The cut chunk is still the policy's actions. The
                            # rewind correction, not an intervention flag, marks
                            # it as the bad branch.
                            job.update(_rlt_transition_job(chunk, frame, state, None, False))
                        if plan is not None and rollout is not None:
                            job["rewind"] = {
                                **plan,
                                "episode_id": rollout.episode_id,
                                "chunk_id": rollout.next_chunk_id,
                            }
                        if job.keys() != {"rlt"}:
                            if start_inference(job):
                                rl_inflight = job
                            else:
                                logger_mp.error("Could not report the rollback to the RL server.")
                    except Exception as report_error:
                        logger_mp.error(
                            "Failed to report interrupted RL chunk: %s", report_error
                        )
                reset_handoff_runtime(handoff)
                ease_steps = max(2, int(round(ROLLBACK_EASE_SECONDS * args.frequency)))
                rollback_sequence = ease_out_playback(raw_rollback, ease_steps)
                RUN_PHASE = POLICY_ROLLBACK
                logger_mp.info(
                    "Rollback requested: replaying %d frames; slowing the last %.2fs of the path to a stop",
                    len(rollback_sequence),
                    ROLLBACK_EASE_SECONDS,
                )

            if rollback_sequence:
                arm_q, tau, left_grip, right_grip = rollback_sequence.pop(0)
                arm_ctrl.ctrl_dual_arm(arm_q, tau)
                arm_ctrl.set_policy_gripper_q(left_grip, right_grip)
                last_arm_q, last_tau = arm_q.copy(), tau.copy()
                last_left_grip, last_right_grip = left_grip, right_grip
                _record_policy_frame(
                    recorder, recording, camera_config, head_img, left_wrist_img, right_wrist_img,
                    arm_ctrl.get_current_dual_arm_q()[:14], arm_q, left_grip, right_grip, RUN_PHASE,
                )
                time.sleep(max(0.0, (1.0 / args.frequency) - (time.time() - loop_start)))
                continue

            if RUN_PHASE == POLICY_ROLLBACK:
                endpoint_q, endpoint_tau, endpoint_left, endpoint_right = (
                    rollback_hold_from_last_command(
                        last_arm_q, last_tau, last_left_grip, last_right_grip
                    )
                )
                measured_q = arm_ctrl.get_current_dual_arm_q()[:14]
                logger_mp.info(
                    "Rollback complete; holding last command "
                    "(meas-cmd max=%.4f rad). Align controllers and press gamepad A.",
                    float(np.max(np.abs(np.asarray(measured_q, dtype=float) - endpoint_q))),
                )
                handoff["hold_q"] = endpoint_q
                handoff["hold_tau"] = endpoint_tau
                handoff["hold_grip"] = np.array(
                    [endpoint_left, endpoint_right], dtype=float
                )
                targets = rollback_endpoint_to_xr_targets(
                    arm_ik, tv_wrapper, endpoint_q, tele_data.head_pose,
                    args.alignment_forward_offset,
                )
                alignment.reset(targets)
                tv_wrapper.set_alignment_targets(targets["left"], targets["right"])
                handoff["overlay_left"], handoff["overlay_right"] = arm_ik.solve_fk_matrix(endpoint_q)
                ALIGNMENT_STATE = alignment.state
                alignment_state_shared[0] = alignment_state_code(alignment.state)
                alignment_log_time = 0.0
                RUN_PHASE = ALIGNING

            current_lr_arm_q = arm_ctrl.get_current_dual_arm_q()[:14]
            sol_q = last_arm_q.copy()
            sol_tauff = last_tau.copy()
            cmd_left_grip = last_left_grip
            cmd_right_grip = last_right_grip
            teleop_stepped = False

            if RUN_PHASE == POLICY_LIVE:
                try:
                    live_targets = rollback_endpoint_to_xr_targets(
                        arm_ik, tv_wrapper, current_lr_arm_q, tele_data.head_pose,
                        args.alignment_forward_offset,
                    )
                    tv_wrapper.set_alignment_targets(live_targets["left"], live_targets["right"])
                    tv_wrapper.set_alignment_current_targets(
                        live_targets["left"], live_targets["right"]
                    )
                except (ValueError, np.linalg.LinAlgError) as marker_error:
                    logger_mp.debug("Unable to update live EEF markers: %s", marker_error)

            if RUN_PHASE == POLICY_LIVE:
                with inference_lock:
                    ready_queue = inference_state["result"]
                    if ready_queue is not None:
                        inference_state["result"] = None
                    completed_error = inference_state["error"]
                    inference_state["error"] = None
                    worker_busy = inference_state["thread"] is not None
                if ready_queue is not None:
                    rl_inflight = None
                if isinstance(ready_queue, dict):
                    kind = ready_queue.get("kind")
                    if kind == "act" and ready_queue.get("hold_actions"):
                        _accept_hold(ready_queue)
                    elif kind == "act":
                        model_actions = ready_queue.get("model_actions")
                        if model_actions is not None:
                            takeover_gate["chunk_len"] = int(np.asarray(model_actions).shape[0])
                        action_queue = list(ready_queue["queue"])
                        rollout.accept_chunk(
                            ready_queue["transition_id"],
                            ready_queue["model_actions"],
                            len(action_queue),
                        )
                    elif kind == "closed":
                        policy_inference_enabled = False
                        RUN_PHASE = POLICY_IDLE
                        logger_mp.info(
                            "RL episode %d ended. Press S to start another.",
                            rollout.episode_id,
                        )
                elif ready_queue is not None:
                    if action_queue:
                        action_queue.extend(ready_queue)
                    else:
                        action_queue = ready_queue
                if completed_error is not None:
                    logger_mp.error("Policy inference failed; retrying: %s", completed_error)
                    if args.rl_online and rl_inflight is not None and start_inference(rl_inflight):
                        worker_busy = True
                        logger_mp.info("Retrying the same RL request.")
                    elif not args.rl_online:
                        # Keep rollout live and let the normal prefetch branch retry
                        # from the current robot pose on the next loop.
                        policy_inference_enabled = True
                    else:
                        policy_inference_enabled = True
                step = action_queue.pop(0) if action_queue else None
                chunk_just_finished = False
                if step is not None:
                    sol_q = step.arm_q
                    sol_tauff = arm_ik.solve_tau(sol_q)
                    cmd_left_grip = step.left_grip
                    cmd_right_grip = step.right_grip
                    if args.rl_online:
                        rollout.on_step()
                        if rollout.chunk_finished() and not action_queue:
                            rl_report_due = True
                            chunk_just_finished = True
                if args.rl_online:
                    if (rl_report_due and not chunk_just_finished
                            and not worker_busy and not action_queue):
                        try:
                            outcome = None
                            chunk = None
                            outcome = RL_OUTCOME_REQUEST
                            RL_OUTCOME_REQUEST = None
                            chunk = rollout.take_open()
                            if chunk is None:
                                rl_report_due = False
                            else:
                                frame, state, arm_q = _prepare_policy_request(
                                    adapter, img_client, arm_ctrl
                                )
                                fields = transition_fields(
                                    int(chunk["actions"].shape[0]), outcome, False
                                )
                                job = _rlt_transition_job(
                                    chunk, frame, state, outcome, False
                                )
                                if not fields["done"] and policy_inference_enabled:
                                    job["act"] = (
                                        frame, state, arm_q,
                                        rollout.episode_id, rollout.next_chunk_id,
                                    )
                                if start_inference(job):
                                    rl_inflight = job
                                    rl_report_due = False
                                    chunk = None
                                    if fields["done"]:
                                        policy_inference_enabled = False
                                else:
                                    rollout.open = chunk
                                    chunk = None
                                    RL_OUTCOME_REQUEST = outcome
                                    logger_mp.error(
                                        "RL transition worker is busy; will retry chunk %s",
                                        rollout.open["transition_id"],
                                    )
                        except Exception as request_error:
                            if chunk is not None:
                                rollout.open = chunk
                            if outcome is not None or RL_OUTCOME_REQUEST is None:
                                RL_OUTCOME_REQUEST = outcome
                            logger_mp.error(
                                "RL transition observation failed; retrying: %s",
                                request_error,
                            )
                    elif (policy_inference_enabled and rollout.open is None
                            and not rl_report_due and not action_queue and not worker_busy):
                        try:
                            request = _rlt_act_request(
                                _prepare_policy_request(adapter, img_client, arm_ctrl)
                            )
                            if start_inference(request):
                                rl_inflight = request
                            else:
                                logger_mp.debug("Policy inference worker is still busy.")
                        except Exception as request_error:
                            logger_mp.error(
                                "Policy observation failed; retrying: %s", request_error
                            )
                            policy_inference_enabled = True
                elif (policy_inference_enabled and
                        len(action_queue) <= args.policy_prefetch_steps and
                        not worker_busy):
                    try:
                        request = _prepare_policy_request(adapter, img_client, arm_ctrl)
                        if not start_inference(request):
                            logger_mp.debug("Policy inference worker is still busy.")
                    except Exception as request_error:
                        logger_mp.error("Policy observation failed; retrying: %s", request_error)
                        policy_inference_enabled = True

            elif RUN_PHASE == ALIGNING:
                sol_q = hold_arm_cmd(handoff["hold_q"], last_arm_q)
                sol_tauff = hold_tau_cmd(handoff["hold_tau"], last_tau)
                cmd_left_grip, cmd_right_grip = hold_gripper_cmd(
                    handoff["hold_grip"], last_left_grip, last_right_grip
                )
                alignment.update(aligned_left_pose, aligned_right_pose, tracking_valid)
                if tracking_valid:
                    tv_wrapper.set_alignment_current_targets(
                        aligned_left_pose, aligned_right_pose
                    )
                elif handoff["hold_q"] is not None:
                    try:
                        hold_left, hold_right = arm_ik.solve_fk_matrix(
                            np.asarray(handoff["hold_q"][:14], dtype=float)
                        )
                        hold_targets = {
                            "left": tv_wrapper.waist_pose_to_openxr_controller(
                                hold_left, tele_data.head_pose, "left"
                            ),
                            "right": tv_wrapper.waist_pose_to_openxr_controller(
                                hold_right, tele_data.head_pose, "right"
                            ),
                        }
                        tv_wrapper.set_alignment_current_targets(
                            hold_targets["left"], hold_targets["right"]
                        )
                    except (ValueError, np.linalg.LinAlgError):
                        pass
                now = time.monotonic()
                if now - alignment_log_time >= 1.0:
                    errors = alignment.last_errors
                    error_text = "left=(n/a, n/a), right=(n/a, n/a)" if errors is None else (
                        f"left=({errors[0][0]:.4f} m, {errors[0][1]:.4f} rad), "
                        f"right=({errors[1][0]:.4f} m, {errors[1][1]:.4f} rad)"
                    )
                    logger_mp.warning(
                        "Alignment pending: %s; tracking_valid=%s ready=%s "
                        "left_valid=%s right_valid=%s source=%s",
                        error_text,
                        tracking_valid,
                        tele_data.motion_data_ready,
                        tele_data.left_wrist_valid,
                        tele_data.right_wrist_valid,
                        getattr(tele_data, "tracking_source", ""),
                    )
                    if not tracking_valid and not tracking_hint_logged:
                        tracking_hint_logged = True
                        logger_mp.warning(
                            "No XR tracking yet. Keep Vuer in XR and provide the configured "
                            "%s tracking input. Press S to skip alignment and resume policy.",
                            args.input_mode,
                        )
                    alignment_log_time = now
                if ALIGN_CONFIRM:
                    ALIGN_CONFIRM = False
                    if not tracking_valid:
                        logger_mp.warning("A rejected: both controller poses must be valid at handoff.")
                    else:
                        was_aligned = alignment.state == ALIGNED
                        if not was_aligned:
                            logger_mp.warning("A accepted before thresholds; using current controller pose.")
                        hold_arm_q = hold_arm_cmd(handoff["hold_q"], last_arm_q)
                        robot_left_start, robot_right_start = arm_ik.solve_fk_matrix(hold_arm_q)
                        if not was_aligned:
                            alignment.state = ALIGNED
                        if not alignment.confirm(
                                aligned_left_waist_pose, aligned_right_waist_pose,
                                robot_left_start, robot_right_start):
                            raise RuntimeError("Unable to confirm alignment handoff")
                        arm_ik.reset_solution_state(hold_arm_q)
                        handoff["previous_q"] = hold_arm_q.copy()
                        handoff["last_command_q"] = hold_arm_q.copy()
                        handoff["blend_started"] = time.monotonic()
                        handoff["max_delta"] = 0.0
                        handoff["tracking_lost"] = False
                        handoff["overlay_left"] = None
                        handoff["overlay_right"] = None
                        RUN_PHASE = TELEOP_LIVE
                        blending = True
                        A_DEBOUNCE_UNTIL = time.monotonic() + A_HANDOFF_DEBOUNCE_S
                        logger_mp.info(
                            "Handoff confirmed. Release gamepad A, then press it again after teleop is active to return policy."
                        )
                ALIGNMENT_STATE = alignment.state
                alignment_state_shared[0] = alignment_state_code(alignment.state)

            elif RUN_PHASE == TELEOP_LIVE:
                teleop_stepped = True
                blending = is_blending(handoff)
                result = compute_relative_arm_command(
                    alignment, arm_ik,
                    aligned_left_waist_pose, aligned_right_waist_pose,
                    tracking_valid, blending, handoff["tracking_lost"],
                    time.monotonic(), handoff["blend_started"],
                    args.alignment_handoff_seconds,
                    args.alignment_handoff_max_joint_speed, args.frequency,
                    handoff["previous_q"], handoff["last_command_q"],
                    last_arm_q, current_lr_arm_q,
                )
                sol_q = result.sol_q
                sol_tauff = result.sol_tauff
                handoff["tracking_lost"] = result.tracking_lost
                handoff["blend_started"] = result.blend_started
                handoff["max_delta"] = max(handoff["max_delta"], result.max_delta)
                handoff["last_command_q"] = np.asarray(sol_q[:14], dtype=float).copy()
                if result.reanchored:
                    logger_mp.info("Tracking recovered; relative handoff reanchored.")
                if result.tracking_lost:
                    now = time.monotonic()
                    if now - handoff["warning_time"] >= 1.0:
                        logger_mp.warning("Tracking lost during handoff; holding last arm command.")
                        handoff["warning_time"] = now
                if blending:
                    cmd_left_grip, cmd_right_grip = hold_gripper_cmd(
                        handoff["hold_grip"], last_left_grip, last_right_grip
                    )

            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
            blending = RUN_PHASE == TELEOP_LIVE and is_blending(handoff)
            if RUN_PHASE == TELEOP_LIVE and not blending:
                arm_ctrl.clear_policy_gripper_direct()
            elif RUN_PHASE != POLICY_IDLE:
                arm_ctrl.set_policy_gripper_q(cmd_left_grip, cmd_right_grip)

            if blending and teleop_stepped:
                handoff["previous_q"] = np.asarray(sol_q[:14], dtype=float).copy()
                handoff["last_command_q"] = handoff["previous_q"].copy()
                elapsed = time.monotonic() - handoff["blend_started"]
                if blend_should_finish(
                        elapsed, args.alignment_handoff_seconds,
                        tracking_valid, arm_ik.last_solve_succeeded):
                    robot_left_now, robot_right_now = arm_ik.solve_fk_matrix(handoff["previous_q"])
                    alignment.reanchor(
                        aligned_left_waist_pose, aligned_right_waist_pose,
                        robot_left_now, robot_right_now,
                    )
                    logger_mp.info(
                        "Handoff blend complete: max step=%.6f rad",
                        handoff["max_delta"],
                    )
                    handoff["hold_q"] = None
                    handoff["hold_grip"] = None
                    handoff["previous_q"] = None
                    handoff["blend_started"] = None
                    blending = False
                    _write_xr_grippers(
                        left_gripper_value, right_gripper_value, tele_data, args.input_mode
                    )
                    arm_ctrl.clear_policy_gripper_direct()
                    A_DEBOUNCE_UNTIL = max(
                        A_DEBOUNCE_UNTIL, time.monotonic() + A_TELEOP_READY_DEBOUNCE_S
                    )
                    logger_mp.info(
                        "Teleoperation handoff active. Human joint commands are recorded for RL. "
                        "Press gamepad A again to return control to policy."
                    )
                    if args.rl_online:
                        _arm_takeover()

            if RUN_PHASE == POLICY_LIVE:
                rollback_buffer.append(sol_q[:14], sol_tauff[:14], cmd_left_grip, cmd_right_grip)
            elif RUN_PHASE == TELEOP_LIVE:
                if blending:
                    rollback_buffer.append(sol_q[:14], sol_tauff[:14], cmd_left_grip, cmd_right_grip)
                else:
                    tele_left_grip, tele_right_grip = _read_grippers(arm_ctrl)
                    rollback_buffer.append(sol_q[:14], sol_tauff[:14], tele_left_grip, tele_right_grip)
                    if args.rl_online:
                        command = qpos_command(sol_q[:14], tele_left_grip, tele_right_grip)
                        if takeover_gate["arming"]:
                            if len(takeover_gate["rows"]) < takeover_gate["chunk_len"]:
                                takeover_gate["rows"].append(command)
                        elif takeover.active and takeover.push(command):
                            _commit_takeover("hold")
                        elif not takeover.active:
                            _arm_takeover()

            last_arm_q = np.asarray(sol_q[:14], dtype=float).copy()
            last_tau = np.asarray(sol_tauff[:14], dtype=float).copy()
            last_left_grip, last_right_grip = cmd_left_grip, cmd_right_grip
            if RUN_PHASE != POLICY_IDLE:
                _record_policy_frame(
                    recorder, recording, camera_config, head_img, left_wrist_img, right_wrist_img,
                    current_lr_arm_q, sol_q, cmd_left_grip, cmd_right_grip, RUN_PHASE,
                )

            if use_zmq_display and head_img is not None and getattr(head_img, "bgr", None) is not None:
                try:
                    if ego_overlay is not None:
                        if RUN_PHASE == ALIGNING and handoff["overlay_left"] is not None:
                            overlay_bgr = ego_overlay.overlay_dual_arm(
                                head_img.bgr,
                                handoff["overlay_left"],
                                handoff["overlay_right"],
                                extra_left=[aligned_left_waist_pose, aligned_right_waist_pose],
                            )
                        else:
                            live_left, live_right = arm_ik.solve_fk_matrix(current_lr_arm_q)
                            overlay_bgr = ego_overlay.overlay_dual_arm(
                                head_img.bgr, live_left, live_right
                            )
                        tv_wrapper.render_to_xr(overlay_bgr)
                    else:
                        tv_wrapper.render_to_xr(head_img.bgr)
                except Exception as overlay_error:
                    logger_mp.warning("ego display failed: %s", overlay_error)
            elif RUN_PHASE == POLICY_IDLE and head_img is not None and getattr(head_img, "bgr", None) is not None:
                try:
                    tv_wrapper.render_to_xr(head_img.bgr)
                except Exception as preview_error:
                    logger_mp.warning("pre-start preview failed: %s", preview_error)

            time.sleep(max(0.0, (1.0 / args.frequency) - (time.time() - loop_start)))

    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt, exiting program...")
    except Exception as error:
        logger_mp.error("Error: %s", error, exc_info=True)
    finally:
        worker = None
        if "inference_state" in locals():
            with inference_lock:
                inference_state["stop"] = True
                worker = inference_state["thread"]
        if remote is not None:
            remote.close()
        if worker is not None:
            worker.join(timeout=2.0)
        try:
            if arm_ctrl is not None:
                arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as home_error:
            logger_mp.error("Failed to go home: %s", home_error)
        try:
            KEY_LISTENER_STOP.set()
            listen_keyboard_thread.join(timeout=1.0)
        except Exception as keyboard_error:
            logger_mp.error("Failed to stop keyboard listener: %s", keyboard_error)
        if tv_wrapper is not None:
            try:
                tv_wrapper.close()
            except Exception as vuer_error:
                logger_mp.error("Failed to close Vuer: %s", vuer_error)
        if recorder is not None:
            try:
                if recording:
                    recorder.save_episode(success=True)
                recorder.close()
            except Exception as record_error:
                logger_mp.error("Failed to close recorder: %s", record_error)
        if img_client is not None:
            img_client.close()
        logger_mp.info("Policy rollout handoff exited.")
