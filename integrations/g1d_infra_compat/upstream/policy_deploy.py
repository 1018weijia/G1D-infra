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


def on_press(key):
    global STOP, START_POLICY, ROLLBACK_REQUEST, ALIGN_CONFIRM, RESUME_POLICY
    global RUN_PHASE, ALIGNMENT_STATE, A_LAST_CHAR_AT
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
        adapter, img_client, arm_ctrl, arm_ik, inference_lock, inference_state, start_inference):
    """Capture the current robot pose and request the next policy chunk."""
    resume_q = arm_ctrl.get_current_dual_arm_q()[:14].copy()
    resume_left_grip, resume_right_grip = _read_grippers(arm_ctrl)
    resume_request = _prepare_policy_request(adapter, img_client, arm_ctrl)
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
            timeout_ms=10000,
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
                "B=rollback, gamepad A=take over, gamepad A again (or S)=return policy, Q=quit."
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
                logger_mp.info(
                    "Policy rollout started from the ready pose and is requesting the first chunk."
                )

            if ((START_POLICY and RUN_PHASE == ALIGNING) or
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
                policy_inference_enabled = False
                invalidate_inference()
                action_queue.clear()
                prefetched_queue.clear()
                reset_handoff_runtime(handoff)
                raw_rollback = rollback_buffer.reverse_playback(exclude_latest=True)
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
                    if action_queue:
                        action_queue.extend(ready_queue)
                    else:
                        action_queue = ready_queue
                if completed_error is not None:
                    logger_mp.error("Policy inference failed; retrying: %s", completed_error)
                    # Keep rollout live and let the normal prefetch branch retry
                    # from the current robot pose on the next loop.
                    policy_inference_enabled = True
                step = action_queue.pop(0) if action_queue else None
                if step is not None:
                    sol_q = step.arm_q
                    sol_tauff = arm_ik.solve_tau(sol_q)
                    cmd_left_grip = step.left_grip
                    cmd_right_grip = step.right_grip
                if (policy_inference_enabled and
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
                    logger_mp.info("Teleoperation handoff active. Press gamepad A again to return control to policy.")

            if RUN_PHASE == POLICY_LIVE:
                rollback_buffer.append(sol_q[:14], sol_tauff[:14], cmd_left_grip, cmd_right_grip)
            elif RUN_PHASE == TELEOP_LIVE:
                if blending:
                    rollback_buffer.append(sol_q[:14], sol_tauff[:14], cmd_left_grip, cmd_right_grip)
                else:
                    tele_left_grip, tele_right_grip = _read_grippers(arm_ctrl)
                    rollback_buffer.append(sol_q[:14], sol_tauff[:14], tele_left_grip, tele_right_grip)

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
