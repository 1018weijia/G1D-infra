"""Single-operator XR data collection. No policy rollout, rollback, or handoff."""
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO, 
                       file=True, 
                       file_path="/home/unitree/unitree_eai_environment/logs",
                       backup_count=100,
                       max_file_size=50*1024*1024,
                       file_name_format="{prog_name}_%Y%m%d.log",
                       )
logger_mp = logging_mp.getLogger(__name__)
import time
import argparse
from multiprocessing import Value, Array, Lock
import threading
import numpy as np
import os 
import sys
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_29_Arm_Internal_Dex1_Controller
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller, Dex1_1_Gripper_Controller
from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX, Inspire_Controller_FTP, Inspire_Controller_DFX_ctrl, Inspire_Controller_FTP_ctrl
from teleop.robot_control.robot_hand_brainco import Brainco_Controller_hand, Brainco_Controller_ctrl
from teleop.robot_control.mobile_control import G1_Mobile_Lift_Controller
from teleop.utils.instruction_map import ControlDataMapper, HandleInstruction

from teleop.teleimager.src.teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.rerun_visualizer import should_log_to_rerun
from teleop.utils.ipc import IPC_Server
from teleop.utils.controller_shortcuts import ControllerShortcutMapper, toggle_start_pause
# from teleop.utils.motion_switcher import MotionSwitcher
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
PAUSED         = False  # Hold current robot command so the operator can reset the scene
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
RECORD_DISCARD = False  # If True when stopping, mark the episode as failed
EPISODE_ID     = 0      # Episode ID (int) for IPC communication
RESUME_BLEND_SECONDS = 0.4
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: READY starts an episode; RECORD_RUNNING always permits stopping it.
#              F/left-X stops a running episode and labels it success=false.
#              R/right-A starts teleop, then toggles pause/resume when not recording.
#              Stopping an episode auto-pauses teleop so the scene can be reset.
#  --> auto  : Auto-transition after saving data.

def on_press(key, episode_id=None):
    global STOP, START, PAUSED, RECORD_TOGGLE, RECORD_DISCARD, EPISODE_ID
    if key == 'r':
        START, PAUSED, action = toggle_start_pause(START, PAUSED, RECORD_RUNNING)
        if action == "ignored_recording":
            logger_mp.warning("[on_press] R ignored: stop recording before pausing teleop.")
        elif action != "started":
            logger_mp.info(f"[on_press] teleop {action} for scene reset.")
    elif key == 'q':
        START = False
        PAUSED = False
        STOP = True
    elif key == 's' and START == True:
        if PAUSED and not RECORD_RUNNING:
            logger_mp.warning("[on_press] S ignored: resume teleop with A/R before recording.")
            return
        if episode_id is not None:
            EPISODE_ID = episode_id
        RECORD_DISCARD = False
        RECORD_TOGGLE = True
    elif key == 'f' and START == True:
        if RECORD_RUNNING:
            RECORD_DISCARD = True
            RECORD_TOGGLE = True
        else:
            logger_mp.warning("[on_press] F ignored: no running episode to mark as failed.")
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, PAUSED, RECORD_RUNNING, READY, RECORD_TOGGLE
    return {
        "START": START,
        "STOP": STOP,
        "PAUSED": PAUSED,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
        "RECORD_TOGGLE": RECORD_TOGGLE,
    }

def smoothstep_resume_gain(elapsed, duration):
    if duration <= 0.0:
        return 1.0
    progress = float(np.clip(elapsed / duration, 0.0, 1.0))
    return progress * progress * (3.0 - 2.0 * progress)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='controller', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1'], default='G1', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex1_internal', 'dex3', 'brainco', 'inspire_ftp', 'inspire_dfx'], help='Select end effector controller')
    # mobile base, elevation and waist control
    parser.add_argument('--base-type', type=str, choices=['mobile_lift', 'lift','legs'], default='mobile_lift', help='Select lower body type')
    parser.add_argument('--use-waist', action = 'store_true', help = 'Enable waist control')
    # mode flags
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--rerun', action='store_true', help='Stream episodes to a Rerun viewer (needs a display; off by default)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', default = True, help = 'Record episodes (default: on)')
    parser.add_argument('--no-record', action = 'store_false', dest = 'record', help = 'Teleoperate without writing episodes')
    parser.add_argument('--task-dir', type = str, default = '/home/unitree/unitree_eai_environment/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')

    args = parser.parse_args()
    logger_mp.info(f"args: {args}")

    from teleop.televuer.tv_wrapper import TeleVuerWrapper

    try:
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                      kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        # image client
        img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
        camera_config = img_client.get_cam_config()
        logger_mp.debug(f"Camera config: {camera_config}")

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=camera_config['head_camera']['binocular'],
                                     img_shape=camera_config['head_camera']['image_shape'],
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     webrtc=camera_config['head_camera']['enable_webrtc'],
                                     webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
                                     arm_reference_mode="head_yaw" # another choice is "head_position".
                                     )

        # Enter debug mode
        # motion_switcher = MotionSwitcher()
        # status, result = motion_switcher.Enter_Debug_Mode()
        # logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")
        
        xr_motion_data_ready = Value('b', False, lock=True)
        # arm
        if args.arm == "G1":
            arm_ik = G1_29_ArmIK()
            if args.ee == "dex1_internal":
                left_gripper_value = Value('d', 0.0, lock=True)        # [input]
                right_gripper_value = Value('d', 0.0, lock=True)       # [input]
                dual_gripper_data_lock = Lock()
                dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
                dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
                arm_ctrl = G1_29_Arm_Internal_Dex1_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock,
                                                              dual_gripper_state_array, dual_gripper_action_array,
                                                              simulation_mode=args.sim, use_waist=args.use_waist,
                                                              xr_motion_data_ready_in=xr_motion_data_ready)
            else:
                arm_ctrl = G1_29_ArmController(simulation_mode=args.sim, use_waist=args.use_waist)

        # end-effector
        if args.ee == "dex3":
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                          dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "dex1": # external
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, 
                                                     dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "inspire_dfx" and args.input_mode == "hand":
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_dfx" and args.input_mode == "controller":
            left_gripper_trigger_in = Value('d', 10.0, lock=True)  # [input]
            left_gripper_squeeze_in = Value('d', 0.0, lock=True)   # [input]
            right_gripper_trigger_in = Value('d', 10.0, lock=True) # [input]
            right_gripper_squeeze_in = Value('d', 0.0, lock=True)  # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX_ctrl(left_gripper_trigger_in, left_gripper_squeeze_in, right_gripper_trigger_in, right_gripper_squeeze_in,
                                                    dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "inspire_ftp" and args.input_mode == "hand":
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_ftp" and args.input_mode == "controller":
            left_gripper_trigger_in = Value('d', 10.0, lock=True)  # [input]
            left_gripper_squeeze_in = Value('d', 0.0, lock=True)   # [input]
            right_gripper_trigger_in = Value('d', 10.0, lock=True) # [input]
            right_gripper_squeeze_in = Value('d', 0.0, lock=True)  # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP_ctrl(left_gripper_trigger_in, left_gripper_squeeze_in, right_gripper_trigger_in, right_gripper_squeeze_in,
                                                    dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "brainco" and args.input_mode == "hand":
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller_hand(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                                dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "brainco" and args.input_mode == "controller":
            left_gripper_trigger_in = Value('d', 10.0, lock=True)  # [input]
            left_gripper_squeeze_in = Value('d', 0.0, lock=True)   # [input]
            right_gripper_trigger_in = Value('d', 10.0, lock=True) # [input]
            right_gripper_squeeze_in = Value('d', 0.0, lock=True)  # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller_ctrl(left_gripper_trigger_in, left_gripper_squeeze_in, right_gripper_trigger_in, right_gripper_squeeze_in,
                                                dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        else:
            pass

        # For mobile base and elevation control
        if args.base_type != "legs":
            try:
                mobile_ctrl = G1_Mobile_Lift_Controller(args.base_type, args.input_mode == "hand")
            except Exception as e:
                STOP = True
                logger_mp.error(f"Failed to initialize mobile base/lift controller: {e}")
                raise
        else:
            mobile_ctrl=None
        control_data_mapper = ControlDataMapper(arm_ctrl.get_current_waist_q()[0])
        handle_instruction = HandleInstruction(args.input_mode == "hand", tv_wrapper, mobile_ctrl)

        # simulation mode
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # record + headless / non-headless mode
        if args.record:
            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.frequency, 
                                     rerun_log = should_log_to_rerun(args.rerun, args.headless))

        logger_mp.info("Please enter the start signal (enter 'r' to start the subsequent program)")
        logger_mp.info("Controller shortcuts: right A=start/pause/resume, left Y=record toggle, left X=fail-stop, right B=quit")
        controller_shortcuts = ControllerShortcutMapper(on_press, get_state)
        last_sol_q = None
        last_sol_tauff = None
        held_sol_q = None
        held_sol_tauff = None
        resume_blend_t0 = None
        READY = True                  # now ready to (1) enter START state
        while not START and not STOP: # wait for start or stop signal.
            if args.input_mode == "controller":
                controller_shortcuts.update(tv_wrapper.get_tele_data())
            time.sleep(0.033)

        logger_mp.info("---------------------🚀start program🚀-------------------------")
        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()
            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record:
                    head_img = img_client.get_head_frame()
            if camera_config['left_wrist_camera']['enable_zmq']:
                if args.record:
                    left_wrist_img = img_client.get_left_wrist_frame()
            if camera_config['right_wrist_camera']['enable_zmq']:
                if args.record:
                    right_wrist_img = img_client.get_right_wrist_frame()

            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()
            if args.input_mode == "controller":
                controller_shortcuts.update(tele_data)
                if STOP:
                    break

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    RECORD_DISCARD = False
                    if recorder.create_episode(episode_id=EPISODE_ID if args.ipc else None):
                        RECORD_RUNNING = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    success = not RECORD_DISCARD
                    RECORD_DISCARD = False
                    RECORD_RUNNING = False
                    recorder.save_episode(success=success)
                    PAUSED = True
                    resume_blend_t0 = None
                    logger_mp.info("Episode stopped; teleop paused. Reset the scene, then press right A / R to resume.")
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)

            follow_xr = (not PAUSED)
            # logger_mp.info(f"tele_data: {tele_data}")
            if follow_xr and args.ee in ("dex3", "inspire_ftp", "inspire_dfx", "brainco") and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif follow_xr and args.ee in ("brainco", "inspire_dfx", "inspire_ftp") and args.input_mode == "controller":
                with left_gripper_trigger_in.get_lock():
                    left_gripper_trigger_in.value = tele_data.left_ctrl_triggerValue
                with left_gripper_squeeze_in.get_lock():
                    left_gripper_squeeze_in.value = tele_data.left_ctrl_squeezeValue
                with right_gripper_trigger_in.get_lock():
                    right_gripper_trigger_in.value = tele_data.right_ctrl_triggerValue
                with right_gripper_squeeze_in.get_lock():
                    right_gripper_squeeze_in.value = tele_data.right_ctrl_squeezeValue
            elif follow_xr and args.ee in ("dex1", "dex1_internal") and args.input_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif follow_xr and args.ee in ("dex1", "dex1_internal") and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            with xr_motion_data_ready.get_lock():
                xr_motion_data_ready.value = tele_data.motion_data_ready
            
            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            # current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()
            left_arm_pose_state, right_arm_pose_state = arm_ik.solve_fk(current_lr_arm_q)
            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.time()
            live_q, live_tauff = arm_ik.solve_ik(tele_data.left_wrist_pose, tele_data.right_wrist_pose, current_lr_arm_q)
            time_ik_end = time.time()
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")

            # For mobile base and elevation control
            height_state = None
            height_action = [0.0]
            move_state = None
            move_action = [0.0, 0.0]
            waist_state = None
            waist_action = None
            if mobile_ctrl is not None:
                height_state = mobile_ctrl.g1_height_state_array_out
                if follow_xr:
                    handle_instruction_data = handle_instruction.get_instruction()
                    vel_data = control_data_mapper.update(ry=handle_instruction_data['ry'])
                    height_action = np.array([vel_data['g1_height']]).tolist()
                    if args.base_type == "mobile_lift":
                        move_state = mobile_ctrl.g1_move_state_array_out
                        vel_data = control_data_mapper.update(lx=handle_instruction_data['lx'], ly=handle_instruction_data['ly'])
                        move_action = np.array([vel_data['mobile_x_vel'], vel_data['mobile_yaw_vel']]).tolist()
                else:
                    control_data_mapper.update(lx=0.0, ly=0.0, ry=0.0)
                    height_action = [0.0]
                    move_action = [0.0, 0.0]
                    if args.base_type == "mobile_lift":
                        move_state = mobile_ctrl.g1_move_state_array_out
                mobile_ctrl.g1_height_action_array_in[0] = height_action[0]
                if args.base_type == "mobile_lift":
                    mobile_ctrl.g1_move_action_array_in[0] = move_action[0]
                    mobile_ctrl.g1_move_action_array_in[1] = move_action[1]

            if args.use_waist:
                waist_state = arm_ctrl.get_current_waist_q()
                if follow_xr:
                    handle_instruction_data = handle_instruction.get_instruction()
                    vel_data = control_data_mapper.update(rx=handle_instruction_data['rx'], current_waist_yaw=waist_state[0])
                    waist_action = np.array([vel_data['waist_yaw_pos']], dtype=float)
                    live_q = np.concatenate([live_q, waist_action])
                elif held_sol_q is not None and np.asarray(held_sol_q).size > 14:
                    waist_action = np.asarray(held_sol_q, dtype=float)[14:15]
                else:
                    waist_action = np.asarray(waist_state[:1], dtype=float)
                    live_q = np.concatenate([live_q, waist_action])

            if PAUSED:
                if held_sol_q is None:
                    held_sol_q = np.asarray(last_sol_q if last_sol_q is not None else live_q, dtype=float).copy()
                    held_sol_tauff = np.asarray(last_sol_tauff if last_sol_tauff is not None else live_tauff, dtype=float).copy()
                sol_q = np.asarray(held_sol_q, dtype=float).copy()
                sol_tauff = np.asarray(held_sol_tauff, dtype=float).copy()
                resume_blend_t0 = None
            else:
                sol_q = np.asarray(live_q, dtype=float).copy()
                sol_tauff = np.asarray(live_tauff, dtype=float).copy()
                if held_sol_q is not None:
                    if resume_blend_t0 is None:
                        resume_blend_t0 = time.time()
                    gain = smoothstep_resume_gain(time.time() - resume_blend_t0, RESUME_BLEND_SECONDS)
                    hold_q = np.asarray(held_sol_q, dtype=float)
                    if hold_q.shape != sol_q.shape:
                        if hold_q.size < sol_q.size:
                            hold_q = np.concatenate([hold_q, sol_q[hold_q.size:]])
                        else:
                            hold_q = hold_q[:sol_q.size]
                    sol_q = (1.0 - gain) * hold_q + gain * sol_q
                    if gain >= 1.0:
                        held_sol_q = None
                        held_sol_tauff = None
                        resume_blend_t0 = None
                last_sol_q = sol_q.copy()
                last_sol_tauff = sol_tauff.copy()

            left_arm_pose_action, right_arm_pose_action = arm_ik.solve_fk(sol_q[:14])

            try:   
                arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
            except Exception as e:
                logger_mp.error(f"Failed to control arms with ik solution: {e}")
                raise e
            # record data
            if args.record:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                # dex hand or gripper
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                elif (args.ee in ("brainco", "inspire_dfx", "inspire_ftp") and args.input_mode == "controller"):
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[7:7+7]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    if camera_config['head_camera']['binocular']:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr[:, :camera_config['head_camera']['image_shape'][1]//2]
                            colors[f"color_{1}"] = head_img.bgr[:, camera_config['head_camera']['image_shape'][1]//2:]
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{2}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{3}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    else:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{1}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{2}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },       
                        "left_arm_pose": {
                            "qpos": left_arm_pose_state.tolist(),
                            "qvel": [],
                            "torque": [],
                        },
                        "right_arm_pose": {
                            "qpos": right_arm_pose_state.tolist(),
                            "qvel": [],
                            "torque": [],
                        },                  
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": {
                            "qpos": current_body_state,
                        }, 

                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        },     
                        "left_arm_pose": {
                            "qpos": left_arm_pose_action.tolist(),
                            "qvel": [],
                            "torque": [],
                        },
                        "right_arm_pose": {
                            "qpos": right_arm_pose_action.tolist(),
                            "qvel": [],
                            "torque": [],
                        },                     
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": {
                            "qpos": current_body_action,
                        }, 

                        
                    }
                    if mobile_ctrl != None:
                        states["torso"] = {
                            "height": np.array(height_state[0]).tolist(),
                            "qvel": np.array(height_state[1]).tolist()
                        }
                        actions["torso"] = {
                            "qvel": np.array(height_action[0]).tolist()
                        }
                        if args.base_type == "mobile_lift":
                            states["chassis"] = {
                                "qvel": np.array(move_state).tolist()  # [x_vel, yaw_vel]
                            }
                            actions["chassis"] = {
                                "qvel": np.array(move_action).tolist()   # [x_vel, yaw_vel]
                            }
                    if args.use_waist and waist_state is not None and waist_action is not None:
                        states["waist"] = {
                            "qpos": waist_state.tolist(),  # [yaw]
                        }
                        actions["waist"] = {
                            "qpos": waist_action.tolist(),  # [yaw]
                        }

                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt, exiting program...")
    except Exception as e:
        logger_mp.error(f"Error: {e}")
    finally:
        try:
            arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")
        
        try:
            if args.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        # try:
        #     if not args.motion:
        #         status, result = motion_switcher.Exit_Debug_Mode()
        #         logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        # except Exception as e:
        #     logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")
        
        try:
            if args.record:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        logger_mp.info("Finally, exiting program.")
