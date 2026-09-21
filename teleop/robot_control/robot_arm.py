import numpy as np
import threading
import time
from enum import IntEnum

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import ( LowCmd_  as hg_LowCmd, LowState_ as hg_LowState) # idl for g1-d
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

import logging_mp
logger_mp = logging_mp.getLogger(__name__)
from teleop.robot_control.dds_utils import wait_for_dds

kTopicLowCommand_Debug  = "rt/lowcmd"
kTopicLowState = "rt/lowstate"

G1_29_Num_Motors = 35
G1_29_Body_Motors = 29
 

class MotorState:
    def __init__(self):
        self.q = None
        self.dq = None

class G1_29_LowState:
    def __init__(self):
        self.mode_machine = None
        self.motor_state = [MotorState() for _ in range(G1_29_Num_Motors)]

class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def GetData(self):
        with self.lock:
            return self.data

    def SetData(self, data):
        with self.lock:
            self.data = data

class G1_29_ArmController:
    def __init__(self, simulation_mode = False, use_waist=False):
        logger_mp.info("Initialize G1_29_ArmController...")
        self.q_target = np.zeros(15)
        self.tauff_target = np.zeros(15)
        self.simulation_mode = simulation_mode

        self.kp_high = 300.0
        self.kd_high = 3.0
        self.kp_low = 80.0
        self.kd_low = 3.0
        self.kp_wrist = 40.0
        self.kd_wrist = 1.5

        self.all_motor_q = None
        self.arm_velocity_limit = 30.0
        self.control_dt = 1.0 / 250.0

        self.use_waist = use_waist

        self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()
        self.lowstate_sub_ready = False

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        wait_for_dds(lambda: self.lowstate_sub_ready, "G1_29_ArmController")

        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        self.all_motor_q = self.get_current_motor_q()
        logger_mp.info(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.info(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger_mp.info("Lock all joints except two arms...")
        arm_indices = set(member.value for member in G1_29_Arm_JointIndex)
        for id in G1_29_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            if id.value in arm_indices:
                if self._Is_wrist_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_wrist
                    self.msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
            else:
                if self._Is_weak_motor(id):
                    if self._Is_waistYaw(id):
                        self.msg.motor_cmd[id].kp = 100
                        self.msg.motor_cmd[id].kd = 2.0
                    else:
                        self.msg.motor_cmd[id].kp = self.kp_low
                        self.msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[id].kp = self.kp_high
                    self.msg.motor_cmd[id].kd = self.kd_high
            self.msg.motor_cmd[id].q  = self.all_motor_q[id]
            
            # if id.value == G1_29_Waist_JointIndex.kWaistYaw.value:
            #     self.q_target[-1] = self.msg.motor_cmd[id].q 
            #     self.tauff_target[-1] = 0
        logger_mp.info("Lock OK!\n")
        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize G1_29_ArmController OK!")

    def _subscribe_motor_state(self):
        while True:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = G1_29_LowState()
                lowstate.mode_machine = msg.mode_machine
                for id in range(G1_29_Num_Motors):
                    lowstate.motor_state[id].q  = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                self.lowstate_buffer.SetData(lowstate)
                self.lowstate_sub_ready = True
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_arm_waist_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_q_target

    def _ctrl_motor_state(self):
        while True:
            start_time = time.time()

            with self.ctrl_lock:
                q_target     = self.q_target.copy()
                tauff_target = self.tauff_target.copy()

            if self.simulation_mode:
                cliped_q_target = q_target
            else:
                cliped_q_target = self.clip_arm_q_target(q_target, velocity_limit = self.arm_velocity_limit)
                    
            for idx, id in enumerate(G1_29_Arm_Waist_JointIndex):
                self.msg.motor_cmd[id].q = cliped_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = tauff_target[idx]

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)

    def ctrl_dual_arm(self, q_target, tauff_target):
        '''Set control target values q & tau of the left and right arm motors.'''
        q_arr = np.atleast_1d(q_target)

        if q_arr.shape[0] == 14:
            with self.ctrl_lock:
                self.q_target[:14] = q_arr[:14]
                self.tauff_target[:14] = tauff_target[:14]
        elif q_arr.shape[0] == 15:
            with self.ctrl_lock:
                self.q_target[:15] = q_arr[:15]
                self.tauff_target[:14] = tauff_target[:14]
        else:
            raise ValueError(f"Invalid q_target shape: {q_arr.shape}")

    def get_mode_machine(self):
        '''Return current dds mode machine.'''
        return self.lowstate_buffer.GetData().mode_machine
    
    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        data = self.lowstate_buffer.GetData()
        return np.array([data.motor_state[id].q for id in G1_29_JointIndex])
    
    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        data = self.lowstate_buffer.GetData()
        return np.array([data.motor_state[id].q for id in G1_29_Arm_JointIndex])

    def get_current_waist_q(self):
        data = self.lowstate_buffer.GetData()
        return np.array([data.motor_state[id].q for id in G1_29_Waist_JointIndex])
    
    def get_current_arm_waist_q(self):
        '''Return current state q of the left and right arm and waist motors.'''
        data = self.lowstate_buffer.GetData()
        return np.array([data.motor_state[id].q for id in G1_29_Arm_Waist_JointIndex])

    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        data = self.lowstate_buffer.GetData()
        return np.array([data.motor_state[id].dq for id in G1_29_Arm_JointIndex])

    def ctrl_dual_arm_go_home(self):
        '''Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero.
        First moves waist to home position, then moves arms.'''
        logger_mp.info("[G1_29_ArmController] ctrl_dual_arm_go_home start...")
        tolerance = 0.05  # Tolerance threshold for joint angles to determine "close to zero"
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(15)
        while current_attempts < max_attempts:
            current_q = self.get_current_arm_waist_q()
            if np.all(np.abs(current_q) < tolerance):
                logger_mp.info("[G1_29_ArmController] Both arms and waist have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)
        
        if current_attempts >= max_attempts:
            logger_mp.warning("[G1_29_ArmController] Arms and waist did not reach home position within timeout.")

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            G1_29_JointIndex.kLeftAnklePitch.value,
            G1_29_JointIndex.kRightAnklePitch.value,
            # Left arm
            G1_29_JointIndex.kLeftShoulderPitch.value,
            G1_29_JointIndex.kLeftShoulderRoll.value,
            G1_29_JointIndex.kLeftShoulderYaw.value,
            G1_29_JointIndex.kLeftElbow.value,
            # Right arm
            G1_29_JointIndex.kRightShoulderPitch.value,
            G1_29_JointIndex.kRightShoulderRoll.value,
            G1_29_JointIndex.kRightShoulderYaw.value,
            G1_29_JointIndex.kRightElbow.value,

            # Waist
            G1_29_Waist_JointIndex.kWaistYaw.value,
        ]
        return motor_index.value in weak_motors
    
    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            G1_29_JointIndex.kLeftWristRoll.value,
            G1_29_JointIndex.kLeftWristPitch.value,
            G1_29_JointIndex.kLeftWristyaw.value,
            G1_29_JointIndex.kRightWristRoll.value,
            G1_29_JointIndex.kRightWristPitch.value,
            G1_29_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_motors

    def _Is_waistYaw(self, motor_index):
        waist_motors = [
            G1_29_Waist_JointIndex.kWaistYaw.value
        ]
        return motor_index.value in waist_motors
class G1_29_Arm_JointIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28

class G1_29_Waist_JointIndex(IntEnum):
    kWaistYaw = 12

class G1_29_Arm_Waist_JointIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28

    # Waist
    kWaistYaw = 12
    
class G1_29_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28
    
    # not used
    kNotUsedJoint0 = 29
    kNotUsedJoint1 = 30
    kNotUsedJoint2 = 31
    kNotUsedJoint3 = 32
    kNotUsedJoint4 = 33
    kNotUsedJoint5 = 34

class G1_29_Internal_Dex1_JointIndex(IntEnum):
    kLeftDex1_1 = 31
    kRightDex1_1 = 33

class G1_29_Arm_Internal_Dex1_Controller:
    def __init__(self, left_gripper_value_in, right_gripper_value_in, dual_gripper_data_lock=None, dual_gripper_state_out=None, dual_gripper_action_out=None,
                 simulation_mode=False, use_waist=False, xr_motion_data_ready_in=None):
        logger_mp.info("Initialize G1_29_Arm_Internal_Dex1_Controller...")

        self.left_gripper_value_in = left_gripper_value_in
        self.right_gripper_value_in = right_gripper_value_in
        self.xr_motion_data_ready_in = xr_motion_data_ready_in
        self.dual_gripper_data_lock = dual_gripper_data_lock
        self.dual_gripper_state_out = dual_gripper_state_out
        self.dual_gripper_action_out = dual_gripper_action_out

        self.simulation_mode = simulation_mode
        self.use_waist = use_waist

        self.left_index = G1_29_Internal_Dex1_JointIndex.kLeftDex1_1.value
        self.right_index = G1_29_Internal_Dex1_JointIndex.kRightDex1_1.value

        self.q_target = np.zeros(15)
        self.tauff_target = np.zeros(15)
        self.gripper_q_target = np.zeros(2)
        self._policy_gripper_direct = False

        self.control_dt = 1.0 / 250.0
        self.arm_velocity_limit = 30.0
        self.running = True

        self.kp_high = 300.0
        self.kd_high = 3.0
        self.kp_low = 80.0
        self.kd_low = 3.0
        self.kp_wrist = 40.0
        self.kd_wrist = 1.5
        self.gripper_kp = 5.0
        self.gripper_kd = 0.05

        from teleop.utils.weighted_moving_filter import WeightedMovingFilter
        self.smooth_filter = WeightedMovingFilter(np.array([0.5, 0.3, 0.2]), 2)

        self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()
        self.lowstate_sub_ready = False

        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        wait_for_dds(lambda: self.lowstate_sub_ready, "G1_29_Arm_Internal_Dex1_Controller")

        self.crc = CRC()
        self.ctrl_lock = threading.Lock()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        self.all_motor_q = self.get_current_motor_q()
        self.gripper_q_target = self.get_current_dual_gripper_q()
        logger_mp.info(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.info(f"Current arm and waist motor state q:\n{self.get_current_arm_waist_q()}\n")
        logger_mp.info(f"Current dual gripper q: {self.gripper_q_target}")

        arm_indices = set(member.value for member in G1_29_Arm_JointIndex)
        gripper_indices = {self.left_index, self.right_index}
        for id in G1_29_JointIndex:
            cmd = self.msg.motor_cmd[id]
            cmd.q = self.all_motor_q[id]
            cmd.dq = 0.0
            cmd.tau = 0.0
            if id.value in gripper_indices:
                cmd.mode = 1
                cmd.kp = self.gripper_kp
                cmd.kd = self.gripper_kd
            elif id.value >= G1_29_Body_Motors:
                cmd.mode = 0
                cmd.kp = 0.0
                cmd.kd = 0.0
            elif id.value in arm_indices:
                cmd.mode = 1
                cmd.kp = self.kp_wrist if self._Is_wrist_motor(id) else self.kp_low
                cmd.kd = self.kd_wrist if self._Is_wrist_motor(id) else self.kd_low
            else:
                cmd.mode = 1
                if self._Is_weak_motor(id):
                    if self._Is_waistYaw(id):
                        cmd.kp = 100
                        cmd.kd = 2.0
                    else:
                        cmd.kp = self.kp_low
                        cmd.kd = self.kd_low
                else:
                    cmd.kp = self.kp_high
                    cmd.kd = self.kd_high

            # if id.value == G1_29_Waist_JointIndex.kWaistYaw.value:
            #     self.q_target[-1] = cmd.q
            #     self.tauff_target[-1] = 0.0

        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize G1_29_Arm_Internal_Dex1_Controller OK!")

    def _subscribe_motor_state(self):
        while self.running:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                self.lowstate_buffer.SetData(msg)
                self.lowstate_sub_ready = True
            time.sleep(0.002)

    def _ctrl_motor_state(self):
        DELTA_GRIPPER_CMD = 0.18
        THUMB_INDEX_DISTANCE_MIN = 5.0
        THUMB_INDEX_DISTANCE_MAX = 7.0
        LEFT_MAPPED_MIN = 0.0
        RIGHT_MAPPED_MIN = 0.0
        LEFT_MAPPED_MAX = 5.40
        RIGHT_MAPPED_MAX = 5.40

        while self.running:
            start_time = time.time()
            with self.ctrl_lock:
                q_target = self.q_target.copy()
                tauff_target = self.tauff_target.copy()
            with self.left_gripper_value_in.get_lock():
                left_gripper_value = self.left_gripper_value_in.value
            with self.right_gripper_value_in.get_lock():
                right_gripper_value = self.right_gripper_value_in.value
            if self.xr_motion_data_ready_in is not None:
                with self.xr_motion_data_ready_in.get_lock():
                    xr_motion_data_ready = self.xr_motion_data_ready_in.value
            else:
                xr_motion_data_ready = True

            if self.simulation_mode:
                cliped_q_target = q_target
            else:
                cliped_q_target = self.clip_arm_q_target(q_target, velocity_limit=self.arm_velocity_limit)

            gripper_state = self.get_current_dual_gripper_q()
            if self._policy_gripper_direct:
                with self.ctrl_lock:
                    gripper_q_target = self.gripper_q_target.copy()
            elif xr_motion_data_ready:
                left_target_action = np.interp(left_gripper_value, [THUMB_INDEX_DISTANCE_MIN, THUMB_INDEX_DISTANCE_MAX], [LEFT_MAPPED_MIN, LEFT_MAPPED_MAX])
                right_target_action = np.interp(right_gripper_value, [THUMB_INDEX_DISTANCE_MIN, THUMB_INDEX_DISTANCE_MAX], [RIGHT_MAPPED_MIN, RIGHT_MAPPED_MAX])
                with self.ctrl_lock:
                    self.gripper_q_target = np.array([left_target_action, right_target_action])
                with self.ctrl_lock:
                    gripper_q_target = self.gripper_q_target.copy()
            else:
                with self.ctrl_lock:
                    self.gripper_q_target = gripper_state.copy()
                with self.ctrl_lock:
                    gripper_q_target = self.gripper_q_target.copy()
            gripper_q_cmd = np.clip(gripper_q_target, gripper_state - DELTA_GRIPPER_CMD, gripper_state + DELTA_GRIPPER_CMD)

            if self.smooth_filter:
                self.smooth_filter.add_data(gripper_q_cmd)
                gripper_q_cmd = self.smooth_filter.filtered_data

            for idx, id in enumerate(G1_29_Arm_Waist_JointIndex):
                self.msg.motor_cmd[id].q = cliped_q_target[idx]
                self.msg.motor_cmd[id].dq = 0.0
                self.msg.motor_cmd[id].tau = tauff_target[idx]

            for idx, id in enumerate((self.left_index, self.right_index)):
                self.msg.motor_cmd[id].mode = 1
                self.msg.motor_cmd[id].q = gripper_q_cmd[idx]
                self.msg.motor_cmd[id].dq = 0.0
                self.msg.motor_cmd[id].tau = 0.0
                self.msg.motor_cmd[id].kp = self.gripper_kp
                self.msg.motor_cmd[id].kd = self.gripper_kd

            if self.dual_gripper_state_out is not None and self.dual_gripper_action_out is not None:
                if self.dual_gripper_data_lock is not None:
                    with self.dual_gripper_data_lock:
                        self.dual_gripper_state_out[:] = gripper_state
                        self.dual_gripper_action_out[:] = gripper_q_cmd
                else:
                    self.dual_gripper_state_out[:] = gripper_state
                    self.dual_gripper_action_out[:] = gripper_q_cmd

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

            sleep_time = max(0.0, self.control_dt - (time.time() - start_time))
            time.sleep(sleep_time)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_arm_waist_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        return current_q + delta / max(motion_scale, 1.0)

    def set_policy_gripper_q(self, left_q, right_q):
        """Command internal grippers directly during policy rollout."""
        with self.ctrl_lock:
            self.gripper_q_target = np.array([float(left_q), float(right_q)], dtype=float)
            self._policy_gripper_direct = True

    def clear_policy_gripper_direct(self):
        """Return gripper control to XR/controller mapping."""
        self._policy_gripper_direct = False

    def ctrl_dual_arm(self, q_target, tauff_target):
        q_arr = np.atleast_1d(q_target)

        with self.ctrl_lock:
            if q_arr.shape[0] == 14:
                self.q_target[:14] = q_arr[:14]
                self.tauff_target[:14] = tauff_target[:14]
            elif q_arr.shape[0] == 15:
                self.q_target[:15] = q_arr[:15]
                self.tauff_target[:14] = tauff_target[:14]
            else:
                raise ValueError(f"Invalid q_target shape: {q_arr.shape}")

    def get_mode_machine(self):
        return self.lowstate_buffer.GetData().mode_machine

    def get_current_motor_q(self):
        data = self.lowstate_buffer.GetData()
        return np.array([data.motor_state[id].q for id in G1_29_JointIndex])

    def get_current_dual_arm_q(self):
        data = self.lowstate_buffer.GetData()
        return np.array([data.motor_state[id].q for id in G1_29_Arm_JointIndex])

    def get_current_waist_q(self):
        data = self.lowstate_buffer.GetData()
        return np.array([data.motor_state[id].q for id in G1_29_Waist_JointIndex])

    def get_current_arm_waist_q(self):
        data = self.lowstate_buffer.GetData()
        return np.array([data.motor_state[id].q for id in G1_29_Arm_Waist_JointIndex])

    def get_current_dual_arm_dq(self):
        data = self.lowstate_buffer.GetData()
        return np.array([data.motor_state[id].dq for id in G1_29_Arm_JointIndex])

    def get_current_dual_gripper_q(self):
        data = self.lowstate_buffer.GetData()
        return np.array([data.motor_state[self.left_index].q, data.motor_state[self.right_index].q])

    def ctrl_dual_arm_go_home(self):
        logger_mp.info("[G1_29_Arm_Internal_Dex1_Controller] ctrl_dual_arm_go_home start...")
        tolerance = 0.05
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(15)
        while current_attempts < max_attempts:
            current_q = self.get_current_arm_waist_q()
            if np.all(np.abs(current_q) < tolerance):
                logger_mp.info("[G1_29_Arm_Internal_Dex1_Controller] Both arms and waist have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)
        if current_attempts >= max_attempts:
            logger_mp.warning("[G1_29_Arm_Internal_Dex1_Controller] Arms and waist did not reach home position within timeout.")

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            G1_29_JointIndex.kLeftAnklePitch.value,
            G1_29_JointIndex.kRightAnklePitch.value,
            G1_29_JointIndex.kLeftShoulderPitch.value,
            G1_29_JointIndex.kLeftShoulderRoll.value,
            G1_29_JointIndex.kLeftShoulderYaw.value,
            G1_29_JointIndex.kLeftElbow.value,
            G1_29_JointIndex.kRightShoulderPitch.value,
            G1_29_JointIndex.kRightShoulderRoll.value,
            G1_29_JointIndex.kRightShoulderYaw.value,
            G1_29_JointIndex.kRightElbow.value,
            G1_29_Waist_JointIndex.kWaistYaw.value,
        ]
        return motor_index.value in weak_motors

    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            G1_29_JointIndex.kLeftWristRoll.value,
            G1_29_JointIndex.kLeftWristPitch.value,
            G1_29_JointIndex.kLeftWristyaw.value,
            G1_29_JointIndex.kRightWristRoll.value,
            G1_29_JointIndex.kRightWristPitch.value,
            G1_29_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_motors

    def _Is_waistYaw(self, motor_index):
        return motor_index.value in [G1_29_Waist_JointIndex.kWaistYaw.value]
