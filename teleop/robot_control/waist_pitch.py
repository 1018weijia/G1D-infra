"""G1-D waist pitch on its own DDS topic, independent of the lowcmd yaw path."""
from pathlib import Path
import threading
import time

import numpy as np
import pinocchio as pin
from multiprocessing import Value

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import MotorCmd_, MotorState_

import logging_mp

logger_mp = logging_mp.getLogger(__name__)
from teleop.robot_control.dds_utils import wait_for_dds


kTopicWaistPitchCmd = "rt/waistpitchcmd"
kTopicWaistPitchState = "rt/waistpitchstate"

# This repo's G1-D URDF names the bend joint waist_pitch_joint. The original
# teleop URDF called the same motor torso_Joint; that name is not in this file.
DEFAULT_URDF = Path(__file__).resolve().parents[2] / "assets" / "g1_D" / "g1_d.urdf"


class G1D_WaistPitchGravityFeedforward:
    """Gravity torque on the waist pitch joint, computed from the G1-D URDF.

    Only the pitch joint and the two arms are kept. Yaw, roll, the floating base,
    and the wheels sit at or below the pitch joint and are locked, so they do not
    enter the configuration vector. The arms are kept because holding them out in
    front changes the holding torque.
    """
    joint_names = ["waist_pitch_joint"] + [
        f"{side}_{name}_joint"
        for side in ("left", "right")
        for name in (
            "shoulder_pitch", "shoulder_roll", "shoulder_yaw",
            "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw",
        )
    ]

    def __init__(self, urdf_path=None):
        urdf_path = Path(urdf_path) if urdf_path is not None else DEFAULT_URDF
        if not urdf_path.is_file():
            raise FileNotFoundError(f"G1-D URDF not found: {urdf_path}")
        model = pin.buildModelFromUrdf(str(urdf_path))
        missing = [name for name in self.joint_names if model.existJointName(name) is False]
        if missing:
            raise ValueError(f"Waist pitch URDF is missing joints: {missing}")
        locked = [
            model.getJointId(name)
            for name in model.names[1:]
            if name not in self.joint_names
        ]
        self.model = pin.buildReducedModel(model, locked, pin.neutral(model))
        self.data = self.model.createData()
        self.idxs = np.array([
            self.model.joints[self.model.getJointId(name)].idx_q
            for name in self.joint_names
        ])
        self.zeros = np.zeros(self.model.nv)

    def compute(self, q):
        """q: [pitch, 14 arm joints] in URDF order. Returns the pitch holding torque."""
        full_q = pin.neutral(self.model)
        full_q[self.idxs] = q
        return pin.rnea(self.model, self.data, full_q, self.zeros, self.zeros)[self.idxs[0]]


class G1D_Mobile_Waist_Pitch_Controller:
    kp = 500.0
    kd = 12.0
    mode = 1                 # FOC with command-timeout protection
    min_angle = 0.0          # rad, upright
    max_angle = 2.3562       # rad, bent over
    travel_seconds = 5.0     # s to sweep the full range while a button is held
    DELTA_WAIST_PITCH_CMD = 0.10  # rad, cap on how far the command may lead the measured angle

    def __init__(self, arm_ctrl, fps=200.0, urdf_path=None):
        logger_mp.info("Initialize G1D_Mobile_Waist_Pitch_Controller...")
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.fps = fps
        # The arm angles swing the gravity load on the pitch joint, so the
        # feedforward reads them live from the arm controller.
        self.arm_ctrl = arm_ctrl
        self.waist_pitch_data_received = False
        # Holding a button sweeps the whole range in travel_seconds, so the speed
        # follows from the limits rather than being tuned separately.
        self.velocity = (self.max_angle - self.min_angle) / self.travel_seconds
        self.gravity_ff = G1D_WaistPitchGravityFeedforward(urdf_path)

        # Direction: -1 = pitch up, 0 = hold, +1 = bend over.
        self.g1_waist_pitch_action_value_in = Value('d', 0.0, lock=True)
        self.g1_waist_pitch_state_value_out = Value('d', 0.0, lock=True)
        self.g1_waist_pitch_action_value_out = Value('d', 0.0, lock=True)

        self.WaistPitchCmd_publisher = ChannelPublisher(kTopicWaistPitchCmd, MotorCmd_)
        self.WaistPitchCmd_publisher.Init()
        self.WaistPitchState_subscriber = ChannelSubscriber(kTopicWaistPitchState, MotorState_)
        self.WaistPitchState_subscriber.Init()

        self.subscribe_waist_pitch_state_thread = threading.Thread(target=self._subscribe_waist_pitch_state)
        self.subscribe_waist_pitch_state_thread.daemon = True
        self.subscribe_waist_pitch_state_thread.start()

        wait_for_dds(lambda: self.waist_pitch_data_received, "G1D_Mobile_Waist_Pitch_Controller")

        self.g1_waist_pitch_action_value_in.value = 0.0
        self.g1_waist_pitch_action_value_out.value = self.g1_waist_pitch_state_value_out.value
        logger_mp.info(f"Current waist pitch q: {self.g1_waist_pitch_state_value_out.value:.4f} rad")

        self.running = True
        self.waist_pitch_control_thread = threading.Thread(target=self.control_thread)
        self.waist_pitch_control_thread.daemon = True
        self.waist_pitch_control_thread.start()

        logger_mp.info("Initialize G1D_Mobile_Waist_Pitch_Controller OK!\n")

    def _subscribe_waist_pitch_state(self):
        while True:
            try:
                waist_pitch_msg = self.WaistPitchState_subscriber.Read()
                if waist_pitch_msg is not None:
                    self.g1_waist_pitch_state_value_out.value = waist_pitch_msg.q
                    if not self.waist_pitch_data_received:
                        self.waist_pitch_data_received = True
            except Exception as e:
                logger_mp.info(f"[_subscribe_waist_pitch_state] Exception: {e}")
                time.sleep(0.1)
            time.sleep(0.005)

    def get_current_waist_pitch_state(self):
        return self.g1_waist_pitch_state_value_out.value

    def get_current_waist_pitch_action(self):
        return self.g1_waist_pitch_action_value_out.value

    def set_waist_pitch_buttons(self, button_bend, button_raise):
        # Pressing both buttons, or neither, is no request.
        direction = 0.0 if bool(button_bend) == bool(button_raise) else (1.0 if button_bend else -1.0)
        self.g1_waist_pitch_action_value_in.value = direction
        return direction

    def ctrl_waist_pitch_go_home(self, tolerance=0.05, timeout=6.0):
        logger_mp.info("[G1D_Mobile_Waist_Pitch_Controller] ctrl_waist_pitch_go_home start...")
        self.set_waist_pitch_buttons(button_bend=False, button_raise=True)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if abs(self.g1_waist_pitch_state_value_out.value - self.min_angle) < tolerance:
                logger_mp.info("[G1D_Mobile_Waist_Pitch_Controller] Waist pitch has reached the home position.")
                break
            time.sleep(0.02)
        else:
            logger_mp.warning(
                "[G1D_Mobile_Waist_Pitch_Controller] Waist pitch did not reach home within "
                f"{timeout}s, stopping at {self.g1_waist_pitch_state_value_out.value:.4f} rad."
            )
        self.set_waist_pitch_buttons(button_bend=False, button_raise=False)
        time.sleep(0.1)  # let the control thread publish the settled command before it stops

    def close(self):
        try:
            self.ctrl_waist_pitch_go_home()
        finally:
            self.running = False
            self.waist_pitch_control_thread.join(timeout=0.5)
            logger_mp.info("[G1D_Mobile_Waist_Pitch_Controller] closed.")

    def ctrl_g1_waist_pitch(self, q_target):
        self.g1_waist_pitch_msg.q = q_target
        self.WaistPitchCmd_publisher.Write(self.g1_waist_pitch_msg)

    def control_thread(self):
        self.g1_waist_pitch_msg = unitree_hg_msg_dds__MotorCmd_()
        self.g1_waist_pitch_msg.dq = 0.0
        self.g1_waist_pitch_msg.kp = self.kp
        self.g1_waist_pitch_msg.kd = self.kd
        self.g1_waist_pitch_msg.mode = self.mode
        q_cmd = self.g1_waist_pitch_state_value_out.value
        try:
            last_time = time.time()
            while self.running:
                self.start_time = time.time()
                # Clamp dt so a scheduling hiccup cannot integrate a large jump.
                dt = min(self.start_time - last_time, 5.0 / self.fps)
                last_time = self.start_time

                direction = self.g1_waist_pitch_action_value_in.value
                measured = self.g1_waist_pitch_state_value_out.value

                if direction > 0.5:
                    velocity = self.velocity
                elif direction < -0.5:
                    velocity = -self.velocity
                else:
                    velocity = 0.0

                q_target = np.clip(q_cmd + velocity * dt, self.min_angle, self.max_angle)
                q_cmd = np.clip(
                    q_target,
                    measured - self.DELTA_WAIST_PITCH_CMD,
                    measured + self.DELTA_WAIST_PITCH_CMD,
                )

                self.g1_waist_pitch_msg.tau = self.gravity_ff.compute(
                    np.concatenate(([measured], self.arm_ctrl.get_current_dual_arm_q()))
                )
                self.ctrl_g1_waist_pitch(q_cmd)
                self.g1_waist_pitch_action_value_out.value = q_cmd

                current_time = time.time()
                time_elapsed = current_time - self.start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("G1D_Mobile_Waist_Pitch_Controller has been closed.")
