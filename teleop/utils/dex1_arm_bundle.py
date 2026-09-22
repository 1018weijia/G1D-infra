"""Shared Dex1 dual-arm controller bootstrap for open-loop and policy paths."""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Array, Lock, Value

from teleop.robot_control.robot_arm import G1_29_Arm_Internal_Dex1_Controller


@dataclass
class Dex1ArmBundle:
    arm_ctrl: G1_29_Arm_Internal_Dex1_Controller
    xr_motion_data_ready: Value
    left_gripper_value: Value
    right_gripper_value: Value
    dual_gripper_data_lock: Lock
    dual_gripper_state_array: Array
    dual_gripper_action_array: Array


def create_dex1_arm_controller(
        xr_motion_data_ready=None, simulation_mode=False, use_waist=False):
    """Build the Dex1 arm controller and the shared memory it reads.

    Pass an existing xr_motion_data_ready Value when a caller will flip it (e.g.
    policy_deploy with XR). Leave it None to create a Value fixed at False —
    required for open-loop replay so the controller never takes the XR gripper
    branch with unset triggers.
    """
    if xr_motion_data_ready is None:
        xr_motion_data_ready = Value("b", False, lock=True)
    left_gripper_value = Value("d", 0.0, lock=True)
    right_gripper_value = Value("d", 0.0, lock=True)
    dual_gripper_data_lock = Lock()
    dual_gripper_state_array = Array("d", 2, lock=False)
    dual_gripper_action_array = Array("d", 2, lock=False)
    arm_ctrl = G1_29_Arm_Internal_Dex1_Controller(
        left_gripper_value,
        right_gripper_value,
        dual_gripper_data_lock,
        dual_gripper_state_array,
        dual_gripper_action_array,
        simulation_mode=simulation_mode,
        use_waist=use_waist,
        xr_motion_data_ready_in=xr_motion_data_ready,
    )
    return Dex1ArmBundle(
        arm_ctrl=arm_ctrl,
        xr_motion_data_ready=xr_motion_data_ready,
        left_gripper_value=left_gripper_value,
        right_gripper_value=right_gripper_value,
        dual_gripper_data_lock=dual_gripper_data_lock,
        dual_gripper_state_array=dual_gripper_state_array,
        dual_gripper_action_array=dual_gripper_action_array,
    )
