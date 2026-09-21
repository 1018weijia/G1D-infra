"""Approximate G1-D head binocular camera model for ego pixel overlay.

Parameters come from Unitree service docs (FOV + optical-center xyz + pitch).
They are NOT factory calibrated intrinsics/extrinsics.  Lift-height changes and
the fixed head<->waist offset make this a visual aid only.
"""
from __future__ import annotations

import numpy as np

# Actual teleop head SBS half-frame (cam_config image_shape [480, 1280]).
HALF_WIDTH = 640
HALF_HEIGHT = 480

# Documented head binocular FOV (degrees).
HEAD_FOV_H_DEG = 115.0
HEAD_FOV_V_DEG = 80.0

# Documented pitch of each eye optical axis, nose-down (degrees).
HEAD_PITCH_DOWN_DEG = 47.6

# Eye centers relative to head center in robot frame (m).  Lateral only;
# absolute base xyz from the doc is not used directly so lift height is less
# tightly coupled.  Left/right signs follow robot convention: +Y = left.
LEFT_EYE_IN_HEAD_M = np.array([0.0, 0.0298, 0.0], dtype=float)
RIGHT_EYE_IN_HEAD_M = np.array([0.0, -0.0298, 0.0], dtype=float)

# Same fixed HEAD->WAIST origin shift used by televuer/tv_wrapper.py.
WAIST_FROM_HEAD_XYZ_M = np.array([0.15, 0.0, 0.45], dtype=float)


def intrinsic_from_fov(width, height, fov_h_deg, fov_v_deg):
    """Estimate a pinhole K from horizontal/vertical FOV."""
    fx = float(width) / (2.0 * np.tan(np.deg2rad(fov_h_deg) * 0.5))
    fy = float(height) / (2.0 * np.tan(np.deg2rad(fov_v_deg) * 0.5))
    return np.array(
        [[fx, 0.0, width * 0.5],
         [0.0, fy, height * 0.5],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )


def _rot_y(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]], dtype=float)


def _robot_to_opencv_cam_rotation(pitch_down_rad):
    """Rotation that maps robot-frame vectors into OpenCV camera axes.

    Robot:  +X forward, +Y left, +Z up
    OpenCV: +X right,   +Y down, +Z forward (optical axis)

    Unpitched mapping:
        cam_x = -robot_y
        cam_y = -robot_z
        cam_z =  robot_x
    Then apply nose-down pitch about robot +Y before the axis remap.
    """
    r_pitch = _rot_y(-pitch_down_rad)  # nose-down
    r_axes = np.array([[0.0, -1.0, 0.0],
                       [0.0, 0.0, -1.0],
                       [1.0, 0.0, 0.0]], dtype=float)
    return r_axes @ r_pitch


def _camera_pose_in_waist(eye_in_head_m, pitch_down_deg):
    """4x4 pose of the OpenCV camera frame expressed in the WAIST frame."""
    pitch = np.deg2rad(pitch_down_deg)
    r_cam_from_waist = _robot_to_opencv_cam_rotation(pitch)
    # Camera position in waist ≈ head origin in waist + eye offset in head.
    t_waist_cam = WAIST_FROM_HEAD_XYZ_M + eye_in_head_m
    t_waist_from_head = np.eye(4, dtype=float)
    t_waist_from_head[:3, 3] = WAIST_FROM_HEAD_XYZ_M
    # Orientation of camera frame relative to waist (same as relative to head
    # while ignoring dynamic head pitch/roll, matching head_yaw teleop).
    t_waist_cam_pose = np.eye(4, dtype=float)
    # Columns of R are camera axes expressed in waist.  r_cam_from_waist maps
    # waist vectors into camera, so R_waist_cam = r_cam_from_waist.T.
    t_waist_cam_pose[:3, :3] = r_cam_from_waist.T
    t_waist_cam_pose[:3, 3] = t_waist_cam
    return t_waist_cam_pose, r_cam_from_waist


def build_head_stereo_model(half_width=HALF_WIDTH, half_height=HALF_HEIGHT):
    """Return left/right K and T_cam_waist (OpenCV camera <- WAIST)."""
    k = intrinsic_from_fov(half_width, half_height, HEAD_FOV_H_DEG, HEAD_FOV_V_DEG)
    left_pose, left_r = _camera_pose_in_waist(LEFT_EYE_IN_HEAD_M, HEAD_PITCH_DOWN_DEG)
    right_pose, right_r = _camera_pose_in_waist(RIGHT_EYE_IN_HEAD_M, HEAD_PITCH_DOWN_DEG)

    def pose_to_cam_from_waist(pose_waist_cam, r_cam_from_waist):
        t = np.eye(4, dtype=float)
        t[:3, :3] = r_cam_from_waist
        t[:3, 3] = -r_cam_from_waist @ pose_waist_cam[:3, 3]
        return t

    return {
        "K": k,
        "D": np.zeros(5, dtype=float),
        "half_width": int(half_width),
        "half_height": int(half_height),
        "T_cam_waist_left": pose_to_cam_from_waist(left_pose, left_r),
        "T_cam_waist_right": pose_to_cam_from_waist(right_pose, right_r),
        "warning": (
            "ego pixel overlay uses approximate FOV intrinsics and fixed "
            "head/waist eye extrinsics; not factory calibration"
        ),
    }
