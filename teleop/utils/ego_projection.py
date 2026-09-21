"""OpenCV overlay of WAIST-frame poses onto the head stereo SBS image."""
from __future__ import annotations

import cv2
import numpy as np

from teleop.utils.ego_camera_config import build_head_stereo_model

AXIS_COLORS_BGR = (
    (0, 0, 255),    # X red
    (0, 255, 0),    # Y green
    (255, 0, 0),    # Z blue
)


def _as_pose4(pose):
    pose = np.asarray(pose, dtype=float).reshape(4, 4)
    if not np.all(np.isfinite(pose)):
        raise ValueError("pose must contain finite values")
    return pose


def project_points_waist_to_uv(points_waist, T_cam_waist, K):
    """Project Nx3 WAIST points to pixel uv.  Invalid depths become nan."""
    pts = np.asarray(points_waist, dtype=float).reshape(-1, 3)
    ones = np.ones((pts.shape[0], 1), dtype=float)
    hom = np.concatenate([pts, ones], axis=1)
    cam = (T_cam_waist @ hom.T).T[:, :3]
    uv = np.full((pts.shape[0], 2), np.nan, dtype=float)
    valid = cam[:, 2] > 1e-6
    if not np.any(valid):
        return uv, valid
    x = cam[valid, 0] / cam[valid, 2]
    y = cam[valid, 1] / cam[valid, 2]
    uv[valid, 0] = K[0, 0] * x + K[0, 2]
    uv[valid, 1] = K[1, 1] * y + K[1, 2]
    return uv, valid


def draw_pose_axes(image_bgr, pose_waist, T_cam_waist, K, axis_len=0.08, thickness=2):
    """Draw RGB axes for one WAIST pose onto a single eye image (in-place)."""
    if image_bgr is None or pose_waist is None:
        return image_bgr
    pose = _as_pose4(pose_waist)
    origin = pose[:3, 3]
    tips = [origin + pose[:3, i] * axis_len for i in range(3)]
    points = np.vstack([origin, tips])
    uv, valid = project_points_waist_to_uv(points, T_cam_waist, K)
    h, w = image_bgr.shape[:2]
    if not valid[0]:
        return image_bgr
    o = uv[0]
    if not (0 <= o[0] < w and 0 <= o[1] < h):
        # Still allow axes that start slightly off-frame if tips are visible.
        pass
    o_pt = (int(round(o[0])), int(round(o[1])))
    for axis in range(3):
        if not valid[axis + 1]:
            continue
        tip = uv[axis + 1]
        if not np.all(np.isfinite(tip)):
            continue
        tip_pt = (int(round(tip[0])), int(round(tip[1])))
        cv2.line(image_bgr, o_pt, tip_pt, AXIS_COLORS_BGR[axis], thickness, cv2.LINE_AA)
    if 0 <= o_pt[0] < w and 0 <= o_pt[1] < h:
        cv2.circle(image_bgr, o_pt, 3, (255, 255, 255), -1, cv2.LINE_AA)
    return image_bgr


class EgoPixelOverlay:
    """Stereo SBS overlay helper for the teleop head image."""

    def __init__(self, half_width=None, half_height=None):
        kwargs = {}
        if half_width is not None:
            kwargs["half_width"] = int(half_width)
        if half_height is not None:
            kwargs["half_height"] = int(half_height)
        self.model = build_head_stereo_model(**kwargs)
        self.K = self.model["K"]
        self.T_left = self.model["T_cam_waist_left"]
        self.T_right = self.model["T_cam_waist_right"]
        self.half_width = self.model["half_width"]
        self.half_height = self.model["half_height"]
        self.warning = self.model["warning"]

    def overlay(self, head_bgr, left_poses=None, right_poses=None, axis_len=0.08):
        """Return a copy of head_bgr with poses drawn on left/right halves.

        ``left_poses`` / ``right_poses`` are sequences of WAIST 4x4 poses to
        draw on the left-eye / right-eye halves respectively.  Using the same
        dual-arm poses for both eyes is the normal stereo case.
        """
        if head_bgr is None:
            return None
        image = np.asarray(head_bgr)
        if image.ndim != 3 or image.shape[0] < self.half_height:
            return image.copy()
        out = image.copy()
        width = out.shape[1]
        mid = width // 2
        left_img = out[:, :mid]
        right_img = out[:, mid:]
        for pose in left_poses or ():
            if pose is None:
                continue
            draw_pose_axes(left_img, pose, self.T_left, self.K, axis_len=axis_len)
        for pose in right_poses or ():
            if pose is None:
                continue
            draw_pose_axes(right_img, pose, self.T_right, self.K, axis_len=axis_len)
        return out

    def overlay_dual_arm(self, head_bgr, left_arm_pose, right_arm_pose, extra_left=None, extra_right=None):
        """Draw left/right arm poses onto both eyes (stereo consistent)."""
        left_list = [left_arm_pose, right_arm_pose]
        right_list = [left_arm_pose, right_arm_pose]
        if extra_left is not None:
            left_list.extend(extra_left if isinstance(extra_left, (list, tuple)) else [extra_left])
            right_list.extend(extra_left if isinstance(extra_left, (list, tuple)) else [extra_left])
        if extra_right is not None:
            left_list.extend(extra_right if isinstance(extra_right, (list, tuple)) else [extra_right])
            right_list.extend(extra_right if isinstance(extra_right, (list, tuple)) else [extra_right])
        return self.overlay(head_bgr, left_poses=left_list, right_poses=right_list)
