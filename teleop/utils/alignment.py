"""Small, frame-explicit dual-arm alignment state machine."""
import time
import numpy as np
from scipy.spatial.transform import Rotation

ALIGNING, ALIGNED, ACTIVE = "ALIGNING", "ALIGNED", "ACTIVE"

def pose_error(actual, target):
    position = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
    rotation = float(np.linalg.norm(Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec()))
    return position, rotation

class DualArmAlignment:
    def __init__(self, targets, position_tolerance=0.04, rotation_tolerance=0.20, stable_seconds=0.5):
        self.targets = targets
        self.position_tolerance = position_tolerance
        self.rotation_tolerance = rotation_tolerance
        self.stable_seconds = stable_seconds
        self.state = ALIGNING
        self._stable_since = None
        self.last_errors = None
        self.hand_start = None
        self.robot_start = None

    def reset(self, targets):
        """Replace XR-world targets and return to the pre-confirmation state."""
        self.targets = {key: np.asarray(value, dtype=float).reshape(4, 4).copy()
                        for key, value in targets.items()}
        self.state = ALIGNING
        self._stable_since = None
        self.hand_start = None
        self.robot_start = None
        self.last_errors = None

    def update(self, left, right, tracking_valid):
        if self.state != ALIGNING:
            return self.state
        valid = tracking_valid and left is not None and right is not None
        if valid:
            errors = (pose_error(left, self.targets["left"]), pose_error(right, self.targets["right"]))
            self.last_errors = errors
            valid = all(p <= self.position_tolerance and r <= self.rotation_tolerance for p, r in errors)
        if valid:
            self._stable_since = self._stable_since or time.monotonic()
            if time.monotonic() - self._stable_since >= self.stable_seconds:
                self.state = ALIGNED
        else:
            self._stable_since = None
        return self.state

    def confirm(self, left, right, robot_left, robot_right):
        if self.state != ALIGNED:
            return False
        self.hand_start = {"left": left.copy(), "right": right.copy()}
        self.robot_start = {"left": robot_left.copy(), "right": robot_right.copy()}
        self.state = ACTIVE
        return True

    def reanchor(self, left, right, robot_left, robot_right):
        """Restart relative control from the current hand and robot poses."""
        if self.state != ACTIVE:
            return False
        self.hand_start = {"left": left.copy(), "right": right.copy()}
        self.robot_start = {"left": robot_left.copy(), "right": robot_right.copy()}
        return True

    def targets_from_hand(self, left, right, gain=1.0):
        if self.state != ACTIVE:
            return None
        gain = float(gain)
        if not np.isfinite(gain) or gain < 0.0 or gain > 1.0:
            raise ValueError("handoff gain must be finite and within [0, 1]")

        def scaled_target(side, current):
            delta = np.linalg.inv(self.hand_start[side]) @ current
            scaled_delta = np.eye(4)
            scaled_delta[:3, 3] = delta[:3, 3] * gain
            rotvec = Rotation.from_matrix(delta[:3, :3]).as_rotvec()
            scaled_delta[:3, :3] = Rotation.from_rotvec(rotvec * gain).as_matrix()
            return self.robot_start[side] @ scaled_delta

        return scaled_target("left", left), scaled_target("right", right)
