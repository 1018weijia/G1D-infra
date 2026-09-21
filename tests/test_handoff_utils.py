"""Hardware-free tests for the rollback-endpoint to XR target conversion."""
import unittest

import numpy as np

from teleop.utils.handoff_utils import rollback_endpoint_to_xr_targets


class _FakeIK:
    """Returns distinct left/right waist-frame EEF poses so sides stay traceable."""

    def __init__(self):
        self.calls = []

    def solve_fk_matrix(self, q):
        self.calls.append(np.asarray(q, dtype=float).copy())
        left = np.eye(4)
        left[:3, 3] = [0.1, 0.2, 0.3]
        right = np.eye(4)
        right[:3, 3] = [-0.1, 0.2, 0.3]
        return left, right


class _FakeTVWrapper:
    """Stand-in for TeleVuerWrapper's waist/OpenXR frame conversions."""

    def __init__(self, broken_side=None):
        self.broken_side = broken_side
        self.offset_calls = []

    def waist_pose_to_openxr_controller(self, waist_pose, head_pose, side):
        target = np.asarray(waist_pose, dtype=float).copy()
        # Mark the OpenXR conversion so tests can tell it ran.
        target[0, 3] += 1.0
        if side == self.broken_side:
            target[3] = [1.0, 0.0, 0.0, 1.0]
        return target

    def apply_openxr_forward_offset(self, target, head_pose, forward_offset):
        self.offset_calls.append(forward_offset)
        shifted = np.asarray(target, dtype=float).copy()
        shifted[2, 3] += forward_offset
        return shifted


class RollbackEndpointToXRTargetsTests(unittest.TestCase):
    def setUp(self):
        self.arm_ik = _FakeIK()
        self.tv = _FakeTVWrapper()
        self.head_pose = np.eye(4)
        self.endpoint = np.linspace(-0.3, 0.3, 14)

    def test_returns_left_and_right_homogeneous_targets(self):
        targets = rollback_endpoint_to_xr_targets(
            self.arm_ik, self.tv, self.endpoint, self.head_pose,
        )
        self.assertEqual(set(targets), {"left", "right"})
        for target in targets.values():
            self.assertEqual(target.shape, (4, 4))
            self.assertTrue(np.allclose(target[3], [0.0, 0.0, 0.0, 1.0]))
        # FK ran on the full 14-joint endpoint, and the OpenXR conversion was applied.
        self.assertTrue(np.allclose(self.arm_ik.calls[0], self.endpoint))
        self.assertAlmostEqual(targets["left"][0, 3], 1.1)
        self.assertAlmostEqual(targets["right"][0, 3], 0.9)

    def test_rejects_wrong_width_or_non_finite_endpoint(self):
        for bad in (np.zeros(13), np.zeros(15), np.full(14, np.nan),
                    np.concatenate([np.zeros(13), [np.inf]])):
            with self.assertRaises(ValueError):
                rollback_endpoint_to_xr_targets(
                    self.arm_ik, self.tv, bad, self.head_pose,
                )

    def test_forward_offset_applied_only_when_non_zero(self):
        zero = rollback_endpoint_to_xr_targets(
            self.arm_ik, self.tv, self.endpoint, self.head_pose, forward_offset=0.0,
        )
        self.assertEqual(self.tv.offset_calls, [])

        shifted = rollback_endpoint_to_xr_targets(
            self.arm_ik, self.tv, self.endpoint, self.head_pose, forward_offset=0.25,
        )
        self.assertEqual(self.tv.offset_calls, [0.25, 0.25])
        for side in ("left", "right"):
            self.assertAlmostEqual(
                shifted[side][2, 3] - zero[side][2, 3], 0.25,
            )

    def test_rejects_non_homogeneous_target(self):
        broken = _FakeTVWrapper(broken_side="right")
        with self.assertRaises(ValueError):
            rollback_endpoint_to_xr_targets(
                self.arm_ik, broken, self.endpoint, self.head_pose,
            )


if __name__ == "__main__":
    unittest.main()
