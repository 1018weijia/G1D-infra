"""Hardware-free tests for the shared arm startup pose."""
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from teleop.utils.ready_pose import (
    ARM_DIM, ReadyPoseError, load_ready_pose, move_to_ready_pose,
    ready_pose_profile, smoothstep,
)


class _FakeArmController:
    def __init__(self, q):
        self.q = np.asarray(q, dtype=float)
        self.commands = []
        self.gripper_commands = []

    def get_current_dual_arm_q(self):
        return self.q.copy()

    def ctrl_dual_arm(self, q, tau):
        self.commands.append((np.asarray(q).copy(), np.asarray(tau).copy()))

    def set_policy_gripper_q(self, left, right):
        self.gripper_commands.append((left, right))


class _FakeIK:
    def __init__(self):
        self.reset_q = None

    def reset_solution_state(self, q):
        self.reset_q = np.asarray(q).copy()

    def solve_tau(self, q):
        return np.asarray(q) * 0.0


class ReadyPoseTests(unittest.TestCase):
    def test_repo_config_is_valid_and_raised(self):
        path = os.path.join(os.path.dirname(__file__), "..", "configs", "ready_pose.json")
        q = load_ready_pose(path)
        self.assertEqual(q.shape, (ARM_DIM,))
        self.assertGreater(float(np.max(np.abs(q))), 0.25)

    def test_profile_ends_exactly_at_target(self):
        current = np.zeros(ARM_DIM)
        target = np.linspace(-0.3, 0.3, ARM_DIM)
        profile = ready_pose_profile(current, target, seconds=3.0, frequency=30.0)
        self.assertEqual(profile.shape, (90, ARM_DIM))
        np.testing.assert_allclose(profile[-1], target)

    def test_profile_eases_in_and_is_monotonic(self):
        profile = ready_pose_profile(
            np.zeros(ARM_DIM), np.ones(ARM_DIM), seconds=2.0, frequency=30.0,
        )
        deltas = np.diff(np.vstack([np.zeros(ARM_DIM), profile]), axis=0)
        self.assertTrue(np.all(deltas >= -1e-12))
        self.assertLess(float(deltas[0].max()), float(deltas[29].max()))

    def test_rejects_unsafe_config(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "ready.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"arm_q": [0.0] * 13 + [4.0]}, handle)
            with self.assertRaises(ReadyPoseError):
                load_ready_pose(path)

    def test_rejects_non_positive_duration(self):
        with self.assertRaises(ReadyPoseError):
            ready_pose_profile(np.zeros(ARM_DIM), np.zeros(ARM_DIM), 0.0, 30.0)

    def test_smoothstep_is_clamped(self):
        self.assertEqual(smoothstep(-1.0), 0.0)
        self.assertEqual(smoothstep(0.0), 0.0)
        self.assertEqual(smoothstep(1.0), 1.0)
        self.assertEqual(smoothstep(2.0), 1.0)

    @mock.patch("teleop.utils.ready_pose.time.sleep", return_value=None)
    def test_move_commands_target_and_preserves_grippers(self, _sleep):
        arm = _FakeArmController(np.zeros(ARM_DIM))
        ik = _FakeIK()
        target = np.linspace(-0.2, 0.2, ARM_DIM)
        result = move_to_ready_pose(
            arm, ik, target, seconds=0.1, frequency=10.0,
            grippers=(1.5, 2.5), settle_seconds=0.0,
        )
        self.assertTrue(result.completed)
        np.testing.assert_allclose(arm.commands[-1][0], target)
        np.testing.assert_allclose(ik.reset_q, target)
        self.assertEqual(arm.gripper_commands[-1], (1.5, 2.5))

    @mock.patch("teleop.utils.ready_pose.time.sleep", return_value=None)
    def test_move_can_abort_before_motion(self, _sleep):
        arm = _FakeArmController(np.zeros(ARM_DIM))
        result = move_to_ready_pose(
            arm, _FakeIK(), np.ones(ARM_DIM), seconds=1.0, frequency=30.0,
            settle_seconds=0.0, stop_requested=lambda: True,
        )
        self.assertFalse(result.completed)
        self.assertEqual(arm.commands, [])


if __name__ == "__main__":
    unittest.main()
