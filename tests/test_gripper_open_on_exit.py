"""Tests for parking both grippers open before control is released."""
import threading
import unittest
from unittest import mock

import numpy as np

from teleop.robot_control.robot_arm import G1_29_Arm_Internal_Dex1_Controller
from teleop.utils.ready_pose import GRIPPER_CLOSED, GRIPPER_OPEN


class FakeGripperController(G1_29_Arm_Internal_Dex1_Controller):
    """Run the real open_grippers() without DDS, recording the commanded target."""

    def __init__(self, measured=(GRIPPER_CLOSED, 3.178)):
        self.control_dt = 1.0 / 250.0
        self.ctrl_lock = threading.Lock()
        self.gripper_q_target = np.array([0.0, 0.0])
        self._policy_gripper_direct = False
        self._measured = np.array(measured, dtype=float)
        self.samples = []
        self.latch_calls = 0

    def get_current_dual_gripper_q(self):
        return self._measured.copy()

    def latch_gripper_command(self):
        self.latch_calls += 1
        self._policy_gripper_direct = True
        return float(self.gripper_q_target[0]), float(self.gripper_q_target[1])


def run_open(measured=(GRIPPER_CLOSED, 3.178), seconds=1.5, open_q=None):
    ctrl = FakeGripperController(measured=measured)

    def fake_sleep(dt):
        ctrl.samples.append(tuple(ctrl.gripper_q_target))

    kwargs = {} if open_q is None else {"open_q": open_q}
    with mock.patch("time.sleep", side_effect=fake_sleep):
        ctrl.open_grippers(seconds=seconds, **kwargs)
    return ctrl


class OpenGrippersTests(unittest.TestCase):
    def test_both_grippers_end_fully_open(self):
        ctrl = run_open()
        np.testing.assert_allclose(ctrl.gripper_q_target, [GRIPPER_OPEN, GRIPPER_OPEN])

    def test_mismatched_start_positions_both_converge(self):
        # This is the reported symptom: one jaw shut, the other half open.
        ctrl = run_open(measured=(GRIPPER_CLOSED, 3.178))
        np.testing.assert_allclose(ctrl.samples[-1], [GRIPPER_OPEN, GRIPPER_OPEN])
        self.assertAlmostEqual(ctrl.samples[0][0], GRIPPER_CLOSED, places=3)
        self.assertAlmostEqual(ctrl.samples[0][1], 3.178, places=3)

    def test_ramp_is_monotonic_for_both_jaws(self):
        ctrl = run_open(measured=(0.0, 0.0))
        left = [s[0] for s in ctrl.samples]
        right = [s[1] for s in ctrl.samples]
        self.assertTrue(all(a <= b + 1e-9 for a, b in zip(left, left[1:])))
        self.assertTrue(all(a <= b + 1e-9 for a, b in zip(right, right[1:])))

    def test_starts_from_measured_not_stale_command(self):
        # A policy rollout can leave a stale open command while the jaw is shut.
        ctrl = FakeGripperController(measured=(0.0, 0.0))
        ctrl.gripper_q_target = np.array([GRIPPER_OPEN, GRIPPER_OPEN])
        with mock.patch("time.sleep", side_effect=lambda dt: ctrl.samples.append(
                tuple(ctrl.gripper_q_target))):
            ctrl.open_grippers(seconds=1.0)
        # Must ease up from the measured 0.0, not jump to the stale 5.40 target.
        self.assertLess(ctrl.samples[0][0], GRIPPER_OPEN * 0.5)

    def test_latches_so_xr_cannot_overwrite(self):
        ctrl = run_open()
        self.assertGreaterEqual(ctrl.latch_calls, 1)
        self.assertTrue(ctrl._policy_gripper_direct)

    def test_smoothstep_has_soft_start(self):
        ctrl = run_open(measured=(0.0, 0.0), seconds=1.5)
        # Ten ticks in, a smoothstep ramp has barely moved; linear would be 2.7%.
        self.assertLess(ctrl.samples[10][0], GRIPPER_OPEN * 0.005)

    def test_already_open_stays_open(self):
        ctrl = run_open(measured=(GRIPPER_OPEN, GRIPPER_OPEN))
        np.testing.assert_allclose(ctrl.gripper_q_target, [GRIPPER_OPEN, GRIPPER_OPEN])
        self.assertTrue(all(abs(s[0] - GRIPPER_OPEN) < 1e-6 for s in ctrl.samples))

    def test_custom_open_value(self):
        ctrl = run_open(open_q=2.0)
        np.testing.assert_allclose(ctrl.gripper_q_target, [2.0, 2.0])

    def test_zero_duration_still_opens(self):
        ctrl = run_open(seconds=0.0)
        np.testing.assert_allclose(ctrl.gripper_q_target, [GRIPPER_OPEN, GRIPPER_OPEN])


class ExitOrderingTests(unittest.TestCase):
    """The exit sequence must open jaws before the spring is withdrawn."""

    def test_open_then_release_then_stop(self):
        import inspect
        from pathlib import Path
        for name in ("collect.py", "policy_deploy.py"):
            src = Path(__file__).resolve().parents[1] / name
            text = src.read_text(encoding="utf-8")
            i_open = text.index("open_grippers(")
            i_release = text.index("release_stiffness(")
            self.assertLess(i_open, i_release, f"{name}: grippers must open before stiffness release")
            # The release call must not already be inside the "stop publishing" path.
            self.assertIn("ctrl_dual_arm_go_home", text[:i_open], f"{name}: home should happen first")


if __name__ == "__main__":
    unittest.main()
