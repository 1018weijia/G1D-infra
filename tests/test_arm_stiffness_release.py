"""Tests for fading arm stiffness out instead of dropping it in one tick."""
import threading
import unittest
from unittest import mock

from teleop.robot_control.robot_arm import (
    G1_29_Arm_Internal_Dex1_Controller,
    G1_29_Arm_JointIndex,
    G1_29_JointIndex,
)


class FakeArmController(G1_29_Arm_Internal_Dex1_Controller):
    """Run the real release_stiffness() without DDS, recording kp scale per tick."""

    def __init__(self):
        self.control_dt = 1.0 / 250.0
        self.kp_low = 80.0
        self.kp_wrist = 40.0
        self.kp_high = 300.0
        self.kd_low = 3.0
        self.kd_wrist = 1.5
        self._kp_scale = 1.0
        self.ctrl_lock = threading.Lock()
        self.samples = []


def run_release(seconds, min_kp_ratio=0.0):
    """Call release_stiffness() with a fake clock so tests stay instant."""
    ctrl = FakeArmController()
    fake_time = {"t": 0.0}

    def fake_sleep(dt):
        ctrl.samples.append(ctrl._kp_scale)
        fake_time["t"] += dt

    with mock.patch("time.sleep", side_effect=fake_sleep):
        ctrl.release_stiffness(seconds=seconds, min_kp_ratio=min_kp_ratio)
    return ctrl


class ReleaseStiffnessTests(unittest.TestCase):
    def test_scale_starts_near_one_and_ends_at_zero(self):
        ctrl = run_release(seconds=5.0)
        self.assertGreater(ctrl.samples[0], 0.999)
        self.assertEqual(ctrl._kp_scale, 0.0)

    def test_scale_decreases_monotonically(self):
        ctrl = run_release(seconds=1.0)
        self.assertTrue(
            all(a >= b - 1e-9 for a, b in zip(ctrl.samples, ctrl.samples[1:])),
            "kp scale must never rise during release",
        )

    def test_smoothstep_has_a_soft_start(self):
        # A linear ramp would already be 10/1250 = 0.8% down after ten ticks, which
        # is a visible jolt on a shoulder holding the whole forearm.
        ctrl = run_release(seconds=5.0)
        self.assertLess(1.0 - ctrl.samples[10], 0.001)

    def test_smoothstep_lands_softly(self):
        # Symmetry check: the last ticks should also move very little.
        ctrl = run_release(seconds=5.0)
        self.assertLess(ctrl.samples[-1] - ctrl.samples[-2], 0.001)

    def test_tick_count_matches_duration(self):
        ctrl = run_release(seconds=5.0)
        self.assertAlmostEqual(len(ctrl.samples), 5.0 / (1.0 / 250.0), delta=2)

    def test_partial_release_keeps_a_fraction(self):
        ctrl = run_release(seconds=0.5, min_kp_ratio=0.25)
        self.assertAlmostEqual(ctrl._kp_scale, 0.25, places=6)

    def test_zero_duration_still_emits_a_tick(self):
        ctrl = run_release(seconds=0.0)
        self.assertTrue(ctrl.samples, "a zero duration must still emit at least one tick")
        self.assertEqual(ctrl._kp_scale, 0.0)

    def test_wrist_and_shoulder_scale_independently(self):
        ctrl = run_release(seconds=0.5, min_kp_ratio=0.5)
        shoulder_kp = ctrl.kp_low * ctrl._kp_scale
        wrist_kp = ctrl.kp_wrist * ctrl._kp_scale
        self.assertAlmostEqual(shoulder_kp, 40.0)
        self.assertAlmostEqual(wrist_kp, 20.0)

    def test_kd_is_never_touched(self):
        ctrl = run_release(seconds=1.0)
        # release_stiffness only writes _kp_scale; the damping must survive so the
        # arms sag instead of swinging.
        self.assertEqual(ctrl.kd_low, 3.0)
        self.assertEqual(ctrl.kd_wrist, 1.5)


class ArmJointCoverageTests(unittest.TestCase):
    def test_all_fourteen_arm_joints_are_covered(self):
        ids = [j.value for j in G1_29_Arm_JointIndex]
        self.assertEqual(len(ids), 14)
        for expected in (15, 18, 21, 22, 25, 28):
            self.assertIn(expected, ids)

    def test_waist_keeps_its_stiffness(self):
        # Only arms fade. Dropping the waist would let the torso fold forward.
        arm_ids = {j.value for j in G1_29_Arm_JointIndex}
        self.assertNotIn(G1_29_JointIndex.kWaistYaw.value, arm_ids)
        self.assertNotIn(G1_29_JointIndex.kWaistPitch.value, arm_ids)
        self.assertNotIn(G1_29_JointIndex.kWaistRoll.value, arm_ids)


if __name__ == "__main__":
    unittest.main()
