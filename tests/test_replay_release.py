"""Tests for replay's release sequence (open jaws, fade stiffness, stop)."""
import os
import unittest

REPLAY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "teleop", "replay.py")


class ReplayReleaseOrderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(REPLAY, encoding="utf-8") as handle:
            cls.text = handle.read()

    def test_exit_opens_grippers(self):
        self.assertIn("open_grippers(", self.text)

    def test_exit_releases_stiffness(self):
        self.assertIn("release_stiffness(", self.text)

    def test_jaws_open_before_stiffness_fades(self):
        i_open = self.text.index("open_grippers(seconds=args.gripper_open_seconds)")
        i_release = self.text.index("release_stiffness(seconds=args.release_seconds)")
        self.assertLess(i_open, i_release)

    def test_go_home_happens_before_release(self):
        i_home = self.text.index("ctrl_dual_arm_go_home()")
        i_open = self.text.index("open_grippers(seconds=args.gripper_open_seconds)")
        self.assertLess(i_home, i_open)

    def test_release_lives_in_the_finally_block(self):
        # If this drifted outside finally, an abort would skip it and drop the arms.
        i_finally = self.text.index("    finally:")
        i_release = self.text.index("release_stiffness(seconds=args.release_seconds)")
        self.assertLess(i_finally, i_release)

    def test_both_flags_exist_with_defaults(self):
        self.assertIn("--release-seconds", self.text)
        self.assertIn("--gripper-open-seconds", self.text)
        self.assertIn("if args.release_seconds > 0:", self.text)
        self.assertIn("if args.gripper_open_seconds > 0:", self.text)

    def test_hold_before_home_is_preserved(self):
        # The 1s "hold the last command" loop predates this change and must survive:
        # it is what keeps the arms from sagging while go_home spins up.
        self.assertIn("Hold the last command so the arms do not sag", self.text)


class ControllerMethodTests(unittest.TestCase):
    """The two methods replay now calls must exist on the real controller class."""

    def test_controller_exposes_both_release_methods(self):
        from teleop.robot_control.robot_arm import G1_29_Arm_Internal_Dex1_Controller
        self.assertTrue(callable(getattr(G1_29_Arm_Internal_Dex1_Controller, "open_grippers", None)))
        self.assertTrue(callable(getattr(G1_29_Arm_Internal_Dex1_Controller, "release_stiffness", None)))


if __name__ == "__main__":
    unittest.main()
