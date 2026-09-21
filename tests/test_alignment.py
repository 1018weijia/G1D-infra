import time
import unittest

import numpy as np

from teleop.utils.alignment import ACTIVE, ALIGNED, ALIGNING, DualArmAlignment


def pose(x, y, z):
    mat = np.eye(4)
    mat[:3, 3] = (x, y, z)
    return mat


class DualArmAlignmentTests(unittest.TestCase):
    def test_reaches_aligned_then_confirm(self):
        targets = {"left": pose(-0.3, 1.2, -0.2), "right": pose(0.3, 1.2, -0.2)}
        alignment = DualArmAlignment(targets, position_tolerance=0.04,
                                     rotation_tolerance=0.20, stable_seconds=0.05)
        self.assertEqual(alignment.state, ALIGNING)
        alignment.update(targets["left"], targets["right"], True)
        self.assertEqual(alignment.state, ALIGNING)
        time.sleep(0.06)
        alignment.update(targets["left"], targets["right"], True)
        self.assertEqual(alignment.state, ALIGNED)
        robot_left = pose(-0.2, 0.1, 0.0)
        robot_right = pose(0.2, 0.1, 0.0)
        self.assertTrue(alignment.confirm(targets["left"], targets["right"], robot_left, robot_right))
        self.assertEqual(alignment.state, ACTIVE)
        moved_left = pose(-0.25, 1.2, -0.2)
        out = alignment.targets_from_hand(moved_left, targets["right"], gain=1.0)
        self.assertIsNotNone(out)
        self.assertAlmostEqual(out[0][0, 3], robot_left[0, 3] + 0.05, places=5)

    def test_invalid_tracking_does_not_align(self):
        targets = {"left": pose(-0.3, 1.2, -0.2), "right": pose(0.3, 1.2, -0.2)}
        alignment = DualArmAlignment(targets, stable_seconds=0.0)
        alignment.update(targets["left"], targets["right"], False)
        self.assertEqual(alignment.state, ALIGNING)


if __name__ == "__main__":
    unittest.main()
