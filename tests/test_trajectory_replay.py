"""Loading and safety checks for episode replay. No hardware."""
import json
import os
import tempfile
import unittest

import numpy as np

from teleop.utils.trajectory_replay import (
    ARM_DIM, TrajectoryError, approach_profile, check_trajectory, load_trajectory,
    smoothstep, summarize,
)


def block(left_arm, right_arm, left_ee, right_ee):
    return {
        "left_arm": {"qpos": list(left_arm), "qvel": [], "torque": []},
        "right_arm": {"qpos": list(right_arm), "qvel": [], "torque": []},
        "left_ee": {"qpos": [left_ee], "qvel": [], "torque": []},
        "right_ee": {"qpos": [right_ee], "qvel": [], "torque": []},
    }


def write_episode(root, frames=10, fps=30.0, success=None, scale=0.01, drop=None,
                  gripper_scalar=False):
    """Write an episode shaped like collect.py's output."""
    data = []
    for i in range(frames):
        left = np.full(7, i * scale)
        right = np.full(7, -i * scale)
        states = block(left, right, 0.5 + i * scale, 5.0 - i * scale)
        actions = block(left + 0.001, right - 0.001, 0.5 + i * scale, 5.0 - i * scale)
        if gripper_scalar:
            actions["left_ee"]["qpos"] = 0.5 + i * scale
        if drop is not None and i == 0:
            actions.pop(drop, None)
        data.append({"idx": i, "colors": {}, "states": states, "actions": actions})

    payload = {"info": {"image": {"fps": fps}}, "text": {"goal": "t"}, "data": data}
    if success is not None:
        payload["success"] = success

    episode_dir = os.path.join(root, "episode_0000")
    os.makedirs(episode_dir, exist_ok=True)
    path = os.path.join(episode_dir, "data.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


class LoadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_shapes_and_rate(self):
        traj = load_trajectory(write_episode(self.tmp.name, frames=12, fps=30.0))
        self.assertEqual(traj.arm_q.shape, (12, ARM_DIM))
        self.assertEqual(traj.grippers.shape, (12, 2))
        self.assertEqual(traj.fps, 30.0)
        self.assertAlmostEqual(traj.duration, 0.4)
        self.assertEqual(len(traj), 12)

    def test_accepts_the_episode_directory(self):
        path = write_episode(self.tmp.name)
        by_dir = load_trajectory(os.path.dirname(path))
        by_file = load_trajectory(path)
        np.testing.assert_allclose(by_dir.arm_q, by_file.arm_q)

    def test_actions_and_states_differ(self):
        path = write_episode(self.tmp.name)
        actions = load_trajectory(path, source="actions")
        states = load_trajectory(path, source="states")
        self.assertFalse(np.allclose(actions.arm_q, states.arm_q))
        np.testing.assert_allclose(actions.arm_q[:, :7], states.arm_q[:, :7] + 0.001)

    def test_arm_halves_are_left_then_right(self):
        traj = load_trajectory(write_episode(self.tmp.name), source="states")
        self.assertTrue(np.all(traj.arm_q[1:, :7] > 0))
        self.assertTrue(np.all(traj.arm_q[1:, 7:] < 0))

    def test_scalar_gripper_value_is_accepted(self):
        """collect.py writes torso.height as a bare float; be tolerant elsewhere too."""
        traj = load_trajectory(write_episode(self.tmp.name, gripper_scalar=True))
        self.assertEqual(traj.grippers.shape[1], 2)

    def test_frame_range_and_stride(self):
        path = write_episode(self.tmp.name, frames=20, fps=30.0)
        traj = load_trajectory(path, start=5, end=15, stride=2)
        self.assertEqual(len(traj), 5)
        self.assertEqual(list(traj.frame_indices), [5, 7, 9, 11, 13])
        # Skipping frames must slow the playback rate, or the replay runs fast.
        self.assertEqual(traj.fps, 15.0)

    def test_success_label_is_carried_through(self):
        self.assertIs(load_trajectory(write_episode(self.tmp.name, success=False)).success, False)
        self.assertIs(load_trajectory(write_episode(self.tmp.name, success=True)).success, True)
        self.assertIsNone(load_trajectory(write_episode(self.tmp.name)).success)

    def test_missing_file(self):
        with self.assertRaises(TrajectoryError):
            load_trajectory(os.path.join(self.tmp.name, "nope.json"))

    def test_truncated_json_names_the_file(self):
        path = os.path.join(self.tmp.name, "data.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('{"info": {}, "data": [')
        with self.assertRaises(TrajectoryError) as ctx:
            load_trajectory(path)
        self.assertIn("valid JSON", str(ctx.exception))

    def test_missing_block_is_reported_with_the_frame(self):
        path = write_episode(self.tmp.name, drop="right_arm")
        with self.assertRaises(TrajectoryError) as ctx:
            load_trajectory(path, source="actions")
        self.assertIn("frame 0", str(ctx.exception))

    def test_empty_range_and_bad_arguments(self):
        path = write_episode(self.tmp.name, frames=5)
        with self.assertRaises(TrajectoryError):
            load_trajectory(path, start=4, end=4)
        with self.assertRaises(TrajectoryError):
            load_trajectory(path, stride=0)
        with self.assertRaises(TrajectoryError):
            load_trajectory(path, source="tactiles")


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.traj = load_trajectory(write_episode(self.tmp.name, frames=10, scale=0.01))

    def tearDown(self):
        self.tmp.cleanup()

    def test_clean_trajectory_passes(self):
        self.assertEqual(check_trajectory(self.traj), [])

    def test_joint_limit(self):
        self.traj.arm_q[3, 2] = 4.0
        problems = check_trajectory(self.traj)
        self.assertTrue(any("over the 3.5 rad limit" in p for p in problems))

    def test_gripper_range(self):
        self.traj.grippers[2, 0] = 9.0
        self.assertTrue(any("gripper" in p for p in check_trajectory(self.traj)))

    def test_jump_between_frames_is_rejected(self):
        self.traj.arm_q[5] += 1.0
        problems = check_trajectory(self.traj)
        self.assertTrue(any("jumps" in p for p in problems), problems)

    def test_non_finite(self):
        self.traj.arm_q[0, 0] = np.nan
        self.assertEqual(check_trajectory(self.traj), ["trajectory contains NaN or inf"])

    def test_single_frame_has_no_step_to_check(self):
        traj = load_trajectory(write_episode(self.tmp.name, frames=1))
        self.assertEqual(check_trajectory(traj), [])


class ApproachTests(unittest.TestCase):
    def test_ends_exactly_on_the_target(self):
        start = np.zeros(ARM_DIM)
        goal = np.linspace(0.0, 0.5, ARM_DIM)
        steps = approach_profile(start, [0.0, 0.0], goal, [5.0, 5.0], seconds=2.0, fps=30.0)
        self.assertEqual(len(steps), 60)
        np.testing.assert_allclose(steps[-1][0], goal)
        np.testing.assert_allclose(steps[-1][1], [5.0, 5.0])

    def test_eases_in_so_the_first_step_is_small(self):
        goal = np.ones(ARM_DIM)
        steps = approach_profile(np.zeros(ARM_DIM), [0.0, 0.0], goal, [0.0, 0.0], 2.0, 30.0)
        first = float(np.max(np.abs(steps[0][0])))
        middle = float(np.max(np.abs(steps[30][0] - steps[29][0])))
        self.assertLess(first, middle)

    def test_monotonic_and_rate_limited(self):
        goal = np.full(ARM_DIM, 1.5)
        steps = approach_profile(np.zeros(ARM_DIM), [0.0, 0.0], goal, [0.0, 0.0], 3.0, 30.0)
        qs = np.stack([q for q, _ in steps])
        deltas = np.diff(qs, axis=0)
        self.assertTrue(np.all(deltas >= -1e-12))
        self.assertLess(float(np.abs(deltas).max()), 0.05)

    def test_zero_duration_jumps_straight_there(self):
        goal = np.ones(ARM_DIM)
        steps = approach_profile(np.zeros(ARM_DIM), [0.0, 0.0], goal, [1.0, 1.0], 0.0, 30.0)
        self.assertEqual(len(steps), 1)
        np.testing.assert_allclose(steps[0][0], goal)

    def test_rejects_wrong_width(self):
        with self.assertRaises(TrajectoryError):
            approach_profile(np.zeros(7), [0, 0], np.zeros(ARM_DIM), [0, 0], 1.0, 30.0)

    def test_smoothstep_endpoints_and_clamping(self):
        self.assertEqual(smoothstep(0.0), 0.0)
        self.assertEqual(smoothstep(1.0), 1.0)
        self.assertEqual(smoothstep(-5.0), 0.0)
        self.assertEqual(smoothstep(5.0), 1.0)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_flags_a_failed_episode(self):
        traj = load_trajectory(write_episode(self.tmp.name, success=False))
        self.assertIn("marked FAILED", summarize(traj))

    def test_reports_the_gap_to_the_first_frame(self):
        traj = load_trajectory(write_episode(self.tmp.name))
        text = summarize(traj, current_q=traj.arm_q[0] + 0.4)
        self.assertIn("0.400 rad", text)


if __name__ == "__main__":
    unittest.main()
