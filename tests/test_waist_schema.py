"""Waist columns stay consistent across a task, and pitch targets stay capped."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from data_convert.constants import MOVEIBLE_LIFT_G1_DEX1_USEWAIST_CONFIG
from data_convert.convert_unitree_json import (
    NO_WAIST_ROBOT_TYPE, WAIST_ROBOT_TYPE, check_waist_consistency, episode_waist_fields,
    robot_type_for_schema,
)
from teleop.robot_control.waist_pitch import tracked_pitch_command


def _episode(root, name, with_waist, with_pitch=None):
    if with_pitch is None:
        with_pitch = with_waist
    episode = os.path.join(root, name)
    os.makedirs(episode, exist_ok=True)
    actions = {"left_arm": {"qpos": [0.0] * 7}}
    states = {"left_arm": {"qpos": [0.0] * 7}}
    if with_waist:
        actions["waist"] = {"qpos": [0.2]}
        states["waist"] = {"qpos": [0.2]}
    if with_pitch:
        actions["waist_pitch"] = {"qpos": [0.4]}
        states["waist_pitch"] = {"qpos": [0.4]}
    payload = {"info": {"image": {"fps": 30}}, "data": [{"states": states, "actions": actions}]}
    with open(os.path.join(episode, "data.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return episode


class ConvertWaistSchemaTests(unittest.TestCase):
    def test_use_waist_config_lists_bend_after_yaw(self):
        for names in (
            MOVEIBLE_LIFT_G1_DEX1_USEWAIST_CONFIG.json_state_data_name,
            MOVEIBLE_LIFT_G1_DEX1_USEWAIST_CONFIG.json_action_data_name,
        ):
            yaw = names.index("waist.qpos")
            self.assertEqual(names[yaw + 1], "waist_pitch.qpos")
        motors = MOVEIBLE_LIFT_G1_DEX1_USEWAIST_CONFIG.motors
        self.assertLess(motors.index("kWaistYaw"), motors.index("kWaistPitch"))

    def test_episode_fields_are_reported_per_group(self):
        with tempfile.TemporaryDirectory() as root:
            plain = _episode(root, "episode_0000", with_waist=False)
            yaw_only = _episode(root, "episode_0001", with_waist=True, with_pitch=False)
            both = _episode(root, "episode_0002", with_waist=True)
            self.assertEqual(episode_waist_fields(plain), (False, False))
            self.assertEqual(episode_waist_fields(yaw_only), (True, False))
            self.assertEqual(episode_waist_fields(both), (True, True))
            self.assertEqual(episode_waist_fields(Path(root) / "missing"), (False, False))

    def test_a_task_that_agrees_on_waist_is_accepted(self):
        with tempfile.TemporaryDirectory() as root:
            dirs = [
                Path(_episode(root, "episode_0000", with_waist=True)),
                Path(_episode(root, "episode_0001", with_waist=True)),
            ]
            dims = {
                "action.waist.qpos": 1, "action.waist_pitch.qpos": 1,
                "observation.state.waist.qpos": 1, "observation.state.waist_pitch.qpos": 1,
            }
            check_waist_consistency(dirs, dims)
        with tempfile.TemporaryDirectory() as root:
            dirs = [
                Path(_episode(root, "episode_0000", with_waist=False)),
                Path(_episode(root, "episode_0001", with_waist=False)),
            ]
            check_waist_consistency(dirs, {"action.left_arm.qpos": 7})

    def test_mixed_waist_episodes_are_refused_not_padded(self):
        with tempfile.TemporaryDirectory() as root:
            dirs = [
                Path(_episode(root, "episode_0000", with_waist=False)),
                Path(_episode(root, "episode_0001", with_waist=True)),
            ]
            with self.assertRaises(ValueError) as ctx:
                check_waist_consistency(dirs, {"action.left_arm.qpos": 7})
            message = str(ctx.exception)
            self.assertIn("episode_0000", message)
            self.assertIn("episode_0001", message)
            self.assertIn("separate datasets", message)

    def test_yaw_without_bend_is_refused(self):
        # collect.py always writes both groups together; a half pair means the
        # file was assembled by hand and would leave the bend unpadded.
        with tempfile.TemporaryDirectory() as root:
            dirs = [
                Path(_episode(root, "episode_0000", with_waist=True)),
                Path(_episode(root, "episode_0001", with_waist=True, with_pitch=False)),
            ]
            with self.assertRaises(ValueError) as ctx:
                check_waist_consistency(dirs, {"action.left_arm.qpos": 7})
            self.assertIn("waist_pitch.qpos", str(ctx.exception))

    def test_schema_that_missed_a_recorded_column_is_refused(self):
        # --schema-episodes 1 scans only the first episode; a waist column that
        # only later episodes carry would otherwise be written out as zeros.
        with tempfile.TemporaryDirectory() as root:
            dirs = [
                Path(_episode(root, "episode_0000", with_waist=True)),
                Path(_episode(root, "episode_0001", with_waist=True)),
            ]
            with self.assertRaises(ValueError) as ctx:
                check_waist_consistency(dirs, {"action.left_arm.qpos": 7})
            self.assertIn("--schema-episodes 0", str(ctx.exception))

    def test_default_robot_type_follows_the_json(self):
        plain = {"action.left_arm.qpos": 7}
        waist = {"action.waist.qpos": 1, "action.waist_pitch.qpos": 1}
        self.assertEqual(robot_type_for_schema(NO_WAIST_ROBOT_TYPE, plain), NO_WAIST_ROBOT_TYPE)
        self.assertEqual(robot_type_for_schema(NO_WAIST_ROBOT_TYPE, waist), WAIST_ROBOT_TYPE)
        self.assertEqual(robot_type_for_schema("Unitree_G1_Dex1", waist), "Unitree_G1_Dex1")


class PitchTrackTests(unittest.TestCase):
    def test_target_cannot_lead_the_measured_angle(self):
        # Full bend asked from upright only advances by the lead cap.
        self.assertAlmostEqual(tracked_pitch_command(2.0, 0.0, 0.10, 0.0, 2.3562), 0.10)

    def test_target_is_clipped_to_the_joint_range(self):
        self.assertAlmostEqual(tracked_pitch_command(-1.0, 0.0, 0.10, 0.0, 2.3562), 0.0)
        self.assertAlmostEqual(tracked_pitch_command(9.0, 2.3562, 0.10, 0.0, 2.3562), 2.3562)


if __name__ == "__main__":
    unittest.main()
