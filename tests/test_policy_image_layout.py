import os
import tempfile
import unittest

import cv2
import numpy as np

from teleop.utils.policy_client import PolicyAdapter, build_zmq_request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINIMAL_YAML = """
common:
  video_height: 384
  video_width: 320
dataset:
  stitch_mode: aspect
  action_mode: qpos
  image_layout: separate
"""


def _write_config(text):
    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    handle.write(text)
    handle.close()
    return handle.name


class PolicyImageLayoutTests(unittest.TestCase):
    def setUp(self):
        self.head = np.zeros((4, 8, 3), dtype=np.uint8)
        self.head[:, :4] = (1, 2, 3)
        self.head[:, 4:] = (4, 5, 6)
        self.left = np.full((4, 5, 3), 7, dtype=np.uint8)
        self.right = np.full((4, 6, 3), 9, dtype=np.uint8)
        self.arm_q = np.zeros(14, dtype=float)

    def test_default_config_stays_stitched(self):
        adapter = PolicyAdapter(
            os.path.join(REPO_ROOT, "configs", "infer_g1d.yaml"),
            swap_wrists=False,
            action_interp_factor=1,
            exec_chunk_steps=0,
            pad_joint_values="",
        )
        observation = adapter.build_model_input(
            self.head, self.left, self.right, self.arm_q, 0.0, 0.0,
        )
        self.assertEqual(observation.image_layout, "stitched")
        self.assertEqual(observation.first_frame.shape, (384, 320, 3))
        request = build_zmq_request(observation, "pick")
        self.assertEqual(request["image_layout"], "stitched")
        self.assertIn("first_frame", request)
        self.assertNotIn("images", request)
        self.assertEqual(request["state"].shape, (16,))

    def test_separate_sends_left_eye_and_wrists(self):
        path = _write_config(MINIMAL_YAML)
        try:
            plain = PolicyAdapter(
                path, swap_wrists=False, action_interp_factor=1,
                exec_chunk_steps=0, pad_joint_values="",
            )
            swapped = PolicyAdapter(
                path, swap_wrists=True, action_interp_factor=1,
                exec_chunk_steps=0, pad_joint_values="",
            )
        finally:
            os.unlink(path)
        observation = plain.build_model_input(
            self.head, self.left, self.right, self.arm_q, 0.1, 0.2,
        )
        self.assertEqual(list(observation.images), ["left_eye", "left_wrist", "right_wrist"])
        self.assertEqual(observation.images["left_eye"].shape, (4, 4, 3))
        np.testing.assert_array_equal(
            observation.images["left_eye"],
            cv2.cvtColor(self.head[:, :4], cv2.COLOR_BGR2RGB),
        )
        np.testing.assert_array_equal(
            observation.images["left_wrist"],
            cv2.cvtColor(self.left, cv2.COLOR_BGR2RGB),
        )
        np.testing.assert_array_equal(
            observation.images["right_wrist"],
            cv2.cvtColor(self.right, cv2.COLOR_BGR2RGB),
        )
        request = build_zmq_request(observation, "pick")
        self.assertEqual(request["image_layout"], "separate")
        self.assertNotIn("first_frame", request)
        self.assertEqual(set(request["images"]), {"left_eye", "left_wrist", "right_wrist"})
        self.assertIsNone(request["rtc_prev"])
        swapped_obs = swapped.build_model_input(
            self.head, self.left, self.right, self.arm_q, 0.0, 0.0,
        )
        np.testing.assert_array_equal(
            swapped_obs.images["left_wrist"],
            cv2.cvtColor(self.right, cv2.COLOR_BGR2RGB),
        )

    def test_unknown_layout_is_rejected(self):
        path = _write_config(MINIMAL_YAML.replace("separate", "mosaic"))
        try:
            with self.assertRaises(ValueError):
                PolicyAdapter(
                    path, swap_wrists=False, action_interp_factor=1,
                    exec_chunk_steps=0, pad_joint_values="",
                )
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
