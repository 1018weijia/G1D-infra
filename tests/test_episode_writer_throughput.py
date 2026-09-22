"""Regressions for the recording stalls seen during the first on-robot session.

Two failures were observed:
  1. Pressing record froze the arm for ~3.7 s, because create_episode() built a
     second RerunLogger and rr.spawn() blocks with no display.
  2. Saving an episode took ~41 s and the next episode could not be started,
     because the writer fell behind a 25 Hz recording and the save waited for
     the queue to happen to be empty.
"""
import json
import os
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.rerun_visualizer import should_log_to_rerun, viewer_can_start


def frame(index, cameras=4, size=(48, 64)):
    arm = {"qpos": [0.1 * index] * 7, "qvel": [], "torque": []}
    states = {
        "left_arm": dict(arm),
        "right_arm": dict(arm),
        "left_ee": {"qpos": [0.5], "qvel": [], "torque": []},
        "right_ee": {"qpos": [0.5], "qvel": [], "torque": []},
    }
    colors = {
        f"color_{c}": np.full((size[0], size[1], 3), (index * 7 + c) % 256, dtype=np.uint8)
        for c in range(cameras)
    }
    return colors, states


def read_episode(episode_dir):
    with open(os.path.join(episode_dir, "data.json"), encoding="utf-8") as handle:
        return json.load(handle)


def wait_ready(recorder, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if recorder.is_ready():
            return time.time()
        time.sleep(0.005)
    raise TimeoutError("recorder did not become ready")


class WriterBase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.recorder = EpisodeWriter(
            task_dir=self.tmpdir.name, task_goal="test", rerun_log=False,
        )

    def tearDown(self):
        self.recorder.close()
        self.tmpdir.cleanup()

    def record(self, frames, success=True):
        self.assertTrue(self.recorder.create_episode())
        for i in range(frames):
            colors, states = frame(i)
            self.recorder.add_item(colors=colors, depths={}, states=states, actions=states)
        episode_dir = self.recorder.episode_dir
        self.recorder.save_episode(success=success)
        wait_ready(self.recorder)
        return episode_dir


class CreateEpisodeStallTests(WriterBase):
    def test_create_episode_does_not_build_a_rerun_logger(self):
        """create_episode() runs in the control loop; rr.spawn() there froze the arm."""
        with mock.patch("teleop.utils.episode_writer.RerunLogger") as spawn:
            self.assertTrue(self.recorder.create_episode())
        spawn.assert_not_called()

    def test_create_episode_is_fast(self):
        start = time.time()
        self.recorder.create_episode()
        self.assertLess(time.time() - start, 0.5)


class SaveCompletesWithBacklogTests(WriterBase):
    def test_save_waits_for_queued_frames_then_completes(self):
        """The save must land after every queued frame, without polling for an empty queue."""
        episode_dir = self.record(40)
        payload = read_episode(episode_dir)
        self.assertEqual(len(payload["data"]), 40)
        self.assertEqual([item["idx"] for item in payload["data"]], list(range(40)))
        self.assertEqual(len(os.listdir(os.path.join(episode_dir, "colors"))), 40 * 4)

    def test_save_is_prompt_once_the_backlog_is_written(self):
        self.recorder.create_episode()
        for i in range(10):
            colors, states = frame(i)
            self.recorder.add_item(colors=colors, depths={}, states=states, actions=states)
        self.recorder.item_data_queue.join()
        requested = time.time()
        self.recorder.save_episode(success=True)
        ready = wait_ready(self.recorder)
        # The old queue.empty() poll added up to a full second here.
        self.assertLess(ready - requested, 0.5)

    def test_two_episodes_back_to_back(self):
        """The reported session: one good episode, then one marked failed."""
        good = self.record(25, success=True)
        bad = self.record(25, success=False)

        self.assertNotEqual(good, bad)
        self.assertTrue(read_episode(good)["success"])
        self.assertFalse(os.path.exists(os.path.join(good, "FAILED")))
        self.assertFalse(read_episode(bad)["success"])
        self.assertTrue(os.path.exists(os.path.join(bad, "FAILED")))

    def test_repeated_save_requests_do_not_close_the_array_twice(self):
        self.recorder.create_episode()
        colors, states = frame(0)
        self.recorder.add_item(colors=colors, depths={}, states=states, actions=states)
        for _ in range(5):
            self.recorder.save_episode(success=True)
        wait_ready(self.recorder)
        self.recorder.save_episode(success=True)
        time.sleep(0.2)
        # A second sentinel would append another '], "success": ...' and break the file.
        payload = read_episode(self.recorder.episode_dir)
        self.assertEqual(len(payload["data"]), 1)

    def test_images_keep_their_content_and_relative_paths(self):
        import cv2

        episode_dir = self.record(3)
        payload = read_episode(episode_dir)
        for index, item in enumerate(payload["data"]):
            self.assertEqual(sorted(item["colors"]), [f"color_{c}" for c in range(4)])
            for camera, rel_path in item["colors"].items():
                self.assertTrue(rel_path.startswith("colors/"), rel_path)
                image = cv2.imread(os.path.join(episode_dir, rel_path))
                self.assertIsNotNone(image, rel_path)
                expected = (index * 7 + int(camera.split("_")[1])) % 256
                self.assertLess(abs(int(image[0, 0, 0]) - expected), 4)


class RerunGatingTests(unittest.TestCase):
    def test_off_unless_requested(self):
        self.assertFalse(should_log_to_rerun(False, env={"DISPLAY": ":0"}))

    def test_headless_wins_over_request(self):
        self.assertFalse(should_log_to_rerun(True, headless=True, env={"DISPLAY": ":0"}))

    def test_requires_a_display(self):
        self.assertFalse(should_log_to_rerun(True, env={}))
        self.assertTrue(should_log_to_rerun(True, env={"DISPLAY": ":0"}))
        self.assertTrue(should_log_to_rerun(True, env={"WAYLAND_DISPLAY": "wayland-0"}))

    def test_viewer_can_start_reads_the_environment(self):
        self.assertFalse(viewer_can_start(env={}))
        self.assertTrue(viewer_can_start(env={"DISPLAY": ":1"}))


if __name__ == "__main__":
    unittest.main()
