import json
import os
import tempfile
import time
import unittest

from teleop.utils.episode_writer import EpisodeWriter


def wait_ready(recorder, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if recorder.is_ready():
            return
        time.sleep(0.01)
    raise TimeoutError("recorder did not become ready")


class EpisodeWriterFailTagTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.recorder = EpisodeWriter(
            task_dir=self.tmpdir.name,
            task_goal="test",
            rerun_log=False,
        )

    def tearDown(self):
        self.recorder.close()
        self.tmpdir.cleanup()

    def _record_one_item(self, success):
        self.assertTrue(self.recorder.create_episode())
        self.recorder.add_item(colors={}, states={"ok": True}, actions={})
        self.recorder.save_episode(success=success)
        wait_ready(self.recorder)
        json_path = os.path.join(self.recorder.episode_dir, "data.json")
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload, self.recorder.episode_dir

    def test_successful_episode_has_success_true_and_no_failed_marker(self):
        payload, episode_dir = self._record_one_item(success=True)
        self.assertTrue(payload["success"])
        self.assertEqual(len(payload["data"]), 1)
        self.assertFalse(os.path.exists(os.path.join(episode_dir, "FAILED")))

    def test_failed_episode_is_saved_and_tagged(self):
        payload, episode_dir = self._record_one_item(success=False)
        self.assertFalse(payload["success"])
        self.assertTrue(os.path.exists(os.path.join(episode_dir, "FAILED")))

    def test_close_does_not_overwrite_failed_label(self):
        self.assertTrue(self.recorder.create_episode())
        self.recorder.add_item(colors={}, states={"ok": True}, actions={})
        self.recorder.save_episode(success=False)
        self.recorder.close()
        json_path = os.path.join(self.recorder.episode_dir, "data.json")
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.assertFalse(payload["success"])
        self.assertTrue(os.path.exists(os.path.join(self.recorder.episode_dir, "FAILED")))
        # close() already joined the worker; avoid double-close in tearDown.
        self.recorder.is_available = True
        self.recorder.stop_worker = True


if __name__ == "__main__":
    unittest.main()
