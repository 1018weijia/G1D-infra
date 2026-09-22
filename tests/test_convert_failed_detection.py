"""Bad-episode detection in the JSON -> LeRobot converter.

Only the detection helpers are exercised here; a full conversion needs the
unitree_lerobot env. See TESTING.md L2.2 for the end-to-end check.
"""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONVERTER = REPO_ROOT / "data_convert" / "convert_unitree_json.py"

_spec = importlib.util.spec_from_file_location("convert_unitree_json", CONVERTER)
convert_unitree_json = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convert_unitree_json)

episode_marked_failed = convert_unitree_json.episode_marked_failed
detect_failed_episode_ids = convert_unitree_json.detect_failed_episode_ids


def write_episode(root, index, success, marker=None, padding=0):
    """Mirror EpisodeWriter's output. marker=None follows `success`."""
    episode_dir = Path(root) / f"episode_{index:04d}"
    episode_dir.mkdir(parents=True)
    payload = {
        "info": {"image": {"fps": 30}, "note": "x" * padding},
        "text": {"goal": "test"},
        "data": [{"idx": 0, "colors": {}, "states": {}, "actions": {}}],
        "success": success,
    }
    (episode_dir / "data.json").write_text(json.dumps(payload, indent=4), encoding="utf-8")
    if marker is None:
        marker = not success
    if marker:
        (episode_dir / "FAILED").write_text("failed\n", encoding="utf-8")
    return episode_dir


class EpisodeMarkedFailedTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = self.tmpdir.name

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_successful_episode_is_not_failed(self):
        self.assertFalse(episode_marked_failed(write_episode(self.root, 0, True)))

    def test_failed_marker_is_detected(self):
        self.assertTrue(episode_marked_failed(write_episode(self.root, 1, False)))

    def test_success_false_without_marker_is_detected(self):
        episode_dir = write_episode(self.root, 2, False, marker=False)
        self.assertFalse((episode_dir / "FAILED").exists())
        self.assertTrue(episode_marked_failed(episode_dir))

    def test_marker_wins_when_json_says_success(self):
        self.assertTrue(episode_marked_failed(write_episode(self.root, 3, True, marker=True)))

    def test_success_flag_found_past_the_tail_window(self):
        """A long data.json must not push "success" out of the bytes we read."""
        padding = 4 * convert_unitree_json.SUCCESS_TAIL_BYTES
        episode_dir = write_episode(self.root, 4, False, marker=False, padding=padding)
        self.assertGreater(os.path.getsize(episode_dir / "data.json"), padding)
        self.assertTrue(episode_marked_failed(episode_dir))

    def test_missing_data_json_is_not_failed(self):
        episode_dir = Path(self.root) / "episode_0005"
        episode_dir.mkdir()
        self.assertFalse(episode_marked_failed(episode_dir))

    def test_detect_returns_source_episode_numbers(self):
        write_episode(self.root, 0, True)
        write_episode(self.root, 3, False)
        write_episode(self.root, 7, False, marker=False)
        write_episode(self.root, 9, True)
        episode_dirs = sorted(Path(self.root).iterdir())
        self.assertEqual(detect_failed_episode_ids(episode_dirs), {3, 7})


if __name__ == "__main__":
    unittest.main()
