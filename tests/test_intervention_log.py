"""Unit tests for intervention JSONL logging."""
import json
import os
import tempfile
import unittest

from teleop.utils.intervention_log import InterventionLogger


class InterventionLogTests(unittest.TestCase):
    def test_disabled_without_episode_dir(self):
        log = InterventionLogger()
        log.log("rollback_start", phase="POLICY_ROLLBACK")
        self.assertIsNone(log.path)

    def test_writes_jsonl_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = InterventionLogger(tmp)
            log.log("rollback_start", phase="POLICY_ROLLBACK")
            log.log("align_ok", phase="ALIGNING", frame_idx=3)
            path = os.path.join(tmp, "intervention.jsonl")
            self.assertTrue(os.path.isfile(path))
            with open(path, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["event"], "rollback_start")
            self.assertEqual(rows[1]["frame_idx"], 3)
            log.clear()
            log.log("teleop_enter", phase="TELEOP_LIVE")
            # cleared logger must not append more
            with open(path, encoding="utf-8") as handle:
                rows_after = [json.loads(line) for line in handle]
            self.assertEqual(len(rows_after), 2)


if __name__ == "__main__":
    unittest.main()
