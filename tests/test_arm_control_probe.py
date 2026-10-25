"""Hardware-free tests for arm-controller detection and stopping."""
import os
import signal
import tempfile
import unittest
from unittest import mock

from teleop.utils import arm_control_probe as probe
from teleop.utils.arm_owner_lock import ArmOwnerLock


def fake_proc_tree(root, entries):
    """Build a minimal /proc tree: {pid: (cmdline_tokens, exe, sigcgt_mask)}."""
    for pid, (tokens, exe, mask) in entries.items():
        pid_dir = os.path.join(root, str(pid))
        os.makedirs(pid_dir, exist_ok=True)
        if tokens is not None:
            with open(os.path.join(pid_dir, "cmdline"), "wb") as handle:
                handle.write(b"\0".join(part.encode() for part in tokens) + b"\0")
        if exe is not None:
            os.symlink(exe, os.path.join(pid_dir, "exe"))
        if mask is not None:
            with open(os.path.join(pid_dir, "status"), "w", encoding="utf-8") as handle:
                handle.write(f"Name:\tpython\nSigCgt:\t{mask:016x}\n")


class TokenMatchingTests(unittest.TestCase):
    def test_path_token_matches_controller(self):
        spec = probe.classify_tokens(
            ["/home/unitree/miniconda3/envs/tv/bin/python", "-u",
             "calib_arm_server.py", "--yes", "--vmax", "0.2"])
        self.assertIsNotNone(spec)
        self.assertEqual(spec.name, "calib_arm_server")

    def test_module_form_matches_replay(self):
        spec = probe.classify_tokens(["python", "-m", "teleop.replay", "--data-json", "x"])
        self.assertIsNotNone(spec)
        self.assertEqual(spec.name, "replay")

    def test_unrelated_process_is_ignored(self):
        self.assertIsNone(probe.classify_tokens(["/usr/bin/python3", "networkd-dispatcher"]))
        self.assertIsNone(probe.classify_tokens(["/usr/bin/python3", "teleimager-server"]))

    def test_substring_of_longer_word_does_not_match(self):
        # "collect_data_converter" must not be treated as the collect entrypoint.
        self.assertIsNone(probe.classify_tokens(["python", "collect_data_converter.py"]))

    def test_editor_holding_the_path_is_not_a_controller(self):
        self.assertFalse(probe.looks_like_python("node", ["node", "calib_arm_server.py"]))
        self.assertFalse(probe.looks_like_python("bash", ["bash", "/x/collect.py"]))
        self.assertTrue(probe.looks_like_python("python3.10", ["python3.10", "collect.py"]))


class SignalScanTests(unittest.TestCase):
    def test_signal_mask_decoding(self):
        # SIGINT(2) -> bit 1, SIGTERM(15) -> bit 14
        mask = (1 << 1) | (1 << 14)
        self.assertEqual(probe.signal_mask_to_set(mask), frozenset({signal.SIGINT, signal.SIGTERM}))

    def test_choose_signal_prefers_sigterm_when_registered(self):
        self.assertEqual(probe.choose_stop_signal(frozenset({signal.SIGTERM})), signal.SIGTERM)
        self.assertEqual(probe.choose_stop_signal(frozenset()), signal.SIGINT)
        # CPython always installs a SIGINT handler even when the program ignores it.
        self.assertEqual(probe.choose_stop_signal(frozenset({signal.SIGINT})), signal.SIGINT)

    def test_scan_finds_controller_and_skips_others(self):
        with tempfile.TemporaryDirectory() as root:
            fake_proc_tree(root, {
                100: (["/env/bin/python", "-u", "calib_arm_server.py", "--yes"], "/env/bin/python", 0),
                200: (["/usr/bin/python3", "teleimager-server"], "/usr/bin/python3", 0),
                300: (["/env/bin/python", "collect.py", "--record"], "/env/bin/python", 0),
            })
            found = probe.scan_controllers(exclude_pids={300}, root=root)
            self.assertEqual([p.pid for p in found], [100])
            self.assertEqual(found[0].spec.name, "calib_arm_server")

    def test_scan_skips_processes_it_cannot_read(self):
        with tempfile.TemporaryDirectory() as root:
            fake_proc_tree(root, {
                100: (None, None, None),  # vanished between listdir and open
            })
            self.assertEqual(probe.scan_controllers(root=root), [])


class LockTests(unittest.TestCase):
    def test_stale_lock_text_is_not_reported_as_held(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "arm.lock")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("deploy pid=74645\n")
            self.assertEqual(probe.read_lock_holder(path), "deploy pid=74645")
            self.assertFalse(probe.lock_is_held(path))

    def test_live_lock_is_reported_as_held(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "arm.lock")
            held = ArmOwnerLock(path, "deploy").acquire()
            try:
                self.assertTrue(probe.lock_is_held(path))
            finally:
                held.release()
            self.assertFalse(probe.lock_is_held(path))


class VerdictTests(unittest.TestCase):
    def _proc(self):
        spec = probe.CONTROLLER_SPECS[0]
        return probe.ControllerProcess(pid=1, spec=spec, cmdline="python x", exe="python")

    def _traffic(self, count, topic="rt/lowcmd", arm=True):
        return probe.TopicTraffic(topic=topic, label="x", arm=arm, count=count)

    def test_process_makes_it_busy(self):
        verdict = probe.overall_verdict([self._proc()], True, [], False)
        self.assertEqual(verdict.state, "busy")

    def test_traffic_without_process_makes_it_busy(self):
        verdict = probe.overall_verdict([], True, [self._traffic(40)], False)
        self.assertEqual(verdict.state, "busy")

    def test_stale_lock_alone_does_not_block(self):
        verdict = probe.overall_verdict([], True, [], False)
        self.assertEqual(verdict.state, "free")

    def test_held_lock_alone_makes_it_busy(self):
        verdict = probe.overall_verdict([], True, [], True)
        self.assertEqual(verdict.state, "busy")

    def test_broken_probe_is_unknown_never_free(self):
        # This is the regression that matters: a dead probe must not read as "free".
        verdict = probe.overall_verdict([], False, [], False)
        self.assertEqual(verdict.state, "unknown")
        self.assertNotEqual(verdict.state, "free")

    def test_base_velocity_traffic_is_reported_but_not_blocking(self):
        probe_result = probe.ProbeResult(
            ok=True, positive_count=10,
            traffic=[self._traffic(50, "rt/cmd_vel_no_limit", arm=False), self._traffic(0)],
        )
        self.assertEqual(probe_result.active_arm_topics, [])
        self.assertEqual(len(probe_result.active_other_topics), 1)
        verdict = probe.overall_verdict([], probe_result.ok, probe_result.active_arm_topics, False)
        self.assertEqual(verdict.state, "free")


class ProbeResultTests(unittest.TestCase):
    def test_empty_arm_topics_is_free_when_probe_ran(self):
        result = probe.ProbeResult(ok=True, positive_count=20, traffic=[
            probe.TopicTraffic("rt/lowcmd", "x", True, 0),
            probe.TopicTraffic("rt/waistpitchcmd", "y", True, 0),
        ])
        self.assertEqual(result.active_arm_topics, [])


class StopSignalTests(unittest.TestCase):
    def test_send_to_missing_process_is_treated_as_done(self):
        with mock.patch.object(probe.os, "kill", side_effect=ProcessLookupError):
            ok, message = probe.send_stop_signal(999999, signal.SIGINT)
        self.assertTrue(ok)
        self.assertIn("已不在", message)

    def test_permission_error_is_reported(self):
        with mock.patch.object(probe.os, "kill", side_effect=PermissionError("denied")):
            ok, message = probe.send_stop_signal(1, signal.SIGINT)
        self.assertFalse(ok)
        self.assertIn("权限", message)

    def test_wait_for_exit_returns_survivors(self):
        with mock.patch.object(probe, "pid_alive", return_value=True):
            survivors = probe.wait_for_exit([123], timeout=0.05, poll=0.01)
        self.assertEqual(survivors, [123])


if __name__ == "__main__":
    unittest.main()
