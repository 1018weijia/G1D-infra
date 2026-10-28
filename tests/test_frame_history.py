import os
import tempfile
import unittest

import numpy as np

from teleop.utils.frame_history import (
    FrameHistory,
    HistoryConfig,
    parse_history_config,
)
from teleop.utils.policy_client import (
    PolicyAdapter,
    PolicyObservation,
    build_zmq_request,
    check_history_protocol,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME_SHAPE = (16, 12, 3)


def _frame(value=0):
    return np.full(FRAME_SHAPE, value, dtype=np.uint8)


def _gradient(height, width):
    """Smooth BGR test image: JPEG is built for camera content, not noise."""
    rows, cols = np.mgrid[0:height, 0:width]
    return np.stack(
        [(cols * 3) % 256, (rows * 5) % 256, ((cols + rows) * 2) % 256], axis=-1
    ).astype(np.uint8)


class FrameHistoryDisabledTests(unittest.TestCase):
    def test_disabled_captures_nothing_and_creates_no_directory(self):
        with tempfile.TemporaryDirectory() as root:
            history = FrameHistory(root=root, enabled=False)
            history.start(now=0.0)
            self.assertIsNone(history.capture(_frame(7), np.zeros(16), now=0.0))
            history.flush()
            self.assertEqual(len(history), 0)
            self.assertEqual(os.listdir(root), [])
            self.assertIsNone(history.directory)
            self.assertEqual(history.select(0.5, 4), [])


class FrameHistorySamplingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.history = FrameHistory(root=self.root, sample_hz=5.0, enabled=True)
        self.history.start(now=100.0)

    def tearDown(self):
        self.history.discard()
        self._tmp.cleanup()

    def test_throttles_to_sample_rate(self):
        self.assertIsNotNone(self.history.capture(_frame(1), np.zeros(16), now=100.0))
        self.assertIsNone(self.history.capture(_frame(2), np.zeros(16), now=100.1))
        self.assertIsNone(self.history.capture(_frame(3), np.zeros(16), now=100.19))
        self.assertIsNotNone(self.history.capture(_frame(4), np.zeros(16), now=100.2))
        self.assertEqual(len(self.history), 2)

    def test_capture_is_noop_before_start_and_after_stop(self):
        history = FrameHistory(root=self.root, sample_hz=5.0, enabled=True)
        self.assertIsNone(history.capture(_frame(1), np.zeros(16), now=0.0))
        history.start(now=0.0)
        self.assertIsNotNone(history.capture(_frame(1), np.zeros(16), now=0.0))
        history.stop()
        self.assertIsNone(history.capture(_frame(1), np.zeros(16), now=5.0))

    def test_writes_one_jpeg_and_index_line_per_sample(self):
        self.history.capture(_frame(3), np.arange(16, dtype=np.float32), now=100.0)
        self.history.capture(_frame(9), np.arange(16, dtype=np.float32), now=100.2)
        self.history.flush()
        directory = self.history.directory
        self.assertTrue(directory)
        files = sorted(os.listdir(directory))
        self.assertEqual(files, ["000000.jpg", "000001.jpg", "index.jsonl"])
        with open(os.path.join(directory, "index.jsonl"), encoding="utf-8") as handle:
            lines = [line for line in handle.read().splitlines() if line]
        self.assertEqual(len(lines), 2)
        self.assertIn('"rollback": false', lines[0])

    def test_state_is_stored_raw_and_as_16_floats(self):
        state = np.arange(16, dtype=np.float32)
        self.history.capture(_frame(3), state, now=100.0)
        self.history.flush()
        samples = self.history.select(0.0, 1, include_first=False)
        self.assertEqual(len(samples), 1)
        np.testing.assert_array_equal(samples[0].state, state)


class FrameHistorySelectionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.history = FrameHistory(root=self.root, sample_hz=5.0, enabled=True)
        self.history.start(now=0.0)
        for index in range(20):
            self.history.capture(
                _frame(index), np.full(16, index, dtype=np.float32), now=index * 0.2
            )
        self.history.flush()

    def tearDown(self):
        self.history.discard()
        self._tmp.cleanup()

    def test_select_honours_interval_and_count(self):
        # Samples sit 0.2s apart; walking back at 0.6s spacing lands every 3rd.
        samples = self.history.select(0.6, 4, include_first=False)
        self.assertEqual([sample.n for sample in samples], [19, 16, 13, 10])
        chosen = [sample.t_rel for sample in samples]
        for newer, older in zip(chosen, chosen[1:]):
            self.assertGreaterEqual(newer - older, 0.6 - 1e-9)

        # A shorter interval must keep more, still respecting the count.
        tighter = self.history.select(0.4, 4, include_first=False)
        self.assertEqual([sample.n for sample in tighter], [19, 17, 15, 13])

    def test_select_never_duplicates_or_pads(self):
        samples = self.history.select(0.5, 100, include_first=True)
        indices = [sample.n for sample in samples]
        self.assertEqual(len(indices), len(set(indices)))
        self.assertLessEqual(len(indices), len(self.history))
        self.assertNotIn(0, indices[:-1])

    def test_short_history_returns_fewer_frames(self):
        small = FrameHistory(root=self.root, sample_hz=5.0, enabled=True)
        small.start(now=0.0)
        small.capture(_frame(1), np.zeros(16), now=0.0)
        small.flush()
        try:
            samples = small.select(0.5, 16, include_first=False)
            self.assertEqual(len(samples), 1)
        finally:
            small.discard()

    def test_include_first_appends_sample_zero_exactly_once(self):
        samples = self.history.select(0.5, 4, include_first=True)
        self.assertEqual(samples[-1].n, 0)
        self.assertEqual([sample.n for sample in samples].count(0), 1)

        already = self.history.select(100.0, 16, include_first=True)
        self.assertEqual(already[-1].n, 0)
        self.assertEqual([sample.n for sample in already].count(0), 1)

    def test_select_returns_zero_count_as_empty(self):
        self.assertEqual(self.history.select(0.5, 0), [])

    def test_count_is_a_hard_cap_including_the_first_frame(self):
        samples = self.history.select(0.0, 3, include_first=True)
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[-1].n, 0)

        single = self.history.select(0.0, 1, include_first=True)
        self.assertEqual([sample.n for sample in single], [0])

        newest_only = self.history.select(0.0, 1, include_first=False)
        self.assertEqual([sample.n for sample in newest_only], [19])

    def test_jpeg_payload_is_decodable_and_matches_frame_order(self):
        samples = self.history.select(0.0, 1, include_first=False)
        import cv2
        decoded = cv2.imdecode(
            np.frombuffer(samples[0].jpeg, np.uint8), cv2.IMREAD_COLOR
        )
        self.assertEqual(decoded.shape, FRAME_SHAPE)


class FrameHistoryRollbackTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.history = FrameHistory(root=self.root, sample_hz=5.0, enabled=True)
        self.history.start(now=0.0)
        for index in range(6):
            self.history.capture(
                _frame(index),
                np.zeros(16),
                now=index * 0.2,
                rolling_back=index in (2, 3, 4),
            )
        self.history.flush()

    def tearDown(self):
        self.history.discard()
        self._tmp.cleanup()

    def test_rollback_samples_are_excluded_by_default(self):
        samples = self.history.select(0.0, 16, include_first=False)
        self.assertEqual([sample.n for sample in samples], [5, 1, 0])

    def test_include_rollback_keeps_them(self):
        samples = self.history.select(
            0.0, 16, include_first=False, include_rollback=True
        )
        self.assertEqual([sample.n for sample in samples], [5, 4, 3, 2, 1, 0])

    def test_rollback_first_frame_is_skipped_only_when_excluded(self):
        history = FrameHistory(root=self.root, sample_hz=5.0, enabled=True)
        history.start(now=0.0)
        history.capture(_frame(1), np.zeros(16), now=0.0, rolling_back=True)
        history.capture(_frame(2), np.zeros(16), now=1.0)
        history.flush()
        try:
            self.assertEqual(
                [s.n for s in history.select(0.0, 16, include_first=True)], [1]
            )
            self.assertEqual(
                [
                    s.n
                    for s in history.select(
                        0.0, 16, include_first=True, include_rollback=True
                    )
                ],
                [1, 0],
            )
        finally:
            history.discard()


class FrameHistoryLifecycleTests(unittest.TestCase):
    def test_discard_removes_directory_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as root:
            history = FrameHistory(root=root, sample_hz=5.0, enabled=True)
            history.start(now=0.0)
            history.capture(_frame(1), np.zeros(16), now=0.0)
            history.flush()
            directory = history.directory
            self.assertTrue(os.path.isdir(directory))
            history.discard()
            self.assertFalse(os.path.exists(directory))
            self.assertIsNone(history.directory)
            history.discard()
            self.assertEqual(os.listdir(root), [])

    def test_each_instance_gets_its_own_directory(self):
        with tempfile.TemporaryDirectory() as root:
            first = FrameHistory(root=root, sample_hz=5.0, enabled=True)
            first.start(now=0.0)
            first.capture(_frame(1), np.zeros(16), now=0.0)
            first.flush()
            second = FrameHistory(root=root, sample_hz=5.0, enabled=True)
            second.start(now=0.0)
            try:
                self.assertTrue(os.path.isdir(second.directory))
                self.assertNotEqual(first.directory, second.directory)
            finally:
                first.discard()
                second.discard()

    def test_discarded_instance_stays_dead(self):
        with tempfile.TemporaryDirectory() as root:
            history = FrameHistory(root=root, sample_hz=5.0, enabled=True)
            history.start(now=0.0)
            history.capture(_frame(1), np.zeros(16), now=0.0)
            directory = history.directory
            history.discard()
            history.start(now=0.0)
            history.capture(_frame(1), np.zeros(16), now=1.0)
            self.assertFalse(history.started)
            self.assertEqual(len(history), 0)
            self.assertFalse(os.path.exists(directory))
            self.assertEqual(os.listdir(root), [])

    def test_resume_keeps_existing_samples_and_start_time(self):
        with tempfile.TemporaryDirectory() as root:
            history = FrameHistory(root=root, sample_hz=5.0, enabled=True)
            history.start(now=0.0)
            history.capture(_frame(1), np.zeros(16), now=0.0)
            history.stop()
            history.start(now=10.0)
            history.capture(_frame(2), np.zeros(16), now=10.0)
            history.flush()
            try:
                self.assertEqual(len(history), 2)
                samples = history.select(0.0, 16, include_first=False)
                self.assertEqual(samples[0].n, 1)
                self.assertAlmostEqual(samples[0].t_rel, 10.0)
            finally:
                history.discard()


class FrameHistoryRingTests(unittest.TestCase):
    def test_ring_evicts_oldest_but_keeps_the_first_sample(self):
        with tempfile.TemporaryDirectory() as root:
            history = FrameHistory(
                root=root, sample_hz=5.0, enabled=True, max_samples=5
            )
            history.start(now=0.0)
            for index in range(12):
                history.capture(_frame(index), np.zeros(16), now=index * 0.2)
            history.flush()
            try:
                self.assertEqual(len(history), 5)
                indices = [sample.n for sample in history._index]
                self.assertEqual(indices[0], 0)
                self.assertEqual(indices[-4:], [8, 9, 10, 11])
                # Evicted JPEGs leave no files behind, including in-flight ones.
                files = sorted(
                    name for name in os.listdir(history.directory)
                    if name.endswith(".jpg")
                )
                self.assertEqual(
                    files,
                    ["000000.jpg", "000008.jpg", "000009.jpg",
                     "000010.jpg", "000011.jpg"],
                )
            finally:
                history.discard()

    def test_dropped_sample_never_occupies_a_ring_slot(self):
        with tempfile.TemporaryDirectory() as root:
            history = FrameHistory(
                root=root, sample_hz=5.0, enabled=True, max_pending=1
            )
            # Never start the writer so every queued capture after the first fails.
            history.enabled = True
            history._started = True
            history._dir = history._new_run_dir()
            os.makedirs(history._dir, exist_ok=True)
            history._t0 = 0.0
            with self.assertLogs("teleop.utils.frame_history", level="WARNING"):
                kept = [
                    history.capture(_frame(i), np.zeros(16), now=i * 0.2)
                    for i in range(4)
                ]
            try:
                self.assertEqual(len(history), 1)
                self.assertEqual([sample is None for sample in kept].count(True), 3)
            finally:
                history.discard()

    def test_max_samples_is_configurable(self):
        config = parse_history_config({"enabled": True, "max_samples": 300})
        self.assertEqual(config.max_samples, 300)
        with self.assertRaises(ValueError) as raised:
            parse_history_config({"enabled": True, "max_samples": 1})
        self.assertIn("history.max_samples", str(raised.exception))


class HistoryConfigTests(unittest.TestCase):
    def test_absent_block_is_disabled(self):
        self.assertEqual(parse_history_config(None), HistoryConfig())

    def test_defaults_apply_to_partial_block(self):
        config = parse_history_config({"enabled": True})
        self.assertTrue(config.enabled)
        self.assertFalse(config.send)
        self.assertEqual(config.count, 16)
        self.assertAlmostEqual(config.period, 0.2)

    def test_unknown_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_history_config({"enabled": True, "countt": 4})

    def test_bad_types_name_the_key(self):
        with self.assertRaises(ValueError) as raised:
            parse_history_config({"sample_hz": "fast"})
        self.assertIn("history.sample_hz", str(raised.exception))
        with self.assertRaises(ValueError) as raised:
            parse_history_config({"count": 2.5})
        self.assertIn("history.count", str(raised.exception))
        with self.assertRaises(ValueError) as raised:
            parse_history_config({"enabled": "yes"})
        self.assertIn("history.enabled", str(raised.exception))

    def test_send_requires_enabled(self):
        with self.assertRaises(ValueError) as raised:
            parse_history_config({"send": True})
        self.assertIn("history.send", str(raised.exception))

    def test_non_positive_sample_rate_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_history_config({"sample_hz": 0})

    def test_adapter_reads_block_from_config_file(self):
        adapter = PolicyAdapter(
            os.path.join(REPO_ROOT, "configs", "infer_g1d.yaml"),
            swap_wrists=False,
            action_interp_factor=1,
            exec_chunk_steps=0,
            pad_joint_values="",
        )
        self.assertFalse(adapter.history_config.enabled)
        self.assertFalse(adapter.history_config.send)


class HistoryProtocolTests(unittest.TestCase):
    def test_zmq_accepts_sending_history(self):
        config = HistoryConfig(enabled=True, send=True)
        check_history_protocol(config, "zmq")
        check_history_protocol(config, "ZMQ")

    def test_websocket_is_refused_when_sending(self):
        config = HistoryConfig(enabled=True, send=True)
        for protocol in ("ws", "websocket"):
            with self.assertRaises(RuntimeError) as raised:
                check_history_protocol(config, protocol)
            self.assertIn("zmq", str(raised.exception))

    def test_websocket_is_fine_when_only_sampling(self):
        check_history_protocol(HistoryConfig(enabled=True, send=False), "ws")

    def test_disabled_config_is_never_blocked(self):
        for protocol in ("ws", "zmq"):
            check_history_protocol(HistoryConfig(), protocol)


class HistoryRequestTests(unittest.TestCase):
    def test_request_without_history_is_unchanged(self):
        observation = PolicyObservation(
            image_layout="stitched",
            state=np.zeros(16, dtype=np.float32),
            first_frame=np.zeros((8, 8, 3), np.uint8),
        )
        request = build_zmq_request(observation, "pick")
        self.assertNotIn("history", request)
        self.assertEqual(
            sorted(request),
            sorted([
                "image_layout", "state", "instruction", "rtc_prev",
                "rtc_inference_delay", "first_frame",
            ]),
        )

    def test_empty_history_list_also_adds_no_key(self):
        observation = PolicyObservation(
            image_layout="stitched",
            state=np.zeros(16, dtype=np.float32),
            first_frame=np.zeros((8, 8, 3), np.uint8),
            history_frames=[],
        )
        self.assertNotIn("history", build_zmq_request(observation, "pick"))

    def test_history_frames_travel_as_jpeg_bytes(self):
        jpeg = b"\xff\xd8jpeg"
        frames = [
            (0, 0.0, jpeg, np.zeros(16, np.float32)),
            (5, 1.0, jpeg, np.ones(16, np.float32)),
        ]
        observation = PolicyObservation(
            image_layout="stitched",
            state=np.zeros(16, dtype=np.float32),
            first_frame=np.zeros((8, 8, 3), np.uint8),
            history_frames=frames,
        )
        request = build_zmq_request(observation, "pick")
        history = request["history"]
        self.assertEqual(len(history["frames"]), 2)
        self.assertEqual(history["frames"][0].dtype, np.uint8)
        self.assertEqual(history["frames"][0].size, len(jpeg))
        self.assertEqual(history["state"].shape, (2, 16))
        np.testing.assert_allclose(history["t_rel"], [0.0, 1.0])
        # The single-frame fields are untouched by the additive key.
        self.assertEqual(request["image_layout"], "stitched")
        self.assertEqual(request["first_frame"].shape, (8, 8, 3))


class HistoryFrameEncodingTests(unittest.TestCase):
    """The stored JPEG must decode to the same picture the model sees live."""

    def setUp(self):
        self.adapter = PolicyAdapter(
            os.path.join(REPO_ROOT, "configs", "infer_g1d.yaml"),
            swap_wrists=False,
            action_interp_factor=1,
            exec_chunk_steps=0,
            pad_joint_values="",
        )
        # Smooth gradients, not noise: JPEG at quality 90 is meant for camera
        # content, and pure noise would exaggerate its error far beyond reality.
        self.head = _gradient(48, 64)
        self.left = _gradient(24, 32)
        self.right = _gradient(32, 24)

    def test_jpeg_round_trip_matches_first_frame(self):
        import cv2

        frame = self.adapter.build_history_frame(self.head, self.left, self.right)
        self.assertEqual(frame.shape, (384, 320, 3))
        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )
        self.assertTrue(ok)
        # cv2 decodes to BGR, which is the order the frame was encoded in.
        decoded = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        observation = self.adapter.build_model_input(
            self.head, self.left, self.right, np.zeros(14, dtype=float), 0.0, 0.0,
        )
        live = observation.first_frame
        self.assertEqual(decoded.shape, live.shape)
        # Lossy at quality 95, but the content must be the same picture.
        difference = np.abs(decoded.astype(np.int16) - live.astype(np.int16))
        self.assertLess(float(difference.mean()), 4.0)

    def test_missing_frame_returns_none(self):
        self.assertIsNone(self.adapter.build_history_frame(None, self.left, self.right))

    def test_swap_wrists_is_applied_like_the_live_path(self):
        import cv2

        swapped = PolicyAdapter(
            os.path.join(REPO_ROOT, "configs", "infer_g1d.yaml"),
            swap_wrists=True,
            action_interp_factor=1,
            exec_chunk_steps=0,
            pad_joint_values="",
        )
        # Swapped stitching is not symmetric, so the frame must match the live
        # canvas that the same adapter builds for the model.
        frame = swapped.build_history_frame(self.head, self.left, self.right)
        live = swapped.build_model_input(
            self.head, self.left, self.right, np.zeros(14, dtype=float), 0.0, 0.0,
        ).first_frame
        np.testing.assert_array_equal(
            frame, cv2.cvtColor(live, cv2.COLOR_RGB2BGR)
        )


if __name__ == "__main__":
    unittest.main()
