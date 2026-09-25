import unittest

from integrations.g1d_infra_compat.operator_events import OperatorEvent, PicoEventMapper
from integrations.g1d_infra_compat.voice_announcer import VoiceAnnouncer


class CompatTest(unittest.TestCase):
    def test_pico_rising_edges_and_x_failure(self):
        mapper = PicoEventMapper()
        sample = {"motion_data_ready": True, "left_x": True}
        self.assertEqual(mapper.update(sample), [OperatorEvent.MARK_FAILED])
        self.assertEqual(mapper.update(sample), [])
        mapper.update({"motion_data_ready": True})
        self.assertEqual(mapper.update({"motion_data_ready": True, "right_a": True}), [OperatorEvent.START_OR_TOGGLE_PAUSE])

    def test_voice_sink_without_audio_backend(self):
        messages = []
        announcer = VoiceAnnouncer(enabled=False, sink=messages.append)
        announcer.announce("trajectory_failed")
        announcer.close()
        self.assertEqual(messages, ["轨迹失败，已标记"])


if __name__ == "__main__":
    unittest.main()
