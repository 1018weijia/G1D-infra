import unittest
from types import SimpleNamespace

from teleop.utils.controller_shortcuts import ControllerShortcutMapper, toggle_start_pause


def tele(**kwargs):
    data = {
        "motion_data_ready": True,
        "right_ctrl_aButton": False,
        "right_ctrl_bButton": False,
        "left_ctrl_aButton": False,
        "left_ctrl_bButton": False,
    }
    data.update(kwargs)
    return SimpleNamespace(**data)


class FakeSession:
    def __init__(self):
        self.keys = []
        self.state = {
            "START": False,
            "STOP": False,
            "PAUSED": False,
            "READY": True,
            "RECORD_RUNNING": False,
            "RECORD_TOGGLE": False,
        }

    def on_press(self, key, episode_id=None):
        self.keys.append(key)
        if key == "r":
            start, paused, action = toggle_start_pause(
                self.state["START"], self.state["PAUSED"], self.state["RECORD_RUNNING"]
            )
            self.state["START"] = start
            self.state["PAUSED"] = paused
            self.last_action = action
        elif key == "q":
            self.state["START"] = False
            self.state["PAUSED"] = False
            self.state["STOP"] = True
        elif key == "s" and self.state["START"] and not (self.state["PAUSED"] and not self.state["RECORD_RUNNING"]):
            self.state["RECORD_TOGGLE"] = True
        elif key == "f" and self.state["START"] and self.state["RECORD_RUNNING"]:
            self.state["RECORD_TOGGLE"] = True

    def get_state(self):
        return dict(self.state)


class ToggleStartPauseTest(unittest.TestCase):
    def test_first_press_starts(self):
        self.assertEqual(toggle_start_pause(False, False, False), (True, False, "started"))

    def test_second_press_pauses(self):
        self.assertEqual(toggle_start_pause(True, False, False), (True, True, "paused"))

    def test_third_press_resumes(self):
        self.assertEqual(toggle_start_pause(True, True, False), (True, False, "resumed"))

    def test_ignored_while_recording(self):
        self.assertEqual(toggle_start_pause(True, False, True), (True, False, "ignored_recording"))


class ControllerShortcutMapperTest(unittest.TestCase):
    def setUp(self):
        self.session = FakeSession()
        self.mapper = ControllerShortcutMapper(self.session.on_press, self.session.get_state)

    def test_ignores_buttons_until_motion_ready(self):
        fired = self.mapper.update(tele(motion_data_ready=False, right_ctrl_aButton=True))
        self.assertEqual(fired, [])
        self.assertEqual(self.session.keys, [])

    def test_right_a_starts_teleop_on_rising_edge(self):
        self.assertEqual(self.mapper.update(tele(right_ctrl_aButton=True)), ["r"])
        self.assertEqual(self.mapper.update(tele(right_ctrl_aButton=True)), [])
        self.assertEqual(self.session.keys, ["r"])
        self.assertTrue(self.session.state["START"])
        self.assertFalse(self.session.state["PAUSED"])

    def test_right_a_pauses_and_resumes_after_start(self):
        self.mapper.update(tele(right_ctrl_aButton=True))
        self.mapper.update(tele())
        self.assertEqual(self.mapper.update(tele(right_ctrl_aButton=True)), ["r"])
        self.assertTrue(self.session.state["PAUSED"])
        self.mapper.update(tele())
        self.assertEqual(self.mapper.update(tele(right_ctrl_aButton=True)), ["r"])
        self.assertFalse(self.session.state["PAUSED"])

    def test_right_a_ignored_while_recording(self):
        self.session.state["START"] = True
        self.session.state["RECORD_RUNNING"] = True
        self.assertEqual(self.mapper.update(tele(right_ctrl_aButton=True)), [])
        self.assertEqual(self.session.keys, [])
        self.assertFalse(self.session.state["PAUSED"])

    def test_left_y_toggles_record_after_start(self):
        self.mapper.update(tele(right_ctrl_aButton=True))
        self.mapper.update(tele())
        self.assertEqual(self.mapper.update(tele(left_ctrl_bButton=True)), ["s"])
        self.assertEqual(self.session.keys, ["r", "s"])

    def test_left_y_ignored_before_start(self):
        self.assertEqual(self.mapper.update(tele(left_ctrl_bButton=True)), [])
        self.assertEqual(self.session.keys, [])

    def test_left_y_ignored_while_paused(self):
        self.session.state["START"] = True
        self.session.state["PAUSED"] = True
        self.assertEqual(self.mapper.update(tele(left_ctrl_bButton=True)), [])
        self.assertEqual(self.session.keys, [])

    def test_left_y_can_stop_while_recorder_is_busy(self):
        self.session.state["START"] = True
        self.session.state["READY"] = False
        self.session.state["RECORD_RUNNING"] = True
        self.assertEqual(self.mapper.update(tele(left_ctrl_bButton=True)), ["s"])

    def test_left_y_ignored_while_toggle_pending(self):
        self.session.state["START"] = True
        self.session.state["RECORD_TOGGLE"] = True
        self.assertEqual(self.mapper.update(tele(left_ctrl_bButton=True)), [])

    def test_left_x_fail_stops_only_running_episode(self):
        self.session.state["START"] = True
        self.assertEqual(self.mapper.update(tele(left_ctrl_aButton=True)), [])
        self.mapper.update(tele())
        self.session.state["RECORD_RUNNING"] = True
        self.assertEqual(self.mapper.update(tele(left_ctrl_aButton=True)), ["f"])
        self.assertEqual(self.session.keys, ["f"])

    def test_right_b_requests_exit(self):
        self.assertEqual(self.mapper.update(tele(right_ctrl_bButton=True)), ["q"])
        self.assertTrue(self.session.state["STOP"])


if __name__ == "__main__":
    unittest.main()
