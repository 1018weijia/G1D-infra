import logging_mp

logger_mp = logging_mp.getLogger(__name__)

MODE_ARM = "arm"
MODE_BODY = "body"


def toggle_start_pause(start, paused, record_running):
    """Apply one R / right-A press to the start/pause state.

    Returns (start, paused, action) where action is:
      started, paused, resumed, or ignored_recording.
    """
    if not start:
        return True, False, "started"
    if record_running:
        return start, paused, "ignored_recording"
    paused = not paused
    return start, paused, "paused" if paused else "resumed"


class ControllerShortcutMapper:
    """Map Pico face-button rising edges to R/S/F/Q state-machine actions.

    Pico / OpenXR mapping used by Vuer, arm mode (default):
      right A -> right_ctrl_aButton  -> R  start, then pause/resume teleop
      left  Y -> left_ctrl_bButton   -> S  start or save episode
      left  X -> left_ctrl_aButton   -> F  stop episode and mark it failed
      right B -> right_ctrl_bButton  -> Q  exit

    With waist_enabled, clicking the right thumbstick (the digital press, not
    a tilt) toggles body mode. In body mode A/B are held waist-pitch buttons
    and do not fire R/Q. Left Y/X stay session keys in both modes.
    """

    def __init__(self, on_press, get_state, waist_enabled=False):
        if not callable(on_press):
            raise ValueError("on_press callback must be provided")
        if not callable(get_state):
            raise ValueError("get_state callback must be provided")
        self.on_press = on_press
        self.get_state = get_state
        self.waist_enabled = bool(waist_enabled)
        self.mode = MODE_ARM
        self.previous = {
            "right_a": False,
            "right_b": False,
            "left_y": False,
            "left_x": False,
            "right_stick": False,
        }

    def update(self, tele_data):
        if not getattr(tele_data, "motion_data_ready", False):
            return []

        current = {
            "right_a": bool(getattr(tele_data, "right_ctrl_aButton", False)),
            "right_b": bool(getattr(tele_data, "right_ctrl_bButton", False)),
            "left_y": bool(getattr(tele_data, "left_ctrl_bButton", False)),
            "left_x": bool(getattr(tele_data, "left_ctrl_aButton", False)),
            "right_stick": bool(getattr(tele_data, "right_ctrl_thumbstick", False)),
        }
        if self.waist_enabled and current["right_stick"] and not self.previous["right_stick"]:
            self.mode = MODE_BODY if self.mode == MODE_ARM else MODE_ARM
            if self.mode == MODE_BODY:
                logger_mp.info(
                    "Pico right stick click -> body mode: hold B to bend, A to straighten, stick left/right to yaw"
                )
            else:
                logger_mp.info("Pico right stick click -> arm mode: waist held; A/B are session keys again")
        body_mode = self.mode == MODE_BODY
        state = self.get_state() or {}
        start = bool(state.get("START", False))
        paused = bool(state.get("PAUSED", False))
        record_toggle = bool(state.get("RECORD_TOGGLE", False))
        record_running = bool(state.get("RECORD_RUNNING", False))
        ready = bool(state.get("READY", False))
        fired = []

        if current["right_a"] and not self.previous["right_a"] and not body_mode:
            if record_running:
                logger_mp.info("Pico right A ignored: stop recording before pausing teleop")
            else:
                self.on_press("r")
                fired.append("r")
                _, _, action = toggle_start_pause(start, paused, record_running)
                if action == "started":
                    logger_mp.info("Pico right A -> R: teleoperation started")
                elif action == "paused":
                    logger_mp.info("Pico right A -> R: teleoperation paused for scene reset")
                else:
                    logger_mp.info("Pico right A -> R: teleoperation resumed")

        if current["left_y"] and not self.previous["left_y"]:
            # READY is false while an episode is open. A stop request must
            # therefore be accepted while RECORD_RUNNING is true; READY only
            # gates creation of the next episode.
            if paused and not record_running:
                logger_mp.info("Pico left Y ignored: press right A to resume teleop first")
            elif start and not record_toggle and (record_running or ready):
                self.on_press("s")
                fired.append("s")
                logger_mp.info("Pico left Y -> S: recording toggled")
            elif start:
                logger_mp.info("Pico left Y ignored: recorder is busy")
            else:
                logger_mp.info("Pico left Y ignored: press right A to start first")

        if current["left_x"] and not self.previous["left_x"]:
            if start and record_running and not record_toggle:
                self.on_press("f")
                fired.append("f")
                logger_mp.info("Pico left X -> F: episode marked failed and stopped")
            else:
                logger_mp.info("Pico left X ignored: no running episode to discard")

        if current["right_b"] and not self.previous["right_b"] and not body_mode:
            self.on_press("q")
            fired.append("q")
            logger_mp.info("Pico right B -> Q: exit requested")

        self.previous = current
        return fired
