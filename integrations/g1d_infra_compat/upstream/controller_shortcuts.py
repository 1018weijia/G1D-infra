import logging_mp

logger_mp = logging_mp.getLogger(__name__)


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

    Pico / OpenXR mapping used by Vuer:
      right A -> right_ctrl_aButton  -> R  start, then pause/resume teleop
      left  Y -> left_ctrl_bButton   -> S  start or save episode
      left  X -> left_ctrl_aButton   -> F  stop episode and mark it failed
      right B -> right_ctrl_bButton  -> Q  exit
    """

    def __init__(self, on_press, get_state):
        if not callable(on_press):
            raise ValueError("on_press callback must be provided")
        if not callable(get_state):
            raise ValueError("get_state callback must be provided")
        self.on_press = on_press
        self.get_state = get_state
        self.previous = {
            "right_a": False,
            "right_b": False,
            "left_y": False,
            "left_x": False,
        }

    def update(self, tele_data):
        if not getattr(tele_data, "motion_data_ready", False):
            return []

        current = {
            "right_a": bool(getattr(tele_data, "right_ctrl_aButton", False)),
            "right_b": bool(getattr(tele_data, "right_ctrl_bButton", False)),
            "left_y": bool(getattr(tele_data, "left_ctrl_bButton", False)),
            "left_x": bool(getattr(tele_data, "left_ctrl_aButton", False)),
        }
        state = self.get_state() or {}
        start = bool(state.get("START", False))
        paused = bool(state.get("PAUSED", False))
        record_toggle = bool(state.get("RECORD_TOGGLE", False))
        record_running = bool(state.get("RECORD_RUNNING", False))
        ready = bool(state.get("READY", False))
        fired = []

        if current["right_a"] and not self.previous["right_a"]:
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

        if current["right_b"] and not self.previous["right_b"]:
            self.on_press("q")
            fired.append("q")
            logger_mp.info("Pico right B -> Q: exit requested")

        self.previous = current
        return fired
