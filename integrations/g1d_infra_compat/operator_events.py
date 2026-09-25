"""Pico button events shared by collection and policy deployment.

The mapping follows G1D-infra: right A starts/pauses/resumes, left Y toggles
recording, left X stops and marks the current episode failed, and right B exits.
The mapper emits rising edges only, so a held button cannot trigger two actions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class OperatorEvent(str, Enum):
    START_OR_TOGGLE_PAUSE = "start_or_toggle_pause"
    TOGGLE_RECORD = "toggle_record"
    MARK_FAILED = "mark_failed"
    QUIT = "quit"


@dataclass
class PicoEventMapper:
    previous: dict[str, bool] | None = None

    def __post_init__(self) -> None:
        if self.previous is None:
            self.previous = {"right_a": False, "right_b": False, "left_y": False, "left_x": False}

    def update(self, sample: Mapping[str, object]) -> list[OperatorEvent]:
        if not bool(sample.get("motion_data_ready", sample.get("motion_ready", False))):
            return []
        current = {
            "right_a": bool(sample.get("right_ctrl_aButton", sample.get("right_a", False))),
            "right_b": bool(sample.get("right_ctrl_bButton", sample.get("right_b", False))),
            "left_y": bool(sample.get("left_ctrl_bButton", sample.get("left_y", False))),
            "left_x": bool(sample.get("left_ctrl_aButton", sample.get("left_x", False))),
        }
        events = [
            event for key, event in (
                ("right_a", OperatorEvent.START_OR_TOGGLE_PAUSE),
                ("left_y", OperatorEvent.TOGGLE_RECORD),
                ("left_x", OperatorEvent.MARK_FAILED),
                ("right_b", OperatorEvent.QUIT),
            )
            if current[key] and not self.previous[key]
        ]
        self.previous = current
        return events
