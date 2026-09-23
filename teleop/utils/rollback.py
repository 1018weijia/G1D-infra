"""Bounded command history used for safe rollback playback."""
from collections import deque
import numpy as np

# Decelerate this much of the reverse path so the arm arrives at rest.
# The tail is time-stretched so several final commands sit on the endpoint.
ROLLBACK_EASE_SECONDS = 0.5
ROLLBACK_EASE_STRETCH = 3


class PolicyRollbackBuffer:
    """Store recent dual-arm and gripper commands for policy rollout rollback."""

    def __init__(self, seconds, frequency):
        self._items = deque(maxlen=max(1, int(np.ceil(seconds * frequency)) + 1))

    def append(self, arm_q, tau, left_grip, right_grip):
        self._items.append((
            np.asarray(arm_q, dtype=float).copy(),
            np.asarray(tau, dtype=float).copy(),
            float(left_grip),
            float(right_grip),
        ))

    def clear(self):
        self._items.clear()

    def reverse_playback(self, exclude_latest=True):
        items = list(self._items)
        if exclude_latest and items:
            items.pop()
        self.clear()
        return list(reversed(items))

    def __len__(self):
        return len(self._items)


def _copy_frame(frame):
    return (
        np.asarray(frame[0], dtype=float).copy(),
        np.asarray(frame[1], dtype=float).copy(),
        float(frame[2]),
        float(frame[3]),
    )


def _lerp_frame(start, end, alpha):
    alpha = float(alpha)
    return (
        np.asarray(start[0], dtype=float) + (np.asarray(end[0], dtype=float) - np.asarray(start[0], dtype=float)) * alpha,
        np.asarray(start[1], dtype=float) + (np.asarray(end[1], dtype=float) - np.asarray(start[1], dtype=float)) * alpha,
        float(start[2]) + (float(end[2]) - float(start[2])) * alpha,
        float(start[3]) + (float(end[3]) - float(start[3])) * alpha,
    )


def _stop_gain(progress):
    """Ease-out with h(0)=0, h(1)=1, h'(0)=3, h'(1)=0."""
    progress = float(np.clip(progress, 0.0, 1.0))
    return 1.0 - (1.0 - progress) ** 3


def ease_out_playback(frames, ease_steps, stretch=ROLLBACK_EASE_STRETCH):
    """Replay the reverse path, then decelerate onto the recorded endpoint.

    The head is unchanged. The tail is time-stretched by ``stretch`` and
    sampled with a zero-end-speed curve, so the last commands are nearly
    identical to the original final pose instead of stopping from cruise speed.
    """
    frames = list(frames)
    count = len(frames)
    if count <= 1 or int(ease_steps) < 2:
        return [_copy_frame(frame) for frame in frames]
    ease_steps = int(min(max(2, int(ease_steps)), count - 1))
    stretch = max(1, int(stretch))
    head = [_copy_frame(frames[index]) for index in range(count - ease_steps)]
    tail = frames[count - ease_steps - 1:]
    out_steps = max(len(tail), ease_steps * stretch)
    output = []
    for step in range(1, out_steps + 1):
        if step == out_steps:
            output.append(_copy_frame(frames[-1]))
            continue
        position = _stop_gain(step / out_steps) * (len(tail) - 1)
        index = int(np.floor(position))
        nxt = min(index + 1, len(tail) - 1)
        output.append(_lerp_frame(tail[index], tail[nxt], position - index))
    return head + output
