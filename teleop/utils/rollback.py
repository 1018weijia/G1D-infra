"""Bounded command history used for safe rollback playback."""
from collections import deque
import numpy as np


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


class RollbackBuffer:
    def __init__(self, seconds, frequency):
        self._items = deque(maxlen=max(1, int(np.ceil(seconds * frequency)) + 1))

    def append(self, q, tau):
        self._items.append((np.asarray(q, dtype=float).copy(), np.asarray(tau, dtype=float).copy()))

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
