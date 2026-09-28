"""Append-only intervention event log for policy rollout recordings."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional


class InterventionLogger:
    """Writes JSONL events next to an episode directory."""

    def __init__(self, episode_dir: Optional[str] = None):
        self.episode_dir = episode_dir
        self.path = (
            os.path.join(episode_dir, "intervention.jsonl") if episode_dir else None
        )
        self._enabled = bool(self.path)

    def bind(self, episode_dir: Optional[str]) -> None:
        self.episode_dir = episode_dir
        self.path = (
            os.path.join(episode_dir, "intervention.jsonl") if episode_dir else None
        )
        self._enabled = bool(self.path)

    def clear(self) -> None:
        self.bind(None)

    def log(self, event: str, phase: str = "", frame_idx: Optional[int] = None,
            **extra: Any) -> None:
        if not self._enabled or not self.path:
            return
        payload = {
            "t": time.time(),
            "event": event,
            "phase": phase,
        }
        if frame_idx is not None:
            payload["frame_idx"] = int(frame_idx)
        payload.update(extra)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
