"""Non-blocking voice announcements for operator state changes.

No audio dependency is required. The first available backend is selected from
``espeak-ng``, ``spd-say`` and ``say``; if none exists, messages are logged and
remain available through the callback for a UI or robot audio service.
"""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading
from collections.abc import Callable

log = logging.getLogger(__name__)


class VoiceAnnouncer:
    MESSAGES = {
        "record_started": "开始录制",
        "record_saved": "录制已保存",
        "trajectory_failed": "轨迹失败，已标记",
        "policy_started": "策略开始执行",
        "takeover_ready": "可以接管",
        "takeover_active": "已接管",
        "policy_resumed": "策略已恢复",
        "tracking_lost": "跟踪丢失，机械臂保持安全姿态",
        "quit": "退出",
    }

    def __init__(self, *, enabled: bool = True, sink: Callable[[str], None] | None = None):
        self.enabled = enabled
        self.sink = sink
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._backend = next((shutil.which(x) for x in ("espeak-ng", "spd-say", "say") if shutil.which(x)), None)
        self._thread = threading.Thread(target=self._worker, name="g1d-voice", daemon=True)
        self._thread.start()

    @property
    def backend(self) -> str | None:
        return self._backend

    def announce(self, event: str, text: str | None = None) -> None:
        message = text or self.MESSAGES.get(event, event)
        log.info("VOICE %s", message)
        if self.sink:
            self.sink(message)
        if self.enabled:
            self._queue.put(message)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=2.0)

    def _worker(self) -> None:
        while True:
            message = self._queue.get()
            if message is None:
                return
            if not self._backend:
                continue
            try:
                if self._backend.endswith("say") and not self._backend.endswith("espeak-ng"):
                    subprocess.run([self._backend, message], check=False, timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    subprocess.run([self._backend, "-v", "zh", message], check=False, timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except (OSError, subprocess.TimeoutExpired) as exc:
                log.warning("voice backend failed: %s", exc)
