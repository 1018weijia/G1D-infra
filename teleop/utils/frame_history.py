"""Opt-in client-side frame history for policies that need long-span context.

The buffer is sampled at a fixed rate from the moment a policy rollout starts,
writes one stitched JPEG plus machine state per sample to local disk, and hands
back only the subset an inference request asks for. Disabled instances create no
directory and mutate no request, so the existing inference path stays identical.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

DEFAULT_ROOT = "/tmp/g1d_frame_history"
# Timestamps that land exactly on the sample period (3.6 - 3.4) fall just under
# it in float arithmetic; this keeps those samples instead of dropping them.
_TIME_EPS = 1e-9


@dataclass(frozen=True)
class HistoryConfig:
    """Client-side settings for the history buffer and for what gets sent."""

    enabled: bool = False
    sample_hz: float = 5.0
    send: bool = False
    interval_seconds: float = 0.5
    count: int = 16
    include_first: bool = True
    include_rollback: bool = False
    jpeg_quality: int = 90
    max_samples: int = 1800

    @property
    def period(self) -> float:
        return 1.0 / self.sample_hz


def _as_bool(key, value):
    if isinstance(value, bool):
        return value
    raise ValueError(f"history.{key} must be a boolean, got {value!r}")


def _as_float(key, value):
    if isinstance(value, bool):
        raise ValueError(f"history.{key} must be a number, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"history.{key} must be a number, got {value!r}") from None


def _as_int(key, value):
    if isinstance(value, bool):
        raise ValueError(f"history.{key} must be an integer, got {value!r}")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"history.{key} must be an integer, got {value!r}") from None
    if number != float(value):
        raise ValueError(f"history.{key} must be an integer, got {value!r}")
    return number


def parse_history_config(raw) -> HistoryConfig:
    """Validate the optional ``history:`` mapping; absent means fully disabled."""
    defaults = HistoryConfig()
    if raw is None:
        return defaults
    if not isinstance(raw, dict):
        raise ValueError(f"history must be a mapping, got {type(raw).__name__}")
    known = set(defaults.__dataclass_fields__)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"history has unknown keys {unknown}; expected {sorted(known)}")
    config = HistoryConfig(
        enabled=_as_bool("enabled", raw.get("enabled", defaults.enabled)),
        sample_hz=_as_float("sample_hz", raw.get("sample_hz", defaults.sample_hz)),
        send=_as_bool("send", raw.get("send", defaults.send)),
        interval_seconds=_as_float(
            "interval_seconds", raw.get("interval_seconds", defaults.interval_seconds)
        ),
        count=_as_int("count", raw.get("count", defaults.count)),
        include_first=_as_bool(
            "include_first", raw.get("include_first", defaults.include_first)
        ),
        include_rollback=_as_bool(
            "include_rollback", raw.get("include_rollback", defaults.include_rollback)
        ),
        jpeg_quality=_as_int("jpeg_quality", raw.get("jpeg_quality", defaults.jpeg_quality)),
        max_samples=_as_int("max_samples", raw.get("max_samples", defaults.max_samples)),
    )
    if config.sample_hz <= 0.0:
        raise ValueError(f"history.sample_hz must be positive, got {config.sample_hz!r}")
    if config.interval_seconds < 0.0:
        raise ValueError(
            f"history.interval_seconds must be non-negative, got {config.interval_seconds!r}"
        )
    if config.count < 0:
        raise ValueError(f"history.count must be non-negative, got {config.count!r}")
    if not 1 <= config.jpeg_quality <= 100:
        raise ValueError(
            f"history.jpeg_quality must be between 1 and 100, got {config.jpeg_quality!r}"
        )
    if config.send and not config.enabled:
        raise ValueError("history.send requires history.enabled: true")
    if config.max_samples < 2:
        raise ValueError(
            f"history.max_samples must be at least 2, got {config.max_samples!r}"
        )
    return config


def load_history_config(config_path: str) -> HistoryConfig:
    """Read the ``history:`` block out of an inference config file."""
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return parse_history_config(data.get("history"))


@dataclass(frozen=True)
class HistorySample:
    """One captured sample: metadata in memory, JPEG bytes loaded on demand."""

    n: int
    t_rel: float
    rollback: bool
    state: np.ndarray
    jpeg: bytes = b""

    def as_request_tuple(self):
        return (self.n, self.t_rel, self.jpeg, self.state)


class FrameHistory:
    """Fixed-rate, disk-backed ring buffer of stitched frames.

    Sampling is throttled by the caller's timestamps; JPEG encoding and the
    index append happen on one background thread with a bounded queue so a slow
    disk drops samples instead of stalling the 30 Hz control loop. The buffer
    keeps at most ``max_samples`` samples, always retaining the run's first one,
    so a long rollout cannot fill the disk.
    """

    def __init__(self, root: Optional[str] = None, sample_hz: float = 5.0,
                 enabled: bool = True, jpeg_quality: int = 90,
                 max_pending: int = 48, max_samples: int = 1800):
        self.enabled = bool(enabled)
        self.sample_hz = max(1e-3, float(sample_hz))
        self.jpeg_quality = int(jpeg_quality)
        self.max_samples = max(2, int(max_samples))
        self._root = root or DEFAULT_ROOT
        self._max_pending = max(1, int(max_pending))
        self._index = deque()
        self._queue = queue.Queue(maxsize=self._max_pending)
        self._thread: Optional[threading.Thread] = None
        self._index_handle = None
        self._dir: Optional[str] = None
        self._t0: Optional[float] = None
        self._started = False
        self._last_capture: Optional[float] = None
        self._next_n = 0
        self._dropped = 0
        self._run_count = 0
        self._evicted = set()
        self._discarded = False
        self._lock = threading.RLock()

    # -- introspection -----------------------------------------------------
    @property
    def period(self) -> float:
        return 1.0 / self.sample_hz

    @property
    def directory(self) -> Optional[str]:
        return self._dir

    @property
    def started(self) -> bool:
        return self._started

    def __len__(self) -> int:
        return len(self._index)

    # -- lifecycle ---------------------------------------------------------
    def start(self, now: Optional[float] = None):
        """Begin (or resume) sampling. Resuming keeps the run's directory."""
        if not self.enabled or self._discarded:
            return
        with self._lock:
            if self._started:
                return
            if self._dir is None:
                self._dir = self._new_run_dir()
                os.makedirs(self._dir, exist_ok=True)
                self._index_handle = open(
                    os.path.join(self._dir, "index.jsonl"), "w", encoding="utf-8"
                )
            if self._t0 is None:
                self._t0 = time.monotonic() if now is None else float(now)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._writer_loop, name="frame-history-writer", daemon=True
                )
                self._thread.start()
            self._last_capture = None
            self._started = True

    def stop(self):
        """Pause sampling without dropping what was captured."""
        with self._lock:
            self._started = False

    def due(self, now: Optional[float] = None) -> bool:
        """True when a capture now would pass the sample-rate throttle."""
        if not (self.enabled and self._started):
            return False
        now = time.monotonic() if now is None else float(now)
        return self._throttle_ok(now)

    def _throttle_ok(self, now: float) -> bool:
        if self._last_capture is None:
            return True
        # Timestamps that land exactly on the sample period must not be dropped
        # by float subtraction error (3.6 - 3.4 is slightly under 0.2).
        return (now - self._last_capture) >= self.period - _TIME_EPS

    def flush(self):
        """Wait for queued samples to reach disk. Tests and shutdown only."""
        if self._thread is None:
            return
        self._queue.join()

    def discard(self):
        """Delete the run's directory. Terminates the instance; safe to repeat."""
        with self._lock:
            self._discarded = True
            self.enabled = False
            self._started = False
            thread = self._thread
            self._thread = None
            handle = self._index_handle
            self._index_handle = None
            directory = self._dir
            self._dir = None
            self._index.clear()
            self._evicted.clear()
            self._t0 = None
            self._last_capture = None
            self._next_n = 0
        if thread is not None:
            # Drop anything still queued: those JPEGs would be written and then
            # immediately deleted, so draining is faster than encoding them.
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._queue.task_done()
            try:
                self._queue.put(None, timeout=1.0)
            except queue.Full:
                pass
            thread.join(timeout=2.0)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
        if directory:
            shutil.rmtree(directory, ignore_errors=True)

    def _evict_overflow(self):
        """Drop the oldest non-first samples once the ring is full.

        The run's first sample is kept so ``include_first`` always has its frame.
        The JPEG is unlinked immediately when it is already on disk; when it is
        still queued for the writer, the n is remembered so the writer removes it
        right after writing instead of leaving an orphan behind.
        """
        while len(self._index) > self.max_samples:
            if self._index[0].n == 0 and len(self._index) > 1:
                victim = self._index[1]
                del self._index[1]
            else:
                victim = self._index.popleft()
            self._evicted.add(victim.n)
            try:
                os.unlink(self._path(victim.n))
            except OSError:
                # Not written yet; the writer will skip it after writing.
                pass
        if len(self._evicted) > 4 * self.max_samples:
            # Keep the guard set small on very long rollouts. The writer only
            # consults it for samples still in flight, which is a tiny window.
            newest = self._next_n - self._max_pending - 1
            self._evicted = {n for n in self._evicted if n >= newest}

    # -- capture -----------------------------------------------------------
    def capture(self, image, state, now: Optional[float] = None,
                rolling_back: bool = False) -> Optional[HistorySample]:
        """Store one sample if the throttle allows; returns it or None."""
        if not (self.enabled and self._started):
            return None
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            if not self._throttle_ok(now):
                return None
            self._last_capture = now
            t0 = self._t0 if self._t0 is not None else now
            sample = HistorySample(
                n=self._next_n,
                t_rel=float(now - t0),
                rollback=bool(rolling_back),
                state=np.asarray(state, dtype=np.float32).reshape(-1)[:16].copy(),
            )
            self._next_n += 1
            frame = np.array(image, dtype=np.uint8, copy=True)
            record = {
                "n": sample.n,
                "t": float(now),
                "t_rel": sample.t_rel,
                "rollback": sample.rollback,
                "state": [float(v) for v in sample.state],
            }
            try:
                self._queue.put_nowait((record, self._path(sample.n), frame))
            except queue.Full:
                # No JPEG will ever exist for this sample, so keep it out of the
                # index instead of holding a ring slot select can never use.
                self._dropped += 1
                if self._dropped == 1:
                    logger.warning(
                        "Frame history writer is behind; dropping samples instead of "
                        "stalling the control loop (dropped=%d)",
                        self._dropped,
                    )
                return None
            self._index.append(sample)
            self._evict_overflow()
            return sample

    def _writer_loop(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                record, path, frame = item
                if not cv2.imwrite(
                        path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]):
                    logger.warning("Frame history failed to write %s", path)
                    continue
                self._append_index(record)
                with self._lock:
                    evicted = record["n"] in self._evicted
                    self._evicted.discard(record["n"])
                if evicted:
                    # The ring dropped this sample while its JPEG was in flight;
                    # unlink now rather than leave an unreferenced file behind.
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
            except Exception as error:
                logger.warning("Frame history writer error: %s", error)
            finally:
                self._queue.task_done()

    def _append_index(self, record):
        with self._lock:
            handle = self._index_handle
            if handle is None:
                return
            handle.write(json.dumps(record) + "\n")
            handle.flush()

    def _path(self, n: int) -> str:
        return os.path.join(self._dir or "", f"{int(n):06d}.jpg")

    def _new_run_dir(self) -> str:
        """Unique per-run directory so a leftover run cannot poison the next one."""
        base = f"{os.getpid()}_{int(time.time() * 1000)}"
        self._run_count += 1
        if self._run_count > 1:
            base = f"{base}_{self._run_count}"
        candidate = os.path.join(self._root, base)
        suffix = 0
        while os.path.exists(candidate):
            suffix += 1
            candidate = os.path.join(self._root, f"{base}_{suffix}")
        return candidate

    def _read_jpeg(self, n: int) -> Optional[bytes]:
        try:
            with open(self._path(n), "rb") as handle:
                return handle.read()
        except OSError:
            return None

    # -- selection ---------------------------------------------------------
    def select(self, interval_seconds: float, count: int,
               include_first: bool = True,
               include_rollback: bool = False) -> list:
        """Pick at most ``count`` samples walking back from the newest one.

        Samples are spaced by at least ``interval_seconds`` of ``t_rel``. When
        ``include_first`` is set, one slot is reserved so the run's first sample
        can be appended last and the policy sees the task's starting pose.
        Frames whose JPEG has not reached disk yet are skipped: this never pads
        or duplicates, which the model would read as real observations.
        """
        if not self.enabled or count <= 0:
            return []
        with self._lock:
            # Snapshot: capture runs on the same thread today, but the writer
            # and a future caller should not be able to mutate the deque here.
            index = list(self._index)
        first = index[0] if index else None
        want_first = bool(
            include_first and first is not None and first.n == 0
            and (include_rollback or not first.rollback)
        )
        budget = count - 1 if want_first else count
        interval = max(0.0, float(interval_seconds))
        picked = []
        picked_n = set()
        previous_t = None
        for sample in reversed(index):
            if len(picked) >= budget:
                break
            if not include_rollback and sample.rollback:
                continue
            if previous_t is not None and (previous_t - sample.t_rel) < interval - _TIME_EPS:
                continue
            jpeg = self._read_jpeg(sample.n)
            if jpeg is None:
                continue
            picked.append(HistorySample(
                n=sample.n, t_rel=sample.t_rel, rollback=sample.rollback,
                state=sample.state, jpeg=jpeg,
            ))
            picked_n.add(sample.n)
            previous_t = sample.t_rel
        if want_first and first.n not in picked_n:
            jpeg = self._read_jpeg(first.n)
            if jpeg is not None:
                picked.append(HistorySample(
                    n=first.n, t_rel=first.t_rel, rollback=first.rollback,
                    state=first.state, jpeg=jpeg,
                ))
        return picked
