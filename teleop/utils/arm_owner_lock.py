"""Exclusive ownership lock so collect and policy_deploy never both own the arms."""
from __future__ import annotations

import fcntl
import os
from typing import Optional

DEFAULT_LOCK_PATH = "/tmp/g1d_arm_owner.lock"


class ArmOwnerLockError(RuntimeError):
    pass


class ArmOwnerLock:
    """fcntl flock wrapper; keep the instance alive for the process lifetime."""

    def __init__(self, path: str, owner: str):
        self.path = path
        self.owner = owner
        self._fh = None

    def acquire(self) -> "ArmOwnerLock":
        os.makedirs(os.path.dirname(self.path) or "/tmp", exist_ok=True)
        self._fh = open(self.path, "a+", encoding="utf-8")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            holder = self._read_holder()
            self._fh.close()
            self._fh = None
            detail = f" (held by {holder})" if holder else ""
            raise ArmOwnerLockError(
                f"Another process already owns the arms{detail}. "
                f"Stop it, or pass --force to bypass (dangerous)."
            ) from exc
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(f"{self.owner} pid={os.getpid()}\n")
        self._fh.flush()
        return self

    def _read_holder(self) -> str:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                return handle.read().strip()
        except OSError:
            return ""

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "ArmOwnerLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def acquire_arm_owner_lock(
        owner: str, path: Optional[str] = None) -> ArmOwnerLock:
    """Acquire exclusive arm ownership for ``owner`` (e.g. collect / deploy)."""
    return ArmOwnerLock(path or DEFAULT_LOCK_PATH, owner).acquire()
