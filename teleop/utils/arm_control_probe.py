"""Detect which processes are commanding the G1-D arms, and stop them politely.

The pure helpers here never touch DDS, hardware, or a TTY so tests stay
hardware-free. The DDS probe imports ``unitree_sdk2py`` lazily: on a machine
without the SDK the caller still gets process and lock-file detection.
"""
from __future__ import annotations

import contextlib
import fcntl
import importlib
import io
import os
import re
import signal
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from teleop.utils.arm_owner_lock import DEFAULT_LOCK_PATH

DEFAULT_INTERFACE = "eth0"
DEFAULT_PROBE_SECONDS = 2.0

# Exit codes shared with scripts/stop_arm_control.py
EXIT_FREE = 0
EXIT_ERROR = 1
EXIT_BUSY = 2
EXIT_DECLINED = 3


@dataclass(frozen=True)
class ControllerSpec:
    """A known entrypoint that can command the arms."""

    name: str
    patterns: Tuple[str, ...]
    stop_note: str


# Longest patterns first so "calib_arm_server" is not shadowed by a shorter stem.
CONTROLLER_SPECS: Tuple[ControllerSpec, ...] = (
    ControllerSpec(
        "calib_arm_server",
        ("calib_arm_server",),
        "收到 SIGTERM/SIGINT 后先回到启动姿态，再停止发布并退出",
    ),
    ControllerSpec(
        "calib_ready_pose",
        ("calib_ready_pose",),
        "退出时先释放手臂再停止发布",
    ),
    ControllerSpec(
        "align_calibrate",
        ("align_calibrate",),
        "退出时释放手臂并停止发布",
    ),
    ControllerSpec(
        "policy_deploy",
        ("policy_deploy",),
        "收到 SIGINT 后先 ctrl_dual_arm_go_home 回零位，再停止发布并退出",
    ),
    ControllerSpec(
        "collect",
        ("collect",),
        "收到 SIGINT 后先 ctrl_dual_arm_go_home 回零位，再停止发布并退出",
    ),
    ControllerSpec(
        "xr_teleop",
        ("teleop_hand_and_arm",),
        "未知清理逻辑：退出后机械臂可能立即失去支撑",
    ),
    ControllerSpec(
        "replay",
        ("teleop.replay", "replay"),
        "收到 SIGINT 后先回 home，再停止发布并退出",
    ),
)

# Anything matching these is only reported as context (a launcher shell), never targeted.
SHELL_HINTS = ("bash", "sh", "zsh", "dash")


@dataclass(frozen=True)
class TopicSpec:
    topic: str
    label: str
    idl_module: str
    idl_name: str
    arm: bool = True


_HG = "unitree_sdk2py.idl.unitree_hg.msg.dds_"
_GO = "unitree_sdk2py.idl.unitree_go.msg.dds_"
_GEOM = "unitree_sdk2py.idl.geometry_msgs.msg.dds_"

CONTROL_TOPICS: Tuple[TopicSpec, ...] = (
    TopicSpec("rt/lowcmd", "手臂/腰/腿 lowcmd", _HG, "LowCmd_", True),
    TopicSpec("rt/waistpitchcmd", "腰部俯仰", _HG, "MotorCmd_", True),
    TopicSpec("rt/brainco/left/cmd", "左手夹爪", _GO, "MotorCmds_", True),
    TopicSpec("rt/brainco/right/cmd", "右手夹爪", _GO, "MotorCmds_", True),
    TopicSpec("rt/cmd_vel_no_limit", "底盘速度", _GEOM, "Twist_", False),
)

# Always published by the robot itself; proves the probe path works. Without it a
# silent control topic could just mean the probe is broken, not that the arms are free.
POSITIVE_TOPIC = TopicSpec("rt/lowstate", "机器人状态(探针自检)", _HG, "LowState_", True)


@dataclass
class ControllerProcess:
    pid: int
    spec: ControllerSpec
    cmdline: str
    exe: str
    caught: FrozenSet[int] = field(default_factory=frozenset)
    stop_signal: int = signal.SIGINT

    @property
    def describes_graceful_exit(self) -> bool:
        return signal.SIGTERM in self.caught or signal.SIGINT in self.caught


@dataclass(frozen=True)
class TopicTraffic:
    topic: str
    label: str
    arm: bool
    count: int
    error: str = ""

    @property
    def active(self) -> bool:
        return self.count > 0


@dataclass
class ProbeResult:
    ok: bool = False
    positive_count: int = 0
    traffic: List[TopicTraffic] = field(default_factory=list)
    error: str = ""

    @property
    def active_arm_topics(self) -> List[TopicTraffic]:
        return [t for t in self.traffic if t.arm and t.active]

    @property
    def active_other_topics(self) -> List[TopicTraffic]:
        return [t for t in self.traffic if not t.arm and t.active]


# ---------------------------------------------------------------- token matching


def token_stems(token: str) -> Set[str]:
    """Return the forms of ``token`` we match on ("a/b/collect.py" -> {"collect.py", "collect"})."""
    base = os.path.basename(token)
    stems = {base}
    if base.endswith(".py"):
        stems.add(base[:-3])
    return stems


def classify_tokens(tokens: Sequence[str]) -> Optional[ControllerSpec]:
    """Pick the controller spec whose pattern appears in one cmdline token."""
    for token in tokens:
        stems = token_stems(token)
        for spec in CONTROLLER_SPECS:
            if stems.intersection(spec.patterns):
                return spec
    return None


def looks_like_python(exe: str, tokens: Sequence[str]) -> bool:
    if exe.startswith("python"):
        return True
    if tokens:
        head = os.path.basename(tokens[0])
        if head.startswith("python"):
            return True
    return False


# ------------------------------------------------------------------ /proc reading


def iter_proc_pids(root: str = "/proc") -> Iterable[int]:
    for entry in os.listdir(root):
        if entry.isdigit():
            yield int(entry)


def read_cmdline(pid: int, root: str = "/proc") -> List[str]:
    try:
        with open(f"{root}/{pid}/cmdline", "rb") as handle:
            raw = handle.read()
    except OSError:
        return []
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def read_exe(pid: int, root: str = "/proc") -> str:
    try:
        return os.path.basename(os.readlink(f"{root}/{pid}/exe"))
    except OSError:
        return ""


def signal_mask_to_set(mask: int) -> FrozenSet[int]:
    return frozenset(sig for sig in range(1, 65) if (mask >> (sig - 1)) & 1)


def read_caught_signals(pid: int, root: str = "/proc") -> FrozenSet[int]:
    try:
        with open(f"{root}/{pid}/status", "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return frozenset()
    match = re.search(r"^SigCgt:\s*([0-9a-fA-F]+)", text, re.MULTILINE)
    if match is None:
        return frozenset()
    return signal_mask_to_set(int(match.group(1), 16))


def choose_stop_signal(caught: FrozenSet[int]) -> int:
    """SIGTERM when the program registered one, else SIGINT (CPython always handles it)."""
    if signal.SIGTERM in caught:
        return signal.SIGTERM
    return signal.SIGINT


def scan_controllers(
    exclude_pids: Iterable[int] = (),
    root: str = "/proc",
) -> List[ControllerProcess]:
    """Find live python processes whose cmdline names a known arm controller."""
    excluded = set(exclude_pids)
    found: List[ControllerProcess] = []
    for pid in iter_proc_pids(root):
        if pid in excluded:
            continue
        tokens = read_cmdline(pid, root)
        if not tokens:
            continue
        spec = classify_tokens(tokens)
        if spec is None:
            continue
        exe = read_exe(pid, root)
        if not looks_like_python(exe, tokens):
            continue  # a launcher shell, or an editor holding the path in argv
        caught = read_caught_signals(pid, root)
        found.append(
            ControllerProcess(
                pid=pid,
                spec=spec,
                cmdline=" ".join(tokens),
                exe=exe or os.path.basename(tokens[0]),
                caught=caught,
                stop_signal=choose_stop_signal(caught),
            )
        )
    return sorted(found, key=lambda item: item.pid)


# ------------------------------------------------------------------------ locking


def read_lock_holder(path: str = DEFAULT_LOCK_PATH) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def lock_is_held(path: str = DEFAULT_LOCK_PATH) -> bool:
    """True only when a live process holds the flock.

    The file usually survives a crash with stale text like ``deploy pid=74645``,
    so the text alone must never be trusted.
    """
    try:
        with open(path, "a+", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return False
    except OSError:
        return False


# --------------------------------------------------------------------- verification


@dataclass(frozen=True)
class Verdict:
    state: str          # "busy" | "free" | "unknown"
    reason: str


def overall_verdict(
    processes: Sequence[ControllerProcess],
    probe_ok: bool,
    active_arm_topics: Sequence[TopicTraffic],
    lock_held: bool,
) -> Verdict:
    """Combine every signal into one verdict.

    A failed probe is never reported as "free": silence from a broken probe must
    not be mistaken for silence on the bus.
    """
    if processes:
        names = ", ".join(sorted({p.spec.name for p in processes}))
        return Verdict("busy", f"发现控制进程：{names}")
    if active_arm_topics:
        topics = ", ".join(t.topic for t in active_arm_topics)
        return Verdict("busy", f"手臂控制话题上仍有数据在发布：{topics}")
    if lock_held:
        return Verdict("busy", "手臂所有权锁被占用（未见进程，可能是别的环境里的控制者）")
    if not probe_ok:
        return Verdict("unknown", "DDS 探针不可用，无法确认总线是否安静")
    return Verdict("free", "没有发现控制进程，手臂控制话题上没有数据")


# ----------------------------------------------------------------------- DDS probe


def load_idl(spec: TopicSpec):
    module = importlib.import_module(spec.idl_module)
    return getattr(module, spec.idl_name)


@contextlib.contextmanager
def _swallow_stdout():
    """unitree_sdk2py prints on every empty read; keep the tool's output readable."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield


def probe_control_topics(
    interface: str = DEFAULT_INTERFACE,
    seconds: float = DEFAULT_PROBE_SECONDS,
    topics: Sequence[TopicSpec] = CONTROL_TOPICS,
) -> ProbeResult:
    """Count messages arriving on each control topic for ``seconds``.

    A topic with any sample means a writer is alive there. The positive control
    (``rt/lowstate``, published by the robot itself) tells us whether the probe
    path works at all.
    """
    result = ProbeResult()
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    except Exception as exc:  # noqa: BLE001 - report, never raise at the CLI
        result.error = f"无法导入 unitree_sdk2py: {exc}"
        return result

    with _swallow_stdout():
        try:
            ChannelFactoryInitialize(0, interface)
        except Exception as exc:  # noqa: BLE001
            result.error = f"DDS 初始化失败（接口 {interface}）: {exc}"
            return result

        subscribers: List[Tuple[TopicSpec, object]] = []
        errors: Dict[str, str] = {}
        for spec in list(topics) + [POSITIVE_TOPIC]:
            try:
                sub = ChannelSubscriber(spec.topic, load_idl(spec))
                sub.Init()
                subscribers.append((spec, sub))
            except Exception as exc:  # noqa: BLE001
                errors[spec.topic] = str(exc)

        counts: Dict[str, int] = {spec.topic: 0 for spec, _ in subscribers}
        deadline = time.time() + max(0.1, float(seconds))
        while time.time() < deadline:
            for spec, sub in subscribers:
                try:
                    if sub.Read(0.05) is not None:
                        counts[spec.topic] += 1
                except Exception as exc:  # noqa: BLE001
                    errors.setdefault(spec.topic, str(exc))

    result.positive_count = counts.get(POSITIVE_TOPIC.topic, 0)
    result.ok = result.positive_count > 0
    if not result.ok and not result.error:
        result.error = (
            f"在接口 {interface} 上收不到 {POSITIVE_TOPIC.topic}，"
            "说明探针本身没通，此时无法判断手臂是否被控制"
        )
    result.traffic = [
        TopicTraffic(
            topic=spec.topic,
            label=spec.label,
            arm=spec.arm,
            count=counts.get(spec.topic, 0),
            error=errors.get(spec.topic, ""),
        )
        for spec in topics
    ]
    return result


# ------------------------------------------------------------------------ stopping


def send_stop_signal(pid: int, sig: int) -> Tuple[bool, str]:
    try:
        os.kill(pid, sig)
        return True, ""
    except ProcessLookupError:
        return True, "进程已不在"
    except PermissionError as exc:
        return False, f"权限不足: {exc}"
    except OSError as exc:
        return False, str(exc)


def wait_for_exit(pids: Iterable[int], timeout: float, poll: float = 0.2) -> List[int]:
    """Return the pids still alive after ``timeout`` seconds."""
    remaining = [pid for pid in pids if pid_alive(pid)]
    deadline = time.time() + max(0.0, float(timeout))
    while remaining and time.time() < deadline:
        time.sleep(poll)
        remaining = [pid for pid in remaining if pid_alive(pid)]
    return remaining


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
