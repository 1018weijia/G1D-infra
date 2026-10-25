#!/usr/bin/env python3
"""检测并结束正在控制 G1-D 机械臂的程序。

流程：
  1. 扫描 /proc，找出 calib_arm_server、policy_deploy、collect、replay 等控制入口
  2. 检查手臂所有权锁 /tmp/g1d_arm_owner.lock 是否被真实持有
  3. 在 DDS 总线上监听控制话题，看有没有数据在发布
  4. 报告结论；有控制者时询问 y/n，输入 y 则按正确信号停下它，再复检

退出码：0 无控制者 / 2 仍有控制者 / 3 用户选择不管 / 1 出错
"""
from __future__ import annotations

import argparse
import os
import signal
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from teleop.utils.arm_control_probe import (  # noqa: E402
    DEFAULT_INTERFACE,
    DEFAULT_PROBE_SECONDS,
    EXIT_BUSY,
    EXIT_DECLINED,
    EXIT_ERROR,
    EXIT_FREE,
    overall_verdict,
    pid_alive,
    probe_control_topics,
    read_lock_holder,
    lock_is_held,
    scan_controllers,
    send_stop_signal,
    wait_for_exit,
)
from teleop.utils.arm_owner_lock import DEFAULT_LOCK_PATH  # noqa: E402

STOP_TIMEOUT_SECONDS = 15.0
SIGNAL_NAMES = {signal.SIGINT: "SIGINT", signal.SIGTERM: "SIGTERM", signal.SIGKILL: "SIGKILL"}


def signal_name(sig: int) -> str:
    return SIGNAL_NAMES.get(sig, str(sig))


def describe_processes(processes) -> None:
    print("发现控制进程：")
    for proc in processes:
        caught = "/".join(sorted(signal_name(s) for s in proc.caught if s in (signal.SIGINT, signal.SIGTERM)))
        print(f"  PID {proc.pid}  [{proc.spec.name}]")
        print(f"    命令  : {proc.cmdline}")
        print(f"    捕获  : {caught or '未捕获 SIGINT/SIGTERM'}"
              f" -> 将发送 {signal_name(proc.stop_signal)}")
        print(f"    退出后: {proc.spec.stop_note}")


def describe_probe(probe) -> None:
    print()
    if probe.ok:
        print(f"DDS 探针正常（{probe.positive_count} 次收到 rt/lowstate 自检消息）")
    else:
        print(f"DDS 探针不可用：{probe.error}")
    active = probe.active_arm_topics
    quiet = [t for t in probe.traffic if t.arm and not t.active and not t.error]
    if active:
        print("手臂控制话题上有数据在发布：")
        for traffic in active:
            print(f"    {traffic.topic:<24} {traffic.count:>5} 条  ({traffic.label})")
    elif probe.ok:
        print("手臂控制话题上没有数据：")
        for traffic in quiet:
            print(f"    {traffic.topic:<24}       0 条  ({traffic.label})")
    errored = [t for t in probe.traffic if t.error]
    for traffic in errored:
        print(f"    {traffic.topic:<24}    订阅失败: {traffic.error}")
    others = probe.active_other_topics
    if others:
        print("注意，底盘/其他非手臂话题仍在发布（本脚本不处理）：")
        for traffic in others:
            print(f"    {traffic.topic:<24} {traffic.count:>5} 条  ({traffic.label})")


def ask_yes_no(prompt: str) -> bool:
    while True:
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            print()
            return False
        if answer in ("y", "yes", "是"):
            return True
        if answer in ("n", "no", "否", ""):
            return False
        print("请输入 y 或 n。")


def repoll(interface: str, seconds: float):
    """Re-run the full detection after stopping, so we trust the bus, not the kill()."""
    remaining = scan_controllers(exclude_pids={os.getpid()})
    probe = probe_control_topics(interface=interface, seconds=seconds)
    return overall_verdict(
        remaining, probe.ok, probe.active_arm_topics, lock_is_held(DEFAULT_LOCK_PATH)
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="检测并结束正在控制机械臂的程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-y", "--yes", action="store_true", help="不询问，发现有控制者就结束")
    parser.add_argument("-n", "--no", action="store_true", help="只检测，不结束任何东西")
    parser.add_argument("--probe-only", action="store_true", help="只报告，不打印 y/n 询问")
    parser.add_argument("--interface", default=os.environ.get("UNITREE_DDSINTERFACE", DEFAULT_INTERFACE),
                        help=f"DDS 网卡（默认 {DEFAULT_INTERFACE}）")
    parser.add_argument("--probe-seconds", type=float, default=DEFAULT_PROBE_SECONDS,
                        help=f"每个话题的监听时长（默认 {DEFAULT_PROBE_SECONDS}s）")
    parser.add_argument("--timeout", type=float, default=STOP_TIMEOUT_SECONDS,
                        help=f"等待进程退出的时长（默认 {STOP_TIMEOUT_SECONDS}s）")
    parser.add_argument("--force", action="store_true",
                        help="进程没在超时内退出时补发 SIGKILL（机械臂会立即失去支撑）")
    args = parser.parse_args()

    if args.yes and args.no:
        print("--yes 和 --no 不能同时使用。", file=sys.stderr)
        return EXIT_ERROR

    print("正在检测机械臂控制者 ...")
    processes = scan_controllers(exclude_pids={os.getpid()})
    probe = probe_control_topics(interface=args.interface, seconds=args.probe_seconds)
    lock_held = lock_is_held(DEFAULT_LOCK_PATH)
    verdict = overall_verdict(processes, probe.ok, probe.active_arm_topics, lock_held)

    if processes:
        describe_processes(processes)
    if lock_held:
        holder = read_lock_holder(DEFAULT_LOCK_PATH)
        print(f"手臂所有权锁被占用：{DEFAULT_LOCK_PATH}"
              + (f"（文件内记录：{holder}）" if holder else ""))
    describe_probe(probe)

    print()
    print(f"结论：{verdict.reason}")

    if verdict.state == "free":
        print("机械臂当前无人控制。")
        return EXIT_FREE

    if verdict.state == "unknown":
        print("无法确认状态，不做任何操作。请确认遥控器/其他控制端是否已断开。", file=sys.stderr)
        return EXIT_ERROR

    if not processes:
        # Busy without a matching process: something outside this repo owns the arms.
        print("没找到可结束的本机进程，因此不会自动处置。"
              "请检查是否有别的机器在通过 DDS 发布，或手动确认后再处理。", file=sys.stderr)
        return EXIT_BUSY

    if args.no or args.probe_only:
        return EXIT_BUSY

    if not args.yes:
        if not ask_yes_no("结束这些控制程序？机械臂会失去支撑并可能下垂 [y/N] "):
            print("未做任何改动。")
            return EXIT_DECLINED

    for proc in processes:
        if signal.SIGTERM not in proc.caught and signal.SIGINT not in proc.caught:
            print(f"警告：PID {proc.pid} 没有捕获 SIGINT/SIGTERM，"
                  "它可能来不及清理就退出，机械臂会立即失去支撑。")
        ok, message = send_stop_signal(proc.pid, proc.stop_signal)
        state = "已发送" if ok else "发送失败"
        print(f"PID {proc.pid}: {state} {signal_name(proc.stop_signal)}"
              + (f"（{message}）" if message else ""))

    print(f"等待进程退出（最多 {args.timeout:.0f}s）...")
    survivors = wait_for_exit([proc.pid for proc in processes], args.timeout)

    if survivors:
        print(f"以下进程没有在超时内退出：{survivors}")
        if args.force:
            for pid in survivors:
                ok, message = send_stop_signal(pid, signal.SIGKILL)
                print(f"PID {pid}: 已发送 SIGKILL" + (f"（{message}）" if message else ""))
            wait_for_exit(survivors, 5.0)
        else:
            print("未补发 SIGKILL。程序可能正在归位，重跑本脚本可再查一次；"
                  "确实需要强杀时加 --force。", file=sys.stderr)
            return EXIT_BUSY

    print()
    print("复检 ...")
    remaining = [proc for proc in processes if pid_alive(proc.pid)]
    if remaining:
        print(f"进程仍在：{[proc.pid for proc in remaining]}", file=sys.stderr)
        return EXIT_BUSY

    final = repoll(args.interface, args.probe_seconds)
    print(f"结论：{final.reason}")
    if final.state == "free":
        print("机械臂现在无人控制。")
        return EXIT_FREE
    if final.state == "unknown":
        return EXIT_ERROR
    return EXIT_BUSY


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已取消。", file=sys.stderr)
        sys.exit(EXIT_ERROR)
