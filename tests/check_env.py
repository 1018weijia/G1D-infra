#!/usr/bin/env python3
"""Offline environment self-check for g1d_infra.

Reports one PASS/FAIL line per item and exits non-zero if anything required is
missing. Nothing here touches the robot, the cameras, or the network.

Imports run in separate interpreters: on aarch64 some native extensions
(pinocchio/libgomp) only load cleanly when they come first, so checking them
in one shared process would report failures the real entrypoints never hit.
"""
import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUNTIME_MODULES = [
    "numpy", "cv2", "yaml", "zmq", "sshkeyboard", "rerun", "logging_mp",
    "unitree_sdk2py", "vuer", "pinocchio", "casadi",
]

REPO_MODULES = [
    "teleop.utils.episode_writer",
    "teleop.utils.policy_client",
    "teleop.utils.policy_handoff",
    "teleop.utils.rollback",
    "teleop.utils.alignment",
    "teleop.utils.handoff_utils",
    "teleop.utils.ready_pose",
    "teleop.utils.trajectory_replay",
    "teleop.utils.dex1_arm_bundle",
    "teleop.replay",
    "teleop.robot_control.robot_arm_ik",
    "teleop.televuer.tv_wrapper",
]

REQUIRED_PATHS = [
    "collect.py",
    "policy_deploy.py",
    "teleop/replay.py",
    "configs/infer_g1d.yaml",
    "configs/alignment_targets.json",
    "configs/ready_pose.json",
    "assets/g1_D/g1_d.urdf",
    "data_convert/convert.sh",
    "data_convert/upload.sh",
    "3rd/lerobot/src/lerobot",
]

CONVERT_MODULES = ["huggingface_hub", "datasets", "av", "torch"]


class Report:
    def __init__(self):
        self.failures = 0

    def check(self, label, ok, detail=""):
        if not ok:
            self.failures += 1
        suffix = f"  ({detail})" if detail and not ok else ""
        print(f"[{'PASS' if ok else 'FAIL'}] {label}{suffix}")

    def warn(self, label, detail=""):
        suffix = f"  ({detail})" if detail else ""
        print(f"[WARN] {label}{suffix}")


def last_line(text):
    lines = [line for line in (text or "").strip().splitlines() if line.strip()]
    return lines[-1][:140] if lines else ""


def import_in_subprocess(name, timeout=180):
    proc = subprocess.run(
        [sys.executable, "-c", f"import {name}"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
    )
    return proc.returncode == 0, last_line(proc.stderr)


def help_runs(args, timeout=180):
    proc = subprocess.run(
        [sys.executable, *args, "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
    )
    return proc.returncode == 0, last_line(proc.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--convert-python", default=os.path.expanduser(
        "~/miniconda3/envs/unitree_lerobot/bin/python"))
    parser.add_argument("--skip-imports", action="store_true",
                        help="only check files and entrypoints")
    args = parser.parse_args()

    report = Report()
    print(f"repo:   {REPO_ROOT}")
    print(f"python: {sys.executable} ({sys.version.split()[0]})\n")

    print("-- files --")
    for rel in REQUIRED_PATHS:
        report.check(rel, os.path.exists(os.path.join(REPO_ROOT, rel)))

    if not args.skip_imports:
        print("\n-- runtime dependencies --")
        for name in RUNTIME_MODULES:
            report.check(name, *import_in_subprocess(name))

        print("\n-- repo modules --")
        for name in REPO_MODULES:
            report.check(name, *import_in_subprocess(name))

    print("\n-- entrypoints --")
    for label, argv in (
        ("collect.py", ["collect.py"]),
        ("policy_deploy.py", ["policy_deploy.py"]),
        ("teleop.replay", ["-m", "teleop.replay"]),
    ):
        report.check(f"{label} --help", *help_runs(argv))

    print("\n-- conversion environment (optional on the robot) --")
    if not os.path.exists(args.convert_python):
        report.warn(f"python not found: {args.convert_python}",
                    "only needed for data_convert/")
    else:
        proc = subprocess.run(
            [args.convert_python, "-c", "import " + ", ".join(CONVERT_MODULES)],
            capture_output=True, text=True,
        )
        if proc.returncode == 0:
            print(f"[PASS] {args.convert_python}: {', '.join(CONVERT_MODULES)}")
        else:
            report.warn(f"{args.convert_python} missing LeRobot deps",
                        last_line(proc.stderr))

    print()
    if report.failures:
        print(f"{report.failures} required check(s) failed")
        return 1
    print("all required checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
