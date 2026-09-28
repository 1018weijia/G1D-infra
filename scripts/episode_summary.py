#!/usr/bin/env python3
"""Summarize Unitree JSON episodes before convert (frames, FAILED, joint steps)."""
from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from teleop.utils.trajectory_replay import TrajectoryError, load_trajectory, summarize


def iter_episode_dirs(task_dir):
    entries = []
    for name in sorted(os.listdir(task_dir)):
        path = os.path.join(task_dir, name)
        if os.path.isdir(path) and name.startswith("episode_"):
            entries.append(path)
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task_dir",
        help="Directory containing episode_XXXX folders",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=0,
        help="Limit how many episodes to print (0 = all)",
    )
    args = parser.parse_args()

    task_dir = os.path.expanduser(args.task_dir)
    if not os.path.isdir(task_dir):
        print(f"Not a directory: {task_dir}", file=sys.stderr)
        return 1

    episodes = iter_episode_dirs(task_dir)
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    if not episodes:
        print(f"No episode_* folders under {task_dir}")
        return 1

    failed = 0
    ok = 0
    for episode_dir in episodes:
        data_path = os.path.join(episode_dir, "data.json")
        has_intervention = os.path.isfile(os.path.join(episode_dir, "intervention.jsonl"))
        has_failed_marker = os.path.isfile(os.path.join(episode_dir, "FAILED"))
        print("=" * 60)
        print(os.path.basename(episode_dir))
        if not os.path.isfile(data_path):
            print("  missing data.json")
            failed += 1
            continue
        try:
            traj = load_trajectory(data_path)
            for line in summarize(traj):
                print(" ", line)
        except TrajectoryError as exc:
            print(f"  load error: {exc}")
            failed += 1
            continue
        if traj.success is False or has_failed_marker:
            failed += 1
            label = "FAILED"
        else:
            ok += 1
            label = "ok"
        print(f"  status  : {label}")
        print(f"  intervention.jsonl: {'yes' if has_intervention else 'no'}")

    print("=" * 60)
    print(f"total={len(episodes)} ok={ok} failed_or_error={failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
