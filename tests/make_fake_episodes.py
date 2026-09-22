#!/usr/bin/env python3
"""Write synthetic episodes in collect.py's on-disk format.

Used by TESTING.md to exercise data_convert/ without a robot. The layout comes
from EpisodeWriter itself, so it stays in sync with what collect.py records.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from teleop.utils.episode_writer import EpisodeWriter  # noqa: E402

CAMERAS = ["color_0", "color_1", "color_2", "color_3"]


def arm_block(values):
    return {"qpos": [float(v) for v in values], "qvel": [], "torque": []}


def frame_payload(step, total):
    phase = 2.0 * np.pi * step / max(total, 1)
    arm = 0.3 * np.sin(phase + np.arange(7) * 0.1)
    return (
        {
            "left_arm": arm_block(arm),
            "right_arm": arm_block(-arm),
            "left_arm_pose": arm_block(np.zeros(6)),
            "right_arm_pose": arm_block(np.zeros(6)),
            "left_ee": arm_block([0.5 + 0.4 * np.sin(phase)]),
            "right_ee": arm_block([0.5 - 0.4 * np.sin(phase)]),
            "body": {"qpos": []},
            "torso": {"height": [0.1 * np.sin(phase)], "qvel": [0.0]},
            "chassis": {"qvel": [0.0, 0.0]},
        },
        {
            "left_arm": arm_block(arm * 1.01),
            "right_arm": arm_block(-arm * 1.01),
            "left_arm_pose": arm_block(np.zeros(6)),
            "right_arm_pose": arm_block(np.zeros(6)),
            "left_ee": arm_block([0.5 + 0.4 * np.sin(phase)]),
            "right_ee": arm_block([0.5 - 0.4 * np.sin(phase)]),
            "body": {"qpos": []},
            "torso": {"qvel": [0.0]},
            "chassis": {"qvel": [0.0, 0.0]},
        },
    )


def fake_image(step, camera_idx, width, height):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, camera_idx % 3] = (step * 7) % 256
    img[height // 4:height // 2, width // 4:width // 2] = 255
    return img


def wait_ready(recorder, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if recorder.is_ready():
            return
        time.sleep(0.02)
    raise TimeoutError("EpisodeWriter did not finish saving")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", default="/tmp/g1d_fake/fake_task")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fail", default="", help="episode indices to mark failed, e.g. 1 or 0,2")
    args = parser.parse_args()

    failed = {int(x) for x in args.fail.replace(",", " ").split()}
    task_dir = os.path.expanduser(args.task_dir)

    recorder = EpisodeWriter(
        task_dir=task_dir,
        task_goal="fake pick and place",
        task_desc="synthetic data for pipeline testing",
        image_size=[args.width, args.height],
        rerun_log=False,
    )
    try:
        for episode in range(args.episodes):
            if not recorder.create_episode():
                raise RuntimeError("create_episode refused; writer is busy")
            for step in range(args.frames):
                states, actions = frame_payload(step, args.frames)
                colors = {
                    name: fake_image(step, idx, args.width, args.height)
                    for idx, name in enumerate(CAMERAS)
                }
                recorder.add_item(colors=colors, depths={}, states=states, actions=actions)
            recorder.save_episode(success=episode not in failed)
            wait_ready(recorder)
            print(f"wrote {recorder.episode_dir} success={episode not in failed}")
    finally:
        recorder.close()

    print(f"\ntask dir: {task_dir}")


if __name__ == "__main__":
    main()
