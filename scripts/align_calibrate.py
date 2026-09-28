#!/usr/bin/env python3
"""Capture current XR controller poses as alignment_targets.json.

Hold both controllers at the desired alignment pose (same OpenXR convention
used by DualArmAlignment), then confirm to write the config.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _matrix_to_list(mat):
    return np.asarray(mat, dtype=float).reshape(4, 4).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=os.path.join(REPO_ROOT, "configs", "alignment_targets.json"),
        help="JSON path to write (default: configs/alignment_targets.json)",
    )
    parser.add_argument("--img-server-ip", default="192.168.123.164")
    parser.add_argument("--network-interface", default=None)
    parser.add_argument("--input-mode", choices=["hand", "controller"], default="controller")
    parser.add_argument("--display-mode", choices=["immersive", "ego", "pass-through"],
                        default="ego")
    parser.add_argument("--samples", type=int, default=10,
                        help="number of valid samples to average before write")
    parser.add_argument("--dry-run", action="store_true",
                        help="print targets only; do not write the file")
    parser.add_argument("--yes", action="store_true",
                        help="skip interactive confirmation")
    args = parser.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from teleop.teleimager.src.teleimager.image_client import ImageClient
    from teleop.televuer.tv_wrapper import TeleVuerWrapper
    from teleop.utils.policy_handoff import is_tracking_valid

    ChannelFactoryInitialize(0, networkInterface=args.network_interface)
    img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
    camera_config = img_client.get_cam_config()
    tv_wrapper = TeleVuerWrapper(
        use_hand_tracking=(args.input_mode == "hand"),
        binocular=camera_config["head_camera"]["binocular"],
        img_shape=camera_config["head_camera"]["image_shape"],
        display_mode=args.display_mode,
        webrtc=camera_config["head_camera"]["enable_webrtc"],
        webrtc_url=(
            f"https://{args.img_server_ip}:"
            f"{camera_config['head_camera']['webrtc_port']}/offer"
        ),
        arm_reference_mode="head_yaw",
    )

    print("Hold both controllers at the alignment pose.")
    print(f"Collecting {args.samples} valid OpenXR samples...")
    left_samples = []
    right_samples = []
    try:
        deadline = time.time() + 60.0
        while len(left_samples) < args.samples and time.time() < deadline:
            tele = tv_wrapper.get_tele_data()
            if is_tracking_valid(tele):
                left_samples.append(np.asarray(tele.left_wrist_pose_openxr, dtype=float))
                right_samples.append(np.asarray(tele.right_wrist_pose_openxr, dtype=float))
                print(f"  sample {len(left_samples)}/{args.samples}")
            time.sleep(0.1)
        if len(left_samples) < args.samples:
            print("Timed out waiting for valid XR tracking.", file=sys.stderr)
            return 1

        left = np.mean(np.stack(left_samples, axis=0), axis=0)
        right = np.mean(np.stack(right_samples, axis=0), axis=0)
        # Re-orthonormalize rotation blocks after averaging.
        for mat in (left, right):
            u, _, vt = np.linalg.svd(mat[:3, :3])
            mat[:3, :3] = u @ vt
            if np.linalg.det(mat[:3, :3]) < 0:
                u[:, -1] *= -1
                mat[:3, :3] = u @ vt

        payload = {"left": _matrix_to_list(left), "right": _matrix_to_list(right)}
        print(json.dumps(payload, indent=2))
        if args.dry_run:
            print("[dry-run] not writing file")
            return 0
        if not args.yes:
            answer = input(f"Write to {args.output}? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                print("Aborted.")
                return 1
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"Wrote {args.output}")
        return 0
    finally:
        try:
            tv_wrapper.close()
        except Exception:
            pass
        try:
            img_client.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
