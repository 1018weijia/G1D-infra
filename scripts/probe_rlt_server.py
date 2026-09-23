#!/usr/bin/env python3
"""Send one fake observation and discard it.

Does not import the robot SDK and does not move the arm. A passing probe
prints the action shape, the clip check, and the transition id.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop.utils.policy_client import PolicyRemoteClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--timeout-ms", type=int, default=180000)
    parser.add_argument(
        "--instruction",
        default="倒豆子",
        help="Must match a cached T5 sentence when the server has no live encoder yet.",
    )
    parser.add_argument("--clip-min", type=float, default=-2.1640625)
    parser.add_argument("--clip-max", type=float, default=6.03125)
    args = parser.parse_args()

    frame = np.zeros((384, 320, 3), dtype=np.uint8)
    state = np.zeros(16, dtype=np.float32)
    client = PolicyRemoteClient(
        args.host, args.port, timeout_ms=args.timeout_ms, protocol="zmq"
    )
    try:
        reply = client.act(frame, state, args.instruction, episode_id=0, chunk_id=0)
        actions = np.asarray(reply["actions"], dtype=np.float32)
        transition_id = str(reply["transition_id"])
        if actions.shape != (64, 16):
            raise SystemExit(f"expected actions (64, 16), got {actions.shape}")
        if not np.isfinite(actions).all():
            raise SystemExit("actions are not finite")
        if float(actions.min()) < args.clip_min or float(actions.max()) > args.clip_max:
            raise SystemExit(
                f"actions outside clip [{args.clip_min}, {args.clip_max}]: "
                f"min={actions.min():.4f} max={actions.max():.4f}"
            )
        discarded = client.discard(transition_id)
        if discarded.get("status") != "ok":
            raise SystemExit(f"discard failed: {discarded}")
    finally:
        client.close()
    print(
        f"probe ok actions={actions.shape} id={transition_id} "
        f"min={float(actions.min()):.4f} max={float(actions.max()):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
