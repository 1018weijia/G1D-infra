#!/usr/bin/env python3
"""Drive PolicyAdapter + PolicyRemoteClient against a server, with fake cameras.

Pair with tests/mock_policy_server.py to check the inference transport without
a robot or a GPU:

    python tests/mock_policy_server.py --bind tcp://127.0.0.1:5555 &
    python tests/check_policy_roundtrip.py --server-port 5555
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from teleop.utils.policy_client import PolicyAdapter, PolicyRemoteClient  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", default=os.path.join(REPO_ROOT, "configs/infer_g1d.yaml"))
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=15555)
    parser.add_argument("--protocol", default="zmq", choices=["zmq", "ws"])
    parser.add_argument("--instruction", default="pick up the red cup")
    parser.add_argument("--timeout-ms", type=int, default=10000)
    args = parser.parse_args()

    adapter = PolicyAdapter(
        config_path=args.config_path,
        instruction=args.instruction,
        action_interp_factor=1,
        exec_chunk_steps=8,
    )

    rng = np.random.default_rng(0)
    head = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    wrist_l = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    wrist_r = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    arm_q = np.zeros(14, dtype=float)

    frame, state = adapter.build_model_input(head, wrist_l, wrist_r, arm_q, 0.0, 0.0)
    print(f"stitched frame {frame.shape} dtype={frame.dtype}, state dim {state.shape[0]}")

    client = PolicyRemoteClient(
        server_host=args.server_host,
        server_port=args.server_port,
        timeout_ms=args.timeout_ms,
        protocol=args.protocol,
    )
    try:
        actions, predict_ms = client.predict(frame, state, args.instruction)
    finally:
        client.close()
    print(f"action chunk {actions.shape}, predict {predict_ms:.1f} ms")

    queue = adapter.build_exec_queue(actions, arm_q)
    print(f"exec queue {len(queue)} steps, first arm_q[:3]={queue[0].arm_q[:3]}, "
          f"grippers=({queue[0].left_grip:.3f}, {queue[0].right_grip:.3f})")
    print("roundtrip OK")


if __name__ == "__main__":
    main()
