#!/usr/bin/env python3
"""Minimal stand-in for the GPU inference server (ZMQ REQ/REP).

Answers every request with a constant-hold action chunk, so the transport,
SSH tunnel, and PolicyRemoteClient can be checked without a model.
"""
import argparse
import pickle
import time

import numpy as np
import zmq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="tcp://127.0.0.1:5555")
    parser.add_argument("--chunk-steps", type=int, default=64)
    parser.add_argument("--requests", type=int, default=0, help="exit after N requests; 0 = forever")
    args = parser.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(args.bind)
    print(f"mock policy server listening on {args.bind}", flush=True)

    served = 0
    try:
        while True:
            req = pickle.loads(sock.recv())
            start = time.time()
            state = np.asarray(req.get("state"), dtype=np.float32).reshape(-1)
            if state.size != 16:
                sock.send(pickle.dumps({"status": "error", "message": f"bad state dim {state.size}"}))
                continue
            layout = req.get("image_layout", "stitched" if req.get("first_frame") is not None else "separate")
            if layout == "stitched":
                frame = req.get("first_frame")
                if frame is None:
                    sock.send(pickle.dumps({"status": "error", "message": "missing first_frame"}))
                    continue
                image_desc = f"frame={np.asarray(frame).shape}"
            elif layout == "separate":
                images = req.get("images") or {}
                if not images:
                    sock.send(pickle.dumps({"status": "error", "message": "missing images"}))
                    continue
                image_desc = "images=" + ",".join(
                    f"{name}:{np.asarray(image).shape}" for name, image in images.items()
                )
            else:
                sock.send(pickle.dumps({"status": "error", "message": f"bad image_layout {layout}"}))
                continue
            actions = np.tile(state, (args.chunk_steps, 1)).astype(np.float32)
            sock.send(pickle.dumps({
                "status": "ok",
                "actions": actions,
                "predict_ms": (time.time() - start) * 1000.0,
            }))
            served += 1
            print(f"served request {served} instruction={req.get('instruction')!r} "
                  f"{image_desc}", flush=True)
            if args.requests and served >= args.requests:
                break
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
