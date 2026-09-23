"""Stage-2 act / transition / discard client, without the robot SDK."""
import pickle
import threading
import unittest

import numpy as np
import zmq

from teleop.utils.policy_client import PolicyRemoteClient
from teleop.utils.rlt_online import (
    REQUEST_KEY,
    RLTRollout,
    chunk_rewards,
    transition_fields,
)


def _serve(sock, seen, stop):
    while not stop.is_set():
        if not sock.poll(200):
            continue
        req = pickle.loads(sock.recv())
        seen.append(req)
        kind = req.get(REQUEST_KEY)
        if kind == "act":
            state = np.asarray(req["observation"]["observation/state"], dtype=np.float32)
            actions = np.tile(state, (4, 1))
            actions[:, 15] = 6.0
            sock.send(pickle.dumps({
                "status": "ok",
                "actions": actions,
                "transition_id": "t-1",
            }))
        elif kind in ("transition", "discard"):
            sock.send(pickle.dumps({"status": "ok", REQUEST_KEY: kind}))
        else:
            state = np.asarray(req["state"], dtype=np.float32)
            sock.send(pickle.dumps({
                "status": "ok",
                "actions": np.tile(state, (4, 1)),
                "predict_ms": 1.0,
            }))


class RLTOnlineTests(unittest.TestCase):
    def test_rewards_and_bootstrap(self):
        success = chunk_rewards(4, "success")
        self.assertEqual(success.tolist(), [0.0, 0.0, 0.0, 1.0])
        failure = transition_fields(4, "failure")
        self.assertEqual(float(failure["rewards"].sum()), 0.0)
        self.assertTrue(failure["done"])
        self.assertEqual(failure["bootstrap_mask"], 0.0)
        cut = transition_fields(4, None, intervention=True)
        self.assertFalse(cut["done"])
        self.assertTrue(cut["intervention"])
        self.assertEqual(cut["bootstrap_mask"], 1.0)

    def test_interrupt_before_and_after_motion(self):
        rollout = RLTRollout()
        self.assertEqual(rollout.begin_episode(), 1)
        actions = np.zeros((4, 16), dtype=np.float32)
        rollout.accept_chunk("a", actions, queue_len=4)
        kind, chunk = rollout.interrupt()
        self.assertEqual(kind, "discard")
        self.assertEqual(chunk["transition_id"], "a")

        rollout.accept_chunk("b", actions, queue_len=4)
        rollout.on_step()
        rollout.on_step()
        kind, chunk = rollout.interrupt()
        self.assertEqual(kind, "transition")
        self.assertTrue(chunk["intervention"])
        self.assertEqual(chunk["executed_steps"], 2)
        self.assertIsNone(rollout.interrupt())

    def test_chunk_finishes_after_every_queued_step(self):
        rollout = RLTRollout()
        rollout.begin_episode()
        rollout.accept_chunk("c", np.zeros((2, 16), dtype=np.float32), queue_len=2)
        self.assertFalse(rollout.chunk_finished())
        rollout.on_step()
        self.assertFalse(rollout.chunk_finished())
        rollout.on_step()
        self.assertTrue(rollout.chunk_finished())

    def test_zmq_act_transition_and_sft_predict(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.bind("tcp://127.0.0.1:0")
        endpoint = sock.getsockopt(zmq.LAST_ENDPOINT).decode()
        host, port = endpoint.rsplit(":", 1)
        host = host.replace("tcp://", "")
        seen = []
        stop = threading.Event()
        thread = threading.Thread(target=_serve, args=(sock, seen, stop), daemon=True)
        thread.start()
        client = PolicyRemoteClient(host, int(port), timeout_ms=2000, protocol="zmq")
        try:
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            state = np.linspace(0.1, 1.6, 16, dtype=np.float32)
            reply = client.act(frame, state, "倒豆子", episode_id=3, chunk_id=1)
            self.assertEqual(reply["transition_id"], "t-1")
            self.assertEqual(reply["actions"].shape, (4, 16))
            self.assertAlmostEqual(float(reply["actions"][0, 15]), 6.0)
            report = client.report_transition(
                reply["transition_id"],
                frame,
                state,
                "倒豆子",
                chunk_rewards(4, "success"),
                done=True,
                bootstrap_mask=0.0,
                action_chunk=reply["actions"],
                episode_id=3,
                chunk_id=1,
            )
            self.assertEqual(report["status"], "ok")
            client.discard("t-unused")
            actions, _predict_ms = client.predict(frame, state, "倒豆子")
            self.assertEqual(actions.shape, (4, 16))
        finally:
            client.close()
            stop.set()
            thread.join(timeout=2)
            sock.close()
            ctx.term()

        kinds = [item.get(REQUEST_KEY) for item in seen]
        self.assertEqual(kinds, ["act", "transition", "discard", None])
        obs = seen[0]["observation"]
        self.assertEqual(obs["prompt"], "倒豆子")
        self.assertEqual(np.asarray(obs["observation/state"]).shape, (16,))
        self.assertEqual(obs["observation/image"].dtype, np.uint8)
        self.assertEqual(seen[0]["identity"]["episode_id"], 3)
        self.assertEqual(seen[1]["action_chunk_space"], "robot")
        self.assertTrue(seen[1]["done"])
        self.assertEqual(seen[1]["bootstrap_mask"], 0.0)
        self.assertEqual(float(np.asarray(seen[1]["rewards"])[-1]), 1.0)


if __name__ == "__main__":
    unittest.main()
