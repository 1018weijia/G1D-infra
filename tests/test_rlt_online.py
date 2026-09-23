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
    TakeoverChunk,
    chunk_rewards,
    outcome_ends_without_chunk,
    qpos_command,
    rewind_frame_count,
    rewind_plan,
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
    def test_outcome_while_waiting_does_not_attach_to_the_next_chunk(self):
        self.assertTrue(outcome_ends_without_chunk("success", False, 0))
        self.assertTrue(outcome_ends_without_chunk("failure", False, 0))
        self.assertFalse(outcome_ends_without_chunk("success", True, 0))
        self.assertFalse(outcome_ends_without_chunk("success", False, 3))
        self.assertFalse(outcome_ends_without_chunk(None, False, 0))

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
        rollout.add_step_reward(0.5)
        self.assertFalse(rollout.chunk_finished())
        rollout.on_step()
        rollout.add_step_reward(1.0)
        self.assertTrue(rollout.chunk_finished())
        chunk = rollout.take_open()
        self.assertEqual(chunk["rewards"].tolist(), [0.5, 1.0])

    def test_step_images_stitch_off_the_control_call(self):
        rollout = RLTRollout()
        rollout.begin_episode()
        rollout.accept_chunk("t-img", np.zeros((1, 16), dtype=np.float32), 1)
        seen = []

        def stitch(head, left, right):
            seen.append((head.shape, left.shape, right.shape))
            return np.zeros((4, 4, 3), dtype=np.uint8)

        blank = np.zeros((2, 2, 3), dtype=np.uint8)
        rollout.on_step(
            state=np.zeros(16, dtype=np.float32),
            cameras=(blank, blank, blank),
            stitch=stitch,
        )
        chunk = rollout.take_open()
        self.assertEqual(len(seen), 1)
        self.assertEqual(chunk["step_observations"][0]["observation/image"].shape, (4, 4, 3))
        self.assertEqual(chunk["step_observations"][0]["observation/state"].shape, (16,))

    def test_takeover_commands_match_policy_layout(self):
        arm = np.arange(1, 15, dtype=np.float32)
        command = qpos_command(arm, 5.5, 4.25)
        self.assertEqual(command.shape, (16,))
        self.assertEqual(command[:7].tolist(), arm[:7].tolist())
        self.assertEqual(command[8:15].tolist(), arm[7:14].tolist())
        self.assertAlmostEqual(float(command[7]), 5.5)
        self.assertAlmostEqual(float(command[15]), 4.25)

    def test_rewind_plan_matches_rlinf_exit_and_credit(self):
        physical = rewind_plan(frames=90, chunk_len=64, stored_chunks=3, include_current=True)
        self.assertEqual(physical["mode"], "exit")
        self.assertEqual(physical["chunks"], 1)
        self.assertEqual(physical["terminal_reward"], -0.2)
        self.assertEqual(rewind_frame_count(20, 64), 19)
        self.assertEqual(rewind_frame_count(0, 64), 63)
        credit = rewind_plan(frames=0, chunk_len=64, stored_chunks=2, include_current=False)
        self.assertEqual(credit["mode"], "credit")
        self.assertEqual(credit["chunks"], 1)
        self.assertEqual(credit["prefix_reward"], 0.1)
        self.assertIsNone(rewind_plan(0, 64, 0, False))

    def test_takeover_chunk_fills_then_closes(self):
        chunk = TakeoverChunk()
        self.assertIsNone(chunk.close())
        chunk.open("t9", episode_id=2, chunk_id=1, chunk_len=2)
        self.assertFalse(chunk.push(qpos_command(np.zeros(14), 1.0, 2.0)))
        self.assertTrue(chunk.push(qpos_command(np.ones(14), 3.0, 4.0)))
        identity, actions, executed = chunk.close()
        self.assertEqual(identity["transition_id"], "t9")
        self.assertEqual(identity["chunk_id"], 1)
        self.assertEqual(executed, 2)
        self.assertEqual(actions.shape, (2, 16))
        self.assertAlmostEqual(float(actions[1, 7]), 3.0)
        self.assertFalse(chunk.active)
        self.assertIsNone(chunk.close())

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
