"""Hardware-free smoke tests for policy rollout handoff helpers."""
import unittest
from types import SimpleNamespace

import numpy as np

from teleop.utils.handoff_utils import limit_joint_step, smoothstep_handoff_gain
from teleop.utils.policy_client import PolicyAdapter, validate_action_chunk
from teleop.utils.policy_handoff import (
    A_GAP_S,
    ALIGNING,
    DEBOUNCE_A,
    IGNORE_A,
    NONE,
    POLICY_IDLE,
    POLICY_LIVE,
    POLICY_ROLLBACK,
    REPEAT_A,
    RESUME_POLICY,
    ROLLBACK,
    START_POLICY,
    TAKEOVER,
    TELEOP_LIVE,
    ButtonRisingEdge,
    blend_should_finish,
    compute_relative_arm_command,
    interpret_key,
    is_tracking_valid,
    rollback_hold_from_last_command,
    stale_key_flags,
)
from teleop.utils.rollback import PolicyRollbackBuffer, ease_out_playback


class PolicyRolloutHandoffTests(unittest.TestCase):
    def test_qpos_conversion_order(self):
        row = np.arange(16, dtype=float)
        arm_q, left_grip, right_grip = PolicyAdapter.qpos_action_to_g1_action(row)
        self.assertEqual(arm_q.shape, (14,))
        self.assertEqual(left_grip, row[7])
        self.assertEqual(right_grip, row[15])
        self.assertTrue(np.array_equal(arm_q[:7], row[:7]))
        self.assertTrue(np.array_equal(arm_q[7:], row[8:15]))

    def test_validate_rejects_bad_chunk(self):
        with self.assertRaises(ValueError):
            validate_action_chunk(np.zeros((0, 16)))
        with self.assertRaises(ValueError):
            validate_action_chunk(np.full((2, 15), np.nan))
        with self.assertRaises(ValueError):
            validate_action_chunk(np.array([np.nan] + [0.0] * 15).reshape(1, 16))

    def test_gripper_range_is_validated_separately(self):
        chunk = np.zeros((2, 16), dtype=float)
        chunk[:, [7, 15]] = 5.2
        validated = validate_action_chunk(chunk)
        self.assertEqual(validated.shape, (2, 16))
        chunk[0, 7] = 5.6
        with self.assertRaises(ValueError):
            validate_action_chunk(chunk)

    def test_policy_rollback_buffer_reverse(self):
        buffer = PolicyRollbackBuffer(seconds=1.0, frequency=10.0)
        for idx in range(5):
            buffer.append(
                np.full(14, idx, dtype=float),
                np.zeros(14),
                float(idx),
                float(idx + 0.5),
            )
        playback = buffer.reverse_playback(exclude_latest=True)
        self.assertEqual(len(playback), 4)
        self.assertEqual(playback[0][0][0], 3.0)
        self.assertEqual(playback[-1][0][0], 0.0)
        self.assertEqual(playback[0][2], 3.0)
        self.assertEqual(len(buffer), 0)

    def test_ease_out_stops_on_the_recorded_endpoint(self):
        frames = []
        for idx in range(30):
            frames.append((
                np.full(14, float(idx), dtype=float),
                np.full(14, float(idx) * 0.1),
                float(idx),
                float(idx) + 0.25,
            ))
        eased = ease_out_playback(frames, ease_steps=12)
        self.assertGreater(len(eased), len(frames))
        self.assertTrue(np.allclose(eased[17][0], frames[17][0]))
        self.assertTrue(np.allclose(eased[-1][0], frames[-1][0]))
        self.assertTrue(np.allclose(eased[-1][1], frames[-1][1]))
        self.assertEqual(eased[-1][2], frames[-1][2])
        self.assertEqual(eased[-1][3], frames[-1][3])
        steps = np.abs(np.diff(np.stack([frame[0] for frame in eased]), axis=0))
        early = float(np.max(steps[:8]))
        final = float(np.max(steps[-6:]))
        self.assertGreater(early, 0.5)
        self.assertLess(final, 0.05)

    def test_gamepad_a_fires_once_per_press(self):
        edge = ButtonRisingEdge()
        self.assertFalse(edge.update(False))
        self.assertTrue(edge.update(True))
        self.assertFalse(edge.update(True))
        self.assertFalse(edge.update(False))
        self.assertTrue(edge.update(True))

    def test_handoff_helpers(self):
        self.assertEqual(smoothstep_handoff_gain(0.0, 0.3), 0.0)
        self.assertEqual(smoothstep_handoff_gain(0.3, 0.3), 1.0)
        limited = limit_joint_step(np.ones(14), np.zeros(14), 0.5 / 30.0)
        self.assertLessEqual(float(np.max(np.abs(limited))), 0.5 / 30.0 + 1e-12)

    def test_rollback_hold_uses_last_command_not_measured_state(self):
        last_q = np.arange(14, dtype=float)
        last_tau = np.full(14, 0.2)
        hold_q, hold_tau, left, right = rollback_hold_from_last_command(
            last_q, last_tau, 1.5, 1.7
        )
        measured_q = last_q + 0.35
        self.assertTrue(np.array_equal(hold_q, last_q))
        self.assertFalse(np.allclose(hold_q, measured_q))
        self.assertTrue(np.array_equal(hold_tau, last_tau))
        self.assertEqual(left, 1.5)
        self.assertEqual(right, 1.7)


class PolicyHandoffKeyTests(unittest.TestCase):
    def test_s_starts_only_from_idle_or_aligning(self):
        self.assertEqual(interpret_key("s", POLICY_IDLE, 1.0, 0.0, 0.0)[0], START_POLICY)
        self.assertEqual(interpret_key("s", ALIGNING, 1.0, 0.0, 0.0)[0], START_POLICY)
        self.assertEqual(interpret_key("s", TELEOP_LIVE, 1.0, 0.0, 0.0)[0], RESUME_POLICY)
        self.assertEqual(interpret_key("s", POLICY_LIVE, 1.0, 0.0, 0.0)[0], NONE)
        self.assertEqual(interpret_key("s", POLICY_ROLLBACK, 1.0, 0.0, 0.0)[0], NONE)

    def test_a_takeover_then_held_repeat_does_not_resume(self):
        first, last_a = interpret_key("a", ALIGNING, 10.0, 0.0, 0.0)
        self.assertEqual(first, TAKEOVER)
        debounce_until = last_a + 0.5
        held, last_a = interpret_key("a", TELEOP_LIVE, 10.05, debounce_until, last_a)
        self.assertEqual(held, DEBOUNCE_A)
        held, last_a = interpret_key("a", TELEOP_LIVE, 10.40, debounce_until, last_a)
        self.assertEqual(held, DEBOUNCE_A)
        after_debounce, last_a = interpret_key("a", TELEOP_LIVE, 10.51, debounce_until, last_a)
        self.assertEqual(after_debounce, REPEAT_A)
        released, _ = interpret_key(
            "a", TELEOP_LIVE, last_a + A_GAP_S, debounce_until, last_a
        )
        self.assertEqual(released, RESUME_POLICY)

    def test_a_ignored_until_rollback(self):
        self.assertEqual(interpret_key("a", POLICY_LIVE, 1.0, 0.0, 0.0)[0], IGNORE_A)
        self.assertEqual(interpret_key("a", POLICY_IDLE, 1.0, 0.0, 0.0)[0], IGNORE_A)

    def test_b_allowed_during_policy_and_teleop(self):
        self.assertEqual(interpret_key("b", POLICY_LIVE, 1.0, 0.0, 0.0)[0], ROLLBACK)
        self.assertEqual(interpret_key("b", TELEOP_LIVE, 1.0, 0.0, 0.0)[0], ROLLBACK)
        self.assertEqual(interpret_key("b", ALIGNING, 1.0, 0.0, 0.0)[0], NONE)

    def test_stale_start_policy_is_dropped_in_live_policy(self):
        start, resume, confirm, rollback = stale_key_flags(
            POLICY_LIVE, True, True, True, False,
        )
        self.assertFalse(start)
        self.assertFalse(resume)
        self.assertFalse(confirm)
        self.assertFalse(rollback)

    def test_blend_finish_requires_tracking_and_ik(self):
        self.assertFalse(blend_should_finish(0.3, 0.3, False, True))
        self.assertFalse(blend_should_finish(0.3, 0.3, True, False))
        self.assertFalse(blend_should_finish(0.2, 0.3, True, True))
        self.assertTrue(blend_should_finish(0.3, 0.3, True, True))

    def test_tracking_valid_requires_both_wrists(self):
        self.assertFalse(is_tracking_valid(SimpleNamespace(
            motion_data_ready=True, left_wrist_valid=True, right_wrist_valid=False,
        )))
        self.assertTrue(is_tracking_valid(SimpleNamespace(
            motion_data_ready=True, left_wrist_valid=True, right_wrist_valid=True,
        )))


class _FakeAlignment:
    def __init__(self):
        self.reanchored = 0

    def reanchor(self, left, right, robot_left, robot_right):
        self.reanchored += 1
        return True

    def targets_from_hand(self, left, right, gain=1.0):
        return left, right


class _FakeIK:
    def __init__(self):
        self.last_solve_succeeded = False
        self.seed = None

    def reset_solution_state(self, q):
        self.seed = np.asarray(q, dtype=float).copy()
        self.last_solve_succeeded = False

    def solve_fk_matrix(self, q):
        return np.eye(4), np.eye(4)

    def solve_ik(self, left, right, seed):
        self.last_solve_succeeded = True
        q = np.asarray(seed, dtype=float)[:14] + 0.2
        return q, np.zeros_like(q)


class RelativeTeleopCommandTests(unittest.TestCase):
    def test_tracking_loss_holds_and_reanchor_restarts_blend(self):
        alignment = _FakeAlignment()
        arm_ik = _FakeIK()
        last = np.zeros(14)
        previous = np.full(14, 0.1)
        lost = compute_relative_arm_command(
            alignment, arm_ik, np.eye(4), np.eye(4),
            tracking_valid=False, blending=True, tracking_lost=False,
            now=1.0, blend_started=0.5, blend_seconds=0.3,
            max_joint_speed=0.5, frequency=30.0,
            previous_q=previous, last_command_q=last,
            last_arm_q=last, current_lr_arm_q=last,
        )
        self.assertTrue(lost.tracking_lost)
        self.assertTrue(np.allclose(lost.sol_q[:14], previous))

        recovered = compute_relative_arm_command(
            alignment, arm_ik, np.eye(4), np.eye(4),
            tracking_valid=True, blending=True, tracking_lost=True,
            now=1.4, blend_started=0.5, blend_seconds=0.3,
            max_joint_speed=0.5, frequency=30.0,
            previous_q=previous, last_command_q=last,
            last_arm_q=last, current_lr_arm_q=last,
        )
        self.assertFalse(recovered.tracking_lost)
        self.assertTrue(recovered.reanchored)
        self.assertEqual(recovered.blend_started, 1.4)
        self.assertEqual(alignment.reanchored, 1)
        self.assertLessEqual(recovered.max_delta, 0.5 / 30.0 + 1e-12)


if __name__ == "__main__":
    unittest.main()
