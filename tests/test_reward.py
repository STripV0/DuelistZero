"""Tests for the sparse terminal reward function."""

from duelist_zero.engine.game_state import GameState, PlayerState, ZoneCard
from duelist_zero.env.reward import compute_reward


def _make_state(
    agent_lp=8000, opp_lp=8000,
    agent_hand=5, opp_hand=5,
    finished=False, winner=-1,
    turn=1,
) -> GameState:
    """Create a GameState with specified parameters."""
    state = GameState()
    p0 = state.players[0]
    p1 = state.players[1]
    p0.lp = agent_lp
    p1.lp = opp_lp
    p0.hand = [1] * agent_hand
    p1.hand = [1] * opp_hand
    state.is_finished = finished
    state.winner = winner
    state.current_turn = turn
    return state


class TestTerminalRewards:
    def test_win_returns_positive(self):
        state = _make_state(finished=True, winner=0, turn=10)
        r = compute_reward(state, perspective=0)
        assert r > 0.0

    def test_loss_returns_minus_one(self):
        state = _make_state(finished=True, winner=1)
        assert compute_reward(state, perspective=0) == -1.0

    def test_draw_returns_zero(self):
        state = _make_state(finished=True, winner=-1)
        assert compute_reward(state, perspective=0) == 0.0


class TestTurnScaling:
    def test_fast_win_higher_than_slow_win(self):
        fast = _make_state(finished=True, winner=0, turn=3)
        slow = _make_state(finished=True, winner=0, turn=18)
        assert compute_reward(fast, 0) > compute_reward(slow, 0)

    def test_turn_5_boundary(self):
        """Turn <= 5 should give max reward of 1.0."""
        for turn in [1, 3, 5]:
            state = _make_state(finished=True, winner=0, turn=turn)
            assert compute_reward(state, 0) == 1.0

    def test_turn_20_boundary(self):
        """Turn >= 20 should give min win reward of 0.3."""
        for turn in [20, 25, 50]:
            state = _make_state(finished=True, winner=0, turn=turn)
            assert abs(compute_reward(state, 0) - 0.3) < 1e-6

    def test_interpolation_midpoint(self):
        """Turn 12-13 should be roughly in the middle."""
        state = _make_state(finished=True, winner=0, turn=12)
        r = compute_reward(state, 0)
        # turn=12: 1.0 - 0.7 * (12-5)/15 = 1.0 - 0.7 * 7/15 ≈ 0.673
        assert 0.3 < r < 1.0

    def test_loss_not_scaled_by_turn(self):
        """Loss is always -1.0 regardless of turn count."""
        for turn in [1, 10, 30]:
            state = _make_state(finished=True, winner=1, turn=turn)
            assert compute_reward(state, 0) == -1.0


class TestIntermediateReward:
    def test_non_terminal_always_zero(self):
        """Intermediate steps should always return 0.0."""
        state = _make_state(agent_lp=4000, opp_lp=6000, finished=False)
        assert compute_reward(state, 0) == 0.0

    def test_non_terminal_zero_regardless_of_state(self):
        """Even dramatic LP differences produce 0.0 during game."""
        state = _make_state(agent_lp=100, opp_lp=8000, finished=False)
        assert compute_reward(state, 0) == 0.0


class TestPerspective:
    def test_player_0_wins(self):
        state = _make_state(finished=True, winner=0, turn=10)
        assert compute_reward(state, 0) > 0.0
        assert compute_reward(state, 1) == -1.0

    def test_player_1_wins(self):
        state = _make_state(finished=True, winner=1, turn=10)
        assert compute_reward(state, 1) > 0.0
        assert compute_reward(state, 0) == -1.0
