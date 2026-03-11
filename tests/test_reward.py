"""Tests for the reward function including PBRS shaping."""

from duelist_zero.engine.game_state import GameState, PlayerState, ZoneCard
from duelist_zero.env.reward import compute_reward, compute_potential


def _make_state(
    agent_lp=8000, opp_lp=8000,
    agent_hand=5, opp_hand=5,
    agent_monsters=0, opp_monsters=0,
    agent_spells=0, opp_spells=0,
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
    # Place monsters
    for i in range(min(agent_monsters, 5)):
        p0.monsters[i] = ZoneCard(code=1, position=1, controller=0, sequence=i)
    for i in range(min(opp_monsters, 5)):
        p1.monsters[i] = ZoneCard(code=1, position=1, controller=1, sequence=i)
    # Place spells
    for i in range(min(agent_spells, 5)):
        p0.spells[i] = ZoneCard(code=1, position=1, controller=0, sequence=i)
    for i in range(min(opp_spells, 5)):
        p1.spells[i] = ZoneCard(code=1, position=1, controller=1, sequence=i)
    state.is_finished = finished
    state.winner = winner
    state.current_turn = turn
    return state


class TestTerminalRewards:
    def test_win_returns_positive(self):
        state = _make_state(finished=True, winner=0, turn=10)
        r, _ = compute_reward(state, perspective=0)
        assert r > 0.0

    def test_loss_returns_minus_one(self):
        state = _make_state(finished=True, winner=1)
        r, _ = compute_reward(state, perspective=0)
        assert r == -1.0

    def test_draw_returns_zero(self):
        state = _make_state(finished=True, winner=-1)
        r, _ = compute_reward(state, perspective=0)
        assert r == 0.0


class TestTurnScaling:
    def test_fast_win_higher_than_slow_win(self):
        fast = _make_state(finished=True, winner=0, turn=3)
        slow = _make_state(finished=True, winner=0, turn=18)
        r_fast, _ = compute_reward(fast, 0)
        r_slow, _ = compute_reward(slow, 0)
        assert r_fast > r_slow

    def test_turn_5_boundary(self):
        """Turn <= 5 should give max reward of 1.0."""
        for turn in [1, 3, 5]:
            state = _make_state(finished=True, winner=0, turn=turn)
            r, _ = compute_reward(state, 0)
            assert r == 1.0

    def test_turn_20_boundary(self):
        """Turn >= 20 should give min win reward of 0.3."""
        for turn in [20, 25, 50]:
            state = _make_state(finished=True, winner=0, turn=turn)
            r, _ = compute_reward(state, 0)
            assert abs(r - 0.3) < 1e-6

    def test_interpolation_midpoint(self):
        """Turn 12-13 should be roughly in the middle."""
        state = _make_state(finished=True, winner=0, turn=12)
        r, _ = compute_reward(state, 0)
        assert 0.3 < r < 1.0

    def test_loss_not_scaled_by_turn(self):
        """Loss is always -1.0 regardless of turn count."""
        for turn in [1, 10, 30]:
            state = _make_state(finished=True, winner=1, turn=turn)
            r, _ = compute_reward(state, 0)
            assert r == -1.0


class TestIntermediateReward:
    def test_non_terminal_zero_without_shaping(self):
        """Without shaping, intermediate steps return 0.0."""
        state = _make_state(agent_lp=4000, opp_lp=6000, finished=False)
        r, _ = compute_reward(state, 0)
        assert r == 0.0

    def test_non_terminal_zero_regardless_of_state(self):
        """Even dramatic LP differences produce 0.0 without shaping."""
        state = _make_state(agent_lp=100, opp_lp=8000, finished=False)
        r, _ = compute_reward(state, 0)
        assert r == 0.0


class TestPerspective:
    def test_player_0_wins(self):
        state = _make_state(finished=True, winner=0, turn=10)
        r0, _ = compute_reward(state, 0)
        r1, _ = compute_reward(state, 1)
        assert r0 > 0.0
        assert r1 == -1.0

    def test_player_1_wins(self):
        state = _make_state(finished=True, winner=1, turn=10)
        r1, _ = compute_reward(state, 1)
        r0, _ = compute_reward(state, 0)
        assert r1 > 0.0
        assert r0 == -1.0


class TestComputePotential:
    def test_equal_state_zero_potential(self):
        """Symmetric game state should have ~zero potential."""
        state = _make_state()
        p = compute_potential(state, 0)
        assert abs(p) < 1e-6

    def test_lp_advantage_positive(self):
        """Agent with LP advantage should have positive potential."""
        state = _make_state(agent_lp=8000, opp_lp=4000)
        p = compute_potential(state, 0)
        assert p > 0.0

    def test_lp_disadvantage_negative(self):
        """Agent with LP disadvantage should have negative potential."""
        state = _make_state(agent_lp=2000, opp_lp=8000)
        p = compute_potential(state, 0)
        assert p < 0.0

    def test_board_advantage(self):
        """More monsters should increase potential."""
        state = _make_state(agent_monsters=3, opp_monsters=0)
        p = compute_potential(state, 0)
        assert p > 0.0

    def test_card_advantage(self):
        """More total cards should increase potential."""
        state = _make_state(agent_hand=6, opp_hand=2,
                           agent_monsters=2, opp_monsters=0)
        p = compute_potential(state, 0)
        assert p > 0.0

    def test_symmetric_perspectives(self):
        """Potential from p0's view should negate potential from p1's view."""
        state = _make_state(agent_lp=6000, opp_lp=4000,
                           agent_monsters=2, opp_monsters=1,
                           agent_hand=4, opp_hand=3)
        p0 = compute_potential(state, 0)
        p1 = compute_potential(state, 1)
        assert abs(p0 + p1) < 1e-6

    def test_potential_bounded(self):
        """Potential should stay in [-1, 1] range."""
        # Extreme advantage
        state = _make_state(agent_lp=8000, opp_lp=0,
                           agent_monsters=5, opp_monsters=0,
                           agent_hand=10, opp_hand=0,
                           agent_spells=5, opp_spells=0)
        p = compute_potential(state, 0)
        assert -1.0 <= p <= 1.0

        # Extreme disadvantage
        p_opp = compute_potential(state, 1)
        assert -1.0 <= p_opp <= 1.0


class TestPBRSShaping:
    def test_shaping_zero_when_disabled(self):
        """With shaping_scale=0, reward should equal terminal-only reward."""
        state = _make_state(finished=False)
        r, _ = compute_reward(state, 0, shaping_scale=0.0)
        assert r == 0.0

    def test_shaping_positive_for_improvement(self):
        """Improving state should give positive shaping reward."""
        state = _make_state(agent_lp=8000, opp_lp=4000, finished=False,
                           agent_monsters=2, opp_monsters=0)
        curr_pot = compute_potential(state, 0)
        prev_pot = 0.0  # was equal before
        r, _ = compute_reward(state, 0, prev_potential=prev_pot,
                              curr_potential=curr_pot, shaping_scale=0.5)
        assert r > 0.0

    def test_shaping_negative_for_deterioration(self):
        """Worsening state should give negative shaping reward."""
        state = _make_state(agent_lp=2000, opp_lp=8000, finished=False,
                           agent_monsters=0, opp_monsters=3)
        curr_pot = compute_potential(state, 0)
        prev_pot = 0.5  # was favorable before
        r, _ = compute_reward(state, 0, prev_potential=prev_pot,
                              curr_potential=curr_pot, shaping_scale=0.5)
        assert r < 0.0

    def test_terminal_potential_zero(self):
        """Terminal state should return potential = 0."""
        state = _make_state(finished=True, winner=0, turn=10)
        _, potential = compute_reward(state, 0, shaping_scale=0.5)
        assert potential == 0.0

    def test_shaping_small_relative_to_terminal(self):
        """Shaping rewards should be small compared to terminal rewards."""
        state = _make_state(agent_lp=6000, opp_lp=4000, finished=False,
                           agent_monsters=2, opp_monsters=1)
        curr_pot = compute_potential(state, 0)
        prev_pot = 0.0
        r, _ = compute_reward(state, 0, prev_potential=prev_pot,
                              curr_potential=curr_pot, shaping_scale=0.5)
        # Shaping should be much smaller than terminal reward
        assert abs(r) < 0.5

    def test_backward_compatible_no_shaping(self):
        """Default args (no shaping) should match original behavior."""
        state = _make_state(finished=True, winner=0, turn=10)
        r, _ = compute_reward(state, 0)
        # Should equal turn-scaled win reward
        expected = 1.0 - 0.7 * (10 - 5) / 15.0
        assert abs(r - expected) < 1e-6
