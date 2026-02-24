"""Tests for the reward function."""

from duelist_zero.engine.game_state import GameState, PlayerState, ZoneCard
from duelist_zero.env.reward import compute_reward, card_count


def _make_state(
    agent_lp=8000, opp_lp=8000,
    agent_hand=5, agent_monsters=0, agent_spells=0,
    opp_hand=5, opp_monsters=0, opp_spells=0,
    finished=False, winner=-1,
) -> GameState:
    """Create a GameState with specified parameters."""
    state = GameState()
    p0 = state.players[0]
    p1 = state.players[1]
    p0.lp = agent_lp
    p1.lp = opp_lp
    p0.hand = [1] * agent_hand
    p1.hand = [1] * opp_hand
    for i in range(agent_monsters):
        p0.monsters[i] = ZoneCard(code=1)
    for i in range(opp_monsters):
        p1.monsters[i] = ZoneCard(code=1)
    for i in range(agent_spells):
        p0.spells[i] = ZoneCard(code=1)
    for i in range(opp_spells):
        p1.spells[i] = ZoneCard(code=1)
    state.is_finished = finished
    state.winner = winner
    return state


class TestTerminalRewards:
    def test_win_returns_plus_one(self):
        state = _make_state(finished=True, winner=0)
        assert compute_reward(state, perspective=0) == 1.0

    def test_loss_returns_minus_one(self):
        state = _make_state(finished=True, winner=1)
        assert compute_reward(state, perspective=0) == -1.0

    def test_draw_returns_zero(self):
        state = _make_state(finished=True, winner=-1)
        assert compute_reward(state, perspective=0) == 0.0

    def test_terminal_ignores_shaping(self):
        """Terminal rewards should not include step penalty or card advantage."""
        state = _make_state(finished=True, winner=0)
        r = compute_reward(
            state, perspective=0,
            prev_lp=(8000, 8000),
            prev_cards=(5, 5),
            step_penalty=0.1,
        )
        assert r == 1.0


class TestCardCount:
    def test_card_count_hand_only(self):
        state = _make_state(agent_hand=5, agent_monsters=0, agent_spells=0)
        assert card_count(state, 0) == 5

    def test_card_count_full(self):
        state = _make_state(agent_hand=3, agent_monsters=2, agent_spells=1)
        assert card_count(state, 0) == 7.5

    def test_card_count_opponent(self):
        state = _make_state(opp_hand=4, opp_monsters=1, opp_spells=2)
        assert card_count(state, 1) == 8.5


class TestCardAdvantageDelta:
    def test_gaining_card_advantage(self):
        """Agent gains a card advantage → positive reward component."""
        state = _make_state(agent_hand=6, opp_hand=4)
        prev_cards = (5, 5)  # was even
        r = compute_reward(state, 0, prev_lp=(8000, 8000), prev_cards=prev_cards, step_penalty=0.0)
        # card advantage delta = (6-4) - (5-5) = 2, scaled by /10 = 0.2
        # LP delta = 0
        assert abs(r - 0.2) < 1e-6

    def test_losing_card_advantage(self):
        """Agent loses card advantage → negative reward component."""
        state = _make_state(agent_hand=4, opp_hand=6)
        prev_cards = (5, 5)
        r = compute_reward(state, 0, prev_lp=(8000, 8000), prev_cards=prev_cards, step_penalty=0.0)
        assert abs(r - (-0.2)) < 1e-6

    def test_no_change(self):
        """No card advantage change → zero card component."""
        state = _make_state(agent_hand=5, opp_hand=5)
        prev_cards = (5, 5)
        r = compute_reward(state, 0, prev_lp=(8000, 8000), prev_cards=prev_cards, step_penalty=0.0)
        assert abs(r) < 1e-6


class TestStepPenalty:
    def test_step_penalty_applied(self):
        """Step penalty should be subtracted on non-terminal steps."""
        state = _make_state()
        r = compute_reward(state, 0, prev_lp=(8000, 8000), prev_cards=(5, 5), step_penalty=0.002)
        assert abs(r - (-0.002)) < 1e-6

    def test_custom_step_penalty(self):
        state = _make_state()
        r = compute_reward(state, 0, prev_lp=(8000, 8000), prev_cards=(5, 5), step_penalty=0.01)
        assert abs(r - (-0.01)) < 1e-6

    def test_zero_step_penalty(self):
        state = _make_state()
        r = compute_reward(state, 0, prev_lp=(8000, 8000), prev_cards=(5, 5), step_penalty=0.0)
        assert abs(r) < 1e-6


class TestCombinedReward:
    def test_lp_and_card_and_penalty(self):
        """All three components should combine additively."""
        # Agent dealt 1000 damage to opponent, gained 1 card advantage
        state = _make_state(agent_lp=8000, opp_lp=7000, agent_hand=6, opp_hand=5)
        prev_lp = (8000, 8000)
        prev_cards = (5, 5)
        r = compute_reward(state, 0, prev_lp=prev_lp, prev_cards=prev_cards, step_penalty=0.002)
        lp_component = 1000 / 8000.0  # 0.125
        card_component = ((6 - 5) - (5 - 5)) / 10.0  # 0.1
        expected = lp_component + card_component - 0.002
        assert abs(r - expected) < 1e-6
