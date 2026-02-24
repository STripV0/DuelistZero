"""
Reward function for GoatEnv.

Terminal: +1.0 on win, -1.0 on loss, 0.0 on draw.
Intermediate: LP-delta shaping + card advantage delta + step penalty.
"""

from typing import Optional

from ..engine.game_state import GameState


def card_count(state: GameState, perspective: int) -> float:
    """Count total cards controlled by a player (hand + field).

    Field cards (monsters, spells) are weighted 1.5x to incentivize
    playing cards from hand onto the field.
    """
    p = state.players[perspective]
    return p.hand_count + 1.5 * p.monster_count + 1.5 * p.spell_count


def compute_reward(
    state: GameState,
    perspective: int,
    prev_lp: Optional[tuple[int, int]] = None,
    prev_cards: Optional[tuple[int, int]] = None,
    step_penalty: float = 0.0,
) -> float:
    """
    Compute the reward for the agent at `perspective`.

    Args:
        state: Current GameState (after the last step)
        perspective: Agent's player index (0 or 1)
        prev_lp: Optional (agent_lp, opp_lp) from before the step,
                 used for LP-delta reward shaping.
        prev_cards: Optional (agent_cards, opp_cards) from before the step,
                    used for card-advantage reward shaping.
        step_penalty: Small penalty per step to discourage stalling.

    Returns:
        +1.0 if agent won, -1.0 if agent lost, 0.0 on draw,
        or a shaped reward during the game.
    """
    if state.is_finished:
        if state.winner == perspective:
            return 1.0
        if state.winner == 1 - perspective:
            return -1.0
        return 0.0

    reward = 0.0

    # LP-delta shaping: (damage dealt to opponent - damage taken) / 8000
    if prev_lp is not None:
        agent_lp = state.players[perspective].lp
        opp_lp = state.players[1 - perspective].lp
        agent_delta = agent_lp - prev_lp[0]   # negative = damage taken
        opp_delta = opp_lp - prev_lp[1]       # negative = damage dealt
        reward += ((-opp_delta) - (-agent_delta)) / 8000.0

    # Card advantage delta: change in (my_cards - opp_cards) / 10
    if prev_cards is not None:
        cur_agent_cards = card_count(state, perspective)
        cur_opp_cards = card_count(state, 1 - perspective)
        prev_advantage = prev_cards[0] - prev_cards[1]
        cur_advantage = cur_agent_cards - cur_opp_cards
        reward += (cur_advantage - prev_advantage) / 10.0

    # Step penalty to discourage stalling
    reward -= step_penalty

    return reward
