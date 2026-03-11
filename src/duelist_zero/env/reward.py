"""
Reward function for GoatEnv.

Terminal reward scaled by turn count, plus optional potential-based reward
shaping (PBRS, Ng et al. 1999) using the difference gamma*Phi(s') - Phi(s).
PBRS is mathematically guaranteed to preserve the optimal policy.
"""

from ..engine.game_state import GameState


def compute_potential(state: GameState, perspective: int) -> float:
    """
    Compute the potential function Phi(s) for PBRS.

    Uses only public game state — no DB lookups required.
    Returns a value roughly in [-1, 1].
    """
    me = state.players[perspective]
    opp = state.players[1 - perspective]

    # LP advantage: clipped to [-1, 1]
    lp_adv = max(-1.0, min(1.0, (me.lp - opp.lp) / 8000.0))

    # Board (monster) advantage: [-1, 1]
    board_adv = (me.monster_count - opp.monster_count) / 5.0

    # Total card advantage (hand + monsters + spells): clipped to [-1, 1]
    my_cards = me.hand_count + me.monster_count + me.spell_count
    opp_cards = opp.hand_count + opp.monster_count + opp.spell_count
    card_adv = max(-1.0, min(1.0, (my_cards - opp_cards) / 10.0))

    return 0.4 * lp_adv + 0.35 * board_adv + 0.25 * card_adv


def compute_reward(
    state: GameState,
    perspective: int,
    prev_potential: float = 0.0,
    curr_potential: float = 0.0,
    gamma: float = 0.99,
    shaping_scale: float = 0.0,
) -> tuple[float, float]:
    """
    Compute the reward for the agent at `perspective`.

    Returns (shaped_reward, current_potential).

    Terminal:
      Win: linearly scaled from 1.0 (turn <= 5) to 0.3 (turn >= 20).
      Loss: flat -1.0.
      Draw: 0.0.
      PBRS terminal convention: curr_potential = 0.

    Non-terminal:
      Base reward is 0.0, plus PBRS shaping if shaping_scale > 0.
    """
    if state.is_finished:
        # Terminal reward
        if state.winner == perspective:
            turn = max(state.current_turn, 1)
            if turn <= 5:
                r_terminal = 1.0
            elif turn >= 20:
                r_terminal = 0.3
            else:
                r_terminal = 1.0 - 0.7 * (turn - 5) / 15.0
        elif state.winner == 1 - perspective:
            r_terminal = -1.0
        else:
            r_terminal = 0.0

        # PBRS absorbing state convention: Phi(terminal) = 0
        terminal_potential = 0.0
        shaping = shaping_scale * (gamma * terminal_potential - prev_potential)
        return r_terminal + shaping, terminal_potential

    # Non-terminal: base reward = 0, plus PBRS shaping
    shaping = shaping_scale * (gamma * curr_potential - prev_potential)
    return shaping, curr_potential
