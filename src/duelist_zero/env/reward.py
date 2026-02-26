"""
Reward function for GoatEnv.

Sparse terminal reward only — no intermediate shaping.
Win reward is scaled by turn count to encourage faster wins.
"""

from ..engine.game_state import GameState


def compute_reward(state: GameState, perspective: int) -> float:
    """
    Compute the reward for the agent at `perspective`.

    Non-terminal steps return 0.0.
    Win: linearly scaled from 1.0 (turn <= 5) to 0.3 (turn >= 20).
    Loss: flat -1.0.
    Draw: 0.0.
    """
    if not state.is_finished:
        return 0.0

    if state.winner == perspective:
        # Fast wins are worth more
        turn = max(state.current_turn, 1)
        if turn <= 5:
            return 1.0
        if turn >= 20:
            return 0.3
        # Linear interpolation between (5, 1.0) and (20, 0.3)
        return 1.0 - 0.7 * (turn - 5) / 15.0

    if state.winner == 1 - perspective:
        return -1.0

    return 0.0
