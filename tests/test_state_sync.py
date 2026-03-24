"""Tests for engine state verification (A1).

Runs heuristic-vs-heuristic games and verifies that the Python GameState
matches the engine's internal state after every process() call.
"""

import numpy as np
import pytest

from duelist_zero.env.goat_env import GoatEnv


class TestStateSync:
    def test_verify_state_10_games(self):
        """Run 10 heuristic-vs-heuristic games, verify_state after every step."""
        env = GoatEnv()
        try:
            for seed in range(10):
                env.reset(seed=seed)
                done = False
                steps = 0
                while not done and steps < 300:
                    mask = env.valid_action_mask()
                    action = int(np.random.choice(np.where(mask)[0]))
                    _, _, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    steps += 1

                    # verify_state is called automatically in debug mode
                    # via GoatEnv.step(), but let's also call explicitly
                    if not done and env._duel is not None:
                        discrepancies = env._duel.verify_state()
                        assert len(discrepancies) == 0, (
                            f"Game {seed}, step {steps}: {discrepancies}"
                        )
        finally:
            env.close()

    def test_verify_state_empty_board(self):
        """verify_state on a fresh duel should find no discrepancies."""
        env = GoatEnv()
        try:
            env.reset(seed=42)
            if env._duel is not None:
                discrepancies = env._duel.verify_state()
                assert isinstance(discrepancies, list)
                # Fresh game may have some cards on field from draw/setup,
                # but should have no sync issues
                assert len(discrepancies) == 0
        finally:
            env.close()
