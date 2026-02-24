"""
Random agent — picks uniformly from valid actions.

Useful as a baseline and for testing the environment.
"""

import numpy as np
from ..env.goat_env import GoatEnv
from ..env.action_space import ACTION_DIM


class RandomAgent:
    """
    Agent that picks uniformly at random from valid (masked) actions.
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        """
        Choose a random valid action.

        Args:
            obs: Current observation (unused by random agent)
            mask: Boolean mask of valid actions

        Returns:
            Action index
        """
        valid = np.where(mask)[0]
        if len(valid) == 0:
            return 0
        return int(self.rng.choice(valid))

    def run_episode(self, env: GoatEnv) -> tuple[float, int]:
        """
        Run a single episode with random actions.

        Returns:
            (total_reward, steps)
        """
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            mask = env.valid_action_mask()
            action = self.act(obs, mask)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        return total_reward, steps
