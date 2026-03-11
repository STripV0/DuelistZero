"""
Tests for the GoatEnv Gymnasium environment.
"""

import numpy as np
import pytest

from duelist_zero.env.goat_env import GoatEnv
from duelist_zero.env.observation import (
    OBSERVATION_DIM, CARD_ID_DIM,
    ACTION_FEATURES_SLOTS, ACTION_FEATURES_DIM,
    HISTORY_LENGTH, HISTORY_FEATURES_DIM,
)
from duelist_zero.env.action_space import ACTION_DIM


@pytest.fixture(scope="module")
def env():
    """Create a single env instance shared across tests in this module."""
    e = GoatEnv()
    yield e
    e.close()


class TestReset:
    def test_reset_returns_valid_obs(self, env):
        obs, info = env.reset(seed=42)
        assert isinstance(obs, dict), f"Expected dict obs, got {type(obs)}"
        assert "features" in obs and "card_ids" in obs
        assert "action_features" in obs and "action_history" in obs
        assert obs["features"].shape == (OBSERVATION_DIM,)
        assert obs["features"].dtype == np.float32
        assert np.all(obs["features"] >= -1.0), "Features has values < -1"
        assert np.all(obs["features"] <= 1.0), "Features has values > 1"
        assert obs["card_ids"].shape == (CARD_ID_DIM,)
        assert obs["card_ids"].dtype == np.float32
        assert np.all(obs["card_ids"] >= 0.0), "Card IDs has values < 0"
        assert obs["action_features"].shape == (ACTION_FEATURES_SLOTS, ACTION_FEATURES_DIM)
        assert obs["action_features"].dtype == np.float32
        assert obs["action_history"].shape == (HISTORY_LENGTH, HISTORY_FEATURES_DIM)
        assert obs["action_history"].dtype == np.float32

    def test_reset_returns_info_dict(self, env):
        _, info = env.reset(seed=1)
        assert isinstance(info, dict)
        assert "turn" in info
        assert "agent_lp" in info
        assert "opp_lp" in info

    def test_reset_twice_works(self, env):
        obs1, _ = env.reset(seed=10)
        obs2, _ = env.reset(seed=20)
        # Different seeds → different observations (usually)
        assert obs1["features"].shape == obs2["features"].shape == (OBSERVATION_DIM,)
        assert obs1["card_ids"].shape == obs2["card_ids"].shape == (CARD_ID_DIM,)
        assert obs1["action_features"].shape == obs2["action_features"].shape == (ACTION_FEATURES_SLOTS, ACTION_FEATURES_DIM)
        assert obs1["action_history"].shape == obs2["action_history"].shape == (HISTORY_LENGTH, HISTORY_FEATURES_DIM)


class TestActionMask:
    def test_mask_shape(self, env):
        env.reset(seed=0)
        mask = env.valid_action_mask()
        assert mask.shape == (ACTION_DIM,), f"Expected ({ACTION_DIM},), got {mask.shape}"
        assert mask.dtype == bool

    def test_mask_has_valid_actions(self, env):
        env.reset(seed=0)
        mask = env.valid_action_mask()
        assert mask.any(), "No valid actions in initial state"

    def test_mask_count_reasonable(self, env):
        env.reset(seed=0)
        mask = env.valid_action_mask()
        n_valid = mask.sum()
        assert 1 <= n_valid <= ACTION_DIM, f"Unexpected valid action count: {n_valid}"


class TestStep:
    def test_step_random_masked_action(self, env):
        env.reset(seed=99)
        mask = env.valid_action_mask()
        action = int(np.random.choice(np.where(mask)[0]))
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, dict)
        assert obs["features"].shape == (OBSERVATION_DIM,)
        assert obs["features"].dtype == np.float32
        assert obs["card_ids"].shape == (CARD_ID_DIM,)
        assert obs["action_features"].shape == (ACTION_FEATURES_SLOTS, ACTION_FEATURES_DIM)
        assert obs["action_history"].shape == (HISTORY_LENGTH, HISTORY_FEATURES_DIM)
        assert -1.0 <= reward <= 1.0, f"Unexpected reward: {reward}"
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_obs_in_range(self, env):
        env.reset(seed=7)
        mask = env.valid_action_mask()
        action = int(np.random.choice(np.where(mask)[0]))
        obs, _, _, _, _ = env.step(action)
        assert np.all(obs["features"] >= -1.0)
        assert np.all(obs["features"] <= 1.0)
        assert np.all(obs["card_ids"] >= 0.0)

    def test_step_until_done(self, env):
        """Play a full game with random masked actions."""
        env.reset(seed=123)
        done = False
        steps = 0
        while not done and steps < 300:
            mask = env.valid_action_mask()
            action = int(np.random.choice(np.where(mask)[0]))
            _, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        assert done, f"Game did not finish in {steps} steps"
        assert -1.0 <= reward <= 1.0, f"Terminal reward out of range: {reward}"


class TestStressTest:
    def test_100_random_duels(self):
        """Run 100 complete duels with random masked actions. All must terminate."""
        env = GoatEnv()
        try:
            results = []
            for i in range(100):
                env.reset(seed=i)
                done = False
                steps = 0
                reward = 0.0
                while not done and steps < 300:
                    mask = env.valid_action_mask()
                    action = int(np.random.choice(np.where(mask)[0]))
                    _, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    steps += 1
                results.append((done, reward, steps))

            # All duels must have terminated
            assert all(r[0] for r in results), "Some duels did not terminate"
            # All terminal rewards must be in valid range
            assert all(-1.0 <= r[1] <= 1.0 for r in results), "Terminal rewards out of range"
            # Average steps should be reasonable (not all truncated)
            avg_steps = sum(r[2] for r in results) / len(results)
            assert avg_steps < 450, f"Average steps too high ({avg_steps:.0f}), likely truncating"
        finally:
            env.close()

    def test_100_duels_with_pbrs(self):
        """Run 100 complete duels with PBRS shaping enabled."""
        env = GoatEnv(shaping_scale=0.5)
        try:
            results = []
            for i in range(100):
                env.reset(seed=i)
                done = False
                steps = 0
                reward = 0.0
                while not done and steps < 300:
                    mask = env.valid_action_mask()
                    action = int(np.random.choice(np.where(mask)[0]))
                    _, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    steps += 1
                results.append((done, reward, steps))

            assert all(r[0] for r in results), "Some duels did not terminate"
            # With PBRS, terminal rewards can include shaping component
            assert all(-2.0 <= r[1] <= 2.0 for r in results), "Terminal rewards out of range"
            avg_steps = sum(r[2] for r in results) / len(results)
            assert avg_steps < 450, f"Average steps too high ({avg_steps:.0f}), likely truncating"
        finally:
            env.close()
