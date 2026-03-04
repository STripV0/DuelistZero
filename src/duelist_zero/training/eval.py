"""
Evaluation utilities for measuring agent strength.

Provides:
- Win-rate evaluation against random / trained opponents
- ELO rating system for checkpoint tracking
- RecurrentAgentFn for LSTM-based agent evaluation
"""

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np


class RecurrentAgentFn:
    """
    Stateful wrapper for recurrent (LSTM) model inference.

    Tracks LSTM hidden state across steps within an episode. Call reset()
    at the start of each new episode to zero the LSTM state.

    Compatible with evaluate() — implements __call__(obs, mask) -> action
    and reset() for episode boundaries.
    """

    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic
        self.state = None
        self.episode_start = np.array([True])

    def reset(self):
        """Reset LSTM state for a new episode."""
        self.state = None
        self.episode_start = np.array([True])

    def __call__(self, obs, mask):
        action, self.state = self.model.predict(
            obs,
            state=self.state,
            episode_start=self.episode_start,
            action_masks=mask,
            deterministic=self.deterministic,
        )
        self.episode_start = np.array([False])
        return int(action)


def evaluate(env, agent_fn, n_episodes: int = 50) -> dict:
    """
    Evaluate an agent on the given env for n_episodes.

    Args:
        env: GoatEnv instance (opponent already configured)
        agent_fn: Callable(obs, mask) -> action_idx.
            If it has a reset() method (e.g. RecurrentAgentFn),
            it will be called at the start of each episode.
        n_episodes: Number of episodes to run

    Returns:
        dict with keys: win_rate, avg_reward, avg_steps, wins, losses, draws
    """
    wins = 0
    losses = 0
    draws = 0
    total_reward = 0.0
    total_steps = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        if hasattr(agent_fn, "reset"):
            agent_fn.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done and steps < 2000:
            mask = env.valid_action_mask()
            action = agent_fn(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated

        total_reward += ep_reward
        total_steps += steps

        if ep_reward > 0:
            wins += 1
        elif ep_reward < 0:
            losses += 1
        else:
            draws += 1

    return {
        "win_rate": wins / n_episodes,
        "avg_reward": total_reward / n_episodes,
        "avg_steps": total_steps / n_episodes,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "n_episodes": n_episodes,
    }


def expected_score(rating_a: float, rating_b: float) -> float:
    """ELO expected score for player A against player B."""
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def update_elo(
    rating_a: float,
    rating_b: float,
    score_a: float,
    k: float = 32.0,
) -> tuple[float, float]:
    """
    Update ELO ratings after a match.

    Args:
        rating_a: Player A's current rating
        rating_b: Player B's current rating
        score_a: Actual score for A (1.0=win, 0.5=draw, 0.0=loss)
        k: K-factor (higher = more volatile ratings)

    Returns:
        (new_rating_a, new_rating_b)
    """
    ea = expected_score(rating_a, rating_b)
    eb = 1.0 - ea
    score_b = 1.0 - score_a

    new_a = rating_a + k * (score_a - ea)
    new_b = rating_b + k * (score_b - eb)
    return new_a, new_b


class EloTracker:
    """
    Tracks ELO ratings for a pool of agent checkpoints.

    Ratings are persisted to a JSON file.
    """

    HEURISTIC_ID = "heuristic"
    RANDOM_ID = HEURISTIC_ID  # back-compat alias
    INITIAL_RATING = 1200.0

    def __init__(self, save_path: Optional[str | Path] = None):
        self.ratings: dict[str, float] = {
            self.RANDOM_ID: self.INITIAL_RATING,
        }
        self.history: list[dict] = []  # [{step, id, rating, opponent, result}, ...]
        self.save_path = Path(save_path) if save_path else None

        if self.save_path and self.save_path.exists():
            self._load()

    def get_rating(self, agent_id: str) -> float:
        return self.ratings.get(agent_id, self.INITIAL_RATING)

    def record_match(
        self,
        agent_id: str,
        opponent_id: str,
        win_rate: float,
        step: int = 0,
    ):
        """
        Record a match result and update ratings.

        Args:
            agent_id: ID of the agent being evaluated
            opponent_id: ID of the opponent
            win_rate: Agent's win rate (0.0 to 1.0)
            step: Training step count
        """
        if agent_id not in self.ratings:
            self.ratings[agent_id] = self.INITIAL_RATING
        if opponent_id not in self.ratings:
            self.ratings[opponent_id] = self.INITIAL_RATING

        new_a, new_b = update_elo(
            self.ratings[agent_id],
            self.ratings[opponent_id],
            win_rate,
        )
        self.ratings[agent_id] = new_a
        self.ratings[opponent_id] = new_b

        self.history.append({
            "step": step,
            "agent": agent_id,
            "opponent": opponent_id,
            "win_rate": round(win_rate, 3),
            "agent_elo": round(new_a, 1),
            "opponent_elo": round(new_b, 1),
        })

        if self.save_path:
            self._save()

    def get_summary(self) -> str:
        """Return a formatted string of current ratings."""
        sorted_ratings = sorted(
            self.ratings.items(), key=lambda x: x[1], reverse=True
        )
        lines = ["ELO Ratings:"]
        for agent_id, rating in sorted_ratings:
            lines.append(f"  {agent_id:>30s}: {rating:.0f}")
        return "\n".join(lines)

    def _save(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "ratings": self.ratings,
            "history": self.history,
        }
        self.save_path.write_text(json.dumps(data, indent=2))

    def _load(self):
        data = json.loads(self.save_path.read_text())
        self.ratings = data.get("ratings", {self.RANDOM_ID: self.INITIAL_RATING})
        self.history = data.get("history", [])
