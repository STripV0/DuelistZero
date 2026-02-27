"""
Stable-Baselines3 callbacks for self-play training.
"""

import random
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from ..env.goat_env import GoatEnv
from .eval import EloTracker, evaluate


class SelfPlayCallback(BaseCallback):
    """
    Callback that implements gated self-play during MaskablePPO training.

    Phase 1 (pre-training): Train vs heuristic (diverse decks) until win rate
    exceeds `self_play_threshold` for `self_play_window` consecutive evals,
    or `heuristic_limit` steps reached.

    Phase 2 (self-play): Per-episode opponent roll — 60% heuristic (diverse
    deck), 20% recent frozen checkpoint (mirror), 20% older frozen checkpoint
    (mirror). Regression gate deactivates self-play if heuristic WR drops.

    Uses a dedicated eval env (main process) and broadcasts opponent
    changes to training envs via env_method().
    """

    def __init__(
        self,
        checkpoint_interval: int = 50_000,
        save_dir: str = "checkpoints",
        eval_episodes: int = 200,
        no_self_play: bool = False,
        self_play_threshold: float = 0.75,
        self_play_window: int = 3,
        regression_gate: float = 0.70,
        heuristic_limit: int = 5_000_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.checkpoint_interval = checkpoint_interval
        self.save_dir = Path(save_dir)
        self.eval_episodes = eval_episodes
        self.no_self_play = no_self_play
        self.self_play_threshold = self_play_threshold
        self.self_play_window = self_play_window
        self.regression_gate = regression_gate
        self.heuristic_limit = heuristic_limit

        # Checkpoint pool: list of (path, checkpoint_id)
        self.pool: list[tuple[Path, str]] = []
        self.elo = EloTracker(self.save_dir / "elo.json")
        self._next_checkpoint_step = checkpoint_interval

        # Self-play gating
        self._self_play_active = False
        self._recent_win_rates: list[float] = []

        # Dedicated eval env (created in _init_callback)
        self._eval_env: Optional[GoatEnv] = None

    def _init_callback(self) -> None:
        """Create the dedicated eval env in the main process."""
        self._eval_env = GoatEnv()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_checkpoint_step:
            self._do_checkpoint()
            self._next_checkpoint_step += self.checkpoint_interval
        return True

    def _on_training_end(self) -> None:
        """Clean up the eval env."""
        if self._eval_env is not None:
            self._eval_env.close()
            self._eval_env = None

    def _do_checkpoint(self):
        step = self.num_timesteps
        ckpt_id = f"ckpt_{step:08d}"
        ckpt_path = self.save_dir / ckpt_id

        # 1. Save checkpoint
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(ckpt_path))
        self.pool.append((ckpt_path, ckpt_id))

        if self.verbose:
            print(f"\n[SelfPlay] Checkpoint saved: {ckpt_id}")

        # 2. Evaluate vs heuristic on dedicated eval env
        env = self._eval_env
        env.set_opponent(None)
        agent_fn = self._make_agent_fn()
        result = evaluate(env, agent_fn, n_episodes=self.eval_episodes)
        win_rate = result["win_rate"]
        self.elo.record_match(ckpt_id, EloTracker.RANDOM_ID, win_rate, step=step)

        if self.verbose:
            print(f"[Eval] vs Heuristic: {win_rate:.0%} "
                  f"({result['wins']}W/{result['losses']}L/{result['draws']}D)")

        # 3. Evaluate vs past checkpoint (mirror match)
        if self._self_play_active and len(self.pool) >= 2:
            past_path, past_id = self._sample_recent_opponent(exclude=ckpt_id)
            past_fn = self._make_opponent_fn(past_path)
            env.set_opponent(past_fn)
            past_result = evaluate(env, agent_fn, n_episodes=self.eval_episodes)
            past_wr = past_result["win_rate"]
            self.elo.record_match(ckpt_id, past_id, past_wr, step=step)
            if self.verbose:
                print(f"[Eval] vs {past_id}: {past_wr:.0%} "
                      f"({past_result['wins']}W/{past_result['losses']}L/"
                      f"{past_result['draws']}D)")

        # 4. Self-play gating (activation and regression)
        self._recent_win_rates.append(win_rate)
        if not self.no_self_play:
            if not self._self_play_active:
                # Check if self-play should activate (threshold met)
                if len(self._recent_win_rates) >= self.self_play_window:
                    recent = self._recent_win_rates[-self.self_play_window :]
                    if all(wr >= self.self_play_threshold for wr in recent):
                        self._self_play_active = True
                        if self.verbose:
                            print(
                                f"[SelfPlay] ACTIVATED — last {self.self_play_window} "
                                f"evals all >= {self.self_play_threshold:.0%}"
                            )
                # Force-activate if heuristic phase has exceeded step limit
                if not self._self_play_active and step >= self.heuristic_limit:
                    self._self_play_active = True
                    if self.verbose:
                        print(
                            f"[SelfPlay] FORCE-ACTIVATED — heuristic limit "
                            f"({self.heuristic_limit:,} steps) reached"
                        )
            else:
                # Regression gate: deactivate if heuristic win rate drops too low
                if win_rate < self.regression_gate:
                    self._self_play_active = False
                    if self.verbose:
                        print(
                            f"[SelfPlay] DEACTIVATED — heuristic win rate "
                            f"{win_rate:.0%} < {self.regression_gate:.0%} gate"
                        )

        # 5. Set training opponent via broadcasting (frozen opponent pool)
        if self._self_play_active and len(self.pool) >= 2:
            # Stratified sampling: recent frozen + older frozen
            recent_path, recent_id = self._sample_recent_opponent(
                exclude=ckpt_id, window=5,
            )
            older_path, older_id = self._sample_full_pool(
                exclude={ckpt_id, recent_id},
            )

            self._broadcast_opponent(
                mode="mixed",
                model_path=str(recent_path),
                past_path=str(older_path),
            )
            if self.verbose:
                print(f"[SelfPlay] 60% heuristic (diverse) + 20% {recent_id} (mirror) + 20% {older_id} (mirror)")
        else:
            # Heuristic-only with diverse decks
            self._broadcast_opponent(mode="heuristic")

        if self.verbose:
            if self.no_self_play:
                phase = "HEURISTIC-ONLY"
            elif self._self_play_active:
                phase = "FROZEN-POOL SELF-PLAY"
            else:
                phase = "PRE-TRAINING"
            print(f"[SelfPlay] Phase: {phase}")
            print(self.elo.get_summary())
            print()

    def _broadcast_opponent(
        self,
        mode: str = "heuristic",
        model_path: Optional[str] = None,
        past_path: Optional[str] = None,
    ) -> None:
        """Set opponent on all training envs (works for both single and vec envs)."""
        venv = self.training_env
        if hasattr(venv, "env_method"):
            # SubprocVecEnv / VecEnv — broadcast to all subprocesses
            venv.env_method(
                "set_opponent_from_path",
                mode=mode,
                model_path=model_path,
                past_path=past_path,
            )
        else:
            # Single env (DummyVecEnv wrapping ActionMasker)
            env = venv.envs[0]
            while hasattr(env, "env"):
                env = env.env
            env.set_opponent_from_path(mode=mode, model_path=model_path, past_path=past_path)

    def _make_agent_fn(self):
        """Create an agent function from the current model."""
        model = self.model

        def agent_fn(obs, mask):
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            return int(action)

        return agent_fn

    def _make_opponent_fn(self, model_path: Path, deterministic: bool = False):
        """Create an opponent function from a saved checkpoint."""
        from sb3_contrib import MaskablePPO

        opp_model = MaskablePPO.load(str(model_path))

        def opponent_fn(obs, mask):
            action, _ = opp_model.predict(
                obs, action_masks=mask, deterministic=deterministic,
            )
            return int(action)

        return opponent_fn

    def _sample_recent_opponent(
        self, exclude: Optional[str] = None, window: int = 5,
    ) -> tuple[Path, str]:
        """Sample from the last N checkpoints only."""
        candidates = [
            (p, cid) for p, cid in self.pool[-window:] if cid != exclude
        ]
        if not candidates:
            candidates = list(self.pool)
        idx = np.random.choice(len(candidates))
        return candidates[idx]

    def _sample_full_pool(
        self, exclude: Optional[set[str]] = None,
    ) -> tuple[Path, str]:
        """Sample uniformly from the entire checkpoint pool."""
        if exclude is None:
            exclude = set()
        candidates = [
            (p, cid) for p, cid in self.pool if cid not in exclude
        ]
        if not candidates:
            candidates = list(self.pool)
        idx = np.random.choice(len(candidates))
        return candidates[idx]
