"""
Stable-Baselines3 callbacks for self-play training.
"""

import random
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .curriculum import CurriculumScheduler
from .eval import EloTracker, evaluate


class SelfPlayCallback(BaseCallback):
    """
    Callback that implements gated self-play during MaskablePPO training.

    Phase 1 (pre-training): Train vs heuristic until win rate exceeds
    `self_play_threshold` for `self_play_window` consecutive evals.

    Phase 2 (self-play): Mixed opponent — 70% current self, 15% heuristic,
    15% sampled past checkpoint. Prevents catastrophic forgetting by
    always retaining exposure to baseline opponents.
    """

    def __init__(
        self,
        checkpoint_interval: int = 50_000,
        save_dir: str = "checkpoints",
        eval_episodes: int = 20,
        curriculum: Optional[CurriculumScheduler] = None,
        self_play_threshold: float = 0.80,
        self_play_window: int = 3,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.checkpoint_interval = checkpoint_interval
        self.save_dir = Path(save_dir)
        self.eval_episodes = eval_episodes
        self.curriculum = curriculum
        self.self_play_threshold = self_play_threshold
        self.self_play_window = self_play_window

        # Checkpoint pool: list of (path, checkpoint_id)
        self.pool: list[tuple[Path, str]] = []
        self.elo = EloTracker(self.save_dir / "elo.json")
        self._next_checkpoint_step = checkpoint_interval

        # Self-play gating
        self._self_play_active = True
        self._recent_win_rates: list[float] = []

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_checkpoint_step:
            self._do_checkpoint()
            self._next_checkpoint_step += self.checkpoint_interval
        return True

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

        # 2. Evaluate vs heuristic (set opponent to None = heuristic)
        env = self._get_unwrapped_env()
        old_opponent = env._opponent_fn

        env.set_opponent(None)
        agent_fn = self._make_agent_fn()
        result = evaluate(env, agent_fn, n_episodes=self.eval_episodes)
        win_rate = result["win_rate"]
        self.elo.record_match(ckpt_id, EloTracker.RANDOM_ID, win_rate, step=step)

        if self.verbose:
            print(f"[Eval] vs Heuristic: {win_rate:.0%} "
                  f"({result['wins']}W/{result['losses']}L/{result['draws']}D)")

        # 3. Curriculum: record eval and check for advancement
        if self.curriculum is not None:
            self.curriculum.record_eval(win_rate, step)
            if self.curriculum.should_advance():
                new_stage = self.curriculum.advance()
                env.set_deck_pool(
                    self.curriculum.deck_pool, self.curriculum.deck_weights,
                )
                if self.verbose:
                    print(f"[Curriculum] Advanced to stage {new_stage}")
                    print(f"[Curriculum] {self.curriculum.stage_summary()}")
            self.curriculum.save_state(self.save_dir / "curriculum.json")

        # 4. Check if self-play should activate
        self._recent_win_rates.append(win_rate)
        if not self._self_play_active:
            if len(self._recent_win_rates) >= self.self_play_window:
                recent = self._recent_win_rates[-self.self_play_window :]
                if all(wr >= self.self_play_threshold for wr in recent):
                    self._self_play_active = True
                    if self.verbose:
                        print(
                            f"[SelfPlay] ACTIVATED — last {self.self_play_window} "
                            f"evals all >= {self.self_play_threshold:.0%}"
                        )

        # 5. Set training opponent
        if self._self_play_active and len(self.pool) >= 1:
            self_fn = self._make_opponent_fn(ckpt_path)
            if len(self.pool) > 1:
                past_path, past_id = self._sample_recent_opponent(exclude=ckpt_id)
                past_fn = self._make_opponent_fn(past_path)
            else:
                past_fn = None
                past_id = "heuristic"
            mixed_fn = self._make_mixed_opponent(self_fn, past_fn)
            env.set_opponent(mixed_fn)
            if self.verbose:
                print(f"[SelfPlay] Mixed opponent: 70% self + 15% heuristic + 15% {past_id}")
        else:
            # Pre-training phase: keep heuristic opponent
            env.set_opponent(old_opponent)

        if self.verbose:
            phase = "SELF-PLAY" if self._self_play_active else "PRE-TRAINING"
            print(f"[SelfPlay] Phase: {phase}")
            print(self.elo.get_summary())
            print()

    def _get_unwrapped_env(self):
        """Get the underlying GoatEnv from SB3's vectorized wrapper."""
        env = self.training_env.envs[0]
        while hasattr(env, "env"):
            env = env.env
        return env

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

    def _make_mixed_opponent(self, self_fn, past_fn):
        """
        Create mixed opponent: 70% current self, 15% heuristic, 15% past checkpoint.

        Per-decision mixing acts like an opponent with varying skill levels,
        similar to epsilon-greedy exploration on the opponent side.

        When past_fn is None (only 1 checkpoint exists), heuristic gets 30%.
        """
        from ..env.heuristic import heuristic_action

        def mixed_fn(obs, mask):
            r = np.random.random()
            if r < 0.15:
                # Heuristic — prevents forgetting how to beat basic play
                return heuristic_action(mask)
            elif r < 0.30:
                # Past checkpoint — retains knowledge of older strategies
                if past_fn is not None:
                    return past_fn(obs, mask)
                return heuristic_action(mask)
            else:
                # Current self — pushes the frontier
                return self_fn(obs, mask)

        return mixed_fn

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
