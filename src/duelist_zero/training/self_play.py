"""
Self-play training pipeline using MaskablePPO from sb3-contrib.

Trains a YuGiOh GOAT format agent via gated self-play:
1. Phase 1: Train against heuristic until 80%+ win rate
2. Phase 2: Self-play with mixed opponents (70% self, 15% heuristic, 15% past)
3. Curriculum: start with mirror match, gradually add diverse decks

Usage:
    uv run python -m duelist_zero.training.self_play --timesteps 2000000
    uv run python -m duelist_zero.training.self_play --timesteps 500000 --no-curriculum
"""

import argparse
from pathlib import Path

import torch.nn as nn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from ..env.goat_env import GoatEnv
from ..network.extractor import CardEmbeddingExtractor
from .callbacks import SelfPlayCallback
from .curriculum import CurriculumScheduler
from .eval import evaluate


def mask_fn(env):
    return env.valid_action_mask()


def train(
    timesteps: int = 500_000,
    checkpoint_interval: int = 50_000,
    save_dir: str = "checkpoints",
    eval_episodes: int = 10,
    resume: str | None = None,
    n_steps: int = 2048,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    step_penalty: float = 0.001,
    mirror_ratio: float = 0.70,
    no_curriculum: bool = False,
    self_play_threshold: float = 0.80,
    verbose: int = 1,
):
    """
    Train a MaskablePPO agent on GoatEnv with gated self-play.

    Args:
        timesteps: Total environment steps to train for
        checkpoint_interval: Steps between checkpoints and evals
        save_dir: Directory for checkpoints and logs
        eval_episodes: Episodes per evaluation round
        resume: Path to a checkpoint to resume from
        n_steps: PPO rollout buffer size
        batch_size: PPO minibatch size
        learning_rate: PPO learning rate
        step_penalty: Per-step penalty (0.0 = disabled)
        mirror_ratio: Curriculum mirror match sampling weight
        no_curriculum: Disable curriculum (use all decks from start)
        self_play_threshold: Win rate vs heuristic to activate self-play
        verbose: Verbosity level (0=silent, 1=progress)
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Set up curriculum or discover full deck pool
    deck_dir = Path(__file__).resolve().parents[3] / "data" / "deck"
    curriculum = None

    if no_curriculum:
        # No curriculum: use all decks with equal weights
        deck_pool = sorted(deck_dir.glob("*.ydk"))
        if verbose and len(deck_pool) > 1:
            print(f"Opponent deck pool: {len(deck_pool)} decks")
            for d in deck_pool:
                print(f"  - {d.name}")
        env = GoatEnv(
            opponent_deck_pool=deck_pool if len(deck_pool) > 1 else None,
            step_penalty=step_penalty,
        )
    else:
        # Curriculum: start with mirror match, expand over time
        curriculum = CurriculumScheduler(
            deck_dir=deck_dir,
            mirror_ratio=mirror_ratio,
        )

        # Load curriculum state if resuming
        curriculum_path = save_path / "curriculum.json"
        if resume and curriculum_path.exists():
            curriculum.load_state(curriculum_path)
            if verbose:
                print(f"Resumed curriculum from {curriculum_path}")
                print(f"  {curriculum.stage_summary()}")

        pool = curriculum.deck_pool
        weights = curriculum.deck_weights
        if verbose:
            print(f"Curriculum enabled ({curriculum.max_stage + 1} stages)")
            print(f"  {curriculum.stage_summary()}")

        env = GoatEnv(
            opponent_deck_pool=pool,
            opponent_deck_weights=weights,
            step_penalty=step_penalty,
        )
    env = ActionMasker(env, mask_fn)

    # Create or load model
    if resume:
        if verbose:
            print(f"Resuming from checkpoint: {resume}")
        model = MaskablePPO.load(resume, env=env)
    else:
        # Only enable tensorboard if installed
        try:
            import tensorboard  # noqa: F401
            tb_log = str(save_path / "tb_logs")
        except ImportError:
            tb_log = None

        # Get vocab_size from the unwrapped GoatEnv
        unwrapped_env = env
        while hasattr(unwrapped_env, "env"):
            unwrapped_env = unwrapped_env.env
        vocab_size = unwrapped_env._card_index.vocab_size

        policy_kwargs = dict(
            features_extractor_class=CardEmbeddingExtractor,
            features_extractor_kwargs=dict(
                embed_dim=32,
                hidden_dim=256,
                vocab_size=vocab_size,
            ),
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=nn.ReLU,
        )

        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            learning_rate=learning_rate,
            ent_coef=0.05,
            clip_range=0.2,
            tensorboard_log=tb_log,
        )

    # Self-play callback
    sp_callback = SelfPlayCallback(
        checkpoint_interval=checkpoint_interval,
        save_dir=str(save_path),
        eval_episodes=eval_episodes,
        curriculum=curriculum,
        self_play_threshold=self_play_threshold,
        verbose=verbose,
    )

    if verbose:
        print("=" * 60)
        print("Duelist Zero — Self-Play Training")
        print("=" * 60)
        print(f"  Timesteps:           {timesteps:,}")
        print(f"  Checkpoint interval: {checkpoint_interval:,}")
        print(f"  Eval episodes:       {eval_episodes}")
        print(f"  Self-play threshold: {self_play_threshold:.0%}")
        print(f"  Save directory:      {save_path}")
        print(f"  Device:              {model.device}")
        print("=" * 60)
        print()

    # Train
    model.learn(
        total_timesteps=timesteps,
        callback=sp_callback,
    )

    # Final save
    final_path = save_path / "final_model"
    model.save(str(final_path))

    if verbose:
        print()
        print("=" * 60)
        print("Training complete!")
        print(f"  Final model: {final_path}")
        print(f"  Checkpoints: {len(sp_callback.pool)}")
        print()
        print(sp_callback.elo.get_summary())
        print("=" * 60)

    # Final evaluation vs heuristic
    if verbose:
        print("\nFinal evaluation vs heuristic (100 episodes)...")
        unwrapped = env.envs[0] if hasattr(env, "envs") else env
        while hasattr(unwrapped, "env"):
            unwrapped = unwrapped.env
        unwrapped.set_opponent(None)

        def agent_fn(obs, mask):
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            return int(action)

        result = evaluate(unwrapped, agent_fn, n_episodes=100)
        print(f"  Win rate: {result['win_rate']:.0%} "
              f"({result['wins']}W/{result['losses']}L/{result['draws']}D)")
        print(f"  Avg steps: {result['avg_steps']:.0f}")

    env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MaskablePPO agent with self-play on GOAT format YuGiOh"
    )
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--checkpoint-interval", type=int, default=50_000,
                        help="Steps between checkpoints and evals")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Directory for checkpoints and logs")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Episodes per evaluation round")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path to resume from")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="PPO rollout buffer size")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="PPO minibatch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--step-penalty", type=float, default=0.001,
                        help="Per-step penalty (0.0 = disabled)")
    parser.add_argument("--mirror-ratio", type=float, default=0.70,
                        help="Curriculum: mirror match sampling weight")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable curriculum learning (use all decks from start)")
    parser.add_argument("--self-play-threshold", type=float, default=0.80,
                        help="Win rate vs heuristic to activate self-play")
    args = parser.parse_args()

    train(
        timesteps=args.timesteps,
        checkpoint_interval=args.checkpoint_interval,
        save_dir=args.save_dir,
        eval_episodes=args.eval_episodes,
        resume=args.resume,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        step_penalty=args.step_penalty,
        mirror_ratio=args.mirror_ratio,
        no_curriculum=args.no_curriculum,
        self_play_threshold=args.self_play_threshold,
    )
