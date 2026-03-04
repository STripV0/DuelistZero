"""
Self-play training pipeline using MaskableRecurrentPPO.

Trains a YuGiOh GOAT format agent with LSTM memory via gated self-play:
1. Phase 1: Train against heuristic (diverse decks) until threshold or step limit
2. Phase 2: Per-episode mixed opponent — 40% heuristic (diverse deck),
   30% recent frozen checkpoint (mirror), 30% older frozen (mirror)

Usage:
    uv run python -m duelist_zero.training.self_play --timesteps 25000000 --n-envs 8
"""

import argparse
from pathlib import Path

import torch.nn as nn
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv

from ..env.goat_env import GoatEnv
from ..network.extractor import CardEmbeddingExtractor
from .callbacks import SelfPlayCallback
from .eval import RecurrentAgentFn, evaluate
from .maskable_recurrent_ppo import MaskableRecurrentPPO


def mask_fn(env):
    return env.valid_action_mask()


def _make_env(
    deck_pool: list[str] | None = None,
    deck_weights: list[float] | None = None,
) -> callable:
    """Return a factory function that creates a GoatEnv + ActionMasker.

    All arguments must be picklable (strings, lists of primitives, None)
    since SubprocVecEnv sends them through multiprocessing.Pipe.
    """
    def _init():
        env = GoatEnv(
            opponent_deck_pool=deck_pool,
            opponent_deck_weights=deck_weights,
        )
        return ActionMasker(env, mask_fn)
    return _init


def train(
    timesteps: int = 25_000_000,
    checkpoint_interval: int = 50_000,
    save_dir: str = "checkpoints",
    eval_episodes: int = 200,
    resume: str | None = None,
    n_steps: int = 512,
    batch_size: int = 128,
    learning_rate: float = 3e-5,
    n_epochs: int = 2,
    n_envs: int = 8,
    no_self_play: bool = False,
    self_play_threshold: float = 0.70,
    regression_gate: float = 0.60,
    heuristic_limit: int = 5_000_000,
    pretrained_embeddings: str | None = None,
    lstm_hidden_size: int = 256,
    n_lstm_layers: int = 1,
    verbose: int = 1,
):
    """
    Train a MaskableRecurrentPPO agent on GoatEnv with gated self-play.

    Args:
        timesteps: Total environment steps to train for
        checkpoint_interval: Steps between checkpoints and evals
        save_dir: Directory for checkpoints and logs
        eval_episodes: Episodes per evaluation round
        resume: Path to a checkpoint to resume from
        n_steps: PPO rollout buffer size (per env)
        batch_size: PPO minibatch size
        learning_rate: PPO learning rate
        n_epochs: PPO epochs per rollout
        n_envs: Number of parallel environments
        no_self_play: Disable self-play (train vs heuristic only)
        self_play_threshold: Win rate vs heuristic to activate self-play
        regression_gate: Deactivate self-play if heuristic win rate drops below this
        heuristic_limit: Force self-play activation after this many steps
        pretrained_embeddings: Path to .npy file with pretrained card embeddings
        lstm_hidden_size: LSTM hidden state dimension
        n_lstm_layers: Number of LSTM layers
        verbose: Verbosity level (0=silent, 1=progress)
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Discover all available decks for diverse opponent games
    deck_dir = Path(__file__).resolve().parents[3] / "data" / "deck"
    deck_files = sorted(deck_dir.glob("*.ydk"))
    deck_pool = [str(p) for p in deck_files] if len(deck_files) > 1 else None
    deck_weights = None  # uniform sampling

    if verbose and deck_pool:
        print(f"Opponent deck pool: {len(deck_files)} decks")
        for d in deck_files:
            print(f"  - {d.name}")

    # Get vocab_size from a temporary GoatEnv (can't access subprocess envs)
    tmp_env = GoatEnv()
    vocab_size = tmp_env._card_index.vocab_size
    tmp_env.close()

    # Create vectorized env
    if n_envs > 1:
        env = SubprocVecEnv([
            _make_env(deck_pool=deck_pool, deck_weights=deck_weights)
            for _ in range(n_envs)
        ])
    else:
        env = GoatEnv(
            opponent_deck_pool=deck_pool,
            opponent_deck_weights=deck_weights,
        )
        env = ActionMasker(env, mask_fn)

    # Create or load model
    if resume:
        if verbose:
            print(f"Resuming from checkpoint: {resume}")
        custom_objects = dict(
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
        )
        if verbose:
            print(f"  Injecting hyperparams: {custom_objects}")
        model = MaskableRecurrentPPO.load(
            resume, env=env, custom_objects=custom_objects,
        )
    else:
        # Only enable tensorboard if installed
        try:
            import tensorboard  # noqa: F401
            tb_log = str(save_path / "tb_logs")
        except ImportError:
            tb_log = None

        extractor_kwargs = dict(
            embed_dim=32,
            hidden_dim=512,
            vocab_size=vocab_size,
            d_model=64,
        )
        if pretrained_embeddings:
            extractor_kwargs["pretrained_embeddings_path"] = pretrained_embeddings

        policy_kwargs = dict(
            features_extractor_class=CardEmbeddingExtractor,
            features_extractor_kwargs=extractor_kwargs,
            net_arch=dict(pi=[512, 256], vf=[512, 256]),
            activation_fn=nn.ReLU,
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=n_lstm_layers,
            enable_critic_lstm=True,
            shared_lstm=False,
        )

        model = MaskableRecurrentPPO(
            "MultiInputLstmPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.99,
            learning_rate=learning_rate,
            ent_coef=0.10,
            clip_range=0.2,
            tensorboard_log=tb_log,
        )

    # Self-play callback
    sp_callback = SelfPlayCallback(
        checkpoint_interval=checkpoint_interval,
        save_dir=str(save_path),
        eval_episodes=eval_episodes,
        no_self_play=no_self_play,
        self_play_threshold=self_play_threshold,
        regression_gate=regression_gate,
        heuristic_limit=heuristic_limit,
        verbose=verbose,
    )

    if verbose:
        print("=" * 60)
        print("Duelist Zero — LSTM Self-Play Training")
        print("=" * 60)
        print(f"  Timesteps:           {timesteps:,}")
        print(f"  Parallel envs:       {n_envs}")
        print(f"  Checkpoint interval: {checkpoint_interval:,}")
        print(f"  Eval episodes:       {eval_episodes}")
        print(f"  Self-play threshold: {self_play_threshold:.0%}")
        print(f"  LSTM hidden size:    {lstm_hidden_size}")
        print(f"  LSTM layers:         {n_lstm_layers}")
        print(f"  n_steps (per env):   {n_steps}")
        print(f"  batch_size:          {batch_size}")
        print(f"  n_epochs:            {n_epochs}")
        print(f"  learning_rate:       {learning_rate}")
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

    # Final evaluation vs heuristic using a dedicated env
    if verbose:
        print("\nFinal evaluation vs heuristic (100 episodes)...")
        eval_env = GoatEnv()
        agent_fn = RecurrentAgentFn(model, deterministic=True)
        result = evaluate(eval_env, agent_fn, n_episodes=100)
        print(f"  Win rate: {result['win_rate']:.0%} "
              f"({result['wins']}W/{result['losses']}L/{result['draws']}D)")
        print(f"  Avg steps: {result['avg_steps']:.0f}")
        eval_env.close()

    env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MaskableRecurrentPPO agent with self-play on GOAT format YuGiOh"
    )
    parser.add_argument("--timesteps", type=int, default=25_000_000,
                        help="Total training timesteps")
    parser.add_argument("--checkpoint-interval", type=int, default=50_000,
                        help="Steps between checkpoints and evals")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Directory for checkpoints and logs")
    parser.add_argument("--eval-episodes", type=int, default=200,
                        help="Episodes per evaluation round")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path to resume from")
    parser.add_argument("--n-steps", type=int, default=512,
                        help="PPO rollout buffer size (per env)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="PPO minibatch size")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Learning rate")
    parser.add_argument("--n-epochs", type=int, default=2,
                        help="PPO epochs per rollout")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel environments")
    parser.add_argument("--no-self-play", action="store_true",
                        help="Disable self-play (train vs heuristic only)")
    parser.add_argument("--self-play-threshold", type=float, default=0.70,
                        help="Win rate vs heuristic to activate self-play")
    parser.add_argument("--regression-gate", type=float, default=0.60,
                        help="Deactivate self-play if heuristic win rate drops below this")
    parser.add_argument("--heuristic-limit", type=int, default=5_000_000,
                        help="Force self-play activation after this many steps vs heuristic")
    parser.add_argument("--pretrained-embeddings", type=str, default=None,
                        help="Path to .npy file with pretrained card embeddings")
    parser.add_argument("--lstm-hidden-size", type=int, default=256,
                        help="LSTM hidden state dimension")
    parser.add_argument("--n-lstm-layers", type=int, default=1,
                        help="Number of LSTM layers")
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
        n_epochs=args.n_epochs,
        n_envs=args.n_envs,
        no_self_play=args.no_self_play,
        self_play_threshold=args.self_play_threshold,
        regression_gate=args.regression_gate,
        heuristic_limit=args.heuristic_limit,
        pretrained_embeddings=args.pretrained_embeddings,
        lstm_hidden_size=args.lstm_hidden_size,
        n_lstm_layers=args.n_lstm_layers,
    )
