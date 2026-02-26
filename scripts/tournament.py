"""Round robin tournament between checkpoint candidates."""

from sb3_contrib import MaskablePPO
from duelist_zero.env.goat_env import GoatEnv
import numpy as np
import sys

candidates = {
    "2250k": "checkpoints/ckpt_02250000",
    "2750k": "checkpoints/ckpt_02750000",
    "3850k": "checkpoints/ckpt_03850000",
    "final": "checkpoints/final_model",
}

N = 200  # games per matchup

# Load all models
models = {}
for name, path in candidates.items():
    models[name] = MaskablePPO.load(path)
    print(f"Loaded {name} from {path}")

env = GoatEnv()


def play_match(env, model_a, model_b, n_games):
    wins_a, wins_b, draws = 0, 0, 0
    for g in range(n_games):
        def opp_fn(obs, mask, _m=model_b):
            action, _ = _m.predict(obs, action_masks=mask, deterministic=True)
            return int(action)

        env.set_opponent(opp_fn)
        obs, _ = env.reset(seed=g + 10000)
        done = False
        steps = 0
        while not done and steps < 300:
            mask = env.valid_action_mask()
            action, _ = model_a.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            steps += 1
        if reward > 0:
            wins_a += 1
        elif reward < 0:
            wins_b += 1
        else:
            draws += 1
    return wins_a, wins_b, draws


names = list(models.keys())
total_wins = {n: 0 for n in names}
total_games = {n: 0 for n in names}

print(f"\nRound robin: {N} games per matchup ({len(names)} agents, {len(names)*(len(names)-1)//2} matchups)\n")

for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        print(f"  {a} vs {b}...", end=" ", flush=True)
        wa, wb, d = play_match(env, models[a], models[b], N)
        total_wins[a] += wa
        total_wins[b] += wb
        total_games[a] += N
        total_games[b] += N
        print(f"{wa}W/{wb}L/{d}D  ({wa/N:.0%})")

env.close()

print()
print("=" * 40)
print(f"{'Agent':>8s}  {'Wins':>5s}  {'Games':>5s}  {'WR':>6s}")
print("-" * 40)
for n in sorted(names, key=lambda x: -total_wins[x]):
    wr = total_wins[n] / total_games[n]
    print(f"{n:>8s}  {total_wins[n]:5d}  {total_games[n]:5d}  {wr:5.0%}")
print("=" * 40)
