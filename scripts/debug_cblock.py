"""Verify: is duel.process() blocking in C++?"""
import signal
import time
import random
import numpy as np
from duelist_zero.env.goat_env import GoatEnv

env = GoatEnv()

# Run games and time each process() call directly
for game in range(50):
    env.reset(seed=game)
    steps = 0
    done = False
    t_game = time.time()

    while not done and steps < 500:
        # Time the entire step
        t_step = time.time()
        mask = env.valid_action_mask()
        action = int(np.random.choice(np.where(mask)[0]))
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step_dt = time.time() - t_step
        steps += 1

        if step_dt > 1.0:
            print(f"  game={game} step={steps} took {step_dt:.2f}s", flush=True)

    game_dt = time.time() - t_game
    if game_dt > 5:
        print(f"SLOW game={game} seed={game} steps={steps} time={game_dt:.1f}s", flush=True)
    if (game + 1) % 10 == 0:
        print(f"  {game+1}/50 done", flush=True)

env.close()
print("Done")
