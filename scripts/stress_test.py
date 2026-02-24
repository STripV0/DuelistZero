"""Quick stress test to find game seeds that cause infinite loops."""
import sys
import time
import signal
import numpy as np
from duelist_zero.env.goat_env import GoatEnv


def timeout_handler(signum, frame):
    raise TimeoutError("Game took too long")


env = GoatEnv()
t_start = time.time()
slow = []
hung = []

for i in range(1000):
    t0 = time.time()
    env.reset(seed=i)
    steps = 0
    done = False

    # Set a 10-second alarm per game
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)
    try:
        while not done and steps < 2000:
            mask = env.valid_action_mask()
            action = int(np.random.choice(np.where(mask)[0]))
            _, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        signal.alarm(0)  # cancel alarm
    except TimeoutError:
        signal.alarm(0)
        elapsed = time.time() - t0
        hung.append((i, steps, elapsed))
        print(f"HUNG: seed={i} steps={steps} elapsed={elapsed:.1f}s", flush=True)
        # Force restart by creating a new env
        env.close()
        env = GoatEnv()
        continue

    elapsed = time.time() - t0
    if elapsed > 2:
        slow.append((i, steps, elapsed))
        print(f"SLOW: seed={i} steps={steps} elapsed={elapsed:.1f}s", flush=True)

    if (i + 1) % 100 == 0:
        print(f"  {i+1}/1000 done ({time.time()-t_start:.0f}s)", flush=True)

total = time.time() - t_start
print(f"\nDone: 1000 games in {total:.0f}s")
print(f"Hung: {len(hung)}, Slow: {len(slow)}")
for h in hung:
    print(f"  HUNG seed={h[0]} steps={h[1]}")
for s in slow:
    print(f"  SLOW seed={s[0]} steps={s[1]} elapsed={s[2]:.1f}s")

env.close()
