"""Debug: is duel.process() itself blocking, or the Python loop?"""
import signal
import time
import types
import numpy as np
from duelist_zero.env.goat_env import GoatEnv

signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError()))

env = GoatEnv()

# Patch _advance to time each process() call
def timed_advance(self):
    duel = self._duel
    iters = 0
    while not duel.state.is_finished:
        t0 = time.time()
        msg = duel.process()
        dt = time.time() - t0
        iters += 1
        if dt > 0.5:
            print(f"    SLOW process() call: {dt:.2f}s (iter {iters})", flush=True)
        if iters > 30:
            duel.state.is_finished = True
            duel.state.winner = -1
            print(f"    CAP HIT at iter {iters}", flush=True)
            return None
        if msg is None:
            return None
        player = getattr(msg, "player", 0)
        if player == self._agent_player:
            return msg
        else:
            self._opponent_response(msg, duel)
    return None

env._advance = types.MethodType(timed_advance, env)

# Run training-like loop
for game in range(200):
    signal.alarm(15)
    try:
        env.reset(seed=game * 7)  # different seeds
        steps = 0
        done = False
        while not done and steps < 500:
            mask = env.valid_action_mask()
            action = int(np.random.choice(np.where(mask)[0]))
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        signal.alarm(0)
    except TimeoutError:
        signal.alarm(0)
        print(f"HUNG game={game} seed={game*7} steps={steps}", flush=True)
        env.close()
        env = GoatEnv()
        env._advance = types.MethodType(timed_advance, env)
    if (game + 1) % 50 == 0:
        print(f"  {game+1}/200 done", flush=True)

env.close()
print("Done")
