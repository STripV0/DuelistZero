"""Debug a hanging game to find where it loops."""
import sys
import signal
import time
import numpy as np
from duelist_zero.env.goat_env import GoatEnv
from duelist_zero.env.action_space import ActionSpace

# Patch _advance to log
env = GoatEnv()
duel_obj = None

original_advance = env._advance.__func__

def debug_advance(self):
    """Instrumented _advance that logs opponent actions."""
    duel = self._duel
    opp_actions = 0
    while not duel.state.is_finished:
        msg = duel.process()
        if msg is None:
            return None
        player = getattr(msg, "player", 0)
        if player == self._agent_player:
            return msg
        else:
            opp_actions += 1
            if opp_actions <= 10 or opp_actions % 100 == 0:
                print(f"  opp_action #{opp_actions}: {type(msg).__name__} player={player}", flush=True)
            if opp_actions > 200:
                mask = self._action_space_handler.get_mask(msg)
                valid = np.where(mask)[0]
                if len(valid) > 0:
                    action = int(np.random.choice(valid))
                    self._action_space_handler.decode(action, msg, duel)
                else:
                    duel.respond_int(0)
            else:
                self._opponent_response(msg, duel)
    return None

# Monkey-patch
import types
env._advance = types.MethodType(debug_advance, env)

seed = 27
print(f"Testing seed={seed}", flush=True)

signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError()))
signal.alarm(15)

try:
    env.reset(seed=seed)
    print(f"  agent_player={env._agent_player}", flush=True)
    steps = 0
    done = False
    while not done and steps < 100:
        mask = env.valid_action_mask()
        action = int(np.random.choice(np.where(mask)[0]))
        print(f"\nStep {steps}: agent action={action}, msg={type(env._pending_msg).__name__ if env._pending_msg else 'None'}", flush=True)
        _, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
except TimeoutError:
    print("\nTIMEOUT - game hung!", flush=True)
finally:
    signal.alarm(0)
    env.close()
