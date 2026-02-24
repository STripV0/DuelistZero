"""Identify if C++ process() or Python loop is the blocker."""
import signal
import time
import types
import numpy as np
from duelist_zero.engine.duel import Duel
from duelist_zero.env.goat_env import GoatEnv
from duelist_zero.core.constants import PROCESSOR_END, PROCESSOR_WAITING
from duelist_zero.core.message_parser import RESPONSE_MESSAGES

# Patch Duel.process to time and count C++ calls
call_count = 0
max_single_call = 0.0

original_process = Duel.process

def timed_process(self):
    global call_count, max_single_call
    deadline = time.monotonic() + 2.0
    while True:
        call_count += 1
        t0 = time.monotonic()
        result = self.core.process(self._pduel)
        dt = time.monotonic() - t0
        if dt > max_single_call:
            max_single_call = dt
        if dt > 0.1:
            print(f"    C++ process() took {dt:.3f}s (call #{call_count})", flush=True)

        if time.monotonic() > deadline:
            print(f"    TIMEOUT in Duel.process() after {call_count} C++ calls", flush=True)
            self.state.is_finished = True
            return None

        msg_data = self.core.get_message(self._pduel)
        if not msg_data:
            flag = (result >> 16) & 0xFF
            if flag & PROCESSOR_END:
                self.state.is_finished = True
                return None
            continue

        messages = self.parser.parse(msg_data)
        for msg in messages:
            self._update_state(msg)
            if msg.msg_type in RESPONSE_MESSAGES:
                self._pending_decision = msg
                return msg

        flag = (result >> 16) & 0xFF
        if flag & PROCESSOR_END:
            self.state.is_finished = True
            return None
        if flag & PROCESSOR_WAITING:
            continue

Duel.process = timed_process

env = GoatEnv()
for game in range(50):
    call_count = 0
    max_single_call = 0.0
    t0 = time.time()
    env.reset(seed=game)
    steps = 0
    done = False
    while not done and steps < 500:
        mask = env.valid_action_mask()
        action = int(np.random.choice(np.where(mask)[0]))
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
    dt = time.time() - t0
    if dt > 3:
        print(f"SLOW game={game} steps={steps} time={dt:.1f}s calls={call_count} max_call={max_single_call:.3f}s", flush=True)
    if (game + 1) % 10 == 0:
        print(f"  {game+1}/50 done", flush=True)

env.close()
print("Done")
