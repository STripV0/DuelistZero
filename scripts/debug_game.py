#!/usr/bin/env python3
"""Debug: trace a single game to find why player 0 always loses."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
from duelist_zero.env.goat_env import GoatEnv
from duelist_zero.core.message_parser import MsgWin, MsgNewTurn, MSG
from duelist_zero.core.constants import PROCESSOR_END, PROCESSOR_WAITING
from duelist_zero.engine.duel import Duel

_orig_process = Duel.process

def traced_process(self):
    while True:
        result = self.core.process(self._pduel)
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
            if isinstance(msg, MsgWin):
                reason_names = {0: "???", 1: "LP=0", 2: "Deck-out", 4: "Exodia"}
                print(f"  WIN: Player {msg.player} by {reason_names.get(msg.reason, msg.reason)}", flush=True)
            if isinstance(msg, MsgNewTurn):
                p0 = self.state.players[0]
                p1 = self.state.players[1]
                print(f"  Turn -> P{msg.player} | "
                      f"P0: LP={p0.lp} Dk={p0.deck_count} Hd={p0.hand_count} | "
                      f"P1: LP={p1.lp} Dk={p1.deck_count} Hd={p1.hand_count}", flush=True)

            if msg.msg_type in {MSG.SELECT_IDLECMD, MSG.SELECT_BATTLECMD, MSG.SELECT_CARD,
                                MSG.SELECT_CHAIN, MSG.SELECT_EFFECTYN, MSG.SELECT_YESNO,
                                MSG.SELECT_OPTION, MSG.SELECT_POSITION, MSG.SELECT_PLACE,
                                MSG.SELECT_TRIBUTE}:
                self._pending_decision = msg
                return msg

        flag = (result >> 16) & 0xFF
        if flag & PROCESSOR_END:
            self.state.is_finished = True
            return None
        if flag & PROCESSOR_WAITING:
            continue

Duel.process = traced_process

env = GoatEnv()
obs, info = env.reset()
done = False
step = 0

while not done and step < 2000:
    mask = env.valid_action_mask()
    valid = np.where(mask)[0]
    action = int(np.random.choice(valid))
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1
    done = terminated or truncated

state = env._duel.state if env._duel else None
print(f"\nFinal: winner={state.winner} LP={state.players[0].lp}/{state.players[1].lp} "
      f"Deck={state.players[0].deck_count}/{state.players[1].deck_count} steps={step}", flush=True)
env.close()
