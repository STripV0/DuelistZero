"""
Unified flat action space for GoatEnv.

All possible decisions the agent can make are mapped to a single
integer index in [0, ACTION_DIM). The `ActionSpace` class provides:
  - get_mask(msg)  → bool ndarray of shape (ACTION_DIM,)
  - decode(idx, msg, duel) → sends the response to the engine

Action layout:
  [0-4]    Idle: summon card 0-4
  [5-9]    Idle: special summon card 0-4
  [10-14]  Idle: set monster 0-4
  [15-19]  Idle: set spell/trap 0-4
  [20-29]  Idle: activate card 0-9
  [30-34]  Idle: reposition monster 0-4
  [35]     Idle: go to Battle Phase
  [36]     Idle: go to End Phase
  [37-41]  Battle: attack with monster 0-4
  [42-46]  Battle: activate card 0-4
  [47]     Battle: go to Main Phase 2
  [48]     Battle: go to End Phase
  [49]     Chain/YesNo/EffectYN: Yes (or select card 0)
  [50]     Chain/YesNo/EffectYN: No (or pass)
  [51-60]  Select card: indices 0-9
  [61]     Select position: ATK
  [62]     Select position: DEF
  [63-70]  Select option: 0-7

Total: 71 actions
"""

import struct
from typing import Optional

import numpy as np

from ..core.constants import LOCATION, POSITION, MSG
from ..core.message_parser import (
    ParsedMessage,
    MsgSelectIdleCmd,
    MsgSelectBattleCmd,
    MsgSelectCard,
    MsgSelectChain,
    MsgSelectEffectYn,
    MsgSelectYesNo,
    MsgSelectOption,
    MsgSelectPosition,
    MsgSelectPlace,
    MsgSelectTribute,
    MsgAnnounceNumber,
    MsgAnnounceRace,
)


# ============================================================
# Action index ranges (inclusive start, exclusive end)
# ============================================================
_IDLE_SUMMON_START = 0       # 0-4   summon card idx 0-4
_IDLE_SPSUMMON_START = 5     # 5-9   special summon card idx 0-4
_IDLE_SET_MON_START = 10     # 10-14 set monster idx 0-4
_IDLE_SET_ST_START = 15      # 15-19 set spell/trap idx 0-4
_IDLE_ACTIVATE_START = 20    # 20-29 activate card idx 0-9
_IDLE_REPOSITION_START = 30  # 30-34 reposition idx 0-4
_IDLE_BATTLE = 35            # go to battle phase
_IDLE_END = 36               # go to end phase

_BATTLE_ATTACK_START = 37    # 37-41 attack with monster idx 0-4
_BATTLE_ACTIVATE_START = 42  # 42-46 activate card idx 0-4
_BATTLE_MAIN2 = 47           # go to main phase 2
_BATTLE_END = 48             # go to end phase

_YESNO_YES = 49              # yes / select first card
_YESNO_NO = 50               # no / pass / cancel

_SELECT_CARD_START = 51      # 51-60 select card idx 0-9

_POS_ATK = 61                # select ATK position
_POS_DEF = 62                # select DEF position

_OPTION_START = 63           # 63-70 select option 0-7

ACTION_DIM = 71


class ActionSpace:
    """
    Maps flat action indices to engine responses.

    Usage:
        space = ActionSpace()
        mask = space.get_mask(msg)
        action = np.random.choice(np.where(mask)[0])
        space.decode(action, msg, duel)
    """

    def get_mask(self, msg: ParsedMessage) -> np.ndarray:
        """
        Return a boolean mask of valid actions for the current message.
        """
        mask = np.zeros(ACTION_DIM, dtype=bool)

        if isinstance(msg, MsgSelectIdleCmd):
            # Summon
            for i, _ in enumerate(msg.summonable[:5]):
                mask[_IDLE_SUMMON_START + i] = True
            # Special summon
            for i, _ in enumerate(msg.spsummonable[:5]):
                mask[_IDLE_SPSUMMON_START + i] = True
            # Set monster
            for i, _ in enumerate(msg.setable_monsters[:5]):
                mask[_IDLE_SET_MON_START + i] = True
            # Set spell/trap
            for i, _ in enumerate(msg.setable_st[:5]):
                mask[_IDLE_SET_ST_START + i] = True
            # Activate
            for i, _ in enumerate(msg.activatable[:10]):
                mask[_IDLE_ACTIVATE_START + i] = True
            # Reposition
            for i, _ in enumerate(msg.repositionable[:5]):
                mask[_IDLE_REPOSITION_START + i] = True
            # Phase transitions
            if msg.can_battle_phase:
                mask[_IDLE_BATTLE] = True
            if msg.can_end_phase:
                mask[_IDLE_END] = True

        elif isinstance(msg, MsgSelectBattleCmd):
            # Attack
            for i, _ in enumerate(msg.attackable[:5]):
                mask[_BATTLE_ATTACK_START + i] = True
            # Activate
            for i, _ in enumerate(msg.activatable[:5]):
                mask[_BATTLE_ACTIVATE_START + i] = True
            if msg.can_main2:
                mask[_BATTLE_MAIN2] = True
            if msg.can_end_phase:
                mask[_BATTLE_END] = True

        elif isinstance(msg, (MsgSelectEffectYn, MsgSelectYesNo)):
            mask[_YESNO_YES] = True
            mask[_YESNO_NO] = True

        elif isinstance(msg, MsgSelectChain):
            if not msg.forced:
                mask[_YESNO_NO] = True  # pass / don't chain
            for i, _ in enumerate(msg.cards[:10]):
                mask[_SELECT_CARD_START + i] = True

        elif isinstance(msg, (MsgSelectCard, MsgSelectTribute)):
            for i, _ in enumerate(msg.cards[:10]):
                mask[_SELECT_CARD_START + i] = True
            if msg.cancelable:
                mask[_YESNO_NO] = True

        elif isinstance(msg, MsgSelectPosition):
            allowed = msg.positions
            if allowed & POSITION.FACEUP_ATTACK:
                mask[_POS_ATK] = True
            if allowed & (POSITION.FACEUP_DEFENSE | POSITION.FACEDOWN_DEFENSE):
                mask[_POS_DEF] = True

        elif isinstance(msg, MsgSelectOption):
            for i, _ in enumerate(msg.options[:8]):
                mask[_OPTION_START + i] = True

        elif isinstance(msg, MsgAnnounceNumber):
            for i, _ in enumerate(msg.options[:8]):
                mask[_OPTION_START + i] = True

        elif isinstance(msg, MsgSelectPlace):
            # SELECT_PLACE is handled internally (random valid zone)
            # We expose a single "confirm" action
            mask[_YESNO_YES] = True

        elif isinstance(msg, MsgAnnounceRace):
            # Auto-handled: pick first valid race(s)
            mask[_YESNO_YES] = True

        # Fallback: always allow YES if nothing else is valid
        if not mask.any():
            mask[_YESNO_YES] = True

        return mask

    def decode(self, action_idx: int, msg: ParsedMessage, duel) -> None:
        """
        Decode action_idx and send the appropriate response to the engine
        via the duel object.

        Args:
            action_idx: Chosen action index
            msg: The pending decision message
            duel: Duel instance (for respond_int / respond_bytes)
        """
        if isinstance(msg, MsgSelectIdleCmd):
            self._decode_idle(action_idx, msg, duel)

        elif isinstance(msg, MsgSelectBattleCmd):
            self._decode_battle(action_idx, msg, duel)

        elif isinstance(msg, MsgSelectEffectYn):
            duel.respond_int(1 if action_idx == _YESNO_YES else 0)

        elif isinstance(msg, MsgSelectYesNo):
            duel.respond_int(1 if action_idx == _YESNO_YES else 0)

        elif isinstance(msg, MsgSelectChain):
            if action_idx == _YESNO_NO:
                duel.respond_int(-1)  # pass
            else:
                card_idx = action_idx - _SELECT_CARD_START
                card_idx = max(0, min(card_idx, len(msg.cards) - 1))
                duel.respond_int(card_idx)

        elif isinstance(msg, (MsgSelectCard, MsgSelectTribute)):
            if action_idx == _YESNO_NO and msg.cancelable:
                duel.respond_int(-1)
            else:
                card_idx = action_idx - _SELECT_CARD_START
                card_idx = max(0, min(card_idx, len(msg.cards) - 1))
                # Response: count(1) + indices(1 each)
                needed = max(msg.min_count, 1)
                # Pick `needed` cards starting from card_idx
                indices = []
                for i in range(needed):
                    indices.append((card_idx + i) % len(msg.cards))
                duel.respond_card_selection(indices)

        elif isinstance(msg, MsgSelectPosition):
            if action_idx == _POS_ATK:
                duel.respond_int(POSITION.FACEUP_ATTACK)
            else:
                # Prefer face-up defense, fall back to face-down
                if msg.positions & POSITION.FACEUP_DEFENSE:
                    duel.respond_int(POSITION.FACEUP_DEFENSE)
                else:
                    duel.respond_int(POSITION.FACEDOWN_DEFENSE)

        elif isinstance(msg, MsgSelectOption):
            opt_idx = action_idx - _OPTION_START
            opt_idx = max(0, min(opt_idx, len(msg.options) - 1))
            duel.respond_int(opt_idx)

        elif isinstance(msg, MsgAnnounceNumber):
            opt_idx = action_idx - _OPTION_START
            opt_idx = max(0, min(opt_idx, len(msg.options) - 1))
            duel.respond_int(opt_idx)

        elif isinstance(msg, MsgAnnounceRace):
            _respond_announce_race(msg, duel)

        elif isinstance(msg, MsgSelectPlace):
            # Handled by the env directly (random valid zone)
            _respond_select_place(msg, duel)

        else:
            # Fallback
            duel.respond_int(0)

    # ============================================================
    # Idle command decoding
    # ============================================================
    def _decode_idle(self, action_idx: int, msg: MsgSelectIdleCmd, duel) -> None:
        """
        Idle command response encoding:
          response = category | (index << 16)
        Categories (from ygopro-core):
          0 = summon, 1 = spsummon, 2 = reposition, 3 = set monster,
          4 = set spell/trap, 5 = activate, 6 = to battle phase,
          7 = to end phase
        """
        if _IDLE_SUMMON_START <= action_idx < _IDLE_SUMMON_START + 5:
            idx = action_idx - _IDLE_SUMMON_START
            idx = min(idx, len(msg.summonable) - 1)
            duel.respond_int(_encode_idle(0, idx))

        elif _IDLE_SPSUMMON_START <= action_idx < _IDLE_SPSUMMON_START + 5:
            idx = action_idx - _IDLE_SPSUMMON_START
            idx = min(idx, len(msg.spsummonable) - 1)
            duel.respond_int(_encode_idle(1, idx))

        elif _IDLE_SET_MON_START <= action_idx < _IDLE_SET_MON_START + 5:
            idx = action_idx - _IDLE_SET_MON_START
            idx = min(idx, len(msg.setable_monsters) - 1)
            duel.respond_int(_encode_idle(3, idx))

        elif _IDLE_SET_ST_START <= action_idx < _IDLE_SET_ST_START + 5:
            idx = action_idx - _IDLE_SET_ST_START
            idx = min(idx, len(msg.setable_st) - 1)
            duel.respond_int(_encode_idle(4, idx))

        elif _IDLE_ACTIVATE_START <= action_idx < _IDLE_ACTIVATE_START + 10:
            idx = action_idx - _IDLE_ACTIVATE_START
            idx = min(idx, len(msg.activatable) - 1)
            duel.respond_int(_encode_idle(5, idx))

        elif _IDLE_REPOSITION_START <= action_idx < _IDLE_REPOSITION_START + 5:
            idx = action_idx - _IDLE_REPOSITION_START
            idx = min(idx, len(msg.repositionable) - 1)
            duel.respond_int(_encode_idle(2, idx))

        elif action_idx == _IDLE_BATTLE:
            duel.respond_int(_encode_idle(6, 0))

        elif action_idx == _IDLE_END:
            duel.respond_int(_encode_idle(7, 0))

        else:
            # Fallback: end phase
            duel.respond_int(_encode_idle(7, 0))

    # ============================================================
    # Battle command decoding
    # ============================================================
    def _decode_battle(self, action_idx: int, msg: MsgSelectBattleCmd, duel) -> None:
        """
        Battle command response encoding:
          response = category | (index << 16)
        Categories (per ygopro-core playerop.cpp):
          0 = activate, 1 = attack, 2 = to main2, 3 = to end phase
        """
        if _BATTLE_ATTACK_START <= action_idx < _BATTLE_ATTACK_START + 5:
            idx = action_idx - _BATTLE_ATTACK_START
            idx = min(idx, len(msg.attackable) - 1)
            duel.respond_int(_encode_battle(1, idx))

        elif _BATTLE_ACTIVATE_START <= action_idx < _BATTLE_ACTIVATE_START + 5:
            idx = action_idx - _BATTLE_ACTIVATE_START
            idx = min(idx, len(msg.activatable) - 1)
            duel.respond_int(_encode_battle(0, idx))

        elif action_idx == _BATTLE_MAIN2:
            duel.respond_int(_encode_battle(2, 0))

        elif action_idx == _BATTLE_END:
            duel.respond_int(_encode_battle(3, 0))

        else:
            duel.respond_int(_encode_battle(3, 0))


# ============================================================
# Helpers
# ============================================================
def _encode_idle(category: int, index: int) -> int:
    """Encode idle command response as a single int: category | (index << 16)."""
    return category | (index << 16)


def _encode_battle(category: int, index: int) -> int:
    """Encode battle command response as a single int: category | (index << 16)."""
    return category | (index << 16)


def _respond_select_place(msg: MsgSelectPlace, duel) -> None:
    """
    Handle SELECT_PLACE by picking the first available valid zone.
    The bit layout is relative to the asking player (msg.player):
      bits 0-6:   asking player's MZONE 0-6
      bits 8-12:  asking player's SZONE 0-4
      bits 16-22: opponent's MZONE 0-6
      bits 24-28: opponent's SZONE 0-4
    Response: count × (player:u8, location:u8, sequence:u8)
    """
    available = ~msg.field_mask & 0xFFFFFFFF
    needed = max(1, msg.count)
    valid_zones = []

    for bit_group, actual_player in [(0, msg.player), (16, 1 - msg.player)]:
        for loc, bit_offset, sub_count in [
            (LOCATION.MZONE, 0, 7),
            (LOCATION.SZONE, 8, 5),
        ]:
            for seq in range(sub_count):
                bit = bit_group + bit_offset + seq
                if available & (1 << bit):
                    valid_zones.append((actual_player, int(loc), seq))

    if not valid_zones:
        valid_zones = [(msg.player, int(LOCATION.MZONE), 0)]

    count_to_pick = min(needed, len(valid_zones))
    selected = valid_zones[:count_to_pick]

    resp = b""
    for p, loc, seq in selected:
        resp += struct.pack("BBB", p, loc, seq)
    duel.respond_bytes(resp)


def _respond_announce_race(msg: MsgAnnounceRace, duel) -> None:
    """
    Handle ANNOUNCE_RACE by picking the first `count` available races.
    Response is a bitmask of selected races.
    """
    available = msg.available
    needed = max(1, msg.count)
    selected = 0
    chosen = 0
    for bit in range(25):  # 25 race types in YGO
        if available & (1 << bit):
            selected |= (1 << bit)
            chosen += 1
            if chosen >= needed:
                break
    if selected == 0:
        selected = available  # fallback: all available
    duel.respond_int(selected)
