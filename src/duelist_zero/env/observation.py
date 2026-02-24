"""
Observation encoder for the GoatEnv.

Converts a GameState into a flat float32 numpy array that the RL agent
can consume. All values are normalized to [0, 1] or [-1, 1].

Layout (OBSERVATION_DIM total):
  [0]     my LP / 8000
  [1]     opp LP / 8000
  [2]     my hand size / 10
  [3]     opp hand size / 10
  [4]     my deck count / 40
  [5]     opp deck count / 40
  [6]     my GY count / 40
  [7]     opp GY count / 40
  [8]     my banished / 40
  [9]     opp banished / 40
  [10-17] phase one-hot (8 phases)
  [18]    turn player flag (1 if my turn)
  [19-78] my monster zone: 5 slots × 12 features
  [79-108] my spell/trap zone: 5 slots × 6 features
  [109-168] opp monster zone: 5 slots × 12 features
  [169-198] opp spell/trap zone: 5 slots × 6 features
  [199-278] my hand: 10 slots × 8 features
  [279-328] recent actions: 10 × 5 features
"""

import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.constants import LOCATION, POSITION, PHASE, TYPE, ATTRIBUTE
from ..engine.game_state import GameState, ZoneCard
from .card_index import CardIndex


# ============================================================
# Dimension constants
# ============================================================
_SCALAR_DIM = 10          # LP, hand, deck, GY, banished × 2 players
_PHASE_DIM = 8            # phase one-hot
_TURN_DIM = 1             # turn player flag
_MONSTER_SLOT_DIM = 12    # features per monster zone slot
_SPELL_SLOT_DIM = 6       # features per spell/trap zone slot
_HAND_SLOT_DIM = 8        # features per hand card slot
_ACTION_HIST_DIM = 5      # features per action history entry

_MONSTER_ZONE_DIM = 5 * _MONSTER_SLOT_DIM   # 60
_SPELL_ZONE_DIM = 5 * _SPELL_SLOT_DIM        # 30
_HAND_DIM = 10 * _HAND_SLOT_DIM              # 80
_ACTION_HIST_TOTAL = 10 * _ACTION_HIST_DIM   # 50

OBSERVATION_DIM = (
    _SCALAR_DIM +
    _PHASE_DIM +
    _TURN_DIM +
    2 * _MONSTER_ZONE_DIM +   # my + opp
    2 * _SPELL_ZONE_DIM +     # my + opp
    _HAND_DIM +
    _ACTION_HIST_TOTAL
)
# = 10 + 8 + 1 + 120 + 60 + 80 + 50 = 329

# Card ID observation: 30 slots for embedding lookup
# [0-4] my monsters, [5-9] my S/T, [10-14] opp monsters (face-up only),
# [15-19] opp S/T (face-up only), [20-29] my hand (up to 10)
CARD_ID_DIM = 30


# Phase index mapping
_PHASE_ORDER = [
    PHASE.DRAW, PHASE.STANDBY, PHASE.MAIN1,
    PHASE.BATTLE_START, PHASE.BATTLE_STEP,
    PHASE.BATTLE, PHASE.MAIN2, PHASE.END,
]

# Action type index mapping for history
_ACTION_TYPES = ["summon", "set", "activate", "attack", "draw"]


# ============================================================
# Card DB loader (for hand card features)
# ============================================================
class CardDB:
    """Lightweight SQLite card database reader."""

    def __init__(self, db_path: str | Path):
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._cache: dict[int, dict] = {}

    def get(self, code: int) -> Optional[dict]:
        """Return card data dict or None if not found."""
        if code in self._cache:
            return self._cache[code]
        cur = self._conn.cursor()
        cur.execute(
            "SELECT type, level, attribute, atk, def FROM datas WHERE id=?",
            (code,)
        )
        row = cur.fetchone()
        if row is None:
            self._cache[code] = None
            return None
        data = {
            "type": row[0],
            "level": row[1],
            "attribute": row[2],
            "atk": row[3],
            "def": row[4],
        }
        self._cache[code] = data
        return data

    def close(self):
        self._conn.close()


# ============================================================
# Feature encoders
# ============================================================
def _encode_monster_slot(card: Optional[ZoneCard], db: Optional[CardDB], visible: bool) -> np.ndarray:
    """
    Encode a single monster zone slot into 12 floats.
    [occupied, face_up, atk_pos, atk/2000, def/2000, level/12,
     is_monster, is_spell, is_trap, attr_dark, attr_light, attr_other]
    """
    v = np.zeros(_MONSTER_SLOT_DIM, dtype=np.float32)
    if card is None:
        return v

    v[0] = 1.0  # occupied
    face_up = bool(card.position & POSITION.FACEUP)
    atk_pos = bool(card.position & POSITION.ATTACK)
    v[1] = float(face_up)
    v[2] = float(atk_pos)

    if face_up and visible and db is not None:
        data = db.get(card.code & 0x7FFFFFFF)
        if data:
            v[3] = min(data["atk"], 5000) / 5000.0 if data["atk"] >= 0 else 0.0
            v[4] = min(data["def"], 5000) / 5000.0 if data["def"] >= 0 else 0.0
            v[5] = min(data["level"], 12) / 12.0
            v[6] = float(bool(data["type"] & TYPE.MONSTER))
            v[7] = float(bool(data["type"] & TYPE.SPELL))
            v[8] = float(bool(data["type"] & TYPE.TRAP))
            v[9] = float(bool(data["attribute"] & ATTRIBUTE.DARK))
            v[10] = float(bool(data["attribute"] & ATTRIBUTE.LIGHT))
            v[11] = float(bool(data["attribute"] & ~(ATTRIBUTE.DARK | ATTRIBUTE.LIGHT)))
    return v


def _encode_spell_slot(card: Optional[ZoneCard], db: Optional[CardDB], visible: bool) -> np.ndarray:
    """
    Encode a single spell/trap zone slot into 6 floats.
    [occupied, face_up, is_spell, is_trap, is_continuous, is_equip]
    """
    v = np.zeros(_SPELL_SLOT_DIM, dtype=np.float32)
    if card is None:
        return v

    v[0] = 1.0
    face_up = bool(card.position & POSITION.FACEUP)
    v[1] = float(face_up)

    if face_up and visible and db is not None:
        data = db.get(card.code & 0x7FFFFFFF)
        if data:
            v[2] = float(bool(data["type"] & TYPE.SPELL))
            v[3] = float(bool(data["type"] & TYPE.TRAP))
            v[4] = float(bool(data["type"] & TYPE.CONTINUOUS))
            v[5] = float(bool(data["type"] & TYPE.EQUIP))
    return v


def _encode_hand_card(code: int, db: Optional[CardDB]) -> np.ndarray:
    """
    Encode a single hand card into 8 floats.
    [level/12, atk/5000, def/5000, is_monster, is_spell, is_trap, attr_dark, attr_light]
    """
    v = np.zeros(_HAND_SLOT_DIM, dtype=np.float32)
    if code == 0 or db is None:
        return v

    data = db.get(code & 0x7FFFFFFF)
    if data:
        v[0] = min(data["level"], 12) / 12.0
        v[1] = min(data["atk"], 5000) / 5000.0 if data["atk"] >= 0 else 0.0
        v[2] = min(data["def"], 5000) / 5000.0 if data["def"] >= 0 else 0.0
        v[3] = float(bool(data["type"] & TYPE.MONSTER))
        v[4] = float(bool(data["type"] & TYPE.SPELL))
        v[5] = float(bool(data["type"] & TYPE.TRAP))
        v[6] = float(bool(data["attribute"] & ATTRIBUTE.DARK))
        v[7] = float(bool(data["attribute"] & ATTRIBUTE.LIGHT))
    return v


def _encode_action(action) -> np.ndarray:
    """
    Encode a single ActionRecord into 5 floats.
    [player, is_summon, is_set, is_activate, is_attack]
    """
    from ..engine.game_state import ActionRecord
    v = np.zeros(_ACTION_HIST_DIM, dtype=np.float32)
    if action is None:
        return v
    v[0] = float(action.player)
    v[1] = float(action.action_type == "summon")
    v[2] = float(action.action_type == "set")
    v[3] = float(action.action_type == "activate")
    v[4] = float(action.action_type == "attack")
    return v


# ============================================================
# Main encoder
# ============================================================
def encode_observation(
    state: GameState,
    perspective: int,
    db: Optional[CardDB] = None,
) -> np.ndarray:
    """
    Encode the full game state from `perspective` player's point of view.

    Args:
        state: Current GameState
        perspective: 0 or 1 — which player is the agent
        db: CardDB for looking up card stats (optional but recommended)

    Returns:
        float32 array of shape (OBSERVATION_DIM,)
    """
    obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    me = state.players[perspective]
    opp = state.players[1 - perspective]
    idx = 0

    # --- Scalars ---
    obs[idx] = me.lp / 8000.0;          idx += 1
    obs[idx] = opp.lp / 8000.0;         idx += 1
    obs[idx] = me.hand_count / 10.0;    idx += 1
    obs[idx] = opp.hand_count / 10.0;   idx += 1
    obs[idx] = me.deck_count / 40.0;    idx += 1
    obs[idx] = opp.deck_count / 40.0;   idx += 1
    obs[idx] = me.grave_count / 40.0;   idx += 1
    obs[idx] = opp.grave_count / 40.0;  idx += 1
    obs[idx] = me.banished_count / 40.0; idx += 1
    obs[idx] = opp.banished_count / 40.0; idx += 1

    # --- Phase one-hot ---
    for phase in _PHASE_ORDER:
        obs[idx] = float(state.current_phase == phase)
        idx += 1

    # --- Turn player ---
    obs[idx] = float(state.turn_player == perspective)
    idx += 1

    # --- My monster zone ---
    for slot in range(5):
        card = me.monsters[slot] if slot < len(me.monsters) else None
        obs[idx:idx + _MONSTER_SLOT_DIM] = _encode_monster_slot(card, db, visible=True)
        idx += _MONSTER_SLOT_DIM

    # --- My spell/trap zone ---
    for slot in range(5):
        card = me.spells[slot] if slot < len(me.spells) else None
        obs[idx:idx + _SPELL_SLOT_DIM] = _encode_spell_slot(card, db, visible=True)
        idx += _SPELL_SLOT_DIM

    # --- Opp monster zone (face-down hidden) ---
    for slot in range(5):
        card = opp.monsters[slot] if slot < len(opp.monsters) else None
        obs[idx:idx + _MONSTER_SLOT_DIM] = _encode_monster_slot(card, db, visible=True)
        idx += _MONSTER_SLOT_DIM

    # --- Opp spell/trap zone (face-down hidden) ---
    for slot in range(5):
        card = opp.spells[slot] if slot < len(opp.spells) else None
        obs[idx:idx + _SPELL_SLOT_DIM] = _encode_spell_slot(card, db, visible=False)
        idx += _SPELL_SLOT_DIM

    # --- My hand (up to 10 cards) ---
    hand = me.hand[:10]
    for i in range(10):
        code = hand[i] if i < len(hand) else 0
        obs[idx:idx + _HAND_SLOT_DIM] = _encode_hand_card(code, db)
        idx += _HAND_SLOT_DIM

    # --- Recent action history (last 10) ---
    recent = state.get_recent_actions(10)
    # Pad to 10
    recent = [None] * (10 - len(recent)) + list(recent)
    for action in recent:
        obs[idx:idx + _ACTION_HIST_DIM] = _encode_action(action)
        idx += _ACTION_HIST_DIM

    assert idx == OBSERVATION_DIM, f"Observation dim mismatch: {idx} != {OBSERVATION_DIM}"
    return obs


# ============================================================
# Card ID encoder (for embedding lookup)
# ============================================================
def encode_card_ids(
    state: GameState,
    perspective: int,
    card_index: CardIndex,
) -> np.ndarray:
    """
    Encode card identity indices for embedding lookup.

    Returns float32 array of shape (CARD_ID_DIM,) with integer-valued floats.
    Index 0 = empty/unknown/face-down. Face-down opponent cards get index 0.
    """
    ids = np.zeros(CARD_ID_DIM, dtype=np.float32)
    me = state.players[perspective]
    opp = state.players[1 - perspective]
    idx = 0

    # My monsters (5 slots, always visible)
    for slot in range(5):
        card = me.monsters[slot] if slot < len(me.monsters) else None
        if card is not None:
            ids[idx] = float(card_index.code_to_index(card.code))
        idx += 1

    # My S/T (5 slots, always visible)
    for slot in range(5):
        card = me.spells[slot] if slot < len(me.spells) else None
        if card is not None:
            ids[idx] = float(card_index.code_to_index(card.code))
        idx += 1

    # Opp monsters (face-up only)
    for slot in range(5):
        card = opp.monsters[slot] if slot < len(opp.monsters) else None
        if card is not None and bool(card.position & POSITION.FACEUP):
            ids[idx] = float(card_index.code_to_index(card.code))
        idx += 1

    # Opp S/T (face-up only)
    for slot in range(5):
        card = opp.spells[slot] if slot < len(opp.spells) else None
        if card is not None and bool(card.position & POSITION.FACEUP):
            ids[idx] = float(card_index.code_to_index(card.code))
        idx += 1

    # My hand (up to 10 cards)
    hand = me.hand[:10]
    for i in range(10):
        if i < len(hand) and hand[i] != 0:
            ids[idx] = float(card_index.code_to_index(hand[i]))
        idx += 1

    assert idx == CARD_ID_DIM
    return ids
