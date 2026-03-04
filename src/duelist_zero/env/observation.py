"""
Observation encoder for the GoatEnv.

Converts a GameState into a flat float32 numpy array that the RL agent
can consume. All values are normalized to [0, 1] or [-1, 1].

Layout (OBSERVATION_DIM = 453):
  [0-9]     scalars (LP, hand, deck, GY, banished × 2)
  [10-17]   phase one-hot (8 phases)
  [18]      turn player flag
  [19]      turn count / 40
  [20-22]   relative advantages (LP, monster count, ATK)
  [23-102]  my monster zone: 5 × 16
  [103-152] my spell/trap zone: 5 × 10
  [153-232] opp monster zone: 5 × 16
  [233-282] opp spell/trap zone: 5 × 10
  [283-432] my hand: 10 × 15
  [433-452] deck identity one-hot (20)

Separate observation keys:
  action_features: (71, 12) — rich per-action features with card IDs, types, stats
  action_history:  (16, 10) — structured history with card IDs and types
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
_GAME_PROGRESS_DIM = 1    # turn count / 40
_RELATIVE_DIM = 3         # LP advantage, monster count advantage, ATK advantage
_MONSTER_SLOT_DIM = 16    # features per monster zone slot
_SPELL_SLOT_DIM = 10      # features per spell/trap zone slot
_HAND_SLOT_DIM = 15       # features per hand card slot
_MONSTER_ZONE_DIM = 5 * _MONSTER_SLOT_DIM   # 60
_SPELL_ZONE_DIM = 5 * _SPELL_SLOT_DIM        # 30
_HAND_DIM = 10 * _HAND_SLOT_DIM              # 80

MAX_DECKS = 20                               # capacity for deck identity one-hot
_DECK_ID_DIM = MAX_DECKS                     # 20

OBSERVATION_DIM = (
    _SCALAR_DIM +
    _PHASE_DIM +
    _TURN_DIM +
    _GAME_PROGRESS_DIM +
    _RELATIVE_DIM +
    2 * _MONSTER_ZONE_DIM +   # my + opp
    2 * _SPELL_ZONE_DIM +     # my + opp
    _HAND_DIM +
    _DECK_ID_DIM
)
# = 10 + 8 + 1 + 1 + 3 + 160 + 100 + 150 + 20 = 453

# Rich action features: 12 features per action slot
ACTION_FEATURES_DIM = 12
ACTION_FEATURES_SLOTS = 71

# Structured action history: 10 features per entry, 16 entries
HISTORY_LENGTH = 16
HISTORY_FEATURES_DIM = 10

# Card ID observation: 50 slots for embedding lookup
# [0-4] my monsters, [5-9] my S/T, [10-14] opp monsters (face-up only),
# [15-19] opp S/T (face-up only), [20-29] my hand (up to 10),
# [30-39] my graveyard (top 10), [40-49] opp graveyard (top 10)
CARD_ID_DIM = 50

# Action-to-card mapping: one card index per action slot
ACTION_CARD_DIM = 71


# Phase index mapping
_PHASE_ORDER = [
    PHASE.DRAW, PHASE.STANDBY, PHASE.MAIN1,
    PHASE.BATTLE_START, PHASE.BATTLE_STEP,
    PHASE.BATTLE, PHASE.MAIN2, PHASE.END,
]

# Location encoding for action features
_LOCATION_ENCODING = {
    LOCATION.HAND: 0.2,
    LOCATION.MZONE: 0.4,
    LOCATION.SZONE: 0.6,
    LOCATION.GRAVE: 0.8,
    LOCATION.REMOVED: 1.0,
}


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
    Encode a single monster zone slot into 16 floats.
    [occupied, face_up, atk_pos, atk/5000, def/5000, level/12,
     is_monster, is_spell, is_trap, attr_dark, attr_light, attr_other,
     is_effect, is_flip, is_fusion, is_ritual]
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
            v[12] = float(bool(data["type"] & TYPE.EFFECT))
            v[13] = float(bool(data["type"] & TYPE.FLIP))
            v[14] = float(bool(data["type"] & TYPE.FUSION))
            v[15] = float(bool(data["type"] & TYPE.RITUAL))
    return v


def _encode_spell_slot(card: Optional[ZoneCard], db: Optional[CardDB], visible: bool) -> np.ndarray:
    """
    Encode a single spell/trap zone slot into 10 floats.
    [occupied, face_up, is_spell, is_trap, is_continuous, is_equip,
     is_quickplay, is_field, is_counter, is_ritual]
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
            v[6] = float(bool(data["type"] & TYPE.QUICKPLAY))
            v[7] = float(bool(data["type"] & TYPE.FIELD))
            v[8] = float(bool(data["type"] & TYPE.COUNTER))
            v[9] = float(bool(data["type"] & TYPE.RITUAL))
    return v


def _encode_hand_card(code: int, db: Optional[CardDB]) -> np.ndarray:
    """
    Encode a single hand card into 15 floats.
    [level/12, atk/5000, def/5000, is_monster, is_spell, is_trap, attr_dark, attr_light,
     is_effect, is_flip, is_fusion, is_ritual, is_quickplay, is_field, is_counter]
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
        v[8] = float(bool(data["type"] & TYPE.EFFECT))
        v[9] = float(bool(data["type"] & TYPE.FLIP))
        v[10] = float(bool(data["type"] & TYPE.FUSION))
        v[11] = float(bool(data["type"] & TYPE.RITUAL))
        v[12] = float(bool(data["type"] & TYPE.QUICKPLAY))
        v[13] = float(bool(data["type"] & TYPE.FIELD))
        v[14] = float(bool(data["type"] & TYPE.COUNTER))
    return v


def _encode_location(loc: int) -> float:
    """Encode a LOCATION bitmask to a scalar."""
    for flag, val in _LOCATION_ENCODING.items():
        if loc & flag:
            return val
    return 0.0


def _encode_action_slot(
    card_code: int,
    action_type: int,
    location: int,
    card_index: 'CardIndex',
    db: Optional[CardDB],
) -> np.ndarray:
    """
    Encode a single action slot into ACTION_FEATURES_DIM (12) floats.
    [card_id, is_summon, is_spsummon, is_set, is_activate, is_attack,
     is_reposition, is_phase_pass_other, ATK/5000, DEF/5000, level/12, location]
    """
    v = np.zeros(ACTION_FEATURES_DIM, dtype=np.float32)
    if card_code != 0:
        v[0] = float(card_index.code_to_index(card_code))
    v[action_type] = 1.0  # action_type is col index 1-7
    if card_code != 0 and db is not None:
        data = db.get(card_code & 0x7FFFFFFF)
        if data:
            v[8] = min(data["atk"], 5000) / 5000.0 if data["atk"] >= 0 else 0.0
            v[9] = min(data["def"], 5000) / 5000.0 if data["def"] >= 0 else 0.0
            v[10] = min(data["level"], 12) / 12.0
    v[11] = _encode_location(location)
    return v


# Action type column indices for action_features
_ACT_SUMMON = 1
_ACT_SPSUMMON = 2
_ACT_SET = 3
_ACT_ACTIVATE = 4
_ACT_ATTACK = 5
_ACT_REPOSITION = 6
_ACT_OTHER = 7


def encode_action_features(
    msg,
    card_index: 'CardIndex',
    db: Optional[CardDB] = None,
) -> np.ndarray:
    """
    Encode rich per-action features for all 71 action slots.

    Returns float32 array of shape (ACTION_FEATURES_SLOTS, ACTION_FEATURES_DIM).
    """
    from ..core.message_parser import (
        MsgSelectIdleCmd,
        MsgSelectBattleCmd,
        MsgSelectCard,
        MsgSelectChain,
        MsgSelectEffectYn,
        MsgSelectTribute,
    )

    arr = np.zeros((ACTION_FEATURES_SLOTS, ACTION_FEATURES_DIM), dtype=np.float32)
    if msg is None:
        return arr

    if isinstance(msg, MsgSelectIdleCmd):
        for i, c in enumerate(msg.summonable[:5]):
            arr[0 + i] = _encode_action_slot(c.code, _ACT_SUMMON, c.location, card_index, db)
        for i, c in enumerate(msg.spsummonable[:5]):
            arr[5 + i] = _encode_action_slot(c.code, _ACT_SPSUMMON, c.location, card_index, db)
        for i, c in enumerate(msg.setable_monsters[:5]):
            arr[10 + i] = _encode_action_slot(c.code, _ACT_SET, c.location, card_index, db)
        for i, c in enumerate(msg.setable_st[:5]):
            arr[15 + i] = _encode_action_slot(c.code, _ACT_SET, c.location, card_index, db)
        for i, c in enumerate(msg.activatable[:10]):
            arr[20 + i] = _encode_action_slot(c.code, _ACT_ACTIVATE, c.location, card_index, db)
        for i, c in enumerate(msg.repositionable[:5]):
            arr[30 + i] = _encode_action_slot(c.code, _ACT_REPOSITION, c.location, card_index, db)
        # toBP (35), toEP (36) — phase transitions
        if msg.can_battle_phase:
            arr[35, _ACT_OTHER] = 1.0
        if msg.can_end_phase:
            arr[36, _ACT_OTHER] = 1.0

    elif isinstance(msg, MsgSelectBattleCmd):
        for i, a in enumerate(msg.attackable[:5]):
            if a.card is not None:
                arr[37 + i] = _encode_action_slot(
                    a.card.code, _ACT_ATTACK, a.card.location, card_index, db
                )
        for i, c in enumerate(msg.activatable[:5]):
            arr[42 + i] = _encode_action_slot(c.code, _ACT_ACTIVATE, c.location, card_index, db)
        # toM2 (47), toEP (48)
        if msg.can_main2:
            arr[47, _ACT_OTHER] = 1.0
        if msg.can_end_phase:
            arr[48, _ACT_OTHER] = 1.0

    elif isinstance(msg, (MsgSelectCard, MsgSelectTribute)):
        for i, c in enumerate(msg.cards[:10]):
            arr[51 + i] = _encode_action_slot(c.code, _ACT_OTHER, c.location, card_index, db)

    elif isinstance(msg, MsgSelectChain):
        for i, c in enumerate(msg.cards[:10]):
            arr[51 + i] = _encode_action_slot(c.code, _ACT_ACTIVATE, c.location, card_index, db)

    elif isinstance(msg, MsgSelectEffectYn):
        arr[49] = _encode_action_slot(msg.code, _ACT_ACTIVATE, msg.location, card_index, db)

    # Yes/No (49, 50), position (61, 62), options (63-70) are left as zeros
    # unless populated above — these are utility actions with no card info

    return arr


def encode_action_history(
    state: GameState,
    perspective: int,
    card_index: 'CardIndex',
    db: Optional[CardDB] = None,
) -> np.ndarray:
    """
    Encode structured action history.

    Returns float32 array of shape (HISTORY_LENGTH, HISTORY_FEATURES_DIM).
    Right-aligned: most recent action at the end.
    """
    arr = np.zeros((HISTORY_LENGTH, HISTORY_FEATURES_DIM), dtype=np.float32)
    recent = state.get_recent_actions(HISTORY_LENGTH)

    # Right-align: pad start with zeros
    offset = HISTORY_LENGTH - len(recent)
    current_turn = state.current_turn

    for i, action in enumerate(recent):
        v = arr[offset + i]
        if action.card_code != 0:
            v[0] = float(card_index.code_to_index(action.card_code))
        # Encode player relative to perspective
        v[1] = float(action.player != perspective)  # 0=me, 1=opponent
        v[2] = float(action.action_type == "summon")
        v[3] = float(action.action_type == "set")
        v[4] = float(action.action_type == "activate")
        v[5] = float(action.action_type == "attack")
        v[6] = float(action.action_type == "draw")
        # Card stats
        if action.card_code != 0 and db is not None:
            data = db.get(action.card_code & 0x7FFFFFFF)
            if data:
                v[7] = min(data["atk"], 5000) / 5000.0 if data["atk"] >= 0 else 0.0
                v[8] = min(data["def"], 5000) / 5000.0 if data["def"] >= 0 else 0.0
        # Turns ago
        v[9] = min(max(current_turn - action.turn, 0), 40) / 40.0

    return arr


# ============================================================
# Main encoder
# ============================================================
def encode_observation(
    state: GameState,
    perspective: int,
    db: Optional[CardDB] = None,
    deck_id: int = 0,
) -> np.ndarray:
    """
    Encode the full game state from `perspective` player's point of view.

    Args:
        state: Current GameState
        perspective: 0 or 1 — which player is the agent
        db: CardDB for looking up card stats (optional but recommended)
        deck_id: Index into the deck pool (0 = Goat Control, up to MAX_DECKS-1)

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

    # --- Turn count ---
    obs[idx] = min(state.current_turn, 40) / 40.0
    idx += 1

    # --- Relative advantage features ---
    lp_diff = (me.lp - opp.lp) / 8000.0
    obs[idx] = max(-1.0, min(1.0, lp_diff))
    idx += 1

    my_mon_count = sum(1 for m in me.monsters if m is not None)
    opp_mon_count = sum(1 for m in opp.monsters if m is not None)
    obs[idx] = max(-1.0, min(1.0, (my_mon_count - opp_mon_count) / 5.0))
    idx += 1

    # Total ATK of attack-position monsters
    my_atk = 0.0
    for m in me.monsters:
        if m is not None and bool(m.position & POSITION.ATTACK) and db is not None:
            data = db.get(m.code & 0x7FFFFFFF)
            if data and data["atk"] >= 0:
                my_atk += data["atk"]
    opp_atk = 0.0
    for m in opp.monsters:
        if m is not None and bool(m.position & POSITION.FACEUP) and bool(m.position & POSITION.ATTACK) and db is not None:
            data = db.get(m.code & 0x7FFFFFFF)
            if data and data["atk"] >= 0:
                opp_atk += data["atk"]
    obs[idx] = max(-1.0, min(1.0, (my_atk - opp_atk) / 10000.0))
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

    # --- Deck identity one-hot ---
    if 0 <= deck_id < MAX_DECKS:
        obs[idx + deck_id] = 1.0
    idx += _DECK_ID_DIM

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

    # My graveyard (top 10, most recent first — public info)
    my_gy = me.graveyard[-10:]
    for i in range(10):
        if i < len(my_gy) and my_gy[-(i + 1)] != 0:
            ids[idx] = float(card_index.code_to_index(my_gy[-(i + 1)]))
        idx += 1

    # Opp graveyard (top 10, most recent first — public info)
    opp_gy = opp.graveyard[-10:]
    for i in range(10):
        if i < len(opp_gy) and opp_gy[-(i + 1)] != 0:
            ids[idx] = float(card_index.code_to_index(opp_gy[-(i + 1)]))
        idx += 1

    assert idx == CARD_ID_DIM
    return ids


# ============================================================
# Action-to-card mapping encoder
# ============================================================
def encode_action_cards(msg, card_index: CardIndex) -> np.ndarray:
    """
    Encode which card each action slot refers to.

    Returns float32 array of shape (ACTION_CARD_DIM,) where each element
    is the card embedding index for the corresponding action. Zero means
    no card (phase transitions, pass, or unused slots).

    Action layout (from action_space.py):
      [0-4]   idle summon, [5-9] idle spsummon, [10-14] idle set monster,
      [15-19] idle set S/T, [20-29] idle activate, [30-34] idle reposition,
      [35] toBP, [36] toEP,
      [37-41] battle attack, [42-46] battle activate,
      [47] toM2, [48] toEP,
      [49] yes, [50] no,
      [51-60] select card 0-9,
      [61] pos ATK, [62] pos DEF, [63-70] option 0-7
    """
    from ..core.message_parser import (
        MsgSelectIdleCmd,
        MsgSelectBattleCmd,
        MsgSelectCard,
        MsgSelectChain,
        MsgSelectEffectYn,
        MsgSelectTribute,
    )

    arr = np.zeros(ACTION_CARD_DIM, dtype=np.float32)
    if msg is None:
        return arr

    if isinstance(msg, MsgSelectIdleCmd):
        for i, c in enumerate(msg.summonable[:5]):
            arr[0 + i] = float(card_index.code_to_index(c.code))
        for i, c in enumerate(msg.spsummonable[:5]):
            arr[5 + i] = float(card_index.code_to_index(c.code))
        for i, c in enumerate(msg.setable_monsters[:5]):
            arr[10 + i] = float(card_index.code_to_index(c.code))
        for i, c in enumerate(msg.setable_st[:5]):
            arr[15 + i] = float(card_index.code_to_index(c.code))
        for i, c in enumerate(msg.activatable[:10]):
            arr[20 + i] = float(card_index.code_to_index(c.code))
        for i, c in enumerate(msg.repositionable[:5]):
            arr[30 + i] = float(card_index.code_to_index(c.code))

    elif isinstance(msg, MsgSelectBattleCmd):
        for i, a in enumerate(msg.attackable[:5]):
            if a.card is not None:
                arr[37 + i] = float(card_index.code_to_index(a.card.code))
        for i, c in enumerate(msg.activatable[:5]):
            arr[42 + i] = float(card_index.code_to_index(c.code))

    elif isinstance(msg, (MsgSelectCard, MsgSelectTribute)):
        for i, c in enumerate(msg.cards[:10]):
            arr[51 + i] = float(card_index.code_to_index(c.code))

    elif isinstance(msg, MsgSelectChain):
        for i, c in enumerate(msg.cards[:10]):
            arr[51 + i] = float(card_index.code_to_index(c.code))

    elif isinstance(msg, MsgSelectEffectYn):
        arr[49] = float(card_index.code_to_index(msg.code))

    return arr
