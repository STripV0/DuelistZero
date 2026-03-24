# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Fast Cython observation encoder for GoatEnv.

Drop-in replacement for the pure Python functions in observation.py.
Uses typed memoryviews, pre-allocated buffers, and inline helpers
to avoid creating intermediate numpy arrays.
"""

import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport fminf, fmaxf

np.import_array()

# ============================================================
# Constants (mirrored from observation.py and constants.py)
# ============================================================

# Dimensions
DEF OBSERVATION_DIM = 462
DEF CARD_ID_DIM = 50
DEF ACTION_FEATURES_SLOTS = 71
DEF ACTION_FEATURES_DIM = 29
DEF HISTORY_LENGTH = 16
DEF HISTORY_FEATURES_DIM = 13
DEF MAX_DECKS = 20

DEF MONSTER_SLOT_DIM = 16
DEF SPELL_SLOT_DIM = 10
DEF HAND_SLOT_DIM = 15

# POSITION flags
DEF POS_FACEUP_ATTACK = 0x1
DEF POS_FACEDOWN_ATTACK = 0x2
DEF POS_FACEUP_DEFENSE = 0x4
DEF POS_FACEDOWN_DEFENSE = 0x8
DEF POS_FACEUP = 0x1 | 0x4   # FACEUP_ATTACK | FACEUP_DEFENSE
DEF POS_FACEDOWN = 0x2 | 0x8  # FACEDOWN_ATTACK | FACEDOWN_DEFENSE
DEF POS_ATTACK = 0x1 | 0x2    # FACEUP_ATTACK | FACEDOWN_ATTACK

# LOCATION flags
DEF LOC_DECK = 0x01
DEF LOC_HAND = 0x02
DEF LOC_MZONE = 0x04
DEF LOC_SZONE = 0x08
DEF LOC_GRAVE = 0x10
DEF LOC_REMOVED = 0x20

# TYPE flags
DEF TYPE_MONSTER = 0x1
DEF TYPE_SPELL = 0x2
DEF TYPE_TRAP = 0x4
DEF TYPE_EFFECT = 0x20
DEF TYPE_FUSION = 0x40
DEF TYPE_RITUAL = 0x80
DEF TYPE_QUICKPLAY = 0x10000
DEF TYPE_CONTINUOUS = 0x20000
DEF TYPE_EQUIP = 0x40000
DEF TYPE_FIELD = 0x80000
DEF TYPE_COUNTER = 0x100000
DEF TYPE_FLIP = 0x200000

# ATTRIBUTE flags
DEF ATTR_DARK = 0x20
DEF ATTR_LIGHT = 0x10

# PHASE values
DEF PHASE_DRAW = 0x01
DEF PHASE_STANDBY = 0x02
DEF PHASE_MAIN1 = 0x04
DEF PHASE_BATTLE_START = 0x08
DEF PHASE_BATTLE_STEP = 0x10
DEF PHASE_BATTLE = 0x80
DEF PHASE_MAIN2 = 0x100
DEF PHASE_END = 0x200

# Action type column indices
DEF ACT_SUMMON = 1
DEF ACT_SPSUMMON = 2
DEF ACT_SET = 3
DEF ACT_ACTIVATE = 4
DEF ACT_ATTACK = 5
DEF ACT_REPOSITION = 6
DEF ACT_OTHER = 7

# Phase order for one-hot encoding (8 phases)
cdef int[8] PHASE_ORDER
PHASE_ORDER[0] = PHASE_DRAW
PHASE_ORDER[1] = PHASE_STANDBY
PHASE_ORDER[2] = PHASE_MAIN1
PHASE_ORDER[3] = PHASE_BATTLE_START
PHASE_ORDER[4] = PHASE_BATTLE_STEP
PHASE_ORDER[5] = PHASE_BATTLE
PHASE_ORDER[6] = PHASE_MAIN2
PHASE_ORDER[7] = PHASE_END

# Import effect flags lookup
from .effect_flags import get_effect_flags as _py_get_effect_flags, _EFFECT_FLAGS


# ============================================================
# Helper: fast clamp
# ============================================================
cdef inline float clampf(float x, float lo, float hi) noexcept nogil:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


# ============================================================
# Helper: encode location to scalar
# ============================================================
cdef inline float encode_location_scalar(int loc) noexcept nogil:
    if loc & LOC_HAND:
        return 0.2
    if loc & LOC_MZONE:
        return 0.4
    if loc & LOC_SZONE:
        return 0.6
    if loc & LOC_GRAVE:
        return 0.8
    if loc & LOC_REMOVED:
        return 1.0
    return 0.0


# ============================================================
# Helper: encode monster slot into buffer at offset
# ============================================================
cdef inline void encode_monster_slot(
    float[:] buf,
    int offset,
    object card,
    object db,
    bint visible,
) noexcept:
    """Encode one monster zone slot (16 floats) into buf[offset:offset+16]."""
    cdef int i
    cdef int pos, card_code
    cdef bint face_up, atk_pos
    cdef int c_type, c_level, c_attr, c_atk, c_def

    # Already zeroed by caller
    if card is None:
        return

    buf[offset] = 1.0  # occupied
    pos = card.position
    face_up = (pos & POS_FACEUP) != 0
    atk_pos = (pos & POS_ATTACK) != 0
    buf[offset + 1] = 1.0 if face_up else 0.0
    buf[offset + 2] = 1.0 if atk_pos else 0.0

    if face_up and visible and db is not None:
        card_code = card.code & 0x7FFFFFFF
        data = db.get(card_code)
        if data is not None:
            c_atk = data["atk"]
            c_def = data["def"]
            c_level = data["level"]
            c_type = data["type"]
            c_attr = data["attribute"]

            buf[offset + 3] = clampf(<float>c_atk, 0.0, 5000.0) / 5000.0 if c_atk >= 0 else 0.0
            buf[offset + 4] = clampf(<float>c_def, 0.0, 5000.0) / 5000.0 if c_def >= 0 else 0.0
            buf[offset + 5] = clampf(<float>c_level, 0.0, 12.0) / 12.0
            buf[offset + 6] = 1.0 if (c_type & TYPE_MONSTER) else 0.0
            buf[offset + 7] = 1.0 if (c_type & TYPE_SPELL) else 0.0
            buf[offset + 8] = 1.0 if (c_type & TYPE_TRAP) else 0.0
            buf[offset + 9] = 1.0 if (c_attr & ATTR_DARK) else 0.0
            buf[offset + 10] = 1.0 if (c_attr & ATTR_LIGHT) else 0.0
            buf[offset + 11] = 1.0 if (c_attr & ~(ATTR_DARK | ATTR_LIGHT)) else 0.0
            buf[offset + 12] = 1.0 if (c_type & TYPE_EFFECT) else 0.0
            buf[offset + 13] = 1.0 if (c_type & TYPE_FLIP) else 0.0
            buf[offset + 14] = 1.0 if (c_type & TYPE_FUSION) else 0.0
            buf[offset + 15] = 1.0 if (c_type & TYPE_RITUAL) else 0.0


# ============================================================
# Helper: encode spell/trap slot into buffer at offset
# ============================================================
cdef inline void encode_spell_slot(
    float[:] buf,
    int offset,
    object card,
    object db,
    bint visible,
) noexcept:
    """Encode one spell/trap zone slot (10 floats) into buf[offset:offset+10]."""
    cdef int pos, card_code, c_type
    cdef bint face_up

    if card is None:
        return

    buf[offset] = 1.0
    pos = card.position
    face_up = (pos & POS_FACEUP) != 0
    buf[offset + 1] = 1.0 if face_up else 0.0

    if face_up and visible and db is not None:
        card_code = card.code & 0x7FFFFFFF
        data = db.get(card_code)
        if data is not None:
            c_type = data["type"]
            buf[offset + 2] = 1.0 if (c_type & TYPE_SPELL) else 0.0
            buf[offset + 3] = 1.0 if (c_type & TYPE_TRAP) else 0.0
            buf[offset + 4] = 1.0 if (c_type & TYPE_CONTINUOUS) else 0.0
            buf[offset + 5] = 1.0 if (c_type & TYPE_EQUIP) else 0.0
            buf[offset + 6] = 1.0 if (c_type & TYPE_QUICKPLAY) else 0.0
            buf[offset + 7] = 1.0 if (c_type & TYPE_FIELD) else 0.0
            buf[offset + 8] = 1.0 if (c_type & TYPE_COUNTER) else 0.0
            buf[offset + 9] = 1.0 if (c_type & TYPE_RITUAL) else 0.0


# ============================================================
# Helper: encode hand card into buffer at offset
# ============================================================
cdef inline void encode_hand_card(
    float[:] buf,
    int offset,
    int code,
    object db,
) noexcept:
    """Encode one hand card (15 floats) into buf[offset:offset+15]."""
    cdef int c_type, c_level, c_attr, c_atk, c_def

    if code == 0 or db is None:
        return

    data = db.get(code & 0x7FFFFFFF)
    if data is not None:
        c_atk = data["atk"]
        c_def = data["def"]
        c_level = data["level"]
        c_type = data["type"]
        c_attr = data["attribute"]

        buf[offset + 0] = clampf(<float>c_level, 0.0, 12.0) / 12.0
        buf[offset + 1] = clampf(<float>c_atk, 0.0, 5000.0) / 5000.0 if c_atk >= 0 else 0.0
        buf[offset + 2] = clampf(<float>c_def, 0.0, 5000.0) / 5000.0 if c_def >= 0 else 0.0
        buf[offset + 3] = 1.0 if (c_type & TYPE_MONSTER) else 0.0
        buf[offset + 4] = 1.0 if (c_type & TYPE_SPELL) else 0.0
        buf[offset + 5] = 1.0 if (c_type & TYPE_TRAP) else 0.0
        buf[offset + 6] = 1.0 if (c_attr & ATTR_DARK) else 0.0
        buf[offset + 7] = 1.0 if (c_attr & ATTR_LIGHT) else 0.0
        buf[offset + 8] = 1.0 if (c_type & TYPE_EFFECT) else 0.0
        buf[offset + 9] = 1.0 if (c_type & TYPE_FLIP) else 0.0
        buf[offset + 10] = 1.0 if (c_type & TYPE_FUSION) else 0.0
        buf[offset + 11] = 1.0 if (c_type & TYPE_RITUAL) else 0.0
        buf[offset + 12] = 1.0 if (c_type & TYPE_QUICKPLAY) else 0.0
        buf[offset + 13] = 1.0 if (c_type & TYPE_FIELD) else 0.0
        buf[offset + 14] = 1.0 if (c_type & TYPE_COUNTER) else 0.0


# ============================================================
# Helper: encode action slot into 2D buffer at row
# ============================================================
cdef inline void encode_action_slot(
    float[:, :] buf,
    int row,
    int card_code,
    int action_type,
    int location,
    object card_index,
    object db,
) noexcept:
    """Encode one action slot (29 floats) into buf[row, :]."""
    cdef int c_atk, c_def, c_level
    cdef int masked_code

    if card_code != 0:
        buf[row, 0] = <float>card_index.code_to_index(card_code)

    # action_type is the column index (1-7)
    buf[row, action_type] = 1.0

    if card_code != 0 and db is not None:
        masked_code = card_code & 0x7FFFFFFF
        data = db.get(masked_code)
        if data is not None:
            c_atk = data["atk"]
            c_def = data["def"]
            c_level = data["level"]
            buf[row, 8] = clampf(<float>c_atk, 0.0, 5000.0) / 5000.0 if c_atk >= 0 else 0.0
            buf[row, 9] = clampf(<float>c_def, 0.0, 5000.0) / 5000.0 if c_def >= 0 else 0.0
            buf[row, 10] = clampf(<float>c_level, 0.0, 12.0) / 12.0

    buf[row, 11] = encode_location_scalar(location)

    # C2: Effect category flags (cols 14-25)
    if card_code != 0:
        masked_code = card_code & 0x7FFFFFFF
        indices = _EFFECT_FLAGS.get(masked_code)
        if indices:
            for flag_idx in indices:
                buf[row, 14 + <int>flag_idx] = 1.0


# ============================================================
# Main encoder: encode_observation_fast
# ============================================================
def encode_observation_fast(
    object state,
    int perspective,
    object db,
    int deck_id,
) -> np.ndarray:
    """
    Fast Cython replacement for encode_observation().

    Returns float32 array of shape (462,).
    """
    cdef np.ndarray[np.float32_t, ndim=1] obs_arr = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    cdef float[:] obs = obs_arr
    cdef int idx = 0
    cdef int slot, i
    cdef float lp_diff, my_atk_total, opp_atk_total
    cdef int my_mon_count, opp_mon_count
    cdef int pos, card_code
    cdef int opp_facedown
    cdef int opp_gy_monsters
    cdef int c_type

    me = state.players[perspective]
    opp = state.players[1 - perspective]

    # --- Scalars (10) ---
    obs[0] = <float>me.lp / 8000.0
    obs[1] = <float>opp.lp / 8000.0
    obs[2] = <float>len(me.hand) / 10.0
    obs[3] = <float>len(opp.hand) / 10.0
    obs[4] = <float>me.deck_count / 40.0
    obs[5] = <float>opp.deck_count / 40.0
    obs[6] = <float>len(me.graveyard) / 40.0
    obs[7] = <float>len(opp.graveyard) / 40.0
    obs[8] = <float>len(me.banished) / 40.0
    obs[9] = <float>len(opp.banished) / 40.0
    idx = 10

    # --- Phase one-hot (8) ---
    cdef int current_phase = state.current_phase
    for i in range(8):
        obs[idx + i] = 1.0 if current_phase == PHASE_ORDER[i] else 0.0
    idx += 8

    # --- Turn player (1) ---
    obs[idx] = 1.0 if state.turn_player == perspective else 0.0
    idx += 1

    # --- Turn count (1) ---
    cdef int current_turn = state.current_turn
    obs[idx] = clampf(<float>current_turn, 0.0, 40.0) / 40.0
    idx += 1

    # --- Relative advantages (3) ---
    lp_diff = <float>(me.lp - opp.lp) / 8000.0
    obs[idx] = clampf(lp_diff, -1.0, 1.0)
    idx += 1

    # Monster count advantage
    my_mon_count = 0
    opp_mon_count = 0
    my_monsters = me.monsters
    opp_monsters = opp.monsters
    for slot in range(5):
        if my_monsters[slot] is not None:
            my_mon_count += 1
        if opp_monsters[slot] is not None:
            opp_mon_count += 1
    obs[idx] = clampf(<float>(my_mon_count - opp_mon_count) / 5.0, -1.0, 1.0)
    idx += 1

    # ATK advantage
    my_atk_total = 0.0
    opp_atk_total = 0.0
    if db is not None:
        for slot in range(5):
            card = my_monsters[slot]
            if card is not None:
                pos = card.position
                if pos & POS_ATTACK:
                    data = db.get(card.code & 0x7FFFFFFF)
                    if data is not None and data["atk"] >= 0:
                        my_atk_total += <float>data["atk"]
            card = opp_monsters[slot]
            if card is not None:
                pos = card.position
                if (pos & POS_FACEUP) and (pos & POS_ATTACK):
                    data = db.get(card.code & 0x7FFFFFFF)
                    if data is not None and data["atk"] >= 0:
                        opp_atk_total += <float>data["atk"]
    obs[idx] = clampf((my_atk_total - opp_atk_total) / 10000.0, -1.0, 1.0)
    idx += 1

    # --- My monster zone: 5 x 16 = 80 ---
    for slot in range(5):
        card = my_monsters[slot] if slot < len(my_monsters) else None
        encode_monster_slot(obs, idx, card, db, True)
        idx += MONSTER_SLOT_DIM

    # --- My spell/trap zone: 5 x 10 = 50 ---
    my_spells = me.spells
    for slot in range(5):
        card = my_spells[slot] if slot < len(my_spells) else None
        encode_spell_slot(obs, idx, card, db, True)
        idx += SPELL_SLOT_DIM

    # --- Opp monster zone: 5 x 16 = 80 ---
    for slot in range(5):
        card = opp_monsters[slot] if slot < len(opp_monsters) else None
        encode_monster_slot(obs, idx, card, db, True)
        idx += MONSTER_SLOT_DIM

    # --- Opp spell/trap zone: 5 x 10 = 50 ---
    opp_spells = opp.spells
    for slot in range(5):
        card = opp_spells[slot] if slot < len(opp_spells) else None
        encode_spell_slot(obs, idx, card, db, False)
        idx += SPELL_SLOT_DIM

    # --- My hand: 10 x 15 = 150 ---
    hand = me.hand
    cdef int hand_len = len(hand)
    for i in range(10):
        if i < hand_len and i < 10:
            encode_hand_card(obs, idx, hand[i], db)
        idx += HAND_SLOT_DIM

    # --- Deck identity one-hot (20) ---
    if 0 <= deck_id < MAX_DECKS:
        obs[idx + deck_id] = 1.0
    idx += MAX_DECKS

    # --- B2: Opponent inference features (9) ---
    obs[idx] = clampf(<float>opp.extra_draws, 0.0, 5.0) / 5.0
    idx += 1

    # Opponent face-down count
    opp_facedown = 0
    for slot in range(5):
        card = opp_monsters[slot]
        if card is not None and (card.position & POS_FACEDOWN):
            opp_facedown += 1
        card = opp_spells[slot]
        if card is not None and (card.position & POS_FACEDOWN):
            opp_facedown += 1
    obs[idx] = clampf(<float>opp_facedown, 0.0, 5.0) / 5.0
    idx += 1

    # Opponent S/T zone age (5 slots)
    for slot in range(5):
        card = opp_spells[slot] if slot < len(opp_spells) else None
        if card is not None and card.set_turn > 0:
            obs[idx] = clampf(<float>(current_turn - card.set_turn), 0.0, 10.0) / 10.0
        idx += 1

    # Opponent graveyard monster count
    opp_gy_monsters = 0
    if db is not None:
        opp_graveyard = opp.graveyard
        for i in range(len(opp_graveyard)):
            code_val = opp_graveyard[i]
            if code_val != 0:
                data = db.get(code_val & 0x7FFFFFFF)
                if data is not None and (data["type"] & TYPE_MONSTER):
                    opp_gy_monsters += 1
    obs[idx] = clampf(<float>opp_gy_monsters, 0.0, 10.0) / 10.0
    idx += 1

    # Opponent banished count
    obs[idx] = clampf(<float>len(opp.banished), 0.0, 10.0) / 10.0
    idx += 1

    return obs_arr


# ============================================================
# Card ID encoder: encode_card_ids_fast
# ============================================================
def encode_card_ids_fast(
    object state,
    int perspective,
    object card_index,
) -> np.ndarray:
    """
    Fast Cython replacement for encode_card_ids().

    Returns float32 array of shape (50,).
    """
    cdef np.ndarray[np.float32_t, ndim=1] ids_arr = np.zeros(CARD_ID_DIM, dtype=np.float32)
    cdef float[:] ids = ids_arr
    cdef int idx = 0
    cdef int slot, i, pos
    cdef int gy_len

    me = state.players[perspective]
    opp = state.players[1 - perspective]
    my_monsters = me.monsters
    my_spells = me.spells
    opp_monsters = opp.monsters
    opp_spells = opp.spells

    # My monsters (5 slots, always visible)
    for slot in range(5):
        card = my_monsters[slot] if slot < len(my_monsters) else None
        if card is not None:
            ids[idx] = <float>card_index.code_to_index(card.code)
        idx += 1

    # My S/T (5 slots, always visible)
    for slot in range(5):
        card = my_spells[slot] if slot < len(my_spells) else None
        if card is not None:
            ids[idx] = <float>card_index.code_to_index(card.code)
        idx += 1

    # Opp monsters (face-up only)
    for slot in range(5):
        card = opp_monsters[slot] if slot < len(opp_monsters) else None
        if card is not None:
            pos = card.position
            if pos & POS_FACEUP:
                ids[idx] = <float>card_index.code_to_index(card.code)
        idx += 1

    # Opp S/T (face-up only)
    for slot in range(5):
        card = opp_spells[slot] if slot < len(opp_spells) else None
        if card is not None:
            pos = card.position
            if pos & POS_FACEUP:
                ids[idx] = <float>card_index.code_to_index(card.code)
        idx += 1

    # My hand (up to 10 cards)
    hand = me.hand
    cdef int hand_len = len(hand)
    for i in range(10):
        if i < hand_len:
            code_val = hand[i]
            if code_val != 0:
                ids[idx] = <float>card_index.code_to_index(code_val)
        idx += 1

    # My graveyard (top 10, most recent first)
    my_gy = me.graveyard
    gy_len = len(my_gy)
    # Take the last 10 entries, iterate most-recent-first
    cdef int gy_start = gy_len - 10 if gy_len > 10 else 0
    cdef int gy_avail = gy_len - gy_start  # min(gy_len, 10)
    for i in range(10):
        if i < gy_avail:
            code_val = my_gy[gy_len - 1 - i]
            if code_val != 0:
                ids[idx] = <float>card_index.code_to_index(code_val)
        idx += 1

    # Opp graveyard (top 10, most recent first)
    opp_gy = opp.graveyard
    gy_len = len(opp_gy)
    gy_start = gy_len - 10 if gy_len > 10 else 0
    gy_avail = gy_len - gy_start
    for i in range(10):
        if i < gy_avail:
            code_val = opp_gy[gy_len - 1 - i]
            if code_val != 0:
                ids[idx] = <float>card_index.code_to_index(code_val)
        idx += 1

    return ids_arr


# ============================================================
# Action features encoder: encode_action_features_fast
# ============================================================
def encode_action_features_fast(
    object msg,
    object card_index,
    object db=None,
    int perspective=0,
    object state=None,
) -> np.ndarray:
    """
    Fast Cython replacement for encode_action_features().

    Returns float32 array of shape (71, 29).
    """
    # Lazy imports (same as Python version)
    from ..core.message_parser import (
        MsgSelectIdleCmd,
        MsgSelectBattleCmd,
        MsgSelectCard,
        MsgSelectChain,
        MsgSelectEffectYn,
        MsgSelectTribute,
    )

    cdef np.ndarray[np.float32_t, ndim=2] arr_np = np.zeros(
        (ACTION_FEATURES_SLOTS, ACTION_FEATURES_DIM), dtype=np.float32
    )
    cdef float[:, :] arr = arr_np

    if msg is None:
        return arr_np

    cdef int i, n
    cdef int max_opp_atk
    cdef bint opp_has_facedown
    cdef float my_atk_raw

    if isinstance(msg, MsgSelectIdleCmd):
        # Summonable (slots 0-4)
        cards = msg.summonable
        n = min(len(cards), 5)
        for i in range(n):
            c = cards[i]
            encode_action_slot(arr, 0 + i, c.code, ACT_SUMMON, c.location, card_index, db)

        # SpSummonable (slots 5-9)
        cards = msg.spsummonable
        n = min(len(cards), 5)
        for i in range(n):
            c = cards[i]
            encode_action_slot(arr, 5 + i, c.code, ACT_SPSUMMON, c.location, card_index, db)

        # Setable monsters (slots 10-14)
        cards = msg.setable_monsters
        n = min(len(cards), 5)
        for i in range(n):
            c = cards[i]
            encode_action_slot(arr, 10 + i, c.code, ACT_SET, c.location, card_index, db)

        # Setable S/T (slots 15-19)
        cards = msg.setable_st
        n = min(len(cards), 5)
        for i in range(n):
            c = cards[i]
            encode_action_slot(arr, 15 + i, c.code, ACT_SET, c.location, card_index, db)

        # Activatable (slots 20-29)
        cards = msg.activatable
        n = min(len(cards), 10)
        for i in range(n):
            c = cards[i]
            encode_action_slot(arr, 20 + i, c.code, ACT_ACTIVATE, c.location, card_index, db)

        # Repositionable (slots 30-34)
        cards = msg.repositionable
        n = min(len(cards), 5)
        for i in range(n):
            c = cards[i]
            encode_action_slot(arr, 30 + i, c.code, ACT_REPOSITION, c.location, card_index, db)

        # Phase transitions
        if msg.can_battle_phase:
            arr[35, ACT_OTHER] = 1.0
        if msg.can_end_phase:
            arr[36, ACT_OTHER] = 1.0

    elif isinstance(msg, MsgSelectBattleCmd):
        # C3: Compute opponent's strongest face-up ATK
        max_opp_atk = 0
        opp_has_facedown = False
        if state is not None and db is not None:
            opp_p = state.players[1 - perspective]
            opp_mons = opp_p.monsters
            for slot in range(5):
                m = opp_mons[slot]
                if m is not None:
                    pos = m.position
                    if pos & POS_FACEUP:
                        opp_data = db.get(m.code & 0x7FFFFFFF)
                        if opp_data is not None and opp_data["atk"] >= 0:
                            if opp_data["atk"] > max_opp_atk:
                                max_opp_atk = opp_data["atk"]
                    if pos & POS_FACEDOWN:
                        opp_has_facedown = True

        # Attackable (slots 37-41)
        attackable = msg.attackable
        n = min(len(attackable), 5)
        for i in range(n):
            a = attackable[i]
            if a.card is not None:
                encode_action_slot(arr, 37 + i, a.card.code, ACT_ATTACK, a.card.location, card_index, db)
                # C3: ATK matchup features
                my_atk_raw = arr[37 + i, 8] * 5000.0
                arr[37 + i, 26] = clampf(<float>max_opp_atk, 0.0, 5000.0) / 5000.0
                arr[37 + i, 27] = clampf((my_atk_raw - <float>max_opp_atk) / 5000.0, -1.0, 1.0)
                arr[37 + i, 28] = 1.0 if opp_has_facedown else 0.0

        # Battle activatable (slots 42-46)
        cards = msg.activatable
        n = min(len(cards), 5)
        for i in range(n):
            c = cards[i]
            encode_action_slot(arr, 42 + i, c.code, ACT_ACTIVATE, c.location, card_index, db)

        # Phase transitions
        if msg.can_main2:
            arr[47, ACT_OTHER] = 1.0
        if msg.can_end_phase:
            arr[48, ACT_OTHER] = 1.0

    elif isinstance(msg, (MsgSelectCard, MsgSelectTribute)):
        cards = msg.cards
        n = min(len(cards), 10)
        for i in range(n):
            c = cards[i]
            encode_action_slot(arr, 51 + i, c.code, ACT_OTHER, c.location, card_index, db)
            # C1: Ownership flags
            if c.controller == perspective:
                arr[51 + i, 12] = 1.0
            else:
                arr[51 + i, 13] = 1.0

    elif isinstance(msg, MsgSelectChain):
        cards = msg.cards
        n = min(len(cards), 10)
        for i in range(n):
            c = cards[i]
            encode_action_slot(arr, 51 + i, c.code, ACT_ACTIVATE, c.location, card_index, db)
            # C1: Ownership flags
            if c.controller == perspective:
                arr[51 + i, 12] = 1.0
            else:
                arr[51 + i, 13] = 1.0

    elif isinstance(msg, MsgSelectEffectYn):
        encode_action_slot(arr, 49, msg.code, ACT_ACTIVATE, msg.location, card_index, db)

    return arr_np


# ============================================================
# Action history encoder: encode_action_history_fast
# ============================================================
def encode_action_history_fast(
    object state,
    int perspective,
    object card_index,
    object db=None,
) -> np.ndarray:
    """
    Fast Cython replacement for encode_action_history().

    Returns float32 array of shape (16, 13).
    """
    cdef np.ndarray[np.float32_t, ndim=2] arr_np = np.zeros(
        (HISTORY_LENGTH, HISTORY_FEATURES_DIM), dtype=np.float32
    )
    cdef float[:, :] arr = arr_np

    recent = state.get_recent_actions(HISTORY_LENGTH)
    cdef int num_recent = len(recent)
    cdef int offset = HISTORY_LENGTH - num_recent
    cdef int current_turn = state.current_turn
    cdef int i, row
    cdef int c_atk, c_def
    cdef float damage_val, destroyed_val

    for i in range(num_recent):
        action = recent[i]
        row = offset + i

        if action.card_code != 0:
            arr[row, 0] = <float>card_index.code_to_index(action.card_code)

        # Player relative to perspective
        arr[row, 1] = 1.0 if action.player != perspective else 0.0

        # Action type one-hots
        action_type = action.action_type
        if action_type == "summon":
            arr[row, 2] = 1.0
        elif action_type == "set":
            arr[row, 3] = 1.0
        elif action_type == "activate":
            arr[row, 4] = 1.0
        elif action_type == "attack":
            arr[row, 5] = 1.0
        elif action_type == "draw":
            arr[row, 6] = 1.0

        # Card stats
        if action.card_code != 0 and db is not None:
            data = db.get(action.card_code & 0x7FFFFFFF)
            if data is not None:
                c_atk = data["atk"]
                c_def = data["def"]
                arr[row, 7] = clampf(<float>c_atk, 0.0, 5000.0) / 5000.0 if c_atk >= 0 else 0.0
                arr[row, 8] = clampf(<float>c_def, 0.0, 5000.0) / 5000.0 if c_def >= 0 else 0.0

        # Turns ago
        arr[row, 9] = clampf(<float>(current_turn - action.turn), 0.0, 40.0) / 40.0

        # B1: Outcome features
        extra = action.extra_info
        damage_val = <float>extra.get("damage", 0)
        arr[row, 10] = clampf(damage_val, 0.0, 8000.0) / 8000.0
        destroyed_val = <float>extra.get("destroyed", 0)
        arr[row, 11] = clampf(destroyed_val, 0.0, 5.0) / 5.0
        arr[row, 12] = 1.0 if extra.get("was_negated", False) else 0.0

    return arr_np
