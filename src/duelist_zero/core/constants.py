"""
Constants for ygopro-core engine.
Ported from ygopro-core/common.h and yugioh-gamepy/ygo/constants.py
"""

from enum import IntFlag, IntEnum, unique


# ============================================================
# Locations
# ============================================================
@unique
class LOCATION(IntFlag):
    DECK = 0x01
    HAND = 0x02
    MZONE = 0x04
    SZONE = 0x08
    GRAVE = 0x10
    REMOVED = 0x20
    EXTRA = 0x40
    OVERLAY = 0x80
    ONFIELD = MZONE | SZONE
    FZONE = 0x100
    PZONE = 0x200


# ============================================================
# Positions
# ============================================================
@unique
class POSITION(IntFlag):
    FACEUP_ATTACK = 0x1
    FACEDOWN_ATTACK = 0x2
    FACEUP_DEFENSE = 0x4
    FACEDOWN_DEFENSE = 0x8
    FACEUP = FACEUP_ATTACK | FACEUP_DEFENSE
    FACEDOWN = FACEDOWN_ATTACK | FACEDOWN_DEFENSE
    ATTACK = FACEUP_ATTACK | FACEDOWN_ATTACK
    DEFENSE = FACEUP_DEFENSE | FACEDOWN_DEFENSE


# ============================================================
# Card Types
# ============================================================
@unique
class TYPE(IntFlag):
    MONSTER = 0x1
    SPELL = 0x2
    TRAP = 0x4
    NORMAL = 0x10
    EFFECT = 0x20
    FUSION = 0x40
    RITUAL = 0x80
    TRAPMONSTER = 0x100
    SPIRIT = 0x200
    UNION = 0x400
    DUAL = 0x800
    TUNER = 0x1000
    SYNCHRO = 0x2000
    TOKEN = 0x4000
    QUICKPLAY = 0x10000
    CONTINUOUS = 0x20000
    EQUIP = 0x40000
    FIELD = 0x80000
    COUNTER = 0x100000
    FLIP = 0x200000
    TOON = 0x400000
    XYZ = 0x800000
    PENDULUM = 0x1000000
    SPSUMMON = 0x2000000
    LINK = 0x4000000
    # Composite
    EXTRA = FUSION | SYNCHRO | XYZ | LINK


# ============================================================
# Attributes
# ============================================================
@unique
class ATTRIBUTE(IntFlag):
    EARTH = 0x01
    WATER = 0x02
    FIRE = 0x04
    WIND = 0x08
    LIGHT = 0x10
    DARK = 0x20
    DIVINE = 0x40


# ============================================================
# Races (Monster Types)
# ============================================================
@unique
class RACE(IntFlag):
    WARRIOR = 0x1
    SPELLCASTER = 0x2
    FAIRY = 0x4
    FIEND = 0x8
    ZOMBIE = 0x10
    MACHINE = 0x20
    AQUA = 0x40
    PYRO = 0x80
    ROCK = 0x100
    WINGEDBEAST = 0x200
    PLANT = 0x400
    INSECT = 0x800
    THUNDER = 0x1000
    DRAGON = 0x2000
    BEAST = 0x4000
    BEASTWARRIOR = 0x8000
    DINOSAUR = 0x10000
    FISH = 0x20000
    SEASERPENT = 0x40000
    REPTILE = 0x80000
    PSYCHIC = 0x100000
    DIVINE = 0x200000
    CREATORGOD = 0x400000
    WYRM = 0x800000
    CYBERSE = 0x1000000


# ============================================================
# Phases
# ============================================================
@unique
class PHASE(IntFlag):
    DRAW = 0x01
    STANDBY = 0x02
    MAIN1 = 0x04
    BATTLE_START = 0x08
    BATTLE_STEP = 0x10
    DAMAGE = 0x20
    DAMAGE_CAL = 0x40
    BATTLE = 0x80
    MAIN2 = 0x100
    END = 0x200


# ============================================================
# Query Flags (for query_card / query_field_card)
# ============================================================
@unique
class QUERY(IntFlag):
    CODE = 0x1
    POSITION = 0x2
    ALIAS = 0x4
    TYPE = 0x8
    LEVEL = 0x10
    RANK = 0x20
    ATTRIBUTE = 0x40
    RACE = 0x80
    ATTACK = 0x100
    DEFENSE = 0x200
    BASE_ATTACK = 0x400
    BASE_DEFENSE = 0x800
    REASON = 0x1000
    REASON_CARD = 0x2000
    EQUIP_CARD = 0x4000
    TARGET_CARD = 0x8000
    OVERLAY_CARD = 0x10000
    COUNTERS = 0x20000
    OWNER = 0x40000
    STATUS = 0x80000
    LSCALE = 0x200000
    RSCALE = 0x400000
    LINK = 0x800000


# ============================================================
# Reasons
# ============================================================
@unique
class REASON(IntFlag):
    DESTROY = 0x1
    RELEASE = 0x2
    TEMPORARY = 0x4
    MATERIAL = 0x8
    SUMMON = 0x10
    BATTLE = 0x20
    EFFECT = 0x40
    COST = 0x80
    ADJUST = 0x100
    LOST_TARGET = 0x200
    RULE = 0x400
    SPSUMMON = 0x800
    DISSUMMON = 0x1000
    FLIP = 0x2000
    DISCARD = 0x4000
    RDAMAGE = 0x8000
    RRECOVER = 0x10000
    RETURN = 0x20000
    FUSION = 0x40000
    SYNCHRO = 0x80000
    RITUAL = 0x100000
    XYZ = 0x200000
    REPLACE = 0x1000000
    DRAW = 0x2000000
    REDIRECT = 0x4000000
    LINK = 0x10000000


# ============================================================
# Duel Creation Options
# ============================================================
DUEL_TEST_MODE = 0x01
DUEL_ATTACK_FIRST_TURN = 0x02
DUEL_OBSOLETE_RULING = 0x08      # GOAT format: ignition priority / old ruling
DUEL_PSEUDO_SHUFFLE = 0x10
DUEL_TAG_MODE = 0x20
DUEL_SIMPLE_AI = 0x40
DUEL_RETURN_DECK_TOP = 0x80

# GOAT format: only needs obsolete ruling (ignition priority)
GOAT_DUEL_OPTIONS = DUEL_OBSOLETE_RULING


# ============================================================
# Message Types (from ygopro-core process() output)
# These are the first byte of each message in the buffer
# ============================================================
class MSG(IntEnum):
    """Message types from the ygopro-core engine."""
    RETRY = 1
    HINT = 2
    WAITING = 3
    START = 4
    WIN = 5
    UPDATE_DATA = 6
    UPDATE_CARD = 7
    REQUEST_DECK = 8
    SELECT_BATTLECMD = 10
    SELECT_IDLECMD = 11
    SELECT_EFFECTYN = 12
    SELECT_YESNO = 13
    SELECT_OPTION = 14
    SELECT_CARD = 15
    SELECT_CHAIN = 16
    SELECT_PLACE = 18
    SELECT_POSITION = 19
    SELECT_TRIBUTE = 20
    SORT_CHAIN = 21
    SELECT_COUNTER = 22
    SELECT_SUM = 23
    SELECT_DISFIELD = 24
    SORT_CARD = 25
    SELECT_UNSELECT_CARD = 26
    CONFIRM_DECKTOP = 30
    CONFIRM_CARDS = 31
    SHUFFLE_DECK = 32
    SHUFFLE_HAND = 33
    REFRESH_DECK = 34
    SWAP_GRAVE_DECK = 35
    SHUFFLE_SET_CARD = 36
    REVERSE_DECK = 37
    DECK_TOP = 38
    SHUFFLE_EXTRA = 39
    NEW_TURN = 40
    NEW_PHASE = 41
    CONFIRM_EXTRATOP = 42
    MOVE = 50
    POS_CHANGE = 53
    SET = 54
    SWAP = 55
    FIELD_DISABLED = 56
    SUMMONING = 60
    SUMMONED = 61
    SPSUMMONING = 62
    SPSUMMONED = 63
    FLIPSUMMONING = 64
    FLIPSUMMONED = 65
    CHAINING = 70
    CHAINED = 71
    CHAIN_SOLVING = 72
    CHAIN_SOLVED = 73
    CHAIN_END = 74
    CHAIN_NEGATED = 75
    CHAIN_DISABLED = 76
    CARD_SELECTED = 80
    RANDOM_SELECTED = 81
    BECOME_TARGET = 83
    DRAW = 90
    DAMAGE = 91
    RECOVER = 92
    EQUIP = 93
    LPUPDATE = 94
    UNEQUIP = 95
    CARD_TARGET = 96
    CANCEL_TARGET = 97
    PAY_LPCOST = 100
    ADD_COUNTER = 101
    REMOVE_COUNTER = 102
    ATTACK = 110
    BATTLE = 111
    ATTACK_DISABLED = 112
    DAMAGE_STEP_START = 113
    DAMAGE_STEP_END = 114
    MISSED_EFFECT = 120
    BE_CHAIN_TARGET = 121
    CREATE_RELATION = 122
    RELEASE_RELATION = 123
    TOSS_COIN = 130
    TOSS_DICE = 131
    ROCK_PAPER_SCISSORS = 132
    HAND_RES = 133
    ANNOUNCE_RACE = 140
    ANNOUNCE_ATTRIB = 141
    ANNOUNCE_CARD = 142
    ANNOUNCE_NUMBER = 143
    CARD_HINT = 160
    TAG_SWAP = 161
    RELOAD_FIELD = 162
    AI_NAME = 163
    SHOW_HINT = 164
    PLAYER_HINT = 165
    MATCH_KILL = 170
    CUSTOM_MSG = 180


# ============================================================
# Process return flags (from ygopro-core common.h)
# ============================================================
PROCESSOR_BUFFER_LEN = 0x0FFFFFFF
PROCESSOR_FLAG = 0xF0000000
PROCESSOR_WAITING = 0x10000000
PROCESSOR_END = 0x20000000


# ============================================================
# Helpful Mappings
# ============================================================
LOCATION_NAMES = {
    LOCATION.DECK: "Deck",
    LOCATION.HAND: "Hand",
    LOCATION.MZONE: "Monster Zone",
    LOCATION.SZONE: "Spell/Trap Zone",
    LOCATION.GRAVE: "Graveyard",
    LOCATION.REMOVED: "Banished",
    LOCATION.EXTRA: "Extra Deck",
}

PHASE_NAMES = {
    PHASE.DRAW: "Draw Phase",
    PHASE.STANDBY: "Standby Phase",
    PHASE.MAIN1: "Main Phase 1",
    PHASE.BATTLE_START: "Battle Start",
    PHASE.BATTLE_STEP: "Battle Step",
    PHASE.DAMAGE: "Damage Step",
    PHASE.DAMAGE_CAL: "Damage Calc",
    PHASE.BATTLE: "Battle Phase",
    PHASE.MAIN2: "Main Phase 2",
    PHASE.END: "End Phase",
}

ATTRIBUTE_NAMES = {
    ATTRIBUTE.EARTH: "EARTH",
    ATTRIBUTE.WATER: "WATER",
    ATTRIBUTE.FIRE: "FIRE",
    ATTRIBUTE.WIND: "WIND",
    ATTRIBUTE.LIGHT: "LIGHT",
    ATTRIBUTE.DARK: "DARK",
    ATTRIBUTE.DIVINE: "DIVINE",
}
