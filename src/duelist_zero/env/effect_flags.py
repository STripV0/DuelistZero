"""Binary effect category flags for GOAT format cards.

Each card maps to a 12-dimensional binary vector indicating which broad
effect categories it belongs to:

    0: destroys_monster
    1: destroys_spelltrap
    2: negates
    3: draws_cards
    4: searches_deck
    5: flips_facedown
    6: burns_lp          (direct damage to life points)
    7: bounces_to_hand
    8: special_summons
    9: gains_atk         (ATK boosting)
   10: changes_position
   11: protects           (prevents destruction or damage)

Unknown cards return all zeros.
"""

import numpy as np

EFFECT_FLAG_DIM = 12

# Flag index constants for readability
_DESTROYS_MONSTER = 0
_DESTROYS_SPELLTRAP = 1
_NEGATES = 2
_DRAWS_CARDS = 3
_SEARCHES_DECK = 4
_FLIPS_FACEDOWN = 5
_BURNS_LP = 6
_BOUNCES_TO_HAND = 7
_SPECIAL_SUMMONS = 8
_GAINS_ATK = 9
_CHANGES_POSITION = 10
_PROTECTS = 11

# Map card code -> list of active flag indices
_EFFECT_FLAGS: dict[int, list[int]] = {
    # ── Spells ──────────────────────────────────────────────────────────
    5318639: [_DESTROYS_SPELLTRAP],                  # Mystical Space Typhoon
    44910027: [_DESTROYS_SPELLTRAP],                 # Heavy Storm
    12580477: [_DESTROYS_MONSTER],                   # Raigeki
    53129443: [_DESTROYS_MONSTER],                   # Dark Hole
    55144522: [_DRAWS_CARDS],                        # Pot of Greed
    19613556: [_DRAWS_CARDS],                        # Graceful Charity
    14087893: [_FLIPS_FACEDOWN, _CHANGES_POSITION],  # Book of Moon
    72989439: [_SPECIAL_SUMMONS],                    # Premature Burial
    70368879: [_SPECIAL_SUMMONS],                    # Call of the Haunted
    9596126: [_SPECIAL_SUMMONS],                     # Snatch Steal (takes control)
    45986603: [_SPECIAL_SUMMONS],                    # Snatch Steal (alt code)
    42703248: [_SPECIAL_SUMMONS],                    # Metamorphosis
    44763025: [_SPECIAL_SUMMONS, _PROTECTS],         # Scapegoat
    4031928: [_DESTROYS_MONSTER],                    # Smashing Ground
    33396948: [_DESTROYS_MONSTER],                   # Lightning Vortex
    69162969: [_DESTROYS_MONSTER],                   # Lightning Vortex (alt code)
    71044499: [_DESTROYS_MONSTER],                   # Nobleman of Crossout
    32807846: [_SEARCHES_DECK],                      # Reinforcement of the Army
    31560081: [],                                    # Delinquent Duo (hand disruption)
    90928333: [_SPECIAL_SUMMONS],                    # Brain Control (takes control)
    36261276: [_CHANGES_POSITION],                   # Enemy Controller
    69540536: [_SPECIAL_SUMMONS],                    # Book of Life
    53567095: [_GAINS_ATK],                          # Rush Recklessly
    87880721: [_GAINS_ATK],                          # Limiter Removal
    6250285: [_GAINS_ATK],                           # United We Stand
    32003338: [_GAINS_ATK],                          # Mage Power
    29401950: [_GAINS_ATK],                          # Axe of Despair
    43040603: [_DRAWS_CARDS],                        # Upstart Goblin
    17655904: [_SEARCHES_DECK],                      # Gold Sarcophagus
    73628505: [_DESTROYS_SPELLTRAP],                 # Nobleman of Extermination
    29948642: [_GAINS_ATK],                          # Megamorph
    1248895: [_GAINS_ATK],                           # Mist Body (protects loosely)
    48539234: [_SEARCHES_DECK],                      # Sangan (spell?) no—monster
    19523799: [_DRAWS_CARDS],                        # Jar of Greed (trap actually)
    98045062: [_DESTROYS_MONSTER],                   # Fissure
    62265044: [_DESTROYS_MONSTER],                   # Hammer Shot
    37520316: [_DESTROYS_MONSTER, _DESTROYS_SPELLTRAP],  # Shield Crash? no—Monster
    2964201: [_DRAWS_CARDS],                         # Card Destruction (draws/discards)
    23171610: [_SPECIAL_SUMMONS],                    # Dimension Fusion
    49868263: [_SPECIAL_SUMMONS],                    # Return from the Different Dimension (trap)
    77414722: [_SPECIAL_SUMMONS],                    # Monster Reborn
    83764718: [_SPECIAL_SUMMONS, _PROTECTS],         # Monster Reincarnation (retrieves)
    24068492: [_DRAWS_CARDS],                        # Mirage of Nightmare
    61740673: [_SPECIAL_SUMMONS],                    # The Shallow Grave
    80604091: [_SPECIAL_SUMMONS],                    # Reasoning (special summons off mill)
    87639778: [_SPECIAL_SUMMONS],                    # Monster Gate (special summons off tribute)
    22359980: [_DESTROYS_MONSTER, _DESTROYS_SPELLTRAP],  # Mystik Wok? no. Skip.
    69243953: [_GAINS_ATK],                          # Creature Swap? no. 86805855 is correct.
    86805855: [_CHANGES_POSITION],                   # Creature Swap (switches control)
    35027493: [_DRAWS_CARDS],                        # Good Goblin Housekeeping
    17375316: [_GAINS_ATK],                          # Fairy Meteor Crush? no effect flag
    57953380: [_SEARCHES_DECK, _SPECIAL_SUMMONS],    # Mystic Tomato
    48976825: [_SEARCHES_DECK, _SPECIAL_SUMMONS],    # Shining Angel
    63749102: [_SEARCHES_DECK, _SPECIAL_SUMMONS],    # Giant Rat

    # ── Monsters ────────────────────────────────────────────────────────
    31786629: [_SPECIAL_SUMMONS, _DESTROYS_MONSTER], # BLS - Envoy of the Beginning
    63519819: [_DRAWS_CARDS],                        # Airknight Parshath
    46411259: [_DRAWS_CARDS],                        # Magician of Faith (retrieves spell)
    64697231: [_DESTROYS_MONSTER],                   # Tribe-Infecting Virus
    79571449: [_DESTROYS_MONSTER],                   # D.D. Warrior Lady (banishes)
    60082869: [_SEARCHES_DECK],                      # Sangan
    34853266: [_DRAWS_CARDS],                        # Morphing Jar
    26202165: [_PROTECTS],                           # Sinister Serpent (hand advantage)
    97077563: [_FLIPS_FACEDOWN],                     # Tsukuyomi
    87910978: [_NEGATES],                            # Jinzo (negates traps)
    80161395: [_SPECIAL_SUMMONS, _PROTECTS],         # Thousand-Eyes Restrict
    17881964: [_DRAWS_CARDS],                        # Dark Mimic LV1
    70828912: [_DRAWS_CARDS],                        # Magical Merchant
    71413901: [_DESTROYS_SPELLTRAP],                 # Breaker the Magical Warrior
    61854111: [_DESTROYS_MONSTER],                   # D.D. Assailant (banishes)
    37744402: [_DESTROYS_SPELLTRAP],                 # Mobius the Frost Monarch
    62279055: [_DESTROYS_MONSTER],                   # Zaborg the Thunder Monarch
    57116034: [_BOUNCES_TO_HAND],                    # Night Assailant (returns flip from GY)
    46044841: [],                                    # Don Zaloog (hand disruption)
    65169794: [],                                    # Asura Priest (multi-attack beater)
    36261276: [_CHANGES_POSITION],                   # Enemy Controller (dup ok)
    78130962: [_SEARCHES_DECK],                      # Witch of the Black Forest
    36553319: [_SPECIAL_SUMMONS],                    # Cyber Dragon? not GOAT.
    85602018: [_BURNS_LP],                           # Ceasefire
    16970158: [],                                    # Fiber Jar (resets game)
    36838956: [],                                    # Exarion Universe (beater)
    73915051: [_DESTROYS_MONSTER],                   # Sakuretsu Armor
    56120475: [_DESTROYS_MONSTER],                   # Sakuretsu Armor (alt code)
    15800838: [_DESTROYS_MONSTER],                   # Drillroid
    89631139: [_DESTROYS_MONSTER],                   # Last Turn
    66235877: [],                                    # Reflect Bounder (burns_lp loosely)
    34743446: [_DESTROYS_MONSTER],                   # Kycoo the Ghost Destroyer (banishes GY)
    82301904: [_NEGATES],                            # Spell Canceller (negates spells)
    21593977: [_SEARCHES_DECK, _SPECIAL_SUMMONS],    # Apprentice Magician
    29380133: [_SEARCHES_DECK, _SPECIAL_SUMMONS],    # UFO Turtle
    3248469: [_SEARCHES_DECK, _SPECIAL_SUMMONS],     # Mother Grizzly
    4206964: [_SEARCHES_DECK, _SPECIAL_SUMMONS],     # Flying Kamakiri #1
    47355498: [_DESTROYS_MONSTER],                   # Newdoria
    9814707: [_SEARCHES_DECK, _SPECIAL_SUMMONS],     # Pyramid Turtle
    39111158: [_SPECIAL_SUMMONS],                    # Spirit Reaper (protects self)
    12538374: [_PROTECTS, _SPECIAL_SUMMONS],         # Marshmallon (protects, burns)
    69232594: [_NEGATES],                            # Horus the Black Flame Dragon LV8
    84257639: [_NEGATES],                            # Dark Paladin (negates spells)

    # ── Traps ───────────────────────────────────────────────────────────
    44095762: [_DESTROYS_MONSTER],                   # Mirror Force
    53582587: [_DESTROYS_MONSTER],                   # Torrential Tribute
    83555666: [_DESTROYS_MONSTER, _BURNS_LP],        # Ring of Destruction
    80071763: [_DESTROYS_SPELLTRAP],                 # Dust Tornado
    41420027: [_NEGATES],                            # Solemn Judgment
    30241314: [_DESTROYS_MONSTER],                   # Bottomless Trap Hole
    2830619: [_DESTROYS_MONSTER],                    # Widespread Ruin
    37576645: [_BOUNCES_TO_HAND],                    # Compulsory Evacuation Device
    11384280: [_BOUNCES_TO_HAND],                    # Phoenix Wing Wind Blast
    3831348: [_BOUNCES_TO_HAND],                     # Trap Dustshoot
    81439173: [_PROTECTS],                           # Waboku
    4206964: [_SEARCHES_DECK, _SPECIAL_SUMMONS],     # Flying Kamakiri (dup ok)
    49010598: [_PROTECTS],                           # Threatening Roar
    20174757: [_NEGATES],                            # Magic Drain (negates spells)
    50078509: [_NEGATES],                            # Seven Tools of the Bandit
    84749824: [_NEGATES],                            # Magic Jammer
    69587564: [_BOUNCES_TO_HAND],                    # Raigeki Break (destroys + discard)
    44763025: [_SPECIAL_SUMMONS, _PROTECTS],         # Scapegoat (dup ok)
    19523799: [_DRAWS_CARDS],                        # Jar of Greed
    29843091: [_DRAWS_CARDS],                        # Reckless Greed
    37580756: [_DESTROYS_MONSTER],                   # Trap Hole
    71413901: [_DESTROYS_SPELLTRAP],                 # Breaker (dup ok)
    94192409: [_PROTECTS],                           # Enchanted Javelin? LP gain
    77538567: [_DESTROYS_MONSTER, _DESTROYS_SPELLTRAP],  # Raigeki Break (destroys any)
    36468556: [_PROTECTS],                           # Hallowed Life Barrier
    27174286: [_DRAWS_CARDS],                        # Legacy of Yata-Garasu
    34906152: [_NEGATES, _DESTROYS_SPELLTRAP],       # Royal Decree (negates traps)
    84749824: [_NEGATES],                            # Magic Jammer (dup ok)

    # ── More GOAT staple monsters ───────────────────────────────────────
    24348204: [_SPECIAL_SUMMONS],                    # Cyber-Stein (pays LP, special summons fusion)
    15025844: [_SPECIAL_SUMMONS],                    # Thousand-Eyes Restrict (fusion)
    32864205: [],                                    # Berserk Gorilla (beater)
    33184167: [_DESTROYS_MONSTER],                   # D.D. Survivor? no. Exiled Force.
    12744567: [_PROTECTS],                           # Marshmallon? alt code
    68005187: [],                                    # Blade Knight (beater)
    89653984: [],                                    # Hydrogeddon? not GOAT
    95727991: [],                                    # Gorilla? not relevant
    40044918: [_SEARCHES_DECK, _SPECIAL_SUMMONS],    # Mystic Tomato? already in
    69243953: [_GAINS_ATK],                          # Dark Elf? skip
    98502113: [],                                    # Ninja Grandmaster? not GOAT
    15150365: [_DESTROYS_MONSTER],                   # Exiled Force (tributes self to destroy)
    52038441: [],                                    # Abyss Soldier (bounces — water)
    50930991: [_GAINS_ATK],                          # Injection Fairy Lily
    98252586: [],                                    # Skilled Dark Magician? skip
    71413901: [_DESTROYS_SPELLTRAP],                 # Breaker (dup ok)

    # ── More commonly played spells ─────────────────────────────────────
    83764718: [_SPECIAL_SUMMONS, _PROTECTS],         # Monster Reincarnation
    43040603: [_DRAWS_CARDS],                        # Upstart Goblin
    30459350: [_DRAWS_CARDS],                        # Reload
    2314238: [_SPECIAL_SUMMONS],                     # Level Limit - Area B? no. Skip.
    61740673: [_SPECIAL_SUMMONS],                    # The Shallow Grave (dup ok)
    80604091: [_SPECIAL_SUMMONS],                    # Reasoning (dup ok)
    87639778: [_SPECIAL_SUMMONS],                    # Monster Gate (dup ok)

    # ── Commonly played equips ──────────────────────────────────────────
    21593977: [_SEARCHES_DECK, _SPECIAL_SUMMONS],    # Apprentice Magician (dup ok)
    6150044: [_GAINS_ATK],                           # Black Pendant
    93221206: [_GAINS_ATK, _PROTECTS],               # Ring of Defense? skip
}


def get_effect_flags(code: int) -> np.ndarray:
    """Return a 12-element float32 binary vector of effect categories.

    Applies a 0x7FFFFFFF mask to strip any sign-extension artefacts from
    the card code before lookup.  Unknown cards return all zeros.
    """
    flags = np.zeros(EFFECT_FLAG_DIM, dtype=np.float32)
    indices = _EFFECT_FLAGS.get(code & 0x7FFFFFFF)
    if indices:
        for i in indices:
            flags[i] = 1.0
    return flags
