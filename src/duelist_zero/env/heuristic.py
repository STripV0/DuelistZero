"""
Heuristic action selection and exploration wrapper for GoatEnv.

The heuristic picks "reasonable" YGO actions by priority from the action mask,
without needing observations. Used as the default opponent policy and for
epsilon-greedy exploration during early training.
"""

import random

import numpy as np

from .action_space import (
    _IDLE_SUMMON_START,
    _IDLE_SPSUMMON_START,
    _IDLE_SET_MON_START,
    _IDLE_SET_ST_START,
    _IDLE_ACTIVATE_START,
    _IDLE_BATTLE,
    _IDLE_END,
    _BATTLE_ATTACK_START,
    _BATTLE_ACTIVATE_START,
    _BATTLE_MAIN2,
    _BATTLE_END,
    _YESNO_NO,
    _SELECT_CARD_START,
    _POS_ATK,
    _POS_DEF,
    ACTION_DIM,
)


def heuristic_action(mask: np.ndarray, is_chain: bool = False) -> int:
    """
    Pick a reasonable action from the valid action mask using priority rules.

    Priority order per message type:
      SELECT_IDLECMD: Summon > SpSummon > Battle Phase > Set Monster > End Phase
      SELECT_BATTLECMD: Attack > Main Phase 2 > End Phase
      SELECT_CHAIN: Pass (50) unless forced, then first card (51+)
      SELECT_POSITION: ATK (61) > DEF (62)
      SELECT_CARD/TRIBUTE: First card (prefer selecting over cancelling)
      Everything else: First valid action

    Args:
        mask: Boolean array of shape (ACTION_DIM,) indicating valid actions.
        is_chain: If True, this is a SELECT_CHAIN message (prefer pass over activate).

    Returns:
        Index of the chosen action.
    """
    # Detect message type from which actions are valid
    has_idle = mask[_IDLE_SUMMON_START:_IDLE_END + 1].any()
    has_battle = mask[_BATTLE_ATTACK_START:_BATTLE_END + 1].any()

    if has_idle:
        # SELECT_IDLECMD priority: Summon > SpSummon > Battle > Set Mon > End
        priority_ranges = [
            range(_IDLE_SUMMON_START, _IDLE_SUMMON_START + 5),     # Summon 0-4
            range(_IDLE_SPSUMMON_START, _IDLE_SPSUMMON_START + 5), # SpSummon 0-4
            [_IDLE_BATTLE],                                         # Battle Phase
            range(_IDLE_SET_MON_START, _IDLE_SET_MON_START + 5),   # Set Monster 0-4
            [_IDLE_END],                                            # End Phase
        ]
        for action_range in priority_ranges:
            for idx in action_range:
                if mask[idx]:
                    return idx

    if has_battle:
        # SELECT_BATTLECMD priority: Attack > Main2 > End
        priority_ranges = [
            range(_BATTLE_ATTACK_START, _BATTLE_ATTACK_START + 5),  # Attack 0-4
            [_BATTLE_MAIN2],                                         # Main Phase 2
            [_BATTLE_END],                                           # End Phase
        ]
        for action_range in priority_ranges:
            for idx in action_range:
                if mask[idx]:
                    return idx

    # SELECT_CHAIN: prefer pass (don't chain) unless forced
    if is_chain and mask[_YESNO_NO]:
        return _YESNO_NO

    # SELECT_CARD / SELECT_TRIBUTE: prefer selecting a card over cancelling
    card_actions = np.where(mask[_SELECT_CARD_START:_SELECT_CARD_START + 10])[0]
    if len(card_actions) > 0:
        return int(_SELECT_CARD_START + card_actions[0])

    # SELECT_POSITION: prefer ATK > DEF
    if mask[_POS_ATK]:
        return _POS_ATK
    if mask[_POS_DEF]:
        return _POS_DEF

    # Fallback: first valid action
    valid = np.where(mask)[0]
    if len(valid) > 0:
        return int(valid[0])
    return 0
