"""
Diagnostic script: Load a MaskablePPO checkpoint and play 5 games
against the heuristic opponent, logging every agent decision.

Usage:
    uv run python scripts/diagnose_agent.py
"""

import sys
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

from sb3_contrib import MaskablePPO

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from duelist_zero.env.goat_env import GoatEnv
from duelist_zero.env.action_space import (
    ACTION_DIM,
    _IDLE_SUMMON_START,
    _IDLE_SPSUMMON_START,
    _IDLE_SET_MON_START,
    _IDLE_SET_ST_START,
    _IDLE_ACTIVATE_START,
    _IDLE_REPOSITION_START,
    _IDLE_BATTLE,
    _IDLE_END,
    _BATTLE_ATTACK_START,
    _BATTLE_ACTIVATE_START,
    _BATTLE_MAIN2,
    _BATTLE_END,
    _YESNO_YES,
    _YESNO_NO,
    _SELECT_CARD_START,
    _POS_ATK,
    _POS_DEF,
    _OPTION_START,
)
from duelist_zero.core.message_parser import (
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
# Human-readable action name mapping
# ============================================================
def action_name(idx: int) -> str:
    """Map a flat action index to a human-readable name."""
    if _IDLE_SUMMON_START <= idx < _IDLE_SUMMON_START + 5:
        return f"IDLE_SUMMON[{idx - _IDLE_SUMMON_START}]"
    if _IDLE_SPSUMMON_START <= idx < _IDLE_SPSUMMON_START + 5:
        return f"IDLE_SPSUMMON[{idx - _IDLE_SPSUMMON_START}]"
    if _IDLE_SET_MON_START <= idx < _IDLE_SET_MON_START + 5:
        return f"IDLE_SET_MON[{idx - _IDLE_SET_MON_START}]"
    if _IDLE_SET_ST_START <= idx < _IDLE_SET_ST_START + 5:
        return f"IDLE_SET_ST[{idx - _IDLE_SET_ST_START}]"
    if _IDLE_ACTIVATE_START <= idx < _IDLE_ACTIVATE_START + 10:
        return f"IDLE_ACTIVATE[{idx - _IDLE_ACTIVATE_START}]"
    if _IDLE_REPOSITION_START <= idx < _IDLE_REPOSITION_START + 5:
        return f"IDLE_REPOSITION[{idx - _IDLE_REPOSITION_START}]"
    if idx == _IDLE_BATTLE:
        return "IDLE_TO_BATTLE"
    if idx == _IDLE_END:
        return "IDLE_TO_END"
    if _BATTLE_ATTACK_START <= idx < _BATTLE_ATTACK_START + 5:
        return f"BATTLE_ATTACK[{idx - _BATTLE_ATTACK_START}]"
    if _BATTLE_ACTIVATE_START <= idx < _BATTLE_ACTIVATE_START + 5:
        return f"BATTLE_ACTIVATE[{idx - _BATTLE_ACTIVATE_START}]"
    if idx == _BATTLE_MAIN2:
        return "BATTLE_TO_M2"
    if idx == _BATTLE_END:
        return "BATTLE_TO_END"
    if idx == _YESNO_YES:
        return "YES"
    if idx == _YESNO_NO:
        return "NO/PASS"
    if _SELECT_CARD_START <= idx < _SELECT_CARD_START + 10:
        return f"SELECT_CARD[{idx - _SELECT_CARD_START}]"
    if idx == _POS_ATK:
        return "POS_ATK"
    if idx == _POS_DEF:
        return "POS_DEF"
    if _OPTION_START <= idx <= 70:
        return f"SELECT_OPTION[{idx - _OPTION_START}]"
    return f"UNKNOWN[{idx}]"


def action_category(idx: int) -> str:
    """Map action index to a broad category for aggregate counting."""
    if _IDLE_SUMMON_START <= idx < _IDLE_SUMMON_START + 5:
        return "summon"
    if _IDLE_SPSUMMON_START <= idx < _IDLE_SPSUMMON_START + 5:
        return "sp_summon"
    if _IDLE_SET_MON_START <= idx < _IDLE_SET_MON_START + 5:
        return "set_monster"
    if _IDLE_SET_ST_START <= idx < _IDLE_SET_ST_START + 5:
        return "set_spell_trap"
    if _IDLE_ACTIVATE_START <= idx < _IDLE_ACTIVATE_START + 10:
        return "idle_activate"
    if _IDLE_REPOSITION_START <= idx < _IDLE_REPOSITION_START + 5:
        return "reposition"
    if idx == _IDLE_BATTLE:
        return "go_battle"
    if idx == _IDLE_END:
        return "go_end"
    if _BATTLE_ATTACK_START <= idx < _BATTLE_ATTACK_START + 5:
        return "attack"
    if _BATTLE_ACTIVATE_START <= idx < _BATTLE_ACTIVATE_START + 5:
        return "battle_activate"
    if idx == _BATTLE_MAIN2:
        return "go_main2"
    if idx == _BATTLE_END:
        return "battle_end"
    if idx == _YESNO_YES:
        return "yes"
    if idx == _YESNO_NO:
        return "no/pass"
    if _SELECT_CARD_START <= idx < _SELECT_CARD_START + 10:
        return "select_card"
    if idx == _POS_ATK:
        return "pos_atk"
    if idx == _POS_DEF:
        return "pos_def"
    if _OPTION_START <= idx <= 70:
        return "select_option"
    return "unknown"


def mask_summary(mask: np.ndarray) -> str:
    """Return short description of which action groups are available."""
    valid = np.where(mask)[0]
    groups = Counter()
    for v in valid:
        groups[action_category(v)] += 1
    parts = [f"{cat}({cnt})" for cat, cnt in sorted(groups.items())]
    return ", ".join(parts)


def msg_type_name(msg) -> str:
    """Return human-readable message type name."""
    return type(msg).__name__


def idle_detail(msg, mask):
    """For idle commands, show what options are available."""
    parts = []
    if hasattr(msg, 'summonable') and len(msg.summonable) > 0:
        parts.append(f"can_summon={len(msg.summonable)}")
    if hasattr(msg, 'spsummonable') and len(msg.spsummonable) > 0:
        parts.append(f"can_spsummon={len(msg.spsummonable)}")
    if hasattr(msg, 'setable_monsters') and len(msg.setable_monsters) > 0:
        parts.append(f"can_set_mon={len(msg.setable_monsters)}")
    if hasattr(msg, 'setable_st') and len(msg.setable_st) > 0:
        parts.append(f"can_set_st={len(msg.setable_st)}")
    if hasattr(msg, 'activatable') and len(msg.activatable) > 0:
        parts.append(f"can_activate={len(msg.activatable)}")
    if hasattr(msg, 'repositionable') and len(msg.repositionable) > 0:
        parts.append(f"can_reposition={len(msg.repositionable)}")
    if hasattr(msg, 'can_battle_phase') and msg.can_battle_phase:
        parts.append("can_BP")
    if hasattr(msg, 'can_end_phase') and msg.can_end_phase:
        parts.append("can_EP")
    return " | ".join(parts) if parts else "none"


def battle_detail(msg, mask):
    """For battle commands, show what options are available."""
    parts = []
    if hasattr(msg, 'attackable') and len(msg.attackable) > 0:
        parts.append(f"can_attack={len(msg.attackable)}")
    if hasattr(msg, 'activatable') and len(msg.activatable) > 0:
        parts.append(f"can_activate={len(msg.activatable)}")
    if hasattr(msg, 'can_main2') and msg.can_main2:
        parts.append("can_M2")
    if hasattr(msg, 'can_end_phase') and msg.can_end_phase:
        parts.append("can_EP")
    return " | ".join(parts) if parts else "none"


# ============================================================
# Main diagnostic
# ============================================================
def main():
    ckpt_path = PROJECT_ROOT / "checkpoints" / "ckpt_00500000"
    n_games = 5

    print("=" * 70)
    print("AGENT DIAGNOSTIC - 500k checkpoint vs heuristic")
    print("=" * 70)
    print(f"Checkpoint: {ckpt_path}")
    print()

    # Create env (heuristic opponent by default when opponent_fn=None)
    env = GoatEnv()

    # Load model
    model = MaskablePPO.load(str(ckpt_path))
    print(f"Model loaded. Device: {model.device}")
    print()

    # Aggregate stats
    total_action_counts = Counter()
    total_msg_type_counts = Counter()
    game_results = []

    # Track idle decision patterns specifically
    idle_choices = Counter()  # what the agent picks during IDLE
    battle_choices = Counter()  # what the agent picks during BATTLE
    idle_had_summon_but_skipped = 0  # times agent could summon but didn't
    idle_had_attack_but_skipped = 0  # times agent could attack but didn't
    total_idle_decisions = 0
    total_battle_decisions = 0

    for game_idx in range(n_games):
        print(f"\n{'='*70}")
        print(f"GAME {game_idx + 1}")
        print(f"{'='*70}")

        obs, info = env.reset()
        agent_player = env._agent_player
        print(f"Agent is player {agent_player}")

        done = False
        step = 0
        ep_reward = 0.0
        lp_history = []
        game_action_counts = Counter()

        while not done and step < 300:
            msg = env._pending_msg
            mask = env.valid_action_mask()
            valid_actions = np.where(mask)[0]

            # Get agent's action (deterministic)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            action = int(action)

            # Log the decision
            msg_name = msg_type_name(msg) if msg else "None"
            act_name = action_name(action)
            cat = action_category(action)

            # Detailed logging for every decision
            detail = ""
            if isinstance(msg, MsgSelectIdleCmd):
                detail = idle_detail(msg, mask)
                total_idle_decisions += 1
                idle_choices[cat] += 1

                # Track missed opportunities
                could_summon = mask[_IDLE_SUMMON_START:_IDLE_SUMMON_START + 5].any()
                could_spsummon = mask[_IDLE_SPSUMMON_START:_IDLE_SPSUMMON_START + 5].any()
                if (could_summon or could_spsummon) and cat not in ("summon", "sp_summon"):
                    idle_had_summon_but_skipped += 1

            elif isinstance(msg, MsgSelectBattleCmd):
                detail = battle_detail(msg, mask)
                total_battle_decisions += 1
                battle_choices[cat] += 1

                could_attack = mask[_BATTLE_ATTACK_START:_BATTLE_ATTACK_START + 5].any()
                if could_attack and cat != "attack":
                    idle_had_attack_but_skipped += 1

            print(f"  Step {step:3d} | {msg_name:25s} | "
                  f"valid=[{mask_summary(mask)}] | "
                  f"chose={act_name:25s} ({cat})")
            if detail:
                print(f"          | detail: {detail}")

            # Track counts
            game_action_counts[cat] += 1
            total_action_counts[cat] += 1
            total_msg_type_counts[msg_name] += 1

            # Step the env
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step += 1
            done = terminated or truncated

            # LP snapshots every 20 steps
            if step % 20 == 0 or done:
                state = env._duel.state if env._duel else None
                if state:
                    agent_lp = state.players[agent_player].lp
                    opp_lp = state.players[1 - agent_player].lp
                    agent_mons = sum(1 for m in state.players[agent_player].monsters if m is not None)
                    opp_mons = sum(1 for m in state.players[1 - agent_player].monsters if m is not None)
                    lp_history.append((step, agent_lp, opp_lp, agent_mons, opp_mons))
                    if step % 20 == 0:
                        print(f"    >>> LP checkpoint: step={step} agent_LP={agent_lp} opp_LP={opp_lp} "
                              f"agent_mons={agent_mons} opp_mons={opp_mons}")

        # Game result
        winner = info.get("winner", None)
        if winner is not None:
            if winner == agent_player:
                result = "WIN"
            elif winner == (1 - agent_player):
                result = "LOSS"
            else:
                result = "DRAW"
        elif truncated:
            result = "TRUNCATED"
        else:
            result = "UNKNOWN"

        game_results.append(result)

        print(f"\n  --- GAME {game_idx + 1} RESULT: {result} in {step} steps "
              f"(cumulative reward: {ep_reward:.3f}) ---")
        print(f"  LP progression:")
        for s, alp, olp, am, om in lp_history:
            print(f"    step {s:3d}: agent_LP={alp:5d}  opp_LP={olp:5d}  "
                  f"agent_mons={am}  opp_mons={om}")
        print(f"  Action distribution this game:")
        for cat, cnt in sorted(game_action_counts.items(), key=lambda x: -x[1]):
            print(f"    {cat:20s}: {cnt:4d}")

    # ============================================================
    # Aggregate summary
    # ============================================================
    env.close()

    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY")
    print("=" * 70)

    print(f"\nGame results: {game_results}")
    wins = sum(1 for r in game_results if r == "WIN")
    losses = sum(1 for r in game_results if r == "LOSS")
    truncs = sum(1 for r in game_results if r == "TRUNCATED")
    print(f"  Wins: {wins}/{n_games}  Losses: {losses}/{n_games}  Truncated: {truncs}/{n_games}")

    print(f"\n--- Action Distribution (all {n_games} games) ---")
    total = sum(total_action_counts.values())
    for cat, cnt in sorted(total_action_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * cnt / total if total > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {cat:20s}: {cnt:5d} ({pct:5.1f}%) {bar}")

    print(f"\n--- Message Type Distribution ---")
    for msg, cnt in sorted(total_msg_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {msg:30s}: {cnt:5d}")

    print(f"\n--- Idle Phase Decisions ({total_idle_decisions} total) ---")
    for cat, cnt in sorted(idle_choices.items(), key=lambda x: -x[1]):
        pct = 100.0 * cnt / total_idle_decisions if total_idle_decisions > 0 else 0
        print(f"  {cat:20s}: {cnt:5d} ({pct:5.1f}%)")

    print(f"\n--- Battle Phase Decisions ({total_battle_decisions} total) ---")
    for cat, cnt in sorted(battle_choices.items(), key=lambda x: -x[1]):
        pct = 100.0 * cnt / total_battle_decisions if total_battle_decisions > 0 else 0
        print(f"  {cat:20s}: {cnt:5d} ({pct:5.1f}%)")

    print(f"\n--- Missed Opportunities ---")
    print(f"  Could summon but didn't: {idle_had_summon_but_skipped} / {total_idle_decisions} idle decisions")
    print(f"  Could attack but didn't: {idle_had_attack_but_skipped} / {total_battle_decisions} battle decisions")

    # Key diagnostic questions
    print(f"\n--- KEY DIAGNOSTIC ANSWERS ---")
    summon_total = total_action_counts.get("summon", 0) + total_action_counts.get("sp_summon", 0)
    attack_total = total_action_counts.get("attack", 0)
    end_total = total_action_counts.get("go_end", 0) + total_action_counts.get("battle_end", 0)
    pass_total = total_action_counts.get("no/pass", 0)
    set_total = total_action_counts.get("set_monster", 0) + total_action_counts.get("set_spell_trap", 0)

    print(f"  Total summons:     {summon_total}")
    print(f"  Total attacks:     {attack_total}")
    print(f"  Total sets:        {set_total}")
    print(f"  Total end/pass:    {end_total + pass_total}")
    print(f"  Is agent passive?  {'YES - agent is barely summoning/attacking!' if (summon_total + attack_total) < (end_total + pass_total) else 'No - agent is taking proactive actions'}")


if __name__ == "__main__":
    main()
