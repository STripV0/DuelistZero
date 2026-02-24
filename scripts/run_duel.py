#!/usr/bin/env python3
"""
Run a complete GOAT duel with random decisions.
This is the end-to-end integration test.

Usage:
    uv run python scripts/run_duel.py
"""

import random
import struct
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from duelist_zero.core.bindings import OcgCore
from duelist_zero.core.callbacks import CallbackManager
from duelist_zero.core.constants import GOAT_DUEL_OPTIONS, MSG
from duelist_zero.core.message_parser import (
    MessageParser,
    RESPONSE_MESSAGES,
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
    MsgWin,
    MsgNewTurn,
    MsgNewPhase,
    MsgDraw,
    MsgSummoning,
    MsgChaining,
    MsgAttack,
    MsgDamage,
    MsgMove,
    try_parse_hint,
    ParsedMessage,
    MsgGeneric,
    MsgHint,
)
from duelist_zero.engine.duel import Duel, load_deck
from duelist_zero.engine.game_state import GameState
from duelist_zero.core.constants import LOCATION, POSITION, PHASE


VERBOSE = True  # Debug: print all message types


def random_response(msg, core, pduel) -> None:
    """Generate a random legal response for any decision message."""

    if isinstance(msg, MsgSelectIdleCmd):
        # Count total options
        total = (len(msg.summonable) + len(msg.spsummonable) +
                 len(msg.repositionable) + len(msg.setable_monsters) +
                 len(msg.setable_st) + len(msg.activatable))
        phase_options = 0
        if msg.can_battle_phase:
            phase_options += 1
        if msg.can_end_phase:
            phase_options += 1

        # Response encoding: (index << 16) | category
        # t (low 16 bits) = category: 0=summon, 1=spsummon, 2=repos, 3=setmon, 4=setst, 5=activate, 6=toBP, 7=toEP, 8=shuffle
        # s (high 16 bits) = index within that category
        if total == 0:
            # No card actions, go to BP or EP
            if msg.can_battle_phase:
                core.set_responsei(pduel, 6)  # t=6, s=0
            else:
                core.set_responsei(pduel, 7)  # t=7, s=0
            return

        # Pick a random action (weighted toward ending phase to avoid infinite loops)
        choice = random.randint(0, total + phase_options - 1)

        offset = 0
        for cat, cards in enumerate([
            msg.summonable, msg.spsummonable, msg.repositionable,
            msg.setable_monsters, msg.setable_st, msg.activatable
        ]):
            if choice < offset + len(cards):
                idx = choice - offset
                core.set_responsei(pduel, (idx << 16) | cat)
                return
            offset += len(cards)

        # Phase transition
        if msg.can_battle_phase and choice == total:
            core.set_responsei(pduel, 6)  # t=6 => to BP
        else:
            core.set_responsei(pduel, 7)  # t=7 => to EP
        return

    elif isinstance(msg, MsgSelectBattleCmd):
        total_atk = len(msg.attackable)
        total_act = len(msg.activatable)
        total = total_atk + total_act

        # Response encoding: (index << 16) | category
        # t=0: activate, t=1: attack, t=2: to M2, t=3: to EP
        if total == 0 or random.random() < 0.3:
            # End battle or go to M2
            if msg.can_main2:
                core.set_responsei(pduel, 2)  # t=2 => to M2
            else:
                core.set_responsei(pduel, 3)  # t=3 => to EP
            return

        choice = random.randint(0, total - 1)
        if choice < total_act:
            core.set_responsei(pduel, (choice << 16) | 0)  # t=0, s=choice
        else:
            idx = choice - total_act
            core.set_responsei(pduel, (idx << 16) | 1)  # t=1, s=idx
        return

    elif isinstance(msg, MsgSelectCard):
        count = random.randint(msg.min_count, msg.max_count)
        indices = random.sample(range(len(msg.cards)), min(count, len(msg.cards)))
        response = struct.pack("B", len(indices))
        for idx in indices:
            response += struct.pack("B", idx)
        core.set_responseb(pduel, response)
        return

    elif isinstance(msg, MsgSelectTribute):
        count = random.randint(msg.min_count, msg.max_count)
        indices = random.sample(range(len(msg.cards)), min(count, len(msg.cards)))
        response = struct.pack("B", len(indices))
        for idx in indices:
            response += struct.pack("B", idx)
        core.set_responseb(pduel, response)
        return

    elif isinstance(msg, MsgSelectChain):
        if not msg.forced and (not msg.cards or random.random() < 0.5):
            core.set_responsei(pduel, -1)  # Don't activate
        else:
            idx = random.randint(0, max(0, len(msg.cards) - 1))
            core.set_responsei(pduel, idx)
        return

    elif isinstance(msg, MsgSelectEffectYn):
        core.set_responsei(pduel, random.randint(0, 1))
        return

    elif isinstance(msg, MsgSelectYesNo):
        core.set_responsei(pduel, random.randint(0, 1))
        return

    elif isinstance(msg, MsgSelectOption):
        core.set_responsei(pduel, random.randint(0, max(0, len(msg.options) - 1)))
        return

    elif isinstance(msg, MsgSelectPosition):
        # Pick a random valid position from the bitmask
        positions = []
        for pos in [POSITION.FACEUP_ATTACK, POSITION.FACEDOWN_ATTACK,
                     POSITION.FACEUP_DEFENSE, POSITION.FACEDOWN_DEFENSE]:
            if msg.positions & pos:
                positions.append(pos)
        if positions:
            core.set_responsei(pduel, random.choice(positions))
        else:
            core.set_responsei(pduel, POSITION.FACEUP_ATTACK)
        return

    elif isinstance(msg, MsgSelectPlace):
        if VERBOSE:
             print(f"    [{msg.msg_type.name}] count={msg.count}, mask=0x{msg.field_mask:08x}")

        # Invert mask to get available zones
        available = ~msg.field_mask & 0xFFFFFFFF
        
        # We need to select `count` zones (or at least 1 if count=0)
        needed = max(1, msg.count)
        valid_zones = []
        
        # The bit layout is relative to the asking player (msg.player):
        #   bits 0-6:   asking player's MZONE slots 0-6
        #   bits 8-12:  asking player's SZONE slots 0-4
        #   bits 16-22: opponent's MZONE slots 0-6
        #   bits 24-28: opponent's SZONE slots 0-4
        # The response bytes are: actual_player(1) + location(1) + sequence(1)
        for bit_group, actual_player in [(0, msg.player), (16, 1 - msg.player)]:
            for loc, bit_offset, sub_count in [(LOCATION.MZONE, 0, 7), (LOCATION.SZONE, 8, 5)]:
                for seq in range(sub_count):
                    bit = bit_group + bit_offset + seq
                    if available & (1 << bit):
                        valid_zones.append((actual_player, loc, seq))
        
        # Randomly select needed amount
        if not valid_zones:
             # Fallback if empty (shouldn't happen)
             selected_zones = [(msg.player, LOCATION.MZONE, 0)]
        else:
             # If we need more than available, take all
             count_to_pick = min(needed, len(valid_zones))
             selected_zones = random.sample(valid_zones, count_to_pick)
        
        resp = b""
        for p, loc, seq in selected_zones:
            resp += struct.pack("BBB", p, loc, seq)
            
        core.set_responseb(pduel, resp)
        return

    # Fallback: send 0 for any unhandled message
    print(f"  ⚠️  Unhandled decision message: {msg.msg_type.name}")
    core.set_responsei(pduel, 0)


def main():
    lib_path = project_root / "lib" / "libocgcore.so"
    db_path = project_root / "data" / "cards.cdb"
    script_dir = project_root / "data" / "script"
    deck_path = project_root / "data" / "deck" / "Goat Control.ydk"

    print("=" * 60)
    print("Duelist Zero — Random Duel")
    print("=" * 60)
    print()

    # Validate files exist
    for path, name in [(lib_path, "libocgcore.so"), (db_path, "cards.cdb"),
                        (deck_path, "deck file")]:
        if not path.exists():
            print(f"❌ {name} not found at {path}")
            sys.exit(1)

    if not script_dir.exists() or not any(script_dir.glob("*.lua")):
        print(f"❌ Lua scripts not found at {script_dir}")
        print("   Run: uv run python scripts/download_data.py")
        sys.exit(1)

    # Load engine
    print("[1] Loading engine...")
    core = OcgCore(lib_path)

    # Setup callbacks
    cb = CallbackManager(core, db_path, script_dir)
    cb.register()
    print("  ✅ Engine loaded, callbacks registered")

    # Load deck
    print("[2] Loading deck...")
    main_deck, extra_deck = load_deck(deck_path)
    print(f"  ✅ Deck loaded: {len(main_deck)} main, {len(extra_deck)} extra")

    # Create duel
    print("[3] Starting duel...")
    seed = random.randint(0, 0xFFFFFFFF)
    pduel = core.create_duel(seed)
    core.set_player_info(pduel, 0, 8000, 5, 1)
    core.set_player_info(pduel, 1, 8000, 5, 1)

    # Load decks for both players (mirror match)
    for player in [0, 1]:
        for code in reversed(main_deck):
            core.new_card(pduel, code, player, player,
                         LOCATION.DECK, 0, POSITION.FACEDOWN_DEFENSE)
        for code in extra_deck:
            core.new_card(pduel, code, player, player,
                         LOCATION.EXTRA, 0, POSITION.FACEDOWN_DEFENSE)

    core.start_duel(pduel, GOAT_DUEL_OPTIONS)
    print(f"  ✅ Duel started (seed={seed})")
    print()

    # Process the duel
    parser = MessageParser()
    turn = 0
    max_turns = 100
    finished = False
    stall_counter = 0
    retry_counter = 0

    while not finished and turn < max_turns:
        result = core.process(pduel)
        msg_data = core.get_message(pduel)

        if not msg_data:
            flag = (result >> 16) & 0xFF
            if flag & 0x02:  # PROCESSOR_END
                print("\n  Duel ended (no winner determined)")
                finished = True
            else:
                stall_counter += 1
                if stall_counter > 50:
                    print(f"\n  ❌ Stall detected — engine stuck in process loop (result=0x{result:08x})")
                    finished = True
            continue
        
        stall_counter = 0  # Reset on progress

        messages = parser.parse(msg_data)

        if VERBOSE:
            msg_names = []
            for m in messages:
                name = m.msg_type.name
                if isinstance(m, MsgChaining):
                    # Lookup card name
                    try:
                       cursor = cb.conn.cursor()
                       cursor.execute("SELECT name FROM texts WHERE id=?", (m.code,))
                       row = cursor.fetchone()
                       if row:
                           name += f"({row[0]})"
                       else:
                           name += f"({m.code})"
                    except:
                       name += f"({m.code})"
                elif isinstance(m, MsgHint):
                    name += f"(type={m.hint_type}, data={m.data})"
                msg_names.append(name)
            print(f"  [msgs: {', '.join(msg_names)}]")

        # Detect RETRY flood (bad responses)
        if len(messages) == 1 and messages[0].msg_type == MSG.RETRY:
            retry_counter += 1
            if retry_counter > 10:
                print(f"\n  ❌ Too many RETRYs — response format is wrong")
                finished = True
                break
            continue
        retry_counter = 0

        for msg in messages:
            # Print interesting events
            if isinstance(msg, MsgNewTurn):
                turn += 1
                print(f"\n{'='*40}")
                print(f"  Turn {turn} — Player {msg.player}")
                print(f"{'='*40}")

            elif isinstance(msg, MsgNewPhase):
                phase_name = {
                    0x01: "Draw", 0x02: "Standby", 0x04: "Main 1",
                    0x08: "Battle Start", 0x10: "Battle Step",
                    0x100: "Main 2", 0x200: "End"
                }.get(msg.phase, f"0x{msg.phase:x}")
                print(f"  📋 Phase: {phase_name}")

            elif isinstance(msg, MsgDraw):
                print(f"  🎴 Player {msg.player} draws {len(msg.cards)} card(s)")

            elif isinstance(msg, MsgSummoning):
                print(f"  ⭐ Player {msg.player} summons card #{msg.code}")

            elif isinstance(msg, MsgChaining):
                print(f"  🔗 Player {msg.player} activates card #{msg.code}")

            elif isinstance(msg, MsgAttack):
                print(f"  ⚔️  Player {msg.attacker_player} attacks!")

            elif isinstance(msg, MsgDamage):
                print(f"  💥 Player {msg.player} takes {msg.amount} damage")

            elif isinstance(msg, MsgWin):
                print(f"\n  🏆 Player {msg.player} wins!")
                finished = True
                break

            elif isinstance(msg, MsgMove):
                pass  # Too noisy

            # Handle decisions
            if msg.msg_type in RESPONSE_MESSAGES:
                if VERBOSE:
                    print(f"  🎯 Responding to: {msg.msg_type.name}")
                random_response(msg, core, pduel)

    if turn >= max_turns:
        print(f"\n  ⏰ Reached turn limit ({max_turns})")

    # Cleanup
    core.end_duel(pduel)
    cb.cleanup()

    print()
    print("=" * 60)
    print("✅ Duel complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
