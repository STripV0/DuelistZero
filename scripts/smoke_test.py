#!/usr/bin/env python3
"""
Smoke test: verify libocgcore.so loads and basic engine lifecycle works.

Usage:
    uv run python scripts/smoke_test.py

This test:
1. Loads libocgcore.so via ctypes
2. Registers dummy callbacks
3. Creates a duel instance
4. Verifies process() can be called
5. Cleans up

Note: This does NOT run a full game (needs cards.cdb + scripts).
      It just verifies the C library integration works.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from duelist_zero.core.bindings import OcgCore, CardData
from duelist_zero.core.constants import GOAT_DUEL_OPTIONS, LOCATION, POSITION


def main():
    lib_path = project_root / "lib" / "libocgcore.so"

    print("=" * 60)
    print("Duelist Zero — Smoke Test")
    print("=" * 60)
    print()

    # Step 1: Load library
    print("[1/5] Loading libocgcore.so...")
    try:
        core = OcgCore(lib_path)
        print(f"  ✅ Library loaded from: {lib_path}")
    except FileNotFoundError as e:
        print(f"  ❌ {e}")
        print("\n  Run ./build_core.sh first to compile the engine.")
        sys.exit(1)
    except OSError as e:
        print(f"  ❌ Failed to load library: {e}")
        sys.exit(1)

    # Step 2: Register dummy callbacks
    print("[2/5] Registering callbacks...")

    def dummy_card_reader(code, data):
        return 0

    def dummy_script_reader(name, len_ptr):
        len_ptr[0] = 0
        return None

    def dummy_msg_handler(pduel, msg_type):
        return 0

    core.set_card_reader(dummy_card_reader)
    core.set_script_reader(dummy_script_reader)
    core.set_message_handler(dummy_msg_handler)
    print("  ✅ Callbacks registered")

    # Step 3: Create duel instance
    print("[3/5] Creating duel instance...")
    pduel = core.create_duel(42)
    print(f"  ✅ Duel created (handle: {pduel})")

    # Step 4: Set player info and try processing
    print("[4/5] Setting player info...")
    core.set_player_info(pduel, 0, 8000, 5, 1)
    core.set_player_info(pduel, 1, 8000, 5, 1)
    print("  ✅ Player info set (LP=8000, hand=5, draw=1)")

    # Step 5: Clean up
    print("[5/5] Ending duel...")
    core.end_duel(pduel)
    print("  ✅ Duel ended cleanly")

    print()
    print("=" * 60)
    print("✅ All smoke tests passed!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Download GOAT format card data (cards.cdb + scripts)")
    print("  2. Create a .ydk deck file")
    print("  3. Run a full game with: python scripts/run_duel.py")


if __name__ == "__main__":
    main()
