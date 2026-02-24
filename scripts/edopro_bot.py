#!/usr/bin/env python3
"""
Connect a trained MaskablePPO agent to EDOPro as a bot opponent.

Usage:
    1. Open EDOPro and host a LAN game (GOAT format)
    2. Run: uv run python scripts/edopro_bot.py
    3. The bot connects and plays using the trained model

Options:
    uv run python scripts/edopro_bot.py --host 127.0.0.1 --port 7911
    uv run python scripts/edopro_bot.py --model checkpoints/final_model
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from duelist_zero.network.bot import EdoProBot


def main():
    parser = argparse.ArgumentParser(
        description="Connect trained YuGiOh AI to EDOPro as a bot opponent"
    )
    parser.add_argument("--model", type=str, default="checkpoints/final_model",
                        help="Path to trained MaskablePPO model (default: checkpoints/final_model)")
    parser.add_argument("--deck", type=str, default=None,
                        help="Path to .ydk deck file (default: data/deck/goat_control.ydk)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="EDOPro host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7911,
                        help="EDOPro host port (default: 7911)")
    parser.add_argument("--name", type=str, default="DuelistZero",
                        help="Bot player name (default: DuelistZero)")
    parser.add_argument("--version", type=str, default="0x1360",
                        help="Protocol version hex (default: 0x1360)")
    args = parser.parse_args()

    kwargs = {
        "model_path": args.model,
        "host": args.host,
        "port": args.port,
        "name": args.name,
        "version": int(args.version, 16),
    }
    if args.deck:
        kwargs["deck_path"] = args.deck

    bot = EdoProBot(**kwargs)
    bot.connect()
    bot.run()


if __name__ == "__main__":
    main()
