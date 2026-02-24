#!/usr/bin/env python3
"""
Download GOAT format data files:
- cards.cdb from ProjectIgnis/BabelCDB
- Card Lua scripts from ProjectIgnis/CardScripts

Usage:
    uv run python scripts/download_data.py
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCRIPT_DIR = DATA_DIR / "script"


def run(cmd, **kwargs):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        sys.exit(1)
    return result


def download_cards_cdb():
    """Download cards.cdb from ProjectIgnis/BabelCDB."""
    cdb_path = DATA_DIR / "cards.cdb"
    if cdb_path.exists():
        print(f"  ✅ cards.cdb already exists ({cdb_path.stat().st_size // 1024}KB)")
        return

    print("  Downloading cards.cdb from ProjectIgnis...")
    # The release has a compiled cards.cdb
    url = "https://github.com/ProjectIgnis/BabelCDB/raw/master/cards.cdb"
    run(f'wget -q -O "{cdb_path}" "{url}"')
    print(f"  ✅ Downloaded cards.cdb ({cdb_path.stat().st_size // 1024}KB)")


def download_scripts():
    """Clone card scripts from ProjectIgnis/CardScripts."""
    if SCRIPT_DIR.exists() and any(SCRIPT_DIR.glob("*.lua")):
        count = len(list(SCRIPT_DIR.glob("*.lua")))
        print(f"  ✅ Scripts already exist ({count} lua files)")
        return

    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    print("  Cloning CardScripts (this may take a moment)...")
    tmp_dir = PROJECT_ROOT / "vendor" / "_cardscripts_tmp"
    if tmp_dir.exists():
        run(f'rm -rf "{tmp_dir}"')

    # Shallow clone to save bandwidth
    run(f'git clone --depth 1 https://github.com/ProjectIgnis/CardScripts.git "{tmp_dir}"')

    # Copy all .lua files to data/script/
    lua_files = list(tmp_dir.rglob("*.lua"))
    print(f"  Found {len(lua_files)} Lua scripts, copying...")
    for f in lua_files:
        dest = SCRIPT_DIR / f.name
        if not dest.exists():
            os.link(str(f), str(dest)) if os.name != 'nt' else __import__('shutil').copy2(str(f), str(dest))

    # Also copy the utility scripts (used by card scripts)
    utility_dir = tmp_dir / "utility"
    if utility_dir.exists():
        for f in utility_dir.glob("*.lua"):
            dest = SCRIPT_DIR / f.name
            if not dest.exists():
                os.link(str(f), str(dest)) if os.name != 'nt' else __import__('shutil').copy2(str(f), str(dest))

    # Clean up temp clone
    run(f'rm -rf "{tmp_dir}"')

    count = len(list(SCRIPT_DIR.glob("*.lua")))
    print(f"  ✅ {count} Lua scripts installed")


def main():
    print("=" * 60)
    print("Duelist Zero — Data Setup")
    print("=" * 60)
    print()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/2] Card Database (cards.cdb)")
    download_cards_cdb()
    print()

    print("[2/2] Card Scripts (Lua)")
    download_scripts()
    print()

    print("=" * 60)
    print("✅ Data setup complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
