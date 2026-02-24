"""
Callback implementations for ygopro-core.

The C engine calls back into Python for two things:
1. card_reader: Look up card data (ATK, DEF, etc.) from cards.cdb
2. script_reader: Load Lua effect scripts from disk

Reference: yugioh-gamepy/ygo/duel.py
"""

import ctypes
import os
import sqlite3
from ctypes import POINTER, c_byte, c_int32
from pathlib import Path
from typing import Optional

from .bindings import CardData, OcgCore


class CardDatabase:
    """SQLite interface to cards.cdb for the card_reader callback."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Card database not found: {self.db_path}")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        # Cache for performance
        self._cache: dict[int, dict] = {}

    def get_card(self, code: int) -> Optional[dict]:
        """Fetch card data by ID. Returns None if not found."""
        if code in self._cache:
            return self._cache[code]

        row = self.cursor.execute(
            "SELECT * FROM datas WHERE id=?", (code,)
        ).fetchone()

        if row is None:
            return None

        data = dict(row)
        self._cache[code] = data
        return data

    def close(self):
        self.conn.close()


class CallbackManager:
    """
    Manages the card_reader and script_reader callbacks for ygopro-core.

    Usage:
        cb_manager = CallbackManager(core, db_path, script_dir)
        cb_manager.register()
        # Now the engine can read cards and scripts via callbacks
    """

    def __init__(self, core: OcgCore, db_path: str | Path, script_dir: str | Path):
        self.core = core
        self.db = CardDatabase(db_path)
        self.script_dir = str(script_dir)

        # Buffer for script data (reused across calls)
        self._script_buf = (c_byte * 131072)()  # 128KB buffer

    def register(self):
        """Register both callbacks with the engine."""
        self.core.set_card_reader(self._card_reader)
        self.core.set_script_reader(self._script_reader)
        self.core.set_message_handler(self._message_handler)

    def _card_reader(self, code: int, data_ptr: POINTER(CardData)) -> int:
        """
        Called by the engine when it needs card data.
        Reads from cards.cdb and fills the CardData struct.
        """
        card = self.db.get_card(code)
        if card is None:
            return 1  # error

        cd = data_ptr[0]
        cd.code = code
        cd.alias = card.get("alias", 0)

        # setcode is stored as a single uint64 in the DB, split into array
        setcode_val = card.get("setcode", 0)
        cd.setcode[0] = setcode_val & 0xFFFF
        cd.setcode[1] = (setcode_val >> 16) & 0xFFFF
        cd.setcode[2] = (setcode_val >> 32) & 0xFFFF
        cd.setcode[3] = (setcode_val >> 48) & 0xFFFF

        cd.type = card.get("type", 0)

        level_val = card.get("level", 0)
        cd.level = level_val & 0xFF
        cd.lscale = (level_val >> 24) & 0xFF
        cd.rscale = (level_val >> 16) & 0xFF

        cd.attribute = card.get("attribute", 0)
        cd.race = card.get("race", 0)
        cd.attack = card.get("atk", 0)
        cd.defense = card.get("def", 0)

        # Link monsters store link marker in defense field
        if cd.type & 0x4000000:  # TYPE_LINK
            cd.link_marker = cd.defense
            cd.defense = 0
        else:
            cd.link_marker = 0

        return 0  # success

    def _script_reader(
        self, name: bytes, len_ptr: POINTER(c_int32)
    ):
        """
        Called by the engine when it needs a Lua card script.
        Reads the .lua file from the script directory.
        Returns c_void_p (address of buffer) or None.
        """
        script_name = name.decode("utf-8", errors="replace") if isinstance(name, bytes) else name

        # The engine requests paths like "./script/c12345678.lua"
        # Try the path as-is first, then relative to script_dir
        paths_to_try = [
            script_name,
            os.path.join(self.script_dir, os.path.basename(script_name)),
        ]

        for path in paths_to_try:
            if os.path.exists(path):
                try:
                    data = open(path, "rb").read()
                    ctypes.memmove(self._script_buf, data, len(data))
                    len_ptr[0] = len(data)
                    return ctypes.addressof(self._script_buf)
                except Exception:
                    break

        # Script not found
        len_ptr[0] = 0
        return None

    def _message_handler(self, pduel, msg_type: int) -> int:
        """
        Called by the engine for certain message types.
        We don't need to handle anything here for basic operation.
        """
        return 0

    def cleanup(self):
        """Close database connection."""
        self.db.close()
