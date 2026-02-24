"""
Python ctypes bindings for libocgcore.so (ygopro-core).

Provides a thin wrapper around the C API defined in ocgapi.h.
Reference: https://github.com/Fluorohydride/ygopro-core/blob/master/ocgapi.h
"""

import ctypes
import os
from ctypes import (
    CDLL,
    CFUNCTYPE,
    POINTER,
    Structure,
    c_byte,
    c_char_p,
    c_int32,
    c_uint8,
    c_uint16,
    c_uint32,
    c_void_p,
)
from pathlib import Path

# Type aliases
byte_p = POINTER(c_byte)
intptr_t = c_void_p  # duel handle


# ============================================================
# card_data struct (mirrors C struct)
# ============================================================
class CardData(Structure):
    """Mirrors the card_data struct in ygopro-core."""
    _fields_ = [
        ("code", c_uint32),
        ("alias", c_uint32),
        ("setcode", c_uint16 * 16),  # SIZE_SETCODE = 16 in card_data.h
        ("type", c_uint32),
        ("level", c_uint32),
        ("attribute", c_uint32),
        ("race", c_uint32),
        ("attack", c_int32),
        ("defense", c_int32),
        ("lscale", c_uint32),
        ("rscale", c_uint32),
        ("link_marker", c_uint32),
    ]


# ============================================================
# Callback function types
# ============================================================
# typedef byte* (*script_reader)(const char* script_name, int* len);
SCRIPT_READER_FUNC = CFUNCTYPE(c_void_p, c_char_p, POINTER(c_int32))

# typedef uint32_t (*card_reader)(uint32_t code, card_data* data);
CARD_READER_FUNC = CFUNCTYPE(c_uint32, c_uint32, POINTER(CardData))

# typedef uint32_t (*message_handler)(intptr_t pduel, uint32_t msg_type);
MESSAGE_HANDLER_FUNC = CFUNCTYPE(c_uint32, intptr_t, c_uint32)


# ============================================================
# Library Loader
# ============================================================
class OcgCore:
    """
    Wrapper around the ygopro-core shared library.

    Usage:
        core = OcgCore("/path/to/libocgcore.so")
        duel = core.create_duel(seed=42)
        core.set_player_info(duel, player=0, lp=8000, start_hand=5, draw_count=1)
        ...
    """

    def __init__(self, lib_path: str | Path | None = None):
        if lib_path is None:
            # Default: look in project lib/ directory
            lib_path = Path(__file__).parent.parent.parent.parent / "lib" / "libocgcore.so"

        lib_path = str(lib_path)
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"libocgcore.so not found at {lib_path}. "
                "Run ./build_core.sh to compile it."
            )

        self.lib = CDLL(lib_path)
        self._setup_functions()
        self._callbacks = []  # prevent GC of callback references

    def _setup_functions(self):
        """Define argument/return types for all C functions."""
        lib = self.lib

        # void set_script_reader(script_reader f)
        lib.set_script_reader.argtypes = [SCRIPT_READER_FUNC]
        lib.set_script_reader.restype = None

        # void set_card_reader(card_reader f)
        lib.set_card_reader.argtypes = [CARD_READER_FUNC]
        lib.set_card_reader.restype = None

        # void set_message_handler(message_handler f)
        lib.set_message_handler.argtypes = [MESSAGE_HANDLER_FUNC]
        lib.set_message_handler.restype = None

        # intptr_t create_duel(uint_fast32_t seed)
        lib.create_duel.argtypes = [c_uint32]
        lib.create_duel.restype = intptr_t

        # void start_duel(intptr_t pduel, uint32_t options)
        lib.start_duel.argtypes = [intptr_t, c_uint32]
        lib.start_duel.restype = None

        # void end_duel(intptr_t pduel)
        lib.end_duel.argtypes = [intptr_t]
        lib.end_duel.restype = None

        # void set_player_info(intptr_t pduel, int32_t playerid, int32_t lp,
        #                      int32_t startcount, int32_t drawcount)
        lib.set_player_info.argtypes = [intptr_t, c_int32, c_int32, c_int32, c_int32]
        lib.set_player_info.restype = None

        # void get_log_message(intptr_t pduel, char* buf)
        lib.get_log_message.argtypes = [intptr_t, c_char_p]
        lib.get_log_message.restype = None

        # int32_t get_message(intptr_t pduel, byte* buf)
        lib.get_message.argtypes = [intptr_t, byte_p]
        lib.get_message.restype = c_int32

        # uint32_t process(intptr_t pduel)
        lib.process.argtypes = [intptr_t]
        lib.process.restype = c_uint32

        # void new_card(intptr_t pduel, uint32_t code, uint8_t owner,
        #               uint8_t playerid, uint8_t location, uint8_t sequence,
        #               uint8_t position)
        lib.new_card.argtypes = [
            intptr_t, c_uint32, c_uint8, c_uint8, c_uint8, c_uint8, c_uint8
        ]
        lib.new_card.restype = None

        # void new_tag_card(intptr_t pduel, uint32_t code, uint8_t owner,
        #                   uint8_t location)
        lib.new_tag_card.argtypes = [intptr_t, c_uint32, c_uint8, c_uint8]
        lib.new_tag_card.restype = None

        # int32_t query_card(intptr_t pduel, uint8_t playerid, uint8_t location,
        #                    uint8_t sequence, uint32_t query_flag, byte* buf,
        #                    int32_t use_cache)
        lib.query_card.argtypes = [
            intptr_t, c_uint8, c_uint8, c_uint8, c_uint32, byte_p, c_int32
        ]
        lib.query_card.restype = c_int32

        # int32_t query_field_count(intptr_t pduel, uint8_t playerid,
        #                           uint8_t location)
        lib.query_field_count.argtypes = [intptr_t, c_uint8, c_uint8]
        lib.query_field_count.restype = c_int32

        # int32_t query_field_card(intptr_t pduel, uint8_t playerid,
        #                          uint8_t location, uint32_t query_flag,
        #                          byte* buf, int32_t use_cache)
        lib.query_field_card.argtypes = [
            intptr_t, c_uint8, c_uint8, c_uint32, byte_p, c_int32
        ]
        lib.query_field_card.restype = c_int32

        # int32_t query_field_info(intptr_t pduel, byte* buf)
        lib.query_field_info.argtypes = [intptr_t, byte_p]
        lib.query_field_info.restype = c_int32

        # void set_responsei(intptr_t pduel, int32_t value)
        lib.set_responsei.argtypes = [intptr_t, c_int32]
        lib.set_responsei.restype = None

        # void set_responseb(intptr_t pduel, byte* buf)
        lib.set_responseb.argtypes = [intptr_t, byte_p]
        lib.set_responseb.restype = None

        # int32_t preload_script(intptr_t pduel, const char* script_name)
        lib.preload_script.argtypes = [intptr_t, c_char_p]
        lib.preload_script.restype = c_int32

    # ============================================================
    # Callback registration
    # ============================================================
    def set_card_reader(self, func):
        """Register a card reader callback. func(code: int, data: CardData) -> int"""
        cb = CARD_READER_FUNC(func)
        self._callbacks.append(cb)  # prevent GC
        self.lib.set_card_reader(cb)

    def set_script_reader(self, func):
        """Register a script reader callback. func(name: bytes, len_ptr) -> byte*"""
        cb = SCRIPT_READER_FUNC(func)
        self._callbacks.append(cb)
        self.lib.set_script_reader(cb)

    def set_message_handler(self, func):
        """Register a message handler callback. func(pduel, msg_type) -> int"""
        cb = MESSAGE_HANDLER_FUNC(func)
        self._callbacks.append(cb)
        self.lib.set_message_handler(cb)

    # ============================================================
    # Duel lifecycle
    # ============================================================
    def create_duel(self, seed: int) -> intptr_t:
        return self.lib.create_duel(c_uint32(seed))

    def start_duel(self, pduel: intptr_t, options: int):
        self.lib.start_duel(pduel, c_uint32(options))

    def end_duel(self, pduel: intptr_t):
        self.lib.end_duel(pduel)

    def set_player_info(self, pduel: intptr_t, player: int,
                        lp: int, start_hand: int, draw_count: int):
        self.lib.set_player_info(pduel, player, lp, start_hand, draw_count)

    # ============================================================
    # Duel processing
    # ============================================================
    def process(self, pduel: intptr_t) -> int:
        return self.lib.process(pduel)

    def get_message(self, pduel: intptr_t, buf_size: int = 4096) -> bytes:
        """Get message buffer from engine. Returns raw bytes."""
        buf = (c_byte * buf_size)()
        length = self.lib.get_message(pduel, buf)
        if length <= 0:
            return b""
        # Convert signed c_byte to unsigned bytes
        return bytes(b & 0xFF for b in buf[:length])

    def get_log_message(self, pduel: intptr_t) -> str:
        buf = ctypes.create_string_buffer(1024)
        self.lib.get_log_message(pduel, buf)
        return buf.value.decode("utf-8", errors="replace")

    # ============================================================
    # Card management
    # ============================================================
    def new_card(self, pduel: intptr_t, code: int, owner: int,
                 playerid: int, location: int, sequence: int, position: int):
        self.lib.new_card(pduel, code, owner, playerid, location, sequence, position)

    # ============================================================
    # Responses
    # ============================================================
    def set_responsei(self, pduel: intptr_t, value: int):
        self.lib.set_responsei(pduel, c_int32(value))

    def set_responseb(self, pduel: intptr_t, data: bytes):
        buf = (c_byte * len(data))(*data)
        self.lib.set_responseb(pduel, buf)

    # ============================================================
    # Card querying
    # ============================================================
    def query_card(self, pduel: intptr_t, player: int, location: int,
                   sequence: int, flags: int, use_cache: bool = False) -> bytes:
        buf = (c_byte * 4096)()
        length = self.lib.query_card(
            pduel, player, location, sequence, flags, buf, int(use_cache)
        )
        if length <= 0:
            return b""
        return bytes(b & 0xFF for b in buf[:length])

    def query_field_count(self, pduel: intptr_t, player: int, location: int) -> int:
        return self.lib.query_field_count(pduel, player, location)

    def query_field_card(self, pduel: intptr_t, player: int, location: int,
                         flags: int, use_cache: bool = False) -> bytes:
        buf = (c_byte * 8192)()
        length = self.lib.query_field_card(
            pduel, player, location, flags, buf, int(use_cache)
        )
        if length <= 0:
            return b""
        return bytes(b & 0xFF for b in buf[:length])
