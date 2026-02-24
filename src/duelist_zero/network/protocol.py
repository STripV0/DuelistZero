"""
YGOPro network protocol primitives.

Packet format: [uint16_le length][uint8 proto_id][payload...]
where length = 1 + len(payload).
"""

import struct
import socket


# ============================================================
# CTOS (Client-To-Server) message types
# ============================================================
CTOS_RESPONSE = 0x01
CTOS_UPDATE_DECK = 0x02
CTOS_HAND_RESULT = 0x03
CTOS_TP_RESULT = 0x04
CTOS_PLAYER_INFO = 0x10
CTOS_CREATE_GAME = 0x11
CTOS_JOIN_GAME = 0x12
CTOS_LEAVE_GAME = 0x13
CTOS_SURRENDER = 0x14
CTOS_TIME_CONFIRM = 0x15
CTOS_CHAT = 0x16
CTOS_HS_TODUELIST = 0x20
CTOS_HS_TOOBSERVER = 0x21
CTOS_HS_READY = 0x22
CTOS_HS_NOTREADY = 0x23
CTOS_HS_KICK = 0x24
CTOS_HS_START = 0x25

# ============================================================
# STOC (Server-To-Client) message types
# ============================================================
STOC_GAME_MSG = 0x01
STOC_ERROR_MSG = 0x02
STOC_SELECT_HAND = 0x03
STOC_SELECT_TP = 0x04
STOC_HAND_RESULT = 0x05
STOC_TP_RESULT = 0x06
STOC_CHANGE_SIDE = 0x07
STOC_WAITING_SIDE = 0x08
STOC_CREATE_GAME = 0x11
STOC_JOIN_GAME = 0x12
STOC_TYPE_CHANGE = 0x13
STOC_LEAVE_GAME = 0x14
STOC_DUEL_START = 0x15
STOC_DUEL_END = 0x16
STOC_REPLAY = 0x17
STOC_TIME_LIMIT = 0x18
STOC_CHAT = 0x19
STOC_HS_PLAYER_ENTER = 0x20
STOC_HS_PLAYER_CHANGE = 0x21
STOC_HS_WATCH_CHANGE = 0x22

# RPS values
RPS_SCISSORS = 1
RPS_ROCK = 2
RPS_PAPER = 3


# ============================================================
# Packet I/O
# ============================================================
def send_packet(sock: socket.socket, proto_id: int, payload: bytes = b"") -> None:
    """Send a framed packet: [uint16_le length][uint8 proto_id][payload]."""
    length = 1 + len(payload)
    header = struct.pack("<HB", length, proto_id)
    sock.sendall(header + payload)


def recv_packet(sock: socket.socket) -> tuple[int, bytes]:
    """Receive a framed packet. Returns (proto_id, payload).

    Raises ConnectionError if the connection is closed.
    """
    header = _recv_exact(sock, 2)
    length = struct.unpack("<H", header)[0]
    if length < 1:
        raise ConnectionError("Invalid packet length")
    data = _recv_exact(sock, length)
    proto_id = data[0]
    payload = data[1:]
    return proto_id, payload


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from socket."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed")
        buf.extend(chunk)
    return bytes(buf)


# ============================================================
# Struct builders
# ============================================================
def build_player_info(name: str = "DuelistZero") -> bytes:
    """Build CTOS_PLAYER_INFO payload: name as UTF-16LE in 20-char buffer."""
    encoded = name.encode("utf-16-le")
    # 20 uint16 chars = 40 bytes, zero-padded
    buf = encoded[:40].ljust(40, b"\x00")
    return buf


def build_join_game(version: int = 0x1360, gameid: int = 0,
                    password: str = "",
                    client_major: int = 41, client_minor: int = 0,
                    core_major: int = 11, core_minor: int = 0) -> bytes:
    """Build CTOS_JOIN_GAME payload.

    EDOPro expects a version2 (ClientVersion) field after the password:
        struct ClientVersion { {u8 major, minor} client, core; }
    Packet is rejected (silently dropped) if this field is missing.
    """
    pass_encoded = password.encode("utf-16-le")
    pass_buf = pass_encoded[:40].ljust(40, b"\x00")
    version2 = struct.pack("<BBBB", client_major, client_minor,
                           core_major, core_minor)
    # HHI = version(u16) + padding(u16) + gameid(u32) — matches C struct alignment
    return struct.pack("<HHI", version, 0, gameid) + pass_buf + version2


def build_update_deck(main: list[int], extra: list[int],
                      side: list[int] | None = None) -> bytes:
    """Build CTOS_UPDATE_DECK payload.

    Format: mainc(i32) + sidec(i32) + card_codes(u32 each)
    Cards are sent as: main + extra + side (concatenated).
    """
    if side is None:
        side = []
    # mainc includes extra deck cards; sidec is side deck
    mainc = len(main) + len(extra)
    sidec = len(side)
    buf = struct.pack("<ii", mainc, sidec)
    for code in main:
        buf += struct.pack("<I", code)
    for code in extra:
        buf += struct.pack("<I", code)
    for code in side:
        buf += struct.pack("<I", code)
    return buf


def build_hand_result(choice: int = RPS_ROCK) -> bytes:
    """Build CTOS_HAND_RESULT payload (RPS choice: 1=scissors, 2=rock, 3=paper)."""
    return struct.pack("<B", choice)


def build_tp_result(go_first: bool = True) -> bytes:
    """Build CTOS_TP_RESULT payload (0=go first, 1=go second)."""
    return struct.pack("<B", 0 if go_first else 1)
