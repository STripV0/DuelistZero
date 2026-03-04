"""
EDOPro bot client — connects to an EDOPro-hosted LAN game
and plays using a trained MaskablePPO model.
"""

import random
import socket
import struct
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.message_parser import (
    MessageParser,
    ParsedMessage,
    MsgStart,
    RESPONSE_MESSAGES,
)
from ..engine.duel import load_deck
from ..engine.game_state import GameState, update_state
from ..env.action_space import ActionSpace
from ..env.observation import (
    encode_observation, encode_card_ids, encode_action_features, encode_action_history,
    CardDB,
)
from ..env.card_index import CardIndex
from .protocol import (
    send_packet, recv_packet,
    CTOS_PLAYER_INFO, CTOS_JOIN_GAME, CTOS_UPDATE_DECK,
    CTOS_HAND_RESULT, CTOS_TP_RESULT, CTOS_RESPONSE,
    CTOS_HS_READY, CTOS_TIME_CONFIRM,
    STOC_GAME_MSG, STOC_ERROR_MSG, STOC_SELECT_HAND, STOC_SELECT_TP,
    STOC_HAND_RESULT, STOC_TP_RESULT, STOC_JOIN_GAME, STOC_TYPE_CHANGE,
    STOC_DUEL_START, STOC_DUEL_END, STOC_TIME_LIMIT,
    STOC_HS_PLAYER_ENTER, STOC_HS_PLAYER_CHANGE, STOC_HS_WATCH_CHANGE,
    STOC_CHANGE_SIDE, STOC_REPLAY, STOC_CHAT,
    build_player_info, build_join_game, build_update_deck,
    build_hand_result, build_tp_result,
    RPS_ROCK,
)

# Default paths
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DB = _PROJECT_ROOT / "data" / "cards.cdb"
_DEFAULT_DECK = _PROJECT_ROOT / "data" / "deck" / "Goat Control.ydk"
_DEFAULT_MODEL = _PROJECT_ROOT / "checkpoints" / "final_model"


class DuelProxy:
    """Mock Duel that captures responses from ActionSpace.decode()."""

    def __init__(self):
        self.response: Optional[bytes] = None

    def respond_int(self, value: int):
        self.response = struct.pack("<i", value)

    def respond_bytes(self, data: bytes):
        self.response = data

    def respond_card_selection(self, indices: list[int]):
        # EDOPro's ygopro-core fork reads responses as int32 arrays:
        #   returns[0] = type (3 = legacy byte format)
        #   returns[1] = count
        #   returns[2..] = indices
        # Use type=0 (uint32 encoding) for compatibility
        self.response = struct.pack("<i", 0)  # type = 0 (uint32)
        self.response += struct.pack("<I", len(indices))  # count
        for idx in indices:
            self.response += struct.pack("<I", idx)  # each index as uint32


class EdoProBot:
    """Bot client that connects to EDOPro and plays using a trained model."""

    def __init__(
        self,
        model_path: str | Path = _DEFAULT_MODEL,
        deck_path: str | Path = _DEFAULT_DECK,
        db_path: str | Path = _DEFAULT_DB,
        host: str = "127.0.0.1",
        port: int = 7911,
        name: str = "DuelistZero",
        version: int = 0x1360,
    ):
        self.host = host
        self.port = port
        self.name = name
        self.version = version

        # Load model (try MaskableRecurrentPPO first, fall back to MaskablePPO)
        print(f"Loading model from {model_path}...")
        try:
            from ..training.maskable_recurrent_ppo import MaskableRecurrentPPO
            self.model = MaskableRecurrentPPO.load(str(model_path))
            self._recurrent = True
            print("Loaded as MaskableRecurrentPPO (LSTM)")
        except Exception:
            from sb3_contrib import MaskablePPO
            self.model = MaskablePPO.load(str(model_path))
            self._recurrent = False
            print("Loaded as MaskablePPO (non-recurrent)")

        # LSTM state tracking for recurrent models
        self._lstm_state = None
        self._episode_start = np.array([True])

        # Detect whether model uses dict or flat observations
        from gymnasium import spaces
        self._dict_obs = isinstance(self.model.observation_space, spaces.Dict)
        print(f"Model observation type: {'dict' if self._dict_obs else 'flat'}")

        # Load deck
        self.main_deck, self.extra_deck = load_deck(deck_path)
        print(f"Deck loaded: {len(self.main_deck)} main + {len(self.extra_deck)} extra")

        # Card DB and index for observation encoding
        self.card_db = CardDB(db_path)
        self.card_index = CardIndex(db_path)

        # Game state
        self.state = GameState()
        self.parser = MessageParser(edopro=True)
        self.action_space = ActionSpace()
        self.proxy = DuelProxy()

        # Which player we are (set during handshake)
        self.my_player = 0
        self.sock: Optional[socket.socket] = None
        self._running = False

    def connect(self):
        """Connect to EDOPro and send handshake."""
        print(f"Connecting to {self.host}:{self.port}...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(120.0)
        self.sock.connect((self.host, self.port))
        print("Connected!")

        # Send player info
        send_packet(self.sock, CTOS_PLAYER_INFO, build_player_info(self.name))

        # Join game
        send_packet(self.sock, CTOS_JOIN_GAME, build_join_game(self.version))
        print(f"Sent player info and join request (version=0x{self.version:04x})")

    def run(self):
        """Main event loop — receive and handle packets."""
        self._running = True
        print("Waiting for server response...")
        try:
            while self._running:
                proto_id, payload = recv_packet(self.sock)
                print(f"  << STOC 0x{proto_id:02x} ({len(payload)} bytes)")
                self._dispatch(proto_id, payload)
        except socket.timeout:
            print("Timed out waiting for server response.")
            print("This usually means a protocol version mismatch.")
            print("Check your EDOPro version (title bar) and pass it with --version.")
        except ConnectionError as e:
            print(f"Connection closed: {e}")
        except KeyboardInterrupt:
            print("\nBot stopped by user")
        finally:
            self.close()

    def close(self):
        """Clean up resources."""
        self._running = False
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        if self.card_db:
            self.card_db.close()
            self.card_db = None

    def _dispatch(self, proto_id: int, payload: bytes):
        """Dispatch a received STOC packet to the appropriate handler."""
        if proto_id == STOC_GAME_MSG:
            self._handle_game_msg(payload)
        elif proto_id == STOC_SELECT_HAND:
            self._handle_select_hand()
        elif proto_id == STOC_SELECT_TP:
            self._handle_select_tp()
        elif proto_id == STOC_TYPE_CHANGE:
            self._handle_type_change(payload)
        elif proto_id == STOC_JOIN_GAME:
            self._handle_join_game(payload)
        elif proto_id == STOC_DUEL_START:
            self._handle_duel_start()
        elif proto_id == STOC_DUEL_END:
            self._handle_duel_end()
        elif proto_id == STOC_TIME_LIMIT:
            self._handle_time_limit(payload)
        elif proto_id == STOC_HAND_RESULT:
            self._handle_hand_result(payload)
        elif proto_id == STOC_TP_RESULT:
            self._handle_tp_result(payload)
        elif proto_id == STOC_REPLAY:
            pass  # Ignore replay data
        elif proto_id == STOC_CHAT:
            pass  # Ignore chat
        elif proto_id == STOC_ERROR_MSG:
            self._handle_error(payload)
        elif proto_id == STOC_HS_PLAYER_ENTER:
            self._handle_player_enter(payload)
        elif proto_id == STOC_HS_PLAYER_CHANGE:
            pass
        elif proto_id == STOC_HS_WATCH_CHANGE:
            pass
        elif proto_id == STOC_CHANGE_SIDE:
            # Side deck change phase - just re-send our deck
            send_packet(self.sock, CTOS_UPDATE_DECK,
                        build_update_deck(self.main_deck, self.extra_deck))
        else:
            print(f"  Unknown STOC: 0x{proto_id:02x} ({len(payload)} bytes)")

    # ============================================================
    # STOC Handlers
    # ============================================================

    def _handle_game_msg(self, payload: bytes):
        """Handle STOC_GAME_MSG containing ygopro-core messages."""
        if not payload:
            return

        try:
            messages = self.parser.parse(payload)
        except Exception as e:
            print(f"  Parse error (msg_id={payload[0] if payload else '?'}): {e}")
            return

        for msg in messages:
            print(f"    MSG: {msg.msg_type.name} player={getattr(msg, 'player', '?')}")

            # Track our game-engine player ID from MSG_START
            if isinstance(msg, MsgStart):
                self.my_player = msg.player_type
                print(f"    Game-engine player ID: {self.my_player}")

            # Update game state
            try:
                update_state(self.state, msg)
            except Exception:
                pass  # state update failures are non-fatal

            # If this is a decision message, respond — EDOPro only
            # sends us decisions that are ours to answer
            if msg.msg_type in RESPONSE_MESSAGES:
                if hasattr(msg, 'cards'):
                    print(f"    >> cards={len(msg.cards)} min={getattr(msg, 'min_count', '?')} "
                          f"max={getattr(msg, 'max_count', '?')} cancel={getattr(msg, 'cancelable', '?')}")
                self._respond_to_decision(msg)

    def _respond_to_decision(self, msg: ParsedMessage):
        """Use the model to decide and send a response."""
        try:
            # Compute observation from our perspective
            if self._dict_obs:
                obs = {
                    "features": encode_observation(
                        self.state, perspective=self.my_player, db=self.card_db
                    ),
                    "card_ids": encode_card_ids(
                        self.state, perspective=self.my_player, card_index=self.card_index
                    ),
                    "action_features": encode_action_features(
                        msg, self.card_index, db=self.card_db
                    ),
                    "action_history": encode_action_history(
                        self.state, perspective=self.my_player,
                        card_index=self.card_index, db=self.card_db
                    ),
                }
            else:
                obs = encode_observation(
                    self.state, perspective=self.my_player, db=self.card_db
                )

            # Get valid action mask
            mask = self.action_space.get_mask(msg)

            # Run model (with LSTM state tracking for recurrent models)
            if self._recurrent:
                action, self._lstm_state = self.model.predict(
                    obs,
                    state=self._lstm_state,
                    episode_start=self._episode_start,
                    action_masks=mask,
                    deterministic=True,
                )
                self._episode_start = np.array([False])
            else:
                action, _ = self.model.predict(
                    obs, action_masks=mask, deterministic=True
                )
            action = int(action)

            # Clamp to valid actions
            if not mask[action]:
                valid = np.where(mask)[0]
                action = int(np.random.choice(valid)) if len(valid) > 0 else 0

            # Decode action to get response bytes
            self.proxy.response = None
            self.action_space.decode(action, msg, self.proxy)

            # Send response
            if self.proxy.response is not None:
                send_packet(self.sock, CTOS_RESPONSE, self.proxy.response)
                print(f"    -> Responded to {msg.msg_type.name}: action={action} "
                      f"response={self.proxy.response.hex()}")
            else:
                print(f"    !! No response generated for {msg.msg_type.name}")
        except Exception as e:
            print(f"    !! Decision error ({msg.msg_type.name}): {e}")
            # Send a fallback response (action 0)
            try:
                mask = self.action_space.get_mask(msg)
                valid = np.where(mask)[0]
                fallback = int(valid[0]) if len(valid) > 0 else 0
                self.proxy.response = None
                self.action_space.decode(fallback, msg, self.proxy)
                if self.proxy.response is not None:
                    send_packet(self.sock, CTOS_RESPONSE, self.proxy.response)
                    print(f"    -> Fallback response: action={fallback}")
            except Exception as e2:
                print(f"    !! Fallback also failed: {e2}")

    def _handle_select_hand(self):
        """Rock-paper-scissors — pick randomly."""
        choice = random.choice([RPS_ROCK, RPS_ROCK, RPS_ROCK])  # always rock for simplicity
        send_packet(self.sock, CTOS_HAND_RESULT, build_hand_result(choice))
        print("RPS: Rock")

    def _handle_select_tp(self):
        """Choose to go second (draws on first turn, more active play)."""
        send_packet(self.sock, CTOS_TP_RESULT, build_tp_result(go_first=False))
        print("Chose to go second")

    def _handle_type_change(self, payload: bytes):
        """Track which player slot we are."""
        if payload:
            type_val = payload[0]
            # Bits 0-3: player position (0=host P1, 1=client P1, etc.)
            # Bit 4: is_host flag
            self.my_player = type_val & 0x0F
            print(f"Assigned as player {self.my_player}")

    def _handle_join_game(self, payload: bytes):
        """Received join confirmation — send deck, then ready."""
        print("Joined game lobby")
        # Must send deck before readying up
        send_packet(self.sock, CTOS_UPDATE_DECK,
                    build_update_deck(self.main_deck, self.extra_deck))
        print(f"Sent deck: {len(self.main_deck)} main + {len(self.extra_deck)} extra")
        send_packet(self.sock, CTOS_HS_READY, b"")

    def _handle_duel_start(self):
        """Duel is starting — send our deck."""
        print("Duel starting! Sending deck...")
        send_packet(self.sock, CTOS_UPDATE_DECK,
                    build_update_deck(self.main_deck, self.extra_deck))
        # Reset game state and LSTM state for the new duel
        self.state.reset()
        self._lstm_state = None
        self._episode_start = np.array([True])

    def _handle_duel_end(self):
        """Duel ended."""
        winner = self.state.winner
        if winner == self.my_player:
            print("Result: Bot WINS!")
        elif winner >= 0:
            print("Result: Bot LOSES")
        else:
            print("Result: DRAW")
        print(f"Final LP — Bot: {self.state.players[self.my_player].lp} | "
              f"Opponent: {self.state.players[1 - self.my_player].lp}")
        self._running = False

    def _handle_time_limit(self, payload: bytes):
        """Respond to time limit check to avoid timeout."""
        send_packet(self.sock, CTOS_TIME_CONFIRM, b"")

    def _handle_hand_result(self, payload: bytes):
        """RPS result received."""
        if len(payload) >= 2:
            res1, res2 = payload[0], payload[1]
            names = {1: "Scissors", 2: "Rock", 3: "Paper"}
            print(f"RPS result: Us={names.get(res1, '?')} vs Them={names.get(res2, '?')}")

    def _handle_tp_result(self, payload: bytes):
        """Turn player result."""
        if payload:
            print(f"Player {payload[0]} goes first")

    def _handle_error(self, payload: bytes):
        """Handle server error."""
        if len(payload) < 1:
            self._running = False
            return
        error_type = payload[0]
        error_names = {1: "Join error", 2: "Deck error", 3: "Side error",
                       4: "Version error", 5: "Version error 2"}
        print(f"Server error: {error_names.get(error_type, f'code {error_type}')}")

        if error_type == 2 and len(payload) >= 24:
            # DeckError struct (with natural alignment):
            #   u8 etype, 3 pad, u32 type, u32 current, u32 min, u32 max, u32 code
            derr_type, current, minimum, maximum, code = struct.unpack_from("<5I", payload, 4)
            derr_names = {
                0: "NONE", 1: "LFLIST", 2: "OCGONLY", 3: "TCGONLY",
                4: "UNKNOWNCARD", 5: "CARDCOUNT", 6: "MAINCOUNT",
                7: "EXTRACOUNT", 8: "SIDECOUNT", 9: "FORBTYPE",
                10: "UNOFFICIALCARD", 11: "INVALIDSIZE",
            }
            print(f"  Deck error type: {derr_names.get(derr_type, derr_type)}")
            print(f"  Count: current={current} min={minimum} max={maximum}")
            print(f"  Card code: {code}")
        elif error_type in (4, 5) and len(payload) >= 8:
            # VersionError
            print(f"  Raw: {payload.hex()}")
        else:
            print(f"  Raw: {payload.hex()}")
        self._running = False

    def _handle_player_enter(self, payload: bytes):
        """A player entered the lobby."""
        if len(payload) >= 40:
            name_bytes = payload[:40]
            # Decode UTF-16LE, strip null terminators
            name = name_bytes.decode("utf-16-le", errors="replace").rstrip("\x00")
            print(f"Player entered: {name}")
