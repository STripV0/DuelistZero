"""
Duel controller — manages the lifecycle of a single duel.

Creates duel instance, loads decks, processes engine messages,
updates game state, and dispatches player decisions.
"""

import random
import struct
from pathlib import Path
from typing import Callable, Optional

from ..core.bindings import OcgCore
from ..core.callbacks import CallbackManager
from ..core.constants import (
    GOAT_DUEL_OPTIONS,
    LOCATION,
    MSG,
    PHASE,
    POSITION,
    PROCESSOR_END,
    PROCESSOR_WAITING,
)
from ..core.message_parser import (
    MessageParser,
    ParsedMessage,
    RESPONSE_MESSAGES,
)
from .game_state import GameState, ZoneCard, update_state


# Type for decision callback
DecisionFunc = Callable[[ParsedMessage, GameState], int | bytes]


def load_deck(filepath: str | Path) -> tuple[list[int], list[int]]:
    """
    Load a .ydk deck file.
    Returns (main_deck, extra_deck) as lists of card codes.
    """
    main = []
    extra = []
    side = []
    current = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line.startswith("!"):
                if "main" in line.lower():
                    current = main
                elif "extra" in line.lower():
                    current = extra
                elif "side" in line.lower():
                    current = side
                continue
            if not line:
                continue
            # Strip inline comments: "71413901  # Card Name" -> "71413901"
            if "#" in line:
                line = line[:line.index("#")].strip()
            if not line:
                continue
            try:
                code = int(line)
                if current is not None:
                    current.append(code)
                else:
                    main.append(code)  # default to main
            except ValueError:
                continue

    return main, extra


class Duel:
    """
    Controls a single duel between two players.

    Usage:
        duel = Duel(core, cb_manager)
        duel.load_decks("deck1.ydk", "deck2.ydk")
        duel.start()

        while not duel.state.is_finished:
            msg = duel.process_turn()
            if msg and msg.msg_type in RESPONSE_MESSAGES:
                response = agent.decide(msg, duel.state)
                duel.respond(response)
    """

    def __init__(self, core: OcgCore, cb_manager: CallbackManager,
                 lp: int = 8000, seed: int | None = None):
        self.core = core
        self.cb_manager = cb_manager
        self.parser = MessageParser()
        self.state = GameState()
        self.lp = lp

        if seed is None:
            seed = random.randint(0, 0xFFFFFFFF)
        self.seed = seed

        self._pduel = None
        self._pending_decision: Optional[ParsedMessage] = None

    def load_decks(self, deck1_path: str | Path, deck2_path: str | Path):
        """Load .ydk decks for both players."""
        self._deck1 = load_deck(deck1_path)
        self._deck2 = load_deck(deck2_path)

    def start(self, options: int | None = None):
        """Initialize and start the duel."""
        if options is None:
            options = GOAT_DUEL_OPTIONS

        # Create duel instance
        self._pduel = self.core.create_duel(self.seed)

        # Set player info
        self.core.set_player_info(self._pduel, 0, self.lp, 5, 1)
        self.core.set_player_info(self._pduel, 1, self.lp, 5, 1)

        self.state.players[0].lp = self.lp
        self.state.players[1].lp = self.lp

        # Load decks into engine
        self._load_deck_to_engine(0, self._deck1)
        self._load_deck_to_engine(1, self._deck2)

        # Start the duel
        self.core.start_duel(self._pduel, options)

    def _load_deck_to_engine(self, player: int, deck: tuple[list[int], list[int]]):
        """Load a deck (main + extra) into the engine for a player."""
        main, extra = deck

        # Shuffle main deck
        shuffled_main = main[:]
        random.shuffle(shuffled_main)

        # Load main deck cards (in reverse so top of deck is first card)
        for code in reversed(shuffled_main):
            self.core.new_card(
                self._pduel, code, player, player,
                LOCATION.DECK, 0, POSITION.FACEDOWN_DEFENSE
            )

        # Load extra deck
        for code in extra:
            self.core.new_card(
                self._pduel, code, player, player,
                LOCATION.EXTRA, 0, POSITION.FACEDOWN_DEFENSE
            )

        self.state.players[player].deck_count = len(main)
        self.state.players[player].extra_count = len(extra)

    def process(self) -> Optional[ParsedMessage]:
        """
        Process engine until it needs a player decision or the duel ends.
        Returns the decision-requesting message, or None if duel ended.
        """
        while True:
            result = self.core.process(self._pduel)
            msg_data = self.core.get_message(self._pduel)

            if not msg_data:
                if result & PROCESSOR_END:
                    self.state.is_finished = True
                    return None
                continue

            messages = self.parser.parse(msg_data)

            for msg in messages:
                if msg.msg_type == MSG.RETRY:
                    self.state.is_finished = True
                    return None

                self._update_state(msg)

                if msg.msg_type in RESPONSE_MESSAGES:
                    self._pending_decision = msg
                    return msg

            if result & PROCESSOR_END:
                self.state.is_finished = True
                return None
            if result & PROCESSOR_WAITING:
                continue

    def respond_int(self, value: int):
        """Send an integer response to the engine."""
        self.core.set_responsei(self._pduel, value)
        self._pending_decision = None

    def respond_bytes(self, data: bytes):
        """Send a byte buffer response to the engine."""
        self.core.set_responseb(self._pduel, data)
        self._pending_decision = None

    def respond_card_selection(self, indices: list[int]):
        """
        Respond to MSG_SELECT_CARD / MSG_SELECT_TRIBUTE with selected indices.
        Encodes as: count(1 byte) + indices(1 byte each)
        """
        response = struct.pack("B", len(indices))
        for idx in indices:
            response += struct.pack("B", idx)
        self.respond_bytes(response)

    def end(self):
        """End the duel and clean up."""
        if self._pduel is not None:
            self.core.end_duel(self._pduel)
            self._pduel = None

    # ============================================================
    # State Update from Messages
    # ============================================================
    def _update_state(self, msg: ParsedMessage):
        """Update game state based on a parsed message."""
        update_state(self.state, msg)
