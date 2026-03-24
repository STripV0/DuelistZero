"""
Game state representation.

Maintains a Pythonic view of the current duel state,
updated by processing engine messages.
"""

from dataclasses import dataclass, field
from typing import Optional

from ..core.constants import LOCATION, MSG, PHASE
from ..core.message_parser import (
    ParsedMessage,
    MsgStart,
    MsgNewTurn,
    MsgNewPhase,
    MsgDraw,
    MsgDamage,
    MsgRecover,
    MsgLPUpdate,
    MsgPayLPCost,
    MsgMove,
    MsgSummoning,
    MsgSet,
    MsgChaining,
    MsgAttack,
    MsgPosChange,
    MsgShuffleHand,
    MsgWin,
)


@dataclass
class ZoneCard:
    """A card in a specific zone on the field."""
    code: int
    position: int = 0  # POSITION flags
    controller: int = 0
    sequence: int = 0
    set_turn: int = 0  # Turn number when this card was placed (for age tracking)


@dataclass
class PlayerState:
    """State for one player."""
    lp: int = 8000
    hand: list[int] = field(default_factory=list)          # card codes
    monsters: list[Optional[ZoneCard]] = field(default_factory=lambda: [None] * 5)
    spells: list[Optional[ZoneCard]] = field(default_factory=lambda: [None] * 5)
    graveyard: list[int] = field(default_factory=list)     # card codes
    banished: list[int] = field(default_factory=list)      # card codes
    deck_count: int = 40
    extra_count: int = 0
    extra_draws: int = 0  # Draws beyond normal draw-phase draw (for opponent inference)

    @property
    def hand_count(self) -> int:
        return len(self.hand)

    @property
    def monster_count(self) -> int:
        return sum(1 for m in self.monsters if m is not None)

    @property
    def spell_count(self) -> int:
        return sum(1 for s in self.spells if s is not None)

    @property
    def grave_count(self) -> int:
        return len(self.graveyard)

    @property
    def banished_count(self) -> int:
        return len(self.banished)


@dataclass
class ActionRecord:
    """A single recorded public action for the action history."""
    turn: int = 0
    player: int = 0
    action_type: str = ""    # "summon", "attack", "activate", "set", "draw", etc.
    card_code: int = 0
    extra_info: dict = field(default_factory=dict)


@dataclass
class GameState:
    """
    Complete observable game state for both players.
    Updated incrementally by the message processor.
    """
    players: list[PlayerState] = field(
        default_factory=lambda: [PlayerState(), PlayerState()]
    )
    current_turn: int = 0
    turn_player: int = 0
    current_phase: int = 0
    chain_count: int = 0

    # Action history (public actions visible to both players)
    action_history: list[ActionRecord] = field(default_factory=list)

    # B1: Pending attack record for damage attribution
    _pending_attack_record: Optional[ActionRecord] = field(default=None, repr=False)

    # Duel result
    winner: int = -1  # -1 = ongoing, 0/1 = player won
    is_finished: bool = False

    def record_action(self, player: int, action_type: str,
                      card_code: int = 0, **extra):
        """Record a public action to the history."""
        self.action_history.append(ActionRecord(
            turn=self.current_turn,
            player=player,
            action_type=action_type,
            card_code=card_code,
            extra_info=extra,
        ))

    def get_recent_actions(self, n: int = 10) -> list[ActionRecord]:
        """Get the N most recent public actions."""
        return self.action_history[-n:]

    def reset(self):
        """Reset state for a new game."""
        self.players = [PlayerState(), PlayerState()]
        self.current_turn = 0
        self.turn_player = 0
        self.current_phase = 0
        self.chain_count = 0
        self.action_history = []
        self._pending_attack_record = None
        self.winner = -1
        self.is_finished = False


# ============================================================
# Standalone state update from messages
# ============================================================
def update_state(state: GameState, msg: ParsedMessage) -> None:
    """Update game state based on a parsed message.

    This is the canonical state-tracking logic, usable both by the
    local Duel controller and by the network bot client.
    """
    if isinstance(msg, MsgStart):
        state.players[0].lp = msg.lp[0]
        state.players[1].lp = msg.lp[1]
        state.players[0].deck_count = msg.deck_count[0]
        state.players[1].deck_count = msg.deck_count[1]
        state.players[0].extra_count = msg.extra_count[0]
        state.players[1].extra_count = msg.extra_count[1]

    elif isinstance(msg, MsgNewTurn):
        state.current_turn += 1
        state.turn_player = msg.player

    elif isinstance(msg, MsgNewPhase):
        state.current_phase = msg.phase
        state._pending_attack_record = None  # Clear stale attack pointer

    elif isinstance(msg, MsgDraw):
        p = state.players[msg.player]
        for code in msg.cards:
            p.hand.append(code & 0x7FFFFFFF)
        p.deck_count = max(0, p.deck_count - len(msg.cards))
        # Track extra draws (beyond normal draw-phase draw)
        if state.current_phase != PHASE.DRAW:
            p.extra_draws += len(msg.cards)
        state.record_action(msg.player, "draw", extra={"count": len(msg.cards)})

    elif isinstance(msg, MsgDamage):
        state.players[msg.player].lp = max(0, state.players[msg.player].lp - msg.amount)
        # B1: Attribute damage to pending attack record
        if state._pending_attack_record is not None:
            rec = state._pending_attack_record
            rec.extra_info["damage"] = rec.extra_info.get("damage", 0) + msg.amount
            state._pending_attack_record = None

    elif isinstance(msg, MsgRecover):
        state.players[msg.player].lp += msg.amount

    elif isinstance(msg, MsgLPUpdate):
        state.players[msg.player].lp = msg.lp

    elif isinstance(msg, MsgPayLPCost):
        state.players[msg.player].lp = max(0, state.players[msg.player].lp - msg.amount)

    elif isinstance(msg, MsgMove):
        _handle_move(state, msg)

    elif isinstance(msg, MsgSummoning):
        state.record_action(msg.player, "summon", msg.code)

    elif isinstance(msg, MsgSet):
        state.record_action(msg.player, "set", msg.code)

    elif isinstance(msg, MsgChaining):
        state.chain_count = msg.chain_count
        state._pending_attack_record = None  # Chain resolution clears attack context
        state.record_action(msg.player, "activate", msg.code)

    elif isinstance(msg, MsgAttack):
        record = ActionRecord(
            turn=state.current_turn,
            player=msg.attacker_player,
            action_type="attack",
            card_code=0,
            extra_info={"target_player": msg.target_player, "damage": 0, "destroyed": 0},
        )
        state.action_history.append(record)
        state._pending_attack_record = record

    elif isinstance(msg, MsgPosChange):
        p = state.players[msg.player]
        loc = msg.location
        seq = msg.sequence
        if loc & LOCATION.MZONE and 0 <= seq < 5:
            if p.monsters[seq] is not None:
                p.monsters[seq].position = msg.cur_position
        state.record_action(msg.player, "reposition", msg.code)

    elif isinstance(msg, MsgShuffleHand):
        state.players[msg.player].hand = list(msg.cards)

    elif isinstance(msg, MsgWin):
        state.winner = msg.player
        state.is_finished = True

    # B1: Handle chain resolution events via msg_type
    elif hasattr(msg, 'msg_type'):
        if msg.msg_type == MSG.CHAIN_END:
            state._pending_attack_record = None
        elif msg.msg_type == MSG.CHAIN_NEGATED:
            # Mark the last activate record as negated
            for rec in reversed(state.action_history):
                if rec.action_type == "activate":
                    rec.extra_info["was_negated"] = True
                    break


def _handle_move(state: GameState, msg: MsgMove) -> None:
    """Handle card movement between zones."""
    code = msg.code

    # Remove from source
    fl = msg.from_location
    fp = msg.from_player
    fs = msg.from_sequence

    if fl & LOCATION.HAND:
        hand = state.players[fp].hand
        for i, c in enumerate(hand):
            if c == code or (c & 0x7FFFFFFF) == (code & 0x7FFFFFFF):
                hand.pop(i)
                break
    elif fl & LOCATION.MZONE and 0 <= fs < 5:
        state.players[fp].monsters[fs] = None
    elif fl & LOCATION.SZONE and 0 <= fs < 5:
        state.players[fp].spells[fs] = None
    elif fl & LOCATION.GRAVE:
        grave = state.players[fp].graveyard
        if code in grave:
            grave.remove(code)
    elif fl & LOCATION.REMOVED:
        bans = state.players[fp].banished
        if code in bans:
            bans.remove(code)
    elif fl & LOCATION.DECK:
        state.players[fp].deck_count = max(0, state.players[fp].deck_count - 1)

    # Add to destination
    tl = msg.to_location
    tp = msg.to_player
    ts = msg.to_sequence
    tpos = msg.to_position

    if tl & LOCATION.HAND:
        state.players[tp].hand.append(code & 0x7FFFFFFF)
    elif tl & LOCATION.MZONE and 0 <= ts < 5:
        state.players[tp].monsters[ts] = ZoneCard(
            code=code, position=tpos, controller=tp, sequence=ts,
            set_turn=state.current_turn,
        )
    elif tl & LOCATION.SZONE and 0 <= ts < 5:
        state.players[tp].spells[ts] = ZoneCard(
            code=code, position=tpos, controller=tp, sequence=ts,
            set_turn=state.current_turn,
        )
    elif tl & LOCATION.GRAVE:
        state.players[tp].graveyard.append(code)
        # B1: Track destruction — field → grave counts as destroyed
        if fl & (LOCATION.MZONE | LOCATION.SZONE):
            # Attribute to most recent activate record in the chain
            for rec in reversed(state.action_history):
                if rec.action_type == "activate":
                    rec.extra_info["destroyed"] = rec.extra_info.get("destroyed", 0) + 1
                    break
    elif tl & LOCATION.REMOVED:
        state.players[tp].banished.append(code)
    elif tl & LOCATION.DECK:
        state.players[tp].deck_count += 1

    state.record_action(tp, "move", code, extra={
        "from": fl, "to": tl
    })
