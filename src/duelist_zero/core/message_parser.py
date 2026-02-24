"""
Message parser for ygopro-core binary protocol.

The engine's process() function produces a byte buffer containing
one or more messages. Each message starts with a 1-byte message ID
followed by message-specific payload data.

This parser reads the buffer sequentially and extracts structured
data for each message type.

Reference: yugioh-gamepy/ygo/duel.py (process_messages, read_cardlist, etc.)
"""

import io
import struct
from dataclasses import dataclass, field
from typing import Any

from .constants import MSG, LOCATION, POSITION, PHASE


# ============================================================
# Parsed Message Types
# ============================================================
@dataclass
class ParsedMessage:
    """Base class for all parsed messages."""
    msg_type: MSG


@dataclass
class MsgDraw(ParsedMessage):
    player: int = 0
    cards: list[int] = field(default_factory=list)


@dataclass
class MsgNewTurn(ParsedMessage):
    player: int = 0


@dataclass
class MsgNewPhase(ParsedMessage):
    phase: int = 0


@dataclass
class MsgWin(ParsedMessage):
    player: int = 0
    reason: int = 0


@dataclass
class MsgDamage(ParsedMessage):
    player: int = 0
    amount: int = 0


@dataclass
class MsgRecover(ParsedMessage):
    player: int = 0
    amount: int = 0


@dataclass
class MsgLPUpdate(ParsedMessage):
    player: int = 0
    lp: int = 0


@dataclass
class MsgPayLPCost(ParsedMessage):
    player: int = 0
    amount: int = 0


@dataclass
class MsgMove(ParsedMessage):
    code: int = 0
    from_player: int = 0
    from_location: int = 0
    from_sequence: int = 0
    from_position: int = 0
    to_player: int = 0
    to_location: int = 0
    to_sequence: int = 0
    to_position: int = 0
    reason: int = 0


@dataclass
class MsgSummoning(ParsedMessage):
    code: int = 0
    player: int = 0
    location: int = 0
    sequence: int = 0
    position: int = 0


@dataclass
class MsgSet(ParsedMessage):
    code: int = 0
    player: int = 0
    location: int = 0
    sequence: int = 0
    position: int = 0


@dataclass
class MsgPosChange(ParsedMessage):
    code: int = 0
    player: int = 0
    location: int = 0
    sequence: int = 0
    prev_position: int = 0
    cur_position: int = 0


@dataclass
class MsgChaining(ParsedMessage):
    code: int = 0
    player: int = 0
    location: int = 0
    sequence: int = 0
    description: int = 0
    chain_count: int = 0


@dataclass
class MsgAttack(ParsedMessage):
    attacker_player: int = 0
    attacker_location: int = 0
    attacker_sequence: int = 0
    target_player: int = 0
    target_location: int = 0
    target_sequence: int = 0


@dataclass
class MsgBattle(ParsedMessage):
    attacker_player: int = 0
    attacker_location: int = 0
    attacker_sequence: int = 0
    attacker_atk: int = 0
    attacker_def: int = 0
    target_player: int = 0
    target_location: int = 0
    target_sequence: int = 0
    target_atk: int = 0
    target_def: int = 0


@dataclass
class CardInfo:
    """A single selectable card option."""
    code: int = 0
    controller: int = 0
    location: int = 0
    sequence: int = 0
    subsequence: int = 0
    data: int = 0  # extra info (e.g., position, effect desc)


@dataclass
class IdleCmdAction:
    """A single action available during idle command selection."""
    action_type: str = ""  # "summon", "spsummon", "reposition", "setmonster",
                           # "setspell", "activate", "toep", "tobp", "tom2"
    card: CardInfo | None = None
    description: int = 0


@dataclass
class MsgSelectIdleCmd(ParsedMessage):
    player: int = 0
    summonable: list[CardInfo] = field(default_factory=list)
    spsummonable: list[CardInfo] = field(default_factory=list)
    repositionable: list[CardInfo] = field(default_factory=list)
    setable_monsters: list[CardInfo] = field(default_factory=list)
    setable_st: list[CardInfo] = field(default_factory=list)
    activatable: list[CardInfo] = field(default_factory=list)
    activatable_descs: list[int] = field(default_factory=list)
    can_battle_phase: bool = False
    can_main2: bool = False
    can_end_phase: bool = False


@dataclass
class BattleCmdAction:
    """A single action during battle command selection."""
    card: CardInfo | None = None
    can_direct_attack: bool = False
    targets: list[CardInfo] = field(default_factory=list)


@dataclass
class MsgSelectBattleCmd(ParsedMessage):
    player: int = 0
    attackable: list[BattleCmdAction] = field(default_factory=list)
    activatable: list[CardInfo] = field(default_factory=list)
    activatable_descs: list[int] = field(default_factory=list)
    can_main2: bool = False
    can_end_phase: bool = False


@dataclass
class MsgSelectCard(ParsedMessage):
    player: int = 0
    cancelable: bool = False
    min_count: int = 0
    max_count: int = 0
    cards: list[CardInfo] = field(default_factory=list)


@dataclass
class MsgSelectChain(ParsedMessage):
    player: int = 0
    spe_count: int = 0
    forced: bool = False
    cards: list[CardInfo] = field(default_factory=list)
    descriptions: list[int] = field(default_factory=list)


@dataclass
class MsgSelectEffectYn(ParsedMessage):
    player: int = 0
    code: int = 0
    controller: int = 0
    location: int = 0
    sequence: int = 0
    description: int = 0


@dataclass
class MsgSelectYesNo(ParsedMessage):
    player: int = 0
    description: int = 0


@dataclass
class MsgSelectOption(ParsedMessage):
    player: int = 0
    options: list[int] = field(default_factory=list)


@dataclass
class MsgSelectPosition(ParsedMessage):
    player: int = 0
    code: int = 0
    positions: int = 0  # bitmask of allowed POSITION values


@dataclass
class MsgSelectPlace(ParsedMessage):
    player: int = 0
    count: int = 0
    field_mask: int = 0


@dataclass
class MsgSelectTribute(ParsedMessage):
    player: int = 0
    cancelable: bool = False
    min_count: int = 0
    max_count: int = 0
    cards: list[CardInfo] = field(default_factory=list)


@dataclass
class MsgHint(ParsedMessage):
    hint_type: int = 0
    player: int = 0
    data: int = 0


@dataclass
class MsgStart(ParsedMessage):
    player_type: int = 0
    lp: list[int] = field(default_factory=list)
    deck_count: list[int] = field(default_factory=list)
    extra_count: list[int] = field(default_factory=list)


@dataclass
class MsgShuffleHand(ParsedMessage):
    player: int = 0
    cards: list[int] = field(default_factory=list)


@dataclass
class MsgConfirmCards(ParsedMessage):
    player: int = 0
    cards: list[CardInfo] = field(default_factory=list)


@dataclass
class MsgTossCoin(ParsedMessage):
    player: int = 0
    results: list[int] = field(default_factory=list)


@dataclass
class MsgTossDice(ParsedMessage):
    player: int = 0
    results: list[int] = field(default_factory=list)


@dataclass
class MsgTagSwap(ParsedMessage):
    player: int = 0
    main_count: int = 0
    extra_count: int = 0
    extra_p_count: int = 0
    hand: list[int] = field(default_factory=list)
    extra: list[int] = field(default_factory=list)
    top_code: int = 0


@dataclass
class MsgAnnounceNumber(ParsedMessage):
    """MSG_ANNOUNCE_NUMBER: player selects a number from a list."""
    player: int = 0
    options: list[int] = field(default_factory=list)


@dataclass
class MsgAnnounceRace(ParsedMessage):
    """MSG_ANNOUNCE_RACE: player declares monster race(s)."""
    player: int = 0
    count: int = 1
    available: int = 0


@dataclass
class MsgGeneric(ParsedMessage):
    """Fallback for messages we don't parse in detail yet."""
    raw_data: bytes = b""


# ============================================================
# Buffer Reader Helper
# ============================================================
class BufferReader:
    """Sequential binary reader over a byte buffer."""

    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)

    def read_u8(self) -> int:
        return struct.unpack("<B", self._buf.read(1))[0]

    def read_i8(self) -> int:
        return struct.unpack("<b", self._buf.read(1))[0]

    def read_u16(self) -> int:
        return struct.unpack("<H", self._buf.read(2))[0]

    def read_i16(self) -> int:
        return struct.unpack("<h", self._buf.read(2))[0]

    def read_u32(self) -> int:
        return struct.unpack("<I", self._buf.read(4))[0]

    def read_i32(self) -> int:
        return struct.unpack("<i", self._buf.read(4))[0]

    def read_u64(self) -> int:
        return struct.unpack("<Q", self._buf.read(8))[0]

    def read_bytes(self, n: int) -> bytes:
        return self._buf.read(n)

    @property
    def remaining(self) -> int:
        pos = self._buf.tell()
        self._buf.seek(0, 2)
        end = self._buf.tell()
        self._buf.seek(pos)
        return end - pos

    def read_card_info(self) -> CardInfo:
        """Read a standard card location tuple: code(4) + controller(1) + location(1) + sequence(1)."""
        return CardInfo(
            code=self.read_u32(),
            controller=self.read_u8(),
            location=self.read_u8(),
            sequence=self.read_u8(),
        )

    def read_card_info_with_subsequence(self) -> CardInfo:
        """Read card info with extra subsequence byte."""
        info = self.read_card_info()
        info.subsequence = self.read_u8()
        return info

    def read_card_info_u32seq(self) -> CardInfo:
        """Read card info with u32 sequence (EDOPro format for summonable etc.)."""
        return CardInfo(
            code=self.read_u32(),
            controller=self.read_u8(),
            location=self.read_u8(),
            sequence=self.read_u32(),
        )

    def read_edopro_loc_info(self) -> CardInfo:
        """Read EDOPro loc_info: code(u32) + ctrl(u8) + loc(u8) + seq(u32) + subseq(u32)."""
        return CardInfo(
            code=self.read_u32(),
            controller=self.read_u8(),
            location=self.read_u8(),
            sequence=self.read_u32(),
            subsequence=self.read_u32(),
        )


# ============================================================
# Main Parser
# ============================================================
class MessageParser:
    """
    Parses the raw byte buffer from ygopro-core process() into
    structured ParsedMessage objects.

    Set edopro=True to handle EDOPro's ygopro-core format differences
    (u32 counts, u64 descriptions, u32 sequences, loc_info structs).
    """

    def __init__(self, edopro: bool = False):
        self.edopro = edopro
        if edopro:
            self._handlers = dict(self._default_handlers)
            self._handlers.update({
                MSG.UPDATE_DATA: MessageParser._skip_remaining_msg,
                MSG.UPDATE_CARD: MessageParser._skip_remaining_msg,
                MSG.SELECT_IDLECMD: MessageParser._parse_select_idle_cmd_edopro,
                MSG.SELECT_BATTLECMD: MessageParser._parse_select_battle_cmd_edopro,
                MSG.SELECT_CARD: MessageParser._parse_select_card_edopro,
                MSG.SELECT_TRIBUTE: MessageParser._parse_select_tribute_edopro,
                MSG.SELECT_CHAIN: MessageParser._parse_select_chain_edopro,
                MSG.SELECT_EFFECTYN: MessageParser._parse_select_effect_yn_edopro,
                MSG.SELECT_YESNO: MessageParser._parse_select_yesno_edopro,
                MSG.SELECT_OPTION: MessageParser._parse_select_option_edopro,
                MSG.SELECT_UNSELECT_CARD: MessageParser._parse_select_unselect_card_edopro,
                MSG.DRAW: MessageParser._parse_draw_edopro,
                MSG.SHUFFLE_HAND: MessageParser._parse_shuffle_hand_edopro,
            })
        else:
            self._handlers = self._default_handlers

    def parse(self, data: bytes) -> list[ParsedMessage]:
        """Parse a complete message buffer. Returns list of messages."""
        messages = []
        reader = BufferReader(data)

        while reader.remaining > 0:
            msg_id = reader.read_u8()

            try:
                msg_type = MSG(msg_id)
            except ValueError:
                # Unknown message, skip rest of buffer
                break

            handler = self._handlers.get(msg_type)
            if handler:
                msg = handler(self, reader)
                messages.append(msg)
            else:
                # Unknown/unhandled message — can't continue since we
                # don't know how many bytes it consumes
                print(f"  ⚠️  UNHANDLED MSG: {msg_type} (id={msg_id}), {reader.remaining} bytes remaining")
                messages.append(MsgGeneric(msg_type=msg_type, raw_data=reader.read_bytes(reader.remaining)))
                break

        return messages

    # ============================================================
    # Per-message parsers
    # ============================================================

    def _parse_hint(self, r: BufferReader) -> MsgHint:
        return MsgHint(
            msg_type=MSG.HINT,
            hint_type=r.read_u8(),
            player=r.read_u8(),
            data=r.read_u32(),
        )

    def _parse_start(self, r: BufferReader) -> MsgStart:
        player_type = r.read_u8()
        lp = [r.read_u32(), r.read_u32()]
        deck = [r.read_u16(), r.read_u16()]
        extra = [r.read_u16(), r.read_u16()]
        return MsgStart(
            msg_type=MSG.START,
            player_type=player_type,
            lp=lp,
            deck_count=deck,
            extra_count=extra,
        )

    def _parse_win(self, r: BufferReader) -> MsgWin:
        return MsgWin(msg_type=MSG.WIN, player=r.read_u8(), reason=r.read_u8())

    def _parse_draw(self, r: BufferReader) -> MsgDraw:
        player = r.read_u8()
        count = r.read_u8()
        cards = [r.read_u32() for _ in range(count)]
        return MsgDraw(msg_type=MSG.DRAW, player=player, cards=cards)

    def _parse_new_turn(self, r: BufferReader) -> MsgNewTurn:
        return MsgNewTurn(msg_type=MSG.NEW_TURN, player=r.read_u8())

    def _parse_new_phase(self, r: BufferReader) -> MsgNewPhase:
        return MsgNewPhase(msg_type=MSG.NEW_PHASE, phase=r.read_u16())

    def _parse_damage(self, r: BufferReader) -> MsgDamage:
        return MsgDamage(msg_type=MSG.DAMAGE, player=r.read_u8(), amount=r.read_u32())

    def _parse_recover(self, r: BufferReader) -> MsgRecover:
        return MsgRecover(msg_type=MSG.RECOVER, player=r.read_u8(), amount=r.read_u32())

    def _parse_lpupdate(self, r: BufferReader) -> MsgLPUpdate:
        return MsgLPUpdate(msg_type=MSG.LPUPDATE, player=r.read_u8(), lp=r.read_u32())

    def _parse_pay_lpcost(self, r: BufferReader) -> MsgPayLPCost:
        return MsgPayLPCost(msg_type=MSG.PAY_LPCOST, player=r.read_u8(), amount=r.read_u32())

    def _parse_move(self, r: BufferReader) -> MsgMove:
        code = r.read_u32()
        fp = r.read_u8(); fl = r.read_u8(); fs = r.read_u8(); fpos = r.read_u8()
        tp = r.read_u8(); tl = r.read_u8(); ts = r.read_u8(); tpos = r.read_u8()
        reason = r.read_u32()
        return MsgMove(
            msg_type=MSG.MOVE, code=code,
            from_player=fp, from_location=fl, from_sequence=fs, from_position=fpos,
            to_player=tp, to_location=tl, to_sequence=ts, to_position=tpos,
            reason=reason,
        )

    def _parse_summoning(self, r: BufferReader) -> MsgSummoning:
        code = r.read_u32()
        p = r.read_u8(); l = r.read_u8(); s = r.read_u8(); pos = r.read_u8()
        return MsgSummoning(msg_type=MSG.SUMMONING, code=code, player=p, location=l, sequence=s, position=pos)

    def _parse_spsummoning(self, r: BufferReader) -> MsgSummoning:
        code = r.read_u32()
        p = r.read_u8(); l = r.read_u8(); s = r.read_u8(); pos = r.read_u8()
        return MsgSummoning(msg_type=MSG.SPSUMMONING, code=code, player=p, location=l, sequence=s, position=pos)

    def _parse_flipsummoning(self, r: BufferReader) -> MsgSummoning:
        code = r.read_u32()
        p = r.read_u8(); l = r.read_u8(); s = r.read_u8(); pos = r.read_u8()
        return MsgSummoning(msg_type=MSG.FLIPSUMMONING, code=code, player=p, location=l, sequence=s, position=pos)

    def _parse_set(self, r: BufferReader) -> MsgSet:
        code = r.read_u32()
        p = r.read_u8(); l = r.read_u8(); s = r.read_u8(); pos = r.read_u8()
        return MsgSet(msg_type=MSG.SET, code=code, player=p, location=l, sequence=s, position=pos)

    def _parse_pos_change(self, r: BufferReader) -> MsgPosChange:
        code = r.read_u32()
        p = r.read_u8(); l = r.read_u8(); s = r.read_u8()
        pp = r.read_u8(); cp = r.read_u8()
        return MsgPosChange(msg_type=MSG.POS_CHANGE, code=code, player=p, location=l, sequence=s, prev_position=pp, cur_position=cp)

    def _parse_chaining(self, r: BufferReader) -> MsgChaining:
        code = r.read_u32()
        _info_loc = r.read_u32()  # info_location (packed controller+location+sequence as u32)
        p = r.read_u8(); l = r.read_u8(); s = r.read_u8()
        desc = r.read_u32()
        cc = r.read_u8()
        return MsgChaining(msg_type=MSG.CHAINING, code=code, player=p, location=l, sequence=s, description=desc, chain_count=cc)

    def _parse_attack(self, r: BufferReader) -> MsgAttack:
        ap = r.read_u8(); al = r.read_u8(); as_ = r.read_u8(); _asub = r.read_u8()
        tp = r.read_u8(); tl = r.read_u8(); ts = r.read_u8(); _tsub = r.read_u8()
        return MsgAttack(msg_type=MSG.ATTACK, attacker_player=ap, attacker_location=al, attacker_sequence=as_, target_player=tp, target_location=tl, target_sequence=ts)

    def _parse_battle(self, r: BufferReader) -> MsgBattle:
        ap = r.read_u8(); al = r.read_u8(); as_ = r.read_u8(); _asub = r.read_u8()
        aatk = r.read_i32(); adef = r.read_i32(); _afl = r.read_u8()
        tp = r.read_u8(); tl = r.read_u8(); ts = r.read_u8(); _tsub = r.read_u8()
        tatk = r.read_i32(); tdef = r.read_i32(); _tfl = r.read_u8()
        return MsgBattle(msg_type=MSG.BATTLE, attacker_player=ap, attacker_location=al, attacker_sequence=as_,
                         attacker_atk=aatk, attacker_def=adef, target_player=tp, target_location=tl,
                         target_sequence=ts, target_atk=tatk, target_def=tdef)

    def _parse_select_idle_cmd(self, r: BufferReader) -> MsgSelectIdleCmd:
        player = r.read_u8()

        # Summonable: code(4) + ctrl(1) + loc(1) + seq(1)
        count = r.read_u8()
        summonable = [r.read_card_info() for _ in range(count)]

        # Special summonable
        count = r.read_u8()
        spsummonable = [r.read_card_info() for _ in range(count)]

        # Repositionable (change position)
        count = r.read_u8()
        repos = [r.read_card_info() for _ in range(count)]

        # Set-able monsters
        count = r.read_u8()
        setmons = [r.read_card_info() for _ in range(count)]

        # Set-able spell/traps
        count = r.read_u8()
        setst = [r.read_card_info() for _ in range(count)]

        # Activatable: code(4) + ctrl(1) + loc(1) + seq(1) + desc(4)
        count = r.read_u8()
        activatable = []
        descs = []
        for _ in range(count):
            ci = r.read_card_info()
            desc = r.read_u32()
            activatable.append(ci)
            descs.append(desc)

        # Phase flags: can_bp(1) + can_ep(1) + can_shuffle(1)
        can_bp = r.read_u8() != 0
        can_ep = r.read_u8() != 0
        _can_shuffle = r.read_u8()  # shuffle hand flag, unused for AI

        return MsgSelectIdleCmd(
            msg_type=MSG.SELECT_IDLECMD,
            player=player,
            summonable=summonable,
            spsummonable=spsummonable,
            repositionable=repos,
            setable_monsters=setmons,
            setable_st=setst,
            activatable=activatable,
            activatable_descs=descs,
            can_battle_phase=can_bp,
            can_end_phase=can_ep,
        )

    def _parse_select_battle_cmd(self, r: BufferReader) -> MsgSelectBattleCmd:
        player = r.read_u8()

        # Activatable: code(4) + ctrl(1) + loc(1) + seq(1) + desc(4)
        act_count = r.read_u8()
        activatable = []
        act_descs = []
        for _ in range(act_count):
            ci = r.read_card_info()
            desc = r.read_u32()
            activatable.append(ci)
            act_descs.append(desc)

        # Attackable: code(4) + ctrl(1) + loc(1) + seq(1) + direct(1)
        atk_count = r.read_u8()
        attackable = []
        for _ in range(atk_count):
            ci = r.read_card_info()
            can_direct = r.read_u8() != 0
            attackable.append(BattleCmdAction(card=ci, can_direct_attack=can_direct))

        can_m2 = r.read_u8() != 0
        can_ep = r.read_u8() != 0

        return MsgSelectBattleCmd(
            msg_type=MSG.SELECT_BATTLECMD,
            player=player,
            attackable=attackable,
            activatable=activatable,
            activatable_descs=act_descs,
            can_main2=can_m2,
            can_end_phase=can_ep,
        )

    def _parse_select_card(self, r: BufferReader) -> MsgSelectCard:
        player = r.read_u8()
        cancelable = r.read_u8() != 0
        min_count = r.read_u8()
        max_count = r.read_u8()
        count = r.read_u8()
        cards = []
        for _ in range(count):
            # code(4) + info_location(4) — packed location from get_select_info_location()
            code = r.read_u32()
            info_loc = r.read_u32()
            ci = CardInfo(
                code=code,
                controller=(info_loc >> 24) & 0xFF,
                location=(info_loc >> 16) & 0xFF,
                sequence=(info_loc >> 8) & 0xFF,
                subsequence=info_loc & 0xFF,
            )
            cards.append(ci)
        return MsgSelectCard(
            msg_type=MSG.SELECT_CARD, player=player,
            cancelable=cancelable, min_count=min_count, max_count=max_count,
            cards=cards,
        )

    def _parse_select_tribute(self, r: BufferReader) -> MsgSelectTribute:
        player = r.read_u8()
        cancelable = r.read_u8() != 0
        min_count = r.read_u8()
        max_count = r.read_u8()
        count = r.read_u8()
        cards = []
        for _ in range(count):
            # code(4) + ctrl(1) + loc(1) + seq(1) + release_param(1)
            ci = r.read_card_info()
            ci.data = r.read_u8()  # release_param
            cards.append(ci)
        return MsgSelectTribute(
            msg_type=MSG.SELECT_TRIBUTE, player=player,
            cancelable=cancelable, min_count=min_count, max_count=max_count,
            cards=cards,
        )

    def _parse_select_chain(self, r: BufferReader) -> MsgSelectChain:
        player = r.read_u8()
        count = r.read_u8()
        spe_count = r.read_u8()
        _hint_timing0 = r.read_u32()
        _hint_timing1 = r.read_u32()
        forced = False
        cards = []
        descs = []
        for _ in range(count):
            _edesc_flag = r.read_u8()
            is_forced = r.read_u8() != 0
            if is_forced:
                forced = True
            code = r.read_u32()
            info_loc = r.read_u32()  # packed info_location
            desc = r.read_u32()
            ci = CardInfo(
                code=code,
                controller=(info_loc >> 24) & 0xFF,
                location=(info_loc >> 16) & 0xFF,
                sequence=(info_loc >> 8) & 0xFF,
                subsequence=info_loc & 0xFF,
            )
            cards.append(ci)
            descs.append(desc)
        return MsgSelectChain(
            msg_type=MSG.SELECT_CHAIN, player=player,
            spe_count=spe_count, forced=forced,
            cards=cards, descriptions=descs,
        )

    def _parse_select_effect_yn(self, r: BufferReader) -> MsgSelectEffectYn:
        player = r.read_u8()
        code = r.read_u32()
        info_loc = r.read_u32()  # packed info_location from get_info_location()
        desc = r.read_u32()
        return MsgSelectEffectYn(
            msg_type=MSG.SELECT_EFFECTYN, player=player,
            code=code,
            controller=(info_loc >> 24) & 0xFF,
            location=(info_loc >> 16) & 0xFF,
            sequence=(info_loc >> 8) & 0xFF,
            description=desc,
        )

    def _parse_select_yesno(self, r: BufferReader) -> MsgSelectYesNo:
        player = r.read_u8()
        desc = r.read_u32()
        return MsgSelectYesNo(msg_type=MSG.SELECT_YESNO, player=player, description=desc)

    def _parse_select_option(self, r: BufferReader) -> MsgSelectOption:
        player = r.read_u8()
        count = r.read_u8()
        options = [r.read_u32() for _ in range(count)]
        return MsgSelectOption(msg_type=MSG.SELECT_OPTION, player=player, options=options)

    def _parse_select_position(self, r: BufferReader) -> MsgSelectPosition:
        player = r.read_u8()
        code = r.read_u32()
        positions = r.read_u8()
        return MsgSelectPosition(msg_type=MSG.SELECT_POSITION, player=player, code=code, positions=positions)

    def _parse_select_place(self, r: BufferReader, msg_type: MSG = MSG.SELECT_PLACE) -> MsgSelectPlace:
        player = r.read_u8()
        count = r.read_u8()
        field_mask = r.read_u32()
        return MsgSelectPlace(msg_type=msg_type, player=player, count=count, field_mask=field_mask)

    def _parse_shuffle_hand(self, r: BufferReader) -> MsgShuffleHand:
        player = r.read_u8()
        count = r.read_u8()
        cards = [r.read_u32() for _ in range(count)]
        return MsgShuffleHand(msg_type=MSG.SHUFFLE_HAND, player=player, cards=cards)

    def _parse_confirm_cards(self, r: BufferReader) -> MsgConfirmCards:
        player = r.read_u8()
        _skip_panel = r.read_u8()  # skip_panel flag (only in MSG_CONFIRM_CARDS)
        count = r.read_u8()
        cards = []
        for _ in range(count):
            ci = CardInfo(
                code=r.read_u32(),
                controller=r.read_u8(),
                location=r.read_u8(),
                sequence=r.read_u8(),
            )
            cards.append(ci)
        return MsgConfirmCards(msg_type=MSG.CONFIRM_CARDS, player=player, cards=cards)

    def _parse_confirm_top(self, r: BufferReader, msg_type: MSG) -> MsgConfirmCards:
        """Parse MSG_CONFIRM_DECKTOP / MSG_CONFIRM_EXTRATOP (no skip_panel byte)."""
        player = r.read_u8()
        count = r.read_u8()
        cards = []
        for _ in range(count):
            ci = CardInfo(
                code=r.read_u32(),
                controller=r.read_u8(),
                location=r.read_u8(),
                sequence=r.read_u8(),
            )
            cards.append(ci)
        return MsgConfirmCards(msg_type=msg_type, player=player, cards=cards)

    def _parse_toss_coin(self, r: BufferReader) -> MsgTossCoin:
        player = r.read_u8()
        count = r.read_u8()
        results = [r.read_u8() for _ in range(count)]
        return MsgTossCoin(msg_type=MSG.TOSS_COIN, player=player, results=results)

    def _parse_toss_dice(self, r: BufferReader) -> MsgTossDice:
        player = r.read_u8()
        count = r.read_u8()
        results = [r.read_u8() for _ in range(count)]
        return MsgTossDice(msg_type=MSG.TOSS_DICE, player=player, results=results)

    def _parse_tag_swap(self, r: BufferReader) -> MsgTagSwap:
        player = r.read_u8()
        main_count = r.read_u8()
        extra_count = r.read_u8()
        extra_p_count = r.read_u8()
        hand_count = r.read_u8()
        top_code = r.read_u32()
        hand = [r.read_u32() for _ in range(hand_count)]
        extra = [r.read_u32() for _ in range(extra_count)]
        return MsgTagSwap(
            msg_type=MSG.TAG_SWAP, player=player,
            main_count=main_count, extra_count=extra_count,
            extra_p_count=extra_p_count, hand=hand, extra=extra,
            top_code=top_code,
        )

    # Zero-payload messages — consume no extra bytes
    def _parse_empty(self, msg_type: MSG):
        def parser(self, r: BufferReader) -> ParsedMessage:
            return ParsedMessage(msg_type=msg_type)
        return parser

    # ============================================================
    # Handler dispatch table
    # ============================================================
    _default_handlers = {
        MSG.HINT: _parse_hint,
        MSG.START: _parse_start,
        MSG.WIN: _parse_win,
        MSG.DRAW: _parse_draw,
        MSG.NEW_TURN: _parse_new_turn,
        MSG.NEW_PHASE: _parse_new_phase,
        MSG.DAMAGE: _parse_damage,
        MSG.RECOVER: _parse_recover,
        MSG.LPUPDATE: _parse_lpupdate,
        MSG.PAY_LPCOST: _parse_pay_lpcost,
        MSG.MOVE: _parse_move,
        MSG.SUMMONING: _parse_summoning,
        MSG.SPSUMMONING: _parse_spsummoning,
        MSG.FLIPSUMMONING: _parse_flipsummoning,
        MSG.SET: _parse_set,
        MSG.POS_CHANGE: _parse_pos_change,
        MSG.CHAINING: _parse_chaining,
        MSG.ATTACK: _parse_attack,
        MSG.BATTLE: _parse_battle,
        MSG.SELECT_IDLECMD: _parse_select_idle_cmd,
        MSG.SELECT_BATTLECMD: _parse_select_battle_cmd,
        MSG.SELECT_CARD: _parse_select_card,
        MSG.SELECT_TRIBUTE: _parse_select_tribute,
        MSG.SELECT_CHAIN: _parse_select_chain,
        MSG.SELECT_EFFECTYN: _parse_select_effect_yn,
        MSG.SELECT_YESNO: _parse_select_yesno,
        MSG.SELECT_OPTION: _parse_select_option,
        MSG.SELECT_POSITION: _parse_select_position,
        MSG.SELECT_PLACE: lambda self, r: self._parse_select_place(r, MSG.SELECT_PLACE),
        MSG.SELECT_DISFIELD: lambda self, r: self._parse_select_place(r, MSG.SELECT_DISFIELD),
        MSG.SHUFFLE_HAND: _parse_shuffle_hand,
        MSG.CONFIRM_CARDS: _parse_confirm_cards,
        MSG.CONFIRM_DECKTOP: lambda self, r: self._parse_confirm_top(r, MSG.CONFIRM_DECKTOP),
        MSG.CONFIRM_EXTRATOP: lambda self, r: self._parse_confirm_top(r, MSG.CONFIRM_EXTRATOP),
        MSG.TOSS_COIN: _parse_toss_coin,
        MSG.TOSS_DICE: _parse_toss_dice,
        MSG.TAG_SWAP: _parse_tag_swap,
    }

    # ============================================================
    # Fixed-size message stubs (consume known byte counts)
    # These are needed so the parser can skip past them without
    # breaking the buffer read position.
    # ============================================================
    @staticmethod
    def _skip_0(r: BufferReader, msg_type: MSG) -> ParsedMessage:
        """0 extra bytes."""
        return ParsedMessage(msg_type=msg_type)

    @staticmethod
    def _skip_1(r: BufferReader, msg_type: MSG) -> ParsedMessage:
        """1 byte payload (e.g. player or chain count)."""
        r.read_u8()
        return ParsedMessage(msg_type=msg_type)

    @staticmethod
    def _skip_4(r: BufferReader, msg_type: MSG) -> ParsedMessage:
        """4 byte payload."""
        r.read_u32()
        return ParsedMessage(msg_type=msg_type)

    @staticmethod
    def _skip_8(r: BufferReader, msg_type: MSG) -> ParsedMessage:
        """8 byte payload (e.g. two card locations)."""
        r.read_u32(); r.read_u32()
        return ParsedMessage(msg_type=msg_type)

    @staticmethod
    def _skip_card_list(r: BufferReader, msg_type: MSG) -> ParsedMessage:
        """count(1) + count * info_location(4)."""
        count = r.read_u8()
        for _ in range(count):
            r.read_u32()  # info_location
        return ParsedMessage(msg_type=msg_type)

    @staticmethod
    def _skip_shuffle_deck(r: BufferReader, msg_type: MSG) -> ParsedMessage:
        """player(1)."""
        r.read_u8()
        return ParsedMessage(msg_type=msg_type)

    @staticmethod
    def _skip_shuffle_set_card(r: BufferReader, msg_type: MSG) -> ParsedMessage:
        """location(1) + count(1) + count * (from_loc(4) + to_loc(4))."""
        _loc = r.read_u8()
        count = r.read_u8()
        for _ in range(count):
            r.read_u32()  # from info
            r.read_u32()  # to info
        return ParsedMessage(msg_type=msg_type)

    @staticmethod
    def _skip_deck_top(r: BufferReader, msg_type: MSG) -> ParsedMessage:
        """player(1) + count(1) + code(4)."""
        r.read_u8(); r.read_u8(); r.read_u32()
        return ParsedMessage(msg_type=msg_type)

    @staticmethod
    def _skip_shuffle_extra(r: BufferReader, msg_type: MSG) -> ParsedMessage:
        """player(1) + count(1) + count * code(4)."""
        r.read_u8()
        count = r.read_u8()
        for _ in range(count):
            r.read_u32()
        return ParsedMessage(msg_type=msg_type)

    @staticmethod
    def _skip_card_hint(r: BufferReader, msg_type: MSG) -> ParsedMessage:
        """info_location(4) + hint_type(1) + value(4)."""
        r.read_u32()  # info_location
        r.read_u8()   # hint_type
        r.read_u32()  # value
        return ParsedMessage(msg_type=msg_type)

    @staticmethod
    def _parse_announce_number(r: BufferReader) -> "MsgAnnounceNumber":
        """player(1) + count(1) + options[count](4 each)."""
        player = r.read_u8()
        count = r.read_u8()
        options = [r.read_u32() for _ in range(count)]
        return MsgAnnounceNumber(msg_type=MSG.ANNOUNCE_NUMBER, player=player, options=options)

    @staticmethod
    def _parse_announce_race(r: BufferReader) -> "MsgAnnounceRace":
        """player(1) + count(1) + available(4)."""
        player = r.read_u8()
        count = r.read_u8()
        available = r.read_u32()
        return MsgAnnounceRace(msg_type=MSG.ANNOUNCE_RACE, player=player, count=count, available=available)

    @staticmethod
    def _skip_add_remove_counter(r: BufferReader, msg_type: MSG) -> ParsedMessage:
        """counter_type(2) + player(1) + location(1) + seq(1) + count(2)."""
        r.read_u16(); r.read_u8(); r.read_u8(); r.read_u8(); r.read_u16()
        return ParsedMessage(msg_type=msg_type)

    @staticmethod
    def _skip_missed_effect(r: BufferReader, msg_type: MSG) -> ParsedMessage:
        """info_location(4) + code(4)."""
        r.read_u32(); r.read_u32()
        return ParsedMessage(msg_type=msg_type)

    @staticmethod
    def _skip_be_chain_target(r: BufferReader, msg_type: MSG) -> ParsedMessage:
        """Same as BECOME_TARGET: count(1) + count * info_location(4)."""
        count = r.read_u8()
        for _ in range(count):
            r.read_u32()
        return ParsedMessage(msg_type=msg_type)

    @staticmethod
    def _skip_random_selected(r: BufferReader, msg_type: MSG) -> ParsedMessage:
        """player(1) + count(1) + count * info_location(4)."""
        r.read_u8()
        count = r.read_u8()
        for _ in range(count):
            r.read_u32()
        return ParsedMessage(msg_type=msg_type)

    @staticmethod
    def _skip_player_hint(r: BufferReader, msg_type: MSG) -> ParsedMessage:
        """player(1) + hint_type(1) + value(4)."""
        r.read_u8()   # player
        r.read_u8()   # hint_type
        r.read_u32()  # value
        return ParsedMessage(msg_type=msg_type)

    # ============================================================
    # EDOPro-specific parsers (wider types, different layouts)
    # ============================================================

    @staticmethod
    def _skip_remaining_msg(self, r: BufferReader) -> ParsedMessage:
        """Skip all remaining bytes (for UPDATE_DATA, UPDATE_CARD, etc.)."""
        r.read_bytes(r.remaining)
        return ParsedMessage(msg_type=MSG.UPDATE_DATA)

    def _parse_draw_edopro(self, r: BufferReader) -> MsgDraw:
        """EDOPro DRAW: player(u8) + count(u32) + cards[count](code_u32 + pos_u32)."""
        player = r.read_u8()
        count = r.read_u32()
        cards = []
        for _ in range(count):
            code = r.read_u32()
            _position = r.read_u32()  # position info, unused
            cards.append(code)
        return MsgDraw(msg_type=MSG.DRAW, player=player, cards=cards)

    def _parse_shuffle_hand_edopro(self, r: BufferReader) -> MsgShuffleHand:
        """EDOPro SHUFFLE_HAND: player(u8) + count(u32) + cards[count](u32)."""
        player = r.read_u8()
        count = r.read_u32()
        cards = [r.read_u32() for _ in range(count)]
        return MsgShuffleHand(msg_type=MSG.SHUFFLE_HAND, player=player, cards=cards)

    def _parse_select_idle_cmd_edopro(self, r: BufferReader) -> MsgSelectIdleCmd:
        player = r.read_u8()

        # Summonable: code(4) + ctrl(1) + loc(1) + seq(4)
        count = r.read_u32()
        summonable = [r.read_card_info_u32seq() for _ in range(count)]

        # Special summonable
        count = r.read_u32()
        spsummonable = [r.read_card_info_u32seq() for _ in range(count)]

        # Repositionable: seq stays u8
        count = r.read_u32()
        repos = [r.read_card_info() for _ in range(count)]

        # Set-able monsters
        count = r.read_u32()
        setmons = [r.read_card_info_u32seq() for _ in range(count)]

        # Set-able spell/traps
        count = r.read_u32()
        setst = [r.read_card_info_u32seq() for _ in range(count)]

        # Activatable: code(4) + ctrl(1) + loc(1) + seq(4) + desc(8) + mode(1)
        count = r.read_u32()
        activatable = []
        descs = []
        for _ in range(count):
            ci = r.read_card_info_u32seq()
            desc = r.read_u64()
            _mode = r.read_u8()
            activatable.append(ci)
            descs.append(desc & 0xFFFFFFFF)  # truncate to u32 for compatibility

        can_bp = r.read_u8() != 0
        can_ep = r.read_u8() != 0
        _can_shuffle = r.read_u8()

        return MsgSelectIdleCmd(
            msg_type=MSG.SELECT_IDLECMD,
            player=player, summonable=summonable, spsummonable=spsummonable,
            repositionable=repos, setable_monsters=setmons, setable_st=setst,
            activatable=activatable, activatable_descs=descs,
            can_battle_phase=can_bp, can_end_phase=can_ep,
        )

    def _parse_select_battle_cmd_edopro(self, r: BufferReader) -> MsgSelectBattleCmd:
        player = r.read_u8()

        # Activatable: code(4) + ctrl(1) + loc(1) + seq(4) + desc(8) + mode(1)
        act_count = r.read_u32()
        activatable = []
        act_descs = []
        for _ in range(act_count):
            ci = r.read_card_info_u32seq()
            desc = r.read_u64()
            _mode = r.read_u8()
            activatable.append(ci)
            act_descs.append(desc & 0xFFFFFFFF)

        # Attackable: code(4) + ctrl(1) + loc(1) + seq(1) + direct(1)
        atk_count = r.read_u32()
        attackable = []
        for _ in range(atk_count):
            ci = r.read_card_info()
            can_direct = r.read_u8() != 0
            attackable.append(BattleCmdAction(card=ci, can_direct_attack=can_direct))

        can_m2 = r.read_u8() != 0
        can_ep = r.read_u8() != 0

        return MsgSelectBattleCmd(
            msg_type=MSG.SELECT_BATTLECMD,
            player=player, attackable=attackable, activatable=activatable,
            activatable_descs=act_descs, can_main2=can_m2, can_end_phase=can_ep,
        )

    def _parse_select_card_edopro(self, r: BufferReader) -> MsgSelectCard:
        player = r.read_u8()
        cancelable = r.read_u8() != 0
        min_count = r.read_u32()
        max_count = r.read_u32()
        count = r.read_u32()
        cards = []
        for _ in range(count):
            ci = r.read_edopro_loc_info()
            cards.append(ci)
        return MsgSelectCard(
            msg_type=MSG.SELECT_CARD, player=player,
            cancelable=cancelable, min_count=min_count, max_count=max_count,
            cards=cards,
        )

    def _parse_select_tribute_edopro(self, r: BufferReader) -> MsgSelectTribute:
        player = r.read_u8()
        cancelable = r.read_u8() != 0
        min_count = r.read_u32()
        max_count = r.read_u32()
        count = r.read_u32()
        cards = []
        for _ in range(count):
            ci = r.read_card_info_u32seq()
            ci.data = r.read_u8()  # release_param
            cards.append(ci)
        return MsgSelectTribute(
            msg_type=MSG.SELECT_TRIBUTE, player=player,
            cancelable=cancelable, min_count=min_count, max_count=max_count,
            cards=cards,
        )

    def _parse_select_chain_edopro(self, r: BufferReader) -> MsgSelectChain:
        player = r.read_u8()
        spe_count = r.read_u8()
        forced = r.read_u8() != 0
        _hint_timing0 = r.read_u32()
        _hint_timing1 = r.read_u32()
        count = r.read_u32()
        cards = []
        descs = []
        for _ in range(count):
            ci = r.read_edopro_loc_info()
            desc = r.read_u64()
            _mode = r.read_u8()
            cards.append(ci)
            descs.append(desc & 0xFFFFFFFF)
        return MsgSelectChain(
            msg_type=MSG.SELECT_CHAIN, player=player,
            spe_count=spe_count, forced=forced,
            cards=cards, descriptions=descs,
        )

    def _parse_select_effect_yn_edopro(self, r: BufferReader) -> MsgSelectEffectYn:
        player = r.read_u8()
        code = r.read_u32()
        ctrl = r.read_u8()
        loc = r.read_u8()
        seq = r.read_u32()
        _subseq = r.read_u32()
        desc = r.read_u64()
        return MsgSelectEffectYn(
            msg_type=MSG.SELECT_EFFECTYN, player=player,
            code=code, controller=ctrl, location=loc, sequence=seq,
            description=desc & 0xFFFFFFFF,
        )

    def _parse_select_yesno_edopro(self, r: BufferReader) -> MsgSelectYesNo:
        player = r.read_u8()
        desc = r.read_u64()
        return MsgSelectYesNo(
            msg_type=MSG.SELECT_YESNO, player=player,
            description=desc & 0xFFFFFFFF,
        )

    def _parse_select_option_edopro(self, r: BufferReader) -> MsgSelectOption:
        player = r.read_u8()
        count = r.read_u8()
        options = [r.read_u64() & 0xFFFFFFFF for _ in range(count)]
        return MsgSelectOption(
            msg_type=MSG.SELECT_OPTION, player=player, options=options,
        )

    def _parse_select_unselect_card_edopro(self, r: BufferReader) -> MsgSelectCard:
        player = r.read_u8()
        _finishable = r.read_u8()
        cancelable = r.read_u8() != 0
        min_count = r.read_u32()
        max_count = r.read_u32()
        count1 = r.read_u32()
        cards = []
        for _ in range(count1):
            ci = r.read_edopro_loc_info()
            cards.append(ci)
        count2 = r.read_u32()
        for _ in range(count2):
            r.read_edopro_loc_info()  # unselect cards, skip
        return MsgSelectCard(
            msg_type=MSG.SELECT_UNSELECT_CARD, player=player,
            cancelable=cancelable, min_count=min_count, max_count=max_count,
            cards=cards,
        )

    # Build the full handler table including stubs
    @classmethod
    def _build_full_handlers(cls):
        """Extend _default_handlers with all stub handlers."""
        h = dict(cls._default_handlers)

        # 0-byte messages
        for msg in [MSG.RETRY, MSG.WAITING, MSG.SUMMONED, MSG.SPSUMMONED,
                     MSG.FLIPSUMMONED, MSG.CHAIN_END, MSG.ATTACK_DISABLED,
                     MSG.DAMAGE_STEP_START, MSG.DAMAGE_STEP_END, MSG.REVERSE_DECK]:
            h[msg] = lambda self, r, mt=msg: cls._skip_0(r, mt)

        # 1-byte messages (player or chain_count)
        for msg in [MSG.SHUFFLE_DECK, MSG.CHAINED, MSG.CHAIN_SOLVING,
                     MSG.CHAIN_SOLVED, MSG.CHAIN_NEGATED, MSG.CHAIN_DISABLED,
                     MSG.SWAP_GRAVE_DECK, MSG.REFRESH_DECK]:
            h[msg] = lambda self, r, mt=msg: cls._skip_1(r, mt)

        # 4-byte messages
        for msg in [MSG.FIELD_DISABLED, MSG.UNEQUIP]:
            h[msg] = lambda self, r, mt=msg: cls._skip_4(r, mt)

        # 8-byte messages (two packed locations)
        for msg in [MSG.EQUIP, MSG.CARD_TARGET, MSG.CANCEL_TARGET]:
            h[msg] = lambda self, r, mt=msg: cls._skip_8(r, mt)

        # count(1) + count * u32
        for msg in [MSG.CARD_SELECTED, MSG.BECOME_TARGET]:
            h[msg] = lambda self, r, mt=msg: cls._skip_card_list(r, mt)

        # Specific formats
        h[MSG.SHUFFLE_SET_CARD] = lambda self, r: cls._skip_shuffle_set_card(r, MSG.SHUFFLE_SET_CARD)
        h[MSG.DECK_TOP] = lambda self, r: cls._skip_deck_top(r, MSG.DECK_TOP)
        h[MSG.SHUFFLE_EXTRA] = lambda self, r: cls._skip_shuffle_extra(r, MSG.SHUFFLE_EXTRA)
        h[MSG.CARD_HINT] = lambda self, r: cls._skip_card_hint(r, MSG.CARD_HINT)
        h[MSG.ADD_COUNTER] = lambda self, r: cls._skip_add_remove_counter(r, MSG.ADD_COUNTER)
        h[MSG.REMOVE_COUNTER] = lambda self, r: cls._skip_add_remove_counter(r, MSG.REMOVE_COUNTER)
        h[MSG.MISSED_EFFECT] = lambda self, r: cls._skip_missed_effect(r, MSG.MISSED_EFFECT)
        h[MSG.BE_CHAIN_TARGET] = lambda self, r: cls._skip_be_chain_target(r, MSG.BE_CHAIN_TARGET)
        h[MSG.RANDOM_SELECTED] = lambda self, r: cls._skip_random_selected(r, MSG.RANDOM_SELECTED)
        h[MSG.SWAP] = lambda self, r: ParsedMessage(msg_type=MSG.SWAP) if not r.read_bytes(16) else ParsedMessage(msg_type=MSG.SWAP)
        h[MSG.PLAYER_HINT] = lambda self, r: cls._skip_player_hint(r, MSG.PLAYER_HINT)
        h[MSG.ANNOUNCE_NUMBER] = lambda self, r: cls._parse_announce_number(r)
        h[MSG.ANNOUNCE_RACE] = lambda self, r: cls._parse_announce_race(r)

        return h


# Build full handler table at import time
MessageParser._default_handlers = MessageParser._build_full_handlers()


# ============================================================
# Convenience: which messages require a player response?
# ============================================================
RESPONSE_MESSAGES = frozenset({
    MSG.SELECT_IDLECMD,
    MSG.SELECT_BATTLECMD,
    MSG.SELECT_CARD,
    MSG.SELECT_TRIBUTE,
    MSG.SELECT_CHAIN,
    MSG.SELECT_EFFECTYN,
    MSG.SELECT_YESNO,
    MSG.SELECT_OPTION,
    MSG.SELECT_POSITION,
    MSG.SELECT_PLACE,
    MSG.SELECT_COUNTER,
    MSG.SELECT_SUM,
    MSG.SELECT_DISFIELD,
    MSG.SELECT_UNSELECT_CARD,
    MSG.SORT_CHAIN,
    MSG.SORT_CARD,
    MSG.ANNOUNCE_RACE,
    MSG.ANNOUNCE_ATTRIB,
    MSG.ANNOUNCE_CARD,
    MSG.ANNOUNCE_NUMBER,
    MSG.ROCK_PAPER_SCISSORS,
})


def try_parse_hint(msg: ParsedMessage) -> "MsgHint | None":
    """Return msg as MsgHint if it is one, otherwise None."""
    if isinstance(msg, MsgHint):
        return msg
    return None
