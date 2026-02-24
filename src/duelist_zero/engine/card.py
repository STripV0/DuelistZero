"""
Card data representation.
"""

from dataclasses import dataclass
from .core.constants import TYPE, ATTRIBUTE, RACE


@dataclass
class Card:
    """Represents a Yu-Gi-Oh card's static data."""
    code: int
    name: str = ""
    alias: int = 0
    type: int = 0
    level: int = 0
    attribute: int = 0
    race: int = 0
    attack: int = 0
    defense: int = 0
    lscale: int = 0
    rscale: int = 0

    @property
    def is_monster(self) -> bool:
        return bool(self.type & TYPE.MONSTER)

    @property
    def is_spell(self) -> bool:
        return bool(self.type & TYPE.SPELL)

    @property
    def is_trap(self) -> bool:
        return bool(self.type & TYPE.TRAP)

    @property
    def is_extra_deck(self) -> bool:
        return bool(self.type & TYPE.EXTRA)

    @property
    def attribute_name(self) -> str:
        from .core.constants import ATTRIBUTE_NAMES
        try:
            return ATTRIBUTE_NAMES.get(ATTRIBUTE(self.attribute), "???")
        except ValueError:
            return "???"

    def __repr__(self) -> str:
        if self.name:
            return f"Card({self.code}, '{self.name}')"
        return f"Card({self.code})"
