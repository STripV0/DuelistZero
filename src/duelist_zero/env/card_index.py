"""
Card index mapping for embedding lookups.

Maps large non-sequential card codes (e.g., 18036057) to contiguous
integer indices suitable for nn.Embedding. Index 0 is reserved for
empty/unknown/face-down cards.
"""

import sqlite3
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DB = _PROJECT_ROOT / "data" / "cards.cdb"


class CardIndex:
    """Bidirectional mapping between card codes and contiguous embedding indices."""

    def __init__(self, db_path: str | Path = _DEFAULT_DB):
        db_path = Path(db_path)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        cur = conn.cursor()
        cur.execute("SELECT id FROM datas ORDER BY id")
        codes = [row[0] for row in cur.fetchall()]
        conn.close()

        # Index 0 = padding/unknown, actual cards start at 1
        self._code_to_idx: dict[int, int] = {}
        for i, code in enumerate(codes, start=1):
            self._code_to_idx[code] = i

    def code_to_index(self, code: int) -> int:
        """Map a card code to its embedding index. Returns 0 for unknown cards."""
        return self._code_to_idx.get(code & 0x7FFFFFFF, 0)

    @property
    def vocab_size(self) -> int:
        """Total embedding table size (num_cards + 1 for padding index 0)."""
        return len(self._code_to_idx) + 1
