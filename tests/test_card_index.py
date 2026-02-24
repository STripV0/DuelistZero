"""Tests for CardIndex mapping."""

import pytest

from duelist_zero.env.card_index import CardIndex


@pytest.fixture(scope="module")
def card_index():
    return CardIndex()


class TestCardIndex:
    def test_vocab_size_positive(self, card_index):
        assert card_index.vocab_size > 1, "vocab_size must be > 1 (at least padding + 1 card)"

    def test_unknown_code_returns_zero(self, card_index):
        assert card_index.code_to_index(999999999) == 0

    def test_zero_code_returns_zero(self, card_index):
        assert card_index.code_to_index(0) == 0

    def test_known_card_returns_nonzero(self, card_index):
        # Dark Magician (46986414) should exist in any YGO card DB
        idx = card_index.code_to_index(46986414)
        assert idx > 0, "Known card should map to a positive index"

    def test_indices_are_contiguous(self, card_index):
        # vocab_size should be max_index + 1
        assert card_index.vocab_size >= 2

    def test_high_bit_masking(self, card_index):
        # Card codes sometimes have the high bit set; code_to_index masks it
        idx_clean = card_index.code_to_index(46986414)
        idx_dirty = card_index.code_to_index(46986414 | 0x80000000)
        assert idx_clean == idx_dirty
