"""Tests for the curriculum learning scheduler."""

import json
import tempfile
from pathlib import Path

import pytest

from duelist_zero.training.curriculum import CurriculumScheduler, DECK_ORDER


@pytest.fixture
def deck_dir():
    """Use the real deck directory."""
    return Path(__file__).resolve().parents[1] / "data" / "deck"


@pytest.fixture
def scheduler(deck_dir):
    return CurriculumScheduler(deck_dir=deck_dir)


class TestStageZero:
    def test_stage_0_has_only_goat_control(self, scheduler):
        pool = scheduler.deck_pool
        assert len(pool) == 1
        assert "Goat Control" in pool[0].stem

    def test_stage_0_weights(self, scheduler):
        assert scheduler.deck_weights == [1.0]


class TestWeightComputation:
    def test_stage_1_weights(self, scheduler):
        scheduler.current_stage = 1
        weights = scheduler.deck_weights
        assert len(weights) == 2
        assert abs(weights[0] - 0.70) < 1e-6
        assert abs(weights[1] - 0.30) < 1e-6

    def test_stage_2_weights(self, scheduler):
        scheduler.current_stage = 2
        weights = scheduler.deck_weights
        assert len(weights) == 3
        assert abs(weights[0] - 0.70) < 1e-6
        assert abs(weights[1] - 0.15) < 1e-6
        assert abs(weights[2] - 0.15) < 1e-6

    def test_weights_sum_to_one(self, scheduler):
        for stage in range(scheduler.max_stage + 1):
            scheduler.current_stage = stage
            assert abs(sum(scheduler.deck_weights) - 1.0) < 1e-6


class TestAdvancement:
    def test_no_advance_below_win_rate(self, scheduler):
        """Should not advance if win rate is below threshold."""
        # Record enough evals with low win rate
        for i in range(10):
            scheduler.record_eval(0.40, (i + 1) * 20_000)
        assert not scheduler.should_advance()

    def test_no_advance_before_min_steps(self, scheduler):
        """Should not advance before min_stage_steps elapsed."""
        # High win rate but not enough steps
        for i in range(10):
            scheduler.record_eval(0.80, (i + 1) * 5_000)  # only 50k total
        assert not scheduler.should_advance()

    def test_plateau_triggers_advancement(self, scheduler):
        """Should advance when win rate plateaus above threshold."""
        # Need 2 * plateau_window (10) evals with flat improvement.
        # Early window and late window both at 0.65 → improvement < 0.03
        base = 100_000
        for i in range(10):
            scheduler.record_eval(0.65, base + (i + 1) * 20_000)
        assert scheduler.should_advance()

    def test_advance_increments_stage(self, scheduler):
        scheduler.current_stage = 0
        scheduler.record_eval(0.70, 200_000)
        new_stage = scheduler.advance()
        assert new_stage == 1
        assert scheduler.current_stage == 1

    def test_advance_clears_eval_history(self, scheduler):
        scheduler.record_eval(0.70, 200_000)
        scheduler.advance()
        assert len(scheduler.eval_history) == 0

    def test_no_advance_past_max(self, scheduler):
        scheduler.current_stage = scheduler.max_stage
        assert not scheduler.should_advance()

    def test_deck_pool_grows_with_stage(self, scheduler):
        for stage in range(scheduler.max_stage + 1):
            scheduler.current_stage = stage
            assert len(scheduler.deck_pool) == stage + 1


class TestSaveLoad:
    def test_round_trip(self, scheduler):
        scheduler.current_stage = 3
        scheduler.stage_start_step = 300_000
        scheduler.record_eval(0.65, 350_000)
        scheduler.record_eval(0.70, 400_000)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        scheduler.save_state(path)

        loaded = CurriculumScheduler(deck_dir=scheduler.deck_dir)
        loaded.load_state(path)

        assert loaded.current_stage == 3
        assert loaded.stage_start_step == 300_000
        assert len(loaded.eval_history) == 2
        assert loaded.eval_history[0] == (0.65, 350_000)

        Path(path).unlink()
