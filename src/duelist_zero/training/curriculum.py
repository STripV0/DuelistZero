"""
Curriculum learning scheduler for deck progression.

Starts with mirror matches (Goat Control only), then gradually introduces
more diverse opponents as the agent masters each stage.
"""

import json
from pathlib import Path
from typing import Optional


# Deck ordering from simple to complex
DECK_ORDER = [
    "Goat Control",
    "Dragons goat",
    "My Warrior Toolbox",
    "Thunder Dragon Chaos Control",
    "Goat Format Chaos Control",
    "GOAT Reasoning Gate",
    "Empty Jar",
]


class CurriculumScheduler:
    """
    Manages curriculum-based deck progression during training.

    Tracks the current stage, records evaluation results, and determines
    when to advance to the next stage based on win rate plateau detection.
    """

    def __init__(
        self,
        deck_dir: str | Path,
        mirror_ratio: float = 0.70,
        win_rate_threshold: float = 0.60,
        min_stage_steps: int = 100_000,
        plateau_window: int = 5,
        plateau_threshold: float = 0.02,
    ):
        self.deck_dir = Path(deck_dir)
        self.mirror_ratio = mirror_ratio
        self.win_rate_threshold = win_rate_threshold
        self.min_stage_steps = min_stage_steps
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold

        # Discover available decks in order
        self._available_decks: list[str] = []
        for name in DECK_ORDER:
            path = self.deck_dir / f"{name}.ydk"
            if path.exists():
                self._available_decks.append(name)

        self.max_stage = max(0, len(self._available_decks) - 1)

        # State
        self.current_stage: int = 0
        self.stage_start_step: int = 0
        self.eval_history: list[tuple[float, int]] = []  # (win_rate, timestep)

    @property
    def deck_pool(self) -> list[Path]:
        """Return deck paths for the current stage."""
        decks = self._available_decks[: self.current_stage + 1]
        return [self.deck_dir / f"{name}.ydk" for name in decks]

    @property
    def deck_weights(self) -> list[float]:
        """Compute sampling weights: mirror gets mirror_ratio, rest split equally."""
        n = self.current_stage + 1
        if n == 1:
            return [1.0]
        remaining = 1.0 - self.mirror_ratio
        other_weight = remaining / (n - 1)
        return [self.mirror_ratio] + [other_weight] * (n - 1)

    def record_eval(self, win_rate: float, timestep: int) -> None:
        """Record an evaluation result."""
        self.eval_history.append((win_rate, timestep))

    def should_advance(self) -> bool:
        """Check if conditions are met to advance to the next stage."""
        if self.current_stage >= self.max_stage:
            return False

        if not self.eval_history:
            return False

        latest_wr, latest_step = self.eval_history[-1]

        # Must have trained enough at current stage
        if latest_step - self.stage_start_step < self.min_stage_steps:
            return False

        # Must meet minimum win rate
        if latest_wr < self.win_rate_threshold:
            return False

        # Check for plateau: win rate improvement over recent evals is small
        if len(self.eval_history) >= self.plateau_window:
            recent = [wr for wr, _ in self.eval_history[-self.plateau_window:]]
            improvement = max(recent) - min(recent)
            if improvement < self.plateau_threshold:
                return True

        return False

    def advance(self) -> int:
        """Advance to the next curriculum stage. Returns the new stage number."""
        self.current_stage = min(self.current_stage + 1, self.max_stage)
        self.stage_start_step = (
            self.eval_history[-1][1] if self.eval_history else 0
        )
        self.eval_history.clear()
        return self.current_stage

    def save_state(self, path: str | Path) -> None:
        """Save curriculum state to JSON."""
        data = {
            "current_stage": self.current_stage,
            "stage_start_step": self.stage_start_step,
            "eval_history": self.eval_history,
            "mirror_ratio": self.mirror_ratio,
            "win_rate_threshold": self.win_rate_threshold,
            "min_stage_steps": self.min_stage_steps,
            "plateau_window": self.plateau_window,
            "plateau_threshold": self.plateau_threshold,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load_state(self, path: str | Path) -> None:
        """Load curriculum state from JSON."""
        data = json.loads(Path(path).read_text())
        self.current_stage = data["current_stage"]
        self.stage_start_step = data["stage_start_step"]
        self.eval_history = [tuple(e) for e in data["eval_history"]]

    def stage_summary(self) -> str:
        """Return a human-readable summary of the current stage."""
        decks = self._available_decks[: self.current_stage + 1]
        weights = self.deck_weights
        lines = [f"Stage {self.current_stage}/{self.max_stage}:"]
        for name, w in zip(decks, weights):
            lines.append(f"  {name}: {w:.0%}")
        return "\n".join(lines)
