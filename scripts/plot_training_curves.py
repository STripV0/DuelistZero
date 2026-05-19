"""Parse training logs and render the curves embedded in the README.

Each training run prints alternating ``[SelfPlay] Checkpoint saved: ckpt_<step>``
and ``[Eval] vs Heuristic: NN% (...)`` lines. This script pairs them into
(timesteps, win_rate) series, detects self-play activation events, and writes
two figures to ``docs/``:

  * training_curve.png        — final architecture run, annotated.
  * architecture_iterations.png — comparison across iterations.

Run from the repo root::

    uv run python scripts/plot_training_curves.py
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CKPT_RE = re.compile(r"\[SelfPlay\] Checkpoint saved: ckpt_(\d+)")
EVAL_RE = re.compile(r"\[Eval\] vs Heuristic: (\d+)%")
ACTIVATED_RE = re.compile(r"\[SelfPlay\] (ACTIVATED|FORCE-ACTIVATED)")
DEACTIVATED_RE = re.compile(r"\[SelfPlay\] DEACTIVATED")


@dataclass
class Run:
    label: str
    path: Path
    color: str
    steps: list[int] = field(default_factory=list)
    wr: list[int] = field(default_factory=list)
    activated_at: int | None = None


def parse(path: Path) -> tuple[list[int], list[int], int | None]:
    """Pair checkpoint+eval lines. Filter resume duplicates by keeping the
    longest contiguous monotonically-increasing prefix."""
    raw_steps: list[int] = []
    raw_wr: list[int] = []
    activated_at: int | None = None
    last_ckpt: int | None = None
    for line in path.read_text(errors="ignore").splitlines():
        if m := CKPT_RE.search(line):
            last_ckpt = int(m.group(1))
            continue
        if m := EVAL_RE.search(line):
            if last_ckpt is not None:
                raw_steps.append(last_ckpt)
                raw_wr.append(int(m.group(1)))
                last_ckpt = None
            continue
        if activated_at is None and ACTIVATED_RE.search(line):
            activated_at = raw_steps[-1] if raw_steps else 0

    steps: list[int] = []
    wr: list[int] = []
    for s, w in zip(raw_steps, raw_wr):
        if steps and s <= steps[-1]:
            break  # resume detected — drop trailing duplicates
        steps.append(s)
        wr.append(w)
    return steps, wr, activated_at


def smooth(y: list[int], window: int = 5) -> np.ndarray:
    if len(y) < window:
        return np.asarray(y, dtype=float)
    arr = np.asarray(y, dtype=float)
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "figure.dpi": 130,
    })


def plot_final(run: Run, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    steps_m = np.array(run.steps) / 1e6
    raw_color = "#94a3b8"
    line_color = "#2563eb"

    ax.scatter(steps_m, run.wr, s=10, color=raw_color, alpha=0.55,
               label="Per-checkpoint eval (200 games)", zorder=2)
    smoothed = smooth(run.wr, window=7)
    ax.plot(steps_m, smoothed, color=line_color, linewidth=2.4,
            label="Smoothed (7-checkpoint avg)", zorder=3)

    ax.axhline(50, color="#64748b", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(steps_m[-1], 51, "50% (coin flip)", ha="right", va="bottom",
            fontsize=9, color="#64748b")

    if run.activated_at is not None:
        x = run.activated_at / 1e6
        ax.axvline(x, color="#dc2626", linestyle="--", linewidth=1.2, alpha=0.75)
        ax.text(x + 0.3, 12, "self-play activated\n(85% gate)", color="#dc2626",
                fontsize=9, va="bottom")

    peak = max(run.wr)
    peak_step = run.steps[run.wr.index(peak)] / 1e6
    ax.annotate(f"peak {peak}%", xy=(peak_step, peak),
                xytext=(peak_step + 1.5, peak + 4),
                fontsize=10, color="#0f172a",
                arrowprops={"arrowstyle": "->", "color": "#0f172a", "lw": 1})

    ax.set_xlabel("Training timesteps (millions)")
    ax.set_ylabel("Win rate vs heuristic (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max(steps_m) * 1.03)
    ax.set_title("Duelist Zero — training curve (Phase C architecture, 16 parallel envs)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_iterations(runs: list[Run], out_path: Path) -> None:
    """Bar chart of peak win rate per iteration with what changed."""
    runs = [r for r in runs if r.steps]
    labels = [r.label.split("  ", 1)[1] for r in runs]
    peaks = [max(r.wr) for r in runs]
    finals = [int(np.mean(r.wr[-10:])) for r in runs]
    colors = [r.color for r in runs]

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    y = np.arange(len(runs))
    bar_h = 0.38
    ax.barh(y - bar_h / 2, peaks, bar_h, color=colors, alpha=0.95, label="Peak win rate")
    ax.barh(y + bar_h / 2, finals, bar_h, color=colors, alpha=0.45,
            label="Late-training avg (last 10 evals)")
    for i, (p, f) in enumerate(zip(peaks, finals)):
        ax.text(p + 1, i - bar_h / 2, f"{p}%", va="center", fontsize=9, color="#0f172a")
        ax.text(f + 1, i + bar_h / 2, f"{f}%", va="center", fontsize=9, color="#475569")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Win rate vs heuristic (%)")
    ax.set_xlim(0, 105)
    ax.axvline(50, color="#64748b", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_title("Architecture iterations — peak vs late-training performance",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    docs = repo / "docs"
    docs.mkdir(exist_ok=True)

    runs = [
        Run("v1  two-stream MLP + 32d embeddings",     repo / "training.log",              "#94a3b8"),
        Run("v2  cross-attention action head",         repo / "training_crossattn_v1.log", "#fb923c"),
        Run("v3  + PBRS reward shaping",               repo / "training_pbrs_v1.log",      "#a3e635"),
        Run("v4  + Phase B (64d embeds, rich history)", repo / "training_arch_v3.log",     "#22d3ee"),
        Run("v5  + tuned hyperparams (no B3 critic)",  repo / "training_ablation_v4.log",  "#a855f7"),
        Run("v6  + Phase C structured semantics",      repo / "training_phase_c.log",      "#2563eb"),
    ]

    for run in runs:
        if run.path.exists():
            run.steps, run.wr, run.activated_at = parse(run.path)
            print(f"{run.label:50s} {len(run.steps):4d} evals  "
                  f"peak {max(run.wr) if run.wr else 0:3d}%  "
                  f"final {run.wr[-1] if run.wr else 0:3d}%")
        else:
            print(f"[skip] {run.path} missing")

    setup_style()

    final = next(r for r in runs if "Phase C" in r.label)
    plot_final(final, docs / "training_curve.png")
    plot_iterations(runs, docs / "architecture_iterations.png")
    print(f"\nWrote {docs/'training_curve.png'}")
    print(f"Wrote {docs/'architecture_iterations.png'}")


if __name__ == "__main__":
    main()
