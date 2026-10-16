"""Success-rate curves over training from the evaluations in ../results (10 seeds x 25 trials per checkpoint; line =
mean of the per-seed success rates, band = +-1 std across seeds).

  <task>_curves.png        Haptic-ACT (F/T), ACT, Diffusion Policy and BC on one task
  act_vs_haptic_act.png    ACT and Haptic-ACT (F/T) on the four tasks, one panel each
Usage: python plot_results.py [--out ../results]"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
METHODS = [("haptic-act", "Haptic-ACT (F/T)", "#d62728"), ("act", "ACT", "#1f77b4"),
           ("dp", "Diffusion Policy (50 DDPM steps)", "#2ca02c"), ("bc", "BC (single-step ACT)", "#7f7f7f")]
TASKS = [("peg", "PegInsertionSide (contact-rich)"), ("pushcube", "PushCube"), ("stackcube", "StackCube"), ("pickcube", "PickCube")]


def curve(task: str, method: str):
    runs = sorted((int(p.stem), json.loads(p.read_text())) for p in (RESULTS / task / method).glob("*.json"))
    x = np.array([step for step, _ in runs]) / 1000
    return x, np.array([r["mean"] for _, r in runs]), np.array([r["std_population"] for _, r in runs])


def draw(ax, task: str, methods):
    for method, label, color in methods:
        x, mean, std = curve(task, method)
        ax.plot(x, mean, "-o", color=color, lw=2, ms=3, label=label, zorder=4 if method == "haptic-act" else 3)
        ax.fill_between(x, np.clip(mean - std, 0, 100), np.clip(mean + std, 0, 100), color=color, alpha=0.15, lw=0)
    ax.set_ylim(-2, 102)
    ax.grid(alpha=0.3)
    ax.set_xlabel("training step (k)")
    ax.set_ylabel("success rate (%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for task, title in TASKS:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        draw(ax, task, METHODS)
        ax.set_title(f"{title}: 10 seeds x 25 trials per checkpoint (mean ± std)")
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(args.out / f"{task}_curves.png", dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (task, title) in zip(axes.flat, TASKS):
        draw(ax, task, [m for m in METHODS if m[0] in ("act", "haptic-act")][::-1])
        ax.set_title(title)
    axes[0, 0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Effect of the force input: ACT vs Haptic-ACT (F/T), 10 seeds x 25 trials per checkpoint")
    fig.tight_layout()
    fig.savefig(args.out / "act_vs_haptic_act.png", dpi=150)
    print("wrote", sorted(p.name for p in args.out.glob("*.png")))


if __name__ == "__main__":
    main()
