"""Blog figure: training return + world-model 8-step error over iterations.

Data: supermimic/blog_data/training_series.csv (per-50-iteration log)
      supermimic/blog_data/bench_by_ckpt.csv (4-motion benchmark per checkpoint)
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "blog_data"
OUT = Path("/Users/master/Desktop/github/brianwang2007xjtu.github.io/public/supermimic-demo/training_curve.png")

SMOOTH_K = 40  # ±40 log rows ≈ ±2000 iterations
PAPER_BAND = (0.007, 0.011)


def smooth(ys: list[float], k: int = SMOOTH_K) -> list[float]:
    out = []
    for i in range(len(ys)):
        lo, hi = max(0, i - k), min(len(ys), i + k)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return out


def main() -> None:
    rows = list(csv.DictReader(open(DATA / "training_series.csv")))
    it = [int(r["iteration"]) for r in rows]
    ret = [float(r["train_ret"]) for r in rows]
    n8 = [float(r["wm_nse8"]) for r in rows]

    bench = list(csv.DictReader(open(DATA / "bench_by_ckpt.csv")))
    pairs = []
    for r in bench:
        digits = re.sub(r"\D", "", r["ckpt"])
        if digits:
            pairs.append((int(digits), float(r["mean_norm_return"])))
    pairs.sort()
    b_it = [p[0] for p in pairs]
    b_ret = [p[1] for p in pairs]

    plt.rcParams.update({
        "font.family": "Helvetica Neue",
        "font.size": 11,
        "axes.edgecolor": "#d2d2d7",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#6e6e73",
        "ytick.color": "#6e6e73",
        "axes.labelcolor": "#1d1d1f",
        "text.color": "#1d1d1f",
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), dpi=200)

    ax1.plot(it, ret, color="#8dc8e8", lw=0.7, alpha=0.5)
    ax1.plot(it, smooth(ret), color="#3d7fbf", lw=1.6, label="training return (smoothed)")
    ax1.scatter(b_it, b_ret, s=24, color="#1d1d1f", zorder=3, label="4-motion benchmark")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("normalised return")
    ax1.set_ylim(0, 1.0)
    ax1.legend(frameon=False, fontsize=9, loc="lower right")
    ax1.set_title("policy", loc="left", fontsize=12, pad=8)

    ax2.plot(it, n8, color="#c9a0a0", lw=0.7, alpha=0.5)
    ax2.plot(it, smooth(n8), color="#9f6262", lw=1.6)
    ax2.axhspan(*PAPER_BAND, color="#9f6262", alpha=0.12)
    ax2.annotate(
        "paper 0.007–0.011",
        xy=(3000, PAPER_BAND[1]),
        xytext=(3000, 0.030),
        fontsize=9,
        color="#9f6262",
        arrowprops=dict(arrowstyle="-", color="#9f6262", lw=0.8),
    )
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("8-step open-loop error")
    ax2.set_ylim(0, 0.08)
    ax2.set_title("world model", loc="left", fontsize=12, pad=8)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print("saved", OUT)


if __name__ == "__main__":
    main()
