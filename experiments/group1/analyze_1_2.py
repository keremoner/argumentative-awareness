"""Experiment 1.2: Warm-up fix evaluation.

Post-hoc re-evaluation of the crossing rule: only flag when round >= t_warmup+1.
Computes FPR (psi=inf, pooled across theta) and TPR (psi in {high, low} at
theta=0.5) for each warmup value. Bar chart plots both.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ALL_SCORES, PRESENTED_SCORES, SCORE_COLORS, load, ensure_dir


WARMUPS = [0, 3, 5, 10]


def crossing_mask_after_warmup(traj: pd.DataFrame, t_warmup: int) -> pd.Series:
    """Per (sim_id, psi, theta_star, score_type): True if ever crosses at
    round > t_warmup."""
    sub = traj[traj["round"] > t_warmup]
    # any crossed within the valid window
    return sub.groupby(["psi", "theta_star", "sim_id", "score_type"],
                       observed=True)["crossed"].any()


def compute_rates(traj: pd.DataFrame, t_warmup: int) -> pd.DataFrame:
    """Fraction of sims that ever cross at t > t_warmup, by (psi, theta, score)."""
    crossed = crossing_mask_after_warmup(traj, t_warmup).reset_index()
    crossed.columns = ["psi", "theta_star", "sim_id", "score_type", "ever"]
    rate = (crossed.groupby(["psi", "theta_star", "score_type"], observed=True)
                  ["ever"].mean().reset_index(name="rate"))
    rate["t_warmup"] = t_warmup
    return rate


def plot_fpr_tpr(rates: pd.DataFrame, out_path: str):
    # FPR: psi=inf, pooled across theta (mean of per-theta rates).
    fpr = (rates[rates["psi"] == "inf"]
           .groupby(["t_warmup", "score_type"], observed=True)["rate"]
           .mean().reset_index())

    # TPR at theta=0.5 for high/low.
    tpr = rates[(rates["psi"].isin(["high", "low"])) &
                (np.isclose(rates["theta_star"], 0.5))]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    titles = ["FPR  (psi=inf, pooled over theta*)",
              "TPR  (psi=high, theta*=0.5)",
              "TPR  (psi=low, theta*=0.5)"]
    datasets = [
        fpr.assign(psi="inf"),
        tpr[tpr["psi"] == "high"],
        tpr[tpr["psi"] == "low"],
    ]
    width = 0.8 / max(1, len(ALL_SCORES))
    x = np.arange(len(WARMUPS))
    for ax, title, data in zip(axes, titles, datasets):
        for i, score in enumerate(ALL_SCORES):
            vals = []
            for w in WARMUPS:
                row = data[(data["t_warmup"] == w) & (data["score_type"] == score)]
                vals.append(float(row["rate"].iloc[0]) if len(row) else 0.0)
            ax.bar(x + i * width - 0.4 + width / 2, vals, width=width,
                   label=score, color=SCORE_COLORS[score],
                   alpha=0.85, edgecolor="black", linewidth=0.3)
            for xi, v in zip(x + i * width - 0.4 + width / 2, vals):
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"t_warmup={w}" for w in WARMUPS])
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0.05, color="red", lw=0.6, ls=":", label="5% ref")
    axes[0].set_ylabel("rate")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Warm-up fix: FPR vs TPR across t_warmup")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="results/group1")
    args = ap.parse_args()

    traj, tau_df, cfg = load(args.in_dir)
    out_dir = ensure_dir(os.path.join(args.in_dir, "exp_1.2"))
    fig_dir = ensure_dir(os.path.join(out_dir, "figures"))

    all_rates = pd.concat([compute_rates(traj, w) for w in WARMUPS],
                          ignore_index=True)
    summary_path = os.path.join(out_dir, "summary_stats.csv")
    all_rates.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")
    print(all_rates.to_string(index=False))

    plot_fpr_tpr(all_rates, os.path.join(fig_dir, "fpr_tpr_warmup.png"))
    print(f"wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
