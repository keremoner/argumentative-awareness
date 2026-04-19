"""Variant-to-variant comparison:
  - Sus(t) trajectory across variants (mean +/- band per score, psi=inf).
  - Per-round score correlation between variants (within-sim, pooled).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ALL_SCORES, SUS_VARIANTS, SCORE_COLORS, load, ensure_dir


def plot_trajectories(traj: pd.DataFrame, out_path: str):
    null = traj[traj["psi"] == "inf"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for score in ALL_SCORES:
        sub = null[null["score_type"] == score]
        g = sub.groupby("round")["running_mean"]
        mean = g.mean()
        lo = g.quantile(0.025)
        hi = g.quantile(0.975)
        ts = mean.index.to_numpy()
        ax.plot(ts, mean.values, color=SCORE_COLORS[score], lw=1.8, label=score)
        ax.fill_between(ts, lo.values, hi.values,
                        color=SCORE_COLORS[score], alpha=0.12)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("round t")
    ax.set_ylabel("Sus(t) (running mean)")
    ax.set_title("Sus(t) trajectory across variants, psi=inf (mean +/- 95% band)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def compute_correlation(traj: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between per-round scores across (variant, variant)
    pairs. Pool all (sim, round, theta, psi) rows.
    """
    wide = traj.pivot_table(
        index=["psi", "theta_star", "sim_id", "round"],
        columns="score_type", values="score", observed=True,
    )
    corr = wide[ALL_SCORES].corr(method="pearson")
    return corr


def plot_correlation(corr: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(ALL_SCORES)))
    ax.set_yticks(range(len(ALL_SCORES)))
    ax.set_xticklabels(ALL_SCORES, rotation=45, ha="right")
    ax.set_yticklabels(ALL_SCORES)
    for i in range(len(ALL_SCORES)):
        for j in range(len(ALL_SCORES)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center",
                    color="black" if abs(corr.iloc[i, j]) < 0.5 else "white",
                    fontsize=8)
    ax.set_title("Pearson correlation of per-round scores\n(pooled across all conditions)")
    fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_pairwise_scatter(traj: pd.DataFrame, out_path: str):
    """Scatter of sus_1 vs each other sus variant, and vs surp2."""
    wide = traj.pivot_table(
        index=["psi", "theta_star", "sim_id", "round"],
        columns="score_type", values="score", observed=True,
    ).reset_index()
    others = [s for s in ALL_SCORES if s != "sus_1"]
    fig, axes = plt.subplots(1, len(others), figsize=(4.0 * len(others), 4.0))
    if len(others) == 1:
        axes = [axes]
    # Subsample for plot clarity
    sample = wide.sample(n=min(len(wide), 20000), random_state=0)
    for ax, other in zip(axes, others):
        ax.scatter(sample["sus_1"], sample[other], s=2, alpha=0.2,
                   color=SCORE_COLORS[other])
        lim = [min(sample["sus_1"].min(), sample[other].min()),
               max(sample["sus_1"].max(), sample[other].max())]
        ax.plot(lim, lim, "r--", lw=0.8)
        corr = float(sample[["sus_1", other]].corr().iloc[0, 1])
        ax.set_xlabel("sus_1")
        ax.set_ylabel(other)
        ax.set_title(f"sus_1 vs {other}  (rho={corr:.3f})")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="results/sus_variants")
    args = ap.parse_args()

    traj, _, _ = load(args.in_dir)
    out_dir = ensure_dir(os.path.join(args.in_dir, "comparison"))
    fig_dir = ensure_dir(os.path.join(out_dir, "figures"))

    plot_trajectories(traj, os.path.join(fig_dir, "trajectories.png"))

    corr = compute_correlation(traj)
    corr.to_csv(os.path.join(out_dir, "score_correlation.csv"))
    print("Pairwise score correlation (pooled):")
    print(corr.round(3).to_string())

    plot_correlation(corr, os.path.join(fig_dir, "correlation_matrix.png"))
    plot_pairwise_scatter(traj, os.path.join(fig_dir, "pairwise_scatter.png"))
    print(f"\nwrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
