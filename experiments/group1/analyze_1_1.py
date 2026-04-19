"""Experiment 1.1: FPR diagnostic under the null (psi=inf).

Characterizes WHEN false crossings happen. Produces:
  - summary_stats.csv: per (theta_star, score) FPR and tau distribution stats.
  - figures/hist_tau.png: histogram of tau per score, pooled over theta.
  - figures/violin_tau.png: violin of tau per score per theta.
  - figures/sus_trajectory.png: mean Sus(t) +/- 95% band vs mean threshold.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (
    ALL_SCORES, PRESENTED_SCORES, NULL_THETAS, SCORE_COLORS, load, ensure_dir,
)


def summary_stats(traj: pd.DataFrame, tau_df: pd.DataFrame, rounds: int):
    # Restrict to null condition.
    tau_null = tau_df[tau_df["psi"] == "inf"].copy()

    rows = []
    for theta in sorted(tau_null["theta_star"].unique()):
        for score in ALL_SCORES:
            col = f"tau_{score}"
            taus = tau_null.loc[tau_null["theta_star"] == theta, col].to_numpy()
            crossed_mask = taus > 0  # tau=-1 means never crossed
            fpr = float(crossed_mask.mean())
            finite = taus[crossed_mask]
            rows.append({
                "theta_star": theta,
                "score": score,
                "n_sims": int(len(taus)),
                "fpr": fpr,
                "tau_median": float(np.median(finite)) if finite.size else np.nan,
                "tau_p05": float(np.quantile(finite, 0.05)) if finite.size else np.nan,
                "tau_p25": float(np.quantile(finite, 0.25)) if finite.size else np.nan,
                "tau_p75": float(np.quantile(finite, 0.75)) if finite.size else np.nan,
                "tau_p95": float(np.quantile(finite, 0.95)) if finite.size else np.nan,
                "tau_frac_le_5": float((finite <= 5).mean()) if finite.size else np.nan,
                "tau_frac_le_10": float((finite <= 10).mean()) if finite.size else np.nan,
            })
    return pd.DataFrame(rows)


def plot_hist_tau(tau_df: pd.DataFrame, out_path: str):
    tau_null = tau_df[tau_df["psi"] == "inf"]
    fig, axes = plt.subplots(1, len(PRESENTED_SCORES),
                             figsize=(5.5 * len(PRESENTED_SCORES), 3.8),
                             sharey=False)
    if len(PRESENTED_SCORES) == 1:
        axes = [axes]
    for ax, score in zip(axes, PRESENTED_SCORES):
        col = f"tau_{score}"
        taus = tau_null[col].to_numpy()
        finite = taus[taus > 0]
        if finite.size == 0:
            ax.set_title(f"{score}: no crossings")
            ax.set_xlabel("first-crossing round tau")
            continue
        # Fine bins for rounds 1..20, coarser after.
        bins_fine = np.arange(1, 21)
        bins_coarse = np.arange(20, int(finite.max()) + 11, 10)
        bins = np.concatenate([bins_fine, bins_coarse])
        ax.hist(finite, bins=bins, color=SCORE_COLORS[score], alpha=0.85,
                edgecolor="black", linewidth=0.3)
        ax.set_title(f"{score}  (n crossed = {finite.size} / {taus.size}, "
                     f"FPR = {finite.size/taus.size:.2%})")
        ax.set_xlabel("first-crossing round tau")
        ax.set_ylabel("count")
        ax.axvline(5, color="red", ls=":", lw=0.8, label="t=5")
        ax.axvline(10, color="orange", ls=":", lw=0.8, label="t=10")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Distribution of first-crossing round (psi=inf, pooled over theta*)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_violin_tau(tau_df: pd.DataFrame, out_path: str, rounds: int):
    tau_null = tau_df[tau_df["psi"] == "inf"]
    fig, axes = plt.subplots(1, len(PRESENTED_SCORES),
                             figsize=(5.0 * len(PRESENTED_SCORES), 4.0),
                             sharey=True)
    if len(PRESENTED_SCORES) == 1:
        axes = [axes]
    for ax, score in zip(axes, PRESENTED_SCORES):
        col = f"tau_{score}"
        data = []
        labels = []
        for theta in NULL_THETAS:
            taus = tau_null.loc[tau_null["theta_star"] == theta, col].to_numpy()
            finite = taus[taus > 0]
            # Replace non-crossings with rounds+1 so the violin reflects the
            # fact that many sims never cross; label them below the plot.
            full = np.concatenate([finite, np.full((taus.size - finite.size,),
                                                   rounds + 1)])
            data.append(full)
            labels.append(f"{theta}\nFPR={finite.size/taus.size:.0%}")
        parts = ax.violinplot(data, showmeans=False, showmedians=True,
                              widths=0.8)
        for body in parts["bodies"]:
            body.set_facecolor(SCORE_COLORS[score])
            body.set_alpha(0.55)
        ax.set_xticks(range(1, len(NULL_THETAS) + 1))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("theta_star")
        ax.set_ylabel("first-crossing round tau (rounds+1 = never)")
        ax.set_title(score)
        ax.axhline(rounds + 1, color="k", ls=":", lw=0.6)
    fig.suptitle("tau by theta_star (psi=inf)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_sus_trajectory(traj: pd.DataFrame, out_path: str):
    null = traj[traj["psi"] == "inf"]
    fig, axes = plt.subplots(1, len(PRESENTED_SCORES),
                             figsize=(5.5 * len(PRESENTED_SCORES), 3.8),
                             sharex=True)
    if len(PRESENTED_SCORES) == 1:
        axes = [axes]
    for ax, score in zip(axes, PRESENTED_SCORES):
        sub = null[null["score_type"] == score]
        # Pivot to (sim_id, theta_star) x round, averaging only over sims at the
        # same theta, then pool.
        pivot_mean = sub.groupby("round")["running_mean"].agg(["mean",
            lambda s: s.quantile(0.025), lambda s: s.quantile(0.975)])
        pivot_mean.columns = ["mean", "lo", "hi"]
        thr = sub.groupby("round")["threshold"].mean()
        ts = pivot_mean.index.to_numpy()
        ax.plot(ts, pivot_mean["mean"], color=SCORE_COLORS[score], lw=1.8,
                label=f"Sus(t) mean")
        ax.fill_between(ts, pivot_mean["lo"], pivot_mean["hi"],
                        color=SCORE_COLORS[score], alpha=0.22, label="95% band")
        ax.plot(ts, thr, color="black", ls="--", lw=1.0,
                label="mean threshold")
        ax.plot(ts, -thr, color="black", ls=":", lw=0.7, alpha=0.7,
                label="-threshold")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(score)
        ax.set_xlabel("round t")
        ax.set_ylabel("Sus(t)")
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Running mean trajectory vs threshold (psi=inf, pooled)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="results/group1")
    args = ap.parse_args()

    traj, tau_df, cfg = load(args.in_dir)
    rounds = cfg["rounds"]

    out_dir = ensure_dir(os.path.join(args.in_dir, "exp_1.1"))
    fig_dir = ensure_dir(os.path.join(out_dir, "figures"))

    summary = summary_stats(traj, tau_df, rounds)
    summary_path = os.path.join(out_dir, "summary_stats.csv")
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")
    print(summary.to_string(index=False))

    plot_hist_tau(tau_df, os.path.join(fig_dir, "hist_tau.png"))
    plot_violin_tau(tau_df, os.path.join(fig_dir, "violin_tau.png"), rounds)
    plot_sus_trajectory(traj, os.path.join(fig_dir, "sus_trajectory.png"))
    print(f"wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
