"""Variance calibration: naive vs corrected formulas.

Key test: under the null, empirical Var[sus_v^(t)] should equal the mean
of sigma^2_corrected^(t) much more closely than the mean of
sigma^2_naive^(t). Ratios should move from ~0.89 (naive, as in Group 1)
toward ~1.0 (corrected).

Also checks the running-mean variance Var[Sus_v(t)] vs
mean(sigma_bar_corrected^2) / t (the threshold's denominator).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ALL_SCORES, SUS_VARIANTS, SCORE_COLORS, VARIANCE_KINDS, load, ensure_dir


def compute_per_round(traj: pd.DataFrame) -> pd.DataFrame:
    null = traj[traj["psi"] == "inf"]
    grp = null.groupby(["score_type", "round"], observed=True)
    emp = grp["score"].var(ddof=1).rename("emp_var_score")
    theo_n = grp["variance_naive"].mean().rename("theo_var_naive")
    theo_c = grp["variance_corrected"].mean().rename("theo_var_corrected")
    df = pd.concat([emp, theo_n, theo_c], axis=1).reset_index()
    df["ratio_naive"] = df["emp_var_score"] / df["theo_var_naive"]
    df["ratio_corrected"] = df["emp_var_score"] / df["theo_var_corrected"]
    return df


def compute_running(traj: pd.DataFrame) -> pd.DataFrame:
    null = traj[traj["psi"] == "inf"].copy()
    null["sb_sq_naive"] = null["running_sigma_naive"] ** 2
    null["sb_sq_corrected"] = null["running_sigma_corrected"] ** 2
    grp = null.groupby(["score_type", "round"], observed=True)
    emp = grp["running_mean"].var(ddof=1).rename("emp_var_sus")
    mn = grp["sb_sq_naive"].mean().rename("mean_sb_sq_naive")
    mc = grp["sb_sq_corrected"].mean().rename("mean_sb_sq_corrected")
    df = pd.concat([emp, mn, mc], axis=1).reset_index()
    df["theo_var_sus_naive"] = df["mean_sb_sq_naive"] / df["round"]
    df["theo_var_sus_corrected"] = df["mean_sb_sq_corrected"] / df["round"]
    df["ratio_naive"] = df["emp_var_sus"] / df["theo_var_sus_naive"]
    df["ratio_corrected"] = df["emp_var_sus"] / df["theo_var_sus_corrected"]
    return df


def plot_per_round_ratio(df: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, kind in zip(axes, VARIANCE_KINDS):
        col = f"ratio_{kind}"
        for score in ALL_SCORES:
            sub = df[df["score_type"] == score].sort_values("round")
            ax.plot(sub["round"], sub[col], color=SCORE_COLORS[score],
                    lw=1.6, label=score)
        ax.axhline(1.0, color="k", ls=":", lw=0.8)
        ax.set_xlabel("round t")
        ax.set_ylabel("Var_emp[score] / mean(sigma^2)")
        ax.set_title(f"per-round variance ratio ({kind})")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Per-round variance calibration: naive vs corrected")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_running_ratio(df: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, kind in zip(axes, VARIANCE_KINDS):
        col = f"ratio_{kind}"
        for score in ALL_SCORES:
            sub = df[df["score_type"] == score].sort_values("round")
            ax.plot(sub["round"], sub[col], color=SCORE_COLORS[score],
                    lw=1.6, label=score)
        ax.axhline(1.0, color="k", ls=":", lw=0.8)
        ax.set_xlabel("round t")
        ax.set_ylabel("Var_emp[Sus(t)] / (mean(sigma_bar^2)/t)")
        ax.set_title(f"running-mean variance ratio ({kind})")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Running-mean variance calibration: naive vs corrected")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def aggregate_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Summarize ratio_naive and ratio_corrected per score."""
    agg = (df.groupby("score_type", observed=True)
             [["ratio_naive", "ratio_corrected"]]
             .agg(["mean", "median"]).reset_index())
    agg.columns = [c if isinstance(c, str) else "_".join(c)
                   for c in agg.columns]
    agg["level"] = label
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="results/sus_variants")
    args = ap.parse_args()

    traj, _, _ = load(args.in_dir)
    out_dir = ensure_dir(os.path.join(args.in_dir, "variance"))
    fig_dir = ensure_dir(os.path.join(out_dir, "figures"))

    per_round = compute_per_round(traj)
    per_round.to_csv(os.path.join(out_dir, "per_round_variance.csv"), index=False)

    running = compute_running(traj)
    running.to_csv(os.path.join(out_dir, "running_variance.csv"), index=False)

    plot_per_round_ratio(per_round, os.path.join(fig_dir, "per_round_ratio.png"))
    plot_running_ratio(running, os.path.join(fig_dir, "running_ratio.png"))

    agg_pr = aggregate_table(per_round, "per_round")
    agg_rn = aggregate_table(running, "running")
    agg = pd.concat([agg_pr, agg_rn], ignore_index=True)
    agg.to_csv(os.path.join(out_dir, "ratio_agg.csv"), index=False)

    print("\n=== Per-round variance ratio (empirical / theoretical) ===")
    print(agg_pr.to_string(index=False))
    print("\n=== Running-mean Var[Sus(t)] ratio ===")
    print(agg_rn.to_string(index=False))
    print(f"\nwrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
