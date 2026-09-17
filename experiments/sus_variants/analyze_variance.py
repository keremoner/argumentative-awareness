"""Variance calibration for the exact state-only null variance.

Key test: under the null, the empirical Var[score^(t)] across sims should equal
the mean of the theoretical sigma^2,(t) reported by the score function, i.e. a
ratio of ~1.0 at every round.

Also checks the running-mean variance Var[Sus(t)] against
mean(sigma_bar^2) / t, which is the threshold's denominator.

There is no longer a naive/corrected pair: sus_1 returns one exact variance
computed as a finite sum over the utterance space, so there is a single ratio
per score.
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


def compute_per_round(traj: pd.DataFrame) -> pd.DataFrame:
    null = traj[traj["psi"] == "inf"]
    grp = null.groupby(["score_type", "round"], observed=True)
    emp = grp["score"].var(ddof=1).rename("emp_var_score")
    theo = grp["variance"].mean().rename("theo_var")
    df = pd.concat([emp, theo], axis=1).reset_index()
    df["ratio"] = df["emp_var_score"] / df["theo_var"]
    return df


def compute_running(traj: pd.DataFrame) -> pd.DataFrame:
    null = traj[traj["psi"] == "inf"].copy()
    null["sb_sq"] = null["running_sigma"] ** 2
    grp = null.groupby(["score_type", "round"], observed=True)
    emp = grp["running_mean"].var(ddof=1).rename("emp_var_sus")
    msb = grp["sb_sq"].mean().rename("mean_sb_sq")
    df = pd.concat([emp, msb], axis=1).reset_index()
    df["theo_var_sus"] = df["mean_sb_sq"] / df["round"]
    df["ratio"] = df["emp_var_sus"] / df["theo_var_sus"]
    return df


def _plot_ratio(df: pd.DataFrame, out_path: str, ylabel: str, title: str):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for score in ALL_SCORES:
        sub = df[df["score_type"] == score].sort_values("round")
        if sub.empty:
            continue
        ax.plot(sub["round"], sub["ratio"], color=SCORE_COLORS[score],
                lw=1.6, label=score)
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("round t")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def aggregate_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Summarize the calibration ratio per score."""
    agg = (df.groupby("score_type", observed=True)["ratio"]
             .agg(["mean", "median"]).reset_index())
    agg.columns = ["score_type", "ratio_mean", "ratio_median"]
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

    _plot_ratio(per_round, os.path.join(fig_dir, "per_round_ratio.png"),
                "Var_emp[score] / mean(sigma^2)",
                "Per-round variance calibration (exact variance)")
    _plot_ratio(running, os.path.join(fig_dir, "running_ratio.png"),
                "Var_emp[Sus(t)] / (mean(sigma_bar^2)/t)",
                "Running-mean variance calibration (exact variance)")

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
