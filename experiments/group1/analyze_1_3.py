"""Experiment 1.3: Per-round variance calibration (theoretical vs empirical).

For each round t and score, compare:
  - Mean of theoretical per-round variance sigma^2,(t) across sims.
  - Empirical variance of per-round score score^(t) across sims.

Ratio ~= 1 means the formula is correct. Ratio > 1 means the formula
underestimates (would inflate FPR, since the test believes the score is
less variable than it actually is).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import PRESENTED_SCORES, SCORE_COLORS, load, ensure_dir


def compute_var_table(traj: pd.DataFrame) -> pd.DataFrame:
    null = traj[traj["psi"] == "inf"]
    # Empirical variance of per-round score across all (sim_id, theta_star).
    grp = null.groupby(["score_type", "round"], observed=True)
    emp = grp["score"].var(ddof=1).rename("emp_var")
    theo = grp["variance"].mean().rename("theo_var_mean")
    n = grp["score"].size().rename("n")
    out = pd.concat([emp, theo, n], axis=1).reset_index()
    out["ratio_emp_over_theo"] = out["emp_var"] / out["theo_var_mean"]
    return out


def plot_variance(df: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(1, len(PRESENTED_SCORES),
                             figsize=(5.5 * len(PRESENTED_SCORES), 4.0))
    if len(PRESENTED_SCORES) == 1:
        axes = [axes]
    for ax, score in zip(axes, PRESENTED_SCORES):
        sub = df[df["score_type"] == score].sort_values("round")
        ts = sub["round"].to_numpy()
        ax.plot(ts, sub["theo_var_mean"], color=SCORE_COLORS[score],
                ls="--", lw=1.5, label="theoretical mean sigma^2,(t)")
        ax.plot(ts, sub["emp_var"], color=SCORE_COLORS[score],
                lw=1.8, label="empirical Var[score^(t)]")
        ax.set_title(f"{score}")
        ax.set_xlabel("round t")
        ax.set_ylabel("variance")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("Per-round variance: theoretical vs empirical (psi=inf)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_ratio(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4.0))
    for score in PRESENTED_SCORES:
        sub = df[df["score_type"] == score].sort_values("round")
        ax.plot(sub["round"], sub["ratio_emp_over_theo"],
                color=SCORE_COLORS[score], lw=1.6, label=score)
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("round t")
    ax.set_ylabel("ratio  Var_emp / mean(sigma^2_theo)")
    ax.set_title("Variance calibration ratio (1.0 = correct)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="results/group1")
    args = ap.parse_args()

    traj, _, _ = load(args.in_dir)
    out_dir = ensure_dir(os.path.join(args.in_dir, "exp_1.3"))
    fig_dir = ensure_dir(os.path.join(out_dir, "figures"))

    df = compute_var_table(traj)
    stats_path = os.path.join(out_dir, "variance_table.csv")
    df.to_csv(stats_path, index=False)
    print(f"wrote {stats_path}")

    # Quick aggregate print.
    agg = (df.groupby("score_type", observed=True)["ratio_emp_over_theo"]
             .agg(["mean", "median", "min", "max"]).reset_index())
    print("Ratio (empirical / theoretical) across rounds:")
    print(agg.to_string(index=False))

    plot_variance(df, os.path.join(fig_dir, "per_round_variance.png"))
    plot_ratio(df, os.path.join(fig_dir, "variance_ratio.png"))
    print(f"wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
