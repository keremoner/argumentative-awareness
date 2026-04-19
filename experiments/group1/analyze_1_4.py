"""Experiment 1.4: Running-mean variance scaling check.

Under the null with iid per-round scores of variance sigma^2, Var[Sus(t)]
should scale as sigma^2 / t. We verify this empirically and compare to the
model-implied scaling sigma_bar^2,(t) / t (what the threshold formula uses).
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


def compute_scaling(traj: pd.DataFrame) -> pd.DataFrame:
    null = traj[traj["psi"] == "inf"]
    grp = null.groupby(["score_type", "round"], observed=True)
    # Empirical variance of running_mean across sims (pooled across theta_star).
    emp = grp["running_mean"].var(ddof=1).rename("emp_var_sus")
    # Model-implied: mean(sigma_bar^2) / t, where sigma_bar is running_sigma.
    # We compute mean(sigma_bar^2) per round, then divide by t.
    sigma_sq = (null.assign(sb2=null["running_sigma"] ** 2)
                     .groupby(["score_type", "round"], observed=True)["sb2"]
                     .mean().rename("mean_sigma_bar_sq"))
    out = pd.concat([emp, sigma_sq], axis=1).reset_index()
    out["theo_var_sus"] = out["mean_sigma_bar_sq"] / out["round"]
    out["ratio_emp_over_theo"] = out["emp_var_sus"] / out["theo_var_sus"]
    return out


def plot_scaling(df: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(1, len(PRESENTED_SCORES),
                             figsize=(5.5 * len(PRESENTED_SCORES), 4.2))
    if len(PRESENTED_SCORES) == 1:
        axes = [axes]
    for ax, score in zip(axes, PRESENTED_SCORES):
        sub = df[df["score_type"] == score].sort_values("round")
        ts = sub["round"].to_numpy()
        emp = sub["emp_var_sus"].to_numpy()
        theo = sub["theo_var_sus"].to_numpy()

        ax.loglog(ts, emp, color=SCORE_COLORS[score], lw=1.8,
                  label="empirical Var[Sus(t)]")
        ax.loglog(ts, theo, color=SCORE_COLORS[score], ls="--", lw=1.2,
                  label="theo: mean(sigma_bar^2)/t")

        # Reference: 1/t slope anchored at t=2 of empirical.
        if emp.size > 1 and np.isfinite(emp[1]) and emp[1] > 0:
            anchor = emp[1] * 2
            ref_1_t = anchor / ts
            ax.loglog(ts, ref_1_t, color="k", ls=":", lw=0.9, label="1/t ref")
            ref_1_sqrt_t = (emp[1] * np.sqrt(2)) / np.sqrt(ts)
            ax.loglog(ts, ref_1_sqrt_t, color="gray", ls=":", lw=0.9,
                      label="1/sqrt(t) ref")

        ax.set_title(score)
        ax.set_xlabel("round t")
        ax.set_ylabel("Var[Sus(t)]")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("Running-mean variance scaling (psi=inf)")
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
    ax.set_ylabel("empirical / theoretical Var[Sus(t)]")
    ax.set_title("Running-mean variance ratio")
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
    out_dir = ensure_dir(os.path.join(args.in_dir, "exp_1.4"))
    fig_dir = ensure_dir(os.path.join(out_dir, "figures"))

    df = compute_scaling(traj)
    stats_path = os.path.join(out_dir, "scaling_table.csv")
    df.to_csv(stats_path, index=False)
    print(f"wrote {stats_path}")

    agg = (df.groupby("score_type", observed=True)["ratio_emp_over_theo"]
             .agg(["mean", "median", "min", "max"]).reset_index())
    print("Sus(t) variance ratio (empirical / theoretical):")
    print(agg.to_string(index=False))

    plot_scaling(df, os.path.join(fig_dir, "sus_variance_scaling.png"))
    plot_ratio(df, os.path.join(fig_dir, "sus_variance_ratio.png"))
    print(f"wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
