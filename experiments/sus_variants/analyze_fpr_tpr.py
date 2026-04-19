"""FPR/TPR comparison across variants x variance kinds.

Reports for each (score, variance_kind, t_warmup):
  - FPR under the null (psi=inf, pooled over theta_star).
  - FPR per theta_star.
  - TPR under persuasive speakers (theta_star=0.5, psi in {high, low}).

Produces:
  - summary_stats.csv (long-format)
  - figures/fpr_per_theta.png  -- null FPR by theta for each (score, kind)
  - figures/fpr_tpr_bars.png   -- FPR (pooled) vs TPR (theta=0.5) bars
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
    ALL_SCORES, NULL_THETAS, VARIANCE_KINDS, SCORE_COLORS,
    SUS_VARIANTS, load, ensure_dir, per_sim_crossed,
)


WARMUPS = [0, 5, 10]


def compute_rates(traj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kind in VARIANCE_KINDS:
        for w in WARMUPS:
            cr = per_sim_crossed(traj, kind, w)
            rates = (cr.groupby(["psi", "theta_star", "score_type"],
                                observed=True)["ever"].mean()
                     .reset_index(name="rate"))
            rates["variance_kind"] = kind
            rates["t_warmup"] = w
            rows.append(rates)
    return pd.concat(rows, ignore_index=True)


def plot_fpr_per_theta(rates: pd.DataFrame, out_path: str):
    """One panel per variance kind; within each, FPR vs theta_star for each score."""
    fpr = rates[(rates["psi"] == "inf") & (rates["t_warmup"] == 0)]
    fig, axes = plt.subplots(1, len(VARIANCE_KINDS),
                             figsize=(6.0 * len(VARIANCE_KINDS), 4.0),
                             sharey=True)
    for ax, kind in zip(axes, VARIANCE_KINDS):
        sub = fpr[fpr["variance_kind"] == kind]
        for score in ALL_SCORES:
            s = sub[sub["score_type"] == score].sort_values("theta_star")
            ax.plot(s["theta_star"], s["rate"], marker="o",
                    color=SCORE_COLORS[score], lw=1.6, label=score)
        ax.axhline(0.05, color="red", ls=":", lw=0.7, label="5% ref")
        ax.axhline(0.023, color="red", ls="--", lw=0.7, alpha=0.6,
                   label="2.3% (nominal c=2)")
        ax.set_xlabel("theta_star")
        ax.set_title(f"FPR under null  (variance = {kind})")
        ax.set_ylim(0, None)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("FPR (cross within 150 rounds)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Null FPR by theta_star x variance kind")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_fpr_tpr_bars(rates: pd.DataFrame, out_path: str):
    """Grid: rows = variance kind, cols = metric (FPR, TPR_high, TPR_low).
    Bars within each panel: one per t_warmup, grouped by score."""
    # Aggregate: FPR pooled over theta for psi=inf; TPR at theta=0.5 for high/low.
    fpr_pool = (rates[rates["psi"] == "inf"]
                .groupby(["variance_kind", "t_warmup", "score_type"],
                         observed=True)["rate"].mean().reset_index())
    tpr = rates[(rates["psi"].isin(["high", "low"])) &
                (np.isclose(rates["theta_star"], 0.5))].copy()

    fig, axes = plt.subplots(len(VARIANCE_KINDS), 3,
                             figsize=(14.5, 3.8 * len(VARIANCE_KINDS)),
                             sharey="row")
    if len(VARIANCE_KINDS) == 1:
        axes = axes[None, :]
    x = np.arange(len(WARMUPS))
    width = 0.8 / len(ALL_SCORES)

    for ri, kind in enumerate(VARIANCE_KINDS):
        panels = [
            ("FPR (pooled over theta*)",
             fpr_pool[fpr_pool["variance_kind"] == kind]),
            ("TPR (psi=high, theta*=0.5)",
             tpr[(tpr["variance_kind"] == kind) & (tpr["psi"] == "high")]),
            ("TPR (psi=low, theta*=0.5)",
             tpr[(tpr["variance_kind"] == kind) & (tpr["psi"] == "low")]),
        ]
        for ci, (title, data) in enumerate(panels):
            ax = axes[ri][ci]
            for i, score in enumerate(ALL_SCORES):
                vals = []
                for w in WARMUPS:
                    row = data[(data["t_warmup"] == w) &
                               (data["score_type"] == score)]
                    vals.append(float(row["rate"].iloc[0]) if len(row) else 0.0)
                ax.bar(x + i * width - 0.4 + width / 2, vals, width=width,
                       label=score, color=SCORE_COLORS[score],
                       edgecolor="black", linewidth=0.3, alpha=0.85)
                for xi, v in zip(x + i * width - 0.4 + width / 2, vals):
                    ax.text(xi, v + 0.01, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=6, rotation=90)
            ax.set_xticks(x)
            ax.set_xticklabels([f"tw={w}" for w in WARMUPS])
            ax.set_title(f"{title}  ({kind})")
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.3)
            if ci == 0:
                ax.set_ylabel("rate")
        axes[ri][0].axhline(0.05, color="red", lw=0.6, ls=":")
    axes[0][0].legend(fontsize=7, loc="upper right", ncol=2)
    fig.suptitle("Per-variant FPR vs TPR (across warm-ups and variance kinds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_fpr_tpr_scatter(rates: pd.DataFrame, out_path: str):
    """Scatter: FPR (x) vs TPR (y) for each (score, variance_kind) at t_warmup=0
    and t_warmup=10. TPR averaged over {high, low} at theta=0.5.
    Useful for spotting dominated variants.
    """
    fpr = (rates[(rates["psi"] == "inf") & (rates["t_warmup"].isin([0, 10]))]
           .groupby(["variance_kind", "t_warmup", "score_type"], observed=True)
           ["rate"].mean().reset_index(name="fpr"))
    tpr = (rates[(rates["psi"].isin(["high", "low"])) &
                 (np.isclose(rates["theta_star"], 0.5)) &
                 (rates["t_warmup"].isin([0, 10]))]
           .groupby(["variance_kind", "t_warmup", "score_type"], observed=True)
           ["rate"].mean().reset_index(name="tpr"))
    merged = fpr.merge(tpr, on=["variance_kind", "t_warmup", "score_type"])

    fig, axes = plt.subplots(1, len(VARIANCE_KINDS),
                             figsize=(5.5 * len(VARIANCE_KINDS), 4.2),
                             sharey=True)
    for ax, kind in zip(axes, VARIANCE_KINDS):
        sub = merged[merged["variance_kind"] == kind]
        for score in ALL_SCORES:
            ss = sub[sub["score_type"] == score]
            for _, row in ss.iterrows():
                marker = "o" if row["t_warmup"] == 0 else "s"
                ax.scatter(row["fpr"], row["tpr"], s=90 if marker == "o" else 70,
                           color=SCORE_COLORS[score], marker=marker,
                           edgecolor="black", linewidth=0.5, alpha=0.85)
            # Connect the two t_warmup points with a thin line to show movement.
            if len(ss) == 2:
                ss_sorted = ss.sort_values("t_warmup")
                ax.plot(ss_sorted["fpr"], ss_sorted["tpr"],
                        color=SCORE_COLORS[score], lw=0.8, alpha=0.5)
            ax.annotate(score, (ss["fpr"].mean(), ss["tpr"].mean()),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.axvline(0.05, color="red", lw=0.6, ls=":", alpha=0.6)
        ax.set_xlabel("FPR (psi=inf, pooled over theta*)")
        ax.set_ylabel("mean TPR (theta*=0.5, psi in high/low)")
        ax.set_title(f"variance = {kind}\n(circle=t_warmup 0, square=t_warmup 10)")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, None)
        ax.set_ylim(0, 1.02)
    fig.suptitle("FPR vs TPR trade-off across variants")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="results/sus_variants")
    args = ap.parse_args()

    traj, tau_df, cfg = load(args.in_dir)
    out_dir = ensure_dir(os.path.join(args.in_dir, "fpr_tpr"))
    fig_dir = ensure_dir(os.path.join(out_dir, "figures"))

    rates = compute_rates(traj)
    rates.to_csv(os.path.join(out_dir, "summary_stats.csv"), index=False)
    print(f"wrote {out_dir}/summary_stats.csv  ({len(rates)} rows)")

    plot_fpr_per_theta(rates, os.path.join(fig_dir, "fpr_per_theta.png"))
    plot_fpr_tpr_bars(rates, os.path.join(fig_dir, "fpr_tpr_bars.png"))
    plot_fpr_tpr_scatter(rates, os.path.join(fig_dir, "fpr_tpr_scatter.png"))
    print(f"wrote figures to {fig_dir}")

    # Pretty-print summary at t_warmup=0 for quick scanning.
    print("\n=== t_warmup=0 summary (psi=inf pooled, theta*=0.5 high/low) ===")
    for kind in VARIANCE_KINDS:
        print(f"\n[variance = {kind}]")
        print(f"  {'score':>8}  {'FPR':>6}  {'TPR_high':>9}  {'TPR_low':>9}")
        for score in ALL_SCORES:
            r = rates[(rates["t_warmup"] == 0) &
                      (rates["variance_kind"] == kind) &
                      (rates["score_type"] == score)]
            fpr = r[r["psi"] == "inf"]["rate"].mean()
            tpr_h = r[(r["psi"] == "high") &
                      (np.isclose(r["theta_star"], 0.5))]["rate"].mean()
            tpr_l = r[(r["psi"] == "low") &
                      (np.isclose(r["theta_star"], 0.5))]["rate"].mean()
            print(f"  {score:>8}  {fpr:>6.3f}  {tpr_h:>9.3f}  {tpr_l:>9.3f}")


if __name__ == "__main__":
    main()
