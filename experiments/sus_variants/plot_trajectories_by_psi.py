"""Per-score Sus(t) trajectory plots, separated by speaker psi.

For each of the 5 scores, one panel showing Sus(t) mean +/- 95% band for
psi=inf (null, pooled over theta_star), psi=high (theta*=0.5), and
psi=low (theta*=0.5). Mean naive and corrected thresholds are overlaid
so the crossing story is visible at a glance.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ALL_SCORES, load, ensure_dir


PSI_CONDITIONS = [
    # (label, psi, theta filter or None)
    ("psi=inf (null, pooled over theta*)", "inf", None),
    ("psi=high (theta*=0.5)", "high", 0.5),
    ("psi=low  (theta*=0.5)", "low", 0.5),
]
PSI_COLORS = {"inf": "tab:blue", "high": "tab:red", "low": "tab:orange"}


def _subset(traj, score, psi, theta):
    m = (traj["score_type"] == score) & (traj["psi"] == psi)
    if theta is not None:
        m &= np.isclose(traj["theta_star"], theta)
    return traj[m]


def plot_grid(traj, out_path, rounds):
    fig, axes = plt.subplots(len(ALL_SCORES), 1,
                             figsize=(9, 2.6 * len(ALL_SCORES)),
                             sharex=True)
    if len(ALL_SCORES) == 1:
        axes = [axes]
    ts = np.arange(1, rounds + 1)

    for ax, score in zip(axes, ALL_SCORES):
        for label, psi, theta in PSI_CONDITIONS:
            sub = _subset(traj, score, psi, theta)
            if len(sub) == 0:
                continue
            g = sub.groupby("round")["running_mean"]
            mean = g.mean().reindex(ts).to_numpy()
            lo = g.quantile(0.025).reindex(ts).to_numpy()
            hi = g.quantile(0.975).reindex(ts).to_numpy()
            ax.plot(ts, mean, color=PSI_COLORS[psi], lw=1.8, label=label)
            ax.fill_between(ts, lo, hi, color=PSI_COLORS[psi], alpha=0.15)

        # Null-condition thresholds (mean across psi=inf sims, pooled over theta).
        null = traj[(traj["score_type"] == score) & (traj["psi"] == "inf")]
        thr_n = null.groupby("round")["threshold_naive"].mean().reindex(ts).to_numpy()
        thr_c = null.groupby("round")["threshold_corrected"].mean().reindex(ts).to_numpy()
        ax.plot(ts, thr_n, color="black", lw=1.0, ls="--",
                label="mean threshold (naive, null)")
        ax.plot(ts, thr_c, color="gray", lw=1.0, ls=":",
                label="mean threshold (corrected, null)")

        ax.axhline(0, color="gray", lw=0.4)
        ax.set_ylabel(f"{score}\nSus(t)")
        ax.grid(alpha=0.3)
        # Auto-clip y to avoid the pers+/pers- signal compressing the null band.
        # Keep threshold and null band visible by not clipping too tight.
        null_mean = null.groupby("round")["running_mean"].mean().reindex(ts)
        ymax_candidates = [
            np.nanmax(thr_n) * 3.0 if np.isfinite(np.nanmax(thr_n)) else 0,
            float(null_mean.abs().max() * 4),
        ]
        # If alt trajectories are huge, still show them but cap at a
        # reasonable multiple of the alt means.
        for label, psi, theta in PSI_CONDITIONS:
            if psi == "inf":
                continue
            sub = _subset(traj, score, psi, theta)
            if len(sub) == 0:
                continue
            alt_mean_end = float(sub.groupby("round")["running_mean"]
                                    .mean().iloc[-1])
            ymax_candidates.append(alt_mean_end * 1.3)
        ymax = max(ymax_candidates)
        if np.isfinite(ymax) and ymax > 0:
            ax.set_ylim(-0.05 * ymax, ymax)

    axes[0].legend(fontsize=7, loc="upper left", ncol=1, framealpha=0.9)
    axes[-1].set_xlabel("round t")
    fig.suptitle("Sus(t) by score x speaker  (mean +/- 95% band, 200 sims per condition)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_null_by_theta(traj, out_path, rounds):
    """For each score, Sus(t) under psi=inf stratified by theta_star."""
    theta_colors = plt.cm.viridis(np.linspace(0.05, 0.95, 5))
    null_thetas = sorted(traj.loc[traj["psi"] == "inf", "theta_star"].unique())
    fig, axes = plt.subplots(len(ALL_SCORES), 1,
                             figsize=(9, 2.4 * len(ALL_SCORES)),
                             sharex=True)
    if len(ALL_SCORES) == 1:
        axes = [axes]
    ts = np.arange(1, rounds + 1)
    for ax, score in zip(axes, ALL_SCORES):
        for th, col in zip(null_thetas, theta_colors):
            sub = traj[(traj["score_type"] == score) &
                       (traj["psi"] == "inf") &
                       (np.isclose(traj["theta_star"], th))]
            g = sub.groupby("round")["running_mean"]
            mean = g.mean().reindex(ts).to_numpy()
            lo = g.quantile(0.025).reindex(ts).to_numpy()
            hi = g.quantile(0.975).reindex(ts).to_numpy()
            ax.plot(ts, mean, color=col, lw=1.6, label=f"theta*={th}")
            ax.fill_between(ts, lo, hi, color=col, alpha=0.12)
        null = traj[(traj["score_type"] == score) & (traj["psi"] == "inf")]
        thr_n = null.groupby("round")["threshold_naive"].mean().reindex(ts).to_numpy()
        ax.plot(ts, thr_n, color="black", lw=0.9, ls="--",
                label="mean threshold (naive)")
        ax.axhline(0, color="gray", lw=0.4)
        ax.set_ylabel(f"{score}\nSus(t)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.9)
    axes[-1].set_xlabel("round t")
    fig.suptitle("Null-only Sus(t) by theta*  (psi=inf, 200 sims per theta)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="results/sus_variants")
    args = ap.parse_args()
    traj, _, cfg = load(args.in_dir)
    rounds = cfg["rounds"]

    out_dir = ensure_dir(os.path.join(args.in_dir, "comparison", "figures"))

    plot_grid(traj, os.path.join(out_dir, "trajectories_by_psi.png"), rounds)
    plot_null_by_theta(traj, os.path.join(out_dir, "trajectories_null_by_theta.png"), rounds)
    print(f"wrote {out_dir}/trajectories_by_psi.png")
    print(f"wrote {out_dir}/trajectories_null_by_theta.png")


if __name__ == "__main__":
    main()
