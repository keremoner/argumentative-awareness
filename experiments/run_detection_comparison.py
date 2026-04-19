"""
Compare the three detection scores (surp2, surp1, sus) as passive observers.

All three tests run in parallel on *identical* utterance streams, so any
difference in trajectories, detection latency, or false-alarm rate is
attributable purely to the score design -- not to simulation noise.

Usage
-----
    python experiments/run_detection_comparison.py --mode pilot
    python experiments/run_detection_comparison.py --mode full

The pilot mode is meant to be fast (~1-2 min) and is used for iterating on
sanity checks before committing to the full sweep.
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt

# Ensure `rsa` is importable when run as a script from anywhere
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rsa.setup import make_thetas, make_world, make_semantics
from rsa.speaker0 import Speaker0
from rsa.listener0 import Listener0
from rsa.speaker1 import Speaker1
from rsa.detection import (
    DetectionListener,
    SequentialTest,
    compute_surp2,
    compute_surp1,
    compute_sus,
)


SCORE_NAMES = ["surp2", "surp1", "sus"]
SCORE_FNS = {"surp2": compute_surp2, "surp1": compute_surp1, "sus": compute_sus}
DEFAULT_PSIS = ["inf", "high", "low"]


@dataclass
class Config:
    theta_space: list = field(default_factory=lambda: make_thetas(0.1, True, True))
    psis: list = field(default_factory=lambda: list(DEFAULT_PSIS))
    test_thetas: list = field(default_factory=lambda: [0.3, 0.5, 0.7])
    speaker_types: list = field(default_factory=lambda: ["inf", "high", "low"])
    alpha: float = 3.0
    c: float = 2.0
    rounds: int = 50
    n_sims: int = 20
    seed: int = 0
    n: int = 1
    m: int = 7
    out_dir: str = "results/detection_comparison"
    tag: str = "pilot"


def run_single(cfg: Config, speaker_type: str, theta_true: float, sim_idx: int):
    """Run a single simulation with all three tests as passive observers.

    Returns a dict with per-round trajectories for each score.
    """
    rng_seed = cfg.seed + hash((speaker_type, theta_true, sim_idx)) % (2**31)
    random.seed(rng_seed)
    np.random.seed(rng_seed % (2**31))

    world = make_world(theta_true, n=cfg.n, m=cfg.m)
    semantics = make_semantics(n=cfg.n)
    s0 = Speaker0(cfg.theta_space, semantics=semantics, world=world)
    l0 = Listener0(cfg.theta_space, s0, semantics=semantics, world=world)
    s1 = Speaker1(cfg.theta_space, l0, semantics=semantics, world=world,
                  alpha=cfg.alpha, psi=speaker_type)

    tests = [SequentialTest(SCORE_FNS[name], name, c=cfg.c,
                            switch_enabled=False)
             for name in SCORE_NAMES]
    listener = DetectionListener(cfg.theta_space, cfg.psis, s1, world, semantics,
                                 tests=tests, alpha=cfg.alpha)

    utts = []
    obs_hist = []
    for _ in range(cfg.rounds):
        obs = world.sample_obs()
        utt = s1.sample_utterance(obs)
        listener.update(utt)
        s1.update(obs)
        l0.update(utt)
        s0.update(obs)
        utts.append(utt)
        obs_hist.append(obs)

    return {
        "speaker_type": speaker_type,
        "theta_true": theta_true,
        "sim_idx": sim_idx,
        "utt_history": utts,
        "tests": {
            t.name: {
                "per_round": np.asarray(t.history["scores"], dtype=float),
                "running_mean": np.asarray(t.history["running_mean"], dtype=float),
                "running_sigma": np.asarray(t.history["running_sigma"], dtype=float),
                "threshold": np.asarray(t.history["threshold"], dtype=float),
                "crossed": np.asarray(t.history["crossed"], dtype=bool),
                "tau": t.tau,
            }
            for t in tests
        },
    }


def run_sweep(cfg: Config):
    conditions = list(product(cfg.speaker_types, cfg.test_thetas))
    all_results = {}
    t0 = time.time()
    total = len(conditions) * cfg.n_sims
    done = 0
    for psi, theta in conditions:
        key = (psi, theta)
        runs = []
        for sim_idx in range(cfg.n_sims):
            runs.append(run_single(cfg, psi, theta, sim_idx))
            done += 1
            if done % max(1, total // 20) == 0:
                elapsed = time.time() - t0
                print(f"  progress: {done}/{total} ({100*done/total:.0f}%) "
                      f"elapsed {elapsed:.1f}s", flush=True)
        all_results[key] = runs
    return all_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _stack_running(results_for_condition, score_name):
    """(n_sims, rounds) array of running means."""
    return np.stack([r["tests"][score_name]["running_mean"] for r in results_for_condition])


def _stack_threshold(results_for_condition, score_name):
    return np.stack([r["tests"][score_name]["threshold"] for r in results_for_condition])


def plot_trajectories(all_results, cfg: Config, path):
    conds = [(psi, th) for psi in cfg.speaker_types for th in cfg.test_thetas]
    nrows, ncols = len(cfg.speaker_types), len(cfg.test_thetas)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows),
                              sharex=True, squeeze=False)
    colors = {"surp2": "tab:blue", "surp1": "tab:orange", "sus": "tab:green"}
    ts = np.arange(1, cfg.rounds + 1)
    for i, psi in enumerate(cfg.speaker_types):
        for j, th in enumerate(cfg.test_thetas):
            ax = axes[i][j]
            runs = all_results[(psi, th)]
            for name in SCORE_NAMES:
                mat = _stack_running(runs, name)
                mean = mat.mean(axis=0)
                lo = np.quantile(mat, 0.025, axis=0)
                hi = np.quantile(mat, 0.975, axis=0)
                ax.plot(ts, mean, color=colors[name], label=name, lw=1.6)
                ax.fill_between(ts, lo, hi, color=colors[name], alpha=0.18)
                thr = _stack_threshold(runs, name).mean(axis=0)
                ax.plot(ts, thr, color=colors[name], lw=0.8, ls="--", alpha=0.7)
            ax.axhline(0, color="k", lw=0.5, alpha=0.5)
            ax.set_title(f"ψ={psi}, θ={th}")
            if j == 0:
                ax.set_ylabel("Sus(t)")
            if i == nrows - 1:
                ax.set_xlabel("round t")
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Sus(t) trajectories (mean ± 95% CI, dashed = mean threshold)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_detection_latency(all_results, cfg: Config, path):
    fig, axes = plt.subplots(1, len(cfg.speaker_types),
                              figsize=(4.0 * len(cfg.speaker_types), 3.2),
                              sharey=True, squeeze=False)
    axes = axes[0]
    for ax, psi in zip(axes, cfg.speaker_types):
        data = []
        labels = []
        for name in SCORE_NAMES:
            taus = []
            for th in cfg.test_thetas:
                for r in all_results[(psi, th)]:
                    tau = r["tests"][name]["tau"]
                    taus.append(tau if tau is not None else cfg.rounds + 1)
            data.append(taus)
            labels.append(name)
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.axhline(cfg.rounds + 1, color="r", ls=":", lw=0.8, label="no-cross")
        ax.set_title(f"ψ={psi}")
        ax.set_ylabel("first-crossing round τ")
        ax.set_ylim(0, cfg.rounds + 2)
    fig.suptitle("Detection latency (across test θ, n_sims per condition)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_false_alarm(all_results, cfg: Config, path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ts = np.arange(1, cfg.rounds + 1)
    for name in SCORE_NAMES:
        crossed_mat = []
        for th in cfg.test_thetas:
            runs = all_results[("inf", th)]
            for r in runs:
                crossed = r["tests"][name]["crossed"]
                # "has crossed by round t" = cumulative OR
                ever = np.maximum.accumulate(crossed.astype(int)).astype(bool)
                crossed_mat.append(ever)
        crossed_mat = np.stack(crossed_mat)
        ax.plot(ts, crossed_mat.mean(axis=0), label=name, lw=1.6)
    ax.set_xlabel("round t")
    ax.set_ylabel("P(any cross by t | ψ=inf)")
    ax.set_title(f"False-alarm rate (c={cfg.c})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_score_correlation(all_results, cfg: Config, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    pairs = [("surp2", "sus"), ("surp2", "surp1")]
    for ax, (a, b) in zip(axes, pairs):
        xs, ys = [], []
        for key, runs in all_results.items():
            for r in runs:
                xs.append(r["tests"][a]["per_round"])
                ys.append(r["tests"][b]["per_round"])
        xs = np.concatenate(xs)
        ys = np.concatenate(ys)
        ax.scatter(xs, ys, s=3, alpha=0.25)
        lim = [min(xs.min(), ys.min()), max(xs.max(), ys.max())]
        ax.plot(lim, lim, "r--", lw=0.8)
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        corr = np.corrcoef(xs, ys)[0, 1]
        ax.set_title(f"{a} vs {b}  (Pearson ρ = {corr:.3f})")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report (text summary)
# ---------------------------------------------------------------------------

def make_text_report(all_results, cfg: Config):
    lines = []
    lines.append(f"=== Detection comparison ({cfg.tag}) ===")
    lines.append(f"theta space: {cfg.theta_space}")
    lines.append(f"test_thetas: {cfg.test_thetas}")
    lines.append(f"speaker types: {cfg.speaker_types}")
    lines.append(f"psi: {cfg.psis}   alpha: {cfg.alpha}   c: {cfg.c}")
    lines.append(f"rounds: {cfg.rounds}   n_sims per cond: {cfg.n_sims}")
    lines.append("")

    for psi in cfg.speaker_types:
        lines.append(f"-- speaker ψ={psi}")
        header = f"  {'θ':>5} " + " ".join(f"{'cross%_'+n:>14}" for n in SCORE_NAMES) \
                 + " | " + " ".join(f"{'median_τ_'+n:>14}" for n in SCORE_NAMES)
        lines.append(header)
        for th in cfg.test_thetas:
            runs = all_results[(psi, th)]
            row = [f"  {th:>5}"]
            cross_vals, tau_vals = [], []
            for name in SCORE_NAMES:
                taus = [r["tests"][name]["tau"] for r in runs]
                crossed = [t is not None for t in taus]
                pct = 100.0 * sum(crossed) / len(crossed)
                finite = [t for t in taus if t is not None]
                med = float(np.median(finite)) if finite else float("nan")
                cross_vals.append(pct)
                tau_vals.append(med)
            row += [f"{v:>14.1f}" for v in cross_vals]
            row += ["|"] + [f"{v:>14.1f}" for v in tau_vals]
            lines.append(" ".join(row))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "full", "custom"], default="pilot")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--n_sims", type=int, default=None)
    parser.add_argument("--c", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.mode == "pilot":
        cfg.rounds = 50
        cfg.n_sims = 20
        cfg.test_thetas = [0.3, 0.5, 0.7]
        cfg.tag = "pilot"
    elif args.mode == "full":
        cfg.rounds = 150
        cfg.n_sims = 100
        cfg.test_thetas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        cfg.tag = "full"

    # per-arg overrides (useful for ad-hoc runs / tuning)
    if args.rounds is not None:  cfg.rounds = args.rounds
    if args.n_sims is not None:  cfg.n_sims = args.n_sims
    if args.c is not None:       cfg.c = args.c
    if args.alpha is not None:   cfg.alpha = args.alpha
    if args.tag is not None:     cfg.tag = args.tag

    os.makedirs(cfg.out_dir, exist_ok=True)
    print(f"[{cfg.tag}] theta_space={cfg.theta_space}")
    print(f"[{cfg.tag}] test_thetas={cfg.test_thetas}, speakers={cfg.speaker_types}")
    print(f"[{cfg.tag}] rounds={cfg.rounds}, n_sims={cfg.n_sims}, "
          f"alpha={cfg.alpha}, c={cfg.c}")

    t0 = time.time()
    results = run_sweep(cfg)
    elapsed = time.time() - t0
    print(f"[{cfg.tag}] sweep finished in {elapsed:.1f}s")

    # Save raw
    pkl_path = os.path.join(cfg.out_dir, f"{cfg.tag}_results.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump({"cfg": cfg.__dict__, "results": results}, fh)
    print(f"[{cfg.tag}] saved {pkl_path}")

    # Plots
    plot_trajectories(results, cfg, os.path.join(cfg.out_dir, f"{cfg.tag}_trajectories.png"))
    plot_detection_latency(results, cfg, os.path.join(cfg.out_dir, f"{cfg.tag}_latency.png"))
    plot_false_alarm(results, cfg, os.path.join(cfg.out_dir, f"{cfg.tag}_false_alarm.png"))
    plot_score_correlation(results, cfg, os.path.join(cfg.out_dir, f"{cfg.tag}_correlation.png"))

    # Text report
    report = make_text_report(results, cfg)
    report_path = os.path.join(cfg.out_dir, f"{cfg.tag}_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report)
    print(f"[{cfg.tag}] wrote report to {report_path}")


if __name__ == "__main__":
    main()
