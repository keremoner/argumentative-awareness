"""
sus-variants diagnostic sims: same design as Group 1's null + alt sweep,
but with five scores running as passive observers (surp2, sus_1, sus_3,
sus_4, sus_4b) and both the naive and corrected variance logged per round.

Usage
-----
    python experiments/sus_variants/run_sims.py           # full: 200 sims, 150 rounds
    python experiments/sus_variants/run_sims.py --smoke   # quick smoke test
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    SUS_VARIANT_FNS,
)


# score_name -> score_fn. surp2 uses the legacy 2-tuple signature (corrected == naive).
SCORE_FNS = {
    "surp2": compute_surp2,
    "sus_1": SUS_VARIANT_FNS["1"],
    "sus_3": SUS_VARIANT_FNS["3"],
    "sus_4": SUS_VARIANT_FNS["4"],
    "sus_4b": SUS_VARIANT_FNS["4b"],
}
SCORE_NAMES = list(SCORE_FNS.keys())


@dataclass
class Config:
    theta_space: list = field(default_factory=lambda: make_thetas(0.1, True, True))
    psis: list = field(default_factory=lambda: ["inf", "high", "low"])
    null_thetas: tuple = (0.1, 0.3, 0.5, 0.7, 0.9)
    alt_thetas: tuple = (0.5,)
    alt_psis: tuple = ("high", "low")
    alpha: float = 3.0
    c: float = 2.0
    rounds: int = 150
    n_sims: int = 200
    seed: int = 0
    n: int = 1
    m: int = 7
    out_dir: str = "results/sus_variants"
    workers: int = 0


def _seed_for(psi, theta, sim_idx, base):
    return (base + hash((psi, round(theta, 6), sim_idx))) % (2**31)


def run_single(task):
    psi, theta_true, sim_idx, cfg_dict = task
    cfg = Config(**cfg_dict)

    rng_seed = _seed_for(psi, theta_true, sim_idx, cfg.seed)
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    world = make_world(theta_true, n=cfg.n, m=cfg.m)
    semantics = make_semantics(n=cfg.n)
    s0 = Speaker0(cfg.theta_space, semantics=semantics, world=world)
    l0 = Listener0(cfg.theta_space, s0, semantics=semantics, world=world)
    s1 = Speaker1(cfg.theta_space, l0, semantics=semantics, world=world,
                  alpha=cfg.alpha, psi=psi)

    tests = [SequentialTest(SCORE_FNS[name], name, c=cfg.c,
                            switch_enabled=False) for name in SCORE_NAMES]
    listener = DetectionListener(cfg.theta_space, cfg.psis, s1, world, semantics,
                                 tests=tests, alpha=cfg.alpha)

    for _ in range(cfg.rounds):
        obs = world.sample_obs()
        utt = s1.sample_utterance(obs)
        listener.update(utt)
        s1.update(obs)
        l0.update(utt)
        s0.update(obs)

    rows = []
    tau_summary = {"psi": psi, "theta_star": theta_true, "sim_id": sim_idx}
    for t in tests:
        h = t.history
        for i in range(len(h["scores"])):
            rows.append({
                "sim_id": sim_idx,
                "psi": psi,
                "theta_star": theta_true,
                "round": i + 1,
                "score_type": t.name,
                "score": float(h["scores"][i]),
                "variance_naive": float(h["variances"][i]),
                "variance_corrected": float(h["variances_corrected"][i]),
                "running_mean": float(h["running_mean"][i]),
                "running_sigma_naive": float(h["running_sigma"][i]),
                "running_sigma_corrected": float(h["running_sigma_corrected"][i]),
                "threshold_naive": float(h["threshold"][i]),
                "threshold_corrected": float(h["threshold_corrected"][i]),
                "crossed_naive": bool(h["crossed"][i]),
                "crossed_corrected": bool(h["crossed_corrected"][i]),
            })
        tau_summary[f"tau_naive_{t.name}"] = (
            -1 if t.tau_naive is None else int(t.tau_naive))
        tau_summary[f"tau_corrected_{t.name}"] = (
            -1 if t.tau_corrected is None else int(t.tau_corrected))
    return rows, tau_summary


def build_task_list(cfg: Config):
    tasks = []
    cfg_dict = {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in cfg.__dict__.items()}
    for th in cfg.null_thetas:
        for sim in range(cfg.n_sims):
            tasks.append(("inf", float(th), sim, cfg_dict))
    for psi in cfg.alt_psis:
        for th in cfg.alt_thetas:
            for sim in range(cfg.n_sims):
                tasks.append((psi, float(th), sim, cfg_dict))
    return tasks


def run_sweep(cfg: Config):
    tasks = build_task_list(cfg)
    n_workers = cfg.workers if cfg.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    print(f"launching {len(tasks)} sims across {n_workers} workers", flush=True)

    all_rows = []
    tau_rows = []
    t0 = time.time()
    done = 0
    total = len(tasks)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(run_single, task) for task in tasks]
        for fut in as_completed(futures):
            rows, tau_summary = fut.result()
            all_rows.extend(rows)
            tau_rows.append(tau_summary)
            done += 1
            if done % max(1, total // 20) == 0 or done == total:
                el = time.time() - t0
                rate = done / el if el > 0 else 0
                eta = (total - done) / rate if rate > 0 else float("inf")
                print(f"  {done}/{total} ({100*done/total:.0f}%) "
                      f"elapsed {el:.1f}s eta {eta:.1f}s", flush=True)
    return all_rows, tau_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n_sims", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.smoke:
        cfg.n_sims = 10
        cfg.rounds = 30
        cfg.out_dir = "results/sus_variants_smoke"
    if args.n_sims is not None:
        cfg.n_sims = args.n_sims
    if args.rounds is not None:
        cfg.rounds = args.rounds
    if args.workers:
        cfg.workers = args.workers
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir

    os.makedirs(cfg.out_dir, exist_ok=True)
    print(f"cfg: rounds={cfg.rounds} n_sims={cfg.n_sims} "
          f"null_thetas={list(cfg.null_thetas)} alt_psis={list(cfg.alt_psis)} "
          f"alt_thetas={list(cfg.alt_thetas)} scores={SCORE_NAMES}")

    t0 = time.time()
    rows, tau_rows = run_sweep(cfg)
    print(f"sweep done in {time.time()-t0:.1f}s", flush=True)

    traj_path = os.path.join(cfg.out_dir, "raw_trajectories.parquet")
    tau_path = os.path.join(cfg.out_dir, "tau_summary.parquet")
    meta_path = os.path.join(cfg.out_dir, "run_config.json")

    df = pd.DataFrame(rows)
    df["psi"] = df["psi"].astype("category")
    df["score_type"] = df["score_type"].astype("category")
    df.to_parquet(traj_path, index=False)

    tau_df = pd.DataFrame(tau_rows)
    tau_df["psi"] = tau_df["psi"].astype("category")
    tau_df.to_parquet(tau_path, index=False)

    import json
    with open(meta_path, "w", encoding="utf-8") as fh:
        serialisable = {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in cfg.__dict__.items()}
        json.dump(serialisable, fh, indent=2)

    print(f"wrote {traj_path} ({len(df):,} rows)")
    print(f"wrote {tau_path} ({len(tau_df):,} rows)")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
