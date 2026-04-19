"""
Group 1 diagnostic sims: run the null sweep (psi=inf across thetas) plus the
alt conditions needed for 1.2's TPR check (psi=high, psi=low at theta=0.5).

All three detection scores (surp2, surp1, sus) are attached as passive
observers; switching is disabled. Per-round trajectories are saved to a
parquet file with one row per (sim_id, round, score_type) so downstream
analysis scripts can slice freely.

Usage
-----
    python experiments/group1/run_sims.py            # full: 200 sims, 150 rounds
    python experiments/group1/run_sims.py --smoke    # quick smoke test
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
    compute_surp1,
    compute_sus,
)


SCORE_NAMES = ["surp2", "surp1", "sus"]
SCORE_FNS = {"surp2": compute_surp2, "surp1": compute_surp1, "sus": compute_sus}


@dataclass
class Config:
    theta_space: list = field(default_factory=lambda: make_thetas(0.1, True, True))
    psis: list = field(default_factory=lambda: ["inf", "high", "low"])
    # Null sweep: psi=inf across all of these.
    null_thetas: tuple = (0.1, 0.3, 0.5, 0.7, 0.9)
    # Alt sweep for 1.2 TPR: psi in {high, low} at theta=0.5.
    alt_thetas: tuple = (0.5,)
    alt_psis: tuple = ("high", "low")
    alpha: float = 3.0
    c: float = 2.0
    rounds: int = 150
    n_sims: int = 200
    seed: int = 0
    n: int = 1
    m: int = 7
    out_dir: str = "results/group1"
    workers: int = 0  # 0 = auto (os.cpu_count())


def _seed_for(psi: str, theta: float, sim_idx: int, base: int) -> int:
    return (base + hash((psi, round(theta, 6), sim_idx))) % (2**31)


def run_single(task):
    """Worker: run one simulation. Returns a list of row dicts (one per
    (round, score)) plus a tau-summary dict.
    """
    (psi, theta_true, sim_idx, cfg_dict) = task
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

    tests = [SequentialTest(SCORE_FNS[name], name, c=cfg.c, switch_enabled=False)
             for name in SCORE_NAMES]
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
        scores = h["scores"]
        for i in range(len(scores)):
            rows.append({
                "sim_id": sim_idx,
                "psi": psi,
                "theta_star": theta_true,
                "round": i + 1,
                "score_type": t.name,
                "score": float(scores[i]),
                "variance": float(h["variances"][i]),
                "running_mean": float(h["running_mean"][i]),
                "running_sigma": float(h["running_sigma"][i]),
                "threshold": float(h["threshold"][i]),
                "crossed": bool(h["crossed"][i]),
            })
        tau_summary[f"tau_{t.name}"] = (-1 if t.tau is None else int(t.tau))
    return rows, tau_summary


def build_task_list(cfg: Config):
    tasks = []
    cfg_dict = {k: v for k, v in cfg.__dict__.items()}
    # Convert tuples to lists so dataclass round-trips cleanly.
    for k, v in list(cfg_dict.items()):
        if isinstance(v, tuple):
            cfg_dict[k] = list(v)
    # Null: psi=inf across null_thetas.
    for th in cfg.null_thetas:
        for sim in range(cfg.n_sims):
            tasks.append(("inf", float(th), sim, cfg_dict))
    # Alt: pers+ / pers- at alt_thetas.
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
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else float("inf")
                print(f"  {done}/{total} ({100*done/total:.0f}%) "
                      f"elapsed {elapsed:.1f}s eta {eta:.1f}s", flush=True)
    return all_rows, tau_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run (10 sims x 30 rounds) to validate the pipeline")
    parser.add_argument("--n_sims", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.smoke:
        cfg.n_sims = 10
        cfg.rounds = 30
        cfg.out_dir = "results/group1_smoke"
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
          f"alt_thetas={list(cfg.alt_thetas)} alpha={cfg.alpha} c={cfg.c} "
          f"n={cfg.n} m={cfg.m}")

    t0 = time.time()
    rows, tau_rows = run_sweep(cfg)
    elapsed = time.time() - t0
    print(f"sweep done in {elapsed:.1f}s", flush=True)

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
