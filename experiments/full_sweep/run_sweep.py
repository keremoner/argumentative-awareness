"""
Full sweep of core detection experiments -- see full_sweep_spec.md.

Grid: theta_star x psi_true x alpha = 9 x 3 x 11 = 297 cells.
200 simulations per cell, 150 rounds each, so 8,910,000 trajectory rows
in total. Two scores (surp2, sus_1 with corrected variance) run as
passive observers per sim.

Output: partitioned parquet dataset at
    results/full_sweep/trajectories/cell=XXXX/part.parquet
plus simulations.parquet, run_config.json, sanity.md.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
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


THETA_STARS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
PSI_TRUES = ("inf", "pers+", "pers-")       # saved labels
PSI_TO_CODE = {"inf": "inf", "pers+": "high", "pers-": "low"}
ALPHAS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0)


@dataclass
class Config:
    theta_space: list = field(default_factory=lambda: make_thetas(0.1, True, True))
    n: int = 1
    m: int = 7
    speaker_level: str = "S1"
    rounds: int = 150
    n_sims: int = 200
    seed: int = 0
    out_dir: str = "results/full_sweep"
    workers: int = 0
    flush_every_cell: bool = True  # write shards per-cell (spec: incremental saves)


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def build_cell_grid():
    """Return list of dicts, one per cell, with cell_id and axes."""
    cells = []
    cid = 0
    for theta in THETA_STARS:
        for psi in PSI_TRUES:
            for alpha in ALPHAS:
                cells.append({
                    "cell_id": cid,
                    "theta_star": float(theta),
                    "psi_true": psi,
                    "alpha": float(alpha),
                })
                cid += 1
    return cells


def _seed_for(cell_id: int, sim_idx: int, base: int) -> int:
    return (base + hash((cell_id, sim_idx))) % (2**31)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def run_single(task):
    """Worker: returns list of row dicts for one (cell_id, sim_id)."""
    (cell, sim_idx, cfg_dict) = task
    cfg = Config(**cfg_dict)

    cell_id = cell["cell_id"]
    theta_star = cell["theta_star"]
    psi_true = cell["psi_true"]
    psi_code = PSI_TO_CODE[psi_true]
    alpha = cell["alpha"]

    rng_seed = _seed_for(cell_id, sim_idx, cfg.seed)
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    world = make_world(theta_star, n=cfg.n, m=cfg.m)
    semantics = make_semantics(n=cfg.n)
    s0 = Speaker0(cfg.theta_space, semantics=semantics, world=world)
    l0 = Listener0(cfg.theta_space, s0, semantics=semantics, world=world)
    s1 = Speaker1(cfg.theta_space, l0, semantics=semantics, world=world,
                  alpha=alpha, psi=psi_code)

    tests = [
        SequentialTest(compute_surp2, "surp2", c=float("inf"),
                       switch_enabled=False),
        SequentialTest(SUS_VARIANT_FNS["1"], "sus_1", c=float("inf"),
                       switch_enabled=False),
    ]
    listener = DetectionListener(cfg.theta_space, ["inf", "high", "low"], s1,
                                 world, semantics, tests=tests, alpha=alpha)

    n_theta = len(cfg.theta_space)
    rows = []
    for t in range(cfg.rounds):
        # Capture pre-update beliefs (these are what the score context uses).
        l1_arr = listener._l1_theta_array()
        l0_arr = listener._l0_theta_array()

        obs = world.sample_obs()
        utt = s1.sample_utterance(obs)
        listener.update(utt)
        s1.update(obs)
        l0.update(utt)
        s0.update(obs)

        # SequentialTest just recorded this round at index t.
        surp2_t = tests[0].history
        sus_t = tests[1].history

        o_idx = int(world.obs_index(obs))
        o_count = int(np.argmax(obs))
        u_idx = int(semantics.utterance_index(utt))

        row = {
            "cell_id": cell_id,
            "sim_id": sim_idx,
            "round": t + 1,
            "theta_star": theta_star,
            "psi_true": psi_true,
            "alpha": alpha,
            "n": cfg.n,
            "m": cfg.m,
            "speaker_level": cfg.speaker_level,
            "u_observed": u_idx,
            "O_true_idx": o_idx,
            "O_true_count": o_count,
            # scores
            "surp2_score": float(surp2_t["scores"][t]),
            "surp2_sigma2": float(surp2_t["variances"][t]),
            "surp2_Sus": float(surp2_t["running_mean"][t]),
            "surp2_sigma_bar2": float(surp2_t["running_sigma"][t]) ** 2,
            "sus1_score": float(sus_t["scores"][t]),
            "sus1_sigma2_naive": float(sus_t["variances"][t]),
            "sus1_sigma2_corrected": float(sus_t["variances_corrected"][t]),
            "sus1_Sus": float(sus_t["running_mean"][t]),
            "sus1_sigma_bar2_naive": float(sus_t["running_sigma"][t]) ** 2,
            "sus1_sigma_bar2_corrected":
                float(sus_t["running_sigma_corrected"][t]) ** 2,
        }
        for i in range(n_theta):
            row[f"L1_theta_{i}"] = float(l1_arr[i])
            row[f"L0_theta_{i}"] = float(l0_arr[i])
        rows.append(row)

    sim_meta = {
        "cell_id": cell_id,
        "sim_id": sim_idx,
        "theta_star": theta_star,
        "psi_true": psi_true,
        "alpha": alpha,
        "n": cfg.n,
        "m": cfg.m,
        "speaker_level": cfg.speaker_level,
        "seed": rng_seed,
    }
    return cell_id, rows, sim_meta


# ---------------------------------------------------------------------------
# Sweep driver with per-cell incremental writes
# ---------------------------------------------------------------------------

def _shard_path(base_dir: str, cell_id: int) -> str:
    d = os.path.join(base_dir, "trajectories", f"cell={cell_id:04d}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "part.parquet")


def _sims_shard_path(base_dir: str, cell_id: int) -> str:
    d = os.path.join(base_dir, "simulations", f"cell={cell_id:04d}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "part.parquet")


def _completed_cells(base_dir: str) -> set:
    """Return set of cell_ids whose trajectory shard already exists."""
    traj_root = os.path.join(base_dir, "trajectories")
    if not os.path.isdir(traj_root):
        return set()
    out = set()
    for name in os.listdir(traj_root):
        if name.startswith("cell="):
            path = os.path.join(traj_root, name, "part.parquet")
            if os.path.isfile(path):
                try:
                    out.add(int(name.split("=")[1]))
                except ValueError:
                    pass
    return out


def run_sweep(cfg: Config, resume: bool = True):
    cells = build_cell_grid()
    n_workers = cfg.workers if cfg.workers > 0 else max(1, (os.cpu_count() or 2) - 1)

    completed = _completed_cells(cfg.out_dir) if resume else set()
    pending_cells = [c for c in cells if c["cell_id"] not in completed]
    print(f"cells total={len(cells)} already_done={len(completed)} "
          f"to_run={len(pending_cells)} workers={n_workers}", flush=True)

    cfg_dict = {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in cfg.__dict__.items()}

    total_rows = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        # Submit all tasks. Keep futures organized by cell_id so we can
        # write each cell's shard as soon as its 200 sims finish.
        pending_by_cell = {c["cell_id"]: cfg.n_sims for c in pending_cells}
        cell_rows = {c["cell_id"]: [] for c in pending_cells}
        cell_meta = {c["cell_id"]: [] for c in pending_cells}
        cell_info = {c["cell_id"]: c for c in pending_cells}

        futures = []
        for c in pending_cells:
            for sim in range(cfg.n_sims):
                futures.append(ex.submit(run_single, (c, sim, cfg_dict)))

        total_sims = len(futures)
        heartbeat_every = max(1, total_sims // 200)
        print(f"submitted {total_sims} sims across {len(pending_cells)} cells; "
              f"heartbeat every {heartbeat_every} sims", flush=True)

        n_done_cells = 0
        n_done_sims = 0
        for fut in as_completed(futures):
            cell_id, rows, sim_meta = fut.result()
            cell_rows[cell_id].extend(rows)
            cell_meta[cell_id].append(sim_meta)
            pending_by_cell[cell_id] -= 1
            n_done_sims += 1
            if n_done_sims % heartbeat_every == 0 and pending_by_cell[cell_id] != 0:
                elapsed = time.time() - t_start
                rate = n_done_sims / elapsed if elapsed > 0 else 0
                eta = ((total_sims - n_done_sims) / rate
                       if rate > 0 else float("inf"))
                print(f"    sims [{n_done_sims}/{total_sims}]  "
                      f"cells [{n_done_cells}/{len(pending_cells)}]  "
                      f"rate={rate:.2f} sim/s  "
                      f"elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)
            if pending_by_cell[cell_id] == 0:
                df = pd.DataFrame(cell_rows[cell_id])
                df.to_parquet(_shard_path(cfg.out_dir, cell_id), index=False)
                meta_df = pd.DataFrame(cell_meta[cell_id])
                meta_df.to_parquet(_sims_shard_path(cfg.out_dir, cell_id),
                                   index=False)
                total_rows += len(df)
                # free memory
                cell_rows.pop(cell_id)
                cell_meta.pop(cell_id)

                n_done_cells += 1
                info = cell_info[cell_id]
                elapsed = time.time() - t_start
                rate = n_done_cells / elapsed if elapsed > 0 else 0
                eta = ((len(pending_cells) - n_done_cells) / rate
                       if rate > 0 else float("inf"))
                print(f"  cell {cell_id:03d} done  "
                      f"theta={info['theta_star']:.1f} psi={info['psi_true']:<5} "
                      f"alpha={info['alpha']:<5}  "
                      f"[{n_done_cells}/{len(pending_cells)}]  "
                      f"elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)
    return total_rows, time.time() - t_start


# ---------------------------------------------------------------------------
# Post-sweep: config + sanity checks
# ---------------------------------------------------------------------------

def git_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                      cwd=_ROOT, stderr=subprocess.DEVNULL)
        return out.decode("ascii").strip()
    except Exception:
        return "unknown"


def write_run_config(cfg: Config, wall_seconds: float, total_rows: int):
    info = {
        "grid": {
            "theta_stars": list(THETA_STARS),
            "psi_trues": list(PSI_TRUES),
            "alphas": list(ALPHAS),
            "n": cfg.n,
            "m": cfg.m,
            "speaker_level": cfg.speaker_level,
            "rounds": cfg.rounds,
            "n_sims_per_cell": cfg.n_sims,
            "n_cells": len(build_cell_grid()),
        },
        "seed_base": cfg.seed,
        "theta_space": list(cfg.theta_space),
        "psi_label_map": PSI_TO_CODE,
        "git_hash": git_hash(),
        "wall_seconds": wall_seconds,
        "total_trajectory_rows": total_rows,
        "run_date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = os.path.join(cfg.out_dir, "run_config.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    print(f"wrote {path}")


def run_sanity_checks(cfg: Config):
    """Spec-defined sanity checks. Returns markdown report string."""
    traj_root = os.path.join(cfg.out_dir, "trajectories")
    df = pd.read_parquet(traj_root)
    sims_root = os.path.join(cfg.out_dir, "simulations")
    sims = pd.read_parquet(sims_root)

    lines = ["# sanity.md", "",
             "Automated checks from `run_sweep.py`. All should pass.", ""]

    # 1. Row count.
    expected_rows = (len(THETA_STARS) * len(PSI_TRUES) * len(ALPHAS)
                     * cfg.n_sims * cfg.rounds)
    ok1 = len(df) == expected_rows
    lines.append(f"**1. Row count.** expected={expected_rows:,}  "
                 f"actual={len(df):,}  "
                 f"=> {'PASS' if ok1 else 'FAIL'}")

    # 2. No NaN/inf in any score column.
    score_cols = [
        "surp2_score", "surp2_sigma2", "surp2_Sus", "surp2_sigma_bar2",
        "sus1_score", "sus1_sigma2_naive", "sus1_sigma2_corrected",
        "sus1_Sus", "sus1_sigma_bar2_naive", "sus1_sigma_bar2_corrected",
    ]
    bad = {c: int((~np.isfinite(df[c])).sum()) for c in score_cols}
    ok2 = all(v == 0 for v in bad.values())
    lines.append(f"**2. No NaN/inf in score columns.** "
                 f"=> {'PASS' if ok2 else 'FAIL'}")
    if not ok2:
        for c, v in bad.items():
            if v:
                lines.append(f"   - `{c}`: {v} non-finite")

    # 3. sus1 corrected <= naive (up to 1e-9).
    diff = df["sus1_sigma2_corrected"] - df["sus1_sigma2_naive"]
    n_violations = int((diff > 1e-9).sum())
    ok3 = n_violations == 0
    lines.append(f"**3. `sus1_sigma2_corrected <= sus1_sigma2_naive`** "
                 f"(up to 1e-9 tol).  violations={n_violations}  "
                 f"=> {'PASS' if ok3 else 'FAIL'}")

    # 4. Beliefs sum to 1.
    n_theta = len(cfg.theta_space)
    l1_cols = [f"L1_theta_{i}" for i in range(n_theta)]
    l0_cols = [f"L0_theta_{i}" for i in range(n_theta)]
    l1_sums = df[l1_cols].sum(axis=1)
    l0_sums = df[l0_cols].sum(axis=1)
    ok4a = np.allclose(l1_sums, 1.0, atol=1e-6)
    ok4b = np.allclose(l0_sums, 1.0, atol=1e-6)
    lines.append(f"**4. Beliefs sum to 1.**  "
                 f"L1 max|sum-1|={float((l1_sums - 1).abs().max()):.2e}  "
                 f"L0 max|sum-1|={float((l0_sums - 1).abs().max()):.2e}  "
                 f"=> {'PASS' if (ok4a and ok4b) else 'FAIL'}")

    # 5. Mean-zero asymptotic under null.
    null = df[(df["psi_true"] == "inf") & (df["round"] == cfg.rounds)]
    mean_surp2 = float(null["surp2_Sus"].mean())
    mean_sus1 = float(null["sus1_Sus"].mean())
    abs_th = 0.05  # loose informational threshold; FPR selection bias is known
    lines.append(f"**5. `Sus(t={cfg.rounds})` mean under null** "
                 f"(pooled over theta*, alpha).")
    lines.append(f"   - surp2: {mean_surp2:+.4f}")
    lines.append(f"   - sus_1: {mean_sus1:+.4f}")
    lines.append(f"   (expected close to 0; slight positive bias acceptable due "
                 f"to multiple-testing selection discussed in Group 1.)")
    lines.append("")

    # Per-axis breakdown (useful diagnostic).
    lines.append("## Per-alpha null Sus(t=150) means")
    lines.append("")
    grp = (null.groupby("alpha")
                .agg(surp2=("surp2_Sus", "mean"),
                     sus1=("sus1_Sus", "mean")).round(4))
    lines.append("```")
    lines.append(grp.to_string())
    lines.append("```")
    lines.append("")

    lines.append("## Dataset shape")
    lines.append("")
    lines.append(f"- trajectories: {len(df):,} rows, "
                 f"{len(df.columns)} columns")
    lines.append(f"- simulations: {len(sims):,} rows")
    lines.append(f"- unique cells: {df['cell_id'].nunique()}")

    status_all = ok1 and ok2 and ok3 and ok4a and ok4b
    lines.insert(3, f"**Overall: {'PASS' if status_all else 'FAIL'}**  ")
    lines.insert(4, "")

    report = "\n".join(lines)
    path = os.path.join(cfg.out_dir, "sanity.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"wrote {path}")
    print(report)
    return status_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: 1 cell, 5 sims, 20 rounds")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--n_sims", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--no_resume", action="store_true",
                    help="rerun even if a cell shard exists")
    ap.add_argument("--skip_sanity", action="store_true")
    ap.add_argument("--only_alpha", type=float, nargs="+", default=None,
                    help="restrict grid to these alpha values")
    ap.add_argument("--only_psi", type=str, nargs="+", default=None,
                    help="restrict grid to these psi_true labels (inf/pers+/pers-)")
    ap.add_argument("--only_theta", type=float, nargs="+", default=None,
                    help="restrict grid to these theta_star values")
    args = ap.parse_args()

    cfg = Config()
    if args.smoke:
        cfg.n_sims = 5
        cfg.rounds = 20
        cfg.out_dir = "results/full_sweep_smoke"
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.workers:
        cfg.workers = args.workers
    if args.n_sims is not None:
        cfg.n_sims = args.n_sims
    if args.rounds is not None:
        cfg.rounds = args.rounds

    os.makedirs(cfg.out_dir, exist_ok=True)

    # Optional grid filter — applied via module globals so build_cell_grid()
    # (also called inside run_sweep / sanity / config) sees the restricted set.
    global THETA_STARS, PSI_TRUES, ALPHAS
    if args.only_theta is not None:
        THETA_STARS = tuple(float(x) for x in args.only_theta)
    if args.only_psi is not None:
        PSI_TRUES = tuple(args.only_psi)
    if args.only_alpha is not None:
        ALPHAS = tuple(float(x) for x in args.only_alpha)

    n_cells = len(build_cell_grid())
    print(f"cfg: cells={n_cells} n_sims/cell={cfg.n_sims} rounds={cfg.rounds}  "
          f"n={cfg.n} m={cfg.m}  out_dir={cfg.out_dir}", flush=True)
    if (args.only_alpha or args.only_psi or args.only_theta):
        print(f"  grid filter: theta={list(THETA_STARS)} "
              f"psi={list(PSI_TRUES)} alpha={list(ALPHAS)}", flush=True)

    total_rows, wall = run_sweep(cfg, resume=not args.no_resume)
    # Combine with any previously-existing shards when reporting totals.
    all_rows = sum(
        len(pd.read_parquet(os.path.join(cfg.out_dir, "trajectories", d, "part.parquet")))
        for d in sorted(os.listdir(os.path.join(cfg.out_dir, "trajectories")))
        if d.startswith("cell=")
    )
    write_run_config(cfg, wall, all_rows)

    # file sizes
    traj_root = os.path.join(cfg.out_dir, "trajectories")
    total_bytes = sum(
        os.path.getsize(os.path.join(traj_root, d, "part.parquet"))
        for d in os.listdir(traj_root) if d.startswith("cell=")
    )
    print(f"total wall: {wall:.0f}s")
    print(f"total trajectory rows (incl. prior shards): {all_rows:,}")
    print(f"total trajectory bytes: {total_bytes/1e9:.2f} GB")

    if not args.skip_sanity:
        run_sanity_checks(cfg)


if __name__ == "__main__":
    main()
