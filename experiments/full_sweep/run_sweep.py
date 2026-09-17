"""
Full sweep of core detection experiments -- see full_sweep_spec.md.

Grid: theta_star x psi_true x alpha = 9 x 3 x 11 = 297 cells.
200 simulations per cell, 150 rounds each, so 8,910,000 trajectory rows
in total. Two scores (surp2, sus_1) run as passive observers per sim; each
reports one exact state-only null variance.

Output: partitioned parquet dataset at
    results/full_sweep/trajectories/cell=XXXX/part.parquet
plus simulations/cell=XXXX/part.parquet, run_config.json, progress.jsonl,
sanity.md.

Memory and durability
---------------------
The dataset is ~30x larger than RAM allows to hold as Python objects, so
nothing proportional to the dataset is ever held in the parent process:

* One task == one cell (``n_sims`` simulations).  The worker builds that
  cell's columns as numpy arrays (~11 MB), writes the shard itself, and
  returns only a small summary dict.  Parent memory is O(1) in the number
  of cells done.
* Results are never retained.  ``concurrent.futures.Future`` keeps its
  ``_result`` alive for as long as the future object exists, so futures are
  dropped as soon as they are consumed and only ``max_inflight`` are ever
  submitted at once.
* Shards are written to ``part.parquet.tmp`` and atomically renamed, so an
  interrupted run can never leave a half-written shard that ``--resume``
  would mistake for finished work.  ``simulations`` is written before
  ``trajectories``, so the trajectory shard's presence implies both.
* Every finished cell is appended to ``progress.jsonl`` and fsync'd, and
  ``run_config.json`` is written at start-up (``status: running``) and
  rewritten at the end, so a crash still leaves full provenance on disk.
* Sanity checks stream the dataset one shard at a time -- they never load
  the full 8.9 M rows.
* A free-memory guard stops submitting new cells if the system gets close
  to exhausting RAM, so the run degrades into a clean resumable stop rather
  than an OOM kill.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    BrokenExecutor,
    ProcessPoolExecutor,
    wait,
)
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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

# Score columns, in on-disk order.  Keep this layout and dtype set stable so a
# resumed sweep and the shards it inherits form one dataset with one schema.
#
# sus_1 now reports a single exact state-only variance, so the naive/corrected
# pair is gone: sus1_sigma2 replaces sus1_sigma2_{naive,corrected} and
# sus1_sigma_bar2 replaces sus1_sigma_bar2_{naive,corrected}.  This schema is
# NOT compatible with results/full_sweep/ (44 columns); v2 output goes to a
# separate directory.
SCORE_COLS = (
    "surp2_score", "surp2_sigma2", "surp2_Sus", "surp2_sigma_bar2",
    "sus1_score", "sus1_sigma2", "sus1_Sus", "sus1_sigma_bar2",
)
INT_COLS = ("sim_id", "round", "u_observed", "O_true_idx", "O_true_count")


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
# Paths and atomic writes
# ---------------------------------------------------------------------------

def _shard_path(base_dir: str, cell_id: int) -> str:
    d = os.path.join(base_dir, "trajectories", f"cell={cell_id:04d}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "part.parquet")


def _sims_shard_path(base_dir: str, cell_id: int) -> str:
    d = os.path.join(base_dir, "simulations", f"cell={cell_id:04d}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "part.parquet")


def _write_parquet_atomic(df: pd.DataFrame, path: str) -> int:
    """Write via a .tmp sibling + os.replace, so readers never see a partial file."""
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)           # atomic on both POSIX and Windows
    return os.path.getsize(path)


def _parquet_rows(path: str):
    """Row count from the parquet footer, or None if unreadable/truncated."""
    try:
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return None


def _sweep_dirs(base_dir: str):
    return (os.path.join(base_dir, "trajectories"),
            os.path.join(base_dir, "simulations"))


def _clear_stale_tmp(base_dir: str) -> int:
    """Remove .tmp shards left behind by an interrupted run."""
    removed = 0
    for root in _sweep_dirs(base_dir):
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            tmp = os.path.join(root, name, "part.parquet.tmp")
            if os.path.isfile(tmp):
                try:
                    os.remove(tmp)
                    removed += 1
                except OSError:
                    pass
    return removed


def _completed_cells(base_dir: str, cfg: Config, verify: bool = True) -> set:
    """
    Cell ids that are genuinely finished: both shards present, and (when
    ``verify``) both with the expected row count.  A shard whose footer is
    unreadable or short is treated as missing so the cell is recomputed.
    """
    traj_root, sims_root = _sweep_dirs(base_dir)
    if not os.path.isdir(traj_root):
        return set()
    want_traj = cfg.n_sims * cfg.rounds
    out = set()
    for name in sorted(os.listdir(traj_root)):
        if not name.startswith("cell="):
            continue
        try:
            cid = int(name.split("=")[1])
        except ValueError:
            continue
        traj = os.path.join(traj_root, name, "part.parquet")
        sims = os.path.join(sims_root, name, "part.parquet")
        if not (os.path.isfile(traj) and os.path.isfile(sims)):
            continue
        if verify:
            if _parquet_rows(traj) != want_traj:
                continue
            if _parquet_rows(sims) != cfg.n_sims:
                continue
        out.add(cid)
    return out


# ---------------------------------------------------------------------------
# Worker: one cell == one task
# ---------------------------------------------------------------------------

def _run_one_sim(cell, sim_idx, cfg, dest, ints, scores, l1_out, l0_out) -> int:
    """
    Run one simulation, writing its ``cfg.rounds`` rows into the preallocated
    column arrays starting at row ``dest``.  Returns the seed used.

    Reseeds the global RNGs first, exactly as the previous one-sim-per-task
    worker did, so results are identical to that layout.
    """
    cell_id = cell["cell_id"]
    theta_star = cell["theta_star"]
    psi_code = PSI_TO_CODE[cell["psi_true"]]
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

    for t in range(cfg.rounds):
        k = dest + t

        # Capture pre-update beliefs (these are what the score context uses).
        l1_out[k, :] = listener._l1_theta_array()
        l0_out[k, :] = np.asarray(l0.state_belief.prob, dtype=float)

        obs = world.sample_obs()
        utt = s1.sample_utterance(obs)
        listener.update(utt)
        s1.update(obs)
        l0.update(utt)
        s0.update(obs)

        # SequentialTest just recorded this round at index t.
        surp2_t = tests[0].history
        sus_t = tests[1].history

        ints["sim_id"][k] = sim_idx
        ints["round"][k] = t + 1
        ints["u_observed"][k] = int(semantics.utterance_index(utt))
        ints["O_true_idx"][k] = int(world.obs_index(obs))
        ints["O_true_count"][k] = int(np.argmax(obs))

        scores["surp2_score"][k] = surp2_t["scores"][t]
        scores["surp2_sigma2"][k] = surp2_t["variances"][t]
        scores["surp2_Sus"][k] = surp2_t["running_mean"][t]
        scores["surp2_sigma_bar2"][k] = surp2_t["running_sigma"][t] ** 2
        scores["sus1_score"][k] = sus_t["scores"][t]
        scores["sus1_sigma2"][k] = sus_t["variances"][t]
        scores["sus1_Sus"][k] = sus_t["running_mean"][t]
        scores["sus1_sigma_bar2"][k] = sus_t["running_sigma"][t] ** 2

    return rng_seed


def run_cell(task):
    """
    Worker entry point: run every sim for one cell, write both shards, and
    return a small summary.  Nothing large crosses the process boundary.
    """
    cell, cfg_dict = task
    cfg = Config(**cfg_dict)
    t0 = time.time()

    cell_id = cell["cell_id"]
    n_theta = len(cfg.theta_space)
    n_rows = cfg.n_sims * cfg.rounds

    # ~11 MB of columns for a 200 x 150 cell, versus ~100 MB as row dicts.
    ints = {c: np.empty(n_rows, dtype=np.int64) for c in INT_COLS}
    scores = {c: np.empty(n_rows, dtype=np.float64) for c in SCORE_COLS}
    l1_out = np.empty((n_rows, n_theta), dtype=np.float64)
    l0_out = np.empty((n_rows, n_theta), dtype=np.float64)
    seeds = np.empty(cfg.n_sims, dtype=np.int64)

    for sim_idx in range(cfg.n_sims):
        seeds[sim_idx] = _run_one_sim(
            cell, sim_idx, cfg, sim_idx * cfg.rounds,
            ints, scores, l1_out, l0_out,
        )

    data = {
        "cell_id": np.full(n_rows, cell_id, dtype=np.int64),
        "sim_id": ints["sim_id"],
        "round": ints["round"],
        "theta_star": np.full(n_rows, cell["theta_star"], dtype=np.float64),
        "psi_true": np.full(n_rows, cell["psi_true"], dtype=object),
        "alpha": np.full(n_rows, cell["alpha"], dtype=np.float64),
        "n": np.full(n_rows, cfg.n, dtype=np.int64),
        "m": np.full(n_rows, cfg.m, dtype=np.int64),
        "speaker_level": np.full(n_rows, cfg.speaker_level, dtype=object),
        "u_observed": ints["u_observed"],
        "O_true_idx": ints["O_true_idx"],
        "O_true_count": ints["O_true_count"],
    }
    data.update({c: scores[c] for c in SCORE_COLS})
    for i in range(n_theta):
        data[f"L1_theta_{i}"] = l1_out[:, i]
        data[f"L0_theta_{i}"] = l0_out[:, i]

    sims_df = pd.DataFrame({
        "cell_id": np.full(cfg.n_sims, cell_id, dtype=np.int64),
        "sim_id": np.arange(cfg.n_sims, dtype=np.int64),
        "theta_star": np.full(cfg.n_sims, cell["theta_star"], dtype=np.float64),
        "psi_true": np.full(cfg.n_sims, cell["psi_true"], dtype=object),
        "alpha": np.full(cfg.n_sims, cell["alpha"], dtype=np.float64),
        "n": np.full(cfg.n_sims, cfg.n, dtype=np.int64),
        "m": np.full(cfg.n_sims, cfg.m, dtype=np.int64),
        "speaker_level": np.full(cfg.n_sims, cfg.speaker_level, dtype=object),
        "seed": seeds,
    })

    # simulations first: the trajectory shard is the resume marker, so it must
    # be the last thing to appear.
    _write_parquet_atomic(sims_df, _sims_shard_path(cfg.out_dir, cell_id))
    n_bytes = _write_parquet_atomic(
        pd.DataFrame(data), _shard_path(cfg.out_dir, cell_id))

    return {
        "cell_id": cell_id,
        "theta_star": cell["theta_star"],
        "psi_true": cell["psi_true"],
        "alpha": cell["alpha"],
        "rows": n_rows,
        "bytes": n_bytes,
        "wall_s": round(time.time() - t0, 2),
    }


# ---------------------------------------------------------------------------
# Memory guard
# ---------------------------------------------------------------------------

def _mem_stats():
    """(available_gb, parent_rss_gb); infs/zeros if psutil is unavailable."""
    try:
        import psutil
    except Exception:
        return float("inf"), 0.0
    try:
        return (psutil.virtual_memory().available / 1e9,
                psutil.Process().memory_info().rss / 1e9)
    except Exception:
        return float("inf"), 0.0


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def _journal(base_dir: str, record: dict) -> None:
    """Append one durable line to progress.jsonl."""
    path = os.path.join(base_dir, "progress.jsonl")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def run_sweep(cfg: Config, resume: bool = True, max_inflight: int = 0,
              min_free_gb: float = 1.0, verify_shards: bool = True):
    cells = build_cell_grid()
    n_workers = cfg.workers if cfg.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    if max_inflight <= 0:
        max_inflight = 2 * n_workers

    n_stale = _clear_stale_tmp(cfg.out_dir)
    if n_stale:
        print(f"removed {n_stale} stale .tmp shard(s) from an interrupted run",
              flush=True)

    completed = _completed_cells(cfg.out_dir, cfg, verify=verify_shards) if resume else set()
    pending_cells = [c for c in cells if c["cell_id"] not in completed]
    avail, _ = _mem_stats()
    print(f"cells total={len(cells)} already_done={len(completed)} "
          f"to_run={len(pending_cells)} workers={n_workers} "
          f"inflight<={max_inflight} free_ram={avail:.1f}GB", flush=True)

    cfg_dict = {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in cfg.__dict__.items()}

    total_rows = 0
    failures = []
    stopped_early = None
    t_start = time.time()

    if not pending_cells:
        return 0, 0.0, failures, None

    queue = iter(pending_cells)
    n_done = 0
    n_target = len(pending_cells)

    # max_tasks_per_child recycles workers so nothing can creep up over a
    # multi-hour run; harmless if the interpreter is too old to support it.
    try:
        ex = ProcessPoolExecutor(max_workers=n_workers, max_tasks_per_child=8)
    except TypeError:
        ex = ProcessPoolExecutor(max_workers=n_workers)

    try:
        inflight = {}

        def _submit_next():
            """Submit one more cell unless the queue is drained or RAM is low."""
            nonlocal stopped_early
            if stopped_early is not None:
                return False
            cell = next(queue, None)
            if cell is None:
                return False
            inflight[ex.submit(run_cell, (cell, cfg_dict))] = cell
            return True

        for _ in range(max_inflight):
            if not _submit_next():
                break

        while inflight:
            done, _pending = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:
                cell = inflight.pop(fut)
                try:
                    summary = fut.result()
                except Exception as exc:            # keep going; cell is retryable
                    failures.append({"cell_id": cell["cell_id"],
                                     "error": f"{type(exc).__name__}: {exc}"})
                    print(f"  cell {cell['cell_id']:03d} FAILED: "
                          f"{type(exc).__name__}: {exc}", flush=True)
                    summary = None
                finally:
                    # Futures keep their result alive; drop ours immediately.
                    del fut

                n_done += 1
                if summary is not None:
                    total_rows += summary["rows"]
                    _journal(cfg.out_dir, summary)

                avail, rss = _mem_stats()
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0.0
                eta = (n_target - n_done) / rate if rate > 0 else float("inf")
                info = summary if summary is not None else cell
                print(f"  cell {info['cell_id']:03d} done  "
                      f"theta={info['theta_star']:.1f} "
                      f"psi={info['psi_true']:<5} "
                      f"alpha={info['alpha']:<5}  "
                      f"[{n_done}/{n_target}]  "
                      f"elapsed={elapsed:.0f}s  eta={eta:.0f}s  "
                      f"rss={rss:.2f}GB free={avail:.1f}GB", flush=True)

                if min_free_gb > 0 and avail < min_free_gb and stopped_early is None:
                    stopped_early = (
                        f"free RAM fell to {avail:.2f} GB (< {min_free_gb:.2f} GB); "
                        f"stopped submitting new cells")
                    print(f"\n!! {stopped_early}\n"
                          f"   finishing {len(inflight)} in-flight cell(s), then "
                          f"exiting cleanly. Everything finished is on disk; "
                          f"rerun the same command to resume.\n", flush=True)

                _submit_next()

    except KeyboardInterrupt:
        stopped_early = "interrupted by user (Ctrl-C)"
        print(f"\n!! {stopped_early}; finished cells are on disk, "
              f"rerun the same command to resume.\n", flush=True)
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    except BrokenExecutor as exc:
        stopped_early = f"worker pool broke: {type(exc).__name__}: {exc}"
        print(f"\n!! {stopped_early}; finished cells are on disk, "
              f"rerun the same command to resume.\n", flush=True)
    finally:
        ex.shutdown(wait=True)

    return total_rows, time.time() - t_start, failures, stopped_early


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


def write_run_config(cfg: Config, wall_seconds: float, total_rows: int,
                     status: str = "complete", extra: dict | None = None):
    info = {
        "status": status,
        # Recorded so watch_progress.py can compute an exact ETA instead of
        # inferring concurrency from cell durations.
        "workers": (cfg.workers if cfg.workers > 0
                    else max(1, (os.cpu_count() or 2) - 1)),
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
    if extra:
        info.update(extra)
    path = os.path.join(cfg.out_dir, "run_config.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    print(f"wrote {path}")


def iter_shards(base_dir: str):
    """Yield (cell_id, trajectory_path) for every shard on disk, in cell order."""
    traj_root = os.path.join(base_dir, "trajectories")
    if not os.path.isdir(traj_root):
        return
    for name in sorted(os.listdir(traj_root)):
        if not name.startswith("cell="):
            continue
        path = os.path.join(traj_root, name, "part.parquet")
        if os.path.isfile(path):
            try:
                yield int(name.split("=")[1]), path
            except ValueError:
                continue


def verify_dataset(cfg: Config, quiet: bool = False):
    """
    Footer-only integrity pass: every shard readable, right row count, both
    halves present.  Returns (ok, report_lines, n_shards, n_rows).
    """
    want_traj = cfg.n_sims * cfg.rounds
    bad, n_shards, n_rows = [], 0, 0
    for cid, path in iter_shards(cfg.out_dir):
        n_shards += 1
        rows = _parquet_rows(path)
        if rows is None:
            bad.append(f"cell={cid:04d}: trajectory shard unreadable/truncated")
            continue
        n_rows += rows
        if rows != want_traj:
            bad.append(f"cell={cid:04d}: {rows:,} trajectory rows "
                       f"(expected {want_traj:,})")
        sims = _sims_shard_path(cfg.out_dir, cid)
        srows = _parquet_rows(sims) if os.path.isfile(sims) else None
        if srows is None:
            bad.append(f"cell={cid:04d}: simulations shard missing/unreadable")
        elif srows != cfg.n_sims:
            bad.append(f"cell={cid:04d}: {srows} simulation rows "
                       f"(expected {cfg.n_sims})")
    lines = [f"- shards on disk: {n_shards} / {len(build_cell_grid())}",
             f"- trajectory rows on disk: {n_rows:,}"]
    lines += ([f"- integrity: OK"] if not bad
              else [f"- integrity: {len(bad)} PROBLEM(S)"] + [f"  - {b}" for b in bad])
    if not quiet:
        print("\n".join(lines))
    return (not bad), lines, n_shards, n_rows


def run_sanity_checks(cfg: Config):
    """
    Spec-defined sanity checks, computed by streaming one cell shard at a
    time (~11 MB resident) instead of loading the whole dataset.  Writes
    sanity.md and returns the overall pass flag.
    """
    n_theta = len(cfg.theta_space)
    l1_cols = [f"L1_theta_{i}" for i in range(n_theta)]
    l0_cols = [f"L0_theta_{i}" for i in range(n_theta)]
    score_cols = list(SCORE_COLS)
    need = (["round", "alpha", "psi_true"] + score_cols + l1_cols + l0_cols)

    n_rows = 0
    n_shards = 0
    nonfinite = {c: 0 for c in score_cols}
    n_var_violations = 0
    max_l1_dev = 0.0
    max_l0_dev = 0.0
    null_sums = {}          # alpha -> [surp2 sum, sus1 sum, count]
    missing_cols = set()

    for _cid, path in iter_shards(cfg.out_dir):
        n_shards += 1
        have = set(pq.ParquetFile(path).schema_arrow.names)
        cols = [c for c in need if c in have]
        missing_cols |= (set(need) - have)
        df = pd.read_parquet(path, columns=cols)
        n_rows += len(df)

        for c in score_cols:
            if c in df:
                nonfinite[c] += int((~np.isfinite(df[c].to_numpy())).sum())

        # The exact variance is a probability-weighted sum of squares, so any
        # negative value is a bug rather than a tolerance issue.
        for c in ("surp2_sigma2", "sus1_sigma2"):
            if c in df.columns:
                n_var_violations += int((df[c].to_numpy() < 0.0).sum())

        if set(l1_cols) <= set(df.columns):
            dev = np.abs(df[l1_cols].to_numpy().sum(axis=1) - 1.0).max()
            max_l1_dev = max(max_l1_dev, float(dev))
        if set(l0_cols) <= set(df.columns):
            dev = np.abs(df[l0_cols].to_numpy().sum(axis=1) - 1.0).max()
            max_l0_dev = max(max_l0_dev, float(dev))

        # Check 5 is over the null cells at the final round only.
        if {"psi_true", "round", "alpha"} <= set(df.columns):
            tail = df[(df["psi_true"] == "inf") & (df["round"] == cfg.rounds)]
            if len(tail):
                for alpha, g in tail.groupby("alpha"):
                    acc = null_sums.setdefault(float(alpha), [0.0, 0.0, 0])
                    acc[0] += float(g["surp2_Sus"].sum())
                    acc[1] += float(g["sus1_Sus"].sum())
                    acc[2] += len(g)
        del df

    lines = ["# sanity.md", "",
             "Automated checks from `run_sweep.py`. All should pass.", ""]

    # 1. Row count.
    expected_rows = (len(THETA_STARS) * len(PSI_TRUES) * len(ALPHAS)
                     * cfg.n_sims * cfg.rounds)
    ok1 = n_rows == expected_rows
    lines.append(f"**1. Row count.** expected={expected_rows:,}  "
                 f"actual={n_rows:,}  "
                 f"=> {'PASS' if ok1 else 'FAIL'}")
    if not ok1:
        lines.append(f"   - {n_shards} of "
                     f"{len(build_cell_grid())} cell shards present; the sweep "
                     f"is incomplete or was run with a grid filter.")

    # 2. No NaN/inf in any score column.
    ok2 = all(v == 0 for v in nonfinite.values())
    lines.append(f"**2. No NaN/inf in score columns.** "
                 f"=> {'PASS' if ok2 else 'FAIL'}")
    if not ok2:
        for c, v in nonfinite.items():
            if v:
                lines.append(f"   - `{c}`: {v} non-finite")

    # 3. Per-round variances are non-negative by construction.
    ok3 = n_var_violations == 0
    lines.append(f"**3. Per-round variances non-negative.** "
                 f"(`surp2_sigma2`, `sus1_sigma2`; the exact score variance is "
                 f"a probability-weighted sum of squares.)  "
                 f"violations={n_var_violations}  "
                 f"=> {'PASS' if ok3 else 'FAIL'}")

    # 4. Beliefs sum to 1.
    ok4 = (max_l1_dev <= 1e-6) and (max_l0_dev <= 1e-6)
    lines.append(f"**4. Beliefs sum to 1.**  "
                 f"L1 max|sum-1|={max_l1_dev:.2e}  "
                 f"L0 max|sum-1|={max_l0_dev:.2e}  "
                 f"=> {'PASS' if ok4 else 'FAIL'}")

    # 5. Mean-zero asymptotic under null.
    tot_n = sum(v[2] for v in null_sums.values())
    mean_surp2 = (sum(v[0] for v in null_sums.values()) / tot_n) if tot_n else float("nan")
    mean_sus1 = (sum(v[1] for v in null_sums.values()) / tot_n) if tot_n else float("nan")
    lines.append(f"**5. `Sus(t={cfg.rounds})` mean under null** "
                 f"(pooled over theta*, alpha).")
    lines.append(f"   - surp2: {mean_surp2:+.4f}")
    lines.append(f"   - sus_1: {mean_sus1:+.4f}")
    lines.append(f"   (expected close to 0; slight positive bias acceptable due "
                 f"to multiple-testing selection discussed in Group 1.)")
    lines.append("")

    # Per-axis breakdown (useful diagnostic).
    lines.append("## Per-alpha null Sus(t=%d) means" % cfg.rounds)
    lines.append("")
    if null_sums:
        grp = pd.DataFrame(
            [{"alpha": a, "surp2": v[0] / v[2], "sus1": v[1] / v[2]}
             for a, v in sorted(null_sums.items()) if v[2]]
        ).set_index("alpha").round(4)
        lines.append("```")
        lines.append(grp.to_string())
        lines.append("```")
    else:
        lines.append("_no null (psi=inf) rows found_")
    lines.append("")

    lines.append("## Dataset shape")
    lines.append("")
    lines.append(f"- trajectories: {n_rows:,} rows across {n_shards} cell shards")
    if missing_cols:
        lines.append(f"- WARNING: columns absent from some shards: "
                     f"{sorted(missing_cols)}")
    ok_int, int_lines, _, _ = verify_dataset(cfg, quiet=True)
    lines.extend(int_lines)

    status_all = ok1 and ok2 and ok3 and ok4 and ok_int
    lines.insert(3, f"**Overall: {'PASS' if status_all else 'FAIL'}**  ")
    lines.insert(4, "")

    report = "\n".join(lines)
    path = os.path.join(cfg.out_dir, "sanity.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"wrote {path}")
    print(report)
    return status_all


def _dataset_totals(cfg: Config):
    """Row count and byte size from parquet footers -- no data is read."""
    rows = 0
    total_bytes = 0
    for _cid, path in iter_shards(cfg.out_dir):
        r = _parquet_rows(path)
        if r is not None:
            rows += r
        total_bytes += os.path.getsize(path)
    return rows, total_bytes


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
    ap.add_argument("--fast_resume", action="store_true",
                    help="trust existing shards without checking their row "
                         "counts (skips reading 297 parquet footers)")
    ap.add_argument("--skip_sanity", action="store_true")
    ap.add_argument("--sanity_only", action="store_true",
                    help="run the sanity checks over whatever is on disk and exit")
    ap.add_argument("--verify_only", action="store_true",
                    help="footer-only integrity check of existing shards, then exit")
    ap.add_argument("--max_inflight", type=int, default=0,
                    help="cells queued at once (default: 2 x workers). Bounds "
                         "peak memory; lower it if RAM is tight.")
    ap.add_argument("--min_free_gb", type=float, default=1.0,
                    help="stop submitting new cells when free RAM drops below "
                         "this, and exit cleanly (0 disables the guard)")
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

    # Optional grid filter -- applied via module globals so build_cell_grid()
    # (also called inside run_sweep / sanity / config) sees the restricted set.
    global THETA_STARS, PSI_TRUES, ALPHAS
    if args.only_theta is not None:
        THETA_STARS = tuple(float(x) for x in args.only_theta)
    if args.only_psi is not None:
        PSI_TRUES = tuple(args.only_psi)
    if args.only_alpha is not None:
        ALPHAS = tuple(float(x) for x in args.only_alpha)

    if args.verify_only:
        ok, _, _, _ = verify_dataset(cfg)
        sys.exit(0 if ok else 1)

    if args.sanity_only:
        ok = run_sanity_checks(cfg)
        sys.exit(0 if ok else 1)

    n_cells = len(build_cell_grid())
    print(f"cfg: cells={n_cells} n_sims/cell={cfg.n_sims} rounds={cfg.rounds}  "
          f"n={cfg.n} m={cfg.m}  out_dir={cfg.out_dir}", flush=True)
    if (args.only_alpha or args.only_psi or args.only_theta):
        print(f"  grid filter: theta={list(THETA_STARS)} "
              f"psi={list(PSI_TRUES)} alpha={list(ALPHAS)}", flush=True)

    # Provenance goes to disk before any work starts, so a crash mid-sweep
    # still leaves a record of what this run was.
    write_run_config(cfg, 0.0, 0, status="running")

    total_rows, wall, failures, stopped_early = run_sweep(
        cfg,
        resume=not args.no_resume,
        max_inflight=args.max_inflight,
        min_free_gb=args.min_free_gb,
        verify_shards=not args.fast_resume,
    )

    all_rows, total_bytes = _dataset_totals(cfg)
    status = "complete"
    if stopped_early is not None:
        status = "incomplete"
    elif failures:
        status = "partial"
    write_run_config(cfg, wall, all_rows, status=status,
                     extra={"failed_cells": failures,
                            "stopped_early": stopped_early})

    print(f"total wall: {wall:.0f}s")
    print(f"total trajectory rows (incl. prior shards): {all_rows:,}")
    print(f"total trajectory bytes: {total_bytes/1e9:.2f} GB")
    if failures:
        print(f"!! {len(failures)} cell(s) failed: "
              f"{[f['cell_id'] for f in failures]}  -- rerun to retry them")

    if not args.skip_sanity:
        run_sanity_checks(cfg)

    if stopped_early is not None or failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
