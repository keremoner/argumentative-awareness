"""
Detector-triggered switching: does switching to a vigilant listener at the
detector's stopping time recover the vigilance benefit?

Four listener policies are run on ONE utterance stream per simulation, so the
comparison is paired (identical world observations and speaker utterances):

    credulous    L1 with psi = {inf} for all T rounds (Fang's coop listener)
    vigilant     L1 with psi = {inf, high, low} from round 1 (Fang's strat listener)
    switch_soft  credulous until the sus_1 test fires at tau, then vigilant whose
                 theta-marginal is inherited from the credulous belief, uniform
                 over psi
    switch_hard  credulous until tau, then vigilant from a uniform joint prior

The stream cannot depend on the listener: ``Speaker1`` reasons about its own L0
model and never observes the actual listener, and only the world and speaker
draw random numbers.  Seeds are ``_seed_for(cell_id, sim_idx, seed_base)``, the
same as the full sweep, so the credulous trajectory here is identical to the
``L1_theta_*`` columns stored in ``results/full_sweep_v2/`` for the same cell
and sim -- ``--check_against_sweep`` verifies that.

Cost is kept down by running one soft ``DetectionListener`` and, at its tau,
forking a fresh uniform vigilant ``Listener1`` for the hard policy; that is
exactly what a hard ``DetectionListener`` would construct (see
``tests/test_switching.py::test_manual_fork_at_tau_equals_hard_switch_listener``).
Hard and soft therefore share one tau per sim.

Per round the shard stores every policy's theta-marginal (so any metric --
E[theta], P(theta*), KL -- can be derived offline), the psi-marginal for the
policies that have one (NaN before the switch), the detector's running
statistic up to tau, and a ``switched`` flag.

Memory, durability and resume follow experiments/full_sweep/run_sweep.py: one
task per cell, atomic shard writes, progress.jsonl, run_config.json written
before work starts, free-RAM guard.  Interrupt it and rerun the same command.

Usage
-----
    python experiments/switching/run_switching.py --time_trial     # size it
    python experiments/switching/run_switching.py --smoke
    python experiments/switching/run_switching.py                  # full run
    python experiments/switching/run_switching.py --sanity_only
"""

from __future__ import annotations

import argparse
import json
import os
import random
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_ROOT, os.path.join(_ROOT, "experiments", "full_sweep")):
    if p not in sys.path:
        sys.path.insert(0, p)

from rsa.setup import make_thetas, make_world, make_semantics  # noqa: E402
from rsa.speaker0 import Speaker0  # noqa: E402
from rsa.listener0 import Listener0  # noqa: E402
from rsa.speaker1 import Speaker1  # noqa: E402
from rsa.listener1 import Listener1  # noqa: E402
from rsa.detection import DetectionListener, SequentialTest, SUS_VARIANT_FNS  # noqa: E402

import run_sweep as _rs  # noqa: E402  (shared grid, seeds, I/O helpers)

PSI_TO_CODE = _rs.PSI_TO_CODE
PSIS = ["inf", "high", "low"]
POLICIES = ("cred", "vig", "soft", "hard")
PSI_POLICIES = ("vig", "soft", "hard")

# Grid axes are module globals so --only_* filters work exactly as in run_sweep.
THETA_STARS = _rs.THETA_STARS
PSI_TRUES = _rs.PSI_TRUES
ALPHAS = _rs.ALPHAS


@dataclass
class Config:
    theta_space: list = field(default_factory=lambda: make_thetas(0.1, True, True))
    n: int = 1
    m: int = 7
    rounds: int = 150
    n_sims: int = 100
    seed: int = 0                 # same seed base as the sweep -> same streams
    c_switch: float = 3.5         # sus_1 cutoff that drives the switch
    out_dir: str = "results/switching"
    workers: int = 0


def build_cell_grid():
    cells, cid = [], 0
    for theta in THETA_STARS:
        for psi in PSI_TRUES:
            for alpha in ALPHAS:
                cells.append({"cell_id": cid, "theta_star": float(theta),
                              "psi_true": psi, "alpha": float(alpha)})
                cid += 1
    return cells


# ---------------------------------------------------------------------------
# Column layout
# ---------------------------------------------------------------------------

def column_names(n_theta: int):
    cols = ["cell_id", "sim_id", "round", "theta_star", "psi_true", "alpha",
            "u_observed", "O_true_idx", "switched",
            "sus1_Sus", "sus1_sigma_bar2"]
    for pol in POLICIES:
        cols += [f"{pol}_theta_{i}" for i in range(n_theta)]
    for pol in PSI_POLICIES:
        cols += [f"{pol}_psi_{p}" for p in PSIS]
    return cols


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _theta_arr(listener, thetas):
    d = listener.marginal_theta()
    return np.fromiter((d.get(t, 0.0) for t in thetas), dtype=float, count=len(thetas))


def _psi_arr(listener):
    d = listener.marginal_psi()
    return np.fromiter((d.get(p, 0.0) for p in PSIS), dtype=float, count=len(PSIS))


def _run_one_sim(cell, sim_idx, cfg, dest, out, thetas):
    """Run one paired stream; fill rows dest..dest+rounds of ``out``. Returns
    (seed, tau)."""
    cell_id = cell["cell_id"]
    theta_star = cell["theta_star"]
    psi_code = PSI_TO_CODE[cell["psi_true"]]
    alpha = cell["alpha"]
    n_theta = len(thetas)

    rng_seed = _rs._seed_for(cell_id, sim_idx, cfg.seed)
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    world = make_world(theta_star, n=cfg.n, m=cfg.m)
    semantics = make_semantics(n=cfg.n)
    s0 = Speaker0(thetas, semantics=semantics, world=world)
    l0 = Listener0(thetas, s0, semantics=semantics, world=world)
    s1 = Speaker1(thetas, l0, semantics=semantics, world=world,
                  alpha=alpha, psi=psi_code)

    cred = Listener1(thetas, ["inf"], s1, world, semantics, "inf", alpha)
    vig = Listener1(thetas, PSIS, s1, world, semantics, "vig", alpha)
    test = SequentialTest(SUS_VARIANT_FNS["1"], "sus_1", c=cfg.c_switch,
                          switch_enabled=True, switch_type="soft")
    det = DetectionListener(thetas, PSIS, s1, world, semantics,
                            tests=[test], alpha=alpha)
    hard = None

    nan3 = np.full(len(PSIS), np.nan)
    for t in range(cfg.rounds):
        obs = world.sample_obs()
        utt = s1.sample_utterance(obs)

        cred.update(utt)
        vig.update(utt)
        det.update(utt)
        if hard is not None:
            hard.update(utt)
        elif det.switched:
            # Fork the hard policy: uniform joint prior, first update next round.
            hard = Listener1(thetas, PSIS, s1, world, semantics, "vig", alpha)

        k = dest + t
        out["u_observed"][k] = int(semantics.utterance_index(utt))
        out["O_true_idx"][k] = int(world.obs_index(obs))
        out["switched"][k] = int(det.switched)
        h = test.history
        if t < len(h["running_mean"]):
            out["sus1_Sus"][k] = h["running_mean"][t]
            out["sus1_sigma_bar2"][k] = h["running_sigma"][t] ** 2
        else:
            out["sus1_Sus"][k] = np.nan
            out["sus1_sigma_bar2"][k] = np.nan

        th = {
            "cred": _theta_arr(cred, thetas),
            "vig": _theta_arr(vig, thetas),
            "soft": _theta_arr(det, thetas),
            "hard": _theta_arr(hard, thetas) if hard is not None
                    else _theta_arr(cred, thetas),
        }
        ps = {
            "vig": _psi_arr(vig),
            "soft": _psi_arr(det) if det.switched else nan3,
            "hard": _psi_arr(hard) if hard is not None else nan3,
        }
        for pol in POLICIES:
            for i in range(n_theta):
                out[f"{pol}_theta_{i}"][k] = th[pol][i]
        for pol in PSI_POLICIES:
            for j, p in enumerate(PSIS):
                out[f"{pol}_psi_{p}"][k] = ps[pol][j]

        s1.update(obs)
        l0.update(utt)
        s0.update(obs)

    tau = det.switched_at if det.switched else -1
    return rng_seed, tau


def run_cell(task):
    cell, cfg_dict = task
    cfg = Config(**cfg_dict)
    t0 = time.time()
    thetas = list(cfg.theta_space)
    n_theta = len(thetas)
    cell_id = cell["cell_id"]
    n_rows = cfg.n_sims * cfg.rounds

    cols = column_names(n_theta)
    int_cols = {"cell_id", "sim_id", "round", "u_observed", "O_true_idx", "switched"}
    out = {}
    for c in cols:
        if c in ("theta_star", "psi_true", "alpha"):
            continue
        out[c] = np.empty(n_rows, dtype=np.int64 if c in int_cols else np.float64)
    out["cell_id"][:] = cell_id
    out["sim_id"][:] = np.repeat(np.arange(cfg.n_sims), cfg.rounds)
    out["round"][:] = np.tile(np.arange(1, cfg.rounds + 1), cfg.n_sims)

    seeds = np.empty(cfg.n_sims, dtype=np.int64)
    taus = np.empty(cfg.n_sims, dtype=np.int64)
    for s in range(cfg.n_sims):
        seeds[s], taus[s] = _run_one_sim(cell, s, cfg, s * cfg.rounds, out, thetas)

    data = {}
    for c in cols:
        if c == "theta_star":
            data[c] = np.full(n_rows, cell["theta_star"])
        elif c == "psi_true":
            data[c] = np.full(n_rows, cell["psi_true"], dtype=object)
        elif c == "alpha":
            data[c] = np.full(n_rows, cell["alpha"])
        else:
            data[c] = out[c]

    sims_df = pd.DataFrame({
        "cell_id": np.full(cfg.n_sims, cell_id, dtype=np.int64),
        "sim_id": np.arange(cfg.n_sims, dtype=np.int64),
        "theta_star": np.full(cfg.n_sims, cell["theta_star"]),
        "psi_true": np.full(cfg.n_sims, cell["psi_true"], dtype=object),
        "alpha": np.full(cfg.n_sims, cell["alpha"]),
        "c_switch": np.full(cfg.n_sims, cfg.c_switch),
        "seed": seeds,
        "tau": taus,
    })
    _rs._write_parquet_atomic(sims_df, _rs._sims_shard_path(cfg.out_dir, cell_id))
    n_bytes = _rs._write_parquet_atomic(pd.DataFrame(data),
                                        _rs._shard_path(cfg.out_dir, cell_id))
    return {
        "cell_id": cell_id, "theta_star": cell["theta_star"],
        "psi_true": cell["psi_true"], "alpha": cell["alpha"],
        "rows": n_rows, "bytes": n_bytes,
        "switch_rate": float((taus >= 0).mean()),
        "median_tau": float(np.median(taus[taus >= 0])) if (taus >= 0).any() else float("nan"),
        "wall_s": round(time.time() - t0, 2),
    }


# ---------------------------------------------------------------------------
# Driver (mirrors run_sweep.run_sweep)
# ---------------------------------------------------------------------------

def run(cfg: Config, resume=True, max_inflight=0, min_free_gb=1.0):
    cells = build_cell_grid()
    n_workers = cfg.workers if cfg.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    max_inflight = max_inflight or 2 * n_workers

    n_stale = _rs._clear_stale_tmp(cfg.out_dir)
    if n_stale:
        print(f"removed {n_stale} stale .tmp shard(s)", flush=True)
    completed = _rs._completed_cells(cfg.out_dir, cfg) if resume else set()
    pending = [c for c in cells if c["cell_id"] not in completed]
    avail, _ = _rs._mem_stats()
    print(f"cells total={len(cells)} already_done={len(completed)} "
          f"to_run={len(pending)} workers={n_workers} free_ram={avail:.1f}GB",
          flush=True)
    cfg_dict = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()}

    failures, stopped_early = [], None
    t_start = time.time()
    if not pending:
        return 0.0, failures, None
    queue = iter(pending)
    n_done, n_target = 0, len(pending)
    try:
        ex = ProcessPoolExecutor(max_workers=n_workers, max_tasks_per_child=8)
    except TypeError:
        ex = ProcessPoolExecutor(max_workers=n_workers)
    try:
        inflight = {}

        def submit_next():
            nonlocal stopped_early
            if stopped_early is not None:
                return False
            c = next(queue, None)
            if c is None:
                return False
            inflight[ex.submit(run_cell, (c, cfg_dict))] = c
            return True

        for _ in range(max_inflight):
            if not submit_next():
                break
        while inflight:
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:
                cell = inflight.pop(fut)
                try:
                    summary = fut.result()
                except Exception as exc:
                    failures.append({"cell_id": cell["cell_id"],
                                     "error": f"{type(exc).__name__}: {exc}"})
                    print(f"  cell {cell['cell_id']:03d} FAILED: {exc}", flush=True)
                    summary = None
                finally:
                    del fut
                n_done += 1
                if summary is not None:
                    _rs._journal(cfg.out_dir, summary)
                avail, rss = _rs._mem_stats()
                el = time.time() - t_start
                eta = (n_target - n_done) / (n_done / el) if n_done else float("inf")
                info = summary or cell
                extra = (f"switch={summary['switch_rate']:.2f} "
                         f"tau~{summary['median_tau']:.0f}  " if summary else "")
                print(f"  cell {info['cell_id']:03d} done  theta={info['theta_star']:.1f} "
                      f"psi={info['psi_true']:<5} alpha={info['alpha']:<5} {extra}"
                      f"[{n_done}/{n_target}] elapsed={el:.0f}s eta={eta:.0f}s "
                      f"rss={rss:.2f}GB free={avail:.1f}GB", flush=True)
                if min_free_gb > 0 and avail < min_free_gb and stopped_early is None:
                    stopped_early = f"free RAM {avail:.2f} GB < {min_free_gb} GB"
                    print(f"\n!! {stopped_early}; draining in-flight cells then "
                          f"exiting cleanly. Rerun to resume.\n", flush=True)
                submit_next()
    except KeyboardInterrupt:
        stopped_early = "interrupted (Ctrl-C)"
        print(f"\n!! {stopped_early}; rerun the same command to resume.\n", flush=True)
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    except BrokenExecutor as exc:
        stopped_early = f"worker pool broke: {exc}"
        print(f"\n!! {stopped_early}; rerun to resume.\n", flush=True)
    finally:
        ex.shutdown(wait=True)
    return time.time() - t_start, failures, stopped_early


def write_run_config(cfg, wall, status, extra=None):
    info = {
        "status": status,
        "workers": cfg.workers if cfg.workers > 0 else max(1, (os.cpu_count() or 2) - 1),
        "grid": {"theta_stars": list(THETA_STARS), "psi_trues": list(PSI_TRUES),
                 "alphas": list(ALPHAS), "n": cfg.n, "m": cfg.m,
                 "rounds": cfg.rounds, "n_sims_per_cell": cfg.n_sims,
                 "n_cells": len(build_cell_grid())},
        "policies": list(POLICIES),
        "switch": {"score": "sus_1", "c": cfg.c_switch, "types": ["soft", "hard"]},
        "seed_base": cfg.seed,
        "theta_space": list(cfg.theta_space),
        "psi_label_map": PSI_TO_CODE,
        "git_hash": _rs.git_hash(),
        "wall_seconds": wall,
        "run_date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        info.update(extra)
    path = os.path.join(cfg.out_dir, "run_config.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())


# ---------------------------------------------------------------------------
# Sanity checks (streamed, one shard at a time)
# ---------------------------------------------------------------------------

def run_sanity(cfg: Config, sweep_dir: str | None = None):
    thetas = list(cfg.theta_space)
    n_theta = len(thetas)
    lines = ["# sanity.md", "", "Automated checks from `run_switching.py`.", ""]
    n_rows = n_shards = 0
    bad_sum = {p: 0 for p in POLICIES}
    bad_psi = {p: 0 for p in PSI_POLICIES}
    pre_mismatch = 0
    flag_mismatch = 0
    nonfinite_theta = 0
    null_switch = [0, 0]
    alt_switch = [0, 0]
    sweep_mismatch, sweep_checked = 0, 0
    sweep_max_err = 0.0

    for cid, path in _rs.iter_shards(cfg.out_dir):
        d = pd.read_parquet(path)
        n_shards += 1
        n_rows += len(d)
        for pol in POLICIES:
            M = d[[f"{pol}_theta_{i}" for i in range(n_theta)]].to_numpy()
            nonfinite_theta += int((~np.isfinite(M)).sum())
            bad_sum[pol] += int((np.abs(M.sum(axis=1) - 1) > 1e-6).sum())
        sw = d["switched"].to_numpy().astype(bool)
        for pol in PSI_POLICIES:
            P = d[[f"{pol}_psi_{p}" for p in PSIS]].to_numpy()
            has = ~np.isnan(P).any(axis=1)
            bad_psi[pol] += int((np.abs(P[has].sum(axis=1) - 1) > 1e-6).sum())
            if pol != "vig":
                # psi must be NaN exactly when not switched
                flag_mismatch += int((has != sw).sum())
        # before the switch, soft == hard == cred exactly
        C = d[[f"cred_theta_{i}" for i in range(n_theta)]].to_numpy()
        for pol in ("soft", "hard"):
            M = d[[f"{pol}_theta_{i}" for i in range(n_theta)]].to_numpy()
            pre_mismatch += int((np.abs(M[~sw] - C[~sw]).max(axis=1) > 1e-12).sum())
        sims = pd.read_parquet(_rs._sims_shard_path(cfg.out_dir, cid))
        fired = (sims["tau"] >= 0)
        tgt = null_switch if sims["psi_true"].iloc[0] == "inf" else alt_switch
        tgt[0] += int(fired.sum()); tgt[1] += len(sims)

        if sweep_dir is not None:
            sp = os.path.join(sweep_dir, "trajectories", f"cell={cid:04d}", "part.parquet")
            if os.path.isfile(sp):
                s = pd.read_parquet(sp, columns=["sim_id", "round"]
                                    + [f"L1_theta_{i}" for i in range(n_theta)])
                s = s[(s["sim_id"] < cfg.n_sims) & (s["round"] <= cfg.rounds)]
                if len(s) == len(d):
                    # sweep stores the PRE-update belief; here it is post-update,
                    # so the sweep's round t+1 equals our round t.
                    S = s[[f"L1_theta_{i}" for i in range(n_theta)]].to_numpy()
                    S = S.reshape(-1, cfg.rounds, n_theta)[:, 1:, :]
                    Cr = C.reshape(-1, cfg.rounds, n_theta)[:, :-1, :]
                    err = float(np.abs(S - Cr).max())
                    sweep_max_err = max(sweep_max_err, err)
                    sweep_checked += 1
                    sweep_mismatch += int(err > 1e-9)
        del d

    expected = len(build_cell_grid()) * cfg.n_sims * cfg.rounds
    ok = []
    def add(flag, text):
        ok.append(flag); lines.append(f"{text}  => {'PASS' if flag else 'FAIL'}")
    add(n_rows == expected, f"**1. Row count.** expected={expected:,} actual={n_rows:,}")
    add(nonfinite_theta == 0, f"**2. Theta marginals finite.** non-finite={nonfinite_theta}")
    add(all(v == 0 for v in bad_sum.values()),
        f"**3. Theta marginals sum to 1** (1e-6). violations={bad_sum}")
    add(all(v == 0 for v in bad_psi.values()),
        f"**4. Psi marginals sum to 1 where present.** violations={bad_psi}")
    add(flag_mismatch == 0,
        f"**5. `switched` flag consistent with psi presence.** mismatches={flag_mismatch}")
    add(pre_mismatch == 0,
        f"**6. Before tau, soft == hard == credulous exactly.** mismatches={pre_mismatch}")
    ns = null_switch[0] / null_switch[1] if null_switch[1] else float("nan")
    as_ = alt_switch[0] / alt_switch[1] if alt_switch[1] else float("nan")
    lines.append(f"**7. Switch rate** at c={cfg.c_switch}: null {ns:.3f} "
                 f"({null_switch[0]}/{null_switch[1]}), persuasive {as_:.3f} "
                 f"({alt_switch[0]}/{alt_switch[1]}). Compare the sweep's FPR/TPR "
                 f"at this c.")
    if sweep_dir is not None:
        add(sweep_checked > 0 and sweep_mismatch == 0,
            f"**8. Credulous trajectory == sweep `L1_theta` on shared seeds.** "
            f"cells checked={sweep_checked} mismatching={sweep_mismatch} "
            f"max|diff|={sweep_max_err:.2e}")
    lines.insert(3, f"**Overall: {'PASS' if all(ok) else 'FAIL'}**")
    lines.insert(4, "")
    lines.append("")
    lines.append(f"- trajectories: {n_rows:,} rows across {n_shards} shards")
    report = "\n".join(lines)
    with open(os.path.join(cfg.out_dir, "sanity.md"), "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report)
    return all(ok)


# ---------------------------------------------------------------------------
# Time trial
# ---------------------------------------------------------------------------

def time_trial(cfg: Config, n_sims=3):
    """Single-process per-sim cost on representative cells, then a projection."""
    thetas = list(cfg.theta_space)
    n_theta = len(thetas)
    probes = [("inf", 1.5), ("inf", 20.0), ("pers+", 3.0), ("pers+", 10.0),
              ("pers-", 20.0)]
    print(f"time trial: {n_sims} sims x {cfg.rounds} rounds per probe, c={cfg.c_switch}")
    per_sim = {}
    for psi, alpha in probes:
        cell = {"cell_id": 0, "theta_star": 0.5, "psi_true": psi, "alpha": alpha}
        out = {c: np.empty(n_sims * cfg.rounds) for c in column_names(n_theta)
               if c not in ("theta_star", "psi_true", "alpha")}
        t0 = time.time()
        taus = [_run_one_sim(cell, s, cfg, s * cfg.rounds, out, thetas)[1]
                for s in range(n_sims)]
        dt = (time.time() - t0) / n_sims
        per_sim[(psi, alpha)] = dt
        print(f"  psi={psi:<5} alpha={alpha:<5}  {dt:6.2f} s/sim   taus={taus}")
    null_cost = np.mean([v for (p, _), v in per_sim.items() if p == "inf"])
    alt_cost = np.mean([v for (p, _), v in per_sim.items() if p != "inf"])
    n_cells = len(build_cell_grid())
    n_null = sum(1 for c in build_cell_grid() if c["psi_true"] == "inf")
    n_workers = cfg.workers if cfg.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    total = (n_null * null_cost + (n_cells - n_null) * alt_cost) * cfg.n_sims
    # parallel efficiency observed on this machine for the sweep was ~0.85
    wall = total / n_workers / 0.85
    print(f"\nprojection for grid={n_cells} cells x {cfg.n_sims} sims on {n_workers} workers:")
    print(f"  null {null_cost:.2f} s/sim, persuasive {alt_cost:.2f} s/sim")
    print(f"  total worker-time {total/3600:.1f} h  ->  wall ~{wall/3600:.1f} h")
    return per_sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--time_trial", action="store_true")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--n_sims", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--c_switch", type=float, default=None)
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--skip_sanity", action="store_true")
    ap.add_argument("--sanity_only", action="store_true")
    ap.add_argument("--check_against_sweep", default="results/full_sweep_v2",
                    help="sweep dir whose L1_theta the credulous policy must "
                         "reproduce on shared seeds ('' to skip)")
    ap.add_argument("--max_inflight", type=int, default=0)
    ap.add_argument("--min_free_gb", type=float, default=1.0)
    ap.add_argument("--only_alpha", type=float, nargs="+", default=None)
    ap.add_argument("--only_psi", type=str, nargs="+", default=None)
    ap.add_argument("--only_theta", type=float, nargs="+", default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.smoke:
        cfg.n_sims, cfg.rounds, cfg.out_dir = 4, 30, "results/switching_smoke"
    if args.out_dir: cfg.out_dir = args.out_dir
    if args.workers: cfg.workers = args.workers
    if args.n_sims is not None: cfg.n_sims = args.n_sims
    if args.rounds is not None: cfg.rounds = args.rounds
    if args.c_switch is not None: cfg.c_switch = args.c_switch

    global THETA_STARS, PSI_TRUES, ALPHAS
    if args.only_theta is not None: THETA_STARS = tuple(float(x) for x in args.only_theta)
    if args.only_psi is not None: PSI_TRUES = tuple(args.only_psi)
    if args.only_alpha is not None: ALPHAS = tuple(float(x) for x in args.only_alpha)

    if args.time_trial:
        time_trial(cfg)
        return

    os.makedirs(cfg.out_dir, exist_ok=True)
    sweep = args.check_against_sweep or None
    if sweep and not os.path.isdir(os.path.join(sweep, "trajectories")):
        sweep = None
    # cell ids only line up with the sweep's on the unfiltered grid
    if sweep and (THETA_STARS, PSI_TRUES, ALPHAS) != (_rs.THETA_STARS, _rs.PSI_TRUES, _rs.ALPHAS):
        sweep = None
    if args.sanity_only:
        sys.exit(0 if run_sanity(cfg, sweep) else 1)

    print(f"cfg: cells={len(build_cell_grid())} n_sims/cell={cfg.n_sims} "
          f"rounds={cfg.rounds} c_switch={cfg.c_switch} out_dir={cfg.out_dir}", flush=True)
    write_run_config(cfg, 0.0, "running")
    wall, failures, stopped = run(cfg, resume=not args.no_resume,
                                  max_inflight=args.max_inflight,
                                  min_free_gb=args.min_free_gb)
    status = "incomplete" if stopped else ("partial" if failures else "complete")
    write_run_config(cfg, wall, status, {"failed_cells": failures, "stopped_early": stopped})
    print(f"total wall: {wall:.0f}s  status={status}")
    if not args.skip_sanity:
        run_sanity(cfg, sweep)
    if stopped or failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
