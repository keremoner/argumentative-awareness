"""
Recompute the sus_1 null variance offline, without re-simulating.

Why this is possible
--------------------
The exact sus_1 variance for a round is a function of the listener state alone:

    V = sum_u P_pred(u) s(u)^2,
    s(u) = sum_O W[O,u] B[O,u],
    W    = colnorm( S1(u|O,inf) * L1(O) ),
    B    = -log S1(u|O,inf) - H(S1(.|O,inf)),

so it needs exactly two things: ``L1(theta)`` and the table ``S1(u|O,inf)``.

``ScoreContext.S1_table`` always asks the speaker for ``psi="inf"``, and in that
branch ``Speaker1`` sets ``beta = 1``, which drops the persuasiveness factor
entirely:

    S1(u|O,inf)  ~  Truth(u;O) * Informativeness(O,u)^alpha,
    Informativeness(O,u) = P_L0(O | u).

That depends only on ``alpha`` and on **L0's** belief over theta -- never on the
speaker's own ``belief_theta``, and never on the cell's true psi.  The sweep
stores ``L1_theta_*`` and ``L0_theta_*`` every round, so the whole context is
recoverable from the parquet and the variance can be recomputed directly.

This is checked rather than assumed: for every row the script recomputes the
sus_1 *score* from the reconstructed context and compares it against the stored
value, aborting the cell if they ever differ by more than ``SCORE_TOL``.  A
matching score means the reconstructed S1 table and weighting matrix are the
ones the simulation actually used, so the variance built from them is right.

What changes and what does not
------------------------------
Only the sus_1 variance columns are recomputed.  Scores are carried over
verbatim from the input sweep, so a before/after comparison isolates the
variance change by construction.  ``surp2`` is untouched: its variance was
already the exact state-only varentropy.

    dropped : sus1_sigma2_naive, sus1_sigma2_corrected,
              sus1_sigma_bar2_naive, sus1_sigma_bar2_corrected
    added   : sus1_sigma2, sus1_sigma_bar2

Usage
-----
    python experiments/full_sweep/recompute_variance.py \
        --in_dir results/full_sweep --out_dir results/full_sweep_v2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rsa.setup import make_thetas, make_world, make_semantics
from rsa.detection.scores import _exact_score_variance, _safe_log

# Reuse the sweep's own schema and atomic-write helpers so the output is
# byte-for-byte the same shape run_sweep.py would have produced.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_sweep as _rs  # noqa: E402

SCORE_COLS = _rs.SCORE_COLS
git_hash = _rs.git_hash
shard_path = _rs._shard_path
sims_shard_path = _rs._sims_shard_path
write_parquet_atomic = _rs._write_parquet_atomic

SCORE_TOL = 1e-9


def recompute_cell(task):
    """Recompute one cell's sus_1 variance columns; return a summary dict."""
    cell_id, in_dir, out_dir, n_sims, rounds = task
    t0 = time.time()

    src = os.path.join(in_dir, "trajectories", f"cell={cell_id:04d}", "part.parquet")
    d = pd.read_parquet(src)

    thetas = make_thetas(0.1, True, True)
    n_theta = len(thetas)
    l1_cols = [f"L1_theta_{i}" for i in range(n_theta)]
    l0_cols = [f"L0_theta_{i}" for i in range(n_theta)]

    alpha = float(d["alpha"].iloc[0])
    theta_star = float(d["theta_star"].iloc[0])
    n = int(d["n"].iloc[0])
    m = int(d["m"].iloc[0])

    world = make_world(theta_star, n=n, m=m)
    semantics = make_semantics(n=n)

    # Fixed tables for this cell.  S0 is uniform over literally true utterances
    # and carries no state, so P_S0(u|O) and the truth table never move.
    truth = semantics.truth_table(world).astype(float)      # (n_obs, n_utt)
    T_O = truth.sum(axis=1)
    P_S0 = truth / T_O[:, None]
    obs_tab = np.asarray(world.obs_prob_table(thetas), dtype=float)

    def s1_table(l0_theta):
        """S1(u|O,inf) for a given L0 belief.

        Reproduces Speaker1._dist_over_utterances_obs_array for psi='inf',
        where beta=1 drops persuasiveness and the row reduces to
        Truth(u;O) * P_L0(O|u)^alpha.  Written directly in numpy because the
        object path rebuilds and re-caches the whole RSA stack per round; this
        is ~30x faster and agrees to 2e-16.  Correctness is not assumed -- the
        per-row score check below re-derives the stored sus1_score from this
        table for every single row.
        """
        pO = obs_tab @ l0_theta
        pO = pO / pO.sum()
        num = P_S0 * pO[:, None]
        den = num.sum(axis=0, keepdims=True)
        info = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        sc = truth * np.power(info, alpha, where=info > 0,
                              out=np.zeros_like(info))
        rs = sc.sum(axis=1, keepdims=True)
        out = np.divide(sc, rs, out=np.zeros_like(sc), where=rs > 0)
        bad = rs[:, 0] == 0
        if bad.any():                       # no truthful utterance carries mass
            out[bad] = truth[bad] / truth[bad].sum(axis=1, keepdims=True)
        return out

    L1 = d[l1_cols].to_numpy(dtype=float)
    L0 = d[l0_cols].to_numpy(dtype=float)
    U = d["u_observed"].to_numpy()
    stored_score = d["sus1_score"].to_numpy(dtype=float)

    n_rows = len(d)
    sigma2 = np.empty(n_rows, dtype=np.float64)
    max_score_err = 0.0

    for k in range(n_rows):
        S1 = s1_table(L0[k])
        log_S1 = _safe_log(S1)
        H_O = -np.sum(S1 * log_S1, axis=1)
        B = -log_S1 - H_O[:, None]

        L1_O = obs_tab @ L1[k]
        L1_O = L1_O / L1_O.sum()

        num = S1 * L1_O[:, None]
        den = num.sum(axis=0, keepdims=True)
        W = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

        p_pred = S1.T @ L1_O
        p_pred = p_pred / p_pred.sum()

        s_all = np.sum(W * B, axis=0)
        err = abs(float(s_all[int(U[k])]) - stored_score[k])
        if err > max_score_err:
            max_score_err = err
        sigma2[k] = _exact_score_variance(W, B, p_pred)

    if max_score_err > SCORE_TOL:
        raise RuntimeError(
            f"cell {cell_id}: reconstructed sus1_score differs from stored by "
            f"{max_score_err:.3e} (> {SCORE_TOL}). The listener state could not "
            f"be reproduced, so the recomputed variance is not trustworthy."
        )

    if (sigma2 < 0).any():
        raise RuntimeError(f"cell {cell_id}: negative exact variance produced")

    # sigma_bar^2(t) = (1/t) sum_{i<=t} sigma^2_i, restarted per simulation.
    sig = sigma2.reshape(n_sims, rounds)
    t_axis = np.arange(1, rounds + 1, dtype=np.float64)
    sigma_bar2 = (np.cumsum(sig, axis=1) / t_axis).reshape(-1)

    # Rebuild in run_sweep's on-disk column order.
    out = {c: d[c].to_numpy() for c in
           ("cell_id", "sim_id", "round", "theta_star", "psi_true", "alpha",
            "n", "m", "speaker_level", "u_observed", "O_true_idx",
            "O_true_count")}
    carried = {"surp2_score", "surp2_sigma2", "surp2_Sus", "surp2_sigma_bar2",
               "sus1_score", "sus1_Sus"}
    for c in SCORE_COLS:
        if c == "sus1_sigma2":
            out[c] = sigma2
        elif c == "sus1_sigma_bar2":
            out[c] = sigma_bar2
        elif c in carried:
            out[c] = d[c].to_numpy()
        else:
            raise KeyError(f"unhandled score column {c!r}")
    for i in range(n_theta):
        out[f"L1_theta_{i}"] = d[f"L1_theta_{i}"].to_numpy()
        out[f"L0_theta_{i}"] = d[f"L0_theta_{i}"].to_numpy()

    n_bytes = write_parquet_atomic(pd.DataFrame(out), shard_path(out_dir, cell_id))

    # simulations/ is metadata only -- copy it across unchanged.
    src_sims = os.path.join(in_dir, "simulations", f"cell={cell_id:04d}",
                            "part.parquet")
    if os.path.isfile(src_sims):
        dst_sims = sims_shard_path(out_dir, cell_id)
        tmp = dst_sims + ".tmp"
        shutil.copyfile(src_sims, tmp)
        os.replace(tmp, dst_sims)

    return {
        "cell_id": cell_id,
        "rows": n_rows,
        "bytes": n_bytes,
        "max_score_err": max_score_err,
        "mean_sigma2": float(sigma2.mean()),
        "wall_s": round(time.time() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="results/full_sweep")
    ap.add_argument("--out_dir", default="results/full_sweep_v2")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit_cells", type=int, default=0,
                    help="only process the first N cells (smoke test)")
    args = ap.parse_args()

    traj_root = os.path.join(args.in_dir, "trajectories")
    cells = sorted(int(n.split("=")[1]) for n in os.listdir(traj_root)
                   if n.startswith("cell="))
    if args.limit_cells:
        cells = cells[:args.limit_cells]

    src_cfg_path = os.path.join(args.in_dir, "run_config.json")
    with open(src_cfg_path, encoding="utf-8") as fh:
        src_cfg = json.load(fh)
    rounds = int(src_cfg["grid"]["rounds"])
    n_sims = int(src_cfg["grid"]["n_sims_per_cell"])

    # Cells already done are skipped, so an interrupted run just resumes.
    done = set()
    out_traj = os.path.join(args.out_dir, "trajectories")
    if os.path.isdir(out_traj):
        for name in os.listdir(out_traj):
            p = os.path.join(out_traj, name, "part.parquet")
            if name.startswith("cell=") and os.path.isfile(p):
                done.add(int(name.split("=")[1]))
    pending = [c for c in cells if c not in done]

    n_workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"cells total={len(cells)} done={len(done)} to_do={len(pending)} "
          f"workers={n_workers}", flush=True)

    t_start = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        it = iter(pending)
        inflight = {}
        for _ in range(2 * n_workers):
            c = next(it, None)
            if c is None:
                break
            inflight[ex.submit(recompute_cell,
                               (c, args.in_dir, args.out_dir, n_sims, rounds))] = c
        n_done = 0
        while inflight:
            fin, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in fin:
                cid = inflight.pop(fut)
                r = fut.result()          # a failed reconstruction aborts loudly
                del fut
                results.append(r)
                n_done += 1
                el = time.time() - t_start
                eta = (len(pending) - n_done) / (n_done / el) if n_done else float("inf")
                print(f"  cell {cid:03d}  rows={r['rows']:,}  "
                      f"score_err={r['max_score_err']:.1e}  "
                      f"[{n_done}/{len(pending)}]  elapsed={el:.0f}s "
                      f"eta={eta:.0f}s", flush=True)
                nxt = next(it, None)
                if nxt is not None:
                    inflight[ex.submit(
                        recompute_cell,
                        (nxt, args.in_dir, args.out_dir, n_sims, rounds))] = nxt

    wall = time.time() - t_start
    worst = max((r["max_score_err"] for r in results), default=0.0)

    cfg = dict(src_cfg)
    cfg.update({
        "status": "complete",
        "derived_from": os.path.abspath(args.in_dir),
        "derivation": (
            "sus_1 null variance recomputed offline from the stored per-round "
            "L1_theta/L0_theta; scores carried over unchanged from the source "
            "sweep. No re-simulation."),
        "max_score_reconstruction_error": worst,
        "git_hash": git_hash(),
        "wall_seconds": wall,
        "run_date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    with open(os.path.join(args.out_dir, "run_config.json"), "w",
              encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    print(f"\ndone in {wall:.0f}s")
    print(f"worst score reconstruction error across all cells: {worst:.3e}")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
