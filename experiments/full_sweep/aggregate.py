"""
Aggregate the full sweep (``results/full_sweep_v2``) into the tables that
``notebooks/sweep_analysis.ipynb`` reads.

The sweep is 297 cells x 200 sims x 150 rounds of an S1 speaker heard by a
credulous L1 whose detector runs as a pure observer (c = inf), recording two
scores (``surp2``, ``sus1``) and the *full* L1 and L0 posteriors over an
11-point theta space every round.  8.9M rows / 2.2 GB -- too much to explore
directly, and a stale summary once produced wrong numbers in this project, so
every table below is rebuilt from the shards on each run (about a minute).

Conventions are aligned with ``experiments/switching/aggregate.py`` so the two
aggregate sets can be joined: ``psi_star`` uses the switching labels
(``inf`` / ``high`` / ``low``), and ``bias_*`` means E[theta] - theta*.

Outputs (in ``results/full_sweep_v2/agg/``)
------------------------------------------
``rounds.parquet``
    Per (theta*, psi*, alpha, round): for each score the mean and sd of the
    per-round score, mean analytic sigma^2, mean and sd of Sus(t), mean
    sigma_bar; for L1 and L0 the mean signed bias, |bias|, squared error,
    posterior sd, entropy, mass on the true theta, and 2-sd coverage.

``runs.parquet``
    One row per simulation: tau of the fixed-z rule for every (score, c) in
    ``C_LIST``; Sus(150) per score; L1 bias at rounds 25/50/100/150 and at
    tau - 1 for every (score, c).  Run-level distributions and "damage before
    detection" come from here.

``panels_{S,V}_{score}.parquet``
    Dense (n_sims, 150) matrices of S_t = t * Sus(t) and V_t = t * sigma_bar^2(t),
    one row per simulation in ``runs.parquet`` order.  Any stopping-rule
    boundary B(t, V_t) can be scored against them in milliseconds; this is how
    the stopping-rule section evaluates Robbins / LIL / warm-up / empirical
    rules without re-simulating.

``utterances.parquet``
    Counts per (theta*, psi*, alpha, obs_count, utt).

Usage
-----
    python experiments/full_sweep/aggregate.py
    python experiments/full_sweep/aggregate.py --workers 8 --sweep_dir results/full_sweep_v2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

SCORES = ("surp2", "sus1")
PSI_MAP = {"inf": "inf", "pers+": "high", "pers-": "low"}
C_LIST = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
HORIZONS = (25, 50, 100, 150)
T_MAX = 150
N_SIMS = 200
LISTENERS = ("L1", "L0")


def _c_tag(c):
    return f"{c:g}"


# ---------------------------------------------------------------------------
# Per-shard work
# ---------------------------------------------------------------------------

def _posterior_stats(P, thetas, theta_star):
    """Per-row summaries of a posterior matrix P (rows sum to 1)."""
    m = P @ thetas
    v = P @ (thetas ** 2) - m ** 2
    sd = np.sqrt(np.maximum(v, 0.0))
    bias = m - theta_star
    ent = -np.sum(P * np.log(np.maximum(P, 1e-300)), axis=1)
    j = int(np.argmin(np.abs(thetas - theta_star)))
    return dict(bias=bias, abias=np.abs(bias), sqerr=bias * bias, sd=sd, entropy=ent,
                p_true=P[:, j], cov=(np.abs(bias) <= 2.0 * sd).astype(float))


def summarize_shard(task):
    path, thetas = task
    thetas = np.asarray(thetas, dtype=float)
    n_theta = len(thetas)
    df = pd.read_parquet(path)
    df = df.sort_values(["sim_id", "round"], kind="stable").reset_index(drop=True)
    n_sims = df["sim_id"].nunique()
    T = int(df["round"].max())
    rnd = df["round"].to_numpy().reshape(n_sims, T)
    if not (rnd == np.arange(1, T + 1)).all():
        raise ValueError(f"{path}: rows are not sim-major / round-minor")

    theta_star = float(df["theta_star"].iloc[0])
    psi = PSI_MAP[str(df["psi_true"].iloc[0])]
    alpha = float(df["alpha"].iloc[0])
    key = dict(theta_star=theta_star, psi_star=psi, alpha=alpha)

    # ---- posterior summaries ---------------------------------------------
    for L in LISTENERS:
        P = df[[f"{L}_theta_{i}" for i in range(n_theta)]].to_numpy(dtype=float)
        st = _posterior_stats(P, thetas, theta_star)
        for k, v in st.items():
            df[f"{L}_{k}"] = v

    for sc in SCORES:
        df[f"{sc}_sigma_bar"] = np.sqrt(np.maximum(df[f"{sc}_sigma_bar2"].to_numpy(dtype=float), 0.0))

    # ---- rounds ------------------------------------------------------------
    mean_cols = ([f"{sc}_{q}" for sc in SCORES for q in ("score", "sigma2", "Sus", "sigma_bar")]
                 + [f"{L}_{q}" for L in LISTENERS
                    for q in ("bias", "abias", "sqerr", "sd", "entropy", "p_true", "cov")])
    sd_cols = [f"{sc}_{q}" for sc in SCORES for q in ("score", "Sus")] + ["L1_bias", "L0_bias"]
    g = df.groupby("round")
    rounds = g[mean_cols].mean()
    sd = g[sd_cols].std(ddof=1)
    sd.columns = [f"{c}_sd" for c in sd.columns]
    rounds = rounds.join(sd)
    rounds["n"] = g.size()
    rounds = rounds.reset_index()

    # ---- dense panels + taus -------------------------------------------------
    t = np.arange(1, T + 1, dtype=float)
    panels = {}
    runs = pd.DataFrame({"sim": np.arange(n_sims)})
    L1_bias = df["L1_bias"].to_numpy().reshape(n_sims, T)
    for sc in SCORES:
        S = df[f"{sc}_Sus"].to_numpy(dtype=float).reshape(n_sims, T) * t
        V = df[f"{sc}_sigma_bar2"].to_numpy(dtype=float).reshape(n_sims, T) * t
        panels[f"S_{sc}"] = S
        panels[f"V_{sc}"] = V
        runs[f"Sus150_{sc}"] = S[:, -1] / T
        for c in C_LIST:
            crossed = S > c * np.sqrt(np.maximum(V, 0.0))
            fired = crossed.any(axis=1)
            tau = np.where(fired, crossed.argmax(axis=1) + 1, -1)
            runs[f"tau_{sc}_c{_c_tag(c)}"] = tau.astype(np.int32)
            # credulous bias just before the alarm (NaN when no alarm or tau == 1)
            idx = tau - 2
            ok = fired & (idx >= 0)
            dmg = np.full(n_sims, np.nan)
            dmg[ok] = L1_bias[np.arange(n_sims)[ok], idx[ok]]
            runs[f"dmg_{sc}_c{_c_tag(c)}"] = dmg
    for h in HORIZONS:
        runs[f"L1_bias_t{h}"] = L1_bias[:, h - 1]
    runs["seed_sim_id"] = df["sim_id"].to_numpy().reshape(n_sims, T)[:, 0]

    # ---- utterances ------------------------------------------------------------
    utts = (df.groupby(["O_true_count", "u_observed"]).size().rename("n").reset_index()
              .rename(columns={"O_true_count": "obs_count", "u_observed": "utt"}))

    for frame in (rounds, runs, utts):
        for k, v in reversed(list(key.items())):
            frame.insert(0, k, v)
    return rounds, runs, utts, panels, key


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", default=os.path.join("results", "full_sweep_v2"))
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.sweep_dir, "agg")
    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    os.makedirs(out_dir, exist_ok=True)

    cfg = json.load(open(os.path.join(args.sweep_dir, "run_config.json"), encoding="utf-8"))
    thetas = list(cfg["theta_space"])
    tdir = os.path.join(args.sweep_dir, "trajectories")
    shards = sorted(os.path.join(tdir, d, "part.parquet") for d in os.listdir(tdir)
                    if d.startswith("cell=") and os.path.isfile(os.path.join(tdir, d, "part.parquet")))
    print(f"{len(shards)} shards on {workers} workers; theta space {thetas}")

    t0 = time.time()
    results = [None] * len(shards)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(summarize_shard, (p, thetas)): i for i, p in enumerate(shards)}
        done = 0
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
            done += 1
            if done % 50 == 0 or done == len(shards):
                el = time.time() - t0
                print(f"  {done}/{len(shards)}  {el:.0f}s", flush=True)

    # keep shard (cell) order so panel rows line up with runs.parquet rows
    R = pd.concat([r[0] for r in results], ignore_index=True)
    RU = pd.concat([r[1] for r in results], ignore_index=True)
    UT = pd.concat([r[2] for r in results], ignore_index=True)
    UT = UT.groupby(["theta_star", "psi_star", "alpha", "obs_count", "utt"], as_index=False)["n"].sum()

    keep = {"theta_star", "alpha"}
    for df in (R,):
        f64 = [c for c in df.columns if df[c].dtype == np.float64 and c not in keep]
        df[f64] = df[f64].astype(np.float32)

    R.to_parquet(os.path.join(out_dir, "rounds.parquet"), index=False)
    RU.to_parquet(os.path.join(out_dir, "runs.parquet"), index=False)
    UT.to_parquet(os.path.join(out_dir, "utterances.parquet"), index=False)
    for name in ("S_surp2", "V_surp2", "S_sus1", "V_sus1"):
        M = np.vstack([r[3][name] for r in results])
        pd.DataFrame(M, columns=[f"t{i}" for i in range(1, M.shape[1] + 1)]).to_parquet(
            os.path.join(out_dir, f"panels_{name}.parquet"), index=False)
    for name in ("rounds", "runs", "utterances"):
        p = os.path.join(out_dir, f"{name}.parquet")
        print(f"  wrote {name}.parquet  {os.path.getsize(p)/1e6:.1f} MB")
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(sweep_dir=args.sweep_dir, git_hash=cfg["git_hash"], theta_space=thetas,
                       scores=list(SCORES), c_list=list(C_LIST), horizons=list(HORIZONS),
                       psi_map=PSI_MAP, n_shards=len(shards), n_rows_runs=int(len(RU)),
                       grid=cfg["grid"], built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                  fh, indent=1)
    print(f"done in {(time.time()-t0)/60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
