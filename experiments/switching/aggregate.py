"""
Aggregate the switching datasets into small tables the analysis notebook reads.

The raw trajectories are ~5 GB / 130M rows across four studies, far too much to
explore interactively.  Every figure in ``notebooks/switching_analysis.ipynb``
is built from the parquet tables written here (a few MB in total), so the
notebook re-runs in seconds.  Raw rows are only needed for per-cell deep dives,
for which the notebook has a ``load_cell`` helper.

One (study, theta*, psi*, alpha, c, switch_type, round) group always lives
inside a single cell shard -- offline studies vary (c, switch_type) within a
cell, the feedback study puts them in the cell key -- so each shard can be
summarised independently and the pieces concatenated.  That makes the whole
pass embarrassingly parallel.

Outputs (in ``results/switching/agg/``)
--------------------------------------
``rounds.parquet``
    Per (study, theta*, psi*, alpha, c, switch_type, round): means and spreads
    of every listener's bias / |bias| / squared error / posterior sd / 2-sd
    coverage, the psi-marginals, the detector state (score, running mean,
    running sigma, boundary, crossed, switched) and the *paired* differences
    between the switching listener and the two references.  ``n`` is the number
    of simulations; ``*_sd`` columns divided by sqrt(n) give standard errors.
    NB: psi-marginal means skip NaN, i.e. they are conditional on having
    switched; ``switched`` gives the mass they are conditional on.

``aligned.parquet``
    The same bias / psi quantities indexed by ``k = round - tau`` (the switch
    is at k = 0), for runs that switched.  This is what post-switch recovery
    is read from.

``runs.parquet``
    One row per (study, cell, condition, simulation, horizon), where horizon is
    round 25/50/100/150 or the run's own ``tau`` / ``tau - 1``: each listener's
    bias and posterior sd at that moment, plus ``tau``.  Run-level
    distributions -- quantiles, histograms, scatter against tau, "how far had
    the persuader got by the time it was caught" -- come from here.  The
    round-averaged table above cannot answer those.

``utterances.parquet``
    Utterance frequencies per (study, theta*, psi*, alpha, c, switch_type,
    obs_count, utt), split by whether the listener had already switched.  The
    speaker-policy figures (Fang's Figure 1 analogue) come from here.  For the
    offline studies the stream does not depend on (c, switch_type), so only one
    condition is counted.

``evasion.parquet`` (study B only)
    Per (theta*, psi*, alpha, c, switch_type, margin bin): how often the
    speaker took the informative / persuasive reference utterance and how much
    policy mass sat on each, as a function of how close ``Sus(t)`` was to the
    boundary.  Pre-switch rounds only (``margin`` is NaN afterwards).

``evasion_rounds.parquet`` (study B only)
    The same quantities per round instead of per margin bin.

Usage
-----
    python experiments/switching/aggregate.py              # all four studies
    python experiments/switching/aggregate.py --studies B  # just one
    python experiments/switching/aggregate.py --workers 8
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

STUDIES = ("A", "B", "Fvig", "Fcred")
LISTENERS = ("cred", "vig", "switch")
PSIS = ("inf", "high", "low")
KEY = ["theta_star", "psi_star", "alpha", "c", "switch_type"]

# k = round - tau window kept in aligned.parquet
K_LO, K_HI = -30, 60

# fixed horizons kept per run in runs.parquet (plus each run's own tau)
HORIZONS = (25, 50, 100, 150)

# margin = Sus(t) - boundary at choice time; always negative before the switch
MARGIN_BINS = [-np.inf, -2.0, -1.0, -0.5, -0.25, -0.1, -0.05, -0.02, -0.01, 0.0]

BASE_COLS = [
    "sim", "round", "theta_star", "psi_star", "alpha", "c", "switch_type",
    "obs_count", "utt", "crossed", "switched",
    "sus1_score", "sus1_sigma2", "sus1_Sus", "sus1_sigma_bar2",
    "E_theta_cred", "std_theta_cred", "E_theta_vig", "std_theta_vig",
    "p_psi_vig_inf", "p_psi_vig_high", "p_psi_vig_low",
    "E_theta_switch", "std_theta_switch",
    "p_psi_switch_inf", "p_psi_switch_high", "p_psi_switch_low",
]
B_COLS = ["margin", "p_chosen", "p_pers_ref", "p_inf_ref",
          "went_informative", "went_persuasive"]

MEAN_COLS = (
    [f"bias_{L}" for L in LISTENERS]
    + [f"abias_{L}" for L in LISTENERS]
    + [f"sqerr_{L}" for L in LISTENERS]
    + [f"std_theta_{L}" for L in LISTENERS]
    + [f"cov_{L}" for L in LISTENERS]
    + [f"p_psi_vig_{p}" for p in PSIS]
    + [f"p_psi_switch_{p}" for p in PSIS]
    + ["crossed", "switched", "sus1_score", "sus1_sigma2", "sus1_Sus",
       "sigma_bar", "threshold",
       "d_switch_vig", "d_switch_cred", "ad_switch_vig", "ad_switch_cred"]
)
SD_COLS = [
    "bias_cred", "bias_vig", "bias_switch", "abias_switch", "sus1_score",
    "d_switch_vig", "d_switch_cred", "ad_switch_vig", "ad_switch_cred",
]


# ---------------------------------------------------------------------------
# Derived per-row quantities
# ---------------------------------------------------------------------------

def _derive(df: pd.DataFrame) -> pd.DataFrame:
    """Add bias / error / coverage / detector columns used by every aggregate."""
    th = df["theta_star"].to_numpy()
    for L in LISTENERS:
        e = df[f"E_theta_{L}"].to_numpy(dtype=float)
        b = e - th
        df[f"bias_{L}"] = b
        df[f"abias_{L}"] = np.abs(b)
        df[f"sqerr_{L}"] = b * b
        # is theta* inside the listener's own +/- 2 sd interval?
        df[f"cov_{L}"] = (np.abs(b) <= 2.0 * df[f"std_theta_{L}"].to_numpy(dtype=float)).astype(float)
    # paired differences: same simulation, same observations, same utterances
    df["d_switch_vig"] = df["bias_switch"] - df["bias_vig"]
    df["d_switch_cred"] = df["bias_switch"] - df["bias_cred"]
    df["ad_switch_vig"] = df["abias_switch"] - df["abias_vig"]
    df["ad_switch_cred"] = df["abias_switch"] - df["abias_cred"]
    df["sigma_bar"] = np.sqrt(df["sus1_sigma_bar2"].to_numpy(dtype=float))
    df["threshold"] = df["c"].to_numpy(dtype=float) * df["sigma_bar"] / np.sqrt(df["round"].to_numpy(dtype=float))
    return df


def _tau_table(df: pd.DataFrame) -> pd.DataFrame:
    """First switched round per (sim, c, switch_type); absent when never switched."""
    sw = df.loc[df["switched"] == 1]
    if sw.empty:
        return pd.DataFrame(columns=["sim", "c", "switch_type", "tau"])
    return (sw.groupby(["sim", "c", "switch_type"], observed=True)["round"]
              .min().rename("tau").reset_index())


# ---------------------------------------------------------------------------
# Per-shard summaries
# ---------------------------------------------------------------------------

def _rounds(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["c", "switch_type", "round"], observed=True)
    out = g[MEAN_COLS].mean()
    sd = g[SD_COLS].std(ddof=1)
    sd.columns = [f"{c}_sd" for c in sd.columns]
    out = out.join(sd)
    out["n"] = g.size()
    return out.reset_index()


def _aligned(df: pd.DataFrame, tau: pd.DataFrame) -> pd.DataFrame:
    if tau.empty:
        return pd.DataFrame()
    d = df.merge(tau, on=["sim", "c", "switch_type"], how="inner")
    d["k"] = d["round"] - d["tau"]
    d = d[(d["k"] >= K_LO) & (d["k"] <= K_HI)]
    if d.empty:
        return pd.DataFrame()
    cols = ([f"bias_{L}" for L in LISTENERS] + [f"abias_{L}" for L in LISTENERS]
            + [f"std_theta_{L}" for L in LISTENERS]
            + [f"p_psi_switch_{p}" for p in PSIS] + [f"p_psi_vig_{p}" for p in PSIS]
            + ["d_switch_vig", "ad_switch_vig", "d_switch_cred", "ad_switch_cred"])
    g = d.groupby(["c", "switch_type", "k"], observed=True)
    out = g[cols].mean()
    sd = g[["d_switch_vig", "ad_switch_vig"]].std(ddof=1)
    sd.columns = [f"{c}_sd" for c in sd.columns]
    out = out.join(sd)
    out["n"] = g.size()
    out["tau_mean"] = g["tau"].mean()
    return out.reset_index()


def _runs(df: pd.DataFrame, tau: pd.DataFrame) -> pd.DataFrame:
    """One row per (sim, condition, horizon): the run-level snapshot that the
    round-averaged table cannot give back.

    Horizons are the fixed rounds in ``HORIZONS`` plus each run's own ``tau``
    and ``tau - 1`` -- the belief at the moment of the alarm, which is what
    "damage done before detection" is read from.  Keeping this means run-level
    distributions (quantiles, histograms, scatter against tau) stay available
    without touching the 5 GB of raw trajectories.
    """
    keep = ["sim", "c", "switch_type", "round", "bias_cred", "bias_vig", "bias_switch",
            "std_theta_switch", "std_theta_cred", "std_theta_vig"]
    fixed = df[df["round"].isin(HORIZONS)][keep].copy()
    fixed["horizon"] = fixed["round"].astype(str)

    parts = [fixed]
    if not tau.empty:
        d = df.merge(tau, on=["sim", "c", "switch_type"], how="inner")
        for label, off in (("tau", 0), ("tau_minus_1", -1)):
            sel = d[d["round"] == d["tau"] + off]
            if not sel.empty:
                s = sel[keep].copy()
                s["horizon"] = label
                parts.append(s)
    out = pd.concat(parts, ignore_index=True)
    out = out.merge(tau, on=["sim", "c", "switch_type"], how="left")
    out["tau"] = out["tau"].fillna(-1).astype(np.int32)
    return out


def _utterances(df: pd.DataFrame, offline: bool) -> pd.DataFrame:
    d = df
    if offline:
        # the (obs, utt) stream is shared by every condition in an offline cell
        first_c = d["c"].iloc[0]
        first_st = d["switch_type"].iloc[0]
        d = d[(d["c"] == first_c) & (d["switch_type"] == first_st)]
    g = (d.groupby(["c", "switch_type", "obs_count", "utt", "switched"], observed=True)
           .size().rename("n").reset_index())
    return g


def _evasion(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pre = df[df["margin"].notna()]
    cols = ["went_informative", "went_persuasive", "p_chosen", "p_inf_ref", "p_pers_ref", "margin"]
    if pre.empty:
        return pd.DataFrame(), pd.DataFrame()
    pre = pre.copy()
    pre["margin_bin"] = pd.cut(pre["margin"], MARGIN_BINS)
    g = pre.groupby(["c", "switch_type", "margin_bin"], observed=True)
    by_bin = g[cols].mean()
    by_bin["n"] = g.size()
    by_bin = by_bin.reset_index()
    by_bin["margin_bin"] = by_bin["margin_bin"].astype(str)

    gr = pre.groupby(["c", "switch_type", "round"], observed=True)
    by_round = gr[cols].mean()
    by_round["n"] = gr.size()
    return by_bin.reset_index(drop=True), by_round.reset_index()


def summarize_shard(task):
    study, path, offline, is_b = task
    cols = BASE_COLS + (B_COLS if is_b else [])
    df = pd.read_parquet(path, columns=cols)
    # ``went_*`` are -1 (not applicable) for an informative speaker
    if is_b:
        for col in ("went_informative", "went_persuasive"):
            df.loc[df[col] < 0, col] = np.nan
    df = _derive(df)
    tau = _tau_table(df)
    keyvals = {k: df[k].iloc[0] for k in ("theta_star", "psi_star", "alpha")}

    def stamp(frame):
        if frame is None or frame.empty:
            return frame
        frame.insert(0, "study", study)
        for k, v in keyvals.items():
            frame.insert(1, k, v)
        return frame

    rounds = stamp(_rounds(df))
    aligned = stamp(_aligned(df, tau))
    runs = stamp(_runs(df, tau))
    utts = stamp(_utterances(df, offline))
    ev_bin = ev_round = None
    if is_b:
        b1, b2 = _evasion(df)
        ev_bin, ev_round = stamp(b1), stamp(b2)
    return rounds, aligned, runs, utts, ev_bin, ev_round


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def shard_tasks(study, root="results"):
    d = os.path.join(root, f"switching_{study}")
    cfg = json.load(open(os.path.join(d, "run_config.json"), encoding="utf-8"))
    offline = cfg["mode"] == "offline"
    is_b = not offline
    tdir = os.path.join(d, "trajectories")
    tasks = []
    for name in sorted(os.listdir(tdir)):
        p = os.path.join(tdir, name, "part.parquet")
        if os.path.isfile(p):
            tasks.append((study, p, offline, is_b))
    return tasks, cfg


def _concat(parts):
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    # every aggregated value is a mean or sd of a quantity in [0, 1] or a small
    # log-scale score; float32 is ample and halves the files
    keep = {"theta_star", "alpha", "c"}
    f64 = [c for c in out.columns if out[c].dtype == np.float64 and c not in keep]
    out[f64] = out[f64].astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", nargs="*", default=list(STUDIES))
    ap.add_argument("--root", default="results")
    ap.add_argument("--out_dir", default=os.path.join("results", "switching", "agg"))
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    os.makedirs(args.out_dir, exist_ok=True)

    tasks = []
    meta = {}
    for st in args.studies:
        t, cfg = shard_tasks(st, args.root)
        tasks += t
        meta[st] = dict(mode=cfg["mode"], speaker_level=cfg["speaker_level"],
                        listener_level=cfg["listener_level"], n_cells=cfg["grid"]["n_cells"],
                        n_sims=cfg["grid"]["n_sims_per_cell"], rounds=cfg["grid"]["rounds"],
                        git_hash=cfg["git_hash"], theta_grid=cfg["theta_grid"],
                        utterance_index=cfg["utterance_index"])
    print(f"{len(tasks)} shards from {args.studies} on {workers} workers")

    R, AL, RU, UT, EB, ER = [], [], [], [], [], []
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(summarize_shard, t) for t in tasks]
        for fut in as_completed(futs):
            r, al, ru, ut, eb, er = fut.result()
            R.append(r); AL.append(al); RU.append(ru); UT.append(ut)
            if eb is not None:
                EB.append(eb); ER.append(er)
            done += 1
            if done % 200 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"  {done}/{len(tasks)} shards  {el:.0f}s  eta {el/done*(len(tasks)-done):.0f}s", flush=True)

    outputs = {"rounds": _concat(R), "aligned": _concat(AL), "runs": _concat(RU),
               "utterances": _concat(UT)}
    if EB:
        outputs["evasion"] = _concat(EB)
        outputs["evasion_rounds"] = _concat(ER)
    for name, df in outputs.items():
        path = os.path.join(args.out_dir, f"{name}.parquet")
        df.to_parquet(path, index=False)
        print(f"  wrote {name}.parquet  rows={len(df):,}  {os.path.getsize(path)/1e6:.1f} MB")

    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(studies=meta, built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       k_window=[K_LO, K_HI], margin_bins=[float(b) for b in MARGIN_BINS]),
                  fh, indent=1)
    print(f"done in {(time.time()-t0)/60:.1f} min -> {args.out_dir}")


if __name__ == "__main__":
    main()
