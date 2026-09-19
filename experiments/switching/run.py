"""
Switching experiments runner -- one entry point, one JSON grid per study.

    python experiments/switching/run.py --grid experiments/switching/grids/A.json --pilot 20
    python experiments/switching/run.py --grid experiments/switching/grids/A.json
    python experiments/switching/run.py --grid experiments/switching/grids/B.json --n_sims 60

Two modes (``"mode"`` in the grid):

``offline``  (Experiments A, C1, C3-for-A)
    The speaker never sees the listener, so utterances are generated once per
    (theta*, psi*, alpha, sim) together with the per-round S1 tables and the
    sus_1 observer statistics.  Every (c, switch_type) condition is then
    derived offline: tau from ``tau_for_rule``, the retrospective trajectory
    from ``splice`` (exact), soft / hard_amnesic from ``switching_trajectory``.
    Speaker: ``S1`` (A) or ``S2_vig`` -- an S2 whose internal listener is the
    always-vigilant L1 carried in the run (C1).

``feedback`` (Experiment B, C3-for-B)
    The S2 speaker's internal listener is a private replica of the actual
    switching ``DetectionListener`` (same detector, c, switch type, S1-inf
    model), so every (cell, sim, c, switch_type) is a fresh simulation.  The
    replica is asserted equal to the actual listener every round.

Every run writes ``trajectories/cell=XXXX/part.parquet`` (one row per
listener-round-condition), ``tau_summary.parquet``, ``run_config.json``,
``progress.jsonl`` and appends to ``timing.md``.  Seeds are
``seed_for(cell_index, sim_index, seed_base)`` and recorded.
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

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rsa.setup import make_thetas, make_world, make_semantics  # noqa: E402
from rsa.speaker0 import Speaker0  # noqa: E402
from rsa.listener0 import Listener0  # noqa: E402
from rsa.speaker1 import Speaker1  # noqa: E402
from rsa.listener1 import Listener1  # noqa: E402
from rsa.speaker2 import Speaker2  # noqa: E402
from rsa.detection import (  # noqa: E402
    DetectionListener, SequentialTest, SUS_VARIANT_FNS,
    tau_for_rule, switching_trajectory, make_switching_listener,
)
from rsa.detection.replay import empty_trajectory, record_round  # noqa: E402

PSIS = ["inf", "high", "low"]
PSI_LABEL = {"inf": "inf", "high": "pers+", "low": "pers-"}
LISTENER_COLS = ("cred", "vig", "switch")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    study="A", mode="offline", speaker_level="S1", listener_level="L1",
    out_dir=None,
    theta_stars=[0.1, 0.3, 0.5, 0.7, 0.9],
    psi_stars=["inf", "high", "low"],
    alphas=[1.5, 3.0, 5.0, 10.0],
    cs=[2.0, 3.5],
    switch_types=["hard", "soft"],
    n_sims=100, rounds=150, n=1, m=7,
    theta_grid=None,
    seed_base=0, workers=0, retro_cache=False,
    score="sus_1",
)


def load_config(path, overrides):
    with open(path, encoding="utf-8") as fh:
        cfg = dict(DEFAULTS)
        cfg.update(json.load(fh))
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    if cfg["theta_grid"] is None:
        cfg["theta_grid"] = make_thetas(0.1)
    if cfg["out_dir"] is None:
        cfg["out_dir"] = os.path.join("results", f"switching_{cfg['study']}")
    if cfg["workers"] <= 0:
        cfg["workers"] = max(1, (os.cpu_count() or 2) - 1)
    cfg["theta_stars"] = [float(x) for x in cfg["theta_stars"]]
    cfg["alphas"] = [float(x) for x in cfg["alphas"]]
    cfg["cs"] = [float(x) for x in cfg["cs"]]
    if cfg["mode"] not in ("offline", "feedback"):
        raise ValueError(f"unknown mode {cfg['mode']!r}")
    if cfg["mode"] == "offline" and cfg["speaker_level"] not in ("S1", "S2_vig", "S2_cred"):
        raise ValueError("offline mode supports speaker_level S1, S2_vig or S2_cred")
    if cfg["listener_level"] not in ("L1", "L2"):
        raise ValueError("listener_level must be L1 or L2")
    if cfg["listener_level"] == "L2" and cfg["speaker_level"] != "S2_cred":
        raise ValueError("listener_level L2 (the C2 detector study) requires speaker_level S2_cred")
    if cfg["mode"] == "feedback" and cfg["speaker_level"] != "S2_replica":
        raise ValueError("feedback mode requires speaker_level S2_replica")
    return cfg


def build_cells(cfg):
    cells = []
    if cfg["mode"] == "offline":
        for th in cfg["theta_stars"]:
            for psi in cfg["psi_stars"]:
                for a in cfg["alphas"]:
                    cells.append(dict(cell_id=len(cells), theta_star=th, psi_star=psi,
                                      alpha=a, c=None, switch_type=None))
    else:
        for th in cfg["theta_stars"]:
            for psi in cfg["psi_stars"]:
                for a in cfg["alphas"]:
                    for c in cfg["cs"]:
                        for st in cfg["switch_types"]:
                            cells.append(dict(cell_id=len(cells), theta_star=th,
                                              psi_star=psi, alpha=a, c=c, switch_type=st))
    return cells


def seed_for(cell_index: int, sim_index: int, base: int) -> int:
    """Seed of the global RNG (speaker utterance sampling): unique per (cell, sim)."""
    return int((base + cell_index * 1_000_003 + sim_index) % (2 ** 31))


def obs_seed_for(sim_index: int, base: int) -> int:
    """Seed of the world's private observation RNG: depends on the sim index
    only, so sim i draws the same observation stream in every cell (identical
    within a theta*, the same uniforms through a different CDF across theta*)
    -- common random numbers for every between-condition comparison."""
    return int((base + 7_919 * (sim_index + 1)) % (2 ** 31))


def n_conditions(cfg):
    return len(cfg["cs"]) * len(cfg["switch_types"]) if cfg["mode"] == "offline" else 1


def rows_per_cell(cfg):
    return cfg["n_sims"] * cfg["rounds"] * n_conditions(cfg)


# ---------------------------------------------------------------------------
# Column buffers
# ---------------------------------------------------------------------------

class Rows:
    """Preallocated column store for one cell."""

    INT = ("cell_id", "sim", "round", "obs", "obs_count", "utt", "crossed", "switched",
           "u_pers_ref", "u_inf_ref", "went_persuasive", "went_informative", "seed")
    F64 = ("sus1_score", "sus1_sigma2", "sus1_Sus", "sus1_sigma_bar2", "margin",
           "p_chosen", "p_pers_ref", "p_inf_ref", "replica_maxdiff")

    def __init__(self, n, feedback):
        self.n = n
        self.feedback = feedback
        self.cols = {}
        for k in self.INT:
            if not feedback and k in ("u_pers_ref", "u_inf_ref", "went_persuasive", "went_informative"):
                continue
            self.cols[k] = np.full(n, -1, dtype=np.int32)
        for k in self.F64:
            if not feedback and k in ("margin", "p_chosen", "p_pers_ref", "p_inf_ref", "replica_maxdiff"):
                continue
            self.cols[k] = np.full(n, np.nan, dtype=np.float64)
        for L in LISTENER_COLS:
            self.cols[f"E_theta_{L}"] = np.full(n, np.nan, dtype=np.float32)
            self.cols[f"std_theta_{L}"] = np.full(n, np.nan, dtype=np.float32)
            if L != "cred":
                for p in PSIS:
                    self.cols[f"p_psi_{L}_{p}"] = np.full(n, np.nan, dtype=np.float32)
        self.cond_c = np.full(n, np.nan, dtype=np.float64)
        self.cond_st = np.empty(n, dtype=object)

    def put_traj(self, sl, name, traj):
        self.cols[f"E_theta_{name}"][sl] = traj.E_theta
        self.cols[f"std_theta_{name}"][sl] = traj.std_theta
        if name != "cred":
            for j, p in enumerate(PSIS):
                self.cols[f"p_psi_{name}_{p}"][sl] = traj.psi[:, j]

    def frame(self, cfg, cell):
        d = {"study": cfg["study"]}
        d.update({k: v for k, v in self.cols.items() if k in ("cell_id", "sim", "round")})
        d["theta_star"] = np.float64(cell["theta_star"])
        d["psi_star"] = cell["psi_star"]
        d["alpha"] = np.float64(cell["alpha"])
        d["c"] = self.cond_c
        d["switch_type"] = self.cond_st
        d["speaker_level"] = cfg["speaker_level"]
        d["listener_level"] = cfg["listener_level"]
        for k, v in self.cols.items():
            if k not in d:
                d[k] = v
        return pd.DataFrame(d)


# ---------------------------------------------------------------------------
# Simulation primitives
# ---------------------------------------------------------------------------

def _stack(thetas, theta_star, psi, alpha, n, m):
    world = make_world(theta_star, n=n, m=m)
    sem = make_semantics(n=n)
    s0 = Speaker0(thetas, semantics=sem, world=world)
    l0 = Listener0(thetas, s0, semantics=sem, world=world)
    s1 = Speaker1(thetas, l0, semantics=sem, world=world, alpha=alpha, psi=psi)
    return world, sem, s0, l0, s1


def _obs_count(obs):
    """Number of effective sessions for an n=1 observation histogram."""
    return int(obs.index(1))


def _test(cfg, c, switch_type, enabled):
    return SequentialTest(SUS_VARIANT_FNS["1"], cfg["score"], c=c,
                          switch_enabled=enabled, switch_type=switch_type)


def run_offline_sim(cfg, cell, sim_idx, rows, base_row, conditions, tau_records):
    """Generate one stream and derive every (c, switch_type) condition offline.

    Fills ``rows`` starting at ``base_row`` (one block of ``rounds`` rows per
    condition, in ``conditions`` order)."""
    thetas = cfg["theta_grid"]
    T = cfg["rounds"]
    seed = seed_for(cell["cell_id"], sim_idx, cfg["seed_base"])
    random.seed(seed); np.random.seed(seed)
    alpha = cell["alpha"]
    world, sem, s0, l0, s1 = _stack(thetas, cell["theta_star"], cell["psi_star"], alpha,
                                    cfg["n"], cfg["m"])
    world.rng = random.Random(obs_seed_for(sim_idx, cfg["seed_base"]))

    obs_test = _test(cfg, float("inf"), "hard", False)
    internal = None
    if cfg["speaker_level"] == "S2_cred":
        # The data-generating S2 models a credulous L1 (``internal``).
        #   listener_level L1 (Fang's cooperative dyad): the listeners invert
        #     S1; ``cred`` below equals ``internal`` round for round, so the
        #     credulous column is the matched S2 <-> credulous-L1 dyad.
        #   listener_level L2 (C2): the listeners invert this same S2, whose
        #     "inf" tables are then the properly specified null.
        internal = Listener1(thetas, ["inf"], s1, world, sem, "inf", alpha)
        s2 = Speaker2(thetas, internal, sem, world, alpha=alpha, psi=cell["psi_star"])
        speaker = s2
        model = s2 if cfg["listener_level"] == "L2" else s1
    else:
        model = s1
        s2 = None
        speaker = s1
    observer = DetectionListener(thetas, PSIS, model, world, sem, tests=[obs_test], alpha=alpha)
    cred = observer.naive                       # c = inf: the observer never switches
    vig = Listener1(thetas, PSIS, model, world, sem, "vig", alpha)

    if cfg["speaker_level"] == "S2_vig":        # C1: S2 modelling Fang's L1-strat == vig
        s2 = Speaker2(thetas, vig, sem, world, alpha=alpha, psi=cell["psi_star"])
        speaker = s2

    cred_t = empty_trajectory(T, thetas, PSIS)
    vig_t = empty_trajectory(T, thetas, PSIS)
    utts, utt_idx, obs_idx, obs_cnt = [], np.empty(T, np.int32), np.empty(T, np.int32), np.empty(T, np.int32)
    for i in range(T):
        obs = world.sample_obs()
        utt = speaker.sample_utterance(obs)
        utts.append(utt)
        utt_idx[i] = sem.utterance_index(utt)
        obs_idx[i] = world.obs_index(obs)
        obs_cnt[i] = _obs_count(obs)
        observer.update(utt)
        vig.update(utt)
        record_round(cred_t, i, cred)
        record_round(vig_t, i, vig)
        if internal is not None:
            internal.update(utt)
        if s2 is not None:
            s2.update(obs)
        s1.update(obs); l0.update(utt); s0.update(obs)

    h = obs_test.history
    sus = np.asarray(h["running_mean"]); sig = np.asarray(h["running_sigma"])
    score = np.asarray(h["scores"]); var = np.asarray(h["variances"])
    t_arr = np.arange(1, T + 1, dtype=float)
    tables = observer.table_history

    for k, (c, st) in enumerate(conditions):
        sl = slice(base_row + k * T, base_row + (k + 1) * T)
        tau = tau_for_rule(sus, sig, c)
        sw = switching_trajectory(cred_t, vig_t, utts, tables, tau, st, thetas, world, sem, alpha)
        rows.cols["cell_id"][sl] = cell["cell_id"]
        rows.cols["sim"][sl] = sim_idx
        rows.cols["seed"][sl] = seed
        rows.cols["round"][sl] = np.arange(1, T + 1)
        rows.cond_c[sl] = c
        rows.cond_st[sl] = st
        rows.cols["obs"][sl] = obs_idx
        rows.cols["obs_count"][sl] = obs_cnt
        rows.cols["utt"][sl] = utt_idx
        rows.cols["sus1_score"][sl] = score
        rows.cols["sus1_sigma2"][sl] = var
        rows.cols["sus1_Sus"][sl] = sus
        rows.cols["sus1_sigma_bar2"][sl] = sig ** 2
        rows.cols["crossed"][sl] = ((sig > 0) & (sus > c * sig / np.sqrt(t_arr))).astype(np.int32)
        rows.cols["switched"][sl] = ((np.arange(1, T + 1) >= tau).astype(np.int32)
                                     if tau is not None else 0)
        rows.put_traj(sl, "cred", cred_t)
        rows.put_traj(sl, "vig", vig_t)
        rows.put_traj(sl, "switch", sw)
        tau_records.append(dict(study=cfg["study"], cell_id=cell["cell_id"], sim=sim_idx, seed=seed,
                                theta_star=cell["theta_star"], psi_star=cell["psi_star"],
                                alpha=alpha, c=c, switch_type=st,
                                tau=(-1 if tau is None else int(tau)),
                                switched=int(tau is not None)))
    return seed


def run_feedback_sim(cfg, cell, sim_idx, rows, base_row, tau_records):
    """One fresh S2-vs-switching-L1 simulation for a (c, switch_type) cell."""
    thetas = cfg["theta_grid"]
    T = cfg["rounds"]
    seed = seed_for(cell["cell_id"], sim_idx, cfg["seed_base"])
    random.seed(seed); np.random.seed(seed)
    alpha, c, st, psi_star = cell["alpha"], cell["c"], cell["switch_type"], cell["psi_star"]
    # The listener's model is S1-inf (psi of this S1 object is irrelevant: only
    # its tables are read); the data-generating speaker is the S2 below.
    world, sem, s0, l0, s1 = _stack(thetas, cell["theta_star"], "inf", alpha, cfg["n"], cfg["m"])
    world.rng = random.Random(obs_seed_for(sim_idx, cfg["seed_base"]))

    def mk_det():
        # Same builder as rsa.game.make_listener("switch", ...), so the runner
        # and the game loop cannot drift apart.
        return make_switching_listener(thetas, PSIS, s1, world, sem, c=c, switch_type=st,
                                       alpha=alpha, name=cfg["score"],
                                       retro_cache=cfg["retro_cache"])
    actual = mk_det()
    replica = mk_det()
    s2 = Speaker2(thetas, replica, sem, world, alpha=alpha, psi=psi_star)
    cred = Listener1(thetas, ["inf"], s1, world, sem, "inf", alpha)
    vig = Listener1(thetas, PSIS, s1, world, sem, "vig", alpha)
    s2_cred_ref = (Speaker2(thetas, cred, sem, world, alpha=alpha, psi=psi_star)
                   if psi_star != "inf" else None)
    test = actual.tests[0]

    sl = slice(base_row, base_row + T)
    rows.cols["cell_id"][sl] = cell["cell_id"]
    rows.cols["sim"][sl] = sim_idx
    rows.cols["seed"][sl] = seed
    rows.cols["round"][sl] = np.arange(1, T + 1)
    rows.cond_c[sl] = c
    rows.cond_st[sl] = st
    cred_t = empty_trajectory(T, thetas, PSIS)
    vig_t = empty_trajectory(T, thetas, PSIS)
    sw_t = empty_trajectory(T, thetas, PSIS)
    max_replica_diff = 0.0

    for i in range(T):
        k = base_row + i
        obs = world.sample_obs()
        # margin at choice time: Sus(t-1) - boundary(t-1) of the live test
        hh = test.history
        if hh["running_mean"] and not actual.switched:
            rows.cols["margin"][k] = hh["running_mean"][-1] - hh["threshold"][-1]
        utt = s2.sample_utterance(obs)
        policy = s2.dist_over_utterances_obs_array(obs, psi_star)
        u_i = sem.utterance_index(utt)
        rows.cols["p_chosen"][k] = policy[u_i]
        inf_policy = s2.dist_over_utterances_obs_array(obs, "inf")
        u_inf = int(np.argmax(inf_policy))
        rows.cols["u_inf_ref"][k] = u_inf
        rows.cols["p_inf_ref"][k] = policy[u_inf]
        rows.cols["went_informative"][k] = int(u_i == u_inf)
        if s2_cred_ref is not None:
            pers_policy = s2_cred_ref.dist_over_utterances_obs_array(obs, psi_star)
            u_pers = int(np.argmax(pers_policy))
            rows.cols["u_pers_ref"][k] = u_pers
            rows.cols["p_pers_ref"][k] = policy[u_pers]
            rows.cols["went_persuasive"][k] = int(u_i == u_pers)

        actual.update(utt); replica.update(utt); cred.update(utt); vig.update(utt)
        d = float(np.abs(actual.theta_array() - replica.theta_array()).max())
        max_replica_diff = max(max_replica_diff, d)
        rows.cols["replica_maxdiff"][k] = d
        if d > 1e-9 or actual.switched != replica.switched:
            raise RuntimeError(f"replica diverged from actual listener at round {i+1}: {d}")

        rows.cols["obs"][k] = world.obs_index(obs)
        rows.cols["obs_count"][k] = _obs_count(obs)
        rows.cols["utt"][k] = u_i
        if len(hh["scores"]) == i + 1:            # test still observing this round
            rows.cols["sus1_score"][k] = hh["scores"][i]
            rows.cols["sus1_sigma2"][k] = hh["variances"][i]
            rows.cols["sus1_Sus"][k] = hh["running_mean"][i]
            rows.cols["sus1_sigma_bar2"][k] = hh["running_sigma"][i] ** 2
            rows.cols["crossed"][k] = int(hh["crossed"][i])
        else:
            rows.cols["crossed"][k] = 0
        rows.cols["switched"][k] = int(actual.switched)
        record_round(cred_t, i, cred); record_round(vig_t, i, vig); record_round(sw_t, i, actual)

        s2.update(obs)
        if s2_cred_ref is not None:
            s2_cred_ref.update(obs)
        s1.update(obs); l0.update(utt); s0.update(obs)

    rows.put_traj(sl, "cred", cred_t)
    rows.put_traj(sl, "vig", vig_t)
    rows.put_traj(sl, "switch", sw_t)
    tau = actual.switched_at
    tau_records.append(dict(study=cfg["study"], cell_id=cell["cell_id"], sim=sim_idx, seed=seed,
                            theta_star=cell["theta_star"], psi_star=psi_star, alpha=alpha,
                            c=c, switch_type=st, tau=(-1 if tau is None else int(tau)),
                            switched=int(tau is not None),
                            replica_maxdiff=max_replica_diff))
    return seed


# ---------------------------------------------------------------------------
# Cell task (one worker task == one cell)
# ---------------------------------------------------------------------------

def _shard_path(out_dir, cell_id):
    d = os.path.join(out_dir, "trajectories", f"cell={cell_id:04d}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "part.parquet")


def _tau_path(out_dir, cell_id):
    d = os.path.join(out_dir, "tau", f"cell={cell_id:04d}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "part.parquet")


def _write_atomic(df, path):
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return os.path.getsize(path)


def run_cell(task):
    cfg, cell = task
    t0 = time.time()
    T = cfg["rounds"]
    feedback = cfg["mode"] == "feedback"
    conditions = [(c, st) for c in cfg["cs"] for st in cfg["switch_types"]] if not feedback else [(cell["c"], cell["switch_type"])]
    n_rows = cfg["n_sims"] * T * len(conditions)
    rows = Rows(n_rows, feedback)
    tau_records = []
    for s in range(cfg["n_sims"]):
        base = s * T * len(conditions)
        if feedback:
            run_feedback_sim(cfg, cell, s, rows, base, tau_records)
        else:
            run_offline_sim(cfg, cell, s, rows, base, conditions, tau_records)
    df = rows.frame(cfg, cell)
    tau_df = pd.DataFrame(tau_records)
    _write_atomic(tau_df, _tau_path(cfg["out_dir"], cell["cell_id"]))
    nbytes = _write_atomic(df, _shard_path(cfg["out_dir"], cell["cell_id"]))
    return dict(cell_id=cell["cell_id"], theta_star=cell["theta_star"], psi_star=cell["psi_star"],
                alpha=cell["alpha"], c=cell["c"], switch_type=cell["switch_type"],
                rows=int(len(df)), bytes=int(nbytes), wall_s=round(time.time() - t0, 2),
                switch_rate=float(tau_df["switched"].mean()) if len(tau_df) else float("nan"),
                median_tau=float(tau_df.loc[tau_df.tau > 0, "tau"].median()) if (tau_df.tau > 0).any() else float("nan"))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _parquet_rows(path):
    try:
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return None


def completed_cells(cfg):
    out = set()
    want = rows_per_cell(cfg)
    root = os.path.join(cfg["out_dir"], "trajectories")
    if not os.path.isdir(root):
        return out
    for name in os.listdir(root):
        if not name.startswith("cell="):
            continue
        cid = int(name.split("=")[1])
        p = os.path.join(root, name, "part.parquet")
        tp = os.path.join(cfg["out_dir"], "tau", name, "part.parquet")
        if os.path.isfile(p) and os.path.isfile(tp) and _parquet_rows(p) == want:
            out.add(cid)
    return out


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def write_run_config(cfg, cells, status, wall_seconds, extra=None):
    sem = make_semantics(n=cfg["n"])
    info = dict(
        status=status,
        study=cfg["study"], mode=cfg["mode"], speaker_level=cfg["speaker_level"],
        listener_level=cfg["listener_level"],
        grid=dict(theta_stars=cfg["theta_stars"], psi_stars=cfg["psi_stars"], alphas=cfg["alphas"],
                  cs=cfg["cs"], switch_types=cfg["switch_types"], n=cfg["n"], m=cfg["m"],
                  rounds=cfg["rounds"], n_sims_per_cell=cfg["n_sims"], n_cells=len(cells),
                  n_conditions_per_cell=n_conditions(cfg)),
        theta_grid=list(cfg["theta_grid"]),
        psi_label_map=PSI_LABEL,
        utterance_index={i: list(u) for i, u in enumerate(sem.utterance_space())},
        obs_index_note="obs = index into world.generate_all_obs(); obs_count = number of effective sessions",
        seed_rule="seed = (seed_base + cell_index * 1000003 + sim_index) mod 2^31; "
                  "random.seed and np.random.seed are both set to it before each sim "
                  "(speaker utterance sampling)",
        obs_seed_rule="obs_seed = (seed_base + 7919 * (sim_index + 1)) mod 2^31 seeds the world's "
                      "private observation RNG: sim i sees the same observation stream in every "
                      "cell of this run (identical within a theta*, same uniforms across theta*)",
        seed_base=cfg["seed_base"], score=cfg["score"], retro_cache=cfg["retro_cache"],
        workers=cfg["workers"], git_hash=git_hash(), wall_seconds=wall_seconds,
        run_date_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        cells=cells,
    )
    if extra:
        info.update(extra)
    path = os.path.join(cfg["out_dir"], "run_config.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)


def journal(cfg, rec):
    with open(os.path.join(cfg["out_dir"], "progress.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def append_timing(cfg, text):
    os.makedirs(cfg["out_dir"], exist_ok=True)
    with open(os.path.join(cfg["out_dir"], "timing.md"), "a", encoding="utf-8") as fh:
        fh.write(text.rstrip() + "\n")


def merge_tau(cfg):
    root = os.path.join(cfg["out_dir"], "tau")
    parts = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, "part.parquet")
        if os.path.isfile(p):
            parts.append(pd.read_parquet(p))
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(os.path.join(cfg["out_dir"], "tau_summary.parquet"), index=False)
    return df


def pilot(cfg, n_sims):
    """Time one cell in-process and extrapolate to the whole grid."""
    cells = build_cells(cfg)
    th = 0.5 if 0.5 in cfg["theta_stars"] else cfg["theta_stars"][0]
    a = 3.0 if 3.0 in cfg["alphas"] else cfg["alphas"][0]
    psi = "high" if "high" in cfg["psi_stars"] else cfg["psi_stars"][0]
    pick = [c for c in cells if c["theta_star"] == th and c["alpha"] == a and c["psi_star"] == psi]
    if cfg["mode"] == "feedback":
        pick = [c for c in pick if c["c"] == 3.5 and c["switch_type"] == "hard"] or pick
    cell = dict(pick[0]); cell["cell_id"] = 9999
    pcfg = dict(cfg); pcfg["n_sims"] = n_sims
    pcfg["out_dir"] = os.path.join(cfg["out_dir"], "_pilot")
    os.makedirs(pcfg["out_dir"], exist_ok=True)
    t0 = time.time()
    summary = run_cell((pcfg, cell))
    wall = time.time() - t0
    per_sim = wall / n_sims
    total_sims = len(cells) * cfg["n_sims"]
    proj_serial = per_sim * total_sims
    proj_wall = proj_serial / cfg["workers"]
    msg = (f"## Pilot ({time.strftime('%Y-%m-%d %H:%M:%S')})\n"
           f"- study {cfg['study']} mode {cfg['mode']} speaker {cfg['speaker_level']}\n"
           f"- pilot cell: theta*={cell['theta_star']} psi*={cell['psi_star']} alpha={cell['alpha']}"
           f" c={cell['c']} switch={cell['switch_type']}, {n_sims} sims x {cfg['rounds']} rounds"
           f" (switch rate {summary['switch_rate']:.2f}, median tau {summary['median_tau']})\n"
           f"- {per_sim:.3f} s per sim (incl. {n_conditions(cfg)} offline conditions)\n"
           f"- grid: {len(cells)} cells x {cfg['n_sims']} sims = {total_sims} sims\n"
           f"- projected: {proj_serial/60:.1f} CPU-min, {proj_wall/60:.1f} wall-min on {cfg['workers']} workers\n")
    print(msg)
    append_timing(cfg, msg)
    return per_sim, proj_wall


def run(cfg, resume=True):
    cells = build_cells(cfg)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    done = completed_cells(cfg) if resume else set()
    todo = [c for c in cells if c["cell_id"] not in done]
    print(f"{len(cells)} cells, {len(done)} already complete, {len(todo)} to run on {cfg['workers']} workers")
    write_run_config(cfg, cells, "running", 0.0)
    t0 = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as ex:
        futs = {ex.submit(run_cell, (cfg, c)): c for c in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            n_done += 1
            el = time.time() - t0
            eta = el / n_done * (len(todo) - n_done)
            rec["elapsed_s"] = round(el, 1)
            journal(cfg, rec)
            print(f"  cell {rec['cell_id']:04d} theta={rec['theta_star']} psi={rec['psi_star']:<4} "
                  f"alpha={rec['alpha']:<5} c={rec['c']} st={rec['switch_type']} "
                  f"switch={rec['switch_rate']:.2f} tau~{rec['median_tau']} "
                  f"[{n_done}/{len(todo)}] elapsed={el:.0f}s eta={eta:.0f}s", flush=True)
    wall = time.time() - t0
    tau_df = merge_tau(cfg)
    write_run_config(cfg, cells, "complete", wall,
                     extra=dict(total_trajectory_rows=int(rows_per_cell(cfg) * len(cells)),
                                n_tau_rows=int(len(tau_df))))
    append_timing(cfg, f"## Run ({time.strftime('%Y-%m-%d %H:%M:%S')})\n"
                       f"- {len(todo)} cells run this invocation ({len(done)} resumed), "
                       f"{cfg['n_sims']} sims/cell, {cfg['workers']} workers\n"
                       f"- actual wall: {wall/60:.1f} min\n")
    print(f"done in {wall/60:.1f} min; tau_summary rows = {len(tau_df)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--pilot", type=int, default=0, help="time one cell with this many sims and exit")
    ap.add_argument("--n_sims", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--retro_cache", action="store_true", default=None)
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.grid, dict(n_sims=args.n_sims, rounds=args.rounds, workers=args.workers,
                                      out_dir=args.out_dir, retro_cache=args.retro_cache))
    if args.pilot:
        pilot(cfg, args.pilot)
        return
    run(cfg, resume=not args.no_resume)


if __name__ == "__main__":
    main()
