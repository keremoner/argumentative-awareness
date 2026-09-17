"""Shared helpers for sus_variants analysis scripts."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


SUS_VARIANTS = ["sus_1"]
ALL_SCORES = ["surp2"] + SUS_VARIANTS
NULL_THETAS = [0.1, 0.3, 0.5, 0.7, 0.9]

# There is one variance now (the exact state-only V_t). Kept as a
# single-element list so the analysis figures keep their panel structure.
VARIANCE_KINDS = ["exact"]

SCORE_COLORS = {
    "surp2":  "tab:blue",
    "sus_1":  "tab:green",
}


def load(in_dir: str):
    traj = pd.read_parquet(os.path.join(in_dir, "raw_trajectories.parquet"))
    tau = pd.read_parquet(os.path.join(in_dir, "tau_summary.parquet"))
    with open(os.path.join(in_dir, "run_config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    return traj, tau, cfg


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def per_sim_crossed(traj: pd.DataFrame, t_warmup: int = 0):
    """Per (psi, theta_star, sim_id, score_type) indicator: did the crossing
    fire at any round > t_warmup?"""
    col = "crossed"
    sub = traj[traj["round"] > t_warmup]
    return (sub.groupby(["psi", "theta_star", "sim_id", "score_type"],
                        observed=True)[col].any()
            .reset_index(name="ever"))
