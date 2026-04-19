"""Shared helpers for Group 1 analysis scripts."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


# Presented scores (surp1 collected but excluded from main plots per spec).
PRESENTED_SCORES = ["surp2", "sus"]
ALL_SCORES = ["surp2", "surp1", "sus"]
NULL_THETAS = [0.1, 0.3, 0.5, 0.7, 0.9]

SCORE_COLORS = {"surp2": "tab:blue", "surp1": "tab:gray", "sus": "tab:green"}


def load(in_dir: str):
    traj = pd.read_parquet(os.path.join(in_dir, "raw_trajectories.parquet"))
    tau = pd.read_parquet(os.path.join(in_dir, "tau_summary.parquet"))
    with open(os.path.join(in_dir, "run_config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    return traj, tau, cfg


def first_crossing_round(crossed: np.ndarray, t_warmup: int = 0):
    """First round t >= t_warmup+1 where crossed=True, else None."""
    idx = np.where(crossed)[0]
    for i in idx:
        t = i + 1  # 1-indexed
        if t > t_warmup:
            return t
    return None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path
