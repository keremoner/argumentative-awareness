"""Smoke tests for experiments/switching/run.py (Task 5).

Runs one tiny cell in each mode in-process and checks the parquet layout and
the invariants the datasets are built on:

* offline: the switching trajectory equals the credulous one before tau and,
  for ``hard``, the always-vigilant one from tau; ``crossed``/``switched`` are
  consistent with the stored Sus and sigma_bar; every condition shares the
  same (obs, utt) stream; seeds are reproducible.
* feedback: the private replica never diverges from the actual listener, the
  chosen utterance always has positive policy probability, and the detector
  columns are NaN after the switch (test frozen).
* C2 (S2_cred / L2) runs.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "experiments", "switching"))

import run as R  # noqa: E402


def _cfg(tmp_path, **kw):
    grid = dict(study="T", mode="offline", speaker_level="S1", listener_level="L1",
                out_dir=str(tmp_path / "out"), theta_stars=[0.5], psi_stars=["high"],
                alphas=[5.0], cs=[2.0, 3.5], switch_types=["hard", "soft"],
                n_sims=3, rounds=20, n=1, m=7, seed_base=7, workers=1)
    grid.update(kw)
    p = tmp_path / "grid.json"
    p.write_text(json.dumps(grid))
    return R.load_config(str(p), {})


def _run_first_cell(cfg):
    cells = R.build_cells(cfg)
    summary = R.run_cell((cfg, cells[0]))
    df = pd.read_parquet(R._shard_path(cfg["out_dir"], 0))
    tau = pd.read_parquet(R._tau_path(cfg["out_dir"], 0))
    return summary, df, tau


def test_offline_layout_and_invariants(tmp_path):
    cfg = _cfg(tmp_path)
    summary, df, tau = _run_first_cell(cfg)
    T, n_sims = cfg["rounds"], cfg["n_sims"]
    n_cond = len(cfg["cs"]) * len(cfg["switch_types"])
    assert len(df) == n_sims * T * n_cond and summary["rows"] == len(df)
    for col in ("study", "cell_id", "sim", "round", "theta_star", "psi_star", "alpha", "c",
                "switch_type", "speaker_level", "listener_level", "obs", "obs_count", "utt",
                "sus1_score", "sus1_sigma2", "sus1_Sus", "sus1_sigma_bar2", "crossed", "switched",
                "E_theta_cred", "std_theta_cred", "E_theta_vig", "std_theta_vig",
                "p_psi_vig_inf", "E_theta_switch", "std_theta_switch", "p_psi_switch_high", "seed"):
        assert col in df.columns, col
    assert set(df["switch_type"]) == {"hard", "soft"} and set(df["c"]) == {2.0, 3.5}
    # same stream across conditions
    base = df[(df.c == 2.0) & (df.switch_type == "hard")].sort_values(["sim", "round"])
    for (c, st), g in df.groupby(["c", "switch_type"]):
        g = g.sort_values(["sim", "round"])
        np.testing.assert_array_equal(g["utt"].to_numpy(), base["utt"].to_numpy())
        np.testing.assert_array_equal(g["obs"].to_numpy(), base["obs"].to_numpy())
        np.testing.assert_array_equal(g["sus1_Sus"].to_numpy(), base["sus1_Sus"].to_numpy())
        # crossed / switched consistent with the stored statistics
        t = g["round"].to_numpy(float); sus = g["sus1_Sus"].to_numpy(); sig = np.sqrt(g["sus1_sigma_bar2"].to_numpy())
        np.testing.assert_array_equal(g["crossed"].to_numpy(), ((sig > 0) & (sus > c * sig / np.sqrt(t))).astype(int))
        for s, gs in g.groupby("sim"):
            cr = gs["crossed"].to_numpy(); sw = gs["switched"].to_numpy()
            trow = tau[(tau.sim == s) & (tau.c == c) & (tau.switch_type == st)].iloc[0]
            if cr.any():
                k = int(np.argmax(cr))
                assert trow.tau == k + 1 and sw[:k].sum() == 0 and sw[k:].all()
                # before tau the switching listener is the credulous one
                np.testing.assert_allclose(gs["E_theta_switch"].to_numpy()[:k], gs["E_theta_cred"].to_numpy()[:k], atol=1e-6)
                if st == "hard":
                    np.testing.assert_allclose(gs["E_theta_switch"].to_numpy()[k:], gs["E_theta_vig"].to_numpy()[k:], atol=1e-6)
                    assert np.isfinite(gs["p_psi_switch_high"].to_numpy()[k:]).all()
                assert np.isnan(gs["p_psi_switch_high"].to_numpy()[:k]).all()
            else:
                assert trow.tau == -1 and sw.sum() == 0
                np.testing.assert_allclose(gs["E_theta_switch"], gs["E_theta_cred"], atol=1e-6)
    assert (df["obs_count"] == 7 - df["obs"]).all()          # obs index 0 == 7 effective sessions
    assert len(tau) == n_sims * n_cond


def test_offline_seeds_reproducible(tmp_path):
    cfg = _cfg(tmp_path, n_sims=2, rounds=15)
    _, df1, _ = _run_first_cell(cfg)
    _, df2, _ = _run_first_cell(cfg)
    pd.testing.assert_frame_equal(df1, df2)
    assert df1["seed"].nunique() == 2
    assert R.seed_for(0, 0, 7) != R.seed_for(1, 0, 7) != R.seed_for(0, 1, 7)


@pytest.mark.parametrize("speaker_level,listener_level",
                         [("S2_vig", "L1"), ("S2_cred", "L1"), ("S2_cred", "L2")])
def test_offline_s2_variants_run(tmp_path, speaker_level, listener_level):
    cfg = _cfg(tmp_path, speaker_level=speaker_level, listener_level=listener_level, n_sims=2, rounds=12,
               switch_types=["hard", "soft"])
    summary, df, tau = _run_first_cell(cfg)
    assert len(df) == 2 * 12 * 4
    assert (df["speaker_level"] == speaker_level).all() and (df["listener_level"] == listener_level).all()
    assert np.isfinite(df["sus1_Sus"]).all()


def test_feedback_layout_and_replica(tmp_path):
    cfg = _cfg(tmp_path, mode="feedback", speaker_level="S2_replica", cs=[2.0], switch_types=["hard"],
               n_sims=3, rounds=25)
    summary, df, tau = _run_first_cell(cfg)
    assert len(df) == 3 * 25
    for col in ("u_pers_ref", "u_inf_ref", "p_chosen", "p_pers_ref", "p_inf_ref", "went_persuasive",
                "went_informative", "margin", "replica_maxdiff"):
        assert col in df.columns, col
    assert df["replica_maxdiff"].max() <= 1e-9
    assert (df["p_chosen"] > 0).all()
    assert ((df["went_persuasive"] == 1) == (df["utt"] == df["u_pers_ref"])).all()
    assert ((df["went_informative"] == 1) == (df["utt"] == df["u_inf_ref"])).all()
    for s, g in df.sort_values("round").groupby("sim"):
        sw = g["switched"].to_numpy()
        trow = tau[tau.sim == s].iloc[0]
        if sw.any():
            k = int(np.argmax(sw))
            assert trow.tau == k + 1
            assert np.isnan(g["sus1_Sus"].to_numpy()[k + 1:]).all()      # test frozen after tau
            assert np.isnan(g["margin"].to_numpy()[k + 1:]).all()
            assert np.isfinite(g["sus1_Sus"].to_numpy()[:k + 1]).all()
        assert np.isnan(g["margin"].to_numpy()[0])                       # no history at round 1
    assert "replica_maxdiff" in tau.columns


def test_feedback_inf_speaker_has_no_persuasive_reference(tmp_path):
    cfg = _cfg(tmp_path, mode="feedback", speaker_level="S2_replica", psi_stars=["inf"], cs=[3.5],
               switch_types=["soft"], n_sims=2, rounds=10)
    _, df, _ = _run_first_cell(cfg)
    assert (df["u_pers_ref"] == -1).all() and (df["went_persuasive"] == -1).all()
    assert (df["u_inf_ref"] >= 0).all()


def test_config_validation(tmp_path):
    with pytest.raises(ValueError):
        _cfg(tmp_path, mode="feedback", speaker_level="S1")
    with pytest.raises(ValueError):
        _cfg(tmp_path, speaker_level="S1", listener_level="L2")     # L2 detector needs the S2_cred speaker
    with pytest.raises(ValueError):
        _cfg(tmp_path, listener_level="L3")
    with pytest.raises(ValueError):
        _cfg(tmp_path, mode="bogus")
    _cfg(tmp_path, speaker_level="S2_cred", listener_level="L1")   # Fang's cooperative dyad: allowed


def test_observation_streams_are_paired_across_cells(tmp_path):
    # Common random numbers: sim i draws the same observations in every cell
    # with the same theta* (offline and feedback alike), and the same uniforms
    # -- hence a monotone-related but different stream -- across theta*.
    cfg = _cfg(tmp_path, theta_stars=[0.3, 0.7], psi_stars=["inf", "high"], alphas=[3.0],
               cs=[2.0], switch_types=["hard"], n_sims=3, rounds=25)
    cells = R.build_cells(cfg)
    frames = {}
    for cell in cells:
        R.run_cell((cfg, cell))
        df = pd.read_parquet(R._shard_path(cfg["out_dir"], cell["cell_id"]))
        frames[(cell["theta_star"], cell["psi_star"])] = df.sort_values(["sim", "round"])
    for th in (0.3, 0.7):
        a, b = frames[(th, "inf")], frames[(th, "high")]
        np.testing.assert_array_equal(a["obs"].values, b["obs"].values)   # same obs, different psi
        assert not np.array_equal(a["utt"].values, b["utt"].values)       # different speaker
    assert not np.array_equal(frames[(0.3, "inf")]["obs"].values, frames[(0.7, "inf")]["obs"].values)

    (tmp_path / "fb").mkdir()
    fcfg = _cfg(tmp_path / "fb", mode="feedback", speaker_level="S2_replica", theta_stars=[0.3],
                psi_stars=["inf", "high"], alphas=[3.0], cs=[2.0], switch_types=["hard", "soft"],
                n_sims=3, rounds=25)
    obs_by_cell = []
    for cell in R.build_cells(fcfg):
        R.run_cell((fcfg, cell))
        df = pd.read_parquet(R._shard_path(fcfg["out_dir"], cell["cell_id"])).sort_values(["sim", "round"])
        obs_by_cell.append(df["obs"].values)
    for o in obs_by_cell[1:]:
        np.testing.assert_array_equal(obs_by_cell[0], o)
    # and the feedback stream of sim i equals the offline stream of sim i at the same theta*
    np.testing.assert_array_equal(obs_by_cell[0], frames[(0.3, "inf")]["obs"].values)
