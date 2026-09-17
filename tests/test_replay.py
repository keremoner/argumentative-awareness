"""Tests for ``rsa.detection.replay`` (Task 4 of switching_experiments_spec.md).

* ``regenerate_tables`` rebuilds the per-round S1 tables from a stored
  (obs, utt) stream exactly.
* ``replay_listener`` on stored tables reproduces a live Listener1 and a live
  DetectionListener (all switch types), including tau.
* ``splice`` equals a direct replay of the retrospective switching listener.
* ``switching_trajectory`` (soft / hard_amnesic) equals a direct replay.
* ``tau_for_rule`` on observer trajectories equals the live test's tau.

Run with:  python -m pytest tests/test_replay.py -v
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from rsa.setup import make_thetas, make_world, make_semantics
from rsa.speaker0 import Speaker0
from rsa.listener0 import Listener0
from rsa.speaker1 import Speaker1
from rsa.listener1 import Listener1
from rsa.detection import (
    DetectionListener, SequentialTest, SUS_VARIANT_FNS,
    replay_listener, splice, tau_for_rule, regenerate_tables, switching_trajectory,
)
from rsa.detection.replay import empty_trajectory, record_round


THETAS = make_thetas(0.1)
PSIS = ["inf", "high", "low"]


def _generate(theta, psi, alpha, rounds, seed):
    """One live stream with cred / vig / observer-detector trajectories."""
    random.seed(seed); np.random.seed(seed)
    world = make_world(theta, n=1, m=7); sem = make_semantics(n=1)
    s0 = Speaker0(THETAS, semantics=sem, world=world)
    l0 = Listener0(THETAS, s0, semantics=sem, world=world)
    s1 = Speaker1(THETAS, l0, semantics=sem, world=world, alpha=alpha, psi=psi)
    cred = Listener1(THETAS, ["inf"], s1, world, sem, "inf", alpha)
    vig = Listener1(THETAS, PSIS, s1, world, sem, "vig", alpha)
    obs_test = SequentialTest(SUS_VARIANT_FNS["1"], "sus_1", c=float("inf"))
    observer = DetectionListener(THETAS, PSIS, s1, world, sem, tests=[obs_test], alpha=alpha)
    utts, obss, tables = [], [], []
    cred_t = empty_trajectory(rounds, THETAS, PSIS)
    vig_t = empty_trajectory(rounds, THETAS, PSIS)
    for i in range(rounds):
        obs = world.sample_obs(); utt = s1.sample_utterance(obs)
        utts.append(utt); obss.append(obs)
        cred.update(utt); vig.update(utt); observer.update(utt)
        record_round(cred_t, i, cred); record_round(vig_t, i, vig)
        s1.update(obs); l0.update(utt); s0.update(obs)
    tables = observer.table_history
    sus = np.array(obs_test.history["running_mean"])
    sig = np.array(obs_test.history["running_sigma"])
    return dict(world=world, sem=sem, alpha=alpha, utts=utts, obss=obss,
                tables=tables, cred=cred_t, vig=vig_t, sus=sus, sig=sig)


def _live_switching(theta, psi, alpha, rounds, seed, c, switch_type):
    """Trajectory of a live DetectionListener on the same seeded stream."""
    random.seed(seed); np.random.seed(seed)
    world = make_world(theta, n=1, m=7); sem = make_semantics(n=1)
    s0 = Speaker0(THETAS, semantics=sem, world=world)
    l0 = Listener0(THETAS, s0, semantics=sem, world=world)
    s1 = Speaker1(THETAS, l0, semantics=sem, world=world, alpha=alpha, psi=psi)
    test = SequentialTest(SUS_VARIANT_FNS["1"], "sus_1", c=c, switch_enabled=True,
                          switch_type=switch_type)
    det = DetectionListener(THETAS, PSIS, s1, world, sem, tests=[test], alpha=alpha)
    traj = empty_trajectory(rounds, THETAS, PSIS)
    for i in range(rounds):
        obs = world.sample_obs(); utt = s1.sample_utterance(obs)
        det.update(utt)
        record_round(traj, i, det)
        s1.update(obs); l0.update(utt); s0.update(obs)
    traj.switched_at = det.switched_at
    return traj


def _assert_traj_equal(a, b, atol=1e-10):
    np.testing.assert_allclose(a.E_theta, b.E_theta, atol=atol)
    np.testing.assert_allclose(a.std_theta, b.std_theta, atol=atol)
    np.testing.assert_allclose(a.theta, b.theta, atol=atol)
    np.testing.assert_array_equal(np.isnan(a.psi), np.isnan(b.psi))
    m = ~np.isnan(a.psi)
    np.testing.assert_allclose(a.psi[m], b.psi[m], atol=atol)
    assert a.switched_at == b.switched_at


# ---------------------------------------------------------------------------

def test_regenerate_tables_matches_snapshots():
    g = _generate(0.3, "high", 5.0, 50, seed=1)
    regen = regenerate_tables(g["utts"], g["obss"], THETAS, g["alpha"], 1, 7)
    obs_idx = [g["world"].obs_index(o) for o in g["obss"]]
    regen_idx = regenerate_tables(g["utts"], obs_idx, THETAS, g["alpha"], 1, 7)
    assert len(regen) == 50
    for i in range(50):
        for p in PSIS:
            np.testing.assert_array_equal(regen[i][p], g["tables"][i][p])
            np.testing.assert_array_equal(regen_idx[i][p], g["tables"][i][p])


@pytest.mark.parametrize("kind", ["cred", "vig"])
def test_replay_listener1_matches_live(kind):
    g = _generate(0.5, "low", 3.0, 60, seed=2)
    world, sem, alpha = g["world"], g["sem"], g["alpha"]

    def factory(sp):
        if kind == "cred":
            return Listener1(THETAS, ["inf"], sp, world, sem, "inf", alpha)
        return Listener1(THETAS, PSIS, sp, world, sem, "vig", alpha)

    traj = replay_listener(g["utts"], g["tables"], factory, THETAS, world, sem)
    _assert_traj_equal(traj, g[kind])
    if kind == "cred":
        assert np.isnan(traj.psi).all()
    else:
        assert not np.isnan(traj.psi).any()


@pytest.mark.parametrize("switch_type", ["hard", "soft", "hard_amnesic"])
def test_replay_detection_listener_matches_live(switch_type):
    theta, psi, alpha, rounds, seed, c = 0.3, "high", 5.0, 60, 3, 2.5
    g = _generate(theta, psi, alpha, rounds, seed)
    world, sem = g["world"], g["sem"]

    def factory(sp):
        t = SequentialTest(SUS_VARIANT_FNS["1"], "sus_1", c=c, switch_enabled=True,
                           switch_type=switch_type)
        return DetectionListener(THETAS, PSIS, sp, world, sem, tests=[t], alpha=alpha)

    replayed = replay_listener(g["utts"], g["tables"], factory, THETAS, world, sem)
    live = _live_switching(theta, psi, alpha, rounds, seed, c, switch_type)
    assert live.switched_at is not None
    _assert_traj_equal(replayed, live)


def test_tau_for_rule_matches_live_test():
    g = _generate(0.3, "high", 5.0, 100, seed=4)
    for c in (1.5, 2.0, 2.5, 3.5, 100.0, float("inf")):
        expected = _live_switching(0.3, "high", 5.0, 100, 4, c, "hard").switched_at
        assert tau_for_rule(g["sus"], g["sig"], c) == expected
    # sanity on the definition itself
    t = np.arange(1, 101)
    assert tau_for_rule(np.ones(100), np.zeros(100), 1.0) is None      # sigma == 0 never fires
    sus = np.zeros(100); sus[9] = 10.0
    assert tau_for_rule(sus, np.ones(100), 1.0) == 10


@pytest.mark.parametrize("theta,psi,seed", [(0.3, "high", 10), (0.7, "low", 11),
                                            (0.5, "high", 12), (0.9, "low", 13)])
def test_splice_equals_direct_retro_replay(theta, psi, seed):
    alpha, rounds, c = 5.0, 80, 2.5
    g = _generate(theta, psi, alpha, rounds, seed)
    tau = tau_for_rule(g["sus"], g["sig"], c)
    assert tau is not None
    spliced = splice(g["cred"], g["vig"], tau)
    direct = _live_switching(theta, psi, alpha, rounds, seed, c, "hard")
    _assert_traj_equal(spliced, direct)
    # before tau: credulous; from tau: vigilant, psi present only from tau
    assert np.isnan(spliced.psi[: tau - 1]).all() and not np.isnan(spliced.psi[tau - 1:]).any()


@pytest.mark.parametrize("switch_type", ["soft", "hard_amnesic", "hard"])
def test_switching_trajectory_matches_direct_replay(switch_type):
    theta, psi, alpha, rounds, seed, c = 0.3, "high", 5.0, 80, 14, 2.5
    g = _generate(theta, psi, alpha, rounds, seed)
    tau = tau_for_rule(g["sus"], g["sig"], c)
    assert tau is not None
    derived = switching_trajectory(g["cred"], g["vig"], g["utts"], g["tables"], tau,
                                   switch_type, THETAS, g["world"], g["sem"], alpha)
    direct = _live_switching(theta, psi, alpha, rounds, seed, c, switch_type)
    _assert_traj_equal(derived, direct)


def test_switching_trajectory_without_tau_is_credulous():
    g = _generate(0.3, "inf", 3.0, 30, seed=15)
    for st in ("hard", "soft", "hard_amnesic"):
        d = switching_trajectory(g["cred"], g["vig"], g["utts"], g["tables"], None,
                                 st, THETAS, g["world"], g["sem"], 3.0)
        np.testing.assert_allclose(d.E_theta, g["cred"].E_theta)
        assert d.switched_at is None and np.isnan(d.psi).all()
