"""Tests for the Fang-form ``Speaker2`` (Task 3 of switching_experiments_spec.md).

    P_S2(u | O, psi, alpha) ~ Truth(u;O) * Inf(u;O)^(alpha*beta) * PersStr(u;psi)^(alpha*(1-beta))
    Inf(u;O) = P_L1(O | u),  PersStr = E_L1[theta | u] (pers+), 1 - E_L1[theta | u] (pers-), 1 (inf)

(a) an informative S2 modelling a credulous L1 with a uniform state prefers the
    most-informative description ("some" is never dominant);
(b) the product-form identity holds exactly over true utterances;
(c) the policy does not depend on ``belief_theta``;
plus: a ``DetectionListener`` inside S2 is consulted through ``peek`` and a
private replica stays identical to the actual listener round by round.

Run with:  python -m pytest tests/test_speaker2.py -v
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from rsa.core import Belief
from rsa.setup import make_thetas, make_world, make_semantics
from rsa.speaker0 import Speaker0
from rsa.listener0 import Listener0
from rsa.speaker1 import Speaker1
from rsa.listener1 import Listener1
from rsa.speaker2 import Speaker2
from rsa.utils import expected_theta
from rsa.detection import DetectionListener, SequentialTest, SUS_VARIANT_FNS


THETAS = make_thetas(0.1)
PSIS = ["inf", "high", "low"]


def _chain(theta_true=0.3, alpha=3.0):
    world = make_world(theta_true, n=1, m=7)
    sem = make_semantics(n=1)
    s0 = Speaker0(THETAS, semantics=sem, world=world)
    l0 = Listener0(THETAS, s0, semantics=sem, world=world)
    s1 = Speaker1(THETAS, l0, semantics=sem, world=world, alpha=alpha, psi="inf")
    return world, sem, s0, l0, s1


def _obs_with_k_effective(world, k):
    for o in world.generate_all_obs():
        if o.index(1) == k:
            return o
    raise AssertionError


def _drive(world, sem, s0, l0, s1, listener, rounds, seed):
    """Advance the whole chain (and ``listener``) on an informative S1 stream."""
    random.seed(seed); np.random.seed(seed)
    for _ in range(rounds):
        obs = world.sample_obs(); utt = s1.sample_utterance(obs)
        listener.update(utt)
        s1.update(obs); l0.update(utt); s0.update(obs)


# ---------------------------------------------------------------------------
# (a) qualitative behaviour of S2-inf with a credulous, uniform L1
# ---------------------------------------------------------------------------

def test_informative_s2_prefers_most_informative_description():
    world, sem, s0, l0, s1 = _chain(alpha=3.0)
    l1 = Listener1(THETAS, PSIS, s1, world, sem, "inf", 3.0)     # credulous, uniform
    s2 = Speaker2(THETAS, l1, sem, world, alpha=3.0, psi="inf")
    for k in range(8):
        obs = _obs_with_k_effective(world, k)
        dist = s2.dist_over_utterances_obs(obs, "inf")
        best = max(dist, key=dist.get)
        assert best[0] != "some", (k, dist)             # Fang: "some" never dominant
        if k == 4:
            assert best == ("most", "effective"), dist
        if k == 0:
            assert best in {("none", "effective"), ("all", "ineffective")}, dist
            assert abs(dist[("none", "effective")] - dist[("all", "ineffective")]) < 1e-12
        if k == 7:
            assert best in {("all", "effective"), ("none", "ineffective")}, dist
        # every true utterance gets positive mass, every false one none
        for u, p in dist.items():
            assert (p > 0) == sem.truth_value(world, obs, u)


def test_persuasive_s2_pulls_towards_its_goal():
    world, sem, s0, l0, s1 = _chain(alpha=3.0)
    l1 = Listener1(THETAS, PSIS, s1, world, sem, "inf", 3.0)
    s2 = Speaker2(THETAS, l1, sem, world, alpha=3.0, psi="high")
    obs = _obs_with_k_effective(world, 2)
    e_after = {u: expected_theta(l1.infer_state(u).marginal(0)) for u in sem.utterance_space()}
    up = s2.dist_over_utterances_obs(obs, "high")
    down = s2.dist_over_utterances_obs(obs, "low")
    # pers+ mass is concentrated on the true utterance with the highest
    # E_L1[theta | u]; pers- on the lowest
    true_u = [u for u in sem.utterance_space() if sem.truth_value(world, obs, u)]
    assert max(up, key=up.get) == max(true_u, key=e_after.get)
    assert max(down, key=down.get) == min(true_u, key=e_after.get)


# ---------------------------------------------------------------------------
# (b) product-form identity
# ---------------------------------------------------------------------------

def _identity_residual(s2, listener, world, sem, psi, obs):
    alpha = s2.alpha
    beta = 1.0 if psi == "inf" else 0.0
    dist = s2.dist_over_utterances_obs(obs, psi)
    # Inf is the one-step-ahead P(O | u) of the listener that would be active
    # after u: peek_obs when the listener can switch, infer_obs otherwise.
    peek_obs = getattr(listener, "peek_obs", None)
    obs_post = peek_obs if peek_obs is not None else listener.infer_obs
    inf = {u: obs_post(u)[obs] for u in sem.utterance_space()}
    pers = s2.get_persuasiveness(psi)
    consts = []
    for u, p in dist.items():
        if not sem.truth_value(world, obs, u):
            assert p == 0.0
            continue
        assert p > 0.0
        consts.append(np.log(p) - alpha * beta * np.log(inf[u])
                      - alpha * (1 - beta) * np.log(pers[u]))
    return float(np.ptp(consts))


@pytest.mark.parametrize("listener_kind", ["cred", "vig", "detection"])
@pytest.mark.parametrize("psi", PSIS)
def test_product_form_identity(listener_kind, psi):
    world, sem, s0, l0, s1 = _chain(alpha=4.0)
    if listener_kind == "cred":
        listener = Listener1(THETAS, PSIS, s1, world, sem, "inf", 4.0)
    elif listener_kind == "vig":
        listener = Listener1(THETAS, PSIS, s1, world, sem, "vig", 4.0)
    else:
        test = SequentialTest(SUS_VARIANT_FNS["1"], "sus_1", c=2.5,
                              switch_enabled=True, switch_type="hard")
        listener = DetectionListener(THETAS, PSIS, s1, world, sem, tests=[test], alpha=4.0)
    _drive(world, sem, s0, l0, s1, listener, rounds=6, seed=5)   # non-trivial state
    s2 = Speaker2(THETAS, listener, sem, world, alpha=4.0, psi=psi)
    for obs in world.generate_all_obs():
        assert _identity_residual(s2, listener, world, sem, psi, obs) < 1e-9


# ---------------------------------------------------------------------------
# (c) no dependence on belief_theta
# ---------------------------------------------------------------------------

def test_policy_independent_of_belief_theta():
    world, sem, s0, l0, s1 = _chain(alpha=3.0)
    l1 = Listener1(THETAS, PSIS, s1, world, sem, "vig", 3.0)
    s2 = Speaker2(THETAS, l1, sem, world, alpha=3.0, psi="high")
    before = {psi: np.array(s2.obs_utt_table_for_psi(psi), copy=True) for psi in PSIS}
    spike = np.zeros(len(THETAS)); spike[-1] = 1.0
    s2.belief_theta = Belief(THETAS, spike)
    s2.clear_caches()
    for psi in PSIS:
        np.testing.assert_array_equal(s2.obs_utt_table_for_psi(psi), before[psi])
    # belief_theta still updates on O like S1's (bookkeeping only)
    obs = _obs_with_k_effective(world, 6)
    s2.update(obs)
    assert s2.belief_theta.prob[-1] > 0.9


def test_fallback_uniform_literal_when_all_scores_vanish():
    thetas11 = make_thetas(0.1, True, True)
    world = make_world(0.3, n=1, m=7); sem = make_semantics(n=1)

    class Certain:                       # listener certain theta = 1 after any u
        def infer_obs(self, u):
            return {o: 1.0 / 8 for o in world.generate_all_obs()}

        def peek(self, u):
            return {th: (1.0 if th == 1.0 else 0.0) for th in thetas11}

    s2 = Speaker2(thetas11, Certain(), sem, world, alpha=3.0, psi="low")
    for obs in world.generate_all_obs():
        probs = s2.dist_over_utterances_obs_array(obs, "low")     # PersStr = 0 for all u
        row = sem.truth_table(world)[world.obs_index(obs)].astype(float)
        np.testing.assert_allclose(probs, row / row.sum())


# ---------------------------------------------------------------------------
# DetectionListener inside S2: peek is used, replica tracks the actual listener
# ---------------------------------------------------------------------------

def test_s2_consults_detection_listener_through_peek(monkeypatch):
    world, sem, s0, l0, s1 = _chain(alpha=3.0)
    test = SequentialTest(SUS_VARIANT_FNS["1"], "sus_1", c=2.5, switch_enabled=True, switch_type="hard")
    det = DetectionListener(THETAS, PSIS, s1, world, sem, tests=[test], alpha=3.0)
    calls = []
    orig = det.peek
    monkeypatch.setattr(det, "peek", lambda u: (calls.append(u), orig(u))[1])
    s2 = Speaker2(THETAS, det, sem, world, alpha=3.0, psi="high")
    s2.get_persuasiveness("high")
    assert sorted(calls) == sorted(sem.utterance_space())
    # and the persuasiveness values are exactly E[theta] of peek
    pers = s2.get_persuasiveness("high")
    for u in sem.utterance_space():
        assert abs(pers[u] - expected_theta(orig(u))) < 1e-12


@pytest.mark.parametrize("switch_type", ["hard", "soft"])
def test_private_replica_tracks_actual_listener(switch_type):
    """Experiment B invariant: S2's internal replica, fed the same public
    utterances, equals the actual switching listener every round (1e-9)."""
    random.seed(9); np.random.seed(9)
    world, sem, s0, l0, s1 = _chain(theta_true=0.3, alpha=5.0)

    def mk():
        t = SequentialTest(SUS_VARIANT_FNS["1"], "sus_1", c=2.0, switch_enabled=True,
                           switch_type=switch_type)
        return DetectionListener(THETAS, PSIS, s1, world, sem, tests=[t], alpha=5.0)
    replica, actual = mk(), mk()
    s2 = Speaker2(THETAS, replica, sem, world, alpha=5.0, psi="high")
    for t in range(60):
        obs = world.sample_obs(); utt = s2.sample_utterance(obs)
        actual.update(utt); replica.update(utt)
        a, r = actual.marginal_theta(), replica.marginal_theta()
        assert max(abs(a[th] - r[th]) for th in THETAS) < 1e-9
        assert actual.switched == replica.switched
        s2.update(obs); s1.update(obs); l0.update(utt); s0.update(obs)
    assert actual.switched, "a pers+ S2 at alpha=5, c=2 should trip the detector"
    assert actual.switched_at == replica.switched_at
