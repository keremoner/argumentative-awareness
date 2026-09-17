"""Tests for detector-triggered switching in ``DetectionListener`` (Tasks 1-2
of switching_experiments_spec.md).

* ``switch_type="hard"`` is retrospective: after the switch the belief equals
  an always-vigilant L1 run on the same utterances with the same snapshotted
  tables (to 1e-10).
* ``switch_type="hard_amnesic"`` reproduces the old behaviour (uniform at tau,
  never sees u_tau, learns from tau+1).
* ``switch_type="soft"`` inherits the credulous theta-marginal, is uniform over
  psi, and now also sees u_tau vigilantly.
* Snapshot tables at round i equal the live speaker's tables at round i.
* ``peek(u)`` equals deepcopy + ``update(u)`` for every u, at several rounds,
  under every switch type, including a round where some u would trigger.
* Mechanics carried over from the previous test file (c = inf never switches,
  the switch fires exactly at tau, only the first enabled test drives it, the
  utterance stream does not depend on the listeners, ...).

Run with:  python -m pytest tests/test_switching.py -v
"""

from __future__ import annotations

import random
from copy import deepcopy

import numpy as np
import pytest

from rsa.setup import make_thetas, make_world, make_semantics
from rsa.speaker0 import Speaker0
from rsa.listener0 import Listener0
from rsa.speaker1 import Speaker1
from rsa.listener1 import Listener1
from rsa.detection import (
    DetectionListener, SequentialTest, SUS_VARIANT_FNS, ScoreContext, SWITCH_TYPES,
)


THETAS = make_thetas(0.1)              # 9-point grid used by the experiments
PSIS = ["inf", "high", "low"]


def _stack(theta_true, speaker_psi, alpha=3.0):
    world = make_world(theta_true, n=1, m=7)
    semantics = make_semantics(n=1)
    s0 = Speaker0(THETAS, semantics=semantics, world=world)
    l0 = Listener0(THETAS, s0, semantics=semantics, world=world)
    s1 = Speaker1(THETAS, l0, semantics=semantics, world=world,
                  alpha=alpha, psi=speaker_psi)
    return world, semantics, s0, l0, s1


def _det(s1, world, semantics, c, switch_type, alpha=3.0, enabled=True,
         retro_cache=False):
    test = SequentialTest(SUS_VARIANT_FNS["1"], "sus_1", c=c,
                          switch_enabled=enabled, switch_type=switch_type)
    det = DetectionListener(THETAS, PSIS, s1, world, semantics,
                            tests=[test], alpha=alpha, retro_cache=retro_cache)
    return det, test


def _theta_vec(listener):
    d = listener.marginal_theta()
    return np.array([d.get(t, 0.0) for t in THETAS])


def _e_theta(listener):
    return float(np.dot(_theta_vec(listener), THETAS))


def _advance(world, s0, l0, s1, obs, utt):
    s1.update(obs)
    l0.update(utt)
    s0.update(obs)


def run_stream(theta_true, speaker_psi, alpha, rounds, seed, make_listeners):
    """Drive one utterance stream through several listeners at once."""
    random.seed(seed)
    np.random.seed(seed)
    world, semantics, s0, l0, s1 = _stack(theta_true, speaker_psi, alpha)
    listeners = make_listeners(s1, world, semantics)
    e_theta = {k: [] for k in listeners}
    utts = []
    for _ in range(rounds):
        obs = world.sample_obs()
        utt = s1.sample_utterance(obs)
        utts.append(utt)
        for k, L in listeners.items():
            L.update(utt)
            e_theta[k].append(_e_theta(L))
        _advance(world, s0, l0, s1, obs, utt)
    return listeners, {k: np.array(v) for k, v in e_theta.items()}, utts


def _find_seed_with_mid_tau(theta, psi, alpha, c, switch_type, rounds,
                            lo=3, hi=None, seeds=range(50)):
    """A seed for which the switch fires at lo <= tau <= hi."""
    hi = rounds - 5 if hi is None else hi
    for seed in seeds:
        def mk(s1, world, sem):
            d, _ = _det(s1, world, sem, c, switch_type, alpha)
            return {"det": d}
        Ls, _, _ = run_stream(theta, psi, alpha, rounds, seed, mk)
        tau = Ls["det"].switched_at
        if tau is not None and lo <= tau <= hi:
            return seed, tau
    pytest.skip("no seed produced a mid-run switch")


# ---------------------------------------------------------------------------
# Task 1 (a): retrospective hard switch == always-vigilant
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theta,psi,alpha", [(0.3, "high", 5.0), (0.7, "low", 5.0)])
def test_retro_hard_switch_equals_always_vigilant(theta, psi, alpha):
    seed, tau0 = _find_seed_with_mid_tau(theta, psi, alpha, 2.5, "hard", 80)

    def mk(s1, world, sem):
        d, _ = _det(s1, world, sem, 2.5, "hard", alpha)
        return {"det": d,
                "vig": Listener1(THETAS, PSIS, s1, world, sem, "vig", alpha),
                "cred": Listener1(THETAS, ["inf"], s1, world, sem, "inf", alpha)}

    random.seed(seed); np.random.seed(seed)
    world, sem, s0, l0, s1 = _stack(theta, psi, alpha)
    Ls = mk(s1, world, sem)
    det, vig, cred = Ls["det"], Ls["vig"], Ls["cred"]
    for t in range(1, 81):
        obs = world.sample_obs(); utt = s1.sample_utterance(obs)
        det.update(utt); vig.update(utt); cred.update(utt)
        if not det.switched:
            # before tau the switching listener *is* the credulous one
            np.testing.assert_allclose(_theta_vec(det), _theta_vec(cred), atol=1e-12)
        else:
            assert det.switch_type == "hard"
            np.testing.assert_allclose(det.state_belief.prob, vig.state_belief.prob,
                                       atol=1e-10)
        _advance(world, s0, l0, s1, obs, utt)
    assert det.switched and det.switched_at == tau0 >= 3

    # The replay used the snapshotted tables: rebuilding an always-vigilant
    # listener from det.table_history alone gives the same belief at tau.
    tau = det.switched_at
    rebuilt = Listener1(THETAS, PSIS, s1, world, sem, "vig", alpha)
    for u, tab in zip(det.utt_history[:tau], det.table_history[:tau]):
        rebuilt.update_with_tables(u, tab)
    np.testing.assert_allclose(rebuilt.state_belief.prob,
                               det.vigilant.hist[tau].prob, atol=1e-10)
    # and the retrospective vigilant listener saw exactly u_1..u_tau before
    # continuing live
    assert det.vigilant.utt_history == det.utt_history


# ---------------------------------------------------------------------------
# Task 1 (b): hard_amnesic reproduces the old behaviour
# ---------------------------------------------------------------------------

def test_hard_amnesic_reproduces_old_behaviour():
    """Old ``hard``: fresh uniform vigilant at tau with no history; it never
    sees u_tau and is updated live from tau+1.  A vigilant Listener1 forked
    *after* the switch round must match it exactly, and the joint at tau is
    uniform."""
    seed, _ = _find_seed_with_mid_tau(0.5, "high", 10.0, 3.0, "hard_amnesic", 120)
    random.seed(seed); np.random.seed(seed)
    world, sem, s0, l0, s1 = _stack(0.5, "high", 10.0)
    det, _ = _det(s1, world, sem, 3.0, "hard_amnesic", 10.0)
    fork = None
    diffs = []
    for t in range(1, 121):
        obs = world.sample_obs(); utt = s1.sample_utterance(obs)
        det.update(utt)
        if fork is not None:
            fork.update(utt)
        elif det.switched:
            joint = det.vigilant.state_belief.prob
            np.testing.assert_allclose(joint, 1.0 / joint.size, atol=1e-12)
            assert det.vigilant.utt_history == []          # no history, no u_tau
            fork = Listener1(THETAS, PSIS, s1, world, sem, "vig", 10.0)
        if fork is not None:
            diffs.append(np.abs(fork.state_belief.prob - det.vigilant.state_belief.prob).max())
        _advance(world, s0, l0, s1, obs, utt)
    assert det.switched and det.switch_type == "hard_amnesic"
    assert max(diffs) < 1e-12


# ---------------------------------------------------------------------------
# Task 1 (c): snapshot tables at round i == live speaker tables at round i
# ---------------------------------------------------------------------------

def test_snapshot_tables_match_live_speaker_each_round():
    random.seed(3); np.random.seed(3)
    world, sem, s0, l0, s1 = _stack(0.3, "high", 3.0)
    det, _ = _det(s1, world, sem, float("inf"), "hard", 3.0)
    live = []
    for t in range(40):
        obs = world.sample_obs(); utt = s1.sample_utterance(obs)
        live.append({p: np.array(s1.obs_utt_table_for_psi(p), copy=True) for p in PSIS})
        det.update(utt)
        _advance(world, s0, l0, s1, obs, utt)
    assert len(det.table_history) == 40
    for i in range(40):
        for p in PSIS:
            np.testing.assert_array_equal(det.table_history[i][p], live[i][p])
    # tables genuinely change over rounds (so the snapshot is not vacuous)
    assert np.abs(live[0]["inf"] - live[-1]["inf"]).max() > 1e-6
    # and are independent copies of the speaker's cache
    s1.obs_utt_table_for_psi("inf")[0, 0] = -1.0
    assert det.table_history[-1]["inf"][0, 0] != -1.0


# ---------------------------------------------------------------------------
# Task 1 (4)-(5): soft keeps the credulous marginal, uniform psi, sees u_tau
# ---------------------------------------------------------------------------

def test_soft_switch_inherits_marginal_and_sees_u_tau():
    seed, _ = _find_seed_with_mid_tau(0.3, "high", 5.0, 2.5, "soft", 80)
    random.seed(seed); np.random.seed(seed)
    world, sem, s0, l0, s1 = _stack(0.3, "high", 5.0)
    det, _ = _det(s1, world, sem, 2.5, "soft", 5.0)
    for t in range(1, 81):
        obs = world.sample_obs(); utt = s1.sample_utterance(obs)
        det.update(utt)
        _advance(world, s0, l0, s1, obs, utt)
        if det.switched:
            break
    assert det.switched and det.switch_type == "soft"
    tau = det.switched_at
    # the credulous listener absorbed u_tau; the seeded joint is its marginal
    # spread uniformly over psi ...
    seeded = det.vigilant.hist[0]
    naive_theta = det.naive.marginal_theta()
    for (th, p), prob in zip(seeded.values, seeded.prob):
        assert abs(prob - naive_theta[th] / len(PSIS)) < 1e-12
    # ... then u_tau was applied vigilantly with the snapshot tables
    ref = Listener1(THETAS, PSIS, s1, world, sem, "vig", 5.0)
    ref.seed_from_theta_marginal(naive_theta)
    ref.update_with_tables(det.utt_history[tau - 1], det.table_history[tau - 1])
    np.testing.assert_allclose(det.state_belief.prob, ref.state_belief.prob, atol=1e-12)
    assert det.vigilant.utt_history == [det.utt_history[tau - 1]]


# ---------------------------------------------------------------------------
# Task 2: cleanup, infer_obs, peek
# ---------------------------------------------------------------------------

def test_l0_machinery_removed():
    world, sem, s0, l0, s1 = _stack(0.3, "inf", 3.0)
    det = DetectionListener(THETAS, PSIS, s1, world, sem)
    assert not hasattr(det, "naive_l0")
    assert not hasattr(det, "_l0_theta_array")
    ctx = det.build_context(sem.utterance_space()[0])
    assert not hasattr(ctx, "L0_O")
    # l0_theta is still accepted (test_sus_variants passes it) but unused
    ctx2 = ScoreContext(THETAS, s1, world, sem, det._l1_theta_array(),
                        sem.utterance_space()[0], l0_theta=np.ones(len(THETAS)) / len(THETAS))
    np.testing.assert_allclose(ctx2.P_prior, ctx.P_prior)


def test_infer_obs_delegates_to_active_sublistener():
    seed, _ = _find_seed_with_mid_tau(0.3, "high", 5.0, 2.5, "hard", 60)
    random.seed(seed); np.random.seed(seed)
    world, sem, s0, l0, s1 = _stack(0.3, "high", 5.0)
    det, _ = _det(s1, world, sem, 2.5, "hard", 5.0)
    u0 = sem.utterance_space()[3]
    checked_pre = checked_post = False
    for t in range(60):
        obs = world.sample_obs(); utt = s1.sample_utterance(obs)
        det.update(utt)
        _advance(world, s0, l0, s1, obs, utt)
        d = det.infer_obs(u0)
        if det.switched:
            ref = det.vigilant.infer_obs(u0); checked_post = True
        else:
            ref = det.naive.infer_obs(u0); checked_pre = True
        assert d == ref
        assert abs(sum(d.values()) - 1.0) < 1e-9
    assert checked_pre and checked_post


def _peek_vs_copy(det):
    """max |peek(u) - deepcopy+update(u)| over u, plus whether any u would switch."""
    sem = det.semantics
    worst = 0.0
    any_switch = False
    for u in sem.utterance_space():
        if det._would_switch(u) is not None:
            any_switch = True
        pk = det.peek(u)
        cp = deepcopy(det)
        cp.update(u)
        ref = cp.marginal_theta()
        worst = max(worst, max(abs(pk[th] - ref[th]) for th in THETAS))
    return worst, any_switch


@pytest.mark.parametrize("switch_type", SWITCH_TYPES)
def test_peek_equals_deepcopy_update(switch_type):
    seed, tau = _find_seed_with_mid_tau(0.3, "high", 5.0, 3.0, switch_type, 60, lo=4)
    random.seed(seed); np.random.seed(seed)
    world, sem, s0, l0, s1 = _stack(0.3, "high", 5.0)
    det, test = _det(s1, world, sem, 3.0, switch_type, 5.0)
    check_rounds = {1, 3, tau - 1, tau, tau + 3, 30}
    saw_trigger_round = False
    for t in range(1, 61):
        obs = world.sample_obs(); utt = s1.sample_utterance(obs)
        if t in check_rounds:
            worst, any_switch = _peek_vs_copy(det)
            assert worst < 1e-10, (t, worst)
            if t == tau:
                # the round where the actual utterance triggers: some
                # candidates cross the boundary, and peek must reflect the
                # switch consequence for those
                assert any_switch and det._would_switch(utt) is not None
                saw_trigger_round = True
            # peek never mutates
            assert det.round == t - 1 and len(det.utt_history) == t - 1
            assert len(test.history["scores"]) == min(t - 1, tau if det.switched else t - 1)
        det.update(utt)
        _advance(world, s0, l0, s1, obs, utt)
    assert det.switched_at == tau and saw_trigger_round


def test_peek_retro_cache_gives_identical_results():
    seed, tau = _find_seed_with_mid_tau(0.3, "high", 5.0, 3.0, "hard", 60, lo=4)
    random.seed(seed); np.random.seed(seed)
    world, sem, s0, l0, s1 = _stack(0.3, "high", 5.0)
    det_a, _ = _det(s1, world, sem, 3.0, "hard", 5.0, retro_cache=False)
    det_b, _ = _det(s1, world, sem, 3.0, "hard", 5.0, retro_cache=True)
    for t in range(1, tau + 5):
        obs = world.sample_obs(); utt = s1.sample_utterance(obs)
        for u in sem.utterance_space():
            pa, pb = det_a.peek(u), det_b.peek(u)
            assert max(abs(pa[th] - pb[th]) for th in THETAS) < 1e-12
        # calling peek twice in a round hits the cache the second time
        for u in sem.utterance_space():
            pb2 = det_b.peek(u)
            assert max(abs(det_a.peek(u)[th] - pb2[th]) for th in THETAS) < 1e-12
        det_a.update(utt); det_b.update(utt)
        _advance(world, s0, l0, s1, obs, utt)
    assert det_a.switched_at == det_b.switched_at == tau
    np.testing.assert_allclose(det_a.state_belief.prob, det_b.state_belief.prob, atol=1e-12)


def test_peek_after_switch_is_vigilant_one_step():
    seed, tau = _find_seed_with_mid_tau(0.3, "high", 5.0, 2.5, "hard", 40)
    random.seed(seed); np.random.seed(seed)
    world, sem, s0, l0, s1 = _stack(0.3, "high", 5.0)
    det, _ = _det(s1, world, sem, 2.5, "hard", 5.0)
    for t in range(1, 41):
        obs = world.sample_obs(); utt = s1.sample_utterance(obs)
        det.update(utt)
        _advance(world, s0, l0, s1, obs, utt)
    assert det.switched
    for u in sem.utterance_space():
        ref = det.vigilant.infer_state(u).marginal(0)
        pk = det.peek(u)
        assert max(abs(pk[th] - ref[th]) for th in THETAS) < 1e-12


def test_unknown_switch_type_raises():
    world, sem, s0, l0, s1 = _stack(0.3, "high", 5.0)
    det, _ = _det(s1, world, sem, 0.01, "bogus", 5.0)
    with pytest.raises(ValueError):
        for t in range(20):
            obs = world.sample_obs(); utt = s1.sample_utterance(obs)
            det.update(utt)
            _advance(world, s0, l0, s1, obs, utt)


# ---------------------------------------------------------------------------
# Mechanics (carried over)
# ---------------------------------------------------------------------------

def test_c_inf_never_switches_and_equals_credulous():
    def mk(s1, world, sem):
        det, _ = _det(s1, world, sem, float("inf"), "soft")
        cred = Listener1(THETAS, ["inf"], s1, world, sem, listener_type="inf", alpha=3.0)
        return {"det": det, "cred": cred}

    Ls, _, _ = run_stream(0.3, "high", 3.0, 60, 1, mk)
    assert not Ls["det"].switched
    assert Ls["det"].switched_at is None
    np.testing.assert_allclose(_theta_vec(Ls["det"]), _theta_vec(Ls["cred"]), atol=1e-12)


def test_switch_fires_exactly_at_tau():
    def mk(s1, world, sem):
        det, test = _det(s1, world, sem, 1.5, "soft")
        mk.test = test
        return {"det": det}

    Ls, _, _ = run_stream(0.3, "high", 3.0, 80, 2, mk)
    det, test = Ls["det"], mk.test
    assert det.switched
    assert det.switched_at == test.tau
    assert det.switch_driver is test
    assert len(test.history["scores"]) == test.tau
    assert test.history["crossed"][test.tau - 1] is True
    assert not any(test.history["crossed"][: test.tau - 1])


def test_switch_disabled_never_switches_even_when_crossing():
    def mk(s1, world, sem):
        det, test = _det(s1, world, sem, 1.5, "soft", enabled=False)
        mk.test = test
        return {"det": det}

    Ls, _, _ = run_stream(0.3, "high", 3.0, 80, 2, mk)
    assert mk.test.tau is not None
    assert not Ls["det"].switched
    assert len(mk.test.history["scores"]) == 80


def test_only_first_enabled_test_drives_the_switch():
    def mk(s1, world, sem):
        t_a = SequentialTest(SUS_VARIANT_FNS["1"], "a", c=1.5, switch_enabled=True,
                             switch_type="hard")
        t_b = SequentialTest(SUS_VARIANT_FNS["1"], "b", c=1.0, switch_enabled=True,
                             switch_type="soft")
        det = DetectionListener(THETAS, PSIS, s1, world, sem, tests=[t_a, t_b], alpha=3.0)
        mk.tests = (t_a, t_b)
        return {"det": det}

    Ls, _, _ = run_stream(0.3, "high", 3.0, 80, 3, mk)
    det = Ls["det"]
    t_a, t_b = mk.tests
    assert det.switched
    taus = [t.tau for t in (t_a, t_b) if t.tau is not None]
    assert det.switched_at == min(taus)
    first_at_min = next(t for t in (t_a, t_b) if t.tau == det.switched_at)
    assert det.switch_driver is first_at_min
    assert det.switch_type == first_at_min.switch_type


def test_after_switch_naive_freezes_and_vigilant_updates():
    seed, _ = _find_seed_with_mid_tau(0.3, "high", 3.0, 1.5, "soft", 100)
    random.seed(seed); np.random.seed(seed)
    world, sem, s0, l0, s1 = _stack(0.3, "high", 3.0)
    det, test = _det(s1, world, sem, 1.5, "soft")
    switched_round = None
    for t in range(1, 101):
        obs = world.sample_obs(); utt = s1.sample_utterance(obs)
        det.update(utt)
        _advance(world, s0, l0, s1, obs, utt)
        if det.switched and switched_round is None:
            switched_round = t
            naive_hist_len = len(det.naive.hist)
            theta_at_switch = _theta_vec(det).copy()
    assert switched_round is not None and switched_round < 100
    assert len(det.naive.hist) == naive_hist_len          # frozen
    assert len(test.history["scores"]) == switched_round  # tests frozen too
    assert np.abs(_theta_vec(det) - theta_at_switch).max() > 1e-6
    assert det.state_belief is det.vigilant.state_belief


def test_utterance_stream_independent_of_listeners():
    def mk_one(s1, world, sem):
        return {"c": Listener1(THETAS, ["inf"], s1, world, sem, "inf", 3.0)}

    def mk_many(s1, world, sem):
        d, _ = _det(s1, world, sem, 1.5, "hard")
        v = Listener1(THETAS, PSIS, s1, world, sem, "vig", 3.0)
        return {"c": Listener1(THETAS, ["inf"], s1, world, sem, "inf", 3.0),
                "det": d, "vig": v}

    _, e1, u1 = run_stream(0.5, "low", 5.0, 50, 11, mk_one)
    _, e2, u2 = run_stream(0.5, "low", 5.0, 50, 11, mk_many)
    assert u1 == u2
    np.testing.assert_allclose(e1["c"], e2["c"], atol=1e-12)


# ---------------------------------------------------------------------------
# Function: does switching do what it is for?
# ---------------------------------------------------------------------------

STRONG = 10.0


def _policies_at(alpha, c=3.0):
    def mk(s1, world, sem):
        hard, _ = _det(s1, world, sem, c, "hard", alpha)
        soft, _ = _det(s1, world, sem, c, "soft", alpha)
        return {
            "credulous": Listener1(THETAS, ["inf"], s1, world, sem, "inf", alpha),
            "vigilant": Listener1(THETAS, PSIS, s1, world, sem, "vig", alpha),
            "switch_hard": hard,
            "switch_soft": soft,
        }
    return mk


def test_switching_recovers_after_tau():
    """Paired on one stream: before tau the switching listeners track the
    credulous one; from tau on, retro equals always-vigilant and soft's
    error shrinks relative to credulous."""
    found = 0
    for seed in range(4):
        Ls, e, _ = run_stream(0.5, "high", STRONG, 150, 200 + seed,
                              _policies_at(STRONG, c=3.0))
        hard, soft = Ls["switch_hard"], Ls["switch_soft"]
        if not hard.switched or hard.switched_at > 100:
            continue
        found += 1
        tau = hard.switched_at
        assert soft.switched_at == tau
        np.testing.assert_allclose(e["switch_hard"][:tau - 1], e["credulous"][:tau - 1], atol=1e-10)
        np.testing.assert_allclose(e["switch_soft"][:tau - 1], e["credulous"][:tau - 1], atol=1e-10)
        np.testing.assert_allclose(e["switch_hard"][tau - 1:], e["vigilant"][tau - 1:], atol=1e-10)
        err_c = abs(e["credulous"][-1] - 0.5)
        err_s = abs(e["switch_soft"][-1] - 0.5)
        assert err_s < err_c, (seed, tau, err_c, err_s)
    assert found >= 2, "too few seeds switched early enough to test recovery"


def test_under_informative_speaker_switch_rarely_fires_and_beliefs_converge():
    n_switch = 0
    for seed in range(4):
        Ls, e, _ = run_stream(0.6, "inf", 3.0, 150, 300 + seed, _policies_at(3.0, c=3.5))
        n_switch += int(Ls["switch_soft"].switched)
        for k in ("credulous", "vigilant"):
            assert abs(e[k][-1] - 0.6) < 0.1, (k, e[k][-1])
    assert n_switch <= 1, "c=3.5 should seldom fire on an honest speaker"
