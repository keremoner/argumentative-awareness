"""Sanity checks for detector-triggered switching in DetectionListener.

The switching machinery was implemented but never exercised by an experiment.
Before building one on it, these tests pin down what it actually does:

* c = inf never switches, and the listener is then exactly the credulous L1.
* A switch fires at the driving test's tau and nowhere else; only the first
  switch-enabled test can drive it; switch_enabled=False never switches.
* Hard switch: the vigilant listener starts from a uniform joint prior.
* Soft switch: it inherits the credulous theta-marginal, uniform over psi.
* After the switch, beliefs come from the vigilant listener and keep updating,
  while the credulous one and the tests are frozen.
* Functionally: a persuasive speaker misleads the credulous listener; the
  vigilant listener resists; the switching listener recovers after tau.
* The utterance stream does not depend on which listeners are attached, so
  policies can be compared on identical noise.

Run with:  python -m pytest tests/test_switching.py -v
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
from rsa.detection import DetectionListener, SequentialTest, SUS_VARIANT_FNS


THETAS = make_thetas(0.1, True, True)
PSIS = ["inf", "high", "low"]
PSI_CODE = {"inf": "inf", "pers+": "high", "pers-": "low"}


def _stack(theta_true, speaker_psi, alpha=3.0):
    world = make_world(theta_true, n=1, m=7)
    semantics = make_semantics(n=1)
    s0 = Speaker0(THETAS, semantics=semantics, world=world)
    l0 = Listener0(THETAS, s0, semantics=semantics, world=world)
    s1 = Speaker1(THETAS, l0, semantics=semantics, world=world,
                  alpha=alpha, psi=speaker_psi)
    return world, semantics, s0, l0, s1


def _switch_listener(s1, world, semantics, c, switch_type, alpha=3.0,
                     enabled=True):
    test = SequentialTest(SUS_VARIANT_FNS["1"], "sus_1", c=c,
                          switch_enabled=enabled, switch_type=switch_type)
    return DetectionListener(THETAS, PSIS, s1, world, semantics,
                             tests=[test], alpha=alpha), test


def _theta_vec(listener):
    d = listener.marginal_theta()
    return np.array([d.get(t, 0.0) for t in THETAS])


def _e_theta(listener):
    return float(np.dot(_theta_vec(listener), THETAS))


def run_stream(theta_true, speaker_psi, alpha, rounds, seed, make_listeners):
    """Drive one utterance stream through several listeners at once.

    ``make_listeners(s1, world, semantics)`` returns a dict name -> listener,
    each exposing ``update(utt)`` and ``marginal_theta()``.  Returns the
    listeners plus per-round E[theta] for each.
    """
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
        s1.update(obs)
        l0.update(utt)
        s0.update(obs)
    return listeners, {k: np.array(v) for k, v in e_theta.items()}, utts


# ---------------------------------------------------------------------------
# Mechanics
# ---------------------------------------------------------------------------

def test_c_inf_never_switches_and_equals_credulous():
    """Pure-observer mode is what the sweep ran: no switch, beliefs are L1^inf."""
    def mk(s1, world, sem):
        det, _ = _switch_listener(s1, world, sem, c=float("inf"), switch_type="soft")
        cred = Listener1(THETAS, ["inf"], s1, world, sem, listener_type="inf", alpha=3.0)
        return {"det": det, "cred": cred}

    Ls, _, _ = run_stream(0.3, "high", 3.0, 60, 1, mk)
    assert not Ls["det"].switched
    assert Ls["det"].switched_at is None
    np.testing.assert_allclose(_theta_vec(Ls["det"]), _theta_vec(Ls["cred"]),
                               atol=1e-12)


def test_switch_fires_exactly_at_tau():
    def mk(s1, world, sem):
        det, test = _switch_listener(s1, world, sem, c=1.5, switch_type="soft")
        mk.test = test
        return {"det": det}

    Ls, _, _ = run_stream(0.3, "high", 3.0, 80, 2, mk)
    det, test = Ls["det"], mk.test
    assert det.switched, "persuasive speaker at c=1.5 should trigger within 80 rounds"
    assert det.switched_at == test.tau
    assert det.switch_driver is test
    # the test observed exactly tau rounds and then stopped
    assert len(test.history["scores"]) == test.tau
    assert test.history["crossed"][test.tau - 1] is True
    assert not any(test.history["crossed"][: test.tau - 1])


def test_switch_disabled_never_switches_even_when_crossing():
    def mk(s1, world, sem):
        det, test = _switch_listener(s1, world, sem, c=1.5, switch_type="soft",
                                     enabled=False)
        mk.test = test
        return {"det": det}

    Ls, _, _ = run_stream(0.3, "high", 3.0, 80, 2, mk)
    assert mk.test.tau is not None, "should still record a crossing"
    assert not Ls["det"].switched
    assert len(mk.test.history["scores"]) == 80   # keeps observing


def test_only_first_enabled_test_drives_the_switch():
    def mk(s1, world, sem):
        t_a = SequentialTest(SUS_VARIANT_FNS["1"], "a", c=1.5, switch_enabled=True,
                             switch_type="hard")
        t_b = SequentialTest(SUS_VARIANT_FNS["1"], "b", c=1.0, switch_enabled=True,
                             switch_type="soft")
        det = DetectionListener(THETAS, PSIS, s1, world, sem, tests=[t_a, t_b],
                                alpha=3.0)
        mk.tests = (t_a, t_b)
        return {"det": det}

    Ls, _, _ = run_stream(0.3, "high", 3.0, 80, 3, mk)
    det = Ls["det"]
    t_a, t_b = mk.tests
    assert det.switched
    # Rule: the switch fires at the earliest crossing of any enabled test; if
    # two cross on the same round, the first in list order drives it.
    taus = [t.tau for t in (t_a, t_b) if t.tau is not None]
    assert det.switched_at == min(taus)
    first_at_min = next(t for t in (t_a, t_b) if t.tau == det.switched_at)
    assert det.switch_driver is first_at_min
    assert det.vigilant is not None
    # and switching happened exactly once
    assert len(det.vigilant.hist) == 1 + (80 - det.switched_at)


def test_manual_fork_at_tau_equals_hard_switch_listener():
    """The experiment runs one soft DetectionListener and, at its tau, forks a
    fresh uniform vigilant Listener1 for the hard policy instead of paying for a
    second detector. That must reproduce a real hard DetectionListener exactly:
    identical tau, identical beliefs at every later round."""
    random.seed(8); np.random.seed(8)
    world, sem, s0, l0, s1 = _stack(0.5, "high", 10.0)
    soft, _ = _switch_listener(s1, world, sem, c=3.0, switch_type="soft", alpha=10.0)
    hard, _ = _switch_listener(s1, world, sem, c=3.0, switch_type="hard", alpha=10.0)
    fork = None
    diffs = []
    for t in range(1, 121):
        obs = world.sample_obs(); utt = s1.sample_utterance(obs)
        soft.update(utt); hard.update(utt)
        if fork is not None:
            fork.update(utt)
        elif soft.switched:
            # created after the switch-round update, never sees that utterance --
            # exactly what DetectionListener._trigger_switch("hard") does
            fork = Listener1(THETAS, PSIS, s1, world, sem, "vig", 10.0)
            assert hard.switched and hard.switched_at == soft.switched_at
        if fork is not None:
            diffs.append(np.abs(fork.state_belief.prob - hard.vigilant.state_belief.prob).max())
        s1.update(obs); l0.update(utt); s0.update(obs)
    assert fork is not None, "expected a switch at alpha=10 under pers+"
    assert max(diffs) < 1e-12


def test_hard_switch_starts_from_uniform_joint():
    def mk(s1, world, sem):
        det, _ = _switch_listener(s1, world, sem, c=1.5, switch_type="hard")
        return {"det": det}

    # Stop the stream at the switch round to inspect the fresh vigilant prior.
    random.seed(4); np.random.seed(4)
    world, sem, s0, l0, s1 = _stack(0.3, "high", 3.0)
    det, test = _switch_listener(s1, world, sem, c=1.5, switch_type="hard")
    for t in range(1, 121):
        obs = world.sample_obs(); utt = s1.sample_utterance(obs)
        det.update(utt); s1.update(obs); l0.update(utt); s0.update(obs)
        if det.switched:
            break
    assert det.switched and det.switched_at == t
    joint = det.vigilant.state_belief.prob
    assert joint.shape == (len(THETAS) * len(PSIS),)
    np.testing.assert_allclose(joint, 1.0 / joint.size, atol=1e-12)
    np.testing.assert_allclose(_theta_vec(det), 1.0 / len(THETAS), atol=1e-12)


def test_soft_switch_inherits_theta_marginal_uniform_over_psi():
    random.seed(5); np.random.seed(5)
    world, sem, s0, l0, s1 = _stack(0.3, "high", 3.0)
    det, test = _switch_listener(s1, world, sem, c=1.5, switch_type="soft")
    for t in range(1, 121):
        obs = world.sample_obs(); utt = s1.sample_utterance(obs)
        det.update(utt); s1.update(obs); l0.update(utt); s0.update(obs)
        if det.switched:
            break
    assert det.switched
    naive_theta = np.array([det.naive.marginal_theta().get(th, 0.0) for th in THETAS])
    vig_theta = _theta_vec(det)
    np.testing.assert_allclose(vig_theta, naive_theta, atol=1e-12)
    psi = det.vigilant.marginal_psi()
    np.testing.assert_allclose([psi[p] for p in PSIS], 1.0 / len(PSIS), atol=1e-12)
    # and the inherited marginal is genuinely non-uniform (it carried information)
    assert np.abs(naive_theta - 1.0 / len(THETAS)).max() > 0.05


def test_after_switch_vigilant_updates_and_naive_freezes():
    random.seed(6); np.random.seed(6)
    world, sem, s0, l0, s1 = _stack(0.3, "high", 3.0)
    det, test = _switch_listener(s1, world, sem, c=1.5, switch_type="soft")
    switched_round = None
    for t in range(1, 101):
        obs = world.sample_obs(); utt = s1.sample_utterance(obs)
        det.update(utt); s1.update(obs); l0.update(utt); s0.update(obs)
        if det.switched and switched_round is None:
            switched_round = t
            naive_hist_len = len(det.naive.hist)
            theta_at_switch = _theta_vec(det).copy()
    assert switched_round is not None and switched_round < 100
    assert len(det.naive.hist) == naive_hist_len          # frozen
    assert len(det.vigilant.hist) == 1 + (100 - switched_round)
    assert np.abs(_theta_vec(det) - theta_at_switch).max() > 1e-6   # moved on
    assert det.state_belief is det.vigilant.state_belief


def test_utterance_stream_independent_of_listeners():
    """Same seed -> same utterances whatever is listening. This is what makes
    a paired comparison of policies on one stream legitimate."""
    def mk_one(s1, world, sem):
        return {"c": Listener1(THETAS, ["inf"], s1, world, sem, "inf", 3.0)}

    def mk_many(s1, world, sem):
        d, _ = _switch_listener(s1, world, sem, c=1.5, switch_type="hard")
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

def _policies(c=3.0):
    def mk(s1, world, sem):
        hard, _ = _switch_listener(s1, world, sem, c=c, switch_type="hard")
        soft, _ = _switch_listener(s1, world, sem, c=c, switch_type="soft")
        return {
            "credulous": Listener1(THETAS, ["inf"], s1, world, sem, "inf", 3.0),
            "vigilant": Listener1(THETAS, PSIS, s1, world, sem, "vig", 3.0),
            "switch_hard": hard,
            "switch_soft": soft,
        }
    return mk


def _policies_at(alpha, c=3.0):
    def mk(s1, world, sem):
        hard, _ = _switch_listener(s1, world, sem, c=c, switch_type="hard", alpha=alpha)
        soft, _ = _switch_listener(s1, world, sem, c=c, switch_type="soft", alpha=alpha)
        return {
            "credulous": Listener1(THETAS, ["inf"], s1, world, sem, "inf", alpha),
            "vigilant": Listener1(THETAS, PSIS, s1, world, sem, "vig", alpha),
            "switch_hard": hard,
            "switch_soft": soft,
        }
    return mk


# Persuasion only bites at large alpha: at alpha <= 3 the credulous listener
# converges to theta* regardless of the speaker's goal, so there is nothing for
# vigilance to protect against. The function tests therefore use alpha = 10,
# where a pers+ speaker drags credulous E[theta] from 0.5 to ~0.65.
STRONG = 10.0


@pytest.mark.parametrize("psi,direction", [("high", +1), ("low", -1)])
def test_vigilant_resists_persuasion_credulous_does_not(psi, direction):
    gaps, biases = [], []
    for seed in range(3):
        _, e, _ = run_stream(0.5, psi, STRONG, 100, 100 + seed, _policies_at(STRONG))
        biases.append(direction * (e["credulous"][-1] - 0.5))
        gaps.append(abs(e["credulous"][-1] - 0.5) - abs(e["vigilant"][-1] - 0.5))
    assert np.mean(biases) > 0.05, f"credulous not pushed in the persuasive direction: {biases}"
    assert np.mean(gaps) > 0.05, f"vigilant not clearly better: gaps={gaps}"


def test_vigilant_attributes_persuasion_to_psi():
    Ls, _, _ = run_stream(0.5, "high", STRONG, 100, 7, _policies_at(STRONG))
    psi = Ls["vigilant"].marginal_psi()
    assert psi["high"] > psi["inf"] and psi["high"] > psi["low"], psi


def test_switching_recovers_after_tau():
    """Paired on one stream: before tau the switch listener tracks credulous
    exactly; after tau its error vs theta* shrinks relative to credulous."""
    found = 0
    for seed in range(4):
        Ls, e, _ = run_stream(0.5, "high", STRONG, 150, 200 + seed,
                              _policies_at(STRONG, c=3.0))
        det = Ls["switch_soft"]
        if not det.switched or det.switched_at > 100:
            continue
        found += 1
        tau = det.switched_at
        # identical up to and including the switch round (soft inherits)
        np.testing.assert_allclose(e["switch_soft"][:tau], e["credulous"][:tau],
                                   atol=1e-10)
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
