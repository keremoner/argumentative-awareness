"""S2's model of a switching listener: informativeness and persuasiveness are
read from the *same* hypothetical listener.

* ``DetectionListener.peek_obs(u)`` equals ``infer_obs(u)`` of the listener
  that would be active after u -- credulous when u does not trigger, the
  would-be vigilant one (shadow / seeded scratch) when it does -- and never
  mutates.
* At a round where some utterance would trigger the switch, an informative S2
  over the switching listener satisfies the product-form identity with
  ``peek_obs`` and *violates* it with the credulous ``infer_obs`` (the old
  behaviour), so the switch is inside S2-inf's utility.

Run with:  python -m pytest tests/test_s2_model.py -v
"""

from __future__ import annotations

import random
from copy import deepcopy

import numpy as np
import pytest

from rsa.listener1 import Listener1
from rsa.speaker2 import Speaker2
from rsa.detection import SWITCH_TYPES
from tests.test_switching import (
    THETAS, PSIS, _stack, _det, _advance, _find_seed_with_mid_tau,
)


def _drive_to(det_factory, theta, psi, alpha, seed, rounds):
    """Fresh stream driven ``rounds`` rounds; returns everything, including an
    always-vigilant twin of the detector's shadow."""
    random.seed(seed); np.random.seed(seed)
    world, sem, s0, l0, s1 = _stack(theta, psi, alpha)
    det, test = det_factory(s1, world, sem)
    vig = Listener1(THETAS, PSIS, s1, world, sem, "vig", alpha)
    for _ in range(rounds):
        obs = world.sample_obs(); utt = s1.sample_utterance(obs)
        det.update(utt); vig.update(utt)
        _advance(world, s0, l0, s1, obs, utt)
    return world, sem, s0, l0, s1, det, vig


def _max_diff(a, b):
    return max(abs(a[k] - b[k]) for k in a)


@pytest.mark.parametrize("switch_type", SWITCH_TYPES)
def test_peek_obs_matches_would_be_active_listener(switch_type):
    c, alpha = 3.0, 5.0
    seed, tau = _find_seed_with_mid_tau(0.3, "high", alpha, c, switch_type, 60, lo=4)
    fac = lambda s1, w, s: _det(s1, w, s, c, switch_type, alpha)
    world, sem, s0, l0, s1, det, vig = _drive_to(fac, 0.3, "high", alpha, seed, tau - 1)
    assert not det.switched
    triggering = [u for u in sem.utterance_space() if det._would_switch(u) is not None]
    assert triggering, "seed should give a trigger round"

    snapshot = deepcopy(det.naive.state_belief.prob), det.round, len(det.utt_history)
    for u in sem.utterance_space():
        got = det.peek_obs(u)
        if u in triggering:
            if switch_type == "hard":
                ref = vig.infer_obs(u)                         # always-vigilant == shadow
            else:
                scratch = Listener1(THETAS, PSIS, s1, world, sem, "vig", alpha)
                scratch.seed_from_theta_marginal(det.naive.marginal_theta())
                ref = scratch.infer_obs(u)
            assert _max_diff(got, det.naive.infer_obs(u)) > 1e-6    # differs from credulous
        else:
            ref = det.naive.infer_obs(u)
        assert _max_diff(got, ref) < 1e-12, u
    # nothing moved
    np.testing.assert_array_equal(det.naive.state_belief.prob, snapshot[0])
    assert (det.round, len(det.utt_history)) == snapshot[1:] and not det.switched


def test_peek_obs_after_switch_is_vigilant():
    seed, tau = _find_seed_with_mid_tau(0.3, "high", 5.0, 2.5, "hard", 40)
    fac = lambda s1, w, s: _det(s1, w, s, 2.5, "hard", 5.0)
    world, sem, *_, det, vig = _drive_to(fac, 0.3, "high", 5.0, seed, tau + 3)
    assert det.switched
    for u in sem.utterance_space():
        assert _max_diff(det.peek_obs(u), det.vigilant.infer_obs(u)) < 1e-12


def _residual(s2, obs_post, sem, world, obs):
    """Spread of log P(u|O) - alpha*log Inf(u;O) over true u (0 iff product form)."""
    dist = s2.dist_over_utterances_obs(obs, "inf")
    vals = [np.log(dist[u]) - s2.alpha * np.log(obs_post(u)[obs])
            for u in sem.utterance_space() if sem.truth_value(world, obs, u)]
    return float(np.ptp(vals))


@pytest.mark.parametrize("switch_type", SWITCH_TYPES)
def test_informative_s2_scores_against_the_switched_listener(switch_type):
    c, alpha = 3.0, 5.0
    seed, tau = _find_seed_with_mid_tau(0.3, "high", alpha, c, switch_type, 60, lo=4)
    fac = lambda s1, w, s: _det(s1, w, s, c, switch_type, alpha)
    world, sem, s0, l0, s1, det, vig = _drive_to(fac, 0.3, "high", alpha, seed, tau - 1)
    triggering = [u for u in sem.utterance_space() if det._would_switch(u) is not None]
    assert triggering
    s2 = Speaker2(THETAS, det, sem, world, alpha=alpha, psi="inf")
    worst_new = worst_old = 0.0
    for obs in world.generate_all_obs():
        if not any(sem.truth_value(world, obs, u) for u in triggering):
            continue                                # the switch cannot matter here
        worst_new = max(worst_new, _residual(s2, det.peek_obs, sem, world, obs))
        worst_old = max(worst_old, _residual(s2, det.infer_obs, sem, world, obs))
    assert worst_new < 1e-9
    assert worst_old > 1e-6, "credulous Inf would have given the same policy"
    # and the persuasive S2 sees the same listener (peek) -- identity in test_speaker2
    for psi in ("high", "low"):
        pers = s2.get_persuasiveness(psi)
        for u in triggering:
            e = sum(th * p for th, p in det.peek(u).items())
            assert abs(pers[u] - (e if psi == "high" else 1 - e)) < 1e-12
