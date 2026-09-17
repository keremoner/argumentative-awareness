"""Checks for the sus_1 score and its exact state-only variance.

The variance returned by ``make_sus_variant("1")`` is

    V = sum_u p(u) s(u)^2                                      (A) direct

which, by the law of total variance, equals

    V = sum_O q(O) v_O  -  K                                   (B) via LTV

with v_O the varentropy of row O and K the within-posterior correction.  The
tests below pin (A), check (B) agrees to machine precision, and confirm both
against Monte Carlo.

Run with:  python -m pytest tests/test_sus_variants.py -v
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from rsa.setup import make_thetas, make_world, make_semantics
from rsa.speaker0 import Speaker0
from rsa.listener0 import Listener0
from rsa.speaker1 import Speaker1
from rsa.detection import (
    DetectionListener,
    ScoreContext,
    SequentialTest,
    compute_sus,
    SUS_VARIANTS,
    SUS_VARIANT_FNS,
    make_sus_variant,
)
from rsa.detection.scores import (
    RETIRED_SUS_VARIANTS,
    _p_pred,
    _weighting_matrix,
    _within_posterior_correction,
)


THETAS = make_thetas(0.1, True, True)
ALPHAS = [1.0, 3.0, 10.0]


def _concentrated(theta_value, mass=0.9):
    """A frozen listener state putting most mass on one theta."""
    p = np.full(len(THETAS), (1.0 - mass) / (len(THETAS) - 1))
    p[THETAS.index(theta_value)] = mass
    return p / p.sum()


FROZEN_STATES = {
    "uniform": np.ones(len(THETAS)) / len(THETAS),
    "theta0.1": _concentrated(0.1),
    "theta0.5": _concentrated(0.5),
    "theta0.9": _concentrated(0.9),
}
ALL_STATES = [(nm, a) for nm in FROZEN_STATES for a in ALPHAS]


def _fresh_stack(theta_true=0.3, alpha=3.0, speaker_psi="inf", n=1, m=7):
    world = make_world(theta_true, n=n, m=m)
    semantics = make_semantics(n=n)
    s0 = Speaker0(THETAS, semantics=semantics, world=world)
    l0 = Listener0(THETAS, s0, semantics=semantics, world=world)
    s1 = Speaker1(THETAS, l0, semantics=semantics, world=world,
                  alpha=alpha, psi=speaker_psi)
    return world, semantics, s0, l0, s1


def _build_ctx(world, semantics, s1, l1_theta, l0_theta, u_obs):
    return ScoreContext(THETAS, s1, world, semantics,
                        l1_theta=l1_theta, u_obs=u_obs,
                        l0_theta=l0_theta)


def _state_quantities(l1_theta, alpha):
    """Bundle the pieces of one frozen listener state."""
    world, semantics, _, l0, s1 = _fresh_stack(alpha=alpha)
    l0_theta = np.asarray(l0.state_belief.prob, dtype=float)
    u0 = semantics.utterance_space()[0]
    ctx = _build_ctx(world, semantics, s1, l1_theta, l0_theta, u0)
    W = _weighting_matrix("1", ctx)
    B = ctx.B_matrix
    p_pred = np.asarray(_p_pred("1", ctx), dtype=float)
    s_all = np.sum(W * B, axis=0)
    var = SUS_VARIANT_FNS["1"](ctx)[1]
    return ctx, W, B, s_all, p_pred, var


@pytest.fixture
def stack():
    random.seed(0)
    np.random.seed(0)
    return _fresh_stack()


@pytest.mark.parametrize("state_name,alpha", ALL_STATES)
def test_score_is_mean_zero(state_name, alpha):
    """E over u of s(u) is 0 -- the property the exact variance relies on."""
    _, _, _, s_all, p_pred, _ = _state_quantities(FROZEN_STATES[state_name], alpha)
    assert float(p_pred @ s_all) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("state_name,alpha", ALL_STATES)
def test_direct_and_ltv_forms_agree(state_name, alpha):
    """Form (A) equals form (B) to machine precision."""
    ctx, W, B, s_all, p_pred, var = _state_quantities(FROZEN_STATES[state_name], alpha)
    form_a = float(p_pred @ s_all ** 2)
    K = _within_posterior_correction(W, B, p_pred)
    form_b = float(np.dot(ctx.L1_O, ctx.varentropy_per_O) - K)
    assert form_a == pytest.approx(form_b, abs=1e-10)
    assert var == pytest.approx(form_a, abs=1e-12)


@pytest.mark.parametrize("state_name,alpha", ALL_STATES)
def test_variance_is_non_negative(state_name, alpha):
    """var >= 0 by construction: a p-weighted sum of squares."""
    *_, var = _state_quantities(FROZEN_STATES[state_name], alpha)
    assert var >= 0.0


@pytest.mark.parametrize("state_name,alpha", [("uniform", 3.0), ("theta0.5", 1.0)])
def test_variance_matches_monte_carlo(state_name, alpha):
    """Sampling u from p_pred reproduces the analytic variance within ~5%."""
    _, _, _, s_all, p_pred, var = _state_quantities(FROZEN_STATES[state_name], alpha)
    rng = np.random.default_rng(12345)
    draws = rng.choice(len(s_all), size=20000, p=p_pred / p_pred.sum())
    emp = float(np.var(s_all[draws]))
    assert emp == pytest.approx(var, rel=0.05), f"empirical {emp} vs analytic {var}"


@pytest.mark.parametrize("state_name,alpha", ALL_STATES)
def test_old_proxy_matched_new_variance_only_in_expectation(state_name, alpha):
    """Documents why the old per-round proxy passed its calibration check.

    The retired implementation returned ``var_naive(u_obs) - K`` where
    ``var_naive(u) = sum_O W[O,u] v_O``.  Averaged over u it equals the new
    exact variance -- which is why pooled calibration looked fine -- but per
    round it is a different number, and one that moves with the score it is
    meant to scale.
    """
    ctx, W, B, s_all, p_pred, var = _state_quantities(FROZEN_STATES[state_name], alpha)
    K = _within_posterior_correction(W, B, p_pred)
    old_per_u = W.T @ ctx.varentropy_per_O - K
    assert float(p_pred @ old_per_u) == pytest.approx(var, abs=1e-10)
    # ...and it genuinely varied round to round, which is the bug being fixed.
    assert old_per_u.std() > 1e-6


def test_compute_sus_is_variant_one(stack):
    """compute_sus is variant 1 -- one object, so they cannot drift apart."""
    world, semantics, _, l0, s1 = stack
    assert compute_sus is SUS_VARIANT_FNS["1"]
    l1_theta = np.ones(len(THETAS)) / len(THETAS)
    l0_theta = np.asarray(l0.state_belief.prob, dtype=float)
    for u in semantics.utterance_space():
        ctx = _build_ctx(world, semantics, s1, l1_theta, l0_theta, u)
        assert len(SUS_VARIANT_FNS["1"](ctx)) == 2, "score_fn must return a 2-tuple"


def test_variance_does_not_depend_on_utterance_heard(stack):
    """The point of the fix: var is a function of listener state alone."""
    world, semantics, _, l0, s1 = stack
    l1_theta = np.ones(len(THETAS)) / len(THETAS)
    l0_theta = np.asarray(l0.state_belief.prob, dtype=float)
    variances = [
        SUS_VARIANT_FNS["1"](_build_ctx(world, semantics, s1, l1_theta, l0_theta, u))[1]
        for u in semantics.utterance_space()
    ]
    assert max(variances) - min(variances) < 1e-12


@pytest.mark.parametrize("variant", RETIRED_SUS_VARIANTS)
def test_retired_variants_raise(variant):
    assert variant not in SUS_VARIANTS
    assert variant not in SUS_VARIANT_FNS
    with pytest.raises(NotImplementedError):
        make_sus_variant(variant)


def test_sequential_test_single_variance(stack):
    """SequentialTest tracks one variance and never needs to clip."""
    world, semantics, _, l0, s1 = stack
    tests = [SequentialTest(SUS_VARIANT_FNS["1"], "sus_1", c=2.0,
                            switch_enabled=False)]
    listener = DetectionListener(THETAS, ["inf", "high", "low"], s1, world,
                                 semantics, tests=tests, alpha=3.0)
    random.seed(0)
    np.random.seed(0)
    for _ in range(15):
        obs = world.sample_obs()
        utt = s1.sample_utterance(obs)
        listener.update(utt)
        s1.update(obs)
        l0.update(utt)
    h = tests[0].history
    assert len(h["variances"]) == 15
    assert all(v >= 0.0 for v in h["variances"])
    assert all(np.isfinite(sc) and sc >= 0.0 for sc in h["running_sigma"])
    assert "variances_corrected" not in h
    assert not hasattr(tests[0], "tau_corrected")


def test_sequential_test_rejects_three_tuple():
    """A 3-tuple score_fn must fail loudly, pointing at the change."""
    t = SequentialTest(lambda ctx: (0.0, 1.0, 1.0), "bad", c=2.0)
    with pytest.raises(ValueError, match="expected 2"):
        t.observe(None)


def test_sequential_test_rejects_negative_variance():
    t = SequentialTest(lambda ctx: (0.0, -1.0), "bad", c=2.0)
    with pytest.raises(ValueError, match="negative"):
        t.observe(None)
