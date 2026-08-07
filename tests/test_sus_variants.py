"""Sanity checks for the sus variant score functions.

Run with:  python -m pytest tests/test_sus_variants.py -v
"""

from __future__ import annotations

import math
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


THETAS = make_thetas(0.1, True, True)


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


@pytest.fixture
def stack():
    random.seed(0)
    np.random.seed(0)
    return _fresh_stack()


def test_variant_1_matches_existing_sus(stack):
    """Sanity 1: variant 1's score and naive variance reproduce compute_sus exactly."""
    world, semantics, _, l0, s1 = stack
    l1_theta = np.ones(len(THETAS)) / len(THETAS)
    l0_theta = np.asarray(l0.state_belief.prob, dtype=float)
    for u in semantics.utterance_space():
        ctx = _build_ctx(world, semantics, s1, l1_theta, l0_theta, u)
        score_old, var_old = compute_sus(ctx)
        score_new, var_n, _ = SUS_VARIANT_FNS["1"](ctx)
        assert math.isclose(score_new, score_old, rel_tol=1e-10, abs_tol=1e-12)
        assert math.isclose(var_n, var_old, rel_tol=1e-10, abs_tol=1e-12)


def test_variants_3_and_4_agree_when_L0_equals_L1(stack):
    """Sanity 2: forcing L_0(theta) = L_1(theta) makes variants 3 and 4 identical."""
    world, semantics, _, _, s1 = stack
    rng = np.random.default_rng(123)
    # Any valid theta distribution will do.
    l_theta = rng.dirichlet(np.ones(len(THETAS)))
    for u in semantics.utterance_space():
        ctx = _build_ctx(world, semantics, s1,
                         l1_theta=l_theta, l0_theta=l_theta.copy(), u_obs=u)
        s3, v3n, v3c = SUS_VARIANT_FNS["3"](ctx)
        s4, v4n, v4c = SUS_VARIANT_FNS["4"](ctx)
        assert math.isclose(s3, s4, rel_tol=1e-10, abs_tol=1e-12)
        assert math.isclose(v3n, v4n, rel_tol=1e-10, abs_tol=1e-12)
        assert math.isclose(v3c, v4c, rel_tol=1e-10, abs_tol=1e-12)


def test_variant_4b_differs_from_4_when_T_varies(stack):
    """Sanity 3: T(O) varies across O in the default n=1,m=7 setup, so 4b != 4."""
    world, semantics, _, _, s1 = stack
    # Confirm the premise: T(O) is not constant.
    truth = semantics.truth_table(world)
    T_O = truth.sum(axis=1)
    assert T_O.max() > T_O.min(), (
        "Test premise failed: T(O) is constant for this setup; 4b would coincide with 4."
    )

    rng = np.random.default_rng(7)
    l_theta = rng.dirichlet(np.ones(len(THETAS)))
    utts = semantics.utterance_space()
    diffs = []
    for u in utts:
        ctx = _build_ctx(world, semantics, s1,
                         l1_theta=l_theta, l0_theta=l_theta.copy(), u_obs=u)
        s4 = SUS_VARIANT_FNS["4"](ctx)[0]
        s4b = SUS_VARIANT_FNS["4b"](ctx)[0]
        diffs.append(abs(s4 - s4b))
    assert max(diffs) > 1e-6, "Variant 4b should differ from 4 when T(O) varies."


def test_corrected_variance_is_at_most_naive(stack):
    """Law of total variance: corrected (= naive - correction) must be <= naive.

    corrected is deliberately *not* clipped per-round -- the identity holds in
    expectation over u, so an individual round may be negative.  What must hold
    is that the p_pred-weighted mean of corrected equals Var_u[s(u)] exactly;
    that is asserted in ``test_ltv_identity_holds_in_expectation``.
    """
    world, semantics, _, l0, s1 = stack
    rng = np.random.default_rng(42)
    for _ in range(10):
        l1 = rng.dirichlet(np.ones(len(THETAS)))
        l0_th = rng.dirichlet(np.ones(len(THETAS)))
        for u in semantics.utterance_space():
            ctx = _build_ctx(world, semantics, s1, l1, l0_th, u)
            for v in SUS_VARIANTS:
                _, var_n, var_c = SUS_VARIANT_FNS[v](ctx)
                assert var_c <= var_n + 1e-10, (v, var_n, var_c)


def test_ltv_identity_holds_in_expectation(stack):
    """E_u[var_corrected(u)] == Var_{u ~ p_pred}[s(u)] exactly, for variant 1.

    This is the property the sequential test actually relies on: it averages
    the per-round corrected variances, so the identity must hold in the mean
    over u, not term by term.  Regression guard for the per-utterance clip
    that used to inflate the running variance at low alpha.
    """
    from rsa.detection.scores import _p_pred

    world, semantics, _, l0, s1 = stack
    utts = list(semantics.utterance_space())
    rng = np.random.default_rng(7)
    for _ in range(10):
        l1 = rng.dirichlet(np.ones(len(THETAS)))
        l0_th = rng.dirichlet(np.ones(len(THETAS)))

        per_u = [SUS_VARIANT_FNS["1"](_build_ctx(world, semantics, s1, l1, l0_th, u))
                 for u in utts]
        s_u = np.array([p[0] for p in per_u])
        vc_u = np.array([p[2] for p in per_u])

        ctx0 = _build_ctx(world, semantics, s1, l1, l0_th, utts[0])
        p = np.asarray(_p_pred("1", ctx0), dtype=float)

        exact = float(p @ s_u**2 - (p @ s_u)**2)
        assert float(p @ vc_u) == pytest.approx(exact, abs=1e-12)


def test_sequential_test_logs_both_variances(stack):
    """SequentialTest should track naive and corrected variances in parallel."""
    world, semantics, _, l0, s1 = stack
    tests = [SequentialTest(SUS_VARIANT_FNS[v], f"sus_{v}", c=2.0,
                            switch_enabled=False) for v in SUS_VARIANTS]
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
    for t in tests:
        h = t.history
        assert len(h["variances"]) == 15
        assert len(h["variances_corrected"]) == 15
        assert all(vc <= vn + 1e-10
                   for vn, vc in zip(h["variances"], h["variances_corrected"]))
        # running sigma stays real and non-negative even if a round is negative
        assert all(np.isfinite(sc) and sc >= 0.0
                   for sc in h["running_sigma_corrected"])



