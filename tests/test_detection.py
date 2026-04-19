"""
Sanity checks for ``rsa.detection``.

Run with: ``python -m pytest tests/test_detection.py -v``
"""

from __future__ import annotations

import math
import random
from copy import deepcopy

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
    compute_surp2,
    compute_surp1,
    compute_sus,
)
from rsa.detection.scores import _safe_log


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

THETAS = make_thetas(0.1, True, True)     # [0.0, 0.1, ..., 1.0]
PSIS = ["inf", "high", "low"]


def _fresh_stack(theta_true=0.3, alpha=3.0, speaker_psi="inf", n=1, m=7):
    """Build a fresh S0 / L0 / S1 stack."""
    world = make_world(theta_true, n=n, m=m)
    semantics = make_semantics(n=n)
    s0 = Speaker0(THETAS, semantics=semantics, world=world)
    l0 = Listener0(THETAS, s0, semantics=semantics, world=world)
    s1 = Speaker1(THETAS, l0, semantics=semantics, world=world,
                  alpha=alpha, psi=speaker_psi)
    return world, semantics, s0, l0, s1


@pytest.fixture
def stack():
    random.seed(0)
    np.random.seed(0)
    return _fresh_stack()


# ---------------------------------------------------------------------------
# 1. Shape / smoke
# ---------------------------------------------------------------------------

def test_scores_finite_and_nonneg_variance(stack):
    world, semantics, s0, l0, s1 = stack
    listener = DetectionListener(THETAS, PSIS, s1, world, semantics)
    for u in semantics.utterance_space():
        ctx = listener.build_context(u)
        for fn in (compute_surp2, compute_surp1, compute_sus):
            score, var = fn(ctx)
            assert math.isfinite(score), f"{fn.__name__} score not finite for {u}"
            assert math.isfinite(var) and var >= 0.0, \
                f"{fn.__name__} variance must be finite and >= 0 for {u}"


def test_predictives_are_valid_distributions(stack):
    world, semantics, s0, l0, s1 = stack
    listener = DetectionListener(THETAS, PSIS, s1, world, semantics)
    u = semantics.utterance_space()[0]
    ctx = listener.build_context(u)
    for name, p in [("P_prior", ctx.P_prior),
                    ("P_post", ctx.P_post),
                    ("L1_O", ctx.L1_O),
                    ("L1_O_given_u", ctx.L1_O_given_u)]:
        assert np.all(p >= -1e-12), f"{name} has negative entries"
        assert abs(p.sum() - 1.0) < 1e-9, f"{name} not normalized (sum={p.sum()})"


# ---------------------------------------------------------------------------
# 2. Algebraic identity: sus(u) == surp2(u) - KL(L1(.|u)||L1(.)) + DeltaH(u)
# ---------------------------------------------------------------------------

def _kl(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0
    return float(np.sum(p[mask] * (np.log(p[mask]) - _safe_log(q)[mask])))


def test_sus_surp2_identity_single_utterance(stack):
    """sus(u) = surp2(u) - KL(L1(O|u) || L1(O)) + (H(P_prior) - E_{O|u}[H(S1(.|O))])."""
    world, semantics, s0, l0, s1 = stack
    listener = DetectionListener(THETAS, PSIS, s1, world, semantics)
    for u in semantics.utterance_space():
        ctx = listener.build_context(u)
        surp2, _ = compute_surp2(ctx)
        sus, _ = compute_sus(ctx)

        kl = _kl(ctx.L1_O_given_u, ctx.L1_O)
        H_pred = -float(np.dot(ctx.P_prior, ctx.log_P_prior))
        E_H_cond = float(np.dot(ctx.L1_O_given_u, ctx.H_per_O))
        delta_H = H_pred - E_H_cond

        predicted_sus = surp2 - kl + delta_H
        assert abs(sus - predicted_sus) < 1e-9, (
            f"identity broke for {u}: "
            f"sus={sus}, surp2={surp2}, kl={kl}, delta_H={delta_H}, "
            f"predicted={predicted_sus}"
        )


def test_identity_after_several_updates(stack):
    """Same identity must hold even after the listener has absorbed rounds."""
    world, semantics, s0, l0, s1 = stack
    listener = DetectionListener(THETAS, PSIS, s1, world, semantics)
    random.seed(42)
    for _ in range(5):
        obs = world.sample_obs()
        utt = s1.sample_utterance(obs)
        # run a round *without* scoring to drive state forward
        listener.naive.update(utt)
        s1.update(obs)
        l0.update(utt)
        s0.update(obs)

    for u in semantics.utterance_space():
        ctx = listener.build_context(u)
        surp2, _ = compute_surp2(ctx)
        sus, _ = compute_sus(ctx)
        kl = _kl(ctx.L1_O_given_u, ctx.L1_O)
        H_pred = -float(np.dot(ctx.P_prior, ctx.log_P_prior))
        E_H_cond = float(np.dot(ctx.L1_O_given_u, ctx.H_per_O))
        predicted = surp2 - kl + (H_pred - E_H_cond)
        assert abs(sus - predicted) < 1e-9


# ---------------------------------------------------------------------------
# 3. Exact null-mean properties (analytic, no MC)
# ---------------------------------------------------------------------------

def test_surp2_analytic_null_mean_is_zero(stack):
    """E_{u ~ P_prior}[surp2(u)] = 0 by definition of entropy."""
    world, semantics, s0, l0, s1 = stack
    listener = DetectionListener(THETAS, PSIS, s1, world, semantics)
    # Use any u just to build ctx; P_prior is independent of u_obs
    ctx = listener.build_context(semantics.utterance_space()[0])
    expected = 0.0
    for u in semantics.utterance_space():
        ctx_u = listener.build_context(u)
        score, _ = compute_surp2(ctx_u)
        expected += ctx.P_prior[ctx_u.u_obs_idx] * score
    assert abs(expected) < 1e-9, f"E[surp2] = {expected}, expected 0"


def test_sus_analytic_null_mean_is_zero(stack):
    """E_{O~L1(O), u~S1(.|O,inf)}[sus(u)] = 0 exactly.

    This is the key motivator for the sus score: the per-O centering by
    H(S1(.|O,inf)) makes the expectation vanish term-by-term in O.
    """
    world, semantics, s0, l0, s1 = stack
    listener = DetectionListener(THETAS, PSIS, s1, world, semantics)
    ctx0 = listener.build_context(semantics.utterance_space()[0])
    S1_table = ctx0.S1_table
    L1_O = ctx0.L1_O

    expected = 0.0
    for u in semantics.utterance_space():
        u_idx = semantics.utterance_index(u)
        # marginal P(u) under null = sum_O L1(O) * S1(u|O,inf) = P_prior[u_idx]
        p_u = float(ctx0.P_prior[u_idx])
        if p_u <= 0:
            continue
        ctx_u = listener.build_context(u)
        score, _ = compute_sus(ctx_u)
        expected += p_u * score
    assert abs(expected) < 1e-9, f"E[sus] = {expected}, expected 0"


# ---------------------------------------------------------------------------
# 4. SequentialTest behaviour
# ---------------------------------------------------------------------------

def test_sequential_test_passive_under_inf(stack):
    """Under the informative speaker, Sus(t) should stay roughly centred."""
    world, semantics, s0, l0, s1 = stack
    random.seed(123)

    tests = [
        SequentialTest(compute_surp2, "surp2", c=float("inf")),
        SequentialTest(compute_surp1, "surp1", c=float("inf")),
        SequentialTest(compute_sus, "sus", c=float("inf")),
    ]
    listener = DetectionListener(THETAS, PSIS, s1, world, semantics, tests=tests)

    rounds = 60
    for _ in range(rounds):
        obs = world.sample_obs()
        utt = s1.sample_utterance(obs)
        listener.update(utt)
        s1.update(obs)
        l0.update(utt)
        s0.update(obs)

    # No test can have crossed because c=inf
    for t in tests:
        assert t.tau is None, f"{t.name} crossed with c=inf"
        assert len(t.history["scores"]) == rounds


def test_sqrt_t_decay_under_null():
    """Running mean |Sus(t)| should shrink roughly as 1/sqrt(t) under null.

    Check the ratio of late-stage to early-stage |Sus|: if the scaling is
    correct, the late ratio should be considerably smaller than 1.
    """
    random.seed(7)
    np.random.seed(7)

    def run_once(rounds):
        world, semantics, s0, l0, s1 = _fresh_stack(theta_true=0.3)
        tests = [
            SequentialTest(compute_surp2, "surp2", c=float("inf")),
            SequentialTest(compute_surp1, "surp1", c=float("inf")),
            SequentialTest(compute_sus, "sus", c=float("inf")),
        ]
        listener = DetectionListener(THETAS, PSIS, s1, world, semantics, tests=tests)
        for _ in range(rounds):
            obs = world.sample_obs()
            utt = s1.sample_utterance(obs)
            listener.update(utt)
            s1.update(obs)
            l0.update(utt)
            s0.update(obs)
        return {t.name: t.history["running_mean"] for t in tests}

    n_sims = 25
    rounds = 80
    by_name = {"surp2": [], "surp1": [], "sus": []}
    for _ in range(n_sims):
        out = run_once(rounds)
        for k, v in out.items():
            by_name[k].append(v)

    arr = {k: np.array(v) for k, v in by_name.items()}   # (n_sims, rounds)

    for name, mat in arr.items():
        early = np.mean(np.abs(mat[:, 9]))       # |Sus(10)|
        late = np.mean(np.abs(mat[:, 79]))       # |Sus(80)|
        # Under null we expect late < early by a clear margin.
        assert late < 0.8 * early + 1e-6, (
            f"{name}: late={late:.4f} not smaller than 0.8*early={0.8*early:.4f}"
        )


def test_switch_triggers_and_freezes(stack):
    """A test with a very low c should trigger a switch quickly under a
    persuasive speaker, and the listener should stop running the tests."""
    world, semantics, s0, l0, s1 = _fresh_stack(theta_true=0.3, speaker_psi="high")
    tests = [SequentialTest(compute_sus, "sus", c=0.1, switch_enabled=True,
                             switch_type="soft")]
    listener = DetectionListener(THETAS, PSIS, s1, world, semantics, tests=tests)

    for _ in range(40):
        obs = world.sample_obs()
        utt = s1.sample_utterance(obs)
        listener.update(utt)
        s1.update(obs)
        l0.update(utt)
        s0.update(obs)

    assert listener.switched, "expected at least one switch"
    # After switch, the test stops receiving updates
    rounds_observed = len(tests[0].history["scores"])
    assert rounds_observed == listener.switched_at
    # Vigilant has an expanded state
    assert set(listener.marginal_psi().keys()) == set(PSIS)
