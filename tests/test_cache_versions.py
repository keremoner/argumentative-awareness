"""Cache/version protocol (see ``rsa.core``).

Every agent caches tables that depend on the agent below it.  These tests move
the lower agent *without* calling the upper agent's ``update`` and check that
the upper agent still answers from fresh state -- the situation that used to
silently serve stale tables.

Run with:  python -m pytest tests/test_cache_versions.py -v
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
from rsa.speaker2 import Speaker2
from rsa.detection import make_switching_listener, TableSpeaker


THETAS = make_thetas(0.1)
PSIS = ["inf", "high", "low"]
U = [("some", "effective"), ("most", "ineffective"), ("some", "ineffective")]


def _chain(theta=0.3, alpha=3.0, psi="inf"):
    world = make_world(theta, n=1, m=7)
    sem = make_semantics(n=1)
    s0 = Speaker0(THETAS, semantics=sem, world=world)
    l0 = Listener0(THETAS, s0, semantics=sem, world=world)
    s1 = Speaker1(THETAS, l0, semantics=sem, world=world, alpha=alpha, psi=psi)
    return world, sem, s0, l0, s1


def _tables(sp):
    return {p: np.array(sp.obs_utt_table_for_psi(p), copy=True) for p in PSIS}


def _assert_tables_equal(a, b):
    for p in PSIS:
        np.testing.assert_allclose(a[p], b[p], atol=1e-12)


def _assert_tables_differ(a, b):
    assert max(np.abs(a[p] - b[p]).max() for p in PSIS) > 1e-6


# ---------------------------------------------------------------------------
# version fingerprints move when (and only when) state moves
# ---------------------------------------------------------------------------

def test_versions_are_nested_and_move_on_update():
    world, sem, s0, l0, s1 = _chain()
    l1 = Listener1(THETAS, PSIS, s1, world, sem, "vig", 3.0)
    s2 = Speaker2(THETAS, l1, sem, world, alpha=3.0, psi="high")
    obs = world.generate_all_obs()[3]

    v = [s0.version, l0.version, s1.version, l1.version, s2.version]
    # nesting: each agent's tuple ends with the one below it
    assert l0.version[1] == s0.version and s1.version[1] == l0.version
    assert l1.version[1] == s1.version and s2.version[1] == l1.version

    l0.update(U[0])                       # moves l0, hence s1, l1, s2 -- not s0
    assert s0.version == v[0]
    assert l0.version != v[1] and s1.version != v[2]
    assert l1.version != v[3] and s2.version != v[4]

    w = s2.version
    s2.get_persuasiveness("high"); s2.obs_utt_table_for_psi("inf")   # reads never bump
    assert s2.version == w
    s2.update(obs)
    assert s2.version != w


# ---------------------------------------------------------------------------
# upper agents drop caches when the lower agent moves without them
# ---------------------------------------------------------------------------

def test_s1_tracks_l0_without_its_own_update():
    world, sem, s0, l0, s1 = _chain()
    before = _tables(s1)
    for u in U:
        l0.update(u)                      # nobody calls s1.update
    after = _tables(s1)
    fresh = _tables(Speaker1(THETAS, l0, semantics=sem, world=world, alpha=3.0))
    _assert_tables_differ(before, after)
    _assert_tables_equal(after, fresh)


def test_l1_tracks_s1_without_its_own_update():
    world, sem, s0, l0, s1 = _chain()
    l1 = Listener1(THETAS, PSIS, s1, world, sem, "vig", 3.0)
    u = U[0]
    before = l1.infer_state(u).prob.copy()
    before_obs = dict(l1.infer_obs(u))
    for x in U:                           # move the speaker's tables underneath l1
        l0.update(x); s1.update(world.generate_all_obs()[2])
    after = l1.infer_state(u).prob.copy()
    after_obs = l1.infer_obs(u)
    twin = Listener1(THETAS, PSIS, s1, world, sem, "vig", 3.0)   # same prior, fresh caches
    np.testing.assert_allclose(after, twin.infer_state(u).prob, atol=1e-12)
    for o in after_obs:
        assert abs(after_obs[o] - twin.infer_obs(u)[o]) < 1e-12
    assert np.abs(after - before).max() > 1e-6
    assert max(abs(after_obs[o] - before_obs[o]) for o in after_obs) > 1e-6


def test_s2_tracks_listener1_without_its_own_update():
    world, sem, s0, l0, s1 = _chain()
    l1 = Listener1(THETAS, PSIS, s1, world, sem, "vig", 3.0)
    s2 = Speaker2(THETAS, l1, sem, world, alpha=3.0, psi="high")
    before = _tables(s2)
    for u in U:
        l1.update(u)                      # nobody calls s2.update
    after = _tables(s2)
    fresh = _tables(Speaker2(THETAS, l1, sem, world, alpha=3.0, psi="high"))
    _assert_tables_differ(before, after)
    _assert_tables_equal(after, fresh)


def test_s2_tracks_detection_listener_without_its_own_update():
    random.seed(3); np.random.seed(3)
    world, sem, s0, l0, s1 = _chain(alpha=5.0, psi="high")
    det = make_switching_listener(THETAS, PSIS, s1, world, sem, c=2.0, switch_type="hard", alpha=5.0)
    s2 = Speaker2(THETAS, det, sem, world, alpha=5.0, psi="high")
    before = _tables(s2)
    for _ in range(25):                   # drive the listener across its switch
        obs = world.sample_obs(); u = s1.sample_utterance(obs)
        det.update(u); s1.update(obs); l0.update(u); s0.update(obs)
    assert det.switched
    after = _tables(s2)
    fresh = _tables(Speaker2(THETAS, det, sem, world, alpha=5.0, psi="high"))
    _assert_tables_differ(before, after)
    _assert_tables_equal(after, fresh)


def test_listener1_tracks_table_speaker_round_index():
    world, sem, s0, l0, s1 = _chain()
    tabs = []
    for u in U:
        tabs.append(_tables(s1)); l0.update(u)
    sp = TableSpeaker(tabs, THETAS, world, sem)
    l1 = Listener1(THETAS, PSIS, sp, world, sem, "vig", 3.0)
    u = U[1]
    sp.round_idx = 0; p0 = l1.infer_state(u).prob.copy()
    sp.round_idx = 2; p2 = l1.infer_state(u).prob.copy()   # no l1.update in between
    ref = Listener1(THETAS, PSIS, sp, world, sem, "vig", 3.0)
    np.testing.assert_allclose(p2, ref.infer_state(u).prob, atol=1e-12)
    assert np.abs(p2 - p0).max() > 1e-6


# ---------------------------------------------------------------------------
# a shared object in two roles cannot leak state across a round boundary
# ---------------------------------------------------------------------------

def test_shared_listener_equals_replica_over_a_full_game():
    """S2 over the public listener itself vs S2 over an identical replica:
    the utterance stream and every belief must coincide, i.e. S2's reads of
    the listener (peeks, cached posteriors) leave no trace that changes what
    the listener later does."""
    def run(use_replica, seed=11, rounds=40):
        random.seed(seed); np.random.seed(seed)
        world, sem, s0, l0, s1 = _chain(alpha=5.0)
        mk = lambda: make_switching_listener(THETAS, PSIS, s1, world, sem, c=2.0,
                                             switch_type="soft", alpha=5.0)
        pub = mk()
        s2 = Speaker2(THETAS, mk() if use_replica else pub, sem, world, alpha=5.0, psi="high")
        utts, E = [], []
        for _ in range(rounds):
            obs = world.sample_obs(); u = s2.sample_utterance(obs); utts.append(u)
            pub.update(u)
            if use_replica:
                s2.listener.update(u)
            E.append(pub.theta_array().copy())
            s2.update(obs); s1.update(obs); l0.update(u); s0.update(obs)
        return utts, np.array(E), pub.switched_at
    a, b = run(False), run(True)
    assert a[0] == b[0] and a[2] == b[2]
    np.testing.assert_allclose(a[1], b[1], atol=1e-12)


# ---------------------------------------------------------------------------
# unknown psi is an error, not a silent default
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [Speaker1, Speaker2])
def test_unknown_psi_raises(cls):
    world, sem, s0, l0, s1 = _chain()
    inner = l0 if cls is Speaker1 else Listener1(THETAS, PSIS, s1, world, sem, "vig", 3.0)
    with pytest.raises(ValueError):
        cls(THETAS, inner, sem, world, alpha=3.0, psi="typo")
    sp = cls(THETAS, inner, sem, world, alpha=3.0, psi="high")
    with pytest.raises(ValueError):
        sp.get_persuasiveness("typo")
