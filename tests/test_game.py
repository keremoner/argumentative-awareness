"""``rsa.game``: S2 vs {credulous, vigilant, switching} L1 with a replica model.

Run with:  python -m pytest tests/test_game.py -v
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from rsa.setup import make_thetas, make_world, make_semantics
from rsa.game import game, game_s1, make_listener, LISTENER_TYPES
from rsa.listener1 import Listener1
from rsa.detection import DetectionListener


THETAS = make_thetas(0.1)
PSIS = ["inf", "high", "low"]


def _env(theta=0.3):
    return make_world(theta, n=1, m=7), make_semantics(n=1)


@pytest.mark.parametrize("listener_type", LISTENER_TYPES)
@pytest.mark.parametrize("speaker_type", ["inf", "high", "low"])
def test_game_runs_with_a_private_replica(listener_type, speaker_type):
    random.seed(1); np.random.seed(1)
    world, sem = _env()
    s2, l1, s1, l0, s0 = game(THETAS, PSIS, sem, world, speaker_type=speaker_type,
                              listener_type=listener_type, alpha=3.0, rounds=12,
                              c=2.5, switch_type="soft", verbose=False)
    assert s2.psi == speaker_type
    assert s2.listener is not l1                                  # a replica, not the public object
    assert type(s2.listener) is type(l1)
    np.testing.assert_allclose(s2.listener.theta_array(), l1.theta_array(), atol=1e-12)
    assert len(l1.utt_history) == 12 and len(s2.hist) == 13
    if listener_type == "switch":
        assert isinstance(l1, DetectionListener)
        assert l1.switched == s2.listener.switched
        assert l1.tests[0].c == 2.5 and l1.tests[0].switch_type == "soft"
    else:
        assert isinstance(l1, Listener1) and l1.listener_type == listener_type


def test_persuasive_s2_trips_the_switching_listener():
    random.seed(9); np.random.seed(9)
    world, sem = _env()
    s2, l1, *_ = game(THETAS, PSIS, sem, world, speaker_type="high", listener_type="switch",
                      alpha=5.0, rounds=60, c=2.0, switch_type="hard", verbose=False)
    assert l1.switched and l1.switched_at == s2.listener.switched_at
    assert l1.marginal_psi()["inf"] < 0.5


def test_replica_divergence_is_an_error():
    random.seed(2); np.random.seed(2)
    world, sem = _env()
    s2, l1, *_ = game(THETAS, PSIS, sem, world, listener_type="vig", rounds=3, verbose=False)
    # tamper with the public listener; a further game step must notice
    from rsa.game import _check_replica
    l1.state_belief.prob[:] = 1.0 / l1.state_belief.prob.size
    with pytest.raises(RuntimeError):
        _check_replica(l1, s2.listener, 4)


def test_game_s1_accepts_switching_listener():
    random.seed(4); np.random.seed(4)
    world, sem = _env()
    s1, l1, l0, s0 = game_s1(THETAS, PSIS, sem, world, speaker_type="low", listener_type="switch",
                             alpha=5.0, rounds=30, c=2.0, switch_type="hard", verbose=False)
    assert isinstance(l1, DetectionListener) and l1.round == 30


def test_make_listener_rejects_unknown_type():
    world, sem = _env()
    with pytest.raises(ValueError):
        make_listener("bogus", THETAS, PSIS, None, world, sem)
