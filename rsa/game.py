"""
Multi-round dyadic interaction games.

Each round: the world generates an observation -> the speaker produces an
utterance -> the listener(s) update -> every agent's belief moves on.

``game_s1``  S1 speaker vs L1.  Fang's dyad (Figures 3-6 of the paper).
``game``     S2 speaker vs L1.  S2 reasons about a *replica* of the public
             listener -- a second listener built with the same parameters and
             fed the same public utterances.  Every listener here is a
             deterministic function of the utterance stream, so the replica's
             state equals the public listener's every round (asserted), and S2
             has an exact model of its audience without reading the audience's
             state directly.  Giving the replica different parameters is how a
             *mis*-specified S2 would be built.

``listener_type`` is ``"inf"`` (credulous), ``"vig"`` (vigilant) or
``"switch"`` (a ``DetectionListener`` that starts credulous and switches to
vigilant when its detector crosses ``c``; see ``rsa.detection``).
"""

import numpy as np

from .speaker0 import Speaker0
from .listener0 import Listener0
from .speaker1 import Speaker1
from .listener1 import Listener1
from .speaker2 import Speaker2
from .detection.listener import make_switching_listener


LISTENER_TYPES = ("inf", "vig", "switch")


def make_listener(listener_type, thetas, psis, s1, world, semantics, alpha=1.0,
                  c=3.5, switch_type="hard"):
    """Build the public L1 of the requested type over the S1 model ``s1``."""
    if listener_type in ("inf", "vig"):
        return Listener1(thetas, psis, s1, world, semantics, listener_type, alpha)
    if listener_type == "switch":
        return make_switching_listener(thetas, psis, s1, world, semantics,
                                       c=c, switch_type=switch_type, alpha=alpha)
    raise ValueError(f"unknown listener_type {listener_type!r}; expected one of {LISTENER_TYPES}")


def _level1_stack(thetas, semantics, world, alpha, s1_psi):
    s0 = Speaker0(thetas, semantics=semantics, world=world)
    l0 = Listener0(thetas, s0, semantics=semantics, world=world)
    s1 = Speaker1(thetas, l0, semantics=semantics, world=world, alpha=alpha, psi=s1_psi)
    return s0, l0, s1


def _check_replica(public, replica, round_no):
    d = float(np.abs(public.theta_array() - replica.theta_array()).max())
    same_switch = getattr(public, "switched", None) == getattr(replica, "switched", None)
    if d > 1e-9 or not same_switch:
        raise RuntimeError(f"replica diverged from the public listener at round {round_no}: {d}")


def game(thetas, psis, semantics, world, speaker_type="inf", listener_type="inf",
         alpha=1.0, rounds=1, c=3.5, switch_type="hard", verbose=True,
         check_replica=True):
    """
    S2 speaker (goal ``speaker_type``) against a public L1 of ``listener_type``.

    ``c`` and ``switch_type`` only matter for ``listener_type="switch"``.
    S2's internal listener is a private replica of the public one; it is
    reachable as ``s2.listener``.

    Returns all agents for analysis:
        (speaker2, listener1, speaker1, listener0, speaker0)
    """
    # S1 only supplies tables to the listeners here; its own psi never matters.
    s0, l0, s1 = _level1_stack(thetas, semantics, world, alpha, "inf")
    l1 = make_listener(listener_type, thetas, psis, s1, world, semantics, alpha, c, switch_type)
    replica = make_listener(listener_type, thetas, psis, s1, world, semantics, alpha, c, switch_type)
    s2 = Speaker2(thetas, replica, semantics=semantics, world=world, alpha=alpha, psi=speaker_type)

    for r in range(rounds):
        obs = world.sample_obs()
        utt = s2.sample_utterance(obs)

        if verbose:
            print(f"Round {r+1}: obs={obs}, utt={utt}")

        # Listeners first (they read the speaker tables that produced utt),
        # then every speaker-side belief.
        l1.update(utt)
        replica.update(utt)
        if check_replica:
            _check_replica(l1, replica, r + 1)
        s2.update(obs)
        s1.update(obs)
        l0.update(utt)
        s0.update(obs)

    return s2, l1, s1, l0, s0


def game_s1(thetas, psis, semantics, world, speaker_type="inf", listener_type="inf",
            alpha=1.0, rounds=1, c=3.5, switch_type="hard", verbose=True):
    """
    Simpler game using S1 as the top-level speaker (no S2).

    Returns: (speaker1, listener1, listener0, speaker0)
    """
    s0, l0, s1 = _level1_stack(thetas, semantics, world, alpha, speaker_type)
    l1 = make_listener(listener_type, thetas, psis, s1, world, semantics, alpha, c, switch_type)

    for r in range(rounds):
        obs = world.sample_obs()
        utt = s1.sample_utterance(obs)

        if verbose:
            print(f"Round {r+1}: obs={obs}, utt={utt}")

        l1.update(utt)
        s1.update(obs)
        l0.update(utt)
        s0.update(obs)

    return s1, l1, l0, s0
