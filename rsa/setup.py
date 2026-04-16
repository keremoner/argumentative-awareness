"""
Quick setup helpers for common configurations.

Provides factory functions to create worlds, agents, and full agent hierarchies
with minimal boilerplate.
"""

from itertools import product
from .core import Semantics, World, Belief
from .environment import generate_all_observations, utterance_is_true, get_obs_prob
from .speaker0 import Speaker0
from .listener0 import Listener0
from .speaker1 import Speaker1
from .listener1 import Listener1
from .speaker2 import Speaker2


def make_world(theta, n=1, m=7):
    """Create a World with given parameters."""
    world_parameters = {"n": n, "m": m}
    return World(theta, world_parameters, get_obs_prob, generate_all_observations)


def make_semantics(n=1):
    """Create Semantics with appropriate utterance space for n patients."""
    quantifiers = ["none", "some", "most", "all"]
    predicates = ["effective", "ineffective"]
    if n > 1:
        utterances = list(product(quantifiers, quantifiers, predicates))
    else:
        utterances = list(product(quantifiers, predicates))
    return Semantics(utterances, utterance_is_true)


def make_thetas(step=0.1, include_zero=False, include_one=False):
    """Create list of theta values. Default: [0.1, 0.2, ..., 0.9]."""
    start = 0 if include_zero else 1
    end = 11 if include_one else 10
    return [round(step * i, 10) for i in range(start, end)]


def make_agents(theta=0.3, n=1, m=7, alpha=3.0, speaker_psi="inf",
                listener_type="vig", psis=None, thetas=None):
    """
    Create a complete agent hierarchy: S0 -> L0 -> S1 -> L1.

    Returns dict with keys: world, semantics, thetas, s0, l0, s1, l1
    """
    if thetas is None:
        thetas = make_thetas()
    if psis is None:
        psis = ["inf", "high", "low"]

    world = make_world(theta, n=n, m=m)
    semantics = make_semantics(n=n)
    s0 = Speaker0(thetas, semantics=semantics, world=world)
    l0 = Listener0(thetas, s0, semantics=semantics, world=world)
    s1 = Speaker1(thetas, l0, semantics=semantics, world=world, alpha=alpha, psi=speaker_psi)
    l1 = Listener1(thetas, psis, s1, semantics=semantics, world=world, alpha=alpha, listener_type=listener_type)

    return {
        "world": world,
        "semantics": semantics,
        "thetas": thetas,
        "psis": psis,
        "s0": s0,
        "l0": l0,
        "s1": s1,
        "l1": l1,
    }


def make_agents_s2(theta=0.3, n=1, m=7, alpha=3.0, speaker_psi="inf",
                   listener_type="vig", psis=None, thetas=None):
    """
    Create a full hierarchy with S2: S0 -> L0 -> S1 -> L1 -> S2.

    Returns dict with keys: world, semantics, thetas, s0, l0, s1, l1, s2
    """
    agents = make_agents(theta=theta, n=n, m=m, alpha=alpha, speaker_psi=speaker_psi,
                         listener_type=listener_type, psis=psis, thetas=thetas)
    s2 = Speaker2(agents["thetas"], agents["l1"], semantics=agents["semantics"],
                  world=agents["world"], alpha=alpha, psi=speaker_psi)
    agents["s2"] = s2
    return agents
