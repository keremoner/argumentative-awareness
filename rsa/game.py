"""
Multi-round dyadic interaction game.

Orchestrates communication between speakers and listeners over multiple rounds.
Each round: world generates observation -> speaker produces utterance -> listener updates.
"""

from .speaker0 import Speaker0
from .listener0 import Listener0
from .speaker1 import Speaker1
from .listener1 import Listener1
from .speaker2 import Speaker2


def game(thetas, psis, semantics, world, speaker_type="inf", listener_type="inf",
         alpha=1.0, rounds=1, verbose=True):
    """
    Run a multi-round dyadic communication game.

    Returns all agents for analysis:
        (speaker2, listener1, speaker1, listener0, speaker0)
    """
    s0 = Speaker0(thetas, semantics=semantics, world=world)
    l0 = Listener0(thetas, s0, semantics=semantics, world=world)
    s1 = Speaker1(thetas, l0, semantics=semantics, world=world, alpha=alpha, psi=speaker_type)
    l1 = Listener1(thetas, psis, s1, semantics=semantics, world=world, alpha=alpha, listener_type=listener_type)
    s2 = Speaker2(thetas, l1, semantics=semantics, world=world, alpha=alpha, psi=speaker_type)

    for r in range(rounds):
        obs = world.sample_obs()
        utt = s2.sample_utterance(obs)

        if verbose:
            print(f"Round {r+1}: obs={obs}, utt={utt}")

        # Update all agents bottom-up
        s2.update(obs)
        l1.update(utt)
        s1.update(obs)
        l0.update(utt)
        s0.update(obs)

    return s2, l1, s1, l0, s0


def game_s1(thetas, psis, semantics, world, speaker_type="inf", listener_type="inf",
            alpha=1.0, rounds=1, verbose=True):
    """
    Simpler game using S1 as the top-level speaker (no S2).

    Returns: (speaker1, listener1, listener0, speaker0)
    """
    s0 = Speaker0(thetas, semantics=semantics, world=world)
    l0 = Listener0(thetas, s0, semantics=semantics, world=world)
    s1 = Speaker1(thetas, l0, semantics=semantics, world=world, alpha=alpha, psi=speaker_type)
    l1 = Listener1(thetas, psis, s1, semantics=semantics, world=world, alpha=alpha, listener_type=listener_type)

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
