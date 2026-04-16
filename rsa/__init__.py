"""
RSA Opinion Dynamics - Organized implementation.

Paper: "A Computational Account of Epistemic Vigilance:
        Learning from Selective Truths through Bayesian Reasoning"

Core modules:
    - core: Belief, Semantics, World
    - environment: observation generation, truth conditions, likelihoods
    - speaker0/listener0: literal (level-0) agents
    - speaker1/listener1: pragmatic (level-1) agents (paper version)
    - speaker2: pragmatic (level-2) speaker
    - game: multi-round dyadic interactions
    - setup: factory functions for quick configuration
    - utils: plotting and analysis helpers

Experimental variants in rsa.experimental:
    - speaker1_variants: alternative informativeness/persuasiveness formulations
    - listener_switch: Listener1Switch, Listener2Switch
"""

from .core import Belief, Semantics, World
from .environment import generate_all_observations, utterance_is_true, get_obs_prob
from .speaker0 import Speaker0
from .listener0 import Listener0
from .speaker1 import Speaker1
from .listener1 import Listener1
from .speaker2 import Speaker2
from .game import game, game_s1
from .setup import make_world, make_semantics, make_thetas, make_agents, make_agents_s2
from .utils import (
    pretty_print, plot_speaker_distribution, plot_speaker_grid,
    plot_theta_posterior, plot_theta_learning, plot_psi_learning,
    expected_theta, find_median, persuasiveness_result,
)
