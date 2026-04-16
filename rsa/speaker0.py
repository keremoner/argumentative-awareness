"""
Literal Speaker S0.

Chooses uniformly among all literally true utterances for a given observation.
Maintains a Bayesian belief over theta that updates with each new observation.
"""

import random
import numpy as np
from copy import deepcopy
from .core import Belief


class Speaker0:
    def __init__(self, thetas, semantics, world):
        self.thetas = thetas
        self.belief_theta = Belief(thetas)
        self.semantics = semantics
        self.world = world
        self.hist = [deepcopy(self.belief_theta)]
        self.utterances_theta = {}
        self._utterances = self.semantics.utterance_space()
        self._theta_to_index = {theta: idx for idx, theta in enumerate(self.thetas)}
        self._obs_utt_probs = self._build_obs_utt_probs()

    def _build_obs_utt_probs(self):
        truth_table = self.semantics.truth_table(self.world).astype(float)
        row_sums = truth_table.sum(axis=1, keepdims=True)
        return np.divide(truth_table, row_sums, out=np.zeros_like(truth_table), where=row_sums > 0)

    def infer_state(self, obs):
        """Posterior P(theta | obs)."""
        likelihoods = [self.world.obs_prob(obs, theta) for theta in self.thetas]
        posterior = Belief(self.thetas, self.belief_theta.prob.copy())
        posterior.update(likelihoods)
        return posterior

    def update(self, obs):
        """Update belief after observing data."""
        self.belief_theta = self.infer_state(obs)
        self.hist.append(deepcopy(self.belief_theta))
        self.utterances_theta = {}
        return self.belief_theta.as_dict()

    def dist_over_utterances_obs(self, obs):
        """Uniform distribution over all literally true utterances for obs."""
        row = self._obs_utt_probs[self.world.obs_index(obs)]
        return dict(zip(self._utterances, row))

    def utterance_prob(self, obs, utt):
        return float(self._obs_utt_probs[self.world.obs_index(obs), self.semantics.utterance_index(utt)])

    def dist_over_utterances_theta(self, theta):
        """P(u | theta) marginalizing over observations."""
        if theta in self.utterances_theta:
            return self.utterances_theta[theta]
        theta_idx = self._theta_to_index[theta]
        obs_probs = self.world.obs_prob_table(self.thetas)[:, theta_idx]
        probs = self._obs_utt_probs.T @ obs_probs
        result = dict(zip(self._utterances, probs))
        self.utterances_theta[theta] = result
        return result

    def sample_utterance(self, obs):
        probs = self._obs_utt_probs[self.world.obs_index(obs)]
        return random.choices(self._utterances, weights=probs, k=1)[0]
