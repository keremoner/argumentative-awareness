"""
Literal Listener L0.

Updates belief over theta by reasoning about a literal speaker S0.
P(theta | u) proportional to P(u | theta) * P(theta)
where P(u | theta) = sum_O P_S0(u | O) * P(O | theta).
"""

from copy import deepcopy
import numpy as np
from .core import Belief


class Listener0:
    def __init__(self, thetas, speaker, world, semantics):
        self.state_belief = Belief(thetas)
        self.speaker = speaker
        self.world = world
        self.semantics = semantics
        self.hist = [deepcopy(self.state_belief)]
        self.prior_utt = None
        self.obs_utt = {}

    def infer_state(self, utt):
        """Posterior P(theta | utt)."""
        likelihoods = []
        for state in self.state_belief.values:
            utt_dist = self.speaker.dist_over_utterances_theta(state)
            likelihoods.append(utt_dist.get(utt, 0.0))
        posterior = Belief(self.state_belief.values, self.state_belief.prob.copy())
        posterior.update(likelihoods)
        return posterior

    def infer_obs(self, utt):
        """P(obs | utt) for all observations."""
        if utt in self.obs_utt:
            return self.obs_utt[utt]
        result = {}
        obs_prob = self.distribution_over_obs()
        utt_priors = self.prior_over_utt()
        for obs in self.world.generate_all_obs():
            result[obs] = self.speaker.utterance_prob(obs, utt) * obs_prob[obs] / utt_priors[utt]
        self.obs_utt[utt] = result
        return result

    def distribution_over_obs(self):
        """P(obs) marginalizing over theta."""
        obs_probs = self.world.obs_prob_table(self.state_belief.values) @ self.state_belief.prob
        return dict(zip(self.world.generate_all_obs(), obs_probs))

    def prior_over_utt(self):
        """P(utt) marginalizing over theta and observations."""
        if self.prior_utt is not None:
            return self.prior_utt
        theta_probs = self.state_belief.prob
        obs_priors = self.world.obs_prob_table(self.state_belief.values) @ theta_probs
        utt_priors = {}
        for utt in self.semantics.utterance_space():
            utt_idx = self.semantics.utterance_index(utt)
            total = np.dot(self.speaker._obs_utt_probs[:, utt_idx], obs_priors)
            utt_priors[utt] = float(total)
        self.prior_utt = utt_priors
        return utt_priors

    def update(self, utt):
        """Update belief after hearing utterance."""
        new_belief = self.infer_state(utt)
        self.state_belief = new_belief
        self.prior_utt = None
        self.obs_utt = {}
        self.hist.append(deepcopy(self.state_belief))
        return self.state_belief.as_dict()
