"""
Pragmatic Speaker S2 (paper version).

Reasons about a pragmatic listener L1 instead of literal listener L0.
Uses state-level informativeness (KL divergence) and mean-centered persuasiveness.

Structural symmetry: all speakers Sn (n>=2) use the same machinery as S2,
only changing the index of their internal listener.
"""

import numpy as np
import random
from copy import deepcopy
from .core import Belief


class Speaker2:
    def __init__(self, thetas, listener, semantics, world, alpha=1.0, psi="inf"):
        """
        thetas: list of possible theta values
        listener: a Listener1 object
        semantics: Semantics object
        world: World object
        alpha: rationality parameter
        psi: speaker goal ("inf", "high"=pers+, "low"=pers-)
        """
        self.thetas = thetas
        self.belief_theta = Belief(thetas)
        self.listener = listener
        self.semantics = semantics
        self.world = world
        self.alpha = alpha
        self.psi = psi
        self.hist = [deepcopy(self.belief_theta)]
        self.prob_hist = []
        self._utterances = self.semantics.utterance_space()
        self._theta_to_index = {theta: idx for idx, theta in enumerate(self.thetas)}

        # Caches
        self.utterance_theta_psi = {}
        self.informativeness_obs_utt = {}
        self.persuasiveness_psi = {}
        self.utterances_obs_psi = {}

    def infer_state(self, obs):
        """Posterior P(theta | obs)."""
        likelihoods = [self.world.obs_prob(obs, theta) for theta in self.thetas]
        posterior = Belief(self.thetas, self.belief_theta.prob.copy())
        posterior.update(likelihoods)
        return posterior

    def update(self, obs):
        """Update belief and clear caches."""
        self.belief_theta = self.infer_state(obs)
        self.hist.append(deepcopy(self.belief_theta))
        self.utterance_theta_psi = {}
        self.informativeness_obs_utt = {}
        self.persuasiveness_psi = {}
        self.utterances_obs_psi = {}
        return self.belief_theta.as_dict()

    def get_informativeness_obs_utt(self, obs, utt):
        """
        State-level informativeness via KL divergence.
        Measures how well the utterance helps L1 recover the speaker's belief about theta.
        """
        if (obs, utt) in self.informativeness_obs_utt:
            return self.informativeness_obs_utt[(obs, utt)]
        speaker_dist = self.infer_state(obs).as_dict()
        listener_dist = self.listener.infer_state(utt).marginal(0)
        result = 0
        for theta, prob in speaker_dist.items():
            if listener_dist[theta] == 0:
                if prob == 0:
                    continue
                else:
                    result = float('-inf')
                    break
            else:
                result += np.log2(listener_dist[theta]) * prob
        self.informativeness_obs_utt[(obs, utt)] = result
        return result

    def get_persuasiveness(self, psi, obs):
        """
        Persuasiveness based on how L1's expected theta shifts.
        Mean-centered: subtracts the average persuasiveness of true utterances.
        """
        if (psi, obs) in self.persuasiveness_psi:
            return self.persuasiveness_psi[(psi, obs)]
        utterances = self._utterances
        result = {u: 0.0 for u in utterances}
        current_listener_belief = self.listener.state_belief.marginal(0)
        current_listener_mean = sum(theta * prob for theta, prob in current_listener_belief.items())

        pers_mean = 0
        amount = 0

        for utt in utterances:
            if psi == "inf":
                result[utt] = 1
            elif psi == "high":
                for state, state_prob in self.listener.infer_state(utt).as_dict().items():
                    result[utt] += state[0] * state_prob
            elif psi == "low":
                for state, state_prob in self.listener.infer_state(utt).as_dict().items():
                    result[utt] += state[0] * state_prob
                result[utt] = 1 - result[utt]

            if self.semantics.truth_value(self.world, obs, utt):
                pers_mean += result[utt]
                amount += 1

        pers_mean /= amount
        for utt in utterances:
            if psi == "high":
                result[utt] = result[utt] - pers_mean
            elif psi == "low":
                result[utt] = pers_mean - result[utt]

        self.persuasiveness_psi[(psi, obs)] = result
        return result

    def dist_over_utterances_obs(self, obs, psi):
        """P_S2(u | O, psi) using informativeness toward L1 and persuasiveness."""
        if (obs, psi) in self.utterances_obs_psi:
            return self.utterances_obs_psi[(obs, psi)]
        utterances = self._utterances
        persuasiveness = self.get_persuasiveness(psi, obs)
        truth_row = self.semantics.truth_table(self.world)[self.world.obs_index(obs)]
        true_utterances = [utt for utt, is_true in zip(utterances, truth_row) if is_true]
        beta = 1.0 if psi == "inf" else 0.0

        scores = []
        for utt, is_true in zip(utterances, truth_row):
            info_val = self.get_informativeness_obs_utt(obs, utt)
            pers_val = persuasiveness[utt]
            if is_true:
                if psi == "inf":
                    score = np.exp(self.alpha * info_val)
                else:
                    if pers_val <= 0:
                        score = 0.0
                    else:
                        score = np.exp(self.alpha * np.log2(pers_val) * (1 - beta))
            else:
                score = 0.0
            scores.append(score)

        scores = np.array(scores)
        if np.sum(scores) == 0:
            for i, utt in enumerate(utterances):
                if utt in true_utterances:
                    scores[i] = 1.0

        probs = scores / np.sum(scores)
        self.utterances_obs_psi[(obs, psi)] = dict(zip(utterances, probs))
        return self.utterances_obs_psi[(obs, psi)]

    def dist_over_utterances_theta(self, theta, psi):
        """P(u | theta, psi) marginalizing over observations."""
        if (theta, psi) in self.utterance_theta_psi:
            return self.utterance_theta_psi[(theta, psi)]
        result = {u: 0.0 for u in self._utterances}
        theta_idx = self._theta_to_index[theta]
        obs_probs = self.world.obs_prob_table(self.thetas)[:, theta_idx]
        for obs_idx, obs in enumerate(self.world.generate_all_obs()):
            obs_prob = obs_probs[obs_idx]
            for utt, prob in self.dist_over_utterances_obs(obs, psi).items():
                result[utt] += prob * obs_prob
        self.utterance_theta_psi[(theta, psi)] = result
        return result

    def sample_utterance(self, obs):
        self.prob_hist.append(self.dist_over_utterances_obs(obs, self.psi))
        dist = self.dist_over_utterances_obs(obs, self.psi)
        utterances, probs = zip(*dist.items())
        return random.choices(utterances, weights=probs, k=1)[0]
