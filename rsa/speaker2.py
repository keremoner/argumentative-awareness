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
        self._obs_list = self.world.generate_all_obs()

        # Caches
        self.utterance_theta_psi = {}
        self.informativeness_obs_utt = {}
        self.persuasiveness_psi = {}
        self.utterances_obs_psi = {}
        self._utterances_obs_psi_array = {}
        self._utterance_theta_psi_array = {}

    def infer_state(self, obs):
        """Posterior P(theta | obs)."""
        likelihoods = self.world.obs_likelihoods(obs, self.thetas)
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
        self._utterances_obs_psi_array = {}
        self._utterance_theta_psi_array = {}
        return self.belief_theta.as_dict()

    def _dist_over_utterances_obs_array(self, obs, psi):
        if (obs, psi) in self._utterances_obs_psi_array:
            return self._utterances_obs_psi_array[(obs, psi)]

        persuasiveness = self.get_persuasiveness(psi, obs)
        truth_row = self.semantics.truth_table(self.world)[self.world.obs_index(obs)]
        scores = np.zeros(len(self._utterances), dtype=float)

        for i, (utt, is_true) in enumerate(zip(self._utterances, truth_row)):
            if is_true:
                info_val = self.get_informativeness_obs_utt(obs, utt)
                pers_val = persuasiveness[utt]
                if psi == "inf":
                    scores[i] = np.exp(self.alpha * info_val)
                elif pers_val > 0:
                    scores[i] = np.exp(self.alpha * np.log2(pers_val))

        score_sum = scores.sum()
        if score_sum == 0:
            scores = truth_row.astype(float)
            score_sum = scores.sum()

        probs = scores / score_sum
        self._utterances_obs_psi_array[(obs, psi)] = probs
        self.utterances_obs_psi[(obs, psi)] = dict(zip(self._utterances, probs))
        return probs

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
                result = float("-inf")
                break
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
        result = {u: 0.0 for u in self._utterances}

        pers_mean = 0.0
        amount = 0
        for utt in self._utterances:
            if psi == "inf":
                result[utt] = 1.0
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
        for utt in self._utterances:
            if psi == "high":
                result[utt] = result[utt] - pers_mean
            elif psi == "low":
                result[utt] = pers_mean - result[utt]

        self.persuasiveness_psi[(psi, obs)] = result
        return result

    def dist_over_utterances_obs(self, obs, psi):
        """P_S2(u | O, psi) using informativeness toward L1 and persuasiveness."""
        if (obs, psi) not in self.utterances_obs_psi:
            self._dist_over_utterances_obs_array(obs, psi)
        return self.utterances_obs_psi[(obs, psi)]

    def dist_over_utterances_theta(self, theta, psi):
        """P(u | theta, psi) marginalizing over observations."""
        if (theta, psi) in self.utterance_theta_psi:
            return self.utterance_theta_psi[(theta, psi)]
        theta_idx = self._theta_to_index[theta]
        obs_probs = self.world.obs_prob_table(self.thetas)[:, theta_idx]
        obs_utt_table = np.vstack([self._dist_over_utterances_obs_array(obs, psi) for obs in self._obs_list])
        probs = obs_utt_table.T @ obs_probs
        result = dict(zip(self._utterances, probs))
        self._utterance_theta_psi_array[(theta, psi)] = probs
        self.utterance_theta_psi[(theta, psi)] = result
        return result

    def dist_over_utterances_theta_array(self, theta, psi):
        if (theta, psi) not in self._utterance_theta_psi_array:
            self.dist_over_utterances_theta(theta, psi)
        return self._utterance_theta_psi_array[(theta, psi)]

    def sample_utterance(self, obs):
        probs = self._dist_over_utterances_obs_array(obs, self.psi)
        self.prob_hist.append(dict(zip(self._utterances, probs)))
        return random.choices(self._utterances, weights=probs, k=1)[0]
