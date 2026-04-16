"""
Pragmatic Speaker S1 (paper version).

Uses observation-level informativeness: Inf(u; O) = P_L0(O | u)
and default persuasiveness: PersStr = E_L0[theta | u] for pers+, 1 - E_L0[theta | u] for pers-.

Utterance probability:
  P_S1(u | O, psi, alpha) proportional to
  Truth(u; O) * Inf(u; O)^(alpha*beta) * PersStr(u; psi)^(alpha*(1-beta))
where beta=1 if psi=inf, beta=0 otherwise.
"""

import numpy as np
import random
from copy import deepcopy
from .core import Belief


class Speaker1:
    def __init__(self, thetas, listener, semantics, world, alpha=1.0, psi="inf"):
        """
        thetas: list of possible theta values
        listener: a Listener0 object
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
        self._obs_utt_table_psi = {}

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
        self._obs_utt_table_psi = {}
        return self.belief_theta.as_dict()

    def _dist_over_utterances_obs_array(self, obs, psi):
        if (obs, psi) in self._utterances_obs_psi_array:
            return self._utterances_obs_psi_array[(obs, psi)]

        persuasiveness = self.get_persuasiveness(psi, obs)
        beta = 1.0 if psi == "inf" else 0.0
        truth_row = self.semantics.truth_table(self.world)[self.world.obs_index(obs)]
        scores = np.zeros(len(self._utterances), dtype=float)

        for i, (utt, is_true) in enumerate(zip(self._utterances, truth_row)):
            if is_true:
                info_val = self.get_informativeness_obs_utt(obs, utt)
                pers_val = persuasiveness[utt]
                scores[i] = (info_val ** (self.alpha * beta)) * (pers_val ** (self.alpha * (1 - beta)))

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
        Observation-level informativeness: P_L0(obs | utt).
        How likely is the literal listener to recover obs from utt.
        """
        if (obs, utt) in self.informativeness_obs_utt:
            return self.informativeness_obs_utt[(obs, utt)]
        result = self.listener.infer_obs(utt)
        for obs_case, prob in result.items():
            self.informativeness_obs_utt[(obs_case, utt)] = prob
        return result[obs]

    def get_persuasiveness(self, psi, obs=None):
        """
        Default persuasiveness (paper version):
          - "inf": PersStr = 1 (no persuasion)
          - "high" (pers+): PersStr = E_L0[theta | u]
          - "low" (pers-): PersStr = 1 - E_L0[theta | u]
        """
        if psi in self.persuasiveness_psi:
            return self.persuasiveness_psi[psi]
        result = {u: 0.0 for u in self._utterances}

        for utt in self._utterances:
            if psi == "inf":
                result[utt] = 1
            elif psi == "high":
                for theta, theta_prob in self.listener.infer_state(utt).as_dict().items():
                    result[utt] += theta * theta_prob
            elif psi == "low":
                for theta, theta_prob in self.listener.infer_state(utt).as_dict().items():
                    result[utt] += theta * theta_prob
                result[utt] = 1 - result[utt]

        self.persuasiveness_psi[psi] = result
        return result

    def dist_over_utterances_obs(self, obs, psi):
        """
        P_S1(u | O, psi) proportional to
        Truth(u;O) * Inf^(alpha*beta) * PersStr^(alpha*(1-beta))
        """
        if (obs, psi) not in self.utterances_obs_psi:
            self._dist_over_utterances_obs_array(obs, psi)
        return self.utterances_obs_psi[(obs, psi)]

    def dist_over_utterances_theta(self, theta, psi):
        """P(u | theta, psi) marginalizing over observations."""
        if (theta, psi) in self.utterance_theta_psi:
            return self.utterance_theta_psi[(theta, psi)]
        theta_idx = self._theta_to_index[theta]
        obs_probs = self.world.obs_prob_table(self.thetas)[:, theta_idx]
        obs_utt_table = self.obs_utt_table_for_psi(psi)
        probs = obs_utt_table.T @ obs_probs
        result = dict(zip(self._utterances, probs))
        self._utterance_theta_psi_array[(theta, psi)] = probs
        self.utterance_theta_psi[(theta, psi)] = result
        return result

    def dist_over_utterances_theta_array(self, theta, psi):
        if (theta, psi) not in self._utterance_theta_psi_array:
            self.dist_over_utterances_theta(theta, psi)
        return self._utterance_theta_psi_array[(theta, psi)]

    def obs_utt_table_for_psi(self, psi):
        if psi not in self._obs_utt_table_psi:
            self._obs_utt_table_psi[psi] = np.vstack(
                [self._dist_over_utterances_obs_array(obs, psi) for obs in self._obs_list]
            )
        return self._obs_utt_table_psi[psi]

    def sample_utterance(self, obs):
        probs = self._dist_over_utterances_obs_array(obs, self.psi)
        return random.choices(self._utterances, weights=probs, k=1)[0]
