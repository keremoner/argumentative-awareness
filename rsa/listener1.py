"""
Pragmatic Listener L1 (paper version).

Maintains a joint belief P(theta, psi) and reasons about S1's goals.

- Credulous (listener_type="inf"): assumes psi=inf only (cooperative world)
- Vigilant (listener_type="vig"): considers all psi types (strategic world)

P_L1(theta, psi | u) ∝ P_L1(theta, psi) * P_S1(u | theta, psi)
"""

import numpy as np
from copy import deepcopy
from .core import Belief


class Listener1:
    def __init__(self, thetas, psis, speaker, world, semantics, listener_type, alpha=1.0):
        """
        thetas: list of theta values
        psis: list of possible speaker goals (e.g. ["inf", "high", "low"])
        speaker: a Speaker1 object
        listener_type: "inf" (credulous) or "vig" (vigilant)
        """
        if listener_type == "inf":
            psis = ["inf"]
        joint_values = [(theta, psi) for theta in thetas for psi in psis]
        self.state_belief = Belief(joint_values)
        self.speaker = speaker
        self.world = world
        self.semantics = semantics
        self.psis = psis
        self.alpha = alpha
        self.listener_type = listener_type
        self.thetas = thetas
        self._theta_to_index = {theta: idx for idx, theta in enumerate(self.thetas)}

        self.hist = [deepcopy(self.state_belief)]
        self.suspicion = []
        self.utt_history = []

        # Caches
        self.obs_psi_utt = {}
        self.obs_psi = None
        self.prior_utt = None
        self.obs_utt = {}
        self.suspicions = {}

    def infer_state(self, utt):
        """Posterior P(theta, psi | utt)."""
        likelihoods = []
        for state in self.state_belief.values:
            utt_dist = self.speaker.dist_over_utterances_theta(state[0], state[1])
            likelihoods.append(utt_dist.get(utt, 0.0))
        posterior = Belief(self.state_belief.values, self.state_belief.prob.copy())
        posterior.update(likelihoods)
        return posterior

    def infer_obs(self, utt):
        """P(obs | utt) marginalizing over psi."""
        if utt in self.obs_utt:
            return self.obs_utt[utt]
        result = {}
        obs_psi_utt = self.infer_obs_psi(utt)
        psi_probs = self.marginal_psi()
        for obs in self.world.generate_all_obs():
            for psi, prob in psi_probs.items():
                result[obs] = result.get(obs, 0.0) + obs_psi_utt[(obs, psi)]
        self.obs_utt[utt] = result
        return result

    def infer_obs_psi(self, utt):
        """P(obs, psi | utt)."""
        if utt in self.obs_psi_utt:
            return self.obs_psi_utt[utt]
        result = {(obs, psi): 0.0 for obs in self.world.generate_all_obs() for psi in self.psis}
        obs_psi = self.distribution_over_obs_psi()
        utt_priors = self.prior_over_utt()
        for obs in self.world.generate_all_obs():
            for psi in self.psis:
                result[(obs, psi)] = (
                    self.speaker.dist_over_utterances_obs(obs, psi)[utt]
                    * obs_psi[(obs, psi)]
                    / utt_priors[utt]
                )
        self.obs_psi_utt[utt] = result
        return result

    def distribution_over_obs_psi(self):
        """P(obs, psi) marginalizing over theta."""
        if self.obs_psi is not None:
            return self.obs_psi
        result = {(obs, psi): 0.0 for obs in self.world.generate_all_obs() for psi in self.psis}
        obs_prob_table = self.world.obs_prob_table(self.thetas)
        for state, state_prob in self.state_belief.as_dict().items():
            theta_idx = self._theta_to_index[state[0]]
            for obs_idx, obs in enumerate(self.world.generate_all_obs()):
                result[(obs, state[1])] += obs_prob_table[obs_idx, theta_idx] * state_prob
        self.obs_psi = result
        return result

    def prior_over_utt(self):
        """P(utt) marginalizing over theta, psi, and observations."""
        if self.prior_utt is not None:
            return self.prior_utt
        utt_priors = {utt: 0.0 for utt in self.semantics.utterance_space()}
        obs_psi_dist = self.distribution_over_obs_psi()
        for obs_case in self.world.generate_all_obs():
            for psi in self.psis:
                speaker_dist = self.speaker.dist_over_utterances_obs(obs_case, psi)
                for utt in self.semantics.utterance_space():
                    utt_priors[utt] += speaker_dist[utt] * obs_psi_dist[(obs_case, psi)]
        self.prior_utt = utt_priors
        return utt_priors

    def update(self, utt):
        """Update belief after hearing utterance."""
        self.suspicion.append(self.get_suspicion(utt))
        self.utt_history.append(utt)
        new_belief = self.infer_state(utt)
        self.state_belief = new_belief
        self.hist.append(deepcopy(self.state_belief))

        # Clear caches
        self.obs_psi_utt = {}
        self.obs_psi = None
        self.prior_utt = None
        self.obs_utt = {}
        return self.state_belief

    def marginal_theta(self):
        """Marginal distribution over theta."""
        theta_probs = {}
        for (theta, psi), p in zip(self.state_belief.values, self.state_belief.prob):
            theta_probs[theta] = theta_probs.get(theta, 0.0) + p
        return theta_probs

    def marginal_psi(self):
        """Marginal distribution over psi (speaker goal)."""
        psi_probs = {}
        for (theta, psi), p in zip(self.state_belief.values, self.state_belief.prob):
            psi_probs[psi] = psi_probs.get(psi, 0.0) + p
        return psi_probs

    def get_suspicion(self, utt):
        """
        Suspicion score: probability that the speaker chose a suboptimal utterance
        (i.e., there exist utterances the informative speaker would prefer).
        """
        if utt in self.suspicions:
            return self.suspicions[utt]
        suspicion = 0.0
        for state, state_prior in self.speaker.listener.infer_state(utt).marginal(0).items():
            speaker_probs = self.speaker.dist_over_utterances_theta(state, "inf")
            for u, prob in speaker_probs.items():
                if np.isclose(prob, speaker_probs[utt], rtol=1e-9, atol=1e-12):
                    continue
                elif prob > speaker_probs[utt]:
                    suspicion += state_prior * prob
        self.suspicions[utt] = suspicion
        return suspicion

    def false_positive_rates(self, thresholds):
        """FPR at different suspicion thresholds (against informative speaker)."""
        fprs = {threshold: 0.0 for threshold in thresholds}
        for state, prob in self.state_belief.marginal(0).items():
            speaker_dist = self.speaker.dist_over_utterances_theta(state, "inf")
            for u in self.semantics.utterance_space():
                for threshold in thresholds:
                    if self.get_suspicion(u) >= threshold:
                        fprs[threshold] += speaker_dist[u] * prob
        return fprs

    def true_positive_rates(self, thresholds, pers_ratio=0.5):
        """TPR at different suspicion thresholds (against persuasive speakers)."""
        tprs = {threshold: 0.0 for threshold in thresholds}
        for state, prob in self.state_belief.marginal(0).items():
            speaker_dist_high = self.speaker.dist_over_utterances_theta(state, "high")
            speaker_dist_low = self.speaker.dist_over_utterances_theta(state, "low")
            for u in self.semantics.utterance_space():
                for threshold in thresholds:
                    if self.get_suspicion(u) >= threshold:
                        tprs[threshold] += speaker_dist_high[u] * prob * pers_ratio
                        tprs[threshold] += speaker_dist_low[u] * prob * (1 - pers_ratio)
        return tprs
