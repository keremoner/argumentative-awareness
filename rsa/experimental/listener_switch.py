"""
Listener1Switch and Listener2Switch.

Hybrid listener that starts as credulous (inf) and switches to vigilant (vig)
when suspicion exceeds a threshold.
"""

import numpy as np
from copy import deepcopy
from ..core import Belief
from ..listener1 import Listener1


class Listener1Switch:
    """
    Hybrid listener that dynamically switches from trusting to vigilant.

    Maintains both internal listeners. Delegates to inf_listener until
    suspicion(utt) >= threshold, then permanently switches to vig_listener.
    """

    def __init__(self, thetas, psis, speaker, world, semantics, threshold=0.6, alpha=1.0):
        self.psis = ["inf"]
        self.thetas = thetas
        joint_values = [(theta, psi) for theta in self.thetas for psi in self.psis]
        self.state_belief = Belief(joint_values)
        self.speaker = speaker
        self.world = world
        self.semantics = semantics
        self.alpha = alpha
        self.threshold = threshold
        self.switched = False
        self.switch_point = 1

        self.vig_listener = Listener1(thetas, psis, speaker, world, semantics,
                                      listener_type="vig", alpha=alpha)
        self.vig_listener.state_belief = deepcopy(self.state_belief)
        self.inf_listener = Listener1(thetas, psis, speaker, world, semantics,
                                      listener_type="inf", alpha=alpha)

        self.hist = [deepcopy(self.state_belief)]
        self.suspicion = []
        self.utt_history = []
        self.suspicions = {}

    def _active(self):
        return self.vig_listener if self.switched else self.inf_listener

    def infer_state(self, utt):
        if self.switched or self.get_suspicion(utt) >= self.threshold:
            return self.vig_listener.infer_state(utt)
        return self.inf_listener.infer_state(utt)

    def infer_obs(self, utt):
        if self.switched or self.get_suspicion(utt) >= self.threshold:
            return self.vig_listener.infer_obs(utt)
        return self.inf_listener.infer_obs(utt)

    def infer_obs_psi(self, utt):
        if self.switched or self.get_suspicion(utt) >= self.threshold:
            return self.vig_listener.infer_obs_psi(utt)
        return self.inf_listener.infer_obs_psi(utt)

    def distribution_over_obs_psi(self):
        return self._active().distribution_over_obs_psi()

    def prior_over_utt(self):
        return self._active().prior_over_utt()

    def update(self, utt):
        current_sus = self.get_suspicion(utt)
        self.suspicion.append(current_sus)
        vig_belief = self.vig_listener.update(utt)
        inf_belief = self.inf_listener.update(utt)

        if current_sus >= self.threshold:
            self.switched = True

        if self.switched:
            new_belief = vig_belief
        else:
            new_belief = inf_belief
            self.switch_point += 1

        self.utt_history.append(utt)
        self.state_belief = new_belief
        self.suspicions = {}
        self.hist.append(deepcopy(self.state_belief))
        return new_belief

    def marginal_theta(self):
        return self._active().marginal_theta()

    def marginal_psi(self):
        return self._active().marginal_psi()

    def get_suspicion(self, utt):
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
        fprs = {threshold: 0.0 for threshold in thresholds}
        for obs in self.world.generate_all_obs():
            speaker_dist = self.speaker.dist_over_utterances_obs(obs, "inf")
            prob = self.world.obs_prob(obs, self.world.theta)
            for u in self.semantics.utterance_space():
                for threshold in thresholds:
                    if self.get_suspicion(u) >= threshold:
                        fprs[threshold] += speaker_dist[u] * prob
        return fprs

    def true_positive_rates(self, thresholds, pers_ratio=0.5):
        tprs = {threshold: 0.0 for threshold in thresholds}
        for obs in self.world.generate_all_obs():
            speaker_dist_high = self.speaker.dist_over_utterances_obs(obs, "high")
            speaker_dist_low = self.speaker.dist_over_utterances_obs(obs, "low")
            prob = self.world.obs_prob(obs, self.world.theta)
            for u in self.semantics.utterance_space():
                for threshold in thresholds:
                    if self.get_suspicion(u) >= threshold:
                        tprs[threshold] += speaker_dist_high[u] * prob * pers_ratio
                        tprs[threshold] += speaker_dist_low[u] * prob * (1 - pers_ratio)
        return tprs


class Listener2Switch:
    """
    Level-2 switch listener. Same logic as Listener1Switch but wraps
    Listener1Switch as the vigilant listener and Listener1 as trusting.
    """

    def __init__(self, thetas, psis, speaker, world, semantics, threshold=0.6, alpha=1.0):
        self.psis = ["inf"]
        self.thetas = thetas
        joint_values = [(theta, psi) for theta in self.thetas for psi in self.psis]
        self.state_belief = Belief(joint_values)
        self.speaker = speaker
        self.world = world
        self.semantics = semantics
        self.alpha = alpha
        self.threshold = threshold
        self.switched = False
        self.switch_point = 1

        self.vig_listener = Listener1Switch(thetas, psis, speaker, world, semantics, alpha=alpha)
        self.vig_listener.state_belief = deepcopy(self.state_belief)
        self.inf_listener = Listener1(thetas, psis, speaker, world, semantics,
                                      listener_type="inf", alpha=alpha)

        self.hist = [deepcopy(self.state_belief)]
        self.suspicion = []
        self.utt_history = []
        self.suspicions = {}

    def _active(self):
        return self.vig_listener if self.switched else self.inf_listener

    def infer_state(self, utt):
        if self.switched or self.get_suspicion(utt) >= self.threshold:
            return self.vig_listener.infer_state(utt)
        return self.inf_listener.infer_state(utt)

    def infer_obs(self, utt):
        if self.switched or self.get_suspicion(utt) >= self.threshold:
            return self.vig_listener.infer_obs(utt)
        return self.inf_listener.infer_obs(utt)

    def infer_obs_psi(self, utt):
        if self.switched or self.get_suspicion(utt) >= self.threshold:
            return self.vig_listener.infer_obs_psi(utt)
        return self.inf_listener.infer_obs_psi(utt)

    def distribution_over_obs_psi(self):
        return self._active().distribution_over_obs_psi()

    def prior_over_utt(self):
        return self._active().prior_over_utt()

    def update(self, utt):
        current_sus = self.get_suspicion(utt)
        self.suspicion.append(current_sus)
        vig_belief = self.vig_listener.update(utt)
        inf_belief = self.inf_listener.update(utt)

        if current_sus >= self.threshold:
            self.switched = True

        if self.switched:
            new_belief = vig_belief
        else:
            new_belief = inf_belief
            self.switch_point += 1

        self.utt_history.append(utt)
        self.state_belief = new_belief
        self.suspicions = {}
        self.hist.append(deepcopy(self.state_belief))
        return new_belief

    def marginal_theta(self):
        return self._active().marginal_theta()

    def marginal_psi(self):
        return self._active().marginal_psi()

    def get_suspicion(self, utt):
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
        fprs = {threshold: 0.0 for threshold in thresholds}
        for obs in self.world.generate_all_obs():
            speaker_dist = self.speaker.dist_over_utterances_obs(obs, "inf")
            prob = self.world.obs_prob(obs, self.world.theta)
            for u in self.semantics.utterance_space():
                for threshold in thresholds:
                    if self.get_suspicion(u) >= threshold:
                        fprs[threshold] += speaker_dist[u] * prob
        return fprs

    def true_positive_rates(self, thresholds, pers_ratio=0.5):
        tprs = {threshold: 0.0 for threshold in thresholds}
        for obs in self.world.generate_all_obs():
            speaker_dist_high = self.speaker.dist_over_utterances_obs(obs, "high")
            speaker_dist_low = self.speaker.dist_over_utterances_obs(obs, "low")
            prob = self.world.obs_prob(obs, self.world.theta)
            for u in self.semantics.utterance_space():
                for threshold in thresholds:
                    if self.get_suspicion(u) >= threshold:
                        tprs[threshold] += speaker_dist_high[u] * prob * pers_ratio
                        tprs[threshold] += speaker_dist_low[u] * prob * (1 - pers_ratio)
        return tprs
