"""
Experimental Speaker1 variants with different informativeness and persuasiveness formulations.

Each variant shares the same interface as Speaker1 but differs in how
informativeness and/or persuasiveness are computed.

Variants:
  - Speaker1_state_inf_def_pers: state-level inf (KL div) + default pers (E[theta])
  - Speaker1_obs_inf_new_pers1: obs-level inf + normalized pers (min-subtracted)
  - Speaker1_state_inf_new_pers1: state-level inf + quadratic pers loss
  - Speaker1_state_inf_new_pers2: state-level inf + goal distribution pers (weighted by theta)
  - Speaker1_state_inf_new_pers4: state-level inf + logarithmic opinion pool goal distribution
"""

import numpy as np
import random
from copy import deepcopy
from ..core import Belief


# ── Shared base: all variants share infer_state, update, dist_over_utterances_theta, sample_utterance ──

class _Speaker1Base:
    """Base class with shared logic for all Speaker1 variants."""

    def __init__(self, thetas, listener, semantics, world, alpha=1.0, psi="inf"):
        self.thetas = thetas
        self.belief_theta = Belief(thetas)
        self.listener = listener
        self.semantics = semantics
        self.world = world
        self.alpha = alpha
        self.psi = psi
        self.hist = [deepcopy(self.belief_theta)]
        self.utterance_theta_psi = {}
        self.informativeness_obs_utt = {}
        self.persuasiveness_psi = {}
        self.utterances_obs_psi = {}

    def infer_state(self, obs):
        likelihoods = [self.world.obs_prob(obs, theta) for theta in self.thetas]
        posterior = Belief(self.thetas, self.belief_theta.prob.copy())
        posterior.update(likelihoods)
        return posterior

    def update(self, obs):
        self.belief_theta = self.infer_state(obs)
        self.hist.append(deepcopy(self.belief_theta))
        self.utterance_theta_psi = {}
        self.informativeness_obs_utt = {}
        self.persuasiveness_psi = {}
        self.utterances_obs_psi = {}
        return self.belief_theta.as_dict()

    def dist_over_utterances_theta(self, theta, psi):
        if (theta, psi) in self.utterance_theta_psi:
            return self.utterance_theta_psi[(theta, psi)]
        result = {u: 0.0 for u in self.semantics.utterance_space()}
        for obs in self.world.generate_all_obs():
            obs_prob = self.world.obs_prob(obs, theta)
            for utt, prob in self.dist_over_utterances_obs(obs, psi).items():
                result[utt] += prob * obs_prob
        self.utterance_theta_psi[(theta, psi)] = result
        return result

    def sample_utterance(self, obs):
        dist = self.dist_over_utterances_obs(obs, self.psi)
        utterances, probs = zip(*dist.items())
        return random.choices(utterances, weights=probs, k=1)[0]


# ── Informativeness helpers ──

def _obs_informativeness(listener, obs, utt, cache):
    """Observation-level: P_L0(obs | utt)."""
    if (obs, utt) in cache:
        return cache[(obs, utt)]
    result = listener.infer_obs(utt)
    for obs_case, prob in result.items():
        cache[(obs_case, utt)] = prob
    return result[obs]


def _state_informativeness(speaker, listener, obs, utt, cache):
    """State-level: KL-divergence based informativeness."""
    if (obs, utt) in cache:
        return cache[(obs, utt)]
    speaker_dist = speaker.infer_state(obs).as_dict()
    listener_dist = listener.infer_state(utt).marginal(0)
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
    cache[(obs, utt)] = result
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Variant 1: State-level informativeness + Default persuasiveness
# ═══════════════════════════════════════════════════════════════════════════

class Speaker1_state_inf_def_pers(_Speaker1Base):
    """State-level informativeness (KL div) + default persuasiveness (E[theta])."""

    def get_informativeness_obs_utt(self, obs, utt):
        return _state_informativeness(self, self.listener, obs, utt, self.informativeness_obs_utt)

    def get_persuasiveness(self, psi, obs=None):
        if psi in self.persuasiveness_psi:
            return self.persuasiveness_psi[psi]
        utterances = self.semantics.utterance_space()
        result = {u: 0.0 for u in utterances}
        for utt in utterances:
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
        if (obs, psi) in self.utterances_obs_psi:
            return self.utterances_obs_psi[(obs, psi)]
        utterances = self.semantics.utterance_space()
        persuasiveness = self.get_persuasiveness(psi, obs)
        true_utterances = [utt for utt in utterances if self.semantics.is_true(utt, obs)]
        scores = []
        for utt in utterances:
            if self.semantics.is_true(utt, obs):
                info_val = self.get_informativeness_obs_utt(obs, utt)
                pers_val = persuasiveness[utt]
                if psi == "inf":
                    score = np.exp2(self.alpha * info_val)
                else:
                    score = np.exp2(pers_val * self.alpha) if pers_val > 0 else 0.0
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


# ═══════════════════════════════════════════════════════════════════════════
# Variant 2: Obs-level informativeness + New persuasiveness (min-subtracted)
# ═══════════════════════════════════════════════════════════════════════════

class Speaker1_obs_inf_new_pers1(_Speaker1Base):
    """Obs-level informativeness + normalized persuasiveness (min subtracted over true utts)."""

    def get_informativeness_obs_utt(self, obs, utt):
        return _obs_informativeness(self.listener, obs, utt, self.informativeness_obs_utt)

    def get_persuasiveness(self, psi, obs):
        if (psi, obs) in self.persuasiveness_psi:
            return self.persuasiveness_psi[(psi, obs)]
        utterances = self.semantics.utterance_space()
        result = {u: 0.0 for u in utterances}
        min_pers = float('inf')
        for utt in utterances:
            if psi == "inf":
                result[utt] = 1
            elif psi == "high":
                for theta, state_prob in self.listener.infer_state(utt).marginal(0).items():
                    result[utt] += theta * state_prob
            elif psi == "low":
                for theta, state_prob in self.listener.infer_state(utt).marginal(0).items():
                    result[utt] += theta * state_prob
                result[utt] = 1 - result[utt]
            if result[utt] < min_pers and self.semantics.is_true(utt, obs):
                min_pers = result[utt]

        for utt in utterances:
            if not self.semantics.is_true(utt, obs):
                result[utt] = 0.0
            else:
                result[utt] = result[utt] - min_pers

        self.persuasiveness_psi[(psi, obs)] = result
        return result

    def dist_over_utterances_obs(self, obs, psi):
        if (obs, psi) in self.utterances_obs_psi:
            return self.utterances_obs_psi[(obs, psi)]
        utterances = self.semantics.utterance_space()
        persuasiveness = self.get_persuasiveness(psi, obs)
        true_utterances = [utt for utt in utterances if self.semantics.is_true(utt, obs)]
        scores = []
        for utt in utterances:
            if self.semantics.is_true(utt, obs):
                info_val = self.get_informativeness_obs_utt(obs, utt)
                pers_val = persuasiveness[utt]
                score = (info_val ** self.alpha) if psi == "inf" else (pers_val ** self.alpha)
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


# ═══════════════════════════════════════════════════════════════════════════
# Variant 3: State-level informativeness + Quadratic persuasiveness loss
# ═══════════════════════════════════════════════════════════════════════════

class Speaker1_state_inf_new_pers1(_Speaker1Base):
    """State-level informativeness + quadratic loss persuasiveness: -((1-E[theta|u])^2)."""

    def get_informativeness_obs_utt(self, obs, utt):
        return _state_informativeness(self, self.listener, obs, utt, self.informativeness_obs_utt)

    def get_persuasiveness(self, psi, obs):
        if (psi, obs) in self.persuasiveness_psi:
            return self.persuasiveness_psi[(psi, obs)]
        utterances = self.semantics.utterance_space()
        result = {u: 0.0 for u in utterances}
        for utt in utterances:
            if psi == "inf":
                result[utt] = 1
            elif psi == "high":
                for theta, state_prob in self.listener.infer_state(utt).marginal(0).items():
                    result[utt] += theta * state_prob
            elif psi == "low":
                for theta, state_prob in self.listener.infer_state(utt).marginal(0).items():
                    result[utt] += theta * state_prob
                result[utt] = 1 - result[utt]

        for utt in utterances:
            if not self.semantics.is_true(utt, obs):
                result[utt] = float('-inf')
            else:
                result[utt] = -((1 - result[utt]) ** 2)

        self.persuasiveness_psi[(psi, obs)] = result
        return result

    def dist_over_utterances_obs(self, obs, psi):
        if (obs, psi) in self.utterances_obs_psi:
            return self.utterances_obs_psi[(obs, psi)]
        utterances = self.semantics.utterance_space()
        persuasiveness = self.get_persuasiveness(psi, obs)
        true_utterances = [utt for utt in utterances if self.semantics.is_true(utt, obs)]
        scores = []
        for utt in utterances:
            if self.semantics.is_true(utt, obs):
                info_val = self.get_informativeness_obs_utt(obs, utt)
                pers_val = persuasiveness[utt]
                if psi == "inf":
                    score = np.exp2(info_val * self.alpha)
                else:
                    score = np.exp2(pers_val * self.alpha)
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


# ═══════════════════════════════════════════════════════════════════════════
# Variant 4: State-level inf + Goal distribution pers (weighted by theta)
# ═══════════════════════════════════════════════════════════════════════════

class Speaker1_state_inf_new_pers2(_Speaker1Base):
    """
    State-level informativeness + goal distribution persuasiveness.
    Goal distribution: weights theta values proportional to (theta - min_theta)^beta.
    Persuasiveness = KL divergence from listener posterior to goal distribution.
    """

    def get_informativeness_obs_utt(self, obs, utt):
        return _state_informativeness(self, self.listener, obs, utt, self.informativeness_obs_utt)

    def get_goal_distribution(self, obs, beta=1.0):
        """Goal distribution weighted by adjusted theta values."""
        state_dist = self.infer_state(obs).as_dict()
        non_zero_items = [(theta, prob) for theta, prob in state_dist.items() if prob > 0]
        if len(non_zero_items) == 0:
            return {theta: 1.0 / len(self.thetas) for theta in self.thetas}
        theta_values = np.array([theta for theta, _ in non_zero_items])
        min_theta = np.min(theta_values)
        adjusted_thetas = theta_values - min_theta
        scores = adjusted_thetas ** beta
        probs = scores / np.sum(scores) if np.sum(scores) > 0 else np.ones_like(scores) / len(scores)
        result = {theta: 0.0 for theta in self.thetas}
        for (theta, _), prob in zip(non_zero_items, probs):
            result[theta] = prob
        return result

    def get_persuasiveness(self, psi="inf", obs=None, beta=1.0):
        """KL divergence from listener posterior to goal distribution."""
        result = {u: 0.0 for u in self.semantics.utterance_space()}
        if psi == "inf":
            return {u: 1.0 for u in self.semantics.utterance_space()}
        goal_dist = self.get_goal_distribution(obs, beta=beta)
        for utt in self.semantics.utterance_space():
            listener_dist = self.listener.infer_state(utt).marginal(0)
            for theta, prob in goal_dist.items():
                if listener_dist[theta] == 0:
                    if prob == 0:
                        continue
                    else:
                        result[utt] = float('-inf')
                        break
                else:
                    result[utt] += np.log2(listener_dist[theta]) * prob
        return result

    def dist_over_utterances_obs(self, obs, psi):
        if (obs, psi) in self.utterances_obs_psi:
            return self.utterances_obs_psi[(obs, psi)]
        utterances = self.semantics.utterance_space()
        persuasiveness = self.get_persuasiveness(psi, obs)
        true_utterances = [utt for utt in utterances if self.semantics.is_true(utt, obs)]
        scores = []
        for utt in utterances:
            if self.semantics.is_true(utt, obs):
                info_val = self.get_informativeness_obs_utt(obs, utt)
                pers_val = persuasiveness[utt]
                if psi == "inf":
                    score = np.exp2(info_val * self.alpha)
                else:
                    score = np.exp2(pers_val * self.alpha)
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


# ═══════════════════════════════════════════════════════════════════════════
# Variant 5: State-level inf + Logarithmic opinion pool goal distribution
# ═══════════════════════════════════════════════════════════════════════════

class Speaker1_state_inf_new_pers4(_Speaker1Base):
    """
    State-level informativeness + logarithmic opinion pool (geometric mean) goal distribution.
    Weights = E[theta | u] for each truthful utterance, normalized by subtracting minimum.
    """

    def get_informativeness_obs_utt(self, obs, utt):
        return _state_informativeness(self, self.listener, obs, utt, self.informativeness_obs_utt)

    def get_goal_distribution(self, obs, beta=1.0):
        """Logarithmic opinion pool of listener posteriors for truthful utterances."""
        truthful_utts = [utt for utt in self.semantics.utterance_space()
                         if self.semantics.is_true(utt, obs)]
        if len(truthful_utts) == 0:
            return {theta: 1.0 / len(self.thetas) for theta in self.thetas}

        posteriors = []
        expected_values = []
        for utt in truthful_utts:
            posterior = self.listener.infer_state(utt).marginal(0)
            posteriors.append(posterior)
            exp_val = sum(theta * posterior[theta] for theta in self.thetas)
            expected_values.append(exp_val)

        min_val = min(expected_values)
        weights = [ev - min_val for ev in expected_values]
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0 / len(expected_values)] * len(expected_values)
        else:
            weights = [w / total_weight for w in weights]

        # Geometric mean in log space
        result = {}
        for theta in self.thetas:
            log_prob = 0.0
            for posterior, weight in zip(posteriors, weights):
                if posterior[theta] > 0:
                    log_prob += weight * np.log2(posterior[theta])
                else:
                    log_prob = float('-inf')
                    break
            result[theta] = np.exp2(log_prob) if log_prob != float('-inf') else 0.0

        total = sum(result.values())
        if total > 0:
            result = {theta: prob / total for theta, prob in result.items()}
        else:
            result = {theta: 1.0 / len(self.thetas) for theta in self.thetas}
        return result

    def get_persuasiveness(self, psi, obs, beta=1.0):
        """KL divergence from listener posterior to log opinion pool goal distribution."""
        result = {u: 0.0 for u in self.semantics.utterance_space()}
        if psi == "inf":
            return {u: 1.0 for u in self.semantics.utterance_space()}
        goal_dist = self.get_goal_distribution(obs, beta=beta)
        for utt in self.semantics.utterance_space():
            listener_dist = self.listener.infer_state(utt).marginal(0)
            for theta, prob in goal_dist.items():
                if listener_dist[theta] == 0:
                    if prob == 0:
                        continue
                    else:
                        result[utt] = float('-inf')
                        break
                else:
                    result[utt] += np.log2(listener_dist[theta]) * prob
        return result

    def dist_over_utterances_obs(self, obs, psi):
        if (obs, psi) in self.utterances_obs_psi:
            return self.utterances_obs_psi[(obs, psi)]
        utterances = self.semantics.utterance_space()
        persuasiveness = self.get_persuasiveness(psi, obs)
        true_utterances = [utt for utt in utterances if self.semantics.is_true(utt, obs)]
        scores = []
        for utt in utterances:
            if self.semantics.is_true(utt, obs):
                info_val = self.get_informativeness_obs_utt(obs, utt)
                pers_val = persuasiveness[utt]
                if psi == "inf":
                    score = np.exp2(info_val * self.alpha)
                else:
                    score = np.exp2(pers_val * self.alpha)
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
