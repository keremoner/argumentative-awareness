"""
Pragmatic Listener L1 (paper version).

Maintains a joint belief P(theta, psi) and reasons about S1's goals.

- Credulous (listener_type="inf"): assumes psi=inf only (cooperative world)
- Vigilant (listener_type="vig"): considers all psi types (strategic world)

P_L1(theta, psi | u) is proportional to P_L1(theta, psi) * P_S1(u | theta, psi)

Two update paths exist and are numerically identical:

* ``update(utt)`` reads the speaker's *current* utterance tables through
  ``speaker.dist_over_utterances_theta_array(theta, psi)``;
* ``update_with_tables(utt, tables_by_psi)`` takes an explicit
  ``{psi: (n_obs, n_utt) array}`` triple and never touches the speaker.  This is
  what makes a retrospective replay possible after the shared speaker object
  has moved on (``DetectionListener`` snapshots the tables every round).
"""

import numpy as np
from copy import deepcopy
from .core import Belief


class Listener1:
    def __init__(self, thetas, psis, speaker, world, semantics, listener_type, alpha=1.0):
        """
        thetas: list of theta values
        psis: list of possible speaker goals (e.g. ["inf", "high", "low"])
        speaker: a Speaker1 object (or anything exposing
                 ``dist_over_utterances_theta_array`` / ``obs_utt_table_for_psi``)
        listener_type: "inf" (credulous) or "vig" (vigilant)
        """
        if listener_type == "inf":
            psis = ["inf"]
        joint_values = [(theta, psi) for theta in thetas for psi in psis]
        self.state_belief = Belief(joint_values)
        self.speaker = speaker
        self.world = world
        self.semantics = semantics
        self.psis = list(psis)
        self.alpha = alpha
        self.listener_type = listener_type
        self.thetas = list(thetas)
        self._theta_to_index = {theta: idx for idx, theta in enumerate(self.thetas)}
        self._psi_to_index = {psi: idx for idx, psi in enumerate(self.psis)}
        self._utterances = self.semantics.utterance_space()
        self._obs_list = self.world.generate_all_obs()
        # (theta, psi) coordinates of every joint state, for the vectorised
        # table-based likelihood
        self._joint_theta_idx = np.array(
            [self._theta_to_index[theta] for theta, _ in joint_values], dtype=int)
        self._joint_psi_idx = np.array(
            [self._psi_to_index[psi] for _, psi in joint_values], dtype=int)

        self.hist = [deepcopy(self.state_belief)]
        self.suspicion = []
        self.utt_history = []

        # Caches
        self.obs_psi_utt = {}
        self.obs_psi = None
        self.prior_utt = None
        self.obs_utt = {}
        self.suspicions = {}
        self.state_utt = {}
        self._state_utt_array = {}
        self._obs_psi_array = None
        self._obs_psi_utt_array = {}
        self._prior_utt_array = None

    # ------------------------------------------------------------------
    # Live path (reads the speaker's current tables)
    # ------------------------------------------------------------------

    def infer_state(self, utt):
        """Posterior P(theta, psi | utt)."""
        if utt in self.state_utt:
            return self.state_utt[utt]
        utt_idx = self.semantics.utterance_index(utt)
        likelihoods = np.array(
            [
                self.speaker.dist_over_utterances_theta_array(theta, psi)[utt_idx]
                for theta, psi in self.state_belief.values
            ],
            dtype=float,
        )
        posterior = Belief(self.state_belief.values, self.state_belief.prob.copy())
        posterior.update(likelihoods)
        self.state_utt[utt] = posterior
        self._state_utt_array[utt] = posterior.prob.copy()
        return posterior

    def _distribution_over_obs_psi_array(self):
        if self._obs_psi_array is not None:
            return self._obs_psi_array

        obs_prob_table = self.world.obs_prob_table(self.thetas)
        result = np.zeros((len(self._obs_list), len(self.psis)), dtype=float)
        for (theta, psi), state_prob in zip(self.state_belief.values, self.state_belief.prob):
            result[:, self._psi_to_index[psi]] += obs_prob_table[:, self._theta_to_index[theta]] * state_prob

        self._obs_psi_array = result
        self.obs_psi = {
            (obs, psi): result[obs_idx, psi_idx]
            for obs_idx, obs in enumerate(self._obs_list)
            for psi_idx, psi in enumerate(self.psis)
        }
        return result

    def infer_obs(self, utt):
        """P(obs | utt) marginalizing over psi."""
        if utt in self.obs_utt:
            return self.obs_utt[utt]
        obs_psi_utt = self._infer_obs_psi_array(utt)
        probs = obs_psi_utt.sum(axis=1)
        result = dict(zip(self._obs_list, probs))
        self.obs_utt[utt] = result
        return result

    def _infer_obs_psi_array(self, utt):
        if utt in self._obs_psi_utt_array:
            return self._obs_psi_utt_array[utt]

        utt_idx = self.semantics.utterance_index(utt)
        obs_psi = self._distribution_over_obs_psi_array()
        utt_prior = self._prior_over_utt_array()[utt_idx]
        result = np.zeros_like(obs_psi)

        for psi_idx, psi in enumerate(self.psis):
            obs_utt = self.speaker.obs_utt_table_for_psi(psi)[:, utt_idx]
            result[:, psi_idx] = obs_utt * obs_psi[:, psi_idx] / utt_prior

        self._obs_psi_utt_array[utt] = result
        self.obs_psi_utt[utt] = {
            (obs, psi): result[obs_idx, psi_idx]
            for obs_idx, obs in enumerate(self._obs_list)
            for psi_idx, psi in enumerate(self.psis)
        }
        return result

    def infer_obs_psi(self, utt):
        """P(obs, psi | utt)."""
        if utt not in self.obs_psi_utt:
            self._infer_obs_psi_array(utt)
        return self.obs_psi_utt[utt]

    def distribution_over_obs_psi(self):
        """P(obs, psi) marginalizing over theta."""
        if self.obs_psi is None:
            self._distribution_over_obs_psi_array()
        return self.obs_psi

    def _prior_over_utt_array(self):
        if self._prior_utt_array is not None:
            return self._prior_utt_array

        obs_psi = self._distribution_over_obs_psi_array()
        utt_priors = np.zeros(len(self._utterances), dtype=float)
        for psi_idx, psi in enumerate(self.psis):
            obs_utt = self.speaker.obs_utt_table_for_psi(psi)
            utt_priors += obs_utt.T @ obs_psi[:, psi_idx]

        self._prior_utt_array = utt_priors
        self.prior_utt = dict(zip(self._utterances, utt_priors))
        return utt_priors

    def prior_over_utt(self):
        """P(utt) marginalizing over theta, psi, and observations."""
        if self.prior_utt is None:
            self._prior_over_utt_array()
        return self.prior_utt

    def _clear_caches(self):
        self.obs_psi_utt = {}
        self.obs_psi = None
        self.prior_utt = None
        self.obs_utt = {}
        self.state_utt = {}
        self._state_utt_array = {}
        self._obs_psi_array = None
        self._obs_psi_utt_array = {}
        self._prior_utt_array = None

    def update(self, utt):
        """Update belief after hearing utterance (live speaker tables)."""
        # Legacy single-utterance suspicion (Fang-style).  It needs the
        # speaker's own internal listener; table-replay stand-ins have none.
        if getattr(self.speaker, "listener", None) is not None:
            self.suspicion.append(self.get_suspicion(utt))
        else:
            self.suspicion.append(float("nan"))
        self.utt_history.append(utt)
        new_belief = self.infer_state(utt)
        self.state_belief = new_belief
        self.hist.append(deepcopy(self.state_belief))
        self._clear_caches()
        return self.state_belief

    # ------------------------------------------------------------------
    # Table path (explicit per-round speaker tables, speaker never consulted)
    # ------------------------------------------------------------------

    def likelihoods_from_tables(self, utt, tables_by_psi):
        """P(u | theta, psi) for every joint state, from explicit tables.

        ``tables_by_psi[psi]`` is the (n_obs, n_utt) matrix P_S(u | O, psi).
        P(u | theta, psi) = sum_O P(O | theta) P_S(u | O, psi).
        """
        utt_idx = self.semantics.utterance_index(utt)
        obs_prob_table = self.world.obs_prob_table(self.thetas)      # (n_obs, n_theta)
        per_psi = np.empty((len(self.psis), len(self.thetas)), dtype=float)
        for psi_idx, psi in enumerate(self.psis):
            col = np.asarray(tables_by_psi[psi], dtype=float)[:, utt_idx]
            per_psi[psi_idx] = obs_prob_table.T @ col
        return per_psi[self._joint_psi_idx, self._joint_theta_idx]

    def infer_state_with_tables(self, utt, tables_by_psi):
        """Posterior P(theta, psi | utt) using explicit tables (no caching, no mutation)."""
        posterior = Belief(self.state_belief.values, self.state_belief.prob.copy())
        posterior.update(self.likelihoods_from_tables(utt, tables_by_psi))
        return posterior

    def update_with_tables(self, utt, tables_by_psi):
        """Update from an explicit table triple rather than the live speaker.

        Identical to ``update`` when ``tables_by_psi`` equals the speaker's
        current tables.  The legacy ``suspicion`` list is not extended.
        """
        self.utt_history.append(utt)
        self.state_belief = self.infer_state_with_tables(utt, tables_by_psi)
        self.hist.append(deepcopy(self.state_belief))
        self._clear_caches()
        return self.state_belief

    def seed_from_theta_marginal(self, theta_probs):
        """Reset the joint belief to ``theta_probs`` spread uniformly over psi.

        This is the soft-switch initialisation: the theta-marginal is inherited
        and the listener is direction-blind (uniform over psi).
        """
        n_psi = len(self.psis)
        prior = np.array(
            [theta_probs.get(theta, 0.0) / n_psi for (theta, psi) in self.state_belief.values],
            dtype=float,
        )
        s = prior.sum()
        if s > 0:
            prior = prior / s
        self.state_belief = Belief(self.state_belief.values, prior)
        self.hist = [deepcopy(self.state_belief)]
        self._clear_caches()
        return self.state_belief

    # ------------------------------------------------------------------
    # Marginals
    # ------------------------------------------------------------------

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

    def theta_array(self):
        """Theta-marginal as an ndarray aligned with ``self.thetas``."""
        out = np.zeros(len(self.thetas), dtype=float)
        np.add.at(out, self._joint_theta_idx, self.state_belief.prob)
        return out

    # ------------------------------------------------------------------
    # Legacy single-utterance suspicion (Fang-style), kept for reference
    # ------------------------------------------------------------------

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
                if prob > speaker_probs[utt]:
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
