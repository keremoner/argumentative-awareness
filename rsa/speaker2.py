"""
Pragmatic Speaker S2 (Fang 2025, "structural symmetry" section).

S2 is the S1 template with the internal listener swapped: instead of a literal
listener L0 it reasons about a pragmatic listener L1.

    P_S2(u | O, psi, alpha)  ~  Truth(u; O) * Inf(u; O)^(alpha*beta)
                                          * PersStr(u; psi)^(alpha*(1-beta))
    Inf(u; O)      = P_L1(O | u)
    PersStr(u;psi) = E_L1[theta | u]        (pers+ / "high")
                   = 1 - E_L1[theta | u]    (pers- / "low")
                   = 1                      (inf)
    beta = 1 if psi == "inf" else 0

Both terms are one-step-ahead quantities of the *same* internal listener.
The listener is generic; it must be updatable on ``u`` by the caller and
expose:

* ``infer_obs(u)`` -> dict O -> P(O | u), and ``infer_state(u).marginal(0)``
  -> dict theta -> P(theta | u): a plain ``Listener1``;
* optionally ``peek_obs(u)`` and ``peek(u)`` with the same signatures.  When
  present they are preferred: a ``DetectionListener`` uses them to put its
  detector update, and the switch that ``u`` may trigger, *inside* both
  utility terms, so that an S2 modelling a switching listener sees the
  listener it would actually be talking to.

``belief_theta`` is updated on O exactly as for S1 but never enters the policy.
When every score is zero (only possible through numerical underflow) the
speaker falls back to a uniform choice among literally true utterances.

Caches are stamped with the internal listener's ``version`` and dropped
automatically when it moves (see ``rsa.core``); ``update`` also clears them.

All speakers Sn (n >= 2) use this machinery with the index of their internal
listener changed.
"""

import numpy as np
import random
from copy import deepcopy
from .core import Belief
from .utils import expected_theta


PSI_VALUES = ("inf", "high", "low")


class Speaker2:
    def __init__(self, thetas, listener, semantics, world, alpha=1.0, psi="inf"):
        """
        thetas: list of possible theta values
        listener: the internal pragmatic listener (Listener1 or DetectionListener)
        semantics: Semantics object
        world: World object
        alpha: rationality parameter
        psi: speaker goal ("inf", "high"=pers+, "low"=pers-)
        """
        if psi not in PSI_VALUES:
            raise ValueError(f"unknown psi {psi!r}; expected one of {PSI_VALUES}")
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
        self._version = 0
        self._dep_version = None
        self.clear_caches()

    # ------------------------------------------------------------------
    # Cache/version protocol
    # ------------------------------------------------------------------

    @property
    def version(self):
        """Fingerprint of everything the tables depend on: this agent's own
        updates and the internal listener's version (recursively)."""
        return (self._version, getattr(self.listener, "version", None))

    def clear_caches(self):
        self.utterance_theta_psi = {}
        self.informativeness_utt = {}
        self.persuasiveness_psi = {}
        self.utterances_obs_psi = {}
        self._utterances_obs_psi_array = {}
        self._utterance_theta_psi_array = {}
        self._obs_utt_table_psi = {}

    def _sync(self):
        """Drop every cache if the internal listener has moved since it was filled."""
        v = getattr(self.listener, "version", None)
        if v != self._dep_version:
            self.clear_caches()
            self._dep_version = v

    # ------------------------------------------------------------------
    # Belief over theta (bookkeeping only; not used by the policy)
    # ------------------------------------------------------------------

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
        self._version += 1
        self.clear_caches()
        return self.belief_theta.as_dict()

    # ------------------------------------------------------------------
    # Utility terms -- one-step-ahead quantities of the internal listener
    # ------------------------------------------------------------------

    def _listener_obs_posterior(self, utt):
        """P_L(O | utt) of the listener that would be active after ``utt``."""
        peek_obs = getattr(self.listener, "peek_obs", None)
        if peek_obs is not None:
            return peek_obs(utt)
        return self.listener.infer_obs(utt)

    def _listener_theta_marginal(self, utt):
        """P_L(theta | utt) of the listener that would be active after ``utt``."""
        peek = getattr(self.listener, "peek", None)
        if peek is not None:
            return peek(utt)
        return self.listener.infer_state(utt).marginal(0)

    def get_informativeness_obs_utt(self, obs, utt):
        """Inf(u; O) = P_L1(O | u): how likely the pragmatic listener recovers O from u."""
        self._sync()
        if utt not in self.informativeness_utt:
            self.informativeness_utt[utt] = self._listener_obs_posterior(utt)
        return self.informativeness_utt[utt][obs]

    def get_persuasiveness(self, psi, obs=None):
        """PersStr(u; psi) for every utterance (independent of O).

          - "inf":  1
          - "high": E_L1[theta | u]
          - "low":  1 - E_L1[theta | u]
        """
        if psi not in PSI_VALUES:
            raise ValueError(f"unknown psi {psi!r}; expected one of {PSI_VALUES}")
        self._sync()
        if psi in self.persuasiveness_psi:
            return self.persuasiveness_psi[psi]
        result = {}
        for utt in self._utterances:
            if psi == "inf":
                result[utt] = 1.0
            else:
                e_theta = expected_theta(self._listener_theta_marginal(utt))
                result[utt] = e_theta if psi == "high" else 1.0 - e_theta
        self.persuasiveness_psi[psi] = result
        return result

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def _dist_over_utterances_obs_array(self, obs, psi):
        self._sync()
        if (obs, psi) in self._utterances_obs_psi_array:
            return self._utterances_obs_psi_array[(obs, psi)]

        beta = 1.0 if psi == "inf" else 0.0
        persuasiveness = self.get_persuasiveness(psi) if beta < 1.0 else None
        truth_row = self.semantics.truth_table(self.world)[self.world.obs_index(obs)]
        scores = np.zeros(len(self._utterances), dtype=float)

        for i, (utt, is_true) in enumerate(zip(self._utterances, truth_row)):
            if not is_true:
                continue
            val = 1.0
            if beta > 0.0:
                val *= self.get_informativeness_obs_utt(obs, utt) ** (self.alpha * beta)
            if beta < 1.0:
                val *= persuasiveness[utt] ** (self.alpha * (1.0 - beta))
            scores[i] = val

        score_sum = scores.sum()
        if score_sum == 0:
            scores = truth_row.astype(float)
            score_sum = scores.sum()

        probs = scores / score_sum
        self._utterances_obs_psi_array[(obs, psi)] = probs
        self.utterances_obs_psi[(obs, psi)] = dict(zip(self._utterances, probs))
        return probs

    def dist_over_utterances_obs(self, obs, psi):
        """P_S2(u | O, psi) as a dict."""
        self._sync()
        if (obs, psi) not in self.utterances_obs_psi:
            self._dist_over_utterances_obs_array(obs, psi)
        return self.utterances_obs_psi[(obs, psi)]

    def dist_over_utterances_obs_array(self, obs, psi):
        """P_S2(u | O, psi) as an array aligned with the utterance space."""
        return self._dist_over_utterances_obs_array(obs, psi)

    def dist_over_utterances_theta(self, theta, psi):
        """P(u | theta, psi) marginalizing over observations."""
        self._sync()
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
        self._sync()
        if (theta, psi) not in self._utterance_theta_psi_array:
            self.dist_over_utterances_theta(theta, psi)
        return self._utterance_theta_psi_array[(theta, psi)]

    def obs_utt_table_for_psi(self, psi):
        self._sync()
        if psi not in self._obs_utt_table_psi:
            self._obs_utt_table_psi[psi] = np.vstack(
                [self._dist_over_utterances_obs_array(obs, psi) for obs in self._obs_list]
            )
        return self._obs_utt_table_psi[psi]

    def sample_utterance(self, obs):
        probs = self._dist_over_utterances_obs_array(obs, self.psi)
        return random.choices(self._utterances, weights=probs, k=1)[0]
