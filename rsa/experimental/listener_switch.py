"""
Listener1Switch, Listener2Switch, and SuspicionSwitchListener.

Hybrid listeners that start as credulous (inf) and switch to vigilant (vig)
when a suspicion criterion is met.

- Listener1Switch / Listener2Switch: switch when a single-utterance suspicion
  score exceeds a fixed threshold.
- SuspicionSwitchListener: switch when the cumulative suspicion score Sus(t)
  exceeds a data-driven threshold c * sigma(t), where sigma(t) is estimated
  from the null variance of the per-round excess-surprise statistic.
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


class SuspicionSwitchListenerS0:
    """
    Suspicion-based switch listener (S0 inversion variant — DEPRECATED).

    Uses S0(u | O) (literal speaker, uniform over true utterances) for the
    observation posterior P(O | u).  This causes E[sus] != 0 under the null
    because the inversion model (S0) does not match the scoring model (S1^inf).

    See ``SuspicionSwitchListener`` for the corrected version that uses
    S1^inf for both inversion and scoring.

    Parameters
    ----------
    thetas, psis, speaker, world, semantics, c, alpha, hard_switch :
        Same as ``SuspicionSwitchListener``.
    """

    def __init__(self, thetas, psis, speaker, world, semantics,
                 c=1.0, alpha=1.0, hard_switch=True):
        self.thetas = thetas
        self.psis = psis
        self.speaker = speaker
        self.world = world
        self.semantics = semantics
        self.c = c
        self.alpha = alpha
        self.hard_switch = hard_switch

        # Switching state
        self.state = "naive"        # "naive" | "vigilant"
        self.switched_at = None     # round at which switch occurred (1-indexed)
        self.round = 0

        # Running statistics for Sus(t) and sigma(t)
        self._sus_sum = 0.0         # cumulative sum of per-round sus scores
        self._var_sum = 0.0         # cumulative sum of per-round variance contributions
        self.Sus_running = 0.0      # Sus(t) = _sus_sum / round
        self.sigma2_running = 0.0   # sigma^2(t) = _var_sum / round

        # Internal listeners (only the active one is updated each round)
        self.inf_listener = Listener1(thetas, psis, speaker, world, semantics,
                                      listener_type="inf", alpha=alpha)
        self.vig_listener = Listener1(thetas, psis, speaker, world, semantics,
                                      listener_type="vig", alpha=alpha)

        # Exposed state_belief mirrors the active listener
        self.state_belief = deepcopy(self.inf_listener.state_belief)

        self.hist = [deepcopy(self.state_belief)]
        self.suspicion = []         # Sus_running value recorded after each round
        self.utt_history = []

        # Precomputed references
        self._obs_list = world.generate_all_obs()
        self._utterances = semantics.utterance_space()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _active(self):
        """Return the currently active internal listener."""
        return self.vig_listener if self.state == "vigilant" else self.inf_listener

    def _invalidate_vig_caches(self):
        """Clear all cached computations on vig_listener after its state_belief
        has been externally modified (hard/soft switch)."""
        vl = self.vig_listener
        vl.obs_psi_utt = {}
        vl.obs_psi = None
        vl.prior_utt = None
        vl.obs_utt = {}
        vl.state_utt = {}
        vl._state_utt_array = {}
        vl._obs_psi_array = None
        vl._obs_psi_utt_array = {}
        vl._prior_utt_array = None

    def _compute_sus_and_var(self, utt):
        """Compute per-round suspicion sus(u, t) and variance contribution.

        Uses L0's current prior (speaker.listener.state_belief) which must be
        the prior *before* the round's belief update -- i.e. this must be called
        before l0.update(utt).

        Returns
        -------
        sus_score : float
            Expected excess surprise of utt under the informative speaker,
            averaged over L0-posterior-weighted consistent observations.
        var_score : float
            Expected null variance of the per-round statistic, used to
            estimate sigma(t).
        """
        utt_idx = self.semantics.utterance_index(utt)
        truth_table = self.semantics.truth_table(self.world)   # n_obs x n_utt (bool)
        consistent_mask = truth_table[:, utt_idx]              # bool, n_obs

        if not consistent_mask.any():
            return 0.0, 0.0

        # ---- Step 2: L0_post(O | u) ----------------------------------------
        # P(O) under L0's current prior over theta
        l0_prior = self.speaker.listener.state_belief.prob     # n_theta
        obs_prob_table = self.world.obs_prob_table(self.thetas)  # n_obs x n_theta
        obs_marginal = obs_prob_table @ l0_prior               # n_obs

        # S0(u | O) = 1 / |{u' : Truth(u', O)}|  for consistent O, else 0
        count_true = truth_table.sum(axis=1).astype(float)    # n_obs
        s0_utt = np.where(consistent_mask,
                          1.0 / np.maximum(count_true, 1.0),
                          0.0)

        weights = s0_utt * obs_marginal
        weight_sum = weights.sum()
        if weight_sum <= 0.0:
            return 0.0, 0.0

        l0_post = weights / weight_sum                         # n_obs

        # ---- Steps 3 & 4: excess surprise and null variance ----------------
        sus_score = 0.0
        var_score = 0.0

        for obs_idx in np.where(consistent_mask)[0]:
            w = l0_post[obs_idx]
            if w == 0.0:
                continue

            obs = self._obs_list[obs_idx]

            # S1(. | O, inf) over the full utterance space (n_utt array)
            s1_dist = self.speaker._dist_over_utterances_obs_array(obs, "inf")

            # Clip for log stability; entropy uses all utterances, not just true
            # ones, per spec ("iterate over full utterance space")
            log_s1 = np.log(np.maximum(s1_dist, 1e-10))
            entropy = -np.dot(s1_dist, log_s1)         # H(S1(.|O,inf))
            neg_log_utt = -log_s1[utt_idx]             # -log S1(u|O,inf)

            # Excess surprise: E_{null}[surprise] = 0 since E_p[-log p] = H(p)
            sus_score += w * (neg_log_utt - entropy)

            # Null variance of -log S1(u | O, inf) under S1(. | O, inf)
            # Var = E[(-log S1)^2] - H^2
            E_sq = np.dot(s1_dist, log_s1 ** 2)
            var_O = E_sq - entropy ** 2
            var_score += w * var_O

        return sus_score, var_score

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, utt):
        """Update listener state after hearing utterance *utt*.

        Must be called before the shared L0 / S1 agents are updated for this
        round (i.e. before l0.update(utt) and s1.update(obs)) so that the
        suspicion score uses L0's prior from *before* the current round.

        Returns the new state_belief from the active listener.
        """
        self.round += 1
        t = self.round

        # Compute per-round suspicion BEFORE any belief update
        sus_t, var_t = self._compute_sus_and_var(utt)

        # Incremental update of running statistics
        self._sus_sum += sus_t
        self._var_sum += var_t
        self.Sus_running = self._sus_sum / t
        self.sigma2_running = self._var_sum / t

        sigma = np.sqrt(max(self.sigma2_running, 0.0))

        # Check switching condition: Sus(t) > c * sigma(t) / sqrt(t)
        # sigma(t) estimates the SD of a single round; Sus(t) is an average of
        # t rounds, so its SD is sigma(t)/sqrt(t).  c acts as a z-score cutoff.
        threshold = self.c * sigma / np.sqrt(t)
        if self.state == "naive" and sigma > 0.0 and self.Sus_running > threshold:
            self.state = "vigilant"
            self.switched_at = t

            if self.hard_switch:
                # Reset vig_listener to a flat uniform prior over all (theta, psi)
                vig_joint = [(theta, psi)
                             for theta in self.thetas for psi in self.psis]
                self.vig_listener.state_belief = Belief(vig_joint)
            else:
                # Soft switch: inherit current theta beliefs, uniform over psi
                theta_probs = self.inf_listener.marginal_theta()
                n_psi = len(self.psis)
                vig_joint = [(theta, psi)
                             for theta in self.thetas for psi in self.psis]
                vig_prior = [theta_probs.get(theta, 0.0) / n_psi
                             for theta, psi in vig_joint]
                self.vig_listener.state_belief = Belief(vig_joint, vig_prior)

            self._invalidate_vig_caches()

        # Update only the active listener's beliefs
        new_belief = self._active().update(utt)

        self.utt_history.append(utt)
        self.suspicion.append(self.Sus_running)
        self.state_belief = new_belief
        self.hist.append(deepcopy(self.state_belief))
        return new_belief

    def infer_state(self, utt):
        return self._active().infer_state(utt)

    def infer_obs(self, utt):
        return self._active().infer_obs(utt)

    def infer_obs_psi(self, utt):
        return self._active().infer_obs_psi(utt)

    def distribution_over_obs_psi(self):
        return self._active().distribution_over_obs_psi()

    def prior_over_utt(self):
        return self._active().prior_over_utt()

    def marginal_theta(self):
        return self._active().marginal_theta()

    def marginal_psi(self):
        return self._active().marginal_psi()

    def false_positive_rates(self, c_values):
        """Per-utterance FPR proxy: probability mass (under informative speaker)
        of utterances whose single-round sus score exceeds c * per-obs-sigma.

        For the true multi-round FPR use ``estimate_fpr`` / ``estimate_tpr``."""
        fprs = {c_val: 0.0 for c_val in c_values}
        for obs in self._obs_list:
            speaker_dist = self.speaker.dist_over_utterances_obs(obs, "inf")
            prob = self.world.obs_prob(obs, self.world.theta)
            for u in self._utterances:
                sus, var = self._compute_sus_and_var(u)
                sigma = np.sqrt(max(var, 0.0))
                for c_val in c_values:
                    if sigma > 0.0 and sus > c_val * sigma:
                        fprs[c_val] += speaker_dist[u] * prob
        return fprs

    def true_positive_rates(self, c_values, pers_ratio=0.5):
        """Per-utterance TPR proxy under persuasive speaker."""
        tprs = {c_val: 0.0 for c_val in c_values}
        for obs in self._obs_list:
            speaker_dist_high = self.speaker.dist_over_utterances_obs(obs, "high")
            speaker_dist_low = self.speaker.dist_over_utterances_obs(obs, "low")
            prob = self.world.obs_prob(obs, self.world.theta)
            for u in self._utterances:
                sus, var = self._compute_sus_and_var(u)
                sigma = np.sqrt(max(var, 0.0))
                for c_val in c_values:
                    if sigma > 0.0 and sus > c_val * sigma:
                        tprs[c_val] += speaker_dist_high[u] * prob * pers_ratio
                        tprs[c_val] += speaker_dist_low[u] * prob * (1 - pers_ratio)
        return tprs


class SuspicionSwitchListener(SuspicionSwitchListenerS0):
    """
    Suspicion-based switch listener (S1^inf-consistent version).

    Uses S1^inf(u | O, inf) — the informative pragmatic speaker — for *both*
    the observation posterior P(O | u) and the excess-surprise scoring.  This
    ensures E[sus] = 0 under the null (when the speaker truly is S1^inf),
    unlike the S0-based variant where the inversion/scoring mismatch causes
    a systematic positive drift.

    Per-round suspicion score:

        P^(i)(O | u) proportional to S1^(i)(u | O, inf) * P_L0^(i)(O)

        sus(u, i) = sum_O P^(i)(O | u) * [-log S1^(i)(u | O, inf) - H(S1^(i)(. | O, inf))]

    Everything else — L0 prior updates, cumulative Sus(t), sigma(t), switch
    mechanism — is identical to the base class.

    Parameters
    ----------
    thetas, psis, speaker, world, semantics, c, alpha, hard_switch :
        Same as ``SuspicionSwitchListenerS0``.
    """

    def _compute_sus_and_var(self, utt):
        """Compute per-round suspicion sus(u, t) and variance contribution.

        Uses S1^inf(u | O, inf) for the observation posterior (key difference
        from the S0 variant).  L0's current prior must still be the prior
        *before* the round's belief update.

        Returns
        -------
        sus_score : float
        var_score : float
        """
        utt_idx = self.semantics.utterance_index(utt)
        truth_table = self.semantics.truth_table(self.world)   # n_obs x n_utt (bool)
        consistent_mask = truth_table[:, utt_idx]              # bool, n_obs

        if not consistent_mask.any():
            return 0.0, 0.0

        # P_L0(O) under L0's current prior over theta
        l0_prior = self.speaker.listener.state_belief.prob     # n_theta
        obs_prob_table = self.world.obs_prob_table(self.thetas)  # n_obs x n_theta
        obs_marginal = obs_prob_table @ l0_prior               # n_obs

        # Precompute S1^inf distribution for each consistent O
        consistent_indices = np.where(consistent_mask)[0]
        n_utt = len(self._utterances)
        s1_dists = {}            # obs_idx -> n_utt array
        s1_utt_given_obs = np.zeros(len(self._obs_list))

        for obs_idx in consistent_indices:
            obs = self._obs_list[obs_idx]
            s1_dist = self.speaker._dist_over_utterances_obs_array(obs, "inf")
            s1_dists[obs_idx] = s1_dist
            s1_utt_given_obs[obs_idx] = s1_dist[utt_idx]

        # S1^inf-based observation posterior (the key change):
        #   P^(i)(O | u) proportional to S1^inf(u | O, inf) * P_L0(O)
        weights = s1_utt_given_obs * obs_marginal
        weight_sum = weights.sum()
        if weight_sum <= 0.0:
            return 0.0, 0.0

        obs_post = weights / weight_sum                        # n_obs

        # Excess surprise and null variance
        sus_score = 0.0
        var_score = 0.0

        for obs_idx in consistent_indices:
            w = obs_post[obs_idx]
            if w == 0.0:
                continue

            s1_dist = s1_dists[obs_idx]

            log_s1 = np.log(np.maximum(s1_dist, 1e-10))
            entropy = -np.dot(s1_dist, log_s1)         # H(S1(.|O,inf))
            neg_log_utt = -log_s1[utt_idx]             # -log S1(u|O,inf)

            sus_score += w * (neg_log_utt - entropy)

            # Var[-log S1(u | O, inf)] under S1(. | O, inf)
            E_sq = np.dot(s1_dist, log_s1 ** 2)
            var_O = E_sq - entropy ** 2
            var_score += w * var_O

        return sus_score, var_score


# ---------------------------------------------------------------------------
# Simulation-based FPR / TPR analysis for SuspicionSwitchListener
# ---------------------------------------------------------------------------

def simulate_suspicion_listener(
    thetas, psis, semantics, world, speaker_type="inf",
    c=1.0, alpha=1.0, rounds=20, n_sims=100, hard_switch=True
):
    """Run *n_sims* independent simulations of the SuspicionSwitchListener game.

    Each simulation creates fresh agents so there is no state contamination
    across trials.  The world's fixed ``theta`` is preserved across simulations.

    Parameters
    ----------
    thetas, psis, semantics, world : standard game objects
    speaker_type : "inf" | "high" | "low" -- true speaker type for the simulation
    c : threshold parameter
    alpha : rationality
    rounds : number of communication rounds per simulation
    n_sims : number of independent Monte-Carlo simulations
    hard_switch : forwarded to SuspicionSwitchListener

    Returns
    -------
    results : list of dicts, one per simulation, each containing:
        ``switched``      : bool -- did the listener ever switch?
        ``switched_at``   : int or None -- round of first switch (1-indexed)
        ``Sus_history``   : list of Sus(t) values per round
        ``state_history`` : list of "naive"/"vigilant" strings per round
        ``theta_history`` : list of marginal_theta() dicts per round
    """
    from ..speaker0 import Speaker0
    from ..listener0 import Listener0
    from ..speaker1 import Speaker1

    results = []

    for _ in range(n_sims):
        s0 = Speaker0(thetas, semantics=semantics, world=world)
        l0 = Listener0(thetas, s0, semantics=semantics, world=world)
        s1 = Speaker1(thetas, l0, semantics=semantics, world=world,
                      alpha=alpha, psi=speaker_type)
        listener = SuspicionSwitchListener(
            thetas, psis, s1, world, semantics,
            c=c, alpha=alpha, hard_switch=hard_switch
        )

        Sus_history = []
        state_history = []
        theta_history = []

        for _ in range(rounds):
            obs = world.sample_obs()
            utt = s1.sample_utterance(obs)

            # Listener updates BEFORE s1/l0 so it sees L0's pre-round prior
            listener.update(utt)
            s1.update(obs)
            l0.update(utt)
            s0.update(obs)

            Sus_history.append(listener.Sus_running)
            state_history.append(listener.state)
            theta_history.append(listener.marginal_theta())

        results.append({
            "switched": listener.state == "vigilant",
            "switched_at": listener.switched_at,
            "Sus_history": Sus_history,
            "state_history": state_history,
            "theta_history": theta_history,
        })

    return results


def estimate_fpr(thetas, psis, semantics, world, c=1.0, alpha=1.0,
                 rounds=20, n_sims=200, hard_switch=True):
    """Estimate FPR(c, t) = P(switched by round t | psi=inf) by simulation.

    Returns
    -------
    fpr_by_round : np.ndarray of shape (rounds,)
        Fraction of simulations that have switched by each round.
    """
    results = simulate_suspicion_listener(
        thetas, psis, semantics, world, speaker_type="inf",
        c=c, alpha=alpha, rounds=rounds, n_sims=n_sims, hard_switch=hard_switch
    )
    switched_by_round = np.zeros(rounds)
    for res in results:
        if res["switched_at"] is not None:
            switched_by_round[res["switched_at"] - 1:] += 1
    return switched_by_round / n_sims


def estimate_tpr(thetas, psis, semantics, world, speaker_type="high",
                 c=1.0, alpha=1.0, rounds=20, n_sims=200, hard_switch=True):
    """Estimate TPR(c, t) = P(switched by round t | psi=persuasive) by simulation.

    Returns
    -------
    tpr_by_round : np.ndarray of shape (rounds,)
        Fraction of simulations that have switched by each round.
    """
    results = simulate_suspicion_listener(
        thetas, psis, semantics, world, speaker_type=speaker_type,
        c=c, alpha=alpha, rounds=rounds, n_sims=n_sims, hard_switch=hard_switch
    )
    switched_by_round = np.zeros(rounds)
    for res in results:
        if res["switched_at"] is not None:
            switched_by_round[res["switched_at"] - 1:] += 1
    return switched_by_round / n_sims
