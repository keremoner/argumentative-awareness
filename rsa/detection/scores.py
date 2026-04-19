"""
Per-round detection scores for the three-score framework.

All scores are computed from the perspective of a credulous listener L1^inf
holding a belief L1^(t)(theta) over the world parameter.  The three scores
differ only in (i) which distribution over observations O is used (prior
vs. posterior) and (ii) whether the log is taken inside or outside the
marginalization over O.

Score surfaces and the null variances used by the sequential test follow
``detection_implementation_spec.md``.  Numerical conventions:
``0 * log 0 = 0`` via a lower clamp on log arguments (``EPS``).  All
scores are finite on the simplex.
"""

from __future__ import annotations

import numpy as np


EPS = 1e-12


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(x, EPS))


def _entropy(p: np.ndarray) -> float:
    return float(-np.dot(p, _safe_log(p)))


def _varentropy(p: np.ndarray) -> float:
    log_p = _safe_log(p)
    H = -np.dot(p, log_p)
    E_sq = np.dot(p, log_p ** 2)
    return float(max(E_sq - H ** 2, 0.0))


class ScoreContext:
    """Lazy bundle of per-round quantities shared across the three scores.

    All scores at a given round depend on the same four objects:
      - ``L1_theta``:   L1^(t)(theta)              (n_theta,)
      - ``L1_O``:       L1^(t)(O)                   (n_obs,)
      - ``S1_table``:   S1^(t)(u | O, inf)          (n_obs, n_utt)
      - ``u_obs_idx``:  index of the observed utterance
    and the derivatives ``L1_O_given_u``, ``P_prior``, ``P_post``.

    Build once per round and pass to each score function -- this guarantees
    the three scores share a consistent snapshot of the listener state.
    """

    def __init__(self, thetas, speaker, world, semantics, l1_theta, u_obs):
        self.thetas = tuple(thetas)
        self.speaker = speaker
        self.world = world
        self.semantics = semantics
        self.u_obs = u_obs
        self.u_obs_idx = semantics.utterance_index(u_obs)

        # Listener's theta marginal (normalized; length = n_theta)
        self.L1_theta = np.asarray(l1_theta, dtype=float)

        self._L1_O = None
        self._S1_table = None
        self._log_S1_table = None
        self._P_prior = None
        self._log_P_prior = None
        self._L1_O_given_u = None
        self._P_post = None
        self._log_P_post = None
        self._H_per_O = None

    @property
    def L1_O(self) -> np.ndarray:
        if self._L1_O is None:
            obs_prob_table = self.world.obs_prob_table(self.thetas)  # n_obs x n_theta
            p = obs_prob_table @ self.L1_theta
            s = p.sum()
            self._L1_O = p / s if s > 0 else p
        return self._L1_O

    @property
    def S1_table(self) -> np.ndarray:
        if self._S1_table is None:
            self._S1_table = np.asarray(
                self.speaker.obs_utt_table_for_psi("inf"), dtype=float
            )
        return self._S1_table

    @property
    def log_S1_table(self) -> np.ndarray:
        if self._log_S1_table is None:
            self._log_S1_table = _safe_log(self.S1_table)
        return self._log_S1_table

    @property
    def H_per_O(self) -> np.ndarray:
        """H(S1(. | O, inf)) for each O."""
        if self._H_per_O is None:
            self._H_per_O = -np.sum(self.S1_table * self.log_S1_table, axis=1)
        return self._H_per_O

    @property
    def P_prior(self) -> np.ndarray:
        """P_prior(u) = sum_O S1(u | O, inf) * L1(O)."""
        if self._P_prior is None:
            p = self.S1_table.T @ self.L1_O
            s = p.sum()
            self._P_prior = p / s if s > 0 else p
        return self._P_prior

    @property
    def log_P_prior(self) -> np.ndarray:
        if self._log_P_prior is None:
            self._log_P_prior = _safe_log(self.P_prior)
        return self._log_P_prior

    @property
    def L1_O_given_u(self) -> np.ndarray:
        """L1(O | u_obs) proportional to S1(u_obs | O, inf) * L1(O)."""
        if self._L1_O_given_u is None:
            s1_col = self.S1_table[:, self.u_obs_idx]
            w = s1_col * self.L1_O
            s = w.sum()
            self._L1_O_given_u = w / s if s > 0 else np.zeros_like(w)
        return self._L1_O_given_u

    @property
    def P_post(self) -> np.ndarray:
        """P_post(u | u_obs) = sum_O S1(u | O, inf) * L1(O | u_obs)."""
        if self._P_post is None:
            p = self.S1_table.T @ self.L1_O_given_u
            s = p.sum()
            self._P_post = p / s if s > 0 else p
        return self._P_post

    @property
    def log_P_post(self) -> np.ndarray:
        if self._log_P_post is None:
            self._log_P_post = _safe_log(self.P_post)
        return self._log_P_post


def compute_surp2(ctx: ScoreContext):
    """Prior predictive surprisal (Eq. 5 of three_scores.pdf).

        surp2(u) = -log P_prior(u) - H(P_prior)
        sigma^2  = Var_{u ~ P_prior}[-log P_prior(u)]   (varentropy)

    Under the null, E[surp2] = 0 by definition of entropy.
    """
    P_prior = ctx.P_prior
    log_P = ctx.log_P_prior
    H = float(-np.dot(P_prior, log_P))
    score = float(-log_P[ctx.u_obs_idx] - H)
    var = float(np.dot(P_prior, log_P ** 2) - H ** 2)
    return score, max(var, 0.0)


def compute_surp1(ctx: ScoreContext):
    """Posterior predictive surprisal (Eq. 6 of three_scores.pdf).

        surp1(u) = -log P_post(u | u) - H(P_post(. | u))
        sigma^2  = Var_{u' ~ P_post(.|u)}[-log P_post(u' | u)]

    Note ``u_obs`` appears twice: once as the scored utterance, once inside
    the conditioning that produced the posterior predictive.
    """
    P_post = ctx.P_post
    log_P = ctx.log_P_post
    H = float(-np.dot(P_post, log_P))
    score = float(-log_P[ctx.u_obs_idx] - H)
    var = float(np.dot(P_post, log_P ** 2) - H ** 2)
    return score, max(var, 0.0)


def compute_sus(ctx: ScoreContext):
    """Observation-level suspicion (Eq. 7 of three_scores.pdf).

        sus(u) = sum_O L1(O|u) [ -log S1(u|O,inf) - H(S1(.|O,inf)) ]
        sigma^2 = sum_O L1(O|u) * Var_{u' ~ S1(.|O,inf)}[-log S1(u'|O,inf)]

    Log is taken inside the marginalization over O.  Under the null
    (u | O ~ S1(. | O, inf)) the per-O excess surprise has mean zero, so
    E[sus] = 0.
    """
    L1_O_post = ctx.L1_O_given_u
    S1_table = ctx.S1_table
    log_S1 = ctx.log_S1_table
    u_idx = ctx.u_obs_idx

    neg_log_u = -log_S1[:, u_idx]                  # n_obs
    H_per_O = ctx.H_per_O                          # n_obs
    per_O_score = neg_log_u - H_per_O              # n_obs
    score = float(np.dot(L1_O_post, per_O_score))

    E_sq_per_O = np.sum(S1_table * log_S1 ** 2, axis=1)
    varentropy_per_O = np.maximum(E_sq_per_O - H_per_O ** 2, 0.0)
    var = float(np.dot(L1_O_post, varentropy_per_O))
    return score, max(var, 0.0)


SCORE_FNS = {
    "surp2": compute_surp2,
    "surp1": compute_surp1,
    "sus": compute_sus,
}
