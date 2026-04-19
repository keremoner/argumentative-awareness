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

    def __init__(self, thetas, speaker, world, semantics, l1_theta, u_obs,
                 l0_theta=None):
        self.thetas = tuple(thetas)
        self.speaker = speaker
        self.world = world
        self.semantics = semantics
        self.u_obs = u_obs
        self.u_obs_idx = semantics.utterance_index(u_obs)

        # Listener's theta marginals (normalized; length = n_theta).
        # ``l0_theta`` is optional so the three core scores (surp2, surp1,
        # sus) continue to work without it; only the sus-variant scores
        # that need L_0 will require it.
        self.L1_theta = np.asarray(l1_theta, dtype=float)
        self.L0_theta = (np.asarray(l0_theta, dtype=float)
                         if l0_theta is not None else None)

        self._L1_O = None
        self._L0_O = None
        self._S1_table = None
        self._log_S1_table = None
        self._P_prior = None
        self._log_P_prior = None
        self._L1_O_given_u = None
        self._P_post = None
        self._log_P_post = None
        self._H_per_O = None
        self._varentropy_per_O = None
        self._truth_table = None
        self._T_O = None
        self._B_matrix = None

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

    @property
    def L0_O(self) -> np.ndarray:
        """L0^(t)(O) = sum_theta P(O | theta) * L0^(t)(theta)."""
        if self._L0_O is None:
            if self.L0_theta is None:
                raise RuntimeError(
                    "ScoreContext was built without L0_theta; variants that "
                    "require L_0 cannot be computed."
                )
            obs_prob_table = self.world.obs_prob_table(self.thetas)
            p = obs_prob_table @ self.L0_theta
            s = p.sum()
            self._L0_O = p / s if s > 0 else p
        return self._L0_O

    @property
    def truth_table(self) -> np.ndarray:
        """Truth matrix, shape (n_obs, n_utt), bool."""
        if self._truth_table is None:
            self._truth_table = self.semantics.truth_table(self.world)
        return self._truth_table

    @property
    def T_O(self) -> np.ndarray:
        """T(O) = number of utterances truthful at O. Shape (n_obs,)."""
        if self._T_O is None:
            self._T_O = self.truth_table.sum(axis=1).astype(float)
        return self._T_O

    @property
    def varentropy_per_O(self) -> np.ndarray:
        """Var_{u' ~ S1(.|O,inf)}[-log S1(u' | O, inf)] for each O."""
        if self._varentropy_per_O is None:
            S1 = self.S1_table
            logS1 = self.log_S1_table
            E_sq = np.sum(S1 * logS1 ** 2, axis=1)
            self._varentropy_per_O = np.maximum(E_sq - self.H_per_O ** 2, 0.0)
        return self._varentropy_per_O

    @property
    def B_matrix(self) -> np.ndarray:
        """B[O, u] = -log S1(u|O,inf) - H(S1(.|O,inf)). Shape (n_obs, n_utt)."""
        if self._B_matrix is None:
            self._B_matrix = -self.log_S1_table - self.H_per_O[:, None]
        return self._B_matrix


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


# ---------------------------------------------------------------------------
# sus variants (see sus_variants_spec for the math)
# ---------------------------------------------------------------------------

SUS_VARIANTS = ("1", "3", "4", "4b")


def _weighting_matrix(variant: str, ctx: "ScoreContext") -> np.ndarray:
    """W_v[O, u], columns normalized. Zero columns left as zero."""
    n_obs, n_utt = ctx.S1_table.shape
    if variant == "1":
        num = ctx.S1_table * ctx.L1_O[:, None]
    elif variant == "3":
        num = ctx.truth_table.astype(float) * ctx.L1_O[:, None]
    elif variant == "4":
        num = ctx.truth_table.astype(float) * ctx.L0_O[:, None]
    elif variant == "4b":
        inv_T = np.where(ctx.T_O > 0, 1.0 / np.maximum(ctx.T_O, 1.0), 0.0)
        num = ctx.truth_table.astype(float) * ctx.L0_O[:, None] * inv_T[:, None]
    else:
        raise ValueError(f"unknown sus variant {variant!r}")
    col_sums = num.sum(axis=0, keepdims=True)
    W = np.divide(num, col_sums, out=np.zeros_like(num),
                  where=col_sums > 0)
    return W


def _p_pred(variant: str, ctx: "ScoreContext") -> np.ndarray:
    """P_pred_v(u') over the utterance space, shape (n_utt,)."""
    if variant in ("1", "3"):
        # sum_O S1(u'|O,inf) * L1(O)
        p = ctx.S1_table.T @ ctx.L1_O
    elif variant == "4":
        # sum_O S1(u'|O,inf) * L0(O)
        p = ctx.S1_table.T @ ctx.L0_O
    elif variant == "4b":
        # sum_O S1(u'|O,inf) * L0(O)/T(O), renormalized
        inv_T = np.where(ctx.T_O > 0, 1.0 / np.maximum(ctx.T_O, 1.0), 0.0)
        weighted = ctx.L0_O * inv_T
        p = ctx.S1_table.T @ weighted
    else:
        raise ValueError(f"unknown sus variant {variant!r}")
    s = p.sum()
    return p / s if s > 0 else p


def make_sus_variant(variant: str):
    """Return a score function ``f(ctx) -> (score, var_naive, var_corrected)``
    for sus variant ``v in {"1","3","4","4b"}``.
    """
    if variant not in SUS_VARIANTS:
        raise ValueError(f"unknown sus variant {variant!r}")

    def f(ctx: "ScoreContext"):
        u_idx = ctx.u_obs_idx
        W = _weighting_matrix(variant, ctx)        # (n_obs, n_utt)
        B = ctx.B_matrix                            # (n_obs, n_utt)
        varent = ctx.varentropy_per_O               # (n_obs,)
        w_u = W[:, u_idx]                           # (n_obs,)

        score = float(np.dot(w_u, B[:, u_idx]))
        var_naive = float(np.dot(w_u, varent))

        # Corrected variance: naive - within-posterior correction.
        mean_B = np.sum(W * B, axis=0)              # (n_utt,)
        E_B_sq = np.sum(W * B ** 2, axis=0)         # (n_utt,)
        var_O_given_u = np.maximum(E_B_sq - mean_B ** 2, 0.0)  # (n_utt,)

        p_pred = _p_pred(variant, ctx)              # (n_utt,)
        correction = float(np.dot(p_pred, var_O_given_u))
        var_corrected = float(max(var_naive - correction, 0.0))

        return score, max(var_naive, 0.0), var_corrected

    f.__name__ = f"compute_sus_{variant}"
    f.__doc__ = (
        f"sus variant {variant}. Returns (score, var_naive, var_corrected); "
        f"var_corrected uses the law-of-total-variance correction that "
        f"removes the within-posterior variance of B(u', O') over W_v(. | u')."
    )
    return f


SUS_VARIANT_FNS = {v: make_sus_variant(v) for v in SUS_VARIANTS}
