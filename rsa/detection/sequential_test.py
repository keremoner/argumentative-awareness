"""
Sequential detection test with running-average threshold rule.

For any per-round score ``s^(i)`` with null variance ``sigma^2,(i)``, the
test accumulates
    Sus(t)         = (1/t) * sum_{i=1}^t s^(i)
    sigmabar^2(t)  = (1/t) * sum_{i=1}^t sigma^2,(i)
and fires when
    Sus(t) > c * sigmabar(t) / sqrt(t).

Score functions return ``(score, var)``.  ``var`` is the exact variance of the
score under the listener's own predictive distribution and is non-negative by
construction, so nothing is clipped here -- a negative variance is a bug in the
score function and is raised rather than silently absorbed.

The test is a pure observer by default -- switching must be opted into via
``switch_enabled``.  The listener (``DetectionListener``) decides what to do
with the signal.
"""

from __future__ import annotations

import numpy as np


def _unpack_score(out):
    if len(out) == 2:
        score, var = out
        return float(score), float(var)
    raise ValueError(
        f"score_fn returned {len(out)} values; expected 2. The naive/corrected "
        f"variance pair was removed when sus_1 moved to its exact state-only "
        f"variance -- see docs/project.typ, section on the sus_1 variance."
    )


class SequentialTest:
    def __init__(self, score_fn, name, c=2.0, switch_enabled=False, switch_type="soft"):
        """
        Parameters
        ----------
        score_fn : callable(ctx) -> (score, variance)
        name : str
        c : float
            z-score cutoff.  ``float('inf')`` disables crossing entirely.
        switch_enabled : bool
            If True, this test is authorised to drive a listener switch at
            its stopping time.
        switch_type : str
            ``"hard"`` or ``"soft"``.
        """
        self.score_fn = score_fn
        self.name = name
        self.c = c
        self.switch_enabled = switch_enabled
        self.switch_type = switch_type

        self.history = {
            "scores": [],
            "variances": [],
            "running_mean": [],
            "running_sigma": [],
            "threshold": [],
            "crossed": [],
        }
        self.tau = None

        self._score_sum = 0.0
        self._var_sum = 0.0
        self._round = 0

    def observe(self, ctx):
        self._round += 1
        t = self._round
        score, var = _unpack_score(self.score_fn(ctx))

        if not (var >= 0.0):
            raise ValueError(
                f"score_fn {self.name!r} returned a negative or NaN variance "
                f"({var!r}) at round {t}. The exact score variance is a sum of "
                f"squares weighted by a probability vector and cannot be "
                f"negative; this indicates a bug in the score function."
            )

        self._score_sum += score
        self._var_sum += var

        running_mean = self._score_sum / t
        sigma = float(np.sqrt(self._var_sum / t))

        thr = self.c * sigma / float(np.sqrt(t)) if np.isfinite(self.c) else float("inf")
        crossed = bool(sigma > 0.0 and running_mean > thr)

        h = self.history
        h["scores"].append(float(score))
        h["variances"].append(float(var))
        h["running_mean"].append(float(running_mean))
        h["running_sigma"].append(sigma)
        h["threshold"].append(float(thr))
        h["crossed"].append(crossed)

        if self.tau is None and crossed:
            self.tau = t
        return crossed

    def reset(self):
        self.history = {k: [] for k in self.history}
        self.tau = None
        self._score_sum = 0.0
        self._var_sum = 0.0
        self._round = 0

    @property
    def running_mean(self):
        return self.history["running_mean"][-1] if self.history["running_mean"] else 0.0

    @property
    def running_sigma(self):
        return self.history["running_sigma"][-1] if self.history["running_sigma"] else 0.0
