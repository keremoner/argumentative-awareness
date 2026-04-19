"""
Sequential detection test with running-average threshold rule.

For any per-round score ``s^(i)`` with null variance ``sigma^2,(i)``, the
test accumulates
    Sus(t)         = (1/t) * sum_{i=1}^t s^(i)
    sigmabar^2(t)  = (1/t) * sum_{i=1}^t sigma^2,(i)
and fires when
    Sus(t) > c * sigmabar(t) / sqrt(t).

The test is a pure observer by default -- switching must be opted into via
``switch_enabled``.  The listener (``DetectionListener``) decides what to
do with the signal.
"""

from __future__ import annotations

import numpy as np


class SequentialTest:
    def __init__(self, score_fn, name, c=2.0, switch_enabled=False, switch_type="soft"):
        """
        Parameters
        ----------
        score_fn : callable(ctx) -> (score, variance)
            Per-round score function, e.g. ``compute_surp2``.
        name : str
            Label for logging (e.g. ``"surp2"``).
        c : float
            z-score cutoff.  ``float('inf')`` disables crossing entirely
            (useful for pure observation / data collection).
        switch_enabled : bool
            If True, this test is authorised to drive a listener switch
            at its stopping time.
        switch_type : str
            ``"hard"`` or ``"soft"`` -- passed to the listener if switching.
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
        self.tau = None  # first crossing time (1-indexed)

        self._score_sum = 0.0
        self._var_sum = 0.0
        self._round = 0

    def observe(self, ctx):
        """Run one round of the test against the supplied ``ScoreContext``.

        Returns
        -------
        crossed : bool
            Whether the running statistic strictly exceeds the threshold
            this round.  ``self.tau`` records the first such round.
        """
        self._round += 1
        t = self._round
        score, var = self.score_fn(ctx)
        self._score_sum += score
        self._var_sum += var

        running_mean = self._score_sum / t
        running_var = self._var_sum / t
        sigma = float(np.sqrt(max(running_var, 0.0)))

        if np.isfinite(self.c):
            threshold = self.c * sigma / float(np.sqrt(t))
        else:
            threshold = float("inf")
        crossed = bool(sigma > 0.0 and running_mean > threshold)

        self.history["scores"].append(float(score))
        self.history["variances"].append(float(var))
        self.history["running_mean"].append(float(running_mean))
        self.history["running_sigma"].append(sigma)
        self.history["threshold"].append(float(threshold))
        self.history["crossed"].append(crossed)

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
