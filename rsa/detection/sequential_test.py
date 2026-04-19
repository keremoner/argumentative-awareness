"""
Sequential detection test with running-average threshold rule.

For any per-round score ``s^(i)`` with null variance ``sigma^2,(i)``, the
test accumulates
    Sus(t)         = (1/t) * sum_{i=1}^t s^(i)
    sigmabar^2(t)  = (1/t) * sum_{i=1}^t sigma^2,(i)
and fires when
    Sus(t) > c * sigmabar(t) / sqrt(t).

Score functions may return either ``(score, var)`` -- in which case the
naive and corrected variances are both set to ``var`` -- or
``(score, var_naive, var_corrected)``.  The test tracks BOTH variances in
parallel: two running sigmas, two thresholds, two crossing indicators, two
first-crossing times.  ``observe()`` returns the naive crossing flag (the
historical behaviour used by switch-driving tests).

The test is a pure observer by default -- switching must be opted into via
``switch_enabled``.  The listener (``DetectionListener``) decides what to
do with the signal.
"""

from __future__ import annotations

import numpy as np


def _unpack_score(out):
    if len(out) == 2:
        score, var = out
        return float(score), float(var), float(var)
    if len(out) == 3:
        score, var_n, var_c = out
        return float(score), float(var_n), float(var_c)
    raise ValueError(
        f"score_fn returned {len(out)} values; expected 2 or 3"
    )


class SequentialTest:
    def __init__(self, score_fn, name, c=2.0, switch_enabled=False, switch_type="soft"):
        """
        Parameters
        ----------
        score_fn : callable(ctx) -> (score, variance) or (score, var_naive, var_corrected)
        name : str
        c : float
            z-score cutoff.  ``float('inf')`` disables crossing entirely.
        switch_enabled : bool
            If True, this test is authorised to drive a listener switch at
            its (naive) stopping time.
        switch_type : str
            ``"hard"`` or ``"soft"``.
        """
        self.score_fn = score_fn
        self.name = name
        self.c = c
        self.switch_enabled = switch_enabled
        self.switch_type = switch_type

        # Legacy keys (``variances`` / ``running_sigma`` / ``threshold`` /
        # ``crossed``) continue to reflect the naive variance, so existing
        # analysis code keeps working.
        self.history = {
            "scores": [],
            "variances": [],
            "variances_corrected": [],
            "running_mean": [],
            "running_sigma": [],
            "running_sigma_corrected": [],
            "threshold": [],
            "threshold_corrected": [],
            "crossed": [],
            "crossed_corrected": [],
        }
        self.tau = None             # legacy: naive first crossing
        self.tau_naive = None
        self.tau_corrected = None

        self._score_sum = 0.0
        self._var_sum_naive = 0.0
        self._var_sum_corrected = 0.0
        self._round = 0

    def observe(self, ctx):
        self._round += 1
        t = self._round
        score, var_n, var_c = _unpack_score(self.score_fn(ctx))

        self._score_sum += score
        self._var_sum_naive += var_n
        self._var_sum_corrected += var_c

        running_mean = self._score_sum / t
        sigma_n = float(np.sqrt(max(self._var_sum_naive / t, 0.0)))
        sigma_c = float(np.sqrt(max(self._var_sum_corrected / t, 0.0)))

        if np.isfinite(self.c):
            thr_n = self.c * sigma_n / float(np.sqrt(t))
            thr_c = self.c * sigma_c / float(np.sqrt(t))
        else:
            thr_n = thr_c = float("inf")

        crossed_n = bool(sigma_n > 0.0 and running_mean > thr_n)
        crossed_c = bool(sigma_c > 0.0 and running_mean > thr_c)

        h = self.history
        h["scores"].append(float(score))
        h["variances"].append(float(var_n))
        h["variances_corrected"].append(float(var_c))
        h["running_mean"].append(float(running_mean))
        h["running_sigma"].append(sigma_n)
        h["running_sigma_corrected"].append(sigma_c)
        h["threshold"].append(float(thr_n))
        h["threshold_corrected"].append(float(thr_c))
        h["crossed"].append(crossed_n)
        h["crossed_corrected"].append(crossed_c)

        if self.tau_naive is None and crossed_n:
            self.tau_naive = t
            self.tau = t
        if self.tau_corrected is None and crossed_c:
            self.tau_corrected = t
        return crossed_n

    def reset(self):
        self.history = {k: [] for k in self.history}
        self.tau = None
        self.tau_naive = None
        self.tau_corrected = None
        self._score_sum = 0.0
        self._var_sum_naive = 0.0
        self._var_sum_corrected = 0.0
        self._round = 0

    @property
    def running_mean(self):
        return self.history["running_mean"][-1] if self.history["running_mean"] else 0.0

    @property
    def running_sigma(self):
        return self.history["running_sigma"][-1] if self.history["running_sigma"] else 0.0
