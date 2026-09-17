"""
Listener wrapper that runs any number of SequentialTest instances in parallel
against an internal credulous L1^inf belief.

By default every attached test is a passive observer -- no switching, no
coupling between tests.  Enabling ``switch_enabled`` on one test authorises
it to drive a switch to a full vigilant L1 at its stopping time tau.

Switch types
------------
``"hard"`` (retrospective)
    A fresh vigilant L1 with a uniform prior over (theta, psi) is updated on the
    *whole* utterance history ``u_1 .. u_tau`` (inclusive of the triggering
    utterance) and then continues live from tau + 1.  The replay at historical
    round i uses the speaker tables *as they were at round i* -- the shared
    speaker object only holds its current tables, so ``update`` snapshots the
    ``{psi: (n_obs, n_utt)}`` triple every round into ``table_history``.
    After the switch the listener's belief equals an always-vigilant L1 run
    on the same utterances.

``"soft"``
    The vigilant L1 inherits the credulous theta-marginal (which has already
    absorbed u_tau credulously) spread uniformly over psi, i.e. it keeps the
    pre-tau credulous belief and is direction-blind at tau.  It is then updated
    on u_tau with the vigilant likelihood, so the triggering utterance is seen
    twice: once credulously (inside the inherited marginal) and once
    vigilantly.  Documented, deliberate; the credulous absorption is left as is.

``"hard_amnesic"``
    The pre-Task-1 ``"hard"`` behaviour, kept only as a contrast condition: a
    fresh uniform vigilant L1 with no history, which never sees u_tau and
    starts learning at tau + 1.
"""

from __future__ import annotations

from copy import deepcopy
import numpy as np

from ..listener1 import Listener1
from .scores import ScoreContext


SWITCH_TYPES = ("hard", "soft", "hard_amnesic")


class DetectionListener:
    def __init__(self, thetas, psis, speaker, world, semantics,
                 tests=None, alpha=1.0, retro_cache=False):
        """
        Parameters
        ----------
        thetas : sequence of float
            Candidate theta values.  Must match the speaker's ``thetas``.
        psis : sequence of str
            Full psi hypothesis space used after switching, e.g.
            ``["inf", "high", "low"]``.  The naive listener always runs
            with ``["inf"]``.
        speaker : Speaker1-like
            The listener's internal speaker model (the S1 whose ``inf`` tables
            define the null).  Anything exposing ``obs_utt_table_for_psi`` and
            ``dist_over_utterances_theta_array`` works, e.g. a replay stub.
        tests : list of SequentialTest, optional
            Zero or more detection tests.  Passive unless ``switch_enabled``.
        retro_cache : bool
            If True, ``peek`` caches the retrospective vigilant belief on the
            current ``utt_history`` once per round (it is the same for every
            candidate utterance; only the final update differs).  Off by
            default; results are identical either way (tested).
        """
        self.thetas = list(thetas)
        self.psis = list(psis)
        self.speaker = speaker
        self.world = world
        self.semantics = semantics
        self.alpha = alpha
        self.tests = list(tests) if tests is not None else []
        self.retro_cache = bool(retro_cache)

        self._theta_to_index = {t: i for i, t in enumerate(self.thetas)}

        self.naive = Listener1(
            self.thetas, ["inf"], speaker, world, semantics,
            listener_type="inf", alpha=alpha,
        )
        self.vigilant = None
        self.switched = False
        self.switched_at = None
        self.switch_driver = None  # which test triggered the switch
        self.switch_type = None
        self.round = 0

        self.utt_history = []
        self.table_history = []    # per round: {psi: (n_obs, n_utt) array}
        self.hist = [deepcopy(self.naive.state_belief)]
        self._retro_cached = None  # (n_replayed, Listener1) when retro_cache

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    @property
    def active(self):
        """The sub-listener currently holding the belief."""
        return self.vigilant if self.switched else self.naive

    @property
    def state_belief(self):
        return self.active.state_belief

    def marginal_theta(self):
        return self.active.marginal_theta()

    def marginal_psi(self):
        return self.active.marginal_psi()

    def theta_array(self):
        return self.active.theta_array()

    def infer_obs(self, utt):
        """q_t(O | u) from the currently active sub-listener (no mutation)."""
        return self.active.infer_obs(utt)

    def _l1_theta_array(self):
        """L1^(t)(theta) as an ndarray aligned with ``self.thetas``."""
        return self.naive.theta_array()

    def build_context(self, u_obs):
        """Construct a ScoreContext from the *current* naive listener state.

        Must be called before the round's belief update.
        """
        return ScoreContext(
            self.thetas, self.speaker, self.world, self.semantics,
            self._l1_theta_array(), u_obs,
        )

    def snapshot_tables(self):
        """Copy of the speaker's current ``{psi: P(u | O, psi)}`` tables."""
        return {
            psi: np.array(self.speaker.obs_utt_table_for_psi(psi), dtype=float, copy=True)
            for psi in self.psis
        }

    def _fresh_vigilant(self):
        return Listener1(
            self.thetas, self.psis, self.speaker, self.world, self.semantics,
            listener_type="vig", alpha=self.alpha,
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, utt):
        """Process one utterance: run tests, update beliefs, maybe switch."""
        self.round += 1
        t = self.round

        # Snapshot the tables that generate/explain u_t *before* anything
        # moves.  The history is what a retrospective switch replays.
        tables = self.snapshot_tables()
        self.utt_history.append(utt)
        self.table_history.append(tables)

        if not self.switched:
            ctx = self.build_context(utt)
            for test in self.tests:
                crossed = test.observe(ctx)
                if (crossed and test.switch_enabled
                        and self.switch_driver is None):
                    self.switch_driver = test

            self.naive.update(utt)

            if self.switch_driver is not None and self.switch_driver.tau == t:
                self._trigger_switch(self.switch_driver.switch_type)
        else:
            self.vigilant.update_with_tables(utt, tables)

        self.hist.append(deepcopy(self.state_belief))
        return self.state_belief

    def _trigger_switch(self, switch_type):
        if switch_type not in SWITCH_TYPES:
            raise ValueError(f"unknown switch_type {switch_type!r}; expected one of {SWITCH_TYPES}")
        self.switched = True
        self.switched_at = self.round
        self.switch_type = switch_type

        vig = self._fresh_vigilant()
        if switch_type == "hard":
            # Retrospective: replay u_1..u_tau with the per-round snapshots.
            for u, tab in zip(self.utt_history, self.table_history):
                vig.update_with_tables(u, tab)
        elif switch_type == "soft":
            # Inherit the credulous theta-marginal (already includes u_tau,
            # credulously), uniform over psi, then see u_tau vigilantly.
            vig.seed_from_theta_marginal(self.naive.marginal_theta())
            vig.update_with_tables(self.utt_history[-1], self.table_history[-1])
        elif switch_type == "hard_amnesic":
            pass  # uniform joint prior, no history, never sees u_tau
        self.vigilant = vig
        self._retro_cached = None

    # ------------------------------------------------------------------
    # One-step peek (no mutation)
    # ------------------------------------------------------------------

    def _would_switch(self, utt):
        """The switch-enabled test that would fire if ``utt`` arrived now, or None."""
        if self.switched:
            return None
        t = self.round + 1
        ctx = None
        for test in self.tests:
            if not test.switch_enabled:
                continue
            if ctx is None:
                ctx = self.build_context(utt)
            out = test.score_fn(ctx)
            score, var = float(out[0]), float(out[1])
            running_mean = (test._score_sum + score) / t
            sigma = float(np.sqrt((test._var_sum + var) / t))
            thr = (test.c * sigma / float(np.sqrt(t))
                   if np.isfinite(test.c) else float("inf"))
            if sigma > 0.0 and running_mean > thr:
                return test
        return None

    def _retro_vigilant(self):
        """Vigilant L1 replayed on the current ``utt_history`` (all rounds)."""
        n = len(self.utt_history)
        if self.retro_cache and self._retro_cached is not None:
            n_done, cached = self._retro_cached
            if n_done == n:
                return cached
        vig = self._fresh_vigilant()
        for u, tab in zip(self.utt_history, self.table_history):
            vig.update_with_tables(u, tab)
        if self.retro_cache:
            self._retro_cached = (n, vig)
        return vig

    def peek(self, utt):
        """Theta-marginal the listener would hold after hearing ``utt`` --
        including the detector update and a switch if ``utt`` would cross the
        boundary this round -- without mutating anything.

        Returns a dict theta -> probability (same layout as ``marginal_theta``).
        """
        if self.switched:
            return self.vigilant.infer_state(utt).marginal(0)

        driver = self._would_switch(utt)
        if driver is None:
            return self.naive.infer_state(utt).marginal(0)

        switch_type = driver.switch_type
        if switch_type == "hard_amnesic":
            n = len(self.thetas)
            return {theta: 1.0 / n for theta in self.thetas}

        tables = self.snapshot_tables()
        if switch_type == "hard":
            vig = self._retro_vigilant()
            return vig.infer_state_with_tables(utt, tables).marginal(0)
        if switch_type == "soft":
            vig = self._fresh_vigilant()
            vig.seed_from_theta_marginal(self.naive.infer_state(utt).marginal(0))
            return vig.infer_state_with_tables(utt, tables).marginal(0)
        raise ValueError(f"unknown switch_type {switch_type!r}")
