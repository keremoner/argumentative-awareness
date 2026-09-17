"""
Listener wrapper that runs any number of SequentialTest instances in parallel
against an internal credulous L1^inf belief.

By default every attached test is a passive observer -- no switching, no
coupling between tests.  Enabling ``switch_enabled`` on one test authorises
it to drive a switch to a full vigilant L1 at its stopping time tau.

Round structure at tau
----------------------
The tests score u_tau against the credulous listener *as it stands* (its
belief before u_tau).  If a switch-enabled test crosses, the switch happens
right there, before any belief update, and u_tau is then absorbed exactly
once -- by the vigilant listener.  The credulous listener freezes at tau - 1;
it is never read again.

Switch types
------------
``"hard"`` (retrospective)
    A vigilant L1 that has been listening from round 1.  The listener keeps a
    ``shadow`` vigilant L1 updated in parallel with the credulous one every
    round, so at tau the shadow simply becomes the active listener: its belief
    equals an always-vigilant L1 run on u_1 .. u_tau, with no replay needed.

``"soft"``
    A fresh vigilant L1 seeded with the credulous theta-marginal *before*
    u_tau, spread uniformly over psi -- it keeps the pre-tau credulous belief
    and is direction-blind at tau -- and then updated on u_tau vigilantly.
    At tau = 1 this coincides with ``"hard"``.

``table_history`` records the speaker's ``{psi: P(u | O, psi)}`` tables as they
were when each utterance arrived.  The listener itself no longer needs them
(every sub-listener is updated live), but the offline runners derive
switching trajectories from stored streams and read them from here.
"""

from __future__ import annotations

from copy import deepcopy
import numpy as np

from ..listener1 import Listener1
from .scores import ScoreContext


SWITCH_TYPES = ("hard", "soft")


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
            Accepted for backward compatibility and ignored.  The hard switch
            used to replay the history inside ``peek``; the shadow vigilant
            listener makes that (and its cache) unnecessary.
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
        # Always-vigilant L1 run alongside the naive one from round 1; becomes
        # the active listener under a hard switch.
        self.shadow = self._fresh_vigilant()
        self.vigilant = None
        self.switched = False
        self.switched_at = None
        self.switch_driver = None  # which test triggered the switch
        self.switch_type = None
        self.round = 0

        self.utt_history = []
        self.table_history = []    # per round: {psi: (n_obs, n_utt) array}
        self.hist = [deepcopy(self.naive.state_belief)]
        self._soft_scratch = None  # reusable seeded vigilant L1 for soft peeks

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

    def _live_tables(self):
        """The speaker's current ``{psi: P(u | O, psi)}`` tables, uncopied."""
        return {psi: self.speaker.obs_utt_table_for_psi(psi) for psi in self.psis}

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
        """Process one utterance: run tests, maybe switch, then update beliefs."""
        self.round += 1
        t = self.round

        # Snapshot the tables that generate/explain u_t *before* anything
        # moves; every sub-listener is updated from this same triple.
        tables = self.snapshot_tables()
        self.utt_history.append(utt)
        self.table_history.append(tables)

        if self.switched:
            self.vigilant.update_with_tables(utt, tables)
        else:
            ctx = self.build_context(utt)
            for test in self.tests:
                crossed = test.observe(ctx)
                if (crossed and test.switch_enabled
                        and self.switch_driver is None):
                    self.switch_driver = test

            if self.switch_driver is not None and self.switch_driver.tau == t:
                # Switch first, then let the vigilant listener absorb u_tau
                # once.  The naive listener freezes at t - 1.
                self._trigger_switch(self.switch_driver.switch_type)
                self.vigilant.update_with_tables(utt, tables)
            else:
                self.naive.update(utt)
                self.shadow.update_with_tables(utt, tables)

        self.hist.append(deepcopy(self.state_belief))
        return self.state_belief

    def _trigger_switch(self, switch_type):
        if switch_type not in SWITCH_TYPES:
            raise ValueError(f"unknown switch_type {switch_type!r}; expected one of {SWITCH_TYPES}")
        self.switched = True
        self.switched_at = self.round
        self.switch_type = switch_type

        if switch_type == "hard":
            # The shadow has seen u_1 .. u_{tau-1} vigilantly; it takes over.
            self.vigilant = self.shadow
        else:
            # Pre-u_tau credulous marginal, uniform over psi.
            vig = self._fresh_vigilant()
            vig.seed_from_theta_marginal(self.naive.marginal_theta())
            self.vigilant = vig

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
        if switch_type == "hard":
            return self.shadow.infer_state(utt).marginal(0)
        if switch_type == "soft":
            if self._soft_scratch is None:
                self._soft_scratch = self._fresh_vigilant()
            scratch = self._soft_scratch
            scratch.seed_from_theta_marginal(self.naive.marginal_theta())
            return scratch.infer_state_with_tables(utt, self._live_tables()).marginal(0)
        raise ValueError(f"unknown switch_type {switch_type!r}; expected one of {SWITCH_TYPES}")
