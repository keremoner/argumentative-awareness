"""
Listener wrapper that runs any number of SequentialTest instances in parallel
against an internal credulous L1^inf belief.

By default every attached test is a passive observer -- no switching, no
coupling between tests.  Enabling ``switch_enabled`` on one test authorises
it to drive a hard/soft switch to a full vigilant L1 at its stopping time.
"""

from __future__ import annotations

from copy import deepcopy
import numpy as np

from ..core import Belief
from ..listener1 import Listener1
from .scores import ScoreContext


class DetectionListener:
    def __init__(self, thetas, psis, speaker, world, semantics,
                 tests=None, alpha=1.0):
        """
        Parameters
        ----------
        thetas : sequence of float
            Candidate theta values.  Must match the speaker's ``thetas``.
        psis : sequence of str
            Full psi hypothesis space used after switching, e.g.
            ``["inf", "high", "low"]``.  The naive listener always runs
            with ``["inf"]``.
        speaker : Speaker1
            Shared speaker instance (same as the one generating utterances).
        tests : list of SequentialTest, optional
            Zero or more detection tests.  Passive unless ``switch_enabled``.
        """
        self.thetas = list(thetas)
        self.psis = list(psis)
        self.speaker = speaker
        self.world = world
        self.semantics = semantics
        self.alpha = alpha
        self.tests = list(tests) if tests is not None else []

        self._theta_to_index = {t: i for i, t in enumerate(self.thetas)}

        self.naive = Listener1(
            self.thetas, ["inf"], speaker, world, semantics,
            listener_type="inf", alpha=alpha,
        )
        self.vigilant = None
        self.switched = False
        self.switched_at = None
        self.switch_driver = None  # which test triggered the switch
        self.round = 0

        self.utt_history = []
        self.hist = [deepcopy(self.naive.state_belief)]

    @property
    def state_belief(self):
        return self.vigilant.state_belief if self.switched else self.naive.state_belief

    def _l1_theta_array(self):
        """L1^(t)(theta) as an ndarray aligned with ``self.thetas``."""
        theta_dict = self.naive.marginal_theta()
        return np.array([theta_dict.get(t, 0.0) for t in self.thetas], dtype=float)

    def build_context(self, u_obs):
        """Construct a ScoreContext from the *current* naive listener state.

        Must be called before the round's belief update.
        """
        return ScoreContext(
            self.thetas, self.speaker, self.world, self.semantics,
            self._l1_theta_array(), u_obs,
        )

    def update(self, utt):
        """Process one utterance: run tests, update beliefs, maybe switch."""
        self.round += 1
        t = self.round

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
            self.vigilant.update(utt)

        self.utt_history.append(utt)
        self.hist.append(deepcopy(self.state_belief))
        return self.state_belief

    def _trigger_switch(self, switch_type):
        self.switched = True
        self.switched_at = self.round

        self.vigilant = Listener1(
            self.thetas, self.psis, self.speaker, self.world, self.semantics,
            listener_type="vig", alpha=self.alpha,
        )

        if switch_type == "soft":
            theta_probs = self.naive.marginal_theta()
            n_psi = len(self.psis)
            joint = self.vigilant.state_belief.values
            prior = np.array(
                [theta_probs.get(theta, 0.0) / n_psi for (theta, psi) in joint],
                dtype=float,
            )
            s = prior.sum()
            if s > 0:
                prior = prior / s
            self.vigilant.state_belief = Belief(joint, prior)
            self.vigilant.hist = [deepcopy(self.vigilant.state_belief)]
        # hard switch: leave vigilant at its default uniform prior

    def marginal_theta(self):
        return (self.vigilant.marginal_theta()
                if self.switched else self.naive.marginal_theta())

    def marginal_psi(self):
        return (self.vigilant.marginal_psi()
                if self.switched else self.naive.marginal_psi())
