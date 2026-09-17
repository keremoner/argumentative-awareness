"""
Offline replay utilities for stored utterance streams.

Every dataset produced by ``experiments/switching`` stores the observation
and the utterance of every round.  Because the S1 tables at round i depend
only on the utterance history (through L0) and on alpha, the per-round
speaker tables can be regenerated exactly from a stored stream
(``regenerate_tables``), after which *any* listener can be replayed on it
without re-simulating the speaker.

``TableSpeaker`` is the stand-in that serves stored tables to a listener.  It
exposes exactly the speaker surface that ``Listener1``, ``DetectionListener``
and ``ScoreContext`` read (``obs_utt_table_for_psi``,
``dist_over_utterances_theta_array``, ``dist_over_utterances_theta``) and
nothing else, so a replayed listener cannot accidentally reach a live speaker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..listener1 import Listener1


PSIS_DEFAULT = ("inf", "high", "low")


@dataclass
class Trajectory:
    """Per-round listener summary (1-based rounds map to 0-based rows)."""
    E_theta: np.ndarray            # (T,)
    std_theta: np.ndarray          # (T,)
    theta: np.ndarray              # (T, n_theta) full theta-marginal
    psi: np.ndarray                # (T, n_psi) psi-marginal, NaN where absent
    switched_at: int | None = None
    thetas: tuple = field(default_factory=tuple)
    psis: tuple = field(default_factory=tuple)

    def __len__(self):
        return len(self.E_theta)

    def copy(self):
        return Trajectory(self.E_theta.copy(), self.std_theta.copy(),
                          self.theta.copy(), self.psi.copy(), self.switched_at,
                          self.thetas, self.psis)


class TableSpeaker:
    """Serves stored per-round tables as if it were the live speaker."""

    def __init__(self, tables_per_round, thetas, world, semantics):
        self.tables = list(tables_per_round)
        self.thetas = list(thetas)
        self.world = world
        self.semantics = semantics
        self.round_idx = 0
        self._utterances = semantics.utterance_space()
        self._theta_to_index = {t: i for i, t in enumerate(self.thetas)}
        self._obs_prob_table = world.obs_prob_table(self.thetas)

    @property
    def psis(self):
        return list(self.tables[0].keys())

    def obs_utt_table_for_psi(self, psi):
        return self.tables[self.round_idx][psi]

    def dist_over_utterances_theta_array(self, theta, psi):
        table = np.asarray(self.obs_utt_table_for_psi(psi), dtype=float)
        return table.T @ self._obs_prob_table[:, self._theta_to_index[theta]]

    def dist_over_utterances_theta(self, theta, psi):
        return dict(zip(self._utterances, self.dist_over_utterances_theta_array(theta, psi)))


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _moments(theta_probs, thetas_arr):
    e = float(theta_probs @ thetas_arr)
    v = float(theta_probs @ (thetas_arr ** 2)) - e * e
    return e, float(np.sqrt(max(v, 0.0)))


def _theta_vec(listener, thetas):
    if hasattr(listener, "theta_array"):
        return np.asarray(listener.theta_array(), dtype=float)
    d = listener.marginal_theta()
    return np.array([d.get(t, 0.0) for t in thetas], dtype=float)


def _psi_vec(listener, psis):
    """psi-marginal aligned with ``psis``; NaN unless the listener carries the
    whole psi space (credulous listeners and un-switched DetectionListeners
    have only ``inf``)."""
    d = listener.marginal_psi() if hasattr(listener, "marginal_psi") else {}
    if all(p in d for p in psis):
        return np.array([d[p] for p in psis], dtype=float)
    return np.full(len(psis), np.nan)


def empty_trajectory(T, thetas, psis=PSIS_DEFAULT):
    return Trajectory(np.empty(T), np.empty(T), np.empty((T, len(thetas))),
                      np.full((T, len(psis)), np.nan), None,
                      tuple(thetas), tuple(psis))


def record_round(traj, i, listener):
    thetas_arr = np.asarray(traj.thetas, dtype=float)
    th = _theta_vec(listener, traj.thetas)
    traj.theta[i] = th
    traj.E_theta[i], traj.std_theta[i] = _moments(th, thetas_arr)
    traj.psi[i] = _psi_vec(listener, traj.psis)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def replay_listener(utts, tables_per_round, listener_factory, thetas, world,
                    semantics, psis=PSIS_DEFAULT):
    """Run any listener on a stored utterance sequence with stored per-round
    speaker tables.

    ``listener_factory(speaker)`` receives a ``TableSpeaker`` and returns the
    listener (a ``Listener1`` or a ``DetectionListener`` built on it).  Round
    i uses ``tables_per_round[i]``.  Returns a ``Trajectory`` with per-round
    E[theta], std[theta], the full theta-marginal, the psi-marginal (NaN when
    the listener has none) and ``switched_at`` if the listener reports one.
    """
    T = len(utts)
    if len(tables_per_round) < T:
        raise ValueError("need one table triple per utterance")
    speaker = TableSpeaker(tables_per_round, thetas, world, semantics)
    listener = listener_factory(speaker)
    traj = empty_trajectory(T, thetas, psis)
    for i, u in enumerate(utts):
        speaker.round_idx = i
        listener.update(u)
        record_round(traj, i, listener)
    traj.switched_at = getattr(listener, "switched_at", None)
    return traj


def splice(credulous_traj, vigilant_traj, tau):
    """Trajectory of the retrospective switching listener: credulous for
    t < tau, always-vigilant for t >= tau (exact by construction, since the
    retrospective vigilant belief at tau equals the always-vigilant one).
    ``tau`` is 1-based; ``None`` means no switch (pure credulous)."""
    out = credulous_traj.copy()
    out.psi[:] = np.nan
    out.switched_at = None
    if tau is None:
        return out
    k = int(tau) - 1
    if k < 0 or k >= len(out):
        raise ValueError(f"tau={tau} outside 1..{len(out)}")
    out.E_theta[k:] = vigilant_traj.E_theta[k:]
    out.std_theta[k:] = vigilant_traj.std_theta[k:]
    out.theta[k:] = vigilant_traj.theta[k:]
    out.psi[k:] = vigilant_traj.psi[k:]
    out.switched_at = int(tau)
    return out


def tau_for_rule(sus_traj, sigma_bar_traj, c):
    """First 1-based round t at which Sus(t) > c * sigma_bar(t) / sqrt(t),
    with sigma_bar(t) > 0; ``None`` if never.  ``sigma_bar_traj`` is the
    running sigma (the square root of the stored ``*_sigma_bar2``)."""
    sus = np.asarray(sus_traj, dtype=float)
    sig = np.asarray(sigma_bar_traj, dtype=float)
    if not np.isfinite(c):
        return None
    t = np.arange(1, len(sus) + 1, dtype=float)
    thr = c * sig / np.sqrt(t)
    hit = np.flatnonzero((sig > 0.0) & (sus > thr))
    return int(hit[0]) + 1 if hit.size else None


def regenerate_tables(utts, obs_seq, thetas, alpha, n, m, psis=PSIS_DEFAULT):
    """Rebuild the S1 tables of every round from a stored (obs, utt) stream by
    replaying the S0 -> L0 -> S1 chain exactly as the game loop does
    (``s1.update(obs)``, ``l0.update(utt)``, ``s0.update(obs)`` after each
    round).  ``tables[i]`` are the tables in force when ``utts[i]`` was
    produced.  ``obs_seq`` entries may be observation tuples or indices."""
    from ..setup import make_world, make_semantics
    from ..speaker0 import Speaker0
    from ..listener0 import Listener0
    from ..speaker1 import Speaker1

    world = make_world(thetas[0], n=n, m=m)      # theta irrelevant for tables
    semantics = make_semantics(n=n)
    all_obs = world.generate_all_obs()
    s0 = Speaker0(thetas, semantics=semantics, world=world)
    l0 = Listener0(thetas, s0, semantics=semantics, world=world)
    s1 = Speaker1(thetas, l0, semantics=semantics, world=world, alpha=alpha, psi="inf")
    tables = []
    for u, o in zip(utts, obs_seq):
        if not isinstance(o, tuple):
            o = all_obs[int(o)]
        tables.append({psi: np.array(s1.obs_utt_table_for_psi(psi), dtype=float, copy=True)
                       for psi in psis})
        s1.update(o)
        l0.update(u)
        s0.update(o)
    return tables


def switching_trajectory(cred_traj, vig_traj, utts, tables_per_round, tau,
                         switch_type, thetas, world, semantics, alpha,
                         psis=PSIS_DEFAULT):
    """Trajectory of a switching listener that fires at ``tau`` (1-based, or
    None), derived offline from the stored stream.

    * ``"hard"``          -- ``splice`` (exact).
    * ``"soft"``          -- credulous for t < tau; at tau the vigilant L1 is
                             seeded with the credulous theta-marginal *after*
                             u_tau, uniform over psi, then updated on
                             u_tau .. u_T with the stored tables.
    * ``"hard_amnesic"``  -- credulous for t < tau; uniform at tau; a fresh
                             vigilant L1 updated on u_{tau+1} .. u_T.
    """
    if tau is None:
        out = cred_traj.copy()
        out.psi[:] = np.nan
        out.switched_at = None
        return out
    if switch_type == "hard":
        return splice(cred_traj, vig_traj, tau)

    T = len(utts)
    k = int(tau) - 1
    out = cred_traj.copy()
    out.psi[:] = np.nan
    out.switched_at = int(tau)
    speaker = TableSpeaker(tables_per_round, thetas, world, semantics)
    vig = Listener1(thetas, list(psis), speaker, world, semantics, "vig", alpha)

    if switch_type == "soft":
        theta_probs = dict(zip(thetas, cred_traj.theta[k]))
        vig.seed_from_theta_marginal(theta_probs)
        for i in range(k, T):
            vig.update_with_tables(utts[i], tables_per_round[i])
            record_round(out, i, vig)
        return out
    if switch_type == "hard_amnesic":
        record_round(out, k, vig)          # uniform at tau, never sees u_tau
        for i in range(k + 1, T):
            vig.update_with_tables(utts[i], tables_per_round[i])
            record_round(out, i, vig)
        return out
    raise ValueError(f"unknown switch_type {switch_type!r}")
