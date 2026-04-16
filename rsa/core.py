"""
Core classes: Belief, Semantics, World.
"""

import numpy as np
import random
from copy import deepcopy


class Belief:
    """Joint probability distribution over discrete states."""

    def __init__(self, values, prior=None):
        """
        values: list of possible states, e.g. [0.1, 0.2, ...] or [(theta, psi), ...]
        prior: optional list of probabilities (uniform if None)
        """
        self.values = values
        if prior is None:
            self.prob = np.ones(len(values)) / len(values)
        else:
            self.prob = np.array(prior, dtype=float)
            self.prob /= np.sum(self.prob)

    def update(self, likelihoods):
        """Bayesian update: multiply by likelihoods and renormalize."""
        likelihoods = np.array(likelihoods, dtype=float)
        self.prob *= likelihoods
        if np.sum(self.prob) > 0:
            self.prob /= np.sum(self.prob)

    def as_dict(self):
        return dict(zip(self.values, self.prob))

    def marginal(self, index):
        """
        Compute marginal distribution over one coordinate of joint states.
        index=0 for theta, index=1 for psi (when values are tuples).
        """
        if not isinstance(self.values[0], tuple):
            return self.as_dict()
        marg = {}
        for val, p in zip(self.values, self.prob):
            key = val[index]
            marg[key] = marg.get(key, 0.0) + p
        return marg


class Semantics:
    """Defines utterance space and truth conditions."""

    def __init__(self, utterances, truth_calc):
        self.utterances = tuple(utterances)
        self.truth_calc = truth_calc
        self._utterance_to_index = {utt: idx for idx, utt in enumerate(self.utterances)}
        self._truth_tables = {}

    def utterance_space(self):
        return self.utterances

    def is_true(self, utt, obs):
        return self.truth_calc(utt, obs)

    def utterance_index(self, utt):
        return self._utterance_to_index[utt]

    def truth_table(self, world):
        cache_key = tuple(world.generate_all_obs())
        if cache_key not in self._truth_tables:
            observations = list(cache_key)
            table = np.zeros((len(observations), len(self.utterances)), dtype=bool)
            for obs_idx, obs in enumerate(observations):
                for utt_idx, utt in enumerate(self.utterances):
                    table[obs_idx, utt_idx] = self.truth_calc(utt, obs)
            self._truth_tables[cache_key] = table
        return self._truth_tables[cache_key]

    def truth_value(self, world, obs, utt):
        return bool(self.truth_table(world)[world.obs_index(obs), self.utterance_index(utt)])


class World:
    """
    Represents the observable world with a latent parameter theta.

    theta: true underlying rate
    world_parameters: dict with 'n' (patients) and 'm' (sessions)
    obs_model: function(obs, theta) -> P(obs | theta)
    obs_generator: function(world_parameters) -> list of all possible observations
    """

    def __init__(self, theta, world_parameters, obs_model, obs_generator):
        self.theta = theta
        self.world_parameters = world_parameters
        self.obs_model = obs_model
        self.obs_generator = obs_generator
        self.all_obs = None
        self._obs_index = None
        self._obs_prob_tables = {}
        self._obs_sample_probs = {}

    def obs_prob(self, obs, theta):
        """Return P(obs | theta)."""
        return self.obs_model(obs, theta)

    def generate_all_obs(self):
        """Generate all possible observations."""
        if self.all_obs is None:
            self.all_obs = self.obs_generator(world_parameters=self.world_parameters)
        return self.all_obs

    def obs_index(self, obs):
        if self._obs_index is None:
            self._obs_index = {obs_case: idx for idx, obs_case in enumerate(self.generate_all_obs())}
        return self._obs_index[obs]

    def obs_prob_table(self, thetas):
        theta_key = tuple(thetas)
        if theta_key not in self._obs_prob_tables:
            observations = self.generate_all_obs()
            table = np.array(
                [[self.obs_model(obs, theta) for theta in theta_key] for obs in observations],
                dtype=float,
            )
            self._obs_prob_tables[theta_key] = table
        return self._obs_prob_tables[theta_key]

    def obs_likelihoods(self, obs, thetas):
        return self.obs_prob_table(thetas)[self.obs_index(obs)]

    def sample_obs(self):
        """Sample an observation according to P(obs | self.theta)."""
        all_obs = self.generate_all_obs()
        if self.theta not in self._obs_sample_probs:
            probs = self.obs_prob_table((self.theta,))[:, 0].copy()
            probs = probs / probs.sum()
            self._obs_sample_probs[self.theta] = probs
        probs = self._obs_sample_probs[self.theta]
        return random.choices(all_obs, weights=probs, k=1)[0]
