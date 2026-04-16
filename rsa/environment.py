"""
Environment functions: observation generation, truth conditions, and likelihoods.

This module defines the clinical trial world where:
- n patients each undergo m treatment sessions
- Each session has probability theta of improvement
- Observations are histograms: obs[k] = number of patients with exactly k effective sessions
- Utterances use quantifiers (none/some/most/all) over (effective/ineffective)
"""

import math
from functools import lru_cache
from scipy.special import binom


def generate_all_observations(world_parameters):
    """
    Generate all possible observation histograms for n patients and m sessions.
    Each histogram is a tuple of length m+1, summing to n.

    obs[k] = number of patients who had exactly k effective sessions out of m.
    """
    n = world_parameters["n"]
    m = world_parameters["m"]
    observations = []

    def helper(current, depth, remaining):
        if depth == m:
            current.append(remaining)
            observations.append(tuple(current))
            current.pop()
            return
        for i in range(remaining + 1):
            current.append(i)
            helper(current, depth + 1, remaining - i)
            current.pop()

    helper([], 0, n)
    return observations


@lru_cache(maxsize=None)
def utterance_is_true(u, obs):
    """
    Evaluate whether utterance u is literally true given observation obs.

    For n=1 (single patient): u = (quantifier, predicate)
    For n>1 (multiple patients): u = (q1, q2, predicate)
      where q2 applies to sessions within each patient,
      and q1 applies across patients.
    """
    if u == ("all", "none"):
        return obs[0] == sum(obs) or obs[-1] == sum(obs)

    if sum(obs) > 1:
        # Multi-patient case: u = (q1, q2, pred)
        q1, q2, pred = u
        n = sum(obs)
        m = len(obs) - 1
        k = 0
        if pred == "ineffective":
            obs = obs[::-1]

        # Step 1: apply q2 to each patient's sessions
        if q2 == "none":
            k = obs[0]
        elif q2 == "some":
            k = sum(obs[1:])
        elif q2 == "most":
            k = sum(obs[math.floor(m / 2) + (m % 2):])
        elif q2 == "all":
            k = obs[-1]

        # Step 2: apply q1 across patients
        if q1 == "none":
            return k == 0
        elif q1 == "some":
            return k >= 1
        elif q1 == "most":
            return k > (n / 2)
        elif q1 == "all":
            return k == n
    else:
        # Single patient case: u = (q2, pred)
        q2, pred = u
        m = len(obs) - 1
        patient_score = obs.index(1) if pred == "effective" else m - obs.index(1)

        if q2 == "none":
            return patient_score == 0
        elif q2 == "some":
            return patient_score > 0
        elif q2 == "most":
            return patient_score > (m / 2)
        elif q2 == "all":
            return patient_score == m


@lru_cache(maxsize=None)
def multinomial(params):
    """Compute multinomial coefficient recursively."""
    if len(params) == 1:
        return 1
    return binom(sum(params), params[-1]) * multinomial(params[:-1])


@lru_cache(maxsize=None)
def get_obs_prob(obs, theta):
    """
    Compute P(obs | theta) for the clinical trial observation model.

    Each patient independently has Binomial(m, theta) effective sessions.
    obs[k] = count of patients with exactly k effective sessions.
    """
    n = sum(obs)
    m = len(obs) - 1
    flat_prob = 1

    def helper(effective):
        return math.comb(m, effective) * (theta ** effective) * ((1 - theta) ** (m - effective))

    for i in range(len(obs)):
        flat_prob *= helper(i) ** obs[i]
    return flat_prob * multinomial(obs)
