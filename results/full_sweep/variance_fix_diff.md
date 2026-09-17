# sus_1 variance fix: before / after

- **old**: `results/full_sweep` -- per-round variance `var_naive(u_obs) - K`, an unbiased-in-expectation proxy that depends on the utterance heard.
- **new**: `results/full_sweep_v2` -- exact `V = sum_u p(u) s(u)^2`, a function of the listener state only.

The two sweeps share their scores exactly (the new one carries them over verbatim and every row was re-derived and checked), so every difference below comes from the variance alone.

All rates pool over `theta*` and `alpha >= 1.5`.

## 1. Operating characteristics

| variance | c | FPR | TPR pers+ | TPR pers- |
|---|---|---|---|---|
| naive | 2 | 0.180 | 0.984 | 0.983 |
| naive | 3 | 0.046 | 0.925 | 0.926 |
| naive | 3.5 | 0.026 | 0.888 | 0.889 |
| naive | 5 | 0.007 | 0.777 | 0.772 |
| naive | 7 | 0.003 | 0.648 | 0.646 |
| corrected | 2 | 0.288 | 0.999 | 0.999 |
| corrected | 3 | 0.118 | 0.995 | 0.994 |
| corrected | 3.5 | 0.091 | 0.989 | 0.988 |
| corrected | 5 | 0.064 | 0.945 | 0.944 |
| corrected | 7 | 0.058 | 0.826 | 0.828 |
| exact | 2 | 0.219 | 0.996 | 0.995 |
| exact | 3 | 0.065 | 0.974 | 0.970 |
| exact | 3.5 | 0.031 | 0.947 | 0.947 |
| exact | 5 | 0.011 | 0.830 | 0.826 |
| exact | 7 | 0.003 | 0.681 | 0.683 |

## 2. Per-alpha null crossing rate at c = 3.5

| variance | 1.5 | 2 | 2.5 | 3 | 4 | 5 | 7 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| naive | 0.000 | 0.000 | 0.001 | 0.006 | 0.009 | 0.021 | 0.068 | 0.051 | 0.049 | 0.058 |
| corrected | 0.374 | 0.217 | 0.035 | 0.013 | 0.016 | 0.023 | 0.069 | 0.051 | 0.049 | 0.058 |
| exact | 0.002 | 0.002 | 0.003 | 0.012 | 0.018 | 0.027 | 0.066 | 0.065 | 0.051 | 0.063 |

## 3. Per-round calibration ratio  Var[s] / E[sigma^2]

Empirical variance of the per-round score across null sims, over the mean reported variance, averaged across rounds. 1.00 is honest.

| variance | 1.5 | 2 | 2.5 | 3 | 4 | 5 | 7 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| naive | 0.662 | 0.737 | 0.792 | 0.850 | 0.914 | 0.952 | 0.984 | 1.001 | 0.993 | 0.988 |
| corrected | 1.005 | 1.000 | 0.995 | 1.003 | 0.995 | 0.993 | 0.993 | 1.002 | 0.993 | 0.988 |
| exact | 1.003 | 1.003 | 0.998 | 1.005 | 0.996 | 0.994 | 0.994 | 1.003 | 0.994 | 0.988 |

## 4. What the old per-round proxy was doing

Measured on the **old** trajectories, null cells only: the correlation between the per-round score and the per-round `sigma2_corrected` that was supposed to scale it, and how often that variance came out non-positive.

| alpha | corr(score, sigma2_corrected) | frac(sigma2_corrected <= 0) |
|---|---|---|
| 1.5 | -0.271 | 0.353 |
| 2 | -0.157 | 0.332 |
| 2.5 | -0.081 | 0.271 |
| 3 | -0.026 | 0.036 |
| 4 | +0.011 | 0.033 |
| 5 | +0.011 | 0.033 |
| 7 | -0.001 | 0.035 |
| 10 | +0.002 | 0.035 |
| 15 | +0.001 | 0.037 |
| 20 | -0.006 | 0.038 |

A variance that correlates with its own numerator is not a scale factor; it is part of the statistic. The exact variance is constant across utterances within a round, so its correlation with the score is identically zero and it is never non-positive.
