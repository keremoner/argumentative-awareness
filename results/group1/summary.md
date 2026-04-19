# Group 1 diagnostics — summary

Config: `n=1, m=7, alpha=3, c=2`, 200 sims × 150 rounds per condition,
switching disabled. Null sweep: `psi=inf, theta_star in {0.1,0.3,0.5,0.7,0.9}`.
Alt sweep (1.2 TPR): `psi in {high,low}, theta_star=0.5`.

See `exp_1.X/summary_stats.csv` (+ figures) for raw numbers. `surp1` was
collected but is not emphasized below per spec.

## 1.1 — FPR diagnostic under the null

Per-theta FPR (fraction crossing within 150 rounds) at `psi=inf`:

| theta_star | surp2 | sus   |
|------------|-------|-------|
| 0.1        | 0.335 | 0.145 |
| 0.3        | 0.170 | 0.075 |
| 0.5        | 0.085 | 0.165 |
| 0.7        | 0.235 | 0.115 |
| 0.9        | 0.330 | 0.130 |
| **mean**   | 0.231 | 0.126 |

- Both scores sit well above the nominal ~2.3% expected from a one-sided
  `c=2` z-threshold. This reproduces the 17-25% FPR problem.
- `tau` distribution (see `exp_1.1/figures/hist_tau.png`,
  `violin_tau.png`): crossings are **not** concentrated only at small `t`.
  Roughly 30-55% of crossings for `surp2` happen at `tau <= 10`, but there
  is a long right tail reaching past round 100. `sus` has a similar pattern
  with more mass in the tail.
- Conclusion: the early-round transient is a real contributor but not the
  whole story — the test keeps drawing fresh false alarms at large `t` too.

## 1.2 — Warm-up fix

FPR (pooled over theta) and TPR (`theta_star=0.5`) vs. warm-up length:

| t_warmup | FPR surp2 | FPR sus | TPR surp2 (high/low) | TPR sus (high/low) |
|----------|-----------|---------|----------------------|--------------------|
| 0        | 0.231     | 0.126   | 0.985 / 0.995        | 1.000 / 1.000      |
| 3        | 0.211     | 0.110   | 0.985 / 0.995        | 1.000 / 1.000      |
| 5        | 0.204     | 0.104   | 0.985 / 0.995        | 1.000 / 1.000      |
| 10       | 0.181     | 0.093   | 0.985 / 0.995        | 1.000 / 1.000      |

- Warm-up **costs essentially nothing in TPR** at `theta_star=0.5` (both
  persuasive speakers saturate TPR well above `t=10`).
- But warm-up **does not fix FPR**: even `t_warmup=10` leaves FPR around
  19% for `surp2` and 9% for `sus` — still far above the nominal 2-3%.
- Consistent with 1.1: a long-tail of late crossings cannot be removed by
  gating only the first few rounds.

## 1.3 — Per-round variance calibration

Ratio `Var_emp[score^(t)] / mean(sigma^2_theo,(t))` pooled over rounds:

| score | mean | median | min   | max   |
|-------|------|--------|-------|-------|
| surp2 | 1.00 | 0.99   | 0.69  | 1.34  |
| sus   | 0.89 | 0.89   | 0.52  | 1.03  |
| surp1 | 0.46 | 0.46   | 0.25  | 0.53  |

- For `surp2` the per-round variance formula is well-calibrated
  (ratio ~1 across all rounds).
- For `sus` the theoretical variance slightly **over**estimates the
  empirical variance (ratio ~0.9 — conservative direction).
- `surp1`'s theoretical variance is about 2x too large (ratio ~0.46). This
  explains why `surp1` has much lower FPR in 1.1 despite being a broadly
  similar quantity: its test is implicitly much more conservative than `c=2`
  because the threshold scales with a variance that exceeds reality.

## 1.4 — Running-mean variance scaling

Ratio `Var_emp[Sus(t)] / (mean(sigma_bar^2(t)) / t)` pooled over rounds:

| score | mean | median | min   | max   |
|-------|------|--------|-------|-------|
| surp2 | 1.06 | 1.06   | 0.94  | 1.13  |
| sus   | 0.84 | 0.84   | 0.65  | 0.94  |
| surp1 | 0.71 | 0.73   | 0.30  | 1.05  |

- For `surp2` the `1/t` shrinkage holds to within ~10% — auto-correlation
  is not breaking the CLT scaling in a major way.
- For `sus` empirical `Var[Sus(t)]` is even smaller than theoretical,
  implying some anti-correlation between rounds (no FPR inflation from
  this direction).
- The log-log plots (`exp_1.4/figures/sus_variance_scaling.png`) confirm
  that the empirical curve tracks the `1/t` reference rather than a slower
  law like `1/sqrt(t)`.

## Overall picture

1. The per-round variance formula and the `1/t` running-mean scaling are
   **both correct** (for `surp2` especially, but also well enough for
   `sus`). So the FPR inflation is **not** caused by a miscomputed
   `sigma^2,(t)` or a broken CLT assumption.
2. Warm-up cannot fix it either: false alarms are spread across all `t`,
   not just the first few rounds.
3. The most plausible remaining cause is **multiple-testing inflation**:
   the sequential test compares `Sus(t)` against `c * sigma_bar(t) / sqrt(t)`
   at every round for 150 rounds, but `c=2` gives only a **pointwise**
   (~2.3%) significance level. Under an iid null, the first-crossing
   probability of a running mean against a fixed-z boundary naturally
   grows with the horizon (roughly log-log in `t`). This is exactly the
   kind of inflation we observe.
4. `surp1`'s theoretical variance is ~2x too large. This warrants a
   separate look (is the inner expectation using the wrong distribution?),
   though it masquerades as conservative rather than inflated FPR, so it
   is not the cause of the core issue.

## Preliminary recommendation for Group 2 and beyond

- **Replace the fixed-z boundary with a horizon-aware one.** Candidates:
  - Law-of-iterated-logarithm boundary:
    `c * sigma_bar(t) * sqrt(2 log log t / t)` (scales with log-log t).
  - Time-uniform confidence sequences
    (e.g. Howard-Ramdas-style betting confidence sequences).
  - Empirically calibrated `c` targeting a specific horizon-level FPR
    (if we commit to a fixed max `T`, we can pick `c` from null-sim
    quantiles and get exact FPR control).
- **Warm-up is cheap insurance**: `t_warmup=5` or `10` has no TPR cost at
  `theta_star=0.5` and trims a bit of FPR. Include it alongside the
  horizon-aware boundary, not as a standalone fix.
- **Investigate `surp1`'s variance formula.** Low priority for the main
  pipeline (since `surp1` is not part of what we present), but the 2x
  mismatch suggests a bug worth identifying before using `surp1`
  anywhere load-bearing.
- **Do not recalibrate `sus`'s variance formula yet.** The ~10% residual
  looks like finite-sample noise plus mild anti-correlation; it is in the
  conservative direction.

## Artifacts

```
results/group1/
  raw_trajectories.parquet   # 630k rows, per (sim,round,score)
  tau_summary.parquet         # 1.4k rows, one per (psi,theta,sim)
  run_config.json
  exp_1.1/summary_stats.csv   + figures/{hist_tau,violin_tau,sus_trajectory}.png
  exp_1.2/summary_stats.csv   + figures/fpr_tpr_warmup.png
  exp_1.3/variance_table.csv  + figures/{per_round_variance,variance_ratio}.png
  exp_1.4/scaling_table.csv   + figures/{sus_variance_scaling,sus_variance_ratio}.png
```
