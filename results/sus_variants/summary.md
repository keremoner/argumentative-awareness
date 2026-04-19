# sus variants — summary

Config: `n=1, m=7, alpha=3, c=2`, 200 sims × 150 rounds per condition,
switching disabled. Null sweep: `psi=inf, theta_star in {0.1,0.3,0.5,0.7,0.9}`.
Alt sweep: `psi in {high,low}, theta_star=0.5`. Five scores attached as
passive observers: `surp2`, `sus_1` (== the original `sus`), `sus_3`,
`sus_4`, `sus_4b`.

Each `sus_v` logs both variance formulas from the spec: **naive**
(`sum_O W_v(O|u)·Var[-log S1(·|O,inf)]`) and **corrected**
(naive minus the law-of-total-variance within-posterior term).

See `fpr_tpr/`, `variance/`, `comparison/` for raw tables and figures.

## Headline table (t_warmup = 0)

FPR is pooled over `theta_star` at `psi=inf`; TPR is at `theta_star=0.5`.

| score  | FPR (naive) | FPR (corr.) | TPR high (naive/corr.) | TPR low (naive/corr.) |
|--------|-------------|-------------|------------------------|-----------------------|
| surp2  | 0.266       | 0.266       | 0.985 / 0.985          | 0.995 / 0.995         |
| sus_1  | 0.142       | 0.211       | 1.000 / 1.000          | 1.000 / 1.000         |
| sus_3  | 0.971       | 0.027       | 1.000 / 0.005          | 1.000 / 0.000         |
| sus_4  | 0.972       | 0.000       | 1.000 / 0.000          | 1.000 / 0.000         |
| sus_4b | 0.948       | 0.000       | 1.000 / 0.000          | 1.000 / 0.000         |

(`corr.` uses each score's `variance_corrected` for the threshold; `surp2`
returns a 2-tuple so naive == corrected.)

## Variance calibration

Empirical / theoretical ratio under the null (psi=inf), pooled across all
rounds:

**Per-round** `Var_emp[score^(t)] / mean(sigma^2_theo,(t))`:

| score  | naive mean | naive median | corrected mean | corrected median |
|--------|------------|--------------|----------------|------------------|
| surp2  | 1.00       | 0.98         | 1.00           | 0.98             |
| sus_1  | **0.88**   | 0.89         | **0.99**       | 0.99             |
| sus_3  | 1.66       | 1.67         | inf            | inf              |
| sus_4  | 1.72       | 1.72         | inf            | inf              |
| sus_4b | 1.59       | 1.59         | inf            | inf              |

**Running-mean** `Var_emp[Sus(t)] / (mean(sigma_bar^2(t)) / t)`:

| score  | naive mean | corrected mean |
|--------|------------|----------------|
| surp2  | 1.02       | 1.02           |
| sus_1  | 0.84       | **0.97**       |
| sus_3  | 16.8       | inf            |
| sus_4  | 17.9       | inf            |
| sus_4b | 14.5       | inf            |

Readings:

- **`sus_1` corrected formula works as advertised.** The per-round ratio
  moves from 0.88 (the Group 1 residual) to 0.99. The running-mean ratio
  also tightens from 0.84 to 0.97. The spec's law-of-total-variance
  correction is quantitatively correct for variant 1.
- **For `sus_3`, `sus_4`, `sus_4b` the naive formula under-estimates
  variance by 1.6-1.7× per round and by ~15× at the running-mean level.**
  This is what drives their 95-97% false-alarm rates under the naive
  threshold: the test thinks the score has much less variance than it
  actually does, so the threshold is far too loose.
- **The "corrected" formula overshoots for variants 3, 4, 4b.** The
  correction exceeds the naive variance, so `sigma^2_corrected` is
  clamped at 0 most rounds. That produces degenerate (zero) thresholds
  and, in the observe-gate `sigma > 0`, zero crossings — hence the all-zero
  corrected FPR/TPR for those three variants.

  **Why this happens:** the LTV identity the spec leans on
  (`Var[sus] = naive - correction`) only holds when `P_pred_v(u)·W_v(O|u)`
  equals the true null joint `L_1(O)·S1(u|O,inf)`. That holds exactly for
  variant 1, where `W_1(O|u)` is the real posterior. For variants 3/4/4b,
  `W_v` is an ad-hoc weighting (truth-restricted listener prior, no
  `S1(u|O,inf)` factor), so the joint `P_pred_v·W_v` is fictitious and the
  identity is no longer valid. The correction term it produces is simply
  too large.

## Variant correlations

Pooled per-round score correlations across all conditions:

```
            surp2  sus_1  sus_3  sus_4  sus_4b
surp2       1.00   0.72   0.52   0.53   0.56
sus_1       0.72   1.00   0.89   0.89   0.91
sus_3       0.52   0.89   1.00   0.99   0.99
sus_4       0.53   0.89   0.99   1.00   1.00
sus_4b      0.56   0.91   0.99   1.00   1.00
```

- `sus_3`, `sus_4`, `sus_4b` are near-duplicates of each other (pairwise
  rho ≥ 0.99). The `1/T(O)` correction in 4b barely moves the number
  relative to 4 in this setup, despite variance-of-T(O) being nonzero.
- `sus_1` correlates ~0.89 with the other three — they track the same
  underlying structure but with different spread.
- All `sus` variants share ~0.5–0.7 correlation with `surp2`.

## Which variant to use?

- **sus_1 + corrected variance** is the only combination that is
  calibration-consistent with the spec's theory. It also has the lowest
  FPR at fixed TPR (of any sus variant), and TPR remains saturated (1.0)
  at `theta_star=0.5` against both persuasive speakers.
- `sus_3`, `sus_4`, `sus_4b` are **dominated** in this study:
    - under the naive threshold they produce 95-97% FPR while adding no
      TPR over `sus_1`;
    - under the corrected threshold they degenerate to zero variance
      (spec artifact, not numerical noise).
- `surp2` remains a useful reference score (calibrated at ~1.0 ratio,
  lower TPR than any sus against a persuasive speaker).

## Does this move the Group 1 needle?

Not really. The per-round variance correction for `sus_1` is quantitatively
successful (0.88 → 0.99), but its effect on FPR is small and in the
**wrong** direction — `sus_1`'s FPR goes from 14.2% (naive) to 21.1%
(corrected) at t_warmup=0. The reason: naive was accidentally conservative,
so correcting it makes the test slightly more aggressive, not less. The
dominant FPR driver is still multiple-testing inflation against the
fixed-`c=2` horizon boundary, as Group 1 concluded.

TPR at `theta_star=0.5` stays at 1.0 for both persuasive speakers
regardless of variance formula. So using the corrected variance is a
free calibration fix for `sus_1` — it doesn't help FPR but it makes the
per-round test honest.

## Recommendations

1. **Adopt `sus_1` with the corrected variance formula** as the working
   observation-level score. Its per-round and running-mean variance are
   both calibrated to within a few percent.
2. **Drop `sus_3`, `sus_4`, `sus_4b`** from the candidate set. They add
   no detection power over `sus_1` and their corrected variance is
   theoretically unsound (the LTV identity does not apply to their
   weightings).
3. The FPR problem (`sus_1` still ~21%) is a **boundary-shape** problem,
   not a variance-formula problem. Next step: horizon-aware threshold
   (LIL boundary, confidence sequences, or empirically-calibrated `c`).
4. If `sus_3`/`sus_4`/`sus_4b` are of theoretical interest, they need a
   different variance derivation (not the LTV route in the spec) — e.g.,
   compute `Var_u[sus_v(u)]` directly from the true null predictive.

## Artifacts

```
results/sus_variants/
  raw_trajectories.parquet     # 1.05M rows, (sim, round, score) with both variances
  tau_summary.parquet           # 1400 rows, per-sim first-crossings for each variance kind
  run_config.json
  fpr_tpr/
    summary_stats.csv
    figures/{fpr_per_theta, fpr_tpr_bars, fpr_tpr_scatter}.png
  variance/
    per_round_variance.csv
    running_variance.csv
    ratio_agg.csv
    figures/{per_round_ratio, running_ratio}.png
  comparison/
    score_correlation.csv
    figures/{trajectories, correlation_matrix, pairwise_scatter}.png
```

Supporting tests: `tests/test_sus_variants.py` (5 tests — variant 1
identity with `compute_sus`, v3==v4 when L0=L1, v4b differs when T(O)
varies, corrected ≤ naive bound, dual-variance logging).
