# sus variants — summary

> **Re-run 2026-09-09.** Variants 3/4/4b are retired (`make_sus_variant` raises
> `NotImplementedError`): they are dominated, and their score is not mean-zero
> under the listener's joint, so the exact variance below does not apply to them.
> `sus_1` now reports a single exact state-only null variance
> `V = sum_u p(u) s(u)^2` instead of a naive/corrected pair. Re-measured, its
> per-round calibration ratio is 1.000 and its running-mean ratio 1.010.
> See `results/full_sweep/variance_fix_diff.md` for the before/after.

Config: `n=1, m=7, alpha=3, c=2`, 200 sims × 150 rounds per condition,
switching disabled. Null sweep: `psi=inf, theta_star in {0.1,0.3,0.5,0.7,0.9}`.
Alt sweep: `psi in {high,low}, theta_star=0.5`. Two scores attached as
passive observers: `surp2` and `sus_1` (== the original `sus`).

Each score logs one exact null variance; there is no naive/corrected pair and
nothing is clipped.

See `fpr_tpr/`, `variance/`, `comparison/` for raw tables and figures.

## Headline table (t_warmup = 0)

FPR is pooled over `theta_star` at `psi=inf`; TPR is at `theta_star=0.5`.

| score  | FPR   | TPR high | TPR low |
|--------|-------|----------|---------|
| surp2  | 0.240 | 1.000    | 0.985   |
| sus_1  | 0.194 | 1.000    | 1.000   |

## Variance calibration

Empirical / theoretical ratio under the null (psi=inf), pooled across all
rounds. One variance per score now, so one column.

**Per-round** `Var_emp[score^(t)] / mean(sigma^2_theo,(t))`:

| score  | mean  | median |
|--------|-------|--------|
| surp2  | 0.994 | 0.984  |
| sus_1  | 1.000 | 0.996  |

**Running-mean** `Var_emp[Sus(t)] / (mean(sigma_bar^2(t)) / t)`:

| score  | mean  |
|--------|-------|
| surp2  | 1.008 |
| sus_1  | 1.010 |


Readings:

- **`sus_1` is honest at both levels.** The per-round ratio is 1.000 and the
  running-mean ratio 1.010. The exact state-only variance needs no correction
  term and cannot go negative, so nothing is clipped anywhere.
- **`surp2` is equally well calibrated** (0.994 / 1.008); its variance was
  always the exact varentropy of the prior predictive.
- **Variants 3, 4, 4b are retired.** They were dominated on every axis, and
  their score is not mean-zero under the listener's joint
  `L_1(O)*S1(u|O,inf)` -- `W_v` is an ad-hoc weighting with the `S1(u|O,inf)`
  factor stripped out, so the implied joint is fictitious. The exact variance
  has no analogue for them. `make_sus_variant` now raises
  `NotImplementedError` for all three.

## Score correlations

Pooled per-round score correlations across all conditions:

```
            surp2  sus_1
score_type              
surp2        1.00   0.72
sus_1        0.72   1.00
```

- `sus_1` and `surp2` share about 0.7 correlation: they track related
  structure, but `sus_1` takes the log inside the O-marginalization and
  is the more sensitive of the two.
