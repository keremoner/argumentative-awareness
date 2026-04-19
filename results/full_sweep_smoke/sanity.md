# sanity.md

Automated checks from `run_sweep.py`. All should pass.
**Overall: PASS**  


**1. Row count.** expected=29,700  actual=29,700  => PASS
**2. No NaN/inf in score columns.** => PASS
**3. `sus1_sigma2_corrected <= sus1_sigma2_naive`** (up to 1e-9 tol).  violations=0  => PASS
**4. Beliefs sum to 1.**  L1 max|sum-1|=5.55e-16  L0 max|sum-1|=5.55e-16  => PASS
**5. `Sus(t=150)` mean under null** (pooled over theta*, alpha).
   - surp2: -0.0127
   - sus_1: -0.0014
   (expected close to 0; slight positive bias acceptable due to multiple-testing selection discussed in Group 1.)

## Per-alpha null Sus(t=150) means

```
        surp2    sus1
alpha                
1.0   -0.0113  0.0086
1.5   -0.0290  0.0060
2.0    0.0222 -0.0074
2.5    0.0083 -0.0033
3.0   -0.0068  0.0046
4.0   -0.0241 -0.0126
5.0    0.0227  0.0176
7.0    0.0117  0.0069
10.0  -0.0740 -0.0368
15.0  -0.0368  0.0054
20.0  -0.0221 -0.0047
```

## Dataset shape

- trajectories: 29,700 rows, 45 columns
- simulations: 1,485 rows
- unique cells: 297