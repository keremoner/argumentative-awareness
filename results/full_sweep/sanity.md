# sanity.md

Automated checks from `run_sweep.py`. All should pass.
**Overall: PASS**  


**1. Row count.** expected=8,910,000  actual=8,910,000  => PASS
**2. No NaN/inf in score columns.** => PASS
**3. `sus1_sigma2_corrected <= sus1_sigma2_naive`** (up to 1e-9 tol).  violations=0  => PASS
**4. Beliefs sum to 1.**  L1 max|sum-1|=6.66e-16  L0 max|sum-1|=6.66e-16  => PASS
**5. `Sus(t=150)` mean under null** (pooled over theta*, alpha).
   - surp2: -0.0007
   - sus_1: +0.0000
   (expected close to 0; slight positive bias acceptable due to multiple-testing selection discussed in Group 1.)

## Per-alpha null Sus(t=150) means

```
        surp2    sus1
alpha                
1.0   -0.0019  0.0003
1.5   -0.0004  0.0011
2.0   -0.0009  0.0006
2.5    0.0009 -0.0003
3.0    0.0005  0.0018
4.0    0.0004  0.0002
5.0   -0.0024 -0.0009
7.0   -0.0023 -0.0019
10.0   0.0003  0.0008
15.0   0.0012  0.0004
20.0  -0.0025 -0.0018
```

## Dataset shape

- trajectories: 8,910,000 rows, 45 columns
- simulations: 59,400 rows
- unique cells: 297