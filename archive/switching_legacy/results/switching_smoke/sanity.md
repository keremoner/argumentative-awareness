# sanity.md

Automated checks from `run_switching.py`.
**Overall: PASS**


**1. Row count.** expected=35,640 actual=35,640  => PASS
**2. Theta marginals finite.** non-finite=0  => PASS
**3. Theta marginals sum to 1** (1e-6). violations={'cred': 0, 'vig': 0, 'soft': 0, 'hard': 0}  => PASS
**4. Psi marginals sum to 1 where present.** violations={'vig': 0, 'soft': 0, 'hard': 0}  => PASS
**5. `switched` flag consistent with psi presence.** mismatches=0  => PASS
**6. Before tau, soft == hard == credulous exactly.** mismatches=0  => PASS
**7. Switch rate** at c=3.5: null 0.030 (12/396), persuasive 0.662 (524/792). Compare the sweep's FPR/TPR at this c.
**8. Credulous trajectory == sweep `L1_theta` on shared seeds.** cells checked=297 mismatching=0 max|diff|=0.00e+00  => PASS

- trajectories: 35,640 rows across 297 shards