## 8. Flagged decisions, and what the grid was cut down to

**ASK-1 — informativeness of an honest S2 against the switching listener.** Implemented as
proposed: `Inf(u;O) = q_t(O|u)` taken from the *currently active* sub-listener before the
round's update, i.e. `DetectionListener.infer_obs` delegates to the naive listener before τ
and to the vigilant one after it, with no detour through `peek`. Routing informativeness
through `peek` as well would be more consistent, but it only ever matters for an honest S2
(the persuasive one has β=0 and never evaluates informativeness), and for an honest S2 the
two agree except on the single round where a candidate would trip the boundary. Not worth
the cost.

**ASK-2 — does the full sweep contain utterances?** It does: `results/full_sweep_v2` stores
`u_observed`, `O_true_idx` and `O_true_count` per round, so the trigger condition in the
spec ("if the parquet lacks utterances, regenerate") never fired. Experiment A regenerates
fresh S1 data regardless, for a reason the spec did not anticipate: the sweep was run with
the **11-point** θ space {0.0, 0.1, …, 1.0} as the agents' hypothesis grid, while these
experiments fix the **9-point** grid Θ = {0.1, …, 0.9}. The grid enters every likelihood the
speaker's internal L0 and the listener's L1 compute, so the stored streams were produced by
a different generative model and cannot be replayed under the 9-point grid. The sweep is
therefore used only as an external reference for first-crossing rates (section 1), where the
two grids agree to within sampling error.

**ASK-3 — caching the retrospective replay inside `peek`.** Not needed. A retro `peek` only
replays the history for candidates that would actually cross the boundary this round, and
before τ almost none do, so the replay branch is rarely taken. The Experiment B pilot timed
0.690 s per simulation without the cache and 0.681 s with it — within noise. The option is
implemented as `DetectionListener(retro_cache=True)` and tested to give bit-identical
results (`tests/test_switching.py::test_peek_retro_cache_gives_identical_results`), but it
is off in every run recorded here.

**Grid reductions against the 6-hour budget.** Pilots were timed before each study and the
numbers are in each study's `timing.md`. Experiment A came in at 0.50 s per simulation and
ran the full default grid (5 θ* × 3 ψ* × 4 α = 60 cells × 100 sims) in 14.7 wall-minutes,
far under its 1.5-hour target, so nothing was cut. Experiment B's single-cell pilot suggested
0.69 s per simulation, but the real grid ran at ≈1.38 s — the ψ*=inf cells rarely switch and
so pay for all 150 rounds of unswitched operation, which a pilot at ψ*=pers+ does not see —
putting the default grid at ~80 wall-minutes. Following the spec's shrink order, B was cut to
**60 simulations per cell** and **α ∈ {1.5, 3, 10}**, dropping α=5 on the evidence of A, where
α=5 and α=10 are nearly indistinguishable on every axis (switch rate 0.99 vs 1.00, credulous
|bias| at τ 0.17 vs 0.18, post-τ |bias| 0.033 vs 0.033). θ* keeps all five values and both c
and both switch types are kept, since those carry the headline contrasts. C1, C2 and C3 use
the same reduced axes. Nothing ran below the 40-simulation floor.

**Actual cost.** The five studies together took **78.5 wall-minutes** on 7 workers: A 14.7 (60 cells x 100 sims), B 41.2 (180 x 60), C1 8.4 (45 x 60), C2 7.9 (45 x 60), C3 6.2 (30 x 60), against the 6-hour budget. Every study overran its pilot projection by roughly 2x for the same reason, recorded in each `timing.md`: the pilot cell is psi*=pers+, which trips the detector early and then costs almost nothing, while the psi*=inf cells never switch and pay for all 150 rounds. A pilot at psi*=inf, or simply doubling the projection, would have sized these correctly first time.

**What is not comparable across studies.** A has 100 simulations per cell and four α values;
B, C1, C2 and C3 have 60 and three. B's α set is a subset of A's, so like-for-like cells can
be read across studies, but the α=5 column exists only in A. The C3 amnesic contrast cannot
be paired with B's retro and soft runs sim-by-sim: in feedback mode the utterance stream
depends on the switch type through the speaker's utility, so each condition is necessarily
its own set of simulations, and only the distributions are comparable.
