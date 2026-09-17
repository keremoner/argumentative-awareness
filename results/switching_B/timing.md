# Timing — Experiment B (S2 modelling the switching L1, feedback)

## Pilot (2026-09-11 06:24:29, repeated 06:24:47 with retro_cache)
- study B mode feedback speaker S2_replica
- pilot cell: theta*=0.5 psi*=pers+ alpha=3.0 c=3.5 switch=hard, 20 sims x 150 rounds
  (switch rate 1.00, median tau 11.0)
- 0.690 s per sim without the peek cache, 0.681 s with it -> ASK-3 cache not needed
- default grid: 240 cells (5 theta* x 3 psi* x 4 alpha x 2 c x 2 switch types) x 100 sims
- projected on the default grid: 275.9 CPU-min, 39.4 wall-min on 7 workers

## Observed cost on the real grid, and the shrink
- The default grid was launched at 06:41 and ran at ~1.38 s per sim, 2x the pilot: the
  psi*=inf cells rarely switch and so run all 150 rounds unswitched, which a pers+ pilot
  never sees. Re-projection: ~80 wall-min, too much to leave room for the C studies
  inside the session budget.
- Shrunk per the spec's order, and relaunched at 10:23:
  sims 100 -> 60, alpha {1.5, 3, 5, 10} -> {1.5, 3, 10}.
  alpha=5 was dropped on the evidence of A, where alpha=5 and alpha=10 are nearly
  identical on every axis (switch rate 0.99 vs 1.00, credulous |bias| at tau 0.17 vs
  0.18, post-tau |bias| 0.033 vs 0.033). All 5 theta*, both c and both switch types kept.
- reduced grid: 180 cells x 60 sims = 10800 sims
- projected on the reduced grid: 248 CPU-min, ~49 wall-min on 7 workers
## Run (2026-09-11 11:04:32)
- 180 cells run this invocation (0 resumed), 60 sims/cell, 7 workers
- actual wall: 41.2 min

## Run (2026-09-11 11:04:32)
- 180 cells run this invocation (0 resumed), 60 sims/cell, 7 workers
- actual wall: 41.2 min

## Projected vs actual
- projected 49 wall-min on the reduced grid, actual 41.2 wall-min. The default
  grid would have taken ~80 min; the shrink brought it inside budget with room
  for C1, C2 and C3.
