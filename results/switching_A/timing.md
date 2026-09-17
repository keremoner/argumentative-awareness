# Timing — Experiment A (S1 speaker, switching L1, offline conditions)

## Pilot (2026-09-11 06:24:10)
- study A mode offline speaker S1
- pilot cell: theta*=0.5 psi*=high alpha=3.0, 20 sims x 150 rounds (switch rate 1.00, median tau 10.0)
- 0.504 s per sim (incl. 6 offline conditions: 2 c x 3 switch types)
- grid: 60 cells x 100 sims = 6000 sims
- projected: 50.4 CPU-min, 7.2 wall-min on 7 workers
- decision: full default grid, no shrink (target for A was 1.5 h)

## Run (2026-09-11 06:40:37)
- 60 cells run this invocation (0 resumed), 100 sims/cell, 7 workers
- actual wall: 14.7 min

## Projected vs actual
- projected 7.2 wall-min, actual 14.7 wall-min (2.0x).
- The pilot cell (psi*=pers+, alpha=3) switches in every run at median tau 10, after
  which the offline conditions are cheap splices. The psi*=inf cells rarely switch and
  pay for all 150 rounds, so a pers+ pilot systematically under-times the grid. Both
  numbers are far inside the budget, so nothing was re-sized.
