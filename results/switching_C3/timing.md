# Timing — Experiment C3 (hard_amnesic contrast in the feedback setting, alpha=3)

## Pilot
- No separate pilot: C3 is Experiment B's feedback machinery with switch_type
  hard_amnesic at a single alpha, so B's measured 1.38 s per sim applies.
- grid: 30 cells (5 theta* x 3 psi* x 1 alpha x 2 c x 1 switch type) x 60 sims = 1800 sims
- projected: 41 CPU-min, ~6 wall-min on 7 workers
## Run (2026-09-11 11:27:18)
- 30 cells run this invocation (0 resumed), 60 sims/cell, 7 workers
- actual wall: 6.2 min

## Run (2026-09-11T09:27:18Z)
- 30 cells, 60 sims/cell, 7 workers
- actual wall: 6.2 min

## Projected vs actual
- projected 6.0 wall-min, actual 6.2 wall-min (1.0x). Same cause as A
  and B: the pilot cell was psi*=pers+, which switches early and then costs little,
  while the psi*=inf cells run all 150 rounds unswitched.
