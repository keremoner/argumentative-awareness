# Timing — Experiment C1 (S2 modelling Fang's vigilant L1-strat, offline)

## Pilot (2026-09-11 10:22:44)
- study C1 mode offline speaker S2_vig
- pilot cell: theta*=0.5 psi*=pers+ alpha=3.0, 10 sims x 150 rounds (switch rate 1.00,
  median tau 10.5)
- 0.521 s per sim (incl. 6 offline conditions)
- grid: 45 cells x 60 sims = 2700 sims  (reduced axes, same as B)
- projected: 23.4 CPU-min, 3.3 wall-min on 7 workers
## Run (2026-09-11 11:13:01)
- 45 cells run this invocation (0 resumed), 60 sims/cell, 7 workers
- actual wall: 8.4 min

## Run (2026-09-11T09:13:01Z)
- 45 cells, 60 sims/cell, 7 workers
- actual wall: 8.4 min

## Projected vs actual
- projected 3.3 wall-min, actual 8.4 wall-min (2.6x). Same cause as A
  and B: the pilot cell was psi*=pers+, which switches early and then costs little,
  while the psi*=inf cells run all 150 rounds unswitched.
