# Timing — Experiment C2 (S2 modelling a credulous L1, switching L2 detector, offline)

## Pilot (2026-09-11 10:22:55)
- study C2 mode offline speaker S2_cred, listener L2
- pilot cell: theta*=0.5 psi*=pers+ alpha=3.0, 10 sims x 150 rounds (switch rate 0.85,
  median tau 3.0)
- 0.585 s per sim (incl. 6 offline conditions)
- grid: 45 cells x 60 sims = 2700 sims  (reduced axes, same as B)
- projected: 26.3 CPU-min, 3.8 wall-min on 7 workers
## Run (2026-09-11 11:21:01)
- 45 cells run this invocation (0 resumed), 60 sims/cell, 7 workers
- actual wall: 7.9 min

## Run (2026-09-11T09:21:01Z)
- 45 cells, 60 sims/cell, 7 workers
- actual wall: 7.9 min

## Projected vs actual
- projected 3.8 wall-min, actual 7.9 wall-min (2.1x). Same cause as A
  and B: the pilot cell was psi*=pers+, which switches early and then costs little,
  while the psi*=inf cells run all 150 rounds unswitched.
