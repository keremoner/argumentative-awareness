# Legacy switching experiment (superseded, do not mix with results/switching_*)

Moved here on 2026-09-11 before the switching experiments were re-implemented
from `switching_experiments_spec.md`.

* `run_switching.py` -- the previous runner (four live policies per stream,
  `hard` = amnesic hard switch, no retrospective replay).
* `test_switching_legacy.py` -- its tests. Note `test_hard_switch_starts_from_uniform_joint`
  and `test_manual_fork_at_tau_equals_hard_switch_listener` describe the *old*
  `"hard"` behaviour, which is now called `switch_type="hard_amnesic"`.
* `results/switching_c3.5/` -- an aborted full-grid run (84 of 297 cells,
  `run_config.json` still says `"status": "running"`; the process was killed
  at 05:58 on 2026-09-11 and no process is running). Its `hard` policy is the
  amnesic one. Parquet shards are git-ignored.
* `results/switching_smoke/` -- the smoke run of the same runner.

The new experiments write to `results/switching_A/`, `results/switching_B/`,
`results/switching_C*/` and `results/switching/report.md`, and use the
retrospective hard switch (`switch_type="hard"`).
