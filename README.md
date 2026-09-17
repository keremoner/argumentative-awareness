# Argumentative awareness

Can a *credulous* RSA listener notice, on its own, that it is being talked to by a
persuasive speaker? The base model is Ke Fang's *A Computational Account of Epistemic
Vigilance*, which compares a credulous listener against an already-vigilant one. This
project asks the question that comes before that: keep the cheap credulous model,
accumulate a per-round statistic of how badly it predicts what it hears, and alarm when
the evidence crosses a boundary.

Start with **`docs/project.typ`** (build to PDF, see below) — it covers the model, the
detection scores, and the current findings, and ends with a repository map and a
symbol-to-code cross-reference.

## Setup

```bash
git clone https://github.com/keremoner/argumentative-awareness.git
cd argumentative-awareness
python -m venv .venv && .venv/Scripts/activate      # Windows; use bin/activate on POSIX
pip install -r requirements.txt
python -m pytest tests/ -q                          # 126 tests, ~70 s
```

## Regenerating the data

**Raw simulation output is deliberately not in git.** It is ~2.4 GB of parquet and is
fully reproducible: every run is seeded by `_seed_for(cell_id, sim_idx, seed_base)`, and
each output directory's `run_config.json` records `seed_base` and the `git_hash` it was
produced under. What *is* tracked is the small stuff — configs, `sanity.md`, `summary.md`,
figures and CSVs — so conclusions and provenance travel with the code.

After a fresh clone, `results/` will contain those summaries but no parquet. Regenerate
whichever you need:

| Command | Output | Approx. time |
|---|---|---|
| `python experiments/full_sweep/run_sweep.py --out_dir results/full_sweep_v2` | `results/full_sweep_v2/` (9 θ × 11 α × 3 ψ, 200 sims × 150 rounds) | ~100 min on 7 workers |
| `python experiments/full_sweep/recompute_variance.py` | rebuilds the variance columns of an existing sweep without re-simulating | ~8 min |
| `python experiments/group1/run_sims.py` | `results/group1/` | a few min |
| `python experiments/sus_variants/run_sims.py` | `results/sus_variants/` | a few min |
| `python experiments/run_detection_comparison.py --mode full` | `results/detection_comparison/` | a few min |
| `bash experiments/switching/run_all.sh` | `results/switching_{B,C1,C2,C3}/` (the switching studies, in order) | ~55 min on 7 workers |
| `python experiments/switching/run.py --grid experiments/switching/grids/A.json` | `results/switching_A/` | ~15 min on 7 workers |

### The switching studies

`experiments/switching/` has one runner and one JSON grid per study, so the grid is data
rather than code:

```bash
python experiments/switching/run.py --grid experiments/switching/grids/A.json --pilot 20
python experiments/switching/run.py --grid experiments/switching/grids/A.json
python experiments/switching/analyze.py --study A
python experiments/switching/report.py        # rebuilds results/switching/report.md
```

`--pilot N` times one cell with `N` simulations and extrapolates to the whole grid,
writing the projection to that study's `timing.md`; run it before launching anything.
`--n_sims`, `--rounds`, `--workers` and `--out_dir` override the grid from the command
line, and runs resume by default. Study A (S1 speaker) derives every (c, switch type)
condition offline from one utterance stream per simulation; study B (S2 speaker modelling
the switching listener) cannot, so each condition is its own set of simulations.
Findings prose lives in `experiments/switching/findings_*.md` and is inserted verbatim
into the report.

The sweep and the two `run_sims.py` studies take `--smoke` for a fast throwaway run
into `results/*_smoke/`;
`run_detection_comparison.py` instead defaults to `--mode pilot`, with `--mode full` for
the real thing. `run_sweep.py` also takes `--n_sims`, `--rounds`,
`--only_alpha/--only_theta/--only_psi` and `--workers`, and resumes by default if
interrupted (`--no_resume` to force a rerun).

### Interrupting and resuming the full sweep

The sweep is built to survive being killed. One task is one grid cell; the worker
writes that cell's parquet shard itself, so the parent process stays at ~130 MB no
matter how far the run has got, and the whole tree peaks around 1.2 GB on 7 workers.
Shards are written to a `.tmp` sibling and atomically renamed, so a shard is either
absent or complete — never half-written.

| Situation | What to do |
|---|---|
| Killed / crashed / Ctrl-C | Rerun the same command. Finished cells are skipped. |
| Want to check what survived | `--verify_only` — footer-only integrity pass, exits 1 on any problem |
| Want sanity checks without rerunning | `--sanity_only` — streams the dataset shard by shard |
| RAM is tight | lower `--workers`, or `--max_inflight`; `--min_free_gb` (default 1.0) stops submitting new cells and exits cleanly rather than being OOM-killed |
| Resume feels slow to start | `--fast_resume` trusts existing shards instead of checking row counts |

Progress is appended to `results/full_sweep/progress.jsonl` (one fsync'd line per
finished cell), and `run_config.json` is written *before* work starts with
`"status": "running"`, then rewritten at the end as `complete`, `partial` (some cells
errored) or `incomplete` (guard tripped or interrupted). A run that did not finish
exits non-zero.

> **Note (2026-09-09): the `sus_1` null variance changed.** `sus_1` now reports
> the exact state-only variance `V = sum_u p(u) s(u)^2` instead of the per-round
> proxy `var_naive(u_obs) - K`. The proxy was unbiased in expectation but varied
> round to round with the utterance heard, correlated with the score it scaled
> (rho = -0.27 at alpha = 1.5) and went non-positive on 35% of rounds. Score
> functions now return `(score, var)` and `SequentialTest` clips nothing.
>
> - `results/full_sweep_v2/` is the current sweep. Its `sus1_sigma2` /
>   `sus1_sigma_bar2` columns replace the `*_naive` / `*_corrected` pairs, and
>   its analyses live in `results/full_sweep_v2/analyses/`.
> - `results/full_sweep/` is kept as the pre-fix record, and
>   `results/full_sweep/variance_fix_diff.md` compares the two.
> - `results/group1/` and `results/sus_variants/` were re-run; their summaries
>   carry a note describing what moved.
>
> The fix removed two effects previously believed real: `sus_1` had appeared to
> have an irreducible false-alarm floor near 6%, and a U-shaped false-alarm rate
> in alpha. Both were artefacts of the proxy.

## Documentation

`docs/project.typ` builds with [Typst](https://typst.app/):

```bash
typst compile --root . docs/project.typ build/typst/project.pdf
typst watch  --root . docs/project.typ build/typst/project.pdf   # live preview
```

See `docs/TYPST.md` for conventions and gotchas. LaTeX is fully superseded; the old
sources are kept for reference in `docs/legacy/`.

## Layout

```
rsa/            model: agents, semantics, detection scores, sequential test, switching + replay
experiments/    runnable studies, one directory per study
notebooks/      analysis notebooks (variance_diagnostics.ipynb is the current one)
results/        per-study outputs; parquet is gitignored, summaries are tracked
                (results/switching/report.md is the switching write-up)
docs/           project.typ (canonical), TYPST.md, legacy/
tests/          pytest suite
archive/        pre-refactor exploratory work, kept for reference only
```
