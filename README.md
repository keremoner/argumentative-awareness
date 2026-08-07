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
python -m pytest tests/ -q                          # 15 tests, ~8 s
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
| `python experiments/full_sweep/run_sweep.py` | `results/full_sweep/` (9 θ × 11 α × 3 ψ, 200 sims × 150 rounds) | ~33 min |
| `python experiments/group1/run_sims.py` | `results/group1/` | a few min |
| `python experiments/sus_variants/run_sims.py` | `results/sus_variants/` | a few min |
| `python experiments/run_detection_comparison.py --mode full` | `results/detection_comparison/` | a few min |

The first three take `--smoke` for a fast throwaway run into `results/*_smoke/`;
`run_detection_comparison.py` instead defaults to `--mode pilot`, with `--mode full` for
the real thing. `run_sweep.py` also takes `--n_sims`, `--rounds`,
`--only_alpha/--only_theta/--only_psi` and `--workers`, and resumes by default if
interrupted (`--no_resume` to force a rerun).

> **Note:** `results/full_sweep/` and `results/sus_variants/` as last generated *predate*
> the `sus_1` variance clip fix, so their `sigma2_corrected` is inflated at α ≤ 2. Re-run
> before quoting low-α corrected numbers.

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
rsa/            model: agents, semantics, detection scores, sequential test
experiments/    runnable studies, one directory per study
notebooks/      analysis notebooks (variance_diagnostics.ipynb is the current one)
results/        per-study outputs; parquet is gitignored, summaries are tracked
docs/           project.typ (canonical), TYPST.md, legacy/
tests/          pytest suite
archive/        pre-refactor exploratory work, kept for reference only
```
