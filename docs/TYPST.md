# Typst in this project

Typst is the document toolchain for this project. LaTeX is no longer used for new
work. Typst is available from the VS Code editor through Tinymist and from a new
terminal as `typst`.

## Documents

- **`project.typ`** — the canonical project reference: the RSA model, the
  detection scores and their variance formulas, the sequential test, and a
  summary of every experiment. Start here.
- `legacy/` — superseded LaTeX and Typst sources, kept for provenance only.
  Their content is folded into `project.typ`; do not edit them.

## Building

Open a `.typ` document and use either:

- `Ctrl+Shift+P` -> `Typst: Preview current file` for a live preview.
- `Terminal` -> `Run Build Task` -> `Typst: Build current document` to create
  `build/typst/<document>.pdf`.
- `Terminal` -> `Run Task` -> `Typst: Watch current document` to rebuild on
  changes.

From a terminal directly:

```
typst compile --root . docs/project.typ build/typst/project.pdf
```

Generated PDFs live under `build/typst/` and are gitignored.

## Conventions

A few Typst gotchas this project has already hit:

- **Leave a space before a parenthesis after a subscript.** `k_t (u | O)` renders
  correctly; `k_t(u | O)` makes the subscript swallow the arguments, giving
  `k_{t(u|O)}`.
- **Use `frac(a, b)` when the denominator is a call.** `a / p_t (u)` puts `(u)`
  outside the fraction; `frac(a, p_t (u))` does not.
- **`~` is a non-breaking space in markup**, not "approximately". Write "about
  10%" or use `$approx$`. Inside `$...$` it is the tilde operator, which is fine.
- Angle brackets are `chevron.l` / `chevron.r` (not `angle.l`) as of Typst 0.15.
