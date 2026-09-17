"""
Before/after comparison for the sus_1 exact-variance fix.

Writes ``results/full_sweep/variance_fix_diff.md`` from two sweeps:

  old : results/full_sweep      (naive / corrected variance pair)
  new : results/full_sweep_v2   (single exact state-only variance)

The scores are identical in both -- the new sweep carries them over verbatim --
so every difference below is attributable to the variance alone.

Streams one cell shard at a time; peak memory is a few tens of MB.

Usage:
    python experiments/full_sweep/variance_fix_diff.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

C_VALUES = [2.0, 3.0, 3.5, 5.0, 7.0]
C_PER_ALPHA = 3.5
MIN_ALPHA = 1.5          # alpha=1 excluded, matching the published tables
T_MAX = 150
N_SIMS = 200

OLD_VARIANTS = {"naive": "sus1_sigma_bar2_naive",
                "corrected": "sus1_sigma_bar2_corrected"}


def _fired(sus, sigbar2, c, rounds):
    """Per-sim indicator: did Sus(t) > c*sigma_bar(t)/sqrt(t) ever fire?"""
    thr = c * np.sqrt(np.maximum(sigbar2, 0.0) / rounds)
    return (sus > thr).any(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_dir", default="results/full_sweep")
    ap.add_argument("--new_dir", default="results/full_sweep_v2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or os.path.join(args.old_dir, "variance_fix_diff.md")

    old_traj = os.path.join(args.old_dir, "trajectories")
    cells = sorted(int(n.split("=")[1]) for n in os.listdir(old_traj)
                   if n.startswith("cell="))
    rounds = np.arange(1, T_MAX + 1, dtype=float)

    # (1) operating characteristics: fired counts by (variant, c, psi)
    oc = {}
    # (2) per-alpha null crossing rate at C_PER_ALPHA
    pa = {}
    # (3) per-round calibration accumulators, keyed (variant, alpha, round)
    cal = {}
    # (4) old-proxy diagnostics per alpha
    diag = {}

    for cell in cells:
        o = pd.read_parquet(
            os.path.join(old_traj, f"cell={cell:04d}", "part.parquet"),
            columns=["psi_true", "alpha", "sus1_Sus", "sus1_score",
                     "sus1_sigma2_corrected", "sus1_sigma2_naive",
                     "sus1_sigma_bar2_naive", "sus1_sigma_bar2_corrected"])
        n = pd.read_parquet(
            os.path.join(args.new_dir, "trajectories", f"cell={cell:04d}",
                         "part.parquet"),
            columns=["sus1_Sus", "sus1_sigma2", "sus1_sigma_bar2"])

        alpha = float(o["alpha"].iloc[0])
        psi = str(o["psi_true"].iloc[0])
        if alpha < MIN_ALPHA:
            continue
        is_null = (psi == "inf")

        sus = o["sus1_Sus"].to_numpy().reshape(N_SIMS, T_MAX)
        panels = {k: o[v].to_numpy().reshape(N_SIMS, T_MAX)
                  for k, v in OLD_VARIANTS.items()}
        panels["exact"] = n["sus1_sigma_bar2"].to_numpy().reshape(N_SIMS, T_MAX)
        # Sus is carried over unchanged; assert rather than assume.
        assert np.array_equal(sus, n["sus1_Sus"].to_numpy().reshape(N_SIMS, T_MAX))

        for variant, sb2 in panels.items():
            for c in C_VALUES:
                f = _fired(sus, sb2, c, rounds)
                k = (variant, c, psi)
                a, b = oc.get(k, (0, 0))
                oc[k] = (a + int(f.sum()), b + f.size)
            if is_null:
                f = _fired(sus, sb2, C_PER_ALPHA, rounds)
                a, b = pa.get((variant, alpha), (0, 0))
                pa[(variant, alpha)] = (a + int(f.sum()), b + f.size)

        if is_null:
            score = o["sus1_score"].to_numpy().reshape(N_SIMS, T_MAX)
            per_round_var = {
                "naive": o["sus1_sigma2_naive"].to_numpy().reshape(N_SIMS, T_MAX),
                "corrected": o["sus1_sigma2_corrected"].to_numpy().reshape(N_SIMS, T_MAX),
                "exact": n["sus1_sigma2"].to_numpy().reshape(N_SIMS, T_MAX),
            }
            for variant, v in per_round_var.items():
                key = (variant, alpha)
                acc = cal.setdefault(key, np.zeros((4, T_MAX)))
                acc[0] += score.sum(axis=0)
                acc[1] += (score ** 2).sum(axis=0)
                acc[2] += v.sum(axis=0)
                acc[3] += score.shape[0]

            # (4) how much did the old per-round proxy move with the score?
            x = score.reshape(-1)
            y = per_round_var["corrected"].reshape(-1)
            d = diag.setdefault(alpha, np.zeros(7))
            d[0] += x.size
            d[1] += x.sum(); d[2] += y.sum()
            d[3] += (x * x).sum(); d[4] += (y * y).sum(); d[5] += (x * y).sum()
            d[6] += int((y <= 0).sum())

        del o, n

    # ---------------- render ----------------
    L = []
    L.append("# sus_1 variance fix: before / after")
    L.append("")
    L.append(f"- **old**: `{args.old_dir}` -- per-round variance "
             f"`var_naive(u_obs) - K`, an unbiased-in-expectation proxy that "
             f"depends on the utterance heard.")
    L.append(f"- **new**: `{args.new_dir}` -- exact "
             f"`V = sum_u p(u) s(u)^2`, a function of the listener state only.")
    L.append("")
    L.append("The two sweeps share their scores exactly (the new one carries "
             "them over verbatim and every row was re-derived and checked), so "
             "every difference below comes from the variance alone.")
    L.append("")
    L.append(f"All rates pool over `theta*` and `alpha >= {MIN_ALPHA}`.")
    L.append("")

    L.append("## 1. Operating characteristics")
    L.append("")
    L.append("| variance | c | FPR | TPR pers+ | TPR pers- |")
    L.append("|---|---|---|---|---|")
    for variant in ("naive", "corrected", "exact"):
        for c in C_VALUES:
            def rate(psi):
                a, b = oc.get((variant, c, psi), (0, 0))
                return a / b if b else float("nan")
            L.append(f"| {variant} | {c:g} | {rate('inf'):.3f} | "
                     f"{rate('pers+'):.3f} | {rate('pers-'):.3f} |")
    L.append("")

    L.append(f"## 2. Per-alpha null crossing rate at c = {C_PER_ALPHA:g}")
    L.append("")
    alphas = sorted({a for (_v, a) in pa})
    L.append("| variance | " + " | ".join(f"{a:g}" for a in alphas) + " |")
    L.append("|---" * (len(alphas) + 1) + "|")
    for variant in ("naive", "corrected", "exact"):
        cells_ = []
        for a in alphas:
            num, den = pa.get((variant, a), (0, 0))
            cells_.append(f"{num/den:.3f}" if den else "-")
        L.append(f"| {variant} | " + " | ".join(cells_) + " |")
    L.append("")

    L.append("## 3. Per-round calibration ratio  Var[s] / E[sigma^2]")
    L.append("")
    L.append("Empirical variance of the per-round score across null sims, over "
             "the mean reported variance, averaged across rounds. 1.00 is honest.")
    L.append("")
    L.append("| variance | " + " | ".join(f"{a:g}" for a in alphas) + " |")
    L.append("|---" * (len(alphas) + 1) + "|")
    for variant in ("naive", "corrected", "exact"):
        cells_ = []
        for a in alphas:
            acc = cal.get((variant, a))
            if acc is None:
                cells_.append("-")
                continue
            nobs = acc[3]
            mean = acc[0] / nobs
            emp = acc[1] / nobs - mean ** 2
            theo = acc[2] / nobs
            ratio = np.divide(emp, theo, out=np.full_like(emp, np.nan),
                              where=theo > 0)
            cells_.append(f"{np.nanmean(ratio):.3f}")
        L.append(f"| {variant} | " + " | ".join(cells_) + " |")
    L.append("")

    L.append("## 4. What the old per-round proxy was doing")
    L.append("")
    L.append("Measured on the **old** trajectories, null cells only: the "
             "correlation between the per-round score and the per-round "
             "`sigma2_corrected` that was supposed to scale it, and how often "
             "that variance came out non-positive.")
    L.append("")
    L.append("| alpha | corr(score, sigma2_corrected) | frac(sigma2_corrected <= 0) |")
    L.append("|---|---|---|")
    for a in alphas:
        d = diag.get(a)
        if d is None:
            continue
        nobs, sx, sy, sxx, syy, sxy, nneg = d
        cov = sxy / nobs - (sx / nobs) * (sy / nobs)
        vx = sxx / nobs - (sx / nobs) ** 2
        vy = syy / nobs - (sy / nobs) ** 2
        r = cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")
        L.append(f"| {a:g} | {r:+.3f} | {nneg / nobs:.3f} |")
    L.append("")
    L.append("A variance that correlates with its own numerator is not a "
             "scale factor; it is part of the statistic. The exact variance is "
             "constant across utterances within a round, so its correlation "
             "with the score is identically zero and it is never non-positive.")
    L.append("")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"wrote {out_path}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
