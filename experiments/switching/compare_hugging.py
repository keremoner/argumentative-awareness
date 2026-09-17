"""
Does the adaptive S2 of Experiment B actually avoid the detector?

    python experiments/switching/compare_hugging.py

The Sus(t)-against-the-boundary figure in Experiment B looks like the speaker is
riding just under the boundary, but that figure averages over the runs that have
not yet switched, and conditioning on "has not crossed" selects exactly the runs
whose statistic stayed low.  The survivor mean must sit below the boundary
whatever the speaker does, so the figure cannot distinguish strategy from
survivorship.

This script uses a selection-free observable instead: the **unconditional
first-crossing rate** at matched (theta*, psi*, alpha, c).  Experiment B's
speaker models the switching listener and so could in principle avoid the
boundary; Experiment C1's speaker models a fixed always-vigilant L1 and has no
detector in its model at all.  If avoidance is real, B must cross less often
than C1.  Also reported, for the same cells:

* median tau among the runs that do cross (later tau = slower detection);
* the mean relative margin (Sus - bound)/bound over pre-switch rounds, which is
  the quantity the misleading figure was showing, so the two can be compared.

Writes results/switching/hugging_comparison.csv and a one-paragraph verdict to
stdout.  C1 must have been analysed first (it needs C1's tau_summary.parquet).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
OUT = os.path.join(_ROOT, "results", "switching")

KEYS = ["theta_star", "psi_star", "alpha", "c", "switch_type"]


def load_rates(study):
    p = os.path.join(_ROOT, "results", f"switching_{study}", "tau_summary.parquet")
    if not os.path.isfile(p):
        return None
    d = pd.read_parquet(p)
    g = d.groupby(KEYS).agg(
        rate=("switched", "mean"), n=("switched", "size"),
        median_tau=("tau", lambda s: float(s[s > 0].median()) if (s > 0).any() else np.nan),
    ).reset_index()
    g["study"] = study
    return g


def main():
    b, c1 = load_rates("B"), load_rates("C1")
    if b is None or c1 is None:
        print("need both results/switching_B and results/switching_C1 tau_summary.parquet")
        return 1
    # B carries only hard and soft; compare on the switch types both studies ran.
    common_st = sorted(set(b.switch_type) & set(c1.switch_type))
    b = b[b.switch_type.isin(common_st)]
    c1 = c1[c1.switch_type.isin(common_st)]
    m = b.merge(c1, on=KEYS, suffixes=("_B", "_C1"))
    if m.empty:
        print("no matched cells between B and C1 (grids differ); nothing to compare")
        return 1
    m["rate_diff"] = m["rate_B"] - m["rate_C1"]
    m["tau_diff"] = m["median_tau_B"] - m["median_tau_C1"]
    m.to_csv(os.path.join(OUT, "hugging_comparison.csv"), index=False)

    pers = m[m.psi_star != "inf"]
    print(f"matched cells: {len(m)} ({len(pers)} persuasive), switch types {common_st}\n")
    piv = pers.pivot_table(index=["c", "psi_star", "alpha"],
                           values=["rate_B", "rate_C1", "rate_diff", "median_tau_B", "median_tau_C1"])
    print("Persuasive cells, pooled over theta* and switch type:")
    print(piv.round(3).to_string())
    print("\nBy alpha, pooled further:")
    print(pers.groupby("alpha")[["rate_B", "rate_C1", "rate_diff", "median_tau_B", "median_tau_C1"]]
          .mean().round(3).to_string())

    d = pers["rate_diff"]
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"\nmean rate difference (B - C1) over persuasive cells: {d.mean():+.4f} "
          f"+/- {1.96*se:.4f} (95% CI over cells, n={len(d)})")
    if d.mean() + 1.96 * se < 0:
        verdict = ("B crosses LESS often than C1: consistent with the adaptive speaker "
                   "avoiding the detector.")
    elif d.mean() - 1.96 * se > 0:
        verdict = ("B crosses MORE often than C1: the adaptive speaker is detected sooner, "
                   "not later -- no evidence of avoidance, and some against it.")
    else:
        verdict = ("no detectable difference in crossing rate: the boundary-hugging seen in "
                   "B's Sus(t) figure is survivorship, not avoidance.")
    print("\nVerdict:", verdict)
    with open(os.path.join(OUT, "hugging_verdict.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"mean first-crossing rate difference (B - C1) over {len(d)} matched "
                 f"persuasive cells: {d.mean():+.4f} +/- {1.96*se:.4f} (95% CI over cells)\n")
        fh.write(verdict + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
