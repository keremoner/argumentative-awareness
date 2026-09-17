"""
Build results/switching/report.md from the analysis CSVs of the studies that
have been run.  Plain-language findings per study are read from
``experiments/switching/findings_<study>.md`` when present (written by hand
after inspecting the analysis) and inserted verbatim.

    python experiments/switching/report.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
OUT = os.path.join(_ROOT, "results", "switching")
PSI_TXT = {"inf": "inf", "high": "pers+", "low": "pers−"}
L_TXT = {"cred": "credulous", "vig": "always-vigilant", "sw": "switching", "switch": "switching"}
ST_TXT = {"hard": "retro", "soft": "soft", "hard_amnesic": "amnesic"}


# Columns that read better without three decimals: grid coordinates and round counts.
_SHORT = {"c", "alpha", "theta_star", "n", "n_switched", "bin",
          "median_tau", "mean_tau", "recovery_median", "recovery_mean",
          "median_tau_B", "median_tau_C1"}


def _num(v, short=False):
    if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
        return "nan"
    if isinstance(v, (float, np.floating)):
        if short:
            return f"{v:g}" if float(v).is_integer() else f"{v:.1f}"
        return f"{v:.3f}"
    return str(v)


def md_table(df, cols, fmt=None, rename=None):
    fmt = fmt or {}
    rename = rename or {}
    head = [rename.get(c, c) for c in cols]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt:
                cells.append(fmt[c](v))
            else:
                cells.append(_num(v, short=c in _SHORT))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def rel(study, name):
    return f"../switching_{study}/analysis/{name}"


def fig(study, name, caption):
    return f"![{caption}]({rel(study, name)})\n\n*{caption}*\n"


def read(study, name):
    p = os.path.join(_ROOT, "results", f"switching_{study}", "analysis", name)
    return pd.read_csv(p) if os.path.isfile(p) else None


def findings(study):
    p = os.path.join(_HERE, f"findings_{study}.md")
    return open(p, encoding="utf-8").read().strip() if os.path.isfile(p) else "_(findings pending)_"


def timing(study):
    p = os.path.join(_ROOT, "results", f"switching_{study}", "timing.md")
    return open(p, encoding="utf-8").read().strip() if os.path.isfile(p) else ""


def cfg(study):
    p = os.path.join(_ROOT, "results", f"switching_{study}", "run_config.json")
    return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else None


def section_setup(studies):
    out = ["## 0. Setup and provenance\n"]
    out.append("Common settings: n=1, m=7, θ-grid {0.1,…,0.9} (9 points), 150 rounds, score `sus_1` with its exact "
               "state-only variance, boundary Sus(t) > c·σ̄(t)/√t, ψ* ∈ {inf, pers+ (code `high`), pers− (code `low`)}.\n")
    rows = []
    for s in studies:
        c = cfg(s)
        if c is None:
            continue
        g = c["grid"]
        rows.append(dict(study=s, mode=c["mode"], speaker=c["speaker_level"], listener=c["listener_level"],
                         theta=str(g["theta_stars"]), alpha=str(g["alphas"]), c=str(g["cs"]),
                         switch=str(g["switch_types"]), sims=g["n_sims_per_cell"], cells=g["n_cells"],
                         wall_min=f"{c['wall_seconds']/60:.1f}", git=c["git_hash"][:8], seed_base=c["seed_base"]))
    out.append(md_table(pd.DataFrame(rows), ["study", "mode", "speaker", "listener", "theta", "alpha", "c", "switch",
                                             "sims", "cells", "wall_min", "git", "seed_base"]))
    out.append("\nSeeds: `seed = (seed_base + cell_index·1000003 + sim_index) mod 2^31` (recorded per row). "
               "Every dataset stores `obs` and `utt` per round, so any listener can be replayed offline "
               "(`rsa/detection/replay.py`).\n")
    for s in studies:
        t = timing(s)
        if t:
            out.append(f"<details><summary>timing.md ({s})</summary>\n\n{t}\n\n</details>\n")
    return "\n".join(out)


def section_sanity():
    out = ["## 1. Sanity check: first-crossing rates vs. the full sweep\n"]
    cmp = read("A", "A1_sweep_comparison.csv")
    pooled = read("A", "A1_rates_pooled.csv")
    if cmp is None:
        return "\n".join(out + ["_Experiment A not analysed yet._"])
    out.append("Rates are the fraction of simulations whose `sus_1` running mean crosses the boundary within 150 rounds, "
               "pooled over θ* ∈ {0.1,0.3,0.5,0.7,0.9}. The sweep reference (`results/full_sweep_v2`, 200 sims/cell) is "
               "recomputed on the same (θ*, α) cells but was generated on the 11-point θ-grid {0.0,…,1.0}, so exact "
               "agreement is not expected.\n")
    p = pooled[pooled.switch_type == pooled.switch_type.iloc[0]]
    ref = pd.read_csv(os.path.join(OUT, "sweep_reference_rates.csv"))
    ref["psi_star"] = ref["psi"].map({"inf": "inf", "pers+": "high", "pers-": "low"})
    rp = ref.groupby(["c", "psi_star"]).agg(rate_sweep=("rate", "mean")).reset_index()
    p = p.merge(rp, on=["c", "psi_star"], how="left")
    p["psi_star"] = p["psi_star"].map(PSI_TXT)
    out.append(md_table(p.sort_values(["c", "psi_star"]), ["c", "psi_star", "rate", "rate_sweep", "median_tau"],
                        rename={"rate": "rate (A)", "rate_sweep": "rate (sweep)", "median_tau": "median τ (A)"}))
    out.append("\nPer α:\n")
    cmp["psi_star"] = cmp["psi_star"].map(PSI_TXT)
    piv = cmp.pivot_table(index=["c", "psi_star"], columns="alpha", values=["rate", "rate_sweep"]).round(3)
    out.append("```\n" + piv.to_string() + "\n```\n")
    out.append(fig("A", "A1_rates_vs_alpha.png", "First-crossing rate vs α, this study (green) against the sweep (grey)."))
    out.append("The comparison applies only to the S1 studies. The sweep was generated by an S1 speaker, so "
               "for the S2 studies below the crossing rate measures level mismatch rather than goal detection, "
               "and no reference series is drawn.\n")
    return "\n".join(out)


def section_offline(study, title):
    out = [f"## {title}\n"]
    c = cfg(study)
    if c is None or read(study, f"{study}1_rates_pooled.csv") is None:
        return "\n".join(out + ["_not run_"])
    out.append(findings(study) + "\n")
    sp = os.path.join(_ROOT, "results", f"switching_{study}", "analysis", f"{study}0_splice_check.txt")
    if os.path.isfile(sp):
        out.append("### Splice identity check\n")
        out.append("```\n" + open(sp, encoding="utf-8").read().strip() + "\n```\n")
    out.append("### τ distribution\n")
    out.append(fig(study, f"{study}1_tau_hist_c3.5.png", "First-crossing time τ at c=3.5 (rows ψ*, columns α)."))
    out.append("### Belief trajectories (Fang-style panels)\n")
    for cc in c["grid"]["cs"]:
        for a in (3.0, 10.0):
            if a in c["grid"]["alphas"]:
                out.append(fig(study, f"{study}2_panels_c{cc}_alpha{a}.png",
                               f"E[θ] by round, rows ψ*, columns θ*; α={a}, c={cc}. Red dashed = θ*."))
    out.append("Other (c, α) panels: " + ", ".join(
        f"[c={cc}, α={a}]({rel(study, f'{study}2_panels_c{cc}_alpha{a}.png')})"
        for cc in c["grid"]["cs"] for a in c["grid"]["alphas"]) + "\n")
    out.append("### |bias| and std over rounds\n")
    for cc in c["grid"]["cs"]:
        out.append(fig(study, f"{study}2_bias_c{cc}.png", f"|E[θ]−θ*| by round, mean over θ* and sims, c={cc}."))
        out.append(fig(study, f"{study}2_std_c{cc}.png", f"std[θ] by round, mean over θ* and sims, c={cc}."))
    avg = read(study, f"{study}2_bias_avg.csv")
    if avg is not None:
        t = avg[avg.c == 3.5].copy()
        t["who"] = np.where(t.listener == "switch", "switch-" + t.switch_type.map(ST_TXT), t.listener.map(L_TXT))
        t = t.drop_duplicates(subset=["who", "psi_star", "alpha"])
        t["psi_star"] = t["psi_star"].map(PSI_TXT)
        piv = t.pivot_table(index=["psi_star", "alpha"], columns="who", values="mean_abias").round(3)
        out.append("Mean |E[θ]−θ*| over all 150 rounds (c=3.5, averaged over θ*):\n")
        out.append("```\n" + piv.to_string() + "\n```\n")
        piv2 = t.pivot_table(index=["psi_star", "alpha"], columns="who", values="abias_r150").round(3)
        out.append("|E[θ]−θ*| at round 150 (c=3.5):\n")
        out.append("```\n" + piv2.to_string() + "\n```\n")
    out.append("### Cost until τ and recovery\n")
    cr = read(study, f"{study}3_cost_recovery.csv")
    if cr is not None:
        t = cr.copy(); t["psi_star"] = t["psi_star"].map(PSI_TXT); t["switch_type"] = t["switch_type"].map(ST_TXT)
        out.append("Recovery = rounds after τ until |bias| of the switching listener ≤ |bias| of the always-vigilant "
                   "listener at the same round + 0.02 (per sim; median over sims that switched; `recovery_frac` = share "
                   "recovered within the horizon). For retro it is 0 by construction.\n")
        out.append(md_table(t.sort_values(["c", "psi_star", "alpha", "switch_type"]),
                            ["c", "psi_star", "alpha", "switch_type", "rate", "median_tau", "abias_cred_at_tau",
                             "bias_cred_at_tau", "recovery_median", "recovery_frac", "mean_abias_sw_post",
                             "mean_abias_cred_post", "mean_abias_vig_post"],
                            rename={"rate": "switch rate", "median_tau": "median τ",
                                    "abias_cred_at_tau": "|bias| cred at τ", "bias_cred_at_tau": "bias cred at τ",
                                    "recovery_median": "recovery (median rounds)", "recovery_frac": "recovery frac",
                                    "mean_abias_sw_post": "mean |bias| switch, t≥τ",
                                    "mean_abias_cred_post": "mean |bias| cred, t≥τ",
                                    "mean_abias_vig_post": "mean |bias| vig, t≥τ"}))
        out.append(fig(study, f"{study}3_cost_recovery.png", "Cost until τ and recovery vs α."))
    out.append("### False-alarm cost (ψ* = inf, conditional on having switched)\n")
    fa = read(study, f"{study}4_false_alarm_cost.csv")
    if fa is not None:
        t = fa.copy(); t["switch_type"] = t["switch_type"].map(ST_TXT)
        out.append(md_table(t.sort_values(["c", "alpha", "switch_type"]),
                            ["c", "alpha", "switch_type", "rate", "n_switched", "median_tau", "abias_cred_r150",
                             "abias_vig_r150", "abias_sw_r150", "std_cred_150", "std_sw_150",
                             "mean_abias_cred_post", "mean_abias_sw_post"],
                            rename={"rate": "false-alarm rate", "abias_cred_r150": "|bias| cred @150",
                                    "abias_vig_r150": "|bias| vig @150", "abias_sw_r150": "|bias| switch @150",
                                    "std_cred_150": "std cred @150", "std_sw_150": "std switch @150",
                                    "mean_abias_cred_post": "mean |bias| cred, t≥τ", "mean_abias_sw_post": "mean |bias| switch, t≥τ"}))
        for cc in c["grid"]["cs"]:
            out.append(fig(study, f"{study}4_false_alarm_c{cc}.png", f"ψ*=inf, sims that switched: |bias| by round, c={cc}."))
    out.append("### Utterance frequency by round\n")
    out.append(fig(study, f"{study}5_utt_freq.png", "Utterance share by round (rows ψ*, columns α), pooled over θ*."))
    sf = read(study, f"{study}5_some_freq.csv")
    if sf is not None:
        t = sf.copy(); t["psi_star"] = t["psi_star"].map(PSI_TXT)
        piv = t.pivot_table(index=["psi_star", "alpha"], columns="block", values="f_some").round(3)
        out.append("Share of rounds using a `some` utterance, by round block:\n")
        out.append("```\n" + piv.to_string() + "\n```\n")
    return "\n".join(out)


def section_feedback(study, title):
    out = [f"## {title}\n"]
    c = cfg(study)
    if c is None or read(study, f"{study}1_rates_pooled.csv") is None:
        return "\n".join(out + ["_not run_"])
    out.append(findings(study) + "\n")
    rp = os.path.join(_ROOT, "results", f"switching_{study}", "analysis", f"{study}0_replica_check.txt")
    if os.path.isfile(rp):
        out.append("Replica check (S2's internal listener vs. the actual listener): " + open(rp).read().strip() + "\n")
    out.append("### Belief trajectories\n")
    for cc in c["grid"]["cs"]:
        for a in (3.0, 10.0):
            if a in c["grid"]["alphas"]:
                out.append(fig(study, f"{study}2_panels_c{cc}_alpha{a}.png",
                               f"E[θ] by round, rows (ψ*, switch type), columns θ*; α={a}, c={cc}."))
    out.append("Other panels: " + ", ".join(
        f"[c={cc}, α={a}]({rel(study, f'{study}2_panels_c{cc}_alpha{a}.png')})"
        for cc in c["grid"]["cs"] for a in c["grid"]["alphas"]) + "\n")
    for cc in c["grid"]["cs"]:
        out.append(fig(study, f"{study}2_bias_c{cc}.png", f"|E[θ]−θ*| by round, mean over θ* and sims, c={cc}."))
    out.append("### Per-round speaker choice (trade-off)\n")
    out.append("`went persuasive` = the chosen utterance is the top choice of a Fang-S2 modelling a *credulous* L1 "
               "and not the top choice of S2-inf; `went informative` = the reverse; the two references coincide on "
               "a large share of rounds (grey in the CSV). Margin = (Sus(t−1) − boundary(t−1)) / boundary(t−1), "
               "pre-switch rounds only.\n")
    for cc in c["grid"]["cs"]:
        out.append(fig(study, f"{study}2_choice_vs_round_c{cc}.png", f"Choice by round, c={cc}."))
        out.append(fig(study, f"{study}2_choice_vs_margin_c{cc}.png", f"Choice vs. distance to the boundary, c={cc}."))
        out.append(fig(study, f"{study}2_policy_mass_c{cc}.png",
                       f"Policy probability on each reference utterance, c={cc}. The continuous "
                       f"version of the same trade-off, and the readable one at low α where the "
                       f"speaker's softmax is too diffuse for an argmax to mean much."))
    cb = read(study, f"{study}2_choice_by_block.csv")
    if cb is not None:
        t = cb[cb.c == 3.5].copy(); t["psi_star"] = t["psi_star"].map(PSI_TXT); t["switch_type"] = t["switch_type"].map(ST_TXT)
        out.append("Fractions by round block (c=3.5):\n")
        out.append(md_table(t, ["psi_star", "alpha", "switch_type", "block", "f_pers_only", "f_inf_only", "f_both", "p_switched"],
                            rename={"f_pers_only": "went persuasive", "f_inf_only": "went informative",
                                    "f_both": "pers = inf", "p_switched": "P(switched)"}))
    out.append("### Sus(t) against the boundary, retro vs soft\n")
    out.append(fig(study, f"{study}3_sus_vs_boundary.png", "Running mean Sus(t) (thin: 12 sims; thick: mean over sims not yet switched) and the boundary; θ*=0.5, c=3.5."))
    out.append("### Achieved persuasion\n")
    pt = read(study, f"{study}4_persuasion.csv")
    if pt is not None:
        t = pt.copy()
        t["who"] = np.where(t.listener == "switch", "switch-" + t.switch_type.map(ST_TXT),
                            t.listener.map(L_TXT) + " (" + t.switch_type.map(ST_TXT) + " sims)")
        t = t[t.psi_star != "inf"]; t["psi_star"] = t["psi_star"].map(PSI_TXT)
        for r in ("r25", "r50", "r100", "r150"):
            piv = t.pivot_table(index=["c", "psi_star", "alpha"], columns="who", values=r).round(3)
            out.append(f"|E[θ]−θ*| of the actual listener at round {r[1:]} (averaged over θ* and sims):\n")
            out.append("```\n" + piv.to_string() + "\n```\n")
        out.append(fig(study, f"{study}4_persuasion.png", "Achieved persuasion at round 150 (bars) and 50 (ticks)."))
    out.append("### Level-mismatch alarm rate under S2-inf\n")
    mm = read(study, f"{study}5_mismatch_alarm.csv")
    if mm is not None:
        t = mm.copy(); t["switch_type"] = t["switch_type"].map(ST_TXT)
        out.append("The actual listener models S1-inf; the honest speaker is an S2-inf. These are not false-alarm "
                   "rates of a correctly specified test: they are *level-mismatch* rates.\n")
        out.append(md_table(t.sort_values(["c", "switch_type", "alpha"]), ["c", "switch_type", "alpha", "rate", "median_tau", "n"],
                            rename={"rate": "alarm rate", "median_tau": "median τ"}))
        out.append(fig(study, f"{study}5_mismatch_alarm.png", "Level-mismatch alarm rate vs α."))
    return "\n".join(out)


def section_hugging():
    """The selection-free boundary-avoidance test (B against C1)."""
    v = os.path.join(OUT, "hugging_verdict.txt")
    if not os.path.isfile(v):
        return ""
    out = ["## 7. Does the adaptive speaker avoid the detector?\n",
           "Experiment B's Sus(t) figure appears to show the speaker riding just under the "
           "boundary, but it averages over the runs that have not yet switched, and that "
           "conditioning forces the survivor mean below the boundary whatever the speaker does. "
           "The selection-free test is the unconditional first-crossing rate at matched "
           "(θ*, ψ*, α, c) against Experiment C1, whose speaker models a fixed vigilant L1 and "
           "has no detector in its model. Run by `experiments/switching/compare_hugging.py`.\n",
           "```\n" + open(v, encoding="utf-8").read().strip() + "\n```\n"]
    p = os.path.join(OUT, "hugging_comparison.csv")
    if os.path.isfile(p):
        d = pd.read_csv(p)
        d = d[d.psi_star != "inf"]
        t = d.groupby("alpha")[["rate_B", "rate_C1", "rate_diff", "median_tau_B", "median_tau_C1"]].mean().reset_index()
        out.append(md_table(t, ["alpha", "rate_B", "rate_C1", "rate_diff", "median_tau_B", "median_tau_C1"],
                            rename={"rate_B": "crossing rate B", "rate_C1": "crossing rate C1",
                                    "rate_diff": "B − C1", "median_tau_B": "median τ B",
                                    "median_tau_C1": "median τ C1"}))
        out.append("\nPooled over persuasive cells only. `peek` shows the speaker the consequence of "
                   "tripping the detector but it maximises a one-round utility, so it cannot trade a "
                   "loss now for staying undetected later; evasion would need a speaker that plans "
                   "over the horizon.\n")
    return "\n".join(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    parts = ["# Switching experiments — report\n",
             "Detector-triggered switching from a credulous L1 to a vigilant L1 (retrospective `hard`, `soft`, "
             "and the amnesic contrast), against S1 and S2 speakers. Spec: `switching_experiments_spec.md`. "
             "Code: `rsa/detection/listener.py`, `rsa/detection/replay.py`, `rsa/speaker2.py`, "
             "`experiments/switching/`.\n"]
    parts.append(section_setup(["A", "B", "C1", "C2", "C3"]))
    parts.append(section_sanity())
    parts.append(section_offline("A", "2. Experiment A — S1 speaker, switching L1 (offline conditions)"))
    parts.append(section_feedback("B", "3. Experiment B — S2 modelling the switching L1 (feedback)"))
    parts.append(section_offline("C1", "4. Experiment C1 — S2 modelling Fang's vigilant L1-strat, switching L1 (offline)"))
    parts.append(section_offline("C2", "5. Experiment C2 — S2 modelling a credulous L1, switching L2 detector (offline)"))
    parts.append(section_feedback("C3", "6. Experiment C3 — hard_amnesic contrast in the feedback setting (α=3)"))
    parts.append(section_hugging())
    parts.append(findings("decisions"))
    path = os.path.join(OUT, "report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(parts) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
