"""
Analysis for the switching experiments.

    python experiments/switching/analyze.py --study A
    python experiments/switching/analyze.py --study B

Streams the per-cell parquet shards written by run.py, aggregates to
round-level and sim-level summaries, writes CSV tables and figures into
``results/switching_<study>/analysis/``.  Plotting is in plots.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import plots  # noqa: E402

PSIS = ["inf", "high", "low"]
CHECK_ROUNDS = (25, 50, 100, 150)
RECOVERY_TOL = 0.02
# Margin bins for the trade-off plot. Pre-switch rounds necessarily have
# Sus(t-1) <= boundary(t-1) -- a positive margin is exactly what makes the test
# fire -- so there is no ">0" bin to fill and the scale runs up to 0 only.
MARGIN_BINS = [-np.inf, -1.0, -0.75, -0.5, -0.25, -0.1, 0.0]
MARGIN_LABELS = ["< −1", "−1..−.75", "−.75..−.5", "−.5..−.25", "−.25..−.1", "−.1..0"]


def load_cfg(out_dir):
    with open(os.path.join(out_dir, "run_config.json"), encoding="utf-8") as fh:
        return json.load(fh)


def iter_shards(out_dir):
    root = os.path.join(out_dir, "trajectories")
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, "part.parquet")
        if os.path.isfile(p):
            yield int(name.split("=")[1]), p


# ---------------------------------------------------------------------------
# Per-shard aggregation
# ---------------------------------------------------------------------------

BASE_COLS = ["sim", "round", "theta_star", "psi_star", "alpha", "c", "switch_type", "utt",
             "switched", "sus1_Sus", "sus1_sigma_bar2",
             "E_theta_cred", "std_theta_cred", "E_theta_vig", "std_theta_vig",
             "E_theta_switch", "std_theta_switch"]
FEEDBACK_COLS = ["margin", "went_persuasive", "went_informative", "replica_maxdiff",
                 "p_pers_ref", "p_inf_ref", "p_chosen"]


def _first_true(mask, axis=-1):
    """Index of first True along axis, -1 if none."""
    any_ = mask.any(axis=axis)
    idx = mask.argmax(axis=axis)
    return np.where(any_, idx, -1)


def aggregate_shard(path, feedback, n_utt, trace_pick=None):
    cols = BASE_COLS + (FEEDBACK_COLS if feedback else [])
    df = pd.read_parquet(path, columns=cols)
    df = df.sort_values(["sim", "c", "switch_type", "round"], kind="stable")
    theta_star = float(df["theta_star"].iloc[0]); psi_star = df["psi_star"].iloc[0]
    alpha = float(df["alpha"].iloc[0])
    conds = df[["c", "switch_type"]].drop_duplicates().sort_values(["c", "switch_type"]).to_records(index=False)
    conds = [(float(c), str(st)) for c, st in conds]
    n_sims = int(df["sim"].nunique()); T = int(df["round"].max()); K = len(conds)
    shape = (n_sims, K, T)

    def arr(col, dtype=float):
        return df[col].to_numpy(dtype=dtype).reshape(shape)

    E = {L: arr(f"E_theta_{L}") for L in ("cred", "vig", "switch")}
    S = {L: arr(f"std_theta_{L}") for L in ("cred", "vig", "switch")}
    AB = {L: np.abs(E[L] - theta_star) for L in E}
    switched = arr("switched", int)
    tau_idx = _first_true(switched == 1, axis=2)                     # (n_sims, K), 0-based, -1 none
    has_tau = tau_idx >= 0
    rounds = np.arange(1, T + 1)

    # Splice check on the stored data: for switch_type "hard" the switching
    # trajectory must equal the credulous one strictly before tau and the
    # always-vigilant one from tau on.  Exact by construction, so any nonzero
    # value here is a bug in the runner, not a tolerance question.
    splice_pre = splice_post = 0.0
    splice_checked = False
    r_idx = np.arange(1, T + 1)[None, :]
    for k, (_c, _st) in enumerate(conds):
        if _st != "hard":
            continue
        splice_checked = True
        tau1 = np.where(has_tau[:, k], tau_idx[:, k] + 1, T + 1)[:, None]   # 1-based, T+1 = never
        pre, post = r_idx < tau1, r_idx >= tau1
        if pre.any():
            splice_pre = max(splice_pre, float(np.abs(E["switch"][:, k] - E["cred"][:, k])[pre].max()))
        if post.any():
            splice_post = max(splice_post, float(np.abs(E["switch"][:, k] - E["vig"][:, k])[post].max()))

    round_rows, sim_rows, choice_round_rows, choice_margin_rows = [], [], [], []
    utt_rows = []
    utt = arr("utt", int)
    # utterance frequency per round (per cell; same stream for every offline condition)
    for i in range(T):
        counts = np.bincount(utt[:, 0, i], minlength=n_utt) / n_sims
        utt_rows.append(dict(theta_star=theta_star, psi_star=psi_star, alpha=alpha, round=i + 1,
                             **{f"f_utt_{k}": counts[k] for k in range(n_utt)}))

    if feedback:
        margin = arr("margin"); wp = arr("went_persuasive", int); wi = arr("went_informative", int)
        sus = arr("sus1_Sus"); sig2 = arr("sus1_sigma_bar2")
        # Probability the actual policy puts on each reference utterance. A
        # continuous read of the same trade-off as the argmax-match fractions,
        # and a far less noisy one at low alpha where the softmax is diffuse.
        pp = arr("p_pers_ref"); pi_ = arr("p_inf_ref"); pc = arr("p_chosen")

    for k, (c, st) in enumerate(conds):
        key = dict(theta_star=theta_star, psi_star=psi_star, alpha=alpha, c=c, switch_type=st)
        sw_mask = has_tau[:, k]
        n_sw = int(sw_mask.sum())
        for i in range(T):
            r = dict(key, round=i + 1, p_switched=float(switched[:, k, i].mean()), n_sw=n_sw)
            for L, tag in (("cred", "cred"), ("vig", "vig"), ("switch", "sw")):
                r[f"E_{tag}"] = float(E[L][:, k, i].mean())
                r[f"abias_{tag}"] = float(AB[L][:, k, i].mean())
                r[f"std_{tag}"] = float(S[L][:, k, i].mean())
                if n_sw:
                    r[f"abias_{tag}_sw"] = float(AB[L][sw_mask, k, i].mean())
                    r[f"std_{tag}_sw"] = float(S[L][sw_mask, k, i].mean())
                else:
                    r[f"abias_{tag}_sw"] = np.nan; r[f"std_{tag}_sw"] = np.nan
            if feedback:
                with np.errstate(all="ignore"):
                    r["Sus_mean"] = float(np.nanmean(sus[:, k, i])) if np.isfinite(sus[:, k, i]).any() else np.nan
                    b = c * np.sqrt(sig2[:, k, i]) / np.sqrt(i + 1)
                    r["bound_mean"] = float(np.nanmean(b)) if np.isfinite(b).any() else np.nan
                if psi_star != "inf":
                    p, q = wp[:, k, i] == 1, wi[:, k, i] == 1
                    choice_round_rows.append(dict(key, round=i + 1, p_switched=r["p_switched"],
                                                  f_pers_only=float((p & ~q).mean()), f_inf_only=float((q & ~p).mean()),
                                                  f_both=float((p & q).mean()), f_neither=float((~p & ~q).mean()),
                                                  mean_p_pers=float(np.nanmean(pp[:, k, i])),
                                                  mean_p_inf=float(np.nanmean(pi_[:, k, i])),
                                                  mean_p_chosen=float(np.nanmean(pc[:, k, i]))))
            round_rows.append(r)

        if feedback and psi_star != "inf":
            m = margin[:, k, :]; bnd = c * np.sqrt(sig2[:, k, :]) / np.sqrt(rounds)[None, :]
            # margin is Sus(t-1) - bound(t-1): relative to the boundary of the previous round
            prev_b = np.concatenate([np.full((n_sims, 1), np.nan), bnd[:, :-1]], axis=1)
            rel = m / prev_b
            valid = np.isfinite(rel)
            bins = np.digitize(rel[valid], MARGIN_BINS[1:-1])
            p, q = (wp[:, k, :] == 1)[valid], (wi[:, k, :] == 1)[valid]
            pp_v, pi_v = pp[:, k, :][valid], pi_[:, k, :][valid]
            for b in range(len(MARGIN_LABELS)):
                sel = bins == b
                n_b = int(sel.sum())
                choice_margin_rows.append(dict(key, bin=b, n=n_b,
                                               n_pers_only=int((p & ~q & sel).sum()), n_inf_only=int((q & ~p & sel).sum()),
                                               n_both=int((p & q & sel).sum()), n_neither=int((~p & ~q & sel).sum()),
                                               sum_p_pers=float(pp_v[sel].sum()) if n_b else 0.0,
                                               sum_p_inf=float(pi_v[sel].sum()) if n_b else 0.0))

        for s in range(n_sims):
            ti = tau_idx[s, k]
            row = dict(key, sim=s, tau=(ti + 1 if ti >= 0 else -1), switched=int(ti >= 0))
            for L, tag in (("cred", "cred"), ("vig", "vig"), ("switch", "sw")):
                row[f"mean_abias_{tag}"] = float(AB[L][s, k].mean())
                row[f"std_{tag}_150"] = float(S[L][s, k, T - 1])
                for rr in CHECK_ROUNDS:
                    row[f"abias_{tag}_r{rr}"] = float(AB[L][s, k, rr - 1]) if rr <= T else np.nan
            if ti >= 0:
                row["bias_cred_at_tau"] = float(E["cred"][s, k, ti] - theta_star)
                row["abias_cred_at_tau"] = float(AB["cred"][s, k, ti])
                row["abias_sw_at_tau"] = float(AB["switch"][s, k, ti])
                ok = AB["switch"][s, k, ti:] <= AB["vig"][s, k, ti:] + RECOVERY_TOL
                j = _first_true(ok[None, :], axis=1)[0]
                row["recovery"] = float(j) if j >= 0 else np.nan
                row["recovered"] = int(j >= 0)
                row["mean_abias_sw_post"] = float(AB["switch"][s, k, ti:].mean())
                row["mean_abias_cred_post"] = float(AB["cred"][s, k, ti:].mean())
                row["mean_abias_vig_post"] = float(AB["vig"][s, k, ti:].mean())
            else:
                for kk in ("bias_cred_at_tau", "abias_cred_at_tau", "abias_sw_at_tau", "recovery",
                           "mean_abias_sw_post", "mean_abias_cred_post", "mean_abias_vig_post"):
                    row[kk] = np.nan
                row["recovered"] = np.nan
            if feedback:
                p, q = wp[s, k] == 1, wi[s, k] == 1
                row["frac_pers_only"] = float((p & ~q).mean()) if psi_star != "inf" else np.nan
                row["frac_inf_only"] = float((q & ~p).mean())
                row["replica_maxdiff"] = float(df["replica_maxdiff"].to_numpy().reshape(shape)[s, k].max())
            sim_rows.append(row)

    traces = []
    if feedback and trace_pick is not None and trace_pick(theta_star, psi_star, alpha):
        for k, (c, st) in enumerate(conds):
            taus = tau_idx[:, k] + 1
            traces.append(dict(psi_star=psi_star, alpha=alpha, switch_type=st, c=c, t=rounds,
                               sus=sus[:, k, :], bound=c * np.sqrt(sig2[:, k, :]) / np.sqrt(rounds)[None, :],
                               taus=taus, rate=float(has_tau[:, k].mean()),
                               median_tau=float(np.median(taus[taus > 0])) if (taus > 0).any() else np.nan))
    return (round_rows, sim_rows, utt_rows, choice_round_rows, choice_margin_rows,
            traces, (splice_pre, splice_post, splice_checked))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def rate_table(tau_df):
    g = tau_df.groupby(["psi_star", "alpha", "c", "switch_type"])
    out = g.agg(rate=("switched", "mean"), n=("switched", "size"),
                median_tau=("tau", lambda s: float(s[s > 0].median()) if (s > 0).any() else np.nan),
                mean_tau=("tau", lambda s: float(s[s > 0].mean()) if (s > 0).any() else np.nan)).reset_index()
    return out


def bias_table(sim_agg):
    long = []
    for L, tag in (("cred", "cred"), ("vig", "vig"), ("switch", "sw")):
        d = sim_agg[["theta_star", "psi_star", "alpha", "c", "switch_type", f"mean_abias_{tag}",
                     f"abias_{tag}_r150", f"std_{tag}_150"]].copy()
        d.columns = ["theta_star", "psi_star", "alpha", "c", "switch_type", "mean_abias", "abias_r150", "std_150"]
        d["listener"] = L
        long.append(d)
    long = pd.concat(long, ignore_index=True)
    per_theta = long.groupby(["listener", "psi_star", "alpha", "c", "switch_type", "theta_star"]).mean(numeric_only=True).reset_index()
    avg = per_theta.groupby(["listener", "psi_star", "alpha", "c", "switch_type"]).mean(numeric_only=True).reset_index().drop(columns="theta_star")
    return per_theta, avg


def cost_recovery_table(sim_agg):
    d = sim_agg[sim_agg.psi_star != "inf"]
    g = d.groupby(["psi_star", "alpha", "c", "switch_type"])
    out = g.agg(n=("switched", "size"), rate=("switched", "mean"),
                mean_tau=("tau", lambda s: float(s[s > 0].mean()) if (s > 0).any() else np.nan),
                median_tau=("tau", lambda s: float(s[s > 0].median()) if (s > 0).any() else np.nan),
                abias_cred_at_tau=("abias_cred_at_tau", "mean"), bias_cred_at_tau=("bias_cred_at_tau", "mean"),
                abias_sw_at_tau=("abias_sw_at_tau", "mean"),
                recovery_median=("recovery", "median"), recovery_mean=("recovery", "mean"),
                recovery_frac=("recovered", "mean"),
                mean_abias_sw_post=("mean_abias_sw_post", "mean"), mean_abias_cred_post=("mean_abias_cred_post", "mean"),
                mean_abias_vig_post=("mean_abias_vig_post", "mean")).reset_index()
    return out


def false_alarm_table(sim_agg):
    d = sim_agg[(sim_agg.psi_star == "inf")]
    g = d.groupby(["alpha", "c", "switch_type"])
    all_ = g.agg(n=("switched", "size"), rate=("switched", "mean")).reset_index()
    sw = d[d.switched == 1].groupby(["alpha", "c", "switch_type"]).agg(
        n_switched=("switched", "size"), median_tau=("tau", "median"),
        abias_cred_r150=("abias_cred_r150", "mean"), abias_vig_r150=("abias_vig_r150", "mean"),
        abias_sw_r150=("abias_sw_r150", "mean"),
        std_cred_150=("std_cred_150", "mean"), std_vig_150=("std_vig_150", "mean"), std_sw_150=("std_sw_150", "mean"),
        mean_abias_cred_post=("mean_abias_cred_post", "mean"), mean_abias_vig_post=("mean_abias_vig_post", "mean"),
        mean_abias_sw_post=("mean_abias_sw_post", "mean")).reset_index()
    return all_.merge(sw, on=["alpha", "c", "switch_type"], how="left")


def persuasion_table(sim_agg):
    long = []
    for L, tag in (("cred", "cred"), ("vig", "vig"), ("switch", "sw")):
        d = sim_agg[["theta_star", "psi_star", "alpha", "c", "switch_type"] + [f"abias_{tag}_r{r}" for r in CHECK_ROUNDS]].copy()
        d.columns = ["theta_star", "psi_star", "alpha", "c", "switch_type"] + [f"r{r}" for r in CHECK_ROUNDS]
        d["listener"] = L
        long.append(d)
    long = pd.concat(long, ignore_index=True)
    per_theta = long.groupby(["listener", "psi_star", "alpha", "c", "switch_type", "theta_star"]).mean(numeric_only=True).reset_index()
    return per_theta.groupby(["listener", "psi_star", "alpha", "c", "switch_type"]).mean(numeric_only=True).reset_index().drop(columns="theta_star")


def some_freq_table(utt_agg, utt_names):
    some_cols = [f"f_utt_{k}" for k, u in enumerate(utt_names) if u[0] == "some"]
    d = utt_agg.copy()
    d["f_some"] = d[some_cols].sum(axis=1)
    d["block"] = pd.cut(d["round"], [0, 10, 50, 150], labels=["1-10", "11-50", "51-150"])
    return d.groupby(["psi_star", "alpha", "block"], observed=True)["f_some"].mean().reset_index()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--in_dir", default=None)
    args = ap.parse_args()
    out_dir = args.in_dir or os.path.join("results", f"switching_{args.study}")
    cfg = load_cfg(out_dir)
    feedback = cfg["mode"] == "feedback"
    an = os.path.join(out_dir, "analysis"); os.makedirs(an, exist_ok=True)
    utt_names = [tuple(v) for _, v in sorted(cfg["utterance_index"].items(), key=lambda kv: int(kv[0]))]
    utt_labels = [f"{q}, {p[:3]}" for q, p in utt_names]
    thetas = cfg["grid"]["theta_stars"]; alphas = cfg["grid"]["alphas"]; cs = cfg["grid"]["cs"]
    sts = cfg["grid"]["switch_types"]; psis = [p for p in PSIS if p in cfg["grid"]["psi_stars"]]
    tag = f"[{args.study}: {cfg['speaker_level']} speaker, {cfg['listener_level']} listener]"

    def trace_pick(th, psi, a):
        return th == 0.5 and psi in ("high", "low") and a in (3.0, 10.0)

    R, Ssim, U, CR, CM, TR = [], [], [], [], [], []
    splice_pre = splice_post = 0.0
    splice_checked = False
    for cid, path in iter_shards(out_dir):
        r, s, u, cr, cm, tr, sp = aggregate_shard(path, feedback, len(utt_names), trace_pick)
        R += r; Ssim += s; U += u; CR += cr; CM += cm; TR += tr
        splice_pre = max(splice_pre, sp[0]); splice_post = max(splice_post, sp[1])
        splice_checked = splice_checked or sp[2]
    round_agg = pd.DataFrame(R); sim_agg = pd.DataFrame(Ssim); utt_agg = pd.DataFrame(U)
    round_agg.to_parquet(os.path.join(an, "round_agg.parquet"), index=False)
    sim_agg.to_parquet(os.path.join(an, "sim_agg.parquet"), index=False)
    utt_agg.to_parquet(os.path.join(an, "utt_agg.parquet"), index=False)
    tau_df = pd.read_parquet(os.path.join(out_dir, "tau_summary.parquet"))
    P = args.study
    n_runs = len(sim_agg)
    n_sw = int(sim_agg["switched"].sum())
    with open(os.path.join(an, f"{P}0_splice_check.txt"), "w", encoding="utf-8") as fh:
        if splice_checked:
            fh.write("\n".join([
                "Retrospective switch identity, checked on the stored trajectories",
                f"({n_runs} (sim, c, switch_type) runs, {n_sw} of which switched):",
                f"  max |E_switch - E_cred| over rounds t <  tau : {splice_pre:.3e}",
                f"  max |E_switch - E_vig|  over rounds t >= tau : {splice_post:.3e}",
                "E[theta] columns are stored as float32; 0.0 is bit-identical agreement.",
                "",
            ]))
        else:
            fh.write("Retrospective switch identity: NOT CHECKED. This study ran no "
                     f"switch_type=\"hard\" condition (it ran {sorted(set(sim_agg['switch_type']))}), "
                     "so there is no retrospective trajectory to compare against the "
                     "always-vigilant one.\n")

    # 1. tau distribution, FPR / TPR
    rates = rate_table(tau_df)
    rates.to_csv(os.path.join(an, f"{P}1_rates_by_cell.csv"), index=False)
    pooled = rates.groupby(["c", "switch_type", "psi_star"]).agg(rate=("rate", "mean"), median_tau=("median_tau", "median")).reset_index()
    pooled.to_csv(os.path.join(an, f"{P}1_rates_pooled.csv"), index=False)
    # The full sweep is an S1-speaker dataset, so it is a valid external
    # reference only for the S1 studies.  For the S2 studies the crossing rates
    # are measuring something else entirely (level mismatch, not goal
    # detection), and overlaying the sweep would invite a false comparison --
    # so those studies get the same plot without a reference series.
    ref_path = os.path.join("results", "switching", "sweep_reference_rates.csv")
    ref = None
    comparable = cfg["speaker_level"] == "S1"
    if comparable and os.path.isfile(ref_path):
        ref = pd.read_csv(ref_path)
        ref["psi_star"] = ref["psi"].map({"inf": "inf", "pers+": "high", "pers-": "low"})
        ref = ref.groupby(["psi_star", "alpha", "c"]).agg(rate=("rate", "mean")).reset_index()
    cmp = rates[rates.switch_type == sts[0]].groupby(["psi_star", "alpha", "c"]).agg(rate=("rate", "mean")).reset_index()
    if ref is not None:
        cmp = cmp.merge(ref.rename(columns={"rate": "rate_sweep"}), on=["psi_star", "alpha", "c"], how="left")
        cmp.to_csv(os.path.join(an, f"{P}1_sweep_comparison.csv"), index=False)
    plots.rate_vs_alpha(cmp, cs, os.path.join(an, f"{P}1_rates_vs_alpha.png"), title=tag, ref=ref)
    for c in cs:
        plots.tau_hist(tau_df, c, alphas, psis, sts[0], os.path.join(an, f"{P}1_tau_hist_c{c}.png"),
                       rounds=cfg["grid"]["rounds"], title=tag)

    # 2. panels and bias/std curves
    per_theta, avg = bias_table(sim_agg)
    per_theta.to_csv(os.path.join(an, f"{P}2_bias_per_theta.csv"), index=False)
    avg.to_csv(os.path.join(an, f"{P}2_bias_avg.csv"), index=False)
    for c in cs:
        for a in alphas:
            fn = os.path.join(an, f"{P}2_panels_c{c}_alpha{a}.png")
            if feedback:
                plots.panels_feedback(round_agg, c, a, thetas, psis, sts, fn, title=tag)
            else:
                plots.panels_offline(round_agg, c, a, thetas, psis, sts, fn, title=tag)
        plots.bias_curves(round_agg, c, "abias", alphas, psis, sts, os.path.join(an, f"{P}2_bias_c{c}.png"),
                          title=tag, ylabel="|E[θ]−θ*|")
        plots.bias_curves(round_agg, c, "std", alphas, psis, sts, os.path.join(an, f"{P}2_std_c{c}.png"),
                          title=tag, ylabel="std[θ]")

    # 3. cost until tau, recovery
    cr = cost_recovery_table(sim_agg)
    cr.to_csv(os.path.join(an, f"{P}3_cost_recovery.csv"), index=False)
    plots.cost_recovery(cr, cs, sts, os.path.join(an, f"{P}3_cost_recovery.png"), title=tag)

    # 4. false-alarm cost (psi* = inf, conditional on switching)
    fa = false_alarm_table(sim_agg)
    fa.to_csv(os.path.join(an, f"{P}4_false_alarm_cost.csv"), index=False)
    for c in cs:
        plots.bias_curves(round_agg, c, "abias", alphas, ["inf"], sts, os.path.join(an, f"{P}4_false_alarm_c{c}.png"),
                          title=tag, ylabel="|E[θ]−θ*|", conditional=True)

    # 5. utterance frequency by round
    plots.utt_freq(utt_agg, alphas, psis, utt_labels, os.path.join(an, f"{P}5_utt_freq.png"), title=tag)
    some_freq_table(utt_agg, utt_names).to_csv(os.path.join(an, f"{P}5_some_freq.csv"), index=False)

    if feedback:
        choice_round = pd.DataFrame(CR); choice_margin = pd.DataFrame(CM)
        choice_round.to_parquet(os.path.join(an, "choice_round.parquet"), index=False)
        choice_margin.to_csv(os.path.join(an, f"{P}2_choice_margin_counts.csv"), index=False)
        for c in cs:
            plots.choice_vs_round(choice_round, c, alphas, sts, os.path.join(an, f"{P}2_choice_vs_round_c{c}.png"), title=tag)
            plots.choice_vs_margin(choice_margin, c, alphas, sts, MARGIN_LABELS,
                                   os.path.join(an, f"{P}2_choice_vs_margin_c{c}.png"), title=tag)
            plots.policy_mass(choice_round, choice_margin, c, alphas, sts, MARGIN_LABELS,
                              os.path.join(an, f"{P}2_policy_mass_c{c}.png"), title=tag)
        cs_ = choice_round.groupby(["psi_star", "alpha", "c", "switch_type"])[
            ["f_pers_only", "f_inf_only", "f_both", "f_neither",
             "mean_p_pers", "mean_p_inf", "mean_p_chosen"]].mean().reset_index()
        cs_.to_csv(os.path.join(an, f"{P}2_choice_summary.csv"), index=False)
        pre = choice_round.copy()
        pre["block"] = pd.cut(pre["round"], [0, 5, 10, 25, 50, 150], labels=["1-5", "6-10", "11-25", "26-50", "51-150"])
        pre.groupby(["psi_star", "alpha", "c", "switch_type", "block"], observed=True)[
            ["f_pers_only", "f_inf_only", "f_both", "p_switched",
             "mean_p_pers", "mean_p_inf"]].mean().reset_index() \
           .to_csv(os.path.join(an, f"{P}2_choice_by_block.csv"), index=False)
        if TR:
            plots.sus_vs_boundary([t for t in TR if t["c"] == 3.5], os.path.join(an, f"{P}3_sus_vs_boundary.png"), title=tag)
        pt = persuasion_table(sim_agg)
        pt.to_csv(os.path.join(an, f"{P}4_persuasion.csv"), index=False)
        plots.persuasion_bars(pt, cs, os.path.join(an, f"{P}4_persuasion.png"), title=tag)
        mm = rates[rates.psi_star == "inf"][["alpha", "c", "switch_type", "rate", "median_tau", "n"]]
        mm.to_csv(os.path.join(an, f"{P}5_mismatch_alarm.csv"), index=False)
        plots.mismatch_alarm(mm, cs, sts, os.path.join(an, f"{P}5_mismatch_alarm.png"), title=tag)
        rep = sim_agg["replica_maxdiff"].max()
        with open(os.path.join(an, f"{P}0_replica_check.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"max |replica - actual| theta-marginal over all rounds and sims: {rep:.3e}\n")
    print(f"analysis written to {an}")


if __name__ == "__main__":
    main()
