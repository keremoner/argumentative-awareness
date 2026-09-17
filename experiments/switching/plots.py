"""
All plotting for the switching experiments lives here.  Every function takes
already-aggregated pandas DataFrames (see analyze.py) and writes one PNG.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"cred": "tab:orange", "vig": "tab:blue", "hard": "tab:green",
          "soft": "tab:purple", "hard_amnesic": "tab:gray"}
LS = {"cred": "-", "vig": "-", "hard": "-", "soft": "-", "hard_amnesic": ":"}
LABELS = {"cred": "credulous", "vig": "always-vigilant", "hard": "switch-retro",
          "soft": "switch-soft", "hard_amnesic": "switch-amnesic"}
PSI_TXT = {"inf": "ψ*=inf", "high": "ψ*=pers+", "low": "ψ*=pers−"}
ST_TXT = {"hard": "retro", "soft": "soft", "hard_amnesic": "amnesic"}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _sel(df, **kw):
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    return df[m]


# ---------------------------------------------------------------------------
# Fang-style panels: rows = psi*, columns = theta*, E[theta] trajectories
# ---------------------------------------------------------------------------

def panels_offline(round_agg, c, alpha, thetas, psis, switch_types, path, title=""):
    fig, axes = plt.subplots(len(psis), len(thetas), figsize=(2.9 * len(thetas), 2.5 * len(psis)),
                             sharex=True, sharey=True, squeeze=False)
    for i, psi in enumerate(psis):
        for j, th in enumerate(thetas):
            ax = axes[i][j]
            sub = _sel(round_agg, psi_star=psi, theta_star=th, alpha=alpha, c=c)
            if len(sub) == 0:
                ax.set_visible(False); continue
            base = _sel(sub, switch_type=switch_types[0]).sort_values("round")
            ax.plot(base["round"], base["E_cred"], color=COLORS["cred"], lw=1.6, label=LABELS["cred"])
            ax.plot(base["round"], base["E_vig"], color=COLORS["vig"], lw=1.6, label=LABELS["vig"])
            for st in switch_types:
                s = _sel(sub, switch_type=st).sort_values("round")
                ax.plot(s["round"], s["E_sw"], color=COLORS[st], ls=LS[st], lw=1.3, label=LABELS[st])
            ax.axhline(th, color="red", ls="--", lw=0.8)
            if i == 0:
                ax.set_title(f"θ*={th}")
            if j == 0:
                ax.set_ylabel(f"{PSI_TXT[psi]}\nE[θ]")
            if i == len(psis) - 1:
                ax.set_xlabel("round")
            ax.set_ylim(0, 1)
    axes[0][0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"{title}  α={alpha}, c={c}  (mean over sims)")
    _save(fig, path)


def panels_feedback(round_agg, c, alpha, thetas, psis, switch_types, path, title=""):
    rows = [(psi, st) for psi in psis for st in switch_types]
    fig, axes = plt.subplots(len(rows), len(thetas), figsize=(2.9 * len(thetas), 2.3 * len(rows)),
                             sharex=True, sharey=True, squeeze=False)
    for i, (psi, st) in enumerate(rows):
        for j, th in enumerate(thetas):
            ax = axes[i][j]
            s = _sel(round_agg, psi_star=psi, theta_star=th, alpha=alpha, c=c, switch_type=st).sort_values("round")
            if len(s) == 0:
                ax.set_visible(False); continue
            ax.plot(s["round"], s["E_cred"], color=COLORS["cred"], lw=1.4, label=LABELS["cred"])
            ax.plot(s["round"], s["E_vig"], color=COLORS["vig"], lw=1.4, label=LABELS["vig"])
            ax.plot(s["round"], s["E_sw"], color=COLORS[st], lw=1.6, label=LABELS[st])
            ax.axhline(th, color="red", ls="--", lw=0.8)
            if i == 0:
                ax.set_title(f"θ*={th}")
            if j == 0:
                ax.set_ylabel(f"{PSI_TXT[psi]}, {ST_TXT[st]}\nE[θ]")
            if i == len(rows) - 1:
                ax.set_xlabel("round")
            ax.set_ylim(0, 1)
            if j == 0:
                ax.legend(fontsize=6, loc="lower right")
    fig.suptitle(f"{title}  α={alpha}, c={c}  (S2 adapts to the listener; each row is its own set of sims)")
    _save(fig, path)


# ---------------------------------------------------------------------------
# |bias| and std vs round, averaged over theta*
# ---------------------------------------------------------------------------

def bias_curves(round_agg, c, metric, alphas, psis, switch_types, path, title="",
                listeners=("cred", "vig"), ylabel=None, conditional=False):
    """metric in {"abias", "std"}; conditional=True uses the *_sw columns
    (means over sims that switched)."""
    suf = "_sw" if conditional else ""
    fig, axes = plt.subplots(len(psis), len(alphas), figsize=(3.2 * len(alphas), 2.5 * len(psis)),
                             sharex=True, sharey="row", squeeze=False)
    for i, psi in enumerate(psis):
        for j, a in enumerate(alphas):
            ax = axes[i][j]
            sub = _sel(round_agg, psi_star=psi, alpha=a, c=c)
            if len(sub) == 0:
                ax.set_visible(False); continue
            base = _sel(sub, switch_type=switch_types[0])
            g = base.groupby("round")
            for L in listeners:
                col = f"{metric}_{L}{suf}"
                ax.plot(g[col].mean().index, g[col].mean().to_numpy(), color=COLORS[L], lw=1.5, label=LABELS[L])
            for st in switch_types:
                g = _sel(sub, switch_type=st).groupby("round")
                col = f"{metric}_sw{suf}"
                ax.plot(g[col].mean().index, g[col].mean().to_numpy(), color=COLORS[st], ls=LS[st], lw=1.3, label=LABELS[st])
            if i == 0:
                ax.set_title(f"α={a}")
            if j == 0:
                ax.set_ylabel(f"{PSI_TXT[psi]}\n{ylabel or metric}")
            if i == len(psis) - 1:
                ax.set_xlabel("round")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"{title}  c={c}  (mean over θ* and sims{', switched sims only' if conditional else ''})")
    _save(fig, path)


# ---------------------------------------------------------------------------
# tau histograms
# ---------------------------------------------------------------------------

def tau_hist(tau_df, c, alphas, psis, switch_type, path, rounds=150, title=""):
    fig, axes = plt.subplots(len(psis), len(alphas), figsize=(3.0 * len(alphas), 2.2 * len(psis)),
                             sharex=True, squeeze=False)
    for i, psi in enumerate(psis):
        for j, a in enumerate(alphas):
            ax = axes[i][j]
            sub = _sel(tau_df, psi_star=psi, alpha=a, c=c, switch_type=switch_type)
            if len(sub) == 0:
                ax.set_visible(False); continue
            taus = sub.loc[sub["tau"] > 0, "tau"]
            ax.hist(taus, bins=np.arange(0, rounds + 6, 5), color=COLORS["hard"], alpha=0.8)
            ax.text(0.97, 0.9, f"rate {len(taus)/len(sub):.2f}\nmedian τ {taus.median() if len(taus) else float('nan'):.0f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8)
            if i == 0:
                ax.set_title(f"α={a}")
            if j == 0:
                ax.set_ylabel(f"{PSI_TXT[psi]}\ncount")
            if i == len(psis) - 1:
                ax.set_xlabel("τ")
    fig.suptitle(f"{title}  first-crossing time τ at c={c}")
    _save(fig, path)


# ---------------------------------------------------------------------------
# Cost until tau and recovery
# ---------------------------------------------------------------------------

def cost_recovery(tab, cs, switch_types, path, title=""):
    """tab columns: psi_star, alpha, c, switch_type, abias_cred_at_tau, recovery_median, recovery_frac."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for c, ls in zip(cs, ["-", "--"]):
        for psi, mk in (("high", "o"), ("low", "s")):
            s = _sel(tab, psi_star=psi, c=c, switch_type=switch_types[0]).sort_values("alpha")
            axes[0].plot(s["alpha"], s["abias_cred_at_tau"], marker=mk, ls=ls, label=f"{PSI_TXT[psi]}, c={c}")
            axes[0].set_ylabel("|E[θ]−θ*| of credulous belief at τ")
    axes[0].set_xlabel("α"); axes[0].set_title("cost until τ"); axes[0].legend(fontsize=7)
    for c, ls in zip(cs, ["-", "--"]):
        for st in switch_types:
            s = tab[(tab.c == c) & (tab.switch_type == st) & (tab.psi_star != "inf")]
            s = s.groupby("alpha").agg(rec=("recovery_median", "mean"), frac=("recovery_frac", "mean")).reset_index()
            axes[1].plot(s["alpha"], s["rec"], marker="o", ls=ls, color=COLORS[st], label=f"{LABELS[st]}, c={c}")
            axes[2].plot(s["alpha"], s["frac"], marker="o", ls=ls, color=COLORS[st], label=f"{LABELS[st]}, c={c}")
    axes[1].set_xlabel("α"); axes[1].set_ylabel("median rounds after τ"); axes[1].set_title("recovery (|bias| ≤ vigilant + 0.02)")
    axes[2].set_xlabel("α"); axes[2].set_ylabel("fraction recovered within horizon"); axes[2].set_title("recovery fraction")
    axes[1].legend(fontsize=7)
    fig.suptitle(title)
    _save(fig, path)


# ---------------------------------------------------------------------------
# Utterance frequency by round
# ---------------------------------------------------------------------------

def utt_freq(utt_agg, alphas, psis, utt_names, path, title="", smooth=9):
    """utt_agg: psi_star, alpha, round, f_utt_0..f_utt_7 (mean over theta* and sims).

    Per-round shares carry ~2% sampling noise, which swamps the trend in a
    stacked area chart, so each series is smoothed with a centred rolling mean
    of ``smooth`` rounds. The quantitative claim lives in the by-block CSV.
    """
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(len(psis), len(alphas), figsize=(3.2 * len(alphas), 2.4 * len(psis)),
                             sharex=True, sharey=True, squeeze=False)
    n_u = len(utt_names)
    for i, psi in enumerate(psis):
        for j, a in enumerate(alphas):
            ax = axes[i][j]
            s = _sel(utt_agg, psi_star=psi, alpha=a).groupby("round")[[f"f_utt_{k}" for k in range(n_u)]].mean()
            if len(s) == 0:
                ax.set_visible(False); continue
            s = s.rolling(smooth, center=True, min_periods=1).mean()
            ys = [s[f"f_utt_{k}"].to_numpy() for k in range(n_u)]
            ax.stackplot(s.index, ys, colors=[cmap(k) for k in range(n_u)], labels=utt_names, alpha=0.85)
            if i == 0:
                ax.set_title(f"α={a}")
            if j == 0:
                ax.set_ylabel(f"{PSI_TXT[psi]}\nutterance share")
            if i == len(psis) - 1:
                ax.set_xlabel("round")
            ax.set_ylim(0, 1)
    axes[0][-1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle(f"{title}  utterance frequency by round "
                 f"(mean over θ* and sims, {smooth}-round rolling mean)")
    _save(fig, path)


# ---------------------------------------------------------------------------
# Experiment B: speaker choice trade-off
# ---------------------------------------------------------------------------

CHOICE_COLS = [("f_pers_only", "went persuasive (≠ informative)", "tab:red"),
               ("f_inf_only", "went informative (≠ persuasive)", "tab:blue"),
               ("f_both", "persuasive = informative", "tab:gray"),
               ("f_neither", "neither", "tab:olive")]


def choice_vs_round(choice_round, c, alphas, switch_types, path, title="", smooth=9):
    """Per-round choice shares, smoothed with a centred rolling mean (the raw
    per-round fractions carry ~5% sampling noise); P(switched) is left raw."""
    psis = ["high", "low"]
    fig, axes = plt.subplots(len(psis), len(alphas), figsize=(3.4 * len(alphas), 2.7 * len(psis)),
                             sharex=True, sharey=True, squeeze=False)
    handles = {}
    for i, psi in enumerate(psis):
        for j, a in enumerate(alphas):
            ax = axes[i][j]
            for st, ls in zip(switch_types, ["-", "--"]):
                s = _sel(choice_round, psi_star=psi, alpha=a, c=c, switch_type=st)
                if len(s) == 0:
                    continue
                g = s.groupby("round").mean(numeric_only=True)
                sm = g.rolling(smooth, center=True, min_periods=1).mean()
                for col, lab, colr in CHOICE_COLS[:2]:
                    ln, = ax.plot(sm.index, sm[col].to_numpy(), color=colr, ls=ls,
                                  label=f"{lab} [{ST_TXT[st]}]")
                    handles[f"{lab} [{ST_TXT[st]}]"] = ln
                ln, = ax.plot(g.index, g["p_switched"].to_numpy(), color="k", ls=ls, lw=0.8,
                              label=f"P(switched) [{ST_TXT[st]}]")
                handles[f"P(switched) [{ST_TXT[st]}]"] = ln
            if i == 0:
                ax.set_title(f"α={a}")
            if j == 0:
                ax.set_ylabel(f"{PSI_TXT[psi]}\nfraction of rounds")
            if i == len(psis) - 1:
                ax.set_xlabel("round")
            ax.set_ylim(0, 1)
    fig.legend(handles.values(), handles.keys(), fontsize=7, loc="upper left",
               bbox_to_anchor=(1.0, 0.92), frameon=False)
    fig.suptitle(f"{title}  persuasive S2's choice by round, c={c} "
                 f"(mean over θ* and sims, {smooth}-round rolling mean)")
    _save(fig, path)


def choice_vs_margin(choice_margin, c, alphas, switch_types, bin_labels, path, title="",
                     min_n=50):
    """Choice shares binned by how close Sus was to the boundary before the round.

    Bins holding fewer than ``min_n`` rounds are dropped rather than plotted:
    at high α the detector fires within a few rounds, so the far-from-boundary
    bins can hold a handful of rounds whose noise would read as structure. The
    per-bin counts are in the accompanying CSV.
    """
    psis = ["high", "low"]
    fig, axes = plt.subplots(len(psis), len(alphas), figsize=(3.4 * len(alphas), 2.8 * len(psis)),
                             sharex=True, sharey=True, squeeze=False)
    x = np.arange(len(bin_labels))
    handles = {}
    for i, psi in enumerate(psis):
        for j, a in enumerate(alphas):
            ax = axes[i][j]
            n_tot = 0
            for st, ls in zip(switch_types, ["-", "--"]):
                s = _sel(choice_margin, psi_star=psi, alpha=a, c=c, switch_type=st)
                if len(s) == 0:
                    continue
                w = s.groupby("bin")[["n_pers_only", "n_inf_only", "n_both", "n_neither", "n"]].sum().reindex(range(len(bin_labels)))
                n_tot += int(w["n"].fillna(0).sum())
                enough = (w["n"].fillna(0) >= min_n).to_numpy()
                for key, lab, colr in (("n_pers_only", "went persuasive", "tab:red"),
                                       ("n_inf_only", "went informative", "tab:blue")):
                    y = np.where(enough, (w[key] / w["n"]).to_numpy(), np.nan)
                    ln, = ax.plot(x, y, marker="o", ms=3, color=colr, ls=ls,
                                  label=f"{lab} [{ST_TXT[st]}]")
                    handles[f"{lab} [{ST_TXT[st]}]"] = ln
            ax.set_xticks(x); ax.set_xticklabels(bin_labels, rotation=60, fontsize=7)
            ax.text(0.03, 0.95, f"n={n_tot}", transform=ax.transAxes, va="top", fontsize=7,
                    color="0.4")
            if i == 0:
                ax.set_title(f"α={a}")
            if j == 0:
                ax.set_ylabel(f"{PSI_TXT[psi]}\nfraction of rounds")
            ax.set_ylim(0, 1)
    fig.legend(handles.values(), handles.keys(), fontsize=7, loc="upper left",
               bbox_to_anchor=(1.0, 0.92), frameon=False)
    fig.supxlabel("(Sus − boundary) / boundary in the round before the choice   "
                  "(pre-switch rounds only; more negative = further below the boundary)",
                  fontsize=8)
    fig.suptitle(f"{title}  persuasive S2's choice vs distance to the boundary, c={c}")
    _save(fig, path)


def policy_mass(choice_round, choice_margin, c, alphas, switch_types, bin_labels, path,
                title="", smooth=9, min_n=50):
    """Probability the persuasive S2's own policy puts on the persuasive-reference
    utterance versus the informative-reference utterance.

    Left block: by round. Right block: by distance to the boundary. This is the
    continuous version of the argmax-match fractions and carries the trade-off
    signal at low α, where the softmax is too diffuse for an argmax to be
    informative.
    """
    psis = ["high", "low"]
    ncol = len(alphas) + len(alphas)
    fig, axes = plt.subplots(len(psis), ncol, figsize=(2.6 * ncol, 2.7 * len(psis)),
                             sharey=True, squeeze=False)
    x = np.arange(len(bin_labels))
    handles = {}
    for i, psi in enumerate(psis):
        for j, a in enumerate(alphas):
            ax = axes[i][j]
            for st, ls in zip(switch_types, ["-", "--"]):
                s = _sel(choice_round, psi_star=psi, alpha=a, c=c, switch_type=st)
                if len(s) == 0:
                    continue
                g = s.groupby("round").mean(numeric_only=True).rolling(smooth, center=True, min_periods=1).mean()
                for col, lab, colr in (("mean_p_pers", "P(persuasive ref)", "tab:red"),
                                       ("mean_p_inf", "P(informative ref)", "tab:blue")):
                    ln, = ax.plot(g.index, g[col].to_numpy(), color=colr, ls=ls, label=f"{lab} [{ST_TXT[st]}]")
                    handles[f"{lab} [{ST_TXT[st]}]"] = ln
            ax.set_title(f"α={a}", fontsize=9)
            ax.set_xlabel("round", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{PSI_TXT[psi]}\npolicy probability")
            ax.set_ylim(0, 1)
        for j, a in enumerate(alphas):
            ax = axes[i][len(alphas) + j]
            for st, ls in zip(switch_types, ["-", "--"]):
                s = _sel(choice_margin, psi_star=psi, alpha=a, c=c, switch_type=st)
                if len(s) == 0:
                    continue
                w = s.groupby("bin")[["sum_p_pers", "sum_p_inf", "n"]].sum().reindex(range(len(bin_labels)))
                enough = (w["n"].fillna(0) >= min_n).to_numpy()
                for key, colr in (("sum_p_pers", "tab:red"), ("sum_p_inf", "tab:blue")):
                    y = np.where(enough, (w[key] / w["n"]).to_numpy(), np.nan)
                    ax.plot(x, y, marker="o", ms=3, color=colr, ls=ls)
            ax.set_xticks(x); ax.set_xticklabels(bin_labels, rotation=60, fontsize=6)
            ax.set_title(f"α={a}, by margin", fontsize=9)
            ax.set_ylim(0, 1)
    fig.legend(handles.values(), handles.keys(), fontsize=7, loc="upper left",
               bbox_to_anchor=(1.0, 0.92), frameon=False)
    fig.suptitle(f"{title}  where the persuasive S2 puts its probability mass, c={c} "
                 f"(left: by round, {smooth}-round rolling mean; right: by margin, pre-switch rounds)")
    _save(fig, path)


def sus_vs_boundary(traces, path, title=""):
    """traces: list of dicts with keys psi_star, alpha, switch_type, t, sus (n_sims x T, NaN after tau),
    bound (n_sims x T), taus."""
    keys = sorted({(d["psi_star"], d["alpha"]) for d in traces})
    sts = ["hard", "soft"]
    fig, axes = plt.subplots(len(keys), 2, figsize=(9, 2.6 * len(keys)), sharex=True, sharey="row", squeeze=False)
    for i, (psi, a) in enumerate(keys):
        for j, st in enumerate(sts):
            ax = axes[i][j]
            d = next((d for d in traces if d["psi_star"] == psi and d["alpha"] == a and d["switch_type"] == st), None)
            if d is None:
                ax.set_visible(False); continue
            t = d["t"]
            for k in range(min(12, d["sus"].shape[0])):
                ax.plot(t, d["sus"][k], color=COLORS[st], alpha=0.35, lw=0.8)
            with np.errstate(all="ignore"):
                ax.plot(t, np.nanmean(d["sus"], axis=0), color=COLORS[st], lw=2, label="mean Sus(t) (sims not yet switched)")
                ax.plot(t, np.nanmean(d["bound"], axis=0), color="k", lw=1.5, label="boundary c·σ̄/√t")
            ax.set_title(f"{PSI_TXT[psi]}, α={a}, {ST_TXT[st]}  (switch rate {d['rate']:.2f}, median τ {d['median_tau']:.0f})", fontsize=9)
            if j == 0:
                ax.set_ylabel("Sus(t)")
            if i == len(keys) - 1:
                ax.set_xlabel("round")
            ax.set_xlim(0, 60)
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"{title}  sus_1 running mean against the boundary (θ*=0.5, c=3.5)")
    _save(fig, path)


def persuasion_bars(tab, cs, path, title=""):
    """tab: psi_star, alpha, c, switch_type, listener, r25, r50, r100, r150 (|E-theta*| averaged over theta*)."""
    rounds = ["r25", "r50", "r100", "r150"]
    psis = ["high", "low"]
    fig, axes = plt.subplots(len(psis), len(cs), figsize=(5.2 * len(cs), 2.8 * len(psis)), sharey=True, squeeze=False)
    alphas = sorted(tab["alpha"].unique())
    for i, psi in enumerate(psis):
        for j, c in enumerate(cs):
            ax = axes[i][j]
            x = np.arange(len(alphas))
            series = [("cred", "hard", COLORS["cred"], LABELS["cred"] + " (retro sims)"),
                      ("vig", "hard", COLORS["vig"], LABELS["vig"] + " (retro sims)"),
                      ("sw", "hard", COLORS["hard"], LABELS["hard"]),
                      ("sw", "soft", COLORS["soft"], LABELS["soft"])]
            width = 0.8 / len(series)
            for k, (L, st, colr, lab) in enumerate(series):
                s = _sel(tab, psi_star=psi, c=c, switch_type=st, listener=L).set_index("alpha").reindex(alphas)
                ax.bar(x + (k - 1.5) * width, s["r150"].to_numpy(), width, color=colr, label=lab)
                ax.plot(x + (k - 1.5) * width, s["r50"].to_numpy(), "k_", ms=8)
            ax.set_xticks(x); ax.set_xticklabels([str(a) for a in alphas])
            ax.set_title(f"{PSI_TXT[psi]}, c={c}")
            if j == 0:
                ax.set_ylabel("|E[θ]−θ*| at round 150\n(tick = round 50)")
            if i == len(psis) - 1:
                ax.set_xlabel("α")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"{title}  achieved persuasion of the actual listener (mean over θ* and sims)")
    _save(fig, path)


def mismatch_alarm(tab, cs, switch_types, path, title=""):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for c, ls in zip(cs, ["-", "--"]):
        for st in switch_types:
            s = _sel(tab, c=c, switch_type=st).sort_values("alpha")
            ax.plot(s["alpha"], s["rate"], marker="o", ls=ls, color=COLORS[st], label=f"{ST_TXT[st]}, c={c}")
    ax.set_xlabel("α"); ax.set_ylabel("alarm rate under S2-inf"); ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.set_title(f"{title}  level-mismatch alarm rate (L1 detector, S2-inf speaker)")
    _save(fig, path)


def rate_vs_alpha(tab, cs, path, title="", ref=None):
    """tab: psi_star, alpha, c, rate. ref: same columns from the sweep (optional)."""
    psis = ["inf", "high", "low"]
    fig, axes = plt.subplots(1, len(psis), figsize=(4 * len(psis), 3.2), sharey=True)
    for j, psi in enumerate(psis):
        ax = axes[j]
        for c, ls in zip(cs, ["-", "--"]):
            s = _sel(tab, psi_star=psi, c=c).sort_values("alpha")
            ax.plot(s["alpha"], s["rate"], marker="o", ls=ls, color="tab:green", label=f"this study, c={c}")
            if ref is not None:
                r = _sel(ref, psi_star=psi, c=c).sort_values("alpha")
                ax.plot(r["alpha"], r["rate"], marker="x", ls=ls, color="tab:gray", label=f"full sweep, c={c}")
        ax.set_title(("FPR " if psi == "inf" else "TPR ") + PSI_TXT[psi]); ax.set_xlabel("α")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("first-crossing rate within 150 rounds"); axes[0].legend(fontsize=7)
    fig.suptitle(title)
    _save(fig, path)
