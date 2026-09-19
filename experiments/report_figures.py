"""
Figures and tables for docs/report.typ.

Reads the aggregated tables in results/full_sweep_v2/agg and
results/switching/agg (built by experiments/full_sweep/aggregate.py and
experiments/switching/aggregate.py) and writes PNGs to results/report/figures.

Score naming used in the report (differs from the parquet column names):

    report name                          parquet prefix
    surp1  prior predictive surprisal    surp2_*
    surp2  posterior predictive surprisal sus1_*

Run:  .conda/python.exe experiments/report_figures.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(ROOT, "results", "full_sweep_v2", "agg")
SWITCH = os.path.join(ROOT, "results", "switching", "agg")
OUT = os.path.join(ROOT, "results", "report", "figures")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- styling --
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA, C_GREEN, C_VIOLET, C_RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")

# fixed assignments (identity -> hue), never cycled
PSI_COL = {"inf": C_BLUE, "high": C_ORANGE, "low": C_VIOLET}
PSI_LAB = {"inf": "informative", "high": "persuade-up", "low": "persuade-down"}
LIS_COL = {"cred": C_ORANGE, "vig": C_BLUE, "switch": C_AQUA}
LIS_LAB = {"cred": "credulous", "vig": "vigilant", "switch": "switching"}
SC_COL = {"surp1": C_MAGENTA, "surp2": C_BLUE}
SC_LAB = {"surp1": "surp1 (prior predictive)", "surp2": "surp2 (posterior predictive)"}
SC_PREFIX = {"surp1": "surp2", "surp2": "sus1"}          # report name -> parquet prefix
STUDY_LAB = {"A": "S1 speaker (study A)", "B": "S2 speaker (study B)"}

BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#1c5cab", "#104281", "#0d366b"]
ORANGES = ["#fbd9c9", "#f5b08e", "#ef8a5e", "#eb6834", "#c94f21", "#9d3b16"]
seq_blue = LinearSegmentedColormap.from_list("seqblue", BLUES)
seq_orange = LinearSegmentedColormap.from_list("seqorange", ORANGES)


def ramp(values, cmap, lo=0.15, hi=1.0):
    values = list(values)
    if len(values) == 1:
        return {values[0]: cmap(hi)}
    return {v: cmap(lo + (hi - lo) * i / (len(values) - 1)) for i, v in enumerate(values)}


plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 9.5, "axes.titleweight": "medium", "axes.labelsize": 8.5,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 1.6, "legend.frameon": False, "legend.fontsize": 8,
    "figure.dpi": 200, "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))


def sel(df, **kw):
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        m &= df[k].isin(v).values if isinstance(v, (list, tuple, set)) else (df[k] == v).values
    return df[m]


def hit(tau):
    return (tau > 0).astype(float)


# ------------------------------------------------------------------- data --
RS = pd.read_parquet(os.path.join(SWEEP, "rounds.parquet"))
RU = pd.read_parquet(os.path.join(SWEEP, "runs.parquet"))
ALPHAS = sorted(RS.alpha.unique())
THETAS = sorted(RS.theta_star.unique())
C_LIST = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
ALPHA_MIN = 1.5                                            # pooled rates exclude alpha = 1

WR = pd.read_parquet(os.path.join(SWITCH, "rounds.parquet"))
WA = pd.read_parquet(os.path.join(SWITCH, "aligned.parquet"))
WE = pd.read_parquet(os.path.join(SWITCH, "evasion.parquet"))
WEr = pd.read_parquet(os.path.join(SWITCH, "evasion_rounds.parquet"))
WT = pd.read_parquet(os.path.join(SWITCH, "runs.parquet"),
                     columns=["study", "alpha", "psi_star", "theta_star", "sim", "c",
                              "switch_type", "horizon", "tau"])
WT = WT[WT.horizon == "150"].drop(columns="horizon")
W_ALPHAS = sorted(WR.alpha.unique())
W_CS = sorted(WR.c.unique())

FOCUS = dict(alpha=3.0, c=3.5)
TABLES = {}

# ======================================================================
# Part 1: S1 speaker, credulous L1 (full sweep)
# ======================================================================

# --- F_S1: null calibration ------------------------------------------------
KEYS = RU[["theta_star", "psi_star", "alpha"]].reset_index(drop=True)
fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex="col")
acol = ramp(ALPHAS, seq_blue)
for i, sc in enumerate(["surp1", "surp2"]):
    px = SC_PREFIX[sc]
    PS = pd.read_parquet(os.path.join(SWEEP, f"panels_S_{px}.parquet")).values   # per-run S_t = t * sus(t)
    PV = pd.read_parquet(os.path.join(SWEEP, f"panels_V_{px}.parquet")).values   # per-run V_t = t * sigma_bar^2(t)
    null = sel(RS, psi_star="inf")
    for a in ALPHAS:
        d = sel(null, alpha=a).groupby("round")[f"{px}_Sus"].mean()
        axes[i, 0].plot(d.index, d.values, color=acol[a], lw=1.1)
        m = ((KEYS.psi_star == "inf") & (KEYS.alpha == a)).values
        # Var[sus(t)] / (sigma_bar^2(t) / t) == Var[S_t] / V_t; pooled over theta* by
        # summing the per-cell numerators and denominators
        num = np.zeros(150); den = np.zeros(150)
        for th in THETAS:
            mm = m & (KEYS.theta_star == th).values
            num += PS[mm].var(axis=0, ddof=1)
            den += PV[mm].mean(axis=0)
        axes[i, 1].plot(np.arange(1, 151), num / den, color=acol[a], lw=1.1)
    axes[i, 0].axhline(0, color=INK, lw=0.8, ls="--")
    axes[i, 1].axhline(1, color=INK, lw=0.8, ls="--")
    axes[i, 0].set_ylabel(sc + "\nmean suspicion score sus(t)")
    axes[i, 1].set_ylabel("Var[sus(t)] / (σ̄²(t) / t)")
    axes[i, 0].set_ylim(-0.25, 0.1)
    axes[i, 1].set_ylim(0.6, 1.4)
axes[0, 0].set_title("running mean under the informative S1")
axes[0, 1].set_title("variance of the running mean, empirical / analytic")
for ax in axes[1]:
    ax.set_xlabel("round t")
handles = [plt.Line2D([], [], color=acol[a], lw=1.6, label=f"{a:g}") for a in ALPHAS]
axes[0, 0].legend(handles=handles, title="α", ncol=3, loc="lower right", fontsize=7, title_fontsize=7.5)
fig.tight_layout()
save(fig, "S1_null_calibration.png")

# pooled numbers for the text
for sc in ["surp1", "surp2"]:
    px = SC_PREFIX[sc]
    null = sel(RS, psi_star="inf")
    TABLES[f"null_{sc}"] = dict(
        mean_sus150=float(sel(null, round=150)[f"{px}_Sus"].mean()),
        var_ratio=float((null[f"{px}_score_sd"] ** 2).mean() / null[f"{px}_sigma2"].mean()),
    )

# --- F_S2: separation at the horizon --------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
dp = {}
for ax, sc in zip(axes, ["surp1", "surp2"]):
    px = SC_PREFIX[sc]
    for psi in ["inf", "high", "low"]:
        x = sel(RU, alpha=FOCUS["alpha"], psi_star=psi)[f"Sus150_{px}"]
        ax.hist(x, bins=70, histtype="step", density=True, lw=1.4, color=PSI_COL[psi], label=PSI_LAB[psi])
    ax.axvline(0, color=INK, lw=0.8, ls="--")
    ax.set_title(SC_LAB[sc])
    ax.set_xlabel("suspicion score sus(150)")
    # d' over the grid
    M = []
    for a in ALPHAS:
        for th in THETAS:
            n = sel(RU, alpha=a, theta_star=th, psi_star="inf")[f"Sus150_{px}"]
            p = sel(RU, alpha=a, theta_star=th, psi_star=["high", "low"])[f"Sus150_{px}"]
            M.append((p.mean() - n.mean()) / n.std(ddof=1))
    dp[sc] = float(np.nanmedian(M))
axes[0].set_ylabel("density")
axes[0].legend(loc="upper right")
fig.tight_layout()
save(fig, "S2_separation.png")
TABLES["dprime_median"] = dp

# --- rates --------------------------------------------------------------------
def sweep_rates(sc, c, alphas=None, horizon=None):
    px = SC_PREFIX[sc]
    d = RU if alphas is None else sel(RU, alpha=alphas)
    tau = d[f"tau_{px}_c{c:g}"]
    h = (tau > 0) if horizon is None else ((tau > 0) & (tau <= horizon))
    d = d.assign(hit=h.astype(float), tau=tau)
    return d


rows = []
for sc in ["surp1", "surp2"]:
    for c in C_LIST:
        d = sweep_rates(sc, c, alphas=[a for a in ALPHAS if a >= ALPHA_MIN])
        fa = d[d.psi_star == "inf"].hit.mean()
        tp_hi = d[d.psi_star == "high"].hit.mean()
        tp_lo = d[d.psi_star == "low"].hit.mean()
        pers = d[d.psi_star != "inf"]
        med = pers[pers.tau > 0].tau.median()
        rows.append(dict(score=sc, c=c, fpr=fa, tpr_high=tp_hi, tpr_low=tp_lo, med_tau=med))
TABLES["sweep_rates"] = rows

# --- F_S3: ROC + rates against alpha ---------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
for ax, sc in zip(axes, ["surp1", "surp2"]):
    for a in ALPHAS:
        pts = []
        for c in C_LIST:
            d = sweep_rates(sc, c, alphas=[a])
            pts.append((d[d.psi_star == "inf"].hit.mean(), d[d.psi_star != "inf"].hit.mean()))
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], "-o", color=acol[a], ms=3, lw=1.2, label=f"{a:g}")
    ax.plot([0, 0.35], [0, 0.35], ls=":", color=MUTED, lw=0.8)
    ax.set_xlim(-0.01, 0.35)
    ax.set_ylim(0, 1.03)
    ax.set_title(SC_LAB[sc])
    ax.set_xlabel("false-alarm rate (informative S1)")
axes[0].set_ylabel("power (persuasive S1)")
axes[1].legend(title="α", ncol=2, loc="lower right", fontsize=7, title_fontsize=7.5)
fig.tight_layout()
save(fig, "S3_roc.png")

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
ccol = ramp([2.0, 3.5, 5.0], seq_orange, lo=0.35)
for ax, sc in zip(axes, ["surp1", "surp2"]):
    for c in [2.0, 3.5, 5.0]:
        fa, pw = [], []
        for a in ALPHAS:
            d = sweep_rates(sc, c, alphas=[a])
            fa.append(d[d.psi_star == "inf"].hit.mean())
            pw.append(d[d.psi_star != "inf"].hit.mean())
        ax.plot(ALPHAS, fa, "-o", color=ccol[c], ms=3, lw=1.3, label=f"c = {c:g}")
        ax.plot(ALPHAS, pw, "--s", color=ccol[c], ms=3, lw=1.3)
    ax.set_xscale("log")
    ax.set_xticks(ALPHAS)
    ax.set_xticklabels([f"{a:g}" for a in ALPHAS], fontsize=7)
    ax.set_xlabel("speaker rationality α")
    ax.set_title(SC_LAB[sc])
    ax.minorticks_off()
axes[0].set_ylabel("rate by round 150")
axes[0].text(0.03, 0.55, "solid: false alarm\ndashed: power", transform=axes[0].transAxes, fontsize=7.5, color=INK2)
axes[1].legend(loc="center right")
fig.tight_layout()
save(fig, "S4_rates_vs_alpha.png")

# --- F_S5: timing --------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
T = np.arange(1, 151)
for sc in ["surp1", "surp2"]:
    d = sweep_rates(sc, FOCUS["c"], alphas=[FOCUS["alpha"]])
    for psi, ls in [("inf", ":"), ("high", "-"), ("low", "--")]:
        tau = d[d.psi_star == psi].tau.values
        curve = [(np.sum((tau > 0) & (tau <= t))) / len(tau) for t in T]
        axes[0].plot(T, curve, color=SC_COL[sc], ls=ls, lw=1.5)
axes[0].set_xscale("log")
axes[0].set_xlabel("round t")
axes[0].set_ylabel("fraction of runs crossed by t")
axes[0].set_title(f"cumulative crossings, α = {FOCUS['alpha']:g}, c = {FOCUS['c']:g}")
h1 = [plt.Line2D([], [], color=SC_COL[s], lw=1.6, label=s) for s in ["surp1", "surp2"]]
h2 = [plt.Line2D([], [], color=INK2, ls=ls, lw=1.4, label=PSI_LAB[p]) for p, ls in [("inf", ":"), ("high", "-"), ("low", "--")]]
axes[0].legend(handles=h1 + h2, loc="upper left", fontsize=7)

for sc in ["surp1", "surp2"]:
    med = []
    for a in ALPHAS:
        d = sweep_rates(sc, FOCUS["c"], alphas=[a])
        p = d[(d.psi_star != "inf") & (d.tau > 0)]
        med.append(p.tau.median())
    axes[1].plot(ALPHAS, med, "-o", color=SC_COL[sc], ms=3.5, label=sc)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xticks(ALPHAS)
axes[1].set_xticklabels([f"{a:g}" for a in ALPHAS], fontsize=7)
axes[1].minorticks_off()
axes[1].set_xlabel("speaker rationality α")
axes[1].set_ylabel("median τ over detected persuasive runs")
axes[1].set_title(f"detection latency, c = {FOCUS['c']:g}")
axes[1].legend()
fig.tight_layout()
save(fig, "S5_latency.png")

# --- F_S6: rate surfaces ----------------------------------------------------
def heat(ax, M, xs, ys, title, vmin=0, vmax=1, cmap=seq_blue, fmt="{:.2f}"):
    im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=7)
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels([f"{y:g}" for y in ys], fontsize=7)
    ax.grid(False)
    ax.set_title(title)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=5.8,
                    color="white" if (v - vmin) / (vmax - vmin) > 0.55 else INK)
    return im


fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))
for i, sc in enumerate(["surp1", "surp2"]):
    FA = np.zeros((len(ALPHAS), len(THETAS)))
    PW = np.zeros_like(FA)
    for r, a in enumerate(ALPHAS):
        for j, th in enumerate(THETAS):
            d = sweep_rates(sc, FOCUS["c"], alphas=[a])
            d = d[d.theta_star == th]
            FA[r, j] = d[d.psi_star == "inf"].hit.mean()
            PW[r, j] = d[d.psi_star != "inf"].hit.mean()
    heat(axes[i, 0], FA, THETAS, ALPHAS, f"{sc}: false-alarm rate", vmax=0.3)
    heat(axes[i, 1], PW, THETAS, ALPHAS, f"{sc}: power")
    axes[i, 0].set_ylabel("α")
for ax in axes[1]:
    ax.set_xlabel("θ*")
fig.tight_layout()
save(fig, "S6_rate_surfaces.png")

# ======================================================================
# Part 2: switching listener
# ======================================================================
st = "hard"

# --- F_W1: accuracy of the three listeners over time ------------------------
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4), sharex=True, sharey=True)
for i, study in enumerate(["A", "B"]):
    for j, psi in enumerate(["inf", "high", "low"]):
        ax = axes[i, j]
        d = sel(WR, study=study, psi_star=psi, alpha=FOCUS["alpha"], c=FOCUS["c"], switch_type=st)
        g = d.groupby("round")[["abias_cred", "abias_vig", "abias_switch"]].mean()
        for lis in ["cred", "vig", "switch"]:
            ax.plot(g.index, g[f"abias_{lis}"], color=LIS_COL[lis], lw=1.5,
                    ls="--" if lis == "switch" else "-", label=LIS_LAB[lis])
        ax.set_xscale("log")
        if i == 0:
            ax.set_title(f"{PSI_LAB[psi]} speaker")
        if j == 0:
            ax.set_ylabel(f"{STUDY_LAB[study]}\nmean |E[θ] − θ*|")
        if i == 1:
            ax.set_xlabel("round t")
axes[0, 2].legend(loc="upper right")
fig.tight_layout()
save(fig, "W1_listener_accuracy.png")

# time-averaged |bias| tables (study B and A), pooled over theta*
def tavg_table(study):
    rows = []
    for a in W_ALPHAS:
        r = dict(alpha=a)
        for psi in ["inf", "high", "low"]:
            d = sel(WR, study=study, psi_star=psi, alpha=a, c=FOCUS["c"], switch_type=st)
            for lis in ["cred", "vig", "switch"]:
                r[f"{psi}_{lis}"] = float(d[f"abias_{lis}"].mean())
        rows.append(r)
    return rows


TABLES["tavg_B"] = tavg_table("B")
TABLES["tavg_A"] = tavg_table("A")

# --- F_W2: cost-benefit plane (regret) ------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
wacol = ramp(W_ALPHAS, seq_blue)
for ax, study in zip(axes, ["A", "B"]):
    for a in W_ALPHAS:
        xs, ys, labs = [], [], []
        for c in W_CS:
            def regret(psis):
                d = sel(WR, study=study, psi_star=psis, alpha=a, c=c, switch_type=st)
                g = d.groupby(["theta_star", "psi_star"])[["abias_cred", "abias_vig", "abias_switch"]].mean()
                return float((g.abias_switch - np.minimum(g.abias_cred, g.abias_vig)).mean())
            xs.append(regret(["inf"]))
            ys.append(regret(["high", "low"]))
            labs.append(c)
        ax.plot(xs, ys, "-o", color=wacol[a], ms=3.2, lw=1.1, label=f"{a:g}")
        if a in (1.0, 3.0, 10.0):
            for x, y, c in zip(xs, ys, labs):
                ax.annotate(f"{c:g}", (x, y), fontsize=5.5, color=INK2, xytext=(2, 2), textcoords="offset points")
    ax.axhline(0, color=INK, lw=0.7, ls="--")
    ax.axvline(0, color=INK, lw=0.7, ls="--")
    ax.set_title(STUDY_LAB[study])
    ax.set_xlabel("regret against the informative speaker")
axes[0].set_ylabel("regret against a persuasive speaker")
axes[0].text(0.98, 0.97, "lower is better on both axes", transform=axes[0].transAxes, ha="right", va="top", fontsize=7, color=INK2)
axes[1].legend(title="α", ncol=2, fontsize=7, title_fontsize=7.5, loc="upper right")
fig.tight_layout()
save(fig, "W2_regret_plane.png")

# --- F_W3: level mismatch: alarm rates for S1 vs S2 speakers ---------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
for c, ls in [(3.5, "-"), (5.0, "--")]:
    for study, col in [("A", C_MAGENTA), ("B", C_BLUE)]:
        fa, pw = [], []
        for a in W_ALPHAS:
            d = sel(WT, study=study, alpha=a, c=c, switch_type=st)
            fa.append(hit(d[d.psi_star == "inf"].tau).mean())
            pw.append(hit(d[d.psi_star != "inf"].tau).mean())
        axes[0].plot(W_ALPHAS, fa, ls=ls, marker="o", ms=3, color=col, lw=1.4,
                     label=f"{STUDY_LAB[study].split(' (')[0]}, c = {c:g}")
        axes[1].plot(W_ALPHAS, pw, ls=ls, marker="o", ms=3, color=col, lw=1.4)
for ax in axes:
    ax.set_xscale("log")
    ax.set_xticks(W_ALPHAS)
    ax.set_xticklabels([f"{a:g}" for a in W_ALPHAS], fontsize=7)
    ax.minorticks_off()
    ax.set_xlabel("speaker rationality α")
axes[0].set_title("switch rate under the informative speaker")
axes[1].set_title("switch rate under a persuasive speaker")
axes[0].set_ylabel("fraction switched by round 150")
axes[0].legend(loc="upper left", fontsize=7)
fig.tight_layout()
save(fig, "W3_level_mismatch.png")

TABLES["B_honest_alarm"] = [
    dict(alpha=a, **{f"c{c:g}": float(hit(sel(WT, study="B", alpha=a, c=c, switch_type=st, psi_star="inf").tau).mean())
                     for c in W_CS}) for a in W_ALPHAS]
TABLES["A_honest_alarm"] = [
    dict(alpha=a, **{f"c{c:g}": float(hit(sel(WT, study="A", alpha=a, c=c, switch_type=st, psi_star="inf").tau).mean())
                     for c in W_CS}) for a in W_ALPHAS]
TABLES["B_pers_power"] = [
    dict(alpha=a, **{f"c{c:g}": float(hit(sel(WT, study="B", alpha=a, c=c, switch_type=st, psi_star=["high", "low"]).tau).mean())
                     for c in W_CS}) for a in W_ALPHAS]

# --- F_W4: recovery after the switch, hard vs soft ------------------------
th0 = 0.3
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6), sharex=True)
for j, psi in enumerate(["inf", "high", "low"]):
    for stt, col in [("hard", C_AQUA), ("soft", C_VIOLET)]:
        d = sel(WA, study="B", psi_star=psi, theta_star=th0, alpha=FOCUS["alpha"], c=FOCUS["c"], switch_type=stt)
        d = d.sort_values("k")
        axes[0, j].plot(d.k, d.bias_switch, color=col, lw=1.5, label=f"{stt} switch")
        axes[1, j].plot(d.k, d[f"p_psi_switch_{psi}"], color=col, lw=1.5, label=f"{stt} switch")
    d = sel(WA, study="B", psi_star=psi, theta_star=th0, alpha=FOCUS["alpha"], c=FOCUS["c"], switch_type="hard").sort_values("k")
    axes[0, j].plot(d.k, d.bias_vig, color=LIS_COL["vig"], lw=1.2, ls="--", label="always vigilant")
    axes[0, j].plot(d.k, d.bias_cred, color=LIS_COL["cred"], lw=1.2, ls=":", label="always credulous")
    axes[1, j].plot(d.k, d[f"p_psi_vig_{psi}"], color=LIS_COL["vig"], lw=1.2, ls="--", label="always vigilant")
    axes[1, j].axhline(1 / 3, color=MUTED, lw=0.8, ls=":")
    for ax in axes[:, j]:
        ax.axvline(0, color=INK, lw=0.8)
    axes[0, j].axhline(0, color=INK, lw=0.6, ls="--")
    axes[0, j].set_title(f"{PSI_LAB[psi]} speaker")
    axes[1, j].set_xlabel("rounds since the switch, k = t − τ")
    axes[1, j].set_ylim(0, 1.02)
axes[0, 0].set_ylabel("signed bias E[θ] − θ*")
axes[1, 0].set_ylabel("P(ψ = ψ*) after the switch")
hh, ll = axes[0, 0].get_legend_handles_labels()
fig.legend(hh, ll, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "W4_switch_recovery.png")

# --- F_W5: the S2 speaker near the boundary -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
bins = list(WE.margin_bin.unique())
order = ['(-inf, -2.0]', '(-2.0, -1.0]', '(-1.0, -0.5]', '(-0.5, -0.25]', '(-0.25, -0.1]',
         '(-0.1, -0.05]', '(-0.05, -0.02]', '(-0.02, -0.01]', '(-0.01, 0.0]']
for a in W_ALPHAS:
    d = sel(WE, study="B", psi_star=["high", "low"], alpha=a, c=FOCUS["c"], switch_type=st)
    g = d.groupby("margin_bin").apply(lambda x: np.average(x.went_informative, weights=x.n)).reindex(order)
    axes[0].plot(range(len(order)), g.values, "-o", ms=3, color=wacol[a], lw=1.2, label=f"{a:g}")
    d = sel(WEr, study="B", psi_star=["high", "low"], alpha=a, c=FOCUS["c"], switch_type=st)
    g = d.groupby("round").apply(lambda x: pd.Series(dict(p=np.average(x.went_informative, weights=x.n), n=x.n.sum())))
    g = g[(g.n >= 300) & (g.index <= 40)]
    axes[1].plot(g.index, g.p, "-o", ms=2.2, color=wacol[a], lw=1.1)
axes[0].set_xticks(range(len(order)))
axes[0].set_xticklabels(["≤ −2", "(−2, −1]", "(−1, −.5]", "(−.5, −.25]", "(−.25, −.1]", "(−.1, −.05]", "(−.05, −.02]", "(−.02, −.01]", "(−.01, 0]"], fontsize=6, rotation=35, ha="right")
axes[0].set_xlabel("margin sus(t) − boundary(t) before the choice")
axes[0].set_ylabel("P(chose the informative S2's utterance)")
axes[0].set_title("persuasive S2 against distance to the boundary")
axes[1].set_xlabel("round t (pre-switch rounds only)")
axes[1].set_title("the same probability against round")
axes[0].legend(title="α", ncol=4, fontsize=7, title_fontsize=7.5, loc="upper left")
axes[0].set_ylim(0.2, 0.75)
fig.tight_layout()
save(fig, "W5_speaker_evasion.png")

# --- F_W6: Fang dyads at level 2 for reference (vigilant vs credulous) -------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
for ax, psi in zip(axes, ["inf", "high"]):
    for study, lis, ls, col, lab in [("Fcred", "cred", "-", C_ORANGE, "credulous L1, S2 modelling it (Fcred)"),
                                     ("Fvig", "vig", "-", C_BLUE, "vigilant L1, S2 modelling it (Fvig)"),
                                     ("A", "cred", "--", C_ORANGE, "credulous L1, S1 speaker (A)"),
                                     ("A", "vig", "--", C_BLUE, "vigilant L1, S1 speaker (A)")]:
        d = sel(WR, study=study, psi_star=psi, alpha=FOCUS["alpha"], c=FOCUS["c"], switch_type=st)
        g = d.groupby("round")[f"abias_{lis}"].mean()
        ax.plot(g.index, g.values, color=col, ls=ls, lw=1.4, label=lab)
    ax.set_xscale("log")
    ax.set_title(f"{PSI_LAB[psi]} speaker")
    ax.set_xlabel("round t")
axes[0].set_ylabel("mean |E[θ] − θ*|")
axes[1].legend(fontsize=6.5)
fig.tight_layout()
save(fig, "W6_fang_dyads.png")

with open(os.path.join(OUT, "..", "tables.json"), "w") as f:
    json.dump(TABLES, f, indent=1)
print(json.dumps(TABLES, indent=1))
