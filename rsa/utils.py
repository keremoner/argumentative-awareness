"""
Utility functions for plotting and analysis.
"""

import pprint
import numpy as np
import matplotlib.pyplot as plt


# ── Pretty printing ──

def pretty_print(title, obj, precision=4):
    """Pretty-print a dict or list with float formatting."""
    print(f"\n=== {title} ===")
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (float, np.floating)):
                print(f"{str(k):>20}: {float(v):.{precision}g}")
            else:
                print(f"{str(k):>20}: {v}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            if isinstance(v, (float, np.floating)):
                print(f"[{i}] {float(v):.{precision}g}")
            else:
                print(f"[{i}] {v}")
    else:
        pprint.pprint(obj)


# ── Plotting ──

UTTERANCE_ORDER_SIMPLE = [
    ("none", "ineffective"),
    ("none", "effective"),
    ("some", "ineffective"),
    ("some", "effective"),
    ("most", "ineffective"),
    ("most", "effective"),
    ("all", "ineffective"),
    ("all", "effective"),
]


def plot_speaker_distribution(distribution, title="Speaker Distribution", alpha=1.0, ax=None):
    """Plot speaker utterance probabilities as horizontal bar chart."""
    utterances = [f"{q2}, {pred}" for (q2, pred) in UTTERANCE_ORDER_SIMPLE]
    probs = [distribution.get((q2, pred), 0.0) for (q2, pred) in UTTERANCE_ORDER_SIMPLE]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.barh(utterances, probs, color="steelblue")
    ax.set_xlabel("Probability")
    if alpha > 0:
        ax.set_title(f"{title} (α={alpha})")
    else:
        ax.set_title(title)
    ax.set_xlim(0, 1)

    for bar, prob in zip(bars, probs):
        if prob > 0.01:
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{prob:.2f}", va="center", fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    return ax


def plot_speaker_grid(distributions, titles, alpha=1.0, ncols=4):
    """
    Plot multiple speaker distributions in a grid.
    distributions: list of dicts
    titles: list of strings
    """
    n = len(distributions)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    for i, (dist, title) in enumerate(zip(distributions, titles)):
        plot_speaker_distribution(dist, title=title, alpha=alpha, ax=axes[i])
    for i in range(n, len(axes)):
        axes[i].set_visible(False)
    plt.tight_layout()
    return fig


def plot_theta_posterior(belief_or_dict, title="Posterior P(θ)", ax=None):
    """Plot posterior distribution over theta."""
    if hasattr(belief_or_dict, 'marginal'):
        theta_dist = belief_or_dict.marginal(0)
    elif hasattr(belief_or_dict, 'as_dict'):
        theta_dist = belief_or_dict.as_dict()
    else:
        theta_dist = belief_or_dict

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    thetas = sorted(theta_dist.keys())
    probs = [theta_dist[t] for t in thetas]
    ax.bar(thetas, probs, width=0.08, color="coral", alpha=0.8)
    ax.set_xlabel("θ (Treatment Effectiveness)")
    ax.set_ylabel("P(θ|u)")
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    plt.tight_layout()
    return ax


def plot_theta_learning(listener, true_theta, title="Theta Learning", ax=None):
    """Plot the evolution of listener's expected theta over rounds."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    expected_thetas = []
    for belief in listener.hist:
        theta_dist = belief.marginal(0) if hasattr(belief.values[0], '__iter__') else belief.as_dict()
        e_theta = sum(t * p for t, p in theta_dist.items())
        expected_thetas.append(e_theta)

    ax.plot(expected_thetas, label="E[θ]", linewidth=2)
    ax.axhline(y=true_theta, color='r', linestyle='--', label=f"True θ={true_theta}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Expected θ")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return ax


def plot_psi_learning(listener, title="Speaker Type Learning", ax=None):
    """Plot the evolution of listener's belief about speaker type over rounds."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    psi_history = {psi: [] for psi in listener.psis}
    for belief in listener.hist:
        psi_dist = belief.marginal(1)
        for psi in listener.psis:
            psi_history[psi].append(psi_dist.get(psi, 0.0))

    for psi, probs in psi_history.items():
        ax.plot(probs, label=f'"{psi}"', linewidth=2)

    ax.set_xlabel("Round")
    ax.set_ylabel("P(ψ|u)")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    return ax


# ── Analysis helpers ──

def expected_theta(theta_dist):
    """Expected value of theta from a distribution dict."""
    return sum(theta * prob for theta, prob in theta_dist.items())


def find_median(theta_dist):
    """Median of a discrete theta distribution."""
    sorted_thetas = sorted(theta_dist.items())
    cumulative_prob = 0.0
    prev_theta = sorted_thetas[0][0]
    for theta, prob in sorted_thetas:
        cumulative_prob += prob
        if cumulative_prob == 0.5:
            return theta
        if cumulative_prob > 0.5:
            return (prev_theta + theta) / 2
        prev_theta = theta
    return sorted_thetas[-1][0]


def persuasiveness_result(speaker_dist, listener):
    """
    Expected theta the listener would infer given speaker's utterance distribution.
    E_u[E_L[theta | u]] where u ~ speaker_dist.
    """
    result = 0
    for utt, prob in speaker_dist.items():
        result += expected_theta(listener.infer_state(utt).marginal(0)) * prob
    return result


def psi_inference_result(speaker_dist, listener):
    """Expected psi inference given speaker's utterance distribution."""
    result = 0
    for utt, prob in speaker_dist.items():
        result += expected_theta(listener.infer_state(utt).marginal(1)) * prob
    return result
