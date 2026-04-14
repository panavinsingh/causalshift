"""Generate paper figures for the separation theorem results.

Produces:
  - Figure 3: Regret vs shift magnitude for all 3 environments (3-panel)
  - Figure 4: Regret gap scaling (linear fit validation)

Usage:
    python -m analysis.plot_separation --input results/full_separation/results.json --output figures/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Paper-quality settings
matplotlib.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "text.usetex": False,  # Set True if LaTeX available
})

ENV_LABELS = {
    "xor": "CausalShift-XOR",
    "chain": "CausalShift-Chain",
    "branch": "CausalShift-Branch",
}

SHIFT_PARAM_LABELS = {
    "xor": r"$\theta$",
    "chain": r"$p_3$",
    "branch": r"$\mu$",
}


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_curves(env_results: dict, shift_param: str) -> tuple:
    """Extract shift values and regrets for both abstractions."""
    shifts_m, regrets_m, ci_m = [], [], []
    shifts_p, regrets_p, ci_p = [], [], []

    for key, data in sorted(env_results["mechanism"].items()):
        shifts_m.append(data["shift_value"])
        regrets_m.append(data["mean_total_regret"])
        ci_m.append(data["ci95_total_regret"])

    for key, data in sorted(env_results["predictive"].items()):
        shifts_p.append(data["shift_value"])
        regrets_p.append(data["mean_total_regret"])
        ci_p.append(data["ci95_total_regret"])

    return (
        np.array(shifts_m), np.array(regrets_m), np.array(ci_m),
        np.array(shifts_p), np.array(regrets_p), np.array(ci_p),
    )


def plot_three_panel(results: dict, output_dir: Path) -> None:
    """Figure 3: Regret vs shift magnitude, 3-panel plot."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for idx, (env_name, shift_param) in enumerate([
        ("xor", "theta"), ("chain", "p3"), ("branch", "mu")
    ]):
        ax = axes[idx]
        env_results = results[env_name]
        s_m, r_m, c_m, s_p, r_p, c_p = extract_curves(env_results, shift_param)

        # Theoretical prediction: regret = theta * T * n_episodes
        T = 200
        n_ep = 100
        theoretical = s_p * T * n_ep

        ax.fill_between(s_p, r_p - c_p, r_p + c_p, alpha=0.2, color="C1")
        ax.fill_between(s_m, r_m - c_m, r_m + c_m, alpha=0.2, color="C0")

        ax.plot(s_m, r_m, "o-", color="C0", label=r"$\phi_m$ (mechanism)", markersize=4, linewidth=1.5)
        ax.plot(s_p, r_p, "s-", color="C1", label=r"$\phi_p$ (predictive)", markersize=4, linewidth=1.5)
        ax.plot(s_p, theoretical, "--", color="gray", alpha=0.6, label=r"Theoretical: $\theta \cdot T$", linewidth=1)

        ax.set_xlabel(SHIFT_PARAM_LABELS[env_name])
        if idx == 0:
            ax.set_ylabel("Total Transfer Regret")
        ax.set_title(ENV_LABELS[env_name])
        ax.legend(loc="upper left", framealpha=0.9)
        ax.set_xlim(-0.02, 0.52)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Separation Theorem: Mechanism-Invariant vs Predictive Abstraction Under Shift",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()

    output_path = output_dir / "fig3_separation_three_panel.pdf"
    fig.savefig(output_path)
    fig.savefig(output_dir / "fig3_separation_three_panel.png")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_gap_scaling(results: dict, output_dir: Path) -> None:
    """Figure 4: Regret gap vs shift, with linear fit."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    colors = {"xor": "C0", "chain": "C2", "branch": "C3"}
    markers = {"xor": "o", "chain": "s", "branch": "^"}

    for env_name, shift_param in [("xor", "theta"), ("chain", "p3"), ("branch", "mu")]:
        env_results = results[env_name]
        s_m, r_m, _, s_p, r_p, _ = extract_curves(env_results, shift_param)
        gaps = r_p - r_m

        ax.plot(
            s_p, gaps,
            marker=markers[env_name], color=colors[env_name],
            label=ENV_LABELS[env_name], markersize=5, linewidth=1.5,
        )

    # Theoretical line
    theta = np.linspace(0, 0.5, 100)
    T = 200
    n_ep = 100
    ax.plot(theta, theta * T * n_ep, "--", color="gray", alpha=0.6, label=r"$\theta \cdot T$ (theory)")

    ax.set_xlabel("Shift Magnitude")
    ax.set_ylabel("Regret Gap (predictive - mechanism)")
    ax.set_title("Regret Gap Scales Linearly with Shift")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    output_path = output_dir / "fig4_gap_scaling.pdf"
    fig.savefig(output_path)
    fig.savefig(output_dir / "fig4_gap_scaling.png")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main(input_path: str, output_dir: str) -> None:
    results = load_results(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_three_panel(results, out)
    plot_gap_scaling(results, out)
    print("All figures generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/full_separation/results.json")
    parser.add_argument("--output", default="figures")
    args = parser.parse_args()
    main(args.input, args.output)
