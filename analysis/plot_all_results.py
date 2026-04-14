"""Generate ALL paper figures from experimental results.

Produces:
  Fig 1: Schematic (created manually in LaTeX/TikZ)
  Fig 2: XOR construction (created manually)
  Fig 3: Separation theorem — 3-panel (already generated)
  Fig 4: Gap scaling (already generated)
  Fig 5: LLM baselines — privileged vs blackbox vs CoT across models
  Fig 6: GPU baselines — all world models + DBC/CBM under shift
  Fig 7: Router ablation — always-m vs always-p vs adaptive vs random
  Fig 8: Combined Pareto / summary comparison
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

SHIFT_LEVELS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ============================================================
# Fig 5: LLM Baselines
# ============================================================
def plot_llm_baselines():
    models = {
        "GPT-5.4": "gpt-5.4",
        "Gemini 3.1 Pro": "gemini-3.1-pro",
        "DeepSeek V3.2": "deepseek-v3.2",
        "Qwen3-235B": "qwen3-235b",
    }
    conditions = ["privileged", "blackbox", "cot"]
    cond_labels = {"privileged": "Privileged", "blackbox": "Blackbox", "cot": "CoT"}
    cond_colors = {"privileged": "C0", "blackbox": "C1", "cot": "C2"}
    cond_markers = {"privileged": "o", "blackbox": "s", "cot": "^"}

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)

    for idx, (model_name, model_key) in enumerate(models.items()):
        ax = axes[idx]
        for cond in conditions:
            fpath = RESULTS_DIR / "llm" / f"{model_key}_{cond}.json"
            if not fpath.exists():
                continue
            data = load_json(fpath)
            thetas = []
            accs = []
            for k, v in sorted(data["summary"].items()):
                thetas.append(v["theta"])
                accs.append(v["accuracy"])
            ax.plot(thetas, accs, marker=cond_markers[cond], color=cond_colors[cond],
                    label=cond_labels[cond], markersize=5, linewidth=1.5)

        ax.set_xlabel(r"Shift $\theta$")
        if idx == 0:
            ax.set_ylabel("Accuracy")
        ax.set_title(model_name)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="Random" if idx == 0 else None)
        ax.legend(loc="lower left", framealpha=0.9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("LLM Planner Accuracy Under Graph-Preserving Shift", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_llm_baselines.pdf")
    fig.savefig(FIGURES_DIR / "fig5_llm_baselines.png")
    print("Saved fig5_llm_baselines")
    plt.close(fig)


# ============================================================
# Fig 6: GPU Baselines (World Models + DBC/CBM)
# ============================================================
def plot_gpu_baselines():
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    baselines = {
        "DBC": ("gpu/dbc_xor.json", "C0", "o"),
        "CBM": ("gpu/cbm_xor.json", "C1", "s"),
        "DreamerV3": ("gpu/dreamerv3_xor_results.json", "C2", "^"),
        "Dreamer-CDP": ("gpu/dreamer_cdp_xor_results.json", "C3", "D"),
        "TransDreamerV3": ("gpu/trans_dreamer_xor_results.json", "C4", "v"),
        "NE-Dreamer": ("gpu/ne_dreamer_xor_results.json", "C5", "P"),
    }

    # Also plot phi_m and phi_p from separation results
    sep = load_json(RESULTS_DIR / "full_separation" / "results.json")

    # phi_m (mechanism-invariant)
    thetas_m = [sep["xor"]["mechanism"][k]["shift_value"] for k in sorted(sep["xor"]["mechanism"].keys())]
    regrets_m = [sep["xor"]["mechanism"][k]["mean_total_regret"] for k in sorted(sep["xor"]["mechanism"].keys())]
    # Filter to our shift levels
    thetas_m_f = [t for t in thetas_m if t in SHIFT_LEVELS]
    regrets_m_f = [r for t, r in zip(thetas_m, regrets_m) if t in SHIFT_LEVELS]
    ax.plot(thetas_m_f, regrets_m_f, "k-", linewidth=2, label=r"$\phi_m$ (mechanism)", zorder=10)

    # phi_p (predictive) — theoretical
    thetas_theory = np.array(SHIFT_LEVELS)
    regrets_theory = thetas_theory * 200 * 100  # theta * T * n_episodes
    ax.plot(thetas_theory, regrets_theory, "k--", linewidth=2, label=r"$\phi_p$ (predictive, theory)", zorder=10)

    for name, (fpath, color, marker) in baselines.items():
        full_path = RESULTS_DIR / fpath
        if not full_path.exists():
            continue
        data = load_json(full_path)
        summary = data["summary"]

        thetas = []
        regrets = []
        for k in sorted(summary.keys()):
            sv = summary[k]["shift_value"]
            if sv in SHIFT_LEVELS:
                thetas.append(sv)
                regrets.append(summary[k]["mean_total_regret"])

        ax.plot(thetas, regrets, marker=marker, color=color, label=name,
                markersize=6, linewidth=1.5, alpha=0.8)

    ax.set_xlabel(r"Shift Magnitude $\theta$")
    ax.set_ylabel("Total Transfer Regret")
    ax.set_title("Learned Abstractions Under Shift: All Baselines on CausalShift-XOR")
    ax.legend(loc="upper left", framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_gpu_baselines.pdf")
    fig.savefig(FIGURES_DIR / "fig6_gpu_baselines.png")
    print("Saved fig6_gpu_baselines")
    plt.close(fig)


# ============================================================
# Fig 7: Router Ablation
# ============================================================
def plot_router_ablation():
    router = load_json(RESULTS_DIR / "router" / "results.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    strategies = {
        "always_m": (r"Always $\phi_m$", "C0", "o"),
        "always_p": (r"Always $\phi_p$", "C1", "s"),
        "adaptive": ("Adaptive Router", "C2", "^"),
        "random": ("Random", "C3", "D"),
    }

    for strat_key, (label, color, marker) in strategies.items():
        thetas = []
        regrets = []
        fracs = []
        for k in sorted(router.keys()):
            data = router[k][strat_key]
            theta = float(k.split("_")[1])
            thetas.append(theta)
            regrets.append(data["mean_regret"])
            fracs.append(data["mean_phi_m_fraction"])

        ax1.plot(thetas, regrets, marker=marker, color=color, label=label,
                 markersize=5, linewidth=1.5)
        ax2.plot(thetas, fracs, marker=marker, color=color, label=label,
                 markersize=5, linewidth=1.5)

    ax1.set_xlabel(r"Shift $\theta$")
    ax1.set_ylabel("Total Regret")
    ax1.set_title("Regret by Routing Strategy")
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel(r"Shift $\theta$")
    ax2.set_ylabel(r"Fraction using $\phi_m$")
    ax2.set_title("Abstraction Selection by Router")
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    fig.suptitle("Adaptive Metacontrol: EXP3-Based Router Ablation", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_router_ablation.pdf")
    fig.savefig(FIGURES_DIR / "fig7_router_ablation.png")
    print("Saved fig7_router_ablation")
    plt.close(fig)


# ============================================================
# Fig 8: Summary Table as Heatmap
# ============================================================
def plot_summary_heatmap():
    """Heatmap showing accuracy at theta=0.5 for all models × conditions."""
    models = ["GPT-5.4", "Gemini 3.1", "DeepSeek V3.2", "Qwen3-235B"]
    model_keys = ["gpt-5.4", "gemini-3.1-pro", "deepseek-v3.2", "qwen3-235b"]
    conditions = ["privileged", "blackbox", "cot"]

    data_matrix = np.zeros((len(models), len(conditions)))

    for i, mkey in enumerate(model_keys):
        for j, cond in enumerate(conditions):
            fpath = RESULTS_DIR / "llm" / f"{mkey}_{cond}.json"
            if fpath.exists():
                d = load_json(fpath)
                data_matrix[i, j] = d["summary"]["theta_0.50"]["accuracy"]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    im = ax.imshow(data_matrix, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(["Privileged", "Blackbox", "CoT"])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    for i in range(len(models)):
        for j in range(len(conditions)):
            val = data_matrix[i, j]
            color = "white" if val < 0.65 else "black"
            ax.text(j, i, f"{val:.1%}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    ax.set_title(r"LLM Accuracy at Maximum Shift ($\theta = 0.5$)", fontsize=13)
    fig.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig8_summary_heatmap.pdf")
    fig.savefig(FIGURES_DIR / "fig8_summary_heatmap.png")
    print("Saved fig8_summary_heatmap")
    plt.close(fig)


# ============================================================
# Statistical Analysis Summary
# ============================================================
def generate_statistical_summary():
    """Generate a comprehensive statistical summary JSON."""
    summary = {
        "separation_theorem": {},
        "llm_baselines": {},
        "gpu_baselines": {},
        "router": {},
    }

    # Separation theorem results
    sep = load_json(RESULTS_DIR / "full_separation" / "results.json")
    for env_name in ["xor", "chain", "branch"]:
        env_data = sep[env_name]
        summary["separation_theorem"][env_name] = {
            "phi_m_regret_at_0.5": env_data["mechanism"][sorted(env_data["mechanism"].keys())[-1]]["mean_total_regret"],
            "phi_p_regret_at_0.5": env_data["predictive"][sorted(env_data["predictive"].keys())[-1]]["mean_total_regret"],
        }

    # LLM results
    for fname in sorted((RESULTS_DIR / "llm").glob("*.json")):
        data = load_json(fname)
        model = data["model"]
        cond = data["condition"]
        key = f"{model}_{cond}"
        summary["llm_baselines"][key] = {
            "accuracy_at_0.0": data["summary"]["theta_0.00"]["accuracy"],
            "accuracy_at_0.5": data["summary"]["theta_0.50"]["accuracy"],
            "s1_match_at_0.5": data["summary"]["theta_0.50"]["s1_match_rate"],
            "s2_match_at_0.5": data["summary"]["theta_0.50"]["s2_match_rate"],
            "error_rate": data["error_rate"],
            "total_calls": data["total_calls"],
        }

    # GPU results
    for fname in sorted((RESULTS_DIR / "gpu").glob("*.json")):
        data = load_json(fname)
        name = fname.stem
        s = data["summary"]
        first_key = sorted(s.keys())[0]
        last_key = sorted(s.keys())[-1]
        summary["gpu_baselines"][name] = {
            "regret_at_source": s[first_key]["mean_total_regret"],
            "regret_at_max_shift": s[last_key]["mean_total_regret"],
            "ci95_at_max_shift": s[last_key].get("ci95_total_regret", 0),
        }

    # Router
    router = load_json(RESULTS_DIR / "router" / "results.json")
    for strat in ["always_m", "always_p", "adaptive", "random"]:
        summary["router"][strat] = {
            "regret_at_0.5": router["theta_0.50"][strat]["mean_regret"],
            "phi_m_fraction_at_0.5": router["theta_0.50"][strat]["mean_phi_m_fraction"],
        }

    out_path = RESULTS_DIR / "statistical_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_path}")
    return summary


if __name__ == "__main__":
    print("Generating all figures...\n")
    plot_llm_baselines()
    plot_gpu_baselines()
    plot_router_ablation()
    plot_summary_heatmap()
    print("\nGenerating statistical summary...")
    summary = generate_statistical_summary()
    print("\nAll figures and analysis complete.")
