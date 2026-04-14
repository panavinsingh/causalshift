"""Full separation experiment across all 3 CausalShift environments.

Runs the frozen-policy transfer protocol for phi_m and phi_p across
XOR, Chain, and Branch environments at multiple shift levels.
Produces paper-ready JSON results for plotting.

Usage:
    python -m experiments.full_separation --seeds 30 --output results/full_separation/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from causalshift.abstractions.mechanism_invariant import MechanismInvariantAbstraction
from causalshift.abstractions.predictive import PredictiveAbstraction
from causalshift.envs.branch import CausalShiftBranch
from causalshift.envs.chain import CausalShiftChain
from causalshift.envs.xor import CausalShiftXOR

HORIZON = 200
TRANSFER_EPISODES = 100
SOURCE_EPISODES = 50

# Environment configs: each defines the env class, shift levels, and abstractions
ENV_CONFIGS = {
    "xor": {
        "make_env": lambda theta: CausalShiftXOR(theta=theta, horizon=HORIZON),
        "shift_levels": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        "shift_param_name": "theta",
        "phi_m": MechanismInvariantAbstraction(causal_indices=[0], discrete=True),
        "phi_p": PredictiveAbstraction(effect_indices=[1], discrete=True),
        "action_from_z": lambda z: int(z) if isinstance(z, (int, np.integer)) else int(z[0]),
    },
    "chain": {
        "make_env": lambda p3: CausalShiftChain.make_shifted(p3=p3, horizon=HORIZON),
        "shift_levels": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        "shift_param_name": "p3",
        "phi_m": MechanismInvariantAbstraction(causal_indices=[0, 1], discrete=False),
        "phi_p": PredictiveAbstraction(effect_indices=[3, 4], discrete=False),
        "action_from_z": lambda z: int(z[0]) if hasattr(z, "__len__") else int(z),
    },
    "branch": {
        "make_env": lambda mu: CausalShiftBranch.make_shifted(mu=mu, horizon=HORIZON),
        "shift_levels": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        "shift_param_name": "mu",
        "phi_m": MechanismInvariantAbstraction(causal_indices=[0], discrete=True),
        "phi_p": PredictiveAbstraction(effect_indices=[3], discrete=True),
        "action_from_z": lambda z: int(z) if isinstance(z, (int, np.integer)) else int(z[0]),
    },
}


def run_frozen_transfer(
    env,
    abstraction,
    action_fn,
    n_episodes: int,
) -> dict:
    """Run episodes with frozen policy A = action_fn(phi(S))."""
    episode_rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=ep * 1000)
        total_reward = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            z = abstraction.abstract(state)
            action = action_fn(z)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

        episode_rewards.append(total_reward)

    optimal_per_episode = HORIZON * 1.0  # Optimal: match X1 every step
    episode_regrets = [optimal_per_episode - r for r in episode_rewards]

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards, ddof=1)),
        "mean_regret": float(np.mean(episode_regrets)),
        "std_regret": float(np.std(episode_regrets, ddof=1)),
        "total_regret": float(np.sum(episode_regrets)),
        "ci95_regret": float(1.96 * np.std(episode_regrets, ddof=1) / np.sqrt(n_episodes)),
    }


def run_env_experiment(
    env_name: str,
    config: dict,
    n_seeds: int,
) -> dict:
    """Run full separation experiment for one environment."""
    results = {"mechanism": {}, "predictive": {}}

    for abstraction_name, phi in [("mechanism", config["phi_m"]), ("predictive", config["phi_p"])]:
        action_fn = config["action_from_z"]

        for shift_val in config["shift_levels"]:
            shift_key = f"{config['shift_param_name']}_{shift_val:.2f}"
            seed_results = []

            for seed in range(n_seeds):
                env = config["make_env"](shift_val)
                result = run_frozen_transfer(
                    env=env,
                    abstraction=phi,
                    action_fn=action_fn,
                    n_episodes=TRANSFER_EPISODES,
                )
                seed_results.append(result)

            # Aggregate across seeds
            mean_regrets = [r["mean_regret"] for r in seed_results]
            total_regrets = [r["total_regret"] for r in seed_results]

            results[abstraction_name][shift_key] = {
                "shift_value": shift_val,
                "mean_total_regret": float(np.mean(total_regrets)),
                "std_total_regret": float(np.std(total_regrets, ddof=1)),
                "ci95_total_regret": float(
                    1.96 * np.std(total_regrets, ddof=1) / np.sqrt(n_seeds)
                ),
                "mean_episode_regret": float(np.mean(mean_regrets)),
                "n_seeds": n_seeds,
            }

    return results


def main(n_seeds: int, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for env_name, config in ENV_CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"Environment: {env_name.upper()}")
        print(f"{'='*70}")

        results = run_env_experiment(env_name, config, n_seeds)
        all_results[env_name] = results

        # Print summary
        for shift_val in config["shift_levels"]:
            shift_key = f"{config['shift_param_name']}_{shift_val:.2f}"
            m = results["mechanism"][shift_key]
            p = results["predictive"][shift_key]
            gap = p["mean_total_regret"] - m["mean_total_regret"]
            print(
                f"  {shift_key}: "
                f"phi_m={m['mean_total_regret']:8.1f} +/- {m['ci95_total_regret']:6.1f} | "
                f"phi_p={p['mean_total_regret']:8.1f} +/- {p['ci95_total_regret']:6.1f} | "
                f"gap={gap:8.1f}"
            )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full CausalShift separation experiment")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--output", type=str, default="results/full_separation")
    args = parser.parse_args()
    main(n_seeds=args.seeds, output_dir=args.output)
