"""GPU baseline experiments: DBC, CBM on CausalShift environments.

Trains each baseline on the source environment, then evaluates
transfer performance under graph-preserving shift.

Usage:
    python -m experiments.gpu_baselines --baseline dbc --env xor --seeds 30 --output results/gpu/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from causalshift.envs.xor import CausalShiftXOR
from causalshift.envs.chain import CausalShiftChain
from causalshift.envs.branch import CausalShiftBranch
from causalshift.baselines.dbc_baseline import DBCBaseline, DBCConfig
from causalshift.baselines.cbm_baseline import CBMBaseline, CBMConfig

HORIZON = 200
TRANSFER_EPISODES = 100
SHIFT_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

ENV_FACTORIES = {
    "xor": {
        "make_source": lambda: CausalShiftXOR(theta=0.0, horizon=HORIZON),
        "make_shifted": lambda theta: CausalShiftXOR(theta=theta, horizon=HORIZON),
        "state_dim": 2,
        "n_actions": 2,
    },
    "chain": {
        "make_source": lambda: CausalShiftChain(horizon=HORIZON),
        "make_shifted": lambda p3: CausalShiftChain.make_shifted(p3=p3, horizon=HORIZON),
        "state_dim": 5,
        "n_actions": 2,
    },
    "branch": {
        "make_source": lambda: CausalShiftBranch(mu=0.0, horizon=HORIZON),
        "make_shifted": lambda mu: CausalShiftBranch.make_shifted(mu=mu, horizon=HORIZON),
        "state_dim": 4,
        "n_actions": 2,
    },
}


def evaluate_transfer(baseline, env, n_episodes: int) -> dict:
    """Evaluate frozen baseline on shifted environment."""
    episode_rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=ep * 1000)
        total_reward = 0.0

        for _ in range(HORIZON):
            action = baseline.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        episode_rewards.append(total_reward)

    optimal = HORIZON * 1.0
    regrets = [optimal - r for r in episode_rewards]

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "mean_regret": float(np.mean(regrets)),
        "std_regret": float(np.std(regrets, ddof=1)),
        "total_regret": float(np.sum(regrets)),
        "ci95_regret": float(1.96 * np.std(regrets, ddof=1) / np.sqrt(n_episodes)),
    }


def run_single_seed(
    baseline_name: str,
    env_name: str,
    seed: int,
) -> dict:
    """Train baseline on source, evaluate transfer across all shifts."""
    env_config = ENV_FACTORIES[env_name]
    np.random.seed(seed)

    source_env = env_config["make_source"]()
    state_dim = env_config["state_dim"]
    n_actions = env_config["n_actions"]

    # Create baseline
    if baseline_name == "dbc":
        baseline = DBCBaseline(state_dim, n_actions, DBCConfig())
    elif baseline_name == "cbm":
        baseline = CBMBaseline(state_dim, n_actions, CBMConfig())
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")

    # Phase 1: Collect source data and train
    print(f"    Collecting source data...", flush=True)
    transitions = baseline.collect_source_data(source_env, n_episodes=100)

    print(f"    Training {baseline_name}...", flush=True)
    if baseline_name == "dbc":
        train_info = baseline.train_encoder(transitions)
        print(f"    Encoder loss: {train_info['final_loss']:.4f}", flush=True)
    elif baseline_name == "cbm":
        causal_scores = baseline.learn_causal_graph(transitions)
        print(f"    Causal scores: {causal_scores}", flush=True)
        print(f"    Selected causal indices: {baseline.causal_indices}", flush=True)

    print(f"    Learning policy...", flush=True)
    baseline.learn_policy(source_env, n_episodes=200)

    # Phase 2: Evaluate transfer (frozen)
    results = {}
    for theta in SHIFT_LEVELS:
        shifted_env = env_config["make_shifted"](theta)
        transfer_result = evaluate_transfer(baseline, shifted_env, TRANSFER_EPISODES)
        results[f"shift_{theta:.2f}"] = {
            "shift_value": theta,
            **transfer_result,
        }

    return {
        "seed": seed,
        "baseline": baseline_name,
        "env": env_name,
        "results": results,
        "metadata": {
            "causal_indices": getattr(baseline, "causal_indices", None),
        },
    }


def main(
    baseline_name: str,
    env_name: str,
    n_seeds: int,
    output_dir: str,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Baseline: {baseline_name} | Env: {env_name} | Seeds: {n_seeds}")

    all_results = []
    for seed in range(n_seeds):
        print(f"\n  Seed {seed + 1}/{n_seeds}", flush=True)
        t = time.time()
        result = run_single_seed(baseline_name, env_name, seed)
        elapsed = time.time() - t
        all_results.append(result)
        print(f"    Done in {elapsed:.0f}s", flush=True)

    # Aggregate across seeds
    summary = {}
    for theta in SHIFT_LEVELS:
        key = f"shift_{theta:.2f}"
        regrets = [r["results"][key]["total_regret"] for r in all_results]
        summary[key] = {
            "shift_value": theta,
            "mean_total_regret": float(np.mean(regrets)),
            "std_total_regret": float(np.std(regrets, ddof=1)),
            "ci95_total_regret": float(1.96 * np.std(regrets, ddof=1) / np.sqrt(n_seeds)),
            "n_seeds": n_seeds,
        }

    # Save
    filename = f"{baseline_name}_{env_name}.json"
    results_file = output_path / filename
    with open(results_file, "w") as f:
        json.dump({
            "baseline": baseline_name,
            "env": env_name,
            "summary": summary,
            "per_seed": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {baseline_name} on {env_name}")
    print(f"{'='*60}")
    for theta in SHIFT_LEVELS:
        key = f"shift_{theta:.2f}"
        s = summary[key]
        print(f"  theta={theta:.2f}: regret={s['mean_total_regret']:8.1f} +/- {s['ci95_total_regret']:6.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True, choices=["dbc", "cbm"])
    parser.add_argument("--env", type=str, default="xor", choices=["xor", "chain", "branch"])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--output", type=str, default="results/gpu")
    args = parser.parse_args()
    main(args.baseline, args.env, args.seeds, args.output)
