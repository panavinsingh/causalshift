"""Experiment 1: Validate the separation theorem on CausalShift-XOR.

This is the paper's central empirical result. We show:
  1. Under source (theta=0): phi_m and phi_p achieve identical performance
  2. Under shift (theta>0): phi_m maintains bounded regret, phi_p suffers linear regret
  3. The gap scales with theta (shift magnitude)

Protocol P:
  Phase 1: 50 source episodes (theta=0), learner updates online
  Phase 2: 100 episodes at each shifted theta', learner continues updating

Usage:
    python -m experiments.xor_separation --seeds 30 --output results/xor_separation/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from causalshift.abstractions.mechanism_invariant import MechanismInvariantAbstraction
from causalshift.abstractions.predictive import PredictiveAbstraction
from causalshift.baselines.ucb_abstract import EpisodeResult, UCBAbstractLearner, run_episode
from causalshift.envs.xor import CausalShiftXOR


# --- Experiment Configuration ---

SHIFT_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
SOURCE_EPISODES = 50
TRANSFER_EPISODES = 100
HORIZON = 200


def optimal_value_per_step(theta: float) -> float:
    """Optimal expected reward per step at shift level theta.

    Optimal policy: A = S1 (match the cause). E[R] = 1.0 regardless of theta.
    """
    return 1.0


def run_single_seed(
    seed: int,
    abstraction_type: str,
) -> dict:
    """Run the full Protocol P for one seed and one abstraction type.

    Protocol P (corrected):
      Phase 1: Train on source (theta=0) for SOURCE_EPISODES episodes, learning online.
      Phase 2: FREEZE the learned policy. Evaluate zero-shot on each shifted theta'.
               No online updates during transfer — this measures whether the
               abstraction preserves the optimal policy under shift.

    This directly tests the theorem: phi_m's abstract MDP is invariant so
    the frozen policy stays optimal; phi_p's abstract MDP changes so the
    frozen policy becomes suboptimal.
    """
    if abstraction_type == "mechanism":
        phi = MechanismInvariantAbstraction.for_xor()
    elif abstraction_type == "predictive":
        phi = PredictiveAbstraction.for_xor()
    else:
        raise ValueError(f"Unknown abstraction: {abstraction_type}")

    learner = UCBAbstractLearner(abstraction=phi, n_actions=2)

    # Phase 1: Train on source
    source_env = CausalShiftXOR(theta=0.0, horizon=HORIZON)
    source_regrets = []
    for ep in range(SOURCE_EPISODES):
        ep_result = run_episode(source_env, learner, optimal_value_per_step=1.0)
        source_regrets.append(ep_result.regret)

    # Extract the learned greedy policy (frozen)
    learned_policy = _extract_greedy_policy(learner)

    results = {}

    # Phase 2: Zero-shot transfer with FROZEN policy (no updates)
    for theta in SHIFT_LEVELS:
        env = CausalShiftXOR(theta=theta, horizon=HORIZON)
        v_star = optimal_value_per_step(theta)

        episode_regrets = []
        cumulative_reward = 0.0

        for ep in range(TRANSFER_EPISODES):
            ep_result = _run_frozen_episode(
                env, phi, learned_policy, v_star
            )
            episode_regrets.append(ep_result.regret)
            cumulative_reward += ep_result.total_reward

        results[f"theta_{theta:.2f}"] = {
            "theta": theta,
            "n_episodes": TRANSFER_EPISODES,
            "mean_episode_regret": float(np.mean(episode_regrets)),
            "std_episode_regret": float(np.std(episode_regrets)),
            "total_regret": float(np.sum(episode_regrets)),
            "mean_reward_per_step": float(
                cumulative_reward / (TRANSFER_EPISODES * HORIZON)
            ),
            "optimal_reward_per_step": v_star,
            "episode_regrets": [float(r) for r in episode_regrets],
        }

    return {
        "seed": seed,
        "abstraction": abstraction_type,
        "source_mean_regret": float(np.mean(source_regrets)),
        "learned_policy": learned_policy,
        "results": results,
    }


def _extract_greedy_policy(learner: UCBAbstractLearner) -> dict[int, int]:
    """Extract the greedy policy from a trained UCB learner.

    Returns: dict mapping abstract_state -> best_action.
    """
    policy = {}
    for z, stats in learner._stats.items():
        best_a = max(range(len(stats)), key=lambda a: stats[a].mean)
        policy[z] = best_a
    return policy


def _run_frozen_episode(
    env,
    abstraction,
    policy: dict[int, int],
    optimal_value_per_step: float,
) -> EpisodeResult:
    """Run one episode with a FROZEN policy (no learning)."""
    state, info = env.reset()
    total_reward = 0.0
    rewards = []
    actions = []

    terminated = truncated = False
    while not (terminated or truncated):
        z = abstraction.abstract(state)
        # Use learned policy; default to action 1 if unseen state
        action = policy.get(int(z) if isinstance(z, (int, np.integer)) else z, 1)
        actions.append(action)

        state, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        total_reward += reward

    steps = len(rewards)
    regret = optimal_value_per_step * steps - total_reward

    return EpisodeResult(
        total_reward=total_reward,
        regret=regret,
        steps=steps,
        rewards=rewards,
        actions=actions,
    )


def run_experiment(n_seeds: int, output_dir: str) -> None:
    """Run the full separation experiment across seeds and abstractions."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = []

    for abstraction_type in ["mechanism", "predictive"]:
        print(f"\n{'='*60}")
        print(f"Abstraction: {abstraction_type}")
        print(f"{'='*60}")

        for seed in range(n_seeds):
            print(f"  Seed {seed + 1}/{n_seeds}...", end=" ", flush=True)
            result = run_single_seed(seed=seed, abstraction_type=abstraction_type)
            all_results.append(result)

            # Print summary for this seed
            source_regret = result["results"]["theta_0.00"]["mean_episode_regret"]
            max_shift_regret = result["results"]["theta_0.50"]["mean_episode_regret"]
            print(f"source_regret={source_regret:.2f}, shift_0.50_regret={max_shift_regret:.2f}")

    # Save raw results
    results_file = output_path / "raw_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to {results_file}")

    # Compute and save summary statistics
    summary = compute_summary(all_results, n_seeds)
    summary_file = output_path / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_file}")

    # Print the key result
    print(f"\n{'='*60}")
    print("SEPARATION RESULT")
    print(f"{'='*60}")
    for theta in SHIFT_LEVELS:
        key = f"theta_{theta:.2f}"
        mech = summary["mechanism"][key]
        pred = summary["predictive"][key]
        print(
            f"  theta={theta:.2f}: "
            f"phi_mech={mech['mean_total_regret']:8.1f} +/- {mech['ci95_total_regret']:6.1f} | "
            f"phi_pred={pred['mean_total_regret']:8.1f} +/- {pred['ci95_total_regret']:6.1f} | "
            f"gap={pred['mean_total_regret'] - mech['mean_total_regret']:8.1f}"
        )


def compute_summary(all_results: list[dict], n_seeds: int) -> dict:
    """Compute per-theta summary statistics across seeds."""
    summary = {}

    for abstraction_type in ["mechanism", "predictive"]:
        seed_results = [r for r in all_results if r["abstraction"] == abstraction_type]
        summary[abstraction_type] = {}

        for theta in SHIFT_LEVELS:
            key = f"theta_{theta:.2f}"
            total_regrets = [r["results"][key]["total_regret"] for r in seed_results]
            mean_regrets = [r["results"][key]["mean_episode_regret"] for r in seed_results]

            mean_total = float(np.mean(total_regrets))
            std_total = float(np.std(total_regrets, ddof=1))
            ci95 = 1.96 * std_total / np.sqrt(len(total_regrets))

            summary[abstraction_type][key] = {
                "theta": theta,
                "mean_total_regret": mean_total,
                "std_total_regret": std_total,
                "ci95_total_regret": float(ci95),
                "mean_episode_regret": float(np.mean(mean_regrets)),
                "n_seeds": len(seed_results),
            }

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CausalShift-XOR separation experiment")
    parser.add_argument("--seeds", type=int, default=30, help="Number of random seeds")
    parser.add_argument(
        "--output",
        type=str,
        default="results/xor_separation",
        help="Output directory",
    )
    args = parser.parse_args()

    run_experiment(n_seeds=args.seeds, output_dir=args.output)
