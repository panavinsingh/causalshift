"""Router experiment: Compare adaptive, oracle, random, always-m, always-p.

For each shift level, runs all 5 routing strategies and records:
  - Total reward (performance)
  - Fraction of steps using phi_m (compute proxy)
  - Regret vs oracle

Usage:
    python -m experiments.router_experiment --seeds 30 --output results/router/
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
from causalshift.envs.xor import CausalShiftXOR
from causalshift.router.adaptive import AdaptiveRouter, OracleRouter, RandomRouter

HORIZON = 200
N_EPISODES = 100
SHIFT_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def action_fn(z):
    return int(z) if isinstance(z, (int, np.integer)) else int(z[0])


def run_always_strategy(
    env, phi, n_episodes: int, horizon: int
) -> dict:
    """Run episodes always using one abstraction."""
    total_reward = 0.0
    for ep in range(n_episodes):
        state, _ = env.reset(seed=ep * 1000)
        for _ in range(horizon):
            z = phi.abstract(state)
            action = action_fn(z)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
    return {
        "total_reward": total_reward,
        "regret": horizon * n_episodes - total_reward,
        "phi_m_fraction": 1.0 if "mech" in phi.name else 0.0,
    }


def run_adaptive(
    theta: float, n_episodes: int, horizon: int, seed: int, cost_ratio: float = 1.0,
) -> dict:
    """Run the adaptive EXP3 router."""
    phi_m = MechanismInvariantAbstraction.for_xor()
    phi_p = PredictiveAbstraction.for_xor()
    router = AdaptiveRouter(
        phi_m=phi_m, phi_p=phi_p, action_fn=action_fn,
        cost_ratio=cost_ratio, horizon_estimate=horizon * n_episodes,
    )

    rng = np.random.default_rng(seed)
    total_reward = 0.0

    for ep in range(n_episodes):
        env = CausalShiftXOR(theta=theta, horizon=horizon)
        state, _ = env.reset(seed=ep * 1000 + seed)

        for _ in range(horizon):
            action, chosen = router.select_and_act(state, rng)
            state, reward, terminated, truncated, _ = env.step(action)
            router.update(reward, chosen)
            total_reward += reward
            if terminated or truncated:
                break

    return {
        "total_reward": total_reward,
        "regret": horizon * n_episodes - total_reward,
        "phi_m_fraction": router.get_routing_fraction(),
    }


def run_random_router(
    theta: float, n_episodes: int, horizon: int, seed: int,
) -> dict:
    phi_m = MechanismInvariantAbstraction.for_xor()
    phi_p = PredictiveAbstraction.for_xor()
    router = RandomRouter(phi_m, phi_p, action_fn)

    rng = np.random.default_rng(seed)
    total_reward = 0.0
    n_m = 0
    n_total = 0

    for ep in range(n_episodes):
        env = CausalShiftXOR(theta=theta, horizon=horizon)
        state, _ = env.reset(seed=ep * 1000 + seed)

        for _ in range(horizon):
            action, chosen = router.select_and_act(state, rng)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            n_total += 1
            if chosen == "mechanism":
                n_m += 1
            if terminated or truncated:
                break

    return {
        "total_reward": total_reward,
        "regret": horizon * n_episodes - total_reward,
        "phi_m_fraction": n_m / n_total if n_total > 0 else 0.5,
    }


def main(n_seeds: int, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for theta in SHIFT_LEVELS:
        print(f"\ntheta={theta:.2f}")
        all_results[f"theta_{theta:.2f}"] = {}

        for strategy_name in ["always_m", "always_p", "adaptive", "random"]:
            seed_results = []

            for seed in range(n_seeds):
                if strategy_name == "always_m":
                    env = CausalShiftXOR(theta=theta, horizon=HORIZON)
                    phi = MechanismInvariantAbstraction.for_xor()
                    result = run_always_strategy(env, phi, N_EPISODES, HORIZON)
                elif strategy_name == "always_p":
                    env = CausalShiftXOR(theta=theta, horizon=HORIZON)
                    phi = PredictiveAbstraction.for_xor()
                    result = run_always_strategy(env, phi, N_EPISODES, HORIZON)
                elif strategy_name == "adaptive":
                    result = run_adaptive(theta, N_EPISODES, HORIZON, seed)
                elif strategy_name == "random":
                    result = run_random_router(theta, N_EPISODES, HORIZON, seed)

                seed_results.append(result)

            regrets = [r["regret"] for r in seed_results]
            fracs = [r["phi_m_fraction"] for r in seed_results]

            summary = {
                "mean_regret": float(np.mean(regrets)),
                "std_regret": float(np.std(regrets, ddof=1)) if len(regrets) > 1 else 0.0,
                "ci95_regret": float(1.96 * np.std(regrets, ddof=1) / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0,
                "mean_phi_m_fraction": float(np.mean(fracs)),
            }

            all_results[f"theta_{theta:.2f}"][strategy_name] = summary
            print(f"  {strategy_name:12s}: regret={summary['mean_regret']:8.1f}, phi_m_frac={summary['mean_phi_m_fraction']:.2f}")

    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--output", type=str, default="results/router")
    args = parser.parse_args()
    main(n_seeds=args.seeds, output_dir=args.output)
