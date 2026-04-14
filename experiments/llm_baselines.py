"""LLM baseline experiment runner — PARALLEL execution.

Runs LLM planners across shift levels on CausalShift-XOR with concurrent API calls.
Uses asyncio + ThreadPoolExecutor for parallel request execution.

Usage:
    python -m experiments.llm_baselines --model gpt-5.4 --condition privileged --seeds 5 --workers 50 --output results/llm/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from causalshift.baselines.llm_planner import PLANNER_REGISTRY
from causalshift.envs.xor import CausalShiftXOR

# States are iid — we don't need full 200-step episodes.
# Sample enough states per condition for statistical power.
HORIZON = 50
N_EPISODES = 10
SHIFT_LEVELS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
# Total calls: 6 shifts * 5 seeds * 10 episodes * 50 steps = 15,000


def generate_all_prompts(n_seeds: int) -> list[dict]:
    """Pre-generate all (state, metadata) pairs for every step of every episode.

    Instead of calling the API step-by-step (sequential, slow), we pre-roll
    all environment trajectories and collect every state the LLM needs to see.
    Then we batch all API calls in parallel.

    For frozen-policy evaluation: the LLM sees each state independently.
    Its action doesn't affect the next state (states are iid), so we can
    pre-generate all states and query the LLM for all of them at once.
    """
    prompts = []

    for theta in SHIFT_LEVELS:
        for seed in range(n_seeds):
            env = CausalShiftXOR(theta=theta, horizon=HORIZON)

            for ep in range(N_EPISODES):
                state, _ = env.reset(seed=seed * 10000 + ep)

                for step in range(HORIZON):
                    prompts.append({
                        "theta": theta,
                        "seed": seed,
                        "episode": ep,
                        "step": step,
                        "state": state.tolist(),
                        "s1": int(state[0]),
                        "s2": int(state[1]),
                    })
                    # Step with a dummy action (doesn't matter — states are iid)
                    state, _, terminated, truncated, _ = env.step(0)
                    if terminated or truncated:
                        break

    return prompts


def call_llm_single(planner, prompt: dict) -> dict:
    """Make a single LLM call. Retries until success — never returns garbage data."""
    state = np.array(prompt["state"], dtype=np.int64)
    max_retries = 10
    for attempt in range(max_retries):
        try:
            action = planner.get_action(state)
            prompt["action"] = action
            prompt["correct"] = int(action == prompt["s1"])
            prompt["attempts"] = attempt + 1
            return prompt
        except Exception as e:
            if attempt == max_retries - 1:
                # Log the failure but mark it clearly — do NOT silently fallback
                prompt["action"] = -1  # Invalid action, clearly flagged
                prompt["correct"] = 0
                prompt["error"] = str(e)
                prompt["attempts"] = max_retries
                return prompt
            time.sleep(2 ** attempt)  # 1, 2, 4, 8, 16, 32, 64, 128, 256 seconds
    return prompt


def run_parallel_experiment(
    model_name: str,
    condition: str,
    n_seeds: int,
    n_workers: int,
    output_dir: str,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_name}, Condition: {condition}")
    print(f"Generating prompts...")

    prompts = generate_all_prompts(n_seeds)
    total_calls = len(prompts)
    print(f"Total API calls: {total_calls}")
    print(f"Workers: {n_workers}")
    print(f"Estimated time: ~{total_calls / n_workers * 3 / 60:.0f} minutes (at ~3s/call)")

    # Create planner
    planner = PLANNER_REGISTRY[model_name](condition)

    # Execute all calls in parallel, write progress to file
    progress_file = output_path / f"{model_name}_{condition}_progress.txt"
    print(f"\nStarting parallel execution...")
    print(f"Progress: tail -f {progress_file}")
    start_time = time.time()
    results = []
    completed = 0
    errors = 0

    def log(msg):
        with open(progress_file, "a") as pf:
            pf.write(msg + "\n")

    log(f"Started: {model_name} {condition} | {total_calls} calls | {n_workers} workers")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(call_llm_single, planner, prompt): i
            for i, prompt in enumerate(prompts)
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if "error" in result:
                errors += 1

            if completed % 100 == 0 or completed == total_calls:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_calls - completed) / rate if rate > 0 else 0
                msg = (
                    f"{completed}/{total_calls} "
                    f"({completed/total_calls*100:.0f}%) | "
                    f"{rate:.1f} calls/sec | "
                    f"ETA: {eta/60:.1f}min | "
                    f"Errors: {errors}"
                )
                log(msg)
                print(f"  {msg}", flush=True)

    elapsed = time.time() - start_time
    print(f"\nCompleted {total_calls} calls in {elapsed:.0f}s ({total_calls/elapsed:.1f} calls/sec)")
    print(f"Errors: {errors}/{total_calls} ({errors/total_calls*100:.1f}%)")

    # Aggregate results by theta
    summary = aggregate_results(results, n_seeds)

    # Save
    filename = f"{model_name}_{condition}.json"
    results_file = output_path / filename
    with open(results_file, "w") as f:
        json.dump({
            "model": model_name,
            "condition": condition,
            "total_calls": total_calls,
            "total_time_sec": elapsed,
            "calls_per_sec": total_calls / elapsed,
            "error_rate": errors / total_calls,
            "summary": summary,
        }, f, indent=2)
    print(f"Results saved to {results_file}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name} ({condition})")
    print(f"{'='*60}")
    for theta_key, data in sorted(summary.items()):
        print(
            f"  theta={data['theta']:.2f}: "
            f"accuracy={data['accuracy']:.3f}, "
            f"s1_match={data['s1_match_rate']:.3f}, "
            f"s2_match={data['s2_match_rate']:.3f}, "
            f"regret_per_step={1-data['accuracy']:.3f}"
        )


def aggregate_results(results: list[dict], n_seeds: int) -> dict:
    """Aggregate per-call results into per-theta summary."""
    from collections import defaultdict

    by_theta = defaultdict(list)
    for r in results:
        by_theta[r["theta"]].append(r)

    summary = {}
    for theta, calls in sorted(by_theta.items()):
        correct = sum(c["correct"] for c in calls)
        total = len(calls)
        s1_matches = sum(1 for c in calls if c["action"] == c["s1"])
        s2_matches = sum(1 for c in calls if c["action"] == c["s2"])

        # Per-seed accuracy
        seed_accuracies = []
        for seed in range(n_seeds):
            seed_calls = [c for c in calls if c["seed"] == seed]
            if seed_calls:
                seed_acc = sum(c["correct"] for c in seed_calls) / len(seed_calls)
                seed_accuracies.append(seed_acc)

        summary[f"theta_{theta:.2f}"] = {
            "theta": theta,
            "accuracy": correct / total,
            "s1_match_rate": s1_matches / total,
            "s2_match_rate": s2_matches / total,
            "total_calls": total,
            "total_correct": correct,
            "regret_per_step": 1.0 - correct / total,
            "total_regret": total - correct,
            "mean_seed_accuracy": float(np.mean(seed_accuracies)) if seed_accuracies else 0,
            "std_seed_accuracy": float(np.std(seed_accuracies, ddof=1)) if len(seed_accuracies) > 1 else 0,
        }

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel LLM baseline experiment")
    from causalshift.baselines.llm_planner import PLANNER_REGISTRY as REG
    parser.add_argument("--model", type=str, required=True, choices=list(REG.keys()))
    parser.add_argument("--condition", type=str, default="privileged", choices=["blackbox", "privileged", "cot"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=50, help="Number of parallel API calls")
    parser.add_argument("--output", type=str, default="results/llm")
    args = parser.parse_args()
    run_parallel_experiment(args.model, args.condition, args.seeds, args.workers, args.output)
