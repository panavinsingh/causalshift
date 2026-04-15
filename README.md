# The Cost of Counterfactuals

**A Separation Theorem for Transfer-Robust Abstraction in Modular Decision Processes**

[![Paper](https://img.shields.io/badge/Paper-AGI--2027-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)]()

## Abstract

We prove a constructive separation theorem showing that two state abstractions with identical source-environment trajectory distributions can exhibit radically different transfer behavior under graph-preserving shift. The *mechanism-invariant abstraction* achieves zero transfer regret, while the *predictive abstraction* suffers linear regret — and no amount of source data can distinguish them. We formalize the *counterfactual cost* as the minimal overhead for mechanism invariance, and validate our theory across 15 baselines including frontier LLMs (GPT-5.4, Gemini 3.1 Pro, DeepSeek V3.2, Qwen3-235B), world models (DreamerV3, Dreamer-CDP, TransDreamerV3, NE-Dreamer), and causal RL methods (DBC, CBM).

## Key Results

| Abstraction | Source (theta=0) | Shifted (theta=0.5) | Behavior |
|------------|-------------|-----------------|----------|
| phi_m (mechanism-invariant) | 0 regret | 0 regret | Transfer-robust |
| phi_p (predictive) | 0 regret | theta x T linear regret | Transfer-fragile |
| DBC, CBM, DreamerV3, etc. | 0 regret | ~theta x T linear regret | Learn phi_p |
| LLMs (privileged) | 100% accuracy | 100% accuracy | Act like phi_m |
| LLMs (blackbox) | ~50-100% | ~49-74% | Between phi_m and phi_p |

## Repository Structure

```
causalshift/                        # Core library (pip install -e .)
├── envs/                           # CausalShift benchmark environments
│   ├── xor.py                      # CausalShift-XOR (theorem construction)
│   ├── chain.py                    # CausalShift-Chain (5-component scaling)
│   ├── branch.py                   # CausalShift-Branch (confounder)
│   └── registry.py                 # Gymnasium registration
├── abstractions/                   # State abstraction implementations
│   ├── base.py                     # Abstract base class
│   ├── mechanism_invariant.py      # phi_m: tracks causal variables
│   └── predictive.py               # phi_p: tracks effect variables
├── baselines/                      # Baseline implementations
│   ├── ucb_abstract.py             # UCB learner on abstract MDPs
│   ├── dbc_baseline.py             # Deep Bisimulation for Control
│   ├── cbm_baseline.py             # Causal Bisimulation Modeling
│   └── llm_planner.py             # LLM planners (GPT-5.4, Gemini, DeepSeek, Qwen)
└── router/                         # Adaptive metacontrol
    └── adaptive.py                 # EXP3-based router

experiments/                        # Experiment scripts
├── full_separation.py              # Separation theorem validation (3 envs)
├── xor_separation.py               # XOR-specific separation with UCB
├── gpu_baselines.py                # DBC/CBM experiments
├── llm_baselines.py                # LLM planner experiments (parallel)
└── router_experiment.py            # Router ablation

notebooks/                          # GPU baseline scripts (Kaggle / Lightning AI)
├── dreamerv3_baseline.py           # DreamerV3 world model
├── dreamer_cdp_baseline.py         # Dreamer-CDP (JEPA-style)
├── trans_dreamer_baseline.py       # TransDreamerV3 (transformer encoder)
└── ne_dreamer_baseline.py          # NE-Dreamer (next-embedding)

analysis/                           # Plotting and statistical analysis
├── plot_separation.py              # Figs 3-4: separation results
└── plot_all_results.py             # Figs 5-8: all baselines + summary

results/                            # All experimental results (JSON)
├── full_separation/                # Theorem validation (3 envs x 30 seeds)
├── llm/                            # LLM baselines (4 models x 3 conditions)
├── gpu/                            # RL baselines (6 methods x 30 seeds)
├── router/                         # Router ablation (4 strategies x 30 seeds)
└── statistical_summary.json        # Aggregate statistics

figures/                            # Publication-ready figures (PDF + PNG)
tests/                              # Unit tests
```

## Quick Start

```bash
# Install core library
pip install -e .

# Install with LLM baseline dependencies
pip install -e ".[llm]"

# Run tests
pytest tests/ -v

# Reproduce separation theorem
python experiments/full_separation.py --seeds 30 --output results/full_separation

# Reproduce router experiment
python experiments/router_experiment.py --seeds 30 --output results/router

# Reproduce GPU baselines (DBC/CBM)
python experiments/gpu_baselines.py --baseline dbc --env xor --seeds 30 --output results/gpu
python experiments/gpu_baselines.py --baseline cbm --env xor --seeds 30 --output results/gpu

# Generate all figures
python analysis/plot_all_results.py
```

## LLM Baselines

LLM experiments require API access and the `[llm]` optional dependencies:

- **GPT-5.4** via Azure OpenAI Service
- **Gemini 3.1 Pro** via GCP Vertex AI
- **DeepSeek V3.2** via AWS Bedrock
- **Qwen3-235B** via GCP Vertex AI

```bash
pip install -e ".[llm]"
python experiments/llm_baselines.py --model gpt-5.4 --condition privileged --seeds 5 --workers 20 --output results/llm
```

## GPU Baselines (World Models)

World model baselines (DreamerV3, Dreamer-CDP, TransDreamerV3, NE-Dreamer) require GPU access. Scripts in `notebooks/` are self-contained and designed to run on Kaggle or Lightning AI.

## Citation

```bibtex
@inproceedings{singh2027cost,
  title={The Cost of Counterfactuals: A Separation Theorem for Transfer-Robust
         Abstraction in Modular Decision Processes},
  author={Singh, Panavin},
  booktitle={Proceedings of the 20th International Conference on
             Artificial General Intelligence (AGI-2027)},
  year={2027},
  publisher={Springer LNAI}
}
```

## License

[MIT](LICENSE)
