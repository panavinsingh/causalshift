"""Causal Bisimulation Modeling (CBM) baseline.

CBM learns causal relationships in the dynamics and reward functions to derive
minimal, task-specific causal state abstractions. Unlike DBC which uses
predictive bisimulation, CBM explicitly models causal structure.

Our theorem predicts: If CBM correctly identifies causal structure, it should
learn a mechanism-invariant abstraction (phi_m behavior). If it relies on
observational correlations, it will learn a predictive abstraction (phi_p behavior).

Reference: Wang et al. (2024) "Building Minimal and Reusable Causal State
Abstractions for Reinforcement Learning" arXiv:2401.12497.

Simplified implementation: learns a causal graph over state components via
intervention-based structure learning, then constructs abstraction from
causally relevant variables.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass


@dataclass
class CBMConfig:
    """Configuration for CBM baseline."""
    hidden_dim: int = 64
    lr: float = 1e-3
    n_interventions: int = 500
    train_steps: int = 3000
    batch_size: int = 64
    causal_threshold: float = 0.1


class CausalGraphLearner(nn.Module):
    """Learns causal relationships between state components and reward.

    For each state component S_i, trains a predictor for reward and next-state
    with and without S_i to determine causal relevance via intervention.
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions

        # Full model: predicts reward from all state components
        self.reward_full = nn.Sequential(
            nn.Linear(state_dim + n_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Per-component ablation models: predict reward without component i
        self.reward_ablated = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim - 1 + n_actions, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(state_dim)
        ])

    def forward_full(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        a_oh = nn.functional.one_hot(a.long(), self.n_actions).float()
        return self.reward_full(torch.cat([s.float(), a_oh], dim=-1)).squeeze(-1)

    def forward_ablated(self, s: torch.Tensor, a: torch.Tensor, ablate_idx: int) -> torch.Tensor:
        a_oh = nn.functional.one_hot(a.long(), self.n_actions).float()
        # Remove component ablate_idx from state
        mask = [i for i in range(self.state_dim) if i != ablate_idx]
        s_ablated = s[:, mask].float()
        return self.reward_ablated[ablate_idx](torch.cat([s_ablated, a_oh], dim=-1)).squeeze(-1)


class CBMBaseline:
    """Causal Bisimulation Modeling — learns causal state abstraction.

    Phase 1: Collect transitions + interventional data from source.
    Phase 2: Learn causal graph (which components causally affect reward).
    Phase 3: Construct abstraction using only causally relevant components.
    Phase 4: Learn policy on the causal abstraction.
    Phase 5: Transfer (frozen) to shifted environments.
    """

    def __init__(self, state_dim: int, n_actions: int, config: CBMConfig | None = None):
        self.config = config or CBMConfig()
        self.state_dim = state_dim
        self.n_actions = n_actions

        self.graph_learner = CausalGraphLearner(state_dim, n_actions, self.config.hidden_dim)
        self.optimizer = optim.Adam(self.graph_learner.parameters(), lr=self.config.lr)

        self.causal_indices: list[int] = []
        self._q_table: dict[tuple, np.ndarray] = {}

    def collect_source_data(self, env, n_episodes: int = 100) -> list[dict]:
        """Collect observational transitions from source."""
        transitions = []
        for ep in range(n_episodes):
            state, _ = env.reset(seed=ep)
            for _ in range(env.horizon):
                action = env.action_space.sample()
                next_state, reward, terminated, truncated, _ = env.step(action)
                transitions.append({
                    "state": state.copy(),
                    "action": action,
                    "reward": reward,
                    "next_state": next_state.copy(),
                })
                state = next_state
                if terminated or truncated:
                    break
        return transitions

    def learn_causal_graph(self, transitions: list[dict]) -> dict[int, float]:
        """Learn which state components causally affect reward.

        Trains full and ablated reward predictors. Components where ablation
        significantly increases prediction error are deemed causal.
        """
        states = torch.tensor(np.array([t["state"] for t in transitions]), dtype=torch.float32)
        actions = torch.tensor([t["action"] for t in transitions], dtype=torch.long)
        rewards = torch.tensor([t["reward"] for t in transitions], dtype=torch.float32)
        n = len(transitions)

        # Train all models
        for step in range(self.config.train_steps):
            idx = torch.randint(0, n, (self.config.batch_size,))
            s, a, r = states[idx], actions[idx], rewards[idx]

            # Full model loss
            r_pred_full = self.graph_learner.forward_full(s, a)
            loss_full = nn.functional.mse_loss(r_pred_full, r)

            # Ablated model losses
            loss_ablated = torch.tensor(0.0)
            for i in range(self.state_dim):
                r_pred_abl = self.graph_learner.forward_ablated(s, a, i)
                loss_ablated += nn.functional.mse_loss(r_pred_abl, r)

            loss = loss_full + loss_ablated
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Compute causal importance: difference in prediction error
        self.graph_learner.eval()
        with torch.no_grad():
            r_pred_full = self.graph_learner.forward_full(states, actions)
            full_error = nn.functional.mse_loss(r_pred_full, rewards).item()

            causal_scores = {}
            for i in range(self.state_dim):
                r_pred_abl = self.graph_learner.forward_ablated(states, actions, i)
                abl_error = nn.functional.mse_loss(r_pred_abl, rewards).item()
                causal_scores[i] = abl_error - full_error

        # Components with score above threshold are causally relevant
        self.causal_indices = [
            i for i, score in causal_scores.items()
            if score > self.config.causal_threshold
        ]

        # Fallback: if nothing passes threshold, use all components
        if not self.causal_indices:
            self.causal_indices = list(range(self.state_dim))

        return causal_scores

    def abstract(self, state: np.ndarray) -> tuple:
        """Apply learned causal abstraction."""
        return tuple(state[i] for i in self.causal_indices)

    def learn_policy(self, env, n_episodes: int = 200, epsilon: float = 0.1) -> None:
        """Learn Q-values on the causal abstraction."""
        lr = 0.1
        gamma = 0.99

        for ep in range(n_episodes):
            state, _ = env.reset(seed=ep + 1000)
            for _ in range(env.horizon):
                z = self.abstract(state)
                if z not in self._q_table:
                    self._q_table[z] = np.zeros(self.n_actions)

                if np.random.random() < epsilon:
                    action = np.random.randint(self.n_actions)
                else:
                    action = int(np.argmax(self._q_table[z]))

                next_state, reward, terminated, truncated, _ = env.step(action)
                z_next = self.abstract(next_state)

                if z_next not in self._q_table:
                    self._q_table[z_next] = np.zeros(self.n_actions)

                target = reward + gamma * np.max(self._q_table[z_next])
                self._q_table[z][action] += lr * (target - self._q_table[z][action])

                state = next_state
                if terminated or truncated:
                    break

    def select_action(self, state: np.ndarray) -> int:
        """Select action using learned causal abstraction + Q-values."""
        z = self.abstract(state)
        if z in self._q_table:
            return int(np.argmax(self._q_table[z]))
        return np.random.randint(self.n_actions)
