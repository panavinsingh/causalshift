"""Deep Bisimulation for Control (DBC) baseline.

DBC learns a latent representation where distances in latent space equal
bisimulation distances in state space. It is a PREDICTIVE abstraction —
it optimizes for reward prediction and transition prediction on the
source distribution, but does NOT preserve causal/mechanism structure.

Our theorem predicts: DBC's learned abstraction will behave like phi_p
under graph-preserving shift (linear regret).

Reference: Zhang et al. (2021) "Learning Invariant Representations for
Reinforcement Learning without Reconstruction" ICLR 2021.

For our discrete binary environments, we implement a simplified version:
a neural network encoder that maps states to latent representations,
trained with bisimulation loss on source data, then frozen for transfer.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass


@dataclass
class DBCConfig:
    """Configuration for DBC baseline."""
    latent_dim: int = 8
    hidden_dim: int = 64
    lr: float = 1e-3
    bisim_coef: float = 0.5
    reward_coef: float = 0.5
    batch_size: int = 64
    train_steps: int = 5000
    gamma: float = 0.99


class DBCEncoder(nn.Module):
    """Encoder network: maps state to latent representation."""

    def __init__(self, state_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class DBCTransitionModel(nn.Module):
    """Predicts next latent state given current latent + action."""

    def __init__(self, latent_dim: int, n_actions: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + n_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.n_actions = n_actions

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        a_onehot = torch.nn.functional.one_hot(a.long(), self.n_actions).float()
        return self.net(torch.cat([z, a_onehot], dim=-1))


class DBCRewardModel(nn.Module):
    """Predicts reward given latent state + action."""

    def __init__(self, latent_dim: int, n_actions: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + n_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.n_actions = n_actions

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        a_onehot = torch.nn.functional.one_hot(a.long(), self.n_actions).float()
        return self.net(torch.cat([z, a_onehot], dim=-1)).squeeze(-1)


class DBCBaseline:
    """Deep Bisimulation for Control — learns predictive state abstraction.

    Training phase: Collect transitions on source environment, train encoder
    with bisimulation metric loss (reward prediction + transition prediction).

    Transfer phase: Freeze encoder, use Q-learning on the learned latent space.
    Measure regret under shifted environments.
    """

    def __init__(self, state_dim: int, n_actions: int, config: DBCConfig | None = None):
        self.config = config or DBCConfig()
        self.state_dim = state_dim
        self.n_actions = n_actions

        self.encoder = DBCEncoder(state_dim, self.config.latent_dim, self.config.hidden_dim)
        self.transition = DBCTransitionModel(self.config.latent_dim, n_actions, self.config.hidden_dim)
        self.reward_model = DBCRewardModel(self.config.latent_dim, n_actions, self.config.hidden_dim)

        self.optimizer = optim.Adam(
            list(self.encoder.parameters())
            + list(self.transition.parameters())
            + list(self.reward_model.parameters()),
            lr=self.config.lr,
        )

        # Q-table on discretized latent space for action selection
        self._q_table: dict[tuple, np.ndarray] = {}

    def collect_source_data(self, env, n_episodes: int = 100) -> list[dict]:
        """Collect transitions from source environment."""
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

    def train_encoder(self, transitions: list[dict]) -> dict:
        """Train encoder with bisimulation loss on source data."""
        states = torch.tensor(np.array([t["state"] for t in transitions]), dtype=torch.float32)
        actions = torch.tensor([t["action"] for t in transitions], dtype=torch.long)
        rewards = torch.tensor([t["reward"] for t in transitions], dtype=torch.float32)
        next_states = torch.tensor(np.array([t["next_state"] for t in transitions]), dtype=torch.float32)

        n = len(transitions)
        losses = []

        for step in range(self.config.train_steps):
            idx = torch.randint(0, n, (self.config.batch_size,))
            s, a, r, ns = states[idx], actions[idx], rewards[idx], next_states[idx]

            # Encode
            z = self.encoder(s)
            z_next = self.encoder(ns)

            # Reward prediction loss
            r_pred = self.reward_model(z, a)
            reward_loss = nn.functional.mse_loss(r_pred, r)

            # Transition prediction loss
            z_next_pred = self.transition(z, a)
            transition_loss = nn.functional.mse_loss(z_next_pred, z_next.detach())

            # Bisimulation loss: pairwise distances in latent space should match
            # reward differences + discounted next-state distances
            idx2 = torch.randint(0, n, (self.config.batch_size,))
            z2 = self.encoder(states[idx2])
            r2 = rewards[idx2]
            z2_next = self.encoder(next_states[idx2])

            latent_dist = torch.norm(z - z2, dim=-1)
            reward_dist = torch.abs(r - r2)
            next_dist = torch.norm(z_next.detach() - z2_next.detach(), dim=-1)
            bisim_target = reward_dist + self.config.gamma * next_dist
            bisim_loss = nn.functional.mse_loss(latent_dist, bisim_target)

            loss = (
                self.config.reward_coef * reward_loss
                + self.config.bisim_coef * bisim_loss
                + transition_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % 500 == 0:
                losses.append(loss.item())

        return {"final_loss": losses[-1] if losses else 0.0, "losses": losses}

    def _discretize_latent(self, z: np.ndarray, resolution: int = 10) -> tuple:
        """Discretize continuous latent for Q-table lookup."""
        return tuple(np.round(z * resolution).astype(int))

    def learn_policy(self, env, n_episodes: int = 200, epsilon: float = 0.1) -> None:
        """Learn Q-values on the encoded source environment."""
        self.encoder.eval()
        lr = 0.1

        for ep in range(n_episodes):
            state, _ = env.reset(seed=ep + 1000)
            for _ in range(env.horizon):
                with torch.no_grad():
                    z = self.encoder(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                    z_key = self._discretize_latent(z.squeeze().numpy())

                if z_key not in self._q_table:
                    self._q_table[z_key] = np.zeros(self.n_actions)

                # Epsilon-greedy
                if np.random.random() < epsilon:
                    action = np.random.randint(self.n_actions)
                else:
                    action = int(np.argmax(self._q_table[z_key]))

                next_state, reward, terminated, truncated, _ = env.step(action)

                with torch.no_grad():
                    z_next = self.encoder(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0))
                    z_next_key = self._discretize_latent(z_next.squeeze().numpy())

                if z_next_key not in self._q_table:
                    self._q_table[z_next_key] = np.zeros(self.n_actions)

                # Q-learning update
                target = reward + self.config.gamma * np.max(self._q_table[z_next_key])
                self._q_table[z_key][action] += lr * (target - self._q_table[z_key][action])

                state = next_state
                if terminated or truncated:
                    break

    def select_action(self, state: np.ndarray) -> int:
        """Select action using frozen encoder + learned Q-values."""
        with torch.no_grad():
            z = self.encoder(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            z_key = self._discretize_latent(z.squeeze().numpy())

        if z_key in self._q_table:
            return int(np.argmax(self._q_table[z_key]))
        return np.random.randint(self.n_actions)
