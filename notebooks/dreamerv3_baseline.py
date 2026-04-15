"""DreamerV3 baseline on CausalShift — Kaggle GPU notebook.

DreamerV3: standard reconstruction-based world model with encoder,
decoder, latent dynamics predictor, and reward head.

Reference: Hafner et al. (2023) "Mastering Diverse Domains through
World Models" arXiv:2301.04104

For our discrete environments, the model:
- Encodes state to latent via MLP encoder
- Reconstructs state via MLP decoder (reconstruction loss)
- Predicts next latent via transition predictor
- Predicts reward via reward head

Trained on source data, then frozen policy is evaluated on transfer.
"""

import subprocess
import sys
import os
import json
import time
import numpy as np

print("Installing dependencies...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "gymnasium", "-q"])

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces


# === CausalShift Environment ===
class CausalShiftXOR(gym.Env):
    def __init__(self, theta=0.0, horizon=200):
        super().__init__()
        self.theta = theta
        self.horizon = horizon
        self.observation_space = spaces.MultiDiscrete([2, 2])
        self.action_space = spaces.Discrete(2)
        self._step_count = 0
        self._state = np.array([0, 0], dtype=np.int64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._state = self._sample()
        self._step_count = 0
        return self._state.copy(), {}

    def step(self, action):
        reward = float(int(action) == self._state[0])
        self._state = self._sample()
        self._step_count += 1
        return self._state.copy(), reward, False, self._step_count >= self.horizon, {}

    def _sample(self):
        s1 = int(self.np_random.random() < 0.5)
        s2 = s1 ^ int(self.np_random.random() < self.theta)
        return np.array([s1, s2], dtype=np.int64)


# === DreamerV3 Model ===
class DreamerV3Model:
    """DreamerV3-style world model: encoder + decoder + latent dynamics."""

    def __init__(self, state_dim=2, n_actions=2, latent_dim=16, hidden_dim=64, lr=1e-3):
        self.n_actions = n_actions
        self.latent_dim = latent_dim
        self.device = torch.device("cuda")
        print(f"  Using device: {self.device}", flush=True)

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        ).to(self.device)

        # DreamerV3: reconstruction decoder (reconstructs state from latent)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        ).to(self.device)

        # Transition predictor
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + n_actions, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        ).to(self.device)

        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim + n_actions, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

        params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.predictor.parameters()) + list(self.reward_head.parameters())
        self.optimizer = optim.Adam(params, lr=lr)
        self._q_table = {}

    def collect_data(self, env, n_episodes=100):
        data = []
        for ep in range(n_episodes):
            s, _ = env.reset(seed=ep)
            for _ in range(env.horizon):
                a = env.action_space.sample()
                ns, r, term, trunc, _ = env.step(a)
                data.append((s.copy(), a, r, ns.copy()))
                s = ns
                if term or trunc: break
        return data

    def train_world_model(self, data, steps=5000, batch_size=64):
        states = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32).to(self.device)
        actions = torch.tensor([d[1] for d in data], dtype=torch.long).to(self.device)
        rewards = torch.tensor([d[2] for d in data], dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array([d[3] for d in data]), dtype=torch.float32).to(self.device)
        n = len(data)

        for step in range(steps):
            idx = torch.randint(0, n, (batch_size,))
            s, a, r, ns = states[idx], actions[idx], rewards[idx], next_states[idx]
            a_oh = torch.nn.functional.one_hot(a, self.n_actions).float()

            z = self.encoder(s)
            z_next_true = self.encoder(ns).detach()  # Stop gradient on target (JEPA)
            z_next_pred = self.predictor(torch.cat([z, a_oh], dim=-1))
            r_pred = self.reward_head(torch.cat([z, a_oh], dim=-1)).squeeze(-1)

            # DreamerV3 loss: reconstruction + transition prediction + reward
            s_recon = self.decoder(z)
            recon_loss = nn.functional.mse_loss(s_recon, s.float())
            pred_loss = nn.functional.mse_loss(z_next_pred, z_next_true)
            reward_loss = nn.functional.mse_loss(r_pred, r)

            loss = recon_loss + pred_loss + reward_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def learn_policy(self, env, n_episodes=200, epsilon=0.1):
        self.encoder.eval()
        for ep in range(n_episodes):
            s, _ = env.reset(seed=ep + 1000)
            for _ in range(env.horizon):
                with torch.no_grad():
                    z = self.encoder(torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device))
                    z_key = tuple(np.round(z.cpu().squeeze().numpy() * 10).astype(int))
                if z_key not in self._q_table:
                    self._q_table[z_key] = np.zeros(self.n_actions)
                if np.random.random() < epsilon:
                    a = np.random.randint(self.n_actions)
                else:
                    a = int(np.argmax(self._q_table[z_key]))
                ns, r, term, trunc, _ = env.step(a)
                with torch.no_grad():
                    zn = self.encoder(torch.tensor(ns, dtype=torch.float32).unsqueeze(0).to(self.device))
                    zn_key = tuple(np.round(zn.cpu().squeeze().numpy() * 10).astype(int))
                if zn_key not in self._q_table:
                    self._q_table[zn_key] = np.zeros(self.n_actions)
                self._q_table[z_key][a] += 0.1 * (r + 0.99 * np.max(self._q_table[zn_key]) - self._q_table[z_key][a])
                s = ns
                if term or trunc: break

    def select_action(self, state):
        with torch.no_grad():
            z = self.encoder(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device))
            z_key = tuple(np.round(z.cpu().squeeze().numpy() * 10).astype(int))
        if z_key in self._q_table:
            return int(np.argmax(self._q_table[z_key]))
        return np.random.randint(self.n_actions)


# === Run experiment ===
SHIFT_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
N_SEEDS = 30
HORIZON = 200
N_EPISODES = 100

all_results = []
for seed in range(N_SEEDS):
    print(f"\nSeed {seed+1}/{N_SEEDS}", flush=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = DreamerV3Model()
    source = CausalShiftXOR(theta=0.0, horizon=HORIZON)
    data = model.collect_data(source, n_episodes=100)
    loss = model.train_world_model(data, steps=5000)
    print(f"  Train loss: {loss:.4f}", flush=True)
    model.learn_policy(source, n_episodes=200)

    results = {}
    for theta in SHIFT_LEVELS:
        env = CausalShiftXOR(theta=theta, horizon=HORIZON)
        total_r = 0.0
        for ep in range(N_EPISODES):
            s, _ = env.reset(seed=ep * 1000)
            for _ in range(HORIZON):
                a = model.select_action(s)
                s, r, term, trunc, _ = env.step(a)
                total_r += r
                if term or trunc: break
        regret = N_EPISODES * HORIZON - total_r
        results[f"shift_{theta:.2f}"] = {"shift_value": theta, "total_regret": float(regret)}
        print(f"  theta={theta:.2f}: regret={regret:.0f}", flush=True)

    all_results.append({"seed": seed, "results": results})

# Aggregate and save
summary = {}
for theta in SHIFT_LEVELS:
    key = f"shift_{theta:.2f}"
    regrets = [r["results"][key]["total_regret"] for r in all_results]
    summary[key] = {
        "shift_value": theta,
        "mean_total_regret": float(np.mean(regrets)),
        "std_total_regret": float(np.std(regrets, ddof=1)),
        "ci95_total_regret": float(1.96 * np.std(regrets, ddof=1) / np.sqrt(N_SEEDS)),
    }

with open("dreamerv3_xor_results.json", "w") as f:
    json.dump({"baseline": "dreamerv3", "env": "xor", "summary": summary, "per_seed": all_results}, f, indent=2)

print("\n" + "=" * 60)
print("RESULTS: DreamerV3 on XOR")
print("=" * 60)
for theta in SHIFT_LEVELS:
    s = summary[f"shift_{theta:.2f}"]
    print(f"  theta={theta:.2f}: regret={s['mean_total_regret']:8.1f} +/- {s['ci95_total_regret']:6.1f}")
