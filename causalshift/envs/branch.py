"""CausalShift-Branch: Branching DAG with hidden confounder.

  - State: S = (X1, X2, X3, X4) in {0,1}^4 (observed) + U in {0,1} (hidden)
  - DAG: X1 -> {X2, X3}, {X2, X3} -> X4, U -> {X2, X3} (confounder)
  - X1(t+1) ~ Bernoulli(0.5) (exogenous, iid)
  - X2 = X1 XOR U XOR B2, X3 = X1 XOR U XOR B3
  - X4 = X2 XOR X3 (deterministic combination)
  - U ~ Bernoulli(mu) — hidden confounder
  - Shift: Change mu (confounder distribution). Source: mu=0.
  - Action: A in {0, 1}
  - Reward: R = 1 if A == X1 (must match the root cause)

Abstractions:
  - phi_m (mechanism-invariant): Z = X1 — the root cause, unaffected by confounder
  - phi_p (predictive): Z = X4 — the downstream aggregate, confounded

At source (mu=0, B2=B3=0): X2=X3=X1, X4=X2 XOR X3=0. phi_p always sees 0.
  Actually this is degenerate. Better: at source, mu=0 and X4 = X2 XOR X3.
  If B2=B3=0 and U=0: X2=X1, X3=X1, X4=0 always. Not useful.

REVISED: Use X4 = X2 AND X3 instead of XOR. Then at source:
  X2=X1, X3=X1, X4 = X1 AND X1 = X1. So phi_p(Z=X4) = X1 = phi_m(Z=X1).
  Under shift (mu>0): U flips X2 and X3 sometimes, making X4 unreliable.

What makes Branch harder than Chain: the confounder U creates correlated noise
in X2 and X3 that is invisible in the observed state. A predictive abstraction
cannot distinguish causal influence of X1 from confounding influence of U.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CausalShiftBranch(gym.Env):
    """Branching DAG SCM-MDP with hidden confounder.

    Args:
        mu: Confounder probability P(U=1). Source: mu=0.
        noise_x2: P(B2=1), additional noise on X2. Default 0.
        noise_x3: P(B3=1), additional noise on X3. Default 0.
        horizon: Episode length.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        mu: float = 0.0,
        noise_x2: float = 0.0,
        noise_x3: float = 0.0,
        horizon: int = 200,
        render_mode: str | None = None,
    ):
        super().__init__()
        assert 0.0 <= mu <= 0.5
        assert 0.0 <= noise_x2 <= 0.5
        assert 0.0 <= noise_x3 <= 0.5

        self.mu = mu
        self.noise_x2 = noise_x2
        self.noise_x3 = noise_x3
        self.horizon = horizon
        self.render_mode = render_mode

        # Observed state: (X1, X2, X3, X4). U is hidden.
        self.observation_space = spaces.MultiDiscrete([2, 2, 2, 2])
        self.action_space = spaces.Discrete(2)

        self._state = np.zeros(4, dtype=np.int64)
        self._step_count = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._state = self._sample_state()
        self._step_count = 0
        return self._state.copy(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action)

        reward = float(action == self._state[0])
        self._state = self._sample_state()
        self._step_count += 1

        terminated = False
        truncated = self._step_count >= self.horizon

        return self._state.copy(), reward, terminated, truncated, self._get_info()

    def _sample_state(self) -> np.ndarray:
        """Sample from the branching SCM with hidden confounder U."""
        x1 = int(self.np_random.random() < 0.5)
        u = int(self.np_random.random() < self.mu)
        b2 = int(self.np_random.random() < self.noise_x2)
        b3 = int(self.np_random.random() < self.noise_x3)

        x2 = x1 ^ u ^ b2
        x3 = x1 ^ u ^ b3
        x4 = x2 & x3  # AND gate: X4=1 iff both X2=1 and X3=1

        return np.array([x1, x2, x3, x4], dtype=np.int64)

    def _get_info(self) -> dict[str, Any]:
        return {
            "mu": self.mu,
            "noise_x2": self.noise_x2,
            "noise_x3": self.noise_x3,
            "step": self._step_count,
        }

    @classmethod
    def make_shifted(cls, mu: float, **kwargs) -> "CausalShiftBranch":
        """Create a shifted environment by changing confounder probability mu."""
        return cls(mu=mu, **kwargs)
