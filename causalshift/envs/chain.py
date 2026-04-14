"""CausalShift-Chain: 5-component linear causal chain for scaling experiments.

REVISED to match the separation theorem construction:
  - State: S = (X1, ..., X5) in {0,1}^5 (discrete, 5 binary variables)
  - DAG: X1 -> X2 -> X3 -> X4 -> X5
  - X1(t+1) ~ Bernoulli(0.5) (exogenous, iid — same pattern as XOR)
  - Mechanism: Xi(t+1) = Xi-1(t+1) XOR Bi, Bi ~ Bernoulli(p_i)
  - Shift: Change p_3 (mechanism at middle of chain). Source: p_3=0.
  - Action: A in {0, 1}
  - Reward: R = 1 if A == X1 (must match the root cause)

Abstractions:
  - phi_m (mechanism-invariant): Z = (X1, X2) — upstream of shifted mechanism
  - phi_p (predictive): Z = (X4, X5) — downstream of shifted mechanism

At source (p_3=0): all Xi equal X1, so phi_m and phi_p carry identical info.
Under shift: X4, X5 become noisy copies of X1, degrading phi_p.

Why 5 components: minimal chain where (a) causal depth matters,
(b) multi-bit abstraction is tested, (c) the shifted mechanism is
interior (not at boundary), creating a clean upstream/downstream split.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CausalShiftChain(gym.Env):
    """5-component binary causal chain SCM-MDP.

    Args:
        noise_probs: Per-mechanism noise probabilities [p2, p3, p4, p5]. Length 4.
            p_i = P(B_i = 1), where Xi = X_{i-1} XOR B_i.
            Source: all zeros. Shift: change p3.
        horizon: Episode length.
    """

    metadata = {"render_modes": ["ansi"]}
    N_COMPONENTS = 5

    # Source: all mechanisms are deterministic copies
    DEFAULT_NOISE = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def __init__(
        self,
        noise_probs: np.ndarray | None = None,
        horizon: int = 200,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.noise_probs = (
            np.array(noise_probs, dtype=np.float64)
            if noise_probs is not None
            else self.DEFAULT_NOISE.copy()
        )
        assert self.noise_probs.shape == (4,)
        assert all(0.0 <= p <= 0.5 for p in self.noise_probs)

        self.horizon = horizon
        self.render_mode = render_mode

        self.observation_space = spaces.MultiDiscrete([2] * self.N_COMPONENTS)
        self.action_space = spaces.Discrete(2)

        self._state = np.zeros(self.N_COMPONENTS, dtype=np.int64)
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

        # Reward: match X1
        reward = float(action == self._state[0])

        # Transition to new state
        self._state = self._sample_state()
        self._step_count += 1

        terminated = False
        truncated = self._step_count >= self.horizon

        return self._state.copy(), reward, terminated, truncated, self._get_info()

    def _sample_state(self) -> np.ndarray:
        """Sample state from the chain SCM: X1 -> X2 -> ... -> X5."""
        state = np.zeros(self.N_COMPONENTS, dtype=np.int64)
        # X1: exogenous
        state[0] = int(self.np_random.random() < 0.5)
        # X2..X5: chain mechanism
        for i in range(1, self.N_COMPONENTS):
            noise = int(self.np_random.random() < self.noise_probs[i - 1])
            state[i] = state[i - 1] ^ noise
        return state

    def _get_info(self) -> dict[str, Any]:
        return {
            "noise_probs": self.noise_probs.tolist(),
            "step": self._step_count,
        }

    @classmethod
    def make_shifted(cls, p3: float, **kwargs) -> "CausalShiftChain":
        """Create a shifted environment by changing p3 (X2->X3 mechanism noise)."""
        noise = cls.DEFAULT_NOISE.copy()
        noise[1] = p3  # index 1 = p3 (noise for X3 = X2 XOR B3)
        return cls(noise_probs=noise, **kwargs)
