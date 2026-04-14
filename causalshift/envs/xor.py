"""CausalShift-XOR: The separation theorem's constructive environment.

REVISED CONSTRUCTION (state-dependent optimal policy):

A 2-component modular SCM-MDP where:
  - State: S = (S1, S2) in {0,1}^2
  - DAG: S1 -> S2
  - S1 dynamics: S1(t+1) ~ Bernoulli(0.5), iid (stochastic, exogenous)
  - Mechanism: S2(t+1) = S1(t+1) XOR B_theta, B_theta ~ Bernoulli(theta)
  - Action: A in {0, 1}
  - Reward: R = 1 if A == S1 (agent must MATCH the cause variable)
  - Shift: theta in [0, 0.5], source theta_0 = 0

Key property: The agent MUST observe the state to act optimally (S1 is random).

Two 1-bit abstractions:
  - phi_m (mechanism-invariant): Z = S1 (tracks cause). Policy A=Z is optimal.
    Under any theta: Z = S1, so A=Z=S1, reward = 1. Abstract MDP is FULLY INVARIANT.
  - phi_p (predictive): Z = S2 (tracks effect). At theta=0: Z=S2=S1, identical to phi_m.
    Under theta>0: Z=S2=S1 XOR noise. Policy A=Z gives reward 1-theta.
    Regret = theta * T (LINEAR in shift and time).

THIS is the separation: identical source behavior, linear regret gap under shift.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CausalShiftXOR(gym.Env):
    """Modular SCM-MDP for the separation theorem.

    The agent observes (S1, S2) and must output A matching S1 to get reward.
    S1 is iid Bernoulli(0.5) each step (exogenous, random).
    S2 = S1 XOR Bernoulli(theta) (downstream effect, noisy under shift).

    At theta=0: S2 = S1, so both components carry the same info.
    At theta>0: S2 is a noisy copy of S1. An agent relying on S2 makes errors.

    Args:
        theta: Shift parameter in [0, 0.5]. At theta=0 (source), S2=S1 deterministically.
        horizon: Episode length.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        theta: float = 0.0,
        horizon: int = 200,
        render_mode: str | None = None,
    ):
        super().__init__()
        assert 0.0 <= theta <= 0.5, f"theta must be in [0, 0.5], got {theta}"

        self.theta = theta
        self.horizon = horizon
        self.render_mode = render_mode

        self.observation_space = spaces.MultiDiscrete([2, 2])
        self.action_space = spaces.Discrete(2)

        self._step_count = 0
        self._state = np.array([0, 0], dtype=np.int64)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._state = self._sample_state()
        self._step_count = 0
        return self._state.copy(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action)

        # Reward: did the agent match S1?
        reward = float(action == self._state[0])

        # Transition: new iid S1, new S2 from mechanism
        self._state = self._sample_state()
        self._step_count += 1

        terminated = False
        truncated = self._step_count >= self.horizon

        return self._state.copy(), reward, terminated, truncated, self._get_info()

    def _sample_state(self) -> np.ndarray:
        """Sample a new state from the SCM."""
        s1 = int(self.np_random.random() < 0.5)
        noise = int(self.np_random.random() < self.theta)
        s2 = s1 ^ noise
        return np.array([s1, s2], dtype=np.int64)

    def _get_info(self) -> dict[str, Any]:
        return {
            "theta": self.theta,
            "s1": int(self._state[0]),
            "s2": int(self._state[1]),
            "step": self._step_count,
        }

    def render(self) -> str | None:
        if self.render_mode == "ansi":
            return (
                f"Step {self._step_count}/{self.horizon} | "
                f"S1={self._state[0]} S2={self._state[1]} | "
                f"theta={self.theta:.2f}"
            )
        return None


def optimal_reward_per_step(theta: float) -> float:
    """Optimal expected reward per step.

    With phi_m (perfect info): always match S1 -> reward = 1.0 per step.
    This is the global optimum regardless of theta.
    """
    return 1.0


def expected_reward_predictive(theta: float) -> float:
    """Expected reward per step when using policy A=Z with phi_p (Z=S2).

    With prob (1-theta): S2=S1, A=S2=S1, reward=1.
    With prob theta: S2≠S1, A=S2≠S1, reward=0.
    E[R] = 1 - theta.
    """
    return 1.0 - theta


class CausalShiftXORAbstracted(gym.Wrapper):
    """Wraps CausalShiftXOR to expose only an abstracted 1-bit observation.

    Args:
        env: CausalShiftXOR environment.
        abstraction: "mechanism" (phi_m: Z=S1) or "predictive" (phi_p: Z=S2).
    """

    VALID_ABSTRACTIONS = ("mechanism", "predictive")

    def __init__(self, env: CausalShiftXOR, abstraction: str = "mechanism"):
        super().__init__(env)
        assert abstraction in self.VALID_ABSTRACTIONS
        self.abstraction = abstraction
        self.observation_space = spaces.Discrete(2)

    def _abstract(self, state: np.ndarray) -> int:
        if self.abstraction == "mechanism":
            return int(state[0])  # Z = S1
        else:
            return int(state[1])  # Z = S2

    def reset(self, **kwargs) -> tuple[int, dict[str, Any]]:
        state, info = self.env.reset(**kwargs)
        info["full_state"] = state.copy()
        info["abstraction"] = self.abstraction
        return self._abstract(state), info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        state, reward, terminated, truncated, info = self.env.step(action)
        info["full_state"] = state.copy()
        info["abstraction"] = self.abstraction
        return self._abstract(state), reward, terminated, truncated, info
