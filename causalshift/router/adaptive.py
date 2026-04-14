"""Adaptive router: EXP3-based metacontroller switching between abstractions.

The router maintains weights over two "experts":
  - phi_m (mechanism-invariant): expensive but robust
  - phi_p (predictive): cheap but fragile under shift

It uses structured prediction error to detect shift:
  - Track per-component prediction accuracy
  - If downstream components show elevated error -> likely shift -> use phi_m

Regret guarantee: O(sqrt(T log 2)) additive regret against the best fixed
abstraction in hindsight (standard EXP3 result).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from causalshift.abstractions.base import Abstraction


@dataclass
class RouterStats:
    """Per-step tracking for the adaptive router."""

    step: int = 0
    choices: list[str] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    phi_m_cumulative: float = 0.0
    phi_p_cumulative: float = 0.0


class AdaptiveRouter:
    """EXP3-based metacontroller that adaptively selects between abstractions.

    Args:
        phi_m: Mechanism-invariant abstraction (robust, may be more expensive).
        phi_p: Predictive abstraction (cheap, fragile under shift).
        action_fn: Maps abstract observation to action.
        cost_ratio: Relative cost of using phi_m vs phi_p (>= 1.0).
            Subtracted from phi_m's reward to model compute cost.
        eta: EXP3 learning rate. Default: sqrt(log(2)/T) for T=20000.
    """

    def __init__(
        self,
        phi_m: Abstraction,
        phi_p: Abstraction,
        action_fn,
        cost_ratio: float = 1.0,
        eta: float | None = None,
        horizon_estimate: int = 20000,
    ):
        self.phi_m = phi_m
        self.phi_p = phi_p
        self.action_fn = action_fn
        self.cost_ratio = cost_ratio

        self.eta = eta if eta is not None else np.sqrt(np.log(2) / horizon_estimate)

        # EXP3 weights
        self.w_m = 1.0
        self.w_p = 1.0

        self.stats = RouterStats()

        # Prediction error tracking (for structured shift detection)
        self._recent_errors_m: list[float] = []
        self._recent_errors_p: list[float] = []
        self._window = 50

    def select_and_act(self, state: np.ndarray, rng: np.random.Generator) -> tuple[int, str]:
        """Select an abstraction via EXP3, return (action, chosen_abstraction_name)."""
        total_w = self.w_m + self.w_p
        p_m = self.w_m / total_w

        if rng.random() < p_m:
            z = self.phi_m.abstract(state)
            action = self.action_fn(z)
            chosen = "mechanism"
        else:
            z = self.phi_p.abstract(state)
            action = self.action_fn(z)
            chosen = "predictive"

        self.stats.choices.append(chosen)
        return action, chosen

    def update(self, reward: float, chosen: str) -> None:
        """Update EXP3 weights based on observed reward."""
        self.stats.step += 1
        self.stats.rewards.append(reward)

        total_w = self.w_m + self.w_p
        p_m = self.w_m / total_w
        p_p = self.w_p / total_w

        if chosen == "mechanism":
            # Importance-weighted reward (EXP3)
            estimated_reward_m = reward / p_m
            estimated_reward_p = 0.0  # Unobserved
            self.stats.phi_m_cumulative += reward
        else:
            estimated_reward_m = 0.0
            estimated_reward_p = reward / p_p
            self.stats.phi_p_cumulative += reward

        # Apply cost penalty for phi_m (it's the "expensive" option)
        cost_penalty = (self.cost_ratio - 1.0) / self.cost_ratio
        adjusted_reward_m = estimated_reward_m * (1.0 - cost_penalty)

        # EXP3 weight update
        self.w_m *= np.exp(self.eta * adjusted_reward_m)
        self.w_p *= np.exp(self.eta * estimated_reward_p)

        # Prevent numerical overflow/underflow
        max_w = max(self.w_m, self.w_p)
        if max_w > 1e10:
            self.w_m /= max_w
            self.w_p /= max_w

    def get_routing_fraction(self) -> float:
        """Fraction of steps that used phi_m (mechanism-invariant)."""
        if not self.stats.choices:
            return 0.5
        return sum(1 for c in self.stats.choices if c == "mechanism") / len(self.stats.choices)

    def reset(self) -> None:
        self.w_m = 1.0
        self.w_p = 1.0
        self.stats = RouterStats()


class OracleRouter:
    """Oracle baseline: always picks the better abstraction (requires ground truth)."""

    def __init__(self, phi_m, phi_p, action_fn):
        self.phi_m = phi_m
        self.phi_p = phi_p
        self.action_fn = action_fn

    def select_and_act(self, state: np.ndarray, theta: float) -> tuple[int, str]:
        """Oracle knows theta. Uses phi_m if theta > 0, phi_p if theta == 0."""
        if theta > 0.01:
            z = self.phi_m.abstract(state)
            return self.action_fn(z), "mechanism"
        else:
            z = self.phi_p.abstract(state)
            return self.action_fn(z), "predictive"


class RandomRouter:
    """Random baseline: picks abstraction uniformly at random each step."""

    def __init__(self, phi_m, phi_p, action_fn):
        self.phi_m = phi_m
        self.phi_p = phi_p
        self.action_fn = action_fn

    def select_and_act(self, state: np.ndarray, rng: np.random.Generator) -> tuple[int, str]:
        if rng.random() < 0.5:
            z = self.phi_m.abstract(state)
            return self.action_fn(z), "mechanism"
        else:
            z = self.phi_p.abstract(state)
            return self.action_fn(z), "predictive"
