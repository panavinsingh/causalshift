"""UCB learner operating on abstract state spaces.

This is the learner class quantified over in Theorem 1. It operates
on the abstract MDP induced by an abstraction phi, selecting actions
via UCB1 on the abstract state-action Q-values.

For discrete abstract states (CausalShift-XOR): standard tabular UCB1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from causalshift.abstractions.base import Abstraction


@dataclass
class UCBStats:
    """Running statistics for a single (abstract_state, action) pair."""

    count: int = 0
    total_reward: float = 0.0

    @property
    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_reward / self.count


class UCBAbstractLearner:
    """UCB1 learner on the abstract MDP induced by abstraction phi.

    For each abstract state z and action a, maintains count N(z,a)
    and mean reward Q(z,a). Selects actions via:
        a* = argmax_a [ Q(z,a) + c * sqrt(log(t) / N(z,a)) ]

    Args:
        abstraction: The state abstraction to apply.
        n_actions: Number of available actions.
        exploration_bonus: UCB exploration constant (default: sqrt(2)).
    """

    def __init__(
        self,
        abstraction: Abstraction,
        n_actions: int,
        exploration_bonus: float = 1.414,
    ):
        self.abstraction = abstraction
        self.n_actions = n_actions
        self.c = exploration_bonus
        self.t = 0

        # Q-table: abstract_state -> action -> stats
        self._stats: dict[int, list[UCBStats]] = {}

    def _get_stats(self, z: int) -> list[UCBStats]:
        if z not in self._stats:
            self._stats[z] = [UCBStats() for _ in range(self.n_actions)]
        return self._stats[z]

    def select_action(self, state: np.ndarray) -> int:
        """Select action via UCB1 on the abstract state."""
        z = self.abstraction.abstract(state)
        stats = self._get_stats(z)
        self.t += 1

        # If any action unvisited in this abstract state, explore it
        for a in range(self.n_actions):
            if stats[a].count == 0:
                return a

        # UCB1 selection
        log_t = np.log(self.t)
        ucb_values = np.array([
            stats[a].mean + self.c * np.sqrt(log_t / stats[a].count)
            for a in range(self.n_actions)
        ])
        return int(np.argmax(ucb_values))

    def update(self, state: np.ndarray, action: int, reward: float) -> None:
        """Update Q-estimates after observing (state, action, reward)."""
        z = self.abstraction.abstract(state)
        stats = self._get_stats(z)
        stats[action].count += 1
        stats[action].total_reward += reward

    def reset(self) -> None:
        """Reset all learned statistics."""
        self._stats.clear()
        self.t = 0


@dataclass
class EpisodeResult:
    """Result of running one episode."""

    total_reward: float
    regret: float
    steps: int
    rewards: list[float] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    abstract_states: list[int] = field(default_factory=list)


def run_episode(
    env,
    learner: UCBAbstractLearner,
    optimal_value_per_step: float | None = None,
) -> EpisodeResult:
    """Run a single episode, updating the learner online.

    Args:
        env: Gymnasium environment.
        learner: UCB learner to use.
        optimal_value_per_step: V* per step for regret computation.
            For XOR at theta': optimal action is a=1, E[R]=1-theta'.
    """
    state, info = env.reset()
    total_reward = 0.0
    rewards = []
    actions = []
    abstract_states = []

    terminated = truncated = False
    while not (terminated or truncated):
        z = learner.abstraction.abstract(state)
        abstract_states.append(int(z) if isinstance(z, (int, np.integer)) else z)

        action = learner.select_action(state)
        actions.append(action)

        next_state, reward, terminated, truncated, info = env.step(action)
        learner.update(state, action, reward)

        rewards.append(reward)
        total_reward += reward
        state = next_state

    steps = len(rewards)
    regret = 0.0
    if optimal_value_per_step is not None:
        regret = optimal_value_per_step * steps - total_reward

    return EpisodeResult(
        total_reward=total_reward,
        regret=regret,
        steps=steps,
        rewards=rewards,
        actions=actions,
        abstract_states=abstract_states,
    )
