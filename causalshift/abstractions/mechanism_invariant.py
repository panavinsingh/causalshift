"""Mechanism-invariant abstraction: Z = h(S_cause).

Tracks the upstream causal variable(s) whose dynamics are invariant
across graph-preserving shifts. This is the abstraction our theorem
proves achieves bounded transfer regret.
"""

from __future__ import annotations

import numpy as np

from causalshift.abstractions.base import Abstraction


class MechanismInvariantAbstraction(Abstraction):
    """Mechanism-invariant abstraction for modular SCM-MDPs.

    For CausalShift-XOR: Z = S1 (the cause variable).
    For CausalShift-Chain: Z = (X1, X2) (upstream of shifted mechanism).

    Args:
        causal_indices: Indices of the state components to retain.
            These should be the variables whose causal mechanisms are
            invariant across the shift family.
        discrete: Whether the abstraction produces discrete states.
    """

    def __init__(self, causal_indices: list[int], discrete: bool = True):
        self._indices = causal_indices
        self._discrete = discrete

    def abstract(self, state: np.ndarray) -> int | np.ndarray:
        extracted = state[self._indices]
        if self._discrete and len(self._indices) == 1:
            return int(extracted[0])
        return extracted

    @property
    def name(self) -> str:
        return f"phi_mech({self._indices})"

    @property
    def abstract_state_size(self) -> int:
        return len(self._indices)

    @classmethod
    def for_xor(cls) -> MechanismInvariantAbstraction:
        """Factory for CausalShift-XOR: Z = S1."""
        return cls(causal_indices=[0], discrete=True)

    @classmethod
    def for_chain(cls, upstream_depth: int = 2) -> MechanismInvariantAbstraction:
        """Factory for CausalShift-Chain: Z = (X1, ..., X_{depth})."""
        return cls(causal_indices=list(range(upstream_depth)), discrete=False)
