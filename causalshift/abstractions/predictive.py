"""Predictive abstraction: Z = g(S_effect).

Tracks downstream/effect variables that are predictively sufficient on the
source distribution but NOT mechanism-invariant. This is the abstraction
our theorem proves suffers linear transfer regret.
"""

from __future__ import annotations

import numpy as np

from causalshift.abstractions.base import Abstraction


class PredictiveAbstraction(Abstraction):
    """Predictive (non-mechanism-invariant) abstraction.

    For CausalShift-XOR: Z = S2 (the effect variable).
    For CausalShift-Chain: Z = (X4, X5) (downstream of shifted mechanism).

    Args:
        effect_indices: Indices of the state components to retain.
            These are variables whose relationship to reward is mediated
            by shiftable mechanisms.
        discrete: Whether the abstraction produces discrete states.
    """

    def __init__(self, effect_indices: list[int], discrete: bool = True):
        self._indices = effect_indices
        self._discrete = discrete

    def abstract(self, state: np.ndarray) -> int | np.ndarray:
        extracted = state[self._indices]
        if self._discrete and len(self._indices) == 1:
            return int(extracted[0])
        return extracted

    @property
    def name(self) -> str:
        return f"phi_pred({self._indices})"

    @property
    def abstract_state_size(self) -> int:
        return len(self._indices)

    @classmethod
    def for_xor(cls) -> PredictiveAbstraction:
        """Factory for CausalShift-XOR: Z = S2."""
        return cls(effect_indices=[1], discrete=True)

    @classmethod
    def for_chain(cls) -> PredictiveAbstraction:
        """Factory for CausalShift-Chain: Z = (X4, X5)."""
        return cls(effect_indices=[3, 4], discrete=False)
