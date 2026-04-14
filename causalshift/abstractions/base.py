"""Base class for state abstractions in modular SCM-MDPs."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Abstraction(ABC):
    """A state abstraction phi: S -> Z for a modular SCM-MDP.

    The abstraction maps full state observations to an abstract state space.
    The key property we study: mechanism-invariant abstractions preserve
    the abstract transition kernel across graph-preserving shifts, while
    predictive abstractions may not.
    """

    @abstractmethod
    def abstract(self, state: np.ndarray) -> int | np.ndarray:
        """Map a full state to an abstract state."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""

    @property
    @abstractmethod
    def abstract_state_size(self) -> int:
        """Cardinality (discrete) or dimensionality (continuous) of Z."""
