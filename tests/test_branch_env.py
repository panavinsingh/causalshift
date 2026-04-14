"""Tests for CausalShift-Branch environment."""

import numpy as np
import pytest

from causalshift.abstractions.mechanism_invariant import MechanismInvariantAbstraction
from causalshift.abstractions.predictive import PredictiveAbstraction
from causalshift.envs.branch import CausalShiftBranch


class TestBranchMechanics:
    def test_x4_equals_x1_at_source(self):
        """At source (mu=0): X2=X1, X3=X1, X4=X1 AND X1=X1."""
        env = CausalShiftBranch(mu=0.0, horizon=1000)
        env.reset(seed=42)
        for _ in range(1000):
            state, _, _, _, _ = env.step(0)
            assert state[3] == state[0], f"At source, X4 should equal X1, got {state}"

    def test_confounder_disrupts_x4(self):
        env = CausalShiftBranch(mu=0.4, horizon=10000)
        env.reset(seed=42)
        disagree = 0
        for _ in range(10000):
            state, _, _, _, _ = env.step(0)
            if state[3] != state[0]:
                disagree += 1
        frac = disagree / 10000
        assert frac > 0.1, f"X4 should disagree with X1 under confounder, got {frac:.1%}"

    def test_x1_unaffected_by_confounder(self):
        env = CausalShiftBranch(mu=0.5, horizon=10000)
        env.reset(seed=42)
        x1_vals = [env.step(0)[0][0] for _ in range(10000)]
        assert abs(np.mean(x1_vals) - 0.5) < 0.02


class TestBranchSeparation:
    def test_phi_m_perfect(self):
        phi_m = MechanismInvariantAbstraction(causal_indices=[0], discrete=True)
        env = CausalShiftBranch.make_shifted(mu=0.4, horizon=5000)
        state, _ = env.reset(seed=42)
        total = 0.0
        for _ in range(5000):
            z = phi_m.abstract(state)
            _, r, _, _, _ = env.step(z)
            total += r
            state = env._state
        assert total / 5000 > 0.99

    def test_phi_p_degrades(self):
        phi_p = PredictiveAbstraction(effect_indices=[3], discrete=True)
        env = CausalShiftBranch.make_shifted(mu=0.3, horizon=10000)
        state, _ = env.reset(seed=42)
        total = 0.0
        for _ in range(10000):
            z = phi_p.abstract(state)
            _, r, _, _, _ = env.step(z)
            total += r
            state = env._state
        mean_r = total / 10000
        assert mean_r < 0.95, f"phi_p should degrade, got {mean_r:.3f}"
