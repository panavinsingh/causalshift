"""Tests for CausalShift-Chain environment."""

import numpy as np
import pytest

from causalshift.abstractions.mechanism_invariant import MechanismInvariantAbstraction
from causalshift.abstractions.predictive import PredictiveAbstraction
from causalshift.envs.chain import CausalShiftChain


class TestChainMechanics:
    def test_all_equal_at_source(self):
        env = CausalShiftChain(horizon=1000)
        env.reset(seed=42)
        for _ in range(1000):
            state, _, _, _, _ = env.step(0)
            assert all(state[i] == state[0] for i in range(5)), (
                f"At source, all components should equal X1, got {state}"
            )

    def test_x1_is_uniform(self):
        env = CausalShiftChain(horizon=10000)
        env.reset(seed=42)
        x1_vals = [env.step(0)[0][0] for _ in range(10000)]
        assert abs(np.mean(x1_vals) - 0.5) < 0.02

    def test_downstream_noisy_under_shift(self):
        env = CausalShiftChain.make_shifted(p3=0.3, horizon=5000)
        env.reset(seed=42)
        disagree_x4 = 0
        for _ in range(5000):
            state, _, _, _, _ = env.step(0)
            if state[3] != state[0]:  # X4 vs X1
                disagree_x4 += 1
        frac = disagree_x4 / 5000
        assert frac > 0.1, f"X4 should disagree with X1 under shift, got {frac:.1%}"

    def test_upstream_clean_under_shift(self):
        env = CausalShiftChain.make_shifted(p3=0.4, horizon=5000)
        env.reset(seed=42)
        for _ in range(5000):
            state, _, _, _, _ = env.step(0)
            # X2 should still equal X1 (p2=0, upstream of shift)
            assert state[1] == state[0], "X2 should equal X1 (upstream of shift at X3)"


class TestChainSeparation:
    def test_phi_m_perfect(self):
        phi_m = MechanismInvariantAbstraction(causal_indices=[0, 1], discrete=False)
        env = CausalShiftChain.make_shifted(p3=0.4, horizon=5000)
        state, _ = env.reset(seed=42)
        total = 0.0
        for _ in range(5000):
            z = phi_m.abstract(state)
            action = int(z[0])  # A = X1 from abstraction
            _, r, _, _, _ = env.step(action)
            total += r
            state = env._state
        assert total / 5000 > 0.99

    def test_phi_p_degrades(self):
        phi_p = PredictiveAbstraction(effect_indices=[3, 4], discrete=False)
        theta = 0.3
        env = CausalShiftChain.make_shifted(p3=theta, horizon=10000)
        state, _ = env.reset(seed=42)
        total = 0.0
        for _ in range(10000):
            z = phi_p.abstract(state)
            action = int(z[0])  # A = X4 from abstraction
            _, r, _, _, _ = env.step(action)
            total += r
            state = env._state
        mean_r = total / 10000
        # X4 disagrees with X1 when noise propagates: expected reward < 1
        assert mean_r < 0.95, f"phi_p should degrade under shift, got {mean_r:.3f}"
