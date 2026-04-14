"""Tests for the revised CausalShift-XOR environment.

Validates:
  1. S1 is iid Bernoulli(0.5) (exogenous, random)
  2. S2 = S1 XOR Bernoulli(theta)
  3. Reward = 1 if A == S1
  4. At theta=0: phi_m and phi_p produce identical abstract states
  5. At theta>0: phi_p gives noisy observations, phi_m stays clean
  6. Frozen policy A=Z has reward 1.0 under phi_m, (1-theta) under phi_p
"""

import numpy as np
import pytest

from causalshift.abstractions.mechanism_invariant import MechanismInvariantAbstraction
from causalshift.abstractions.predictive import PredictiveAbstraction
from causalshift.envs.xor import CausalShiftXOR, CausalShiftXORAbstracted


class TestEnvironmentMechanics:
    def test_s1_is_uniform(self):
        env = CausalShiftXOR(theta=0.0, horizon=10000)
        env.reset(seed=42)
        s1_values = []
        for _ in range(10000):
            state, _, _, _, _ = env.step(0)
            s1_values.append(state[0])
        assert abs(np.mean(s1_values) - 0.5) < 0.02

    def test_s2_equals_s1_at_theta_zero(self):
        env = CausalShiftXOR(theta=0.0, horizon=1000)
        env.reset(seed=42)
        for _ in range(1000):
            state, _, _, _, _ = env.step(0)
            assert state[0] == state[1], "At theta=0, S2 must equal S1"

    def test_s2_noisy_at_theta_half(self):
        env = CausalShiftXOR(theta=0.5, horizon=10000)
        env.reset(seed=42)
        disagreements = 0
        for _ in range(10000):
            state, _, _, _, _ = env.step(0)
            if state[0] != state[1]:
                disagreements += 1
        frac = disagreements / 10000
        assert abs(frac - 0.5) < 0.03, f"Expected ~50% disagreement, got {frac:.1%}"

    def test_reward_matches_s1(self):
        env = CausalShiftXOR(theta=0.3, horizon=100)
        state, _ = env.reset(seed=42)
        for _ in range(100):
            action = int(state[0])  # Match S1
            _, reward, _, _, _ = env.step(action)
            assert reward == 1.0, "Matching S1 should always give reward 1"
            state, _ = env.reset()

    def test_reward_zero_on_mismatch(self):
        env = CausalShiftXOR(theta=0.0, horizon=100)
        state, _ = env.reset(seed=42)
        for _ in range(100):
            action = 1 - int(state[0])  # Opposite of S1
            _, reward, _, _, _ = env.step(action)
            assert reward == 0.0, "Mismatching S1 should give reward 0"
            state, _ = env.reset()


class TestAbstractionEquivalence:
    def test_identical_at_source(self):
        phi_m = MechanismInvariantAbstraction.for_xor()
        phi_p = PredictiveAbstraction.for_xor()
        env = CausalShiftXOR(theta=0.0, horizon=500)
        state, _ = env.reset(seed=42)
        for _ in range(500):
            assert phi_m.abstract(state) == phi_p.abstract(state)
            state, _, _, _, _ = env.step(0)

    def test_diverge_under_shift(self):
        phi_m = MechanismInvariantAbstraction.for_xor()
        phi_p = PredictiveAbstraction.for_xor()
        env = CausalShiftXOR(theta=0.3, horizon=5000)
        state, _ = env.reset(seed=42)
        n_diff = 0
        for _ in range(5000):
            if phi_m.abstract(state) != phi_p.abstract(state):
                n_diff += 1
            state, _, _, _, _ = env.step(0)
        assert n_diff / 5000 > 0.2, f"Expected ~30% divergence, got {n_diff/5000:.1%}"


class TestSeparationTheorem:
    """The central empirical claim: frozen policy A=Z gives different rewards."""

    def test_phi_m_perfect_under_any_shift(self):
        """phi_m: policy A=Z=S1 gives reward 1.0 regardless of theta."""
        for theta in [0.0, 0.2, 0.5]:
            phi_m = MechanismInvariantAbstraction.for_xor()
            env = CausalShiftXOR(theta=theta, horizon=5000)
            state, _ = env.reset(seed=42)
            total = 0.0
            for _ in range(5000):
                z = phi_m.abstract(state)
                _, reward, _, _, _ = env.step(z)  # A = Z = S1
                total += reward
                state = env._state
            assert total / 5000 > 0.99, f"phi_m should get ~1.0 at theta={theta}, got {total/5000}"

    def test_phi_p_degrades_linearly(self):
        """phi_p: policy A=Z=S2 gives reward (1-theta)."""
        for theta, expected in [(0.0, 1.0), (0.2, 0.8), (0.4, 0.6)]:
            phi_p = PredictiveAbstraction.for_xor()
            env = CausalShiftXOR(theta=theta, horizon=10000)
            state, _ = env.reset(seed=42)
            total = 0.0
            for _ in range(10000):
                z = phi_p.abstract(state)
                _, reward, _, _, _ = env.step(z)  # A = Z = S2
                total += reward
                state = env._state
            mean_r = total / 10000
            assert abs(mean_r - expected) < 0.03, (
                f"phi_p at theta={theta}: expected ~{expected}, got {mean_r:.3f}"
            )

    def test_regret_gap_scales_with_theta(self):
        """The regret gap between phi_p and phi_m should be ~theta * T."""
        phi_m = MechanismInvariantAbstraction.for_xor()
        phi_p = PredictiveAbstraction.for_xor()
        T = 10000
        theta = 0.3

        env_m = CausalShiftXOR(theta=theta, horizon=T)
        state_m, _ = env_m.reset(seed=99)
        reward_m = 0.0
        for _ in range(T):
            z = phi_m.abstract(state_m)
            _, r, _, _, _ = env_m.step(z)
            reward_m += r
            state_m = env_m._state

        env_p = CausalShiftXOR(theta=theta, horizon=T)
        state_p, _ = env_p.reset(seed=99)
        reward_p = 0.0
        for _ in range(T):
            z = phi_p.abstract(state_p)
            _, r, _, _, _ = env_p.step(z)
            reward_p += r
            state_p = env_p._state

        gap = reward_m - reward_p  # Should be ~theta * T = 3000
        # Allow margin but should be clearly positive and large
        assert gap > 0.2 * T * theta, (
            f"Regret gap should be ~{theta*T:.0f}, got {gap:.0f}"
        )
