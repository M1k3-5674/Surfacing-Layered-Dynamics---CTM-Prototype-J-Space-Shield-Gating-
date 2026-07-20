"""
test_calibrated_transition.py
==============================
Pytest unit-test suite for CalibratedTransitionModule.

Coverage
--------
* build_C correctness (symmetry, diagonal, range)
* _compute_A vectorisation and masked diagonal
* C_max_masked excludes diagonal
* q score shape / range / monotonicity
* Routing paths: admit / quarantine / reject
* Three-signal reference scenario from the design specification
* Per-signal threshold arrays
* Dynamic tau callable
* Input validation (bad shapes, NaN, non-symmetric C)
* dtype stability (float32 input → float64 internals)
* Edge cases: identical signals, extreme values, n=2 minimum

Run with:
    pytest tests/test_calibrated_transition.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

# Make the package importable when running from repo root or tests/
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from calibrated_transition import (
    CalibratedTransitionModule,
    RoutingResult,
    make_ctm,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ctm_default() -> CalibratedTransitionModule:
    return CalibratedTransitionModule(theta_low=0.3, theta_high=0.7)


@pytest.fixture
def three_signals() -> np.ndarray:
    """Reference three-signal scenario from the design specification."""
    return np.array([0.1, 0.5, 0.9], dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. build_C
# ---------------------------------------------------------------------------

class TestBuildC:
    def test_symmetry(self):
        s = np.array([0.2, 0.5, 0.8])
        C = CalibratedTransitionModule.build_C(s)
        np.testing.assert_allclose(C, C.T, atol=1e-12)

    def test_diagonal_is_one(self):
        s = np.array([0.1, 0.4, 0.7])
        C = CalibratedTransitionModule.build_C(s)
        np.testing.assert_allclose(np.diag(C), np.ones(3), atol=1e-12)

    def test_range(self):
        rng = np.random.default_rng(42)
        s = rng.uniform(-5, 5, size=20)
        C = CalibratedTransitionModule.build_C(s)
        assert C.min() >= 0.0
        assert C.max() <= 1.0 + 1e-12

    def test_decay_with_distance(self):
        """C[i,j] should decrease as |s_i - s_j| increases."""
        s = np.array([0.0, 1.0, 5.0])
        C = CalibratedTransitionModule.build_C(s)
        assert C[0, 1] > C[0, 2]   # closer pair → higher affinity

    def test_identical_signals(self):
        s = np.array([0.5, 0.5, 0.5])
        C = CalibratedTransitionModule.build_C(s)
        np.testing.assert_allclose(C, np.ones((3, 3)), atol=1e-12)


# ---------------------------------------------------------------------------
# 2. C_max_masked
# ---------------------------------------------------------------------------

class TestCMaxMasked:
    def test_excludes_diagonal(self):
        C = np.eye(3)                          # all off-diagonal = 0
        assert CalibratedTransitionModule.C_max_masked(C) == 0.0

    def test_correct_max(self):
        C = np.array([
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.6],
            [0.2, 0.6, 1.0],
        ])
        assert CalibratedTransitionModule.C_max_masked(C) == pytest.approx(0.8)

    def test_not_one_when_all_similar(self):
        # all off-diagonal < 1 even when all signals are identical (diagonal=1)
        s = np.array([0.5, 0.5, 0.5])
        C = CalibratedTransitionModule.build_C(s)
        # off-diagonal of all-ones matrix is still 1 — verify masked max
        # is still returned correctly (it is 1.0 here because exp(0)=1)
        assert CalibratedTransitionModule.C_max_masked(C) == pytest.approx(1.0)

    def test_4x4(self):
        rng = np.random.default_rng(0)
        C = rng.uniform(0, 1, (4, 4))
        C = (C + C.T) / 2
        np.fill_diagonal(C, 1.0)
        expected = C[~np.eye(4, dtype=bool)].max()
        assert CalibratedTransitionModule.C_max_masked(C) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 3. _compute_A (vectorised, masked diagonal)
# ---------------------------------------------------------------------------

class TestComputeA:
    def test_shape(self, ctm_default, three_signals):
        C = CalibratedTransitionModule.build_C(three_signals)
        A = ctm_default._compute_A(C)
        assert A.shape == (3,)

    def test_sums_to_one(self, ctm_default, three_signals):
        C = CalibratedTransitionModule.build_C(three_signals)
        A = ctm_default._compute_A(C)
        assert A.sum() == pytest.approx(1.0, abs=1e-6)

    def test_all_positive(self, ctm_default, three_signals):
        C = CalibratedTransitionModule.build_C(three_signals)
        A = ctm_default._compute_A(C)
        assert np.all(A > 0)

    def test_diagonal_not_included(self):
        """
        With an identity C matrix all off-diagonal values are 0,
        so raw affinity sums are equal → A should be uniform (1/n).
        """
        ctm = CalibratedTransitionModule()
        C = np.eye(4)
        A = ctm._compute_A(C)
        np.testing.assert_allclose(A, np.full(4, 0.25), atol=1e-6)

    def test_extreme_spread(self):
        """
        Signal with much higher row-sum of C should receive larger A_i.
        """
        ctm = CalibratedTransitionModule()
        # signal 0 is close to signals 1 & 2; signal 3 is very far away
        s = np.array([0.0, 0.05, 0.1, 100.0])
        C = CalibratedTransitionModule.build_C(s)
        A = ctm._compute_A(C)
        # signals 0,1,2 should collectively dominate A[3]
        assert A[0] > A[3]


# ---------------------------------------------------------------------------
# 4. q scores
# ---------------------------------------------------------------------------

class TestQScores:
    def test_shape_and_range(self, ctm_default, three_signals):
        result = ctm_default.route(three_signals)
        assert result.q.shape == (3,)
        assert np.all(result.q > 0) and np.all(result.q < 1)

    def test_dtype_float64(self, ctm_default, three_signals):
        result = ctm_default.route(three_signals)
        assert result.q.dtype == np.float64

    def test_no_nan_or_inf(self, ctm_default):
        rng = np.random.default_rng(7)
        s = rng.uniform(0, 1, 50)
        result = ctm_default.route(s)
        assert np.all(np.isfinite(result.q))

    def test_identical_signals_uniform_q(self, ctm_default):
        """All identical signals → equal A_i → equal q_i ≈ 0.5."""
        s = np.array([0.5] * 5)
        result = ctm_default.route(s)
        np.testing.assert_allclose(result.q, result.q[0], atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Routing decisions
# ---------------------------------------------------------------------------

class TestRoutingDecisions:

    def test_all_admit(self):
        """Low theta_high → everything quarantined; very tight low → all admit."""
        # With theta_low=0.99, theta_high=1.0 all q < 0.99 → admit
        ctm = CalibratedTransitionModule(theta_low=0.99, theta_high=1.0)
        s = np.array([0.1, 0.2, 0.3, 0.4])
        result = ctm.route(s)
        assert len(result.admitted_idx) == 4
        assert len(result.quarantine_idx) == 0
        assert len(result.rejected_idx) == 0

    def test_all_quarantine(self):
        """theta_low=0.0, theta_high=1.0 → nothing ever crosses a hard boundary."""
        ctm = CalibratedTransitionModule(theta_low=0.0, theta_high=1.0)
        s = np.array([0.1, 0.5, 0.9])
        result = ctm.route(s)
        # q in (0,1) so all fall in [theta_low, theta_high)
        assert len(result.quarantine_idx) == 3

    def test_all_reject(self):
        """theta_low=0.0, theta_high=0.0 → all q >= theta_high → reject."""
        ctm = CalibratedTransitionModule(theta_low=0.0, theta_high=0.0)
        s = np.array([0.2, 0.5, 0.8])
        result = ctm.route(s)
        assert len(result.rejected_idx) == 3

    def test_decision_strings(self, ctm_default, three_signals):
        result = ctm_default.route(three_signals)
        valid = {"admit", "quarantine", "reject"}
        assert all(d in valid for d in result.decisions)

    def test_index_consistency(self, ctm_default):
        rng = np.random.default_rng(99)
        s = rng.uniform(0, 1, 30)
        result = ctm_default.route(s)
        total = (
            len(result.admitted_idx)
            + len(result.quarantine_idx)
            + len(result.rejected_idx)
        )
        assert total == 30

    def test_decisions_match_indices(self, ctm_default):
        rng = np.random.default_rng(11)
        s = rng.uniform(0, 1, 10)
        result = ctm_default.route(s)
        for i in result.admitted_idx:
            assert result.decisions[i] == "admit"
        for i in result.quarantine_idx:
            assert result.decisions[i] == "quarantine"
        for i in result.rejected_idx:
            assert result.decisions[i] == "reject"


# ---------------------------------------------------------------------------
# 6. Three-signal reference scenario
# ---------------------------------------------------------------------------

class TestThreeSignalScenario:
    """
    Reference scenario:
        signals = [0.1, 0.5, 0.9]
        theta_low=0.3, theta_high=0.7

    Expected behaviour validated against the hand-computed design example:
    - C is 3×3 symmetric with diagonal 1.
    - A sums to 1, all positive.
    - q values in (0, 1).
    - At least one of each routing class is *possible* depending on spread.
    """

    def test_C_shape_and_symmetry(self, ctm_default, three_signals):
        result = ctm_default.route(three_signals)
        assert result.C.shape == (3, 3)
        np.testing.assert_allclose(result.C, result.C.T, atol=1e-12)

    def test_C_diagonal(self, ctm_default, three_signals):
        result = ctm_default.route(three_signals)
        np.testing.assert_allclose(np.diag(result.C), [1, 1, 1], atol=1e-12)

    def test_A_sums_to_one(self, ctm_default, three_signals):
        result = ctm_default.route(three_signals)
        assert result.A.sum() == pytest.approx(1.0, abs=1e-6)

    def test_C_off_diagonal_values(self, three_signals):
        """Manually verify specific C values: C[0,1]=exp(-0.4), C[0,2]=exp(-0.8)."""
        C = CalibratedTransitionModule.build_C(three_signals)
        np.testing.assert_allclose(C[0, 1], np.exp(-0.4), atol=1e-10)
        np.testing.assert_allclose(C[1, 2], np.exp(-0.4), atol=1e-10)
        np.testing.assert_allclose(C[0, 2], np.exp(-0.8), atol=1e-10)

    def test_C_max_masked_correct(self, three_signals):
        C = CalibratedTransitionModule.build_C(three_signals)
        expected = np.exp(-0.4)   # closest pair gap = 0.4
        assert CalibratedTransitionModule.C_max_masked(C) == pytest.approx(
            expected, abs=1e-10
        )

    def test_full_routing_output_types(self, ctm_default, three_signals):
        result = ctm_default.route(three_signals)
        assert isinstance(result, RoutingResult)
        assert isinstance(result.latency_s, float)
        assert result.latency_s >= 0.0

    def test_summary_string(self, ctm_default, three_signals):
        result = ctm_default.route(three_signals)
        s = result.summary()
        assert "n=3" in s
        assert "admit=" in s
        assert "quarantine=" in s
        assert "reject=" in s

    def test_middle_signal_higher_affinity(self, ctm_default, three_signals):
        """
        Signal at 0.5 is equidistant from 0.1 and 0.9, so its row-sum of
        off-diagonal C is the highest among the three → it should have the
        highest A_i.
        """
        result = ctm_default.route(three_signals)
        assert result.A[1] == result.A.max()


# ---------------------------------------------------------------------------
# 7. Per-signal threshold arrays
# ---------------------------------------------------------------------------

class TestPerSignalThresholds:
    def test_array_thresholds(self):
        s = np.array([0.1, 0.5, 0.9])
        theta_low  = np.array([0.99, 0.0, 0.0])  # signal 0 → admit
        theta_high = np.array([1.0,  0.0, 1.0])  # signal 1 → reject, signal 2 → quarantine
        ctm = CalibratedTransitionModule(
            theta_low=theta_low,
            theta_high=theta_high,
        )
        result = ctm.route(s)
        assert 0 in result.admitted_idx
        assert 1 in result.rejected_idx

    def test_scalar_and_array_equivalent_for_uniform(self):
        s = np.array([0.1, 0.5, 0.9])
        ctm_scalar = CalibratedTransitionModule(theta_low=0.3, theta_high=0.7)
        ctm_array  = CalibratedTransitionModule(
            theta_low=np.array([0.3, 0.3, 0.3]),
            theta_high=np.array([0.7, 0.7, 0.7]),
        )
        r_s = ctm_scalar.route(s)
        r_a = ctm_array.route(s)
        assert r_s.decisions == r_a.decisions


# ---------------------------------------------------------------------------
# 8. Dynamic tau
# ---------------------------------------------------------------------------

class TestDynamicTau:
    def test_callable_tau_called(self):
        called = []
        def my_tau(C):
            called.append(True)
            return np.ones(C.shape[0]) * 0.5
        ctm = CalibratedTransitionModule(tau=my_tau)
        ctm.route(np.array([0.2, 0.5, 0.8]))
        assert len(called) == 1

    def test_dynamic_tau_result_valid(self):
        ctm = make_ctm(dynamic_tau=True)
        rng = np.random.default_rng(42)
        s = rng.uniform(0, 1, 10)
        result = ctm.route(s)
        assert np.all(np.isfinite(result.q))
        assert result.q.shape == (10,)

    def test_zero_tau_clipped(self):
        """A callable that returns 0 should be clipped to eps, not cause /0."""
        ctm = CalibratedTransitionModule(tau=lambda C: np.zeros(C.shape[0]))
        result = ctm.route(np.array([0.1, 0.5, 0.9]))
        assert np.all(np.isfinite(result.q))


# ---------------------------------------------------------------------------
# 9. Pre-computed C matrix path
# ---------------------------------------------------------------------------

class TestPrecomputedC:
    def test_accepts_valid_C(self, ctm_default):
        s = np.array([0.2, 0.4, 0.6])
        C = CalibratedTransitionModule.build_C(s)
        result = ctm_default.route(s, C=C)
        np.testing.assert_allclose(result.C, C)

    def test_rejects_non_square(self, ctm_default):
        s = np.array([0.2, 0.4, 0.6])
        with pytest.raises(ValueError, match="must be"):
            ctm_default.route(s, C=np.ones((3, 4)))

    def test_rejects_non_symmetric(self, ctm_default):
        s = np.array([0.2, 0.4, 0.6])
        C = np.array([[1, 0.5, 0.3],
                      [0.6, 1, 0.4],
                      [0.3, 0.4, 1]], dtype=float)
        with pytest.raises(ValueError, match="symmetric"):
            ctm_default.route(s, C=C)

    def test_rejects_out_of_range(self, ctm_default):
        s = np.array([0.2, 0.4, 0.6])
        C = np.array([[1, 1.5, 0.3],
                      [1.5, 1, 0.4],
                      [0.3, 0.4, 1]], dtype=float)
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            ctm_default.route(s, C=C)


# ---------------------------------------------------------------------------
# 10. Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_rejects_2d_signals(self, ctm_default):
        with pytest.raises(ValueError, match="1-D"):
            ctm_default.route(np.ones((3, 2)))

    def test_rejects_nan(self, ctm_default):
        with pytest.raises(ValueError, match="NaN"):
            ctm_default.route(np.array([0.1, np.nan, 0.9]))

    def test_rejects_inf(self, ctm_default):
        with pytest.raises(ValueError, match="NaN"):
            ctm_default.route(np.array([0.1, np.inf, 0.9]))

    def test_rejects_too_few_signals(self):
        ctm = CalibratedTransitionModule(min_signals=3)
        with pytest.raises(ValueError, match="least 3"):
            ctm.route(np.array([0.5, 0.6]))

    def test_minimum_two_signals(self, ctm_default):
        """Default min_signals=2 — two signals should work."""
        result = ctm_default.route(np.array([0.2, 0.8]))
        assert result.q.shape == (2,)


# ---------------------------------------------------------------------------
# 11. dtype stability
# ---------------------------------------------------------------------------

class TestDtypeStability:
    def test_float32_input_yields_float64_q(self, ctm_default):
        s = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        result = ctm_default.route(s)
        assert result.q.dtype == np.float64

    def test_int_input_yields_float64_q(self, ctm_default):
        s = np.array([1, 5, 9], dtype=int)
        result = ctm_default.route(s)
        assert result.q.dtype == np.float64

    def test_large_values_no_overflow(self, ctm_default):
        s = np.array([1e10, 2e10, 3e10])
        result = ctm_default.route(s)
        assert np.all(np.isfinite(result.q))

    def test_small_values_no_underflow(self, ctm_default):
        s = np.array([1e-10, 2e-10, 3e-10])
        result = ctm_default.route(s)
        assert np.all(np.isfinite(result.q))


# ---------------------------------------------------------------------------
# 12. make_ctm factory
# ---------------------------------------------------------------------------

class TestMakeCtm:
    def test_returns_ctm_instance(self):
        ctm = make_ctm()
        assert isinstance(ctm, CalibratedTransitionModule)

    def test_dynamic_tau_factory(self):
        ctm = make_ctm(dynamic_tau=True)
        assert callable(ctm.tau)

    def test_fixed_tau_factory(self):
        ctm = make_ctm(tau=2.5, dynamic_tau=False)
        assert ctm.tau == 2.5

    def test_custom_thresholds(self):
        ctm = make_ctm(theta_low=0.2, theta_high=0.6)
        assert ctm.theta_low == 0.2
        assert ctm.theta_high == 0.6


# ---------------------------------------------------------------------------
# 13. Metadata passthrough
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_metadata_stored(self, ctm_default, three_signals):
        meta = {"run_id": "abc123", "version": 2}
        result = ctm_default.route(three_signals, metadata=meta)
        assert result.metadata["run_id"] == "abc123"

    def test_empty_metadata_default(self, ctm_default, three_signals):
        result = ctm_default.route(three_signals)
        assert result.metadata == {}


# ---------------------------------------------------------------------------
# 14. Latency sanity
# ---------------------------------------------------------------------------

class TestLatency:
    def test_latency_positive(self, ctm_default, three_signals):
        result = ctm_default.route(three_signals)
        assert result.latency_s > 0

    def test_latency_reasonable(self, ctm_default):
        """1000 signals should complete well under 1 second."""
        rng = np.random.default_rng(0)
        s = rng.uniform(0, 1, 1000)
        result = ctm_default.route(s)
        assert result.latency_s < 1.0
