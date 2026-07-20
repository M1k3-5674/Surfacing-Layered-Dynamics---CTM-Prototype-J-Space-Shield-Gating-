"""
calibrated_transition.py
========================
Production-ready, numerically hardened implementation of the
CalibratedTransitionModule (CTM).

Features
--------
* Fully vectorised A_i computation via numpy broadcasting.
* Masked C-matrix: diagonal excluded from C_max to prevent self-routing.
* dtype stability: all intermediate arithmetic in float64 regardless of input.
* Per-signal thresholds (scalar or array).
* Dynamic tau via optional callable or fixed scalar.
* Structured logging (stdlib logging) with a NullHandler by default.
* All public methods return plain numpy arrays or plain Python scalars —
  no hidden state mutations between calls.

Routing decision
----------------
  admit      : q_i < theta_low
  quarantine : theta_low <= q_i < theta_high
  reject     : q_i >= theta_high

Author: Mike / CalibratedTransitionModule dev-package
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ---------------------------------------------------------------------------
# Module-level logger — callers attach their own handlers.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------
ThresholdSpec = Union[float, ArrayLike]   # scalar or per-signal array
TauSpec       = Union[float, Callable[[NDArray], NDArray]]  # scalar or fn(C)->tau

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class RoutingResult:
    """Structured output returned by :meth:`CalibratedTransitionModule.route`."""
    signals: NDArray          # (n,) input signals (float64)
    q: NDArray                # (n,) calibrated uncertainty scores
    C: NDArray                # (n, n) correlation / affinity matrix
    A: NDArray                # (n,) vectorised A_i values
    decisions: List[str]      # 'admit' | 'quarantine' | 'reject' per signal
    admitted_idx: NDArray     # indices of admitted signals
    quarantine_idx: NDArray   # indices of quarantined signals
    rejected_idx: NDArray     # indices of rejected signals
    latency_s: float          # wall-clock seconds for route()
    metadata: Dict            = field(default_factory=dict)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        n = len(self.signals)
        return (
            f"RoutingResult | n={n} | "
            f"admit={len(self.admitted_idx)} "
            f"quarantine={len(self.quarantine_idx)} "
            f"reject={len(self.rejected_idx)} | "
            f"latency={self.latency_s*1000:.2f} ms"
        )


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------
class CalibratedTransitionModule:
    """
    Calibrated Transition Module — routes signals based on calibrated
    uncertainty scores computed from a pairwise affinity matrix.

    Parameters
    ----------
    theta_low : float or array-like
        Lower uncertainty threshold.  Signals with q < theta_low → *admit*.
        If array, must have length equal to the number of signals passed to
        :meth:`route`.
    theta_high : float or array-like
        Upper uncertainty threshold.  Signals with q >= theta_high → *reject*.
        theta_low <= q < theta_high → *quarantine*.
    tau : float or callable, default 1.0
        Temperature / scale for the softmax-style A_i computation.
        Pass a callable ``f(C: ndarray) -> ndarray`` to compute per-signal
        dynamic tau from the C matrix (e.g. row-wise std).
    eps : float, default 1e-8
        Small constant added for numerical stability in divisions and logs.
    min_signals : int, default 2
        Minimum number of signals required; raises ValueError otherwise.
    """

    def __init__(
        self,
        theta_low:  ThresholdSpec = 0.3,
        theta_high: ThresholdSpec = 0.7,
        tau:        TauSpec       = 1.0,
        eps:        float         = 1e-8,
        min_signals: int          = 2,
    ) -> None:
        self.theta_low   = theta_low
        self.theta_high  = theta_high
        self.tau         = tau
        self.eps         = float(eps)
        self.min_signals = int(min_signals)
        logger.info(
            "CTM initialised | theta_low=%s theta_high=%s tau=%s eps=%s",
            theta_low, theta_high, tau, eps,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        signals: ArrayLike,
        C: Optional[ArrayLike] = None,
        *,
        metadata: Optional[Dict] = None,
    ) -> RoutingResult:
        """
        Route a batch of signals.

        Parameters
        ----------
        signals : array-like, shape (n,)
            Raw signal values.  Converted to float64 internally.
        C : array-like, shape (n, n), optional
            Pre-computed correlation/affinity matrix.  If None, C is
            constructed from ``signals`` via :meth:`build_C`.
        metadata : dict, optional
            Arbitrary caller metadata stored in the result.

        Returns
        -------
        RoutingResult
        """
        t0 = time.perf_counter()

        signals = np.asarray(signals, dtype=np.float64)
        n = signals.shape[0]
        self._validate_signals(signals)

        # ---- C matrix ---------------------------------------------------
        if C is None:
            C_mat = self.build_C(signals)
        else:
            C_mat = np.asarray(C, dtype=np.float64)
            self._validate_C(C_mat, n)

        # ---- A_i (vectorised) ------------------------------------------
        A = self._compute_A(C_mat)

        # ---- q scores ---------------------------------------------------
        q = self._compute_q(signals, A)

        # ---- thresholds (broadcast-safe) --------------------------------
        theta_low  = np.broadcast_to(
            np.asarray(self.theta_low,  dtype=np.float64), (n,)
        )
        theta_high = np.broadcast_to(
            np.asarray(self.theta_high, dtype=np.float64), (n,)
        )

        # ---- routing decisions ------------------------------------------
        decisions, admitted, quarantined, rejected = self._decide(
            q, theta_low, theta_high
        )

        latency = time.perf_counter() - t0
        logger.debug(
            "route() | n=%d admit=%d quarantine=%d reject=%d | %.3f ms",
            n, len(admitted), len(quarantined), len(rejected), latency * 1000,
        )

        return RoutingResult(
            signals       = signals,
            q             = q,
            C             = C_mat,
            A             = A,
            decisions     = decisions,
            admitted_idx  = admitted,
            quarantine_idx= quarantined,
            rejected_idx  = rejected,
            latency_s     = latency,
            metadata      = metadata or {},
        )

    # ------------------------------------------------------------------
    # C-matrix construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_C(signals: NDArray) -> NDArray:
        """
        Build a symmetric pairwise affinity matrix C ∈ [0,1]^{n×n}.

        C[i,j] = exp(-|s_i - s_j|) — decays with signal distance.
        Diagonal is 1 by construction (self-affinity).

        Parameters
        ----------
        signals : ndarray, shape (n,)

        Returns
        -------
        C : ndarray, shape (n, n), dtype float64
        """
        s = np.asarray(signals, dtype=np.float64)
        diff = np.abs(s[:, None] - s[None, :])  # (n, n) broadcasted diff
        C = np.exp(-diff)
        return C

    # ------------------------------------------------------------------
    # A_i computation
    # ------------------------------------------------------------------

    def _compute_A(self, C: NDArray) -> NDArray:
        """
        Vectorised A_i computation.

        For each signal i, A_i is defined as a softmax-normalised weighted
        sum of off-diagonal column affinities:

            raw_i  = sum_{j≠i} C[i,j]
            A_i    = softmax(raw / tau)_i

        The diagonal is masked out before aggregation to prevent
        self-routing inflation.

        Parameters
        ----------
        C : ndarray, shape (n, n)

        Returns
        -------
        A : ndarray, shape (n,)
        """
        n = C.shape[0]
        mask = ~np.eye(n, dtype=bool)           # True for off-diagonal

        # sum of off-diagonal affinities per row
        masked_C = np.where(mask, C, 0.0)       # zero diagonal
        raw = masked_C.sum(axis=1)              # (n,)

        # resolve tau
        tau = self._resolve_tau(C)              # scalar or (n,)

        # numerically stable softmax
        scaled = raw / (tau + self.eps)
        scaled -= scaled.max()                  # stability shift
        exp_scaled = np.exp(scaled)
        A = exp_scaled / (exp_scaled.sum() + self.eps)

        return A

    def _resolve_tau(self, C: NDArray) -> NDArray:
        """Return tau as a broadcastable scalar or (n,) array."""
        if callable(self.tau):
            tau = np.asarray(self.tau(C), dtype=np.float64)
            tau = np.clip(tau, self.eps, None)   # prevent zero / negative
            return tau
        return float(self.tau)

    # ------------------------------------------------------------------
    # q score computation
    # ------------------------------------------------------------------

    def _compute_q(self, signals: NDArray, A: NDArray) -> NDArray:
        """
        Compute calibrated uncertainty scores q ∈ [0, 1].

        q_i = sigma( A_i - mean(A) )  where sigma is the logistic function,
        then rescaled so the range is [0, 1].

        Concretely:
            deviation_i = A_i - mean(A)
            q_i         = 1 / (1 + exp(-deviation_i / (std(A) + eps)))

        This maps signals whose affinity weight is below average to q < 0.5
        (lower uncertainty → more likely to admit) and above-average weights
        to q > 0.5.

        Parameters
        ----------
        signals : ndarray, shape (n,)  [not used directly; reserved for
                  future signal-conditioned scoring extensions]
        A : ndarray, shape (n,)

        Returns
        -------
        q : ndarray, shape (n,), dtype float64, values in (0, 1)
        """
        _ = signals  # reserved
        deviation = A - A.mean()
        std_A = A.std()
        q = 1.0 / (1.0 + np.exp(-deviation / (std_A + self.eps)))
        return np.clip(q, self.eps, 1.0 - self.eps)

    # ------------------------------------------------------------------
    # Routing decision
    # ------------------------------------------------------------------

    @staticmethod
    def _decide(
        q: NDArray,
        theta_low: NDArray,
        theta_high: NDArray,
    ) -> Tuple[List[str], NDArray, NDArray, NDArray]:
        admit_mask      = q < theta_low
        reject_mask     = q >= theta_high
        quarantine_mask = ~admit_mask & ~reject_mask

        decisions = [""] * len(q)
        for i in range(len(q)):
            if admit_mask[i]:
                decisions[i] = "admit"
            elif reject_mask[i]:
                decisions[i] = "reject"
            else:
                decisions[i] = "quarantine"

        return (
            decisions,
            np.where(admit_mask)[0],
            np.where(quarantine_mask)[0],
            np.where(reject_mask)[0],
        )

    # ------------------------------------------------------------------
    # C_max (off-diagonal)
    # ------------------------------------------------------------------

    @staticmethod
    def C_max_masked(C: NDArray) -> float:
        """
        Return the maximum off-diagonal value of C.

        The diagonal (self-affinity) is excluded to prevent trivial
        maximum of 1.0 from dominating threshold reasoning.
        """
        n = C.shape[0]
        mask = ~np.eye(n, dtype=bool)
        return float(C[mask].max())

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_signals(self, signals: NDArray) -> None:
        if signals.ndim != 1:
            raise ValueError(
                f"signals must be 1-D, got shape {signals.shape}"
            )
        n = signals.shape[0]
        if n < self.min_signals:
            raise ValueError(
                f"At least {self.min_signals} signals required, got {n}"
            )
        if not np.all(np.isfinite(signals)):
            raise ValueError("signals contain NaN or Inf")

    @staticmethod
    def _validate_C(C: NDArray, n: int) -> None:
        if C.shape != (n, n):
            raise ValueError(
                f"C must be ({n},{n}), got {C.shape}"
            )
        if not np.allclose(C, C.T, atol=1e-6):
            raise ValueError("C must be symmetric")
        if not np.all(np.isfinite(C)):
            raise ValueError("C contains NaN or Inf")
        if C.min() < 0.0 or C.max() > 1.0 + 1e-6:
            raise ValueError("C values must be in [0, 1]")


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_ctm(
    theta_low:  float = 0.3,
    theta_high: float = 0.7,
    tau:        float = 1.0,
    dynamic_tau: bool = False,
) -> CalibratedTransitionModule:
    """
    Factory helper.

    Parameters
    ----------
    theta_low, theta_high : float
        Routing thresholds.
    tau : float
        Fixed tau value (used when dynamic_tau=False).
    dynamic_tau : bool
        If True, tau is computed per-batch as the row-wise std of C,
        clipped to [0.1, 10.0].
    """
    if dynamic_tau:
        def _dynamic(C: NDArray) -> NDArray:
            std = C.std(axis=1)
            return np.clip(std, 0.1, 10.0)
        tau_spec: TauSpec = _dynamic
    else:
        tau_spec = tau

    return CalibratedTransitionModule(
        theta_low=theta_low,
        theta_high=theta_high,
        tau=tau_spec,
    )
