"""
harness.py
==========
Synthetic test harness for the CalibratedTransitionModule.

Test suites
-----------
1. Normal     : uniformly-distributed signals, default thresholds.
2. Adversarial: crafted edge cases — duplicates, near-threshold clusters,
                extreme outliers, high-correlation blocks.
3. Scale      : n ∈ {10, 100, 500, 1000, 5000} — measures latency vs. n.
4. Threshold  : sweeps theta_low ∈ [0.1, 0.9] and reports quarantine rate.

Metrics collected
-----------------
  quarantine_rate : fraction of signals routed to quarantine
  admit_rate      : fraction admitted
  reject_rate     : fraction rejected
  latency_ms      : wall-clock time for route() in milliseconds
  auc             : ROC-AUC treating quarantine+reject as "positive class"
                    (computed via trapezoidal rule from scipy, with a
                    synthetic label based on true signal value > 0.6)

Usage
-----
    python harness.py                     # run all suites, print report
    python harness.py --suite scale       # run only the scale suite
    python harness.py --out results.json  # write JSON summary
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from scipy.stats import roc_auc_score as _scipy_auc
    def _auc(y_true, y_score):
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(_scipy_auc(y_true, y_score))
except ImportError:
    # Fallback: trapezoidal AUC without scipy
    def _auc(y_true, y_score):  # type: ignore[misc]
        y_true = np.asarray(y_true, dtype=float)
        y_score = np.asarray(y_score, dtype=float)
        if len(np.unique(y_true)) < 2:
            return float("nan")
        order = np.argsort(-y_score)
        y_true = y_true[order]
        n_pos = y_true.sum()
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        tp = np.cumsum(y_true)
        fp = np.arange(1, len(y_true) + 1) - tp
        tpr = tp / n_pos
        fpr = fp / n_neg
        return float(np.trapz(tpr, fpr))

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from calibrated_transition import CalibratedTransitionModule, make_ctm, RoutingResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SuiteResult:
    suite:           str
    n_signals:       int
    admit_rate:      float
    quarantine_rate: float
    reject_rate:     float
    latency_ms:      float
    auc:             float
    notes:           str = ""
    extra:           Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


# ---------------------------------------------------------------------------
# Synthetic dataset generators
# ---------------------------------------------------------------------------

class DatasetFactory:
    """Generates synthetic signal arrays for various test scenarios."""

    @staticmethod
    def normal(n: int = 200, seed: int = 0) -> np.ndarray:
        """Uniformly distributed signals in [0, 1]."""
        return np.random.default_rng(seed).uniform(0, 1, n)

    @staticmethod
    def near_threshold(n: int = 60, theta: float = 0.5, spread: float = 0.02,
                       seed: int = 1) -> np.ndarray:
        """
        Signals clustered tightly around a threshold — stress-tests boundary
        conditions that could oscillate between routing buckets.
        """
        rng = np.random.default_rng(seed)
        return np.clip(rng.normal(theta, spread, n), 0, 1)

    @staticmethod
    def duplicates(n: int = 50, n_unique: int = 3, seed: int = 2) -> np.ndarray:
        """Only n_unique distinct values repeated — tests identical-signal paths."""
        rng = np.random.default_rng(seed)
        vals = rng.uniform(0, 1, n_unique)
        return vals[rng.integers(0, n_unique, n)]

    @staticmethod
    def outliers(n: int = 100, n_outliers: int = 5, seed: int = 3) -> np.ndarray:
        """
        Bulk of signals in [0.2, 0.8] with a handful of extreme values near
        0 or 1 — tests tail behaviour of C and A computations.
        """
        rng = np.random.default_rng(seed)
        normal = rng.uniform(0.2, 0.8, n - n_outliers)
        extreme = np.concatenate([
            rng.uniform(0.0, 0.01, n_outliers // 2),
            rng.uniform(0.99, 1.0, n_outliers - n_outliers // 2),
        ])
        return np.concatenate([normal, extreme])

    @staticmethod
    def block_correlated(n_blocks: int = 4, block_size: int = 25,
                         seed: int = 4) -> np.ndarray:
        """
        Signals in tight blocks — C matrix has strong intra-block affinity and
        weak inter-block affinity.
        """
        rng = np.random.default_rng(seed)
        centers = rng.uniform(0.1, 0.9, n_blocks)
        signals = []
        for c in centers:
            signals.append(np.clip(rng.normal(c, 0.02, block_size), 0, 1))
        return np.concatenate(signals)

    @staticmethod
    def monotone(n: int = 100) -> np.ndarray:
        """Linearly spaced — deterministic, good for visual inspection."""
        return np.linspace(0.0, 1.0, n)


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def _compute_metrics(
    result: RoutingResult,
    true_label_threshold: float = 0.6,
) -> Dict[str, float]:
    """
    Compute routing rates and AUC.

    AUC: treats quarantine+reject as the 'positive' prediction.
    Ground truth: signal > true_label_threshold → positive.
    """
    n = len(result.signals)
    admit_rate      = len(result.admitted_idx)    / n
    quarantine_rate = len(result.quarantine_idx)  / n
    reject_rate     = len(result.rejected_idx)    / n

    # positive prediction score = q (higher q → more likely to flag)
    y_score = result.q
    y_true  = (result.signals > true_label_threshold).astype(int)
    auc     = _auc(y_true, y_score)

    return {
        "admit_rate":      admit_rate,
        "quarantine_rate": quarantine_rate,
        "reject_rate":     reject_rate,
        "latency_ms":      result.latency_s * 1000,
        "auc":             auc,
    }


# ---------------------------------------------------------------------------
# Individual suites
# ---------------------------------------------------------------------------

def run_normal_suite(ctm: CalibratedTransitionModule) -> SuiteResult:
    signals = DatasetFactory.normal(n=200)
    result  = ctm.route(signals)
    m       = _compute_metrics(result)
    return SuiteResult(suite="normal", n_signals=200, **m)


def run_adversarial_suite(ctm: CalibratedTransitionModule) -> List[SuiteResult]:
    results = []
    scenarios = {
        "near_threshold":   DatasetFactory.near_threshold(),
        "duplicates":       DatasetFactory.duplicates(),
        "outliers":         DatasetFactory.outliers(),
        "block_correlated": DatasetFactory.block_correlated(),
        "monotone":         DatasetFactory.monotone(),
    }
    for name, signals in scenarios.items():
        try:
            result = ctm.route(signals)
            m      = _compute_metrics(result)
            results.append(SuiteResult(
                suite=f"adversarial/{name}",
                n_signals=len(signals),
                notes=f"adversarial scenario: {name}",
                **m,
            ))
        except Exception as exc:  # noqa: BLE001
            results.append(SuiteResult(
                suite=f"adversarial/{name}",
                n_signals=len(signals),
                admit_rate=float("nan"),
                quarantine_rate=float("nan"),
                reject_rate=float("nan"),
                latency_ms=float("nan"),
                auc=float("nan"),
                notes=f"FAILED: {exc}",
            ))
    return results


def run_scale_suite(ctm: CalibratedTransitionModule) -> List[SuiteResult]:
    sizes   = [10, 50, 100, 500, 1000, 5000]
    results = []
    rng     = np.random.default_rng(77)
    for n in sizes:
        signals = rng.uniform(0, 1, n)
        result  = ctm.route(signals)
        m       = _compute_metrics(result)
        results.append(SuiteResult(
            suite=f"scale/n={n}",
            n_signals=n,
            notes=f"n={n} scale test",
            **m,
        ))
    return results


def run_threshold_sweep(
    theta_lows: Optional[np.ndarray] = None,
    n_signals: int = 200,
    seed: int = 0,
) -> List[SuiteResult]:
    """
    Sweep theta_low while keeping theta_high = theta_low + 0.3.
    Measures how quarantine_rate varies with thresholds.
    """
    if theta_lows is None:
        theta_lows = np.linspace(0.1, 0.7, 13)
    signals = DatasetFactory.normal(n=n_signals, seed=seed)
    results = []
    for tl in theta_lows:
        th = min(tl + 0.3, 1.0)
        ctm = CalibratedTransitionModule(theta_low=float(tl), theta_high=float(th))
        result = ctm.route(signals)
        m = _compute_metrics(result)
        results.append(SuiteResult(
            suite=f"threshold_sweep/tl={tl:.2f}",
            n_signals=n_signals,
            notes=f"theta_low={tl:.2f} theta_high={th:.2f}",
            **m,
        ))
    return results


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

_COL_WIDTH = 28

def _print_header():
    cols = ["Suite", "n", "admit%", "quar%", "reject%", "lat_ms", "AUC"]
    print("─" * 100)
    print(
        f"{'Suite':<{_COL_WIDTH}} {'n':>6} "
        f"{'admit%':>8} {'quar%':>8} {'reject%':>8} "
        f"{'lat_ms':>8} {'AUC':>7}"
    )
    print("─" * 100)

def _print_row(r: SuiteResult):
    def pct(v): return f"{v*100:.1f}" if not (v != v) else "nan"
    def flt(v): return f"{v:.2f}" if not (v != v) else "nan"
    print(
        f"{r.suite:<{_COL_WIDTH}} {r.n_signals:>6} "
        f"{pct(r.admit_rate):>8} {pct(r.quarantine_rate):>8} {pct(r.reject_rate):>8} "
        f"{flt(r.latency_ms):>8} {flt(r.auc):>7}"
    )

def print_report(all_results: List[SuiteResult]):
    _print_header()
    for r in all_results:
        _print_row(r)
    print("─" * 100)


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def run_all(
    suite_filter: Optional[str] = None,
    out_path: Optional[str] = None,
    theta_low:  float = 0.3,
    theta_high: float = 0.7,
    tau:        float = 1.0,
    dynamic_tau: bool = False,
) -> List[SuiteResult]:
    ctm = make_ctm(
        theta_low=theta_low,
        theta_high=theta_high,
        tau=tau,
        dynamic_tau=dynamic_tau,
    )
    all_results: List[SuiteResult] = []

    suites = {
        "normal":    lambda: [run_normal_suite(ctm)],
        "adversarial": lambda: run_adversarial_suite(ctm),
        "scale":     lambda: run_scale_suite(ctm),
        "threshold": lambda: run_threshold_sweep(),
    }

    for name, fn in suites.items():
        if suite_filter and not name.startswith(suite_filter):
            continue
        print(f"\n▶  Running suite: {name} …")
        rows = fn()
        if isinstance(rows, SuiteResult):
            rows = [rows]
        all_results.extend(rows)

    print()
    print_report(all_results)

    if out_path:
        serialisable = [asdict(r) for r in all_results]
        with open(out_path, "w") as fh:
            json.dump(serialisable, fh, indent=2)
        print(f"\nResults written to {out_path}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CTM Test Harness — run synthetic test suites."
    )
    parser.add_argument(
        "--suite",
        choices=["normal", "adversarial", "scale", "threshold"],
        default=None,
        help="Run only a specific suite (default: all).",
    )
    parser.add_argument("--out",    default=None, help="Path to write JSON results.")
    parser.add_argument("--theta-low",  type=float, default=0.3)
    parser.add_argument("--theta-high", type=float, default=0.7)
    parser.add_argument("--tau",        type=float, default=1.0)
    parser.add_argument("--dynamic-tau", action="store_true")
    args = parser.parse_args()

    run_all(
        suite_filter=args.suite,
        out_path=args.out,
        theta_low=args.theta_low,
        theta_high=args.theta_high,
        tau=args.tau,
        dynamic_tau=args.dynamic_tau,
    )
