"""
integration_example.py
=======================
Shows how to integrate CalibratedTransitionModule with:

  1. An Evidence Ledger  — append-only structured log of routing decisions
     (JSON-Lines format, one record per signal, fully serialisable).
  2. A Verification Loop — re-processes quarantined signals with tightened
     thresholds and human-review hooks, then writes final disposition.

The module is self-contained; no external services are required.

Run
---
    python integration_example.py                  # demo run
    python integration_example.py --ledger-path my_ledger.jsonl
    python integration_example.py --signals 0.1 0.45 0.72 0.9
"""

from __future__ import annotations

import argparse
import json
import uuid
import datetime
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from calibrated_transition import (
    CalibratedTransitionModule,
    RoutingResult,
    make_ctm,
)


# ===========================================================================
# Evidence Ledger
# ===========================================================================

@dataclass
class LedgerEntry:
    """
    Single record in the Evidence Ledger.

    Fields follow a minimal schema that can be extended with
    domain-specific fields via the `extra` dict.
    """
    # ---- identifiers -------------------------------------------------------
    entry_id:     str   = field(default_factory=lambda: str(uuid.uuid4()))
    batch_id:     str   = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:    str   = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z"
    )

    # ---- signal payload ----------------------------------------------------
    signal_index: int   = 0
    signal_value: float = 0.0

    # ---- CTM outputs -------------------------------------------------------
    q:            float = 0.0          # calibrated uncertainty score
    A_i:          float = 0.0          # affinity weight
    C_max:        float = 0.0          # off-diagonal C max for the batch

    # ---- routing -----------------------------------------------------------
    decision:     str   = "quarantine" # admit | quarantine | reject
    theta_low:    float = 0.3
    theta_high:   float = 0.7

    # ---- verification loop -------------------------------------------------
    verification_round: int  = 0      # 0 = initial, 1+ = re-processed
    final_decision:     Optional[str] = None  # set after verification
    human_reviewed:     bool = False

    # ---- metadata ----------------------------------------------------------
    extra: Dict = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), default=str)


class EvidenceLedger:
    """
    Append-only Evidence Ledger backed by a JSON-Lines file.

    Each call to :meth:`write_batch` appends one :class:`LedgerEntry`
    per signal.  The ledger is flushed to disk after each batch.

    Parameters
    ----------
    path : str
        Filesystem path for the .jsonl file.
    mode : str
        ``'a'`` to append to an existing ledger (default) or
        ``'w'`` to overwrite.
    """

    def __init__(self, path: str = "evidence_ledger.jsonl", mode: str = "a") -> None:
        self.path = path
        self._fh  = open(path, mode, encoding="utf-8")   # kept open for efficiency
        self._count = 0

    # ------------------------------------------------------------------
    def write_batch(
        self,
        result: RoutingResult,
        batch_id: Optional[str] = None,
        verification_round: int = 0,
        extra_per_signal: Optional[List[Dict]] = None,
    ) -> List[LedgerEntry]:
        """
        Write one ledger entry per signal in *result*.

        Parameters
        ----------
        result : RoutingResult
        batch_id : str, optional
            Shared identifier for all entries in this batch.
        verification_round : int
            0 for initial routing, 1+ for re-verification passes.
        extra_per_signal : list of dicts, optional
            Per-signal extra metadata; must match len(result.signals).

        Returns
        -------
        list of LedgerEntry
        """
        if batch_id is None:
            batch_id = str(uuid.uuid4())

        c_max = CalibratedTransitionModule.C_max_masked(result.C)
        theta_low  = float(np.asarray(result.metadata.get("theta_low",  0.3)).flat[0])
        theta_high = float(np.asarray(result.metadata.get("theta_high", 0.7)).flat[0])
        entries    = []

        for i, (sig, q, A_i, decision) in enumerate(
            zip(result.signals, result.q, result.A, result.decisions)
        ):
            extra = {}
            if extra_per_signal and i < len(extra_per_signal):
                extra = extra_per_signal[i]

            entry = LedgerEntry(
                batch_id           = batch_id,
                signal_index       = i,
                signal_value       = float(sig),
                q                  = float(q),
                A_i                = float(A_i),
                C_max              = c_max,
                decision           = decision,
                theta_low          = theta_low,
                theta_high         = theta_high,
                verification_round = verification_round,
                extra              = extra,
            )
            self._fh.write(entry.to_jsonl() + "\n")
            self._count += 1
            entries.append(entry)

        self._fh.flush()
        return entries

    # ------------------------------------------------------------------
    def read_all(self) -> List[Dict]:
        """Read and parse all entries currently in the ledger file."""
        self._fh.flush()
        with open(self.path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def quarantined_entries(self) -> List[Dict]:
        return [e for e in self.read_all() if e["decision"] == "quarantine"
                and e["final_decision"] is None]

    def __len__(self) -> int:
        return self._count

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ===========================================================================
# Verification Loop
# ===========================================================================

# Signature for a human-review hook:
# Given a LedgerEntry dict, return True to approve (admit), False to reject.
HumanReviewHook = Callable[[Dict], bool]


def _default_human_review(entry: Dict) -> bool:
    """
    Stub reviewer — in production, replace with an actual human-in-the-loop
    interface (e.g., push to a review queue, await webhook, etc.).

    This stub approves entries with q < 0.6 and rejects the rest.
    """
    return entry["q"] < 0.6


class VerificationLoop:
    """
    Re-processes quarantined signals through up to *max_rounds* of
    progressively tighter thresholds.

    Algorithm
    ---------
    Round 1: Narrow the quarantine band by ``band_shrink`` on each side.
    Round 2: If still quarantined, invoke the human-review hook.
    Round 3+: Human review always called; further band shrinkage ignored.

    Final dispositions are written back to the ledger as new entries with
    ``verification_round`` > 0.

    Parameters
    ----------
    ledger : EvidenceLedger
    ctm : CalibratedTransitionModule
        Base CTM; thresholds are modified per-round.
    max_rounds : int, default 2
    band_shrink : float, default 0.05
        Amount to tighten theta_low (raise) and theta_high (lower) per round.
    human_review : callable, optional
        Hook called on round 2+; receives a ledger entry dict.
    """

    def __init__(
        self,
        ledger:        EvidenceLedger,
        ctm:           CalibratedTransitionModule,
        max_rounds:    int   = 2,
        band_shrink:   float = 0.05,
        human_review:  Optional[HumanReviewHook] = None,
    ) -> None:
        self.ledger       = ledger
        self.ctm          = ctm
        self.max_rounds   = max_rounds
        self.band_shrink  = band_shrink
        self.human_review = human_review or _default_human_review

    # ------------------------------------------------------------------
    def run(
        self,
        quarantined_entries: List[Dict],
        batch_id_prefix: str = "verify",
    ) -> List[Dict]:
        """
        Run the verification loop on a list of quarantined ledger entries.

        Returns
        -------
        List of final LedgerEntry dicts (one per input entry) with
        ``final_decision`` populated.
        """
        if not quarantined_entries:
            print("  [VerificationLoop] No quarantined entries to process.")
            return []

        pending   = list(quarantined_entries)
        finalised = []

        for rnd in range(1, self.max_rounds + 2):
            if not pending:
                break

            print(f"  [VerificationLoop] Round {rnd}: {len(pending)} signal(s) pending.")

            tighten    = self.band_shrink * (rnd - 1)
            theta_low  = min(float(pending[0]["theta_low"])  + tighten, 0.95)
            theta_high = max(float(pending[0]["theta_high"]) - tighten, theta_low + 0.01)

            signals = np.array([e["signal_value"] for e in pending], dtype=np.float64)

            # Re-route with tightened thresholds
            ctm_tight = CalibratedTransitionModule(
                theta_low=theta_low, theta_high=theta_high,
            )
            meta = {"theta_low": theta_low, "theta_high": theta_high}
            result = ctm_tight.route(signals, metadata=meta)

            still_quarantined = []
            batch_id = f"{batch_id_prefix}_r{rnd}_{uuid.uuid4().hex[:6]}"
            entries  = self.ledger.write_batch(
                result,
                batch_id=batch_id,
                verification_round=rnd,
            )

            for orig_entry, new_entry, decision in zip(pending, entries, result.decisions):
                if decision == "quarantine":
                    if rnd >= 2:
                        # Escalate to human review
                        approved = self.human_review(asdict(new_entry))
                        final    = "admit" if approved else "reject"
                        new_entry.human_reviewed = True
                        new_entry.final_decision = final
                        print(
                            f"    Signal[{new_entry.signal_index}] "
                            f"q={new_entry.q:.3f} → human_review → {final}"
                        )
                        finalised.append(asdict(new_entry))
                    else:
                        still_quarantined.append(orig_entry)
                else:
                    new_entry.final_decision = decision
                    print(
                        f"    Signal[{new_entry.signal_index}] "
                        f"q={new_entry.q:.3f} → {decision}"
                    )
                    finalised.append(asdict(new_entry))

            pending = still_quarantined

        # Any remaining (shouldn't happen, but safeguard)
        for e in pending:
            e["final_decision"] = "reject"
            e["verification_round"] = self.max_rounds + 1
            finalised.append(e)

        return finalised


# ===========================================================================
# End-to-end demo
# ===========================================================================

def run_demo(
    signals:      Optional[List[float]] = None,
    ledger_path:  str  = "evidence_ledger.jsonl",
    theta_low:    float = 0.3,
    theta_high:   float = 0.7,
    verbose:      bool  = True,
) -> None:
    """
    Full end-to-end integration example.

    1. Route a batch of signals through CTM.
    2. Write all decisions to the Evidence Ledger.
    3. Pass quarantined signals through the Verification Loop.
    4. Print a ledger summary.
    """
    if signals is None:
        signals = [0.05, 0.15, 0.42, 0.48, 0.55, 0.71, 0.88, 0.95]

    sig_arr = np.array(signals, dtype=np.float64)

    # ------------------------------------------------------------------
    print("=" * 60)
    print("CalibratedTransitionModule — Integration Example")
    print("=" * 60)
    print(f"Signals:      {sig_arr.tolist()}")
    print(f"theta_low:    {theta_low}   theta_high: {theta_high}")
    print(f"Ledger path:  {ledger_path}")
    print()

    # ---- Step 1: Initial routing -----------------------------------------
    ctm = make_ctm(theta_low=theta_low, theta_high=theta_high)
    result = ctm.route(
        sig_arr,
        metadata={"theta_low": theta_low, "theta_high": theta_high},
    )

    print("── Initial Routing ──")
    for i, (sig, q, decision) in enumerate(
        zip(result.signals, result.q, result.decisions)
    ):
        marker = {"admit": "✓", "quarantine": "?", "reject": "✗"}[decision]
        print(f"  [{marker}] signal[{i}]={sig:.3f}  q={q:.4f}  → {decision}")

    print()
    print(f"  Admitted:    {len(result.admitted_idx)}")
    print(f"  Quarantined: {len(result.quarantine_idx)}")
    print(f"  Rejected:    {len(result.rejected_idx)}")
    print(f"  C_max (off-diag): {CalibratedTransitionModule.C_max_masked(result.C):.4f}")
    print()

    # ---- Step 2: Write to Evidence Ledger --------------------------------
    with EvidenceLedger(ledger_path, mode="w") as ledger:
        batch_id = f"demo_{uuid.uuid4().hex[:8]}"
        entries = ledger.write_batch(
            result,
            batch_id=batch_id,
            extra_per_signal=[{"source": "demo"} for _ in range(len(signals))],
        )
        print(f"── Evidence Ledger ──")
        print(f"  Written {len(entries)} entries to {ledger_path}")
        print()

        # ---- Step 3: Verification Loop for quarantined -------------------
        quarantined = ledger.quarantined_entries()

        if quarantined:
            print("── Verification Loop ──")
            loop = VerificationLoop(
                ledger=ledger,
                ctm=ctm,
                max_rounds=2,
                band_shrink=0.05,
            )
            final_entries = loop.run(quarantined, batch_id_prefix="verify")
            print()

            admit_final  = sum(1 for e in final_entries if e["final_decision"] == "admit")
            reject_final = sum(1 for e in final_entries if e["final_decision"] == "reject")
            human_count  = sum(1 for e in final_entries if e.get("human_reviewed"))
            print(f"  Verification outcomes:")
            print(f"    → Final admit:   {admit_final}")
            print(f"    → Final reject:  {reject_final}")
            print(f"    → Human reviews: {human_count}")
            print()

        # ---- Step 4: Ledger summary -------------------------------------
        all_entries = ledger.read_all()
        print("── Ledger Summary ──")
        print(f"  Total records: {len(all_entries)}")
        by_decision = {}
        for e in all_entries:
            d = e["decision"]
            by_decision[d] = by_decision.get(d, 0) + 1
        for d, cnt in sorted(by_decision.items()):
            print(f"  {d:<12}: {cnt}")
        print()

        # Print last 3 ledger entries as sample JSON
        if verbose:
            print("── Sample Ledger Entries (last 3) ──")
            for e in all_entries[-3:]:
                print(json.dumps(e, indent=2, default=str))
                print()

    print(f"Done. Full ledger written to: {os.path.abspath(ledger_path)}")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CTM Integration Example — Evidence Ledger + Verification Loop"
    )
    parser.add_argument(
        "--signals", type=float, nargs="+", default=None,
        help="Space-separated signal values (default: built-in demo set).",
    )
    parser.add_argument("--ledger-path", default="evidence_ledger.jsonl")
    parser.add_argument("--theta-low",   type=float, default=0.3)
    parser.add_argument("--theta-high",  type=float, default=0.7)
    parser.add_argument("--quiet",       action="store_true",
                        help="Suppress sample ledger JSON output.")
    args = parser.parse_args()

    run_demo(
        signals     = args.signals,
        ledger_path = args.ledger_path,
        theta_low   = args.theta_low,
        theta_high  = args.theta_high,
        verbose     = not args.quiet,
    )
