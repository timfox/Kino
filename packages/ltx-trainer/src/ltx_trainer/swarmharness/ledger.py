"""Local SwarmCredit ledger with signed attribution entries (Sec. 3.4)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.swarmharness.credit import AttributionResult
from ltx_trainer.swarmharness.identity import CreditReceipt, verify_credit_receipt


@dataclass
class LedgerEntry:
    timestamp: float
    task_id: str
    node_id: str
    delta: float
    signed_receipt: dict[str, Any] | None = None


@dataclass
class LocalCreditLedger:
    """Per-node ledger of credit balance and append-only delta history."""

    balances: dict[str, float] = field(default_factory=dict)
    entries: list[LedgerEntry] = field(default_factory=list)
    pending_receipts: list[dict[str, Any]] = field(default_factory=list)

    def balance(self, node_id: str) -> float:
        return self.balances.get(node_id, 0.0)

    def ensure_account(self, node_id: str, initial: float = 0.0) -> None:
        if node_id not in self.balances:
            self.balances[node_id] = initial

    def apply_signed_attribution(
        self,
        signed: dict[str, Any],
        *,
        require_countersign: bool = True,
    ) -> dict[str, Any]:
        """
        Apply credit deltas after submitter countersignature (Sec. 5.4).
        Returns summary; raises ValueError if signature invalid.
        """
        if require_countersign and not verify_credit_receipt(signed):
            raise ValueError("invalid or missing submitter signature on credit receipt")
        payload = signed["receipt"]
        task_id = str(payload["task_id"])
        ts = float(payload.get("issued_at", time.time()))
        applied: dict[str, float] = {}
        for node_id, delta in payload["deltas"].items():
            d = float(delta)
            self.ensure_account(node_id)
            self.balances[node_id] += d
            self.entries.append(
                LedgerEntry(timestamp=ts, task_id=task_id, node_id=node_id, delta=d, signed_receipt=signed)
            )
            applied[node_id] = d
        submitter = str(payload["submitter_id"])
        pool = float(payload["credit_pool"])
        self.ensure_account(submitter)
        self.balances[submitter] -= pool
        self.entries.append(
            LedgerEntry(
                timestamp=ts,
                task_id=task_id,
                node_id=submitter,
                delta=-pool,
                signed_receipt=signed,
            )
        )
        return {"task_id": task_id, "applied": applied, "submitter_debit": pool}

    def apply_attribution_result(
        self,
        task_id: str,
        submitter_id: str,
        result: AttributionResult,
        signed: dict[str, Any],
    ) -> dict[str, Any]:
        _ = result
        return self.apply_signed_attribution(signed)

    def summary(self) -> dict[str, Any]:
        return {
            "accounts": len(self.balances),
            "entries": len(self.entries),
            "balances": {k: round(v, 6) for k, v in sorted(self.balances.items())},
        }
