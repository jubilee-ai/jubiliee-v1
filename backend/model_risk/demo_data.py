"""Deterministic synthetic MRM fields derived from model name (demo)."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

TIERS = ("low", "medium", "high", "critical")
LIFECYCLE = (
    "intake",
    "development",
    "validation",
    "deployment",
    "monitoring",
    "retirement",
)
MONITORING_STATUS = ("healthy", "watch", "breach", "unknown")
GOVERNANCE = ("Small bank", "Regional bank", "Large bank", "Credit union")


def _h(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)


def pick(name: str, salt: str, choices: tuple[str, ...]) -> str:
    return choices[_h(f"{name}:{salt}") % len(choices)]


def tier(name: str) -> str:
    return pick(name, "tier", TIERS)


def lifecycle_stage(name: str) -> str:
    return pick(name, "lc", LIFECYCLE)


def monitoring_status(name: str) -> str:
    return pick(name, "mon", MONITORING_STATUS)


def governance_profile(name: str) -> str:
    return pick(name, "gov", GOVERNANCE)


def lifecycle_stages(name: str, current: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    idx = LIFECYCLE.index(current) if current in LIFECYCLE else 0
    base = datetime.now(timezone.utc) - timedelta(days=120)
    for i, label in enumerate(LIFECYCLE):
        if i < idx:
            st = "complete"
        elif i == idx:
            st = "current"
        else:
            st = "pending"
        entered = (base + timedelta(days=18 * i)).isoformat()
        out.append(
            {
                "id": label,
                "label": label.replace("_", " ").title(),
                "status": st,
                "entered_at": entered if st != "pending" else None,
                "notes": None,
            }
        )
    return out


def psi_csi_rows(name: str) -> list[dict[str, Any]]:
    feats = ["income", "age", "credit_score", "dti_ratio", "employment_years"]
    rows = []
    for j, f in enumerate(feats):
        r = (_h(f"{name}:{f}") % 1000) / 10000.0
        psi = round(0.02 + r, 4)
        csi = round(0.01 + r * 0.7, 4) if j % 2 == 0 else None
        rows.append({"feature": f, "psi": psi, "csi": csi})
    return rows


def backtest_rows(name: str) -> list[dict[str, Any]]:
    return [
        {
            "period": "Q4 2025",
            "metric": "Brier score",
            "value": round(0.18 + (_h(name + "b1") % 500) / 10000.0, 4),
            "benchmark": 0.2,
            "pass": True,
        },
        {
            "period": "Q1 2026",
            "metric": "ROC-AUC",
            "value": round(0.74 + (_h(name + "b2") % 300) / 10000.0, 4),
            "benchmark": 0.72,
            "pass": (_h(name + "b2") % 5) != 0,
        },
    ]


def approvals_seed(name: str) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    return [
        {
            "id": f"{name}-a1",
            "stage": "Model validation",
            "status": "pending",
            "requested_at": (now - timedelta(days=2)).isoformat(),
            "decided_at": None,
            "actor_role": "validation",
            "comment": None,
        },
        {
            "id": f"{name}-a2",
            "stage": "Board risk",
            "status": "approved",
            "requested_at": (now - timedelta(days=40)).isoformat(),
            "decided_at": (now - timedelta(days=35)).isoformat(),
            "actor_role": "board",
            "comment": "Annual review complete.",
        },
    ]


def documents_seed(name: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{name}-d1",
            "title": "Model risk policy attestation",
            "kind": "policy",
            "href": None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "id": f"{name}-d2",
            "title": "Validation summary",
            "kind": "validation",
            "href": None,
            "updated_at": None,
        },
    ]


def reviews_seed(name: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{name}-r1",
            "review_type": "Periodic validation",
            "scheduled_for": (datetime.now(timezone.utc) + timedelta(days=90)).date().isoformat(),
            "completed_at": None,
            "outcome": None,
            "owner": "Model risk",
        }
    ]


def monitoring_narrative(status: str) -> str:
    return {
        "healthy": "Drift within policy thresholds; no escalation.",
        "watch": "Elevated PSI on select features; monitoring increased.",
        "breach": "Threshold breach detected; remediation plan required.",
        "unknown": "Insufficient monitoring history for this period.",
    }.get(status, "")
