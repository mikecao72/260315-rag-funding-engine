from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RuleResult:
    allowed: bool
    reasons: list[str]


def apply_basic_constraints(recommendations: list[dict[str, Any]], consult_facts: dict[str, Any]) -> RuleResult:
    """Prototype hard-gate constraints.

    Current rules:
    1) Avoid returning multiple codes with same alpha prefix family unless one is very high confidence.
    2) If consult type appears as nurse-focused in facts, downselect out obvious GP-only prefixes.

    NOTE: Placeholder until full ACC schedule rule graph is implemented.
    """
    reasons: list[str] = []
    seen = set()
    filtered = []

    for rec in recommendations:
        code = rec.get("code", "")
        prefix = "".join([c for c in code if c.isalpha()])
        if prefix in seen:
            reasons.append(f"Removed duplicate family candidate: {code}")
            continue
        seen.add(prefix)
        filtered.append(rec)

    facts = (consult_facts or {}).get("facts", {})
    consult_type = str(facts.get("consult_type") or "").lower()
    if "nurse" in consult_type:
        before = len(filtered)
        filtered = [r for r in filtered if not str(r.get("code", "")).startswith("GP")]
        if len(filtered) != before:
            reasons.append("Filtered GP-prefixed candidates for nurse consult type")

    recommendations[:] = filtered
    return RuleResult(allowed=True, reasons=reasons)
