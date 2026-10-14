#!/usr/bin/env python3
"""1-Line Drop-In Replacement for Instructor / BAML Structured Output Extraction.

Replaces ``instructor.from_openai(client).chat.completions.create(response_model=...)``
with a sub-millisecond System 1: Instinct multi-field schema extractor that populates:
  - ``ChoiceField`` (single categorical classification)
  - ``BooleanField`` (calibrated binary flag)
  - ``MultiChoiceField`` (multi-label set extraction)
  - ``ScoreField`` (bounded continuous score)

Runs locally in < 0.5 ms with zero LLM API calls, starting from 0 training examples
and supporting live 1-shot online updates via ``learn_from_tier2()``.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Type

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from system1 import (
    BooleanField,
    ChoiceField,
    DecisionSchema,
    MultiChoiceField,
    ScoreField,
    SystemOneEngine,
)


class IncidentTriageModel(DecisionSchema):
    """Multi-field structured extraction schema replacing an Instructor Pydantic model."""

    department = ChoiceField(
        options=["security_ops", "billing_finance", "platform_infra", "customer_success"],
        descriptions={
            "security_ops": [
                "unauthorized access credential leak token compromise phish breach firewall",
                "suspicious login mfa bypass privilege escalation malware intrusion",
            ],
            "billing_finance": [
                "duplicate invoice charge refund wire transfer tax receipt subscription payment",
                "credit card declined overcharge billing dispute payout settlement",
            ],
            "platform_infra": [
                "database outage latency spike 502 bad gateway kubernetes pod crash oom",
                "api timeout replication lag deployment rollback dns failure",
            ],
            "customer_success": [
                "feature request onboarding question workspace seat invite documentation",
                "how to export csv dashboard walkthrough training session",
            ],
        },
    )
    page_oncall = BooleanField(
        true_description="urgent critical production outage security breach data loss immediate escalation",
        false_description="routine question minor request standard billing inquiry invoice charge refund receipt subscription non-urgent",
        threshold=0.50,
    )
    impact_tags = MultiChoiceField(
        options=["auth", "payments", "database", "api"],
        descriptions={
            "auth": "login password token mfa sso credential permission",
            "payments": "invoice charge refund card subscription wire payout",
            "database": "postgres replica query deadlock migration storage",
            "api": "502 gateway timeout webhook endpoint latency rate limit",
        },
        threshold=0.52,
    )
    urgency_score = ScoreField(
        min_value=0.0,
        max_value=1.0,
        low_description="routine question documentation minor cosmetic request",
        high_description="critical production outage security breach unauthorized access",
    )


@dataclass(frozen=True)
class StructuredExtractionResult:
    """Typed response returned by SystemOneInstructorClient.create()."""

    values: Dict[str, Any]
    confidence: float
    latency_ms: float


class SystemOneInstructorClient:
    """Drop-in replacement for Instructor's ``client.chat.completions.create(response_model=...)``."""

    def __init__(self, response_model: Type[DecisionSchema] = IncidentTriageModel) -> None:
        self.response_model = response_model
        self.engine = SystemOneEngine(response_model, strict_mode=False, use_cache=True)

    def create(self, prompt: str) -> StructuredExtractionResult:
        """Extracts all schema fields simultaneously in a single non-autoregressive pass."""
        t0 = time.perf_counter()
        decision = self.engine.decide(prompt, record_receipt=False)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        conf = float(min(decision.confidences.values())) if decision.confidences else 1.0
        return StructuredExtractionResult(
            values=dict(decision.values),
            confidence=round(conf, 4),
            latency_ms=round(latency_ms, 3),
        )

    def adapt_unlabeled(self, texts: Sequence[str]) -> Dict[str, Any]:
        """Adapts schema prototypes to unlabeled domain traffic."""
        return self.engine.adapt_zero_shot(texts, margin_threshold=0.08, learning_rate=0.30)

    def teach_example(self, prompt: str, structured_output: Dict[str, Any]) -> None:
        """Online 1-shot update across all schema fields simultaneously."""
        self.engine.learn_from_tier2(prompt, structured_output, preserve_calibration=True)


def main() -> int:
    client = SystemOneInstructorClient(IncidentTriageModel)
    samples = [
        "Urgent security breach: unauthorized token compromise and suspicious login bypassing MFA",
        "Duplicate invoice charge on our credit card subscription, need an immediate refund receipt",
        "Critical production outage: postgres database deadlock causing 502 gateway API timeouts",
        "Routine onboarding question: how to export CSV from dashboard and invite a workspace seat",
    ]
    print("=== System 1: Instinct — Instructor Structured Extraction Drop-In Demo ===")
    for text in samples:
        res = client.create(text)
        print(
            f"  dept={res.values['department']:18s} | page_oncall={str(res.values['page_oncall']):5s} | "
            f"tags={str(res.values['impact_tags']):24s} | urgency={res.values['urgency_score']:.2f} | "
            f"latency={res.latency_ms:.3f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
