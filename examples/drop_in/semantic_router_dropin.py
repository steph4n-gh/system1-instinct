#!/usr/bin/env python3
"""1-Line Drop-In Replacement for Semantic Router (aurelio-labs/semantic-router).

Demonstrates:
1. Out-of-the-box Zero-Shot routing (0 training examples, 0 API keys, <0.2ms).
2. Unsupervised Test-Time Adaptation (`adapt_zero_shot`) on unlabeled domain traffic.
3. 1-Shot Online Flywheel (`BoundedReviewFallback`) for edge-case takeover.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from system1 import BoundedReviewFallback, ChoiceField, DecisionSchema, SystemOneEngine


class SemanticRoute:
    """Drop-in compatible Route definition mirroring semantic_router.Route."""

    def __init__(self, name: str, utterances: Sequence[str]) -> None:
        self.name = name
        self.utterances = list(utterances)


@dataclass
class RouteMatchResult:
    """Result object supporting both attribute access (`res.name`) and dict access (`res['name']`)."""

    name: Optional[str]
    used_fallback: bool
    confidence: float
    latency_ms: float

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class InstinctRouteLayer:
    """Drop-in replacement for semantic_router.RouteLayer powered by System 1: Instinct."""

    def __init__(
        self,
        routes: Sequence[SemanticRoute],
        *,
        fallback_fn: Optional[Callable[[str], str]] = None,
        strict_mode: bool = False,
        confidence_threshold: float = 0.0,
    ) -> None:
        options = [r.name for r in routes]
        descriptions = {r.name: list(r.utterances) for r in routes}
        schema = DecisionSchema(
            fields={
                "route": ChoiceField(
                    options=options,
                    descriptions=descriptions,
                )
            },
            schema_name="SemanticRouterDropInSchema",
        )
        self.engine = SystemOneEngine(schema, strict_mode=strict_mode, use_cache=True)
        self.confidence_threshold = float(confidence_threshold)
        self.fallback: Optional[BoundedReviewFallback] = None
        if fallback_fn is not None:
            self.fallback = BoundedReviewFallback(
                self.engine,
                review_fn=lambda prompt: {"route": fallback_fn(prompt)},
                auto_online_update=True,
            )

    def __call__(self, text: str) -> RouteMatchResult:
        t0 = time.perf_counter()
        if self.fallback is not None:
            res = self.fallback.decide(text)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            conf = float(res.get("confidences", {}).get("route", 1.0))
            name: Optional[str] = str(res["values"]["route"])
            if self.confidence_threshold > 0.0 and conf < self.confidence_threshold:
                name = None
            return RouteMatchResult(
                name=name,
                used_fallback=bool(res.used_fallback),
                confidence=conf,
                latency_ms=round(dt_ms, 3),
            )
        dec = self.engine.decide(text, record_receipt=False)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        conf = float(dec.confidences["route"])
        name = str(dec.values["route"])
        if self.confidence_threshold > 0.0 and conf < self.confidence_threshold:
            name = None
        return RouteMatchResult(
            name=name,
            used_fallback=False,
            confidence=conf,
            latency_ms=round(dt_ms, 3),
        )

    def adapt_unlabeled(self, texts: Sequence[str]) -> Dict[str, Any]:
        """Adapts route prototypes to unlabeled domain traffic with 0 teacher calls."""
        return self.engine.adapt_zero_shot(texts, margin_threshold=0.08, learning_rate=0.30)


def build_default_router(confidence_threshold: float = 0.38) -> InstinctRouteLayer:
    """Constructs a 4-route InstinctRouteLayer with OOD confidence gating."""
    routes = [
        SemanticRoute(
            name="billing_and_refunds",
            utterances=[
                "refund my duplicate charge on billing invoice",
                "cancel annual subscription and send payment receipt",
                "overcharged on monthly statement credit card dispute",
            ],
        ),
        SemanticRoute(
            name="technical_support",
            utterances=[
                "production API returning 502 bad gateway and database timeout",
                "webhook delivery failing with stack trace exception and crash",
                "kubernetes pod crashloop and SDK connection error",
            ],
        ),
        SemanticRoute(
            name="security_escalation",
            utterances=[
                "suspicious login attempt and leaked API key credential compromise",
                "urgent security breach unauthorized access to customer records",
                "phishing email and compromised account MFA lock",
            ],
        ),
        SemanticRoute(
            name="enterprise_sales",
            utterances=[
                "enterprise pricing quote for 500 seats with custom SLA contract",
                "schedule an executive demo and procurement review for SSO",
                "volume discount licensing agreement for enterprise deployment",
            ],
        ),
    ]
    return InstinctRouteLayer(routes, confidence_threshold=confidence_threshold)


def run_demo() -> Dict[str, Any]:
    router = build_default_router(confidence_threshold=0.285)
    router.adapt_unlabeled(
        [
            "Suspicious login attempt and leaked API key credential compromise SEC_ALPHA_9",
            "Refund my duplicate charge on billing invoice BILL_OMEGA_4",
            "Production API returning 502 bad gateway and database timeout OPS_SIGMA_7",
        ]
    )

    probes = [
        ("Please refund the duplicate charge on my billing invoice", "billing_and_refunds"),
        ("Alert triggered for security code SEC_ALPHA_9 on production tenant", "security_escalation"),
        ("Cluster alert on node OPS_SIGMA_7 experiencing service degradation", "technical_support"),
    ]
    correct = sum(1 for q, exp in probes if router(q)["name"] == exp)
    return {"total": len(probes), "correct": correct, "accuracy": correct / len(probes)}


def main() -> int:
    print("=== System 1: Instinct — Semantic Router Drop-In Replacement Demo ===")
    router = build_default_router(confidence_threshold=0.38)
    for query in [
        "Please refund the duplicate charge on my billing invoice",
        "Production API returning 502 bad gateway and database timeout",
        "What is the best sourdough pizza recipe in Naples?",
    ]:
        res = router(query)
        print(f"  [0-Shot] route={str(res.name):<20} | conf={res.confidence:.2f} | latency={res.latency_ms:.3f} ms | query={query!r}")

    # Step 2: Unsupervised test-time adaptation on unlabeled logs with internal codewords
    router.adapt_unlabeled(
        [
            "Suspicious login attempt and leaked API key credential compromise SEC_ALPHA_9",
            "Refund my duplicate charge on billing invoice BILL_OMEGA_4",
            "Production API returning 502 bad gateway and database timeout OPS_SIGMA_7",
        ]
    )
    for query in [
        "Alert triggered for security code SEC_ALPHA_9 on production tenant",
        "Cluster alert on node OPS_SIGMA_7 experiencing service degradation",
    ]:
        res = router(query)
        print(f"  [Adapted] route={str(res.name):<19} | conf={res.confidence:.2f} | latency={res.latency_ms:.3f} ms | query={query!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
