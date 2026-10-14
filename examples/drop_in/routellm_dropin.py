#!/usr/bin/env python3
"""1-Line Drop-In Replacement for RouteLLM (Controller / Model Cascade Router).

Replaces ``routellm.controller.Controller`` with a sub-millisecond System 1: Instinct
router that dispatches queries across three execution tiers:
  - ``system1_local``: Deterministic fast-path for FAQ, billing, status, and simple intent
  - ``fast_small_model``: Standard summarization, formatting, translation, and drafting
  - ``frontier_reasoning_model``: Deep multi-step architecture, formal proofs, and debugging

Requires zero external API keys, starts with 0 training examples, and supports
live ``adapt_zero_shot()`` and ``learn_from_tier2()`` online refinement.
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

from system1 import BooleanField, ChoiceField, DecisionSchema, ScoreField, SystemOneEngine


class RouteLLMCascadeSchema(DecisionSchema):
    """Multi-field routing schema for LLM cost/quality cascade control."""

    target_tier = ChoiceField(
        options=[
            "system1_local",
            "fast_small_model",
            "frontier_reasoning_model",
        ],
        descriptions={
            "system1_local": [
                "simple status check account balance greeting faq password reset lookup",
                "short deterministic intent routing classification triage ticket tagging",
                "instant factual status query order tracking simple yes no question",
            ],
            "fast_small_model": [
                "summarize article rewrite email translate paragraph extract bullet points",
                "draft polite customer reply format json markdown table standard copywriting",
                "routine single step explanation grammar correction basic code snippet",
            ],
            "frontier_reasoning_model": [
                "complex multi-step distributed systems architecture race condition root cause",
                "formal mathematical proof dynamic programming algorithm optimization security audit",
                "deep debugging subtle concurrency deadlock compiler type theory theorem",
            ],
        },
    )
    requires_deep_reasoning = BooleanField(
        true_description="complex multi-step reasoning formal proof distributed architecture debugging",
        false_description="simple lookup faq status check or standard single-pass summarization",
        threshold=0.55,
    )
    complexity_score = ScoreField(
        min_value=0.0,
        max_value=1.0,
        low_description="simple status check greeting faq lookup basic formatting",
        high_description="complex multi-step architecture formal mathematical proof concurrency debugging",
    )


@dataclass(frozen=True)
class RouteDecision:
    """Result returned by SystemOneRouteController.route()."""

    prompt: str
    selected_tier: str
    selected_model: str
    requires_deep_reasoning: bool
    complexity_score: float
    confidence: float
    latency_ms: float


class SystemOneRouteController:
    """Drop-in replacement for ``routellm.controller.Controller``."""

    def __init__(
        self,
        strong_model: str = "gpt-5-reasoning",
        weak_model: str = "gpt-4o-mini",
        local_model: str = "system1-instinct-local",
        confidence_threshold: float = 0.20,
    ) -> None:
        self.strong_model = strong_model
        self.weak_model = weak_model
        self.local_model = local_model
        self.confidence_threshold = float(confidence_threshold)
        self.engine = SystemOneEngine(RouteLLMCascadeSchema, strict_mode=False, use_cache=True)
        self._tier_to_model = {
            "system1_local": self.local_model,
            "fast_small_model": self.weak_model,
            "frontier_reasoning_model": self.strong_model,
        }

    def route(self, prompt: str) -> RouteDecision:
        """Routes a prompt to the optimal model tier in sub-millisecond latency."""
        t0 = time.perf_counter()
        decision = self.engine.decide(prompt, record_receipt=False)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        tier = str(decision.values["target_tier"])
        deep = bool(decision.values["requires_deep_reasoning"])
        complexity = float(decision.values["complexity_score"])
        conf = float(min(decision.confidences.values())) if decision.confidences else 1.0

        # Safety escalation: if confidence is below threshold, escalate one tier up
        if conf < self.confidence_threshold and tier == "system1_local":
            tier = "fast_small_model"

        return RouteDecision(
            prompt=prompt,
            selected_tier=tier,
            selected_model=self._tier_to_model.get(tier, self.strong_model),
            requires_deep_reasoning=deep,
            complexity_score=round(complexity, 4),
            confidence=round(conf, 4),
            latency_ms=round(latency_ms, 3),
        )

    def adapt_unlabeled(self, prompts: Sequence[str]) -> Dict[str, Any]:
        """Adapts prototypes online from unlabeled production traffic."""
        return self.engine.adapt_zero_shot(prompts, margin_threshold=0.08, learning_rate=0.30)

    def record_feedback(self, prompt: str, true_tier: str) -> None:
        """Online 1-shot update when a downstream evaluator overrides the routed tier."""
        self.engine.learn_from_tier2(
            prompt,
            {
                "target_tier": true_tier,
                "requires_deep_reasoning": (true_tier == "frontier_reasoning_model"),
            },
            preserve_calibration=True,
        )


def main() -> int:
    controller = SystemOneRouteController()
    probes = [
        ("Simple status check on my account balance and password reset lookup", "system1_local"),
        ("Summarize this article into bullet points and draft a polite customer reply", "fast_small_model"),
        (
            "Debug this subtle concurrency deadlock and race condition in our distributed systems architecture",
            "frontier_reasoning_model",
        ),
    ]
    print("=== System 1: Instinct — RouteLLM Drop-In Replacement Demo ===")
    for text, expected in probes:
        res = controller.route(text)
        print(
            f"  [{res.selected_tier:24s} -> {res.selected_model:22s}] "
            f"(expected={expected}, complexity={res.complexity_score:.2f}, latency={res.latency_ms:.3f} ms)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
