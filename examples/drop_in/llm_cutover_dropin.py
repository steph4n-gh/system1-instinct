#!/usr/bin/env python3
"""Drop-In Procedure 5: Wrap Any LLM Call -> Observe -> Auto-Cutover to Local .s1m.

Demonstrates the complete real procedure for replacing a recurring LLM API call
(e.g. OpenAI / Anthropic / Gemini ticket classifier) with System 1: Instinct:
  Step 1: Start with a 0-shot schema and wrap your LLM function with BoundedReviewFallback.
  Step 2: Stream traffic through the wrapper. Uncertain queries call the LLM fallback
          and update System 1 online in <1 ms via Sherman-Morrison RLS.
  Step 3: As System 1 learns the boundary, LLM calls drop to 0% and 100% of traffic
          is served locally in ~0.3 ms.
  Step 4: Save the learned skill to a portable `.s1m` file for zero-dependency deployment.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from system1 import (
    BooleanField,
    BoundedReviewFallback,
    ChoiceField,
    CompiledSystemOneModel,
    DecisionSchema,
    SystemOneCompiler,
    SystemOneEngine,
)


# ---------------------------------------------------------------------------
# Step 1: Define the structured decision schema you currently ask an LLM for
# ---------------------------------------------------------------------------
class SupportTicketSchema(DecisionSchema):
    queue = ChoiceField(
        options=["billing", "infrastructure", "security", "account_access"],
        descriptions={
            "billing": [
                "duplicate invoice charge credit card refund subscription receipt",
                "wire transfer payout tax form monthly statement overcharge",
            ],
            "infrastructure": [
                "kubernetes pod crashloop 502 bad gateway database replica lag",
                "api timeout webhook failure memory leak deployment rollback",
            ],
            "security": [
                "unauthorized login credential leak phishing token compromise",
                "suspicious ip firewall alert privilege escalation data breach",
            ],
            "account_access": [
                "reset password mfa authenticator locked out sso saml login",
                "invite team member workspace role permission seat assignment",
            ],
        },
    )
    urgent = BooleanField(
        true_description="production outage security breach unauthorized compromise locked out critical",
        false_description="routine billing receipt tax form invite member standard question",
        threshold=0.52,
    )


# Simulated upstream LLM teacher (in production, this is your openai/anthropic call)
def upstream_llm_teacher(prompt: str) -> Dict[str, object]:
    p = prompt.lower()
    if any(k in p for k in ("invoice", "refund", "charge", "ledger", "wire", "vat")):
        return {"queue": "billing", "urgent": False}
    if any(k in p for k in ("502", "kubernetes", "database", "crash", "kafka", "redis")):
        return {"queue": "infrastructure", "urgent": True}
    if any(k in p for k in ("breach", "phishing", "token", "unauthorized", "cve", "siem")):
        return {"queue": "security", "urgent": True}
    return {"queue": "account_access", "urgent": "locked" in p}


def main() -> int:
    # Step 1: Start from a cold-start 0-shot engine (or optional starter examples)
    engine = SystemOneEngine(SupportTicketSchema, strict_mode=True, use_cache=False)

    # Teach 3 starter examples per class online (or via BoundedReviewFallback)
    seed_examples: List[Tuple[str, Dict[str, object]]] = [
        ("Please refund the duplicate charge on my monthly invoice", {"queue": "billing", "urgent": False}),
        ("Download our annual subscription receipt and VAT tax form", {"queue": "billing", "urgent": False}),
        ("Wire transfer payout settlement is missing from our billing statement", {"queue": "billing", "urgent": False}),
        ("Production API returning 502 bad gateway after kubernetes deployment", {"queue": "infrastructure", "urgent": True}),
        ("Postgres database replica lag spiking and webhook delivery timing out", {"queue": "infrastructure", "urgent": True}),
        ("Worker pod crashloop due to memory leak in production cluster", {"queue": "infrastructure", "urgent": True}),
        ("Unauthorized login attempt and leaked API token detected on firewall", {"queue": "security", "urgent": True}),
        ("Suspicious phishing email and privilege escalation alert in logs", {"queue": "security", "urgent": True}),
        ("Customer data breach alert from anomalous IP address", {"queue": "security", "urgent": True}),
        ("Locked out of my account after resetting MFA authenticator phone", {"queue": "account_access", "urgent": True}),
        ("How do I invite a team member and update workspace role permissions?", {"queue": "account_access", "urgent": False}),
        ("SSO SAML login redirect fails when assigning a new seat", {"queue": "account_access", "urgent": False}),
    ]
    for prompt, label in seed_examples:
        engine.learn(prompt, label)

    llm_call_count = 0

    def tracked_llm_teacher(prompt: str) -> Dict[str, object]:
        nonlocal llm_call_count
        llm_call_count += 1
        return upstream_llm_teacher(prompt)

    fallback = BoundedReviewFallback(
        engine,
        review_fn=tracked_llm_teacher,
        max_reviews=50,
        auto_online_update=True,
    )

    # Wave 1: Novel domain jargon (e.g. 'Kafka consumer lag', 'SIEM alert', 'ledger wire')
    wave_1_requests: List[Tuple[str, Dict[str, object]]] = [
        ("Refund my duplicate credit card charge on the billing invoice", {"queue": "billing", "urgent": False}),
        ("Kafka consumer lag and Redis cluster crash causing 502 timeouts", {"queue": "infrastructure", "urgent": True}),
        ("SIEM alert: CVE exploit and unauthorized token compromise", {"queue": "security", "urgent": True}),
        ("Locked out of SSO SAML login after MFA authenticator reset", {"queue": "account_access", "urgent": True}),
        ("Ledger wire payout overcharge on monthly VAT statement", {"queue": "billing", "urgent": False}),
        ("Invite a new team member seat with workspace role permissions", {"queue": "account_access", "urgent": False}),
    ]

    # Wave 2: Repeat traffic on taught patterns
    wave_2_requests = list(wave_1_requests)

    # Wave 3: Unseen generalized phrasings (never seen in seed_examples or Wave 1/2)
    wave_3_unseen_requests: List[Tuple[str, Dict[str, object]]] = [
        ("Duplicate credit card charge on our monthly billing invoice needs a refund", {"queue": "billing", "urgent": False}),
        ("Redis cluster crash and Kafka consumer lag triggering 502 gateway timeouts", {"queue": "infrastructure", "urgent": True}),
        ("Unauthorized API token compromise and CVE exploit flagged in SIEM alert", {"queue": "security", "urgent": True}),
        ("MFA authenticator reset locked out my account from SSO SAML login", {"queue": "account_access", "urgent": True}),
        ("Monthly VAT billing statement has a wire transfer payout overcharge", {"queue": "billing", "urgent": False}),
        ("Update workspace role permissions to invite a new team member seat", {"queue": "account_access", "urgent": False}),
    ]

    waves = [
        ("Wave 1 (Initial Stream + Novel Jargon)", wave_1_requests),
        ("Wave 2 (Repeat Stream)", wave_2_requests),
        ("Wave 3 (Unseen Generalized Phrasings)", wave_3_unseen_requests),
    ]

    print("=== System 1: Instinct — LLM Observe -> Online RLS -> Local Cutover Demo ===")
    for wave_label, requests in waves:
        calls_before = llm_call_count
        correct = 0
        latencies: List[float] = []
        for prompt, expected in requests:
            t0 = time.perf_counter()
            res = fallback.decide(prompt)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            vals = res["values"]
            if vals["queue"] == expected["queue"] and bool(vals["urgent"]) == bool(expected["urgent"]):
                correct += 1
        wave_llm_calls = llm_call_count - calls_before
        local_rate = 1.0 - (wave_llm_calls / len(requests))
        avg_ms = sum(latencies) / len(latencies)
        print(
            f"  {wave_label}: Joint Accuracy={correct}/{len(requests)} ({correct/len(requests):.0%}) | "
            f"Served Locally={local_rate:.0%} (LLM Calls={wave_llm_calls}) | Avg Latency={avg_ms:.3f} ms"
        )

    with tempfile.TemporaryDirectory() as tmp:
        s1m_path = Path(tmp) / "support_ticket_instinct.s1m"
        engine.save(s1m_path)
        reloaded_engine = SystemOneEngine.load(s1m_path)
        probe = "Redis cluster crash and Kafka consumer lag triggering 502 gateway timeouts"
        dec = reloaded_engine.decide(probe, record_receipt=False)
        assert reloaded_engine.strict_mode is True
        assert not dec.is_ambiguous
        assert dec.values == {"queue": "infrastructure", "urgent": True}
        print(
            f"\nSaved & Reloaded .s1m ({s1m_path.stat().st_size / 1024:.1f} KiB, strict_mode={reloaded_engine.strict_mode}): "
            f"queue={dec.values['queue']!r}, urgent={dec.values['urgent']}, is_ambiguous={dec.is_ambiguous} "
            f"(latency={dec.latency_ms:.3f} ms, 0 LLM calls)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
