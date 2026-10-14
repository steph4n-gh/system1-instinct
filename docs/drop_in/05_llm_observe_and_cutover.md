# Drop-In Guide 5: Wrap Any Live LLM Call $\to$ Auto-Cutover to `.s1m`

Have an existing OpenAI, Anthropic, or Gemini function that classifies tickets, flags urgency, or routes requests? Wrap it with [`BoundedReviewFallback`](../../examples/drop_in/llm_cutover_dropin.py) so System 1 **learns online from your LLM on uncertain traffic** and cuts over to **100% local sub-millisecond decisions**.

---

## 1. Step-by-Step Procedure

### Step 1: Wrap Your Existing LLM Function with `BoundedReviewFallback`
```python
from system1 import BoundedReviewFallback, SystemOneEngine
from examples.drop_in.llm_cutover_dropin import SupportTicketSchema, upstream_llm_teacher

# Start with a strict System 1 engine (0-shot cold-start or optional starter examples)
engine = SystemOneEngine(SupportTicketSchema, strict_mode=True, use_cache=False)

# Wrap your LLM callable: uncertain inputs call upstream_llm_teacher and update System 1 online
fallback = BoundedReviewFallback(
    engine,
    review_fn=upstream_llm_teacher,
    max_reviews=50,
    auto_online_update=True,
)
```

### Step 2: Send Traffic Through `fallback.decide(prompt)` (or Teach Directly with `engine.learn`)
- **Wave 1 (Novel Traffic)**: When unfamiliar jargon arrives (e.g., `"Kafka consumer lag"`, `"SIEM alert"`, `"SSO SAML login"`), System 1 detects ambiguity (`is_ambiguous=True`), calls `upstream_llm_teacher`, and applies an exact Rank-1 Sherman-Morrison RLS update (`preserve_calibration=True`) in `<1 ms`. (You can also teach directly without a wrapper via `engine.learn(prompt, labels)` or `engine.learn_from_outcome(prompt, labels, reward=1.0)`.)
- **Wave 2 (Recurring Traffic)**: When the same operational patterns recur, System 1 resolves them locally with high confidence — **LLM calls drop to `0`** and latency drops to **`~0.4 ms`**.
- **Wave 3 (Unseen Generalized Phrasings)**: When brand-new phrasings of those newly learned patterns arrive (e.g., `"Redis cluster crash and Kafka consumer lag triggering 502 gateway timeouts"` or `"Unauthorized API token compromise and CVE exploit flagged in SIEM alert"`), System 1 generalizes across them with **100% joint accuracy and `0` LLM calls**.

### Step 3: Save & Reload the Portable `.s1m` Skill
```python
# Save directly from the engine (preserves strict_mode=True and online-learned weights)
engine.save("support_ticket_instinct.s1m")

# Reload anywhere with zero external dependencies
reloaded_engine = SystemOneEngine.load("support_ticket_instinct.s1m")
decision = reloaded_engine.decide(
    "Redis cluster crash and Kafka consumer lag triggering 502 gateway timeouts",
    record_receipt=False,
)
assert not decision.is_ambiguous
```

---

## 2. Run It & See the Real Result

Run the standalone script from the repository root:

```bash
python examples/drop_in/llm_cutover_dropin.py
```

### Real Output
```text
=== System 1: Instinct — LLM Observe -> Online RLS -> Local Cutover Demo ===
  Wave 1 (Initial Stream + Novel Jargon): Joint Accuracy=6/6 (100%) | Served Locally=83% (LLM Calls=1) | Avg Latency=1.354 ms
  Wave 2 (Repeat Stream): Joint Accuracy=6/6 (100%) | Served Locally=100% (LLM Calls=0) | Avg Latency=0.888 ms
  Wave 3 (Unseen Generalized Phrasings): Joint Accuracy=6/6 (100%) | Served Locally=100% (LLM Calls=0) | Avg Latency=0.971 ms

Saved & Reloaded .s1m (12.0 KiB, strict_mode=True): queue='infrastructure', urgent=True, is_ambiguous=False (latency=1.793 ms, 0 LLM calls)
```
