# Drop-In Guide 5: Wrap Any Live LLM Call $\to$ Auto-Cutover to `.s1m`

Have an existing OpenAI, Anthropic, or Gemini function that classifies tickets, flags urgency, or routes requests? Wrap it with [`BoundedReviewFallback`](../../examples/drop_in/llm_cutover_dropin.py) so System 1 **learns online from your LLM on uncertain traffic** and cuts over to **100% local sub-millisecond decisions**.

---

## 1. Step-by-Step Procedure

### Step 1: Wrap Your Existing LLM Function with `BoundedReviewFallback`
```python
from system1 import BoundedReviewFallback, SystemOneEngine
from examples.drop_in.llm_cutover_dropin import SupportTicketSchema, upstream_llm_teacher

# Start with a strict calibrated System 1 engine
engine = SystemOneEngine(SupportTicketSchema, model=starter_model, strict_mode=True)

# Wrap your LLM callable: uncertain inputs call upstream_llm_teacher and update System 1 online
fallback = BoundedReviewFallback(
    engine,
    review_fn=upstream_llm_teacher,
    max_reviews=50,
    auto_online_update=True,
)
```

### Step 2: Send Traffic Through `fallback.decide(prompt)`
- **Wave 1 (Novel Traffic)**: When unfamiliar jargon arrives (e.g., `"Kafka consumer lag"`, `"SIEM alert"`, `"SSO SAML login"`), System 1 detects ambiguity (`is_ambiguous=True`), calls `upstream_llm_teacher`, and applies an exact Rank-1 Sherman-Morrison RLS update (`preserve_calibration=True`) in `<1 ms`.
- **Wave 2 & Wave 3 (Recurring Traffic)**: System 1 now recognizes those patterns locally with high confidence — **LLM calls drop to `0`** and latency drops to **`~0.37 ms`**.

### Step 3: Save the Portable `.s1m` Skill
```python
engine.model.save("support_ticket_instinct.s1m")
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
  Wave 1: Joint Accuracy=6/6 (100%) | Served Locally=0% (LLM Calls=6) | Avg Latency=2.536 ms
  Wave 2: Joint Accuracy=6/6 (100%) | Served Locally=100% (LLM Calls=0) | Avg Latency=0.434 ms
  Wave 3: Joint Accuracy=6/6 (100%) | Served Locally=100% (LLM Calls=0) | Avg Latency=0.370 ms

Saved & Reloaded .s1m (45.1 KiB): queue='infrastructure', urgent=True (latency=1.239 ms, 0 LLM calls)
```
