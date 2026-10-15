# How to wrap a live LLM call and cut over to a `.s1m` skill

To offload an OpenAI, Anthropic, or Gemini classification function to local CPU execution, wrap it with [`BoundedReviewFallback`](../../examples/drop_in/llm_cutover_dropin.py). `SystemOneEngine` escalates uncertain requests (`is_ambiguous=True`) to your LLM callable, updates its weights online in `<1 ms`, and saves a portable `.s1m` skill.

---

## 1. Wrap your LLM function with `BoundedReviewFallback`

Instantiate `SystemOneEngine` with `strict_mode=True` and pass your LLM callable as `review_fn` to `BoundedReviewFallback`:

```python
from system1 import BoundedReviewFallback, SystemOneEngine
from examples.drop_in.llm_cutover_dropin import SupportTicketSchema, upstream_llm_teacher

# Start with a strict SystemOneEngine at 0 examples or with starter examples.
engine = SystemOneEngine(SupportTicketSchema, strict_mode=True, use_cache=False)

# Wrap your LLM callable so uncertain inputs call upstream_llm_teacher and update weights online.
fallback = BoundedReviewFallback(
    engine,
    review_fn=upstream_llm_teacher,
    max_reviews=50,
    auto_online_update=True,
)
```

## 2. Route live traffic through `fallback.decide(prompt)`

Call `fallback.decide(prompt)` for each incoming request, or call `engine.learn(prompt, labels)` and `engine.learn_from_outcome(prompt, labels, reward=1.0)` directly:

- **Wave 1: novel traffic.** When unfamiliar jargon such as `"Kafka consumer lag"`, `"SIEM alert"`, or `"SSO SAML login"` arrives, `engine.decide(...)` returns `is_ambiguous=True`. `BoundedReviewFallback` calls `upstream_llm_teacher` and applies a rank-1 Sherman-Morrison RLS update (`preserve_calibration=True`) in `<1 ms`.
- **Wave 2: recurring traffic.** When those patterns recur, `SystemOneEngine` resolves them locally in `~0.9 ms` with `0` LLM calls.
- **Wave 3: unseen phrasings.** When new phrasings of those learned patterns arrive, such as `"Redis cluster crash and Kafka consumer lag triggering 502 gateway timeouts"`, `SystemOneEngine` resolves them locally with `100%` joint accuracy and `0` LLM calls.

## 3. Save and reload the `.s1m` skill

To persist the online-updated engine to disk and reload it in another process without an LLM client, call `engine.save(...)` and `SystemOneEngine.load(...)`:

```python
# Save the engine state, including strict_mode=True and online-learned weights.
engine.save("support_ticket_instinct.s1m")

# Reload the skill with zero external dependencies.
reloaded_engine = SystemOneEngine.load("support_ticket_instinct.s1m")
decision = reloaded_engine.decide(
    "Redis cluster crash and Kafka consumer lag triggering 502 gateway timeouts",
    record_receipt=False,
)
assert not decision.is_ambiguous
```

---

## 4. Run the standalone three-wave cutover script

To run the complete three-wave simulation from the repository root, execute:

```bash
python examples/drop_in/llm_cutover_dropin.py
```

The script prints:

```text
=== System 1: Instinct — LLM Observe -> Online RLS -> Local Cutover Demo ===
  Wave 1 (Initial Stream + Novel Jargon): Joint Accuracy=6/6 (100%) | Served Locally=83% (LLM Calls=1) | Avg Latency=1.354 ms
  Wave 2 (Repeat Stream): Joint Accuracy=6/6 (100%) | Served Locally=100% (LLM Calls=0) | Avg Latency=0.888 ms
  Wave 3 (Unseen Generalized Phrasings): Joint Accuracy=6/6 (100%) | Served Locally=100% (LLM Calls=0) | Avg Latency=0.971 ms

Saved & Reloaded .s1m (12.0 KiB, strict_mode=True): queue='infrastructure', urgent=True, is_ambiguous=False (latency=1.793 ms, 0 LLM calls)
```
