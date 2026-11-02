# How to wrap a teacher callable and cut over to a `.s1m` skill

To offload a classification function to local CPU execution, wrap the callable with [`BoundedReviewFallback`](../../examples/drop_in/llm_cutover_dropin.py). The checked-in script uses a keyword stub (`upstream_llm_teacher`), not a live OpenAI, Anthropic, or Gemini client. Live-LLM-teacher evaluation is planned. `SystemOneEngine` escalates uncertain requests (`is_ambiguous=True`) to that callable, updates its weights online, and saves a portable `.s1m` skill. Update time is host-specific; see the [README honest limits](../../README.md#honest-limits).

---

## 1. Wrap your LLM function with `BoundedReviewFallback`

Instantiate `SystemOneEngine` with `strict_mode=True` and pass your LLM callable as `review_fn` to `BoundedReviewFallback`:

```python
from system1 import BoundedReviewFallback, SystemOneEngine
from examples.drop_in.llm_cutover_dropin import SupportTicketSchema, upstream_llm_teacher

# Start with a strict SystemOneEngine at 0 examples or with starter examples.
engine = SystemOneEngine(SupportTicketSchema, strict_mode=True, use_cache=False)

# Wrap a teacher callable. In this example upstream_llm_teacher is a keyword stub, not a live LLM.
fallback = BoundedReviewFallback(
    engine,
    review_fn=upstream_llm_teacher,
    max_reviews=50,
    auto_online_update=True,
)
```

## 2. Route live traffic through `fallback.decide(prompt)`

Call `fallback.decide(prompt)` for each incoming request, or call `engine.learn(prompt, labels)` and `engine.learn_from_outcome(prompt, labels, reward=1.0)` directly:

- **Wave 1: novel traffic.** When unfamiliar jargon such as `"Kafka consumer lag"`, `"SIEM alert"`, or `"SSO SAML login"` arrives, `engine.decide(...)` returns `is_ambiguous=True`. `BoundedReviewFallback` calls `upstream_llm_teacher` and applies a rank-1 Sherman-Morrison RLS update (`preserve_calibration=True`).
- **Wave 2: recurring traffic.** When those patterns recur, `SystemOneEngine` can resolve them locally with `0` calls to the stub.
- **Wave 3: unseen phrasings.** New phrasings of those learned patterns, such as `"Redis cluster crash and Kafka consumer lag triggering 502 gateway timeouts"`, are scored locally. Each wave in the script is 6 requests. A 6/6 result on that toy set is not a published accuracy claim.

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

One run on the shared 4-vCPU Intel Xeon VM from the [README honest limits](../../README.md#honest-limits) printed:

```text
=== System 1: Instinct — LLM Observe -> Online RLS -> Local Cutover Demo ===
  Wave 1 (Initial Stream + Novel Jargon): Joint Accuracy=6/6 (100%) | Served Locally=83% (LLM Calls=1) | Avg Latency=2.002 ms
  Wave 2 (Repeat Stream): Joint Accuracy=6/6 (100%) | Served Locally=100% (LLM Calls=0) | Avg Latency=1.214 ms
  Wave 3 (Unseen Generalized Phrasings): Joint Accuracy=6/6 (100%) | Served Locally=100% (LLM Calls=0) | Avg Latency=1.433 ms

Saved & Reloaded .s1m (56.0 KiB, strict_mode=True): queue='infrastructure', urgent=True, is_ambiguous=False (latency=2.265 ms, 0 LLM calls)
```

Those times and the 56.0 KiB size are from that run. The teacher calls are the keyword stub. Live-LLM-teacher evaluation is planned.
