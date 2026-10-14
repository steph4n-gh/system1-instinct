# Drop-In Guide 1: Replace `Semantic Router` (`RouteLayer`)

Replace `semantic_router.RouteLayer` + OpenAI/Cohere embedding calls with [`InstinctRouteLayer`](../../examples/drop_in/semantic_router_dropin.py) — running **100% locally** in **~1.3 ms** with **zero API keys**, **out-of-distribution (OOD) rejection**, and **unsupervised zero-shot drift adaptation**.

---

## 1. Before vs. After

### Before (`semantic-router` + external embedding API)
```python
from semantic_router import Route, RouteLayer
from semantic_router.encoders import OpenAIEncoder

routes = [
    Route(name="billing_and_refunds", utterances=["refund my duplicate charge on billing invoice"]),
    Route(name="technical_support", utterances=["production API returning 502 bad gateway"]),
]
# Requires OPENAI_API_KEY and a network round-trip on every query
rl = RouteLayer(encoder=OpenAIEncoder(), routes=routes)
route_choice = rl("Please refund the duplicate charge on my billing invoice")
```

### After (`System 1: Instinct` — 0 API keys, ~1.3 ms local)
```python
from examples.drop_in import InstinctRouteLayer, SemanticRoute

routes = [
    SemanticRoute(
        name="billing_and_refunds",
        utterances=[
            "refund my duplicate charge on billing invoice",
            "cancel annual subscription and send payment receipt",
        ],
    ),
    SemanticRoute(
        name="technical_support",
        utterances=[
            "production API returning 502 bad gateway and database timeout",
            "webhook delivery failing with stack trace exception and crash",
        ],
    ),
]
rl = InstinctRouteLayer(routes, confidence_threshold=0.38)
route_choice = rl("Please refund the duplicate charge on my billing invoice")
print(route_choice.name, route_choice.latency_ms)  # -> "billing_and_refunds", ~1.1 ms
```

---

## 2. Step-by-Step Procedure

### Step 1: Route In-Domain Queries & Reject Unrelated (OOD) Traffic
Pass `confidence_threshold=0.38` to `InstinctRouteLayer`. In-domain queries return their matched route name; unrelated queries (like recipes or weather) return `name=None`:

```python
assert rl("Please refund the duplicate charge on my billing invoice").name == "billing_and_refunds"
assert rl("What is the best sourdough pizza recipe in Naples?").name is None
```

### Step 2: Adapt to Unseen Jargon with Zero Labels (`adapt_unlabeled`)
In production, users start using new internal project codes or error tags (e.g., `SEC_ALPHA_9`, `OPS_SIGMA_7`) that were not in your original `utterances`. Feed unlabeled production logs where those new terms co-occur with known words into `rl.adapt_unlabeled(...)`:

```python
rl.adapt_unlabeled([
    "Suspicious login attempt and leaked API key credential compromise SEC_ALPHA_9",
    "Production API returning 502 bad gateway and database timeout OPS_SIGMA_7",
])

# Now queries containing ONLY the new internal codeword route accurately!
res = rl("Cluster alert on node OPS_SIGMA_7 experiencing service degradation")
assert res.name == "technical_support"
```

---

## 3. Run It & See the Real Result

Run the standalone script from the repository root:

```bash
python examples/drop_in/semantic_router_dropin.py
```

### Real Output
```text
=== System 1: Instinct — Semantic Router Drop-In Replacement Demo ===
  [0-Shot] route=billing_and_refunds  | conf=0.49 | latency=1.182 ms | query='Please refund the duplicate charge on my billing invoice'
  [0-Shot] route=technical_support    | conf=0.51 | latency=0.842 ms | query='Production API returning 502 bad gateway and database timeout'
  [0-Shot] route=None                 | conf=0.31 | latency=0.794 ms | query='What is the best sourdough pizza recipe in Naples?'
  [Adapted] route=security_escalation | conf=0.45 | latency=0.815 ms | query='Alert triggered for security code SEC_ALPHA_9 on production tenant'
  [Adapted] route=technical_support   | conf=0.47 | latency=0.788 ms | query='Cluster alert on node OPS_SIGMA_7 experiencing service degradation'
```

### Benchmark Scorecard (`evaluate_drop_in_replacements.py`)
- **In-Domain Accuracy (0-Shot)**: **`100.0%`** (`12/12`)
- **Out-of-Distribution Rejection**: **`100.0%`** (`4/4` returned `None`)
- **Unseen Jargon Accuracy**: **`0.0%` (Before) $\to$ `100.0%` (After `adapt_zero_shot`, 0 labels)**
- **Median Latency (P50)**: **`1.363 ms`**
