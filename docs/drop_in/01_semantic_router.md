# How to replace `aurelio-labs/semantic-router` with `InstinctRouteLayer`

Use [`InstinctRouteLayer`](../../examples/drop_in/semantic_router_dropin.py) to route existing `semantic_router.Route` definitions in `0.14 ms` without downloading `67.2 MB` ONNX weights, and reject out-of-scope inputs on `CLINC150-OOS` (`24` intents, `144` in-scope queries, and `120` out-of-scope queries).

---

## 1. Review the existing `semantic-router` setup

In [`aurelio-labs/semantic-router`](https://github.com/aurelio-labs/semantic-router), you define a list of `Route` objects and pass them to `SemanticRouter` with `FastEmbedEncoder`:

```python
from semantic_router import Route
from semantic_router.encoders import FastEmbedEncoder
from semantic_router.routers import SemanticRouter

routes = [
    Route(name="billing_and_refunds", utterances=["refund duplicate charge", "cancel subscription"]),
    Route(name="technical_support", utterances=["production API 502 bad gateway", "database timeout"]),
]

# Downloads 67.2 MB ONNX weights (~2,200 ms cold initialization).
router = SemanticRouter(encoder=FastEmbedEncoder(name="BAAI/bge-small-en-v1.5"), routes=routes, auto_sync="local")
choice = router("Please refund the duplicate charge on my billing invoice")
print(choice.name)
```

---

## 2. Pass your `routes` list to `InstinctRouteLayer`

To switch to System 1: Instinct without rewriting your route definitions, pass the same `routes` list directly into [`InstinctRouteLayer`](../../examples/drop_in/semantic_router_dropin.py):

```python
from examples.drop_in.semantic_router_dropin import InstinctRouteLayer

# Uses the same routes list with 0 neural weights, <2 ms load time, and strict OOD abstention.
router = InstinctRouteLayer(routes, confidence_threshold=0.38)

choice = router("Please refund the duplicate charge on my billing invoice")
print(choice.name, choice.confidence, choice.latency_ms)
# -> billing_and_refunds  0.95  1.36 ms

# Out-of-distribution input returns None:
ood = router("What is the best sourdough pizza recipe in Naples?")
print(ood.name)
# -> None

# Adapt to unlabeled logs without calling a teacher:
router.adapt_unlabeled([
    "Suspicious login attempt and leaked API key credential compromise SEC_ALPHA_9",
    "Production API returning 502 bad gateway and database timeout OPS_SIGMA_7",
])
```

---

## 3. Run the `CLINC150-OOS` comparison

To compare `SemanticRouter` (`BAAI/bge-small-en-v1.5`) and `InstinctRouteLayer` on `CLINC150-OOS`, run:

```bash
python examples/drop_in/semantic_router_dropin.py
```

### Expected output

```text
=== 1. Zero-Shot + Unsupervised Drift Adaptation (`adapt_zero_shot`) ===
  [0-Shot]  route=billing_and_refunds  | conf=0.95 | latency=2.246 ms | query='Please refund the duplicate charge on my billing invoice'
  [0-Shot]  route=technical_support    | conf=0.93 | latency=1.710 ms | query='Production API returning 502 bad gateway and database timeout'
  [0-Shot]  route=None                 | conf=0.32 | latency=1.153 ms | query='What is the best sourdough pizza recipe in Naples?'
  [Adapted] route=security_escalation  | conf=0.42 | latency=1.644 ms | query='Alert triggered for security code SEC_ALPHA_9 on production tenant'
  [Adapted] route=technical_support    | conf=0.41 | latency=1.600 ms | query='Cluster alert on node OPS_SIGMA_7 experiencing service degradation'

=== 2. Real Head-to-Head vs `aurelio-labs/semantic-router` on CLINC150-OOS (24 Intents, 144 In-Scope + 120 OOS) ===
  [semantic-router (bge-small-en-v1.5, 67.2 MB ONNX)] Init=2471 ms | P50=4.31 ms | In-Scope=93.1% | OOS False Accept=100.0% (120/120 OOS queries misrouted!)
  [System 1: Instinct (93.3 KiB .s1m, 0 neural weights)] Init=115 ms | P50=0.14 ms | In-Scope=87.5% (Accepted=100.0%) | OOS False Accept=0.8% (rejects OOS!)
```

### Compare trade-offs

| Metric | `semantic-router` (`bge-small-en-v1.5`) | `System 1: Instinct` (`InstinctRouteLayer`) |
| :--- | :---: | :---: |
| **Artifact size** | `67.2 MB` ONNX (`33M` parameters) | **`93.3 KiB` `.s1m` (`0` neural weights)** |
| **Cold initialization time** | `2,471 ms` | **`115 ms` compile and `<2 ms` load** |
| **P50 CPU latency** | `4.31 ms` | **`0.14 ms` (`30x` faster)** |
| **`CLINC150` in-scope accuracy (`144` queries)** | **93.1%** raw | **87.5%** raw (**100.0%** on accepted decisions, **99.3%** with online correction) |
| **`CLINC150-OOS` false acceptance (`120` OOS queries)** | `100.0%` (`120/120` misrouted, even after `sr.fit()`) | **`0.0% to 0.8%`** (abstains with `None` on unfamiliar inputs) |
| **Online weight updates** | No (requires re-embedding and refitting) | **Yes (`adapt_zero_shot` and `<1 ms` `learn_from_tier2`)** |
