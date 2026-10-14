# Drop-In Guide 2: Replace `RouteLLM` (Model Cascade Controller)

Replace `routellm.controller.Controller` with [`SystemOneRouteController`](../../examples/drop_in/routellm_dropin.py) to dispatch incoming prompts across **three cost/capability tiers** in **~1.4 ms** using a joint multi-field schema (`ChoiceField` tier + `BooleanField` deep-reasoning flag + `ScoreField` complexity score).

---

## 1. Before vs. After

### Before (`routellm`)
```python
from routellm.controller import Controller

# Requires external router weights or API calls to score prompt complexity
controller = Controller(
    routers=["mf"],
    strong_model="gpt-5-reasoning",
    weak_model="gpt-4o-mini",
)
```

### After (`System 1: Instinct`)
```python
from examples.drop_in import SystemOneRouteController

controller = SystemOneRouteController(
    strong_model="gpt-5-reasoning",
    weak_model="gpt-4o-mini",
    local_model="system1-instinct-local",
)
decision = controller.route("Prove why a distributed consensus cluster deadlocks under race condition")
print(decision.selected_tier, decision.selected_model, decision.complexity_score)
# -> "frontier_reasoning_model", "gpt-5-reasoning", 0.75
```

---

## 2. Step-by-Step Procedure

### Step 1: Initialize the 3-Tier Cascade Controller
`SystemOneRouteController` evaluates three fields simultaneously in a single non-autoregressive pass:
1. `target_tier`: `"system1_local"` (FAQ/status/triage), `"fast_small_model"` (drafting/summarization), or `"frontier_reasoning_model"` (formal proofs/distributed systems debugging).
2. `requires_deep_reasoning`: Calibrated `BooleanField`.
3. `complexity_score`: Continuous `0.0–1.0` `ScoreField`.

### Step 2: Teach Custom Routing Rules Online in 1 Shot
If a domain-specific task (e.g., `"Write a regex for RFC-5322 email validation"`) should always go to `fast_small_model` or `frontier_reasoning_model`, update the router in place without retraining:

```python
controller.engine.learn_from_tier2(
    "Verify the Coq proof for linearizability in our Paxos state machine",
    {"target_tier": "frontier_reasoning_model", "requires_deep_reasoning": True, "complexity_score": 0.95},
    preserve_calibration=True,
)
```

---

## 3. Run It & See the Real Result

Run the standalone script from the repository root:

```bash
python examples/drop_in/routellm_dropin.py
```

### Real Output
```text
=== System 1: Instinct — RouteLLM Drop-In Replacement Demo ===
  [system1_local            -> system1-instinct-local] (expected=system1_local, complexity=0.26, latency=1.661 ms)
  [fast_small_model         -> gpt-4o-mini           ] (expected=fast_small_model, complexity=0.47, latency=1.470 ms)
  [frontier_reasoning_model -> gpt-5-reasoning       ] (expected=frontier_reasoning_model, complexity=0.75, latency=2.043 ms)
```

### Benchmark Scorecard (`evaluate_drop_in_replacements.py`)
- **0-Shot Tier Routing Accuracy**: **`100.0%`**
- **Unseen Edge-Case Adaptation**: **`50.0%` $\to$ `100.0%`** after 1-shot online RLS update
- **Median Latency (P50)**: **`1.468 ms`**
