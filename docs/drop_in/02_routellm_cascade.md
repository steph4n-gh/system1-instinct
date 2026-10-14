# Drop-In Guide 2: Replacing `RouteLLM` & Routing Inside `langchain-core` LCEL

This guide shows how to replace `routellm.controller.Controller` or an LLM router inside a real **`langchain_core` (`RunnableLambda`) LCEL pipeline**, evaluated across **`256` held-out `BANKING77` customer tickets (`32` intents, clean + character-typo stress)** against `sentence-transformers/all-MiniLM-L6-v2` (`91.1 MB` ONNX).

---

## 1. Before (`RouteLLM` / Standard LangChain LLM Router)

```python
from routellm.controller import Controller

# Requires external API keys or 80-500 MB router checkpoints:
controller = Controller(
    routers=["mf"],
    strong_model="gpt-5-reasoning",
    weak_model="gpt-4o-mini",
)
response = controller.chat.completions.create(
    model="router-mf-0.11593",
    messages=[{"role": "user", "content": "Simple status check on my account balance"}],
)
```

---

## 2. Drop-In Replacement (`SystemOneRouteController` & `langchain_core` LCEL)

```python
from langchain_core.runnables import RunnableLambda
from examples.drop_in.routellm_dropin import SystemOneRouteController

controller = SystemOneRouteController(
    strong_model="gpt-5-reasoning",
    weak_model="gpt-4o-mini",
    local_model="system1-instinct-local",
)

# Drop directly into any LangChain LCEL pipeline:
lcel_router = RunnableLambda(lambda prompt: controller.route(prompt))
decision = lcel_router.invoke("Debug this subtle concurrency deadlock in our distributed cluster")
print(decision.selected_tier, decision.selected_model, decision.latency_ms)
# -> frontier_reasoning_model  gpt-5-reasoning  1.47 ms

# Teach overrides online in <1 ms:
controller.record_feedback("Write a formal proof for Raft leader election safety", "frontier_reasoning_model")
```

---

## 3. Run the Live `langchain-core` `BANKING77` Benchmark (`256` Held-Out Tickets)

```bash
python examples/drop_in/routellm_dropin.py
```

### Real Output

```text
=== 1. RouteLLM 3-Tier Cascade Drop-In (`SystemOneRouteController`) ===
  [system1_local            -> system1-instinct-local] (expected=system1_local, complexity=0.26, latency=2.328 ms)
  [fast_small_model         -> gpt-4o-mini           ] (expected=fast_small_model, complexity=0.47, latency=1.893 ms)
  [frontier_reasoning_model -> gpt-5-reasoning       ] (expected=frontier_reasoning_model, complexity=0.75, latency=4.018 ms)

=== 2. Real `langchain_core` LCEL Pipeline on BANKING77 (32 Intents, 256 Held-Out Tickets) ===
  [Classical Unigram TF-IDF (85 KiB)]         Clean=77.3% | Typo=66.0% | P50=0.18 ms
  [all-MiniLM-L6-v2 (91.1 MB ONNX, 22M)]      Clean=91.0% | Typo=70.7% (-20.3% drop!) | P50=1.32 ms
  [System 1: Instinct Static (120 KiB .s1m)]  Clean=87.9% | Typo=74.2% (+3.5% vs MiniLM!) | P50=0.22 ms
  [System 1: Instinct + Online Flywheel]      Clean=91.4% (surpasses MiniLM after 48/256 teacher reviews!)
```
