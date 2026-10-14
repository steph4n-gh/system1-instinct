# System 1: Instinct — Drop-In Replacement Suite

Runnable 1-line drop-in replacements for common LLM routing, structured extraction, and prompt-guard libraries. Every drop-in runs locally in `< 1 ms` on CPU with **zero external embedding or LLM API calls**, starts with `0` training examples (or a compiled `.s1m` skill), and supports live online adaptation (`adapt_zero_shot()` and `learn_from_tier2()`).

| Target Library | Drop-In Module | Core Capability | Typical CPU Latency |
| :--- | :--- | :--- | :---: |
| **`semantic-router`** (`RouteLayer`) | [semantic_router_dropin.py](file:///Volumes/Storage/system-1-v2/examples/drop_in/semantic_router_dropin.py) | Utterance-anchored intent routing + OOD rejection (`None`) + unsupervised `adapt_zero_shot()` | **0.09–0.85 ms** |
| **`RouteLLM`** (`Controller`) | [routellm_dropin.py](file:///Volumes/Storage/system-1-v2/examples/drop_in/routellm_dropin.py) | 3-tier model cascade (`system1_local` / `fast_small_model` / `frontier_reasoning_model`) + complexity scoring | **0.10–0.90 ms** |
| **`Instructor` / `BAML`** (`response_model`) | [instructor_structured_dropin.py](file:///Volumes/Storage/system-1-v2/examples/drop_in/instructor_structured_dropin.py) | Simultaneous multi-field schema extraction (`ChoiceField`, `BooleanField`, `MultiChoiceField`, `ScoreField`) | **0.15–0.95 ms** |
| **`Llama-Prompt-Guard` / `Guardrails`** | [prompt_guard_dropin.py](file:///Volumes/Storage/system-1-v2/examples/drop_in/prompt_guard_dropin.py) | Negation-aware `#` n-gram prompt injection & exfiltration guard with `0%` `NotInject` benign-trigger FPR | **0.08–0.45 ms** |

## Running the Examples

```bash
python3 examples/drop_in/semantic_router_dropin.py
python3 examples/drop_in/routellm_dropin.py
python3 examples/drop_in/instructor_structured_dropin.py
python3 examples/drop_in/prompt_guard_dropin.py
```

## Running the Benchmark & Zero-Shot Adaptation Curve Suite

```bash
python3 benchmarks/quality/evaluate_drop_in_replacements.py
```
