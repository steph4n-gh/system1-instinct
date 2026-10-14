# Example Catalog

Run examples from a repository checkout with **System 1: Instinct** installed.

## 1. Interactive Showcase & Real Open-Source Drop-Ins

| Example | What It Demonstrates |
| :--- | :--- |
| **[showcase_break_and_teach.py](showcase_break_and_teach.py)** | **"Break-Me & Teach-Me" Live Showcase & REPL**: Compiles a `23.7 KiB` `.s1m` skill from 6 examples/option, runs side-by-side against `BAAI/bge-small-en-v1.5` (`67.2 MB` ONNX) on negation flips, OOS garbage, and mobile typos, and teaches a brand-new codeword in 1 line via `engine.learn(...)`. Pass `--interactive` to test your own prompts live. |
| **[drop_in/instructor_structured_dropin.py](drop_in/instructor_structured_dropin.py)** | Real `FastAPI` + `Instructor` / Pydantic 4-field structured triage endpoint (`ChoiceField` + `BooleanField` + `MultiChoiceField` + `ScoreField`) converging to `0%` LLM escalation in `0.45 ms` ([guide](../docs/drop_in/03_instructor_structured_outputs.md)). |
| **[drop_in/semantic_router_dropin.py](drop_in/semantic_router_dropin.py)** | Drop-in `InstinctRouteLayer` accepting real `semantic_router.Route` objects (`0.14 ms` CPU latency, `0.0%` OOS false acceptance) ([guide](../docs/drop_in/01_semantic_router.md)). |
| **[drop_in/routellm_dropin.py](drop_in/routellm_dropin.py)** | Drop-in `langchain-core` `RunnableLambda` cascade router with character-typo resilience and online learning ([guide](../docs/drop_in/02_routellm_cascade.md)). |
| **[drop_in/prompt_guard_dropin.py](drop_in/prompt_guard_dropin.py)** | Sub-millisecond prompt-injection & negation-polarity gate (`0.0%` `NotInject` benign false-positive rate) ([guide](../docs/drop_in/04_prompt_guard_security.md)). |
| **[drop_in/llm_cutover_dropin.py](drop_in/llm_cutover_dropin.py)** | End-to-end 3-wave progression from `0` examples $\to$ online learning $\to$ compiled `.s1m` cutover with `0` teacher calls ([guide](../docs/drop_in/05_llm_observe_and_cutover.md)). |

```bash
python examples/showcase_break_and_teach.py
python examples/drop_in/instructor_structured_dropin.py
python examples/drop_in/semantic_router_dropin.py
python examples/drop_in/routellm_dropin.py
python examples/drop_in/prompt_guard_dropin.py
python examples/drop_in/llm_cutover_dropin.py
```

---

## 2. Core Skill Teaching & Verification Workflows

The core lifecycle is **teach → check a candidate → adopt → run locally → correct** ([correction guide](../docs/guides/correcting_skills.md), [teaching guide](../docs/guides/training_experts.md)).

| Example | What It Demonstrates |
| :--- | :--- |
| **[teach_skill.py](teach_skill.py)** | Minimal 8-lesson skill compilation, reload, and CLI inspection ([beginner walkthrough](../docs/guides/first_skill.md)). |
| **[support_triage.py](support_triage.py)** | Taught department routing, split-conformal calibration, and saved/reloaded `.s1m` evaluation. |
| **[banking_support.py](banking_support.py)** | Three `BANKING77` intents evaluated using original public labels. |
| **[model_routing.py](model_routing.py)** | Stated model-complexity routing policy (`local_small` / `standard_chat` / `frontier_reasoning`). |
| **[agent_guard.py](agent_guard.py)** | Operation classification paired with explicit `PolicyEngine` permissions and Ed25519 audit receipts. |
| **[observe_routing.py](observe_routing.py)** | Automatic teacher observation, statistical promotion gate, and offline `.s1m` reuse. |
| **[train_expert.py](train_expert.py)** | Programmatic compilation, online `learn()` updates, and multi-head composition. |
| **[teaching_by_doing/](teaching_by_doing/README.md)** | Browser workspace for interactive lesson editing, candidate assessment, and explicit adoption. |
| **[inbox_zero/](inbox_zero/README.md)** | Local HTTP classification pilot for Inbox Zero's 7 email categories. |
| **[teaching/](teaching/README.md)** | Datasets and recorded evaluation outputs for the core teaching scripts. |

```bash
python examples/teach_skill.py
python examples/support_triage.py
python examples/observe_routing.py
python examples/banking_support.py
```
