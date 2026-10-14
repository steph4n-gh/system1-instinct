# System 1: Instinct — Step-by-Step Drop-In Replacement Guides

Every guide below follows a **real, reproducible procedure** you can run in one command, showing:
1. **Before**: The standard external library or LLM API call (`Semantic Router`, `RouteLLM`, `Instructor`, `Prompt-Guard`, `OpenAI/Anthropic`, or `poke-env`).
2. **Procedure**: How to drop in **System 1: Instinct** (`0`-shot cold start $\to$ online adaptation $\to$ portable `.s1m` skill).
3. **Procedural Verification**: Guides 1–5 use compact, inspectable prompt sets (`8–16` prompts) so you can trace every decision, OOD abstention, and online update (`adapt_zero_shot` / `learn_from_tier2`) step by step. For external public dataset benchmarks (`BANKING77`, `CLINC150-OOS`, `all-MiniLM-L6-v2` / `bge-small-en-v1.5`, and `poke-env` Showdown), see [External Public Benchmarks](../../benchmarks/quality/drop_in_and_adaptation_report.md).

| # | Drop-In Guide | Replaces | Runnable Script | Procedural Verification | P50 Latency |
| :-: | :--- | :--- | :--- | :--- | :---: |
| **1** | [Semantic Router Drop-In](01_semantic_router.md) | `semantic_router.RouteLayer` | [`semantic_router_dropin.py`](../../examples/drop_in/semantic_router_dropin.py) | **In-Domain** · **OOD Reject** · **0% $\to$ 100%** Unseen Jargon (`adapt_zero_shot`) | **1.36 ms** |
| **2** | [RouteLLM Cascade Drop-In](02_routellm_cascade.md) | `routellm.controller.Controller` | [`routellm_dropin.py`](../../examples/drop_in/routellm_dropin.py) | **3-Tier Cascade Routing** (`system1_local` vs `fast` vs `frontier`) | **1.47 ms** |
| **3** | [Instructor / BAML Structured Drop-In](03_instructor_structured_outputs.md) | `instructor.from_openai(...).create()` | [`instructor_structured_dropin.py`](../../examples/drop_in/instructor_structured_dropin.py) | **4-Field Joint Extraction** (`Choice` + `Bool` + `Multi` + `Score`) | **2.73 ms** |
| **4** | [Prompt-Guard Security Drop-In](04_prompt_guard_security.md) | `Llama-Prompt-Guard` / `Guardrails` | [`prompt_guard_dropin.py`](../../examples/drop_in/prompt_guard_dropin.py) | **0.0% `NotInject` Benign FPR** · **Attack Recall** | **0.81 ms** |
| **5** | [LLM Observe $\to$ Local `.s1m` Cutover](05_llm_observe_and_cutover.md) | Recurring OpenAI / Anthropic / Gemini calls | [`llm_cutover_dropin.py`](../../examples/drop_in/llm_cutover_dropin.py) | **Wave 1: 6 LLM calls $\to$ Wave 2: 0 LLM calls** (**100% Local** at **0.37 ms**) | **0.37 ms** |
| **6** | [`poke-env` Showdown Reflex Agent](06_poke_env_showdown_agent.md) | Slow LLM / static rule game bots | [`evaluate_poke_env_showdown.py`](../../benchmarks/quality/evaluate_poke_env_showdown.py) | **100% vs `Random`** · **98% vs `MaxBasePower`** · **62% vs `SimpleHeuristics`** | **0.093 ms** |

---

## Run All Drop-In Smoke Checks & Public Dataset Adaptation Curves

```bash
python benchmarks/quality/evaluate_drop_in_replacements.py
```

See the full public dataset adaptation curves and neural embedding baseline comparison in [drop_in_and_adaptation_report.md](../../benchmarks/quality/drop_in_and_adaptation_report.md).
