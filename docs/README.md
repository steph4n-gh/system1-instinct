# System 1: Instinct Documentation

Current stable runtime: **1.0.0**.

**System 1: Instinct** is a sub-millisecond non-autoregressive decision runtime. Define a structured decision (`ChoiceField`, `BooleanField`, `MultiChoiceField`, `ScoreField`), start **zero-shot** from option descriptions or teach from examples, adapt online via `adapt_zero_shot` and `BoundedReviewFallback`, and deploy portable `.s1m` skills locally with zero LLM calls.

---

## 1. Real Drop-In Replacement Guides

Start with a real, runnable procedure in **[docs/drop_in/README.md](drop_in/README.md)**:

| Guide | What You Replace | Runnable Example |
| :--- | :--- | :--- |
| **[1. Semantic Router Drop-In](drop_in/01_semantic_router.md)** | `semantic_router.RouteLayer` | [`examples/drop_in/semantic_router_dropin.py`](../examples/drop_in/semantic_router_dropin.py) |
| **[2. RouteLLM Cascade Drop-In](drop_in/02_routellm_cascade.md)** | `routellm.controller.Controller` | [`examples/drop_in/routellm_dropin.py`](../examples/drop_in/routellm_dropin.py) |
| **[3. Instructor / BAML Structured Drop-In](drop_in/03_instructor_structured_outputs.md)** | `instructor.from_openai(...).create()` | [`examples/drop_in/instructor_structured_dropin.py`](../examples/drop_in/instructor_structured_dropin.py) |
| **[4. Prompt-Guard Security Drop-In](drop_in/04_prompt_guard_security.md)** | `Llama-Prompt-Guard` / `Guardrails` | [`examples/drop_in/prompt_guard_dropin.py`](../examples/drop_in/prompt_guard_dropin.py) |
| **[5. Live LLM Observe $\to$ `.s1m` Cutover](drop_in/05_llm_observe_and_cutover.md)** | Recurring OpenAI / Anthropic / Gemini calls | [`examples/drop_in/llm_cutover_dropin.py`](../examples/drop_in/llm_cutover_dropin.py) |
| **[6. `poke-env` Showdown Reflex Agent](drop_in/06_poke_env_showdown_agent.md)** | Slow LLM / static game bots | [`benchmarks/quality/evaluate_poke_env_showdown.py`](../benchmarks/quality/evaluate_poke_env_showdown.py) |

---

## 2. Core Guides & Integrations

| Guide | Purpose |
| :--- | :--- |
| [Quickstart](../README.md#quickstart) | Install System 1: Instinct and run your first zero-shot or taught skill |
| [Your first skill](guides/first_skill.md) | Editable lesson file, candidate checks, and local reuse |
| [Correct, compare, and adopt](guides/correcting_skills.md) | Multi-field `TeachingSession` lifecycle, qualification, and rollback |
| [Teaching guide](guides/training_experts.md) | Label formats, compilation flags (`--hierarchical`, `--mondrian`), and online updates |
| [Teacher observation (`TypeSafeClient`)](typesafe.md) | Observe a live teacher and automatically cut over to local `.s1m` execution |
| [Application integrations](integrations.md) | Connect a skill to LangChain, MCP, FastAPI/ASGI, or gRPC (with live hot-reload) |
| [Deterministic policy guard](guides/policy_guard.md) | Enforce explicit tool permissions, Ed25519 receipts, and SHA-256 hash-chained ledgers |
| [Deployment boundaries](deployment.md) | Security boundaries, resource limits, and production checklists |
| [Observability](observability/README.md) | Prometheus metrics and OpenTelemetry tracing |
| [Example catalog](../examples/README.md) | Complete index of runnable examples and interactive labs |

---

## 3. Architecture, Mathematics & Benchmarks

| Document | Scope |
| :--- | :--- |
| [Fresh Traffic with a Fallible Teacher](../benchmarks/heldout_traffic/REPORT.md) | HWU64, 10% noisy lessons, matched 1% local error budget, exact cache/classifier/Instinct comparison, frozen gates, binary `.s1m` hybrid fix, and independent verification |
| [6-Category Benchmark Gauntlet](../benchmarks/quality/instinct_gauntlet_report.md) | `BANKING77` (32/77), `CLINC150-OOS`, `NotInject`, authored drift probes, 4-field extraction, and 0-shot routing |
| [Drop-In & Adaptation Curve Report](../benchmarks/quality/drop_in_and_adaptation_report.md) | `0`-shot $\to$ `adapt_zero_shot` $\to$ `1`-shot $\to$ `3`-shot $\to$ `5`-shot $\to$ `.s1m` curves on `BANKING77` & `CLINC150` |
| [`poke-env` Showdown Benchmark Report](../benchmarks/quality/poke_env_showdown_report.md) | Live `gen9randombattle` Showdown results against `RandomPlayer`, `MaxBasePowerPlayer`, and `SimpleHeuristicsPlayer` |
| [Architecture Reference](architecture/technical_specification.md) | Runtime components, `.s1m` v2 binary format, and module map |
| [Technical Brief](paper/system1_technical_brief.md) | Concise overview of the runtime and empirical results |
| [Whitepaper](paper/system1_whitepaper.md) | Full design, mathematical formulation, and workload evaluations |
| [Mathematical Notes](paper/conformal_gating.md) | Split-conformal prediction, Mondrian calibration, dual OOD gating, and rank-1 RLS updates |
| [Benchmark Suite Index](../benchmarks/quality/README.md) | All reproducible benchmark runners and datasets |
