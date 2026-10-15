# System 1: Instinct documentation

Current stable runtime: **1.0.0**.

**System 1: Instinct** is a sub-millisecond non-autoregressive decision runtime. Each skill uses a `DecisionSchema` (`ChoiceField`, `BooleanField`, `MultiChoiceField`, and `ScoreField`), runs zero-shot from option descriptions or fits weights from labeled examples, updates online via `adapt_zero_shot` and `BoundedReviewFallback`, and serializes to a `.s1m` artifact.

---

## 1. Framework drop-in replacement guides

The index in [docs/drop_in/README.md](drop_in/README.md) covers drop-in integrations for external Python libraries:

| Guide | Target interface | Runnable script |
| :--- | :--- | :--- |
| [1. Semantic Router drop-in](drop_in/01_semantic_router.md) | `semantic_router.RouteLayer` | [`examples/drop_in/semantic_router_dropin.py`](../examples/drop_in/semantic_router_dropin.py) |
| [2. RouteLLM cascade drop-in](drop_in/02_routellm_cascade.md) | `routellm.controller.Controller` | [`examples/drop_in/routellm_dropin.py`](../examples/drop_in/routellm_dropin.py) |
| [3. Instructor structured output drop-in](drop_in/03_instructor_structured_outputs.md) | `instructor.from_openai(...).create()` | [`examples/drop_in/instructor_structured_dropin.py`](../examples/drop_in/instructor_structured_dropin.py) |
| [4. Prompt-Guard security drop-in](drop_in/04_prompt_guard_security.md) | `Llama-Prompt-Guard` and `Guardrails` | [`examples/drop_in/prompt_guard_dropin.py`](../examples/drop_in/prompt_guard_dropin.py) |
| [5. Live LLM observation and `.s1m` cutover](drop_in/05_llm_observe_and_cutover.md) | OpenAI, Anthropic, and Gemini calls | [`examples/drop_in/llm_cutover_dropin.py`](../examples/drop_in/llm_cutover_dropin.py) |
| [6. `poke-env` Showdown reflex agent](drop_in/06_poke_env_showdown_agent.md) | `poke_env.player.Player` battle bots | [`benchmarks/quality/evaluate_poke_env_showdown.py`](../benchmarks/quality/evaluate_poke_env_showdown.py) |

---

## 2. Core guides and integrations

| Guide | Scope |
| :--- | :--- |
| [Quickstart](../README.md#quickstart) | Environment setup and runnable quickstart commands |
| [Teach your first skill](guides/first_skill.md) | Tutorial for building a two-choice routing skill from an editable JSON file |
| [Correct, compare, and adopt a skill](guides/correcting_skills.md) | `TeachingSession` lifecycle, candidate assessment, qualification, and rollback |
| [Teach and compile a decision skill](guides/training_experts.md) | Label formats, `SystemOneCompiler`, `TfidfProjector`, and online updates |
| [Teacher observation (`TypeSafeClient`)](typesafe.md) | Reference for `system1.compat.typesafe.Client`, promotion policies, and `.s1m` export |
| [Application integrations](integrations.md) | Reference for LangChain, MCP, FastAPI, ASGI, and gRPC adapters |
| [Add an explicit tool policy](guides/policy_guard.md) | How to configure `SystemOneGuard`, `PolicyEngine`, Ed25519 receipts, and `ActionLedger` |
| [Deployment boundaries](deployment.md) | Security boundaries, `.s1m` loader limits, and production checklists |
| [Observability](observability/README.md) | Prometheus metrics and OpenTelemetry tracing |
| [Example catalog](../examples/README.md) | Index of runnable scripts and interactive labs |

---

## 3. Architecture, mathematics, and benchmarks

| Document | Scope |
| :--- | :--- |
| [Fresh traffic with a fallible teacher](../benchmarks/heldout_traffic/REPORT.md) | `HWU64` with `10%` noisy lessons, `1%` and `5%` local error budgets, cache and classifier comparisons, and binary `.s1m` verification |
| [Six-category benchmark report](../benchmarks/quality/instinct_gauntlet_report.md) | `BANKING77` (32 and 77 intents), `CLINC150-OOS`, `NotInject`, authored drift probes, four-field extraction, and zero-shot routing |
| [Drop-in and adaptation curve report](../benchmarks/quality/drop_in_and_adaptation_report.md) | Zero-shot, `adapt_zero_shot`, few-shot, and compiled `.s1m` curves on `BANKING77` and `CLINC150` |
| [`poke-env` Showdown benchmark report](../benchmarks/quality/poke_env_showdown_report.md) | Live `gen9randombattle` Showdown results against `RandomPlayer`, `MaxBasePowerPlayer`, and `SimpleHeuristicsPlayer` |
| [Architecture reference](architecture/technical_specification.md) | Runtime components, `.s1m` v2 binary format, and module map |
| [Technical brief](paper/system1_technical_brief.md) | Summary of the runtime and empirical results |
| [Whitepaper](paper/system1_whitepaper.md) | Full design, mathematical formulation, and workload evaluations |
| [Mathematical notes](paper/conformal_gating.md) | Split-conformal prediction, Mondrian calibration, dual OOD gating, and rank-1 RLS updates |
| [Benchmark suite index](../benchmarks/quality/README.md) | Reproducible benchmark runners and datasets |
