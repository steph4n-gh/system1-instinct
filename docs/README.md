# System 1: Instinct documentation

Current stable runtime: **1.4.0**. Source documentation reviewed 14 October 2026.
System 1 is not an LLM. Start with one journey: **choose a decision → show
examples (or start zero-shot) → check a candidate → adopt → correct and compare**.
A person or a running teacher can supply the examples, `adapt_zero_shot(...)` can
adapt prototypes from unlabeled live traffic, and `BoundedReviewFallback` can close
the loop by escalating uncertain boundary cases to a reviewer and updating the
local skill. See the [CHANGELOG](../CHANGELOG.md) for current 1.4.0 capabilities.

## Use the product

| Start here | Purpose |
|---|---|
| [README quickstart](../README.md#quickstart) | Install System 1 and run a complete taught skill |
| [Drop-In Replacements](../examples/drop_in/README.md) | 1-line local replacements for Semantic Router, RouteLLM, Instructor / BAML, and Llama-Prompt-Guard / Guardrails |
| [Your first skill](guides/first_skill.md) | What an example means, an editable lesson file, corrections and readiness |
| [Correct, compare, and adopt](guides/correcting_skills.md) | Retain lessons across single- and multi-field schemas, inspect before/after answers, and explicitly adopt a passing candidate |
| [Teach by doing](../examples/teaching_by_doing/README.md) | Use the same candidate/adoption flow by filing fictional documents |
| [Teaching guide](guides/training_experts.md) | Label formats, separate evidence, compilation and corrections |
| [Teacher observation](typesafe.md) | Callback contract, actual Jev/Gemini evidence, promotion and reuse |
| [Example catalog](../examples/README.md) | Every example's role and supported or experimental scope |
| [Application integrations](integrations.md) | Connect a checked skill to LangChain, MCP, ASGI, gRPC or monitoring |
| [Tool permissions](guides/policy_guard.md) | Add an explicit permission rule when an answer would trigger an action |
| [Deployment boundaries](deployment.md) | Permissions, identity, audit, network paths and service limits |
| [Observability](observability/README.md) | Optional metrics/tracing and dashboard |
| [Container templates](../deploy/README.md) | Experimental local Docker/Kubernetes starting points |

## Understand the implementation and evidence

| Document | Scope |
|---|---|
| [Architecture reference](architecture/technical_specification.md) | Actual components, defaults, code links and lifecycle |
| [Technical brief](paper/system1_technical_brief.md) | Short product and evidence overview |
| [Whitepaper](paper/system1_whitepaper.md) | Current design, measurements and limitations |
| [Mathematical notes](paper/conformal_gating.md) | Established results and their assumptions |
| [v1.4 benchmark gauntlet report](../benchmarks/quality/v1_4_gauntlet_report.md) | Six-category offline evaluation of v1.2 baseline vs. v1.4 static vs. v1.4 closed-loop flywheel |
| [Drop-in & adaptation curve report](../benchmarks/quality/drop_in_and_adaptation_report.md) | Drop-in replacement benchmarks + `0`-shot $\to$ `adapt_zero_shot` $\to$ `1`-shot $\to$ `3`-shot $\to$ `5`-shot $\to$ `Compiled .s1m` on `BANKING77` & `CLINC150` |
| [`poke-env` Showdown benchmark report](../benchmarks/quality/poke_env_showdown_report.md) | Live `gen9randombattle` Showdown benchmark against `RandomPlayer`, `MaxBasePowerPlayer`, and `SimpleHeuristicsPlayer` |
| [Workload and evidence index](../benchmarks/quality/README.md) | One overview of task scope, held-out or reused evidence, baselines, successes and failed targets |
| [Real-document workflow test](../benchmarks/quality/document_workflow/README.md) | BBC topic routing: fewer accepted mistakes with more review; every candidate rejected by default adoption gates |
| [Teaching examples](../examples/teaching/README.md) | Authored lessons, current counts and historical development |
| [Benchmark index](../benchmarks/README.md) | Separate quality, throughput and historical runners |
| [Research and integration examples](../examples/README.md) | Inbox Zero, n8n, games and composition labs, with each experiment's limits |

The quality index links the v1.4 gauntlet, drop-in & adaptation curves, Pokémon
Showdown benchmark, public-data, teacher, email, BBC workflow, n8n and game reports.
Research remains available without being a prerequisite for teaching one useful skill.

The papers are maintainer-authored documentation, not peer-reviewed publications.
Quality claims must identify the task, evidence split and acceptance rule. Local
computation, acceptance without review and correctness are separate measurements.
Do not compare unlike timing cohorts as a measured speedup. See
[contributing](../CONTRIBUTING.md) for the documentation check and release procedure.
