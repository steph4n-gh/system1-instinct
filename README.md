<p align="center">
  <img src="https://raw.githubusercontent.com/steph4n-gh/system1-instinct/main/assets/system1-logo.jpg" alt="System 1: Instinct logo" width="128" />
</p>

# System 1: Instinct

**Sub-millisecond non-autoregressive decision runtime and zero-shot adaptive skills.**

**System 1: Instinct is not an LLM.** Define a structured decision (`ChoiceField`, `BooleanField`, `MultiChoiceField`, `ScoreField`), start **zero-shot** from option descriptions or teach from examples, adapt online in sub-millisecond time (`adapt_zero_shot` / `learn_from_tier2`), and save portable `.s1m` skills that run locally on CPU or Apple Silicon MLX with zero language-model weights and zero API calls.

[![CI](https://github.com/steph4n-gh/system1-instinct/actions/workflows/ci.yml/badge.svg)](https://github.com/steph4n-gh/system1-instinct/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://github.com/steph4n-gh/system1-instinct/blob/main/pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](https://github.com/steph4n-gh/system1-instinct/blob/main/LICENSE)

[Drop-In Guides](docs/drop_in/README.md) · [Quickstart](#quickstart) · [Benchmarks](#benchmarks) · [Integrations](#integrations--production-deployment) · [Documentation](docs/README.md)

---

## Why System 1: Instinct?

| Capability | What It Does | Measured Performance |
| :--- | :--- | :--- |
| **Sub-Millisecond Decisions** | Single-pass matrix heads over spectral & lexical features (no token generation) | **`0.09–0.44 ms`** median CPU latency |
| **Zero-Shot Cold Start** | Routes immediately from `ChoiceField` descriptions using sibling-contrastive exemplar banks | **`81–100%`** 0-shot accuracy (`0` examples) |
| **Unsupervised Drift Adaptation** | `engine.adapt_zero_shot(unlabeled_texts)` bridges new domain jargon with `0` teacher calls | **`0% -> 100%`** unseen codeword recovery |
| **1-Shot Online RLS Learning** | `engine.learn_from_tier2(prompt, label)` updates weights via Sherman-Morrison in `<1 ms` | Monotonic `0`-shot $\to$ `1`-shot $\to$ `5`-shot gains |
| **Strict Conformal & OOD Gating** | Mondrian split-conformal sets + dual familiarity/prototype out-of-distribution abstention | **`0.0%`** false acceptance on `CLINC150-OOS` |
| **Portable `.s1m` Skills** | Self-contained, checksummed artifact with Ed25519 receipt signing & SHA-256 audit ledger | **`20–155 KiB`** per saved `.s1m` skill |

---

## Step-by-Step Drop-In Replacement Guides

Replace slow LLM routers, guardrails, and structured extractors in **one command**. Every guide in **[docs/drop_in/README.md](docs/drop_in/README.md)** includes before/after code, a runnable script, and real measured output:

| # | Drop-In Guide | Replaces | Run Command | Measured Result | P50 Latency |
| :-: | :--- | :--- | :--- | :--- | :---: |
| **1** | **[Semantic Router](docs/drop_in/01_semantic_router.md)** | `semantic_router.RouteLayer` | `python examples/drop_in/semantic_router_dropin.py` | **100% In-Domain · 100% OOD Reject · 100% Adapted** | **`1.36 ms`** |
| **2** | **[RouteLLM Cascade](docs/drop_in/02_routellm_cascade.md)** | `routellm.controller.Controller` | `python examples/drop_in/routellm_dropin.py` | **100% 3-Tier Cascade Routing** | **`1.47 ms`** |
| **3** | **[Instructor / BAML](docs/drop_in/03_instructor_structured_outputs.md)** | `instructor.from_openai(...).create()` | `python examples/drop_in/instructor_structured_dropin.py` | **100% 4-Field Joint Extraction** | **`2.73 ms`** |
| **4** | **[Prompt-Guard Security](docs/drop_in/04_prompt_guard_security.md)** | `Llama-Prompt-Guard` / `Guardrails` | `python examples/drop_in/prompt_guard_dropin.py` | **0.0% `NotInject` Benign FPR · 100% Attack Recall** | **`0.81 ms`** |
| **5** | **[LLM Auto-Cutover](docs/drop_in/05_llm_observe_and_cutover.md)** | Recurring OpenAI / Anthropic calls | `python examples/drop_in/llm_cutover_dropin.py` | **Wave 1: 6 LLM calls $\to$ Wave 2: 0 LLM calls (100% Local)** | **`0.37 ms`** |
| **6** | **[`poke-env` Showdown Agent](docs/drop_in/06_poke_env_showdown_agent.md)** | Slow LLM / static game bots | `python benchmarks/quality/evaluate_poke_env_showdown.py` | **100% vs `Random` · 98% vs `MaxPower` · 62% vs `Heuristics`** | **`0.093 ms`** |

---

## Quickstart

```bash
git clone https://github.com/steph4n-gh/system1-instinct.git
cd system1-instinct
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

### 1. Start Zero-Shot (0 Examples, 0 API Keys)

```python
from system1 import BooleanField, ChoiceField, DecisionSchema, SystemOneEngine

class TicketTriage(DecisionSchema):
    queue = ChoiceField(
        options=["billing", "infrastructure", "security"],
        descriptions={
            "billing": ["refund duplicate charge on billing invoice", "annual subscription receipt"],
            "infrastructure": ["production API 502 bad gateway timeout", "kubernetes pod crashloop"],
            "security": ["unauthorized login and leaked API token", "phishing credential compromise"],
        },
    )
    urgent = BooleanField(
        true_description="critical production outage security breach unauthorized access",
        false_description="routine billing invoice receipt standard question",
    )

engine = SystemOneEngine(TicketTriage)
decision = engine.decide("Production API returning 502 bad gateway after deployment", record_receipt=False)
print(decision.values)  # -> {'queue': 'infrastructure', 'urgent': True}
```

### 2. Zero-Shot & Online Adaptation (`adapt_zero_shot` + `learn_from_tier2`)

```python
# A. Unsupervised test-time adaptation on unlabeled traffic (0 labels, 0 teacher calls)
engine.adapt_zero_shot([
    "Unauthorized login and leaked API token with alert code SEC_ZETA_9",
    "Production API 502 bad gateway timeout on cluster OPS_OMEGA_4",
])

# B. 1-shot supervised online RLS update (<1 ms, preserves calibration)
engine.learn_from_tier2(
    "Wire settlement discrepancy on ledger batch 44",
    {"queue": "billing", "urgent": False},
    preserve_calibration=True,
)
```

### 3. Closed-Loop Flywheel (`BoundedReviewFallback` + `TeachingSession`)

Wrap any teacher callable (`review_fn`) so uncertain boundary cases escalate to your reviewer, record lessons into `TeachingSession`, and update the local skill online:

```python
from system1 import BoundedReviewFallback, TeachingSession

session = TeachingSession(".system1/triage-session", TicketTriage)
fallback = BoundedReviewFallback(
    engine,
    review_fn=lambda prompt: {"queue": "billing", "urgent": False},
    teaching_session=session,
    auto_record_lessons=True,
    auto_online_update=True,
    lesson_split="auto",
)
res = fallback.decide("Please refund my duplicate invoice charge")
```

---

## Benchmarks

### 1. Zero-Shot $\to$ Few-Shot $\to$ Compiled `.s1m` Adaptation Curves
Reproduce with `python benchmarks/quality/evaluate_drop_in_replacements.py` ([full report](benchmarks/quality/drop_in_and_adaptation_report.md)):

| Multi-Class Benchmark (24 Intents) | `0`-Shot (Schema Only) | `0`-Shot + `adapt_zero_shot` | `1`-Shot Online RLS | `3`-Shot Online RLS | `5`-Shot Online RLS | Compiled `.s1m` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`BANKING77`** (`192` test queries) | `53.6%` | **`55.2%`** | **`57.3%`** | **`65.6%`** | **`70.3%`** | **`90.1%`** (`0.22 ms`) |
| **`CLINC150`** (`120` test queries) | `61.7%` | **`62.5%`** | **`65.8%`** | **`72.5%`** | **`82.5%`** | **`87.5%`** (`0.20 ms`) |

### 2. 6-Category Industry-Standard Decision Benchmark Gauntlet
Reproduce 100% offline with `python benchmarks/quality/evaluate_v1_4_gauntlet.py` ([full report](benchmarks/quality/v1_4_gauntlet_report.md)):

| Category | Workload | Classical Linear Baseline | System 1: Instinct (Static) | System 1: Instinct (+ Flywheel) |
| :--- | :--- | :---: | :---: | :---: |
| **1. High-Cardinality Intent Routing** | `BANKING77` (32 Intents, Clean / Typo) | 77.3% / 66.0% | **87.9%** / **74.2%** | **91.4%** |
| **1b. Full 77-Class Scaling** | `BANKING77` (77 Intents, Clean / Typo) | 62.0% / 45.5% | **79.5%** / **58.3%** | **88.1%** |
| **2. Multi-Domain + OOS Rejection** | `CLINC150-OOS` (In-Scope Acc / OOS False Accept) | 84.7% / 3.3% FA | **86.1%** / **0.0% FA** | **99.3%** / **0.0% FA** |
| **3. Injection & NotInject Negatives** | `NotInject` Benign-Trigger FPR / Polarity Acc | 10.0% FPR / 70.0% | **0.0% FPR** / **90.0%** | **0.0% FPR** / **100.0%** |
| **4. Authored Drift Probes (106)** | `banking_support` / `sms_triage` / `model_routing` | 86.2% / 77.5% / 77.8% cov | **100.0%** / **82.5%** / **88.9% cov** | **100.0%** / **87.5%** / **94.4% cov** |
| **5. Multi-Field Structured Triage** | 4-Field Joint Accuracy (`Choice`+`Bool`+`Multi`+`Score`) | 97.1% | **100.0%** (`0.40 ms`) | **100.0%** (`0%` Wave 3 esc) |
| **6. Zero-Shot Cold-Start Routing** | `intent_routing` / `security_triage` (0-Shot) | 51.0% / 50.0% | **81.0%** / **85.0%** | **100.0%** |

### 3. Official `poke-env` + Pokémon Showdown Benchmark (`gen9randombattle`)
Reproduce with `python benchmarks/quality/evaluate_poke_env_showdown.py --battles 50` ([full report & analysis](docs/drop_in/06_poke_env_showdown_agent.md)):

| System 1: Instinct Agent | vs `RandomPlayer` | vs `MaxBasePowerPlayer` | vs `SimpleHeuristicsPlayer` | P50 Decision Latency |
| :--- | :---: | :---: | :---: | :---: |
| **`System 1 Zero-Shot (0 Examples)`** | **100.0%** (`50/50`) | **98.0%** (`49/50`) | **42.0%** (`21/50`) | `3.21 ms` |
| **`System 1 Zero-Shot Adaptive (Online)`** | **100.0%** (`50/50`) | **92.0%** (`46/50`) | **62.0%** (`31/50`) | `3.20 ms` |
| **`System 1 Compiled (.s1m)`** | **98.0%** (`49/50`) | **82.0%** (`41/50`) | **60.0%** (`30/50`) | **`0.093 ms`** |

---

## Integrations & Production Deployment

| Integration | Documentation |
| :--- | :--- |
| **Step-by-Step Drop-In Guides** | [docs/drop_in/README.md](docs/drop_in/README.md) |
| **Teaching & Candidate Qualification** | [docs/guides/correcting_skills.md](docs/guides/correcting_skills.md) · [docs/guides/training_experts.md](docs/guides/training_experts.md) |
| **Live Teacher Observation (`TypeSafeClient`)** | [docs/typesafe.md](docs/typesafe.md) |
| **LangChain, MCP, FastAPI/ASGI, gRPC** | [docs/integrations.md](docs/integrations.md) |
| **Deterministic Policy Guard & Audit Ledger** | [docs/guides/policy_guard.md](docs/guides/policy_guard.md) · [docs/deployment.md](docs/deployment.md) |
| **Architecture & Mathematical Foundations** | [Technical Spec](docs/architecture/technical_specification.md) · [Whitepaper](docs/paper/system1_whitepaper.md) · [Conformal Math](docs/paper/conformal_gating.md) |

---

## Release & Development

**Version 1.0.0:** **System 1: Instinct** complete release featuring zero-shot sibling-contrastive exemplar routing, unsupervised test-time adaptation (`adapt_zero_shot`), contrastively-centered multi-class online RLS updates (`learn_from_tier2`), spectral `PPMI + SVD` hybrid projection (`HybridProjector.fit`), hierarchical coarse-to-fine routing (`--hierarchical`), dual-gate OOD abstention (`--familiarity-alpha`, `--prototype-alpha`), multi-field `TeachingSession`, and `BoundedReviewFallback`. See the [changelog](CHANGELOG.md).

```bash
python -m pip install -e '.[dev,langchain]'
python -m pytest tests/ -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [Apache License 2.0](LICENSE).
