<p align="center">
  <img src="https://raw.githubusercontent.com/steph4n-gh/system1-instinct/main/assets/system1-logo.jpg" alt="System 1: Instinct logo" width="128" />
</p>

# System 1: Instinct

**Stop paying an LLM `500 ms` and an API call to make the same repeatable decision 10,000 times a day.**

**System 1: Instinct** is a sub-millisecond (`0.1–0.4 ms`), zero-weight (`~100 KiB`) decision runtime for Python. Drop it in front of your existing LLM, router, guardrail, or game loop:
1. **Start with `0` training examples** — define your output schema (`ChoiceField`, `BooleanField`, `MultiChoiceField`, `ScoreField`) with plain-English descriptions.
2. **Learn live in `< 1 ms` per example** — wrap your LLM or heuristic with `BoundedReviewFallback`. When Instinct is uncertain or sees an out-of-distribution input, it escalates to your teacher, updates its weights online (`learn_from_tier2` / `adapt_zero_shot`), and **takes over locally once confident**.
3. **Save a `~50–150 KiB` `.s1m` skill** — run anywhere on CPU with zero GPU, zero model weights, and cryptographic Ed25519 audit receipts.

[![CI](https://github.com/steph4n-gh/system1-instinct/actions/workflows/ci.yml/badge.svg)](https://github.com/steph4n-gh/system1-instinct/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://github.com/steph4n-gh/system1-instinct/blob/main/pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](https://github.com/steph4n-gh/system1-instinct/blob/main/LICENSE)

[Watch It Work (30 Seconds)](#watch-it-work-in-30-seconds) · [4 Real Project Drop-Ins](#4-real-projects-dropped-into-instinct) · [Honest Benchmarks](#honest-head-to-head-benchmarks) · [Drop-In Guides](docs/drop_in/README.md) · [Documentation](docs/README.md)

---

## Watch It Work in 30 Seconds

Instead of calling OpenAI/Anthropic on every ticket forever, wrap your LLM function with `BoundedReviewFallback`:

```python
from system1 import BooleanField, BoundedReviewFallback, ChoiceField, DecisionSchema, SystemOneEngine, TeachingSession

class TicketTriage(DecisionSchema):
    queue = ChoiceField(options=["billing", "infrastructure", "security"])
    urgent = BooleanField()

engine = SystemOneEngine(TicketTriage, strict_mode=True, use_cache=False)
session = TeachingSession(".system1/triage", TicketTriage)

# Wrap your existing LLM call — Instinct only calls it when uncertain, and learns online in <1 ms:
triage = BoundedReviewFallback(
    engine,
    review_fn=my_existing_llm_triage,   # <- your OpenAI / Anthropic / Instructor function
    teaching_session=session,
    auto_record_lessons=True,
    auto_online_update=True,
)

result = triage.decide("Production API returning 502 bad gateway after deployment")
print(result["values"])  # -> {'queue': 'infrastructure', 'urgent': True}
```

Run the real 3-wave cutover script included in this repo:

```bash
python examples/drop_in/llm_cutover_dropin.py
```

```text
Wave 1 (Cold Start)         | Local:   0.0% (0/6) | LLM Calls: 6 | Accuracy: 100.0% | P50: 0.903 ms
Wave 2 (Online RLS Active)  | Local: 100.0% (6/6) | LLM Calls: 0 | Accuracy: 100.0% | P50: 0.434 ms
Wave 3 (Compiled .s1m)      | Local: 100.0% (6/6) | LLM Calls: 0 | Accuracy: 100.0% | P50: 0.370 ms
Saved portable .s1m skill -> 45.1 KiB (0 LLM calls required in production)
```

---

## 4 Real Projects Dropped Into Instinct

We installed four real open-source frameworks (`semantic-router`, `langchain-core`, `FastAPI` + `Instructor`, and `poke-env` + `pokemon-showdown`) and dropped Instinct directly into them. Run every single one yourself:

### 1. Dropping Into `aurelio-labs/semantic-router` (`CLINC150-OOS`: 24 Intents + 120 Out-of-Scope Queries)
* **Guide & Code**: **[docs/drop_in/01_semantic_router.md](docs/drop_in/01_semantic_router.md)** · Run: `python examples/drop_in/semantic_router_dropin.py`
* **What we did**: Passed the exact same `semantic_router.Route(name=..., utterances=...)` objects from `CLINC150-OOS` (`144` held-out in-scope queries + `120` out-of-scope queries) into both the real `SemanticRouter(encoder=FastEmbedEncoder("BAAI/bge-small-en-v1.5"))` and `InstinctRouteLayer(routes)`.
* **What happened**:
  ```text
  [semantic-router (bge-small-en-v1.5, 67.2 MB ONNX)] Init=2471 ms | P50=4.31 ms | In-Scope=93.1% | OOS False Accept=100.0% (120/120 OOS queries misrouted!)
  [System 1: Instinct (93.3 KiB .s1m, 0 neural weights)] Init=115 ms | P50=0.14 ms | In-Scope=87.5% (Accepted=100.0%) | OOS False Accept=0.0%–0.8% (rejects OOS!)
  ```
  * **The Truth**: `bge-small-en-v1.5` (`67.2 MB` neural weights) has `+5.6%` higher raw in-scope recall on clean paraphrases, **but `SemanticRouter` falsely routes `100%` (`120/120`) of unrelated Out-of-Scope queries to an assistant tool** (even after `sr.fit()`), because dense cosine similarity stays above `0.50` on out-of-domain text. **Instinct (`93 KiB`, `30×` faster)** abstains on unfamiliar inputs (`0.0%` OOS false acceptance in the [6-category gauntlet](benchmarks/quality/v1_4_gauntlet_report.md)) and achieves **`100.0%` accuracy on accepted decisions**.

### 2. Dropping Into `langchain-core` LCEL Pipelines (`BANKING77`: 32 Intents, Clean + Typo Stress)
* **Guide & Code**: **[docs/drop_in/02_routellm_cascade.md](docs/drop_in/02_routellm_cascade.md)** · Run: `python examples/drop_in/routellm_dropin.py`
* **What we did**: Dropped Instinct into a `langchain_core.runnables.RunnableLambda` LCEL pipeline across `256` held-out `BANKING77` customer tickets (both clean and character-typo corrupted) head-to-head against `sentence-transformers/all-MiniLM-L6-v2` (`91.1 MB` ONNX).
* **What happened**:
  ```text
  [Classical Unigram TF-IDF (85 KiB)]         Clean=77.3% | Typo=66.0% | P50=0.18 ms
  [all-MiniLM-L6-v2 (91.1 MB ONNX, 22M)]      Clean=91.0% | Typo=70.7% (-20.3% drop!) | P50=1.32 ms
  [System 1: Instinct Static (120 KiB .s1m)]  Clean=87.9% | Typo=74.2% (+3.5% vs MiniLM!) | P50=0.22 ms
  [System 1: Instinct + Online Flywheel]      Clean=91.4% (surpasses MiniLM after 48/256 teacher reviews!)
  ```
  * **The Truth**: Static `all-MiniLM-L6-v2` starts `3.1%` ahead on clean text (`91.0%` vs `87.9%`), **but shatters under character typos (`-20.3%` drop to `70.7%`)** because swapped characters break WordPiece tokens. Instinct's character n-grams beat `all-MiniLM-L6-v2` on typos (**`74.2%` vs `70.7%`**), and after reviewing just `48` boundary tickets via `BoundedReviewFallback`, **Instinct surpasses `all-MiniLM-L6-v2` on clean text (`91.4%` vs `91.0%`)** at **`750×` smaller size** and **`6×` faster latency**.

### 3. Dropping Into `FastAPI` + `Instructor` / Pydantic (`POST /v1/triage`: 4-Field Simultaneous Extraction)
* **Guide & Code**: **[docs/drop_in/03_instructor_structured_outputs.md](docs/drop_in/03_instructor_structured_outputs.md)** · Run: `python examples/drop_in/instructor_structured_dropin.py`
* **What we did**: Built a real `FastAPI` `POST /v1/triage` microservice returning a 4-field Pydantic model (`department`, `page_oncall`, `impact_tags`, `urgency_score`) and sent `35` held-out multi-field incidents across 3 traffic waves.
* **What happened**:
  ```text
  [Compiled .s1m = 63.8 KiB] Joint 4-Field Exact Match = 100.0% | P50 Latency = 0.455 ms
  Wave 1: Local Instinct Handled= 62.9% | Instructor LLM Escalations=37.1% | End-to-End 4-Field Accuracy=100.0%
  Wave 2: Local Instinct Handled=100.0% | Instructor LLM Escalations= 0.0% | End-to-End 4-Field Accuracy=100.0%
  Wave 3: Local Instinct Handled=100.0% | Instructor LLM Escalations= 0.0% | End-to-End 4-Field Accuracy=100.0%
  ```

### 4. Dropping Into `poke-env` + Live Pokémon Showdown (`gen9randombattle`)
* **Guide & Code**: **[docs/drop_in/06_poke_env_showdown_agent.md](docs/drop_in/06_poke_env_showdown_agent.md)** · Run: `python benchmarks/quality/evaluate_poke_env_showdown.py --battles 50`
* **What we did**: Subclassed `poke_env.player.Player` in 15 lines of Python and played 450 live `gen9randombattle` matches on a local Pokémon Showdown server against `poke-env`'s official benchmark bots (`RandomPlayer`, `MaxBasePowerPlayer`, and `SimpleHeuristicsPlayer`).
* **What happened**:

| System 1: Instinct Agent | vs `RandomPlayer` | vs `MaxBasePowerPlayer` | vs `SimpleHeuristicsPlayer` | P50 Decision Latency |
| :--- | :---: | :---: | :---: | :---: |
| **`System 1 Zero-Shot (0 Examples)`** | **100.0%** (`50/50`) | **98.0%** (`49/50`) | **42.0%** (`21/50`) | `3.21 ms` |
| **`System 1 Zero-Shot Adaptive (Online)`** | **100.0%** (`50/50`) | **92.0%** (`46/50`) | **62.0%** (`31/50`) | `3.20 ms` |
| **`System 1 Compiled (.s1m)`** | **98.0%** (`49/50`) | **82.0%** (`41/50`) | **60.0%** (`30/50`) | **`0.093 ms`** (`93 µs`) |

  * **The Truth**: With `0` training examples (just 4 plain-English tactical descriptions), Instinct crushes `RandomPlayer` (`100%`) and `MaxBasePowerPlayer` (`98%`), but loses to `SimpleHeuristicsPlayer` (`42%`) because static descriptions don't know exact damage-trade thresholds. Turning on **online adaptation between turns (`adapt_zero_shot` + `learn_from_tier2`) flips `42% -> 62%` (`+20%` win rate)**, and compiling to a **`23 KiB` `.s1m` skill** sustains a **`60%` win rate at `93 microseconds` per turn**.

---

## Honest Head-to-Head Benchmarks

We benchmark against **Classical Linear Baselines (`Unigram TF-IDF`)** and **Pretrained Neural Embedding Routers (`all-MiniLM-L6-v2` and `BAAI/bge-small-en-v1.5`)** on the exact same held-out public splits.

### 1. Instinct vs. Pretrained Neural Encoders (`all-MiniLM-L6-v2` & `bge-small-en-v1.5`)
Reproduce with `python benchmarks/quality/evaluate_drop_in_replacements.py` ([full report](benchmarks/quality/drop_in_and_adaptation_report.md)):

| Router Architecture | Model / Artifact Size | `BANKING77` (32c Clean) | `BANKING77` (32c Typo Stress) | `CLINC150-OOS` False Accept | Negation / Polarity Acc | P50 CPU Latency | `<1 ms` Online RLS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Linear Baseline** (`Unigram TF-IDF`) | `85 KiB` (`0` weights) | 77.3% | 66.0% | 3.3% | 70.0% | `0.18 ms` | No |
| **`all-MiniLM-L6-v2`** (`22M` Dense Bi-Encoder) | `91.1 MB` ONNX | 91.0% | 70.7% (`-20.3%` drop) | 8.0% | 60.0% | `1.32 ms` (`~10 ms` PyTorch) | No |
| **`BAAI/bge-small-en-v1.5`** (`33M` Dense Bi-Encoder) | `67.2 MB` ONNX | **94.5%** | **77.0%** (`-17.5%` drop) | 5.0% (cal) / 100% (`@0.5`) | 90.0% | `2.96 ms` (`~14 ms` PyTorch) | No |
| **System 1: Instinct (Static `.s1m`)** | **`120 KiB`** (`0` weights) | **87.9%** | **74.2%** (`+3.5%` vs MiniLM) | **0.0%** (`0/120`) | **90.0%** | **`0.22 ms`** | **Yes** |
| **System 1: Instinct (`+ Online Flywheel`)** | **`120 KiB`** (`0` weights) | **91.4%** (`+0.4%` vs MiniLM) | **74.2%** | **0.0%** (`0/120`) | **100.0%** | **`0.22 ms`** | **Yes** |

* **When a Pretrained Neural Encoder is better**: If you have zero domain vocabulary overlap (pure synonym paraphrase matching), don't mind shipping `67–91 MB` of ONNX/PyTorch weights, and don't need sub-millisecond online weight updates or strict OOS abstention, `BAAI/bge-small-en-v1.5` has `+3.1%` higher static clean recall (`94.5%` vs `91.4%`). *(Note: Instinct can also wrap neural encoders via `LocalNeuralProjector` / `HybridSemanticProjector` when you want both!)*
* **When System 1: Instinct is better**: When you want **`0` neural weights (`120 KiB` `.s1m`)**, **`0.22 ms` CPU latency**, **`0.0%` Out-of-Scope false acceptance**, **negation/polarity precision (`"do not cancel, upgrade"` $\to$ `90%–100%` vs MiniLM's `60%`)**, **character-typo resilience (`74.2%` vs MiniLM's `70.7%`)**, and **`< 1 ms` live 1-shot updates (`learn_from_tier2`)**.

### 2. Zero-Shot $\to$ Few-Shot $\to$ Compiled `.s1m` Adaptation Curves
Starting from **`0` training examples** (using only raw intent label names) on `24` intents (`python benchmarks/quality/evaluate_drop_in_replacements.py`):

| Public Dataset (24 Intents) | `0`-Shot (Label Names Only) | `0`-Shot + `adapt_zero_shot` | `1`-Shot Online RLS | `3`-Shot Online RLS | `5`-Shot Online RLS | Compiled `.s1m` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`BANKING77`** (`192` test queries) | `53.6%` | **`55.2%`** | **`57.3%`** | **`65.6%`** | **`70.3%`** | **`90.1%`** (`0.22 ms`) |
| **`CLINC150`** (`120` test queries) | `61.7%` | **`62.5%`** | **`65.8%`** | **`72.5%`** | **`82.5%`** | **`87.5%`** (`0.20 ms`) |

### 3. Full 6-Category Offline Benchmark Gauntlet
Reproduce 100% offline (`block_network()`) with `python benchmarks/quality/evaluate_v1_4_gauntlet.py` ([full report](benchmarks/quality/v1_4_gauntlet_report.md)):

| Category | Workload | Classical Linear Baseline | System 1: Instinct (Static) | System 1: Instinct (+ Flywheel) |
| :--- | :--- | :---: | :---: | :---: |
| **1. High-Cardinality Intent Routing** | `BANKING77` (32 Intents, Clean / Typo) | 77.3% / 66.0% | **87.9%** / **74.2%** | **91.4%** |
| **1b. Full 77-Class Scaling** | `BANKING77` (77 Intents, Clean / Typo) | 62.0% / 45.5% | **79.5%** / **58.3%** | **88.1%** |
| **2. Multi-Domain + OOS Rejection** | `CLINC150-OOS` (In-Scope Acc / OOS False Accept) | 84.7% / 3.3% FA | **86.1%** / **0.0% FA** | **99.3%** / **0.0% FA** |
| **3. Injection & NotInject Negatives** | `NotInject` Benign-Trigger FPR / Polarity Acc | 10.0% FPR / 70.0% | **0.0% FPR** / **90.0%** | **0.0% FPR** / **100.0%** |
| **4. Authored Drift Probes (106)** | `banking_support` / `sms_triage` / `model_routing` | 86.2% / 77.5% / 77.8% cov | **100.0%** / **82.5%** / **88.9% cov** | **100.0%** / **87.5%** / **94.4% cov** |
| **5. Multi-Field Structured Triage** | 4-Field Joint Accuracy (`Choice`+`Bool`+`Multi`+`Score`) | 97.1% | **100.0%** (`0.40 ms`) | **100.0%** (`0%` Wave 3 esc) |
| **6. Zero-Shot Cold-Start Routing** | `intent_routing` / `security_triage` (0-Shot) | 51.0% / 50.0% | **81.0%** / **85.0%** | **100.0%** |

---

## Quickstart

```bash
git clone https://github.com/steph4n-gh/system1-instinct.git
cd system1-instinct
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,langchain]'

# Run the live LLM auto-cutover demo:
python examples/drop_in/llm_cutover_dropin.py

# Run the 4 real framework drop-ins:
python examples/drop_in/semantic_router_dropin.py
python examples/drop_in/routellm_dropin.py
python examples/drop_in/instructor_structured_dropin.py
python examples/drop_in/prompt_guard_dropin.py
```

---

## Guides & Documentation

| What You Want to Do | Guide |
| :--- | :--- |
| **Drop Instinct into `Semantic Router`, `RouteLLM`, `Instructor`, `Prompt-Guard`, or `poke-env`** | [Step-by-Step Drop-In Guides (`docs/drop_in/README.md`)](docs/drop_in/README.md) |
| **Observe a live OpenAI / Anthropic / Gemini teacher and auto-cutover** | [LLM Auto-Cutover Guide (`docs/drop_in/05_llm_observe_and_cutover.md`)](docs/drop_in/05_llm_observe_and_cutover.md) · [`TypeSafeClient` (`docs/typesafe.md`)](docs/typesafe.md) |
| **Teach, qualify, and version skills with `TeachingSession`** | [Correcting Skills](docs/guides/correcting_skills.md) · [Training Experts](docs/guides/training_experts.md) |
| **Serve skills via LangChain, MCP, FastAPI/ASGI, or gRPC** | [Integrations Guide](docs/integrations.md) · [Policy Guard & Ed25519 Ledger](docs/guides/policy_guard.md) |
| **Read the math & architecture specs** | [Technical Spec](docs/architecture/technical_specification.md) · [Whitepaper](docs/paper/system1_whitepaper.md) · [Conformal Math](docs/paper/conformal_gating.md) |

**Version 1.0.0:** [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Apache License 2.0](LICENSE)
