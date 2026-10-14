<p align="center">
  <img src="https://raw.githubusercontent.com/steph4n-gh/system1-instinct/main/assets/system1-logo.jpg" alt="System 1: Instinct logo" width="128" />
</p>

# System 1: Instinct

**Stop paying an LLM `500 ms` and an API call to make the same repeatable decision 10,000 times a day.**

**System 1: Instinct** is a sub-millisecond (`0.1–0.4 ms`), zero-weight (`~100 KiB`) decision runtime for Python. You put it in front of your existing LLM, local model, or tool router:
- **Start at `0` examples** with a typed schema (`ChoiceField`, `BooleanField`, `MultiChoiceField`, `ScoreField`).
- **Teach it `3 to 10` real examples per option**—either automatically by watching your LLM answer live traffic, or by passing a handful of examples directly.
- **Carry `70%–95%+` of repeatable traffic locally in `0.2 ms` on CPU**, while automatically escalating ambiguous or unfamiliar inputs (`is_ambiguous=True`) back to your LLM.

[![CI](https://github.com/steph4n-gh/system1-instinct/actions/workflows/ci.yml/badge.svg)](https://github.com/steph4n-gh/system1-instinct/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://github.com/steph4n-gh/system1-instinct/blob/main/pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](https://github.com/steph4n-gh/system1-instinct/blob/main/LICENSE)

[How It Works (0 to 10 Examples)](#from-zero-to-carrying-real-load-what-teaching-actually-looks-like) · [Instinct vs. Local LLMs](#when-to-use-instinct-vs-a-local-llm-or-fine-tuned-model) · [Real Project Drop-Ins](#drop-into-real-projects) · [Receipts & Public Benchmarks](#receipts--public-benchmarks) · [Documentation](docs/README.md)

---

## From Zero to Carrying Real Load: What Teaching Actually Looks Like

A raw `0`-shot classifier is **not** magic on its own—if you force a `0`-shot model to guess across 20+ intents with zero examples, it only gets `~53%–62%`. Nobody wants a `53%` router in production.

What makes **System 1: Instinct** work in production from Request #1 is **Conformal Abstention + Sub-Millisecond Online Learning**:

### Step 1: Start at Zero (`0` Examples)
Define your structured decision. At `0` examples, Instinct only answers obvious matches and **abstains (`is_ambiguous=True`) on everything else**, handing uncertain requests to your LLM so your production accuracy starts at **`100%`**:

```python
from system1 import BooleanField, BoundedReviewFallback, ChoiceField, DecisionSchema, SystemOneEngine, TeachingSession

class SupportTriage(DecisionSchema):
    queue = ChoiceField(
        options=["refunds", "account_access", "technical_bug"],
        descriptions={
            "refunds": "Chargebacks, duplicate billing, invoice disputes, cancel and refund",
            "account_access": "Password reset, locked out, 2FA code not arriving, SSO login failure",
            "technical_bug": "502 bad gateway, crash on startup, API timeout, webhook failing",
        },
    )
    urgent = BooleanField(description="True if production is down, security is compromised, or money is blocked")

engine = SystemOneEngine(SupportTriage, strict_mode=True, use_cache=False)
```

### Step 2: Teach It `3 to 10` Examples per Option (Two Ways)

You do **not** need `5,000,000` examples or a GPU training pipeline. You either let your LLM teach Instinct automatically on live traffic, or feed `3–10` examples per option yourself:

```python
# WAY A: Automatic (0 manual labeling) — Wrap your LLM call.
# Whenever Instinct is unsure, it calls `my_llm_triage`, returns the LLM's answer,
# and updates its own weights in < 0.5 ms so it can handle the next similar request locally.
triage = BoundedReviewFallback(
    engine,
    review_fn=my_llm_triage,
    teaching_session=TeachingSession(".system1/triage", SupportTriage),
    auto_record_lessons=True,
    auto_online_update=True,
)
result = triage.decide("Our SSO login keeps throwing SAML signature error for all users")

# WAY B: Explicit (< 0.5 ms per example) — Teach a few examples or fix a production miss on the spot:
engine.learn_from_tier2(
    "charged twice on my Visa ending 4412, do not upgrade, just refund the duplicate",
    {"queue": "refunds", "urgent": False},
)
engine.learn_from_tier2(
    "production webhook endpoint returning 502 bad gateway since the 9am deploy",
    {"queue": "technical_bug", "urgent": True},
)
```

Run the **"Break-Me & Teach-Me" Live Showcase** ([`examples/showcase_break_and_teach.py`](examples/showcase_break_and_teach.py))—or pass `--interactive` to type your own curveballs and teach it in 1 line against `BAAI/bge-small-en-v1.5` (`67.2 MB` ONNX):

```bash
python examples/showcase_break_and_teach.py
# Or launch the interactive REPL:
python examples/showcase_break_and_teach.py --interactive
```

```text
[Setup from 6 examples/option (30 total teacher logs)]
  • Neural Vector Router (`BAAI/bge-small-en-v1.5`) : 67,200 KiB ONNX | Init:  208.8 ms
  • System 1: Instinct (`.s1m` compiled skill)      :   23.7 KiB .s1m | Init:   40.3 ms

ACT 1: ROUTINE TRAFFIC — Replacing 450 ms LLM Calls in ~0.2 ms ($0 API Cost)
  Query: 'Please cancel my monthly subscription and stop billing me' -> LOCAL .s1m (0.377 ms): cancel_subscription ✅
  Query: 'My Visa card was billed twice for the same invoice today'  -> LOCAL .s1m (0.213 ms): refund_duplicate_charge ✅

ACT 2: 'BREAK ME' — 3 Real-World Curveballs Where Static Vector Routers Fail
  [Trap 1: Negation Flip] 'Please do NOT cancel my subscription, I want to upgrade to Pro instead'
    • bge-small-en-v1.5 (4.23 ms, cos=0.80) -> cancel_subscription      ❌ MISROUTED!
    • System 1: Instinct (0.40 ms, fam=12.10) -> upgrade_plan             ✅
  [Trap 2: Out-of-Scope Garbage] 'What is the best temperature to bake sourdough bread in a Dutch oven?'
    • bge-small-en-v1.5 (3.81 ms, cos=0.46) -> upgrade_plan             ❌ MISROUTED!
    • System 1: Instinct (0.32 ms, fam=4.80) -> ABSTAIN_TO_LLM           ✅
  [Trap 3: Heavy Mobile Typos] 'chagred twcie on my vsia crad endign 4412, pls refnud'
    • bge-small-en-v1.5 (3.94 ms, cos=0.67) -> reset_account_access     ❌ MISROUTED!
    • System 1: Instinct (0.39 ms, fam=5.85) -> refund_duplicate_charge  ✅

ACT 3: 'TEACH ME' — Teaching a Brand-New Jargon Term in 1 Line (< 4 ms)
  1. Unseen codeword ('Customer is asking for a Project-Sunburst adjustment on their account'):
     Before : ABSTAIN_TO_LLM (Safe! 0.313 ms)
  2. Teach 1 example: engine.learn_from_tier2(prompt, {'action': 'refund_duplicate_charge', 'urgent': True})
  3. Next ticket ('Please apply the Project-Sunburst adjustment for this workspace'):
     • bge-small-en-v1.5 (Static)   : upgrade_plan (cos=0.73) ❌ MISROUTED!
     • System 1: Instinct (Taught)  : refund_duplicate_charge (urgent=True) in 0.512 ms ✅
```

You can also run the 3-wave teacher auto-cutover demo in [`examples/drop_in/llm_cutover_dropin.py`](examples/drop_in/llm_cutover_dropin.py).

### How Many Examples Does It Actually Take?

Here is the real progression from **`0` examples** to a compiled **`.s1m` skill**:

| Examples per Option | What Happens Inside Instinct | Traffic Handled Locally (`0.2 ms`) | Traffic Sent to Teacher LLM | End-to-End Accuracy |
| :---: | :--- | :---: | :---: | :---: |
| **`0` examples** | Cold start from option descriptions; abstains on anything non-obvious | **`0%–20%`** | `80%–100%` | **`100%`** *(Teacher answers)* |
| **`1–2` examples / option** | Instant Sherman-Morrison rank-1 RLS update (`<0.5 ms` per example) | **`40%–60%`** | `40%–60%` | **`~90%–95%`** |
| **`3–5` examples / option** | Captures domain jargon, character typos (`char 3–5` n-grams), and `#` negation scope | **`55%–85%`** | `15%–45%` | **`~93%–98%`** |
| **`8–15` examples / option** | Auto-compiles a calibrated **`.s1m` skill (`~50–120 KiB`)** with spectral `PPMI+SVD` features | **`70%–95%+`** | `5%–30%` | **`~96%–100%`** |
| **`25–50` examples / option** | Standalone `.s1m` saturates; out-of-scope inputs still cleanly rejected (`0.0%` OOS FA) | **`80%–100%`** | `0%–20%` | **`~99%–100%`** |
| **`10,000+` examples** | **Wrong tool for new gains** — linear + spectral hyperplanes plateau around `50–100` examples/option | *Plateau* | — | Use a fine-tuned model / local LLM |

---

## When to Use Instinct vs. a Local LLM or Fine-Tuned Model

We don't oversell what a `100 KiB` non-autoregressive runtime can do, and we don't undersell where it beats heavy neural models. Here is the exact architectural boundary:

| Question / Scenario | Best Tool | Why |
| :--- | :--- | :--- |
| **"I have `3 to 30` examples per option (or a live teacher LLM) and want to stop paying `500 ms` for repeatable routing, triage, tool selection, or guardrails."** | **System 1: Instinct**<br>(`~100 KiB` `.s1m`, `0.2 ms` CPU) | Offloads `70%–100%` of repetitive structured decisions in `0.2 ms` on CPU with `0` neural weights, `0.0%` out-of-scope false acceptance, character-typo resilience, and `<0.5 ms` 1-shot online fixes (`learn_from_tier2`). |
| **"I have `3` static examples per class, NO teacher model in the loop, and users use open-vocabulary synonyms (`'automobile'` when my examples only said `'car'`)."** | **Pretrained Bi-Encoder**<br>(`bge-small` / `all-MiniLM`, `70–90 MB`, `1.5–4 ms`) | Internet pretraining already knows `'automobile' == 'car'` on Day 0 (`+3% to +5%` higher static clean recall)—at the cost of `70–90 MB` weights, no `<1 ms` online updates, and misrouting `100%` of out-of-scope queries unless wrapped by Instinct's conformal gate. |
| **"The decision requires multi-hop logic, reading comprehension across paragraphs, arithmetic, or I have `50,000` complex labeled examples."** | **Local Small LLM / Fine-Tuned Decision Model**<br>(`0.5B–8B`, `20–200 ms` GPU/NPU) | No n-gram + spectral hyperplane (`10` examples or `5,000,000` examples) can reason through a 4-page contract or multi-step deduction. |
| **"How do Instinct and a Local LLM work together?"** | **`Tier 1 Instinct` (`0.2 ms`) $\to$ `Tier 2 Local LLM` (`50 ms`)** | Put Instinct in front of your local LLM via `BoundedReviewFallback`. Instinct absorbs the `70%–85%` of fast reflex decisions and rejects out-of-scope garbage in `0.2 ms`, waking up your local LLM **only** on the `15%–30%` of hard/ambiguous inputs (`is_ambiguous=True`). |

---

## Drop Into Real Projects

We installed real open-source frameworks (`semantic-router`, `langchain-core`, `FastAPI` + `Instructor`, and `poke-env` + `pokemon-showdown`) and built runnable drop-in replacements in [`examples/drop_in/`](examples/drop_in/) and [`docs/drop_in/`](docs/drop_in/README.md):

### 1. Drop Into `FastAPI` + `Instructor` / Pydantic (`POST /v1/triage`)
* **Guide & Code**: **[docs/drop_in/03_instructor_structured_outputs.md](docs/drop_in/03_instructor_structured_outputs.md)** · Run: `python examples/drop_in/instructor_structured_dropin.py`
* **How it works**: A real `FastAPI` endpoint returning a 4-field Pydantic model (`department`, `page_oncall`, `impact_tags`, `urgency_score`). In Wave 1, Instinct handles confident tickets in `0.45 ms` and escalates uncertain ones (`37.1%`) to the `Instructor` teacher. By Wave 2 & Wave 3, Instinct handles **`100.0%` (`35/35`) of tickets locally in `0.455 ms` with `0` LLM calls and `100.0%` 4-field exact match**.

### 2. Drop Into `aurelio-labs/semantic-router`
* **Guide & Code**: **[docs/drop_in/01_semantic_router.md](docs/drop_in/01_semantic_router.md)** · Run: `python examples/drop_in/semantic_router_dropin.py`
* **How it works**: Pass your existing `semantic_router.Route(name=..., utterances=...)` objects directly into `InstinctRouteLayer(routes)`. Runs **`30×` faster (`0.14 ms` vs `4.31 ms`)**, shrinks memory from `67.2 MB` ONNX to a **`93.3 KiB` `.s1m` file**, and stops `SemanticRouter` from falsely accepting `100%` of Out-of-Scope queries (**`0.0%–0.8%` OOS False Acceptance**).

### 3. Drop Into `langchain-core` / `RouteLLM` Cascades
* **Guide & Code**: **[docs/drop_in/02_routellm_cascade.md](docs/drop_in/02_routellm_cascade.md)** · Run: `python examples/drop_in/routellm_dropin.py`
* **How it works**: Plug `RunnableLambda` into your LangChain LCEL pipeline to route queries at `0.22 ms` CPU latency, survive character-typo corruption (`+3.5%` higher accuracy under typos than `all-MiniLM-L6-v2`), and learn online from frontier fallback answers.

### 4. Drop Into `Llama-Prompt-Guard` / Polarity & Security Gates
* **Guide & Code**: **[docs/drop_in/04_prompt_guard_security.md](docs/drop_in/04_prompt_guard_security.md)** · Run: `python examples/drop_in/prompt_guard_dropin.py`
* **How it works**: Pre-screen prompts in `0.21 ms` with `#` negation-scope token transformations—achieving **`0.0%` false positives** on benign technical prompts containing words like `"ignore"` or `"system"`, and **`90% -> 100%` accuracy on negation flips** (`"do not cancel, upgrade"`), where `all-MiniLM-L6-v2` gets trapped at `60.0%`.

### 5. Drop Into `poke-env` + Live Pokémon Showdown (`gen9randombattle`)
* **Guide & Code**: **[docs/drop_in/06_poke_env_showdown_agent.md](docs/drop_in/06_poke_env_showdown_agent.md)** · Run: `python benchmarks/quality/evaluate_poke_env_showdown.py --battles 50`
* **How it works**: Subclass `poke_env.player.Player` in 15 lines of Python. Starting from `0` examples (just 4 tactical option descriptions), Instinct beats `RandomPlayer` (`100%`) and `MaxBasePowerPlayer` (`98%`), and uses **online turn-by-turn adaptation (`adapt_zero_shot` + `learn_from_tier2`) to climb from `42%` (cold 0-shot) $\to$ `62%` win rate against `SimpleHeuristicsPlayer`**, compiling down to a **`23 KiB` `.s1m` reflex policy running at `0.093 ms` (`93 µs`) per turn**.

---

## Quickstart

```bash
git clone https://github.com/steph4n-gh/system1-instinct.git
cd system1-instinct
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,langchain]'

# 1. Watch 0-shot -> online learning -> 0-LLM-call .s1m cutover in 30 seconds:
python examples/drop_in/llm_cutover_dropin.py

# 2. Run the real framework drop-ins (FastAPI+Instructor, Semantic Router, LangChain, Prompt-Guard):
python examples/drop_in/instructor_structured_dropin.py
python examples/drop_in/semantic_router_dropin.py
python examples/drop_in/routellm_dropin.py
python examples/drop_in/prompt_guard_dropin.py
```

---

## Receipts & Public Benchmarks

To verify that System 1: Instinct works on **external, publicly authored datasets** (rather than only self-authored demo prompts), every benchmark below runs offline on standard academic/industry splits (`BANKING77`, `CLINC150-OOS`, `NotInject`) and live `smogon/pokemon-showdown` battles.

### 1. Head-to-Head vs. Pretrained Neural Encoders (`all-MiniLM-L6-v2` & `BAAI/bge-small-en-v1.5`)
Reproduce with `python benchmarks/quality/evaluate_drop_in_replacements.py` ([full report](benchmarks/quality/drop_in_and_adaptation_report.md)):

| Router Architecture | Model / Artifact Size | `BANKING77` (32c Clean) | `BANKING77` (32c Typo Stress) | `CLINC150-OOS` False Accept | Negation / Polarity Acc | P50 CPU Latency | `<1 ms` Online RLS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Linear Baseline** (`Unigram TF-IDF`) | `85 KiB` (`0` weights) | 77.3% | 66.0% | 3.3% | 70.0% | `0.18 ms` | No |
| **`all-MiniLM-L6-v2`** (`22M` Dense Bi-Encoder) | `91.1 MB` ONNX | 91.0% | 70.7% (`-20.3%` drop) | 8.0% | 60.0% | `1.32 ms` (`~10 ms` PyTorch) | No |
| **`BAAI/bge-small-en-v1.5`** (`33M` Dense Bi-Encoder) | `67.2 MB` ONNX | **94.5%** | **77.0%** (`-17.5%` drop) | 5.0% (cal) / 100% (`@0.5`) | 90.0% | `2.96 ms` (`~14 ms` PyTorch) | No |
| **System 1: Instinct (Static `.s1m`)** | **`120 KiB`** (`0` weights) | **87.9%** | **74.2%** (`+3.5%` vs MiniLM) | **0.0%** (`0/120`) | **90.0%** | **`0.22 ms`** | **Yes** |
| **System 1: Instinct (`+ Online Flywheel`)** | **`120 KiB`** (`0` weights) | **91.4%** (`+0.4%` vs MiniLM) | **74.2%** | **0.0%** (`0/120`) | **100.0%** | **`0.22 ms`** | **Yes** |

### 2. Sample-Efficiency Curves on Public Datasets (`0` to `40` Examples / Class)
Measured on `24` intents from `CLINC150` (`120` test queries) and `BANKING77` (`192` test queries):

| Examples / Class | Total Examples (`24` classes) | `CLINC150` Raw Online Acc | `CLINC150` Local Offload % | `CLINC150` End-to-End Acc | `BANKING77` Raw Online Acc | `BANKING77` Local Offload % | `BANKING77` End-to-End Acc |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0` (Cold)** | **`0`** | `61.7%` | `0.0%` *(100% teacher)* | **`100.0%`** | `53.6%` | `0.0%` *(100% teacher)* | **`100.0%`** |
| **`1` / class** | **`24`** | `65.8%` | **`59.2%` local** | **`88.3%`** | `57.3%` | **`42.2%` local** | **`92.2%`** |
| **`3` / class** | **`72`** | `72.5%` | **`60.8%` local** | **`90.8%`** | `65.6%` | **`48.4%` local** | **`93.2%`** |
| **`5` / class** | **`120`** | `82.5%` | **`85.0%` local** | **`90.8%`** | `70.3%` | **`50.0%` local** | **`95.3%`** |
| **`8` / class** | **`192`** | `85.8%` | **`65.0%` local** *(compiled)* | **`98.3%`** | `72.4%` | **`56.2%` local** *(compiled)* | **`96.9%`** |
| **`10` / class** | **`240`** | `85.8%` (`.s1m`) | **`70.0%` local** *(compiled)* | **`99.2%`** | `77.1%` (`.s1m`) | **`64.6%` local** *(compiled)* | **`94.8%`** |
| **`24–40` / class** | **`576–960`** | **`90.8%`** | **`87.5%` local** *(compiled)* | **`99.2%`** | **`90.1%`** (`.s1m`) | **`70.3%` local** *(compiled)* | **`100.0%`** |

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

## Guides & Documentation

| What You Want to Do | Guide |
| :--- | :--- |
| **Drop Instinct into `Semantic Router`, `RouteLLM`, `Instructor`, `Prompt-Guard`, or `poke-env`** | [Step-by-Step Drop-In Guides (`docs/drop_in/README.md`)](docs/drop_in/README.md) |
| **Observe a live OpenAI / Anthropic / Gemini teacher and auto-cutover** | [LLM Auto-Cutover Guide (`docs/drop_in/05_llm_observe_and_cutover.md`)](docs/drop_in/05_llm_observe_and_cutover.md) · [`TypeSafeClient` (`docs/typesafe.md`)](docs/typesafe.md) |
| **Teach, qualify, and version skills with `TeachingSession`** | [Correcting Skills](docs/guides/correcting_skills.md) · [Training Experts](docs/guides/training_experts.md) |
| **Serve skills via LangChain, MCP, FastAPI/ASGI, or gRPC** | [Integrations Guide](docs/integrations.md) · [Policy Guard & Ed25519 Ledger](docs/guides/policy_guard.md) |
| **Read the math & architecture specs** | [Technical Spec](docs/architecture/technical_specification.md) · [Whitepaper](docs/paper/system1_whitepaper.md) · [Conformal Math](docs/paper/conformal_gating.md) |

**Version 1.0.0:** [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Apache License 2.0](LICENSE)
