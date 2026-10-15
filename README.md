<p align="center">
  <img src="assets/system1-logo.jpg" alt="System 1: Instinct logo" width="128" />
</p>

# System 1: Instinct

**Stop paying an LLM `500 ms` and an API call to make the same repeatable decision 10,000 times a day.**

**System 1: Instinct** is a sub-millisecond (`0.1 to 0.4 ms`), zero-weight (`~100 KiB`) decision runtime for Python. Place it in front of your LLM, local model, or tool router:
- **Start at `0` examples.** Define a typed schema with `ChoiceField`, `BooleanField`, `MultiChoiceField`, and `ScoreField`.
- **Teach `3 to 10` real examples per option.** Update weights online as your LLM answers live traffic, or pass labeled examples directly.
- **Run `70% to 95%+` of repeatable traffic locally in `0.2 ms` on CPU.** Escalate ambiguous or unfamiliar inputs (`is_ambiguous=True`) back to your LLM.

[![CI](https://github.com/steph4n-gh/system1-instinct/actions/workflows/ci.yml/badge.svg)](https://github.com/steph4n-gh/system1-instinct/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

[How teaching works (0 to 10 examples)](#go-from-zero-examples-to-local-execution) · [Instinct vs. local LLMs](#choose-between-instinct-and-a-local-llm) · [Drop-in replacements](#drop-instinct-into-existing-projects) · [Benchmarks and receipts](#verify-public-benchmarks-and-receipts) · [Documentation](docs/README.md)

---

## Go from zero examples to local execution

A raw zero-shot classifier is not enough on its own. If you force a zero-shot model to guess across 20 or more intents with zero examples, it reaches only `53% to 62%` accuracy. Nobody wants a `53%` router in production.

**System 1: Instinct** combines conformal abstention with sub-millisecond online updates so you can deploy safely from the first request.

### Teach a schema in five seconds

Define your decision schema with `ChoiceField`, `BooleanField`, `MultiChoiceField`, or `ScoreField`. At **`0` examples**, `SystemOneEngine` abstains (`is_ambiguous=True`) on uncertain requests so it never guesses blindly on day zero. After you teach it **`3 to 10` examples per option** from application outcomes or an LLM teacher, the engine generalizes to unseen phrasings in `< 1 ms` and still rejects out-of-scope inputs:

```python
from system1 import BooleanField, BoundedReviewFallback, ChoiceField, DecisionSchema, SystemOneEngine

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

# 1. BEFORE (0 examples): Cold-start strict mode abstains instead of guessing blindly
engine = SystemOneEngine(SupportTriage, strict_mode=True, use_cache=False)
unseen_ticket = "Production webhook endpoint returning 502 bad gateway and API timeout"
before = engine.decide(unseen_ticket, strict=True, record_receipt=False)
print(before.is_ambiguous)  # -> True (escalates to your fallback or LLM on Day 0)

# 2. TEACH (< 0.5 ms per lesson, $0 LLM cost): Teach from app outcomes or LLM fallback
engine.learn_from_outcome(
    "Charged twice on my Visa invoice, please refund the duplicate billing charge",
    outcome={"queue": "refunds", "urgent": False},
)
engine.learn_from_outcome(
    "Locked out of SSO login because my 2FA reset code is not arriving",
    outcome={"queue": "account_access", "urgent": True},
)
engine.learn(
    "API timeout and 502 bad gateway crashing production startup after deploy",
    {"queue": "technical_bug", "urgent": True},
)

# Or wrap any LLM callable so uncertain queries teach Instinct automatically:
fallback = BoundedReviewFallback(
    engine,
    review_fn=lambda prompt: {"queue": "technical_bug", "urgent": True},
    auto_online_update=True,
)
fallback.decide("Webhook delivery failing with 502 bad gateway timeout in production")

# 3. AFTER: Unseen phrasing resolves locally in < 1 ms, and out-of-scope input abstains
after = engine.decide(unseen_ticket, strict=True, record_receipt=False)
print(after.is_ambiguous, after.values)
# -> False {'queue': 'technical_bug', 'urgent': True} (0 LLM calls, < 1 ms on CPU)

oos = engine.decide("What is the best recipe for sourdough bread?", strict=True, record_receipt=False)
print(oos.is_ambiguous, oos.conformal_sets)
# -> True {'queue': [], 'urgent': []} (safe out-of-scope abstention)

# 4. Save and reload a portable ~12 KiB .s1m skill in two lines (preserves strict calibration)
engine.save("triage.s1m")
reloaded = SystemOneEngine.load("triage.s1m")
```

### Choose how to teach Instinct in production

An LLM teacher (`GPT-4o`, `Claude`, or `Gemini`) guesses from text before execution happens. Your application context is often a better teacher because it observes what worked after execution for `$0` and `0` LLM tokens:

| Teaching path | API entrypoint | Cost and speed | Best for |
| :--- | :--- | :---: | :--- |
| **1. Application context and outcomes** | `engine.learn_from_outcome(prompt, outcome=...)`<br>`engine.learn(prompt, labels)` | **`$0` LLM cost**<br>`< 0.5 ms` RLS update | Ground-truth outcomes: UI clicks, user edits, tool exit codes (`200 OK`), support ticket resolutions, or game rewards. |
| **2. Shadowing an LLM teacher** | `BoundedReviewFallback(engine, review_fn=..., auto_online_update=True)`<br>`TypeSafeClient(...)` | **`0` manual labeling**<br>Auto-cuts over to `.s1m` | Wrapping an OpenAI, Anthropic, or local LLM call so uncertain requests teach Instinct automatically. |
| **3. Historical logs or unlabeled traffic** | `SystemOneCompiler(Schema).compile(exemplars)`<br>`engine.adapt_zero_shot(unlabeled_prompts)` | **Offline CPU fit**<br>`< 120 KiB` `.s1m` file | Compiling `10 to 50` historical logs per option into a versioned `.s1m` artifact, or adapting to domain jargon without labels. |

To compare online teaching against `BAAI/bge-small-en-v1.5` (`67.2 MB` ONNX), run [`examples/showcase_break_and_teach.py`](examples/showcase_break_and_teach.py). Pass `--interactive` to enter your own prompts and teach the engine in one line:

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
  2. Teach 1 example: engine.learn(prompt, {'action': 'refund_duplicate_charge', 'urgent': True})
  3. Next ticket ('Please apply the Project-Sunburst adjustment for this workspace'):
     • bge-small-en-v1.5 (Static)   : upgrade_plan (cos=0.73) ❌ MISROUTED!
     • System 1: Instinct (Taught)  : refund_duplicate_charge (urgent=True) in 0.512 ms ✅
```

Run these end-to-end replay scripts to inspect real-world behavior:
- **Live GitHub issue triage (`astral-sh/uv` and `facebook/react` OOS).** Run [`examples/dogfood/live_github_triage.py`](examples/dogfood/live_github_triage.py), which uses the snapshot in [`examples/dogfood/uv_live_issues_snapshot.json`](examples/dogfood/uv_live_issues_snapshot.json). The script replays `260` chronologically ordered `astral-sh/uv` issues (`#491` through `#22542`) plus `25` `facebook/react` out-of-scope issues, with zero synthetic strings and zero temporal leakage. Traffic progresses from **`0` examples (`0.0%` offload, `100%` escalated)** to **online warm-up (`40.0%` offload)**. Compiling `.s1m` checkpoints reaches **`56.7%` offload at `15 per class`, `76.7%` at `25 per class`, and `93.3%` local offload at `40 per class` in `0.23 ms` with `100.0%` local precision, `96.0%` OOS abstention, and `100%` `.s1m` reload parity**:

```bash
python examples/dogfood/live_github_triage.py
# Or pull fresh live issues on demand via GitHub CLI (`gh api`):
python examples/dogfood/live_github_triage.py --live --repo astral-sh/uv
```

- **Live US FDA enforcement recall triage (`openFDA` and `astral-sh/uv` OOS).** Run [`examples/dogfood/live_fda_recall_triage.py`](examples/dogfood/live_fda_recall_triage.py), which uses the snapshot in [`examples/dogfood/fda_live_recalls_snapshot.json`](examples/dogfood/fda_live_recalls_snapshot.json). The script replays `297` chronologically ordered US FDA enforcement recall reports (`20121212` through `20261007`, `33 per cell` across `food`, `drug`, and `device` crossed with `Class I`, `Class II`, and `Class III`) plus `25` `astral-sh/uv` out-of-scope software issues, with zero synthetic strings and zero temporal leakage. Each decision predicts a **3-field multi-head schema** (`regulatory_center`, `hazard_class`, and `urgent_class_i_escalation`). Traffic progresses from **`0` examples (`0.0%` offload, `100%` escalated)** to **online warm-up (`10.0%` offload at `100.0%` precision)**. Compiling `.s1m` checkpoints reaches **`51.4%` offload at `12 per cell`, `61.1%` at `18 per cell`, and `70.8%` local offload at `25 per cell` in `0.30 ms` with `98.0%` joint 3-field precision, `100.0%` center precision, `100.0%` OOS abstention, and `100%` `.s1m` reload parity**:

```bash
python examples/dogfood/live_fda_recall_triage.py
# Or pull fresh live enforcement recall reports on demand from api.fda.gov:
python examples/dogfood/live_fda_recall_triage.py --live
```

- **Three-wave teacher auto-cutover demo.** Run [`examples/drop_in/llm_cutover_dropin.py`](examples/drop_in/llm_cutover_dropin.py) to observe an upstream teacher and cut over to a local `.s1m` skill.

### Measure sample efficiency from 0 to 50 examples

The table below shows how local offload and accuracy scale from **`0` examples** to a compiled **`.s1m` skill**:

| Examples per option | What happens inside Instinct | Traffic handled locally (`0.2 ms`) | Traffic sent to teacher LLM | End-to-end accuracy |
| :---: | :--- | :---: | :---: | :---: |
| **`0` examples** | Cold start from option descriptions. Abstains on non-obvious inputs. | **`0% to 20%`** | `80% to 100%` | **`100%`** *(Teacher answers)* |
| **`1 to 2` examples per option** | Sherman-Morrison rank-1 RLS update (`<0.5 ms` per example). | **`40% to 60%`** | `40% to 60%` | **`~90% to 95%`** |
| **`3 to 5` examples per option** | Captures domain jargon, character typos (`char 3 to 5` n-grams), and `#` negation scope. | **`55% to 85%`** | `15% to 45%` | **`~93% to 98%`** |
| **`8 to 15` examples per option** | Auto-compiles a calibrated **`.s1m` skill (`~50 to 120 KiB`)** with spectral `PPMI+SVD` features. | **`70% to 95%+`** | `5% to 30%` | **`~96% to 100%`** |
| **`25 to 50` examples per option** | Standalone `.s1m` saturates. Out-of-scope inputs still abstain (`0.0%` OOS false accept). | **`80% to 100%`** | `0% to 20%` | **`~99% to 100%`** |
| **`10,000+` examples** | **Wrong tool for new gains.** Linear and spectral hyperplanes plateau around `50 to 100` examples per option. | *Plateau* | None | Use a fine-tuned model or local LLM |

> **Scope and fallible-teacher boundary.** The progression above summarizes `CLINC150` and `BANKING77` adaptation curves with an oracle teacher fallback at $\alpha = 0.10$ marginal conformal gating. We also tested a **fallible teacher** (`10%` label noise on `64`-class `HWU64`) on `1,394` fresh qualification items. Under a strict **`1%` (`0.01`) family-wise conditional local error budget**, neither Instinct nor a classical logistic classifier can statistically certify `<= 1%` upper-bound risk. At a **`5%` (`0.05`) local error budget**, all generalizing methods qualify on `5 of 5` seeds. `instinct_lexical` offloads `24.41%` vs. `simple_classifier`'s `20.46%` at `1.75%` error, and `instinct_hybrid` reaches the lowest local error rate at `1.36%` with `5 of 5` `.s1m` reloadability. See [Fresh HWU64 traffic with a fallible teacher (`benchmarks/heldout_traffic/REPORT.md`)](benchmarks/heldout_traffic/REPORT.md).

---

## Choose between Instinct and a local LLM

A `100 KiB` non-autoregressive runtime has clear boundaries. Use the table below to pick the right model for your workload:

| Scenario | Best tool | Rationale |
| :--- | :--- | :--- |
| **"I have `3 to 30` examples per option (or a live teacher LLM) and want to stop paying `500 ms` for repeatable routing, triage, tool selection, or guardrails."** | **System 1: Instinct**<br>(`~100 KiB` `.s1m`, `0.2 ms` CPU) | Offloads `70% to 100%` of repetitive structured decisions in `0.2 ms` on CPU with `0` neural weights, `0.0%` out-of-scope false acceptance, character-typo resilience, and `<0.5 ms` rank-1 online updates (`learn_from_tier2`). |
| **"I have `3` static examples per class, no teacher model in the loop, and users use open-vocabulary synonyms (`'automobile'` when my examples only said `'car'`)."** | **Pretrained bi-encoder**<br>(`bge-small` or `all-MiniLM`, `70 to 90 MB`, `1.5 to 4 ms`) | Pretraining already maps `'automobile'` to `'car'` on day zero (`+3% to +5%` higher static clean recall). The trade-off is `70 to 90 MB` of weights, no `<1 ms` online updates, and `100%` false acceptance on out-of-scope queries unless wrapped by a conformal gate. |
| **"The decision requires multi-hop logic, reading comprehension across paragraphs, arithmetic, or `50,000` complex labeled examples."** | **Local small LLM or fine-tuned decision model**<br>(`0.5B to 8B`, `20 to 200 ms` GPU or NPU) | An n-gram and spectral hyperplane cannot reason through a four-page contract or perform multi-step deduction. |
| **"How do Instinct and a local LLM work together?"** | **`Tier 1 Instinct` (`0.2 ms`) to `Tier 2 Local LLM` (`50 ms`)** | Put Instinct in front of your local LLM via `BoundedReviewFallback`. Instinct handles `70% to 85%` of fast decisions and rejects out-of-scope traffic in `0.2 ms`, waking your local LLM only on the `15% to 30%` of ambiguous inputs (`is_ambiguous=True`). |

---

## Drop Instinct into existing projects

The [`examples/drop_in/`](examples/drop_in/) and [`docs/drop_in/`](docs/drop_in/README.md) directories contain runnable drop-in replacements for `semantic-router`, `langchain-core`, `FastAPI` with `Instructor`, and `poke-env` with `pokemon-showdown`:

### 1. Replace `Instructor` in a `FastAPI` endpoint (`POST /v1/triage`)
- **Guide and code.** Read [`docs/drop_in/03_instructor_structured_outputs.md`](docs/drop_in/03_instructor_structured_outputs.md) and run `python examples/drop_in/instructor_structured_dropin.py`.
- **Behavior.** A `FastAPI` endpoint returns a four-field Pydantic model (`department`, `page_oncall`, `impact_tags`, and `urgency_score`). In Wave 1, Instinct handles confident tickets in `0.45 ms` and escalates uncertain ones (`37.1%`) to the `Instructor` teacher. In Wave 2 and Wave 3, Instinct handles **`100.0%` (`35 of 35`) of tickets locally in `0.455 ms` with `0` LLM calls and `100.0%` four-field exact match**.

### 2. Replace `aurelio-labs/semantic-router`
- **Guide and code.** Read [`docs/drop_in/01_semantic_router.md`](docs/drop_in/01_semantic_router.md) and run `python examples/drop_in/semantic_router_dropin.py`.
- **Behavior.** Pass your existing `semantic_router.Route(name=..., utterances=...)` objects directly into `InstinctRouteLayer(routes)`. Instinct runs **`30x` faster (`0.14 ms` vs. `4.31 ms`)**, cuts artifact size from `67.2 MB` ONNX to a **`93.3 KiB` `.s1m` file**, and reduces out-of-scope false acceptance from `100%` to **`0.0% to 0.8%`**.

### 3. Replace `RouteLLM` cascades in `langchain-core`
- **Guide and code.** Read [`docs/drop_in/02_routellm_cascade.md`](docs/drop_in/02_routellm_cascade.md) and run `python examples/drop_in/routellm_dropin.py`.
- **Behavior.** Plug `RunnableLambda` into your LangChain LCEL pipeline to route queries at `0.22 ms` CPU latency, tolerate character-typo corruption (`+3.5%` higher accuracy under typos than `all-MiniLM-L6-v2`), and learn online from frontier fallback answers.

### 4. Replace `Llama-Prompt-Guard` for polarity and security checks
- **Guide and code.** Read [`docs/drop_in/04_prompt_guard_security.md`](docs/drop_in/04_prompt_guard_security.md) and run `python examples/drop_in/prompt_guard_dropin.py`.
- **Behavior.** Pre-screen prompts in `0.21 ms` with `#` negation-scope token transformations. Instinct reaches **`0.0%` false positives** on benign technical prompts containing words like `"ignore"` or `"system"`, and **`90%` to `100%` accuracy on negation flips** (`"do not cancel, upgrade"`), where `all-MiniLM-L6-v2` scores `60.0%`.

### 5. Run a `poke-env` agent on live Pokémon Showdown (`gen9randombattle`)
- **Guide and code.** Read [`docs/drop_in/06_poke_env_showdown_agent.md`](docs/drop_in/06_poke_env_showdown_agent.md) and run `python benchmarks/quality/evaluate_poke_env_showdown.py --battles 50`.
- **Behavior.** Subclass `poke_env.player.Player` in 15 lines of Python. Starting from `0` examples and four tactical option descriptions, Instinct beats `RandomPlayer` (`100%`) and `MaxBasePowerPlayer` (`98%`). Online turn-by-turn updates (`adapt_zero_shot` and `learn_from_tier2`) raise its win rate from **`42%` (cold zero-shot) to `62%` against `SimpleHeuristicsPlayer`**, compiling into a **`23 KiB` `.s1m` policy that runs in `0.093 ms` (`93 µs`) per turn**.

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

# 2. Run the framework drop-ins (FastAPI+Instructor, Semantic Router, LangChain, Prompt-Guard):
python examples/drop_in/instructor_structured_dropin.py
python examples/drop_in/semantic_router_dropin.py
python examples/drop_in/routellm_dropin.py
python examples/drop_in/prompt_guard_dropin.py
```

---

## Verify public benchmarks and receipts

Every benchmark below runs offline on public datasets (`BANKING77`, `CLINC150-OOS`, and `NotInject`) and live `smogon/pokemon-showdown` battles.

### 1. Compare against pretrained neural encoders (`all-MiniLM-L6-v2` and `BAAI/bge-small-en-v1.5`)
To reproduce this table, run `python benchmarks/quality/evaluate_drop_in_replacements.py` ([full report](benchmarks/quality/drop_in_and_adaptation_report.md)):

| Router architecture | Artifact size | `BANKING77` (32 classes clean) | `BANKING77` (32 classes typo stress) | `CLINC150-OOS` false accept | Negation and polarity accuracy | P50 CPU latency | `<1 ms` online RLS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical linear baseline** (`Unigram TF-IDF`) | `85 KiB` (`0` weights) | 77.3% | 66.0% | 3.3% | 70.0% | `0.18 ms` | No |
| **`all-MiniLM-L6-v2`** (`22M` dense bi-encoder) | `91.1 MB` ONNX | 91.0% | 70.7% (`-20.3%` drop) | 8.0% | 60.0% | `1.32 ms` (`~10 ms` PyTorch) | No |
| **`BAAI/bge-small-en-v1.5`** (`33M` dense bi-encoder) | `67.2 MB` ONNX | **94.5%** | **77.0%** (`-17.5%` drop) | 5.0% calibrated, 100% at `0.5` | 90.0% | `2.96 ms` (`~14 ms` PyTorch) | No |
| **System 1: Instinct (Static `.s1m`)** | **`120 KiB`** (`0` weights) | **87.9%** | **74.2%** (`+3.5%` vs. MiniLM) | **0.0%** (`0 of 120`) | **90.0%** | **`0.22 ms`** | **Yes** |
| **System 1: Instinct (`+ Online correction`)** | **`120 KiB`** (`0` weights) | **91.4%** (`+0.4%` vs. MiniLM) | **74.2%** | **0.0%** (`0 of 120`) | **100.0%** | **`0.22 ms`** | **Yes** |

### 2. Inspect sample-efficiency curves on public datasets (`0` to `40` examples per class)
Measured on `24` intents from `CLINC150` (`120` test queries) and `BANKING77` (`192` test queries):

| Examples per class | Total examples (`24` classes) | `CLINC150` raw online accuracy | `CLINC150` local offload % | `CLINC150` end-to-end accuracy | `BANKING77` raw online accuracy | `BANKING77` local offload % | `BANKING77` end-to-end accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0` (Cold)** | **`0`** | `61.7%` | `0.0%` *(100% teacher)* | **`100.0%`** | `53.6%` | `0.0%` *(100% teacher)* | **`100.0%`** |
| **`1` per class** | **`24`** | `65.8%` | **`42.5%` local** | **`92.5%`** | `57.3%` | **`27.6%` local** | **`96.4%`** |
| **`3` per class** | **`72`** | `72.5%` | **`36.7%` local** | **`97.5%`** | `65.6%` | **`27.6%` local** | **`97.9%`** |
| **`5` per class** | **`120`** | `82.5%` | **`37.5%` local** | **`97.5%`** | `70.3%` | **`33.9%` local** | **`97.4%`** |
| **`8` per class** | **`192`** | `85.8%` | **`40.0%` local** | **`99.2%`** | `72.4%` | **`40.1%` local** | **`96.9%`** |
| **`10` per class** | **`240`** | `85.8%` (`.s1m`) | **`80.0%` local** *(compiled)* | **`97.5%`** | `77.1%` (`.s1m`) | **`78.1%` local** *(compiled)* | **`90.1%`** |
| **`all` (`10 to 25` per class)** | **`240 to 600`** | **`87.5%`** (`.s1m`) | **`79.2%` local** *(compiled)* | **`95.8%`** | **`90.1%`** (`.s1m`) | **`68.8%` local** *(compiled)* | **`99.5%`** |

### 3. Run the six-category offline benchmark suite
To reproduce this table offline (`block_network()`), run `python benchmarks/quality/evaluate_instinct_gauntlet.py` ([full report](benchmarks/quality/instinct_gauntlet_report.md)):

| Category | Workload | Classical linear baseline | System 1: Instinct (Static) | System 1: Instinct (+ Online replay) |
| :--- | :--- | :---: | :---: | :---: |
| **1. High-cardinality intent routing** | `BANKING77` (32 intents, clean and typo) | 77.3% clean, 66.0% typo | **87.9%** clean, **74.2%** typo | **91.4%** |
| **1b. Full 77-class scaling** | `BANKING77` (77 intents, clean and typo) | 62.0% clean, 45.5% typo | **79.5%** clean, **58.3%** typo | **88.1%** |
| **2. Multi-domain and OOS rejection** | `CLINC150-OOS` (in-scope accuracy and OOS false accept) | 84.7%, 3.3% false accept | **86.1%**, **0.0% false accept** | **99.3%**, **0.0% false accept** |
| **3. Injection and NotInject negatives** | `NotInject` benign-trigger FPR and polarity accuracy | 10.0% FPR, 70.0% accuracy | **0.0% FPR**, **90.0% accuracy** | **0.0% FPR**, **100.0% accuracy** |
| **4. Authored drift probes (106)** | `banking_support`, `sms_triage`, and `model_routing` coverage | 86.2%, 77.5%, 77.8% | **100.0%**, **82.5%**, **88.9%** | **100.0%**, **87.5%**, **94.4%** |
| **5. Multi-field structured triage** | 4-field joint accuracy (`ChoiceField`, `BooleanField`, `MultiChoiceField`, and `ScoreField`) | 97.1% | **100.0%** (`0.40 ms`) | **100.0%** (`0%` Wave 3 escalation) |
| **6. Zero-shot cold-start routing** | `intent_routing` and `security_triage` (0-shot) | 51.0%, 50.0% | **81.0%**, **85.0%** | **100.0%** |

> **Closed-loop replay vs. fresh noisy-teacher traffic.** In the table above, `+ Online replay` columns measure closed-loop oracle correction and replay on the evaluated prompt set. The end-to-end curves in Section 2 use an oracle fallback at $\alpha = 0.10$. We also evaluated **fresh held-out `HWU64` traffic (`64` intents) taught by a `10%`-noisy teacher under strict `1%` (`0.01`) and standard `5%` (`0.05`) family-wise local accepted-error gates**. See [`benchmarks/heldout_traffic/REPORT.md`](benchmarks/heldout_traffic/REPORT.md) and its [protocol and verifier](benchmarks/heldout_traffic/README.md).

---

## Read the documentation

| Goal | Guide |
| :--- | :--- |
| **Drop Instinct into `Semantic Router`, `RouteLLM`, `Instructor`, `Prompt-Guard`, or `poke-env`** | [Drop-in replacement guides (`docs/drop_in/README.md`)](docs/drop_in/README.md) |
| **Observe a live OpenAI, Anthropic, or Gemini teacher and auto-cutover** | [LLM auto-cutover guide (`docs/drop_in/05_llm_observe_and_cutover.md`)](docs/drop_in/05_llm_observe_and_cutover.md) · [`TypeSafeClient` reference (`docs/typesafe.md`)](docs/typesafe.md) |
| **Teach, qualify, and version skills with `TeachingSession`** | [Correct, compare, and adopt a skill](docs/guides/correcting_skills.md) · [Teach and compile a decision skill](docs/guides/training_experts.md) |
| **Connect skills to LangChain, MCP, FastAPI, ASGI, or gRPC** | [Application integrations](docs/integrations.md) · [Add an explicit tool policy](docs/guides/policy_guard.md) |
| **Inspect the math and architecture specifications** | [Technical specification](docs/architecture/technical_specification.md) · [Whitepaper](docs/paper/system1_whitepaper.md) · [Conformal gating notes](docs/paper/conformal_gating.md) |

**Version 1.0.0:** [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Apache License 2.0](LICENSE)
