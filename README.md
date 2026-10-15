<p align="center">
  <img src="assets/system1-logo.jpg" alt="System 1: Instinct logo" width="128" />
</p>

# System 1: Instinct

**Stop paying an LLM `500 ms` and an API call to make the same repeatable decision 10,000 times a day.**

**System 1: Instinct** is a sub-millisecond (`0.1 to 0.4 ms`) decision runtime for Python with **no neural network weights** (`~100 KiB` `.s1m` skill). Place it in front of your LLM, local model, or tool router:
- **Start at `0` examples.** Define a typed schema with `ChoiceField`, `BooleanField`, `MultiChoiceField`, and `ScoreField`.
- **Teach `3 to 10` real examples per option.** Update linear and spectral weights online as your LLM answers live traffic, or pass labeled examples directly.
- **Run repeatable traffic locally in `0.2 ms` on CPU.** Escalate ambiguous or unfamiliar inputs (`is_ambiguous=True`) back to your LLM.

[![CI](https://github.com/steph4n-gh/system1-instinct/actions/workflows/ci.yml/badge.svg)](https://github.com/steph4n-gh/system1-instinct/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

[How teaching works](#go-from-zero-examples-to-local-execution) · [Instinct vs. local LLMs](#choose-between-instinct-and-a-local-llm) · [Drop-in replacements](#drop-instinct-into-existing-projects) · [Benchmarks and receipts](#verify-public-benchmarks-and-receipts) · [Documentation](docs/README.md)

---

## Go from zero examples to local execution

A raw zero-shot classifier is not enough on its own. If you force a zero-shot model to guess across 20 or more intents with zero examples, it gets roughly half of them wrong. Nobody wants an uncalibrated router guessing in production.

**System 1: Instinct** combines conformal abstention with sub-millisecond online updates so you can deploy safely from the first request.

### Teach a schema in five seconds

Define your decision schema with `ChoiceField`, `BooleanField`, `MultiChoiceField`, or `ScoreField`. At **`0` examples**, `SystemOneEngine` abstains (`is_ambiguous=True`) on uncertain requests so it never guesses blindly on day zero. After you teach it a few examples per option from application outcomes or an LLM teacher, the engine generalizes to unseen phrasings in `< 1 ms` and still rejects out-of-scope inputs:

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

### Evaluate and recalibrate a skill on held-out traffic

Before promoting a `.s1m` skill or adjusting your abstention target `alpha`, run `system1.evaluate()` (or `system1 eval` from the CLI) on a held-out JSONL or JSON dataset. It reports overall accuracy alongside selective accepted accuracy (`Acc(Acc)`), separating local fast-path coverage from in-distribution conformal ambiguity (`Ambiguity Abstain`) and out-of-distribution rejection (`Unfamiliar (OOD)`). To refit temperature scaling and conformal quantiles on fresh calibration rows without retraining decision weights (`weights`, `biases`, `P`, `B`), call `engine.recalibrate(calib_rows, alpha=0.10)` or pass `--recalibrate`:

```python
import system1

recal = reloaded.recalibrate("calib.jsonl", alpha=0.10)
reloaded.save("fda_triage_recal.s1m")

report = system1.evaluate("fda_triage_recal.s1m", "heldout.jsonl", alpha=0.10)
print(report.summary_table())
```

```bash
system1 eval fda_triage.s1m heldout.jsonl --recalibrate calib.jsonl -o fda_triage_recal.s1m --alpha 0.10
```

```text
[SYSTEM 1 EVALUATION REPORT] EvaluationReport Schema: FdaRecallTriageSchema (alpha=0.100, n=80)
Recalibration:       ECE 0.0626 -> 0.0575 | NLL 0.2180 -> 0.1767 | T=0.400, qhat=0.4939
--------------------------------------------------------------------------------------------
Overall Accuracy:    76.25%   | Accepted Accuracy:   90.20%   | Conformal Coverage:  71.25%
Fast-Path Coverage:  63.75%   | Ambiguity Abstain:   23.75%   | Unfamiliar (OOD):    12.50%
Calibration ECE:     0.0617    | Brier Score:         0.1787    | Negative Log-Lik:    0.3216
Avg Set Size:        0.97      | Singleton Rate:      79.58%   | Latency p50 / p95:   0.324 / 0.436 ms
Deferral Breakdown:  conformal_ambiguity=19, low_feature_familiarity=5, low_semantic_familiarity=5
--------------------------------------------------------------------------------------------
Field                   Acc  Acc(Acc)  ConfCov  SetSz      ECE    Brier      NLL        Cov/Abs/Unf
--------------------------------------------------------------------------------------------
hazard_class         81.2%    91.4%   81.2%   1.10   0.0935   0.2913   0.5452        72%/18%/10%
regulatory_center    97.5%   100.0%   81.2%   0.81   0.0173   0.0312   0.0610         81%/8%/11%
urgent_class_i_escalation   86.2%    88.2%   82.5%   1.00   0.0742   0.2137   0.3588          85%/8%/8%
--------------------------------------------------------------------------------------------
```

### Choose how to teach Instinct in production

An LLM teacher (`GPT-4o`, `Claude`, or `Gemini`) guesses from text before execution happens. Your application context is often a better teacher because it observes what worked after execution for `$0` and `0` LLM tokens:

| Teaching path | API entrypoint | Cost and speed | Best for |
| :--- | :--- | :---: | :--- |
| **1. Application context and outcomes** | `engine.learn_from_outcome(prompt, outcome=...)`<br>`engine.learn(prompt, labels)` | **`$0` LLM cost**<br>`< 0.5 ms` RLS update | Ground-truth outcomes: UI clicks, user edits, tool exit codes (`200 OK`), support ticket resolutions, or game rewards. |
| **2. Shadowing an LLM teacher** | `BoundedReviewFallback(engine, review_fn=..., auto_online_update=True)`<br>`TypeSafeClient(...)` | **`0` manual labeling**<br>Auto-cuts over to `.s1m` | Wrapping an OpenAI, Anthropic, or local LLM call so uncertain requests teach Instinct automatically. |
| **3. Historical logs or unlabeled traffic** | `SystemOneCompiler(Schema).compile(exemplars)`<br>`engine.adapt_zero_shot(unlabeled_prompts)` | **Offline CPU fit**<br>`< 120 KiB` `.s1m` file | Compiling historical logs per option into a versioned `.s1m` artifact, or adapting to domain jargon without labels. |

To compare online teaching against `BAAI/bge-small-en-v1.5` (`67.2 MB` ONNX), run [`examples/showcase_break_and_teach.py`](examples/showcase_break_and_teach.py). Pass `--interactive` to enter your own prompts and teach the engine in one line:

```bash
python examples/showcase_break_and_teach.py
# Or launch the interactive REPL:
python examples/showcase_break_and_teach.py --interactive
```

```text
[Setup from 6 examples/option (30 total teacher logs)]
  • Neural Vector Router (`BAAI/bge-small-en-v1.5`) : 67,200 KiB ONNX | Init:  208.8 ms
  • System 1: Instinct (`.s1m` compiled skill)      :   48.6 KiB .s1m | Init:   27.7 ms

ACT 1: ROUTINE TRAFFIC — Replacing 450 ms LLM Calls in ~0.2 ms ($0 API Cost)
  Query: 'Please cancel my monthly subscription and stop billing me' -> LOCAL .s1m (0.356 ms): cancel_subscription ✅
  Query: 'My Visa card was billed twice for the same invoice today'  -> LOCAL .s1m (0.245 ms): refund_duplicate_charge ✅

ACT 2: 'BREAK ME' — 3 Real-World Curveballs Where Static Vector Routers Fail
  [Trap 1: Negation Flip] 'Please do NOT cancel my subscription, I want to upgrade to Pro instead'
    • bge-small-en-v1.5 (4.23 ms, cos=0.80) -> cancel_subscription      ❌ MISROUTED!
    • System 1: Instinct (0.17 ms, fam=13.95) -> upgrade_plan             ✅
  [Trap 2: Out-of-Scope Garbage] 'What is the best temperature to bake sourdough bread in a Dutch oven?'
    • bge-small-en-v1.5 (3.81 ms, cos=0.46) -> upgrade_plan             ❌ MISROUTED!
    • System 1: Instinct (0.16 ms, fam=5.83) -> ABSTAIN_TO_LLM           ✅
  [Trap 3: Heavy Mobile Typos] 'chagred twcie on my vsia crad endign 9921, pls refnud'
    • bge-small-en-v1.5 (3.94 ms, cos=0.67) -> reset_account_access     ❌ MISROUTED!
    • System 1: Instinct (0.16 ms, fam=5.98) -> refund_duplicate_charge  ✅

ACT 3: 'TEACH ME' — Teaching a Brand-New Jargon Term in 1 Line (< 5 ms)
  1. Unseen codeword ('Customer is asking for a Project-Sunburst adjustment on their account'):
     Before : ABSTAIN_TO_LLM (Safe! 0.171 ms)
  2. Teach 1 example: engine.learn(prompt, {'action': 'refund_duplicate_charge', 'urgent': True})
  3. Next ticket ('Please apply the Project-Sunburst adjustment for this workspace'):
     • bge-small-en-v1.5 (Static)   : upgrade_plan (cos=0.73) ❌ MISROUTED!
     • System 1: Instinct (Taught)  : refund_duplicate_charge (urgent=True) in 0.329 ms ✅
```

Run these end-to-end replay scripts to inspect real-world behavior on chronological public data:
- **Live GitHub issue triage (`astral-sh/uv` and `facebook/react` OOS).** Run [`examples/dogfood/live_github_triage.py`](examples/dogfood/live_github_triage.py) (or pass `--live` to fetch fresh issues via `gh api`). It replays `260` chronologically ordered `astral-sh/uv` issues plus `25` out-of-scope `facebook/react` issues from cold start (`100%` escalated) through online warm-up to a compiled `.s1m` skill that handles `93.3%` of held-out issues locally in `0.23 ms` (`100.0%` accepted precision, `96.0%` OOS abstention).
- **Live US FDA enforcement recall triage (`openFDA` and `astral-sh/uv` OOS).** Run [`examples/dogfood/live_fda_recall_triage.py`](examples/dogfood/live_fda_recall_triage.py) (or pass `--live` to query `api.fda.gov`). It replays `297` chronological FDA recall reports across a 3-field schema (`regulatory_center`, `hazard_class`, and `urgent_class_i_escalation`) plus `25` out-of-scope software issues, reaching `65.3%` local offload in `0.33 ms` (`95.7%` joint 3-field precision, `100.0%` OOS abstention).
- **Three-wave teacher auto-cutover demo.** Run [`examples/drop_in/llm_cutover_dropin.py`](examples/drop_in/llm_cutover_dropin.py) to observe an upstream teacher and cut over to a local `.s1m` skill.

---

## Choose between Instinct and a local LLM

A `~100 KiB` non-autoregressive runtime has clear boundaries. Use the table below to pick the right model for your workload:

| Scenario | Best tool | Rationale |
| :--- | :--- | :--- |
| **"I have `3 to 30` examples per option (or a live teacher LLM) and want to stop paying `500 ms` for repeatable routing, triage, tool selection, or guardrails."** | **System 1: Instinct**<br>(`~100 KiB` `.s1m`, `0.2 ms` CPU) | Offloads repetitive structured decisions in `0.2 ms` on CPU with no neural network weights, conformal OOD abstention, character-typo resilience, and `<0.5 ms` rank-1 online updates. |
| **"I have `3` static examples per class, no teacher model in the loop, and users use open-vocabulary synonyms (`'automobile'` when my examples only said `'car'`)."** | **Pretrained bi-encoder**<br>(`bge-small` or `all-MiniLM`, `70 to 90 MB`, `1.5 to 4 ms`) | Pretraining maps `'automobile'` to `'car'` on day zero. The trade-off is `70 to 90 MB` of neural weights, no `<1 ms` online updates, and false acceptance on out-of-scope queries unless wrapped by a conformal gate. |
| **"The decision requires multi-hop logic, reading comprehension across paragraphs, arithmetic, or `50,000` complex labeled examples."** | **Local small LLM or fine-tuned decision model**<br>(`0.5B to 8B`, `20 to 200 ms` GPU or NPU) | Linear and spectral hyperplanes plateau around `50 to 100` examples per option and cannot reason through a four-page contract or perform multi-step deduction. |
| **"How do Instinct and a local LLM work together?"** | **`Tier 1 Instinct` (`0.2 ms`) to `Tier 2 Local LLM` (`50 ms`)** | Put Instinct in front of your local LLM via `BoundedReviewFallback`. Instinct handles clear repeat traffic and rejects out-of-scope inputs in `0.2 ms`, waking your local LLM only on ambiguous inputs (`is_ambiguous=True`). |

---

## Drop Instinct into existing projects

The [`examples/drop_in/`](examples/drop_in/) and [`docs/drop_in/`](docs/drop_in/README.md) directories contain runnable drop-in replacements for `semantic-router`, `langchain-core`, `FastAPI` with `Instructor`, and `poke-env` with `pokemon-showdown`:

### 1. Replace `Instructor` in a `FastAPI` endpoint (`POST /v1/triage`)
- **Guide and code.** Read [`docs/drop_in/03_instructor_structured_outputs.md`](docs/drop_in/03_instructor_structured_outputs.md) and run `python examples/drop_in/instructor_structured_dropin.py`.
- **Behavior.** A `FastAPI` endpoint returns a four-field Pydantic model (`department`, `page_oncall`, `impact_tags`, and `urgency_score`), escalating uncertain Wave 1 requests to the `Instructor` teacher and cutting over to local `<0.5 ms` `.s1m` execution once taught.

### 2. Replace `aurelio-labs/semantic-router`
- **Guide and code.** Read [`docs/drop_in/01_semantic_router.md`](docs/drop_in/01_semantic_router.md) and run `python examples/drop_in/semantic_router_dropin.py`.
- **Behavior.** Pass your existing `semantic_router.Route(name=..., utterances=...)` objects directly into `InstinctRouteLayer(routes)` to cut routing latency to `0.14 ms` (`93.3 KiB` `.s1m` vs. `67.2 MB` ONNX) while abstaining on out-of-scope queries.

### 3. Replace `RouteLLM` cascades in `langchain-core`
- **Guide and code.** Read [`docs/drop_in/02_routellm_cascade.md`](docs/drop_in/02_routellm_cascade.md) and run `python examples/drop_in/routellm_dropin.py`.
- **Behavior.** Plug `RunnableLambda` into your LangChain LCEL pipeline to route queries on CPU, tolerate character-typo corruption, and learn online from frontier fallback answers.

### 4. Replace `Llama-Prompt-Guard` for polarity and security checks
- **Guide and code.** Read [`docs/drop_in/04_prompt_guard_security.md`](docs/drop_in/04_prompt_guard_security.md) and run `python examples/drop_in/prompt_guard_dropin.py`.
- **Behavior.** Pre-screen prompts in `0.21 ms` with `#` negation-scope token transformations so benign technical prompts containing words like `"ignore"` or `"system"` are not falsely blocked and negation flips (`"do not cancel, upgrade"`) route accurately.

### 5. Run a `poke-env` agent on live Pokémon Showdown (`gen9randombattle`)
- **Guide and code.** Read [`docs/drop_in/06_poke_env_showdown_agent.md`](docs/drop_in/06_poke_env_showdown_agent.md) and run `python benchmarks/quality/evaluate_poke_env_showdown.py --battles 50`.
- **Behavior.** Subclass `poke_env.player.Player` in 15 lines of Python; online turn-by-turn adaptation compiles into a `23 KiB` `.s1m` policy that runs in `0.093 ms` per turn against `SimpleHeuristicsPlayer`.

---

## Quickstart

```bash
python -m pip install system1-instinct

# Or from a source checkout:
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

Every benchmark runs offline on public datasets (`BANKING77`, `CLINC150-OOS`, `HWU64`, and `NotInject`) and live `smogon/pokemon-showdown` battles. To reproduce the head-to-head encoder comparison below, run `python benchmarks/quality/evaluate_drop_in_replacements.py`:

| Router architecture | Artifact size | `BANKING77` (32 classes clean) | `BANKING77` (32 classes typo stress) | `CLINC150-OOS` false accept (`120` OOS) | Negation and polarity accuracy | P50 CPU latency | `<1 ms` online RLS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical linear baseline** (`Unigram TF-IDF`) | `85 KiB` (no neural weights) | 77.3% | 66.0% | 3.3% | 70.0% | `0.18 ms` | No |
| **`all-MiniLM-L6-v2`** (`22M` dense bi-encoder) | `91.1 MB` ONNX | 91.0% | 70.7% | 8.0% | 60.0% | `1.32 ms` | No |
| **`BAAI/bge-small-en-v1.5`** (`33M` dense bi-encoder) | `67.2 MB` ONNX | **94.5%** | **77.0%** | 5.0% calibrated (100% at `0.5`) | 90.0% | `2.96 ms` | No |
| **System 1: Instinct (Static `.s1m`)** | **`120 KiB`** (no neural weights) | **87.9%** | **74.2%** | **0.0%** (`0 of 120`) | **90.0%** | **`0.22 ms`** | **Yes** |
| **System 1: Instinct (`+ Online correction`)** | **`120 KiB`** (no neural weights) | **91.4%** | **74.2%** | **0.0%** (`0 of 120`) | **100.0%** | **`0.22 ms`** | **Yes** |

> **Full benchmark tables and fallible-teacher boundary.**
> - **Shot-by-shot adaptation curves (`0` to `40` examples/class) and the 6-category gauntlet.** See [`benchmarks/quality/README.md`](benchmarks/quality/README.md), [`drop_in_and_adaptation_report.md`](benchmarks/quality/drop_in_and_adaptation_report.md), and [`instinct_gauntlet_report.md`](benchmarks/quality/instinct_gauntlet_report.md).
> - **Fresh `64`-class `HWU64` traffic taught by a `10%`-noisy teacher.** Under a strict `1%` (`0.01`) family-wise conditional local error budget across `7,031` qualification items and `1,076` test requests, neither Instinct nor a classical logistic classifier can certify `<= 1%` risk (`0.0%` coverage). At the standard `5%` (`0.05`) local error budget, all learned methods qualify on `5 of 5` seeds with `0` budget misses, `instinct_lexical` offloads `27.68%` at `1.68%` local error vs. `simple_classifier`'s `20.46%` at `1.73%` error, and `instinct_hybrid` reaches `1.37%` local error (`5 of 5` `.s1m` reloadable). See [`benchmarks/heldout_traffic/REPORT.md`](benchmarks/heldout_traffic/REPORT.md).

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
