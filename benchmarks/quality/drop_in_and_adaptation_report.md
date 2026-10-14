# System 1: Instinct (`1.0.0`) — Adaptation Curves, Neural Baseline Comparison & Drop-In Verification

## 1. Public Benchmark Adaptation Curves (`BANKING77` & `CLINC150`)

Evaluated on held-out public intent datasets (`BANKING77` and `CLINC150`). Starts from **`0` training examples**
(using only the raw intent label names as descriptions), then adapts online via unsupervised `adapt_zero_shot()`,
`1`-shot, `3`-shot, `5`-shot `learn_from_tier2()`, and full `.s1m` compilation:

| Public Dataset (`24` Intents) | `0`-Shot (Label Names Only) | `0`-Shot + Unsupervised `adapt_zero_shot` | `1`-Shot Online | `3`-Shot Online | `5`-Shot Online | Compiled `.s1m` | Compiled P50 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`BANKING77`** (`192` test queries) | 53.6% | 55.2% | **57.3%** | **65.6%** | **70.3%** | **90.1%** | **0.350 ms** |
| **`CLINC150`** (`120` test queries) | 61.7% | 62.5% | **65.8%** | **72.5%** | **82.5%** | **87.5%** | **0.329 ms** |

---

## 2. Head-to-Head vs. Pretrained Neural Embedding Routers & Classical Linear Baseline

Compares **System 1: Instinct (`0` neural weights)** against both a **Classical Linear Baseline (Unigram TF-IDF)**
and standard **Pretrained Neural Embedding Routers (`all-MiniLM-L6-v2` and `BAAI/bge-small-en-v1.5` via ONNX)**
on the exact same held-out splits from the [6-category benchmark gauntlet](v1_4_gauntlet_report.md):

| Router Architecture | Model / Artifact Size | `BANKING77` (32c Clean) | `BANKING77` (32c Typo Stress) | `CLINC150-OOS` False Accept | Negation / Polarity Acc | P50 CPU Latency | `<1 ms` Online RLS Update |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Linear Baseline** (`Unigram TF-IDF`) | `85 KiB` (`0` weights) | 77.3% | 66.0% | 3.3% | 70.0% | `0.18 ms` | No |
| **`all-MiniLM-L6-v2`** (`22M` Dense Bi-Encoder) | `91.1 MB` ONNX | 91.0% | 70.7% (`-20.3%` drop) | 8.0% | 60.0% | `1.32 ms` (`~10 ms` PyTorch) | No |
| **`BAAI/bge-small-en-v1.5`** (`33M` Dense Bi-Encoder) | `67.2 MB` ONNX | **94.5%** | **77.0%** (`-17.5%` drop) | 5.0% (cal) / 100% (`@0.5`) | 90.0% | `2.96 ms` (`~14 ms` PyTorch) | No |
| **System 1: Instinct (Static `.s1m`)** | **`120 KiB`** (`0` weights) | **87.9%** | **74.2%** (`+3.5%` vs MiniLM) | **0.0%** (`0/120`) | **90.0%** | **`0.22 ms`** | **Yes** |
| **System 1: Instinct (`+ Online Flywheel`)** | **`120 KiB`** (`0` weights) | **91.4%** (`+0.4%` vs MiniLM) | **74.2%** | **0.0%** (`0/120`) | **100.0%** | **`0.22 ms`** | **Yes** |

**Key Takeaways:**
- **Competitive with `all-MiniLM-L6-v2` at `750x` smaller size and `6x–45x` lower latency**: On clean `BANKING77` (`32` intents), System 1: Instinct (`120 KiB`, `0` neural weights, `0.22 ms`) reaches `87.9%` statically and **`91.4%`** with one pass of online RLS adaptation—surpassing `all-MiniLM-L6-v2` (`91.0%`, `91.1 MB` ONNX).
- **Higher robustness on Character Typos & Negation Polarity than `all-MiniLM-L6-v2`**: Subword WordPiece tokenizers shatter under character swaps (`all-MiniLM-L6-v2` drops `-20.3%` to `70.7%`) and bi-encoder pooling blurs negated clauses (`"do not cancel, upgrade"` -> `60.0%`), whereas System 1's bounded character n-grams (`(3, 5)`) and negation-scope binding (`neg:cancel`, `#`) achieve **`74.2%`** on typos and **`90.0% -> 100.0%`** on polarity contrasts.
- **`0.0%` Out-of-Scope False Acceptance**: Cosine-threshold neural routers accept `5.0%–8.0%` of `CLINC150-OOS` out-of-scope queries even after threshold calibration, whereas System 1's dual familiarity + prototype + Mondrian conformal gate achieves **`0.0%` false acceptance (`0/120`)**.

---

## 3. Runnable Drop-In Replacement Functional Verification (`examples/drop_in/`)

> **Note:** The 4 micro-suites below (`8–16` inspectable prompts each) are **functional smoke tests and procedural examples**
> accompanying the step-by-step guides in [`docs/drop_in/README.md`](../../docs/drop_in/README.md). They verify API drop-in
> compatibility, sub-millisecond latency, and live `adapt_zero_shot` / `learn_from_tier2` mechanics rather than serving as
> large-scale dataset benchmarks.

| Procedural Drop-In Example | Verified Capability | Smoke Check | Secondary Verification | P50 CPU Latency |
| :--- | :--- | :---: | :---: | :---: |
| **`semantic-router`** ([guide](../../docs/drop_in/01_semantic_router.md)) | In-Domain + OOD + `adapt_zero_shot` | **100.0%** | OOD Reject: **100.0%** · Unseen Jargon: `0% -> 100%` | **1.217 ms** |
| **`RouteLLM`** ([guide](../../docs/drop_in/02_routellm_cascade.md)) | 3-Tier Complexity Cascade | **100.0%** | Zero External API Calls | **1.434 ms** |
| **`Instructor` / `BAML`** ([guide](../../docs/drop_in/03_instructor_structured_outputs.md)) | 4-Field Joint Extraction | **100.0%** | Dept: **100.0%** / Alert: **100.0%** | **1.825 ms** |
| **`Llama-Prompt-Guard`** ([guide](../../docs/drop_in/04_prompt_guard_security.md)) | Attack Recall + `NotInject` Guard | **100.0%** | `NotInject` Benign FPR: **0.0%** | **0.810 ms** |
