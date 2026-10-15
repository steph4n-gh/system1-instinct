# System 1: Instinct Quality and Adaptation Benchmarks

**Raw accuracy, accepted correctness, acceptance coverage, and execution latency are measured separately.**

| Benchmark Suite | Scope and Reproduction Command | Measured Results |
| :--- | :--- | :--- |
| **[Fresh Traffic with a Fallible Teacher](../heldout_traffic/REPORT.md)** | [Frozen protocol and commands](../heldout_traffic/README.md) | `HWU64`, five `10%` teacher-noise seeds: `0.0%` qualified coverage at the strict `1%` error budget (`7,031` qualification items); at the standard `5%` error budget, all learned methods qualify on `5/5` seeds with `0` budget misses (`instinct_lexical` `27.68%` offload at `1.68%` local error vs. `simple_classifier` `20.46%` at `1.73%`; `instinct_hybrid` `1.37%` local error and `5/5` `.s1m` reloadable). |
| **[Public Adaptation Curves and Neural Baseline Comparison](drop_in_and_adaptation_report.md)** | `python benchmarks/quality/evaluate_drop_in_replacements.py` | `BANKING77` (`24` intents): `53.6%` (`0`-shot) to `90.1%` (`.s1m`); `CLINC150` (`24` intents): `61.7%` (`0`-shot) to `87.5%` (`.s1m`). Compares `System 1: Instinct` (`120 KiB`, no neural weights, `0.22 ms`) against `all-MiniLM-L6-v2` (`91.1 MB` ONNX) and `BAAI/bge-small-en-v1.5` (`67.2 MB` ONNX), plus smoke checks for [`examples/drop_in/`](../../examples/drop_in/README.md). |
| **[6-Category Decision Benchmark Gauntlet](instinct_gauntlet_report.md)** | `python benchmarks/quality/evaluate_instinct_gauntlet.py` | Compares a classical linear baseline against System 1: Instinct (`Static .s1m` and `+ Online replay`) across `BANKING77` (`32` and `77` intents, clean and typo), `CLINC150-OOS` (`0.0%` OOS false acceptance), `NotInject` (`0.0%` benign FPR), `106` authored drift probes, 4-field extraction (`100.0%`), and 0-shot routing. |
| **[`poke-env` Pokémon Showdown Benchmark](poke_env_showdown_report.md)** | `python benchmarks/quality/evaluate_poke_env_showdown.py --battles 50` | Live `gen9randombattle` Showdown matches: `100.0%` vs. `RandomPlayer`, `98.0%` vs. `MaxBasePowerPlayer`, and `62.0%` (Zero-Shot Adaptive) / `60.0%` (`Compiled .s1m` at `0.093 ms` P50) vs. `SimpleHeuristicsPlayer`. |
| **[Three Bounded Public-Data Skills](workloads/README.md)** | `python benchmarks/quality/evaluate_release.py` | Banking support (`97.5%` accepted accuracy), Assistant commands (`99.4%` accepted accuracy), and SMS triage (`97.7%` accepted accuracy) at `0.34 to 0.44 ms` median latency. |
| **[Authored Drift and Qualification Probes](quality_round/README.md)** | `python benchmarks/quality/evaluate_quality_round.py` | `106` non-template drift probes testing vocabulary shift, class imbalance, and complexity routing. |
| **[Public SpamAssassin Email](public_email/README.md)** | `python benchmarks/quality/evaluate_public_email.py --download --representative` | Checksum-pinned SpamAssassin corpus (`603/618` accepted decisions correct on representative grouped split). |
| **[Real-Document Correction Workflow](document_workflow/README.md)** | `python benchmarks/quality/document_workflow/run.py` | Five BBC news topics on a 400-article holdout testing `TeachingSession` candidate review and strict adoption gates. |

---

## 1. Sample-efficiency curves (`0` to `40` examples per class)

Measured on `24` intents from `CLINC150` (`120` test queries) and `BANKING77` (`192` test queries) with an oracle fallback at $\alpha = 0.10$ ([full report](drop_in_and_adaptation_report.md)):

| Examples per class | Total examples (`24` classes) | `CLINC150` raw accuracy | `CLINC150` local offload % | `CLINC150` end-to-end accuracy | `BANKING77` raw accuracy | `BANKING77` local offload % | `BANKING77` end-to-end accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0` (Cold)** | **`0`** | `61.7%` | `0.0%` *(100% teacher)* | **`100.0%`** | `53.6%` | `0.0%` *(100% teacher)* | **`100.0%`** |
| **`1` per class** | **`24`** | `65.8%` | **`42.5%` local** | **`92.5%`** | `57.3%` | **`27.6%` local** | **`96.4%`** |
| **`3` per class** | **`72`** | `72.5%` | **`36.7%` local** | **`97.5%`** | `65.6%` | **`27.6%` local** | **`97.9%`** |
| **`5` per class** | **`120`** | `82.5%` | **`37.5%` local** | **`97.5%`** | `70.3%` | **`33.9%` local** | **`97.4%`** |
| **`8` per class** | **`192`** | `85.8%` | **`40.0%` local** | **`99.2%`** | `72.4%` | **`40.1%` local** | **`96.9%`** |
| **`10` per class** | **`240`** | `85.8%` (`.s1m`) | **`80.0%` local** *(compiled)* | **`97.5%`** | `77.1%` (`.s1m`) | **`78.1%` local** *(compiled)* | **`90.1%`** |
| **`all` (`10 to 25` per class)** | **`240 to 600`** | **`87.5%`** (`.s1m`) | **`79.2%` local** *(compiled)* | **`95.8%`** | **`90.1%`** (`.s1m`) | **`68.8%` local** *(compiled)* | **`99.5%`** |

> **Why `BANKING77` end-to-end accuracy dips at `10 per class` (`96.9%` to `90.1%`).** Through `8 per class`, Instinct runs in conservative online mode (`40.1%` handled locally, `59.9%` escalated to the `100%`-accurate oracle teacher). At `10 per class`, switching to a compiled `.s1m` skill nearly doubles local handling (`40.1%` to `78.1%`) while raw model accuracy is `77.1%` and the conformal split has only a few calibration examples per class at $\alpha = 0.10$. By `25 per class`, both raw accuracy (`90.1%`) and conformal calibration tighten, bringing end-to-end accuracy to `99.5%` (`68.8%` handled locally).

---

## 2. Six-category offline benchmark gauntlet

To reproduce this table offline (`block_network()`), run `python benchmarks/quality/evaluate_instinct_gauntlet.py` ([full report](instinct_gauntlet_report.md)):

| Category | Workload | Classical linear baseline | System 1: Instinct (Static) | System 1: Instinct (+ Online replay) |
| :--- | :--- | :---: | :---: | :---: |
| **1. High-cardinality intent routing** | `BANKING77` (32 intents, clean and typo) | 77.3% clean, 66.0% typo | **87.9%** clean, **74.2%** typo | **91.4%** |
| **1b. Full 77-class scaling** | `BANKING77` (77 intents, clean and typo) | 62.0% clean, 45.5% typo | **79.5%** clean, **58.3%** typo | **88.1%** |
| **2. Multi-domain and OOS rejection** | `CLINC150-OOS` (in-scope accuracy and OOS false accept) | 84.7%, 3.3% false accept | **86.1%**, **0.0% false accept** | **99.3%**, **0.0% false accept** |
| **3. Injection and NotInject negatives** | `NotInject` benign-trigger FPR and polarity accuracy | 10.0% FPR, 70.0% accuracy | **0.0% FPR**, **90.0% accuracy** | **0.0% FPR**, **100.0% accuracy** |
| **4. Authored drift probes (106)** | `banking_support`, `sms_triage`, and `model_routing` coverage | 86.2%, 77.5%, 77.8% | **100.0%**, **82.5%**, **88.9%** | **100.0%**, **87.5%**, **94.4%** |
| **5. Multi-field structured triage** | 4-field joint accuracy (`ChoiceField`, `BooleanField`, `MultiChoiceField`, and `ScoreField`) | 97.1% | **100.0%** (`0.40 ms`) | **100.0%** (`0%` Wave 3 escalation) |
| **6. Zero-shot cold-start routing** | `intent_routing` and `security_triage` (0-shot) | 51.0%, 50.0% | **81.0%**, **85.0%** | **100.0%** |

---

## Quick Reproduction Commands

```bash
# 1. Public Adaptation Curves, Neural Baseline Comparison & Drop-In Smoke Checks
python benchmarks/quality/evaluate_drop_in_replacements.py

# 2. 6-Category Industry-Standard Decision Benchmark Gauntlet
python benchmarks/quality/evaluate_instinct_gauntlet.py

# 3. Official poke-env + Pokémon Showdown Benchmark
python benchmarks/quality/evaluate_poke_env_showdown.py --battles 50

# 4. Primary Taught Skill Release Checks
python benchmarks/quality/evaluate_release.py
```

---

## Teaching Comparison

| Teaching Method | When to Use | Reproduce / Guide |
| :--- | :--- | :--- |
| **Zero-Shot (`ChoiceField.descriptions`)** | Day-0 deployment with `0` labeled examples | [Drop-In Guides](../../docs/drop_in/README.md) |
| **Online RLS (`learn_from_tier2` + `BoundedReviewFallback`)** | Live 1-shot boundary adaptation in `<1 ms` | [LLM Auto-Cutover Guide](../../docs/drop_in/05_llm_observe_and_cutover.md) |
| **Offline `TeachingSession` + `system1 compile`** | High-accuracy compiled `.s1m` skill with spectral `PPMI + SVD` | [Training Experts Guide](../../docs/guides/training_experts.md) |
