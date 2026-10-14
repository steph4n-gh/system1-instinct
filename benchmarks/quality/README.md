# System 1: Instinct — Quality & Adaptation Benchmark Suite

**Raw accuracy, accepted correctness, acceptance coverage, and execution latency are measured separately.**

| Benchmark Suite | Scope & Reproduction Command | Measured Results |
| :--- | :--- | :--- |
| **[Public Adaptation Curves & Neural Baseline Comparison](drop_in_and_adaptation_report.md)** | `python benchmarks/quality/evaluate_drop_in_replacements.py` | `BANKING77` (`24` intents): `53.6%` (`0`-shot) $\to$ `55.2%` (`adapt_zero_shot`) $\to$ `57.3%` (`1`-shot) $\to$ `65.6%` (`3`-shot) $\to$ `70.3%` (`5`-shot) $\to$ **`90.1%`** (`.s1m`); `CLINC150` (`24` intents): `61.7%` (`0`-shot) $\to$ **`87.5%`** (`.s1m`). Also compares `System 1: Instinct` (`120 KiB`, `0.22 ms`) head-to-head against `all-MiniLM-L6-v2` (`91.1 MB` ONNX) and `BAAI/bge-small-en-v1.5` (`67.2 MB` ONNX), plus functional smoke checks for [`examples/drop_in/`](../../examples/drop_in/README.md). |
| **[6-Category Decision Benchmark Gauntlet](instinct_gauntlet_report.md)** | `python benchmarks/quality/evaluate_instinct_gauntlet.py` | Compares a Classical Linear Baseline against System 1: Instinct (`Static .s1m` and `+ Online Flywheel`) across `BANKING77` (32/77 intents, clean & typo), `CLINC150-OOS` (`0.0%` OOS false acceptance), `NotInject` (`0.0%` benign FPR), 106 authored drift probes, 4-field extraction (`100.0%`), and 0-shot routing. |
| **[`poke-env` Pokémon Showdown Benchmark](poke_env_showdown_report.md)** | `python benchmarks/quality/evaluate_poke_env_showdown.py --battles 50` | Live `gen9randombattle` Showdown matches: **`100.0%`** vs `RandomPlayer`, **`98.0%`** vs `MaxBasePowerPlayer`, and **`62.0%`** (Zero-Shot Adaptive) / **`60.0%`** (`Compiled .s1m` at `0.093 ms` P50) vs `SimpleHeuristicsPlayer`. |
| **[Three Bounded Public-Data Skills](workloads/README.md)** | `python benchmarks/quality/evaluate_release.py` | Banking support (`97.5%` accepted acc), Assistant commands (`99.4%` accepted acc), and SMS triage (`97.7%` accepted acc) at `0.34–0.44 ms` median latency. |
| **[Authored Drift & Qualification Probes](quality_round/README.md)** | `python benchmarks/quality/evaluate_quality_round.py` | 106 non-template drift probes testing vocabulary shift, class imbalance, and complexity routing. |
| **[Public SpamAssassin Email](public_email/README.md)** | `python benchmarks/quality/evaluate_public_email.py --download --representative` | Real checksum-pinned SpamAssassin corpus (`603/618` accepted decisions correct on representative grouped split). |
| **[Real-Document Correction Workflow](document_workflow/README.md)** | `python benchmarks/quality/document_workflow/run.py` | Five BBC news topics on a 400-article holdout testing `TeachingSession` candidate review and strict adoption gates. |

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
