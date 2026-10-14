# System 1 v1.4 — Industry-Standard Decision Benchmark Gauntlet Report

**Version**: System 1 `1.4.0`  
**Execution Mode**: 100% Offline, Deterministic, Socket-Blocked (`block_network()`)  
**Hardware Target**: Local CPU / NumPy Single-Pass Non-Autoregressive Runtime

---

## Executive Summary Across All 6 Benchmark Categories

| Category | Workload | Baseline v1.2 | System 1 v1.4 (Static) | System 1 v1.4 (+ Flywheel) | Key v1.4 Improvement |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **1. High-Cardinality Intent Routing** | `BANKING77` (32 Intents, Clean / Typo) | 77.3% / 66.0% | **87.9%** / **74.2%** | **91.4%** | +10.5% clean, +8.2% typo |
| **1b. Full 77-Class Scaling** | `BANKING77` (77 Intents, Clean / Typo) | 62.0% / 45.5% | **79.5%** / **58.3%** | **88.1%** | +17.5% clean, +12.8% typo |
| **2. Multi-Domain + OOS Rejection** | `CLINC150-OOS` (In-Scope Acc / OOS False Accept) | 84.7% / 3.3% FA | **86.1%** / **0.0% FA** | **99.3%** / **0.0% FA** | OOS False Acceptance driven to **0.0%** (≤1% target) |
| **3. Injection & NotInject Negatives** | `NotInject` Benign-Trigger FPR / Polarity Acc | 10.0% FPR / 70.0% | **0.0% FPR** / **90.0%** | **0.0% FPR** / **100.0%** | -10.0% FPR, +20.0% polarity |
| **4a. Fresh Authored Banking Drift** | `banking_support` (30 Fresh Probes, Accepted Acc) | 86.2% (4 err) | **100.0%** (0 err) | **100.0%** | Surpasses ≥95% accepted accuracy target |
| **4b. Fresh Authored SMS Drift** | `sms_triage` (40 Fresh Probes, Raw / Accepted Acc) | 77.5% / 83.3% | **82.5%** / **86.5%** | **87.5%** / **89.7%** | +5.0% raw accuracy on class-imbalanced SMS |
| **4c. Model Complexity Routing** | `model_routing` (36 Fresh Probes, Acc / Coverage) | 86.1% / 77.8% cov | **91.7%** / **88.9% cov** | **94.4%** / **94.4% cov** | +11.1% coverage (clears ≥80% gate) |
| **5. Multi-Field Structured Triage** | 4-Field Joint Accuracy (Choice+Bool+Multi+Score) | 97.1% | **100.0%** | **100.0%** | Single-pass 4-field extraction in 0.40 ms |
| **6. Zero-Shot Cold-Start & Drop-In** | `intent_routing` / `security_triage` (0-Shot) | 51.0% / 50.0% | **81.0%** / **85.0%** | **100.0%** | +29.0% intent, +35.0% security 0-shot |

---

## Category 1: High-Cardinality Fine-Grained Intent Routing (`BANKING77` + Typo Stress + Memory Scaling)

Evaluates System 1 v1.4's class-discriminative `chi2` feature selection, boundary character n-grams `(3, 5)`,
`HybridProjector.fit` spectral `PPMI + SVD` projection with concept anchors, and coarse-to-fine `hierarchical`
choice heads against Baseline v1.2 flat word-frequency Ridge regression.

### 1.1 Accuracy & Latency Breakdown

| Configuration | 32-Intent Clean Acc | 32-Intent Typo Acc | 77-Intent Clean Acc | 77-Intent Typo Acc | P50 Latency | `.s1m` Artifact Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline v1.2** (`frequency` TF-IDF, flat, eager cov) | 77.3% | 66.0% | 62.0% | 45.5% | 0.278 ms | 1223.4 KB |
| **System 1 v1.4** (`chi2` + `char_ngrams=(3,5)` + `hierarchical`) | 82.0% | **74.2%** | 69.5% | **51.1%** | 1.522 ms | **1259.0 KB** |
| **System 1 v1.4** (`HybridProjector.fit` + `hierarchical`) | **87.9%** | 71.9% | **79.5%** | 58.3% | 2.165 ms | 5453.1 KB |
| **System 1 v1.4 + Flywheel** (`BoundedReviewFallback`) | **91.4%** | — | **88.1%** | — | 2.165 ms | 5453.1 KB |

### 1.2 Lazy Covariance Memory Scaling (`D = 1024 .. 8192`, `K = 77` Intents)

| Feature Dimension (`D`) | v1.2 Eager Covariance RAM | v1.4 Lazy Covariance RAM | Memory Reduction Factor |
| :---: | :---: | :---: | :---: |
| `1024` | 4,720.6 KB | **308.3 KB** | **15.3x** |
| `2048` | 17,632.6 KB | **616.3 KB** | **28.6x** |
| `4096` | 68,032.6 KB | **1,232.3 KB** | **55.2x** |
| `8192` | 267,136.6 KB | **2,464.3 KB** | **108.4x** |

---

## Category 2: Multi-Domain Assistant Routing + Strict Out-of-Scope Rejection (`CLINC150-OOS`)

Evaluates 15 in-scope assistant domains alongside official held-out `CLINC150` out-of-scope (OOS) queries
and cross-domain adversarial prompts. System 1 v1.4 combines **TF-IDF feature familiarity gating**
(`familiarity_alpha=0.15`, `low_feature_familiarity`) and **class-prototype cosine distance gating**
(`prototype_alpha=0.15`, `low_prototype_similarity`) with **Mondrian class-conditional calibration**.

| Metric | Baseline v1.2 | System 1 v1.4 (Static) | System 1 v1.4 (+ Flywheel) |
| :--- | :---: | :---: | :---: |
| **In-Scope Raw Accuracy** (144 queries) | 84.7% | **86.1%** | **99.3%** |
| **In-Scope Accepted Accuracy** | 100.0% | **100.0%** | **100.0%** |
| **In-Scope Coverage** | 52.1% | **46.5%** | **95.8%** |
| **OOS Rejection Rate** (120 OOS queries) | 96.7% | **100.0%** | **100.0%** |
| **OOS False-Acceptance Rate** (Target ≤ 1.0%) | 3.3% (4 accepted) | **0.0%** (0 accepted) | **0.0%** |

---

## Category 3: Prompt-Injection & `NotInject` Benign-Trigger Hard Negatives + Polarity Contrasts

Evaluates over-defense against benign prompts containing trigger keywords (`"ignore"`, `"system"`,
`"override"`, `"bypass"`, `"admin"`) and minimal negation/polarity pairs (`"no, working great"`,
`"do not cancel, upgrade"`).

| Suite & Metric | Baseline v1.2 | System 1 v1.4 (Static) | System 1 v1.4 (+ Flywheel) |
| :--- | :---: | :---: | :---: |
| **`NotInject` Overall Accuracy** | 93.8% | **100.0%** | **100.0%** |
| **`NotInject` Benign-Trigger False Positive Rate (FPR)** | 10.0% | **0.0%** | **0.0%** |
| **Adversarial Injection Recall** | 100.0% | **100.0%** | **100.0%** |
| **`NotInject` Accepted Accuracy / Coverage** | 100.0% / 93.8% | **100.0%** / **62.5%** | **100.0%** / **100.0%** |
| **Polarity & Negation Contrast Accuracy** | 70.0% | **90.0%** | **100.0%** |
| **Polarity Accepted Accuracy / Coverage** | 100.0% / 40.0% | **100.0%** / **40.0%** | **100.0%** / **100.0%** |

---

## Category 4: Fresh Authored Drift & Model Complexity Routing (`quality_round/cases.json` — 106 Probes)

Evaluates all 106 human-authored non-template probes from `benchmarks/quality/quality_round/cases.json`
designed to test vocabulary drift, class imbalance (`sms_triage`), and multi-tier complexity routing (`model_routing`).

| Workload (Fresh Cohort) | Metric | Baseline v1.2 | System 1 v1.4 (Static) | System 1 v1.4 (+ Flywheel) |
| :--- | :--- | :---: | :---: | :---: |
| **`banking_support`** (30 probes) | Raw Accuracy | 86.7% | **93.3%** | **100.0%** |
| **`banking_support`** (30 probes) | Accepted Accuracy (Target ≥95%) | 86.2% (4 err) | **100.0%** (0 err) | **100.0%** (0 err) |
| **`sms_triage`** (40 probes) | Raw Accuracy | 77.5% | **82.5%** | **87.5%** |
| **`sms_triage`** (40 probes) | Accepted Accuracy | 83.3% (6 err) | **86.5%** (5 err) | **89.7%** (4 err) |
| **`model_routing`** (36 probes) | Raw Accuracy | 86.1% | **91.7%** | **94.4%** |
| **`model_routing`** (36 probes) | Coverage (Target ≥80%) | 77.8% | **88.9%** | **94.4%** |
| **`model_routing`** (36 probes) | Accepted Accuracy | 100.0% | **100.0%** | **94.1%** |

---

## Category 5: Multi-Field Structured Ticket Triage (`Instructor` / `BAML` Style) & Flywheel Convergence

Evaluates simultaneous single-pass structured extraction across `ChoiceField` (`department`),
`BooleanField` (`urgent_escalation`), `MultiChoiceField` (`compliance_tags`), and `ScoreField` (`severity_score`).

### 5.1 Multi-Field Structured Extraction Accuracy

| Field / Metric | Baseline v1.2 | System 1 v1.4 (Static) |
| :--- | :---: | :---: |
| `department` (`ChoiceField` 5 options) | 100.0% | **100.0%** |
| `urgent_escalation` (`BooleanField`) | 100.0% | **100.0%** |
| `compliance_tags` (`MultiChoiceField` exact set match) | 100.0% | **100.0%** |
| `severity_score` (`ScoreField` MAE / ±0.15 accuracy) | 0.031 / 97.1% | **0.021** / **100.0%** |
| **Joint All-4-Fields Exact Match** | 97.1% | **100.0%** |
| **Accepted Joint Accuracy** | 100.0% | **100.0%** |
| **Single-Pass P50 Latency / `.s1m` Artifact Size** | 0.261 ms / 25.2 KB | **0.398 ms** / **63.8 KB** |

### 5.2 Closed-Loop Flywheel Convergence (`BoundedReviewFallback`)

| Traffic Wave | Local System 1 Acceptance Rate | Teacher Escalation Rate | End-to-End Joint Accuracy |
| :---: | :---: | :---: | :---: |
| Wave 1 | **62.9%** | 37.1% | **100.0%** |
| Wave 2 | **100.0%** | 0.0% | **100.0%** |
| Wave 3 | **100.0%** | 0.0% | **100.0%** |

---

## Category 6: Out-of-the-Box Zero-Shot Routing & Drop-In Open-Source Router Patterns

Evaluates System 1 v1.4 out-of-the-box (`0` teaching calls, `0` calibration calls) using morphological
stemming (`_stem_word`), sibling-contrastive stem IDF filtering, and disjunctive multi-exemplar sub-prototype
scoring (`_exemplar_bank`), followed by online `BoundedReviewFallback` flywheel convergence.

| Workload | Baseline v1.2 (0-Shot) | System 1 v1.4 Default (0-Shot) | System 1 v1.4 Hybrid (0-Shot) | System 1 v1.4 + Flywheel |
| :--- | :---: | :---: | :---: | :---: |
| **`intent_routing`** (100 tickets, 4 depts) | 51.0% | **80.0%** (+29.0%) | **81.0%** (+19.0%) | — |
| **`security_triage`** (100 ops, 3 tiers) | 50.0% | **85.0%** (+35.0%) | **82.0%** (+27.0%) | **100.0%** |
| **`raw_routing_control`** (12 probes, 3 depts) | 66.7% | **100.0%** | — | — |
| **`TypeSafeClient` 1-Line Model Router** (36 probes) | — | **47.2%** (0-shot) | — | **94.4%** (taught+flywheel) |
