# System 1: Instinct — Drop-In Replacement & Zero-Shot Adaptation Report

## 1. Drop-In Replacement Suite (`examples/drop_in/`)

| Drop-In Replacement | Primary Metric | Result | Secondary Metric | P50 CPU Latency |
| :--- | :--- | :---: | :---: | :---: |
| **`semantic-router`** (`SystemOneRouteLayer`) | In-Domain Route Accuracy | **100.0%** | OOD Rejection: **100.0%** (`adapt_zero_shot`: `0% -> 100%`) | **1.363 ms** |
| **`RouteLLM`** (`SystemOneRouteController`) | 3-Tier Cascade Accuracy | **100.0%** | Zero External API Calls | **1.468 ms** |
| **`Instructor` / `BAML`** (`SystemOneInstructorClient`) | Joint Multi-Field Accuracy | **100.0%** | Dept: **100.0%** / Alert: **100.0%** | **2.729 ms** |
| **`Llama-Prompt-Guard`** (`SystemOnePromptGuard`) | Attack Detection Recall | **100.0%** | `NotInject` Benign FPR: **0.0%** | **0.811 ms** |

---

## 2. Zero-Shot $\to$ Few-Shot $\to$ Compiled `.s1m` Adaptation Curves

Starts from **`0` training examples** (using only the raw intent label names as descriptions),
then adapts online via unsupervised `adapt_zero_shot()`, `1`-shot, `3`-shot, `5`-shot `learn_from_tier2()`,
and full `.s1m` compilation:

| Benchmark (`24` Intents) | `0`-Shot (Label Names Only) | `0`-Shot + Unsupervised `adapt_zero_shot` | `1`-Shot Online | `3`-Shot Online | `5`-Shot Online | Compiled `.s1m` | Compiled P50 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`BANKING77`** (`192` test queries) | 53.6% | 55.2% | **57.3%** | **65.6%** | **70.3%** | **90.1%** | **0.339 ms** |
| **`CLINC150`** (`120` test queries) | 61.7% | 62.5% | **65.8%** | **72.5%** | **82.5%** | **87.5%** | **0.312 ms** |
