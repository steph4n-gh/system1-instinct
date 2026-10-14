# System 1: Instinct — Official `poke-env` + Pokémon Showdown Benchmark Report

**Engine**: System 1: Instinct `1.0.0` Non-Autoregressive Mathematical Decision Runtime  
**Simulator**: Local `smogon/pokemon-showdown` Server + `poke-env` (`gen9randombattle`)  
**LLM API Calls**: `0` (`$0.00` cost, sub-millisecond local CPU execution)

---

## 1. Live `gen9randombattle` Showdown Matchups vs. Official `poke-env` Baselines

| System 1: Instinct Agent | Opponent (`poke-env` Official) | Battles | Wins / Losses | Win Rate | Turn Decisions | P50 Decision Latency | Online Adaptations |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **System 1 Zero-Shot (0 Examples)** | `RandomPlayer` | 50 | `50W - 0L` | **100.0%** | 1036 | **3.177 ms** | 0 |
| **System 1 Zero-Shot (0 Examples)** | `MaxBasePowerPlayer` | 50 | `49W - 1L` | **98.0%** | 829 | **3.217 ms** | 0 |
| **System 1 Zero-Shot (0 Examples)** | `SimpleHeuristicsPlayer` | 50 | `21W - 29L` | **42.0%** | 863 | **3.220 ms** | 0 |
| **System 1 Zero-Shot Adaptive (Online)** | `RandomPlayer` | 50 | `50W - 0L` | **100.0%** | 1051 | **3.166 ms** | 5 |
| **System 1 Zero-Shot Adaptive (Online)** | `MaxBasePowerPlayer` | 50 | `46W - 4L` | **92.0%** | 831 | **3.201 ms** | 4 |
| **System 1 Zero-Shot Adaptive (Online)** | `SimpleHeuristicsPlayer` | 50 | `31W - 19L` | **62.0%** | 867 | **3.238 ms** | 4 |
| **System 1 Compiled (.s1m)** | `RandomPlayer` | 50 | `49W - 1L` | **98.0%** | 1148 | **0.097 ms** | 0 |
| **System 1 Compiled (.s1m)** | `MaxBasePowerPlayer` | 50 | `41W - 9L` | **82.0%** | 874 | **0.090 ms** | 0 |
| **System 1 Compiled (.s1m)** | `SimpleHeuristicsPlayer` | 50 | `30W - 20L` | **60.0%** | 869 | **0.093 ms** | 0 |

---

## 2. Zero-Shot Cold-Start & Online Adaptation Tactical Suite

Evaluates how System 1: Instinct handles unseen battle-state signals starting from `0` training examples,
after unsupervised test-time prototype adaptation (`adapt_zero_shot`), after 1-shot online
updates (`learn_from_tier2`), and when compiled into a `.s1m` skill:

| Metric | Result |
| :--- | :---: |
| **Zero-Shot Canonical State Accuracy (`0` Examples)** | **100.0%** |
| **Zero-Shot Unseen Jargon State Accuracy (Before Adaptation)** | 20.0% |
| **Unsupervised `adapt_zero_shot` Unseen Jargon Accuracy (`0` Labels)** | **100.0%** |
| **1-Shot Online `learn_from_tier2` Jargon Accuracy** | **100.0%** |
| **Compiled `.s1m` Tactical Skill Accuracy** | **100.0%** |
| **Zero-Shot P50 Latency / Compiled `.s1m` P50 Latency** | **1.628 ms** / **1.230 ms** |
| **Compiled `.s1m` Artifact Size** | **155.6 KB** |
