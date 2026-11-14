# System 1: Instinct — Benchmarks

All primary benchmarks are 100% reproducible from a local checkout with zero external API keys:

| Suite | Command | Report | What It Measures |
| :--- | :--- | :--- | :--- |
| **1. Public Adaptation Curves & Neural Baseline Comparison** | `python benchmarks/quality/evaluate_drop_in_replacements.py` | [drop_in_and_adaptation_report.md](quality/drop_in_and_adaptation_report.md) | `0`-shot $\to$ `adapt_zero_shot` $\to$ `1/3/5`-shot $\to$ `.s1m` curves on `BANKING77` & `CLINC150` + head-to-head vs `all-MiniLM-L6-v2` (`91.1 MB` ONNX) and `BAAI/bge-small-en-v1.5` (`67.2 MB` ONNX) + functional smoke verification of [`examples/drop_in/`](../examples/drop_in/README.md) |
| **2. 6-Category Decision Gauntlet** | `python benchmarks/quality/evaluate_instinct_gauntlet.py` | [instinct_gauntlet_report.md](quality/instinct_gauntlet_report.md) | `BANKING77` (32 & 77 intents, clean & typo), `CLINC150-OOS` (120 OOS rejections), `NotInject` benign triggers, 106 drift probes, 4-field triage, and 0-shot cold-start routing vs Classical Linear Baseline |
| **3. `poke-env` Showdown Benchmark** | `python benchmarks/quality/evaluate_poke_env_showdown.py --battles 50` | [poke_env_showdown_report.md](quality/poke_env_showdown_report.md) | Live `gen9randombattle` Showdown matches vs `RandomPlayer` (`100%`), `MaxBasePowerPlayer` (`98%`), and `SimpleHeuristicsPlayer` (`62%` Adaptive / `60%` `.s1m` at `0.093 ms`) |
| **4. Primary Skill Release Gate** | `python benchmarks/quality/evaluate_release.py` | [quality/README.md](quality/README.md) | Offline verification of `support_triage`, `model_routing`, and `agent_guard` against ≥95% accepted correctness and ≥80% coverage gates |
| **5. Held-Out Noisy Teacher Traffic (`HWU64`)** | `python benchmarks/heldout_traffic/run.py` (after `prepare.py`; verified with `verify.py`) | [heldout_traffic/REPORT.md](heldout_traffic/REPORT.md) | `64`-class `HWU64` with `10%` noisy teacher lessons at `1%` and `5%` local error budgets + independent receipt verification |
| **6. Offline Prequential Cassette Replay (`system1.cassette.v1`)** | `python benchmarks/prequential/run_causal_stream.py --json` | [prequential/README.md](prequential/README.md) | Socket-blocked causal test-then-train replay measuring teacher-call reduction, conditional local error, and OOS abstention |

See the [Quality Benchmark Suite Index](quality/README.md) for additional domain workloads ([workloads/README.md](quality/workloads/README.md), [public_email/README.md](quality/public_email/README.md), and [document_workflow/README.md](quality/document_workflow/README.md)).
