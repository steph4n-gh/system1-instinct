# System 1: Instinct — Benchmarks

All primary benchmarks are 100% reproducible from a local checkout with zero external API keys:

| Suite | Command | Report | What It Measures |
| :--- | :--- | :--- | :--- |
| **1. Public Adaptation Curves & Neural Baseline Comparison** | `python benchmarks/quality/evaluate_drop_in_replacements.py` | [drop_in_and_adaptation_report.md](quality/drop_in_and_adaptation_report.md) | `0`-shot $\to$ `adapt_zero_shot` $\to$ `1/3/5`-shot $\to$ `.s1m` curves on `BANKING77` & `CLINC150` + head-to-head vs `all-MiniLM-L6-v2` (`91.1 MB` ONNX) and `BAAI/bge-small-en-v1.5` (`67.2 MB` ONNX) + functional smoke verification of [`examples/drop_in/`](../examples/drop_in/README.md) |
| **2. 6-Category Decision Gauntlet** | `python benchmarks/quality/evaluate_v1_4_gauntlet.py` | [v1_4_gauntlet_report.md](quality/v1_4_gauntlet_report.md) | `BANKING77` (32 & 77 intents, clean & typo), `CLINC150-OOS` (120 OOS rejections), `NotInject` benign triggers, 106 drift probes, 4-field triage, and 0-shot cold-start routing vs Classical Linear Baseline |
| **3. `poke-env` Showdown Benchmark** | `python benchmarks/quality/evaluate_poke_env_showdown.py --battles 50` | [poke_env_showdown_report.md](quality/poke_env_showdown_report.md) | Live `gen9randombattle` Showdown matches vs `RandomPlayer` (`100%`), `MaxBasePowerPlayer` (`98%`), and `SimpleHeuristicsPlayer` (`62%` Adaptive / `60%` `.s1m` at `0.093 ms`) |
| **4. Primary Skill Release Gate** | `python benchmarks/quality/evaluate_release.py` | [quality/README.md](quality/README.md) | Offline verification of `support_triage`, `model_routing`, and `agent_guard` against ≥95% accepted correctness and ≥80% coverage gates |

See the [Quality Benchmark Suite Index](quality/README.md) for additional domain workloads ([workloads/README.md](quality/workloads/README.md), [public_email/README.md](quality/public_email/README.md), [document_workflow/README.md](quality/document_workflow/README.md), and [n8n_gauntlet/README.md](quality/n8n_gauntlet/README.md)).
