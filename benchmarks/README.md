# Benchmarks

The [v1.4 industry-standard decision benchmark gauntlet](quality/v1_4_gauntlet_report.md)
(`python benchmarks/quality/evaluate_v1_4_gauntlet.py`) evaluates System 1 1.4.0
offline across six categories: `BANKING77` (32 and 77 intents, clean and typo stress),
`CLINC150-OOS` (in-scope routing + 120 out-of-scope rejections), `NotInject`
benign-trigger hard negatives and polarity contrasts, 106 fresh authored drift
probes, multi-field (`Choice` + `Boolean` + `MultiChoice` + `Score`) 3-wave
flywheel convergence, and zero-shot cold-start projectors.

The [drop-in replacement & zero-shot-to-few-shot adaptation benchmark](quality/drop_in_and_adaptation_report.md)
(`python benchmarks/quality/evaluate_drop_in_replacements.py`) evaluates the four
drop-in replacements ([`examples/drop_in/`](../examples/drop_in/README.md):
`Semantic Router`, `RouteLLM`, `Instructor / BAML`, and `Llama-Prompt-Guard / Guardrails`)
and measures the full adaptation trajectory (`0`-shot $\to$ unsupervised `adapt_zero_shot` $\to$
`1`-shot $\to$ `3`-shot $\to$ `5`-shot $\to$ `Compiled .s1m`) on `BANKING77` and `CLINC150`.

The [live `poke-env` Pokémon Showdown benchmark](quality/poke_env_showdown_report.md)
(`python benchmarks/quality/evaluate_poke_env_showdown.py --battles 50`) evaluates
`SystemOneShowdownPlayer` (`0`-shot, `0`-shot + online adaptation, and `Compiled .s1m`)
in real-time `gen9randombattle` Showdown battles against `RandomPlayer` (`100.0%` win rate),
`MaxBasePowerPlayer` (`98.0%` win rate), and `SimpleHeuristicsPlayer` (`62.0%` adaptive / `60.0%` compiled win rate at `0.093 ms` P50 latency).

The [full-scope n8n gauntlet](quality/n8n_gauntlet/README.md) covers all 150 CLINC
and 77 BANKING77 intents. Quality gates still fail; development improvements and
the real teacher-disconnection rehearsal do not establish qualification. Full
artifacts and raw results stay in Git and are excluded from Python distributions.

The [1.0.2 quality round](quality/quality_round/README.md) adds stricter qualification,
a measured SMS takeover improvement and fresh authored failures. System 1 remains
a small local decision runtime, not an LLM.

The [real-email experiment](quality/public_email/README.md) adds public
SpamAssassin teaching and evaluation with no credentials: 603/618 accepted
decisions correct on a representative grouped split, with a separate failed
source-shift experiment retained. Neither result qualifies seven-category routing.

Use the [quality suite](quality/README.md) for current evaluation commands and the
[public-workload report](quality/workloads/README.md) for banking, assistant and
SMS results, the classical baseline and actual Jev/Gemini takeover recordings.
All reported quality must distinguish raw correctness, local acceptance and
correctness among accepted decisions. Failed targets stay in the evidence.

[run_triple_crown_benchmark.py](run_triple_crown_benchmark.py) is a historical
three-domain simulation inspired by OpenHands, Instructor and Semantic Router
use cases. It does not execute those packages or establish drop-in compatibility
with them. Its cloud baseline timings, nominal labels and token costs are
synthetic. The archived [scorecard](results/triple_crown_scorecard.json) retains
that provenance and its historical numbers; its speedup and savings fields must
not be quoted as measured provider results.

The current workload runner compares against an actual local TF-IDF/logistic
regression baseline. Live teacher observations are recorded separately, and
teacher/local timings on different inputs are not presented as matched speedups.
