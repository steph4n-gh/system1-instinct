# Frozen fresh-traffic comparison

This is an offline simulation on a previously unused public intent workload,
HWU64. It tests generalization after learning from a fallible teacher. It is not
production traffic, timestamped drift, an LLM evaluation, or a claim about all
structured decision tasks.

The user-selected budget is **1% error among locally accepted answers**, with a
**10% noisy simulated teacher**. We compare an exact cache, word/character TF-IDF
plus one logistic regression classifier, a flat lexical Instinct ablation, and
the hybrid/hierarchical Instinct configuration used in the existing gauntlet.
`protocol.json` fixes settings before fitting or final scoring.

The first runner revision stopped before final inference when a full hybrid
skill exceeded the reader's header limit. A pre-score harness amendment retains
that deployment failure and evaluates this model in memory. Predictor settings,
data, thresholds, risk accounting and decision rule are unchanged. Reloadable
models undergo save/load parity checks; unsupported hybrid artifacts do not
receive a successful deployment claim.

## Data and phases

Use the published HWU64 train/valid/test split from Zhang et al.'s
[Few-Shot Intent Detection repository](https://github.com/jianguoz/Few-Shot-Intent-Detection/tree/fa9553312b1dba54b1be0aae2d30b13171f898e5/Datasets/HWU64).
HWU64 originates with Liu et al., *Benchmarking Natural Language Understanding
Services for Building Conversational Agents* (2019),
[original data](https://github.com/xliuhw/NLU-Evaluation-Data/tree/f6071b496b17d71e6eb43f543af0707f4ff30557)
([CC BY 4.0 license](https://github.com/xliuhw/NLU-Evaluation-Data/blob/f6071b496b17d71e6eb43f543af0707f4ff30557/LICENSE)).
Download pinned files and check all six SHA-256 hashes. Do not commit raw texts.

Normalize with casefold and whitespace collapse solely for duplicate detection.
Remove label-conflicting texts and **all** texts shared across source partitions;
collapse duplicates within a partition. Split the cleaned training partition by
fixed text hashes into 20 teaching and 10 trusted calibration examples per class;
use the rest only for independent gate qualification. Published validation is
threshold development; published test is final scoring. All five cohorts must
be disjoint. Hash ordering also fixes the final sequence. There are no authentic
timestamps: phase separation does not establish temporal robustness.

For each of five frozen seeds, corrupt exactly two teaching labels per class.
Wrong labels come from another intent in the same domain when possible. This is
persistent prompt-specific noise, not independent re-voting. All methods get the
same noisy lessons. All receive the same clean calibration/development/
qualification labels; those labels may calibrate/gate but **may not refit** the
predictor or vocabulary. Instinct retains its native strict review gate at
alpha=0.1; alpha is not treated as conditional accepted-error control. Its extra
threshold uses the same external qualification procedure as the classifier.

Choose one confidence threshold per method/seed on development data: maximum
coverage at <=1% empirical accepted error, at least 50 accepted examples, and
whole confidence ties. Qualify that **fixed** threshold on the remaining train
partition using a one-sided exact binomial upper bound, with failure probability
0.05/(4 methods * 5 seeds). If the bound exceeds 1%, accept nothing. Do not search
qualification thresholds, change settings, or use test answers for gate choice.
This finite-sample qualification assumes exchangeability with future in-scope
requests; it is not a guarantee under drift. No accepted samples means undefined
conditional error, not perfect accuracy. No qualification hits cannot certify a
cache gate.

Fit, calibrate, predict development/qualification, and save immutable gates and
predictions before opening final labels. Score final fresh requests exactly once.
No final request is learned or cached during this measurement. Also predict all
teaching inputs again as a **separate exact-repeat diagnostic**. Repeats are not
fresh evidence or used to qualify the fresh gate; applying it to repeats is not
a risk guarantee. The exact cache's raw repeat error reveals teacher poisoning.

Fallback teacher predictions are a paired, deterministic 10% corruption of the
final gold labels (rounded to nearest integer). They are generated only by the
scorer and never supplied to the fitted models. Report total cascade errors
separately: a 1% local budget does not turn a 10% fallible fallback into a 1%
end-to-end service. Report measured single-request local p50/p95, fit time,
serialized size, and *modeled* teacher latency at 50/500ms. No paid API is called;
API prices and real teacher latency are unknown.

## Decision rule and verification

Instinct earns its extra generalization machinery on this workload only if its
qualified fresh coverage exceeds the classifier by >=5 percentage points in at
least four of five seeds, and a paired two-stage bootstrap (noise seeds and test
requests) gives a positive 95% lower bound for the mean coverage gain. Report
final budget misses even when qualification passed. Also report whether the
lexical ablation suffices. Size/fit/latency costs remain explicit tradeoffs.
Failure is a bounded negative result, not proof that the architecture is useless.

`verify.py` imports neither Instinct nor the runner. It rebuilds the data split
and noisy lessons, recomputes gate thresholds, exact binomial bounds, raw/final
metrics and the paired comparison from recorded predictions. Source/prediction/
gate hashes are checked before metrics. The recorded per-request evidence has
text hashes, labels and predictions, not raw utterances. Focused tests include
confidence ties, boundary counts, overlap and tamper controls.

## Reproduce

Use a separate environment; these are experiment dependencies, not runtime deps.
The original run used Python 3.12.9, NumPy 2.5.4, SciPy 1.18.1 and scikit-learn
1.7.2. Set BLAS to one thread for consistent local timing.

```bash
uv venv --python 3.12.9 scratch/heldout-venv
uv pip install --python scratch/heldout-venv/bin/python numpy==2.5.4 scipy==1.18.1 scikit-learn==1.7.2 cryptography
scratch/heldout-venv/bin/python benchmarks/heldout_traffic/prepare.py --data-dir scratch/hwu64 --download
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  PYTHONPATH=src scratch/heldout-venv/bin/python benchmarks/heldout_traffic/run.py \
  --data-dir scratch/hwu64 --output-dir scratch/heldout-replay
scratch/heldout-venv/bin/python benchmarks/heldout_traffic/verify.py \
  --data-dir scratch/hwu64 --output-dir scratch/heldout-replay
```

The runner refuses an existing results directory. Replays use another output
directory; timings need not reproduce bit for bit. Source and gate receipts
identify the evaluated revision. The final report is added only after scoring.

Recorded result: [fresh-traffic report](REPORT.md). To check the archived evidence
without refitting, download the data and invoke `verify.py` against
`benchmarks/heldout_traffic/results`. The exact evaluated code is commit
`a660f1e1c59a25eace725c5dd6b2a7bc97f2308f`; later commits add reporting and
corruption checks without changing the predictor settings or frozen protocol.

Run focused checks, including the five archived-evidence corruption controls:

```bash
uv pip install --python scratch/heldout-venv/bin/python pytest
PYTHONPATH=src:benchmarks/heldout_traffic SYSTEM1_HELDOUT_DATA_DIR=scratch/hwu64 \
  scratch/heldout-venv/bin/python -m pytest -q benchmarks/heldout_traffic/test_protocol.py
```
