# Fresh HWU64 traffic with a fallible teacher

**Result: Instinct did not earn its extra generalization machinery on this workload.**

The simple classifier averaged **73.83% raw fresh accuracy**, versus **70.37%** for hybrid Instinct and **61.38%** for the flat lexical ablation. All four methods had **0% independently qualified offload** at the requested **1% locally accepted error budget**. Conditional local error is undefined when nothing is accepted; this is not 100% accepted accuracy.

Five frozen teacher-noise seeds used the same 1,076 fresh public test requests. These are five fitted models, not 5,380 independent requests. Training labels had exactly 128 errors among 1,280 lessons (10%). The exact cache matched none of the fresh requests and repeated all 1,280 teaching answers with exactly 128 errors. Repetition did not remove teacher poisoning.

## Measurements

Means over the five seeds; latency columns average per-seed single-request quantiles. Serialized sizes exclude installed dependencies and process RAM; the classifier uses uncompressed pickle, Instinct uses compressed `.s1m`, and the cache uses JSON. These formats and numeric precision differ.

| Method | Raw fresh accuracy | Qualified fresh coverage | Local p50 | Local p95 | Fit + calibration | Serialized bytes (MiB) | Reloadable |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Exact cache | No matches | 0.0% | 0.0001 ms | 0.0002 ms | 0.0001 s | 0.066 | 5/5 |
| TF-IDF + logistic regression | 73.83% | 0.0% | 1.0253 ms | 1.3046 ms | 0.6160 s | 6.426 | 5/5 |
| Instinct: flat lexical | 61.38% | 0.0% | 1.7722 ms | 2.7174 ms | 0.7337 s | 0.662 | 5/5 |
| Instinct: hybrid + hierarchy | 70.37% | 0.0% | 2.4119 ms | 2.8661 ms | 4.6780 s | 3.569 | 0/5 |

Hybrid Instinct was slower to fit and execute than the classifier, though its serialized file was smaller. No method saved a teacher call under the qualified policy. At modeled teacher latencies of 50 or 500ms, every cascade still calls the teacher for all fresh requests and adds its own local overhead. These are latency scenarios, not measured API latency, billing or paid calls.

The paired teacher makes 108/1,076 final errors per seed (10.04%, due to rounding). Because all qualified gates abstain, that is also every qualified cascade's final error rate. A 1% local budget does not establish a 1% end-to-end budget.

## Why the gates failed

Each threshold was chosen once on development data at <=1% empirical accepted error, then independently qualified. A one-sided exact binomial upper bound had to be <=1%, using failure probability 0.05/20 = 0.0025 for the family of four methods and five seeds. No threshold was retuned on qualification or final answers. Even zero errors require at least 597 accepted qualification examples with this accounting.

Counts below describe the **rejected candidate thresholds**, not deployed acceptance. Final candidate performance is diagnostic only and cannot rescue a failed qualification gate.

| Seed | Classifier qualification errors/accepted (upper bound) | Hybrid qualification errors/accepted (upper bound) | Classifier fresh candidate errors/accepted | Hybrid fresh candidate errors/accepted |
| ---: | :--- | :--- | :--- | :--- |
| 17 | 6/674 (2.46%) | 37/1112 (5.13%) | 1/104 | 4/179 |
| 29 | 15/1164 (2.52%) | 24/909 (4.50%) | 3/176 | 2/146 |
| 43 | 35/1477 (3.71%) | 46/1292 (5.26%) | 2/234 | 4/198 |
| 71 | 13/1201 (2.22%) | 38/1350 (4.32%) | 2/186 | 6/206 |
| 101 | 10/814 (2.76%) | 35/1012 (5.39%) | 2/135 | 7/168 |

Hybrid qualification point errors were 2.64–3.56%, already above the budget before accounting for uncertainty. The classifier had 0.89–2.37% point errors; its upper bounds also exceeded the budget. The flat lexical model found no eligible development threshold in four seeds; seed 71 qualified 563 candidates with 21 errors (6.56% upper bound), then had 3/78 errors on the final candidate set. The cache had no fresh matches to qualify.

The preregistered complexity rule required >=5 percentage points of qualified fresh coverage gain in at least four seeds and a positive paired bootstrap lower bound. Both Instinct variants had zero gain in every seed and failed this rule. No final answer informed thresholds or model settings.

## Deployment failure

All five full hybrid models saved a `.s1m` whose JSON header was **1,268,694–1,269,769 bytes**, above the reader's **1,048,576-byte** header limit. Saving succeeded; reloading raised `ValueError: Saved skill header exceeds size limit`. Learned dense subspace components are serialized as nested JSON lists. The files averaged 3.57MiB overall; none was a deployable portable skill in this configuration.

The first run stopped at this save/load check **before final inference or scoring** ([failure receipt](preflight_failure/failure.json)). A harness-only amendment retained unsupported models in memory and recorded the reload failure. It did not relax admission limits, change predictors, tune thresholds, or alter the data. Reloadable models passed prediction/confidence/review parity checks on ten development inputs. This is a separate execution finding from the negative accuracy result.

## Provenance and limits

- Runtime under evaluation: `0b1072453749855f78669b73c54b938959627e3a` (PR #1 head).
- Initial protocol commit: `87379662ed4fff9e5c8d4948781ac9787b9b07e8`.
- Pre-score amended runner commit: `a660f1e1c59a25eace725c5dd6b2a7bc97f2308f`; predictor settings and protocol hash unchanged.
- Pinned HWU64 published split: `jianguoz/Few-Shot-Intent-Detection@fa9553312b1dba54b1be0aae2d30b13171f898e5`; [data attribution and protocol](README.md).
- Clean cohorts: 1,280 noisy teaching, 640 trusted calibration, 1,006 development, 7,031 independent qualification, 1,076 final requests. Removed 73 within-partition duplicate rows; no conflicting or cross-partition duplicate texts remained. All five cohorts are disjoint under casefold/whitespace normalization.
- Every method had the same trusted label allowance. The 8,677 clean calibration/development/qualification labels are a substantial audit cost, supplied by this public dataset; a noisy teacher alone cannot provide them.
- Hardware: Apple M4 Pro, 24GiB RAM; Python 3.12.9, macOS 27.0 arm64; NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.7.2; one BLAS thread; all inference measured one request at a time, warmup excluded; receipt signing disabled for all classifier comparisons.
- Offline fitting/inference with socket connections blocked; no paid teacher API or pretrained neural encoder.
- Public in-scope requests have no authentic traffic timestamps. This tests fresh generalization after noisy batch teaching, not temporal drift, a live LLM, continual online learning, multi-field extraction, receipts, or policy enforcement.
- Risk qualification relies on exchangeability with future requests. Native conformal alpha=0.1 is a marginal-set setting; it is not interpreted as a 1% accepted-error bound.

This result supports withholding general offload forecasts and measuring a simple classifier first. It does not justify removing every Instinct component or claiming that the simple classifier is ready to deploy at 1% error. Both need an independently qualified acceptance policy on their target workload.

## Evidence and validation

- [Frozen protocol](protocol.json), [prediction/source manifest](results/manifest.json), [gates](results/gates.json), [raw per-request predictions](results/predictions.json.gz), [cost measurements](results/costs.json), [source/environment receipt](results/provenance.json), [independently verified metrics](results/verified_report.json). Raw utterances are not committed.
- The independent verifier imports neither Instinct nor the runner, rebuilds cohorts/noisy labels, derives thresholds, checks binomial bounds by a different numerical method, and recomputes all metrics and the paired comparison.
- 16 experiment checks passed, including five corruption controls: changed evidence hash, falsely enabled gate despite recomputed manifest, duplicate final ID despite recomputed manifest, silently repaired noisy cache entry, and altered final answer key.
- Existing runtime suite: 1,170 passed, five optional skips on Python 3.12. No runtime algorithm changed in this PR.

Reproduction and explicit negative-control commands are in the [protocol README](README.md). The full raw prediction archive is approximately 11MiB; it includes development and qualification evidence so threshold selection is reviewable, not just final aggregates.
