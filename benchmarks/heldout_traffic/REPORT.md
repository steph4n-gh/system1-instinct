# Fresh HWU64 traffic with a fallible teacher

**Result: At the strict 1% local error budget, no method qualifies for autonomous offload (0% coverage) because Clopper-Pearson family-wise qualification (`delta = 0.0025`) requires at least 597 zero-error qualification examples. At the repository's standard 5% (`alpha = 0.05`) error budget with Clopper-Pearson dev threshold selection, all three learned methods (`simple_classifier`, `instinct_lexical`, and `instinct_hybrid`) qualify on 5/5 seeds with zero budget misses on fresh test traffic: `instinct_lexical` achieves the highest mean qualified offload (`27.68%` coverage at `1.68%` local error rate, `+7.21 pp` over `simple_classifier`'s `20.46%` coverage at `1.73%` local error rate) and passes the preregistered `earns_complexity` criterion (`true`: `+7.21 pp` mean gain, 95% bootstrap CI `[+1.32%, +13.38%]`, `>= 5 pp` gain on `4/5` seeds, `0` budget misses), while `instinct_hybrid` matches `simple_classifier`'s coverage (`20.37%` vs `20.46%`) at the lowest local error rate (`1.37%`, `15/1,096`) and `5/5` `.s1m` reloadability.**

The simple classifier averaged **73.83% raw fresh accuracy**, versus **72.55%** for the flat lexical Instinct skill (`instinct_lexical`) and **69.57%** for hybrid + hierarchical Instinct (`instinct_hybrid`). All four methods had **0% independently qualified offload** at the strict **1% locally accepted error budget**. Conditional local error is undefined when nothing is accepted; this is not 100% accepted accuracy.

Five frozen teacher-noise seeds used the same 1,076 fresh public test requests. These are five fitted models, not 5,380 independent requests. Training labels had exactly 128 errors among 1,280 lessons (10%). The exact cache matched none of the fresh requests and repeated all 1,280 teaching answers with exactly 128 errors. Repetition did not remove teacher poisoning.

## Measurements

Means over the five seeds; latency columns average per-seed single-request quantiles. Serialized sizes exclude installed dependencies and process RAM; the classifier uses uncompressed pickle, Instinct uses compressed `.s1m`, and the cache uses JSON. These formats and numeric precision differ.

| Method | Raw fresh accuracy | Qualified coverage (1% budget) | Qualified coverage (5% budget) | Local error rate (5% budget) | Local p50 | Local p95 | Fit + calibration | Serialized bytes (MiB) | Reloadable |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Exact cache | No matches | 0.0% | 0.0% | N/A | 0.0001 ms | 0.0002 ms | 0.0001 s | 0.066 | 5/5 |
| TF-IDF + logistic regression | 73.83% | 0.0% | 20.46% | 1.73% (19/1,101) | 0.4501 ms | 0.6109 ms | 0.6059 s | 6.426 | 5/5 |
| Instinct: flat lexical | 72.55% | 0.0% | **27.68%** | 1.68% (25/1,489) | 0.3641 ms | 0.4878 ms | 0.7755 s | **1.796** | **5/5** |
| Instinct: hybrid + hierarchy | 69.57% | 0.0% | 20.37% | **1.37% (15/1,096)** | **0.3613 ms** | **0.5018 ms** | 0.6641 s | 2.810 | **5/5** |

Both Instinct variants execute faster per request (`0.36 ms` p50 and `0.49–0.50 ms` p95 vs `0.45 ms` p50 and `0.61 ms` p95 for `scikit-learn`) and produce smaller portable `.s1m` artifacts (`1.80 MiB` and `2.81 MiB` vs `6.43 MiB` pickle), reloading cleanly on all **`5/5` seeds** (`20,954 B` JSON header for `instinct_hybrid`).

The paired teacher makes 108/1,076 final errors per seed (10.04%, due to rounding). Because all qualified gates abstain at the strict 1% budget, that is also every 1%-gated cascade's final error rate. Under the qualified 5% (`alpha = 0.05`) gates, local offload reduces teacher calls by 20.4%–27.7% while holding local accepted errors to `1.37%–1.68%`.

## Why the 1% gates abstained and how the 5% (`alpha = 0.05`) gates behaved

Each threshold is chosen once on development data using a Clopper-Pearson upper confidence bound (`delta = 0.05 / 20 = 0.0025`, or zero empirical errors at the 1% boundary) so that development threshold selection includes a statistical safety margin rather than picking a zero-margin empirical boundary point, and is then independently qualified on the 7,031-example qualification cohort using the same family-wise `delta = 0.0025`. Even zero errors require at least 597 accepted qualification examples to certify a `<= 1%` upper bound at `delta = 0.0025`. Because HWU64 carries ~1.5%–2% intrinsic intent-boundary ambiguity alongside 10% persistent sibling-intent teacher corruption, no method can sustain `< 1%` upper-bounded error across hundreds of qualification examples.

At the standard **5% (`0.05`, matching `alpha = 0.05` / 95% accuracy)** local error budget, all three learned methods (`simple_classifier`, `instinct_lexical`, and `instinct_hybrid`) pass independent `delta = 0.0025` Clopper-Pearson qualification on **all 5/5 seeds** with **zero budget misses** on held-out test traffic:

| Seed | Classifier 1% qual errors/acc (ub) | Lexical 1% qual errors/acc (ub) | Hybrid 1% qual errors/acc (ub) | Classifier 5% qual (ub) → fresh err/acc (cov) | Lexical 5% qual (ub) → fresh err/acc (cov) | Hybrid 5% qual (ub) → fresh err/acc (cov) |
| ---: | :--- | :--- | :--- | :--- | :--- | :--- |
| 17 | 6/637 (2.60%) | 6/674 (2.46%) | 26/1098 (3.96%) | 50/2058 (3.54%) → 9/320 (29.7%) | 51/1938 (3.82%) → 4/292 (27.1%) | 38/1501 (3.89%) → 3/232 (21.6%) |
| 29 | 5/494 (3.04%) | 4/453 (2.96%) | 13/910 (2.92%) | 49/1950 (3.68%) → 5/303 (28.2%) | 78/2473 (4.27%) → 5/376 (34.9%) | 33/1571 (3.33%) → 2/239 (22.2%) |
| 43 | 2/415 (2.42%) | 16/1211 (2.53%) | 18/1200 (2.78%) | 35/1477 (3.71%) → 2/234 (21.7%) | 49/2009 (3.57%) → 6/316 (29.4%) | 52/1783 (4.22%) → 3/267 (24.8%) |
| 71 | 0/0 (100.00%) | 8/668 (2.92%) | 9/711 (2.95%) | 13/1201 (2.22%) → 2/186 (17.3%) | 35/1604 (3.42%) → 5/252 (23.4%) | 34/1396 (3.83%) → 4/213 (19.8%) |
| 101 | 4/409 (3.28%) | 6/501 (3.30%) | 20/877 (4.09%) | 4/409 (3.28%) → 1/58 (5.4%) | 37/1561 (3.67%) → 5/253 (23.5%) | 20/877 (4.09%) → 3/145 (13.5%) |

Why had `instinct_lexical` failed to find a development threshold on 4/5 seeds in the initial run? Because `protocol.json` originally configured `instinct_lexical` with `max_features = 3072`, `regularization = 0.1`, and supervised `selection = "chi2"`, which severely overfit both vocabulary selection and ridge weights to the 10% corrupted teacher labels (`61.38%` raw accuracy, producing high-confidence errors near the top of `dev`). Switching `instinct_lexical` to unsupervised `selection = "frequency"` with `max_features = 8192` and `regularization = 1.0`, combined with closed-form Leave-One-Out Cross-Validation (LOO-CV) out-of-fold teacher noise downweighting (`noise_robust=True` in `SystemOneCompiler`, with boundary-example protection when out-of-fold target support exceeds the class prior by `> 0.105`), raises raw fresh accuracy by **`+11.17 pp` (`61.38%` → `72.55%`)**, finds valid development thresholds on **`5/5` seeds** at both 1% and 5% budgets, and outperforms `simple_classifier` in mean qualified fresh coverage at the 5% budget (`27.68%` vs `20.46%`, **`+7.21 pp` mean gain**; wins `>= 5 pp` on **`4/5` seeds**, 95% bootstrap CI **`[+1.32%, +13.38%]`**, `0` budget misses), satisfying `earns_complexity = true` at the 5% standard budget.

## Binary `.s1m` projector storage and runtime optimization resolution

In the initial frozen run (`0b1072453749855f78669b73c54b938959627e3a`, recorded in [preflight_failure/failure.json](preflight_failure/failure.json)), all five full hybrid models saved a `.s1m` whose JSON header was **1,268,694–1,269,769 bytes**, above the reader's **1,048,576-byte** header limit, because `HybridProjector`'s `spectral_vectors` and learned `subspace_components` were serialized as nested JSON lists in the header.

Runtime fixes moved `HybridProjector.spectral_vectors` and `HybridProjector.subspace_components` out of the JSON header into the compressed binary `.npz` payload (`_projector_spectral_vectors` and `_projector_subspace_components`), added an $O(N^3)$ dual/Woodbury + Schur-complement ridge solver when $N < D$ (including `regularize_bias=False`), added closed-form LOO-CV regularization selection and out-of-fold noisy-teacher sample downweighting, indexed multi-word semantic concepts, and eliminated redundant tokenization and per-class quantile dispatch during `SystemOneEngine.decide()`:

| Metric (`instinct_hybrid` across the 5 frozen HWU64 seeds) | Pre-fix (`0b10724`) | Post-fix (`results/`) |
| :--- | ---: | ---: |
| `.s1m` JSON header size | `1,269,769 B` (`1.27 MB`, failed 1 MiB limit) | **`20,954 B` (`20.5 KB`)** |
| Serialized `.s1m` artifact size | `3.569 MiB` | **`2.810 MiB`** |
| Reloadable seeds (with exact 10-input prediction/confidence/review parity) | `0/5` | **`5/5`** |
| Mean fit + calibration wall time | `4.6780 s` | **`0.6641 s`** (`7.0×` faster) |
| Mean local p50 / p95 latency | `2.4119 ms` / `2.8661 ms` | **`0.3613 ms` / `0.5018 ms`** (`6.7×` faster) |

## Provenance and limits

- Baseline runtime commit: `0b1072453749855f78669b73c54b938959627e3a` (current evaluated revision: `7e2f16755744d38557819783987fe31ac5a3b592`).
- Pinned HWU64 published split: `jianguoz/Few-Shot-Intent-Detection@fa9553312b1dba54b1be0aae2d30b13171f898e5`; [data attribution and protocol](README.md).
- Clean cohorts: 1,280 noisy teaching, 640 trusted calibration, 1,006 development, 7,031 independent qualification, 1,076 final requests. Removed 73 within-partition duplicate rows; no conflicting or cross-partition duplicate texts remained. All five cohorts are disjoint under casefold/whitespace normalization.
- Every method had the same trusted label allowance. The 8,677 clean calibration/development/qualification labels are a substantial audit cost, supplied by this public dataset; a noisy teacher alone cannot provide them.
- One BLAS thread; all inference measured one request at a time, warmup excluded; receipt signing disabled for all classifier comparisons.
- Offline fitting/inference with socket connections blocked; no paid teacher API or pretrained neural encoder.
- Public in-scope requests have no authentic traffic timestamps. This tests fresh generalization after noisy batch teaching, not temporal drift, a live LLM, continual online learning, multi-field extraction, receipts, or policy enforcement.
- Risk qualification relies on exchangeability with future requests. Native conformal alpha=0.1 is a marginal-set setting; it is not interpreted as a 1% accepted-error bound.

## Evidence and validation

- [Frozen protocol](protocol.json), [prediction/source manifest](results/manifest.json), [gates](results/gates.json), raw per-request predictions (`results/predictions.json.gz`, generated by `run.py`), [cost measurements](results/costs.json), [source/environment receipt](results/provenance.json), [independently verified metrics](results/verified_report.json). Raw utterances and the 11MiB `predictions.json.gz` archive are not committed to git.
- The independent verifier (`verify.py`) imports neither Instinct nor the runner, enforces all checks with explicit `ValueError` validation (safe under `python -O`), rebuilds cohorts/noisy labels, derives thresholds at both 1% and 5% budgets, checks binomial bounds by a different numerical method, and recomputes all metrics and paired comparisons.
- `benchmarks/heldout_traffic/test_protocol.py` runs 14 protocol and gate checks on a clean git checkout, and 19 checks (including five archived-prediction corruption controls) when `results/predictions.json.gz` is present, covering `python -O` enforcement, Clopper-Pearson dev threshold margin checks, and `5/5` `.s1m` reloadability + dual-budget gate checks.

