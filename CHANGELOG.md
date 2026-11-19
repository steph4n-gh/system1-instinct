# Changelog

## 1.3.0 - 2026-11-18

**System 1: Instinct (`1.3.0`)** - Adds soft-label distillation, compiled decision callables, an opt-in `.s1m` v3 format, a wire-compatible decision gateway, cost-aware calibration, explanations, active example selection, an opt-in guard shield, and a multi-tenant skill fleet. Tightens input validation across skill loading, online learning, receipts, and HTTP servers.

- **Soft Distillation & Few-Shot Routing**:
  - Add `SystemOneCompiler.compile_soft_distilled(prompts, soft_probs, ...)` and `soft_targets=` on `compile(...)` and `compile_and_save(...)` to fit `ChoiceField` and `BooleanField` heads to teacher probability rows.
  - Add `PragmaticCompositeProjector` (chi-squared-filtered TF-IDF plus structural text channels) and `SmartsCrossZeroFewRouter` (BM25F, criteria matching, and ridge routing from zero or a few examples per class, with no pretrained weights).
  - Add `OnlineMondrianConformal`, a per-class online quantile tracker. It targets long-run coverage under drift and does not give a finite-sample guarantee.
- **Compiled Callables & Opt-In `.s1m` v3**:
  - Add `SystemOneEngine.compile_callable()` / `compile_engine_callable(engine)` and `TypeSafeClient.compile_callable(questions)` / `compile_typesafe_callable(...)`. Each returns a `FastDecisionCallable` backed by a `PackedMultiQuestionKernel` that scores every `ChoiceField`, `BooleanField`, and `ScoreField` head in one matrix product. The callable returns `values`, `confidences`, and `is_ambiguous` and no receipt. It rebuilds its kernel when the engine's weights, schema, or effective `alpha` change.
  - Add `format_version=3` and `embed_vocab=True` to `save(...)`, `to_bytes(...)`, `compile_and_save(...)`, and `TypeSafeClient.export_model(...)`. The default stays `format_version=2`. `S1M\x03` allows up to 512 fields (`MAX_SKILL_FIELDS_V3`) and aligns arrays to 16 KiB pages. `load(..., mmap_mode=True)`, the default, memory-maps v3 files; call `close()` on the model or engine to release the mapping. v1 and v2 files load as before.
  - Add `use_cache=` to `SystemOneModel.forward_single(...)` and `evaluate(...)`, and `set_embedding_cache_maxsize` / `clear_embedding_caches` for the module-level embedding caches.
- **Universal Decision Gateway (`system1.compat.universal`)**:
  - Add `UniversalDecisionService`, `UniversalDecisionHTTPServer`, `create_universal_asgi_app`, and `mount_universal_gateway`. They accept Gemini Decision, Laya, and Jev payloads on `POST /v1beta/decisions`, `/v1/decisions`, `/v1/systemone`, and `/v1/laya/decide`, distill teacher probabilities into one `AutoCutoverLifecycle` per schema, and isolate schema drift. The HTTP server refuses non-loopback hosts. Requests with `NaN`, `Infinity`, or duplicate JSON keys are rejected, and body-size caps, header authentication, and `Origin` allowlists apply.
- **Score Cutover & Cost-Aware Calibration**:
  - `AutoCutoverLifecycle` accepts a `ScoreField` primary field. A score prediction agrees with the teacher when it lies within `CutoverConfig.score_tolerance` (default `0.15`) times the field's range. Add `post_compile_hook=`, `decide_local(..., accept_predicate=..., cache_key=...)`, and the `is_cutover` property.
  - Add `UtilityCalibrator`, which picks the Bayes expected-cost action from an `(actions, classes)` cost matrix and gates it with Conformal Risk Control. When the calibration data cannot certify the risk bound, `is_vacuous` is `True` and `predict` returns `safe_action`. `UtilityCascadeRouter` applies the same rule to a local, weak, and strong model ladder.
  - Add `CumulativeLogitOrdinalHead`, `SoftOrdinalHead`, `BinaryPlattBetaCalibrator`, `cqr_ordinal_intervals`, and `should_use_soft_ordinal`.
- **Explanations & Active Teaching**:
  - Add `SystemOneEngine.explain(prompt, field_name=None)` and `explain_all(prompt)` (`FastExplainer`), returning additive token attributions, exemplar witnesses, and counterfactual flips. `sign_compact_witness` and `verify_compact_witness` sign and check a compact witness with Ed25519.
  - Add `ActiveTeacherSelector`, which ranks unlabeled examples by RLS variance, margin entropy, and k-center coverage.
- **Opt-In Guard Shield**:
  - `SystemOneGuardHook` and `SystemOnePromptGuard` accept `shield_mode` (`"off"` by default, `"canonicalize"`, or `"sliding_window"`), and `SystemOneGuardHook` also accepts a `GuardShield` instance. The shield applies NFKC normalization, removes invisible characters, decodes embedded tool payloads, and scores sliding windows against unsafe prototypes before dispatch, with thresholds calibrated from benign text.
- **Multi-Tenant Skill Fleet**:
  - Add `MultiTenantSkillFleet`, which serves many tenants from one shared projector and base trunk with compact per-tenant arenas, a bounded token-hash cache (`token_cache_limit=4096`, `set_fleet_token_cache_limit`), and pickle-free delta serialization (`fleet_delta_to_bytes` / `fleet_delta_from_bytes`).

### Behavior changes

- **Optional keyword arguments for model callables come from the signature alone**:
  - `SystemOneEngine` chooses the optional keyword arguments it passes to `model.encode`, `model.forward_single`, `model.learn_from_tier2`, a head's `online_update`, and a conformal predictor's `predict_set` from that callable's signature. A callable with explicit parameters receives only the ones it names. A callable that accepts `**kwargs` receives every optional keyword argument.
  - The engine does not inspect what a `**kwargs` callable does with those arguments. A wrapper that forwards them to a stricter callable gets that callable's `TypeError`, and the error propagates to the engine's caller. `learn` leaves the engine's model version and lesson count unchanged when this happens.
  - In 1.2.0 the engine retried `encode` and `forward_single` with fewer keyword arguments after a `TypeError`. It no longer retries. Give a forwarding wrapper explicit parameters, or filter `**kwargs` before forwarding.
- **Stricter validation**:
  - Skill loading, compiled export, batch decisions, and online updates raise `ValueError` for non-finite or mis-shaped weights, biases, covariance matrices, exemplar banks, and calibration arrays, and for values that overflow `float32`.
  - `compute_receipt_digest` raises `ValueError` for non-finite numbers or for `confidences`, `conformal_sets`, or `probabilities` values that are not mappings. `create_decision_receipt` records non-finite confidences and probabilities as `0.0`. Receipt verification returns `False` for a receipt that contains `NaN` or `Infinity`.
  - The MCP Streamable HTTP server and the Inbox Zero ASGI app parse `Content-Type` strictly as `application/json` with RFC 9110 parameters and answer `415` to malformed values. The MCP server still accepts a request without `Content-Type`.
  - `SemanticSystemOneCache` rejects `str`, `bytes`, and non-numeric `embedding` values.

## 1.2.0 - 2026-11-13

**System 1: Instinct (`1.2.0`)** - Consolidates the internal `1.1` milestone and `1.2` runtime, calibration, cutover, and MCP server upgrades under `Development Status :: 5 - Production/Stable`.

- **`ActionLedger` WAL Durability & `O(1)` Verification Watermarks**:
  - Add configurable SQLite `durability_mode` (`"process_wal"` default, `"group_commit"`, `"strict_sync"`, `"high_throughput_normal"`) and `enforce_durability=True` (requiring persistent storage and rejecting `"high_throughput_normal"`).
  - Add atomic single-commit `ActionLedger.record_coalesced_read_only_execution(...)` for `RiskLevel.READ_ONLY` actions and `O(1)` hash-chain verification watermarks invalidated on `PRAGMA data_version` or `total_changes()` drift.
- **Vectorized Batch Inference**:
  - Add `SystemOneEngine.decide_batch(prompts, ...)` with single-call numerical parity across `DeterministicSemanticProjector`, `TfidfProjector`, `HybridProjector`, and `CompiledSystemOneModel`.
- **Discriminative Hybrid Representation (`v1_1`, `v1_2_fisher`, `v1_2_rbf`)**:
  - Add `HybridProjector.fit(...)` (`semantic_mode="v1_1"`), `HybridProjector.fit_v1_2(...)` (stemmed/skip-bigram/char-ngram TF-IDF, schema-anchored spectral SVD, and optional LOO-filtered class-centroid RBF-Cosine landmarks via `attach_rbf_landmarks` in `"v1_2_rbf"`), `compute_strict_loo_weights` leave-one-out ridge margin filtering blended with class-balanced ridge weights (`proto_blend=0.22` in `"v1_2_fisher"` and `"v1_2_rbf"`), and pure-NumPy `DecisionCalibrator.fit_imbalance_aware(...)`.
- **Selective, Factor-Graph Joint & Sequential Risk Control**:
  - Add `SelectiveRiskCalibrator` (`"ltt_fst"`, `"ltt_bonferroni"`, `"rcps_joint"`), `AdaptiveConformalTracker`, `JointSchemaCalibrator` / `SystemOneEngine.calibrate_joint(train_examples, calibration_examples, ...)` (pairwise factor-graph exact MAP decoding + Joint LTT qualification), and `SequentialRiskMonitor` (`"capital_ons"`, `"ons_betting"`, `"capital_eb"`, `"fixed_mixture"` e-processes).
- **Autonomous Cold-Start Cutover (`AutoCutoverLifecycle`)**:
  - Add `AutoCutoverLifecycle` and `CutoverConfig` (`COLD_START_ABSTAIN` -> `MICRO_COMPILED_WARMUP` -> `SELECTIVE_HOT_SWAP` -> `DRIFT_RECALIBRATE`) with quarantine-filtered disjoint train/calibration reservoirs, wired into `BoundedReviewFallback(..., auto_compile=True)` and `TypeSafeClient(..., auto_cutover_lifecycle=...)`.
- **Runnable MCP Stdio & Streamable HTTP Guard Servers**:
  - Add `MCPStdioProxy` (`system1 mcp-proxy` / `system1 serve --transport mcp-stdio`) and `MCPStreamableHTTPServer` (`system1 serve --transport mcp-http` / `system1 mcp-proxy --transport http`, MCP `2025-03-26` Streamable HTTP with constant-time Bearer auth, `Origin` allowlists, `Mcp-Session-Id` lifecycle, `--durability-mode`, and `GET /healthz`).
- **Offline Prequential Cassette Benchmark**:
  - Add `benchmarks/prequential/` (`system1.cassette.v1` socket-blocked causal replay harness).
- Add a frozen fresh HWU64/noisy-teacher comparison (`benchmarks/heldout_traffic/`) against an exact cache and a simple classifier at a matched 1% local error budget, with provenance receipts and independent verification (`verify.py` enforcing explicit `ValueError` checks even under `python -O`).
- Serialize `HybridProjector` spectral and learned subspace matrices as binary `.npz` arrays inside `.s1m` artifacts instead of nested JSON lists in the header (reducing 64-class HWU64 hybrid headers from `1.27 MB` to `20.9 KB` and restoring `5/5` reloadability), while retaining backward-compatible loading for legacy JSON-inline projector fields.
- Accelerate `SystemOneCompiler._solve_ridge` via an $O(N^3)$ dual/Woodbury solve when $N < D$, eliminate redundant `TfidfProjector` tokenization during compilation, and cache `HybridProjector` stem/hash tokens (cutting 64-class hybrid compile + calibration time from `4.68 s` to `1.74 s`).
- Bound cache vectors and aliases during eviction, invalidation and clearing, and promote semantic cache hits in LRU order.
- Validate complete online lessons across all schema fields before mutating any field head, exemplar bank, or model version.
- Preserve cold-start exemplar banks and online prototype state across skill saves and reloads with strict shape/finiteness validation. Newly saved exemplar arrays require the updated reader; existing skills remain readable.
- Recheck permission and context synchronously on submission and before deferred async reviewers start, and classify deterministic input/validation errors inside async reviewers without penalizing the circuit breaker.
- Bound every skill file read and account for covariance, calibration objects, exemplar state and spectral reconstruction during model admission. Loaded heads no longer rebuild discarded seed embeddings.
- Include FastAPI/httpx test dependencies, refresh the lockfile and constrain BLAS test threads in CI.

## 1.0.0 — 2026-10-14

**System 1: Instinct (`1.0.0`)** — Sub-millisecond non-autoregressive decision runtime, zero-shot unsupervised adaptation (`adapt_zero_shot`), and calibration-preserving online RLS skills.

- **Zero-Shot Unsupervised Adaptation & Contrastive Multi-Class Prototype Learning**:
  - `CompiledSystemOneModel.adapt_zero_shot(prompts, margin_threshold=..., learning_rate=...)` and `SystemOneEngine.adapt_zero_shot(...)` for unsupervised test-time prototype adaptation without teacher calls or ground-truth labels.
  - Contrastively-centered multi-class exemplar banks (`DecisionFieldHead._exemplar_bank`) and online shot prototypes (`_online_shots_p`, `_online_shots_c`) so 0-shot, 1-shot, 3-shot, and 5-shot `learn_from_tier2` updates monotonically improve accuracy across 24–77 class routing tasks (`BANKING77` and `CLINC150`).
- **Representation, Hierarchical Routing & Dual-Gate OOD Abstention**:
  - `TfidfProjector.fit(..., selection="chi2", labels=..., char_ngrams=(3, 5))` for supervised $\chi^2$ feature selection and character n-gram typo resilience, and `HybridProjector.fit(...)` for corpus-fitted spectral (`PPMI + truncated SVD`) dense embeddings combined with $\chi^2$ sparse lexical features.
  - Two-stage hierarchical coarse-to-fine routing (`hierarchical=True`), Mondrian class-conditional conformal calibration (`mondrian=True`), and dual OOD gating (`familiarity_alpha` and `prototype_alpha`) in `SystemOneCompiler`, `TeachingSession`, and the `system1 compile` / `system1 teach` CLI commands.
- **Multi-Field `TeachingSession` & Closed-Loop Boundary Flywheel**:
  - `TeachingSession` supports `ChoiceField`, `BooleanField`, `MultiChoiceField`, `ScoreField`, and multi-field schemas with `split="auto"` stratified routing and file-locked concurrent access.
  - `BoundedReviewFallback` (`auto_record_lessons=True`, `auto_online_update=True`) and `preserve_calibration=True` in `learn_from_tier2` so boundary escalations update local skills in sub-millisecond time while preserving conformal calibration.
- **Drop-In Replacements & Benchmarks**:
  - Six step-by-step procedural drop-in guides in [docs/drop_in/](docs/drop_in/README.md) and runnable scripts in [examples/drop_in/](examples/drop_in/README.md) for **Semantic Router**, **RouteLLM**, **Instructor / BAML**, **Llama-Prompt-Guard / Guardrails**, **Live LLM Auto-Cutover**, and **`poke-env` Pokémon Showdown**.
  - Reproducible evaluation suites in [instinct_gauntlet_report.md](benchmarks/quality/instinct_gauntlet_report.md), [drop_in_and_adaptation_report.md](benchmarks/quality/drop_in_and_adaptation_report.md), and [poke_env_showdown_report.md](benchmarks/quality/poke_env_showdown_report.md).
