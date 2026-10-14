# Changelog

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
