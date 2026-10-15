# Teacher observation and `TypeSafeClient` reference

`system1.compat.typesafe.Client` and `AsyncClient` implement the Jev structured decision interface: typed state and questions in, typed answers out. In `auto_cutover` mode, the client observes answers from an upstream teacher such as Jev, an OpenAI, Anthropic, or Gemini callback, or recorded labels. Once a compiled candidate skill passes its `PromotionPolicy`, the client switches the call site to local `.s1m` execution.

## Offline and live observation examples

The deterministic offline observation script runs from the repository root:

```bash
python examples/observe_routing.py
```

`examples/observe_routing.py` observes a synthetic support-ticket rule teacher under default promotion settings and disconnects the teacher after promotion. It then evaluates `40` held-out tickets locally, saves `routing.s1m`, verifies identical outputs after reload, and writes a JSON report to `.system1/observe-routing/`.

In both the offline run and the recorded live Jev run ([`benchmarks/quality/results/jev_live_cutover.json`](../benchmarks/quality/results/jev_live_cutover.json)), the client promoted after `357` observations:
- **Fitting split.** `215` observations.
- **Calibration split.** `71` observations, divided between temperature scaling and conformal quantile estimation.
- **Validation split.** `71` observations. The candidate matched the teacher and accepted all `71` of `71` observations on validation attempt `10`.
- **Post-cutover local evaluation.** The compiled skill matched the teacher and accepted all `40` of `40` held-out tickets in `~0.4 ms` per decision from a `5 KiB` `.s1m` file.

With `--teacher jev`, the script calls the live Jev API using `TYPESAFE_API_KEY` and makes billable API requests:

```bash
python examples/observe_routing.py --teacher jev --output-dir .system1/observe-jev
```

## Public-workload cutover results

On the six-intent assistant workload ([`benchmarks/quality/workloads/README.md`](../benchmarks/quality/workloads/README.md)), live Jev `1.13.0` and Gemini `2.5 Flash` each qualified after `358` observations. With teacher callbacks and network access disabled after cutover, each compiled `.s1m` skill accepted `171` of `180` unseen test requests with `167` of `171` (`97.7%`) right and `9` requests flagged for review.

On the banking workload (`412` observations), the client remained on the teacher and did not promote. On the SMS workload, the skill promoted under the default `80%` agreement threshold, while the stricter `95%` accepted-accuracy target (`PromotionPolicy(min_accepted_agreement=0.95)`) rejected promotion.

## Python integration contract

```python
from system1.compat.typesafe import Client, Choice

questions = {
    "queue": Choice(
        instructions="Route the request by its topic.",
        criteria={"billing": "Invoices and payments", "access": "Login and passwords"},
    )
}

# The upstream teacher answers until the candidate skill passes validation.
with Client(mode="auto_cutover", zero_egress=False) as client:
    response = client.systemone(state={"topic": "invoice"}, questions=questions)
    print(response.choices.queue.choice)
    print(client.is_cutover)
    # Export succeeds only after promotion completes.
    if client.is_cutover:
        client.export_model("queue.s1m")
```

- **Custom teacher callback (`baseline_handler`).** `Client(..., baseline_handler=teacher)` observes a Python callable instead of the Jev HTTP API. The callable receives `(state, questions)` and returns a mapping with an `"answers"` dictionary where each entry contains `"type"` and `"choice"`, `"noul"`, `"score"`, or `"value"`. Setting `zero_egress=True` blocks the built-in Jev HTTP transport, not custom code inside `baseline_handler`.
- **Reloading a promoted `.s1m` skill (`model_path`).** `Client(mode="auto_cutover", model_path="queue.s1m")` loads a saved `.s1m` file without a teacher or API key. The `questions` mapping passed to `client.systemone(...)` must match the saved schema. The `.s1m` artifact restores built-in hashed, `TfidfProjector`, and `HybridProjector` settings, weights, temperatures, conformal scores, and strict-mode flags. Custom external projectors passed to `Client(...)` must implement `projector_digest()`.
- **Multi-exemplar option criteria.** `Choice(criteria=...)` accepts a description string per option, a semicolon-separated string of phrases, or a list or tuple of example strings per option. Multiple phrases per option compile into sibling-contrastive sub-prototypes for zero-shot cold-start routing before cutover.

<a id="what-promotion-means"></a>
## What promotion means

By default, `auto_cutover` compiles candidates from observed teacher labels only (`augment=False`). Normalized duplicate prompts, request IDs, and lineage groups cannot cross the fitting, calibration, and validation folds. Requests from the same conversation thread or document share a group when supplied with matching lineage or group identifiers.

### Default `PromotionPolicy` thresholds

- **Complete-decision agreement.** Every field in the schema must match the teacher's output on a validation item, including exact `MultiChoice` sets.
- **Minimum agreement (`min_agreement=0.80`).** At least `80%` of validation groups must match the teacher, and agreement among locally accepted validation groups must also be at least `80%`.
- **Minimum local acceptance (`min_local_acceptance=0.80`).** The candidate skill must accept at least `80%` of validation groups locally (`is_ambiguous=False`).
- **Zero critical false-allows.** Security or critical fields must record zero false-allow errors on the validation split.
- **High-agreement override (`min_accepted_agreement`).** `PromotionPolicy(min_accepted_agreement=0.95)` requires both a `95%` point estimate and a one-sided binomial lower bound on accepted validation decisions. The [`1.0.2` quality round report](../benchmarks/quality/quality_round/README.md) records measurements with `dimension=2048`, `regularization=0.1`, and `cutover_threshold=400`.

### Statistical error-budget allocation

With `require_statistical_bound=True` (the default), validation attempt `k` allocates `(1 - confidence) / (2 * k * (k + 1))` to the agreement lower bound and to the acceptance lower bound. When `PromotionPolicy` sets `min_accepted_agreement`, attempt `k` allocates `(1 - confidence) / (3 * k * (k + 1))` across all three bounds. The sum across all attempts `k >= 1` stays within `1 - confidence`. Each validation attempt requires a fresh disjoint validation block.

`cutover_threshold=50` sets only the minimum observation count before validation attempt `1` can run. Under the default binomial bounds, promotion requires hundreds of independent observations. After promotion, `client.systemone(...)` executes locally and can still flag individual requests with `is_ambiguous` or `abstain`.

## Public API and SDK differences

`system1.compat.typesafe` targets the decision interface of [`typesafe-sdk-python` at commit `2ce5c65`](https://github.com/typesafe-ai/typesafe-sdk-python/tree/2ce5c65f13646cab6e6f782328194c9d85f3300a).

| Component | Behavior |
|---|---|
| `Client` and `AsyncClient` | Sync and async context managers, `close()` and `aclose()`, `systemone(...)`, constructor model defaults, `background_compile=True`, `wait_for_compilation()`, and `export_teaching_session(path)`. |
| Structured `state` | Accepts strings, JSON objects, and JSON arrays. Canonical JSON text feeds local feature extraction, and the client sends the original JSON value over HTTP when calling Jev. |
| Questions | Supports `Choice`, `Noul`, `NoulCriteria`, and ordinal `Score`, plus dicts and objects implementing `model_dump()`. |
| Responses | Provides `.answers`, `.choices`, `.nouls`, `.scores`, attribute access, and token usage on mutable response dictionaries. |
| Ordinal `Score` | Supports `2` to `10` ordered integer levels, returning per-level probabilities, an integer-keyed legend, and the probability-weighted mean score. Observed scalar scores map to the nearest level during fitting. |
| Local extensions | Supports continuous `ScoreField` ranges, `MultiChoice`, signed receipts, conformal sets, `background_compile=True`, `export_teaching_session(path)`, and `.s1m` export. |
| Unsupported SDK options | Custom HTTP transports, retry policies, custom headers, extra request bodies, custom Pydantic response models, and remote model-management endpoints raise `NotImplementedError`. |

## Artifact format and online updates

- **`.s1m` v2 format.** Newly saved skills write format v2 with Least Ambiguous Set Classifier (LAC) conformal scores (`1 - probability(label)`). System 1: Instinct 1.0.0 reads both v1 (APS) and v2 (LAC) artifacts and rejects corrupted headers, mismatched schemas, or nonfinite weights.
- **Calibration and online updates.** `engine.calibrate(...)` stores conformal scores in the compiled model. Low-level `learn_from_tier2(...)` defaults to `preserve_calibration=False` and invalidates calibration on updated heads until recalibration. Passing `preserve_calibration=True` to `learn_from_tier2(...)`, or calling `engine.learn(...)`, `engine.learn_from_outcome(...)`, or `BoundedReviewFallback(..., auto_online_update=True)`, applies Sherman-Morrison directional gain scaling to preserve conformal quantiles.
- **Exporting to `TeachingSession`.** `client.export_teaching_session(path)` writes all observed fitting, calibration, and validation records into a persistent [`TeachingSession`](guides/correcting_skills.md) directory.
