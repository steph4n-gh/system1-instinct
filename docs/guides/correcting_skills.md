# Correct, compare, and adopt a skill

When a skill makes a mistake or encounters a new edge case, you can update it in two ways:

1. **Apply an online rank-1 RLS update (`<1 ms`).** Call `engine.learn(prompt, {"team": "support"})` or `engine.learn_from_outcome(prompt, {"team": "support"}, outcome="resolved", reward=1.0)`, or wrap your upstream LLM or reviewer with `BoundedReviewFallback(..., auto_online_update=True)`. By default (`preserve_calibration=True`), `SystemOneEngine` applies a Sherman-Morrison rank-1 covariance update and scales calibration quantiles in place. To persist the updated skill, call `engine.save("support-route.s1m")`.
2. **Run a governed candidate session (`TeachingSession`).** Store your lessons, incumbent skill, and candidate revision in one directory. Record a correction, compile a candidate, compare its evaluation answers against the incumbent skill, and call `session.adopt()` only after the candidate passes your regression and error gates.

To install System 1 1.0.0 or newer, run:

```bash
python -m pip install 'system1-instinct>=1.0.0'
```

`TeachingSession` supports single-field and multi-field schemas (`ChoiceField`, `BooleanField`, `MultiChoiceField`, and `ScoreField`) when at least one field enables ambiguity escalation (the schema default). The session rejects schemas that disable escalation on all fields. By default, `TeachingSession` fits a `TfidfProjector` with up to 1,024 features, ridge regularization `0.1`, and strict decisions at `alpha=0.05`. You can also configure `selection="chi2"`, `char_ngrams`, `mondrian=True`, `familiarity_alpha`, `prototype_alpha`, `hierarchical` routing, and `split="auto"` stratified routing.

To launch the browser workspace from a repository checkout, run `python -m examples.teaching_by_doing`. Click **Review candidate** to check a proposed document-filing skill and **Adopt candidate** to activate a passing revision. See the [workspace guide](../../examples/teaching_by_doing/README.md) for details on its sample data.

## Prepare the decision splits

Keep your labeled examples in three disjoint splits:

| Split | Purpose |
|---|---|
| `teach` | Fit the numerical decision head |
| `calibrate` | Set review thresholds on held-out examples |
| `evaluate` | Compare candidate and incumbent answers on unseen examples |

Keep related customer threads and near-duplicate messages in the same split. Once you inspect an evaluation failure and add a lesson for it, treat that check as regression evidence and reserve fresh examples for final verification.

Create `lessons.json` with representative examples for every option in each split:

```json
{
  "teach": [
    ["Refund my payment", "billing"],
    ["The application crashes", "support"]
  ],
  "calibrate": [
    ["Please correct my invoice", "billing"],
    ["The software fails during startup", "support"]
  ],
  "evaluate": [
    ["Why was I billed twice?", "billing"],
    ["The checkout screen freezes", "support"]
  ]
}
```

The six rows above show the JSON layout. A real task requires enough calibration and evaluation rows to pass your coverage and accuracy gates.

## Assess a candidate in Python

To load `lessons.json` into a `TeachingSession` and compile a candidate skill, run:

```python
import json
from pathlib import Path
from system1 import ChoiceField, DecisionSchema, TeachingSession

class SupportRoute(DecisionSchema):
    team = ChoiceField(options=["billing", "support"])

session = TeachingSession(".system1/support-session", SupportRoute)
lessons = json.loads(Path("lessons.json").read_text())
session.record_many([
    {"input": text, "label": label, "split": split, "source": "human"}
    for split, pairs in lessons.items()
    for text, label in pairs
])
report = session.assess()
print(json.dumps(report, indent=2))
```

`session.assess()` compiles, saves, and reloads a candidate `.s1m` file, then evaluates both the candidate and any adopted incumbent skill on the `evaluate` split in strict mode. Assessment requires teaching examples for every option, at least two calibration examples, and at least one evaluation example.

To reopen an existing session directory, call `TeachingSession(".system1/support-session")`. Inspect `session.records` for stored lessons, `session.report` and `session.snapshot()` for assessment status, and `session.current_path` and `session.candidate_path` for artifact paths. Treat a session directory as a single-writer local workspace. `TeachingSession` caps each input at 8,192 characters.

## Inspect the assessment report

Check three metrics in the dictionary returned by `session.assess()`:

- **Raw accuracy (`raw_accuracy`).** The fraction of `evaluate` rows where the top guess matches the label, including guesses flagged for review. Low raw accuracy indicates conflicting labels, missing lessons, or insufficient features.
- **Coverage and accepted errors (`coverage` and `accepted_errors`).** `coverage` is the fraction of `evaluate` rows that the engine accepts without review (`is_ambiguous=False`). `accepted_errors` counts wrong predictions that the engine accepted without review.
- **Regressions (`regressions` and `raw_regressions`).** `regressions` counts `evaluate` rows that the incumbent skill answered right without review, but the candidate now gets wrong or sends to review. `raw_regressions` counts previously right top guesses that became wrong.

The report also includes `candidate` and `incumbent` summaries, per-row comparisons in `cases`, pattern summaries in `diagnostics`, and failed gate names in `reasons`.

By default, `session.assess()` requires `min_accuracy=0.8`, `min_coverage=0.5`, `max_accepted_errors=0`, and `max_regressions=0`. To set custom gates for your workload, pass them to `session.assess()`:

```python
report = session.assess(
    min_accuracy=0.8,
    min_coverage=0.5,
    max_accepted_errors=0,
    max_regressions=0,
)
```

Set your thresholds before inspecting candidate results. Do not lower a gate just to force a failing candidate through.

## Correct a lesson and reassess

Suppose the skill predicts `billing` for `"The payment page crashes"`. Under your routing rule, broken software belongs to `support`. To record the lesson and re-run assessment, call `session.record(...)`:

```python
session.record("The payment page crashes", "support", source="human")
report = session.assess()
print(json.dumps(report, indent=2))
```

`session.record(...)` writes to the `teach` split by default. Recording the same normalized input in the same split replaces its previous label instead of creating duplicate entries. If that input already exists in `calibrate` or `evaluate`, `session.record(...)` raises an error so a lesson cannot also serve as its own evaluation test. To delete an entry, call `session.remove(text)`.

If the new candidate fails a gate, the previously adopted skill remains active. Add several distinct examples of the boundary case rather than repeating one sentence, then check `session.assess()` again.

## Adopt a passing candidate

When `report["passed"]` is `True`, promote the candidate to the active skill by calling `session.adopt()`:

```python
session.adopt()
answer = session.predict("Please send my payment receipt")
print(answer)
```

`session.adopt()` verifies that the lessons, candidate artifact, and incumbent artifact have not changed since the passing `session.assess()` call. If you edit lessons or files after assessment, run `session.assess()` again before adopting. `session.engine` and `session.predict()` always run the adopted skill and never run an unapproved candidate. Before the first adoption, `session.engine` is `None`.

Explicit adoption in `TeachingSession` is separate from the automatic promotion policy in [`TypeSafeClient`](../typesafe.md) and from deterministic tool permissions in [`SystemOneGuard`](policy_guard.md).

<a id="the-same-workflow-from-the-cli"></a>
## Run the session workflow from the CLI

First, write your schema to `schema.json`:

```python
Path("schema.json").write_text(json.dumps(SupportRoute().to_dict()))
```

Next, initialize the session, load `lessons.json`, add a correction, and run assessment from the command line:

```bash
system1 teach .system1/support-session --schema schema.json
system1 teach .system1/support-session --dataset lessons.json
system1 teach .system1/support-session --record "The payment page crashes" --label support
system1 teach .system1/support-session --assess --json
```

Use `--split` to target `teach`, `calibrate`, or `evaluate` when recording a single example, or `--remove "text"` to delete a record. Running `system1 teach .system1/support-session --json` with no action flag prints the current session snapshot. The CLI exits with code `0` when assessment passes, `2` when assessment fails its gates or receives invalid CLI flags, and `1` on operational or data errors.

To adopt a passing candidate from the CLI, run:

```bash
system1 teach .system1/support-session --adopt --json
```

For multi-field schemas (`ChoiceField`, `BooleanField`, `MultiChoiceField`, and `ScoreField`), pass a dictionary of field labels to `session.record(text, {"field_a": ..., "field_b": ...}, split="auto")`, or pass `teaching_session=session` to `BoundedReviewFallback(teaching_session=session, review_fn=..., auto_record_lessons=True, auto_online_update=True)`. For lower-level compilation options, see [Teach and compile a decision skill](training_experts.md). For an empirical study of iterative corrections on real documents, see the [BBC correction workflow report](../../benchmarks/quality/document_workflow/README.md).

## Qualify a candidate on held-out traffic

To require statistical qualification on an independent cohort before adoption, pass `require_qualification=True` to `session.assess()` and then call `session.qualify(...)`:

```python
report = session.assess(require_qualification=True)
# Only a passing development assessment can proceed to qualification.
fresh = json.loads(Path("fresh-cases.json").read_text())
qualification = session.qualify(
    fresh,
    source="Reserved independently reviewed requests with collection dates",
    min_accepted_accuracy=0.95,
    min_coverage=0.80,
    confidence=0.95,
)
print(qualification["candidate"], qualification["reasons"])
# Review the qualification output before calling session.adopt().
```

Each item in `fresh-cases.json` is an object with `input`, `label`, and an optional `group` identifier. Related records can share a `group` within one split, but a `group` cannot span multiple splits. Qualification accepts at most one row per independent group.

Both the accepted-accuracy lower bound and the coverage lower bound must pass. `TeachingSession` computes exact one-sided binomial bounds and splits the total error budget `1 - confidence` across all qualification attempts in the session. Attempt `k` allocates `(1 - confidence) / (2 * k * (k + 1))` to each bound, so repeated attempts cannot reset the statistical error budget. Every qualification attempt freezes its cohort, source, and policy before running the first prediction, and consumes those rows and groups for the life of the session.

To run qualification and adoption from the CLI, execute:

```bash
system1 teach .system1/support-session --assess --require-qualification
system1 teach .system1/support-session --qualification fresh-cases.json --source "Independent collection"
system1 teach .system1/support-session --adopt
```

## Roll back to an approved revision

To inspect previously adopted revisions and restore an earlier `.s1m` artifact while keeping all recorded lessons, call `session.rollback(revision)`:

```python
print(session.history)
revision = session.history[0]["revision"]
session.rollback(revision)
print(session.decision_details("Please refund my payment"))
```

`session.history` lists each approved revision digest, timestamp, and qualification status. `session.rollback(revision)` verifies the archive digest, schema, and approval metadata before restoring the artifact, and marks any pending candidate assessment stale.

To view history and roll back from the CLI, run:

```bash
system1 teach .system1/support-session --history
system1 teach .system1/support-session --rollback FULL_APPROVED_REVISION_DIGEST
system1 teach .system1/support-session --predict "Please refund my payment" --json
```

## Select the choice solver

`TeachingSession` uses ridge regression by default. To fit `ChoiceField` heads with cross-entropy logistic optimization instead, pass `choice_solver="logistic"` when creating a new session:

```python
session = TeachingSession(".system1/logistic-session", SupportRoute, choice_solver="logistic")
```

To use `choice_solver="logistic"`, install `python -m pip install 'system1-instinct[teaching]'`, which adds SciPy for fitting. Saved `.s1m` artifacts run inference with NumPy alone and do not require SciPy at decision time.

## Evaluate and recalibrate a compiled skill

When you want to audit an existing `.s1m` artifact, `SystemOneEngine`, or `TeachingSession` directory on a held-out JSONL or JSON dataset without modifying recorded lessons, call `system1.evaluate()` or run `system1 eval`:

```python
import system1
from system1 import CompiledSystemOneModel

model = CompiledSystemOneModel.load(".system1/support-session/current.s1m")
recal = model.recalibrate("calib.jsonl", alpha=0.05)
model.save("support-recalibrated.s1m")

report = system1.evaluate(model, "heldout.jsonl", alpha=0.05)
print(report.overall_accuracy, report.accepted_accuracy, report.conformal_coverage)
print(report.coverage, report.abstain_rate, report.unfamiliar_rate)
```

`model.recalibrate()` and `engine.recalibrate()` refit temperature scaling and conformal nonconformity quantiles (`qhat`, `calibration_scores`, and `class_calibration_scores`) on held-out examples while leaving the linear decision hyperplanes (`weights`, `biases`, `P`, and `B`) and projector state untouched.

To run dataset evaluation and post-hoc recalibration from the CLI, execute:

```bash
system1 eval .system1/support-session/current.s1m heldout.jsonl --recalibrate calib.jsonl -o support-recalibrated.s1m --alpha 0.05
system1 eval support-recalibrated.s1m heldout.jsonl --alpha 0.05 --json
```

