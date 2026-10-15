# Teach and compile a decision skill

If you are building your first skill, start with the [Teach your first skill](first_skill.md) tutorial. To manage teaching, calibration, and evaluation splits with candidate promotion gates, use [`TeachingSession`](correcting_skills.md). Use this how-to guide when you want to call `SystemOneCompiler`, `TfidfProjector`, or the online learning methods on `SystemOneEngine` directly.

`SystemOneCompiler` and `SystemOneEngine` fit lightweight numerical decision heads over fixed text and numeric features, save them to `.s1m` files, and run them locally with NumPy. You can build a skill in three steps:

1. **Initialize a strict engine at zero examples.** Call `engine = SystemOneEngine(SupportRoute, strict_mode=True)`. Before you teach it, the engine abstains (`is_ambiguous=True`) on every input instead of guessing.
2. **Teach online or compile a batch dataset.** For online updates in `<1 ms`, call `engine.learn(prompt, labels)` or `engine.learn_from_outcome(prompt, labels, outcome="resolved", reward=1.0)`, or wrap an LLM with `BoundedReviewFallback(..., auto_online_update=True)`. For batch compilation, call `SystemOneCompiler(SupportRoute).compile(examples, augment=False)`.
3. **Save and reload the `.s1m` artifact.** Call `engine.save("support-route.s1m")` or `skill.save("support-route.s1m")`, and reload the artifact with `SystemOneEngine.load("support-route.s1m")` or `CompiledSystemOneModel.load("support-route.s1m")`.

## Compile and run a skill in Python

To run a complete batch compilation from the repository root, execute:

```bash
python examples/teach_skill.py
system1 decide "Please refund this payment" --model .system1/support-route.s1m --json
```

For multi-split demonstrations, see [the four teaching examples](../../examples/teaching/README.md). To compile eight examples in Python, save the `.s1m` artifact, and run a strict decision, execute:

```python
from system1 import ChoiceField, DecisionSchema, SystemOneEngine
from system1.compiler import CompiledSystemOneModel, SystemOneCompiler

class SupportRoute(DecisionSchema):
    team = ChoiceField(options=["billing", "support"])

examples = {"team": [
    ("Refund my payment", "billing"),
    ("Correct the invoice", "billing"),
    ("Subscription charged twice", "billing"),
    ("Payment receipt requested", "billing"),
    ("Debug this exception", "support"),
    ("Fix a software crash", "support"),
    ("API request fails", "support"),
    ("Investigate error logs", "support"),
]}

skill = SystemOneCompiler(SupportRoute).compile(examples, augment=False)
skill.save("support-route.s1m")

loaded = CompiledSystemOneModel.load("support-route.s1m")
engine = SystemOneEngine(loaded.schema, model=loaded, strict_mode=True)
decision = engine.decide("Please refund this payment")
print(decision.values)
print("Needs review:", decision.is_ambiguous)
```

Pass `augment=False` so `SystemOneCompiler` uses only your supplied examples and raises an error on invalid labels, unknown fields, or missing options. With `strict_mode=True`, `SystemOneEngine` uses held-out calibration scores to construct conformal prediction sets. Check `decision.is_ambiguous` on every call and route ambiguous inputs to a human reviewer or upstream LLM.

### Teach a skill online from zero examples

To teach an engine online without batch compilation, call `engine.learn_from_outcome(...)` or `engine.learn(...)`, then save with `engine.save(...)`. Both methods default to `preserve_calibration=True`:

```python
from system1 import ChoiceField, DecisionSchema, SystemOneEngine


class SupportRoute(DecisionSchema):
    team = ChoiceField(options=["billing", "support"])


engine = SystemOneEngine(SupportRoute, strict_mode=True)

for prompt, team in [
    ("Refund my duplicate credit card charge and fix the invoice", "billing"),
    ("Why was my subscription billed twice on this receipt?", "billing"),
    ("Please update our billing address and send the tax invoice", "billing"),
    ("Application crashes with a stack trace when calling the API", "support"),
    ("Debug this software exception and fix the failing endpoint", "support"),
    ("Production server error logs show a timeout crash on startup", "support"),
]:
    engine.learn_from_outcome(prompt, {"team": team}, outcome="resolved", reward=1.0)

engine.save("support-route.s1m")
loaded_engine = SystemOneEngine.load("support-route.s1m")
decision = loaded_engine.decide("My credit card was billed twice for the subscription invoice")
print(decision.values, "Needs review:", decision.is_ambiguous)
```

## Configure a TF-IDF or hybrid projector

By default, `SystemOneCompiler` uses a deterministic feature hashing projector. To fit an explicit vocabulary with supervised feature selection and character n-grams, pass a fitted `TfidfProjector` to `SystemOneCompiler`, or pass a `HybridProjector` for TF-IDF plus spectral `PPMI+SVD` features:

```python
from system1 import TfidfProjector

# Fit the vocabulary only on the teaching split.
projector = TfidfProjector.fit(
    [text for text, label in examples["team"]],
    selection="chi2",
    labels=[label for text, label in examples["team"]],
    char_ngrams=(3, 5),
)
calibration = {"team": [
    ("Please correct my invoice total", "billing"),
    ("The application crashes during startup", "support"),
]}
skill = SystemOneCompiler(
    SupportRoute,
    projector=projector,
    mondrian=True,
    familiarity_alpha=0.05,
    prototype_alpha=0.05,
).compile(
    examples, augment=False, calibration_exemplars=calibration,
)
skill.save("support-route.s1m")
loaded = CompiledSystemOneModel.load("support-route.s1m")
```

`TfidfProjector` caps its vocabulary at 2,048 terms by default, lowercases the first 8,192 characters of each input, and extracts Unicode word tokens of length 2 to 128. If an input contains no known vocabulary features, strict mode marks the decision as needing review.

Because `TfidfProjector` freezes its vocabulary and IDF weights inside the `.s1m` file, always fit the projector on your teaching split alone and pass a separate `calibration_exemplars` mapping to `compile(...)`. Fitting `TfidfProjector` on all rows before letting `compile(...)` split calibration internally leaks calibration vocabulary into the projector.

<a id="optional-classification-fitting"></a>
## Fit a choice head with logistic regression

To fit `ChoiceField` heads with cross-entropy optimization instead of the default ridge solver, install the `teaching` extra and pass `choice_solver="logistic"`:

```bash
python -m pip install 'system1[teaching]>=1.0.0'
```

In Python, pass `choice_solver="logistic"` and `regularization` to `SystemOneCompiler`:

```python
compiler = SystemOneCompiler(
    SupportRoute,
    projector=projector,
    choice_solver="logistic",
    regularization=0.1,
)
```

`choice_solver="logistic"` uses SciPy during compilation and writes standard NumPy weights to the `.s1m` file, so inference requires only NumPy. See the [Inbox Zero email comparison](../../examples/inbox_zero/README.md) for benchmark results comparing ridge and logistic heads.

## Teach your data

To compile a skill from the command line, format your dataset as a JSON field mapping:

```json
{
  "team": [
    ["Refund my payment", "billing"],
    ["Investigate this software crash", "support"]
  ]
}
```

You can also supply a JSON list of record objects with `{"prompt": "...", "labels": {"team": "billing"}}` and optional numeric `telemetry`. First, export your schema to `schema.json` with `Path("schema.json").write_text(json.dumps(SupportRoute().to_dict()))`, then run:

```bash
system1 compile --schema schema.json --dataset examples.json --output support-route.s1m --json
system1 decide "Please refund this payment" --model support-route.s1m --json
```

When you pass `--dataset`, `system1 compile` disables synthetic template augmentation by default and fails if the file is missing or malformed. Pass `--augment` only if you want synthetic templates added.

## Separate teaching, calibration, and evaluation data

Divide your labeled data into three disjoint partitions:

1. **Teaching.** Examples used to fit the projector and decision weights.
2. **Calibration.** Held-out examples used to fit temperature scaling and conformal quantiles.
3. **Evaluation.** Unseen examples held out of both teaching and calibration to measure accuracy, coverage, and accepted errors.

When you do not pass `calibration_exemplars`, `SystemOneCompiler` reserves roughly 25% of unique normalized `(prompt, telemetry)` inputs for calibration. For `ChoiceField`, `BooleanField`, and `MultiChoiceField`, the compiler splits that calibration fold again between temperature scaling and conformal quantile estimation.

Because `SystemOneCompiler` groups only exact normalized inputs, it cannot detect paraphrases, multi-message customer threads, or adjacent states from the same game episode. Split related groups yourself and pass `calibration_exemplars=held_out_examples` in Python or `--calibration-dataset calibration.json` on the CLI. Passing `calibration_split=0` in Python produces an uncalibrated skill that always requests review in strict mode.

Note that `system1 compile --dataset` expects a field mapping (`{"team": [[prompt, label], ...]}`) or record list, whereas `system1 teach --dataset` expects a split mapping (`{"teach": [...], "calibrate": [...], "evaluate": [...]}`). See [Run the session workflow from the CLI](correcting_skills.md#the-same-workflow-from-the-cli) for the `system1 teach` format.

To run the offline teaching benchmark comparison, execute:

```bash
python benchmarks/quality/evaluate_teaching.py
```

See [Benchmark suite index](../../benchmarks/quality/README.md#teaching-comparison) and [Public-workload evidence](../../benchmarks/quality/workloads/README.md) for full results.

## Update and recalibrate a skill

To fix an error in a deployed skill, correct the mislabeled example, add representative examples of the failure pattern, recompile, and verify on held-out evaluation data.

When you update a live engine in place:
- **High-level online learning (`engine.learn(...)`, `engine.learn_from_outcome(...)`, and `BoundedReviewFallback(..., auto_online_update=True)`).** Defaults to `preserve_calibration=True`, applying Sherman-Morrison directional gain scaling to preserve conformal quantiles across rank-1 updates.
- **Low-level tier updates (`learn_from_tier2(...)`, `learn_from_system2(...)`, and `learn_from_tier3(...)`).** Defaults to `preserve_calibration=False`, which invalidates calibration on modified heads so strict mode requests review until you call `engine.calibrate(...)` or pass `preserve_calibration=True`.

For multi-step synthetic bootstrapping, online updates, and skill composition, see [`examples/train_expert.py`](../../examples/train_expert.py).
