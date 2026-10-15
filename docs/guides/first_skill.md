# Teach your first skill

In this tutorial, we build a local routing skill that sends payment questions to `billing` and broken-software reports to `support`. We do not need an LLM, an API key, or a GPU.

If you prefer a browser interface, open the [document teaching workspace](../../examples/teaching_by_doing/README.md). In this tutorial, we teach the skill from Python and an editable JSON file.

## Step 1: Teach six examples in Python

First, let us create a two-option `SupportRoute` schema in Python, watch `SystemOneEngine` abstain at zero examples, teach it six labeled messages, and save the `.s1m` skill:

```python
from system1 import ChoiceField, DecisionSchema, SystemOneEngine


class SupportRoute(DecisionSchema):
    team = ChoiceField(options=["billing", "support"])


# 1. Start at 0 examples. Strict mode abstains instead of guessing.
engine = SystemOneEngine(SupportRoute, strict_mode=True)
before = engine.decide("Duplicate charge on my credit card invoice", record_receipt=False)
print("Before teaching:", before.values["team"], "is_ambiguous=", before.is_ambiguous)

# 2. Teach 6 examples online (<1 ms per rank-1 Sherman-Morrison update).
for message, team in [
    ("Refund my duplicate credit card charge and fix the invoice", "billing"),
    ("Why was my subscription billed twice on this receipt?", "billing"),
    ("Please update our billing address and send the tax invoice", "billing"),
    ("Application crashes with a stack trace when calling the API", "support"),
    ("Debug this software exception and fix the failing endpoint", "support"),
    ("Production server error logs show a timeout crash on startup", "support"),
]:
    engine.learn_from_outcome(message, {"team": team}, outcome="resolved", reward=1.0)

# 3. Check an unseen billing phrasing and an out-of-scope question.
after = engine.decide("My credit card was billed twice for the subscription invoice", record_receipt=False)
print("After teaching:", after.values["team"], "is_ambiguous=", after.is_ambiguous)

oos = engine.decide("What is the best recipe for sourdough bread?", record_receipt=False)
print("Out of scope: is_ambiguous=", oos.is_ambiguous)

# 4. Save and reload the .s1m skill.
engine.save(".system1/support-route.s1m")
reloaded = SystemOneEngine.load(".system1/support-route.s1m")
```

When we run this snippet, we see three outputs:

```text
Before teaching: billing is_ambiguous= True
After teaching: billing is_ambiguous= False
Out of scope: is_ambiguous= True
```

At zero examples, `before.is_ambiguous` is `True`. After six online examples, `after.values["team"]` resolves to `"billing"` with `is_ambiguous=False`, while the sourdough question abstains with `is_ambiguous=True`.

## Step 2: Compile a skill from a JSON file

Now, let us compile a skill from a JSON lesson file on disk. If you have not installed the package yet, run the commands in the [README quickstart](../../README.md#quickstart) first.

From the repository root, run `examples/teach_skill.py`:

```bash
python examples/teach_skill.py
```

You should see this output:

```text
Suggested team: billing
Needs review: True
```

The script reads [first_skill.json](../../examples/teaching/first_skill.json), compiles its eight examples, writes `.system1/support-route.s1m`, and evaluates a test prompt:

```json
{
  "team": [
    ["Refund my payment", "billing"],
    ["Correct the invoice", "billing"],
    ["Subscription charged twice", "billing"],
    ["Payment receipt requested", "billing"],
    ["Debug this exception", "support"],
    ["Fix a software crash", "support"],
    ["API request fails", "support"],
    ["Investigate error logs", "support"]
  ]
}
```

Each pair in `"team"` maps an input message to either `"billing"` or `"support"`. Because `SystemOneCompiler` splits these eight examples between fitting and held-out calibration, the tiny calibration fold still marks the decision with `Needs review: True`.

## Step 3: Edit the lessons and re-run

Next, copy `first_skill.json` into `.system1/my-lessons.json` so we can add our own messages:

```bash
cp examples/teaching/first_skill.json .system1/my-lessons.json
```

Open `.system1/my-lessons.json` in your editor and add two more `[message, answer]` pairs for `"billing"` and `"support"`. Then run `examples/teach_skill.py` with `--lessons` and `--output`:

```bash
python examples/teach_skill.py --lessons .system1/my-lessons.json --prompt "Please send my payment receipt" --output .system1/candidate.s1m
```

You should see this output:

```text
Suggested team: billing
Needs review: True
```

The script compiles `.system1/my-lessons.json`, writes `.system1/candidate.s1m` without overwriting `.system1/support-route.s1m`, and prints the predicted team and review flag. To define your own output options, see the [teaching guide](training_experts.md#teach-your-data).

## Step 4: Check a larger four-team example

To see how a skill behaves when we separate teaching, calibration, and evaluation messages, run `examples/support_triage.py`:

```bash
python examples/support_triage.py
```

The script compiles a four-department skill (`billing`, `technical_support`, `account_security`, and `sales`), evaluates `104` held-out messages outside the teaching and calibration splits, and prints a summary header followed by per-message verdicts:

```text
support_triage: 192 teaching + 112 calibration + 104 unseen cases
Teach + calibrate: 115.5 ms | Saved skill: 62.8 KiB
Save: 1.8 ms | Load: 1.0 ms
Unseen accuracy: starter 76.9% -> taught 94.2%
Answered without review: 92/104 | Accuracy on those: 98.9%
```

Here, raw accuracy rises from `76.9%` to `94.2%`. On the `92` messages that the skill accepts without review (`ANSWER`), accuracy reaches `98.9%`, while the remaining `12` uncertain messages print `REVIEW`.

## Step 5: Load and reuse the saved `.s1m` skill

Now, load `.system1/support-route.s1m` in Python and run a decision without reading the JSON lessons file:

```python
from system1 import CompiledSystemOneModel, SystemOneEngine

# Load directly into SystemOneEngine (preserves strict_mode from save time).
engine = SystemOneEngine.load(".system1/support-route.s1m")

# Or load the CompiledSystemOneModel artifact explicitly:
skill = CompiledSystemOneModel.load(".system1/support-route.s1m")
engine = SystemOneEngine(skill.schema, model=skill, strict_mode=True)

answer = engine.decide("Please refund this payment", record_receipt=False)
print("Suggested team:", answer.values["team"])
print("Needs review:", answer.is_ambiguous)
```

This prints:

```text
Suggested team: billing
Needs review: True
```

## Step 6: Correct a mistake

Suppose a customer sends `"The payment page crashes when I open it."` and the skill predicts `billing`. Under our rule, broken software belongs to `support` even when the message mentions payment.

To fix the decision online in Python, call `engine.learn(...)`:

```python
engine.learn("The payment page crashes when I open it.", {"team": "support"})
fixed = engine.decide("The payment page crashes when I open it.", record_receipt=False)
print("Updated team:", fixed.values["team"])
```

You should see:

```text
Updated team: support
```

To persist the correction in `.system1/my-lessons.json`, add the pair to the `"team"` array, or update its label if that sentence already exists in the file:

```json
["The payment page crashes when I open it.", "support"]
```

Re-run `examples/teach_skill.py` with `--prompt "The payment page crashes when I open it."`:

```bash
python examples/teach_skill.py --lessons .system1/my-lessons.json --prompt "The payment page crashes when I open it." --output .system1/candidate.s1m
```

You should see:

```text
Suggested team: support
Needs review: True
```

When you are ready to manage separate teaching, calibration, and evaluation splits with regression gates, continue to [Correct, compare, and adopt a skill](correcting_skills.md). To collect lessons automatically from an upstream LLM, see the [teacher observation reference](../typesafe.md) and [Wrap an LLM call with `BoundedReviewFallback`](../drop_in/05_llm_observe_and_cutover.md).
