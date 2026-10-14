# Teach your first skill

Teaching System 1 means **showing it examples of a decision you already know how
to make**. You do not need an LLM, an API key or a GPU. We will teach one small
job: choose the team that should receive a customer message.

Our rule is simple: payment questions go to `billing`; broken software goes to
`support`. You choose that rule. System 1 learns to apply it to other messages.

If you prefer a visual first lesson, try the
[document teaching workspace](../../examples/teaching_by_doing/README.md).
You choose folders for fictional documents, and the page records those actions
as examples. It lets you compare a candidate with the approved skill, explicitly
adopt a passing revision, correct a choice and download the skill.
The walkthrough below introduces teaching using an editable JSON file.

## What is a label?

A label is just **the answer you want**. These are three teaching examples:

| Message System 1 will see | Answer you want |
|---|---|
| Refund my payment | `billing` |
| Correct the invoice | `billing` |
| Fix a software crash | `support` |

You are showing complete messages and their correct destination. You do not need
to highlight keywords, write a prompt for an LLM or describe an algorithm.
The answer names must match the two choices exactly: `billing` or `support`.

## Fast path in Python: 0 examples $\to$ 6 online examples $\to$ `.s1m`

Before creating any files, you can teach an engine interactively in Python. With
**0 examples**, `strict_mode=True` safely abstains (`is_ambiguous=True`). After
**6 online examples** via `engine.learn_from_outcome(...)` (or `engine.learn(...)`),
System 1 routes unseen phrasings locally in `<1 ms`, still abstains on unrelated
topics, and saves to a portable `.s1m` file:

```python
from system1 import ChoiceField, DecisionSchema, SystemOneEngine


class SupportRoute(DecisionSchema):
    team = ChoiceField(options=["billing", "support"])


# 1. Start at 0 examples — strict mode safely abstains instead of guessing
engine = SystemOneEngine(SupportRoute, strict_mode=True)
before = engine.decide("Duplicate charge on my credit card invoice", record_receipt=False)
assert before.is_ambiguous is True

# 2. Teach 6 examples online (<1 ms per Rank-1 Sherman-Morrison update)
for message, team in [
    ("Refund my duplicate credit card charge and fix the invoice", "billing"),
    ("Why was my subscription billed twice on this receipt?", "billing"),
    ("Please update our billing address and send the tax invoice", "billing"),
    ("Application crashes with a stack trace when calling the API", "support"),
    ("Debug this software exception and fix the failing endpoint", "support"),
    ("Production server error logs show a timeout crash on startup", "support"),
]:
    engine.learn_from_outcome(message, {"team": team}, outcome="resolved", reward=1.0)

# 3. Unseen phrasings now resolve locally with high confidence
after = engine.decide("My credit card was billed twice for the subscription invoice", record_receipt=False)
assert after.values["team"] == "billing" and after.is_ambiguous is False

# Out-of-scope inputs still abstain safely
oos = engine.decide("What is the best recipe for sourdough bread?", record_receipt=False)
assert oos.is_ambiguous is True

# 4. Save & reload the portable .s1m skill
engine.save(".system1/support-route.s1m")
reloaded = SystemOneEngine.load(".system1/support-route.s1m")
```

## Run the first lesson

From a repository checkout with System 1 installed, run:

```bash
python examples/teach_skill.py
```

If you have not installed the project yet, use the
[README setup commands](../../README.md#quickstart), then return here.

The script reads [first_skill.json](../../examples/teaching/first_skill.json),
teaches from its eight examples, saves a skill and tries a different message.
The file uses this simple layout:

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

`team` names the decision. Each pair contains a message followed by the answer
you want. That is the whole teaching format for this example.

The first run prints:

```text
Suggested team: billing
Needs review: True
```

`billing` is its best guess. **Needs review means a person should still decide.**
Eight examples split between fitting and held-out calibration show the batch
compile process, but leave a tiny calibration fold so review is requested.
Saving a skill does not certify that it is ready.

The saved `.s1m` file contains the reusable skill. System 1 applies learned
patterns to new messages; it does not need the JSON file at decision time.

## Teach your own messages

Copy the lesson file so you can edit it. The first run creates `.system1/`, which
is ignored by Git:

```bash
cp examples/teaching/first_skill.json .system1/my-lessons.json
```

Open `.system1/my-lessons.json` in a text editor. Replace or add messages that
people actually send to your team, keeping the same two possible answers. Then:

```bash
python examples/teach_skill.py --lessons .system1/my-lessons.json --prompt "Please send my payment receipt"
```

Each run rereads your lessons and saves the newly taught skill. Editing the JSON
alone does not change an already saved skill. Use `--output .system1/candidate.s1m`
to save a candidate separately while keeping the previous skill.

For a different job, define different output choices as described in the
[teaching API guide](training_experts.md#teach-your-data).

## How do I know it is ready?

Keep some real messages out of the lessons. Decide their correct teams yourself,
then compare System 1's answers. Track three separate things:

- **How often the best guess is right, including guesses sent for review.** This is raw accuracy.
- **How often it is wrong without requesting review.** These mistakes could reach your workflow.
- **How often it needs review.** A skill that asks about every message is not yet doing the job automatically.

The complete workflow uses examples in three ways:

| In plain language | Name used in the code | Purpose |
|---|---|---|
| Show it how to decide | Teaching | Learn the decision |
| Help it recognize uncertainty | Calibration | Decide when to ask for review |
| Give it a final check | Evaluation | Measure behavior on messages kept out of the first two groups |

The starter script handles a small calibration split automatically. Its tiny
file leaves too little evidence, so review is expected. For a fuller working
example with separate groups and a report, run:

```bash
python examples/support_triage.py
```

That example includes four department choices and shows both correct decisions
and mistakes. Its larger file is a demonstration, not a substitute for your own
messages. There is no magic number of examples: start with the task's common
cases and confusing distinctions, then use the separate checks to decide what
is missing. Never teach from your final check and still call that same check new.

If raw guesses are wrong, inspect the rule, lessons and whether the text features
capture the distinction. More calibration alone does not repair wrong guesses.
If guesses are mostly right but often need review, inspect representative
calibration examples and the cases being withheld. Reteaching changes the skill,
so check it again.

## Reuse the saved skill

After saving a `.s1m` file (via `engine.save(...)` or `skill.save(...)`), you can
load it directly into a ready-to-use `SystemOneEngine` without reading the
lesson file or calling a teacher:

```python
from system1 import CompiledSystemOneModel, SystemOneEngine

# Direct one-line load (preserves strict_mode from when the engine was saved)
engine = SystemOneEngine.load(".system1/support-route.s1m")

# Or load the CompiledSystemOneModel artifact explicitly:
skill = CompiledSystemOneModel.load(".system1/support-route.s1m")
engine = SystemOneEngine(skill.schema, model=skill, strict_mode=True)

answer = engine.decide("Please refund this payment", record_receipt=False)
print(answer.values["team"])
print("Needs review:", answer.is_ambiguous)
```

Saving and reopening a skill preserves its calibration state; it does not
blindly approve every answer. Your application chooses what to do when review is
requested (`answer.is_ambiguous is True`). Keep the earlier file while checking
a revised candidate, then choose which saved skill your application loads.

## What does correcting a mistake mean?

Suppose a new message says **“The payment page crashes when I open it.”** and
System 1 suggests `billing`. Under our rule the right answer is `support`: the
problem is broken software, even though it mentions payment.

You can correct it immediately in Python with an online Rank-1 Sherman-Morrison
update (`<1 ms`, preserving calibration by default):

```python
engine.learn("The payment page crashes when I open it.", {"team": "support"})
```

Or add the pair to your JSON lessons file, remembering the comma between JSON entries:

```json
["The payment page crashes when I open it.", "support"]
```

If that exact message is already present with the wrong answer, **replace the
wrong answer**. Do not leave two conflicting lessons. Add a few different real
examples of the same distinction, rather than copying the same sentence many
times. Keep ordinary billing and ordinary support messages too.

Rerun the teaching command. Then try **different** messages, such as “The checkout
screen freezes” and “Why was I billed twice?” Showing that the corrected sentence
now works is useful, but it does not show that the skill handles new cases.
A correction can also affect other decisions, so check both teams again.

For an ongoing skill, use the
[correct, compare, and adopt workflow](correcting_skills.md), available in System 1 1.0.0+. It retains separate
teaching, calibration and evaluation examples, builds a candidate, and compares
old and new answers before you explicitly activate it. It reports lost useful
answers as regressions even when the revised skill merely sends them to review.
An unsuccessful candidate leaves the approved skill available.

## Can someone else supply the lessons?

Yes. A colleague, reviewed historical decisions, a rule in your existing software,
a conventional classifier, an OpenAI, Anthropic, or Gemini model, or another API can provide the same
message-and-answer pairs. JSON is simply a way to store them. The
[observation workflow](../typesafe.md) and [`BoundedReviewFallback`](../drop_in/05_llm_observe_and_cutover.md)
collect answers from a running teacher and keep that teacher responsible until
System 1 qualifies for local takeover.

System 1 remains a small local decision skill regardless of who taught it.
