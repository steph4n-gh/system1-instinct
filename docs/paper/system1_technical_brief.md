# System 1: Instinct — technical brief

**System 1: Instinct 1.0.0 · 14 October 2026 · Maintainer-authored implementation brief**

This brief describes **System 1: Instinct 1.0.0** (see the [6-category benchmark gauntlet report](../../benchmarks/quality/instinct_gauntlet_report.md) and [drop-in & adaptation report](../../benchmarks/quality/drop_in_and_adaptation_report.md)). Each linked experiment
retains its original source revision and evidence; the benchmark gauntlet adds six
offline evaluation categories and neural baseline comparisons alongside historical quality results.

System 1 turns a repeatable decision into a small reusable local skill. Define the
choices, show examples, check a candidate, adopt it, then correct and compare when
it makes mistakes. A person, JSON data, a rule, Jev, Gemini or another callback can
supply labels. **System 1 is not an LLM:** its built-in path fits numerical decision
heads without language-model weights, token generation or a GPU. It does not
inherit a teacher's general knowledge.

## One teaching and correction workflow

`TeachingSession` keeps single-field
or multi-field schemas (`ChoiceField`, `BooleanField`, `MultiChoiceField`,
`ScoreField`), explicit lessons, separate calibration examples and evaluation checks
together (including `split="auto"` stratified routing). At least one field
must keep ambiguity escalation enabled, so uncertainty requests review. A
correction replaces a same-input lesson. `assess()` builds, saves and reloads a
candidate and compares it with the approved skill on the same checks. It reports
raw accuracy, accepted errors, review coverage, per-class counts, descriptive
bounds, changed answers and regressions. Reused checks remain development evidence.
`adopt()` activates only a passing candidate whose lessons, artifacts and current
skill have not changed. Editing or failing an assessment leaves the approved
skill available.

`assess(require_qualification=True)` also requires `qualify()` to pass
on one separately supplied fresh cohort. Provenance, targets and labels are
frozen before scoring. Consumed inputs and supplied groups cannot become lessons
or another qualification cohort, and one candidate gets one request even when
scoring fails or is interrupted. The caller must supply independent,
representative examples; the software cannot verify that assumption.
Accepted-correctness and coverage lower bounds use a fixed session confidence
budget: each bound spends `(1 - confidence) / (2 * k * (k + 1))` at attempt `k`.
This is distinct from conformal prediction-set coverage and development checks.

Adoption archives exact approved bytes and their evidence. `rollback()` validates
and restores a retained revision, preserves lessons and requires a new assessment
before further adoption. `decision_details()` reports the active revision and
observable review conditions, without automatically calling a teacher or teaching
from predictions.

This is an additive wrapper around the existing compiler and strict engine. It
uses the existing TF-IDF or `HybridProjector` representation and a ridge or optional logistic head, with
the solver fixed when the session is created, no new dependency or
saved-format change. The [walkthrough](../guides/correcting_skills.md) is the
reference for Python, CLI, defaults and evidence limits. The
[document workspace](../../examples/teaching_by_doing/README.md) exposes the same
workflow through filing actions. Its authored sample checks demonstrate the
process; they do not qualify a customer's document workload.
The release introduces no pretrained language model and does not
weaken the existing full-scope quality targets.

## How a saved skill works

Text features and optional numerical telemetry feed small NumPy decision heads.
The default compiler uses fixed hashed features; the optional TF-IDF projector
fits a vocabulary on teaching text and freezes it for inference. Separate
examples calibrate uncertainty. A `.s1m` file preserves the schema, feature
configuration, weights and review behavior. New inputs pass through those shared
features, rather than requiring an exact cached match.

Optional MLX supports Apple Silicon decision-head operations; its benefit depends
on workload. Optional research adapters use fixed pretrained text encoders whose
separate weights and execution costs belong in their measurements. The
compiler also offers opt-in logistic fitting with SciPy during teaching and the
existing NumPy head during inference. These choices do not change the default
ridge compiler or the limits of its narrow output schema.

Automatic observation is a separate existing path: the teacher keeps answering
until promotion checks pass. Later local decisions can still request review.
Neither manual adoption nor automatic promotion grants permission to perform an
action; the application must honor review and its own execution policy.

## What the evidence supports

Three fixed public-data tasks accepted 96.1–98.3% of 1,301 originally held-out
cases, with 97.5–99.4% correctness among accepted decisions. Teaching and
calibration took 0.23–1.77 seconds once labels existed; saved skills were
20.4–47.0 KiB and median local decisions took 0.34–0.44 ms. These NumPy CPU
measurements exclude data preparation, startup, audit work and downstream actions.
The matched conventional TF-IDF/logistic baseline ran faster locally on all three.

Actual Jev and Gemini each taught the six-intent assistant takeover after 358
observations: 171/180 test requests accepted, 167/171 correct, with no further
teacher calls and identical reloaded results. Other teacher/task combinations
deferred or missed the independent quality target. These bounded measurements
do not establish general provider parity or superiority over conventional
classifiers. See the [original workload report](../../benchmarks/quality/workloads/README.md).

The [BBC workflow test](../../benchmarks/quality/document_workflow/README.md)
measured one correction round on five news topics. On 400 previously unscored
articles, targeted feedback reduced accepted errors 16→8 while acceptance fell
373→337. Raw correctness rose 369→373, but the paired interval includes no gain.
All candidates failed the unchanged zero-error adoption checks, so none became
a serving skill. This is evidence of a review/quality tradeoff and candidate
rejection; it does not establish successful real-data adoption or superiority
over its ordinary-addition control.

Classical linear baselines preserve important limitations: under the unigram TF-IDF baseline, fresh authored SMS probes reached
only 27/33 correct accepted answers; public email has a source-shift failure and
a retrospective improvement; Inbox Zero remains review-only; the full-scope n8n
skills fail accepted-accuracy targets and CLINC unfamiliar-input rejection.
In the [6-category benchmark gauntlet](../../benchmarks/quality/instinct_gauntlet_report.md),
supervised `chi2` + character n-gram TF-IDF, `HybridProjector.fit` spectral
`PPMI+SVD`, two-stage `hierarchical` routing, Mondrian class-conditional
calibration, and dual OOD gates (`familiarity_alpha` + `prototype_alpha`) raise
`BANKING77` 32-intent accuracy from `77.3% -> 87.9%` (`91.4%` post-flywheel) and
77-intent accuracy from `62.0% -> 79.5%` (`88.1%` post-flywheel), reduce
`CLINC150-OOS` false acceptance to `0.0%` (`0/120` accepted OOS), eliminate
`NotInject` benign-trigger false positives (`10.0% -> 0.0%` FPR, `70.0% -> 100.0%`
polarity contrast accuracy), and lift `quality_round` fresh `banking_support`
accepted accuracy to `100.0%` (`0` errors) and `model_routing` coverage to `88.9%`
(`100.0%` accepted accuracy).
The [workload and evidence index](../../benchmarks/quality/README.md) links every
protocol, baseline, result and reproduction path. Reused checks are regression
evidence. New candidates still need representative independent confirmation.

## Product value and boundaries

The practical value is the complete path from explicit examples to a checked,
portable local skill that can be corrected without silently replacing a working
revision. Feature hashing, ridge regression, conformal prediction and signatures
are established techniques. This is an integration and usability claim, not a
new learning algorithm. Poor raw predictions, excessive review and unsuccessful
completed tasks remain distinct problems; better workflow does not by itself
solve them.

Optional `PolicyEngine`/`SystemOneGuard` permissions, software Ed25519 receipts
and a SHA-256-chained ledger support enforcement and auditing. Applications must
authenticate callers, protect keys and enforce tool boundaries. See
[deployment boundaries](../deployment.md).

Start with [your first skill](../guides/first_skill.md), then
[correct, compare, and adopt](../guides/correcting_skills.md). The
[architecture reference](../architecture/technical_specification.md),
[teacher adapter](../typesafe.md), [whitepaper](system1_whitepaper.md) and
[example catalog](../../examples/README.md) cover implementation, evidence and
bounded research labs. This brief and the whitepaper are maintainer-authored,
not peer-reviewed publications.
