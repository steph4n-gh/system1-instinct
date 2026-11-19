# Mathematical notes: uncertainty, prototype geometry and online correction

**System 1: Instinct 1.3.0 · 18 November 2026 · Implementation notes, not new theorems**

This note describes the runtime mathematics in
**System 1: Instinct 1.3.0** (see the [6-category benchmark gauntlet report](../../benchmarks/quality/instinct_gauntlet_report.md) and [drop-in & adaptation report](../../benchmarks/quality/drop_in_and_adaptation_report.md)) and the assumptions needed to interpret it,
including Mondrian class-conditional calibration, dual OOD familiarity and prototype similarity gating, the teaching workflow, and calibration-preserving online RLS updates. Section 7 covers the calibration methods added in 1.3.0. The [whitepaper](system1_whitepaper.md) contains the
product evaluation; the [architecture reference](../architecture/technical_specification.md)
maps behavior to code. Earlier claims of guaranteed safe execution, universal
margin improvement and fixed microsecond update times are withdrawn.

## 1. Split-conformal prediction sets

Fix a categorical scoring model before evaluating the conformal calibration fold.
Let its probabilities be $p_y(x)$ and let $n$ calibration pairs and the next test
pair be exchangeable. Examples-only skills use the LAC nonconformity score

$$s(x,y)=1-p_y(x).$$

Sort calibration scores as $s_{(1)}\le\cdots\le s_{(n)}$ and set

$$k=\lceil(n+1)(1-\alpha)\rceil,\qquad
q=\begin{cases}s_{(k)}&k\le n,\\1&k>n.\end{cases}$$

Because LAC scores lie in $[0,1]$, the second case includes every label. With no
calibration evidence, strict mode also retains all labels. For a calibrated skill,

$$C(x)=\{y:s(x,y)\le q\}.$$

Under the stated assumptions,

$$\Pr\{Y_{n+1}\in C(X_{n+1})\}\ge1-\alpha.$$

For distinct scores, the test score's pooled rank is uniform on $1,\ldots,n+1$.
When $k\le n$, inclusion has probability $k/(n+1)\ge1-\alpha$; when $k>n$ it is
certain. Ties handled by including scores equal to the threshold are conservative.
This is the established split-conformal argument, not a new System 1 result.
See [Angelopoulos and Bates, §2](https://arxiv.org/html/2107.07511v6#S2).

### What the guarantee does not say

Let $A$ be the event that the application accepts a local prediction. Marginal
set coverage does not imply

$$\Pr\{\hat Y=Y\mid A\}\ge1-\alpha.$$

For example, set misses can be concentrated in a small accepted subset while
other inputs receive broad sets. The coverage statement also does not establish
accuracy for each class, simultaneous coverage across fields, adversarial
robustness or safe tool execution. Exchangeability can fail under traffic drift,
related templates or adaptive collection. An empty set requests review; it does
not by itself detect distribution shift.

## 2. Score definitions and release behavior

[ConformalPredictor](../../src/system1/calibration.py) supports LAC and a legacy,
deterministic APS score. APS sorts labels by descending probability and sums mass
through the candidate label. Calibration and prediction use the same score and
ordering. Strict inversion includes labels whose score is at most the stored
quantile; it does not add the next label after crossing that threshold.

New examples-only categorical/Boolean skills use LAC. Legacy and schema-augmented
skills use APS. Format-v2 and opt-in v3 artifacts record this choice, while v1 files retain
legacy APS semantics. Both score functions lie in $[0,1]$ for valid normalized
probabilities; insufficient evidence conservatively includes all labels.

The current source computes p-value tail counts using left insertion into sorted
calibration scores. Equal scores remain included, preserving the prior inclusive
tail-count definition for both LAC and APS.

The compiler separates temperature fitting from conformal scoring so that the
probability transformation is not fitted on the same labels used to claim set
coverage. Explicit calibration must also be separate from head fitting and final
evaluation. Repeated normalized prompts do not create new independent evidence.

Strict categorical decisions require a singleton set for acceptance. Non-strict
mode can suppress review using margin, a cardinality-dependent confidence floor
and a top-to-runner-up ratio; these are heuristics and do not inherit the strict
coverage interpretation. They do not apply as a universal safety test to all
field types. MultiChoice uses a complete assignment score and continuous scores
use residual intervals. Field-level `escalate_on_ambiguity` determines which
uncertain fields affect the overall review flag.

The published public-data experiments use strict mode and report **acceptance**,
**correctness among accepted decisions** and **raw correctness** separately.
The optional `PromotionPolicy(min_accepted_agreement=.95)` added in 1.0.2 also
tests agreement conditional on an accepted validation group. Its point estimate
and exact lower bound must pass, with no accepted evidence causing deferral.
This adds a third exact bound to the error budget: at attempt `k`, each spends
`(1 - confidence) / (3 * k * (k + 1))`. The default two-bound budget is unchanged.
Teacher agreement is distinct from correctness against independent labels; the
[quality round](../../benchmarks/quality/quality_round/README.md) reports both
qualification and subsequent diagnostic failures.

Default automatic promotion tests 80% agreement and 80% acceptance; that is a
separate policy from alpha 0.05 or a 95% accepted-correctness workload target.
The [SMS observation result](../../benchmarks/quality/workloads/README.md)
promotes under the former and misses the latter.

### Independent qualification in the 1.2.0 teaching session

`TeachingSession.assess()` compares a fixed candidate with the approved skill on
recurring development checks. Its empirical counts and descriptive bounds cannot
be treated as fresh qualification after the checks have guided corrections.
`assess(require_qualification=True)` additionally requires a separately supplied
cohort scored by `qualify()`, after its candidate, targets and source are frozen.

For qualification attempt $k\ge1$, let $N_k$ be the number of independently sampled
groups, $A_k$ the number accepted, and $S_k$ the accepted-correct count. There is
one supplied row per group. The two quantities being assessed are acceptance
probability and correctness conditional on acceptance for this fixed candidate
on the represented population. Using the existing exact one-sided binomial lower
bound $L(s,n;\delta)$, qualification requires

$$L(S_k,A_k;\delta_k)\ge t_{\rm correctness},\qquad
L(A_k,N_k;\delta_k)\ge t_{\rm coverage}.$$

With no accepted observations, accepted correctness is undefined and qualification
fails. For $0<s\le n$, the bound inverts the binomial upper tail; when $s=n$,
$L(n,n;\delta)=\delta^{1/n}$. A zero success count gives a zero lower bound.
The defaults are $t_{\rm correctness}=0.95$ and $t_{\rm coverage}=0.8$.

The session fixes confidence $c$ on its first consumed qualification request.
Every later request must use the same $c$. Each bound at attempt $k$ receives

$$\delta_k=\frac{1-c}{2k(k+1)},\qquad
\sum_{k=1}^{\infty}2\delta_k=1-c.$$

Under independent representative sampling, with each candidate fixed before its
fresh cohort is scored, the union bound therefore controls bound failures across
the session's qualification attempts at $1-c$. Candidates may have been developed
from earlier evidence; a later cohort must remain independent conditional on that
history. This does not establish future accuracy after population drift, per-class
accuracy, or conformity of caller-supplied data to the assumptions.

The complete request is persisted before scoring. One candidate can consume one
request, including a failed or interrupted evaluation. Inputs and supplied groups
are excluded from later qualification and teaching within the session. These
checks catch supplied lineage and normalized overlap; they cannot prove that
unidentified paraphrases or source families are independent. The confidence scope
is the retained session, rather than arbitrary experiments in other directories.
This gate is separate from both conformal calibration and the automatic teacher
promotion policy above.

## 3. Prototype centering: a conditional geometry result

The schema-seeded model can center class prototypes and normalize the residuals.
This is centering, not covariance whitening. It is separate from fitting ridge
heads to representative labeled examples. The following identity describes a
special geometry; it is not a guarantee about arbitrary text features.

Let $w_j=b+v_j$ for $K\ge2$, with mutually orthogonal $v_j$ of equal nonzero norm
$c$ in dimension $D\ge K$. Subtract $\mu=K^{-1}\sum_jw_j$ and write
$w'_j=v_j-K^{-1}\sum_mv_m$. Then

$$\langle w'_j,w'_k\rangle=c^2(\delta_{jk}-1/K),\qquad
\|w'_j\|^2=c^2(K-1)/K.$$

Normalizing therefore gives

$$\langle\tilde w_j,\tilde w_k\rangle=-\frac1{K-1}\quad(j\ne k).$$

This reaches the simplex bound for the maximum signed pairwise inner product:
for any $K$ unit vectors,

$$0\le\left\|\sum_ju_j\right\|^2
=K+\sum_{j\ne k}\langle u_j,u_k\rangle$$

implies $\max_{j\ne k}\langle u_j,u_k\rangle\ge-1/(K-1)$.
The bound is attained by a regular simplex when dimension permits. It is not the
Welch bound on absolute correlation. For two distinct prototypes, centering makes
them antipodal; coincident prototypes have zero residual and cannot be normalized
by this formula. Real correlated prototypes need not become a regular simplex,
and greater separation at prototypes does not prove better test accuracy.

## 4. Ridge fitting and optional online correction

For augmented features $Z=[X,\mathbf1]$, targets $Y$, and the compiler's diagonal
regularizer $R$, the fitted coefficients satisfy

$$\Theta=(Z^TZ+R)^{-1}Z^TY.$$

The implementation solves these equations with NumPy (`np.linalg.solve`, through
the equivalent $N\times N$ dual system when there are fewer samples than features)
and forms the inverse only when it is retained for online correction; it has a
pseudoinverse/least-squares fallback. Ridge fitting does not use the Cholesky
solver described by older drafts. Bias regularization and sample
weights must be included when comparing against a batch formula.

For ideal arithmetic, an invertible matrix $A$ with inverse $P$, and a new feature
vector $z$, Sherman–Morrison gives

$$P'=(A+zz^T)^{-1}=P-\frac{Pzz^TP}{1+z^TPz},$$

provided the denominator is nonzero. Updating $B'=B+zy^T$ and setting
$\Theta'=P'B'$ yields the unweighted batch solution for the augmented data.
This follows directly by multiplying $A+zz^T$ by the proposed inverse.

The optional runtime update also supports forgetting $f\in(0,1]$ (default 0.995):

$$P'=\frac1f\left(P-\frac{Pzz^TP}{f+z^TPz}\right),\qquad
B'=fB+zy^T.$$

This describes exponentially weighted evidence, not the original unweighted
batch objective when $f<1$. Finite precision and runtime stabilization further
limit exact equivalence. A positive diagonal alone does not prove positive
definiteness; rescaling a matrix alone does not improve its condition number.
No fixed latency, unlimited stability or immunity to forgetting is implied.

The `TeachingSession` workflow retains corrected examples across single-field and
multi-field schemas and
recompiles a separate TF-IDF or `HybridProjector` candidate, then recalibrates it on its separate
calibration records. It does not mutate
the approved skill during assessment. When `learn_from_tier2` is called with the
default `preserve_calibration=False`,
changed weights invalidate their old calibration in memory and saved artifacts, so strict inference requests review until separate recalibration.
When `learn_from_tier2(..., preserve_calibration=True)` is used (such as inside
`BoundedReviewFallback(..., auto_online_update=True)`), System 1 preserves
existing conformal calibration scores and uses exact Sherman-Morrison directional
gain scaling $y_{\text{step}} = ((\lambda_f + g) y_{\text{desired}} - \lambda_f W_t x) / g$
(with $g = x^T P_t x$) so that only the target class and competing positive runner-up
logits are adjusted while non-competing classes remain unperturbed. Even so, a
teacher label is fallible supervision, and online updates do not re-establish
formal split-conformal exchangeability; periodic batch recompilation on retained
lessons remains recommended.

### Mondrian class-conditional calibration and dual OOD gating

With `ConformalPredictor(mondrian=True)`, System 1 partitions the calibration
fold by true class $c\in\{1,\ldots,K\}$ and computes per-class quantiles
$q_c = s_{c,(\lceil (n_c + 1)(1 - \alpha)\rceil)}$ (falling back to the marginal
quantile when $n_c < 3$). Candidate label $y$ is included in $C(x)$ iff
$s(x, y) \le q_y$, preventing majority classes from starving minority classes on
imbalanced workloads (`sms_triage`). Additionally, two orthogonal OOD gates can
escalate unfamiliar inputs before conformal singleton acceptance:

1. **Feature familiarity gate** (`familiarity_alpha > 0`): Escalates with
   `abstention_reason="low_feature_familiarity"` when known vocabulary mass
   falls below the empirical $\alpha_{\text{fam}}$ calibration quantile.
2. **Prototype cosine similarity gate** (`prototype_alpha > 0`): Computes
   unit-normalized class centroids $\mu_c$ during compilation and escalates with
   `abstention_reason="low_prototype_similarity"` when
   $\max_c \langle \hat{x}, \mu_c \rangle$ falls below the empirical
   $\alpha_{\text{proto}}$ calibration quantile.

The compiler's optional logistic choice fit instead minimizes summed
cross-entropy plus $\lambda\|W\|_F^2/2$, with an unpenalized bias, using SciPy
L-BFGS-B at teaching time. Stored coefficients are scaled by 0.25 to match the
runtime's choice-logit scaling. NumPy inference and separate calibration are
unchanged. The ridge batch-equivalence formulas above do not describe incremental
logistic fitting: retain corrections and recompile to preserve that objective.
Since 1.2.0, the teaching session fixes its ridge or optional logistic solver at
creation. Adoption archives exact approved bytes and evidence; restoring a
validated approved revision keeps lessons and invalidates the pending assessment.
Restoration does not recompute an online update or make old qualification data fresh.

## 5. Reading the evidence

The [three-workload report](../../benchmarks/quality/workloads/README.md) and
[6-category benchmark gauntlet](../../benchmarks/quality/instinct_gauntlet_report.md) record
measured quality, errors, timings, baseline behavior and failed takeovers.
They do not experimentally prove exchangeability or the conditions of the
prototype identity. Deterministic tool permissions and signed audit records are
separate software mechanisms; see [deployment boundaries](../deployment.md).

The new session's adoption checks are an explicit empirical policy over recurring
development cases: raw accuracy, acceptance coverage, accepted-error count and
loss of previously accepted correct answers. Their default zero-error requirement
is neither a conformal theorem nor a future-error guarantee. Reusing checks to
choose corrections does not create independent validation evidence. The
[BBC workflow experiment](../../benchmarks/quality/document_workflow/README.md)
keeps a separate final holdout: a candidate meets its 95% accepted-correctness /
80% coverage targets yet is correctly refused by the stricter development
adoption policy. Its narrower accepted set also yields fewer correct automatic
answers. Neither a Wilson interval nor a small accepted-error count proves a
better decision policy without considering coverage and application costs.
The separate 1.2.0 qualification gate adds fresh-data lower-bound checks under
the assumptions above; it does not retroactively qualify this experiment or any
full-scope workload. No new encoder or lesson-selection result is asserted here.

## 6. Selective risk, factor-graph joint LTT, and sequential monitoring (1.2.0)

Version 1.2.0 adds finite-sample selective risk gates, imbalance-aware
calibrators, factor-graph joint decoding, and anytime-valid online risk
monitors in [calibration.py](../../src/system1/calibration.py):

- **`SelectiveRiskCalibrator`:** Selects a threshold $\lambda$ on scalar
  confidence or margin scores using Learn Then Test (`"ltt_fst"` fixed-sequence
  testing or `"ltt_bonferroni"`) for conditional selective error
  $\Pr(\hat{Y} \ne Y \mid g(X) \ge \lambda) \le \alpha$, or Risk-Controlling
  Prediction Sets (`"rcps_joint"`) for joint selective risk
  $\Pr(\hat{Y} \ne Y \text{ and } g(X) \ge \lambda) \le \alpha$, each backed
  by exact binomial tail p-values and `clopper_pearson_upper` bounds at failure
  probability $\delta$. When no threshold qualifies, `should_accept(...)`
  fails closed (`False`).
- **`DecisionCalibrator.fit_imbalance_aware`:** Applies class-prior logit
  adjustment (`-tau * log(pi)`), balanced temperature scaling, and a fail-closed
  LTT margin gate using pure NumPy without requiring SciPy at runtime.
- **`JointSchemaCalibrator` (`engine.calibrate_joint`):** Fits pairwise label
  log-prior potentials over structured schema combinations (`fit_structure`),
  runs exact MAP decoding (`decode_batch` / `decode_one`), and certifies
  complete-assignment acceptance with `SelectiveRiskCalibrator(method="ltt_fst")`.
- **`AdaptiveConformalTracker` & `SequentialRiskMonitor`:** Track live
  residual error after deployment. `AdaptiveConformalTracker` updates online
  $\alpha_t$ via Gibbs-Candes Adaptive Conformal Inference with a bounded
  sliding window and error-triggered `margin_floor` ratcheting.
  `SequentialRiskMonitor` maintains a nonnegative supermartingale (e-process)
  across audited error indicators in `"capital_ons"`, `"ons_betting"`,
  `"capital_eb"`, or `"fixed_mixture"` mode, raising an anytime-valid drift
  alarm whenever the e-value reaches $1/\delta$.

## 7. Additions in 1.3.0

The methods below are in [calibration.py](../../src/system1/calibration.py)
unless stated otherwise. Apart from soft distillation, which is a compiler
method, they are standalone: `SystemOneEngine.decide` does not call them.
`UtilityCascadeRouter` combines the cost-aware, ordinal and binary calibrators.

### Online Mondrian threshold tracking

`OnlineMondrianConformal` keeps one LAC threshold per class. Its prediction set is
$\{k: 1-p_k\le q_k\}$, and `should_accept` requires a singleton. Each class $c$
stores $u_c\in[0.01,16]$ with $q_c=1-e^{-u_c}$ ($q_c=1$ at the upper bound), a
working miss level $\alpha_c\in[\alpha_{\min},\alpha_{\max}]$, where
$\alpha_{\min}=\max(0.015,0.35\alpha)$ and $\alpha_{\max}=\min(0.40,2.5\alpha)$,
and an error average $\bar e_c$. After observing true class $y$ with score
$s=1-p_y$ and miss indicator $e=\mathbf 1\{s>q_y\}$ (or a caller-supplied
indicator), only class $y$ changes:

$$\bar e_y\leftarrow0.85\,\bar e_y+0.15\,e,\qquad
b=\begin{cases}1.6&\bar e_y>1.6\alpha,\\1&\text{otherwise,}\end{cases}$$

$$\alpha_y\leftarrow\operatorname{clip}\bigl(\alpha_y+\gamma b(\alpha-e),\,
\alpha_{\min},\alpha_{\max}\bigr),\qquad
u_y\leftarrow\operatorname{clip}\bigl(u_y+\eta b(e-\alpha_y),\,0.01,16\bigr).$$

After a miss while $b=1.6$, $u_y$ is also raised to at least
$(1-w)u_y+w\min(-\log(1-s),15.5)$, computed from the value before the step.
The defaults are $\alpha=0.10$, $\gamma=\max(0.03,0.4\alpha)$, $\eta=0.30$ and
$w=0.12$.

This tracker has **no finite-sample coverage guarantee**. The step resembles
adaptive conformal inference ([Gibbs and Candès](https://arxiv.org/abs/2106.00170)),
whose long-run miss-rate bound assumes a fixed target level and an unclipped
update. This tracker moves its per-class target, clips and boosts its step, and
can jump upward, so that bound does not carry over. The repository proves no
long-run coverage bound for it either, and records no measurement of its
coverage. A class that is rarely observed is rarely updated.

### Cost-aware decisions with Conformal Risk Control

`UtilityCalibrator` takes a cost matrix $C\in\mathbb R^{A\times K}$ indexed by
action and true class. The safe action $a_s$ defaults to
$\arg\min_a\max_kC_{ak}$, the action with the smallest worst-case cost. With
$c_{\rm esc}=\max_kC_{a_sk}$ and $B=\max_{a,k}C_{ak}$, the default
`loss_mode="normalized_excess"` scores

$$L_{ak}=\operatorname{clip}\!\left(\frac{C_{ak}-c_{\rm esc}}{B-c_{\rm esc}},0,1\right)
\quad\text{when }C_{ak}>c_{\rm esc},\qquad L_{ak}=0\text{ otherwise},$$

and `loss_mode="indicator"` uses $L_{ak}=\mathbf 1\{C_{ak}>c_{\rm esc}\}$. The
safe action therefore has zero loss for every class. For probabilities $p(x)$,
the Bayes action is $a^*(x)=\arg\min_a\sum_kp_k(x)C_{ak}$. At threshold
$\lambda$ the policy returns $a^*(x)$ when $\sum_kp_k(x)L_{a^*(x)k}\le\lambda$
and $a_s$ otherwise. With `ladder_mode=True`, used by `UtilityCascadeRouter`, a
Bayes action that precedes $a_s$ in action order is replaced by the first action
from $a^*(x)$ through $a_s$ whose expected loss is at most $\lambda$.

`fit_crc` evaluates the policy on $n$ calibration pairs over a grid $\Lambda$ of
501 points in $[0,1]$. Let $\ell_i(\lambda)\in[0,1]$ be the realized loss of
pair $i$ and $\bar\ell_i(\lambda)=\max_{\lambda'\in\Lambda,\lambda'\le\lambda}\ell_i(\lambda')$
its running maximum, which is nondecreasing in $\lambda$. The calibrator selects

$$\hat\lambda=\max\Bigl\{\lambda\in\Lambda:\frac{\sum_{i=1}^n\bar\ell_i(\lambda)+1}{n+1}\le\alpha\Bigr\}.$$

If no grid point qualifies, or $1/(n+1)>\alpha$, then `lam_crc` is $-1$,
`is_vacuous` is `True`, and `predict` always returns the safe action. Otherwise,
if the $n$ calibration pairs and the test pair are exchangeable and the
probabilities (including any temperature) do not depend on the calibration
pairs, the Conformal Risk Control argument
([Angelopoulos et al.](https://arxiv.org/abs/2208.02814)) applied to
$\bar\ell$ gives

$$\mathbb E\bigl[\ell_{n+1}(\hat\lambda)\bigr]\le\mathbb E\bigl[\bar\ell_{n+1}(\hat\lambda)\bigr]\le\alpha.$$

This bounds the expected normalized cost in excess of the safe action's worst
case (or, with `"indicator"`, the probability of exceeding it), averaged over
inputs. It does not bound expected cost itself, the loss conditional on a
non-safe action being chosen, or the loss for each class. Temperature fitting is
off by default (`fit_temperature=False`). When it is on, `fit` fits the
temperature on rows kept apart from the risk-control rows: the training rows
when separate calibration rows are supplied, or otherwise the first
$\max(8,\min(\lfloor n/3\rfloor,128))$ rows when $n\ge24$. That split takes rows
in the given order, so the order must not depend on the data.

### Ordinal heads and intervals

`CumulativeLogitOrdinalHead` fits a proportional-odds model,
$\Pr(Y\le j\mid x)=\sigma(\theta_j-w^\top x)$ for $j<K-1$, with increasing
thresholds $\theta_j=\theta_{j-1}+\operatorname{softplus}(\delta_j)+10^{-4}$.
It minimizes mean negative log-likelihood plus $\tfrac{\lambda}{2}\|w\|^2$
(default $\lambda=0.01$) with a NumPy L-BFGS routine. `SoftOrdinalHead` fits a
linear softmax by cross-entropy, with the same penalty on weights only, to soft
targets $q_{ik}\propto\exp\bigl(-(k-y_i)^2/(2\sigma^2)\bigr)$ (default
$\sigma=0.5$; $\sigma=0$ gives one-hot targets). Its docstring also names a
Laplace kernel, but only the Gaussian kernel is implemented.
`should_use_soft_ordinal(K)` is true for $2\le K\le5$; this is a fixed size rule,
not a statistical test.

`cqr_ordinal_intervals` takes the lower and upper indices $\ell(x),h(x)$ where
each probability row's cumulative distribution first reaches $\alpha/2$ and
$1-\alpha/2$, scores calibration pairs by $s_i=\max(\ell_i-y_i,\,y_i-h_i)$, and
sets $\hat q$ to the $\lceil(n+1)(1-\alpha)\rceil$-th smallest score. The test
interval is $[\ell-\hat q,\,h+\hat q]$ intersected with $\{0,\ldots,K-1\}$; when
$\lceil(n+1)(1-\alpha)\rceil>n$ it is the full range. A label lies in the
unclipped interval exactly when its score is at most $\hat q$, so the
split-conformal argument of section 1 gives marginal coverage of at least
$1-\alpha$ under the same exchangeability and fixed-model assumptions. Clipping
to the label range removes no possible label. An empty interval, possible when
$\hat q<0$, is replaced by its midpoint label, which can only add coverage.

`BinaryPlattBetaCalibrator` returns the input probabilities unchanged below
`min_platt_samples` (default 74) calibration examples, fits two-parameter Platt
scaling below `min_beta_samples` (default 150), and fits three-parameter beta
calibration from there on. These thresholds are fixed rules; the calibrator
makes no coverage statement.

### Soft-distilled heads

`SystemOneCompiler.compile_soft_distilled` (in [compiler.py](../../src/system1/compiler.py))
fits Choice and Boolean heads to teacher probability rows. It splits the rows
into two to five stratified folds, chooses the regularization (and, when hard
labels are supplied, their blend weight) from small grids and fits a temperature
using out-of-fold predictions, computes LAC calibration
scores from those same out-of-fold probabilities, and then refits the head on all
rows. The scores use the supplied hard labels, or the teacher's most probable
label when none are supplied. The deployed head is not the model that produced
the scores, and the rows that set the quantile also chose the hyperparameters and
temperature, so the split-conformal guarantee of section 1 does not apply. No
coverage guarantee is stated for these heads. Without hard labels, any coverage
reading is relative to the teacher's top label, not to correct answers.

### Score fields in automatic promotion

Since 1.3, `AutoCutoverLifecycle` accepts a `ScoreField` primary field. A local
score agrees with the teacher when it lies within `CutoverConfig.score_tolerance`
(default 0.15) times the field's range. That agreement indicator is what the
lifecycle's `SelectiveRiskCalibrator` margin gate (section 6) counts as correct,
so its risk statement is about agreement within the tolerance. The tolerance is a
policy choice, not a calibrated interval.
