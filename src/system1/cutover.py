"""Autonomous cold-start to selective hot-swap lifecycle for System 1: Instinct.

Implements ``AutoCutoverLifecycle``, managing cold-start abstention,
quarantine-filtered disjoint train/calibration reservoirs, micro-compiled hybrid
hot-swaps, OOD centroid + Adaptive Conformal Inference gating, and anytime-valid
e-process drift recalibration with zero pretrained weights.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
from dataclasses import dataclass
from enum import Enum
import math
import re
import threading
import time
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Type, Union

import numpy as np

from system1.calibration import (
    AdaptiveConformalTracker,
    SelectiveRiskCalibrator,
    SequentialRiskMonitor,
)
from system1.compiler import SystemOneCompiler, VALID_SEMANTIC_MODES
from system1.core.embeddings import (
    HybridProjector,
    SubwordSemanticEmbeddings,
    compute_strict_loo_weights,
)
from system1.core.schema import (
    BooleanField,
    ChoiceField,
    DecisionField,
    DecisionSchema,
    ScoreField,
    _score_half_span,
    _score_normalized_margin,
    _score_to_unit_ratio,
    _score_within_rel_tolerance,
)
from system1.engine import SystemOneEngine


_ACTIVE_CANARY_TX_TLS = threading.local()
_UNTRACKED_CANARY_SENTINEL = object()
_MISSING_CANARY_BINDING = object()


@contextlib.contextmanager
def _bind_active_canary_tx(*targets: Any, tx_token: Any) -> Any:
    bindings = getattr(_ACTIVE_CANARY_TX_TLS, "bindings", None)
    if not isinstance(bindings, dict):
        bindings = {}
        _ACTIVE_CANARY_TX_TLS.bindings = bindings
    saved: List[Tuple[int, Any]] = []
    seen_ids: set[int] = set()
    for target in targets:
        if target is None:
            continue
        tid = id(target)
        if tid in seen_ids:
            continue
        seen_ids.add(tid)
        saved.append((tid, bindings.get(tid, _MISSING_CANARY_BINDING)))
        bindings[tid] = tx_token
    try:
        yield
    finally:
        for tid, prev_val in saved:
            if prev_val is _MISSING_CANARY_BINDING:
                bindings.pop(tid, None)
            else:
                bindings[tid] = prev_val


def _normalize_cache_key(prompt: str) -> str:
    return " ".join(prompt.casefold().split())


def _resolve_cache_key(prompt: str, cache_key: Optional[str] = None) -> str:
    if isinstance(cache_key, str) and cache_key:
        return cache_key
    return _normalize_cache_key(prompt)


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _char_4grams(words: Sequence[str]) -> Set[str]:
    joined = " " + " ".join(words) + " "
    return {joined[i : i + 4] for i in range(max(0, len(joined) - 3))}


class CutoverState(str, Enum):
    """Operational lifecycle states for autonomous System 1 cutover."""

    COLD_START_ABSTAIN = "COLD_START_ABSTAIN"
    MICRO_COMPILED_WARMUP = "MICRO_COMPILED_WARMUP"
    SELECTIVE_HOT_SWAP = "SELECTIVE_HOT_SWAP"
    DRIFT_RECALIBRATE = "DRIFT_RECALIBRATE"


@dataclass(frozen=True)
class CutoverConfig:
    """Configuration for ``AutoCutoverLifecycle``."""

    milestones: Tuple[int, ...] = (
        56,
        96,
        144,
        208,
        288,
        384,
        512,
        672,
        864,
        1120,
        1440,
        1800,
    )
    k_min: int = 5
    alpha_target: float = 0.05
    canary_rate: float = 0.05
    sparse_dim: int = 3072
    dense_dim: int = 192
    hybrid_alpha: float = 0.78
    familiarity_alpha: float = 0.07
    prototype_alpha: float = 0.07
    cls_sim_q: float = 0.06
    cls_min_abs: float = 0.31
    enable_quarantine: bool = True
    semantic_mode: str = "default"
    drift_delta: float = 0.05
    score_tolerance: float = 0.15

    def __post_init__(self) -> None:
        if not self.milestones or any(int(m) < 2 for m in self.milestones):
            raise ValueError("milestones must contain positive integers >= 2")
        if int(self.k_min) < 1:
            raise ValueError(f"k_min must be >= 1, got {self.k_min}")
        if not (0.0 < float(self.alpha_target) < 1.0):
            raise ValueError(f"alpha_target must be in (0, 1), got {self.alpha_target}")
        if not (0.0 < float(self.canary_rate) <= 1.0):
            raise ValueError(f"canary_rate must be in (0, 1], got {self.canary_rate}")
        if int(self.sparse_dim) < 16 or int(self.dense_dim) < 16:
            raise ValueError("sparse_dim and dense_dim must be >= 16")
        if not (0.0 <= float(self.hybrid_alpha) <= 1.0):
            raise ValueError(f"hybrid_alpha must be in [0, 1], got {self.hybrid_alpha}")
        if self.semantic_mode not in VALID_SEMANTIC_MODES:
            raise ValueError(
                f"Unsupported semantic_mode {self.semantic_mode!r}; expected one of {VALID_SEMANTIC_MODES}"
            )
        if not (0.0 < float(self.drift_delta) < 1.0):
            raise ValueError(f"drift_delta must be in (0, 1), got {self.drift_delta}")
        if not (0.0 < float(self.score_tolerance) <= 1.0):
            raise ValueError(f"score_tolerance must be in (0, 1], got {self.score_tolerance}")


@dataclass(frozen=True)
class CutoverStepResult:
    """Structured result emitted by ``AutoCutoverLifecycle.step``."""

    values: Dict[str, Any]
    used_teacher: bool
    state: str
    is_canary: bool = False
    quarantined: bool = False
    margin: float = 0.0
    compile_triggered: bool = False


class AutoCutoverLifecycle:
    """Thread-safe autonomous cold-start -> micro-compile -> selective hot-swap manager.

    Progresses through four explicit lifecycle states:
      1. ``COLD_START_ABSTAIN``: escalates all queries to teacher and builds
         quarantine-filtered disjoint train (60%) and calibration (40%) pools.
      2. ``MICRO_COMPILED_WARMUP``: compiles hybrid local model at milestone
         admitted counts; holds back immature classes (< ``k_min``).
      3. ``SELECTIVE_HOT_SWAP``: fast-paths mature classes when familiarity,
         prototype, per-class subword centroid coherence, LTT/shadow margin gate
         ``tau``, and ``AdaptiveConformalTracker`` all agree.
      4. ``DRIFT_RECALIBRATE``: sparse canary teacher audits feed an anytime-valid
         ``SequentialRiskMonitor`` (e-process) and sliding window to trigger
         recalibration and re-compilation upon concept or prior shift.
    """

    def __init__(
        self,
        schema: Union[DecisionSchema, Type[DecisionSchema]],
        teacher: Optional[Callable[[str], Any]] = None,
        *,
        config: Optional[CutoverConfig] = None,
        post_compile_hook: Optional[Callable[[Any, Sequence[Tuple[str, Dict[str, Any]]]], bool]] = None,
        **overrides: Any,
    ) -> None:
        if isinstance(schema, type) and issubclass(schema, DecisionSchema):
            self.schema: DecisionSchema = schema()
        elif isinstance(schema, DecisionSchema):
            self.schema = schema
        else:
            raise TypeError(f"Expected DecisionSchema subclass or instance, got {type(schema).__name__}")

        if config is None:
            self.config = CutoverConfig(**overrides)
        elif overrides:
            self.config = dataclasses.replace(config, **overrides)
        else:
            self.config = config

        self.teacher = teacher
        self.post_compile_hook = post_compile_hook
        self._lock = threading.RLock()

        primary_name = ""
        primary_field: Optional[DecisionField] = None
        for fname, fdef in self.schema.fields.items():
            if isinstance(fdef, (ChoiceField, BooleanField)):
                primary_name = fname
                primary_field = fdef
                break
        if primary_field is None:
            primary_name, primary_field = next(iter(self.schema.fields.items()))
        self.primary_field_name: str = primary_name
        self.primary_field: DecisionField = primary_field

        if isinstance(primary_field, ChoiceField):
            self.total_schema_classes = max(2, len(primary_field.options))
        elif isinstance(primary_field, ScoreField):
            self.total_schema_classes = 4
        elif isinstance(primary_field, BooleanField):
            self.total_schema_classes = 2
        else:
            self.total_schema_classes = 2

        self.subword = SubwordSemanticEmbeddings(self.config.dense_dim)
        self.engine = SystemOneEngine(
            self.schema,
            dimension=self.config.dense_dim,
            use_cache=False,
            strict_mode=True,
            min_calibration_lessons=1_000_000,
            min_override_corroborations=2,
        )
        self.aci = AdaptiveConformalTracker(
            alpha_target=self.config.alpha_target,
            gamma=0.025,
            alpha_min=0.01,
            alpha_max=self.config.alpha_target,
            escalation_learning_rate_scale=0.25,
        )
        self.drift_monitor = SequentialRiskMonitor(
            alpha=self.config.alpha_target,
            delta=self.config.drift_delta,
            mode="capital_ons",
        )

        self.admitted_keys: Set[str] = set()
        self.admitted_vecs: List[np.ndarray] = []
        self.admitted_labels: List[str] = []
        self.admitted_values: List[Dict[str, Any]] = []
        self._vec_buf = np.empty((0, self.config.dense_dim), dtype=np.float32)
        self.ng_df: Dict[str, int] = {}
        self.admitted_lengths: List[int] = []

        self.train_by_class: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        self.calib_by_class: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        self._calib_cache_keys: Dict[str, List[str]] = {}
        self.class_counts: Dict[str, int] = {}
        self.quarantine: List[Tuple[str, str, Dict[str, Any], np.ndarray, List[str], Set[str], str]] = []

        self.state: str = CutoverState.COLD_START_ABSTAIN.value
        self.compiled_active: bool = False
        self.active_classes: Set[str] = set()
        self.mature_classes: Set[str] = set()
        self.selective_cal: Optional[SelectiveRiskCalibrator] = None
        self.active_tau: Optional[float] = None
        self.tau_source: str = "none"
        self.eta_hat: float = 0.0
        self.last_compile_n: int = 0
        self.next_milestone_idx: int = 0
        self.pending_recalibrate: bool = False

        self.class_centroids: Dict[str, np.ndarray] = {}
        self.class_sim_floors: Dict[str, float] = {}
        self.teacher_key_labels: Dict[str, Dict[str, Tuple[int, Dict[str, Any]]]] = {}
        self.clean_cache: Dict[str, Dict[str, Any]] = {}
        self._conflicted_cache_keys: Set[str] = set()
        self.recent_unseen_window: Deque[int] = collections.deque(maxlen=40)

        self.step_count: int = 0
        self.local_hits: int = 0
        self.local_accept_eligible: int = 0
        self.canary_checks: int = 0
        self.canary_window: Deque[int] = collections.deque(maxlen=20)
        self._pending_canaries: Dict[str, Tuple[Dict[str, Any], float, bool]] = {}
        self._pending_canary_stack: Dict[str, List[Tuple[object, Tuple[Dict[str, Any], float, bool]]]] = {}
        self._untracked_canaries: Dict[str, Tuple[Dict[str, Any], float, bool]] = {}
        self._untracked_canary_de: Dict[str, int] = {}
        self.cutover_events: List[Dict[str, Any]] = []
        self.recalibrate_events: List[Dict[str, Any]] = []
        self.quarantined_total: int = 0
        self.compile_time_total_ms: float = 0.0

    @property
    def is_cutover(self) -> bool:
        """Whether the lifecycle is actively promoted in SELECTIVE_HOT_SWAP."""
        return (
            self.state == CutoverState.SELECTIVE_HOT_SWAP.value
            and bool(self.compiled_active)
            and self.engine is not None
        )

    def _coerce_teacher_values(self, teacher_output: Any) -> Dict[str, Any]:
        if hasattr(teacher_output, "values") and isinstance(getattr(teacher_output, "values", None), Mapping):
            raw = dict(teacher_output.values)
        elif isinstance(teacher_output, Mapping):
            if "values" in teacher_output and isinstance(teacher_output["values"], Mapping):
                raw = dict(teacher_output["values"])
            else:
                raw = dict(teacher_output)
        elif len(self.schema.fields) == 1:
            only_name = next(iter(self.schema.fields.keys()))
            raw = {only_name: teacher_output}
        else:
            raise TypeError(
                f"Teacher output must be a Mapping for multi-field schema, got {type(teacher_output).__name__}"
            )
        return self.schema.validate_decision(raw)

    def _primary_label_key(self, values: Mapping[str, Any]) -> str:
        val = values[self.primary_field_name]
        if isinstance(self.primary_field, ScoreField):
            lo = float(self.primary_field.min_value)
            hi = float(self.primary_field.max_value)
            if hi <= lo:
                norm = 0.0
            else:
                norm = min(1.0, max(0.0, float(_score_to_unit_ratio(float(val), lo, hi))))
            return f"bin_{min(3, int(norm * 4.0))}"
        if isinstance(self.primary_field, BooleanField):
            return "True" if bool(val) else "False"
        if isinstance(val, (tuple, list)):
            return "|".join(str(v) for v in val)
        return str(val)

    def _values_match(
        self,
        pred_values: Mapping[str, Any],
        target_values: Mapping[str, Any],
        schema: Optional[DecisionSchema] = None,
    ) -> bool:
        """Compare predicted and target values across fields using tolerance for ScoreField."""
        target_schema = schema or self.schema
        for fname, f_def in target_schema.fields.items():
            if fname not in pred_values or fname not in target_values:
                return False
            p_val = pred_values[fname]
            t_val = target_values[fname]
            if isinstance(f_def, ScoreField):
                try:
                    p_f = float(p_val)
                    t_f = float(t_val)
                except (TypeError, ValueError):
                    return False
                if not (math.isfinite(p_f) and math.isfinite(t_f)):
                    return False
                if not _score_within_rel_tolerance(
                    p_f,
                    t_f,
                    f_def.min_value,
                    f_def.max_value,
                    float(self.config.score_tolerance),
                ):
                    return False
            elif isinstance(f_def, BooleanField):
                if bool(p_val) != bool(t_val):
                    return False
            else:
                if p_val != t_val:
                    return False
        return True

    def _decision_min_margin(
        self,
        res: Any,
        engine: Optional[SystemOneEngine],
        field_names: Iterable[str],
    ) -> float:
        """Return minimum field confidence/margin across fields, supporting ScoreField intervals."""
        margins: List[float] = []
        reg_preds = getattr(engine, "regression_conformal_predictors", {}) if engine is not None else {}
        for fname in field_names:
            if fname in res.margins:
                margins.append(float(res.margins[fname]))
                continue
            f_def = self.schema.fields.get(fname)
            if isinstance(f_def, ScoreField):
                half_span = _score_half_span(f_def.min_value, f_def.max_value)
                conf = float(res.confidences.get(fname, 0.0))
                rcp = reg_preds.get(fname)
                alpha_eff = max(1e-6, float(getattr(res, "alpha", self.config.alpha_target)))
                if (
                    half_span > 0.0
                    and rcp is not None
                    and getattr(rcp, "is_calibrated", False)
                    and len(getattr(rcp, "residuals", ())) >= math.ceil(1.0 / alpha_eff) - 1
                ):
                    iv = rcp.predict_interval(float(res.values[fname]), alpha=alpha_eff)
                    ratio = _score_normalized_margin(float(iv.margin), f_def.min_value, f_def.max_value)
                    half_w_margin = max(0.0, 1.0 - ratio)
                    margins.append(min(conf, half_w_margin) if ratio < 1.0 else conf)
                else:
                    margins.append(conf)
            else:
                margins.append(float(res.margins.get(fname, 0.0)))
        return min(margins) if margins else 0.0

    def _admit_record_locked(
        self,
        prompt: str,
        label_key: str,
        values: Dict[str, Any],
        vec: np.ndarray,
        words: List[str],
        ngs: Set[str],
        *,
        cache_key: Optional[str] = None,
    ) -> None:
        key = _resolve_cache_key(prompt, cache_key)
        if key in self.admitted_keys:
            return
        idx = len(self.admitted_vecs)
        if idx >= self._vec_buf.shape[0]:
            cur_cap = self._vec_buf.shape[0]
            grow = 64 if cur_cap == 0 else max(64, min(1000, cur_cap))
            new_buf = np.zeros((cur_cap + grow, self.config.dense_dim), dtype=np.float32)
            if cur_cap > 0:
                new_buf[:cur_cap] = self._vec_buf
            self._vec_buf = new_buf
        self._vec_buf[idx] = vec
        self.admitted_keys.add(key)
        self.admitted_vecs.append(vec)
        self.admitted_labels.append(label_key)
        self.admitted_values.append(dict(values))
        for g in ngs:
            self.ng_df[g] = self.ng_df.get(g, 0) + 1
        self.admitted_lengths.append(len(words))

        cnt = self.class_counts.get(label_key, 0)
        self.class_counts[label_key] = cnt + 1
        if cnt < 2 or (cnt % 5 in (0, 2, 4)):
            self.train_by_class.setdefault(label_key, []).append((prompt, dict(values)))
        else:
            self.calib_by_class.setdefault(label_key, []).append((prompt, dict(values)))
            self._calib_cache_keys.setdefault(label_key, []).append(key)

    def _try_admit_or_quarantine_locked(
        self,
        prompt: str,
        values: Dict[str, Any],
        *,
        cache_key: Optional[str] = None,
    ) -> bool:
        """Return True if quarantined, False if admitted or already known."""
        label_key = self._primary_label_key(values)
        self.recent_unseen_window.append(0 if label_key in self.active_classes else 1)
        key = _resolve_cache_key(prompt, cache_key)
        k_cnts = self.teacher_key_labels.setdefault(key, {})
        has_conflict = any(
            lbl_prev != label_key or not self._values_match(vals_prev, values)
            for lbl_prev, (_, vals_prev) in k_cnts.items()
        )
        prev_cnt, _ = k_cnts.get(label_key, (0, values))
        k_cnts[label_key] = (prev_cnt + 1, dict(values))
        if has_conflict:
            self._conflicted_cache_keys.add(key)
            self.clean_cache.pop(key, None)
        elif (
            key not in self._conflicted_cache_keys
            and len(k_cnts) == 1
            and (prev_cnt + 1) >= 2
            and label_key in self.mature_classes
        ):
            self.clean_cache[key] = dict(values)
        if key in self.admitted_keys:
            return False

        words = _tokenize_words(prompt)
        ngs = _char_4grams(words)
        vec = self.subword.encode(prompt)

        if not self.config.enable_quarantine:
            self._admit_record_locked(prompt, label_key, values, vec, words, ngs, cache_key=key)
            return False

        n_adm = len(self.admitted_vecs)
        if n_adm < 20:
            self._admit_record_locked(prompt, label_key, values, vec, words, ngs, cache_key=key)
            return False

        p90_len = float(np.percentile(self.admitted_lengths, 90))
        if len(words) > max(26, 2.3 * p90_len):
            self.quarantined_total += 1
            self.quarantine.append((prompt, label_key, dict(values), vec, words, ngs, key))
            return True

        ng_cov = sum(1 for g in ngs if self.ng_df.get(g, 0) >= 1) / max(1, len(ngs))
        sims = self._vec_buf[:n_adm] @ vec
        max_sim = float(np.max(sims))

        if ng_cov < 0.42 and max_sim < 0.30:
            self.quarantined_total += 1
            self.quarantine.append((prompt, label_key, dict(values), vec, words, ngs, key))
            return True

        cls_cnt = self.class_counts.get(label_key, 0)
        if cls_cnt >= 2:
            same_sims = [
                float(sims[i]) for i in range(n_adm) if self.admitted_labels[i] == label_key
            ]
            same_max = max(same_sims) if same_sims else 0.0
            top_k = np.argsort(-sims)[: min(8, n_adm)]
            knn_agree = sum(1 for i in top_k if self.admitted_labels[i] == label_key)
            if (max_sim >= 0.28 or ng_cov >= 0.58) and (
                same_max >= 0.20 or knn_agree >= 1 or ng_cov >= 0.65
            ):
                self._admit_record_locked(prompt, label_key, values, vec, words, ngs, cache_key=key)
                return False
            self.quarantined_total += 1
            self.quarantine.append((prompt, label_key, dict(values), vec, words, ngs, key))
            return True

        if ng_cov >= 0.38 or max_sim >= 0.28:
            peers = [
                q
                for q in self.quarantine
                if q[1] == label_key
                and len(q[4]) <= max(26, 2.3 * p90_len)
                and float(np.dot(q[3], vec)) >= 0.14
            ]
            if peers:
                for q in peers:
                    self.quarantine.remove(q)
                    q_key = q[6] if len(q) > 6 else None
                    self._admit_record_locked(q[0], q[1], q[2], q[3], q[4], q[5], cache_key=q_key)
                self._admit_record_locked(prompt, label_key, values, vec, words, ngs, cache_key=key)
                return False

        self.quarantined_total += 1
        self.quarantine.append((prompt, label_key, dict(values), vec, words, ngs, key))
        return True

    def _estimate_noise_rate(
        self,
        proj: Any,
        tr_pairs: Sequence[Tuple[str, Dict[str, Any]]],
        active_list: Sequence[str],
    ) -> float:
        if len(tr_pairs) < 40 or len(active_list) < 2:
            return 0.0
        opt_to_idx = {o: i for i, o in enumerate(active_list)}
        tr_texts = [p for p, _ in tr_pairs]
        X_tr = proj.project_batch(tr_texts).astype(np.float32)
        Y_tr = np.zeros((len(tr_pairs), len(active_list)), dtype=np.float32)
        for i, (_, vals) in enumerate(tr_pairs):
            Y_tr[i, opt_to_idx[self._primary_label_key(vals)]] = 1.0
        sw, _ = compute_strict_loo_weights(X_tr, Y_tr)
        if len(tr_pairs) >= 80:
            return float(min(0.115, float(np.mean(sw < 0.68)) * 1.25))
        return float(min(0.08, float(np.mean(sw < 0.52)) * 1.15))

    def _select_shadow_threshold_locked(
        self,
        margins_cal: List[float],
        correct_cal: List[bool],
        eta_hat: float,
        n_classes: int,
    ) -> Optional[float]:
        if len(margins_cal) < 20:
            return None
        unseen_rate = sum(self.recent_unseen_window) / max(1, len(self.recent_unseen_window))
        per_cls = len(margins_cal) / max(1, n_classes)

        if self.total_schema_classes >= 45:
            floor_min = 0.91 if (n_classes < 45 or per_cls < 9.5) else 0.86
            ucb_z = 1.50
            max_deb = 0.030
            max_ucb = 0.070
            max_grid = 0.985
            min_sub = 15
            eta_sub = min(eta_hat, 0.09)
        elif unseen_rate > 0.25 and len(margins_cal) < 100:
            floor_min = 0.84
            ucb_z = 1.45
            max_deb = 0.028
            max_ucb = 0.068
            max_grid = 0.96
            min_sub = 18
            eta_sub = eta_hat
        elif len(margins_cal) < 55 or per_cls < 3.6:
            floor_min = 0.72
            ucb_z = 1.35
            max_deb = 0.028
            max_ucb = 0.068
            max_grid = 0.96
            min_sub = 18
            eta_sub = eta_hat
        else:
            floor_min = 0.66
            ucb_z = 1.35
            max_deb = 0.028
            max_ucb = 0.068
            max_grid = 0.96
            min_sub = 18
            eta_sub = eta_hat

        grid = [round(floor_min + 0.025 * i, 4) for i in range(14)]
        grid = [g for g in grid if g <= max_grid]
        if self.total_schema_classes >= 45:
            for extra in (0.975, 0.985):
                if extra not in grid and floor_min <= extra:
                    grid.append(extra)
            grid.sort()
        denom = max(0.5, 1.0 - eta_sub * n_classes / max(1, n_classes - 1))
        for th in grid:
            sub = [ok for m, ok in zip(margins_cal, correct_cal) if m >= th]
            m_n = len(sub)
            if m_n < min_sub:
                continue
            p_err = 1.0 - (sum(sub) / m_n)
            p_deb = max(0.0, (p_err - eta_sub) / denom)
            ucb = p_deb + ucb_z * math.sqrt(max(p_deb * (1.0 - p_deb), 0.012) / m_n)
            if ucb <= max_ucb and p_deb <= max_deb:
                return float(th)
        return None

    def _build_active_schema(self, active_list: Sequence[str]) -> DecisionSchema:
        if len(self.schema.fields) == 1 and isinstance(self.primary_field, ChoiceField):
            return DecisionSchema(
                schema_name=self.schema.schema_name,
                fields={
                    self.primary_field_name: ChoiceField(
                        options=list(active_list),
                        descriptions={
                            o: self.primary_field.descriptions[o]
                            for o in active_list
                            if o in getattr(self.primary_field, "descriptions", {})
                        }
                        or None,
                        description=self.primary_field.description,
                    )
                },
            )
        new_fields: Dict[str, DecisionField] = {}
        for fname, fdef in self.schema.fields.items():
            if fname == self.primary_field_name and isinstance(fdef, ChoiceField):
                new_fields[fname] = ChoiceField(
                    options=list(active_list),
                    descriptions={
                        o: fdef.descriptions[o]
                        for o in active_list
                        if o in getattr(fdef, "descriptions", {})
                    }
                    or None,
                    description=fdef.description,
                )
            else:
                new_fields[fname] = fdef
        return DecisionSchema(schema_name=self.schema.schema_name, fields=new_fields)

    def _maybe_micro_compile_locked(self) -> bool:
        n_adm = len(self.admitted_vecs)
        hit_milestone = (
            self.next_milestone_idx < len(self.config.milestones)
            and n_adm >= self.config.milestones[self.next_milestone_idx]
        )
        new_mature = {c for c, cnt in self.class_counts.items() if cnt >= self.config.k_min}
        has_new_mature = (
            len(new_mature - self.active_classes) >= 2 and (n_adm - self.last_compile_n >= 24)
        )

        if (
            not hit_milestone
            and not (self.pending_recalibrate and n_adm - self.last_compile_n >= 16)
            and not has_new_mature
        ):
            return False

        while (
            self.next_milestone_idx < len(self.config.milestones)
            and n_adm >= self.config.milestones[self.next_milestone_idx]
        ):
            self.next_milestone_idx += 1

        active_list = sorted(
            c
            for c, tr in self.train_by_class.items()
            if len(tr) >= 2 and len(self.calib_by_class.get(c, ())) >= 1
        )
        if len(active_list) < 2:
            return False

        tr_pairs = [x for c in active_list for x in self.train_by_class[c]]
        cal_pairs = [x for c in active_list for x in self.calib_by_class[c]]
        cal_keys: List[str] = []
        for c in active_list:
            cls_pairs = self.calib_by_class[c]
            cls_keys = self._calib_cache_keys.get(c, ())
            if len(cls_keys) == len(cls_pairs):
                cal_keys.extend(cls_keys)
            else:
                cal_keys.extend(_normalize_cache_key(p_c) for p_c, _ in cls_pairs)
        if len(tr_pairs) < 24 or len(cal_pairs) < 12:
            return False

        t0 = time.perf_counter()
        active_schema = self._build_active_schema(active_list)
        tr_texts = [p for p, _ in tr_pairs]
        tr_labels = [self._primary_label_key(vals) for _, vals in tr_pairs]

        if self.config.semantic_mode in ("v1_2_fisher", "v1_2_rbf"):
            proj: Any = HybridProjector.fit_v1_2(
                tr_texts,
                labels=tr_labels,
                schema=active_schema,
                sparse_dim=self.config.sparse_dim,
                dense_dim=self.config.dense_dim,
                alpha=self.config.hybrid_alpha,
                attach_rbf=(self.config.semantic_mode == "v1_2_rbf"),
                backend="numpy",
            )
        else:
            try:
                proj = HybridProjector.fit(
                    tr_texts,
                    labels=tr_labels,
                    sparse_dim=self.config.sparse_dim,
                    dense_dim=self.config.dense_dim,
                    alpha=self.config.hybrid_alpha,
                    selection="frequency",
                    char_ngrams=(3, 5),
                    backend="numpy",
                )
            except np.linalg.LinAlgError:
                proj = HybridProjector.fit(
                    tr_texts,
                    labels=tr_labels,
                    sparse_dim=self.config.sparse_dim,
                    dense_dim=self.config.dense_dim,
                    alpha=self.config.hybrid_alpha,
                    max_vocab=128,
                    spectral_dim=16,
                    selection="frequency",
                    char_ngrams=(3, 5),
                    backend="numpy",
                )

        exemplars_map: Dict[str, List[Tuple[str, Any]]] = {
            fname: [(p, vals[fname]) for p, vals in tr_pairs if fname in vals]
            for fname in active_schema.fields
        }
        cal_exemplars_map: Dict[str, List[Tuple[str, Any]]] = {
            fname: [(p, vals[fname]) for p, vals in cal_pairs if fname in vals]
            for fname in active_schema.fields
        }

        comp = SystemOneCompiler(
            active_schema,
            projector=proj,
            dimension=proj.dimension,
            backend="numpy",
            regularization=0.5,
            auto_regularization=True,
            noise_robust=True,
            hierarchical="auto",
            familiarity_alpha=self.config.familiarity_alpha,
            prototype_alpha=self.config.prototype_alpha,
            mondrian=False,
            semantic_mode=self.config.semantic_mode,
        )
        compiled = comp.compile(
            exemplars=exemplars_map,
            calibration_exemplars=cal_exemplars_map,
            augment=False,
        )

        self.eta_hat = self._estimate_noise_rate(proj, tr_pairs, active_list)
        sub_eng = SystemOneEngine(
            active_schema,
            model=compiled,
            strict_mode=True,
            use_cache=False,
            aci_tracker=self.aci,
        )
        sub_eng.default_alpha = self.config.alpha_target
        if self.post_compile_hook is not None and self.post_compile_hook(sub_eng, tr_pairs):
            seen_cal: Set[str] = set()
            dedup_cal: List[Tuple[str, Dict[str, Any]]] = []
            for p_c, vals_c in cal_pairs:
                if p_c not in seen_cal:
                    seen_cal.add(p_c)
                    dedup_cal.append((p_c, dict(vals_c)))
            if dedup_cal:
                for cal_obj in sub_eng.calibrators.values():
                    if getattr(cal_obj, "logit_adjustments", None) is not None:
                        cal_obj.logit_adjustments = None
                sub_eng.calibrate(
                    dedup_cal,
                    familiarity_alpha=self.config.familiarity_alpha,
                    prototype_alpha=self.config.prototype_alpha,
                )
            if hasattr(sub_eng, "_packed_kernel_cache"):
                sub_eng._packed_kernel_cache = None
            exp_cache = getattr(sub_eng, "_explainer_cache", None)
            if exp_cache is not None:
                exp_cache.clear()

        cal_prompts = [p for p, _ in cal_pairs]
        cal_decs = sub_eng.decide_batch(
            cal_prompts,
            strict=False,
            record_receipt=False,
            use_cache=False,
        )
        margins_cal: List[float] = []
        correct_cal: List[bool] = []
        high_conf_cache: List[Tuple[str, Dict[str, Any]]] = []
        for (p_c, vals_c), k_c, d_c in zip(cal_pairs, cal_keys, cal_decs):
            if d_c.abstention_reason in ("low_feature_familiarity", "low_semantic_familiarity"):
                continue
            min_margin = self._decision_min_margin(d_c, sub_eng, active_schema.fields)
            is_ok = self._values_match(d_c.values, vals_c, schema=active_schema)
            margins_cal.append(min_margin)
            correct_cal.append(is_ok)
            if d_c.abstention_reason is None and is_ok and min_margin >= 0.85:
                high_conf_cache.append((k_c, dict(vals_c)))

        k_act = len(active_list)
        alpha_calib = min(
            0.18,
            self.config.alpha_target
            + self.eta_hat * (1.0 - self.config.alpha_target * k_act / max(1, k_act - 1)),
        )
        sel_cal = SelectiveRiskCalibrator(
            score_name="margin",
            method="ltt_bonferroni",
            alpha=alpha_calib,
            delta=0.05,
            grid_size=6,
            max_frac=0.90,
        )
        if len(margins_cal) >= 20:
            sel_cal.fit(margins_cal, correct_cal)

        if sel_cal.is_qualified() and sel_cal.threshold is not None:
            tau = float(sel_cal.threshold)
            tau_src = "ltt_bonferroni"
        else:
            tau = self._select_shadow_threshold_locked(margins_cal, correct_cal, self.eta_hat, k_act)
            tau_src = "debiased_shadow" if tau is not None else "none"

        by_c: Dict[str, List[np.ndarray]] = {}
        for v, lbl in zip(self.admitted_vecs, self.admitted_labels):
            by_c.setdefault(lbl, []).append(v)
        self.class_centroids = {}
        self.class_sim_floors = {}
        for c, vs in by_c.items():
            mat = np.stack(vs, axis=0)
            cent = np.mean(mat, axis=0)
            cent = cent / max(1e-8, float(np.linalg.norm(cent)))
            self.class_centroids[c] = cent
            sims = mat @ cent
            self.class_sim_floors[c] = (
                float(np.quantile(sims, self.config.cls_sim_q)) if len(sims) >= 5 else 0.30
            )

        self.engine = sub_eng
        self.selective_cal = sel_cal
        self.active_tau = tau
        self.tau_source = tau_src
        self.active_classes = set(active_list)
        self.mature_classes = {
            c for c in active_list if self.class_counts.get(c, 0) >= self.config.k_min
        }
        self.compiled_active = True
        self.last_compile_n = n_adm
        self.pending_recalibrate = False
        self.aci.margin_floor = tau
        self.drift_monitor = SequentialRiskMonitor(
            alpha=min(0.45, max(0.02, self.config.alpha_target + self.eta_hat)),
            delta=self.config.drift_delta,
            mode="capital_ons",
        )

        for k_c, vals_c in high_conf_cache:
            if k_c not in self.clean_cache and k_c not in self._conflicted_cache_keys:
                self.clean_cache[k_c] = dict(vals_c)

        if tau is not None and len(self.mature_classes) >= 2:
            self.state = CutoverState.SELECTIVE_HOT_SWAP.value
        else:
            self.state = CutoverState.MICRO_COMPILED_WARMUP.value

        compile_ms = (time.perf_counter() - t0) * 1000.0
        self.compile_time_total_ms += compile_ms
        self.cutover_events.append(
            {
                "t": self.step_count,
                "state": self.state,
                "n_admitted": n_adm,
                "quarantined_current": len(self.quarantine),
                "active_classes": k_act,
                "mature_classes": len(self.mature_classes),
                "n_train": len(tr_pairs),
                "n_calib": len(cal_pairs),
                "eta_hat": round(self.eta_hat, 4),
                "alpha_calib": round(alpha_calib, 4),
                "tau": round(tau, 4) if tau is not None else None,
                "tau_source": tau_src,
                "compile_ms": round(compile_ms, 2),
            }
        )
        return True

    def _record_canary_check_locked(
        self,
        pred_values: Dict[str, Any],
        y_teacher_values: Dict[str, Any],
        *,
        margin: float,
        from_cache: bool,
    ) -> None:
        self.canary_checks += 1
        is_err = not self._values_match(pred_values, y_teacher_values)
        self.canary_window.append(1 if is_err else 0)
        self.aci.update_accepted(is_err, margin=(None if from_cache else margin))
        mon_step = self.drift_monitor.update(is_err, step=self.step_count)

        window_alarm = (
            len(self.canary_window) >= 12
            and (sum(self.canary_window) / len(self.canary_window))
            > (self.config.alpha_target + self.eta_hat + 0.08)
        )
        if mon_step.alarm or window_alarm:
            self.pending_recalibrate = True
            self.state = CutoverState.DRIFT_RECALIBRATE.value
            self.recalibrate_events.append(
                {
                    "t": self.step_count,
                    "canary_err": round(sum(self.canary_window) / max(1, len(self.canary_window)), 4),
                    "window": len(self.canary_window),
                    "e_value": round(mon_step.e_value, 4),
                    "e_alarm": bool(mon_step.alarm),
                    "window_alarm": bool(window_alarm),
                }
            )
            self.canary_window.clear()
            self.drift_monitor.reset()

    def _evaluate_local_candidate_locked(
        self,
        prompt: str,
        *,
        cache_key: Optional[str] = None,
        accept_predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]], float, bool, bool]:
        """Return (accepted_fastpath, candidate_values, margin, is_canary, from_cache)."""
        key = _resolve_cache_key(prompt, cache_key)
        canary_period = max(1, int(round(1.0 / self.config.canary_rate)))

        if key in self.clean_cache:
            cached_vals = dict(self.clean_cache[key])
            self.local_accept_eligible += 1
            is_canary = (self.local_accept_eligible % canary_period) == 0
            return (not is_canary), cached_vals, 1.0, is_canary, True

        if key in self._conflicted_cache_keys:
            return False, None, 0.0, False, False

        if not self.compiled_active or self.active_tau is None:
            return False, None, 0.0, False, False

        res = self.engine.decide(prompt, strict=False, record_receipt=False, use_cache=False)
        vals = dict(res.values)
        pred_key = self._primary_label_key(vals)
        margin = self._decision_min_margin(res, self.engine, res.values)

        if (
            res.abstention_reason is None
            and pred_key in self.mature_classes
            and margin >= self.active_tau
            and self.aci.should_accept_margin(margin)
        ):
            cent = self.class_centroids.get(pred_key)
            floor = self.class_sim_floors.get(pred_key, 0.30)
            if cent is None or float(np.dot(self.subword.encode(prompt), cent)) >= max(
                self.config.cls_min_abs, floor * 0.90
            ):
                if accept_predicate is not None and not accept_predicate(vals):
                    return False, None, margin, False, False
                self.local_accept_eligible += 1
                is_canary = (self.local_accept_eligible % canary_period) == 0
                return (not is_canary), vals, margin, is_canary, False

        return False, None, margin, False, False

    _supports_canary_tx_arg: bool = True

    def _sync_pending_canary_locked(self, key: str) -> None:
        stack = self._pending_canary_stack.get(key)
        if stack:
            self._pending_canaries[key] = stack[-1][1]
        elif key in self._untracked_canaries:
            self._pending_canary_stack.pop(key, None)
            self._pending_canaries[key] = self._untracked_canaries[key]
        else:
            self._pending_canary_stack.pop(key, None)
            self._pending_canaries.pop(key, None)

    def _clear_untracked_canary(self, prompt: str, *, cache_key: Optional[str] = None) -> None:
        try:
            key = _resolve_cache_key(prompt, cache_key)
        except Exception:
            return
        with self._lock:
            self._untracked_canaries.pop(key, None)
            prev_de = self._untracked_canary_de.pop(key, 0)
            if prev_de:
                self.local_accept_eligible = max(0, self.local_accept_eligible - prev_de)
            self._sync_pending_canary_locked(key)

    def _decide_local_with_tx(
        self,
        prompt: str,
        *,
        cache_key: Optional[str] = None,
        accept_predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
        _track_tx: bool = True,
    ) -> Tuple[Optional[Dict[str, Any]], Tuple[int, int, int, str, Any]]:
        """Attempt local decision and return (values_or_none, rollback_tx_token)."""
        with self._lock:
            self._maybe_micro_compile_locked()
            key = _resolve_cache_key(prompt, cache_key)
            if not _track_tx and key in self._untracked_canaries:
                self._untracked_canaries.pop(key, None)
                prev_de = self._untracked_canary_de.pop(key, 0)
                if prev_de:
                    self.local_accept_eligible = max(0, self.local_accept_eligible - prev_de)
                self._sync_pending_canary_locked(key)
            e0 = self.local_accept_eligible
            accepted, vals, margin, is_canary, from_cache = self._evaluate_local_candidate_locked(
                prompt,
                cache_key=key,
                accept_predicate=accept_predicate,
            )
            de = self.local_accept_eligible - e0
            if accepted and vals is not None:
                self.step_count += 1
                self.local_hits += 1
                return vals, (de, 1, 1, key, None)
            canary_tok: Any = None
            if is_canary and vals is not None:
                if (
                    key in self._pending_canaries
                    and key not in self._pending_canary_stack
                    and key not in self._untracked_canaries
                ):
                    self._untracked_canaries[key] = self._pending_canaries[key]
                added_pend = (vals, margin, from_cache)
                if _track_tx:
                    canary_tok = object()
                    self._pending_canary_stack.setdefault(key, []).append((canary_tok, added_pend))
                else:
                    canary_tok = _UNTRACKED_CANARY_SENTINEL
                    self._untracked_canaries[key] = added_pend
                    self._untracked_canary_de[key] = de
                self._sync_pending_canary_locked(key)
            return None, (de, 0, 0, key, canary_tok)

    def _rollback_local_tx(self, tx: Tuple[int, int, int, str, Any]) -> None:
        """Roll back local_accept_eligible, counters, and pending canary state from a failed request."""
        de, ds, dh, key, canary_tok = tx
        with self._lock:
            if de:
                self.local_accept_eligible = max(0, self.local_accept_eligible - de)
            if ds:
                self.step_count = max(0, self.step_count - ds)
            if dh:
                self.local_hits = max(0, self.local_hits - dh)
            if canary_tok is _UNTRACKED_CANARY_SENTINEL:
                self._untracked_canaries.pop(key, None)
                self._untracked_canary_de.pop(key, None)
                self._sync_pending_canary_locked(key)
            elif canary_tok is not None:
                stack = self._pending_canary_stack.get(key)
                if stack is not None:
                    self._pending_canary_stack[key] = [pair for pair in stack if pair[0] is not canary_tok]
                    self._sync_pending_canary_locked(key)

    def decide_local(
        self,
        prompt: str,
        *,
        cache_key: Optional[str] = None,
        accept_predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return validated local decision values when safe, or None to escalate/canary."""
        vals, _ = self._decide_local_with_tx(
            prompt,
            cache_key=cache_key,
            accept_predicate=accept_predicate,
            _track_tx=False,
        )
        return vals

    def observe_teacher(
        self,
        prompt: str,
        teacher_values: Any,
        *,
        cache_key: Optional[str] = None,
        _canary_tx: Optional[Any] = None,
    ) -> bool:
        """Ingest a validated teacher answer, update canary drift state, and admit/compile."""
        tls_map = getattr(_ACTIVE_CANARY_TX_TLS, "bindings", None)
        has_bound = isinstance(tls_map, dict) and id(self) in tls_map
        eff_canary_tx = tls_map[id(self)] if has_bound else _canary_tx
        try:
            validated = self._coerce_teacher_values(teacher_values)
        except BaseException:
            if not has_bound and eff_canary_tx is None:
                self._clear_untracked_canary(prompt, cache_key=cache_key)
            raise
        with self._lock:
            key = _resolve_cache_key(prompt, cache_key)
            pending: Optional[Tuple[Dict[str, Any], float, bool]] = None
            if has_bound or eff_canary_tx is not None:
                tx_tok = (
                    eff_canary_tx[4]
                    if (isinstance(eff_canary_tx, tuple) and len(eff_canary_tx) >= 5)
                    else None
                )
                if tx_tok is _UNTRACKED_CANARY_SENTINEL:
                    pending = self._untracked_canaries.pop(key, None)
                    self._untracked_canary_de.pop(key, None)
                    self._sync_pending_canary_locked(key)
                elif tx_tok is not None:
                    stack = self._pending_canary_stack.get(key)
                    if stack is not None:
                        new_stack: List[Tuple[object, Tuple[Dict[str, Any], float, bool]]] = []
                        for pair in stack:
                            if pending is None and pair[0] is tx_tok:
                                pending = pair[1]
                            else:
                                new_stack.append(pair)
                        self._pending_canary_stack[key] = new_stack
                        self._sync_pending_canary_locked(key)
            else:
                pending = self._untracked_canaries.pop(key, None)
                self._untracked_canary_de.pop(key, None)
                if pending is None and key in self._pending_canaries and key not in self._pending_canary_stack:
                    pending = self._pending_canaries.pop(key, None)
                self._sync_pending_canary_locked(key)
            if pending is not None:
                pred_vals, margin, from_cache = pending
                self._record_canary_check_locked(
                    pred_vals,
                    validated,
                    margin=margin,
                    from_cache=from_cache,
                )
            quarantined = self._try_admit_or_quarantine_locked(prompt, validated, cache_key=key)
            self._maybe_micro_compile_locked()
            self.step_count += 1
            return not quarantined

    def step(self, prompt: str) -> CutoverStepResult:
        """Execute one closed-loop step against ``self.teacher``."""
        if self.teacher is None or not callable(self.teacher):
            raise RuntimeError(
                "AutoCutoverLifecycle.step() requires a callable 'teacher'; "
                "use decide_local() + observe_teacher() when driving an external fallback."
            )

        with self._lock:
            compile_triggered = self._maybe_micro_compile_locked()
            e0 = self.local_accept_eligible
            try:
                accepted, vals, margin, is_canary, from_cache = self._evaluate_local_candidate_locked(prompt)
                if accepted and vals is not None:
                    self.step_count += 1
                    self.local_hits += 1
                    return CutoverStepResult(
                        values=vals,
                        used_teacher=False,
                        state=self.state,
                        is_canary=False,
                        quarantined=False,
                        margin=margin,
                        compile_triggered=compile_triggered,
                    )

                teacher_raw = self.teacher(prompt)
                validated = self._coerce_teacher_values(teacher_raw)
            except BaseException:
                self.local_accept_eligible = e0
                raise
            if is_canary and vals is not None:
                self._record_canary_check_locked(
                    vals,
                    validated,
                    margin=margin,
                    from_cache=from_cache,
                )
            quarantined = self._try_admit_or_quarantine_locked(prompt, validated)
            compile_triggered = self._maybe_micro_compile_locked() or compile_triggered
            self.step_count += 1
            return CutoverStepResult(
                values=validated,
                used_teacher=True,
                state=self.state,
                is_canary=is_canary,
                quarantined=quarantined,
                margin=margin,
                compile_triggered=compile_triggered,
            )

    def summary(self) -> Dict[str, Any]:
        """Return diagnostic snapshot of reservoirs, lifecycle state, and drift events."""
        with self._lock:
            return {
                "state": self.state,
                "step_count": self.step_count,
                "local_hits": self.local_hits,
                "compiled_active": self.compiled_active,
                "n_admitted": len(self.admitted_vecs),
                "quarantined_current": len(self.quarantine),
                "quarantined_total": self.quarantined_total,
                "active_classes": len(self.active_classes),
                "mature_classes": len(self.mature_classes),
                "active_tau": self.active_tau,
                "tau_source": self.tau_source,
                "eta_hat": round(self.eta_hat, 4),
                "canary_checks": self.canary_checks,
                "cutover_events": list(self.cutover_events),
                "recalibrate_events": list(self.recalibrate_events),
                "compile_time_total_ms": round(self.compile_time_total_ms, 2),
            }


__all__ = [
    "AutoCutoverLifecycle",
    "CutoverConfig",
    "CutoverState",
    "CutoverStepResult",
]
