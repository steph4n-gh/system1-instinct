"""Sub-100us exact additive attribution, exemplar witness, and counterfactual explainer."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import zlib

import numpy as np

from system1.core.embeddings import (
    _ANCHOR_STOPWORDS,
    HybridProjector,
    _CLAUSE_SPLIT_RE,
    _NEGATION_SKIP_MODIFIERS,
    _NEGATION_TOKENS,
    _normalize_text,
    _stem_word as _emb_stem_word,
    _tokenize_with_negation_scope,
)
from system1.core.model import _has_custom_validate_decision
from system1.core.schema import BooleanField, ChoiceField
from system1.core.text import (
    _CONTRACTION_RE,
    MAX_TERM_LENGTH,
    MAX_TEXT_LENGTH,
    TfidfProjector,
    _stem_word as _tfidf_stem_word,
)

_SPAN_WORD_RE = re.compile(r"(?u)\b\w+\b")
_CONTRACTION_SPAN_RE = re.compile(
    r"(?iu)\b(?:can't|won't|isn't|wasn't|didn't|don't|doesn't|couldn't|shouldn't|"
    r"wouldn't|aren't|weren't|haven't|hasn't|hadn't|\w+)\b"
)
_GL10_NODES, _GL10_WEIGHTS = np.polynomial.legendre.leggauss(10)
_IG_T = (0.5 * (_GL10_NODES + 1.0)).astype(np.float64).reshape(-1, 1)
_IG_W = (0.5 * _GL10_WEIGHTS).astype(np.float64).reshape(-1, 1)
_MAX_EXPLAIN_OOV_CACHE = 32768


def _boolean_threshold_logit_shift(threshold: float) -> float:
    t = float(threshold)
    if not math.isfinite(t):
        raise ValueError(f"threshold must be finite, got {threshold!r}")
    if t <= 0.0:
        return -750.0
    if t > 1.0:
        return 750.0
    if t == 1.0:
        return math.nextafter(53.0 * math.log(2.0), math.inf)
    return math.log(t) - math.log1p(-t)


def _normalize_head_weights_and_options(
    weights: np.ndarray,
    biases: np.ndarray,
    options: Optional[Sequence[str]],
    *,
    threshold: float = 0.5,
    temperature_scale: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    """Normalize ChoiceField or BooleanField (1, D) head into a multi-class matrix pair."""
    w = np.asarray(weights, dtype=np.float64)
    b = np.asarray(biases, dtype=np.float64).reshape(-1)
    if w.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix, got shape {w.shape}")
    opts = tuple(str(o) for o in options) if options is not None else ()

    if w.shape[0] == 1 and (not opts or len(opts) == 2):
        # Convert scalar BooleanField head w*x + b >= inv_sigmoid(t) into 2-class symmetric logits.
        logit_shift = _boolean_threshold_logit_shift(threshold)
        b_adj = float(b[0]) - (1.0 / float(temperature_scale)) * logit_shift
        w2 = np.vstack([-0.5 * w[0], 0.5 * w[0]])
        b2 = np.array([-0.5 * b_adj, 0.5 * b_adj], dtype=np.float64)
        return w2, b2, opts if len(opts) == 2 else ("False", "True")

    if not opts:
        opts = tuple(str(i) for i in range(w.shape[0]))
    if len(opts) != w.shape[0]:
        raise ValueError(f"Options length {len(opts)} does not match weight rows {w.shape[0]}")
    if b.shape[0] != w.shape[0]:
        raise ValueError(f"Bias length {b.shape[0]} does not match weight rows {w.shape[0]}")
    return w, b, opts


def _l2_normalize_batch(mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] == 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    good = (norms[:, 0] > 1e-12) & np.isfinite(norms[:, 0])
    out = np.zeros_like(arr, dtype=np.float32)
    if np.any(good):
        out[good] = (arr[good] / norms[good]).astype(np.float32)
    return out


def _project_one(projector: Any, text: str, *, recency_weighted: bool = False) -> np.ndarray:
    use_rw = bool(recency_weighted)
    if hasattr(projector, "project"):
        try:
            out = projector.project(text, recency_weighted=use_rw)
        except TypeError:
            out = projector.project(text)
    else:
        out = projector(text)
    if isinstance(out, tuple):
        out = out[0]
    vec = np.asarray(out, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-12 and math.isfinite(norm):
        return (vec / norm).astype(np.float32)
    return np.zeros_like(vec, dtype=np.float32)


def _project_texts(
    projector: Any,
    texts: Sequence[str],
    *,
    recency_weighted: bool = False,
) -> np.ndarray:
    text_list = list(texts)
    if not text_list:
        dim = int(getattr(projector, "dimension", 0))
        return np.empty((0, dim), dtype=np.float32)
    use_rw = bool(recency_weighted)
    project_batch = getattr(projector, "project_batch", None)
    if callable(project_batch):
        try:
            out = project_batch(text_list, recency_weighted=use_rw)
        except TypeError:
            if not use_rw:
                out = project_batch(text_list)
            else:
                out = np.vstack([_project_one(projector, t, recency_weighted=True) for t in text_list])
        if isinstance(out, tuple):
            out = out[0]
        return _l2_normalize_batch(np.asarray(out, dtype=np.float32))
    return np.vstack([_project_one(projector, t, recency_weighted=use_rw) for t in text_list])


def _normalize_label_str(label: Any) -> str:
    if isinstance(label, (bool, np.bool_)):
        return "True" if bool(label) else "False"
    return str(label)


def _canonical_explain_field(schema: Any, field_name: Optional[str]) -> str:
    explainable = [
        name for name, f_def in schema.fields.items() if isinstance(f_def, (ChoiceField, BooleanField))
    ]
    if field_name is None:
        if not explainable:
            raise ValueError("Schema contains no ChoiceField or BooleanField to explain")
        return explainable[0]
    if field_name not in schema.fields:
        raise ValueError(f"Unknown schema field {field_name!r}")
    if not isinstance(schema.fields[field_name], (ChoiceField, BooleanField)):
        raise ValueError(f"FastExplainer supports ChoiceField and BooleanField, got {type(schema.fields[field_name]).__name__}")
    return field_name


def _extract_calibrated_head_params(
    model: Any,
    fname: str,
    *,
    calibrator: Any = None,
    temperature_scale: float = 4.0,
    recency_weighted: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...], float, float, Dict[str, Any]]:
    """Extract calibrated effective weights, biases, scale, and threshold matching engine decide()."""
    if _has_custom_validate_decision(getattr(model, "schema", None)):
        raise ValueError("FastExplainer does not support custom DecisionSchema.validate_decision")
    f_def = model.schema.fields[fname]
    field_heads = getattr(model, "_field_heads", None)
    compiled_heads = getattr(model, "heads", None)
    field_head = field_heads.get(fname) if isinstance(field_heads, Mapping) else None
    compiled_head = compiled_heads.get(fname) if isinstance(compiled_heads, Mapping) else None
    head = field_head if field_head is not None else compiled_head
    if head is None:
        raise KeyError(f"Unknown field head {fname!r} on model")

    for candidate_head in (field_head, compiled_head):
        if candidate_head is None:
            continue
        if (
            bool(getattr(candidate_head, "hierarchical", False))
            or getattr(candidate_head, "coarse_weights", None) is not None
        ):
            raise ValueError(
                f"FastExplainer cannot faithfully represent hierarchical head for field {fname!r}"
            )
        if getattr(candidate_head, "_exemplar_bank", None) is not None:
            raise ValueError(
                f"FastExplainer cannot faithfully represent nonlinear exemplar-bank head for field {fname!r}"
            )

    base_scale = float(temperature_scale)
    if base_scale <= 0.0:
        raise ValueError(f"temperature_scale must be positive, got {temperature_scale}")

    if calibrator is not None:
        raw_temp = float(getattr(calibrator, "temperature", 1.0))
        adj_raw = getattr(calibrator, "logit_adjustments", None)
    else:
        src_head = compiled_head if compiled_head is not None else head
        raw_temp = float(getattr(src_head, "temperature", 1.0))
        adj_raw = getattr(src_head, "logit_adjustments", None)

    temp = max(1e-4, raw_temp)
    eff_scale = base_scale / temp

    w = np.asarray(head.weights, dtype=np.float64)
    b = np.asarray(head.biases, dtype=np.float64).reshape(-1).copy()
    is_boolean = isinstance(f_def, BooleanField)
    is_zero_head = bool(not np.any(w) and not np.any(b))
    raw_w_f32 = (
        np.ascontiguousarray(head.weights, dtype=np.float32)[:1]
        if is_boolean
        else np.ascontiguousarray(head.weights, dtype=np.float32)
    )
    raw_b_f32 = (
        np.ascontiguousarray(head.biases, dtype=np.float32).reshape(-1)[:1]
        if is_boolean
        else np.ascontiguousarray(head.biases, dtype=np.float32).reshape(-1)
    )
    cal_adj_2: Optional[np.ndarray] = None

    if adj_raw is not None:
        adj = np.asarray(adj_raw, dtype=np.float64).reshape(-1)
        if is_boolean or w.shape[0] == 1:
            if adj.size == 2:
                delta_adj = float(adj[1] - adj[0])
                cal_adj_2 = adj.astype(np.float64).copy()
            elif adj.size == 1:
                delta_adj = float(adj[0])
                cal_adj_2 = np.array([-0.5 * float(adj[0]), 0.5 * float(adj[0])], dtype=np.float64)
            else:
                raise ValueError(
                    f"Unsupported logit_adjustments length {adj.size} for BooleanField {fname!r}"
                )
            b[0] += delta_adj / base_scale
            adj_eff = np.array([delta_adj / temp], dtype=np.float64)
        elif adj.size == w.shape[0]:
            b += adj / base_scale
            adj_eff = (adj / temp).astype(np.float64)
        elif adj.size > 0:
            raise ValueError(
                f"logit_adjustments length {adj.size} != head outputs {w.shape[0]} for field {fname!r}"
            )
        else:
            adj_eff = np.zeros(1 if is_boolean else w.shape[0], dtype=np.float64)
    else:
        adj_eff = np.zeros(1 if is_boolean else w.shape[0], dtype=np.float64)

    default_rec = bool(getattr(model, "recency_weighted", getattr(model.projector, "recency_weighted", False)))
    eff_recency = default_rec if recency_weighted is None else bool(recency_weighted)
    learned_overrides: Dict[str, Any] = {}
    if hasattr(model, "_build_override_eval"):
        src_ov = getattr(model, "_learned_prompt_overrides", None)
        src_rec = getattr(model, "_learned_prompt_override_recency", {})
        if isinstance(src_ov, Mapping):
            for p_k, f_map in src_ov.items():
                if isinstance(p_k, str) and isinstance(f_map, Mapping) and fname in f_map:
                    rec_flag = src_rec.get(p_k, default_rec) if isinstance(src_rec, Mapping) else default_rec
                    if bool(rec_flag) == eff_recency:
                        learned_overrides[p_k] = f_map[fname]

    opts = ("False", "True") if is_boolean else tuple(f_def.options)
    thr = float(getattr(f_def, "threshold", 0.5)) if is_boolean else 0.5
    meta = {
        "is_boolean": is_boolean,
        "is_zero_head": is_zero_head,
        "adj_eff": adj_eff,
        "learned_overrides": learned_overrides,
        "recency_weighted": eff_recency,
        "raw_w_f32": raw_w_f32,
        "raw_b_f32": raw_b_f32,
        "base_scale": base_scale,
        "cal_temp": temp,
        "cal_adj_2": cal_adj_2,
    }
    return w, b, opts, thr, eff_scale, meta


@dataclass(frozen=True, slots=True)
class SpanAttribution:
    """Single word/span attribution record."""

    index: int
    start: int
    end: int
    token: str
    target_logit_attr: float
    margin_attr: float


@dataclass(frozen=True, slots=True)
class ExplanationResult:
    """Full explanation payload for a single System 1 decision."""

    predicted_label: str
    runner_up_label: str
    predicted_logit: float
    runner_up_logit: float
    margin: float
    baseline_target_logit: float
    baseline_margin: float
    additivity_error_target: float
    additivity_error_margin: float
    spans: Tuple[SpanAttribution, ...]
    nearest_exemplar_idx: int
    nearest_exemplar_label: str
    nearest_exemplar_text: str
    nearest_exemplar_similarity: float
    contrast_exemplar_idx: int
    contrast_exemplar_label: str
    contrast_exemplar_text: str
    contrast_exemplar_similarity: float
    counterfactual_found: bool
    counterfactual_k: int
    counterfactual_tokens: Tuple[str, ...]
    counterfactual_new_label: str
    counterfactual_new_margin: float
    latency_us: float


class FastExplainer:
    """Exact additive, exemplar witness, and counterfactual explainer for System 1 heads."""

    def __init__(
        self,
        *,
        projector: Any,
        weights: np.ndarray,
        biases: np.ndarray,
        options: Optional[Sequence[str]] = None,
        train_texts: Optional[Sequence[str]] = None,
        train_labels: Optional[Sequence[Any]] = None,
        temperature_scale: float = 4.0,
        threshold: float = 0.5,
        recency_weighted: Optional[bool] = None,
        is_boolean: Optional[bool] = None,
        learned_overrides: Optional[Mapping[str, Any]] = None,
        is_zero_head: bool = False,
        adj_eff: Optional[np.ndarray] = None,
        _bool_raw_meta: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.projector = projector
        self.scale = float(temperature_scale)
        self.threshold = float(threshold)
        w_in = np.asarray(weights)
        inferred_bool = w_in.ndim == 2 and w_in.shape[0] == 1 and (options is None or len(options) in (0, 2))
        self._is_boolean = bool(inferred_bool if is_boolean is None else is_boolean)
        self._bool_logit_shift = _boolean_threshold_logit_shift(self.threshold)
        self.recency_weighted = bool(
            getattr(projector, "recency_weighted", False) if recency_weighted is None else recency_weighted
        )
        self._learned_overrides: Dict[str, Any] = (
            {" ".join(str(k).strip().casefold().split()): v for k, v in learned_overrides.items() if str(k).strip()}
            if learned_overrides
            else {}
        )
        self._is_zero_head = bool(is_zero_head)

        w_norm, b_norm, resolved_options = _normalize_head_weights_and_options(
            weights,
            biases,
            options,
            threshold=threshold,
            temperature_scale=self.scale,
        )
        self.options = resolved_options
        self.label_to_idx = {lbl: i for i, lbl in enumerate(self.options)}
        self.num_classes = len(self.options)
        self.W_eff = np.ascontiguousarray(w_norm * self.scale, dtype=np.float64)
        self.b_eff = np.ascontiguousarray(b_norm * self.scale, dtype=np.float64)
        if adj_eff is not None:
            self._adj_eff = np.asarray(adj_eff, dtype=np.float64).reshape(-1)
        else:
            self._adj_eff = np.zeros(1 if self._is_boolean else self.num_classes, dtype=np.float64)

        self._bool_has_scalar_raw = False
        self._bool_raw_w_f32 = np.empty((0, 0), dtype=np.float32)
        self._bool_raw_b_f32 = np.empty(0, dtype=np.float32)
        self._bool_raw_base_scale = 4.0
        self._bool_cal_temp = 1.0
        self._bool_cal_adj_2: Optional[np.ndarray] = None
        if self._is_boolean:
            if _bool_raw_meta is not None:
                self._bool_has_scalar_raw = True
                self._bool_raw_w_f32 = np.ascontiguousarray(_bool_raw_meta["raw_w_f32"], dtype=np.float32)[:1]
                self._bool_raw_b_f32 = np.ascontiguousarray(
                    _bool_raw_meta["raw_b_f32"], dtype=np.float32
                ).reshape(-1)[:1]
                self._bool_raw_base_scale = float(_bool_raw_meta["base_scale"])
                self._bool_cal_temp = float(_bool_raw_meta["cal_temp"])
                raw_c_adj = _bool_raw_meta.get("cal_adj_2")
                self._bool_cal_adj_2 = (
                    None if raw_c_adj is None else np.ascontiguousarray(raw_c_adj, dtype=np.float64).reshape(2)
                )
            elif w_in.ndim == 2 and w_in.shape[0] == 1:
                self._bool_has_scalar_raw = True
                self._bool_raw_w_f32 = np.ascontiguousarray(w_in[:1], dtype=np.float32)
                b_flat = np.asarray(biases, dtype=np.float32).reshape(-1)
                self._bool_raw_b_f32 = np.ascontiguousarray(
                    b_flat[:1] if b_flat.size >= 1 else np.zeros(1, dtype=np.float32),
                    dtype=np.float32,
                )
                self._bool_raw_base_scale = float(temperature_scale)
                self._bool_cal_temp = 1.0
                self._bool_cal_adj_2 = (
                    np.array(
                        [-0.5 * float(self._adj_eff[0]), 0.5 * float(self._adj_eff[0])],
                        dtype=np.float64,
                    )
                    if (self._adj_eff.size == 1 and float(self._adj_eff[0]) != 0.0)
                    else None
                )

        if train_texts and train_labels:
            if len(train_texts) != len(train_labels):
                raise ValueError("train_texts and train_labels must have equal length")
            self.train_texts = tuple(str(t) for t in train_texts)
            self.train_labels = tuple(_normalize_label_str(lbl) for lbl in train_labels)
        else:
            self.train_texts = tuple(self.options)
            self.train_labels = tuple(self.options)

        unknown = [lbl for lbl in self.train_labels if lbl not in self.label_to_idx]
        if unknown:
            raise ValueError(f"Unknown exemplar label(s) {unknown!r} not in options {self.options!r}")
        self.train_label_idx = np.array([self.label_to_idx[lbl] for lbl in self.train_labels], dtype=np.int32)

        has_hybrid_fast = (
            not self.recency_weighted
            and isinstance(projector, HybridProjector)
            and projector._tfidf_projector is not None
            and not (
                projector._fitted
                and projector._subspace_components is not None
                and float(projector._subspace_weight) > 0.0
            )
        )
        if has_hybrid_fast:
            self._mode = "hybrid"
        elif not self.recency_weighted and isinstance(projector, TfidfProjector):
            self._mode = "tfidf"
        else:
            self._mode = "generic"
        self.is_hybrid = self._mode == "hybrid"

        X_train = _project_texts(projector, self.train_texts, recency_weighted=self.recency_weighted)
        self.X_train_full = np.ascontiguousarray(X_train, dtype=np.float32)
        self._cache_lock = threading.Lock()

        if self._mode == "hybrid":
            hp = projector
            self.tfidf = hp._tfidf_projector
            self.dense_engine = hp._dense_engine
            self.sd = int(hp.sparse_dim)
            self.dd = int(hp.dense_dim)
            self.spd = int(hp._direct_spectral_dim)
            self.base_dim = self.sd + self.dd + self.spd

            self.w_sparse = math.sqrt(float(hp.alpha))
            if self.spd > 0:
                self.w_dense = math.sqrt((1.0 - float(hp.alpha)) * max(0.0, 1.0 - float(hp.spectral_blend)))
                self.w_spec = math.sqrt((1.0 - float(hp.alpha)) * max(0.0, float(hp.spectral_blend)))
            else:
                self.w_dense = math.sqrt(1.0 - float(hp.alpha))
                self.w_spec = 0.0

            self.has_rbf = hp._rbf_landmarks is not None and float(hp._rbf_weight) > 0.0
            if self.has_rbf:
                self.wb = math.sqrt(1.0 - float(hp._rbf_weight))
                self.wr = math.sqrt(float(hp._rbf_weight))
                self.rbf_gamma = float(hp._rbf_gamma)
                self.landmarks = np.asarray(hp._rbf_landmarks, dtype=np.float64)
                self.num_landmarks = self.landmarks.shape[0]
                self.L_sp_T = np.ascontiguousarray(self.landmarks[:, : self.sd].T)
                self.L_dn = np.ascontiguousarray(self.landmarks[:, self.sd : self.sd + self.dd])
                self.L_sc = np.ascontiguousarray(self.landmarks[:, self.sd + self.dd :])
                self.W_base = self.W_eff[:, : self.base_dim] * self.wb
                self.W_rbf = self.W_eff[:, self.base_dim :] * self.wr
                u0 = np.full(self.num_landmarks, 1.0 / math.sqrt(self.num_landmarks), dtype=np.float64)
                self.b_rbf0 = self.W_rbf @ u0
            else:
                self.wb = 1.0
                self.wr = 0.0
                self.rbf_gamma = 0.0
                self.landmarks = None
                self.num_landmarks = 0
                self.L_sp_T = None
                self.L_dn = None
                self.L_sc = None
                self.W_base = self.W_eff
                self.W_rbf = None
                self.b_rbf0 = np.zeros(self.num_classes, dtype=np.float64)

            self.W_sp = np.ascontiguousarray(self.W_base[:, : self.sd])
            self.W_sp_T = np.ascontiguousarray(self.W_sp.T)
            self.W_dn = np.ascontiguousarray(self.W_base[:, self.sd : self.sd + self.dd])
            self.W_sc = np.ascontiguousarray(self.W_base[:, self.sd + self.dd :])

            table64 = np.asarray(self.dense_engine._table, dtype=np.float64)
            self.W_bucket_dn = np.ascontiguousarray(self.W_dn @ table64.T)
            self.table_f32 = np.ascontiguousarray(self.dense_engine._table, dtype=np.float32)

            if self.spd > 0 and hp._spectral_vectors is not None and hp._spectral_vocab:
                spec_vecs = np.asarray(hp._spectral_vectors, dtype=np.float64)
                self.spec_vocab = hp._spectral_vocab
                self.spec_weights = np.asarray(hp._spectral_weights, dtype=np.float64)
                self.spec_vecs_f32 = np.ascontiguousarray(hp._spectral_vectors, dtype=np.float32)
                self.W_term_sc = np.ascontiguousarray(self.W_sc @ spec_vecs.T)
            else:
                self.spec_vocab = {}
                self.spec_weights = np.empty(0, dtype=np.float64)
                self.spec_vecs_f32 = np.empty((0, 0), dtype=np.float32)
                self.W_term_sc = np.empty((self.num_classes, 0), dtype=np.float64)

            if self.has_rbf:
                self.L_bucket_dn = np.ascontiguousarray(self.L_dn @ table64.T)
                self.L_term_sc = (
                    np.ascontiguousarray(self.L_sc @ self.spec_vecs_f32.astype(np.float64).T)
                    if self.spec_vecs_f32.size
                    else np.empty((self.num_landmarks, 0), dtype=np.float64)
                )
            else:
                self.L_bucket_dn = None
                self.L_term_sc = None

            self.X_train_sp_T = np.ascontiguousarray(self.X_train_full[:, : self.sd].T)
            self.X_train_rest = np.ascontiguousarray(self.X_train_full[:, self.sd :])

            self._hash_token = self.dense_engine._hash_token
            self._multiword_by_space3 = self.dense_engine._multiword_by_space3
            self._bi_hash_cache: Dict[Tuple[str, str], int] = {}
            self._word_dn_cache: Dict[
                str, Tuple[int, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
            ] = {}
            for w in [word for word in self.tfidf._word_vocab if " " not in word and ":" not in word]:
                self._get_word_dn_entry(w)
        elif self._mode == "tfidf":
            self.tfidf = projector
            self.sd = int(projector.dimension)
            self.has_rbf = False
            self.b_rbf0 = np.zeros(self.num_classes, dtype=np.float64)
            self.W_sp = np.ascontiguousarray(self.W_eff)
            self.W_sp_T = np.ascontiguousarray(self.W_sp.T)
            self.X_train_sp_T = np.ascontiguousarray(self.X_train_full.T)
            self.X_train_rest = None
        else:
            self.tfidf = None
            self.sd = int(self.W_eff.shape[1])
            self.has_rbf = False
            self.b_rbf0 = np.zeros(self.num_classes, dtype=np.float64)

        self._class_masks = [self.train_label_idx == c for c in range(self.num_classes)]
        self._other_masks = [~m for m in self._class_masks]
        self._has_same = [bool(np.any(m)) for m in self._class_masks]
        self._has_other = [bool(np.any(m)) for m in self._other_masks]

        self._tls = threading.local()
        if self._mode in ("tfidf", "hybrid"):
            self._idf_f64 = np.asarray(self.tfidf._idf_f64, dtype=np.float64)
            self._scale_f64 = np.asarray(self.tfidf._scale_f64, dtype=np.float64)
            self._base_weight_f64 = self._idf_f64 * self._scale_f64
            self._char_arr_cache: Dict[
                str, Tuple[np.ndarray, np.ndarray, bool, Tuple[int, ...], Tuple[Tuple[int, int], ...]]
            ] = {}
            for w in self.tfidf._word_vocab:
                if " " not in w and ":" not in w and len(w) <= MAX_TERM_LENGTH:
                    self._get_char_arr(w)

    @classmethod
    def from_compiled_model(
        cls,
        model: Any,
        field_name: Optional[str] = None,
        *,
        train_texts: Optional[Sequence[str]] = None,
        train_labels: Optional[Sequence[Any]] = None,
        temperature_scale: float = 4.0,
    ) -> FastExplainer:
        """Build a FastExplainer directly from a CompiledSystemOneModel or SystemOneModel."""
        fname = _canonical_explain_field(model.schema, field_name)
        w, b, opts, thr, eff_scale, meta = _extract_calibrated_head_params(
            model,
            fname,
            calibrator=None,
            temperature_scale=temperature_scale,
        )
        return cls(
            projector=model.projector,
            weights=w,
            biases=b,
            options=opts,
            train_texts=train_texts,
            train_labels=train_labels,
            temperature_scale=eff_scale,
            threshold=thr,
            recency_weighted=meta["recency_weighted"],
            is_boolean=meta["is_boolean"],
            learned_overrides=meta["learned_overrides"],
            is_zero_head=meta["is_zero_head"],
            adj_eff=meta["adj_eff"],
            _bool_raw_meta=meta,
        )

    @classmethod
    def from_engine(
        cls,
        engine: Any,
        field_name: Optional[str] = None,
        *,
        train_texts: Optional[Sequence[str]] = None,
        train_labels: Optional[Sequence[Any]] = None,
        temperature_scale: float = 4.0,
    ) -> FastExplainer:
        """Build a FastExplainer bound to a live SystemOneEngine head."""
        if getattr(engine, "joint_calibrator", None) is not None:
            raise ValueError("FastExplainer does not support engines with active joint_calibrator")
        if _has_custom_validate_decision(getattr(engine, "schema", None)) or _has_custom_validate_decision(
            getattr(getattr(engine, "model", None), "schema", None)
        ):
            raise ValueError("FastExplainer does not support custom DecisionSchema.validate_decision")
        fname = _canonical_explain_field(engine.model.schema, field_name)
        calibrators = getattr(engine, "calibrators", None)
        calibrator = calibrators.get(fname) if isinstance(calibrators, Mapping) else None
        eng_rec = bool(getattr(engine, "recency_weighted", getattr(engine.model, "recency_weighted", False)))
        w, b, opts, thr, eff_scale, meta = _extract_calibrated_head_params(
            engine.model,
            fname,
            calibrator=calibrator,
            temperature_scale=temperature_scale,
            recency_weighted=eng_rec,
        )
        return cls(
            projector=engine.model.projector,
            weights=w,
            biases=b,
            options=opts,
            train_texts=train_texts,
            train_labels=train_labels,
            temperature_scale=eff_scale,
            threshold=thr,
            recency_weighted=meta["recency_weighted"],
            is_boolean=meta["is_boolean"],
            learned_overrides=meta["learned_overrides"],
            is_zero_head=meta["is_zero_head"],
            adj_eff=meta["adj_eff"],
            _bool_raw_meta=meta,
        )

    def _get_char_arr(
        self, w: str
    ) -> Tuple[np.ndarray, np.ndarray, bool, Tuple[int, ...], Tuple[Tuple[int, int], ...]]:
        entry = self._char_arr_cache.get(w)
        if entry is None:
            hits = self.tfidf._word_char_hits(w)
            if not hits:
                computed = (np.empty(0, dtype=np.intp), np.empty(0, dtype=np.int32), False, (), ())
            else:
                u_idx, u_cnt = np.unique(np.asarray(hits, dtype=np.intp), return_counts=True)
                has_mult = bool(np.any(u_cnt > 1))
                u_idx_tup = tuple(int(x) for x in u_idx.tolist())
                u_pair_tup = (
                    tuple((int(i), int(c)) for i, c in zip(u_idx_tup, u_cnt.tolist())) if has_mult else ()
                )
                computed = (u_idx, u_cnt.astype(np.int32), has_mult, u_idx_tup, u_pair_tup)
            with self._cache_lock:
                existing = self._char_arr_cache.get(w)
                if existing is not None:
                    return existing
                if len(self._char_arr_cache) < _MAX_EXPLAIN_OOV_CACHE:
                    self._char_arr_cache[w] = computed
                entry = computed
        return entry

    def _get_word_dn_entry(
        self, w: str
    ) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        entry = self._word_dn_cache.get(w)
        if entry is None:
            hash_fn = self._hash_token
            h_word = hash_fn(w)
            if len(w) >= 3:
                sub_idx_list: List[int] = []
                for n in (3, 4):
                    for j in range(len(w) - n + 1):
                        sub_idx_list.append(hash_fn(f"sub:{w[j:j + n]}"))
                sub_arr = np.asarray(sub_idx_list, dtype=np.intp)
                sub_vec = (0.35 * np.sum(self.table_f32[sub_arr], axis=0)).astype(np.float32)
                sub_w = 0.35 * np.sum(self.W_bucket_dn[:, sub_arr], axis=1)
                sub_l = (
                    0.35 * np.sum(self.L_bucket_dn[:, sub_arr], axis=1)
                    if self.has_rbf and self.L_bucket_dn is not None
                    else None
                )
                computed = (h_word, sub_vec, sub_w, sub_l)
            else:
                computed = (h_word, None, None, None)
            with self._cache_lock:
                existing = self._word_dn_cache.get(w)
                if existing is not None:
                    return existing
                if len(self._word_dn_cache) < _MAX_EXPLAIN_OOV_CACHE:
                    self._word_dn_cache[w] = computed
                entry = computed
        return entry

    def _bi_hash(self, a: str, b: str) -> int:
        key = (a, b)
        h = self._bi_hash_cache.get(key)
        if h is None:
            computed = self._hash_token(f"bi:{a}_{b}")
            with self._cache_lock:
                existing = self._bi_hash_cache.get(key)
                if existing is not None:
                    return existing
                if len(self._bi_hash_cache) < _MAX_EXPLAIN_OOV_CACHE:
                    self._bi_hash_cache[key] = computed
                h = computed
        return h

    def _tokenize_spans(self, text: str) -> List[Tuple[int, int, str, str]]:
        """Extract surface word spans `(start, end, surface, clean_lower)`."""
        clipped = text[:MAX_TEXT_LENGTH]
        spans: List[Tuple[int, int, str, str]] = []
        for m in _CONTRACTION_SPAN_RE.finditer(clipped):
            surf = m.group(0)
            low = surf.lower()
            if "'" in low:
                replaced = _CONTRACTION_RE.sub(lambda mm: mm.group(0).replace("'", ""), low)
                if replaced != low:
                    spans.append((m.start(), m.end(), surf, replaced))
                else:
                    for sub_m in _SPAN_WORD_RE.finditer(surf):
                        spans.append((
                            m.start() + sub_m.start(),
                            m.start() + sub_m.end(),
                            sub_m.group(0),
                            sub_m.group(0).lower(),
                        ))
            else:
                spans.append((m.start(), m.end(), surf, low))
        return spans

    def _extract_tfidf_fast(
        self,
        text: str,
        spans: List[Tuple[int, int, str, str]],
        kept_mask: Optional[List[bool]] = None,
        *,
        need_trace: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, float, Optional[Dict[str, Any]]]:
        tfidf = self.tfidf
        w_get = tfidf._word_vocab.get
        bi_first = tfidf._bigram_first_words
        cnt_buf = getattr(self._tls, "cnt_buf", None)
        if cnt_buf is None or cnt_buf.shape[0] != self.sd:
            cnt_buf = np.zeros(self.sd, dtype=np.int32)
            self._tls.cnt_buf = cnt_buf
        touched: List[int] = []
        cleaned = False
        try:
            if kept_mask is None:
                active_spans = [(j, s[3]) for j, s in enumerate(spans)]
                clipped = text[:MAX_TEXT_LENGTH].lower()
                lowered = _CONTRACTION_RE.sub(lambda m: m.group(0).replace("'", ""), clipped) if "'" in clipped else clipped
            else:
                active_spans = [(j, spans[j][3]) for j in range(len(spans)) if kept_mask[j]]
                lowered = " ".join(w for _, w in active_spans)

            words_2: List[Tuple[int, str]] = [(j, w) for j, w in active_spans if len(w) >= 2]
            w_counts: Dict[str, int] = {}
            w_to_spans: Dict[str, List[int]] = {}
            w_unigrams: List[Tuple[int, List[int]]] = []

            for j, w in words_2:
                w_counts[w] = w_counts.get(w, 0) + 1
                if need_trace:
                    lst = w_to_spans.get(w)
                    if lst is None:
                        w_to_spans[w] = [j]
                    else:
                        lst.append(j)
                idx = w_get(w)
                if idx is not None:
                    if cnt_buf[idx] == 0:
                        touched.append(idx)
                    cnt_buf[idx] += 1

            if need_trace:
                for w, sp_list in w_to_spans.items():
                    idx = w_get(w)
                    if idx is not None:
                        w_unigrams.append((idx, sp_list))

            bi_events: List[Tuple[int, int, int]] = []
            if bi_first and len(words_2) > 1:
                for i in range(len(words_2) - 1):
                    j0, a = words_2[i]
                    if a in bi_first:
                        j1, b = words_2[i + 1]
                        idx = w_get(f"{a} {b}")
                        if idx is not None:
                            if cnt_buf[idx] == 0:
                                touched.append(idx)
                            cnt_buf[idx] += 1
                            if need_trace:
                                bi_events.append((idx, j0, j1))

            negated_words: set[str] = set()
            special_events: List[Tuple[int, Tuple[int, ...], float]] = []
            has_neg = not _NEGATION_TOKENS.isdisjoint(w_counts)
            if has_neg or tfidf._use_skip_bigrams:
                cursor = 0
                n_w2 = len(words_2)
                for clause in _CLAUSE_SPLIT_RE.split(lowered):
                    c_toks = [w for w in _SPAN_WORD_RE.findall(clause) if len(w) >= 2]
                    n_c = len(c_toks)
                    c_spans: List[int] = []
                    if need_trace:
                        for cw in c_toks:
                            while cursor < n_w2 and words_2[cursor][1] != cw:
                                cursor += 1
                            if cursor < n_w2:
                                c_spans.append(words_2[cursor][0])
                                cursor += 1
                            else:
                                c_spans.append(active_spans[-1][0] if active_spans else 0)
                    if tfidf._use_skip_bigrams and n_c >= 3:
                        for i in range(n_c - 2):
                            idx = w_get(f"skip:{c_toks[i]}_{c_toks[i + 2]}")
                            if idx is not None:
                                if cnt_buf[idx] == 0:
                                    touched.append(idx)
                                cnt_buf[idx] += 1
                                if need_trace:
                                    special_events.append((idx, (c_spans[i], c_spans[i + 2]), 0.5))
                    if has_neg:
                        for i, w in enumerate(c_toks):
                            if w in _NEGATION_TOKENS:
                                bound: List[str] = []
                                bound_spans: List[int] = []
                                for j in range(i + 1, min(i + 4, n_c)):
                                    target = c_toks[j]
                                    if target in _NEGATION_TOKENS:
                                        break
                                    if target in _NEGATION_SKIP_MODIFIERS:
                                        continue
                                    bound.append(target)
                                    if need_trace:
                                        bound_spans.append(c_spans[j])
                                    negated_words.add(target)
                                    for tok in (f"neg:{target}", f"{w}_{target}"):
                                        idx = w_get(tok)
                                        if idx is not None:
                                            if cnt_buf[idx] == 0:
                                                touched.append(idx)
                                            cnt_buf[idx] += 1
                                            if need_trace:
                                                special_events.append((idx, (c_spans[i], c_spans[j]), 0.5))
                                    if j > i + 1:
                                        idx = w_get(f"{w} {target}")
                                        if idx is not None:
                                            if cnt_buf[idx] == 0:
                                                touched.append(idx)
                                            cnt_buf[idx] += 1
                                            if need_trace:
                                                special_events.append((idx, (c_spans[i], c_spans[j]), 0.5))
                                    if len(bound) == 2:
                                        idx = w_get(f"{w}_{bound[0]}_{bound[1]}")
                                        if idx is not None:
                                            if cnt_buf[idx] == 0:
                                                touched.append(idx)
                                            cnt_buf[idx] += 1
                                            if need_trace:
                                                special_events.append(
                                                    (idx, (c_spans[i], bound_spans[0], bound_spans[1]), 1.0 / 3.0)
                                                )
                                        break

            unneg_words: List[
                Tuple[str, Optional[int], Tuple[int, ...], Tuple[Tuple[int, int], ...], bool]
            ] = []
            use_stems = tfidf._use_stems
            use_chars = bool(tfidf._char_lens)
            char_chunks: List[np.ndarray] = []
            if use_stems or use_chars:
                for w, wc in w_counts.items():
                    if w not in negated_words and len(w) <= MAX_TERM_LENGTH:
                        s_idx = None
                        if use_stems and len(w) >= 4:
                            st = _tfidf_stem_word(w)
                            if st != w and len(st) >= 3:
                                s_idx = w_get(f"stem:{st}")
                                if s_idx is not None:
                                    if cnt_buf[s_idx] == 0:
                                        touched.append(s_idx)
                                    cnt_buf[s_idx] += wc
                        u_c_tup: Tuple[int, ...] = ()
                        u_pair_tup: Tuple[Tuple[int, int], ...] = ()
                        has_mult = False
                        if use_chars:
                            u_c_idx, u_c_mult, has_mult, u_c_tup, u_pair_tup = self._get_char_arr(w)
                            if u_c_idx.size > 0:
                                char_chunks.append(u_c_idx)
                                if has_mult:
                                    cnt_buf[u_c_idx] += wc * u_c_mult
                                else:
                                    cnt_buf[u_c_idx] += wc
                        if need_trace and (s_idx is not None or u_c_tup):
                            unneg_words.append((w, s_idx, u_c_tup, u_pair_tup, has_mult))

            if not touched and not char_chunks:
                cleaned = True
                trace = {
                    "w_to_spans": w_to_spans,
                    "w_unigrams": w_unigrams,
                    "unneg_words": unneg_words,
                    "bi_events": bi_events,
                    "special_events": special_events,
                    "touched": np.empty(0, dtype=np.intp),
                    "act_occ": np.empty(0, dtype=np.float64),
                } if need_trace else None
                return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64), 0.0, trace

            if char_chunks:
                if touched:
                    char_chunks.append(np.asarray(touched, dtype=np.intp))
                act_idx = np.unique(np.concatenate(char_chunks))
            else:
                act_idx = np.asarray(touched, dtype=np.intp)

            cnts = cnt_buf[act_idx]
            cnt_buf[act_idx] = 0
            cleaned = True
        finally:
            if not cleaned:
                cnt_buf.fill(0)

        if int(np.max(cnts)) <= 1:
            cnts_f = None
            tf_uni = 1.0
            act_raw = self._base_weight_f64[act_idx].copy()
        else:
            cnts_f = cnts.astype(np.float64)
            tf = np.where(cnts == 1, 1.0, 1.0 + np.log(cnts_f))
            tf_uni = tf
            act_raw = self._base_weight_f64[act_idx] * tf

        if negated_words and tfidf._neg_targets_in_vocab:
            active_tgts = [w for w in negated_words if w in tfidf._neg_targets_in_vocab]
            if active_tgts:
                supp_map = tfidf._get_neg_suppress_map()
                suppressed: set[int] = set()
                for tgt in active_tgts:
                    s_idxs = supp_map.get(tgt)
                    if s_idxs:
                        suppressed.update(s_idxs)
                if suppressed:
                    for pos_i, fidx in enumerate(act_idx):
                        if int(fidx) in suppressed:
                            tf_i = 1.0 if cnts_f is None else float(tf_uni[pos_i])
                            act_raw[pos_i] = 0.15 * tf_i * float(self._idf_f64[fidx])

        norm_sp = float(np.linalg.norm(act_raw))
        if need_trace:
            act_occ = act_raw if cnts_f is None else (act_raw / cnts_f)
            trace = {
                "w_to_spans": w_to_spans,
                "w_unigrams": w_unigrams,
                "unneg_words": unneg_words,
                "bi_events": bi_events,
                "special_events": special_events,
                "touched": act_idx,
                "act_occ": act_occ,
            }
        else:
            trace = None
        return act_idx, act_raw, norm_sp, trace

    def _extract_hybrid_dense_and_spec_fast(
        self,
        text: str,
        spans: List[Tuple[int, int, str, str]],
        kept_mask: Optional[List[bool]] = None,
        *,
        need_trace: bool = True,
    ) -> Tuple[np.ndarray, float, np.ndarray, float, Optional[Dict[str, Any]]]:
        n_spans = len(spans)
        if kept_mask is None:
            clean_text = _normalize_text(text)
            active_spans = [(j, spans[j][3]) for j in range(n_spans)]
        else:
            raw_joined = " ".join(spans[j][2] for j in range(n_spans) if kept_mask[j])
            clean_text = _normalize_text(raw_joined) if raw_joined else ""
            active_spans = [(j, spans[j][3]) for j in range(n_spans) if kept_mask[j]]

        if not clean_text:
            return (
                np.zeros(self.dd, dtype=np.float64),
                0.0,
                np.zeros(max(1, self.spd), dtype=np.float64),
                0.0,
                None,
            )

        words, negated_positions = _tokenize_with_negation_scope(clean_text)
        if len(words) == len(active_spans):
            word_spans = [j for j, _ in active_spans]
        else:
            word_spans = []
            cur = 0
            n_act_sp = len(active_spans)
            for w in words:
                while cur < n_act_sp and active_spans[cur][1] != w:
                    cur += 1
                if cur < n_act_sp:
                    word_spans.append(active_spans[cur][0])
                    cur += 1
                else:
                    word_spans.append(active_spans[-1][0] if active_spans else 0)

        dn_buckets: List[int] = []
        dn_weights: List[float] = []
        sub_accum = np.zeros(self.dd, dtype=np.float32)
        has_sub_accum = False

        dn_word_events: List[Tuple[int, int, float, Optional[np.ndarray], Optional[np.ndarray]]] = []
        dn_unary_events: List[Tuple[int, int, float]] = []
        dn_bi_events: List[Tuple[int, int, int, float]] = []
        dn_mw_events: List[Tuple[List[int], int, float]] = []
        hash_token = self._hash_token

        if " " in clean_text:
            seen_k3: set[str] = set()
            pos_sp = clean_text.find(" ")
            clean_word_ranges: Optional[List[Tuple[int, int]]] = None
            while pos_sp != -1:
                if 0 < pos_sp < len(clean_text) - 1:
                    k3 = clean_text[pos_sp - 1 : pos_sp + 2]
                    if k3 not in seen_k3:
                        seen_k3.add(k3)
                        cands = self._multiword_by_space3.get(k3)
                        if cands is not None:
                            for term, first_is_neg, term_hash in cands:
                                pos_match = clean_text.find(term)
                                if pos_match != -1:
                                    wt = 3.5
                                    if not first_is_neg:
                                        clause_prefix = _CLAUSE_SPLIT_RE.split(clean_text[:pos_match])[-1]
                                        prefix_words = [
                                            pw
                                            for pw in _SPAN_WORD_RE.findall(clause_prefix)
                                            if pw not in _NEGATION_SKIP_MODIFIERS
                                        ]
                                        if prefix_words and prefix_words[-1] in _NEGATION_TOKENS:
                                            wt = -2.5
                                    dn_buckets.append(term_hash)
                                    dn_weights.append(wt)
                                    if need_trace:
                                        if clean_word_ranges is None:
                                            clean_word_ranges = [m.span() for m in _SPAN_WORD_RE.finditer(clean_text)]
                                        end_match = pos_match + len(term)
                                        hit_spans = [
                                            word_spans[wi]
                                            for wi, (ws, we) in enumerate(clean_word_ranges)
                                            if we > pos_match and ws < end_match and wi < len(word_spans)
                                        ]
                                        if not hit_spans:
                                            hit_spans = [word_spans[0]] if word_spans else [0]
                                        dn_mw_events.append((hit_spans, term_hash, wt))
                pos_sp = clean_text.find(" ", pos_sp + 1)

        for pos, word in enumerate(words):
            sp_j = word_spans[pos]
            word_weight = math.log1p(len(word)) / math.sqrt(1.0 + pos * 0.05)
            if pos in negated_positions:
                cue = negated_positions[pos]
                h1 = hash_token(f"neg:{word}")
                h2 = hash_token(f"neg:{cue}_{word}")
                h3 = hash_token(word)
                w1, w2, w3 = word_weight * 2.8, word_weight * 2.0, -word_weight * 0.9
                dn_buckets.extend((h1, h2, h3))
                dn_weights.extend((w1, w2, w3))
                if need_trace:
                    dn_unary_events.append((sp_j, h1, w1))
                    dn_unary_events.append((sp_j, h2, w2))
                    dn_unary_events.append((sp_j, h3, w3))
                continue

            if word in _NEGATION_TOKENS:
                h = hash_token(word)
                w = word_weight * 0.6
                dn_buckets.append(h)
                dn_weights.append(w)
                if need_trace:
                    dn_unary_events.append((sp_j, h, w))
                continue

            h_word, sub_vec, sub_w, sub_l = self._get_word_dn_entry(word)
            w_word = word_weight * 1.5
            dn_buckets.append(h_word)
            dn_weights.append(w_word)
            if sub_vec is not None:
                sub_accum += sub_vec
                has_sub_accum = True
            if need_trace:
                dn_word_events.append((sp_j, h_word, w_word, sub_w, sub_l))

        for i in range(len(words) - 1):
            bh = self._bi_hash(words[i], words[i + 1])
            dn_buckets.append(bh)
            dn_weights.append(1.2)
            if need_trace:
                dn_bi_events.append((word_spans[i], word_spans[i + 1], bh, 0.6))

        if dn_buckets or has_sub_accum:
            raw_dn32 = sub_accum
            if dn_buckets:
                b_arr = np.asarray(dn_buckets, dtype=np.intp)
                w_arr = np.asarray(dn_weights, dtype=np.float32)
                raw_dn32 = raw_dn32 + (w_arr @ self.table_f32[b_arr])
            norm_dn = float(np.linalg.norm(raw_dn32))
            if norm_dn > 1e-12:
                u_dn = raw_dn32.astype(np.float64) / norm_dn
            else:
                u_dn = np.zeros(self.dd, dtype=np.float64)
                norm_dn = 0.0
        else:
            u_dn = np.zeros(self.dd, dtype=np.float64)
            norm_dn = 0.0

        spec_counts: Dict[int, int] = {}
        spec_w_map: Dict[int, List[int]] = {}
        spec_bi_events: List[Tuple[int, int, int]] = []
        if self.spd > 0 and self.spec_vocab:
            spec_words, spec_neg = _tokenize_with_negation_scope(clean_text, pattern=r"\b\w\w+\b", max_len=64)
            if spec_words:
                sp_idx_list: List[int] = []
                if need_trace:
                    cur = 0
                    n_w = len(words)
                    for sw in spec_words:
                        while cur < n_w and words[cur] != sw:
                            cur += 1
                        if cur < n_w:
                            sp_idx_list.append(word_spans[cur])
                            cur += 1
                        else:
                            sp_idx_list.append(word_spans[-1] if word_spans else 0)

                sv_get = self.spec_vocab.get
                for pos, sw in enumerate(spec_words):
                    sp_j = sp_idx_list[pos] if need_trace else 0
                    if pos in spec_neg:
                        cue = spec_neg[pos]
                        for tm in (f"neg:{sw}", f"{cue}_{sw}"):
                            tidx = sv_get(tm)
                            if tidx is not None:
                                spec_counts[tidx] = spec_counts.get(tidx, 0) + 1
                                if need_trace:
                                    spec_w_map.setdefault(tidx, []).append(sp_j)
                    elif sw not in _ANCHOR_STOPWORDS:
                        tidx = sv_get(sw)
                        if tidx is not None:
                            spec_counts[tidx] = spec_counts.get(tidx, 0) + 1
                            if need_trace:
                                spec_w_map.setdefault(tidx, []).append(sp_j)
                        st = _emb_stem_word(sw)
                        if st != sw and len(st) >= 3:
                            tidx_s = sv_get(f"stem:{st}")
                            if tidx_s is not None:
                                spec_counts[tidx_s] = spec_counts.get(tidx_s, 0) + 1
                                if need_trace:
                                    spec_w_map.setdefault(tidx_s, []).append(sp_j)
                for i in range(len(spec_words) - 1):
                    if spec_words[i] not in _ANCHOR_STOPWORDS or spec_words[i + 1] not in _ANCHOR_STOPWORDS:
                        tidx = sv_get(f"{spec_words[i]}_{spec_words[i + 1]}")
                        if tidx is not None:
                            spec_counts[tidx] = spec_counts.get(tidx, 0) + 1
                            if need_trace:
                                spec_bi_events.append((tidx, sp_idx_list[i], sp_idx_list[i + 1]))

                if spec_counts:
                    n_sc = len(spec_counts)
                    sc_idx = np.empty(n_sc, dtype=np.intp)
                    sc_w = np.empty(n_sc, dtype=np.float32)
                    sc_occ: Dict[int, float] = {}
                    for pos_s, (tidx, cnt) in enumerate(spec_counts.items()):
                        wt_s = (1.0 + math.log(cnt)) * float(self.spec_weights[tidx])
                        sc_idx[pos_s] = tidx
                        sc_w[pos_s] = wt_s
                        if need_trace:
                            sc_occ[tidx] = wt_s / float(cnt)
                    spec_accum = sc_w @ self.spec_vecs_f32[sc_idx]
                    norm_sc = float(np.linalg.norm(spec_accum))
                    if norm_sc >= 1e-12:
                        u_sc = spec_accum.astype(np.float64) / norm_sc
                    else:
                        u_sc = np.zeros(self.spd, dtype=np.float64)
                        norm_sc = 0.0
                else:
                    sc_occ = {}
                    u_sc = np.zeros(self.spd, dtype=np.float64)
                    norm_sc = 0.0
            else:
                sc_occ = {}
                u_sc = np.zeros(self.spd, dtype=np.float64)
                norm_sc = 0.0
        else:
            sc_occ = {}
            u_sc = np.zeros(max(1, self.spd), dtype=np.float64)
            norm_sc = 0.0

        trace = {
            "dn_word_events": dn_word_events,
            "dn_unary_events": dn_unary_events,
            "dn_bi_events": dn_bi_events,
            "dn_mw_events": dn_mw_events,
            "sc_occ": sc_occ,
            "spec_w_map": spec_w_map,
            "spec_bi_events": spec_bi_events,
        } if need_trace else None
        return u_dn, norm_dn, u_sc, norm_sc, trace

    def _eval_boolean_raw_logit_f32(self, raw_logit: float) -> int:
        if self.threshold <= 0.0:
            return 1
        if self.threshold > 1.0:
            return 0
        scaled_z = (
            float(raw_logit) / 0.25
            if self._bool_raw_base_scale == 4.0
            else float(raw_logit) * self._bool_raw_base_scale
        )
        arr = np.array([-0.5 * scaled_z, 0.5 * scaled_z], dtype=np.float64)
        if self._bool_cal_adj_2 is not None:
            arr = arr + self._bool_cal_adj_2
        scaled = arr / self._bool_cal_temp
        s = scaled - float(np.max(scaled))
        exp = np.exp(s)
        prob_true = float((exp / float(np.sum(exp)))[1])
        return 1 if prob_true >= self.threshold else 0

    def _eval_boolean_embedding_winner(self, emb: np.ndarray) -> int:
        x_f32 = np.asarray(emb, dtype=np.float32).reshape(1, -1)
        raw_l = (x_f32 @ self._bool_raw_w_f32.T + self._bool_raw_b_f32).ravel()[0]
        return self._eval_boolean_raw_logit_f32(float(raw_l))

    def _eval_boolean_text_winner(self, text: str) -> int:
        emb = _project_one(self.projector, text, recency_weighted=self.recency_weighted)
        return self._eval_boolean_embedding_winner(emb)

    def _select_top2_text(self, text: str, logits: np.ndarray) -> Tuple[int, int]:
        if self._is_boolean and self._bool_has_scalar_raw:
            c1 = self._eval_boolean_text_winner(text)
            return c1, (0 if c1 == 1 else 1)
        return self._select_top2(logits)

    def _select_winner_kept(
        self,
        logits: np.ndarray,
        spans: List[Tuple[int, int, str, str]],
        kept_mask: List[bool],
    ) -> int:
        if self._is_boolean and self._bool_has_scalar_raw:
            kept_text = " ".join(spans[j][2] for j in range(len(spans)) if kept_mask[j])
            return self._eval_boolean_text_winner(kept_text)
        return self._select_winner(logits)

    def _select_winner(self, logits: np.ndarray) -> int:
        if self._is_boolean and logits.size == 2:
            if self.threshold <= 0.0:
                return 1
            if self.threshold > 1.0:
                return 0
            s_cal = float(logits[1] - logits[0]) + self._bool_logit_shift
            half_s = 0.5 * s_cal
            s_max = half_s if half_s >= -half_s else -half_s
            e0 = math.exp(-half_s - s_max)
            e1 = math.exp(half_s - s_max)
            prob_true = e1 / (e0 + e1)
            return 1 if prob_true >= self.threshold else 0
        return int(np.argmax(logits))

    def _select_top2(self, logits: np.ndarray) -> Tuple[int, int]:
        if self._is_boolean and logits.size == 2:
            c1 = self._select_winner(logits)
            return c1, (0 if c1 == 1 else 1)
        order = np.argsort(-logits, kind="stable")
        c1 = int(order[0])
        c2 = int(order[1]) if len(order) > 1 else c1
        return c1, c2

    def _fast_logits_for_kept(
        self,
        text: str,
        spans: List[Tuple[int, int, str, str]],
        kept_mask: List[bool],
    ) -> Tuple[int, np.ndarray]:
        if self._mode == "generic":
            kept_text = " ".join(spans[j][2] for j in range(len(spans)) if kept_mask[j])
            x_sub_f32 = _project_one(self.projector, kept_text, recency_weighted=self.recency_weighted)
            x_sub = x_sub_f32.astype(np.float64)
            logits = (self.W_eff @ x_sub) + self.b_eff
            c_win = (
                self._eval_boolean_embedding_winner(x_sub_f32)
                if (self._is_boolean and self._bool_has_scalar_raw)
                else self._select_winner(logits)
            )
            return c_win, logits

        act_idx, act_raw, norm_sp, _ = self._extract_tfidf_fast(text, spans, kept_mask=kept_mask, need_trace=False)
        if not self.is_hybrid:
            if norm_sp > 1e-12:
                v_sp = (act_raw / norm_sp).astype(np.float32).astype(np.float64)
                logits = (v_sp @ self.W_sp_T[act_idx]) + self.b_eff
            else:
                logits = self.b_eff.copy()
            return self._select_winner_kept(logits, spans, kept_mask), logits

        u_dn, norm_dn, u_sc, norm_sc, _ = self._extract_hybrid_dense_and_spec_fast(
            text, spans, kept_mask=kept_mask, need_trace=False
        )
        sp_ok = norm_sp > 1e-12
        dn_ok = norm_dn > 1e-12
        sc_ok = self.spd > 0 and norm_sc > 1e-12
        v_sp = (act_raw / norm_sp).astype(np.float32).astype(np.float64) if sp_ok else act_raw

        b_sq = (
            (self.w_sparse ** 2 if sp_ok else 0.0)
            + (self.w_dense ** 2 if dn_ok else 0.0)
            + (self.w_spec ** 2 if sc_ok else 0.0)
        )
        base_norm = math.sqrt(b_sq)
        if base_norm <= 1e-12:
            base_logits = self.W_base[:, 0] + self.b_eff
            if self.has_rbf and self.L_sp_T is not None and self.W_rbf is not None:
                sims = self.L_sp_T[0]
                phi = np.exp(self.rbf_gamma * (sims - 1.0)).astype(np.float32).astype(np.float64)
                u_rbf = phi / float(np.linalg.norm(phi))
                logits = base_logits + self.W_rbf @ u_rbf
            else:
                logits = base_logits
            return self._select_winner_kept(logits, spans, kept_mask), logits

        g_sp = (self.w_sparse / base_norm) if sp_ok else 0.0
        g_dn = (self.w_dense / base_norm) if dn_ok else 0.0
        g_sc = (self.w_spec / base_norm) if sc_ok else 0.0

        logits = self.b_eff.copy()
        if sp_ok:
            logits += g_sp * (v_sp @ self.W_sp_T[act_idx])
        if dn_ok:
            logits += g_dn * (self.W_dn @ u_dn)
        if sc_ok:
            logits += g_sc * (self.W_sc @ u_sc)

        if self.has_rbf and self.L_sp_T is not None and self.L_dn is not None and self.L_sc is not None and self.W_rbf is not None:
            sims = np.zeros(self.num_landmarks, dtype=np.float64)
            if sp_ok:
                sims += g_sp * (v_sp @ self.L_sp_T[act_idx])
            if dn_ok:
                sims += g_dn * (self.L_dn @ u_dn)
            if sc_ok:
                sims += g_sc * (self.L_sc @ u_sc)
            phi = np.exp(self.rbf_gamma * (sims - 1.0)).astype(np.float32).astype(np.float64)
            pnorm = float(np.linalg.norm(phi))
            if pnorm > 1e-12:
                logits += self.W_rbf @ (phi / pnorm)

        return self._select_winner_kept(logits, spans, kept_mask), logits

    def _accumulate_tfidf_span_attrs(
        self,
        trace: Dict[str, Any],
        act_idx: np.ndarray,
        w_sp_c1_act: np.ndarray,
        w_sp_mg_act: np.ndarray,
        sp_coef: float,
        t_attrs: List[float],
        m_attrs: List[float],
    ) -> None:
        occ_scaled = trace["act_occ"] * sp_coef
        act_list = act_idx.tolist()
        c1_map = dict(zip(act_list, (w_sp_c1_act * occ_scaled).tolist()))
        mg_map = dict(zip(act_list, (w_sp_mg_act * occ_scaled).tolist()))

        for idx, sp_list in trace["w_unigrams"]:
            dt = c1_map[idx]
            dm = mg_map[idx]
            for sp_j in sp_list:
                t_attrs[sp_j] += dt
                m_attrs[sp_j] += dm

        w_to_spans = trace["w_to_spans"]
        for w, s_idx, u_c_tup, u_pair_tup, has_mult in trace["unneg_words"]:
            sp_list = w_to_spans[w]
            dt = c1_map[s_idx] if s_idx is not None else 0.0
            dm = mg_map[s_idx] if s_idx is not None else 0.0
            if u_c_tup:
                if has_mult:
                    for ci, cm in u_pair_tup:
                        dt += cm * c1_map[ci]
                        dm += cm * mg_map[ci]
                else:
                    for ci in u_c_tup:
                        dt += c1_map[ci]
                        dm += mg_map[ci]
            if dt != 0.0 or dm != 0.0:
                for sp_j in sp_list:
                    t_attrs[sp_j] += dt
                    m_attrs[sp_j] += dm

        for fidx, j0, j1 in trace["bi_events"]:
            dt = 0.5 * c1_map[fidx]
            dm = 0.5 * mg_map[fidx]
            t_attrs[j0] += dt
            t_attrs[j1] += dt
            m_attrs[j0] += dm
            m_attrs[j1] += dm

        for fidx, sp_tuple, share in trace["special_events"]:
            dt = share * c1_map[fidx]
            dm = share * mg_map[fidx]
            for sp_j in sp_tuple:
                t_attrs[sp_j] += dt
                m_attrs[sp_j] += dm

    def _explain_generic(
        self, text: str, spans: List[Tuple[int, int, str, str]]
    ) -> Tuple[np.ndarray, int, int, float, List[float], List[float], float, float, np.ndarray]:
        n_spans = len(spans)
        x_f32 = _project_one(self.projector, text, recency_weighted=self.recency_weighted)
        x_full = x_f32.astype(np.float64)
        logits = (self.W_eff @ x_full) + self.b_eff
        if self._is_boolean and self._bool_has_scalar_raw:
            c1 = self._eval_boolean_embedding_winner(x_f32)
            c2 = 0 if c1 == 1 else 1
        else:
            c1, c2 = self._select_top2(logits)
        margin = float(logits[c1] - logits[c2])
        sims_all = (self.X_train_full @ x_full.astype(np.float32)).astype(np.float64)

        if n_spans == 0:
            return logits, c1, c2, margin, [], [], float(logits[c1]), margin, sims_all

        b_target = float(self.b_eff[c1])
        b_margin = float(self.b_eff[c1] - self.b_eff[c2])

        loo_texts = [" ".join(spans[k][2] for k in range(n_spans) if k != j) for j in range(n_spans)]
        X_loo = _project_texts(self.projector, loo_texts, recency_weighted=self.recency_weighted).astype(np.float64)
        logits_loo = (X_loo @ self.W_eff.T) + self.b_eff
        raw_t = logits[c1] - logits_loo[:, c1]
        raw_m = margin - (logits_loo[:, c1] - logits_loo[:, c2])

        res_t = (float(logits[c1]) - b_target) - float(np.sum(raw_t))
        abs_t = float(np.sum(np.abs(raw_t)))
        t_arr = raw_t + (res_t * (np.abs(raw_t) / abs_t) if abs_t > 1e-12 else (res_t / n_spans))

        res_m = (margin - b_margin) - float(np.sum(raw_m))
        abs_m = float(np.sum(np.abs(raw_m)))
        m_arr = raw_m + (res_m * (np.abs(raw_m) / abs_m) if abs_m > 1e-12 else (res_m / n_spans))

        return logits, c1, c2, margin, t_arr.tolist(), m_arr.tolist(), b_target, b_margin, sims_all

    def explain(self, text: str, *, include_counterfactual: bool = True) -> ExplanationResult:
        """Compute exact additive attribution, exemplar witness, and minimal counterfactual removal."""
        t0 = time.perf_counter_ns()
        spans = self._tokenize_spans(text)
        n_spans = len(spans)

        norm_key = " ".join(text.strip().casefold().split()) if text.strip() else ""
        ov_val = self._learned_overrides.get(norm_key) if norm_key else None
        override_hit = False
        if ov_val is not None:
            emb_ov = _project_one(self.projector, text, recency_weighted=self.recency_weighted)
            if (not bool(np.any(emb_ov))) or self._is_zero_head:
                override_hit = True
                if self._is_boolean:
                    z_raw = 12.0 if bool(ov_val) else -12.0
                    s_shifted = (z_raw * self.scale + float(self._adj_eff[0])) - self._bool_logit_shift
                    logits = np.array([-0.5 * s_shifted, 0.5 * s_shifted], dtype=np.float64)
                    if self._bool_has_scalar_raw:
                        c1 = self._eval_boolean_raw_logit_f32(z_raw)
                        c2 = 0 if c1 == 1 else 1
                    else:
                        c1, c2 = self._select_top2(logits)
                else:
                    z_raw = np.full(self.num_classes, -12.0, dtype=np.float64)
                    ov_lbl = _normalize_label_str(ov_val)
                    if ov_lbl in self.label_to_idx:
                        z_raw[self.label_to_idx[ov_lbl]] = 12.0
                    logits = z_raw * self.scale + self._adj_eff
                    c1, c2 = self._select_top2(logits)
                margin = float(logits[c1] - logits[c2])
                t_attrs = [0.0] * n_spans
                m_attrs = [0.0] * n_spans
                b_target = float(logits[c1])
                b_margin = margin
                sims_all = (self.X_train_full @ emb_ov.astype(np.float32)).astype(np.float64)

        if override_hit:
            pass
        elif self._mode == "generic":
            logits, c1, c2, margin, t_attrs, m_attrs, b_target, b_margin, sims_all = self._explain_generic(
                text, spans
            )
        elif self._mode == "tfidf":
            act_idx, act_raw, norm_sp, sp_trace = self._extract_tfidf_fast(text, spans, need_trace=True)
            sp_ok = norm_sp > 1e-12
            v_sp = (act_raw / norm_sp).astype(np.float32).astype(np.float64) if sp_ok else act_raw
            logits = (v_sp @ self.W_sp_T[act_idx]) + self.b_eff if sp_ok else self.b_eff.copy()

            c1, c2 = self._select_top2_text(text, logits)
            margin = float(logits[c1] - logits[c2])

            t_attrs = [0.0] * n_spans
            m_attrs = [0.0] * n_spans
            if sp_ok:
                w_sp_act = self.W_sp_T[act_idx]
                w_c1 = w_sp_act[:, c1]
                w_mg = w_c1 - w_sp_act[:, c2]
                self._accumulate_tfidf_span_attrs(sp_trace, act_idx, w_c1, w_mg, 1.0 / norm_sp, t_attrs, m_attrs)

            b_target = float(self.b_eff[c1])
            b_margin = float(self.b_eff[c1] - self.b_eff[c2])
            sims_all = (
                (v_sp.astype(np.float32) @ self.X_train_sp_T[act_idx]).astype(np.float64)
                if sp_ok
                else np.zeros(len(self.train_texts), dtype=np.float64)
            )
        else:
            act_idx, act_raw, norm_sp, sp_trace = self._extract_tfidf_fast(text, spans, need_trace=True)
            u_dn, norm_dn, u_sc, norm_sc, hd_trace = self._extract_hybrid_dense_and_spec_fast(
                text, spans, need_trace=True
            )
            sp_ok = norm_sp > 1e-12
            dn_ok = norm_dn > 1e-12
            sc_ok = self.spd > 0 and norm_sc > 1e-12
            v_sp = (act_raw / norm_sp).astype(np.float32).astype(np.float64) if sp_ok else act_raw

            b_sq = (
                (self.w_sparse ** 2 if sp_ok else 0.0)
                + (self.w_dense ** 2 if dn_ok else 0.0)
                + (self.w_spec ** 2 if sc_ok else 0.0)
            )
            base_norm = math.sqrt(b_sq)

            if base_norm <= 1e-12:
                logits = self.W_base[:, 0] + self.b_eff
                if self.has_rbf and self.L_sp_T is not None and self.W_rbf is not None:
                    sims = self.L_sp_T[0]
                    phi = np.exp(self.rbf_gamma * (sims - 1.0)).astype(np.float32).astype(np.float64)
                    pnorm = float(np.linalg.norm(phi))
                    if pnorm > 1e-12:
                        logits = logits + self.W_rbf @ (phi / pnorm)
                c1, c2 = self._select_top2_text(text, logits)
                margin = float(logits[c1] - logits[c2])
                t_attrs = [0.0] * n_spans
                m_attrs = [0.0] * n_spans
                b_target = float(logits[c1])
                b_margin = margin
                sims_all = self.X_train_sp_T[0].astype(np.float64)
            else:
                g_sp = (self.w_sparse / base_norm) if sp_ok else 0.0
                g_dn = (self.w_dense / base_norm) if dn_ok else 0.0
                g_sc = (self.w_spec / base_norm) if sc_ok else 0.0

                logits = self.b_eff.copy()
                if sp_ok:
                    logits += g_sp * (v_sp @ self.W_sp_T[act_idx])
                if dn_ok:
                    logits += g_dn * (self.W_dn @ u_dn)
                if sc_ok:
                    logits += g_sc * (self.W_sc @ u_sc)

                if self.has_rbf and self.L_sp_T is not None and self.L_dn is not None and self.L_sc is not None and self.W_rbf is not None:
                    sims = np.zeros(self.num_landmarks, dtype=np.float64)
                    if sp_ok:
                        sims += g_sp * (v_sp @ self.L_sp_T[act_idx])
                    if dn_ok:
                        sims += g_dn * (self.L_dn @ u_dn)
                    if sc_ok:
                        sims += g_sc * (self.L_sc @ u_sc)
                    phi = np.exp(self.rbf_gamma * (sims - 1.0)).astype(np.float32).astype(np.float64)
                    pnorm = float(np.linalg.norm(phi))
                    if pnorm > 1e-12:
                        u_rbf = phi / pnorm
                        logits += self.W_rbf @ u_rbf
                    else:
                        u_rbf = np.zeros(self.num_landmarks, dtype=np.float64)
                else:
                    sims = np.empty(0, dtype=np.float64)
                    u_rbf = np.empty(0, dtype=np.float64)

                c1, c2 = self._select_top2_text(text, logits)
                margin = float(logits[c1] - logits[c2])

                if self.has_rbf and self.W_rbf is not None and self.L_sp_T is not None:
                    # 10-point Gauss-Legendre path integral for RBF + L2 normalization nonlinearity.
                    W_pair_rbf = np.vstack([self.W_rbf[c1], self.W_rbf[c1] - self.W_rbf[c2]])
                    Phi_q = np.exp(self.rbf_gamma * (_IG_T * sims.reshape(1, -1) - 1.0))
                    r_q = np.linalg.norm(Phi_q, axis=1, keepdims=True)
                    U_q = Phi_q / np.maximum(1e-12, r_q)
                    Z_q = U_q @ W_pair_rbf.T
                    scale_q = (_IG_W * (self.rbf_gamma / np.maximum(1e-12, r_q))) * Phi_q
                    beta = np.sum(scale_q, axis=0)[None, :] * W_pair_rbf - Z_q.T @ (scale_q * U_q)

                    w_sp_act = self.W_sp_T[act_idx]
                    l_sp_act = self.L_sp_T[act_idx] if sp_ok else None
                    w_eff_sp_c1 = w_sp_act[:, c1] + (l_sp_act @ beta[0] if l_sp_act is not None else 0.0)
                    w_eff_sp_mg = (w_sp_act[:, c1] - w_sp_act[:, c2]) + (
                        l_sp_act @ beta[1] if l_sp_act is not None else 0.0
                    )
                    b_target = float(self.b_eff[c1] + self.b_rbf0[c1])
                    b_margin = float((self.b_eff[c1] - self.b_eff[c2]) + (self.b_rbf0[c1] - self.b_rbf0[c2]))
                else:
                    beta = None
                    w_sp_act = self.W_sp_T[act_idx]
                    w_eff_sp_c1 = w_sp_act[:, c1]
                    w_eff_sp_mg = w_sp_act[:, c1] - w_sp_act[:, c2]
                    b_target = float(self.b_eff[c1])
                    b_margin = float(self.b_eff[c1] - self.b_eff[c2])

                t_attrs = [0.0] * n_spans
                m_attrs = [0.0] * n_spans

                if sp_ok:
                    self._accumulate_tfidf_span_attrs(
                        sp_trace, act_idx, w_eff_sp_c1, w_eff_sp_mg, g_sp / norm_sp, t_attrs, m_attrs
                    )

                if dn_ok and hd_trace is not None:
                    dn_coef = g_dn / norm_dn
                    wb_c1 = self.W_bucket_dn[c1]
                    wb_mg = wb_c1 - self.W_bucket_dn[c2]
                    if beta is not None and self.L_bucket_dn is not None:
                        b0_dn = beta[0] @ self.L_bucket_dn
                        b1_dn = beta[1] @ self.L_bucket_dn
                        for sp_j, h_word, w_wt, sub_w, sub_l in hd_trace["dn_word_events"]:
                            dt = (float(wb_c1[h_word]) + float(b0_dn[h_word])) * w_wt
                            dm = (float(wb_mg[h_word]) + float(b1_dn[h_word])) * w_wt
                            if sub_w is not None and sub_l is not None:
                                dt += float(sub_w[c1]) + float(beta[0] @ sub_l)
                                dm += float(sub_w[c1] - sub_w[c2]) + float(beta[1] @ sub_l)
                            t_attrs[sp_j] += dt * dn_coef
                            m_attrs[sp_j] += dm * dn_coef
                        for sp_j, b_idx, wt in hd_trace["dn_unary_events"]:
                            c_wt = wt * dn_coef
                            t_attrs[sp_j] += (float(wb_c1[b_idx]) + float(b0_dn[b_idx])) * c_wt
                            m_attrs[sp_j] += (float(wb_mg[b_idx]) + float(b1_dn[b_idx])) * c_wt
                        for sp0, sp1, b_idx, half_wt in hd_trace["dn_bi_events"]:
                            c_wt = half_wt * dn_coef
                            dt = (float(wb_c1[b_idx]) + float(b0_dn[b_idx])) * c_wt
                            dm = (float(wb_mg[b_idx]) + float(b1_dn[b_idx])) * c_wt
                            t_attrs[sp0] += dt
                            t_attrs[sp1] += dt
                            m_attrs[sp0] += dm
                            m_attrs[sp1] += dm
                        for hit_spans, b_idx, wt in hd_trace["dn_mw_events"]:
                            c_wt = (wt / len(hit_spans)) * dn_coef
                            dt = (float(wb_c1[b_idx]) + float(b0_dn[b_idx])) * c_wt
                            dm = (float(wb_mg[b_idx]) + float(b1_dn[b_idx])) * c_wt
                            for sp_j in hit_spans:
                                t_attrs[sp_j] += dt
                                m_attrs[sp_j] += dm
                    else:
                        for sp_j, h_word, w_wt, sub_w, _ in hd_trace["dn_word_events"]:
                            dt = float(wb_c1[h_word]) * w_wt
                            dm = float(wb_mg[h_word]) * w_wt
                            if sub_w is not None:
                                dt += float(sub_w[c1])
                                dm += float(sub_w[c1] - sub_w[c2])
                            t_attrs[sp_j] += dt * dn_coef
                            m_attrs[sp_j] += dm * dn_coef
                        for sp_j, b_idx, wt in hd_trace["dn_unary_events"]:
                            c_wt = wt * dn_coef
                            t_attrs[sp_j] += float(wb_c1[b_idx]) * c_wt
                            m_attrs[sp_j] += float(wb_mg[b_idx]) * c_wt
                        for sp0, sp1, b_idx, half_wt in hd_trace["dn_bi_events"]:
                            c_wt = half_wt * dn_coef
                            dt = float(wb_c1[b_idx]) * c_wt
                            dm = float(wb_mg[b_idx]) * c_wt
                            t_attrs[sp0] += dt
                            t_attrs[sp1] += dt
                            m_attrs[sp0] += dm
                            m_attrs[sp1] += dm
                        for hit_spans, b_idx, wt in hd_trace["dn_mw_events"]:
                            c_wt = (wt / len(hit_spans)) * dn_coef
                            dt = float(wb_c1[b_idx]) * c_wt
                            dm = float(wb_mg[b_idx]) * c_wt
                            for sp_j in hit_spans:
                                t_attrs[sp_j] += dt
                                m_attrs[sp_j] += dm

                if sc_ok and hd_trace is not None:
                    sc_coef = g_sc / norm_sc
                    wt_c1 = self.W_term_sc[c1]
                    wt_mg = wt_c1 - self.W_term_sc[c2]
                    if beta is not None and self.L_term_sc is not None:
                        wt_c1 = wt_c1 + beta[0] @ self.L_term_sc
                        wt_mg = wt_mg + beta[1] @ self.L_term_sc
                    sc_occ = hd_trace["sc_occ"]
                    for tidx, sp_list in hd_trace["spec_w_map"].items():
                        vc = sc_occ[tidx] * sc_coef
                        dt = float(wt_c1[tidx]) * vc
                        dm = float(wt_mg[tidx]) * vc
                        for sp_j in sp_list:
                            t_attrs[sp_j] += dt
                            m_attrs[sp_j] += dm
                    for tidx, sp0, sp1 in hd_trace["spec_bi_events"]:
                        vc = 0.5 * sc_occ[tidx] * sc_coef
                        dt = float(wt_c1[tidx]) * vc
                        dm = float(wt_mg[tidx]) * vc
                        t_attrs[sp0] += dt
                        t_attrs[sp1] += dt
                        m_attrs[sp0] += dm
                        m_attrs[sp1] += dm

                rest_pieces: List[np.ndarray] = [(u_dn * (self.wb * g_dn)).astype(np.float32)]
                if self.spd > 0:
                    rest_pieces.append((u_sc * (self.wb * g_sc)).astype(np.float32))
                if self.has_rbf:
                    rest_pieces.append((u_rbf * self.wr).astype(np.float32))
                v_rest = np.concatenate(rest_pieces) if len(rest_pieces) > 1 else rest_pieces[0]
                sims_all = (self.X_train_rest @ v_rest).astype(np.float64)
                if sp_ok:
                    sp_scaled = (v_sp * (self.wb * g_sp)).astype(np.float32)
                    sims_all += (sp_scaled @ self.X_train_sp_T[act_idx]).astype(np.float64)

        sum_t = sum(t_attrs) + b_target
        sum_m = sum(m_attrs) + b_margin
        err_t = abs(sum_t - float(logits[c1]))
        err_m = abs(sum_m - margin)

        same_mask = self._class_masks[c1]
        if self._has_same[c1]:
            near_idx = int(np.argmax(np.where(same_mask, sims_all, -1e9)))
        else:
            near_idx = int(np.argmax(sims_all))
        other_mask = self._other_masks[c1]
        if self._has_other[c1]:
            ctr_idx = int(np.argmax(np.where(other_mask, sims_all, -1e9)))
        else:
            ctr_idx = near_idx

        cf_found = False
        cf_k = 0
        cf_tokens: Tuple[str, ...] = ()
        cf_new_lbl = self.options[c1]
        cf_new_margin = margin

        if include_counterfactual:
            pos_order = [
                j for j in sorted(range(n_spans), key=lambda idx: m_attrs[idx], reverse=True) if m_attrs[j] > 0.0
            ]
            if pos_order:
                max_try = min(5, len(pos_order))
                cum_attr = 0.0
                est_idx = max_try - 1
                for s in range(max_try):
                    cum_attr += m_attrs[pos_order[s]]
                    if margin - cum_attr <= 0.0:
                        est_idx = s
                        break

                def _probe(k_idx: int) -> Tuple[bool, int, np.ndarray]:
                    kept_k = [True] * n_spans
                    for s in range(k_idx + 1):
                        kept_k[pos_order[s]] = False
                    nc, nl = self._fast_logits_for_kept(text, spans, kept_k)
                    return (nc != c1), nc, nl

                flipped, nc_est, nl_est = _probe(est_idx)
                if flipped:
                    best_idx, best_nc, best_nl = est_idx, nc_est, nl_est
                    while best_idx > 0:
                        prev_flip, p_nc, p_nl = _probe(best_idx - 1)
                        if prev_flip:
                            best_idx, best_nc, best_nl = best_idx - 1, p_nc, p_nl
                        else:
                            break
                    cf_found = True
                    cf_k = best_idx + 1
                    cf_tokens = tuple(spans[pos_order[s]][2] for s in range(cf_k))
                    cf_new_lbl = self.options[best_nc]
                    cf_new_margin = float(best_nl[c1] - best_nl[best_nc])
                else:
                    for step in range(est_idx + 1, max_try):
                        fl, nc_s, nl_s = _probe(step)
                        if fl:
                            cf_found = True
                            cf_k = step + 1
                            cf_tokens = tuple(spans[pos_order[s]][2] for s in range(cf_k))
                            cf_new_lbl = self.options[nc_s]
                            cf_new_margin = float(nl_s[c1] - nl_s[nc_s])
                            break

        elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
        span_objs = tuple(
            SpanAttribution(
                index=j,
                start=spans[j][0],
                end=spans[j][1],
                token=spans[j][2],
                target_logit_attr=t_attrs[j],
                margin_attr=m_attrs[j],
            )
            for j in range(n_spans)
        )
        return ExplanationResult(
            predicted_label=self.options[c1],
            runner_up_label=self.options[c2],
            predicted_logit=float(logits[c1]),
            runner_up_logit=float(logits[c2]),
            margin=margin,
            baseline_target_logit=b_target,
            baseline_margin=b_margin,
            additivity_error_target=err_t,
            additivity_error_margin=err_m,
            spans=span_objs,
            nearest_exemplar_idx=near_idx,
            nearest_exemplar_label=self.train_labels[near_idx],
            nearest_exemplar_text=self.train_texts[near_idx],
            nearest_exemplar_similarity=float(sims_all[near_idx]),
            contrast_exemplar_idx=ctr_idx,
            contrast_exemplar_label=self.train_labels[ctr_idx],
            contrast_exemplar_text=self.train_texts[ctr_idx],
            contrast_exemplar_similarity=float(sims_all[ctr_idx]),
            counterfactual_found=cf_found,
            counterfactual_k=cf_k,
            counterfactual_tokens=cf_tokens,
            counterfactual_new_label=cf_new_lbl,
            counterfactual_new_margin=cf_new_margin,
            latency_us=elapsed_us,
        )


def pack_compact_witness(exp: ExplanationResult, *, top_k: int = 4) -> Dict[str, Any]:
    """Build a compact JSON-serializable witness payload suitable for DecisionWitnessReceipt."""
    top_spans = sorted(exp.spans, key=lambda s: abs(s.margin_attr), reverse=True)[:top_k]
    return {
        "v": 1,
        "p": exp.predicted_label,
        "r": exp.runner_up_label,
        "m": round(exp.margin, 4),
        "b0": round(exp.baseline_margin, 4),
        "ts": [[s.index, s.start, s.end, round(s.margin_attr, 4)] for s in top_spans],
        "ex": [exp.nearest_exemplar_idx, round(exp.nearest_exemplar_similarity, 4)],
        "cx": [exp.contrast_exemplar_idx, round(exp.contrast_exemplar_similarity, 4)],
        "cf": [int(exp.counterfactual_found), exp.counterfactual_k, list(exp.counterfactual_tokens)],
    }


def encoded_witness_sizes(exp: ExplanationResult, *, top_k: int = 4) -> Dict[str, int]:
    """Return byte sizes of compact witness representations (minified JSON, zlib-compressed)."""
    payload = pack_compact_witness(exp, top_k=top_k)
    compact_json = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    compressed = zlib.compress(compact_json, level=6)
    return {
        "compact_json_bytes": len(compact_json),
        "zlib_bytes": len(compressed),
    }


def sign_compact_witness(exp: ExplanationResult, signing_key: Any, *, top_k: int = 4) -> Dict[str, Any]:
    """Sign a compact explanation witness using an Ed25519 private key."""
    from system1.receipt import public_key_bytes, sign_payload

    witness = pack_compact_witness(exp, top_k=top_k)
    signature = sign_payload(witness, signing_key)
    return {
        "witness": witness,
        "signature": signature,
        "signer_public_key": public_key_bytes(signing_key).hex(),
    }


def verify_compact_witness(signed_witness: Mapping[str, Any], public_key: Any) -> bool:
    """Verify a compact explanation witness against an independently trusted Ed25519 public key."""
    from system1.receipt import verify_payload

    if not isinstance(signed_witness, Mapping):
        return False
    witness = signed_witness.get("witness")
    signature = signed_witness.get("signature")
    if not isinstance(witness, Mapping) or not isinstance(signature, str):
        return False
    return bool(verify_payload(witness, signature, public_key))


__all__ = [
    "SpanAttribution",
    "ExplanationResult",
    "FastExplainer",
    "pack_compact_witness",
    "encoded_witness_sizes",
    "sign_compact_witness",
    "verify_compact_witness",
]
