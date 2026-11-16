"""Teach compact local decision skills from labeled examples using NumPy ridge
regression, held-out uncertainty calibration, and portable .s1m serialization.

Schema-derived synthetic examples remain available for bootstrapping demos.
"""

from __future__ import annotations

import gc
import inspect
import io
import json
import math
import mmap
import os
import struct
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import AbstractSet, Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import numpy as np

from system1.core.model import (
    DecisionFieldHead,
    DeterministicSemanticProjector,
    ModelInferenceResult,
    RawFieldEvaluation,
    SystemOneModel,
    _MAX_STRIDED_CLONE_SPAN_BYTES,
    _accepts_recency_weighted,
    _is_active_zero_head,
    _is_native_scalar_projector,
    _needs_2d_layout_preservation,
    _normalize_raw_projection,
    _normalize_raw_projection_batch,
    _reconstruct_2d_layout_array,
    _stable_sigmoid,
    _stable_softmax,
    _validate_2d_layout_cloneable,
)


def _accepts_return_norms(fn: Any) -> bool:
    """Return True when a projector batch callable accepts a `return_norms` keyword argument."""
    if not callable(fn):
        return False
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if "return_norms" in params:
        p = params["return_norms"]
        return p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
from system1.core.schema import (
    BooleanField,
    ChoiceField,
    DecisionField,
    DecisionSchema,
    MultiChoiceField,
    ScoreField,
    _score_abs_diff_clamped,
    _score_from_unit_prob,
    _score_full_span_clamped,
    _score_half_span,
    _score_to_unit_ratio,
)
from system1.calibration import (
    ConformalPredictor,
    DecisionCalibrator,
    _normalize_rows_f32_safe,
    _normalize_vec_f32_safe,
    _safe_centroid_mean_f64,
)
from system1.core.embeddings import compute_strict_loo_weights
from system1.core.telemetry import TelemetryProjector
from system1.core.text import (
    DEFAULT_MAX_FEATURES,
    MAX_PRAG_FEATURES,
    PragmaticCompositeProjector,
    TfidfProjector,
)
from system1.cache import SemanticSystemOneCache

MAGIC_HEADER = b"S1M\x02"
MAGIC_HEADER_V3 = b"S1M\x03\x00\x00\x00\x00"
_V3_HDR_STRUCT = struct.Struct("<8sQ")
S1M_V3_PAGE_SIZE = 16384
MAX_SKILL_FIELDS_V3 = 512
MAX_SKILL_BYTES_V3 = 256 * 1024 * 1024
MAX_SKILL_HEADER_BYTES_V3 = 16 * 1024 * 1024
MAX_SKILL_ARRAY_BYTES_V3 = 256 * 1024 * 1024
MAX_SKILL_RUNTIME_BYTES_V3 = 512 * 1024 * 1024
_ALLOWED_V3_DTYPES = frozenset(("float16", "float32", "float64"))


def _strict_json_parse_float(num_str: str) -> float:
    """Reject JSON float tokens that overflow IEEE-754 float64 to +/-inf."""
    val = float(num_str)
    if not math.isfinite(val):
        raise ValueError(f"Saved skill header contains overflowing float token: {num_str!r}")
    return val


def _strict_header_bool(
    mapping: Mapping[str, Any],
    key: str,
    default: bool,
    context: str,
) -> bool:
    """Validate an optional boolean header field without coercing non-bool types."""
    if key not in mapping:
        return default
    raw = mapping[key]
    if type(raw) is not bool:
        raise ValueError(f"{context} field {key!r} must be a boolean, got {raw!r}")
    return raw


def _call_raw_norm_if_callable(
    projector: Any,
    prompt: str,
    *,
    recency_weighted: bool = False,
) -> Optional[float]:
    """Call `projector.raw_norm` when callable, returning None on missing or invalid values."""
    rn_fn = getattr(projector, "raw_norm", None) if projector is not None else None
    if not callable(rn_fn):
        return None
    try:
        rn_val = (
            rn_fn(prompt, recency_weighted=recency_weighted)
            if _accepts_recency_weighted(rn_fn)
            else rn_fn(prompt)
        )
        if rn_val is not None and not isinstance(rn_val, (bool, np.bool_)):
            return float(rn_val)
    except Exception:
        pass
    return None


def _extract_projector_raw_norm(
    projector: Any,
    prompt: str,
    *,
    recency_weighted: bool = False,
) -> Optional[float]:
    """Retrieve a prompt's raw norm from `raw_norm` or `project_with_norm`."""
    rn_val = _call_raw_norm_if_callable(projector, prompt, recency_weighted=recency_weighted)
    if rn_val is not None:
        return rn_val
    pwn_fn = getattr(projector, "project_with_norm", None) if projector is not None else None
    if callable(pwn_fn):
        try:
            raw_pwn = (
                pwn_fn(prompt, recency_weighted=recency_weighted)
                if _accepts_recency_weighted(pwn_fn)
                else pwn_fn(prompt)
            )
            if (
                isinstance(raw_pwn, tuple)
                and len(raw_pwn) >= 2
                and raw_pwn[1] is not None
                and not isinstance(raw_pwn[1], (bool, np.bool_))
            ):
                return float(raw_pwn[1])
        except Exception:
            pass
    return None


def _validate_compiled_logit_adj(raw_adj: Any, definition: Any, name: str) -> Optional[np.ndarray]:
    """Validate and freeze optional per-class logit_adjustments for a compiled head."""
    if raw_adj is None:
        return None
    if isinstance(definition, BooleanField):
        exp_len = 2
    elif isinstance(definition, ChoiceField):
        exp_len = len(definition.options)
    else:
        raise ValueError(
            f"Saved head {name!r} of type {type(definition).__name__} does not support logit_adj"
        )
    arr = np.asarray(raw_adj)
    if arr.dtype == np.dtype(np.bool_) or arr.dtype.kind not in ("i", "u", "f"):
        raise ValueError(f"Saved head {name!r} logit_adj must have numeric dtype, got {arr.dtype}")
    if arr.ndim != 1 or arr.shape != (exp_len,):
        raise ValueError(
            f"Saved head {name!r} has incompatible logit_adj dimensions {arr.shape} != ({exp_len},)"
        )
    out = np.ascontiguousarray(arr, dtype=np.float64).copy()
    if not bool(np.all(np.isfinite(out))):
        raise ValueError(f"Saved head {name!r} logit_adj contains non-finite values")
    out.setflags(write=False)
    return out


def _strict_header_float(
    mapping: Mapping[str, Any],
    key: str,
    default: float,
    context: str,
    *,
    min_val: Optional[float] = None,
    min_inclusive: bool = True,
    max_val: Optional[float] = None,
    max_inclusive: bool = True,
) -> float:
    """Validate an optional finite numeric header field within inclusive/exclusive bounds."""
    if key not in mapping:
        return default
    raw = mapping[key]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{context} field {key!r} must be a finite number, got {raw!r}")
    try:
        val = float(raw)
    except OverflowError as exc:
        raise ValueError(f"{context} field {key!r} overflows float64: {raw!r}") from exc
    if not math.isfinite(val):
        raise ValueError(f"{context} field {key!r} must be finite, got {raw!r}")
    if min_val is not None and (val < min_val if min_inclusive else val <= min_val):
        raise ValueError(f"{context} field {key!r} out of range: {val!r}")
    if max_val is not None and (val > max_val if max_inclusive else val >= max_val):
        raise ValueError(f"{context} field {key!r} out of range: {val!r}")
    return val

# Fixed budgets for portable, bounded skills. Check before decoding/allocating.
MAX_SKILL_BYTES = 64 * 1024 * 1024
MAX_SKILL_HEADER_BYTES = 1024 * 1024
MAX_SKILL_ARRAY_BYTES = 128 * 1024 * 1024
MAX_SKILL_RUNTIME_BYTES = 256 * 1024 * 1024
MAX_SKILL_DIMENSION = 8192
MAX_SKILL_FIELDS = 64
MAX_SKILL_OPTIONS = 1024
MAX_SKILL_CALIBRATION_SCORES = 1_000_000
MAX_SKILL_COMPRESSION_RATIO = 1024
EXEMPLAR_ARRAYS = ("exemplar_centroids", "exemplar_vectors", "exemplar_mean", "initial_w", "shots_p", "shots_c")
VALID_SEMANTIC_MODES = ("default", "v1_1", "v1_2_fisher", "v1_2_rbf")
HEAD_TENSOR_SUFFIXES = (
    "w",
    "b",
    "P",
    "B",
    "calib_scores",
    "logit_adj",
    "prototypes",
    "proto_scores",
    "coarse_w",
    "coarse_b",
    "fine_w",
    "fine_b",
    *EXEMPLAR_ARRAYS,
)
PROJECTOR_TENSOR_NAMES = frozenset((
    "_projector_tfidf_idf",
    "_projector_prag_base_idf",
    "_projector_prag_idf",
    "_projector_spectral_vectors",
    "_projector_spectral_basis",
    "_projector_subspace_components",
    "_projector_rbf_landmarks",
    "_projector_hybrid_tfidf_idf",
    "_projector_dense_table",
))


def _head_tensor_name(field_name: str, suffix: str, *, namespaced: bool = False) -> str:
    """Return canonical serialized tensor key for (field_name, suffix)."""
    if namespaced:
        return f"0h{len(field_name)}_{field_name}_{suffix}"
    return f"{field_name}_{suffix}"


def _schema_needs_namespaced_head_tensors(schema: DecisionSchema) -> bool:
    """Return True if un-namespaced head tensor keys could collide."""
    seen: set[str] = set(PROJECTOR_TENSOR_NAMES)
    for fname in schema.fields:
        if fname.startswith("0h"):
            return True
        for suffix in HEAD_TENSOR_SUFFIXES:
            legacy_key = f"{fname}_{suffix}"
            if legacy_key in seen:
                return True
            seen.add(legacy_key)
    return False


def _resolve_tensor_namespaced_flag(schema: DecisionSchema, meta: Mapping[str, Any]) -> bool:
    """Validate and resolve the serialized head tensor namespace mode from header metadata."""
    ns_val = meta.get("tensor_namespace")
    if ns_val is None:
        return False
    if type(ns_val) is str and ns_val == "len_v1":
        return True
    raise ValueError(f"Saved skill uses unsupported tensor_namespace: {ns_val!r}")


def _snap_f16(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Quantize a finite weight/matrix buffer through IEEE float16 so compiled predictions match .s1m reload bit-for-bit."""
    if arr is None:
        return None
    a32 = np.asarray(arr, dtype=np.float32)
    if a32.size == 0:
        return a32
    a16 = a32.astype(np.float16)
    return a16.astype(np.float32) if np.all(np.isfinite(a16)) else a32


def _centered_log_odds(probs: np.ndarray, eps: float = 5e-3) -> np.ndarray:
    """Map probability simplex rows into zero-sum centered log-odds."""
    p = np.clip(np.asarray(probs, dtype=np.float64), float(eps), 1.0)
    p = p / np.maximum(1e-12, p.sum(axis=1, keepdims=True))
    lp = np.log(p)
    return lp - lp.mean(axis=1, keepdims=True)


def _soft_fit_temperature_np(logits: np.ndarray, P_T: np.ndarray) -> float:
    """Pure-NumPy grid + golden-section temperature search minimizing KL(P_T || softmax(logits / T))."""
    logits64 = np.asarray(logits, dtype=np.float64)
    P64 = np.clip(np.asarray(P_T, dtype=np.float64), 0.0, 1.0)
    P64 = P64 / np.maximum(1e-12, P64.sum(axis=1, keepdims=True))
    if logits64.shape[0] == 0:
        return 1.0

    def _obj(log_t: float) -> float:
        lt = max(-2.5, min(2.5, float(log_t)))
        T = math.exp(lt)
        s = logits64 / T
        s = s - np.max(s, axis=1, keepdims=True)
        lse = np.log(np.maximum(1e-12, np.sum(np.exp(s), axis=1, keepdims=True)))
        nll = -float(np.mean(np.sum(P64 * (s - lse), axis=1)))
        return nll + 0.01 * (lt * lt)

    grid = np.linspace(-2.5, 2.5, 41)
    vals = [_obj(float(g)) for g in grid]
    best_i = int(np.argmin(vals))
    lo = float(grid[max(0, best_i - 1)])
    hi = float(grid[min(len(grid) - 1, best_i + 1)])
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - phi * (hi - lo)
    d = lo + phi * (hi - lo)
    fc, fd = _obj(c), _obj(d)
    for _ in range(40):
        if fc < fd:
            hi = d
            d, fd = c, fc
            c = hi - phi * (hi - lo)
            fc = _obj(c)
        else:
            lo = c
            c, fc = d, fd
            d = lo + phi * (hi - lo)
            fd = _obj(d)
    best_log_t = 0.5 * (lo + hi)
    return float(math.exp(max(-2.5, min(2.5, best_log_t))))


def _soft_student_sample_weights(
    P_clean: np.ndarray,
    y_lbl_idx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute entropy-weighted + bounded class-balanced sample weights."""
    N, K = P_clean.shape
    entropy = -np.sum(P_clean * np.log(np.clip(P_clean, 1e-12, 1.0)), axis=1) / math.log(max(2, K))
    sw = 0.45 + 0.55 * (1.0 - np.clip(entropy, 0.0, 1.0))
    idx = np.asarray(y_lbl_idx, dtype=np.int64) if y_lbl_idx is not None else np.argmax(P_clean, axis=1)
    counts = np.bincount(idx, minlength=K).astype(np.float64)
    present = counts > 0
    k_eff = max(1, int(np.sum(present)))
    cls_w = np.ones(K, dtype=np.float64)
    cls_w[present] = np.clip(float(N) / (float(k_eff) * counts[present]), 0.35, 3.0)
    sw = sw * cls_w[idx]
    mean_sw = float(np.mean(sw))
    if mean_sw > 1e-12:
        sw = sw / mean_sw
    return sw


def _solve_soft_ridge_student_raw(
    X: np.ndarray,
    P_T: np.ndarray,
    Y_hard_oh: Optional[np.ndarray] = None,
    *,
    beta: float = 0.25,
    reg: float = 0.5,
    eps_smooth: float = 5e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form dual/primal weighted ridge over centered soft log-odds mixed with optional hard targets."""
    X64 = np.asarray(X, dtype=np.float64)
    N, D = X64.shape
    P_raw = np.clip(np.asarray(P_T, dtype=np.float64), 0.0, 1.0)
    K = P_raw.shape[1]
    row_s = P_raw.sum(axis=1, keepdims=True)
    P_norm = np.where(row_s > 1e-12, P_raw / np.maximum(1e-12, row_s), 1.0 / max(1, K))
    P_clean = (1.0 - eps_smooth) * P_norm + eps_smooth / K
    P_clean = P_clean / np.maximum(1e-12, P_clean.sum(axis=1, keepdims=True))

    y_idx = np.argmax(Y_hard_oh, axis=1) if Y_hard_oh is not None else np.argmax(P_clean, axis=1)
    sw = _soft_student_sample_weights(P_clean, y_idx)

    Z_soft = _centered_log_odds(P_clean, eps=eps_smooth)
    if Y_hard_oh is not None and beta > 0.0:
        P_hard = (1.0 - eps_smooth) * np.asarray(Y_hard_oh, dtype=np.float64) + eps_smooth / K
        Y_centered = _centered_log_odds(P_hard, eps=eps_smooth)
        Z = (1.0 - float(beta)) * Z_soft + float(beta) * Y_centered
    else:
        Z = Z_soft

    w_sum = max(1e-12, float(np.sum(sw)))
    mu_X = (sw @ X64) / w_sum
    mu_Z = (sw @ Z) / w_sum
    sw_sqrt = np.sqrt(np.maximum(1e-8, sw))[:, None]
    Xc = (X64 - mu_X) * sw_sqrt
    Zc = (Z - mu_Z) * sw_sqrt

    reg_val = max(1e-4, float(reg))
    if 0 < N < D:
        gram = np.eye(N, dtype=np.float64) + (1.0 / reg_val) * (Xc @ Xc.T)
        try:
            alpha_dual = np.linalg.solve(gram, Zc)
            U = ((1.0 / reg_val) * (Xc.T @ alpha_dual)).T
        except np.linalg.LinAlgError:
            A = Xc.T @ Xc + reg_val * np.eye(D, dtype=np.float64)
            U = np.linalg.lstsq(A, Xc.T @ Zc, rcond=None)[0].T
    else:
        A = Xc.T @ Xc + reg_val * np.eye(D, dtype=np.float64)
        try:
            U = np.linalg.solve(A, Xc.T @ Zc).T
        except np.linalg.LinAlgError:
            U = np.linalg.lstsq(A, Xc.T @ Zc, rcond=None)[0].T
    c = mu_Z - U @ mu_X
    return U, c, Z, sw


def _oof_conservative_soft_conformal(
    p_oof: np.ndarray,
    P_T: np.ndarray,
    *,
    y_true_idx: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Finite-sample weighted conformal LAC quantile + Laplace OOF confidence floor."""
    N, K = p_oof.shape
    if N == 0:
        return 1.0, 0.0
    y_target = (
        np.asarray(y_true_idx, dtype=np.int64)
        if y_true_idx is not None
        else np.argmax(P_T, axis=1)
    )
    scores_arg = 1.0 - p_oof[np.arange(N), y_target]
    w_arg = np.ones(N, dtype=np.float64) if y_true_idx is not None else np.maximum(1e-6, np.max(P_T, axis=1))
    order = np.argsort(scores_arg, kind="mergesort")
    v_s = scores_arg[order]
    w_s = w_arg[order]
    w_sum = float(np.sum(w_s))
    n_eff = (w_sum * w_sum) / max(1e-12, float(np.sum(w_s * w_s)))
    target = (1.0 - float(alpha)) * (n_eff + 1.0) / n_eff
    if target >= 1.0:
        q_lac = float(np.max(v_s)) if N >= 20 else 1.0
    else:
        cdf = np.cumsum(w_s) / w_sum
        idx = int(np.searchsorted(cdf, target, side="left"))
        q_lac = float(v_s[min(idx, len(v_s) - 1)])

    top_pred = np.argmax(p_oof, axis=1)
    top_prob = np.max(p_oof, axis=1)
    is_err = (top_pred != y_target).astype(np.float64)
    singletons = np.sum((1.0 - p_oof) <= q_lac + 1e-12, axis=1) == 1
    cands = sorted(set([0.0] + [float(x) for x in np.round(top_prob, 4)]))
    chosen_floor = 0.0
    found = False
    for fl in cands:
        mask = singletons & (top_prob >= fl)
        n_m = int(np.sum(mask))
        if n_m == 0:
            continue
        if (float(np.sum(is_err[mask])) + 0.5) / (n_m + 1.0) <= float(alpha) + 1e-9:
            chosen_floor = fl
            found = True
            break
    if not found:
        chosen_floor = float(np.max(top_prob)) + 1e-3
    elif N <= 80 and np.any(singletons):
        chosen_floor = max(chosen_floor, float(np.quantile(top_prob[singletons], 0.35)))
    return float(q_lac), float(chosen_floor)


def _roundtrip_unit_f16(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Snap a row-wise unit-norm matrix through float16 and re-normalize each non-zero row."""
    if arr is None:
        return None
    a32 = np.asarray(arr, dtype=np.float32)
    if a32.size == 0 or a32.ndim != 2:
        return a32
    a16 = a32.astype(np.float16)
    if not np.all(np.isfinite(a16)):
        return a32
    rt = a16.astype(np.float32)
    norms = np.linalg.norm(rt, axis=1, keepdims=True)
    return (rt / np.where(norms < 1e-12, 1.0, norms)).astype(np.float32)


def _read_skill_bytes(path: Union[str, Path]) -> bytes:
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_SKILL_BYTES + 1)
    if len(data) > MAX_SKILL_BYTES:
        raise ValueError("Saved skill file exceeds size limit")
    return data


def _read_skill_arrays(data: bytes, schema: DecisionSchema, dimension: int, runtime_bytes: int = 0) -> Dict[str, np.ndarray]:
    """Preflight ZIP and NPY headers before NumPy can allocate any array."""
    # Bound central-directory parsing too: ZipFile creates one object per entry.
    end = data.rfind(b"PK\x05\x06", max(0, len(data) - 65557))
    if end < 0 or end + 22 > len(data):
        raise ValueError("Saved skill has no valid NPZ directory")
    _, disk, directory_disk, disk_count, count, directory_size, directory_offset, comment_size = struct.unpack(
        "<4s4H2LH", data[end:end + 22]
    )
def _validate_skill_shapes_and_runtime_delta(
    schema: DecisionSchema,
    dimension: int,
    shapes: Mapping[str, Tuple[int, ...]],
    *,
    heads_meta: Optional[Mapping[str, Any]] = None,
    namespaced: bool = False,
) -> int:
    """Validate saved projector and head array shapes and return extra Python/covariance runtime bytes."""
    runtime_delta = 0

    tfidf_idf_shape = shapes.get("_projector_tfidf_idf")
    if tfidf_idf_shape is not None and tfidf_idf_shape != (dimension,):
        raise ValueError("Saved TF-IDF idf array has invalid shape")
    hyb_idf_shape = shapes.get("_projector_hybrid_tfidf_idf")
    if hyb_idf_shape is not None and (
        len(hyb_idf_shape) != 1 or not (1 <= hyb_idf_shape[0] <= dimension)
    ):
        raise ValueError("Saved hybrid TF-IDF idf array has invalid shape")
    prag_base_idf_shape = shapes.get("_projector_prag_base_idf")
    if prag_base_idf_shape is not None and (
        len(prag_base_idf_shape) != 1 or not (1 <= prag_base_idf_shape[0] <= dimension)
    ):
        raise ValueError("Saved pragmatic base TF-IDF idf array has invalid shape")
    prag_idf_shape = shapes.get("_projector_prag_idf")
    if prag_idf_shape is not None and (
        len(prag_idf_shape) != 1 or not (0 <= prag_idf_shape[0] <= min(dimension, MAX_PRAG_FEATURES))
    ):
        raise ValueError("Saved pragmatic TF-IDF idf array has invalid shape")
    spec_vec_shape = shapes.get("_projector_spectral_vectors")
    if spec_vec_shape is not None and (
        len(spec_vec_shape) != 2
        or not (1 <= spec_vec_shape[0] <= 2048)
        or not (1 <= spec_vec_shape[1] <= dimension)
    ):
        raise ValueError("Saved hybrid projector spectral_vectors have invalid shape")
    spec_basis_shape = shapes.get("_projector_spectral_basis")
    if spec_basis_shape is not None and (
        len(spec_basis_shape) != 2
        or not (1 <= spec_basis_shape[0] <= dimension)
        or not (1 <= spec_basis_shape[1] <= dimension)
    ):
        raise ValueError("Saved hybrid projector spectral_basis has invalid shape")
    sub_comp_shape = shapes.get("_projector_subspace_components")
    if sub_comp_shape is not None and (
        len(sub_comp_shape) != 2
        or not (1 <= sub_comp_shape[0] <= dimension)
        or not (1 <= sub_comp_shape[1] <= dimension)
    ):
        raise ValueError("Saved hybrid projector subspace_components have invalid shape")
    rbf_shape = shapes.get("_projector_rbf_landmarks")
    if rbf_shape is not None and (
        len(rbf_shape) != 2
        or not (1 <= rbf_shape[0] <= MAX_SKILL_OPTIONS)
        or not (1 <= rbf_shape[1] <= dimension)
        or rbf_shape[0] + rbf_shape[1] != dimension
    ):
        raise ValueError("Saved hybrid projector rbf_landmarks have invalid shape")
    dense_tbl_shape = shapes.get("_projector_dense_table")
    if dense_tbl_shape is not None and (
        len(dense_tbl_shape) != 2
        or not (1 <= dense_tbl_shape[0] <= 250_000)
        or not (1 <= dense_tbl_shape[1] <= dimension)
    ):
        raise ValueError("Saved hybrid projector dense_table has invalid shape")

    for name, definition in schema.fields.items():
        def _k(suffix: str, _fn: str = name) -> str:
            return _head_tensor_name(_fn, suffix, namespaced=namespaced)

        widths = (
            (1, 2)
            if isinstance(definition, (BooleanField, ScoreField))
            else (len(definition.options),)
            if hasattr(definition, "options")
            else (1,)
        )
        weight_shape = shapes.get(_k("w"), ())
        if weight_shape not in tuple((width, dimension) for width in widths):
            raise ValueError(f"Saved head {name!r} has incompatible weight dimensions")
        width = weight_shape[0]
        valid_b_shapes = (
            ((1,), (width,))
            if isinstance(definition, (BooleanField, ScoreField))
            else ((width,),)
        )
        b_shape = shapes.get(_k("b"))
        if b_shape not in valid_b_shapes:
            raise ValueError(f"Saved head {name!r} has incompatible b dimensions")
        required_shapes = {
            "P": (dimension + 1, dimension + 1),
            "B": (dimension + 1, width),
        }
        for suffix, shape in required_shapes.items():
            actual = shapes.get(_k(suffix))
            if actual is not None and actual != shape:
                raise ValueError(f"Saved head {name!r} has incompatible {suffix} dimensions")
        logit_adj_shape = shapes.get(_k("logit_adj"))
        if logit_adj_shape is not None:
            if isinstance(definition, BooleanField):
                exp_adj_len = 2
            elif isinstance(definition, ChoiceField):
                exp_adj_len = len(definition.options)
            else:
                raise ValueError(
                    f"Saved head {name!r} of type {type(definition).__name__} does not support logit_adj"
                )
            if logit_adj_shape != (exp_adj_len,):
                raise ValueError(f"Saved head {name!r} has incompatible logit_adj dimensions")
            runtime_delta += 8 * exp_adj_len
        calibration = shapes.get(_k("calib_scores"))
        if calibration is not None and (len(calibration) != 1 or calibration[0] > MAX_SKILL_CALIBRATION_SCORES):
            raise ValueError("Saved skill calibration exceeds limits")
        proto_shape = shapes.get(_k("prototypes"))
        if proto_shape is not None:
            valid_proto_rows = (1, 2) if isinstance(definition, BooleanField) else (width,)
            if len(proto_shape) != 2 or proto_shape[0] not in valid_proto_rows or proto_shape[1] != dimension:
                raise ValueError(f"Saved head {name!r} has incompatible prototypes dimensions")
        proto_scores_shape = shapes.get(_k("proto_scores"))
        if proto_scores_shape is not None and (
            len(proto_scores_shape) != 1 or proto_scores_shape[0] > MAX_SKILL_CALIBRATION_SCORES
        ):
            raise ValueError("Saved skill prototype calibration exceeds limits")
        coarse_w_shape = shapes.get(_k("coarse_w"))
        coarse_b_shape = shapes.get(_k("coarse_b"))
        if (coarse_w_shape is None) != (coarse_b_shape is None):
            raise ValueError(f"Saved head {name!r} has incomplete coarse hierarchical arrays")
        if coarse_w_shape is not None:
            if (
                len(coarse_w_shape) != 2
                or not (1 <= coarse_w_shape[0] <= width)
                or coarse_w_shape[1] != dimension
                or coarse_b_shape != (coarse_w_shape[0],)
            ):
                raise ValueError(f"Saved head {name!r} has incompatible coarse hierarchical dimensions")
        fine_w_shape = shapes.get(_k("fine_w"))
        fine_b_shape = shapes.get(_k("fine_b"))
        if (fine_w_shape is None) != (fine_b_shape is None):
            raise ValueError(f"Saved head {name!r} has incomplete fine hierarchical arrays")
        if fine_w_shape is not None:
            if fine_w_shape != (width, dimension) or fine_b_shape != (width,):
                raise ValueError(f"Saved head {name!r} has incompatible fine hierarchical dimensions")
        if _k("B") in shapes and _k("P") not in shapes:
            runtime_delta += 8 * (dimension + 1) ** 2
        for suffix in ("calib_scores", "proto_scores"):
            runtime_delta += 32 * shapes.get(_k(suffix), (0,))[0]
        if isinstance(heads_meta, Mapping):
            h_info = heads_meta.get(name)
            if isinstance(h_info, Mapping):
                raw_cls_scores = h_info.get("class_calibration_scores")
                if raw_cls_scores is not None:
                    if not isinstance(raw_cls_scores, Mapping):
                        raise ValueError(f"Saved head {name!r} class_calibration_scores must be a mapping")
                    total_cls_scores = 0
                    for c_vals in raw_cls_scores.values():
                        if not isinstance(c_vals, (list, tuple)):
                            raise ValueError(f"Saved head {name!r} class_calibration_scores entry must be a list")
                        total_cls_scores += len(c_vals)
                        if total_cls_scores > MAX_SKILL_CALIBRATION_SCORES:
                            raise ValueError(f"Saved head {name!r} class_calibration_scores exceeds limit")
                    runtime_delta += 56 * len(raw_cls_scores) + 32 * total_cls_scores
        for suffix in EXEMPLAR_ARRAYS:
            shape = shapes.get(_k(suffix))
            if shape is None:
                continue
            valid = (shape == (dimension,)) if suffix == "exemplar_mean" else (
                len(shape) == 2 and shape[1] == dimension
                and (suffix not in ("exemplar_centroids", "initial_w") or shape[0] == width)
            )
            if not isinstance(definition, ChoiceField) or not valid:
                raise ValueError(f"Saved head {name!r} has incompatible exemplar dimensions")
            if suffix in ("shots_p", "shots_c"):
                runtime_delta += 160 * shape[0]
    return runtime_delta


def _validate_tensor_layouts_header(
    meta: Mapping[str, Any],
    shapes: Mapping[str, Tuple[int, ...]],
    *,
    non_f32_2d_names: AbstractSet[str] = frozenset(),
    full_allocation: bool = False,
) -> Tuple[Dict[str, Tuple[int, int, int]], int]:
    """Validate optional serialized 2D stride/alignment metadata and return extra runtime bytes."""
    if "tensor_layouts" not in meta:
        return {}, 0
    raw_layouts = meta["tensor_layouts"]
    if not isinstance(raw_layouts, dict):
        raise ValueError("Saved skill tensor_layouts header must be a mapping")
    validated: Dict[str, Tuple[int, int, int]] = {}
    extra_bytes = 0
    for tname, lspec in raw_layouts.items():
        if not isinstance(tname, str) or tname not in shapes or tname in non_f32_2d_names:
            raise ValueError(f"Saved skill tensor_layouts references unexpected tensor {tname!r}")
        sh = shapes[tname]
        if len(sh) != 2 or sh[0] <= 0 or sh[1] <= 0:
            raise ValueError(f"Saved skill tensor_layouts entry {tname!r} requires non-empty 2D tensor")
        if not isinstance(lspec, list) or len(lspec) != 3 or any(type(v) is not int for v in lspec):
            raise ValueError(f"Saved skill tensor_layouts entry {tname!r} must be [mod64, s0, s1]")
        mod64, s0, s1 = int(lspec[0]), int(lspec[1]), int(lspec[2])
        K, D = int(sh[0]), int(sh[1])
        itemsize = 4
        nbytes_f32 = itemsize * K * D
        if not (0 <= mod64 < 64):
            raise ValueError(f"Saved skill tensor_layouts mod64 {mod64} out of range for {tname!r}")
        if (s0 % itemsize) != 0 or (s1 % itemsize) != 0:
            raise ValueError(f"Saved skill tensor_layouts strides ({s0}, {s1}) incompatible with float32 for {tname!r}")
        if (K == 1 and s0 != 0) or (D == 1 and s1 != 0):
            raise ValueError(f"Saved skill tensor_layouts singleton stride invalid for {tname!r}")
        min_off = min(0, (K - 1) * s0) + min(0, (D - 1) * s1)
        max_off = max(0, (K - 1) * s0) + max(0, (D - 1) * s1)
        span_bytes = (max_off - min_off) + itemsize
        max_allowed_span = min(_MAX_STRIDED_CLONE_SPAN_BYTES, max(1 << 20, 64 * nbytes_f32))
        if span_bytes <= 0 or span_bytes < nbytes_f32 or span_bytes > max_allowed_span:
            raise ValueError(
                f"Saved skill tensor_layouts span {span_bytes} exceeds limit {max_allowed_span} for {tname!r}"
            )
        a0, a1 = abs(s0), abs(s1)
        aliased = (K > 1 and a0 == 0) or (D > 1 and a1 == 0)
        if not aliased and K > 1 and D > 1:
            g = math.gcd(a0, a1)
            aliased = (a1 // g) < K and (a0 // g) < D
        if aliased:
            raise ValueError(f"Saved skill tensor_layouts aliases elements for {tname!r}")
        extra_bytes += (span_bytes + 128) if full_allocation else ((span_bytes - nbytes_f32) + 128)
        validated[tname] = (mod64, s0, s1)
    return validated, extra_bytes


def _read_skill_arrays(
    data: bytes,
    schema: DecisionSchema,
    dimension: int,
    runtime_bytes: int,
    *,
    namespaced: bool = False,
    meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """Validate and deserialize `.s1m` v1/v2 NPZ arrays within strict bounds."""
    if not data.startswith(b"PK\x03\x04"):
        raise ValueError("Saved skill does not contain a valid NPZ array archive")
    eocd = data.rfind(b"PK\x05\x06", max(0, len(data) - 65557))
    if eocd < 0 or eocd + 22 > len(data):
        raise ValueError("Saved skill NPZ archive is missing or malformed")
    disk, directory_disk, entries, disk_count, directory_size, directory_offset, comment_size = struct.unpack_from(
        "<HHHHIIH", data, eocd + 4
    )
    end, count = directory_offset + directory_size, entries
    if (
        directory_offset > eocd
        or entries != disk_count
        or directory_size == 0
        or end != eocd
        or data[directory_offset : directory_offset + 4] != b"PK\x01\x02"
    ):
        raise ValueError("Saved skill NPZ directory is malformed")
    if (disk or directory_disk or disk_count != count or count > MAX_SKILL_FIELDS * (12 + len(EXEMPLAR_ARRAYS)) + 7
            or directory_size > MAX_SKILL_HEADER_BYTES or directory_offset + directory_size != end
            or end + 22 + comment_size != len(data)):
        raise ValueError("Saved skill NPZ directory exceeds limits or is malformed")
    expected = {
        f"{_head_tensor_name(name, suffix, namespaced=namespaced)}.npy"
        for name in schema.fields
        for suffix in HEAD_TENSOR_SUFFIXES
    } | {
        "_projector_spectral_vectors.npy",
        "_projector_subspace_components.npy",
        "_projector_rbf_landmarks.npy",
        "_projector_tfidf_idf.npy",
        "_projector_hybrid_tfidf_idf.npy",
        "_projector_prag_base_idf.npy",
        "_projector_prag_idf.npy",
    }
    shapes: Dict[str, Tuple[int, ...]] = {}
    effective_array_bytes = 0
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        names = [info.filename for info in members]
        if (len(members) != count or len(set(names)) != len(names) or not set(names) <= expected
                or sum(info.file_size for info in members) > MAX_SKILL_ARRAY_BYTES):
            raise ValueError("Saved skill contains unexpected, duplicate or oversized arrays")
        for info in members:
            if (info.flag_bits & 1 or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                    or info.file_size > MAX_SKILL_ARRAY_BYTES
                    or info.file_size > max(1, info.compress_size) * MAX_SKILL_COMPRESSION_RATIO):
                raise ValueError("Saved skill array compression exceeds limits")
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("Unsupported saved skill array format")
                if (dtype.kind != "f" or dtype.itemsize not in (2, 4, 8) or len(shape) not in (1, 2)
                        or any(type(n) is not int or n < 0 for n in shape)):
                    raise ValueError("Saved skill arrays must be bounded float16/float32/float64 vectors or matrices")
                byte_count = math.prod(shape) * dtype.itemsize
                if byte_count > MAX_SKILL_ARRAY_BYTES or stream.tell() + byte_count != info.file_size:
                    raise ValueError("Saved skill array shape exceeds its declared size or budget")
                effective_array_bytes += info.file_size + (byte_count if dtype.itemsize == 2 else 0)
                shapes[info.filename[:-4]] = shape
        if effective_array_bytes > MAX_SKILL_ARRAY_BYTES:
            raise ValueError("Saved skill arrays exceed upcast array budget")
        runtime_bytes += _validate_skill_shapes_and_runtime_delta(
            schema,
            dimension,
            shapes,
            heads_meta=meta.get("heads") if isinstance(meta, Mapping) else None,
            namespaced=namespaced,
        )
        _, layout_extra_bytes = _validate_tensor_layouts_header(meta or {}, shapes)
        runtime_bytes += layout_extra_bytes
        runtime_bytes += 2 * effective_array_bytes
        if runtime_bytes > MAX_SKILL_RUNTIME_BYTES:
            raise ValueError("Saved skill runtime allocation exceeds memory budget")
        # Every member has been checked, including optional covariance arrays.
        with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
            return {name: arrays[name] for name in shapes}


def _projector_config(projector: Any, *, include_arrays: bool = True) -> Dict[str, Any]:
    """Persist the settings that define the feature space, not just its width."""
    from system1.core.embeddings import HybridProjector

    if type(projector) is DeterministicSemanticProjector:
        return {"type": "deterministic", "dimension": projector.dimension,
                "recency_weighted": projector.recency_weighted}
    if type(projector) is TfidfProjector:
        return projector.to_config(include_arrays=include_arrays)
    if isinstance(projector, PragmaticCompositeProjector):
        return projector.to_config(include_arrays=include_arrays)
    if isinstance(projector, HybridProjector):
        return projector.to_config(include_arrays=include_arrays)
    config = {"type": "external", "class": f"{type(projector).__module__}.{type(projector).__qualname__}",
              "dimension": getattr(projector, "dimension", None)}
    if callable(getattr(projector, "projector_digest", None)):
        config["digest"] = projector.projector_digest()
    return config


def _validate_projector_header_strict(projector_config: Any, dimension: int) -> None:
    if not isinstance(projector_config, Mapping) or not isinstance(projector_config.get("type"), str):
        raise ValueError("Saved projector config must be a mapping with a string type")
    kind = projector_config["type"]
    if kind == "deterministic":
        unexpected = set(projector_config.keys()) - {"type", "dimension", "recency_weighted"}
        if unexpected:
            raise ValueError(f"Saved deterministic projector config contains unexpected keys: {sorted(unexpected)}")
        dim_cfg = projector_config.get("dimension")
        if type(dim_cfg) is not int or dim_cfg != dimension:
            raise ValueError("Saved projector dimension does not match the skill")
        _strict_header_bool(
            projector_config, "recency_weighted", False, "Saved deterministic projector"
        )
    elif kind == "external":
        unexpected = set(projector_config.keys()) - {"type", "class", "dimension", "digest"}
        if unexpected:
            raise ValueError(f"Saved external projector config contains unexpected keys: {sorted(unexpected)}")
        cls_name = projector_config.get("class")
        if not isinstance(cls_name, str) or not cls_name.strip():
            raise ValueError("Saved external projector config must define a non-empty class string")
        if "dimension" not in projector_config:
            raise ValueError("Saved external projector config must define dimension")
        ext_dim = projector_config.get("dimension")
        if ext_dim is not None and (type(ext_dim) is not int or ext_dim != dimension):
            raise ValueError("Saved external projector dimension does not match the skill")
        if "digest" in projector_config:
            dig = projector_config["digest"]
            if not isinstance(dig, str) or not dig.strip():
                raise ValueError("Saved external projector digest must be a non-empty string")


@dataclass
class CompiledHeadWeights:
    """Calibrated weight and bias matrices for a single schema field."""

    field_name: str
    field_type: str
    weights: np.ndarray  # shape: (K, D) or (1, D)
    biases: np.ndarray   # shape: (K,) or (1,)
    temperature: float = 1.0
    logit_adjustments: Optional[np.ndarray] = None
    conformal_quantile: float = 0.0
    options: Tuple[str, ...] = ()
    P: Optional[np.ndarray] = None  # Inverted regularized covariance matrix (D+1, D+1)
    B: Optional[np.ndarray] = None  # Cross-covariance matrix (D+1, K)
    margin_threshold: float = 0.20
    forgetting_factor: float = 0.995
    relative_odds_ratio: float = 1.5
    confidence_floor_tau0: float = 0.15
    escalate_on_ambiguity: bool = True
    calibration_scores: Tuple[float, ...] = ()
    score_method: str = "aps"
    mondrian: bool = False
    class_calibration_scores: Dict[str, Tuple[float, ...]] = field(default_factory=dict)
    familiarity_threshold: float = 0.0
    prototypes: Optional[np.ndarray] = None
    prototype_threshold: float = 0.0
    prototype_scores: Tuple[float, ...] = ()
    hierarchical: bool = False
    cluster_assignments: Tuple[int, ...] = ()
    coarse_weights: Optional[np.ndarray] = None
    coarse_biases: Optional[np.ndarray] = None
    fine_weights: Optional[np.ndarray] = None
    fine_biases: Optional[np.ndarray] = None
    exemplar_state: Optional[Dict[str, Any]] = None


class CompiledSystemOneModel:
    """Static compiled System 1 decision model loaded from a .s1m binary.

    Executes single-pass non-autoregressive forward evaluation using pre-distilled
    optimal decision hyperplanes and calibrated conformal non-conformity bounds.
    """

    def __init__(
        self,
        schema: DecisionSchema,
        heads: Dict[str, CompiledHeadWeights],
        dimension: int = 384,
        projector: Optional[Any] = None,
        backend: str = "numpy",
        metadata: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
        cache_threshold: float = 0.98,
        cache_capacity: int = 2048,
        forgetting_factor: float = 0.995,
        recency_weighted: bool = False,
    ) -> None:
        self.schema = schema
        self.heads = heads
        self.dimension = dimension
        self.backend = backend
        self.metadata = metadata or {}
        self.created_at = float(self.metadata.get("created_at", 0.0))
        self.forgetting_factor = float(forgetting_factor)
        self.recency_weighted = bool(recency_weighted)
        if type(projector) is TfidfProjector and self.recency_weighted:
            raise ValueError("TF-IDF does not support recency weighting")
        if isinstance(projector, PragmaticCompositeProjector) and self.recency_weighted:
            raise ValueError("PragmaticCompositeProjector does not support recency weighting")
        self.projector = (
            projector
            if projector is not None
            else DeterministicSemanticProjector(
                dimension=self.dimension,
                recency_weighted=self.recency_weighted,
            )
        )
        self.use_cache = bool(use_cache)
        self.cache = SemanticSystemOneCache(
            capacity=cache_capacity,
            similarity_threshold=cache_threshold,
        )
        self.model_version: int = 1
        self.telemetry_projector = TelemetryProjector(embedding_dim=self.dimension)
        self._lock = threading.RLock()
        self._mmap_obj: Optional[mmap.mmap] = None
        self._mmap_fh: Optional[Any] = None
        self._learned_prompt_overrides: Dict[str, Dict[str, Any]] = {}
        self._learned_prompt_override_recency: Dict[str, bool] = {}
        raw_overrides = self.metadata.get("learned_prompt_overrides")
        raw_override_recency = self.metadata.get("learned_prompt_override_recency")
        if isinstance(raw_overrides, Mapping):
            for p_key, f_map in raw_overrides.items():
                if isinstance(p_key, str) and isinstance(f_map, Mapping):
                    valid_f: Dict[str, Any] = {}
                    for f_k, f_v in f_map.items():
                        if f_k in self.schema.fields:
                            try:
                                valid_f[f_k] = self.schema.fields[f_k].validate_value(f_v)
                            except (TypeError, ValueError):
                                pass
                    if valid_f:
                        self._learned_prompt_overrides[p_key] = valid_f
                        if isinstance(raw_override_recency, Mapping) and p_key in raw_override_recency:
                            self._learned_prompt_override_recency[p_key] = bool(raw_override_recency[p_key])
                        else:
                            self._learned_prompt_override_recency[p_key] = bool(self.recency_weighted)

        # Build live heads for evaluation
        self._field_heads: Dict[str, DecisionFieldHead] = {}
        for name, f_def in self.schema.fields.items():
            head = DecisionFieldHead(
                f_def,
                dimension=self.dimension,
                backend=self.backend,
                projector=self.projector,
                forgetting_factor=self.forgetting_factor,
                initialize_weights=False,
            )
            if name in self.heads:
                ch = self.heads[name]
                ch.logit_adjustments = _validate_compiled_logit_adj(
                    getattr(ch, "logit_adjustments", None),
                    f_def,
                    name,
                )
                head.set_weights(ch.weights, ch.biases)
                if ch.exemplar_state is not None:
                    head.restore_exemplar_state(ch.exemplar_state)
                if not isinstance(f_def, BooleanField) and getattr(ch, "hierarchical", False):
                    if (
                        ch.coarse_weights is None
                        or ch.coarse_biases is None
                        or len(ch.cluster_assignments) != ch.weights.shape[0]
                    ):
                        raise ValueError(f"Compiled head {name!r} has incomplete hierarchical parameters")
                    head.set_hierarchical_weights(
                        cluster_assignments=ch.cluster_assignments,
                        coarse_weights=ch.coarse_weights,
                        coarse_biases=ch.coarse_biases,
                        fine_weights=ch.fine_weights,
                        fine_biases=ch.fine_biases,
                    )
                if ch.P is not None or ch.B is not None:
                    head.init_covariance(P=ch.P, B=ch.B)
            self._field_heads[name] = head

    @staticmethod
    def _close_mmap_handle(mm: Optional[mmap.mmap], fh: Optional[Any]) -> None:
        """Closes file handle immediately and unmaps mmap when no NumPy views remain."""
        try:
            if mm is not None:
                try:
                    mm.close()
                except BufferError:
                    gc.collect()
                    try:
                        mm.close()
                    except BufferError:
                        # External read-only NumPy views still reference mm; GC will unmap it safely.
                        pass
        finally:
            if fh is not None:
                fh.close()

    def close(self) -> None:
        """Releases any memory-mapped buffers held by this compiled model."""
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            mm = getattr(self, "_mmap_obj", None)
            fh = getattr(self, "_mmap_fh", None)
            if mm is None and fh is None:
                return
            self._mmap_obj = None
            self._mmap_fh = None
            if hasattr(self, "_field_heads") and self._field_heads:
                self._field_heads.clear()
            if hasattr(self, "heads") and self.heads:
                self.heads.clear()
            if hasattr(self, "_online_prototype_bank"):
                self._online_prototype_bank.clear()
            self.projector = None
            cache = getattr(self, "cache", None)
            if cache is not None and hasattr(cache, "clear"):
                cache.clear()
            gc.collect()
            self._close_mmap_handle(mm, fh)

    def __enter__(self) -> CompiledSystemOneModel:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _build_override_eval(head: DecisionFieldHead, override_val: Any) -> RawFieldEvaluation:
        """Construct a deterministic high-confidence RawFieldEvaluation for an explicitly taught OOV prompt."""
        f_def = head.field_def
        if isinstance(f_def, ChoiceField):
            opts = f_def.options
            logits = np.full(len(opts), -12.0, dtype=np.float32)
            if override_val in opts:
                logits[opts.index(override_val)] = 12.0
            return head.evaluate_logits(logits)
        if isinstance(f_def, BooleanField):
            z = 12.0 if bool(override_val) else -12.0
            logits = (
                np.array([z], dtype=np.float32)
                if head.weights.shape[0] == 1
                else np.array([z, -z], dtype=np.float32)
            )
            return head.evaluate_logits(logits)
        if isinstance(f_def, MultiChoiceField):
            selected = (
                set(override_val)
                if isinstance(override_val, (list, tuple, set, frozenset))
                else {override_val}
            )
            logits = np.array([12.0 if opt in selected else -12.0 for opt in f_def.options], dtype=np.float32)
            return head.evaluate_logits(logits)
        if isinstance(f_def, ScoreField):
            return head.evaluate_logits(head.format_target_vector(override_val))
        return head.forward(np.zeros(head.dimension, dtype=np.float32))

    @property
    def prototypes(self) -> Dict[str, np.ndarray]:
        """Return mapping of field name to unit-normalized class prototype matrix (K, D)."""
        return {
            name: ch.prototypes
            for name, ch in self.heads.items()
            if getattr(ch, "prototypes", None) is not None
        }

    def compute_prototype_similarity(
        self, field_name: str, embedding: np.ndarray
    ) -> Optional[float]:
        """Compute maximum cosine similarity between an embedding and class prototypes for field_name."""
        ch = self.heads.get(field_name)
        if ch is None or ch.prototypes is None or ch.prototypes.size == 0:
            return None
        emb_raw = np.asarray(embedding)
        emb_flat = emb_raw.ravel()
        if emb_flat.ndim != 1 or ch.prototypes.ndim != 2 or ch.prototypes.shape[1] != emb_flat.shape[0]:
            return None
        emb_unit, _, emb_valid = _normalize_vec_f32_safe(emb_flat)
        if not emb_valid:
            return 0.0
        proto_all, valid_rows = _normalize_rows_f32_safe(ch.prototypes)
        if not np.any(valid_rows):
            return None
        proto_unit = proto_all[valid_rows]
        best_sim = float(np.max(proto_unit @ emb_unit))
        online_bank = getattr(self, "_online_prototype_bank", {}).get(field_name)
        if online_bank:
            for shot in online_bank:
                s_raw = np.asarray(shot).ravel()
                if s_raw.shape[0] == emb_unit.shape[0]:
                    with np.errstate(over="ignore", invalid="ignore"):
                        sim_s = float(np.dot(s_raw.astype(np.float32), emb_unit))
                    if sim_s > best_sim:
                        best_sim = sim_s
        head = self._field_heads.get(field_name)
        online_shots = getattr(head, "_online_shots_p", None) if head is not None else None
        if online_shots:
            for group in online_shots:
                for shot in group:
                    s_raw = np.asarray(shot).ravel()
                    if s_raw.shape[0] == emb_unit.shape[0]:
                        s_unit, _, s_valid = _normalize_vec_f32_safe(s_raw)
                        if s_valid:
                            sim_s = float(np.dot(s_unit, emb_unit))
                            if sim_s > best_sim:
                                best_sim = sim_s
        return best_sim

    def _encode_with_norm(
        self,
        prompt: str,
        telemetry: Optional[Any] = None,
        recency_weighted: Optional[bool] = None,
    ) -> Tuple[np.ndarray, Optional[float]]:
        if not isinstance(prompt, str):
            raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}")
        use_recency = self.recency_weighted if recency_weighted is None else bool(recency_weighted)
        raw_norm: Optional[float] = None
        pwn_fn = getattr(self.projector, "project_with_norm", None)
        proj_fn = getattr(self.projector, "project", None)
        pwn_rw = _accepts_recency_weighted(pwn_fn)
        proj_rw = _accepts_recency_weighted(proj_fn)
        if callable(pwn_fn) and telemetry is None and (pwn_rw or not proj_rw):
            emb, raw_norm = (
                pwn_fn(prompt, recency_weighted=use_recency) if pwn_rw else pwn_fn(prompt)
            )
        elif callable(proj_fn):
            emb = proj_fn(prompt, recency_weighted=use_recency) if proj_rw else proj_fn(prompt)
            if callable(pwn_fn) and telemetry is None:
                _, raw_norm = pwn_fn(prompt)
        else:
            emb = self.projector(prompt)
        emb = _normalize_raw_projection(
            emb,
            self.dimension,
            zero_unit0=not isinstance(self.projector, (TfidfProjector, PragmaticCompositeProjector)),
        )
        if telemetry is not None:
            emb = self.telemetry_projector.fuse(emb, telemetry)
        return emb, raw_norm

    def encode(
        self,
        prompt: str,
        telemetry: Optional[Any] = None,
        recency_weighted: Optional[bool] = None,
    ) -> np.ndarray:
        """Projects input prompt into normalized dense vector, fusing continuous telemetry if present."""
        return self._encode_with_norm(prompt, telemetry=telemetry, recency_weighted=recency_weighted)[0]

    def encode_batch(
        self,
        prompts: Sequence[str],
        telemetry: Optional[Sequence[Any]] = None,
        recency_weighted: Optional[bool] = None,
    ) -> np.ndarray:
        """Batch-projects a sequence of prompts into normalized dense vectors."""
        if len(prompts) == 0:
            return np.zeros((0, self.dimension), dtype=np.float32)
        if telemetry is not None and len(telemetry) == len(prompts):
            return np.stack(
                [
                    self.encode(p, telemetry=telemetry[i], recency_weighted=recency_weighted)
                    for i, p in enumerate(prompts)
                ],
                axis=0,
            ).astype(np.float32)
        return np.stack(
            [self.encode(p, recency_weighted=recency_weighted) for p in prompts],
            axis=0,
        ).astype(np.float32)

    def forward_single(
        self,
        prompt: str,
        telemetry: Optional[Any] = None,
        embedding: Optional[np.ndarray] = None,
        recency_weighted: Optional[bool] = None,
        *,
        use_cache: Optional[bool] = None,
    ) -> ModelInferenceResult:
        """Executes a forward evaluation over the schema."""
        t0 = time.perf_counter()
        use_recency = self.recency_weighted if recency_weighted is None else bool(recency_weighted)
        eff_use_cache = self.use_cache if use_cache is None else bool(use_cache)
        raw_norm: Optional[float] = None
        if embedding is not None:
            emb = np.asarray(embedding, dtype=np.float32).flatten()
        else:
            emb, raw_norm = self._encode_with_norm(prompt, telemetry=telemetry, recency_weighted=use_recency)

        # Check Tier 0 Semantic System 1 Cache
        emb_finite = bool(np.all(np.isfinite(emb)))
        cur_version = getattr(self, "model_version", 1)
        schema_dig = self.schema.schema_digest()
        if eff_use_cache and emb_finite:
            hit = self.cache.get(
                prompt,
                embedding=emb,
                telemetry=telemetry,
                schema_digest=schema_dig,
                model_version=cur_version, strict=True,
                recency_weighted=use_recency,
            )
            if hit is not None:
                entry, sim = hit
                if isinstance(entry.result, ModelInferenceResult):
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    return ModelInferenceResult(
                        prompt=prompt,
                        embedding=emb,
                        fields=entry.result.fields,
                        inference_latency_ms=elapsed_ms,
                        raw_norm=raw_norm if raw_norm is not None else entry.result.raw_norm,
                    )

        override_map: Optional[Dict[str, Any]] = None
        if (
            emb_finite
            and self._learned_prompt_overrides
            and embedding is None
            and telemetry is None
            and isinstance(prompt, str)
            and prompt.strip()
        ):
            norm_key = " ".join(prompt.strip().casefold().split())
            if self._learned_prompt_override_recency.get(norm_key, bool(self.recency_weighted)) == bool(use_recency):
                override_map = self._learned_prompt_overrides.get(norm_key)

        is_zero_emb_eval = not bool(np.any(emb))
        evals: Dict[str, RawFieldEvaluation] = {}
        for name, head in self._field_heads.items():
            if (
                override_map is not None
                and name in override_map
                and (is_zero_emb_eval or _is_active_zero_head(head.field_def, head.weights, head.biases))
            ):
                evals[name] = self._build_override_eval(head, override_map[name])
            else:
                evals[name] = head.forward(emb, prompt=prompt if embedding is None else None)
        latency = (time.perf_counter() - t0) * 1000.0
        result = ModelInferenceResult(
            prompt=prompt,
            embedding=emb,
            fields=evals,
            inference_latency_ms=latency,
            raw_norm=raw_norm,
        )

        if eff_use_cache and emb_finite:
            self.cache.put(
                prompt,
                result,
                embedding=emb,
                telemetry=telemetry,
                schema_digest=schema_dig,
                model_version=cur_version, strict=True,
                source="evaluation",
                explicit_embedding=(embedding is not None),
                recency_weighted=use_recency,
            )

        return result

    def forward_batch(
        self,
        prompts: Sequence[str],
        telemetry: Optional[Sequence[Any]] = None,
        recency_weighted: Optional[bool] = None,
    ) -> List[ModelInferenceResult]:
        """Batched forward evaluation over multiple inputs."""
        use_recency = self.recency_weighted if recency_weighted is None else bool(recency_weighted)
        if telemetry is not None and len(telemetry) == len(prompts):
            return [self.forward_single(p, telemetry[i], recency_weighted=use_recency) for i, p in enumerate(prompts)]
        return [self.forward_single(p, recency_weighted=use_recency) for p in prompts]

    def evaluate(
        self,
        prompt: str,
        telemetry: Optional[Any] = None,
        embedding: Optional[np.ndarray] = None,
        recency_weighted: Optional[bool] = None,
    ) -> ModelInferenceResult:
        """Alias for forward_single with optional continuous telemetry vector and precomputed embedding."""
        return self.forward_single(
            prompt,
            telemetry=telemetry,
            embedding=embedding,
            recency_weighted=recency_weighted,
        )

    def learn_from_tier2(
        self,
        prompt: str,
        target: Union[Mapping[str, Any], Any],
        telemetry: Optional[Any] = None,
        embedding: Optional[np.ndarray] = None,
        forgetting_factor: Optional[float] = None,
        recency_weighted: Optional[bool] = None,
        *,
        preserve_calibration: bool = False,
        reward_scale: float = 1.0,
    ) -> Dict[str, Any]:
        """Apply an optional online correction and invalidate changed calibration.

        Separate recalibration is needed before strict local acceptance. Retaining
        examples and recompiling is the simplest reproducible update workflow."""
        with self._lock:
            target_dict = self.schema.validate_lesson(target, allow_unknown=True)
            eff_ff = float(forgetting_factor if forgetting_factor is not None else self.forgetting_factor)
            if not math.isfinite(eff_ff) or not (0.0 < eff_ff <= 1.0):
                raise ValueError(f"Forgetting factor lambda_f must be in (0, 1.0], got {eff_ff}")
            eff_reward_scale = float(np.clip(reward_scale, 0.0, 1.0))
            t0 = time.perf_counter()
            use_recency = self.recency_weighted if recency_weighted is None else bool(recency_weighted)

            if embedding is not None:
                emb = np.asarray(embedding, dtype=np.float32).flatten()
            else:
                emb = self.encode(prompt, telemetry=telemetry, recency_weighted=use_recency)
            if len(emb) != self.dimension:
                raise ValueError(f"Expected embedding of length {self.dimension}, got {len(emb)}")
            if not np.all(np.isfinite(emb)):
                raise ValueError("embedding contains NaN or Inf non-finite values")
            x_aug = np.append(emb, 1.0).astype(np.float32)

            updated_fields: List[str] = []
            update_durations_ms: List[float] = []
            is_zero_emb = not bool(np.any(emb))
            is_cold_skill = bool(self.metadata.get("cold_start", False))
            prompt_for_head = (
                prompt if (embedding is None and isinstance(prompt, str) and prompt.strip()) else None
            )
            prompt_for_eval = prompt_for_head
            norm_key = (
                " ".join(prompt.strip().casefold().split())
                if (embedding is None and telemetry is None and isinstance(prompt, str) and prompt.strip())
                else None
            )

            for f_name, target_val in target_dict.items():
                if f_name in self._field_heads:
                    head = self._field_heads[f_name]
                    y_target = head.format_target_vector(target_val)
                    is_cold_head = head._exemplar_bank is not None or is_cold_skill
                    skip_field = False
                    if is_zero_emb:
                        dt = 0.0
                        if norm_key is not None and eff_reward_scale >= 0.5:
                            if len(self._learned_prompt_overrides) >= 4096 and norm_key not in self._learned_prompt_overrides:
                                oldest_key = next(iter(self._learned_prompt_overrides))
                                self._learned_prompt_overrides.pop(oldest_key, None)
                                self._learned_prompt_override_recency.pop(oldest_key, None)
                            self._learned_prompt_overrides.setdefault(norm_key, {})[f_name] = (
                                self.schema.fields[f_name].validate_value(target_val)
                            )
                            self._learned_prompt_override_recency[norm_key] = bool(use_recency)
                    elif is_cold_head:
                        if preserve_calibration and eff_reward_scale >= 0.5:
                            if norm_key is not None:
                                if len(self._learned_prompt_overrides) >= 4096 and norm_key not in self._learned_prompt_overrides:
                                    oldest_key = next(iter(self._learned_prompt_overrides))
                                    self._learned_prompt_overrides.pop(oldest_key, None)
                                    self._learned_prompt_override_recency.pop(oldest_key, None)
                                self._learned_prompt_overrides.setdefault(norm_key, {})[f_name] = (
                                    self.schema.fields[f_name].validate_value(target_val)
                                )
                                self._learned_prompt_override_recency[norm_key] = bool(use_recency)
                        elif not preserve_calibration:
                            if norm_key is not None and norm_key in self._learned_prompt_overrides:
                                self._learned_prompt_overrides[norm_key].pop(f_name, None)
                                if not self._learned_prompt_overrides[norm_key]:
                                    self._learned_prompt_overrides.pop(norm_key, None)
                                    self._learned_prompt_override_recency.pop(norm_key, None)
                        dt = head.online_update(
                            x_aug,
                            y_target,
                            forgetting_factor=eff_ff,
                            prompt=prompt_for_head,
                            preserve_calibration=preserve_calibration,
                            reward_scale=eff_reward_scale,
                        )
                    else:
                        if preserve_calibration:
                            if head.covariance_inv is None or head.cross_covariance is None:
                                head.init_covariance(regularization=max(4.0, float(getattr(head, "regularization", 1.0))))
                            x_step = np.append(emb, 0.0).astype(np.float32)
                            step_ff = 1.0 if eff_ff >= 0.95 else eff_ff
                            ch_ref = self.heads.get(f_name)
                            f_type = getattr(ch_ref, "field_type", None) or getattr(getattr(head, "definition", None), "field_type", "choice")
                            ch_temp = max(0.05, float(getattr(ch_ref, "temperature", 1.0))) if ch_ref is not None else 1.0
                            score_method = getattr(ch_ref, "score_method", "lac") if ch_ref is not None else "lac"
                            raw_cal_scores = getattr(ch_ref, "calibration_scores", ()) if ch_ref is not None else ()
                            ch_q = float(getattr(ch_ref, "conformal_quantile", 0.05)) if ch_ref is not None else 0.05
                            if score_method == "lac":
                                pos_scores = [float(s) for s in raw_cal_scores if 1e-6 < float(s) < 0.99]
                                eff_lac_q = max(ch_q if ch_q < 0.99 else 0.01, min(pos_scores) if pos_scores else 0.01)
                                q_goal = max(1e-4, min(0.035, eff_lac_q * 0.65))
                            else:
                                q_goal = 0.02
                            target_top_p = min(0.999, max(0.95, 1.0 - q_goal))
                            hier_scale = 1.45 if getattr(head, "hierarchical", False) else 1.0
                            wx0 = (head.weights @ emb).astype(np.float32)
                            b0 = np.asarray(head.biases, dtype=np.float32).flatten()
                            if f_type == "choice":
                                target_idx = int(np.argmax(y_target))
                                z0 = wx0 + b0
                                z_desired = z0.copy()
                                for k_idx in range(len(z_desired)):
                                    if k_idx != target_idx and z_desired[k_idx] > 0.10:
                                        z_desired[k_idx] = -0.15
                                other_scaled = np.array(
                                    [z_desired[k_idx] / (0.25 * ch_temp) for k_idx in range(len(z_desired)) if k_idx != target_idx],
                                    dtype=np.float64,
                                )
                                if len(other_scaled) > 0:
                                    m_other = float(np.max(other_scaled))
                                    log_sum_other = m_other + math.log(float(np.sum(np.exp(other_scaled - m_other))))
                                else:
                                    log_sum_other = 0.0
                                req_target_scaled = log_sum_other + math.log(max(2.0, (1.0 - q_goal) / max(1e-6, q_goal)))
                                req_target_raw = float(np.clip(0.25 * ch_temp * req_target_scaled * hier_scale + 0.15, 1.8, 7.5))
                                z_desired[target_idx] = max(float(z0[target_idx]), req_target_raw)
                                y_desired = (z_desired - b0).astype(np.float32)
                            elif f_type == "boolean":
                                req_scaled_mag = math.log((1.0 - q_goal) / max(1e-6, q_goal))
                                req_raw_mag = float(np.clip(0.25 * req_scaled_mag * hier_scale, 2.0, 6.0))
                                want_true = float(y_target[0]) > 0.0
                                b_val = float(b0[0]) if len(b0) > 0 else 0.0
                                y_desired = np.array([(req_raw_mag if want_true else -req_raw_mag) - b_val], dtype=np.float32)
                            elif f_type in ("multi_choice", "multichoice"):
                                ev_mc0 = head.forward(emb, prompt=prompt_for_eval)
                                probs_mc0 = np.asarray(ev_mc0.raw_probabilities, dtype=np.float64)
                                y_desired = wx0.copy()
                                z_mag = float(np.clip(0.25 * math.log((1.0 - q_goal) / max(1e-6, q_goal)), 1.8, 5.5))
                                for k_idx in range(len(y_target)):
                                    want_pos = float(y_target[k_idx]) > 0.0
                                    p_k = float(probs_mc0[k_idx]) if k_idx < len(probs_mc0) else 0.5
                                    err_k = (1.0 - p_k) if want_pos else p_k
                                    if err_k > q_goal:
                                        desired_logit = (z_mag if want_pos else -z_mag) - float(b0[k_idx])
                                        y_desired[k_idx] = np.float32(np.clip(desired_logit, -5.5, 5.5))
                            elif f_type == "score":
                                b_val = float(b0[0]) if len(b0) > 0 else 0.0
                                y_desired = np.array([float(y_target[0]) - b_val], dtype=np.float32)
                            else:
                                y_desired = y_target

                            if eff_reward_scale < 1.0:
                                y_desired = (wx0 + eff_reward_scale * (y_desired - wx0)).astype(np.float32)

                            dt = 0.0
                            if len(self._field_heads) > 1:
                                ev_pre = head.forward(emb, prompt=prompt_for_eval)
                                logits_pre = np.asarray(ev_pre.logits, dtype=np.float64) / ch_temp
                                eff_q_skip = ch_q if 0.0 < ch_q < 0.90 else 0.04
                                if f_type == "choice" and len(logits_pre) >= 2:
                                    shifted_pre = logits_pre - np.max(logits_pre)
                                    exp_pre = np.exp(shifted_pre)
                                    probs_c_pre = exp_pre / np.sum(exp_pre)
                                    pred_idx_pre = int(np.argmax(probs_c_pre))
                                    p_top_pre = float(probs_c_pre[pred_idx_pre])
                                    if pred_idx_pre == target_idx and (1.0 - p_top_pre) <= max(1e-3, eff_q_skip) and p_top_pre >= 0.95:
                                        skip_field = True
                                elif f_type in ("multi_choice", "multichoice"):
                                    probs_pre = np.asarray(ev_pre.raw_probabilities, dtype=np.float64)
                                    s_pre = float(np.max(np.minimum(probs_pre, 1.0 - probs_pre))) if len(probs_pre) > 0 else 0.0
                                    if set(ev_pre.selected_value) == set(target_val) and s_pre <= max(0.04, eff_q_skip):
                                        skip_field = True
                                elif f_type == "boolean":
                                    p_sel_pre = float(ev_pre.confidence)
                                    if bool(ev_pre.selected_value) == bool(target_val) and (1.0 - p_sel_pre) <= max(0.04, eff_q_skip) and p_sel_pre >= 0.95:
                                        skip_field = True
                                elif f_type == "score":
                                    if abs(float(ev_pre.selected_value) - float(target_val)) <= min(0.04, max(0.01, eff_q_skip * 0.8)):
                                        skip_field = True
                            if not skip_field:
                                # Exact Sherman-Morrison directional gain scaling so W_next @ emb == y_desired in 1 step
                                p_mat = head.covariance_inv
                                g_val = float(x_step @ (p_mat @ x_step)) if p_mat is not None else 0.0
                                if g_val > 1e-6:
                                    y_step = ((step_ff + g_val) * y_desired - step_ff * wx0) / g_val
                                    y_step = np.clip(y_step, -40.0, 40.0).astype(np.float32)
                                else:
                                    y_step = y_desired
                                dt = head.online_update(
                                    x_step,
                                    y_step,
                                    forgetting_factor=step_ff,
                                    prompt=prompt_for_head,
                                )
                                if eff_reward_scale >= 0.5:
                                    for step_i in range(4):
                                        ev = head.forward(emb, prompt=prompt_for_eval)
                                        logits_ev = np.asarray(ev.logits, dtype=np.float64) / ch_temp
                                        if f_type == "choice" and len(logits_ev) >= 2:
                                            shifted_ev = logits_ev - np.max(logits_ev)
                                            exp_ev = np.exp(shifted_ev)
                                            probs_cal = exp_ev / np.sum(exp_ev)
                                            s_p = np.sort(probs_cal)
                                            if (
                                                ev.selected_value == target_val
                                                and float(s_p[-1]) >= target_top_p
                                                and float(s_p[-1] - s_p[-2]) >= 0.55
                                            ):
                                                break
                                        elif f_type in ("multi_choice", "multichoice"):
                                            probs_mc = np.asarray(ev.raw_probabilities, dtype=np.float64)
                                            max_err_mc = float(np.max(np.minimum(probs_mc, 1.0 - probs_mc))) if len(probs_mc) > 0 else 0.0
                                            if (
                                                set(ev.selected_value) == set(target_val)
                                                and max_err_mc <= max(0.03, ch_q * 0.9)
                                            ):
                                                break
                                        elif f_type == "score":
                                            if abs(float(ev.selected_value) - float(target_val)) <= 0.02:
                                                break
                                        elif f_type == "boolean":
                                            p_sel = float(ev.confidence)
                                            if bool(ev.selected_value) == bool(target_val) and p_sel >= target_top_p:
                                                break
                                        elif bool(ev.selected_value) == bool(target_val) and ev.confidence >= 0.90:
                                            break
                                        wx_curr = (head.weights @ emb).astype(np.float32)
                                        p_curr = head.covariance_inv
                                        g_curr = float(x_step @ (p_curr @ x_step)) if p_curr is not None else 0.0
                                        if g_curr > 1e-6:
                                            y_refine = ((step_ff + g_curr) * y_desired - step_ff * wx_curr) / g_curr
                                            y_refine = np.clip(y_refine, -40.0, 40.0).astype(np.float32)
                                        else:
                                            y_refine = y_desired
                                        dt += head.online_update(
                                            x_step,
                                            y_refine,
                                            forgetting_factor=step_ff,
                                            prompt=prompt_for_head,
                                        )
                            if norm_key is not None and eff_reward_scale >= 0.5:
                                if len(self._learned_prompt_overrides) >= 4096 and norm_key not in self._learned_prompt_overrides:
                                    oldest_key = next(iter(self._learned_prompt_overrides))
                                    self._learned_prompt_overrides.pop(oldest_key, None)
                                    self._learned_prompt_override_recency.pop(oldest_key, None)
                                self._learned_prompt_overrides.setdefault(norm_key, {})[f_name] = (
                                    self.schema.fields[f_name].validate_value(target_val)
                                )
                                self._learned_prompt_override_recency[norm_key] = bool(use_recency)
                        else:
                            if norm_key is not None and norm_key in self._learned_prompt_overrides:
                                self._learned_prompt_overrides[norm_key].pop(f_name, None)
                                if not self._learned_prompt_overrides[norm_key]:
                                    self._learned_prompt_overrides.pop(norm_key, None)
                                    self._learned_prompt_override_recency.pop(norm_key, None)
                            dt = head.online_update(
                                x_aug,
                                y_target,
                                forgetting_factor=eff_ff,
                                prompt=prompt_for_head,
                            )
                    update_durations_ms.append(dt)

                    # Sync back to compiled head weights
                    if f_name in self.heads:
                        ch = self.heads[f_name]
                        ch.weights = head.weights
                        ch.biases = head.biases
                        ch.P = head.covariance_inv
                        ch.B = head.cross_covariance
                        ch.exemplar_state = head.export_exemplar_state()
                        if head.hierarchical:
                            ch.coarse_weights = head.coarse_weights
                            ch.coarse_biases = head.coarse_biases
                            ch.fine_weights = head.fine_weights
                            ch.fine_biases = head.fine_biases
                        if not is_zero_emb and ch.prototypes is not None and ch.prototypes.shape[1] == len(emb):
                            e_norm = float(np.linalg.norm(emb))
                            if e_norm > 1e-12:
                                emb_u = emb / e_norm
                                proto_idx: Optional[int] = None
                                if ch.field_type == "choice" and len(y_target) == ch.prototypes.shape[0]:
                                    proto_idx = int(np.argmax(y_target))
                                elif ch.field_type == "boolean" and ch.prototypes.shape[0] == 2:
                                    proto_idx = 1 if float(y_target[0]) > 0.0 else 0
                                if not skip_field and proto_idx is not None and 0 <= proto_idx < ch.prototypes.shape[0]:
                                    cur_p = ch.prototypes[proto_idx]
                                    sim_before = float(np.max(ch.prototypes @ emb_u)) if np.any(cur_p) else 0.0
                                    if not preserve_calibration or ch.prototype_threshold <= 0.0 or not np.any(cur_p):
                                        blend = (0.25 if preserve_calibration else 0.15) * eff_reward_scale
                                        updated_p = emb_u if not np.any(cur_p) else ((1.0 - blend) * cur_p + blend * emb_u)
                                        up_norm = float(np.linalg.norm(updated_p))
                                        if up_norm > 1e-12:
                                            ch.prototypes = ch.prototypes.copy()
                                            ch.prototypes[proto_idx] = (updated_p / up_norm).astype(np.float32)
                                    elif eff_reward_scale >= 0.5 and sim_before < ch.prototype_threshold:
                                        if not hasattr(self, "_online_prototype_bank"):
                                            self._online_prototype_bank = {}
                                        bank = self._online_prototype_bank.setdefault(f_name, [])
                                        if len(bank) >= 256:
                                            bank.pop(0)
                                        bank.append(emb_u.astype(np.float32))
                                if preserve_calibration and eff_reward_scale >= 0.5 and ch.prototype_threshold > 0.0:
                                    sim_after = self.compute_prototype_similarity(f_name, emb_u)
                                    if sim_after is not None and sim_after < ch.prototype_threshold:
                                        ch.prototype_threshold = max(0.0, sim_after - 1e-4)
                        if (
                            preserve_calibration
                            and eff_reward_scale >= 0.5
                            and ch.familiarity_threshold > 0.0
                            and isinstance(prompt, str)
                            and prompt.strip()
                        ):
                            p_raw_norm = _extract_projector_raw_norm(
                                self.projector, prompt, recency_weighted=use_recency
                            )
                            if (
                                p_raw_norm is not None
                                and math.isfinite(p_raw_norm)
                                and 0.0 < p_raw_norm < ch.familiarity_threshold
                            ):
                                ch.familiarity_threshold = max(0.0, p_raw_norm - 1e-4)
                        if preserve_calibration and not ch.calibration_scores:
                            if ch.field_type in ("choice", "boolean", "multi_choice", "multichoice"):
                                k_opts = max(2, len(ch.options))
                                q_init = float((k_opts - 1.0) / (3.5 + k_opts))
                                ch.score_method = "lac"
                                ch.calibration_scores = tuple(q_init for _ in range(20))
                                ch.conformal_quantile = q_init
                            elif ch.field_type == "score":
                                f_def_sc = self.schema.fields.get(f_name)
                                q_init_sc = (
                                    0.10 * _score_half_span(f_def_sc.min_value, f_def_sc.max_value)
                                    if isinstance(f_def_sc, ScoreField)
                                    else 0.05
                                )
                                ch.calibration_scores = tuple(q_init_sc for _ in range(20))
                                ch.conformal_quantile = q_init_sc
                        elif not preserve_calibration:
                            # Old evidence describes the previous weights, including on disk.
                            ch.calibration_scores = ()
                            ch.class_calibration_scores = {}
                            ch.familiarity_threshold = 0.0
                            ch.prototype_threshold = 0.0
                            ch.prototype_scores = ()
                            ch.conformal_quantile = 0.0
                            ch.temperature = 1.0
                            ch.logit_adjustments = None
                    updated_fields.append(f_name)

            self.model_version = getattr(self, "model_version", 1) + 1
            if self.use_cache and self.cache is not None:
                if embedding is None and hasattr(self.cache, "evict_prompt"):
                    self.cache.evict_prompt(prompt)
                if hasattr(self.cache, "invalidate_prior_versions"):
                    self.cache.invalidate_prior_versions(self.model_version)

            rank1_ms = float(sum(update_durations_ms)) if update_durations_ms else 0.0
            total_ms = (time.perf_counter() - t0) * 1000.0

            # Evaluate model to get fresh result and store in Tier 0 cache
            override_map = None
            if norm_key is not None and self._learned_prompt_override_recency.get(norm_key, bool(self.recency_weighted)) == bool(use_recency):
                override_map = self._learned_prompt_overrides.get(norm_key)
            evals: Dict[str, RawFieldEvaluation] = {}
            for name, head in self._field_heads.items():
                if (
                    override_map is not None
                    and name in override_map
                    and (is_zero_emb or _is_active_zero_head(head.field_def, head.weights, head.biases))
                ):
                    evals[name] = self._build_override_eval(head, override_map[name])
                else:
                    evals[name] = head.forward(emb, prompt=prompt_for_eval)
            inference_res = ModelInferenceResult(
                prompt=prompt,
                embedding=emb,
                fields=evals,
                inference_latency_ms=rank1_ms,
            )
            if self.use_cache:
                self.cache.put(
                    prompt,
                    inference_res,
                    embedding=emb,
                    telemetry=telemetry,
                    schema_digest=self.schema.schema_digest(),
                    model_version=self.model_version, strict=True,
                    source="tier2",
                    explicit_embedding=(embedding is not None),
                    recency_weighted=use_recency,
                )

            return {
                "status": "updated",
                "update_latency_ms": round(rank1_ms, 4),
                "total_latency_ms": round(total_ms, 4),
                "updated_fields": updated_fields,
            }

    def adapt_zero_shot(
        self,
        prompts: Union[str, Sequence[str]],
        *,
        telemetry: Optional[Sequence[Any]] = None,
        margin_threshold: float = 0.18,
        learning_rate: float = 0.25,
        recency_weighted: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Unsupervised test-time zero-shot prototype and exemplar adaptation across schema heads."""
        with self._lock:
            t0 = time.perf_counter()
            if isinstance(prompts, str):
                prompt_seq: Sequence[str] = [prompts]
            else:
                prompt_seq = list(prompts)
            if len(prompt_seq) == 0:
                return {"status": "no_op", "adapted_fields": {}, "adapted_samples": 0, "latency_ms": 0.0}

            embs = self.encode_batch(prompt_seq, telemetry=telemetry, recency_weighted=recency_weighted)
            adapted_fields: Dict[str, Any] = {}
            total_adapted = 0
            for f_name, head in self._field_heads.items():
                if hasattr(head, "adapt_zero_shot"):
                    info = head.adapt_zero_shot(
                        embs,
                        prompts=prompt_seq,
                        margin_threshold=margin_threshold,
                        learning_rate=learning_rate,
                    )
                    if info.get("adapted_samples", 0) > 0:
                        adapted_fields[f_name] = info
                        total_adapted = max(total_adapted, int(info["adapted_samples"]))
                        if f_name in self.heads:
                            ch = self.heads[f_name]
                            ch.weights = head.weights
                            ch.biases = head.biases
                            ch.P = head.covariance_inv
                            ch.B = head.cross_covariance
                            ch.exemplar_state = head.export_exemplar_state()

            if total_adapted > 0:
                self.model_version = getattr(self, "model_version", 1) + 1
                if self.use_cache and self.cache is not None and hasattr(self.cache, "invalidate_prior_versions"):
                    self.cache.invalidate_prior_versions(self.model_version)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "status": "adapted" if adapted_fields else "no_op",
                "adapted_fields": adapted_fields,
                "adapted_samples": total_adapted,
                "latency_ms": round(elapsed_ms, 4),
            }

    # Cognitive dual-process and execution pipeline aliases
    learn_from_system2 = learn_from_tier2
    learn_from_tier3 = learn_from_tier2

    def learn(
        self,
        prompt: str,
        target: Union[Mapping[str, Any], Any] = None,
        telemetry: Optional[Any] = None,
        embedding: Optional[np.ndarray] = None,
        forgetting_factor: Optional[float] = None,
        recency_weighted: Optional[bool] = None,
        *,
        preserve_calibration: bool = True,
        outcome: Optional[Any] = None,
        reward: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Online teaching with retained heuristic gates, not preserved coverage."""
        if forgetting_factor is not None:
            ff_val = float(forgetting_factor)
            if not math.isfinite(ff_val) or not (0.0 < ff_val <= 1.0):
                raise ValueError(f"Forgetting factor lambda_f must be in (0, 1.0], got {ff_val}")
        eff_ff = forgetting_factor
        eff_reward_scale = 1.0
        if reward is not None:
            r_val = float(reward)
            if not math.isfinite(r_val):
                raise ValueError("reward must be a finite number")
            if r_val <= 0.0:
                skipped: Dict[str, Any] = {
                    "status": "skipped_non_positive_reward",
                    "update_latency_ms": 0.0,
                    "total_latency_ms": 0.0,
                    "updated_fields": [],
                    "reward": r_val,
                }
                if outcome is not None:
                    skipped["outcome_recorded"] = True
                return skipped
            eff_reward_scale = float(np.clip(r_val, 0.0, 1.0))
            if r_val != 1.0:
                base_ff = float(forgetting_factor if forgetting_factor is not None else self.forgetting_factor)
                eff_ff = float(np.clip(1.0 - (1.0 - base_ff) * min(r_val, 2.0), 0.5, 1.0))

        resolved_target = target
        if resolved_target is None and outcome is not None:
            if (
                isinstance(outcome, Mapping)
                and "target" in outcome
                and not (set(outcome.keys()) & set(self.schema.fields.keys()))
            ):
                resolved_target = outcome["target"]
            else:
                resolved_target = outcome
        resolved_target = self.schema.validate_lesson(resolved_target)

        res = self.learn_from_tier2(
            prompt,
            resolved_target,
            telemetry=telemetry,
            embedding=embedding,
            forgetting_factor=eff_ff,
            recency_weighted=recency_weighted,
            preserve_calibration=preserve_calibration,
            reward_scale=eff_reward_scale,
        )
        if outcome is not None:
            res["outcome_recorded"] = True
        if reward is not None:
            res["reward"] = float(reward)
        return res

    def learn_from_outcome(
        self,
        prompt: str,
        target: Union[Mapping[str, Any], Any] = None,
        telemetry: Optional[Any] = None,
        embedding: Optional[np.ndarray] = None,
        forgetting_factor: Optional[float] = None,
        recency_weighted: Optional[bool] = None,
        *,
        preserve_calibration: bool = True,
        outcome: Optional[Any] = None,
        reward: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Online teaching from application context, user actions, or business outcomes."""
        return self.learn(
            prompt,
            target,
            telemetry=telemetry,
            embedding=embedding,
            forgetting_factor=forgetting_factor,
            recency_weighted=recency_weighted,
            preserve_calibration=preserve_calibration,
            outcome=outcome,
            reward=reward,
        )

    def recalibrate(
        self,
        examples: Union[Sequence[Any], str, Path],
        alpha: float = 0.05,
        *,
        n_bins: int = 10,
        familiarity_alpha: Optional[float] = None,
        prototype_alpha: Optional[float] = None,
        mondrian: Optional[bool] = None,
    ) -> Any:
        """Recalibrate temperature scaling and conformal quantiles on held-out examples without modifying weights."""
        from system1.engine import SystemOneEngine

        engine = SystemOneEngine(
            schema=self.schema,
            model=self,
            dimension=self.dimension,
            backend=self.backend,
            projector=self.projector,
            forgetting_factor=self.forgetting_factor,
            recency_weighted=self.recency_weighted,
            use_cache=False,
        )
        return engine.recalibrate(
            examples,
            alpha=alpha,
            n_bins=n_bins,
            familiarity_alpha=familiarity_alpha,
            prototype_alpha=prototype_alpha,
            mondrian=mondrian,
        )

    def evaluate_dataset(
        self,
        rows: Union[Sequence[Any], str, Path],
        *,
        alpha: Optional[float] = None,
        n_bins: int = 10,
        strict_mode: Optional[bool] = None,
        use_cache: bool = False,
    ) -> Any:
        """Evaluate the compiled model on a labeled dataset and return an EvaluationReport."""
        from system1.engine import SystemOneEngine

        eff_strict = bool(self.metadata.get("strict_mode", True)) if strict_mode is None else bool(strict_mode)
        engine = SystemOneEngine(
            schema=self.schema,
            model=self,
            dimension=self.dimension,
            backend=self.backend,
            projector=self.projector,
            forgetting_factor=self.forgetting_factor,
            recency_weighted=self.recency_weighted,
            strict_mode=eff_strict,
            use_cache=use_cache,
        )
        return engine.evaluate_dataset(
            rows,
            alpha=alpha,
            n_bins=n_bins,
            strict_mode=eff_strict,
            use_cache=use_cache,
        )

    def _build_export_header_and_arrays(
        self,
        include_covariance: bool = False,
        *,
        compact: Optional[bool] = None,
        format_version: int = 2,
        embed_vocab: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        """Builds metadata dictionary and contiguous array payload for .s1m serialization."""
        if format_version not in (2, 3):
            raise ValueError(f"Unsupported .s1m format_version {format_version!r}; expected 2 or 3")
        if format_version == 3 and not 0 < len(self.schema.fields) <= MAX_SKILL_FIELDS_V3:
            raise ValueError(
                f"Skill field count {len(self.schema.fields)} exceeds v3 limit {MAX_SKILL_FIELDS_V3}"
            )

        heads_meta: Dict[str, Any] = {}
        exemplar_states = {name: head.export_exemplar_state() for name, head in self._field_heads.items()}
        for name, ch in sorted(self.heads.items()):
            h_entry: Dict[str, Any] = {
                "field_type": ch.field_type,
                "temperature": float(ch.temperature),
                "conformal_quantile": float(ch.conformal_quantile),
                "score_method": ch.score_method,
                "margin_threshold": float(getattr(ch, "margin_threshold", 0.20)),
                "forgetting_factor": float(getattr(ch, "forgetting_factor", 0.995)),
                "relative_odds_ratio": float(getattr(ch, "relative_odds_ratio", 1.5)),
                "confidence_floor_tau0": float(getattr(ch, "confidence_floor_tau0", 0.15)),
                "escalate_on_ambiguity": bool(getattr(ch, "escalate_on_ambiguity", True)),
                "options": list(ch.options),
                "weights_shape": list(ch.weights.shape),
                "biases_shape": list(ch.biases.shape),
            }
            if getattr(ch, "mondrian", False):
                h_entry["mondrian"] = True
            class_scores = getattr(ch, "class_calibration_scores", None)
            if class_scores and (
                getattr(ch, "mondrian", False)
                or sum(len(v) for v in class_scores.values()) <= 2048
            ):
                h_entry["class_calibration_scores"] = {
                    str(k): [float(v) for v in vals] for k, vals in sorted(class_scores.items())
                }
            fam_thresh = float(getattr(ch, "familiarity_threshold", 0.0))
            if fam_thresh > 0.0:
                h_entry["familiarity_threshold"] = fam_thresh
            proto_thresh = float(getattr(ch, "prototype_threshold", 0.0))
            if proto_thresh > 0.0:
                h_entry["prototype_threshold"] = proto_thresh
            f_def_h = self.schema.fields.get(name)
            if not isinstance(f_def_h, BooleanField) and getattr(ch, "hierarchical", False):
                if (
                    ch.coarse_weights is None
                    or ch.coarse_biases is None
                    or len(getattr(ch, "cluster_assignments", ()) or ()) != ch.weights.shape[0]
                    or (ch.fine_weights is None) != (ch.fine_biases is None)
                ):
                    raise ValueError(f"Head {name!r} has incomplete hierarchical parameters")
                h_entry["hierarchical"] = True
                h_entry["cluster_assignments"] = [int(c) for c in ch.cluster_assignments]
            state = exemplar_states.get(name)
            if state is not None:
                h_entry["exemplar_state"] = {key: value for key, value in state.items() if key != "arrays"}
            heads_meta[name] = h_entry

        serialized_meta: Dict[str, Any] = {
            **self.metadata,
            "forgetting_factor": self.forgetting_factor,
            "recency_weighted": self.recency_weighted,
        }
        # Loaded metadata may describe lessons that have since been cleared.
        serialized_meta.pop("learned_prompt_overrides", None)
        serialized_meta.pop("learned_prompt_override_recency", None)
        if self._learned_prompt_overrides:
            serialized_meta["learned_prompt_overrides"] = {
                p_k: {
                    f_k: (list(f_v) if isinstance(f_v, tuple) else f_v)
                    for f_k, f_v in f_map.items()
                }
                for p_k, f_map in self._learned_prompt_overrides.items()
            }
            serialized_meta["learned_prompt_override_recency"] = {
                p_k: bool(self._learned_prompt_override_recency.get(p_k, self.recency_weighted))
                for p_k in self._learned_prompt_overrides
            }

        use_binary_tfidf_idf = (
            type(self.projector) is TfidfProjector
            and (compact is True or len(self.projector._terms) > 128)
        )
        use_binary_prag_tfidf = (
            isinstance(self.projector, PragmaticCompositeProjector)
            and (
                compact is True
                or (len(self.projector.base_tfidf._terms) + len(self.projector._prag_terms)) > 128
            )
        )
        header_meta: Dict[str, Any] = {
            "version": int(format_version),
            "projector": _projector_config(
                self.projector,
                include_arrays=(
                    not use_binary_tfidf_idf
                    if type(self.projector) is TfidfProjector
                    else (
                        not use_binary_prag_tfidf
                        if isinstance(self.projector, PragmaticCompositeProjector)
                        else False
                    )
                ),
            ),
            "schema_name": self.schema.schema_name,
            "schema_dict": self.schema.to_dict(),
            "schema_digest": self.schema.schema_digest(),
            "dimension": self.dimension,
            "created_at": getattr(self, "created_at", 0.0),
            "metadata": serialized_meta,
            "heads": heads_meta,
        }
        if format_version == 3:
            header_meta["embed_vocab"] = bool(embed_vocab)
        namespaced_tensors = _schema_needs_namespaced_head_tensors(self.schema)
        if namespaced_tensors:
            header_meta["tensor_namespace"] = "len_v1"

        def _is_f_2d_tensor(arr: np.ndarray) -> bool:
            return bool(
                isinstance(arr, np.ndarray)
                and arr.ndim == 2
                and arr.size > 0
                and arr.flags["F_CONTIGUOUS"]
                and not arr.flags["C_CONTIGUOUS"]
            )

        def _pack_array(arr: np.ndarray) -> np.ndarray:
            is_f_2d = _is_f_2d_tensor(arr)
            with np.errstate(over="ignore", invalid="ignore"):
                a32 = np.asarray(arr).astype(np.float32, order="K", copy=False)
            if not bool(np.all(np.isfinite(a32))):
                raise ValueError("Compiled model array must be finite in float32")
            if a32.ndim == 2 and a32.size > 0:
                if format_version == 3:
                    if compact is True:
                        with np.errstate(over="ignore", invalid="ignore"):
                            a16 = a32.astype(np.float16, order="K")
                        if np.all(np.isfinite(a16)):
                            return np.asfortranarray(a16) if is_f_2d else np.ascontiguousarray(a16)
                    return np.asfortranarray(a32) if is_f_2d else np.ascontiguousarray(a32)
                with np.errstate(over="ignore", invalid="ignore"):
                    a16 = a32.astype(np.float16, order="K")
                if np.all(np.isfinite(a16)) and (
                    compact is True
                    or (compact is None and np.array_equal(a32, a16.astype(np.float32)))
                ):
                    return np.asfortranarray(a16) if is_f_2d else a16
            if format_version == 3:
                return np.asfortranarray(a32) if is_f_2d else np.ascontiguousarray(a32)
            return np.asfortranarray(a32) if is_f_2d else a32

        tensor_layouts: Dict[str, List[int]] = {}

        def _pack_named_array(tname: str, arr: np.ndarray) -> np.ndarray:
            arr_np = np.asarray(arr)
            if _needs_2d_layout_preservation(arr_np):
                clone_spec = _validate_2d_layout_cloneable(arr_np, np.dtype(np.float32))
                if clone_spec is not None:
                    mod64, s0_f32, s1_f32, _, _ = clone_spec
                    tensor_layouts[tname] = [int(mod64), int(s0_f32), int(s1_f32)]
                return _pack_array(np.array(arr_np, order="C", copy=True))
            return _pack_array(arr_np)

        def _pack_f32_vec(arr: np.ndarray, label: str) -> np.ndarray:
            with np.errstate(over="ignore", invalid="ignore"):
                out = np.asarray(arr, dtype=np.float32)
            if not bool(np.all(np.isfinite(out))):
                raise ValueError(f"{label} must be finite in float32")
            return out

        arrays_to_save: Dict[str, np.ndarray] = {}
        from system1.core.embeddings import HybridProjector

        if use_binary_tfidf_idf:
            arrays_to_save["_projector_tfidf_idf"] = np.asarray(self.projector._idf, dtype=np.float64)
        if use_binary_prag_tfidf:
            arrays_to_save["_projector_prag_base_idf"] = np.asarray(
                self.projector.base_tfidf._idf, dtype=np.float64
            )
            if self.projector.prag_dim > 0:
                arrays_to_save["_projector_prag_idf"] = np.asarray(
                    self.projector.prag_idf, dtype=np.float64
                )
        if isinstance(self.projector, HybridProjector) and getattr(self.projector, "_fitted", False):
            if getattr(self.projector, "_spectral_vectors", None) is not None:
                arrays_to_save["_projector_spectral_vectors"] = _pack_named_array(
                    "_projector_spectral_vectors", self.projector._spectral_vectors
                )
            if getattr(self.projector, "_subspace_components", None) is not None:
                arrays_to_save["_projector_subspace_components"] = _pack_named_array(
                    "_projector_subspace_components", self.projector._subspace_components
                )
            if getattr(self.projector, "_rbf_landmarks", None) is not None:
                arrays_to_save["_projector_rbf_landmarks"] = _pack_named_array(
                    "_projector_rbf_landmarks", self.projector._rbf_landmarks
                )
            hybrid_tfidf = getattr(self.projector, "_tfidf_projector", None)
            if hybrid_tfidf is not None and getattr(hybrid_tfidf, "_idf", None) is not None:
                arrays_to_save["_projector_hybrid_tfidf_idf"] = np.asarray(
                    hybrid_tfidf._idf, dtype=np.float64
                )

        if format_version == 3 and embed_vocab and isinstance(self.projector, HybridProjector):
            dense_eng = self.projector._dense_engine
            arrays_to_save["_projector_dense_table"] = np.ascontiguousarray(
                dense_eng._table, dtype=np.float32
            )
            if getattr(self.projector, "_spectral_basis", None) is not None:
                arrays_to_save["_projector_spectral_basis"] = _pack_named_array(
                    "_projector_spectral_basis", self.projector._spectral_basis
                )
            header_meta["dense_embed_vocab"] = {
                "vocab_size": int(dense_eng.vocab_size),
                "seed": int(dense_eng.seed),
                "explicit_alpha": bool(getattr(self.projector, "_explicit_alpha", True)),
                "multiword_by_space3": {
                    str(k): [[str(e[0]), bool(e[1]), int(e[2])] for e in entries]
                    for k, entries in dense_eng._multiword_by_space3.items()
                },
                "sparse": (
                    self.projector._tfidf_projector.to_config(include_arrays=False)
                    if self.projector._tfidf_projector is not None
                    else None
                ),
            }

        for name, ch in sorted(self.heads.items()):
            def _hk(suffix: str, _fn: str = name) -> str:
                return _head_tensor_name(_fn, suffix, namespaced=namespaced_tensors)

            arrays_to_save[_hk("w")] = _pack_named_array(_hk("w"), ch.weights)
            b_arr = np.asarray(ch.biases)
            if b_arr.ndim != 1:
                raise ValueError(f"Head {name!r} biases must be 1D, got shape {b_arr.shape}")
            arrays_to_save[_hk("b")] = _pack_f32_vec(b_arr, f"Head {name!r} biases")
            state = exemplar_states.get(name)
            if state is not None:
                for suffix, arr in state["arrays"].items():
                    arr_np = np.asarray(arr)
                    arrays_to_save[_hk(suffix)] = (
                        _pack_named_array(_hk(suffix), arr_np) if arr_np.ndim == 2 else arr_np
                    )
            if hasattr(ch, "calibration_scores") and len(ch.calibration_scores) > 0:
                arrays_to_save[_hk("calib_scores")] = np.asarray(ch.calibration_scores, dtype=np.float64)
            if getattr(ch, "logit_adjustments", None) is not None:
                adj_valid = _validate_compiled_logit_adj(
                    ch.logit_adjustments, self.schema.fields[name], name
                )
                if adj_valid is not None:
                    arrays_to_save[_hk("logit_adj")] = np.ascontiguousarray(adj_valid, dtype=np.float64)
            if getattr(ch, "prototypes", None) is not None and np.any(ch.prototypes):
                proto_32 = _pack_f32_vec(ch.prototypes, f"Head {name!r} prototypes")
                with np.errstate(over="ignore", invalid="ignore"):
                    proto_16 = proto_32.astype(np.float16)
                if (
                    compact is not False
                    and (format_version == 2 or compact is True)
                    and np.all(np.isfinite(proto_16))
                ):
                    arrays_to_save[_hk("prototypes")] = proto_16
                else:
                    arrays_to_save[_hk("prototypes")] = _pack_array(proto_32)
            if getattr(ch, "prototype_scores", ()) and len(ch.prototype_scores) > 0:
                arrays_to_save[_hk("proto_scores")] = np.asarray(ch.prototype_scores, dtype=np.float64)
            if (
                getattr(ch, "hierarchical", False)
                and ch.coarse_weights is not None
                and ch.coarse_biases is not None
            ):
                cb_arr = np.asarray(ch.coarse_biases)
                if cb_arr.ndim != 1:
                    raise ValueError(f"Head {name!r} coarse_biases must be 1D, got shape {cb_arr.shape}")
                arrays_to_save[_hk("coarse_w")] = _pack_named_array(_hk("coarse_w"), ch.coarse_weights)
                arrays_to_save[_hk("coarse_b")] = _pack_f32_vec(
                    cb_arr, f"Head {name!r} coarse biases"
                )
                if ch.fine_weights is not None and ch.fine_biases is not None:
                    fb_arr = np.asarray(ch.fine_biases)
                    if fb_arr.ndim != 1:
                        raise ValueError(f"Head {name!r} fine_biases must be 1D, got shape {fb_arr.shape}")
                    arrays_to_save[_hk("fine_w")] = _pack_named_array(_hk("fine_w"), ch.fine_weights)
                    arrays_to_save[_hk("fine_b")] = _pack_f32_vec(
                        fb_arr, f"Head {name!r} fine biases"
                    )
            if include_covariance:
                if ch.P is not None:
                    arrays_to_save[_hk("P")] = _pack_named_array(_hk("P"), ch.P)
                if ch.B is not None:
                    arrays_to_save[_hk("B")] = _pack_named_array(_hk("B"), ch.B)

        if tensor_layouts:
            header_meta["tensor_layouts"] = tensor_layouts

        if format_version == 3:
            for tname, arr in list(arrays_to_save.items()):
                is_f_2d = _is_f_2d_tensor(arr)
                if arr.dtype.name not in _ALLOWED_V3_DTYPES:
                    arr_norm = (
                        np.asfortranarray(arr, dtype=np.float32)
                        if is_f_2d
                        else np.ascontiguousarray(arr, dtype=np.float32)
                    )
                else:
                    arr_norm = np.asfortranarray(arr) if is_f_2d else np.ascontiguousarray(arr)
                arrays_to_save[tname] = arr_norm

        return header_meta, arrays_to_save

    def _to_bytes_v3(
        self,
        include_covariance: bool = False,
        *,
        compact: Optional[bool] = None,
        embed_vocab: bool = False,
    ) -> bytes:
        header_meta, arrays_to_save = self._build_export_header_and_arrays(
            include_covariance=include_covariance,
            compact=compact,
            format_version=3,
            embed_vocab=embed_vocab,
        )
        arrays_ordered = sorted(arrays_to_save.items(), key=lambda kv: kv[0])

        def _v3_manifest_entry(arr: np.ndarray, offset: int) -> Dict[str, Any]:
            entry: Dict[str, Any] = {
                "dtype": arr.dtype.name,
                "shape": [int(d) for d in arr.shape],
                "offset": int(offset),
                "nbytes": int(arr.nbytes),
            }
            if (
                arr.ndim == 2
                and arr.size > 0
                and arr.flags["F_CONTIGUOUS"]
                and not arr.flags["C_CONTIGUOUS"]
            ):
                entry["order"] = "F"
            return entry

        manifest: Dict[str, Dict[str, Any]] = {
            tname: _v3_manifest_entry(arr, 0) for tname, arr in arrays_ordered
        }
        header_meta["tensors"] = manifest
        header_bytes = json.dumps(header_meta, sort_keys=True, allow_nan=False).encode("utf-8")
        if len(header_bytes) > MAX_SKILL_HEADER_BYTES_V3:
            raise ValueError("Saved v3 skill header exceeds size limit")

        cursor = 0
        for _ in range(8):
            cursor = (
                (16 + len(header_bytes) + S1M_V3_PAGE_SIZE - 1) // S1M_V3_PAGE_SIZE
            ) * S1M_V3_PAGE_SIZE
            next_manifest: Dict[str, Dict[str, Any]] = {}
            for idx, (tname, arr) in enumerate(arrays_ordered):
                align = (
                    S1M_V3_PAGE_SIZE
                    if (idx == 0 or arr.nbytes >= 65536 or tname == "_projector_dense_table")
                    else 64
                )
                cursor = ((cursor + align - 1) // align) * align
                next_manifest[tname] = _v3_manifest_entry(arr, cursor)
                cursor += int(arr.nbytes)
            header_meta["tensors"] = next_manifest
            next_header_bytes = json.dumps(header_meta, sort_keys=True, allow_nan=False).encode("utf-8")
            if len(next_header_bytes) > MAX_SKILL_HEADER_BYTES_V3:
                raise ValueError("Saved v3 skill header exceeds size limit")
            if next_manifest == manifest and next_header_bytes == header_bytes:
                header_bytes = next_header_bytes
                manifest = next_manifest
                break
            header_bytes = next_header_bytes
            manifest = next_manifest
        else:
            # Final pass to guarantee offsets remain consistent with settled header_len.
            cursor = (
                (16 + len(header_bytes) + S1M_V3_PAGE_SIZE - 1) // S1M_V3_PAGE_SIZE
            ) * S1M_V3_PAGE_SIZE
            for idx, (tname, arr) in enumerate(arrays_ordered):
                align = (
                    S1M_V3_PAGE_SIZE
                    if (idx == 0 or arr.nbytes >= 65536 or tname == "_projector_dense_table")
                    else 64
                )
                cursor = ((cursor + align - 1) // align) * align
                manifest[tname]["offset"] = int(cursor)
                cursor += int(arr.nbytes)
            header_meta["tensors"] = manifest
            header_bytes = json.dumps(header_meta, sort_keys=True, allow_nan=False).encode("utf-8")

        total_len = ((cursor + S1M_V3_PAGE_SIZE - 1) // S1M_V3_PAGE_SIZE) * S1M_V3_PAGE_SIZE
        if total_len > MAX_SKILL_BYTES_V3:
            raise ValueError("Saved v3 skill file exceeds size limit")

        out = bytearray(total_len)
        _V3_HDR_STRUCT.pack_into(out, 0, MAGIC_HEADER_V3, len(header_bytes))
        out[16 : 16 + len(header_bytes)] = header_bytes
        for tname, arr in arrays_ordered:
            spec = manifest[tname]
            nbytes = int(spec["nbytes"])
            if nbytes > 0:
                offset = int(spec["offset"])
                if spec.get("order") == "F":
                    out[offset : offset + nbytes] = arr.tobytes(order="F")
                else:
                    out[offset : offset + nbytes] = np.ascontiguousarray(arr).tobytes()
        return bytes(out)

    def to_bytes(
        self,
        include_covariance: bool = False,
        *,
        compact: Optional[bool] = None,
        format_version: int = 2,
        embed_vocab: bool = False,
    ) -> bytes:
        """Serializes the compiled model into .s1m binary bytes (v2 compact zip or v3 page-aligned mmap).

        By default (format_version=2), excludes covariance matrices and emits S1M\\x02
        bytes (<20 KB for typical skills). Set format_version=3 for S1M\\x03 page-aligned
        zero-copy mmap storage supporting up to 512 fields and optional embed_vocab.
        """
        with self._lock:
            if format_version == 3:
                return self._to_bytes_v3(
                    include_covariance=include_covariance,
                    compact=compact,
                    embed_vocab=embed_vocab,
                )
            header_meta, arrays_to_save = self._build_export_header_and_arrays(
                include_covariance=include_covariance,
                compact=compact,
                format_version=format_version,
                embed_vocab=False,
            )
            header_bytes = json.dumps(header_meta, sort_keys=True).encode("utf-8")
            header_len = len(header_bytes)
            if header_len > MAX_SKILL_HEADER_BYTES:
                raise ValueError("Saved skill header exceeds size limit")

            npz_buf = io.BytesIO()
            np.savez_compressed(npz_buf, **arrays_to_save)
            npz_bytes = npz_buf.getvalue()

            # Wire format: MAGIC (4 bytes) + uint32(header_len) + header_bytes + npz_bytes
            out = bytearray()
            out.extend(MAGIC_HEADER)
            out.extend(struct.pack(">I", header_len))
            out.extend(header_bytes)
            out.extend(npz_bytes)
            return bytes(out)

    def save(
        self,
        path: Union[str, Path],
        include_covariance: bool = False,
        *,
        compact: Optional[bool] = None,
        format_version: int = 2,
        embed_vocab: bool = False,
    ) -> None:
        """Saves compiled model to a .s1m binary file on disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            self.to_bytes(
                include_covariance=include_covariance,
                compact=compact,
                format_version=format_version,
                embed_vocab=embed_vocab,
            )
        )

    @staticmethod
    def _coerce_readonly_float_array(arr: Optional[np.ndarray], dtype: Any) -> Optional[np.ndarray]:
        """Convert array to target runtime float dtype without overflow warnings and mark read-only."""
        if arr is None:
            return None
        target = np.dtype(dtype)
        if arr.dtype == target:
            return arr
        with np.errstate(over="ignore", invalid="ignore"):
            out = np.asarray(arr).astype(target, order="K", copy=False)
        out.setflags(write=False)
        return out

    @staticmethod
    def _validate_hybrid_projector_config_and_tensors(
        projector_config: Mapping[str, Any],
        array_map: Mapping[str, np.ndarray],
        dimension: int,
    ) -> Tuple[
        int,
        Optional[TfidfProjector],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Validate HybridProjector config scalars, TF-IDF header, and tensor shapes before reconstruction."""
        from system1.core.embeddings import HybridProjector

        if not isinstance(projector_config, Mapping):
            raise ValueError("Saved hybrid projector config must be a mapping")
        if set(projector_config) - HybridProjector.CONFIG_KEYS:
            raise ValueError("Saved hybrid projector contains unsupported settings")

        sparse = projector_config.get("sparse_dim")
        dense = projector_config.get("dense_dim")
        if type(sparse) is not int or type(dense) is not int or sparse <= 0 or dense <= 0:
            raise ValueError("Saved hybrid projector dimensions are invalid")

        def _req_num(val: Any, name: str) -> float:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"Saved hybrid projector {name} must be finite")
            fval = float(val)
            if not math.isfinite(fval):
                raise ValueError(f"Saved hybrid projector {name} must be finite")
            return fval

        def _req_bool(key: str, default: bool = False) -> bool:
            if key not in projector_config:
                return default
            val = projector_config[key]
            if type(val) is not bool:
                raise ValueError(f"Saved hybrid projector {key} must be a boolean")
            return val

        alpha = _req_num(projector_config.get("alpha", 0.5), "alpha")
        if not (0.0 < alpha < 1.0):
            raise ValueError("Saved hybrid projector alpha must be in (0, 1)")
        seed = projector_config.get("seed", 42)
        if type(seed) is not int:
            raise ValueError("Saved hybrid projector seed must be an integer")
        if _req_num(projector_config.get("svd_power", 0.5), "svd_power") < 0.0:
            raise ValueError("Saved hybrid projector svd_power must be nonnegative")
        spectral_blend = _req_num(projector_config.get("spectral_blend", 0.35), "spectral_blend")
        if not (0.0 <= spectral_blend <= 1.0):
            raise ValueError("Saved hybrid projector spectral_blend must be in [0, 1]")
        subspace_weight = _req_num(projector_config.get("subspace_weight", 0.0), "subspace_weight")
        if not (0.0 <= subspace_weight <= 1.0):
            raise ValueError("Saved hybrid projector subspace_weight must be in [0, 1]")
        if _req_num(projector_config.get("context_smooth", 1.0), "context_smooth") < 0.0:
            raise ValueError("Saved hybrid projector context_smooth must be nonnegative")
        if _req_num(projector_config.get("rbf_gamma", 3.0), "rbf_gamma") <= 0.0:
            raise ValueError("Saved hybrid projector rbf_gamma must be positive")
        rbf_weight = _req_num(projector_config.get("rbf_weight", 0.0), "rbf_weight")
        if not (0.0 <= rbf_weight < 1.0):
            raise ValueError("Saved hybrid projector rbf_weight must be in [0, 1)")

        fitted = _req_bool("fitted", False)
        _req_bool("use_subspace", False)
        direct_spectral = _req_bool("direct_spectral", False)
        raw_ds_dim = projector_config.get("direct_spectral_dim", 0)
        if type(raw_ds_dim) is not int or not (0 <= raw_ds_dim <= 1024):
            raise ValueError("Saved hybrid projector direct_spectral_dim is invalid")

        _HYBRID_TFIDF_KEY_MAP = (
            ("tfidf_use_stems", "use_stems"),
            ("tfidf_stem_weight", "stem_weight"),
            ("tfidf_use_skip_bigrams", "use_skip_bigrams"),
            ("tfidf_skip_weight", "skip_weight"),
            ("tfidf_word_quota_frac", "word_quota_frac"),
            ("tfidf_char_weight", "char_weight"),
        )
        _HYBRID_ALL_TFIDF_KEYS = {
            "tfidf_terms",
            "tfidf_idf",
            "tfidf_char_ngrams",
            *(src_k for src_k, _ in _HYBRID_TFIDF_KEY_MAP),
        }
        has_any_tfidf_cfg = bool(set(projector_config.keys()) & _HYBRID_ALL_TFIDF_KEYS)
        has_hybrid_idf_arr = "_projector_hybrid_tfidf_idf" in array_map
        top_tfidf: Optional[TfidfProjector] = None
        if not fitted:
            if has_any_tfidf_cfg or has_hybrid_idf_arr:
                raise ValueError("Unfitted saved hybrid projector cannot carry TF-IDF state")
        elif "tfidf_terms" not in projector_config:
            if has_any_tfidf_cfg or has_hybrid_idf_arr:
                raise ValueError("Saved hybrid projector has incomplete TF-IDF configuration")
        else:
            tf_terms = projector_config["tfidf_terms"]
            if not isinstance(tf_terms, (list, tuple)) or not 0 < len(tf_terms) <= sparse:
                raise ValueError("Saved hybrid TF-IDF vocabulary exceeds sparse_dim")
            top_tf_cfg: Dict[str, Any] = {"type": "tfidf", "terms": list(tf_terms)}
            if "tfidf_char_ngrams" in projector_config:
                top_tf_cfg["char_ngrams"] = projector_config["tfidf_char_ngrams"]
            for src_k, dst_k in _HYBRID_TFIDF_KEY_MAP:
                if src_k in projector_config:
                    top_tf_cfg[dst_k] = projector_config[src_k]
            if len(tf_terms) > DEFAULT_MAX_FEATURES:
                top_tf_cfg["max_features"] = len(tf_terms)
            bin_idf = array_map.get("_projector_hybrid_tfidf_idf")
            if "tfidf_idf" in projector_config:
                top_tf_cfg["idf"] = projector_config["tfidf_idf"]
                TfidfProjector.validate_header_config(top_tf_cfg, allow_missing_idf=False)
                top_tfidf = TfidfProjector.from_config(top_tf_cfg)
                if bin_idf is not None:
                    if (
                        bin_idf.ndim != 1
                        or len(bin_idf) != len(tf_terms)
                        or not np.all(np.isfinite(bin_idf))
                        or not np.allclose(
                            np.asarray(top_tfidf._idf, dtype=np.float64),
                            bin_idf,
                            rtol=0.0,
                            atol=1e-5,
                        )
                    ):
                        raise ValueError(
                            "Saved hybrid projector inline tfidf_idf mismatches binary IDF tensor"
                        )
            else:
                if bin_idf is None:
                    raise ValueError("Saved hybrid projector missing TF-IDF idf weights")
                TfidfProjector.validate_header_config(top_tf_cfg, allow_missing_idf=True)
                top_tfidf = TfidfProjector.from_config(top_tf_cfg, idf_array=bin_idf)

        raw_terms = projector_config.get("spectral_terms", ())
        raw_weights = projector_config.get("spectral_weights", ())
        if not isinstance(raw_terms, (list, tuple)) or not isinstance(raw_weights, (list, tuple)):
            raise ValueError("Saved hybrid projector spectral terms and weights must be sequences")
        terms: List[str] = []
        for tm in raw_terms:
            if not isinstance(tm, str) or not tm.strip():
                raise ValueError("Saved hybrid spectral terms must be unique nonempty strings")
            terms.append(tm)
        if len(set(terms)) != len(terms):
            raise ValueError("Saved hybrid spectral terms must be unique nonempty strings")
        if len(terms) > 2048:
            raise ValueError("Saved hybrid spectral terms exceed maximum allowed size")
        for w_val in raw_weights:
            _req_num(w_val, "spectral_weights")
        if len(terms) != len(raw_weights):
            raise ValueError("Saved hybrid projector spectral terms and weights length mismatch")
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                spec_w_f32 = np.asarray([float(w_val) for w_val in raw_weights], dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Saved hybrid projector spectral weights are invalid") from exc
        if not bool(np.all(np.isfinite(spec_w_f32))):
            raise ValueError("Saved hybrid projector spectral weights contain non-finite values")

        if len(terms) > 2048:
            raise ValueError("Saved hybrid projector spectral_terms exceeds limit")

        def _parse_inline_matrix_f32(raw_val: Any, cfg_key: str) -> np.ndarray:
            if not isinstance(raw_val, (list, tuple)) or len(raw_val) == 0:
                raise ValueError(f"Saved hybrid projector {cfg_key} is invalid")
            if len(raw_val) > 2048:
                raise ValueError(f"Saved hybrid projector {cfg_key} exceeds row limit")
            row_len: Optional[int] = None
            for row in raw_val:
                if not isinstance(row, (list, tuple)) or len(row) == 0:
                    raise ValueError(f"Saved hybrid projector {cfg_key} is invalid")
                if row_len is None:
                    row_len = len(row)
                    if row_len > MAX_SKILL_DIMENSION:
                        raise ValueError(f"Saved hybrid projector {cfg_key} exceeds dimension limit")
                elif len(row) != row_len:
                    raise ValueError(f"Saved hybrid projector {cfg_key} is ragged")
                for elem in row:
                    if isinstance(elem, bool) or not isinstance(elem, (int, float)):
                        raise ValueError(f"Saved hybrid projector {cfg_key} contains invalid element")
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    arr = np.asarray(raw_val, dtype=np.float32)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Saved hybrid projector {cfg_key} is invalid") from exc
            if arr.ndim != 2 or not bool(np.all(np.isfinite(arr))):
                raise ValueError(f"Saved hybrid projector {cfg_key} contains non-finite values")
            arr.setflags(write=False)
            return arr

        def _inline_or_map_f32(map_key: str, cfg_key: str) -> Optional[np.ndarray]:
            bin_raw = array_map.get(map_key)
            bin_arr = (
                CompiledSystemOneModel._coerce_readonly_float_array(bin_raw, np.float32)
                if bin_raw is not None
                else None
            )
            inline_arr: Optional[np.ndarray] = None
            if cfg_key in projector_config and projector_config[cfg_key] is not None:
                inline_arr = _parse_inline_matrix_f32(projector_config[cfg_key], cfg_key)
            if bin_arr is not None:
                if inline_arr is not None and (
                    inline_arr.shape != bin_arr.shape
                    or not np.allclose(inline_arr, bin_arr, rtol=0.0, atol=1e-4)
                ):
                    raise ValueError(
                        f"Saved hybrid projector inline {cfg_key} mismatches binary tensor"
                    )
                return bin_arr
            return inline_arr

        spec_arr = _inline_or_map_f32("_projector_spectral_vectors", "spectral_vectors")
        basis_arr = CompiledSystemOneModel._coerce_readonly_float_array(
            array_map.get("_projector_spectral_basis"), np.float32
        )
        sub_arr = _inline_or_map_f32("_projector_subspace_components", "subspace_components")
        rbf_arr = _inline_or_map_f32("_projector_rbf_landmarks", "rbf_landmarks")

        if not fitted:
            if (
                terms
                or len(raw_weights) > 0
                or (spec_arr is not None and len(spec_arr) > 0)
                or basis_arr is not None
                or (sub_arr is not None and len(sub_arr) > 0)
                or (rbf_arr is not None and len(rbf_arr) > 0)
                or (direct_spectral and raw_ds_dim != 0)
            ):
                raise ValueError("Unfitted saved hybrid projector cannot carry fitted tensors")

        if spec_arr is not None and len(spec_arr) > 0:
            if spec_arr.ndim != 2 or spec_arr.shape[0] != len(terms) or len(terms) == 0:
                raise ValueError("Saved hybrid projector spectral_vectors have invalid shape")
            if direct_spectral:
                if (
                    not (1 <= spec_arr.shape[1] <= 1024)
                    or (raw_ds_dim != 0 and spec_arr.shape[1] != raw_ds_dim)
                    or basis_arr is not None
                ):
                    raise ValueError("Saved hybrid projector direct spectral_vectors have invalid shape")
                eff_spec_dim = int(spec_arr.shape[1])
            else:
                if not (1 <= spec_arr.shape[1] <= dense):
                    raise ValueError("Saved hybrid projector spectral_vectors have invalid shape")
                if basis_arr is not None and (
                    basis_arr.ndim != 2 or basis_arr.shape != (spec_arr.shape[1], dense)
                ):
                    raise ValueError("Saved hybrid projector _projector_spectral_basis shape mismatch")
                eff_spec_dim = 0
        else:
            if terms or basis_arr is not None or (direct_spectral and raw_ds_dim != 0):
                raise ValueError("Saved hybrid projector missing spectral_vectors")
            eff_spec_dim = 0

        base_dim = sparse + dense + eff_spec_dim
        if sub_arr is not None and len(sub_arr) > 0:
            if sub_arr.ndim != 2 or not (1 <= sub_arr.shape[0] <= base_dim) or sub_arr.shape[1] != base_dim:
                raise ValueError("Saved hybrid projector subspace_components have invalid shape")
        rbf_rows = 0
        if rbf_arr is not None and len(rbf_arr) > 0:
            if (
                rbf_arr.ndim != 2
                or not (1 <= rbf_arr.shape[0] <= MAX_SKILL_OPTIONS)
                or rbf_arr.shape[1] != base_dim
            ):
                raise ValueError("Saved hybrid projector rbf_landmarks have invalid shape")
            eff_rbf_weight = _req_num(projector_config.get("rbf_weight", 0.14), "rbf_weight")
            if not (0.0 < eff_rbf_weight < 1.0):
                raise ValueError(
                    "Saved hybrid projector rbf_weight must be in (0, 1) when rbf_landmarks are present"
                )
            rbf_rows = int(rbf_arr.shape[0])

        if base_dim + rbf_rows != dimension:
            raise ValueError("Saved hybrid projector dimensions do not match the skill")
        return eff_spec_dim, top_tfidf, spec_arr, sub_arr, rbf_arr

    @classmethod
    def _reconstruct_model_from_parts(
        cls,
        meta: Mapping[str, Any],
        array_map: Mapping[str, np.ndarray],
        *,
        projector: Optional[Any],
        backend: str,
        schema: DecisionSchema,
        dimension: int,
        projector_config: Optional[Mapping[str, Any]],
        saved_tfidf: Optional[Any],
        saved_prag: Optional[Any],
        format_version: int,
    ) -> CompiledSystemOneModel:
        from system1.core.embeddings import HybridProjector, SubwordSemanticEmbeddings

        def _as_f32(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
            return cls._coerce_readonly_float_array(arr, np.float32)

        def _as_f64(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
            return cls._coerce_readonly_float_array(arr, np.float64)

        array_map = dict(array_map)
        validated_layouts, _ = _validate_tensor_layouts_header(
            meta,
            {k: tuple(np.asarray(v).shape) for k, v in array_map.items()},
        )

        def _restore_f32_layout(tname: str, arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
            out_arr = _as_f32(arr)
            if out_arr is not None and tname in validated_layouts:
                if not bool(np.all(np.isfinite(out_arr))):
                    raise ValueError(f"Saved tensor {tname!r} contains non-finite values")
                mod64, s0, s1 = validated_layouts[tname]
                out_arr = _reconstruct_2d_layout_array(
                    out_arr, np.array([mod64, s0, s1], dtype=np.int64)
                )
            return out_arr

        is_v3 = format_version == 3
        embed_vocab_flag = bool(meta.get("embed_vocab", False)) if is_v3 else False

        if (projector_config is None or projector_config.get("type") != "tfidf") and (
            "_projector_tfidf_idf" in array_map
        ):
            raise ValueError("Saved skill contains unexpected TF-IDF idf array")
        if (projector_config is None or projector_config.get("type") != "pragmatic_tfidf") and (
            "_projector_prag_base_idf" in array_map or "_projector_prag_idf" in array_map
        ):
            raise ValueError("Saved skill contains unexpected pragmatic TF-IDF idf array")
        if (projector_config is None or projector_config.get("type") != "hybrid") and (
            "_projector_spectral_vectors" in array_map
            or "_projector_spectral_basis" in array_map
            or "_projector_subspace_components" in array_map
            or "_projector_rbf_landmarks" in array_map
            or "_projector_hybrid_tfidf_idf" in array_map
            or "_projector_dense_table" in array_map
        ):
            raise ValueError("Saved skill contains unexpected hybrid projector arrays")
        if (not is_v3 or not embed_vocab_flag) and (
            "_projector_dense_table" in array_map or "_projector_spectral_basis" in array_map
        ):
            raise ValueError("Saved skill contains unexpected dense vocab projector arrays")
        if (
            not is_v3
            or not embed_vocab_flag
            or projector_config is None
            or projector_config.get("type") != "hybrid"
        ) and "dense_embed_vocab" in meta:
            raise ValueError("Saved skill contains unexpected dense_embed_vocab metadata")

        for proj_arr_key in (
            "_projector_tfidf_idf",
            "_projector_prag_base_idf",
            "_projector_prag_idf",
            "_projector_hybrid_tfidf_idf",
        ):
            if proj_arr_key in array_map:
                conv_f64 = _as_f64(array_map[proj_arr_key])
                assert conv_f64 is not None
                if not np.all(np.isfinite(conv_f64)):
                    raise ValueError(f"Saved projector array {proj_arr_key!r} contains non-finite values")
                array_map[proj_arr_key] = conv_f64

        for proj_arr_key in (
            "_projector_spectral_vectors",
            "_projector_spectral_basis",
            "_projector_subspace_components",
            "_projector_rbf_landmarks",
            "_projector_dense_table",
        ):
            if proj_arr_key in array_map:
                conv_f32 = _restore_f32_layout(proj_arr_key, array_map[proj_arr_key])
                assert conv_f32 is not None
                if not np.all(np.isfinite(conv_f32)):
                    raise ValueError(f"Saved projector array {proj_arr_key!r} contains non-finite values")
                array_map[proj_arr_key] = conv_f32

        if projector_config is not None and projector_config["type"] == "tfidf" and saved_tfidf is None:
            saved_tfidf = TfidfProjector.from_config(
                projector_config,
                idf_array=array_map.get("_projector_tfidf_idf"),
            )
        if projector_config is not None and projector_config["type"] == "pragmatic_tfidf" and saved_prag is None:
            saved_prag = PragmaticCompositeProjector.from_config(
                projector_config,
                tfidf_idf_array=array_map.get("_projector_prag_base_idf"),
                prag_idf_array=array_map.get("_projector_prag_idf"),
            )

        saved_hybrid = None
        if projector_config is not None and projector_config["type"] == "hybrid":
            eff_spec_dim, top_tfidf, spec_arr, sub_arr, rbf_arr = (
                cls._validate_hybrid_projector_config_and_tensors(
                    projector_config,
                    array_map,
                    dimension,
                )
            )

            if is_v3 and embed_vocab_flag:
                if "_projector_dense_table" not in array_map:
                    raise ValueError(
                        "Saved v3 skill with embed_vocab=True missing _projector_dense_table"
                    )
                vocab_meta = meta.get("dense_embed_vocab")
                if not isinstance(vocab_meta, Mapping):
                    raise ValueError("Saved v3 skill missing dense_embed_vocab metadata")
                if set(vocab_meta) - {"vocab_size", "seed", "explicit_alpha", "multiword_by_space3", "sparse"}:
                    raise ValueError("Saved v3 skill dense_embed_vocab contains unsupported settings")
                vocab_size = vocab_meta.get("vocab_size")
                if type(vocab_size) is not int or not 1 <= vocab_size <= 250_000:
                    raise ValueError("Saved v3 skill dense_embed_vocab vocab_size is invalid")
                dense_dim = int(projector_config["dense_dim"])
                dense_table = array_map["_projector_dense_table"]
                if dense_table.ndim != 2 or dense_table.shape != (vocab_size, dense_dim):
                    raise ValueError("Saved v3 skill _projector_dense_table shape mismatch")

                raw_mw = vocab_meta.get("multiword_by_space3", {})
                if not isinstance(raw_mw, Mapping):
                    raise ValueError("Saved v3 skill multiword_by_space3 must be a mapping")
                mw_map: Dict[str, Tuple[Tuple[str, bool, int], ...]] = {}
                for k_str, entries in raw_mw.items():
                    if not isinstance(k_str, str) or not isinstance(entries, (list, tuple)):
                        raise ValueError("Saved v3 skill multiword_by_space3 entry is invalid")
                    parsed_entries: List[Tuple[str, bool, int]] = []
                    for entry in entries:
                        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                            raise ValueError("Saved v3 skill multiword_by_space3 tuple is invalid")
                        term_s, neg_f, tok_idx = entry
                        if (
                            not isinstance(term_s, str)
                            or type(neg_f) is not bool
                            or type(tok_idx) is not int
                            or not 0 <= tok_idx < vocab_size
                        ):
                            raise ValueError("Saved v3 skill multiword_by_space3 item out of bounds")
                        parsed_entries.append((term_s, neg_f, tok_idx))
                    mw_map[k_str] = tuple(parsed_entries)

                sparse_cfg = vocab_meta.get("sparse")
                tfidf_proj: Optional[Any] = None
                if top_tfidf is None and sparse_cfg is not None:
                    raise ValueError(
                        "Saved v3 skill sparse projector config mismatches top-level hybrid header"
                    )
                if top_tfidf is not None and sparse_cfg is None:
                    raise ValueError(
                        "Saved v3 skill missing sparse projector config for fitted hybrid TF-IDF"
                    )
                if sparse_cfg is not None:
                    if not isinstance(sparse_cfg, Mapping):
                        raise ValueError("Saved v3 skill sparse projector config is invalid")
                    TfidfProjector.validate_header_config(sparse_cfg, allow_missing_idf=True)
                    tfidf_proj = TfidfProjector.from_config(
                        sparse_cfg,
                        idf_array=array_map.get("_projector_hybrid_tfidf_idf"),
                    )
                    if tfidf_proj.dimension > int(projector_config["sparse_dim"]):
                        raise ValueError("Saved hybrid TF-IDF vocabulary exceeds sparse_dim")
                    if top_tfidf is not None and (
                        tfidf_proj.to_config(include_arrays=True)
                        != top_tfidf.to_config(include_arrays=True)
                    ):
                        raise ValueError(
                            "Saved v3 skill sparse projector config mismatches top-level hybrid header"
                        )

                dense_engine = object.__new__(SubwordSemanticEmbeddings)
                dense_engine.dimension = dense_dim
                dense_engine.vocab_size = vocab_size
                vm_seed = vocab_meta.get("seed", projector_config.get("seed", 42))
                if type(vm_seed) is not int or vm_seed != int(projector_config.get("seed", 42)):
                    raise ValueError("Saved v3 skill dense_embed_vocab seed mismatches hybrid projector seed")
                dense_engine.seed = vm_seed
                dense_engine._table = dense_table
                dense_engine._multiword_by_space3 = mw_map

                hp = object.__new__(HybridProjector)
                hp.sparse_dim = int(projector_config["sparse_dim"])
                hp.dense_dim = dense_dim
                hp.dimension = int(dimension)
                hp.alpha = float(projector_config.get("alpha", 0.5))
                hp._explicit_alpha = _strict_header_bool(
                    vocab_meta, "explicit_alpha", True, "Saved v3 skill dense_embed_vocab"
                )
                hp.seed = int(projector_config.get("seed", 42))
                hp.svd_power = float(projector_config.get("svd_power", 0.5))
                hp.spectral_blend = float(projector_config.get("spectral_blend", 0.35))
                hp.backend = "numpy"
                hp._dense_engine = dense_engine
                hp._fitted = bool(projector_config.get("fitted", False))
                hp._tfidf_projector = tfidf_proj
                spec_terms = tuple(str(t) for t in projector_config.get("spectral_terms", ()))
                spec_weights = tuple(float(w) for w in projector_config.get("spectral_weights", ()))
                hp._spectral_terms = spec_terms
                hp._spectral_vocab = {t: idx for idx, t in enumerate(spec_terms)}
                hp._spectral_weights = spec_weights
                with np.errstate(over="ignore", invalid="ignore"):
                    spec_w_arr = (
                        np.asarray(spec_weights, dtype=np.float32)
                        if spec_weights
                        else np.empty(0, dtype=np.float32)
                    )
                if not bool(np.all(np.isfinite(spec_w_arr))):
                    raise ValueError("Saved hybrid projector spectral weights contain non-finite values")
                hp._spectral_weights_arr = spec_w_arr
                hp._spectral_vectors = _as_f32(spec_arr)
                hp._spectral_basis = _as_f32(array_map.get("_projector_spectral_basis"))
                hp._subspace_components = _as_f32(sub_arr)
                hp._subspace_weight = float(
                    projector_config.get("subspace_weight", 0.20 if sub_arr is not None else 0.0)
                )
                hp._use_subspace = bool(
                    projector_config.get("use_subspace", hp._subspace_components is not None)
                )
                hp._direct_spectral = bool(projector_config.get("direct_spectral", False))
                hp._direct_spectral_dim = eff_spec_dim
                if (
                    hp._spectral_vectors is not None
                    and hp._spectral_vectors.size > 0
                    and not hp._direct_spectral
                    and hp._spectral_basis is None
                ):
                    vocab_dense = dense_engine.encode_batch(list(spec_terms)).astype(np.float32)
                    aligned_basis = (hp._spectral_vectors.T @ vocab_dense).astype(np.float32)
                    b_norms = np.linalg.norm(aligned_basis, axis=1, keepdims=True)
                    if np.all(b_norms.flatten() > 1e-6):
                        hp._spectral_basis = (aligned_basis / b_norms).astype(np.float32)
                    else:
                        hp._spectral_basis = hp._build_spectral_basis(hp._spectral_vectors.shape[1])
                hp._context_smooth = float(projector_config.get("context_smooth", 1.0))
                hp._rbf_landmarks = _as_f32(rbf_arr)
                hp._rbf_gamma = float(projector_config.get("rbf_gamma", 3.0))
                hp._rbf_weight = float(projector_config.get("rbf_weight", 0.14)) if rbf_arr is not None else 0.0
                hp._digest = None
                hp._fit_base_cache = {}
                saved_hybrid = hp
            else:
                hyb_kwargs: Dict[str, Any] = {
                    "backend": backend,
                    "spectral_vectors": _as_f32(spec_arr),
                    "subspace_components": _as_f32(sub_arr),
                }
                if "_projector_hybrid_tfidf_idf" in array_map:
                    hyb_kwargs["tfidf_idf_array"] = array_map["_projector_hybrid_tfidf_idf"]
                if rbf_arr is not None:
                    hyb_kwargs["rbf_landmarks"] = _as_f32(rbf_arr)
                try:
                    saved_hybrid = HybridProjector.from_config(projector_config, **hyb_kwargs)
                except TypeError:
                    hyb_kwargs.pop("tfidf_idf_array", None)
                    hyb_kwargs.pop("rbf_landmarks", None)
                    saved_hybrid = HybridProjector.from_config(projector_config, **hyb_kwargs)

        if projector_config is not None:
            _validate_projector_header_strict(projector_config, dimension)
            if projector is not None:
                if projector_config["type"] == "hybrid" and saved_hybrid is not None:
                    if (
                        not isinstance(projector, HybridProjector)
                        or projector.projector_digest() != saved_hybrid.projector_digest()
                    ):
                        raise ValueError("Supplied projector does not match the saved skill's feature space")
                elif projector_config["type"] == "tfidf" and saved_tfidf is not None:
                    if (
                        type(projector) is not TfidfProjector
                        or _projector_config(projector, include_arrays=True)
                        != _projector_config(saved_tfidf, include_arrays=True)
                    ):
                        raise ValueError("Supplied projector does not match the saved skill's feature space")
                elif projector_config["type"] == "pragmatic_tfidf" and saved_prag is not None:
                    if (
                        not isinstance(projector, PragmaticCompositeProjector)
                        or projector.projector_digest() != saved_prag.projector_digest()
                    ):
                        raise ValueError("Supplied projector does not match the saved skill's feature space")
                elif _projector_config(projector) != projector_config:
                    raise ValueError("Supplied projector does not match the saved skill's feature space")
            elif projector_config["type"] == "deterministic":
                det_rw = _strict_header_bool(
                    projector_config, "recency_weighted", False, "Saved deterministic projector"
                )
                if type(projector_config.get("dimension")) is not int or projector_config["dimension"] != dimension:
                    raise ValueError("Saved projector dimension does not match the skill")
                projector = DeterministicSemanticProjector(
                    dimension=dimension,
                    recency_weighted=det_rw,
                )
            elif saved_hybrid is not None:
                projector = saved_hybrid
            elif saved_tfidf is not None:
                projector = saved_tfidf
            elif saved_prag is not None:
                projector = saved_prag
            else:
                raise ValueError("This skill requires its original external projector; pass projector= when loading")

        top_created_at = _strict_header_float(
            meta, "created_at", 0.0, "Saved skill header", min_val=0.0, min_inclusive=True
        )
        saved_meta = meta.get("metadata", {})
        if not isinstance(saved_meta, dict):
            raise ValueError("Saved skill metadata must be a mapping")
        is_cold_skill = _strict_header_bool(saved_meta, "cold_start", False, "Saved skill metadata")
        meta_ff = _strict_header_float(
            saved_meta,
            "forgetting_factor",
            0.995,
            "Saved skill metadata",
            min_val=0.0,
            min_inclusive=False,
            max_val=1.0,
            max_inclusive=True,
        )
        meta_rw = _strict_header_bool(saved_meta, "recency_weighted", False, "Saved skill metadata")
        meta_created_at = _strict_header_float(
            saved_meta, "created_at", top_created_at, "Saved skill metadata", min_val=0.0, min_inclusive=True
        )
        _strict_header_bool(saved_meta, "strict_mode", True, "Saved skill metadata")
        _strict_header_float(
            saved_meta,
            "default_alpha",
            0.05,
            "Saved skill metadata",
            min_val=0.0,
            min_inclusive=False,
            max_val=1.0,
            max_inclusive=False,
        )

        heads: Dict[str, CompiledHeadWeights] = {}
        namespaced_tensors = _resolve_tensor_namespaced_flag(schema, meta)
        for name, h_info in meta["heads"].items():
            def _hk(suffix: str, _fn: str = name) -> str:
                return _head_tensor_name(_fn, suffix, namespaced=namespaced_tensors)

            if not isinstance(h_info, Mapping):
                raise ValueError(f"Saved head {name!r} metadata must be a mapping")
            ctx = f"Saved head {name!r}"
            if name not in schema.fields or h_info.get("field_type") != schema.fields[name].field_type:
                raise ValueError(f"Saved head {name!r} does not match the schema")
            definition = schema.fields[name]
            if _hk("w") not in array_map or _hk("b") not in array_map:
                raise ValueError(f"Saved head {name!r} is missing required weights")
            w = _restore_f32_layout(_hk("w"), array_map[_hk("w")])
            assert w is not None
            b = _as_f32(array_map[_hk("b")])
            assert b is not None
            widths = (
                (1, 2)
                if isinstance(definition, (BooleanField, ScoreField))
                else (len(definition.options),)
                if hasattr(definition, "options")
                else (1,)
            )
            valid_b_shapes = (
                ((1,), (w.shape[0],))
                if isinstance(definition, (BooleanField, ScoreField))
                else ((w.shape[0],),)
            )
            if (
                w.ndim != 2
                or w.shape[0] not in widths
                or w.shape[1] != dimension
                or b.shape not in valid_b_shapes
            ):
                raise ValueError(f"Saved head {name!r} has incompatible weight dimensions")
            if not np.all(np.isfinite(w)) or not np.all(np.isfinite(b)):
                raise ValueError(f"Saved head {name!r} contains non-finite weights")
            temperature = _strict_header_float(
                h_info, "temperature", 1.0, ctx, min_val=0.0, min_inclusive=False
            )
            if h_info.get("score_method", "aps") not in ("aps", "lac"):
                raise ValueError(f"Saved head {name!r} uses an unsupported conformal score")
            if isinstance(definition, (ChoiceField, MultiChoiceField)) and tuple(h_info.get("options", ())) != definition.options:
                raise ValueError(f"Saved head {name!r} options do not match the schema")
            p_mat = _restore_f32_layout(_hk("P"), array_map[_hk("P")]) if _hk("P") in array_map else None
            b_mat = _restore_f32_layout(_hk("B"), array_map[_hk("B")]) if _hk("B") in array_map else None
            if any(a is not None and not np.all(np.isfinite(a)) for a in (p_mat, b_mat)):
                raise ValueError(f"Saved head {name!r} contains non-finite covariance")
            calib_scores_arr = array_map[_hk("calib_scores")] if _hk("calib_scores") in array_map else None
            calib_scores = tuple(float(s) for s in calib_scores_arr) if calib_scores_arr is not None else ()
            if any(not math.isfinite(score) or score < 0 for score in calib_scores):
                raise ValueError(f"Saved head {name!r} contains invalid calibration scores")
            logit_adj_arr = _validate_compiled_logit_adj(
                array_map.get(_hk("logit_adj")),
                definition,
                name,
            )
            raw_class_scores = h_info.get("class_calibration_scores", {})
            parsed_class_scores: Dict[str, Tuple[float, ...]] = {}
            if raw_class_scores is not None:
                if not isinstance(raw_class_scores, dict):
                    raise ValueError(f"Saved head {name!r} class_calibration_scores must be a mapping")
                allowed_classes = (
                    set(definition.options)
                    if isinstance(definition, (ChoiceField, MultiChoiceField))
                    else ({"False", "True"} if isinstance(definition, BooleanField) else set())
                )
                total_class_scores = 0
                for c_key, c_vals in raw_class_scores.items():
                    if str(c_key) not in allowed_classes or not isinstance(c_vals, (list, tuple)):
                        raise ValueError(f"Saved head {name!r} has invalid class_calibration_scores entry {c_key!r}")
                    total_class_scores += len(c_vals)
                    if total_class_scores > MAX_SKILL_CALIBRATION_SCORES:
                        raise ValueError(f"Saved head {name!r} class_calibration_scores exceeds limit")
                    parsed_vals: List[float] = []
                    for v in c_vals:
                        if isinstance(v, bool) or not isinstance(v, (int, float)):
                            raise ValueError(f"Saved head {name!r} contains non-numeric class calibration score")
                        fv = float(v)
                        if not math.isfinite(fv) or fv < 0.0:
                            raise ValueError(f"Saved head {name!r} contains invalid class calibration score")
                        parsed_vals.append(fv)
                    parsed_class_scores[str(c_key)] = tuple(parsed_vals)
            fam_thresh = _strict_header_float(
                h_info, "familiarity_threshold", 0.0, ctx, min_val=0.0, min_inclusive=True
            )
            proto_thresh = _strict_header_float(
                h_info, "prototype_threshold", 0.0, ctx, min_val=0.0, min_inclusive=True
            )
            raw_proto_arr = array_map[_hk("prototypes")] if _hk("prototypes") in array_map else None
            proto_mat = _as_f32(raw_proto_arr)
            if proto_mat is not None and not np.all(np.isfinite(proto_mat)):
                raise ValueError(f"Saved head {name!r} contains non-finite prototypes")
            if proto_mat is not None and proto_mat.ndim == 2:
                proto_mat, _ = _normalize_rows_f32_safe(proto_mat)
            if (
                proto_mat is None
                and not is_cold_skill
                and "exemplar_state" not in h_info
                and isinstance(definition, (ChoiceField, BooleanField))
                and np.any(w)
            ):
                if w.shape[0] >= 2:
                    proto_mat, _ = _normalize_rows_f32_safe(w)
                elif isinstance(definition, BooleanField) and w.shape[0] == 1:
                    w_u, _, w_ok = _normalize_vec_f32_safe(w[0])
                    if not w_ok and np.any(w[0]):
                        w64 = np.asarray(w[0], dtype=np.float64)
                        w_norm = max(1e-12, float(np.linalg.norm(w64)))
                        w_u = (w64 / w_norm).astype(np.float32)
                    proto_mat = np.vstack([-w_u, w_u]).astype(np.float32)
            proto_scores_arr = array_map[_hk("proto_scores")] if _hk("proto_scores") in array_map else None
            proto_scores = tuple(float(s) for s in proto_scores_arr) if proto_scores_arr is not None else ()
            if any(not math.isfinite(score) or score < -1.0 - 1e-4 or score > 1.0 + 1e-4 for score in proto_scores):
                raise ValueError(f"Saved head {name!r} contains invalid prototype calibration scores")
            is_hier = _strict_header_bool(h_info, "hierarchical", False, ctx)
            raw_assign = h_info.get("cluster_assignments", ())
            if is_hier and (
                not isinstance(raw_assign, (list, tuple))
                or any(type(c) is not int for c in raw_assign)
            ):
                raise ValueError(f"Saved head {name!r} has invalid cluster_assignments")
            cluster_assign = tuple(int(c) for c in raw_assign) if isinstance(raw_assign, (list, tuple)) else ()
            coarse_w = _restore_f32_layout(_hk("coarse_w"), array_map[_hk("coarse_w")]) if _hk("coarse_w") in array_map else None
            coarse_b = _as_f32(array_map[_hk("coarse_b")]) if _hk("coarse_b") in array_map else None
            fine_w = _restore_f32_layout(_hk("fine_w"), array_map[_hk("fine_w")]) if _hk("fine_w") in array_map else None
            fine_b = _as_f32(array_map[_hk("fine_b")]) if _hk("fine_b") in array_map else None
            if any(a is not None and not np.all(np.isfinite(a)) for a in (coarse_w, coarse_b, fine_w, fine_b)):
                raise ValueError(f"Saved head {name!r} contains non-finite hierarchical weights")
            if is_hier:
                if coarse_w is None or coarse_b is None or len(cluster_assign) != w.shape[0]:
                    raise ValueError(f"Saved head {name!r} has incomplete hierarchical routing configuration")
                if any(c < 0 or c >= coarse_w.shape[0] for c in cluster_assign):
                    raise ValueError(f"Saved head {name!r} has out-of-range cluster_assignments")

            exemplar_arrays = {
                suffix: _restore_f32_layout(_hk(suffix), array_map[_hk(suffix)])
                for suffix in EXEMPLAR_ARRAYS
                if _hk(suffix) in array_map
            }
            for ex_arr in exemplar_arrays.values():
                if ex_arr is None or not np.all(np.isfinite(ex_arr)):
                    raise ValueError(f"Saved head {name!r} contains non-finite exemplar arrays")
            if "exemplar_state" in h_info:
                exemplar_meta = h_info["exemplar_state"]
                if not isinstance(exemplar_meta, Mapping) or (
                    set(exemplar_meta.keys()) - {"counts", "online_shots_are_pseudo"}
                ):
                    raise ValueError(f"Saved head {name!r} has an invalid exemplar_state header")
                exemplar_state: Optional[Dict[str, Any]] = {
                    **exemplar_meta,
                    "arrays": exemplar_arrays,
                }
            else:
                if exemplar_arrays:
                    raise ValueError(f"Saved head {name!r} has unexpected exemplar arrays without exemplar_state")
                exemplar_state = None

            cq_max = None if isinstance(definition, ScoreField) else (1.0 + 1e-6)
            conformal_q = _strict_header_float(
                h_info,
                "conformal_quantile",
                0.0,
                ctx,
                min_val=0.0,
                min_inclusive=True,
                max_val=cq_max,
                max_inclusive=True,
            )
            margin_t = _strict_header_float(
                h_info, "margin_threshold", 0.20, ctx, min_val=0.0, min_inclusive=True
            )
            head_ff = _strict_header_float(
                h_info,
                "forgetting_factor",
                0.995,
                ctx,
                min_val=0.0,
                min_inclusive=False,
                max_val=1.0,
                max_inclusive=True,
            )
            rel_odds = _strict_header_float(
                h_info, "relative_odds_ratio", 1.5, ctx, min_val=0.0, min_inclusive=False
            )
            conf_tau0 = _strict_header_float(
                h_info, "confidence_floor_tau0", 0.15, ctx, min_val=0.0, min_inclusive=True
            )
            esc_ambig = _strict_header_bool(h_info, "escalate_on_ambiguity", True, ctx)
            is_mondrian = _strict_header_bool(h_info, "mondrian", False, ctx)

            heads[name] = CompiledHeadWeights(
                field_name=name,
                field_type=h_info["field_type"],
                weights=w,
                biases=b,
                temperature=temperature,
                logit_adjustments=logit_adj_arr,
                conformal_quantile=conformal_q,
                score_method=h_info.get("score_method", "aps"),
                options=tuple(h_info.get("options", [])),
                P=p_mat,
                B=b_mat,
                margin_threshold=margin_t,
                forgetting_factor=head_ff,
                relative_odds_ratio=rel_odds,
                confidence_floor_tau0=conf_tau0,
                escalate_on_ambiguity=esc_ambig,
                calibration_scores=calib_scores,
                mondrian=is_mondrian,
                class_calibration_scores=parsed_class_scores,
                familiarity_threshold=fam_thresh,
                prototypes=proto_mat,
                prototype_threshold=proto_thresh,
                prototype_scores=proto_scores,
                hierarchical=is_hier,
                cluster_assignments=cluster_assign,
                coarse_weights=coarse_w,
                coarse_biases=coarse_b,
                fine_weights=fine_w,
                fine_biases=fine_b,
                exemplar_state=exemplar_state,
            )

        inst = cls(
            schema=schema,
            heads=heads,
            dimension=dimension,
            projector=projector,
            backend=backend,
            metadata=dict(saved_meta),
            forgetting_factor=meta_ff,
            recency_weighted=meta_rw,
        )
        inst.created_at = meta_created_at
        return inst

    @staticmethod
    def _validate_projector_and_budget(
        schema: DecisionSchema,
        dimension: int,
        projector_config: Optional[Mapping[str, Any]],
        *,
        cov_fields: int,
        max_runtime_bytes: int,
        extra_runtime_bytes: int = 0,
    ) -> Tuple[Optional[Any], Optional[Any], int]:
        """Validates projector configuration and estimates bounded runtime memory allocation."""
        widths = sum(max(2, len(getattr(f, "options", ()))) for f in schema.fields.values())
        runtime_bytes = (
            16 * cov_fields * (dimension + 1) ** 2
            + 16 * widths * (dimension + 1)
            + int(extra_runtime_bytes)
        )
        saved_tfidf = None
        saved_prag = None
        if projector_config is not None:
            _validate_projector_header_strict(projector_config, dimension)
            kind = projector_config["type"]
            if kind == "tfidf":
                tfidf_terms, _ = TfidfProjector.validate_header_config(
                    projector_config, allow_missing_idf=True
                )
                if len(tfidf_terms) != dimension:
                    raise ValueError("Saved TF-IDF dimension does not match the skill")
                if "idf" in projector_config:
                    saved_tfidf = TfidfProjector.from_config(projector_config)
            if kind == "pragmatic_tfidf":
                _, _, prag_total_dim = PragmaticCompositeProjector.validate_header_config(
                    projector_config, allow_missing_idf=True
                )
                if prag_total_dim != dimension:
                    raise ValueError("Saved pragmatic TF-IDF dimension does not match the skill")
                runtime_bytes += 64 * prag_total_dim
                base_cfg = projector_config.get("base_tfidf", {})
                if "prag_idf" in projector_config and isinstance(base_cfg, dict) and "idf" in base_cfg:
                    saved_prag = PragmaticCompositeProjector.from_config(projector_config)
            if kind == "deterministic" and (
                type(projector_config.get("dimension")) is not int
                or projector_config.get("dimension") != dimension
            ):
                raise ValueError("Saved projector dimension does not match the skill")
            if kind == "hybrid":
                from system1.core.embeddings import HybridProjector

                if set(projector_config) - HybridProjector.CONFIG_KEYS:
                    raise ValueError("Saved hybrid projector contains unsupported settings")
                sparse, dense = projector_config.get("sparse_dim"), projector_config.get("dense_dim")
                ds_flag = projector_config.get("direct_spectral", False)
                if "direct_spectral" in projector_config and type(ds_flag) is not bool:
                    raise ValueError("Saved hybrid projector direct_spectral must be a boolean")
                direct_spec = projector_config.get("direct_spectral_dim", 0) if ds_flag else 0
                raw_rbf = projector_config.get("rbf_landmarks")
                if raw_rbf is not None and not isinstance(raw_rbf, (list, tuple)):
                    raise ValueError("Saved hybrid projector rbf_landmarks must be a sequence")
                inline_rbf_rows = len(raw_rbf) if raw_rbf is not None else 0
                if (
                    type(sparse) is not int
                    or type(dense) is not int
                    or sparse <= 0
                    or dense <= 0
                    or type(direct_spec) is not int
                    or direct_spec < 0
                    or sparse + dense + direct_spec + inline_rbf_rows > dimension
                ):
                    raise ValueError("Saved hybrid projector dimensions do not match the skill")
                tf_terms_raw = projector_config.get("tfidf_terms", ())
                spec_terms_raw = projector_config.get("spectral_terms", ())
                if not isinstance(tf_terms_raw, (list, tuple)):
                    raise ValueError("Saved hybrid projector tfidf_terms must be a sequence")
                if not isinstance(spec_terms_raw, (list, tuple)):
                    raise ValueError("Saved hybrid projector spectral_terms must be a sequence")
                runtime_bytes += 16 * 25000 * dense
                runtime_bytes += 64 * len(tf_terms_raw)
                runtime_bytes += 16 * len(spec_terms_raw) * dense
                for mat_key in ("spectral_vectors", "subspace_components", "rbf_landmarks"):
                    mat_val = projector_config.get(mat_key)
                    if mat_val is None:
                        continue
                    if not isinstance(mat_val, (list, tuple)) or any(
                        not isinstance(row, (list, tuple)) for row in mat_val
                    ):
                        raise ValueError(f"Saved hybrid projector {mat_key} must be a 2D sequence")
                    runtime_bytes += 8 * sum(len(row) for row in mat_val)
        if runtime_bytes > max_runtime_bytes:
            raise ValueError("Saved skill runtime allocation exceeds memory budget")
        return saved_tfidf, saved_prag, runtime_bytes

    @staticmethod
    def _is_readonly_mmap(mm: mmap.mmap) -> bool:
        with memoryview(mm) as mv:
            return bool(mv.readonly)

    @classmethod
    def _from_v3_buffer(
        cls,
        buf: Union[bytes, bytearray, memoryview, mmap.mmap],
        buf_len: int,
        projector: Optional[Any] = None,
        backend: str = "numpy",
    ) -> CompiledSystemOneModel:
        """Preflight-validates and constructs a zero-copy CompiledSystemOneModel over a v3 buffer."""
        if not isinstance(buf, (bytes, bytearray, memoryview, mmap.mmap)):
            raise ValueError("Invalid System 1 v3 Model binary: unsupported buffer type")
        actual_len = buf.nbytes if isinstance(buf, memoryview) else len(buf)
        if (
            type(buf_len) is not int
            or buf_len != actual_len
            or buf_len < 16
            or buf_len > MAX_SKILL_BYTES_V3
        ):
            raise ValueError("Saved v3 skill file size is out of bounds")
        if isinstance(buf, bytearray):
            buf = memoryview(bytes(buf))
        elif isinstance(buf, mmap.mmap):
            if not cls._is_readonly_mmap(buf):
                buf = memoryview(bytes(buf))
        elif isinstance(buf, memoryview):
            if not buf.c_contiguous:
                buf = memoryview(buf.tobytes())
            else:
                if buf.ndim != 1 or buf.format != "B":
                    buf = buf.cast("B")
                buf_obj = getattr(buf, "obj", None)
                is_ro_backing = type(buf_obj) is bytes or (
                    isinstance(buf_obj, mmap.mmap) and cls._is_readonly_mmap(buf_obj)
                )
                if not buf.readonly or not is_ro_backing:
                    buf = memoryview(buf.tobytes())
        elif type(buf) is not bytes:
            buf = memoryview(bytes(buf))

        magic, hdr_len = _V3_HDR_STRUCT.unpack_from(buf, 0)
        if magic != MAGIC_HEADER_V3:
            raise ValueError("Invalid System 1 v3 Model binary: missing or invalid magic header")
        if hdr_len == 0 or hdr_len > MAX_SKILL_HEADER_BYTES_V3:
            raise ValueError("Saved v3 skill header exceeds size limit")
        header_end = 16 + hdr_len
        if buf_len < header_end:
            raise ValueError("Corrupted System 1 v3 Model binary: truncated header")

        try:
            header_json = bytes(buf[16:header_end]).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Corrupted System 1 v3 header encoding: {exc}") from exc

        def _reject_nonfinite(tok: str) -> Any:
            raise ValueError(f"Saved v3 skill header contains non-finite constant: {tok!r}")

        meta = json.loads(header_json, parse_constant=_reject_nonfinite, parse_float=_strict_json_parse_float)
        if not isinstance(meta, dict) or meta.get("version") != 3:
            raise ValueError("Unsupported or inconsistent saved v3 skill format version")
        if "embed_vocab" in meta and type(meta["embed_vocab"]) is not bool:
            raise ValueError("Saved v3 skill embed_vocab flag must be a boolean")

        fields_data = meta["schema_dict"]["fields"]
        if not isinstance(fields_data, dict) or not 0 < len(fields_data) <= MAX_SKILL_FIELDS_V3:
            raise ValueError("Saved v3 skill field count exceeds limits")
        for definition in fields_data.values():
            if len(definition.get("options", ())) > MAX_SKILL_OPTIONS:
                raise ValueError("Saved v3 skill option count exceeds limits")
        schema = DecisionSchema.from_dict(meta["schema_dict"])
        if meta.get("schema_digest") != schema.schema_digest():
            raise ValueError("Saved skill schema digest does not match its definition")
        dimension = meta.get("dimension")
        if type(dimension) is not int or not 0 < dimension <= MAX_SKILL_DIMENSION:
            raise ValueError("Saved skill dimension must be a positive integer within limits")
        if set(meta.get("heads", ())) != set(schema.fields):
            raise ValueError("Saved skill heads must exactly match its schema")

        raw_tensors = meta.get("tensors")
        if not isinstance(raw_tensors, dict) or len(raw_tensors) > 17 * MAX_SKILL_FIELDS_V3 + 32:
            raise ValueError("Saved v3 skill tensor manifest is invalid or exceeds limit")

        namespaced = _resolve_tensor_namespaced_flag(schema, meta)
        allowed_names = set(PROJECTOR_TENSOR_NAMES)
        for fname in schema.fields:
            for suf in HEAD_TENSOR_SUFFIXES:
                allowed_names.add(_head_tensor_name(fname, suf, namespaced=namespaced))

        max_dim_bound = max(MAX_SKILL_CALIBRATION_SCORES, 250_000)
        total_array_bytes = 0
        intervals: List[Tuple[int, int, str]] = []
        for tname, spec in raw_tensors.items():
            if not isinstance(tname, str) or tname not in allowed_names:
                raise ValueError(f"Saved v3 skill contains unexpected tensor: {tname!r}")
            if not isinstance(spec, Mapping):
                raise ValueError(f"Saved v3 skill tensor {tname!r} spec must be a mapping")
            dt_name = spec.get("dtype")
            if not isinstance(dt_name, str) or dt_name not in _ALLOWED_V3_DTYPES:
                raise ValueError(f"Saved v3 skill tensor {tname!r} has unsupported dtype {dt_name!r}")
            dt = np.dtype(dt_name)
            shape = spec.get("shape")
            if (
                not isinstance(shape, list)
                or len(shape) > 3
                or any(type(d) is not int or d < 0 or d > max_dim_bound for d in shape)
            ):
                raise ValueError(f"Saved v3 skill tensor {tname!r} has invalid shape {shape!r}")
            raw_order = spec.get("order", "C")
            if type(raw_order) is not str or raw_order not in ("C", "F"):
                raise ValueError(f"Saved v3 skill tensor {tname!r} has unsupported order {raw_order!r}")
            if raw_order == "F" and len(shape) != 2:
                raise ValueError(f"Saved v3 skill tensor {tname!r} order 'F' requires a 2D shape, got {shape!r}")
            count = 1
            for d in shape:
                count *= d
            expected_nbytes = count * dt.itemsize
            nbytes = spec.get("nbytes")
            if (
                type(nbytes) is not int
                or nbytes != expected_nbytes
                or nbytes < 0
                or nbytes > MAX_SKILL_ARRAY_BYTES_V3
            ):
                raise ValueError(f"Saved v3 skill tensor {tname!r} nbytes mismatch")
            offset = spec.get("offset")
            if (
                type(offset) is not int
                or offset < header_end
                or (offset % 64) != 0
                or offset + nbytes > buf_len
            ):
                raise ValueError(f"Saved v3 skill tensor {tname!r} offset is misaligned or out of bounds")
            if (nbytes >= 65536 or tname == "_projector_dense_table") and (offset % S1M_V3_PAGE_SIZE) != 0:
                raise ValueError(f"Saved v3 skill tensor {tname!r} is not {S1M_V3_PAGE_SIZE}-byte page aligned")
            total_array_bytes += nbytes
            if total_array_bytes > MAX_SKILL_ARRAY_BYTES_V3:
                raise ValueError("Saved v3 skill array payload exceeds memory budget")
            if nbytes > 0:
                intervals.append((offset, offset + nbytes, tname))

        intervals.sort()
        prev_end = header_end
        for start, end, tname in intervals:
            if start < prev_end:
                raise ValueError(f"Saved v3 skill tensor {tname!r} overlaps another tensor region")
            prev_end = end

        shapes_v3: Dict[str, Tuple[int, ...]] = {
            tname: tuple(spec["shape"]) for tname, spec in raw_tensors.items()
        }
        idf_tensor_names = frozenset((
            "_projector_tfidf_idf",
            "_projector_prag_base_idf",
            "_projector_prag_idf",
            "_projector_hybrid_tfidf_idf",
        ))
        logit_adj_names = frozenset(
            _head_tensor_name(fname, "logit_adj", namespaced=namespaced)
            for fname in schema.fields
        )
        score_f64_names = frozenset(
            _head_tensor_name(fname, sfx, namespaced=namespaced)
            for fname in schema.fields
            for sfx in ("calib_scores", "proto_scores")
        )
        upcast_bytes_v3 = 0
        for tname, spec in raw_tensors.items():
            dt_bytes = np.dtype(spec["dtype"]).itemsize
            cnt = int(math.prod(shapes_v3[tname]))
            if tname in idf_tensor_names or tname in logit_adj_names:
                if dt_bytes != 8:
                    upcast_bytes_v3 += 8 * cnt
            elif tname not in score_f64_names and dt_bytes != 4:
                upcast_bytes_v3 += 4 * cnt
        extra_runtime_bytes = _validate_skill_shapes_and_runtime_delta(
            schema,
            dimension,
            shapes_v3,
            heads_meta=meta.get("heads"),
            namespaced=namespaced,
        )
        _, layout_extra_bytes_v3 = _validate_tensor_layouts_header(
            meta,
            shapes_v3,
            non_f32_2d_names=idf_tensor_names | logit_adj_names | score_f64_names,
            full_allocation=True,
        )
        extra_runtime_bytes += upcast_bytes_v3 + layout_extra_bytes_v3

        cov_fields = sum(
            1
            for fname in schema.fields
            if (
                _head_tensor_name(fname, "P", namespaced=namespaced) in raw_tensors
                or _head_tensor_name(fname, "B", namespaced=namespaced) in raw_tensors
            )
        )
        projector_config = meta.get("projector")
        saved_tfidf, saved_prag, _ = cls._validate_projector_and_budget(
            schema,
            dimension,
            projector_config,
            cov_fields=cov_fields,
            max_runtime_bytes=MAX_SKILL_RUNTIME_BYTES_V3,
            extra_runtime_bytes=extra_runtime_bytes,
        )

        views: Dict[str, np.ndarray] = {}
        try:
            for tname, spec in raw_tensors.items():
                shape = tuple(spec["shape"])
                dt = np.dtype(spec["dtype"])
                order_flag = spec.get("order", "C")
                count = int(math.prod(shape)) if shape else 1
                if count == 0:
                    arr = np.empty(shape, dtype=dt, order=order_flag)
                elif order_flag == "F":
                    arr = np.ndarray(shape, dtype=dt, buffer=buf, offset=spec["offset"], order="F")
                else:
                    arr = np.frombuffer(buf, dtype=dt, count=count, offset=spec["offset"]).reshape(shape)
                arr.setflags(write=False)
                views[tname] = arr

            return cls._reconstruct_model_from_parts(
                meta,
                views,
                projector=projector,
                backend=backend,
                schema=schema,
                dimension=dimension,
                projector_config=projector_config,
                saved_tfidf=saved_tfidf,
                saved_prag=saved_prag,
                format_version=3,
            )
        except Exception:
            views.clear()
            raise

    @classmethod
    def from_bytes(
        cls,
        data: Union[bytes, bytearray, memoryview],
        projector: Optional[Any] = None,
        backend: str = "numpy",
    ) -> CompiledSystemOneModel:
        """Deserializes a CompiledSystemOneModel from .s1m binary bytes (v1, v2, or v3)."""
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise ValueError("Invalid System 1 Model binary: unsupported buffer type")
        mv = memoryview(data)
        total_bytes = mv.nbytes
        if total_bytes > max(MAX_SKILL_BYTES, MAX_SKILL_BYTES_V3):
            raise ValueError("Saved skill file exceeds size limit")
        if total_bytes < 8:
            raise ValueError("Invalid System 1 Model binary: missing or invalid magic header")

        if not mv.c_contiguous:
            frozen: Optional[bytes] = mv.tobytes()
            mv_b = memoryview(frozen)
        else:
            mv_b = mv if (mv.ndim == 1 and mv.format == "B") else mv.cast("B")
            frozen = None

        magic8 = bytes(mv_b[:8])
        if magic8 == MAGIC_HEADER_V3:
            if total_bytes > MAX_SKILL_BYTES_V3:
                raise ValueError("Saved v3 skill file size is out of bounds")
            if frozen is None and not (
                type(data) is bytes
                or (
                    isinstance(data, memoryview)
                    and data.readonly
                    and type(getattr(data, "obj", None)) is bytes
                )
            ):
                mv_b = memoryview(mv_b.tobytes())
            return cls._from_v3_buffer(mv_b, total_bytes, projector=projector, backend=backend)

        if total_bytes > MAX_SKILL_BYTES:
            raise ValueError("Saved skill file exceeds size limit")
        if magic8[:4] not in (b"S1M\x01", MAGIC_HEADER):
            raise ValueError("Invalid System 1 Model binary: missing or invalid magic header")

        raw_bytes = frozen if frozen is not None else (data if type(data) is bytes else mv_b.tobytes())
        header_len = struct.unpack(">I", raw_bytes[4:8])[0]
        if header_len > MAX_SKILL_HEADER_BYTES:
            raise ValueError("Saved skill header exceeds size limit")
        header_end = 8 + header_len
        if len(raw_bytes) < header_end:
            raise ValueError("Corrupted System 1 Model binary: truncated header")

        try:
            header_json = raw_bytes[8:header_end].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Corrupted System 1 header encoding: {exc}") from exc

        def _reject_nonfinite(tok: str) -> Any:
            raise ValueError(f"Saved skill header contains non-finite constant: {tok!r}")

        meta = json.loads(header_json, parse_constant=_reject_nonfinite, parse_float=_strict_json_parse_float)
        if meta.get("version") not in (1, 2) or meta["version"] != raw_bytes[3]:
            raise ValueError("Unsupported or inconsistent saved skill format version")

        fields_data = meta["schema_dict"]["fields"]
        if not isinstance(fields_data, dict) or not 0 < len(fields_data) <= MAX_SKILL_FIELDS:
            raise ValueError("Saved skill field count exceeds limits")
        for definition in fields_data.values():
            if len(definition.get("options", ())) > MAX_SKILL_OPTIONS:
                raise ValueError("Saved skill option count exceeds limits")
        schema = DecisionSchema.from_dict(meta["schema_dict"])
        if meta.get("schema_digest") != schema.schema_digest():
            raise ValueError("Saved skill schema digest does not match its definition")
        dimension = meta["dimension"]
        if type(dimension) is not int or not 0 < dimension <= MAX_SKILL_DIMENSION:
            raise ValueError("Saved skill dimension must be a positive integer within limits")
        if set(meta["heads"]) != set(schema.fields):
            raise ValueError("Saved skill heads must exactly match its schema")
        projector_config = meta.get("projector")
        cov_fields = (
            len(schema.fields)
            if (projector_config is None or projector_config.get("type") == "deterministic")
            else 0
        )
        saved_tfidf, saved_prag, runtime_bytes = cls._validate_projector_and_budget(
            schema,
            dimension,
            projector_config,
            cov_fields=cov_fields,
            max_runtime_bytes=MAX_SKILL_RUNTIME_BYTES,
        )
        namespaced = _resolve_tensor_namespaced_flag(schema, meta)
        npz_file = _read_skill_arrays(
            raw_bytes[header_end:],
            schema,
            dimension,
            runtime_bytes,
            namespaced=namespaced,
            meta=meta,
        )
        return cls._reconstruct_model_from_parts(
            meta,
            npz_file,
            projector=projector,
            backend=backend,
            schema=schema,
            dimension=dimension,
            projector_config=projector_config,
            saved_tfidf=saved_tfidf,
            saved_prag=saved_prag,
            format_version=int(meta["version"]),
        )

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        projector: Optional[Any] = None,
        backend: str = "numpy",
        *,
        mmap_mode: bool = True,
    ) -> CompiledSystemOneModel:
        """Loads a CompiledSystemOneModel from a .s1m file (using zero-copy mmap for S1M\\x03)."""
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"Model file not found: {target}")
        file_size = target.stat().st_size
        with open(target, "rb") as _hdr_fh:
            magic8 = _hdr_fh.read(8)
        if magic8 == MAGIC_HEADER_V3:
            if file_size > MAX_SKILL_BYTES_V3:
                raise ValueError("Saved v3 skill file exceeds size limit")
            if mmap_mode:
                fh = open(target, "rb")
                mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
                try:
                    model = cls._from_v3_buffer(mm, len(mm), projector=projector, backend=backend)
                except Exception:
                    cls._close_mmap_handle(mm, fh)
                    raise
                model._mmap_obj = mm
                model._mmap_fh = fh
                return model
            return cls.from_bytes(target.read_bytes(), projector=projector, backend=backend)

        data = _read_skill_bytes(target)
        return cls.from_bytes(data, projector=projector, backend=backend)


class SystemOneCompiler:
    """Teach and compile small decision heads.

    Takes a DecisionSchema and training exemplars (or generates synthetic exemplars),
    fits closed-form Ridge Regression hyperplanes:
        W* = (X^T X + lambda I)^(-1) X^T Y
    calibrates temperature scaling and conformal prediction bounds, and serializes
    the resulting model to a compact .s1m binary. ``choice_solver="logistic"``
    optionally fits ChoiceField heads with cross-entropy using SciPy at teaching
    time. Saved heads still run with NumPy; other field types keep their existing
    fitting methods.
    """

    def __init__(
        self,
        schema: Union[DecisionSchema, Type[DecisionSchema], Mapping[str, Any]],
        *,
        projector: Optional[Any] = None,
        dimension: int = 384,
        regularization: float = 1.0,
        backend: str = "numpy",
        forgetting_factor: float = 0.995,
        relative_odds_ratio: float = 1.5,
        confidence_floor_tau0: float = 0.15,
        recency_weighted: bool = False,
        choice_solver: str = "ridge",
        mondrian: bool = False,
        familiarity_alpha: float = 0.0,
        prototype_alpha: float = 0.0,
        hierarchical: Union[bool, str] = "auto",
        auto_regularization: bool = True,
        noise_robust: bool = True,
        label_noise_robust: Optional[bool] = None,
        semantic_mode: str = "default",
        proto_blend: Optional[float] = None,
        fisher_reg: Optional[float] = None,
    ) -> None:
        if isinstance(schema, type) and issubclass(schema, DecisionSchema):
            self.schema = schema()
        elif isinstance(schema, DecisionSchema):
            self.schema = schema
        elif isinstance(schema, Mapping):
            from system1.compat.typesafe import _build_dynamic_schema
            self.schema = _build_dynamic_schema(schema)
        else:
            raise TypeError(f"Unsupported schema type: {type(schema).__name__}")

        self.dimension = dimension
        self.regularization = float(regularization)
        if not math.isfinite(self.regularization) or self.regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        if choice_solver not in ("ridge", "logistic"):
            raise ValueError("choice_solver must be 'ridge' or 'logistic'")
        self.choice_solver = choice_solver
        self.mondrian = bool(mondrian)
        self.familiarity_alpha = float(familiarity_alpha)
        if not math.isfinite(self.familiarity_alpha) or not 0.0 <= self.familiarity_alpha < 1.0:
            raise ValueError("familiarity_alpha must be finite and in [0, 1)")
        self.prototype_alpha = float(prototype_alpha)
        if not math.isfinite(self.prototype_alpha) or not 0.0 <= self.prototype_alpha < 1.0:
            raise ValueError("prototype_alpha must be finite and in [0, 1)")
        if not isinstance(hierarchical, bool) and hierarchical != "auto":
            raise ValueError("hierarchical must be a boolean or 'auto'")
        self.hierarchical: Union[bool, str] = hierarchical
        if not isinstance(auto_regularization, bool):
            raise ValueError("auto_regularization must be a boolean")
        self.auto_regularization = auto_regularization
        resolved_noise_robust = noise_robust if label_noise_robust is None else label_noise_robust
        if not isinstance(resolved_noise_robust, bool):
            raise ValueError("noise_robust must be a boolean")
        self.noise_robust = resolved_noise_robust
        self.backend = backend
        self.forgetting_factor = float(forgetting_factor)
        self.relative_odds_ratio = float(relative_odds_ratio)
        self.confidence_floor_tau0 = float(confidence_floor_tau0)
        self.recency_weighted = bool(recency_weighted)
        if not isinstance(semantic_mode, str) or semantic_mode not in VALID_SEMANTIC_MODES:
            raise ValueError(
                f"Unsupported semantic_mode {semantic_mode!r}; expected one of {VALID_SEMANTIC_MODES}"
            )
        self.semantic_mode = semantic_mode
        if proto_blend is not None:
            pb = float(proto_blend)
            if not math.isfinite(pb) or not (0.0 <= pb <= 1.0):
                raise ValueError("proto_blend must be finite and in [0, 1]")
            self.proto_blend: Optional[float] = pb
        else:
            self.proto_blend = None
        if fisher_reg is not None:
            fr = float(fisher_reg)
            if not math.isfinite(fr) or fr <= 0.0:
                raise ValueError("fisher_reg must be finite and positive")
            self.fisher_reg: Optional[float] = fr
        else:
            self.fisher_reg = None
        self._explicit_projector = projector is not None
        if projector is not None:
            self.projector = projector
            if hasattr(projector, "dimension"):
                self.dimension = projector.dimension
        else:
            self.projector = DeterministicSemanticProjector(
                dimension=self.dimension,
                recency_weighted=self.recency_weighted,
            )
        self.telemetry_projector = TelemetryProjector(embedding_dim=self.dimension)

    @staticmethod
    def _solve_logistic(
        X: np.ndarray,
        Y: np.ndarray,
        regularization: float,
        sample_weights: Optional[np.ndarray] = None,
    ):
        """Fit sum cross-entropy + lambda/2 * ||W||², with an unpenalized bias.

        The runtime divides choice logits by .25, so scale the stored coefficients
        by .25. Calibration then sees the fitted logits, including for two choices.
        No optimizer object or dependency is serialized into the skill.
        """
        try:
            from scipy.optimize import minimize
        except ImportError as exc:
            raise ImportError("Logistic teaching requires pip install 'system1-instinct[teaching]'") from exc

        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        n, dimension = X.shape
        choices = Y.shape[1]
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
            raise ValueError("Logistic teaching requires finite features and targets")
        if np.any(Y.sum(axis=0) == 0):
            raise ValueError("Logistic teaching needs fitting examples for every choice")

        sw_col: Optional[np.ndarray] = None
        if sample_weights is not None:
            sw_raw = np.maximum(1e-6, np.asarray(sample_weights, dtype=np.float64).flatten())
            sw_col = (sw_raw * (n / float(np.sum(sw_raw))))[:, None]

        if 0 < n < dimension and regularization > 0.0:
            q_mat, r_mat = np.linalg.qr(X.T, mode="reduced")
            z_mat = r_mat.T
            r_dim = z_mat.shape[1]

            def objective_subspace(flat):
                parameters = flat.reshape(choices, r_dim + 1)
                w_sub, biases = parameters[:, :-1], parameters[:, -1]
                logits = z_mat @ w_sub.T + biases
                shifted = logits - logits.max(axis=1, keepdims=True)
                exp = np.exp(shifted)
                total = exp.sum(axis=1, keepdims=True)
                if sw_col is None:
                    loss = (np.log(total).sum() - np.sum(Y * shifted)
                            + .5 * regularization * np.sum(w_sub * w_sub)) / n
                    residual = exp / total - Y
                else:
                    loss = (np.sum(sw_col * np.log(total)) - np.sum(sw_col * Y * shifted)
                            + .5 * regularization * np.sum(w_sub * w_sub)) / n
                    residual = sw_col * (exp / total - Y)
                gradient = np.empty_like(parameters)
                gradient[:, :-1] = (residual.T @ z_mat + regularization * w_sub) / n
                gradient[:, -1] = residual.mean(axis=0)
                return float(loss), gradient.ravel()

            result = minimize(
                objective_subspace,
                np.zeros(choices * (r_dim + 1)),
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 1000, "gtol": 1e-6, "ftol": 1e-10},
            )
            if not result.success or not np.all(np.isfinite(result.x)):
                raise RuntimeError("Logistic teaching did not converge; no skill was compiled")
            parameters = result.x.reshape(choices, r_dim + 1)
            weights_full = parameters[:, :-1] @ q_mat.T
            return (.25 * weights_full).astype(np.float32), (.25 * parameters[:, -1]).astype(np.float32)

        def objective(flat):
            parameters = flat.reshape(choices, dimension + 1)
            weights, biases = parameters[:, :-1], parameters[:, -1]
            logits = X @ weights.T + biases
            shifted = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(shifted)
            total = exp.sum(axis=1, keepdims=True)
            if sw_col is None:
                loss = (np.log(total).sum() - np.sum(Y * shifted)
                        + .5 * regularization * np.sum(weights * weights)) / n
                residual = exp / total - Y
            else:
                loss = (np.sum(sw_col * np.log(total)) - np.sum(sw_col * Y * shifted)
                        + .5 * regularization * np.sum(weights * weights)) / n
                residual = sw_col * (exp / total - Y)
            gradient = np.empty_like(parameters)
            gradient[:, :-1] = (residual.T @ X + regularization * weights) / n
            gradient[:, -1] = residual.mean(axis=0)
            return float(loss), gradient.ravel()

        result = minimize(objective, np.zeros(choices * (dimension + 1)), jac=True,
                          method="L-BFGS-B", options={"maxiter": 1000, "gtol": 1e-6, "ftol": 1e-10})
        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError("Logistic teaching did not converge; no skill was compiled")
        parameters = result.x.reshape(choices, dimension + 1)
        return (.25 * parameters[:, :-1]).astype(np.float32), (.25 * parameters[:, -1]).astype(np.float32)

    def generate_synthetic_exemplars(
        self,
        samples_per_choice: int = 20,
        teacher: str = "synthetic",
    ) -> Dict[str, List[Tuple[str, Any]]]:
        """Generates rich synthetic training exemplars for all schema fields."""
        exemplars: Dict[str, List[Tuple[str, Any]]] = {}

        for field_name, f_def in self.schema.fields.items():
            field_samples: List[Tuple[str, Any]] = []

            if isinstance(f_def, ChoiceField):
                templates = [
                    "User request regarding {opt}: {desc}",
                    "Please process {clean_opt} operation immediately",
                    "Handle inquiry: {desc}",
                    "Action required: {clean_opt}. Details: {desc}",
                    "Status update for {clean_opt}",
                    "Incoming event classified under {clean_opt}: {desc}",
                    "Requesting {clean_opt} assistance",
                    "Dispatch to {clean_opt} handler: {desc}",
                    "Log entry indicating {clean_opt} condition",
                    "Notification: {desc}",
                ]
                for opt in f_def.options:
                    desc_raw = f_def.descriptions.get(opt, "")
                    if isinstance(desc_raw, (list, tuple)):
                        desc = " ".join(desc_raw)
                        extra_exs = list(desc_raw)
                    else:
                        desc = str(desc_raw)
                        extra_exs = [desc]

                    clean_opt = opt.replace("_", " ").replace("-", " ")
                    # Add base variations
                    for t in templates:
                        prompt = t.format(opt=opt, clean_opt=clean_opt, desc=desc)
                        field_samples.append((prompt, opt))

                    for ex in extra_exs:
                        if ex.strip():
                            field_samples.append((ex.strip(), opt))
                            field_samples.append((f"{opt}: {ex.strip()}", opt))
                            # Add sub-phrase splits
                            for phrase in ex.replace(" and ", ", ").replace(";", ",").split(","):
                                p = phrase.strip()
                                if p and len(p) > 2:
                                    field_samples.append((p, opt))
                                    field_samples.append((f"User reporting {clean_opt}: {p}", opt))

                    # Common domain expansions for triage & support
                    domain_synonyms = {
                        "technical": [
                            "server crashed with error trace", "software bug in backend service",
                            "NullPointerException thrown by api service", "unhandled exception in runtime",
                            "application crash and downtime", "database connection failure",
                            "system crash dump", "stack trace error log",
                        ],
                        "billing": [
                            "why did you charge my credit card", "unauthorized charge on card for renewal",
                            "refund request for invoice", "dispute subscription fee",
                            "payment failed for monthly plan", "billing invoice question",
                        ],
                        "account": [
                            "forgot my password reset link", "cannot pass MFA token verification",
                            "locked out of user account login", "change profile email address",
                            "login trouble authentication failed", "two factor auth code not working",
                        ],
                    }
                    if opt in domain_synonyms:
                        for s in domain_synonyms[opt]:
                            field_samples.append((s, opt))

                    # Expand to reach requested sample count
                    idx = 0
                    while len([s for s in field_samples if s[1] == opt]) < samples_per_choice:
                        field_samples.append((
                            f"Automated test case {idx} for {clean_opt}: {desc} ({opt})",
                            opt,
                        ))
                        idx += 1

            elif isinstance(f_def, BooleanField):
                true_templates = [
                    f"Confirmed state true: {f_def.true_description}",
                    f"Affirmative condition: {f_def.true_description}",
                    f"Verified true: {f_def.description or f_def.true_description}",
                    f"Status true: {f_def.true_description}",
                    f"Criteria satisfied true: {f_def.true_description}",
                ]
                false_templates = [
                    f"Confirmed state false: {f_def.false_description}",
                    f"Negative condition: {f_def.false_description}",
                    f"Verified false: {f_def.false_description}",
                    f"Status false: {f_def.false_description}",
                    f"Criteria unsatisfied false: {f_def.false_description}",
                ]
                for p in true_templates:
                    field_samples.append((p, True))
                for p in false_templates:
                    field_samples.append((p, False))

                idx = 0
                while len(field_samples) < max(samples_per_choice * 2, 20):
                    field_samples.append((f"Synthetic positive {idx}: {f_def.true_description}", True))
                    field_samples.append((f"Synthetic negative {idx}: {f_def.false_description}", False))
                    idx += 1

            elif isinstance(f_def, MultiChoiceField):
                templates = [
                    "Apply tag {clean_opt}: {desc}",
                    "{clean_opt} condition active: {desc}",
                    "Tagged with {clean_opt}",
                    "Feature detected: {clean_opt}",
                    "Operational signal for {clean_opt}: {desc}",
                    "Notification concerning {clean_opt} status",
                    "Event flagged with {clean_opt}",
                    "Trigger for {clean_opt}: {desc}",
                ]
                for opt in f_def.options:
                    desc_raw = f_def.descriptions.get(opt, "")
                    desc = " ".join(desc_raw) if isinstance(desc_raw, (list, tuple)) else str(desc_raw)
                    clean_opt = opt.replace("_", " ").replace("-", " ")
                    for t in templates[:max(3, samples_per_choice // 2)]:
                        field_samples.append((t.format(clean_opt=clean_opt, desc=desc), (opt,)))

                # Add pairwise tag combinations if multiple options
                if len(f_def.options) >= 2:
                    for idx_a in range(len(f_def.options) - 1):
                        opt_a = f_def.options[idx_a]
                        opt_b = f_def.options[idx_a + 1]
                        field_samples.append((
                            f"Combined indicators for {opt_a} and {opt_b}",
                            (opt_a, opt_b),
                        ))

                # Add neutral / unflagged negative samples
                neutral_negatives = [
                    "Normal standard operation with no active tags",
                    "Benign routine background event",
                    "General routine request with no tags",
                    "Good morning, standard greeting and conversation",
                    "Routine heartbeat check passed",
                    "Standard documentation page lookup",
                    "Regular ping status report",
                    "System idling normally with zero alerts",
                ]
                for neg_text in neutral_negatives:
                    field_samples.append((neg_text, ()))

            elif isinstance(f_def, ScoreField):
                low_d = f_def.low_description
                high_d = f_def.high_description
                desc_mid = f_def.description or "Intermediate balanced condition"
                score_levels = [
                    (0.0, f"Low boundary value: {low_d}"),
                    (0.0, f"Minimum rating condition: {low_d}"),
                    (0.01, f"Safe operational state: {low_d}"),
                    (0.01, f"Benign routine task: {low_d}"),
                    (0.01, f"Clean verification passed: {low_d}"),
                    (0.02, f"Read-only inspection and analysis: {low_d}"),
                    (0.02, f"Directory listing and file hierarchy inspection: {low_d}"),
                    (0.05, f"Harmless standard check: {low_d}"),
                    (0.10, f"Non-destructive operation: {low_d}"),
                    (0.45, f"Moderate rating indicator: {desc_mid}"),
                    (0.50, f"Intermediate moderate value: {desc_mid}"),
                    (0.50, f"Median balanced score: {desc_mid}"),
                    (0.55, f"Borderline operational action: {desc_mid}"),
                    (0.75, f"Elevated high indicator: {high_d}"),
                    (0.90, f"Critical security violation: {high_d}"),
                    (0.95, f"Severe vulnerability exploit: {high_d}"),
                    (0.98, f"High boundary value: {high_d}"),
                    (1.0, f"Maximum critical rating: {high_d}"),
                ]
                for fraction, prompt in score_levels:
                    target_val = float(_score_from_unit_prob(fraction, f_def.min_value, f_def.max_value))
                    field_samples.append((prompt, target_val))

            exemplars[field_name] = field_samples

        return exemplars

    def _solve_ridge(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        regularization: float,
        sample_weights: Optional[np.ndarray] = None,
        regularize_bias: bool = True,
        return_covariance: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Solves closed-form Ridge Regression with optional sample weights and regularized bias.

        tilde{X} = [X, 1]
        A = tilde{X}^T W tilde{X} + lambda I'
        W_tilde = solve(A, tilde{X}^T W Y)
        Returns: (weights: shape (K, D), biases: shape (K,))
        Or if return_covariance=True: (weights, biases, P, B)
        """
        N, D = X.shape
        K = Y.shape[1] if Y.ndim > 1 else 1
        Y_mat = Y.reshape(N, K).astype(np.float32)

        X_aug = np.column_stack([X, np.ones((N, 1), dtype=np.float32)])
        if sample_weights is not None:
            sw = np.asarray(sample_weights, dtype=np.float32).flatten()
            sw_sqrt = np.sqrt(np.maximum(1e-8, sw))[:, None]
            X_aug = X_aug * sw_sqrt
            Y_mat = Y_mat * sw_sqrt

        if 0 < N < D and float(regularization) > 0.0:
            if regularize_bias:
                X64 = X_aug.astype(np.float64)
                Y64 = Y_mat.astype(np.float64)
                d_inv = np.full(D + 1, 1.0 / float(regularization), dtype=np.float64)
                d_inv[D] = 1.0 / max(1.0, float(regularization))
                XD = X64 * d_inv
                gram = np.eye(N, dtype=np.float64) + XD @ X64.T
                try:
                    alpha_dual = np.linalg.solve(gram, Y64)
                    sol = (XD.T @ alpha_dual).astype(np.float32)
                    weights = sol[:D, :].T.astype(np.float32)
                    biases = sol[D, :].astype(np.float32)
                    if not return_covariance:
                        return weights, biases
                    K_inv_XD = np.linalg.solve(gram, XD).astype(np.float32)
                    P = -(XD.astype(np.float32).T @ K_inv_XD)
                    P[np.diag_indices(D + 1)] += d_inv.astype(np.float32)
                    B = (X_aug.T @ Y_mat).astype(np.float32)
                    return weights, biases, P, B
                except np.linalg.LinAlgError:
                    pass
            else:
                try:
                    if sample_weights is not None:
                        w_vec = np.maximum(1e-8, np.asarray(sample_weights, dtype=np.float64).flatten())
                        w_sum = float(np.sum(w_vec))
                        mu_X = ((w_vec @ X.astype(np.float64)) / w_sum).astype(np.float32)
                        mu_Y = ((w_vec @ np.asarray(Y, dtype=np.float64).reshape(N, K)) / w_sum).astype(np.float32)
                        sw_f32 = np.sqrt(w_vec).astype(np.float32)[:, None]
                        Xc = (X - mu_X) * sw_f32
                        Yc = (np.asarray(Y, dtype=np.float32).reshape(N, K) - mu_Y) * sw_f32
                    else:
                        w_sum = float(N)
                        mu_X = np.mean(X, axis=0, dtype=np.float64).astype(np.float32)
                        mu_Y = np.mean(Y_mat, axis=0, dtype=np.float64).astype(np.float32)
                        Xc = X - mu_X
                        Yc = Y_mat - mu_Y
                    inv_reg = 1.0 / float(regularization)
                    inv_reg_f32 = np.float32(inv_reg)
                    gram = np.eye(N, dtype=np.float64) + inv_reg * (Xc @ Xc.T).astype(np.float64)
                    alpha_dual = np.linalg.solve(gram, Yc.astype(np.float64)).astype(np.float32)
                    theta = inv_reg_f32 * (Xc.T @ alpha_dual)
                    weights = theta.T.astype(np.float32)
                    biases = (mu_Y - mu_X @ theta).astype(np.float32)
                    if not return_covariance:
                        return weights, biases
                    K_inv_Xc = np.linalg.solve(gram, (inv_reg_f32 * Xc).astype(np.float64)).astype(np.float32)
                    S_inv = -(inv_reg_f32 * (Xc.T @ K_inv_Xc))
                    S_inv[np.diag_indices(D)] += inv_reg_f32
                    v = S_inv @ mu_X
                    P = np.empty((D + 1, D + 1), dtype=np.float32)
                    P[:D, :D] = S_inv
                    P[:D, D] = -v
                    P[D, :D] = -v
                    P[D, D] = np.float32(1.0 / w_sum + float(mu_X @ v))
                    B = (X_aug.T @ Y_mat).astype(np.float32)
                    return weights, biases, P, B
                except np.linalg.LinAlgError:
                    pass

        reg_matrix = float(regularization) * np.eye(D + 1, dtype=np.float32)
        if not regularize_bias:
            reg_matrix[D, D] = 0.0
        else:
            reg_matrix[D, D] = max(1.0, float(regularization))

        A = X_aug.T @ X_aug + reg_matrix
        B = X_aug.T @ Y_mat

        if not return_covariance:
            try:
                sol = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                sol, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            return sol[:D, :].T.astype(np.float32), sol[D, :].astype(np.float32)

        try:
            P = np.linalg.inv(A)
            sol = P @ B
        except np.linalg.LinAlgError:
            P = np.linalg.pinv(A)
            sol, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

        weights = sol[:D, :].T  # shape: (K, D)
        biases = sol[D, :]      # shape: (K,)
        return (
            weights.astype(np.float32),
            biases.astype(np.float32),
            P.astype(np.float32),
            B.astype(np.float32),
        )

    @staticmethod
    def _prompt_key(prompt: str) -> str:
        return " ".join(prompt.casefold().split())

    def _sample_key(self, row: Sequence[Any]) -> tuple:
        """Identify the input, including numeric state when the question repeats."""
        telemetry = row[2] if len(row) > 2 else None
        vector = ()
        if telemetry is not None:
            values, _ = self.telemetry_projector.extract_vector(telemetry)
            if not np.all(np.isfinite(values)):
                raise ValueError("Teaching telemetry must contain finite numbers")
            vector = tuple(float(value) for value in values)
        return self._prompt_key(row[0]), telemetry is not None, vector

    def _validate_exemplars(self, dataset: Mapping[str, Sequence]) -> Dict[str, list]:
        if not isinstance(dataset, Mapping):
            raise ValueError("Examples must map schema field names to labeled samples")
        unknown = set(dataset) - set(self.schema.fields)
        if unknown:
            raise ValueError(f"Unknown example fields: {sorted(unknown)}")
        validated = {}
        for name, rows in dataset.items():
            field_def = self.schema.fields[name]
            validated[name] = []
            try:
                rows = iter(rows)
            except TypeError as exc:
                raise ValueError(f"Examples for {name!r} must be a sequence of labeled samples") from exc
            for index, row in enumerate(rows):
                if not isinstance(row, (list, tuple)) or len(row) not in (2, 3):
                    raise ValueError(f"{name}[{index}] must contain prompt, label, and optional telemetry")
                prompt, label = row[:2]
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"{name}[{index}] requires a nonempty text prompt")
                if label is None:
                    raise ValueError(f"{name}[{index}] requires an explicit label")
                try:
                    label = field_def.validate_value(label)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid label for {name}[{index}]: {exc}") from exc
                self._sample_key(row)
                validated[name].append((prompt, label, *row[2:]))
        return validated

    def _split_samples(self, samples: list, field_def: Optional[DecisionField], fraction: float) -> Tuple[list, list]:
        groups: Dict[tuple, list] = {}
        for row in samples:
            groups.setdefault(self._sample_key(row), []).append(row)
        # Stratify discrete fields, retaining at least one training group per class.
        strata: Dict[Any, list] = {}
        for key, rows in groups.items():
            label = rows[0][1] if isinstance(field_def, (ChoiceField, BooleanField)) else None
            strata.setdefault(label, []).append(key)
        rng = np.random.RandomState(42)
        held_out = set()
        for keys in strata.values():
            keys = sorted(keys)
            rng.shuffle(keys)
            count = min(len(keys) - 1, max(1, int(len(keys) * fraction))) if fraction else 0
            held_out.update(keys[:count])
        training = [row for row in samples if self._sample_key(row) not in held_out]
        calibration = [row for row in samples if self._sample_key(row) in held_out]
        return training, calibration

    @staticmethod
    def _compute_class_prototypes(
        X_train: np.ndarray,
        Y_train: np.ndarray,
        *,
        is_boolean: bool = False,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute unit-normalized class feature centroids (shape (K, D))."""
        X_mat = np.asarray(X_train, dtype=np.float64)
        N, D = X_mat.shape
        sw = (
            np.maximum(1e-6, np.asarray(sample_weights, dtype=np.float64).flatten())
            if sample_weights is not None
            else None
        )

        def _safe_weighted_mean(rows: np.ndarray, weights: Optional[np.ndarray]) -> Optional[np.ndarray]:
            return _safe_centroid_mean_f64(rows, weights)

        if is_boolean:
            y_flat = np.asarray(Y_train, dtype=np.float32).flatten()
            protos = np.zeros((2, D), dtype=np.float32)
            for idx, mask in enumerate((y_flat < 0.0, y_flat > 0.0)):
                if np.any(mask):
                    vec = _safe_weighted_mean(X_mat[mask], sw[mask] if sw is not None else None)
                    if vec is not None:
                        unit_v, _, ok = _normalize_vec_f32_safe(vec, min_norm=0.0)
                        if ok:
                            protos[idx] = unit_v
            res = _roundtrip_unit_f16(protos)
            assert res is not None
            return res

        Y_mat = np.asarray(Y_train, dtype=np.float32)
        K = Y_mat.shape[1] if Y_mat.ndim > 1 else 1
        Y_mat = Y_mat.reshape(N, K)
        protos = np.zeros((K, D), dtype=np.float32)
        for k in range(K):
            mask = Y_mat[:, k] > 0.5
            if np.any(mask):
                vec = _safe_weighted_mean(X_mat[mask], sw[mask] if sw is not None else None)
                if vec is not None:
                    unit_v, _, ok = _normalize_vec_f32_safe(vec, min_norm=0.0)
                    if ok:
                        protos[k] = unit_v
        res = _roundtrip_unit_f16(protos)
        assert res is not None
        return res

    def _compute_loo_ridge_and_sample_weights(
        self,
        X_train: np.ndarray,
        Y_onehot: np.ndarray,
        *,
        auto_regularization: bool = True,
        noise_robust: bool = True,
    ) -> Tuple[float, Optional[np.ndarray]]:
        """Closed-form Leave-One-Out CV ridge selection and out-of-fold noise downweighting."""
        base_reg = float(self.regularization)
        N, D = X_train.shape
        K = Y_onehot.shape[1]
        if N < 40 or K < 2 or (N / K) < 4.0 or (not auto_regularization and not noise_robust):
            return base_reg, None

        X64 = np.asarray(X_train, dtype=np.float64)
        Y64 = np.asarray(Y_onehot, dtype=np.float64)
        if not np.all(np.isfinite(X64)) or not np.all(np.isfinite(Y64)):
            return base_reg, None

        mu_X = np.mean(X64, axis=0)
        mu_Y = np.mean(Y64, axis=0)
        Xc = X64 - mu_X
        Yc = Y64 - mu_Y
        try:
            if N <= D:
                s2, U = np.linalg.eigh(Xc @ Xc.T)
            else:
                U, s_vals, _ = np.linalg.svd(Xc, full_matrices=False)
                s2 = s_vals * s_vals
        except np.linalg.LinAlgError:
            return base_reg, None

        s2 = np.maximum(0.0, s2)
        UT_Yc = U.T @ Yc
        U_sq = U * U

        best_reg = base_reg
        if auto_regularization and base_reg < 1.0:
            candidates = sorted(
                {round(val, 6) for val in (base_reg, 0.15, 0.35, 0.75, 1.0) if val >= base_reg and val > 0.0}
            )
            best_brier = float("inf")
            for reg in candidates:
                f = s2 / (s2 + reg)
                h_diag = (1.0 / N) + U_sq @ f
                denom = np.maximum(1e-4, 1.0 - h_diag)[:, None]
                Y_hat = mu_Y + U @ (f[:, None] * UT_Yc)
                loo_res = (Y_hat - Y64) / denom
                brier = float(np.mean(np.sum(loo_res * loo_res, axis=1)))
                if reg == base_reg or brier < best_brier * 0.995:
                    best_brier = brier
                    best_reg = float(reg)

        if not noise_robust:
            return best_reg, None

        loo_reg = max(best_reg, 0.5)
        f_loo = s2 / (s2 + loo_reg)
        h_diag_loo = (1.0 / N) + U_sq @ f_loo
        denom_loo = np.maximum(1e-4, 1.0 - h_diag_loo)[:, None]
        Y_hat_loo = mu_Y + U @ (f_loo[:, None] * UT_Yc)
        Y_loo = Y64 + (Y_hat_loo - Y64) / denom_loo

        true_cls = np.argmax(Y64, axis=1)
        s_true = Y_loo[np.arange(N), true_cls]
        rival_mask = (Y64 <= 0.5) & (Y_loo > mu_Y[None, :])
        s_other = np.max(np.where(rival_mask, Y_loo, -1e9), axis=1)
        deficit = s_other - s_true

        if not np.any(deficit > 0.18):
            return best_reg, None

        sw = np.where(
            deficit > 0.18,
            np.clip(1.0 - 2.2 * (deficit - 0.18), 0.20, 1.0),
            1.0,
        ).astype(np.float32)
        s_true_excess = s_true - mu_Y[true_cls]
        sw = np.where(s_true_excess > 0.105, np.maximum(sw, 0.75), sw).astype(np.float32)

        for k in range(K):
            cls_idx = np.where(true_cls == k)[0]
            if len(cls_idx) > 0:
                best_in_cls = int(cls_idx[int(np.argmin(deficit[cls_idx]))])
                if sw[best_in_cls] < 0.85:
                    sw[best_in_cls] = 0.85
                cls_sum = float(np.sum(sw[cls_idx]))
                if cls_sum < 1.0:
                    sw[cls_idx] = np.clip(sw[cls_idx] * (1.0 / max(1e-6, cls_sum)), 0.20, 1.0)

        if bool(np.all(sw > 0.999)):
            return best_reg, None
        return best_reg, sw

    @staticmethod
    def _spherical_kmeans_clusters(
        prototypes: np.ndarray,
        num_clusters: int,
    ) -> Tuple[int, ...]:
        """Deterministic spherical k-means partitioning of K class centroids into M clusters."""
        protos = np.asarray(prototypes, dtype=np.float64)
        K, D = protos.shape
        M = max(1, min(int(num_clusters), K))
        if M <= 1:
            return tuple(0 for _ in range(K))
        if M >= K:
            return tuple(range(K))

        unit_f32, _ = _normalize_rows_f32_safe(protos)
        unit_protos = unit_f32.astype(np.float64)

        # Deterministic farthest-first traversal (k-means++ on cosine similarity)
        sim_matrix = unit_protos @ unit_protos.T
        first_idx = int(np.argmin(np.mean(sim_matrix, axis=1)))
        center_indices = [first_idx]
        for _ in range(1, M):
            max_sim_to_chosen = np.max(sim_matrix[:, center_indices], axis=1)
            for chosen in center_indices:
                max_sim_to_chosen[chosen] = 2.0
            next_idx = int(np.argmin(max_sim_to_chosen))
            center_indices.append(next_idx)

        centers = unit_protos[center_indices].copy()
        assignments = np.zeros(K, dtype=np.int64)

        for _ in range(25):
            sims = unit_protos @ centers.T  # (K, M)
            new_assignments = np.argmax(sims, axis=1)

            # Ensure no cluster is empty
            for m in range(M):
                if not np.any(new_assignments == m):
                    counts = np.bincount(new_assignments, minlength=M)
                    donor_clusters = np.where(counts > 1)[0]
                    if len(donor_clusters) == 0:
                        break
                    donor_mask = np.isin(new_assignments, donor_clusters)
                    donor_indices = np.where(donor_mask)[0]
                    assigned_sims = sims[donor_indices, new_assignments[donor_indices]]
                    move_idx = int(donor_indices[int(np.argmin(assigned_sims))])
                    new_assignments[move_idx] = m

            if np.array_equal(new_assignments, assignments):
                break
            assignments = new_assignments

            for m in range(M):
                members = unit_protos[assignments == m]
                if len(members) > 0:
                    c_vec = np.mean(members, axis=0)
                    c_norm = float(np.linalg.norm(c_vec))
                    if c_norm > 1e-12:
                        centers[m] = c_vec / c_norm

        return tuple(int(c) for c in assignments)

    def _fit_hierarchical_choice_head(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        prototypes: np.ndarray,
        *,
        regularize_bias: bool = True,
        regularization: Optional[float] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Tuple[Tuple[int, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit two-stage coarse cluster router + intra-cluster fine specialist heads."""
        N, D = X_train.shape
        K = Y_train.shape[1]
        eff_reg = float(self.regularization if regularization is None else regularization)
        num_clusters = min(K, max(2, min(int(round(math.sqrt(K))), max(2, K // 2))))
        cluster_assignments = self._spherical_kmeans_clusters(prototypes, num_clusters)
        M = max(cluster_assignments) + 1 if cluster_assignments else 1

        Y_coarse = np.zeros((N, M), dtype=np.float32)
        for k, c_idx in enumerate(cluster_assignments):
            Y_coarse[:, c_idx] += Y_train[:, k]
        Y_coarse = np.clip(Y_coarse, 0.0, 1.0)

        coarse_weights, coarse_biases = self._solve_ridge(
            X_train,
            Y_coarse,
            eff_reg,
            sample_weights=sample_weights,
            regularize_bias=regularize_bias,
        )
        # Zero-center coarse router across clusters so coarse boost is zero-sum
        if M > 1:
            coarse_weights = (coarse_weights - np.mean(coarse_weights, axis=0, keepdims=True)).astype(np.float32)
            coarse_biases = (coarse_biases - np.mean(coarse_biases)).astype(np.float32)

        fine_weights = np.zeros((K, D), dtype=np.float32)
        fine_biases = np.zeros(K, dtype=np.float32)

        for m in range(M):
            member_classes = [k for k, c_idx in enumerate(cluster_assignments) if c_idx == m]
            if len(member_classes) <= 1:
                continue
            cluster_mask = np.sum(Y_train[:, member_classes], axis=1) > 0.5
            if int(np.sum(cluster_mask)) < len(member_classes):
                X_sub = X_train
                Y_sub = Y_train[:, member_classes]
                sw_sub = sample_weights
            else:
                X_sub = X_train[cluster_mask]
                Y_sub = Y_train[cluster_mask][:, member_classes]
                sw_sub = sample_weights[cluster_mask] if sample_weights is not None else None

            w_sub, b_sub = self._solve_ridge(
                X_sub,
                Y_sub,
                max(0.25, eff_reg * 0.5),
                sample_weights=sw_sub,
                regularize_bias=regularize_bias,
            )
            # Zero-center specialist fine head within the cluster for sibling contrast
            w_sub = (w_sub - np.mean(w_sub, axis=0, keepdims=True)).astype(np.float32)
            b_sub = (b_sub - np.mean(b_sub)).astype(np.float32)
            for local_i, global_k in enumerate(member_classes):
                fine_weights[global_k] = w_sub[local_i]
                fine_biases[global_k] = float(b_sub[local_i])

        return (
            cluster_assignments,
            coarse_weights.astype(np.float32),
            coarse_biases.astype(np.float32),
            fine_weights.astype(np.float32),
            fine_biases.astype(np.float32),
        )

    @staticmethod
    def _compute_hierarchical_logits(
        X: np.ndarray,
        weights: np.ndarray,
        biases: np.ndarray,
        cluster_assignments: Sequence[int],
        coarse_weights: Optional[np.ndarray],
        coarse_biases: Optional[np.ndarray],
        fine_weights: Optional[np.ndarray],
        fine_biases: Optional[np.ndarray],
    ) -> np.ndarray:
        """Evaluate combined coarse-to-fine raw logits matching DecisionFieldHead.compute_logits."""
        logits = X @ weights.T + biases
        if (
            coarse_weights is not None
            and coarse_biases is not None
            and len(cluster_assignments) == weights.shape[0]
            and len(X) > 0
        ):
            coarse_logits = X @ coarse_weights.T + coarse_biases
            coarse_probs = _stable_softmax(coarse_logits / 0.25, temperature=1.0)
            assign = np.asarray(cluster_assignments, dtype=np.int64)
            mapped_coarse = coarse_logits[:, assign]
            mapped_probs = coarse_probs[:, assign]
            if fine_weights is not None and fine_biases is not None:
                spec_logits = X @ fine_weights.T + fine_biases
                logits = (
                    0.65 * logits
                    + (0.20 + 0.30 * mapped_probs) * spec_logits
                    + 0.20 * mapped_coarse
                ).astype(np.float32)
            else:
                logits = (0.80 * logits + 0.20 * mapped_coarse).astype(np.float32)
        return logits

    @staticmethod
    def _calibrate_prototype_scores(
        X_conf: np.ndarray,
        prototypes: Optional[np.ndarray],
        prototype_alpha: float,
    ) -> Tuple[float, Tuple[float, ...]]:
        """Compute sorted calibration prototype similarities and empirical alpha quantile."""
        if prototype_alpha <= 0.0 or prototypes is None or prototypes.size == 0 or len(X_conf) == 0:
            return 0.0, ()
        protos_all, valid_rows = _normalize_rows_f32_safe(prototypes)
        if not np.any(valid_rows):
            return 0.0, ()
        protos_valid = protos_all[valid_rows]
        x64 = np.asarray(X_conf, dtype=np.float64)
        if x64.ndim != 2 or len(x64) == 0:
            return 0.0, ()
        fin_rows = np.all(np.isfinite(x64), axis=1)
        if not np.any(fin_rows):
            return 0.0, ()
        X_unit, _ = _normalize_rows_f32_safe(x64[fin_rows])
        max_sims = np.max(X_unit @ protos_valid.T, axis=1).astype(np.float64)
        finite_sims = max_sims[np.isfinite(max_sims)]
        if len(finite_sims) == 0:
            return 0.0, ()
        sorted_sims = np.sort(finite_sims)
        scores_tuple = tuple(float(v) for v in sorted_sims)
        m = len(sorted_sims)
        rank = int(math.floor((m + 1) * prototype_alpha))
        if rank >= 1:
            thresh = float(max(0.0, float(sorted_sims[rank - 1])))
        else:
            min_sim = float(sorted_sims[0])
            thresh = float(max(0.0, min_sim * 0.90)) if min_sim > 0.0 else 0.0
        return thresh, scores_tuple

    @staticmethod
    def _normalize_soft_probs(
        raw_probs: Any,
        f_def: Union[ChoiceField, BooleanField],
        *,
        expected_len: Optional[int] = None,
    ) -> np.ndarray:
        arr = np.asarray(raw_probs, dtype=np.float64)
        if isinstance(f_def, BooleanField):
            if arr.ndim == 1:
                arr = np.column_stack([1.0 - arr, arr])
            elif arr.ndim == 2 and arr.shape[1] == 1:
                p_true = arr[:, 0]
                arr = np.column_stack([1.0 - p_true, p_true])
            elif arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(f"soft_probs for BooleanField {f_def.name!r} must have shape (N, 2) or (N,)")
        elif isinstance(f_def, ChoiceField):
            K = len(f_def.options)
            if arr.ndim != 2 or arr.shape[1] != K:
                raise ValueError(f"soft_probs for ChoiceField {f_def.name!r} must have shape (N, {K})")
        else:
            raise ValueError(f"Soft distillation supports ChoiceField and BooleanField, got {type(f_def).__name__}")
        if arr.shape[0] < 2:
            raise ValueError("Soft distillation requires at least 2 examples")
        if expected_len is not None and arr.shape[0] != expected_len:
            raise ValueError(f"soft_probs length ({arr.shape[0]}) does not match exemplars ({expected_len})")
        if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
            raise ValueError("soft_probs must contain finite non-negative probabilities")
        row_sums = arr.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 1e-12):
            raise ValueError("Each soft_probs row must sum to a positive value")
        return arr / row_sums

    @staticmethod
    def _encode_texts_with_projector(
        texts: Sequence[str],
        projector: Any,
        *,
        recency_weighted: bool = False,
    ) -> np.ndarray:
        text_list = [str(t) for t in texts]
        if not text_list:
            dim = int(getattr(projector, "dimension", 0) or 0)
            return np.empty((0, dim), dtype=np.float32)
        eff_rw = bool(recency_weighted) and not isinstance(projector, (TfidfProjector, PragmaticCompositeProjector))
        proj_fn = getattr(projector, "project", None)
        pwn_fn = getattr(projector, "project_with_norm", None)
        batch_fn = getattr(projector, "project_batch", None)
        one_rw = _accepts_recency_weighted(proj_fn)
        pwn_rw = _accepts_recency_weighted(pwn_fn)
        batch_rw = _accepts_recency_weighted(batch_fn)
        if (
            (not _is_native_scalar_projector(projector))
            and callable(batch_fn)
            and (batch_rw or not (one_rw or pwn_rw))
        ):
            raw_mat = (
                batch_fn(text_list, recency_weighted=eff_rw)
                if batch_rw
                else batch_fn(text_list)
            )
            raw_mat_arr = np.asarray(raw_mat)
            dim_out = int(raw_mat_arr.shape[1]) if raw_mat_arr.ndim == 2 else 0
            return _normalize_raw_projection_batch(raw_mat_arr, len(text_list), dim_out, zero_unit0=False)
        rows = []
        for p in text_list:
            if callable(pwn_fn) and (pwn_rw or not one_rw):
                v, _ = pwn_fn(p, recency_weighted=eff_rw) if pwn_rw else pwn_fn(p)
            elif callable(proj_fn):
                v = proj_fn(p, recency_weighted=eff_rw) if one_rw else proj_fn(p)
            else:
                v = projector(p)
            v_arr = np.asarray(v).ravel()
            rows.append(_normalize_raw_projection(v_arr, int(v_arr.shape[0]), zero_unit0=False))
        return np.stack(rows, axis=0)

    @staticmethod
    def _stratified_folds(y_idx: np.ndarray, k_classes: int) -> Tuple[int, np.ndarray]:
        N = len(y_idx)
        n_folds = min(5, max(2, N // max(4, k_classes)))
        if N < n_folds:
            n_folds = max(2, N)
        fold_ids = np.zeros(N, dtype=np.int64)
        fc = 0
        for c in range(max(1, k_classes)):
            idxs = np.where(y_idx == c)[0]
            for i in idxs:
                fold_ids[i] = fc % n_folds
                fc += 1
        return n_folds, fold_ids

    def _soft_distill_discrete_head(
        self,
        field_name: str,
        f_def: Union[ChoiceField, BooleanField],
        prompts: Sequence[str],
        P_T: np.ndarray,
        *,
        y_true_idx: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        beta_grid: Sequence[float] = (0.15, 0.35, 0.55),
        reg_grid: Sequence[float] = (0.15, 0.5, 1.5, 5.0),
        X_precomputed: Optional[np.ndarray] = None,
        fold_projector_factory: Optional[Any] = None,
        mondrian: bool = False,
        fam_thresh: float = 0.0,
    ) -> CompiledHeadWeights:
        N, K = P_T.shape
        y_arg = (
            np.asarray(y_true_idx, dtype=np.int64)
            if y_true_idx is not None
            else np.argmax(P_T, axis=1)
        )
        Y_hard_oh = np.eye(K, dtype=np.float64)[y_arg] if y_true_idx is not None else None
        X_all = (
            np.asarray(X_precomputed, dtype=np.float32)
            if X_precomputed is not None
            else self._encode_texts_with_projector(prompts, self.projector, recency_weighted=self.recency_weighted)
        )

        n_folds, fold_ids = self._stratified_folds(y_arg, K)
        fold_features: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        prompt_list = [str(p) for p in prompts]
        for f in range(n_folds):
            tri = np.where(fold_ids != f)[0]
            vai = np.where(fold_ids == f)[0]
            if len(tri) == 0 or len(vai) == 0:
                continue
            if fold_projector_factory is not None:
                fold_proj = fold_projector_factory(
                    [prompt_list[i] for i in tri],
                    [int(y_arg[i]) for i in tri],
                )
                Xf_tr = self._encode_texts_with_projector([prompt_list[i] for i in tri], fold_proj).astype(np.float64)
                Xf_va = self._encode_texts_with_projector([prompt_list[i] for i in vai], fold_proj).astype(np.float64)
            else:
                Xf_tr = X_all[tri].astype(np.float64)
                Xf_va = X_all[vai].astype(np.float64)
            fold_features.append((tri, vai, Xf_tr, Xf_va))

        betas = tuple(float(b) for b in beta_grid) if Y_hard_oh is not None else (0.0,)
        regs = tuple(max(1e-4, float(r)) for r in reg_grid)
        best_nll = float("inf")
        best_beta = betas[0]
        best_reg = regs[0]
        best_oof_logits = np.zeros((N, K), dtype=np.float64)

        for beta_v in betas:
            for reg_v in regs:
                oof_logits = np.zeros((N, K), dtype=np.float64)
                for tri, vai, Xf_tr, Xf_va in fold_features:
                    U_f, c_f, _, _ = _solve_soft_ridge_student_raw(
                        Xf_tr,
                        P_T[tri],
                        (Y_hard_oh[tri] if Y_hard_oh is not None else None),
                        beta=beta_v,
                        reg=reg_v,
                    )
                    oof_logits[vai] = Xf_va @ U_f.T + c_f
                p_cand = _stable_softmax(oof_logits, temperature=1.0)
                nll = -float(np.mean(np.sum(P_T * np.log(np.clip(p_cand, 1e-12, 1.0)), axis=1)))
                if nll < best_nll:
                    best_nll = nll
                    best_beta = beta_v
                    best_reg = reg_v
                    best_oof_logits = oof_logits

        T_soft = _soft_fit_temperature_np(best_oof_logits, P_T)
        p_oof = _stable_softmax(best_oof_logits / T_soft, temperature=1.0)
        q_lac, p_floor = _oof_conservative_soft_conformal(
            p_oof,
            P_T,
            y_true_idx=(y_arg if y_true_idx is not None else None),
            alpha=float(alpha),
        )

        lac_scores = 1.0 - p_oof[np.arange(N), y_arg]
        sorted_calib_scores = np.sort(lac_scores)
        options = f_def.options if isinstance(f_def, ChoiceField) else ("False", "True")
        per_class_scores = (
            {
                options[c]: tuple(float(v) for v in np.sort(lac_scores[y_arg == c]))
                for c in range(K)
                if np.any(y_arg == c)
            }
            if mondrian
            else {}
        )
        err_mask = np.argmax(p_oof, axis=1) != y_arg
        if np.any(err_mask):
            sorted_p = -np.sort(-p_oof[err_mask], axis=1)
            runner_diff = sorted_p[:, 0] - (sorted_p[:, 1] if K > 1 else 0.0)
            m_thresh = float(np.clip(np.quantile(runner_diff, 0.95) + 0.05, 0.15, 0.85))
        else:
            m_thresh = 0.20

        U_full, c_full, Z_full, sw_full = _solve_soft_ridge_student_raw(
            X_all.astype(np.float64),
            P_T,
            Y_hard_oh,
            beta=best_beta,
            reg=best_reg,
        )
        if isinstance(f_def, BooleanField):
            weights = (0.25 * (U_full[1:2] - U_full[0:1])).astype(np.float32)
            biases = np.array([0.25 * float(c_full[1] - c_full[0])], dtype=np.float32)
            target_scaled = (0.25 * (Z_full[:, 1] - Z_full[:, 0])).reshape(-1, 1).astype(np.float32)
            head_options: Tuple[str, ...] = ()
        else:
            weights = (0.25 * U_full).astype(np.float32)
            biases = (0.25 * c_full).astype(np.float32)
            target_scaled = (0.25 * Z_full).astype(np.float32)
            head_options = f_def.options

        _, _, P_mat, B_mat = self._solve_ridge(
            X_all,
            target_scaled,
            best_reg,
            sample_weights=sw_full.astype(np.float32),
            regularize_bias=False,
            return_covariance=True,
        )
        tau0 = max(0.0, float(p_floor) - (1.0 / max(1, K)))
        return CompiledHeadWeights(
            field_name=field_name,
            field_type=f_def.field_type,
            weights=weights,
            biases=biases,
            temperature=float(T_soft),
            conformal_quantile=float(q_lac),
            options=head_options,
            P=P_mat,
            B=B_mat,
            margin_threshold=m_thresh,
            forgetting_factor=self.forgetting_factor,
            relative_odds_ratio=self.relative_odds_ratio,
            confidence_floor_tau0=float(tau0),
            escalate_on_ambiguity=getattr(f_def, "escalate_on_ambiguity", True),
            calibration_scores=tuple(float(s) for s in sorted_calib_scores),
            score_method="lac",
            mondrian=bool(mondrian),
            class_calibration_scores=per_class_scores,
            familiarity_threshold=float(fam_thresh),
        )

    def compile_soft_distilled(
        self,
        prompts: Union[Sequence[str], Mapping[str, Sequence[str]]],
        *,
        soft_probs: Union[Sequence[Sequence[float]], Mapping[str, Sequence[Sequence[float]]]],
        hard_labels: Optional[Union[Sequence[Any], Mapping[str, Sequence[Any]]]] = None,
        field_name: Optional[str] = None,
        alpha: Optional[float] = None,
        beta_grid: Sequence[float] = (0.15, 0.35, 0.55),
        reg_grid: Sequence[float] = (0.15, 0.5, 1.5, 5.0),
        semantic_mode: Optional[str] = None,
        pragmatic: bool = False,
        max_features: int = 2048,
        prag_max_dim: int = 384,
    ) -> CompiledSystemOneModel:
        """Distill continuous teacher probabilities into a calibrated zero-egress .s1m student."""
        eff_alpha = 0.05 if alpha is None else float(alpha)
        if not math.isfinite(eff_alpha) or not (0.0 < eff_alpha < 1.0):
            raise ValueError("alpha must be finite and in (0, 1)")
        eff_semantic_mode = self.semantic_mode if semantic_mode is None else semantic_mode
        if not isinstance(eff_semantic_mode, str) or eff_semantic_mode not in VALID_SEMANTIC_MODES:
            raise ValueError(
                f"Unsupported semantic_mode {eff_semantic_mode!r}; expected one of {VALID_SEMANTIC_MODES}"
            )

        if isinstance(prompts, Mapping):
            if not isinstance(soft_probs, Mapping):
                raise ValueError("soft_probs must be a mapping when prompts is a mapping")
            prompt_map: Dict[str, List[str]] = {k: [str(x) for x in v] for k, v in prompts.items()}
            soft_map_raw: Dict[str, Any] = dict(soft_probs)
            hard_map_raw: Dict[str, Any] = dict(hard_labels) if isinstance(hard_labels, Mapping) else {}
        else:
            if isinstance(soft_probs, Mapping):
                raise ValueError("soft_probs must be a sequence/array when prompts is a sequence")
            if field_name is not None:
                target_field = field_name
            elif len(self.schema.fields) == 1:
                target_field = next(iter(self.schema.fields))
            else:
                raise ValueError("field_name is required for multi-field schemas")
            prompt_map = {target_field: [str(x) for x in prompts]}
            soft_map_raw = {target_field: soft_probs}
            hard_map_raw = {target_field: hard_labels} if hard_labels is not None else {}

        if not prompt_map or set(prompt_map) != set(soft_map_raw):
            raise ValueError("prompts and soft_probs must cover the same schema field(s)")

        parsed_fields: Dict[str, Dict[str, Any]] = {}
        all_texts: List[str] = []
        proj_labels: List[str] = []
        multi_field = len(prompt_map) > 1

        for fname, p_list in prompt_map.items():
            f_def = self.schema.fields.get(fname)
            if f_def is None or not isinstance(f_def, (ChoiceField, BooleanField)):
                raise ValueError(f"Unsupported field for soft distillation: {fname!r}")
            P_norm = self._normalize_soft_probs(soft_map_raw[fname], f_def, expected_len=len(p_list))
            y_idx_arr: Optional[np.ndarray] = None
            raw_hard = hard_map_raw.get(fname)
            if raw_hard is not None:
                if len(raw_hard) != len(p_list):
                    raise ValueError(f"hard_labels length mismatch for field {fname!r}")
                if isinstance(f_def, ChoiceField):
                    opt_idx = {opt: i for i, opt in enumerate(f_def.options)}
                    idx_list = []
                    for lbl in raw_hard:
                        if lbl not in opt_idx:
                            raise ValueError(f"Invalid hard label {lbl!r} for field {fname!r}")
                        idx_list.append(opt_idx[lbl])
                    y_idx_arr = np.asarray(idx_list, dtype=np.int64)
                else:
                    y_idx_arr = np.asarray([1 if bool(lbl) else 0 for lbl in raw_hard], dtype=np.int64)
            parsed_fields[fname] = {"f_def": f_def, "prompts": p_list, "P_T": P_norm, "y_idx": y_idx_arr}
            lbl_source = y_idx_arr if y_idx_arr is not None else np.argmax(P_norm, axis=1)
            for txt, idx_val in zip(p_list, lbl_source):
                all_texts.append(txt)
                proj_labels.append(f"{fname}:{int(idx_val)}" if multi_field else str(int(idx_val)))

        from system1.core.embeddings import HybridProjector

        needs_hybrid_fit = (
            eff_semantic_mode in ("v1_1", "v1_2_fisher", "v1_2_rbf")
            and (
                not self._explicit_projector
                or (isinstance(self.projector, HybridProjector) and not getattr(self.projector, "_fitted", False))
            )
        )
        fold_factory = None
        if needs_hybrid_fit:
            fit_labels = proj_labels if len(set(proj_labels)) >= 2 else None
            extra_fit_kw: Dict[str, Any] = {}
            if isinstance(self.projector, HybridProjector):
                extra_fit_kw["sparse_dim"] = self.projector.sparse_dim
                extra_fit_kw["dense_dim"] = self.projector.dense_dim
                extra_fit_kw["alpha"] = self.projector.alpha
                extra_fit_kw["seed"] = self.projector.seed
            if eff_semantic_mode == "v1_1":
                fitted_proj = HybridProjector.fit(
                    all_texts,
                    labels=fit_labels,
                    backend=self.backend,
                    **extra_fit_kw,
                )
            else:
                fitted_proj = HybridProjector.fit_v1_2(
                    all_texts,
                    labels=fit_labels,
                    schema=self.schema,
                    backend=self.backend,
                    attach_rbf=bool(eff_semantic_mode == "v1_2_rbf" and fit_labels is not None),
                    **extra_fit_kw,
                )
            self.projector = fitted_proj
            self.dimension = int(fitted_proj.dimension)
            self.telemetry_projector = TelemetryProjector(embedding_dim=self.dimension)
        elif pragmatic and (not self._explicit_projector or not isinstance(self.projector, PragmaticCompositeProjector)):
            base_cap = min(int(max_features), MAX_SKILL_DIMENSION - int(prag_max_dim))
            eff_min_df = 1 if len(all_texts) < 24 else (2 if len(all_texts) < 48 else 3)
            fit_labels = proj_labels if len(set(proj_labels)) >= 2 else None
            fitted_prag = PragmaticCompositeProjector.fit(
                all_texts,
                labels=fit_labels,
                max_features=max(16, base_cap),
                prag_max_dim=min(int(prag_max_dim), MAX_PRAG_FEATURES),
                min_df=eff_min_df,
                char_ngrams=(3, 5),
            )
            self.projector = fitted_prag
            self.dimension = int(fitted_prag.dimension)
            self.telemetry_projector = TelemetryProjector(embedding_dim=self.dimension)
            if not multi_field:
                def fold_factory(tr_texts: List[str], tr_lbls: List[int]) -> PragmaticCompositeProjector:
                    flabels = [str(v) for v in tr_lbls] if len(set(tr_lbls)) >= 2 else None
                    mdf = 1 if len(tr_texts) < 24 else (2 if len(tr_texts) < 48 else 3)
                    return PragmaticCompositeProjector.fit(
                        tr_texts,
                        labels=flabels,
                        max_features=max(16, base_cap),
                        prag_max_dim=min(int(prag_max_dim), MAX_PRAG_FEATURES),
                        min_df=mdf,
                        char_ngrams=(3, 5),
                    )
        elif not self._explicit_projector:
            feat_cap = min(int(max_features), MAX_SKILL_DIMENSION)
            fit_labels = proj_labels if len(set(proj_labels)) >= 2 else None
            fitted_tfidf = TfidfProjector.fit(
                all_texts,
                max_features=max(16, feat_cap),
                labels=fit_labels,
                selection="chi2" if fit_labels is not None else "frequency",
                char_ngrams=(3, 5),
            )
            self.projector = fitted_tfidf
            self.dimension = int(fitted_tfidf.dimension)
            self.telemetry_projector = TelemetryProjector(embedding_dim=self.dimension)
            if not multi_field:
                def fold_factory(tr_texts: List[str], tr_lbls: List[int]) -> TfidfProjector:
                    flabels = [str(v) for v in tr_lbls] if len(set(tr_lbls)) >= 2 else None
                    return TfidfProjector.fit(
                        tr_texts,
                        max_features=max(16, feat_cap),
                        labels=flabels,
                        selection="chi2" if flabels is not None else "frequency",
                        char_ngrams=(3, 5),
                    )

        eff_recency = (
            False
            if isinstance(self.projector, (TfidfProjector, PragmaticCompositeProjector))
            else self.recency_weighted
        )
        compiled_heads: Dict[str, CompiledHeadWeights] = {}
        sample_counts: Dict[str, Dict[str, int]] = {}
        for fname, f_def in self.schema.fields.items():
            info = parsed_fields.get(fname)
            if info is None:
                head = DecisionFieldHead(f_def, dimension=self.dimension, projector=self.projector)
                compiled_heads[fname] = CompiledHeadWeights(
                    field_name=fname,
                    field_type=f_def.field_type,
                    weights=head.weights,
                    biases=head.biases,
                    temperature=1.0,
                    conformal_quantile=0.0,
                    options=getattr(f_def, "options", ()),
                    forgetting_factor=self.forgetting_factor,
                    relative_odds_ratio=self.relative_odds_ratio,
                    confidence_floor_tau0=self.confidence_floor_tau0,
                    escalate_on_ambiguity=getattr(f_def, "escalate_on_ambiguity", True),
                )
                continue
            head_w = self._soft_distill_discrete_head(
                fname,
                info["f_def"],
                info["prompts"],
                info["P_T"],
                y_true_idx=info["y_idx"],
                alpha=eff_alpha,
                beta_grid=beta_grid,
                reg_grid=reg_grid,
                fold_projector_factory=fold_factory,
                mondrian=self.mondrian,
            )
            compiled_heads[fname] = head_w
            sample_counts[fname] = {
                "fit": len(info["prompts"]),
                "calibration": len(info["prompts"]),
                "temperature": len(info["prompts"]),
                "generated": 0,
            }

        return CompiledSystemOneModel(
            schema=self.schema,
            heads=compiled_heads,
            dimension=self.dimension,
            projector=self.projector,
            backend=self.backend,
            forgetting_factor=self.forgetting_factor,
            recency_weighted=eff_recency,
            metadata={
                "regularization": self.regularization,
                "choice_solver": "soft_ridge_oof",
                "teaching_mode": "soft_distilled",
                "pragmatic": bool(isinstance(self.projector, PragmaticCompositeProjector)),
                "semantic_mode": eff_semantic_mode,
                "alpha": eff_alpha,
                "sample_counts": sample_counts,
                "compiled_at": time.time(),
            },
        )

    def compile(
        self,
        exemplars: Optional[Dict[str, Sequence[Tuple[str, Any]]]] = None,
        *,
        samples_per_choice: int = 20,
        teacher: str = "synthetic",
        calibration_split: float = 0.25,
        augment: bool = True,
        calibration_exemplars: Optional[Dict[str, Sequence[Tuple[str, Any]]]] = None,
        mondrian: Optional[bool] = None,
        familiarity_alpha: Optional[float] = None,
        prototype_alpha: Optional[float] = None,
        hierarchical: Optional[Union[bool, str]] = None,
        auto_regularization: Optional[bool] = None,
        noise_robust: Optional[bool] = None,
        label_noise_robust: Optional[bool] = None,
        semantic_mode: Optional[str] = None,
        proto_blend: Optional[float] = None,
        fisher_reg: Optional[float] = None,
        soft_targets: Optional[Mapping[str, Sequence[Sequence[float]]]] = None,
    ) -> CompiledSystemOneModel:
        """Compile a local model, optionally using supplied examples only.

        Set ``augment=False`` to teach only from the examples you supply.
        Supply ``calibration_exemplars`` to control the held-out calibration set,
        or reserve a fraction of unique inputs with ``calibration_split``.
        An input is the normalized prompt plus its optional numeric telemetry.
        A zero split produces an uncalibrated model; it never calibrates on training data.
        Related paraphrases/workflows should be split by the caller before compilation.
        """
        eff_mondrian = self.mondrian if mondrian is None else bool(mondrian)
        eff_familiarity_alpha = (
            self.familiarity_alpha if familiarity_alpha is None else float(familiarity_alpha)
        )
        if not math.isfinite(eff_familiarity_alpha) or not 0.0 <= eff_familiarity_alpha < 1.0:
            raise ValueError("familiarity_alpha must be finite and in [0, 1)")
        eff_prototype_alpha = (
            self.prototype_alpha if prototype_alpha is None else float(prototype_alpha)
        )
        if not math.isfinite(eff_prototype_alpha) or not 0.0 <= eff_prototype_alpha < 1.0:
            raise ValueError("prototype_alpha must be finite and in [0, 1)")
        eff_hierarchical = self.hierarchical if hierarchical is None else hierarchical
        if not isinstance(eff_hierarchical, bool) and eff_hierarchical != "auto":
            raise ValueError("hierarchical must be a boolean or 'auto'")
        if auto_regularization is not None and not isinstance(auto_regularization, bool):
            raise ValueError("auto_regularization must be a boolean")
        eff_auto_reg = self.auto_regularization if auto_regularization is None else auto_regularization
        req_noise = noise_robust if label_noise_robust is None else label_noise_robust
        if req_noise is not None and not isinstance(req_noise, bool):
            raise ValueError("noise_robust must be a boolean")
        eff_noise_robust = self.noise_robust if req_noise is None else req_noise
        eff_semantic_mode = self.semantic_mode if semantic_mode is None else semantic_mode
        if not isinstance(eff_semantic_mode, str) or eff_semantic_mode not in VALID_SEMANTIC_MODES:
            raise ValueError(
                f"Unsupported semantic_mode {eff_semantic_mode!r}; expected one of {VALID_SEMANTIC_MODES}"
            )
        raw_pb = self.proto_blend if proto_blend is None else float(proto_blend)
        if raw_pb is None:
            eff_proto_blend = 0.22 if eff_semantic_mode in ("v1_2_fisher", "v1_2_rbf") else 0.0
        else:
            if not math.isfinite(raw_pb) or not (0.0 <= raw_pb <= 1.0):
                raise ValueError("proto_blend must be finite and in [0, 1]")
            eff_proto_blend = float(raw_pb)
        raw_fr = self.fisher_reg if fisher_reg is None else float(fisher_reg)
        if raw_fr is not None:
            if not math.isfinite(raw_fr) or raw_fr <= 0.0:
                raise ValueError("fisher_reg must be finite and positive")
            eff_fisher_reg: Optional[float] = float(raw_fr)
        else:
            eff_fisher_reg = None
        if not math.isfinite(calibration_split) or not 0.0 <= calibration_split < 1.0:
            raise ValueError("calibration_split must be finite and in [0, 1)")
        if not augment and exemplars is None:
            raise ValueError("Teaching without augmentation requires explicit exemplars")
        if calibration_exemplars is not None and exemplars is None:
            raise ValueError("Explicit calibration requires explicit training exemplars")
        if exemplars is None:
            dataset = self.generate_synthetic_exemplars(
                samples_per_choice=samples_per_choice,
                teacher=teacher,
            )
        else:
            dataset = exemplars

        dataset = self._validate_exemplars(dataset)
        calibration_data = (
            self._validate_exemplars(calibration_exemplars)
            if calibration_exemplars is not None else None
        )
        if calibration_data is not None:
            training_inputs = {self._sample_key(row) for rows in dataset.values() for row in rows}
            calibration_inputs = {self._sample_key(row) for rows in calibration_data.values() for row in rows}
            if training_inputs & calibration_inputs:
                raise ValueError("Training and calibration inputs must be disjoint")

        validated_soft: Dict[str, np.ndarray] = {}
        if soft_targets is not None:
            if not isinstance(soft_targets, Mapping):
                raise ValueError("soft_targets must be a mapping of field_name -> probability sequences")
            for s_fname, s_probs in soft_targets.items():
                f_def_s = self.schema.fields.get(s_fname)
                if f_def_s is None or not isinstance(f_def_s, (ChoiceField, BooleanField)):
                    raise ValueError(f"soft_targets supports ChoiceField and BooleanField only, got {s_fname!r}")
                raw_rows = dataset.get(s_fname, [])
                validated_soft[s_fname] = self._normalize_soft_probs(
                    s_probs, f_def_s, expected_len=len(raw_rows)
                )

        compiled_heads: Dict[str, CompiledHeadWeights] = {}
        sample_counts: Dict[str, Dict[str, int]] = {}
        partitioned_fields: Dict[str, Optional[Dict[str, Any]]] = {}
        all_train_texts: List[str] = []
        discrete_texts: List[str] = []
        discrete_labels: List[str] = []
        multi_field_schema = len(self.schema.fields) > 1

        for field_name, f_def in self.schema.fields.items():
            samples = list(dataset.get(field_name, []))
            if not samples:
                if not augment:
                    raise ValueError(f"Teaching requires examples for field {field_name!r}")
                partitioned_fields[field_name] = None
                continue

            soft_probs_field = validated_soft.get(field_name)
            if soft_probs_field is not None:
                calibration_samples = list(calibration_data.get(field_name, [])) if calibration_data is not None else []
                num_train = len(samples)
                num_calib = len(calibration_samples)
                sample_counts[field_name] = {
                    "fit": num_train,
                    "calibration": num_calib if num_calib > 0 else num_train,
                    "temperature": num_calib // 2 if num_calib >= 2 else num_train,
                    "generated": 0,
                }
                for row in samples:
                    p_txt = row[0]
                    if isinstance(p_txt, str):
                        all_train_texts.append(p_txt)
                        discrete_texts.append(p_txt)
                        lbl_str = str(row[1])
                        discrete_labels.append(f"{field_name}:{lbl_str}" if multi_field_schema else lbl_str)
                partitioned_fields[field_name] = {
                    "samples": samples + calibration_samples,
                    "num_train": num_train,
                    "num_calib": num_calib,
                    "num_temperature": (num_calib // 2 if num_calib >= 2 else 0),
                    "generated_count": 0,
                    "soft_probs": soft_probs_field,
                }
                continue

            # Teaching from examples keeps repeated inputs in one partition.
            # Preserve the existing synthetic-demo recipe for compatibility.
            if calibration_data is not None:
                calibration_samples = list(calibration_data.get(field_name, []))
            elif not augment:
                samples, calibration_samples = self._split_samples(samples, f_def, calibration_split)
            else:
                calibration_samples = []
            if not augment:
                observed = {row[1] for row in samples} if isinstance(f_def, (ChoiceField, BooleanField)) else None
                required = set(f_def.options) if isinstance(f_def, ChoiceField) else {False, True}
                if observed is not None and not required.issubset(observed):
                    raise ValueError(f"Training examples for {field_name!r} must cover every class")
            real_training_count = len(samples)
            if augment:
                # Class balance and synthetic augmentation across all field types
                if isinstance(f_def, ChoiceField):
                    counts = {opt: sum(1 for s in samples if s[1] == opt) for opt in f_def.options}
                    max_c = max(counts.values()) if counts else 0
                    target_c = max(max_c, samples_per_choice, 15)
                    synth_all = self.generate_synthetic_exemplars(samples_per_choice=target_c)
                    synth_field = synth_all.get(field_name, [])

                    for opt in f_def.options:
                        curr_samples = [s for s in samples if s[1] == opt]
                        deficit = target_c - len(curr_samples)
                        if deficit > 0:
                            synth_opt = [s for s in synth_field if s[1] == opt]
                            if not synth_opt:
                                synth_opt = [(f"Operation {opt}: {f_def.descriptions.get(opt, opt)}", opt)]
                            for idx in range(deficit):
                                samples.append(synth_opt[idx % len(synth_opt)])

                elif isinstance(f_def, BooleanField):
                    true_count = sum(1 for s in samples if bool(s[1]))
                    false_count = sum(1 for s in samples if not bool(s[1]))
                    target_c = max(true_count, false_count, samples_per_choice, 15)
                    synth_all = self.generate_synthetic_exemplars(samples_per_choice=target_c)
                    synth_field = synth_all.get(field_name, [])

                    true_deficit = target_c - true_count
                    false_deficit = target_c - false_count
                    if true_deficit > 0:
                        synth_true = [s for s in synth_field if bool(s[1])]
                        for idx in range(true_deficit):
                            samples.append(synth_true[idx % len(synth_true)])
                    if false_deficit > 0:
                        synth_false = [s for s in synth_field if not bool(s[1])]
                        for idx in range(false_deficit):
                            samples.append(synth_false[idx % len(synth_false)])

                elif isinstance(f_def, MultiChoiceField):
                    synth_all = self.generate_synthetic_exemplars(samples_per_choice=15)
                    synth_field = synth_all.get(field_name, [])
                    for p, lab in synth_field:
                        samples.append((p, lab))

                elif isinstance(f_def, ScoreField):
                    synth_all = self.generate_synthetic_exemplars(samples_per_choice=samples_per_choice)
                    synth_field = synth_all.get(field_name, [])

                    def _sc_unit(sample_pair: Any) -> float:
                        return float(
                            _score_to_unit_ratio(float(sample_pair[1]), f_def.min_value, f_def.max_value)
                        )

                    low_synth = [s for s in synth_field if _sc_unit(s) <= 0.33]
                    mid_synth = [s for s in synth_field if 0.33 < _sc_unit(s) <= 0.67]
                    high_synth = [s for s in synth_field if _sc_unit(s) > 0.67]

                    low_real = [s for s in samples if _sc_unit(s) <= 0.33]
                    mid_real = [s for s in samples if 0.33 < _sc_unit(s) <= 0.67]
                    high_real = [s for s in samples if _sc_unit(s) > 0.67]

                    target_c = max(len(low_real), len(mid_real), len(high_real), samples_per_choice, 15)

                    if low_synth:
                        for idx in range(target_c - len(low_real)):
                            samples.append(low_synth[idx % len(low_synth)])
                    if mid_synth:
                        for idx in range(target_c - len(mid_real)):
                            samples.append(mid_synth[idx % len(mid_synth)])
                    if high_synth:
                        for idx in range(target_c - len(high_real)):
                            samples.append(high_synth[idx % len(high_synth)])

                # Synthetic templates must not duplicate held-out inputs either.
                held_out = {self._sample_key(row) for row in calibration_samples}
                samples = [row for row in samples if self._sample_key(row) not in held_out]
            if not samples:
                raise ValueError(f"No training examples remain for field {field_name!r}")
            generated_count = len(samples) - real_training_count
            if augment and calibration_data is None:
                rng = np.random.RandomState(42)
                samples = [samples[i] for i in rng.permutation(len(samples))]
                count = max(1, int(len(samples) * calibration_split)) if calibration_split and len(samples) >= 4 else 0
                samples, calibration_samples = samples[:len(samples) - count], samples[len(samples) - count:]
            # Temperature and conformal calibration also need disjoint inputs.
            num_temperature = 0
            if isinstance(f_def, ChoiceField) and (not augment or eff_mondrian):
                temperature_samples, conformal_samples = self._split_samples(
                    calibration_samples, f_def if eff_mondrian else None, 0.5
                )
                if conformal_samples:
                    num_temperature = len(temperature_samples)
                    calibration_samples = temperature_samples + conformal_samples
            elif isinstance(f_def, ChoiceField) and len(calibration_samples) >= 2:
                if len(calibration_samples) >= 40 or len(calibration_samples) <= 20:
                    num_temperature = len(calibration_samples) // 2
                else:
                    num_temperature = len(calibration_samples) - 20
            num_train, num_calib = len(samples), len(calibration_samples)
            sample_counts[field_name] = {
                "fit": num_train, "calibration": num_calib,
                "temperature": num_temperature,
                "generated": generated_count,
            }
            for row in samples:
                p_txt = row[0]
                if isinstance(p_txt, str):
                    all_train_texts.append(p_txt)
                    if isinstance(f_def, (ChoiceField, BooleanField)):
                        discrete_texts.append(p_txt)
                        lbl_str = str(row[1])
                        discrete_labels.append(f"{field_name}:{lbl_str}" if multi_field_schema else lbl_str)
            partitioned_fields[field_name] = {
                "samples": samples + calibration_samples,
                "num_train": num_train,
                "num_calib": num_calib,
                "num_temperature": num_temperature,
                "generated_count": generated_count,
            }

        if eff_semantic_mode in ("v1_1", "v1_2_fisher", "v1_2_rbf") and all_train_texts:
            from system1.core.embeddings import HybridProjector

            should_fit_proj = not self._explicit_projector or (
                isinstance(self.projector, HybridProjector)
                and not getattr(self.projector, "_fitted", False)
            )
            has_all_discrete = (
                len(discrete_labels) == len(all_train_texts)
                and len(set(discrete_labels)) >= 2
            )
            has_some_discrete = len(set(discrete_labels)) >= 2
            fit_labels = discrete_labels if has_all_discrete else None

            if should_fit_proj:
                extra_fit_kw: Dict[str, Any] = {}
                if isinstance(self.projector, HybridProjector):
                    extra_fit_kw["sparse_dim"] = self.projector.sparse_dim
                    extra_fit_kw["dense_dim"] = self.projector.dense_dim
                    extra_fit_kw["alpha"] = self.projector.alpha
                    extra_fit_kw["seed"] = self.projector.seed

                if eff_semantic_mode == "v1_1":
                    fitted_proj = HybridProjector.fit(
                        all_train_texts,
                        labels=fit_labels,
                        backend=self.backend,
                        **extra_fit_kw,
                    )
                else:
                    fitted_proj = HybridProjector.fit_v1_2(
                        all_train_texts,
                        labels=fit_labels,
                        schema=self.schema,
                        backend=self.backend,
                        attach_rbf=bool(eff_semantic_mode == "v1_2_rbf" and has_all_discrete),
                        **extra_fit_kw,
                    )
                    if eff_semantic_mode == "v1_2_rbf" and not has_all_discrete and has_some_discrete:
                        fitted_proj.attach_rbf_landmarks(discrete_texts, discrete_labels)
                self.projector = fitted_proj
                self.dimension = int(fitted_proj.dimension)
                self.telemetry_projector = TelemetryProjector(embedding_dim=self.dimension)
            elif (
                eff_semantic_mode == "v1_2_rbf"
                and isinstance(self.projector, HybridProjector)
                and getattr(self.projector, "_rbf_landmarks", None) is None
                and has_some_discrete
            ):
                cloned_proj = HybridProjector.from_config(
                    self.projector.to_config(),
                    backend=self.backend,
                )
                cloned_proj.attach_rbf_landmarks(discrete_texts, discrete_labels)
                self.projector = cloned_proj
                self.dimension = int(cloned_proj.dimension)
                self.telemetry_projector = TelemetryProjector(embedding_dim=self.dimension)

        for field_name, f_def in self.schema.fields.items():
            part = partitioned_fields[field_name]
            if part is None:
                # Fall back to default initialized head weights
                head = DecisionFieldHead(f_def, dimension=self.dimension, projector=self.projector)
                compiled_heads[field_name] = CompiledHeadWeights(
                    field_name=field_name,
                    field_type=f_def.field_type,
                    weights=head.weights,
                    biases=head.biases,
                    temperature=1.0,
                    conformal_quantile=0.0,
                    options=getattr(f_def, "options", ()),
                    forgetting_factor=self.forgetting_factor,
                    relative_odds_ratio=self.relative_odds_ratio,
                    confidence_floor_tau0=self.confidence_floor_tau0,
                    escalate_on_ambiguity=getattr(f_def, "escalate_on_ambiguity", True),
                )
                continue

            samples = part["samples"]
            num_train = part["num_train"]
            num_calib = part["num_calib"]
            num_temperature = part["num_temperature"]
            generated_count = part["generated_count"]

            # Prompts and labels
            prompts = [s[0] for s in samples]
            raw_labels = [s[1] for s in samples]

            conf_start_idx = (num_train + num_temperature) if isinstance(f_def, ChoiceField) else num_train
            if conf_start_idx >= len(prompts):
                conf_start_idx = num_train
            need_fam = (
                eff_familiarity_alpha > 0.0
                and (
                    callable(getattr(self.projector, "raw_norm", None))
                    or callable(getattr(self.projector, "project_with_norm", None))
                )
                and conf_start_idx < len(prompts)
            )

            # Encode all prompts (and optional continuous telemetry) to embedding matrix X
            X_list = []
            fam_vals: List[float] = []
            proj_cache: Dict[str, Tuple[np.ndarray, Optional[float]]] = {}
            unique_prompts: List[str] = []
            need_norm_set: set[str] = set()
            seen_prompts: set[str] = set()
            for idx_s, s in enumerate(samples):
                p = s[0]
                if isinstance(p, str):
                    if p not in seen_prompts:
                        seen_prompts.add(p)
                        unique_prompts.append(p)
                    if need_fam and idx_s >= conf_start_idx:
                        need_norm_set.add(p)

            eff_rw = bool(self.recency_weighted) and not isinstance(self.projector, (TfidfProjector, PragmaticCompositeProjector))
            proj_fn = getattr(self.projector, "project", None)
            pwn_fn = getattr(self.projector, "project_with_norm", None)
            batch_fn = getattr(self.projector, "project_batch", None)
            one_rw = _accepts_recency_weighted(proj_fn)
            pwn_rw = _accepts_recency_weighted(pwn_fn)
            batch_rw = _accepts_recency_weighted(batch_fn)
            batch_rn = _accepts_return_norms(batch_fn)
            if (
                unique_prompts
                and (not _is_native_scalar_projector(self.projector))
                and callable(batch_fn)
                and (batch_rw or not (one_rw or pwn_rw))
            ):
                want_any_norm = bool(need_norm_set)
                raw_norms_batch: Optional[np.ndarray] = None
                if want_any_norm and batch_rn:
                    if batch_rw:
                        raw_mat, raw_norms_batch = batch_fn(
                            unique_prompts,
                            recency_weighted=eff_rw,
                            return_norms=True,
                        )
                    else:
                        raw_mat, raw_norms_batch = batch_fn(
                            unique_prompts,
                            return_norms=True,
                        )
                elif batch_rw:
                    raw_mat = batch_fn(
                        unique_prompts,
                        recency_weighted=eff_rw,
                    )
                else:
                    raw_mat = batch_fn(unique_prompts)
                normed_mat = _normalize_raw_projection_batch(
                    raw_mat,
                    len(unique_prompts),
                    self.dimension,
                    zero_unit0=False,
                )
                for u_idx, p in enumerate(unique_prompts):
                    if raw_norms_batch is not None and p in need_norm_set:
                        rn_val_b: Optional[float] = float(raw_norms_batch[u_idx])
                    elif p in need_norm_set:
                        rn_val_b = None
                        if callable(pwn_fn):
                            _, pwn_n_b = pwn_fn(p, recency_weighted=eff_rw) if pwn_rw else pwn_fn(p)
                            if pwn_n_b is not None:
                                rn_val_b = float(pwn_n_b)
                        if rn_val_b is None:
                            rn_val_b = _call_raw_norm_if_callable(self.projector, p, recency_weighted=eff_rw)
                    else:
                        rn_val_b = None
                    proj_cache[p] = (normed_mat[u_idx], rn_val_b)

            for idx_s, s in enumerate(samples):
                p = s[0]
                t = s[2] if len(s) > 2 else None
                want_raw = need_fam and idx_s >= conf_start_idx
                cached = proj_cache.get(p) if isinstance(p, str) else None
                if cached is not None:
                    emb_base, rn_val = cached
                    if want_raw and rn_val is None:
                        if callable(pwn_fn):
                            _, pwn_n_c = pwn_fn(p, recency_weighted=eff_rw) if pwn_rw else pwn_fn(p)
                            rn_val = (
                                float(pwn_n_c)
                                if pwn_n_c is not None
                                else _call_raw_norm_if_callable(self.projector, p, recency_weighted=eff_rw)
                            )
                        else:
                            rn_val = _call_raw_norm_if_callable(self.projector, p, recency_weighted=eff_rw)
                        proj_cache[p] = (emb_base, rn_val)
                    emb = emb_base
                else:
                    if callable(pwn_fn) and t is None and (pwn_rw or not one_rw):
                        raw_emb, pwn_norm = (
                            pwn_fn(p, recency_weighted=eff_rw) if pwn_rw else pwn_fn(p)
                        )
                        rn_val = (
                            float(pwn_norm)
                            if (want_raw and pwn_norm is not None)
                            else (_call_raw_norm_if_callable(self.projector, p, recency_weighted=eff_rw) if want_raw else None)
                        )
                    elif callable(proj_fn):
                        raw_emb = proj_fn(p, recency_weighted=eff_rw) if one_rw else proj_fn(p)
                        if want_raw and callable(pwn_fn) and t is None:
                            _, pwn_norm = pwn_fn(p)
                            rn_val = (
                                float(pwn_norm)
                                if pwn_norm is not None
                                else _call_raw_norm_if_callable(self.projector, p, recency_weighted=eff_rw)
                            )
                        else:
                            rn_val = _call_raw_norm_if_callable(self.projector, p, recency_weighted=eff_rw) if want_raw else None
                    else:
                        raw_emb = self.projector(p)
                        rn_val = _call_raw_norm_if_callable(self.projector, p, recency_weighted=eff_rw) if want_raw else None
                    emb_base = _normalize_raw_projection(raw_emb, self.dimension, zero_unit0=False)
                    if isinstance(p, str):
                        proj_cache[p] = (emb_base, rn_val)
                    emb = emb_base
                if want_raw and rn_val is not None:
                    fam_vals.append(rn_val)
                if t is not None:
                    emb = self.telemetry_projector.fuse(emb, t)
                X_list.append(emb)
            X_all = np.stack(X_list, axis=0).astype(np.float32)

            fam_thresh = 0.0
            if fam_vals:
                pos_fam = [v for v in fam_vals if math.isfinite(v) and v > 0.0]
                if pos_fam:
                    sorted_fam = np.sort(np.asarray(pos_fam, dtype=np.float64))
                    idx_f = max(0, int(math.floor((len(sorted_fam) + 1) * eff_familiarity_alpha)) - 1)
                    idx_f = min(idx_f, len(sorted_fam) - 1)
                    fam_thresh = float(sorted_fam[idx_f])

            N = len(prompts)
            X_train = X_all[:num_train]
            X_calib = X_all[num_train:]
            X_conf = X_calib[num_temperature:] if isinstance(f_def, ChoiceField) else X_calib
            if len(X_conf) == 0:
                X_conf = X_calib

            soft_probs_field = part.get("soft_probs")
            if soft_probs_field is not None and isinstance(f_def, (ChoiceField, BooleanField)):
                y_raw_train = raw_labels[:num_train]
                if isinstance(f_def, ChoiceField):
                    opt_idx_map = {opt: idx for idx, opt in enumerate(f_def.options)}
                    y_idx_train = np.asarray([opt_idx_map[lbl] for lbl in y_raw_train], dtype=np.int64)
                else:
                    y_idx_train = np.asarray([1 if bool(lbl) else 0 for lbl in y_raw_train], dtype=np.int64)

                if num_calib == 0:
                    compiled_heads[field_name] = self._soft_distill_discrete_head(
                        field_name,
                        f_def,
                        prompts[:num_train],
                        soft_probs_field,
                        y_true_idx=y_idx_train,
                        alpha=0.05,
                        X_precomputed=X_train,
                        mondrian=eff_mondrian,
                        fam_thresh=fam_thresh,
                    )
                else:
                    K_s = soft_probs_field.shape[1]
                    Y_hard_s = np.eye(K_s, dtype=np.float64)[y_idx_train]
                    n_f_s, f_ids_s = self._stratified_folds(y_idx_train, K_s)
                    best_b_s, best_r_s, best_nll_s = 0.25, 0.5, float("inf")
                    X_tr64 = X_train.astype(np.float64)
                    for beta_s in (0.15, 0.35, 0.55):
                        for reg_s in (0.15, 0.5, 1.5, 5.0):
                            oof_s = np.zeros((num_train, K_s), dtype=np.float64)
                            for f_k in range(n_f_s):
                                tri_s = np.where(f_ids_s != f_k)[0]
                                vai_s = np.where(f_ids_s == f_k)[0]
                                if len(tri_s) == 0 or len(vai_s) == 0:
                                    continue
                                U_fk, c_fk, _, _ = _solve_soft_ridge_student_raw(
                                    X_tr64[tri_s], soft_probs_field[tri_s], Y_hard_s[tri_s], beta=beta_s, reg=reg_s
                                )
                                oof_s[vai_s] = X_tr64[vai_s] @ U_fk.T + c_fk
                            p_s = _stable_softmax(oof_s, temperature=1.0)
                            nll_s = -float(np.mean(np.sum(soft_probs_field * np.log(np.clip(p_s, 1e-12, 1.0)), axis=1)))
                            if nll_s < best_nll_s:
                                best_nll_s, best_b_s, best_r_s = nll_s, beta_s, reg_s
                    U_s, c_s, Z_s, sw_s = _solve_soft_ridge_student_raw(
                        X_tr64, soft_probs_field, Y_hard_s, beta=best_b_s, reg=best_r_s
                    )
                    if isinstance(f_def, BooleanField):
                        w_s = (0.25 * (U_s[1:2] - U_s[0:1])).astype(np.float32)
                        b_s = np.array([0.25 * float(c_s[1] - c_s[0])], dtype=np.float32)
                        t_scaled = (0.25 * (Z_s[:, 1] - Z_s[:, 0])).reshape(-1, 1).astype(np.float32)
                        h_opts: Tuple[str, ...] = ()
                        raw_cal = (X_calib.astype(np.float64) @ U_s.T + c_s)
                        y_cal_idx = np.asarray([1 if bool(l) else 0 for l in raw_labels[num_train:]], dtype=np.int64)
                    else:
                        w_s = (0.25 * U_s).astype(np.float32)
                        b_s = (0.25 * c_s).astype(np.float32)
                        t_scaled = (0.25 * Z_s).astype(np.float32)
                        h_opts = f_def.options
                        opt_idx_map = {opt: idx for idx, opt in enumerate(f_def.options)}
                        raw_cal = X_calib.astype(np.float64) @ U_s.T + c_s
                        y_cal_idx = np.asarray([opt_idx_map[l] for l in raw_labels[num_train:]], dtype=np.int64)
                    _, _, P_s, B_s = self._solve_ridge(
                        X_train,
                        t_scaled,
                        best_r_s,
                        sample_weights=sw_s.astype(np.float32),
                        regularize_bias=False,
                        return_covariance=True,
                    )
                    n_temp_s = max(0, min(num_temperature, len(raw_cal) - 1))
                    if n_temp_s >= 1:
                        y_t_oh = np.eye(K_s, dtype=np.float64)[y_cal_idx[:n_temp_s]]
                        T_cal = _soft_fit_temperature_np(raw_cal[:n_temp_s], y_t_oh)
                        cal_logits_c = raw_cal[n_temp_s:]
                        cal_y_c = y_cal_idx[n_temp_s:]
                    else:
                        T_cal = 1.0
                        cal_logits_c = raw_cal
                        cal_y_c = y_cal_idx
                    p_cal_c = _stable_softmax(cal_logits_c / T_cal, temperature=1.0)
                    sc_cal = 1.0 - p_cal_c[np.arange(len(cal_y_c)), cal_y_c]
                    sorted_sc_cal = np.sort(sc_cal)
                    n_sc_c = len(sorted_sc_cal)
                    k_c = int(math.ceil((n_sc_c + 1) * 0.95))
                    conf_q_c = (1.0 if k_c > n_sc_c else float(sorted_sc_cal[k_c - 1])) if n_sc_c else 0.0
                    c_opts_lbl = f_def.options if isinstance(f_def, ChoiceField) else ("False", "True")
                    cls_sc_c = (
                        {
                            c_opts_lbl[c]: tuple(float(v) for v in np.sort(sc_cal[cal_y_c == c]))
                            for c in range(K_s)
                            if np.any(cal_y_c == c)
                        }
                        if eff_mondrian
                        else {}
                    )
                    compiled_heads[field_name] = CompiledHeadWeights(
                        field_name=field_name,
                        field_type=f_def.field_type,
                        weights=w_s,
                        biases=b_s,
                        temperature=float(T_cal),
                        conformal_quantile=float(conf_q_c),
                        options=h_opts,
                        P=P_s,
                        B=B_s,
                        forgetting_factor=self.forgetting_factor,
                        relative_odds_ratio=self.relative_odds_ratio,
                        confidence_floor_tau0=self.confidence_floor_tau0,
                        escalate_on_ambiguity=getattr(f_def, "escalate_on_ambiguity", True),
                        calibration_scores=tuple(float(v) for v in sorted_sc_cal),
                        score_method="lac",
                        mondrian=eff_mondrian,
                        class_calibration_scores=cls_sc_c,
                        familiarity_threshold=fam_thresh,
                    )
                continue

            # Construct target Y matrix
            if isinstance(f_def, ChoiceField):
                opt_to_idx = {opt: idx for idx, opt in enumerate(f_def.options)}
                K = len(f_def.options)
                Y_all = np.zeros((N, K), dtype=np.float32)
                for i, lab in enumerate(raw_labels):
                    Y_all[i, opt_to_idx[lab]] = 1.0

                Y_train = Y_all[:num_train]
                Y_calib = Y_all[num_train:]

                if generated_count == 0 and (eff_auto_reg or eff_noise_robust):
                    eff_reg, sw_choice = self._compute_loo_ridge_and_sample_weights(
                        X_train,
                        Y_train,
                        auto_regularization=eff_auto_reg,
                        noise_robust=eff_noise_robust,
                    )
                else:
                    eff_reg, sw_choice = self.regularization, None

                prototypes = self._compute_class_prototypes(
                    X_train, Y_train, is_boolean=False, sample_weights=sw_choice
                )
                proto_thresh, proto_scores = self._calibrate_prototype_scores(
                    X_conf, prototypes, eff_prototype_alpha
                )

                if self.choice_solver == "logistic":
                    weights, biases = self._solve_logistic(
                        X_train, Y_train, eff_reg, sample_weights=sw_choice
                    )
                    P_mat = B_mat = None
                else:
                    weights, biases, P_mat, B_mat = self._solve_ridge(
                        X_train,
                        Y_train,
                        eff_reg,
                        sample_weights=sw_choice,
                        regularize_bias=augment,
                        return_covariance=True,
                    )
                    if eff_proto_blend > 0.0 and K >= 2 and X_train.shape[0] >= K:
                        sw_strict, clean_mask = compute_strict_loo_weights(X_train, Y_train)
                        if sw_choice is not None:
                            sw_strict = np.minimum(sw_choice, sw_strict)
                        use_clean = int(np.sum(clean_mask)) >= K
                        X_c = X_train[clean_mask] if use_clean else X_train
                        Y_c = Y_train[clean_mask] if use_clean else Y_train
                        sw_c = sw_strict[clean_mask] if use_clean else sw_strict
                        cls_c = np.maximum(1.0, np.sum(Y_c, axis=0))
                        Y_bal = (Y_c / cls_c[None, :]) * float(len(X_c) / K)
                        f_reg = (
                            float(eff_fisher_reg)
                            if eff_fisher_reg is not None
                            else max(0.25, float(eff_reg))
                        )
                        w_bal, b_bal = self._solve_ridge(
                            X_c,
                            Y_bal,
                            f_reg,
                            sample_weights=sw_c,
                            regularize_bias=augment,
                            return_covariance=False,
                        )
                        wn = np.linalg.norm(w_bal, axis=1, keepdims=True)
                        wu = w_bal / np.where(wn < 1e-12, 1.0, wn)
                        rn = np.linalg.norm(weights, axis=1, keepdims=True)
                        ru = weights / np.where(rn < 1e-12, 1.0, rn)
                        mean_rn = float(np.mean(rn))
                        w_comb = (1.0 - eff_proto_blend) * ru + eff_proto_blend * wu
                        cn = np.linalg.norm(w_comb, axis=1, keepdims=True)
                        weights = ((w_comb / np.where(cn < 1e-12, 1.0, cn)) * mean_rn).astype(
                            np.float32
                        )
                        biases = (
                            (1.0 - eff_proto_blend) * biases
                            + eff_proto_blend * (b_bal - np.mean(b_bal))
                        ).astype(np.float32)
                weights = _snap_f16(weights)
                assert weights is not None
                P_mat = _snap_f16(P_mat)
                B_mat = _snap_f16(B_mat)

                use_hierarchical = False
                if self.choice_solver == "ridge" and num_train >= K:
                    if eff_hierarchical is True and K >= 3:
                        use_hierarchical = True
                    elif eff_hierarchical == "auto" and K >= 16:
                        use_hierarchical = True

                cluster_assignments: Tuple[int, ...] = ()
                coarse_weights: Optional[np.ndarray] = None
                coarse_biases: Optional[np.ndarray] = None
                fine_weights: Optional[np.ndarray] = None
                fine_biases: Optional[np.ndarray] = None

                if use_hierarchical:
                    (
                        cluster_assignments,
                        coarse_weights,
                        coarse_biases,
                        fine_weights,
                        fine_biases,
                    ) = self._fit_hierarchical_choice_head(
                        X_train,
                        Y_train,
                        prototypes,
                        regularize_bias=augment,
                        regularization=eff_reg,
                        sample_weights=sw_choice,
                    )
                    coarse_weights = _snap_f16(coarse_weights)
                    fine_weights = _snap_f16(fine_weights)
                    raw_logits_calib = self._compute_hierarchical_logits(
                        X_calib,
                        weights,
                        biases,
                        cluster_assignments,
                        coarse_weights,
                        coarse_biases,
                        fine_weights,
                        fine_biases,
                    )
                else:
                    raw_logits_calib = X_calib @ weights.T + biases

                # Calibrate temperature and conformal bounds on independent held-out folds
                logits_calib = raw_logits_calib / 0.25
                calib_labels = np.argmax(Y_calib, axis=1).copy()

                logits_temp, labels_temp = logits_calib[:num_temperature], calib_labels[:num_temperature]
                logits_conf, labels_conf = logits_calib[num_temperature:], calib_labels[num_temperature:]

                calibrator = DecisionCalibrator()
                learned_temp = 1.0
                if len(labels_temp):
                    calibrator.fit(logits_temp, labels_temp)
                    learned_temp = float(calibrator.temperature)
                probs_calib = calibrator.calibrate_logits(logits_conf)

                # Adaptive Prediction Sets (APS) scoring matching calibration.py on held-out conformal fold
                nonconf_scores = []
                error_margins = []
                per_class_scores: Dict[str, List[float]] = {opt: [] for opt in f_def.options}
                for i in range(len(labels_conf)):
                    p_row = probs_calib[i]
                    s_idx = np.argsort(-p_row, kind="mergesort")
                    top_i = int(s_idx[0])
                    runner_i = int(s_idx[1]) if len(s_idx) > 1 else top_i
                    true_c = int(labels_conf[i])
                    if top_i != true_c:
                        error_margins.append(float(p_row[top_i] - p_row[runner_i]))

                    cum_sum = 0.0
                    score_val = 1.0
                    for c_i in s_idx:
                        cum_sum += float(p_row[c_i])
                        if c_i == true_c:
                            score_val = cum_sum
                            break
                    sc = score_val if augment else 1.0 - float(p_row[true_c])
                    nonconf_scores.append(sc)
                    per_class_scores[f_def.options[true_c]].append(sc)

                m_thresh = float(np.clip(np.quantile(error_margins, 0.95) + 0.05, 0.15, 0.85)) if error_margins else 0.20

                sorted_calib_scores = np.sort(np.asarray(nonconf_scores, dtype=np.float64))
                alpha = 0.05
                n_scores = len(sorted_calib_scores)
                k = int(math.ceil((n_scores + 1) * (1.0 - alpha)))
                conformal_q = (1.0 if k > n_scores else float(sorted_calib_scores[k - 1])) if n_scores else 0.0

                class_scores_tuples = (
                    {
                        opt: tuple(float(v) for v in np.sort(np.asarray(vals, dtype=np.float64)))
                        for opt, vals in per_class_scores.items()
                        if vals
                    }
                    if eff_mondrian
                    else {}
                )

                compiled_heads[field_name] = CompiledHeadWeights(
                    field_name=field_name,
                    field_type=f_def.field_type,
                    weights=weights,
                    biases=biases,
                    temperature=learned_temp,
                    score_method="aps" if augment else "lac",
                    conformal_quantile=conformal_q,
                    options=f_def.options,
                    P=P_mat,
                    B=B_mat,
                    margin_threshold=m_thresh,
                    forgetting_factor=self.forgetting_factor,
                    relative_odds_ratio=self.relative_odds_ratio,
                    confidence_floor_tau0=self.confidence_floor_tau0,
                    escalate_on_ambiguity=getattr(f_def, "escalate_on_ambiguity", True),
                    calibration_scores=tuple(float(s) for s in sorted_calib_scores),
                    mondrian=eff_mondrian,
                    class_calibration_scores=class_scores_tuples,
                    familiarity_threshold=fam_thresh,
                    prototype_threshold=proto_thresh,
                    prototype_scores=proto_scores,
                    prototypes=prototypes,
                    hierarchical=use_hierarchical,
                    cluster_assignments=cluster_assignments,
                    coarse_weights=coarse_weights,
                    coarse_biases=coarse_biases,
                    fine_weights=fine_weights,
                    fine_biases=fine_biases,
                )

            elif isinstance(f_def, BooleanField):
                # Logit-space targets: +1.0 for True, -1.0 for False
                Y_all = np.array([1.0 if lab else -1.0 for lab in raw_labels], dtype=np.float32).reshape(-1, 1)
                Y_train = Y_all[:num_train]
                pos_mask = (Y_train.flatten() == 1.0)
                neg_mask = (Y_train.flatten() == -1.0)
                n_pos = int(np.sum(pos_mask))
                n_neg = int(np.sum(neg_mask))
                if generated_count == 0 and (eff_auto_reg or eff_noise_robust):
                    Y_bool_oh = np.column_stack([neg_mask.astype(np.float32), pos_mask.astype(np.float32)])
                    eff_reg_bool, sw_noise = self._compute_loo_ridge_and_sample_weights(
                        X_train,
                        Y_bool_oh,
                        auto_regularization=eff_auto_reg,
                        noise_robust=eff_noise_robust,
                    )
                else:
                    eff_reg_bool, sw_noise = self.regularization, None
                prototypes = self._compute_class_prototypes(
                    X_train, Y_train, is_boolean=True, sample_weights=sw_noise
                )
                proto_thresh, proto_scores = self._calibrate_prototype_scores(
                    X_conf, prototypes, eff_prototype_alpha
                )
                if n_pos > 0 and n_neg > 0:
                    sw = np.zeros(num_train, dtype=np.float32)
                    sw[pos_mask] = (0.5 * num_train) / n_pos
                    sw[neg_mask] = (0.5 * num_train) / n_neg
                    if sw_noise is not None:
                        sw = sw * sw_noise
                else:
                    sw = sw_noise
                weights, biases, P_mat, B_mat = self._solve_ridge(
                    X_train, Y_train, eff_reg_bool, sample_weights=sw, regularize_bias=True, return_covariance=True
                )
                weights = _snap_f16(weights)
                assert weights is not None
                P_mat = _snap_f16(P_mat)
                B_mat = _snap_f16(B_mat)

                # Calibrate conformal bounds on X_calib
                logits_calib = (X_calib @ weights.T + biases).flatten()
                probs_calib = _stable_sigmoid(logits_calib, temperature=0.25)
                calib_bools = [bool(lab) for lab in (raw_labels[num_train:])]
                nonconf_scores = []
                error_margins_bool = []
                per_class_bool: Dict[str, List[float]] = {"False": [], "True": []}
                for i in range(len(calib_bools)):
                    p_true = float(probs_calib[i])
                    p_false = 1.0 - p_true
                    p_row = [p_false, p_true]
                    true_idx = 1 if calib_bools[i] else 0
                    top_b = p_true >= 0.5
                    if top_b != calib_bools[i]:
                        error_margins_bool.append(abs(p_true - p_false))

                    # APS score
                    s_idx = np.argsort(-np.array(p_row), kind="mergesort")
                    cum_s = 0.0
                    score_val = 1.0
                    for c_i in s_idx:
                        cum_s += p_row[c_i]
                        if c_i == true_idx:
                            score_val = cum_s
                            break
                    sc = score_val if augment else 1.0 - float(p_row[true_idx])
                    nonconf_scores.append(sc)
                    per_class_bool["True" if calib_bools[i] else "False"].append(sc)

                m_thresh_bool = float(np.clip(np.quantile(error_margins_bool, 0.95) + 0.05, 0.15, 0.85)) if error_margins_bool else 0.20

                sorted_calib_scores = np.sort(np.asarray(nonconf_scores, dtype=np.float64))
                alpha = 0.05
                n_scores = len(sorted_calib_scores)
                k = int(math.ceil((n_scores + 1) * (1.0 - alpha)))
                conformal_q = (1.0 if k > n_scores else float(sorted_calib_scores[k - 1])) if n_scores else 0.0

                class_scores_bool = (
                    {
                        opt: tuple(float(v) for v in np.sort(np.asarray(vals, dtype=np.float64)))
                        for opt, vals in per_class_bool.items()
                        if vals
                    }
                    if eff_mondrian
                    else {}
                )

                compiled_heads[field_name] = CompiledHeadWeights(
                    field_name=field_name,
                    field_type=f_def.field_type,
                    weights=weights,
                    biases=biases,
                    temperature=1.0,
                    conformal_quantile=conformal_q,
                    options=("False", "True"),
                    P=P_mat,
                    B=B_mat,
                    margin_threshold=m_thresh_bool,
                    score_method="aps" if augment else "lac",
                    forgetting_factor=self.forgetting_factor,
                    relative_odds_ratio=self.relative_odds_ratio,
                    confidence_floor_tau0=self.confidence_floor_tau0,
                    escalate_on_ambiguity=getattr(f_def, "escalate_on_ambiguity", True),
                    calibration_scores=tuple(float(s) for s in sorted_calib_scores),
                    mondrian=eff_mondrian,
                    class_calibration_scores=class_scores_bool,
                    familiarity_threshold=fam_thresh,
                    prototype_threshold=proto_thresh,
                    prototype_scores=proto_scores,
                    prototypes=prototypes,
                )

            elif isinstance(f_def, MultiChoiceField):
                opt_to_idx = {opt: idx for idx, opt in enumerate(f_def.options)}
                K = len(f_def.options)
                N_train = num_train
                weights = np.zeros((K, self.dimension), dtype=np.float32)
                biases = np.zeros(K, dtype=np.float32)
                Y_multi_pos = np.zeros((N_train, K), dtype=np.float32)

                for k, opt in enumerate(f_def.options):
                    y_train_k = np.array([
                        1.0 if (isinstance(raw_labels[i], (list, tuple, set)) and opt in raw_labels[i]) or raw_labels[i] == opt else -1.0
                        for i in range(N_train)
                    ], dtype=np.float32)
                    pos_mask = (y_train_k == 1.0)
                    Y_multi_pos[pos_mask, k] = 1.0
                    neg_mask = (y_train_k == -1.0)
                    n_pos = int(np.sum(pos_mask))
                    n_neg = int(np.sum(neg_mask))
                    sw = np.zeros(N_train, dtype=np.float32)
                    sw[pos_mask] = (0.5 * N_train) / max(1, n_pos)
                    sw[neg_mask] = (0.5 * N_train) / max(1, n_neg)

                    w_k, b_k = self._solve_ridge(
                        X_train, y_train_k, self.regularization, sample_weights=sw, regularize_bias=True
                    )
                    weights[k] = w_k.flatten()
                    biases[k] = float(b_k[0])

                prototypes = self._compute_class_prototypes(X_train, Y_multi_pos, is_boolean=False)
                proto_thresh, proto_scores = self._calibrate_prototype_scores(
                    X_conf, prototypes, eff_prototype_alpha
                )

                X_mc_aug = np.column_stack([X_train, np.ones((N_train, 1), dtype=np.float32)])
                reg_mc = float(self.regularization) * np.eye(self.dimension + 1, dtype=np.float32)
                reg_mc[self.dimension, self.dimension] = max(1.0, float(self.regularization))
                A_mc = X_mc_aug.T @ X_mc_aug + reg_mc
                try:
                    P_mat = np.linalg.inv(A_mc).astype(np.float32)
                except np.linalg.LinAlgError:
                    P_mat = np.linalg.pinv(A_mc).astype(np.float32)
                w_aug = np.hstack([weights, biases.reshape(-1, 1)])
                B_mat = (A_mc @ w_aug.T).astype(np.float32)
                weights = _snap_f16(weights)
                assert weights is not None
                P_mat = _snap_f16(P_mat)
                B_mat = _snap_f16(B_mat)

                # Calibrate conformal bounds on X_calib
                logits_calib = X_calib @ weights.T + biases
                probs_calib = _stable_sigmoid(logits_calib, temperature=0.25)
                calib_labels = raw_labels[num_train:]
                nonconf_scores = []
                for i in range(len(probs_calib)):
                    active_set = set(calib_labels[i] if isinstance(calib_labels[i], (list, tuple, set)) else [calib_labels[i]])
                    scores_i = [
                        float(1.0 - probs_calib[i, k] if opt in active_set else probs_calib[i, k])
                        for k, opt in enumerate(f_def.options)
                    ]
                    nonconf_scores.append(float(np.max(scores_i)) if scores_i else 0.05)
                sorted_calib_scores = np.sort(np.asarray(nonconf_scores, dtype=np.float64))
                alpha = 0.05
                n_scores = len(sorted_calib_scores)
                k = int(math.ceil((n_scores + 1) * (1.0 - alpha)))
                conformal_q = (1.0 if k > n_scores else float(sorted_calib_scores[k - 1])) if n_scores else 0.0

                compiled_heads[field_name] = CompiledHeadWeights(
                    field_name=field_name,
                    field_type=f_def.field_type,
                    weights=weights,
                    biases=biases,
                    temperature=1.0,
                    conformal_quantile=conformal_q,
                    options=f_def.options,
                    P=P_mat,
                    B=B_mat,
                    forgetting_factor=self.forgetting_factor,
                    relative_odds_ratio=self.relative_odds_ratio,
                    confidence_floor_tau0=self.confidence_floor_tau0,
                    escalate_on_ambiguity=getattr(f_def, "escalate_on_ambiguity", True),
                    calibration_scores=tuple(float(s) for s in sorted_calib_scores),
                    familiarity_threshold=fam_thresh,
                    prototype_threshold=proto_thresh,
                    prototype_scores=proto_scores,
                    prototypes=prototypes,
                )

            elif isinstance(f_def, ScoreField):
                raw_floats = np.array([float(lab) for lab in raw_labels], dtype=np.float64).reshape(-1, 1)
                # Map target into logit space matching sigmoid evaluation: prob = sigmoid(z / 0.25) => z = 0.25 * ln(r / (1 - r))
                with np.errstate(over="ignore", invalid="ignore"):
                    r = np.clip(_score_to_unit_ratio(raw_floats, f_def.min_value, f_def.max_value), 0.01, 0.99)
                    Z_all = (0.25 * np.log(r / (1.0 - r))).astype(np.float32)
                Z_train = Z_all[:num_train]
                weights, biases, P_mat, B_mat = self._solve_ridge(
                    X_train, Z_train, self.regularization, regularize_bias=augment, return_covariance=True
                )
                weights = _snap_f16(weights)
                assert weights is not None
                P_mat = _snap_f16(P_mat)
                B_mat = _snap_f16(B_mat)

                # Calibrate residual bounds on X_calib
                logits_calib = (X_calib @ weights.T + biases).flatten()
                probs_calib = _stable_sigmoid(logits_calib, temperature=0.25)
                pred_vals = _score_from_unit_prob(probs_calib, f_def.min_value, f_def.max_value)
                true_calib_floats = raw_floats[num_train:].flatten()
                residuals = [
                    _score_abs_diff_clamped(float(pred_vals[i]), float(true_calib_floats[i]))
                    for i in range(len(pred_vals))
                ]
                sorted_residuals = np.sort(np.asarray(residuals, dtype=np.float64))
                alpha = 0.05
                n_res = len(sorted_residuals)
                k = int(math.ceil((n_res + 1) * (1.0 - alpha)))
                full_range = _score_full_span_clamped(f_def.min_value, f_def.max_value)
                conformal_q = (full_range if k > n_res else float(sorted_residuals[k - 1])) if n_res else 0.0

                compiled_heads[field_name] = CompiledHeadWeights(
                    field_name=field_name,
                    field_type=f_def.field_type,
                    weights=weights,
                    biases=biases,
                    temperature=1.0,
                    conformal_quantile=conformal_q,
                    options=(),
                    P=P_mat,
                    B=B_mat,
                    forgetting_factor=self.forgetting_factor,
                    relative_odds_ratio=self.relative_odds_ratio,
                    confidence_floor_tau0=self.confidence_floor_tau0,
                    escalate_on_ambiguity=getattr(f_def, "escalate_on_ambiguity", True),
                    calibration_scores=tuple(float(s) for s in sorted_residuals),
                    familiarity_threshold=fam_thresh,
                )

        return CompiledSystemOneModel(
            schema=self.schema,
            heads=compiled_heads,
            dimension=self.dimension,
            projector=self.projector,
            backend=self.backend,
            forgetting_factor=self.forgetting_factor,
            recency_weighted=self.recency_weighted,
            metadata={
                "regularization": self.regularization,
                "choice_solver": self.choice_solver,
                "mondrian": eff_mondrian,
                "familiarity_alpha": eff_familiarity_alpha,
                "prototype_alpha": eff_prototype_alpha,
                "hierarchical": eff_hierarchical,
                "semantic_mode": eff_semantic_mode,
                "proto_blend": eff_proto_blend,
                "fisher_reg": eff_fisher_reg,
                "teacher": teacher if augment else None,
                "teaching_mode": "schema_augmented" if augment else "examples",
                "sample_counts": sample_counts,
                "compiled_at": time.time(),
            },
        )

    def compile_and_save(
        self,
        output_path: Union[str, Path],
        exemplars: Optional[Dict[str, Sequence[Tuple[str, Any]]]] = None,
        samples_per_choice: int = 20,
        teacher: str = "synthetic",
        *,
        augment: bool = True,
        calibration_split: float = 0.25,
        calibration_exemplars: Optional[Dict[str, Sequence[Tuple[str, Any]]]] = None,
        mondrian: Optional[bool] = None,
        familiarity_alpha: Optional[float] = None,
        prototype_alpha: Optional[float] = None,
        hierarchical: Optional[Union[bool, str]] = None,
        auto_regularization: Optional[bool] = None,
        noise_robust: Optional[bool] = None,
        label_noise_robust: Optional[bool] = None,
        semantic_mode: Optional[str] = None,
        proto_blend: Optional[float] = None,
        fisher_reg: Optional[float] = None,
        soft_targets: Optional[Mapping[str, Sequence[Sequence[float]]]] = None,
        format_version: int = 2,
        embed_vocab: bool = False,
    ) -> CompiledSystemOneModel:
        """Compiles model and writes directly to target .s1m binary file."""
        model = self.compile(
            exemplars=exemplars,
            samples_per_choice=samples_per_choice,
            teacher=teacher,
            augment=augment,
            calibration_split=calibration_split,
            calibration_exemplars=calibration_exemplars,
            mondrian=mondrian,
            familiarity_alpha=familiarity_alpha,
            prototype_alpha=prototype_alpha,
            hierarchical=hierarchical,
            auto_regularization=auto_regularization,
            noise_robust=noise_robust,
            label_noise_robust=label_noise_robust,
            semantic_mode=semantic_mode,
            proto_blend=proto_blend,
            fisher_reg=fisher_reg,
            soft_targets=soft_targets,
        )
        model.save(
            output_path,
            format_version=format_version,
            embed_vocab=embed_vocab,
        )
        return model

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        projector: Optional[Any] = None,
        backend: str = "numpy",
        *,
        mmap_mode: bool = True,
    ) -> CompiledSystemOneModel:
        """Loads a compiled model from a .s1m file."""
        return CompiledSystemOneModel.load(
            path,
            projector=projector,
            backend=backend,
            mmap_mode=mmap_mode,
        )


__all__ = [
    "SystemOneCompiler",
    "CompiledSystemOneModel",
    "CompiledHeadWeights",
    "MAGIC_HEADER",
    "MAGIC_HEADER_V3",
    "S1M_V3_PAGE_SIZE",
    "MAX_SKILL_FIELDS_V3",
    "VALID_SEMANTIC_MODES",
    "compute_strict_loo_weights",
]
