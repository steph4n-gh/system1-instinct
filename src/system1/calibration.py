"""System 1 Calibration & Split Conformal Prediction Engine.

Implements:
1. Temperature Scaling & Platt Scaling with numerical stability.
2. Proper scoring rules: Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
   Negative Log-Likelihood (NLL), and Sanders-Murphy Brier Score Decomposition
   (Reliability, Resolution, Uncertainty).
3. Split Conformal Prediction with finite-sample, distribution-free mathematical
   coverage guarantees (1 - alpha).
"""

from __future__ import annotations

import bisect
import hashlib
import itertools
import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np

from system1.core.model import (
    _assert_real_numeric_array_like,
    _validate_real_scalar,
)
from system1.core.schema import (
    _f64_to_exact_2m1074,
    _format_score_interval,
    _score_abs_diff_clamped,
    _score_full_span_clamped,
    _score_half_span,
    _score_interval_bounds,
)


def _normalize_rows_f32_safe(arr: Any) -> Tuple[np.ndarray, np.ndarray]:
    """L2-normalize 2D rows in scaled float64 without overflow, returning (unit_f32, valid_mask)."""
    a64 = np.asarray(arr, dtype=np.float64)
    if a64.ndim != 2 or a64.shape[0] == 0 or a64.shape[1] == 0:
        cols = a64.shape[1] if a64.ndim == 2 else 0
        return np.zeros((0, cols), dtype=np.float32), np.zeros((0,), dtype=bool)
    row_finite = np.all(np.isfinite(a64), axis=1)
    safe_a = np.where(row_finite[:, None], a64, 0.0)
    scales = np.max(np.abs(safe_a), axis=1, keepdims=True)
    pos_scale = (scales[:, 0] > 0.0) & row_finite
    denom_scale = np.where(pos_scale[:, None], scales, 1.0)
    scaled = safe_a / denom_scale
    scaled_norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    with np.errstate(over="ignore", invalid="ignore"):
        raw_norms = np.where(scales >= 1.0, np.inf, scales * scaled_norms)
    valid = (
        pos_scale
        & np.isfinite(scaled_norms[:, 0])
        & (scaled_norms[:, 0] > 0.0)
        & ((scales[:, 0] >= 1.0) | (raw_norms[:, 0] > 1e-12))
    )
    denom_norm = np.where(valid[:, None], scaled_norms, 1.0)
    unit_64 = np.where(valid[:, None], scaled / denom_norm, 0.0)
    return unit_64.astype(np.float32), valid


def _normalize_vec_f32_safe(
    vec: Any,
    *,
    min_norm: float = 1e-12,
) -> Tuple[np.ndarray, float, bool]:
    """L2-normalize a 1D vector in scaled float64, returning (unit_f32, norm_f64, is_valid)."""
    v64 = np.asarray(vec, dtype=np.float64).ravel()
    if v64.size == 0 or not np.all(np.isfinite(v64)):
        return np.zeros((v64.size,), dtype=np.float32), float("nan") if v64.size else 0.0, False
    scale = float(np.max(np.abs(v64)))
    if scale == 0.0:
        return np.zeros_like(v64, dtype=np.float32), 0.0, False
    scaled = v64 / scale
    scaled_norm = float(np.linalg.norm(scaled))
    if not math.isfinite(scaled_norm) or scaled_norm <= 0.0:
        return np.zeros_like(v64, dtype=np.float32), float("nan"), False
    with np.errstate(over="ignore"):
        raw_norm = scale * scaled_norm
    if min_norm > 0.0 and scale < 1.0 and raw_norm <= min_norm:
        return np.zeros_like(v64, dtype=np.float32), raw_norm, False
    unit = (scaled / scaled_norm).astype(np.float32)
    return unit, raw_norm, True


def _safe_centroid_mean_f64(rows: Any, weights: Optional[Any] = None) -> Optional[np.ndarray]:
    """Compute overflow-free, permutation-invariant exact centroid mean in 2^-1074 fixed point."""
    r_arr = np.asarray(rows)
    if r_arr.ndim != 2 or r_arr.shape[0] == 0 or r_arr.shape[1] == 0:
        return None
    fin = np.all(np.isfinite(r_arr), axis=1)
    if not bool(np.any(fin)):
        return None
    sub = np.asarray(r_arr[fin], dtype=np.float64)
    w_ints: Optional[List[int]] = None
    den_int: int
    if weights is not None:
        w_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w_arr.shape[0] != r_arr.shape[0]:
            raise ValueError(
                f"Sample weights length {w_arr.shape[0]} != rows count {r_arr.shape[0]}"
            )
        w_sub = w_arr[fin]
        w_ok = np.isfinite(w_sub) & (w_sub > 0.0)
        if not bool(np.any(w_ok)):
            return None
        sub = sub[w_ok]
        w_pos = w_sub[w_ok]
        w_ints = [_f64_to_exact_2m1074(float(w)) for w in w_pos]
        w_sum_int = sum(w_ints)
        if w_sum_int <= 0:
            return None
        den_int = w_sum_int << 1074
    else:
        den_int = sub.shape[0] << 1074
    n_cols = sub.shape[1]
    out = np.zeros(n_cols, dtype=np.float64)
    for j in range(n_cols):
        col = sub[:, j]
        if not bool(np.any(col)):
            out[j] = 0.0
            continue
        if w_ints is None:
            num_int = sum(_f64_to_exact_2m1074(float(v)) for v in col)
        else:
            num_int = sum(wi * _f64_to_exact_2m1074(float(v)) for wi, v in zip(w_ints, col))
        out[j] = num_int / den_int
    return out


def _binom_cdf(k: int, n: int, p: float) -> float:
    """Exact Binomial(n, p) CDF P(X <= k) computed in log space (no scipy required)."""
    if n <= 0 or k >= n or p <= 0.0:
        return 1.0 if k >= 0 else 0.0
    if k < 0 or p >= 1.0:
        return 0.0
    log_p = math.log(p)
    log_q = math.log1p(-p)
    if k <= n // 2:
        cur = n * log_q
        terms = [cur]
        log_odds = log_p - log_q
        for j in range(1, k + 1):
            cur += math.log(n - j + 1) - math.log(j) + log_odds
            terms.append(cur)
        max_lp = max(terms)
        if max_lp < -745.0:
            return 0.0
        return float(min(1.0, math.exp(max_lp) * sum(math.exp(t - max_lp) for t in terms)))
    cur = n * log_p
    terms = [cur]
    log_inv_odds = log_q - log_p
    for j in range(n - 1, k, -1):
        cur += math.log(j + 1) - math.log(n - j) + log_inv_odds
        terms.append(cur)
    max_lp = max(terms)
    if max_lp < -745.0:
        return 1.0
    upper = min(1.0, math.exp(max_lp) * sum(math.exp(t - max_lp) for t in terms))
    return float(max(0.0, 1.0 - upper))


def clopper_pearson_upper(errors: int, n: int, delta: float = 0.05) -> float:
    """Exact one-sided (1 - delta) Clopper-Pearson upper confidence bound."""
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not (0.0 < float(delta) < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    delta_f = float(delta)
    if n <= 0 or errors >= n:
        return 1.0
    if errors <= 0:
        return float(min(1.0, max(0.0, -math.expm1(math.log(delta_f) / float(n)))))
    lo = 0.0
    hi = 1.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if _binom_cdf(errors, n, mid) > delta_f:
            lo = mid
        else:
            hi = mid
    return float(hi)


@dataclass(frozen=True)
class BrierDecomposition:
    """Sanders-Murphy decomposition of the Brier Score.

    Identity: total_brier = reliability - resolution + uncertainty.
    - reliability >= 0: calibration error (lower is better, 0 is perfectly calibrated).
    - resolution >= 0: discriminative sorting ability (higher is better).
    - uncertainty >= 0: inherent variance of base rates (fixed for a given dataset).
    - empirical_brier: direct unbinned sample mean squared error.
    """

    total_brier: float
    reliability: float
    resolution: float
    uncertainty: float
    empirical_brier: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationMetrics:
    """Comprehensive calibration diagnostic metrics."""

    expected_calibration_error: float
    maximum_calibration_error: float
    negative_log_likelihood: float
    brier: BrierDecomposition
    optimal_temperature: float
    bin_accuracies: Tuple[float, ...]
    bin_confidences: Tuple[float, ...]
    bin_counts: Tuple[int, ...]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["bin_accuracies"] = list(self.bin_accuracies)
        d["bin_confidences"] = list(self.bin_confidences)
        d["bin_counts"] = list(self.bin_counts)
        return d


@dataclass(frozen=True)
class ConformalPredictionSet:
    """Result of evaluating a test input through a split conformal predictor."""

    field_name: str
    target_coverage: float  # 1 - alpha (e.g. 0.95)
    alpha: float
    prediction_set: Tuple[str, ...]
    conformal_threshold: float
    is_ambiguous: bool
    is_empty: bool
    p_values: Dict[str, float]
    margin: float = 0.0
    margin_threshold: float = 0.0
    margin_gate_active: bool = False
    raw_is_ambiguous: bool = False
    odds_ratio: float = 1.0
    relative_odds_ratio_threshold: float = 0.0
    confidence_floor: float = 0.0
    needs_escalation: bool = False
    abstention_reason: Optional[str] = None
    prototype_similarity: Optional[float] = None
    prototype_threshold: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "target_coverage": self.target_coverage,
            "alpha": self.alpha,
            "prediction_set": list(self.prediction_set),
            "conformal_threshold": self.conformal_threshold,
            "is_ambiguous": self.is_ambiguous,
            "is_empty": self.is_empty,
            "p_values": self.p_values,
            "margin": self.margin,
            "margin_threshold": self.margin_threshold,
            "margin_gate_active": self.margin_gate_active,
            "raw_is_ambiguous": self.raw_is_ambiguous,
            "needs_escalation": self.needs_escalation,
            "odds_ratio": self.odds_ratio,
            "relative_odds_ratio_threshold": self.relative_odds_ratio_threshold,
            "confidence_floor": self.confidence_floor,
            "abstention_reason": self.abstention_reason,
            "prototype_similarity": self.prototype_similarity,
            "prototype_threshold": self.prototype_threshold,
        }


def _stable_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature."""
    temp = max(1e-4, float(temperature))
    with np.errstate(over="ignore", invalid="ignore"):
        arr = np.asarray(logits, dtype=np.float64)
        scaled = arr / temp
        row_finite = np.all(np.isfinite(arr), axis=-1) & np.all(np.isfinite(scaled), axis=-1)
        safe_scaled = np.where(row_finite[..., None], scaled, 0.0)
        max_val = np.max(safe_scaled, axis=-1, keepdims=True)
        exp = np.exp(safe_scaled - max_val)
        out = exp / np.sum(exp, axis=-1, keepdims=True)
        if not bool(np.all(row_finite)):
            out = np.where(row_finite[..., None], out, np.nan)
        return out


def compute_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Computes Negative Log-Likelihood under categorical distribution."""
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    if n == 0:
        return 0.0
    correct_probs = probs[np.arange(n), labels]
    return float(-np.mean(np.log(correct_probs)))


def compute_ece_and_bins(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float, Tuple[float, ...], Tuple[float, ...], Tuple[int, ...]]:
    """Calculates ECE, MCE, and binned calibration stats across [0, 1]."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    if n == 0:
        return 0.0, 0.0, (), (), ()

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_accs: List[float] = []
    bin_confs: List[float] = []
    bin_counts: List[int] = []

    for b in range(n_bins):
        low, high = bin_boundaries[b], bin_boundaries[b + 1]
        mask = (confidences > low) & (confidences <= high) if b > 0 else (confidences >= low) & (confidences <= high)
        count = int(np.sum(mask))
        bin_counts.append(count)
        if count > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            gap = abs(bin_acc - bin_conf)
            ece += (count / n) * gap
            mce = max(mce, gap)
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)

    return float(ece), float(mce), tuple(bin_accs), tuple(bin_confs), tuple(bin_counts)


def compute_brier_decomposition(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> BrierDecomposition:
    """Computes Sanders-Murphy decomposition of multi-class Brier Score.

    Formula:
      BS = Reliability - Resolution + Uncertainty
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n, k = probs.shape
    if n == 0:
        return BrierDecomposition(0.0, 0.0, 0.0, 0.0)

    # One-hot true outcomes Y
    y_one_hot = np.zeros((n, k), dtype=np.float64)
    y_one_hot[np.arange(n), labels] = 1.0

    # Empirical sample mean squared error
    empirical_brier = float(np.mean(np.sum((probs - y_one_hot) ** 2, axis=1)))

    # Overall base rate \bar{y}
    base_rate = np.mean(y_one_hot, axis=0)
    uncertainty = float(np.sum(base_rate * (1.0 - base_rate)))

    # Binning predictions by max confidence
    confidences = np.max(probs, axis=1)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    reliability = 0.0
    resolution = 0.0

    for b in range(n_bins):
        low, high = bin_boundaries[b], bin_boundaries[b + 1]
        mask = (confidences > low) & (confidences <= high) if b > 0 else (confidences >= low) & (confidences <= high)
        n_m = int(np.sum(mask))
        if n_m > 0:
            weight = n_m / n
            p_m = np.mean(probs[mask], axis=0)
            y_m = np.mean(y_one_hot[mask], axis=0)

            reliability += weight * float(np.sum((p_m - y_m) ** 2))
            resolution += weight * float(np.sum((y_m - base_rate) ** 2))

    rel = float(max(0.0, reliability))
    res = float(max(0.0, resolution))
    unc = float(uncertainty)
    binned_brier = float(max(0.0, rel - res + unc))

    return BrierDecomposition(
        total_brier=binned_brier,
        reliability=rel,
        resolution=res,
        uncertainty=unc,
        empirical_brier=empirical_brier,
    )


class DecisionCalibrator:
    """Calibrator implementing temperature scaling and Platt scaling."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature: float = max(1e-3, float(temperature))
        self.metrics: Optional[CalibrationMetrics] = None
        self.is_calibrated = False
        self.is_binary: bool = False
        self.logit_adjustments: Optional[np.ndarray] = None
        self.selective_calibrator: Optional["SelectiveRiskCalibrator"] = None
        self.margin_threshold: Optional[float] = None

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        *,
        n_bins: int = 10,
        temp_bounds: Tuple[float, float] = (0.15, 10.0),
    ) -> CalibrationMetrics:
        """Finds optimal temperature T > 0 minimizing NLL on calibration logits."""
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        self.is_calibrated = False
        self.is_binary = logits.ndim == 1

        if len(logits) == 0:
            self.temperature = 1.0
            self.metrics = CalibrationMetrics(
                expected_calibration_error=1.0,
                maximum_calibration_error=1.0,
                negative_log_likelihood=float('inf'),
                brier=BrierDecomposition(1.0, 1.0, 0.0, 0.0, 1.0),
                optimal_temperature=1.0,
                bin_accuracies=tuple([0.0]*n_bins),
                bin_confidences=tuple([0.0]*n_bins),
                bin_counts=tuple([0]*n_bins),
            )
            return self.metrics

        if logits.ndim == 1:
            self.is_binary = True
            logits = np.column_stack([-logits / 2.0, logits / 2.0])

        a, b = temp_bounds
        inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
        inv_phi2 = (3.0 - math.sqrt(5.0)) / 2.0

        c = a + inv_phi2 * (b - a)
        d = a + inv_phi * (b - a)

        def eval_temp(t: float) -> float:
            scaled = logits / t
            max_val = np.max(scaled, axis=-1, keepdims=True)
            exp = np.exp(scaled - max_val)
            probs = exp / np.sum(exp, axis=-1, keepdims=True)
            return compute_nll(probs, labels)

        fc = eval_temp(c)
        fd = eval_temp(d)

        for _ in range(60):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = a + inv_phi2 * (b - a)
                fc = eval_temp(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + inv_phi * (b - a)
                fd = eval_temp(d)

        best_temp = float((a + b) / 2.0)
        self.temperature = best_temp
        self.is_calibrated = True

        calibrated_probs = self.calibrate_logits(logits)
        nll = compute_nll(calibrated_probs, labels)
        ece, mce, bin_accs, bin_confs, bin_counts = compute_ece_and_bins(
            calibrated_probs, labels, n_bins=n_bins
        )
        brier = compute_brier_decomposition(calibrated_probs, labels, n_bins=n_bins)

        self.metrics = CalibrationMetrics(
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            negative_log_likelihood=nll,
            brier=brier,
            optimal_temperature=best_temp,
            bin_accuracies=bin_accs,
            bin_confidences=bin_confs,
            bin_counts=bin_counts,
        )
        return self.metrics

    def calibrate_logits(self, logits: np.ndarray) -> np.ndarray:
        """Applies learned logit adjustment and temperature scaling to raw logits."""
        logits_chk = _assert_real_numeric_array_like(logits, "logits")
        arr = np.asarray(logits_chk, dtype=np.float64)
        temp = _validate_real_scalar(self.temperature, "DecisionCalibrator temperature")
        if not math.isfinite(temp) or temp <= 0.0:
            raise ValueError(f"DecisionCalibrator temperature must be positive and finite, got {temp!r}")
        raw_adj = self.logit_adjustments
        if raw_adj is None:
            adj = None
        else:
            adj_chk = _assert_real_numeric_array_like(raw_adj, "DecisionCalibrator logit_adjustments")
            adj = np.asarray(adj_chk, dtype=np.float64)
            exp_adj_len = 2 if (arr.ndim == 0 or self.is_binary or (arr.ndim == 1 and len(arr) == 1)) else arr.shape[-1]
            if adj.ndim != 1 or adj.shape != (exp_adj_len,) or not bool(np.all(np.isfinite(adj))):
                raise ValueError(
                    f"DecisionCalibrator logit_adjustments must be finite with shape ({exp_adj_len},), got {adj.shape}"
                )

        def _single_binary_logit(z: float) -> np.ndarray:
            if not math.isfinite(z):
                return np.array([np.nan, np.nan], dtype=np.float64)
            two = np.array([-0.5 * z, 0.5 * z], dtype=np.float64)
            if adj is not None and adj.shape == (2,):
                two = two + adj
            scaled_two = two / temp
            if not bool(np.all(np.isfinite(scaled_two))):
                return np.array([np.nan, np.nan], dtype=np.float64)
            return _stable_softmax(scaled_two, temperature=1.0)

        with np.errstate(over="ignore", invalid="ignore"):
            if arr.ndim == 0:
                return _single_binary_logit(float(arr))
            if self.is_binary:
                if arr.ndim == 1:
                    finite_1d = np.isfinite(arr)
                    two = np.column_stack([-arr / 2.0, arr / 2.0])
                    if adj is not None and adj.shape == (2,):
                        two = two + adj
                    scaled_two = two / temp
                    out = _stable_softmax(scaled_two, temperature=1.0)
                    row_ok = finite_1d & np.all(np.isfinite(scaled_two), axis=-1)
                    if not bool(np.all(row_ok)):
                        out = np.where(row_ok[:, None], out, np.nan)
                    return out

            if arr.ndim == 1:
                if len(arr) == 1:
                    return _single_binary_logit(float(arr[0]))
                if not bool(np.all(np.isfinite(arr))):
                    return np.full(arr.shape, np.nan, dtype=np.float64)
                if adj is not None and adj.shape == (len(arr),):
                    arr = arr + adj
                scaled = arr / temp
                if not bool(np.all(np.isfinite(scaled))):
                    return np.full(arr.shape, np.nan, dtype=np.float64)
                s = scaled - float(np.max(scaled))
                exp = np.exp(s)
                return exp / float(np.sum(exp))

            row_finite = np.all(np.isfinite(arr), axis=-1)
            if adj is not None and adj.shape == (arr.shape[-1],):
                arr = arr + adj
            scaled = arr / temp
            row_finite = row_finite & np.all(np.isfinite(scaled), axis=-1)
            safe_scaled = np.where(row_finite[..., None], scaled, 0.0)
            max_val = np.max(safe_scaled, axis=-1, keepdims=True)
            exp = np.exp(safe_scaled - max_val)
            out = exp / np.sum(exp, axis=-1, keepdims=True)
            if not bool(np.all(row_finite)):
                out = np.where(row_finite[..., None], out, np.nan)
            return out

    def fit_imbalance_aware(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        *,
        class_priors: Optional[Sequence[float]] = None,
        tau: float = 0.5,
        temp_bounds: Tuple[float, float] = (0.5, 5.0),
        l2_reg: float = 0.02,
        alpha: float = 0.05,
        delta: float = 0.05,
        seed: int = 42,
    ) -> "DecisionCalibrator":
        """Fit class-prior logit adjustment, balanced temperature scaling, and fail-closed LTT margin gate."""
        arr_raw = np.asarray(logits, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        self.is_calibrated = False
        self.selective_calibrator = None
        self.logit_adjustments = None
        self.margin_threshold = 1.01

        if arr_raw.size == 0 or len(y) == 0:
            self.is_binary = arr_raw.ndim == 1
            self.temperature = 1.0
            return self

        if arr_raw.ndim == 1:
            self.is_binary = True
            arr_2d = np.column_stack([-arr_raw / 2.0, arr_raw / 2.0])
        elif arr_raw.ndim == 2:
            self.is_binary = False
            arr_2d = arr_raw
        else:
            raise ValueError(f"logits must be 1D or 2D, got {arr_raw.ndim}D")

        n_samples, k_classes = arr_2d.shape
        if len(y) != n_samples:
            raise ValueError(f"logits rows ({n_samples}) != labels len ({len(y)})")
        if k_classes < 2:
            raise ValueError(f"Expected at least 2 classes, got {k_classes}")
        if np.any((y < 0) | (y >= k_classes)):
            raise ValueError(f"labels must be in [0, {k_classes - 1}]")
        if not math.isfinite(float(tau)) or float(tau) < 0.0:
            raise ValueError(f"tau must be finite and >= 0, got {tau}")
        if not math.isfinite(float(l2_reg)) or float(l2_reg) < 0.0:
            raise ValueError(f"l2_reg must be finite and >= 0, got {l2_reg}")
        t_lo, t_hi = float(temp_bounds[0]), float(temp_bounds[1])
        if not (0.0 < t_lo <= t_hi) or not math.isfinite(t_lo) or not math.isfinite(t_hi):
            raise ValueError(f"temp_bounds must satisfy 0 < lo <= hi, got {temp_bounds!r}")

        if class_priors is not None:
            pi = np.asarray(list(class_priors), dtype=np.float64)
            if pi.ndim != 1 or len(pi) != k_classes or np.any(~np.isfinite(pi)) or np.any(pi < 0.0):
                raise ValueError(f"class_priors must be non-negative finite vector of length {k_classes}")
            pi_sum = float(np.sum(pi))
            if pi_sum <= 0.0:
                raise ValueError("class_priors must sum to > 0")
            pi = np.maximum(pi / pi_sum, 1e-12)
        else:
            counts_all = np.bincount(y, minlength=k_classes)
            pi = np.maximum(1, counts_all).astype(np.float64) / float(max(1, n_samples))

        self.logit_adjustments = (-float(tau) * np.log(np.clip(pi, 1e-12, 1.0))).astype(np.float64)

        rng = np.random.RandomState(int(seed))
        t_parts: List[np.ndarray] = []
        g_parts: List[np.ndarray] = []
        for c in range(k_classes):
            c_idx = np.where(y == c)[0]
            if len(c_idx) == 0:
                continue
            perm = c_idx[rng.permutation(len(c_idx))]
            half = len(perm) // 2
            if len(perm) == 1:
                t_parts.append(perm)
            else:
                t_parts.append(perm[:half])
                g_parts.append(perm[half:])

        t_idx = np.concatenate(t_parts) if t_parts else np.arange(0, dtype=np.int64)
        g_idx = np.concatenate(g_parts) if g_parts else np.arange(0, dtype=np.int64)

        adj_2d = arr_2d + self.logit_adjustments
        if len(t_idx) >= 2:
            y_t = y[t_idx]
            counts_t = np.bincount(y_t, minlength=k_classes)
            class_w = np.where(counts_t > 0, len(y_t) / (float(k_classes) * np.maximum(1, counts_t)), 1.0)
            sw = class_w[y_t]
            sw_mean = float(np.mean(sw))
            if sw_mean > 0.0:
                sw = sw / sw_mean
            adj_t = adj_2d[t_idx]
            ts = np.geomspace(t_lo, t_hi, 80)
            best_t = 1.0
            best_loss = float("inf")
            row_ar = np.arange(len(y_t))
            for cand_t in ts:
                probs_t = np.clip(_stable_softmax(adj_t, temperature=float(cand_t)), 1e-15, 1.0 - 1e-15)
                nll = float(-np.mean(sw * np.log(probs_t[row_ar, y_t])))
                loss = nll + float(l2_reg) * (math.log(float(cand_t)) ** 2)
                if loss < best_loss:
                    best_loss = loss
                    best_t = float(cand_t)
            self.temperature = best_t
        else:
            self.temperature = 1.0
        self.is_calibrated = True

        cal_ltt = SelectiveRiskCalibrator(
            score_name="margin",
            method="ltt_fst",
            alpha=float(alpha),
            delta=float(delta),
            grid_size=25,
            max_frac=0.95,
        )
        self.selective_calibrator = cal_ltt
        if len(g_idx) >= 1:
            adj_g = adj_2d[g_idx]
            p_g = _stable_softmax(adj_g, temperature=self.temperature)
            part_g = np.partition(p_g, -2, axis=1)
            m_g = part_g[:, -1] - part_g[:, -2]
            corr_g = np.argmax(p_g, axis=1) == y[g_idx]
            cert = cal_ltt.fit(m_g, corr_g)
            if cert.qualified and cert.threshold is not None:
                self.margin_threshold = float(cert.threshold)
            else:
                self.margin_threshold = 1.01
        else:
            self.margin_threshold = 1.01
        return self

    def should_accept_probs(self, probs: np.ndarray) -> Union[bool, np.ndarray]:
        """Return whether top-1 minus runner-up probability margin meets calibrated gate threshold."""
        p = np.asarray(probs, dtype=np.float64)
        thresh = self.margin_threshold if self.margin_threshold is not None else 0.0
        if p.ndim == 1:
            if len(p) < 2:
                return bool(1.0 >= thresh)
            part = np.partition(p, -2)
            return bool(float(part[-1] - part[-2]) >= thresh)
        if p.ndim == 2:
            if p.shape[1] < 2:
                return np.ones(p.shape[0], dtype=bool) & (1.0 >= thresh)
            part = np.partition(p, -2, axis=1)
            margins = part[:, -1] - part[:, -2]
            return margins >= thresh
        raise ValueError(f"probs must be 1D or 2D, got {p.ndim}D")


class ConformalPredictor:
    """Split conformal sets using LAC (1 - p(label)) or legacy APS scores.

    Strict sets have marginal coverage under exchangeable calibration/test
    examples and an independently fitted scoring model. They do not guarantee
    correctness conditional on acceptance. Non-strict margin heuristics may
    suppress review and do not preserve the strict coverage interpretation.
    """

    def __init__(
        self,
        field_name: Union[str, Sequence[str]] = "default",
        options: Sequence[str] = (),
        margin_threshold: Optional[float] = None,
        *,
        relative_odds_ratio: Optional[float] = 1.5,
        confidence_floor_tau0: Optional[float] = 0.15,
        score_method: str = "aps",
        mondrian: bool = False,
        mondrian_shrinkage: float = 10.0,
        familiarity_threshold: float = 0.0,
        prototype_threshold: float = 0.0,
        prototypes: Optional[np.ndarray] = None,
    ) -> None:
        if score_method not in ("aps", "lac"):
            raise ValueError(f"Unknown conformal score method: {score_method!r}")
        if not isinstance(field_name, str) and not options:
            options = field_name
            field_name = "default"
        self.score_method = score_method
        self.field_name = str(field_name)
        self.options = tuple(options)
        self.option_to_idx = {opt: idx for idx, opt in enumerate(self.options)}
        self.calibration_scores: np.ndarray = np.array([], dtype=np.float64)
        self.class_calibration_scores: Dict[str, np.ndarray] = {}
        self.mondrian: bool = bool(mondrian)
        self.mondrian_shrinkage: float = float(mondrian_shrinkage)
        self.familiarity_threshold: float = float(familiarity_threshold)
        self.familiarity_scores: np.ndarray = np.array([], dtype=np.float64)
        self.prototype_threshold: float = float(prototype_threshold)
        self.prototype_scores: np.ndarray = np.array([], dtype=np.float64)
        if prototypes is not None:
            p_raw = np.asarray(prototypes)
            if p_raw.ndim == 2 and np.all(np.isfinite(np.asarray(p_raw, dtype=np.float64))):
                self.prototypes = _normalize_rows_f32_safe(p_raw)[0]
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    self.prototypes = np.asarray(p_raw, dtype=np.float32)
        else:
            self.prototypes = None
        self.is_calibrated: bool = False
        self.quantile: Optional[float] = None
        self.margin_threshold: Optional[float] = (
            float(margin_threshold) if margin_threshold is not None else None
        )
        self.calibrated_margin_threshold: float = 0.0
        self.relative_odds_ratio: float = (
            float(relative_odds_ratio) if relative_odds_ratio is not None else 1.5
        )
        self.confidence_floor_tau0: float = (
            float(confidence_floor_tau0) if confidence_floor_tau0 is not None else 0.15
        )

    def compute_prototype_similarity(self, embedding: np.ndarray) -> Optional[float]:
        """Compute maximum cosine similarity between an input embedding and class prototypes."""
        if self.prototypes is None or self.prototypes.size == 0:
            return None
        emb_flat = np.asarray(embedding).ravel()
        if emb_flat.ndim != 1 or self.prototypes.ndim != 2 or self.prototypes.shape[1] != emb_flat.shape[0]:
            return None
        emb_unit, _, emb_valid = _normalize_vec_f32_safe(emb_flat)
        if not emb_valid:
            return 0.0
        proto_all, valid_rows = _normalize_rows_f32_safe(self.prototypes)
        if not np.any(valid_rows):
            return None
        proto_unit = proto_all[valid_rows]
        sims = proto_unit @ emb_unit
        return float(np.max(sims))

    def calibrate_familiarity(
        self,
        familiarity_values: Sequence[float],
        *,
        alpha_ood: float = 0.01,
    ) -> float:
        """Fit a held-out lower conformal threshold on non-negative familiarity scores."""
        if not (0.0 <= alpha_ood < 1.0) or not math.isfinite(alpha_ood):
            raise ValueError(f"alpha_ood must be in [0, 1), got {alpha_ood}")
        vals = np.asarray(familiarity_values, dtype=np.float64).flatten()
        vals = vals[np.isfinite(vals) & (vals > 0.0)]
        if len(vals) == 0 or alpha_ood == 0.0:
            self.familiarity_scores = np.array([], dtype=np.float64)
            self.familiarity_threshold = 0.0
            return 0.0
        self.familiarity_scores = np.sort(vals)
        m = len(self.familiarity_scores)
        rank = int(math.floor((m + 1) * alpha_ood))
        self.familiarity_threshold = float(self.familiarity_scores[rank - 1]) if rank >= 1 else 0.0
        return self.familiarity_threshold

    def calibrate_prototypes(
        self,
        prototype_values: Union[Sequence[float], np.ndarray],
        labels: Optional[Sequence[Any]] = None,
        *,
        alpha_prototype: float = 0.01,
        alpha: Optional[float] = None,
        prototypes: Optional[np.ndarray] = None,
    ) -> float:
        """Fit a held-out lower conformal threshold on max class-prototype cosine similarities."""
        eff_alpha = float(alpha) if alpha is not None else float(alpha_prototype)
        if not (0.0 <= eff_alpha < 1.0) or not math.isfinite(eff_alpha):
            raise ValueError(f"alpha_prototype must be in [0, 1), got {eff_alpha}")
        if prototypes is not None:
            p_raw = np.asarray(prototypes)
            if p_raw.ndim == 2 and np.all(np.isfinite(np.asarray(p_raw, dtype=np.float64))):
                self.prototypes = _normalize_rows_f32_safe(p_raw)[0]
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    self.prototypes = np.asarray(p_raw, dtype=np.float32)
        arr = np.asarray(prototype_values, dtype=np.float64)
        if arr.ndim == 2:
            if self.prototypes is None and labels is not None and len(labels) == arr.shape[0] and len(self.options) > 0:
                protos = np.zeros((len(self.options), arr.shape[1]), dtype=np.float32)
                for c_idx, opt in enumerate(self.options):
                    mask = [i for i, lbl in enumerate(labels) if str(lbl) == str(opt)]
                    if mask:
                        c_mean = _safe_centroid_mean_f64(arr[mask])
                        if c_mean is not None:
                            u_vec, _, ok = _normalize_vec_f32_safe(c_mean, min_norm=0.0)
                            if ok:
                                protos[c_idx] = u_vec
                self.prototypes = protos
            if self.prototypes is not None and self.prototypes.size > 0 and len(arr) > 0:
                sims = [self.compute_prototype_similarity(row) for row in arr]
                vals = np.asarray([s for s in sims if s is not None], dtype=np.float64)
            else:
                vals = np.array([], dtype=np.float64)
        else:
            vals = arr.flatten()
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0 or eff_alpha == 0.0:
            self.prototype_scores = np.array([], dtype=np.float64)
            self.prototype_threshold = 0.0
            return 0.0
        self.prototype_scores = np.sort(vals)
        m = len(self.prototype_scores)
        rank = int(math.floor((m + 1) * eff_alpha))
        if rank >= 1:
            self.prototype_threshold = float(max(0.0, float(self.prototype_scores[rank - 1])))
        else:
            # For small calibration cohorts where (m + 1) * alpha < 1, use a conservative
            # margin below the minimum observed in-distribution prototype similarity.
            min_sim = float(self.prototype_scores[0])
            self.prototype_threshold = float(max(0.0, min_sim * 0.90)) if min_sim > 0.0 else 0.0
        return self.prototype_threshold

    def class_quantile(self, option: str, alpha: float = 0.05) -> float:
        """Compute global or Mondrian (exact when n_y >= ceil(1/alpha)-1, empirical-Bayes blended when small) quantile."""
        n = len(self.calibration_scores)
        if not self.is_calibrated or n == 0:
            return 1.0
        k_global = int(math.ceil((n + 1) * (1.0 - alpha)))
        q_global = 1.0 if k_global > n else float(self.calibration_scores[k_global - 1])
        if not self.mondrian:
            return q_global
        c_scores = self.class_calibration_scores.get(option)
        if c_scores is None or len(c_scores) == 0:
            return q_global
        n_c = len(c_scores)
        k_c = int(math.ceil((n_c + 1) * (1.0 - alpha)))
        if k_c <= n_c:
            return float(c_scores[k_c - 1])
        if q_global >= 1.0:
            return 1.0
        if n_c == 1:
            q_emp = float(c_scores[0])
        else:
            pos = (n_c - 1) * (1.0 - alpha)
            lo = int(math.floor(pos))
            hi = min(n_c - 1, lo + 1)
            frac = pos - lo
            c_lo = float(c_scores[lo])
            q_emp = c_lo + frac * (float(c_scores[hi]) - c_lo)
        shrink = max(1.0, self.mondrian_shrinkage)
        w = float(n_c) / (float(n_c) + shrink)
        # Floor at q_global when k_c > n_c so tiny per-class cohorts never shrink below marginal coverage.
        return float(max(q_global, w * q_emp + (1.0 - w) * q_global))

    def calibrate(
        self,
        calibrated_probs: np.ndarray,
        ground_truth_labels: Sequence[str],
        margin_threshold: Optional[float] = None,
        *,
        relative_odds_ratio: Optional[float] = None,
        confidence_floor_tau0: Optional[float] = None,
        mondrian: Optional[bool] = None,
        familiarity_values: Optional[Sequence[float]] = None,
        alpha_ood: float = 0.0,
        familiarity_alpha: float = 0.0,
        prototype_values: Optional[Union[Sequence[float], np.ndarray]] = None,
        embeddings: Optional[np.ndarray] = None,
        alpha_prototype: float = 0.0,
        prototype_alpha: float = 0.0,
        prototypes: Optional[np.ndarray] = None,
    ) -> None:
        """Fits the configured nonconformity score and optional margin/familiarity/prototype thresholds."""
        probs = np.asarray(calibrated_probs, dtype=np.float64)
        if len(probs) != len(ground_truth_labels):
            raise ValueError("Mismatched probabilities and labels length")
        if len(probs) == 0:
            raise ValueError("Cannot calibrate conformal predictor with 0 samples")

        if relative_odds_ratio is not None:
            self.relative_odds_ratio = float(relative_odds_ratio)
        if confidence_floor_tau0 is not None:
            self.confidence_floor_tau0 = float(confidence_floor_tau0)
        if mondrian is not None:
            self.mondrian = bool(mondrian)
        if prototypes is not None:
            self.prototypes = np.asarray(prototypes, dtype=np.float32)

        scores: List[float] = []
        per_class_scores: Dict[str, List[float]] = {opt: [] for opt in self.options}
        error_margins: List[float] = []

        for i, label in enumerate(ground_truth_labels):
            if label not in self.option_to_idx:
                raise ValueError(f"Unknown label {label!r} not in options {self.options}")
            true_idx = self.option_to_idx[label]
            row = probs[i]

            sorted_indices = np.argsort(-row, kind="mergesort")
            top_idx = int(sorted_indices[0])
            runner_up_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else top_idx
            margin_i = float(row[top_idx] - row[runner_up_idx])

            if top_idx != true_idx:
                # Top candidate was an error: record its margin
                error_margins.append(margin_i)

            cum_sum = 0.0
            score_val = 1.0
            for idx in sorted_indices:
                cum_sum += float(row[idx])
                if idx == true_idx:
                    score_val = cum_sum
                    break
            s_i = 1.0 - float(row[true_idx]) if self.score_method == "lac" else score_val
            scores.append(s_i)
            per_class_scores[label].append(s_i)

        self.calibration_scores = np.sort(np.asarray(scores, dtype=np.float64))
        self.class_calibration_scores = {
            opt: np.sort(np.asarray(vals, dtype=np.float64))
            for opt, vals in per_class_scores.items()
            if vals
        }
        self.is_calibrated = True

        eff_alpha_ood = float(familiarity_alpha) if familiarity_alpha > 0.0 else float(alpha_ood)
        eff_alpha_proto = float(prototype_alpha) if prototype_alpha > 0.0 else float(alpha_prototype)
        eff_proto_vals = prototype_values if prototype_values is not None else embeddings
        if familiarity_values is not None and eff_alpha_ood > 0.0:
            self.calibrate_familiarity(familiarity_values, alpha_ood=eff_alpha_ood)
        if eff_proto_vals is not None and eff_alpha_proto > 0.0:
            self.calibrate_prototypes(
                eff_proto_vals,
                labels=ground_truth_labels,
                alpha_prototype=eff_alpha_proto,
                prototypes=prototypes,
            )

        if margin_threshold is not None:
            self.margin_threshold = float(margin_threshold)
            self.calibrated_margin_threshold = self.margin_threshold
        elif self.margin_threshold is not None:
            self.calibrated_margin_threshold = self.margin_threshold
        else:
            # Calibrate threshold above observed error margins
            if error_margins:
                # Heuristic margin threshold above observed error margins
                self.calibrated_margin_threshold = float(
                    np.clip(np.quantile(error_margins, 0.95) + 0.05, 0.15, 0.85)
                )
            else:
                self.calibrated_margin_threshold = 0.35

    def predict_set(
        self,
        calibrated_probs: np.ndarray,
        *,
        alpha: float = 0.05,
        margin_threshold: Optional[float] = None,
        relative_odds_ratio: Optional[float] = None,
        confidence_floor_tau0: Optional[float] = None,
        strict: bool = False,
        familiarity_score: Optional[float] = None,
        prototype_similarity: Optional[float] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> ConformalPredictionSet:
        """Invert the configured score to construct a prediction set.

        Applies Cardinality-Scaled Margin Gate & Relative Odds Ratio Dominance:
        When raw_is_ambiguous (set size > 1), false-positive ambiguity escalation is suppressed
        ONLY IF strict=False AND:
          1. Absolute margin difference: margin >= eff_margin_thresh
          2. Absolute confidence floor scaled by cardinality: top_prob >= (1/K) + eff_tau0
          3. Relative odds ratio dominance: (top_prob / max(1e-6, runner_up_prob)) >= eff_gamma
        These heuristics do not guarantee precision or preserve strict conformal
        coverage; evaluate their behavior on the intended workload.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Significance level alpha must be in (0, 1), got {alpha}")

        probs = np.asarray(calibrated_probs, dtype=np.float64).flatten()
        if len(probs) != len(self.options):
            raise ValueError(
                f"Probabilities length ({len(probs)}) does not match options ({len(self.options)})"
            )

        if prototype_similarity is None and embedding is not None and self.prototypes is not None:
            prototype_similarity = self.compute_prototype_similarity(embedding)

        n = len(self.calibration_scores)
        sorted_indices = np.argsort(-probs, kind="mergesort")
        top_prob = float(probs[sorted_indices[0]]) if len(sorted_indices) > 0 else 0.0
        runner_up_prob = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
        margin = float(top_prob - runner_up_prob) if len(sorted_indices) > 1 else 1.0

        low_familiarity = (
            familiarity_score is not None
            and self.familiarity_threshold > 0.0
            and (not math.isfinite(familiarity_score) or familiarity_score < self.familiarity_threshold - 1e-12)
        )
        low_prototype_similarity = (
            prototype_similarity is not None
            and self.prototype_threshold > 0.0
            and (not math.isfinite(prototype_similarity) or prototype_similarity < self.prototype_threshold - 1e-6)
        )
        oos_abstain = low_familiarity or low_prototype_similarity
        abstention_reason: Optional[str] = None
        if low_familiarity:
            abstention_reason = "low_feature_familiarity"
        elif low_prototype_similarity:
            abstention_reason = "low_semantic_familiarity"

        if not self.is_calibrated or n == 0:
            if oos_abstain:
                q_hat = 1.0 if strict else (1.0 - alpha)
                prediction_set: List[str] = []
            elif strict:
                q_hat = 1.0
                prediction_set = list(self.options)
            else:
                q_hat = 1.0 - alpha
                prediction_set = []
                cum_mass = 0.0
                total_mass = float(np.sum(probs))
                if total_mass >= q_hat - 1e-7 and total_mass >= 0.5:
                    for idx in sorted_indices:
                        opt = self.options[idx]
                        p = float(probs[idx])
                        prediction_set.append(opt)
                        cum_mass += p
                        if cum_mass >= q_hat - 1e-7:
                            break
        else:
            k = int(math.ceil((n + 1) * (1.0 - alpha)))
            if k > n:
                q_hat = 1.0
            else:
                q_hat = float(self.calibration_scores[k - 1])

            prediction_set = []
            cum_mass = 0.0
            total_mass = float(np.sum(probs))
            if oos_abstain:
                prediction_set = []
            # Invalid/degenerate probabilities abstain; this is not a calibrated OOD test.
            elif self.score_method == "lac":
                if k > n:
                    prediction_set = list(self.options)
                elif np.all(np.isfinite(probs)) and np.all(probs >= 0) and abs(total_mass - 1.0) <= (1e-8 + 1e-5):
                    if self.mondrian and self.class_calibration_scores:
                        prediction_set = [
                            self.options[idx]
                            for idx in sorted_indices
                            if 1.0 - float(probs[idx]) <= self.class_quantile(self.options[idx], alpha) + 1e-7
                        ]
                    else:
                        prediction_set = [self.options[idx] for idx in sorted_indices
                                          if 1.0 - float(probs[idx]) <= q_hat + 1e-7]
            elif strict:
                # Invert the calibrated score s(x,y) = cumulative mass through y.
                # Including the next label after crossing q_hat inflates every set.
                # Empty sets request review; insufficient evidence includes all labels.
                if k > n:
                    prediction_set = list(self.options)
                elif np.all(np.isfinite(probs)) and np.all(probs >= 0) and abs(total_mass - 1.0) <= (1e-8 + 1e-5):
                    for idx in sorted_indices:
                        cum_mass += float(probs[idx])
                        opt = self.options[idx]
                        thresh = self.class_quantile(opt, alpha) if (self.mondrian and self.class_calibration_scores) else q_hat
                        if cum_mass <= thresh + 1e-7:
                            prediction_set.append(opt)
            elif total_mass >= q_hat - 1e-7 and total_mass >= 0.5:
                for idx in sorted_indices:
                    opt = self.options[idx]
                    p = float(probs[idx])
                    prediction_set.append(opt)
                    cum_mass += p
                    if cum_mass >= q_hat - 1e-7:
                        break

        if n > 0:
            if self.score_method == "lac":
                candidate_scores = 1.0 - probs
            else:
                candidate_scores = np.empty_like(probs)
                candidate_scores[sorted_indices] = np.cumsum(probs[sorted_indices])
            # Calibration scores are sorted when fitted/loaded. Left insertion
            # includes equal scores, preserving the original >= tail count.
            ranks = n - np.searchsorted(self.calibration_scores, candidate_scores, side="left")
            p_values = {opt: float((1 + int(rank)) / (n + 1))
                        for opt, rank in zip(self.options, ranks)}
        else:
            p_values = {opt: float(p) for opt, p in zip(self.options, probs)}

        # Cardinality-Scaled Margin Gate & Relative Odds Ratio
        eff_margin_thresh = (
            float(margin_threshold)
            if margin_threshold is not None
            else (self.margin_threshold if self.margin_threshold is not None else self.calibrated_margin_threshold)
        )
        eff_gamma = (
            float(relative_odds_ratio)
            if relative_odds_ratio is not None
            else self.relative_odds_ratio
        )
        eff_tau0 = (
            float(confidence_floor_tau0)
            if confidence_floor_tau0 is not None
            else self.confidence_floor_tau0
        )

        raw_is_ambiguous = len(prediction_set) > 1
        is_empty = len(prediction_set) == 0

        # Cardinality-scaled confidence floor: p_(1) >= 1/K + tau_0
        k_card = max(1, len(self.options))
        conf_floor = float((1.0 / float(k_card)) + eff_tau0)

        # Relative odds ratio: p_(1) / max(1e-6, p_(2))
        odds_ratio = float(top_prob / max(1e-6, runner_up_prob))

        # Margin-Based Conformal Dominance Gating (suppressed in strict mode)
        margin_gate_active = bool(
            not strict
            and raw_is_ambiguous
            and (eff_margin_thresh > 0.0)
            and (margin >= eff_margin_thresh)
            and (top_prob >= conf_floor)
            and (odds_ratio >= eff_gamma)
        )
        if oos_abstain or (strict and (not self.is_calibrated or n == 0)):
            is_ambiguous = True
            needs_escalation = True
        else:
            is_ambiguous = raw_is_ambiguous and not margin_gate_active
            needs_escalation = is_empty or is_ambiguous

        return ConformalPredictionSet(
            field_name=self.field_name,
            target_coverage=1.0 - alpha,
            alpha=alpha,
            prediction_set=tuple(prediction_set),
            conformal_threshold=q_hat,
            is_ambiguous=is_ambiguous,
            is_empty=is_empty,
            p_values=p_values,
            margin=margin,
            margin_threshold=eff_margin_thresh,
            margin_gate_active=margin_gate_active,
            raw_is_ambiguous=raw_is_ambiguous,
            odds_ratio=odds_ratio,
            relative_odds_ratio_threshold=eff_gamma,
            confidence_floor=conf_floor,
            needs_escalation=needs_escalation,
            abstention_reason=abstention_reason,
            prototype_similarity=prototype_similarity,
            prototype_threshold=self.prototype_threshold,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "options": list(self.options),
            "margin_threshold": self.margin_threshold,
            "calibrated_margin_threshold": self.calibrated_margin_threshold,
            "relative_odds_ratio": self.relative_odds_ratio,
            "confidence_floor_tau0": self.confidence_floor_tau0,
            "score_method": self.score_method,
            "mondrian": self.mondrian,
            "mondrian_shrinkage": self.mondrian_shrinkage,
            "familiarity_threshold": self.familiarity_threshold,
            "familiarity_scores": [float(s) for s in self.familiarity_scores],
            "prototype_threshold": self.prototype_threshold,
            "prototype_scores": [float(s) for s in self.prototype_scores],
            "calibration_scores": [float(s) for s in self.calibration_scores],
            "class_calibration_scores": {
                k: [float(v) for v in vals] for k, vals in self.class_calibration_scores.items()
            },
            "is_calibrated": self.is_calibrated,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ConformalPredictor:
        pred = cls(
            field_name=str(data["field_name"]),
            options=list(data["options"]),
            margin_threshold=data.get("margin_threshold"),
            relative_odds_ratio=float(data.get("relative_odds_ratio", 1.5)),
            confidence_floor_tau0=float(data.get("confidence_floor_tau0", 0.15)),
            score_method=str(data.get("score_method", "lac")),
            mondrian=bool(data.get("mondrian", False)),
            mondrian_shrinkage=float(data.get("mondrian_shrinkage", 10.0)),
            familiarity_threshold=float(data.get("familiarity_threshold", 0.0)),
            prototype_threshold=float(data.get("prototype_threshold", 0.0)),
        )
        pred.calibrated_margin_threshold = float(data.get("calibrated_margin_threshold", 0.20))
        fam_scores = data.get("familiarity_scores", [])
        if fam_scores:
            pred.familiarity_scores = np.sort(np.asarray(fam_scores, dtype=np.float64))
        proto_scores = data.get("prototype_scores", [])
        if proto_scores:
            pred.prototype_scores = np.sort(np.asarray(proto_scores, dtype=np.float64))
        scores = data.get("calibration_scores", [])
        if scores:
            pred.calibration_scores = np.sort(np.asarray(scores, dtype=np.float64))
            pred.is_calibrated = bool(data.get("is_calibrated", True))
        class_scores = data.get("class_calibration_scores", {})
        if isinstance(class_scores, Mapping):
            pred.class_calibration_scores = {
                str(k): np.sort(np.asarray(v, dtype=np.float64))
                for k, v in class_scores.items()
                if v
            }
        return pred


@dataclass(frozen=True)
class RegressionConformalInterval:
    """Conformal prediction interval for continuous score fields with 1 - alpha coverage."""

    field_name: str
    target_coverage: float
    alpha: float
    point_prediction: float
    lower_bound: float
    upper_bound: float
    margin: float
    is_calibrated: bool

    def to_interval_string(self) -> str:
        return _format_score_interval(self.lower_bound, self.upper_bound)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RegressionConformalPredictor:
    """Split conformal predictor for continuous bounded regression scores."""

    def __init__(self, field_name: str, min_value: float = 0.0, max_value: float = 1.0) -> None:
        self.field_name = field_name
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.residuals: np.ndarray = np.array([], dtype=np.float64)
        self.is_calibrated: bool = False

    def calibrate(self, predictions: Sequence[float], ground_truth: Sequence[float]) -> None:
        preds = np.asarray(predictions, dtype=np.float64)
        targets = np.asarray(ground_truth, dtype=np.float64)
        if len(preds) != len(targets):
            raise ValueError("Predictions and targets length mismatch")
        if len(preds) == 0:
            raise ValueError("Cannot calibrate regression conformal predictor with 0 samples")
        if not (bool(np.all(np.isfinite(preds))) and bool(np.all(np.isfinite(targets)))):
            raise ValueError("Predictions and targets must be finite")
        res = np.array(
            [_score_abs_diff_clamped(float(p_v), float(t_v)) for p_v, t_v in zip(preds.flat, targets.flat)],
            dtype=np.float64,
        )
        self.residuals = np.sort(res)
        self.is_calibrated = True

    def predict_interval(
        self,
        point_prediction: float,
        *,
        alpha: float = 0.05,
        strict: bool = False,
    ) -> RegressionConformalInterval:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Significance level alpha must be in (0, 1), got {alpha}")
        y_hat = float(point_prediction)
        val_range = _score_full_span_clamped(self.min_value, self.max_value)
        half_span = _score_half_span(self.min_value, self.max_value)
        n = len(self.residuals)

        if not self.is_calibrated or n == 0:
            if strict:
                margin = val_range
                low = self.min_value
                high = self.max_value
            else:
                margin = (1.0 - alpha) * half_span
                low, high = _score_interval_bounds(y_hat, margin, self.min_value, self.max_value)
        else:
            k = int(math.ceil((n + 1) * (1.0 - alpha)))
            if k > n:
                margin = val_range
                low = self.min_value
                high = self.max_value
            else:
                margin = float(self.residuals[k - 1])
                low, high = _score_interval_bounds(y_hat, margin, self.min_value, self.max_value)
        return RegressionConformalInterval(
            field_name=self.field_name,
            target_coverage=1.0 - alpha,
            alpha=alpha,
            point_prediction=y_hat,
            lower_bound=low,
            upper_bound=high,
            margin=margin,
            is_calibrated=self.is_calibrated,
        )


@dataclass(frozen=True)
class RecalibrationMetrics:
    """Per-field post-hoc temperature scaling and conformal quantile recalibration summary."""

    field_name: str
    ece_before: float
    ece_after: float
    nll_before: float
    nll_after: float
    temperature_before: float
    temperature: float
    qhat_before: float
    qhat: float
    alpha: float
    n_examples: int
    brier_before: float = 0.0
    brier_after: float = 0.0
    familiarity_threshold: float = 0.0
    prototype_threshold: float = 0.0
    calibration_scores: Tuple[float, ...] = ()
    class_calibration_scores: Dict[str, Tuple[float, ...]] = field(default_factory=dict)

    @property
    def optimal_temperature(self) -> float:
        return self.temperature

    @property
    def temperature_after(self) -> float:
        return self.temperature

    @property
    def previous_temperature(self) -> float:
        return self.temperature_before

    @property
    def updated_temperature(self) -> float:
        return self.temperature

    @property
    def calibrated_temperature(self) -> float:
        return self.temperature

    @property
    def conformal_quantile(self) -> float:
        return self.qhat

    @property
    def qhat_after(self) -> float:
        return self.qhat

    @property
    def previous_conformal_quantile(self) -> float:
        return self.qhat_before

    @property
    def updated_conformal_quantile(self) -> float:
        return self.qhat

    @property
    def calibrated_quantile(self) -> float:
        return self.qhat

    @property
    def calibration_sample_count(self) -> int:
        return self.n_examples

    @property
    def initial_nll(self) -> float:
        return self.nll_before

    @property
    def calibrated_nll(self) -> float:
        return self.nll_after

    @property
    def initial_ece(self) -> float:
        return self.ece_before

    @property
    def calibrated_ece(self) -> float:
        return self.ece_after

    @property
    def initial_brier(self) -> float:
        return self.brier_before

    @property
    def calibrated_brier(self) -> float:
        return self.brier_after

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
            "nll_before": self.nll_before,
            "nll_after": self.nll_after,
            "temperature_before": self.temperature_before,
            "temperature": self.temperature,
            "qhat_before": self.qhat_before,
            "qhat": self.qhat,
            "alpha": self.alpha,
            "n_examples": self.n_examples,
            "brier_before": self.brier_before,
            "brier_after": self.brier_after,
            "familiarity_threshold": self.familiarity_threshold,
            "prototype_threshold": self.prototype_threshold,
            "calibration_scores": list(self.calibration_scores),
            "class_calibration_scores": {k: list(v) for k, v in self.class_calibration_scores.items()},
        }

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key) and not key.startswith("_") and not callable(getattr(self, key)):
            return getattr(self, key)
        raise KeyError(key)

    def keys(self) -> Any:
        return self.to_dict().keys()

    def items(self) -> Any:
        return self.to_dict().items()

    def values(self) -> Any:
        return self.to_dict().values()

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and not key.startswith("_") and hasattr(self, key) and not callable(getattr(self, key))


@dataclass(frozen=True)
class RecalibrationReport:
    """Aggregate and per-field post-hoc recalibration diagnostics."""

    ece_before: float
    ece_after: float
    nll_before: float
    nll_after: float
    temperature_before: float
    temperature: float
    qhat_before: float
    qhat: float
    alpha: float
    n_examples: int
    per_field: Dict[str, RecalibrationMetrics] = field(default_factory=dict)
    schema_name: str = ""
    brier_before: float = 0.0
    brier_after: float = 0.0

    @property
    def per_field_metrics(self) -> Dict[str, RecalibrationMetrics]:
        return self.per_field

    @property
    def field_metrics(self) -> Dict[str, RecalibrationMetrics]:
        return self.per_field

    @property
    def fields(self) -> Dict[str, RecalibrationMetrics]:
        return self.per_field

    @property
    def total_examples(self) -> int:
        return self.n_examples

    @property
    def calibration_size(self) -> int:
        return self.n_examples

    @property
    def initial_ece(self) -> float:
        return self.ece_before

    @property
    def calibrated_ece(self) -> float:
        return self.ece_after

    @property
    def initial_nll(self) -> float:
        return self.nll_before

    @property
    def calibrated_nll(self) -> float:
        return self.nll_after

    @property
    def initial_brier(self) -> float:
        return self.brier_before

    @property
    def calibrated_brier(self) -> float:
        return self.brier_after

    @property
    def optimal_temperature(self) -> float:
        return self.temperature

    @property
    def temperature_after(self) -> float:
        return self.temperature

    @property
    def conformal_quantile(self) -> float:
        return self.qhat

    @property
    def qhat_after(self) -> float:
        return self.qhat

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
            "nll_before": self.nll_before,
            "nll_after": self.nll_after,
            "brier_before": self.brier_before,
            "brier_after": self.brier_after,
            "temperature_before": self.temperature_before,
            "temperature": self.temperature,
            "qhat_before": self.qhat_before,
            "qhat": self.qhat,
            "alpha": self.alpha,
            "n_examples": self.n_examples,
            "per_field": {k: v.to_dict() for k, v in self.per_field.items()},
        }

    def __getitem__(self, key: str) -> Any:
        if key in self.per_field:
            return self.per_field[key]
        if hasattr(self, key) and not key.startswith("_") and not callable(getattr(self, key)):
            return getattr(self, key)
        raise KeyError(key)

    def keys(self) -> Any:
        return self.to_dict().keys()

    def items(self) -> Any:
        return self.to_dict().items()

    def values(self) -> Any:
        return self.to_dict().values()

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and (
            key in self.per_field
            or (not key.startswith("_") and hasattr(self, key) and not callable(getattr(self, key)))
        )


@dataclass(frozen=True)
class SelectiveRiskCertificate:
    """Finite-sample LTT/RCPS selective acceptance certificate."""

    threshold: Optional[float]
    method: str
    risk_type: str
    alpha: float
    delta: float
    qualified: bool
    n_qual: int
    n_accepted_qual: int
    errors_qual: int
    upper_bound: Optional[float]
    score_name: str = "margin"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SelectiveRiskCalibrator:
    """Finite-sample selective-risk threshold calibrator (LTT-FST, LTT-Bonferroni, RCPS-Joint).

    Controls either conditional selective risk Pr(err | g(X) >= lambda) <= alpha
    via Learn Then Test (Angelopoulos et al.) or joint risk Pr(err and g(X) >= lambda) <= alpha
    via Bates et al. RCPS with exact Clopper-Pearson bounds. Fail-closed until qualified.
    """

    VALID_METHODS = ("ltt_fst", "ltt_bonferroni", "rcps_joint")

    def __init__(
        self,
        *,
        score_name: str = "margin",
        method: str = "ltt_fst",
        alpha: float = 0.05,
        delta: float = 0.05,
        grid_size: int = 15,
        max_frac: float = 0.55,
    ) -> None:
        if method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}, got {method!r}")
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not (0.0 < float(delta) < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if int(grid_size) < 1:
            raise ValueError(f"grid_size must be >= 1, got {grid_size}")
        if not (0.0 < float(max_frac) <= 1.0):
            raise ValueError(f"max_frac must be in (0, 1], got {max_frac}")
        self.score_name = str(score_name)
        self.method = str(method)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.grid_size = int(grid_size)
        self.max_frac = float(max_frac)
        self.threshold: Optional[float] = None
        self.certificate: Optional[SelectiveRiskCertificate] = None

    def is_qualified(self) -> bool:
        return (
            self.certificate is not None
            and self.certificate.qualified
            and self.threshold is not None
            and math.isfinite(self.threshold)
        )

    def should_accept(self, score: float) -> bool:
        """Fail-closed selective gate: returns True only when qualified and score >= threshold."""
        if not self.is_qualified():
            return False
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            return False
        assert self.threshold is not None
        return float(score) >= float(self.threshold)

    def fit(
        self,
        scores: Sequence[float],
        correct: Sequence[bool],
        *,
        alpha: Optional[float] = None,
        delta: Optional[float] = None,
        method: Optional[str] = None,
    ) -> SelectiveRiskCertificate:
        use_method = str(method or self.method)
        if use_method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}, got {use_method!r}")
        use_alpha = float(self.alpha if alpha is None else alpha)
        use_delta = float(self.delta if delta is None else delta)
        if not (0.0 < use_alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {use_alpha}")
        if not (0.0 < use_delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {use_delta}")

        s_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
        c_arr = np.asarray(correct, dtype=bool).reshape(-1)
        if len(s_arr) != len(c_arr):
            raise ValueError(f"scores and correct must have the same length ({len(s_arr)} vs {len(c_arr)})")
        if len(s_arr) > 0 and not np.all(np.isfinite(s_arr)):
            raise ValueError("scores must contain only finite values")

        n = int(len(s_arr))
        risk_type = "joint" if use_method == "rcps_joint" else "conditional"
        best_lam: Optional[float] = None
        best_ub: Optional[float] = None
        best_m = 0
        best_errs = 0

        if n > 0:
            s_desc = np.sort(s_arr)[::-1]
            if use_method == "rcps_joint":
                lo_m = max(1, min(10, n) if n < 100 else max(10, int(0.01 * n)))
                hi_m = max(lo_m, int(0.80 * n)) if n >= 2 else n
                count = min(self.grid_size, hi_m - lo_m + 1)
                idxs = np.unique(np.round(np.geomspace(lo_m, hi_m, count)).astype(int) - 1)
                thresholds = sorted({float(s_desc[i]) for i in idxs})
                m_eff = max(1, len(thresholds))
                for lam in reversed(thresholds):
                    mask = s_arr >= lam
                    errs = int(np.sum(~c_arr[mask]))
                    ub = clopper_pearson_upper(errs, n, delta=use_delta / m_eff)
                    if ub <= use_alpha:
                        best_lam = lam
                        best_ub = ub
                        best_m = int(np.sum(mask))
                        best_errs = errs
                    else:
                        break
            elif use_method == "ltt_bonferroni":
                min_m = max(1, math.ceil(math.log(use_delta / self.grid_size) / math.log(1.0 - use_alpha)))
                if n >= min_m:
                    max_m = max(min_m, int(round(self.max_frac * n)))
                    hi_m = min(n, max_m)
                    count = min(self.grid_size, hi_m - min_m + 1)
                    idxs = np.unique(np.round(np.geomspace(min_m, hi_m, count)).astype(int) - 1)
                    thresholds = sorted({float(s_desc[i]) for i in idxs})
                    m_eff = max(1, len(thresholds))
                    for lam in thresholds:
                        mask = s_arr >= lam
                        m = int(np.sum(mask))
                        errs = int(np.sum(~c_arr[mask]))
                        pval = _binom_cdf(errs, m, use_alpha)
                        if pval <= use_delta / m_eff:
                            best_lam = lam
                            best_ub = clopper_pearson_upper(errs, m, delta=use_delta / m_eff)
                            best_m = m
                            best_errs = errs
                            break
            else:  # ltt_fst
                min_m = max(1, math.ceil(math.log(use_delta) / math.log(1.0 - use_alpha)))
                if n >= min_m:
                    max_m = max(min_m, int(round(self.max_frac * n)))
                    hi_m = min(n, max_m)
                    count = min(self.grid_size, hi_m - min_m + 1)
                    idxs = np.unique(np.round(np.geomspace(min_m, hi_m, count)).astype(int) - 1)
                    thresholds = sorted({float(s_desc[i]) for i in idxs}, reverse=True)
                    for lam in thresholds:
                        mask = s_arr >= lam
                        m = int(np.sum(mask))
                        errs = int(np.sum(~c_arr[mask]))
                        pval = _binom_cdf(errs, m, use_alpha)
                        if pval <= use_delta:
                            best_lam = lam
                            best_ub = clopper_pearson_upper(errs, m, delta=use_delta)
                            best_m = m
                            best_errs = errs
                        else:
                            break

        cert = SelectiveRiskCertificate(
            threshold=best_lam,
            method=use_method,
            risk_type=risk_type,
            alpha=use_alpha,
            delta=use_delta,
            qualified=best_lam is not None,
            n_qual=n,
            n_accepted_qual=best_m,
            errors_qual=best_errs,
            upper_bound=best_ub,
            score_name=self.score_name,
        )
        self.threshold = best_lam
        self.certificate = cert
        return cert

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score_name": self.score_name,
            "method": self.method,
            "alpha": self.alpha,
            "delta": self.delta,
            "grid_size": self.grid_size,
            "max_frac": self.max_frac,
            "threshold": self.threshold,
            "certificate": self.certificate.to_dict() if self.certificate is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SelectiveRiskCalibrator":
        obj = cls(
            score_name=str(data.get("score_name", "margin")),
            method=str(data.get("method", "ltt_fst")),
            alpha=float(data.get("alpha", 0.05)),
            delta=float(data.get("delta", 0.05)),
            grid_size=int(data.get("grid_size", 15)),
            max_frac=float(data.get("max_frac", 0.55)),
        )
        cert_raw = data.get("certificate")
        if isinstance(cert_raw, Mapping):
            obj.certificate = SelectiveRiskCertificate(**dict(cert_raw))
            obj.threshold = obj.certificate.threshold
        else:
            raw_t = data.get("threshold")
            obj.threshold = float(raw_t) if raw_t is not None else None
        return obj


class AdaptiveConformalTracker:
    """Online Adaptive Conformal Inference (ACI) tracker with bounded sliding window and margin ratchet."""

    def __init__(
        self,
        *,
        alpha_target: float = 0.05,
        gamma: float = 0.02,
        alpha_min: float = 0.01,
        alpha_max: Optional[float] = None,
        margin_floor: Optional[float] = None,
        max_window: int = 512,
        escalation_learning_rate_scale: float = 0.25,
    ) -> None:
        if not (0.0 < float(alpha_target) < 1.0):
            raise ValueError(f"alpha_target must be in (0, 1), got {alpha_target}")
        if float(gamma) < 0.0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if int(max_window) < 1:
            raise ValueError(f"max_window must be >= 1, got {max_window}")
        eff_max = float(alpha_target) if alpha_max is None else float(alpha_max)
        eff_min = float(alpha_min)
        if not (0.0 < eff_min <= eff_max < 1.0):
            raise ValueError(f"Require 0 < alpha_min <= alpha_max < 1, got [{eff_min}, {eff_max}]")
        self.alpha_target = float(alpha_target)
        self.gamma = float(gamma)
        self.alpha_min = eff_min
        self.alpha_max = eff_max
        self.alpha_t = float(alpha_target)
        self.margin_floor: Optional[float] = float(margin_floor) if margin_floor is not None else None
        self.max_window = int(max_window)
        self.escalation_learning_rate_scale = float(escalation_learning_rate_scale)
        self.steps = 0
        self.accepted_steps = 0
        self.accepted_errors = 0
        self.escalation_steps = 0
        self._scores: deque[float] = deque(maxlen=max(1, self.max_window))
        self._sorted_scores: List[float] = []

    def should_accept_margin(self, margin: float) -> bool:
        if self.margin_floor is None:
            return True
        if not math.isfinite(float(margin)):
            return False
        return float(margin) >= float(self.margin_floor)

    def update_accepted(self, error: bool, *, margin: Optional[float] = None) -> float:
        """Update online ACI alpha_t after observing a locally accepted outcome."""
        err_f = 1.0 if error else 0.0
        self.steps += 1
        self.accepted_steps += 1
        self.accepted_errors += int(err_f)
        self.alpha_t = float(
            np.clip(
                self.alpha_t + self.gamma * (self.alpha_target - err_f),
                self.alpha_min,
                self.alpha_max,
            )
        )
        if error and margin is not None and math.isfinite(float(margin)):
            floor = float(margin) + 1e-4
            self.margin_floor = floor if self.margin_floor is None else max(self.margin_floor, floor)
        return self.alpha_t

    def record_escalation(self, *, predicted_wrong: bool = False, margin: Optional[float] = None) -> None:
        self.steps += 1
        self.escalation_steps += 1
        if (
            predicted_wrong
            and margin is not None
            and math.isfinite(float(margin))
            and self.margin_floor is not None
            and float(margin) >= self.margin_floor
        ):
            self.margin_floor = max(self.margin_floor, float(margin) + 1e-4)

    def push_calibration_score(self, score: float) -> np.ndarray:
        """Append a nonconformity score into the bounded sliding window and return sorted scores."""
        val = float(score)
        if self.max_window <= 0:
            return np.array([], dtype=np.float64)
        if len(self._scores) >= self.max_window:
            oldest = self._scores[0]
            rm_idx = bisect.bisect_left(self._sorted_scores, oldest)
            if rm_idx < len(self._sorted_scores) and self._sorted_scores[rm_idx] == oldest:
                self._sorted_scores.pop(rm_idx)
            else:
                self._scores.popleft()
                self._sorted_scores = sorted(self._scores)
        self._scores.append(val)
        bisect.insort(self._sorted_scores, val)
        return np.asarray(self._sorted_scores, dtype=np.float64)

    def rescore_calibration_window(
        self,
        pairs: Sequence[Tuple[str, Any]],
        predictor: Callable[[str, Any], Tuple[float, float, bool]],
        *,
        default_floor: float = 0.30,
    ) -> Tuple[np.ndarray, float]:
        """Re-score a bounded calibration window after online weight updates and ratchet margin_floor."""
        window = list(pairs)[-self.max_window :] if self.max_window > 0 else []
        scores: List[float] = []
        err_margins: List[float] = []
        for prompt_c, label_c in window:
            nc_score, margin_c, is_correct = predictor(prompt_c, label_c)
            scores.append(float(nc_score))
            if not is_correct and math.isfinite(float(margin_c)):
                err_margins.append(float(margin_c))
        self._scores = deque(scores, maxlen=max(1, self.max_window))
        self._sorted_scores = sorted(self._scores)
        new_floor = (max(err_margins) + 1e-4) if err_margins else float(default_floor)
        if self.margin_floor is not None:
            self.margin_floor = max(self.margin_floor, new_floor)
        else:
            self.margin_floor = new_floor
        return np.asarray(self._sorted_scores, dtype=np.float64), float(self.margin_floor)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "alpha_target": self.alpha_target,
            "alpha_t": self.alpha_t,
            "gamma": self.gamma,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "margin_floor": self.margin_floor,
            "max_window": self.max_window,
            "window_len": len(self._scores),
            "steps": self.steps,
            "accepted_steps": self.accepted_steps,
            "accepted_errors": self.accepted_errors,
            "escalation_steps": self.escalation_steps,
        }


def _is_bool_scalar(val: Any) -> bool:
    return isinstance(val, (bool, np.bool_)) or getattr(getattr(val, "dtype", None), "kind", None) == "b"


def _validate_real_1d_vec(vec: Any, expected_len: int, label: str) -> np.ndarray:
    if isinstance(vec, (str, bytes, bytearray, Mapping)):
        raise ValueError(f"{label} must be a 1D numeric sequence, not {type(vec).__name__}")
    if isinstance(vec, np.ndarray):
        if vec.dtype == np.bool_ or vec.dtype.kind not in ("i", "u", "f"):
            raise ValueError(f"{label} must have real numeric dtype, got {vec.dtype}")
        if vec.ndim != 1 or vec.shape[0] != expected_len:
            raise ValueError(f"{label} must have shape ({expected_len},), got {vec.shape}")
        arr = np.asarray(vec, dtype=np.float64)
    else:
        try:
            items = list(vec)
        except TypeError as exc:
            raise ValueError(f"{label} must be a 1D numeric sequence, got {type(vec).__name__}") from exc
        if len(items) != expected_len:
            raise ValueError(f"{label} must have length {expected_len}, got {len(items)}")
        for v in items:
            if _is_bool_scalar(v) or not isinstance(v, (int, float, np.integer, np.floating)):
                raise ValueError(f"{label} entries must be real numbers, got {v!r}")
        arr = np.asarray(items, dtype=np.float64)
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{label} must contain only finite values")
    return arr


class OnlineMondrianConformal:
    """Strictly O(1) per-class online adaptive quantile tracker in log-nonconformity space.

    Uses ACI-style adaptive tracking toward long-run target coverage under
    distribution shift rather than a static distribution-free finite-sample guarantee.
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.10,
        q_init: Optional[Sequence[float]] = None,
        gamma_aci: Optional[float] = None,
        eta_u: float = 0.30,
        jump_w: float = 0.12,
    ) -> None:
        if _is_bool_scalar(num_classes) or not isinstance(num_classes, (int, np.integer)):
            raise ValueError(f"num_classes must be an integer >= 1, got {num_classes!r}")
        nc = int(num_classes)
        if nc < 1:
            raise ValueError("num_classes must be >= 1")
        if _is_bool_scalar(alpha) or not isinstance(alpha, (int, float, np.integer, np.floating)):
            raise ValueError("alpha must be finite and in (0, 1)")
        a = float(alpha)
        if not math.isfinite(a) or not (0.0 < a < 1.0):
            raise ValueError("alpha must be finite and in (0, 1)")
        if gamma_aci is not None:
            if _is_bool_scalar(gamma_aci) or not isinstance(gamma_aci, (int, float, np.integer, np.floating)):
                raise ValueError(f"gamma_aci must be finite and positive, got {gamma_aci!r}")
            g_aci = float(gamma_aci)
            if not math.isfinite(g_aci) or g_aci <= 0.0:
                raise ValueError(f"gamma_aci must be finite and positive, got {gamma_aci!r}")
        else:
            g_aci = max(0.03, 0.4 * a)
        if _is_bool_scalar(eta_u) or not isinstance(eta_u, (int, float, np.integer, np.floating)):
            raise ValueError(f"eta_u must be finite and positive, got {eta_u!r}")
        eta_val = float(eta_u)
        if not math.isfinite(eta_val) or eta_val <= 0.0:
            raise ValueError(f"eta_u must be finite and positive, got {eta_u!r}")
        if _is_bool_scalar(jump_w) or not isinstance(jump_w, (int, float, np.integer, np.floating)):
            raise ValueError(f"jump_w must be finite and in [0, 1], got {jump_w!r}")
        jump_val = float(jump_w)
        if not math.isfinite(jump_val) or not (0.0 <= jump_val <= 1.0):
            raise ValueError(f"jump_w must be finite and in [0, 1], got {jump_w!r}")
        self.num_classes = nc
        self.alpha_target = a
        self.gamma_aci = g_aci
        self.eta_u = eta_val
        self.jump_w = jump_val
        if q_init is not None:
            q0 = _validate_real_1d_vec(q_init, self.num_classes, "q_init")
        else:
            q0 = np.full(self.num_classes, 0.5, dtype=np.float64)
        self.u_max = 16.0
        self.q_c = np.clip(q0.copy(), 0.0, 1.0)
        self.u_c = np.where(
            self.q_c >= 1.0 - math.exp(-(self.u_max - 1e-6)),
            self.u_max,
            np.maximum(0.0, -np.log(np.clip(1.0 - self.q_c, 1e-12, 1.0))),
        )
        self.alpha_c = np.full(self.num_classes, a, dtype=np.float64)
        self.err_ewma = np.full(self.num_classes, a, dtype=np.float64)
        self.alpha_min = max(0.015, 0.35 * a)
        self.alpha_max = min(0.40, 2.5 * a)
        self.steps = 0

    def thresholds(self) -> np.ndarray:
        """Return per-class nonconformity score thresholds in [0, 1]."""
        return self.q_c

    def update(
        self,
        y: int,
        score_y: Optional[float] = None,
        err_y: Optional[int] = None,
        *,
        probs: Optional[Sequence[float]] = None,
    ) -> None:
        """O(1) update after observing true class y and either (score_y, err_y) or probs."""
        if _is_bool_scalar(y) or not isinstance(y, (int, np.integer)):
            raise ValueError(f"Class index y must be an integer, got {y!r}")
        cls_idx = int(y)
        if cls_idx < 0 or cls_idx >= self.num_classes:
            raise IndexError(f"Class index {cls_idx} out of range [0, {self.num_classes})")
        err_int: Optional[int] = None
        if err_y is not None:
            if _is_bool_scalar(err_y) or not isinstance(err_y, (int, np.integer)):
                raise ValueError(f"err_y must be 0 or 1 (non-boolean integer), got {err_y!r}")
            err_int = int(err_y)
            if err_int not in (0, 1):
                raise ValueError(f"err_y must be 0 or 1, got {err_y!r}")
        if probs is not None:
            p_arr = self._validate_probs(probs)
            if score_y is None:
                s_val = 1.0 - float(p_arr[cls_idx])
            else:
                if _is_bool_scalar(score_y) or not isinstance(score_y, (int, float, np.integer, np.floating)):
                    raise ValueError(f"score_y must be finite in [0, 1], got {score_y!r}")
                s_val = float(score_y)
        else:
            if score_y is None:
                raise ValueError("Either score_y or probs must be provided to update()")
            if _is_bool_scalar(score_y) or not isinstance(score_y, (int, float, np.integer, np.floating)):
                raise ValueError(f"score_y must be finite in [0, 1], got {score_y!r}")
            s_val = float(score_y)
        if not math.isfinite(s_val) or s_val < -1e-9 or s_val > 1.0 + 1e-9:
            raise ValueError(f"Nonconformity score {s_val!r} must be finite in [0, 1]")
        s_val = min(1.0, max(0.0, s_val))
        e_flag = (1 if s_val > float(self.q_c[cls_idx]) else 0) if err_int is None else err_int

        s_clip = s_val if s_val < 1.0 - 1e-7 else (1.0 - 1e-7)
        u_obs = -math.log(max(1.0 - s_clip, 1e-7))
        if u_obs > self.u_max:
            u_obs = self.u_max
        fe = 1.0 if e_flag else 0.0
        self.err_ewma[cls_idx] = 0.85 * self.err_ewma[cls_idx] + 0.15 * fe
        shift = self.err_ewma[cls_idx] > 1.6 * self.alpha_target
        boost = 1.6 if shift else 1.0
        a_next = self.alpha_c[cls_idx] + self.gamma_aci * boost * (self.alpha_target - fe)
        if a_next < self.alpha_min:
            a_next = self.alpha_min
        elif a_next > self.alpha_max:
            a_next = self.alpha_max
        self.alpha_c[cls_idx] = a_next
        u_next = self.u_c[cls_idx] + self.eta_u * boost * (fe - a_next)
        if e_flag and shift:
            jump_cap = max(0.01, self.u_max - 0.5)
            u_jump = (1.0 - self.jump_w) * self.u_c[cls_idx] + self.jump_w * min(u_obs, jump_cap)
            if u_jump > u_next:
                u_next = u_jump
        if u_next < 0.01:
            u_next = 0.01
        elif u_next > self.u_max:
            u_next = self.u_max
        self.u_c[cls_idx] = u_next
        self.q_c[cls_idx] = 1.0 if u_next >= self.u_max - 1e-6 else (1.0 - math.exp(-u_next))
        self.steps += 1

    def _validate_probs(self, probs: Sequence[float]) -> np.ndarray:
        """Validate 1D finite probability vector in [0.0, 1.0] (rejecting bool/str/object)."""
        p_arr = _validate_real_1d_vec(probs, self.num_classes, "probs")
        if not bool(np.all((p_arr >= 0.0) & (p_arr <= 1.0))):
            raise ValueError("probs entries must lie in [0.0, 1.0]")
        return p_arr

    def predict_set(self, probs: Sequence[float]) -> List[int]:
        """Return class indices whose nonconformity score (1 - p_k) <= q_c[k]."""
        p_arr = self._validate_probs(probs)
        scores = 1.0 - p_arr
        return [k for k in range(self.num_classes) if float(scores[k]) <= float(self.q_c[k]) + 1e-12]

    def should_accept(self, probs: Sequence[float]) -> bool:
        """Accept locally when the Mondrian prediction set is a singleton."""
        return len(self.predict_set(probs)) == 1

    def state_dict(self) -> Dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "alpha_target": self.alpha_target,
            "gamma_aci": self.gamma_aci,
            "eta_u": self.eta_u,
            "jump_w": self.jump_w,
            "q_c": [float(v) for v in self.q_c],
            "u_c": [float(v) for v in self.u_c],
            "alpha_c": [float(v) for v in self.alpha_c],
            "err_ewma": [float(v) for v in self.err_ewma],
            "steps": self.steps,
        }

    @classmethod
    def from_state_dict(cls, data: Mapping[str, Any]) -> "OnlineMondrianConformal":
        if not isinstance(data, Mapping):
            raise ValueError("OnlineMondrianConformal state must be a mapping")
        if "num_classes" not in data or "alpha_target" not in data:
            raise ValueError("OnlineMondrianConformal state requires 'num_classes' and 'alpha_target'")
        if "gamma_aci" in data:
            gamma_val = data["gamma_aci"]
            if gamma_val is None:
                raise ValueError("gamma_aci cannot be None when explicitly present in state_dict")
        else:
            gamma_val = None
        if "eta_u" in data:
            eta_val = data["eta_u"]
            if eta_val is None:
                raise ValueError("eta_u cannot be None when explicitly present in state_dict")
        else:
            eta_val = 0.30
        if "jump_w" in data:
            jump_val = data["jump_w"]
            if jump_val is None:
                raise ValueError("jump_w cannot be None when explicitly present in state_dict")
        else:
            jump_val = 0.12
        obj = cls(
            num_classes=data["num_classes"],
            alpha=data["alpha_target"],
            q_init=None,
            gamma_aci=gamma_val,
            eta_u=eta_val,
            jump_w=jump_val,
        )
        if "q_c" in data and data["q_c"] is not None:
            q_arr = _validate_real_1d_vec(data["q_c"], obj.num_classes, "q_c").copy()
            if float(np.min(q_arr)) < 0.0 or float(np.max(q_arr)) > 1.0:
                raise ValueError("q_c values must lie in [0.0, 1.0]")
            obj.q_c = q_arr
        if "u_c" in data and data["u_c"] is not None:
            u_arr = _validate_real_1d_vec(data["u_c"], obj.num_classes, "u_c").copy()
            if float(np.min(u_arr)) < 0.0 or float(np.max(u_arr)) > obj.u_max:
                raise ValueError(f"u_c values must lie in [0.0, {obj.u_max}]")
            obj.u_c = u_arr
        else:
            obj.u_c = np.where(
                obj.q_c >= 1.0 - math.exp(-(obj.u_max - 1e-6)),
                obj.u_max,
                -np.log(np.clip(1.0 - obj.q_c, 1e-12, 1.0)),
            )
        if "alpha_c" in data and data["alpha_c"] is not None:
            a_arr = _validate_real_1d_vec(data["alpha_c"], obj.num_classes, "alpha_c").copy()
            if float(np.min(a_arr)) <= 0.0 or float(np.max(a_arr)) >= 1.0:
                raise ValueError("alpha_c values must lie in (0.0, 1.0)")
            obj.alpha_c = a_arr
        if "err_ewma" in data and data["err_ewma"] is not None:
            e_arr = _validate_real_1d_vec(data["err_ewma"], obj.num_classes, "err_ewma").copy()
            if float(np.min(e_arr)) < 0.0 or float(np.max(e_arr)) > 1.0:
                raise ValueError("err_ewma values must lie in [0.0, 1.0]")
            obj.err_ewma = e_arr
        raw_steps = data.get("steps", 0)
        if _is_bool_scalar(raw_steps):
            raise ValueError(f"steps must be a nonnegative integer, got {raw_steps!r}")
        if isinstance(raw_steps, (int, np.integer)):
            steps = int(raw_steps)
        elif (
            isinstance(raw_steps, (float, np.floating))
            and math.isfinite(float(raw_steps))
            and float(raw_steps).is_integer()
        ):
            steps = int(raw_steps)
        else:
            raise ValueError(f"steps must be a nonnegative integer, got {raw_steps!r}")
        if steps < 0:
            raise ValueError(f"steps must be nonnegative, got {steps}")
        obj.steps = steps
        return obj


VALID_MONITOR_MODES = (
    "capital_ons",
    "ons_betting",
    "capital_eb",
    "fixed_mixture",
)


@dataclass(frozen=True)
class JointSelectiveCertificate:
    """Finite-sample Joint LTT selective-risk certificate for factor-graph MAP decoding."""

    qualified: bool
    threshold: Optional[float]
    score_kind: str
    alpha: float
    delta: float
    empirical_risk: float
    empirical_coverage: float
    n_calibration: int
    num_states: int
    field_names: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qualified": bool(self.qualified),
            "threshold": None if self.threshold is None else float(self.threshold),
            "score_kind": str(self.score_kind),
            "alpha": float(self.alpha),
            "delta": float(self.delta),
            "empirical_risk": float(self.empirical_risk),
            "empirical_coverage": float(self.empirical_coverage),
            "n_calibration": int(self.n_calibration),
            "num_states": int(self.num_states),
            "field_names": list(self.field_names),
        }


def _normalize_joint_field_value(opts: Tuple[Any, ...], raw_val: Any) -> Any:
    if opts == (False, True):
        if isinstance(raw_val, bool):
            return raw_val
        if isinstance(raw_val, str):
            low = raw_val.strip().lower()
            if low in ("true", "1", "yes"):
                return True
            if low in ("false", "0", "no"):
                return False
        return bool(raw_val)
    return raw_val


class JointSchemaCalibrator:
    """Pairwise factor-graph exact MAP decoder with Joint LTT selective-risk qualification."""

    def __init__(
        self,
        schema: Any,
        *,
        gamma: float = 1.0,
        alpha: float = 0.05,
        delta: float = 0.05,
        score_kind: str = "margin",
        smoothing: float = 1e-3,
        grid_size: int = 25,
        max_frac: float = 0.95,
    ) -> None:
        if score_kind not in ("margin", "prob"):
            raise ValueError(f"score_kind must be 'margin' or 'prob', got {score_kind!r}")
        if not math.isfinite(float(gamma)) or float(gamma) < 0.0:
            raise ValueError(f"gamma must be finite and >= 0, got {gamma}")
        if not (0.0 < float(alpha) < 1.0) or not math.isfinite(float(alpha)):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not (0.0 < float(delta) < 1.0) or not math.isfinite(float(delta)):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if not math.isfinite(float(smoothing)) or float(smoothing) <= 0.0:
            raise ValueError(f"smoothing must be finite and > 0, got {smoothing}")

        field_options: Dict[str, Tuple[Any, ...]] = {}
        if isinstance(schema, Mapping):
            for fname, opts in schema.items():
                opt_t = tuple(opts)
                if len(opt_t) < 2:
                    raise ValueError(f"Field {fname!r} must have at least 2 options")
                field_options[str(fname)] = opt_t
        else:
            from system1.schema import BooleanField, ChoiceField, DecisionSchema

            if isinstance(schema, type) and issubclass(schema, DecisionSchema):
                schema_obj = schema()
            elif isinstance(schema, DecisionSchema):
                schema_obj = schema
            else:
                raise TypeError(f"Expected DecisionSchema or Mapping, got {type(schema).__name__}")
            for fname, fdef in schema_obj.fields.items():
                if isinstance(fdef, ChoiceField):
                    field_options[fname] = tuple(fdef.options)
                elif isinstance(fdef, BooleanField):
                    field_options[fname] = (False, True)

        if not (2 <= len(field_options) <= 8):
            raise ValueError(
                f"JointSchemaCalibrator supports 2 to 8 discrete fields, got {len(field_options)}"
            )
        num_states = math.prod(len(opts) for opts in field_options.values())
        if num_states > 4096:
            raise ValueError(
                f"Joint state space too large ({num_states} > 4096); reduce fields or cardinality"
            )

        self.field_names: Tuple[str, ...] = tuple(field_options.keys())
        self.field_options: Dict[str, Tuple[Any, ...]] = field_options
        self._opt_index: Dict[str, Dict[Any, int]] = {
            fname: {opt: idx for idx, opt in enumerate(opts)}
            for fname, opts in field_options.items()
        }
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.score_kind = str(score_kind)
        self.smoothing = float(smoothing)
        self.grid_size = int(grid_size)
        self.max_frac = float(max_frac)
        self.num_states = int(num_states)

        combos = list(itertools.product(*[range(len(field_options[fn])) for fn in self.field_names]))
        self._combos: List[Tuple[int, ...]] = combos
        self._combo_idx = np.asarray(combos, dtype=np.intp)
        self._combo_prior = np.zeros(len(combos), dtype=np.float64)
        self.certificate: Optional[JointSelectiveCertificate] = None

    def fit_structure(self, train_labels: Sequence[Mapping[str, Any]]) -> "JointSchemaCalibrator":
        """Estimate pairwise empirical log-compatibility potentials from training labels."""
        if not train_labels:
            raise ValueError("train_labels must be non-empty")
        m_fields = len(self.field_names)
        prior = np.zeros(self.num_states, dtype=np.float64)
        for i in range(m_fields):
            fi = self.field_names[i]
            oi = self.field_options[fi]
            idx_i_map = self._opt_index[fi]
            for j in range(i + 1, m_fields):
                fj = self.field_names[j]
                oj = self.field_options[fj]
                idx_j_map = self._opt_index[fj]
                cnt = np.full((len(oi), len(oj)), self.smoothing, dtype=np.float64)
                for row in train_labels:
                    vi = _normalize_joint_field_value(oi, row[fi])
                    vj = _normalize_joint_field_value(oj, row[fj])
                    cnt[idx_i_map[vi], idx_j_map[vj]] += 1.0
                pair_logp = np.log(cnt / float(np.sum(cnt)))
                prior += self.gamma * pair_logp[self._combo_idx[:, i], self._combo_idx[:, j]]
        self._combo_prior = prior
        return self

    def _coerce_probs_matrix(
        self,
        fname: str,
        raw: Any,
        n_rows: Optional[int],
    ) -> np.ndarray:
        opts = self.field_options[fname]
        k_opts = len(opts)
        if isinstance(raw, np.ndarray):
            arr = np.asarray(raw, dtype=np.float64)
            if arr.ndim == 1:
                if len(arr) != k_opts:
                    raise ValueError(f"Expected length {k_opts} for field {fname!r}, got {len(arr)}")
                arr = arr.reshape(1, k_opts)
            elif arr.ndim != 2 or arr.shape[1] != k_opts:
                raise ValueError(f"Expected shape (N, {k_opts}) for field {fname!r}, got {arr.shape}")
            if n_rows is not None and arr.shape[0] != n_rows:
                raise ValueError(f"Row count mismatch for field {fname!r}: {arr.shape[0]} != {n_rows}")
            return arr

        if isinstance(raw, Mapping):
            rows_list = [raw]
        elif isinstance(raw, Sequence):
            rows_list = list(raw)
        else:
            arr = np.asarray(raw, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] != k_opts:
                raise ValueError(f"Expected {k_opts} columns for field {fname!r}, got {arr.shape[1]}")
            return arr

        if n_rows is not None and len(rows_list) != n_rows:
            raise ValueError(f"Row count mismatch for field {fname!r}: {len(rows_list)} != {n_rows}")
        out = np.zeros((len(rows_list), k_opts), dtype=np.float64)
        for r_i, item in enumerate(rows_list):
            if isinstance(item, Mapping):
                for c_i, opt in enumerate(opts):
                    if opt in item:
                        out[r_i, c_i] = float(item[opt])
                    elif str(opt) in item:
                        out[r_i, c_i] = float(item[str(opt)])
            else:
                vec = np.asarray(item, dtype=np.float64)
                if len(vec) != k_opts:
                    raise ValueError(f"Expected length {k_opts} row for field {fname!r}, got {len(vec)}")
                out[r_i] = vec
        return out

    def decode_batch(
        self,
        probs_by_field: Mapping[str, Any],
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
        """Decode exact factor-graph MAP assignments and selective acceptance decisions."""
        missing = [fn for fn in self.field_names if fn not in probs_by_field]
        if missing:
            raise ValueError(f"probs_by_field is missing fields: {missing}")

        mats: List[np.ndarray] = []
        n_rows: Optional[int] = None
        for fname in self.field_names:
            mat = self._coerce_probs_matrix(fname, probs_by_field[fname], n_rows)
            n_rows = mat.shape[0]
            mats.append(mat)

        n = 0 if n_rows is None else int(n_rows)
        if n == 0:
            return [], np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=bool)

        energies = np.broadcast_to(self._combo_prior, (n, self.num_states)).copy()
        for m, mat in enumerate(mats):
            log_p = np.log(np.clip(mat, 1e-12, 1.0))
            energies += log_p[:, self._combo_idx[:, m]]

        energies -= np.max(energies, axis=1, keepdims=True)
        post = np.exp(energies)
        post /= np.sum(post, axis=1, keepdims=True)

        top1_idx = np.argmax(post, axis=1)
        j_probs = post[np.arange(n), top1_idx].astype(np.float64)
        if self.num_states >= 2:
            second_probs = np.partition(post, -2, axis=1)[:, -2]
            j_margs = (j_probs - second_probs).astype(np.float64)
        else:
            j_margs = j_probs.copy()

        j_dicts: List[Dict[str, Any]] = []
        for s_i in top1_idx.tolist():
            combo = self._combos[s_i]
            j_dicts.append(
                {
                    fname: self.field_options[fname][combo[m]]
                    for m, fname in enumerate(self.field_names)
                }
            )

        if self.certificate is not None and self.certificate.qualified and self.certificate.threshold is not None:
            scores = j_probs if self.score_kind == "prob" else j_margs
            j_acc = (scores >= float(self.certificate.threshold)).astype(bool)
        else:
            j_acc = np.zeros(n, dtype=bool)

        return j_dicts, j_probs, j_margs, j_acc

    def decode_one(self, probs_by_field: Mapping[str, Any]) -> Tuple[Dict[str, Any], float, float, bool]:
        """Decode a single instance via the factor-graph MAP decoder."""
        j_dicts, j_probs, j_margs, j_acc = self.decode_batch(probs_by_field)
        return j_dicts[0], float(j_probs[0]), float(j_margs[0]), bool(j_acc[0])

    def calibrate(
        self,
        calib_probs_by_field: Mapping[str, Any],
        calib_labels: Sequence[Mapping[str, Any]],
    ) -> JointSelectiveCertificate:
        """Fit a finite-sample Joint LTT threshold over factor-graph joint correctness."""
        j_dicts, j_probs, j_margs, _ = self.decode_batch(calib_probs_by_field)
        if len(j_dicts) != len(calib_labels):
            raise ValueError(
                f"calib_probs_by_field length ({len(j_dicts)}) != calib_labels length ({len(calib_labels)})"
            )

        correct = np.array(
            [
                all(
                    _normalize_joint_field_value(self.field_options[fn], pred[fn])
                    == _normalize_joint_field_value(self.field_options[fn], truth[fn])
                    for fn in self.field_names
                )
                for pred, truth in zip(j_dicts, calib_labels)
            ],
            dtype=bool,
        )
        scores = j_probs if self.score_kind == "prob" else j_margs
        cal_ltt = SelectiveRiskCalibrator(
            score_name=f"factor_graph_{self.score_kind}",
            method="ltt_fst",
            alpha=self.alpha,
            delta=self.delta,
            grid_size=self.grid_size,
            max_frac=self.max_frac,
        )
        cert = cal_ltt.fit(scores, correct)
        emp_risk = (
            float(cert.errors_qual) / float(cert.n_accepted_qual)
            if cert.n_accepted_qual > 0
            else 0.0
        )
        emp_cov = (
            float(cert.n_accepted_qual) / float(cert.n_qual)
            if cert.n_qual > 0
            else 0.0
        )
        joint_cert = JointSelectiveCertificate(
            qualified=bool(cert.qualified),
            threshold=None if cert.threshold is None else float(cert.threshold),
            score_kind=self.score_kind,
            alpha=self.alpha,
            delta=self.delta,
            empirical_risk=emp_risk,
            empirical_coverage=emp_cov,
            n_calibration=int(len(calib_labels)),
            num_states=self.num_states,
            field_names=self.field_names,
        )
        self.certificate = joint_cert
        return joint_cert

    def digest(self) -> str:
        """Return deterministic SHA-256 hex digest covering structure potentials and Joint LTT threshold."""
        payload = {
            "field_names": list(self.field_names),
            "field_options": {
                fn: [str(o) for o in opts] for fn, opts in self.field_options.items()
            },
            "gamma": round(self.gamma, 6),
            "alpha": round(self.alpha, 6),
            "delta": round(self.delta, 6),
            "score_kind": self.score_kind,
            "qualified": bool(self.certificate.qualified) if self.certificate is not None else False,
            "threshold": (
                None
                if (self.certificate is None or self.certificate.threshold is None)
                else round(float(self.certificate.threshold), 6)
            ),
            "prior": [round(float(v), 6) for v in self._combo_prior.tolist()],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SequentialMonitorStep:
    """Single audited observation step returned by SequentialRiskMonitor."""

    step: int
    audited_count: int
    e_value: float
    threshold: float
    alarm: bool
    alarm_step: Optional[int]
    empirical_error_rate: float
    bet_fraction: float


class SequentialRiskMonitor:
    """Anytime-valid sequential error monitor backed by nonnegative supermartingales (e-processes)."""

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        delta: float = 0.05,
        mode: str = "capital_ons",
    ) -> None:
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not (0.0 < float(delta) < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {delta!r}")
        if mode == "naive_binom_peeking" or mode not in VALID_MONITOR_MODES:
            raise ValueError(
                f"Unsupported sequential monitor mode {mode!r}; expected one of {VALID_MONITOR_MODES}"
            )
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.mode = str(mode)
        self._threshold = 1.0 / self.delta
        self._lam_max = 0.90 / self.alpha
        self.reset()

    def reset(self) -> None:
        """Reset all wealth, betting, and alarm state."""
        self._audited_count: int = 0
        self._errors: int = 0
        self._e_value: float = 1.0
        self._alarm_triggered: bool = False
        self._alarm_step: Optional[int] = None
        self._a_cap: float = 0.0
        self._c_cap: float = 1.0
        self._a_ons: float = 1.0
        self._sum_x: float = 0.0
        self._sum_sq: float = 0.0
        if self.mode == "capital_ons":
            self._lam = min(1.5, self._lam_max)
        elif self.mode == "ons_betting":
            self._lam = min(0.5, self._lam_max)
        else:
            self._lam = 0.0
        raw_lams = np.array([1.0, 2.0, 4.0, 6.0, 8.0, 12.0], dtype=np.float64)
        filtered = raw_lams[raw_lams <= self._lam_max]
        if filtered.size == 0:
            filtered = np.array([min(self._lam_max, 0.5 * self._lam_max)], dtype=np.float64)
        self._mix_lams = filtered
        self._mix_weights = np.ones(len(filtered), dtype=np.float64) / float(len(filtered))
        self._mix_wealths = np.ones(len(filtered), dtype=np.float64)

    @property
    def e_value(self) -> float:
        return self._e_value

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def alarm_triggered(self) -> bool:
        return self._alarm_triggered

    @property
    def alarm_step(self) -> Optional[int]:
        return self._alarm_step

    @property
    def audited_count(self) -> int:
        return self._audited_count

    def update(self, is_error: bool, *, step: Optional[int] = None) -> SequentialMonitorStep:
        """Ingest one audited error indicator and return updated e-process state."""
        self._audited_count += 1
        k = self._audited_count
        x = 1.0 if bool(is_error) else 0.0
        if is_error:
            self._errors += 1
        step_val = k if step is None else int(step)
        diff = x - self.alpha

        if self.mode == "fixed_mixture":
            self._mix_wealths *= 1.0 + self._mix_lams * diff
            e_val = float(np.dot(self._mix_weights, self._mix_wealths))
            bet_used = float(np.dot(self._mix_weights, self._mix_lams))
        elif self.mode == "ons_betting":
            bet_used = float(min(self._lam, self._lam_max))
            factor = 1.0 + bet_used * diff
            self._e_value = float(self._e_value * factor)
            e_val = self._e_value
            g = diff / max(1e-12, factor)
            z = -g
            self._a_ons += z * z
            self._lam = float(
                np.clip(
                    self._lam - (2.0 / (2.0 - math.log(3.0))) * z / self._a_ons,
                    0.0,
                    self._lam_max,
                )
            )
        elif self.mode == "capital_ons":
            bet_used = float(min(self._lam, self._lam_max))
            w_k = 1.0 / (float(k) * float(k + 1))
            factor = 1.0 + bet_used * diff
            self._a_cap = (self._a_cap + w_k) * factor
            self._c_cap -= w_k
            e_val = float(self._a_cap + self._c_cap)
            g = diff / max(1e-12, factor)
            z = -g
            self._a_ons += z * z
            self._lam = float(
                np.clip(
                    self._lam - (2.0 / (2.0 - math.log(3.0))) * z / self._a_ons,
                    0.0,
                    self._lam_max,
                )
            )
        else:  # capital_eb
            w_k = 1.0 / (float(k) * float(k + 1))
            mu_hat = (20.0 * self.alpha + self._sum_x) / (20.0 + float(k - 1))
            v_hat = (0.25 + self._sum_sq) / (1.0 + float(k - 1))
            eb_step = math.sqrt(2.0 * math.log(1.0 / self.delta) / ((float(k) + 20.0) * max(1e-4, v_hat)))
            kelly = max(0.0, (mu_hat - self.alpha) / (self.alpha * (1.0 - self.alpha)))
            bet_used = float(np.clip(max(kelly, 0.5 * eb_step), 0.0, self._lam_max))
            self._a_cap = (self._a_cap + w_k) * (1.0 + bet_used * diff)
            self._c_cap -= w_k
            e_val = float(self._a_cap + self._c_cap)
            self._sum_x += x
            self._sum_sq += (x - mu_hat) ** 2

        self._e_value = e_val
        if e_val >= self._threshold and not self._alarm_triggered:
            self._alarm_triggered = True
            self._alarm_step = step_val

        return SequentialMonitorStep(
            step=step_val,
            audited_count=k,
            e_value=e_val,
            threshold=self._threshold,
            alarm=self._alarm_triggered,
            alarm_step=self._alarm_step,
            empirical_error_rate=float(self._errors) / float(k),
            bet_fraction=bet_used,
        )

    def update_batch(
        self,
        events: Sequence[Union[bool, Tuple[int, bool]]],
    ) -> List[SequentialMonitorStep]:
        """Ingest a sequence of error booleans or (step, is_error) tuples."""
        out: List[SequentialMonitorStep] = []
        for ev in events:
            if isinstance(ev, tuple):
                t_step, err_flag = ev
                out.append(self.update(bool(err_flag), step=int(t_step)))
            else:
                out.append(self.update(bool(ev)))
        return out


# ---------------------------------------------------------------------------
# Wave 3: Pure-NumPy Decision Utility, Ordinal Calibration & Platt/Beta Head
# ---------------------------------------------------------------------------


def _expit_np(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Numerically bounded pure-NumPy logistic sigmoid."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        arr = np.asarray(x, dtype=np.float64)
        nan_mask = np.isnan(arr)
        arr_safe = np.where(nan_mask, 0.0, arr)
        clipped = np.clip(arr_safe, -60.0, 60.0)
        out = np.where(nan_mask, np.nan, 1.0 / (1.0 + np.exp(-clipped)))
    if np.ndim(x) == 0:
        return float(out)
    return out


def _softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable pure-NumPy softmax failing closed with NaN on non-finite slices."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        arr = np.asarray(logits, dtype=np.float64)
        if arr.size == 0:
            return arr.copy()
        slice_ok = np.all(np.isfinite(arr), axis=axis, keepdims=True)
        arr_safe = np.where(slice_ok, arr, 0.0)
        shifted = arr_safe - np.max(arr_safe, axis=axis, keepdims=True)
        exps = np.exp(np.clip(shifted, -60.0, 60.0))
        denom = np.maximum(np.sum(exps, axis=axis, keepdims=True), 1e-12)
        out = exps / denom
        return np.where(slice_ok, out, np.nan)


def _safe_l2_norm(v: np.ndarray) -> float:
    """Compute overflow-safe Euclidean norm of a finite 1D vector via max-abs scaling."""
    v64 = np.asarray(v, dtype=np.float64).ravel()
    if v64.size == 0:
        return 0.0
    if not bool(np.all(np.isfinite(v64))):
        return float("nan")
    scale = float(np.max(np.abs(v64)))
    if scale == 0.0:
        return 0.0
    with np.errstate(over="ignore", under="ignore"):
        return float(scale * np.linalg.norm(v64 / scale))


def _minimize_lbfgs_np(
    fun: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    x0: np.ndarray,
    *,
    bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
    maxiter: int = 250,
    gtol: float = 1e-6,
    ftol: float = 1e-12,
    m: int = 10,
) -> Tuple[np.ndarray, float, int]:
    """Pure-NumPy box-projected L-BFGS optimizer with Armijo backtracking line search."""
    x = np.asarray(x0, dtype=np.float64).copy().ravel()
    n = x.size
    if bounds is not None:
        if len(bounds) != n:
            raise ValueError(f"Expected {n} bounds entries, got {len(bounds)}")
        lb = np.array([-np.inf if lo is None else float(lo) for lo, _ in bounds], dtype=np.float64)
        ub = np.array([np.inf if hi is None else float(hi) for _, hi in bounds], dtype=np.float64)
    else:
        lb = np.full(n, -np.inf, dtype=np.float64)
        ub = np.full(n, np.inf, dtype=np.float64)

    x = np.clip(x, lb, ub)
    f_raw, g_raw = fun(x)
    f = float(f_raw)
    g = np.asarray(g_raw, dtype=np.float64).ravel()
    if not math.isfinite(f) or not bool(np.all(np.isfinite(g))):
        raise ValueError("Initial optimization objective or gradient is non-finite")

    def _project_grad(xv: np.ndarray, gv: np.ndarray) -> np.ndarray:
        pg = gv.copy()
        pg[(xv <= lb + 1e-12) & (gv > 0.0)] = 0.0
        pg[(xv >= ub - 1e-12) & (gv < 0.0)] = 0.0
        return pg

    history: List[Tuple[np.ndarray, np.ndarray, float]] = []
    iters = 0
    for iters in range(1, max(1, int(maxiter)) + 1):
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            pg = _project_grad(x, g)
            if pg.size == 0 or float(np.max(np.abs(pg))) <= gtol:
                break

            # Two-loop L-BFGS search direction on projected gradient.
            q = pg.copy()
            alphas = [0.0] * len(history)
            for idx in range(len(history) - 1, -1, -1):
                s_i, y_i, rho_i = history[idx]
                alpha_i = rho_i * float(np.dot(s_i, q))
                alphas[idx] = alpha_i
                q = q - alpha_i * y_i

            if history:
                s_last, y_last, _ = history[-1]
                yy = float(np.dot(y_last, y_last))
                sy_last = float(np.dot(s_last, y_last))
                if math.isfinite(yy) and math.isfinite(sy_last) and sy_last > 0.0:
                    gamma = max(min(sy_last / max(yy, 1e-12), 1e6), 1e-6)
                else:
                    gamma = 1.0
            else:
                gamma = 1.0

            r = gamma * q
            for idx, (s_i, y_i, rho_i) in enumerate(history):
                beta_i = rho_i * float(np.dot(y_i, r))
                r = r + s_i * (alphas[idx] - beta_i)

            d = -r
            d[(x <= lb + 1e-12) & (d < 0.0)] = 0.0
            d[(x >= ub - 1e-12) & (d > 0.0)] = 0.0
            g_norm = _safe_l2_norm(g)
            gd = float(np.dot(g, d))
            if (
                not bool(np.all(np.isfinite(d)))
                or math.isnan(gd)
                or (
                    math.isfinite(gd)
                    and (not math.isfinite(g_norm) or gd >= -1e-14 * max(1.0, g_norm))
                )
            ):
                d = -pg
                history.clear()

            step = 1.0
            accepted = False
            x_next = x
            f_next = f
            g_next = g
            for _ in range(36):
                cand = np.clip(x + step * d, lb, ub)
                if not bool(np.all(np.isfinite(cand))):
                    step *= 0.5
                    continue
                delta_x = cand - x
                slope = float(np.dot(g, delta_x))
                if math.isnan(slope) or slope >= 0.0:
                    step *= 0.5
                    continue
                cand_f, cand_g = fun(cand)
                cand_f_f = float(cand_f)
                cand_g_arr = np.asarray(cand_g, dtype=np.float64).ravel()
                armijo_ok = (
                    cand_f_f < f
                    if slope == -math.inf
                    else cand_f_f <= f + 1e-4 * slope
                )
                if (
                    math.isfinite(cand_f_f)
                    and bool(np.all(np.isfinite(cand_g_arr)))
                    and armijo_ok
                ):
                    x_next = cand
                    f_next = cand_f_f
                    g_next = cand_g_arr
                    accepted = True
                    break
                step *= 0.5

            if not accepted:
                if history:
                    history.clear()
                    continue
                break

            s_vec = x_next - x
            y_vec = g_next - g
            rel_f = abs(f - f_next) / max(1.0, abs(f))
            x, f, g = x_next, f_next, g_next
            if rel_f <= ftol and float(np.max(np.abs(s_vec))) <= 1e-11:
                break

            if bool(np.all(np.isfinite(s_vec))) and bool(np.all(np.isfinite(y_vec))):
                sy = float(np.dot(s_vec, y_vec))
                ss = float(np.dot(s_vec, s_vec))
                if math.isfinite(sy) and math.isfinite(ss) and sy > 1e-10 * max(ss, 1e-12):
                    inv_sy = 1.0 / sy
                    if math.isfinite(inv_sy):
                        history.append((s_vec, y_vec, inv_sy))
                        if len(history) > m:
                            history.pop(0)

    return x, f, iters


def should_use_soft_ordinal(n_classes: int) -> bool:
    """Return True when soft ordinal heads apply (small ordinal schemas 2 <= K <= 5)."""
    return 2 <= int(n_classes) <= 5


class UtilityCalibrator:
    """Bayes expected-cost decision rule + Conformal Risk Control (CRC) gating.

    Uses pure-NumPy temperature scaling and finite-sample CRC calibration over an
    ``(A, K)`` cost matrix ``C[action, true_class]``. When finite-sample data
    cannot certify non-vacuous excess-risk control, ``lam_crc`` is set to
    ``-1.0``, ``is_vacuous`` is True, and ``predict`` falls back to ``safe_action``.
    """

    def __init__(
        self,
        cost_matrix: Any,
        *,
        safe_action: Optional[int] = None,
        alpha: float = 0.05,
        loss_mode: str = "normalized_excess",
        fit_temperature: bool = False,
        temperature: float = 1.0,
        ladder_mode: bool = False,
    ) -> None:
        matrix = _coerce_real_numeric_array(cost_matrix, name="cost_matrix")
        if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 2:
            raise ValueError(
                f"cost_matrix must be 2D (n_actions >= 1, n_classes >= 2), got shape {matrix.shape}"
            )
        if not bool(np.all(np.isfinite(matrix))):
            raise ValueError("cost_matrix must contain only finite values")
        if _is_bool_scalar(alpha) or not isinstance(alpha, (int, float, np.integer, np.floating)):
            raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
        alpha_f = float(alpha)
        if not math.isfinite(alpha_f) or not (0.0 < alpha_f < 1.0):
            raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
        if loss_mode not in {"normalized_excess", "indicator"}:
            raise ValueError(f"Unsupported loss_mode {loss_mode!r}")
        if _is_bool_scalar(temperature) or not isinstance(
            temperature, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"temperature must be finite and positive, got {temperature!r}")
        temp_f = float(temperature)
        if not math.isfinite(temp_f) or temp_f <= 0.0:
            raise ValueError(f"temperature must be finite and positive, got {temperature!r}")

        self.cost_matrix = matrix
        self.n_actions, self.n_classes = matrix.shape
        if safe_action is None:
            self.safe_action = int(np.argmin(np.max(matrix, axis=1)))
        else:
            if _is_bool_scalar(safe_action) or not isinstance(safe_action, (int, np.integer)):
                raise ValueError(f"safe_action must be an integer, got {safe_action!r}")
            sa_i = int(safe_action)
            if not (0 <= sa_i < self.n_actions):
                raise ValueError(f"safe_action {safe_action} out of range [0, {self.n_actions})")
            self.safe_action = sa_i
        self.alpha = alpha_f
        self.loss_mode = loss_mode
        self.fit_temperature = bool(fit_temperature)
        self.temperature = max(1e-3, temp_f)
        self.ladder_mode = bool(ladder_mode)

        c_scale = float(np.max(np.abs(self.cost_matrix)))
        self._cost_scale: float = c_scale
        self._scaled_cost_matrix: np.ndarray = (
            (self.cost_matrix / c_scale)
            if c_scale > 0.0
            else np.zeros_like(self.cost_matrix, dtype=np.float64)
        )
        if 0.0 < c_scale < 1.0:
            _, exp_c = math.frexp(c_scale)
            self._upscaled_cost_matrix: Optional[np.ndarray] = np.ldexp(
                self.cost_matrix,
                1 - exp_c,
            )
        else:
            self._upscaled_cost_matrix = None
        eval_cost_mat = (
            self._upscaled_cost_matrix
            if self._upscaled_cost_matrix is not None
            else self.cost_matrix
        )
        self._abs_eval_cost_matrix: np.ndarray = np.abs(eval_cost_mat)
        self._abs_cost_matrix: np.ndarray = np.abs(self.cost_matrix)
        self._exact_cost_rows: Optional[List[Tuple[Fraction, ...]]] = None
        self.c_esc = float(np.max(self.cost_matrix[self.safe_action]))
        self.B = float(np.max(self.cost_matrix))
        self._c_esc_scaled: float = float(np.max(self._scaled_cost_matrix[self.safe_action]))
        with np.errstate(over="ignore", invalid="ignore"):
            raw_span = self.B - self.c_esc
        self.denom = (
            float(raw_span)
            if (math.isfinite(raw_span) and raw_span > 1e-9)
            else (float("inf") if not math.isfinite(raw_span) else max(1e-9, float(raw_span)))
        )
        self.L = np.zeros_like(self.cost_matrix, dtype=np.float64)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            pos_mask = self.cost_matrix > self.c_esc
            if math.isfinite(raw_span):
                if raw_span > 0.0 and bool(np.any(pos_mask)):
                    self.L[pos_mask] = np.clip(
                        (self.cost_matrix[pos_mask] - self.c_esc) / raw_span,
                        0.0,
                        1.0,
                    )
            else:
                span_scale = max(abs(self.B), abs(self.c_esc))
                if span_scale > 0.0 and bool(np.any(pos_mask)):
                    c_sp = self.cost_matrix / span_scale
                    esc_sp = self.c_esc / span_scale
                    denom_sp = (self.B / span_scale) - esc_sp
                    if math.isfinite(denom_sp) and denom_sp > 0.0:
                        self.L[pos_mask] = np.clip(
                            (c_sp[pos_mask] - esc_sp) / denom_sp,
                            0.0,
                            1.0,
                        )
        self._indicator_loss_matrix: np.ndarray = (self.cost_matrix > self.c_esc).astype(np.float64)

        self.lam_crc: float = -1.0
        self._is_fitted: bool = False

    @staticmethod
    def _valid_prob_rows(p2: np.ndarray) -> np.ndarray:
        if p2.size == 0:
            return np.zeros(p2.shape[0], dtype=bool)
        with np.errstate(over="ignore", invalid="ignore"):
            fin = np.all(np.isfinite(p2), axis=1)
            nonneg = np.all(p2 >= -1e-9, axis=1)
            safe_p = np.where(fin[:, None], p2, 0.0)
            rmax = np.max(np.abs(safe_p), axis=1, keepdims=True)
            safe_m = np.where(rmax > 0.0, rmax, 1.0)
            pos_sum = (rmax[:, 0] > 0.0) & (np.sum(safe_p / safe_m, axis=1) > 0.0)
        return fin & nonneg & pos_sum

    @staticmethod
    def _normalize_prob_rows(p_rows: np.ndarray) -> np.ndarray:
        if p_rows.size == 0:
            return p_rows
        with np.errstate(over="ignore", invalid="ignore"):
            raw_sums = np.sum(p_rows, axis=1, keepdims=True)
            already_unit = np.isfinite(raw_sums) & (np.abs(raw_sums - 1.0) <= 1e-9)
            rmax = np.max(np.abs(p_rows), axis=1, keepdims=True)
            safe_m = np.where(rmax > 0.0, rmax, 1.0)
            scaled = p_rows / safe_m
            scaled_sums = np.sum(scaled, axis=1, keepdims=True)
            normed = scaled / np.where(scaled_sums > 0.0, scaled_sums, 1.0)
            return np.where(already_unit, p_rows, normed)

    def _validate_cal_labels(self, y_raw: Any, n_rows: int) -> np.ndarray:
        if isinstance(y_raw, (str, bytes, bytearray, memoryview, Mapping, set, frozenset)):
            raise ValueError("Labels must be a 1D array of integer class indices")
        if not isinstance(y_raw, np.ndarray) and _contains_bool_element(y_raw):
            raise ValueError("Labels must have integer dtype, not boolean")
        y_np = np.asarray(y_raw)
        if y_np.ndim != 1:
            raise ValueError(f"Labels must be a 1D array of integer class indices, got shape {y_np.shape}")
        if y_np.dtype == np.dtype(np.object_):
            out_vals: List[int] = []
            for v in y_np.flat:
                if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
                    raise ValueError(f"Labels must have integer dtype, got element {type(v).__name__}")
                iv = int(v)
                if iv < -9223372036854775808 or iv > 9223372036854775807:
                    raise ValueError(f"Label integer index out of int64 bounds: {iv}")
                out_vals.append(iv)
            y_arr = np.asarray(out_vals, dtype=np.int64).ravel()
        else:
            if y_np.dtype == np.dtype(np.bool_) or y_np.dtype.kind not in ("i", "u"):
                raise ValueError(f"Labels must have integer dtype, got {y_np.dtype}")
            if (
                y_np.dtype == np.dtype(np.uint64)
                and y_np.size > 0
                and bool(np.any(y_np > np.uint64(np.iinfo(np.int64).max)))
            ):
                raise ValueError("Label integer indices exceed int64 bounds")
            y_arr = y_np.astype(np.int64, copy=False).ravel()
        if y_arr.size > 0 and (int(np.min(y_arr)) < 0 or int(np.max(y_arr)) >= self.n_classes):
            raise ValueError(f"Labels must lie in [0, {self.n_classes}), got [{int(np.min(y_arr))}, {int(np.max(y_arr))}]")
        return y_arr

    @property
    def is_vacuous(self) -> bool:
        """True when CRC has no certified non-safe threshold (lam_crc < 0)."""
        return (not self._is_fitted) or (self.lam_crc < 0.0)

    def _fit_temperature(self, probs: np.ndarray, y: np.ndarray) -> float:
        p_norm = self._normalize_prob_rows(probs)
        logits = np.log(np.clip(p_norm, 1e-12, 1.0))
        n = logits.shape[0]
        rows = np.arange(n)
        y_onehot = np.zeros_like(logits)
        y_onehot[rows, y] = 1.0

        def _obj(params: np.ndarray) -> Tuple[float, np.ndarray]:
            beta = float(params[0])
            inv_t = math.exp(beta)
            scaled = inv_t * logits
            shifted = scaled - np.max(scaled, axis=1, keepdims=True)
            exps = np.exp(np.clip(shifted, -60.0, 60.0))
            denom = np.sum(exps, axis=1, keepdims=True)
            log_p = shifted - np.log(np.maximum(denom, 1e-12))
            p = exps / np.maximum(denom, 1e-12)
            nll = -float(np.mean(log_p[rows, y])) + 0.5 * 1e-4 * (beta ** 2)
            d_inv_t = float(np.mean(np.sum((p - y_onehot) * logits, axis=1)))
            grad_beta = inv_t * d_inv_t + 1e-4 * beta
            return nll, np.array([grad_beta], dtype=np.float64)

        opt, _, _ = _minimize_lbfgs_np(
            _obj,
            np.array([0.0], dtype=np.float64),
            bounds=[(math.log(0.1), math.log(10.0))],
            maxiter=80,
        )
        return float(1.0 / math.exp(float(opt[0])))

    def transform_probs(self, probs: np.ndarray) -> np.ndarray:
        """Apply learned temperature scaling when ``temperature != 1.0``."""
        p = _coerce_real_numeric_array(probs, name="probs")
        single = p.ndim == 1
        p2 = p.reshape(1, -1) if single else p
        if p2.ndim != 2 or p2.shape[1] != self.n_classes:
            raise ValueError(f"Expected {self.n_classes} classes, got shape {p.shape}")
        valid = self._valid_prob_rows(p2)
        scaled = np.full(p2.shape, np.nan, dtype=np.float64)
        if bool(np.any(valid)):
            p_ok = self._normalize_prob_rows(p2[valid])
            if not math.isfinite(self.temperature) or self.temperature <= 0.0:
                return scaled[0] if single else scaled
            if abs(self.temperature - 1.0) <= 1e-12:
                scaled[valid] = p_ok
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    logits = np.log(np.clip(p_ok, 1e-12, 1.0))
                    s_ok = _softmax_np(logits / max(self.temperature, 1e-3), axis=1)
                s_fin = np.all(np.isfinite(s_ok), axis=1)
                if bool(np.any(s_fin)):
                    v_idx = np.flatnonzero(valid)[s_fin]
                    scaled[v_idx] = s_ok[s_fin]
        return scaled[0] if single else scaled

    def expected_costs(self, probs: np.ndarray, *, apply_temperature: bool = True) -> np.ndarray:
        """Return expected cost matrix product ``probs @ C.T`` of shape ``(..., A)``."""
        p = (
            self.transform_probs(probs)
            if apply_temperature
            else _coerce_real_numeric_array(probs, name="probs")
        )
        single = p.ndim == 1
        p2 = p.reshape(1, -1) if single else p
        if p2.ndim != 2 or p2.shape[1] != self.n_classes:
            raise ValueError(f"Expected {self.n_classes} classes, got shape {p.shape}")
        valid = self._valid_prob_rows(p2)
        costs = np.full((p2.shape[0], self.n_actions), np.nan, dtype=np.float64)
        if bool(np.any(valid)):
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                p_ok = self._normalize_prob_rows(p2[valid])
                c_ok = p_ok @ self.cost_matrix.T
                cond_ok = p_ok @ self._abs_cost_matrix.T
            gamma_ec = (float(self.n_classes) + 2.0) * float(np.finfo(np.float64).eps)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                need_exact = ~np.isfinite(c_ok) | (
                    (cond_ok > 0.0) & (np.abs(c_ok) <= 1024.0 * gamma_ec * cond_ok)
                )
            bad_ec_rows = np.flatnonzero(np.any(need_exact, axis=1))
            if bad_ec_rows.size > 0:
                if self._exact_cost_rows is None:
                    self._exact_cost_rows = [
                        tuple(Fraction(float(val)) for val in row)
                        for row in self.cost_matrix
                    ]
                for r_i in bad_ec_rows:
                    p_fracs = tuple(Fraction(float(pv)) for pv in p_ok[r_i])
                    for a_i in range(self.n_actions):
                        tot = sum(
                            (p_f * c_f for p_f, c_f in zip(p_fracs, self._exact_cost_rows[a_i]) if p_f != 0),
                            Fraction(0),
                        )
                        try:
                            val_f = float(tot)
                        except OverflowError:
                            val_f = float("nan")
                        c_ok[r_i, a_i] = val_f if math.isfinite(val_f) else float("nan")
            c_fin = np.all(np.isfinite(c_ok), axis=1)
            if bool(np.any(c_fin)):
                v_idx = np.flatnonzero(valid)[c_fin]
                costs[v_idx] = c_ok[c_fin]
        return costs[0] if single else costs

    def _expected_policy_costs(self, p_norm: np.ndarray) -> np.ndarray:
        """Evaluate expected costs using raw or power-of-two upscaled costs, falling back per-row to scaled costs."""
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            costs = p_norm @ self.cost_matrix.T
            if self._upscaled_cost_matrix is not None:
                costs = p_norm @ self._upscaled_cost_matrix.T
            bad_rows = ~np.all(np.isfinite(costs), axis=1)
            if bool(np.any(bad_rows)):
                costs[bad_rows] = p_norm[bad_rows] @ self._scaled_cost_matrix.T
        return costs

    def _select_bayes_actions(self, p_norm: np.ndarray, costs: np.ndarray) -> np.ndarray:
        """Select Bayes-optimal actions with exact Fraction tie-breaking on floating-point ties."""
        a_bayes = np.argmin(costs, axis=1).astype(np.int64)
        if self.n_actions <= 1 or p_norm.shape[0] == 0:
            return a_bayes
        row_idx = np.arange(costs.shape[0])
        row_mins = costs[row_idx, a_bayes][:, None]
        gamma = (float(self.n_classes) + 2.0) * float(np.finfo(np.float64).eps)
        subnorm_tol = float(self.n_classes) * float(np.finfo(np.float64).tiny)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            cond_abs = p_norm @ self._abs_eval_cost_matrix.T
            err_bound = gamma * cond_abs + subnorm_tol
            pair_tol = err_bound + err_bound[row_idx, a_bayes][:, None]
            diff = costs - row_mins
            tie_mask = (
                (costs == row_mins)
                | (~np.isfinite(pair_tol))
                | (~np.isfinite(diff))
                | (diff <= pair_tol)
            )
        tie_rows = np.flatnonzero(np.sum(tie_mask, axis=1) > 1)
        if tie_rows.size == 0:
            return a_bayes
        if self._exact_cost_rows is None:
            self._exact_cost_rows = [
                tuple(Fraction(float(val)) for val in row)
                for row in self.cost_matrix
            ]
        memo: Dict[Tuple[bytes, Tuple[int, ...]], int] = {}
        for r_i in tie_rows:
            cands = np.flatnonzero(tie_mask[r_i])
            cands_tup = tuple(int(c) for c in cands)
            if bool(np.all(self.cost_matrix[cands] == self.cost_matrix[cands[0]])):
                a_bayes[r_i] = cands_tup[0]
                continue
            p_bytes = np.ascontiguousarray(p_norm[r_i], dtype=np.float64).tobytes()
            m_key = (p_bytes, cands_tup)
            cached = memo.get(m_key)
            if cached is not None:
                a_bayes[r_i] = cached
                continue
            p_fracs = tuple(Fraction(float(pv)) for pv in p_norm[r_i])
            best_act = cands_tup[0]
            best_val: Optional[Fraction] = None
            for ca in cands_tup:
                row_fracs = self._exact_cost_rows[ca]
                tot = sum((p_f * c_f for p_f, c_f in zip(p_fracs, row_fracs) if p_f != 0), Fraction(0))
                if best_val is None or tot < best_val:
                    best_val = tot
                    best_act = ca
            memo[m_key] = best_act
            a_bayes[r_i] = best_act
        return a_bayes

    def _apply_policy(self, probs_2d: np.ndarray, lam: float) -> np.ndarray:
        n = probs_2d.shape[0]
        acts = np.full(n, self.safe_action, dtype=np.int64)
        if not math.isfinite(float(lam)) or lam < 0.0 or n == 0:
            return acts
        valid = self._valid_prob_rows(probs_2d)
        if not bool(np.any(valid)):
            return acts
        v_idx = np.flatnonzero(valid)
        policy_loss = self._indicator_loss_matrix if self.loss_mode == "indicator" else self.L
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            p_norm = self._normalize_prob_rows(probs_2d[v_idx])
            costs_v = self._expected_policy_costs(p_norm)
            eh_v = p_norm @ policy_loss.T
        row_fin = np.all(np.isfinite(costs_v), axis=1) & np.all(np.isfinite(eh_v), axis=1)
        if not bool(np.any(row_fin)):
            return acts
        v_idx = v_idx[row_fin]
        p_norm = p_norm[row_fin]
        costs_v = costs_v[row_fin]
        eh_v = eh_v[row_fin]
        a_bayes = self._select_bayes_actions(p_norm, costs_v)
        if self.ladder_mode:
            for pos, orig_i in enumerate(v_idx):
                ab = int(a_bayes[pos])
                if ab <= self.safe_action:
                    chosen = self.safe_action
                    for cand in range(ab, self.safe_action + 1):
                        if cand == self.safe_action or float(eh_v[pos, cand]) <= lam + 1e-12:
                            chosen = cand
                            break
                    acts[orig_i] = chosen
                else:
                    acts[orig_i] = ab if float(eh_v[pos, ab]) <= lam + 1e-12 else self.safe_action
            return acts
        eh_best = eh_v[np.arange(len(v_idx)), a_bayes]
        acts[v_idx] = np.where(eh_best <= lam + 1e-12, a_bayes, self.safe_action).astype(np.int64)
        return acts

    def fit_crc(
        self,
        probs_cal: np.ndarray,
        y_cal: np.ndarray,
        alpha: Optional[float] = None,
    ) -> float:
        """Select largest threshold ``lam`` in ``[0, 1]`` satisfying CRC bound ``<= alpha``."""
        if alpha is None:
            eff_alpha = self.alpha
        elif _is_bool_scalar(alpha) or not isinstance(alpha, (int, float, np.integer, np.floating)):
            raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
        else:
            eff_alpha = float(alpha)
        if not math.isfinite(eff_alpha) or not (0.0 < eff_alpha < 1.0):
            raise ValueError(f"alpha must lie in (0, 1), got {eff_alpha!r}")
        p_in = _coerce_real_numeric_array(probs_cal, name="probs_cal")
        if p_in.ndim != 2 or p_in.shape[1] != self.n_classes:
            raise ValueError(f"probs_cal must have shape (N, {self.n_classes}), got {p_in.shape}")
        if p_in.shape[0] > 0 and not bool(np.all(self._valid_prob_rows(p_in))):
            raise ValueError("probs_cal contains non-finite or invalid probability rows")
        y_arr = self._validate_cal_labels(y_cal, p_in.shape[0])
        p_cal = self.transform_probs(p_in)
        if p_cal.shape[0] > 0 and not bool(np.all(np.isfinite(p_cal))):
            raise ValueError("probs_cal produced non-finite values after temperature scaling")
        n = y_arr.shape[0]
        if p_cal.shape[0] != n or n == 0 or (1.0 / float(n + 1) > eff_alpha + 1e-12):
            self.lam_crc = -1.0
            self._is_fitted = True
            return -1.0

        lam_best = -1.0
        # Pointwise non-decreasing upper envelope preserves finite-sample CRC validity across ladder switches.
        env_losses = np.zeros(n, dtype=np.float64)
        for lam in np.linspace(0.0, 1.0, 501):
            lam_f = float(lam)
            acts = self._apply_policy(p_cal, lam_f)
            if self.loss_mode == "indicator":
                losses = (self.cost_matrix[acts, y_arr] > self.c_esc).astype(np.float64)
            else:
                losses = self.L[acts, y_arr]
            env_losses = np.maximum(env_losses, losses)
            crc_ub = (float(np.sum(env_losses)) + 1.0) / float(n + 1)
            if crc_ub <= eff_alpha + 1e-12:
                lam_best = lam_f

        self.lam_crc = lam_best
        self._is_fitted = True
        return lam_best

    def fit(
        self,
        probs: np.ndarray,
        y_true: np.ndarray,
        *,
        cal_probs: Optional[np.ndarray] = None,
        cal_y: Optional[np.ndarray] = None,
    ) -> "UtilityCalibrator":
        """Fit optional temperature scaling and finite-sample CRC risk threshold."""
        if (cal_probs is None) != (cal_y is None):
            raise ValueError("cal_probs and cal_y must both be provided or both be None")
        p_train = _coerce_real_numeric_array(probs, name="probs")
        if p_train.ndim != 2 or p_train.shape[1] != self.n_classes:
            raise ValueError(f"probs must have shape (N, {self.n_classes}), got {p_train.shape}")
        y_train = self._validate_cal_labels(y_true, p_train.shape[0])
        if p_train.shape[0] != y_train.shape[0] or y_train.size == 0:
            raise ValueError("probs and y_true must be non-empty and match in length")
        if not bool(np.all(self._valid_prob_rows(p_train))):
            raise ValueError("probs contains non-finite or invalid probability rows")

        if cal_probs is not None and cal_y is not None:
            p_crc = _coerce_real_numeric_array(cal_probs, name="cal_probs")
            if p_crc.ndim != 2 or p_crc.shape[1] != self.n_classes:
                raise ValueError(f"cal_probs must have shape (N, {self.n_classes}), got {p_crc.shape}")
            y_crc = self._validate_cal_labels(cal_y, p_crc.shape[0])
            if p_crc.shape[0] > 0 and not bool(np.all(self._valid_prob_rows(p_crc))):
                raise ValueError("cal_probs contains non-finite or invalid probability rows")
            if self.fit_temperature and y_train.size >= 8:
                self.temperature = self._fit_temperature(p_train, y_train)
        elif self.fit_temperature and y_train.size >= 24:
            # Keep temperature scaling and CRC holdout samples strictly disjoint.
            n_temp = max(8, min(y_train.size // 3, 128))
            self.temperature = self._fit_temperature(p_train[:n_temp], y_train[:n_temp])
            p_crc = p_train[n_temp:]
            y_crc = y_train[n_temp:]
        else:
            p_crc = p_train
            y_crc = y_train

        if p_crc.shape[0] != y_crc.shape[0] or y_crc.size == 0:
            self.lam_crc = -1.0
            self._is_fitted = True
            return self
        self.fit_crc(p_crc, y_crc, alpha=self.alpha)
        return self

    def calibrate(
        self,
        probs: np.ndarray,
        y_true: np.ndarray,
    ) -> "UtilityCalibrator":
        """Alias for ``fit`` matching calibrator conventions."""
        return self.fit(probs, y_true)

    def predict(
        self,
        probs: np.ndarray,
        *,
        use_crc: bool = True,
        apply_temperature: bool = True,
    ) -> np.ndarray:
        """Select risk-calibrated actions of shape ``(N,)``."""
        p = _coerce_real_numeric_array(probs, name="probs")
        single = p.ndim == 1
        p_in = p.reshape(1, -1) if single else p
        if apply_temperature:
            p2 = self.transform_probs(p_in)
        else:
            if p_in.ndim != 2 or p_in.shape[1] != self.n_classes:
                raise ValueError(f"Expected {self.n_classes} classes, got shape {p.shape}")
            p2 = p_in
        if not use_crc:
            n = p2.shape[0]
            out = np.full(n, self.safe_action, dtype=np.int64)
            valid = self._valid_prob_rows(p2)
            if bool(np.any(valid)):
                v_idx = np.flatnonzero(valid)
                with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                    p_norm = self._normalize_prob_rows(p2[v_idx])
                    costs_v = self._expected_policy_costs(p_norm)
                c_fin = np.all(np.isfinite(costs_v), axis=1)
                if bool(np.any(c_fin)):
                    out[v_idx[c_fin]] = self._select_bayes_actions(p_norm[c_fin], costs_v[c_fin])
            return out[:1] if single else out
        lam = -1.0 if self.is_vacuous else self.lam_crc
        out = self._apply_policy(p2, lam)
        return out[:1] if single else out


def _contains_bool_element(obj: Any) -> bool:
    if isinstance(obj, (bool, np.bool_)):
        return True
    if isinstance(obj, np.ndarray):
        if obj.dtype == np.dtype(np.bool_):
            return True
        if obj.dtype == np.object_:
            return any(_contains_bool_element(v) for v in obj.flat)
        return False
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray, memoryview)):
        return any(_contains_bool_element(v) for v in obj)
    return False


def _coerce_real_numeric_array(arr: Any, *, name: str) -> np.ndarray:
    """Convert numeric array-like to float64, rejecting booleans and non-real dtypes."""
    if isinstance(arr, (str, bytes, bytearray, memoryview, Mapping, set, frozenset)):
        raise ValueError(f"{name} must have real numeric dtype, got {type(arr).__name__}")
    if _contains_bool_element(arr):
        raise ValueError(f"{name} must have real numeric dtype, not boolean")
    raw = np.asarray(arr)
    if raw.dtype == np.dtype(np.bool_):
        raise ValueError(f"{name} must have real numeric dtype, got {raw.dtype}")
    if raw.dtype == np.dtype(np.object_):
        vals: List[float] = []
        for v in raw.flat:
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
                raise ValueError(
                    f"{name} must have real numeric dtype, got element of type {type(v).__name__}"
                )
            try:
                vals.append(float(v))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} contains non-convertible numeric value {v!r}") from exc
        return np.asarray(vals, dtype=np.float64).reshape(raw.shape)
    if raw.dtype.kind not in ("i", "u", "f"):
        raise ValueError(f"{name} must have real numeric dtype, got {raw.dtype}")
    return np.asarray(raw, dtype=np.float64)


def _validate_ordinal_class_indices(y: Any, *, name: str) -> np.ndarray:
    """Validate 1D ordinal class labels strictly without lossy float/object truncation."""
    if isinstance(y, (str, bytes, bytearray, Mapping, set, frozenset)):
        raise ValueError(f"{name} must be a 1D array of integer class indices")
    if not isinstance(y, np.ndarray) and _contains_bool_element(y):
        raise ValueError(f"{name} must contain integer class indices, not booleans")
    raw = np.asarray(y)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a 1D array of integer class indices")
    if raw.dtype == np.dtype(np.object_):
        out_vals: List[int] = []
        for v in raw.flat:
            if isinstance(v, (bool, np.bool_)):
                raise ValueError(f"{name} must contain integer class indices, not booleans")
            if isinstance(v, (int, np.integer)):
                iv = int(v)
                if iv < -9223372036854775808 or iv > 9223372036854775807:
                    raise ValueError(f"{name} integer index out of int64 bounds: {iv}")
                out_vals.append(iv)
            elif isinstance(v, (float, np.floating)):
                fv = float(v)
                if (
                    not math.isfinite(fv)
                    or fv != math.floor(fv)
                    or fv < -9223372036854775808.0
                    or fv >= 9223372036854775808.0
                ):
                    raise ValueError(f"{name} must contain exact integer class indices, got {v!r}")
                out_vals.append(int(fv))
            else:
                raise ValueError(
                    f"{name} must contain integer class indices, got {type(v).__name__}"
                )
        return np.asarray(out_vals, dtype=np.int64).ravel()
    if np.issubdtype(raw.dtype, np.floating):
        if raw.size > 0 and (
            not bool(np.all(np.isfinite(raw)))
            or not bool(np.all(raw == np.floor(raw)))
            or not bool(
                np.all((raw >= -9223372036854775808.0) & (raw < 9223372036854775808.0))
            )
        ):
            raise ValueError(f"{name} must contain exact integer class indices within int64 bounds")
        return np.asarray(raw, dtype=np.int64).ravel()
    if raw.dtype.kind in ("i", "u"):
        if (
            raw.dtype == np.dtype(np.uint64)
            and raw.size > 0
            and bool(np.any(raw > np.uint64(np.iinfo(np.int64).max)))
        ):
            raise ValueError(f"{name} integer indices exceed int64 bounds")
        return np.asarray(raw, dtype=np.int64).ravel()
    raise ValueError(f"{name} must have integer or exact-integral dtype, got {raw.dtype}")


def _validate_nonneg_finite_scalar(value: Any, *, name: str) -> float:
    """Validate a finite non-negative real scalar parameter (rejecting booleans)."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite non-negative number, got bool")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite non-negative number, got {type(value).__name__}")
    val = float(value)
    if not math.isfinite(val) or val < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return val


def _validate_int_scalar(value: Any, *, name: str, min_val: int) -> int:
    """Validate a finite integer scalar parameter (rejecting booleans and non-integral floats)."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, got bool")
    if isinstance(value, (int, np.integer)):
        ival = int(value)
    elif isinstance(value, (float, np.floating)):
        fv = float(value)
        if not math.isfinite(fv) or fv != math.floor(fv):
            raise ValueError(f"{name} must be a finite integer, got {value!r}")
        ival = int(fv)
    else:
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    if ival < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {ival}")
    return ival


class CumulativeLogitOrdinalHead:
    """Proportional-odds cumulative logit model trained with pure-NumPy L-BFGS."""

    def __init__(
        self,
        n_classes: Optional[int] = None,
        *,
        l2: float = 1e-2,
        max_iter: int = 300,
    ) -> None:
        self.n_classes = _validate_int_scalar(n_classes, name="n_classes", min_val=2) if n_classes is not None else None
        self.l2 = _validate_nonneg_finite_scalar(l2, name="l2")
        self.max_iter = max(10, _validate_int_scalar(max_iter, name="max_iter", min_val=1))
        self.weights_: Optional[np.ndarray] = None
        self.thresholds_: Optional[np.ndarray] = None

    @staticmethod
    def _unpack_thresholds(delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        k_minus_1 = delta.size
        thetas = np.empty(k_minus_1, dtype=np.float64)
        slopes = np.ones(k_minus_1, dtype=np.float64)
        thetas[0] = delta[0]
        for j in range(1, k_minus_1):
            g = float(np.clip(delta[j], -30.0, 30.0))
            step = math.log1p(math.exp(g)) if g < 20.0 else g
            thetas[j] = thetas[j - 1] + step + 1e-4
            slopes[j] = float(_expit_np(g))
        return thetas, slopes

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CumulativeLogitOrdinalHead":
        X_arr = _coerce_real_numeric_array(X, name="X")
        y_arr = _validate_ordinal_class_indices(y, name="y")
        if (
            X_arr.ndim != 2
            or X_arr.shape[0] != y_arr.shape[0]
            or y_arr.size == 0
            or not bool(np.all(np.isfinite(X_arr)))
        ):
            raise ValueError("X must be finite 2D and match non-empty 1D y")
        n, d = X_arr.shape
        k = int(self.n_classes) if self.n_classes is not None else int(np.max(y_arr)) + 1
        if k < 2:
            raise ValueError(f"Ordinal head requires at least 2 classes, got {k}")
        if int(np.min(y_arr)) < 0 or int(np.max(y_arr)) >= k:
            raise ValueError(f"Labels must lie in [0, {k - 1}]")
        self.n_classes = k

        y_le_idx = y_arr.copy()
        delta0 = np.zeros(k - 1, dtype=np.float64)
        delta0[0] = -0.5 * float(k - 2) if k > 2 else 0.0
        for j in range(1, k - 1):
            delta0[j] = 0.0
        x0 = np.concatenate([np.zeros(d, dtype=np.float64), delta0])

        def _obj(params: np.ndarray) -> Tuple[float, np.ndarray]:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                if not bool(np.all(np.isfinite(params))):
                    return float("inf"), np.full(params.shape, np.nan, dtype=np.float64)
                w = params[:d]
                delta = params[d:]
                thetas, slopes = self._unpack_thresholds(delta)
                if not bool(np.all(np.isfinite(thetas))) or not bool(np.all(np.isfinite(slopes))):
                    return float("inf"), np.full(params.shape, np.nan, dtype=np.float64)
                eta = X_arr @ w
                if not bool(np.all(np.isfinite(eta))):
                    return float("inf"), np.full(params.shape, np.nan, dtype=np.float64)

                f_hi = np.ones(n, dtype=np.float64)
                mask_hi = y_le_idx < (k - 1)
                if np.any(mask_hi):
                    diff_hi = thetas[y_le_idx[mask_hi]] - eta[mask_hi]
                    if not bool(np.all(np.isfinite(diff_hi))):
                        return float("inf"), np.full(params.shape, np.nan, dtype=np.float64)
                    f_hi[mask_hi] = _expit_np(diff_hi)

                f_lo = np.zeros(n, dtype=np.float64)
                mask_lo = y_le_idx > 0
                if np.any(mask_lo):
                    diff_lo = thetas[y_le_idx[mask_lo] - 1] - eta[mask_lo]
                    if not bool(np.all(np.isfinite(diff_lo))):
                        return float("inf"), np.full(params.shape, np.nan, dtype=np.float64)
                    f_lo[mask_lo] = _expit_np(diff_lo)

                probs = np.maximum(f_hi - f_lo, 1e-12)
                reg_term = 0.5 * self.l2 * float(np.sum(w * w))
                loss = -float(np.mean(np.log(probs))) + reg_term
                if not math.isfinite(loss):
                    return float("inf"), np.full(params.shape, np.nan, dtype=np.float64)

                s_hi = np.where(mask_hi, f_hi * (1.0 - f_hi), 0.0)
                s_lo = np.where(mask_lo, f_lo * (1.0 - f_lo), 0.0)
                inv_pn = 1.0 / (probs * float(n))

                d_eta = (s_hi - s_lo) * inv_pn
                grad_w = X_arr.T @ d_eta + self.l2 * w

                grad_theta = np.zeros(k - 1, dtype=np.float64)
                if np.any(mask_hi):
                    np.add.at(grad_theta, y_le_idx[mask_hi], -s_hi[mask_hi] * inv_pn[mask_hi])
                if np.any(mask_lo):
                    np.add.at(grad_theta, y_le_idx[mask_lo] - 1, s_lo[mask_lo] * inv_pn[mask_lo])

                suffix_sum = np.cumsum(grad_theta[::-1])[::-1]
                grad_delta = suffix_sum.copy()
                grad_delta[1:] = suffix_sum[1:] * slopes[1:]
                grad = np.concatenate([grad_w, grad_delta])
                if not bool(np.all(np.isfinite(grad))):
                    return float("inf"), np.full(params.shape, np.nan, dtype=np.float64)
                return loss, grad

        f0, g0 = _obj(x0)
        if not math.isfinite(f0) or not bool(np.all(np.isfinite(g0))):
            raise ValueError("Ordinal optimization failed due to non-finite initial objective or gradient")
        opt, _, _ = _minimize_lbfgs_np(_obj, x0, maxiter=self.max_iter)
        f_fin, g_fin = _obj(opt)
        g0_inf = float(np.max(np.abs(g0))) if g0.size > 0 else 0.0
        thetas_fin, _ = self._unpack_thresholds(opt[d:]) if bool(np.all(np.isfinite(opt))) else (np.array([np.nan]), np.array([]))
        if (
            not bool(np.all(np.isfinite(opt)))
            or not math.isfinite(f_fin)
            or not bool(np.all(np.isfinite(g_fin)))
            or not bool(np.all(np.isfinite(thetas_fin)))
        ):
            raise ValueError("Ordinal optimization failed to produce finite parameters")
        if g0_inf > 1e-6 and (np.array_equal(opt, x0) or not (f_fin < f0)):
            raise ValueError("Ordinal optimization failed to converge due to numerical overflow")
        self.weights_ = opt[:d].copy()
        self.thresholds_ = thetas_fin
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None or self.thresholds_ is None or self.n_classes is None:
            raise RuntimeError("CumulativeLogitOrdinalHead must be fit before calling predict_proba")
        X_arr = _coerce_real_numeric_array(X, name="X")
        single = X_arr.ndim == 1
        X2 = X_arr.reshape(1, -1) if single else X_arr
        w = np.asarray(self.weights_, dtype=np.float64)
        thetas = np.asarray(self.thresholds_, dtype=np.float64).reshape(-1)
        k = int(self.n_classes)
        if X2.ndim != 2 or w.ndim != 1 or X2.shape[1] != w.shape[0] or thetas.shape != (k - 1,):
            raise ValueError(
                f"Input/model shape mismatch in CumulativeLogitOrdinalHead: X={X2.shape}, w={w.shape}, thresholds={thetas.shape}"
            )
        n = X2.shape[0]
        probs = np.full((n, k), np.nan, dtype=np.float64)
        params_ok = (
            bool(np.all(np.isfinite(w)))
            and bool(np.all(np.isfinite(thetas)))
            and (thetas.size <= 1 or bool(np.all(thetas[1:] >= (thetas[:-1] - 1e-12))))
        )
        if params_ok and n > 0:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                eta = X2 @ w
                diff = thetas[None, :] - eta[:, None]
                row_ok = (
                    np.all(np.isfinite(X2), axis=1)
                    & np.isfinite(eta)
                    & np.all(np.isfinite(diff), axis=1)
                )
                if bool(np.any(row_ok)):
                    cum = np.ones((int(np.sum(row_ok)), k), dtype=np.float64)
                    cum[:, : k - 1] = _expit_np(diff[row_ok])
                    p_ok = np.empty((cum.shape[0], k), dtype=np.float64)
                    p_ok[:, 0] = cum[:, 0]
                    if k > 1:
                        p_ok[:, 1:] = cum[:, 1:] - cum[:, :-1]
                    p_ok = np.clip(p_ok, 1e-12, 1.0)
                    p_ok /= np.sum(p_ok, axis=1, keepdims=True)
                    p_fin = np.all(np.isfinite(p_ok), axis=1)
                    good_idx = np.flatnonzero(row_ok)[p_fin]
                    if good_idx.size > 0:
                        probs[good_idx] = p_ok[p_fin]
        return probs[0] if single else probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _predict_ordinal_class(self.predict_proba(X))

    def expected_value(self, X: np.ndarray) -> np.ndarray:
        return _ordinal_expected_value(self.predict_proba(X), self.n_classes)


class SoftOrdinalHead:
    """Gaussian/Laplace soft-target linear softmax ordinal head in pure NumPy."""

    def __init__(
        self,
        n_classes: Optional[int] = None,
        *,
        sigma: float = 0.5,
        l2: float = 1e-2,
        max_iter: int = 300,
    ) -> None:
        self.n_classes = _validate_int_scalar(n_classes, name="n_classes", min_val=2) if n_classes is not None else None
        self.sigma = _validate_nonneg_finite_scalar(sigma, name="sigma")
        self.l2 = _validate_nonneg_finite_scalar(l2, name="l2")
        self.max_iter = max(10, _validate_int_scalar(max_iter, name="max_iter", min_val=1))
        self.weights_: Optional[np.ndarray] = None
        self.bias_: Optional[np.ndarray] = None

    @staticmethod
    def build_soft_targets(y: np.ndarray, n_classes: int, sigma: float) -> np.ndarray:
        k = _validate_int_scalar(n_classes, name="n_classes", min_val=2)
        sig_f = _validate_nonneg_finite_scalar(sigma, name="sigma")
        y_idx = _validate_ordinal_class_indices(y, name="y")
        n = y_idx.shape[0]
        if n == 0:
            raise ValueError("y must be non-empty")
        if int(np.min(y_idx)) < 0 or int(np.max(y_idx)) >= k:
            raise ValueError(f"Labels must lie in [0, {k - 1}]")
        if sig_f <= 0.0:
            q = np.zeros((n, k), dtype=np.float64)
            q[np.arange(n), y_idx] = 1.0
            return q
        ks = np.arange(k, dtype=np.float64)[None, :]
        diff = ks - y_idx[:, None].astype(np.float64)
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            scaled = diff / sig_f
            logits = -0.5 * (scaled * scaled)
            logits[np.arange(n), y_idx] = 0.0
            exps = np.exp(logits)
            denom = np.sum(exps, axis=1, keepdims=True)
            out = exps / denom
        if not bool(np.all(np.isfinite(out))):
            q = np.zeros((n, k), dtype=np.float64)
            q[np.arange(n), y_idx] = 1.0
            return q
        return out

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftOrdinalHead":
        X_arr = _coerce_real_numeric_array(X, name="X")
        y_arr = _validate_ordinal_class_indices(y, name="y")
        if (
            X_arr.ndim != 2
            or X_arr.shape[0] != y_arr.shape[0]
            or y_arr.size == 0
            or not bool(np.all(np.isfinite(X_arr)))
        ):
            raise ValueError("X must be finite 2D and match non-empty 1D y")
        n, d = X_arr.shape
        k = int(self.n_classes) if self.n_classes is not None else int(np.max(y_arr)) + 1
        if k < 2:
            raise ValueError(f"SoftOrdinalHead requires at least 2 classes, got {k}")
        if int(np.min(y_arr)) < 0 or int(np.max(y_arr)) >= k:
            raise ValueError(f"Labels must lie in [0, {k - 1}]")
        self.n_classes = k

        q = self.build_soft_targets(y_arr, k, self.sigma)
        w_len = d * k
        x0 = np.zeros(w_len + k, dtype=np.float64)

        def _obj(params: np.ndarray) -> Tuple[float, np.ndarray]:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                if not bool(np.all(np.isfinite(params))):
                    return float("inf"), np.full(params.shape, np.nan, dtype=np.float64)
                w = params[:w_len].reshape(d, k)
                b = params[w_len:]
                logits = X_arr @ w + b
                if not bool(np.all(np.isfinite(logits))):
                    return float("inf"), np.full(params.shape, np.nan, dtype=np.float64)
                shifted = logits - np.max(logits, axis=1, keepdims=True)
                if not bool(np.all(np.isfinite(shifted))):
                    return float("inf"), np.full(params.shape, np.nan, dtype=np.float64)
                exps = np.exp(np.clip(shifted, -60.0, 60.0))
                denom = np.maximum(np.sum(exps, axis=1, keepdims=True), 1e-12)
                log_p = shifted - np.log(denom)
                p = exps / denom
                reg_term = 0.5 * self.l2 * float(np.sum(w * w))
                loss = -float(np.mean(np.sum(q * log_p, axis=1))) + reg_term
                if not math.isfinite(loss):
                    return float("inf"), np.full(params.shape, np.nan, dtype=np.float64)
                dz = (p - q) / float(n)
                gw = (X_arr.T @ dz) + self.l2 * w
                gb = np.sum(dz, axis=0)
                grad = np.concatenate([gw.ravel(), gb])
                if not bool(np.all(np.isfinite(grad))):
                    return float("inf"), np.full(params.shape, np.nan, dtype=np.float64)
                return loss, grad

        f0, g0 = _obj(x0)
        if not math.isfinite(f0) or not bool(np.all(np.isfinite(g0))):
            raise ValueError("Ordinal optimization failed due to non-finite initial objective or gradient")
        opt, _, _ = _minimize_lbfgs_np(_obj, x0, maxiter=self.max_iter)
        f_fin, g_fin = _obj(opt)
        g0_inf = float(np.max(np.abs(g0))) if g0.size > 0 else 0.0
        if (
            not bool(np.all(np.isfinite(opt)))
            or not math.isfinite(f_fin)
            or not bool(np.all(np.isfinite(g_fin)))
        ):
            raise ValueError("Ordinal optimization failed to produce finite parameters")
        if g0_inf > 1e-6 and (np.array_equal(opt, x0) or not (f_fin < f0)):
            raise ValueError("Ordinal optimization failed to converge due to numerical overflow")
        self.weights_ = opt[:w_len].reshape(d, k).copy()
        self.bias_ = opt[w_len:].copy()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None or self.bias_ is None:
            raise RuntimeError("SoftOrdinalHead must be fit before calling predict_proba")
        X_arr = _coerce_real_numeric_array(X, name="X")
        single = X_arr.ndim == 1
        X2 = X_arr.reshape(1, -1) if single else X_arr
        w = np.asarray(self.weights_, dtype=np.float64)
        b = np.asarray(self.bias_, dtype=np.float64).reshape(-1)
        if X2.ndim != 2 or w.ndim != 2 or X2.shape[1] != w.shape[0] or b.shape != (w.shape[1],):
            raise ValueError(
                f"Input/model shape mismatch in SoftOrdinalHead: X={X2.shape}, w={w.shape}, b={b.shape}"
            )
        k = w.shape[1]
        probs = np.full((X2.shape[0], k), np.nan, dtype=np.float64)
        if bool(np.all(np.isfinite(w))) and bool(np.all(np.isfinite(b))) and X2.shape[0] > 0:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                lin = X2 @ w
                logits = lin + b
                row_ok = (
                    np.all(np.isfinite(X2), axis=1)
                    & np.all(np.isfinite(lin), axis=1)
                    & np.all(np.isfinite(logits), axis=1)
                )
                if bool(np.any(row_ok)):
                    p_ok = _softmax_np(logits[row_ok], axis=1)
                    p_fin = np.all(np.isfinite(p_ok), axis=1)
                    good_idx = np.flatnonzero(row_ok)[p_fin]
                    if good_idx.size > 0:
                        probs[good_idx] = p_ok[p_fin]
        return probs[0] if single else probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _predict_ordinal_class(self.predict_proba(X))

    def expected_value(self, X: np.ndarray) -> np.ndarray:
        return _ordinal_expected_value(self.predict_proba(X), self.n_classes)


def _validate_ordinal_prob_matrix(
    probs: np.ndarray, *, name: str = "probs", allow_1d: bool = False
) -> np.ndarray:
    arr = _coerce_real_numeric_array(probs, name=name).copy()
    was_1d = bool(allow_1d and arr.ndim == 1)
    if was_1d:
        check = arr.reshape(1, -1)
    elif arr.ndim == 2:
        check = arr
    else:
        shape_desc = "1D or 2D" if allow_1d else "2D"
        raise ValueError(f"{name} must be a {shape_desc} array, got shape {arr.shape}")
    if check.shape[1] < 1:
        raise ValueError(f"{name} must have at least 1 class column, got shape {arr.shape}")
    if check.shape[0] > 0:
        if not bool(np.all(np.isfinite(check))):
            raise ValueError(f"{name} contains non-finite values")
        if bool(np.any((check < -1e-12) | (check > 1.0 + 1e-12))):
            raise ValueError(f"{name} entries must lie in [0.0, 1.0]")
        row_sums = np.sum(check, axis=1)
        if not bool(np.all(np.isfinite(row_sums))) or bool(np.any(np.abs(row_sums - 1.0) > 1e-4)):
            raise ValueError(f"{name} rows must sum to 1.0")
        check = np.clip(check, 0.0, 1.0)
        norm_sums = np.sum(check, axis=1, keepdims=True)
        if bool(np.any(norm_sums <= 0.0)):
            raise ValueError(f"{name} rows must have positive sum")
        check = check / norm_sums
    return check[0] if was_1d else check


def _predict_ordinal_class(proba: np.ndarray) -> np.ndarray:
    p = _validate_ordinal_prob_matrix(proba, name="ordinal probabilities", allow_1d=True)
    if p.ndim == 1:
        return np.array([int(np.argmax(p))], dtype=np.int64)
    return np.argmax(p, axis=1).astype(np.int64)


def _ordinal_expected_value(proba: np.ndarray, n_classes: Optional[int]) -> np.ndarray:
    p = _validate_ordinal_prob_matrix(proba, name="ordinal probabilities", allow_1d=True)
    classes = np.arange(n_classes or p.shape[-1], dtype=np.float64)
    return p @ classes


def _cdf_quantiles_from_validated(p: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(alpha, (bool, np.bool_)) or not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    cdf = np.cumsum(p, axis=1)
    mask_lo = cdf >= (0.5 * float(alpha)) - 1e-12
    mask_hi = cdf >= (1.0 - 0.5 * float(alpha)) - 1e-12
    has_lo = np.any(mask_lo, axis=1)
    has_hi = np.any(mask_hi, axis=1)
    lo = np.where(has_lo, np.argmax(mask_lo, axis=1), 0).astype(np.int64)
    hi = np.where(has_hi, np.argmax(mask_hi, axis=1), p.shape[1] - 1).astype(np.int64)
    hi = np.maximum(hi, lo)
    return lo, hi


def _cdf_ordinal_quantiles(probs: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(alpha, (bool, np.bool_)) or not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    p = _validate_ordinal_prob_matrix(probs, name="probs", allow_1d=False)
    return _cdf_quantiles_from_validated(p, float(alpha))


def cqr_ordinal_intervals(
    cal_probs: np.ndarray,
    cal_y: np.ndarray,
    test_probs: np.ndarray,
    *,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Finite-sample Conformalized Quantile Regression intervals for ordinal labels.

    Returns integer arrays ``(lower, upper)`` with guaranteed finite-sample marginal
    coverage ``>= 1 - alpha`` under exchangeability.
    """
    p_cal = _validate_ordinal_prob_matrix(cal_probs, name="cal_probs", allow_1d=False)
    y_cal = _validate_ordinal_class_indices(cal_y, name="cal_y")
    p_test = _validate_ordinal_prob_matrix(test_probs, name="test_probs", allow_1d=False)
    if p_cal.shape[1] != p_test.shape[1]:
        raise ValueError("cal_probs and test_probs must be 2D arrays with matching n_classes")
    if p_cal.shape[0] != y_cal.shape[0] or y_cal.size == 0:
        raise ValueError("cal_probs and cal_y must be non-empty and match in length")
    if bool(np.any((y_cal < 0) | (y_cal >= p_cal.shape[1]))):
        raise ValueError(f"cal_y indices must lie in [0, {p_cal.shape[1] - 1}]")
    if isinstance(alpha, (bool, np.bool_)) or not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")

    k = p_cal.shape[1]
    n = y_cal.shape[0]
    cal_lo, cal_hi = _cdf_quantiles_from_validated(p_cal, float(alpha))
    scores = np.maximum(cal_lo - y_cal, y_cal - cal_hi).astype(np.float64)

    level = math.ceil((n + 1) * (1.0 - float(alpha))) / float(n)
    if level > 1.0:
        lo_out = np.zeros(p_test.shape[0], dtype=np.int64)
        hi_out = np.full(p_test.shape[0], k - 1, dtype=np.int64)
        return lo_out, hi_out

    q_idx = min(n, max(1, int(math.ceil((n + 1) * (1.0 - float(alpha)))))) - 1
    q_hat = int(round(float(np.sort(scores)[q_idx])))

    test_lo, test_hi = _cdf_quantiles_from_validated(p_test, float(alpha))
    lo_out = np.clip(test_lo - q_hat, 0, k - 1).astype(np.int64)
    hi_out = np.clip(test_hi + q_hat, 0, k - 1).astype(np.int64)
    mid = np.clip(np.round(0.5 * (test_lo + test_hi)).astype(np.int64), 0, k - 1)
    empty = lo_out > hi_out
    if np.any(empty):
        lo_out = np.where(empty, mid, lo_out)
        hi_out = np.where(empty, mid, hi_out)
    return lo_out, hi_out


class BinaryPlattBetaCalibrator:
    """Adaptive Platt / Beta binary probability calibrator in pure NumPy.

    Returns identity calibration when ``n_cal < min_platt_samples`` (default 74),
    fits 2-parameter Platt scaling for ``min_platt_samples <= n_cal < min_beta_samples``,
    and fits 3-parameter Beta calibration when ``n_cal >= min_beta_samples``.
    """

    def __init__(
        self,
        *,
        min_platt_samples: int = 74,
        min_beta_samples: int = 150,
        reg: float = 1e-3,
        max_iter: int = 200,
    ) -> None:
        self.min_platt_samples = max(1, int(min_platt_samples))
        self.min_beta_samples = max(1, int(min_beta_samples))
        self.reg = max(0.0, float(reg))
        self.max_iter = max(10, int(max_iter))
        self.mode: str = "unfitted"
        self.n_cal: int = 0
        self.params_: Optional[np.ndarray] = None

    @staticmethod
    def _extract_positive_probs(probs: np.ndarray) -> np.ndarray:
        arr = _coerce_real_numeric_array(probs, name="probs")
        if arr.ndim == 2:
            if arr.shape[1] != 2:
                raise ValueError(f"Expected binary (N, 2) probabilities, got {arr.shape}")
            p1 = arr[:, 1]
        elif arr.ndim == 1:
            p1 = arr
        else:
            raise ValueError(f"Expected 1D or 2D probabilities, got ndim={arr.ndim}")
        if not bool(np.all(np.isfinite(arr))):
            raise ValueError("BinaryPlattBetaCalibrator received non-finite probabilities")
        if arr.size > 0 and (float(np.min(arr)) < -1e-9 or float(np.max(arr)) > 1.0 + 1e-9):
            raise ValueError("BinaryPlattBetaCalibrator probabilities must lie in [0, 1]")
        return np.where(p1 == 0.0, 0.0, np.clip(p1, 0.0, 1.0))

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "BinaryPlattBetaCalibrator":
        p1 = self._extract_positive_probs(probs)
        y_raw = _coerce_real_numeric_array(y_true, name="y_true")
        if y_raw.ndim != 1:
            raise ValueError(f"y_true must be 1D, got shape {y_raw.shape}")
        y = y_raw.ravel()
        if p1.shape[0] != y.shape[0] or y.size == 0:
            raise ValueError("probs and y_true must be non-empty and match in length")
        if np.any((y != 0.0) & (y != 1.0)):
            raise ValueError("BinaryPlattBetaCalibrator requires binary {0, 1} labels")

        n = y.shape[0]
        self.n_cal = int(n)
        if n < self.min_platt_samples:
            self.mode = "identity"
            self.params_ = np.array([1.0, 0.0], dtype=np.float64)
            return self
        p1_eps = np.clip(p1, 1e-6, 1.0 - 1e-6)
        if n < self.min_beta_samples:
            self.mode = "platt"
            z = np.log(p1_eps / (1.0 - p1_eps))

            def _obj_platt(params: np.ndarray) -> Tuple[float, np.ndarray]:
                w, c = float(params[0]), float(params[1])
                eta = np.clip(w * z + c, -60.0, 60.0)
                q = _expit_np(eta)
                bce = float(np.mean(np.logaddexp(0.0, eta) - y * eta))
                loss = bce + 0.5 * self.reg * ((w - 1.0) ** 2 + c ** 2)
                diff = (q - y) / float(n)
                dw = float(np.dot(diff, z)) + self.reg * (w - 1.0)
                dc = float(np.sum(diff)) + self.reg * c
                return loss, np.array([dw, dc], dtype=np.float64)

            opt, _, _ = _minimize_lbfgs_np(
                _obj_platt,
                np.array([1.0, 0.0], dtype=np.float64),
                bounds=[(0.0, 20.0), (-20.0, 20.0)],
                maxiter=self.max_iter,
            )
            self.params_ = opt.copy()
        else:
            self.mode = "beta"
            x_pos = np.log(p1_eps)
            x_neg = -np.log(1.0 - p1_eps)

            def _obj_beta(params: np.ndarray) -> Tuple[float, np.ndarray]:
                a, b, c = float(params[0]), float(params[1]), float(params[2])
                eta = np.clip(a * x_pos + b * x_neg + c, -60.0, 60.0)
                q = _expit_np(eta)
                bce = float(np.mean(np.logaddexp(0.0, eta) - y * eta))
                loss = bce + 0.5 * self.reg * ((a - 1.0) ** 2 + (b - 1.0) ** 2 + c ** 2)
                diff = (q - y) / float(n)
                da = float(np.dot(diff, x_pos)) + self.reg * (a - 1.0)
                db = float(np.dot(diff, x_neg)) + self.reg * (b - 1.0)
                dc = float(np.sum(diff)) + self.reg * c
                return loss, np.array([da, db, dc], dtype=np.float64)

            opt, _, _ = _minimize_lbfgs_np(
                _obj_beta,
                np.array([1.0, 1.0, 0.0], dtype=np.float64),
                bounds=[(0.0, 20.0), (0.0, 20.0), (-20.0, 20.0)],
                maxiter=self.max_iter,
            )
            self.params_ = opt.copy()
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Return calibrated positive-class probability vector of shape ``(N,)``."""
        if self.params_ is None or self.mode == "unfitted":
            raise RuntimeError("BinaryPlattBetaCalibrator must be fit before transforming")
        p1 = self._extract_positive_probs(probs)
        if self.mode == "identity":
            return p1
        p1_eps = np.clip(p1, 1e-6, 1.0 - 1e-6)
        if self.mode == "platt":
            w, c = float(self.params_[0]), float(self.params_[1])
            z = np.log(p1_eps / (1.0 - p1_eps))
            out = _expit_np(w * z + c)
        else:
            a, b, c = float(self.params_[0]), float(self.params_[1]), float(self.params_[2])
            out = _expit_np(a * np.log(p1_eps) - b * np.log(1.0 - p1_eps) + c)
        return np.clip(np.asarray(out, dtype=np.float64), 1e-6, 1.0 - 1e-6)

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        """Return calibrated two-column probability matrix ``[1 - p, p]``."""
        p_cal = self.transform(probs)
        return np.stack([1.0 - p_cal, p_cal], axis=1)


__all__ = [
    "AdaptiveConformalTracker",
    "BinaryPlattBetaCalibrator",
    "BrierDecomposition",
    "CalibrationMetrics",
    "ConformalPredictionSet",
    "ConformalPredictor",
    "CumulativeLogitOrdinalHead",
    "DecisionCalibrator",
    "JointSchemaCalibrator",
    "JointSelectiveCertificate",
    "OnlineMondrianConformal",
    "RecalibrationMetrics",
    "RecalibrationReport",
    "RegressionConformalInterval",
    "RegressionConformalPredictor",
    "SelectiveRiskCalibrator",
    "SelectiveRiskCertificate",
    "SequentialMonitorStep",
    "SequentialRiskMonitor",
    "SoftOrdinalHead",
    "UtilityCalibrator",
    "VALID_MONITOR_MODES",
    "clopper_pearson_upper",
    "compute_brier_decomposition",
    "compute_ece_and_bins",
    "compute_nll",
    "cqr_ordinal_intervals",
    "should_use_soft_ordinal",
]
