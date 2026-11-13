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

import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np


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
    scaled = np.asarray(logits, dtype=np.float64) / temp
    max_val = np.max(scaled, axis=-1, keepdims=True)
    exp = np.exp(scaled - max_val)
    return exp / np.sum(exp, axis=-1, keepdims=True)


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
        arr = np.asarray(logits, dtype=np.float64)
        temp = self.temperature
        adj = self.logit_adjustments
        if self.is_binary:
            if arr.ndim == 0:
                z = float(arr)
                if adj is not None and adj.shape == (2,):
                    z += float(adj[1] - adj[0])
                half_z = z / (2.0 * temp)
                s_max = half_z if half_z >= -half_z else -half_z
                e0 = math.exp(-half_z - s_max)
                e1 = math.exp(half_z - s_max)
                inv = 1.0 / (e0 + e1)
                return np.array([e0 * inv, e1 * inv], dtype=np.float64)
            if arr.ndim == 1:
                arr = np.column_stack([-arr / 2.0, arr / 2.0])
                if adj is not None and adj.shape == (2,):
                    arr = arr + adj
                return _stable_softmax(arr / temp)

        if arr.ndim == 1:
            if len(arr) == 1:
                z = float(arr[0])
                if adj is not None and adj.shape == (2,):
                    z += float(adj[1] - adj[0])
                half_z = z / (2.0 * temp)
                s_max = half_z if half_z >= -half_z else -half_z
                e0 = math.exp(-half_z - s_max)
                e1 = math.exp(half_z - s_max)
                inv = 1.0 / (e0 + e1)
                return np.array([e0 * inv, e1 * inv], dtype=np.float64)
            if adj is not None and adj.shape == (len(arr),):
                arr = arr + adj
            scaled = arr / temp
            s = scaled - float(np.max(scaled))
            exp = np.exp(s)
            return exp / float(np.sum(exp))

        if adj is not None and adj.shape == (arr.shape[-1],):
            arr = arr + adj
        scaled = arr / temp
        max_val = np.max(scaled, axis=-1, keepdims=True)
        exp = np.exp(scaled - max_val)
        return exp / np.sum(exp, axis=-1, keepdims=True)

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
        self.prototypes: Optional[np.ndarray] = (
            np.asarray(prototypes, dtype=np.float32) if prototypes is not None else None
        )
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
        emb = np.asarray(embedding, dtype=np.float32).flatten()
        if emb.ndim != 1 or self.prototypes.ndim != 2 or self.prototypes.shape[1] != emb.shape[0]:
            return None
        e_norm = float(np.linalg.norm(emb))
        if e_norm < 1e-12 or not math.isfinite(e_norm):
            return 0.0
        emb_unit = emb / e_norm
        p_norms = np.linalg.norm(self.prototypes, axis=1, keepdims=True)
        valid_rows = (p_norms.flatten() > 1e-12)
        if not np.any(valid_rows):
            return None
        proto_unit = self.prototypes[valid_rows] / np.where(p_norms[valid_rows] < 1e-12, 1.0, p_norms[valid_rows])
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
            self.prototypes = np.asarray(prototypes, dtype=np.float32)
        arr = np.asarray(prototype_values, dtype=np.float64)
        if arr.ndim == 2:
            if self.prototypes is None and labels is not None and len(labels) == arr.shape[0] and len(self.options) > 0:
                protos = np.zeros((len(self.options), arr.shape[1]), dtype=np.float32)
                for c_idx, opt in enumerate(self.options):
                    mask = [i for i, lbl in enumerate(labels) if str(lbl) == str(opt)]
                    if mask:
                        c_mean = np.mean(arr[mask], axis=0)
                        c_norm = float(np.linalg.norm(c_mean))
                        if c_norm > 1e-12:
                            protos[c_idx] = (c_mean / c_norm).astype(np.float32)
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
            self.prototype_threshold = float(self.prototype_scores[rank - 1])
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

            sorted_indices = np.argsort(-row)
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
        sorted_indices = np.argsort(-probs)
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
        return f"[{self.lower_bound:.4f}, {self.upper_bound:.4f}]"

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
        res = np.abs(preds - targets)
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
        val_range = self.max_value - self.min_value
        n = len(self.residuals)

        if not self.is_calibrated or n == 0:
            if strict:
                margin = val_range
                low = self.min_value
                high = self.max_value
            else:
                margin = val_range * ((1.0 - alpha) / 2.0)
                low = max(self.min_value, y_hat - margin)
                high = min(self.max_value, y_hat + margin)
        else:
            k = int(math.ceil((n + 1) * (1.0 - alpha)))
            if k > n:
                margin = val_range
                low = self.min_value
                high = self.max_value
            else:
                margin = float(self.residuals[k - 1])
                low = max(self.min_value, y_hat - margin)
                high = min(self.max_value, y_hat + margin)
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
        self._scores: List[float] = []

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
        self._scores.append(float(score))
        if len(self._scores) > self.max_window:
            self._scores = self._scores[-self.max_window :]
        return np.sort(np.asarray(self._scores, dtype=np.float64))

    def rescore_calibration_window(
        self,
        pairs: Sequence[Tuple[str, Any]],
        predictor: Callable[[str, Any], Tuple[float, float, bool]],
        *,
        default_floor: float = 0.30,
    ) -> Tuple[np.ndarray, float]:
        """Re-score a bounded calibration window after online weight updates and ratchet margin_floor."""
        window = list(pairs)[-self.max_window :]
        scores: List[float] = []
        err_margins: List[float] = []
        for prompt_c, label_c in window:
            nc_score, margin_c, is_correct = predictor(prompt_c, label_c)
            scores.append(float(nc_score))
            if not is_correct and math.isfinite(float(margin_c)):
                err_margins.append(float(margin_c))
        self._scores = list(scores)
        new_floor = (max(err_margins) + 1e-4) if err_margins else float(default_floor)
        if self.margin_floor is not None:
            self.margin_floor = max(self.margin_floor, new_floor)
        else:
            self.margin_floor = new_floor
        return np.sort(np.asarray(self._scores, dtype=np.float64)), float(self.margin_floor)

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


__all__ = [
    "AdaptiveConformalTracker",
    "BrierDecomposition",
    "CalibrationMetrics",
    "ConformalPredictionSet",
    "ConformalPredictor",
    "DecisionCalibrator",
    "JointSchemaCalibrator",
    "JointSelectiveCertificate",
    "RecalibrationMetrics",
    "RecalibrationReport",
    "RegressionConformalInterval",
    "RegressionConformalPredictor",
    "SelectiveRiskCalibrator",
    "SelectiveRiskCertificate",
    "SequentialMonitorStep",
    "SequentialRiskMonitor",
    "VALID_MONITOR_MODES",
    "clopper_pearson_upper",
    "compute_brier_decomposition",
    "compute_ece_and_bins",
    "compute_nll",
]
