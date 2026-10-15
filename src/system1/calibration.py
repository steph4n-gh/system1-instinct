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

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np


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
        """Applies learned temperature to raw logits to yield calibrated probabilities."""
        logits = np.asarray(logits, dtype=np.float64)
        if self.is_binary:
            if logits.ndim == 0:
                logits = np.array([[-float(logits) / 2.0, float(logits) / 2.0]], dtype=np.float64)
                scaled = logits / self.temperature
                return _stable_softmax(scaled)[0]
            elif logits.ndim == 1:
                logits = np.column_stack([-logits / 2.0, logits / 2.0])
                scaled = logits / self.temperature
                return _stable_softmax(scaled)

        if logits.ndim == 1 and len(logits) == 1:
            z = logits[0]
            logits = np.array([-z / 2.0, z / 2.0], dtype=np.float64)
        scaled = logits / self.temperature
        if scaled.ndim == 1:
            scaled = scaled.reshape(1, -1)
            single = True
        else:
            single = False
        max_val = np.max(scaled, axis=-1, keepdims=True)
        exp = np.exp(scaled - max_val)
        probs = exp / np.sum(exp, axis=-1, keepdims=True)
        return probs[0] if single else probs


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
        return float(w * q_emp + (1.0 - w) * q_global)

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
                elif np.all(np.isfinite(probs)) and np.all(probs >= 0) and np.isclose(total_mass, 1.0):
                    if self.mondrian and self.class_calibration_scores:
                        prediction_set = [
                            self.options[idx]
                            for idx in sorted_indices
                            if 1.0 - float(probs[idx]) <= self.class_quantile(self.options[idx], alpha) + 1e-12
                        ]
                    else:
                        prediction_set = [self.options[idx] for idx in sorted_indices
                                          if 1.0 - float(probs[idx]) <= q_hat + 1e-12]
            elif strict:
                # Invert the calibrated score s(x,y) = cumulative mass through y.
                # Including the next label after crossing q_hat inflates every set.
                # Empty sets request review; insufficient evidence includes all labels.
                if k > n:
                    prediction_set = list(self.options)
                elif np.all(np.isfinite(probs)) and np.all(probs >= 0) and np.isclose(total_mass, 1.0):
                    for idx in sorted_indices:
                        cum_mass += float(probs[idx])
                        opt = self.options[idx]
                        thresh = self.class_quantile(opt, alpha) if (self.mondrian and self.class_calibration_scores) else q_hat
                        if cum_mass <= thresh + 1e-12:
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


__all__ = [
    "BrierDecomposition",
    "CalibrationMetrics",
    "ConformalPredictionSet",
    "ConformalPredictor",
    "DecisionCalibrator",
    "RecalibrationMetrics",
    "RecalibrationReport",
    "RegressionConformalInterval",
    "RegressionConformalPredictor",
    "compute_brier_decomposition",
    "compute_ece_and_bins",
    "compute_nll",
]
