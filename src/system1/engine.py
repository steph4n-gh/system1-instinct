"""System 1 Decision Engine Runtime.

Orchestrates non-autoregressive single-pass schema evaluation, calibrated
confidence scoring, conformal prediction sets, cryptographic receipt generation,
and ActionLedger audit integration.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import secrets
import statistics
import threading
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from system1.calibration import (
    CalibrationMetrics,
    ConformalPredictionSet,
    ConformalPredictor,
    DecisionCalibrator,
    RecalibrationMetrics,
    RecalibrationReport,
    RegressionConformalInterval,
    RegressionConformalPredictor,
    compute_brier_decomposition,
    compute_ece_and_bins,
    compute_nll,
)
from system1.core import (
    BooleanField,
    ChoiceField,
    DecisionField,
    DecisionSchema,
    MultiChoiceField,
    RawFieldEvaluation,
    ScoreField,
    SystemOneModel,
)
from system1.core.model import _stable_sigmoid
from system1.cache import SemanticSystemOneCache, _validate_cache_inputs
from system1.core.telemetry import TelemetryProjector
from system1.ledger import ActionLedger, LedgerWriteError
from system1.receipt import (
    DecisionWitnessReceipt,
    create_decision_receipt,
    public_key_fingerprint,
)


@dataclass(frozen=True)
class FieldEvaluationMetrics:
    """Per-field accuracy, selective accuracy, abstention/OOD rates, conformal coverage, and calibration metrics."""

    field_name: str
    field_type: str
    total_examples: int
    accepted_examples: int
    abstained_examples: int
    unfamiliar_examples: int
    accuracy: float
    accepted_accuracy: float
    conformal_coverage: float
    average_set_size: float
    singleton_rate: float
    ece: float
    brier_score: float
    nll: float
    coverage: float
    abstain_rate: float
    unfamiliar_rate: float
    abstention_reasons: Dict[str, int] = field(default_factory=dict)
    mae: float = 0.0

    @property
    def n_examples(self) -> int:
        return self.total_examples

    @property
    def selective_accuracy(self) -> float:
        return self.accepted_accuracy

    @property
    def empirical_set_coverage(self) -> float:
        return self.conformal_coverage

    @property
    def mean_set_size(self) -> float:
        return self.average_set_size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "field_type": self.field_type,
            "total_examples": self.total_examples,
            "n_examples": self.total_examples,
            "accepted_examples": self.accepted_examples,
            "abstained_examples": self.abstained_examples,
            "unfamiliar_examples": self.unfamiliar_examples,
            "accuracy": self.accuracy,
            "accepted_accuracy": self.accepted_accuracy,
            "selective_accuracy": self.accepted_accuracy,
            "conformal_coverage": self.conformal_coverage,
            "empirical_set_coverage": self.conformal_coverage,
            "average_set_size": self.average_set_size,
            "mean_set_size": self.average_set_size,
            "singleton_rate": self.singleton_rate,
            "ece": self.ece,
            "brier_score": self.brier_score,
            "nll": self.nll,
            "coverage": self.coverage,
            "abstain_rate": self.abstain_rate,
            "unfamiliar_rate": self.unfamiliar_rate,
            "abstention_reasons": dict(self.abstention_reasons),
            "mae": self.mae,
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
class EvaluationReport:
    """Dataset-level and per-field evaluation summary for a System 1 engine or compiled .s1m skill."""

    schema_name: str
    alpha: float
    total_examples: int
    accepted_examples: int
    abstained_examples: int
    unfamiliar_examples: int
    overall_accuracy: float
    accepted_accuracy: float
    conformal_coverage: float
    average_set_size: float
    singleton_rate: float
    ece: float
    brier_score: float
    nll: float
    coverage: float
    abstain_rate: float
    unfamiliar_rate: float
    abstention_reasons: Dict[str, int] = field(default_factory=dict)
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    fields: Dict[str, FieldEvaluationMetrics] = field(default_factory=dict)
    schema_digest: str = ""
    strict_mode: bool = False

    @property
    def n_examples(self) -> int:
        return self.total_examples

    @property
    def accuracy(self) -> float:
        return self.overall_accuracy

    @property
    def selective_accuracy(self) -> float:
        return self.accepted_accuracy

    @property
    def empirical_set_coverage(self) -> float:
        return self.conformal_coverage

    @property
    def mean_set_size(self) -> float:
        return self.average_set_size

    @property
    def field_metrics(self) -> Dict[str, FieldEvaluationMetrics]:
        return self.fields

    @property
    def per_field_metrics(self) -> Dict[str, FieldEvaluationMetrics]:
        return self.fields

    @property
    def per_field(self) -> Dict[str, FieldEvaluationMetrics]:
        return self.fields

    @property
    def latency_p50_ms(self) -> float:
        return self.p50_latency_ms

    @property
    def latency_p95_ms(self) -> float:
        return self.p95_latency_ms

    @property
    def latency_p99_ms(self) -> float:
        return self.p99_latency_ms

    @property
    def latency_ms_p50(self) -> float:
        return self.p50_latency_ms

    @property
    def latency_ms_p95(self) -> float:
        return self.p95_latency_ms

    def summary_table(self) -> str:
        lines = [
            f"EvaluationReport(schema={self.schema_name}, alpha={self.alpha:.3f}, n={self.total_examples})",
            f"  accuracy={self.overall_accuracy:.2%} accepted_accuracy={self.accepted_accuracy:.2%} conformal_coverage={self.conformal_coverage:.2%}",
            f"  routing: coverage={self.coverage:.2%} abstain_rate={self.abstain_rate:.2%} unfamiliar_rate={self.unfamiliar_rate:.2%}",
            f"  calibration: ece={self.ece:.4f} brier={self.brier_score:.4f} nll={self.nll:.4f}",
        ]
        for fname, fm in self.fields.items():
            lines.append(
                f"  - {fname} ({fm.field_type}): acc={fm.accuracy:.1%} acc_accepted={fm.accepted_accuracy:.1%} "
                f"conf_cov={fm.conformal_coverage:.1%} set_size={fm.average_set_size:.2f} ece={fm.ece:.4f} "
                f"cov/abs/unf={fm.coverage:.0%}/{fm.abstain_rate:.0%}/{fm.unfamiliar_rate:.0%}"
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        fields_dict = {k: v.to_dict() for k, v in self.fields.items()}
        return {
            "schema_name": self.schema_name,
            "schema_digest": self.schema_digest,
            "strict_mode": self.strict_mode,
            "alpha": self.alpha,
            "total_examples": self.total_examples,
            "n_examples": self.total_examples,
            "accepted_examples": self.accepted_examples,
            "abstained_examples": self.abstained_examples,
            "unfamiliar_examples": self.unfamiliar_examples,
            "overall_accuracy": self.overall_accuracy,
            "accuracy": self.overall_accuracy,
            "accepted_accuracy": self.accepted_accuracy,
            "selective_accuracy": self.accepted_accuracy,
            "conformal_coverage": self.conformal_coverage,
            "empirical_set_coverage": self.conformal_coverage,
            "average_set_size": self.average_set_size,
            "mean_set_size": self.average_set_size,
            "singleton_rate": self.singleton_rate,
            "ece": self.ece,
            "brier_score": self.brier_score,
            "nll": self.nll,
            "coverage": self.coverage,
            "abstain_rate": self.abstain_rate,
            "unfamiliar_rate": self.unfamiliar_rate,
            "abstention_reasons": dict(self.abstention_reasons),
            "mean_latency_ms": self.mean_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "latency_ms_p50": self.p50_latency_ms,
            "latency_ms_p95": self.p95_latency_ms,
            "fields": fields_dict,
            "field_metrics": fields_dict,
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def __getitem__(self, key: str) -> Any:
        if key in self.fields:
            return self.fields[key]
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
            key in self.fields
            or (not key.startswith("_") and hasattr(self, key) and not callable(getattr(self, key)))
        )


def _normalize_eval_rows(
    rows: Any,
    schema: DecisionSchema,
) -> List[Tuple[str, Dict[str, Any], Optional[Any]]]:
    if isinstance(rows, (str, Path)):
        path = Path(rows)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        raw_text = path.read_text(encoding="utf-8").strip()
        if not raw_text:
            raise ValueError(f"Dataset file is empty: {path}")
        if path.suffix.lower() == ".jsonl":
            parsed_rows: Any = [
                json.loads(line)
                for line in raw_text.splitlines()
                if line.strip()
            ]
        else:
            try:
                parsed_rows = json.loads(raw_text)
            except json.JSONDecodeError:
                lines = [line for line in raw_text.splitlines() if line.strip()]
                if len(lines) > 1:
                    parsed_rows = [json.loads(line) for line in lines]
                else:
                    raise
        rows = parsed_rows

    if isinstance(rows, Mapping):
        if "evaluate" in rows and isinstance(rows["evaluate"], Sequence) and "evaluate" not in schema.fields:
            rows = rows["evaluate"]
        elif "calibrate" in rows and isinstance(rows["calibrate"], Sequence) and len(rows) == 1 and "calibrate" not in schema.fields:
            rows = rows["calibrate"]
        elif "examples" in rows and isinstance(rows["examples"], Sequence) and "examples" not in schema.fields:
            rows = rows["examples"]
        else:
            by_prompt: Dict[str, Tuple[str, Dict[str, Any], Optional[Any]]] = {}
            for field_name, field_items in rows.items():
                if field_name not in schema.fields:
                    raise ValueError(f"Unknown field {field_name!r} for schema {schema.schema_name}")
                if not isinstance(field_items, Sequence) or isinstance(field_items, (str, bytes)):
                    raise ValueError(f"Field {field_name!r} must map to a sequence of examples")
                for item in field_items:
                    if isinstance(item, Mapping):
                        p_str = item.get("prompt", item.get("text", item.get("input", item.get("query"))))
                        lbl_val = item.get("label", item.get("answer", item.get("target")))
                        telem = item.get("telemetry")
                    elif isinstance(item, (tuple, list)) and len(item) in (2, 3):
                        p_str = item[0]
                        lbl_val = item[1]
                        telem = item[2] if len(item) == 3 else None
                    else:
                        raise ValueError(f"Invalid example item in field {field_name!r}: {item!r}")
                    if not isinstance(p_str, str) or not p_str.strip():
                        raise ValueError("Example prompt must be a non-empty string")
                    val_checked = schema.fields[field_name].validate_value(lbl_val)
                    if p_str not in by_prompt:
                        by_prompt[p_str] = (p_str, {field_name: val_checked}, telem)
                    else:
                        existing_map = by_prompt[p_str][1]
                        if field_name in existing_map and existing_map[field_name] != val_checked:
                            raise ValueError(
                                f"Conflicting labels for repeated prompt {p_str!r} on field {field_name!r}"
                            )
                        existing_map[field_name] = val_checked
            normalized = list(by_prompt.values())
            if not normalized:
                raise ValueError("Dataset must contain at least one example")
            return normalized

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError(f"Expected sequence of examples or path to dataset, got {type(rows).__name__}")

    normalized_list: List[Tuple[str, Dict[str, Any], Optional[Any]]] = []
    single_field_name = next(iter(schema.fields.keys())) if len(schema.fields) == 1 else None

    for idx, item in enumerate(rows):
        telemetry: Optional[Any] = None
        raw_labels: Optional[Dict[str, Any]] = None
        if isinstance(item, Mapping):
            prompt = item.get("prompt", item.get("text", item.get("input", item.get("query"))))
            telemetry = item.get("telemetry")
            wrapper_mapping: Optional[Dict[str, Any]] = None
            for w_key in ("labels", "values", "target", "output", "expected"):
                if w_key in item and isinstance(item[w_key], Mapping) and w_key not in schema.fields:
                    wrapper_mapping = dict(item[w_key])
                    break
            if wrapper_mapping is not None:
                raw_labels = wrapper_mapping
            else:
                direct = {k: item[k] for k in schema.fields if k in item}
                if direct:
                    raw_labels = direct
                elif any(k in item for k in ("label", "answer", "target", "output", "expected")):
                    lbl_obj = item.get(
                        "label",
                        item.get("answer", item.get("target", item.get("output", item.get("expected")))),
                    )
                    if isinstance(lbl_obj, Mapping):
                        raw_labels = dict(lbl_obj)
                    elif single_field_name is not None:
                        raw_labels = {single_field_name: lbl_obj}
                    else:
                        raise ValueError(
                            f"Row {idx} provides a scalar label for multi-field schema {schema.schema_name}; pass a dict of field labels"
                        )
        elif isinstance(item, (tuple, list)) and len(item) in (2, 3):
            prompt = item[0]
            lbl_obj = item[1]
            telemetry = item[2] if len(item) == 3 else None
            if isinstance(lbl_obj, Mapping):
                raw_labels = dict(lbl_obj)
            elif single_field_name is not None:
                raw_labels = {single_field_name: lbl_obj}
            else:
                raise ValueError(
                    f"Row {idx} provides a scalar label for multi-field schema {schema.schema_name}; pass a dict of field labels"
                )
        else:
            raise ValueError(f"Row {idx} has unsupported format: {type(item).__name__}")

        if not isinstance(prompt, str):
            raise ValueError(f"Row {idx} prompt must be a string")
        if not raw_labels:
            raise ValueError(f"Row {idx} is missing ground-truth labels")

        validated_labels: Dict[str, Any] = {}
        for fname, fval in raw_labels.items():
            if fname not in schema.fields:
                raise ValueError(f"Row {idx} references unknown field {fname!r} for schema {schema.schema_name}")
            validated_labels[fname] = schema.fields[fname].validate_value(fval)
        normalized_list.append((prompt, validated_labels, telemetry))

    if not normalized_list:
        raise ValueError("Dataset must contain at least one example")
    return normalized_list


@dataclass(frozen=True)
class BenchmarkReport:
    """Performance and latency benchmark report."""

    schema_name: str
    total_decisions: int
    mean_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_decisions_per_sec: float
    target_latency_ms: float = 20.0
    baseline_label: str = "Cloud SaaS API"
    baseline_latency_ms: float = 0.0
    speedup_factor: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionResult:
    """Strongly-typed outcome of a System 1 decision evaluation."""

    schema_name: str
    schema_digest: str
    prompt: str
    values: Dict[str, Any]
    confidences: Dict[str, float]
    conformal_sets: Dict[str, List[str]]
    probabilities: Dict[str, Dict[str, float]]
    is_ambiguous: bool
    latency_ms: float
    alpha: float
    receipt: Optional[DecisionWitnessReceipt]
    margins: Dict[str, float] = field(default_factory=dict)
    margin_thresholds: Dict[str, float] = field(default_factory=dict)
    margin_gate_active: Dict[str, bool] = field(default_factory=dict)
    odds_ratios: Dict[str, float] = field(default_factory=dict)
    relative_odds_ratio_thresholds: Dict[str, float] = field(default_factory=dict)
    confidence_floors: Dict[str, float] = field(default_factory=dict)
    ambiguous_fields: List[str] = field(default_factory=list)
    escalated_fields: List[str] = field(default_factory=list)
    telemetry: Optional[Any] = None
    is_cache_hit: bool = False
    embedding: Optional[np.ndarray] = None
    familiarity_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def abstention_reason(self) -> Optional[str]:
        if isinstance(self.metadata, dict):
            return self.metadata.get("abstention_reason")
        return None

    @property
    def abstention_reasons(self) -> Dict[str, str]:
        if isinstance(self.metadata, dict):
            reasons = self.metadata.get("abstention_reasons")
            if isinstance(reasons, dict):
                return reasons
        return {}

    def __getattr__(self, item: str) -> Any:
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        values = self.__dict__.get("values")
        if values is not None and item in values:
            return values[item]
        raise AttributeError(f"'DecisionResult' object has no attribute {item!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_digest": self.schema_digest,
            "prompt": self.prompt,
            "values": self.values,
            "confidences": self.confidences,
            "conformal_sets": self.conformal_sets,
            "probabilities": self.probabilities,
            "is_ambiguous": self.is_ambiguous,
            "latency_ms": self.latency_ms,
            "alpha": self.alpha,
            "receipt": self.receipt.to_dict() if self.receipt is not None else None,
            "margins": self.margins,
            "margin_thresholds": self.margin_thresholds,
            "margin_gate_active": self.margin_gate_active,
            "odds_ratios": self.odds_ratios,
            "relative_odds_ratio_thresholds": self.relative_odds_ratio_thresholds,
            "confidence_floors": self.confidence_floors,
            "ambiguous_fields": self.ambiguous_fields,
            "escalated_fields": self.escalated_fields,
            "telemetry": self.telemetry,
            "is_cache_hit": self.is_cache_hit,
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


class SystemOneEngine:
    """Local structured decisions with calibration, caching and optional audit evidence.

    Use strict_mode=True for taught skills and honor the returned review flags.
    Prediction-set coverage needs exchangeable held-out evidence; it is not
    accepted-answer accuracy or tool authorization. Performance is workload-dependent."""

    def __init__(
        self,
        schema: Union[DecisionSchema, Type[DecisionSchema]],
        *,
        model: Optional[Any] = None,
        dimension: int = 384,
        backend: str = "auto",
        signing_key: Optional[Ed25519PrivateKey] = None,
        ledger: Optional[ActionLedger] = None,
        projector: Optional[Any] = None,
        contrastive_whitening: bool = True,
        use_cache: bool = True,
        cache_threshold: float = 0.98,
        cache_capacity: int = 2048,
        enable_margin_gating: bool = True,
        margin_threshold: Optional[float] = None,
        forgetting_factor: float = 0.995,
        relative_odds_ratio: Optional[float] = 1.5,
        confidence_floor_tau0: Optional[float] = 0.15,
        recency_weighted: bool = False,
        fail_closed_ledger: bool = False,
        strict_mode: bool = False,
        policy_scope: str = "",
        model_path: Optional[Union[str, Path]] = None,
        auto_reload: bool = False,
        compiled_model: Optional[Any] = None,
    ) -> None:
        if model is None and compiled_model is not None:
            model = compiled_model
        if isinstance(schema, type) and issubclass(schema, DecisionSchema):
            self.schema: DecisionSchema = schema()
        elif isinstance(schema, DecisionSchema):
            self.schema = schema
        else:
            raise TypeError(f"Expected DecisionSchema subclass or instance, got {type(schema).__name__}")

        self.dimension = dimension
        self.backend = backend
        self.signing_key: Optional[Ed25519PrivateKey] = signing_key
        self.ledger: Optional[ActionLedger] = ledger
        self.fail_closed_ledger = bool(fail_closed_ledger)
        self.projector = projector
        self.contrastive_whitening = contrastive_whitening
        self.use_cache = bool(use_cache)
        self.cache = SemanticSystemOneCache(
            capacity=cache_capacity,
            similarity_threshold=cache_threshold,
        )
        self.enable_margin_gating = bool(enable_margin_gating)
        self.margin_threshold = margin_threshold
        self.forgetting_factor = float(forgetting_factor)
        self.relative_odds_ratio = (
            float(relative_odds_ratio) if relative_odds_ratio is not None else None
        )
        self.confidence_floor_tau0 = (
            float(confidence_floor_tau0) if confidence_floor_tau0 is not None else None
        )
        self.recency_weighted = bool(recency_weighted)
        self.strict_mode = bool(strict_mode)
        self.default_alpha: float = 0.05
        self.policy_scope = str(policy_scope)
        self.policy_epoch: int = 0
        self.model_version: int = 1
        self._lock = threading.RLock()
        self.model_path: Optional[Path] = Path(model_path).resolve() if model_path is not None else None
        self.auto_reload: bool = bool(auto_reload)
        self._model_stat: Optional[Tuple[int, int, str]] = None

        if model is None and self.model_path is not None and self.model_path.exists():
            from system1.compiler import CompiledSystemOneModel, _read_skill_bytes

            raw = _read_skill_bytes(self.model_path)
            st = self.model_path.stat()
            loaded = CompiledSystemOneModel.from_bytes(
                raw,
                projector=projector,
                backend=backend if backend != "auto" else "numpy",
            )
            if loaded.schema.schema_digest() != self.schema.schema_digest():
                raise ValueError("Loaded skill schema digest does not match engine schema")
            model = loaded
            self._model_stat = (st.st_mtime_ns, st.st_size, hashlib.sha256(raw).hexdigest())

        # Non-autoregressive decision model
        if model is not None:
            self._sync_from_model(model)
        else:
            self.model = SystemOneModel(
                self.schema,
                dimension=self.dimension,
                backend=self.backend,
                projector=self.projector,
                contrastive_whitening=self.contrastive_whitening,
                forgetting_factor=self.forgetting_factor,
                recency_weighted=self.recency_weighted,
            )
            self.dimension = self.model.dimension
            self.backend = self.model.backend
            self.projector = self.model.projector
            self._init_calibrators_and_predictors()

    def _init_calibrators_and_predictors(self) -> None:
        self.calibrators: Dict[str, DecisionCalibrator] = {
            name: DecisionCalibrator(temperature=1.0)
            for name in self.schema.fields.keys()
        }
        self.conformal_predictors: Dict[str, ConformalPredictor] = {}
        self.regression_conformal_predictors: Dict[str, RegressionConformalPredictor] = {}
        for name, f in self.schema.fields.items():
            if isinstance(f, ChoiceField):
                self.conformal_predictors[name] = ConformalPredictor(
                    name,
                    f.options,
                    margin_threshold=self.margin_threshold,
                    relative_odds_ratio=self.relative_odds_ratio,
                    confidence_floor_tau0=self.confidence_floor_tau0,
                )
            elif isinstance(f, BooleanField):
                self.conformal_predictors[name] = ConformalPredictor(
                    name,
                    ("False", "True"),
                    margin_threshold=self.margin_threshold,
                    relative_odds_ratio=self.relative_odds_ratio,
                    confidence_floor_tau0=self.confidence_floor_tau0,
                )
            elif isinstance(f, MultiChoiceField):
                self.conformal_predictors[name] = ConformalPredictor(
                    name,
                    f.options,
                    margin_threshold=self.margin_threshold,
                    relative_odds_ratio=self.relative_odds_ratio,
                    confidence_floor_tau0=self.confidence_floor_tau0,
                )
            elif isinstance(f, ScoreField):
                self.regression_conformal_predictors[name] = RegressionConformalPredictor(
                    name, min_value=f.min_value, max_value=f.max_value
                )

    def _sync_from_model(self, model: Any) -> None:
        self.model = model
        self.model_version = max(getattr(self, "model_version", 1), int(getattr(model, "model_version", 1)))
        self.dimension = getattr(model, "dimension", self.dimension)
        self.backend = getattr(model, "backend", self.backend)
        self.projector = getattr(model, "projector", self.projector)
        self.recency_weighted = getattr(model, "recency_weighted", self.recency_weighted)
        model_meta = getattr(model, "metadata", None)
        if isinstance(model_meta, Mapping) and "default_alpha" in model_meta:
            raw_alpha = model_meta.get("default_alpha")
            if isinstance(raw_alpha, (int, float)) and not isinstance(raw_alpha, bool) and 0.0 < float(raw_alpha) < 1.0:
                self.default_alpha = float(raw_alpha)
        self._init_calibrators_and_predictors()
        # Lessons belong to the active skill; a replacement retires the old ones.
        self._learned_prompt_overrides = {}
        self._learned_prompt_override_recency: Dict[str, bool] = {}
        model_rec = getattr(model, "_learned_prompt_override_recency", {})
        if hasattr(model, "_learned_prompt_overrides") and isinstance(model._learned_prompt_overrides, Mapping):
            for p_key, f_map in model._learned_prompt_overrides.items():
                if isinstance(p_key, str) and isinstance(f_map, Mapping):
                    self._learned_prompt_overrides[p_key] = copy.deepcopy(dict(f_map))
                    if isinstance(model_rec, Mapping) and p_key in model_rec:
                        self._learned_prompt_override_recency[p_key] = bool(model_rec[p_key])
                    else:
                        self._learned_prompt_override_recency[p_key] = bool(self.recency_weighted)

        if hasattr(model, "heads"):
            for name, ch in model.heads.items():
                if name in self.calibrators:
                    self.calibrators[name].temperature = getattr(ch, "temperature", 1.0)
                if name in self.conformal_predictors:
                    self.conformal_predictors[name].score_method = getattr(ch, "score_method", "aps")
                    calib_scores = getattr(ch, "calibration_scores", ())
                    if len(calib_scores) > 0:
                        self.conformal_predictors[name].calibration_scores = np.sort(
                            np.asarray(calib_scores, dtype=np.float64)
                        )
                        self.conformal_predictors[name].is_calibrated = True
                    q = getattr(ch, "conformal_quantile", 0.0)
                    if q > 0:
                        self.conformal_predictors[name].quantile = q
                        self.conformal_predictors[name].is_calibrated = True
                    m = getattr(ch, "margin_threshold", None)
                    if m is not None and self.margin_threshold is None:
                        self.conformal_predictors[name].calibrated_margin_threshold = float(m)
                    ror = getattr(ch, "relative_odds_ratio", None)
                    if ror is not None and self.relative_odds_ratio is None:
                        self.conformal_predictors[name].relative_odds_ratio = float(ror)
                    cft = getattr(ch, "confidence_floor_tau0", None)
                    if cft is not None and self.confidence_floor_tau0 is None:
                        self.conformal_predictors[name].confidence_floor_tau0 = float(cft)
                    class_scores = getattr(ch, "class_calibration_scores", None)
                    if class_scores:
                        self.conformal_predictors[name].class_calibration_scores = {
                            k: np.sort(np.asarray(v, dtype=np.float64))
                            for k, v in class_scores.items()
                            if len(v) > 0
                        }
                        self.conformal_predictors[name].mondrian = bool(getattr(ch, "mondrian", False))
                    fam_t = float(getattr(ch, "familiarity_threshold", 0.0))
                    if fam_t > 0.0:
                        self.conformal_predictors[name].familiarity_threshold = fam_t
                    proto_t = float(getattr(ch, "prototype_threshold", 0.0))
                    if proto_t > 0.0:
                        self.conformal_predictors[name].prototype_threshold = proto_t
                    proto_scores = getattr(ch, "prototype_scores", ())
                    if len(proto_scores) > 0:
                        self.conformal_predictors[name].prototype_scores = np.sort(
                            np.asarray(proto_scores, dtype=np.float64)
                        )
                    protos = getattr(ch, "prototypes", None)
                    if protos is not None:
                        self.conformal_predictors[name].prototypes = np.asarray(protos, dtype=np.float32)
                if name in self.regression_conformal_predictors:
                    calib_scores = getattr(ch, "calibration_scores", ())
                    if len(calib_scores) > 0:
                        self.regression_conformal_predictors[name].residuals = np.sort(
                            np.asarray(calib_scores, dtype=np.float64)
                        )
                        self.regression_conformal_predictors[name].is_calibrated = True

    def reload_if_changed(self) -> bool:
        """Atomically reload the compiled .s1m model from model_path if modified on disk."""
        if self.model_path is None or not self.model_path.exists():
            return False
        try:
            st = self.model_path.stat()
        except OSError:
            return False
        if self._model_stat is not None and (st.st_mtime_ns, st.st_size) == self._model_stat[:2]:
            return False
        from system1.compiler import CompiledSystemOneModel, _read_skill_bytes
        try:
            raw = _read_skill_bytes(self.model_path)
        except (OSError, ValueError):
            return False
        if not raw:
            return False
        digest = hashlib.sha256(raw).hexdigest()
        if self._model_stat is not None and digest == self._model_stat[2]:
            self._model_stat = (st.st_mtime_ns, st.st_size, digest)
            return False
        try:
            new_model = CompiledSystemOneModel.from_bytes(
                raw,
                backend=self.backend if self.backend != "auto" else "numpy",
            )
        except Exception:
            # Ignore partial/truncated mid-write files; keep current model until write completes.
            return False
        if new_model.schema.schema_digest() != self.schema.schema_digest():
            raise ValueError("Reloaded skill schema digest does not match engine schema")
        with self._lock:
            self._sync_from_model(new_model)
            self.model_version += 1
            new_model.model_version = self.model_version
            self._model_stat = (st.st_mtime_ns, st.st_size, digest)
            if self.use_cache and self.cache is not None:
                self.cache.clear()
        return True

    def _model_digest(self) -> str:
        """Computes deterministic digest of active model weights."""
        if hasattr(self.model, "model_digest") and callable(self.model.model_digest):
            return self.model.model_digest()
        heads = getattr(self.model, "heads", getattr(self.model, "_field_heads", {}))
        if not heads:
            return f"v{self.model_version}"
        h = hashlib.sha256(f"v:{self.model_version}".encode("utf-8"))
        for fname in sorted(heads.keys()):
            hd = heads[fname]
            w = getattr(hd, "weights", None)
            b = getattr(hd, "biases", None)
            if w is not None:
                h.update(np.asarray(w, dtype=np.float32).tobytes())
            if b is not None:
                h.update(np.asarray(b, dtype=np.float32).tobytes())
        return h.hexdigest()

    def _projector_digest(self) -> str:
        """Computes deterministic digest of neural projector."""
        if hasattr(self.projector, "projector_digest") and callable(self.projector.projector_digest):
            return self.projector.projector_digest()
        from system1.compiler import _projector_config
        p_str = json.dumps(_projector_config(self.projector), sort_keys=True)
        return hashlib.sha256(p_str.encode("utf-8")).hexdigest()

    def _calibration_digest(self) -> str:
        """Computes deterministic digest of calibration parameters and conformal nonconformity scores."""
        h = hashlib.sha256()
        for fname in sorted(self.schema.fields.keys()):
            calib = self.calibrators.get(fname)
            if calib is not None:
                h.update(f"{fname}:temp:{calib.temperature:.6f}".encode("utf-8"))
            cp = self.conformal_predictors.get(fname)
            if cp is not None and getattr(cp, "is_calibrated", False):
                scores = getattr(cp, "calibration_scores", np.array([]))
                mondrian_flag = bool(getattr(cp, "mondrian", False))
                fam_thresh = float(getattr(cp, "familiarity_threshold", 0.0))
                proto_thresh = float(getattr(cp, "prototype_threshold", 0.0))
                h.update(
                    f"{fname}:{cp.score_method}:m={mondrian_flag}:ft={fam_thresh:.6f}:pt={proto_thresh:.6f}:cp_scores:{scores.tobytes()}".encode("utf-8")
                )
                for cls_lbl, cls_arr in sorted(getattr(cp, "class_calibration_scores", {}).items()):
                    h.update(f":cls:{cls_lbl}:{np.asarray(cls_arr, dtype=np.float64).tobytes()}".encode("utf-8"))
            rcp = self.regression_conformal_predictors.get(fname)
            if rcp is not None and getattr(rcp, "is_calibrated", False):
                res = getattr(rcp, "residuals", np.array([]))
                h.update(f"{fname}:rcp_res:{res.tobytes()}".encode("utf-8"))
        return h.hexdigest()

    def encode(
        self,
        prompt: str,
        telemetry: Optional[Any] = None,
        recency_weighted: Optional[bool] = None,
    ) -> np.ndarray:
        """Encodes prompt into semantic embedding vector."""
        use_recency = self.recency_weighted if recency_weighted is None else bool(recency_weighted)
        if hasattr(self.model, "encode"):
            try:
                return self.model.encode(prompt, telemetry=telemetry, recency_weighted=use_recency)
            except TypeError:
                return self.model.encode(prompt, telemetry=telemetry)
        if self.projector is not None:
            try:
                emb = self.projector.project(prompt, recency_weighted=use_recency)
            except TypeError:
                emb = self.projector.project(prompt)
            norm = float(np.linalg.norm(emb))
            return (emb / norm).astype(np.float32) if norm > 1e-12 else emb
        v = np.zeros(self.dimension, dtype=np.float32)
        v[0] = 1.0
        return v

    def calibrate(
        self,
        dataset: Sequence[Tuple[str, Mapping[str, Any]]],
        *,
        n_bins: int = 10,
        familiarity_alpha: float = 0.0,
        prototype_alpha: float = 0.0,
    ) -> Dict[str, CalibrationMetrics]:
        """Calibrate on held-out examples and preserve the evidence in compiled skills.

        Repeated normalized prompts are one observation. Temperature fitting and
        conformal calibration use disjoint prompts, including for multi-label and
        continuous outputs. Callers must keep these examples out of teaching.
        """
        if not dataset:
            raise ValueError("Calibration dataset must contain at least one example")
        if not math.isfinite(familiarity_alpha) or not 0.0 <= familiarity_alpha < 1.0:
            raise ValueError("familiarity_alpha must be finite and in [0, 1)")
        if not math.isfinite(prototype_alpha) or not 0.0 <= prototype_alpha < 1.0:
            raise ValueError("prototype_alpha must be finite and in [0, 1)")
        grouped = {}
        for prompt, labels in dataset:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Calibration prompts must be non-empty strings")
            key = " ".join(prompt.casefold().split())
            validated = {}
            for name, value in labels.items():
                if name not in self.schema.fields:
                    raise ValueError(f"Unknown calibration field: {name!r}")
                validated[name] = self.schema.fields[name].validate_value(value)
            if key in grouped and grouped[key][1] != validated:
                raise ValueError("Conflicting labels for repeated calibration prompt")
            grouped[key] = (prompt, validated)

        with self._lock, getattr(self.model, "_lock", nullcontext()):
            rows = list(grouped.values())
            results = self.model.forward_batch([row[0] for row in rows])
            metrics = {}
            for name, definition in self.schema.fields.items():
                indices = [i for i, row in enumerate(rows) if name in row[1]]
                if not indices:
                    continue
                rng = np.random.RandomState(42)
                rng.shuffle(indices)
                count = len(indices) // 2
                logits = np.asarray([results[i].fields[name].logits for i in indices], dtype=np.float64)
                labels = [rows[i][1][name] for i in indices]
                calibrator = DecisionCalibrator()
                if isinstance(definition, ChoiceField):
                    targets = np.asarray([definition.options.index(label) for label in labels])
                elif isinstance(definition, BooleanField):
                    targets = np.asarray(labels, dtype=np.int64)
                elif isinstance(definition, MultiChoiceField):
                    targets = np.asarray([[int(opt in label) for opt in definition.options] for label in labels])
                else:
                    midpoint = (definition.min_value + definition.max_value) / 2
                    targets = np.asarray([int(label >= midpoint) for label in labels])

                temp_logits = logits[:count]
                temp_targets = targets[:count]
                if isinstance(definition, MultiChoiceField):
                    temp_logits, temp_targets = temp_logits.flatten(), temp_targets.flatten()
                elif isinstance(definition, ScoreField) or (isinstance(definition, BooleanField) and logits.shape[1] == 1):
                    temp_logits = temp_logits.flatten()
                metrics[name] = calibrator.fit(temp_logits, temp_targets, n_bins=n_bins)
                # Runtime always passes one head's vector, rather than a batch of
                # binary scalars. Temperature is the complete inference parameter.
                calibrator.is_binary = False
                self.calibrators[name] = calibrator
                conf_logits = logits[count:]
                conf_labels = labels[count:]
                conf_indices = indices[count:]
                conf_embeddings = np.asarray(
                    [results[i].embedding for i in conf_indices],
                    dtype=np.float32,
                )
                fam_vals: Optional[List[float]] = None
                if familiarity_alpha > 0.0 and conf_indices:
                    proj = getattr(self.model, "projector", self.projector)
                    if proj is not None and hasattr(proj, "raw_norm"):
                        fam_vals = [float(proj.raw_norm(rows[i][0])) for i in conf_indices]
                    else:
                        fam_vals = [float(np.linalg.norm(e)) for e in conf_embeddings]
                compiled = getattr(self.model, "heads", {}).get(name)
                proto_mat = (
                    getattr(compiled, "prototypes", None)
                    if compiled is not None
                    else getattr(self.conformal_predictors.get(name), "prototypes", None)
                )

                if isinstance(definition, (ChoiceField, BooleanField)):
                    if isinstance(definition, BooleanField) and conf_logits.shape[1] == 1:
                        z = conf_logits.reshape(-1)
                        conf_logits = np.column_stack([-z / 2, z / 2])
                    probs = calibrator.calibrate_logits(conf_logits)
                    predictor = self.conformal_predictors[name]
                    predictor.calibrate(
                        probs,
                        [str(label) for label in conf_labels],
                        familiarity_values=fam_vals,
                        familiarity_alpha=familiarity_alpha,
                        embeddings=conf_embeddings if len(conf_embeddings) > 0 else None,
                        prototypes=proto_mat,
                        prototype_alpha=prototype_alpha,
                    )
                    scores = predictor.calibration_scores
                elif isinstance(definition, MultiChoiceField):
                    probs = _stable_sigmoid(conf_logits / calibrator.temperature, temperature=1.0)
                    errors = np.where(targets[count:] == 1, 1.0 - probs, probs)
                    predictor = self.conformal_predictors[name]
                    predictor.calibration_scores = np.sort(np.max(errors, axis=1))
                    predictor.is_calibrated = True
                    scores = predictor.calibration_scores
                else:
                    probs = _stable_sigmoid(conf_logits.reshape(-1) / calibrator.temperature, temperature=1.0)
                    predictions = definition.min_value + (definition.max_value - definition.min_value) * probs
                    predictor = self.regression_conformal_predictors[name]
                    predictor.calibrate(predictions, conf_labels)
                    scores = predictor.residuals

                if compiled is not None and hasattr(compiled, "calibration_scores"):
                    compiled.temperature = calibrator.temperature
                    compiled.calibration_scores = tuple(float(score) for score in scores)
                    compiled.score_method = getattr(predictor, "score_method", "aps")
                    if isinstance(definition, (ChoiceField, BooleanField)):
                        compiled.familiarity_threshold = float(getattr(predictor, "familiarity_threshold", 0.0))
                        compiled.prototype_threshold = float(getattr(predictor, "prototype_threshold", 0.0))
                        p_scores = getattr(predictor, "prototype_scores", np.array([]))
                        compiled.prototype_scores = tuple(float(s) for s in p_scores)
                        c_scores = getattr(predictor, "class_calibration_scores", {})
                        if c_scores:
                            compiled.class_calibration_scores = {
                                str(k): tuple(float(v) for v in vals) for k, vals in c_scores.items()
                            }
                    k = int(math.ceil((len(scores) + 1) * (1.0 - self.default_alpha)))
                    upper = definition.max_value - definition.min_value if isinstance(definition, ScoreField) else 1.0
                    compiled.conformal_quantile = float(scores[k - 1]) if k <= len(scores) else upper
            self.model_version = max(self.model_version, int(getattr(self.model, "model_version", 1))) + 1
            if hasattr(self.model, "model_version"):
                self.model.model_version = self.model_version
            if getattr(self.model, "cache", None) is not None:
                self.model.cache.clear()
            if self.cache is not None:
                self.cache.clear()
            return metrics

    def recalibrate(
        self,
        examples: Union[Sequence[Any], str, Path],
        alpha: float = 0.05,
        *,
        n_bins: int = 10,
        familiarity_alpha: Optional[float] = None,
        prototype_alpha: Optional[float] = None,
        mondrian: Optional[bool] = None,
    ) -> RecalibrationReport:
        """Recalibrate temperature scaling and conformal quantiles on held-out examples without modifying weights."""
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(float(alpha))
            or not (0.0 < float(alpha) < 1.0)
        ):
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        alpha_val = float(alpha)
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")

        model_meta = getattr(self.model, "metadata", None)
        if familiarity_alpha is not None:
            if not math.isfinite(float(familiarity_alpha)) or not (0.0 <= float(familiarity_alpha) < 1.0):
                raise ValueError("familiarity_alpha must be finite and in [0, 1)")
            eff_fam_alpha = float(familiarity_alpha)
        elif isinstance(model_meta, Mapping) and isinstance(model_meta.get("familiarity_alpha"), (int, float)) and not isinstance(model_meta.get("familiarity_alpha"), bool):
            cand_fam = float(model_meta["familiarity_alpha"])
            eff_fam_alpha = cand_fam if (math.isfinite(cand_fam) and 0.0 <= cand_fam < 1.0) else 0.0
        else:
            eff_fam_alpha = 0.0

        if prototype_alpha is not None:
            if not math.isfinite(float(prototype_alpha)) or not (0.0 <= float(prototype_alpha) < 1.0):
                raise ValueError("prototype_alpha must be finite and in [0, 1)")
            eff_proto_alpha = float(prototype_alpha)
        elif isinstance(model_meta, Mapping) and isinstance(model_meta.get("prototype_alpha"), (int, float)) and not isinstance(model_meta.get("prototype_alpha"), bool):
            cand_proto = float(model_meta["prototype_alpha"])
            eff_proto_alpha = cand_proto if (math.isfinite(cand_proto) and 0.0 <= cand_proto < 1.0) else 0.0
        else:
            eff_proto_alpha = 0.0

        raw_rows = _normalize_eval_rows(examples, self.schema)
        grouped: Dict[Tuple[str, str], Tuple[str, Dict[str, Any], Optional[Any]]] = {}
        for prompt, validated, telemetry in raw_rows:
            if not prompt.strip():
                raise ValueError("Recalibration prompt must be a non-empty string")
            telem_key = json.dumps(telemetry, sort_keys=True, default=str) if telemetry is not None else ""
            key = (" ".join(prompt.casefold().split()), telem_key)
            if key in grouped and grouped[key][1] != validated:
                raise ValueError("Conflicting labels for repeated calibration prompt")
            grouped[key] = (prompt, validated, telemetry)

        with self._lock, getattr(self.model, "_lock", nullcontext()):
            self.default_alpha = alpha_val
            if hasattr(self.model, "metadata") and isinstance(self.model.metadata, dict):
                self.model.metadata["default_alpha"] = alpha_val
                if familiarity_alpha is not None:
                    self.model.metadata["familiarity_alpha"] = eff_fam_alpha
                if prototype_alpha is not None:
                    self.model.metadata["prototype_alpha"] = eff_proto_alpha
                if mondrian is not None:
                    self.model.metadata["mondrian"] = bool(mondrian)

            rows = list(grouped.values())
            prev_model_cache = getattr(self.model, "use_cache", None)
            try:
                if prev_model_cache is not None:
                    self.model.use_cache = False
                if any(r[2] is not None for r in rows):
                    results = [self.model.forward_single(r[0], telemetry=r[2]) for r in rows]
                else:
                    results = self.model.forward_batch([r[0] for r in rows])
            finally:
                if prev_model_cache is not None:
                    self.model.use_cache = prev_model_cache
            report_fields: Dict[str, RecalibrationMetrics] = {}

            for name, definition in self.schema.fields.items():
                indices = [i for i, row in enumerate(rows) if name in row[1]]
                if not indices:
                    continue

                prev_cal = self.calibrators.get(name)
                prev_temp = float(prev_cal.temperature) if prev_cal is not None else 1.0
                compiled = getattr(self.model, "heads", {}).get(name)
                if (
                    name in self.conformal_predictors
                    and self.conformal_predictors[name].is_calibrated
                    and len(self.conformal_predictors[name].calibration_scores) > 0
                ):
                    sc_prev = self.conformal_predictors[name].calibration_scores
                    k_prev = int(math.ceil((len(sc_prev) + 1) * (1.0 - alpha_val)))
                    prev_q = 1.0 if k_prev > len(sc_prev) else float(sc_prev[k_prev - 1])
                elif compiled is not None and float(getattr(compiled, "conformal_quantile", 0.0)) > 0.0:
                    prev_q = float(compiled.conformal_quantile)
                elif (
                    name in self.regression_conformal_predictors
                    and self.regression_conformal_predictors[name].is_calibrated
                    and len(self.regression_conformal_predictors[name].residuals) > 0
                ):
                    res_prev = self.regression_conformal_predictors[name].residuals
                    k_prev = int(math.ceil((len(res_prev) + 1) * (1.0 - alpha_val)))
                    upper_p = float(definition.max_value - definition.min_value) if isinstance(definition, ScoreField) else 1.0
                    prev_q = upper_p if k_prev > len(res_prev) else float(res_prev[k_prev - 1])
                else:
                    prev_q = 0.0

                rng = np.random.RandomState(42)
                n_total = len(indices)
                logits = np.asarray([results[i].fields[name].logits for i in indices], dtype=np.float64)
                labels = [rows[i][1][name] for i in indices]

                if n_total >= 4 and isinstance(definition, (ChoiceField, BooleanField)):
                    buckets: Dict[Any, List[int]] = {}
                    for pos, lbl in enumerate(labels):
                        buckets.setdefault(lbl, []).append(pos)
                    temp_pos: List[int] = []
                    conf_pos: List[int] = []
                    if any(len(b) >= 2 for b in buckets.values()):
                        for b in buckets.values():
                            b_copy = list(b)
                            rng.shuffle(b_copy)
                            if len(b_copy) >= 2:
                                split_k = len(b_copy) // 2
                                temp_pos.extend(b_copy[:split_k])
                                conf_pos.extend(b_copy[split_k:])
                            else:
                                conf_pos.append(b_copy[0])
                        if not temp_pos or not conf_pos:
                            perm = list(range(n_total))
                            rng.shuffle(perm)
                            temp_pos = perm[: n_total // 2]
                            conf_pos = perm[n_total // 2 :]
                    else:
                        perm = list(range(n_total))
                        rng.shuffle(perm)
                        temp_pos = perm[: n_total // 2]
                        conf_pos = perm[n_total // 2 :]
                elif n_total >= 4:
                    perm = list(range(n_total))
                    rng.shuffle(perm)
                    temp_pos = perm[: n_total // 2]
                    conf_pos = perm[n_total // 2 :]
                else:
                    temp_pos = list(range(n_total))
                    conf_pos = list(range(n_total))

                calibrator = DecisionCalibrator(temperature=prev_temp)
                if isinstance(definition, ScoreField):
                    calibrator.is_binary = False
                    val_span = max(1e-6, float(definition.max_value - definition.min_value))
                    p_ratio = _stable_sigmoid(logits.flatten() / max(1e-4, prev_temp), temperature=1.0)
                    y_ratio = np.clip(
                        (np.asarray(labels, dtype=np.float64) - float(definition.min_value)) / val_span,
                        0.0,
                        1.0,
                    )
                    p_clip = np.clip(p_ratio, 1e-12, 1.0 - 1e-12)
                    init_ece = float(np.mean(np.abs(p_ratio - y_ratio))) if len(logits) > 0 else 0.0
                    init_brier = float(np.mean((p_ratio - y_ratio) ** 2)) if len(logits) > 0 else 0.0
                    init_nll = (
                        float(-np.mean(y_ratio * np.log(p_clip) + (1.0 - y_ratio) * np.log(1.0 - p_clip)))
                        if len(logits) > 0
                        else 0.0
                    )
                    cal_ece = init_ece
                    cal_brier = init_brier
                    cal_nll = init_nll
                    targets = np.zeros(len(labels), dtype=np.int64)
                else:
                    if isinstance(definition, ChoiceField):
                        targets = np.asarray([definition.options.index(label) for label in labels], dtype=np.int64)
                    elif isinstance(definition, BooleanField):
                        targets = np.asarray(labels, dtype=np.int64)
                    else:
                        targets = np.asarray([[int(opt in label) for opt in definition.options] for label in labels], dtype=np.int64)

                    temp_logits = logits[temp_pos]
                    temp_targets = targets[temp_pos]
                    is_1d_temp = False
                    if isinstance(definition, MultiChoiceField):
                        temp_logits, temp_targets = temp_logits.flatten(), temp_targets.flatten()
                        is_1d_temp = True
                    elif isinstance(definition, BooleanField) and logits.shape[1] == 1:
                        temp_logits = temp_logits.flatten()
                        is_1d_temp = True

                    prev_cal_eval = DecisionCalibrator(temperature=prev_temp)
                    prev_cal_eval.is_binary = bool(is_1d_temp or temp_logits.ndim == 1)
                    prev_probs = prev_cal_eval.calibrate_logits(temp_logits)
                    if prev_probs.ndim == 1:
                        prev_probs = np.column_stack([1.0 - prev_probs, prev_probs])
                    init_nll = float(compute_nll(prev_probs, temp_targets)) if len(temp_logits) > 0 else 0.0
                    init_ece = float(compute_ece_and_bins(prev_probs, temp_targets, n_bins=n_bins)[0]) if len(temp_logits) > 0 else 0.0
                    init_brier = float(compute_brier_decomposition(prev_probs, temp_targets, n_bins=n_bins).empirical_brier) if len(temp_logits) > 0 else 0.0

                    calibrator.fit(temp_logits, temp_targets, n_bins=n_bins)
                    post_cal_eval = DecisionCalibrator(temperature=calibrator.temperature)
                    post_cal_eval.is_binary = bool(is_1d_temp or temp_logits.ndim == 1)
                    post_probs = post_cal_eval.calibrate_logits(temp_logits)
                    if post_probs.ndim == 1:
                        post_probs = np.column_stack([1.0 - post_probs, post_probs])
                    cal_nll = float(compute_nll(post_probs, temp_targets)) if len(temp_logits) > 0 else 0.0
                    cal_ece = float(compute_ece_and_bins(post_probs, temp_targets, n_bins=n_bins)[0]) if len(temp_logits) > 0 else 0.0
                    cal_brier = float(compute_brier_decomposition(post_probs, temp_targets, n_bins=n_bins).empirical_brier) if len(temp_logits) > 0 else 0.0
                    calibrator.is_binary = False

                    if len(temp_logits) > 0 and init_nll <= cal_nll:
                        calibrator.temperature = prev_temp
                        cal_nll = init_nll
                        cal_ece = init_ece
                        cal_brier = init_brier

                self.calibrators[name] = calibrator

                conf_logits = logits[conf_pos]
                conf_labels = [labels[p] for p in conf_pos]
                conf_indices = [indices[p] for p in conf_pos]
                conf_embeddings = np.asarray(
                    [results[i].embedding for i in conf_indices],
                    dtype=np.float32,
                )
                fam_vals: Optional[List[float]] = None
                if eff_fam_alpha > 0.0 and conf_indices:
                    proj = getattr(self.model, "projector", self.projector)
                    if proj is not None and hasattr(proj, "raw_norm"):
                        fam_vals = [float(proj.raw_norm(rows[i][0])) for i in conf_indices]
                    else:
                        fam_vals = [float(np.linalg.norm(e)) for e in conf_embeddings]
                proto_mat = (
                    getattr(compiled, "prototypes", None)
                    if compiled is not None
                    else getattr(self.conformal_predictors.get(name), "prototypes", None)
                )

                if isinstance(definition, (ChoiceField, BooleanField)):
                    if isinstance(definition, BooleanField) and conf_logits.shape[1] == 1:
                        z = conf_logits.reshape(-1)
                        conf_logits = np.column_stack([-z / 2, z / 2])
                    probs = calibrator.calibrate_logits(conf_logits)
                    predictor = self.conformal_predictors[name]
                    if mondrian is not None:
                        predictor.mondrian = bool(mondrian)
                    if familiarity_alpha is not None and eff_fam_alpha == 0.0:
                        predictor.familiarity_threshold = 0.0
                        predictor.familiarity_scores = np.array([], dtype=np.float64)
                    if prototype_alpha is not None and eff_proto_alpha == 0.0:
                        predictor.prototype_threshold = 0.0
                        predictor.prototype_scores = np.array([], dtype=np.float64)
                    predictor.calibrate(
                        probs,
                        [str(label) for label in conf_labels],
                        familiarity_values=fam_vals,
                        familiarity_alpha=eff_fam_alpha,
                        embeddings=conf_embeddings if (eff_proto_alpha > 0.0 and len(conf_embeddings) > 0) else None,
                        prototypes=proto_mat if eff_proto_alpha > 0.0 else None,
                        prototype_alpha=eff_proto_alpha,
                    )
                    if mondrian is False:
                        predictor.class_calibration_scores = {}
                    scores = predictor.calibration_scores
                    k = int(math.ceil((len(scores) + 1) * (1.0 - alpha_val)))
                    new_q = 1.0 if k > len(scores) else float(scores[k - 1])
                    predictor.quantile = new_q
                elif isinstance(definition, MultiChoiceField):
                    probs = _stable_sigmoid(conf_logits / calibrator.temperature, temperature=1.0)
                    conf_targets = targets[conf_pos]
                    errors = np.where(conf_targets == 1, 1.0 - probs, probs)
                    predictor = self.conformal_predictors[name]
                    predictor.calibration_scores = np.sort(np.max(errors, axis=1))
                    predictor.is_calibrated = True
                    scores = predictor.calibration_scores
                    k = int(math.ceil((len(scores) + 1) * (1.0 - alpha_val)))
                    new_q = 1.0 if k > len(scores) else float(scores[k - 1])
                    predictor.quantile = new_q
                else:
                    probs = _stable_sigmoid(conf_logits.reshape(-1) / calibrator.temperature, temperature=1.0)
                    predictions = definition.min_value + (definition.max_value - definition.min_value) * probs
                    predictor = self.regression_conformal_predictors[name]
                    predictor.calibrate(predictions, conf_labels)
                    scores = predictor.residuals
                    k = int(math.ceil((len(scores) + 1) * (1.0 - alpha_val)))
                    upper = float(definition.max_value - definition.min_value)
                    new_q = upper if k > len(scores) else float(scores[k - 1])

                fam_thresh_val = float(getattr(predictor, "familiarity_threshold", 0.0)) if isinstance(definition, (ChoiceField, BooleanField)) else 0.0
                proto_thresh_val = float(getattr(predictor, "prototype_threshold", 0.0)) if isinstance(definition, (ChoiceField, BooleanField)) else 0.0
                c_scores_dict: Dict[str, Tuple[float, ...]] = {}
                if isinstance(definition, (ChoiceField, BooleanField)):
                    raw_c_scores = getattr(predictor, "class_calibration_scores", {})
                    if raw_c_scores:
                        c_scores_dict = {
                            str(k_cls): tuple(float(v) for v in vals)
                            for k_cls, vals in raw_c_scores.items()
                        }

                scores_tuple = tuple(float(score) for score in scores)
                if compiled is not None and hasattr(compiled, "calibration_scores"):
                    compiled.temperature = float(calibrator.temperature)
                    compiled.calibration_scores = scores_tuple
                    compiled.score_method = getattr(predictor, "score_method", "aps")
                    compiled.conformal_quantile = float(new_q)
                    if isinstance(definition, (ChoiceField, BooleanField)):
                        if mondrian is not None:
                            compiled.mondrian = bool(mondrian)
                        compiled.margin_threshold = float(
                            getattr(predictor, "calibrated_margin_threshold", getattr(compiled, "margin_threshold", 0.20))
                        )
                        compiled.familiarity_threshold = fam_thresh_val
                        compiled.prototype_threshold = proto_thresh_val
                        if getattr(predictor, "prototypes", None) is not None:
                            compiled.prototypes = np.asarray(predictor.prototypes, dtype=np.float32).copy()
                        p_scores = getattr(predictor, "prototype_scores", np.array([]))
                        compiled.prototype_scores = tuple(float(s) for s in p_scores) if len(p_scores) > 0 else ()
                        compiled.class_calibration_scores = c_scores_dict if c_scores_dict else {}

                report_fields[name] = RecalibrationMetrics(
                    field_name=name,
                    ece_before=init_ece,
                    ece_after=cal_ece,
                    nll_before=init_nll,
                    nll_after=cal_nll,
                    temperature_before=prev_temp,
                    temperature=float(calibrator.temperature),
                    qhat_before=prev_q,
                    qhat=float(new_q),
                    alpha=alpha_val,
                    n_examples=n_total,
                    brier_before=init_brier,
                    brier_after=cal_brier,
                    familiarity_threshold=fam_thresh_val,
                    prototype_threshold=proto_thresh_val,
                    calibration_scores=scores_tuple,
                    class_calibration_scores=c_scores_dict,
                )

            self.model_version = max(self.model_version, int(getattr(self.model, "model_version", 1))) + 1
            if hasattr(self.model, "model_version"):
                self.model.model_version = self.model_version
            if self.cache is not None:
                self.cache.clear()
            if hasattr(self.model, "cache") and getattr(self.model, "cache", None) is not None:
                self.model.cache.clear()

            f_list = list(report_fields.values())
            n_f_rep = max(1, len(f_list))
            return RecalibrationReport(
                ece_before=float(sum(m.ece_before for m in f_list)) / n_f_rep if f_list else 0.0,
                ece_after=float(sum(m.ece_after for m in f_list)) / n_f_rep if f_list else 0.0,
                nll_before=float(sum(m.nll_before for m in f_list)) / n_f_rep if f_list else 0.0,
                nll_after=float(sum(m.nll_after for m in f_list)) / n_f_rep if f_list else 0.0,
                temperature_before=float(sum(m.temperature_before for m in f_list)) / n_f_rep if f_list else 1.0,
                temperature=float(sum(m.temperature for m in f_list)) / n_f_rep if f_list else 1.0,
                qhat_before=float(sum(m.qhat_before for m in f_list)) / n_f_rep if f_list else 0.0,
                qhat=float(sum(m.qhat for m in f_list)) / n_f_rep if f_list else 0.0,
                alpha=alpha_val,
                n_examples=len(rows),
                per_field=report_fields,
                schema_name=self.schema.schema_name,
                brier_before=float(sum(m.brier_before for m in f_list)) / n_f_rep if f_list else 0.0,
                brier_after=float(sum(m.brier_after for m in f_list)) / n_f_rep if f_list else 0.0,
            )

    def evaluate_dataset(
        self,
        rows: Union[Sequence[Any], str, Path],
        *,
        alpha: Optional[float] = None,
        n_bins: int = 10,
        strict_mode: Optional[bool] = None,
        use_cache: bool = False,
    ) -> EvaluationReport:
        """Evaluate the engine on a labeled dataset and return an EvaluationReport."""
        eff_alpha = self.default_alpha if alpha is None else float(alpha)
        if (
            isinstance(alpha, bool)
            or not isinstance(eff_alpha, (int, float))
            or not math.isfinite(eff_alpha)
            or not (0.0 < eff_alpha < 1.0)
        ):
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")

        normalized_rows = _normalize_eval_rows(rows, self.schema)
        eff_strict = self.strict_mode if strict_mode is None else bool(strict_mode)

        decisions: List[DecisionResult] = [
            self.decide(
                prompt,
                telemetry=telemetry,
                alpha=eff_alpha,
                record_receipt=False,
                strict=eff_strict,
                use_cache=use_cache,
            )
            for prompt, _, telemetry in normalized_rows
        ]

        total_examples = len(normalized_rows)
        latencies = sorted(float(d.latency_ms) for d in decisions)
        percentiles = np.percentile(latencies, [50, 95, 99]) if latencies else [0.0, 0.0, 0.0]
        mean_lat = float(statistics.mean(latencies)) if latencies else 0.0
        p50_lat = float(percentiles[0])
        p95_lat = float(percentiles[1])
        p99_lat = float(percentiles[2])

        field_reports: Dict[str, FieldEvaluationMetrics] = {}
        global_abstention_reasons: Dict[str, int] = {}
        total_field_evals = 0
        total_set_size_sum = 0.0
        total_singleton_count = 0

        row_all_correct = [True] * total_examples
        row_all_covered = [True] * total_examples

        for fname, f_def in self.schema.fields.items():
            f_rows = [
                (idx, decisions[idx], normalized_rows[idx][1][fname])
                for idx in range(total_examples)
                if fname in normalized_rows[idx][1]
            ]
            if not f_rows:
                continue
            n_f = len(f_rows)
            f_correct = 0
            f_accepted = 0
            f_accepted_correct = 0
            f_abstained = 0
            f_unfamiliar = 0
            f_covered = 0
            f_set_size_sum = 0.0
            f_singleton = 0
            f_mae_sum = 0.0
            f_reasons: Dict[str, int] = {}

            for row_idx, res, gt_val in f_rows:
                is_unfamiliar = fname in res.abstention_reasons
                is_abstain = (not is_unfamiliar) and (fname in res.ambiguous_fields)
                is_accepted = (not is_unfamiliar) and (not is_abstain)

                if is_unfamiliar:
                    f_unfamiliar += 1
                    r_key = str(res.abstention_reasons[fname])
                    f_reasons[r_key] = f_reasons.get(r_key, 0) + 1
                elif is_abstain:
                    f_abstained += 1
                    r_key = "conformal_ambiguity"
                    f_reasons[r_key] = f_reasons.get(r_key, 0) + 1
                else:
                    f_accepted += 1

                pred_val = res.values.get(fname)
                c_set = res.conformal_sets.get(fname, [])

                if isinstance(f_def, ChoiceField):
                    correct_i = str(pred_val) == str(gt_val)
                    covered_i = str(gt_val) in c_set
                    set_size_i = float(len(c_set))
                    singleton_i = len(c_set) == 1
                elif isinstance(f_def, BooleanField):
                    correct_i = bool(pred_val) == bool(gt_val)
                    gt_str = "True" if bool(gt_val) else "False"
                    covered_i = (gt_str in c_set) or (bool(gt_val) in c_set)
                    set_size_i = float(len(c_set))
                    singleton_i = len(c_set) == 1
                elif isinstance(f_def, MultiChoiceField):
                    pred_set = set(pred_val) if isinstance(pred_val, (list, tuple, set)) else {pred_val}
                    gt_set = set(gt_val) if isinstance(gt_val, (list, tuple, set)) else {gt_val}
                    correct_i = pred_set == gt_set
                    mc_cp = self.conformal_predictors.get(fname)
                    if is_unfamiliar:
                        covered_i = False
                    elif mc_cp is not None and mc_cp.is_calibrated and len(mc_cp.calibration_scores) > 0:
                        mc_scores = mc_cp.calibration_scores
                        k_mc = int(math.ceil((len(mc_scores) + 1) * (1.0 - eff_alpha)))
                        if k_mc > len(mc_scores):
                            covered_i = True
                        else:
                            q_mc = float(mc_scores[k_mc - 1])
                            row_probs = res.probabilities.get(fname, {})
                            covered_i = all(
                                ((1.0 - float(row_probs.get(opt, 0.5))) <= q_mc + 1e-6)
                                if opt in gt_set
                                else (float(row_probs.get(opt, 0.5)) <= q_mc + 1e-6)
                                for opt in f_def.options
                            )
                    else:
                        covered_i = gt_set.issubset(set(c_set))
                    set_size_i = float(len(c_set))
                    singleton_i = is_accepted
                else:
                    val_span = max(1e-6, float(f_def.max_value - f_def.min_value))
                    covered_i = False
                    rcp = self.regression_conformal_predictors.get(fname)
                    if is_unfamiliar or not c_set:
                        covered_i = False
                    elif rcp is not None and rcp.is_calibrated:
                        interval = rcp.predict_interval(float(pred_val), alpha=eff_alpha, strict=eff_strict)
                        covered_i = (interval.lower_bound - 1e-6) <= float(gt_val) <= (interval.upper_bound + 1e-6)
                    elif isinstance(c_set[0], str) and c_set[0].startswith("[") and c_set[0].endswith("]"):
                        try:
                            parts = c_set[0][1:-1].split(",")
                            lo_v, hi_v = float(parts[0].strip()), float(parts[1].strip())
                            covered_i = (lo_v - 1e-4) <= float(gt_val) <= (hi_v + 1e-4)
                        except (ValueError, IndexError):
                            covered_i = False
                    abs_err = abs(float(pred_val) - float(gt_val))
                    f_mae_sum += abs_err
                    correct_i = abs_err <= 0.10 * val_span
                    set_size_i = float(len(c_set))
                    singleton_i = is_accepted

                if correct_i:
                    f_correct += 1
                    if is_accepted:
                        f_accepted_correct += 1
                else:
                    row_all_correct[row_idx] = False
                if covered_i:
                    f_covered += 1
                else:
                    row_all_covered[row_idx] = False
                f_set_size_sum += set_size_i
                if singleton_i:
                    f_singleton += 1

            if isinstance(f_def, ChoiceField):
                probs_mat = np.asarray(
                    [
                        [float(res.probabilities.get(fname, {}).get(opt, 0.0)) for opt in f_def.options]
                        for _, res, _ in f_rows
                    ],
                    dtype=np.float64,
                )
                row_sums = np.sum(probs_mat, axis=1, keepdims=True)
                probs_mat = np.where(row_sums > 1e-12, probs_mat / np.where(row_sums > 1e-12, row_sums, 1.0), 1.0 / max(1, len(f_def.options)))
                targets_arr = np.asarray([f_def.options.index(gt_val) for _, _, gt_val in f_rows], dtype=np.int64)
                f_ece = float(compute_ece_and_bins(probs_mat, targets_arr, n_bins=n_bins)[0])
                f_brier = float(compute_brier_decomposition(probs_mat, targets_arr, n_bins=n_bins).empirical_brier)
                f_nll = float(compute_nll(probs_mat, targets_arr))
            elif isinstance(f_def, BooleanField):
                probs_pos = np.asarray(
                    [float(res.probabilities.get(fname, {}).get("True", 0.5)) for _, res, _ in f_rows],
                    dtype=np.float64,
                )
                probs_2d = np.column_stack([1.0 - probs_pos, probs_pos])
                targets_arr = np.asarray([1 if bool(gt_val) else 0 for _, _, gt_val in f_rows], dtype=np.int64)
                f_ece = float(compute_ece_and_bins(probs_2d, targets_arr, n_bins=n_bins)[0])
                f_brier = float(compute_brier_decomposition(probs_2d, targets_arr, n_bins=n_bins).empirical_brier)
                f_nll = float(compute_nll(probs_2d, targets_arr))
            elif isinstance(f_def, MultiChoiceField):
                probs_flat = np.asarray(
                    [
                        float(res.probabilities.get(fname, {}).get(opt, 0.5))
                        for _, res, _ in f_rows
                        for opt in f_def.options
                    ],
                    dtype=np.float64,
                )
                probs_2d = np.column_stack([1.0 - probs_flat, probs_flat])
                targets_flat = np.asarray(
                    [
                        1 if opt in (gt_val if isinstance(gt_val, (list, tuple, set)) else (gt_val,)) else 0
                        for _, _, gt_val in f_rows
                        for opt in f_def.options
                    ],
                    dtype=np.int64,
                )
                f_ece = float(compute_ece_and_bins(probs_2d, targets_flat, n_bins=n_bins)[0])
                f_brier = float(compute_brier_decomposition(probs_2d, targets_flat, n_bins=n_bins).empirical_brier)
                f_nll = float(compute_nll(probs_2d, targets_flat))
            else:
                val_span = max(1e-6, float(f_def.max_value - f_def.min_value))
                probs_pos = np.clip(
                    np.asarray(
                        [float(res.probabilities.get(fname, {}).get("score_ratio", 0.5)) for _, res, _ in f_rows],
                        dtype=np.float64,
                    ),
                    1e-12,
                    1.0 - 1e-12,
                )
                y_ratio = np.clip(
                    np.asarray(
                        [(float(gt_val) - float(f_def.min_value)) / val_span for _, _, gt_val in f_rows],
                        dtype=np.float64,
                    ),
                    0.0,
                    1.0,
                )
                f_ece = float(np.mean(np.abs(probs_pos - y_ratio)))
                f_brier = float(np.mean((probs_pos - y_ratio) ** 2))
                f_nll = float(-np.mean(y_ratio * np.log(probs_pos) + (1.0 - y_ratio) * np.log(1.0 - probs_pos)))

            f_cov_rate = float(f_accepted) / float(n_f)
            f_abs_rate = float(f_abstained) / float(n_f)
            f_unf_rate = float(f_unfamiliar) / float(n_f)
            f_acc = float(f_correct) / float(n_f)
            f_acc_accepted = float(f_accepted_correct) / float(f_accepted) if f_accepted > 0 else 0.0

            field_reports[fname] = FieldEvaluationMetrics(
                field_name=fname,
                field_type=f_def.field_type,
                total_examples=n_f,
                accepted_examples=f_accepted,
                abstained_examples=f_abstained,
                unfamiliar_examples=f_unfamiliar,
                accuracy=f_acc,
                accepted_accuracy=f_acc_accepted,
                conformal_coverage=float(f_covered) / float(n_f),
                average_set_size=float(f_set_size_sum) / float(n_f),
                singleton_rate=float(f_singleton) / float(n_f),
                ece=float(f_ece),
                brier_score=float(f_brier),
                nll=float(f_nll),
                coverage=f_cov_rate,
                abstain_rate=f_abs_rate,
                unfamiliar_rate=f_unf_rate,
                abstention_reasons=f_reasons,
                mae=float(f_mae_sum) / float(n_f) if isinstance(f_def, ScoreField) else 0.0,
            )

            total_field_evals += n_f
            total_set_size_sum += f_set_size_sum
            total_singleton_count += f_singleton

        row_accepted = 0
        row_accepted_correct = 0
        row_abstained = 0
        row_unfamiliar = 0
        for row_idx, res in enumerate(decisions):
            if res.abstention_reasons:
                row_unfamiliar += 1
                primary_r = next(iter(res.abstention_reasons.values()))
                global_abstention_reasons[str(primary_r)] = global_abstention_reasons.get(str(primary_r), 0) + 1
            elif res.is_ambiguous:
                row_abstained += 1
                global_abstention_reasons["conformal_ambiguity"] = (
                    global_abstention_reasons.get("conformal_ambiguity", 0) + 1
                )
            else:
                row_accepted += 1
                if row_all_correct[row_idx]:
                    row_accepted_correct += 1

        overall_cov = float(row_accepted) / float(total_examples)
        overall_abs = float(row_abstained) / float(total_examples)
        overall_unf = float(row_unfamiliar) / float(total_examples)

        denom_f = max(1, total_field_evals)
        overall_acc = float(sum(1 for ok in row_all_correct if ok)) / float(total_examples)
        accepted_acc = (
            float(row_accepted_correct) / float(row_accepted)
            if row_accepted > 0
            else 0.0
        )
        overall_emp_cov = float(sum(1 for cov in row_all_covered if cov)) / float(total_examples)
        mean_ece = (
            float(statistics.mean(m.ece for m in field_reports.values()))
            if field_reports
            else 0.0
        )
        mean_brier = (
            float(statistics.mean(m.brier_score for m in field_reports.values()))
            if field_reports
            else 0.0
        )
        mean_nll = (
            float(statistics.mean(m.nll for m in field_reports.values()))
            if field_reports
            else 0.0
        )

        return EvaluationReport(
            schema_name=self.schema.schema_name,
            alpha=eff_alpha,
            total_examples=total_examples,
            accepted_examples=row_accepted,
            abstained_examples=row_abstained,
            unfamiliar_examples=row_unfamiliar,
            overall_accuracy=overall_acc,
            accepted_accuracy=accepted_acc,
            conformal_coverage=overall_emp_cov,
            average_set_size=float(total_set_size_sum) / float(denom_f),
            singleton_rate=float(total_singleton_count) / float(denom_f),
            ece=mean_ece,
            brier_score=mean_brier,
            nll=mean_nll,
            coverage=overall_cov,
            abstain_rate=overall_abs,
            unfamiliar_rate=overall_unf,
            abstention_reasons=global_abstention_reasons,
            mean_latency_ms=round(mean_lat, 4),
            p50_latency_ms=round(p50_lat, 4),
            p95_latency_ms=round(p95_lat, 4),
            p99_latency_ms=round(p99_lat, 4),
            fields=field_reports,
            schema_digest=self.schema.schema_digest(),
            strict_mode=eff_strict,
        )

    def decide(
        self,
        prompt: str,
        *,
        telemetry: Optional[Any] = None,
        embedding: Optional[np.ndarray] = None,
        alpha: Optional[float] = None,
        record_receipt: bool = True,
        ledger: Optional[ActionLedger] = None,
        margin_threshold: Optional[float] = None,
        relative_odds_ratio: Optional[float] = None,
        confidence_floor_tau0: Optional[float] = None,
        recency_weighted: Optional[bool] = None,
        fail_closed_ledger: Optional[bool] = None,
        strict: Optional[bool] = None,
        policy_scope: Optional[str] = None,
        use_cache: Optional[bool] = None,
    ) -> DecisionResult:
        """Evaluate a prompt in one non-autoregressive pass.

        With ``record_receipt=False``, skip receipt creation/signing and return
        ``receipt=None``. This applies to both fresh decisions and cache hits.
        """
        if self.auto_reload and self.model_path is not None:
            self.reload_if_changed()
        # A decision, its uncertainty, and its receipt use one model snapshot.
        with self._lock, getattr(self.model, "_lock", nullcontext()):
            recency_weighted = self.recency_weighted if recency_weighted is None else bool(recency_weighted)
            eff_use_cache = self.use_cache if use_cache is None else bool(use_cache)
            version = getattr(self.model, "model_version", self.model_version)
            if version != self.model_version:
                if hasattr(self.model, "_field_heads"):
                    self._sync_from_model(self.model)
                self.model_version = version
                for name in self.schema.fields:
                    head = getattr(self.model, "heads", {}).get(name)
                    if not getattr(head, "calibration_scores", ()):
                        self.calibrators[name] = DecisionCalibrator()
                        if name in self.conformal_predictors:
                            cp_reset = self.conformal_predictors[name]
                            cp_reset.is_calibrated = False
                            cp_reset.familiarity_threshold = 0.0
                            cp_reset.prototype_threshold = 0.0
                            cp_reset.familiarity_scores = np.array([], dtype=np.float64)
                            cp_reset.prototype_scores = np.array([], dtype=np.float64)
                            cp_reset.class_calibration_scores = {}
                        if name in self.regression_conformal_predictors:
                            self.regression_conformal_predictors[name].is_calibrated = False
            if alpha is None:
                alpha = self.default_alpha
            if not isinstance(prompt, str):
                raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}")
            if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not (0.0 < alpha < 1.0) or math.isnan(alpha) or math.isinf(alpha):
                raise ValueError(f"alpha must be in (0, 1), got {alpha}")
            if margin_threshold is not None and (margin_threshold < 0.0 or math.isnan(margin_threshold) or math.isinf(margin_threshold)):
                raise ValueError(f"margin_threshold must be non-negative, got {margin_threshold}")
            if relative_odds_ratio is not None and (relative_odds_ratio <= 0.0 or math.isnan(relative_odds_ratio) or math.isinf(relative_odds_ratio)):
                raise ValueError(f"relative_odds_ratio must be strictly positive, got {relative_odds_ratio}")
            if confidence_floor_tau0 is not None and (confidence_floor_tau0 < 0.0 or math.isnan(confidence_floor_tau0) or math.isinf(confidence_floor_tau0)):
                raise ValueError(f"confidence_floor_tau0 must be non-negative, got {confidence_floor_tau0}")
            if embedding is not None:
                if not isinstance(embedding, (np.ndarray, list, tuple)):
                    raise TypeError(f"embedding must be array-like, got {type(embedding).__name__}")
                arr_emb = np.asarray(embedding, dtype=np.float32).flatten()
                if not np.all(np.isfinite(arr_emb)):
                    raise ValueError("embedding must contain only finite numbers (no NaN or Inf)")
                if arr_emb.shape[0] != self.dimension:
                    raise ValueError(f"embedding dimension mismatch: expected {self.dimension}, got {arr_emb.shape[0]}")
                query_emb = arr_emb
            else:
                query_emb = None
            if telemetry is not None:
                if not isinstance(telemetry, (dict, list, tuple, np.ndarray, float, int)):
                    raise TypeError(f"telemetry must be a dict, sequence, ndarray, or numeric, got {type(telemetry).__name__}")

            t_start = time.perf_counter()

            is_fail_closed = self.fail_closed_ledger if fail_closed_ledger is None else bool(fail_closed_ledger)
            is_strict = self.strict_mode if strict is None else bool(strict)
            eff_scope = self.policy_scope if policy_scope is None else str(policy_scope)
            eff_m_thresh = margin_threshold if margin_threshold is not None else self.margin_threshold
            eff_gamma = relative_odds_ratio if relative_odds_ratio is not None else self.relative_odds_ratio
            eff_tau0 = confidence_floor_tau0 if confidence_floor_tau0 is not None else self.confidence_floor_tau0

            active_ledger = ledger or self.ledger
            truth_ledger_head = ""
            ledger_record_id: Optional[str] = None

            if active_ledger is not None:
                try:
                    truth_ledger_head = active_ledger.head_hash()
                except Exception as ex:
                    if is_fail_closed:
                        raise LedgerWriteError(f"Fail-closed ledger inspection failed: {ex}") from ex
                    truth_ledger_head = ""

            # 1. Cache lookup with request and model context
            if eff_use_cache:
                # These digests bind cache entries; uncached decisions do not use them.
                m_dig = self._model_digest()
                p_dig = self._projector_digest()
                c_dig = self._calibration_digest()
                cached = self.cache.get(
                    prompt,
                    embedding=embedding,
                    telemetry=telemetry,
                    schema_digest=self.schema.schema_digest(),
                    model_version=self.model_version,
                    policy_scope=eff_scope,
                    alpha=alpha,
                    margin_threshold=eff_m_thresh if self.enable_margin_gating else 0.0,
                    strict=is_strict,
                    relative_odds_ratio=eff_gamma,
                    confidence_floor_tau0=eff_tau0,
                    recency_weighted=recency_weighted,
                    model_digest=m_dig,
                    projector_digest=p_dig,
                    calibration_digest=c_dig,
                    policy_epoch=self.policy_epoch,
                    enforce_durability=is_fail_closed,
                )
                if cached is None:
                    if query_emb is None:
                        query_emb = self.encode(prompt, telemetry=telemetry, recency_weighted=recency_weighted)
                    cached = self.cache.get(
                        prompt,
                        embedding=query_emb,
                        telemetry=telemetry,
                        schema_digest=self.schema.schema_digest(),
                        model_version=self.model_version,
                        policy_scope=eff_scope,
                        alpha=alpha,
                        margin_threshold=eff_m_thresh if self.enable_margin_gating else 0.0,
                        strict=is_strict,
                        relative_odds_ratio=eff_gamma,
                        confidence_floor_tau0=eff_tau0,
                        recency_weighted=recency_weighted,
                        model_digest=m_dig,
                        projector_digest=p_dig,
                        calibration_digest=c_dig,
                        policy_epoch=self.policy_epoch,
                        enforce_durability=is_fail_closed,
                    )

                if cached is not None:
                    entry, sim = cached
                    if isinstance(entry.result, DecisionResult):
                        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                        hit_is_ambiguous = bool(getattr(entry.result, "is_ambiguous", False))
                        hit_ambiguous_fields = copy.deepcopy(getattr(entry.result, "ambiguous_fields", []))
                        hit_escalated_fields = copy.deepcopy(getattr(entry.result, "escalated_fields", []))

                        hit_receipt = None
                        if record_receipt:
                            hit_receipt = create_decision_receipt(
                                schema_name=self.schema.schema_name,
                                schema_digest=self.schema.schema_digest(),
                                prompt=prompt,
                                values=copy.deepcopy(entry.result.values),
                                confidences=copy.deepcopy(entry.result.confidences),
                                conformal_sets=copy.deepcopy(entry.result.conformal_sets),
                                probabilities=copy.deepcopy(entry.result.probabilities),
                                latency_ms=elapsed_ms,
                                is_ambiguous=hit_is_ambiguous,
                                truth_ledger_head=truth_ledger_head,
                                ledger_record_id=None,
                                signing_key=self.signing_key,
                            )
                            if active_ledger is not None:
                                try:
                                    ledger_record_id = active_ledger.record_decision_receipt(hit_receipt)
                                    hit_receipt = hit_receipt.with_ledger_record(ledger_record_id, self.signing_key)
                                except Exception as ex:
                                    if is_fail_closed:
                                        raise LedgerWriteError(f"Fail-closed ledger recording failed on cache hit: {ex}") from ex
                                    pass

                        return DecisionResult(
                            schema_name=self.schema.schema_name,
                            schema_digest=self.schema.schema_digest(),
                            prompt=prompt,
                            values=copy.deepcopy(entry.result.values),
                            confidences=copy.deepcopy(entry.result.confidences),
                            conformal_sets=copy.deepcopy(entry.result.conformal_sets),
                            probabilities=copy.deepcopy(entry.result.probabilities),
                            is_ambiguous=hit_is_ambiguous,
                            latency_ms=elapsed_ms,
                            alpha=alpha,
                            receipt=hit_receipt,
                            margins=copy.deepcopy(entry.result.margins),
                            margin_thresholds=copy.deepcopy(entry.result.margin_thresholds),
                            margin_gate_active=copy.deepcopy(entry.result.margin_gate_active),
                            odds_ratios=copy.deepcopy(getattr(entry.result, "odds_ratios", {})),
                            relative_odds_ratio_thresholds=copy.deepcopy(getattr(entry.result, "relative_odds_ratio_thresholds", {})),
                            confidence_floors=copy.deepcopy(getattr(entry.result, "confidence_floors", {})),
                            ambiguous_fields=hit_ambiguous_fields,
                            escalated_fields=hit_escalated_fields,
                            telemetry=copy.deepcopy(telemetry) if telemetry is not None else None,
                            is_cache_hit=True,
                            embedding=entry.embedding if entry.embedding is not None else query_emb,
                            familiarity_score=getattr(entry.result, "familiarity_score", None),
                            metadata=copy.deepcopy(getattr(entry.result, "metadata", {})),
                        )

            with self._lock:
                user_emb = query_emb if embedding is not None else None
                try:
                    raw_result = self.model.forward_single(
                        prompt,
                        telemetry=telemetry,
                        embedding=user_emb,
                        recency_weighted=recency_weighted,
                        use_cache=False,
                    )
                except TypeError:
                    try:
                        raw_result = self.model.forward_single(
                            prompt,
                            telemetry=telemetry,
                            embedding=user_emb,
                            recency_weighted=recency_weighted,
                        )
                    except TypeError:
                        raw_result = self.model.forward_single(
                            prompt,
                            telemetry=telemetry,
                            embedding=user_emb,
                        )

            values: Dict[str, Any] = {}
            confidences: Dict[str, float] = {}
            conformal_sets: Dict[str, List[str]] = {}
            probabilities: Dict[str, Dict[str, float]] = {}
            margins: Dict[str, float] = {}
            margin_thresholds: Dict[str, float] = {}
            margin_gate_active: Dict[str, bool] = {}
            odds_ratios: Dict[str, float] = {}
            relative_odds_ratio_thresholds: Dict[str, float] = {}
            confidence_floors: Dict[str, float] = {}
            ambiguous_fields: List[str] = []
            escalated_fields: List[str] = []
            is_ambiguous = False

            from system1.core.text import TfidfProjector
            prompt_raw_norm: Optional[float] = getattr(raw_result, "raw_norm", None)
            needs_raw_norm = any(
                float(getattr(cp, "familiarity_threshold", 0.0)) > 0.0
                for cp in self.conformal_predictors.values()
            ) or any(
                float(getattr(ch, "familiarity_threshold", 0.0)) > 0.0
                for ch in getattr(self.model, "heads", {}).values()
            )
            if prompt_raw_norm is None and embedding is None and needs_raw_norm and hasattr(self.projector, "raw_norm"):
                try:
                    prompt_raw_norm = float(self.projector.raw_norm(prompt))
                except Exception:
                    prompt_raw_norm = None

            learned_overrides: Dict[str, Any] = {}
            if embedding is None and telemetry is None and isinstance(prompt, str) and prompt.strip():
                norm_key = " ".join(prompt.strip().casefold().split())
                model_ov = getattr(self.model, "_learned_prompt_overrides", {})
                model_rec = getattr(self.model, "_learned_prompt_override_recency", {})
                if isinstance(model_ov, Mapping) and norm_key in model_ov:
                    if model_rec.get(norm_key, bool(self.recency_weighted)) == bool(recency_weighted):
                        learned_overrides.update(model_ov[norm_key])
                eng_ov = getattr(self, "_learned_prompt_overrides", {})
                eng_rec = getattr(self, "_learned_prompt_override_recency", {})
                if isinstance(eng_ov, Mapping) and norm_key in eng_ov:
                    if eng_rec.get(norm_key, bool(self.recency_weighted)) == bool(recency_weighted):
                        learned_overrides.update(eng_ov[norm_key])

            prototype_similarities: Dict[str, float] = {}
            abstention_reasons: Dict[str, str] = {}

            for name, f_def in self.schema.fields.items():
                raw_eval = raw_result.fields[name]
                calibrator = self.calibrators[name]
                has_learned_override = name in learned_overrides

                proto_sim: Optional[float] = None
                if not has_learned_override:
                    if hasattr(self.model, "compute_prototype_similarity") and callable(self.model.compute_prototype_similarity):
                        try:
                            proto_sim = self.model.compute_prototype_similarity(name, raw_result.embedding)
                        except Exception:
                            proto_sim = None
                    if proto_sim is None:
                        cp_for_proto = self.conformal_predictors.get(name)
                        if cp_for_proto is not None and getattr(cp_for_proto, "prototypes", None) is not None:
                            try:
                                proto_sim = cp_for_proto.compute_prototype_similarity(raw_result.embedding)
                            except Exception:
                                proto_sim = None
                    if proto_sim is not None:
                        prototype_similarities[name] = float(proto_sim)

                if isinstance(f_def, ChoiceField):
                    calibrated_p = calibrator.calibrate_logits(raw_eval.logits)
                    best_idx = int(np.argmax(calibrated_p))
                    selected_choice = f_def.options[best_idx]
                    conf = float(calibrated_p[best_idx])

                    prob_dict = {opt: float(calibrated_p[i]) for i, opt in enumerate(f_def.options)}
                    values[name] = selected_choice
                    confidences[name] = conf
                    probabilities[name] = prob_dict

                    conformal = self.conformal_predictors.get(name)
                    is_zero_emb = not bool(np.any(raw_result.embedding))
                    override_matched = has_learned_override and (
                        is_zero_emb or selected_choice == learned_overrides.get(name)
                    )
                    if conformal is not None and not override_matched:
                        try:
                            cset = conformal.predict_set(
                                calibrated_p,
                                alpha=alpha,
                                margin_threshold=eff_m_thresh if (self.enable_margin_gating and not is_strict) else 0.0,
                                relative_odds_ratio=eff_gamma if not is_strict else None,
                                confidence_floor_tau0=eff_tau0 if not is_strict else None,
                                strict=is_strict,
                                familiarity_score=None if has_learned_override else prompt_raw_norm,
                                prototype_similarity=None if has_learned_override else proto_sim,
                            )
                        except TypeError:
                            cset = conformal.predict_set(calibrated_p, alpha=alpha)
                        conformal_sets[name] = list(cset.prediction_set)
                        margins[name] = getattr(cset, "margin", 1.0)
                        margin_thresholds[name] = getattr(cset, "margin_threshold", 0.0)
                        margin_gate_active[name] = getattr(cset, "margin_gate_active", False)
                        odds_ratios[name] = getattr(cset, "odds_ratio", 1.0)
                        relative_odds_ratio_thresholds[name] = getattr(cset, "relative_odds_ratio_threshold", 0.0)
                        confidence_floors[name] = getattr(cset, "confidence_floor", 0.0)
                        cset_reason = getattr(cset, "abstention_reason", None)
                        if cset_reason:
                            abstention_reasons[name] = str(cset_reason)

                        is_empty_set = getattr(cset, "is_empty", len(cset.prediction_set) == 0)
                        cardinality = len(cset.prediction_set)
                        if is_strict:
                            field_ambiguous = (cardinality != 1) or cset.is_ambiguous or is_empty_set
                        else:
                            field_ambiguous = is_empty_set or cset.is_ambiguous

                        if field_ambiguous:
                            ambiguous_fields.append(name)
                            if getattr(f_def, "escalate_on_ambiguity", True):
                                is_ambiguous = True
                                escalated_fields.append(name)
                    else:
                        conformal_sets[name] = [selected_choice]
                        margins[name] = 1.0

                elif isinstance(f_def, BooleanField):
                    calibrated_p = calibrator.calibrate_logits(raw_eval.logits)
                    prob_true = float(calibrated_p[1])
                    val = bool(prob_true >= f_def.threshold)
                    conf = prob_true if val else (1.0 - prob_true)

                    values[name] = val
                    confidences[name] = conf
                    probabilities[name] = {"False": float(calibrated_p[0]), "True": prob_true}

                    conformal = self.conformal_predictors.get(name)
                    is_zero_emb = not bool(np.any(raw_result.embedding))
                    override_matched = has_learned_override and (
                        is_zero_emb or val == bool(learned_overrides.get(name))
                    )
                    if conformal is not None and not override_matched:
                        try:
                            cset = conformal.predict_set(
                                calibrated_p,
                                alpha=alpha,
                                margin_threshold=eff_m_thresh if (self.enable_margin_gating and not is_strict) else 0.0,
                                relative_odds_ratio=eff_gamma if not is_strict else None,
                                confidence_floor_tau0=eff_tau0 if not is_strict else None,
                                strict=is_strict,
                                familiarity_score=None if has_learned_override else prompt_raw_norm,
                                prototype_similarity=None if has_learned_override else proto_sim,
                            )
                        except TypeError:
                            cset = conformal.predict_set(calibrated_p, alpha=alpha)
                        conformal_sets[name] = list(cset.prediction_set)
                        margins[name] = getattr(cset, "margin", 1.0)
                        margin_thresholds[name] = getattr(cset, "margin_threshold", 0.0)
                        margin_gate_active[name] = getattr(cset, "margin_gate_active", False)
                        odds_ratios[name] = getattr(cset, "odds_ratio", 1.0)
                        relative_odds_ratio_thresholds[name] = getattr(cset, "relative_odds_ratio_threshold", 0.0)
                        confidence_floors[name] = getattr(cset, "confidence_floor", 0.0)
                        cset_reason = getattr(cset, "abstention_reason", None)
                        if cset_reason:
                            abstention_reasons[name] = str(cset_reason)

                        is_empty_set = getattr(cset, "is_empty", len(cset.prediction_set) == 0)
                        cardinality = len(cset.prediction_set)
                        if is_strict:
                            field_ambiguous = (cardinality != 1) or cset.is_ambiguous or is_empty_set
                        else:
                            field_ambiguous = is_empty_set or cset.is_ambiguous

                        if field_ambiguous:
                            ambiguous_fields.append(name)
                            if getattr(f_def, "escalate_on_ambiguity", True):
                                is_ambiguous = True
                                escalated_fields.append(name)
                    else:
                        conformal_sets[name] = ["True" if val else "False"]
                        margins[name] = 1.0

                elif isinstance(f_def, MultiChoiceField):
                    calibrated_temp = calibrator.temperature
                    scaled_logits = raw_eval.logits / max(1e-4, calibrated_temp)
                    probs = _stable_sigmoid(scaled_logits, temperature=1.0)
                    selected = tuple(
                        f_def.options[i] for i, p in enumerate(probs) if p >= f_def.threshold
                    )
                    conf = float(np.mean([max(p, 1.0 - p) for p in probs])) if len(probs) > 0 else 1.0
                    values[name] = selected
                    confidences[name] = conf
                    probabilities[name] = {opt: float(probs[i]) for i, opt in enumerate(f_def.options)}
                    conformal_sets[name] = list(selected)

                    eff_boundary_margin = eff_m_thresh if (self.enable_margin_gating and eff_m_thresh is not None) else 0.05
                    is_boundary_uncertain = any(
                        abs(p - f_def.threshold) < eff_boundary_margin
                        for p in probs
                    )
                    mc_ambiguous = False
                    if is_strict and not has_learned_override:
                        conformal = self.conformal_predictors[name]
                        scores = conformal.calibration_scores
                        k = int(math.ceil((len(scores) + 1) * (1.0 - alpha)))
                        if not conformal.is_calibrated or k > len(scores):
                            conformal_sets[name] = list(f_def.options)
                            mc_ambiguous = True
                        else:
                            # Compiler scores bound the largest error over all labels.
                            # Each label has a set of possible binary assignments. Only
                            # a unique complete assignment can be accepted, including {}.
                            q = float(scores[k - 1])
                            can_include = (1.0 - probs) <= q + 1e-6
                            can_exclude = probs <= q + 1e-6
                            conformal_sets[name] = [opt for opt, possible in zip(f_def.options, can_include) if possible]
                            unique = np.logical_xor(can_include, can_exclude)
                            consistent = can_include == np.array([opt in selected for opt in f_def.options])
                            mc_ambiguous = not bool(np.all(unique & consistent))

                    if mc_ambiguous:
                        ambiguous_fields.append(name)
                        if getattr(f_def, "escalate_on_ambiguity", True):
                            is_ambiguous = True
                            escalated_fields.append(name)

                elif isinstance(f_def, ScoreField):
                    calibrator = self.calibrators.get(name)
                    calibrated_temp = calibrator.temperature if calibrator is not None else 1.0
                    scaled_logit = float(raw_eval.logits[0] / max(1e-4, calibrated_temp))
                    prob = float(_stable_sigmoid(np.array([scaled_logit]), temperature=1.0)[0])
                    val = f_def.min_value + (f_def.max_value - f_def.min_value) * prob
                    conf = float(getattr(raw_eval, "confidence", max(prob, 1.0 - prob)))
                    values[name] = val
                    confidences[name] = conf
                    probabilities[name] = {"score_ratio": prob}
                    reg_conformal = self.regression_conformal_predictors.get(name)
                    is_uncalibrated = reg_conformal is None or not reg_conformal.is_calibrated
                    if not is_uncalibrated and not has_learned_override:
                        interval = reg_conformal.predict_interval(val, alpha=alpha, strict=is_strict)
                        conformal_sets[name] = [interval.to_interval_string()]
                        interval_margin = interval.margin
                    else:
                        if is_strict and not has_learned_override:
                            interval_margin = f_def.max_value - f_def.min_value
                            low_b = f_def.min_value
                            high_b = f_def.max_value
                        else:
                            interval_margin = (f_def.max_value - f_def.min_value) * ((1.0 - alpha) / 2.0) if not has_learned_override else 0.0
                            low_b = max(f_def.min_value, val - interval_margin)
                            high_b = min(f_def.max_value, val + interval_margin)
                        conformal_sets[name] = [f"[{low_b:.4f}, {high_b:.4f}]"]

                    score_range = max(1e-6, f_def.max_value - f_def.min_value)
                    normalized_margin = interval_margin / score_range
                    score_ambiguous = False
                    if is_strict and not has_learned_override:
                        eff_score_thresh = eff_m_thresh if (self.enable_margin_gating and eff_m_thresh is not None) else 0.5
                        score_ambiguous = is_uncalibrated or (normalized_margin > eff_score_thresh)

                    if score_ambiguous:
                        ambiguous_fields.append(name)
                        if getattr(f_def, "escalate_on_ambiguity", True):
                            is_ambiguous = True
                            escalated_fields.append(name)

            # Unknown vocabulary carries no evidence. Do not accept a class prior
            # as a taught answer when a fixed-vocabulary skill sees no features.
            familiarity_triggered = False
            prototype_triggered = False
            if isinstance(self.projector, TfidfProjector) and not np.any(raw_result.embedding):
                for name, definition in self.schema.fields.items():
                    if name in learned_overrides:
                        continue
                    familiarity_triggered = True
                    conformal_sets[name] = []
                    abstention_reasons.setdefault(name, "low_feature_familiarity")
                    if name not in ambiguous_fields:
                        ambiguous_fields.append(name)
                    if getattr(definition, "escalate_on_ambiguity", True):
                        is_ambiguous = True
                        if name not in escalated_fields:
                            escalated_fields.append(name)
            if prompt_raw_norm is not None:
                for name, definition in self.schema.fields.items():
                    if name in learned_overrides:
                        continue
                    head_obj = self.model.heads.get(name) if hasattr(self.model, "heads") else None
                    cp_obj = self.conformal_predictors.get(name)
                    fam_t = max(
                        float(getattr(head_obj, "familiarity_threshold", 0.0)) if head_obj is not None else 0.0,
                        float(getattr(cp_obj, "familiarity_threshold", 0.0)) if cp_obj is not None else 0.0,
                    )
                    if fam_t > 0.0 and prompt_raw_norm < fam_t - 1e-12:
                        familiarity_triggered = True
                        conformal_sets[name] = []
                        abstention_reasons.setdefault(name, "low_feature_familiarity")
                        if name not in ambiguous_fields:
                            ambiguous_fields.append(name)
                        if getattr(definition, "escalate_on_ambiguity", True):
                            is_ambiguous = True
                            if name not in escalated_fields:
                                escalated_fields.append(name)
            for name, definition in self.schema.fields.items():
                if name in learned_overrides:
                    continue
                head_obj = self.model.heads.get(name) if hasattr(self.model, "heads") else None
                cp_obj = self.conformal_predictors.get(name)
                proto_t = max(
                    float(getattr(head_obj, "prototype_threshold", 0.0)) if head_obj is not None else 0.0,
                    float(getattr(cp_obj, "prototype_threshold", 0.0)) if cp_obj is not None else 0.0,
                )
                p_sim = prototype_similarities.get(name)
                if proto_t > 0.0 and p_sim is not None and p_sim < proto_t - 1e-6:
                    prototype_triggered = True
                    conformal_sets[name] = []
                    abstention_reasons.setdefault(name, "low_semantic_familiarity")
                    if name not in ambiguous_fields:
                        ambiguous_fields.append(name)
                    if getattr(definition, "escalate_on_ambiguity", True):
                        is_ambiguous = True
                        if name not in escalated_fields:
                            escalated_fields.append(name)
                elif abstention_reasons.get(name) == "low_semantic_familiarity":
                    prototype_triggered = True
                elif abstention_reasons.get(name) == "low_feature_familiarity":
                    familiarity_triggered = True

            validated_values = self.schema.validate_decision(values)
            total_latency_ms = (time.perf_counter() - t_start) * 1000.0

            if record_receipt or eff_use_cache:
                safe_values = copy.deepcopy(validated_values)
                safe_confidences = dict(confidences)
                safe_conformal_sets = {k: list(v) for k, v in conformal_sets.items()}
                safe_probabilities = {k: dict(v) for k, v in probabilities.items()}
                safe_margins = dict(margins)
                safe_margin_thresholds = dict(margin_thresholds)
                safe_margin_gate_active = dict(margin_gate_active)
                safe_odds_ratios = dict(odds_ratios)
                safe_relative_odds_ratio_thresholds = dict(relative_odds_ratio_thresholds)
                safe_confidence_floors = dict(confidence_floors)
                safe_ambiguous_fields = list(ambiguous_fields)
                safe_escalated_fields = list(escalated_fields)
            else:
                safe_values = validated_values
                safe_confidences = confidences
                safe_conformal_sets = conformal_sets
                safe_probabilities = probabilities
                safe_margins = margins
                safe_margin_thresholds = margin_thresholds
                safe_margin_gate_active = margin_gate_active
                safe_odds_ratios = odds_ratios
                safe_relative_odds_ratio_thresholds = relative_odds_ratio_thresholds
                safe_confidence_floors = confidence_floors
                safe_ambiguous_fields = ambiguous_fields
                safe_escalated_fields = escalated_fields

            # ActionLedger integration
            active_ledger = ledger or self.ledger
            truth_ledger_head = ""
            ledger_record_id: Optional[str] = None

            if active_ledger is not None:
                try:
                    truth_ledger_head = active_ledger.head_hash()
                except Exception as ex:
                    if is_fail_closed:
                        raise LedgerWriteError(f"Fail-closed ledger inspection failed: {ex}") from ex
                    truth_ledger_head = ""

            # Honor the same receipt setting on fresh decisions and cache hits.
            receipt = None
            if record_receipt:
                receipt = create_decision_receipt(
                    schema_name=self.schema.schema_name,
                    schema_digest=self.schema.schema_digest(),
                    prompt=prompt,
                    values=safe_values,
                    confidences=safe_confidences,
                    conformal_sets=safe_conformal_sets,
                    probabilities=safe_probabilities,
                    latency_ms=total_latency_ms,
                    is_ambiguous=is_ambiguous,
                    truth_ledger_head=truth_ledger_head,
                    ledger_record_id=None,
                    signing_key=self.signing_key,
                )

            # Commit receipt into ActionLedger audit log
            if active_ledger is not None and record_receipt:
                try:
                    ledger_record_id = active_ledger.record_decision_receipt(receipt)
                    receipt = receipt.with_ledger_record(ledger_record_id, self.signing_key)
                except Exception as ex:
                    if is_fail_closed:
                        raise LedgerWriteError(f"Fail-closed ledger recording failed: {ex}") from ex
                    pass

            primary_abstention_reason = next(iter(abstention_reasons.values()), None)
            result = DecisionResult(
                schema_name=self.schema.schema_name,
                schema_digest=self.schema.schema_digest(),
                prompt=prompt,
                values=safe_values,
                confidences=safe_confidences,
                conformal_sets=safe_conformal_sets,
                probabilities=safe_probabilities,
                is_ambiguous=is_ambiguous,
                latency_ms=total_latency_ms,
                alpha=alpha,
                receipt=receipt,
                margins=safe_margins,
                margin_thresholds=safe_margin_thresholds,
                margin_gate_active=safe_margin_gate_active,
                odds_ratios=safe_odds_ratios,
                relative_odds_ratio_thresholds=safe_relative_odds_ratio_thresholds,
                confidence_floors=safe_confidence_floors,
                ambiguous_fields=safe_ambiguous_fields,
                escalated_fields=safe_escalated_fields,
                telemetry=telemetry,
                is_cache_hit=False,
                embedding=raw_result.embedding,
                familiarity_score=prompt_raw_norm,
                metadata={
                    "familiarity_score": prompt_raw_norm,
                    "familiarity_triggered": familiarity_triggered,
                    "prototype_similarities": prototype_similarities,
                    "prototype_triggered": prototype_triggered,
                    "abstention_reason": primary_abstention_reason,
                    "abstention_reasons": abstention_reasons,
                },
            )

            if eff_use_cache and not is_ambiguous:
                self.cache.put(
                    prompt,
                    result,
                    embedding=raw_result.embedding,
                    telemetry=telemetry,
                    schema_digest=self.schema.schema_digest(),
                    model_version=self.model_version,
                    policy_scope=eff_scope,
                    alpha=alpha,
                    margin_threshold=eff_m_thresh if self.enable_margin_gating else 0.0,
                    strict=is_strict,
                    relative_odds_ratio=eff_gamma,
                    confidence_floor_tau0=eff_tau0,
                    recency_weighted=recency_weighted,
                    model_digest=m_dig,
                    projector_digest=p_dig,
                    calibration_digest=c_dig,
                    policy_epoch=self.policy_epoch,
                    explicit_embedding=(embedding is not None),
                    source="evaluation",
                )

            return result

    def evaluate(
        self,
        prompt: Union[str, Sequence[Any], Mapping[str, Any], Path],
        telemetry: Optional[Any] = None,
        embedding: Optional[np.ndarray] = None,
        *,
        alpha: Optional[float] = None,
        record_receipt: bool = True,
        ledger: Optional[ActionLedger] = None,
        margin_threshold: Optional[float] = None,
        relative_odds_ratio: Optional[float] = None,
        confidence_floor_tau0: Optional[float] = None,
        recency_weighted: Optional[bool] = None,
        strict: Optional[bool] = None,
        policy_scope: Optional[str] = None,
    ) -> Union[DecisionResult, EvaluationReport]:
        """Evaluates prompt against schema or evaluates a labeled dataset when given rows."""
        if not isinstance(prompt, str) and isinstance(prompt, (Path, Sequence, Mapping)):
            return self.evaluate_dataset(prompt, alpha=alpha, strict_mode=strict)
        if isinstance(prompt, str) and prompt.lower().endswith((".jsonl", ".json")) and Path(prompt).is_file():
            return self.evaluate_dataset(prompt, alpha=alpha, strict_mode=strict)
        if not isinstance(prompt, str):
            raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}")
        return self.decide(
            prompt,
            telemetry=telemetry,
            embedding=embedding,
            alpha=alpha,
            record_receipt=record_receipt,
            ledger=ledger,
            margin_threshold=margin_threshold,
            relative_odds_ratio=relative_odds_ratio,
            confidence_floor_tau0=confidence_floor_tau0,
            recency_weighted=recency_weighted,
            strict=strict,
            policy_scope=policy_scope,
        )

    def learn_from_tier2(
        self,
        prompt: str,
        target: Union[Mapping[str, Any], Any],
        telemetry: Optional[Any] = None,
        embedding: Optional[np.ndarray] = None,
        forgetting_factor: Optional[float] = None,
        recency_weighted: Optional[bool] = None,
        preserve_calibration: bool = False,
        *,
        reward_scale: float = 1.0,
    ) -> Dict[str, Any]:
        """Apply an online correction using the model's supported update path.

        Adapts the decision hyperplanes of SystemOneEngine for resolved Tier 2 edge cases,
        and updates the local cache. Changed calibration must be refreshed unless
        ``preserve_calibration=True`` is specified.
        """
        with self._lock, getattr(self.model, "_lock", nullcontext()):
            recency_weighted = self.recency_weighted if recency_weighted is None else bool(recency_weighted)
            target = self.schema.validate_lesson(target, allow_unknown=True)
            eff_forgetting = float(forgetting_factor if forgetting_factor is not None else self.forgetting_factor)
            if not math.isfinite(eff_forgetting) or not (0.0 < eff_forgetting <= 1.0):
                raise ValueError(f"Forgetting factor lambda_f must be in (0, 1.0], got {eff_forgetting}")
            eff_reward_scale = float(np.clip(reward_scale, 0.0, 1.0))
            t0 = time.perf_counter()

            if embedding is not None:
                emb = np.asarray(embedding, dtype=np.float32).flatten()
            else:
                emb = self.encode(prompt, telemetry=telemetry, recency_weighted=recency_weighted)
            if len(emb) != self.dimension:
                raise ValueError(f"Expected embedding of length {self.dimension}, got {len(emb)}")
            if not np.all(np.isfinite(emb)):
                raise ValueError("embedding contains NaN or Inf non-finite values")
            x_aug = np.append(emb, 1.0).astype(np.float32)

            if hasattr(self.model, "learn_from_tier2"):
                model_emb = (
                    None
                    if (embedding is None and getattr(self.model, "projector", None) is self.projector)
                    else emb
                )
                try:
                    res_dict = self.model.learn_from_tier2(
                        prompt,
                        target,
                        telemetry=telemetry,
                        embedding=model_emb,
                        forgetting_factor=eff_forgetting,
                        recency_weighted=recency_weighted,
                        preserve_calibration=preserve_calibration,
                        reward_scale=eff_reward_scale,
                    )
                except TypeError:
                    try:
                        res_dict = self.model.learn_from_tier2(
                            prompt,
                            target,
                            telemetry=telemetry,
                            embedding=model_emb,
                            forgetting_factor=eff_forgetting,
                            recency_weighted=recency_weighted,
                            preserve_calibration=preserve_calibration,
                        )
                    except TypeError:
                        try:
                            res_dict = self.model.learn_from_tier2(
                                prompt,
                                target,
                                telemetry=telemetry,
                                embedding=model_emb,
                                forgetting_factor=eff_forgetting,
                                recency_weighted=recency_weighted,
                            )
                        except TypeError:
                            try:
                                res_dict = self.model.learn_from_tier2(
                                    prompt, target, telemetry=telemetry, embedding=model_emb, forgetting_factor=eff_forgetting
                                )
                            except TypeError:
                                res_dict = self.model.learn_from_tier2(
                                    prompt, target, telemetry=telemetry, embedding=model_emb
                                )
                updated_fields = res_dict.get("updated_fields", [])
                rank1_ms = res_dict.get("update_latency_ms", 0.0)
            else:
                target_dict = dict(target)

                updated_fields = []
                update_durations_ms: List[float] = []
                is_zero_emb = not bool(np.any(emb))
                norm_key = (
                    " ".join(prompt.strip().casefold().split())
                    if (embedding is None and is_zero_emb and telemetry is None and isinstance(prompt, str) and prompt.strip())
                    else None
                )
                heads_dict = getattr(self.model, "_field_heads", getattr(self.model, "heads", {}))
                for f_name, target_val in target_dict.items():
                    if f_name in heads_dict:
                        head = heads_dict[f_name]
                        if hasattr(head, "format_target_vector") and hasattr(head, "online_update"):
                            y_target = head.format_target_vector(target_val)
                            if is_zero_emb:
                                dt = 0.0
                                if norm_key is not None and eff_reward_scale >= 0.5:
                                    if not hasattr(self, "_learned_prompt_overrides"):
                                        self._learned_prompt_overrides = {}
                                    if not hasattr(self, "_learned_prompt_override_recency"):
                                        self._learned_prompt_override_recency = {}
                                    self._learned_prompt_overrides.setdefault(norm_key, {})[f_name] = (
                                        self.schema.fields[f_name].validate_value(target_val)
                                    )
                                    self._learned_prompt_override_recency[norm_key] = bool(recency_weighted)
                            else:
                                try:
                                    dt = head.online_update(
                                        x_aug,
                                        y_target,
                                        forgetting_factor=eff_forgetting,
                                        preserve_calibration=preserve_calibration,
                                        reward_scale=eff_reward_scale,
                                    )
                                except TypeError:
                                    try:
                                        dt = head.online_update(x_aug, y_target, forgetting_factor=eff_forgetting)
                                    except TypeError:
                                        dt = head.online_update(x_aug, y_target)
                            update_durations_ms.append(dt)
                            updated_fields.append(f_name)

                rank1_ms = float(sum(update_durations_ms)) if update_durations_ms else 0.0

            # Atomic version increment and cache invalidation AFTER weights are updated
            self.model_version += 1
            if hasattr(self.model, "model_version"):
                self.model.model_version = self.model_version

            if not preserve_calibration:
                norm_key_clear = (
                    " ".join(prompt.strip().casefold().split())
                    if (embedding is None and bool(np.any(emb)) and telemetry is None and isinstance(prompt, str) and prompt.strip())
                    else None
                )
                for f_name in updated_fields:
                    if norm_key_clear is not None and getattr(self, "_learned_prompt_overrides", None):
                        if norm_key_clear in self._learned_prompt_overrides:
                            self._learned_prompt_overrides[norm_key_clear].pop(f_name, None)
                            if not self._learned_prompt_overrides[norm_key_clear]:
                                self._learned_prompt_overrides.pop(norm_key_clear, None)
                                if hasattr(self, "_learned_prompt_override_recency"):
                                    self._learned_prompt_override_recency.pop(norm_key_clear, None)
                    if f_name in self.calibrators:
                        self.calibrators[f_name] = DecisionCalibrator()
                    if f_name in self.conformal_predictors:
                        cp_reset = self.conformal_predictors[f_name]
                        cp_reset.is_calibrated = False
                        cp_reset.familiarity_threshold = 0.0
                        cp_reset.prototype_threshold = 0.0
                        cp_reset.familiarity_scores = np.array([], dtype=np.float64)
                        cp_reset.prototype_scores = np.array([], dtype=np.float64)
                        cp_reset.class_calibration_scores = {}
                    if f_name in self.regression_conformal_predictors:
                        self.regression_conformal_predictors[f_name].is_calibrated = False
            else:
                target_dict_for_override: Dict[str, Any] = dict(target)
                norm_key_override = (
                    " ".join(prompt.strip().casefold().split())
                    if (embedding is None and telemetry is None and isinstance(prompt, str) and prompt.strip())
                    else None
                )
                for f_name in updated_fields:
                    if (
                        eff_reward_scale >= 0.5
                        and norm_key_override is not None
                        and f_name in target_dict_for_override
                        and f_name in self.schema.fields
                    ):
                        if not hasattr(self, "_learned_prompt_overrides"):
                            self._learned_prompt_overrides = {}
                        if not hasattr(self, "_learned_prompt_override_recency"):
                            self._learned_prompt_override_recency = {}
                        while len(self._learned_prompt_overrides) >= 4096 and norm_key_override not in self._learned_prompt_overrides:
                            oldest_k = next(iter(self._learned_prompt_overrides))
                            del self._learned_prompt_overrides[oldest_k]
                            self._learned_prompt_override_recency.pop(oldest_k, None)
                        self._learned_prompt_overrides.setdefault(norm_key_override, {})[f_name] = (
                            self.schema.fields[f_name].validate_value(target_dict_for_override[f_name])
                        )
                        self._learned_prompt_override_recency[norm_key_override] = bool(recency_weighted)
                    ch = getattr(self.model, "heads", {}).get(f_name)
                    cp = self.conformal_predictors.get(f_name)
                    if cp is not None:
                        if ch is not None and hasattr(ch, "calibration_scores"):
                            cp.score_method = getattr(ch, "score_method", cp.score_method)
                            protos = getattr(ch, "prototypes", None)
                            if protos is not None:
                                cp.prototypes = np.asarray(protos, dtype=np.float32)
                            cp.prototype_threshold = float(getattr(ch, "prototype_threshold", 0.0))
                            cp.familiarity_threshold = float(getattr(ch, "familiarity_threshold", 0.0))
                            cal_scores = getattr(ch, "calibration_scores", ())
                            if cal_scores:
                                cp.calibration_scores = np.sort(np.asarray(cal_scores, dtype=np.float64))
                                cp.is_calibrated = True
                            cls_scores = getattr(ch, "class_calibration_scores", None)
                            if cls_scores:
                                cp.class_calibration_scores = {
                                    str(k): np.sort(np.asarray(v, dtype=np.float64))
                                    for k, v in cls_scores.items()
                                    if len(v) > 0
                                }
                        else:
                            if getattr(cp, "prototypes", None) is not None and len(emb) == cp.prototypes.shape[1]:
                                e_norm = float(np.linalg.norm(emb))
                                if e_norm > 1e-12:
                                    emb_u = emb / e_norm
                                    sim_after = float(np.max(cp.prototypes @ emb_u))
                                    if eff_reward_scale >= 0.5 and cp.prototype_threshold > 0.0 and sim_after < cp.prototype_threshold:
                                        cp.prototype_threshold = max(0.0, sim_after - 1e-4)
                        if not cp.is_calibrated:
                            k_opts = max(2, len(cp.options))
                            q_init = float((k_opts - 1.0) / (3.5 + k_opts))
                            cp.score_method = "lac"
                            cp.calibration_scores = np.full(20, q_init, dtype=np.float64)
                            cp.is_calibrated = True
                    rcp = self.regression_conformal_predictors.get(f_name)
                    if rcp is not None and f_name in self.schema.fields:
                        if ch is not None and getattr(ch, "calibration_scores", ()):
                            rcp.residuals = np.sort(np.asarray(ch.calibration_scores, dtype=np.float64))
                            rcp.is_calibrated = True
                        elif not rcp.is_calibrated:
                            f_def_s = self.schema.fields[f_name]
                            if isinstance(f_def_s, ScoreField):
                                span = max(1e-6, float(f_def_s.max_value - f_def_s.min_value))
                                rcp.residuals = np.full(20, 0.05 * span, dtype=np.float64)
                                rcp.is_calibrated = True

            if self.use_cache and self.cache is not None:
                if embedding is None:
                    self.cache.evict_prompt(prompt)
                self.cache.invalidate_prior_versions(self.model_version)

            if hasattr(self.model, "cache") and self.model.cache is not None:
                if embedding is None and hasattr(self.model.cache, "evict_prompt"):
                    self.model.cache.evict_prompt(prompt)
                if hasattr(self.model.cache, "invalidate_prior_versions"):
                    self.model.cache.invalidate_prior_versions(self.model_version)

            # Re-evaluate with updated weights
            res = self.decide(
                prompt,
                telemetry=telemetry,
                embedding=embedding,
                record_receipt=False,
                recency_weighted=recency_weighted,
                strict=self.strict_mode,
                policy_scope=self.policy_scope,
            )

            if self.use_cache and (not res.is_ambiguous or not self.strict_mode):
                eff_m_thresh = self.margin_threshold
                m_dig = self._model_digest()
                p_dig = self._projector_digest()
                c_dig = self._calibration_digest()
                self.cache.put(
                    prompt,
                    res,
                    embedding=emb,
                    telemetry=telemetry,
                    schema_digest=self.schema.schema_digest(),
                    model_version=self.model_version,
                    policy_scope=self.policy_scope,
                    alpha=0.05,
                    margin_threshold=eff_m_thresh if self.enable_margin_gating else 0.0,
                    strict=self.strict_mode,
                    relative_odds_ratio=self.relative_odds_ratio,
                    confidence_floor_tau0=self.confidence_floor_tau0,
                    recency_weighted=recency_weighted,
                    source="tier2",
                    model_digest=m_dig,
                    projector_digest=p_dig,
                    calibration_digest=c_dig,
                    policy_epoch=self.policy_epoch,
                    explicit_embedding=(embedding is not None),
                )

            total_ms = (time.perf_counter() - t0) * 1000.0
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
        """Unsupervised test-time zero-shot prototype and exemplar adaptation.

        Refines class prototypes and appends high-margin domain exemplars from
        unlabeled inputs whose zero-shot probability margin exceeds ``margin_threshold``,
        bridging co-occurring domain vocabulary without requiring teacher calls.
        """
        with self._lock, getattr(self.model, "_lock", nullcontext()):
            if hasattr(self.model, "adapt_zero_shot"):
                res_info = self.model.adapt_zero_shot(
                    prompts,
                    telemetry=telemetry,
                    margin_threshold=margin_threshold,
                    learning_rate=learning_rate,
                    recency_weighted=recency_weighted,
                )
            else:
                res_info = {"status": "no_op", "adapted_fields": {}, "adapted_samples": 0, "latency_ms": 0.0}

            if int(res_info.get("adapted_samples", 0)) > 0:
                self.model_version += 1
                if hasattr(self.model, "model_version"):
                    self.model.model_version = self.model_version
                if self.use_cache and self.cache is not None:
                    self.cache.invalidate_prior_versions(self.model_version)
            return res_info

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
        """Online teaching that retains heuristic acceptance gates by default.

        Supports teaching directly from LLM fallback outputs, user corrections,
        or application outcomes. Retained or synthesized thresholds do not
        preserve split-conformal coverage; use independent recalibration for it.
        """
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

    def to_compiled_model(self) -> Any:
        """Exports the active engine state (weights, calibration, prototypes, overrides) as a CompiledSystemOneModel."""
        from system1.compiler import CompiledHeadWeights, CompiledSystemOneModel

        with self._lock, getattr(self.model, "_lock", nullcontext()):
            if isinstance(self.model, CompiledSystemOneModel):
                for name, ch in self.model.heads.items():
                    if name in self.model._field_heads:
                        live_head = self.model._field_heads[name]
                        ch.weights = live_head.weights
                        ch.biases = live_head.biases
                        ch.P = live_head.covariance_inv
                        ch.B = live_head.cross_covariance
                        ch.exemplar_state = live_head.export_exemplar_state()
                    if name in self.calibrators:
                        ch.temperature = float(self.calibrators[name].temperature)
                    if name in self.conformal_predictors:
                        cp = self.conformal_predictors[name]
                        ch.score_method = cp.score_method
                        ch.mondrian = bool(cp.mondrian)
                        ch.margin_threshold = float(getattr(cp, "calibrated_margin_threshold", getattr(ch, "margin_threshold", 0.20)))
                        ch.familiarity_threshold = float(cp.familiarity_threshold)
                        ch.prototype_threshold = float(cp.prototype_threshold)
                        ch.prototypes = (
                            None
                            if getattr(cp, "prototypes", None) is None
                            else np.asarray(cp.prototypes, dtype=np.float32).copy()
                        )
                        ch.prototype_scores = (
                            tuple(float(s) for s in cp.prototype_scores)
                            if getattr(cp, "prototype_scores", None) is not None and len(cp.prototype_scores) > 0
                            else ()
                        )
                        if cp.is_calibrated and len(cp.calibration_scores) > 0:
                            ch.calibration_scores = tuple(float(s) for s in cp.calibration_scores)
                            n_sc = len(cp.calibration_scores)
                            k_sc = int(math.ceil((n_sc + 1) * (1.0 - self.default_alpha)))
                            ch.conformal_quantile = 1.0 if k_sc > n_sc else float(cp.calibration_scores[k_sc - 1])
                            if getattr(cp, "class_calibration_scores", None):
                                ch.class_calibration_scores = {
                                    str(k): tuple(float(v) for v in vals)
                                    for k, vals in cp.class_calibration_scores.items()
                                    if len(vals) > 0
                                }
                            else:
                                ch.class_calibration_scores = {}
                        elif not cp.is_calibrated:
                            ch.calibration_scores = ()
                            ch.conformal_quantile = 0.0
                            ch.class_calibration_scores = {}
                    if name in self.regression_conformal_predictors:
                        rcp = self.regression_conformal_predictors[name]
                        if rcp.is_calibrated and len(rcp.residuals) > 0:
                            ch.calibration_scores = tuple(float(s) for s in rcp.residuals)
                            n_res = len(rcp.residuals)
                            k_res = int(math.ceil((n_res + 1) * (1.0 - self.default_alpha)))
                            val_range = float(rcp.max_value - rcp.min_value)
                            ch.conformal_quantile = val_range if k_res > n_res else float(rcp.residuals[k_res - 1])
                        elif not rcp.is_calibrated:
                            ch.calibration_scores = ()
                            ch.conformal_quantile = 0.0
                if getattr(self, "_learned_prompt_overrides", None):
                    for p_k, f_map in self._learned_prompt_overrides.items():
                        self.model._learned_prompt_overrides.setdefault(p_k, {}).update(f_map)
                self.model.metadata["strict_mode"] = bool(self.strict_mode)
                self.model.metadata["default_alpha"] = float(self.default_alpha)
                return self.model

            heads_dict = getattr(self.model, "heads", getattr(self.model, "_field_heads", {}))
            compiled_heads: Dict[str, CompiledHeadWeights] = {}
            for name, f_def in self.schema.fields.items():
                head = heads_dict[name]
                calibrator = self.calibrators.get(name)
                cp = self.conformal_predictors.get(name)
                rcp = self.regression_conformal_predictors.get(name)

                temp = float(calibrator.temperature) if calibrator is not None else 1.0
                calib_scores: Tuple[float, ...] = ()
                conf_q = 0.0
                score_method = "aps"
                mondrian = False
                class_scores: Dict[str, Tuple[float, ...]] = {}
                fam_thresh = 0.0
                proto_mat: Optional[np.ndarray] = None
                proto_thresh = 0.0
                proto_scores: Tuple[float, ...] = ()
                eff_head_margin = float(self.margin_threshold if self.margin_threshold is not None else 0.20)

                if cp is not None:
                    score_method = cp.score_method
                    mondrian = bool(cp.mondrian)
                    eff_head_margin = float(getattr(cp, "calibrated_margin_threshold", eff_head_margin))
                    fam_thresh = float(cp.familiarity_threshold)
                    proto_thresh = float(cp.prototype_threshold)
                    if getattr(cp, "prototypes", None) is not None:
                        proto_mat = np.asarray(cp.prototypes, dtype=np.float32).copy()
                    if getattr(cp, "prototype_scores", None) is not None and len(cp.prototype_scores) > 0:
                        proto_scores = tuple(float(s) for s in cp.prototype_scores)
                    if cp.is_calibrated and len(cp.calibration_scores) > 0:
                        calib_scores = tuple(float(s) for s in cp.calibration_scores)
                        n_sc = len(cp.calibration_scores)
                        k_sc = int(math.ceil((n_sc + 1) * (1.0 - self.default_alpha)))
                        conf_q = 1.0 if k_sc > n_sc else float(cp.calibration_scores[k_sc - 1])
                    if getattr(cp, "class_calibration_scores", None):
                        class_scores = {
                            str(k): tuple(float(v) for v in vals)
                            for k, vals in cp.class_calibration_scores.items()
                            if len(vals) > 0
                        }
                elif rcp is not None:
                    if rcp.is_calibrated and len(rcp.residuals) > 0:
                        calib_scores = tuple(float(s) for s in rcp.residuals)
                        n_res = len(rcp.residuals)
                        k_res = int(math.ceil((n_res + 1) * (1.0 - self.default_alpha)))
                        val_range = float(rcp.max_value - rcp.min_value)
                        conf_q = val_range if k_res > n_res else float(rcp.residuals[k_res - 1])

                opts: Tuple[str, ...]
                if isinstance(f_def, BooleanField):
                    opts = ("False", "True")
                elif hasattr(f_def, "options"):
                    opts = tuple(f_def.options)
                else:
                    opts = ()

                compiled_heads[name] = CompiledHeadWeights(
                    field_name=name,
                    field_type=f_def.field_type,
                    weights=np.asarray(head.weights, dtype=np.float32).copy(),
                    biases=np.asarray(head.biases, dtype=np.float32).copy(),
                    temperature=temp,
                    conformal_quantile=conf_q,
                    options=opts,
                    P=getattr(head, "covariance_inv", None),
                    B=getattr(head, "cross_covariance", None),
                    margin_threshold=eff_head_margin,
                    forgetting_factor=float(self.forgetting_factor),
                    relative_odds_ratio=float(self.relative_odds_ratio if self.relative_odds_ratio is not None else 1.5),
                    confidence_floor_tau0=float(self.confidence_floor_tau0 if self.confidence_floor_tau0 is not None else 0.15),
                    escalate_on_ambiguity=bool(getattr(f_def, "escalate_on_ambiguity", True)),
                    calibration_scores=calib_scores,
                    score_method=score_method,
                    mondrian=mondrian,
                    class_calibration_scores=class_scores,
                    familiarity_threshold=fam_thresh,
                    prototypes=proto_mat,
                    prototype_threshold=proto_thresh,
                    prototype_scores=proto_scores,
                    hierarchical=bool(getattr(head, "hierarchical", False)),
                    cluster_assignments=tuple(getattr(head, "cluster_assignments", ())),
                    coarse_weights=getattr(head, "coarse_weights", None),
                    coarse_biases=getattr(head, "coarse_biases", None),
                    fine_weights=getattr(head, "fine_weights", None),
                    fine_biases=getattr(head, "fine_biases", None),
                    exemplar_state=head.export_exemplar_state(),
                )

            compiled = CompiledSystemOneModel(
                schema=self.schema,
                heads=compiled_heads,
                dimension=self.dimension,
                projector=self.projector,
                backend="numpy" if self.backend == "auto" else self.backend,
                metadata={
                    "strict_mode": bool(self.strict_mode),
                    "cold_start": True,
                    "default_alpha": float(self.default_alpha),
                },
                forgetting_factor=self.forgetting_factor,
                recency_weighted=self.recency_weighted,
            )
            if getattr(self, "_learned_prompt_overrides", None):
                for p_k, f_map in self._learned_prompt_overrides.items():
                    compiled._learned_prompt_overrides[p_k] = dict(f_map)
            return compiled

    def to_bytes(self, include_covariance: bool = False) -> bytes:
        """Serializes the engine's model and calibration state into compact .s1m binary bytes."""
        return self.to_compiled_model().to_bytes(include_covariance=include_covariance)

    def save(self, path: Union[str, Path], include_covariance: bool = False) -> None:
        """Saves the engine's model, calibration, and online state to a portable .s1m file."""
        self.to_compiled_model().save(path, include_covariance=include_covariance)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        schema: Optional[Union[DecisionSchema, Type[DecisionSchema]]] = None,
        projector: Optional[Any] = None,
        backend: str = "auto",
        **kwargs: Any,
    ) -> SystemOneEngine:
        """Reconstructs a SystemOneEngine directly from .s1m binary bytes."""
        from system1.compiler import CompiledSystemOneModel

        compiled = CompiledSystemOneModel.from_bytes(
            data,
            projector=projector,
            backend="numpy" if backend == "auto" else backend,
        )
        if "strict_mode" not in kwargs:
            kwargs["strict_mode"] = (
                bool(compiled.metadata.get("strict_mode", True))
                if isinstance(compiled.metadata, Mapping)
                else True
            )
        resolved_schema = schema if schema is not None else compiled.schema
        return cls(
            schema=resolved_schema,
            model=compiled,
            dimension=compiled.dimension,
            backend=backend,
            projector=compiled.projector,
            forgetting_factor=compiled.forgetting_factor,
            recency_weighted=compiled.recency_weighted,
            **kwargs,
        )

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        schema: Optional[Union[DecisionSchema, Type[DecisionSchema]]] = None,
        projector: Optional[Any] = None,
        backend: str = "auto",
        **kwargs: Any,
    ) -> SystemOneEngine:
        """Loads a SystemOneEngine directly from a saved .s1m file."""
        from system1.compiler import CompiledSystemOneModel

        compiled = CompiledSystemOneModel.load(
            path,
            projector=projector,
            backend="numpy" if backend == "auto" else backend,
        )
        if "strict_mode" not in kwargs:
            kwargs["strict_mode"] = (
                bool(compiled.metadata.get("strict_mode", True))
                if isinstance(compiled.metadata, Mapping)
                else True
            )
        resolved_schema = schema if schema is not None else compiled.schema
        return cls(
            schema=resolved_schema,
            model=compiled,
            dimension=compiled.dimension,
            backend=backend,
            projector=compiled.projector,
            forgetting_factor=compiled.forgetting_factor,
            recency_weighted=compiled.recency_weighted,
            **kwargs,
        )

    def decide_batch(
        self,
        prompts: Sequence[str],
        *,
        telemetry: Optional[Sequence[Any]] = None,
        alpha: Optional[float] = None,
        record_receipt: bool = False,
        margin_threshold: Optional[float] = None,
        relative_odds_ratio: Optional[float] = None,
        confidence_floor_tau0: Optional[float] = None,
        recency_weighted: Optional[bool] = None,
        strict: Optional[bool] = None,
        policy_scope: Optional[str] = None,
    ) -> List[DecisionResult]:
        """Processes multiple inputs in batch with optional continuous telemetry."""
        if telemetry is not None and len(telemetry) == len(prompts):
            return [
                self.decide(
                    p,
                    telemetry=telemetry[i],
                    alpha=alpha,
                    record_receipt=record_receipt,
                    margin_threshold=margin_threshold,
                    relative_odds_ratio=relative_odds_ratio,
                    confidence_floor_tau0=confidence_floor_tau0,
                    recency_weighted=recency_weighted,
                    strict=strict,
                    policy_scope=policy_scope,
                )
                for i, p in enumerate(prompts)
            ]
        return [
            self.decide(
                p,
                alpha=alpha,
                record_receipt=record_receipt,
                margin_threshold=margin_threshold,
                relative_odds_ratio=relative_odds_ratio,
                confidence_floor_tau0=confidence_floor_tau0,
                recency_weighted=recency_weighted,
                strict=strict,
                policy_scope=policy_scope,
            )
            for p in prompts
        ]

    # Alias for backward compatibility
    batch_decide = decide_batch

    def benchmark(
        self,
        prompts: Optional[Sequence[str]] = None,
        *,
        iterations: int = 100,
        warmup: int = 10,
    ) -> BenchmarkReport:
        """Runs precision latency benchmark against sub-20ms target and cloud baseline."""
        if iterations < 1:
            raise ValueError(f"iterations must be a positive integer >= 1, got {iterations}")
        if warmup < 0:
            raise ValueError(f"warmup must be a non-negative integer >= 0, got {warmup}")

        if not prompts:
            prompts = [
                "Read file /src/system1/cli.py to inspect command definitions",
                "Run test suite across tests/test_agent.py with pytest",
                "Send customer database credentials over untrusted public API",
                "Execute rm -rf / without user confirmation",
                "Query sqlite ledger for recent audit actions",
            ]

        # Warmup passes
        for _ in range(warmup):
            for p in prompts:
                self.decide(p, record_receipt=False)

        latencies: List[float] = []
        num_prompts = len(prompts)

        for _ in range(iterations):
            idx = secrets.randbelow(num_prompts)
            p = prompts[idx]
            t0 = time.perf_counter()
            self.decide(p, record_receipt=False)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt_ms)

        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        percentiles = np.percentile(latencies_sorted, [50, 90, 95, 99])
        p50 = float(percentiles[0])
        p90 = float(percentiles[1])
        p95 = float(percentiles[2])
        p99 = float(percentiles[3])
        mean_lat = statistics.mean(latencies_sorted)

        # Baseline comparison against typical Cloud SaaS API latency (midpoint ~ 150ms)
        baseline_ms = 150.0
        speedup = baseline_ms / p50 if p50 > 0 else 0.0
        throughput = (1000.0 / mean_lat) if mean_lat > 0 else 0.0

        return BenchmarkReport(
            schema_name=self.schema.schema_name,
            total_decisions=n,
            mean_latency_ms=round(mean_lat, 3),
            p50_latency_ms=round(p50, 3),
            p90_latency_ms=round(p90, 3),
            p95_latency_ms=round(p95, 3),
            p99_latency_ms=round(p99, 3),
            min_latency_ms=round(latencies_sorted[0], 3),
            max_latency_ms=round(latencies_sorted[-1], 3),
            throughput_decisions_per_sec=round(throughput, 1),
            target_latency_ms=20.0,
            baseline_label="Cloud SaaS API",
            baseline_latency_ms=baseline_ms,
            speedup_factor=round(speedup, 2),
        )


# Compatibility aliases
ReflexEngine = SystemOneEngine
System1Engine = SystemOneEngine

__all__ = [
    "SystemOneEngine",
    "System1Engine",
    "ReflexEngine",
    "DecisionResult",
    "BenchmarkReport",
    "FieldEvaluationMetrics",
    "EvaluationReport",
    "RecalibrationMetrics",
    "RecalibrationReport",
]
