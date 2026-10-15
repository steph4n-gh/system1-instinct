"""Teach compact local decision skills from labeled examples using NumPy ridge
regression, held-out uncertainty calibration, and portable .s1m serialization.

Schema-derived synthetic examples remain available for bootstrapping demos.
"""

from __future__ import annotations

import io
import json
import math
import os
import struct
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import numpy as np

from system1.core.model import (
    DecisionFieldHead,
    DeterministicSemanticProjector,
    ModelInferenceResult,
    RawFieldEvaluation,
    SystemOneModel,
    _stable_sigmoid,
    _stable_softmax,
)
from system1.core.schema import (
    BooleanField,
    ChoiceField,
    DecisionField,
    DecisionSchema,
    MultiChoiceField,
    ScoreField,
)
from system1.calibration import ConformalPredictor, DecisionCalibrator
from system1.core.telemetry import TelemetryProjector
from system1.core.text import TfidfProjector
from system1.cache import SemanticSystemOneCache

MAGIC_HEADER = b"S1M\x02"

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
    if (disk or directory_disk or disk_count != count or count > MAX_SKILL_FIELDS * (11 + len(EXEMPLAR_ARRAYS))
            or directory_size > MAX_SKILL_HEADER_BYTES or directory_offset + directory_size != end
            or end + 22 + comment_size != len(data)):
        raise ValueError("Saved skill NPZ directory exceeds limits or is malformed")
    expected = {
        f"{name}_{suffix}.npy"
        for name in schema.fields
        for suffix in (
            "w",
            "b",
            "P",
            "B",
            "calib_scores",
            "prototypes",
            "proto_scores",
            "coarse_w",
            "coarse_b",
            "fine_w",
            "fine_b",
        ) + EXEMPLAR_ARRAYS
    } | {
        "_projector_spectral_vectors.npy",
        "_projector_subspace_components.npy",
    }
    shapes: Dict[str, Tuple[int, ...]] = {}
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
                if (dtype.kind != "f" or dtype.itemsize not in (4, 8) or len(shape) not in (1, 2)
                        or any(type(n) is not int or n < 0 for n in shape)):
                    raise ValueError("Saved skill arrays must be bounded float32/float64 vectors or matrices")
                byte_count = math.prod(shape) * dtype.itemsize
                if byte_count > MAX_SKILL_ARRAY_BYTES or stream.tell() + byte_count != info.file_size:
                    raise ValueError("Saved skill array shape exceeds its declared size or budget")
                shapes[info.filename[:-4]] = shape
        spec_vec_shape = shapes.get("_projector_spectral_vectors")
        if spec_vec_shape is not None and (
            len(spec_vec_shape) != 2
            or not (1 <= spec_vec_shape[0] <= 1024)
            or not (1 <= spec_vec_shape[1] <= dimension)
        ):
            raise ValueError("Saved hybrid projector spectral_vectors have invalid shape")
        sub_comp_shape = shapes.get("_projector_subspace_components")
        if sub_comp_shape is not None and (
            len(sub_comp_shape) != 2
            or not (1 <= sub_comp_shape[0] <= dimension)
            or sub_comp_shape[1] != dimension
        ):
            raise ValueError("Saved hybrid projector subspace_components have invalid shape")
        for name, definition in schema.fields.items():
            widths = (1, 2) if isinstance(definition, BooleanField) else (len(definition.options),) if hasattr(definition, "options") else (1,)
            weight_shape = shapes.get(f"{name}_w", ())
            if weight_shape not in tuple((width, dimension) for width in widths):
                raise ValueError(f"Saved head {name!r} has incompatible weight dimensions")
            width = weight_shape[0]
            required_shapes = {"b": (width,), "P": (dimension + 1, dimension + 1), "B": (dimension + 1, width)}
            for suffix, shape in required_shapes.items():
                actual = shapes.get(f"{name}_{suffix}")
                if (suffix == "b" or actual is not None) and actual != shape:
                    raise ValueError(f"Saved head {name!r} has incompatible {suffix} dimensions")
            calibration = shapes.get(f"{name}_calib_scores")
            if calibration is not None and (len(calibration) != 1 or calibration[0] > MAX_SKILL_CALIBRATION_SCORES):
                raise ValueError("Saved skill calibration exceeds limits")
            proto_shape = shapes.get(f"{name}_prototypes")
            if proto_shape is not None:
                valid_proto_rows = (1, 2) if isinstance(definition, BooleanField) else (width,)
                if len(proto_shape) != 2 or proto_shape[0] not in valid_proto_rows or proto_shape[1] != dimension:
                    raise ValueError(f"Saved head {name!r} has incompatible prototypes dimensions")
            proto_scores_shape = shapes.get(f"{name}_proto_scores")
            if proto_scores_shape is not None and (
                len(proto_scores_shape) != 1 or proto_scores_shape[0] > MAX_SKILL_CALIBRATION_SCORES
            ):
                raise ValueError("Saved skill prototype calibration exceeds limits")
            coarse_w_shape = shapes.get(f"{name}_coarse_w")
            coarse_b_shape = shapes.get(f"{name}_coarse_b")
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
            fine_w_shape = shapes.get(f"{name}_fine_w")
            fine_b_shape = shapes.get(f"{name}_fine_b")
            if (fine_w_shape is None) != (fine_b_shape is None):
                raise ValueError(f"Saved head {name!r} has incomplete fine hierarchical arrays")
            if fine_w_shape is not None:
                if fine_w_shape != (width, dimension) or fine_b_shape != (width,):
                    raise ValueError(f"Saved head {name!r} has incompatible fine hierarchical dimensions")
            if f"{name}_B" in shapes and f"{name}_P" not in shapes:
                runtime_bytes += 8 * (dimension + 1) ** 2
            for suffix in ("calib_scores", "proto_scores"):
                runtime_bytes += 32 * shapes.get(f"{name}_{suffix}", (0,))[0]
            for suffix in EXEMPLAR_ARRAYS:
                shape = shapes.get(f"{name}_{suffix}")
                if shape is None:
                    continue
                valid = (shape == (dimension,)) if suffix == "exemplar_mean" else (
                    len(shape) == 2 and shape[1] == dimension
                    and (suffix not in ("exemplar_centroids", "initial_w") or shape[0] == width)
                )
                if not isinstance(definition, ChoiceField) or not valid:
                    raise ValueError(f"Saved head {name!r} has incompatible exemplar dimensions")
                if suffix in ("shots_p", "shots_c"):
                    runtime_bytes += 160 * shape[0]  # Retained per-row ndarray views and list slots.
        runtime_bytes += 2 * sum(info.file_size for info in members)
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
        return projector.to_config()
    if type(projector) is HybridProjector:
        return projector.to_config(include_arrays=include_arrays)
    config = {"type": "external", "class": f"{type(projector).__module__}.{type(projector).__qualname__}",
              "dimension": getattr(projector, "dimension", None)}
    if callable(getattr(projector, "projector_digest", None)):
        config["digest"] = projector.projector_digest()
    return config


@dataclass
class CompiledHeadWeights:
    """Calibrated weight and bias matrices for a single schema field."""

    field_name: str
    field_type: str
    weights: np.ndarray  # shape: (K, D) or (1, D)
    biases: np.ndarray   # shape: (K,) or (1,)
    temperature: float = 1.0
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
                head.set_weights(ch.weights, ch.biases)
                if ch.exemplar_state is not None:
                    head.restore_exemplar_state(ch.exemplar_state)
                if (
                    getattr(ch, "hierarchical", False)
                    and ch.coarse_weights is not None
                    and ch.coarse_biases is not None
                    and len(ch.cluster_assignments) == ch.weights.shape[0]
                ):
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
                else np.array([-z, z], dtype=np.float32)
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
        emb = np.asarray(embedding, dtype=np.float32).flatten()
        if emb.ndim != 1 or ch.prototypes.ndim != 2 or ch.prototypes.shape[1] != emb.shape[0]:
            return None
        e_norm = float(np.linalg.norm(emb))
        if e_norm < 1e-12 or not math.isfinite(e_norm):
            return 0.0
        emb_unit = emb / e_norm
        p_norms = np.linalg.norm(ch.prototypes, axis=1, keepdims=True)
        valid_rows = p_norms.flatten() > 1e-12
        if not np.any(valid_rows):
            return None
        proto_unit = ch.prototypes[valid_rows] / np.where(
            p_norms[valid_rows] < 1e-12, 1.0, p_norms[valid_rows]
        )
        best_sim = float(np.max(proto_unit @ emb_unit))
        online_bank = getattr(self, "_online_prototype_bank", {}).get(field_name)
        if online_bank:
            for shot in online_bank:
                s_arr = np.asarray(shot, dtype=np.float32).flatten()
                if s_arr.shape[0] == emb_unit.shape[0]:
                    sim_s = float(np.dot(s_arr, emb_unit))
                    if sim_s > best_sim:
                        best_sim = sim_s
        head = self._field_heads.get(field_name)
        online_shots = getattr(head, "_online_shots_p", None) if head is not None else None
        if online_shots:
            for group in online_shots:
                for shot in group:
                    s_arr = np.asarray(shot, dtype=np.float32).flatten()
                    if s_arr.shape[0] == emb_unit.shape[0]:
                        s_norm = float(np.linalg.norm(s_arr))
                        if s_norm > 1e-12:
                            sim_s = float(np.dot(s_arr / s_norm, emb_unit))
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
        if hasattr(self.projector, "project_with_norm") and telemetry is None:
            try:
                emb, raw_norm = self.projector.project_with_norm(prompt, recency_weighted=use_recency)
            except TypeError:
                emb, raw_norm = self.projector.project_with_norm(prompt)
        elif hasattr(self.projector, "project"):
            try:
                emb = self.projector.project(prompt, recency_weighted=use_recency)
            except TypeError:
                emb = self.projector.project(prompt)
        else:
            emb = self.projector(prompt)
        norm = float(np.linalg.norm(emb))
        if norm > 1e-12:
            emb = emb / norm
        else:
            emb = np.zeros(self.dimension, dtype=np.float32)
            if not isinstance(self.projector, TfidfProjector):
                emb[0] = 1.0
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
        cur_version = getattr(self, "model_version", 1)
        if eff_use_cache:
            schema_dig = self.schema.schema_digest()
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
        if self._learned_prompt_overrides and embedding is None and telemetry is None and isinstance(prompt, str) and prompt.strip():
            norm_key = " ".join(prompt.strip().casefold().split())
            if self._learned_prompt_override_recency.get(norm_key, bool(self.recency_weighted)) == bool(use_recency):
                override_map = self._learned_prompt_overrides.get(norm_key)

        is_zero_emb_eval = not bool(np.any(emb))
        evals: Dict[str, RawFieldEvaluation] = {}
        for name, head in self._field_heads.items():
            if (
                override_map is not None
                and name in override_map
                and (is_zero_emb_eval or (not np.any(head.weights) and not np.any(head.biases)))
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

        if eff_use_cache:
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

            # Bump model_version and evict prompt only after all inputs are validated
            self.model_version = getattr(self, "model_version", 1) + 1
            if self.use_cache and self.cache is not None:
                if embedding is None and hasattr(self.cache, "evict_prompt"):
                    self.cache.evict_prompt(prompt)
                if hasattr(self.cache, "invalidate_prior_versions"):
                    self.cache.invalidate_prior_versions(self.model_version)

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
                            and hasattr(self.projector, "raw_norm")
                            and isinstance(prompt, str)
                            and prompt.strip()
                        ):
                            try:
                                p_raw_norm = float(self.projector.raw_norm(prompt))
                                if p_raw_norm > 0.0 and p_raw_norm < ch.familiarity_threshold:
                                    ch.familiarity_threshold = max(0.0, p_raw_norm - 1e-4)
                            except Exception:
                                pass
                        if preserve_calibration and not ch.calibration_scores:
                            if ch.field_type in ("choice", "boolean", "multi_choice", "multichoice"):
                                k_opts = max(2, len(ch.options))
                                q_init = float((k_opts - 1.0) / (3.5 + k_opts))
                                ch.score_method = "lac"
                                ch.calibration_scores = tuple(q_init for _ in range(20))
                                ch.conformal_quantile = q_init
                            elif ch.field_type == "score":
                                f_def_sc = self.schema.fields.get(f_name)
                                span = (
                                    max(1e-6, float(f_def_sc.max_value - f_def_sc.min_value))
                                    if isinstance(f_def_sc, ScoreField)
                                    else 1.0
                                )
                                q_init_sc = 0.05 * span
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
                    updated_fields.append(f_name)

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
                    and (is_zero_emb or (not np.any(head.weights) and not np.any(head.biases)))
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

    def to_bytes(self, include_covariance: bool = False) -> bytes:
        """Serializes the compiled model into compact .s1m binary bytes.

        By default, excludes covariance matrices to keep the binary ultra-compact (<20 KB).
        Covariance is automatically initialized on-demand if online learning is invoked.
        """
        with self._lock:
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
                if getattr(ch, "hierarchical", False) and getattr(ch, "cluster_assignments", ()):
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

            header_meta = {
                "version": 2,
                "projector": _projector_config(self.projector, include_arrays=False),
                "schema_name": self.schema.schema_name,
                "schema_dict": self.schema.to_dict(),
                "schema_digest": self.schema.schema_digest(),
                "dimension": self.dimension,
                "created_at": getattr(self, "created_at", 0.0),
                "metadata": serialized_meta,
                "heads": heads_meta,
            }
            header_bytes = json.dumps(header_meta, sort_keys=True).encode("utf-8")
            header_len = len(header_bytes)
            if header_len > MAX_SKILL_HEADER_BYTES:
                raise ValueError("Saved skill header exceeds size limit")

            # Pack numpy weights into compressed zip buffer
            npz_buf = io.BytesIO()
            arrays_to_save: Dict[str, np.ndarray] = {}
            from system1.core.embeddings import HybridProjector

            if type(self.projector) is HybridProjector and getattr(self.projector, "_fitted", False):
                if getattr(self.projector, "_spectral_vectors", None) is not None:
                    arrays_to_save["_projector_spectral_vectors"] = np.asarray(
                        self.projector._spectral_vectors, dtype=np.float32
                    )
                if getattr(self.projector, "_subspace_components", None) is not None:
                    arrays_to_save["_projector_subspace_components"] = np.asarray(
                        self.projector._subspace_components, dtype=np.float32
                    )
            for name, ch in sorted(self.heads.items()):
                arrays_to_save[f"{name}_w"] = ch.weights.astype(np.float32)
                arrays_to_save[f"{name}_b"] = ch.biases.astype(np.float32)
                state = exemplar_states.get(name)
                if state is not None:
                    for suffix, arr in state["arrays"].items():
                        arrays_to_save[f"{name}_{suffix}"] = np.asarray(arr)
                if hasattr(ch, "calibration_scores") and len(ch.calibration_scores) > 0:
                    arrays_to_save[f"{name}_calib_scores"] = np.asarray(ch.calibration_scores, dtype=np.float64)
                if getattr(ch, "prototypes", None) is not None and np.any(ch.prototypes):
                    arrays_to_save[f"{name}_prototypes"] = np.asarray(ch.prototypes, dtype=np.float32)
                if getattr(ch, "prototype_scores", ()) and len(ch.prototype_scores) > 0:
                    arrays_to_save[f"{name}_proto_scores"] = np.asarray(ch.prototype_scores, dtype=np.float64)
                if (
                    getattr(ch, "hierarchical", False)
                    and ch.coarse_weights is not None
                    and ch.coarse_biases is not None
                ):
                    arrays_to_save[f"{name}_coarse_w"] = np.asarray(ch.coarse_weights, dtype=np.float32)
                    arrays_to_save[f"{name}_coarse_b"] = np.asarray(ch.coarse_biases, dtype=np.float32)
                    if ch.fine_weights is not None and ch.fine_biases is not None:
                        arrays_to_save[f"{name}_fine_w"] = np.asarray(ch.fine_weights, dtype=np.float32)
                        arrays_to_save[f"{name}_fine_b"] = np.asarray(ch.fine_biases, dtype=np.float32)
                if include_covariance:
                    if ch.P is not None:
                        arrays_to_save[f"{name}_P"] = ch.P.astype(np.float32)
                    if ch.B is not None:
                        arrays_to_save[f"{name}_B"] = ch.B.astype(np.float32)
            np.savez_compressed(npz_buf, **arrays_to_save)
            npz_bytes = npz_buf.getvalue()

            # Wire format: MAGIC (4 bytes) + uint32(header_len) + header_bytes + npz_bytes
            out = bytearray()
            out.extend(MAGIC_HEADER)
            out.extend(struct.pack(">I", header_len))
            out.extend(header_bytes)
            out.extend(npz_bytes)
            return bytes(out)

    def save(self, path: Union[str, Path], include_covariance: bool = False) -> None:
        """Saves compiled model to a .s1m binary file on disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.to_bytes(include_covariance=include_covariance))

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        projector: Optional[Any] = None,
        backend: str = "numpy",
    ) -> CompiledSystemOneModel:
        """Deserializes a CompiledSystemOneModel from binary bytes."""
        if len(data) > MAX_SKILL_BYTES:
            raise ValueError("Saved skill file exceeds size limit")
        if len(data) < 8 or data[:4] not in (b"S1M\x01", MAGIC_HEADER):
            raise ValueError("Invalid System 1 Model binary: missing or invalid magic header")

        header_len = struct.unpack(">I", data[4:8])[0]
        if header_len > MAX_SKILL_HEADER_BYTES:
            raise ValueError("Saved skill header exceeds size limit")
        header_end = 8 + header_len
        if len(data) < header_end:
            raise ValueError("Corrupted System 1 Model binary: truncated header")

        header_json = data[8:header_end].decode("utf-8")
        meta = json.loads(header_json)
        if meta.get("version") not in (1, 2) or meta["version"] != data[3]:
            raise ValueError("Unsupported or inconsistent saved skill format version")

        # Reconstruct schema
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
        # Includes constructor/NumPy temporaries and covariance defaults when applicable.
        widths = sum(max(2, len(getattr(f, "options", ()))) for f in schema.fields.values())
        projector_config = meta.get("projector")
        cov_fields = (
            len(schema.fields)
            if (projector_config is None or projector_config.get("type") == "deterministic")
            else 0
        )
        runtime_bytes = 16 * cov_fields * (dimension + 1) ** 2 + 16 * widths * (dimension + 1)
        saved_tfidf = None
        saved_hybrid = None
        if projector_config is not None:
            kind = projector_config["type"]
            if kind == "tfidf":
                saved_tfidf = TfidfProjector.from_config(projector_config)
                if saved_tfidf.dimension != dimension:
                    raise ValueError("Saved TF-IDF dimension does not match the skill")
            if kind == "deterministic" and projector_config.get("dimension") != dimension:
                raise ValueError("Saved projector dimension does not match the skill")
            if kind == "hybrid":
                from system1.core.embeddings import HybridProjector
                if set(projector_config) - HybridProjector.CONFIG_KEYS:
                    raise ValueError("Saved hybrid projector contains unsupported settings")
                sparse, dense = projector_config.get("sparse_dim"), projector_config.get("dense_dim")
                if (type(sparse) is not int or type(dense) is not int or sparse <= 0 or dense <= 0
                        or sparse + dense != dimension):
                    raise ValueError("Saved hybrid projector dimensions do not match the skill")
                runtime_bytes += 16 * 25000 * dense
                runtime_bytes += 16 * len(projector_config.get("spectral_terms", ())) * dense
                runtime_bytes += 8 * sum(len(row) for key in ("spectral_vectors", "subspace_components")
                                         for row in (projector_config.get(key) or ()))
                if runtime_bytes > MAX_SKILL_RUNTIME_BYTES:
                    raise ValueError("Saved skill runtime allocation exceeds memory budget")
        if runtime_bytes > MAX_SKILL_RUNTIME_BYTES:
            raise ValueError("Saved skill runtime allocation exceeds memory budget")
        npz_file = _read_skill_arrays(data[header_end:], schema, dimension, runtime_bytes)
        if (projector_config is None or projector_config.get("type") != "hybrid") and (
            "_projector_spectral_vectors" in npz_file or "_projector_subspace_components" in npz_file
        ):
            raise ValueError("Saved skill contains unexpected hybrid projector arrays")
        if projector_config is not None and projector_config["type"] == "hybrid":
            saved_hybrid = HybridProjector.from_config(
                projector_config,
                backend=backend,
                spectral_vectors=npz_file.get("_projector_spectral_vectors"),
                subspace_components=npz_file.get("_projector_subspace_components"),
            )
        if projector_config is not None:
            if projector is not None:
                if projector_config["type"] == "hybrid" and saved_hybrid is not None:
                    if (
                        type(projector) is not HybridProjector
                        or projector.projector_digest() != saved_hybrid.projector_digest()
                    ):
                        raise ValueError("Supplied projector does not match the saved skill's feature space")
                elif _projector_config(projector) != projector_config:
                    raise ValueError("Supplied projector does not match the saved skill's feature space")
            elif projector_config["type"] == "deterministic":
                projector = DeterministicSemanticProjector(
                    dimension=projector_config["dimension"],
                    recency_weighted=projector_config["recency_weighted"],
                )
            elif saved_hybrid is not None:
                projector = saved_hybrid
            elif saved_tfidf is not None:
                projector = saved_tfidf
            else:
                raise ValueError("This skill requires its original external projector; pass projector= when loading")

        saved_meta = meta.get("metadata", {})
        if not isinstance(saved_meta, dict):
            raise ValueError("Saved skill metadata must be a mapping")
        is_cold_skill = bool(saved_meta.get("cold_start", False))

        # Reconstruct head weights
        heads: Dict[str, CompiledHeadWeights] = {}
        for name, h_info in meta["heads"].items():
            if name not in schema.fields or h_info["field_type"] != schema.fields[name].field_type:
                raise ValueError(f"Saved head {name!r} does not match the schema")
            definition = schema.fields[name]
            w = npz_file[f"{name}_w"]
            b = npz_file[f"{name}_b"]
            widths = (1, 2) if isinstance(definition, BooleanField) else (len(definition.options),) if hasattr(definition, "options") else (1,)
            if w.ndim != 2 or w.shape[0] not in widths or w.shape[1] != dimension or b.shape != (w.shape[0],):
                raise ValueError(f"Saved head {name!r} has incompatible weight dimensions")
            if not np.all(np.isfinite(w)) or not np.all(np.isfinite(b)):
                raise ValueError(f"Saved head {name!r} contains non-finite weights")
            temperature = float(h_info.get("temperature", 1.0))
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError(f"Saved head {name!r} has an invalid temperature")
            if h_info.get("score_method", "aps") not in ("aps", "lac"):
                raise ValueError(f"Saved head {name!r} uses an unsupported conformal score")
            if isinstance(definition, (ChoiceField, MultiChoiceField)) and tuple(h_info.get("options", ())) != definition.options:
                raise ValueError(f"Saved head {name!r} options do not match the schema")
            p_mat = npz_file[f"{name}_P"] if f"{name}_P" in npz_file else None
            b_mat = npz_file[f"{name}_B"] if f"{name}_B" in npz_file else None
            if any(a is not None and not np.all(np.isfinite(a)) for a in (p_mat, b_mat)):
                raise ValueError(f"Saved head {name!r} contains non-finite covariance")
            calib_scores_arr = npz_file[f"{name}_calib_scores"] if f"{name}_calib_scores" in npz_file else None
            calib_scores = tuple(float(s) for s in calib_scores_arr) if calib_scores_arr is not None else ()
            if any(not math.isfinite(score) or score < 0 for score in calib_scores):
                raise ValueError(f"Saved head {name!r} contains invalid calibration scores")
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
            fam_thresh = float(h_info.get("familiarity_threshold", 0.0))
            if not math.isfinite(fam_thresh) or fam_thresh < 0:
                raise ValueError(f"Saved head {name!r} has an invalid familiarity_threshold")
            proto_thresh = float(h_info.get("prototype_threshold", 0.0))
            if not math.isfinite(proto_thresh) or proto_thresh < 0:
                raise ValueError(f"Saved head {name!r} has an invalid prototype_threshold")
            proto_mat = npz_file[f"{name}_prototypes"] if f"{name}_prototypes" in npz_file else None
            if proto_mat is not None and not np.all(np.isfinite(proto_mat)):
                raise ValueError(f"Saved head {name!r} contains non-finite prototypes")
            if (
                proto_mat is None
                and not is_cold_skill
                and "exemplar_state" not in h_info
                and isinstance(definition, (ChoiceField, BooleanField))
                and np.any(w)
            ):
                if w.shape[0] >= 2:
                    w_norms = np.linalg.norm(w, axis=1, keepdims=True)
                    w_norms = np.where(w_norms < 1e-12, 1.0, w_norms)
                    proto_mat = (w / w_norms).astype(np.float32)
                elif isinstance(definition, BooleanField) and w.shape[0] == 1:
                    w_norm = max(1e-12, float(np.linalg.norm(w[0])))
                    w_u = (w[0] / w_norm).astype(np.float32)
                    proto_mat = np.vstack([-w_u, w_u]).astype(np.float32)
            proto_scores_arr = npz_file[f"{name}_proto_scores"] if f"{name}_proto_scores" in npz_file else None
            proto_scores = tuple(float(s) for s in proto_scores_arr) if proto_scores_arr is not None else ()
            if any(not math.isfinite(score) or score < 0 for score in proto_scores):
                raise ValueError(f"Saved head {name!r} contains invalid prototype calibration scores")
            is_hier = bool(h_info.get("hierarchical", False))
            raw_assign = h_info.get("cluster_assignments", ())
            if is_hier and (
                not isinstance(raw_assign, (list, tuple))
                or any(type(c) is not int for c in raw_assign)
            ):
                raise ValueError(f"Saved head {name!r} has invalid cluster_assignments")
            cluster_assign = tuple(int(c) for c in raw_assign) if isinstance(raw_assign, (list, tuple)) else ()
            coarse_w = npz_file[f"{name}_coarse_w"] if f"{name}_coarse_w" in npz_file else None
            coarse_b = npz_file[f"{name}_coarse_b"] if f"{name}_coarse_b" in npz_file else None
            fine_w = npz_file[f"{name}_fine_w"] if f"{name}_fine_w" in npz_file else None
            fine_b = npz_file[f"{name}_fine_b"] if f"{name}_fine_b" in npz_file else None
            if any(a is not None and not np.all(np.isfinite(a)) for a in (coarse_w, coarse_b, fine_w, fine_b)):
                raise ValueError(f"Saved head {name!r} contains non-finite hierarchical weights")
            if is_hier:
                if coarse_w is None or coarse_b is None or len(cluster_assign) != w.shape[0]:
                    raise ValueError(f"Saved head {name!r} has incomplete hierarchical routing configuration")
                if any(c < 0 or c >= coarse_w.shape[0] for c in cluster_assign):
                    raise ValueError(f"Saved head {name!r} has out-of-range cluster_assignments")

            exemplar_arrays = {
                suffix: npz_file[f"{name}_{suffix}"]
                for suffix in EXEMPLAR_ARRAYS
                if f"{name}_{suffix}" in npz_file
            }
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

            heads[name] = CompiledHeadWeights(
                field_name=name,
                field_type=h_info["field_type"],
                weights=w,
                biases=b,
                temperature=float(h_info.get("temperature", 1.0)),
                conformal_quantile=float(h_info.get("conformal_quantile", 0.0)),
                score_method=h_info.get("score_method", "aps"),
                options=tuple(h_info.get("options", [])),
                P=p_mat,
                B=b_mat,
                margin_threshold=float(h_info.get("margin_threshold", 0.20)),
                forgetting_factor=float(h_info.get("forgetting_factor", 0.995)),
                relative_odds_ratio=float(h_info.get("relative_odds_ratio", 1.5)),
                confidence_floor_tau0=float(h_info.get("confidence_floor_tau0", 0.15)),
                escalate_on_ambiguity=bool(h_info.get("escalate_on_ambiguity", True)),
                calibration_scores=calib_scores,
                mondrian=bool(h_info.get("mondrian", False)),
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

        return cls(
            schema=schema,
            heads=heads,
            dimension=dimension,
            projector=projector,
            backend=backend,
            metadata=saved_meta,
            forgetting_factor=float(saved_meta.get("forgetting_factor", 0.995)),
            recency_weighted=bool(saved_meta.get("recency_weighted", False)),
        )

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        projector: Optional[Any] = None,
        backend: str = "numpy",
    ) -> CompiledSystemOneModel:
        """Loads a CompiledSystemOneModel from a .s1m file."""
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"Model file not found: {target}")
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
            raise ImportError("Logistic teaching requires pip install 'system1[teaching]'") from exc

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
                val_range = max(1e-6, f_def.max_value - f_def.min_value)
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
                    target_val = f_def.min_value + fraction * val_range
                    field_samples.append((prompt, float(target_val)))

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
        X_mat = np.asarray(X_train, dtype=np.float32)
        N, D = X_mat.shape
        sw = (
            np.maximum(1e-6, np.asarray(sample_weights, dtype=np.float64).flatten())
            if sample_weights is not None
            else None
        )
        if is_boolean:
            y_flat = np.asarray(Y_train, dtype=np.float32).flatten()
            protos = np.zeros((2, D), dtype=np.float32)
            for idx, mask in enumerate((y_flat < 0.0, y_flat > 0.0)):
                if np.any(mask):
                    vec = (
                        np.average(X_mat[mask], axis=0, weights=sw[mask])
                        if sw is not None
                        else np.mean(X_mat[mask], axis=0)
                    )
                    norm = float(np.linalg.norm(vec))
                    if norm > 1e-12:
                        protos[idx] = (vec / norm).astype(np.float32)
            return protos

        Y_mat = np.asarray(Y_train, dtype=np.float32)
        K = Y_mat.shape[1] if Y_mat.ndim > 1 else 1
        Y_mat = Y_mat.reshape(N, K)
        protos = np.zeros((K, D), dtype=np.float32)
        for k in range(K):
            mask = Y_mat[:, k] > 0.5
            if np.any(mask):
                vec = (
                    np.average(X_mat[mask], axis=0, weights=sw[mask])
                    if sw is not None
                    else np.mean(X_mat[mask], axis=0)
                )
                norm = float(np.linalg.norm(vec))
                if norm > 1e-12:
                    protos[k] = (vec / norm).astype(np.float32)
        return protos

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

        norms = np.linalg.norm(protos, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        unit_protos = protos / norms

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
        protos = np.asarray(prototypes, dtype=np.float32)
        p_norms = np.linalg.norm(protos, axis=1, keepdims=True)
        valid_rows = (p_norms.flatten() > 1e-12)
        if not np.any(valid_rows):
            return 0.0, ()
        protos_valid = (protos[valid_rows] / p_norms[valid_rows]).astype(np.float32)
        X_mat = np.asarray(X_conf, dtype=np.float32)
        x_norms = np.linalg.norm(X_mat, axis=1, keepdims=True)
        x_norms = np.where(x_norms < 1e-12, 1.0, x_norms)
        X_unit = (X_mat / x_norms).astype(np.float32)
        max_sims = np.max(X_unit @ protos_valid.T, axis=1).astype(np.float64)
        finite_sims = max_sims[np.isfinite(max_sims)]
        if len(finite_sims) == 0:
            return 0.0, ()
        sorted_sims = np.sort(finite_sims)
        scores_tuple = tuple(float(v) for v in sorted_sims)
        m = len(sorted_sims)
        rank = int(math.floor((m + 1) * prototype_alpha))
        if rank >= 1:
            thresh = float(sorted_sims[rank - 1])
        else:
            min_sim = float(sorted_sims[0])
            thresh = float(max(0.0, min_sim * 0.90)) if min_sim > 0.0 else 0.0
        return thresh, scores_tuple

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

        compiled_heads: Dict[str, CompiledHeadWeights] = {}
        sample_counts: Dict[str, Dict[str, int]] = {}

        for field_name, f_def in self.schema.fields.items():
            samples = list(dataset.get(field_name, []))
            if not samples:
                if not augment:
                    raise ValueError(f"Teaching requires examples for field {field_name!r}")
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
                    val_range = max(1e-6, f_def.max_value - f_def.min_value)
                    synth_all = self.generate_synthetic_exemplars(samples_per_choice=samples_per_choice)
                    synth_field = synth_all.get(field_name, [])

                    low_synth = [s for s in synth_field if float(s[1]) <= f_def.min_value + 0.33 * val_range]
                    mid_synth = [s for s in synth_field if f_def.min_value + 0.33 * val_range < float(s[1]) <= f_def.min_value + 0.67 * val_range]
                    high_synth = [s for s in synth_field if float(s[1]) > f_def.min_value + 0.67 * val_range]

                    low_real = [s for s in samples if float(s[1]) <= f_def.min_value + 0.33 * val_range]
                    mid_real = [s for s in samples if f_def.min_value + 0.33 * val_range < float(s[1]) <= f_def.min_value + 0.67 * val_range]
                    high_real = [s for s in samples if float(s[1]) > f_def.min_value + 0.67 * val_range]

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
            samples = samples + calibration_samples

            # Prompts and labels
            prompts = [s[0] for s in samples]
            raw_labels = [s[1] for s in samples]

            conf_start_idx = (num_train + num_temperature) if isinstance(f_def, ChoiceField) else num_train
            if conf_start_idx >= len(prompts):
                conf_start_idx = num_train
            need_fam = eff_familiarity_alpha > 0.0 and hasattr(self.projector, "raw_norm") and conf_start_idx < len(prompts)

            # Encode all prompts (and optional continuous telemetry) to embedding matrix X
            X_list = []
            fam_vals: List[float] = []
            proj_cache: Dict[str, Tuple[np.ndarray, Optional[float]]] = {}
            for idx_s, s in enumerate(samples):
                p = s[0]
                t = s[2] if len(s) > 2 else None
                want_raw = need_fam and idx_s >= conf_start_idx
                cached = proj_cache.get(p) if isinstance(p, str) else None
                if cached is not None:
                    emb_base, rn_val = cached
                    if want_raw and rn_val is None:
                        rn_val = float(self.projector.raw_norm(p))
                        proj_cache[p] = (emb_base, rn_val)
                    emb = emb_base
                else:
                    if hasattr(self.projector, "project"):
                        if self.recency_weighted and not isinstance(self.projector, TfidfProjector):
                            try:
                                raw_emb = self.projector.project(p, recency_weighted=self.recency_weighted)
                            except TypeError:
                                raw_emb = self.projector.project(p)
                        else:
                            raw_emb = self.projector.project(p)
                    else:
                        raw_emb = self.projector(p)
                    rn_val = float(self.projector.raw_norm(p)) if want_raw else None
                    norm = float(np.linalg.norm(raw_emb))
                    emb_base = (
                        (raw_emb / norm).astype(np.float32)
                        if norm > 1e-12
                        else np.asarray(raw_emb, dtype=np.float32)
                    )
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
                    s_idx = np.argsort(-p_row)
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
                    s_idx = np.argsort(-np.array(p_row))
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
                val_range = max(1e-6, f_def.max_value - f_def.min_value)
                raw_floats = np.array([float(lab) for lab in raw_labels], dtype=np.float32).reshape(-1, 1)
                # Map target into logit space matching sigmoid evaluation: prob = sigmoid(z / 0.25) => z = 0.25 * ln(r / (1 - r))
                r = np.clip((raw_floats - f_def.min_value) / val_range, 0.01, 0.99)
                Z_all = (0.25 * np.log(r / (1.0 - r))).astype(np.float32)
                Z_train = Z_all[:num_train]
                weights, biases, P_mat, B_mat = self._solve_ridge(
                    X_train, Z_train, self.regularization, regularize_bias=augment, return_covariance=True
                )

                # Calibrate residual bounds on X_calib
                logits_calib = (X_calib @ weights.T + biases).flatten()
                probs_calib = _stable_sigmoid(logits_calib, temperature=0.25)
                pred_vals = f_def.min_value + val_range * probs_calib
                true_calib_floats = raw_floats[num_train:].flatten()
                residuals = [
                    float(abs(pred_vals[i] - true_calib_floats[i]))
                    for i in range(len(pred_vals))
                ]
                sorted_residuals = np.sort(np.asarray(residuals, dtype=np.float64))
                alpha = 0.05
                n_res = len(sorted_residuals)
                k = int(math.ceil((n_res + 1) * (1.0 - alpha)))
                conformal_q = (float(val_range) if k > n_res else float(sorted_residuals[k - 1])) if n_res else 0.0

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
        )
        model.save(output_path)
        return model

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        projector: Optional[Any] = None,
        backend: str = "numpy",
    ) -> CompiledSystemOneModel:
        """Loads a compiled model from a .s1m file."""
        return CompiledSystemOneModel.load(path, projector=projector, backend=backend)


__all__ = [
    "SystemOneCompiler",
    "CompiledSystemOneModel",
    "CompiledHeadWeights",
    "MAGIC_HEADER",
]
