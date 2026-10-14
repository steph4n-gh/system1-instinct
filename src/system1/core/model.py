"""System 1 Core Non-Autoregressive Decision Model.

Executes single-pass schema evaluation directly mapping input prompts to
typed decision fields without token-by-token autoregressive decoding.
Supports Apple Silicon Metal (MLX), NumPy BLAS, and deterministic offline projection.

Zero external dependencies beyond NumPy (and optional Apple Silicon MLX).
Zero cryptography, SQLite, or disk I/O dependencies.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union
import numpy as np

from system1.core.embeddings import (
    _ANCHOR_STOPWORDS,
    _stem_word,
    match_semantic_concepts,
)
from system1.core.schema import (
    BooleanField,
    ChoiceField,
    DecisionField,
    DecisionSchema,
    MultiChoiceField,
    ScoreField,
)
from system1.core.telemetry import TelemetryProjector

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    mx = None
    HAS_MLX = False


class DeterministicSemanticProjector:
    """Deterministic normalized lexical features for local decisions.

    SHA-256-derived word and character features use 384 dimensions by default.
    Shared features can support generalization, but similarity does not guarantee
    semantic equivalence or distinguish all unrelated concepts."""

    def __init__(self, dimension: int = 384, recency_weighted: bool = False):
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self.dimension = dimension
        self.recency_weighted = bool(recency_weighted)
        self._digest = hashlib.sha256(
            f'{{"dimension": {self.dimension}, "recency_weighted": {"true" if self.recency_weighted else "false"}, "type": "deterministic_hash"}}'.encode("utf-8")
        ).hexdigest()

    def projector_digest(self) -> str:
        return self._digest

    def _hash_to_sparse_vec(self, token: str, weight: float = 1.0) -> np.ndarray:
        """Map a token or n-gram deterministically to a sparse pseudo-random vector."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        h = hashlib.sha256(token.encode("utf-8")).digest()
        for i in range(6):
            chunk = h[i * 4 : (i + 1) * 4]
            val = int.from_bytes(chunk, byteorder="big", signed=False)
            idx = val % self.dimension
            sign = 1.0 if ((val >> 16) & 1) == 0 else -1.0
            vec[idx] += sign * weight
        return vec

    def project(self, text: str, recency_weighted: Optional[bool] = None) -> np.ndarray:
        """Project input text into a deterministic dense vector.

        Supports recency weighting for multi-turn agent traces / command histories.
        When recency_weighted=True, trailing tokens receive full weight 1.0 and historical
        context decays backwards from the end:
            weight(i) = log(1 + len(w_i)) / sqrt(1.0 + 0.05 * (N - 1 - i))
        When recency_weighted=False (default), standard forward position decay applies:
            weight(i) = log(1 + len(w_i)) / sqrt(1.0 + 0.05 * i)
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected text to be a string, got {type(text).__name__}")
        if not text or not text.strip():
            v = np.zeros(self.dimension, dtype=np.float32)
            v[0] = 1.0
            return v

        use_recency = self.recency_weighted if recency_weighted is None else bool(recency_weighted)

        clean_text = text.lower().strip()
        if len(clean_text) > 8192:
            clean_text = clean_text[:4096] + " " + clean_text[-4096:]
        words = re.findall(r"\b\w+\b", clean_text)
        n_words = len(words)

        vec = np.zeros(self.dimension, dtype=np.float32)

        # 1. Whole word tokens with position decay and length weighting
        for pos, word in enumerate(words):
            dist = (n_words - 1 - pos) if use_recency else pos
            word_weight = math.log1p(len(word)) / math.sqrt(1.0 + dist * 0.05)
            vec += self._hash_to_sparse_vec(f"word:{word}", weight=word_weight * 2.0)

            # Subword 3-grams and 4-grams for morphological/synonym tolerance
            if len(word) >= 3:
                for n in (3, 4):
                    for j in range(len(word) - n + 1):
                        ngram = word[j : j + n]
                        vec += self._hash_to_sparse_vec(f"ng:{ngram}", weight=0.5)

        # 2. Word pairs / bigrams for local contextual semantics
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            vec += self._hash_to_sparse_vec(f"bi:{bigram}", weight=1.5)

        # 3. Global character 4-grams across word boundaries
        boundary_text = f" {clean_text} "
        for j in range(max(0, len(boundary_text) - 4)):
            c_ngram = boundary_text[j : j + 4]
            vec += self._hash_to_sparse_vec(f"char4:{c_ngram}", weight=0.3)

        # L2-normalization
        max_abs = float(np.max(np.abs(vec))) if len(vec) > 0 else 0.0
        if max_abs < 1e-12 or not np.isfinite(max_abs):
            v = np.zeros(self.dimension, dtype=np.float32)
            v[0] = 1.0
            return v
        scaled = vec / max_abs
        norm = float(np.linalg.norm(scaled))
        if norm < 1e-12:
            vec_zero = np.zeros(self.dimension, dtype=np.float32)
            vec_zero[0] = 1.0
            return vec_zero
        return (scaled / norm).astype(np.float32)

    def _project_concept_aware(self, text: str) -> np.ndarray:
        """Project input text with morphological stems, stopword downweighting, and concept anchors for zero-shot exemplar banks."""
        if not text or not text.strip():
            v = np.zeros(self.dimension, dtype=np.float32)
            v[0] = 1.0
            return v

        clean_text = text.lower().strip()
        if len(clean_text) > 8192:
            clean_text = clean_text[:4096] + " " + clean_text[-4096:]
        words = re.findall(r"\b\w+\b", clean_text)
        n_words = len(words)
        vec = np.zeros(self.dimension, dtype=np.float32)

        stems: List[str] = []
        for pos, word in enumerate(words):
            word_weight = math.log1p(len(word)) / math.sqrt(1.0 + pos * 0.05)
            if word in _ANCHOR_STOPWORDS:
                stems.append(word)
                vec += self._hash_to_sparse_vec(f"word:{word}", weight=word_weight * 0.35)
                continue

            vec += self._hash_to_sparse_vec(f"word:{word}", weight=word_weight * 2.0)
            stem = _stem_word(word)
            stems.append(stem)
            if len(stem) >= 3:
                vec += self._hash_to_sparse_vec(f"stem:{stem}", weight=word_weight * 1.8)
                if stem != word:
                    vec += self._hash_to_sparse_vec(f"word:{stem}", weight=word_weight * 1.2)

            if len(word) >= 3:
                for n in (3, 4):
                    for j in range(len(word) - n + 1):
                        ngram = word[j : j + n]
                        vec += self._hash_to_sparse_vec(f"ng:{ngram}", weight=0.4)

        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            vec += self._hash_to_sparse_vec(f"bi:{bigram}", weight=1.5)
            if stems[i] not in _ANCHOR_STOPWORDS or stems[i + 1] not in _ANCHOR_STOPWORDS:
                s_bigram = f"{stems[i]}_{stems[i+1]}"
                vec += self._hash_to_sparse_vec(f"sbi:{s_bigram}", weight=1.2)

        len_scale = math.sqrt(max(1.0, n_words / 4.0))
        for c_name, c_score in match_semantic_concepts(clean_text).items():
            vec += self._hash_to_sparse_vec(
                f"concept:{c_name}",
                weight=min(7.5, c_score * 2.1) * len_scale,
            )

        boundary_text = f" {clean_text} "
        for j in range(max(0, len(boundary_text) - 4)):
            c_ngram = boundary_text[j : j + 4]
            vec += self._hash_to_sparse_vec(f"char4:{c_ngram}", weight=0.18)

        max_abs = float(np.max(np.abs(vec))) if len(vec) > 0 else 0.0
        if max_abs < 1e-12 or not np.isfinite(max_abs):
            v = np.zeros(self.dimension, dtype=np.float32)
            v[0] = 1.0
            return v
        scaled = vec / max_abs
        norm = float(np.linalg.norm(scaled))
        if norm < 1e-12:
            vec_zero = np.zeros(self.dimension, dtype=np.float32)
            vec_zero[0] = 1.0
            return vec_zero
        return (scaled / norm).astype(np.float32)

    encode = project

    def project_batch(self, texts: Sequence[str], recency_weighted: Optional[bool] = None) -> np.ndarray:
        """Batched projection of multiple text inputs."""
        if texts is None:
            raise TypeError("texts must be a sequence of strings, got None")
        if not hasattr(texts, "__iter__"):
            raise TypeError(f"texts must be a sequence of strings, got {type(texts).__name__}")
        if len(texts) == 0:
            return np.empty((0, self.dimension), dtype=np.float32)
        for idx, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(f"All elements in texts must be strings, got {type(t).__name__} at index {idx}")
        return np.stack([self.project(t, recency_weighted=recency_weighted) for t in texts], axis=0).astype(np.float32)

    def similarity(self, text_a: str, text_b: str, recency_weighted: Optional[bool] = None) -> float:
        """Computes cosine similarity between two text strings."""
        v_a = self.project(text_a, recency_weighted=recency_weighted)
        v_b = self.project(text_b, recency_weighted=recency_weighted)
        return float(np.dot(v_a, v_b))


def _stable_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature."""
    temp = max(1e-4, float(temperature))
    scaled = np.asarray(logits, dtype=np.float64) / temp
    max_val = np.max(scaled, axis=-1, keepdims=True)
    exp = np.exp(scaled - max_val)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _stable_sigmoid(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable sigmoid with temperature."""
    temp = max(1e-4, float(temperature))
    scaled = np.asarray(logits, dtype=np.float64) / temp
    pos_mask = scaled >= 0
    out = np.empty_like(scaled)
    out[pos_mask] = 1.0 / (1.0 + np.exp(-scaled[pos_mask]))
    exp_neg = np.exp(scaled[~pos_mask])
    out[~pos_mask] = exp_neg / (1.0 + exp_neg)
    return out


# Public aliases for stable softmax and sigmoid
stable_softmax = _stable_softmax
stable_sigmoid = _stable_sigmoid


@dataclass(frozen=True)
class RawFieldEvaluation:
    """Intermediate raw evaluation for a single decision field before calibration."""

    field_name: str
    field_type: str
    logits: np.ndarray
    raw_probabilities: np.ndarray
    options: Tuple[str, ...] = ()
    selected_value: Any = None
    confidence: float = 0.0


@dataclass(frozen=True)
class ModelInferenceResult:
    """Raw model output across all fields for an evaluated prompt."""

    prompt: str
    embedding: np.ndarray
    fields: Dict[str, RawFieldEvaluation]
    inference_latency_ms: float


class DecisionFieldHead:
    """Head projecting dense embeddings to logits for a specific schema field."""

    def __init__(
        self,
        field_def: DecisionField,
        dimension: int = 384,
        backend: str = "numpy",
        projector: Optional[Any] = None,
        contrastive_whitening: bool = True,
        forgetting_factor: float = 0.995,
        initialize_weights: bool = True,
    ) -> None:
        self.field_def = field_def
        self.dimension = dimension
        self.backend = backend
        self.projector = projector
        self.contrastive_whitening = contrastive_whitening
        self.forgetting_factor = float(forgetting_factor)
        self.weights: np.ndarray
        self.biases: np.ndarray
        self.mx_weights: Any = None
        self.mx_biases: Any = None
        self.covariance_inv: Optional[np.ndarray] = None
        self.cross_covariance: Optional[np.ndarray] = None
        self.regularization: float = 1.0
        self.hierarchical: bool = False
        self.cluster_assignments: Tuple[int, ...] = ()
        self.coarse_weights: Optional[np.ndarray] = None
        self.coarse_biases: Optional[np.ndarray] = None
        self.fine_weights: Optional[np.ndarray] = None
        self.fine_biases: Optional[np.ndarray] = None
        self._exemplar_bank: Optional[Tuple[np.ndarray, List[np.ndarray]]] = None
        self._exemplar_mean: Optional[np.ndarray] = None
        self._exemplar_projector: Optional[DeterministicSemanticProjector] = None
        self._initial_weights: Optional[np.ndarray] = None
        self._online_shots_p: Optional[List[List[np.ndarray]]] = None
        self._online_shots_c: Optional[List[List[np.ndarray]]] = None
        if initialize_weights:
            self._init_head()
        else:
            width = len(field_def.options) if isinstance(field_def, (ChoiceField, MultiChoiceField)) else 1
            self.weights = np.zeros((width, dimension), dtype=np.float32)
            self.biases = np.zeros(width, dtype=np.float32)

    def _build_concept_aware_exemplar_bank(self) -> None:
        """Constructs the concept-aware prototype and sub-prototype bank for a ChoiceField."""
        if not isinstance(self.field_def, ChoiceField):
            return
        num_options = len(self.field_def.options)
        if num_options <= 1:
            return
        ex_proj = DeterministicSemanticProjector(dimension=self.dimension)
        self._exemplar_projector = ex_proj
        stem_opt_df: Dict[str, set[int]] = {}
        for idx, opt in enumerate(self.field_def.options):
            desc = self.field_def.descriptions.get(opt, "")
            clean_opt = opt.replace("_", " ").replace("-", " ")
            texts_k = (
                list(desc)
                if isinstance(desc, (list, tuple)) and len(desc) > 0
                else [
                    s.strip()
                    for s in re.split(r"[;,]", str(desc) if desc else clean_opt)
                    if s.strip()
                ]
            )
            for t_str in texts_k:
                for w in re.findall(r"\b\w+\b", str(t_str).lower()):
                    if w not in _ANCHOR_STOPWORDS and len(w) >= 2:
                        st = _stem_word(w)
                        stem_opt_df.setdefault(st, set()).add(idx)
        shared_thresh = max(2, (num_options + 1) // 2)
        shared_stems = {s for s, opts in stem_opt_df.items() if len(opts) >= shared_thresh}

        W_cons = np.zeros((num_options, self.dimension), dtype=np.float32)
        raw_sub_banks: List[np.ndarray] = []
        for idx, opt in enumerate(self.field_def.options):
            desc = self.field_def.descriptions.get(opt, "")
            clean_opt = opt.replace("_", " ").replace("-", " ")
            texts_k = (
                list(desc)
                if isinstance(desc, (list, tuple)) and len(desc) > 0
                else [
                    s.strip()
                    for s in re.split(r"[;,]", str(desc) if desc else clean_opt)
                    if s.strip()
                ]
            )
            filtered_clauses: List[str] = []
            sub_vecs: List[np.ndarray] = []
            for ex in texts_k:
                ex_str = str(ex).strip()
                if not ex_str:
                    continue
                ex_words = re.findall(r"\b\w+\b", ex_str.lower())
                uniq_words = [
                    w
                    for w in ex_words
                    if w not in _ANCHOR_STOPWORDS and _stem_word(w) not in shared_stems
                ]
                clause_text = " ".join(uniq_words) if len(uniq_words) >= 2 else ex_str
                filtered_clauses.append(clause_text)
                v_sub = ex_proj._project_concept_aware(clause_text)
                n_sub = float(np.linalg.norm(v_sub))
                sub_vecs.append(v_sub / (n_sub if n_sub > 1e-12 else 1.0))

            cons_text = " ".join(filtered_clauses).strip() or clean_opt
            v_cons = ex_proj._project_concept_aware(cons_text)
            n_cons = float(np.linalg.norm(v_cons))
            W_cons[idx] = v_cons / (n_cons if n_cons > 1e-12 else 1.0)
            if sub_vecs:
                raw_sub_banks.append(np.stack(sub_vecs, axis=0).astype(np.float32))
            else:
                raw_sub_banks.append(W_cons[idx : idx + 1].copy())

        mu_cons = np.mean(W_cons, axis=0, keepdims=True)
        W_cons_c = W_cons - mu_cons
        norms_c = np.linalg.norm(W_cons_c, axis=1, keepdims=True)
        norms_c = np.where(norms_c < 1e-12, 1.0, norms_c)
        W_cons_norm = (W_cons_c / norms_c).astype(np.float32)

        centered_sub_banks: List[np.ndarray] = []
        for bank in raw_sub_banks:
            b_c = bank - mu_cons
            b_norms = np.linalg.norm(b_c, axis=1, keepdims=True)
            b_norms = np.where(b_norms < 1e-12, 1.0, b_norms)
            centered_sub_banks.append((b_c / b_norms).astype(np.float32))
        self._exemplar_mean = mu_cons.flatten().astype(np.float32)
        self._exemplar_bank = (W_cons_norm, centered_sub_banks)

    def _init_head(self) -> None:
        if self.projector is None:
            self.projector = DeterministicSemanticProjector(dimension=self.dimension)
        projector = self.projector

        if isinstance(self.field_def, ChoiceField):
            num_options = len(self.field_def.options)
            W = np.zeros((num_options, self.dimension), dtype=np.float32)
            has_multi_exemplars = False
            for idx, opt in enumerate(self.field_def.options):
                desc = self.field_def.descriptions.get(opt, "")
                clean_opt = opt.replace("_", " ").replace("-", " ")
                if isinstance(desc, (list, tuple)) and len(desc) > 0:
                    has_multi_exemplars = True
                    ex_vecs = []
                    for ex in desc:
                        ex_text = str(ex).strip()
                        if ex_text:
                            v = projector.project(ex_text)
                            norm_v = float(np.linalg.norm(v))
                            ex_vecs.append(v / (norm_v if norm_v > 1e-12 else 1.0))
                    if ex_vecs:
                        centroid = np.mean(ex_vecs, axis=0)
                        norm_c = float(np.linalg.norm(centroid))
                        W[idx] = centroid / (norm_c if norm_c > 1e-12 else 1.0)
                    else:
                        semantic_text = f"{self.field_def.name} option {opt} {clean_opt}".strip()
                        v = projector.project(semantic_text)
                        norm_v = float(np.linalg.norm(v))
                        W[idx] = v / (norm_v if norm_v > 1e-12 else 1.0)
                else:
                    semantic_text = f"{self.field_def.name} option {opt} {clean_opt}: {desc}".strip()
                    v = projector.project(semantic_text)
                    norm_v = float(np.linalg.norm(v))
                    W[idx] = v / (norm_v if norm_v > 1e-12 else 1.0)

            has_delimited_exemplar_lists = (
                len(self.field_def.descriptions) == num_options
                and all(
                    isinstance(self.field_def.descriptions.get(opt, ""), str)
                    and (
                        ";" in str(self.field_def.descriptions.get(opt, ""))
                        or str(self.field_def.descriptions.get(opt, "")).count(",") >= 3
                    )
                    for opt in self.field_def.options
                )
            )

            # Contrastive centering and whitening across choices
            if self.contrastive_whitening and num_options > 1:
                mu = np.mean(W, axis=0, keepdims=True)
                W_centered = W - mu
                norms = np.linalg.norm(W_centered, axis=1, keepdims=True)
                norms = np.where(norms < 1e-12, 1.0, norms)
                self.weights = (W_centered / norms).astype(np.float32)

                if has_multi_exemplars or has_delimited_exemplar_lists or num_options > 8:
                    self._build_concept_aware_exemplar_bank()
            else:
                norms = np.linalg.norm(W, axis=1, keepdims=True)
                norms = np.where(norms < 1e-12, 1.0, norms)
                self.weights = (W / norms).astype(np.float32)

            self._initial_weights = self.weights.copy()
            self.biases = np.zeros(num_options, dtype=np.float32)

        elif isinstance(self.field_def, BooleanField):
            t_text = f"{self.field_def.name}: {self.field_def.true_description}"
            f_text = f"{self.field_def.name}: {self.field_def.false_description}"
            t_vec = projector.project(t_text)
            f_vec = projector.project(f_text)
            t_norm = t_vec / (np.linalg.norm(t_vec) or 1.0)
            f_norm = f_vec / (np.linalg.norm(f_vec) or 1.0)
            diff = t_norm - f_norm
            norm = np.linalg.norm(diff)
            w = (diff / norm if norm > 0 else t_norm).astype(np.float32)
            self.weights = w.reshape(1, self.dimension)
            thresh = self.field_def.threshold
            b0 = math.log(thresh / (1.0 - thresh)) if 0.0 < thresh < 1.0 else 0.0
            self.biases = np.array([-b0], dtype=np.float32)

        elif isinstance(self.field_def, MultiChoiceField):
            num_options = len(self.field_def.options)
            W = np.zeros((num_options, self.dimension), dtype=np.float32)
            for idx, opt in enumerate(self.field_def.options):
                desc = self.field_def.descriptions.get(opt, "")
                clean_opt = opt.replace("_", " ").replace("-", " ")
                if isinstance(desc, (list, tuple)) and len(desc) > 0:
                    ex_vecs = []
                    for ex in desc:
                        ex_text = str(ex).strip()
                        if ex_text:
                            v = projector.project(ex_text)
                            norm_v = float(np.linalg.norm(v))
                            ex_vecs.append(v / (norm_v if norm_v > 1e-12 else 1.0))
                    if ex_vecs:
                        centroid = np.mean(ex_vecs, axis=0)
                        norm_c = float(np.linalg.norm(centroid))
                        W[idx] = centroid / (norm_c if norm_c > 1e-12 else 1.0)
                    else:
                        semantic_text = f"{self.field_def.name} tag {opt} {clean_opt}".strip()
                        v = projector.project(semantic_text)
                        norm_v = float(np.linalg.norm(v))
                        W[idx] = v / (norm_v if norm_v > 1e-12 else 1.0)
                else:
                    semantic_text = f"{self.field_def.name} tag {opt} {clean_opt}: {desc}".strip()
                    v = projector.project(semantic_text)
                    norm_v = float(np.linalg.norm(v))
                    W[idx] = v / (norm_v if norm_v > 1e-12 else 1.0)

            norms = np.linalg.norm(W, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            self.weights = (W / norms).astype(np.float32)

            thresh = self.field_def.threshold
            b0 = (0.25 * math.log(thresh / (1.0 - thresh))) if 0.0 < thresh < 1.0 else 0.0
            # Apply baseline margin to prevent unflagged background text from spuriously triggering tags
            self.biases = np.full(num_options, b0 - 0.15, dtype=np.float32)

        elif isinstance(self.field_def, ScoreField):
            l_text = f"{self.field_def.name} low: {self.field_def.low_description}"
            h_text = f"{self.field_def.name} high: {self.field_def.high_description}"
            l_vec = projector.project(l_text)
            h_vec = projector.project(h_text)
            l_norm = l_vec / (np.linalg.norm(l_vec) or 1.0)
            h_norm = h_vec / (np.linalg.norm(h_vec) or 1.0)
            diff = h_norm - l_norm
            norm = np.linalg.norm(diff)
            w = (diff / norm if norm > 0 else h_norm).astype(np.float32)
            self.weights = w.reshape(1, self.dimension)
            self.biases = np.zeros(1, dtype=np.float32)
        else:
            raise TypeError(f"Unsupported field type: {type(self.field_def).__name__}")

        if self.backend == "mlx" and HAS_MLX and mx is not None:
            self.mx_weights = mx.array(self.weights)
            self.mx_biases = mx.array(self.biases)

    def set_weights(
        self,
        weights: np.ndarray,
        biases: Optional[np.ndarray] = None,
        *,
        clear_exemplar_bank: bool = True,
    ) -> None:
        """Updates head weights and biases, synchronizing MLX buffers if active."""
        if clear_exemplar_bank:
            self._exemplar_bank = None
            self._exemplar_mean = None
            self._online_shots_p = None
            self._online_shots_c = None
        self.weights = np.asarray(weights, dtype=np.float32)
        if biases is not None:
            self.biases = np.asarray(biases, dtype=np.float32)
        else:
            self.biases = np.zeros(self.weights.shape[0], dtype=np.float32)
        if self.backend == "mlx" and HAS_MLX and mx is not None:
            self.mx_weights = mx.array(self.weights)
            self.mx_biases = mx.array(self.biases)

    def set_hierarchical_weights(
        self,
        *,
        cluster_assignments: Sequence[int],
        coarse_weights: np.ndarray,
        coarse_biases: np.ndarray,
        fine_weights: Optional[np.ndarray] = None,
        fine_biases: Optional[np.ndarray] = None,
    ) -> None:
        """Configures two-stage coarse-to-fine hierarchical routing parameters."""
        cw = np.asarray(coarse_weights, dtype=np.float32)
        cb = np.asarray(coarse_biases, dtype=np.float32).flatten()
        assign = tuple(int(c) for c in cluster_assignments)
        if cw.ndim != 2 or cb.shape[0] != cw.shape[0]:
            raise ValueError("coarse_weights and coarse_biases shape mismatch")
        if any(c < 0 or c >= cw.shape[0] for c in assign):
            raise ValueError("cluster_assignments contains out-of-bounds cluster index")
        self.hierarchical = True
        self.cluster_assignments = assign
        self.coarse_weights = cw
        self.coarse_biases = cb
        self.fine_weights = (
            np.asarray(fine_weights, dtype=np.float32) if fine_weights is not None else None
        )
        self.fine_biases = (
            np.asarray(fine_biases, dtype=np.float32).flatten() if fine_biases is not None else None
        )

    def init_covariance(
        self,
        P: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        regularization: float = 1.0,
    ) -> None:
        """Initializes covariance inverse P and cross-covariance B for Sherman-Morrison online distillation.

        If P is None, initializes P = (1/lambda) * I_(D+1) and B = lambda * [W, b]^T,
        so (P B)^T == [W, b] identically reproducing existing weights.
        """
        self.regularization = max(1e-4, float(regularization))
        D = self.dimension
        if P is not None:
            self.covariance_inv = np.asarray(P, dtype=np.float32)
        else:
            self.covariance_inv = (1.0 / self.regularization) * np.eye(D + 1, dtype=np.float32)

        if B is not None:
            self.cross_covariance = np.asarray(B, dtype=np.float32)
        else:
            w_mat = self.weights  # shape (K, D)
            b_vec = self.biases.reshape(-1, 1)  # shape (K, 1)
            w_aug = np.hstack([w_mat, b_vec])  # shape (K, D+1)
            self.cross_covariance = (self.regularization * w_aug.T).astype(np.float32)

    def format_target_vector(self, target_value: Any) -> np.ndarray:
        """Converts high-level field target value into numeric regression target vector y*."""
        if isinstance(self.field_def, ChoiceField):
            opts = self.field_def.options
            y = np.zeros(len(opts), dtype=np.float32)
            if target_value in opts:
                idx = opts.index(target_value)
                y[idx] = 1.0
            elif isinstance(target_value, int) and 0 <= target_value < len(opts):
                y[target_value] = 1.0
            return y
        elif isinstance(self.field_def, BooleanField):
            val = bool(target_value)
            return np.array([1.0 if val else -1.0], dtype=np.float32)
        elif isinstance(self.field_def, ScoreField):
            val = float(target_value)
            val_range = max(1e-6, self.field_def.max_value - self.field_def.min_value)
            # Inverse sigmoid logit transformation matching _solve_ridge:
            # prob = sigmoid(z / 0.25) => z = 0.25 * ln(r / (1 - r))
            r = float(np.clip((val - self.field_def.min_value) / val_range, 1e-4, 1.0 - 1e-4))
            logit_target = 0.25 * math.log(r / (1.0 - r))
            return np.array([logit_target], dtype=np.float32)
        elif isinstance(self.field_def, MultiChoiceField):
            opts = self.field_def.options
            y = np.full(len(opts), -1.0, dtype=np.float32)
            if isinstance(target_value, (list, tuple, set)):
                for idx, opt in enumerate(opts):
                    if opt in target_value:
                        y[idx] = 1.0
            elif target_value in opts:
                y[opts.index(target_value)] = 1.0
            return y
        return np.asarray(target_value, dtype=np.float32)

    def export_exemplar_state(self) -> Optional[Dict[str, Any]]:
        """Snapshot the nonlinear routing state that cannot be recovered from weights."""
        if self._exemplar_bank is None:
            return None
        centroids, banks = self._exemplar_bank
        arrays = {"exemplar_centroids": centroids.copy()}
        counts: Dict[str, List[int]] = {}
        for key, groups in (("exemplar_vectors", banks), ("shots_p", self._online_shots_p), ("shots_c", self._online_shots_c)):
            if groups is not None:
                counts[key] = [len(group) for group in groups]
                arrays[key] = np.concatenate([
                    np.asarray(group).reshape(-1, self.dimension) for group in groups
                ], axis=0)
        if self._exemplar_mean is not None:
            arrays["exemplar_mean"] = self._exemplar_mean.copy()
        if self._initial_weights is not None:
            arrays["initial_w"] = self._initial_weights.copy()
        return {"arrays": arrays, "counts": counts,
                "online_shots_are_pseudo": bool(getattr(self, "_online_shots_are_pseudo", False))}

    def restore_exemplar_state(self, state: Dict[str, Any]) -> None:
        arrays, counts = state["arrays"], state["counts"]
        width = len(self.field_def.options) if isinstance(self.field_def, ChoiceField) else 0
        if not width or self.hierarchical or "exemplar_centroids" not in arrays or "exemplar_vectors" not in arrays:
            raise ValueError("Saved exemplar state requires a non-hierarchical choice head")
        for key, arr in arrays.items():
            expected = (self.dimension,) if key == "exemplar_mean" else (width, self.dimension)
            if key in ("exemplar_vectors", "shots_p", "shots_c"):
                sizes = counts.get(key)
                if (not isinstance(sizes, (list, tuple)) or len(sizes) != width
                        or any(type(n) is not int or n < 0 for n in sizes)):
                    raise ValueError("Saved exemplar group counts are invalid")
                expected = (sum(sizes), self.dimension)
            if arr.shape != expected or not np.all(np.isfinite(arr)):
                raise ValueError(f"Saved exemplar array {key!r} is invalid")
        groups: Dict[str, List[np.ndarray]] = {}
        for key in ("exemplar_vectors", "shots_p", "shots_c"):
            if key in arrays:
                groups[key] = list(np.split(arrays[key], np.cumsum(counts[key])[:-1]))
        self._exemplar_bank = (arrays["exemplar_centroids"], groups["exemplar_vectors"])
        self._exemplar_mean = arrays.get("exemplar_mean")
        self._initial_weights = arrays.get("initial_w")
        self._online_shots_p = [list(group) for group in groups["shots_p"]] if "shots_p" in groups else None
        self._online_shots_c = [list(group) for group in groups["shots_c"]] if "shots_c" in groups else None
        self._online_shots_are_pseudo = state.get("online_shots_are_pseudo", False)
        self._exemplar_projector = DeterministicSemanticProjector(dimension=self.dimension)

    def _update_exemplar_bank_entry(
        self,
        emb_feat: np.ndarray,
        target_k: int,
        prompt: Optional[str] = None,
        *,
        blend_rate: float = 0.28,
        attenuate_competitors: bool = True,
    ) -> None:
        """Updates the live multi-exemplar sub-prototype bank for ChoiceField class target_k."""
        if not isinstance(self.field_def, ChoiceField) or self.hierarchical:
            return
        K = self.weights.shape[0]
        D = self.dimension
        if target_k < 0 or target_k >= K:
            return
        if self._exemplar_bank is None:
            return
        if prompt is not None and self._exemplar_projector is not None:
            c_vec = self._exemplar_projector._project_concept_aware(prompt)
            if self._exemplar_mean is not None and len(self._exemplar_mean) == D:
                c_vec = (c_vec - self._exemplar_mean).astype(np.float32)
            c_norm = float(np.linalg.norm(c_vec))
            if c_norm <= 1e-12:
                return
            c_full = (c_vec / c_norm).astype(np.float32)
            # Boost novel co-occurring vocabulary not already present in the initial description of target_k
            opt_name = self.field_def.options[target_k]
            desc_k = self.field_def.descriptions.get(opt_name, "")
            txts_k = desc_k if isinstance(desc_k, (list, tuple)) else [str(desc_k), opt_name.replace("_", " ")]
            known_stems_k = {
                _stem_word(w)
                for t in txts_k
                for w in re.findall(r"\b\w+\b", str(t).lower())
            }
            words = re.findall(r"\b\w+\b", prompt.lower())
            novel_words = [
                w
                for w in words
                if w not in _ANCHOR_STOPWORDS and _stem_word(w) not in known_stems_k
            ]
            if novel_words and len(novel_words) < len(words):
                c_nov_raw = self._exemplar_projector._project_concept_aware(" ".join(novel_words))
                if self._exemplar_mean is not None and len(self._exemplar_mean) == D:
                    c_nov_raw = (c_nov_raw - self._exemplar_mean).astype(np.float32)
                cn_norm = float(np.linalg.norm(c_nov_raw))
                if cn_norm > 1e-12:
                    c_nov = (c_nov_raw / cn_norm).astype(np.float32)
                    c_mix = 0.45 * c_full + 0.55 * c_nov
                    cm_norm = float(np.linalg.norm(c_mix))
                    c_unit = (c_mix / (cm_norm if cm_norm > 1e-12 else 1.0)).astype(np.float32)
                else:
                    c_unit = c_full
            else:
                c_unit = c_full
        else:
            c_vec = np.asarray(emb_feat, dtype=np.float32).flatten()[:D].copy()
            if self._exemplar_mean is not None and len(self._exemplar_mean) == D:
                c_vec = (c_vec - self._exemplar_mean).astype(np.float32)
            c_norm = float(np.linalg.norm(c_vec))
            if c_norm <= 1e-12:
                return
            c_unit = (c_vec / c_norm).astype(np.float32)

        w_cons, sub_banks = self._exemplar_bank
        w_cons_new = w_cons.copy()
        w_blend = (1.0 - blend_rate) * w_cons_new[target_k] + blend_rate * c_unit
        wb_norm = float(np.linalg.norm(w_blend))
        if wb_norm > 1e-12:
            w_cons_new[target_k] = (w_blend / wb_norm).astype(np.float32)

        sim_w_thresh = 0.35 if K > 8 else 0.15
        sim_b_thresh = 0.42 if K > 8 else 0.20

        new_sub_banks = list(sub_banks)
        for k_idx in range(K):
            if k_idx == target_k:
                bank_k = new_sub_banks[k_idx]
                if bank_k.size > 0:
                    updated_bank = np.vstack([bank_k[-63:], c_unit.reshape(1, -1)]).astype(np.float32)
                else:
                    updated_bank = c_unit.reshape(1, -1).astype(np.float32)
                new_sub_banks[k_idx] = updated_bank
            elif attenuate_competitors:
                sim_w = float(w_cons_new[k_idx] @ c_unit)
                if sim_w > sim_w_thresh:
                    w_orth = w_cons_new[k_idx] - 0.35 * sim_w * c_unit
                    wo_norm = float(np.linalg.norm(w_orth))
                    if wo_norm > 1e-12:
                        w_cons_new[k_idx] = (w_orth / wo_norm).astype(np.float32)
                bank_k = new_sub_banks[k_idx]
                if bank_k.size > 0:
                    sims_b = bank_k @ c_unit
                    mask_high = sims_b > sim_b_thresh
                    if np.any(mask_high):
                        b_mod = bank_k.copy()
                        b_mod[mask_high] = b_mod[mask_high] - 0.45 * np.outer(sims_b[mask_high], c_unit)
                        b_norms = np.linalg.norm(b_mod, axis=1, keepdims=True)
                        b_norms = np.where(b_norms < 1e-12, 1.0, b_norms)
                        new_sub_banks[k_idx] = (b_mod / b_norms).astype(np.float32)
        self._exemplar_bank = (w_cons_new, new_sub_banks)

    def adapt_zero_shot(
        self,
        embeddings: np.ndarray,
        prompts: Optional[Union[str, Sequence[str]]] = None,
        *,
        margin_threshold: float = 0.18,
        learning_rate: float = 0.25,
    ) -> Dict[str, Any]:
        """Unsupervised test-time zero-shot prototype and exemplar adaptation.

        Refines class prototypes and appends high-margin domain exemplars from
        unlabeled inputs whose zero-shot probability margin exceeds ``margin_threshold``,
        bridging co-occurring domain vocabulary without requiring teacher calls.
        """
        embs = np.asarray(embeddings, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        if embs.size == 0 or not isinstance(self.field_def, ChoiceField):
            return {"adapted_samples": 0, "class_counts": {}}

        prompt_list: Optional[List[str]] = None
        if isinstance(prompts, str):
            prompt_list = [prompts]
        elif prompts is not None:
            prompt_list = [str(p) for p in prompts]

        # Ensure _exemplar_bank exists on non-hierarchical ChoiceField in proper concept-aware space
        if self._exemplar_bank is None and not self.hierarchical:
            if isinstance(self.projector, DeterministicSemanticProjector):
                self._build_concept_aware_exemplar_bank()
            else:
                K_init = self.weights.shape[0]
                w_init = self.weights.copy()
                b_init = [self.weights[idx : idx + 1].copy() for idx in range(K_init)]
                self._exemplar_bank = (w_init, b_init)

        raw_logits = self.compute_logits(embs, prompts=prompts)
        if raw_logits.ndim == 1:
            raw_logits = raw_logits.reshape(1, -1)
        probs = _stable_softmax(raw_logits / 0.25, temperature=1.0)
        N, K = probs.shape
        if K < 2:
            return {"adapted_samples": 0, "class_counts": {}}

        # High-cardinality multi-class orthogonal subspace adaptation (K > 8)
        if K > 8 and prompt_list is not None and len(prompt_list) == N and self._exemplar_projector is not None:
            logits_sorted = np.sort(raw_logits, axis=1)
            logit_margins = logits_sorted[:, -1] - logits_sorted[:, -2]
            preds = np.argmax(raw_logits, axis=1)
            shots_p: List[List[np.ndarray]] = [[] for _ in range(K)]
            shots_c: List[List[np.ndarray]] = [[] for _ in range(K)]
            class_counts: Dict[str, int] = {}
            adapted_samples = 0
            m_th = 0.10
            top_m = 3
            for k_idx in range(K):
                idxs = [
                    i
                    for i in range(N)
                    if int(preds[i]) == k_idx and float(logit_margins[i]) >= m_th
                ]
                idxs.sort(key=lambda i: float(logit_margins[i]), reverse=True)
                selected = idxs[:top_m]
                if selected:
                    opt_name = self.field_def.options[k_idx]
                    class_counts[opt_name] = len(selected)
                    adapted_samples += len(selected)
                    for i in selected:
                        vp = embs[i]
                        np_n = float(np.linalg.norm(vp))
                        if np_n > 1e-12:
                            shots_p[k_idx].append((vp / np_n).astype(np.float32))
                        vc = self._exemplar_projector._project_concept_aware(prompt_list[i])
                        nc_n = float(np.linalg.norm(vc))
                        if nc_n > 1e-12:
                            shots_c[k_idx].append((vc / nc_n).astype(np.float32))
            if adapted_samples > 0 and (
                self._online_shots_p is None or getattr(self, "_online_shots_are_pseudo", False)
            ):
                self._online_shots_p = shots_p
                self._online_shots_c = shots_c
                self._online_shots_are_pseudo = True
            return {"adapted_samples": adapted_samples, "class_counts": class_counts}

        mu_bg = np.mean(self.weights, axis=0)
        if N >= 4:
            valid_embs = [embs[i] for i in range(N) if float(np.linalg.norm(embs[i])) > 1e-12]
            if len(valid_embs) >= 4:
                mu_bg = (0.70 * mu_bg + 0.30 * np.mean(valid_embs, axis=0)).astype(np.float32)

        class_vecs: Dict[int, List[np.ndarray]] = {}
        class_weights_map: Dict[int, List[float]] = {}
        class_counts = {}
        adapted_samples = 0

        for i in range(N):
            emb_i = embs[i]
            if float(np.linalg.norm(emb_i)) <= 1e-12:
                continue
            p_i = probs[i]
            sorted_p = np.sort(p_i)
            margin = float(sorted_p[-1] - sorted_p[-2])
            if margin < margin_threshold:
                continue
            k_best = int(np.argmax(p_i))
            conf_w = float(
                min(1.0, (margin - margin_threshold) / max(0.1, 1.0 - margin_threshold) + 0.35)
            )
            u_i = emb_i - mu_bg
            u_norm = float(np.linalg.norm(u_i))
            if u_norm > 1e-12:
                u_unit = (u_i / u_norm).astype(np.float32)
                class_vecs.setdefault(k_best, []).append(u_unit)
                class_weights_map.setdefault(k_best, []).append(conf_w)
                opt_name = self.field_def.options[k_best]
                class_counts[opt_name] = class_counts.get(opt_name, 0) + 1
                adapted_samples += 1
                p_str = prompt_list[i] if (prompt_list is not None and i < len(prompt_list)) else None
                self._update_exemplar_bank_entry(
                    emb_i,
                    k_best,
                    prompt=p_str,
                    blend_rate=float(min(0.35, learning_rate * conf_w)),
                    attenuate_competitors=False,
                )

        if adapted_samples > 0:
            new_W = self.weights.copy()
            for k_idx, vecs in class_vecs.items():
                ws = np.asarray(class_weights_map[k_idx], dtype=np.float32).reshape(-1, 1)
                pseudo_proto = np.sum(np.stack(vecs, axis=0) * ws, axis=0) / max(1e-6, float(np.sum(ws)))
                p_norm = float(np.linalg.norm(pseudo_proto))
                if p_norm > 1e-12:
                    pseudo_unit = (pseudo_proto / p_norm).astype(np.float32)
                    eff_lr = float(min(0.45, learning_rate * min(1.5, 0.7 + 0.15 * len(vecs))))
                    w_blended = (1.0 - eff_lr) * new_W[k_idx] + eff_lr * pseudo_unit
                    wb_norm = float(np.linalg.norm(w_blended))
                    if wb_norm > 1e-12:
                        new_W[k_idx] = (w_blended / wb_norm).astype(np.float32)
            self.set_weights(new_W, self.biases, clear_exemplar_bank=False)

        return {"adapted_samples": adapted_samples, "class_counts": class_counts}

    def online_update(
        self,
        x_aug: np.ndarray,
        y_target: np.ndarray,
        forgetting_factor: Optional[float] = None,
        *,
        prompt: Optional[str] = None,
        preserve_calibration: bool = False,
    ) -> float:
        r"""Closed-form rank-1 Sherman-Morrison update with exponential forgetting factor.

        P_{t+1} = (1 / \lambda_f) [ P_t - (P_t x x^T P_t) / (\lambda_f + x^T P_t x) ]
        B_{t+1} = \lambda_f B_t + x y^T
        W_{t+1} = (P_{t+1} B_{t+1})^T
        """
        t0 = time.perf_counter()
        if self.covariance_inv is None or self.cross_covariance is None:
            self.init_covariance(regularization=self.regularization)

        lam_f = float(
            forgetting_factor if forgetting_factor is not None else self.forgetting_factor
        )
        if not (0.0 < lam_f <= 1.0):
            raise ValueError(f"Forgetting factor lambda_f must be in (0, 1.0], got {lam_f}")

        P = self.covariance_inv
        B = self.cross_covariance
        D = self.dimension
        K = self.weights.shape[0]

        x = np.asarray(x_aug, dtype=np.float32).flatten()
        if len(x) == D:
            # If prompt embedding without bias 1, augment
            x = np.append(x, 1.0).astype(np.float32)
        elif len(x) != D + 1:
            raise ValueError(f"Expected x_aug of length {D} or {D+1}, got {len(x)}")

        if not np.all(np.isfinite(x)):
            raise ValueError("x_aug contains NaN or Inf non-finite values")

        y = np.asarray(y_target, dtype=np.float32).flatten()
        if not np.all(np.isfinite(y)):
            raise ValueError("y_target contains NaN or Inf non-finite values")
        if len(y) != K:
            raise ValueError(f"Expected target vector y of length {K}, got {len(y)}")

        x_step = x
        y_step = y
        if not preserve_calibration and isinstance(self.field_def, ChoiceField) and K > 8:
            pos_max = float(np.max(y))
            if pos_max > 0.0:
                neg_balanced = -pos_max / float(max(1, K - 1))
                y_step = np.where(y > 0.0, y, neg_balanced).astype(np.float32)
        if preserve_calibration and float(np.linalg.norm(x[:D])) > 1e-12:
            x_step = np.append(x[:D], 0.0).astype(np.float32)
            v_dir = P @ x_step
            g = float(x_step @ v_dir)
            if g > 1e-12:
                z0 = (self.weights @ x[:D] + self.biases).astype(np.float64)
                b0 = self.biases.astype(np.float64)
                z_desired = z0.copy()
                if isinstance(self.field_def, ChoiceField) and K >= 2:
                    t_idx = int(np.argmax(y))
                    comp_max = float(np.max([z0[i] for i in range(K) if i != t_idx]))
                    for i in range(K):
                        if i != t_idx and z0[i] > 0.15:
                            z_desired[i] = 0.05
                    comp_after = float(np.max([z_desired[i] for i in range(K) if i != t_idx]))
                    z_desired[t_idx] = max(float(z0[t_idx]), comp_after + 1.45, 1.45)
                elif isinstance(self.field_def, BooleanField):
                    sign = 1.0 if float(y[0]) >= 0.0 else -1.0
                    z_desired[0] = sign * max(abs(float(z0[0])), 1.45)
                elif isinstance(self.field_def, MultiChoiceField):
                    for i in range(K):
                        sign = 1.0 if float(y[i]) >= 0.0 else -1.0
                        z_desired[i] = sign * max(abs(float(z0[i])), 0.90)
                elif isinstance(self.field_def, ScoreField):
                    z_desired[0] = float(y[0])
                y_des = z_desired - b0
                z_curr_dir = z0 - b0
                y_step = (((lam_f + g) * y_des - lam_f * z_curr_dir) / g).astype(np.float32)

        # v = P @ x_step (shape: D+1,)
        v = P @ x_step
        denom = float(lam_f + x_step @ v)
        if denom <= 1e-12 or not math.isfinite(denom):
            # Degenerate input or numerical singularity: preserve internal state
            return (time.perf_counter() - t0) * 1000.0

        # Rank-1 update with exponential forgetting factor:
        # P_{t+1} = (1 / lam_f) * [ P_t - (v v^T) / denom ]
        P_next = (1.0 / lam_f) * (P - (np.outer(v, v) / denom))
        P_next = 0.5 * (P_next + P_next.T)

        # Cross-covariance update: B_{t+1} = lam_f * B_t + x y^T
        B_next = (lam_f * B) + np.outer(x_step, y_step)

        # Covariance bounding: prevent covariance windup along unexcited subspace directions
        p_max = float(50.0 / max(1e-4, self.regularization))
        max_d = float(np.max(np.diag(P_next)))
        if max_d > p_max and max_d > 0:
            scale = p_max / max_d
            P_next *= scale
            B_next *= (1.0 / scale)

        # Enforce strict positive-definiteness on diagonal against floating-point drift
        np.fill_diagonal(P_next, np.maximum(np.diag(P_next), 1e-6))

        # Weight update: W_aug = (P_{t+1} @ B_{t+1})^T
        W_aug = (P_next @ B_next).T
        new_W = W_aug[:, :D].astype(np.float32)
        new_b = W_aug[:, D].astype(np.float32)

        # Update live cold-start exemplar bank if active
        if not self.hierarchical and isinstance(self.field_def, ChoiceField):
            if self._exemplar_bank is not None:
                target_k = int(np.argmax(y))
                if 0 <= target_k < K and float(y[target_k]) > 0.0:
                    if K <= 8:
                        self._update_exemplar_bank_entry(
                            x[:D],
                            target_k,
                            prompt=prompt,
                            blend_rate=0.35 if preserve_calibration else 0.28,
                            attenuate_competitors=True,
                        )
                    elif isinstance(self.projector, DeterministicSemanticProjector):
                        if self._online_shots_p is None or getattr(self, "_online_shots_are_pseudo", False):
                            self._online_shots_p = [[] for _ in range(K)]
                            self._online_shots_c = [[] for _ in range(K)]
                            self._online_shots_are_pseudo = False
                        xp = x[:D].copy()
                        np_norm = float(np.linalg.norm(xp))
                        if np_norm > 1e-12:
                            self._online_shots_p[target_k].append((xp / np_norm).astype(np.float32))
                        if prompt is not None and self._exemplar_projector is not None:
                            xc = self._exemplar_projector._project_concept_aware(prompt)
                            nc_norm = float(np.linalg.norm(xc))
                            if nc_norm > 1e-12:
                                self._online_shots_c[target_k].append((xc / nc_norm).astype(np.float32))

        # Also update coarse cluster router and fine cluster specialist weights if hierarchical routing is active
        if (
            self.hierarchical
            and self.coarse_weights is not None
            and self.coarse_biases is not None
            and len(self.cluster_assignments) == K
        ):
            target_k = int(np.argmax(y))
            if 0 <= target_k < K and float(y[target_k]) > 0.0:
                target_c = self.cluster_assignments[target_k]
                if 0 <= target_c < self.coarse_weights.shape[0]:
                    x_feat = x[:D]
                    c_logits = self.coarse_weights @ x_feat + self.coarse_biases
                    err_c = np.zeros_like(c_logits)
                    if c_logits[target_c] < 1.2:
                        err_c[target_c] = 1.2 - c_logits[target_c]
                    for c_idx in range(len(c_logits)):
                        if c_idx != target_c and c_logits[c_idx] > 0.1:
                            err_c[c_idx] = -0.1 - c_logits[c_idx]
                    step = float(min(0.35, 1.0 / max(1.0, float(x_feat @ x_feat))))
                    self.coarse_weights = (self.coarse_weights + step * np.outer(err_c, x_feat)).astype(np.float32)
                    if (
                        self.fine_weights is not None
                        and self.fine_biases is not None
                        and self.fine_weights.shape == (K, D)
                        and self.fine_biases.shape == (K,)
                    ):
                        f_logits = self.fine_weights @ x_feat + self.fine_biases
                        err_f = np.zeros_like(f_logits)
                        if f_logits[target_k] < 1.5:
                            err_f[target_k] = 1.5 - f_logits[target_k]
                        for k_idx in range(K):
                            if k_idx != target_k and f_logits[k_idx] > 0.1:
                                err_f[k_idx] = -0.1 - f_logits[k_idx]
                        self.fine_weights = (self.fine_weights + step * np.outer(err_f, x_feat)).astype(np.float32)

        # Atomic commit (preserve _exemplar_bank across online updates)
        self.covariance_inv = P_next.astype(np.float32)
        self.cross_covariance = B_next.astype(np.float32)
        self.set_weights(new_W, new_b, clear_exemplar_bank=False)
        return (time.perf_counter() - t0) * 1000.0

    def evaluate_logits(self, logits: np.ndarray) -> RawFieldEvaluation:
        """Constructs a RawFieldEvaluation from precomputed logits."""
        logits = np.asarray(logits, dtype=np.float64).flatten()
        if isinstance(self.field_def, ChoiceField):
            scaled_logits = logits / 0.25
            probs = _stable_softmax(scaled_logits, temperature=1.0)
            best_idx = int(np.argmax(probs))
            selected = self.field_def.options[best_idx]
            conf = float(probs[best_idx])
            return RawFieldEvaluation(
                field_name=self.field_def.name,
                field_type=self.field_def.field_type,
                logits=scaled_logits,
                raw_probabilities=probs,
                options=self.field_def.options,
                selected_value=selected,
                confidence=conf,
            )

        elif isinstance(self.field_def, BooleanField):
            prob = float(_stable_sigmoid(logits, temperature=0.25)[0])
            val = bool(prob >= self.field_def.threshold)
            conf = prob if val else (1.0 - prob)
            scaled_z = float(logits[0] / 0.25)
            two_class_logits = np.array([-scaled_z / 2.0, scaled_z / 2.0], dtype=np.float64)
            return RawFieldEvaluation(
                field_name=self.field_def.name,
                field_type=self.field_def.field_type,
                logits=two_class_logits,
                raw_probabilities=np.array([1.0 - prob, prob], dtype=np.float64),
                options=("False", "True"),
                selected_value=val,
                confidence=conf,
            )

        elif isinstance(self.field_def, MultiChoiceField):
            scaled_logits = logits / 0.25
            probs = _stable_sigmoid(scaled_logits, temperature=1.0)
            selected = tuple(
                self.field_def.options[i]
                for i, p in enumerate(probs)
                if p >= self.field_def.threshold
            )
            conf = float(np.mean([max(p, 1.0 - p) for p in probs])) if len(probs) > 0 else 1.0
            return RawFieldEvaluation(
                field_name=self.field_def.name,
                field_type=self.field_def.field_type,
                logits=scaled_logits,
                raw_probabilities=probs,
                options=self.field_def.options,
                selected_value=selected,
                confidence=conf,
            )

        elif isinstance(self.field_def, ScoreField):
            scaled_logits = logits / 0.25
            prob = float(_stable_sigmoid(scaled_logits, temperature=1.0)[0])
            val = self.field_def.min_value + (self.field_def.max_value - self.field_def.min_value) * prob
            return RawFieldEvaluation(
                field_name=self.field_def.name,
                field_type=self.field_def.field_type,
                logits=scaled_logits,
                raw_probabilities=np.array([prob], dtype=np.float64),
                options=(),
                selected_value=float(val),
                confidence=float(max(prob, 1.0 - prob)),
            )

        raise TypeError(f"Unknown field type: {self.field_def}")

    def compute_logits(
        self,
        embedding: np.ndarray,
        prompts: Optional[Union[str, Sequence[str]]] = None,
    ) -> np.ndarray:
        """Compute raw logits (including two-stage coarse-to-fine routing when active)."""
        emb = np.asarray(embedding, dtype=np.float32)
        single = emb.ndim == 1
        if single:
            emb = emb.reshape(1, -1)

        if (
            self.backend == "mlx"
            and HAS_MLX
            and mx is not None
            and self.mx_weights is not None
            and not self.hierarchical
            and self._exemplar_bank is None
        ):
            mx_emb = mx.array(emb)
            mx_logits = mx.matmul(mx_emb, mx.transpose(self.mx_weights)) + self.mx_biases
            mx.eval(mx_logits)
            logits = np.array(mx_logits)
        else:
            logits = emb @ self.weights.T + self.biases
            if self._exemplar_bank is not None and not self.hierarchical:
                w_cons, sub_banks = self._exemplar_bank
                if prompts is not None and self._exemplar_projector is not None:
                    if isinstance(prompts, str):
                        c_emb = self._exemplar_projector._project_concept_aware(prompts).reshape(1, -1)
                    else:
                        c_emb = np.stack(
                            [self._exemplar_projector._project_concept_aware(str(p)) for p in prompts],
                            axis=0,
                        ).astype(np.float32)
                else:
                    c_emb = emb
                cons_logits = c_emb @ w_cons.T
                max_ex = np.zeros_like(logits)
                for k, bank in enumerate(sub_banks):
                    if bank.size > 0:
                        max_ex[:, k] = np.max(c_emb @ bank.T, axis=1)
                w0_logits = (
                    emb @ self._initial_weights.T
                    if self._online_shots_p is not None and self._initial_weights is not None
                    else logits
                )
                base_logits = (0.20 * w0_logits + 0.40 * cons_logits + 0.60 * max_ex).astype(np.float32)
                if self._online_shots_p is not None and self.weights.shape[0] > 8:
                    K_cls = self.weights.shape[0]
                    D_dim = self.dimension
                    all_p = [v for k in range(K_cls) for v in self._online_shots_p[k]]
                    all_c = (
                        [v for k in range(K_cls) for v in self._online_shots_c[k]]
                        if self._online_shots_c is not None
                        else []
                    )
                    if all_p:
                        mu_p = np.mean(all_p, axis=0)
                        mu_c = np.mean(all_c, axis=0) if all_c else np.zeros(D_dim, dtype=np.float32)
                        W_shot_p = np.zeros((K_cls, D_dim), dtype=np.float32)
                        W_shot_c = np.zeros((K_cls, D_dim), dtype=np.float32)
                        for k in range(K_cls):
                            if self._online_shots_p[k]:
                                cp = np.mean(self._online_shots_p[k], axis=0) - mu_p
                                np_n = float(np.linalg.norm(cp))
                                if np_n > 1e-12:
                                    W_shot_p[k] = (cp / np_n).astype(np.float32)
                            if self._online_shots_c is not None and self._online_shots_c[k]:
                                cc = np.mean(self._online_shots_c[k], axis=0) - mu_c
                                nc_n = float(np.linalg.norm(cc))
                                if nc_n > 1e-12:
                                    W_shot_c[k] = (cc / nc_n).astype(np.float32)
                        shot_logits = 0.55 * (emb @ W_shot_p.T) + 0.45 * (c_emb @ W_shot_c.T)
                        avg_shots = float(len(all_p)) / float(max(1, K_cls))
                        alpha = min(0.65, 0.28 * avg_shots / (1.0 + 0.25 * avg_shots))
                        base_logits = ((1.0 - alpha) * base_logits + alpha * shot_logits).astype(np.float32)
                logits = base_logits
            elif (
                self.hierarchical
                and self.coarse_weights is not None
                and self.coarse_biases is not None
                and len(self.cluster_assignments) == self.weights.shape[0]
            ):
                coarse_logits = emb @ self.coarse_weights.T + self.coarse_biases
                coarse_probs = _stable_softmax(coarse_logits / 0.25, temperature=1.0)
                assign = np.asarray(self.cluster_assignments, dtype=np.int64)
                mapped_coarse = coarse_logits[:, assign]
                mapped_probs = coarse_probs[:, assign]
                if self.fine_weights is not None and self.fine_biases is not None:
                    spec_logits = emb @ self.fine_weights.T + self.fine_biases
                    logits = (
                        0.65 * logits
                        + (0.20 + 0.30 * mapped_probs) * spec_logits
                        + 0.20 * mapped_coarse
                    ).astype(np.float32)
                else:
                    logits = (0.80 * logits + 0.20 * mapped_coarse).astype(np.float32)

        return logits.flatten() if single else logits

    def forward(self, embedding: np.ndarray, prompt: Optional[str] = None) -> RawFieldEvaluation:
        """Projects dense embedding into logits and raw probabilities."""
        logits = self.compute_logits(embedding, prompts=prompt)
        return self.evaluate_logits(logits)


class SystemOneModel:
    """Complete non-autoregressive decision model executing single-pass schema evaluation.

    Encodes input text into a high-dimensional dense representation and routes it
    through calibrated multi-task decision heads.
    Supports Apple Silicon Metal (MLX) GPU execution and optimized NumPy BLAS.
    """

    def __init__(
        self,
        schema: Union[DecisionSchema, Type[DecisionSchema]],
        *,
        dimension: int = 384,
        backend: str = "auto",
        projector: Optional[Any] = None,
        contrastive_whitening: bool = True,
        forgetting_factor: float = 0.995,
        recency_weighted: bool = False,
    ) -> None:
        if isinstance(schema, type) and issubclass(schema, DecisionSchema):
            self.schema: DecisionSchema = schema()
        elif isinstance(schema, DecisionSchema):
            self.schema = schema
        else:
            raise TypeError(f"Expected DecisionSchema subclass or instance, got {type(schema).__name__}")

        self.dimension = dimension
        self.backend = self._resolve_backend(backend)
        self.contrastive_whitening = contrastive_whitening
        self.forgetting_factor = float(forgetting_factor)
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

        self.heads: Dict[str, DecisionFieldHead] = {
            name: DecisionFieldHead(
                f,
                dimension=self.dimension,
                backend=self.backend,
                projector=self.projector,
                contrastive_whitening=self.contrastive_whitening,
                forgetting_factor=self.forgetting_factor,
            )
            for name, f in self.schema.fields.items()
        }

        self._lock = threading.RLock()

        # Pre-warm backend kernels (e.g. Metal GPU JIT shader pipeline compilation)
        if self.backend == "mlx" and HAS_MLX and mx is not None:
            dummy_emb = np.zeros((1, self.dimension), dtype=np.float32)
            for head in self.heads.values():
                head.forward(dummy_emb)

    def _resolve_backend(self, backend: str) -> str:
        if backend == "auto":
            return "mlx" if HAS_MLX else "numpy"
        if backend == "mlx":
            if not HAS_MLX:
                raise RuntimeError("MLX requested but mlx is not installed on this system")
            return "mlx"
        return "numpy"

    def encode(
        self,
        prompt: str,
        telemetry: Optional[Any] = None,
        recency_weighted: Optional[bool] = None,
    ) -> np.ndarray:
        """Encodes prompt into a normalized dense embedding vector, fusing continuous telemetry if present."""
        if not isinstance(prompt, str):
            raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}")
        use_recency = self.recency_weighted if recency_weighted is None else bool(recency_weighted)
        if hasattr(self.projector, "project"):
            try:
                emb = self.projector.project(prompt, recency_weighted=use_recency)
            except TypeError:
                emb = self.projector.project(prompt)
        else:
            emb = self.projector(prompt)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        if telemetry is not None:
            emb = self.telemetry_projector.fuse(emb, telemetry)
        return emb

    def encode_batch(
        self,
        prompts: Sequence[str],
        telemetry: Optional[Sequence[Any]] = None,
        recency_weighted: Optional[bool] = None,
    ) -> np.ndarray:
        """Encodes a sequence of prompts (and optional telemetry) into normalized dense embedding vectors."""
        if prompts is None:
            raise TypeError("prompts must be a sequence of strings, got None")
        if not hasattr(prompts, "__iter__"):
            raise TypeError(f"prompts must be a sequence of strings, got {type(prompts).__name__}")
        if len(prompts) == 0:
            return np.empty((0, self.dimension), dtype=np.float32)
        for idx, p in enumerate(prompts):
            if not isinstance(p, str):
                raise TypeError(f"All elements in prompts must be strings, got {type(p).__name__} at index {idx}")
        use_recency = self.recency_weighted if recency_weighted is None else bool(recency_weighted)
        if telemetry is not None and len(telemetry) == len(prompts):
            return np.stack([self.encode(p, telemetry[i], recency_weighted=use_recency) for i, p in enumerate(prompts)], axis=0).astype(np.float32)
        if hasattr(self.projector, "project_batch"):
            try:
                embs = self.projector.project_batch(prompts, recency_weighted=use_recency)
            except TypeError:
                embs = self.projector.project_batch(prompts)
        else:
            embs = np.stack([self.encode(p, recency_weighted=use_recency) for p in prompts], axis=0).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return (embs / norms).astype(np.float32)

    def forward_single(
        self,
        prompt: str,
        telemetry: Optional[Any] = None,
        embedding: Optional[np.ndarray] = None,
        recency_weighted: Optional[bool] = None,
    ) -> ModelInferenceResult:
        """Executes a single non-autoregressive forward pass over the schema."""
        start_t = time.perf_counter()
        use_recency = self.recency_weighted if recency_weighted is None else bool(recency_weighted)
        if embedding is not None:
            emb = np.asarray(embedding, dtype=np.float32).flatten()
        else:
            emb = self.encode(prompt, telemetry=telemetry, recency_weighted=use_recency)

        field_results: Dict[str, RawFieldEvaluation] = {}
        prompt_for_head = prompt if embedding is None else None
        for name, head in self.heads.items():
            field_results[name] = head.forward(emb, prompt=prompt_for_head)

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return ModelInferenceResult(
            prompt=prompt,
            embedding=emb,
            fields=field_results,
            inference_latency_ms=elapsed_ms,
        )

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
        preserve_calibration: bool = False,
    ) -> Dict[str, Any]:
        """Closed-form rank-1 Sherman-Morrison online update on the metal (<0.1ms)."""
        with self._lock:
            target = self.schema.validate_lesson(target)
            t0 = time.perf_counter()
            if embedding is not None:
                emb = np.asarray(embedding, dtype=np.float32).flatten()
            else:
                emb = self.encode(prompt, telemetry=telemetry, recency_weighted=recency_weighted)
            x_aug = np.append(emb, 1.0).astype(np.float32)

            target_dict: Dict[str, Any]
            if isinstance(target, Mapping):
                target_dict = dict(target)
            else:
                first_field = next(iter(self.schema.fields.keys()))
                target_dict = {first_field: target}

            updated_fields: List[str] = []
            update_durations_ms: List[float] = []
            prompt_for_head = prompt if (isinstance(prompt, str) and prompt.strip()) else None

            for f_name, target_val in target_dict.items():
                if f_name in self.heads:
                    head = self.heads[f_name]
                    y_target = head.format_target_vector(target_val)
                    dt = head.online_update(
                        x_aug,
                        y_target,
                        forgetting_factor=forgetting_factor,
                        prompt=prompt_for_head,
                        preserve_calibration=preserve_calibration,
                    )
                    update_durations_ms.append(dt)
                    updated_fields.append(f_name)

            rank1_ms = float(sum(update_durations_ms)) if update_durations_ms else 0.0
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
            for f_name, head in self.heads.items():
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
        """First-class online teaching method that preserves calibration by default."""
        eff_ff = forgetting_factor
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
        if resolved_target is None:
            raise ValueError("learn() requires a target label/mapping or outcome")

        res = self.learn_from_tier2(
            prompt,
            resolved_target,
            telemetry=telemetry,
            embedding=embedding,
            forgetting_factor=eff_ff,
            recency_weighted=recency_weighted,
            preserve_calibration=preserve_calibration,
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

    def forward_batch(
        self,
        prompts: Sequence[str],
        telemetry: Optional[Sequence[Any]] = None,
        recency_weighted: Optional[bool] = None,
    ) -> List[ModelInferenceResult]:
        """Batched forward pass over multiple inputs with Metal GPU and NumPy BLAS optimization."""
        if prompts is None:
            raise TypeError("prompts must be a sequence of strings, got None")
        if not hasattr(prompts, "__iter__"):
            raise TypeError(f"prompts must be a sequence of strings, got {type(prompts).__name__}")
        if len(prompts) == 0:
            return []

        start_t = time.perf_counter()
        use_recency = self.recency_weighted if recency_weighted is None else bool(recency_weighted)
        embs = self.encode_batch(prompts, telemetry=telemetry, recency_weighted=use_recency)

        head_logits: Dict[str, np.ndarray] = {}
        for name, head in self.heads.items():
            head_logits[name] = head.compute_logits(embs, prompts=prompts)

        total_elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        per_sample_latency = total_elapsed_ms / len(prompts)

        results: List[ModelInferenceResult] = []
        for i, p in enumerate(prompts):
            f_res: Dict[str, RawFieldEvaluation] = {}
            for name, head in self.heads.items():
                f_res[name] = head.evaluate_logits(head_logits[name][i])
            results.append(
                ModelInferenceResult(
                    prompt=p,
                    embedding=embs[i],
                    fields=f_res,
                    inference_latency_ms=per_sample_latency,
                )
            )
        return results


__all__ = [
    "DeterministicSemanticProjector",
    "DecisionFieldHead",
    "RawFieldEvaluation",
    "ModelInferenceResult",
    "SystemOneModel",
    "HAS_MLX",
    "_stable_softmax",
    "_stable_sigmoid",
    "stable_softmax",
    "stable_sigmoid",
]
