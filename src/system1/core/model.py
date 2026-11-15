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
import struct
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Type, Union
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


_WORD_RE = re.compile(r"\b\w+\b")
_UNPACK_6I = struct.Struct(">6I").unpack_from
_SHA256 = hashlib.sha256


class DeterministicSemanticProjector:
    """Deterministic normalized lexical features for local decisions.

    SHA-256-derived word and character features use 384 dimensions by default.
    Shared features can support generalization, but similarity does not guarantee
    semantic equivalence or distinguish all unrelated concepts."""

    def __init__(
        self,
        dimension: int = 384,
        recency_weighted: bool = False,
        *,
        token_cache: bool = True,
        max_token_cache: int = 65536,
    ):
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self.dimension = dimension
        self.recency_weighted = bool(recency_weighted)
        self.token_cache_enabled = bool(token_cache)
        self.max_token_cache = max(1, int(max_token_cache))
        self._tok_hash_cache: Dict[str, Tuple[int, int, int, int, int, int, float, float, float, float, float, float]] = {}
        self._digest = hashlib.sha256(
            f'{{"dimension": {self.dimension}, "recency_weighted": {"true" if self.recency_weighted else "false"}, "type": "deterministic_hash"}}'.encode("utf-8")
        ).hexdigest()

    def projector_digest(self) -> str:
        return self._digest

    def clear_token_cache(self) -> None:
        self._tok_hash_cache.clear()

    def _hash_token(
        self, token: str
    ) -> Tuple[int, int, int, int, int, int, float, float, float, float, float, float]:
        if self.token_cache_enabled:
            cached = self._tok_hash_cache.get(token)
            if cached is not None:
                return cached
        digest = _SHA256(token.encode("utf-8")).digest()
        v0, v1, v2, v3, v4, v5 = _UNPACK_6I(digest)
        dim = self.dimension
        res = (
            v0 % dim,
            v1 % dim,
            v2 % dim,
            v3 % dim,
            v4 % dim,
            v5 % dim,
            1.0 if ((v0 >> 16) & 1) == 0 else -1.0,
            1.0 if ((v1 >> 16) & 1) == 0 else -1.0,
            1.0 if ((v2 >> 16) & 1) == 0 else -1.0,
            1.0 if ((v3 >> 16) & 1) == 0 else -1.0,
            1.0 if ((v4 >> 16) & 1) == 0 else -1.0,
            1.0 if ((v5 >> 16) & 1) == 0 else -1.0,
        )
        if self.token_cache_enabled:
            if len(self._tok_hash_cache) >= self.max_token_cache:
                self._tok_hash_cache.clear()
            self._tok_hash_cache[token] = res
        return res

    def _hash_to_sparse_vec(self, token: str, weight: float = 1.0) -> np.ndarray:
        """Map a token or n-gram deterministically to a sparse pseudo-random vector."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        i0, i1, i2, i3, i4, i5, s0, s1, s2, s3, s4, s5 = self._hash_token(token)
        vec[i0] += s0 * weight
        vec[i1] += s1 * weight
        vec[i2] += s2 * weight
        vec[i3] += s3 * weight
        vec[i4] += s4 * weight
        vec[i5] += s5 * weight
        return vec

    def _accumulate_and_normalize(
        self,
        weights_map: Dict[str, float],
        local_memo: Optional[Dict[str, Tuple[int, int, int, int, int, int, float, float, float, float, float, float]]] = None,
    ) -> np.ndarray:
        dim = self.dimension
        if not weights_map:
            out = np.zeros(dim, dtype=np.float32)
            out[0] = 1.0
            return out

        vec_list = [0.0] * dim
        tok_cache_on = self.token_cache_enabled
        cache = self._tok_hash_cache if tok_cache_on else local_memo
        max_cache = self.max_token_cache
        cache_get = cache.get if cache is not None else None
        sha256 = _SHA256
        unpack_6i = _UNPACK_6I

        for tok, w in weights_map.items():
            if w == 0.0:
                continue
            res = cache_get(tok) if cache_get is not None else None
            if res is None:
                digest = sha256(tok.encode("utf-8")).digest()
                v0, v1, v2, v3, v4, v5 = unpack_6i(digest)
                res = (
                    v0 % dim,
                    v1 % dim,
                    v2 % dim,
                    v3 % dim,
                    v4 % dim,
                    v5 % dim,
                    1.0 if ((v0 >> 16) & 1) == 0 else -1.0,
                    1.0 if ((v1 >> 16) & 1) == 0 else -1.0,
                    1.0 if ((v2 >> 16) & 1) == 0 else -1.0,
                    1.0 if ((v3 >> 16) & 1) == 0 else -1.0,
                    1.0 if ((v4 >> 16) & 1) == 0 else -1.0,
                    1.0 if ((v5 >> 16) & 1) == 0 else -1.0,
                )
                if cache is not None:
                    if tok_cache_on and len(cache) >= max_cache:
                        cache.clear()
                    cache[tok] = res
            i0, i1, i2, i3, i4, i5, s0, s1, s2, s3, s4, s5 = res
            vec_list[i0] += s0 * w
            vec_list[i1] += s1 * w
            vec_list[i2] += s2 * w
            vec_list[i3] += s3 * w
            vec_list[i4] += s4 * w
            vec_list[i5] += s5 * w

        vec = np.array(vec_list, dtype=np.float32)
        max_abs = float(np.max(np.abs(vec))) if len(vec) > 0 else 0.0
        if max_abs < 1e-12 or not math.isfinite(max_abs):
            vec.fill(0.0)
            vec[0] = 1.0
            return vec
        vec /= max_abs
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec /= norm
        else:
            vec.fill(0.0)
            vec[0] = 1.0
        return vec

    def _extract_token_weights(self, text: str, use_recency: bool) -> Dict[str, float]:
        """Extracts weighted word, n-gram, bigram, and character features for base projection."""
        if not text or not text.strip():
            return {}

        clean_text = text.lower().strip()
        if len(clean_text) > 8192:
            clean_text = clean_text[:4096] + " " + clean_text[-4096:]
        words = _WORD_RE.findall(clean_text)
        n_words = len(words)

        wmap: Dict[str, float] = {}
        wmap_get = wmap.get
        log1p = math.log1p
        sqrt = math.sqrt

        for pos, word in enumerate(words):
            dist = (n_words - 1 - pos) if use_recency else pos
            word_weight = log1p(len(word)) / sqrt(1.0 + dist * 0.05)
            wk = f"word:{word}"
            wmap[wk] = wmap_get(wk, 0.0) + word_weight * 2.0

            wlen = len(word)
            if wlen >= 3:
                for n in (3, 4):
                    for j in range(wlen - n + 1):
                        nk = f"ng:{word[j : j + n]}"
                        wmap[nk] = wmap_get(nk, 0.0) + 0.5

        for i in range(n_words - 1):
            bk = f"bi:{words[i]}_{words[i + 1]}"
            wmap[bk] = wmap_get(bk, 0.0) + 1.5

        boundary_text = f" {clean_text} "
        limit = len(boundary_text) - 4
        for j in range(max(0, limit)):
            ck = f"char4:{boundary_text[j : j + 4]}"
            wmap[ck] = wmap_get(ck, 0.0) + 0.3

        return wmap

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
        use_recency = self.recency_weighted if recency_weighted is None else bool(recency_weighted)
        wmap = self._extract_token_weights(text, use_recency)
        return self._accumulate_and_normalize(wmap)

    def _project_concept_aware(self, text: str) -> np.ndarray:
        """Project input text with morphological stems, stopword downweighting, and concept anchors for zero-shot exemplar banks."""
        _, c_vec = self.project_both(text, need_base=False, need_concept=True)
        assert c_vec is not None
        return c_vec

    def project_both(
        self,
        text: str,
        *,
        recency_weighted: Optional[bool] = None,
        need_base: bool = True,
        need_concept: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Single-pass extraction sharing regex, ngrams, bigrams, and char4 slices."""
        if not text or not text.strip():
            v_b = None
            v_c = None
            if need_base:
                v_b = np.zeros(self.dimension, dtype=np.float32)
                v_b[0] = 1.0
            if need_concept:
                v_c = np.zeros(self.dimension, dtype=np.float32)
                v_c[0] = 1.0
            return v_b, v_c

        use_recency = self.recency_weighted if recency_weighted is None else bool(recency_weighted)
        clean_text = text.lower().strip()
        if len(clean_text) > 8192:
            clean_text = clean_text[:4096] + " " + clean_text[-4096:]

        words = _WORD_RE.findall(clean_text)
        n_words = len(words)
        base_map: Optional[Dict[str, float]] = {} if need_base else None
        conc_map: Optional[Dict[str, float]] = {} if need_concept else None

        log1p = math.log1p
        sqrt = math.sqrt
        stems: List[str] = [""] * n_words if need_concept else []

        for pos, word in enumerate(words):
            lw = log1p(len(word))
            is_stop = word in _ANCHOR_STOPWORDS
            if need_base:
                assert base_map is not None
                dist = (n_words - 1 - pos) if use_recency else pos
                w_base = lw / sqrt(1.0 + dist * 0.05)
                wk = f"word:{word}"
                base_map[wk] = base_map.get(wk, 0.0) + w_base * 2.0

            if need_concept:
                assert conc_map is not None
                w_conc = lw / sqrt(1.0 + pos * 0.05)
                wk = f"word:{word}"
                if is_stop:
                    conc_map[wk] = conc_map.get(wk, 0.0) + w_conc * 0.35
                    stems[pos] = word
                else:
                    conc_map[wk] = conc_map.get(wk, 0.0) + w_conc * 2.0
                    stem = _stem_word(word)
                    stems[pos] = stem
                    if len(stem) >= 3:
                        sk = f"stem:{stem}"
                        conc_map[sk] = conc_map.get(sk, 0.0) + w_conc * 1.8
                        if stem != word:
                            wsk = f"word:{stem}"
                            conc_map[wsk] = conc_map.get(wsk, 0.0) + w_conc * 1.2

            wlen = len(word)
            if wlen >= 3:
                emit_conc_ng = need_concept
                for n in (3, 4):
                    for j in range(wlen - n + 1):
                        nk = f"ng:{word[j : j + n]}"
                        if need_base:
                            assert base_map is not None
                            base_map[nk] = base_map.get(nk, 0.0) + 0.5
                        if emit_conc_ng:
                            assert conc_map is not None
                            conc_map[nk] = conc_map.get(nk, 0.0) + 0.4

        for i in range(n_words - 1):
            bk = f"bi:{words[i]}_{words[i + 1]}"
            if need_base:
                assert base_map is not None
                base_map[bk] = base_map.get(bk, 0.0) + 1.5
            if need_concept:
                assert conc_map is not None
                conc_map[bk] = conc_map.get(bk, 0.0) + 1.5
                s0, s1 = stems[i], stems[i + 1]
                if s0 not in _ANCHOR_STOPWORDS or s1 not in _ANCHOR_STOPWORDS:
                    sbk = f"sbi:{s0}_{s1}"
                    conc_map[sbk] = conc_map.get(sbk, 0.0) + 1.2

        if need_concept and clean_text:
            assert conc_map is not None
            len_scale = sqrt(max(1.0, n_words / 4.0))
            for c_name, c_score in match_semantic_concepts(clean_text).items():
                ck = f"concept:{c_name}"
                conc_map[ck] = conc_map.get(ck, 0.0) + min(7.5, c_score * 2.1) * len_scale

        if clean_text:
            boundary_text = f" {clean_text} "
            limit = len(boundary_text) - 4
            for j in range(max(0, limit)):
                ck = f"char4:{boundary_text[j : j + 4]}"
                if need_base:
                    assert base_map is not None
                    base_map[ck] = base_map.get(ck, 0.0) + 0.3
                if need_concept:
                    assert conc_map is not None
                    conc_map[ck] = conc_map.get(ck, 0.0) + 0.18

        local_memo = {} if (need_base and need_concept and not self.token_cache_enabled) else None
        v_base = (
            self._accumulate_and_normalize(base_map, local_memo=local_memo)
            if need_base and base_map is not None
            else None
        )
        v_conc = (
            self._accumulate_and_normalize(conc_map, local_memo=local_memo)
            if need_concept and conc_map is not None
            else None
        )
        return v_base, v_conc

    encode = project

    def project_batch(self, texts: Sequence[str], recency_weighted: Optional[bool] = None) -> np.ndarray:
        """Batched projection of multiple text inputs using shared token hashing and BLAS."""
        if texts is None:
            raise TypeError("texts must be a sequence of strings, got None")
        if not hasattr(texts, "__iter__"):
            raise TypeError(f"texts must be a sequence of strings, got {type(texts).__name__}")
        n_docs = len(texts)
        if n_docs == 0:
            return np.empty((0, self.dimension), dtype=np.float32)
        for idx, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(f"All elements in texts must be strings, got {type(t).__name__} at index {idx}")

        use_recency = self.recency_weighted if recency_weighted is None else bool(recency_weighted)
        if n_docs <= 4:
            return np.stack([self._accumulate_and_normalize(self._extract_token_weights(t, use_recency)) for t in texts], axis=0).astype(np.float32)

        dim = self.dimension
        tok_cache_on = self.token_cache_enabled
        cache = self._tok_hash_cache if tok_cache_on else {}
        max_cache = self.max_token_cache
        cache_get = cache.get
        sha256 = _SHA256
        unpack_6i = _UNPACK_6I

        avg_len = sum(len(t) for t in texts) / n_docs
        chunk_size = 256 if avg_len <= 512 else (64 if avg_len <= 2048 else 32)
        out = np.empty((n_docs, dim), dtype=np.float32)

        for start in range(0, n_docs, chunk_size):
            end = min(start + chunk_size, n_docs)
            c_len = end - start
            unique_tokens: Dict[str, int] = {}
            doc_maps: List[Dict[str, float]] = []
            empty_rows: List[int] = []

            for local_i, global_i in enumerate(range(start, end)):
                wmap = self._extract_token_weights(texts[global_i], use_recency)
                if not wmap:
                    doc_maps.append({})
                    empty_rows.append(local_i)
                    continue
                for tok, w in wmap.items():
                    if w != 0.0 and tok not in unique_tokens:
                        unique_tokens[tok] = len(unique_tokens)
                doc_maps.append(wmap)

            n_uniq = len(unique_tokens)
            if n_uniq == 0:
                chunk = np.zeros((c_len, dim), dtype=np.float32)
                chunk[:, 0] = 1.0
                out[start:end] = chunk
                continue

            i_mat = np.empty((n_uniq, 6), dtype=np.int32)
            s_mat = np.empty((n_uniq, 6), dtype=np.float64)
            for tok, u_idx in unique_tokens.items():
                res = cache_get(tok)
                if res is None:
                    digest = sha256(tok.encode("utf-8")).digest()
                    v0, v1, v2, v3, v4, v5 = unpack_6i(digest)
                    res = (
                        v0 % dim,
                        v1 % dim,
                        v2 % dim,
                        v3 % dim,
                        v4 % dim,
                        v5 % dim,
                        1.0 if ((v0 >> 16) & 1) == 0 else -1.0,
                        1.0 if ((v1 >> 16) & 1) == 0 else -1.0,
                        1.0 if ((v2 >> 16) & 1) == 0 else -1.0,
                        1.0 if ((v3 >> 16) & 1) == 0 else -1.0,
                        1.0 if ((v4 >> 16) & 1) == 0 else -1.0,
                        1.0 if ((v5 >> 16) & 1) == 0 else -1.0,
                    )
                    if tok_cache_on and len(cache) >= max_cache:
                        cache.clear()
                    cache[tok] = res
                i_mat[u_idx, 0] = res[0]
                i_mat[u_idx, 1] = res[1]
                i_mat[u_idx, 2] = res[2]
                i_mat[u_idx, 3] = res[3]
                i_mat[u_idx, 4] = res[4]
                i_mat[u_idx, 5] = res[5]
                s_mat[u_idx, 0] = res[6]
                s_mat[u_idx, 1] = res[7]
                s_mat[u_idx, 2] = res[8]
                s_mat[u_idx, 3] = res[9]
                s_mat[u_idx, 4] = res[10]
                s_mat[u_idx, 5] = res[11]

            proj_mat = np.zeros((n_uniq, dim), dtype=np.float64)
            u_arange = np.arange(n_uniq, dtype=np.int32)
            for col in range(6):
                np.add.at(proj_mat, (u_arange, i_mat[:, col]), s_mat[:, col])

            w_mat = np.zeros((c_len, n_uniq), dtype=np.float64)
            for local_i, wmap in enumerate(doc_maps):
                if wmap:
                    cols = [unique_tokens[tok] for tok, w in wmap.items() if w != 0.0]
                    vals = [w for w in wmap.values() if w != 0.0]
                    if cols:
                        w_mat[local_i, cols] = np.asarray(vals, dtype=np.float64)

            vecs = (w_mat @ proj_mat).astype(np.float32)
            max_abs = np.max(np.abs(vecs), axis=1, keepdims=True)
            valid = (max_abs[:, 0] >= 1e-12) & np.isfinite(max_abs[:, 0])
            if empty_rows:
                valid[empty_rows] = False

            if np.any(valid):
                vecs[valid] /= max_abs[valid]
                norms = np.linalg.norm(vecs[valid], axis=1, keepdims=True)
                good_norm = (norms[:, 0] > 1e-12) & np.isfinite(norms[:, 0])
                valid_indices = np.flatnonzero(valid)
                if np.any(good_norm):
                    good_rows = valid_indices[good_norm]
                    vecs[good_rows] /= norms[good_norm]
                if np.any(~good_norm):
                    bad_rows = valid_indices[~good_norm]
                    vecs[bad_rows] = 0.0
                    vecs[bad_rows, 0] = 1.0
            if np.any(~valid):
                vecs[~valid] = 0.0
                vecs[~valid, 0] = 1.0
            out[start:end] = vecs

        return out

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


_FLOAT32_EPS = float(np.finfo(np.float32).eps)


def _has_custom_validate_decision(schema: Any) -> bool:
    """Return True if a schema instance or subclass overrides DecisionSchema.validate_decision."""
    if schema is None:
        return False
    schema_cls = schema if isinstance(schema, type) else type(schema)
    if not isinstance(schema, type) and "validate_decision" in getattr(schema, "__dict__", {}):
        return True
    if (
        isinstance(schema_cls, type)
        and issubclass(schema_cls, DecisionSchema)
        and schema_cls.validate_decision is not DecisionSchema.validate_decision
    ):
        return True
    return False


def _pack_f64_scalar(val: Any) -> Optional[bytes]:
    """Serialize a float scalar into exact IEEE-754 bytes for collision-free cache/signature checks."""
    if val is None:
        return None
    f = float(val)
    if f == 0.0:
        f = 0.0
    return struct.pack("<d", f)


def _owned_readonly_array(arr: Any, dtype: Any = None) -> np.ndarray:
    """Return an owned C-contiguous read-only copy without mutating caller array flags."""
    if dtype is None:
        raw = np.asarray(arr)
        target = raw.dtype
    else:
        raw = arr
        target = dtype
    out = np.array(raw, dtype=target, order="C", copy=True)
    out.setflags(write=False)
    return out


def _pack_array_sig(arr: Any) -> Optional[Tuple[Tuple[int, ...], str, bytes]]:
    """Serialize an array's shape, dtype, and contiguous raw bytes for signature comparisons."""
    if arr is None:
        return None
    a = np.asarray(arr)
    return (a.shape, str(a.dtype), np.ascontiguousarray(a).tobytes())


def _exact_choice_probs_row(
    emb_row: np.ndarray,
    w_raw: np.ndarray,
    b_raw: np.ndarray,
    adj: Any,
    temp: float,
) -> Tuple[np.ndarray, bool]:
    """Recompute single-row Choice probabilities matching DecisionFieldHead dot order."""
    with np.errstate(over="ignore", invalid="ignore"):
        z_exact = (emb_row @ w_raw.T + b_raw).astype(np.float64).ravel()
        z_cal_ex = ((z_exact / 0.25) + adj) / temp
        if bool(np.all(np.isfinite(z_exact))) and bool(np.all(np.isfinite(z_cal_ex))):
            m_ex = float(np.max(z_cal_ex))
            ez_ex = np.exp(z_cal_ex - m_ex)
            sum_ex = float(np.sum(ez_ex))
            if math.isfinite(sum_ex) and sum_ex > 0.0:
                p_ex = ez_ex / sum_ex
                if bool(np.all(np.isfinite(p_ex))):
                    return p_ex, True
    return np.full(w_raw.shape[0], np.nan, dtype=np.float64), False


def _exact_bool_probs_row(
    emb_row: np.ndarray,
    w_raw: np.ndarray,
    b_raw: np.ndarray,
    adj0: float,
    adj1: float,
    temp: float,
) -> Tuple[float, float, float, float, float, bool]:
    """Recompute single-row Boolean probabilities matching DecisionFieldHead dot order."""
    with np.errstate(over="ignore", invalid="ignore"):
        zv_ex = float((emb_row @ w_raw.T + b_raw)[0, 0])
        sz_ex = zv_ex / 0.25
        s0_ex = (-0.5 * sz_ex + adj0) / temp
        s1_ex = (0.5 * sz_ex + adj1) / temp
        if math.isfinite(zv_ex) and math.isfinite(s0_ex) and math.isfinite(s1_ex):
            mb_ex = s0_ex if s0_ex >= s1_ex else s1_ex
            e0_ex = math.exp(s0_ex - mb_ex)
            e1_ex = math.exp(s1_ex - mb_ex)
            den_ex = e0_ex + e1_ex
            if math.isfinite(den_ex) and den_ex > 0.0:
                pf_ex = e0_ex / den_ex
                pt_ex = e1_ex / den_ex
                if math.isfinite(pt_ex) and math.isfinite(pf_ex):
                    return zv_ex, s0_ex, s1_ex, pt_ex, pf_ex, True
    return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), False


def _normalize_raw_projection(
    raw: Any,
    dimension: int,
    *,
    zero_unit0: bool = False,
) -> np.ndarray:
    """Validate finiteness before L2-normalizing a raw embedding, preserving NaN rows fail-closed."""
    with np.errstate(over="ignore", invalid="ignore"):
        raw_arr = np.asarray(raw).ravel()
        if raw_arr.shape[0] != dimension:
            raise ValueError(
                f"Projector output dimension mismatch: expected {dimension}, got {raw_arr.shape[0]}"
            )
        if not bool(np.all(np.isfinite(raw_arr))):
            return np.full(dimension, np.nan, dtype=np.float32)
        vec = raw_arr.astype(np.float32, copy=False)
        norm = float(np.linalg.norm(vec)) if bool(np.all(np.isfinite(vec))) else float("inf")
        if not math.isfinite(norm):
            vec64 = raw_arr.astype(np.float64, copy=False)
            norm64 = float(np.linalg.norm(vec64))
            if not math.isfinite(norm64):
                return np.full(dimension, np.nan, dtype=np.float32)
            if norm64 > 1e-12:
                out = (vec64 / norm64).astype(np.float32)
                return out if bool(np.all(np.isfinite(out))) else np.full(dimension, np.nan, dtype=np.float32)
            out = np.zeros(dimension, dtype=np.float32)
            if zero_unit0 and dimension > 0:
                out[0] = 1.0
            return out
        if norm > 1e-12:
            out = (vec / norm).astype(np.float32, copy=False)
            return out if bool(np.all(np.isfinite(out))) else np.full(dimension, np.nan, dtype=np.float32)
        out = np.zeros(dimension, dtype=np.float32)
        if zero_unit0 and dimension > 0:
            out[0] = 1.0
        return out


def _normalize_raw_projection_batch(
    raw_mat: Any,
    n_rows: int,
    dimension: int,
    *,
    zero_unit0: bool = False,
) -> np.ndarray:
    """Validate finiteness row-wise before L2-normalizing a projection batch, leaving bad rows NaN."""
    with np.errstate(over="ignore", invalid="ignore"):
        raw_arr = np.asarray(raw_mat)
        if raw_arr.ndim != 2 or raw_arr.shape != (n_rows, dimension):
            raise ValueError(
                f"Projector batch output shape mismatch: expected ({n_rows}, {dimension}), got {raw_arr.shape}"
            )
        out = np.full((n_rows, dimension), np.nan, dtype=np.float32)
        if n_rows == 0:
            return out
        finite_rows = np.all(np.isfinite(raw_arr), axis=1)
        if not bool(np.any(finite_rows)):
            return out
        good_raw = raw_arr[finite_rows]
        good_f32 = good_raw.astype(np.float32, copy=False)
        norms = np.linalg.norm(good_f32, axis=1)
        finite_norms = np.all(np.isfinite(good_f32), axis=1) & np.isfinite(norms)
        unit_mask = finite_norms & (norms > 1e-12)
        zero_mask = finite_norms & (norms <= 1e-12)
        good_idx = np.flatnonzero(finite_rows)
        if bool(np.any(unit_mask)):
            normed = (good_f32[unit_mask] / norms[unit_mask, None]).astype(np.float32, copy=False)
            ok_u = np.all(np.isfinite(normed), axis=1)
            if bool(np.any(ok_u)):
                out[good_idx[unit_mask][ok_u]] = normed[ok_u]
        if bool(np.any(zero_mask)):
            z_rows = good_idx[zero_mask]
            out[z_rows] = 0.0
            if zero_unit0 and dimension > 0:
                out[z_rows, 0] = 1.0
        overflow_mask = ~finite_norms
        if bool(np.any(overflow_mask)):
            for pos, row_raw in zip(good_idx[overflow_mask], good_raw[overflow_mask]):
                out[pos] = _normalize_raw_projection(row_raw, dimension, zero_unit0=zero_unit0)
        return out


def _validate_packed_engine_support(eng: Any) -> None:
    """Validate that an engine and all heads/predictors are supported by PackedMultiQuestionKernel."""
    schema = getattr(eng, "schema", None)
    fields = getattr(schema, "fields", None)
    if not fields:
        raise ValueError("PackedMultiQuestionKernel requires a non-empty schema")
    model = getattr(eng, "model", None)
    if model is None:
        raise ValueError("PackedMultiQuestionKernel requires an attached model")
    if _has_custom_validate_decision(schema) or _has_custom_validate_decision(
        getattr(model, "schema", None)
    ):
        raise ValueError(
            "PackedMultiQuestionKernel does not support custom DecisionSchema.validate_decision"
        )
    if getattr(eng, "selective_calibrators", None):
        raise ValueError(
            "PackedMultiQuestionKernel does not support engines with active selective_calibrators"
        )
    if getattr(eng, "joint_calibrator", None) is not None:
        raise ValueError(
            "PackedMultiQuestionKernel does not support engines with active joint_calibrator"
        )
    aci_tracker = getattr(eng, "aci_tracker", None)
    if aci_tracker is not None and getattr(aci_tracker, "margin_floor", None) is not None:
        raise ValueError(
            "PackedMultiQuestionKernel does not support engines with active aci_tracker margin_floor"
        )

    heads = getattr(model, "heads", {}) or {}
    field_heads = getattr(model, "_field_heads", {}) or {}
    calibrators = getattr(eng, "calibrators", {}) or {}
    conformal_predictors = getattr(eng, "conformal_predictors", {}) or {}
    reg_conformal_predictors = getattr(eng, "regression_conformal_predictors", {}) or {}

    for fname, f_def in fields.items():
        if not isinstance(f_def, (ChoiceField, BooleanField, ScoreField)):
            raise ValueError(
                f"PackedMultiQuestionKernel does not support field {fname!r} of type {type(f_def).__name__}"
            )
        if fname not in heads:
            raise ValueError(f"PackedMultiQuestionKernel missing model head for {fname!r}")
        h = heads[fname]
        live_h = field_heads.get(fname)
        candidates = (h,) if (live_h is None or live_h is h) else (h, live_h)
        for cand in candidates:
            if getattr(cand, "hierarchical", False) and getattr(cand, "coarse_weights", None) is not None:
                raise ValueError(f"PackedMultiQuestionKernel does not support hierarchical head {fname!r}")
            if getattr(cand, "_exemplar_bank", None) is not None:
                raise ValueError(f"PackedMultiQuestionKernel does not support exemplar bank on {fname!r}")
            fam_h = float(getattr(cand, "familiarity_threshold", 0.0))
            if not math.isfinite(fam_h) or fam_h > 0.0:
                raise ValueError(f"PackedMultiQuestionKernel does not support familiarity_threshold on {fname!r}")
            proto_h = float(getattr(cand, "prototype_threshold", 0.0))
            if not math.isfinite(proto_h) or proto_h > 0.0:
                raise ValueError(f"PackedMultiQuestionKernel does not support prototype_threshold on {fname!r}")

        if fname not in calibrators:
            raise ValueError(f"PackedMultiQuestionKernel missing calibrator for {fname!r}")

        if isinstance(f_def, (ChoiceField, BooleanField)):
            if fname not in conformal_predictors:
                raise ValueError(
                    f"PackedMultiQuestionKernel missing conformal predictor for {fname!r}"
                )
            cp = conformal_predictors[fname]
            if bool(getattr(cp, "mondrian", False)) and bool(
                getattr(cp, "class_calibration_scores", None)
            ):
                raise ValueError(
                    f"PackedMultiQuestionKernel does not support Mondrian predictor on {fname!r}"
                )
            fam_cp = float(getattr(cp, "familiarity_threshold", 0.0))
            if not math.isfinite(fam_cp) or fam_cp > 0.0:
                raise ValueError(
                    f"PackedMultiQuestionKernel does not support familiarity_threshold on {fname!r}"
                )
            proto_cp = float(getattr(cp, "prototype_threshold", 0.0))
            if not math.isfinite(proto_cp) or proto_cp > 0.0:
                raise ValueError(
                    f"PackedMultiQuestionKernel does not support prototype_threshold on {fname!r}"
                )
        else:
            if fname not in reg_conformal_predictors:
                raise ValueError(
                    f"PackedMultiQuestionKernel missing regression conformal predictor for {fname!r}"
                )


def _compute_engine_inference_signature(
    eng: Any,
    storage_mode: str,
    *,
    effective_alpha: Optional[float] = None,
) -> Tuple[Any, ...]:
    """Build a collision-free byte-exact signature of all engine state affecting inference."""
    model = getattr(eng, "model", None)
    eng_schema = getattr(eng, "schema", None)
    model_schema = None if model is None else getattr(model, "schema", None)
    fields = getattr(eng_schema, "fields", None) or {}
    heads = getattr(model, "heads", {}) if model is not None else {}
    field_heads = getattr(model, "_field_heads", {}) if model is not None else {}

    calibrators = getattr(eng, "calibrators", {}) or {}
    cal_sig = tuple(
        (
            name,
            _pack_f64_scalar(getattr(cal, "temperature", 1.0)),
            _pack_array_sig(getattr(cal, "logit_adjustments", None)),
        )
        for name, cal in sorted(calibrators.items())
    )

    conformal_predictors = getattr(eng, "conformal_predictors", {}) or {}
    cp_sig = tuple(
        (
            name,
            bool(getattr(cp, "is_calibrated", False)),
            str(getattr(cp, "score_method", "")),
            _pack_array_sig(np.asarray(getattr(cp, "calibration_scores", ()), dtype=np.float64)),
            _pack_f64_scalar(getattr(cp, "margin_threshold", None)),
            _pack_f64_scalar(getattr(cp, "calibrated_margin_threshold", 0.0)),
            _pack_f64_scalar(getattr(cp, "relative_odds_ratio", None)),
            _pack_f64_scalar(getattr(cp, "confidence_floor_tau0", None)),
            bool(getattr(cp, "mondrian", False)),
            bool(getattr(cp, "class_calibration_scores", None)),
            _pack_f64_scalar(getattr(cp, "familiarity_threshold", 0.0)),
            _pack_f64_scalar(getattr(cp, "prototype_threshold", 0.0)),
        )
        for name, cp in sorted(conformal_predictors.items())
    )

    reg_conformal_predictors = getattr(eng, "regression_conformal_predictors", {}) or {}
    rcp_sig = tuple(
        (
            name,
            bool(getattr(rcp, "is_calibrated", False)),
            _pack_array_sig(np.asarray(getattr(rcp, "residuals", ()), dtype=np.float64)),
            _pack_f64_scalar(getattr(rcp, "min_value", None)),
            _pack_f64_scalar(getattr(rcp, "max_value", None)),
        )
        for name, rcp in sorted(reg_conformal_predictors.items())
    )

    schema_fields_sig = tuple(
        (
            str(fname),
            type(f_def).__name__,
            tuple(getattr(f_def, "options", ())),
            _pack_f64_scalar(getattr(f_def, "threshold", 0.5))
            if hasattr(f_def, "threshold")
            else None,
            _pack_f64_scalar(getattr(f_def, "min_value", 0.0))
            if hasattr(f_def, "min_value")
            else None,
            _pack_f64_scalar(getattr(f_def, "max_value", 1.0))
            if hasattr(f_def, "max_value")
            else None,
            _pack_f64_scalar(getattr(f_def, "default", None))
            if isinstance(f_def, ScoreField) and getattr(f_def, "default", None) is not None
            else None,
            bool(getattr(f_def, "escalate_on_ambiguity", True)),
        )
        for fname, f_def in fields.items()
    )

    def _head_snapshot(h_obj: Any) -> Optional[Tuple[Any, ...]]:
        if h_obj is None:
            return None
        return (
            _pack_array_sig(getattr(h_obj, "weights", None)),
            _pack_array_sig(getattr(h_obj, "biases", None)),
            _pack_f64_scalar(getattr(h_obj, "temperature_scale", 0.25)),
            bool(getattr(h_obj, "hierarchical", False)),
            getattr(h_obj, "coarse_weights", None) is not None,
            getattr(h_obj, "_exemplar_bank", None) is not None,
            _pack_f64_scalar(getattr(h_obj, "familiarity_threshold", 0.0)),
            _pack_f64_scalar(getattr(h_obj, "prototype_threshold", 0.0)),
        )

    heads_sig = tuple(
        (str(fname), _head_snapshot(heads.get(fname)), _head_snapshot(field_heads.get(fname)))
        for fname in fields
    )

    aci_obj = getattr(eng, "aci_tracker", None)
    aci_sig = (
        None
        if aci_obj is None
        else (
            _pack_f64_scalar(getattr(aci_obj, "alpha_t", 0.0)),
            _pack_f64_scalar(getattr(aci_obj, "margin_floor", None)),
        )
    )

    default_rec = bool(getattr(eng, "recency_weighted", False))

    def _ov_snapshot(src_obj: Any) -> Tuple[Any, ...]:
        ov_map = getattr(src_obj, "_learned_prompt_overrides", None)
        if not isinstance(ov_map, Mapping) or not ov_map:
            return ()
        rec_map = getattr(src_obj, "_learned_prompt_override_recency", {})
        return tuple(
            (
                str(pk),
                bool(rec_map.get(pk, default_rec)) if isinstance(rec_map, Mapping) else default_rec,
                tuple(sorted((str(fk), type(fv).__name__, repr(fv)) for fk, fv in f_map.items()))
                if isinstance(f_map, Mapping)
                else (),
            )
            for pk, f_map in sorted(ov_map.items())
        )

    validate_sig = (
        getattr(type(eng_schema), "validate_decision", None) if eng_schema is not None else None,
        getattr(eng_schema, "__dict__", {}).get("validate_decision")
        if eng_schema is not None
        else None,
        None if model_schema is None else getattr(type(model_schema), "validate_decision", None),
        None
        if model_schema is None
        else getattr(model_schema, "__dict__", {}).get("validate_decision"),
    )
    schema_sig = (
        id(eng_schema) if eng_schema is not None else 0,
        eng_schema.schema_digest()
        if eng_schema is not None and hasattr(eng_schema, "schema_digest")
        else None,
        0 if model_schema is None else id(model_schema),
        None
        if model_schema is None or not hasattr(model_schema, "schema_digest")
        else model_schema.schema_digest(),
        schema_fields_sig,
    )
    proj_obj = getattr(model, "projector", getattr(eng, "projector", None))
    return (
        str(storage_mode),
        int(getattr(eng, "model_version", 1)),
        int(getattr(model, "model_version", getattr(eng, "model_version", 1)))
        if model is not None
        else 1,
        int(getattr(eng, "policy_epoch", 0)),
        int(getattr(eng, "dimension", 0)),
        (
            id(proj_obj),
            "" if proj_obj is None else type(proj_obj).__name__,
            int(getattr(proj_obj, "dimension", 0)),
            bool(getattr(proj_obj, "recency_weighted", False)),
        ),
        "" if model is None else type(model).__name__,
        _pack_f64_scalar(getattr(eng, "default_alpha", 0.05)),
        _pack_f64_scalar(effective_alpha),
        bool(getattr(eng, "strict_mode", True)),
        bool(getattr(eng, "enable_margin_gating", False)),
        _pack_f64_scalar(getattr(eng, "margin_threshold", None)),
        _pack_f64_scalar(getattr(eng, "relative_odds_ratio", None)),
        _pack_f64_scalar(getattr(eng, "confidence_floor_tau0", None)),
        default_rec,
        bool(getattr(eng, "selective_calibrators", None)),
        getattr(eng, "joint_calibrator", None) is not None,
        validate_sig,
        aci_sig,
        (_ov_snapshot(eng), _ov_snapshot(model)),
        schema_sig,
        heads_sig,
        cal_sig,
        cp_sig,
        rcp_sig,
    )


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
    raw_norm: Optional[float] = None


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
        self._p_tmp_buf: Optional[np.ndarray] = None
        self._b_tmp_buf: Optional[np.ndarray] = None
        self._known_stems_per_option: List[Set[str]] = []
        if initialize_weights:
            self._init_head()
        else:
            width = len(field_def.options) if isinstance(field_def, (ChoiceField, MultiChoiceField)) else 1
            self.weights = np.zeros((width, dimension), dtype=np.float32)
            self.biases = np.zeros(width, dtype=np.float32)
        self._precompute_known_stems()

    def _precompute_known_stems(self) -> None:
        """Cache stemmed option descriptions once per head for fast online exemplar updates."""
        stems_list: List[Set[str]] = []
        if isinstance(self.field_def, ChoiceField):
            for opt_name in self.field_def.options:
                desc_k = self.field_def.descriptions.get(opt_name, "")
                txts_k = desc_k if isinstance(desc_k, (list, tuple)) else [str(desc_k), opt_name.replace("_", " ")]
                stems_list.append(
                    {
                        _stem_word(w)
                        for t in txts_k
                        for w in _WORD_RE.findall(str(t).lower())
                    }
                )
        self._known_stems_per_option = stems_list

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
        with np.errstate(over="ignore", invalid="ignore"):
            w_f32 = np.asarray(weights, dtype=np.float32)
            b_f32 = (
                np.asarray(biases, dtype=np.float32)
                if biases is not None
                else np.zeros(w_f32.shape[0], dtype=np.float32)
            )
        if not bool(np.all(np.isfinite(w_f32))) or not bool(np.all(np.isfinite(b_f32))):
            raise ValueError("Head weights and biases must be finite in float32")
        if clear_exemplar_bank:
            self._exemplar_bank = None
            self._exemplar_mean = None
            self._online_shots_p = None
            self._online_shots_c = None
        self.weights = w_f32
        self.biases = b_f32
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
        with np.errstate(over="ignore", invalid="ignore"):
            cw = np.asarray(coarse_weights, dtype=np.float32)
            cb = np.asarray(coarse_biases, dtype=np.float32).flatten()
            fw = np.asarray(fine_weights, dtype=np.float32) if fine_weights is not None else None
            fb = np.asarray(fine_biases, dtype=np.float32).flatten() if fine_biases is not None else None
        if (
            not bool(np.all(np.isfinite(cw)))
            or not bool(np.all(np.isfinite(cb)))
            or (fw is not None and not bool(np.all(np.isfinite(fw))))
            or (fb is not None and not bool(np.all(np.isfinite(fb))))
        ):
            raise ValueError("Hierarchical weights and biases must be finite in float32")
        assign = tuple(int(c) for c in cluster_assignments)
        if cw.ndim != 2 or cb.shape[0] != cw.shape[0]:
            raise ValueError("coarse_weights and coarse_biases shape mismatch")
        if any(c < 0 or c >= cw.shape[0] for c in assign):
            raise ValueError("cluster_assignments contains out-of-bounds cluster index")
        self.hierarchical = True
        self.cluster_assignments = assign
        self.coarse_weights = cw
        self.coarse_biases = cb
        self.fine_weights = fw
        self.fine_biases = fb

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
                    np.asarray(group, dtype=np.float32).reshape(-1, self.dimension) for group in groups
                ], axis=0)
        if self._exemplar_mean is not None:
            arrays["exemplar_mean"] = self._exemplar_mean.copy()
        if self._initial_weights is not None:
            arrays["initial_w"] = self._initial_weights.copy()
        return {"arrays": arrays, "counts": counts,
                "online_shots_are_pseudo": bool(getattr(self, "_online_shots_are_pseudo", False))}

    def restore_exemplar_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("Saved exemplar state must be a mapping")
        arrays = state.get("arrays")
        counts = state.get("counts")
        if not isinstance(arrays, Mapping) or not isinstance(counts, Mapping):
            raise ValueError("Saved exemplar state requires 'arrays' and 'counts' mappings")
        allowed_arrays = {
            "exemplar_centroids",
            "exemplar_vectors",
            "shots_p",
            "shots_c",
            "exemplar_mean",
            "initial_w",
        }
        if set(arrays.keys()) - allowed_arrays or set(counts.keys()) - {"exemplar_vectors", "shots_p", "shots_c"}:
            raise ValueError("Saved exemplar state contains unsupported keys")
        if ("shots_p" in arrays) != ("shots_c" in arrays) or ("shots_p" in counts) != ("shots_c" in counts):
            raise ValueError("Saved exemplar state requires both shots_p and shots_c when online shots are present")
        online_shots_are_pseudo = state.get("online_shots_are_pseudo", False)
        if not isinstance(online_shots_are_pseudo, bool):
            raise ValueError("Saved exemplar state 'online_shots_are_pseudo' must be a boolean")
        width = len(self.field_def.options) if isinstance(self.field_def, ChoiceField) else 0
        if not width or self.hierarchical or "exemplar_centroids" not in arrays or "exemplar_vectors" not in arrays:
            raise ValueError("Saved exemplar state requires a non-hierarchical choice head")
        for key, arr in arrays.items():
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"Saved exemplar array {key!r} must be a numpy ndarray")
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
        self._online_shots_are_pseudo = online_shots_are_pseudo
        self._exemplar_projector = DeterministicSemanticProjector(dimension=self.dimension)

    def _update_exemplar_bank_entry(
        self,
        emb_feat: np.ndarray,
        target_k: int,
        prompt: Optional[str] = None,
        *,
        blend_rate: float = 0.28,
        attenuate_competitors: bool = True,
        concept_embedding: Optional[np.ndarray] = None,
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
        if (concept_embedding is not None or prompt is not None) and self._exemplar_projector is not None:
            if concept_embedding is not None:
                c_vec = concept_embedding
            else:
                assert prompt is not None
                c_vec = self._exemplar_projector._project_concept_aware(prompt)
            if self._exemplar_mean is not None and len(self._exemplar_mean) == D:
                c_vec = (c_vec - self._exemplar_mean).astype(np.float32)
            c_norm = float(np.linalg.norm(c_vec))
            if c_norm <= 1e-12:
                return
            c_full = (c_vec / c_norm).astype(np.float32)
            if prompt is not None:
                # Boost novel co-occurring vocabulary not already present in the initial description of target_k
                if target_k < len(self._known_stems_per_option):
                    known_stems_k = self._known_stems_per_option[target_k]
                else:
                    opt_name = self.field_def.options[target_k]
                    desc_k = self.field_def.descriptions.get(opt_name, "")
                    txts_k = desc_k if isinstance(desc_k, (list, tuple)) else [str(desc_k), opt_name.replace("_", " ")]
                    known_stems_k = {
                        _stem_word(w)
                        for t in txts_k
                        for w in _WORD_RE.findall(str(t).lower())
                    }
                words = _WORD_RE.findall(prompt.lower())
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
        reward_scale: float = 1.0,
        concept_embedding: Optional[np.ndarray] = None,
    ) -> float:
        r"""Closed-form rank-1 Sherman-Morrison update with exponential forgetting factor.

        P_{t+1} = (1 / \lambda_f) [ P_t - (P_t x x^T P_t) / (\lambda_f + x^T P_t x) ]
        B_{t+1} = \lambda_f B_t + x y^T
        W_{t+1} = (P_{t+1} B_{t+1})^T
        """
        t0 = time.perf_counter()
        lam_f = float(
            forgetting_factor if forgetting_factor is not None else self.forgetting_factor
        )
        if not math.isfinite(lam_f) or not (0.0 < lam_f <= 1.0):
            raise ValueError(f"Forgetting factor lambda_f must be in (0, 1.0], got {lam_f}")

        D = self.dimension
        K = self.weights.shape[0]

        x = np.asarray(x_aug, dtype=np.float32).ravel()
        if len(x) == D:
            # If prompt embedding without bias 1, augment
            x = np.append(x, np.float32(1.0))
        elif len(x) != D + 1:
            raise ValueError(f"Expected x_aug of length {D} or {D+1}, got {len(x)}")

        if not np.all(np.isfinite(x)):
            raise ValueError("x_aug contains NaN or Inf non-finite values")

        y = np.asarray(y_target, dtype=np.float32).ravel()
        if not np.all(np.isfinite(y)):
            raise ValueError("y_target contains NaN or Inf non-finite values")
        if len(y) != K:
            raise ValueError(f"Expected target vector y of length {K}, got {len(y)}")

        if self.covariance_inv is None or self.cross_covariance is None:
            self.init_covariance(regularization=self.regularization)

        P = self.covariance_inv
        B = self.cross_covariance
        assert P is not None and B is not None
        if not P.flags.writeable:
            P = P.copy()
            self.covariance_inv = P
        if not B.flags.writeable:
            B = B.copy()
            self.cross_covariance = B

        eff_reward_scale = float(min(1.0, max(0.0, reward_scale)))
        x_step = x
        y_step = y
        if not preserve_calibration and isinstance(self.field_def, ChoiceField) and K > 8:
            pos_max = float(np.max(y))
            if pos_max > 0.0:
                neg_balanced = -pos_max / float(max(1, K - 1))
                y_step = np.where(y > 0.0, y, neg_balanced).astype(np.float32)

        v: Optional[np.ndarray] = None
        z_curr_dir: Optional[np.ndarray] = None
        if preserve_calibration and float(np.linalg.norm(x[:D])) > 1e-12:
            x_step = np.empty(D + 1, dtype=np.float32)
            x_step[:D] = x[:D]
            x_step[D] = 0.0
            v_dir = P @ x_step
            g = float(x_step @ v_dir)
            if g > 1e-12:
                v = v_dir
                z0 = (self.weights @ x[:D]).astype(np.float64) + self.biases.astype(np.float64)
                b0 = self.biases.astype(np.float64)
                z_desired = z0.copy()
                if isinstance(self.field_def, ChoiceField) and K >= 2:
                    t_idx = int(np.argmax(y))
                    for i in range(K):
                        if i != t_idx and z0[i] > 0.15:
                            z_desired[i] = 0.05
                    comp_after = float(max(z_desired[i] for i in range(K) if i != t_idx))
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
                if eff_reward_scale < 1.0:
                    z_desired = z0 + eff_reward_scale * (z_desired - z0)
                y_des = z_desired - b0
                z_curr_dir = z0 - b0
                y_step = (((lam_f + g) * y_des - lam_f * z_curr_dir) / g).astype(np.float32)

        if v is None:
            v = P @ x_step
        denom = float(lam_f + float(x_step @ v))
        if denom <= 1e-12 or not math.isfinite(denom):
            # Degenerate input or numerical singularity: preserve internal state
            return (time.perf_counter() - t0) * 1000.0

        # In-place rank-1 P update using preallocated scratch buffer
        if self._p_tmp_buf is None or self._p_tmp_buf.shape != P.shape or self._p_tmp_buf.dtype != P.dtype:
            self._p_tmp_buf = np.empty_like(P)
        buf = self._p_tmp_buf
        np.multiply.outer(v, v, out=buf)
        buf *= np.float32(1.0 / denom)
        P -= buf
        if lam_f != 1.0:
            P *= np.float32(1.0 / lam_f)

        # In-place cross-covariance update: B_{t+1} = lam_f * B_t + x y^T
        if lam_f != 1.0:
            B *= np.float32(lam_f)
        if self._b_tmp_buf is None or self._b_tmp_buf.shape != B.shape or self._b_tmp_buf.dtype != B.dtype:
            self._b_tmp_buf = np.empty_like(B)
        np.multiply.outer(x_step, y_step, out=self._b_tmp_buf)
        B += self._b_tmp_buf

        # Covariance bounding via diagonal view: prevent covariance windup along unexcited directions
        p_diag = P.diagonal()
        p_max = float(50.0 / max(1e-4, self.regularization))
        max_d = float(np.max(p_diag))
        clamped = False
        if max_d > p_max and max_d > 0.0:
            scale = np.float32(p_max / max_d)
            P *= scale
            B *= np.float32(1.0 / scale)
            clamped = True

        # Enforce strict positive-definiteness on diagonal against floating-point drift
        if float(np.min(p_diag)) < 1e-6:
            idx = np.arange(P.shape[0])
            small = p_diag < 1e-6
            P[idx[small], idx[small]] = np.float32(1e-6)
            clamped = True

        # Exact O(D*K) Sherman-Morrison weight update when un-clamped
        if not clamped:
            if z_curr_dir is not None and x_step[D] == 0.0:
                z_curr = z_curr_dir.astype(np.float32)
            else:
                z_curr = self.weights @ x_step[:D]
                if x_step[D] != 0.0:
                    z_curr = z_curr + self.biases * x_step[D]
            gain = (v * np.float32(1.0 / denom)).astype(np.float32)
            err = (y_step - z_curr).astype(np.float32)
            new_W = self.weights + np.multiply.outer(err, gain[:D])
            if gain[D] != 0.0:
                new_b = self.biases + err * gain[D]
            else:
                new_b = self.biases
        else:
            P[:] = np.float32(0.5) * (P + P.T)
            W_aug = (P @ B).T
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
                            blend_rate=(0.35 if preserve_calibration else 0.28) * eff_reward_scale,
                            attenuate_competitors=(eff_reward_scale >= 0.5),
                            concept_embedding=concept_embedding,
                        )
                    elif eff_reward_scale >= 0.5 and isinstance(self.projector, DeterministicSemanticProjector):
                        if self._online_shots_p is None or getattr(self, "_online_shots_are_pseudo", False):
                            self._online_shots_p = [[] for _ in range(K)]
                            self._online_shots_c = [[] for _ in range(K)]
                            self._online_shots_are_pseudo = False
                        xp = x[:D].copy()
                        np_norm = float(np.linalg.norm(xp))
                        if np_norm > 1e-12:
                            self._online_shots_p[target_k].append((xp / np_norm).astype(np.float32))
                        if (concept_embedding is not None or prompt is not None) and self._exemplar_projector is not None:
                            xc = (
                                concept_embedding
                                if concept_embedding is not None
                                else self._exemplar_projector._project_concept_aware(prompt or "")
                            )
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
        *,
        concept_embedding: Optional[np.ndarray] = None,
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
                if concept_embedding is not None:
                    c_emb = np.asarray(concept_embedding, dtype=np.float32)
                    if c_emb.ndim == 1:
                        c_emb = c_emb.reshape(1, -1)
                elif prompts is not None and self._exemplar_projector is not None:
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

    def forward(
        self,
        embedding: np.ndarray,
        prompt: Optional[str] = None,
        *,
        concept_embedding: Optional[np.ndarray] = None,
    ) -> RawFieldEvaluation:
        """Projects dense embedding into logits and raw probabilities."""
        logits = self.compute_logits(embedding, prompts=prompt, concept_embedding=concept_embedding)
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
        emb = _normalize_raw_projection(emb, self.dimension, zero_unit0=False)
        if telemetry is not None:
            emb = self.telemetry_projector.fuse(emb, telemetry)
        return emb, raw_norm

    def encode(
        self,
        prompt: str,
        telemetry: Optional[Any] = None,
        recency_weighted: Optional[bool] = None,
    ) -> np.ndarray:
        """Encodes prompt into a normalized dense embedding vector, fusing continuous telemetry if present."""
        return self._encode_with_norm(prompt, telemetry=telemetry, recency_weighted=recency_weighted)[0]

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
        return np.stack([self.encode(p, recency_weighted=use_recency) for p in prompts], axis=0).astype(np.float32)

    def forward_single(
        self,
        prompt: str,
        telemetry: Optional[Any] = None,
        embedding: Optional[np.ndarray] = None,
        recency_weighted: Optional[bool] = None,
        *,
        concept_embedding: Optional[np.ndarray] = None,
    ) -> ModelInferenceResult:
        """Executes a single non-autoregressive forward pass over the schema."""
        start_t = time.perf_counter()
        use_recency = self.recency_weighted if recency_weighted is None else bool(recency_weighted)
        raw_norm: Optional[float] = None
        c_emb: Optional[np.ndarray] = concept_embedding
        if embedding is not None:
            emb = np.asarray(embedding, dtype=np.float32).flatten()
        elif (
            telemetry is None
            and type(self.projector) is DeterministicSemanticProjector
            and any(h._exemplar_bank is not None and not h.hierarchical for h in self.heads.values())
        ):
            if not isinstance(prompt, str):
                raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}")
            emb_raw, c_emb = self.projector.project_both(
                prompt,
                recency_weighted=use_recency,
                need_base=True,
                need_concept=True,
            )
            assert emb_raw is not None
            emb = _normalize_raw_projection(emb_raw, self.dimension, zero_unit0=False)
        else:
            emb, raw_norm = self._encode_with_norm(prompt, telemetry=telemetry, recency_weighted=use_recency)

        field_results: Dict[str, RawFieldEvaluation] = {}
        prompt_for_head = prompt if embedding is None else None
        for name, head in self.heads.items():
            field_results[name] = head.forward(emb, prompt=prompt_for_head, concept_embedding=c_emb)

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return ModelInferenceResult(
            prompt=prompt,
            embedding=emb,
            fields=field_results,
            inference_latency_ms=elapsed_ms,
            raw_norm=raw_norm,
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
        *,
        reward_scale: float = 1.0,
        concept_embedding: Optional[np.ndarray] = None,
        allow_prompt_for_head: bool = False,
    ) -> Dict[str, Any]:
        """Closed-form rank-1 Sherman-Morrison online update on the metal (<0.1ms)."""
        with self._lock:
            target_dict = self.schema.validate_lesson(target, allow_unknown=True)
            eff_ff = float(forgetting_factor if forgetting_factor is not None else self.forgetting_factor)
            if not math.isfinite(eff_ff) or not (0.0 < eff_ff <= 1.0):
                raise ValueError(f"Forgetting factor lambda_f must be in (0, 1.0], got {eff_ff}")
            t0 = time.perf_counter()
            c_emb = concept_embedding
            needs_concept = any(
                f_name in self.heads
                and self.heads[f_name]._exemplar_bank is not None
                and not self.heads[f_name].hierarchical
                for f_name in target_dict
            )
            if embedding is not None:
                emb = np.asarray(embedding, dtype=np.float32).flatten()
            elif (
                needs_concept
                and telemetry is None
                and type(self.projector) is DeterministicSemanticProjector
                and isinstance(prompt, str)
                and prompt.strip()
            ):
                use_recency = self.recency_weighted if recency_weighted is None else bool(recency_weighted)
                emb_raw, c_emb = self.projector.project_both(
                    prompt,
                    recency_weighted=use_recency,
                    need_base=True,
                    need_concept=True,
                )
                assert emb_raw is not None
                norm = np.linalg.norm(emb_raw)
                emb = (emb_raw / norm).astype(np.float32) if norm > 0 else emb_raw
            else:
                emb = self.encode(prompt, telemetry=telemetry, recency_weighted=recency_weighted)
            if len(emb) != self.dimension:
                raise ValueError(f"Expected embedding of length {self.dimension}, got {len(emb)}")
            if not np.all(np.isfinite(emb)):
                raise ValueError("embedding contains NaN or Inf non-finite values")
            x_aug = np.append(emb, 1.0).astype(np.float32)

            updated_fields: List[str] = []
            update_durations_ms: List[float] = []
            prompt_for_head = (
                prompt
                if ((embedding is None or allow_prompt_for_head) and isinstance(prompt, str) and prompt.strip())
                else None
            )

            for f_name, target_val in target_dict.items():
                if f_name in self.heads:
                    head = self.heads[f_name]
                    y_target = head.format_target_vector(target_val)
                    dt = head.online_update(
                        x_aug,
                        y_target,
                        forgetting_factor=eff_ff,
                        prompt=prompt_for_head,
                        preserve_calibration=preserve_calibration,
                        reward_scale=reward_scale,
                        concept_embedding=c_emb,
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
        if forgetting_factor is not None:
            ff_check = float(forgetting_factor)
            if not math.isfinite(ff_check) or not (0.0 < ff_check <= 1.0):
                raise ValueError(f"Forgetting factor lambda_f must be in (0, 1.0], got {ff_check}")
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
        if resolved_target is None:
            raise ValueError("learn() requires a target label/mapping or outcome")
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


def _fast_deterministic_project(
    proj: DeterministicSemanticProjector,
    text: str,
    recency_weighted: Optional[bool] = None,
) -> np.ndarray:
    """Bit-equivalent single-pass projection for DeterministicSemanticProjector."""
    if not isinstance(text, str):
        raise TypeError(f"Expected text to be a string, got {type(text).__name__}")
    dim = proj.dimension
    clean_text = text.lower().strip()
    if not clean_text:
        out = np.zeros(dim, dtype=np.float32)
        out[0] = 1.0
        return out
    if len(clean_text) > 8192:
        clean_text = clean_text[:4096] + " " + clean_text[-4096:]

    words = _WORD_RE.findall(clean_text)
    n_words = len(words)
    use_recency = proj.recency_weighted if recency_weighted is None else bool(recency_weighted)
    cache = proj._tok_hash_cache
    tok_cache_on = proj.token_cache_enabled
    max_cache = proj.max_token_cache
    cache_get = cache.get if tok_cache_on else None
    sha256 = _SHA256
    unpack_6i = _UNPACK_6I
    sqrt = math.sqrt
    log1p = math.log1p

    def _populate_tok(
        key: str,
    ) -> Tuple[int, int, int, int, int, int, float, float, float, float, float, float]:
        v0, v1, v2, v3, v4, v5 = unpack_6i(sha256(key.encode("utf-8")).digest())
        entry = (
            v0 % dim,
            v1 % dim,
            v2 % dim,
            v3 % dim,
            v4 % dim,
            v5 % dim,
            1.0 if ((v0 >> 16) & 1) == 0 else -1.0,
            1.0 if ((v1 >> 16) & 1) == 0 else -1.0,
            1.0 if ((v2 >> 16) & 1) == 0 else -1.0,
            1.0 if ((v3 >> 16) & 1) == 0 else -1.0,
            1.0 if ((v4 >> 16) & 1) == 0 else -1.0,
            1.0 if ((v5 >> 16) & 1) == 0 else -1.0,
        )
        if tok_cache_on:
            if len(cache) >= max_cache:
                cache.clear()
            cache[key] = entry
        return entry

    vec_list = [0.0] * dim

    for pos, word in enumerate(words):
        dist = (n_words - 1 - pos) if use_recency else pos
        ww = (log1p(len(word)) / sqrt(1.0 + dist * 0.05)) * 2.0
        wk = "word:" + word
        res = cache_get(wk) if cache_get is not None else None
        if res is None:
            res = _populate_tok(wk)
        i0, i1, i2, i3, i4, i5, s0, s1, s2, s3, s4, s5 = res
        vec_list[i0] += s0 * ww
        vec_list[i1] += s1 * ww
        vec_list[i2] += s2 * ww
        vec_list[i3] += s3 * ww
        vec_list[i4] += s4 * ww
        vec_list[i5] += s5 * ww

        wlen = len(word)
        if wlen >= 3:
            for j in range(wlen - 2):
                nk = "ng:" + word[j : j + 3]
                res = cache_get(nk) if cache_get is not None else None
                if res is None:
                    res = _populate_tok(nk)
                i0, i1, i2, i3, i4, i5, s0, s1, s2, s3, s4, s5 = res
                vec_list[i0] += s0 * 0.5
                vec_list[i1] += s1 * 0.5
                vec_list[i2] += s2 * 0.5
                vec_list[i3] += s3 * 0.5
                vec_list[i4] += s4 * 0.5
                vec_list[i5] += s5 * 0.5
            if wlen >= 4:
                for j in range(wlen - 3):
                    nk = "ng:" + word[j : j + 4]
                    res = cache_get(nk) if cache_get is not None else None
                    if res is None:
                        res = _populate_tok(nk)
                    i0, i1, i2, i3, i4, i5, s0, s1, s2, s3, s4, s5 = res
                    vec_list[i0] += s0 * 0.5
                    vec_list[i1] += s1 * 0.5
                    vec_list[i2] += s2 * 0.5
                    vec_list[i3] += s3 * 0.5
                    vec_list[i4] += s4 * 0.5
                    vec_list[i5] += s5 * 0.5

    for i in range(n_words - 1):
        bk = "bi:" + words[i] + "_" + words[i + 1]
        res = cache_get(bk) if cache_get is not None else None
        if res is None:
            res = _populate_tok(bk)
        i0, i1, i2, i3, i4, i5, s0, s1, s2, s3, s4, s5 = res
        vec_list[i0] += s0 * 1.5
        vec_list[i1] += s1 * 1.5
        vec_list[i2] += s2 * 1.5
        vec_list[i3] += s3 * 1.5
        vec_list[i4] += s4 * 1.5
        vec_list[i5] += s5 * 1.5

    btext = " " + clean_text + " "
    blimit = max(0, len(btext) - 4)
    for j in range(blimit):
        ck = "char4:" + btext[j : j + 4]
        res = cache_get(ck) if cache_get is not None else None
        if res is None:
            res = _populate_tok(ck)
        i0, i1, i2, i3, i4, i5, s0, s1, s2, s3, s4, s5 = res
        vec_list[i0] += s0 * 0.3
        vec_list[i1] += s1 * 0.3
        vec_list[i2] += s2 * 0.3
        vec_list[i3] += s3 * 0.3
        vec_list[i4] += s4 * 0.3
        vec_list[i5] += s5 * 0.3

    vec = np.array(vec_list, dtype=np.float32)
    max_abs = float(np.max(np.abs(vec))) if len(vec) > 0 else 0.0
    if max_abs < 1e-12 or not math.isfinite(max_abs):
        vec.fill(0.0)
        vec[0] = 1.0
        return vec
    vec /= max_abs
    norm = float(np.linalg.norm(vec))
    if norm > 1e-12:
        vec /= norm
    else:
        vec.fill(0.0)
        vec[0] = 1.0
    return vec


_VALID_PACKED_STORAGE_MODES = ("float32", "float16", "float16_upcast")


class PackedMultiQuestionKernel:
    """Contiguous multi-head kernel packing Choice, Boolean, and Score heads into one GEMM/GEMV."""

    def __init__(
        self,
        *,
        projector: Any,
        dimension: int,
        strict_mode: bool,
        alpha: float,
        enable_margin_gating: bool,
        margin_threshold: Optional[float],
        recency_weighted: bool,
        W_packed: np.ndarray,
        b_packed: np.ndarray,
        choice_names: Tuple[str, ...],
        c_options: Tuple[Tuple[str, ...], ...],
        c_K: np.ndarray,
        c_offsets: np.ndarray,
        c_seg_idx: np.ndarray,
        c_qhat: np.ndarray,
        c_is_cal: np.ndarray,
        c_is_lac: np.ndarray,
        c_eff_m: np.ndarray,
        c_gamma: np.ndarray,
        c_tau0: np.ndarray,
        bool_names: Tuple[str, ...],
        b_thresh: np.ndarray,
        b_qhat: np.ndarray,
        b_is_cal: np.ndarray,
        b_is_lac: np.ndarray,
        b_eff_m: np.ndarray,
        b_gamma: np.ndarray,
        b_tau0: np.ndarray,
        score_names: Tuple[str, ...],
        s_min: np.ndarray,
        s_range: np.ndarray,
        s_norm_margin: np.ndarray,
        s_raw_scale: np.ndarray,
        s_is_cal: np.ndarray,
        storage_mode: str = "float32",
        c_escalate: Optional[np.ndarray] = None,
        b_escalate: Optional[np.ndarray] = None,
        s_escalate: Optional[np.ndarray] = None,
        learned_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
        model_learned_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
        c_temp: Optional[np.ndarray] = None,
        c_adj_div_temp: Optional[Tuple[np.ndarray, ...]] = None,
        c_zero_head: Optional[np.ndarray] = None,
        b_temp: Optional[np.ndarray] = None,
        b_adj_div_temp: Optional[np.ndarray] = None,
        b_zero_head: Optional[np.ndarray] = None,
        s_zero_head: Optional[np.ndarray] = None,
        c_adj: Optional[Tuple[np.ndarray, ...]] = None,
        b_adj0: Optional[np.ndarray] = None,
        b_adj1: Optional[np.ndarray] = None,
        c_raw_w: Optional[Tuple[np.ndarray, ...]] = None,
        c_raw_b: Optional[Tuple[np.ndarray, ...]] = None,
        b_raw_w: Optional[Tuple[np.ndarray, ...]] = None,
        b_raw_b: Optional[Tuple[np.ndarray, ...]] = None,
        s_default: Optional[np.ndarray] = None,
        zero_emb_unit0: bool = False,
    ) -> None:
        self.projector = projector
        self.dimension = dimension
        self.strict = strict_mode
        self.alpha = alpha
        self.enable_margin_gating = enable_margin_gating
        self.margin_threshold = margin_threshold
        self.recency_weighted = recency_weighted
        self.storage_mode = storage_mode

        self.W_packed = _owned_readonly_array(W_packed)
        self.b_packed = _owned_readonly_array(b_packed)
        self.M_total = int(self.W_packed.shape[1])

        self.choice_names = choice_names
        self.c_options = c_options
        self.c_K = _owned_readonly_array(c_K, dtype=np.int32)
        self.Q_c = len(choice_names)
        self.M_c = int(np.sum(self.c_K)) if self.Q_c else 0
        self.c_offsets = _owned_readonly_array(c_offsets, dtype=np.int32)
        self.c_seg_idx = _owned_readonly_array(c_seg_idx, dtype=np.int32)
        self.c_qhat = _owned_readonly_array(c_qhat, dtype=np.float64)
        self.c_is_cal = _owned_readonly_array(c_is_cal, dtype=bool)
        self.c_is_lac = _owned_readonly_array(c_is_lac, dtype=bool)
        self.c_eff_m = _owned_readonly_array(c_eff_m, dtype=np.float64)
        self.c_gamma = _owned_readonly_array(c_gamma, dtype=np.float64)
        self.c_tau0 = _owned_readonly_array(c_tau0, dtype=np.float64)
        self.c_all_lac = bool(self.Q_c and np.all(self.c_is_lac) and np.all(self.c_is_cal))
        self.c_all_aps_cal = bool(self.Q_c and np.all(~self.c_is_lac) and np.all(self.c_is_cal))
        self.K_max = int(self.c_K.max()) if self.Q_c else 0
        self.c_pad_rows = _owned_readonly_array(
            np.repeat(np.arange(self.Q_c, dtype=np.int32), self.c_K)
            if self.Q_c
            else np.empty(0, dtype=np.int32),
            dtype=np.int32,
        )
        self.c_pad_cols = _owned_readonly_array(
            np.concatenate([np.arange(int(k), dtype=np.int32) for k in self.c_K])
            if self.Q_c
            else np.empty(0, dtype=np.int32),
            dtype=np.int32,
        )
        self.c_conf_floor = _owned_readonly_array(
            (1.0 / np.maximum(1.0, self.c_K)) + self.c_tau0
            if self.Q_c
            else np.empty(0, dtype=np.float64),
            dtype=np.float64,
        )

        self.bool_names = bool_names
        self.Q_b = len(bool_names)
        self.b_thresh = _owned_readonly_array(b_thresh, dtype=np.float64)
        self.b_qhat = _owned_readonly_array(b_qhat, dtype=np.float64)
        self.b_is_cal = _owned_readonly_array(b_is_cal, dtype=bool)
        self.b_is_lac = _owned_readonly_array(b_is_lac, dtype=bool)
        self.b_eff_m = _owned_readonly_array(b_eff_m, dtype=np.float64)
        self.b_gamma = _owned_readonly_array(b_gamma, dtype=np.float64)
        self.b_tau0 = _owned_readonly_array(b_tau0, dtype=np.float64)
        self.b_all_lac = bool(self.Q_b and np.all(self.b_is_lac) and np.all(self.b_is_cal))
        self.b_all_aps_cal = bool(self.Q_b and np.all(~self.b_is_lac) and np.all(self.b_is_cal))
        self.b_conf_floor = _owned_readonly_array(
            0.5 + self.b_tau0 if self.Q_b else np.empty(0, dtype=np.float64),
            dtype=np.float64,
        )

        self.score_names = score_names
        self.Q_s = len(score_names)
        self.s_min = _owned_readonly_array(s_min, dtype=np.float64)
        self.s_range = _owned_readonly_array(s_range, dtype=np.float64)
        self.s_default = _owned_readonly_array(
            self.s_min if s_default is None else s_default,
            dtype=np.float64,
        )
        self.s_norm_margin = _owned_readonly_array(s_norm_margin, dtype=np.float64)
        self.s_raw_scale = _owned_readonly_array(s_raw_scale, dtype=np.float64)
        self.s_is_cal = _owned_readonly_array(s_is_cal, dtype=bool)
        self.eff_score_thresh = (
            float(margin_threshold)
            if (enable_margin_gating and margin_threshold is not None)
            else 0.5
        )
        self.c_escalate = _owned_readonly_array(
            np.ones(self.Q_c, dtype=bool)
            if c_escalate is None
            else np.asarray(c_escalate, dtype=bool).reshape(-1),
            dtype=bool,
        )
        self.b_escalate = _owned_readonly_array(
            np.ones(self.Q_b, dtype=bool)
            if b_escalate is None
            else np.asarray(b_escalate, dtype=bool).reshape(-1),
            dtype=bool,
        )
        self.s_escalate = _owned_readonly_array(
            np.ones(self.Q_s, dtype=bool)
            if s_escalate is None
            else np.asarray(s_escalate, dtype=bool).reshape(-1),
            dtype=bool,
        )
        self.c_any_escalate = bool(self.Q_c and np.any(self.c_escalate))
        self.b_any_escalate = bool(self.Q_b and np.any(self.b_escalate))
        self.s_any_escalate = bool(self.Q_s and np.any(self.s_escalate))
        self.any_escalating_field = self.c_any_escalate or self.b_any_escalate or self.s_any_escalate
        from system1.core.text import TfidfProjector

        self.is_tfidf_projector = isinstance(projector, TfidfProjector)
        self.learned_overrides: Dict[str, Dict[str, Any]] = (
            {str(k): dict(v) for k, v in learned_overrides.items()}
            if learned_overrides
            else {}
        )
        self.model_learned_overrides: Dict[str, Dict[str, Any]] = (
            {str(k): dict(v) for k, v in model_learned_overrides.items()}
            if model_learned_overrides
            else {}
        )
        self.c_temp = _owned_readonly_array(
            c_temp if c_temp is not None else np.ones(self.Q_c, dtype=np.float64),
            dtype=np.float64,
        )
        if c_adj is not None:
            self.c_adj: Tuple[np.ndarray, ...] = tuple(
                _owned_readonly_array(np.asarray(arr, dtype=np.float64).reshape(-1), dtype=np.float64)
                for arr in c_adj
            )
            self.c_adj_div_temp = tuple(
                _owned_readonly_array(arr / float(t), dtype=np.float64)
                for arr, t in zip(self.c_adj, self.c_temp)
            )
        elif c_adj_div_temp is not None:
            self.c_adj_div_temp = tuple(
                _owned_readonly_array(np.asarray(arr, dtype=np.float64).reshape(-1), dtype=np.float64)
                for arr in c_adj_div_temp
            )
            self.c_adj = tuple(
                _owned_readonly_array(arr * float(t), dtype=np.float64)
                for arr, t in zip(self.c_adj_div_temp, self.c_temp)
            )
        else:
            self.c_adj = tuple(
                _owned_readonly_array(np.zeros(int(k), dtype=np.float64), dtype=np.float64)
                for k in self.c_K
            )
            self.c_adj_div_temp = tuple(
                _owned_readonly_array(np.zeros(int(k), dtype=np.float64), dtype=np.float64)
                for k in self.c_K
            )
        self.c_adj_flat = _owned_readonly_array(
            np.concatenate(self.c_adj) if self.Q_c else np.empty(0, dtype=np.float64),
            dtype=np.float64,
        )
        self.c_temp_flat = _owned_readonly_array(
            self.c_temp[self.c_seg_idx] if self.Q_c else np.empty(0, dtype=np.float64),
            dtype=np.float64,
        )
        self.c_zero_head = _owned_readonly_array(
            c_zero_head if c_zero_head is not None else np.zeros(self.Q_c, dtype=bool),
            dtype=bool,
        )
        self.b_temp = _owned_readonly_array(
            b_temp if b_temp is not None else np.ones(self.Q_b, dtype=np.float64),
            dtype=np.float64,
        )
        if b_adj0 is not None and b_adj1 is not None:
            self.b_adj0 = _owned_readonly_array(
                np.asarray(b_adj0, dtype=np.float64).reshape(-1), dtype=np.float64
            )
            self.b_adj1 = _owned_readonly_array(
                np.asarray(b_adj1, dtype=np.float64).reshape(-1), dtype=np.float64
            )
            self.b_adj_div_temp = _owned_readonly_array(
                (self.b_adj1 - self.b_adj0) / self.b_temp, dtype=np.float64
            )
        elif b_adj_div_temp is not None:
            self.b_adj_div_temp = _owned_readonly_array(
                np.asarray(b_adj_div_temp, dtype=np.float64).reshape(-1), dtype=np.float64
            )
            self.b_adj0 = _owned_readonly_array(
                np.zeros(self.Q_b, dtype=np.float64), dtype=np.float64
            )
            self.b_adj1 = _owned_readonly_array(
                self.b_adj_div_temp * self.b_temp, dtype=np.float64
            )
        else:
            self.b_adj0 = _owned_readonly_array(
                np.zeros(self.Q_b, dtype=np.float64), dtype=np.float64
            )
            self.b_adj1 = _owned_readonly_array(
                np.zeros(self.Q_b, dtype=np.float64), dtype=np.float64
            )
            self.b_adj_div_temp = _owned_readonly_array(
                np.zeros(self.Q_b, dtype=np.float64), dtype=np.float64
            )
        self.b_zero_head = _owned_readonly_array(
            b_zero_head if b_zero_head is not None else np.zeros(self.Q_b, dtype=bool),
            dtype=bool,
        )
        self.s_zero_head = _owned_readonly_array(
            s_zero_head if s_zero_head is not None else np.zeros(self.Q_s, dtype=bool),
            dtype=bool,
        )
        self.Q_total = self.Q_c + self.Q_b + self.Q_s
        self.zero_emb_unit0 = bool(zero_emb_unit0)
        self._inference_sig: Optional[Tuple[Any, ...]] = None

        def _freeze_raw_tuple(
            seq: Optional[Tuple[np.ndarray, ...]],
        ) -> Tuple[np.ndarray, ...]:
            if not seq:
                return ()
            frozen: List[np.ndarray] = []
            for item in seq:
                arr = np.asarray(item)
                dt = arr.dtype if arr.dtype in (np.float32, np.float64) else np.float32
                frozen.append(_owned_readonly_array(arr, dtype=dt))
            return tuple(frozen)

        self.c_raw_w = _freeze_raw_tuple(c_raw_w)
        self.c_raw_b = _freeze_raw_tuple(c_raw_b)
        self.b_raw_w = _freeze_raw_tuple(b_raw_w)
        self.b_raw_b = _freeze_raw_tuple(b_raw_b)
        self.c_w_l1_max = _owned_readonly_array(
            np.array(
                [
                    float(np.max(np.sum(np.abs(w), axis=1))) if w.size else 0.0
                    for w in self.c_raw_w
                ],
                dtype=np.float64,
            )
            if len(self.c_raw_w) == self.Q_c and self.Q_c > 0
            else np.empty(0, dtype=np.float64),
            dtype=np.float64,
        )
        self.c_b_abs_max = _owned_readonly_array(
            np.array(
                [float(np.max(np.abs(b))) if b.size else 0.0 for b in self.c_raw_b],
                dtype=np.float64,
            )
            if len(self.c_raw_b) == self.Q_c and self.Q_c > 0
            else np.empty(0, dtype=np.float64),
            dtype=np.float64,
        )
        self.b_w_l1 = _owned_readonly_array(
            np.array(
                [float(np.sum(np.abs(w))) if w.size else 0.0 for w in self.b_raw_w],
                dtype=np.float64,
            )
            if len(self.b_raw_w) == self.Q_b and self.Q_b > 0
            else np.empty(0, dtype=np.float64),
            dtype=np.float64,
        )
        self.b_b_abs = _owned_readonly_array(
            np.array(
                [float(np.max(np.abs(b))) if b.size else 0.0 for b in self.b_raw_b],
                dtype=np.float64,
            )
            if len(self.b_raw_b) == self.Q_b and self.Q_b > 0
            else np.empty(0, dtype=np.float64),
            dtype=np.float64,
        )

    @classmethod
    def from_engine(
        cls,
        eng: Any,
        *,
        storage_mode: str = "float32",
        alpha: Optional[float] = None,
        strict: Optional[bool] = None,
        margin_threshold: Optional[float] = None,
        relative_odds_ratio: Optional[float] = None,
        confidence_floor_tau0: Optional[float] = None,
        recency_weighted: Optional[bool] = None,
    ) -> "PackedMultiQuestionKernel":
        if storage_mode not in _VALID_PACKED_STORAGE_MODES:
            raise ValueError(
                f"storage_mode must be one of {_VALID_PACKED_STORAGE_MODES}, got {storage_mode!r}"
            )
        eng_lock = getattr(eng, "_lock", None)
        mdl_lock = getattr(getattr(eng, "model", None), "_lock", None)
        with (eng_lock if eng_lock is not None else nullcontext()), (
            mdl_lock if mdl_lock is not None else nullcontext()
        ):
            return cls._from_engine_locked(
                eng,
                storage_mode=storage_mode,
                alpha=alpha,
                strict=strict,
                margin_threshold=margin_threshold,
                relative_odds_ratio=relative_odds_ratio,
                confidence_floor_tau0=confidence_floor_tau0,
                recency_weighted=recency_weighted,
            )

    @classmethod
    def _from_engine_locked(
        cls,
        eng: Any,
        *,
        storage_mode: str = "float32",
        alpha: Optional[float] = None,
        strict: Optional[bool] = None,
        margin_threshold: Optional[float] = None,
        relative_odds_ratio: Optional[float] = None,
        confidence_floor_tau0: Optional[float] = None,
        recency_weighted: Optional[bool] = None,
    ) -> "PackedMultiQuestionKernel":
        _validate_packed_engine_support(eng)
        fields = eng.schema.fields
        model = eng.model
        aci_tracker = getattr(eng, "aci_tracker", None)
        heads = model.heads
        field_heads = getattr(model, "_field_heads", {})
        raw_alpha = (
            (aci_tracker.alpha_t if aci_tracker is not None else eng.default_alpha)
            if alpha is None
            else alpha
        )
        if (
            isinstance(raw_alpha, bool)
            or not isinstance(raw_alpha, (int, float))
            or not math.isfinite(float(raw_alpha))
            or not (0.0 < float(raw_alpha) < 1.0)
        ):
            raise ValueError(f"alpha must be in (0, 1), got {raw_alpha!r}")
        eff_alpha = float(raw_alpha)
        strict_mode = bool(eng.strict_mode if strict is None else strict)
        enable_margin_gating = bool(getattr(eng, "enable_margin_gating", False))
        eng_margin_threshold = eng.margin_threshold if margin_threshold is None else margin_threshold
        eng_rel_odds = getattr(eng, "relative_odds_ratio", None) if relative_odds_ratio is None else relative_odds_ratio
        eng_tau0 = getattr(eng, "confidence_floor_tau0", None) if confidence_floor_tau0 is None else confidence_floor_tau0
        eff_recency = bool(getattr(eng, "recency_weighted", False) if recency_weighted is None else recency_weighted)
        default_ov_rec = bool(getattr(eng, "recency_weighted", False))
        learned_overrides_map: Dict[str, Dict[str, Any]] = {}
        model_overrides_map: Dict[str, Dict[str, Any]] = {}
        has_model_override_eval = hasattr(model, "_build_override_eval")
        for src_obj in (model, eng):
            src_ov = getattr(src_obj, "_learned_prompt_overrides", None)
            src_rec = getattr(src_obj, "_learned_prompt_override_recency", {})
            if isinstance(src_ov, Mapping):
                for p_key, f_map in src_ov.items():
                    if isinstance(p_key, str) and isinstance(f_map, Mapping):
                        rec_flag = src_rec.get(p_key, default_ov_rec) if isinstance(src_rec, Mapping) else default_ov_rec
                        if bool(rec_flag) == eff_recency:
                            learned_overrides_map.setdefault(p_key, {}).update(f_map)
                            if src_obj is model and has_model_override_eval:
                                model_overrides_map.setdefault(p_key, {}).update(f_map)

        def _resolve_gate_params(cp_obj: Any) -> Tuple[float, float, float]:
            if strict_mode or not enable_margin_gating:
                eff_m = 0.0
            elif eng_margin_threshold is not None:
                eff_m = float(eng_margin_threshold)
            elif getattr(cp_obj, "margin_threshold", None) is not None:
                eff_m = float(cp_obj.margin_threshold)
            else:
                eff_m = float(getattr(cp_obj, "calibrated_margin_threshold", 0.0))

            eff_gamma = (
                float(eng_rel_odds)
                if eng_rel_odds is not None
                else float(getattr(cp_obj, "relative_odds_ratio", 1.5))
            )
            eff_tau0 = (
                float(eng_tau0)
                if eng_tau0 is not None
                else float(getattr(cp_obj, "confidence_floor_tau0", 0.15))
            )
            return eff_m, eff_gamma, eff_tau0

        W_cols: List[np.ndarray] = []
        b_cols: List[np.ndarray] = []

        choice_names: List[str] = []
        c_options: List[Tuple[str, ...]] = []
        c_K: List[int] = []
        c_offsets: List[int] = []
        c_seg_idx: List[int] = []
        c_qhat: List[float] = []
        c_is_cal: List[bool] = []
        c_is_lac: List[bool] = []
        c_eff_m: List[float] = []
        c_gamma: List[float] = []
        c_tau0: List[float] = []
        c_escalate: List[bool] = []
        c_temp: List[float] = []
        c_adj: List[np.ndarray] = []
        c_zero_head: List[bool] = []
        c_raw_w: List[np.ndarray] = []
        c_raw_b: List[np.ndarray] = []

        bool_names: List[str] = []
        b_thresh: List[float] = []
        b_qhat: List[float] = []
        b_is_cal: List[bool] = []
        b_is_lac: List[bool] = []
        b_eff_m: List[float] = []
        b_gamma: List[float] = []
        b_tau0: List[float] = []
        b_escalate: List[bool] = []
        b_temp: List[float] = []
        b_adj0: List[float] = []
        b_adj1: List[float] = []
        b_zero_head: List[bool] = []
        b_raw_w: List[np.ndarray] = []
        b_raw_b: List[np.ndarray] = []

        score_names: List[str] = []
        s_min: List[float] = []
        s_range: List[float] = []
        s_default: List[float] = []
        s_norm_margin: List[float] = []
        s_raw_scale: List[float] = []
        s_is_cal: List[bool] = []
        s_escalate: List[bool] = []
        s_zero_head: List[bool] = []

        def _validate_head(fname: str) -> Any:
            h = heads[fname]
            live_h = field_heads.get(fname)
            if getattr(h, "hierarchical", False) and getattr(h, "coarse_weights", None) is not None:
                raise ValueError(f"PackedMultiQuestionKernel does not support hierarchical head {fname!r}")
            if getattr(h, "_exemplar_bank", None) is not None or (
                live_h is not None and getattr(live_h, "_exemplar_bank", None) is not None
            ):
                raise ValueError(f"PackedMultiQuestionKernel does not support exemplar bank on {fname!r}")
            if float(getattr(h, "familiarity_threshold", 0.0)) > 0.0:
                raise ValueError(f"PackedMultiQuestionKernel does not support familiarity_threshold on {fname!r}")
            if float(getattr(h, "prototype_threshold", 0.0)) > 0.0:
                raise ValueError(f"PackedMultiQuestionKernel does not support prototype_threshold on {fname!r}")
            return h

        def _validate_cp(fname: str) -> Any:
            cp = eng.conformal_predictors[fname]
            if getattr(cp, "mondrian", False) and getattr(cp, "class_calibration_scores", None):
                raise ValueError(f"PackedMultiQuestionKernel does not support Mondrian predictor on {fname!r}")
            if float(getattr(cp, "familiarity_threshold", 0.0)) > 0.0:
                raise ValueError(f"PackedMultiQuestionKernel does not support familiarity_threshold on {fname!r}")
            if float(getattr(cp, "prototype_threshold", 0.0)) > 0.0:
                raise ValueError(f"PackedMultiQuestionKernel does not support prototype_threshold on {fname!r}")
            return cp

        for fname, f_def in fields.items():
            if not isinstance(f_def, (ChoiceField, BooleanField, ScoreField)):
                raise ValueError(
                    f"PackedMultiQuestionKernel does not support field {fname!r} of type {type(f_def).__name__}"
                )

        cursor = 0
        for fname, f_def in fields.items():
            if isinstance(f_def, ChoiceField):
                head = _validate_head(fname)
                cp = _validate_cp(fname)
                ci = len(choice_names)
                choice_names.append(fname)
                c_options.append(tuple(f_def.options))
                K = len(f_def.options)
                c_K.append(K)
                cal = eng.calibrators[fname]
                temp = float(cal.temperature)
                if not math.isfinite(temp) or temp <= 0.0:
                    raise ValueError(f"Invalid calibrator temperature for {fname!r}: {temp!r}")
                W = np.asarray(head.weights, dtype=np.float64)
                b = np.asarray(head.biases, dtype=np.float64).reshape(-1)
                raw_w_arr = np.asarray(head.weights)
                raw_b_arr = np.asarray(head.biases).reshape(-1)
                c_raw_w.append(
                    _owned_readonly_array(
                        raw_w_arr,
                        dtype=raw_w_arr.dtype if raw_w_arr.dtype in (np.float32, np.float64) else np.float32,
                    )
                )
                c_raw_b.append(
                    _owned_readonly_array(
                        raw_b_arr,
                        dtype=raw_b_arr.dtype if raw_b_arr.dtype in (np.float32, np.float64) else np.float32,
                    )
                )
                raw_adj = getattr(cal, "logit_adjustments", None)
                if raw_adj is not None:
                    adj_vec = np.asarray(raw_adj, dtype=np.float64).reshape(-1)
                    if adj_vec.shape[0] != K or not np.all(np.isfinite(adj_vec)):
                        raise ValueError(f"Invalid calibrator logit_adjustments for {fname!r}")
                else:
                    adj_vec = np.zeros(K, dtype=np.float64)
                c_temp.append(temp)
                c_adj.append(adj_vec)
                c_zero_head.append(bool(not np.any(head.weights) and not np.any(head.biases)))
                W_cols.append(W.T)
                b_cols.append(b)
                c_offsets.append(cursor)
                c_seg_idx.extend([ci] * K)
                cursor += K

                n_cal = len(cp.calibration_scores)
                cal_ok = bool(cp.is_calibrated and n_cal > 0)
                c_is_cal.append(cal_ok)
                c_is_lac.append(cp.score_method == "lac")
                if cal_ok:
                    k_idx = math.ceil((n_cal + 1) * (1.0 - eff_alpha))
                    c_qhat.append(1.0 if k_idx > n_cal else float(cp.calibration_scores[k_idx - 1]))
                else:
                    c_qhat.append(1.0 - eff_alpha)
                gm, gg, gt = _resolve_gate_params(cp)
                c_eff_m.append(gm)
                c_gamma.append(gg)
                c_tau0.append(gt)
                c_escalate.append(bool(getattr(f_def, "escalate_on_ambiguity", True)))

        for fname, f_def in fields.items():
            if isinstance(f_def, BooleanField):
                head = _validate_head(fname)
                cp = _validate_cp(fname)
                bool_names.append(fname)
                cal = eng.calibrators[fname]
                temp = float(cal.temperature)
                if not math.isfinite(temp) or temp <= 0.0:
                    raise ValueError(f"Invalid calibrator temperature for {fname!r}: {temp!r}")
                W = np.asarray(head.weights, dtype=np.float64)[:1]
                b = np.asarray(head.biases, dtype=np.float64).reshape(-1)[:1]
                raw_w_arr = np.asarray(head.weights)[:1]
                raw_b_arr = np.asarray(head.biases).reshape(-1)[:1]
                b_raw_w.append(
                    _owned_readonly_array(
                        raw_w_arr,
                        dtype=raw_w_arr.dtype if raw_w_arr.dtype in (np.float32, np.float64) else np.float32,
                    )
                )
                b_raw_b.append(
                    _owned_readonly_array(
                        raw_b_arr,
                        dtype=raw_b_arr.dtype if raw_b_arr.dtype in (np.float32, np.float64) else np.float32,
                    )
                )
                adj = getattr(cal, "logit_adjustments", None)
                if adj is not None and len(adj) == 2:
                    adj0_val = float(adj[0])
                    adj1_val = float(adj[1])
                    if not (math.isfinite(adj0_val) and math.isfinite(adj1_val)):
                        raise ValueError(f"Invalid calibrator logit_adjustments for {fname!r}")
                elif adj is not None and len(adj) != 0:
                    adj_arr = np.asarray(adj, dtype=np.float64).reshape(-1)
                    if not np.all(np.isfinite(adj_arr)):
                        raise ValueError(f"Invalid calibrator logit_adjustments for {fname!r}")
                    adj0_val = 0.0
                    adj1_val = 0.0
                else:
                    adj0_val = 0.0
                    adj1_val = 0.0
                b_temp.append(temp)
                b_adj0.append(adj0_val)
                b_adj1.append(adj1_val)
                b_zero_head.append(bool(not np.any(head.weights) and not np.any(head.biases)))
                W_cols.append(W.T)
                b_cols.append(b)
                b_thresh.append(float(f_def.threshold))

                n_cal = len(cp.calibration_scores)
                cal_ok = bool(cp.is_calibrated and n_cal > 0)
                b_is_cal.append(cal_ok)
                b_is_lac.append(cp.score_method == "lac")
                if cal_ok:
                    k_idx = math.ceil((n_cal + 1) * (1.0 - eff_alpha))
                    b_qhat.append(1.0 if k_idx > n_cal else float(cp.calibration_scores[k_idx - 1]))
                else:
                    b_qhat.append(1.0 - eff_alpha)
                gm, gg, gt = _resolve_gate_params(cp)
                b_eff_m.append(gm)
                b_gamma.append(gg)
                b_tau0.append(gt)
                b_escalate.append(bool(getattr(f_def, "escalate_on_ambiguity", True)))

        for fname, f_def in fields.items():
            if isinstance(f_def, ScoreField):
                head = _validate_head(fname)
                score_names.append(fname)
                cal = eng.calibrators[fname]
                raw_t = float(cal.temperature)
                if not math.isfinite(raw_t) or raw_t <= 0.0:
                    raise ValueError(f"Invalid calibrator temperature for {fname!r}: {raw_t!r}")
                temp = max(1e-4, raw_t)
                W = np.asarray(head.weights, dtype=np.float64)[:1]
                b = np.asarray(head.biases, dtype=np.float64).reshape(-1)[:1]
                W_cols.append(W.T)
                b_cols.append(b)
                s_raw_scale.append(temp)
                rng_val = float(f_def.max_value - f_def.min_value)
                s_min.append(float(f_def.min_value))
                s_range.append(rng_val)
                s_default.append(
                    float(f_def.default)
                    if getattr(f_def, "default", None) is not None
                    else float(f_def.min_value)
                )
                rcp = eng.regression_conformal_predictors[fname]
                n_res = len(rcp.residuals)
                cal_ok = bool(rcp.is_calibrated and n_res > 0)
                s_is_cal.append(cal_ok)
                if cal_ok:
                    k_idx = math.ceil((n_res + 1) * (1.0 - eff_alpha))
                    m = rng_val if k_idx > n_res else float(rcp.residuals[k_idx - 1])
                else:
                    m = rng_val if strict_mode else rng_val * ((1.0 - eff_alpha) / 2.0)
                s_norm_margin.append(m / max(1e-6, rng_val))
                s_escalate.append(bool(getattr(f_def, "escalate_on_ambiguity", True)))
                s_zero_head.append(bool(not np.any(head.weights) and not np.any(head.biases)))

        arr_dtype = np.float32 if storage_mode == "float32" else np.float16
        with np.errstate(over="ignore", invalid="ignore"):
            W_packed = np.ascontiguousarray(np.hstack(W_cols), dtype=arr_dtype)
            b_packed = np.ascontiguousarray(np.concatenate(b_cols), dtype=arr_dtype)
        if not (np.all(np.isfinite(W_packed)) and np.all(np.isfinite(b_packed))):
            raise ValueError(
                f"PackedMultiQuestionKernel weights/biases non-finite or overflowed {storage_mode}"
            )

        from system1.core.text import PragmaticCompositeProjector, TfidfProjector

        proj_obj = getattr(model, "projector", eng.projector)
        zero_emb_unit0 = bool(
            type(model).__name__ == "CompiledSystemOneModel"
            and not isinstance(proj_obj, (TfidfProjector, PragmaticCompositeProjector))
        )
        kernel = cls(
            projector=proj_obj,
            dimension=int(eng.dimension),
            strict_mode=strict_mode,
            alpha=eff_alpha,
            enable_margin_gating=enable_margin_gating,
            margin_threshold=eng_margin_threshold,
            recency_weighted=eff_recency,
            W_packed=W_packed,
            b_packed=b_packed,
            choice_names=tuple(choice_names),
            c_options=tuple(c_options),
            c_K=np.asarray(c_K, dtype=np.int32),
            c_offsets=np.asarray(c_offsets, dtype=np.int32),
            c_seg_idx=np.asarray(c_seg_idx, dtype=np.int32),
            c_qhat=np.asarray(c_qhat, dtype=np.float64),
            c_is_cal=np.asarray(c_is_cal, dtype=bool),
            c_is_lac=np.asarray(c_is_lac, dtype=bool),
            c_eff_m=np.asarray(c_eff_m, dtype=np.float64),
            c_gamma=np.asarray(c_gamma, dtype=np.float64),
            c_tau0=np.asarray(c_tau0, dtype=np.float64),
            bool_names=tuple(bool_names),
            b_thresh=np.asarray(b_thresh, dtype=np.float64),
            b_qhat=np.asarray(b_qhat, dtype=np.float64),
            b_is_cal=np.asarray(b_is_cal, dtype=bool),
            b_is_lac=np.asarray(b_is_lac, dtype=bool),
            b_eff_m=np.asarray(b_eff_m, dtype=np.float64),
            b_gamma=np.asarray(b_gamma, dtype=np.float64),
            b_tau0=np.asarray(b_tau0, dtype=np.float64),
            score_names=tuple(score_names),
            s_min=np.asarray(s_min, dtype=np.float64),
            s_range=np.asarray(s_range, dtype=np.float64),
            s_norm_margin=np.asarray(s_norm_margin, dtype=np.float64),
            s_raw_scale=np.asarray(s_raw_scale, dtype=np.float64),
            s_is_cal=np.asarray(s_is_cal, dtype=bool),
            storage_mode=storage_mode,
            c_escalate=np.asarray(c_escalate, dtype=bool),
            b_escalate=np.asarray(b_escalate, dtype=bool),
            s_escalate=np.asarray(s_escalate, dtype=bool),
            learned_overrides=learned_overrides_map,
            model_learned_overrides=model_overrides_map,
            c_temp=np.asarray(c_temp, dtype=np.float64),
            c_zero_head=np.asarray(c_zero_head, dtype=bool),
            b_temp=np.asarray(b_temp, dtype=np.float64),
            b_zero_head=np.asarray(b_zero_head, dtype=bool),
            s_zero_head=np.asarray(s_zero_head, dtype=bool),
            c_adj=tuple(c_adj),
            b_adj0=np.asarray(b_adj0, dtype=np.float64),
            b_adj1=np.asarray(b_adj1, dtype=np.float64),
            c_raw_w=tuple(c_raw_w),
            c_raw_b=tuple(c_raw_b),
            b_raw_w=tuple(b_raw_w),
            b_raw_b=tuple(b_raw_b),
            s_default=np.asarray(s_default, dtype=np.float64),
            zero_emb_unit0=zero_emb_unit0,
        )
        kernel._inference_sig = _compute_engine_inference_signature(
            eng, storage_mode, effective_alpha=eff_alpha
        )
        return kernel

    def forward_embeddings(
        self, X: np.ndarray, *, prompts: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
        """Single contiguous BLAS GEMV (B=1) / GEMM (B>1) + vectorized postprocessing."""
        is_1d = X.ndim == 1
        B_in = 1 if is_1d else int(X.shape[0])
        if B_in == 0:
            return {
                "p_c": np.empty((0, self.M_c), dtype=np.float64),
                "best_idx": np.empty((0, self.Q_c), dtype=np.int32),
                "c_conf": np.empty((0, self.Q_c), dtype=np.float64),
                "in_c": np.empty((0, self.Q_c), dtype=np.int32),
                "p_true": np.empty((0, self.Q_b), dtype=np.float64),
                "b_val": np.empty((0, self.Q_b), dtype=bool),
                "b_conf": np.empty((0, self.Q_b), dtype=np.float64),
                "in_b": np.empty((0, self.Q_b), dtype=np.int32),
                "s_val": np.empty((0, self.Q_s), dtype=np.float64),
                "p_s": np.empty((0, self.Q_s), dtype=np.float64),
                "s_conf": np.empty((0, self.Q_s), dtype=np.float64),
                "is_ambiguous": np.empty((0,), dtype=bool),
            }

        with np.errstate(over="ignore", invalid="ignore"):
            if self.storage_mode == "float32":
                Xm = X if X.dtype == np.float32 else X.astype(np.float32, copy=False)
                logits = Xm @ self.W_packed + self.b_packed
            elif self.storage_mode == "float16":
                Xm = X if X.dtype == np.float16 else X.astype(np.float16, copy=False)
                logits = Xm @ self.W_packed + self.b_packed
            else:
                Xm = X if X.dtype == np.float32 else X.astype(np.float32, copy=False)
                logits = Xm @ self.W_packed.astype(np.float32) + self.b_packed.astype(np.float32)

        x_row_nonfin = (
            np.array([not bool(np.all(np.isfinite(Xm)))], dtype=bool)
            if is_1d
            else (~np.all(np.isfinite(Xm), axis=1))
        )

        if logits.dtype != np.float64:
            logits = logits.astype(np.float64)
        if is_1d:
            logits = logits[None, :]
        B = logits.shape[0]
        Xm_2d = Xm.reshape(1, -1) if is_1d else Xm
        c_model_ov = np.zeros((B, self.Q_c), dtype=bool)
        b_model_ov = np.zeros((B, self.Q_b), dtype=bool)

        if prompts is not None and self.model_learned_overrides:
            if not logits.flags.writeable:
                logits = logits.copy()
            for r, raw_p in enumerate(prompts):
                if r >= B:
                    break
                if bool(x_row_nonfin[r]):
                    continue
                if not isinstance(raw_p, str) or not raw_p.strip():
                    continue
                norm_key = " ".join(raw_p.strip().casefold().split())
                m_ov = self.model_learned_overrides.get(norm_key)
                if not m_ov:
                    continue
                is_zero_r = not bool(np.any(Xm if is_1d else Xm[r]))
                for ci, fname in enumerate(self.choice_names):
                    if fname in m_ov and (is_zero_r or bool(self.c_zero_head[ci])):
                        s0 = int(self.c_offsets[ci])
                        K = int(self.c_K[ci])
                        z_ov = np.full(K, -12.0, dtype=np.float64)
                        if m_ov[fname] in self.c_options[ci]:
                            z_ov[self.c_options[ci].index(m_ov[fname])] = 12.0
                        logits[r, s0 : s0 + K] = z_ov
                        c_model_ov[r, ci] = True
                for bi, fname in enumerate(self.bool_names):
                    if fname in m_ov and (is_zero_r or bool(self.b_zero_head[bi])):
                        logits[r, self.M_c + bi] = 12.0 if bool(m_ov[fname]) else -12.0
                        b_model_ov[r, bi] = True
                for si, fname in enumerate(self.score_names):
                    if fname in m_ov and (is_zero_r or bool(self.s_zero_head[si])):
                        val_f = float(m_ov[fname])
                        rng_f = max(1e-6, float(self.s_range[si]))
                        ratio = float(np.clip((val_f - float(self.s_min[si])) / rng_f, 1e-4, 1.0 - 1e-4))
                        logit_t = float(np.float32(0.25 * math.log(ratio / (1.0 - ratio))))
                        logits[r, self.M_c + self.Q_b + si] = logit_t

        zc = logits[:, : self.M_c]
        zb = logits[:, self.M_c : self.M_c + self.Q_b]
        zs = logits[:, self.M_c + self.Q_b :]

        if self.Q_c > 0:
            with np.errstate(over="ignore", invalid="ignore"):
                zc_cal = ((zc / 0.25) + self.c_adj_flat[None, :]) / self.c_temp_flat[None, :]
                seg_max = np.maximum.reduceat(zc_cal, self.c_offsets, axis=1)
                exp_z = np.exp(zc_cal - seg_max[:, self.c_seg_idx])
                seg_sum = np.add.reduceat(exp_z, self.c_offsets, axis=1)
                p_c = exp_z / seg_sum[:, self.c_seg_idx]

            p_pad = np.full((B, self.Q_c, self.K_max), -1.0, dtype=np.float64)
            p_pad[:, self.c_pad_rows, self.c_pad_cols] = p_c
            best_idx = np.argmax(p_pad, axis=2)
            p_sorted = -np.sort(-p_pad, axis=2)
            c_conf = p_sorted[:, :, 0]
            bad_c_col = (~np.isfinite(zc)) | (~np.isfinite(zc_cal)) | (~np.isfinite(p_c))
            bad_c_pad = np.zeros((B, self.Q_c, self.K_max), dtype=bool)
            bad_c_pad[:, self.c_pad_rows, self.c_pad_cols] = bad_c_col
            c_nonfin = np.any(bad_c_pad, axis=2) | (~np.isfinite(c_conf)) | x_row_nonfin[:, None]

            x_inf = None
            if (
                self.storage_mode == "float32"
                and len(self.c_raw_w) == self.Q_c
                and self.c_w_l1_max.shape[0] == self.Q_c
            ):
                with np.errstate(over="ignore", invalid="ignore"):
                    x_inf = np.max(
                        np.abs(np.where(np.isfinite(Xm_2d), Xm_2d, 0.0)), axis=1
                    ).astype(np.float64)
                    dz_bound_c = (
                        4.0
                        * (float(self.dimension) + 4.0)
                        * _FLOAT32_EPS
                        * (x_inf[:, None] * self.c_w_l1_max[None, :] + self.c_b_abs_max[None, :])
                    )
                    dz_cal_bound_c = (dz_bound_c / 0.25) / self.c_temp[None, :]
                    dp_bound_c = np.minimum(1.0, dz_cal_bound_c + 1e-9)

                    z_cal_pad = np.full((B, self.Q_c, self.K_max), -np.inf, dtype=np.float64)
                    z_cal_pad[:, self.c_pad_rows, self.c_pad_cols] = zc_cal
                    z_cal_sorted = -np.sort(-z_cal_pad, axis=2)
                    if self.K_max >= 2:
                        near_c = (z_cal_sorted[:, :, 0] - z_cal_sorted[:, :, 1]) <= 2.0 * dz_cal_bound_c
                    else:
                        near_c = np.zeros((B, self.Q_c), dtype=bool)

                    csum_chk = np.cumsum(np.maximum(p_sorted, 0.0), axis=2)
                    prev_csum_chk = np.concatenate(
                        [np.zeros((B, self.Q_c, 1), dtype=np.float64), csum_chk[:, :, :-1]],
                        axis=2,
                    )
                    lac_thr_c = self.c_qhat[None, :, None] + 1e-7
                    near_lac_c = np.any(
                        (p_pad >= 0.0)
                        & (np.abs((1.0 - p_pad) - lac_thr_c) <= dp_bound_c[:, :, None]),
                        axis=2,
                    )
                    aps_cal_thr_c = self.c_qhat[None, :, None] + (1e-7 if self.strict else -1e-7)
                    uncal_thr_c = (1.0 - self.alpha) + (1e-7 if self.strict else -1e-7)
                    aps_thr_pad_c = np.where(
                        self.c_is_cal[None, :, None], aps_cal_thr_c, uncal_thr_c
                    )
                    aps_buf_c = (float(self.K_max) + 1.0) * dp_bound_c[:, :, None] + 1e-6
                    near_aps_c = np.any(
                        (p_sorted >= 0.0)
                        & (
                            (np.abs(csum_chk - aps_thr_pad_c) <= aps_buf_c)
                            | (np.abs(prev_csum_chk - aps_thr_pad_c) <= aps_buf_c)
                        ),
                        axis=2,
                    )
                    near_c |= np.where(
                        self.c_is_cal[None, :],
                        np.where(self.c_is_lac[None, :], near_lac_c, near_aps_c),
                        np.zeros((B, self.Q_c), dtype=bool) if self.strict else near_aps_c,
                    )
                    if not self.strict and np.any(self.c_eff_m > 0.0):
                        ru_chk = (
                            np.maximum(0.0, p_sorted[:, :, 1])
                            if self.K_max >= 2
                            else np.zeros((B, self.Q_c), dtype=np.float64)
                        )
                        marg_chk = c_conf - ru_chk
                        near_mg_c = (self.c_eff_m[None, :] > 0.0) & (
                            (np.abs(marg_chk - self.c_eff_m[None, :]) <= 2.0 * dp_bound_c + 1e-9)
                            | (np.abs(c_conf - self.c_conf_floor[None, :]) <= dp_bound_c + 1e-9)
                            | (
                                np.abs(c_conf - self.c_gamma[None, :] * np.maximum(1e-6, ru_chk))
                                <= (1.0 + self.c_gamma[None, :]) * dp_bound_c + 1e-9
                            )
                            | (np.abs(ru_chk - 1e-6) <= dp_bound_c + 1e-9)
                        )
                        near_c |= near_mg_c

                    near_c &= (~c_nonfin) & (~self.c_zero_head[None, :]) & (~c_model_ov)
                if np.any(near_c):
                    r_idxs, c_idxs = np.nonzero(near_c)
                    for r_pos, ci_pos in zip(r_idxs.tolist(), c_idxs.tolist()):
                        s0 = int(self.c_offsets[ci_pos])
                        K = int(self.c_K[ci_pos])
                        p_ex, ok_ex = _exact_choice_probs_row(
                            Xm_2d[r_pos : r_pos + 1],
                            self.c_raw_w[ci_pos],
                            self.c_raw_b[ci_pos],
                            self.c_adj[ci_pos],
                            float(self.c_temp[ci_pos]),
                        )
                        if ok_ex:
                            p_c[r_pos, s0 : s0 + K] = p_ex
                            p_pad[r_pos, ci_pos, :K] = p_ex
                            best_i_ex = int(np.argmax(p_ex))
                            best_idx[r_pos, ci_pos] = best_i_ex
                            c_conf[r_pos, ci_pos] = float(p_ex[best_i_ex])
                            p_sorted[r_pos, ci_pos] = -np.sort(-p_pad[r_pos, ci_pos])
                        else:
                            p_c[r_pos, s0 : s0 + K] = np.nan
                            p_pad[r_pos, ci_pos, :K] = np.nan
                            best_idx[r_pos, ci_pos] = 0
                            c_conf[r_pos, ci_pos] = np.nan
                            c_nonfin[r_pos, ci_pos] = True

            if self.c_all_lac:
                in_c = np.sum(
                    (p_pad >= 0.0) & ((1.0 - p_pad) <= self.c_qhat[None, :, None] + 1e-7),
                    axis=2,
                )
            elif self.c_all_aps_cal:
                csum = np.cumsum(np.maximum(p_sorted, 0.0), axis=2)
                if self.strict:
                    in_c = np.sum(
                        (p_sorted >= 0.0) & (csum <= self.c_qhat[None, :, None] + 1e-7),
                        axis=2,
                    )
                else:
                    prev_csum = np.concatenate(
                        [np.zeros((B, self.Q_c, 1), dtype=np.float64), csum[:, :, :-1]],
                        axis=2,
                    )
                    in_c = np.maximum(
                        1,
                        np.sum(
                            (p_sorted >= 0.0) & (prev_csum < self.c_qhat[None, :, None] - 1e-7),
                            axis=2,
                        ),
                    )
            else:
                csum = np.cumsum(np.maximum(p_sorted, 0.0), axis=2)
                prev_csum = np.concatenate(
                    [np.zeros((B, self.Q_c, 1), dtype=np.float64), csum[:, :, :-1]],
                    axis=2,
                )
                in_lac = np.sum(
                    (p_pad >= 0.0) & ((1.0 - p_pad) <= self.c_qhat[None, :, None] + 1e-7),
                    axis=2,
                )
                if self.strict:
                    in_aps = np.sum(
                        (p_sorted >= 0.0) & (csum <= self.c_qhat[None, :, None] + 1e-7),
                        axis=2,
                    )
                    in_uncal = np.broadcast_to(self.c_K[None, :], (B, self.Q_c))
                else:
                    in_aps = np.maximum(
                        1,
                        np.sum(
                            (p_sorted >= 0.0) & (prev_csum < self.c_qhat[None, :, None] - 1e-7),
                            axis=2,
                        ),
                    )
                    in_uncal = np.maximum(
                        1,
                        np.sum(
                            (p_sorted >= 0.0) & (prev_csum < (1.0 - self.alpha - 1e-7)),
                            axis=2,
                        ),
                    )
                in_c = np.where(
                    self.c_is_cal[None, :],
                    np.where(self.c_is_lac[None, :], in_lac, in_aps),
                    in_uncal,
                )

            if self.strict:
                c_ambig = (in_c != 1) | (~self.c_is_cal[None, :])
            else:
                runner_up = (
                    np.maximum(0.0, p_sorted[:, :, 1]) if self.K_max >= 2 else np.zeros((B, self.Q_c))
                )
                margin = c_conf - runner_up
                odds = c_conf / np.maximum(1e-6, runner_up)
                mg = (
                    (in_c > 1)
                    & (self.c_eff_m[None, :] > 0.0)
                    & (margin >= self.c_eff_m[None, :])
                    & (c_conf >= self.c_conf_floor[None, :])
                    & (odds >= self.c_gamma[None, :])
                )
                c_ambig = (in_c == 0) | ((in_c > 1) & (~mg))
            if np.any(c_nonfin):
                best_idx = np.array(best_idx, dtype=np.int32, copy=True)
                best_idx[c_nonfin] = 0
                in_c = np.array(in_c, dtype=np.int32, copy=True)
                in_c[c_nonfin] = 0
                c_ambig[c_nonfin] = True
                c_conf[c_nonfin] = np.nan
        else:
            p_c = np.empty((B, 0), dtype=np.float64)
            best_idx = in_c = np.empty((B, 0), dtype=np.int32)
            c_conf = np.empty((B, 0), dtype=np.float64)
            c_ambig = c_nonfin = np.empty((B, 0), dtype=bool)

        if self.Q_b > 0:
            with np.errstate(over="ignore", invalid="ignore"):
                scaled_zb = zb / 0.25
                s0 = (-0.5 * scaled_zb + self.b_adj0[None, :]) / self.b_temp[None, :]
                s1 = (0.5 * scaled_zb + self.b_adj1[None, :]) / self.b_temp[None, :]
                mb = np.maximum(s0, s1)
                e0 = np.exp(s0 - mb)
                e1 = np.exp(s1 - mb)
                denom_b = e0 + e1
                p_false = e0 / denom_b
                p_true = e1 / denom_b
                b_val = p_true >= self.b_thresh[None, :]
                b_conf = np.where(b_val, p_true, p_false)
                b_top = np.maximum(p_true, p_false)
            b_nonfin = (
                (~np.isfinite(zb))
                | (~np.isfinite(s0))
                | (~np.isfinite(s1))
                | (~np.isfinite(p_true))
                | (~np.isfinite(p_false))
                | (~np.isfinite(b_conf))
                | x_row_nonfin[:, None]
            )

            if (
                self.storage_mode == "float32"
                and len(self.b_raw_w) == self.Q_b
                and self.b_w_l1.shape[0] == self.Q_b
            ):
                with np.errstate(over="ignore", invalid="ignore"):
                    if self.Q_c == 0 or x_inf is None:
                        x_inf = np.max(
                            np.abs(np.where(np.isfinite(Xm_2d), Xm_2d, 0.0)), axis=1
                        ).astype(np.float64)
                    dz_bound_b = (
                        4.0
                        * (float(self.dimension) + 4.0)
                        * _FLOAT32_EPS
                        * (x_inf[:, None] * self.b_w_l1[None, :] + self.b_b_abs[None, :])
                    )
                    dz_cal_b = (dz_bound_b / 0.25) / self.b_temp[None, :]
                    dp_bound_b = np.minimum(1.0, 0.5 * dz_cal_b + 1e-9)

                    near_b = (np.abs(p_true - self.b_thresh[None, :]) <= dp_bound_b) | (
                        np.abs(p_true - 0.5) <= dp_bound_b
                    )
                    lac_thr_b = self.b_qhat[None, :] + 1e-7
                    near_lac_b = (
                        np.abs((1.0 - p_true) - lac_thr_b) <= dp_bound_b
                    ) | (np.abs((1.0 - p_false) - lac_thr_b) <= dp_bound_b)
                    aps_cal_thr_b = self.b_qhat[None, :] + (1e-7 if self.strict else -1e-7)
                    uncal_thr_b = (1.0 - self.alpha) + (1e-7 if self.strict else -1e-7)
                    aps_thr_b = np.where(self.b_is_cal[None, :], aps_cal_thr_b, uncal_thr_b)
                    near_aps_b = np.abs(b_top - aps_thr_b) <= dp_bound_b
                    near_b |= np.where(
                        self.b_is_cal[None, :],
                        np.where(self.b_is_lac[None, :], near_lac_b, near_aps_b),
                        np.zeros((B, self.Q_b), dtype=bool) if self.strict else near_aps_b,
                    )
                    if not self.strict and np.any(self.b_eff_m > 0.0):
                        b_bot_chk = 1.0 - b_top
                        marg_b_chk = b_top - b_bot_chk
                        near_mg_b = (self.b_eff_m[None, :] > 0.0) & (
                            (np.abs(marg_b_chk - self.b_eff_m[None, :]) <= 2.0 * dp_bound_b + 1e-9)
                            | (np.abs(b_top - self.b_conf_floor[None, :]) <= dp_bound_b + 1e-9)
                            | (
                                np.abs(b_top - self.b_gamma[None, :] * np.maximum(1e-6, b_bot_chk))
                                <= (1.0 + self.b_gamma[None, :]) * dp_bound_b + 1e-9
                            )
                            | (np.abs(b_bot_chk - 1e-6) <= dp_bound_b + 1e-9)
                        )
                        near_b |= near_mg_b

                    near_b &= (~b_nonfin) & (~self.b_zero_head[None, :]) & (~b_model_ov)
                if np.any(near_b):
                    r_idxs, b_idxs = np.nonzero(near_b)
                    for r_pos, bi_pos in zip(r_idxs.tolist(), b_idxs.tolist()):
                        zv_ex, s0_ex, s1_ex, pt_ex, pf_ex, ok_ex = _exact_bool_probs_row(
                            Xm_2d[r_pos : r_pos + 1],
                            self.b_raw_w[bi_pos],
                            self.b_raw_b[bi_pos],
                            float(self.b_adj0[bi_pos]),
                            float(self.b_adj1[bi_pos]),
                            float(self.b_temp[bi_pos]),
                        )
                        if ok_ex:
                            bv_ex = bool(pt_ex >= float(self.b_thresh[bi_pos]))
                            p_true[r_pos, bi_pos] = pt_ex
                            b_val[r_pos, bi_pos] = bv_ex
                            b_conf[r_pos, bi_pos] = pt_ex if bv_ex else pf_ex
                            b_top[r_pos, bi_pos] = pt_ex if pt_ex >= pf_ex else pf_ex
                            zb[r_pos, bi_pos] = zv_ex
                            s0[r_pos, bi_pos] = s0_ex
                            s1[r_pos, bi_pos] = s1_ex
                            p_false[r_pos, bi_pos] = pf_ex
                        else:
                            p_true[r_pos, bi_pos] = float("nan")
                            b_val[r_pos, bi_pos] = False
                            b_conf[r_pos, bi_pos] = float("nan")
                            b_nonfin[r_pos, bi_pos] = True

            if self.b_all_lac:
                in_b = ((1.0 - p_true) <= self.b_qhat[None, :] + 1e-7).astype(np.int32) + (
                    (1.0 - p_false) <= self.b_qhat[None, :] + 1e-7
                ).astype(np.int32)
            elif self.b_all_aps_cal:
                if self.strict:
                    in_b = (b_top <= self.b_qhat[None, :] + 1e-7).astype(np.int32) + (
                        1.0 <= self.b_qhat[None, :] + 1e-7
                    ).astype(np.int32)
                else:
                    in_b = 1 + (b_top < self.b_qhat[None, :] - 1e-7).astype(np.int32)
            else:
                in_lac_b = ((1.0 - p_true) <= self.b_qhat[None, :] + 1e-7).astype(np.int32) + (
                    (1.0 - p_false) <= self.b_qhat[None, :] + 1e-7
                ).astype(np.int32)
                if self.strict:
                    in_aps_b = (b_top <= self.b_qhat[None, :] + 1e-7).astype(np.int32) + (
                        1.0 <= self.b_qhat[None, :] + 1e-7
                    ).astype(np.int32)
                    in_uncal_b = np.full((B, self.Q_b), 2, dtype=np.int32)
                else:
                    in_aps_b = 1 + (b_top < self.b_qhat[None, :] - 1e-7).astype(np.int32)
                    in_uncal_b = 1 + (b_top < (1.0 - self.alpha - 1e-7)).astype(np.int32)
                in_b = np.where(
                    self.b_is_cal[None, :],
                    np.where(self.b_is_lac[None, :], in_lac_b, in_aps_b),
                    in_uncal_b,
                )

            if self.strict:
                b_ambig = (in_b != 1) | (~self.b_is_cal[None, :])
            else:
                b_bot = 1.0 - b_top
                margin_b = b_top - b_bot
                odds_b = b_top / np.maximum(1e-6, b_bot)
                mg_b = (
                    (in_b > 1)
                    & (self.b_eff_m[None, :] > 0.0)
                    & (margin_b >= self.b_eff_m[None, :])
                    & (b_top >= self.b_conf_floor[None, :])
                    & (odds_b >= self.b_gamma[None, :])
                )
                b_ambig = (in_b == 0) | ((in_b > 1) & (~mg_b))
            if np.any(b_nonfin):
                b_val[b_nonfin] = False
                in_b[b_nonfin] = 0
                b_ambig[b_nonfin] = True
                b_conf[b_nonfin] = np.nan
        else:
            p_true = b_conf = np.empty((B, 0), dtype=np.float64)
            b_val = b_ambig = b_nonfin = np.empty((B, 0), dtype=bool)
            in_b = np.empty((B, 0), dtype=np.int32)

        if self.Q_s > 0:
            with np.errstate(over="ignore", invalid="ignore"):
                scaled_zs = zs / 0.25
                zs_cal = scaled_zs / self.s_raw_scale[None, :]
                p_s = _stable_sigmoid(zs_cal)
                s_val = self.s_min[None, :] + self.s_range[None, :] * p_s
                p_s_raw = _stable_sigmoid(scaled_zs)
                s_conf = np.maximum(p_s_raw, 1.0 - p_s_raw)
            s_nonfin = (
                (~np.isfinite(zs))
                | (~np.isfinite(scaled_zs))
                | (~np.isfinite(zs_cal))
                | (~np.isfinite(p_s))
                | (~np.isfinite(s_val))
                | (~np.isfinite(p_s_raw))
                | (~np.isfinite(s_conf))
                | x_row_nonfin[:, None]
            )
            if self.strict:
                s_ambig = np.broadcast_to(
                    (~self.s_is_cal) | (self.s_norm_margin > self.eff_score_thresh),
                    (B, self.Q_s),
                )
            else:
                s_ambig = np.zeros((B, self.Q_s), dtype=bool)
            if np.any(s_nonfin):
                if not s_ambig.flags.writeable:
                    s_ambig = np.array(s_ambig, dtype=bool, copy=True)
                s_ambig[s_nonfin] = True
                s_val[s_nonfin] = np.broadcast_to(self.s_default[None, :], s_val.shape)[s_nonfin]
                s_conf[s_nonfin] = np.nan
        else:
            p_s = s_val = s_conf = np.empty((B, 0), dtype=np.float64)
            s_ambig = s_nonfin = np.empty((B, 0), dtype=bool)

        if self.is_tfidf_projector:
            zero_rows = (
                np.array([not bool(np.any(Xm))], dtype=bool)
                if is_1d
                else (~np.any(Xm, axis=1))
            )
            if np.any(zero_rows):
                if self.Q_c:
                    in_c[zero_rows, :] = 0
                    c_ambig[zero_rows, :] = True
                if self.Q_b:
                    in_b[zero_rows, :] = 0
                    b_ambig[zero_rows, :] = True
                if self.Q_s:
                    if not s_ambig.flags.writeable:
                        s_ambig = np.array(s_ambig, dtype=bool, copy=True)
                    s_ambig[zero_rows, :] = True

        if prompts is not None and self.learned_overrides:
            for r, raw_p in enumerate(prompts):
                if r >= B:
                    break
                if not isinstance(raw_p, str) or not raw_p.strip():
                    continue
                norm_key = " ".join(raw_p.strip().casefold().split())
                ov = self.learned_overrides.get(norm_key)
                if not ov:
                    continue
                is_zero_r = not bool(np.any(Xm if is_1d else Xm[r]))
                for ci, fname in enumerate(self.choice_names):
                    if (
                        not c_nonfin[r, ci]
                        and fname in ov
                        and (is_zero_r or self.c_options[ci][int(best_idx[r, ci])] == ov[fname])
                    ):
                        in_c[r, ci] = 1
                        c_ambig[r, ci] = False
                for bi, fname in enumerate(self.bool_names):
                    if (
                        not b_nonfin[r, bi]
                        and fname in ov
                        and (is_zero_r or bool(b_val[r, bi]) == bool(ov[fname]))
                    ):
                        in_b[r, bi] = 1
                        b_ambig[r, bi] = False
                for si, fname in enumerate(self.score_names):
                    if not s_nonfin[r, si] and fname in ov:
                        if not s_ambig.flags.writeable:
                            s_ambig = np.array(s_ambig, dtype=bool, copy=True)
                        s_ambig[r, si] = False

        any_nonfin_row = x_row_nonfin.copy()
        if self.Q_c:
            any_nonfin_row |= np.any(c_nonfin, axis=1)
        if self.Q_b:
            any_nonfin_row |= np.any(b_nonfin, axis=1)
        if self.Q_s:
            any_nonfin_row |= np.any(s_nonfin, axis=1)

        row_ambig = (
            (np.any(c_ambig[:, self.c_escalate], axis=1) if self.c_any_escalate else np.zeros(B, dtype=bool))
            | (np.any(b_ambig[:, self.b_escalate], axis=1) if self.b_any_escalate else np.zeros(B, dtype=bool))
            | (np.any(s_ambig[:, self.s_escalate], axis=1) if self.s_any_escalate else np.zeros(B, dtype=bool))
            | any_nonfin_row
        )

        return {
            "p_c": p_c,
            "best_idx": best_idx,
            "c_conf": c_conf,
            "in_c": in_c,
            "p_true": p_true,
            "b_val": b_val,
            "b_conf": b_conf,
            "in_b": in_b,
            "s_val": s_val,
            "p_s": p_s,
            "s_conf": s_conf,
            "is_ambiguous": row_ambig,
        }

    def encode_one(self, prompt: str) -> np.ndarray:
        if type(self.projector) is DeterministicSemanticProjector:
            raw = _fast_deterministic_project(
                self.projector, prompt, recency_weighted=self.recency_weighted
            )
            return _normalize_raw_projection(
                raw, self.dimension, zero_unit0=self.zero_emb_unit0
            )
        if not isinstance(prompt, str):
            raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}")
        if hasattr(self.projector, "project"):
            try:
                raw = self.projector.project(prompt, recency_weighted=self.recency_weighted)
            except TypeError:
                raw = self.projector.project(prompt)
        else:
            raw = self.projector(prompt)
        return _normalize_raw_projection(
            raw, self.dimension, zero_unit0=self.zero_emb_unit0
        )

    def encode_batch(self, prompts: Sequence[str]) -> np.ndarray:
        if len(prompts) == 0:
            return np.zeros((0, self.dimension), dtype=np.float32)
        for idx, p in enumerate(prompts):
            if not isinstance(p, str):
                raise TypeError(
                    f"All elements in prompts must be strings, got {type(p).__name__} at index {idx}"
                )
        return np.stack([self.encode_one(p) for p in prompts], axis=0).astype(np.float32, copy=False)

    def decide_prompt(self, prompt: str) -> Dict[str, Any]:
        return self.forward_embeddings(self.encode_one(prompt), prompts=(prompt,))

    def decide_batch(self, prompts: Sequence[str]) -> Dict[str, Any]:
        return self.forward_embeddings(self.encode_batch(prompts), prompts=prompts)


class FastDecisionOutput:
    """Slotted decision output avoiding per-call Pydantic/receipt overhead on hot paths."""

    __slots__ = ("values", "confidences", "is_ambiguous")

    def __init__(
        self,
        values: Dict[str, Any],
        confidences: Dict[str, float],
        is_ambiguous: bool,
    ) -> None:
        self.values = values
        self.confidences = confidences
        self.is_ambiguous = is_ambiguous

    def to_dict(self) -> Dict[str, Any]:
        return {
            "values": self.values,
            "confidences": self.confidences,
            "is_ambiguous": self.is_ambiguous,
        }


@dataclass(frozen=True, slots=True)
class _BoundCallableState:
    kernel: PackedMultiQuestionKernel
    projector: Any
    recency_weighted: bool
    is_det_proj: bool
    small_q: bool
    W_t: np.ndarray
    b: np.ndarray
    strict: bool
    alpha: float
    Q_c: int
    Q_b: int
    Q_s: int
    c_options: Tuple[Tuple[str, ...], ...]
    c_names: Tuple[str, ...]
    c_K_list: Tuple[int, ...]
    c_starts: Tuple[int, ...]
    c_temp: Tuple[float, ...]
    c_adj: Tuple[Tuple[float, ...], ...]
    c_qhats: Tuple[float, ...]
    c_cals: Tuple[bool, ...]
    c_lacs: Tuple[bool, ...]
    c_eff_m: Tuple[float, ...]
    c_gamma: Tuple[float, ...]
    c_conf_floor: Tuple[float, ...]
    b_names: Tuple[str, ...]
    b_thresh: Tuple[float, ...]
    b_temp: Tuple[float, ...]
    b_adj0: Tuple[float, ...]
    b_adj1: Tuple[float, ...]
    b_qhats: Tuple[float, ...]
    b_cals: Tuple[bool, ...]
    b_lacs: Tuple[bool, ...]
    b_eff_m: Tuple[float, ...]
    b_gamma: Tuple[float, ...]
    b_conf_floor: Tuple[float, ...]
    b_offset: int
    s_names: Tuple[str, ...]
    s_min: Tuple[float, ...]
    s_range: Tuple[float, ...]
    s_norm_margin: Tuple[float, ...]
    s_raw_scale: Tuple[float, ...]
    s_cals: Tuple[bool, ...]
    s_eff_thresh: float
    s_offset: int
    c_escalate: Tuple[bool, ...]
    b_escalate: Tuple[bool, ...]
    s_escalate: Tuple[bool, ...]
    any_escalating_field: bool
    is_tfidf_projector: bool
    learned_overrides: Mapping[str, Mapping[str, Any]]
    schema_id: int
    schema_digest: Optional[str]
    model_schema_id: int
    model_schema_digest: Optional[str]
    model_version: int
    policy_epoch: int
    engine_sig: Tuple[Any, ...] = ()
    dimension_f: float = 0.0
    c_zero_head: Tuple[bool, ...] = ()
    b_zero_head: Tuple[bool, ...] = ()
    c_raw_w: Tuple[np.ndarray, ...] = ()
    c_raw_b: Tuple[np.ndarray, ...] = ()
    c_w_l1_max: Tuple[float, ...] = ()
    c_b_abs_max: Tuple[float, ...] = ()
    b_raw_w: Tuple[np.ndarray, ...] = ()
    b_raw_b: Tuple[np.ndarray, ...] = ()
    b_w_l1: Tuple[float, ...] = ()
    b_b_abs: Tuple[float, ...] = ()
    s_default: Tuple[float, ...] = ()


def _snapshot_engine_schema_state(eng: Any) -> Tuple[Any, Any, int, Optional[str], int, Optional[str]]:
    live_schema = getattr(eng, "schema", None)
    live_model_schema = getattr(getattr(eng, "model", None), "schema", None)
    s_id = id(live_schema) if live_schema is not None else 0
    s_dig = (
        live_schema.schema_digest()
        if live_schema is not None and hasattr(live_schema, "schema_digest")
        else None
    )
    ms_id = id(live_model_schema) if live_model_schema is not None else 0
    ms_dig = (
        live_model_schema.schema_digest()
        if live_model_schema is not None and hasattr(live_model_schema, "schema_digest")
        else None
    )
    return live_schema, live_model_schema, s_id, s_dig, ms_id, ms_dig


class FastDecisionCallable:
    """Pre-bound callable compiled from a SystemOneEngine (zero lock/validation/receipt overhead)."""

    def __init__(self, eng: Any, *, storage_mode: str = "float32") -> None:
        self._storage_mode = storage_mode
        self._aci_tracker = getattr(eng, "aci_tracker", None)
        self._eng_ref = eng
        self._rebind_lock = threading.Lock()
        eng_lock = getattr(eng, "_lock", None)
        mdl_lock = getattr(getattr(eng, "model", None), "_lock", None)
        with (eng_lock if eng_lock is not None else nullcontext()), (
            mdl_lock if mdl_lock is not None else nullcontext()
        ):
            live_schema, live_model_schema, s_id, s_dig, ms_id, ms_dig = _snapshot_engine_schema_state(eng)
            self._schema_ref = live_schema
            self._model_schema_ref = live_model_schema
            mv = int(getattr(eng, "model_version", 1))
            pe = int(getattr(eng, "policy_epoch", 0))
            if hasattr(eng, "compile_packed_kernel"):
                kernel = eng.compile_packed_kernel(storage_mode=storage_mode)
            else:
                kernel = PackedMultiQuestionKernel.from_engine(eng, storage_mode=storage_mode)
            live_sig = getattr(kernel, "_inference_sig", None)
            if live_sig is None:
                live_sig = _compute_engine_inference_signature(
                    eng, storage_mode, effective_alpha=kernel.alpha
                )
            self._state = self._build_state(
                kernel,
                storage_mode=storage_mode,
                schema_id=s_id,
                schema_digest=s_dig,
                model_schema_id=ms_id,
                model_schema_digest=ms_dig,
                model_version=mv,
                policy_epoch=pe,
                engine_sig=live_sig,
            )

    @staticmethod
    def _build_state(
        kernel: PackedMultiQuestionKernel,
        *,
        storage_mode: str,
        schema_id: int = 0,
        schema_digest: Optional[str] = None,
        model_schema_id: int = 0,
        model_schema_digest: Optional[str] = None,
        model_version: int = 1,
        policy_epoch: int = 0,
        engine_sig: Tuple[Any, ...] = (),
    ) -> _BoundCallableState:
        W_t = _owned_readonly_array(kernel.W_packed, dtype=np.float32)
        b = _owned_readonly_array(kernel.b_packed, dtype=np.float32)
        return _BoundCallableState(
            kernel=kernel,
            projector=kernel.projector,
            recency_weighted=kernel.recency_weighted,
            is_det_proj=type(kernel.projector) is DeterministicSemanticProjector,
            small_q=kernel.Q_total <= 8 and storage_mode == "float32",
            W_t=W_t,
            b=b,
            strict=kernel.strict,
            alpha=kernel.alpha,
            Q_c=kernel.Q_c,
            Q_b=kernel.Q_b,
            Q_s=kernel.Q_s,
            c_options=tuple(tuple(opts) for opts in kernel.c_options),
            c_names=tuple(kernel.choice_names),
            c_K_list=tuple(int(x) for x in kernel.c_K),
            c_starts=tuple(int(x) for x in kernel.c_offsets),
            c_temp=tuple(float(x) for x in kernel.c_temp),
            c_adj=tuple(tuple(float(v) for v in arr) for arr in kernel.c_adj),
            c_qhats=tuple(float(x) for x in kernel.c_qhat),
            c_cals=tuple(bool(x) for x in kernel.c_is_cal),
            c_lacs=tuple(bool(x) for x in kernel.c_is_lac),
            c_eff_m=tuple(float(x) for x in kernel.c_eff_m),
            c_gamma=tuple(float(x) for x in kernel.c_gamma),
            c_conf_floor=tuple(float(x) for x in kernel.c_conf_floor),
            b_names=tuple(kernel.bool_names),
            b_thresh=tuple(float(x) for x in kernel.b_thresh),
            b_temp=tuple(float(x) for x in kernel.b_temp),
            b_adj0=tuple(float(x) for x in kernel.b_adj0),
            b_adj1=tuple(float(x) for x in kernel.b_adj1),
            b_qhats=tuple(float(x) for x in kernel.b_qhat),
            b_cals=tuple(bool(x) for x in kernel.b_is_cal),
            b_lacs=tuple(bool(x) for x in kernel.b_is_lac),
            b_eff_m=tuple(float(x) for x in kernel.b_eff_m),
            b_gamma=tuple(float(x) for x in kernel.b_gamma),
            b_conf_floor=tuple(float(x) for x in kernel.b_conf_floor),
            b_offset=kernel.M_c,
            s_names=tuple(kernel.score_names),
            s_min=tuple(float(x) for x in kernel.s_min),
            s_range=tuple(float(x) for x in kernel.s_range),
            s_norm_margin=tuple(float(x) for x in kernel.s_norm_margin),
            s_raw_scale=tuple(float(x) for x in kernel.s_raw_scale),
            s_cals=tuple(bool(x) for x in kernel.s_is_cal),
            s_eff_thresh=float(kernel.eff_score_thresh),
            s_offset=kernel.M_c + kernel.Q_b,
            c_escalate=tuple(bool(x) for x in kernel.c_escalate),
            b_escalate=tuple(bool(x) for x in kernel.b_escalate),
            s_escalate=tuple(bool(x) for x in kernel.s_escalate),
            any_escalating_field=kernel.any_escalating_field,
            is_tfidf_projector=kernel.is_tfidf_projector,
            learned_overrides=kernel.learned_overrides,
            schema_id=schema_id,
            schema_digest=schema_digest,
            model_schema_id=model_schema_id,
            model_schema_digest=model_schema_digest,
            model_version=model_version,
            policy_epoch=policy_epoch,
            engine_sig=engine_sig,
            dimension_f=float(kernel.dimension),
            c_zero_head=tuple(bool(x) for x in kernel.c_zero_head),
            b_zero_head=tuple(bool(x) for x in kernel.b_zero_head),
            c_raw_w=kernel.c_raw_w,
            c_raw_b=kernel.c_raw_b,
            c_w_l1_max=tuple(float(x) for x in kernel.c_w_l1_max),
            c_b_abs_max=tuple(float(x) for x in kernel.c_b_abs_max),
            b_raw_w=kernel.b_raw_w,
            b_raw_b=kernel.b_raw_b,
            b_w_l1=tuple(float(x) for x in kernel.b_w_l1),
            b_b_abs=tuple(float(x) for x in kernel.b_b_abs),
            s_default=tuple(float(x) for x in kernel.s_default),
        )

    def _bind_kernel(self, kernel: PackedMultiQuestionKernel, *, storage_mode: str) -> None:
        self._state = self._build_state(
            kernel,
            storage_mode=storage_mode,
            engine_sig=getattr(kernel, "_inference_sig", None) or (),
        )

    def __getattr__(self, name: str) -> Any:
        state = object.__getattribute__(self, "_state")
        return getattr(state, name)

    @staticmethod
    def _validate_live_constraints(
        eng_ref: Any, live_schema: Any, live_model_schema: Any, tracker: Any
    ) -> None:
        if eng_ref is not None:
            _validate_packed_engine_support(eng_ref)
            return
        if _has_custom_validate_decision(live_schema) or _has_custom_validate_decision(
            live_model_schema
        ):
            raise ValueError(
                "PackedMultiQuestionKernel does not support custom DecisionSchema.validate_decision"
            )
        if tracker is not None and getattr(tracker, "margin_floor", None) is not None:
            raise ValueError(
                "PackedMultiQuestionKernel does not support engines with active aci_tracker margin_floor"
            )

    def __call__(
        self, prompt: Optional[str] = None, *, embedding: Optional[np.ndarray] = None
    ) -> FastDecisionOutput:
        st = self._state
        eng_ref = self._eng_ref
        if eng_ref is not None:
            live_schema, live_model_schema, s_id, s_dig, ms_id, ms_dig = _snapshot_engine_schema_state(
                eng_ref
            )
            tracker = getattr(eng_ref, "aci_tracker", None)
            self._validate_live_constraints(eng_ref, live_schema, live_model_schema, tracker)
            eff_alpha = (
                float(tracker.alpha_t)
                if tracker is not None
                else float(getattr(eng_ref, "default_alpha", st.alpha))
            )
            live_sig = _compute_engine_inference_signature(
                eng_ref, self._storage_mode, effective_alpha=eff_alpha
            )
            if eff_alpha != st.alpha or live_sig != st.engine_sig:
                with self._rebind_lock:
                    eng_lock = getattr(eng_ref, "_lock", None)
                    mdl_lock = getattr(getattr(eng_ref, "model", None), "_lock", None)
                    with (eng_lock if eng_lock is not None else nullcontext()), (
                        mdl_lock if mdl_lock is not None else nullcontext()
                    ):
                        st = self._state
                        live_schema, live_model_schema, s_id, s_dig, ms_id, ms_dig = (
                            _snapshot_engine_schema_state(eng_ref)
                        )
                        tracker = getattr(eng_ref, "aci_tracker", None)
                        self._validate_live_constraints(eng_ref, live_schema, live_model_schema, tracker)
                        eff_alpha = (
                            float(tracker.alpha_t)
                            if tracker is not None
                            else float(getattr(eng_ref, "default_alpha", st.alpha))
                        )
                        live_sig = _compute_engine_inference_signature(
                            eng_ref, self._storage_mode, effective_alpha=eff_alpha
                        )
                        if eff_alpha != st.alpha or live_sig != st.engine_sig:
                            if hasattr(eng_ref, "compile_packed_kernel"):
                                new_kernel = eng_ref.compile_packed_kernel(
                                    storage_mode=self._storage_mode
                                )
                            else:
                                new_kernel = PackedMultiQuestionKernel.from_engine(
                                    eng_ref,
                                    storage_mode=self._storage_mode,
                                    alpha=eff_alpha,
                                )
                            live_mv = int(getattr(eng_ref, "model_version", st.model_version))
                            live_pe = int(getattr(eng_ref, "policy_epoch", st.policy_epoch))
                            st = self._build_state(
                                new_kernel,
                                storage_mode=self._storage_mode,
                                schema_id=s_id,
                                schema_digest=s_dig,
                                model_schema_id=ms_id,
                                model_schema_digest=ms_dig,
                                model_version=live_mv,
                                policy_epoch=live_pe,
                                engine_sig=getattr(new_kernel, "_inference_sig", None) or live_sig,
                            )
                            self._state = st
                            self._schema_ref = live_schema
                            self._model_schema_ref = live_model_schema
        else:
            self._validate_live_constraints(
                None, self._schema_ref, self._model_schema_ref, self._aci_tracker
            )
        if embedding is None:
            if prompt is None:
                raise ValueError("Either prompt or embedding must be provided")
            emb = st.kernel.encode_one(prompt)
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                emb = (
                    embedding
                    if isinstance(embedding, np.ndarray) and embedding.dtype == np.float32
                    else np.asarray(embedding, dtype=np.float32)
                )
            if emb.ndim != 1:
                emb = emb.ravel()
            if emb.shape[0] != st.kernel.dimension:
                raise ValueError(
                    f"embedding dimension mismatch: expected {st.kernel.dimension}, got {emb.shape[0]}"
                )

        has_prompt_override = bool(
            embedding is None
            and st.learned_overrides
            and isinstance(prompt, str)
            and prompt.strip()
            and (" ".join(prompt.strip().casefold().split()) in st.learned_overrides)
        )
        if not st.small_q or has_prompt_override:
            out = st.kernel.forward_embeddings(
                emb, prompts=(prompt,) if has_prompt_override else None
            )
            vals: Dict[str, Any] = {}
            confs: Dict[str, float] = {}
            for ci, fn in enumerate(st.c_names):
                vals[fn] = st.c_options[ci][int(out["best_idx"][0, ci])]
                confs[fn] = float(out["c_conf"][0, ci])
            for bi, fn in enumerate(st.b_names):
                vals[fn] = bool(out["b_val"][0, bi])
                confs[fn] = float(out["b_conf"][0, bi])
            for si, fn in enumerate(st.s_names):
                vals[fn] = float(out["s_val"][0, si])
                confs[fn] = float(out["s_conf"][0, si])
            return FastDecisionOutput(vals, confs, bool(out["is_ambiguous"][0]))

        with np.errstate(over="ignore", invalid="ignore"):
            z_list: List[float] = (emb @ st.W_t + st.b).tolist()
        vals = {}
        confs = {}
        emb_ok = bool(np.all(np.isfinite(emb)))
        x_inf_val = (
            float(np.max(np.abs(emb)))
            if (emb_ok and (st.Q_c or st.Q_b) and emb.size)
            else 0.0
        )
        ambig = (not emb_ok) or bool(
            st.is_tfidf_projector and not bool(np.any(emb)) and st.any_escalating_field
        )
        exp = math.exp
        isfinite = math.isfinite
        strict = st.strict

        for ci in range(st.Q_c):
            fn = st.c_names[ci]
            K = st.c_K_list[ci]
            s0 = st.c_starts[ci]
            temp = st.c_temp[ci]
            adj_vec = st.c_adj[ci]
            z_cal = [(z_list[s0 + j] / 0.25 + adj_vec[j]) / temp for j in range(K)]
            if (not emb_ok) or not (
                all(isfinite(z_list[s0 + j]) for j in range(K))
                and all(isfinite(v) for v in z_cal)
            ):
                vals[fn] = st.c_options[ci][0]
                confs[fn] = float("nan")
                ambig = True
                continue
            m = max(z_cal)
            exps = [exp(v - m) for v in z_cal]
            sum_e = sum(exps)
            if not (isfinite(sum_e) and sum_e > 0.0):
                vals[fn] = st.c_options[ci][0]
                confs[fn] = float("nan")
                ambig = True
                continue
            inv_sum = 1.0 / sum_e
            probs = [e * inv_sum for e in exps]
            if not all(isfinite(pv) for pv in probs):
                vals[fn] = st.c_options[ci][0]
                confs[fn] = float("nan")
                ambig = True
                continue
            if len(st.c_raw_w) == st.Q_c and not st.c_zero_head[ci]:
                dz_b = (
                    4.0
                    * (st.dimension_f + 4.0)
                    * _FLOAT32_EPS
                    * (x_inf_val * st.c_w_l1_max[ci] + st.c_b_abs_max[ci])
                )
                dz_cal_b = (dz_b / 0.25) / temp
                dp_b = min(1.0, dz_cal_b + 1e-9)
                near_ci = False
                if K >= 2:
                    zcs = sorted(z_cal, reverse=True)
                    if (zcs[0] - zcs[1]) <= 2.0 * dz_cal_b:
                        near_ci = True
                if not near_ci:
                    if st.c_cals[ci]:
                        qh_chk = st.c_qhats[ci]
                        if st.c_lacs[ci]:
                            thr_lac = qh_chk + 1e-7
                            if any(abs((1.0 - pv) - thr_lac) <= dp_b for pv in probs):
                                near_ci = True
                        else:
                            aps_thr = qh_chk + (1e-7 if strict else -1e-7)
                            aps_buf = (K + 1.0) * dp_b + 1e-6
                            ps_chk = sorted(probs, reverse=True)
                            cs_chk = 0.0
                            for pv in ps_chk:
                                if abs(cs_chk - aps_thr) <= aps_buf:
                                    near_ci = True
                                    break
                                cs_chk += pv
                                if abs(cs_chk - aps_thr) <= aps_buf:
                                    near_ci = True
                                    break
                    elif not strict:
                        aps_thr = (1.0 - st.alpha) - 1e-7
                        aps_buf = (K + 1.0) * dp_b + 1e-6
                        ps_chk = sorted(probs, reverse=True)
                        cs_chk = 0.0
                        for pv in ps_chk:
                            if abs(cs_chk - aps_thr) <= aps_buf:
                                near_ci = True
                                break
                            cs_chk += pv
                            if abs(cs_chk - aps_thr) <= aps_buf:
                                near_ci = True
                                break
                if (not near_ci) and (not strict) and st.c_eff_m[ci] > 0.0:
                    ps_chk = sorted(probs, reverse=True)
                    tp_chk = ps_chk[0]
                    ru_chk = ps_chk[1] if K > 1 else 0.0
                    mg_chk = (tp_chk - ru_chk) if K > 1 else 1.0
                    eff_m_chk = st.c_eff_m[ci]
                    gam_chk = st.c_gamma[ci]
                    if (
                        abs(mg_chk - eff_m_chk) <= 2.0 * dp_b + 1e-9
                        or abs(tp_chk - st.c_conf_floor[ci]) <= dp_b + 1e-9
                        or abs(tp_chk - gam_chk * (ru_chk if ru_chk > 1e-6 else 1e-6))
                        <= (1.0 + gam_chk) * dp_b + 1e-9
                        or abs(ru_chk - 1e-6) <= dp_b + 1e-9
                    ):
                        near_ci = True
                if near_ci:
                    p_ex, ok_ex = _exact_choice_probs_row(
                        emb.reshape(1, -1),
                        st.c_raw_w[ci],
                        st.c_raw_b[ci],
                        np.asarray(adj_vec, dtype=np.float64),
                        temp,
                    )
                    if not ok_ex:
                        vals[fn] = st.c_options[ci][0]
                        confs[fn] = float("nan")
                        ambig = True
                        continue
                    probs = p_ex.tolist()
            best_i = 0
            best_p = probs[0]
            for j in range(1, K):
                if probs[j] > best_p:
                    best_p = probs[j]
                    best_i = j
            vals[fn] = st.c_options[ci][best_i]
            confs[fn] = best_p

            ps: Optional[List[float]] = None
            if not st.c_cals[ci]:
                if strict:
                    cnt = K
                else:
                    ps = sorted(probs, reverse=True)
                    cs = 0.0
                    cnt = 0
                    thr = 1.0 - st.alpha - 1e-7
                    for pv in ps:
                        cs += pv
                        cnt += 1
                        if cs >= thr:
                            break
            else:
                qh = st.c_qhats[ci]
                if st.c_lacs[ci]:
                    cnt = sum(1 for pv in probs if (1.0 - pv) <= qh + 1e-7)
                elif strict:
                    ps = sorted(probs, reverse=True)
                    cs = 0.0
                    cnt = 0
                    for pv in ps:
                        cs += pv
                        if cs <= qh + 1e-7:
                            cnt += 1
                        else:
                            break
                else:
                    ps = sorted(probs, reverse=True)
                    cs = 0.0
                    cnt = 0
                    for pv in ps:
                        cs += pv
                        cnt += 1
                        if cs >= qh - 1e-7:
                            break

            field_ambig = False
            if strict:
                if (not st.c_cals[ci]) or (cnt != 1):
                    field_ambig = True
            else:
                if cnt == 0:
                    field_ambig = True
                elif cnt > 1:
                    eff_m = st.c_eff_m[ci]
                    if eff_m > 0.0:
                        if ps is None:
                            ps = sorted(probs, reverse=True)
                        top_p = ps[0]
                        run_p = ps[1] if K > 1 else 0.0
                        marg = (top_p - run_p) if K > 1 else 1.0
                        odds = top_p / (run_p if run_p > 1e-6 else 1e-6)
                        if not (
                            (marg >= eff_m)
                            and (top_p >= st.c_conf_floor[ci])
                            and (odds >= st.c_gamma[ci])
                        ):
                            field_ambig = True
                    else:
                        field_ambig = True
            if field_ambig and st.c_escalate[ci]:
                ambig = True

        for bi in range(st.Q_b):
            fn = st.b_names[bi]
            zv = z_list[st.b_offset + bi]
            scaled_z = zv / 0.25
            temp_b = st.b_temp[bi]
            s0 = (-0.5 * scaled_z + st.b_adj0[bi]) / temp_b
            s1 = (0.5 * scaled_z + st.b_adj1[bi]) / temp_b
            if (not emb_ok) or not (isfinite(zv) and isfinite(s0) and isfinite(s1)):
                vals[fn] = False
                confs[fn] = float("nan")
                ambig = True
                continue
            mb = s0 if s0 >= s1 else s1
            e0 = exp(s0 - mb)
            e1 = exp(s1 - mb)
            denom_b = e0 + e1
            if not (isfinite(denom_b) and denom_b > 0.0):
                vals[fn] = False
                confs[fn] = float("nan")
                ambig = True
                continue
            pf = e0 / denom_b
            pt = e1 / denom_b
            if not (isfinite(pt) and isfinite(pf)):
                vals[fn] = False
                confs[fn] = float("nan")
                ambig = True
                continue
            if len(st.b_raw_w) == st.Q_b and not st.b_zero_head[bi]:
                dz_b = (
                    4.0
                    * (st.dimension_f + 4.0)
                    * _FLOAT32_EPS
                    * (x_inf_val * st.b_w_l1[bi] + st.b_b_abs[bi])
                )
                dp_b = min(1.0, 0.5 * ((dz_b / 0.25) / temp_b) + 1e-9)
                top_chk = pt if pt >= pf else pf
                near_bi = abs(pt - st.b_thresh[bi]) <= dp_b or abs(pt - 0.5) <= dp_b
                if not near_bi:
                    if st.b_cals[bi]:
                        qh_chk = st.b_qhats[bi]
                        if st.b_lacs[bi]:
                            thr_lac = qh_chk + 1e-7
                            if (
                                abs((1.0 - pt) - thr_lac) <= dp_b
                                or abs((1.0 - pf) - thr_lac) <= dp_b
                            ):
                                near_bi = True
                        else:
                            aps_thr = qh_chk + (1e-7 if strict else -1e-7)
                            if abs(top_chk - aps_thr) <= dp_b:
                                near_bi = True
                    elif not strict:
                        aps_thr = (1.0 - st.alpha) - 1e-7
                        if abs(top_chk - aps_thr) <= dp_b:
                            near_bi = True
                if (not near_bi) and (not strict) and st.b_eff_m[bi] > 0.0:
                    bot_chk = 1.0 - top_chk
                    mg_chk = top_chk - bot_chk
                    eff_m_chk = st.b_eff_m[bi]
                    gam_chk = st.b_gamma[bi]
                    if (
                        abs(mg_chk - eff_m_chk) <= 2.0 * dp_b + 1e-9
                        or abs(top_chk - st.b_conf_floor[bi]) <= dp_b + 1e-9
                        or abs(top_chk - gam_chk * (bot_chk if bot_chk > 1e-6 else 1e-6))
                        <= (1.0 + gam_chk) * dp_b + 1e-9
                        or abs(bot_chk - 1e-6) <= dp_b + 1e-9
                    ):
                        near_bi = True
                if near_bi:
                    _, _, _, pt, pf, ok_ex = _exact_bool_probs_row(
                        emb.reshape(1, -1),
                        st.b_raw_w[bi],
                        st.b_raw_b[bi],
                        st.b_adj0[bi],
                        st.b_adj1[bi],
                        temp_b,
                    )
                    if not ok_ex:
                        vals[fn] = False
                        confs[fn] = float("nan")
                        ambig = True
                        continue
            bv = pt >= st.b_thresh[bi]
            vals[fn] = bv
            confs[fn] = pt if bv else pf

            top = pt if pt >= pf else pf
            if not st.b_cals[bi]:
                if strict:
                    cnt = 2
                else:
                    cnt = 1 if top >= (1.0 - st.alpha - 1e-7) else 2
            else:
                qh = st.b_qhats[bi]
                if st.b_lacs[bi]:
                    cnt = (1 if (1.0 - pt) <= qh + 1e-7 else 0) + (
                        1 if (1.0 - pf) <= qh + 1e-7 else 0
                    )
                elif strict:
                    cnt = (1 if top <= qh + 1e-7 else 0) + (1 if 1.0 <= qh + 1e-7 else 0)
                else:
                    cnt = 1 if top >= qh - 1e-7 else 2

            field_ambig = False
            if strict:
                if (not st.b_cals[bi]) or (cnt != 1):
                    field_ambig = True
            else:
                if cnt == 0:
                    field_ambig = True
                elif cnt > 1:
                    eff_m = st.b_eff_m[bi]
                    if eff_m > 0.0:
                        bot = 1.0 - top
                        marg = top - bot
                        odds = top / (bot if bot > 1e-6 else 1e-6)
                        if not (
                            (marg >= eff_m)
                            and (top >= st.b_conf_floor[bi])
                            and (odds >= st.b_gamma[bi])
                        ):
                            field_ambig = True
                    else:
                        field_ambig = True
            if field_ambig and st.b_escalate[bi]:
                ambig = True

        for si in range(st.Q_s):
            fn = st.s_names[si]
            zv = z_list[st.s_offset + si]
            scaled_zs = zv / 0.25
            z_cal = scaled_zs / st.s_raw_scale[si]
            if (not emb_ok) or not (isfinite(zv) and isfinite(scaled_zs) and isfinite(z_cal)):
                vals[fn] = st.s_default[si]
                confs[fn] = float("nan")
                ambig = True
                continue
            if z_cal >= 0.0:
                ps = 1.0 / (1.0 + exp(-z_cal))
            else:
                ez = exp(z_cal)
                ps = ez / (1.0 + ez)
            s_val = st.s_min[si] + st.s_range[si] * ps
            if scaled_zs >= 0.0:
                ps_raw = 1.0 / (1.0 + exp(-scaled_zs))
            else:
                ez_raw = exp(scaled_zs)
                ps_raw = ez_raw / (1.0 + ez_raw)
            s_conf = ps_raw if ps_raw >= 0.5 else (1.0 - ps_raw)
            if not (isfinite(s_val) and isfinite(s_conf)):
                vals[fn] = st.s_default[si]
                confs[fn] = float("nan")
                ambig = True
                continue
            vals[fn] = s_val
            confs[fn] = s_conf
            if (
                strict
                and st.s_escalate[si]
                and ((not st.s_cals[si]) or (st.s_norm_margin[si] > st.s_eff_thresh))
            ):
                ambig = True

        return FastDecisionOutput(vals, confs, ambig)


__all__ = [
    "DeterministicSemanticProjector",
    "_fast_deterministic_project",
    "DecisionFieldHead",
    "RawFieldEvaluation",
    "ModelInferenceResult",
    "SystemOneModel",
    "PackedMultiQuestionKernel",
    "FastDecisionOutput",
    "FastDecisionCallable",
    "HAS_MLX",
    "_stable_softmax",
    "_stable_sigmoid",
    "stable_softmax",
    "stable_sigmoid",
]
