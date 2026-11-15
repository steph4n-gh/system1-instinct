"""Shared-trunk multi-tenant arena and O(1) per-tenant Mondrian conformal updates."""
from __future__ import annotations

import ast
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import io
import json
import math
import re
import struct
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import zlib

import numpy as np

from system1.core.model import (
    _accepts_recency_weighted,
    _encode_projector_batch,
    _encode_projector_one,
    _model_zero_unit0,
)
from system1.core.schema import BooleanField, ChoiceField
from system1.core.text import PragmaticCompositeProjector, TfidfProjector

_TOK_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_DEFAULT_TOK_CACHE: OrderedDict[str, Tuple[int, ...]] = OrderedDict()
_DEFAULT_TOK_CACHE_LIMIT = 4096
_GLOBAL_TOK_LOCK = threading.Lock()

MAX_FLEET_ARCHIVE_RAW_BYTES = 64 * 1024 * 1024
MAX_FLEET_ARCHIVE_EXPANDED_BYTES = 128 * 1024 * 1024
MAX_FLEET_RUNTIME_BYTES = 256 * 1024 * 1024
MAX_FLEET_TENANTS = 100_000
MAX_FLEET_ARCHIVE_ENTRIES = 32
MAX_FLEET_DIRECTORY_BYTES = 65536
MAX_FLEET_SPARSE_CAP = 255
MAX_FLEET_EXEMPLAR_CAP = 255
_FLEET_REQUIRED_ARRAYS = frozenset(
    {
        "W_base",
        "b_base",
        "U_shared",
        "base_tok_df",
        "q_init",
        "dW_q",
        "dW_scale",
        "P_blk",
        "sp_hash",
        "sp_cls",
        "sp_w",
        "sp_len",
        "ex_vecs_q",
        "ex_scales",
        "ex_cls",
        "ex_len",
        "q_c",
        "u_c",
        "alpha_c",
        "err_ewma",
    }
)
_FLEET_ALLOWED_ARCHIVE_MEMBERS = _FLEET_REQUIRED_ARRAYS | frozenset(
    {
        "meta_ints",
        "meta_floats",
        "classes_utf8",
        "raw_W_auth",
        "raw_b_auth",
        "raw_cal_temp",
        "raw_cal_adj",
    }
)


def _read_npy_header_from_stream(stream: Any, uncompressed_size: int) -> Tuple[np.dtype, Tuple[int, ...]]:
    """Inspect an NPY header without allocating its payload buffer."""
    magic = stream.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("Invalid NPY member magic header in fleet snapshot")
    ver = stream.read(2)
    if len(ver) != 2 or ver[0] not in (1, 2, 3):
        raise ValueError(f"Unsupported NPY header version in fleet snapshot: {ver!r}")
    major = ver[0]
    if major == 1:
        len_raw = stream.read(2)
        if len(len_raw) != 2:
            raise ValueError("Truncated NPY v1 header length in fleet snapshot")
        hdr_len = struct.unpack("<H", len_raw)[0]
        prefix_len = 10
    else:
        len_raw = stream.read(4)
        if len(len_raw) != 4:
            raise ValueError("Truncated NPY v2/v3 header length in fleet snapshot")
        hdr_len = struct.unpack("<I", len_raw)[0]
        prefix_len = 12
    if hdr_len <= 0 or hdr_len > 4096:
        raise ValueError(f"Implausible NPY header length in fleet snapshot: {hdr_len}")
    hdr_bytes = stream.read(hdr_len)
    if len(hdr_bytes) != hdr_len:
        raise ValueError("Truncated NPY header dictionary in fleet snapshot")
    encoding = "latin1" if major < 3 else "utf-8"
    hdr = ast.literal_eval(hdr_bytes.decode(encoding).strip())
    if not isinstance(hdr, dict):
        raise ValueError("Invalid NPY header metadata object in fleet snapshot")
    descr = hdr.get("descr")
    shape_raw = hdr.get("shape")
    if not isinstance(shape_raw, tuple) or not all(
        isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0 for dim in shape_raw
    ):
        raise ValueError(f"Invalid NPY shape in fleet snapshot: {shape_raw!r}")
    dtype = np.dtype(descr)
    if dtype.hasobject or dtype.kind not in ("f", "i", "u"):
        raise ValueError(f"Unsupported non-numeric NPY dtype in fleet snapshot: {descr!r}")
    expected_bytes = prefix_len + hdr_len + int(math.prod(shape_raw)) * int(dtype.itemsize)
    if expected_bytes != int(uncompressed_size):
        raise ValueError(
            f"Fleet snapshot NPY byte count mismatch: expected {expected_bytes}, zip reports {uncompressed_size}"
        )
    return dtype, tuple(int(dim) for dim in shape_raw)


def set_fleet_token_cache_limit(maxsize: int) -> None:
    """Configure the bounded LRU cap for module-level fleet token hashing."""
    global _DEFAULT_TOK_CACHE_LIMIT
    limit = max(0, int(maxsize))
    with _GLOBAL_TOK_LOCK:
        _DEFAULT_TOK_CACHE_LIMIT = limit
        while len(_DEFAULT_TOK_CACHE) > limit:
            _DEFAULT_TOK_CACHE.popitem(last=False)


def clear_fleet_token_cache() -> None:
    """Clear the shared module-level fleet token hash cache."""
    with _GLOBAL_TOK_LOCK:
        _DEFAULT_TOK_CACHE.clear()


def tokenize_hashes(
    text: str,
    *,
    cache: Optional[OrderedDict[str, Tuple[int, ...]]] = None,
    cache_limit: Optional[int] = None,
    cache_lock: Optional[Any] = None,
) -> Tuple[int, ...]:
    """Hash normalized tokens into 16-bit buckets with bounded LRU eviction."""
    target_cache = _DEFAULT_TOK_CACHE if cache is None else cache
    lock = _GLOBAL_TOK_LOCK if (cache is None and cache_lock is None) else cache_lock
    limit = _DEFAULT_TOK_CACHE_LIMIT if cache_limit is None else max(0, int(cache_limit))
    if limit > 0:
        if lock is not None:
            with lock:
                cached = target_cache.get(text)
                if cached is not None:
                    target_cache.move_to_end(text)
                    return cached
        else:
            cached = target_cache.get(text)
            if cached is not None:
                target_cache.move_to_end(text)
                return cached

    hashes = tuple(
        zlib.crc32(w.encode("utf-8")) & 0xFFFF
        for w in _TOK_RE.findall(text.lower())
        if len(w) >= 2
    )
    if limit > 0:
        if lock is not None:
            with lock:
                target_cache[text] = hashes
                target_cache.move_to_end(text)
                while len(target_cache) > limit:
                    target_cache.popitem(last=False)
        else:
            target_cache[text] = hashes
            target_cache.move_to_end(text)
            while len(target_cache) > limit:
                target_cache.popitem(last=False)
    return hashes


def _softmax_temp(logits: np.ndarray, temp: float = 0.25) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        arr = np.asarray(logits, dtype=np.float64)
        if arr.size == 0:
            return arr
        t = float(temp)
        if not math.isfinite(t) or t <= 0.0 or not bool(np.all(np.isfinite(arr))):
            return np.full(arr.shape, np.nan, dtype=np.float64)
        z = (arr - np.max(arr)) / t
        if bool(np.any(np.isnan(z))) or bool(np.any(z > 0.0)):
            return np.full(arr.shape, np.nan, dtype=np.float64)
        e = np.exp(z)
        s = float(np.sum(e))
        if not math.isfinite(s) or s <= 0.0:
            return np.full(arr.shape, np.nan, dtype=np.float64)
        out = e / s
        if not bool(np.all(np.isfinite(out))):
            return np.full(arr.shape, np.nan, dtype=np.float64)
        return out


def _round_up_16(val: int) -> int:
    return ((max(16, int(val)) + 15) // 16) * 16


def _ceil_float16_unit(vals: np.ndarray) -> np.ndarray:
    """Conservatively round unit-interval values upward to float16 so thresholds never shrink."""
    v64 = np.clip(np.asarray(vals, dtype=np.float64), 0.0, 1.0)
    q16 = v64.astype(np.float16)
    mask = q16.astype(np.float64) < v64
    if bool(np.any(mask)):
        q16 = np.where(mask, np.nextafter(q16, np.float16(1.0)), q16)
    return q16


def _prepare_shared_subspace(W_base: np.ndarray, r_sub: int, seed: int = 42) -> np.ndarray:
    """Build a (D, r_sub) orthonormal float32 basis anchored on SVD of W_base."""
    W64 = np.asarray(W_base, dtype=np.float64)
    _, d = W64.shape
    _, _, Vt = np.linalg.svd(W64, full_matrices=False)
    U = Vt.T
    target_cols = min(int(r_sub), d)
    if U.shape[1] >= target_cols:
        basis = U[:, :target_cols]
    else:
        rng = np.random.default_rng(seed)
        extra = target_cols - U.shape[1]
        C = rng.standard_normal((d, extra))
        if U.shape[1] > 0:
            C = C - U @ (U.T @ C)
        Q, _ = np.linalg.qr(C)
        if U.shape[1] > 0:
            Q = Q - U @ (U.T @ Q)
            Q, _ = np.linalg.qr(Q)
            basis = np.hstack([U, Q[:, :extra]])
        else:
            basis = Q[:, :extra]
    if target_cols < r_sub:
        pad = np.zeros((d, r_sub - target_cols), dtype=np.float64)
        basis = np.hstack([basis, pad])
    return np.ascontiguousarray(basis, dtype=np.float32)


def _adapt_subspace_width(U_shared: np.ndarray, W_base: np.ndarray, r_sub: int) -> np.ndarray:
    U = np.asarray(U_shared, dtype=np.float32)
    d = W_base.shape[1]
    if U.shape[0] != d:
        raise ValueError(f"U_shared rows {U.shape[0]} != feature dimension {d}")
    if U.shape[1] == r_sub:
        return np.ascontiguousarray(U, dtype=np.float32)
    if U.shape[1] > r_sub:
        return np.ascontiguousarray(U[:, :r_sub], dtype=np.float32)
    full = _prepare_shared_subspace(W_base, r_sub)
    full[:, : U.shape[1]] = U
    return full


@dataclass(frozen=True, slots=True)
class TenantDecisionResult:
    """Structured prediction and Mondrian conformal set for one tenant query."""

    tenant_id: int
    predicted_index: int
    predicted_label: str
    confidence: float
    conformal_set: Tuple[str, ...]
    is_ambiguous: bool
    logits: np.ndarray


class MultiTenantSkillFleet:
    """Shared frozen projector + base trunk + compact contiguous per-tenant arenas."""

    BASE_HASH_SIZE = 4096

    def __init__(
        self,
        projector: Any,
        classes: Sequence[str],
        W_base: np.ndarray,
        b_base: np.ndarray,
        U_shared: Optional[np.ndarray] = None,
        base_tok_df: Optional[np.ndarray] = None,
        q_init: Optional[np.ndarray] = None,
        num_tenants: int = 500,
        r_sub: int = 64,
        m_sparse: int = 48,
        m_exemplars: int = 16,
        lam_rls: float = 1.5,
        alpha: float = 0.10,
        *,
        token_cache_limit: int = 4096,
        is_boolean: Optional[bool] = None,
        bool_threshold: float = 0.5,
        recency_weighted: bool = False,
        zero_unit0: Optional[bool] = None,
        _raw_source_head: Optional[Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray]]] = None,
    ) -> None:
        self.projector = projector
        self.classes = [str(c) for c in classes]
        if not self.classes:
            raise ValueError("classes must be non-empty")
        self.num_classes = len(self.classes)
        self.c2i = {c: i for i, c in enumerate(self.classes)}
        if is_boolean is None:
            self.is_boolean = tuple(self.classes) == ("False", "True")
        elif isinstance(is_boolean, (bool, np.bool_)):
            self.is_boolean = bool(is_boolean)
        else:
            raise ValueError(f"is_boolean must be a bool or None, got {is_boolean!r}")
        if self.is_boolean and self.num_classes != 2:
            raise ValueError(f"is_boolean=True requires 2 classes, got {self.num_classes}")

        if isinstance(bool_threshold, (bool, np.bool_)) or not isinstance(
            bool_threshold, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"bool_threshold must be finite in [0.0, 1.0], got {bool_threshold!r}")
        bt_val = float(bool_threshold)
        if not math.isfinite(bt_val) or not (0.0 <= bt_val <= 1.0):
            raise ValueError(f"bool_threshold must be finite in [0.0, 1.0], got {bool_threshold!r}")
        if not self.is_boolean and bt_val != 0.5:
            raise ValueError("Non-default bool_threshold requires is_boolean=True")
        self.bool_threshold = bt_val

        if not isinstance(recency_weighted, (bool, np.bool_)):
            raise ValueError(f"recency_weighted must be a bool, got {recency_weighted!r}")
        self.recency_weighted = bool(recency_weighted)
        if zero_unit0 is not None and not isinstance(zero_unit0, (bool, np.bool_)):
            raise ValueError(f"zero_unit0 must be a bool or None, got {zero_unit0!r}")
        if isinstance(self.projector, (TfidfProjector, PragmaticCompositeProjector)):
            if zero_unit0 is True:
                raise ValueError(
                    "zero_unit0=True is incompatible with TfidfProjector and PragmaticCompositeProjector"
                )
            self._zero_unit0 = False
        else:
            self._zero_unit0 = True if zero_unit0 is None else bool(zero_unit0)
        self._accepts_recency = _accepts_recency_weighted(getattr(self.projector, "project", None))
        self._batch_accepts_recency = _accepts_recency_weighted(
            getattr(self.projector, "project_batch", None)
        )
        with np.errstate(over="ignore", invalid="ignore"):
            self.W_base = np.asarray(W_base, dtype=np.float32)
            self.b_base = np.asarray(b_base, dtype=np.float32).reshape(-1)
        if self.W_base.ndim != 2 or self.W_base.shape[0] != self.num_classes or self.W_base.shape[1] < 1:
            raise ValueError(f"W_base shape {self.W_base.shape} must be ({self.num_classes}, >=1)")
        if self.b_base.shape[0] != self.num_classes:
            raise ValueError(f"b_base length {self.b_base.shape[0]} != num_classes {self.num_classes}")
        if not bool(np.all(np.isfinite(self.W_base))) or float(np.max(np.abs(self.W_base))) > 1e12:
            raise ValueError("W_base must contain finite values within [-1e12, 1e12]")
        if not bool(np.all(np.isfinite(self.b_base))) or float(np.max(np.abs(self.b_base))) > 1e12:
            raise ValueError("b_base must contain finite values within [-1e12, 1e12]")
        if _raw_source_head is not None:
            self._raw_source_head: Optional[Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray]]] = (
                self._validate_and_freeze_raw_source_head(
                    _raw_source_head,
                    num_classes=self.num_classes,
                    dim=int(self.W_base.shape[1]),
                    is_boolean=self.is_boolean,
                    W_base=self.W_base,
                    b_base=self.b_base,
                )
            )
        else:
            self._raw_source_head = None

        self.num_tenants = int(num_tenants)
        if self.num_tenants <= 0:
            raise ValueError(f"num_tenants must be positive, got {self.num_tenants}")
        # Keep r_sub >= K and a multiple of 16 to avoid low-rank collapse across block-diagonal RLS.
        self.r_sub = _round_up_16(max(int(r_sub), self.num_classes))
        self.blk_sz = 16
        self.n_blk = self.r_sub // self.blk_sz

        with np.errstate(over="ignore", invalid="ignore"):
            self.U_shared = (
                _prepare_shared_subspace(self.W_base, self.r_sub)
                if U_shared is None
                else _adapt_subspace_width(U_shared, self.W_base, self.r_sub)
            )
        if (
            self.U_shared.size > 0
            and (not bool(np.all(np.isfinite(self.U_shared))) or float(np.max(np.abs(self.U_shared))) > 1e12)
        ):
            raise ValueError("U_shared must contain finite values within [-1e12, 1e12]")
        if base_tok_df is None:
            self.base_tok_df = np.zeros(self.BASE_HASH_SIZE, dtype=np.uint16)
        else:
            df_arr = np.asarray(base_tok_df, dtype=np.uint16).reshape(-1)
            if df_arr.shape[0] != self.BASE_HASH_SIZE:
                raise ValueError(f"base_tok_df length {df_arr.shape[0]} != {self.BASE_HASH_SIZE}")
            self.base_tok_df = df_arr

        if isinstance(m_sparse, bool) or not isinstance(m_sparse, (int, np.integer)):
            raise ValueError(
                f"m_sparse must be an integer in [0, {MAX_FLEET_SPARSE_CAP}], got {m_sparse!r}"
            )
        m_sparse_i = int(m_sparse)
        if not (0 <= m_sparse_i <= MAX_FLEET_SPARSE_CAP):
            raise ValueError(f"m_sparse must be in [0, {MAX_FLEET_SPARSE_CAP}], got {m_sparse_i}")
        self.m_sparse = m_sparse_i

        if isinstance(m_exemplars, bool) or not isinstance(m_exemplars, (int, np.integer)):
            raise ValueError(
                f"m_exemplars must be an integer in [0, {MAX_FLEET_EXEMPLAR_CAP}], got {m_exemplars!r}"
            )
        m_exemplars_i = int(m_exemplars)
        if not (0 <= m_exemplars_i <= MAX_FLEET_EXEMPLAR_CAP):
            raise ValueError(
                f"m_exemplars must be in [0, {MAX_FLEET_EXEMPLAR_CAP}], got {m_exemplars_i}"
            )
        self.m_exemplars = m_exemplars_i
        if isinstance(lam_rls, bool) or not isinstance(lam_rls, (int, float, np.integer, np.floating)):
            raise ValueError(f"lam_rls must be finite in [1e-12, 1e12], got {lam_rls!r}")
        self.lam_rls = float(lam_rls)
        if not math.isfinite(self.lam_rls) or not (1e-12 <= self.lam_rls <= 1e12):
            raise ValueError(f"lam_rls must be finite in [1e-12, 1e12], got {self.lam_rls!r}")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.integer, np.floating)):
            raise ValueError(f"alpha must be finite in (0.0, 1.0), got {alpha!r}")
        self.alpha = float(alpha)
        if not math.isfinite(self.alpha) or not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be finite in (0.0, 1.0), got {self.alpha!r}")
        self.token_cache_limit = max(0, int(token_cache_limit))
        self._tok_cache: OrderedDict[str, Tuple[int, ...]] = OrderedDict()

        N, K, R = self.num_tenants, self.num_classes, self.r_sub
        self.dW_q = np.zeros((N, K, R), dtype=np.int8)
        self.dW_scale = np.zeros((N, K, 1), dtype=np.float16)
        self.P_blk = np.broadcast_to(
            (np.eye(self.blk_sz, dtype=np.float32) / self.lam_rls),
            (N, self.n_blk, self.blk_sz, self.blk_sz),
        ).copy()

        self.sp_hash = np.zeros((N, self.m_sparse), dtype=np.uint16)
        self.sp_cls = np.zeros((N, self.m_sparse), dtype=np.uint16)
        self.sp_w = np.zeros((N, self.m_sparse), dtype=np.float16)
        self.sp_len = np.zeros(N, dtype=np.uint8)

        self.ex_vecs_q = np.zeros((N, self.m_exemplars, R), dtype=np.int8)
        self.ex_scales = np.zeros((N, self.m_exemplars), dtype=np.float16)
        self.ex_cls = np.full((N, self.m_exemplars), -1, dtype=np.int16)
        self.ex_len = np.zeros(N, dtype=np.uint8)

        q0_f64 = np.full(K, 0.5, dtype=np.float64) if q_init is None else np.asarray(q_init, dtype=np.float64).reshape(-1)
        if q0_f64.shape[0] != K:
            raise ValueError(f"q_init length {q0_f64.shape[0]} != num_classes {K}")
        if not bool(np.all(np.isfinite(q0_f64))):
            raise ValueError("q_init must contain only finite values")
        q_init_f64 = np.clip(q0_f64, 1e-4, 1.0)
        self.q_init = q_init_f64.astype(np.float32)
        u_init = np.where(
            self.q_init >= 1.0 - math.exp(-(16.0 - 1e-6)),
            16.0,
            -np.log(np.clip(1.0 - self.q_init.astype(np.float64), 1e-12, 1.0)),
        )
        self.q_c = np.broadcast_to(_ceil_float16_unit(q_init_f64), (N, K)).copy()
        self.u_c = np.broadcast_to(u_init.astype(np.float16), (N, K)).copy()
        self.alpha_c = np.full((N, K), self.alpha, dtype=np.float64)
        self.err_ewma = np.full((N, K), self.alpha, dtype=np.float16)

        self._lock = threading.RLock()
        self._active_tid = 0
        self._active_views: Tuple[Any, ...] = self._bind_views(0)

    def _validate_tid(self, tid: int) -> int:
        if isinstance(tid, bool) or not isinstance(tid, (int, np.integer)):
            raise ValueError(f"tenant_id must be an integer, got {tid!r}")
        idx = int(tid)
        if not 0 <= idx < self.num_tenants:
            raise IndexError(f"tenant_id {idx} out of range [0, {self.num_tenants})")
        return idx

    def _resolve_class_idx(self, label: Union[int, str]) -> int:
        if isinstance(label, str):
            if label not in self.c2i:
                raise ValueError(f"Unknown class label {label!r}")
            return self.c2i[label]
        if isinstance(label, bool) or not isinstance(label, (int, np.integer)):
            raise ValueError(f"Class label must be a string or integer index, got {label!r}")
        idx = int(label)
        if not 0 <= idx < self.num_classes:
            raise ValueError(f"Class index {idx} out of range [0, {self.num_classes})")
        return idx

    def _encode_one(self, text: str) -> np.ndarray:
        return _encode_projector_one(
            self.projector,
            text,
            int(self.W_base.shape[1]),
            recency_weighted=self.recency_weighted,
            zero_unit0=self._zero_unit0,
            accepts_recency=self._accepts_recency,
        )

    def _project_batch(self, texts: Sequence[str]) -> np.ndarray:
        return _encode_projector_batch(
            self.projector,
            texts,
            int(self.W_base.shape[1]),
            recency_weighted=self.recency_weighted,
            zero_unit0=self._zero_unit0,
            accepts_recency=self._accepts_recency,
            batch_accepts_recency=self._batch_accepts_recency,
        )

    def _bind_views(self, tid: int) -> Tuple[Any, ...]:
        return (
            self.dW_q[tid],
            self.dW_scale[tid],
            self.P_blk[tid],
            self.sp_hash[tid],
            self.sp_cls[tid],
            self.sp_w[tid],
            int(self.sp_len[tid]),
            self.ex_vecs_q[tid],
            self.ex_scales[tid],
            self.ex_cls[tid],
            int(self.ex_len[tid]),
            self.q_c[tid],
        )

    def _snapshot_views(self, tid: int) -> Tuple[Any, ...]:
        """Take an atomic local copy of tenant `tid`'s arena slices under `self._lock`."""
        with self._lock:
            n_sp = int(self.sp_len[tid])
            n_ex = int(self.ex_len[tid])
            return (
                self.dW_q[tid].copy(),
                self.dW_scale[tid].copy(),
                self.sp_hash[tid, :n_sp].copy(),
                self.sp_cls[tid, :n_sp].copy(),
                self.sp_w[tid, :n_sp].copy(),
                n_sp,
                self.ex_vecs_q[tid, :n_ex].copy(),
                self.ex_scales[tid, :n_ex].copy(),
                self.ex_cls[tid, :n_ex].copy(),
                n_ex,
                self.q_c[tid].copy(),
            )

    def switch_tenant(self, tid: int) -> None:
        """Bind arena slices for `tid` without copying."""
        with self._lock:
            idx = self._validate_tid(tid)
            self._active_tid = idx
            self._active_views = self._bind_views(idx)

    def arena_nbytes(self) -> int:
        """Return total byte footprint of per-tenant arenas across the fleet."""
        arrays = (
            self.dW_q,
            self.dW_scale,
            self.P_blk,
            self.sp_hash,
            self.sp_cls,
            self.sp_w,
            self.sp_len,
            self.ex_vecs_q,
            self.ex_scales,
            self.ex_cls,
            self.ex_len,
            self.q_c,
            self.u_c,
            self.alpha_c,
            self.err_ewma,
        )
        return sum(int(a.nbytes) for a in arrays)

    @staticmethod
    def _validate_and_freeze_raw_source_head(
        raw_head: Any,
        *,
        num_classes: int,
        dim: int,
        is_boolean: bool,
        W_base: np.ndarray,
        b_base: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray]]:
        if not isinstance(raw_head, (tuple, list)) or len(raw_head) != 4:
            raise ValueError("_raw_source_head must be a 4-tuple (W_raw, b_raw, cal_temp, cal_adj)")
        raw_w_in, raw_b_in, raw_cal_t, raw_cal_adj = raw_head
        if isinstance(raw_cal_t, (bool, np.bool_)) or not isinstance(
            raw_cal_t, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"_raw_source_head cal_temp must be positive and finite, got {raw_cal_t!r}")
        cal_t = float(raw_cal_t)
        if not math.isfinite(cal_t) or cal_t <= 0.0:
            raise ValueError(f"_raw_source_head cal_temp must be positive and finite, got {raw_cal_t!r}")

        raw_w_arr = np.asarray(raw_w_in)
        raw_b_arr = np.asarray(raw_b_in)
        allowed_raw_dtypes = (np.dtype(np.float32), np.dtype(np.float64))
        if raw_w_arr.dtype not in allowed_raw_dtypes or raw_b_arr.dtype not in allowed_raw_dtypes:
            raise ValueError(
                f"_raw_source_head weights/biases must be float32 or float64, got {raw_w_arr.dtype}/{raw_b_arr.dtype}"
            )
        exp_rows = 1 if is_boolean else int(num_classes)
        if raw_w_arr.ndim != 2 or raw_w_arr.shape != (exp_rows, int(dim)):
            raise ValueError(
                f"_raw_source_head W shape {raw_w_arr.shape} != expected ({exp_rows}, {int(dim)})"
            )
        if raw_b_arr.ndim != 1 or raw_b_arr.shape != (exp_rows,):
            raise ValueError(
                f"_raw_source_head b shape {raw_b_arr.shape} != expected ({exp_rows},)"
            )
        if not (bool(np.all(np.isfinite(raw_w_arr))) and bool(np.all(np.isfinite(raw_b_arr)))):
            raise ValueError("_raw_source_head contains non-finite weights or biases")

        w_dtype = (
            np.float64
            if (raw_w_arr.dtype == np.dtype(np.float64) or raw_b_arr.dtype == np.dtype(np.float64))
            else np.float32
        )
        w_auth = np.array(raw_w_arr, dtype=w_dtype, copy=True)
        b_auth = np.array(raw_b_arr, dtype=w_dtype, copy=True)
        w_auth.setflags(write=False)
        b_auth.setflags(write=False)

        adj_frozen: Optional[np.ndarray] = None
        if raw_cal_adj is not None:
            raw_adj_arr = np.asarray(raw_cal_adj)
            if raw_adj_arr.dtype == np.dtype(np.bool_) or raw_adj_arr.dtype.kind not in ("i", "u", "f"):
                raise ValueError(
                    f"_raw_source_head cal_adj must have numeric dtype, got {raw_adj_arr.dtype}"
                )
            if raw_adj_arr.ndim != 1 or raw_adj_arr.shape != (int(num_classes),):
                raise ValueError(
                    f"_raw_source_head cal_adj shape {raw_adj_arr.shape} != expected ({int(num_classes)},)"
                )
            adj_frozen = np.array(raw_adj_arr, dtype=np.float64, copy=True)
            if not bool(np.all(np.isfinite(adj_frozen))):
                raise ValueError("_raw_source_head cal_adj contains non-finite values")
            adj_frozen.setflags(write=False)

        raw_w64 = np.asarray(w_auth, dtype=np.float64)
        raw_b64 = np.asarray(b_auth, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            if is_boolean:
                delta_adj = float(adj_frozen[1] - adj_frozen[0]) if adj_frozen is not None else 0.0
                w_bool_fold = raw_w64[0] / cal_t
                b_bool_fold = (float(raw_b64[0]) + 0.25 * delta_adj) / cal_t
                w_fold = np.vstack([-0.5 * w_bool_fold, 0.5 * w_bool_fold]).astype(np.float32)
                b_fold = np.array([-0.5 * b_bool_fold, 0.5 * b_bool_fold], dtype=np.float32)
            else:
                adj_vec = adj_frozen if adj_frozen is not None else np.zeros(int(num_classes), dtype=np.float64)
                w_fold = (raw_w64 / cal_t).astype(np.float32)
                b_fold = ((raw_b64 + 0.25 * adj_vec) / cal_t).astype(np.float32)
        if not (bool(np.all(np.isfinite(w_fold))) and bool(np.all(np.isfinite(b_fold)))):
            raise ValueError("_raw_source_head overflowed float32 bounds after folding calibration")
        if float(np.max(np.abs(w_fold))) > 1e12 or float(np.max(np.abs(b_fold))) > 1e12:
            raise ValueError("_raw_source_head folded head exceeds [-1e12, 1e12] magnitude bound")
        if not (
            bool(np.allclose(w_fold, W_base, rtol=1e-5, atol=1e-5))
            and bool(np.allclose(b_fold, b_base, rtol=1e-5, atol=1e-5))
        ):
            raise ValueError("_raw_source_head does not match folded W_base / b_base")
        return w_auth, b_auth, cal_t, adj_frozen

    def base_nbytes(self) -> int:
        """Return byte footprint of shared trunk weights and subspace basis."""
        raw_bytes = 0
        if self._raw_source_head is not None:
            raw_bytes = (
                int(self._raw_source_head[0].nbytes)
                + int(self._raw_source_head[1].nbytes)
                + (0 if self._raw_source_head[3] is None else int(self._raw_source_head[3].nbytes))
            )
        return int(
            self.W_base.nbytes
            + self.b_base.nbytes
            + self.U_shared.nbytes
            + self.base_tok_df.nbytes
            + self.q_init.nbytes
            + raw_bytes
        )

    @classmethod
    def from_compiled_model(
        cls,
        model: Any,
        field_name: Optional[str] = None,
        *,
        base_texts: Sequence[str] = (),
        num_tenants: int = 500,
        r_sub: int = 64,
        m_sparse: int = 48,
        m_exemplars: int = 16,
        lam_rls: float = 1.5,
        alpha: float = 0.10,
        token_cache_limit: int = 4096,
    ) -> MultiTenantSkillFleet:
        """Create a fleet from an existing CompiledSystemOneModel or SystemOneEngine."""
        resolved = getattr(model, "model", model)
        schema = getattr(model, "schema", None) or resolved.schema
        if field_name is None:
            cand = [
                name for name, f_def in schema.fields.items() if isinstance(f_def, (ChoiceField, BooleanField))
            ]
            if not cand:
                raise ValueError("Model schema has no ChoiceField or BooleanField for fleet initialization")
            field_name = cand[0]
        if field_name not in schema.fields:
            raise ValueError(f"Unknown field {field_name!r}")

        f_def = schema.fields[field_name]
        if not isinstance(f_def, (ChoiceField, BooleanField)):
            raise ValueError(
                f"MultiTenantSkillFleet supports ChoiceField and BooleanField, got {type(f_def).__name__}"
            )
        field_heads = getattr(resolved, "_field_heads", None)
        compiled_heads = getattr(resolved, "heads", None)
        ch = compiled_heads.get(field_name) if isinstance(compiled_heads, Mapping) else None
        live_h = field_heads.get(field_name) if isinstance(field_heads, Mapping) else None
        head = live_h if live_h is not None else ch
        if head is None:
            raise ValueError(f"Model has no head for field {field_name!r}")
        if (
            not isinstance(f_def, BooleanField)
            and getattr(head, "hierarchical", False)
            and getattr(head, "coarse_weights", None) is not None
            and getattr(head, "coarse_biases", None) is not None
            and len(getattr(head, "cluster_assignments", ()) or ()) == np.asarray(head.weights).shape[0]
        ):
            raise ValueError(f"MultiTenantSkillFleet does not support hierarchical head {field_name!r}")

        calibrators = getattr(model, "calibrators", None)
        cal_obj = calibrators.get(field_name) if isinstance(calibrators, Mapping) else None
        if cal_obj is not None:
            temp = float(getattr(cal_obj, "temperature", 1.0))
            raw_adj = getattr(cal_obj, "logit_adjustments", None)
        else:
            src_meta = ch if ch is not None else head
            temp = float(getattr(src_meta, "temperature", 1.0) or 1.0)
            raw_adj = getattr(src_meta, "logit_adjustments", None)
        if not math.isfinite(temp) or temp <= 0.0:
            raise ValueError(f"Calibration temperature for {field_name!r} must be positive and finite, got {temp!r}")
        adj = None if raw_adj is None else np.asarray(raw_adj, dtype=np.float64).reshape(-1)
        if adj is not None and not bool(np.all(np.isfinite(adj))):
            raise ValueError(f"Calibration logit_adjustments for {field_name!r} contain non-finite values")

        raw_w_in = np.asarray(head.weights)
        raw_b_in = np.asarray(head.biases).reshape(-1)
        w_dtype = np.float64 if (raw_w_in.dtype == np.float64 or raw_b_in.dtype == np.float64) else np.float32
        raw_w_auth = np.array(raw_w_in, dtype=w_dtype, copy=True)
        raw_b_auth = np.array(raw_b_in, dtype=w_dtype, copy=True).reshape(-1)
        raw_w_auth.setflags(write=False)
        raw_b_auth.setflags(write=False)
        raw_w64 = np.asarray(raw_w_auth, dtype=np.float64)
        raw_b64 = np.asarray(raw_b_auth, dtype=np.float64).reshape(-1)
        if not (np.all(np.isfinite(raw_w64)) and np.all(np.isfinite(raw_b64))):
            raise ValueError(f"Field head {field_name!r} contains non-finite weights or biases")

        is_bool_head = bool(isinstance(f_def, BooleanField) or raw_w64.shape[0] == 1)
        bool_threshold = float(getattr(f_def, "threshold", 0.5)) if is_bool_head else 0.5
        if not math.isfinite(bool_threshold) or not (0.0 <= bool_threshold <= 1.0):
            raise ValueError(f"BooleanField threshold for {field_name!r} must be in [0.0, 1.0], got {bool_threshold!r}")
        with np.errstate(over="ignore", invalid="ignore"):
            if is_bool_head:
                classes = ["False", "True"]
                delta_adj = float(adj[1] - adj[0]) if (adj is not None and adj.size == 2) else 0.0
                w_diff_eff = raw_w64[0] / temp
                b_diff_eff = (float(raw_b64[0]) + 0.25 * delta_adj) / temp
                W_base = np.vstack([-0.5 * w_diff_eff, 0.5 * w_diff_eff]).astype(np.float32)
                b_base = np.array([-0.5 * b_diff_eff, 0.5 * b_diff_eff], dtype=np.float32)
            elif isinstance(f_def, ChoiceField):
                classes = [str(o) for o in f_def.options]
                adj_vec = (
                    adj
                    if (adj is not None and adj.size == len(classes))
                    else np.zeros(len(classes), dtype=np.float64)
                )
                W_base = (raw_w64 / temp).astype(np.float32)
                b_base = ((raw_b64 + 0.25 * adj_vec) / temp).astype(np.float32)
            else:
                raise ValueError(f"MultiTenantSkillFleet supports ChoiceField and BooleanField, got {type(f_def).__name__}")
        if not (bool(np.all(np.isfinite(W_base))) and bool(np.all(np.isfinite(b_base)))):
            raise ValueError(f"Calibrated base head {field_name!r} overflowed float32 bounds")

        base_tok_df = np.zeros(cls.BASE_HASH_SIZE, dtype=np.uint16)
        if base_texts:
            for txt in base_texts:
                for h in set(tokenize_hashes(txt, cache_limit=token_cache_limit)):
                    bucket = h % cls.BASE_HASH_SIZE
                    if base_tok_df[bucket] < 65535:
                        base_tok_df[bucket] += 1
        else:
            tfidf = getattr(resolved.projector, "_tfidf_projector", resolved.projector)
            vocab = getattr(tfidf, "_word_vocab", None)
            if isinstance(vocab, dict):
                for term in vocab:
                    for h in tokenize_hashes(term, cache_limit=0):
                        bucket = h % cls.BASE_HASH_SIZE
                        if base_tok_df[bucket] < 65535:
                            base_tok_df[bucket] += 1

        q_init = np.full(len(classes), 0.5, dtype=np.float32)
        cp_map = getattr(model, "conformal_predictors", None)
        cp = cp_map.get(field_name) if isinstance(cp_map, Mapping) else None
        score_src = cp if (cp is not None and (getattr(cp, "is_calibrated", False) or len(getattr(cp, "calibration_scores", ())) > 0)) else ch
        if score_src is not None:
            cls_scores = getattr(score_src, "class_calibration_scores", {}) or {}
            raw_q = getattr(score_src, "conformal_quantile", getattr(score_src, "quantile", None))
            cal_scores = getattr(score_src, "calibration_scores", ())
            has_fallback_q = False
            fallback_q = 0.0
            if raw_q is not None:
                fq = float(raw_q)
                if math.isfinite(fq) and (fq > 0.0 or (bool(getattr(score_src, "is_calibrated", False)) and fq >= 0.0)):
                    fallback_q = fq
                    has_fallback_q = True
            if (not has_fallback_q) and len(cal_scores) > 0:
                n_s = len(cal_scores)
                k = min(n_s, max(1, int(math.ceil((n_s + 1) * (1.0 - alpha))))) - 1
                fallback_q = float(sorted(cal_scores)[k])
                has_fallback_q = True
            for idx, c_name in enumerate(classes):
                c_vals = cls_scores.get(c_name)
                if c_vals is not None and len(c_vals) > 0:
                    n_c = len(c_vals)
                    k = min(n_c, max(1, int(math.ceil((n_c + 1) * (1.0 - alpha))))) - 1
                    q_init[idx] = float(sorted(c_vals)[k])
                elif has_fallback_q:
                    q_init[idx] = fallback_q

        return cls(
            projector=resolved.projector,
            classes=classes,
            W_base=W_base,
            b_base=b_base,
            base_tok_df=base_tok_df,
            q_init=q_init,
            num_tenants=num_tenants,
            r_sub=r_sub,
            m_sparse=m_sparse,
            m_exemplars=m_exemplars,
            lam_rls=lam_rls,
            alpha=alpha,
            token_cache_limit=token_cache_limit,
            is_boolean=is_bool_head,
            bool_threshold=bool_threshold,
            recency_weighted=bool(getattr(resolved, "recency_weighted", False)),
            zero_unit0=_model_zero_unit0(resolved, resolved.projector),
            _raw_source_head=(
                raw_w_auth,
                raw_b_auth,
                temp,
                None if (adj is None or adj.size != len(classes)) else adj.copy(),
            ),
        )

    def adapt_tenant_batch(
        self,
        tid: int,
        texts: Sequence[str],
        labels: Sequence[Union[int, str]],
        precomputed_x: Optional[np.ndarray] = None,
        epochs: int = 2,
    ) -> None:
        """Adapt tenant `tid` in-place from a batch of few-shot labeled overrides or jargon."""
        tid = self._validate_tid(tid)
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have equal length")
        if not texts:
            return
        y_indices = [self._resolve_class_idx(lbl) for lbl in labels]
        X = (
            np.asarray(precomputed_x, dtype=np.float32)
            if precomputed_x is not None
            else self._project_batch(texts)
        )
        if (
            X.ndim != 2
            or X.shape != (len(texts), self.W_base.shape[1])
            or not bool(np.all(np.isfinite(X)))
        ):
            raise ValueError(
                f"adapt_tenant_batch features must be finite with shape ({len(texts)}, {self.W_base.shape[1]})"
            )
        with self._lock:
            m_sp = int(self.m_sparse)
            m_ex = int(self.m_exemplars)
            if not (0 <= m_sp <= min(MAX_FLEET_SPARSE_CAP, self.sp_hash.shape[1])):
                raise ValueError(
                    f"m_sparse {m_sp} invalid for uint8/buffer capacity {self.sp_hash.shape[1]}"
                )
            if not (0 <= m_ex <= min(MAX_FLEET_EXEMPLAR_CAP, self.ex_vecs_q.shape[1])):
                raise ValueError(
                    f"m_exemplars {m_ex} invalid for uint8/buffer capacity {self.ex_vecs_q.shape[1]}"
                )
            dW = self.dW_q[tid].astype(np.float32) * self.dW_scale[tid].astype(np.float32)
            P_t = self.P_blk[tid].copy()

            sp_map: Dict[Tuple[int, int], float] = {}
            n_sp = int(self.sp_len[tid])
            for i in range(n_sp):
                sp_map[(int(self.sp_hash[tid, i]), int(self.sp_cls[tid, i]))] = float(self.sp_w[tid, i])

            ex_list_v: List[np.ndarray] = []
            ex_list_c: List[int] = []

            bsz = self.blk_sz
            n_blk = self.n_blk
            K = self.num_classes

            for ep in range(max(1, int(epochs))):
                for idx, (txt, y) in enumerate(zip(texts, y_indices)):
                    x = X[idx]
                    z = x @ self.U_shared
                    base_log64, base_ok = self._eval_base_and_check_raw(x)
                    if not base_ok:
                        raise ValueError("Non-finite or overflowing base head evaluation during adapt_tenant_batch")
                    base_log = base_log64.astype(np.float32)
                    if not bool(np.all(np.isfinite(base_log))):
                        raise ValueError("Base logits overflowed float32 during adapt_tenant_batch")
                    pred_sub = base_log + dW @ z
                    tgt = pred_sub.copy()
                    if pred_sub[y] < 0.80:
                        tgt[y] = 0.80
                    wrong_mask = (np.arange(K) != y) & (pred_sub > 0.10)
                    tgt[wrong_mask] = -0.25
                    err = tgt - pred_sub
                    if float(np.max(np.abs(err))) > 1e-4:
                        k_full = np.empty(self.r_sub, dtype=np.float32)
                        for b in range(n_blk):
                            s0, s1 = b * bsz, (b + 1) * bsz
                            zb = z[s0:s1]
                            Pb = P_t[b]
                            Pzb = Pb @ zb
                            denom = 1.0 + float(zb @ Pzb)
                            kb = Pzb / denom
                            if ep == 0:
                                Pb -= np.outer(kb, Pzb)
                            k_full[s0:s1] = kb
                        dW += np.outer(err, k_full)

                    if ep == 0:
                        th = set(
                            tokenize_hashes(
                                txt,
                                cache=self._tok_cache,
                                cache_limit=self.token_cache_limit,
                                cache_lock=self._lock,
                            )
                        )
                        novel = []
                        for h in th:
                            nw = 1.0 / (1.0 + 0.2 * float(self.base_tok_df[h % self.BASE_HASH_SIZE]))
                            if nw > 0.25:
                                novel.append((h, nw))
                        if novel and self.m_sparse > 0:
                            pred_c = int(np.argmax(base_log))
                            tot_w = sum(nw for _, nw in novel) + 1e-6
                            for h, nw in novel:
                                frac = nw / tot_w
                                sp_map[(h, y)] = sp_map.get((h, y), 0.0) + 0.90 * frac
                                if pred_c != y:
                                    sp_map[(h, pred_c)] = sp_map.get((h, pred_c), 0.0) - 0.50 * frac
                        z_norm = z / (float(np.linalg.norm(z)) + 1e-8)
                        ex_list_v.append(z_norm)
                        ex_list_c.append(int(y))

            with np.errstate(over="ignore", invalid="ignore"):
                max_abs = np.max(np.abs(dW), axis=1, keepdims=True)
                scale_q = np.maximum(max_abs / 127.0, 1e-6).astype(np.float16)
                dW_q_new = np.clip(np.round(dW / scale_q.astype(np.float32)), -127, 127).astype(np.int8)
            if (
                not bool(np.all(np.isfinite(P_t)))
                or (P_t.size > 0 and float(np.max(np.abs(P_t))) > 1e12)
                or not bool(np.all(np.isfinite(dW)))
                or not bool(np.all(np.isfinite(scale_q)))
            ):
                raise ValueError(
                    "adapt_tenant_batch produced non-finite or out-of-range tenant parameters"
                )

            staged_sp: Optional[Tuple[int, np.ndarray, np.ndarray, np.ndarray]] = None
            if m_sp > 0:
                top_sp = sorted(sp_map.items(), key=lambda kv: abs(kv[1]), reverse=True)[:m_sp]
                n_sp_new = len(top_sp)
                if n_sp_new > MAX_FLEET_SPARSE_CAP or n_sp_new > self.sp_hash.shape[1]:
                    raise ValueError(f"Sparse count {n_sp_new} exceeds capacity")
                sp_h_new = np.empty(n_sp_new, dtype=np.uint16)
                sp_c_new = np.empty(n_sp_new, dtype=np.uint16)
                sp_w_new = np.empty(n_sp_new, dtype=np.float16)
                with np.errstate(over="ignore", invalid="ignore"):
                    for i, ((h, ci), wv) in enumerate(top_sp):
                        sp_h_new[i] = h
                        sp_c_new[i] = ci
                        sp_w_new[i] = np.float16(wv)
                if not bool(np.all(np.isfinite(sp_w_new))):
                    raise ValueError("adapt_tenant_batch produced non-finite sparse weights")
                staged_sp = (n_sp_new, sp_h_new, sp_c_new, sp_w_new)

            staged_ex: Optional[Tuple[int, np.ndarray, np.ndarray, np.ndarray]] = None
            if ex_list_v and m_ex > 0:
                by_cls: Dict[int, List[np.ndarray]] = defaultdict(list)
                for v, ci in zip(ex_list_v, ex_list_c):
                    by_cls[ci].append(v)
                chosen_v: List[np.ndarray] = []
                chosen_c: List[int] = []
                for ci, vs in by_cls.items():
                    if len(chosen_v) >= m_ex:
                        break
                    cen = np.mean(vs, axis=0)
                    cen /= float(np.linalg.norm(cen)) + 1e-8
                    chosen_v.append(cen)
                    chosen_c.append(ci)
                shot_idx = 0
                while len(chosen_v) < m_ex:
                    added = False
                    for ci, vs in by_cls.items():
                        if shot_idx < len(vs) and len(chosen_v) < m_ex:
                            chosen_v.append(vs[shot_idx])
                            chosen_c.append(ci)
                            added = True
                    if not added:
                        break
                    shot_idx += 1

                n_ex_new = len(chosen_v)
                if n_ex_new > MAX_FLEET_EXEMPLAR_CAP or n_ex_new > self.ex_vecs_q.shape[1]:
                    raise ValueError(f"Exemplar count {n_ex_new} exceeds capacity")
                ex_s_new = np.empty(n_ex_new, dtype=np.float16)
                ex_vq_new = np.empty((n_ex_new, self.r_sub), dtype=np.int8)
                ex_c_new = np.empty(n_ex_new, dtype=np.int16)
                with np.errstate(over="ignore", invalid="ignore"):
                    for i, (vec, ci) in enumerate(zip(chosen_v, chosen_c)):
                        if not bool(np.all(np.isfinite(vec))):
                            raise ValueError("adapt_tenant_batch produced non-finite exemplar vector")
                        sc = max(float(np.max(np.abs(vec))) / 127.0, 1e-6)
                        ex_s_new[i] = np.float16(sc)
                        ex_vq_new[i] = np.clip(np.round(vec / sc), -127, 127).astype(np.int8)
                        ex_c_new[i] = ci
                if not bool(np.all(np.isfinite(ex_s_new))):
                    raise ValueError("adapt_tenant_batch produced non-finite exemplar scales")
                staged_ex = (n_ex_new, ex_s_new, ex_vq_new, ex_c_new)

            self.P_blk[tid] = P_t
            self.dW_scale[tid] = scale_q
            self.dW_q[tid] = dW_q_new
            if staged_sp is not None:
                n_sp_new, sp_h_new, sp_c_new, sp_w_new = staged_sp
                self.sp_len[tid] = n_sp_new
                if n_sp_new > 0:
                    self.sp_hash[tid, :n_sp_new] = sp_h_new
                    self.sp_cls[tid, :n_sp_new] = sp_c_new
                    self.sp_w[tid, :n_sp_new] = sp_w_new
            if staged_ex is not None:
                n_ex_new, ex_s_new, ex_vq_new, ex_c_new = staged_ex
                self.ex_len[tid] = n_ex_new
                if n_ex_new > 0:
                    self.ex_scales[tid, :n_ex_new] = ex_s_new
                    self.ex_vecs_q[tid, :n_ex_new] = ex_vq_new
                    self.ex_cls[tid, :n_ex_new] = ex_c_new

            if tid == self._active_tid:
                self._active_views = self._bind_views(tid)

    def _eval_base_and_check_raw(self, x: np.ndarray) -> Tuple[np.ndarray, bool]:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            x_f32 = np.asarray(x, dtype=np.float32).reshape(-1)
            if x_f32.shape != (self.W_base.shape[1],) or not bool(np.all(np.isfinite(x_f32))):
                return np.full(self.num_classes, np.nan, dtype=np.float64), False
            if self._raw_source_head is not None:
                raw_w_auth, raw_b_auth, cal_t, cal_adj = self._raw_source_head
                x_in = x_f32.astype(raw_w_auth.dtype, copy=False)
                raw_lin = x_in @ raw_w_auth.T + raw_b_auth
                z_col = raw_lin.astype(np.float64) / 0.25
                if not bool(np.all(np.isfinite(raw_lin))) or not bool(np.all(np.isfinite(z_col))):
                    return np.full(self.num_classes, np.nan, dtype=np.float64), False
                if self.is_boolean:
                    two = np.array([-0.5 * float(z_col[0]), 0.5 * float(z_col[0])], dtype=np.float64)
                    if cal_adj is not None and cal_adj.shape == (2,):
                        two = two + cal_adj
                    cal_scaled = two / cal_t
                else:
                    arr = z_col.reshape(-1)
                    if cal_adj is not None and cal_adj.shape == (arr.shape[0],):
                        arr = arr + cal_adj
                    cal_scaled = arr / cal_t
                if not bool(np.all(np.isfinite(cal_scaled))):
                    return np.full(self.num_classes, np.nan, dtype=np.float64), False
                return 0.25 * cal_scaled, True
            base_log = (self.W_base @ x_f32 + self.b_base).astype(np.float64)
            if not bool(np.all(np.isfinite(base_log))):
                return np.full(self.num_classes, np.nan, dtype=np.float64), False
            return base_log, True

    def compute_component_logits(
        self, tid: int, text: str, x: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Decompose base, subspace RLS, sparse lexical, and exemplar logits for tenant `tid`."""
        tid = self._validate_tid(tid)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            if x is None:
                x_in = self._encode_one(text)
            else:
                x_in = np.asarray(x, dtype=np.float32).reshape(-1)
            dW_qt, dW_st, sp_ht, sp_ct, sp_wt, n_sp, ex_vt, ex_st, ex_ct, n_ex, _ = (
                self._snapshot_views(tid)
            )
            base_log64, base_ok = self._eval_base_and_check_raw(x_in)
            if not base_ok:
                nan_arr = np.full(self.num_classes, np.nan, dtype=np.float32)
                return {
                    "base": nan_arr.copy(),
                    "lowrank_rls": nan_arr.copy(),
                    "sparse_only": nan_arr.copy(),
                    "exemplar_only": nan_arr.copy(),
                    "hybrid_full": nan_arr.copy(),
                }
            z = x_in @ self.U_shared
            sub_log = (dW_qt.astype(np.float32) @ z) * dW_st.reshape(-1).astype(np.float32)

            sp_log = np.zeros(self.num_classes, dtype=np.float32)
            if n_sp > 0:
                th = set(
                    tokenize_hashes(
                        text,
                        cache=self._tok_cache,
                        cache_limit=self.token_cache_limit,
                        cache_lock=self._lock,
                    )
                )
                for i in range(n_sp):
                    if int(sp_ht[i]) in th:
                        sp_log[int(sp_ct[i])] += float(sp_wt[i])

            ex_log = np.zeros(self.num_classes, dtype=np.float32)
            if n_ex > 0:
                z_norm = z / (float(np.linalg.norm(z)) + 1e-8)
                vecs = ex_vt[:n_ex].astype(np.float32) * ex_st[:n_ex, None].astype(np.float32)
                sims = vecs @ z_norm
                for i in range(n_ex):
                    ms = float(sims[i])
                    if ms > 0.62:
                        ci = int(ex_ct[i])
                        bonus = 0.70 * ((ms - 0.62) / 0.38)
                        if bonus > ex_log[ci]:
                            ex_log[ci] = bonus

            def _checked_f32(arr64: np.ndarray) -> np.ndarray:
                out = np.asarray(arr64, dtype=np.float32)
                if not bool(np.all(np.isfinite(arr64))) or not bool(np.all(np.isfinite(out))):
                    return np.full(self.num_classes, np.nan, dtype=np.float32)
                return out

            return {
                "base": _checked_f32(base_log64),
                "lowrank_rls": _checked_f32(base_log64 + sub_log),
                "sparse_only": _checked_f32(base_log64 + sp_log),
                "exemplar_only": _checked_f32(base_log64 + ex_log),
                "hybrid_full": _checked_f32(base_log64 + sub_log + sp_log + ex_log),
            }

    def _eval_live(self, tid: int, text: str) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        tid = self._validate_tid(tid)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            x = self._encode_one(text)
            dW_qt, dW_st, sp_ht, sp_ct, sp_wt, n_sp, ex_vt, ex_st, ex_ct, n_ex, q_ct = (
                self._snapshot_views(tid)
            )
            base_log64, base_ok = self._eval_base_and_check_raw(x)
            if not base_ok:
                return (
                    0,
                    np.full(self.num_classes, np.nan, dtype=np.float32),
                    np.zeros(self.num_classes, dtype=bool),
                    np.full(self.num_classes, np.nan, dtype=np.float64),
                )
            z = x @ self.U_shared
            sub_log = (dW_qt.astype(np.float32) @ z) * dW_st.reshape(-1).astype(np.float32)
            logits64 = base_log64 + sub_log.astype(np.float64)
            if n_sp > 0:
                th = set(
                    tokenize_hashes(
                        text,
                        cache=self._tok_cache,
                        cache_limit=self.token_cache_limit,
                        cache_lock=self._lock,
                    )
                )
                for i in range(n_sp):
                    if int(sp_ht[i]) in th:
                        logits64[int(sp_ct[i])] += float(sp_wt[i])
            if n_ex > 0:
                z_norm = z / (float(np.linalg.norm(z)) + 1e-8)
                vecs = ex_vt[:n_ex].astype(np.float32) * ex_st[:n_ex, None].astype(np.float32)
                sims = vecs @ z_norm
                ex_bonus: Dict[int, float] = {}
                for i in range(n_ex):
                    ms = float(sims[i])
                    if ms > 0.62:
                        ci = int(ex_ct[i])
                        b = 0.70 * ((ms - 0.62) / 0.38)
                        if b > ex_bonus.get(ci, 0.0):
                            ex_bonus[ci] = b
                for ci, b in ex_bonus.items():
                    logits64[ci] += b
            logits_f32 = logits64.astype(np.float32)
            if (
                not bool(np.all(np.isfinite(z)))
                or not bool(np.all(np.isfinite(logits64)))
                or not bool(np.all(np.isfinite(logits_f32)))
            ):
                return (
                    0,
                    np.full(self.num_classes, np.nan, dtype=np.float32),
                    np.zeros(self.num_classes, dtype=bool),
                    np.full(self.num_classes, np.nan, dtype=np.float64),
                )
            probs = _softmax_temp(logits64, temp=0.25)
            if not bool(np.all(np.isfinite(probs))):
                return (
                    0,
                    np.full(self.num_classes, np.nan, dtype=np.float32),
                    np.zeros(self.num_classes, dtype=bool),
                    np.full(self.num_classes, np.nan, dtype=np.float64),
                )
            scores = 1.0 - probs
            pred_set = scores <= (q_ct.astype(np.float64) + 1e-12)
            if self.is_boolean and len(probs) == 2:
                pred_idx = 1 if float(probs[1]) >= self.bool_threshold else 0
            else:
                pred_idx = int(np.argmax(logits64))
            return pred_idx, logits_f32, pred_set, probs

    def decide_live(self, tid: int, text: str) -> Tuple[int, np.ndarray, np.ndarray]:
        """Evaluate `text` for tenant `tid` and return `(pred_idx, logits, conformal_mask)`."""
        pred_idx, logits_f32, pred_set, _ = self._eval_live(tid, text)
        return pred_idx, logits_f32, pred_set

    def decide(self, tid: int, text: str) -> TenantDecisionResult:
        """Evaluate `text` for tenant `tid` and return structured `TenantDecisionResult`."""
        pred_idx, logits_f32, pred_set, probs = self._eval_live(tid, text)
        probs_ok = bool(np.all(np.isfinite(probs)))
        cset = tuple(self.classes[int(i)] for i in np.flatnonzero(pred_set)) if probs_ok else ()
        conf = float(probs[pred_idx]) if probs_ok else float("nan")
        return TenantDecisionResult(
            tenant_id=int(tid),
            predicted_index=pred_idx,
            predicted_label=self.classes[pred_idx],
            confidence=conf,
            conformal_set=cset,
            is_ambiguous=(not probs_ok) or (len(cset) != 1),
            logits=logits_f32,
        )

    @staticmethod
    def _validate_zip_directory(buf_bytes: bytes) -> None:
        """Validate ZIP EOCD and central-directory headers before zipfile allocates entry objects."""
        if len(buf_bytes) < 22 or buf_bytes[:4] != b"PK\x03\x04":
            raise ValueError("Invalid fleet snapshot zip header")
        eocd = buf_bytes.rfind(b"PK\x05\x06", max(0, len(buf_bytes) - 65557))
        if eocd < 0 or eocd + 22 > len(buf_bytes):
            raise ValueError("Fleet snapshot zip EOCD record missing or malformed")
        disk, directory_disk, entries, disk_count, directory_size, directory_offset, comment_size = (
            struct.unpack_from("<HHHHIIH", buf_bytes, eocd + 4)
        )
        if (
            disk != 0
            or directory_disk != 0
            or entries != disk_count
            or entries == 0
            or entries > MAX_FLEET_ARCHIVE_ENTRIES
            or directory_size == 0
            or directory_size > MAX_FLEET_DIRECTORY_BYTES
            or directory_offset > eocd
            or directory_offset + directory_size != eocd
            or eocd + 22 + comment_size != len(buf_bytes)
        ):
            raise ValueError(
                f"Fleet snapshot zip directory invalid or exceeds limits (entries={entries}, size={directory_size})"
            )
        pos = directory_offset
        cd_count = 0
        while pos < eocd:
            if pos + 46 > eocd or buf_bytes[pos : pos + 4] != b"PK\x01\x02":
                raise ValueError("Malformed central directory entry in fleet snapshot")
            fname_len, extra_len, fcomment_len = struct.unpack_from("<HHH", buf_bytes, pos + 28)
            pos += 46 + fname_len + extra_len + fcomment_len
            cd_count += 1
            if cd_count > MAX_FLEET_ARCHIVE_ENTRIES or pos > eocd:
                raise ValueError("Fleet snapshot central directory exceeds entry limit or bounds")
        if pos != eocd or cd_count != entries:
            raise ValueError("Fleet snapshot central directory count/offset mismatch")

    @staticmethod
    def _materialize_snapshot_bytes(data: Any) -> bytes:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise ValueError("Fleet snapshot data must be non-empty bytes")
        mv = memoryview(data)
        raw_len = mv.nbytes
        if raw_len == 0:
            raise ValueError("Fleet snapshot data must be non-empty bytes")
        if raw_len > MAX_FLEET_ARCHIVE_RAW_BYTES:
            raise ValueError(
                f"Fleet snapshot raw size {raw_len} exceeds {MAX_FLEET_ARCHIVE_RAW_BYTES} bytes"
            )
        return data if type(data) is bytes else mv.tobytes()

    @classmethod
    def _preflight_snapshot_bytes(
        cls,
        data: Union[bytes, bytearray, memoryview],
        *,
        projector: Any,
        max_runtime_bytes: int,
    ) -> Tuple[Tuple[int, int, int, int, int], Dict[str, Tuple[np.dtype, Tuple[int, ...]]]]:
        """Validate fleet archive bounds, NPY headers, array dtypes/shapes, and memory budget."""
        import zipfile

        if isinstance(max_runtime_bytes, bool) or int(max_runtime_bytes) <= 0:
            raise ValueError(f"max_runtime_bytes must be positive, got {max_runtime_bytes}")

        buf_bytes = cls._materialize_snapshot_bytes(data)
        cls._validate_zip_directory(buf_bytes)

        try:
            zf_raw = zipfile.ZipFile(io.BytesIO(buf_bytes), "r")
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Invalid fleet snapshot zip archive: {exc}") from exc

        total_expanded = 0
        seen_names: set[str] = set()
        headers: Dict[str, Tuple[np.dtype, Tuple[int, ...]]] = {}
        with zf_raw:
            infos = zf_raw.infolist()
            if len(infos) > MAX_FLEET_ARCHIVE_ENTRIES:
                raise ValueError(f"Too many archive entries in fleet snapshot: {len(infos)}")
            for info in infos:
                if info.is_dir():
                    raise ValueError(f"Unexpected directory entry in fleet snapshot: {info.filename!r}")
                if info.flag_bits & 0x1:
                    raise ValueError(f"Encrypted fleet snapshot entry is forbidden: {info.filename!r}")
                if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise ValueError(
                        f"Unsupported zip compression method {info.compress_type} for {info.filename!r}"
                    )
                if not info.filename.endswith(".npy") or "/" in info.filename or "\\" in info.filename:
                    raise ValueError(f"Invalid member name in fleet snapshot: {info.filename!r}")
                key = info.filename[:-4]
                if key not in _FLEET_ALLOWED_ARCHIVE_MEMBERS:
                    raise ValueError(f"Unexpected member {key!r} in fleet snapshot")
                if key in seen_names:
                    raise ValueError(f"Duplicate member {key!r} in fleet snapshot")
                seen_names.add(key)

                if info.file_size > MAX_FLEET_ARCHIVE_EXPANDED_BYTES:
                    raise ValueError(
                        f"Fleet snapshot member {key!r} expanded size {info.file_size} exceeds limit"
                    )
                total_expanded += int(info.file_size)
                if total_expanded > MAX_FLEET_ARCHIVE_EXPANDED_BYTES:
                    raise ValueError(
                        f"Fleet snapshot expanded size {total_expanded} exceeds {MAX_FLEET_ARCHIVE_EXPANDED_BYTES}"
                    )
                with zf_raw.open(info, "r") as member_stream:
                    headers[key] = _read_npy_header_from_stream(member_stream, info.file_size)

        missing = _FLEET_REQUIRED_ARRAYS - seen_names
        if missing:
            raise ValueError(f"Fleet snapshot missing required array(s): {sorted(missing)}")

        dW_q_dtype, dW_q_shape = headers["dW_q"]
        if len(dW_q_shape) != 3 or dW_q_dtype != np.dtype(np.int8):
            raise ValueError(f"Invalid dW_q spec: dtype={dW_q_dtype}, shape={dW_q_shape}")
        T, K, R = dW_q_shape
        if not (1 <= T <= MAX_FLEET_TENANTS):
            raise ValueError(f"num_tenants {T} outside [1, {MAX_FLEET_TENANTS}]")
        if not (1 <= K <= 65535):
            raise ValueError(f"num_classes {K} outside [1, 65535]")
        if not (16 <= R <= 4096) or (R % 16) != 0 or R < K:
            raise ValueError(f"Invalid subspace dimension r_sub={R} for K={K}")

        W_dtype, W_shape = headers["W_base"]
        float_dtypes = (np.dtype(np.float16), np.dtype(np.float32))
        if W_dtype not in float_dtypes or len(W_shape) != 2 or W_shape[0] != K or W_shape[1] < 1:
            raise ValueError(f"Invalid W_base spec: dtype={W_dtype}, shape={W_shape}, expected ({K}, D)")
        D = W_shape[1]
        proj_dim = getattr(projector, "dimension", None)
        if proj_dim is not None and int(proj_dim) != D:
            raise ValueError(f"Projector dimension {proj_dim} != snapshot feature dimension {D}")

        sp_dtype, sp_shape = headers["sp_hash"]
        if (
            sp_dtype != np.dtype(np.uint16)
            or len(sp_shape) != 2
            or sp_shape[0] != T
            or sp_shape[1] > MAX_FLEET_SPARSE_CAP
        ):
            raise ValueError(
                f"Invalid sp_hash spec: dtype={sp_dtype}, shape={sp_shape}, expected ({T}, <={MAX_FLEET_SPARSE_CAP})"
            )
        M_sp = sp_shape[1]

        ex_dtype, ex_shape = headers["ex_vecs_q"]
        if (
            ex_dtype != np.dtype(np.int8)
            or len(ex_shape) != 3
            or ex_shape[0] != T
            or ex_shape[1] > MAX_FLEET_EXEMPLAR_CAP
            or ex_shape[2] != R
        ):
            raise ValueError(
                f"Invalid ex_vecs_q spec: dtype={ex_dtype}, shape={ex_shape}, expected ({T}, <={MAX_FLEET_EXEMPLAR_CAP}, {R})"
            )
        M_ex = ex_shape[1]

        alpha_dtypes = (np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64))
        expected_specs: Dict[str, Tuple[Tuple[np.dtype, ...], Tuple[int, ...]]] = {
            "b_base": (float_dtypes, (K,)),
            "U_shared": (float_dtypes, (D, R)),
            "base_tok_df": ((np.dtype(np.uint16),), (cls.BASE_HASH_SIZE,)),
            "q_init": (float_dtypes, (K,)),
            "dW_scale": (float_dtypes, (T, K, 1)),
            "P_blk": (float_dtypes, (T, R // 16, 16, 16)),
            "sp_cls": ((np.dtype(np.uint16),), (T, M_sp)),
            "sp_w": (float_dtypes, (T, M_sp)),
            "sp_len": ((np.dtype(np.uint8),), (T,)),
            "ex_scales": (float_dtypes, (T, M_ex)),
            "ex_cls": ((np.dtype(np.int16),), (T, M_ex)),
            "ex_len": ((np.dtype(np.uint8),), (T,)),
            "q_c": (float_dtypes, (T, K)),
            "u_c": (float_dtypes, (T, K)),
            "alpha_c": (alpha_dtypes, (T, K)),
            "err_ewma": (float_dtypes, (T, K)),
        }
        for arr_name, (exp_dtypes, exp_shape) in expected_specs.items():
            actual_dtype, actual_shape = headers[arr_name]
            if actual_dtype not in exp_dtypes:
                raise ValueError(
                    f"Fleet snapshot array {arr_name!r} dtype {actual_dtype} not in {exp_dtypes}"
                )
            if actual_shape != exp_shape:
                raise ValueError(
                    f"Fleet snapshot array {arr_name!r} shape {actual_shape} != expected {exp_shape}"
                )

        if "meta_ints" in headers:
            mi_dtype, mi_shape = headers["meta_ints"]
            if mi_dtype != np.dtype(np.int64) or mi_shape not in ((4,), (6,), (7,)):
                raise ValueError(f"Invalid meta_ints spec: dtype={mi_dtype}, shape={mi_shape}")
        if "meta_floats" in headers:
            mf_dtype, mf_shape = headers["meta_floats"]
            if mf_dtype != np.dtype(np.float64) or mf_shape not in ((2,), (3,)):
                raise ValueError(f"Invalid meta_floats spec: dtype={mf_dtype}, shape={mf_shape}")
        if "classes_utf8" in headers:
            c_dtype, c_shape = headers["classes_utf8"]
            if c_dtype != np.dtype(np.uint8) or len(c_shape) != 1 or c_shape[0] > 1_048_576:
                raise ValueError(f"Invalid classes_utf8 spec: dtype={c_dtype}, shape={c_shape}")

        raw_req_keys = ("raw_W_auth", "raw_b_auth", "raw_cal_temp")
        raw_present = [k for k in raw_req_keys if k in headers]
        raw_runtime = 0
        if raw_present:
            if len(raw_present) != len(raw_req_keys):
                missing_raw = [k for k in raw_req_keys if k not in headers]
                raise ValueError(f"Incomplete raw source head members in snapshot: missing {missing_raw}")
            rw_dtype, rw_shape = headers["raw_W_auth"]
            rb_dtype, rb_shape = headers["raw_b_auth"]
            rt_dtype, rt_shape = headers["raw_cal_temp"]
            raw_head_dtypes = (np.dtype(np.float32), np.dtype(np.float64))
            if (
                rw_dtype not in raw_head_dtypes
                or len(rw_shape) != 2
                or rw_shape[1] != D
                or rw_shape[0] not in (1, K)
            ):
                raise ValueError(
                    f"Invalid raw_W_auth spec: dtype={rw_dtype}, shape={rw_shape}, expected ((1|{K}), {D})"
                )
            if rb_dtype not in raw_head_dtypes or rb_shape != (rw_shape[0],):
                raise ValueError(
                    f"Invalid raw_b_auth spec: dtype={rb_dtype}, shape={rb_shape}, expected ({rw_shape[0]},)"
                )
            if rt_dtype not in alpha_dtypes or rt_shape != (1,):
                raise ValueError(f"Invalid raw_cal_temp spec: dtype={rt_dtype}, shape={rt_shape}")
            raw_runtime += rw_dtype.itemsize * rw_shape[0] * D + rb_dtype.itemsize * rw_shape[0] + 8
            if "raw_cal_adj" in headers:
                ra_dtype, ra_shape = headers["raw_cal_adj"]
                if ra_dtype not in alpha_dtypes or ra_shape != (K,):
                    raise ValueError(
                        f"Invalid raw_cal_adj spec: dtype={ra_dtype}, shape={ra_shape}, expected ({K},)"
                    )
                raw_runtime += 8 * K
        elif "raw_cal_adj" in headers:
            raise ValueError("Snapshot contains raw_cal_adj without raw_W_auth/raw_b_auth/raw_cal_temp")

        # Runtime allocation footprint across float32 base/P_blk + float64 alpha_c + int8/float16/uint16 arenas.
        base_runtime = 4 * (K * D + K + D * R + K) + 2 * cls.BASE_HASH_SIZE
        arena_runtime = (
            T * K * R * 1
            + T * K * 2
            + T * (R // 16) * 256 * 4
            + T * M_sp * (2 + 2 + 2)
            + T * M_ex * (R * 1 + 2 + 2)
            + T * (1 + 1)
            + T * K * (2 * 3 + 8)
        )
        est_runtime = int(base_runtime + arena_runtime + raw_runtime)
        if est_runtime > int(max_runtime_bytes) or total_expanded > int(max_runtime_bytes) + 65536:
            raise ValueError(
                f"Fleet snapshot runtime memory {est_runtime} (expanded {total_expanded}) exceeds budget {int(max_runtime_bytes)} bytes"
            )
        return (T, K, R, M_sp, M_ex), headers

    def update_conformal(
        self,
        tid: int,
        y: Union[int, str],
        probs_or_logits: np.ndarray,
        *,
        is_logits: bool = False,
        eta_u: float = 0.30,
        jump_w: float = 0.12,
        gamma_aci: Optional[float] = None,
    ) -> float:
        """Perform an O(1) per-tenant, per-class Mondrian conformal quantile update."""
        tid = self._validate_tid(tid)
        yi = self._resolve_class_idx(y)
        if type(is_logits) is not bool:
            raise ValueError(f"is_logits must be a bool, got {is_logits!r}")
        if isinstance(eta_u, (bool, np.bool_)) or not isinstance(eta_u, (int, float, np.integer, np.floating)):
            raise ValueError(f"eta_u must be finite and positive, got {eta_u!r}")
        eta_val = float(eta_u)
        if not math.isfinite(eta_val) or eta_val <= 0.0:
            raise ValueError(f"eta_u must be finite and positive, got {eta_u!r}")
        if isinstance(jump_w, (bool, np.bool_)) or not isinstance(jump_w, (int, float, np.integer, np.floating)):
            raise ValueError(f"jump_w must be finite in [0.0, 1.0], got {jump_w!r}")
        jump_val = float(jump_w)
        if not math.isfinite(jump_val) or not (0.0 <= jump_val <= 1.0):
            raise ValueError(f"jump_w must be finite in [0.0, 1.0], got {jump_w!r}")
        if gamma_aci is not None:
            if isinstance(gamma_aci, (bool, np.bool_)) or not isinstance(gamma_aci, (int, float, np.integer, np.floating)):
                raise ValueError(f"gamma_aci must be finite and positive, got {gamma_aci!r}")
            g_aci = float(gamma_aci)
            if not math.isfinite(g_aci) or g_aci <= 0.0:
                raise ValueError(f"gamma_aci must be finite and positive, got {gamma_aci!r}")
        else:
            g_aci = max(0.03, 0.4 * self.alpha)

        def _materialize_real_sequence(obj: Any) -> Any:
            if isinstance(obj, (str, bytes, bytearray, Mapping)):
                raise ValueError("probs_or_logits must be a numeric sequence or array")
            if isinstance(obj, np.ndarray):
                if obj.dtype == np.bool_ or obj.dtype.kind not in ("i", "u", "f"):
                    raise ValueError(f"probs_or_logits must have real numeric dtype, got {obj.dtype}")
                return obj
            if hasattr(obj, "__iter__"):
                out_list: List[Any] = []
                for el in obj:
                    if isinstance(el, np.ndarray):
                        if el.dtype == np.bool_ or el.dtype.kind not in ("i", "u", "f"):
                            raise ValueError(f"probs_or_logits must have real numeric dtype, got {el.dtype}")
                        out_list.append(el)
                    elif hasattr(el, "__iter__") and not isinstance(el, (str, bytes, bytearray, Mapping)):
                        out_list.append(_materialize_real_sequence(el))
                    else:
                        if (
                            isinstance(el, (bool, np.bool_))
                            or getattr(getattr(el, "dtype", None), "kind", None) == "b"
                            or not isinstance(el, (int, float, np.integer, np.floating))
                        ):
                            raise ValueError(f"probs_or_logits entries must be real numbers, got {el!r}")
                        out_list.append(el)
                return out_list
            raise ValueError(f"probs_or_logits must be a numeric sequence or array, got {type(obj).__name__}")

        validated_input = _materialize_real_sequence(probs_or_logits)
        arr = np.asarray(validated_input, dtype=np.float64).reshape(-1)
        if arr.shape[0] != self.num_classes:
            raise ValueError(f"Expected length {self.num_classes}, got {arr.shape[0]}")
        if not bool(np.all(np.isfinite(arr))):
            raise ValueError("probs_or_logits must contain only finite values")
        if not is_logits and not bool(np.all((arr >= 0.0) & (arr <= 1.0))):
            raise ValueError("Probabilities must lie in [0.0, 1.0] when is_logits=False")
        probs = _softmax_temp(arr, temp=0.25) if is_logits else arr
        if not bool(np.all(np.isfinite(probs))):
            raise ValueError("Computed probabilities from probs_or_logits contain non-finite values")
        score_y = 1.0 - float(np.clip(probs[yi], 0.0, 1.0))

        with self._lock:
            q_cur = float(self.q_c[tid, yi])
            err_y = 1 if score_y > q_cur else 0

            s_clip = min(score_y, 1.0 - 1e-7)
            u_obs = min(16.0, -math.log(max(1.0 - s_clip, 1e-7)))
            alpha_min = max(0.015, 0.35 * self.alpha)
            alpha_max = min(0.40, 2.5 * self.alpha)
            fe = float(err_y)

            ewma_next = 0.85 * float(self.err_ewma[tid, yi]) + 0.15 * fe
            shift = ewma_next > 1.6 * self.alpha
            boost = 1.6 if shift else 1.0
            a_next = float(self.alpha_c[tid, yi]) + g_aci * boost * (self.alpha - fe)
            a_next = min(max(a_next, alpha_min), alpha_max)

            u_cur = float(self.u_c[tid, yi])
            u_next = u_cur + eta_val * boost * (fe - a_next)
            if err_y and shift:
                u_jump = (1.0 - jump_val) * u_cur + jump_val * min(u_obs, 15.5)
                if u_jump > u_next:
                    u_next = u_jump
            u_next = min(max(u_next, 0.01), 16.0)
            q_next = 1.0 if u_next >= 15.5 else (1.0 - math.exp(-u_next))
            q_f16 = np.float16(q_next)
            if float(q_f16) < q_next:
                q_f16 = np.nextafter(q_f16, np.float16(1.0))

            self.err_ewma[tid, yi] = np.float16(ewma_next)
            self.alpha_c[tid, yi] = float(a_next)
            self.u_c[tid, yi] = np.float16(u_next)
            self.q_c[tid, yi] = q_f16
            if tid == self._active_tid:
                self._active_views = self._bind_views(tid)
            return float(q_f16)

    def fleet_delta_to_bytes(self) -> bytes:
        """Serialize all per-tenant arenas without the shared projector via compressed NumPy stream."""
        def _pack_base_or_cov(arr: np.ndarray, name: str) -> np.ndarray:
            arr_f32 = np.asarray(arr, dtype=np.float32)
            if not bool(np.all(np.isfinite(arr_f32))):
                raise ValueError(f"Fleet array {name!r} contains non-finite values")
            if arr_f32.size > 0 and float(np.max(np.abs(arr_f32))) > 1e12:
                raise ValueError(f"Fleet array {name!r} exceeds safe float32 magnitude bound")
            return arr_f32

        with self._lock:
            for arena_name, arena_arr in (
                ("dW_scale", self.dW_scale),
                ("sp_w", self.sp_w),
                ("ex_scales", self.ex_scales),
                ("q_c", self.q_c),
                ("u_c", self.u_c),
                ("alpha_c", self.alpha_c),
                ("err_ewma", self.err_ewma),
            ):
                if not bool(np.all(np.isfinite(arena_arr))):
                    raise ValueError(f"Fleet array {arena_name!r} contains non-finite values")

            q_init_f32 = np.asarray(self.q_init, dtype=np.float32)
            if not bool(np.all(np.isfinite(q_init_f32))):
                raise ValueError("Fleet array 'q_init' contains non-finite values")

            w_out = _pack_base_or_cov(self.W_base, "W_base")
            b_out = _pack_base_or_cov(self.b_base, "b_base")
            u_out = _pack_base_or_cov(self.U_shared, "U_shared")
            p_out = _pack_base_or_cov(self.P_blk, "P_blk")

            buf = io.BytesIO()
            classes_bytes = np.frombuffer(json.dumps(self.classes).encode("utf-8"), dtype=np.uint8)
            meta_ints = np.array(
                [
                    self.num_tenants,
                    self.r_sub,
                    self.m_sparse,
                    self.m_exemplars,
                    int(self.is_boolean),
                    int(self.recency_weighted),
                    int(self._zero_unit0),
                ],
                dtype=np.int64,
            )
            meta_floats = np.array([self.lam_rls, self.alpha, self.bool_threshold], dtype=np.float64)
            snapshot_arrays: Dict[str, Any] = {
                "meta_ints": meta_ints,
                "meta_floats": meta_floats,
                "classes_utf8": classes_bytes,
                "W_base": w_out,
                "b_base": b_out,
                "U_shared": u_out,
                "base_tok_df": self.base_tok_df,
                "q_init": q_init_f32,
                "dW_q": self.dW_q,
                "dW_scale": self.dW_scale,
                "P_blk": p_out,
                "sp_hash": self.sp_hash,
                "sp_cls": self.sp_cls,
                "sp_w": self.sp_w,
                "sp_len": self.sp_len,
                "ex_vecs_q": self.ex_vecs_q,
                "ex_scales": self.ex_scales,
                "ex_cls": self.ex_cls,
                "ex_len": self.ex_len,
                "q_c": self.q_c,
                "u_c": self.u_c,
                "alpha_c": self.alpha_c,
                "err_ewma": self.err_ewma,
            }
            if self._raw_source_head is not None:
                raw_w_auth, raw_b_auth, cal_t, cal_adj = self._raw_source_head
                snapshot_arrays["raw_W_auth"] = raw_w_auth
                snapshot_arrays["raw_b_auth"] = raw_b_auth
                snapshot_arrays["raw_cal_temp"] = np.array([cal_t], dtype=np.float64)
                if cal_adj is not None:
                    snapshot_arrays["raw_cal_adj"] = cal_adj
            np.savez_compressed(buf, **snapshot_arrays)
            return buf.getvalue()

    @classmethod
    def fleet_delta_from_bytes(
        cls,
        data: Union[bytes, bytearray, memoryview],
        *,
        projector: Any,
        classes: Optional[Sequence[str]] = None,
        token_cache_limit: int = 4096,
        max_runtime_bytes: int = MAX_FLEET_RUNTIME_BYTES,
    ) -> MultiTenantSkillFleet:
        """Restore a MultiTenantSkillFleet from bytes produced by `fleet_delta_to_bytes()`."""
        raw_bytes = cls._materialize_snapshot_bytes(data)
        (T, K, R, M_sp, M_ex), _ = cls._preflight_snapshot_bytes(
            raw_bytes,
            projector=projector,
            max_runtime_bytes=max_runtime_bytes,
        )
        with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as zf:
            if classes is not None:
                resolved_classes = [str(c) for c in classes]
            elif "classes_utf8" in zf:
                try:
                    loaded_cls = json.loads(bytes(zf["classes_utf8"].tobytes()).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Snapshot classes_utf8 contains invalid UTF-8 JSON: {exc}") from exc
                if not isinstance(loaded_cls, list):
                    raise ValueError("Snapshot classes_utf8 must decode to a JSON list")
                resolved_classes = [str(c) for c in loaded_cls]
            else:
                raise ValueError("Snapshot lacks classes metadata; pass classes= explicitly")
            if len(resolved_classes) != K:
                raise ValueError(f"Snapshot classes length {len(resolved_classes)} != W_base rows {K}")

            is_boolean_meta: Optional[bool] = None
            recency_meta = False
            zero_unit0_meta: Optional[bool] = None
            if "meta_ints" in zf:
                m_int = zf["meta_ints"]
                num_tenants, r_sub, m_sparse, m_exemplars = (
                    int(m_int[0]),
                    int(m_int[1]),
                    int(m_int[2]),
                    int(m_int[3]),
                )
                if num_tenants != T:
                    raise ValueError(f"Snapshot meta_ints num_tenants {num_tenants} != array dim {T}")
                if r_sub < 1 or _round_up_16(max(r_sub, K)) != R:
                    raise ValueError(f"Snapshot meta_ints r_sub {r_sub} incompatible with array dim {R}")
                if m_sparse != M_sp or not (0 <= m_sparse <= MAX_FLEET_SPARSE_CAP):
                    raise ValueError(
                        f"Snapshot meta_ints m_sparse {m_sparse} incompatible with capacity [0, {min(M_sp, MAX_FLEET_SPARSE_CAP)}]"
                    )
                if m_exemplars != M_ex or not (0 <= m_exemplars <= MAX_FLEET_EXEMPLAR_CAP):
                    raise ValueError(
                        f"Snapshot meta_ints m_exemplars {m_exemplars} incompatible with capacity [0, {min(M_ex, MAX_FLEET_EXEMPLAR_CAP)}]"
                    )
                if len(m_int) >= 6:
                    raw_bool_flag = int(m_int[4])
                    raw_rec_flag = int(m_int[5])
                    if raw_bool_flag not in (0, 1) or raw_rec_flag not in (0, 1):
                        raise ValueError(
                            f"Invalid snapshot meta_ints boolean flags: ({raw_bool_flag}, {raw_rec_flag})"
                        )
                    is_boolean_meta = bool(raw_bool_flag)
                    if is_boolean_meta and K != 2:
                        raise ValueError(f"Snapshot is_boolean=True requires 2 classes, got {K}")
                    recency_meta = bool(raw_rec_flag)
                    if len(m_int) == 7:
                        raw_z0_flag = int(m_int[6])
                        if raw_z0_flag not in (0, 1):
                            raise ValueError(
                                f"Invalid snapshot meta_ints zero_unit0 flag: {raw_z0_flag}"
                            )
                        zero_unit0_meta = bool(raw_z0_flag)
            else:
                num_tenants, r_sub, m_sparse, m_exemplars = T, R, M_sp, M_ex
                if m_sparse > MAX_FLEET_SPARSE_CAP or m_exemplars > MAX_FLEET_EXEMPLAR_CAP:
                    raise ValueError("Snapshot array capacities exceed uint8 limit 255")

            bool_thr_meta = 0.5
            if "meta_floats" in zf:
                m_flt = zf["meta_floats"]
                lam_rls, alpha = float(m_flt[0]), float(m_flt[1])
                if len(m_flt) == 3:
                    bool_thr_meta = float(m_flt[2])
            else:
                lam_rls, alpha = 1.5, 0.10
            if not math.isfinite(lam_rls) or not (1e-12 <= lam_rls <= 1e12):
                raise ValueError(f"Invalid snapshot lam_rls: {lam_rls!r}")
            if not math.isfinite(alpha) or not (0.0 < alpha < 1.0):
                raise ValueError(f"Invalid snapshot alpha: {alpha!r}")
            if not math.isfinite(bool_thr_meta) or not (0.0 <= bool_thr_meta <= 1.0):
                raise ValueError(f"Invalid snapshot bool_threshold: {bool_thr_meta!r}")
            if is_boolean_meta is False and bool_thr_meta != 0.5:
                raise ValueError("Snapshot bool_threshold requires is_boolean=True")

            sp_len_arr = zf["sp_len"]
            ex_len_arr = zf["ex_len"]
            if sp_len_arr.dtype != np.dtype(np.uint8) or int(np.min(sp_len_arr)) < 0 or int(np.max(sp_len_arr)) > m_sparse:
                raise ValueError(f"Snapshot sp_len invalid or outside [0, {m_sparse}]")
            if ex_len_arr.dtype != np.dtype(np.uint8) or int(np.min(ex_len_arr)) < 0 or int(np.max(ex_len_arr)) > m_exemplars:
                raise ValueError(f"Snapshot ex_len invalid or outside [0, {m_exemplars}]")

            sp_cls_arr = zf["sp_cls"]
            if sp_cls_arr.size > 0 and (int(np.min(sp_cls_arr)) < 0 or int(np.max(sp_cls_arr)) >= K):
                raise ValueError(f"Snapshot sp_cls index outside [0, {K})")

            ex_cls_arr = zf["ex_cls"]
            if ex_cls_arr.size > 0:
                if int(np.min(ex_cls_arr)) < -1 or int(np.max(ex_cls_arr)) >= K:
                    raise ValueError(f"Snapshot ex_cls index outside [-1, {K})")
                active_ex = np.arange(M_ex, dtype=np.int64)[None, :] < ex_len_arr.astype(np.int64)[:, None]
                if bool(np.any(ex_cls_arr[active_ex] < 0)):
                    raise ValueError("Snapshot contains negative ex_cls in active exemplar slot")

            def _load_checked_raw(
                name: str,
                *,
                min_v: Optional[float] = None,
                max_v: Optional[float] = None,
                max_abs: Optional[float] = None,
                strict_gt: Optional[float] = None,
                strict_lt: Optional[float] = None,
            ) -> np.ndarray:
                raw = zf[name]
                arr64 = np.asarray(raw, dtype=np.float64)
                if not bool(np.all(np.isfinite(arr64))):
                    raise ValueError(f"Snapshot array {name!r} contains non-finite values")
                if arr64.size > 0:
                    lo = float(np.min(arr64))
                    hi = float(np.max(arr64))
                    if max_abs is not None and max(abs(lo), abs(hi)) > max_abs:
                        raise ValueError(
                            f"Snapshot array {name!r} exceeds safe magnitude bound {max_abs}"
                        )
                    if min_v is not None and lo < min_v:
                        raise ValueError(f"Snapshot {name!r} below minimum {min_v}: {lo}")
                    if strict_gt is not None and lo <= strict_gt:
                        raise ValueError(f"Snapshot {name!r} must be > {strict_gt}: {lo}")
                    if max_v is not None and hi > max_v:
                        raise ValueError(f"Snapshot {name!r} above maximum {max_v}: {hi}")
                    if strict_lt is not None and hi >= strict_lt:
                        raise ValueError(f"Snapshot {name!r} must be < {strict_lt}: {hi}")
                return raw

            raw_W_base = _load_checked_raw("W_base", max_abs=1e12)
            raw_b_base = _load_checked_raw("b_base", max_abs=1e12)
            raw_U_shared = _load_checked_raw("U_shared", max_abs=1e12)
            raw_q_init = _load_checked_raw("q_init", min_v=0.0, max_v=1.0)
            raw_dW_scale = _load_checked_raw("dW_scale", min_v=0.0, max_v=65504.0)
            raw_P_blk = _load_checked_raw("P_blk", max_abs=1e12)
            raw_sp_w = _load_checked_raw("sp_w", max_abs=65504.0)
            raw_ex_scales = _load_checked_raw("ex_scales", min_v=0.0, max_v=65504.0)
            raw_q_c = _load_checked_raw("q_c", min_v=0.0, max_v=1.0)
            raw_u_c = _load_checked_raw("u_c", min_v=0.0, max_v=16.0)
            raw_alpha_c = _load_checked_raw("alpha_c", strict_gt=0.0, strict_lt=1.0)
            raw_err_ewma = _load_checked_raw("err_ewma", min_v=0.0, max_v=1.0)

            with np.errstate(over="ignore", invalid="ignore"):
                W_base_arr = raw_W_base.astype(np.float32)
                b_base_arr = raw_b_base.astype(np.float32)
                U_shared_arr = raw_U_shared.astype(np.float32)
                base_tok_df_arr = zf["base_tok_df"].astype(np.uint16)
                q_init_arr = raw_q_init.astype(np.float32)
                dW_q_arr = zf["dW_q"]
                dW_scale_arr = raw_dW_scale.astype(np.float16)
                P_blk_arr = raw_P_blk.astype(np.float32)
                sp_hash_arr = zf["sp_hash"]
                sp_w_arr = raw_sp_w.astype(np.float16)
                ex_vecs_q_arr = zf["ex_vecs_q"]
                ex_scales_arr = raw_ex_scales.astype(np.float16)
                q_c_arr = raw_q_c.copy() if raw_q_c.dtype == np.dtype(np.float16) else raw_q_c.astype(np.float64)
                u_c_arr = raw_u_c.astype(np.float16)
                alpha_c_arr = raw_alpha_c.astype(np.float64)
                err_ewma_arr = raw_err_ewma.astype(np.float16)

            for arr_name, arr_val in (
                ("W_base", W_base_arr),
                ("b_base", b_base_arr),
                ("U_shared", U_shared_arr),
                ("q_init", q_init_arr),
                ("dW_scale", dW_scale_arr),
                ("P_blk", P_blk_arr),
                ("sp_w", sp_w_arr),
                ("ex_scales", ex_scales_arr),
                ("q_c", q_c_arr),
                ("u_c", u_c_arr),
                ("alpha_c", alpha_c_arr),
                ("err_ewma", err_ewma_arr),
            ):
                if not bool(np.all(np.isfinite(arr_val))):
                    raise ValueError(f"Snapshot array {arr_name!r} contains non-finite values")

            for arr_name, arr_val in (
                ("W_base", W_base_arr),
                ("b_base", b_base_arr),
                ("U_shared", U_shared_arr),
                ("P_blk", P_blk_arr),
            ):
                if arr_val.size > 0 and float(np.max(np.abs(arr_val))) > 1e12:
                    raise ValueError(f"Snapshot array {arr_name!r} exceeds safe float32 magnitude bound")

            if dW_scale_arr.size > 0 and float(np.min(dW_scale_arr)) < 0.0:
                raise ValueError("Snapshot dW_scale must be nonnegative")
            if ex_scales_arr.size > 0 and float(np.min(ex_scales_arr)) < 0.0:
                raise ValueError("Snapshot ex_scales must be nonnegative")
            if float(np.min(q_init_arr)) < 0.0 or float(np.max(q_init_arr)) > 1.0:
                raise ValueError("Snapshot q_init outside [0.0, 1.0]")
            if float(np.min(q_c_arr)) < 0.0 or float(np.max(q_c_arr)) > 1.0:
                raise ValueError("Snapshot q_c outside [0.0, 1.0]")
            if q_c_arr.dtype != np.dtype(np.float16):
                q_c_arr = _ceil_float16_unit(q_c_arr)
            if u_c_arr.size > 0 and (float(np.min(u_c_arr)) < 0.0 or float(np.max(u_c_arr)) > 16.0):
                raise ValueError("Snapshot u_c outside [0.0, 16.0]")
            if float(np.min(alpha_c_arr)) <= 0.0 or float(np.max(alpha_c_arr)) >= 1.0:
                raise ValueError("Snapshot alpha_c outside (0.0, 1.0)")
            if float(np.min(err_ewma_arr)) < 0.0 or float(np.max(err_ewma_arr)) > 1.0:
                raise ValueError("Snapshot err_ewma outside [0.0, 1.0]")

            raw_head_meta: Optional[Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray]]] = None
            if "raw_W_auth" in zf:
                raw_w_meta = zf["raw_W_auth"]
                raw_b_meta = zf["raw_b_auth"]
                raw_t_arr = np.asarray(zf["raw_cal_temp"])
                if (
                    raw_t_arr.dtype == np.dtype(np.bool_)
                    or raw_t_arr.dtype.kind not in ("i", "u", "f")
                    or raw_t_arr.shape != (1,)
                ):
                    raise ValueError(
                        f"Invalid snapshot raw_cal_temp spec: dtype={raw_t_arr.dtype}, shape={raw_t_arr.shape}"
                    )
                cal_t_meta = float(np.asarray(raw_t_arr, dtype=np.float64).reshape(-1)[0])
                raw_adj_meta = zf["raw_cal_adj"] if "raw_cal_adj" in zf else None
                raw_head_meta = (raw_w_meta, raw_b_meta, cal_t_meta, raw_adj_meta)

            fleet = cls(
                projector=projector,
                classes=resolved_classes,
                W_base=W_base_arr,
                b_base=b_base_arr,
                U_shared=U_shared_arr,
                base_tok_df=base_tok_df_arr,
                q_init=q_init_arr,
                num_tenants=num_tenants,
                r_sub=r_sub,
                m_sparse=m_sparse,
                m_exemplars=m_exemplars,
                lam_rls=lam_rls,
                alpha=alpha,
                token_cache_limit=token_cache_limit,
                is_boolean=is_boolean_meta,
                bool_threshold=bool_thr_meta,
                recency_weighted=recency_meta,
                zero_unit0=zero_unit0_meta,
                _raw_source_head=raw_head_meta,
            )
            fleet.dW_q[:] = dW_q_arr
            fleet.dW_scale[:] = dW_scale_arr
            fleet.P_blk[:] = P_blk_arr
            fleet.sp_hash[:] = sp_hash_arr
            fleet.sp_cls[:] = sp_cls_arr
            fleet.sp_w[:] = sp_w_arr
            fleet.sp_len[:] = sp_len_arr
            fleet.ex_vecs_q[:] = ex_vecs_q_arr
            fleet.ex_scales[:] = ex_scales_arr
            fleet.ex_cls[:] = ex_cls_arr
            fleet.ex_len[:] = ex_len_arr
            fleet.q_c[:] = q_c_arr
            fleet.u_c[:] = u_c_arr
            fleet.alpha_c[:] = alpha_c_arr
            fleet.err_ewma[:] = err_ewma_arr
            fleet._active_views = fleet._bind_views(fleet._active_tid)
            return fleet


__all__ = [
    "MultiTenantSkillFleet",
    "TenantDecisionResult",
    "clear_fleet_token_cache",
    "set_fleet_token_cache_limit",
    "tokenize_hashes",
]
