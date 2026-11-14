"""Shared-trunk multi-tenant arena and O(1) per-tenant Mondrian conformal updates."""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import io
import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import zlib

import numpy as np

from system1.core.schema import BooleanField, ChoiceField

_TOK_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_DEFAULT_TOK_CACHE: OrderedDict[str, Tuple[int, ...]] = OrderedDict()
_DEFAULT_TOK_CACHE_LIMIT = 4096


def set_fleet_token_cache_limit(maxsize: int) -> None:
    """Configure the bounded LRU cap for module-level fleet token hashing."""
    global _DEFAULT_TOK_CACHE_LIMIT
    limit = max(0, int(maxsize))
    _DEFAULT_TOK_CACHE_LIMIT = limit
    while len(_DEFAULT_TOK_CACHE) > limit:
        _DEFAULT_TOK_CACHE.popitem(last=False)


def clear_fleet_token_cache() -> None:
    """Clear the shared module-level fleet token hash cache."""
    _DEFAULT_TOK_CACHE.clear()


def tokenize_hashes(
    text: str,
    *,
    cache: Optional[OrderedDict[str, Tuple[int, ...]]] = None,
    cache_limit: Optional[int] = None,
) -> Tuple[int, ...]:
    """Hash normalized tokens into 16-bit buckets with bounded LRU eviction."""
    target_cache = _DEFAULT_TOK_CACHE if cache is None else cache
    limit = _DEFAULT_TOK_CACHE_LIMIT if cache_limit is None else max(0, int(cache_limit))
    if limit > 0:
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
        target_cache[text] = hashes
        target_cache.move_to_end(text)
        while len(target_cache) > limit:
            target_cache.popitem(last=False)
    return hashes


def _softmax_temp(logits: np.ndarray, temp: float = 0.25) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    z = (arr - np.max(arr)) / float(temp)
    e = np.exp(np.clip(z, -50.0, 50.0))
    return e / np.sum(e)


def _round_up_16(val: int) -> int:
    return ((max(16, int(val)) + 15) // 16) * 16


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
    ) -> None:
        self.projector = projector
        self.classes = [str(c) for c in classes]
        if not self.classes:
            raise ValueError("classes must be non-empty")
        self.num_classes = len(self.classes)
        self.c2i = {c: i for i, c in enumerate(self.classes)}
        self.W_base = np.asarray(W_base, dtype=np.float32)
        self.b_base = np.asarray(b_base, dtype=np.float32).reshape(-1)
        if self.W_base.ndim != 2 or self.W_base.shape[0] != self.num_classes:
            raise ValueError(f"W_base shape {self.W_base.shape} must be ({self.num_classes}, D)")
        if self.b_base.shape[0] != self.num_classes:
            raise ValueError(f"b_base length {self.b_base.shape[0]} != num_classes {self.num_classes}")

        self.num_tenants = int(num_tenants)
        if self.num_tenants <= 0:
            raise ValueError(f"num_tenants must be positive, got {self.num_tenants}")
        # Keep r_sub >= K and a multiple of 16 to avoid low-rank collapse across block-diagonal RLS.
        self.r_sub = _round_up_16(max(int(r_sub), self.num_classes))
        self.blk_sz = 16
        self.n_blk = self.r_sub // self.blk_sz

        self.U_shared = (
            _prepare_shared_subspace(self.W_base, self.r_sub)
            if U_shared is None
            else _adapt_subspace_width(U_shared, self.W_base, self.r_sub)
        )
        if base_tok_df is None:
            self.base_tok_df = np.zeros(self.BASE_HASH_SIZE, dtype=np.uint16)
        else:
            df_arr = np.asarray(base_tok_df, dtype=np.uint16).reshape(-1)
            if df_arr.shape[0] != self.BASE_HASH_SIZE:
                raise ValueError(f"base_tok_df length {df_arr.shape[0]} != {self.BASE_HASH_SIZE}")
            self.base_tok_df = df_arr

        self.m_sparse = max(0, int(m_sparse))
        self.m_exemplars = max(0, int(m_exemplars))
        self.lam_rls = float(lam_rls)
        if self.lam_rls <= 0.0:
            raise ValueError(f"lam_rls must be positive, got {self.lam_rls}")
        self.alpha = float(alpha)
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

        q0 = np.full(K, 0.5, dtype=np.float32) if q_init is None else np.asarray(q_init, dtype=np.float32).reshape(-1)
        if q0.shape[0] != K:
            raise ValueError(f"q_init length {q0.shape[0]} != num_classes {K}")
        self.q_init = np.clip(q0, 1e-4, 0.999).astype(np.float32)
        u_init = -np.log(np.clip(1.0 - self.q_init, 1e-4, 1.0))
        self.q_c = np.broadcast_to(self.q_init.astype(np.float16), (N, K)).copy()
        self.u_c = np.broadcast_to(u_init.astype(np.float16), (N, K)).copy()
        self.alpha_c = np.full((N, K), self.alpha, dtype=np.float16)
        self.err_ewma = np.full((N, K), self.alpha, dtype=np.float16)

        self._active_tid = 0
        self._active_views: Tuple[Any, ...] = self._bind_views(0)

    def _validate_tid(self, tid: int) -> int:
        idx = int(tid)
        if not 0 <= idx < self.num_tenants:
            raise IndexError(f"tenant_id {idx} out of range [0, {self.num_tenants})")
        return idx

    def _resolve_class_idx(self, label: Union[int, str]) -> int:
        if isinstance(label, str):
            if label not in self.c2i:
                raise ValueError(f"Unknown class label {label!r}")
            return self.c2i[label]
        idx = int(label)
        if not 0 <= idx < self.num_classes:
            raise ValueError(f"Class index {idx} out of range [0, {self.num_classes})")
        return idx

    def _project_batch(self, texts: Sequence[str]) -> np.ndarray:
        text_list = list(texts)
        project_batch = getattr(self.projector, "project_batch", None)
        if callable(project_batch):
            out = project_batch(text_list)
            if isinstance(out, tuple):
                out = out[0]
            return np.asarray(out, dtype=np.float32)
        return np.vstack([np.asarray(self.projector.project(t), dtype=np.float32) for t in text_list])

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

    def switch_tenant(self, tid: int) -> None:
        """Bind arena slices for `tid` without copying."""
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

    def base_nbytes(self) -> int:
        """Return byte footprint of shared trunk weights and subspace basis."""
        return int(
            self.W_base.nbytes
            + self.b_base.nbytes
            + self.U_shared.nbytes
            + self.base_tok_df.nbytes
            + self.q_init.nbytes
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
        schema = resolved.schema
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
        heads_map = getattr(resolved, "_field_heads", None)
        if heads_map is None:
            heads_map = getattr(resolved, "heads", {})
        head = heads_map[field_name]
        raw_w = np.asarray(head.weights, dtype=np.float32)
        raw_b = np.asarray(head.biases, dtype=np.float32).reshape(-1)
        if isinstance(f_def, BooleanField) or raw_w.shape[0] == 1:
            classes = ["False", "True"]
            t_clamped = min(max(float(getattr(f_def, "threshold", 0.5)), 1e-6), 1.0 - 1e-6)
            b_adj = float(raw_b[0]) - 0.25 * math.log(t_clamped / (1.0 - t_clamped))
            W_base = np.vstack([-0.5 * raw_w[0], 0.5 * raw_w[0]]).astype(np.float32)
            b_base = np.array([-0.5 * b_adj, 0.5 * b_adj], dtype=np.float32)
        elif isinstance(f_def, ChoiceField):
            classes = [str(o) for o in f_def.options]
            W_base = raw_w
            b_base = raw_b
        else:
            raise ValueError(f"MultiTenantSkillFleet supports ChoiceField and BooleanField, got {type(f_def).__name__}")

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
        compiled_heads = getattr(resolved, "heads", None)
        ch = compiled_heads.get(field_name) if isinstance(compiled_heads, dict) else None
        if ch is not None:
            cls_scores = getattr(ch, "class_calibration_scores", {}) or {}
            fallback_q = float(getattr(ch, "conformal_quantile", 0.0) or 0.0)
            cal_scores = getattr(ch, "calibration_scores", ()) or ()
            if fallback_q <= 0.0 and cal_scores:
                n_s = len(cal_scores)
                k = min(n_s, max(1, int(math.ceil((n_s + 1) * (1.0 - alpha))))) - 1
                fallback_q = float(sorted(cal_scores)[k])
            for idx, c_name in enumerate(classes):
                c_vals = cls_scores.get(c_name)
                if c_vals:
                    n_c = len(c_vals)
                    k = min(n_c, max(1, int(math.ceil((n_c + 1) * (1.0 - alpha))))) - 1
                    q_init[idx] = float(sorted(c_vals)[k])
                elif fallback_q > 0.0:
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
        dW = self.dW_q[tid].astype(np.float32) * self.dW_scale[tid].astype(np.float32)
        P_t = self.P_blk[tid]

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
                base_log = self.W_base @ x + self.b_base
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
                    th = set(tokenize_hashes(txt, cache=self._tok_cache, cache_limit=self.token_cache_limit))
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

        max_abs = np.max(np.abs(dW), axis=1, keepdims=True)
        scale_q = np.maximum(max_abs / 127.0, 1e-6).astype(np.float16)
        self.dW_scale[tid] = scale_q
        self.dW_q[tid] = np.clip(np.round(dW / scale_q.astype(np.float32)), -127, 127).astype(np.int8)

        if self.m_sparse > 0:
            top_sp = sorted(sp_map.items(), key=lambda kv: abs(kv[1]), reverse=True)[: self.m_sparse]
            self.sp_len[tid] = len(top_sp)
            for i, ((h, ci), wv) in enumerate(top_sp):
                self.sp_hash[tid, i] = h
                self.sp_cls[tid, i] = ci
                self.sp_w[tid, i] = np.float16(wv)

        if ex_list_v and self.m_exemplars > 0:
            by_cls: Dict[int, List[np.ndarray]] = defaultdict(list)
            for v, ci in zip(ex_list_v, ex_list_c):
                by_cls[ci].append(v)
            chosen_v: List[np.ndarray] = []
            chosen_c: List[int] = []
            for ci, vs in by_cls.items():
                if len(chosen_v) >= self.m_exemplars:
                    break
                cen = np.mean(vs, axis=0)
                cen /= float(np.linalg.norm(cen)) + 1e-8
                chosen_v.append(cen)
                chosen_c.append(ci)
            shot_idx = 0
            while len(chosen_v) < self.m_exemplars:
                added = False
                for ci, vs in by_cls.items():
                    if shot_idx < len(vs) and len(chosen_v) < self.m_exemplars:
                        chosen_v.append(vs[shot_idx])
                        chosen_c.append(ci)
                        added = True
                if not added:
                    break
                shot_idx += 1

            self.ex_len[tid] = len(chosen_v)
            for i, (vec, ci) in enumerate(zip(chosen_v, chosen_c)):
                sc = max(float(np.max(np.abs(vec))) / 127.0, 1e-6)
                self.ex_scales[tid, i] = np.float16(sc)
                self.ex_vecs_q[tid, i] = np.clip(np.round(vec / sc), -127, 127).astype(np.int8)
                self.ex_cls[tid, i] = ci

        if tid == self._active_tid:
            self._active_views = self._bind_views(tid)

    def compute_component_logits(
        self, tid: int, text: str, x: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Decompose base, subspace RLS, sparse lexical, and exemplar logits for tenant `tid`."""
        tid = self._validate_tid(tid)
        if x is None:
            x = np.asarray(self.projector.project(text), dtype=np.float32)
        if tid != self._active_tid:
            self.switch_tenant(tid)
        dW_qt, dW_st, _, sp_ht, sp_ct, sp_wt, n_sp, ex_vt, ex_st, ex_ct, n_ex, _ = self._active_views

        base_log = self.W_base @ x + self.b_base
        z = x @ self.U_shared
        sub_log = (dW_qt.astype(np.float32) @ z) * dW_st.reshape(-1).astype(np.float32)

        sp_log = np.zeros(self.num_classes, dtype=np.float32)
        if n_sp > 0:
            th = set(tokenize_hashes(text, cache=self._tok_cache, cache_limit=self.token_cache_limit))
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

        return {
            "base": base_log,
            "lowrank_rls": base_log + sub_log,
            "sparse_only": base_log + sp_log,
            "exemplar_only": base_log + ex_log,
            "hybrid_full": base_log + sub_log + sp_log + ex_log,
        }

    def decide_live(self, tid: int, text: str) -> Tuple[int, np.ndarray, np.ndarray]:
        """Evaluate `text` for tenant `tid` and return `(pred_idx, logits, conformal_mask)`."""
        tid = self._validate_tid(tid)
        x = np.asarray(self.projector.project(text), dtype=np.float32)
        if tid != self._active_tid:
            self.switch_tenant(tid)
        dW_qt, dW_st, _, sp_ht, sp_ct, sp_wt, n_sp, ex_vt, ex_st, ex_ct, n_ex, q_ct = self._active_views
        base_log = self.W_base @ x + self.b_base
        z = x @ self.U_shared
        sub_log = (dW_qt.astype(np.float32) @ z) * dW_st.reshape(-1).astype(np.float32)
        logits = base_log + sub_log
        if n_sp > 0:
            th = set(tokenize_hashes(text, cache=self._tok_cache, cache_limit=self.token_cache_limit))
            for i in range(n_sp):
                if int(sp_ht[i]) in th:
                    logits[int(sp_ct[i])] += float(sp_wt[i])
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
                logits[ci] += b
        probs = _softmax_temp(logits, temp=0.25)
        scores = 1.0 - probs
        pred_set = scores <= q_ct.astype(np.float64)
        return int(np.argmax(logits)), logits, pred_set

    def decide(self, tid: int, text: str) -> TenantDecisionResult:
        """Evaluate `text` for tenant `tid` and return structured `TenantDecisionResult`."""
        pred_idx, logits, pred_set = self.decide_live(tid, text)
        probs = _softmax_temp(logits, temp=0.25)
        cset = tuple(self.classes[int(i)] for i in np.flatnonzero(pred_set))
        return TenantDecisionResult(
            tenant_id=int(tid),
            predicted_index=pred_idx,
            predicted_label=self.classes[pred_idx],
            confidence=float(probs[pred_idx]),
            conformal_set=cset,
            is_ambiguous=(len(cset) != 1),
            logits=logits,
        )

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
        arr = np.asarray(probs_or_logits, dtype=np.float64).reshape(-1)
        if arr.shape[0] != self.num_classes:
            raise ValueError(f"Expected length {self.num_classes}, got {arr.shape[0]}")
        probs = _softmax_temp(arr, temp=0.25) if is_logits else arr
        score_y = 1.0 - float(np.clip(probs[yi], 0.0, 1.0))
        q_cur = float(self.q_c[tid, yi])
        err_y = 1 if score_y > q_cur else 0

        s_clip = min(score_y, 0.9995)
        u_obs = min(5.0, -math.log(max(1.0 - s_clip, 5e-4)))
        g_aci = float(gamma_aci) if gamma_aci is not None else max(0.03, 0.4 * self.alpha)
        alpha_min = max(0.015, 0.35 * self.alpha)
        alpha_max = min(0.40, 2.5 * self.alpha)
        fe = float(err_y)

        ewma_next = 0.85 * float(self.err_ewma[tid, yi]) + 0.15 * fe
        shift = ewma_next > 1.6 * self.alpha
        boost = 1.6 if shift else 1.0
        a_next = float(self.alpha_c[tid, yi]) + g_aci * boost * (self.alpha - fe)
        a_next = min(max(a_next, alpha_min), alpha_max)

        u_cur = float(self.u_c[tid, yi])
        u_next = u_cur + float(eta_u) * boost * (fe - a_next)
        if err_y and shift:
            u_jump = (1.0 - float(jump_w)) * u_cur + float(jump_w) * min(u_obs, 4.2)
            if u_jump > u_next:
                u_next = u_jump
        u_next = min(max(u_next, 0.01), 5.0)
        q_next = 1.0 - math.exp(-u_next)

        self.err_ewma[tid, yi] = np.float16(ewma_next)
        self.alpha_c[tid, yi] = np.float16(a_next)
        self.u_c[tid, yi] = np.float16(u_next)
        self.q_c[tid, yi] = np.float16(q_next)
        if tid == self._active_tid:
            self._active_views = self._bind_views(tid)
        return q_next

    def fleet_delta_to_bytes(self) -> bytes:
        """Serialize all per-tenant arenas without the shared projector via compressed NumPy stream."""
        buf = io.BytesIO()
        classes_bytes = np.frombuffer(json.dumps(self.classes).encode("utf-8"), dtype=np.uint8)
        meta_ints = np.array([self.num_tenants, self.r_sub, self.m_sparse, self.m_exemplars], dtype=np.int64)
        meta_floats = np.array([self.lam_rls, self.alpha], dtype=np.float64)
        np.savez_compressed(
            buf,
            meta_ints=meta_ints,
            meta_floats=meta_floats,
            classes_utf8=classes_bytes,
            W_base=self.W_base.astype(np.float16),
            b_base=self.b_base.astype(np.float16),
            U_shared=self.U_shared.astype(np.float16),
            base_tok_df=self.base_tok_df,
            q_init=self.q_init.astype(np.float16),
            dW_q=self.dW_q,
            dW_scale=self.dW_scale,
            P_blk=self.P_blk.astype(np.float16),
            sp_hash=self.sp_hash,
            sp_cls=self.sp_cls,
            sp_w=self.sp_w,
            sp_len=self.sp_len,
            ex_vecs_q=self.ex_vecs_q,
            ex_scales=self.ex_scales,
            ex_cls=self.ex_cls,
            ex_len=self.ex_len,
            q_c=self.q_c,
            u_c=self.u_c,
            alpha_c=self.alpha_c,
            err_ewma=self.err_ewma,
        )
        return buf.getvalue()

    @classmethod
    def fleet_delta_from_bytes(
        cls,
        data: bytes,
        *,
        projector: Any,
        classes: Optional[Sequence[str]] = None,
        token_cache_limit: int = 4096,
    ) -> MultiTenantSkillFleet:
        """Restore a MultiTenantSkillFleet from bytes produced by `fleet_delta_to_bytes()`."""
        with np.load(io.BytesIO(data), allow_pickle=False) as zf:
            if classes is not None:
                resolved_classes = list(classes)
            elif "classes_utf8" in zf:
                resolved_classes = json.loads(bytes(zf["classes_utf8"].tobytes()).decode("utf-8"))
            else:
                raise ValueError("Snapshot lacks classes metadata; pass classes= explicitly")

            if "meta_ints" in zf:
                m_int = zf["meta_ints"]
                num_tenants, r_sub, m_sparse, m_exemplars = (
                    int(m_int[0]),
                    int(m_int[1]),
                    int(m_int[2]),
                    int(m_int[3]),
                )
            else:
                num_tenants = int(zf["dW_q"].shape[0])
                r_sub = int(zf["dW_q"].shape[2])
                m_sparse = int(zf["sp_hash"].shape[1])
                m_exemplars = int(zf["ex_vecs_q"].shape[1])

            if "meta_floats" in zf:
                m_flt = zf["meta_floats"]
                lam_rls, alpha = float(m_flt[0]), float(m_flt[1])
            else:
                lam_rls, alpha = 1.5, 0.10

            fleet = cls(
                projector=projector,
                classes=resolved_classes,
                W_base=zf["W_base"].astype(np.float32),
                b_base=zf["b_base"].astype(np.float32),
                U_shared=zf["U_shared"].astype(np.float32),
                base_tok_df=zf["base_tok_df"].astype(np.uint16),
                q_init=zf["q_init"].astype(np.float32),
                num_tenants=num_tenants,
                r_sub=r_sub,
                m_sparse=m_sparse,
                m_exemplars=m_exemplars,
                lam_rls=lam_rls,
                alpha=alpha,
                token_cache_limit=token_cache_limit,
            )
            fleet.dW_q[:] = zf["dW_q"]
            fleet.dW_scale[:] = zf["dW_scale"]
            fleet.P_blk[:] = zf["P_blk"].astype(np.float32)
            fleet.sp_hash[:] = zf["sp_hash"]
            fleet.sp_cls[:] = zf["sp_cls"]
            fleet.sp_w[:] = zf["sp_w"]
            fleet.sp_len[:] = zf["sp_len"]
            fleet.ex_vecs_q[:] = zf["ex_vecs_q"]
            fleet.ex_scales[:] = zf["ex_scales"]
            fleet.ex_cls[:] = zf["ex_cls"]
            fleet.ex_len[:] = zf["ex_len"]
            fleet.q_c[:] = zf["q_c"]
            fleet.u_c[:] = zf["u_c"]
            fleet.alpha_c[:] = zf["alpha_c"]
            fleet.err_ewma[:] = zf["err_ewma"]
            fleet._active_views = fleet._bind_views(fleet._active_tid)
            return fleet


__all__ = [
    "MultiTenantSkillFleet",
    "TenantDecisionResult",
    "clear_fleet_token_cache",
    "set_fleet_token_cache_limit",
    "tokenize_hashes",
]
