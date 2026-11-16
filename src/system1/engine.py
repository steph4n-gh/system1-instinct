"""System 1 Decision Engine Runtime.

Orchestrates non-autoregressive single-pass schema evaluation, calibrated
confidence scoring, conformal prediction sets, cryptographic receipt generation,
and ActionLedger audit integration.
"""

from __future__ import annotations

import copy
import dis
import functools
import hashlib
import inspect
import json
import math
import secrets
import statistics
import sys
import threading
import time
import types
from collections import OrderedDict, defaultdict, deque
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
    JointSchemaCalibrator,
    JointSelectiveCertificate,
    RecalibrationMetrics,
    RecalibrationReport,
    RegressionConformalInterval,
    RegressionConformalPredictor,
    _normalize_rows_f32_safe,
    _normalize_vec_f32_safe,
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
from system1.core.model import (
    FastDecisionCallable,
    PackedMultiQuestionKernel,
    _accepts_recency_weighted,
    _assert_real_numeric_array_like,
    _check_nested_real_seq,
    _clone_preserving_2d_layout,
    _coerce_inference_array_f32_or_f64,
    _coerce_real_1d_f32,
    _compute_engine_inference_signature,
    _compute_hierarchical_logits_np,
    _encode_projector_one,
    _has_custom_validate_decision,
    _model_zero_unit0,
    _normalize_raw_projection,
    _normalize_raw_projection_batch,
    _resolve_authoritative_head,
    _stable_sigmoid,
    _validate_2d_layout_cloneable,
    _validate_forgetting_factor,
    _validate_head_weight_bias_shapes,
    _validate_packed_engine_support,
    _validate_real_scalar,
)
from system1.cache import SemanticSystemOneCache, _validate_cache_inputs
from system1.core.schema import (
    _f64_to_exact_2m1074,
    _format_score_interval,
    _score_abs_diff_clamped,
    _score_abs_diff_exact_2m1074,
    _score_from_unit_prob,
    _score_full_span_clamped,
    _score_half_span,
    _score_interval_bounds,
    _score_mae_from_int_sum,
    _score_normalized_margin,
    _score_to_unit_ratio,
    _score_within_rel_tolerance,
)
from system1.core.embeddings import HybridProjector
from system1.core.telemetry import TelemetryProjector
from system1.core.text import PragmaticCompositeProjector
from system1.ledger import ActionLedger, LedgerWriteError
from system1.receipt import (
    DecisionWitnessReceipt,
    create_decision_receipt,
    public_key_fingerprint,
)


_UNSET_SLOT = object()


def _is_immutable_or_non_state(val: Any) -> bool:
    if val is None or isinstance(
        val,
        (bool, int, float, complex, str, bytes, type, types.ModuleType, DecisionSchema, DecisionField),
    ):
        return True
    if inspect.isroutine(val):
        return True
    if hasattr(val, "acquire") and hasattr(val, "release"):
        return True
    return False


def _collect_instance_slot_names(obj: Any) -> Tuple[str, ...]:
    names: List[str] = []
    seen: set[str] = set()
    for base_cls in type(obj).__mro__:
        cls_slots = base_cls.__dict__.get("__slots__")
        if cls_slots is None:
            continue
        raw_slots = (cls_slots,) if isinstance(cls_slots, str) else cls_slots
        cls_prefix = base_cls.__name__.lstrip("_")
        for s_name in raw_slots:
            if not isinstance(s_name, str) or s_name in ("__dict__", "__weakref__"):
                continue
            attr_name = s_name
            if attr_name.startswith("__") and not attr_name.endswith("__") and cls_prefix:
                attr_name = f"_{cls_prefix}{attr_name}"
            if attr_name not in seen:
                seen.add(attr_name)
                names.append(attr_name)
    return tuple(names)


def _snapshot_obj_dict_and_slots(
    val: Any,
    obj_id: int,
    seen_ids: set[int],
    state_snaps: List[Tuple[Any, Any]],
    dict_snaps: List[Tuple[Any, Dict[str, Any]]],
    slot_snaps: List[Tuple[Any, List[Tuple[str, Any]]]],
    memo: Dict[int, Any],
    skip_ids: Optional[set[int]] = None,
    container_snaps: Optional[List[Tuple[Any, ...]]] = None,
) -> bool:
    inst_dict = getattr(val, "__dict__", None)
    slot_names = _collect_instance_slot_names(val)
    if type(inst_dict) is not dict and not slot_names:
        return False
    if obj_id not in seen_ids:
        seen_ids.add(obj_id)
        if type(inst_dict) is dict:
            d_snap = {
                k: _snapshot_duck_graph(
                    v,
                    seen_ids,
                    state_snaps,
                    dict_snaps,
                    slot_snaps,
                    memo,
                    skip_ids,
                    container_snaps,
                )
                for k, v in inst_dict.items()
            }
            dict_snaps.append((val, d_snap))
        if slot_names:
            s_snap = [
                (
                    s_name,
                    _snapshot_duck_graph(
                        getattr(val, s_name, _UNSET_SLOT),
                        seen_ids,
                        state_snaps,
                        dict_snaps,
                        slot_snaps,
                        memo,
                        skip_ids,
                        container_snaps,
                    )
                    if hasattr(val, s_name)
                    else _UNSET_SLOT,
                )
                for s_name in slot_names
            ]
            slot_snaps.append((val, s_snap))
    return True


def _iter_structured_ndarray_objects(arr: np.ndarray) -> Any:
    if not arr.dtype.hasobject:
        return
    if arr.dtype == np.object_:
        yield from arr.flat
        return
    if arr.dtype.names:
        for fname in arr.dtype.names:
            if arr.dtype.fields[fname][0].hasobject:
                yield from _iter_structured_ndarray_objects(arr[fname])
    elif arr.dtype.subdtype is not None:
        sub_dt, _ = arr.dtype.subdtype
        if sub_dt.hasobject:
            for item in arr.flat:
                if isinstance(item, np.ndarray):
                    yield from _iter_structured_ndarray_objects(item)


def _clone_void_scalar(src: np.void) -> np.void:
    buf = np.empty(1, dtype=src.dtype)
    for fn in src.dtype.names or ():
        fdt = src.dtype.fields[fn][0]
        fv = src[fn]
        if fdt == np.object_:
            buf[fn][0] = fv
        elif fdt.names:
            buf[fn][0] = _clone_void_scalar(fv)
        else:
            buf[fn][0] = fv
    return buf[0]


def _void_state_equal(a: np.void, b: np.void) -> bool:
    if a.dtype != b.dtype:
        return False
    for fn in a.dtype.names or ():
        fdt = a.dtype.fields[fn][0]
        av = a[fn]
        bv = b[fn]
        if fdt == np.object_:
            if av is not bv:
                return False
        elif fdt.names:
            if not _void_state_equal(av, bv):
                return False
        elif not _ndarray_state_equal(np.asarray(av), np.asarray(bv)):
            return False
    return True


def _snapshot_duck_graph(
    val: Any,
    seen_ids: set[int],
    state_snaps: List[Tuple[Any, Any]],
    dict_snaps: List[Tuple[Any, Dict[str, Any]]],
    slot_snaps: List[Tuple[Any, List[Tuple[str, Any]]]],
    memo: Dict[int, Any],
    skip_ids: Optional[set[int]] = None,
    container_snaps: Optional[List[Tuple[Any, ...]]] = None,
) -> Any:
    """Recursively snapshot mutable arrays, containers, nested heads, and duck objects."""
    if _is_immutable_or_non_state(val):
        return val
    obj_id = id(val)
    if skip_ids is not None and obj_id in skip_ids:
        return val
    if obj_id in memo:
        return memo[obj_id]
    if isinstance(val, np.ndarray):
        memo[obj_id] = val
        try:
            cp = _clone_preserving_2d_layout(val, writeable=bool(val.flags.writeable))
        except ValueError:
            cp = np.array(val, copy=True)
            if not val.flags.writeable:
                cp.setflags(write=False)
        if container_snaps is not None:
            container_snaps.append(("ndarray", val, cp, bool(val.flags.writeable)))
        if val.dtype == np.object_:
            for elem in val.flat:
                _snapshot_duck_graph(
                    elem,
                    seen_ids,
                    state_snaps,
                    dict_snaps,
                    slot_snaps,
                    memo,
                    skip_ids,
                    container_snaps,
                )
        elif val.dtype != np.object_ and val.dtype.hasobject:
            for elem in _iter_structured_ndarray_objects(val):
                _snapshot_duck_graph(
                    elem,
                    seen_ids,
                    state_snaps,
                    dict_snaps,
                    slot_snaps,
                    memo,
                    skip_ids,
                    container_snaps,
                )
        if type(val) is not np.ndarray:
            _snapshot_obj_dict_and_slots(
                val,
                obj_id,
                seen_ids,
                state_snaps,
                dict_snaps,
                slot_snaps,
                memo,
                skip_ids,
                container_snaps,
            )
        return val
    if isinstance(val, np.void) and val.dtype.names:
        memo[obj_id] = val
        snap_void = _clone_void_scalar(val)
        for fname in val.dtype.names:
            _snapshot_duck_graph(
                val[fname],
                seen_ids,
                state_snaps,
                dict_snaps,
                slot_snaps,
                memo,
                skip_ids,
                container_snaps,
            )
        if container_snaps is not None:
            container_snaps.append(("np_void", val, snap_void, bool(val.flags.writeable)))
        return val
    if isinstance(val, bytearray):
        memo[obj_id] = val
        if container_snaps is not None:
            container_snaps.append(("bytearray", val, bytes(val)))
        if type(val) is not bytearray:
            _snapshot_obj_dict_and_slots(
                val,
                obj_id,
                seen_ids,
                state_snaps,
                dict_snaps,
                slot_snaps,
                memo,
                skip_ids,
                container_snaps,
            )
        return val
    if isinstance(val, dict):
        memo[obj_id] = val
        has_df = isinstance(val, defaultdict)
        df_snap = (
            _snapshot_duck_graph(
                val.default_factory,
                seen_ids,
                state_snaps,
                dict_snaps,
                slot_snaps,
                memo,
                skip_ids,
                container_snaps,
            )
            if (has_df and val.default_factory is not None)
            else None
        )
        items_snap = [
            (
                _snapshot_duck_graph(
                    k, seen_ids, state_snaps, dict_snaps, slot_snaps, memo, skip_ids, container_snaps
                ),
                _snapshot_duck_graph(
                    v, seen_ids, state_snaps, dict_snaps, slot_snaps, memo, skip_ids, container_snaps
                ),
            )
            for k, v in val.items()
        ]
        if container_snaps is not None:
            container_snaps.append(("dict", val, items_snap, has_df, df_snap))
        if type(val) is not dict:
            _snapshot_obj_dict_and_slots(
                val,
                obj_id,
                seen_ids,
                state_snaps,
                dict_snaps,
                slot_snaps,
                memo,
                skip_ids,
                container_snaps,
            )
        return val
    if isinstance(val, list):
        memo[obj_id] = val
        items_snap = [
            _snapshot_duck_graph(
                x, seen_ids, state_snaps, dict_snaps, slot_snaps, memo, skip_ids, container_snaps
            )
            for x in val
        ]
        if container_snaps is not None:
            container_snaps.append(("list", val, items_snap))
        if type(val) is not list:
            _snapshot_obj_dict_and_slots(
                val,
                obj_id,
                seen_ids,
                state_snaps,
                dict_snaps,
                slot_snaps,
                memo,
                skip_ids,
                container_snaps,
            )
        return val
    if isinstance(val, deque):
        memo[obj_id] = val
        items_snap = [
            _snapshot_duck_graph(
                x, seen_ids, state_snaps, dict_snaps, slot_snaps, memo, skip_ids, container_snaps
            )
            for x in val
        ]
        if container_snaps is not None:
            container_snaps.append(("deque", val, items_snap))
        if type(val) is not deque:
            _snapshot_obj_dict_and_slots(
                val,
                obj_id,
                seen_ids,
                state_snaps,
                dict_snaps,
                slot_snaps,
                memo,
                skip_ids,
                container_snaps,
            )
        return val
    if isinstance(val, set):
        memo[obj_id] = val
        items_snap = [
            _snapshot_duck_graph(
                x, seen_ids, state_snaps, dict_snaps, slot_snaps, memo, skip_ids, container_snaps
            )
            for x in val
        ]
        if container_snaps is not None:
            container_snaps.append(("set", val, items_snap))
        if type(val) is not set:
            _snapshot_obj_dict_and_slots(
                val,
                obj_id,
                seen_ids,
                state_snaps,
                dict_snaps,
                slot_snaps,
                memo,
                skip_ids,
                container_snaps,
            )
        return val
    if isinstance(val, (tuple, frozenset)):
        memo[obj_id] = val
        for x in val:
            _snapshot_duck_graph(
                x, seen_ids, state_snaps, dict_snaps, slot_snaps, memo, skip_ids, container_snaps
            )
        if type(val) not in (tuple, frozenset):
            _snapshot_obj_dict_and_slots(
                val,
                obj_id,
                seen_ids,
                state_snaps,
                dict_snaps,
                slot_snaps,
                memo,
                skip_ids,
                container_snaps,
            )
        return val
    if callable(getattr(val, "_snapshot_state", None)) and callable(getattr(val, "_restore_state", None)):
        memo[obj_id] = val
        if obj_id not in seen_ids:
            seen_ids.add(obj_id)
            state_snaps.append((val, val._snapshot_state()))
        return val
    memo[obj_id] = val
    _snapshot_obj_dict_and_slots(
        val,
        obj_id,
        seen_ids,
        state_snaps,
        dict_snaps,
        slot_snaps,
        memo,
        skip_ids,
        container_snaps,
    )
    return val


def _ndarray_state_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind in ("f", "c"):
        return bool(np.array_equal(a, b, equal_nan=True))
    if a.dtype.names:
        return all(_ndarray_state_equal(a[fname], b[fname]) for fname in a.dtype.names)
    if a.dtype.hasobject:
        if a.dtype == np.object_:
            return all(x is y for x, y in zip(a.flat, b.flat))
        if a.dtype.subdtype is not None:
            return all(
                _ndarray_state_equal(np.asarray(x), np.asarray(y))
                for x, y in zip(a.flat, b.flat)
            )
        return False
    return bool(np.array_equal(a, b))


def _restore_duck_containers(container_snaps: Sequence[Tuple[Any, ...]]) -> None:
    for entry in container_snaps:
        kind = entry[0]
        if kind == "ndarray":
            _, arr, cp, orig_w = entry
            if arr.dtype != cp.dtype:
                continue
            needs_copy = (arr.shape != cp.shape) or bool(orig_w) or (not _ndarray_state_equal(arr, cp))
            if needs_copy:
                try:
                    if not arr.flags.writeable:
                        arr.setflags(write=True)
                    if arr.shape != cp.shape:
                        arr.resize(cp.shape, refcheck=False)
                    np.copyto(arr, cp)
                    if not orig_w and arr.flags.writeable:
                        arr.setflags(write=False)
                except (ValueError, BufferError):
                    pass
            elif not orig_w and arr.flags.writeable:
                try:
                    arr.setflags(write=False)
                except (ValueError, BufferError):
                    pass
        elif kind == "np_void":
            _, void_obj, snap_void, orig_w = entry
            if not _void_state_equal(void_obj, snap_void):
                try:
                    if not void_obj.flags.writeable:
                        void_obj.setflags(write=True)
                    for fname in void_obj.dtype.names or ():
                        void_obj[fname] = snap_void[fname]
                    if not orig_w and void_obj.flags.writeable:
                        void_obj.setflags(write=False)
                except (ValueError, BufferError, TypeError):
                    pass
            elif not orig_w and void_obj.flags.writeable:
                try:
                    void_obj.setflags(write=False)
                except (ValueError, BufferError, TypeError):
                    pass
            elif not orig_w and void_obj.flags.writeable:
                try:
                    void_obj.setflags(write=False)
                except (ValueError, BufferError, TypeError):
                    pass
        elif kind == "bytearray":
            _, ba, raw_b = entry
            ba[:] = raw_b
        elif kind == "dict":
            _, dct, items_snap, has_df, df_snap = entry
            dct.clear()
            dct.update(items_snap)
            if has_df:
                dct.default_factory = df_snap
        elif kind == "list":
            _, lst, items_snap = entry
            lst[:] = items_snap
        elif kind == "deque":
            _, dq, items_snap = entry
            dq.clear()
            dq.extend(items_snap)
        elif kind == "set":
            _, st, items_snap = entry
            st.clear()
            st.update(items_snap)


def _clone_head_attr_value(val: Any) -> Any:
    """Shallow-copy arrays and standard mutable containers without choking on locks/callables."""
    if isinstance(val, np.ndarray):
        try:
            return _clone_preserving_2d_layout(val, writeable=bool(val.flags.writeable))
        except ValueError:
            cp = np.array(val, copy=True)
            if not val.flags.writeable:
                cp.setflags(write=False)
            return cp
    if isinstance(val, bytearray):
        return bytearray(val)
    if isinstance(val, dict):
        return copy.copy(val)
    if isinstance(val, list):
        return list(val)
    if isinstance(val, deque):
        return deque(val, maxlen=val.maxlen)
    if isinstance(val, set):
        return set(val)
    return copy.deepcopy(val)


def _safe_stack_effect(ins: Any, *, default: int = 0) -> int:
    try:
        if ins.arg is None:
            return dis.stack_effect(ins.opcode, jump=False)
        return dis.stack_effect(ins.opcode, ins.arg, jump=False)
    except (ValueError, TypeError):
        return default


_EPHEMERAL_SLOT_KEYS = frozenset(
    {
        "consts",
        "raw_str",
        "raw_int",
        "raw_key_tuple",
        "map_val_consts",
        "map_tainted_keys",
        "map_unk_key_tainted",
        "tuple_len",
        "tuple_tainted_idxs",
    }
)


def _analyze_code_kw_flow(
    code: types.CodeType,
    taint_vars: Optional[frozenset[str]] = None,
    candidate_keys: frozenset[str] = frozenset(),
    closure_vars: frozenset[str] = frozenset(),
    known_code_returns: Optional[Mapping[int, frozenset[Tuple[str, Any]]]] = None,
    outer_code_defs: Optional[Mapping[str, types.CodeType]] = None,
    known_code_called_params: Optional[Mapping[int, frozenset[Tuple[str, Any, bool, int]]]] = None,
    ext_target_codes: Optional[Mapping[Any, Tuple[types.CodeType, int]]] = None,
) -> Tuple[
    set[str],
    set[Tuple[str, str]],
    set[str],
    set[str],
    set[Tuple[str, str]],
    set[str],
    set[str],
    Dict[int, types.CodeType],
    Dict[int, Tuple[types.CodeType, bool]],
    frozenset[str],
    Dict[str, types.CodeType],
    Dict[int, frozenset[str]],
    set[str],
    frozenset[str],
    frozenset[Tuple[str, Any]],
    Dict[int, Tuple[types.CodeType, bool]],
    Tuple[set[str], set[Tuple[str, str]], set[str]],
    frozenset[Tuple[str, Any, bool, int]],
]:
    """Analyze bytecode callable targets, mutated container taint, and active constants."""
    unpacked_names: set[str] = set()
    unpacked_owner_attrs: set[Tuple[str, str]] = set()
    unpacked_leaf_attrs: set[str] = set()
    arg_names: set[str] = set()
    arg_owner_attrs: set[Tuple[str, str]] = set()
    arg_leaf_attrs: set[str] = set()
    active_consts: set[str] = set()
    unpacked_codes: Dict[int, types.CodeType] = {}
    tainted_codes: Dict[int, Tuple[types.CodeType, bool]] = {}
    def_tainted_params: Dict[int, frozenset[str]] = {}
    called_closure_names: set[str] = set()
    returned_slots: set[Tuple[str, Any]] = set()
    invoked_local_codes: Dict[int, Tuple[types.CodeType, bool]] = {}
    called_params: set[Tuple[str, Any, bool, int]] = set()
    ret_invoked_names: set[str] = set()
    ret_invoked_owner_attrs: set[Tuple[str, str]] = set()
    ret_invoked_leaf_attrs: set[str] = set()

    tainted_boxes: set[int] = set()
    unpacked_boxes: set[int] = set()
    box_elem_tags: Dict[int, set[Tuple[str, Any]]] = {}
    mutated_seq_boxes: set[int] = set()
    dict_boxes: set[int] = set()
    box_key_elems: Dict[int, Dict[Any, set[Tuple[str, Any]]]] = {}
    box_unk_key_elems: Dict[int, set[Tuple[str, Any]]] = {}
    box_tainted_keys: Dict[int, set[Any]] = {}
    box_unk_tainted: set[int] = set()
    locals_env: Dict[str, set[Tuple[str, Any]]] = {
        vn: {("kw_taint", True)} for vn in (taint_vars or ())
    }
    pos_only_n_p = getattr(code, "co_posonlyargcount", 0)
    kw_cap_n_p = code.co_argcount + code.co_kwonlyargcount
    for p_idx, p_name in enumerate(code.co_varnames[: code.co_argcount]):
        ent_p = locals_env.setdefault(str(p_name), set())
        ent_p.add(("param_idx", p_idx))
        if p_idx >= pos_only_n_p:
            ent_p.add(("param_name", str(p_name)))
    for k_p_name in code.co_varnames[code.co_argcount : kw_cap_n_p]:
        locals_env.setdefault(str(k_p_name), set()).add(("param_name", str(k_p_name)))
    has_varargs_p = bool(code.co_flags & inspect.CO_VARARGS)
    if has_varargs_p and kw_cap_n_p < len(code.co_varnames):
        locals_env.setdefault(str(code.co_varnames[kw_cap_n_p]), set()).add(
            ("varargs_param", code.co_argcount)
        )
    if (code.co_flags & inspect.CO_VARKEYWORDS):
        vk_idx_p = kw_cap_n_p + (1 if has_varargs_p else 0)
        if vk_idx_p < len(code.co_varnames):
            locals_env.setdefault(str(code.co_varnames[vk_idx_p]), set()).add(("varkw_param", None))
    for cvn in closure_vars:
        locals_env.setdefault(cvn, set()).add(("closure_taint", True))
    _mut_dict_methods = frozenset(
        {
            "update",
            "setdefault",
            "append",
            "extend",
            "extendleft",
            "insert",
            "add",
            "symmetric_difference_update",
        }
    )
    _two_arg_mut_methods = frozenset({"setdefault", "insert"})

    def _collect_val_consts(val: Any, dst: set[str], depth: int = 0) -> None:
        if isinstance(val, types.CodeType):
            return
        if isinstance(val, str):
            if val in candidate_keys:
                dst.add(val)
            return
        if depth >= 3:
            return
        if isinstance(val, (tuple, list, set, frozenset)):
            for item in val:
                _collect_val_consts(item, dst, depth + 1)
        elif isinstance(val, Mapping):
            for k_item, v_item in val.items():
                _collect_val_consts(k_item, dst, depth + 1)
                _collect_val_consts(v_item, dst, depth + 1)

    def _slot_is_tainted(slot: set[Tuple[str, Any]]) -> bool:
        if taint_vars is None or ("kw_taint", True) in slot:
            return True
        return any(k == "box_id" and v in tainted_boxes for k, v in slot)

    def _slot_consts(slot: set[Tuple[str, Any]]) -> frozenset[str]:
        parts = [v for k, v in slot if k == "consts" and v]
        return frozenset().union(*parts) if parts else frozenset()

    def _flush_consts(*slots: set[Tuple[str, Any]]) -> None:
        for sl in slots:
            for k, v in sl:
                if k == "consts" and v:
                    active_consts.update(v)

    def _taint_receiver_slot(slot: set[Tuple[str, Any]]) -> None:
        for kind, val in slot:
            if kind in ("name", "recv_var") and isinstance(val, str) and val not in ("self", "cls"):
                ent = locals_env.setdefault(val, set())
                ent.add(("kw_taint", True))
                ent.discard(("lit_map", None))
                for ek, ev in list(ent):
                    if ek == "box_id":
                        tainted_boxes.add(ev)
                        unpacked_boxes.add(ev)
            elif kind in ("box_id", "recv_box"):
                tainted_boxes.add(val)
                unpacked_boxes.add(val)

    def _extract_pos_def_tainted_params(
        sub_code: Optional[types.CodeType],
        def_slot: set[Tuple[str, Any]],
    ) -> frozenset[str]:
        if not isinstance(sub_code, types.CodeType) or not (
            _slot_is_tainted(def_slot) or ("closure_taint", True) in def_slot
        ):
            return frozenset()
        pos_params = sub_code.co_varnames[: sub_code.co_argcount]
        n_defs = next((v for k, v in def_slot if k == "tuple_len"), None)
        t_idxs = next((v for k, v in def_slot if k == "tuple_tainted_idxs"), None)
        if isinstance(n_defs, int):
            start_d = max(0, len(pos_params) - n_defs)
            idxs = t_idxs if isinstance(t_idxs, frozenset) else frozenset()
            return frozenset(
                str(pos_params[start_d + i])
                for i in idxs
                if 0 <= start_d + i < len(pos_params)
            )
        return frozenset(str(p) for p in pos_params)

    def _extract_kw_def_tainted_params(
        sub_code: Optional[types.CodeType],
        kwdef_slot: set[Tuple[str, Any]],
    ) -> frozenset[str]:
        if not isinstance(sub_code, types.CodeType) or not (
            _slot_is_tainted(kwdef_slot) or ("closure_taint", True) in kwdef_slot
        ):
            return frozenset()
        kwonly_params = sub_code.co_varnames[
            sub_code.co_argcount : sub_code.co_argcount + sub_code.co_kwonlyargcount
        ]
        t_keys_parts = [v for k, v in kwdef_slot if k == "map_tainted_keys" and v]
        if ("map_unk_key_tainted", True) not in kwdef_slot and t_keys_parts:
            all_tk = frozenset().union(*t_keys_parts)
            matched = frozenset(str(p) for p in kwonly_params if p in all_tk)
            return matched or all_tk
        return frozenset(str(p) for p in kwonly_params)

    def _activate_nested_slots(
        slots: Sequence[set[Tuple[str, Any]]],
        *,
        is_unpacked: bool,
        with_tainted_args: bool,
    ) -> None:
        for sl in slots:
            sl_tp_parts = [v for k, v in sl if k == "fn_tainted_params" and v]
            sl_tp = frozenset().union(*sl_tp_parts) if sl_tp_parts else frozenset()
            for kind, val in sl:
                if kind == "fn_def_consts" and val:
                    active_consts.update(val)
                elif kind == "code" and isinstance(val, types.CodeType):
                    cid_s = id(val)
                    if is_unpacked:
                        unpacked_codes[cid_s] = val
                    prev_entry = tainted_codes.get(cid_s)
                    prev_flag = prev_entry[1] if prev_entry is not None else False
                    tainted_codes[cid_s] = (val, prev_flag or with_tainted_args or is_unpacked)
                    if sl_tp:
                        def_tainted_params[cid_s] = (
                            def_tainted_params.get(cid_s, frozenset()) | sl_tp
                        )

    def _record(slot: set[Tuple[str, Any]], *, unpacked: bool) -> None:
        names_dst = unpacked_names if unpacked else arg_names
        oattrs_dst = unpacked_owner_attrs if unpacked else arg_owner_attrs
        lattrs_dst = unpacked_leaf_attrs if unpacked else arg_leaf_attrs
        for kind, val in slot:
            if kind == "name":
                names_dst.add(val)
            elif kind == "attr":
                oattrs_dst.add(val)
            elif kind == "leaf_attr":
                lattrs_dst.add(val)
            elif kind == "call_origin" and isinstance(val, frozenset):
                for ok, ov in val:
                    if ok == "name":
                        names_dst.add(ov)
                    elif ok == "attr":
                        oattrs_dst.add(ov)
                    elif ok == "leaf_attr":
                        lattrs_dst.add(ov)

    def _load_var_slot(varname: str) -> set[Tuple[str, Any]]:
        slot = (set(locals_env[varname]) if varname in locals_env else set()) | {("name", varname)}
        if (
            outer_code_defs
            and varname in outer_code_defs
            and (varname in code.co_freevars or varname not in code.co_varnames)
            and not any(k == "code" for k, _ in slot)
        ):
            slot.add(("code", outer_code_defs[varname]))
        if taint_vars is None or varname in taint_vars:
            slot.add(("kw_taint", True))
        if varname in closure_vars:
            slot.add(("closure_taint", True))
        if any(k == "box_id" and v in tainted_boxes for k, v in slot):
            slot.add(("kw_taint", True))
            slot.discard(("lit_map", None))
        return slot

    ins_list = list(dis.get_instructions(code))

    _RET_COPY_KEYS = frozenset(
        {
            "code",
            "fn_def_consts",
            "fn_tainted_params",
            "closure_taint",
            "kw_taint",
            "name",
            "attr",
            "leaf_attr",
            "call_origin",
            "call_param_ret",
            "param_dep",
            "seq_items",
            "seq_items_ex",
        }
    )

    def _record_called_params_from_slots(
        slots: Sequence[set[Tuple[str, Any]]],
        *,
        is_unpacked: bool,
        taint_mode: int,
    ) -> None:
        for sl in slots:
            for k, v in sl:
                if k in ("param_idx", "param_name", "varargs_from", "varkw_any"):
                    called_params.add((k, v, is_unpacked, taint_mode))

    def _record_ret_invoked(slots: Sequence[set[Tuple[str, Any]]]) -> None:
        for sl in slots:
            for k, v in sl:
                if k == "call_origin" and isinstance(v, frozenset):
                    for ok, ov in v:
                        if ok == "name":
                            ret_invoked_names.add(ov)
                        elif ok == "attr":
                            ret_invoked_owner_attrs.add(ov)
                        elif ok == "leaf_attr":
                            ret_invoked_leaf_attrs.add(ov)

    def _container_bids(slot: set[Tuple[str, Any]]) -> set[int]:
        bids_out: set[int] = set()
        for bk, bv in slot:
            if bk in ("box_id", "recv_box") and isinstance(bv, int):
                bids_out.add(bv)
            elif bk in ("name", "recv_var") and isinstance(bv, str) and bv in locals_env:
                for lk, lv in locals_env[bv]:
                    if lk == "box_id" and isinstance(lv, int):
                        bids_out.add(lv)
        return bids_out

    def _seq_nesting_depth(val: Any) -> int:
        rep_d = 0
        if isinstance(val, tuple) and len(val) == 4 and isinstance(val[1], bool):
            for sub in list(val[0]) + list(val[3]):
                if isinstance(sub, frozenset):
                    for sk, sv in sub:
                        if sk in ("seq_items", "seq_items_ex"):
                            rep_d = max(rep_d, _seq_nesting_depth(sv))
            if isinstance(val[2], frozenset):
                for sk, sv in val[2]:
                    if sk in ("seq_items", "seq_items_ex"):
                        rep_d = max(rep_d, _seq_nesting_depth(sv))
            return 1 + rep_d
        if isinstance(val, tuple):
            for el in val:
                if isinstance(el, frozenset):
                    for sk, sv in el:
                        if sk in ("seq_items", "seq_items_ex"):
                            rep_d = max(rep_d, _seq_nesting_depth(sv))
            return 1 + rep_d
        return 0

    def _freeze_elem_slot(s: set[Tuple[str, Any]]) -> frozenset[Tuple[str, Any]]:
        out_kv: set[Tuple[str, Any]] = set()
        for kv in s:
            if kv[0] in _EPHEMERAL_SLOT_KEYS or kv[0] in ("lit_map", "null", "unk_key"):
                continue
            if kv[0] in ("seq_items", "seq_items_ex") and _seq_nesting_depth(kv[1]) >= 8:
                continue
            out_kv.add(kv)
        return frozenset(out_kv)

    def _record_container_elem(
        cont_slot: set[Tuple[str, Any]],
        elem_slot: set[Tuple[str, Any]],
        *,
        seq_mut: bool = False,
    ) -> None:
        el_frozen = _freeze_elem_slot(elem_slot)
        for bid_c in _container_bids(cont_slot):
            if el_frozen:
                box_elem_tags.setdefault(bid_c, set()).update(el_frozen)
            if seq_mut:
                mutated_seq_boxes.add(bid_c)

    def _lookup_container_elems(cont_slot: set[Tuple[str, Any]]) -> frozenset[Tuple[str, Any]]:
        merged: set[Tuple[str, Any]] = set()
        for bid_c in _container_bids(cont_slot):
            tags = box_elem_tags.get(bid_c)
            if tags:
                merged.update(tags)
        return frozenset(merged)

    def _extract_const_keys(slot: set[Tuple[str, Any]]) -> Optional[frozenset[Any]]:
        if _slot_is_tainted(slot) or ("unk_key", True) in slot:
            return None
        cks = frozenset(v for k, v in slot if k == "const_key")
        return cks if cks else None

    def _record_dict_entry(
        cont_slot: set[Tuple[str, Any]],
        ckeys: Optional[frozenset[Any]],
        val_slot: set[Tuple[str, Any]],
    ) -> None:
        el_fr = _freeze_elem_slot(val_slot)
        val_t = _slot_is_tainted(val_slot)
        for bid_d in _container_bids(cont_slot):
            dict_boxes.add(bid_d)
            if ckeys is not None:
                for ck in ckeys:
                    if el_fr:
                        box_key_elems.setdefault(bid_d, {}).setdefault(ck, set()).update(el_fr)
                    if val_t:
                        box_tainted_keys.setdefault(bid_d, set()).add(ck)
            else:
                if el_fr:
                    box_unk_key_elems.setdefault(bid_d, set()).update(el_fr)
                if val_t:
                    box_unk_tainted.add(bid_d)

    def _merge_dict_into(dst_slot: set[Tuple[str, Any]], src_slot: set[Tuple[str, Any]]) -> None:
        src_bids = _container_bids(src_slot)
        for dst_bid in _container_bids(dst_slot):
            dict_boxes.add(dst_bid)
            copied_dict = False
            for sb in src_bids:
                if sb in dict_boxes:
                    copied_dict = True
                    for k_m, v_m in box_key_elems.get(sb, {}).items():
                        box_key_elems.setdefault(dst_bid, {}).setdefault(k_m, set()).update(v_m)
                    if box_unk_key_elems.get(sb):
                        box_unk_key_elems.setdefault(dst_bid, set()).update(box_unk_key_elems[sb])
                    if box_tainted_keys.get(sb):
                        box_tainted_keys.setdefault(dst_bid, set()).update(box_tainted_keys[sb])
                    if sb in box_unk_tainted:
                        box_unk_tainted.add(dst_bid)
            if not copied_dict and _slot_is_tainted(src_slot):
                box_unk_tainted.add(dst_bid)

    def _lookup_keyed_elems(
        cont_slot: set[Tuple[str, Any]],
        key_slot: Optional[set[Tuple[str, Any]]] = None,
    ) -> Tuple[frozenset[Tuple[str, Any]], bool]:
        ckeys_lk = _extract_const_keys(key_slot) if key_slot is not None else None
        out_tags: set[Tuple[str, Any]] = set()
        is_t_read = False
        param_items: set[Tuple[str, Any]] = {kv for kv in cont_slot if kv[0] in ("varkw_param", "varargs_param")}
        for pk, pv in cont_slot:
            if pk == "recv_var" and isinstance(pv, str) and pv in locals_env:
                for lk, lv in locals_env[pv]:
                    if lk in ("varkw_param", "varargs_param"):
                        param_items.add((lk, lv))
        if ("varkw_param", None) in param_items:
            if ckeys_lk is not None:
                for ck in ckeys_lk:
                    if isinstance(ck, str):
                        out_tags.add(("param_name", ck))
            else:
                out_tags.add(("varkw_any", None))
            if _slot_is_tainted(cont_slot):
                is_t_read = True
        for pk, va_base in param_items:
            if pk == "varargs_param" and isinstance(va_base, int):
                if ckeys_lk is not None and all(
                    isinstance(ck, int) and not isinstance(ck, bool) and ck >= 0
                    for ck in ckeys_lk
                ):
                    for ck in ckeys_lk:
                        out_tags.add(("param_idx", int(va_base) + int(ck)))
                else:
                    out_tags.add(("varargs_from", int(va_base)))
                if _slot_is_tainted(cont_slot):
                    is_t_read = True
        bids_lk = _container_bids(cont_slot)
        for bid_lk in bids_lk:
            if bid_lk in dict_boxes:
                if ckeys_lk is not None:
                    for ck in ckeys_lk:
                        out_tags.update(box_key_elems.get(bid_lk, {}).get(ck, ()))
                        if ck in box_tainted_keys.get(bid_lk, ()):
                            is_t_read = True
                    out_tags.update(box_unk_key_elems.get(bid_lk, ()))
                    if bid_lk in box_unk_tainted:
                        is_t_read = True
                else:
                    for vs in box_key_elems.get(bid_lk, {}).values():
                        out_tags.update(vs)
                    out_tags.update(box_unk_key_elems.get(bid_lk, ()))
                    if (
                        bid_lk in tainted_boxes
                        or bid_lk in box_unk_tainted
                        or bool(box_tainted_keys.get(bid_lk))
                    ):
                        is_t_read = True
            else:
                out_tags.update(box_elem_tags.get(bid_lk, ()))
                if bid_lk in tainted_boxes:
                    is_t_read = True
        if not bids_lk and not param_items and _slot_is_tainted(cont_slot):
            is_t_read = True
        if (key_slot is not None and _slot_is_tainted(key_slot)) or _slot_is_tainted(set(out_tags)):
            is_t_read = True
        return frozenset(out_tags), is_t_read

    def _unpack_seq_repr(
        slot: set[Tuple[str, Any]],
    ) -> Optional[
        Tuple[
            Tuple[frozenset[Tuple[str, Any]], ...],
            bool,
            frozenset[Tuple[str, Any]],
            Tuple[frozenset[Tuple[str, Any]], ...],
        ]
    ]:
        ex_reps = [
            v
            for k, v in slot
            if k == "seq_items_ex" and isinstance(v, tuple) and len(v) == 4
        ]
        if ex_reps:
            return max(ex_reps, key=lambda r: (len(r[0]) + len(r[3]), len(r[2])))
        exacts = [v for k, v in slot if k == "seq_items" and isinstance(v, tuple)]
        if exacts:
            return (max(exacts, key=len), False, frozenset(), ())
        return None

    def _unpack_seq_with_boxes(
        slot: set[Tuple[str, Any]],
    ) -> Optional[
        Tuple[
            Tuple[frozenset[Tuple[str, Any]], ...],
            bool,
            frozenset[Tuple[str, Any]],
            Tuple[frozenset[Tuple[str, Any]], ...],
        ]
    ]:
        rep = _unpack_seq_repr(slot)
        bids_u = _container_bids(slot)
        if any(b in mutated_seq_boxes for b in bids_u):
            base_elems = _collect_rep_elem_tags(rep) if rep is not None else frozenset()
            dyn_u = _lookup_container_elems(slot)
            return ((), True, base_elems | dyn_u, ())
        return rep

    def _collect_rep_elem_tags(
        rep: Tuple[
            Tuple[frozenset[Tuple[str, Any]], ...],
            bool,
            frozenset[Tuple[str, Any]],
            Tuple[frozenset[Tuple[str, Any]], ...],
        ],
    ) -> frozenset[Tuple[str, Any]]:
        pref_r, hs_r, se_r, suff_r = rep
        parts: List[frozenset[Tuple[str, Any]]] = list(pref_r) + list(suff_r)
        if hs_r and se_r:
            parts.append(se_r)
        return frozenset().union(*parts) if parts else frozenset()

    def _concat_seq_reps(
        lhs_rep: Tuple[
            Tuple[frozenset[Tuple[str, Any]], ...],
            bool,
            frozenset[Tuple[str, Any]],
            Tuple[frozenset[Tuple[str, Any]], ...],
        ],
        rhs_rep: Tuple[
            Tuple[frozenset[Tuple[str, Any]], ...],
            bool,
            frozenset[Tuple[str, Any]],
            Tuple[frozenset[Tuple[str, Any]], ...],
        ],
    ) -> Tuple[
        Tuple[frozenset[Tuple[str, Any]], ...],
        bool,
        frozenset[Tuple[str, Any]],
        Tuple[frozenset[Tuple[str, Any]], ...],
    ]:
        d_pref, d_hs, d_se, d_suff = lhs_rep
        i_pref, i_hs, i_se, i_suff = rhs_rep
        if not d_hs and not i_hs:
            return (d_pref + i_pref, False, frozenset(), ())
        if not d_hs and i_hs:
            return (d_pref + i_pref, True, i_se, i_suff)
        if d_hs and not i_hs:
            return (d_pref, True, d_se.union(*d_suff), i_pref)
        return (d_pref, True, d_se.union(i_se, *d_suff, *i_pref), i_suff)

    def _set_seq_repr(
        dst_slot: set[Tuple[str, Any]],
        pref: Tuple[frozenset[Tuple[str, Any]], ...],
        has_star: bool,
        star_el: frozenset[Tuple[str, Any]],
        suff: Tuple[frozenset[Tuple[str, Any]], ...],
    ) -> None:
        for kv in [x for x in dst_slot if x[0] in ("seq_items", "seq_items_ex")]:
            dst_slot.discard(kv)
        max_exact = max(64, len(ins_list))
        if not has_star and len(pref) <= max_exact:
            dst_slot.add(("seq_items", pref))
            return
        keep_pref = pref[:32]
        if has_star:
            keep_suff = suff[-32:]
            mid_items = pref[len(keep_pref) :] + suff[: max(0, len(suff) - len(keep_suff))]
        else:
            rem_pref = pref[len(keep_pref) :]
            keep_suff = rem_pref[-32:]
            mid_items = rem_pref[: max(0, len(rem_pref) - len(keep_suff))]
        merged_star = star_el.union(*mid_items) if mid_items else star_el
        dst_slot.add(("seq_items_ex", (keep_pref, True, merged_star, keep_suff)))

    def _collect_slot_param_tags(slots: Sequence[set[Tuple[str, Any]]]) -> frozenset[Tuple[str, Any]]:
        out_pt: set[Tuple[str, Any]] = set()

        def _collect_fs(fs: frozenset[Tuple[str, Any]]) -> None:
            for k_f, v_f in fs:
                if k_f in ("param_idx", "param_name", "varargs_from", "varkw_any"):
                    out_pt.add((k_f, v_f))
                elif k_f == "varargs_param":
                    out_pt.add(("varargs_from", v_f))
                elif k_f == "varkw_param":
                    out_pt.add(("varkw_any", None))
                elif k_f == "param_dep" and isinstance(v_f, frozenset):
                    out_pt.update(v_f)

        for sl in slots:
            _collect_fs(frozenset(sl))
            for k_s, v_s in sl:
                if k_s == "seq_items" and isinstance(v_s, tuple):
                    for el_pt in v_s:
                        _collect_fs(el_pt)
                elif k_s == "seq_items_ex" and isinstance(v_s, tuple) and len(v_s) == 4:
                    for el_pt in v_s[0]:
                        _collect_fs(el_pt)
                    _collect_fs(v_s[2])
                    for el_pt in v_s[3]:
                        _collect_fs(el_pt)
            for bid_p in _container_bids(sl):
                if bid_p in dict_boxes:
                    for dv_pt in box_key_elems.get(bid_p, {}).values():
                        _collect_fs(frozenset(dv_pt))
                    _collect_fs(frozenset(box_unk_key_elems.get(bid_p, ())))
                elif bid_p in box_elem_tags:
                    _collect_fs(frozenset(box_elem_tags[bid_p]))
        return frozenset(out_pt)

    def _apply_known_call_returns(
        callable_slots: Sequence[set[Tuple[str, Any]]],
        arg_slots_rev: Sequence[set[Tuple[str, Any]]],
        ret_slot: set[Tuple[str, Any]],
        ret_bid: int,
        *,
        is_tainted_call: bool,
        allow_pos_arg_fwd: bool,
        kw_arg_map: Optional[Mapping[str, set[Tuple[str, Any]]]] = None,
        unk_kw_slot: Optional[set[Tuple[str, Any]]] = None,
        star_pos_slot: Optional[set[Tuple[str, Any]]] = None,
        attach_param_dep: bool = True,
    ) -> None:
        orig_tags: set[Tuple[str, Any]] = set()
        is_partial = False
        is_seq_ctor = False
        is_dict_ctor = False

        def _copy_ret_src_slot(arg_src: set[Tuple[str, Any]]) -> None:
            if _slot_is_tainted(arg_src):
                tainted_boxes.add(ret_bid)
                ret_slot.add(("kw_taint", True))
            for ak, av in arg_src:
                if ak in _RET_COPY_KEYS or ak in ("param_idx", "param_name", "varargs_from", "varkw_any"):
                    ret_slot.add((ak, av))
                elif ak in ("box_id", "recv_box"):
                    ret_slot.add(("recv_box", av))
                    if av in tainted_boxes:
                        tainted_boxes.add(ret_bid)
                        ret_slot.add(("kw_taint", True))

        def _resolve_param_sources(
            cv_c: types.CodeType,
            bound_off: int,
            pk: str,
            pv: Any,
        ) -> List[set[Tuple[str, Any]]]:
            cv_po = getattr(cv_c, "co_posonlyargcount", 0)
            cv_pos_names_r = cv_c.co_varnames[cv_po : cv_c.co_argcount]
            cv_kw_end = cv_c.co_argcount + cv_c.co_kwonlyargcount
            cv_kw_names_r = set(str(x) for x in cv_c.co_varnames[cv_po:cv_kw_end])
            out_srcs: List[set[Tuple[str, Any]]] = []
            if pk == "param_idx" and isinstance(pv, int):
                eff_idx = pv - bound_off
                if eff_idx < 0:
                    return out_srcs
                if allow_pos_arg_fwd and 0 <= eff_idx < len(arg_slots_rev):
                    out_srcs.append(arg_slots_rev[len(arg_slots_rev) - 1 - eff_idx])
                elif not allow_pos_arg_fwd and star_pos_slot:
                    out_srcs.append(star_pos_slot)
            elif pk == "param_name" and isinstance(pv, str):
                if kw_arg_map and pv in kw_arg_map:
                    out_srcs.append(kw_arg_map[pv])
                elif allow_pos_arg_fwd and pv in cv_pos_names_r:
                    eff_i = cv_po + cv_pos_names_r.index(pv) - bound_off
                    if 0 <= eff_i < len(arg_slots_rev):
                        out_srcs.append(arg_slots_rev[len(arg_slots_rev) - 1 - eff_i])
                elif not allow_pos_arg_fwd and star_pos_slot and pv in cv_pos_names_r:
                    if (cv_po + cv_pos_names_r.index(pv) - bound_off) >= 0:
                        out_srcs.append(star_pos_slot)
                elif unk_kw_slot and pv in cv_kw_names_r:
                    out_srcs.append(unk_kw_slot)
            elif pk == "varargs_from" and isinstance(pv, int):
                eff_va = max(0, pv - bound_off)
                if allow_pos_arg_fwd:
                    for idx_va in range(eff_va, len(arg_slots_rev)):
                        out_srcs.append(arg_slots_rev[len(arg_slots_rev) - 1 - idx_va])
                if star_pos_slot:
                    out_srcs.append(star_pos_slot)
            elif pk == "varkw_any":
                if kw_arg_map:
                    for kw_k, kw_v in kw_arg_map.items():
                        if kw_k not in cv_kw_names_r:
                            out_srcs.append(kw_v)
                if unk_kw_slot:
                    out_srcs.append(unk_kw_slot)
            return out_srcs

        def _lookup_ext_target(ak: str, av: Any) -> Tuple[Optional[types.CodeType], int]:
            if ak == "code" and isinstance(av, types.CodeType):
                return av, 0
            if ak in ("name", "attr", "leaf_attr") and ext_target_codes is not None and (ak, av) in ext_target_codes:
                return ext_target_codes[(ak, av)]
            return None, 0

        def _subst_slot_fset(
            cv_c: types.CodeType,
            bound_off: int,
            slot_fset: frozenset[Tuple[str, Any]],
            depth: int = 0,
            *,
            allow_param_subst: bool = True,
        ) -> frozenset[Tuple[str, Any]]:
            out_kv: set[Tuple[str, Any]] = set()
            had_resolved_param = False
            for rk_s, rv_s in slot_fset:
                if rk_s in ("param_idx", "param_name", "varargs_from", "varkw_any"):
                    if not allow_param_subst:
                        continue
                    srcs = _resolve_param_sources(cv_c, bound_off, rk_s, rv_s)
                    if srcs:
                        had_resolved_param = True
                        for arg_src in srcs:
                            if _slot_is_tainted(arg_src):
                                out_kv.add(("kw_taint", True))
                            for ak, av in arg_src:
                                if ak in _RET_COPY_KEYS or ak in ("param_idx", "param_name", "varargs_from", "varkw_any"):
                                    out_kv.add((ak, av))
                                elif ak in ("box_id", "recv_box"):
                                    out_kv.add(("recv_box", av))
                                    if av in tainted_boxes:
                                        out_kv.add(("kw_taint", True))
                    elif depth > 0:
                        out_kv.add((rk_s, rv_s))
                elif rk_s == "call_param_ret" and isinstance(rv_s, frozenset) and depth < 8:
                    if not allow_param_subst:
                        continue
                    for pk_c, pv_c in rv_s:
                        for arg_src in _resolve_param_sources(cv_c, bound_off, pk_c, pv_c):
                            resid_cp = {
                                (ak, av)
                                for ak, av in arg_src
                                if ak in ("param_idx", "param_name", "varargs_from", "varkw_any")
                            }
                            if resid_cp:
                                out_kv.add(("call_param_ret", frozenset(resid_cp)))
                            for ak, av in arg_src:
                                sub_c_call, sub_boff = _lookup_ext_target(ak, av)
                                if sub_c_call is not None and known_code_returns:
                                    sub_ret = known_code_returns.get(id(sub_c_call))
                                    if sub_ret:
                                        out_kv.update(
                                            _subst_slot_fset(
                                                sub_c_call,
                                                sub_boff,
                                                sub_ret,
                                                depth + 1,
                                                allow_param_subst=False,
                                            )
                                        )
                elif rk_s == "seq_items" and isinstance(rv_s, tuple) and depth < 8:
                    out_kv.add(
                        (
                            "seq_items",
                            tuple(
                                _subst_slot_fset(
                                    cv_c,
                                    bound_off,
                                    el_f,
                                    depth + 1,
                                    allow_param_subst=allow_param_subst,
                                )
                                for el_f in rv_s
                            ),
                        )
                    )
                elif rk_s == "seq_items_ex" and isinstance(rv_s, tuple) and len(rv_s) == 4 and depth < 8:
                    out_kv.add(
                        (
                            "seq_items_ex",
                            (
                                tuple(
                                    _subst_slot_fset(
                                        cv_c,
                                        bound_off,
                                        el_f,
                                        depth + 1,
                                        allow_param_subst=allow_param_subst,
                                    )
                                    for el_f in rv_s[0]
                                ),
                                rv_s[1],
                                _subst_slot_fset(
                                    cv_c,
                                    bound_off,
                                    rv_s[2],
                                    depth + 1,
                                    allow_param_subst=allow_param_subst,
                                ),
                                tuple(
                                    _subst_slot_fset(
                                        cv_c,
                                        bound_off,
                                        el_f,
                                        depth + 1,
                                        allow_param_subst=allow_param_subst,
                                    )
                                    for el_f in rv_s[3]
                                ),
                            ),
                        )
                    )
                elif rk_s == "param_dep" and isinstance(rv_s, frozenset):
                    fwd_deps: set[Tuple[str, Any]] = set()
                    for dp_k, dp_v in rv_s:
                        for arg_src in _resolve_param_sources(cv_c, bound_off, dp_k, dp_v):
                            if _slot_is_tainted(arg_src):
                                out_kv.add(("kw_taint", True))
                            fwd_deps.update(_collect_slot_param_tags([arg_src]))
                    if fwd_deps:
                        out_kv.add(("param_dep", frozenset(fwd_deps)))
                elif rk_s == "code" and isinstance(rv_s, types.CodeType):
                    out_kv.add((rk_s, rv_s))
                    if allow_param_subst and orig_tags and rv_s.co_freevars:
                        out_kv.add(("call_origin", frozenset(orig_tags)))
                elif rk_s == "call_origin" and isinstance(rv_s, frozenset):
                    if allow_param_subst and orig_tags:
                        out_kv.add(("call_origin", rv_s | frozenset(orig_tags)))
                    else:
                        out_kv.add((rk_s, rv_s))
                elif rk_s != "name":
                    out_kv.add((rk_s, rv_s))
            if not had_resolved_param:
                for rk_s, rv_s in slot_fset:
                    if rk_s == "name":
                        out_kv.add((rk_s, rv_s))
            return frozenset(out_kv)

        def _apply_ret_fset(
            cv_e: types.CodeType,
            boff_e: int,
            ret_fset: frozenset[Tuple[str, Any]],
            depth_r: int = 0,
            *,
            allow_param_subst: bool = True,
        ) -> None:
            if depth_r >= 8:
                return
            for rk, rv in ret_fset:
                if rk in ("param_idx", "param_name", "varargs_from", "varkw_any"):
                    if not allow_param_subst:
                        continue
                    for src_s in _resolve_param_sources(cv_e, boff_e, rk, rv):
                        _copy_ret_src_slot(src_s)
                elif rk == "call_param_ret" and isinstance(rv, frozenset):
                    if not allow_param_subst:
                        continue
                    for pk_c, pv_c in rv:
                        for src_s in _resolve_param_sources(cv_e, boff_e, pk_c, pv_c):
                            resid_cp = {
                                (ak_r, av_r)
                                for ak_r, av_r in src_s
                                if ak_r in ("param_idx", "param_name", "varargs_from", "varkw_any")
                            }
                            if resid_cp:
                                ret_slot.add(("call_param_ret", frozenset(resid_cp)))
                            for ak_c, av_c in src_s:
                                sub_c_call, sub_boff = _lookup_ext_target(ak_c, av_c)
                                if sub_c_call is not None and known_code_returns:
                                    sub_ret = known_code_returns.get(id(sub_c_call))
                                    if sub_ret:
                                        _apply_ret_fset(
                                            sub_c_call,
                                            sub_boff,
                                            sub_ret,
                                            depth_r + 1,
                                            allow_param_subst=False,
                                        )
                elif rk == "kw_taint" and rv:
                    tainted_boxes.add(ret_bid)
                    ret_slot.add(("kw_taint", True))
                elif rk == "ret_elems" and isinstance(rv, frozenset):
                    box_elem_tags.setdefault(ret_bid, set()).update(
                        _subst_slot_fset(
                            cv_e,
                            boff_e,
                            rv,
                            depth_r + 1,
                            allow_param_subst=allow_param_subst,
                        )
                    )
                elif rk == "ret_dict" and isinstance(rv, tuple) and len(rv) == 2:
                    dict_boxes.add(ret_bid)
                    for rk_d, rv_d in rv[0]:
                        sub_vd = _subst_slot_fset(
                            cv_e,
                            boff_e,
                            rv_d,
                            depth_r + 1,
                            allow_param_subst=allow_param_subst,
                        )
                        if sub_vd:
                            box_key_elems.setdefault(ret_bid, {}).setdefault(rk_d, set()).update(sub_vd)
                        if any(k_x == "kw_taint" and v_x for k_x, v_x in sub_vd):
                            box_tainted_keys.setdefault(ret_bid, set()).add(rk_d)
                    if rv[1]:
                        sub_ud = _subst_slot_fset(
                            cv_e,
                            boff_e,
                            rv[1],
                            depth_r + 1,
                            allow_param_subst=allow_param_subst,
                        )
                        if sub_ud:
                            box_unk_key_elems.setdefault(ret_bid, set()).update(sub_ud)
                        if any(k_x == "kw_taint" and v_x for k_x, v_x in sub_ud):
                            box_unk_tainted.add(ret_bid)
                elif rk == "seq_items" and isinstance(rv, tuple):
                    ret_slot.add(
                        (
                            "seq_items",
                            tuple(
                                _subst_slot_fset(
                                    cv_e,
                                    boff_e,
                                    el_f,
                                    depth_r + 1,
                                    allow_param_subst=allow_param_subst,
                                )
                                for el_f in rv
                            ),
                        )
                    )
                elif rk == "seq_items_ex" and isinstance(rv, tuple) and len(rv) == 4:
                    ret_slot.add(
                        (
                            "seq_items_ex",
                            (
                                tuple(
                                    _subst_slot_fset(
                                        cv_e,
                                        boff_e,
                                        el_f,
                                        depth_r + 1,
                                        allow_param_subst=allow_param_subst,
                                    )
                                    for el_f in rv[0]
                                ),
                                rv[1],
                                _subst_slot_fset(
                                    cv_e,
                                    boff_e,
                                    rv[2],
                                    depth_r + 1,
                                    allow_param_subst=allow_param_subst,
                                ),
                                tuple(
                                    _subst_slot_fset(
                                        cv_e,
                                        boff_e,
                                        el_f,
                                        depth_r + 1,
                                        allow_param_subst=allow_param_subst,
                                    )
                                    for el_f in rv[3]
                                ),
                            ),
                        )
                    )
                elif rk == "param_dep" and isinstance(rv, frozenset):
                    fwd_deps_r: set[Tuple[str, Any]] = set()
                    for dp_k, dp_v in rv:
                        for src_s in _resolve_param_sources(cv_e, boff_e, dp_k, dp_v):
                            if _slot_is_tainted(src_s):
                                tainted_boxes.add(ret_bid)
                                ret_slot.add(("kw_taint", True))
                            fwd_deps_r.update(_collect_slot_param_tags([src_s]))
                    if fwd_deps_r:
                        ret_slot.add(("param_dep", frozenset(fwd_deps_r)))
                elif rk in _RET_COPY_KEYS:
                    ret_slot.add((rk, rv))

        target_codes: List[Tuple[types.CodeType, int]] = []
        seen_target_ids: set[Tuple[int, int]] = set()
        for cs in callable_slots:
            for ck, cv in cs:
                if ck in ("name", "attr", "leaf_attr"):
                    orig_tags.add((ck, cv))
                    if (ck == "name" and cv == "partial") or (ck == "leaf_attr" and cv == "partial"):
                        is_partial = True
                    if ck == "name" and cv in ("list", "tuple"):
                        is_seq_ctor = True
                    if ck == "name" and cv == "dict" and "dict" not in locals_env:
                        is_dict_ctor = True
                    if ext_target_codes is not None and (ck, cv) in ext_target_codes:
                        ext_c, ext_boff = ext_target_codes[(ck, cv)]
                        tkey = (id(ext_c), ext_boff)
                        if tkey not in seen_target_ids:
                            seen_target_ids.add(tkey)
                            target_codes.append((ext_c, ext_boff))
                elif ck == "call_origin" and isinstance(cv, frozenset):
                    orig_tags.update(cv)
                elif ck == "code" and isinstance(cv, types.CodeType):
                    tkey = (id(cv), 0)
                    if tkey not in seen_target_ids:
                        seen_target_ids.add(tkey)
                        target_codes.append((cv, 0))
        for cv, bound_off in target_codes:
            cid_c = id(cv)
            prev_inv = invoked_local_codes.get(cid_c)
            prev_flag = prev_inv[1] if prev_inv is not None else False
            invoked_local_codes[cid_c] = (cv, prev_flag or is_tainted_call)
            if known_code_called_params and cid_c in known_code_called_params:
                for cp_kind, cp_key, cp_unp, cp_mode in known_code_called_params[cid_c]:
                    srcs_cp = _resolve_param_sources(cv, bound_off, cp_kind, cp_key)
                    if srcs_cp:
                        _record_called_params_from_slots(
                            srcs_cp,
                            is_unpacked=cp_unp,
                            taint_mode=(2 if cp_mode == 2 else (1 if cp_mode == 1 else 0)),
                        )
                    eff_taint_cp = (cp_mode == 2) or (cp_mode == 1 and is_tainted_call)
                    for arg_s in srcs_cp:
                        sub_inv_t = bool(eff_taint_cp or _slot_is_tainted(arg_s) or ("closure_taint", True) in arg_s)
                        for ak_i, av_i in arg_s:
                            sub_c_cp, _ = _lookup_ext_target(ak_i, av_i)
                            if sub_c_cp is not None:
                                scid = id(sub_c_cp)
                                prev_si = invoked_local_codes.get(scid)
                                prev_sf = prev_si[1] if prev_si is not None else False
                                invoked_local_codes[scid] = (sub_c_cp, prev_sf or sub_inv_t)
                        if (
                            eff_taint_cp
                            or _slot_is_tainted(arg_s)
                            or ("closure_taint", True) in arg_s
                            or ("iter_taint", True) in arg_s
                        ):
                            _record_ret_invoked([arg_s])
                            _activate_nested_slots(
                                [arg_s],
                                is_unpacked=cp_unp,
                                with_tainted_args=bool(eff_taint_cp or _slot_is_tainted(arg_s)),
                            )
                            if cp_unp:
                                _record(arg_s, unpacked=True)
                                _record(arg_s, unpacked=False)
                            elif eff_taint_cp:
                                _record(arg_s, unpacked=False)
                            if ("closure_taint", True) in arg_s:
                                for ak_c, av_c in arg_s:
                                    if ak_c == "name":
                                        called_closure_names.add(av_c)
            if known_code_returns and cid_c in known_code_returns:
                _apply_ret_fset(cv, bound_off, known_code_returns[cid_c], 0)
        if orig_tags:
            ret_slot.add(("call_origin", frozenset(orig_tags)))
        if allow_pos_arg_fwd and arg_slots_rev:
            first_arg = arg_slots_rev[-1]
            if is_partial:
                for pk, pv in first_arg:
                    if pk in ("code", "fn_def_consts", "fn_tainted_params", "closure_taint", "name", "attr", "leaf_attr", "call_origin"):
                        ret_slot.add((pk, pv))
            if is_seq_ctor and len(arg_slots_rev) == 1:
                seq_rep_fa = _unpack_seq_with_boxes(first_arg)
                if seq_rep_fa is not None:
                    _set_seq_repr(ret_slot, *seq_rep_fa)
                for sk, sv in first_arg:
                    if sk in ("recv_box", "recv_var"):
                        ret_slot.add((sk, sv))
                dyn_fa = _lookup_container_elems(first_arg)
                if dyn_fa:
                    for bid_f in _container_bids(ret_slot):
                        box_elem_tags.setdefault(bid_f, set()).update(dyn_fa)
        if is_dict_ctor:
            dict_boxes.add(ret_bid)
            if allow_pos_arg_fwd and len(arg_slots_rev) == 1:
                src_dc = arg_slots_rev[0]
                if any(sb in dict_boxes for sb in _container_bids(src_dc)):
                    _merge_dict_into(ret_slot, src_dc)
                else:
                    _record_dict_entry(ret_slot, None, src_dc)
            if kw_arg_map:
                for dk_c, dv_c in kw_arg_map.items():
                    _record_dict_entry(ret_slot, frozenset({dk_c}), dv_c)
            if unk_kw_slot:
                _record_dict_entry(ret_slot, None, unk_kw_slot)
        all_known_targets = (
            bool(target_codes)
            and known_code_returns is not None
            and all(id(cv_k) in known_code_returns for cv_k, _ in target_codes)
        )
        if attach_param_dep and not all_known_targets and not is_partial and not is_seq_ctor and not is_dict_ctor:
            call_arg_list: List[set[Tuple[str, Any]]] = list(arg_slots_rev) + list(callable_slots)
            if kw_arg_map:
                call_arg_list.extend(kw_arg_map.values())
            if unk_kw_slot:
                call_arg_list.append(unk_kw_slot)
            if star_pos_slot:
                call_arg_list.append(star_pos_slot)
            dep_p_tags = _collect_slot_param_tags(call_arg_list)
            if dep_p_tags:
                ret_slot.add(("param_dep", dep_p_tags))

    local_varnames_fset = frozenset(str(v) for v in code.co_varnames)

    def _strip_local_names_fset(
        sf: frozenset[Tuple[str, Any]],
        depth: int = 0,
    ) -> frozenset[Tuple[str, Any]]:
        out_s: set[Tuple[str, Any]] = set()
        for k_l, v_l in sf:
            if k_l == "name" and v_l in local_varnames_fset:
                continue
            if k_l == "seq_items" and isinstance(v_l, tuple) and depth < 8:
                out_s.add(("seq_items", tuple(_strip_local_names_fset(el, depth + 1) for el in v_l)))
            elif k_l == "seq_items_ex" and isinstance(v_l, tuple) and len(v_l) == 4 and depth < 8:
                out_s.add(
                    (
                        "seq_items_ex",
                        (
                            tuple(_strip_local_names_fset(el, depth + 1) for el in v_l[0]),
                            v_l[1],
                            _strip_local_names_fset(v_l[2], depth + 1),
                            tuple(_strip_local_names_fset(el, depth + 1) for el in v_l[3]),
                        ),
                    )
                )
            else:
                out_s.add((k_l, v_l))
        return frozenset(out_s)

    def _slots_have_param(slots: Sequence[set[Tuple[str, Any]]]) -> bool:
        return bool(_collect_slot_param_tags(slots))

    def _run_pass() -> None:
        stack: List[set[Tuple[str, Any]]] = []
        pending_kw_names: Optional[Tuple[str, ...]] = None
        for ins_idx, ins in enumerate(ins_list):
            op = ins.opname
            arg_int = ins.arg if isinstance(ins.arg, int) else 0
            cur_kw_names: Optional[Tuple[str, ...]] = None
            if op != "EXTENDED_ARG":
                cur_kw_names = pending_kw_names
                pending_kw_names = None
            if op == "KW_NAMES":
                if isinstance(ins.argval, tuple) and all(isinstance(x, str) for x in ins.argval):
                    pending_kw_names = ins.argval
                continue
            if op == "PUSH_NULL":
                stack.append({("null", None)})
                continue
            if op == "LOAD_GLOBAL":
                if arg_int & 1:
                    stack.append({("null", None)})
                stack.append({("name", str(ins.argval))})
                continue
            if op == "LOAD_CONST":
                c_slot: set[Tuple[str, Any]] = set()
                if isinstance(ins.argval, types.CodeType):
                    c_slot.add(("code", ins.argval))
                else:
                    if ins.argval is None or isinstance(ins.argval, (str, int, float, bool)):
                        c_slot.add(("const_key", ins.argval))
                    elif isinstance(ins.argval, slice):
                        c_slot.add(("is_slice", True))
                    if isinstance(ins.argval, str):
                        c_slot.add(("raw_str", ins.argval))
                    elif isinstance(ins.argval, int) and not isinstance(ins.argval, bool):
                        c_slot.add(("raw_int", ins.argval))
                    elif isinstance(ins.argval, tuple) and all(isinstance(x, str) for x in ins.argval):
                        c_slot.add(("raw_key_tuple", ins.argval))
                    if candidate_keys:
                        c_strs: set[str] = set()
                        _collect_val_consts(ins.argval, c_strs)
                        if c_strs:
                            c_slot.add(("consts", frozenset(c_strs)))
                stack.append(c_slot)
                continue
            if op == "LOAD_SMALL_INT":
                raw_i = ins.argval if isinstance(ins.argval, int) and not isinstance(ins.argval, bool) else arg_int
                stack.append({("raw_int", int(raw_i)), ("const_key", int(raw_i))})
                continue
            if op == "RETURN_CONST":
                if isinstance(ins.argval, types.CodeType):
                    returned_slots.add(("code", ins.argval))
                elif candidate_keys:
                    _collect_val_consts(ins.argval, active_consts)
                continue
            if op == "BUILD_SLICE":
                popped_bs = [stack.pop() if stack else set() for _ in range(max(0, arg_int))]
                _flush_consts(*popped_bs)
                bs_slot: set[Tuple[str, Any]] = {("is_slice", True)}
                if any(_slot_is_tainted(s) for s in popped_bs):
                    bs_slot.add(("kw_taint", True))
                stack.append(bs_slot)
                continue
            if op in (
                "LOAD_NAME",
                "LOAD_DEREF",
                "LOAD_CLOSURE",
                "LOAD_FAST",
                "LOAD_FAST_BORROW",
                "LOAD_FAST_CHECK",
                "LOAD_FAST_AND_CLEAR",
            ):
                stack.append(_load_var_slot(str(ins.argval)))
                continue
            if op in ("LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW_LOAD_FAST_BORROW") and (
                isinstance(ins.argval, tuple) and len(ins.argval) == 2
            ):
                for vn in ins.argval:
                    stack.append(_load_var_slot(str(vn)))
                continue
            if op == "LOAD_ATTR":
                base = stack.pop() if stack else set()
                _flush_consts(base)
                attr_name = str(ins.argval)
                derived: set[Tuple[str, Any]] = {("leaf_attr", attr_name)}
                for kind, val in base:
                    if kind == "name":
                        derived.add(("attr", (val, attr_name)))
                        derived.add(("recv_var", val))
                    elif kind == "attr":
                        derived.add(("attr", (val[1], attr_name)))
                    elif kind == "box_id":
                        derived.add(("recv_box", val))
                if _slot_is_tainted(base):
                    derived.add(("kw_taint", True))
                if arg_int & 1:
                    stack.append(derived)
                    stack.append({("null", None)})
                else:
                    stack.append(derived)
                continue
            if op == "LOAD_SUPER_ATTR":
                # Pops self, class, super; pushes method (+ optional self/NULL).
                base = stack[-1] if stack else set()
                eff = _safe_stack_effect(ins, default=-2)
                pops = 3
                for _ in range(pops):
                    if stack:
                        _flush_consts(stack.pop())
                push_cnt = eff + pops
                derived_sup: set[Tuple[str, Any]] = {("leaf_attr", str(ins.argval)), ("attr", ("super", str(ins.argval)))}
                for kind, val in base:
                    if kind == "name":
                        derived_sup.add(("attr", (val, str(ins.argval))))
                if _slot_is_tainted(base):
                    derived_sup.add(("kw_taint", True))
                for j in range(max(0, push_cnt)):
                    stack.append(derived_sup if j == 0 else {("null", None)})
                continue
            if op == "MAKE_FUNCTION":
                pops = (
                    1
                    if ins.arg is None
                    else max(
                        1 + sum(1 for r_bit in (0x01, 0x02, 0x04, 0x08) if (arg_int & r_bit)),
                        1 - _safe_stack_effect(ins, default=0),
                    )
                )
                popped_mf = [stack.pop() if stack else set() for _ in range(pops)]
                sub_code_mf = next(
                    (
                        v
                        for k, v in (popped_mf[0] if popped_mf else ())
                        if k == "code" and isinstance(v, types.CodeType)
                    ),
                    None,
                )
                slot_roles: Dict[int, int] = {}
                if ins.arg is not None and len(popped_mf) > 1:
                    cur_r = len(popped_mf) - 1
                    for r_bit in (0x01, 0x02, 0x04, 0x08):
                        if (arg_int & r_bit) and cur_r >= 1:
                            slot_roles[cur_r] = r_bit
                            cur_r -= 1
                fn_slot: set[Tuple[str, Any]] = set()
                def_c: set[str] = set()
                ign_c: set[str] = set()
                keep_val_c: set[str] = set()
                tp_names: set[str] = set()
                mf_p_deps: set[Tuple[str, Any]] = set()
                has_ct = False
                for mf_idx, sm in enumerate(popped_mf):
                    r_bit = slot_roles.get(mf_idx, 0)
                    if r_bit not in (0x04, 0x10):
                        mf_p_deps.update(_collect_slot_param_tags([sm]))
                    if r_bit == 0x01:
                        tp_names.update(_extract_pos_def_tainted_params(sub_code_mf, sm))
                    elif r_bit == 0x02:
                        tp_names.update(_extract_kw_def_tainted_params(sub_code_mf, sm))
                    for kind, val in sm:
                        if kind == "code" and mf_idx == 0:
                            fn_slot.add(("code", val))
                        elif kind == "consts" and val:
                            def_c.update(val)
                            if r_bit in (0x02, 0x04, 0x10):
                                ign_c.update(val)
                            else:
                                keep_val_c.update(val)
                        elif kind == "map_val_consts" and val and r_bit == 0x02:
                            keep_val_c.update(val)
                        elif r_bit in (0x04, 0x10):
                            continue
                        elif kind == "fn_def_consts" and val:
                            def_c.update(val)
                            keep_val_c.update(val)
                        elif kind == "fn_tainted_params" and val:
                            tp_names.update(val)
                        elif kind in ("closure_taint", "kw_taint"):
                            has_ct = True
                        elif kind == "box_id" and val in tainted_boxes:
                            has_ct = True
                clean_def_c = (def_c - ign_c) | keep_val_c
                if clean_def_c:
                    fn_slot.add(("fn_def_consts", frozenset(clean_def_c)))
                if tp_names:
                    tp_fset = frozenset(tp_names)
                    fn_slot.add(("fn_tainted_params", tp_fset))
                    has_ct = True
                    if isinstance(sub_code_mf, types.CodeType):
                        def_tainted_params[id(sub_code_mf)] = (
                            def_tainted_params.get(id(sub_code_mf), frozenset()) | tp_fset
                        )
                if has_ct:
                    fn_slot.add(("closure_taint", True))
                if mf_p_deps:
                    fn_slot.add(("param_dep", frozenset(mf_p_deps)))
                stack.append(fn_slot)
                continue
            if op == "SET_FUNCTION_ATTRIBUTE":
                fn_slot = stack.pop() if stack else set()
                attr_slot = stack.pop() if stack else set()
                new_fn_slot = set(fn_slot)
                if arg_int in (0x04, 0x10):
                    stack.append(new_fn_slot)
                    continue
                sub_code_sfa = next(
                    (v for k, v in new_fn_slot if k == "code" and isinstance(v, types.CodeType)),
                    None,
                )
                if arg_int == 0x02:
                    mvc_parts = [v for k, v in attr_slot if k == "map_val_consts" and v]
                    attr_c = frozenset().union(*mvc_parts) if mvc_parts else frozenset()
                else:
                    attr_c = _slot_consts(attr_slot)
                prev_dc = [v for k, v in new_fn_slot if k == "fn_def_consts" and v]
                ext_dc = [v for k, v in attr_slot if k == "fn_def_consts" and v]
                merged_dc = (
                    attr_c.union(*prev_dc, *ext_dc) if (attr_c or prev_dc or ext_dc) else frozenset()
                )
                if merged_dc:
                    new_fn_slot = {kv for kv in new_fn_slot if kv[0] != "fn_def_consts"}
                    new_fn_slot.add(("fn_def_consts", merged_dc))
                new_tp = (
                    _extract_pos_def_tainted_params(sub_code_sfa, attr_slot)
                    if arg_int == 0x01
                    else (
                        _extract_kw_def_tainted_params(sub_code_sfa, attr_slot)
                        if arg_int == 0x02
                        else frozenset()
                    )
                )
                prev_tp = [v for k, v in new_fn_slot if k == "fn_tainted_params" and v]
                merged_tp = new_tp.union(*prev_tp) if (new_tp or prev_tp) else frozenset()
                if merged_tp:
                    new_fn_slot = {kv for kv in new_fn_slot if kv[0] != "fn_tainted_params"}
                    new_fn_slot.add(("fn_tainted_params", merged_tp))
                    new_fn_slot.add(("closure_taint", True))
                    if isinstance(sub_code_sfa, types.CodeType):
                        def_tainted_params[id(sub_code_sfa)] = (
                            def_tainted_params.get(id(sub_code_sfa), frozenset()) | merged_tp
                        )
                if _slot_is_tainted(attr_slot) or ("closure_taint", True) in attr_slot:
                    new_fn_slot.add(("closure_taint", True))
                sfa_p_deps = _collect_slot_param_tags([attr_slot])
                if sfa_p_deps:
                    prev_pd = frozenset().union(*(v for k, v in new_fn_slot if k == "param_dep" and v))
                    new_fn_slot = {kv for kv in new_fn_slot if kv[0] != "param_dep"}
                    new_fn_slot.add(("param_dep", prev_pd | sfa_p_deps))
                stack.append(new_fn_slot)
                continue
            if op in ("STORE_FAST", "STORE_NAME", "STORE_DEREF"):
                top = stack.pop() if stack else set()
                _flush_consts(top)
                ent_sf = locals_env.setdefault(str(ins.argval), set())
                ent_sf.update({kv for kv in top if kv[0] not in _EPHEMERAL_SLOT_KEYS})
                if not any(k == "const_key" for k, _ in top):
                    ent_sf.add(("unk_key", True))
                continue
            if op == "STORE_FAST_STORE_FAST" and isinstance(ins.argval, tuple) and len(ins.argval) == 2:
                t1 = stack.pop() if stack else set()
                t2 = stack.pop() if stack else set()
                _flush_consts(t1, t2)
                e1 = locals_env.setdefault(str(ins.argval[0]), set())
                e1.update({kv for kv in t1 if kv[0] not in _EPHEMERAL_SLOT_KEYS})
                if not any(k == "const_key" for k, _ in t1):
                    e1.add(("unk_key", True))
                e2 = locals_env.setdefault(str(ins.argval[1]), set())
                e2.update({kv for kv in t2 if kv[0] not in _EPHEMERAL_SLOT_KEYS})
                if not any(k == "const_key" for k, _ in t2):
                    e2.add(("unk_key", True))
                continue
            if op == "STORE_FAST_LOAD_FAST" and isinstance(ins.argval, tuple) and len(ins.argval) == 2:
                t1 = stack.pop() if stack else set()
                _flush_consts(t1)
                e1 = locals_env.setdefault(str(ins.argval[0]), set())
                e1.update({kv for kv in t1 if kv[0] not in _EPHEMERAL_SLOT_KEYS})
                if not any(k == "const_key" for k, _ in t1):
                    e1.add(("unk_key", True))
                stack.append(_load_var_slot(str(ins.argval[1])))
                continue
            if op == "SWAP":
                if 1 <= arg_int <= len(stack):
                    stack[-1], stack[-arg_int] = stack[-arg_int], stack[-1]
                continue
            if op == "COPY":
                stack.append(set(stack[-arg_int]) if 1 <= arg_int <= len(stack) else set())
                continue
            if op in ("BUILD_TUPLE", "BUILD_LIST", "BUILD_SET"):
                popped_seq = [stack.pop() if stack else set() for _ in range(arg_int)]
                elems_fwd = list(reversed(popped_seq))
                res_seq: set[Tuple[str, Any]] = set()
                if op in ("BUILD_LIST", "BUILD_SET"):
                    bid = ins_idx + 1
                    res_seq.add(("box_id", bid))
                    if bid in tainted_boxes or any(
                        _slot_is_tainted(s) or ("closure_taint", True) in s for s in popped_seq
                    ):
                        tainted_boxes.add(bid)
                        res_seq.add(("kw_taint", True))
                elif any(_slot_is_tainted(s) or ("closure_taint", True) in s for s in popped_seq):
                    res_seq.add(("kw_taint", True))
                seq_c = frozenset().union(*(_slot_consts(s) for s in popped_seq)) if popped_seq else frozenset()
                if seq_c:
                    res_seq.add(("consts", seq_c))
                if op == "BUILD_TUPLE":
                    res_seq.add(("tuple_len", arg_int))
                    t_idxs = frozenset(
                        i
                        for i, s in enumerate(elems_fwd)
                        if _slot_is_tainted(s) or ("closure_taint", True) in s
                    )
                    if t_idxs:
                        res_seq.add(("tuple_tainted_idxs", t_idxs))
                    if elems_fwd:
                        first_arg_tags = frozenset(
                            kv for kv in elems_fwd[0] if kv[0] in ("name", "attr", "leaf_attr")
                        )
                        if first_arg_tags:
                            res_seq.add(("first_arg_slot", first_arg_tags))
                if op in ("BUILD_TUPLE", "BUILD_LIST"):
                    res_seq.add(
                        (
                            "seq_items",
                            tuple(_freeze_elem_slot(s) for s in elems_fwd),
                        )
                    )
                if op == "BUILD_LIST":
                    for el_bl in elems_fwd:
                        el_bl_f = _freeze_elem_slot(el_bl)
                        if el_bl_f:
                            box_elem_tags.setdefault(bid, set()).update(el_bl_f)
                for s in popped_seq:
                    for k, v in s:
                        if k in ("code", "fn_def_consts", "fn_tainted_params", "closure_taint"):
                            res_seq.add((k, v))
                        elif k in ("box_id", "recv_box"):
                            res_seq.add(("recv_box", v))
                        elif k in ("name", "recv_var") and v not in ("self", "cls"):
                            res_seq.add(("recv_var", v))
                stack.append(res_seq)
                continue
            if op == "BUILD_MAP":
                bid = ins_idx + 1
                dict_boxes.add(bid)
                popped_bm = [stack.pop() if stack else set() for _ in range(2 * arg_int)]
                vals_bm = popped_bm[0::2]
                keys_bm = popped_bm[1::2]
                had_t = any(_slot_is_tainted(s) for s in popped_bm)
                bm_c = frozenset().union(*(_slot_consts(s) for s in popped_bm)) if popped_bm else frozenset()
                val_c = frozenset().union(*(_slot_consts(s) for s in vals_bm)) if vals_bm else frozenset()
                t_keys: set[str] = set()
                unk_kt = False
                has_val_ct = False
                for k_s, v_s in zip(keys_bm, vals_bm):
                    if ("closure_taint", True) in v_s:
                        has_val_ct = True
                    if _slot_is_tainted(v_s) or ("closure_taint", True) in v_s:
                        k_str = next((v for k, v in k_s if k == "raw_str"), None)
                        if isinstance(k_str, str):
                            t_keys.add(k_str)
                        else:
                            unk_kt = True
                res_map: set[Tuple[str, Any]] = {("box_id", bid)}
                if bid in unpacked_boxes:
                    res_map.add(("kw_unpack", None))
                else:
                    res_map.add(("lit_map", None))
                if had_t or bid in tainted_boxes:
                    tainted_boxes.add(bid)
                    res_map.add(("kw_taint", True))
                if has_val_ct:
                    res_map.add(("closure_taint", True))
                if bm_c:
                    res_map.add(("consts", bm_c))
                if val_c:
                    res_map.add(("map_val_consts", val_c))
                if t_keys:
                    res_map.add(("map_tainted_keys", frozenset(t_keys)))
                if unk_kt:
                    res_map.add(("map_unk_key_tainted", True))
                for k_s_bm, v_s_bm in zip(keys_bm, vals_bm):
                    _record_dict_entry(res_map, _extract_const_keys(k_s_bm), v_s_bm)
                stack.append(res_map)
                continue
            if op == "BUILD_CONST_KEY_MAP":
                bid = ins_idx + 1
                dict_boxes.add(bid)
                popped_ckm = [stack.pop() if stack else set() for _ in range(arg_int + 1)]
                keys_slot = popped_ckm[0] if popped_ckm else set()
                vals_fwd = list(reversed(popped_ckm[1:]))
                had_t = any(_slot_is_tainted(s) for s in popped_ckm)
                ckm_c = frozenset().union(*(_slot_consts(s) for s in popped_ckm)) if popped_ckm else frozenset()
                val_c = frozenset().union(*(_slot_consts(s) for s in vals_fwd)) if vals_fwd else frozenset()
                key_tup = next((v for k, v in keys_slot if k == "raw_key_tuple"), None)
                t_keys = set()
                unk_kt = False
                has_val_ct = False
                for idx_v, v_s in enumerate(vals_fwd):
                    if ("closure_taint", True) in v_s:
                        has_val_ct = True
                    if _slot_is_tainted(v_s) or ("closure_taint", True) in v_s:
                        if isinstance(key_tup, tuple) and idx_v < len(key_tup):
                            t_keys.add(key_tup[idx_v])
                        else:
                            unk_kt = True
                res_ckm: set[Tuple[str, Any]] = {("box_id", bid)}
                if bid in unpacked_boxes:
                    res_ckm.add(("kw_unpack", None))
                else:
                    res_ckm.add(("lit_map", None))
                if had_t or bid in tainted_boxes:
                    tainted_boxes.add(bid)
                    res_ckm.add(("kw_taint", True))
                if has_val_ct:
                    res_ckm.add(("closure_taint", True))
                if ckm_c:
                    res_ckm.add(("consts", ckm_c))
                if val_c:
                    res_ckm.add(("map_val_consts", val_c))
                if t_keys:
                    res_ckm.add(("map_tainted_keys", frozenset(t_keys)))
                if unk_kt:
                    res_ckm.add(("map_unk_key_tainted", True))
                for idx_v, v_s_ckm in enumerate(vals_fwd):
                    ck_set = (
                        frozenset({key_tup[idx_v]})
                        if isinstance(key_tup, tuple) and idx_v < len(key_tup)
                        else None
                    )
                    _record_dict_entry(res_ckm, ck_set, v_s_ckm)
                stack.append(res_ckm)
                continue
            if op in ("DICT_MERGE", "DICT_UPDATE"):
                src_map = stack.pop() if stack else set()
                _flush_consts(src_map)
                if 1 <= arg_int <= len(stack):
                    dst_map = stack[-arg_int]
                    _flush_consts(dst_map)
                    dst_map.discard(("lit_map", None))
                    dst_map.discard(("null", None))
                    dst_map.add(("kw_unpack", None))
                    if _slot_is_tainted(src_map):
                        dst_map.add(("kw_taint", True))
                        _taint_receiver_slot(dst_map)
                    _merge_dict_into(dst_map, src_map)
                continue
            if op == "MAP_ADD":
                val_s = stack.pop() if stack else set()
                key_s = stack.pop() if stack else set()
                _flush_consts(val_s, key_s)
                if 1 <= arg_int <= len(stack):
                    dst_map = stack[-arg_int]
                    if _slot_is_tainted(val_s) or _slot_is_tainted(key_s):
                        dst_map.discard(("lit_map", None))
                        dst_map.add(("kw_taint", True))
                        _taint_receiver_slot(dst_map)
                    _record_dict_entry(dst_map, _extract_const_keys(key_s), val_s)
                continue
            if op in ("LIST_APPEND", "SET_ADD", "LIST_EXTEND", "SET_UPDATE"):
                item_s = stack.pop() if stack else set()
                _flush_consts(item_s)
                if 1 <= arg_int <= len(stack) and _slot_is_tainted(item_s):
                    dst_cont = stack[-arg_int]
                    dst_cont.add(("kw_taint", True))
                    _taint_receiver_slot(dst_cont)
                if op in ("LIST_APPEND", "LIST_EXTEND") and 1 <= arg_int <= len(stack):
                    dst_seq_s = stack[-arg_int]
                    dst_rep = _unpack_seq_repr(dst_seq_s)
                    if dst_rep is not None:
                        d_pref, d_hs, d_se, d_suff = dst_rep
                        if op == "LIST_APPEND":
                            el_fr = _freeze_elem_slot(item_s)
                            if not d_hs:
                                _set_seq_repr(dst_seq_s, d_pref + (el_fr,), False, frozenset(), ())
                            else:
                                _set_seq_repr(dst_seq_s, d_pref, True, d_se, d_suff + (el_fr,))
                        else:
                            it_rep = _unpack_seq_with_boxes(item_s)
                            if it_rep is not None:
                                m_pref, m_hs, m_se, m_suff = _concat_seq_reps(dst_rep, it_rep)
                                _set_seq_repr(dst_seq_s, m_pref, m_hs, m_se, m_suff)
                            else:
                                it_tag = frozenset({("kw_taint", True)}) if _slot_is_tainted(item_s) else frozenset()
                                if not d_hs:
                                    _set_seq_repr(dst_seq_s, d_pref, True, it_tag, ())
                                else:
                                    _set_seq_repr(dst_seq_s, d_pref, True, d_se.union(it_tag, *d_suff), ())
                continue
            if op == "STORE_SLICE":
                stop_s = stack.pop() if stack else set()
                start_s = stack.pop() if stack else set()
                cont_s = stack.pop() if stack else set()
                val_s = stack.pop() if stack else set()
                _flush_consts(stop_s, start_s, cont_s, val_s)
                if _slot_is_tainted(val_s) or _slot_is_tainted(start_s) or _slot_is_tainted(stop_s):
                    _taint_receiver_slot(cont_s)
                rep_ss = _unpack_seq_with_boxes(val_s)
                sl_elems_ss = (
                    set(_collect_rep_elem_tags(rep_ss))
                    if rep_ss is not None
                    else (set(_lookup_container_elems(val_s)) or set(_freeze_elem_slot(val_s)))
                )
                _record_container_elem(cont_s, sl_elems_ss, seq_mut=True)
                continue
            if op == "STORE_SUBSCR":
                key_s = stack.pop() if stack else set()
                cont_s = stack.pop() if stack else set()
                val_s = stack.pop() if stack else set()
                _flush_consts(key_s, cont_s, val_s)
                if _slot_is_tainted(val_s) or _slot_is_tainted(key_s):
                    _taint_receiver_slot(cont_s)
                if ("is_slice", True) in key_s:
                    rep_ss = _unpack_seq_with_boxes(val_s)
                    sl_elems_ss = (
                        set(_collect_rep_elem_tags(rep_ss))
                        if rep_ss is not None
                        else (set(_lookup_container_elems(val_s)) or set(_freeze_elem_slot(val_s)))
                    )
                    _record_container_elem(cont_s, sl_elems_ss, seq_mut=True)
                else:
                    bids_ss = _container_bids(cont_s)
                    if bids_ss:
                        ck_ss = _extract_const_keys(key_s)
                        for b_ss in bids_ss:
                            if b_ss in dict_boxes:
                                _record_dict_entry({("box_id", b_ss)}, ck_ss, val_s)
                            else:
                                _record_container_elem({("box_id", b_ss)}, val_s, seq_mut=True)
                    else:
                        _record_container_elem(cont_s, val_s, seq_mut=True)
                continue
            if op == "GET_ITER":
                if stack:
                    top_it = stack[-1]
                    _flush_consts(top_it)
                    if _slot_is_tainted(top_it):
                        stack[-1] = {("kw_taint", True), ("iter_taint", True)}
                    else:
                        stack[-1] = set()
                continue
            if op == "FOR_ITER":
                iter_s = stack[-1] if stack else set()
                stack.append({("kw_taint", True)} if _slot_is_tainted(iter_s) else set())
                continue
            if op == "UNPACK_SEQUENCE":
                top = stack.pop() if stack else set()
                _flush_consts(top)
                has_t = _slot_is_tainted(top)
                box_t = any(k == "box_id" and v in tainted_boxes for k, v in top)
                seq_rep = _unpack_seq_with_boxes(top)
                pushed_u: Optional[List[set[Tuple[str, Any]]]] = None
                if seq_rep is not None:
                    pref_u, hs_u, se_u, suff_u = seq_rep
                    if not hs_u and len(pref_u) == arg_int:
                        pushed_u = [set(el) for el in pref_u]
                    elif hs_u:
                        pushed_u = []
                        star_and_suff_u = se_u.union(*suff_u) if suff_u else se_u
                        star_and_pref_u = se_u.union(*pref_u) if pref_u else se_u
                        for idx_u in range(arg_int):
                            dist_end_u = arg_int - idx_u
                            if idx_u < len(pref_u):
                                pushed_u.append(set(pref_u[idx_u]))
                            elif dist_end_u <= len(suff_u) and idx_u >= len(pref_u):
                                cand_u = set(suff_u[len(suff_u) - dist_end_u])
                                if idx_u == len(pref_u):
                                    cand_u.update(star_and_suff_u)
                                else:
                                    cand_u.update(se_u)
                                pushed_u.append(cand_u)
                            elif dist_end_u > len(suff_u):
                                pushed_u.append(set(star_and_pref_u | star_and_suff_u))
                            else:
                                pushed_u.append(set(star_and_suff_u))
                if pushed_u is not None:
                    for el_s in reversed(pushed_u):
                        if box_t or _slot_is_tainted(el_s):
                            el_s.add(("kw_taint", True))
                        stack.append(el_s)
                else:
                    recv_fb = {kv for kv in top if kv[0] in ("recv_box", "recv_var", "call_origin")}
                    recv_fb.update(_lookup_container_elems(top))
                    for _ in range(arg_int):
                        fb_s = set(recv_fb)
                        if has_t:
                            fb_s.add(("kw_taint", True))
                        stack.append(fb_s)
                continue
            if op == "UNPACK_EX":
                top = stack.pop() if stack else set()
                _flush_consts(top)
                before = arg_int & 0xFF
                after = arg_int >> 8
                has_t = _slot_is_tainted(top)
                box_t = any(k == "box_id" and v in tainted_boxes for k, v in top)
                seq_rep = _unpack_seq_with_boxes(top)
                pushed_ex: List[set[Tuple[str, Any]]] = []
                if seq_rep is not None:
                    pref_x, hs_x, se_x, suff_x = seq_rep
                    if not hs_x and len(pref_x) >= before + after:
                        before_slots = [set(pref_x[i]) for i in range(before)]
                        star_mid = pref_x[before : len(pref_x) - after]
                        after_slots = [set(pref_x[len(pref_x) - after + j]) for j in range(after)]
                        star_s: set[Tuple[str, Any]] = set()
                        if star_mid:
                            star_s.add(("seq_items", star_mid))
                        if box_t or any(
                            _slot_is_tainted(set(x)) or ("closure_taint", True) in x
                            for x in star_mid
                        ):
                            star_s.add(("kw_taint", True))
                        pushed_ex = before_slots + [star_s] + after_slots
                        for el_s in pushed_ex:
                            if box_t or _slot_is_tainted(el_s):
                                el_s.add(("kw_taint", True))
                    elif hs_x:
                        star_extra_parts: List[frozenset[Tuple[str, Any]]] = [se_x]
                        star_plus_suff_x = se_x.union(*suff_x) if suff_x else se_x
                        star_plus_pref_x = se_x.union(*pref_x) if pref_x else se_x
                        before_slots = [
                            set(pref_x[i]) if i < len(pref_x) else set(star_plus_suff_x)
                            for i in range(before)
                        ]
                        if len(pref_x) > before:
                            star_extra_parts.extend(pref_x[before:])
                        after_slots = []
                        for j in range(after):
                            dist_end = after - j
                            if dist_end <= len(suff_x):
                                after_slots.append(set(suff_x[len(suff_x) - dist_end]))
                            else:
                                after_slots.append(set(star_plus_pref_x))
                        if len(suff_x) > after:
                            star_extra_parts.extend(suff_x[: len(suff_x) - after])
                        merged_star_el = frozenset().union(*star_extra_parts)
                        star_s = set()
                        if merged_star_el or has_t:
                            rem_p = pref_x[before:] if len(pref_x) > before else ()
                            rem_s = suff_x[: len(suff_x) - after] if len(suff_x) > after else ()
                            star_s.add(("seq_items_ex", (rem_p, True, se_x, rem_s)))
                        if box_t or has_t or _slot_is_tainted(set(merged_star_el)):
                            star_s.add(("kw_taint", True))
                        pushed_ex = before_slots + [star_s] + after_slots
                        for el_s in pushed_ex:
                            if box_t or _slot_is_tainted(el_s):
                                el_s.add(("kw_taint", True))
                if not pushed_ex:
                    n_push = before + 1 + after
                    recv_fb_ex = {kv for kv in top if kv[0] in ("recv_box", "recv_var", "call_origin")}
                    recv_fb_ex.update(_lookup_container_elems(top))
                    pushed_ex = []
                    for _ in range(n_push):
                        fb_ex = set(recv_fb_ex)
                        if has_t:
                            fb_ex.add(("kw_taint", True))
                        pushed_ex.append(fb_ex)
                for el_s in reversed(pushed_ex):
                    stack.append(el_s)
                continue
            if op == "LIST_TO_TUPLE" or (op == "CALL_INTRINSIC_1" and arg_int == 6):
                top_lt = stack.pop() if stack else set()
                _flush_consts(top_lt)
                tup_rep = _unpack_seq_with_boxes(top_lt)
                res_lt: set[Tuple[str, Any]] = {
                    kv
                    for kv in top_lt
                    if kv[0] not in ("box_id", "lit_map", "seq_items", "seq_items_ex")
                    and kv[0] not in _EPHEMERAL_SLOT_KEYS
                }
                if _slot_is_tainted(top_lt):
                    res_lt.add(("kw_taint", True))
                if tup_rep is not None:
                    _set_seq_repr(res_lt, *tup_rep)
                else:
                    res_lt.update(_lookup_container_elems(top_lt))
                stack.append(res_lt)
                continue
            if op == "CALL_FUNCTION_EX":
                eff = _safe_stack_effect(
                    ins,
                    default=(-3 if ins.arg is None else (-2 - (1 if (arg_int & 1) else 0))),
                )
                total_pop = max(0, 1 - eff)
                popped = [stack.pop() if stack else set() for _ in range(total_pop)]
                _flush_consts(*popped)
                if ins.arg is None or (arg_int & 1):
                    kw_slot = popped[0] if len(popped) >= 1 else set()
                    pos_slot = popped[1] if len(popped) >= 2 else set()
                    callable_slots = popped[2:]
                    is_plain_lit_map = (
                        ("lit_map", None) in kw_slot
                        and not _slot_is_tainted(kw_slot)
                        and not any(
                            k == "kw_unpack" or (k == "box_id" and v in unpacked_boxes)
                            for k, v in kw_slot
                        )
                    )
                    has_kw_map = (
                        bool(kw_slot)
                        and ("null", None) not in kw_slot
                        and not is_plain_lit_map
                    )
                else:
                    kw_slot = set()
                    pos_slot = popped[0] if len(popped) >= 1 else set()
                    callable_slots = popped[1:]
                    has_kw_map = False
                kw_tainted = _slot_is_tainted(kw_slot)
                pos_tainted = _slot_is_tainted(pos_slot)
                closure_tainted = any(
                    ("closure_taint", True) in s or ("iter_taint", True) in s
                    for s in callable_slots
                )
                recv_tainted = any(_slot_is_tainted(s) for s in callable_slots)
                any_tainted = kw_tainted or pos_tainted or closure_tainted or recv_tainted
                for slot in callable_slots:
                    if has_kw_map and kw_tainted:
                        _record(slot, unpacked=True)
                        _record(slot, unpacked=False)
                    elif kw_tainted or pos_tainted:
                        _record(slot, unpacked=False)
                if kw_tainted or pos_tainted:
                    is_mut_dict_call_ex = any(
                        (k == "leaf_attr" and v in _mut_dict_methods)
                        or (k == "name" and v in _mut_dict_methods)
                        for s in callable_slots
                        for k, v in s
                    )
                    if is_mut_dict_call_ex:
                        for s in callable_slots:
                            _taint_receiver_slot(s)
                        pos_seq = next(
                            (
                                v
                                for k, v in pos_slot
                                if k == "seq_items" and isinstance(v, tuple) and len(v) >= 1
                            ),
                            None,
                        )
                        if pos_seq is not None:
                            _taint_receiver_slot(set(pos_seq[0]))
                        else:
                            _taint_receiver_slot(pos_slot)
                if (
                    has_kw_map
                    and kw_tainted
                    and any(
                        (k == "name" and v == "partial") or (k == "leaf_attr" and v == "partial")
                        for s in callable_slots
                        for k, v in s
                    )
                ):
                    fa = next((v for k, v in pos_slot if k == "first_arg_slot"), None)
                    if fa:
                        _record(set(fa), unpacked=True)
                        _record(set(fa), unpacked=False)
                if closure_tainted:
                    for s in callable_slots:
                        if ("closure_taint", True) in s:
                            for k, v in s:
                                if k == "name":
                                    called_closure_names.add(v)
                if any_tainted:
                    _record_ret_invoked(callable_slots)
                    _activate_nested_slots(
                        callable_slots + [pos_slot, kw_slot],
                        is_unpacked=bool(has_kw_map and kw_tainted),
                        with_tainted_args=bool(kw_tainted or pos_tainted),
                    )
                ex_kw_map: Dict[str, set[Tuple[str, Any]]] = {}
                ex_unk_kw: set[Tuple[str, Any]] = set()
                if kw_slot and ("null", None) not in kw_slot:
                    kw_bids_ex = _container_bids(kw_slot)
                    has_known_dict_kw = any(bk in dict_boxes for bk in kw_bids_ex)
                    for bk in kw_bids_ex:
                        if bk in dict_boxes:
                            for dk_e, dv_e in box_key_elems.get(bk, {}).items():
                                if isinstance(dk_e, str):
                                    dst_dk = ex_kw_map.setdefault(dk_e, set())
                                    dst_dk.update(dv_e)
                                    if dk_e in box_tainted_keys.get(bk, ()):
                                        dst_dk.add(("kw_taint", True))
                                elif dv_e or dk_e in box_tainted_keys.get(bk, ()):
                                    ex_unk_kw.update(dv_e)
                                    if dk_e in box_tainted_keys.get(bk, ()):
                                        ex_unk_kw.add(("kw_taint", True))
                            if box_unk_key_elems.get(bk):
                                ex_unk_kw.update(box_unk_key_elems[bk])
                            if bk in box_unk_tainted:
                                ex_unk_kw.add(("kw_taint", True))
                        elif has_kw_map:
                            if bk in box_elem_tags:
                                ex_unk_kw.update(box_elem_tags[bk])
                            if bk in tainted_boxes:
                                ex_unk_kw.add(("kw_taint", True))
                    if not has_known_dict_kw and has_kw_map:
                        for k_uk, v_uk in _freeze_elem_slot(kw_slot):
                            ex_unk_kw.add(("varkw_any", None) if k_uk == "varkw_param" else (k_uk, v_uk))
                        if kw_tainted:
                            ex_unk_kw.add(("kw_taint", True))
                pos_rep_ex = _unpack_seq_with_boxes(pos_slot) if (pos_slot and ("null", None) not in pos_slot) else None
                if pos_rep_ex is not None and not pos_rep_ex[1]:
                    ex_pos_rev: Sequence[set[Tuple[str, Any]]] = [set(el_f) for el_f in reversed(pos_rep_ex[0])]
                    ex_allow_pos = True
                    ex_star_pos: Optional[set[Tuple[str, Any]]] = None
                else:
                    ex_pos_rev = ()
                    ex_allow_pos = False
                    if pos_slot and ("null", None) not in pos_slot:
                        raw_sp = (
                            _collect_rep_elem_tags(pos_rep_ex)
                            if pos_rep_ex is not None
                            else (_lookup_container_elems(pos_slot) or _freeze_elem_slot(pos_slot))
                        )
                        ex_star_pos = {
                            ("varargs_from", v_sp) if k_sp == "varargs_param" else (k_sp, v_sp)
                            for k_sp, v_sp in raw_sp
                        }
                        if pos_tainted:
                            ex_star_pos.add(("kw_taint", True))
                    else:
                        ex_star_pos = None
                ex_taint_mode = 2 if any_tainted else (1 if _slots_have_param([pos_slot, kw_slot]) else 0)
                _record_called_params_from_slots(
                    callable_slots,
                    is_unpacked=bool(has_kw_map),
                    taint_mode=ex_taint_mode,
                )
                ret_bid = ins_idx + 1
                ret_slot: set[Tuple[str, Any]] = {("box_id", ret_bid)}
                c_p_tags_ex = {
                    (k_p, v_p)
                    for cs in callable_slots
                    for k_p, v_p in cs
                    if k_p in ("param_idx", "param_name", "varargs_from", "varkw_any")
                }
                if c_p_tags_ex:
                    ret_slot.add(("call_param_ret", frozenset(c_p_tags_ex)))
                if any_tainted:
                    tainted_boxes.add(ret_bid)
                    ret_slot.add(("kw_taint", True))
                _apply_known_call_returns(
                    callable_slots,
                    ex_pos_rev,
                    ret_slot,
                    ret_bid,
                    is_tainted_call=bool(kw_tainted or pos_tainted),
                    allow_pos_arg_fwd=ex_allow_pos,
                    kw_arg_map=ex_kw_map,
                    unk_kw_slot=ex_unk_kw,
                    star_pos_slot=ex_star_pos,
                )
                stack.append(ret_slot)
                continue
            if op in ("CALL", "CALL_KW"):
                eff = _safe_stack_effect(ins, default=-(arg_int + 1))
                total_pop = max(0, 1 - eff)
                popped = [stack.pop() if stack else set() for _ in range(total_pop)]
                kw_names_cnt = 1 if op == "CALL_KW" else 0
                _flush_consts(*popped[kw_names_cnt:])
                arg_slots = popped[kw_names_cnt : kw_names_cnt + arg_int]
                callable_slots = popped[kw_names_cnt + arg_int :]
                pos_rev_slots: Sequence[set[Tuple[str, Any]]] = ()
                allow_pos_fwd = False
                kw_arg_map: Dict[str, set[Tuple[str, Any]]] = {}
                if op == "CALL":
                    if cur_kw_names is not None and len(cur_kw_names) <= len(arg_slots):
                        m_kw = len(cur_kw_names)
                        pos_rev_slots = arg_slots[m_kw:]
                        allow_pos_fwd = True
                        for kw_i, kw_name in enumerate(cur_kw_names):
                            kw_arg_map[str(kw_name)] = arg_slots[m_kw - 1 - kw_i]
                    else:
                        pos_rev_slots = arg_slots
                        allow_pos_fwd = True
                elif op == "CALL_KW" and popped:
                    kw_names_tup = next(
                        (v for k, v in popped[0] if k == "raw_key_tuple" and isinstance(v, tuple)),
                        None,
                    )
                    if kw_names_tup is not None and len(kw_names_tup) <= len(arg_slots):
                        m_kw = len(kw_names_tup)
                        pos_rev_slots = arg_slots[m_kw:]
                        allow_pos_fwd = True
                        for kw_i, kw_name in enumerate(kw_names_tup):
                            kw_arg_map[str(kw_name)] = arg_slots[m_kw - 1 - kw_i]
                args_tainted = arg_int > 0 and any(_slot_is_tainted(s) for s in arg_slots)
                nested_call_tainted = args_tainted or any(
                    ("closure_taint", True) in s or ("iter_taint", True) in s
                    for s in callable_slots
                )
                recv_tainted = any(_slot_is_tainted(s) for s in callable_slots)
                any_tainted = args_tainted or nested_call_tainted or recv_tainted
                call_method_names = {
                    v[1] if k == "attr" else v
                    for s in callable_slots
                    for k, v in s
                    if k in ("leaf_attr", "name") or (k == "attr" and isinstance(v, tuple))
                }
                unbound_methods = {
                    v[1] if k == "attr" else v
                    for s in callable_slots
                    for k, v in s
                    if (
                        k == "attr"
                        and v[0] not in ("self", "cls")
                        and v[1] in _mut_dict_methods
                    )
                    or (k == "name" and v in _mut_dict_methods)
                }
                if pos_rev_slots:
                    if call_method_names & {"append", "add"}:
                        el_app = pos_rev_slots[-2] if (unbound_methods and len(pos_rev_slots) >= 2) else pos_rev_slots[-1]
                        targets_app = [pos_rev_slots[-1]] if (unbound_methods and len(pos_rev_slots) >= 2) else callable_slots
                        is_seq_m = "append" in call_method_names
                        for tgt_s in targets_app:
                            _record_container_elem(tgt_s, el_app, seq_mut=is_seq_m)
                    if "insert" in call_method_names:
                        if unbound_methods and len(pos_rev_slots) >= 3:
                            _record_container_elem(pos_rev_slots[-1], pos_rev_slots[-3], seq_mut=True)
                        elif len(pos_rev_slots) >= 2:
                            for tgt_s in callable_slots:
                                _record_container_elem(tgt_s, pos_rev_slots[-2], seq_mut=True)
                    if call_method_names & {"extend", "extendleft"}:
                        src_ext = pos_rev_slots[-2] if (unbound_methods and len(pos_rev_slots) >= 2) else pos_rev_slots[-1]
                        targets_ext = [pos_rev_slots[-1]] if (unbound_methods and len(pos_rev_slots) >= 2) else callable_slots
                        rep_ext = _unpack_seq_with_boxes(src_ext)
                        ext_el_set = set(_collect_rep_elem_tags(rep_ext)) if rep_ext is not None else (
                            set(_lookup_container_elems(src_ext)) or _freeze_elem_slot(src_ext)
                        )
                        if ext_el_set:
                            for tgt_s in targets_ext:
                                _record_container_elem(tgt_s, set(ext_el_set), seq_mut=True)
                    if "setdefault" in call_method_names:
                        if unbound_methods and len(pos_rev_slots) >= 3:
                            targets_sd, key_sd, val_sd = [pos_rev_slots[-1]], pos_rev_slots[-2], pos_rev_slots[-3]
                        elif len(pos_rev_slots) >= 2:
                            targets_sd, key_sd, val_sd = callable_slots, pos_rev_slots[-1], pos_rev_slots[-2]
                        else:
                            targets_sd, key_sd, val_sd = [], None, None
                        for tgt_s in targets_sd:
                            if val_sd is not None and any(b in dict_boxes for b in _container_bids(tgt_s)):
                                _record_dict_entry(
                                    tgt_s,
                                    _extract_const_keys(key_sd) if key_sd is not None else None,
                                    val_sd,
                                )
                            elif val_sd is not None:
                                _record_container_elem(tgt_s, val_sd)
                if "update" in call_method_names:
                    is_unbound_upd = "update" in unbound_methods
                    targets_upd = [pos_rev_slots[-1]] if (is_unbound_upd and len(pos_rev_slots) >= 1) else callable_slots
                    srcs_upd = pos_rev_slots[:-1] if (is_unbound_upd and len(pos_rev_slots) >= 1) else pos_rev_slots
                    for tgt_u in targets_upd:
                        tgt_is_dict = any(tb in dict_boxes for tb in _container_bids(tgt_u))
                        for src_u in srcs_upd:
                            if any(sb in dict_boxes for sb in _container_bids(src_u)):
                                _merge_dict_into(tgt_u, src_u)
                            elif tgt_is_dict:
                                src_rep_u = _unpack_seq_with_boxes(src_u)
                                el_u = (
                                    _collect_rep_elem_tags(src_rep_u)
                                    if src_rep_u is not None
                                    else (_lookup_container_elems(src_u) or _freeze_elem_slot(src_u))
                                )
                                tmp_u = set(el_u)
                                if _slot_is_tainted(src_u):
                                    tmp_u.add(("kw_taint", True))
                                _record_dict_entry(tgt_u, None, tmp_u)
                            else:
                                rep_u = _unpack_seq_with_boxes(src_u)
                                ext_u = (
                                    set(_collect_rep_elem_tags(rep_u))
                                    if rep_u is not None
                                    else (set(_lookup_container_elems(src_u)) or _freeze_elem_slot(src_u))
                                )
                                if ext_u:
                                    _record_container_elem(tgt_u, set(ext_u))
                        if kw_arg_map:
                            for kw_k_u, kw_v_u in kw_arg_map.items():
                                _record_dict_entry(tgt_u, frozenset({kw_k_u}), kw_v_u)
                if call_method_names & {"pop", "remove", "clear", "reverse", "sort"}:
                    is_unbound_seq_mut = any(
                        k == "attr"
                        and isinstance(v, tuple)
                        and v[0] == "list"
                        and v[0] not in locals_env
                        and v[1] in ("pop", "remove", "clear", "reverse", "sort")
                        for s in callable_slots
                        for k, v in s
                    )
                    targets_sm = [pos_rev_slots[-1]] if (is_unbound_seq_mut and pos_rev_slots) else callable_slots
                    for tgt_m in targets_sm:
                        for bid_m in _container_bids(tgt_m):
                            if bid_m not in dict_boxes:
                                mutated_seq_boxes.add(bid_m)
                if args_tainted:
                    is_mut_dict_call = any(
                        (k == "leaf_attr" and v in _mut_dict_methods)
                        or (k == "name" and v in _mut_dict_methods)
                        for s in callable_slots
                        for k, v in s
                    )
                    if is_mut_dict_call:
                        for s in callable_slots:
                            _taint_receiver_slot(s)
                        if unbound_methods:
                            min_args = 3 if (unbound_methods & _two_arg_mut_methods) else 2
                            if len(arg_slots) >= min_args:
                                _taint_receiver_slot(arg_slots[-1])
                    if arg_slots and any(
                        (k == "name" and v == "partial") or (k == "leaf_attr" and v == "partial")
                        for s in callable_slots
                        for k, v in s
                    ):
                        _record(arg_slots[-1], unpacked=False)
                    for slot in callable_slots:
                        _record(slot, unpacked=False)
                if any(("closure_taint", True) in s for s in callable_slots):
                    for s in callable_slots:
                        if ("closure_taint", True) in s:
                            for k, v in s:
                                if k == "name":
                                    called_closure_names.add(v)
                if any_tainted:
                    _record_ret_invoked(callable_slots)
                if nested_call_tainted:
                    iter_t = any(("iter_taint", True) in s for s in callable_slots + arg_slots)
                    _activate_nested_slots(
                        callable_slots + arg_slots,
                        is_unpacked=False,
                        with_tainted_args=bool(args_tainted or iter_t),
                    )
                else:
                    iter_t = False
                _record_called_params_from_slots(
                    callable_slots,
                    is_unpacked=False,
                    taint_mode=(2 if any_tainted else (1 if _slots_have_param(arg_slots) else 0)),
                )
                ret_bid = ins_idx + 1
                ret_slot = {("box_id", ret_bid)}
                c_p_tags = {
                    (k_p, v_p)
                    for cs in callable_slots
                    for k_p, v_p in cs
                    if k_p in ("param_idx", "param_name", "varargs_from", "varkw_any")
                }
                if c_p_tags:
                    ret_slot.add(("call_param_ret", frozenset(c_p_tags)))
                if "copy" in call_method_names and not pos_rev_slots and not kw_arg_map:
                    for tgt_cp in callable_slots:
                        if any(b in dict_boxes for b in _container_bids(tgt_cp)):
                            _merge_dict_into(ret_slot, tgt_cp)
                handled_dict_ret = False
                if call_method_names & {"get", "pop", "setdefault"}:
                    is_unbound_dict_lk = any(
                        k == "attr"
                        and isinstance(v, tuple)
                        and v[0] == "dict"
                        and v[0] not in locals_env
                        and v[1] in ("get", "pop", "setdefault")
                        for s in callable_slots
                        for k, v in s
                    )
                    if is_unbound_dict_lk and len(pos_rev_slots) >= 2:
                        lk_targets: List[set[Tuple[str, Any]]] = [pos_rev_slots[-1]]
                        key_lk_slot: Optional[set[Tuple[str, Any]]] = pos_rev_slots[-2]
                        def_lk_slot: Optional[set[Tuple[str, Any]]] = (
                            pos_rev_slots[-3] if len(pos_rev_slots) >= 3 else None
                        )
                    else:
                        lk_targets = list(callable_slots)
                        key_lk_slot = pos_rev_slots[-1] if len(pos_rev_slots) >= 1 else None
                        def_lk_slot = (
                            pos_rev_slots[-2]
                            if len(pos_rev_slots) >= 2
                            else kw_arg_map.get("default")
                        )
                    all_lk_bids: set[int] = set()
                    for ts in lk_targets:
                        all_lk_bids.update(_container_bids(ts))
                    all_tgt_dict = bool(all_lk_bids) and all(b in dict_boxes for b in all_lk_bids)
                    lk_tainted_any = False
                    for ts in lk_targets:
                        lk_tags, lk_t = _lookup_keyed_elems(ts, key_lk_slot)
                        if lk_t:
                            lk_tainted_any = True
                        if lk_tags:
                            ret_slot.update(
                                kv
                                for kv in lk_tags
                                if kv[0] in _RET_COPY_KEYS
                                or kv[0] in ("box_id", "param_idx", "param_name", "varargs_from", "varkw_any")
                            )
                    if def_lk_slot is not None:
                        ret_slot.update(_freeze_elem_slot(def_lk_slot))
                        if _slot_is_tainted(def_lk_slot):
                            lk_tainted_any = True
                    if all_tgt_dict:
                        handled_dict_ret = True
                        if ret_bid in tainted_boxes or lk_tainted_any:
                            tainted_boxes.add(ret_bid)
                            ret_slot.add(("kw_taint", True))
                    elif lk_tainted_any:
                        tainted_boxes.add(ret_bid)
                        ret_slot.add(("kw_taint", True))
                if not handled_dict_ret and (any_tainted or ret_bid in tainted_boxes):
                    tainted_boxes.add(ret_bid)
                    ret_slot.add(("kw_taint", True))
                _apply_known_call_returns(
                    callable_slots,
                    pos_rev_slots,
                    ret_slot,
                    ret_bid,
                    is_tainted_call=bool(args_tainted or iter_t),
                    allow_pos_arg_fwd=allow_pos_fwd,
                    kw_arg_map=kw_arg_map,
                    attach_param_dep=not handled_dict_ret,
                )
                stack.append(ret_slot)
                continue
            if op == "RETURN_VALUE":
                ret_top = stack.pop() if stack else set()
                _flush_consts(ret_top)
                for rk, rv in ret_top:
                    if rk == "name" and rv in local_varnames_fset:
                        continue
                    if rk in ("seq_items", "seq_items_ex"):
                        returned_slots.update(_strip_local_names_fset(frozenset({(rk, rv)})))
                    elif rk in _RET_COPY_KEYS or rk in ("param_idx", "param_name", "varargs_from", "varkw_any"):
                        returned_slots.add((rk, rv))
                    elif rk == "box_id" and rv in tainted_boxes:
                        returned_slots.add(("kw_taint", True))
                ret_bids_rv = _container_bids(ret_top)
                has_dict_rv = False
                for r_bid in ret_bids_rv:
                    if r_bid in dict_boxes:
                        has_dict_rv = True
                        k_map_r = box_key_elems.get(r_bid, {})
                        t_keys_r = box_tainted_keys.get(r_bid, set())
                        rv_k_items: List[Tuple[Any, frozenset[Tuple[str, Any]]]] = []
                        for dk_r in set(k_map_r.keys()) | set(t_keys_r):
                            dv_clean = set(_strip_local_names_fset(frozenset(k_map_r.get(dk_r, ()))))
                            if dk_r in t_keys_r:
                                dv_clean.add(("kw_taint", True))
                            if dv_clean:
                                rv_k_items.append((dk_r, frozenset(dv_clean)))
                        rv_unk_set = set(_strip_local_names_fset(frozenset(box_unk_key_elems.get(r_bid, ()))))
                        if r_bid in box_unk_tainted:
                            rv_unk_set.add(("kw_taint", True))
                        returned_slots.add(("ret_dict", (frozenset(rv_k_items), frozenset(rv_unk_set))))
                if not has_dict_rv:
                    ret_el_fset = _strip_local_names_fset(_lookup_container_elems(ret_top))
                    if ret_el_fset:
                        returned_slots.add(("ret_elems", ret_el_fset))
                continue
            eff = _safe_stack_effect(ins, default=0)
            if op in (
                "POP_TOP",
                "END_FOR",
                "POP_ITER",
                "STORE_ATTR",
            ):
                for _ in range(max(0, -eff)):
                    if stack:
                        _flush_consts(stack.pop())
            elif eff < 0:
                popped_f = [stack.pop() if stack else set() for _ in range(max(1, 1 - eff))]
                _flush_consts(*popped_f)
                had_t_f = any(_slot_is_tainted(s) for s in popped_f)
                res_f: set[Tuple[str, Any]] = {("kw_taint", True)} if had_t_f else set()
                if (
                    op == "BINARY_OP"
                    and str(getattr(ins, "argrepr", "")) in ("+=", "|=")
                    and len(popped_f) >= 2
                ):
                    argrepr_bi = str(getattr(ins, "argrepr", ""))
                    if _slot_is_tainted(popped_f[0]):
                        _taint_receiver_slot(popped_f[-1])
                    if argrepr_bi == "+=":
                        it_rep_p = _unpack_seq_with_boxes(popped_f[0])
                        el_add = (
                            set(_collect_rep_elem_tags(it_rep_p))
                            if it_rep_p is not None
                            else (set(_lookup_container_elems(popped_f[0])) or _freeze_elem_slot(popped_f[0]))
                        )
                        _record_container_elem(popped_f[-1], set(el_add), seq_mut=True)
                    else:
                        if any(
                            b in dict_boxes
                            for b in (_container_bids(popped_f[-1]) | _container_bids(popped_f[0]))
                        ):
                            _merge_dict_into(popped_f[-1], popped_f[0])
                        else:
                            el_add_m = _lookup_container_elems(popped_f[0]) or _freeze_elem_slot(popped_f[0])
                            _record_container_elem(popped_f[-1], set(el_add_m))
                    for k_b, v_b in popped_f[-1]:
                        if k_b == "box_id":
                            res_f.add(("box_id", v_b))
                elif (
                    op == "BINARY_SUBSCR"
                    or (op == "BINARY_OP" and str(getattr(ins, "argrepr", "")) == "[]")
                ) and len(popped_f) >= 2:
                    key_sub_s = popped_f[0]
                    cont_sub_s = popped_f[1]
                    idx_val = next((v for k, v in key_sub_s if k == "raw_int"), None)
                    seq_rep_sub = _unpack_seq_with_boxes(cont_sub_s)
                    box_t_sub = any(k == "box_id" and v in tainted_boxes for k, v in cont_sub_s)
                    if ("is_slice", True) in key_sub_s:
                        if seq_rep_sub is not None:
                            sl_el2 = _collect_rep_elem_tags(seq_rep_sub)
                            res_f = set()
                            _set_seq_repr(res_f, (), True, sl_el2, ())
                            if box_t_sub or _slot_is_tainted(set(sl_el2)) or _slot_is_tainted(key_sub_s):
                                res_f.add(("kw_taint", True))
                        else:
                            lk_sl, lk_sl_t = _lookup_keyed_elems(cont_sub_s, None)
                            res_f = set(lk_sl)
                            if lk_sl_t or _slot_is_tainted(key_sub_s):
                                res_f.add(("kw_taint", True))
                            for k_c_sub, v_c_sub in cont_sub_s:
                                if k_c_sub == "call_origin":
                                    res_f.add((k_c_sub, v_c_sub))
                    elif seq_rep_sub is not None:
                        pref_sub, hs_sub, se_sub, suff_sub = seq_rep_sub
                        chosen_el: Optional[frozenset[Tuple[str, Any]]] = None
                        if isinstance(idx_val, int):
                            if not hs_sub:
                                eff_i = idx_val if idx_val >= 0 else len(pref_sub) + idx_val
                                if 0 <= eff_i < len(pref_sub):
                                    chosen_el = pref_sub[eff_i]
                            else:
                                if 0 <= idx_val < len(pref_sub):
                                    chosen_el = pref_sub[idx_val]
                                elif idx_val >= len(pref_sub):
                                    chosen_el = se_sub.union(*suff_sub) if suff_sub else se_sub
                                elif idx_val < 0 and (-idx_val) <= len(suff_sub):
                                    chosen_el = suff_sub[len(suff_sub) + idx_val]
                                else:
                                    chosen_el = se_sub.union(*pref_sub) if pref_sub else se_sub
                        else:
                            chosen_el = _collect_rep_elem_tags(seq_rep_sub)
                        if chosen_el is not None:
                            res_f = set(chosen_el)
                            if box_t_sub or _slot_is_tainted(res_f) or _slot_is_tainted(key_sub_s):
                                res_f.add(("kw_taint", True))
                    else:
                        lk_tags, lk_tainted = _lookup_keyed_elems(cont_sub_s, key_sub_s)
                        res_f = set(lk_tags)
                        if lk_tainted:
                            res_f.add(("kw_taint", True))
                        for k_c_sub, v_c_sub in cont_sub_s:
                            if k_c_sub == "call_origin":
                                res_f.add((k_c_sub, v_c_sub))
                elif op == "BINARY_SLICE" and len(popped_f) >= 3:
                    seq_rep_sl = _unpack_seq_with_boxes(popped_f[2])
                    if seq_rep_sl is not None:
                        sl_el = _collect_rep_elem_tags(seq_rep_sl)
                        _set_seq_repr(res_f, (), True, sl_el, ())
                    else:
                        for k_c_sl, v_c_sl in popped_f[2]:
                            if k_c_sl == "call_origin":
                                res_f.add((k_c_sl, v_c_sl))
                elif (
                    op == "BINARY_OP"
                    and str(getattr(ins, "argrepr", "")) == "+"
                    and len(popped_f) >= 2
                ):
                    lhs_rep = _unpack_seq_with_boxes(popped_f[1])
                    rhs_rep = _unpack_seq_with_boxes(popped_f[0])
                    if lhs_rep is not None and rhs_rep is not None:
                        c_pref, c_hs, c_se, c_suff = _concat_seq_reps(lhs_rep, rhs_rep)
                        _set_seq_repr(res_f, c_pref, c_hs, c_se, c_suff)
                stack.append(res_f)
            elif eff > 0:
                for _ in range(eff):
                    stack.append(set())
            elif op.startswith(("UNARY_", "GET_", "TO_")) and stack:
                _flush_consts(stack[-1])
                if _slot_is_tainted(stack[-1]):
                    stack[-1] = {("kw_taint", True)}
        _flush_consts(*stack)

    def _pass_sig() -> Tuple[Any, ...]:
        return (
            frozenset(tainted_boxes),
            frozenset(unpacked_boxes),
            frozenset(unpacked_names),
            frozenset(unpacked_owner_attrs),
            frozenset(unpacked_leaf_attrs),
            frozenset(arg_names),
            frozenset(arg_owner_attrs),
            frozenset(arg_leaf_attrs),
            frozenset(active_consts),
            frozenset(unpacked_codes.keys()),
            frozenset((k, v[1]) for k, v in tainted_codes.items()),
            frozenset((vn, frozenset(sl)) for vn, sl in locals_env.items()),
            frozenset(def_tainted_params.items()),
            frozenset(called_closure_names),
            frozenset(returned_slots),
            frozenset(invoked_local_codes.items()),
            frozenset(called_params),
            frozenset((bk, frozenset(bv)) for bk, bv in box_elem_tags.items()),
            frozenset(mutated_seq_boxes),
            frozenset(dict_boxes),
            frozenset(
                (bk, frozenset((dk, frozenset(dv)) for dk, dv in bmap.items()))
                for bk, bmap in box_key_elems.items()
            ),
            frozenset((bk, frozenset(bv)) for bk, bv in box_unk_key_elems.items()),
            frozenset((bk, frozenset(bt)) for bk, bt in box_tainted_keys.items()),
            frozenset(box_unk_tainted),
            frozenset(ret_invoked_names),
            frozenset(ret_invoked_owner_attrs),
            frozenset(ret_invoked_leaf_attrs),
        )

    prev_sig: Optional[Tuple[Any, ...]] = None
    max_passes = max(16, len(ins_list) + 4)
    for pass_i in range(max_passes):
        _run_pass()
        cur_sig = _pass_sig()
        if pass_i >= 1 and cur_sig == prev_sig:
            break
        prev_sig = cur_sig

    tainted_local_names = frozenset(vn for vn, sl in locals_env.items() if _slot_is_tainted(sl))
    closure_local_names = frozenset(vn for vn, sl in locals_env.items() if ("closure_taint", True) in sl)
    local_code_defs: Dict[str, types.CodeType] = {}
    for vn, sl in locals_env.items():
        loc_code_v: Optional[types.CodeType] = None
        loc_tp: set[str] = set()
        for kind, val in sl:
            if kind == "code" and isinstance(val, types.CodeType):
                loc_code_v = val
                local_code_defs[vn] = val
            elif kind == "fn_tainted_params" and val:
                loc_tp.update(val)
        if loc_code_v is not None and loc_tp:
            def_tainted_params[id(loc_code_v)] = (
                def_tainted_params.get(id(loc_code_v), frozenset()) | frozenset(loc_tp)
            )
    return (
        unpacked_names,
        unpacked_owner_attrs,
        unpacked_leaf_attrs,
        arg_names,
        arg_owner_attrs,
        arg_leaf_attrs,
        active_consts,
        unpacked_codes,
        tainted_codes,
        tainted_local_names,
        local_code_defs,
        def_tainted_params,
        called_closure_names,
        closure_local_names,
        frozenset(returned_slots),
        invoked_local_codes,
        (ret_invoked_names, ret_invoked_owner_attrs, ret_invoked_leaf_attrs),
        frozenset(called_params),
    )


def _extract_kw_unpack_targets(
    code: types.CodeType,
    taint_vars: Optional[frozenset[str]] = None,
) -> Tuple[set[str], set[Tuple[str, str]], set[str], set[str], set[Tuple[str, str]], set[str]]:
    """Return bytecode callable targets for CALL_FUNCTION_EX (**kwargs) and non-zero-arg CALLs."""
    return _analyze_code_kw_flow(code, taint_vars)[:6]


def _build_cell_owner_map(
    root_code: types.CodeType,
    cap: int = 256,
) -> Dict[int, Dict[str, int]]:
    """Map each nested code id to enclosing declaring code ids for its cell/free vars."""
    cell_owners: Dict[int, Dict[str, int]] = {}
    work: deque[Tuple[types.CodeType, Dict[str, int]]] = deque(
        [(root_code, {str(fv): id(root_code) for fv in root_code.co_freevars})]
    )
    while work and len(cell_owners) < cap:
        c_node, env_map = work.popleft()
        cid_node = id(c_node)
        if cid_node in cell_owners:
            continue
        node_cells: Dict[str, int] = {}
        for fv in c_node.co_freevars:
            fv_s = str(fv)
            if fv_s in env_map:
                node_cells[fv_s] = env_map[fv_s]
        for cv in c_node.co_cellvars:
            node_cells[str(cv)] = cid_node
        cell_owners[cid_node] = node_cells
        child_env = dict(env_map)
        for cv in c_node.co_cellvars:
            child_env[str(cv)] = cid_node
        const_q: List[Any] = list(c_node.co_consts)
        while const_q:
            const_item = const_q.pop()
            if isinstance(const_item, types.CodeType):
                if id(const_item) not in cell_owners:
                    work.append((const_item, child_env))
            elif isinstance(const_item, tuple):
                const_q.extend(const_item)
    return cell_owners


def _inspect_varkw_wrapper(
    fn_obj: Any,
    candidate_keys: frozenset[str],
    owner_self: Any = None,
) -> Tuple[set[str], bool]:
    """Return (referenced_candidate_keys, calls_super) for a **kwargs callable."""
    referenced: set[str] = set()

    def _collect_const_strings(val: Any, dst: set[str], depth: int = 0) -> None:
        if isinstance(val, str):
            if val in candidate_keys:
                dst.add(val)
            return
        if depth >= 3:
            return
        if isinstance(val, (tuple, list, set, frozenset)):
            for item in val:
                _collect_const_strings(item, dst, depth + 1)
        elif isinstance(val, Mapping):
            for k_item, v_item in val.items():
                _collect_const_strings(k_item, dst, depth + 1)
                _collect_const_strings(v_item, dst, depth + 1)

    def _unwrap_callable_and_owner(obj: Any, default_owner: Any) -> Tuple[Optional[Any], Any, set[str]]:
        cur = obj
        cur_owner = default_owner
        partial_kw: set[str] = set()
        while isinstance(cur, functools.partial):
            if cur.keywords:
                for pk in cur.keywords:
                    if pk in candidate_keys:
                        partial_kw.add(pk)
            cur = cur.func
        if isinstance(cur, (staticmethod, classmethod)):
            cur = cur.__func__
        elif hasattr(cur, "__func__") and hasattr(cur, "__self__"):
            cur_owner = cur.__self__
            cur = cur.__func__
        try:
            cur = inspect.unwrap(cur)
        except Exception:
            pass
        if not isinstance(getattr(cur, "__code__", None), types.CodeType) and callable(cur) and not isinstance(cur, type):
            call_attr = inspect.getattr_static(type(cur), "__call__", None)
            if isinstance(call_attr, (staticmethod, classmethod)):
                call_attr = call_attr.__func__
            if callable(call_attr):
                cur_owner = cur
                cur = call_attr
        return cur, cur_owner, partial_kw

    def _extract_taint_var_names(
        fn_target: Any,
        fn_owner: Any,
        depth: int,
    ) -> Optional[frozenset[str]]:
        code_t = getattr(fn_target, "__code__", None)
        if not isinstance(code_t, types.CodeType):
            return None
        if depth == 0:
            try:
                sig_t = inspect.signature(fn_target)
                vk = frozenset(
                    p.name for p in sig_t.parameters.values() if p.kind == inspect.Parameter.VAR_KEYWORD
                )
                if vk:
                    return vk
            except (TypeError, ValueError):
                pass
            n_before = (
                code_t.co_argcount
                + code_t.co_kwonlyargcount
                + (1 if (code_t.co_flags & inspect.CO_VARARGS) else 0)
            )
            if (code_t.co_flags & inspect.CO_VARKEYWORDS) and n_before < len(code_t.co_varnames):
                return frozenset((str(code_t.co_varnames[n_before]),))
            return frozenset()
        n_params = (
            code_t.co_argcount
            + code_t.co_kwonlyargcount
            + (1 if (code_t.co_flags & inspect.CO_VARARGS) else 0)
            + (1 if (code_t.co_flags & inspect.CO_VARKEYWORDS) else 0)
        )
        start_p = 1 if (fn_owner is not None and code_t.co_argcount >= 1) else 0
        return frozenset(str(v) for v in code_t.co_varnames[start_p:n_params])

    def _collect_explicit_params(fn_target: Any) -> None:
        try:
            sub_sig = inspect.signature(fn_target)
            for p_name, p_obj in sub_sig.parameters.items():
                if (
                    p_obj.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                    and p_name in candidate_keys
                ):
                    referenced.add(p_name)
        except (TypeError, ValueError):
            pass

    root_owner = getattr(fn_obj, "__self__", None) if owner_self is None else owner_self
    root_method_name = getattr(fn_obj, "__name__", None)
    root_mro = set(type(root_owner).__mro__) if root_owner is not None else set()
    root_calls_super = False
    max_depth = 6
    max_callables = 64
    queue: deque[Tuple[Any, Any, int]] = deque([(fn_obj, root_owner, 0)])
    visited_call_keys: set[Tuple[int, int, bool]] = set()
    visited_depths: Dict[int, int] = {}
    sig_param_fn_ids: set[int] = set()
    invoked_ret_fn_ids: set[int] = set()
    code_info_cache: Dict[
        Tuple[int, int, Optional[frozenset[str]], bool],
        Tuple[
            set[str],
            set[str],
            set[Tuple[str, str]],
            set[str],
            set[str],
            set[Tuple[str, str]],
            set[str],
            set[str],
            set[str],
            set[Tuple[str, str]],
            set[str],
        ],
    ] = {}

    def _mark_unpacked_edge(
        cand: Any,
        sub_owner: Any,
        *,
        unpacked: bool,
        ret_invoked: bool = False,
    ) -> None:
        if cand is None or isinstance(cand, type) or not callable(cand):
            return
        u_cand, _, partial_kw = _unwrap_callable_and_owner(cand, sub_owner)
        if u_cand is None:
            return
        referenced.update(partial_kw)
        u_id = id(u_cand)
        if ret_invoked:
            invoked_ret_fn_ids.add(u_id)
        if unpacked:
            sig_param_fn_ids.add(u_id)
            if visited_depths.get(u_id, 0) > 0:
                _collect_explicit_params(u_cand)

    def _enqueue(cand: Any, sub_owner: Any, nxt_depth: int) -> None:
        if cand is None or nxt_depth > max_depth or len(visited_call_keys) >= max_callables:
            return
        if isinstance(cand, type) or not callable(cand):
            return
        queue.append((cand, getattr(cand, "__self__", sub_owner), nxt_depth))

    def _consider_val(
        val: Any,
        sub_owner: Any,
        local_names: set[str],
        nxt_depth: int,
        *,
        edge_unpacked: bool = False,
        edge_can_fwd: bool = False,
        ret_invoked: bool = False,
        via_name: Optional[str] = None,
        unpacked_owner_attrs: Optional[set[Tuple[str, str]]] = None,
        unpacked_leaf_attrs: Optional[set[str]] = None,
        arg_owner_attrs: Optional[set[Tuple[str, str]]] = None,
        arg_leaf_attrs: Optional[set[str]] = None,
        ret_invoked_names: Optional[set[str]] = None,
        ret_invoked_owner_attrs: Optional[set[Tuple[str, str]]] = None,
        ret_invoked_leaf_attrs: Optional[set[str]] = None,
        is_root_code: bool = False,
        visited_types: Optional[set[int]] = None,
    ) -> None:
        nonlocal root_calls_super
        if not isinstance(val, (type, types.ModuleType)) and not callable(val):
            _collect_const_strings(val, referenced)
        if nxt_depth > max_depth or len(visited_call_keys) >= max_callables or val is None:
            return
        if isinstance(val, type):
            if visited_types is None:
                visited_types = set()
            tid = id(val)
            if tid in visited_types:
                return
            visited_types.add(tid)
            uo_attrs = unpacked_owner_attrs or set()
            ul_attrs = unpacked_leaf_attrs or set()
            ao_attrs = arg_owner_attrs or set()
            al_attrs = arg_leaf_attrs or set()
            ri_oattrs = ret_invoked_owner_attrs or set()
            ri_lattrs = ret_invoked_leaf_attrs or set()
            # Inspect custom __init__ if defined directly on val.
            raw_init = val.__dict__.get("__init__")
            if raw_init is not None and (edge_unpacked or edge_can_fwd or ret_invoked):
                _mark_unpacked_edge(raw_init, None, unpacked=edge_unpacked, ret_invoked=ret_invoked)
                _enqueue(raw_init, None, nxt_depth)
            for sub_n in local_names:
                if val in root_mro and sub_n == root_method_name:
                    if is_root_code and (
                        ((via_name, sub_n) in uo_attrs) if via_name else (sub_n in ul_attrs)
                    ):
                        root_calls_super = True
                    continue
                raw_sub = inspect.getattr_static(val, sub_n, _UNSET_SLOT)
                if raw_sub is _UNSET_SLOT or isinstance(raw_sub, property):
                    continue
                is_sub_unp = ((via_name, sub_n) in uo_attrs) if via_name else (sub_n in ul_attrs)
                is_sub_ri = ((via_name, sub_n) in ri_oattrs) if via_name else (sub_n in ri_lattrs)
                is_sub_fwd = (
                    is_sub_unp
                    or is_sub_ri
                    or (((via_name, sub_n) in ao_attrs) if via_name else (sub_n in al_attrs))
                )
                if isinstance(raw_sub, type):
                    _consider_val(
                        raw_sub,
                        None,
                        local_names,
                        nxt_depth,
                        edge_unpacked=is_sub_unp,
                        edge_can_fwd=is_sub_fwd,
                        ret_invoked=is_sub_ri,
                        via_name=sub_n,
                        unpacked_owner_attrs=unpacked_owner_attrs,
                        unpacked_leaf_attrs=unpacked_leaf_attrs,
                        arg_owner_attrs=arg_owner_attrs,
                        arg_leaf_attrs=arg_leaf_attrs,
                        ret_invoked_names=ret_invoked_names,
                        ret_invoked_owner_attrs=ret_invoked_owner_attrs,
                        ret_invoked_leaf_attrs=ret_invoked_leaf_attrs,
                        is_root_code=is_root_code,
                        visited_types=visited_types,
                    )
                    continue
                if isinstance(raw_sub, staticmethod):
                    sub_fn, sub_own = raw_sub.__func__, None
                elif isinstance(raw_sub, classmethod):
                    sub_fn, sub_own = raw_sub.__func__, val
                else:
                    sub_fn, sub_own = raw_sub, None
                if not isinstance(sub_fn, type) and callable(sub_fn):
                    if is_sub_unp or is_sub_fwd:
                        _mark_unpacked_edge(sub_fn, sub_own, unpacked=is_sub_unp, ret_invoked=is_sub_ri)
                        _enqueue(sub_fn, sub_own, nxt_depth)
            return
        if callable(val):
            if edge_unpacked or edge_can_fwd or ret_invoked:
                _mark_unpacked_edge(val, sub_owner, unpacked=edge_unpacked, ret_invoked=ret_invoked)
                _enqueue(val, sub_owner, nxt_depth)
            if isinstance(
                val,
                (
                    types.FunctionType,
                    types.MethodType,
                    types.BuiltinFunctionType,
                    types.BuiltinMethodType,
                    functools.partial,
                ),
            ):
                return
        if isinstance(val, types.ModuleType):
            mod_d = val.__dict__
            uo_attrs = unpacked_owner_attrs or set()
            ul_attrs = unpacked_leaf_attrs or set()
            ao_attrs = arg_owner_attrs or set()
            al_attrs = arg_leaf_attrs or set()
            ri_oattrs = ret_invoked_owner_attrs or set()
            ri_lattrs = ret_invoked_leaf_attrs or set()
            for sub_n in local_names:
                sub_v = mod_d.get(sub_n)
                if sub_v is None:
                    continue
                is_sub_unp = ((via_name, sub_n) in uo_attrs) if via_name else (sub_n in ul_attrs)
                is_sub_ri = ((via_name, sub_n) in ri_oattrs) if via_name else (sub_n in ri_lattrs)
                is_sub_fwd = (
                    is_sub_unp
                    or is_sub_ri
                    or (((via_name, sub_n) in ao_attrs) if via_name else (sub_n in al_attrs))
                )
                if isinstance(sub_v, type):
                    _consider_val(
                        sub_v,
                        None,
                        local_names,
                        nxt_depth,
                        edge_unpacked=is_sub_unp,
                        edge_can_fwd=is_sub_fwd,
                        ret_invoked=is_sub_ri,
                        via_name=sub_n,
                        unpacked_owner_attrs=unpacked_owner_attrs,
                        unpacked_leaf_attrs=unpacked_leaf_attrs,
                        arg_owner_attrs=arg_owner_attrs,
                        arg_leaf_attrs=arg_leaf_attrs,
                        ret_invoked_names=ret_invoked_names,
                        ret_invoked_owner_attrs=ret_invoked_owner_attrs,
                        ret_invoked_leaf_attrs=ret_invoked_leaf_attrs,
                        is_root_code=is_root_code,
                    )
                elif callable(sub_v):
                    if is_sub_unp or is_sub_fwd:
                        _mark_unpacked_edge(sub_v, None, unpacked=is_sub_unp, ret_invoked=is_sub_ri)
                        _enqueue(sub_v, None, nxt_depth)
        elif not isinstance(val, (str, bytes, bytearray, int, float, bool, np.ndarray, Mapping, list, tuple, set, frozenset)):
            uo_attrs = unpacked_owner_attrs or set()
            ul_attrs = unpacked_leaf_attrs or set()
            ao_attrs = arg_owner_attrs or set()
            al_attrs = arg_leaf_attrs or set()
            ri_oattrs = ret_invoked_owner_attrs or set()
            ri_lattrs = ret_invoked_leaf_attrs or set()
            for sub_n in local_names:
                raw_sub = inspect.getattr_static(val, sub_n, _UNSET_SLOT)
                if raw_sub is _UNSET_SLOT or isinstance(raw_sub, property):
                    continue
                if isinstance(raw_sub, types.MemberDescriptorType):
                    try:
                        raw_sub = raw_sub.__get__(val, type(val))
                    except Exception:
                        continue
                is_sub_unp = ((via_name, sub_n) in uo_attrs) if via_name else (sub_n in ul_attrs)
                is_sub_ri = ((via_name, sub_n) in ri_oattrs) if via_name else (sub_n in ri_lattrs)
                is_sub_fwd = (
                    is_sub_unp
                    or is_sub_ri
                    or (((via_name, sub_n) in ao_attrs) if via_name else (sub_n in al_attrs))
                )
                if isinstance(raw_sub, (staticmethod, classmethod)):
                    raw_sub = raw_sub.__func__
                if isinstance(raw_sub, type):
                    _consider_val(
                        raw_sub,
                        None,
                        local_names,
                        nxt_depth,
                        edge_unpacked=is_sub_unp,
                        edge_can_fwd=is_sub_fwd,
                        ret_invoked=is_sub_ri,
                        via_name=sub_n,
                        unpacked_owner_attrs=unpacked_owner_attrs,
                        unpacked_leaf_attrs=unpacked_leaf_attrs,
                        arg_owner_attrs=arg_owner_attrs,
                        arg_leaf_attrs=arg_leaf_attrs,
                        ret_invoked_names=ret_invoked_names,
                        ret_invoked_owner_attrs=ret_invoked_owner_attrs,
                        ret_invoked_leaf_attrs=ret_invoked_leaf_attrs,
                        is_root_code=is_root_code,
                    )
                elif callable(raw_sub):
                    if is_sub_unp or is_sub_fwd:
                        _mark_unpacked_edge(raw_sub, val, unpacked=is_sub_unp, ret_invoked=is_sub_ri)
                        _enqueue(raw_sub, val, nxt_depth)

    while queue and len(visited_call_keys) < max_callables:
        cand_fn, cur_owner, depth = queue.popleft()
        unwrapped, cur_owner, partial_kw = _unwrap_callable_and_owner(cand_fn, cur_owner)
        if unwrapped is None:
            continue
        u_id = id(unwrapped)
        is_ret_invoked_ctx = depth >= 1 and u_id in invoked_ret_fn_ids
        visit_key = (u_id, id(cur_owner) if cur_owner is not None else 0, is_ret_invoked_ctx)
        if visit_key in visited_call_keys:
            continue
        visited_call_keys.add(visit_key)
        visited_depths[u_id] = max(visited_depths.get(u_id, depth), depth)
        referenced.update(partial_kw)

        if depth > 0 and u_id in sig_param_fn_ids:
            _collect_explicit_params(unwrapped)

        code = getattr(unwrapped, "__code__", None)
        if not isinstance(code, types.CodeType):
            continue

        taint_here = _extract_taint_var_names(unwrapped, cur_owner, depth)
        cache_key_ci = (id(code), id(cur_owner) if cur_owner is not None else 0, taint_here, is_ret_invoked_ctx)
        cached_ci = code_info_cache.get(cache_key_ci)
        if cached_ci is None:
            cell_owner_by_code = _build_cell_owner_map(code)
            max_ret_rounds = max(16, min(256, len(cell_owner_by_code) + 8))
            ref_names_ext: set[str] = set()
            rn_seen: set[int] = set()
            rn_q: deque[types.CodeType] = deque([code])
            while rn_q and len(rn_seen) < 256:
                c_rn = rn_q.popleft()
                if id(c_rn) in rn_seen:
                    continue
                rn_seen.add(id(c_rn))
                ref_names_ext.update(str(n) for n in c_rn.co_names)
                ref_names_ext.update(str(fv) for fv in c_rn.co_freevars)
                for const_rn in c_rn.co_consts:
                    if isinstance(const_rn, types.CodeType) and id(const_rn) not in rn_seen:
                        rn_q.append(const_rn)

            ext_target_codes: Dict[Any, Tuple[types.CodeType, int]] = {}

            def _reg_ext(tag_key: Any, raw_obj: Any, def_owner: Any) -> None:
                if len(ext_target_codes) >= 256 or raw_obj is None:
                    return
                cur_p = raw_obj
                partial_pos_cnt = 0
                while isinstance(cur_p, functools.partial):
                    partial_pos_cnt += len(cur_p.args)
                    cur_p = cur_p.func
                u_fn, u_own, _ = _unwrap_callable_and_owner(cur_p, def_owner)
                if u_fn is None:
                    return
                u_code = getattr(u_fn, "__code__", None)
                if not isinstance(u_code, types.CodeType):
                    return
                ext_target_codes[tag_key] = (
                    u_code,
                    (1 if u_own is not None else 0) + partial_pos_cnt,
                )

            def _reg_ext_attrs(prefix_name: str, holder: Any) -> None:
                if (
                    holder is None
                    or len(ext_target_codes) >= 256
                    or isinstance(
                        holder,
                        (
                            types.FunctionType,
                            types.MethodType,
                            types.BuiltinFunctionType,
                            types.BuiltinMethodType,
                            functools.partial,
                            int,
                            float,
                            str,
                            bytes,
                            bool,
                            tuple,
                            list,
                            dict,
                            set,
                            frozenset,
                        ),
                    )
                ):
                    return
                holder_is_cls_mod = isinstance(holder, (type, types.ModuleType))
                holder_dict = getattr(holder, "__dict__", None)
                for sub_n in ref_names_ext:
                    if len(ext_target_codes) >= 256:
                        break
                    raw_sub = inspect.getattr_static(holder, sub_n, _UNSET_SLOT)
                    if raw_sub is _UNSET_SLOT or isinstance(raw_sub, property):
                        continue
                    was_mem_sub = isinstance(raw_sub, types.MemberDescriptorType)
                    if was_mem_sub and not isinstance(holder, type):
                        try:
                            raw_sub = raw_sub.__get__(holder, type(holder))
                        except Exception:
                            continue
                    if isinstance(raw_sub, staticmethod):
                        sub_own = None
                    elif isinstance(raw_sub, classmethod):
                        sub_own = holder if isinstance(holder, type) else type(holder)
                    elif holder_is_cls_mod or was_mem_sub:
                        sub_own = None
                    elif isinstance(holder_dict, Mapping) and sub_n in holder_dict:
                        sub_own = None
                    elif hasattr(type(raw_sub), "__get__"):
                        sub_own = holder
                    else:
                        sub_own = None
                    _reg_ext(("attr", (prefix_name, sub_n)), raw_sub, sub_own)

            fn_globals_ext = getattr(unwrapped, "__globals__", None)
            if isinstance(fn_globals_ext, Mapping):
                for n_g in ref_names_ext:
                    if n_g in fn_globals_ext:
                        val_g = fn_globals_ext[n_g]
                        _reg_ext(("name", n_g), val_g, None)
                        _reg_ext_attrs(n_g, val_g)

            fn_closure_ext = getattr(unwrapped, "__closure__", None)
            if fn_closure_ext:
                free_ext_names = code.co_freevars if isinstance(getattr(code, "co_freevars", None), tuple) else ()
                for c_idx_e, cell_e in enumerate(fn_closure_ext):
                    c_name_e = free_ext_names[c_idx_e] if c_idx_e < len(free_ext_names) else None
                    if not c_name_e or c_name_e not in ref_names_ext:
                        continue
                    try:
                        val_ce = cell_e.cell_contents
                    except ValueError:
                        continue
                    _reg_ext(("name", c_name_e), val_ce, None)
                    _reg_ext_attrs(c_name_e, val_ce)

            if cur_owner is not None:
                owner_is_type_ext = isinstance(cur_owner, type)
                owner_dict_ext = getattr(cur_owner, "__dict__", None)
                for n_o in ref_names_ext:
                    if len(ext_target_codes) >= 256:
                        break
                    raw_attr_e = inspect.getattr_static(cur_owner, n_o, _UNSET_SLOT)
                    if raw_attr_e is _UNSET_SLOT or isinstance(raw_attr_e, property):
                        continue
                    was_mem_o = isinstance(raw_attr_e, types.MemberDescriptorType)
                    if was_mem_o and not owner_is_type_ext:
                        try:
                            raw_attr_e = raw_attr_e.__get__(cur_owner, type(cur_owner))
                        except Exception:
                            continue
                    if isinstance(raw_attr_e, staticmethod):
                        attr_own_e = None
                    elif isinstance(raw_attr_e, classmethod):
                        attr_own_e = cur_owner if owner_is_type_ext else type(cur_owner)
                    elif owner_is_type_ext or was_mem_o:
                        attr_own_e = None
                    elif isinstance(owner_dict_ext, Mapping) and n_o in owner_dict_ext:
                        attr_own_e = None
                    elif hasattr(type(raw_attr_e), "__get__"):
                        attr_own_e = cur_owner
                    else:
                        attr_own_e = None
                    _reg_ext(("attr", ("self", n_o)), raw_attr_e, attr_own_e)
                    _reg_ext(("attr", ("cls", n_o)), raw_attr_e, attr_own_e)
                    if ("leaf_attr", n_o) not in ext_target_codes:
                        _reg_ext(("leaf_attr", n_o), raw_attr_e, attr_own_e)
                    _reg_ext_attrs(n_o, raw_attr_e)

            known_code_returns: Dict[int, frozenset[Tuple[str, Any]]] = {}
            known_code_called_params: Dict[int, frozenset[Tuple[str, Any, bool, int]]] = {}
            cross_tainted_cells: set[Tuple[int, str]] = set()
            for _round_ret in range(max_ret_rounds):
                prev_kcr = dict(known_code_returns)
                prev_kccp = dict(known_code_called_params)
                prev_cells = set(cross_tainted_cells)
                c_local_names: set[str] = set()
                c_u_names: set[str] = set()
                c_u_oattrs: set[Tuple[str, str]] = set()
                c_u_lattrs: set[str] = set()
                c_a_names: set[str] = set()
                c_a_oattrs: set[Tuple[str, str]] = set()
                c_a_lattrs: set[str] = set()
                c_ri_names: set[str] = set()
                c_ri_oattrs: set[Tuple[str, str]] = set()
                c_ri_lattrs: set[str] = set()
                c_consts: set[str] = set()
                seen_sub_codes: set[Tuple[int, Optional[frozenset[str]], frozenset[str]]] = set()
                local_code_defs: Dict[str, types.CodeType] = {}
                local_code_def_taints: Dict[int, frozenset[str]] = {}
                round_tainted_cells: set[Tuple[int, str]] = set()
                round_closure_cells: set[Tuple[int, str]] = set()
                visited_taint_cids: set[int] = set()
                peek_candidates: Dict[int, types.CodeType] = {
                    id(ec): ec for ec, _ in ext_target_codes.values()
                }
                own_root_cells = frozenset(
                    str(cv) for cv in code.co_cellvars if (id(code), str(cv)) in cross_tainted_cells
                )
                root_taint = (taint_here | own_root_cells) if taint_here is not None else None
                code_stack: List[Tuple[types.CodeType, Optional[frozenset[str]], frozenset[str]]] = [
                    (code, root_taint, frozenset())
                ]
                while code_stack:
                    c, c_taint, c_closure = code_stack.pop()
                    sub_key = (id(c), c_taint, c_closure)
                    if sub_key in seen_sub_codes:
                        continue
                    seen_sub_codes.add(sub_key)
                    visited_taint_cids.add(id(c))
                    c_local_names.update(c.co_names)
                    c_local_names.update(c.co_freevars)
                    (
                        un_c,
                        uo_c,
                        ul_c,
                        an_c,
                        ao_c,
                        al_c,
                        act_consts_c,
                        unp_codes_c,
                        taint_codes_c,
                        taint_locals_c,
                        loc_codes_c,
                        def_tp_c,
                        called_clos_c,
                        clos_locals_c,
                        ret_slot_c,
                        invoked_c,
                        (ri_n_c, ri_oa_c, ri_la_c),
                        called_params_c,
                    ) = _analyze_code_kw_flow(
                        c,
                        c_taint,
                        candidate_keys,
                        c_closure,
                        known_code_returns=known_code_returns,
                        outer_code_defs=local_code_defs,
                        known_code_called_params=known_code_called_params,
                        ext_target_codes=ext_target_codes,
                    )
                    if ret_slot_c:
                        known_code_returns[id(c)] = known_code_returns.get(id(c), frozenset()) | ret_slot_c
                    if called_params_c:
                        known_code_called_params[id(c)] = (
                            known_code_called_params.get(id(c), frozenset()) | called_params_c
                        )
                    sub_map = cell_owner_by_code.get(id(c), {})
                    for tv in taint_locals_c:
                        owner_cid = sub_map.get(tv)
                        if owner_cid is not None:
                            cell_key = (owner_cid, tv)
                            round_tainted_cells.add(cell_key)
                            if tv in c.co_freevars:
                                cross_tainted_cells.add(cell_key)
                    for cv_n in clos_locals_c:
                        owner_cid = sub_map.get(cv_n)
                        if owner_cid is not None:
                            round_closure_cells.add((owner_cid, cv_n))
                    for cid_inv, (sub_inv_c, _) in invoked_c.items():
                        peek_candidates[cid_inv] = sub_inv_c
                    c_u_names.update(un_c)
                    c_u_oattrs.update(uo_c)
                    c_u_lattrs.update(ul_c)
                    c_a_names.update(an_c)
                    c_a_oattrs.update(ao_c)
                    c_a_lattrs.update(al_c)
                    c_ri_names.update(ri_n_c)
                    c_ri_oattrs.update(ri_oa_c)
                    c_ri_lattrs.update(ri_la_c)
                    c_consts.update(act_consts_c)
                    local_code_defs.update(loc_codes_c)
                    for cid_dt, tp_dt in def_tp_c.items():
                        local_code_def_taints[cid_dt] = (
                            local_code_def_taints.get(cid_dt, frozenset()) | tp_dt
                        )
                    all_unp_codes: Dict[int, types.CodeType] = dict(unp_codes_c)
                    all_taint_codes: Dict[int, Tuple[types.CodeType, bool]] = dict(taint_codes_c)
                    if depth >= 1 and c is code and u_id in invoked_ret_fn_ids and ret_slot_c:
                        ret_scan_slots: List[frozenset[Tuple[str, Any]]] = [ret_slot_c]
                        for rk_s, rv_s in ret_slot_c:
                            if rk_s == "seq_items" and isinstance(rv_s, tuple):
                                ret_scan_slots.extend(rv_s)
                            elif rk_s == "seq_items_ex" and isinstance(rv_s, tuple) and len(rv_s) == 4:
                                ret_scan_slots.extend(rv_s[0])
                                if rv_s[2]:
                                    ret_scan_slots.append(rv_s[2])
                                ret_scan_slots.extend(rv_s[3])
                            elif rk_s == "ret_elems" and isinstance(rv_s, frozenset):
                                ret_scan_slots.append(rv_s)
                            elif rk_s == "ret_dict" and isinstance(rv_s, tuple) and len(rv_s) == 2:
                                for _, dv_s in rv_s[0]:
                                    ret_scan_slots.append(dv_s)
                                if rv_s[1]:
                                    ret_scan_slots.append(rv_s[1])
                        for sl_ret in ret_scan_slots:
                            for rk, rv in sl_ret:
                                if rk == "code" and isinstance(rv, types.CodeType):
                                    all_taint_codes[id(rv)] = (rv, True)
                                    if u_id in sig_param_fn_ids:
                                        all_unp_codes[id(rv)] = rv
                                elif rk == "call_origin" and isinstance(rv, frozenset):
                                    for ok, ov in rv:
                                        if ok == "name":
                                            c_ri_names.add(ov)
                                            (c_u_names if u_id in sig_param_fn_ids else c_a_names).add(ov)
                                        elif ok == "attr":
                                            c_ri_oattrs.add(ov)
                                            (c_u_oattrs if u_id in sig_param_fn_ids else c_a_oattrs).add(ov)
                                        elif ok == "leaf_attr":
                                            c_ri_lattrs.add(ov)
                                            (c_u_lattrs if u_id in sig_param_fn_ids else c_a_lattrs).add(ov)
                                elif u_id in sig_param_fn_ids:
                                    if rk == "name":
                                        c_u_names.add(rv)
                                    elif rk == "attr":
                                        c_u_oattrs.add(rv)
                                    elif rk == "leaf_attr":
                                        c_u_lattrs.add(rv)
                                else:
                                    if rk == "name":
                                        c_a_names.add(rv)
                                    elif rk == "attr":
                                        c_a_oattrs.add(rv)
                                    elif rk == "leaf_attr":
                                        c_a_lattrs.add(rv)
                    for fn_var in c_u_names:
                        loc_c = local_code_defs.get(fn_var)
                        if loc_c is not None:
                            all_unp_codes[id(loc_c)] = loc_c
                            all_taint_codes[id(loc_c)] = (loc_c, True)
                    for fn_var in c_a_names:
                        loc_c = local_code_defs.get(fn_var)
                        if loc_c is not None:
                            all_taint_codes[id(loc_c)] = (loc_c, True)
                    for fn_var in called_clos_c:
                        loc_c = local_code_defs.get(fn_var)
                        if loc_c is not None and id(loc_c) not in all_taint_codes:
                            all_taint_codes[id(loc_c)] = (loc_c, False)
                    for unp_c in all_unp_codes.values():
                        pos_only_n = getattr(unp_c, "co_posonlyargcount", 0)
                        kw_cap_n = unp_c.co_argcount + unp_c.co_kwonlyargcount
                        for pname in unp_c.co_varnames[pos_only_n:kw_cap_n]:
                            if pname in candidate_keys:
                                c_consts.add(pname)
                    active_cells = cross_tainted_cells | round_tainted_cells
                    for sub_c, called_t in all_taint_codes.values():
                        sub_n_p = (
                            sub_c.co_argcount
                            + sub_c.co_kwonlyargcount
                            + (1 if (sub_c.co_flags & inspect.CO_VARARGS) else 0)
                            + (1 if (sub_c.co_flags & inspect.CO_VARKEYWORDS) else 0)
                        )
                        sub_p = (
                            frozenset(str(v) for v in sub_c.co_varnames[:sub_n_p])
                            if called_t
                            else frozenset()
                        )
                        def_p = local_code_def_taints.get(id(sub_c), frozenset())
                        sub_child_map = cell_owner_by_code.get(id(sub_c), {})
                        free_t = frozenset(
                            str(fv)
                            for fv in sub_c.co_freevars
                            if sub_child_map.get(str(fv)) is not None
                            and (sub_child_map[str(fv)], str(fv)) in active_cells
                        )
                        own_cell_t = frozenset(
                            str(cv)
                            for cv in sub_c.co_cellvars
                            if (id(sub_c), str(cv)) in cross_tainted_cells
                        )
                        free_clos = frozenset(
                            str(fv)
                            for fv in sub_c.co_freevars
                            if sub_child_map.get(str(fv)) is not None
                            and (sub_child_map[str(fv)], str(fv)) in round_closure_cells
                        )
                        next_taint = (
                            (sub_p | def_p | free_t | own_cell_t)
                            if c_taint is not None
                            else (sub_p | def_p)
                        )
                        code_stack.append((sub_c, next_taint, free_clos))
                peek_q = [sc for cid_p, sc in peek_candidates.items() if cid_p not in visited_taint_cids]
                peek_seen: set[int] = set(visited_taint_cids)
                peek_outer_defs = dict(local_code_defs)
                peek_order: List[types.CodeType] = []
                peek_limit = max(64, max_ret_rounds * 2, len(peek_candidates) + 64)
                while peek_q and len(peek_seen) < peek_limit:
                    p_c = peek_q.pop()
                    pid = id(p_c)
                    if pid in peek_seen:
                        continue
                    peek_seen.add(pid)
                    peek_order.append(p_c)
                    p_res = _analyze_code_kw_flow(
                        p_c,
                        frozenset(),
                        frozenset(),
                        frozenset(),
                        known_code_returns=known_code_returns,
                        outer_code_defs=peek_outer_defs,
                        known_code_called_params=known_code_called_params,
                        ext_target_codes=ext_target_codes,
                    )
                    peek_outer_defs.update(p_res[10])
                    for cid_dt, tp_dt in p_res[11].items():
                        local_code_def_taints[cid_dt] = (
                            local_code_def_taints.get(cid_dt, frozenset()) | tp_dt
                        )
                    p_ret = p_res[14]
                    if p_ret:
                        known_code_returns[pid] = known_code_returns.get(pid, frozenset()) | p_ret
                    p_cp = p_res[17]
                    if p_cp:
                        known_code_called_params[pid] = (
                            known_code_called_params.get(pid, frozenset()) | p_cp
                        )
                    for sub_cid, (sub_pc, _) in p_res[15].items():
                        if sub_cid not in peek_seen:
                            peek_q.append(sub_pc)
                for p_c in reversed(peek_order):
                    pid = id(p_c)
                    p_res = _analyze_code_kw_flow(
                        p_c,
                        frozenset(),
                        frozenset(),
                        frozenset(),
                        known_code_returns=known_code_returns,
                        outer_code_defs=peek_outer_defs,
                        known_code_called_params=known_code_called_params,
                        ext_target_codes=ext_target_codes,
                    )
                    for cid_dt, tp_dt in p_res[11].items():
                        local_code_def_taints[cid_dt] = (
                            local_code_def_taints.get(cid_dt, frozenset()) | tp_dt
                        )
                    p_ret = p_res[14]
                    if p_ret:
                        known_code_returns[pid] = known_code_returns.get(pid, frozenset()) | p_ret
                    p_cp = p_res[17]
                    if p_cp:
                        known_code_called_params[pid] = (
                            known_code_called_params.get(pid, frozenset()) | p_cp
                        )
                if (
                    known_code_returns == prev_kcr
                    and known_code_called_params == prev_kccp
                    and cross_tainted_cells == prev_cells
                ):
                    break
            cached_ci = (
                c_local_names,
                c_u_names,
                c_u_oattrs,
                c_u_lattrs,
                c_a_names,
                c_a_oattrs,
                c_a_lattrs,
                c_consts,
                c_ri_names,
                c_ri_oattrs,
                c_ri_lattrs,
            )
            code_info_cache[cache_key_ci] = cached_ci
        (
            local_names,
            u_names,
            u_oattrs,
            u_lattrs,
            a_names,
            a_oattrs,
            a_lattrs,
            c_consts,
            c_ri_names,
            c_ri_oattrs,
            c_ri_lattrs,
        ) = cached_ci
        referenced.update(c_consts)

        if depth == 0 and "super" in local_names:
            root_calls_super = True

        nxt_depth = depth + 1
        fn_globals = getattr(unwrapped, "__globals__", None)
        if isinstance(fn_globals, Mapping):
            for n in local_names:
                if n in fn_globals:
                    is_unp = n in u_names
                    is_ri = n in c_ri_names
                    is_fwd = is_unp or is_ri or (n in a_names)
                    _consider_val(
                        fn_globals[n],
                        None,
                        local_names,
                        nxt_depth,
                        edge_unpacked=is_unp,
                        edge_can_fwd=is_fwd,
                        ret_invoked=is_ri,
                        via_name=n,
                        unpacked_owner_attrs=u_oattrs,
                        unpacked_leaf_attrs=u_lattrs,
                        arg_owner_attrs=a_oattrs,
                        arg_leaf_attrs=a_lattrs,
                        ret_invoked_names=c_ri_names,
                        ret_invoked_owner_attrs=c_ri_oattrs,
                        ret_invoked_leaf_attrs=c_ri_lattrs,
                        is_root_code=(depth == 0),
                    )

        fn_closure = getattr(unwrapped, "__closure__", None)
        if fn_closure:
            free_names = code.co_freevars if isinstance(getattr(code, "co_freevars", None), tuple) else ()
            for c_idx, cell in enumerate(fn_closure):
                try:
                    val_c = cell.cell_contents
                except ValueError:
                    continue
                c_name = free_names[c_idx] if c_idx < len(free_names) else None
                if (
                    c_name == "__class__"
                    and isinstance(val_c, type)
                    and cur_owner is not None
                    and isinstance(cur_owner, val_c)
                ):
                    continue
                is_unp = c_name in u_names if c_name else False
                is_ri = c_name in c_ri_names if c_name else False
                is_fwd = is_unp or is_ri or (c_name in a_names if c_name else bool(a_names))
                _consider_val(
                    val_c,
                    None,
                    local_names,
                    nxt_depth,
                    edge_unpacked=is_unp,
                    edge_can_fwd=is_fwd,
                    ret_invoked=is_ri,
                    via_name=c_name,
                    unpacked_owner_attrs=u_oattrs,
                    unpacked_leaf_attrs=u_lattrs,
                    arg_owner_attrs=a_oattrs,
                    arg_leaf_attrs=a_lattrs,
                    ret_invoked_names=c_ri_names,
                    ret_invoked_owner_attrs=c_ri_oattrs,
                    ret_invoked_leaf_attrs=c_ri_lattrs,
                    is_root_code=(depth == 0),
                )

        if cur_owner is not None:
            owner_is_type = isinstance(cur_owner, type)
            for n in local_names:
                raw_attr = inspect.getattr_static(cur_owner, n, _UNSET_SLOT)
                if raw_attr is _UNSET_SLOT or isinstance(raw_attr, property):
                    continue
                if isinstance(raw_attr, types.MemberDescriptorType) and not owner_is_type:
                    try:
                        raw_attr = raw_attr.__get__(cur_owner, type(cur_owner))
                    except Exception:
                        continue
                if isinstance(raw_attr, staticmethod):
                    raw_attr = raw_attr.__func__
                    attr_owner = None
                elif isinstance(raw_attr, classmethod):
                    raw_attr = raw_attr.__func__
                    attr_owner = cur_owner if owner_is_type else type(cur_owner)
                else:
                    attr_owner = cur_owner
                is_unp = ("self", n) in u_oattrs or (None, n) in u_oattrs or n in u_lattrs
                is_ri = ("self", n) in c_ri_oattrs or (None, n) in c_ri_oattrs or n in c_ri_lattrs
                is_fwd = (
                    is_unp
                    or is_ri
                    or (("self", n) in a_oattrs or (None, n) in a_oattrs or n in a_lattrs)
                )
                _consider_val(
                    raw_attr,
                    attr_owner,
                    local_names,
                    nxt_depth,
                    edge_unpacked=is_unp,
                    edge_can_fwd=is_fwd,
                    ret_invoked=is_ri,
                    via_name=n,
                    unpacked_owner_attrs=u_oattrs,
                    unpacked_leaf_attrs=u_lattrs,
                    arg_owner_attrs=a_oattrs,
                    arg_leaf_attrs=a_lattrs,
                    ret_invoked_names=c_ri_names,
                    ret_invoked_owner_attrs=c_ri_oattrs,
                    ret_invoked_leaf_attrs=c_ri_lattrs,
                    is_root_code=(depth == 0),
                )

    return referenced, root_calls_super


def _resolve_supported_kwargs(
    fn: Any,
    candidate_kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], frozenset[str]]:
    """Inspect a callable (unwrapping MRO **kwargs overrides) to filter supported kwargs upfront."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {}, frozenset()

    params = sig.parameters
    explicit_names: set[str] = {
        name
        for name, p in params.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_var_kw:
        cand_key_set = frozenset(candidate_kwargs)
        ref_keys, calls_super = _inspect_varkw_wrapper(fn, cand_key_set)
        explicit_names.update(ref_keys)
        owner_self = getattr(fn, "__self__", None)
        method_name = getattr(fn, "__name__", None)
        if calls_super and owner_self is not None and isinstance(method_name, str):
            for base_cls in type(owner_self).__mro__[1:]:
                raw_attr = base_cls.__dict__.get(method_name)
                if raw_attr is None:
                    continue
                target_fn = (
                    raw_attr.__func__
                    if isinstance(raw_attr, (staticmethod, classmethod))
                    else raw_attr
                )
                try:
                    base_sig = inspect.signature(target_fn)
                except (TypeError, ValueError):
                    continue
                base_params = base_sig.parameters
                explicit_names.update(
                    name
                    for name, p in base_params.items()
                    if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                )
                if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in base_params.values()):
                    has_var_kw = False
                    break
                sub_refs, sub_super = _inspect_varkw_wrapper(target_fn, cand_key_set, owner_self=owner_self)
                explicit_names.update(sub_refs)
                if not sub_super:
                    has_var_kw = True
                    break

    kw_fset = frozenset(explicit_names)
    if has_var_kw:
        return dict(candidate_kwargs), frozenset(candidate_kwargs)
    return {k: v for k, v in candidate_kwargs.items() if k in kw_fset}, kw_fset


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


def _parse_jsonl_rows(raw_text: str, *, source: str = "input") -> List[Any]:
    rows: List[Any] = []
    for line_no, line in enumerate(raw_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as ex:
            raise ValueError(f"Invalid JSON in {source} line {line_no}: {ex.msg}") from ex
    return rows


def _validated_head_logit_adjustments(
    raw_adj: Any,
    definition: Any,
    field_name: str,
) -> Optional[np.ndarray]:
    """Validate optional per-class logit adjustments against field definition."""
    if raw_adj is None:
        return None
    if isinstance(definition, BooleanField):
        exp_len = 2
    elif isinstance(definition, ChoiceField):
        exp_len = len(definition.options)
    else:
        raise ValueError(
            f"Field {field_name!r} of type {type(definition).__name__} does not support logit_adjustments"
        )
    if not isinstance(raw_adj, np.ndarray):
        _check_nested_real_seq(raw_adj, f"logit_adjustments for {field_name!r}")
    arr = np.asarray(raw_adj)
    if arr.dtype == np.dtype(np.bool_) or arr.dtype.kind not in ("i", "u", "f"):
        raise ValueError(f"logit_adjustments for {field_name!r} must have numeric dtype, got {arr.dtype}")
    if arr.ndim != 1 or arr.shape != (exp_len,):
        raise ValueError(
            f"logit_adjustments for {field_name!r} shape {arr.shape} != expected ({exp_len},)"
        )
    arr64 = np.asarray(arr, dtype=np.float64)
    if not bool(np.all(np.isfinite(arr64))):
        raise ValueError(f"logit_adjustments for {field_name!r} contains non-finite values")
    return np.array(arr64, dtype=np.float64, copy=True)


def _extract_prompt_raw_norm(
    result_obj: Any,
    prompt_str: str,
    proj_obj: Any,
    *,
    recency_flag: bool = False,
) -> Optional[float]:
    """Retrieve a prompt's pre-normalization norm from an inference result or projector."""
    rn_attr = getattr(result_obj, "raw_norm", None) if result_obj is not None else None
    if rn_attr is not None and not isinstance(rn_attr, (bool, np.bool_)):
        try:
            return float(rn_attr)
        except Exception:
            pass
    if proj_obj is None:
        return None
    raw_norm_fn = getattr(proj_obj, "raw_norm", None)
    if callable(raw_norm_fn):
        try:
            rn_val = (
                raw_norm_fn(prompt_str, recency_weighted=recency_flag)
                if _accepts_recency_weighted(raw_norm_fn)
                else raw_norm_fn(prompt_str)
            )
            if rn_val is not None and not isinstance(rn_val, (bool, np.bool_)):
                return float(rn_val)
        except Exception:
            pass
    pwn_fn = getattr(proj_obj, "project_with_norm", None)
    if callable(pwn_fn):
        try:
            raw_pwn = (
                pwn_fn(prompt_str, recency_weighted=recency_flag)
                if _accepts_recency_weighted(pwn_fn)
                else pwn_fn(prompt_str)
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
            parsed_rows: Any = _parse_jsonl_rows(raw_text, source=str(path))
        else:
            try:
                parsed_rows = json.loads(raw_text)
            except json.JSONDecodeError:
                lines = [line for line in raw_text.splitlines() if line.strip()]
                if len(lines) > 1:
                    parsed_rows = _parse_jsonl_rows(raw_text, source=str(path))
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
        min_calibration_lessons: int = 0,
        min_override_corroborations: int = 1,
        max_calibration_window: int = 512,
        selective_calibrators: Optional[Mapping[str, Any]] = None,
        aci_tracker: Optional[Any] = None,
        joint_calibrator: Optional[Any] = None,
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
        self.forgetting_factor = _validate_forgetting_factor(forgetting_factor)
        self.relative_odds_ratio = (
            float(relative_odds_ratio) if relative_odds_ratio is not None else None
        )
        self.confidence_floor_tau0 = (
            float(confidence_floor_tau0) if confidence_floor_tau0 is not None else None
        )
        self.recency_weighted = bool(recency_weighted)
        self.strict_mode = bool(strict_mode)
        self.default_alpha: float = 0.05
        if int(min_calibration_lessons) < 0:
            raise ValueError(f"min_calibration_lessons must be >= 0, got {min_calibration_lessons}")
        if int(min_override_corroborations) < 1:
            raise ValueError(f"min_override_corroborations must be >= 1, got {min_override_corroborations}")
        if int(max_calibration_window) < 1:
            raise ValueError(f"max_calibration_window must be >= 1, got {max_calibration_window}")
        self.min_calibration_lessons: int = int(min_calibration_lessons)
        self.min_override_corroborations: int = int(min_override_corroborations)
        self.max_calibration_window: int = int(max_calibration_window)
        self.selective_calibrators: Dict[str, Any] = dict(selective_calibrators or {})
        self.aci_tracker: Optional[Any] = aci_tracker
        self.joint_calibrator: Optional[Any] = joint_calibrator
        self._online_lesson_count: int = 0
        self._override_counts: Dict[str, Dict[str, Tuple[Any, int]]] = {}
        self.policy_scope = str(policy_scope)
        self.policy_epoch: int = 0
        self.model_version: int = 1
        self._lock = threading.RLock()
        self.model_path: Optional[Path] = Path(model_path).resolve() if model_path is not None else None
        self.auto_reload: bool = bool(auto_reload)
        self._model_stat: Optional[Tuple[int, int, str]] = None
        self._packed_kernel_cache: Optional[Tuple[Tuple[Any, ...], Any]] = None
        self._explainer_cache: Dict[str, Tuple[Tuple[Any, ...], Any]] = {}

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
        self._packed_kernel_cache = None
        if hasattr(self, "_explainer_cache"):
            self._explainer_cache.clear()
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
        self._override_counts = {}
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
                    self.calibrators[name].logit_adjustments = _validated_head_logit_adjustments(
                        getattr(ch, "logit_adjustments", None),
                        self.schema.fields[name],
                        name,
                    )
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
                adj = getattr(calib, "logit_adjustments", None)
                if adj is not None:
                    h.update(f"{fname}:adj:{np.asarray(adj, dtype=np.float64).tobytes()}".encode("utf-8"))
                m_th = getattr(calib, "margin_threshold", None)
                if m_th is not None:
                    h.update(f"{fname}:mth:{float(m_th):.6f}".encode("utf-8"))
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
        jcal = getattr(self, "joint_calibrator", None)
        if jcal is not None and hasattr(jcal, "digest"):
            h.update(f"joint:{jcal.digest()}".encode("utf-8"))
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
            encode_fn = self.model.encode
            enc_kw, _ = _resolve_supported_kwargs(
                encode_fn,
                {"telemetry": telemetry, "recency_weighted": use_recency},
            )
            return encode_fn(prompt, **enc_kw)
        if self.projector is not None:
            return _encode_projector_one(
                self.projector,
                prompt,
                self.dimension,
                recency_weighted=use_recency,
                zero_unit0=_model_zero_unit0(self.model, self.projector),
            )
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
                    midpoint = 0.5 * float(definition.min_value) + 0.5 * float(definition.max_value)
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
                    eff_rec_cal = bool(getattr(self, "recency_weighted", False))
                    fam_vals = []
                    for i, emb_c in zip(conf_indices, conf_embeddings):
                        rn_c = _extract_prompt_raw_norm(
                            results[i],
                            rows[i][0],
                            proj,
                            recency_flag=eff_rec_cal,
                        )
                        fam_vals.append(float(np.linalg.norm(emb_c)) if rn_c is None else rn_c)
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
                    predictions = _score_from_unit_prob(probs, definition.min_value, definition.max_value)
                    predictor = self.regression_conformal_predictors[name]
                    predictor.calibrate(predictions, conf_labels)
                    scores = predictor.residuals

                if compiled is not None and hasattr(compiled, "calibration_scores"):
                    compiled.temperature = calibrator.temperature
                    compiled.logit_adjustments = _validated_head_logit_adjustments(
                        getattr(calibrator, "logit_adjustments", None),
                        definition,
                        name,
                    )
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
                    upper = (
                        _score_full_span_clamped(definition.min_value, definition.max_value)
                        if isinstance(definition, ScoreField)
                        else 1.0
                    )
                    compiled.conformal_quantile = float(scores[k - 1]) if k <= len(scores) else upper
            self.model_version = max(self.model_version, int(getattr(self.model, "model_version", 1))) + 1
            if hasattr(self.model, "model_version"):
                self.model.model_version = self.model_version
            self._packed_kernel_cache = None
            if hasattr(self, "_explainer_cache"):
                self._explainer_cache.clear()
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
                    upper_p = _score_full_span_clamped(definition.min_value, definition.max_value) if isinstance(definition, ScoreField) else 1.0
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
                    p_ratio = _stable_sigmoid(logits.flatten() / max(1e-4, prev_temp), temperature=1.0)
                    with np.errstate(over="ignore", invalid="ignore"):
                        y_ratio = np.clip(
                            _score_to_unit_ratio(
                                np.asarray(labels, dtype=np.float64),
                                definition.min_value,
                                definition.max_value,
                            ),
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
                    eff_rec_rcal = bool(getattr(self, "recency_weighted", False))
                    fam_vals = []
                    for i, emb_c in zip(conf_indices, conf_embeddings):
                        rn_c = _extract_prompt_raw_norm(
                            results[i],
                            rows[i][0],
                            proj,
                            recency_flag=eff_rec_rcal,
                        )
                        fam_vals.append(float(np.linalg.norm(emb_c)) if rn_c is None else rn_c)
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
                    predictions = _score_from_unit_prob(probs, definition.min_value, definition.max_value)
                    predictor = self.regression_conformal_predictors[name]
                    predictor.calibrate(predictions, conf_labels)
                    scores = predictor.residuals
                    k = int(math.ceil((len(scores) + 1) * (1.0 - alpha_val)))
                    upper = _score_full_span_clamped(definition.min_value, definition.max_value)
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
                    compiled.logit_adjustments = _validated_head_logit_adjustments(
                        getattr(calibrator, "logit_adjustments", None),
                        definition,
                        name,
                    )
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
            self._packed_kernel_cache = None
            if hasattr(self, "_explainer_cache"):
                self._explainer_cache.clear()
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

    def calibrate_joint(
        self,
        train_examples: Sequence[Tuple[str, Mapping[str, Any]]],
        calibration_examples: Sequence[Tuple[str, Mapping[str, Any]]],
        *,
        gamma: float = 1.0,
        alpha: Optional[float] = None,
        delta: float = 0.05,
        score_kind: str = "margin",
        grid_size: int = 25,
        max_frac: float = 0.95,
    ) -> Any:
        """Fit pairwise factor-graph MAP structure on train_examples and Joint LTT on calibration_examples."""
        if not train_examples:
            raise ValueError("train_examples must be non-empty")
        if not calibration_examples:
            raise ValueError("calibration_examples must be non-empty")

        eff_alpha = self.default_alpha if alpha is None else float(alpha)
        jcal = JointSchemaCalibrator(
            self.schema,
            gamma=gamma,
            alpha=eff_alpha,
            delta=delta,
            score_kind=score_kind,
            grid_size=grid_size,
            max_frac=max_frac,
        )
        jcal.fit_structure([lbl for _, lbl in train_examples])

        with self._lock:
            prev_jcal = self.joint_calibrator
            self.joint_calibrator = None
            try:
                cal_prompts = [p for p, _ in calibration_examples]
                decs = self.decide_batch(
                    cal_prompts,
                    alpha=eff_alpha,
                    strict=False,
                    record_receipt=False,
                    use_cache=False,
                )
                calib_probs: Dict[str, np.ndarray] = {}
                for jf in jcal.field_names:
                    f_def = self.schema.fields[jf]
                    if isinstance(f_def, BooleanField):
                        calib_probs[jf] = np.array(
                            [
                                [d.probabilities[jf]["False"], d.probabilities[jf]["True"]]
                                for d in decs
                            ],
                            dtype=np.float64,
                        )
                    else:
                        opts = f_def.options
                        calib_probs[jf] = np.array(
                            [[d.probabilities[jf][o] for o in opts] for d in decs],
                            dtype=np.float64,
                        )
                cert = jcal.calibrate(calib_probs, [lbl for _, lbl in calibration_examples])
                self.joint_calibrator = jcal
                self._packed_kernel_cache = None
                if hasattr(self, "_explainer_cache"):
                    self._explainer_cache.clear()
                if self.cache is not None:
                    self.cache.clear()
                if hasattr(self.model, "cache") and getattr(self.model, "cache", None) is not None:
                    self.model.cache.clear()
                return cert
            except Exception:
                self.joint_calibrator = prev_jcal
                raise

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
        eff_use_cache = bool(use_cache)

        if not eff_use_cache and all(telem is None for _, _, telem in normalized_rows):
            decisions = self.decide_batch(
                [prompt for prompt, _, _ in normalized_rows],
                alpha=eff_alpha,
                record_receipt=False,
                strict=eff_strict,
                use_cache=False,
            )
        else:
            decisions = [
                self.decide(
                    prompt,
                    telemetry=telemetry,
                    alpha=eff_alpha,
                    record_receipt=False,
                    strict=eff_strict,
                    use_cache=eff_use_cache,
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
            f_mae_int_sum = 0
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
                    full_span_s = _score_full_span_clamped(f_def.min_value, f_def.max_value)
                    cov_slack = min(1e-6, 1e-6 * full_span_s) if full_span_s > 0.0 else 0.0
                    str_slack = min(1e-4, 1e-4 * full_span_s) if full_span_s > 0.0 else 0.0
                    covered_i = False
                    rcp = self.regression_conformal_predictors.get(fname)
                    if is_unfamiliar or not c_set:
                        covered_i = False
                    elif rcp is not None and rcp.is_calibrated:
                        interval = rcp.predict_interval(float(pred_val), alpha=eff_alpha, strict=eff_strict)
                        covered_i = (interval.lower_bound - cov_slack) <= float(gt_val) <= (interval.upper_bound + cov_slack)
                    elif isinstance(c_set[0], str) and c_set[0].startswith("[") and c_set[0].endswith("]"):
                        try:
                            parts = c_set[0][1:-1].split(",")
                            lo_v, hi_v = float(parts[0].strip()), float(parts[1].strip())
                            covered_i = (lo_v - str_slack) <= float(gt_val) <= (hi_v + str_slack)
                        except (ValueError, IndexError):
                            covered_i = False
                    f_mae_int_sum += _score_abs_diff_exact_2m1074(float(pred_val), float(gt_val))
                    correct_i = _score_within_rel_tolerance(
                        float(pred_val),
                        float(gt_val),
                        f_def.min_value,
                        f_def.max_value,
                        0.10,
                    )
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
                        [float(_score_to_unit_ratio(float(gt_val), f_def.min_value, f_def.max_value)) for _, _, gt_val in f_rows],
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
                mae=_score_mae_from_int_sum(f_mae_int_sum, n_f) if (isinstance(f_def, ScoreField) and n_f > 0) else 0.0,
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
                alpha = self.aci_tracker.alpha_t if self.aci_tracker is not None else self.default_alpha
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
                raw_emb = _assert_real_numeric_array_like(embedding, "embedding")
                with np.errstate(over="ignore", invalid="ignore"):
                    arr_emb = np.asarray(raw_emb, dtype=np.float32).flatten()
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
                    if bool(np.all(np.isfinite(query_emb))):
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
                fwd_fn = self.model.forward_single
                fwd_kw, _ = _resolve_supported_kwargs(
                    fwd_fn,
                    {
                        "telemetry": telemetry,
                        "embedding": user_emb,
                        "recency_weighted": recency_weighted,
                        "use_cache": False,
                    },
                )
                raw_result = fwd_fn(prompt, **fwd_kw)

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
            needs_raw_norm = (
                any(
                    float(getattr(cp, "familiarity_threshold", 0.0)) > 0.0
                    for cp in self.conformal_predictors.values()
                )
                or any(
                    float(getattr(ch, "familiarity_threshold", 0.0)) > 0.0
                    for ch in getattr(self.model, "heads", {}).values()
                )
                or any(
                    float(getattr(ch, "familiarity_threshold", 0.0)) > 0.0
                    for ch in getattr(self.model, "_field_heads", {}).values()
                )
            )
            if prompt_raw_norm is None and embedding is None and needs_raw_norm:
                eff_rec = (
                    bool(self.recency_weighted)
                    if recency_weighted is None
                    else bool(recency_weighted)
                )
                prompt_raw_norm = _extract_prompt_raw_norm(
                    None, prompt, self.projector, recency_flag=eff_rec
                )

            emb_finite = bool(np.all(np.isfinite(raw_result.embedding)))
            learned_overrides: Dict[str, Any] = {}
            if emb_finite and embedding is None and telemetry is None and isinstance(prompt, str) and prompt.strip():
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
                conformal = self.conformal_predictors.get(name)
                live_h = (
                    self.model._field_heads.get(name)
                    if hasattr(self.model, "_field_heads") and isinstance(self.model._field_heads, Mapping)
                    else None
                )
                head_obj = self.model.heads.get(name) if hasattr(self.model, "heads") else None
                fam_thresh = max(
                    float(getattr(live_h, "familiarity_threshold", 0.0)) if live_h is not None else 0.0,
                    float(getattr(head_obj, "familiarity_threshold", 0.0)) if head_obj is not None else 0.0,
                    float(getattr(conformal, "familiarity_threshold", 0.0)) if conformal is not None else 0.0,
                )
                proto_thresh = max(
                    float(getattr(live_h, "prototype_threshold", 0.0)) if live_h is not None else 0.0,
                    float(getattr(head_obj, "prototype_threshold", 0.0)) if head_obj is not None else 0.0,
                    float(getattr(conformal, "prototype_threshold", 0.0)) if conformal is not None else 0.0,
                )
                is_zero_emb = not bool(np.any(raw_result.embedding))
                enforce_calibrated_ood = (not is_zero_emb) and (fam_thresh > 0.0 or proto_thresh > 0.0)

                proto_sim: Optional[float] = None
                if (not has_learned_override) or enforce_calibrated_ood:
                    if hasattr(self.model, "compute_prototype_similarity") and callable(self.model.compute_prototype_similarity):
                        try:
                            proto_sim = self.model.compute_prototype_similarity(name, raw_result.embedding)
                        except Exception:
                            proto_sim = None
                    if proto_sim is None and conformal is not None and getattr(conformal, "prototypes", None) is not None:
                        try:
                            proto_sim = conformal.compute_prototype_similarity(raw_result.embedding)
                        except Exception:
                            proto_sim = None
                    if proto_sim is not None:
                        prototype_similarities[name] = float(proto_sim)

                ood_blocked = False
                if enforce_calibrated_ood:
                    if (
                        fam_thresh > 0.0
                        and prompt_raw_norm is not None
                        and (not math.isfinite(prompt_raw_norm) or prompt_raw_norm < fam_thresh - 1e-12)
                    ):
                        ood_blocked = True
                    if proto_thresh > 0.0 and proto_sim is not None and proto_sim < proto_thresh - 1e-6:
                        ood_blocked = True

                if isinstance(f_def, ChoiceField):
                    with np.errstate(over="ignore", invalid="ignore"):
                        calibrated_p = calibrator.calibrate_logits(raw_eval.logits)
                    logits_finite = bool(np.all(np.isfinite(raw_eval.logits)))
                    if not (emb_finite and logits_finite and bool(np.all(np.isfinite(calibrated_p)))):
                        values[name] = f_def.options[0]
                        confidences[name] = float("nan")
                        probabilities[name] = {opt: float("nan") for opt in f_def.options}
                        conformal_sets[name] = []
                        abstention_reasons[name] = "non_finite_input"
                        ambiguous_fields.append(name)
                        if getattr(f_def, "escalate_on_ambiguity", True):
                            escalated_fields.append(name)
                        is_ambiguous = True
                        continue
                    best_idx = int(np.argmax(calibrated_p))
                    selected_choice = f_def.options[best_idx]
                    conf = float(calibrated_p[best_idx])

                    prob_dict = {opt: float(calibrated_p[i]) for i, opt in enumerate(f_def.options)}
                    values[name] = selected_choice
                    confidences[name] = conf
                    probabilities[name] = prob_dict

                    override_matched = (
                        has_learned_override
                        and not ood_blocked
                        and (is_zero_emb or selected_choice == learned_overrides.get(name))
                    )
                    if conformal is not None and not override_matched:
                        ps_fn = conformal.predict_set
                        ps_kw, _ = _resolve_supported_kwargs(
                            ps_fn,
                            {
                                "alpha": alpha,
                                "margin_threshold": eff_m_thresh if (self.enable_margin_gating and not is_strict) else 0.0,
                                "relative_odds_ratio": eff_gamma if not is_strict else None,
                                "confidence_floor_tau0": eff_tau0 if not is_strict else None,
                                "strict": is_strict,
                                "familiarity_score": (None if (has_learned_override and not enforce_calibrated_ood) else prompt_raw_norm),
                                "prototype_similarity": (None if (has_learned_override and not enforce_calibrated_ood) else proto_sim),
                            },
                        )
                        cset = ps_fn(calibrated_p, **ps_kw)
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
                        pred_excluded = selected_choice not in cset.prediction_set
                        if is_strict:
                            field_ambiguous = (
                                (cardinality != 1)
                                or cset.is_ambiguous
                                or is_empty_set
                                or pred_excluded
                            )
                        else:
                            field_ambiguous = is_empty_set or cset.is_ambiguous or pred_excluded

                        if field_ambiguous:
                            ambiguous_fields.append(name)
                            if getattr(f_def, "escalate_on_ambiguity", True):
                                is_ambiguous = True
                                escalated_fields.append(name)
                    else:
                        conformal_sets[name] = [selected_choice]
                        margins[name] = 1.0

                elif isinstance(f_def, BooleanField):
                    with np.errstate(over="ignore", invalid="ignore"):
                        calibrated_p = calibrator.calibrate_logits(raw_eval.logits)
                    logits_finite = bool(np.all(np.isfinite(raw_eval.logits)))
                    if not (emb_finite and logits_finite and bool(np.all(np.isfinite(calibrated_p)))):
                        values[name] = False
                        confidences[name] = float("nan")
                        probabilities[name] = {"False": float("nan"), "True": float("nan")}
                        conformal_sets[name] = []
                        abstention_reasons[name] = "non_finite_input"
                        ambiguous_fields.append(name)
                        if getattr(f_def, "escalate_on_ambiguity", True):
                            escalated_fields.append(name)
                        is_ambiguous = True
                        continue
                    prob_true = float(calibrated_p[1])
                    val = bool(prob_true >= f_def.threshold)
                    conf = prob_true if val else (1.0 - prob_true)

                    values[name] = val
                    confidences[name] = conf
                    probabilities[name] = {"False": float(calibrated_p[0]), "True": prob_true}

                    override_matched = (
                        has_learned_override
                        and not ood_blocked
                        and (is_zero_emb or val == bool(learned_overrides.get(name)))
                    )
                    if conformal is not None and not override_matched:
                        ps_fn = conformal.predict_set
                        ps_kw, _ = _resolve_supported_kwargs(
                            ps_fn,
                            {
                                "alpha": alpha,
                                "margin_threshold": eff_m_thresh if (self.enable_margin_gating and not is_strict) else 0.0,
                                "relative_odds_ratio": eff_gamma if not is_strict else None,
                                "confidence_floor_tau0": eff_tau0 if not is_strict else None,
                                "strict": is_strict,
                                "familiarity_score": (None if (has_learned_override and not enforce_calibrated_ood) else prompt_raw_norm),
                                "prototype_similarity": (None if (has_learned_override and not enforce_calibrated_ood) else proto_sim),
                            },
                        )
                        cset = ps_fn(calibrated_p, **ps_kw)
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
                        pred_excluded = ("True" if val else "False") not in cset.prediction_set
                        if is_strict:
                            field_ambiguous = (
                                (cardinality != 1)
                                or cset.is_ambiguous
                                or is_empty_set
                                or pred_excluded
                            )
                        else:
                            field_ambiguous = is_empty_set or cset.is_ambiguous or pred_excluded

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
                    with np.errstate(over="ignore", invalid="ignore"):
                        scaled_logits = raw_eval.logits / max(1e-4, calibrated_temp)
                        probs = _stable_sigmoid(scaled_logits, temperature=1.0)
                    logits_finite = bool(np.all(np.isfinite(raw_eval.logits))) and bool(
                        np.all(np.isfinite(scaled_logits))
                    )
                    if not (emb_finite and logits_finite and bool(np.all(np.isfinite(probs)))):
                        values[name] = ()
                        confidences[name] = float("nan")
                        probabilities[name] = {opt: float("nan") for opt in f_def.options}
                        conformal_sets[name] = []
                        abstention_reasons[name] = "non_finite_input"
                        ambiguous_fields.append(name)
                        if getattr(f_def, "escalate_on_ambiguity", True):
                            escalated_fields.append(name)
                        is_ambiguous = True
                        continue
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
                    with np.errstate(over="ignore", invalid="ignore"):
                        raw_logit = float(raw_eval.logits[0])
                        scaled_logit = float(raw_logit / max(1e-4, calibrated_temp))
                        prob = float(_stable_sigmoid(np.array([scaled_logit]), temperature=1.0)[0])
                        val = float(_score_from_unit_prob(prob, f_def.min_value, f_def.max_value))
                        conf = float(getattr(raw_eval, "confidence", max(prob, 1.0 - prob)))
                    logits_finite = math.isfinite(raw_logit) and math.isfinite(scaled_logit)
                    if not (emb_finite and logits_finite and math.isfinite(val) and math.isfinite(conf)):
                        values[name] = (
                            float(f_def.default)
                            if getattr(f_def, "default", None) is not None
                            else float(f_def.min_value)
                        )
                        confidences[name] = float("nan")
                        probabilities[name] = {"score_ratio": float("nan")}
                        conformal_sets[name] = []
                        abstention_reasons[name] = "non_finite_input"
                        ambiguous_fields.append(name)
                        if getattr(f_def, "escalate_on_ambiguity", True):
                            escalated_fields.append(name)
                        is_ambiguous = True
                        continue
                    values[name] = val
                    confidences[name] = conf
                    probabilities[name] = {"score_ratio": prob}
                    reg_conformal = self.regression_conformal_predictors.get(name)
                    is_uncalibrated = reg_conformal is None or not reg_conformal.is_calibrated
                    half_span = _score_half_span(f_def.min_value, f_def.max_value)
                    if not is_uncalibrated and not has_learned_override:
                        interval = reg_conformal.predict_interval(val, alpha=alpha, strict=is_strict)
                        conformal_sets[name] = [interval.to_interval_string()]
                        interval_margin = interval.margin
                    else:
                        if is_strict and not has_learned_override:
                            interval_margin = _score_full_span_clamped(f_def.min_value, f_def.max_value)
                            low_b = f_def.min_value
                            high_b = f_def.max_value
                        else:
                            interval_margin = (1.0 - alpha) * half_span if not has_learned_override else 0.0
                            low_b, high_b = _score_interval_bounds(val, interval_margin, f_def.min_value, f_def.max_value)
                        conformal_sets[name] = [_format_score_interval(low_b, high_b)]

                    if has_learned_override:
                        normalized_margin = 0.0
                    else:
                        normalized_margin = _score_normalized_margin(
                            float(interval_margin),
                            f_def.min_value,
                            f_def.max_value,
                        )
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
                    live_h = (
                        self.model._field_heads.get(name)
                        if hasattr(self.model, "_field_heads") and isinstance(self.model._field_heads, Mapping)
                        else None
                    )
                    head_obj = self.model.heads.get(name) if hasattr(self.model, "heads") else None
                    cp_obj = self.conformal_predictors.get(name)
                    fam_t = max(
                        float(getattr(live_h, "familiarity_threshold", 0.0)) if live_h is not None else 0.0,
                        float(getattr(head_obj, "familiarity_threshold", 0.0)) if head_obj is not None else 0.0,
                        float(getattr(cp_obj, "familiarity_threshold", 0.0)) if cp_obj is not None else 0.0,
                    )
                    if name in learned_overrides and fam_t <= 0.0:
                        continue
                    if fam_t > 0.0 and (not math.isfinite(prompt_raw_norm) or prompt_raw_norm < fam_t - 1e-12):
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
                live_h = (
                    self.model._field_heads.get(name)
                    if hasattr(self.model, "_field_heads") and isinstance(self.model._field_heads, Mapping)
                    else None
                )
                head_obj = self.model.heads.get(name) if hasattr(self.model, "heads") else None
                cp_obj = self.conformal_predictors.get(name)
                proto_t = max(
                    float(getattr(live_h, "prototype_threshold", 0.0)) if live_h is not None else 0.0,
                    float(getattr(head_obj, "prototype_threshold", 0.0)) if head_obj is not None else 0.0,
                    float(getattr(cp_obj, "prototype_threshold", 0.0)) if cp_obj is not None else 0.0,
                )
                if name in learned_overrides and proto_t <= 0.0:
                    continue
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

            if is_strict:
                for name, definition in self.schema.fields.items():
                    if name in learned_overrides:
                        continue
                    sel_cal = self.selective_calibrators.get(name) if hasattr(self, "selective_calibrators") else None
                    if sel_cal is not None:
                        score_kind = getattr(sel_cal, "score_name", "margin")
                        if score_kind in ("p1", "top_prob", "confidence"):
                            sel_val = float(confidences.get(name, 0.0))
                        else:
                            sel_val = float(margins.get(name, 0.0))
                        if not sel_cal.should_accept(sel_val):
                            conformal_sets[name] = []
                            abstention_reasons.setdefault(name, "selective_risk_gate")
                            if name not in ambiguous_fields:
                                ambiguous_fields.append(name)
                            if getattr(definition, "escalate_on_ambiguity", True):
                                is_ambiguous = True
                                if name not in escalated_fields:
                                    escalated_fields.append(name)
                    if (
                        getattr(self, "aci_tracker", None) is not None
                        and getattr(self.aci_tracker, "margin_floor", None) is not None
                        and name in margins
                        and not self.aci_tracker.should_accept_margin(margins[name])
                    ):
                        conformal_sets[name] = []
                        abstention_reasons.setdefault(name, "aci_margin_floor")
                        if name not in ambiguous_fields:
                            ambiguous_fields.append(name)
                        if getattr(definition, "escalate_on_ambiguity", True):
                            is_ambiguous = True
                            if name not in escalated_fields:
                                escalated_fields.append(name)

            joint_extra_meta: Optional[Dict[str, Any]] = None
            jcal = getattr(self, "joint_calibrator", None)
            if (
                jcal is not None
                and hasattr(jcal, "decode_batch")
                and not abstention_reasons
                and not learned_overrides
            ):
                joint_fields = tuple(getattr(jcal, "field_names", ()))
                if joint_fields and all(jf in probabilities for jf in joint_fields):
                    probs_by_field_1: Dict[str, np.ndarray] = {}
                    for jf in joint_fields:
                        jf_def = self.schema.fields[jf]
                        if isinstance(jf_def, BooleanField):
                            probs_by_field_1[jf] = np.array(
                                [[probabilities[jf]["False"], probabilities[jf]["True"]]],
                                dtype=np.float64,
                            )
                        else:
                            probs_by_field_1[jf] = np.array(
                                [[probabilities[jf][o] for o in jf_def.options]],
                                dtype=np.float64,
                            )
                    j_dicts, j_probs, j_margs, j_acc = jcal.decode_batch(probs_by_field_1)
                    values.update(j_dicts[0])
                    acc_0 = bool(j_acc[0])
                    joint_extra_meta = {
                        "joint_map_prob": float(j_probs[0]),
                        "joint_map_margin": float(j_margs[0]),
                        "joint_accepted": acc_0,
                    }
                    if is_strict:
                        if acc_0:
                            for jf in joint_fields:
                                conformal_sets[jf] = [str(j_dicts[0][jf])]
                            ambiguous_fields = [f for f in ambiguous_fields if f not in joint_fields]
                            escalated_fields = [f for f in escalated_fields if f not in joint_fields]
                            is_ambiguous = bool(escalated_fields)
                        else:
                            for jf in joint_fields:
                                conformal_sets[jf] = []
                                abstention_reasons.setdefault(jf, "joint_selective_risk_gate")
                                if jf not in ambiguous_fields:
                                    ambiguous_fields.append(jf)
                                if getattr(self.schema.fields[jf], "escalate_on_ambiguity", True):
                                    is_ambiguous = True
                                    if jf not in escalated_fields:
                                        escalated_fields.append(jf)

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
            result_meta: Dict[str, Any] = {
                "familiarity_score": prompt_raw_norm,
                "familiarity_triggered": familiarity_triggered,
                "prototype_similarities": prototype_similarities,
                "prototype_triggered": prototype_triggered,
                "abstention_reason": primary_abstention_reason,
                "abstention_reasons": abstention_reasons,
            }
            if joint_extra_meta is not None:
                result_meta.update(joint_extra_meta)
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
                metadata=result_meta,
            )

            if eff_use_cache and not is_ambiguous and emb_finite:
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
            if not isinstance(prompt, str):
                raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}")
            recency_weighted = self.recency_weighted if recency_weighted is None else bool(recency_weighted)
            target = self.schema.validate_lesson(target, allow_unknown=True)
            eff_forgetting = _validate_forgetting_factor(
                forgetting_factor if forgetting_factor is not None else self.forgetting_factor
            )
            eff_reward_scale_in = _validate_real_scalar(reward_scale, "reward_scale")
            eff_reward_scale = float(np.clip(eff_reward_scale_in, 0.0, 1.0))
            t0 = time.perf_counter()

            c_emb: Optional[np.ndarray] = None
            allow_prompt_for_head = (
                embedding is None and isinstance(prompt, str) and bool(prompt.strip())
            )
            if embedding is not None:
                emb = _coerce_real_1d_f32(embedding, "embedding")
            else:
                from system1.core.model import DeterministicSemanticProjector

                heads_map = getattr(self.model, "heads", {})
                needs_concept = any(
                    f_name in heads_map
                    and getattr(heads_map[f_name], "_exemplar_bank", None) is not None
                    and not getattr(heads_map[f_name], "hierarchical", False)
                    for f_name in target
                )
                if (
                    needs_concept
                    and telemetry is None
                    and type(self.projector) is DeterministicSemanticProjector
                    and isinstance(prompt, str)
                    and prompt.strip()
                ):
                    emb_raw, c_emb = self.projector.project_both(
                        prompt,
                        recency_weighted=recency_weighted,
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

            model = self.model
            seen_head_ids: set[int] = set()
            model_state_snaps: List[Tuple[Any, Any]] = []
            model_dict_snaps: List[Tuple[Any, Dict[str, Any]]] = []
            model_slot_snaps: List[Tuple[Any, List[Tuple[str, Any]]]] = []
            model_container_snaps: List[Tuple[Any, ...]] = []
            snap_memo: Dict[int, Any] = {}
            skip_snap_ids: set[int] = {id(self), id(model)}
            if self.projector is not None:
                skip_snap_ids.add(id(self.projector))
            m_proj = getattr(model, "projector", None)
            if m_proj is not None:
                skip_snap_ids.add(id(m_proj))
            head_map_snaps: List[Tuple[str, bool, Any, Optional[List[Tuple[Any, Any]]]]] = []
            for attr_name in ("_field_heads", "heads"):
                has_attr = hasattr(model, attr_name)
                h_map = getattr(model, attr_name, None)
                if has_attr:
                    map_items = list(h_map.items()) if isinstance(h_map, dict) else None
                    head_map_snaps.append((attr_name, True, h_map, map_items))
                else:
                    head_map_snaps.append((attr_name, False, None, None))
                if isinstance(h_map, Mapping):
                    for h_obj in h_map.values():
                        _snapshot_duck_graph(
                            h_obj,
                            seen_head_ids,
                            model_state_snaps,
                            model_dict_snaps,
                            model_slot_snaps,
                            snap_memo,
                            skip_snap_ids,
                            model_container_snaps,
                        )
            skip_model_attrs = frozenset({
                "schema",
                "projector",
                "_lock",
                "heads",
                "_field_heads",
                "model_version",
                "_learned_prompt_overrides",
                "_learned_prompt_override_recency",
                "_online_prototype_bank",
                "cache",
            })
            orig_duck_model_keys: Optional[set[str]] = None
            duck_model_extra_snap: Optional[Dict[str, Any]] = None
            if type(getattr(model, "__dict__", None)) is dict:
                orig_duck_model_keys = set(model.__dict__.keys())
                duck_model_extra_snap = {
                    mk: _snapshot_duck_graph(
                        mv,
                        seen_head_ids,
                        model_state_snaps,
                        model_dict_snaps,
                        model_slot_snaps,
                        snap_memo,
                        skip_snap_ids,
                        model_container_snaps,
                    )
                    for mk, mv in model.__dict__.items()
                    if mk not in skip_model_attrs
                }
            model_own_slots = [
                s_n for s_n in _collect_instance_slot_names(model) if s_n not in skip_model_attrs
            ]
            if model_own_slots:
                model_slot_snaps.append((
                    model,
                    [
                        (
                            s_n,
                            _snapshot_duck_graph(
                                getattr(model, s_n, _UNSET_SLOT),
                                seen_head_ids,
                                model_state_snaps,
                                model_dict_snaps,
                                model_slot_snaps,
                                snap_memo,
                                skip_snap_ids,
                                model_container_snaps,
                            )
                            if hasattr(model, s_n)
                            else _UNSET_SLOT,
                        )
                        for s_n in model_own_slots
                    ],
                ))

            has_model_ver = hasattr(model, "model_version")
            prev_model_ver = getattr(model, "model_version", None)
            has_m_ov = hasattr(model, "_learned_prompt_overrides")
            m_ov_snap = (
                {k: dict(v) for k, v in model._learned_prompt_overrides.items()}
                if has_m_ov
                else None
            )
            has_m_ov_rec = hasattr(model, "_learned_prompt_override_recency")
            m_ov_rec_snap = (
                dict(model._learned_prompt_override_recency)
                if has_m_ov_rec
                else None
            )
            has_m_proto = hasattr(model, "_online_prototype_bank")
            m_proto_snap = (
                {k: [arr.copy() if isinstance(arr, np.ndarray) else _clone_head_attr_value(arr) for arr in v] for k, v in model._online_prototype_bank.items()}
                if has_m_proto
                else None
            )
            m_cache = getattr(model, "cache", None)
            m_cache_snap = (
                m_cache._snapshot_state()
                if (m_cache is not None and hasattr(m_cache, "_snapshot_state"))
                else None
            )

            prev_eng_ver = self.model_version
            has_lesson_cnt = "_online_lesson_count" in self.__dict__
            prev_lesson_cnt = self.__dict__.get("_online_lesson_count", 0)
            has_eng_ov = hasattr(self, "_learned_prompt_overrides")
            eng_ov_snap = (
                {k: dict(v) for k, v in self._learned_prompt_overrides.items()}
                if has_eng_ov
                else None
            )
            has_eng_ov_rec = hasattr(self, "_learned_prompt_override_recency")
            eng_ov_rec_snap = (
                dict(self._learned_prompt_override_recency)
                if has_eng_ov_rec
                else None
            )
            has_eng_ov_cnt = hasattr(self, "_override_counts")
            eng_ov_cnt_snap = (
                {k: dict(v) for k, v in self._override_counts.items()}
                if has_eng_ov_cnt
                else None
            )
            cal_snap = {k: copy.deepcopy(v) for k, v in self.calibrators.items()}
            cp_snap = {k: copy.deepcopy(v) for k, v in self.conformal_predictors.items()}
            rcp_snap = {k: copy.deepcopy(v) for k, v in self.regression_conformal_predictors.items()}
            prev_packed_kernel = getattr(self, "_packed_kernel_cache", None)
            has_exp_cache = hasattr(self, "_explainer_cache")
            exp_cache_snap = dict(self._explainer_cache) if has_exp_cache else None
            eng_cache_snap = (
                self.cache._snapshot_state()
                if (self.cache is not None and hasattr(self.cache, "_snapshot_state"))
                else None
            )

            try:
                if hasattr(self.model, "learn_from_tier2"):
                    learn_fn = self.model.learn_from_tier2
                    candidate_learn_kw = {
                        "telemetry": telemetry,
                        "embedding": emb,
                        "forgetting_factor": eff_forgetting,
                        "recency_weighted": recency_weighted,
                        "preserve_calibration": preserve_calibration,
                        "reward_scale": eff_reward_scale,
                        "concept_embedding": c_emb,
                        "allow_prompt_for_head": allow_prompt_for_head,
                    }
                    learn_kw, explicit_learn_names = _resolve_supported_kwargs(learn_fn, candidate_learn_kw)
                    if (
                        "embedding" in learn_kw
                        and embedding is None
                        and getattr(self.model, "projector", None) is self.projector
                        and "reward_scale" in explicit_learn_names
                        and "allow_prompt_for_head" not in explicit_learn_names
                    ):
                        learn_kw["embedding"] = None
                    res_dict = learn_fn(prompt, target, **learn_kw)
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
                                    ou_fn = head.online_update
                                    ou_kw, _ = _resolve_supported_kwargs(
                                        ou_fn,
                                        {
                                            "forgetting_factor": eff_forgetting,
                                            "preserve_calibration": preserve_calibration,
                                            "reward_scale": eff_reward_scale,
                                        },
                                    )
                                    dt = ou_fn(x_aug, y_target, **ou_kw)
                                update_durations_ms.append(dt)
                                updated_fields.append(f_name)

                    rank1_ms = float(sum(update_durations_ms)) if update_durations_ms else 0.0

                self.model_version += 1
                self._online_lesson_count = getattr(self, "_online_lesson_count", 0) + 1
                if hasattr(self.model, "model_version"):
                    self.model.model_version = self.model_version

                min_cal_lessons = getattr(self, "min_calibration_lessons", 0)
                min_corrob = max(1, int(getattr(self, "min_override_corroborations", 1)))

                if not preserve_calibration:
                    norm_key_clear = (
                        " ".join(prompt.strip().casefold().split())
                        if (embedding is None and bool(np.any(emb)) and telemetry is None and isinstance(prompt, str) and prompt.strip())
                        else None
                    )
                    for f_name in updated_fields:
                        if norm_key_clear is not None:
                            if getattr(self, "_override_counts", None) and norm_key_clear in self._override_counts:
                                self._override_counts[norm_key_clear].pop(f_name, None)
                                if not self._override_counts[norm_key_clear]:
                                    self._override_counts.pop(norm_key_clear, None)
                            if getattr(self, "_learned_prompt_overrides", None) and norm_key_clear in self._learned_prompt_overrides:
                                self._learned_prompt_overrides[norm_key_clear].pop(f_name, None)
                                if not self._learned_prompt_overrides[norm_key_clear]:
                                    self._learned_prompt_overrides.pop(norm_key_clear, None)
                                    if hasattr(self, "_learned_prompt_override_recency"):
                                        self._learned_prompt_override_recency.pop(norm_key_clear, None)
                        if f_name in self.calibrators:
                            self.calibrators[f_name] = DecisionCalibrator()
                        if (
                            hasattr(self.model, "heads")
                            and isinstance(self.model.heads, Mapping)
                            and f_name in self.model.heads
                        ):
                            self.model.heads[f_name].logit_adjustments = None
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
                            validated_val = self.schema.fields[f_name].validate_value(target_dict_for_override[f_name])
                            if not hasattr(self, "_override_counts"):
                                self._override_counts = {}
                            field_counts = self._override_counts.setdefault(norm_key_override, {})
                            prev_entry = field_counts.get(f_name)
                            if prev_entry is not None and prev_entry[0] == validated_val:
                                corrob_count = prev_entry[1] + 1
                            else:
                                corrob_count = 1
                            field_counts[f_name] = (validated_val, corrob_count)

                            if not hasattr(self, "_learned_prompt_overrides"):
                                self._learned_prompt_overrides = {}
                            if not hasattr(self, "_learned_prompt_override_recency"):
                                self._learned_prompt_override_recency = {}
                            if corrob_count >= min_corrob:
                                while len(self._learned_prompt_overrides) >= 4096 and norm_key_override not in self._learned_prompt_overrides:
                                    oldest_k = next(iter(self._learned_prompt_overrides))
                                    del self._learned_prompt_overrides[oldest_k]
                                    self._learned_prompt_override_recency.pop(oldest_k, None)
                                    self._override_counts.pop(oldest_k, None)
                                self._learned_prompt_overrides.setdefault(norm_key_override, {})[f_name] = validated_val
                                self._learned_prompt_override_recency[norm_key_override] = bool(recency_weighted)
                            elif norm_key_override in self._learned_prompt_overrides:
                                self._learned_prompt_overrides[norm_key_override].pop(f_name, None)
                                if not self._learned_prompt_overrides[norm_key_override]:
                                    self._learned_prompt_overrides.pop(norm_key_override, None)
                                    self._learned_prompt_override_recency.pop(norm_key_override, None)
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
                            if not cp.is_calibrated and self._online_lesson_count >= min_cal_lessons:
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
                            elif not rcp.is_calibrated and self._online_lesson_count >= min_cal_lessons:
                                f_def_s = self.schema.fields[f_name]
                                if isinstance(f_def_s, ScoreField):
                                    half_span_s = _score_half_span(f_def_s.min_value, f_def_s.max_value)
                                    rcp.residuals = np.full(20, 0.10 * half_span_s, dtype=np.float64)
                                    rcp.is_calibrated = True

                self._packed_kernel_cache = None
                res = self.decide(
                    prompt,
                    telemetry=telemetry,
                    embedding=(emb if embedding is not None else None),
                    record_receipt=False,
                    recency_weighted=recency_weighted,
                    strict=self.strict_mode,
                    policy_scope=self.policy_scope,
                    fail_closed_ledger=False,
                    use_cache=False,
                )

                if hasattr(self, "_explainer_cache"):
                    self._explainer_cache.clear()
                if self.use_cache and self.cache is not None:
                    if embedding is None:
                        self.cache.evict_prompt(prompt)
                    self.cache.invalidate_prior_versions(self.model_version)

                if hasattr(self.model, "cache") and self.model.cache is not None:
                    if embedding is None and hasattr(self.model.cache, "evict_prompt"):
                        self.model.cache.evict_prompt(prompt)
                    if hasattr(self.model.cache, "invalidate_prior_versions"):
                        self.model.cache.invalidate_prior_versions(self.model_version)

                if self.use_cache and self.cache is not None and (not res.is_ambiguous or not self.strict_mode):
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
            except BaseException:
                for h_obj, snap in model_state_snaps:
                    h_obj._restore_state(snap)
                for h_obj, dict_snap in model_dict_snaps:
                    h_obj.__dict__.clear()
                    h_obj.__dict__.update(dict_snap)
                _restore_duck_containers(model_container_snaps)
                for slot_obj, slot_vals in model_slot_snaps:
                    for s_name, s_val in slot_vals:
                        try:
                            if s_val is _UNSET_SLOT:
                                if hasattr(slot_obj, s_name):
                                    delattr(slot_obj, s_name)
                            else:
                                setattr(slot_obj, s_name, s_val)
                        except Exception:
                            pass
                for attr_name, existed, orig_map, orig_items in head_map_snaps:
                    if existed:
                        if orig_items is not None and isinstance(orig_map, dict):
                            orig_map.clear()
                            orig_map.update(orig_items)
                        try:
                            setattr(model, attr_name, orig_map)
                        except Exception:
                            pass
                    elif hasattr(model, attr_name):
                        try:
                            delattr(model, attr_name)
                        except Exception:
                            pass
                if orig_duck_model_keys is not None and duck_model_extra_snap is not None:
                    for mk in list(model.__dict__):
                        if mk not in orig_duck_model_keys:
                            try:
                                delattr(model, mk)
                            except Exception:
                                pass
                    for mk, mv in duck_model_extra_snap.items():
                        try:
                            setattr(model, mk, mv)
                        except Exception:
                            pass
                if has_model_ver:
                    model.model_version = prev_model_ver
                if has_m_ov and m_ov_snap is not None:
                    model._learned_prompt_overrides.clear()
                    model._learned_prompt_overrides.update(m_ov_snap)
                elif hasattr(model, "_learned_prompt_overrides"):
                    delattr(model, "_learned_prompt_overrides")
                if has_m_ov_rec and m_ov_rec_snap is not None:
                    model._learned_prompt_override_recency.clear()
                    model._learned_prompt_override_recency.update(m_ov_rec_snap)
                elif hasattr(model, "_learned_prompt_override_recency"):
                    delattr(model, "_learned_prompt_override_recency")
                if has_m_proto and m_proto_snap is not None:
                    model._online_prototype_bank.clear()
                    model._online_prototype_bank.update(m_proto_snap)
                elif hasattr(model, "_online_prototype_bank"):
                    delattr(model, "_online_prototype_bank")
                if m_cache is not None and m_cache_snap is not None:
                    m_cache._restore_state(m_cache_snap)

                self.model_version = prev_eng_ver
                if has_lesson_cnt:
                    self._online_lesson_count = prev_lesson_cnt
                else:
                    self.__dict__.pop("_online_lesson_count", None)
                if has_eng_ov and eng_ov_snap is not None:
                    self._learned_prompt_overrides.clear()
                    self._learned_prompt_overrides.update(eng_ov_snap)
                elif hasattr(self, "_learned_prompt_overrides"):
                    delattr(self, "_learned_prompt_overrides")
                if has_eng_ov_rec and eng_ov_rec_snap is not None:
                    self._learned_prompt_override_recency.clear()
                    self._learned_prompt_override_recency.update(eng_ov_rec_snap)
                elif hasattr(self, "_learned_prompt_override_recency"):
                    delattr(self, "_learned_prompt_override_recency")
                if has_eng_ov_cnt and eng_ov_cnt_snap is not None:
                    self._override_counts.clear()
                    self._override_counts.update(eng_ov_cnt_snap)
                elif hasattr(self, "_override_counts"):
                    delattr(self, "_override_counts")
                self.calibrators.clear()
                self.calibrators.update(cal_snap)
                self.conformal_predictors.clear()
                self.conformal_predictors.update(cp_snap)
                self.regression_conformal_predictors.clear()
                self.regression_conformal_predictors.update(rcp_snap)
                self._packed_kernel_cache = prev_packed_kernel
                if has_exp_cache and exp_cache_snap is not None:
                    self._explainer_cache.clear()
                    self._explainer_cache.update(exp_cache_snap)
                elif hasattr(self, "_explainer_cache"):
                    delattr(self, "_explainer_cache")
                if self.cache is not None and eng_cache_snap is not None:
                    self.cache._restore_state(eng_cache_snap)
                raise

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
        margin_threshold = _validate_real_scalar(margin_threshold, "margin_threshold")
        learning_rate = _validate_real_scalar(learning_rate, "learning_rate")
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
                self._packed_kernel_cache = None
                if hasattr(self, "_explainer_cache"):
                    self._explainer_cache.clear()
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
            _validate_forgetting_factor(forgetting_factor)
        eff_ff = forgetting_factor
        eff_reward_scale = 1.0
        if reward is not None:
            r_val = _validate_real_scalar(reward, "reward")
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
                base_ff = _validate_forgetting_factor(
                    forgetting_factor if forgetting_factor is not None else self.forgetting_factor
                )
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

        def _checked_f32_copy(arr: Any, label: str) -> np.ndarray:
            with np.errstate(over="ignore", invalid="ignore"):
                arr_chk = _assert_real_numeric_array_like(arr, label)
                arr_in = np.asarray(arr_chk)
                if (
                    arr_in.ndim == 2
                    and arr_in.size > 0
                    and (
                        (not arr_in.flags["C_CONTIGUOUS"] and not arr_in.flags["F_CONTIGUOUS"])
                        or int(arr_in.ctypes.data % 16) != 0
                    )
                ):
                    from system1.core.model import _owned_readonly_array

                    out = _owned_readonly_array(arr_in, dtype=np.float32)
                else:
                    out = np.array(arr_in, dtype=np.float32, copy=True, order="K")
            if not bool(np.all(np.isfinite(out))):
                raise ValueError(f"{label} must be finite in float32 for compiled export")
            return out

        def _checked_unit_prototypes(arr: Any, label: str) -> np.ndarray:
            raw_p_chk = _assert_real_numeric_array_like(arr, label)
            raw_p = np.asarray(raw_p_chk, dtype=np.float64)
            if raw_p.ndim != 2 or not bool(np.all(np.isfinite(raw_p))):
                raise ValueError(f"{label} must be finite 2D array for compiled export")
            unit_p, _ = _normalize_rows_f32_safe(raw_p)
            return unit_p

        def _extract_validated_hierarchy(
            h_obj: Any, fname: str, width: int
        ) -> Tuple[
            bool,
            Tuple[int, ...],
            Optional[np.ndarray],
            Optional[np.ndarray],
            Optional[np.ndarray],
            Optional[np.ndarray],
        ]:
            is_h = bool(getattr(h_obj, "hierarchical", False))
            if not is_h:
                return False, (), None, None, None, None
            raw_cw = getattr(h_obj, "coarse_weights", None)
            raw_cb = getattr(h_obj, "coarse_biases", None)
            raw_assign = getattr(h_obj, "cluster_assignments", ())
            if raw_cw is None or raw_cb is None:
                raise ValueError(
                    f"Field head {fname!r} hierarchical weights missing coarse_weights/coarse_biases"
                )
            cw_arr = _checked_f32_copy(raw_cw, f"Head {fname!r} coarse_weights")
            cb_arr = _checked_f32_copy(raw_cb, f"Head {fname!r} coarse_biases")
            if (
                cw_arr.ndim != 2
                or cw_arr.shape[1] != self.dimension
                or cb_arr.ndim != 1
                or cb_arr.shape != (cw_arr.shape[0],)
            ):
                raise ValueError(f"Field head {fname!r} hierarchical coarse shape mismatch")
            assign_t = tuple(int(c) for c in raw_assign)
            if len(assign_t) != width or any(c < 0 or c >= cw_arr.shape[0] for c in assign_t):
                raise ValueError(f"Field head {fname!r} invalid hierarchical cluster_assignments")
            raw_fw = getattr(h_obj, "fine_weights", None)
            raw_fb = getattr(h_obj, "fine_biases", None)
            if (raw_fw is None) != (raw_fb is None):
                raise ValueError(
                    f"Field head {fname!r} fine_weights and fine_biases must both be provided or both be None"
                )
            fw_arr: Optional[np.ndarray] = None
            fb_arr: Optional[np.ndarray] = None
            if raw_fw is not None and raw_fb is not None:
                fw_arr = _checked_f32_copy(raw_fw, f"Head {fname!r} fine_weights")
                fb_arr = _checked_f32_copy(raw_fb, f"Head {fname!r} fine_biases")
                if (
                    fw_arr.ndim != 2
                    or fw_arr.shape != (width, self.dimension)
                    or fb_arr.ndim != 1
                    or fb_arr.shape != (width,)
                ):
                    raise ValueError(f"Field head {fname!r} hierarchical fine shape mismatch")
            return True, assign_t, cw_arr, cb_arr, fw_arr, fb_arr

        with self._lock, getattr(self.model, "_lock", nullcontext()):
            if isinstance(self.model, CompiledSystemOneModel):
                for name, ch in self.model.heads.items():
                    if name in self.model._field_heads:
                        live_head = self.model._field_heads[name]
                        f_def = self.schema.fields[name]
                        _validate_head_weight_bias_shapes(
                            f_def,
                            np.asarray(live_head.weights),
                            np.asarray(live_head.biases),
                            self.dimension,
                            name,
                        )
                        ch.weights = _checked_f32_copy(live_head.weights, f"Head {name!r} weights")
                        ch.biases = _checked_f32_copy(live_head.biases, f"Head {name!r} biases")
                        ch.P = (
                            None
                            if live_head.covariance_inv is None
                            else _checked_f32_copy(live_head.covariance_inv, f"Head {name!r} P")
                        )
                        ch.B = (
                            None
                            if live_head.cross_covariance is None
                            else _checked_f32_copy(live_head.cross_covariance, f"Head {name!r} B")
                        )
                        ch.exemplar_state = live_head.export_exemplar_state()
                        is_h, assign_t, cw_arr, cb_arr, fw_arr, fb_arr = _extract_validated_hierarchy(
                            live_head, name, int(ch.weights.shape[0])
                        )
                        ch.hierarchical = is_h
                        ch.cluster_assignments = assign_t
                        ch.coarse_weights = cw_arr
                        ch.coarse_biases = cb_arr
                        ch.fine_weights = fw_arr
                        ch.fine_biases = fb_arr
                    if name in self.calibrators:
                        ch.temperature = float(self.calibrators[name].temperature)
                        ch.logit_adjustments = _validated_head_logit_adjustments(
                            getattr(self.calibrators[name], "logit_adjustments", None),
                            self.schema.fields.get(name),
                            name,
                        )
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
                            else _checked_unit_prototypes(cp.prototypes, f"Head {name!r} prototypes")
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
                            val_range = _score_full_span_clamped(rcp.min_value, rcp.max_value)
                            ch.conformal_quantile = val_range if k_res > n_res else float(rcp.residuals[k_res - 1])
                        elif not rcp.is_calibrated:
                            ch.calibration_scores = ()
                            ch.conformal_quantile = 0.0
                if getattr(self, "_learned_prompt_overrides", None):
                    eng_ov_rec = getattr(self, "_learned_prompt_override_recency", {})
                    for p_k, f_map in self._learned_prompt_overrides.items():
                        if isinstance(f_map, Mapping) and f_map:
                            self.model._learned_prompt_overrides.setdefault(p_k, {}).update(f_map)
                            if isinstance(eng_ov_rec, Mapping) and p_k in eng_ov_rec:
                                self.model._learned_prompt_override_recency[p_k] = bool(eng_ov_rec[p_k])
                            elif p_k not in self.model._learned_prompt_override_recency:
                                self.model._learned_prompt_override_recency[p_k] = bool(self.recency_weighted)
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
                logit_adj = (
                    _validated_head_logit_adjustments(
                        getattr(calibrator, "logit_adjustments", None),
                        f_def,
                        name,
                    )
                    if calibrator is not None
                    else None
                )
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
                        proto_mat = _checked_unit_prototypes(cp.prototypes, f"Head {name!r} prototypes")
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
                        val_range = _score_full_span_clamped(rcp.min_value, rcp.max_value)
                        conf_q = val_range if k_res > n_res else float(rcp.residuals[k_res - 1])

                opts: Tuple[str, ...]
                if isinstance(f_def, BooleanField):
                    opts = ("False", "True")
                elif hasattr(f_def, "options"):
                    opts = tuple(f_def.options)
                else:
                    opts = ()

                w_export = _checked_f32_copy(head.weights, f"Head {name!r} weights")
                b_export = _checked_f32_copy(head.biases, f"Head {name!r} biases")
                _validate_head_weight_bias_shapes(f_def, w_export, b_export, self.dimension, name)
                raw_p = getattr(head, "covariance_inv", None)
                raw_b_cov = getattr(head, "cross_covariance", None)
                is_h, assign_t, cw_arr, cb_arr, fw_arr, fb_arr = _extract_validated_hierarchy(
                    head, name, int(w_export.shape[0])
                )
                compiled_heads[name] = CompiledHeadWeights(
                    field_name=name,
                    field_type=f_def.field_type,
                    weights=w_export,
                    biases=b_export,
                    temperature=temp,
                    logit_adjustments=logit_adj,
                    conformal_quantile=conf_q,
                    options=opts,
                    P=None if raw_p is None else _checked_f32_copy(raw_p, f"Head {name!r} P"),
                    B=None if raw_b_cov is None else _checked_f32_copy(raw_b_cov, f"Head {name!r} B"),
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
                    hierarchical=is_h,
                    cluster_assignments=assign_t,
                    coarse_weights=cw_arr,
                    coarse_biases=cb_arr,
                    fine_weights=fw_arr,
                    fine_biases=fb_arr,
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
            for src_obj in (self.model, self):
                src_ov = getattr(src_obj, "_learned_prompt_overrides", None)
                src_rec = getattr(src_obj, "_learned_prompt_override_recency", {})
                src_def_rec = bool(getattr(src_obj, "recency_weighted", self.recency_weighted))
                if isinstance(src_ov, Mapping):
                    for p_k, f_map in src_ov.items():
                        if isinstance(f_map, Mapping) and f_map:
                            compiled._learned_prompt_overrides.setdefault(p_k, {}).update(f_map)
                            if isinstance(src_rec, Mapping) and p_k in src_rec:
                                compiled._learned_prompt_override_recency[p_k] = bool(src_rec[p_k])
                            elif p_k not in compiled._learned_prompt_override_recency:
                                compiled._learned_prompt_override_recency[p_k] = src_def_rec
            return compiled

    def _packed_kernel_cache_key(self, storage_mode: str) -> Tuple[Any, ...]:
        aci_obj = getattr(self, "aci_tracker", None)
        eff_alpha = float(aci_obj.alpha_t) if aci_obj is not None else float(self.default_alpha)
        return _compute_engine_inference_signature(self, storage_mode, effective_alpha=eff_alpha)

    def compile_packed_kernel(self, *, storage_mode: str = "float32") -> PackedMultiQuestionKernel:
        """Compiles and caches a fused multi-question inference kernel for this engine."""
        model = getattr(self, "model", None)
        model_lock = getattr(model, "_lock", nullcontext()) if model is not None else nullcontext()
        with self._lock, model_lock:
            _validate_packed_engine_support(self)
            aci_tracker = getattr(self, "aci_tracker", None)
            raw_alpha = aci_tracker.alpha_t if aci_tracker is not None else self.default_alpha
            if (
                isinstance(raw_alpha, bool)
                or not isinstance(raw_alpha, (int, float))
                or not math.isfinite(float(raw_alpha))
                or not (0.0 < float(raw_alpha) < 1.0)
            ):
                raise ValueError(f"alpha must be in (0, 1), got {raw_alpha!r}")
            captured_alpha = float(raw_alpha)
            cache_key = _compute_engine_inference_signature(
                self, storage_mode, effective_alpha=captured_alpha
            )
            cached = getattr(self, "_packed_kernel_cache", None)
            if (
                cached is not None
                and cached[0] == cache_key
                and cached[1].alpha == captured_alpha
            ):
                return cached[1]
            kernel = PackedMultiQuestionKernel.from_engine(
                self,
                storage_mode=storage_mode,
                alpha=captured_alpha,
            )
            if kernel.alpha != captured_alpha:
                raise ValueError(
                    f"PackedMultiQuestionKernel alpha mismatch: expected {captured_alpha}, got {kernel.alpha}"
                )
            kernel._inference_sig = cache_key
            self._packed_kernel_cache = (cache_key, kernel)
            return kernel

    def compile_callable(self, *, storage_mode: str = "float32") -> FastDecisionCallable:
        """Returns a zero-overhead decision callable backed by a packed multi-question kernel."""
        return FastDecisionCallable(self, storage_mode=storage_mode)

    def explain(
        self,
        prompt: str,
        field_name: Optional[str] = None,
        *,
        include_counterfactual: bool = True,
        train_texts: Optional[Sequence[str]] = None,
        train_labels: Optional[Sequence[Any]] = None,
    ) -> Any:
        """Returns additive token attributions, exemplar witnesses, and counterfactual flips."""
        if not isinstance(prompt, str):
            raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}")
        from system1.explain import FastExplainer, _canonical_explain_field

        resolved_field = _canonical_explain_field(self.schema, field_name)
        if self.auto_reload and self.model_path is not None:
            self.reload_if_changed()
        with self._lock:
            model = getattr(self, "model", None)
            if _has_custom_validate_decision(self.schema) or _has_custom_validate_decision(
                getattr(model, "schema", None)
            ):
                raise ValueError(
                    "FastExplainer does not support custom DecisionSchema.validate_decision"
                )
            if getattr(self, "joint_calibrator", None) is not None:
                raise ValueError(
                    "FastExplainer does not support engines with active joint_calibrator"
                )
            if train_texts is None and train_labels is None:
                cache_key = self._packed_kernel_cache_key("float32")
                cached = self._explainer_cache.get(resolved_field)
                if cached is not None and cached[0] == cache_key:
                    explainer = cached[1]
                else:
                    explainer = FastExplainer.from_engine(self, resolved_field)
                    self._explainer_cache[resolved_field] = (cache_key, explainer)
            else:
                explainer = FastExplainer.from_engine(
                    self,
                    resolved_field,
                    train_texts=train_texts,
                    train_labels=train_labels,
                )
        return explainer.explain(prompt, include_counterfactual=include_counterfactual)

    def explain_all(
        self,
        prompt: str,
        *,
        include_counterfactual: bool = True,
    ) -> Dict[str, Any]:
        """Explains every ChoiceField and BooleanField in the schema for one prompt."""
        if not isinstance(prompt, str):
            raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}")
        return {
            name: self.explain(prompt, name, include_counterfactual=include_counterfactual)
            for name, f_def in self.schema.fields.items()
            if isinstance(f_def, (ChoiceField, BooleanField))
        }

    def close(self) -> None:
        """Releases any memory-mapped resources held by the underlying model."""
        with self._lock:
            self._packed_kernel_cache = None
            if hasattr(self, "_explainer_cache"):
                self._explainer_cache.clear()
            model = getattr(self, "model", None)
            close_fn = getattr(model, "close", None)
            if callable(close_fn):
                close_fn()

    def __enter__(self) -> SystemOneEngine:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def to_bytes(
        self,
        include_covariance: bool = False,
        *,
        compact: Optional[bool] = None,
        format_version: int = 2,
        embed_vocab: bool = False,
    ) -> bytes:
        """Serializes the engine's model and calibration state into compact .s1m binary bytes."""
        return self.to_compiled_model().to_bytes(
            include_covariance=include_covariance,
            compact=compact,
            format_version=format_version,
            embed_vocab=embed_vocab,
        )

    def save(
        self,
        path: Union[str, Path],
        include_covariance: bool = False,
        *,
        compact: Optional[bool] = None,
        format_version: int = 2,
        embed_vocab: bool = False,
    ) -> None:
        """Saves the engine's model, calibration, and online state to a portable .s1m file."""
        self.to_compiled_model().save(
            path,
            include_covariance=include_covariance,
            compact=compact,
            format_version=format_version,
            embed_vocab=embed_vocab,
        )

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
        mmap_mode: bool = True,
        **kwargs: Any,
    ) -> SystemOneEngine:
        """Loads a SystemOneEngine directly from a saved .s1m file."""
        from system1.compiler import CompiledSystemOneModel

        compiled = CompiledSystemOneModel.load(
            path,
            projector=projector,
            backend="numpy" if backend == "auto" else backend,
            mmap_mode=mmap_mode,
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

    def _decide_batch_vectorized(
        self,
        prompts: Sequence[str],
        *,
        alpha: Optional[float] = None,
        margin_threshold: Optional[float] = None,
        relative_odds_ratio: Optional[float] = None,
        confidence_floor_tau0: Optional[float] = None,
        recency_weighted: Optional[bool] = None,
        strict: Optional[bool] = None,
    ) -> List[DecisionResult]:
        """Vectorized chunked pure-NumPy batch inference path for ChoiceField/BooleanField schemas."""
        from system1.core.text import TfidfProjector

        n_total = len(prompts)
        if n_total == 0:
            return []

        if self.auto_reload and self.model_path is not None:
            self.reload_if_changed()

        with self._lock, getattr(self.model, "_lock", nullcontext()):
            recency_flag = self.recency_weighted if recency_weighted is None else bool(recency_weighted)
            version = getattr(self.model, "model_version", self.model_version)
            if version != self.model_version:
                if hasattr(self.model, "_field_heads"):
                    self._sync_from_model(self.model)
                self.model_version = version
                for fname in self.schema.fields:
                    head_obj = getattr(self.model, "heads", {}).get(fname)
                    if not getattr(head_obj, "calibration_scores", ()):
                        self.calibrators[fname] = DecisionCalibrator()
                        if fname in self.conformal_predictors:
                            cp_reset = self.conformal_predictors[fname]
                            cp_reset.is_calibrated = False
                            cp_reset.familiarity_threshold = 0.0
                            cp_reset.prototype_threshold = 0.0
                            cp_reset.familiarity_scores = np.array([], dtype=np.float64)
                            cp_reset.prototype_scores = np.array([], dtype=np.float64)
                            cp_reset.class_calibration_scores = {}
                        if fname in self.regression_conformal_predictors:
                            self.regression_conformal_predictors[fname].is_calibrated = False

            eff_alpha = (
                (self.aci_tracker.alpha_t if self.aci_tracker is not None else self.default_alpha)
                if alpha is None
                else alpha
            )
            if (
                isinstance(eff_alpha, bool)
                or not isinstance(eff_alpha, (int, float))
                or not (0.0 < eff_alpha < 1.0)
                or math.isnan(eff_alpha)
                or math.isinf(eff_alpha)
            ):
                raise ValueError(f"alpha must be in (0, 1), got {eff_alpha}")
            eff_alpha = float(eff_alpha)
            if margin_threshold is not None and (
                isinstance(margin_threshold, bool)
                or not isinstance(margin_threshold, (int, float))
                or margin_threshold < 0.0
                or math.isnan(margin_threshold)
                or math.isinf(margin_threshold)
            ):
                raise ValueError(f"margin_threshold must be non-negative, got {margin_threshold}")
            if relative_odds_ratio is not None and (
                isinstance(relative_odds_ratio, bool)
                or not isinstance(relative_odds_ratio, (int, float))
                or relative_odds_ratio <= 0.0
                or math.isnan(relative_odds_ratio)
                or math.isinf(relative_odds_ratio)
            ):
                raise ValueError(f"relative_odds_ratio must be strictly positive, got {relative_odds_ratio}")
            if confidence_floor_tau0 is not None and (
                isinstance(confidence_floor_tau0, bool)
                or not isinstance(confidence_floor_tau0, (int, float))
                or confidence_floor_tau0 < 0.0
                or math.isnan(confidence_floor_tau0)
                or math.isinf(confidence_floor_tau0)
            ):
                raise ValueError(f"confidence_floor_tau0 must be non-negative, got {confidence_floor_tau0}")

            is_strict = self.strict_mode if strict is None else bool(strict)
            eff_m_thresh = margin_threshold if margin_threshold is not None else self.margin_threshold
            eff_gamma = relative_odds_ratio if relative_odds_ratio is not None else self.relative_odds_ratio
            eff_tau0 = confidence_floor_tau0 if confidence_floor_tau0 is not None else self.confidence_floor_tau0

            proj_obj = getattr(self.model, "projector", self.projector)
            is_tfidf_proj = isinstance(proj_obj, TfidfProjector)
            zero_unit0 = _model_zero_unit0(self.model, proj_obj)
            is_hybrid_proj = isinstance(proj_obj, HybridProjector)
            schema_name = self.schema.schema_name
            schema_digest = self.schema.schema_digest()
            custom_validate = _has_custom_validate_decision(self.schema)

            online_bank_map = getattr(self.model, "_online_prototype_bank", {})
            model_field_heads = getattr(self.model, "_field_heads", {})
            needs_raw_norm = (
                any(
                    float(getattr(cp, "familiarity_threshold", 0.0)) > 0.0
                    for cp in self.conformal_predictors.values()
                )
                or any(
                    float(getattr(ch, "familiarity_threshold", 0.0)) > 0.0
                    for ch in getattr(self.model, "heads", {}).values()
                )
                or any(
                    float(getattr(ch, "familiarity_threshold", 0.0)) > 0.0
                    for ch in model_field_heads.values()
                )
            )
            sel_cals = getattr(self, "selective_calibrators", {}) or {}
            aci = getattr(self, "aci_tracker", None)
            aci_has_floor = aci is not None and getattr(aci, "margin_floor", None) is not None
            jcal = getattr(self, "joint_calibrator", None)

            field_specs: List[Dict[str, Any]] = []
            for fname, f_def in self.schema.fields.items():
                head, meta_head = _resolve_authoritative_head(self.model, fname)
                calibrator = self.calibrators[fname]
                conformal = self.conformal_predictors.get(fname)
                is_bool = isinstance(f_def, BooleanField)
                options = ("False", "True") if is_bool else tuple(f_def.options)
                k_opts = len(options)

                raw_w_arr = np.asarray(head.weights)
                raw_b_arr = np.asarray(head.biases)
                _validate_head_weight_bias_shapes(f_def, raw_w_arr, raw_b_arr, self.dimension, fname)
                raw_w_arr = _coerce_inference_array_f32_or_f64(
                    raw_w_arr, f"Field head {fname!r} weights", active_first_only=is_bool
                )
                raw_b_arr = _coerce_inference_array_f32_or_f64(
                    raw_b_arr, f"Field head {fname!r} biases", active_first_only=is_bool
                )
                _validate_2d_layout_cloneable(
                    raw_w_arr,
                    np.dtype(raw_w_arr.dtype if raw_w_arr.dtype in (np.float32, np.float64) else np.float32),
                )
                raw_w_val = raw_w_arr[:1] if is_bool else raw_w_arr
                raw_b_val = raw_b_arr[:1] if is_bool else raw_b_arr
                if not (np.all(np.isfinite(raw_w_val)) and np.all(np.isfinite(raw_b_val))):
                    raise ValueError(f"Field head {fname!r} contains non-finite weights or biases")
                with np.errstate(over="ignore", invalid="ignore"):
                    if raw_w_arr.dtype not in (np.float32, np.float64):
                        raw_w_arr = raw_w_arr.astype(np.float32)
                    if raw_b_arr.dtype not in (np.float32, np.float64):
                        raw_b_arr = raw_b_arr.astype(np.float32)
                    w_mat = np.asarray(raw_w_arr[:1] if is_bool else raw_w_arr, dtype=np.float32)
                    b_vec = np.asarray(raw_b_arr[:1] if is_bool else raw_b_arr, dtype=np.float32)

                cluster_assign_raw = getattr(head, "cluster_assignments", ()) or ()
                is_hier = bool(not is_bool and getattr(head, "hierarchical", False))
                raw_coarse_w: Optional[np.ndarray] = None
                raw_coarse_b: Optional[np.ndarray] = None
                raw_fine_w: Optional[np.ndarray] = None
                raw_fine_b: Optional[np.ndarray] = None
                coarse_w: Optional[np.ndarray] = None
                coarse_b: Optional[np.ndarray] = None
                fine_w: Optional[np.ndarray] = None
                fine_b: Optional[np.ndarray] = None
                cluster_assign: Optional[np.ndarray] = None
                if is_hier:
                    if getattr(head, "coarse_weights", None) is None or getattr(head, "coarse_biases", None) is None:
                        raise ValueError(
                            f"Field head {fname!r} hierarchical weights missing coarse_weights/coarse_biases"
                        )
                    raw_coarse_w = _coerce_inference_array_f32_or_f64(
                        head.coarse_weights, f"Field head {fname!r} hierarchical coarse weights"
                    )
                    raw_coarse_b = _coerce_inference_array_f32_or_f64(
                        head.coarse_biases, f"Field head {fname!r} hierarchical coarse biases"
                    )
                    if raw_coarse_b.ndim != 1:
                        raise ValueError(f"Field head {fname!r} coarse_biases must be 1D, got shape {raw_coarse_b.shape}")
                    if not (np.all(np.isfinite(raw_coarse_w)) and np.all(np.isfinite(raw_coarse_b))):
                        raise ValueError(f"Field head {fname!r} contains non-finite hierarchical coarse weights")
                    cluster_assign = np.asarray(cluster_assign_raw, dtype=np.int64)
                    if (
                        raw_coarse_w.ndim != 2
                        or raw_coarse_w.shape[0] < 1
                        or raw_coarse_w.shape[1] != self.dimension
                        or raw_coarse_b.shape != (raw_coarse_w.shape[0],)
                        or cluster_assign.ndim != 1
                        or cluster_assign.size != raw_w_arr.shape[0]
                        or (
                            cluster_assign.size > 0
                            and np.any((cluster_assign < 0) | (cluster_assign >= raw_coarse_w.shape[0]))
                        )
                    ):
                        raise ValueError(f"Field head {fname!r} has invalid hierarchical cluster configuration")
                    _validate_2d_layout_cloneable(raw_coarse_w, np.dtype(raw_coarse_w.dtype))
                    has_fw = getattr(head, "fine_weights", None) is not None
                    has_fb = getattr(head, "fine_biases", None) is not None
                    if has_fw != has_fb:
                        raise ValueError(
                            f"Field head {fname!r} fine_weights and fine_biases must both be provided or both be None"
                        )
                    if has_fw and has_fb:
                        raw_fine_w = _coerce_inference_array_f32_or_f64(
                            head.fine_weights, f"Field head {fname!r} hierarchical fine weights"
                        )
                        raw_fine_b = _coerce_inference_array_f32_or_f64(
                            head.fine_biases, f"Field head {fname!r} hierarchical fine biases"
                        )
                        if raw_fine_b.ndim != 1:
                            raise ValueError(f"Field head {fname!r} fine_biases must be 1D, got shape {raw_fine_b.shape}")
                        if (
                            raw_fine_w.ndim != 2
                            or raw_fine_w.shape != (raw_w_arr.shape[0], self.dimension)
                            or raw_fine_b.shape != (raw_w_arr.shape[0],)
                        ):
                            raise ValueError(f"Field head {fname!r} hierarchical fine tensor shape mismatch")
                        _validate_2d_layout_cloneable(raw_fine_w, np.dtype(raw_fine_w.dtype))
                        if not (np.all(np.isfinite(raw_fine_w)) and np.all(np.isfinite(raw_fine_b))):
                            raise ValueError(f"Field head {fname!r} contains non-finite hierarchical fine weights")
                    coarse_w = raw_coarse_w
                    coarse_b = raw_coarse_b
                    fine_w = raw_fine_w
                    fine_b = raw_fine_b

                fam_t = max(
                    float(getattr(head, "familiarity_threshold", 0.0)),
                    float(getattr(meta_head, "familiarity_threshold", 0.0)) if meta_head is not None else 0.0,
                    float(getattr(conformal, "familiarity_threshold", 0.0)) if conformal is not None else 0.0,
                )
                proto_t = max(
                    float(getattr(head, "prototype_threshold", 0.0)),
                    float(getattr(meta_head, "prototype_threshold", 0.0)) if meta_head is not None else 0.0,
                    float(getattr(conformal, "prototype_threshold", 0.0)) if conformal is not None else 0.0,
                )

                proto_chunks: List[np.ndarray] = []
                head_protos = getattr(head, "prototypes", None)
                if head_protos is None and meta_head is not None:
                    head_protos = getattr(meta_head, "prototypes", None)
                with np.errstate(over="ignore", invalid="ignore"):
                    if head_protos is not None and np.ndim(head_protos) == 2 and head_protos.shape[1] == self.dimension:
                        hp_unit, hvalid = _normalize_rows_f32_safe(head_protos)
                        if np.any(hvalid):
                            proto_chunks.append(hp_unit[hvalid])
                    if not proto_chunks and conformal is not None and getattr(conformal, "prototypes", None) is not None:
                        if np.ndim(conformal.prototypes) == 2 and conformal.prototypes.shape[1] == self.dimension:
                            cp_unit, cvalid = _normalize_rows_f32_safe(conformal.prototypes)
                            if np.any(cvalid):
                                proto_chunks.append(cp_unit[cvalid])
                    if head_protos is not None and online_bank_map:
                        bank = online_bank_map.get(fname)
                        if bank:
                            bank_arr = [
                                np.asarray(s, dtype=np.float32).flatten()
                                for s in bank
                                if np.asarray(s).size == self.dimension
                            ]
                            if bank_arr:
                                proto_chunks.append(np.stack(bank_arr, axis=0))
                    if head_protos is not None and model_field_heads:
                        mfh = model_field_heads.get(fname)
                        shots_p = getattr(mfh, "_online_shots_p", None) if mfh is not None else None
                        if shots_p:
                            shot_vecs: List[np.ndarray] = []
                            for grp in shots_p:
                                for s in grp:
                                    s_raw = np.asarray(s).ravel()
                                    if s_raw.shape[0] == self.dimension:
                                        s_unit, _, s_valid = _normalize_vec_f32_safe(s_raw)
                                        if s_valid:
                                            shot_vecs.append(s_unit)
                            if shot_vecs:
                                proto_chunks.append(np.stack(shot_vecs, axis=0))
                protos_unit = np.concatenate(proto_chunks, axis=0) if proto_chunks else None

                cp_fam_t = float(getattr(conformal, "familiarity_threshold", 0.0)) if conformal is not None else 0.0
                cp_proto_t = float(getattr(conformal, "prototype_threshold", 0.0)) if conformal is not None else 0.0
                if conformal is not None:
                    eff_m = (
                        float(eff_m_thresh)
                        if (self.enable_margin_gating and not is_strict and eff_m_thresh is not None)
                        else (
                            (
                                conformal.margin_threshold
                                if conformal.margin_threshold is not None
                                else conformal.calibrated_margin_threshold
                            )
                            if (self.enable_margin_gating and not is_strict)
                            else 0.0
                        )
                    )
                    eff_g = float(eff_gamma) if (not is_strict and eff_gamma is not None) else conformal.relative_odds_ratio
                    eff_t0 = float(eff_tau0) if (not is_strict and eff_tau0 is not None) else conformal.confidence_floor_tau0
                    conf_floor = float((1.0 / max(1, k_opts)) + eff_t0)
                    n_cal = len(conformal.calibration_scores)
                    if not conformal.is_calibrated or n_cal == 0:
                        k_q = n_cal + 1
                        q_hat = 1.0 if is_strict else (1.0 - eff_alpha)
                    else:
                        k_q = int(math.ceil((n_cal + 1) * (1.0 - eff_alpha)))
                        q_hat = 1.0 if k_q > n_cal else float(conformal.calibration_scores[k_q - 1])
                    if conformal.mondrian and conformal.class_calibration_scores and n_cal > 0:
                        q_per_cls = np.array(
                            [conformal.class_quantile(opt, eff_alpha) for opt in options],
                            dtype=np.float64,
                        )
                    else:
                        q_per_cls = None
                else:
                    eff_m = 0.0
                    eff_g = 0.0
                    conf_floor = 0.0
                    n_cal = 0
                    k_q = 1
                    q_hat = 1.0
                    q_per_cls = None

                field_specs.append(
                    {
                        "name": fname,
                        "f_def": f_def,
                        "is_bool": is_bool,
                        "bool_threshold": float(getattr(f_def, "threshold", 0.5)),
                        "options": options,
                        "k_opts": k_opts,
                        "escalate": bool(getattr(f_def, "escalate_on_ambiguity", True)),
                        "w": w_mat,
                        "b": b_vec,
                        "raw_w": raw_w_arr,
                        "raw_b": raw_b_arr,
                        "is_hier": is_hier,
                        "coarse_w": coarse_w,
                        "coarse_b": coarse_b,
                        "fine_w": fine_w,
                        "fine_b": fine_b,
                        "raw_coarse_w": raw_coarse_w,
                        "raw_coarse_b": raw_coarse_b,
                        "raw_fine_w": raw_fine_w,
                        "raw_fine_b": raw_fine_b,
                        "cluster_assign": cluster_assign,
                        "calibrator": calibrator,
                        "conformal": conformal,
                        "fam_t": fam_t,
                        "proto_t": proto_t,
                        "cp_fam_t": cp_fam_t,
                        "cp_proto_t": cp_proto_t,
                        "protos_unit": protos_unit,
                        "eff_m": float(eff_m),
                        "eff_g": float(eff_g),
                        "conf_floor": float(conf_floor),
                        "n_cal": n_cal,
                        "k_q": k_q,
                        "q_hat": float(q_hat),
                        "q_per_cls": q_per_cls,
                        "sel_cal": sel_cals.get(fname),
                    }
                )

            results: List[DecisionResult] = []
            chunk_size = 512
            for start in range(0, n_total, chunk_size):
                end = min(n_total, start + chunk_size)
                chunk_prompts = prompts[start:end]
                m_docs = len(chunk_prompts)
                t_chunk_start = time.perf_counter()

                raw_norms_arr: Optional[np.ndarray] = None
                enc_rows: List[np.ndarray] = []
                rn_list: Optional[List[float]] = [] if needs_raw_norm else None
                has_model_encode_with_norm = hasattr(self.model, "_encode_with_norm")
                any_rn_observed = False
                for p in chunk_prompts:
                    if has_model_encode_with_norm:
                        emb_p, rn_p = self.model._encode_with_norm(p, recency_weighted=recency_flag)
                    else:
                        emb_p = self.model.encode(p, recency_weighted=recency_flag)
                        rn_p = None
                    with np.errstate(over="ignore", invalid="ignore"):
                        enc_rows.append(np.asarray(emb_p, dtype=np.float32).ravel())
                    if rn_list is not None:
                        if rn_p is None:
                            rn_p = _extract_prompt_raw_norm(
                                None, p, proj_obj, recency_flag=recency_flag
                            )
                        if rn_p is not None:
                            any_rn_observed = True
                        rn_list.append(float(rn_p) if rn_p is not None else float("nan"))
                X = np.stack(enc_rows, axis=0).astype(np.float32, copy=False)
                if rn_list is not None and (
                    callable(getattr(proj_obj, "raw_norm", None))
                    or callable(getattr(proj_obj, "project_with_norm", None))
                    or any_rn_observed
                ):
                    raw_norms_arr = np.asarray(rn_list, dtype=np.float64)
                nonfin_rows = ~np.all(np.isfinite(X), axis=1)
                x_inf_chunk = np.max(
                    np.abs(np.where(np.isfinite(X), X, 0.0)), axis=1
                ).astype(np.float64)
                if is_tfidf_proj:
                    zero_emb_mask = (~nonfin_rows) & (~np.any(X, axis=1))
                else:
                    zero_emb_mask = np.zeros(m_docs, dtype=bool)
                row_any_nonfin = nonfin_rows.copy()

                row_values: List[Dict[str, Any]] = [{} for _ in range(m_docs)]
                row_confs: List[Dict[str, float]] = [{} for _ in range(m_docs)]
                row_probs: List[Dict[str, Dict[str, float]]] = [{} for _ in range(m_docs)]
                row_csets: List[Dict[str, List[str]]] = [{} for _ in range(m_docs)]
                row_margins: List[Dict[str, float]] = [{} for _ in range(m_docs)]
                row_mthresh: List[Dict[str, float]] = [{} for _ in range(m_docs)]
                row_mgact: List[Dict[str, bool]] = [{} for _ in range(m_docs)]
                row_odds: List[Dict[str, float]] = [{} for _ in range(m_docs)]
                row_ror: List[Dict[str, float]] = [{} for _ in range(m_docs)]
                row_cfloor: List[Dict[str, float]] = [{} for _ in range(m_docs)]
                row_proto_sims: List[Dict[str, float]] = [{} for _ in range(m_docs)]
                row_reasons: List[Dict[str, str]] = [{} for _ in range(m_docs)]
                row_amb_fields: List[List[str]] = [[] for _ in range(m_docs)]
                row_esc_fields: List[List[str]] = [[] for _ in range(m_docs)]
                row_is_amb = np.zeros(m_docs, dtype=bool)
                row_fam_trig = np.zeros(m_docs, dtype=bool)
                row_proto_trig = np.zeros(m_docs, dtype=bool)
                probs_by_field_arr: Dict[str, np.ndarray] = {}

                for spec in field_specs:
                    fname = spec["name"]
                    options = spec["options"]
                    k_opts = spec["k_opts"]
                    calibrator = spec["calibrator"]
                    conformal = spec["conformal"]

                    with np.errstate(over="ignore", invalid="ignore"):
                        if spec["is_hier"]:
                            has_f64_h = any(
                                arr is not None and arr.dtype == np.float64
                                for arr in (
                                    spec["raw_w"],
                                    spec["raw_b"],
                                    spec["raw_coarse_w"],
                                    spec["raw_coarse_b"],
                                    spec["raw_fine_w"],
                                    spec["raw_fine_b"],
                                )
                            )
                            logits = np.full(
                                (m_docs, k_opts),
                                np.nan,
                                dtype=np.float64 if has_f64_h else np.float32,
                            )
                            for r_i in range(m_docs):
                                if nonfin_rows[r_i]:
                                    continue
                                x_ex = X[r_i : r_i + 1]
                                base_ex = x_ex @ spec["raw_w"].T + spec["raw_b"]
                                hier_ex = _compute_hierarchical_logits_np(
                                    x_ex,
                                    base_ex,
                                    spec["raw_coarse_w"],
                                    spec["raw_coarse_b"],
                                    spec["cluster_assign"],
                                    spec["raw_fine_w"],
                                    spec["raw_fine_b"],
                                )
                                logits[r_i] = hier_ex[0]
                        else:
                            logits = X @ spec["w"].T + spec["b"]

                        if not spec["is_hier"]:
                            raw_w_s = np.asarray(spec["raw_w"], dtype=np.float64)
                            raw_b_s = np.asarray(spec["raw_b"], dtype=np.float64)
                            w_l1 = (
                                float(np.sum(np.abs(raw_w_s[0])))
                                if spec["is_bool"]
                                else (float(np.max(np.sum(np.abs(raw_w_s), axis=1))) if raw_w_s.size else 0.0)
                            )
                            b_abs = (
                                abs(float(raw_b_s.ravel()[0]))
                                if (spec["is_bool"] and raw_b_s.size)
                                else (float(np.max(np.abs(raw_b_s))) if raw_b_s.size else 0.0)
                            )
                            l1_mag = x_inf_chunk * w_l1 + b_abs
                            pot_ovf_rows = (~np.isfinite(l1_mag)) | (l1_mag >= 1e37)
                            dz_bound = (
                                4.0
                                * (float(self.dimension) + 4.0)
                                * float(np.finfo(np.float32).eps)
                                * l1_mag
                            )
                            temp_est = max(1e-12, abs(float(getattr(calibrator, "temperature", 1.0))))
                            dp_bound = np.minimum(1.0, (dz_bound / 0.25) / temp_est + 1e-6)
                        else:
                            pot_ovf_rows = np.zeros(m_docs, dtype=bool)
                            dz_bound = np.empty(0, dtype=np.float64)
                            dp_bound = np.empty(0, dtype=np.float64)

                        if spec["is_bool"]:
                            z_col = logits[:, 0].astype(np.float64) / 0.25
                            raw_logits_nonfin = (~np.isfinite(logits[:, 0])) | (~np.isfinite(z_col))
                            two_logits = np.column_stack([-z_col / 2.0, z_col / 2.0])
                            probs = calibrator.calibrate_logits(two_logits)
                            if probs.ndim == 1:
                                probs = probs.reshape(1, -1)
                            if not spec["is_hier"]:
                                p_nonfin = ~np.all(np.isfinite(probs), axis=1)
                                cal_adj_b = getattr(calibrator, "logit_adjustments", None)
                                cal_nonfin = np.zeros(m_docs, dtype=bool)
                                z_cal_col = z_col
                                adj_ulp_b = 0.0
                                with np.errstate(over="ignore", invalid="ignore"):
                                    if cal_adj_b is not None:
                                        adj_b_arr = np.asarray(cal_adj_b, dtype=np.float64).reshape(-1)
                                        if adj_b_arr.size == 2:
                                            two_cal = two_logits + adj_b_arr.reshape(1, 2)
                                            z_cal_col = two_cal[:, 1] - two_cal[:, 0]
                                            cal_nonfin = ~np.all(np.isfinite(two_cal), axis=1)
                                            if bool(np.all(np.isfinite(adj_b_arr))):
                                                adj_ulp_b = 8.0 * float(np.finfo(np.float64).eps) * float(np.max(np.abs(adj_b_arr)))
                                    dp_bound_eff = np.minimum(1.0, dp_bound + adj_ulp_b / temp_est)
                                    row_finite = (~nonfin_rows) & (~raw_logits_nonfin) & (~p_nonfin) & (~cal_nonfin)
                                    near_rows = (
                                        (np.abs(probs[:, 1] - spec["bool_threshold"]) <= dp_bound_eff)
                                        | (np.abs(probs[:, 1] - 0.5) <= dp_bound_eff)
                                        | (np.abs(z_cal_col) <= 2.0 * (dz_bound / 0.25) + adj_ulp_b + 1e-9)
                                        | (~np.isfinite(dp_bound_eff))
                                        | raw_logits_nonfin
                                        | p_nonfin
                                        | cal_nonfin
                                        | pot_ovf_rows
                                    )
                                    if 0.0 < spec["bool_threshold"] < 1.0:
                                        thr_b = float(spec["bool_threshold"])
                                        z_thr = temp_est * math.log(thr_b / (1.0 - thr_b))
                                        near_rows |= np.abs(z_cal_col - z_thr) <= 2.0 * (dz_bound / 0.25) + adj_ulp_b + 1e-9
                                    if conformal is not None:
                                        near_rows |= (
                                            np.any(
                                                np.abs((1.0 - probs) - spec["q_hat"]) <= dp_bound_eff[:, None] + 1e-6,
                                                axis=1,
                                            )
                                            | (
                                                np.abs(np.abs(2.0 * probs[:, 1] - 1.0) - spec["eff_m"])
                                                <= 2.0 * dp_bound_eff + 1e-6
                                            )
                                            | (
                                                np.abs(np.maximum(probs[:, 1], probs[:, 0]) - spec["conf_floor"])
                                                <= dp_bound_eff + 1e-6
                                            )
                                            | (
                                                np.abs(
                                                    np.maximum(probs[:, 1], probs[:, 0])
                                                    - spec["eff_g"] * np.maximum(1e-6, np.minimum(probs[:, 1], probs[:, 0]))
                                                )
                                                <= (1.0 + spec["eff_g"]) * dp_bound_eff + 1e-6
                                            )
                                        )
                                        if spec["q_per_cls"] is not None:
                                            near_rows |= np.any(
                                                np.abs((1.0 - probs) - spec["q_per_cls"][None, :])
                                                <= dp_bound_eff[:, None] + 1e-6,
                                                axis=1,
                                            )
                                if spec["sel_cal"] is not None or aci_has_floor:
                                    near_rows |= row_finite
                                if spec["raw_w"].dtype == np.float64 or spec["raw_b"].dtype == np.float64:
                                    near_rows |= ~nonfin_rows
                                near_rows &= ~nonfin_rows
                                if np.any(near_rows):
                                    probs = np.array(probs, dtype=np.float64, copy=True)
                                    raw_w_b = spec["raw_w"]
                                    raw_b_b = spec["raw_b"]
                                    with np.errstate(over="ignore", invalid="ignore"):
                                        for r_i in np.flatnonzero(near_rows).tolist():
                                            x_ex = X[r_i : r_i + 1]
                                            lin_ex = x_ex @ raw_w_b.T
                                            raw_sum_ex = lin_ex[:, :1] + raw_b_b[:1]
                                            z_ex = float(raw_sum_ex[0, 0]) / 0.25
                                            raw_logits_nonfin[r_i] = not (
                                                math.isfinite(float(lin_ex.ravel()[0]))
                                                and math.isfinite(float(raw_sum_ex.ravel()[0]))
                                                and math.isfinite(z_ex)
                                            )
                                            if raw_logits_nonfin[r_i]:
                                                probs[r_i] = np.nan
                                            else:
                                                probs[r_i] = calibrator.calibrate_logits(
                                                    np.array([-0.5 * z_ex, 0.5 * z_ex], dtype=np.float64)
                                                ).reshape(-1)
                            p_true = probs[:, 1]
                            b_vals = p_true >= spec["bool_threshold"]
                            confs = np.where(b_vals, p_true, 1.0 - p_true)
                            for i in range(m_docs):
                                pt = float(p_true[i])
                                row_values[i][fname] = bool(b_vals[i])
                                row_confs[i][fname] = float(confs[i])
                                row_probs[i][fname] = {"False": float(probs[i, 0]), "True": pt}
                        else:
                            z_mat = logits.astype(np.float64) / 0.25
                            raw_logits_nonfin = (~np.all(np.isfinite(logits), axis=1)) | (~np.all(np.isfinite(z_mat), axis=1))
                            probs = calibrator.calibrate_logits(z_mat)
                            if probs.ndim == 1:
                                probs = probs.reshape(1, -1)
                            if not spec["is_hier"]:
                                p_nonfin = ~np.all(np.isfinite(probs), axis=1)
                                cal_adj_c = getattr(calibrator, "logit_adjustments", None)
                                cal_nonfin = np.zeros(m_docs, dtype=bool)
                                z_cal_mat = z_mat
                                adj_ulp_c = 0.0
                                with np.errstate(over="ignore", invalid="ignore"):
                                    if cal_adj_c is not None:
                                        adj_c_arr = np.asarray(cal_adj_c, dtype=np.float64).reshape(-1)
                                        if adj_c_arr.size == k_opts:
                                            z_cal_mat = z_mat + adj_c_arr.reshape(1, k_opts)
                                            cal_nonfin = ~np.all(np.isfinite(z_cal_mat), axis=1)
                                            if bool(np.all(np.isfinite(adj_c_arr))):
                                                adj_ulp_c = 8.0 * float(np.finfo(np.float64).eps) * float(np.max(np.abs(adj_c_arr)))
                                    dp_bound_eff = np.minimum(1.0, dp_bound + adj_ulp_c / temp_est)
                                    row_finite = (~nonfin_rows) & (~raw_logits_nonfin) & (~p_nonfin) & (~cal_nonfin)
                                    z_s = -np.sort(-logits.astype(np.float64), axis=1)
                                    z_cal_s = -np.sort(-z_cal_mat, axis=1)
                                    p_s = -np.sort(-probs, axis=1)
                                    near_rows = (
                                        (
                                            (
                                                np.any((z_s[:, :-1] - z_s[:, 1:]) <= 2.0 * dz_bound[:, None] + 1e-9, axis=1)
                                                | np.any(
                                                    (z_cal_s[:, :-1] - z_cal_s[:, 1:])
                                                    <= 2.0 * (dz_bound[:, None] / 0.25) + adj_ulp_c + 1e-9,
                                                    axis=1,
                                                )
                                            )
                                            if k_opts >= 2
                                            else np.zeros(m_docs, dtype=bool)
                                        )
                                        | (~np.isfinite(dp_bound_eff))
                                        | raw_logits_nonfin
                                        | p_nonfin
                                        | cal_nonfin
                                        | pot_ovf_rows
                                    )
                                    if conformal is not None:
                                        near_rows |= np.any(
                                            np.abs((1.0 - probs) - spec["q_hat"]) <= dp_bound_eff[:, None] + 1e-6,
                                            axis=1,
                                        )
                                        near_rows |= np.any(
                                            np.abs(np.cumsum(np.maximum(p_s, 0.0), axis=1) - spec["q_hat"])
                                            <= (float(k_opts) + 1.0) * dp_bound_eff[:, None] + 1e-6,
                                            axis=1,
                                        )
                                        if spec["q_per_cls"] is not None:
                                            near_rows |= np.any(
                                                np.abs((1.0 - probs) - spec["q_per_cls"][None, :])
                                                <= dp_bound_eff[:, None] + 1e-6,
                                                axis=1,
                                            )
                                        if k_opts >= 2:
                                            top_chk = p_s[:, 0]
                                            run_chk = p_s[:, 1]
                                            near_rows |= (
                                                (np.abs((top_chk - run_chk) - spec["eff_m"]) <= 2.0 * dp_bound_eff + 1e-6)
                                                | (np.abs(top_chk - spec["conf_floor"]) <= dp_bound_eff + 1e-6)
                                                | (
                                                    np.abs(top_chk - spec["eff_g"] * np.maximum(1e-6, run_chk))
                                                    <= (1.0 + spec["eff_g"]) * dp_bound_eff + 1e-6
                                                )
                                                | (np.abs(run_chk - 1e-6) <= dp_bound_eff + 1e-6)
                                            )
                                        else:
                                            near_rows |= (
                                                (np.abs(p_s[:, 0] - spec["conf_floor"]) <= dp_bound_eff + 1e-6)
                                                | (np.abs(1.0 - spec["eff_m"]) <= 2.0 * dp_bound_eff + 1e-6)
                                            )
                                if spec["sel_cal"] is not None or aci_has_floor:
                                    near_rows |= row_finite
                                if spec["raw_w"].dtype == np.float64 or spec["raw_b"].dtype == np.float64:
                                    near_rows |= ~nonfin_rows
                                near_rows &= ~nonfin_rows
                                if np.any(near_rows):
                                    probs = np.array(probs, dtype=np.float64, copy=True)
                                    raw_w_c = spec["raw_w"]
                                    raw_b_c = spec["raw_b"]
                                    for r_i in np.flatnonzero(near_rows).tolist():
                                        x_ex = X[r_i : r_i + 1]
                                        lin_ex = x_ex @ raw_w_c.T
                                        raw_sum_ex = lin_ex + raw_b_c
                                        z_ex = raw_sum_ex.astype(np.float64).ravel() / 0.25
                                        raw_logits_nonfin[r_i] = not (
                                            bool(np.all(np.isfinite(lin_ex)))
                                            and bool(np.all(np.isfinite(raw_sum_ex)))
                                            and bool(np.all(np.isfinite(z_ex)))
                                        )
                                        if raw_logits_nonfin[r_i]:
                                            probs[r_i] = np.nan
                                        else:
                                            probs[r_i] = calibrator.calibrate_logits(z_ex).reshape(-1)
                            best_idx = np.argmax(probs, axis=1)
                            confs = probs[np.arange(m_docs), best_idx]
                            for i in range(m_docs):
                                bi = int(best_idx[i])
                                prow = probs[i]
                                row_values[i][fname] = options[bi]
                                row_confs[i][fname] = float(confs[i])
                                row_probs[i][fname] = {opt: float(prow[j]) for j, opt in enumerate(options)}
                    field_nonfin = nonfin_rows | raw_logits_nonfin | (~np.all(np.isfinite(probs), axis=1))
                    if np.any(field_nonfin):
                        probs = np.array(probs, dtype=np.float64, copy=True)
                        probs[field_nonfin] = np.nan
                    probs_by_field_arr[fname] = probs
                    row_any_nonfin |= field_nonfin

                    protos_unit = spec["protos_unit"]
                    if protos_unit is not None:
                        with np.errstate(over="ignore", invalid="ignore"):
                            p_sims = np.max(X @ protos_unit.T, axis=1).astype(np.float64)
                        if np.any(zero_emb_mask):
                            p_sims = np.where(zero_emb_mask, 0.0, p_sims)
                        for i in range(m_docs):
                            row_proto_sims[i][fname] = float(p_sims[i])
                    else:
                        p_sims = None

                    if conformal is None:
                        for i in range(m_docs):
                            if field_nonfin[i]:
                                row_values[i][fname] = False if spec["is_bool"] else options[0]
                                row_confs[i][fname] = float("nan")
                                row_probs[i][fname] = (
                                    {"False": float("nan"), "True": float("nan")}
                                    if spec["is_bool"]
                                    else {opt: float("nan") for opt in options}
                                )
                                row_csets[i][fname] = []
                                row_reasons[i][fname] = "non_finite_input"
                                row_amb_fields[i].append(fname)
                                if spec["escalate"]:
                                    row_esc_fields[i].append(fname)
                                row_is_amb[i] = True
                            else:
                                if spec["is_bool"]:
                                    row_csets[i][fname] = ["True" if row_values[i][fname] else "False"]
                                else:
                                    row_csets[i][fname] = [str(row_values[i][fname])]
                                row_margins[i][fname] = 1.0
                        continue

                    cp_fam_t = spec["cp_fam_t"]
                    cp_proto_t = spec["cp_proto_t"]
                    low_fam = (
                        ((raw_norms_arr < cp_fam_t - 1e-12) | ~np.isfinite(raw_norms_arr))
                        if (raw_norms_arr is not None and cp_fam_t > 0.0)
                        else np.zeros(m_docs, dtype=bool)
                    )
                    low_proto = (
                        ((p_sims < cp_proto_t - 1e-6) | ~np.isfinite(p_sims))
                        if (p_sims is not None and cp_proto_t > 0.0)
                        else np.zeros(m_docs, dtype=bool)
                    )
                    oos_mask = low_fam | low_proto

                    sorted_idx = np.argsort(-probs, axis=1, kind="mergesort")
                    top_p = probs[np.arange(m_docs), sorted_idx[:, 0]] if k_opts > 0 else np.zeros(m_docs)
                    if k_opts > 1:
                        run_p = probs[np.arange(m_docs), sorted_idx[:, 1]]
                        marg_arr = top_p - run_p
                    else:
                        run_p = np.zeros(m_docs, dtype=np.float64)
                        marg_arr = np.ones(m_docs, dtype=np.float64)
                    odds_arr = top_p / np.maximum(1e-6, run_p)

                    n_cal = spec["n_cal"]
                    k_q = spec["k_q"]
                    q_hat = spec["q_hat"]
                    q_per_cls = spec["q_per_cls"]
                    eff_m = spec["eff_m"]
                    eff_g = spec["eff_g"]
                    conf_floor = spec["conf_floor"]

                    if not conformal.is_calibrated or n_cal == 0:
                        if is_strict:
                            cset_rows = [[] if oos_mask[i] else list(options) for i in range(m_docs)]
                        else:
                            total_mass = np.sum(probs, axis=1)
                            sorted_probs = np.take_along_axis(probs, sorted_idx, axis=1)
                            cset_rows = []
                            for i in range(m_docs):
                                if oos_mask[i] or total_mass[i] < q_hat - 1e-7 or total_mass[i] < 0.5:
                                    cset_rows.append([])
                                else:
                                    s_row: List[str] = []
                                    cmass = 0.0
                                    for j_pos in range(k_opts):
                                        idx_o = int(sorted_idx[i, j_pos])
                                        s_row.append(options[idx_o])
                                        cmass += float(sorted_probs[i, j_pos])
                                        if cmass >= q_hat - 1e-7:
                                            break
                                    cset_rows.append(s_row)
                    else:
                        total_mass = np.sum(probs, axis=1)
                        valid_p = (
                            np.all(np.isfinite(probs), axis=1)
                            & np.all(probs >= 0.0, axis=1)
                            & (np.abs(total_mass - 1.0) <= (1e-8 + 1e-5))
                        )
                        if conformal.score_method == "lac":
                            if k_q > n_cal:
                                cset_rows = [[] if oos_mask[i] else list(options) for i in range(m_docs)]
                            else:
                                sorted_probs = np.take_along_axis(probs, sorted_idx, axis=1)
                                if q_per_cls is not None:
                                    thr_sorted = q_per_cls[sorted_idx]
                                    mask_sorted = (1.0 - sorted_probs) <= (thr_sorted + 1e-7)
                                else:
                                    mask_sorted = (1.0 - sorted_probs) <= (q_hat + 1e-7)
                                mask_sorted &= valid_p[:, None]
                                mask_sorted[oos_mask] = False
                                cset_rows = [
                                    [options[int(sorted_idx[i, j])] for j in np.flatnonzero(mask_sorted[i])]
                                    if np.any(mask_sorted[i])
                                    else []
                                    for i in range(m_docs)
                                ]
                        elif is_strict:
                            if k_q > n_cal:
                                cset_rows = [[] if oos_mask[i] else list(options) for i in range(m_docs)]
                            else:
                                sorted_probs = np.take_along_axis(probs, sorted_idx, axis=1)
                                cum_mass = np.cumsum(sorted_probs, axis=1)
                                if q_per_cls is not None:
                                    thr_sorted = q_per_cls[sorted_idx]
                                    mask_sorted = cum_mass <= (thr_sorted + 1e-7)
                                else:
                                    mask_sorted = cum_mass <= (q_hat + 1e-7)
                                mask_sorted &= valid_p[:, None]
                                mask_sorted[oos_mask] = False
                                cset_rows = [
                                    [options[int(sorted_idx[i, j])] for j in np.flatnonzero(mask_sorted[i])]
                                    if np.any(mask_sorted[i])
                                    else []
                                    for i in range(m_docs)
                                ]
                        else:
                            sorted_probs = np.take_along_axis(probs, sorted_idx, axis=1)
                            cset_rows = []
                            for i in range(m_docs):
                                if oos_mask[i] or total_mass[i] < q_hat - 1e-7 or total_mass[i] < 0.5:
                                    cset_rows.append([])
                                else:
                                    s_row = []
                                    cmass = 0.0
                                    for j_pos in range(k_opts):
                                        idx_o = int(sorted_idx[i, j_pos])
                                        s_row.append(options[idx_o])
                                        cmass += float(sorted_probs[i, j_pos])
                                        if cmass >= q_hat - 1e-7:
                                            break
                                    cset_rows.append(s_row)

                    uncal_or_empty = (not conformal.is_calibrated) or (n_cal == 0)
                    for i in range(m_docs):
                        cset_list = cset_rows[i]
                        card = len(cset_list)
                        raw_amb = card > 1
                        is_empty_set = card == 0
                        m_i = float(marg_arr[i])
                        tp_i = float(top_p[i])
                        odds_i = float(odds_arr[i])
                        mg_act = bool(
                            (not is_strict)
                            and raw_amb
                            and (eff_m > 0.0)
                            and (m_i >= eff_m)
                            and (tp_i >= conf_floor)
                            and (odds_i >= eff_g)
                        )
                        if oos_mask[i] or (is_strict and uncal_or_empty):
                            cset_amb = True
                        else:
                            cset_amb = raw_amb and not mg_act

                        row_csets[i][fname] = cset_list
                        row_margins[i][fname] = m_i
                        row_mthresh[i][fname] = eff_m
                        row_mgact[i][fname] = mg_act
                        row_odds[i][fname] = odds_i
                        row_ror[i][fname] = eff_g
                        row_cfloor[i][fname] = conf_floor

                        if field_nonfin[i]:
                            row_values[i][fname] = False if spec["is_bool"] else options[0]
                            row_confs[i][fname] = float("nan")
                            row_probs[i][fname] = (
                                {"False": float("nan"), "True": float("nan")}
                                if spec["is_bool"]
                                else {opt: float("nan") for opt in options}
                            )
                            row_csets[i][fname] = []
                            row_reasons[i][fname] = "non_finite_input"
                            row_amb_fields[i].append(fname)
                            row_is_amb[i] = True
                            if spec["escalate"]:
                                row_esc_fields[i].append(fname)
                            continue

                        if low_fam[i]:
                            row_reasons[i][fname] = "low_feature_familiarity"
                        elif low_proto[i]:
                            row_reasons[i][fname] = "low_semantic_familiarity"

                        pred_lbl_i = (
                            ("True" if row_values[i][fname] else "False")
                            if spec["is_bool"]
                            else str(row_values[i][fname])
                        )
                        pred_excluded_i = pred_lbl_i not in cset_list
                        field_amb = (
                            (card != 1 or cset_amb or is_empty_set or pred_excluded_i)
                            if is_strict
                            else (is_empty_set or cset_amb or pred_excluded_i)
                        )
                        if field_amb:
                            row_amb_fields[i].append(fname)
                            if spec["escalate"]:
                                row_is_amb[i] = True
                                row_esc_fields[i].append(fname)

                if is_tfidf_proj and np.any(zero_emb_mask):
                    for i in np.flatnonzero(zero_emb_mask):
                        row_fam_trig[i] = True
                        for spec in field_specs:
                            fname = spec["name"]
                            row_csets[i][fname] = []
                            row_reasons[i].setdefault(fname, "low_feature_familiarity")
                            if fname not in row_amb_fields[i]:
                                row_amb_fields[i].append(fname)
                            if spec["escalate"]:
                                row_is_amb[i] = True
                                if fname not in row_esc_fields[i]:
                                    row_esc_fields[i].append(fname)

                for spec in field_specs:
                    fname = spec["name"]
                    fam_t = spec["fam_t"]
                    proto_t = spec["proto_t"]
                    if raw_norms_arr is not None and fam_t > 0.0:
                        low_f_idxs = np.flatnonzero(
                            (raw_norms_arr < (fam_t - 1e-12)) | (~np.isfinite(raw_norms_arr))
                        )
                        for i in low_f_idxs:
                            row_fam_trig[i] = True
                            row_csets[i][fname] = []
                            row_reasons[i].setdefault(fname, "low_feature_familiarity")
                            if fname not in row_amb_fields[i]:
                                row_amb_fields[i].append(fname)
                            if spec["escalate"]:
                                row_is_amb[i] = True
                                if fname not in row_esc_fields[i]:
                                    row_esc_fields[i].append(fname)
                    if proto_t > 0.0 and spec["protos_unit"] is not None:
                        for i in range(m_docs):
                            p_sim = row_proto_sims[i].get(fname)
                            if p_sim is not None and p_sim < (proto_t - 1e-6):
                                row_proto_trig[i] = True
                                row_csets[i][fname] = []
                                row_reasons[i].setdefault(fname, "low_semantic_familiarity")
                                if fname not in row_amb_fields[i]:
                                    row_amb_fields[i].append(fname)
                                if spec["escalate"]:
                                    row_is_amb[i] = True
                                    if fname not in row_esc_fields[i]:
                                        row_esc_fields[i].append(fname)
                            elif row_reasons[i].get(fname) == "low_semantic_familiarity":
                                row_proto_trig[i] = True
                            elif row_reasons[i].get(fname) == "low_feature_familiarity":
                                row_fam_trig[i] = True
                    else:
                        for i in range(m_docs):
                            rsn = row_reasons[i].get(fname)
                            if rsn == "low_semantic_familiarity":
                                row_proto_trig[i] = True
                            elif rsn == "low_feature_familiarity":
                                row_fam_trig[i] = True

                if is_strict:
                    for spec in field_specs:
                        fname = spec["name"]
                        sel_cal = spec["sel_cal"]
                        if sel_cal is not None:
                            score_kind = getattr(sel_cal, "score_name", "margin")
                            use_conf = score_kind in ("p1", "top_prob", "confidence")
                            for i in range(m_docs):
                                s_val = float(row_confs[i][fname] if use_conf else row_margins[i].get(fname, 0.0))
                                if not sel_cal.should_accept(s_val):
                                    row_csets[i][fname] = []
                                    row_reasons[i].setdefault(fname, "selective_risk_gate")
                                    if fname not in row_amb_fields[i]:
                                        row_amb_fields[i].append(fname)
                                    if spec["escalate"]:
                                        row_is_amb[i] = True
                                        if fname not in row_esc_fields[i]:
                                            row_esc_fields[i].append(fname)
                        if aci_has_floor:
                            for i in range(m_docs):
                                if fname in row_margins[i] and not aci.should_accept_margin(row_margins[i][fname]):
                                    row_csets[i][fname] = []
                                    row_reasons[i].setdefault(fname, "aci_margin_floor")
                                    if fname not in row_amb_fields[i]:
                                        row_amb_fields[i].append(fname)
                                    if spec["escalate"]:
                                        row_is_amb[i] = True
                                        if fname not in row_esc_fields[i]:
                                            row_esc_fields[i].append(fname)

                joint_extra_meta: Optional[List[Optional[Dict[str, Any]]]] = None
                if jcal is not None and hasattr(jcal, "decode_batch"):
                    j_dicts, j_probs, j_margs, j_acc = jcal.decode_batch(probs_by_field_arr)
                    joint_fields = tuple(getattr(jcal, "field_names", ()))
                    joint_extra_meta = [None] * m_docs
                    for i in range(m_docs):
                        if row_reasons[i]:
                            continue
                        row_values[i].update(j_dicts[i])
                        acc_i = bool(j_acc[i])
                        joint_extra_meta[i] = {
                            "joint_map_prob": float(j_probs[i]),
                            "joint_map_margin": float(j_margs[i]),
                            "joint_accepted": acc_i,
                        }
                        if is_strict:
                            if acc_i:
                                for jf in joint_fields:
                                    row_csets[i][jf] = [str(j_dicts[i][jf])]
                                row_amb_fields[i] = [f for f in row_amb_fields[i] if f not in joint_fields]
                                row_esc_fields[i] = [f for f in row_esc_fields[i] if f not in joint_fields]
                                row_is_amb[i] = bool(row_esc_fields[i])
                            else:
                                for jf in joint_fields:
                                    row_csets[i][jf] = []
                                    row_reasons[i].setdefault(jf, "joint_selective_risk_gate")
                                    if jf not in row_amb_fields[i]:
                                        row_amb_fields[i].append(jf)
                                    if getattr(self.schema.fields[jf], "escalate_on_ambiguity", True):
                                        row_is_amb[i] = True
                                        if jf not in row_esc_fields[i]:
                                            row_esc_fields[i].append(jf)

                per_item_ms = ((time.perf_counter() - t_chunk_start) * 1000.0) / m_docs
                for i in range(m_docs):
                    vals_i = self.schema.validate_decision(row_values[i]) if custom_validate else row_values[i]
                    fam_score_i = float(raw_norms_arr[i]) if raw_norms_arr is not None else None
                    reasons_i = row_reasons[i]
                    meta_i: Dict[str, Any] = {
                        "familiarity_score": fam_score_i,
                        "familiarity_triggered": bool(row_fam_trig[i]),
                        "prototype_similarities": row_proto_sims[i],
                        "prototype_triggered": bool(row_proto_trig[i]),
                        "abstention_reason": next(iter(reasons_i.values()), None),
                        "abstention_reasons": reasons_i,
                    }
                    if joint_extra_meta is not None and joint_extra_meta[i] is not None:
                        meta_i.update(joint_extra_meta[i])
                    results.append(
                        DecisionResult(
                            schema_name=schema_name,
                            schema_digest=schema_digest,
                            prompt=chunk_prompts[i],
                            values=vals_i,
                            confidences=row_confs[i],
                            conformal_sets=row_csets[i],
                            probabilities=row_probs[i],
                            is_ambiguous=bool(row_is_amb[i]),
                            latency_ms=per_item_ms,
                            alpha=eff_alpha,
                            receipt=None,
                            margins=row_margins[i],
                            margin_thresholds=row_mthresh[i],
                            margin_gate_active=row_mgact[i],
                            odds_ratios=row_odds[i],
                            relative_odds_ratio_thresholds=row_ror[i],
                            confidence_floors=row_cfloor[i],
                            ambiguous_fields=row_amb_fields[i],
                            escalated_fields=row_esc_fields[i],
                            telemetry=None,
                            is_cache_hit=False,
                            embedding=X[i],
                            familiarity_score=fam_score_i,
                            metadata=meta_i,
                        )
                    )
            return results

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
        use_cache: Optional[bool] = False,
    ) -> List[DecisionResult]:
        """Processes multiple inputs in batch with optional continuous telemetry."""
        if len(prompts) == 0:
            return []

        eff_use_cache = self.use_cache if use_cache is None else bool(use_cache)
        matching_telemetry = telemetry is not None and len(telemetry) == len(prompts)
        any_telemetry = matching_telemetry and any(t is not None for t in telemetry)
        proj_obj = getattr(self.model, "projector", self.projector)

        can_vectorize = (
            not record_receipt
            and not eff_use_cache
            and not any_telemetry
            and all(isinstance(p, str) for p in prompts)
            and type(self.model).__name__ in ("CompiledSystemOneModel", "SystemOneModel")
            and hasattr(self.model, "heads")
            and callable(getattr(proj_obj, "project_batch", None))
            and not hasattr(proj_obj, "project_with_norm")
            and all(isinstance(f, (ChoiceField, BooleanField)) for f in self.schema.fields.values())
            and all(
                getattr(h, "_exemplar_bank", None) is None
                and getattr(h, "backend", "numpy") != "mlx"
                for h in getattr(self.model, "heads", {}).values()
            )
            and all(
                getattr(h, "_exemplar_bank", None) is None
                and getattr(h, "backend", "numpy") != "mlx"
                for h in getattr(self.model, "_field_heads", {}).values()
            )
            and not getattr(self.model, "_learned_prompt_overrides", None)
            and not getattr(self, "_learned_prompt_overrides", None)
        )
        if can_vectorize:
            return self._decide_batch_vectorized(
                prompts,
                alpha=alpha,
                margin_threshold=margin_threshold,
                relative_odds_ratio=relative_odds_ratio,
                confidence_floor_tau0=confidence_floor_tau0,
                recency_weighted=recency_weighted,
                strict=strict,
            )

        if matching_telemetry:
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
                    use_cache=eff_use_cache,
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
                use_cache=eff_use_cache,
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


def compile_engine_callable(
    engine: SystemOneEngine,
    *,
    storage_mode: str = "float32",
) -> FastDecisionCallable:
    """Compiles a low-overhead fast decision callable for a SystemOneEngine."""
    return engine.compile_callable(storage_mode=storage_mode)


# Compatibility aliases
ReflexEngine = SystemOneEngine
System1Engine = SystemOneEngine

__all__ = [
    "SystemOneEngine",
    "System1Engine",
    "ReflexEngine",
    "compile_engine_callable",
    "DecisionResult",
    "BenchmarkReport",
    "FieldEvaluationMetrics",
    "EvaluationReport",
    "RecalibrationMetrics",
    "RecalibrationReport",
]
