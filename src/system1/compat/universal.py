"""Universal wire-protocol decision gateway for Gemini Decision 1, Convai Laya, and TypeSafe Jev.

Implements a zero-egress-safe, loopback-restricted HTTP and ASGI gateway with:
  - Auto-detected wire contracts (``gemini``, ``laya``, ``jev``) and multimodal text extraction
  - Bijective ``QuestionContract`` mapping across choice, noul, multichoice, continuous,
    and up-to-64-level ordinal score questions
  - Soft-probability Ridge distillation into ``AutoCutoverLifecycle``
  - Strict JSON parsing (rejecting ``NaN``/``Infinity`` and duplicate keys), header auth
    (``Authorization: Bearer`` / ``x-goog-api-key`` / ``x-api-key``), ``Origin`` allowlisting,
    body size caps, schema drift isolation, and ``freeze_learning`` support
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import http.server
import json
import math
import re
import socketserver
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from system1.compat.typesafe import DotDict, Usage, ZeroEgressViolationError
from system1.core.schema import (
    BooleanField,
    ChoiceField,
    DecisionField,
    DecisionSchema,
    MultiChoiceField,
    ScoreField,
)
from system1.cutover import (
    _ACTIVE_CANARY_TX_TLS,
    AutoCutoverLifecycle,
    CutoverConfig,
    CutoverState,
    _bind_active_canary_tx,
    _normalize_cache_key,
)

_SAFE_IDENT_RE = re.compile(r"[^A-Za-z0-9_]+")
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_GATEWAY_POST_ROUTES = frozenset(
    ("/v1beta/decisions", "/v1/systemone", "/v1/laya/decide", "/v1/decisions")
)


_HTTP_TOKEN_CHARS = frozenset(
    "!#$%&'*+-.^_`|~0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)


def _is_json_media_type(ctype: str) -> bool:
    """Return True iff Content-Type parses strictly as application/json with valid RFC 9110 parameters."""
    if not isinstance(ctype, str):
        return False
    if any((ord(ch) < 0x20 and ch != "\t") or ord(ch) == 0x7F or ord(ch) > 0xFF for ch in ctype):
        return False
    s = ctype.strip(" \t")
    n = len(s)
    if n == 0:
        return False

    pos = 0
    t_start = pos
    while pos < n and s[pos] in _HTTP_TOKEN_CHARS:
        pos += 1
    media_type = s[t_start:pos].lower()
    if pos >= n or s[pos] != "/":
        return False
    pos += 1
    s_start = pos
    while pos < n and s[pos] in _HTTP_TOKEN_CHARS:
        pos += 1
    media_subtype = s[s_start:pos].lower()
    if media_type != "application" or media_subtype != "json":
        return False

    while pos < n and s[pos] in (" ", "\t"):
        pos += 1
    if pos < n and s[pos] != ";":
        return False

    seen_charset: Optional[str] = None
    while pos < n:
        if s[pos] != ";":
            return False
        pos += 1
        while pos < n and s[pos] in (" ", "\t"):
            pos += 1
        if pos >= n:
            break
        if s[pos] == ";":
            continue

        k_start = pos
        while pos < n and s[pos] in _HTTP_TOKEN_CHARS:
            pos += 1
        p_key = s[k_start:pos].lower()
        if not p_key or pos >= n or s[pos] != "=":
            return False
        pos += 1
        if pos >= n:
            return False

        if s[pos] == '"':
            pos += 1
            q_chars: List[str] = []
            closed = False
            while pos < n:
                ch = s[pos]
                if ch == '"':
                    pos += 1
                    closed = True
                    break
                if ch == "\\":
                    pos += 1
                    if pos >= n:
                        return False
                    esc = s[pos]
                    esc_ord = ord(esc)
                    if not (
                        esc in (" ", "\t")
                        or (0x21 <= esc_ord <= 0x7E)
                        or (0x80 <= esc_ord <= 0xFF)
                    ):
                        return False
                    q_chars.append(esc)
                    pos += 1
                else:
                    ch_ord = ord(ch)
                    if not (
                        ch in (" ", "\t")
                        or ch_ord == 0x21
                        or (0x23 <= ch_ord <= 0x5B)
                        or (0x5D <= ch_ord <= 0x7E)
                        or (0x80 <= ch_ord <= 0xFF)
                    ):
                        return False
                    q_chars.append(ch)
                    pos += 1
            if not closed:
                return False
            p_val = "".join(q_chars)
        else:
            v_start = pos
            while pos < n and s[pos] in _HTTP_TOKEN_CHARS:
                pos += 1
            p_val = s[v_start:pos]
            if not p_val:
                return False

        while pos < n and s[pos] in (" ", "\t"):
            pos += 1
        if pos < n and s[pos] != ";":
            return False
        if p_key == "charset":
            if seen_charset is not None:
                return False
            p_val_low = p_val.lower()
            if p_val_low not in ("utf-8", "utf8"):
                return False
            seen_charset = p_val_low
    return True


class InteropError(ValueError):
    """Base error for Universal Decision Gateway validation and HTTP protocol failures."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = int(status_code)


class PayloadTooLargeError(InteropError):
    """Raised when an HTTP request body exceeds ``max_body_bytes``."""

    def __init__(self, message: str = "Request payload exceeds size limit") -> None:
        super().__init__(message, status_code=413)


class UnsupportedMediaError(InteropError):
    """Raised when ``Content-Type`` is not ``application/json``."""

    def __init__(self, message: str = "Content-Type must be application/json") -> None:
        super().__init__(message, status_code=415)


class SchemaValidationError(InteropError):
    """Raised when ``questions``, ``input``, or ``state`` violates schema contracts."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=422)


class AuthError(InteropError):
    """Raised when Bearer or API key authentication fails."""

    def __init__(self, message: str = "Invalid or missing authentication credential") -> None:
        super().__init__(message, status_code=401)


class ForbiddenOriginError(InteropError):
    """Raised when an ``Origin`` header is not on the configured allowlist."""

    def __init__(self, message: str = "Disallowed Origin header") -> None:
        super().__init__(message, status_code=403)


def _reject_nonfinite_json(token: str) -> Any:
    raise InteropError(f"Non-finite JSON constant is not allowed: {token!r}", status_code=400)


def _parse_strict_json_float(token: str) -> float:
    try:
        val = float(token)
    except ValueError as exc:
        raise InteropError(f"Invalid JSON float token: {token!r}", status_code=400) from exc
    if not math.isfinite(val):
        raise InteropError(f"Non-finite JSON number is not allowed: {token!r}", status_code=400)
    return val


def _reject_duplicate_json_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise InteropError(f"Duplicate JSON object key is not allowed: {k!r}", status_code=400)
        out[k] = v
    return out


_MAX_JSON_NESTING_DEPTH = 64


def _validate_payload_tree(root: Any, *, max_depth: int = _MAX_JSON_NESTING_DEPTH) -> None:
    stack: List[Tuple[Any, int]] = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, Mapping):
            if depth >= max_depth:
                raise InteropError(
                    f"JSON nesting depth exceeds maximum allowed limit ({max_depth})",
                    status_code=400,
                )
            for k, v in node.items():
                stack.append((k, depth + 1))
                stack.append((v, depth + 1))
        elif isinstance(node, (list, tuple, set, frozenset)):
            if depth >= max_depth:
                raise InteropError(
                    f"JSON nesting depth exceeds maximum allowed limit ({max_depth})",
                    status_code=400,
                )
            for v in node:
                stack.append((v, depth + 1))
        elif isinstance(node, str):
            try:
                node.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise InteropError(
                    f"JSON string contains invalid unpaired UTF-16 surrogate: {exc}",
                    status_code=400,
                ) from exc
        elif isinstance(node, (float, np.floating)) and not isinstance(node, bool):
            if not math.isfinite(float(node)):
                raise InteropError(f"Non-finite numeric value in payload: {node!r}", status_code=400)


_validate_container_depth = _validate_payload_tree


def strict_json_loads(raw: bytes | str) -> Any:
    """Decode UTF-8 JSON while strictly rejecting NaN/Infinity constants and duplicate keys."""
    if isinstance(raw, (bytes, bytearray, memoryview)):
        try:
            text = bytes(raw).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InteropError(f"Invalid UTF-8 JSON body: {exc}", status_code=400) from exc
    elif isinstance(raw, str):
        text = raw
    else:
        raise InteropError(f"Expected bytes or str JSON body, got {type(raw).__name__}", status_code=400)

    try:
        parsed = json.loads(
            text,
            parse_constant=_reject_nonfinite_json,
            parse_float=_parse_strict_json_float,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except InteropError:
        raise
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise InteropError(f"Malformed JSON body: {exc}", status_code=400) from exc
    _validate_container_depth(parsed)
    return parsed


def strict_json_dumps(payload: Any) -> bytes:
    """Serialize payload to UTF-8 JSON while disallowing NaN/Infinity."""
    try:
        return json.dumps(payload, allow_nan=False, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise InteropError(f"Unable to serialize JSON payload: {exc}", status_code=500) from exc


def _safe_error_json_bytes(message: str) -> bytes:
    """Serialize an error message safely to UTF-8 JSON even if the text contains surrogates."""
    clean_msg = str(message).encode("utf-8", "backslashreplace").decode("utf-8")
    return strict_json_dumps({"error": clean_msg})


def parse_finite_float(value: Any, *, field_name: str = "value") -> float:
    """Coerce numeric or stringified float (Laya edge case) into a validated finite float."""
    if isinstance(value, bool):
        raise InteropError(f"Expected float for {field_name}, got bool", status_code=422)
    try:
        num = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InteropError(f"Invalid float value {value!r} for {field_name}", status_code=422) from exc
    if not math.isfinite(num):
        raise InteropError(f"Non-finite float {value!r} for {field_name}", status_code=422)
    return 0.0 if num == 0.0 else num


def _parse_unit_interval_float(val: Any, *, field_name: str) -> float:
    f = parse_finite_float(val, field_name=field_name)
    if f < -1e-6 or f > 1.0 + 1e-6:
        raise InteropError(f"Out-of-range unit interval value for {field_name}: {f!r}", status_code=422)
    return float(min(1.0, max(0.0, f)))


def coerce_probability_dict(raw_probs: Mapping[Any, Any], keys: Sequence[str]) -> Dict[str, float]:
    """Coerce stringified/numeric probability mapping onto target keys and normalize to 1.0."""
    if not isinstance(raw_probs, Mapping):
        raise InteropError("probabilities must be a mapping", status_code=422)
    str_map: Dict[str, Any] = {}
    for rk, rv in raw_probs.items():
        sk = str(rk)
        if sk in str_map:
            raise InteropError(f"Duplicate probability key after stringification: {sk!r}", status_code=422)
        str_map[sk] = rv
    parsed: Dict[str, float] = {}
    for k in keys:
        sk = str(k)
        v = str_map.get(sk, 0.0)
        raw_p = parse_finite_float(v, field_name=f"probabilities[{k!r}]")
        if raw_p < -1e-9:
            raise InteropError(f"Negative probability {raw_p!r} for {k!r}", status_code=422)
        parsed[sk] = 0.0 if raw_p <= 0.0 else raw_p
    max_p = max(parsed.values(), default=0.0)
    if not math.isfinite(max_p) or max_p <= 0.0:
        raise InteropError("Probability vector has non-positive sum", status_code=422)
    scaled = {k: float(v / max_p) for k, v in parsed.items()}
    total = math.fsum(scaled.values())
    if not math.isfinite(total) or total <= 0.0:
        raise InteropError("Probability vector has non-positive sum", status_code=422)
    out = {k: float(v / total) for k, v in scaled.items()}
    if not all(math.isfinite(v) and v >= 0.0 for v in out.values()) or not math.isclose(
        math.fsum(out.values()), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise InteropError("Probability vector failed normalization", status_code=422)
    return out


def detect_wire_format(path: str, payload: Mapping[str, Any]) -> str:
    """Infer competitor wire contract from HTTP route, model string, and payload keys."""
    clean_path = (path or "").split("?", 1)[0].rstrip("/")
    model = str(payload.get("model") or "").strip().lower()
    if clean_path.endswith("/v1beta/decisions") or "gemini-decision" in model or model.startswith("gemini"):
        return "gemini"
    if clean_path.endswith("/v1/laya/decide") or model.startswith("laya"):
        return "laya"
    if (
        clean_path.endswith("/v1/systemone")
        or model.startswith("jev")
        or ("state" in payload and "input" not in payload and "contents" not in payload)
    ):
        return "jev"
    if "input" in payload or "contents" in payload:
        return "gemini"
    return "jev"


def _canonical_structured_key(node: Any, depth: int = 0) -> str:
    """Return a bijective, type-tagged, length-prefixed cache key for structured inputs."""
    if depth >= _MAX_JSON_NESTING_DEPTH:
        raise InteropError(
            f"Input structure nesting depth exceeds maximum allowed limit ({_MAX_JSON_NESTING_DEPTH})",
            status_code=400,
        )
    if node is None:
        return "N;"
    if isinstance(node, (bool, np.bool_)):
        return "B1;" if bool(node) else "B0;"
    if isinstance(node, (int, np.integer)):
        return f"I{int(node)};"
    if isinstance(node, (float, np.floating)):
        fv = float(node)
        if not math.isfinite(fv):
            raise InteropError(f"Non-finite numeric value in input: {node!r}", status_code=400)
        if fv == 0.0:
            fv = 0.0
        return f"F{fv.hex()};"
    if isinstance(node, str):
        return f"S{len(node)}:{node};"
    if isinstance(node, list):
        inner = "".join(_canonical_structured_key(item, depth + 1) for item in node)
        return f"L{len(node)}:{inner};"
    if isinstance(node, tuple):
        inner = "".join(_canonical_structured_key(item, depth + 1) for item in node)
        return f"Q{len(node)}:{inner};"
    if isinstance(node, (set, frozenset)):
        items = sorted(_canonical_structured_key(item, depth + 1) for item in node)
        return f"T{len(items)}:{''.join(items)};"
    if isinstance(node, Mapping):
        kv_pairs = sorted(
            (
                (
                    _canonical_structured_key(k, depth + 1),
                    _canonical_structured_key(v, depth + 1),
                )
                for k, v in node.items()
            ),
            key=lambda kv: kv[0],
        )
        inner = "".join(f"{k_s}{v_s}" for k_s, v_s in kv_pairs)
        return f"M{len(kv_pairs)}:{inner};"
    tname = type(node).__qualname__
    sval = str(node)
    return f"O{len(tname)}:{tname}:{len(sval)}:{sval};"


def _input_structure_signature(node: Any, depth: int = 0) -> str:
    """Return a canonical shape+key+type skeleton signature for schema-support gating."""
    if depth >= _MAX_JSON_NESTING_DEPTH:
        raise InteropError(
            f"Input structure nesting depth exceeds maximum allowed limit ({_MAX_JSON_NESTING_DEPTH})",
            status_code=400,
        )
    if node is None:
        return "null"
    if isinstance(node, (bool, np.bool_)):
        return "bool"
    if isinstance(node, (int, float, np.integer, np.floating)):
        return "num"
    if isinstance(node, str):
        return "str"
    if isinstance(node, (list, tuple)):
        elem_sigs = sorted({_input_structure_signature(item, depth + 1) for item in node})
        tag = "L" if isinstance(node, list) else "Q"
        return f"{tag}[{','.join(elem_sigs)}]"
    if isinstance(node, (set, frozenset)):
        elem_sigs = sorted({_input_structure_signature(item, depth + 1) for item in node})
        return f"T[{','.join(elem_sigs)}]"
    if isinstance(node, Mapping):
        items = sorted(
            (
                (
                    _canonical_structured_key(k, depth + 1),
                    json.dumps(str(k), ensure_ascii=False),
                    _input_structure_signature(v, depth + 1),
                )
                for k, v in node.items()
            ),
            key=lambda item: item[0],
        )
        return "M{" + ",".join(f"{k_json}:{v_sig}" for _, k_json, v_sig in items) + "}"
    return f"O({type(node).__qualname__})"


def _canonicalize_for_semantic_json(node: Any, depth: int = 0) -> Tuple[Any, bool]:
    """Convert structured Python/JSON values into canonical JSON types and track non-empty leaves."""
    if depth >= _MAX_JSON_NESTING_DEPTH:
        raise InteropError(
            f"Input structure nesting depth exceeds maximum allowed limit ({_MAX_JSON_NESTING_DEPTH})",
            status_code=400,
        )
    if node is None:
        return None, False
    if isinstance(node, (bool, np.bool_)):
        return bool(node), True
    if isinstance(node, (int, np.integer)):
        return int(node), True
    if isinstance(node, (float, np.floating)):
        fv = float(node)
        if not math.isfinite(fv):
            raise InteropError(f"Non-finite numeric value in input: {node!r}", status_code=400)
        if fv == 0.0:
            fv = 0.0
        return fv, True
    if isinstance(node, str):
        return node, bool(node.strip())
    if isinstance(node, (list, tuple)):
        out_list: List[Any] = []
        any_nonempty = False
        for item in node:
            c_item, item_nonempty = _canonicalize_for_semantic_json(item, depth + 1)
            out_list.append(c_item)
            if item_nonempty:
                any_nonempty = True
        return out_list, any_nonempty
    if isinstance(node, (set, frozenset)):
        ordered = sorted(node, key=lambda x: _canonical_structured_key(x, depth + 1))
        out_set: List[Any] = []
        any_nonempty = False
        for item in ordered:
            c_item, item_nonempty = _canonicalize_for_semantic_json(item, depth + 1)
            out_set.append(c_item)
            if item_nonempty:
                any_nonempty = True
        return out_set, any_nonempty
    if isinstance(node, Mapping):
        items = sorted(
            node.items(),
            key=lambda kv: _canonical_structured_key(kv[0], depth + 1),
        )
        out_map: Dict[str, Any] = {}
        any_nonempty = False
        for k, v in items:
            c_val, val_nonempty = _canonicalize_for_semantic_json(v, depth + 1)
            out_map[str(k)] = c_val
            if val_nonempty:
                any_nonempty = True
        return out_map, any_nonempty
    s = str(node)
    return s, bool(s.strip())


_PRIMARY_SEMANTIC_KEYS: Tuple[str, ...] = ("input", "state", "contents", "prompt")
_GATEWAY_ENVELOPE_CONTROL_KEYS: FrozenSet[str] = frozenset({"questions", "slot", "force_teacher", "model"})


def extract_semantic_text(payload: Mapping[str, Any]) -> Tuple[str, Any]:
    """Extract canonical semantic representation and original structured input from any provider payload."""
    present_env: List[Tuple[str, Any]] = [
        (key, payload[key])
        for key in _PRIMARY_SEMANTIC_KEYS
        if key in payload and payload[key] is not None
    ]
    if not present_env:
        raise SchemaValidationError("Request must include non-empty 'input', 'state', or 'contents'")
    extra_env: Dict[str, Any] = {
        k: v
        for k, v in payload.items()
        if k not in _PRIMARY_SEMANTIC_KEYS and k not in _GATEWAY_ENVELOPE_CONTROL_KEYS
    }
    for _, env_val in present_env:
        _validate_payload_tree(env_val)
    for ext_val in extra_env.values():
        _validate_payload_tree(ext_val)
    raw_input = present_env[0][1]
    if len(present_env) == 1 and isinstance(raw_input, str):
        primary_text = raw_input.strip()
        if not primary_text:
            raise SchemaValidationError("Extracted input/state text cannot be empty")
        if not extra_env:
            return primary_text, raw_input
    else:
        primary_target = raw_input if len(present_env) == 1 else {k: v for k, v in present_env}
        canon_primary, has_nonempty = _canonicalize_for_semantic_json(primary_target)
        if not has_nonempty:
            raise SchemaValidationError("Extracted input/state text cannot be empty")
        if not extra_env:
            text = json.dumps(canon_primary, ensure_ascii=False, sort_keys=True, separators=(", ", ": "))
            if not text:
                raise SchemaValidationError("Extracted input/state text cannot be empty")
            return text, raw_input

    combined_env: Dict[str, Any] = {**{k: v for k, v in present_env}, **extra_env}
    canon_combined, _ = _canonicalize_for_semantic_json(combined_env)
    text = json.dumps(canon_combined, ensure_ascii=False, sort_keys=True, separators=(", ", ": "))
    if not text:
        raise SchemaValidationError("Extracted input/state text cannot be empty")
    return text, raw_input


@dataclass(frozen=True)
class QuestionContract:
    """Normalized specification for a single decision question across all 3 formats."""

    external_name: str
    internal_name: str
    q_type: str  # "choice" | "noul" | "score" | "multichoice"
    options: Tuple[str, ...] = ()
    descriptions: Dict[str, str] = field(default_factory=dict)
    instructions: str = ""
    threshold: float = 0.5
    true_description: str = "True / Positive"
    false_description: str = "False / Negative"
    min_value: float = 0.0
    max_value: float = 1.0
    low_description: str = "Minimum boundary"
    high_description: str = "Maximum boundary"
    criteria_legend: Tuple[str, ...] = ()

    def to_decision_field(self) -> DecisionField:
        if self.q_type == "choice":
            return ChoiceField(
                options=list(self.options),
                descriptions=dict(self.descriptions) if self.descriptions else None,
                description=self.instructions,
            )
        if self.q_type == "noul":
            return BooleanField(
                threshold=self.threshold,
                true_description=self.true_description,
                false_description=self.false_description,
                description=self.instructions,
            )
        if self.q_type == "score":
            if self.criteria_legend:
                return ChoiceField(
                    options=[str(i) for i in range(len(self.criteria_legend))],
                    descriptions={str(i): txt for i, txt in enumerate(self.criteria_legend)},
                    description=self.instructions,
                    metadata={"typesafe_score_levels": tuple(self.criteria_legend)},
                )
            return ScoreField(
                min_value=self.min_value,
                max_value=self.max_value,
                low_description=self.low_description,
                high_description=self.high_description,
                description=self.instructions,
            )
        if self.q_type == "multichoice":
            return MultiChoiceField(
                options=list(self.options),
                descriptions=dict(self.descriptions) if self.descriptions else None,
                threshold=self.threshold,
                description=self.instructions,
            )
        raise SchemaValidationError(f"Unsupported question type: {self.q_type!r}")


@dataclass(frozen=True)
class CompiledContractSchema:
    """Bijective schema mapping between external JSON question keys and System 1 fields."""

    schema: DecisionSchema
    contracts: Tuple[QuestionContract, ...]
    ext_to_int: Dict[str, str]
    int_to_ext: Dict[str, str]
    fingerprint: str

    def primary_contract(self) -> QuestionContract:
        return self.contracts[0]


def _sanitize_field_identifier(raw_name: str, index: int) -> str:
    cleaned = _SAFE_IDENT_RE.sub("_", str(raw_name)).strip("_")
    if not cleaned:
        cleaned = "field"
    return f"q{index}_{cleaned}"[:120]


def compile_questions_contract(questions: Any) -> CompiledContractSchema:
    """Compile Gemini/Laya/Jev ``questions`` mappings into a bijective DecisionSchema."""
    if not isinstance(questions, Mapping) or not questions:
        raise SchemaValidationError("'questions' must be a non-empty mapping")

    contracts: List[QuestionContract] = []
    fields: Dict[str, DecisionField] = {}
    ext_to_int: Dict[str, str] = {}
    int_to_ext: Dict[str, str] = {}
    fingerprint_payload: Dict[str, Any] = {}

    seen_stripped: Set[str] = set()
    ordered_items: List[Tuple[str, Any]] = []
    for ext_name_raw, spec in questions.items():
        ext_name = str(ext_name_raw)
        stripped = ext_name.strip()
        if not stripped:
            raise SchemaValidationError("Question name cannot be empty")
        if stripped in seen_stripped:
            raise SchemaValidationError(f"Duplicate normalized question name: {stripped!r}")
        seen_stripped.add(stripped)
        ordered_items.append((ext_name, spec))
    ordered_items.sort(key=lambda kv: (kv[0].strip(), kv[0]))

    for idx, (ext_name, spec) in enumerate(ordered_items):
        if not isinstance(spec, Mapping):
            raise SchemaValidationError(f"Question {ext_name!r} spec must be a mapping")
        q_type = str(spec.get("type") or "").strip().lower()
        if q_type == "boolean":
            q_type = "noul"
        instructions = str(spec.get("instructions") or spec.get("description") or "")
        internal_name = _sanitize_field_identifier(ext_name, idx)

        if q_type in ("choice", "multichoice"):
            raw_criteria = spec.get("criteria")
            raw_options = spec.get("options")
            raw_desc_map: Dict[str, str] = {}
            if isinstance(raw_criteria, Mapping):
                raw_opt_list = list(raw_criteria.keys())
                for k, v in raw_criteria.items():
                    if not isinstance(k, bool):
                        raw_desc_map[str(k).strip()] = str(v)
            elif isinstance(raw_criteria, (list, tuple)):
                raw_opt_list = list(raw_criteria)
                if isinstance(spec.get("descriptions"), Mapping):
                    for k, v in spec.get("descriptions", {}).items():
                        if not isinstance(k, bool):
                            raw_desc_map[str(k).strip()] = str(v)
            elif isinstance(raw_options, Mapping):
                raw_opt_list = list(raw_options.keys())
                for k, v in raw_options.items():
                    if not isinstance(k, bool):
                        raw_desc_map[str(k).strip()] = str(v)
            elif isinstance(raw_options, (list, tuple)):
                raw_opt_list = list(raw_options)
                if isinstance(spec.get("descriptions"), Mapping):
                    for k, v in spec.get("descriptions", {}).items():
                        if not isinstance(k, bool):
                            raw_desc_map[str(k).strip()] = str(v)
            else:
                raise SchemaValidationError(f"Choice question {ext_name!r} requires 'criteria' or 'options'")

            cleaned_opts: List[str] = []
            for raw_opt in raw_opt_list:
                if isinstance(raw_opt, bool):
                    raise SchemaValidationError(f"Option in question {ext_name!r} cannot be boolean")
                opt_s = str(raw_opt).strip()
                if not opt_s:
                    raise SchemaValidationError(f"Empty option in question {ext_name!r}")
                cleaned_opts.append(opt_s)
            options = tuple(cleaned_opts)
            descriptions = {opt: raw_desc_map.get(opt, opt if isinstance(raw_criteria or raw_options, (list, tuple)) else "") for opt in options}

            if q_type == "choice" and not (2 <= len(options) <= 255):
                raise SchemaValidationError(
                    f"Choice question {ext_name!r} requires between 2 and 255 options, got {len(options)}"
                )
            if q_type == "multichoice" and not (1 <= len(options) <= 255):
                raise SchemaValidationError(
                    f"Multichoice question {ext_name!r} requires between 1 and 255 options, got {len(options)}"
                )
            if len(set(options)) != len(options):
                raise SchemaValidationError(f"Duplicate options in question {ext_name!r}")

            norm_descriptions = {opt: descriptions.get(opt, "") for opt in options}
            thresh = parse_finite_float(spec.get("threshold", 0.5), field_name="threshold")
            if q_type == "multichoice" and not (0.0 <= thresh <= 1.0):
                raise SchemaValidationError(
                    f"Multichoice question {ext_name!r} threshold must be in [0, 1], got {thresh}"
                )
            contract = QuestionContract(
                external_name=ext_name,
                internal_name=internal_name,
                q_type=q_type,
                options=options,
                descriptions=descriptions,
                instructions=instructions,
                threshold=thresh,
            )
            fp_entry: Dict[str, Any] = {
                "type": q_type,
                "options": list(options),
                "descriptions": norm_descriptions,
                "instructions": instructions,
            }
            if q_type == "multichoice":
                fp_entry["threshold"] = thresh
            fingerprint_payload[ext_name] = fp_entry

        elif q_type == "noul":
            thresh = parse_finite_float(spec.get("threshold", 0.5), field_name="threshold")
            if not (0.0 < thresh < 1.0):
                raise SchemaValidationError(f"Noul question {ext_name!r} threshold must be in (0, 1)")
            crit = spec.get("criteria")
            t_desc = spec.get("true_description")
            f_desc = spec.get("false_description")
            if isinstance(crit, Mapping):
                t_desc = t_desc or crit.get("true") or crit.get("True") or crit.get("yes")
                f_desc = f_desc or crit.get("false") or crit.get("False") or crit.get("no")
            elif isinstance(crit, (list, tuple)) and len(crit) == 2:
                f_desc = f_desc or str(crit[0])
                t_desc = t_desc or str(crit[1])
            elif isinstance(crit, str) and not t_desc:
                t_desc = crit
            contract = QuestionContract(
                external_name=ext_name,
                internal_name=internal_name,
                q_type="noul",
                instructions=instructions,
                threshold=thresh,
                true_description=str(t_desc or instructions or f"{ext_name} is true"),
                false_description=str(f_desc or f"not ({t_desc or instructions or ext_name})"),
            )
            fingerprint_payload[ext_name] = {
                "type": "noul",
                "threshold": thresh,
                "instructions": instructions,
                "true_description": contract.true_description,
                "false_description": contract.false_description,
            }

        elif q_type == "score":
            crit = spec.get("criteria")
            min_v = spec.get("min_value")
            max_v = spec.get("max_value")
            if isinstance(crit, (list, tuple)) and min_v is None and max_v is None:
                if not (2 <= len(crit) <= 64):
                    raise SchemaValidationError(
                        f"Ordinal Score {ext_name!r} criteria requires 2..64 levels, got {len(crit)}"
                    )
                legend = tuple(str(x) for x in crit)
                contract = QuestionContract(
                    external_name=ext_name,
                    internal_name=internal_name,
                    q_type="score",
                    options=tuple(str(i) for i in range(len(legend))),
                    criteria_legend=legend,
                    instructions=instructions,
                )
                fingerprint_payload[ext_name] = {
                    "type": "score_ordinal",
                    "criteria": list(legend),
                    "instructions": instructions,
                }
            else:
                lo = parse_finite_float(0.0 if min_v is None else min_v, field_name="min_value")
                hi = parse_finite_float(1.0 if max_v is None else max_v, field_name="max_value")
                if hi <= lo:
                    raise SchemaValidationError(f"Score {ext_name!r} max_value ({hi}) must exceed min_value ({lo})")
                contract = QuestionContract(
                    external_name=ext_name,
                    internal_name=internal_name,
                    q_type="score",
                    instructions=instructions,
                    min_value=lo,
                    max_value=hi,
                    low_description=str(spec.get("low_description") or f"Low {lo}"),
                    high_description=str(spec.get("high_description") or f"High {hi}"),
                )
                fingerprint_payload[ext_name] = {
                    "type": "score_continuous",
                    "min": lo,
                    "max": hi,
                    "instructions": instructions,
                    "low_description": contract.low_description,
                    "high_description": contract.high_description,
                }
        else:
            raise SchemaValidationError(f"Unsupported question type {q_type!r} for {ext_name!r}")

        contracts.append(contract)
        try:
            fields[internal_name] = contract.to_decision_field()
        except SchemaValidationError:
            raise
        except (ValueError, TypeError) as exc:
            raise SchemaValidationError(str(exc)) from exc
        ext_to_int[ext_name] = internal_name
        int_to_ext[internal_name] = ext_name

    fp = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    try:
        schema = DecisionSchema(fields=fields, schema_name=f"universal_{fp}")
    except SchemaValidationError:
        raise
    except (ValueError, TypeError) as exc:
        raise SchemaValidationError(str(exc)) from exc
    return CompiledContractSchema(
        schema=schema,
        contracts=tuple(contracts),
        ext_to_int=ext_to_int,
        int_to_ext=int_to_ext,
        fingerprint=fp,
    )


def diff_schemas(prev: CompiledContractSchema, curr: CompiledContractSchema) -> Dict[str, Any]:
    """Compare two compiled contracts and report structured schema drift details."""
    prev_map = {c.external_name: c for c in prev.contracts}
    curr_map = {c.external_name: c for c in curr.contracts}
    added_q = sorted(set(curr_map) - set(prev_map))
    removed_q = sorted(set(prev_map) - set(curr_map))
    changed: Dict[str, Dict[str, Any]] = {}
    for qname in sorted(set(prev_map) & set(curr_map)):
        pc = prev_map[qname]
        cc = curr_map[qname]
        pc_is_ord = pc.q_type == "score" and bool(pc.criteria_legend)
        cc_is_ord = cc.q_type == "score" and bool(cc.criteria_legend)
        if pc.q_type != cc.q_type or pc_is_ord != cc_is_ord:
            changed[qname] = {
                "reason": "type_changed",
                "from": "score_ordinal" if pc_is_ord else pc.q_type,
                "to": "score_ordinal" if cc_is_ord else cc.q_type,
            }
        elif pc.criteria_legend != cc.criteria_legend:
            changed[qname] = {
                "reason": "criteria_changed",
                "from": list(pc.criteria_legend),
                "to": list(cc.criteria_legend),
            }
        elif pc.options != cc.options:
            changed[qname] = {
                "reason": "options_changed",
                "added_options": sorted(set(cc.options) - set(pc.options)),
                "removed_options": sorted(set(pc.options) - set(cc.options)),
            }
        elif pc.q_type in ("noul", "multichoice") and float(pc.threshold) != float(cc.threshold):
            changed[qname] = {
                "reason": "threshold_changed",
                "from": pc.threshold,
                "to": cc.threshold,
            }
        elif pc.q_type == "score" and not pc_is_ord and (
            float(pc.min_value) != float(cc.min_value)
            or float(pc.max_value) != float(cc.max_value)
        ):
            changed[qname] = {
                "reason": "bounds_changed",
                "from": [pc.min_value, pc.max_value],
                "to": [cc.min_value, cc.max_value],
            }
        elif pc.instructions != cc.instructions:
            changed[qname] = {
                "reason": "instructions_changed",
                "from": pc.instructions,
                "to": cc.instructions,
            }
        elif (
            pc.descriptions != cc.descriptions
            or pc.true_description != cc.true_description
            or pc.false_description != cc.false_description
            or pc.low_description != cc.low_description
            or pc.high_description != cc.high_description
        ):
            changed[qname] = {
                "reason": "descriptions_changed",
                "question": qname,
            }
    return {
        "drifted": bool(added_q or removed_q or changed),
        "added_questions": added_q,
        "removed_questions": removed_q,
        "changed_questions": changed,
        "prev_fingerprint": prev.fingerprint,
        "curr_fingerprint": curr.fingerprint,
    }


def parse_teacher_answers(
    raw_response: Mapping[str, Any],
    compiled_schema: CompiledContractSchema,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Parse upstream Gemini/Laya/Jev response into lifecycle values, soft targets, and external answers."""
    raw_answers = raw_response.get("answers") or raw_response.get("decisions")
    if not isinstance(raw_answers, Mapping):
        raise InteropError("Upstream response missing 'answers' / 'decisions' mapping", status_code=502)

    lifecycle_values: Dict[str, Any] = {}
    soft_targets: Dict[str, Dict[str, Any]] = {}
    normalized_external: Dict[str, Any] = {}

    for c in compiled_schema.contracts:
        ans = raw_answers.get(c.external_name)
        if not isinstance(ans, Mapping):
            raise InteropError(f"Upstream response missing answer for {c.external_name!r}", status_code=502)

        if c.q_type == "choice":
            raw_probs = ans.get("probabilities")
            probs: Optional[Dict[str, float]] = None
            if isinstance(raw_probs, Mapping) and raw_probs:
                probs = coerce_probability_dict(raw_probs, c.options)
            raw_choice = ans.get("choice", ans.get("value"))
            if raw_choice is None and probs is not None:
                raw_choice = max(c.options, key=lambda opt: probs[opt])
            choice = str(raw_choice)
            if choice not in c.options:
                raise InteropError(f"Invalid upstream choice {choice!r} for {c.external_name!r}", status_code=502)
            if probs is None:
                probs = {opt: (1.0 if opt == choice else 0.0) for opt in c.options}
            conf = parse_finite_float(ans.get("confidence", probs[choice]), field_name="confidence")
            conf = float(min(1.0, max(0.0, conf)))
            lifecycle_values[c.internal_name] = choice
            soft_targets[c.internal_name] = {"type": "choice", "probs": probs, "choice": choice}
            normalized_external[c.external_name] = {
                "type": "choice",
                "choice": choice,
                "value": choice,
                "confidence": conf,
                "probabilities": probs,
                "conformal_set": ans.get("conformal_set"),
            }

        elif c.q_type == "noul":
            if "noul" in ans and ans["noul"] is not None:
                p_true = parse_finite_float(ans["noul"], field_name="noul")
                p_true = float(min(1.0, max(0.0, p_true)))
                val_bool = (
                    bool(ans["value"])
                    if "value" in ans and isinstance(ans["value"], bool)
                    else (p_true >= c.threshold)
                )
            elif "value" in ans:
                if isinstance(ans["value"], bool):
                    val_bool = ans["value"]
                    p_true = 1.0 if val_bool else 0.0
                else:
                    p_true = float(min(1.0, max(0.0, parse_finite_float(ans["value"], field_name="value"))))
                    val_bool = p_true >= c.threshold
            else:
                raise InteropError(
                    f"Upstream noul answer missing 'noul'/'value' for {c.external_name!r}",
                    status_code=502,
                )
            conf = parse_finite_float(ans.get("confidence", max(p_true, 1.0 - p_true)), field_name="confidence")
            conf = float(min(1.0, max(0.0, conf)))
            lifecycle_values[c.internal_name] = val_bool
            soft_targets[c.internal_name] = {"type": "noul", "p_true": p_true, "value": val_bool}
            normalized_external[c.external_name] = {
                "type": "noul",
                "noul": p_true,
                "value": val_bool,
                "confidence": conf,
                "conformal_set": ans.get("conformal_set"),
            }

        elif c.q_type == "score":
            if c.criteria_legend:
                max_idx = len(c.options) - 1
                raw_probs = ans.get("probabilities")
                has_explicit_score = ("score" in ans and ans["score"] is not None) or (
                    "value" in ans and ans["value"] is not None
                )
                if isinstance(raw_probs, Mapping) and raw_probs:
                    probs = coerce_probability_dict(raw_probs, c.options)
                    exp_score = float(math.fsum(int(k) * probs[k] for k in c.options))
                    if has_explicit_score:
                        raw_s_val = ans["score"] if ("score" in ans and ans["score"] is not None) else ans["value"]
                        explicit_s = parse_finite_float(raw_s_val, field_name="score")
                        if explicit_s < 0.0 or explicit_s > float(max_idx):
                            raise InteropError(
                                f"Ordinal score {explicit_s!r} out of bounds [0, {max_idx}] for {c.external_name!r}",
                                status_code=502,
                            )
                        if abs(explicit_s - exp_score) > 1e-3:
                            raise InteropError(
                                f"Ordinal score {explicit_s!r} mismatches probability expectation {exp_score:.6f} for {c.external_name!r}",
                                status_code=502,
                            )
                    ord_idx = str(max(c.options, key=lambda opt: probs[opt]))
                else:
                    if not has_explicit_score:
                        raise InteropError(
                            f"Upstream score answer missing 'score'/'value' for {c.external_name!r}",
                            status_code=502,
                        )
                    raw_s_val = ans["score"] if ("score" in ans and ans["score"] is not None) else ans["value"]
                    raw_s = parse_finite_float(raw_s_val, field_name="score")
                    if raw_s < 0.0 or raw_s > float(max_idx):
                        raise InteropError(
                            f"Ordinal score {raw_s!r} out of bounds [0, {max_idx}] for {c.external_name!r}",
                            status_code=502,
                        )
                    if max_idx == 0 or raw_s >= float(max_idx):
                        probs = {str(i): (1.0 if i == max_idx else 0.0) for i in range(max_idx + 1)}
                    else:
                        lo = int(math.floor(raw_s))
                        hi = lo + 1
                        frac = float(raw_s - float(lo))
                        w_lo = float(1.0 - frac)
                        w_hi = float(1.0 - w_lo)
                        probs = {
                            str(i): (w_lo if i == lo else (w_hi if i == hi else 0.0))
                            for i in range(max_idx + 1)
                        }
                    exp_score = float(math.fsum(int(k) * probs[k] for k in c.options))
                    ord_idx = str(max(c.options, key=lambda opt: (probs[opt], -abs(int(opt) - exp_score))))
                conf = parse_finite_float(ans.get("confidence", probs[ord_idx]), field_name="confidence")
                conf = float(min(1.0, max(0.0, conf)))
                legend = {str(i): txt for i, txt in enumerate(c.criteria_legend)}
                lifecycle_values[c.internal_name] = ord_idx
                soft_targets[c.internal_name] = {"type": "score_ordinal", "probs": probs, "score": exp_score}
                normalized_external[c.external_name] = {
                    "type": "score",
                    "score": exp_score,
                    "value": exp_score,
                    "confidence": conf,
                    "probabilities": probs,
                    "legend": legend,
                    "conformal_set": ans.get("conformal_set"),
                }
            else:
                raw_s = parse_finite_float(ans.get("score", ans.get("value")), field_name="score")
                clipped = float(min(c.max_value, max(c.min_value, raw_s)))
                conf = parse_finite_float(ans.get("confidence", 1.0), field_name="confidence")
                conf = float(min(1.0, max(0.0, conf)))
                lifecycle_values[c.internal_name] = clipped
                soft_targets[c.internal_name] = {"type": "score_continuous", "score": clipped}
                normalized_external[c.external_name] = {
                    "type": "score",
                    "score": clipped,
                    "value": clipped,
                    "confidence": conf,
                    "conformal_set": ans.get("conformal_set"),
                }

        elif c.q_type == "multichoice":
            vals = ans.get("choices", ans.get("value", ()))
            if vals is None:
                vals = ()
            elif not isinstance(vals, (list, tuple)):
                raise InteropError(
                    f"Upstream multichoice 'choices'/'value' must be a sequence for {c.external_name!r}",
                    status_code=502,
                )
            selected: List[str] = []
            seen_mc: set[str] = set()
            for v in vals:
                if isinstance(v, bool) or str(v) not in c.options:
                    raise InteropError(
                        f"Invalid upstream multichoice option {v!r} for {c.external_name!r}",
                        status_code=502,
                    )
                sv = str(v)
                if sv not in seen_mc:
                    seen_mc.add(sv)
                    selected.append(sv)
            conf = parse_finite_float(ans.get("confidence", 1.0), field_name="confidence")
            conf = float(min(1.0, max(0.0, conf)))
            lifecycle_values[c.internal_name] = selected
            soft_targets[c.internal_name] = {"type": "multichoice", "choices": selected}
            normalized_external[c.external_name] = {
                "type": "multichoice",
                "choices": selected,
                "value": selected,
                "confidence": conf,
            }

    return lifecycle_values, soft_targets, normalized_external


class SoftDistillingLifecycle:
    """Wraps System 1 AutoCutoverLifecycle with soft-probability Ridge distillation
    and calibrated multi-primitive local output extraction.
    """

    def __init__(
        self,
        contract: CompiledContractSchema,
        *,
        distill_mode: str = "soft",
        soft_blend: float = 1.0,
        config: Optional[CutoverConfig] = None,
    ) -> None:
        if distill_mode not in ("soft", "hard"):
            raise ValueError(f"distill_mode must be 'soft' or 'hard', got {distill_mode!r}")
        self.contract = contract
        self.distill_mode = distill_mode
        self.soft_blend = float(min(1.0, max(0.0, soft_blend)))
        self.config = config or CutoverConfig(
            milestones=(48, 88, 136, 192, 264, 352, 464),
            k_min=5,
            alpha_target=0.05,
            canary_rate=0.05,
            sparse_dim=2048,
            dense_dim=192,
            enable_quarantine=False,
        )
        self.lifecycle = AutoCutoverLifecycle(
            copy.deepcopy(contract.schema),
            config=self.config,
            post_compile_hook=self._distill_sub_engine_before_qualification,
        )
        self._soft_records_by_prompt: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._structure_counts: Dict[str, int] = {}
        self._structure_field_counts: Dict[Tuple[str, str, Any], int] = {}
        self._distilled_compile_count: int = 0
        self._soft_distill_applied_count: int = 0
        self._frozen_eligible_counter: int = 0
        self._lock = threading.RLock()

    @property
    def state(self) -> str:
        return self.lifecycle.state

    @property
    def is_cutover(self) -> bool:
        return (
            self.lifecycle.state == CutoverState.SELECTIVE_HOT_SWAP
            and self.lifecycle.engine is not None
            and not self.lifecycle.pending_recalibrate
        )

    def _distill_sub_engine_before_qualification(
        self,
        target_engine: Any,
        train_pairs: Sequence[Tuple[str, Dict[str, Any]]],
    ) -> bool:
        """Distill recorded teacher soft-probability distributions before calibration and qualification."""
        with self._lock:
            if (
                target_engine is None
                or self.distill_mode != "soft"
                or self.soft_blend <= 0.0
                or not train_pairs
            ):
                return False

            self._distilled_compile_count += 1
            train_prompts = [p for p, _ in train_pairs]
            train_vals = [vals for _, vals in train_pairs]
            model = target_engine.model
            with np.errstate(over="ignore", invalid="ignore"):
                X_all = np.stack([model.encode(p) for p in train_prompts], axis=0).astype(np.float32)
            if not bool(np.all(np.isfinite(X_all))):
                return False

            from system1.compiler import SystemOneCompiler

            helper_compiler = SystemOneCompiler(
                self.contract.schema,
                dimension=model.dimension,
                regularization=float(model.metadata.get("regularization", 0.1)),
            )

            updated_any = False
            field_heads = getattr(model, "_field_heads", {})
            for c in self.contract.contracts:
                compiled_head = model.heads.get(c.internal_name)
                live_head = field_heads.get(c.internal_name)
                if compiled_head is None and live_head is None:
                    continue
                ref_head = live_head if live_head is not None else compiled_head
                assert ref_head is not None

                idx_keep = [i for i, vals in enumerate(train_vals) if c.internal_name in vals]
                if not idx_keep:
                    continue
                if len(idx_keep) == len(train_prompts):
                    X_train = X_all
                    sub_prompts = train_prompts
                    sub_vals = train_vals
                else:
                    X_train = X_all[idx_keep]
                    sub_prompts = [train_prompts[i] for i in idx_keep]
                    sub_vals = [train_vals[i] for i in idx_keep]

                if c.q_type in ("choice", "score") and (c.q_type == "choice" or c.criteria_legend):
                    head_options = tuple(getattr(ref_head.field_def, "options", c.options))
                    K = len(head_options)
                    if ref_head.weights.shape[0] != K or ref_head.weights.shape[1] != X_train.shape[1]:
                        continue
                    opt_idx = {opt: i for i, opt in enumerate(head_options)}
                    Y_hard = np.zeros((len(sub_prompts), K), dtype=np.float32)
                    Y_soft = np.zeros((len(sub_prompts), K), dtype=np.float32)
                    has_soft = False
                    for i, (prompt, vals) in enumerate(zip(sub_prompts, sub_vals)):
                        hard_label = str(vals[c.internal_name])
                        y_hard = np.zeros(K, dtype=np.float32)
                        if hard_label in opt_idx:
                            y_hard[opt_idx[hard_label]] = 1.0
                        Y_hard[i] = y_hard
                        soft_entry = self._soft_records_by_prompt.get(prompt, {}).get(c.internal_name)
                        if soft_entry and "probs" in soft_entry and isinstance(soft_entry["probs"], Mapping):
                            with np.errstate(over="ignore", invalid="ignore"):
                                p_vec = np.array(
                                    [float(soft_entry["probs"].get(opt, 0.0)) for opt in head_options],
                                    dtype=np.float32,
                                )
                                s = float(np.sum(p_vec))
                            if bool(np.all(np.isfinite(p_vec))) and math.isfinite(s) and s > 0.0:
                                with np.errstate(over="ignore", invalid="ignore"):
                                    p_vec = p_vec / s
                                if bool(np.all(np.isfinite(p_vec))):
                                    if float(np.max(np.abs(p_vec - y_hard))) > 1e-6:
                                        has_soft = True
                                    Y_soft[i] = p_vec
                                    continue
                        Y_soft[i] = y_hard

                    if has_soft and not getattr(ref_head, "hierarchical", False):
                        eff_reg, sw_choice = helper_compiler._compute_loo_ridge_and_sample_weights(
                            X_train,
                            Y_hard,
                            auto_regularization=True,
                            noise_robust=True,
                        )
                        delta_Y = (self.soft_blend * (Y_soft - Y_hard)).astype(np.float32)
                        dw, db = helper_compiler._solve_ridge(
                            X_train,
                            delta_Y,
                            eff_reg,
                            sample_weights=sw_choice,
                            regularize_bias=False,
                            return_covariance=False,
                        )
                        new_w = (ref_head.weights + dw).astype(np.float32)
                        new_b = (ref_head.biases + db).astype(np.float32)
                        if live_head is not None:
                            live_head.weights = new_w
                            live_head.biases = new_b
                        if compiled_head is not None:
                            compiled_head.weights = new_w
                            compiled_head.biases = new_b
                        updated_any = True

                elif c.q_type == "noul":
                    Y_hard = np.zeros((len(sub_prompts), 1), dtype=np.float32)
                    Y_soft = np.zeros((len(sub_prompts), 1), dtype=np.float32)
                    has_soft = False
                    for i, (prompt, vals) in enumerate(zip(sub_prompts, sub_vals)):
                        y_hard = 1.0 if bool(vals[c.internal_name]) else -1.0
                        Y_hard[i, 0] = y_hard
                        soft_entry = self._soft_records_by_prompt.get(prompt, {}).get(c.internal_name)
                        if soft_entry and "p_true" in soft_entry:
                            p_true = float(soft_entry["p_true"])
                            if math.isfinite(p_true) and 0.0 <= p_true <= 1.0:
                                y_soft = 2.0 * p_true - 1.0
                                if abs(y_soft - y_hard) > 1e-6:
                                    has_soft = True
                                Y_soft[i, 0] = y_soft
                                continue
                        Y_soft[i, 0] = y_hard
                    if has_soft:
                        Y_bool_oh = np.column_stack(
                            [(Y_hard[:, 0] < 0).astype(np.float32), (Y_hard[:, 0] > 0).astype(np.float32)]
                        )
                        eff_reg_b, sw_b = helper_compiler._compute_loo_ridge_and_sample_weights(
                            X_train,
                            Y_bool_oh,
                            auto_regularization=True,
                            noise_robust=True,
                        )
                        delta_Y = (self.soft_blend * (Y_soft - Y_hard)).astype(np.float32)
                        dw, db = helper_compiler._solve_ridge(
                            X_train,
                            delta_Y,
                            eff_reg_b,
                            sample_weights=sw_b,
                            regularize_bias=True,
                            return_covariance=False,
                        )
                        new_w = (ref_head.weights + dw).astype(np.float32)
                        new_b = (ref_head.biases + db).astype(np.float32)
                        if live_head is not None:
                            live_head.weights = new_w
                            live_head.biases = new_b
                        if compiled_head is not None:
                            compiled_head.weights = new_w
                            compiled_head.biases = new_b
                        updated_any = True

            if updated_any:
                self._soft_distill_applied_count += 1
            return updated_any

    def _canonical_discrete_field_val(self, f_def: Any, val: Any) -> Tuple[bool, Any]:
        if isinstance(f_def, BooleanField):
            return True, bool(val)
        if isinstance(f_def, ChoiceField):
            return True, str(val)
        if isinstance(f_def, MultiChoiceField):
            if isinstance(val, (list, tuple)):
                return True, tuple(str(x) for x in val)
            return True, (str(val),)
        return False, None

    def _has_structure_support_locked(
        self,
        structure_sig: Optional[str],
        vals: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        if structure_sig is None:
            return True
        req_min = max(2, int(self.lifecycle.config.k_min))
        if self._structure_counts.get(structure_sig, 0) < req_min:
            return False
        if vals is not None:
            for fname, f_def in self.lifecycle.schema.fields.items():
                if fname not in vals:
                    return False
                is_disc, canon_v = self._canonical_discrete_field_val(f_def, vals[fname])
                if is_disc and self._structure_field_counts.get((structure_sig, fname, canon_v), 0) < req_min:
                    return False
        return True

    _supports_canary_tx_arg: bool = True

    def observe_teacher(
        self,
        prompt: str,
        lifecycle_values: Dict[str, Any],
        soft_targets: Optional[Dict[str, Dict[str, Any]]] = None,
        *,
        cache_key: Optional[str] = None,
        structure_sig: Optional[str] = None,
        _canary_tx: Optional[Any] = None,
    ) -> bool:
        tls_map = getattr(_ACTIVE_CANARY_TX_TLS, "bindings", None)
        has_bound = isinstance(tls_map, dict) and id(self) in tls_map
        eff_canary_tx = tls_map[id(self)] if has_bound else _canary_tx
        with self._lock:
            had_soft = prompt in self._soft_records_by_prompt
            prev_soft = self._soft_records_by_prompt.get(prompt)
            prev_distilled_cnt = self._distilled_compile_count
            prev_applied_cnt = self._soft_distill_applied_count
            had_sc = False
            prev_sc: Optional[int] = None
            inc_fks: List[Tuple[str, str, str]] = []
            try:
                if soft_targets:
                    self._soft_records_by_prompt[prompt] = copy.deepcopy(soft_targets)
                if structure_sig is not None:
                    had_sc = structure_sig in self._structure_counts
                    prev_sc = self._structure_counts.get(structure_sig)
                    self._structure_counts[structure_sig] = self._structure_counts.get(structure_sig, 0) + 1
                    for fname, f_def in self.lifecycle.schema.fields.items():
                        if fname in lifecycle_values:
                            is_disc, canon_v = self._canonical_discrete_field_val(f_def, lifecycle_values[fname])
                            if is_disc:
                                fk = (structure_sig, fname, canon_v)
                                self._structure_field_counts[fk] = self._structure_field_counts.get(fk, 0) + 1
                                inc_fks.append(fk)
                if has_bound or eff_canary_tx is not None:
                    inner_tx = (
                        eff_canary_tx[0]
                        if isinstance(eff_canary_tx, tuple) and len(eff_canary_tx) >= 1
                        else None
                    )
                    with _bind_active_canary_tx(
                        self.lifecycle,
                        tx_token=(inner_tx if inner_tx is not None else (0, 0, 0, "", None)),
                    ):
                        return self.lifecycle.observe_teacher(
                            prompt,
                            lifecycle_values,
                            cache_key=cache_key,
                        )
                return self.lifecycle.observe_teacher(prompt, lifecycle_values, cache_key=cache_key)
            except BaseException:
                self._distilled_compile_count = prev_distilled_cnt
                self._soft_distill_applied_count = prev_applied_cnt
                if soft_targets:
                    if had_soft:
                        self._soft_records_by_prompt[prompt] = prev_soft  # type: ignore[assignment]
                    else:
                        self._soft_records_by_prompt.pop(prompt, None)
                if structure_sig is not None:
                    if had_sc:
                        self._structure_counts[structure_sig] = prev_sc  # type: ignore[assignment]
                    else:
                        self._structure_counts.pop(structure_sig, None)
                    for fk in inc_fks:
                        cur_c = self._structure_field_counts.get(fk, 0) - 1
                        if cur_c <= 0:
                            self._structure_field_counts.pop(fk, None)
                        else:
                            self._structure_field_counts[fk] = cur_c
                if not has_bound and eff_canary_tx is None and hasattr(self.lifecycle, "_clear_untracked_canary"):
                    self.lifecycle._clear_untracked_canary(prompt, cache_key=cache_key)
                raise

    def _evaluate_frozen_local_locked(
        self,
        prompt: str,
        *,
        cache_key: Optional[str] = None,
        structure_sig: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Any], bool]:
        """Read-only fast-path check mirroring AutoCutoverLifecycle without mutating state."""
        lc = self.lifecycle
        engine = lc.engine
        if (
            not lc.compiled_active
            or lc.active_tau is None
            or engine is None
            or lc.pending_recalibrate
            or lc.state == CutoverState.DRIFT_RECALIBRATE.value
        ):
            return None, None, False
        canary_period = max(1, int(round(1.0 / lc.config.canary_rate)))
        key = cache_key if (isinstance(cache_key, str) and cache_key) else _normalize_cache_key(prompt)
        if key in lc.clean_cache:
            self._frozen_eligible_counter += 1
            if ((lc.local_accept_eligible + self._frozen_eligible_counter) % canary_period) == 0:
                return None, None, False
            cached_vals = dict(lc.clean_cache[key])
            return cached_vals, None, True
        if key in lc._conflicted_cache_keys:
            return None, None, False
        if not self._has_structure_support_locked(structure_sig):
            return None, None, False

        res = engine.decide(prompt, strict=False, record_receipt=False, use_cache=False)
        vals = dict(res.values)
        pred_key = lc._primary_label_key(vals)
        margin = lc._decision_min_margin(res, engine, res.values)
        if (
            res.abstention_reason is None
            and pred_key in lc.mature_classes
            and margin >= lc.active_tau
            and lc.aci.should_accept_margin(margin)
        ):
            cent = lc.class_centroids.get(pred_key)
            floor = lc.class_sim_floors.get(pred_key, 0.30)
            if cent is None or float(np.dot(lc.subword.encode(prompt), cent)) >= max(
                lc.config.cls_min_abs, floor * 0.90
            ):
                if not self._has_structure_support_locked(structure_sig, vals):
                    return None, None, False
                self._frozen_eligible_counter += 1
                if ((lc.local_accept_eligible + self._frozen_eligible_counter) % canary_period) == 0:
                    return None, None, False
                return vals, res, False
        return None, None, False

    def _format_local_answers(
        self,
        local_vals: Dict[str, Any],
        res: Any,
        *,
        from_cache: bool = False,
    ) -> Dict[str, Any]:
        """Build external wire answers dict from local System 1 decision outputs."""
        if not from_cache and res is None:
            raise InteropError("Missing student decision result", status_code=422)
        answers_out: Dict[str, Any] = {}
        for c in self.contract.contracts:
            iname = c.internal_name
            if from_cache:
                if iname not in local_vals:
                    raise InteropError(f"Missing cached value for field {iname!r}", status_code=422)
                raw_cached = local_vals[iname]
                if c.q_type == "choice":
                    if not isinstance(raw_cached, str) or raw_cached not in c.options:
                        raise InteropError(f"Invalid cached choice value for {iname!r}: {raw_cached!r}", status_code=422)
                    choice = raw_cached
                    probs = {o: (1.0 if o == choice else 0.0) for o in c.options}
                    answers_out[c.external_name] = {
                        "type": "choice",
                        "choice": choice,
                        "value": choice,
                        "confidence": 1.0,
                        "probabilities": probs,
                        "conformal_set": [choice],
                    }
                elif c.q_type == "noul":
                    if not isinstance(raw_cached, (bool, np.bool_)):
                        raise InteropError(f"Invalid cached noul value for {iname!r}: {raw_cached!r}", status_code=422)
                    val_b = bool(raw_cached)
                    p_true = 1.0 if val_b else 0.0
                    answers_out[c.external_name] = {
                        "type": "noul",
                        "noul": p_true,
                        "value": val_b,
                        "confidence": 1.0,
                        "conformal_set": [val_b],
                    }
                elif c.q_type == "score":
                    if c.criteria_legend:
                        lbl = raw_cached if isinstance(raw_cached, str) else str(raw_cached)
                        if isinstance(raw_cached, (bool, np.bool_)) or lbl not in c.options:
                            raise InteropError(f"Invalid cached score option for {iname!r}: {raw_cached!r}", status_code=422)
                        probs = {o: (1.0 if o == lbl else 0.0) for o in c.options}
                        exp_score = float(int(lbl))
                        legend = {str(i): txt for i, txt in enumerate(c.criteria_legend)}
                        answers_out[c.external_name] = {
                            "type": "score",
                            "score": exp_score,
                            "value": exp_score,
                            "confidence": 1.0,
                            "probabilities": probs,
                            "legend": legend,
                            "conformal_set": [lbl],
                        }
                    else:
                        val_s = parse_finite_float(raw_cached, field_name=f"{iname}.score")
                        answers_out[c.external_name] = {
                            "type": "score",
                            "score": val_s,
                            "value": val_s,
                            "confidence": 1.0,
                            "conformal_set": [],
                        }
                elif c.q_type == "multichoice":
                    if not isinstance(raw_cached, (list, tuple)) or not all(
                        isinstance(ch, str) and ch in c.options for ch in raw_cached
                    ):
                        raise InteropError(f"Invalid cached multichoice value for {iname!r}: {raw_cached!r}", status_code=422)
                    chosen = [str(ch) for ch in raw_cached]
                    answers_out[c.external_name] = {
                        "type": "multichoice",
                        "choices": chosen,
                        "value": chosen,
                        "confidence": 1.0,
                    }
                continue

            if c.q_type == "choice":
                probs_raw = res.probabilities.get(iname) or {}
                raw_choice = local_vals.get(iname, res.values.get(iname))
                if not isinstance(raw_choice, str) or raw_choice not in c.options:
                    raise InteropError(f"Invalid choice value for {iname!r}: {raw_choice!r}", status_code=422)
                choice = raw_choice
                probs = coerce_probability_dict(
                    probs_raw
                    if probs_raw
                    else {o: (1.0 if o == choice else 0.0) for o in c.options},
                    c.options,
                )
                conf = _parse_unit_interval_float(
                    res.confidences.get(iname, probs.get(choice, 1.0)),
                    field_name=f"{iname}.confidence",
                )
                cset = res.conformal_sets.get(iname)
                answers_out[c.external_name] = {
                    "type": "choice",
                    "choice": choice,
                    "value": choice,
                    "confidence": conf,
                    "probabilities": probs,
                    "conformal_set": list(cset) if cset else [choice],
                }
            elif c.q_type == "noul":
                probs_raw = res.probabilities.get(iname) or {}
                if probs_raw:
                    if not isinstance(probs_raw, Mapping):
                        raise InteropError(f"Invalid noul probabilities for {iname!r}", status_code=422)
                    if "False" in probs_raw or len(probs_raw) > 1:
                        p_true = coerce_probability_dict(probs_raw, ("False", "True"))["True"]
                    elif "True" in probs_raw:
                        p_true = _parse_unit_interval_float(
                            probs_raw["True"],
                            field_name=f"{iname}.probabilities['True']",
                        )
                    else:
                        raise InteropError(f"Noul probabilities missing 'True'/'False' for {iname!r}", status_code=422)
                else:
                    raw_vb = local_vals.get(iname, res.values.get(iname))
                    if not isinstance(raw_vb, (bool, np.bool_)):
                        raise InteropError(f"Invalid noul value for {iname!r}: {raw_vb!r}", status_code=422)
                    val_b = bool(raw_vb)
                    c_val = _parse_unit_interval_float(
                        res.confidences.get(iname, 0.9),
                        field_name=f"{iname}.confidence",
                    )
                    p_true = float(c_val if val_b else (1.0 - c_val))
                val_b = bool(p_true >= c.threshold)
                conf = float(max(p_true, 1.0 - p_true))
                cset = res.conformal_sets.get(iname)
                answers_out[c.external_name] = {
                    "type": "noul",
                    "noul": p_true,
                    "value": val_b,
                    "confidence": conf,
                    "conformal_set": list(cset) if cset else [val_b],
                }
            elif c.q_type == "score":
                if c.criteria_legend:
                    probs_raw = res.probabilities.get(iname) or {}
                    raw_fallback = local_vals.get(iname, res.values.get(iname))
                    fallback_str = str(raw_fallback) if raw_fallback is not None else ""
                    if not probs_raw and (
                        isinstance(raw_fallback, (bool, np.bool_)) or fallback_str not in c.options
                    ):
                        raise InteropError(
                            f"Invalid ordinal score option for {iname!r}: {raw_fallback!r}",
                            status_code=422,
                        )
                    probs = coerce_probability_dict(
                        probs_raw
                        if probs_raw
                        else {o: (1.0 if o == fallback_str else 0.0) for o in c.options},
                        c.options,
                    )
                    exp_score = parse_finite_float(
                        sum(int(k) * probs[k] for k in c.options),
                        field_name=f"{iname}.score",
                    )
                    conf = _parse_unit_interval_float(
                        max(probs.values()),
                        field_name=f"{iname}.confidence",
                    )
                    legend = {str(i): txt for i, txt in enumerate(c.criteria_legend)}
                    raw_cset = res.conformal_sets.get(iname)
                    cset_fallback = fallback_str if fallback_str in c.options else max(c.options, key=lambda o: probs[o])
                    answers_out[c.external_name] = {
                        "type": "score",
                        "score": exp_score,
                        "value": exp_score,
                        "confidence": conf,
                        "probabilities": probs,
                        "legend": legend,
                        "conformal_set": list(raw_cset) if raw_cset else [cset_fallback],
                    }
                else:
                    raw_s = local_vals.get(iname, res.values.get(iname))
                    val_s = parse_finite_float(raw_s, field_name=f"{iname}.score")
                    conf = _parse_unit_interval_float(
                        res.confidences.get(iname, 1.0),
                        field_name=f"{iname}.confidence",
                    )
                    answers_out[c.external_name] = {
                        "type": "score",
                        "score": val_s,
                        "value": val_s,
                        "confidence": conf,
                        "conformal_set": list(res.conformal_sets.get(iname) or ()),
                    }
            elif c.q_type == "multichoice":
                raw_mc = local_vals.get(iname, res.values.get(iname))
                if not isinstance(raw_mc, (list, tuple)) or not all(
                    isinstance(ch, str) and ch in c.options for ch in raw_mc
                ):
                    raise InteropError(f"Invalid multichoice value for {iname!r}: {raw_mc!r}", status_code=422)
                chosen = [str(ch) for ch in raw_mc]
                conf = _parse_unit_interval_float(
                    res.confidences.get(iname, 1.0),
                    field_name=f"{iname}.confidence",
                )
                answers_out[c.external_name] = {
                    "type": "multichoice",
                    "choices": chosen,
                    "value": chosen,
                    "confidence": conf,
                }
        return answers_out

    def _decide_local_with_tx(
        self,
        prompt: str,
        *,
        freeze: bool = False,
        cache_key: Optional[str] = None,
        structure_sig: Optional[str] = None,
        _track_tx: bool = True,
    ) -> Tuple[Optional[Dict[str, Any]], Tuple[Optional[Any], int]]:
        """Attempt autonomous local execution and return (answers_or_none, rollback_tx_token)."""
        with self._lock:
            if not self.is_cutover:
                return None, (None, 0)
            eff_key = (
                cache_key
                if (isinstance(cache_key, str) and cache_key)
                else _normalize_cache_key(prompt)
            )
            if freeze:
                f0 = self._frozen_eligible_counter
                local_vals, res, from_cache = self._evaluate_frozen_local_locked(
                    prompt,
                    cache_key=eff_key,
                    structure_sig=structure_sig,
                )
                df = self._frozen_eligible_counter - f0
                if local_vals is None or (not from_cache and res is None):
                    return None, (None, df)
                try:
                    return self._format_local_answers(local_vals, res, from_cache=from_cache), (None, df)
                except InteropError:
                    if df:
                        self._frozen_eligible_counter = max(0, self._frozen_eligible_counter - df)
                    return None, (None, 0)

            if (
                eff_key not in self.lifecycle.clean_cache
                and not self._has_structure_support_locked(structure_sig)
            ):
                return None, (None, 0)

            local_vals, inner_tx = self.lifecycle._decide_local_with_tx(
                prompt,
                cache_key=eff_key,
                accept_predicate=(
                    (lambda cand_vals: self._has_structure_support_locked(structure_sig, cand_vals))
                    if structure_sig is not None
                    else None
                ),
                _track_tx=_track_tx,
            )
            if local_vals is None:
                return None, (inner_tx, 0)
            from_cache = eff_key in self.lifecycle.clean_cache
            if from_cache:
                res = None
            else:
                engine = self.lifecycle.engine
                if engine is None:
                    self.lifecycle._rollback_local_tx(inner_tx)
                    return None, (None, 0)
                try:
                    res = engine.decide(prompt, record_receipt=False)
                except BaseException:
                    self.lifecycle._rollback_local_tx(inner_tx)
                    raise
            try:
                return self._format_local_answers(local_vals, res, from_cache=from_cache), (inner_tx, 0)
            except InteropError:
                self.lifecycle._rollback_local_tx(inner_tx)
                return None, (None, 0)

    def _rollback_local_tx(self, tx: Tuple[Optional[Any], int]) -> None:
        inner_tx, df = tx
        if inner_tx is not None:
            self.lifecycle._rollback_local_tx(inner_tx)
        if df:
            with self._lock:
                self._frozen_eligible_counter = max(0, self._frozen_eligible_counter - df)

    def decide_local(
        self,
        prompt: str,
        *,
        freeze: bool = False,
        cache_key: Optional[str] = None,
        structure_sig: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Attempt autonomous local execution when in SELECTIVE_HOT_SWAP."""
        answers, _ = self._decide_local_with_tx(
            prompt,
            freeze=freeze,
            cache_key=cache_key,
            structure_sig=structure_sig,
            _track_tx=False,
        )
        return answers


def format_provider_response(
    *,
    wire_format: str,
    model_requested: str,
    answers: Dict[str, Any],
    input_tokens: int,
    local_execution: bool,
    schema_fingerprint: str,
    drift_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build wire-compatible response payload for Gemini Decision 1, Laya, or Jev."""
    if wire_format == "gemini":
        out_model = model_requested or "gemini-decision-1"
    elif wire_format == "laya":
        out_model = model_requested or "laya"
    else:
        out_model = "system1-local" if local_execution else (model_requested or "jev-1.13.0")

    payload: Dict[str, Any] = {
        "model": out_model,
        "answers": answers,
        "decisions": answers,
        "local_execution": bool(local_execution),
        "provider_format": wire_format,
        "schema_fingerprint": schema_fingerprint,
        "usage": {
            "input_tokens": int(input_tokens),
            "output_tokens": 0,
            "total_tokens": int(input_tokens),
        },
    }
    if drift_info and drift_info.get("drifted"):
        payload["schema_drift_detected"] = True
        payload["schema_drift"] = drift_info
    else:
        payload["schema_drift_detected"] = False
    return payload


def _canonicalize_origin(origin: str) -> Optional[Tuple[str, str, int]]:
    """Parse an HTTP(S) Origin into normalized ``(scheme, hostname, port)`` or ``None``."""
    if not isinstance(origin, str):
        return None
    s = origin.strip()
    if not s or any(ch.isspace() for ch in s):
        return None
    try:
        parsed = urllib.parse.urlsplit(s)
        raw_port = parsed.port
    except ValueError:
        return None

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        return None
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        return None
    if parsed.query or parsed.fragment or parsed.path not in ("", "/"):
        return None
    if not parsed.hostname:
        return None

    netloc = parsed.netloc
    if netloc.startswith("["):
        end_bracket = netloc.find("]")
        if end_bracket < 0:
            return None
        tail = netloc[end_bracket + 1 :]
        if tail and (not tail.startswith(":") or not tail[1:].isdigit()):
            return None
    elif ":" in netloc:
        _, port_part = netloc.split(":", 1)
        if not port_part.isdigit():
            return None

    eff_port = int(raw_port) if raw_port is not None else (443 if scheme == "https" else 80)
    if not (1 <= eff_port <= 65535):
        return None
    return (scheme, parsed.hostname.lower(), eff_port)


class UniversalDecisionService:
    """Thread-safe multi-provider gateway engine with shadow-distill-cutover and drift isolation."""

    def __init__(
        self,
        *,
        upstream_handlers: Optional[
            Mapping[str, Callable[[str, Mapping[str, Any], CompiledContractSchema], Mapping[str, Any]]]
        ] = None,
        upstream_urls: Optional[Mapping[str, str]] = None,
        bearer_token: Optional[str] = None,
        require_auth: bool = True,
        allowed_origins: Optional[Sequence[str]] = (
            "http://127.0.0.1",
            "http://localhost",
            "https://app.system1.local",
        ),
        max_body_bytes: int = 262_144,
        distill_mode: str = "soft",
        soft_blend: float = 0.5,
        cutover_config: Optional[CutoverConfig] = None,
        freeze_learning: bool = False,
        max_unpromoted_lifecycles: int = 256,
    ) -> None:
        self.upstream_handlers = dict(upstream_handlers or {})
        self.upstream_urls = dict(upstream_urls or {})
        if not isinstance(require_auth, bool):
            raise ValueError(f"require_auth must be explicitly True or False, got {require_auth!r}")
        self.require_auth = require_auth
        if bearer_token is not None and not isinstance(bearer_token, str):
            raise TypeError(f"bearer_token must be a string or None, got {type(bearer_token).__name__}")
        if self.require_auth and (bearer_token is None or not bearer_token.strip()):
            raise ValueError(
                "UniversalDecisionService requires an explicit non-empty bearer_token when require_auth=True; "
                "pass require_auth=False to configure unauthenticated loopback operation"
            )
        self.bearer_token = bearer_token
        self.allowed_origins = set(allowed_origins or ())
        parsed_origins: Set[Tuple[str, str, int]] = set()
        for orig in self.allowed_origins:
            canon = _canonicalize_origin(orig)
            if canon is None:
                raise ValueError(f"Invalid allowed origin entry: {orig!r}")
            parsed_origins.add(canon)
        self._allowed_origin_tuples: FrozenSet[Tuple[str, str, int]] = frozenset(parsed_origins)
        self.max_body_bytes = int(max_body_bytes)
        self.distill_mode = distill_mode
        self.soft_blend = float(soft_blend)
        self.cutover_config = cutover_config
        self.freeze_learning = bool(freeze_learning)
        if isinstance(max_unpromoted_lifecycles, bool) or not isinstance(max_unpromoted_lifecycles, int) or max_unpromoted_lifecycles < 1:
            raise ValueError(
                f"max_unpromoted_lifecycles must be a positive int, got {max_unpromoted_lifecycles!r}"
            )
        self.max_unpromoted_lifecycles: int = int(max_unpromoted_lifecycles)
        self._lifecycles: OrderedDict[str, SoftDistillingLifecycle] = OrderedDict()
        self._last_schema_by_slot: OrderedDict[str, CompiledContractSchema] = OrderedDict()
        self._lc_inflight_obs: Dict[int, int] = defaultdict(int)
        self._lc_committed_ops: Dict[int, int] = defaultdict(int)
        self._lc_pending_new_bind: Dict[str, SoftDistillingLifecycle] = {}
        self._lock = threading.RLock()
        self.stats: Dict[str, int] = {
            "total_requests": 0,
            "local_fastpath_requests": 0,
            "upstream_teacher_calls": 0,
            "schema_drift_events": 0,
            "auth_failures": 0,
            "origin_failures": 0,
        }

    def set_freeze_learning(self, frozen: bool) -> None:
        """Enable or disable online learning/admittance (resets frozen canary counters cleanly)."""
        with self._lock:
            self.freeze_learning = bool(frozen)
            for lc in self._lifecycles.values():
                lc._frozen_eligible_counter = 0

    def check_headers(self, headers: Mapping[str, str]) -> None:
        lower = {str(k).lower(): str(v) for k, v in headers.items()}
        origin = lower.get("origin")
        if origin is not None and self._allowed_origin_tuples:
            canon_origin = _canonicalize_origin(origin)
            if canon_origin is None or canon_origin not in self._allowed_origin_tuples:
                with self._lock:
                    self.stats["origin_failures"] += 1
                raise ForbiddenOriginError(f"Disallowed Origin header: {origin!r}")

        if self.require_auth or self.bearer_token is not None:
            token = self.bearer_token or ""
            auth = lower.get("authorization", "")
            x_api_key = lower.get("x-goog-api-key", "") or lower.get("x-api-key", "")
            candidate = ""
            if auth.lower().startswith("bearer "):
                candidate = auth[7:].strip()
            elif x_api_key:
                candidate = x_api_key.strip()
            if not token or not candidate or not hmac.compare_digest(
                candidate.encode("utf-8"), token.encode("utf-8")
            ):
                with self._lock:
                    self.stats["auth_failures"] += 1
                raise AuthError("Invalid or missing Bearer / x-goog-api-key credential")

    def preflight_http_headers(self, headers: Mapping[str, str]) -> Optional[int]:
        """Validate auth/origin headers and declared Content-Length before streaming request body."""
        lower = {str(k).lower(): str(v) for k, v in headers.items()}
        self.check_headers(lower)
        raw_len = lower.get("content-length")
        if raw_len is None or not raw_len.strip():
            return None
        try:
            content_len = int(raw_len.strip())
        except ValueError as exc:
            raise InteropError(f"Invalid Content-Length header: {raw_len!r}", status_code=400) from exc
        if content_len < 0:
            raise InteropError(f"Negative Content-Length header: {content_len}", status_code=400)
        if content_len > self.max_body_bytes:
            raise PayloadTooLargeError(
                f"Request Content-Length {content_len} exceeds {self.max_body_bytes} bytes"
            )
        return content_len

    def _call_upstream(
        self,
        wire_format: str,
        raw_payload: Mapping[str, Any],
        semantic_text: str,
        contract: CompiledContractSchema,
    ) -> Mapping[str, Any]:
        with self._lock:
            self.stats["upstream_teacher_calls"] += 1

        if wire_format in self.upstream_handlers:
            return self.upstream_handlers[wire_format](semantic_text, raw_payload, contract)

        url = self.upstream_urls.get(wire_format) or self.upstream_urls.get("default")
        if not url:
            raise InteropError(
                f"No upstream teacher configured for wire format {wire_format!r}", status_code=502
            )

        req = urllib.request.Request(
            url,
            data=strict_json_dumps(dict(raw_payload)),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw_resp_bytes = resp.read()
        except ZeroEgressViolationError:
            raise
        except urllib.error.URLError as exc:
            raise InteropError(f"Upstream teacher HTTP error: {exc}", status_code=502) from exc
        try:
            return strict_json_loads(raw_resp_bytes)
        except InteropError as exc:
            raise InteropError(f"Invalid upstream teacher JSON: {exc}", status_code=502) from exc

    def _record_slot_schema_locked(
        self,
        slot_key: str,
        contract: CompiledContractSchema,
    ) -> Optional[Dict[str, Any]]:
        prev_contract = self._last_schema_by_slot.get(slot_key)
        drift_info: Optional[Dict[str, Any]] = None
        if prev_contract is not None and prev_contract.fingerprint != contract.fingerprint:
            drift_info = diff_schemas(prev_contract, contract)
            self.stats["schema_drift_events"] += 1
        self._last_schema_by_slot[slot_key] = contract
        self._last_schema_by_slot.move_to_end(slot_key)
        while len(self._last_schema_by_slot) > 1024:
            self._last_schema_by_slot.popitem(last=False)
        return drift_info

    @staticmethod
    def _extract_upstream_input_tokens(upstream_raw: Any, default_tokens: int) -> int:
        raw_usage = upstream_raw.get("usage") if isinstance(upstream_raw, Mapping) else None
        if raw_usage is None:
            return default_tokens
        if not isinstance(raw_usage, Mapping):
            raise InteropError("Upstream response 'usage' must be a JSON object", status_code=502)
        if "input_tokens" not in raw_usage or raw_usage["input_tokens"] is None:
            return default_tokens
        raw_tok = raw_usage["input_tokens"]
        if isinstance(raw_tok, (bool, np.bool_)):
            raise InteropError(
                "Upstream response 'usage.input_tokens' must be a non-negative integer",
                status_code=502,
            )
        if isinstance(raw_tok, (int, np.integer)):
            tok = int(raw_tok)
        elif isinstance(raw_tok, (float, np.floating)):
            f_tok = float(raw_tok)
            if not math.isfinite(f_tok) or not f_tok.is_integer():
                raise InteropError(
                    "Upstream response 'usage.input_tokens' must be a finite integer",
                    status_code=502,
                )
            tok = int(f_tok)
        else:
            raise InteropError(
                "Upstream response 'usage.input_tokens' must be a non-negative integer",
                status_code=502,
            )
        if tok < 0:
            raise InteropError(
                f"Upstream response 'usage.input_tokens' must be non-negative, got {tok}",
                status_code=502,
            )
        return tok

    def _prune_orphaned_lc_bookkeeping_locked(self) -> None:
        live_lc_ids = {id(v) for v in self._lifecycles.values()}
        inflight_lc_ids = {lc_id for lc_id, cnt in self._lc_inflight_obs.items() if cnt > 0}
        keep_committed = live_lc_ids | inflight_lc_ids
        for k_id in [k_id for k_id in self._lc_committed_ops if k_id not in keep_committed]:
            self._lc_committed_ops.pop(k_id, None)
        for pend_k, pend_lc in list(self._lc_pending_new_bind.items()):
            if self._lifecycles.get(pend_k) is not pend_lc and id(pend_lc) not in inflight_lc_ids:
                self._lc_pending_new_bind.pop(pend_k, None)

    def _enforce_unpromoted_lifecycle_cap_locked(self, *, reserve_for_new: bool = False) -> None:
        limit = max(0, self.max_unpromoted_lifecycles - (1 if reserve_for_new else 0))
        unpromoted_keys = [k for k, v in self._lifecycles.items() if not v.is_cutover]
        while len(unpromoted_keys) > limit:
            evict_key = unpromoted_keys.pop(0)
            self._lifecycles.pop(evict_key, None)
        self._prune_orphaned_lc_bookkeeping_locked()

    def _get_or_create_lifecycle(
        self,
        slot_key: str,
        contract: CompiledContractSchema,
    ) -> Tuple[SoftDistillingLifecycle, Optional[Dict[str, Any]]]:
        with self._lock:
            drift_info = self._record_slot_schema_locked(slot_key, contract)
            lc_key = f"{slot_key}:{contract.fingerprint}"
            if lc_key not in self._lifecycles:
                self._enforce_unpromoted_lifecycle_cap_locked(reserve_for_new=True)
                self._lifecycles[lc_key] = SoftDistillingLifecycle(
                    contract,
                    distill_mode=self.distill_mode,
                    soft_blend=self.soft_blend,
                    config=self.cutover_config,
                )
            else:
                self._lifecycles.move_to_end(lc_key)
                self._enforce_unpromoted_lifecycle_cap_locked(reserve_for_new=False)
            return self._lifecycles[lc_key], drift_info

    def handle_request(
        self,
        *,
        path: str,
        headers: Mapping[str, str],
        raw_body: bytes,
    ) -> Tuple[int, Dict[str, Any]]:
        if len(raw_body) > self.max_body_bytes:
            raise PayloadTooLargeError(f"Request body {len(raw_body)} exceeds {self.max_body_bytes} bytes")
        lower_h = {str(k).lower(): str(v) for k, v in headers.items()}
        self.check_headers(lower_h)
        ctype = lower_h.get("content-type", "")
        if not _is_json_media_type(ctype):
            raise UnsupportedMediaError(f"Expected Content-Type: application/json, got {ctype!r}")

        payload = strict_json_loads(raw_body)
        if not isinstance(payload, Mapping):
            raise InteropError("Request JSON root must be an object", status_code=400)
        freeze_hdr = lower_h.get("x-system1-freeze-learning")
        freeze_override = (
            freeze_hdr.strip().lower() in ("1", "true", "yes")
            if freeze_hdr is not None
            else None
        )
        return self.process_payload(
            path=path,
            payload=payload,
            slot_hint=lower_h.get("x-system1-slot"),
            freeze_learning=freeze_override,
        )

    def process_payload(
        self,
        *,
        path: str,
        payload: Mapping[str, Any],
        slot_hint: Optional[str] = None,
        freeze_learning: Optional[bool] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        if not isinstance(payload, Mapping):
            raise SchemaValidationError("Request payload must be a JSON object")
        _validate_payload_tree(payload)
        if slot_hint is not None:
            if not isinstance(slot_hint, str):
                raise InteropError("slot_hint must be a string", status_code=400)
            try:
                slot_hint.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise InteropError(
                    f"slot_hint contains invalid unpaired UTF-16 surrogate: {exc}",
                    status_code=400,
                ) from exc
        raw_model = payload.get("model")
        if raw_model is not None and not isinstance(raw_model, str):
            raise SchemaValidationError("Request 'model' must be a string")
        wire_format = detect_wire_format(path, payload)
        local_tx_ref: List[Any] = [None, None]
        try:
            return self._process_payload_tx(
                wire_format=wire_format,
                payload=payload,
                slot_hint=slot_hint,
                freeze_learning=freeze_learning,
                local_tx_ref=local_tx_ref,
            )
        except BaseException:
            tx_lc, tx_tok = local_tx_ref[0], local_tx_ref[1]
            if tx_tok is not None and tx_lc is not None and hasattr(tx_lc, "_rollback_local_tx"):
                tx_lc._rollback_local_tx(tx_tok)
            raise

    def _process_payload_tx(
        self,
        *,
        wire_format: str,
        payload: Mapping[str, Any],
        slot_hint: Optional[str],
        freeze_learning: Optional[bool],
        local_tx_ref: List[Any],
    ) -> Tuple[int, Dict[str, Any]]:
        semantic_text, _ = extract_semantic_text(payload)
        context_view = {
            k: v
            for k, v in payload.items()
            if k not in ("questions", "slot", "force_teacher", "model")
        }
        req_cache_key = _canonical_structured_key(context_view)
        primary_view = {
            k: v
            for k, v in context_view.items()
            if k in _PRIMARY_SEMANTIC_KEYS and v is not None
        }
        extra_view = {
            k: v
            for k, v in context_view.items()
            if k not in _PRIMARY_SEMANTIC_KEYS
        }
        req_struct_sig = (
            _input_structure_signature(primary_view)
            if not extra_view
            else f"{_input_structure_signature(primary_view)}|extra:{_canonical_structured_key(extra_view)}"
        )
        contract = compile_questions_contract(payload.get("questions"))
        slot_key = str(slot_hint or payload.get("slot") or wire_format)
        is_frozen = self.freeze_learning if freeze_learning is None else bool(freeze_learning)
        lc_key = f"{slot_key}:{contract.fingerprint}"
        est_input_tokens = max(1, len(semantic_text.split()))

        with self._lock:
            self.stats["total_requests"] += 1
            existing_lc = self._lifecycles.get(lc_key)

        force_teacher = bool(payload.get("force_teacher", False))
        if not force_teacher and existing_lc is not None:
            if hasattr(existing_lc, "_decide_local_with_tx"):
                local_answers, local_tx = existing_lc._decide_local_with_tx(
                    semantic_text,
                    freeze=is_frozen,
                    cache_key=req_cache_key,
                    structure_sig=req_struct_sig,
                )
                local_tx_ref[0] = existing_lc
                local_tx_ref[1] = local_tx
            else:
                local_answers = existing_lc.decide_local(
                    semantic_text,
                    freeze=is_frozen,
                    cache_key=req_cache_key,
                    structure_sig=req_struct_sig,
                )
            if local_answers is not None:
                with self._lock:
                    prev_c_loc = self._last_schema_by_slot.get(slot_key)
                    preview_drift_loc = (
                        diff_schemas(prev_c_loc, contract)
                        if (prev_c_loc is not None and prev_c_loc.fingerprint != contract.fingerprint)
                        else None
                    )
                body = format_provider_response(
                    wire_format=wire_format,
                    model_requested=str(payload.get("model") or ""),
                    answers=local_answers,
                    input_tokens=est_input_tokens,
                    local_execution=True,
                    schema_fingerprint=contract.fingerprint,
                    drift_info=preview_drift_loc,
                )
                strict_json_dumps(body)
                with self._lock:
                    if self._lifecycles.get(lc_key) is existing_lc:
                        self._lifecycles.move_to_end(lc_key)
                        self._lc_committed_ops[id(existing_lc)] += 1
                        self._lc_pending_new_bind.pop(lc_key, None)
                    self._enforce_unpromoted_lifecycle_cap_locked(reserve_for_new=False)
                    drift_info = self._record_slot_schema_locked(slot_key, contract)
                    self.stats["local_fastpath_requests"] += 1
                if drift_info != preview_drift_loc:
                    body = format_provider_response(
                        wire_format=wire_format,
                        model_requested=str(payload.get("model") or ""),
                        answers=local_answers,
                        input_tokens=est_input_tokens,
                        local_execution=True,
                        schema_fingerprint=contract.fingerprint,
                        drift_info=drift_info,
                    )
                    strict_json_dumps(body)
                local_tx_ref[1] = None
                return 200, body

        upstream_raw = self._call_upstream(wire_format, payload, semantic_text, contract)
        lifecycle_vals, soft_targets, clean_answers = parse_teacher_answers(upstream_raw, contract)
        in_tok = self._extract_upstream_input_tokens(upstream_raw, est_input_tokens)
        model_requested = str(payload.get("model") or "")
        with self._lock:
            prev_c = self._last_schema_by_slot.get(slot_key)
            preview_drift = (
                diff_schemas(prev_c, contract)
                if (prev_c is not None and prev_c.fingerprint != contract.fingerprint)
                else None
            )
        preview_body = format_provider_response(
            wire_format=wire_format,
            model_requested=model_requested,
            answers=clean_answers,
            input_tokens=in_tok,
            local_execution=False,
            schema_fingerprint=contract.fingerprint,
            drift_info=preview_drift,
        )
        try:
            strict_json_dumps(preview_body)
        except Exception as exc:
            raise InteropError(
                f"Upstream response contains invalid strict JSON values: {exc}",
                status_code=502,
            ) from exc
        if not is_frozen:
            obs_canary_tx: Optional[Any] = None
            with self._lock:
                lc = self._lifecycles.get(lc_key)
                if lc is None:
                    self._enforce_unpromoted_lifecycle_cap_locked(reserve_for_new=True)
                    lc = SoftDistillingLifecycle(
                        contract,
                        distill_mode=self.distill_mode,
                        soft_blend=self.soft_blend,
                        config=self.cutover_config,
                    )
                    self._lifecycles[lc_key] = lc
                    self._lc_pending_new_bind[lc_key] = lc
                else:
                    self._lifecycles.move_to_end(lc_key)
                self._lc_inflight_obs[id(lc)] += 1
            if lc is not existing_lc and local_tx_ref[1] is not None and existing_lc is not None:
                if hasattr(existing_lc, "_rollback_local_tx"):
                    existing_lc._rollback_local_tx(local_tx_ref[1])
                local_tx_ref[1] = None
            elif lc is existing_lc:
                obs_canary_tx = local_tx_ref[1]
            try:
                with _bind_active_canary_tx(
                    lc,
                    tx_token=(obs_canary_tx if obs_canary_tx is not None else ((0, 0, 0, "", None), 0)),
                ):
                    lc.observe_teacher(
                        semantic_text,
                        lifecycle_vals,
                        soft_targets=soft_targets,
                        cache_key=req_cache_key,
                        structure_sig=req_struct_sig,
                    )
            except BaseException:
                with self._lock:
                    rem_in = self._lc_inflight_obs.get(id(lc), 0) - 1
                    if rem_in <= 0:
                        self._lc_inflight_obs.pop(id(lc), None)
                    else:
                        self._lc_inflight_obs[id(lc)] = rem_in
                    pend_lc = self._lc_pending_new_bind.get(lc_key)
                    if pend_lc is lc and rem_in <= 0 and self._lc_committed_ops.get(id(lc), 0) == 0:
                        inner_lc = lc.lifecycle
                        if (
                            not lc.is_cutover
                            and inner_lc.step_count == 0
                            and inner_lc.local_hits == 0
                            and not inner_lc.admitted_keys
                            and not inner_lc.quarantine
                            and not inner_lc.teacher_key_labels
                        ):
                            self._lc_pending_new_bind.pop(lc_key, None)
                            if self._lifecycles.get(lc_key) is lc:
                                self._lifecycles.pop(lc_key, None)
                    self._enforce_unpromoted_lifecycle_cap_locked(reserve_for_new=False)
                raise
            local_tx_ref[1] = None
            with self._lock:
                rem_in = self._lc_inflight_obs.get(id(lc), 0) - 1
                if rem_in <= 0:
                    self._lc_inflight_obs.pop(id(lc), None)
                else:
                    self._lc_inflight_obs[id(lc)] = rem_in
                if self._lifecycles.get(lc_key) is lc or rem_in > 0:
                    self._lc_committed_ops[id(lc)] += 1
                    if self._lc_pending_new_bind.get(lc_key) is lc:
                        self._lc_pending_new_bind.pop(lc_key, None)
                self._prune_orphaned_lc_bookkeeping_locked()
        else:
            local_tx_ref[1] = None

        with self._lock:
            if lc_key in self._lifecycles:
                self._lifecycles.move_to_end(lc_key)
            self._enforce_unpromoted_lifecycle_cap_locked(reserve_for_new=False)
            drift_info = self._record_slot_schema_locked(slot_key, contract)

        body = format_provider_response(
            wire_format=wire_format,
            model_requested=model_requested,
            answers=clean_answers,
            input_tokens=in_tok,
            local_execution=False,
            schema_fingerprint=contract.fingerprint,
            drift_info=drift_info,
        )
        return 200, body


class _GatewayThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class UniversalDecisionsHTTPServer:
    """Stdlib 127.0.0.1 loopback HTTP gateway supporting Gemini / Laya / Jev endpoints."""

    def __init__(
        self,
        service: UniversalDecisionService,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        bind_and_activate: bool = True,
    ) -> None:
        if host not in _LOOPBACK_HOSTS:
            raise ValueError(f"Gateway must bind strictly to loopback, got {host!r}")
        if type(bind_and_activate) is not bool:
            raise TypeError("bind_and_activate must be a bool")
        self.service = service

        class _Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self_h, fmt: str, *args: Any) -> None:
                return

            def do_GET(self_h) -> None:
                if self_h.path.split("?", 1)[0] == "/healthz":
                    data = strict_json_dumps({"status": "ok", "stats": dict(service.stats)})
                    self_h.send_response(200)
                    self_h.send_header("Content-Type", "application/json")
                    self_h.send_header("Content-Length", str(len(data)))
                    self_h.end_headers()
                    self_h.wfile.write(data)
                    return
                err = strict_json_dumps({"error": "Not found"})
                self_h.send_response(404)
                self_h.send_header("Content-Type", "application/json")
                self_h.send_header("Content-Length", str(len(err)))
                self_h.end_headers()
                self_h.wfile.write(err)

            def do_POST(self_h) -> None:
                clean_path = self_h.path.split("?", 1)[0]
                if clean_path not in _GATEWAY_POST_ROUTES:
                    err = _safe_error_json_bytes(f"Unknown route: {clean_path}")
                    self_h.send_response(404)
                    self_h.send_header("Content-Type", "application/json")
                    self_h.send_header("Content-Length", str(len(err)))
                    self_h.end_headers()
                    self_h.wfile.write(err)
                    return
                body_consumed = False
                try:
                    req_headers = dict(self_h.headers.items())
                    service.preflight_http_headers(req_headers)
                    raw_len = self_h.headers.get("Content-Length")
                    if raw_len is None:
                        raise InteropError("Missing Content-Length header", status_code=400)
                    try:
                        content_len = int(raw_len)
                    except ValueError as exc:
                        raise InteropError(
                            f"Invalid Content-Length header: {raw_len!r}", status_code=400
                        ) from exc
                    if content_len < 0:
                        raise InteropError(
                            f"Invalid negative Content-Length: {content_len}", status_code=400
                        )
                    if content_len > service.max_body_bytes:
                        raise PayloadTooLargeError(f"Content-Length {content_len} exceeds limit")
                    raw_body = self_h.rfile.read(content_len)
                    body_consumed = True
                    status, body = service.handle_request(
                        path=clean_path,
                        headers=req_headers,
                        raw_body=raw_body,
                    )
                    encoded = strict_json_dumps(body)
                except InteropError as exc:
                    if not body_consumed:
                        self_h.close_connection = True
                    status = exc.status_code
                    encoded = _safe_error_json_bytes(str(exc))
                except ZeroEgressViolationError as exc:
                    status = 403
                    encoded = _safe_error_json_bytes(str(exc))
                except Exception as exc:
                    status = 500
                    encoded = _safe_error_json_bytes(f"Internal error: {exc}")
                self_h.send_response(status)
                self_h.send_header("Content-Type", "application/json")
                self_h.send_header("Content-Length", str(len(encoded)))
                self_h.end_headers()
                self_h.wfile.write(encoded)

        self._httpd = _GatewayThreadingServer(
            (host, port), _Handler, bind_and_activate=bind_and_activate
        )
        if bind_and_activate:
            self.host, self.port = self._httpd.server_address[:2]
        else:
            self.host, self.port = host, int(port)
        self.base_url = f"http://{self.host}:{self.port}"
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "UniversalDecisionsHTTPServer":
        if self._thread is None:
            self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> "UniversalDecisionsHTTPServer":
        return self.start()

    def __exit__(self, *args: Any) -> None:
        self.stop()


class UniversalDecisionASGIApp:
    """Framework-free ASGI 3.0 wrapper exposing the Universal Decision Gateway."""

    def __init__(
        self,
        service: UniversalDecisionService,
        *,
        fallback_app: Optional[Any] = None,
    ) -> None:
        self.service = service
        self.fallback_app = fallback_app

    @staticmethod
    async def _send_raw_json(send: Callable, status: int, encoded: bytes) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": int(status),
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(encoded)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": encoded, "more_body": False})

    @staticmethod
    async def _send_json(send: Callable, status: int, payload: Mapping[str, Any]) -> None:
        encoded = strict_json_dumps(dict(payload))
        await UniversalDecisionASGIApp._send_raw_json(send, status, encoded)

    async def __call__(self, scope: Mapping[str, Any], receive: Callable, send: Callable) -> None:
        if scope.get("type") != "http":
            if self.fallback_app is not None:
                await self.fallback_app(scope, receive, send)
            return

        method = str(scope.get("method", "GET")).upper()
        path = str(scope.get("path", "")).split("?", 1)[0]

        if method == "GET" and path == "/healthz":
            await self._send_json(send, 200, {"status": "ok", "stats": dict(self.service.stats)})
            return

        if method != "POST" or path not in _GATEWAY_POST_ROUTES:
            if self.fallback_app is not None:
                await self.fallback_app(scope, receive, send)
            else:
                await self._send_raw_json(send, 404, _safe_error_json_bytes(f"Unknown route: {path}"))
            return

        raw_headers = {
            (k.decode("latin1") if isinstance(k, bytes) else str(k)): (
                v.decode("latin1") if isinstance(v, bytes) else str(v)
            )
            for k, v in scope.get("headers", ())
        }
        try:
            self.service.preflight_http_headers(raw_headers)
        except InteropError as exc:
            await self._send_raw_json(send, exc.status_code, _safe_error_json_bytes(str(exc)))
            return
        except ZeroEgressViolationError as exc:
            await self._send_raw_json(send, 403, _safe_error_json_bytes(str(exc)))
            return

        chunks: List[bytes] = []
        total_bytes = 0
        more_body = True
        while more_body:
            msg = await receive()
            if msg.get("type") == "http.disconnect":
                return
            chunk = bytes(msg.get("body") or b"")
            total_bytes += len(chunk)
            if total_bytes > self.service.max_body_bytes:
                await self._send_json(
                    send,
                    413,
                    {"error": f"Request body exceeds {self.service.max_body_bytes} bytes"},
                )
                return
            chunks.append(chunk)
            more_body = bool(msg.get("more_body", False))

        try:
            status, body = self.service.handle_request(
                path=path,
                headers=raw_headers,
                raw_body=b"".join(chunks),
            )
            encoded = strict_json_dumps(dict(body))
        except InteropError as exc:
            status = exc.status_code
            encoded = _safe_error_json_bytes(str(exc))
        except ZeroEgressViolationError as exc:
            status = 403
            encoded = _safe_error_json_bytes(str(exc))
        except Exception as exc:
            status = 500
            encoded = _safe_error_json_bytes(f"Internal error: {exc}")
        await self._send_raw_json(send, status, encoded)


def add_universal_decisions_gateway(app: Any, service: UniversalDecisionService) -> None:
    """Register Gemini, Laya, and Jev gateway POST endpoints onto a FastAPI / Starlette app."""
    from starlette.requests import Request as _StarletteRequest
    from starlette.responses import Response as _StarletteResponse

    async def _dispatch(request: _StarletteRequest) -> _StarletteResponse:
        raw_headers = dict(request.headers)
        try:
            service.preflight_http_headers(raw_headers)
            chunks: List[bytes] = []
            total_bytes = 0
            async for chunk in request.stream():
                if not chunk:
                    continue
                data = bytes(chunk)
                total_bytes += len(data)
                if total_bytes > service.max_body_bytes:
                    raise PayloadTooLargeError(
                        f"Request body exceeds {service.max_body_bytes} bytes"
                    )
                chunks.append(data)
            status, payload = service.handle_request(
                path=request.url.path,
                headers=raw_headers,
                raw_body=b"".join(chunks),
            )
            encoded = strict_json_dumps(payload)
        except InteropError as exc:
            status = exc.status_code
            encoded = _safe_error_json_bytes(str(exc))
        except ZeroEgressViolationError as exc:
            status = 403
            encoded = _safe_error_json_bytes(str(exc))
        return _StarletteResponse(
            content=encoded,
            status_code=status,
            media_type="application/json",
        )

    _dispatch.__annotations__ = {"request": _StarletteRequest, "return": _StarletteResponse}
    for route_path in sorted(_GATEWAY_POST_ROUTES):
        app.add_api_route(route_path, _dispatch, methods=["POST"])


class _DecisionsResource:
    """Gemini-style ``client.decisions.create(...)`` namespace."""

    def __init__(self, client: "UniversalDecisionsClient") -> None:
        self._client = client

    def create(
        self,
        *,
        model: str = "gemini-decision-1",
        input: Any,
        questions: Mapping[str, Any],
        slot: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> DotDict:
        payload: Dict[str, Any] = {
            "model": model,
            "input": input,
            "questions": dict(questions),
        }
        if slot is not None:
            payload["slot"] = slot
        return self._client._post("/v1beta/decisions", payload, extra_headers=extra_headers)


class UniversalDecisionsClient:
    """Drop-in client exposing ``client.decisions.create(...)`` and ``client.systemone(...)``."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        service: Optional[UniversalDecisionService] = None,
        api_key: Optional[str] = "test-universal-secret",
        origin: Optional[str] = None,
    ) -> None:
        if not base_url and service is None:
            raise ValueError("Provide either base_url or in-process service")
        self.base_url = base_url.rstrip("/") if base_url else None
        self.service = service
        self.api_key = api_key
        self.origin = origin
        self.decisions = _DecisionsResource(self)

    def _to_dot_response(self, data: Mapping[str, Any]) -> DotDict:
        raw_answers = data.get("answers", {})
        wrapped: Dict[str, Any] = {}
        for k, v in raw_answers.items():
            wrapped[k] = DotDict(dict(v)) if isinstance(v, Mapping) else v
        ans_dot = DotDict(wrapped)
        usage_raw = data.get("usage", {})
        usage = Usage(
            input_tokens=int(usage_raw.get("input_tokens", 0)),
            output_tokens=int(usage_raw.get("output_tokens", 0)),
            total_tokens=int(usage_raw.get("total_tokens", 0)),
        )
        return DotDict(
            {
                "model": data.get("model"),
                "answers": ans_dot,
                "decisions": ans_dot,
                "usage": usage,
                "local_execution": bool(data.get("local_execution", False)),
                "provider_format": data.get("provider_format"),
                "schema_fingerprint": data.get("schema_fingerprint"),
                "schema_drift_detected": bool(data.get("schema_drift_detected", False)),
                "schema_drift": data.get("schema_drift"),
            }
        )

    def _post(
        self,
        route: str,
        payload: Mapping[str, Any],
        *,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> DotDict:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.origin:
            headers["Origin"] = self.origin
        if extra_headers:
            headers.update({str(k): str(v) for k, v in extra_headers.items()})
        raw_bytes = strict_json_dumps(dict(payload))

        if self.service is not None and self.base_url is None:
            status, body = self.service.handle_request(path=route, headers=headers, raw_body=raw_bytes)
            if status >= 400:
                raise InteropError(str(body.get("error", body)), status_code=status)
            return self._to_dot_response(body)

        assert self.base_url is not None
        req = urllib.request.Request(f"{self.base_url}{route}", data=raw_bytes, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return self._to_dot_response(strict_json_loads(resp.read()))
        except urllib.error.HTTPError as exc:
            err_body = exc.read()
            try:
                msg = strict_json_loads(err_body).get("error", err_body.decode("utf-8", "replace"))
            except Exception:
                msg = err_body.decode("utf-8", "replace")
            raise InteropError(str(msg), status_code=exc.code) from exc

    def systemone(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        model: str = "jev-1.13.0",
        slot: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> DotDict:
        route = "/v1/laya/decide" if model.lower().startswith("laya") else "/v1/systemone"
        payload: Dict[str, Any] = {
            "model": model,
            "state": state,
            "questions": dict(questions),
        }
        if slot is not None:
            payload["slot"] = slot
        return self._post(route, payload, extra_headers=extra_headers)


UniversalDecisionHTTPServer = UniversalDecisionsHTTPServer
UniversalDecisionClient = UniversalDecisionsClient

__all__ = [
    "InteropError",
    "PayloadTooLargeError",
    "UnsupportedMediaError",
    "SchemaValidationError",
    "AuthError",
    "ForbiddenOriginError",
    "strict_json_loads",
    "strict_json_dumps",
    "parse_finite_float",
    "coerce_probability_dict",
    "detect_wire_format",
    "extract_semantic_text",
    "QuestionContract",
    "CompiledContractSchema",
    "compile_questions_contract",
    "diff_schemas",
    "parse_teacher_answers",
    "SoftDistillingLifecycle",
    "format_provider_response",
    "UniversalDecisionService",
    "UniversalDecisionsHTTPServer",
    "UniversalDecisionHTTPServer",
    "UniversalDecisionASGIApp",
    "add_universal_decisions_gateway",
    "UniversalDecisionsClient",
    "UniversalDecisionClient",
]
