"""Deterministic system1.cassette.v1 offline JSONL record/replay utilities."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
import socket
from typing import Any, Dict, Iterator, List, Literal, Mapping, Optional, Sequence, Tuple

from system1.core.schema import DecisionSchema


DEFAULT_SEED = 20261113
CASSETTE_VERSION = "system1.cassette.v1"
SECRET_PATTERN = re.compile(
    r"(Bearer\s+[A-Za-z0-9._\-]+|sk-[A-Za-z0-9_\-]{12,}|AIza[0-9A-Za-z_\-]{20,})"
)


class NetworkBlockedError(RuntimeError):
    """Raised when any socket connect is attempted while network blocking is active."""


class CassetteMissError(KeyError):
    """Raised in fail-closed replay mode when a prompt and schema key is missing."""


_NETWORK_BLOCKED = False


def block_network() -> None:
    """Block outbound TCP/UDP/Unix socket connections for offline CI replay."""
    global _NETWORK_BLOCKED
    if _NETWORK_BLOCKED:
        return

    def _blocked_connect(*args: Any, **kwargs: Any) -> Any:
        raise NetworkBlockedError("Network sockets are blocked during offline cassette replay")

    socket.socket.connect = _blocked_connect  # type: ignore[method-assign]
    socket.create_connection = _blocked_connect  # type: ignore[assignment]
    _NETWORK_BLOCKED = True


@contextmanager
def network_blocked() -> Iterator[None]:
    """Context manager temporarily blocking outbound socket connections."""
    orig_connect = socket.socket.connect
    orig_create = socket.create_connection

    def _blocked_connect(*args: Any, **kwargs: Any) -> Any:
        raise NetworkBlockedError("Network sockets are blocked during offline cassette replay")

    socket.socket.connect = _blocked_connect  # type: ignore[method-assign]
    socket.create_connection = _blocked_connect  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = orig_connect  # type: ignore[method-assign]
        socket.create_connection = orig_create  # type: ignore[assignment]


def normalize_prompt(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def scrub_secrets(text: str) -> str:
    return SECRET_PATTERN.sub("[REDACTED]", text)


def canonical_schema_dict(schema: DecisionSchema) -> Dict[str, Any]:
    fields = []
    for name, fdef in schema.fields.items():
        fields.append(
            {
                "name": name,
                "kind": getattr(fdef, "field_type", "choice"),
                "options": list(getattr(fdef, "options", ())),
                "description": getattr(fdef, "description", "") or "",
            }
        )
    return {"schema_name": schema.schema_name, "fields": fields}


def compute_schema_sha256(schema: DecisionSchema) -> str:
    payload = json.dumps(
        canonical_schema_dict(schema),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_request_key(
    *,
    provider: str,
    model: str,
    system_prompt: str,
    prompt_text: str,
    schema_sha256: str,
    temperature: float = 0.0,
    seed: Optional[int] = DEFAULT_SEED,
) -> str:
    envelope = {
        "schema_version": CASSETTE_VERSION,
        "provider": provider,
        "model": model,
        "system_prompt": normalize_prompt(system_prompt),
        "prompt_text": normalize_prompt(scrub_secrets(prompt_text)),
        "schema_sha256": schema_sha256,
        "temperature": round(float(temperature), 6),
        "seed": seed,
    }
    encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class StreamEvent:
    """Single chronological event for prequential stream evaluation."""

    t: int
    regime: str
    prompt: str
    true_label: str
    is_oos: bool = False
    is_repeat: bool = False
    source_dataset: str = "synthetic_support"


class CassetteTeacher:
    """Offline system1.cassette.v1 JSONL teacher callable compatible with BoundedReviewFallback.

    Strips ``eval_audit`` on load so runtime teacher lookups cannot access ground-truth labels.
    """

    def __init__(
        self,
        cassette_path: Path,
        schema: DecisionSchema,
        *,
        provider: str = "recorded_sim",
        model: str = "gemini-2.5-flash-sim-v1",
        system_prompt: str = "Classify the request into exactly one schema field value.",
        temperature: float = 0.0,
        seed: Optional[int] = DEFAULT_SEED,
        strict_fifo: bool = False,
    ) -> None:
        self.cassette_path = Path(cassette_path)
        self.schema = schema
        self.schema_sha256 = compute_schema_sha256(schema)
        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = float(temperature)
        self.seed = seed
        self.strict_fifo = strict_fifo
        self.index: Dict[str, List[Dict[str, Any]]] = {}
        self.cursors: Dict[str, int] = {}
        self.call_count: int = 0
        self.total_recorded_teacher_latency_ms: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self._load_replay_index()

    def _load_replay_index(self) -> None:
        with self.cassette_path.open("r", encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("schema_version") != CASSETTE_VERSION:
                    raise ValueError(f"Unsupported cassette version on line {line_no}")
                if rec.get("record_type") == "header":
                    continue
                if rec.get("schema_sha256") != self.schema_sha256:
                    raise ValueError(
                        f"Schema mismatch on line {line_no}: {rec.get('schema_sha256')} != {self.schema_sha256}"
                    )
                sanitized = {
                    "request_key": rec["request_key"],
                    "response": dict(rec["response"]),
                    "telemetry": dict(rec.get("telemetry", {})),
                    "sequence_hint": rec.get("sequence_hint", 0),
                }
                self.index.setdefault(rec["request_key"], []).append(sanitized)

        for key in self.index:
            self.index[key].sort(key=lambda r: r.get("sequence_hint") or 0)

    def __call__(self, prompt_text: str) -> Dict[str, Any]:
        req_key = compute_request_key(
            provider=self.provider,
            model=self.model,
            system_prompt=self.system_prompt,
            prompt_text=prompt_text,
            schema_sha256=self.schema_sha256,
            temperature=self.temperature,
            seed=self.seed,
        )
        entries = self.index.get(req_key)
        if not entries:
            preview = scrub_secrets(prompt_text[:80])
            raise CassetteMissError(f"Cassette miss for {req_key} ({preview!r})")
        idx = self.cursors.get(req_key, 0) if self.strict_fifo else 0
        if idx >= len(entries):
            idx = len(entries) - 1
        if self.strict_fifo:
            self.cursors[req_key] = idx + 1
        entry = entries[idx]
        self.call_count += 1
        telem = entry["telemetry"]
        self.total_recorded_teacher_latency_ms += float(telem.get("latency_ms", 0.0))
        self.total_input_tokens += int(telem.get("input_tokens", 0))
        self.total_output_tokens += int(telem.get("output_tokens", 0))
        return dict(entry["response"]["parsed"])


def load_cassette_events(cassette_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load cassette header and chronological events (including eval_audit for test-then-train scoring)."""
    header: Dict[str, Any] = {}
    events: List[Dict[str, Any]] = []
    with Path(cassette_path).open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("schema_version") != CASSETTE_VERSION:
                raise ValueError(f"Unsupported cassette version on line {line_no}")
            if rec.get("record_type") == "header":
                header = rec
                continue
            events.append(
                {
                    "sequence_hint": int(rec.get("sequence_hint", len(events))),
                    "regime": str(rec.get("regime", "stream")),
                    "prompt_text": str(rec["request"]["prompt_text"]),
                    "eval_audit": dict(rec.get("eval_audit", {})),
                }
            )
    events.sort(key=lambda item: item["sequence_hint"])
    return header, events


def write_cassette_jsonl(
    path: Path,
    schema: DecisionSchema,
    field_name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    cassette_id: str = "support-stream-v1",
    provider: str = "recorded_sim",
    model: str = "gemini-2.5-flash-sim-v1",
    system_prompt: str = "Classify the request into exactly one schema field value.",
    noise_rate: float = 0.0,
    rng_seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Deterministically write a system1.cassette.v1 JSONL file with secret scrubbing."""
    rng = random.Random(rng_seed)
    schema_sha = compute_schema_sha256(schema)
    schema_dict = canonical_schema_dict(schema)
    options = list(getattr(schema.fields[field_name], "options", ()))

    header = {
        "schema_version": CASSETTE_VERSION,
        "record_type": "header",
        "cassette_id": cassette_id,
        "created_at": "2026-11-13T12:00:00Z",
        "provider": provider,
        "model": model,
        "schema_sha256": schema_sha,
        "field_name": field_name,
        "options": options,
        "redaction_policy": "strip_auth_secrets_v1",
    }

    records: List[Dict[str, Any]] = []
    prompt_cache: Dict[str, Tuple[str, bool]] = {}
    for idx, row in enumerate(rows):
        prompt = scrub_secrets(str(row["prompt"]))
        true_label = str(row["true_label"])
        teacher_label = row.get("teacher_label")
        is_oos = bool(row.get("is_oos", False))
        regime = str(row.get("regime", "stream"))
        norm_p = normalize_prompt(prompt)

        if teacher_label is not None:
            pred = str(teacher_label)
            correct = (not is_oos) and (pred == true_label)
        elif norm_p in prompt_cache:
            pred, correct = prompt_cache[norm_p]
        else:
            if not is_oos and noise_rate > 0.0 and rng.random() < noise_rate:
                alts = [o for o in options if o != true_label]
                pred = rng.choice(alts) if alts else true_label
                correct = False
            else:
                pred = true_label if true_label in options else options[0]
                correct = not is_oos
            prompt_cache[norm_p] = (pred, correct)

        req_key = compute_request_key(
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            prompt_text=prompt,
            schema_sha256=schema_sha,
            temperature=0.0,
            seed=DEFAULT_SEED,
        )
        in_toks = max(24, len(prompt.split()) * 2 + 96)
        out_toks = 10
        lat_ms = round(135.0 + min(180.0, len(prompt) * 0.4) + rng.uniform(0.0, 25.0), 2)
        parsed = {field_name: pred}
        records.append(
            {
                "schema_version": CASSETTE_VERSION,
                "record_type": "interaction",
                "request_key": req_key,
                "schema_sha256": schema_sha,
                "provider": provider,
                "model": model,
                "sequence_hint": idx,
                "regime": regime,
                "request": {
                    "prompt_text": prompt,
                    "system_prompt": system_prompt,
                    "schema": schema_dict,
                    "temperature": 0.0,
                    "seed": DEFAULT_SEED,
                },
                "response": {
                    "parsed": parsed,
                    "raw_text": json.dumps(parsed, separators=(",", ":")),
                    "finish_reason": "stop",
                    "forced_choice": is_oos,
                },
                "telemetry": {
                    "latency_ms": lat_ms,
                    "ttft_ms": round(lat_ms * 0.72, 2),
                    "input_tokens": in_toks,
                    "output_tokens": out_toks,
                    "cost_usd": round((in_toks * 0.15 + out_toks * 0.60) / 1_000_000.0, 6),
                },
                "eval_audit": {
                    "true_label": true_label,
                    "is_oos": is_oos,
                    "teacher_correct": correct,
                },
            }
        )

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n")
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")

    return {"cassette_path": str(out_path), "events": len(records), "schema_sha256": schema_sha}
