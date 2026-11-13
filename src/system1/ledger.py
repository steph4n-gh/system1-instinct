"""System 1 Lightweight Tamper-Evident SQLite Action Ledger.

Provides cryptographic audit logging and hash chaining for decision receipts,
ensuring non-repudiation, tamper detection, and historical chain verification.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

from system1.receipt import (
    canonical_json,
    check_decision_receipt_integrity,
    compute_receipt_digest,
    fingerprint,
    utc_now,
    verify_decision_witness_receipt,
)

_ZERO_HASH = "0" * 64

VALID_DURABILITY_MODES: Tuple[str, ...] = (
    "process_wal",
    "strict_sync",
    "group_commit",
    "high_throughput_normal",
)


class LedgerError(RuntimeError):
    """Base exception for ledger operations."""
    pass


class LedgerWriteError(LedgerError):
    """Raised when writing to the audit ledger fails in fail-closed mode."""
    pass


class IntegrityError(LedgerError):
    """Raised when cryptographic chain validation fails."""
    pass


def _trust_key_token(trusted_public_key: Optional[Any]) -> Optional[str]:
    if trusted_public_key is None:
        return None
    if hasattr(trusted_public_key, "public_bytes_raw"):
        try:
            raw = trusted_public_key.public_bytes_raw()
            if isinstance(raw, (bytes, bytearray)):
                return raw.hex()
        except Exception:
            pass
    if isinstance(trusted_public_key, (bytes, bytearray)):
        return bytes(trusted_public_key).hex()
    if isinstance(trusted_public_key, str):
        return trusted_public_key
    return f"id:{id(trusted_public_key)}:{repr(trusted_public_key)}"


@dataclass
class _GroupWriteRequest:
    kind: str
    kwargs: Dict[str, Any]
    event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Optional[BaseException] = None


class ActionLedger:
    """Lightweight, tamper-evident SQLite audit ledger with hash chaining."""

    VALID_DURABILITY_MODES = VALID_DURABILITY_MODES

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        db_path: Optional[str | Path] = None,
        read_only: bool = False,
        require_durable: bool = False,
        enforce_durability: bool = False,
        durability_mode: str = "process_wal",
        fullfsync: Optional[bool] = None,
        group_commit_ms: float = 4.0,
        group_commit_max_batch: int = 32,
    ) -> None:
        self.path = str(db_path if db_path is not None else path)
        self.read_only = bool(read_only)
        mode = str(durability_mode).strip()
        if mode not in VALID_DURABILITY_MODES:
            raise ValueError(
                f"Unsupported durability_mode '{durability_mode}'; expected one of {VALID_DURABILITY_MODES}"
            )
        self.durability_mode = mode
        self.fullfsync = (mode == "strict_sync") if fullfsync is None else bool(fullfsync)
        self.group_commit_ms = max(0.0, float(group_commit_ms))
        self.group_commit_max_batch = max(1, int(group_commit_max_batch))
        self.require_durable = bool(require_durable)
        self.enforce_durability = bool(enforce_durability)

        if self.read_only and self.path == ":memory:":
            raise ValueError("Read-only ledger requires an existing durable database file")
        if (require_durable or enforce_durability) and not self.is_durable:
            raise ValueError(
                "Enforcement profile requires a persistent, durable ActionLedger; in-memory storage is prohibited"
            )
        if enforce_durability and self.durability_mode == "high_throughput_normal":
            raise ValueError(
                "enforce_durability prohibits durability_mode='high_throughput_normal'"
            )

        uri = False
        database = self.path
        if self.path != ":memory:":
            p = Path(self.path).expanduser().resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            self.path = str(p)
            if self.read_only:
                database = p.as_uri() + "?mode=ro"
                uri = True
            else:
                database = str(p)

        self._connection = sqlite3.connect(
            database,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
            uri=uri,
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()

        # O(1) verification watermark invalidated on PRAGMA data_version or total_changes drift.
        self._cache_verified: bool = False
        self._cache_data_version: Optional[int] = None
        self._expected_total_changes: int = int(self._connection.total_changes)
        self._verified_seq: int = 0
        self._verified_hash: str = _ZERO_HASH
        self._verified_trust_tokens: set[str] = set()

        self._gc_cond = threading.Condition()
        self._gc_queue: deque[_GroupWriteRequest] = deque()
        self._gc_closing: bool = False
        self._gc_thread: Optional[threading.Thread] = None

        self._configure()
        if not self.read_only:
            self._initialize()
            self._expected_total_changes = int(self._connection.total_changes)
            if self.durability_mode == "group_commit":
                self._gc_thread = threading.Thread(
                    target=self._group_commit_loop,
                    name="system1-ledger-group-commit",
                    daemon=True,
                )
                self._gc_thread.start()

    @property
    def is_durable(self) -> bool:
        """Returns True if the ledger writes to persistent disk storage with full synchronous durability."""
        p = str(self.path).strip()
        if p == ":memory:" or p.startswith("file::memory:") or p.startswith(":memory:"):
            return False
        return True

    def _configure(self) -> None:
        sync_mode = "NORMAL" if self.durability_mode == "high_throughput_normal" else "FULL"
        fullfsync_pragma = "ON" if self.fullfsync else "OFF"
        with self._lock:
            self._connection.execute("PRAGMA busy_timeout = 5000;")
            self._connection.execute("PRAGMA foreign_keys = ON;")
            if not self.read_only:
                self._connection.execute("PRAGMA journal_mode = WAL;")
                self._connection.execute(f"PRAGMA synchronous = {sync_mode};")
                self._connection.execute(f"PRAGMA fullfsync = {fullfsync_pragma};")

    def durability_info(self) -> Dict[str, Any]:
        """Inspect live SQLite durability pragmas and configured ledger mode."""
        with self._lock:
            jm = self._connection.execute("PRAGMA journal_mode;").fetchone()
            sync = self._connection.execute("PRAGMA synchronous;").fetchone()
            ff = self._connection.execute("PRAGMA fullfsync;").fetchone()
        return {
            "durability_mode": self.durability_mode,
            "is_durable": self.is_durable,
            "read_only": self.read_only,
            "journal_mode": str(jm[0]) if jm is not None else "",
            "synchronous": int(sync[0]) if sync is not None else -1,
            "fullfsync": bool(ff[0]) if ff is not None else self.fullfsync,
            "group_commit_ms": self.group_commit_ms,
            "group_commit_max_batch": self.group_commit_max_batch,
        }

    def _initialize(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS audit_entries (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    tenant_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    action_id TEXT,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    entry_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_entries(action_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_audit_entry_hash ON audit_entries(entry_hash);

                CREATE TABLE IF NOT EXISTS ledger_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                INSERT OR IGNORE INTO ledger_meta(key, value) VALUES
                    ('audit_head_hash', '0000000000000000000000000000000000000000000000000000000000000000'),
                    ('audit_head_sequence', '0');
                """
            )

    def _current_data_version(self, connection: sqlite3.Connection) -> int:
        row = connection.execute("PRAGMA data_version").fetchone()
        return int(row[0]) if row is not None else 0

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            raise LedgerError("Cannot perform write transaction on read-only ledger")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.rollback()
                self._cache_verified = False
                self._verified_trust_tokens.clear()
                raise
            else:
                self._connection.commit()
                self._expected_total_changes = int(self._connection.total_changes)
                self._cache_data_version = self._current_data_version(self._connection)

    def _finish_write_commit_locked(
        self,
        *,
        tip_sequence: int,
        tip_hash: str,
        trust_tokens: Optional[set[str]] = None,
    ) -> None:
        self._cache_verified = True
        self._verified_seq = int(tip_sequence)
        self._verified_hash = str(tip_hash)
        if trust_tokens is not None:
            self._verified_trust_tokens = set(trust_tokens)

    def flush(self) -> None:
        """Wait for any pending group-commit batch to flush to SQLite."""
        if self.durability_mode != "group_commit":
            return
        with self._gc_cond:
            pending = list(self._gc_queue)
        for req in pending:
            req.event.wait()

    def close(self) -> None:
        if self._gc_thread is not None:
            with self._gc_cond:
                self._gc_closing = True
                self._gc_cond.notify_all()
            if self._gc_thread.is_alive() and threading.current_thread() is not self._gc_thread:
                self._gc_thread.join(timeout=5.0)
            self._gc_thread = None
        with self._lock:
            self._connection.close()

    def __enter__(self) -> ActionLedger:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def audit_head(self) -> Tuple[int, str]:
        """Returns (sequence, head_hash) of the latest audit record."""
        with self._lock:
            row = self._connection.execute(
                "SELECT sequence, entry_hash FROM audit_entries ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            actual = (
                (0, _ZERO_HASH) if row is None else (int(row["sequence"]), str(row["entry_hash"]))
            )
            meta = dict(self._connection.execute("SELECT key, value FROM ledger_meta").fetchall())
            persisted = (
                int(meta.get("audit_head_sequence", "-1")),
                str(meta.get("audit_head_hash", "")),
            )
            if persisted != actual:
                raise IntegrityError("Persisted audit head does not match the chain tip")
            return actual

    def head_hash(self) -> str:
        """Returns the current audit head hash string."""
        return self.audit_head()[1]

    def _verify_integrity_locked(
        self,
        connection: sqlite3.Connection,
        trusted_public_key: Optional[Any] = None,
        *,
        full_scan: bool = False,
    ) -> bool:
        """Verify cryptographic chain integrity; uses O(1) watermark when unmodified."""
        dv = self._current_data_version(connection)
        tc = int(connection.total_changes)
        token = _trust_key_token(trusted_public_key)

        if (
            not full_scan
            and self._cache_verified
            and self._cache_data_version == dv
            and tc == self._expected_total_changes
        ):
            meta = dict(connection.execute("SELECT key, value FROM ledger_meta").fetchall())
            if (
                meta.get("audit_head_hash") == self._verified_hash
                and int(meta.get("audit_head_sequence", "-1")) == self._verified_seq
            ):
                tip = connection.execute(
                    "SELECT sequence, entry_hash FROM audit_entries ORDER BY sequence DESC LIMIT 1"
                ).fetchone()
                actual_seq = 0 if tip is None else int(tip["sequence"])
                actual_hash = _ZERO_HASH if tip is None else str(tip["entry_hash"])
                if actual_seq == self._verified_seq and actual_hash == self._verified_hash:
                    if token is None or token in self._verified_trust_tokens or actual_seq == 0:
                        return True

        entries = connection.execute(
            "SELECT * FROM audit_entries ORDER BY sequence ASC"
        ).fetchall()
        previous = _ZERO_HASH
        last_sequence = 0

        for row in entries:
            if row["previous_hash"] != previous:
                self._cache_verified = False
                self._verified_trust_tokens.clear()
                return False
            try:
                payload_data = json.loads(row["payload_json"])
                body = {
                    "event_id": row["event_id"],
                    "tenant_id": row["tenant_id"],
                    "principal_id": row["principal_id"],
                    "scope": row["scope"],
                    "action_id": row["action_id"],
                    "event_type": row["event_type"],
                    "payload": payload_data,
                    "previous_hash": row["previous_hash"],
                    "created_at": row["created_at"],
                }
            except (TypeError, ValueError, json.JSONDecodeError):
                self._cache_verified = False
                self._verified_trust_tokens.clear()
                return False

            if fingerprint(body) != row["entry_hash"]:
                self._cache_verified = False
                self._verified_trust_tokens.clear()
                return False

            if trusted_public_key is not None:
                target_receipt = None
                if isinstance(payload_data, dict):
                    if isinstance(payload_data.get("receipt"), dict) and "envelope" in payload_data["receipt"]:
                        target_receipt = payload_data["receipt"]
                    elif "envelope" in payload_data:
                        target_receipt = payload_data

                if row["event_type"] in ("system1_decision", "decision_receipt", "action_receipt"):
                    if not target_receipt:
                        self._cache_verified = False
                        self._verified_trust_tokens.clear()
                        return False
                    try:
                        if not verify_decision_witness_receipt(target_receipt, public_key=trusted_public_key):
                            self._cache_verified = False
                            self._verified_trust_tokens.clear()
                            return False
                    except Exception:
                        self._cache_verified = False
                        self._verified_trust_tokens.clear()
                        return False
                elif target_receipt is not None:
                    try:
                        if not verify_decision_witness_receipt(target_receipt, public_key=trusted_public_key):
                            self._cache_verified = False
                            self._verified_trust_tokens.clear()
                            return False
                    except Exception:
                        self._cache_verified = False
                        self._verified_trust_tokens.clear()
                        return False

            previous = row["entry_hash"]
            last_sequence = int(row["sequence"])

        meta = dict(connection.execute("SELECT key, value FROM ledger_meta").fetchall())
        if meta.get("audit_head_hash") != previous or int(meta.get("audit_head_sequence", "-1")) != last_sequence:
            self._cache_verified = False
            self._verified_trust_tokens.clear()
            return False

        self._cache_verified = True
        self._cache_data_version = dv
        self._expected_total_changes = tc
        self._verified_seq = last_sequence
        self._verified_hash = previous
        if token is not None:
            self._verified_trust_tokens.add(token)
        return True

    def _read_meta_head_locked(self, connection: sqlite3.Connection) -> Tuple[int, str]:
        meta = dict(connection.execute("SELECT key, value FROM ledger_meta").fetchall())
        return int(meta.get("audit_head_sequence", "0")), str(meta.get("audit_head_hash", _ZERO_HASH))

    def _insert_entry_locked(
        self,
        connection: sqlite3.Connection,
        *,
        event_type: str,
        action_id: Optional[str],
        tenant_id: str,
        principal_id: str,
        scope: str,
        entry_payload: Mapping[str, Any],
        previous_sequence: int,
        previous_hash: str,
        update_meta: bool = True,
    ) -> Tuple[int, str]:
        created_at = utc_now()
        event_id = f"audit_{fingerprint([previous_hash, event_type, created_at])[:32]}"
        payload_dict = dict(entry_payload)
        body = {
            "event_id": event_id,
            "tenant_id": tenant_id,
            "principal_id": principal_id,
            "scope": scope,
            "action_id": action_id,
            "event_type": event_type,
            "payload": payload_dict,
            "previous_hash": previous_hash,
            "created_at": created_at,
        }
        entry_hash = fingerprint(body)
        cursor = connection.execute(
            """
            INSERT INTO audit_entries(
                event_id, tenant_id, principal_id, scope, action_id, event_type,
                payload_json, previous_hash, entry_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                tenant_id,
                principal_id,
                scope,
                action_id,
                event_type,
                canonical_json(payload_dict),
                previous_hash,
                entry_hash,
                created_at,
            ),
        )
        sequence = int(cursor.lastrowid)
        if sequence <= previous_sequence:
            raise IntegrityError("Audit sequence did not advance")
        if update_meta:
            connection.execute(
                "UPDATE ledger_meta SET value=? WHERE key='audit_head_hash'", (entry_hash,)
            )
            connection.execute(
                "UPDATE ledger_meta SET value=? WHERE key='audit_head_sequence'", (str(sequence),)
            )
        return sequence, entry_hash

    def _validate_and_prepare_decision(
        self,
        receipt: Mapping[str, Any] | Any,
        *,
        tenant_id: str,
        principal_id: str,
        scope: str,
        trusted_public_key: Optional[Any],
    ) -> Tuple[str, str, str, Optional[str], Dict[str, Any]]:
        payload = receipt.to_dict() if hasattr(receipt, "to_dict") else dict(receipt)
        eff_tenant = payload.get("tenant_id") or tenant_id
        eff_principal = payload.get("principal_id") or principal_id
        eff_scope = payload.get("scope") or scope
        if trusted_public_key is not None:
            if not verify_decision_witness_receipt(payload, public_key=trusted_public_key):
                raise IntegrityError("Incoming receipt failed signature verification against trust anchor.")
        decision_id = payload.get("decision_id")
        receipt_digest = payload.get("receipt_digest") or (
            receipt.compute_digest() if hasattr(receipt, "compute_digest") else None
        )
        entry_payload: Dict[str, Any] = {
            "decision_id": decision_id,
            "schema_name": payload.get("schema_name"),
            "schema_digest": payload.get("schema_digest"),
            "prompt_digest": payload.get("prompt_digest"),
            "receipt_digest": receipt_digest,
            "is_ambiguous": payload.get("is_ambiguous"),
        }
        if "envelope" in payload:
            entry_payload["envelope"] = payload["envelope"]
        if "signer_public_key" in payload:
            entry_payload["signer_public_key"] = payload["signer_public_key"]
        entry_payload["receipt"] = payload
        action_id = payload.get("action_id") or decision_id
        return eff_tenant, eff_principal, eff_scope, action_id, entry_payload

    def _exec_append_locked(
        self,
        connection: sqlite3.Connection,
        *,
        receipt: Mapping[str, Any] | Any,
        tenant_id: str = "system1",
        principal_id: str = "engine",
        scope: str = "decision",
        trusted_public_key: Optional[Any] = None,
        head_state: Optional[Tuple[int, str]] = None,
        update_meta: bool = True,
    ) -> Tuple[int, str, Optional[str]]:
        eff_tenant, eff_principal, eff_scope, action_id, entry_payload = self._validate_and_prepare_decision(
            receipt,
            tenant_id=tenant_id,
            principal_id=principal_id,
            scope=scope,
            trusted_public_key=trusted_public_key,
        )
        previous_sequence, previous_hash = head_state if head_state is not None else self._read_meta_head_locked(connection)
        seq, entry_hash = self._insert_entry_locked(
            connection,
            event_type="system1_decision",
            action_id=action_id,
            tenant_id=eff_tenant,
            principal_id=eff_principal,
            scope=eff_scope,
            entry_payload=entry_payload,
            previous_sequence=previous_sequence,
            previous_hash=previous_hash,
            update_meta=update_meta,
        )
        return seq, entry_hash, _trust_key_token(trusted_public_key)

    def _exec_outcome_locked(
        self,
        connection: sqlite3.Connection,
        *,
        action_id: str,
        receipt_digest: str,
        status: str,
        result_payload: Optional[Mapping[str, Any]] = None,
        error_message: Optional[str] = None,
        tenant_id: str = "system1",
        principal_id: str = "engine",
        scope: str = "execution_outcome",
        head_state: Optional[Tuple[int, str]] = None,
        update_meta: bool = True,
    ) -> Tuple[int, str]:
        valid_statuses = {"SUCCEEDED", "FAILED", "ERROR", "CANCELLED", "ABORTED", "INDETERMINATE"}
        if str(status).upper() not in valid_statuses:
            raise LedgerWriteError(f"Invalid execution outcome status: {status}")

        prior = connection.execute(
            "SELECT * FROM audit_entries WHERE (action_id = ? OR json_extract(payload_json, '$.decision_id') = ?) AND event_type IN ('system1_decision', 'decision_receipt') ORDER BY sequence DESC LIMIT 1",
            (action_id, action_id),
        ).fetchone()
        if not prior:
            raise LedgerWriteError(f"Orphan execution outcome: no prior authorization found for action_id {action_id}")

        try:
            prior_p = json.loads(prior["payload_json"])
            found_digest = prior_p.get("receipt_digest") or prior_p.get("receipt", {}).get("receipt_digest")
        except Exception:
            found_digest = None

        if not found_digest:
            raise LedgerWriteError("Explicit permission requires a mandatory exact nonempty digest")
        if found_digest != receipt_digest:
            raise LedgerWriteError(
                f"Wrong action association: outcome receipt_digest {receipt_digest} does not match prior {found_digest}"
            )
        if prior["tenant_id"] != tenant_id or prior["principal_id"] != principal_id:
            raise LedgerWriteError(
                f"Authorization mismatch: outcome context ({tenant_id}/{principal_id}) does not match prior ({prior['tenant_id']}/{prior['principal_id']})"
            )
        if scope != "execution_outcome" and prior["scope"] != scope:
            raise LedgerWriteError(
                f"Authorization mismatch: outcome scope ({scope}) does not match prior ({prior['scope']})"
            )

        previous_sequence, previous_hash = head_state if head_state is not None else self._read_meta_head_locked(connection)
        entry_payload = {
            "action_id": action_id,
            "prior_receipt_digest": receipt_digest,
            "status": str(status),
            "error_message": error_message,
            "result_summary": dict(result_payload) if result_payload is not None else {},
        }
        return self._insert_entry_locked(
            connection,
            event_type="execution_outcome",
            action_id=action_id,
            tenant_id=tenant_id,
            principal_id=principal_id,
            scope=scope,
            entry_payload=entry_payload,
            previous_sequence=previous_sequence,
            previous_hash=previous_hash,
            update_meta=update_meta,
        )

    def _exec_action_locked(
        self,
        connection: sqlite3.Connection,
        *,
        action: str,
        proposal: Mapping[str, Any] | Any,
        action_id: Optional[str] = None,
        tenant_id: str = "system1",
        principal_id: str = "engine",
        scope: str = "action",
        head_state: Optional[Tuple[int, str]] = None,
        update_meta: bool = True,
    ) -> Tuple[int, str]:
        payload = dict(proposal) if isinstance(proposal, Mapping) else {"proposal": str(proposal)}
        act_id = (
            action_id
            or payload.get("action_id")
            or payload.get("id")
            or f"act_{fingerprint([action, str(payload)])[:16]}"
        )
        previous_sequence, previous_hash = head_state if head_state is not None else self._read_meta_head_locked(connection)
        return self._insert_entry_locked(
            connection,
            event_type=str(action),
            action_id=act_id,
            tenant_id=tenant_id,
            principal_id=principal_id,
            scope=scope,
            entry_payload=payload,
            previous_sequence=previous_sequence,
            previous_hash=previous_hash,
            update_meta=update_meta,
        )

    def _exec_coalesced_ro_locked(
        self,
        connection: sqlite3.Connection,
        *,
        receipt: Mapping[str, Any] | Any,
        reversibility: str,
        action_id: Optional[str] = None,
        receipt_digest: Optional[str] = None,
        status: str = "SUCCEEDED",
        result_payload: Optional[Mapping[str, Any]] = None,
        error_message: Optional[str] = None,
        tenant_id: str = "system1",
        principal_id: str = "engine",
        scope: str = "decision",
        outcome_scope: str = "execution_outcome",
        trusted_public_key: Optional[Any] = None,
        head_state: Optional[Tuple[int, str]] = None,
        update_meta: bool = True,
    ) -> Tuple[int, str, str, str, Optional[str]]:
        payload = receipt.to_dict() if hasattr(receipt, "to_dict") else dict(receipt)
        eff_rev = str(reversibility or payload.get("reversibility") or "").strip().upper()
        if eff_rev != "READ_ONLY":
            raise LedgerWriteError(
                f"Coalesced ledger recording is strictly restricted to READ_ONLY actions (got '{reversibility}')"
            )
        valid_statuses = {"SUCCEEDED", "FAILED", "ERROR", "CANCELLED", "ABORTED", "INDETERMINATE"}
        if str(status).upper() not in valid_statuses:
            raise LedgerWriteError(f"Invalid execution outcome status: {status}")

        eff_tenant, eff_principal, eff_scope, derived_action_id, entry_payload = self._validate_and_prepare_decision(
            receipt,
            tenant_id=tenant_id,
            principal_id=principal_id,
            scope=scope,
            trusted_public_key=trusted_public_key,
        )
        found_digest = entry_payload.get("receipt_digest")
        if not found_digest:
            raise LedgerWriteError("Explicit permission requires a mandatory exact nonempty digest")
        if receipt_digest is not None and found_digest != receipt_digest:
            raise LedgerWriteError(
                f"Wrong action association: outcome receipt_digest {receipt_digest} does not match prior {found_digest}"
            )
        eff_action_id = action_id or derived_action_id
        if not eff_action_id:
            raise LedgerWriteError("Coalesced execution outcome requires a valid action_id")
        if outcome_scope != "execution_outcome" and eff_scope != outcome_scope:
            raise LedgerWriteError(
                f"Authorization mismatch: outcome scope ({outcome_scope}) does not match prior ({eff_scope})"
            )

        previous_sequence, previous_hash = head_state if head_state is not None else self._read_meta_head_locked(connection)
        seq1, decision_hash = self._insert_entry_locked(
            connection,
            event_type="system1_decision",
            action_id=eff_action_id,
            tenant_id=eff_tenant,
            principal_id=eff_principal,
            scope=eff_scope,
            entry_payload=entry_payload,
            previous_sequence=previous_sequence,
            previous_hash=previous_hash,
            update_meta=False,
        )
        outcome_payload = {
            "action_id": eff_action_id,
            "prior_receipt_digest": found_digest,
            "status": str(status),
            "error_message": error_message,
            "result_summary": dict(result_payload) if result_payload is not None else {},
        }
        seq2, outcome_hash = self._insert_entry_locked(
            connection,
            event_type="execution_outcome",
            action_id=eff_action_id,
            tenant_id=eff_tenant,
            principal_id=eff_principal,
            scope=outcome_scope,
            entry_payload=outcome_payload,
            previous_sequence=seq1,
            previous_hash=decision_hash,
            update_meta=update_meta,
        )
        return seq2, outcome_hash, decision_hash, outcome_hash, _trust_key_token(trusted_public_key)

    def _submit_group_write(self, kind: str, **kwargs: Any) -> Any:
        if self.read_only:
            raise LedgerError("Cannot perform write transaction on read-only ledger")
        req = _GroupWriteRequest(kind=kind, kwargs=kwargs)
        with self._gc_cond:
            if self._gc_closing:
                raise LedgerError("Cannot write to a closed ActionLedger")
            self._gc_queue.append(req)
            self._gc_cond.notify_all()
        req.event.wait()
        if req.error is not None:
            raise req.error
        return req.result

    def _group_commit_loop(self) -> None:
        while True:
            with self._gc_cond:
                while not self._gc_queue and not self._gc_closing:
                    self._gc_cond.wait()
                if not self._gc_queue and self._gc_closing:
                    return
                if (
                    self.group_commit_ms > 0.0
                    and not self._gc_closing
                    and len(self._gc_queue) < self.group_commit_max_batch
                ):
                    deadline = time.monotonic() + (self.group_commit_ms / 1000.0)
                    while (
                        not self._gc_closing
                        and len(self._gc_queue) < self.group_commit_max_batch
                    ):
                        remaining = deadline - time.monotonic()
                        if remaining <= 0.0:
                            break
                        self._gc_cond.wait(timeout=remaining)
                batch: List[_GroupWriteRequest] = []
                while self._gc_queue and len(batch) < self.group_commit_max_batch:
                    batch.append(self._gc_queue.popleft())

            if not batch:
                continue

            try:
                with self._transaction() as connection:
                    initial_trust_key = next(
                        (r.kwargs.get("trusted_public_key") for r in batch if r.kwargs.get("trusted_public_key") is not None),
                        None,
                    )
                    if not self._verify_integrity_locked(connection, initial_trust_key, full_scan=False):
                        raise IntegrityError("Ledger cryptographic integrity validation failed during group commit.")
                    cur_seq, cur_hash = self._read_meta_head_locked(connection)
                    batch_tokens = set(self._verified_trust_tokens)
                    committed_reqs: List[_GroupWriteRequest] = []

                    for idx, req in enumerate(batch):
                        sp = f"gc_item_{idx}"
                        connection.execute(f"SAVEPOINT {sp}")
                        try:
                            req_key = req.kwargs.get("trusted_public_key")
                            if req_key is not None:
                                tok = _trust_key_token(req_key)
                                if tok not in batch_tokens and cur_seq > 0:
                                    if not self._verify_integrity_locked(connection, req_key, full_scan=True):
                                        raise IntegrityError("Ledger trust key verification failed during group commit.")
                                    batch_tokens.add(tok)

                            if req.kind == "append":
                                new_seq, new_hash, tok = self._exec_append_locked(
                                    connection,
                                    head_state=(cur_seq, cur_hash),
                                    update_meta=False,
                                    **req.kwargs,
                                )
                                cur_seq, cur_hash = new_seq, new_hash
                                batch_tokens = {tok} if tok is not None else set()
                                req.result = new_hash
                            elif req.kind == "outcome":
                                kw = {k: v for k, v in req.kwargs.items() if k != "trusted_public_key"}
                                new_seq, new_hash = self._exec_outcome_locked(
                                    connection,
                                    head_state=(cur_seq, cur_hash),
                                    update_meta=False,
                                    **kw,
                                )
                                cur_seq, cur_hash = new_seq, new_hash
                                req.result = new_hash
                            elif req.kind == "action":
                                kw = {k: v for k, v in req.kwargs.items() if k != "trusted_public_key"}
                                new_seq, new_hash = self._exec_action_locked(
                                    connection,
                                    head_state=(cur_seq, cur_hash),
                                    update_meta=False,
                                    **kw,
                                )
                                cur_seq, cur_hash = new_seq, new_hash
                                req.result = new_hash
                            elif req.kind == "coalesced_ro":
                                new_seq, new_hash, dec_hash, out_hash, tok = self._exec_coalesced_ro_locked(
                                    connection,
                                    head_state=(cur_seq, cur_hash),
                                    update_meta=False,
                                    **req.kwargs,
                                )
                                cur_seq, cur_hash = new_seq, new_hash
                                batch_tokens = {tok} if tok is not None else set()
                                req.result = (dec_hash, out_hash)
                            else:
                                raise LedgerError(f"Unknown group write operation: {req.kind}")
                            connection.execute(f"RELEASE SAVEPOINT {sp}")
                            committed_reqs.append(req)
                        except BaseException as item_exc:
                            connection.execute(f"ROLLBACK TO SAVEPOINT {sp}")
                            connection.execute(f"RELEASE SAVEPOINT {sp}")
                            req.error = item_exc

                    if committed_reqs:
                        connection.execute(
                            "UPDATE ledger_meta SET value=? WHERE key='audit_head_hash'", (cur_hash,)
                        )
                        connection.execute(
                            "UPDATE ledger_meta SET value=? WHERE key='audit_head_sequence'", (str(cur_seq),)
                        )
                if committed_reqs:
                    with self._lock:
                        self._finish_write_commit_locked(
                            tip_sequence=cur_seq,
                            tip_hash=cur_hash,
                            trust_tokens=batch_tokens,
                        )
            except BaseException as batch_exc:
                for req in batch:
                    if req.error is None:
                        req.error = batch_exc
            finally:
                for req in batch:
                    req.event.set()

    def append(
        self,
        receipt: Mapping[str, Any] | Any,
        *,
        tenant_id: str = "system1",
        principal_id: str = "engine",
        scope: str = "decision",
        trusted_public_key: Optional[Any] = None,
    ) -> str:
        """Appends a System 1 decision receipt to the tamper-evident audit ledger."""
        if self.durability_mode == "group_commit":
            return str(
                self._submit_group_write(
                    "append",
                    receipt=receipt,
                    tenant_id=tenant_id,
                    principal_id=principal_id,
                    scope=scope,
                    trusted_public_key=trusted_public_key,
                )
            )
        with self._transaction() as connection:
            if trusted_public_key is not None:
                payload = receipt.to_dict() if hasattr(receipt, "to_dict") else dict(receipt)
                if not verify_decision_witness_receipt(payload, public_key=trusted_public_key):
                    raise IntegrityError("Incoming receipt failed signature verification against trust anchor.")
            if not self._verify_integrity_locked(connection, trusted_public_key, full_scan=False):
                raise IntegrityError("Ledger cryptographic integrity validation failed during append.")
            seq, entry_hash, tok = self._exec_append_locked(
                connection,
                receipt=receipt,
                tenant_id=tenant_id,
                principal_id=principal_id,
                scope=scope,
                trusted_public_key=trusted_public_key,
            )
        with self._lock:
            self._finish_write_commit_locked(
                tip_sequence=seq,
                tip_hash=entry_hash,
                trust_tokens={tok} if tok is not None else set(),
            )
        return entry_hash

    def record_decision_receipt(
        self,
        receipt: Mapping[str, Any] | Any,
        *,
        tenant_id: str = "system1",
        principal_id: str = "engine",
        scope: str = "decision",
        trusted_public_key: Optional[Any] = None,
    ) -> str:
        """Appends a System 1 decision receipt to the tamper-evident audit ledger."""
        return self.append(
            receipt,
            tenant_id=tenant_id,
            principal_id=principal_id,
            scope=scope,
            trusted_public_key=trusted_public_key,
        )

    def record_execution_outcome(
        self,
        *,
        action_id: str,
        receipt_digest: str,
        status: str,
        result_payload: Optional[Mapping[str, Any]] = None,
        error_message: Optional[str] = None,
        tenant_id: str = "system1",
        principal_id: str = "engine",
        scope: str = "execution_outcome",
        trusted_public_key: Optional[Any] = None,
    ) -> str:
        """Records an execution outcome linked cryptographically to prior authorization receipt."""
        if self.durability_mode == "group_commit":
            return str(
                self._submit_group_write(
                    "outcome",
                    action_id=action_id,
                    receipt_digest=receipt_digest,
                    status=status,
                    result_payload=result_payload,
                    error_message=error_message,
                    tenant_id=tenant_id,
                    principal_id=principal_id,
                    scope=scope,
                    trusted_public_key=trusted_public_key,
                )
            )
        with self._transaction() as connection:
            if not self._verify_integrity_locked(connection, trusted_public_key, full_scan=False):
                raise IntegrityError("Ledger cryptographic integrity validation failed during outcome recording.")
            seq, entry_hash = self._exec_outcome_locked(
                connection,
                action_id=action_id,
                receipt_digest=receipt_digest,
                status=status,
                result_payload=result_payload,
                error_message=error_message,
                tenant_id=tenant_id,
                principal_id=principal_id,
                scope=scope,
            )
        with self._lock:
            self._finish_write_commit_locked(tip_sequence=seq, tip_hash=entry_hash)
        return entry_hash

    def record_action(
        self,
        action: str,
        proposal: Mapping[str, Any] | Any,
        *,
        action_id: Optional[str] = None,
        tenant_id: str = "system1",
        principal_id: str = "engine",
        scope: str = "action",
        trusted_public_key: Optional[Any] = None,
    ) -> str:
        """Appends an arbitrary action proposal or audit event to the tamper-evident ledger."""
        if self.durability_mode == "group_commit":
            return str(
                self._submit_group_write(
                    "action",
                    action=action,
                    proposal=proposal,
                    action_id=action_id,
                    tenant_id=tenant_id,
                    principal_id=principal_id,
                    scope=scope,
                    trusted_public_key=trusted_public_key,
                )
            )
        with self._transaction() as connection:
            if not self._verify_integrity_locked(connection, trusted_public_key, full_scan=False):
                raise IntegrityError("Ledger cryptographic integrity validation failed during action recording.")
            seq, entry_hash = self._exec_action_locked(
                connection,
                action=action,
                proposal=proposal,
                action_id=action_id,
                tenant_id=tenant_id,
                principal_id=principal_id,
                scope=scope,
            )
        with self._lock:
            self._finish_write_commit_locked(tip_sequence=seq, tip_hash=entry_hash)
        return entry_hash

    def record_coalesced_read_only_execution(
        self,
        receipt: Mapping[str, Any] | Any,
        *,
        reversibility: str,
        action_id: Optional[str] = None,
        receipt_digest: Optional[str] = None,
        status: str = "SUCCEEDED",
        result_payload: Optional[Mapping[str, Any]] = None,
        error_message: Optional[str] = None,
        tenant_id: str = "system1",
        principal_id: str = "engine",
        scope: str = "decision",
        outcome_scope: str = "execution_outcome",
        trusted_public_key: Optional[Any] = None,
    ) -> Tuple[str, str]:
        """Coalesces READ_ONLY decision receipt and execution outcome into one atomic SQLite commit."""
        if self.durability_mode == "group_commit":
            res = self._submit_group_write(
                "coalesced_ro",
                receipt=receipt,
                reversibility=reversibility,
                action_id=action_id,
                receipt_digest=receipt_digest,
                status=status,
                result_payload=result_payload,
                error_message=error_message,
                tenant_id=tenant_id,
                principal_id=principal_id,
                scope=scope,
                outcome_scope=outcome_scope,
                trusted_public_key=trusted_public_key,
            )
            return str(res[0]), str(res[1])
        with self._transaction() as connection:
            if not self._verify_integrity_locked(connection, trusted_public_key, full_scan=False):
                raise IntegrityError("Ledger cryptographic integrity validation failed during coalesced recording.")
            seq, tip_hash, dec_hash, out_hash, tok = self._exec_coalesced_ro_locked(
                connection,
                receipt=receipt,
                reversibility=reversibility,
                action_id=action_id,
                receipt_digest=receipt_digest,
                status=status,
                result_payload=result_payload,
                error_message=error_message,
                tenant_id=tenant_id,
                principal_id=principal_id,
                scope=scope,
                outcome_scope=outcome_scope,
                trusted_public_key=trusted_public_key,
            )
        with self._lock:
            self._finish_write_commit_locked(
                tip_sequence=seq,
                tip_hash=tip_hash,
                trust_tokens={tok} if tok is not None else set(),
            )
        return dec_hash, out_hash

    def verify_integrity(self, trusted_public_key: Optional[Any] = None) -> bool:
        """Verifies the complete cryptographic hash chain and meta head, and validates entry signatures when trusted_public_key is supplied."""
        with self._lock:
            self._connection.execute("BEGIN")
            try:
                return self._verify_integrity_locked(self._connection, trusted_public_key, full_scan=True)
            finally:
                self._connection.rollback()

    def verify_receipt_record(
        self, receipt: Mapping[str, Any] | Any, trusted_public_key: Optional[Any] = None,
    ) -> bool:
        """Check exact receipt inclusion and the complete chain in one snapshot."""
        payload = receipt.to_dict() if hasattr(receipt, "to_dict") else dict(receipt)
        record_id = payload.get("ledger_record_id")
        if not record_id:
            return False
        try:
            valid = (verify_decision_witness_receipt(payload, public_key=trusted_public_key)
                     if trusted_public_key is not None else check_decision_receipt_integrity(payload))
            if not valid:
                return False
            with self._lock:
                self._connection.execute("BEGIN")
                try:
                    if not self._verify_integrity_locked(self._connection, trusted_public_key, full_scan=False):
                        return False
                    row = self._connection.execute(
                        "SELECT payload_json FROM audit_entries WHERE entry_hash=? AND event_type='system1_decision'",
                        (record_id,),
                    ).fetchone()
                    if row is None:
                        return False
                    stored = json.loads(row["payload_json"]).get("receipt", {})
                    return (stored.get("decision_id") == payload.get("decision_id")
                            and compute_receipt_digest(stored) == compute_receipt_digest(payload))
                finally:
                    self._connection.rollback()
        except (TypeError, ValueError, sqlite3.Error):
            return False



    def entries(self) -> List[Dict[str, Any]]:
        """Returns all audit entries ordered by sequence."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM audit_entries ORDER BY sequence ASC"
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                try:
                    d["payload"] = json.loads(d.get("payload_json", "{}"))
                except Exception:
                    pass
                results.append(d)
            return results

    def records(self) -> List[Dict[str, Any]]:
        """Alias for entries()."""
        return self.entries()


__all__ = [
    "VALID_DURABILITY_MODES",
    "ActionLedger",
    "IntegrityError",
    "LedgerError",
    "LedgerWriteError",
]
