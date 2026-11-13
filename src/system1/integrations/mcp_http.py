"""Runnable MCP Streamable HTTP (2025-03-26) guard server for System 1: Instinct.

Serves JSON-RPC 2.0 MCP traffic over ``http.server.ThreadingHTTPServer`` with:
- ``POST /mcp``: single/batch JSON-RPC requests, notifications (202), JSON or SSE
- ``GET /healthz``: runtime status, active sessions, ledger durability, signer key id
- ``DELETE /mcp``: explicit ``Mcp-Session-Id`` teardown

Security and resource controls:
- Strict ``Origin`` allowlist validation (HTTP 403 on disallowed browser origins)
- Constant-time ``Authorization: Bearer`` check via ``hmac.compare_digest`` (HTTP 401)
- Bounded request bodies via ``max_body_bytes`` (HTTP 413 Payload Too Large)
- Explicit ``Mcp-Session-Id`` session lifecycle (400 missing / 404 unknown)
- Optional upstream MCP stdio subprocess forwarding via ``UpstreamMCPProcess``
- Shared thread-safe ``SystemOneMCPProxy`` + ``SystemOneGuardHook`` + ``ActionLedger`` + Ed25519 receipts
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import hmac
import http.server
import itertools
import json
import os
import secrets
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from system1.guard import (
    DecisionOutcome,
    GuardInterceptionResult,
    PolicyEngine,
    PolicyRule,
    SystemOneGuardHook,
)
from system1.integrations.mcp import SystemOneMCPBlockedError, SystemOneMCPProxy
from system1.integrations.mcp_stdio import (
    DEFAULT_MAX_LINE_BYTES,
    UpstreamError,
    UpstreamMCPProcess,
    UpstreamRPCError,
    _load_or_create_signing_key,
    _load_policy_file,
)
from system1.ledger import ActionLedger
from system1.receipt import public_key_bytes


MCP_PROTOCOL_VERSION = "2025-03-26"
DEFAULT_MAX_BODY_BYTES = 1_048_576  # 1 MiB


@dataclasses.dataclass
class MCPStreamableHTTPConfig:
    """Configuration for the MCP Streamable HTTP Guard Daemon."""

    host: str = "127.0.0.1"
    port: int = 8765
    bearer_token: Optional[str] = None
    allowed_origins: Tuple[str, ...] = ("http://127.0.0.1", "http://localhost")
    require_origin: bool = False
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    tenant_id: str = "tenant_mcp_http"
    principal_id: str = "agent_mcp_http"
    alpha: float = 0.20
    min_confidence: float = 0.50
    enforcement_profile: bool = False
    model_path: Optional[str] = None
    ledger_path: Optional[str] = ":memory:"
    durability_mode: str = "process_wal"
    upstream_cmd: Optional[Sequence[str]] = None
    upstream_timeout_s: float = 30.0
    shutdown_timeout_s: float = 3.0
    server_name: str = "system1-mcp-guard-http"
    server_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if int(self.max_body_bytes) <= 0:
            raise ValueError(f"max_body_bytes must be positive, got {self.max_body_bytes}")
        if int(self.port) < 0 or int(self.port) > 65535:
            raise ValueError(f"port must be in [0, 65535], got {self.port}")
        self.allowed_origins = tuple(self.allowed_origins)


@dataclasses.dataclass
class _SessionRecord:
    session_id: str
    created_at: float
    last_seen_at: float
    client_info: Dict[str, Any]
    request_count: int = 0


class _ThreadSafeTrackingMCPProxy(SystemOneMCPProxy):
    """Thread-safe ``SystemOneMCPProxy`` capturing per-thread interception receipts."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._local = threading.local()
        self._eval_lock = threading.RLock()

    def evaluate_mcp_call(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        executor: Optional[Callable[[str, Mapping[str, Any]], Any]] = None,
        client_session_id: Optional[str] = None,
        context_prompt: Optional[str] = None,
    ) -> GuardInterceptionResult:
        with self._eval_lock:
            res = super().evaluate_mcp_call(
                tool_name,
                arguments,
                executor=executor,
                client_session_id=client_session_id,
                context_prompt=context_prompt,
            )
        self._local.last_interception = res
        return res

    def pop_last_interception(self) -> Optional[GuardInterceptionResult]:
        res = getattr(self._local, "last_interception", None)
        self._local.last_interception = None
        return res


def default_tool_registry() -> Dict[str, Dict[str, Any]]:
    """Default local tool catalog and executors for standalone mode or testing."""
    return {
        "search_docs": {
            "description": "Read-only search over approved documentation index.",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            "handler": lambda args: {"matches": [f"doc:{args.get('query', '')}"], "ok": True},
        },
        "read_file": {
            "description": "Read-only file inspection inside permitted workspace.",
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            "handler": lambda args: {"path": str(args.get("path", "")), "content": "ok"},
        },
        "delete_records": {
            "description": "Destructive database record deletion requiring policy clearance.",
            "inputSchema": {
                "type": "object",
                "properties": {"table": {"type": "string"}},
                "required": ["table"],
            },
            "handler": lambda args: {"deleted_from": str(args.get("table", "")), "ok": True},
        },
        "ambiguous_action": {
            "description": "Action whose prompt intentionally hits conformal ambiguity.",
            "inputSchema": {
                "type": "object",
                "properties": {"payload": {"type": "string"}},
                "required": ["payload"],
            },
            "handler": lambda args: {"executed": True, "payload": args.get("payload", "")},
        },
    }


class MCPGuardRuntime:
    """Shared thread-safe runtime behind ``MCPStreamableHTTPServer``."""

    def __init__(
        self,
        config: MCPStreamableHTTPConfig,
        *,
        guard: Optional[SystemOneGuardHook] = None,
        policy_engine: Optional[PolicyEngine] = None,
        ledger: Optional[ActionLedger] = None,
        signing_key: Optional[Ed25519PrivateKey] = None,
        tools: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        self.config = config
        self.started_at = time.time()
        self.signing_key = signing_key or Ed25519PrivateKey.generate()
        pub_raw = public_key_bytes(self.signing_key.public_key())
        self.signer_pubkey = base64.b64encode(pub_raw).decode("ascii")
        self.signer_fp = hashlib.sha256(pub_raw).hexdigest()

        self._owns_ledger = False
        if ledger is not None:
            self.ledger: Optional[ActionLedger] = ledger
        elif config.ledger_path is not None:
            self.ledger = ActionLedger(
                config.ledger_path,
                enforce_durability=config.enforcement_profile,
                durability_mode=config.durability_mode,
            )
            self._owns_ledger = True
        else:
            self.ledger = None

        if guard is not None:
            self.guard = guard
        else:
            eff_policy = policy_engine or PolicyEngine(
                rules=[
                    PolicyRule(
                        rule_id="deny_destructive_delete",
                        action_pattern=r"^(delete_|drop_)",
                        outcome=DecisionOutcome.DENY,
                        reason="Destructive deletion blocked by System 1 policy rule",
                    ),
                ]
            )
            self.guard = SystemOneGuardHook(
                policy_engine=eff_policy,
                signing_key=self.signing_key,
                ledger=self.ledger,
                alpha=config.alpha,
                min_confidence=config.min_confidence,
                fail_closed_ledger=(self.ledger is not None),
                enforcement_profile=config.enforcement_profile,
            )

        self.proxy = _ThreadSafeTrackingMCPProxy(
            guard=self.guard,
            ledger=self.ledger,
            alpha=config.alpha,
            min_confidence=config.min_confidence,
            tenant_id=config.tenant_id,
            principal_id=config.principal_id,
            model_path=config.model_path,
        )

        self.tools: Dict[str, Dict[str, Any]] = (
            {k: dict(v) for k, v in tools.items()} if tools is not None else default_tool_registry()
        )
        self._sessions: Dict[str, _SessionRecord] = {}
        self._session_lock = threading.RLock()

        self._upstream_seq = itertools.count(1)
        self._upstream_init_lock = threading.Lock()
        self._upstream_initialized = False
        self.upstream: Optional[UpstreamMCPProcess] = None
        if config.upstream_cmd:
            self.upstream = UpstreamMCPProcess(
                config.upstream_cmd,
                max_line_bytes=max(DEFAULT_MAX_LINE_BYTES, config.max_body_bytes),
                shutdown_timeout_s=config.shutdown_timeout_s,
            )

    def _next_upstream_id(self) -> str:
        return f"s1http_{next(self._upstream_seq)}"

    def _ensure_upstream_initialized(self, init_params: Mapping[str, Any]) -> None:
        if self.upstream is None:
            return
        with self._upstream_init_lock:
            if self._upstream_initialized:
                return
            resp = self.upstream.request(
                {
                    "jsonrpc": "2.0",
                    "id": self._next_upstream_id(),
                    "method": "initialize",
                    "params": dict(init_params),
                },
                timeout_s=self.config.upstream_timeout_s,
            )
            if "error" in resp:
                err = resp["error"]
                raise UpstreamRPCError(
                    int(err.get("code", -32603)),
                    str(err.get("message", "Upstream initialize error")),
                    err.get("data"),
                )
            self.upstream.send_notification(
                {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
            )
            self._upstream_initialized = True

    def is_origin_allowed(self, origin: Optional[str]) -> bool:
        if origin is None:
            return not self.config.require_origin
        norm = origin.strip().rstrip("/")
        if not norm:
            return not self.config.require_origin
        for allowed in self.config.allowed_origins:
            a = allowed.strip().rstrip("/")
            if norm == a:
                return True
            if norm.startswith(a + ":"):
                port_part = norm[len(a) + 1 :]
                if port_part.isdigit():
                    return True
        return False

    def verify_bearer(self, auth_header: Optional[str]) -> bool:
        expected = self.config.bearer_token
        if not expected:
            return True
        if not auth_header:
            return False
        parts = auth_header.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return False
        provided = parts[1].strip()
        return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))

    def create_session(self, client_info: Mapping[str, Any]) -> str:
        sid = f"s1mcp_{secrets.token_urlsafe(24)}"
        now = time.time()
        with self._session_lock:
            self._sessions[sid] = _SessionRecord(
                session_id=sid,
                created_at=now,
                last_seen_at=now,
                client_info=dict(client_info),
                request_count=1,
            )
        return sid

    def validate_session(self, sid: str) -> bool:
        with self._session_lock:
            rec = self._sessions.get(sid)
            if rec is None:
                return False
            rec.last_seen_at = time.time()
            rec.request_count += 1
            return True

    def terminate_session(self, sid: str) -> bool:
        with self._session_lock:
            return self._sessions.pop(sid, None) is not None

    def active_session_count(self) -> int:
        with self._session_lock:
            return len(self._sessions)

    @staticmethod
    def _rpc_error(
        req_id: Any,
        code: int,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        err: Dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return {"jsonrpc": "2.0", "id": req_id, "error": err}

    def _enrich_allowed_result(
        self,
        result: Dict[str, Any],
        interception: Optional[GuardInterceptionResult],
        session_id: Optional[str],
    ) -> Dict[str, Any]:
        meta = result.setdefault("_meta", {}).setdefault("system1", {})
        if session_id:
            meta["session_id"] = session_id
        if interception is None:
            return result
        if interception.policy_decision is not None:
            meta["action_id"] = interception.policy_decision.action_id
            meta["policy_decision_id"] = interception.policy_decision.decision_id
        receipt = getattr(interception, "receipt", None) or (
            interception.decision_result.receipt if interception.decision_result is not None else None
        )
        if receipt is not None:
            meta["decision_id"] = receipt.decision_id
            meta["ledger_record_id"] = receipt.ledger_record_id
            meta["signer_public_key"] = receipt.signer_public_key
            if not meta.get("digest"):
                meta["digest"] = receipt.compute_digest()
            if receipt.envelope is not None:
                meta["signature"] = receipt.envelope.signature
                meta["signer_fingerprint"] = receipt.envelope.signer_fingerprint
        return result

    def _enrich_blocked_data(
        self,
        interception: Optional[GuardInterceptionResult],
        base_data: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = dict(base_data or {})
        if interception is None:
            return data
        data["outcome"] = interception.outcome.value
        if interception.policy_decision is not None:
            data["action_id"] = interception.policy_decision.action_id
            data["policy_decision"] = interception.policy_decision.outcome.value
            data["reason"] = interception.policy_decision.reason
        if interception.decision_result is not None:
            dr = interception.decision_result
            data["is_ambiguous"] = bool(dr.is_ambiguous)
            data["conformal_sets"] = {k: list(v) for k, v in dr.conformal_sets.items()}
            data["confidences"] = {k: float(v) for k, v in dr.confidences.items()}
            data["escalated_fields"] = list(getattr(dr, "escalated_fields", []) or [])
        b_receipt = getattr(interception, "receipt", None) or (
            interception.decision_result.receipt if interception.decision_result is not None else None
        )
        if b_receipt is not None:
            data["decision_id"] = b_receipt.decision_id
            data["ledger_record_id"] = b_receipt.ledger_record_id
            if b_receipt.envelope is not None:
                data["signature"] = b_receipt.envelope.signature
                data["signer_fingerprint"] = b_receipt.envelope.signer_fingerprint
        return data

    def _execute_tool(self, tool_name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        if self.upstream is not None:
            self._ensure_upstream_initialized(
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": self.config.server_name, "version": self.config.server_version},
                }
            )
            resp = self.upstream.request(
                {
                    "jsonrpc": "2.0",
                    "id": self._next_upstream_id(),
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": dict(arguments)},
                },
                timeout_s=self.config.upstream_timeout_s,
            )
            if "error" in resp:
                err = resp["error"]
                raise UpstreamRPCError(
                    int(err.get("code", -32603)),
                    str(err.get("message", "Upstream tool error")),
                    err.get("data"),
                )
            res = resp.get("result")
            return dict(res) if isinstance(res, Mapping) else {"result": res}

        entry = self.tools.get(tool_name)
        if entry is None:
            raise KeyError(f"Unknown tool: {tool_name}")
        handler = entry["handler"]
        out = handler(arguments)
        if isinstance(out, dict) and "content" in out:
            return dict(out)
        return {
            "content": [{"type": "text", "text": json.dumps(out, sort_keys=True)}],
            "structuredContent": out,
        }

    def dispatch_rpc_item(
        self,
        item: Mapping[str, Any],
        *,
        session_id: Optional[str],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Process a single JSON-RPC 2.0 request/notification item.

        Returns ``(response_dict_or_None, newly_created_session_id_or_None)``.
        """
        is_notification = "id" not in item
        req_id = item.get("id")
        if item.get("jsonrpc") != "2.0" or not isinstance(item.get("method"), str):
            if is_notification:
                return self._rpc_error(None, -32600, "Invalid Request"), None
            return self._rpc_error(req_id, -32600, "Invalid Request"), None

        method: str = item["method"]
        params = item.get("params") or {}
        if not isinstance(params, Mapping):
            return self._rpc_error(req_id, -32602, "Invalid params: expected object"), None

        if is_notification:
            if method == "tools/call":
                return (
                    self._rpc_error(
                        None,
                        -32600,
                        "Invalid Request: tools/call cannot be sent as a JSON-RPC notification",
                    ),
                    None,
                )
            return None, None

        if method == "initialize":
            if self.upstream is not None:
                try:
                    self._ensure_upstream_initialized(params)
                except UpstreamError as exc:
                    return self._rpc_error(req_id, -32603, f"Upstream initialize error: {exc}"), None
            client_info = params.get("clientInfo") if isinstance(params.get("clientInfo"), Mapping) else {}
            new_sid = self.create_session(client_info)
            result = {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {"listChanged": False},
                    "experimental": {
                        "system1Guard": {
                            "signerFingerprint": self.signer_fp,
                            "alpha": self.config.alpha,
                            "minConfidence": self.config.min_confidence,
                        }
                    },
                },
                "serverInfo": {
                    "name": self.config.server_name,
                    "version": self.config.server_version,
                },
            }
            return {"jsonrpc": "2.0", "id": req_id, "result": result}, new_sid

        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}, None

        if method == "tools/list":
            if self.upstream is not None:
                try:
                    self._ensure_upstream_initialized(
                        {
                            "protocolVersion": MCP_PROTOCOL_VERSION,
                            "capabilities": {},
                            "clientInfo": {
                                "name": self.config.server_name,
                                "version": self.config.server_version,
                            },
                        }
                    )
                    resp = self.upstream.request(
                        {
                            "jsonrpc": "2.0",
                            "id": self._next_upstream_id(),
                            "method": "tools/list",
                            "params": dict(params),
                        },
                        timeout_s=self.config.upstream_timeout_s,
                    )
                    if "error" in resp:
                        err = resp["error"]
                        return (
                            self._rpc_error(
                                req_id,
                                int(err.get("code", -32603)),
                                str(err.get("message", "Upstream error")),
                                err.get("data"),
                            ),
                            None,
                        )
                    return {"jsonrpc": "2.0", "id": req_id, "result": resp.get("result", {"tools": []})}, None
                except UpstreamError as exc:
                    return self._rpc_error(req_id, -32603, f"Upstream tools/list failed: {exc}"), None
            tool_defs = [
                {
                    "name": tname,
                    "description": tinfo.get("description", ""),
                    "inputSchema": tinfo.get("inputSchema", {"type": "object"}),
                }
                for tname, tinfo in sorted(self.tools.items())
            ]
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tool_defs}}, None

        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(tool_name, str) or not tool_name:
                return self._rpc_error(req_id, -32602, "Invalid params: missing tool name"), None
            if not isinstance(arguments, Mapping):
                return self._rpc_error(req_id, -32602, "Invalid params: arguments must be an object"), None
            if self.upstream is None and tool_name not in self.tools:
                return self._rpc_error(req_id, -32601, f"Tool not found: {tool_name}"), None

            self.proxy.pop_last_interception()
            try:
                resp = self.proxy.handle_call(
                    dict(item),
                    lambda t_name, t_args: self._execute_tool(t_name, t_args),
                    client_session_id=session_id,
                    context_prompt=params.get("_context") if isinstance(params.get("_context"), str) else None,
                )
            except SystemOneMCPBlockedError as exc:
                interception = self.proxy.pop_last_interception()
                err_data = self._enrich_blocked_data(interception, {"blocked_reason": str(exc)})
                return (
                    self._rpc_error(req_id, -32000, f"System 1 Guard blocked tools/call: {exc}", err_data),
                    None,
                )
            except UpstreamRPCError as exc:
                self.proxy.pop_last_interception()
                return self._rpc_error(req_id, exc.code, exc.rpc_message, exc.data), None
            except UpstreamError as exc:
                self.proxy.pop_last_interception()
                return self._rpc_error(req_id, -32603, f"Upstream tools/call failed: {exc}"), None

            interception = self.proxy.pop_last_interception()
            if "error" in resp:
                err_obj = dict(resp["error"])
                err_obj["data"] = self._enrich_blocked_data(interception, err_obj.get("data"))
                return {"jsonrpc": "2.0", "id": req_id, "error": err_obj}, None

            result_obj = dict(resp.get("result", {}))
            self._enrich_allowed_result(result_obj, interception, session_id)
            return {"jsonrpc": "2.0", "id": req_id, "result": result_obj}, None

        return self._rpc_error(req_id, -32601, f"Method not found: {method}"), None

    def close(self) -> None:
        if self.upstream is not None:
            self.upstream.close()
        if self._owns_ledger and self.ledger is not None:
            self.ledger.close()


def _wants_sse(accept_header: Optional[str]) -> bool:
    """Return True when client explicitly requests SSE framing (``text/event-stream``)."""
    if not accept_header:
        return False
    lower = accept_header.lower()
    if "text/event-stream" not in lower:
        return False
    if "application/json" not in lower:
        return True
    parts = [p.strip() for p in lower.split(",") if p.strip()]
    for p in parts:
        if p.startswith("text/event-stream"):
            return True
        if p.startswith("application/json"):
            return False
    return False


class _MCPStreamableHTTPHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True

    def log_message(self, format: str, *args: Any) -> None:
        return

    @property
    def runtime(self) -> MCPGuardRuntime:
        return self.server.runtime  # type: ignore[attr-defined]

    def _send_json(
        self,
        status_code: int,
        payload: Any,
        *,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_empty(
        self,
        status_code: int,
        *,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.send_response(status_code)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()

    def _send_sse_events(
        self,
        events: Sequence[Any],
        *,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        chunks = []
        for ev in events:
            payload_str = json.dumps(ev, separators=(",", ":"), allow_nan=False)
            chunks.append(f"event: message\ndata: {payload_str}\n\n".encode("utf-8"))
        body = b"".join(chunks)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _check_origin_and_auth(self) -> bool:
        origin = self.headers.get("Origin")
        if not self.runtime.is_origin_allowed(origin):
            self._send_json(
                403,
                {"error": "Forbidden", "reason": f"Disallowed Origin header: {origin!r}"},
            )
            return False
        if not self.runtime.verify_bearer(self.headers.get("Authorization")):
            self._send_json(
                401,
                {"error": "Unauthorized", "reason": "Invalid or missing Bearer token"},
                extra_headers={"WWW-Authenticate": "Bearer"},
            )
            return False
        return True

    def _read_bounded_body(self) -> Optional[bytes]:
        max_bytes = self.runtime.config.max_body_bytes
        cl_raw = self.headers.get("Content-Length")
        if cl_raw is not None:
            try:
                content_length = int(cl_raw)
            except ValueError:
                self._send_json(400, {"error": "Invalid Content-Length header"})
                return None
            if content_length < 0:
                self._send_json(400, {"error": "Negative Content-Length header"})
                return None
            if content_length > max_bytes:
                self.close_connection = True
                self._send_json(
                    413,
                    {
                        "error": "Payload Too Large",
                        "max_body_bytes": max_bytes,
                        "content_length": content_length,
                    },
                )
                return None
            raw = self.rfile.read(content_length)
            if len(raw) != content_length:
                self._send_json(400, {"error": "Incomplete request body"})
                return None
            return raw

        te = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in te:
            buf = bytearray()
            while True:
                size_line = self.rfile.readline(64)
                if not size_line:
                    self._send_json(400, {"error": "Malformed chunked body"})
                    return None
                try:
                    chunk_size = int(size_line.split(b";", 1)[0].strip(), 16)
                except ValueError:
                    self._send_json(400, {"error": "Malformed chunk size"})
                    return None
                if chunk_size == 0:
                    while True:
                        trailer = self.rfile.readline(1024)
                        if not trailer or trailer in (b"\r\n", b"\n"):
                            break
                    break
                if len(buf) + chunk_size > max_bytes:
                    self.close_connection = True
                    self._send_json(413, {"error": "Payload Too Large", "max_body_bytes": max_bytes})
                    return None
                chunk = self.rfile.read(chunk_size)
                if len(chunk) != chunk_size:
                    self._send_json(400, {"error": "Incomplete chunked body"})
                    return None
                buf.extend(chunk)
                self.rfile.read(2)
            return bytes(buf)

        return b""

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] == "/healthz":
            if not self.runtime.is_origin_allowed(self.headers.get("Origin")):
                self._send_json(403, {"error": "Forbidden", "reason": "Disallowed Origin"})
                return
            self._send_json(
                200,
                {
                    "status": "ok",
                    "transport": "streamable-http",
                    "protocol_version": MCP_PROTOCOL_VERSION,
                    "active_sessions": self.runtime.active_session_count(),
                    "ledger_enabled": self.runtime.ledger is not None,
                    "durability_mode": self.runtime.config.durability_mode,
                    "signer_fingerprint": self.runtime.signer_fp,
                    "uptime_s": round(time.time() - self.runtime.started_at, 3),
                },
            )
            return
        self._send_json(404, {"error": "Not Found"})

    def do_DELETE(self) -> None:
        if self.path.split("?", 1)[0] != "/mcp":
            self._send_json(404, {"error": "Not Found"})
            return
        if not self._check_origin_and_auth():
            return
        sid = self.headers.get("Mcp-Session-Id")
        if not sid:
            self._send_json(400, {"error": "Missing Mcp-Session-Id header"})
            return
        if not self.runtime.terminate_session(sid):
            self._send_json(404, {"error": "Session not found or already terminated"})
            return
        self._send_json(200, {"terminated": True, "session_id": sid})

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/mcp":
            self._send_json(404, {"error": "Not Found"})
            return
        if not self._check_origin_and_auth():
            return

        ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if ctype and ctype != "application/json":
            self._send_json(415, {"error": "Unsupported Media Type; expected application/json"})
            return

        raw_body = self._read_bounded_body()
        if raw_body is None:
            return
        if not raw_body:
            self._send_json(
                400,
                self.runtime._rpc_error(None, -32600, "Invalid Request: empty body"),
            )
            return

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json(
                400,
                self.runtime._rpc_error(None, -32700, f"Parse error: {exc}"),
            )
            return

        is_batch = isinstance(payload, list)
        if is_batch:
            if len(payload) == 0:
                self._send_json(
                    400,
                    self.runtime._rpc_error(None, -32600, "Invalid Request: empty JSON-RPC batch"),
                )
                return
            items = payload
        elif isinstance(payload, dict):
            items = [payload]
        else:
            self._send_json(
                400,
                self.runtime._rpc_error(None, -32600, "Invalid Request: expected JSON object or array"),
            )
            return

        is_pure_initialize = (
            not is_batch
            and isinstance(items[0], Mapping)
            and items[0].get("method") == "initialize"
            and "id" in items[0]
        )
        sid = self.headers.get("Mcp-Session-Id")
        if not is_pure_initialize:
            if not sid:
                self._send_json(
                    400,
                    {"error": "Missing Mcp-Session-Id header; call initialize first"},
                )
                return
            if not self.runtime.validate_session(sid):
                self._send_json(
                    404,
                    {"error": "Unknown or expired Mcp-Session-Id"},
                )
                return

        responses: List[Dict[str, Any]] = []
        created_sid: Optional[str] = None
        for item in items:
            if not isinstance(item, Mapping):
                responses.append(self.runtime._rpc_error(None, -32600, "Invalid Request: item not an object"))
                continue
            resp, new_sid = self.runtime.dispatch_rpc_item(item, session_id=sid or created_sid)
            if new_sid is not None:
                created_sid = new_sid
            if resp is not None:
                responses.append(resp)

        extra_headers: Dict[str, str] = {}
        if created_sid is not None:
            extra_headers["Mcp-Session-Id"] = created_sid
        elif sid:
            extra_headers["Mcp-Session-Id"] = sid

        if not responses:
            self._send_empty(202, extra_headers=extra_headers)
            return

        if len(responses) == 1 and responses[0].get("id") is None and responses[0].get("error", {}).get("code") == -32600:
            self._send_json(400, responses[0], extra_headers=extra_headers)
            return

        use_sse = _wants_sse(self.headers.get("Accept"))
        if use_sse:
            self._send_sse_events(responses if is_batch else [responses[0]], extra_headers=extra_headers)
            return

        self._send_json(200, responses if is_batch else responses[0], extra_headers=extra_headers)


class MCPStreamableHTTPServer(http.server.ThreadingHTTPServer):
    """Loopback multi-threaded Streamable HTTP MCP Guard Daemon."""

    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 128

    def __init__(
        self,
        config: Optional[MCPStreamableHTTPConfig] = None,
        *,
        guard: Optional[SystemOneGuardHook] = None,
        policy_engine: Optional[PolicyEngine] = None,
        ledger: Optional[ActionLedger] = None,
        signing_key: Optional[Ed25519PrivateKey] = None,
        tools: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        cfg = config or MCPStreamableHTTPConfig()
        self.runtime = MCPGuardRuntime(
            cfg,
            guard=guard,
            policy_engine=policy_engine,
            ledger=ledger,
            signing_key=signing_key,
            tools=tools,
        )
        super().__init__((cfg.host, cfg.port), _MCPStreamableHTTPHandler)
        self._thread: Optional[threading.Thread] = None

    @property
    def bound_port(self) -> int:
        return int(self.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://{self.server_address[0]}:{self.bound_port}"

    def start_background(self) -> None:
        self._thread = threading.Thread(
            target=self.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
            name="mcp-streamable-http",
        )
        self._thread.start()

    def stop(self) -> None:
        self.shutdown()
        self.server_close()
        self.runtime.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def run_mcp_http_from_args(
    args: argparse.Namespace,
    *,
    error_fn: Callable[[str], Any],
) -> int:
    """Construct an ``MCPStreamableHTTPServer`` from parsed CLI args and serve until interrupted."""
    cmd = list(getattr(args, "upstream_cmd", []) or [])
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    signing_key = _load_or_create_signing_key(getattr(args, "identity_dir", None))
    policy = _load_policy_file(getattr(args, "policy", None))
    ledger_path = getattr(args, "ledger", None)
    durability_mode = str(getattr(args, "durability_mode", "process_wal"))
    ledger = ActionLedger(ledger_path, require_durable=True, durability_mode=durability_mode) if ledger_path else None

    if getattr(args, "enforcement_profile", False) and (signing_key is None or ledger is None):
        if ledger is not None:
            ledger.close()
        error_fn("--enforcement-profile requires both --identity-dir and --ledger")

    bearer_token = getattr(args, "bearer_token", None) or os.environ.get("SYSTEM1_MCP_BEARER_TOKEN")
    raw_origins = getattr(args, "allowed_origins", None)
    allowed_origins = (
        tuple(raw_origins)
        if raw_origins
        else ("http://127.0.0.1", "http://localhost")
    )
    host = str(getattr(args, "host", "127.0.0.1"))
    raw_port = getattr(args, "port", 8765)
    port = 8765 if raw_port is None else int(raw_port)
    # Match serve default cleanly when caller uses serve defaults without specifying --port explicitly
    if port == 50051 and str(getattr(args, "transport", "")) == "mcp-http":
        port = 8765

    config = MCPStreamableHTTPConfig(
        host=host,
        port=port,
        bearer_token=bearer_token,
        allowed_origins=allowed_origins,
        require_origin=bool(getattr(args, "require_origin", False)),
        max_body_bytes=int(getattr(args, "max_line_bytes", DEFAULT_MAX_BODY_BYTES)),
        tenant_id=str(getattr(args, "tenant_id", "tenant_mcp_http")),
        principal_id=str(getattr(args, "principal_id", "agent_mcp_http")),
        alpha=float(getattr(args, "alpha", 0.20)),
        min_confidence=float(getattr(args, "min_confidence", 0.50)),
        enforcement_profile=bool(getattr(args, "enforcement_profile", False)),
        model_path=getattr(args, "model", None),
        ledger_path=ledger_path,
        durability_mode=durability_mode,
        upstream_cmd=cmd or None,
        upstream_timeout_s=float(getattr(args, "upstream_timeout", 30.0)),
        shutdown_timeout_s=float(getattr(args, "shutdown_timeout", 3.0)),
    )

    server = MCPStreamableHTTPServer(
        config,
        policy_engine=policy,
        ledger=ledger,
        signing_key=signing_key,
    )
    print(
        json.dumps(
            {
                "event": "mcp_http_listening",
                "url": f"{server.base_url}/mcp",
                "healthz": f"{server.base_url}/healthz",
                "protocol_version": MCP_PROTOCOL_VERSION,
                "durability_mode": durability_mode,
                "signer_fingerprint": server.runtime.signer_fp,
            }
        ),
        file=sys.stderr,
        flush=True,
    )
    try:
        server.serve_forever()
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
        server.runtime.close()
        if ledger is not None:
            ledger.close()


__all__ = [
    "DEFAULT_MAX_BODY_BYTES",
    "MCPGuardRuntime",
    "MCPStreamableHTTPConfig",
    "MCPStreamableHTTPServer",
    "MCP_PROTOCOL_VERSION",
    "default_tool_registry",
    "run_mcp_http_from_args",
]
