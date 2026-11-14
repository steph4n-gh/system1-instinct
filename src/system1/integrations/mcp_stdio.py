"""Runnable MCP JSON-RPC 2.0 stdio proxy/guard for System 1: Instinct.

Spawns an upstream MCP server subprocess over newline-delimited JSON-RPC 2.0
``stdio`` transport, forwards lifecycle and discovery methods (``initialize``,
``notifications/initialized``, ``tools/list``, ``ping``), and enforces
``SystemOneMCPProxy`` + ``SystemOneGuardHook`` policy, conformal abstention,
and Ed25519 ``ActionLedger`` witness receipts on every ``tools/call``.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, BinaryIO, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from system1.guard import (
    DecisionOutcome,
    GuardInterceptionResult,
    PolicyEngine,
    PolicyRule,
    SystemOneGuardHook,
)
from system1.integrations.mcp import SystemOneMCPProxy
from system1.ledger import ActionLedger
from system1.receipt import load_private_key, save_keypair


DEFAULT_MAX_LINE_BYTES = 1_048_576  # 1 MiB
_DRAIN_CHUNK_BYTES = 65_536


class UpstreamError(RuntimeError):
    """Base error for upstream MCP server communication failures."""


class UpstreamTimeoutError(UpstreamError):
    """Raised when the upstream MCP subprocess fails to respond within timeout."""


class UpstreamExitedError(UpstreamError):
    """Raised when the upstream MCP subprocess terminates unexpectedly."""

    def __init__(self, message: str, returncode: Optional[int] = None) -> None:
        super().__init__(message)
        self.returncode = returncode


class UpstreamProtocolError(UpstreamError):
    """Raised when the upstream MCP subprocess emits invalid or oversized frames."""


class UpstreamRPCError(UpstreamError):
    """Raised when the upstream MCP subprocess returns a JSON-RPC error object."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.rpc_message = message
        self.data = data


@dataclasses.dataclass
class MCPStdioProxyConfig:
    """Configuration for the stdio MCP proxy/guard."""

    upstream_cmd: Sequence[str]
    max_line_bytes: int = DEFAULT_MAX_LINE_BYTES
    upstream_timeout_s: float = 30.0
    shutdown_timeout_s: float = 3.0
    tenant_id: str = "tenant_mcp_default"
    principal_id: str = "agent_mcp_client"
    alpha: float = 0.20
    min_confidence: float = 0.50
    enforcement_profile: bool = False
    model_path: Optional[str] = None
    cwd: Optional[str] = None
    env: Optional[Mapping[str, str]] = None


MCPProxyConfig = MCPStdioProxyConfig


def read_bounded_jsonrpc_line(stream: BinaryIO, max_bytes: int) -> Tuple[Optional[bytes], bool]:
    """Read a single newline-delimited JSON-RPC frame bounded by ``max_bytes``.

    Returns ``(line_bytes, was_oversized)``:
    - ``(None, False)`` indicates EOF.
    - ``(None, True)`` indicates the frame exceeded ``max_bytes`` and was drained.
    - ``(bytes, False)`` indicates a valid frame within ``max_bytes`` with CR/LF stripped.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    raw = stream.readline(max_bytes + 1)
    if not raw:
        return None, False

    if len(raw) > max_bytes and not raw.endswith(b"\n"):
        # Drain through the end of the oversized frame so stream framing stays aligned.
        while True:
            chunk = stream.readline(_DRAIN_CHUNK_BYTES)
            if not chunk or chunk.endswith(b"\n"):
                break
        return None, True

    stripped = raw.rstrip(b"\r\n")
    if len(stripped) > max_bytes:
        return None, True

    return stripped, False


def write_jsonrpc_message(stream: BinaryIO, payload: Mapping[str, Any]) -> None:
    """Serialize a JSON-RPC 2.0 payload as a single NDJSON line and flush."""
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if b"\n" in encoded or b"\r" in encoded:
        raise ValueError("JSON-RPC stdio frame must not contain raw newlines")
    stream.write(encoded + b"\n")
    stream.flush()


class UpstreamMCPProcess:
    """Manages an upstream MCP server subprocess over stdio NDJSON framing."""

    def __init__(
        self,
        cmd: Sequence[str],
        *,
        max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
        shutdown_timeout_s: float = 3.0,
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        on_notification: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        if not cmd:
            raise ValueError("upstream_cmd must be non-empty")
        self.cmd = list(cmd)
        self.max_line_bytes = int(max_line_bytes)
        self.shutdown_timeout_s = float(shutdown_timeout_s)
        self._on_notification = on_notification

        self._write_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: Dict[Any, queue.Queue] = {}
        self._closed = False
        self._eof = False
        self._stderr_tail: Deque[str] = collections.deque(maxlen=100)

        proc_env = dict(os.environ) if env is None else dict(env)
        proc_env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

        self.proc = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            close_fds=True,
            cwd=cwd,
            env=proc_env,
        )

        self._stdout_thread = threading.Thread(
            target=self._stdout_reader_loop,
            name="mcp-upstream-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._stderr_drain_loop,
            name="mcp-upstream-stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    @property
    def stderr_tail(self) -> List[str]:
        return list(self._stderr_tail)

    def _fail_all_pending(self, exc: Exception) -> None:
        with self._pending_lock:
            pending_items = list(self._pending.values())
            self._pending.clear()
        for q in pending_items:
            q.put(exc)

    def _stdout_reader_loop(self) -> None:
        assert self.proc.stdout is not None
        try:
            while True:
                line, oversized = read_bounded_jsonrpc_line(
                    self.proc.stdout, self.max_line_bytes
                )
                if oversized:
                    self._fail_all_pending(
                        UpstreamProtocolError(
                            f"Upstream response frame exceeded {self.max_line_bytes} bytes"
                        )
                    )
                    continue
                if line is None:
                    break
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    self._fail_all_pending(
                        UpstreamProtocolError(f"Upstream emitted invalid JSON: {exc}")
                    )
                    continue

                if not isinstance(msg, dict):
                    self._fail_all_pending(
                        UpstreamProtocolError("Upstream emitted non-object JSON-RPC frame")
                    )
                    continue

                if "id" in msg and msg["id"] is not None:
                    req_id = msg["id"]
                    with self._pending_lock:
                        q = self._pending.pop(req_id, None)
                    if q is not None:
                        q.put(msg)
                elif "method" in msg and self._on_notification is not None:
                    try:
                        self._on_notification(msg)
                    except Exception:
                        pass
        finally:
            self._eof = True
            rc = self._resolve_returncode()
            self._fail_all_pending(
                UpstreamExitedError(
                    f"Upstream MCP server exited (returncode={rc})",
                    returncode=rc,
                )
            )

    def _resolve_returncode(self, wait_s: float = 0.25) -> Optional[int]:
        rc = self.proc.poll()
        if rc is None:
            try:
                rc = self.proc.wait(timeout=wait_s)
            except subprocess.TimeoutExpired:
                rc = self.proc.poll()
        return rc

    def _stderr_drain_loop(self) -> None:
        assert self.proc.stderr is not None
        while True:
            line, oversized = read_bounded_jsonrpc_line(self.proc.stderr, 16_384)
            if oversized:
                self._stderr_tail.append("[stderr line truncated]")
                continue
            if line is None:
                break
            self._stderr_tail.append(line.decode("utf-8", errors="replace"))

    def is_alive(self) -> bool:
        return self.proc.poll() is None and not self._eof

    def send_notification(self, payload: Mapping[str, Any]) -> None:
        if self._closed or not self.is_alive():
            rc = self._resolve_returncode()
            raise UpstreamExitedError(
                f"Cannot send notification: upstream exited (returncode={rc})",
                returncode=rc,
            )
        assert self.proc.stdin is not None
        with self._write_lock:
            try:
                write_jsonrpc_message(self.proc.stdin, payload)
            except BrokenPipeError as exc:
                rc = self._resolve_returncode()
                raise UpstreamExitedError(
                    f"Upstream stdin pipe broken (returncode={rc})",
                    returncode=rc,
                ) from exc

    def request(self, payload: Mapping[str, Any], *, timeout_s: float) -> Dict[str, Any]:
        req_id = payload.get("id")
        if req_id is None:
            raise ValueError("JSON-RPC request must include a non-null 'id'")
        if self._closed or not self.is_alive():
            rc = self._resolve_returncode()
            raise UpstreamExitedError(
                f"Upstream MCP server is not running (returncode={rc})",
                returncode=rc,
            )

        resp_queue: queue.Queue = queue.Queue(maxsize=1)
        with self._pending_lock:
            if req_id in self._pending:
                raise ValueError(f"Duplicate in-flight request id: {req_id!r}")
            self._pending[req_id] = resp_queue

        assert self.proc.stdin is not None
        try:
            with self._write_lock:
                write_jsonrpc_message(self.proc.stdin, payload)
        except BrokenPipeError as exc:
            with self._pending_lock:
                self._pending.pop(req_id, None)
            rc = self._resolve_returncode()
            raise UpstreamExitedError(
                f"Upstream stdin pipe broken (returncode={rc})",
                returncode=rc,
            ) from exc

        try:
            item = resp_queue.get(timeout=timeout_s)
        except queue.Empty as exc:
            with self._pending_lock:
                self._pending.pop(req_id, None)
            raise UpstreamTimeoutError(
                f"Upstream MCP server timed out after {timeout_s:.3f}s for request {req_id!r}"
            ) from exc

        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.proc.stdin is not None and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except OSError:
            pass

        if self.proc.poll() is None:
            try:
                self.proc.wait(timeout=self.shutdown_timeout_s)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=1.0)

        self._stdout_thread.join(timeout=1.0)
        self._stderr_thread.join(timeout=1.0)


class _TrackingMCPProxy(SystemOneMCPProxy):
    """Thin SystemOneMCPProxy subclass capturing the last interception without double-evaluating."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._local = threading.local()

    def evaluate_mcp_call(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        executor: Optional[Callable] = None,
        client_session_id: Optional[str] = None,
        context_prompt: Optional[str] = None,
    ) -> GuardInterceptionResult:
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


class MCPStdioProxy:
    """MCP JSON-RPC 2.0 stdio proxy enforcing System 1 policy, conformal abstention, and Ed25519 receipts."""

    def __init__(
        self,
        config: MCPStdioProxyConfig,
        *,
        guard: Optional[SystemOneGuardHook] = None,
        policy: Optional[PolicyEngine] = None,
        ledger: Optional[ActionLedger] = None,
        signing_key: Optional[Ed25519PrivateKey] = None,
    ) -> None:
        self.config = config
        self._out_lock = threading.Lock()
        self._active_out_stream: Optional[BinaryIO] = None

        if guard is None:
            self.guard = SystemOneGuardHook(
                policy=policy,
                ledger=ledger,
                signing_key=signing_key,
                alpha=config.alpha,
                min_confidence=config.min_confidence,
                fail_closed_ledger=(ledger is not None),
                enforcement_profile=config.enforcement_profile,
            )
        else:
            self.guard = guard

        self.mcp_proxy = _TrackingMCPProxy(
            guard=self.guard,
            alpha=config.alpha,
            min_confidence=config.min_confidence,
            tenant_id=config.tenant_id,
            principal_id=config.principal_id,
            model_path=config.model_path,
        )

        self.upstream = UpstreamMCPProcess(
            config.upstream_cmd,
            max_line_bytes=config.max_line_bytes,
            shutdown_timeout_s=config.shutdown_timeout_s,
            cwd=config.cwd,
            env=config.env,
            on_notification=self._forward_upstream_notification,
        )

    def _forward_upstream_notification(self, msg: Dict[str, Any]) -> None:
        stream = self._active_out_stream
        if stream is not None:
            with self._out_lock:
                try:
                    write_jsonrpc_message(stream, msg)
                except OSError:
                    pass

    @staticmethod
    def _jsonrpc_error(
        req_id: Any,
        code: int,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        err: Dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return {"jsonrpc": "2.0", "id": req_id, "error": err}

    def _enrich_blocked_error(
        self,
        resp: Dict[str, Any],
        interception: Optional[GuardInterceptionResult],
    ) -> Dict[str, Any]:
        if interception is None or "error" not in resp:
            return resp
        data = resp["error"].setdefault("data", {})
        if interception.policy_decision is not None:
            data.setdefault("action_id", interception.policy_decision.action_id)
        if interception.decision_result is not None:
            dr = interception.decision_result
            data["is_ambiguous"] = bool(dr.is_ambiguous)
            data["conformal_sets"] = {k: list(v) for k, v in dr.conformal_sets.items()}
            data["confidences"] = {k: float(v) for k, v in dr.confidences.items()}
            data["escalated_fields"] = list(getattr(dr, "escalated_fields", []) or [])
        return resp

    def _enrich_allowed_result(
        self,
        resp: Dict[str, Any],
        interception: Optional[GuardInterceptionResult],
    ) -> Dict[str, Any]:
        if interception is None or "result" not in resp:
            return resp
        meta = resp["result"].setdefault("_meta", {}).setdefault("system1", {})
        if interception.policy_decision is not None:
            meta["action_id"] = interception.policy_decision.action_id
        receipt = interception.receipt
        if receipt is not None:
            meta["decision_id"] = receipt.decision_id
            meta["ledger_record_id"] = receipt.ledger_record_id
            meta["signer_public_key"] = receipt.signer_public_key
            if receipt.envelope is not None and receipt.envelope.signature:
                meta["signature"] = receipt.envelope.signature
                meta["signer_fingerprint"] = receipt.envelope.signer_fingerprint
        return resp

    def _forward_tools_call(
        self,
        req_id: Any,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        upstream_req = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        upstream_resp = self.upstream.request(
            upstream_req, timeout_s=self.config.upstream_timeout_s
        )
        if "error" in upstream_resp:
            err = upstream_resp["error"]
            raise UpstreamRPCError(
                int(err.get("code", -32603)),
                str(err.get("message", "Upstream tool error")),
                err.get("data"),
            )
        if "result" not in upstream_resp:
            raise UpstreamProtocolError("Upstream tools/call response missing 'result'")
        return upstream_resp["result"]

    def process_raw_line(self, raw_line: bytes) -> Optional[Dict[str, Any]]:
        """Process one bounded NDJSON line from the MCP client."""
        if not raw_line.strip():
            return None

        try:
            text = raw_line.decode("utf-8")
            data = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return self._jsonrpc_error(None, -32700, f"Parse error: {exc}")

        if not isinstance(data, dict):
            return self._jsonrpc_error(
                None,
                -32600,
                "Invalid Request: JSON-RPC 2.0 payload must be a JSON object",
            )

        raw_id = data.get("id")
        safe_id = raw_id if isinstance(raw_id, (str, int, type(None))) else None

        if data.get("jsonrpc") != "2.0":
            return self._jsonrpc_error(
                safe_id,
                -32600,
                "Invalid Request: 'jsonrpc' must equal '2.0'",
            )

        method = data.get("method")
        if not isinstance(method, str) or not method:
            return self._jsonrpc_error(
                safe_id,
                -32600,
                "Invalid Request: 'method' must be a non-empty string",
            )

        is_notification = "id" not in data

        # Security invariant: notifications MUST NOT invoke tools/call.
        if is_notification:
            if method == "tools/call":
                return self._jsonrpc_error(
                    None,
                    -32600,
                    "Invalid Request: 'tools/call' requires a non-null JSON-RPC 'id'",
                )
            try:
                self.upstream.send_notification(data)
            except UpstreamExitedError:
                return None
            return None

        if raw_id is None:
            return self._jsonrpc_error(
                None,
                -32600,
                "Invalid Request: MCP request 'id' must not be null",
            )

        if method != "tools/call":
            allowed, err_resp, _ = self.mcp_proxy.intercept_jsonrpc(data)
            if not allowed:
                return err_resp
            try:
                return self.upstream.request(
                    data, timeout_s=self.config.upstream_timeout_s
                )
            except UpstreamTimeoutError as exc:
                return self._jsonrpc_error(
                    raw_id,
                    -32002,
                    str(exc),
                    {"status": "UPSTREAM_TIMEOUT", "timeout_s": self.config.upstream_timeout_s},
                )
            except UpstreamExitedError as exc:
                return self._jsonrpc_error(
                    raw_id,
                    -32003,
                    str(exc),
                    {"status": "UPSTREAM_EXITED", "returncode": exc.returncode},
                )
            except UpstreamProtocolError as exc:
                return self._jsonrpc_error(
                    raw_id,
                    -32004,
                    str(exc),
                    {"status": "UPSTREAM_PROTOCOL_ERROR"},
                )

        self.mcp_proxy.pop_last_interception()
        upstream_failure: Optional[Exception] = None

        def upstream_executor(tool_name: str, arguments: Dict[str, Any]) -> Any:
            nonlocal upstream_failure
            try:
                return self._forward_tools_call(raw_id, tool_name, arguments)
            except Exception as exc:
                upstream_failure = exc
                raise

        resp = self.mcp_proxy.handle_call(data, upstream_executor)
        interception = self.mcp_proxy.pop_last_interception()

        if "error" in resp:
            if upstream_failure is not None:
                if isinstance(upstream_failure, UpstreamTimeoutError):
                    return self._jsonrpc_error(
                        raw_id,
                        -32002,
                        str(upstream_failure),
                        {
                            "status": "UPSTREAM_TIMEOUT",
                            "timeout_s": self.config.upstream_timeout_s,
                            "action_id": (
                                interception.policy_decision.action_id
                                if interception and interception.policy_decision
                                else None
                            ),
                        },
                    )
                if isinstance(upstream_failure, UpstreamExitedError):
                    return self._jsonrpc_error(
                        raw_id,
                        -32003,
                        str(upstream_failure),
                        {
                            "status": "UPSTREAM_EXITED",
                            "returncode": upstream_failure.returncode,
                            "action_id": (
                                interception.policy_decision.action_id
                                if interception and interception.policy_decision
                                else None
                            ),
                        },
                    )
                if isinstance(upstream_failure, UpstreamRPCError):
                    return self._jsonrpc_error(
                        raw_id,
                        upstream_failure.code,
                        upstream_failure.rpc_message,
                        upstream_failure.data,
                    )
            return self._enrich_blocked_error(resp, interception)

        return self._enrich_allowed_result(resp, interception)

    def run_stdio_loop(
        self,
        in_stream: Optional[BinaryIO] = None,
        out_stream: Optional[BinaryIO] = None,
    ) -> int:
        """Read NDJSON requests from ``in_stream`` and write responses to ``out_stream`` until EOF."""
        stdin_bin = in_stream or sys.stdin.buffer
        stdout_bin = out_stream or sys.stdout.buffer
        self._active_out_stream = stdout_bin

        try:
            while True:
                line, oversized = read_bounded_jsonrpc_line(
                    stdin_bin, self.config.max_line_bytes
                )
                if oversized:
                    err = self._jsonrpc_error(
                        None,
                        -32600,
                        f"Invalid Request: frame exceeds max_line_bytes ({self.config.max_line_bytes})",
                        {"status": "OVERSIZED_FRAME", "max_line_bytes": self.config.max_line_bytes},
                    )
                    with self._out_lock:
                        write_jsonrpc_message(stdout_bin, err)
                    continue
                if line is None:
                    break

                resp = self.process_raw_line(line)
                if resp is not None:
                    with self._out_lock:
                        write_jsonrpc_message(stdout_bin, resp)
            return 0
        finally:
            self._active_out_stream = None
            self.close()

    def close(self) -> None:
        self.upstream.close()


def _load_or_create_signing_key(identity_dir: Optional[str]) -> Optional[Ed25519PrivateKey]:
    if not identity_dir:
        return None
    id_path = Path(identity_dir)
    id_path.mkdir(parents=True, exist_ok=True)
    key_file = id_path / "identity.key"
    if key_file.is_file():
        return load_private_key(key_file)
    key = Ed25519PrivateKey.generate()
    save_keypair(key, id_path)
    return key


def _load_policy_file(policy_path: Optional[str]) -> Optional[PolicyEngine]:
    if not policy_path:
        return None
    data = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "rules" in data:
        raw_rules = data["rules"]
    elif isinstance(data, list):
        raw_rules = data
    else:
        raise ValueError("Policy JSON must be a list of rules or {'rules': [...]}")
    rules: List[PolicyRule] = []
    for item in raw_rules:
        item = dict(item)
        if "outcome" in item and isinstance(item["outcome"], str):
            item["outcome"] = DecisionOutcome(item["outcome"])
        rules.append(PolicyRule(**item))
    return PolicyEngine(rules=rules)


def add_mcp_proxy_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register MCP stdio proxy flags on ``parser``."""
    parser.add_argument("--ledger", help="Path to durable SQLite ActionLedger")
    parser.add_argument("--identity-dir", help="Directory containing Ed25519 identity.key")
    parser.add_argument("--policy", help="Path to JSON PolicyEngine rules file")
    parser.add_argument("--model", help="Optional compiled .s1m guard skill path")
    parser.add_argument("--alpha", type=float, default=0.20, help="Conformal significance level (default: 0.20)")
    parser.add_argument("--min-confidence", type=float, default=0.50, help="Minimum confidence threshold (default: 0.50)")
    parser.add_argument("--enforcement-profile", action="store_true", help="Require explicit deterministic PolicyRule ALLOW grants")
    parser.add_argument("--tenant-id", default="tenant_mcp_default", help="Authenticated tenant_id")
    parser.add_argument("--principal-id", default="agent_mcp_client", help="Authenticated principal_id")
    parser.add_argument("--max-line-bytes", type=int, default=DEFAULT_MAX_LINE_BYTES, help="Maximum NDJSON frame size in bytes")
    parser.add_argument("--upstream-timeout", type=float, default=30.0, help="Upstream request timeout in seconds")
    parser.add_argument("--shutdown-timeout", type=float, default=3.0, help="Upstream graceful shutdown timeout in seconds")
    parser.add_argument("--bearer-token", help="Bearer token required on Streamable HTTP /mcp requests")
    parser.add_argument(
        "--allowed-origin",
        action="append",
        dest="allowed_origins",
        help="Allowed browser Origin for Streamable HTTP requests (repeatable)",
    )
    parser.add_argument(
        "--require-origin",
        action="store_true",
        help="Reject Streamable HTTP requests missing an Origin header",
    )
    parser.add_argument(
        "--durability-mode",
        choices=("process_wal", "strict_sync", "group_commit", "high_throughput_normal"),
        default="process_wal",
        help="SQLite ActionLedger durability mode (default: process_wal)",
    )
    parser.add_argument("upstream_cmd", nargs=argparse.REMAINDER, help="Upstream MCP command after '--'")
    return parser


def build_mcp_proxy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="system1 mcp-proxy",
        description="Run the System 1 MCP JSON-RPC 2.0 stdio or Streamable HTTP proxy/guard",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "http", "streamable-http"),
        default="stdio",
        help="MCP proxy transport: 'stdio' (default) or 'http'/'streamable-http'",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="HTTP bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="HTTP bind port (default: 8765)")
    return add_mcp_proxy_arguments(parser)


def run_mcp_stdio_from_args(
    args: argparse.Namespace,
    *,
    error_fn: Callable[[str], Any],
) -> int:
    """Construct an MCPStdioProxy or MCPStreamableHTTPServer from parsed CLI arguments and execute."""
    transport = str(getattr(args, "transport", "stdio"))
    if transport in ("http", "streamable-http", "mcp-http"):
        from system1.integrations.mcp_http import run_mcp_http_from_args

        return run_mcp_http_from_args(args, error_fn=error_fn)

    cmd = list(getattr(args, "upstream_cmd", []) or [])
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        error_fn("Provide upstream MCP server command after '--'")

    signing_key = _load_or_create_signing_key(getattr(args, "identity_dir", None))
    policy = _load_policy_file(getattr(args, "policy", None))
    ledger_path = getattr(args, "ledger", None)
    ledger = ActionLedger(ledger_path, require_durable=True) if ledger_path else None

    if getattr(args, "enforcement_profile", False) and (signing_key is None or ledger is None):
        if ledger is not None:
            ledger.close()
        error_fn("--enforcement-profile requires both --identity-dir and --ledger")

    config = MCPStdioProxyConfig(
        upstream_cmd=cmd,
        max_line_bytes=int(getattr(args, "max_line_bytes", DEFAULT_MAX_LINE_BYTES)),
        upstream_timeout_s=float(getattr(args, "upstream_timeout", 30.0)),
        shutdown_timeout_s=float(getattr(args, "shutdown_timeout", 3.0)),
        tenant_id=str(getattr(args, "tenant_id", "tenant_mcp_default")),
        principal_id=str(getattr(args, "principal_id", "agent_mcp_client")),
        alpha=float(getattr(args, "alpha", 0.20)),
        min_confidence=float(getattr(args, "min_confidence", 0.50)),
        enforcement_profile=bool(getattr(args, "enforcement_profile", False)),
        model_path=getattr(args, "model", None),
    )
    proxy = MCPStdioProxy(config, policy=policy, ledger=ledger, signing_key=signing_key)
    try:
        return proxy.run_stdio_loop()
    finally:
        if ledger is not None:
            ledger.close()


def run_mcp_stdio_cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_mcp_proxy_parser()
    args = parser.parse_args(argv)
    return run_mcp_stdio_from_args(args, error_fn=parser.error)


__all__ = [
    "DEFAULT_MAX_LINE_BYTES",
    "MCPProxyConfig",
    "MCPStdioProxy",
    "MCPStdioProxyConfig",
    "UpstreamError",
    "UpstreamExitedError",
    "UpstreamMCPProcess",
    "UpstreamProtocolError",
    "UpstreamRPCError",
    "UpstreamTimeoutError",
    "add_mcp_proxy_arguments",
    "build_mcp_proxy_parser",
    "read_bounded_jsonrpc_line",
    "run_mcp_stdio_cli",
    "run_mcp_stdio_from_args",
    "write_jsonrpc_message",
]


if __name__ == "__main__":
    raise SystemExit(run_mcp_stdio_cli())
