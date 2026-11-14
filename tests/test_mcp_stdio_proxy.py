"""End-to-end tests for MCPStdioProxy, SystemOneMCPProxy, and CLI MCP stdio transport."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

import pytest

from system1.guard import DecisionOutcome, RiskLevel
from system1.integrations import (
    MCPStdioProxy,
    MCPStdioProxyConfig,
    SystemOneMCPProxy,
)
from system1.integrations.mcp_stdio import (
    read_bounded_jsonrpc_line,
    write_jsonrpc_message,
)
from system1.ledger import ActionLedger
from system1.receipt import load_public_key, verify_decision_witness_receipt


MOCK_UPSTREAM_CODE = """
import json
import os
import sys
import time
from system1.integrations.mcp_stdio import read_bounded_jsonrpc_line, write_jsonrpc_message

MOCK_TOOLS = [
    {
        "name": "read_file",
        "description": "Read local documentation file",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "purpose": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "delete_database",
        "description": "Delete database",
        "inputSchema": {"type": "object", "properties": {"target": {"type": "string"}}},
    },
    {
        "name": "update_record",
        "description": "Update local record",
        "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "purpose": {"type": "string"}}},
    },
    {
        "name": "slow_tool",
        "description": "Simulate slow tool",
        "inputSchema": {"type": "object", "properties": {"sleep_s": {"type": "number"}, "purpose": {"type": "string"}}},
    },
    {
        "name": "crash_tool",
        "description": "Simulate upstream crash",
        "inputSchema": {"type": "object", "properties": {"purpose": {"type": "string"}}},
    },
]

def main():
    log_file = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else None
    stdin_bin = sys.stdin.buffer
    stdout_bin = sys.stdout.buffer

    def log_event(ev):
        if not log_file:
            return
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev) + "\\n")
            fh.flush()

    while True:
        line, oversized = read_bounded_jsonrpc_line(stdin_bin, 1_048_576)
        if oversized:
            continue
        if line is None:
            break
        if not line.strip():
            continue
        msg = json.loads(line.decode("utf-8"))
        method = msg.get("method")
        req_id = msg.get("id")
        if "id" not in msg:
            log_event({"kind": "notification", "method": method, "params": msg.get("params")})
            continue
        if method == "initialize":
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "mock-upstream-mcp", "version": "0.1.0"},
                    "capabilities": {"tools": {}},
                },
            }
        elif method == "ping":
            resp = {"jsonrpc": "2.0", "id": req_id, "result": {}}
        elif method == "tools/list":
            resp = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": MOCK_TOOLS}}
        elif method == "tools/call":
            params = msg.get("params") or {}
            tool_name = params.get("name", "")
            arguments = params.get("arguments") or {}
            log_event({"kind": "tools/call", "name": tool_name, "arguments": arguments})
            if tool_name == "slow_tool":
                time.sleep(float(arguments.get("sleep_s", 1.5)))
                resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": "slow_done"}]}}
            elif tool_name == "crash_tool":
                os._exit(42)
            else:
                target = arguments.get("path", arguments.get("target", "ok"))
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": f"upstream:{tool_name}:{target}"}]},
                }
        else:
            resp = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}
        write_jsonrpc_message(stdout_bin, resp)

if __name__ == "__main__":
    main()
"""


def _write_default_policy(path: Path) -> None:
    rules = {
        "rules": [
            {
                "rule_id": "veto_sensitive_targets",
                "outcome": "DENY",
                "risk": int(RiskLevel.IRREVERSIBLE),
                "denied_targets": ["/etc/shadow", "/etc/passwd"],
            },
            {
                "rule_id": "veto_dangerous_tools",
                "outcome": "DENY",
                "risk": int(RiskLevel.IRREVERSIBLE),
                "denied_tools": ["delete_database", "shell_exec"],
            },
        ]
    }
    path.write_text(json.dumps(rules), encoding="utf-8")


def _spawn_proxy(
    tmp_dir: Path,
    *,
    subcommand: str = "mcp-proxy",
    policy_path: Optional[Path] = None,
    ledger_path: Optional[Path] = None,
    identity_dir: Optional[Path] = None,
    upstream_log: Optional[Path] = None,
    max_line_bytes: int = 1_048_576,
    upstream_timeout: float = 5.0,
    alpha: float = 0.20,
    min_confidence: float = 0.50,
    enforcement_profile: bool = False,
) -> subprocess.Popen:
    upstream_script = tmp_dir / "mock_upstream.py"
    upstream_script.write_text(MOCK_UPSTREAM_CODE, encoding="utf-8")

    cmd: List[str] = [sys.executable, "-m", "system1.cli"]
    if subcommand == "serve":
        cmd.extend(["serve", "--transport", "mcp-stdio"])
    else:
        cmd.append("mcp-proxy")
    cmd.extend(
        [
            "--max-line-bytes",
            str(max_line_bytes),
            "--upstream-timeout",
            str(upstream_timeout),
            "--alpha",
            str(alpha),
            "--min-confidence",
            str(min_confidence),
            "--tenant-id",
            "mcp_tenant",
            "--principal-id",
            "mcp_user",
        ]
    )
    if policy_path is not None:
        cmd.extend(["--policy", str(policy_path)])
    if ledger_path is not None:
        cmd.extend(["--ledger", str(ledger_path)])
    if identity_dir is not None:
        cmd.extend(["--identity-dir", str(identity_dir)])
    if enforcement_profile:
        cmd.append("--enforcement-profile")

    log_arg = str(upstream_log) if upstream_log is not None else "-"
    cmd.extend(["--", sys.executable, str(upstream_script), log_arg])

    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        env=env,
    )


def _rpc_call(proc: subprocess.Popen, payload: Dict[str, Any]) -> Dict[str, Any]:
    assert proc.stdin is not None and proc.stdout is not None
    write_jsonrpc_message(proc.stdin, payload)
    line, oversized = read_bounded_jsonrpc_line(proc.stdout, 1_048_576)
    assert not oversized
    assert line is not None, (
        "Proxy closed stdout unexpectedly: "
        + (proc.stderr.read().decode(errors="replace") if proc.stderr else "")
    )
    return json.loads(line.decode("utf-8"))


def _read_upstream_log(log_path: Path) -> List[Dict[str, Any]]:
    if not log_path.is_file():
        return []
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_in_process_mcp_proxy_defaults_catalog_and_notifications() -> None:
    tools = [
        {"name": "read_doc", "description": "Read docs"},
        {"name": "list_tickets", "description": "List tickets"},
    ]
    proxy = SystemOneMCPProxy(tools=tools)
    assert proxy.guard.alpha == pytest.approx(0.20)

    list_resp = proxy.handle_call(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        lambda _n, _a: pytest.fail("executor must not run on tools/list"),
    )
    assert list_resp["result"]["tools"] == tools

    notif_resp = proxy.handle_call(
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        lambda _n, _a: pytest.fail("executor must not run on notification"),
    )
    assert notif_resp == {}


@pytest.mark.parametrize("subcommand", ["mcp-proxy", "serve"])
def test_stdio_proxy_lifecycle_allowed_call_and_ed25519_receipt(tmp_path: Path, subcommand: str) -> None:
    policy_path = tmp_path / "policy.json"
    ledger_path = tmp_path / "audit.sqlite"
    id_dir = tmp_path / "identity"
    log_path = tmp_path / "upstream.jsonl"
    _write_default_policy(policy_path)

    proc = _spawn_proxy(
        tmp_path,
        subcommand=subcommand,
        policy_path=policy_path,
        ledger_path=ledger_path,
        identity_dir=id_dir,
        upstream_log=log_path,
    )
    try:
        init_resp = _rpc_call(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0"},
                },
            },
        )
        assert init_resp["result"]["serverInfo"]["name"] == "mock-upstream-mcp"

        assert proc.stdin is not None
        write_jsonrpc_message(proc.stdin, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        assert _rpc_call(proc, {"jsonrpc": "2.0", "id": 2, "method": "ping"}) == {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {},
        }

        list_resp = _rpc_call(proc, {"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}})
        assert [t["name"] for t in list_resp["result"]["tools"]] == [
            "read_file",
            "delete_database",
            "update_record",
            "slow_tool",
            "crash_tool",
        ]

        call_resp = _rpc_call(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "read_file",
                    "arguments": {
                        "path": "README.md",
                        "purpose": "Inspect read-only project documentation in README.md",
                    },
                },
            },
        )
        assert call_resp["result"]["content"][0]["text"] == "upstream:read_file:README.md"
        s1_meta = call_resp["result"]["_meta"]["system1"]
        assert s1_meta["status"] == "ALLOW"
        assert len(s1_meta["digest"]) == 64
        assert len(s1_meta["ledger_record_id"]) == 64
        assert s1_meta["signature"]
    finally:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        assert proc.wait(timeout=5) == 0

    events = _read_upstream_log(log_path)
    assert any(e.get("kind") == "notification" and e.get("method") == "notifications/initialized" for e in events)
    assert len([e for e in events if e.get("kind") == "tools/call"]) == 1

    pub_key = load_public_key(id_dir / "identity.pub")
    with ActionLedger(ledger_path, read_only=True) as ledger:
        assert ledger.verify_integrity(trusted_public_key=pub_key) is True
        action_entries = [e for e in ledger.entries() if e["action_id"] == s1_meta["action_id"]]
        assert [e["event_type"] for e in action_entries] == ["system1_decision", "execution_outcome"]
        assert verify_decision_witness_receipt(action_entries[0]["payload"]["receipt"], public_key=pub_key) is True


def test_stdio_proxy_policy_block_and_conformal_abstention(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.json"
    ledger_path = tmp_path / "audit.sqlite"
    id_dir = tmp_path / "identity"
    log_path = tmp_path / "upstream.jsonl"
    _write_default_policy(policy_path)

    proc = _spawn_proxy(
        tmp_path,
        policy_path=policy_path,
        ledger_path=ledger_path,
        identity_dir=id_dir,
        upstream_log=log_path,
    )
    try:
        r_tool = _rpc_call(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {
                    "name": "delete_database",
                    "arguments": {"target": "prod_db", "purpose": "Inspect read-only docs"},
                },
            },
        )
        assert r_tool["error"]["code"] == -32000
        assert r_tool["error"]["data"]["rule_id"] == "veto_dangerous_tools"

        r_target = _rpc_call(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 11,
                "method": "tools/call",
                "params": {
                    "name": "read_file",
                    "arguments": {"path": "/etc/shadow", "purpose": "Inspect read-only file"},
                },
            },
        )
        assert r_target["error"]["code"] == -32000
        assert r_target["error"]["data"]["rule_id"] == "veto_sensitive_targets"

        r_amb = _rpc_call(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 12,
                "method": "tools/call",
                "params": {
                    "name": "update_record",
                    "arguments": {
                        "path": "config.json",
                        "purpose": "Inspect or modify local cache file and send webhook",
                    },
                },
            },
        )
        assert r_amb["error"]["code"] == -32000
        assert r_amb["error"]["data"]["outcome"] == "REQUIRE_APPROVAL"
        assert r_amb["error"]["data"]["rule_id"] == "system1_conformal_ambiguity"
        assert len(r_amb["error"]["data"]["conformal_sets"]["is_safe"]) > 1
    finally:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        assert proc.wait(timeout=5) == 0

    assert [e for e in _read_upstream_log(log_path) if e.get("kind") == "tools/call"] == []


def test_stdio_proxy_malformed_json_and_oversized_frame_recovery(tmp_path: Path) -> None:
    log_path = tmp_path / "upstream.jsonl"
    proc = _spawn_proxy(tmp_path, upstream_log=log_path, max_line_bytes=4096)
    try:
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(b"{not valid json\n")
        proc.stdin.flush()
        line, _ = read_bounded_jsonrpc_line(proc.stdout, 65_536)
        assert line is not None and json.loads(line.decode("utf-8"))["error"]["code"] == -32700

        write_jsonrpc_message(
            proc.stdin,
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "read_file", "arguments": {"path": "README.md"}},
            },
        )
        line, _ = read_bounded_jsonrpc_line(proc.stdout, 65_536)
        assert line is not None and json.loads(line.decode("utf-8"))["error"]["code"] == -32600

        oversized = b'{"jsonrpc":"2.0","id":99,"method":"ping","params":{"pad":"' + (b"A" * 32_768) + b'"}}\n'
        proc.stdin.write(oversized)
        proc.stdin.flush()
        line, _ = read_bounded_jsonrpc_line(proc.stdout, 65_536)
        assert line is not None
        err = json.loads(line.decode("utf-8"))
        assert err["error"]["code"] == -32600
        assert err["error"]["data"]["status"] == "OVERSIZED_FRAME"

        assert _rpc_call(proc, {"jsonrpc": "2.0", "id": "ok", "method": "ping"}) == {
            "jsonrpc": "2.0",
            "id": "ok",
            "result": {},
        }
    finally:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        assert proc.wait(timeout=5) == 0


def test_stdio_proxy_timeout_and_crash_record_failed_outcomes(tmp_path: Path) -> None:
    upstream_script = tmp_path / "mock_upstream.py"
    upstream_script.write_text(MOCK_UPSTREAM_CODE, encoding="utf-8")

    # Timeout
    t_ledger = tmp_path / "timeout.sqlite"
    with ActionLedger(t_ledger, require_durable=True) as ledger:
        cfg = MCPStdioProxyConfig(
            upstream_cmd=[sys.executable, str(upstream_script), "-"],
            upstream_timeout_s=0.2,
        )
        proxy = MCPStdioProxy(cfg, ledger=ledger)
        try:
            resp = proxy.process_raw_line(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "slow_tool",
                            "arguments": {"sleep_s": 1.2, "purpose": "Inspect read-only README.md"},
                        },
                    }
                ).encode("utf-8")
            )
            assert resp is not None
            assert resp["error"]["code"] == -32002
            assert resp["error"]["data"]["status"] == "UPSTREAM_TIMEOUT"
            action_id = resp["error"]["data"]["action_id"]
        finally:
            proxy.close()

    with ActionLedger(t_ledger, read_only=True) as ro:
        assert any(
            e["event_type"] == "execution_outcome" and e["action_id"] == action_id and e["payload"]["status"] == "FAILED"
            for e in ro.entries()
        )

    # Crash
    c_ledger = tmp_path / "crash.sqlite"
    with ActionLedger(c_ledger, require_durable=True) as ledger:
        cfg_c = MCPStdioProxyConfig(
            upstream_cmd=[sys.executable, str(upstream_script), "-"],
            upstream_timeout_s=2.0,
        )
        proxy_c = MCPStdioProxy(cfg_c, ledger=ledger)
        try:
            resp_c = proxy_c.process_raw_line(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {
                            "name": "crash_tool",
                            "arguments": {"purpose": "Inspect read-only README.md"},
                        },
                    }
                ).encode("utf-8")
            )
            assert resp_c is not None
            assert resp_c["error"]["code"] == -32003
            assert resp_c["error"]["data"]["returncode"] == 42
            follow = proxy_c.process_raw_line(
                json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}}).encode("utf-8")
            )
            assert follow is not None and follow["error"]["code"] == -32003
        finally:
            proxy_c.close()
