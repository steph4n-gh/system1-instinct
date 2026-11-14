"""Tests for System 1 v1.2 AutoCutoverLifecycle, fallback/compat wiring, MCP Streamable HTTP server, and CLI flags."""

from __future__ import annotations

import concurrent.futures
import http.client
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from system1 import (
    ActionLedger,
    AutoCutoverLifecycle,
    BoundedReviewFallback,
    ChoiceField,
    CutoverConfig,
    CutoverState,
    DecisionOutcome,
    DecisionSchema,
    MCPStreamableHTTPConfig,
    MCPStreamableHTTPServer,
    MCP_PROTOCOL_VERSION,
    PolicyEngine,
    PolicyRule,
    SystemOneEngine,
    TypeSafeClient,
)
from system1.cli import build_parser
from system1.compat.typesafe import Choice
from system1.receipt import Ed25519PrivateKey


def _routing_schema() -> DecisionSchema:
    return DecisionSchema(
        schema_name="CutoverLifecycleTest",
        fields={
            "intent": ChoiceField(options=["billing", "refund", "technical"]),
        },
    )


def _diverse_prompt(idx: int) -> str:
    refunds = [
        "Please issue a refund for my order",
        "Need a refund for duplicate charge",
        "Customer wants refund back to card",
        "Refund requested for damaged package",
    ]
    billings = [
        "Update billing invoice payment method",
        "Monthly billing statement question",
        "Invoice tax ID update needed",
        "Check billing cycle renewal date",
    ]
    techs = [
        "API server returning gateway timeout",
        "Database connection pool error",
        "Application crash on login page",
        "SSL certificate handshake failure",
    ]
    bucket = idx % 3
    if bucket == 0:
        return f"{refunds[(idx // 3) % 4]} case {idx}"
    if bucket == 1:
        return f"{billings[(idx // 3) % 4]} case {idx}"
    return f"{techs[(idx // 3) % 4]} case {idx}"


def test_auto_cutover_lifecycle_cold_start_hotswap_quarantine_and_drift() -> None:
    schema = _routing_schema()
    drift_flag = {"flipped": False}

    def teacher(prompt: str) -> Dict[str, str]:
        if drift_flag["flipped"]:
            return {"intent": "technical"}
        low = prompt.lower()
        if "refund" in low:
            return {"intent": "refund"}
        if "billing" in low or "invoice" in low:
            return {"intent": "billing"}
        return {"intent": "technical"}

    cfg = CutoverConfig(
        milestones=(48, 80, 120),
        k_min=4,
        alpha_target=0.12,
        canary_rate=0.50,
        familiarity_alpha=0.01,
        prototype_alpha=0.01,
    )
    lifecycle = AutoCutoverLifecycle(schema, teacher=teacher, config=cfg)
    assert lifecycle.state == CutoverState.COLD_START_ABSTAIN.value

    first = lifecycle.step(_diverse_prompt(0))
    assert first.used_teacher is True
    assert first.state == CutoverState.COLD_START_ABSTAIN.value

    for i in range(1, 96):
        lifecycle.step(_diverse_prompt(i))
    # Exercise repeats so local fast-path and clean cache record hits after milestone 80.
    for i in range(18):
        lifecycle.step(_diverse_prompt(i))

    snap = lifecycle.summary()
    assert snap["compiled_active"] is True
    assert snap["state"] == CutoverState.SELECTIVE_HOT_SWAP.value
    assert snap["local_hits"] > 0
    assert snap["active_tau"] is not None

    # Poisoned over-long OOD sample is quarantined when enable_quarantine=True.
    poison = "zxqvwkjhgfdsapoiuytrewq mnbvcxz " + " ".join(["blarg"] * 30)
    poison_step = lifecycle.step(poison)
    assert poison_step.quarantined is True

    # Sustained teacher disagreement on locally served refund/billing prompts triggers DRIFT_RECALIBRATE.
    drift_flag["flipped"] = True
    reached_drift = False
    refund_billing_indices = [k for k in range(84) if (k % 3) in (0, 1)]
    for i in range(60):
        idx = refund_billing_indices[i % len(refund_billing_indices)]
        lifecycle.step(_diverse_prompt(idx))
        if lifecycle.state == CutoverState.DRIFT_RECALIBRATE.value:
            reached_drift = True
            break

    assert reached_drift is True
    assert len(lifecycle.recalibrate_events) >= 1


def test_bounded_review_fallback_and_typesafe_client_auto_cutover() -> None:
    schema = _routing_schema()
    engine = SystemOneEngine(schema)
    cfg = CutoverConfig(
        milestones=(48, 80),
        k_min=4,
        alpha_target=0.15,
        canary_rate=0.25,
        familiarity_alpha=0.01,
        prototype_alpha=0.01,
    )

    def review_fn(prompt: str) -> Dict[str, str]:
        low = prompt.lower()
        if "refund" in low:
            return {"intent": "refund"}
        if "billing" in low or "invoice" in low:
            return {"intent": "billing"}
        return {"intent": "technical"}

    fb = BoundedReviewFallback(
        engine=engine,
        review_fn=review_fn,
        schema=schema,
        auto_compile=True,
        auto_compile_config=cfg,
    )
    try:
        out = None
        for i in range(96):
            out = fb.decide(_diverse_prompt(i))
        for i in range(18):
            out = fb.decide(_diverse_prompt(i))
        assert out is not None
        assert "cutover_state" in out
        assert out["cutover_state"] == CutoverState.SELECTIVE_HOT_SWAP.value
        assert fb.cutover_lifecycle is not None
        assert fb.cutover_lifecycle.local_hits > 0
    finally:
        fb.close()

    def baseline_handler(state: str, _questions: Any) -> Dict[str, Any]:
        low = state.lower()
        val = "refund" if "refund" in low else ("billing" if "billing" in low else "technical")
        return {"answers": {"intent": {"type": "choice", "choice": val}}}

    client = TypeSafeClient(
        mode="auto_cutover",
        zero_egress=True,
        baseline_handler=baseline_handler,
        auto_cutover_lifecycle=CutoverConfig(
            milestones=(48, 80),
            k_min=4,
            alpha_target=0.15,
            canary_rate=0.25,
            familiarity_alpha=0.01,
            prototype_alpha=0.01,
        ),
    )
    try:
        questions = {
            "intent": Choice(
                "Ticket intent",
                criteria={"refund": "Refund", "billing": "Billing", "technical": "Technical"},
            )
        }
        assert client.auto_cutover_lifecycle is not None
        resp = None
        for i in range(96):
            resp = client.decide(_diverse_prompt(i), questions=questions)
            assert resp.answers.intent.choice in ("refund", "billing", "technical")
            assert resp.egress_bytes == 0
        for i in range(18):
            resp = client.decide(_diverse_prompt(i), questions=questions)
            assert resp.answers.intent.choice in ("refund", "billing", "technical")
            assert resp.egress_bytes == 0
        assert resp is not None
        assert resp.cutover_state == CutoverState.SELECTIVE_HOT_SWAP.value
        assert len(client._cutover_lifecycles) == 1
        live_lc = next(iter(client._cutover_lifecycles.values()))
        assert live_lc.compiled_active is True
        assert live_lc.local_hits > 0
    finally:
        client.close()


def _http_request(
    port: int,
    method: str,
    path: str,
    *,
    body: Any = None,
    raw_body: bytes | None = None,
    headers: Dict[str, str] | None = None,
) -> Tuple[int, Dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
    try:
        hdrs = dict(headers or {})
        payload: bytes | None = raw_body
        if payload is None and body is not None:
            payload = json.dumps(body).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
        conn.request(method, path, body=payload, headers=hdrs)
        resp = conn.getresponse()
        data = resp.read()
        resp_headers = {k.lower(): v for k, v in resp.getheaders()}
        return resp.status, resp_headers, data
    finally:
        conn.close()


def test_mcp_streamable_http_server_protocol_security_and_ledger(tmp_path: Path) -> None:
    executed_calls: list[Dict[str, Any]] = []
    tools = {
        "read_status": {
            "description": "Read status indicator",
            "inputSchema": {"type": "object", "properties": {"id": {"type": "integer"}}},
            "handler": lambda args: (
                executed_calls.append({"tool": "read_status", "args": dict(args)})
                or {"ok": True, "id": args.get("id")}
            ),
        },
        "delete_db": {
            "description": "Dangerous delete tool",
            "inputSchema": {"type": "object"},
            "handler": lambda args: (
                executed_calls.append({"tool": "delete_db", "args": dict(args)})
                or {"deleted": True}
            ),
        },
    }

    policy = PolicyEngine()
    policy.add_rule(
        PolicyRule(
            rule_id="allow-read-status",
            outcome=DecisionOutcome.ALLOW,
            scopes=["mcp:tools:call"],
            allowed_tools=["read_status"],
            reason="Explicit read_status allow",
        )
    )
    policy.add_rule(
        PolicyRule(
            rule_id="deny-delete-db",
            outcome=DecisionOutcome.DENY,
            scopes=["mcp:tools:call"],
            tools=["delete_db"],
            denied_tools=["delete_db"],
            reason="Delete blocked by policy",
        )
    )

    priv_key = Ed25519PrivateKey.generate()
    pub_key = priv_key.public_key()
    ledger_path = tmp_path / "mcp_http_ledger.db"

    cfg = MCPStreamableHTTPConfig(
        host="127.0.0.1",
        port=0,
        bearer_token="secret-tok",
        allowed_origins=("http://localhost:3000",),
        require_origin=True,
        max_body_bytes=4096,
        enforcement_profile=True,
        ledger_path=str(ledger_path),
        durability_mode="group_commit",
    )
    server = MCPStreamableHTTPServer(
        cfg,
        policy_engine=policy,
        signing_key=priv_key,
        tools=tools,
    )
    server.start_background()
    try:
        port = server.bound_port
        auth_ok = {
            "Authorization": "Bearer secret-tok",
            "Origin": "http://localhost:3000",
        }

        st, _, hbody = _http_request(port, "GET", "/healthz", headers={"Origin": "http://localhost:3000"})
        assert st == 200
        health = json.loads(hbody.decode("utf-8"))
        assert health["protocol_version"] == MCP_PROTOCOL_VERSION
        assert health["durability_mode"] == "group_commit"

        st, _, _ = _http_request(
            port,
            "POST",
            "/mcp",
            body={"jsonrpc": "2.0", "id": 1, "method": "ping"},
            headers={"Origin": "http://localhost:3000"},
        )
        assert st == 401

        st, _, _ = _http_request(
            port,
            "POST",
            "/mcp",
            body={"jsonrpc": "2.0", "id": 1, "method": "ping"},
            headers={"Authorization": "Bearer secret-tok", "Origin": "https://evil.example"},
        )
        assert st == 403

        st, _, _ = _http_request(
            port,
            "POST",
            "/mcp",
            raw_body=b"{" + b"a" * 5000 + b"}",
            headers={**auth_ok, "Content-Type": "application/json"},
        )
        assert st == 413

        st, hdrs, ibody = _http_request(
            port,
            "POST",
            "/mcp",
            body={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"clientInfo": {"name": "pytest"}},
            },
            headers=auth_ok,
        )
        assert st == 200
        sid = hdrs.get("mcp-session-id")
        assert sid
        init_json = json.loads(ibody.decode("utf-8"))
        assert init_json["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION

        st, _, _ = _http_request(
            port,
            "POST",
            "/mcp",
            body={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers={**auth_ok, "Mcp-Session-Id": sid},
        )
        assert st == 202

        st, _, _ = _http_request(
            port,
            "POST",
            "/mcp",
            body={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            headers={**auth_ok, "Mcp-Session-Id": "missing-session"},
        )
        assert st == 404

        st, _, cbody = _http_request(
            port,
            "POST",
            "/mcp",
            body={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "read_status", "arguments": {"id": 42}},
            },
            headers={**auth_ok, "Mcp-Session-Id": sid},
        )
        assert st == 200
        call_json = json.loads(cbody.decode("utf-8"))
        meta = call_json["result"]["_meta"]["system1"]
        assert meta["session_id"] == sid
        assert meta["signature"]
        assert meta["ledger_record_id"]
        assert call_json["result"]["structuredContent"] == {"ok": True, "id": 42}

        before_count = len(executed_calls)
        st, _, dbody = _http_request(
            port,
            "POST",
            "/mcp",
            body={
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "delete_db", "arguments": {}},
            },
            headers={**auth_ok, "Mcp-Session-Id": sid},
        )
        assert st == 200
        deny_json = json.loads(dbody.decode("utf-8"))
        assert "error" in deny_json
        assert len(executed_calls) == before_count

        st, shdrs, sbody = _http_request(
            port,
            "POST",
            "/mcp",
            body={"jsonrpc": "2.0", "id": 5, "method": "tools/list"},
            headers={**auth_ok, "Mcp-Session-Id": sid, "Accept": "text/event-stream"},
        )
        assert st == 200
        assert shdrs.get("content-type", "").startswith("text/event-stream")
        assert b"data: " in sbody
        assert b'"read_status"' in sbody

        st, _, bbody = _http_request(
            port,
            "POST",
            "/mcp",
            body=[
                {"jsonrpc": "2.0", "id": 6, "method": "ping"},
                {"jsonrpc": "2.0", "id": 7, "method": "tools/list"},
            ],
            headers={**auth_ok, "Mcp-Session-Id": sid},
        )
        assert st == 200
        batch_json = json.loads(bbody.decode("utf-8"))
        assert isinstance(batch_json, list) and len(batch_json) == 2

        def _concurrent_call(idx: int) -> int:
            code, _, raw = _http_request(
                port,
                "POST",
                "/mcp",
                body={
                    "jsonrpc": "2.0",
                    "id": 100 + idx,
                    "method": "tools/call",
                    "params": {"name": "read_status", "arguments": {"id": idx}},
                },
                headers={**auth_ok, "Mcp-Session-Id": sid},
            )
            assert code == 200
            parsed = json.loads(raw.decode("utf-8"))
            assert parsed["result"]["structuredContent"]["id"] == idx
            return idx

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futs = [pool.submit(_concurrent_call, idx) for idx in range(6)]
            assert [f.result() for f in futs] == list(range(6))

        st, _, _ = _http_request(
            port,
            "DELETE",
            "/mcp",
            headers={**auth_ok, "Mcp-Session-Id": sid},
        )
        assert st == 200
        st, _, _ = _http_request(
            port,
            "POST",
            "/mcp",
            body={"jsonrpc": "2.0", "id": 8, "method": "ping"},
            headers={**auth_ok, "Mcp-Session-Id": sid},
        )
        assert st == 404
    finally:
        server.stop()

    verifier = ActionLedger(ledger_path, read_only=True)
    try:
        assert len(verifier.records()) >= 7
        assert verifier.verify_integrity(trusted_public_key=pub_key) is True
    finally:
        verifier.close()


def test_cli_parses_mcp_http_and_durability_flags() -> None:
    parser = build_parser()

    serve_args = parser.parse_args(
        [
            "serve",
            "--transport",
            "mcp-http",
            "--port",
            "8765",
            "--bearer-token",
            "tok",
            "--allowed-origin",
            "http://localhost:3000",
            "--require-origin",
            "--durability-mode",
            "group_commit",
        ]
    )
    assert serve_args.transport == "mcp-http"
    assert serve_args.port == 8765
    assert serve_args.bearer_token == "tok"
    assert serve_args.allowed_origins == ["http://localhost:3000"]
    assert serve_args.require_origin is True
    assert serve_args.durability_mode == "group_commit"

    proxy_args = parser.parse_args(
        [
            "mcp-proxy",
            "--transport",
            "streamable-http",
            "--durability-mode",
            "process_wal",
        ]
    )
    assert proxy_args.transport == "streamable-http"
    assert proxy_args.durability_mode == "process_wal"
