# Application integrations reference

This reference describes the runtime adapters that connect a [compiled skill](guides/first_skill.md) or a configured [`SystemOneGuard`](guides/policy_guard.md) to LangChain, Model Context Protocol (MCP), FastAPI and ASGI, gRPC, and OpenTelemetry or Prometheus.

## LangChain (`system1-instinct[langchain]`)

`SystemOneGuardCallbackHandler` intercepts LangChain tool invocations and evaluates each call against a configured `SystemOneGuard`. This adapter requires the `system1-instinct[langchain]` extra (`python -m pip install 'system1-instinct[langchain]'`):

```python
from system1.integrations import SystemOneGuardCallbackHandler

# guard is a configured SystemOneGuard with an active ActionLedger and signing key.
handler = SystemOneGuardCallbackHandler(
    guard=guard, tenant_id="demo", principal_id="agent-worker",
)
result = your_tool.invoke(tool_arguments, config={"callbacks": [handler]})
```

- **Proposal scope.** `SystemOneGuardCallbackHandler` sets `scope="langchain:tools:exec"` and passes the serialized tool input in the `ActionProposal`. Policy rules in `PolicyEngine` must match `langchain:tools:exec`.
- **Failure behavior.** When `guard.evaluate_proposal(...)` returns `DecisionOutcome.DENY` or `DecisionOutcome.REQUIRE_APPROVAL`, the callback raises `SystemOneGuardBlockedException`.
- **Direct callable wrapper.** `wrap_langchain_tool` wraps a Python callable with the same `SystemOneGuard` check.

## Model Context Protocol (`SystemOneMCPProxy`, `MCPStdioProxy`, and `MCPStreamableHTTPServer`)

- **In-process guard (`SystemOneMCPProxy`).** Defined in [`src/system1/integrations/mcp.py`](https://github.com/steph4n-gh/system1-instinct/blob/main/src/system1/integrations/mcp.py) (also exported as `System1MCPProxy`). Call `handle_call(request, executor)` or `async_handle_call(request, executor)` in Python (`alpha=0.20` default, optional `tools` list served on `tools/list`, `{}` returned for JSON-RPC notifications without `id`). [`tests/test_integrations.py`](https://github.com/steph4n-gh/system1-instinct/blob/main/tests/test_integrations.py) defines the request and response schema. Diagnostic guard mode (`enforcement_profile=False`) does not produce signed enforcement receipts.
- **Runnable stdio proxy (`MCPStdioProxy`).** Defined in [`src/system1/integrations/mcp_stdio.py`](https://github.com/steph4n-gh/system1-instinct/blob/main/src/system1/integrations/mcp_stdio.py) and launched via `system1 mcp-proxy [flags] -- <upstream_cmd> [args...]` or `system1 serve --transport mcp-stdio [flags] -- <upstream_cmd> [args...]`. Wraps an upstream MCP stdio child process with bounded JSON-RPC line framing (`max_line_bytes=4_194_304`), `SystemOneGuard` evaluation (`scope="mcp:tools:exec"`), and upstream timeout handling (`upstream_timeout_s=30.0`), writing diagnostics only to `stderr`.
- **Streamable HTTP guard server (`MCPStreamableHTTPServer`).** Defined in `src/system1/integrations/mcp_http.py` (protocol `2025-03-26`) and launched via `system1 serve --transport mcp-http [flags]` or `system1 mcp-proxy --transport http [flags]`. Binds to loopback `127.0.0.1` by default (`--host`, `--port`), verifies optional `Authorization: Bearer` tokens (`--bearer-token`), enforces `Origin` allowlists (`--allowed-origin`, `--require-origin`), manages `Mcp-Session-Id` state across `POST /mcp` and `DELETE /mcp` with bounded request bodies (`max_body_bytes=4_194_304`), evaluates tool calls under `scope="mcp:tools:call"`, attaches receipt metadata in `result._meta.system1`, and configures SQLite WAL durability (`--durability-mode process_wal|strict_sync|group_commit|high_throughput_normal`).
- **Measured stdio latency (Apple Silicon macOS, single-threaded BLAS, 100 calls after warmup).** Direct upstream stdio `tools/call` is `0.050 ms` p50 (`0.087 ms` p95). Proxied deterministic policy `ALLOW` without a SQLite ledger is `0.314 ms` p50 (`0.443 ms` p95). Proxied conformal guard with Ed25519 receipt signing and no SQLite ledger is `0.647 ms` p50 (`0.872 ms` p95). Enabling a durable SQLite `ActionLedger` (`require_durable=True`, two `PRAGMA synchronous=FULL` commits per allowed call) measures `101.832 ms` p50 (`165.266 ms` p95) on macOS APFS due to `F_FULLFSYNC`.

## FastAPI and ASGI (`add_system1_gateway`)

`add_system1_gateway(app, schema=YourSchema)` registers `SystemOneGatewayMiddleware` on an ASGI application. The middleware returns a local HTTP response when `SystemOneEngine` produces an unambiguous decision (`is_ambiguous=False`) and forwards ambiguous requests to the downstream ASGI application.

- **Dependencies.** Requires an external ASGI framework such as `fastapi` or `starlette`.
- **Request-size and route-header bounds.** `max_body_bytes` defaults to `1_048_576` (1 MiB) and rejects oversize bodies with HTTP `413` (`x-system1-status: rejected_oversize`) before buffering the full payload. Header-triggered routing via `x-system1-route: true` is disabled by default (`allow_route_header=False`) and supports an optional shared secret (`route_header_token`, verified with `hmac.compare_digest` against `x-system1-route-token`).
- **Security boundary.** `SystemOneGatewayMiddleware` runs before downstream route dependencies. Authentication and caller authorization still belong in outer middleware or a reverse proxy, as detailed in [Deployment boundaries](deployment.md#asgi-gateway).

## gRPC (`system1-instinct[grpc]`)

The `system1-instinct[grpc]` extra provides the gRPC server and CLI entrypoint:

```bash
python -m pip install 'system1-instinct[grpc]'
system1 serve --grpc --host 127.0.0.1 --port 50051
```

- **Protocol buffer definition.** [`src/system1/proto/system1.proto`](https://github.com/steph4n-gh/system1-instinct/blob/main/src/system1/proto/system1.proto) defines the wire contract for generated clients in Python and other languages.
- **Transport and security.** The `system1 serve --grpc` CLI command binds to `127.0.0.1` over plaintext gRPC without caller authentication, persistent signing keys, or a durable `ActionLedger`. Production services use `system1.grpc_server.serve(...)` in Python with explicit `schemas`, `guard`, and transport security. Experimental container templates live in [`deploy/README.md`](https://github.com/steph4n-gh/system1-instinct/blob/main/deploy/README.md).

## Observability (`system1-instinct[observability]` and `system1-instinct[otel]`)

The `system1-instinct[observability]` extra enables Prometheus metrics, and `system1-instinct[otel]` enables OpenTelemetry tracing. See the [observability reference](observability/README.md) and [`grafana-dashboard.json`](observability/grafana-dashboard.json). Enabling a metrics server or trace exporter opens network listeners or outbound connections in the host process.

## Inbox Zero email classification (`system1-instinct[http]`)

The [Inbox Zero integration](../examples/inbox_zero/README.md) provides JSON lesson sets, an authenticated local HTTP endpoint (`system1-instinct[http]`), and a provider patch for seven email categories. It uses `TfidfProjector` and a ridge decision head with no external model weights. The provider starts in review-only mode and leaves uncertain classifications unprocessed when the configuration omits an LLM fallback. For a credential-free public dataset baseline, see the [public spam and ham experiment](../benchmarks/quality/public_email/README.md).
