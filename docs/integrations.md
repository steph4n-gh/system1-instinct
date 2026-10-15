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

## Model Context Protocol (`System1MCPProxy`)

`System1MCPProxy` intercepts JSON-RPC tool calls before dispatch. The proxy takes a configured `SystemOneGuard` and an authenticated caller identity, and exposes `handle_call(request, executor)` and `async_handle_call(request, executor)`. [`tests/test_integrations.py`](https://github.com/steph4n-gh/system1-instinct/blob/main/tests/test_integrations.py) defines the request and response schema. Diagnostic guard mode (`enforcement_profile=False`) does not produce signed enforcement receipts.

## FastAPI and ASGI (`add_system1_gateway`)

`add_system1_gateway(app, schema=YourSchema)` registers `SystemOneGatewayMiddleware` on an ASGI application. The middleware returns a local HTTP response when `SystemOneEngine` produces an unambiguous decision (`is_ambiguous=False`) and forwards ambiguous requests to the downstream ASGI application.

- **Dependencies.** Requires an external ASGI framework such as `fastapi` or `starlette`.
- **Security boundary.** `SystemOneGatewayMiddleware` runs before downstream route dependencies. Authentication, authorization, and request-body size limits belong in an outer middleware or reverse proxy, as detailed in [Deployment boundaries](deployment.md#asgi-gateway).

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
