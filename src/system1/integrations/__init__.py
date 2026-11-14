"""System 1 Framework Integrations.

Native, drop-in adapters and middleware for application stacks:
- mcp: In-process Model Context Protocol (MCP) JSON-RPC guard.
- mcp_stdio: Runnable Model Context Protocol (MCP) JSON-RPC 2.0 stdio proxy/guard.
- mcp_http: Runnable Model Context Protocol (MCP) Streamable HTTP (2025-03-26) guard server.
- fastapi: ASGI gateway middleware for local routing and escalation to the downstream app.
- langchain: Agent tool interceptor and callback handler with Ed25519 witness receipts.
- observability: Prometheus-compatible /metrics exporter for decision engine telemetry.
- otel: OpenTelemetry tracing integration for decision spans.
"""

from __future__ import annotations

from system1.integrations.mcp import (
    SystemOneMCPBlockedError,
    SystemOneMCPProxy,
    wrap_mcp_tool,
)
from system1.integrations.mcp_http import (
    MCPGuardRuntime,
    MCPStreamableHTTPConfig,
    MCPStreamableHTTPServer,
    MCP_PROTOCOL_VERSION,
)
from system1.integrations.mcp_stdio import (
    MCPStdioProxy,
    MCPStdioProxyConfig,
)
from system1.integrations.fastapi import (
    SystemOneGatewayMiddleware,
    UniversalDecisionASGIApp,
    UniversalDecisionService,
    add_system1_gateway,
    add_universal_decisions_gateway,
    create_universal_asgi_app,
    mount_universal_gateway,
)
from system1.integrations.langchain import (
    SystemOneGuardBlockedException,
    SystemOneGuardCallbackHandler,
    SystemOneToolInterceptor,
    wrap_langchain_tool,
)
from system1.integrations.drop_in import (
    IncidentTriageModel,
    InstinctRouteLayer,
    PromptGuardResult,
    PromptGuardSchema,
    RouteDecision,
    RouteLLMCascadeSchema,
    RouteMatchResult,
    SemanticRoute,
    StructuredExtractionResult,
    SystemOneInstructorClient,
    SystemOnePromptGuard,
    SystemOneRouteController,
    SystemOneRouteLayer,
    UtilityCascadeRouter,
    CascadeRouteDecision,
    build_default_router,
)

# Compatibility alias
System1MCPProxy = SystemOneMCPProxy

__all__ = [
    # MCP
    "MCPGuardRuntime",
    "MCPStreamableHTTPConfig",
    "MCPStreamableHTTPServer",
    "MCP_PROTOCOL_VERSION",
    "MCPStdioProxy",
    "MCPStdioProxyConfig",
    "SystemOneMCPBlockedError",
    "SystemOneMCPProxy",
    "System1MCPProxy",
    "wrap_mcp_tool",
    # FastAPI / ASGI
    "SystemOneGatewayMiddleware",
    "UniversalDecisionASGIApp",
    "UniversalDecisionService",
    "add_system1_gateway",
    "add_universal_decisions_gateway",
    "create_universal_asgi_app",
    "mount_universal_gateway",
    # LangChain / Agent
    "SystemOneGuardBlockedException",
    "SystemOneGuardCallbackHandler",
    "SystemOneToolInterceptor",
    "wrap_langchain_tool",
    # Drop-In Ecosystem Replacements
    "SemanticRoute",
    "RouteMatchResult",
    "SystemOneRouteLayer",
    "InstinctRouteLayer",
    "build_default_router",
    "RouteLLMCascadeSchema",
    "RouteDecision",
    "CascadeRouteDecision",
    "SystemOneRouteController",
    "UtilityCascadeRouter",
    "IncidentTriageModel",
    "StructuredExtractionResult",
    "SystemOneInstructorClient",
    "PromptGuardSchema",
    "PromptGuardResult",
    "SystemOnePromptGuard",
]

# Optional: Prometheus observability (requires `prometheus_client`)
try:
    from system1.integrations.observability import SystemOneMetricsExporter

    System1MetricsExporter = SystemOneMetricsExporter
    __all__.extend(["SystemOneMetricsExporter", "System1MetricsExporter"])
except ImportError:
    pass

# Optional: OpenTelemetry tracing (requires `opentelemetry-api`)
try:
    from system1.integrations.otel import SystemOneOTelInstrumentor

    __all__.append("SystemOneOTelInstrumentor")
except ImportError:
    pass
