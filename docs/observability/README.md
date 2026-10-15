# Observability reference

Optional Prometheus metrics and OpenTelemetry spans describe local decisions and review signals. They do not measure whether an external teacher or human reviewer completed a review. Starting a metrics server or configuring a remote exporter changes the network behavior of the host process.

## Installation

```bash
python -m pip install 'system1-instinct[observability,otel]'
```

Install `system1-instinct[observability]` or `system1-instinct[otel]` individually when you need only one integration. The base package does not require either dependency.

## Prometheus metrics (`SystemOneMetricsExporter`)

The example below instruments an untaught diagnostic schema, which requests review in strict mode:

```python
from system1 import ChoiceField, DecisionSchema, System1Engine
from system1.integrations.observability import SystemOneMetricsExporter

class Route(DecisionSchema):
    queue = ChoiceField(options=["billing", "support"])

engine = System1Engine(Route, strict_mode=True)
metrics = SystemOneMetricsExporter()
metrics.instrument(engine)
result = engine.decide("Please correct my invoice")
print("Needs review:", result.is_ambiguous)
```

`metrics.start_server(port=9090)` exposes `/metrics` using the default listen address of `prometheus_client.start_http_server` without authentication. Restrict network access through your deployment boundary. Example Prometheus scrape configuration:

```yaml
scrape_configs:
  - job_name: system1
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:9090']
```

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `system1_decisions_total` | Counter | `schema`, `outcome`, `cache_hit` | Instrumented decision calls, where `outcome` is the first returned field value |
| `system1_decision_latency_seconds` | Histogram | `schema` | Reported decision latency converted from milliseconds |
| `system1_escalations_total` | Counter | `schema`, `reason` | Decisions flagged for review, not completed external escalations |
| `system1_cache_hit_ratio` | Gauge | None | Cumulative hits divided by decisions since exporter creation |
| `system1_conformal_set_size` | Histogram | `schema` | Conformal set sizes recorded across fields |
| `system1_ledger_entries_total` | Gauge | None | Last observed ledger sequence depth |

Latency buckets are `0.0001`, `0.0005`, `0.001`, `0.005`, `0.01`, `0.05`, `0.1`, and `1.0` seconds. The reason labels `ambiguous`, `low_margin`, and `ood` come from an internal heuristic, and `ood` is not an independently calibrated distribution-shift detector. This histogram measures in-memory decision latency rather than full request, durable authorization, or tool execution latency. Bound `schema` and `outcome` label cardinality in your application.

`metrics.record_decision(...)` records a decision manually when you do not wrap `decide()`:

```python
metrics.record_decision(
    result=result,
    schema_name=result.schema_name,
    cache_hit=result.is_cache_hit,
    escalated=result.is_ambiguous,
)
```

Do not call `metrics.record_decision(...)` on an already instrumented engine. The wrapper updates the ledger gauge on a best-effort basis when the engine has a ledger, or you can call `metrics.update_ledger_gauge(ledger.audit_head()[0])` directly. A ledger-read failure leaves the previous gauge value in place.

## OpenTelemetry tracing (`SystemOneOTelInstrumentor`)

The example below exports spans to the console:

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from system1 import ChoiceField, DecisionSchema, System1Engine
from system1.integrations.otel import SystemOneOTelInstrumentor

class Route(DecisionSchema):
    queue = ChoiceField(options=["billing", "support"])

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
engine = System1Engine(Route, strict_mode=True)
SystemOneOTelInstrumentor(tracer_provider=provider).instrument(engine)
engine.decide("Please correct my invoice")
provider.shutdown()
```

Spans record the `system1.*` attribute names: `system1.schema`, `system1.latency_ms`, `system1.latency_seconds`, `system1.outcome`, `system1.cache_hit`, `system1.is_ambiguous`, and `system1.conformal_set_size_max`. The span name is `system1.decide.<schema>`. An ambiguity flag represents a review request, not an external service call.

Remote OTLP export requires a separate exporter package such as `opentelemetry-exporter-otlp-proto-grpc`, which is not part of `system1-instinct[otel]`. Configure its collector, transport, and access controls in your application. Both `SystemOneMetricsExporter` and `SystemOneOTelInstrumentor` can wrap the same engine when you attach each exporter once.

## Grafana dashboard

[grafana-dashboard.json](grafana-dashboard.json) defines panels for latency quantiles, decisions per second, review-signal rates, cumulative cache hit ratio, conformal-set buckets, and ledger depth. It uses the `system1-decision-engine` UID so importing the file updates existing dashboards.

[Metrics implementation](../../src/system1/integrations/observability.py) · [Tracing implementation](../../src/system1/integrations/otel.py) · [Deployment boundaries](../deployment.md).
