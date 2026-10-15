# Configure deployment boundaries and runtime limits

System 1: Instinct 1.0.0 provides NumPy-based compilation, conformal calibration, local `.s1m` inference, teacher observation, closed-loop review fallback (`BoundedReviewFallback`), and deterministic tool authorization (`SystemOneGuard`). See the [six-category benchmark report](../benchmarks/quality/instinct_gauntlet_report.md), [drop-in replacement guides](drop_in/README.md), [changelog](../CHANGELOG.md), and [benchmark suite index](../benchmarks/quality/README.md). Follow the procedures below to enforce security boundaries, statistical checks, and resource limits in production.

## Enforce deterministic tool authorization

To gate tool execution in production, instantiate `SystemOneGuard` with `enforcement_profile=True`, explicit `PolicyEngine` rules, a persistent Ed25519 signing key, and a durable `ActionLedger`:

1. **Require explicit policy matches.** In `enforcement_profile=True`, any proposal without a matching `PolicyRule` returns `DecisionOutcome.DENY`. When multiple rules match, `DENY` overrides `REQUIRE_APPROVAL`, which overrides `ALLOW`.
2. **Bind caller identity from your session.** `SystemOneGuard` evaluates the `tenant_id`, `principal_id`, and `scope` fields on `ActionProposal`, but does not authenticate the caller. Populate those fields from your authenticated session and keep signing keys and `PolicyEngine` rules outside agent control.
3. **Restrict tool arguments and targets.** Constrain argument values in `PolicyRule(argument_limits=...)` and normalize file paths, shell arguments, and URLs according to the target executor before calling `guard.evaluate_proposal(proposal)`.
4. **Enforce outcomes in the host application.** Check `auth.outcome == DecisionOutcome.ALLOW` before executing the tool, and pass only the authorized arguments to the tool. `SystemOneGuard` is an in-process Python check, not an operating-system sandbox. To prevent replay of a captured receipt, track consumed `action_id` values in your application.
5. **Reserve diagnostic mode for local testing.** When `enforcement_profile=False` (the default), `SystemOneGuard` can approve proposals using its statistical classifier and does not require a durable ledger or signing key. Do not use diagnostic mode to gate privileged tools.

## Calibrate and validate statistical skills

Follow these rules when deploying statistical decisions from `SystemOneEngine`:

1. **Do not use the zero-shot seed classifier as a security boundary.** On the historical `0.2.2` security-triage benchmark, the untaught seed classifier reached `50%` raw accuracy and misclassified `8` of `35` `BLOCK` prompts as `ALLOW`. Always use `SystemOneGuard(enforcement_profile=True)` for tool permissions.
2. **Maintain exchangeability between calibration and live traffic.** Split-conformal prediction (`strict_mode=True`) guarantees marginal label coverage only when calibration examples and live inputs are exchangeable and the scoring head is fixed before calibration. It does not guarantee conditional correctness on every accepted singleton, joint coverage across all fields simultaneously, or adversarial robustness. An empty conformal set (`conformal_sets[field] == []`) signals abstention, not proof of distribution shift. See [Angelopoulos and Bates (2021)](https://arxiv.org/abs/2107.07511).
3. **Keep teaching, calibration, and evaluation data disjoint.** Measure accuracy, coverage, and accepted-error rates on held-out traffic from your target environment. Setting `alpha` does not fix a constant escalation rate. `SystemOneEngine.decide()` returns structured values and `is_ambiguous` flags, leaving fallback execution to your application or `BoundedReviewFallback`.
4. **Recalibrate when strict coverage guarantees are required.** `engine.learn(...)` and `engine.learn_from_outcome(...)` default to `preserve_calibration=True`, which scales existing quantiles via Sherman-Morrison directional gain (or synthesizes twenty initial scores when starting from zero examples). When you require formal split-conformal coverage after model updates, pass `preserve_calibration=False` and call `engine.calibrate(...)` on fresh held-out examples.

## Verify audit receipts and protect ledger files

To verify signed receipts and maintain an intact `ActionLedger` chain:

1. **Supply an independently trusted public key.** Call `verify_decision_witness_receipt(receipt, public_key=trusted_key)` in Python or `system1 verify-receipt receipt.json --public-key trusted.pub` on the CLI. Calling `verify_decision_witness_receipt` without `public_key` returns `False`. Never trust a public key extracted from the receipt itself. For unsigned local integrity checks, use `check_decision_receipt_integrity(receipt)`.
2. **Verify ledger inclusion against the SQLite chain.** `ActionLedger` stores a linear SHA-256 hash chain in SQLite and binds `ledger_record_id` inside the signed receipt envelope. Call `ledger.verify_receipt_record(receipt, trusted_public_key=trusted_key)` to verify both the stored record and the full hash chain. Because a linear SQLite chain is not a hardware enclave or external Merkle log, export periodic ledger-head digests to external append-only storage if you need truncation detection on a compromised host.
3. **Persist your Ed25519 keypair across restarts.** Generating a new signing key on every process start breaks single-key verification across ledger history. Store the private key with restricted filesystem permissions and back it up alongside the SQLite ledger.
4. **Log policy denials separately if required.** Deterministic policy denials and approval requests do not always emit signed witness receipts. Record those events in your application logs if your audit policy requires a log of every rejected attempt.
5. **Restrict access to ledger and receipt files.** Receipts and SQLite ledgers store prompts, tool arguments, and execution results in plaintext. Apply filesystem permissions and encryption at rest according to your data-handling policy.
6. **Audit outbound network paths.** `SystemOneEngine.decide()` makes no network calls. When you use `TypeSafeClient`, `BoundedReviewFallback`, OpenTelemetry or Prometheus exporters, or custom projectors, audit their outbound network connections. Setting `zero_egress=True` on `TypeSafeClient` disables only that client's HTTP transport, not the entire Python process.

<a id="asgi-gateway"></a>
## Place authentication outside the ASGI gateway

When you attach `SystemOneGatewayMiddleware` (via `add_system1_gateway`):

1. **Enforce authentication and request-size limits in outer middleware.** Because `SystemOneGatewayMiddleware` returns local HTTP responses before FastAPI route dependencies run, place authentication, rate limiting, and body-size caps in your reverse proxy or outer ASGI middleware.
2. **Scope `route_paths` to classification endpoints.** `SystemOneGatewayMiddleware` matches path prefixes and any `POST` carrying `x-reflex-route: true`. Use it only on classification endpoints. If you pass a custom `is_fastpath_fn`, verify its acceptance criteria on held-out traffic.

## Bind principal identity in LangChain and MCP adapters

When you attach `SystemOneGuardCallbackHandler` or `System1MCPProxy`:

1. **Assign one `principal_id` per authenticated user or service.** Do not use per-request LangChain `run_id` values as `principal_id`.
2. **Attach callbacks directly to tool objects and keep `raise_error=True`.** Allow `SystemOneGuardBlockedException` to propagate so blocked tool calls cannot execute.

<a id="grpc"></a>
## Secure the gRPC service boundary

When you deploy the gRPC server (`system1[grpc]`):

1. **Do not expose the CLI diagnostic server to untrusted networks.** `system1 serve --grpc` binds to `127.0.0.1` over plaintext gRPC without TLS, caller authentication, a persistent signing key, or a durable `ActionLedger`.
2. **Configure production servers through `system1.grpc_server.serve(...)`.** Register explicit schema objects with `serve(schemas={"support": schema})`, attach a configured `SystemOneGuard`, and terminate TLS and caller authentication at your service proxy.
3. **Pre-register every `schema_name`.** Remote gRPC requests can select only built-in schemas (`triage`, `guard`, and their canonical aliases) or exact names passed to `serve(schemas=...)`. Unknown schema names fail immediately and never load Python modules, local files, or inline JSON from remote input.

<a id="saved-skill-resource-limits"></a>
## Enforce `.s1m` loader resource budgets

`CompiledSystemOneModel.load(...)` and `SystemOneEngine.load(...)` enforce fixed admission budgets before materializing arrays:

| Resource | Maximum allowed value |
| :--- | :--- |
| `.s1m` file size | `64 MiB` |
| JSON header size | `1 MiB` |
| Fields per schema | `64` |
| Options per field | `1,024` |
| Feature dimension | `8,192` |
| Calibration scores per field | `1,000,000` |
| NPZ archive members per field | `17` |
| Expanded array payload | `128 MiB` |
| Per-member compression ratio | `1,024:1` |
| Estimated model allocation budget | `256 MiB` |

The loader disables Python `pickle`, verifies NPY header shapes and `float32` or `float64` dtypes before reading arrays, and raises `ValueError` on malformed or over-budget files. `HybridProjector` skills store their spectral and subspace matrices as binary `.npz` members (`_projector_spectral_vectors` and `_projector_subspace_components`). Both v1 (APS) and v2 (LAC) `.s1m` files remain readable in System 1: Instinct 1.0.0.

## Measure end-to-end latency on target hardware

Reported sub-millisecond latencies (`0.1 to 0.4 ms`) measure median in-memory `engine.decide(...)` calls on CPU. They do not include SQLite disk syncs in `ActionLedger`, network hops in gRPC or HTTP, or downstream tool execution. Benchmark your full request path on your target hardware, and monitor `ActionLedger` write latency as the SQLite hash chain grows. Avoid installing the unrelated `reflex` web framework in the same virtual environment as `system1`, because `system1` provides the legacy `reflex` compatibility namespace.
