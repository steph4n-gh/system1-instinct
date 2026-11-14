# Add an explicit tool policy

A taught skill predicts a structured decision, not permission to execute a privileged tool. When a decision triggers a tool call, use `SystemOneGuard` with `enforcement_profile=True`, `PolicyEngine`, an Ed25519 signing key, and `ActionLedger` to enforce deterministic permissions after [teaching your first skill](first_skill.md).

## Authorize a tool call and record a signed receipt

To authorize a single `read_config` lookup for `"agent-worker"`, record the execution outcome in a hash-chained SQLite ledger, and verify the signed chain, run:

```python
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from system1 import (
    ActionLedger, ActionProposal, DecisionOutcome, PolicyEngine, PolicyRule,
    RiskLevel, SystemOneGuard, load_private_key, save_keypair,
)

key_path = Path(".system1/demo-identity/identity.key")
if not key_path.exists():
    save_keypair(Ed25519PrivateKey.generate(), key_path.parent)
signing_key = load_private_key(key_path)

policy = PolicyEngine(rules=[PolicyRule(
    rule_id="read_service_name",
    tools=["read_config"],
    allowed_principals=["agent-worker"],
    allowed_tenants=["demo"],
    allowed_scopes=["config:read"],
    argument_limits={"key": ["service_name"]},
    outcome=DecisionOutcome.ALLOW,
    risk=RiskLevel.READ_ONLY,
)])

with ActionLedger(".system1/demo-audit.sqlite", require_durable=True) as ledger:
    guard = SystemOneGuard(
        policy=policy, ledger=ledger, signing_key=signing_key,
        enforcement_profile=True,
    )
    proposal = ActionProposal.create(
        tenant_id="demo", principal_id="agent-worker", scope="config:read",
        tool="read_config", arguments={"key": "service_name"},
        canonical_target="config:service_name", purpose="Inspect service name",
    )
    auth = guard.evaluate_proposal(proposal)
    if auth.outcome != DecisionOutcome.ALLOW:
        raise PermissionError(auth.reason)

    # Execute the authorized operation and arguments.
    result = {"service_name": "system1-demo"}[proposal.arguments["key"]]
    ledger.record_execution_outcome(
        action_id=proposal.action_id,
        receipt_digest=auth.receipt.compute_digest(),
        status="SUCCEEDED", result_payload={"value": result},
        tenant_id=proposal.tenant_id, principal_id=proposal.principal_id,
        scope=proposal.scope, trusted_public_key=signing_key.public_key(),
    )
    assert ledger.verify_integrity(trusted_public_key=signing_key.public_key())
    print(result)
```

## Enforce caller identity and argument limits

Derive `tenant_id`, `principal_id`, and `scope` from your authenticated application session, restrict tool arguments and targets in `PolicyRule`, and prevent agent code from calling tools outside `SystemOneGuard`. Without `enforcement_profile=True`, `SystemOneGuard` runs in diagnostic mode and can allow actions through statistical classification. Before granting consequential tool permissions, review [Deployment boundaries](../deployment.md).

## Configure ActionLedger WAL durability and coalesced READ_ONLY writes

`ActionLedger` supports four `durability_mode` profiles:

- `"process_wal"` (default): SQLite WAL mode with `synchronous=NORMAL`.
- `"group_commit"`: Background writer thread batching concurrent commits over `group_commit_ms` (default `4.0 ms`).
- `"strict_sync"`: `synchronous=FULL` with `fullfsync` on commit (used when per-commit hardware barriers are required).
- `"high_throughput_normal"`: Non-strict throughput profile (rejected when `enforce_durability=True`).

For enforced persistent ledgers and atomic `READ_ONLY` execution auditing:

```python
with ActionLedger(
    ".system1/demo-audit.sqlite",
    require_durable=True,
    enforce_durability=True,
    durability_mode="process_wal",
) as ledger:
    decision_hash, outcome_hash = ledger.record_coalesced_read_only_execution(
        auth.receipt,
        reversibility="READ_ONLY",
        action_id=proposal.action_id,
        receipt_digest=auth.receipt.compute_digest(),
        status="SUCCEEDED",
        result_payload={"value": "system1-demo"},
        tenant_id=proposal.tenant_id,
        principal_id=proposal.principal_id,
        scope=proposal.scope,
        trusted_public_key=signing_key.public_key(),
    )
```

Mutating or destructive operations continue to require two separate commits (`append` / `record_decision_receipt` before execution, then `record_execution_outcome` after execution). On unmodified connections, `ActionLedger` uses an `O(1)` verification watermark (`PRAGMA data_version` and `total_changes()`) and reverifies the full hash chain whenever external SQLite writes or reopening occur.
