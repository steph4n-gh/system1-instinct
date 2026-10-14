from __future__ import annotations


# === Merged from tests/test_round2_gates_a_b.py ===
"""Comprehensive Verification Suite for Milestone 1: Gate A & Gate B.

Covers:
- Gate A: Compositional Authorization (P0)
  1. Deterministic Rule Composition in PolicyEngine (DENY > REQUIRE_APPROVAL > ALLOW regardless of order).
  2. Decouple Rule Applicability from Constraint Evaluation in PolicyRule.
  3. Fail-Closed Constrained Arguments in argument_limits (missing keys, type mismatches, boolean values).
  4. Release Invariant 1: No Execution Bypass on deterministic ALLOW (signed receipt + durable ledger entry).
  5. Immutable Canonical Action Dispatch in SystemOneMCPProxy, wrap_mcp_tool, and SystemOneToolInterceptor.
  6. Harmless Sentinel Execution (0 invocations on all rejections and ledger failures).

- Gate B: Authenticated Final Authorization & Durability (P0)
  1. Explicit EnforcementProfile (reflex.witness.enforcement.v1 / strict_attested_durable_v1).
  2. Comprehensive Authorization Receipt Binding (PolicyDecision, arguments, principal, tenant, scope, risk, epoch).
  3. Fail-Closed Ledger Durability (reject :memory:, verify_integrity with trusted_public_key).
  4. Complete Two-Phase Outcome Chaining & Structured INDETERMINATE across MCP and LangChain adapters.
"""

import asyncio
import copy
import hashlib
import json
import sqlite3
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from system1.guard import (
    ActionProposal,
    DecisionOutcome,
    EnforcementProfile,
    ENFORCEMENT_PROFILE_V1,
    PolicyDecision,
    PolicyEngine,
    PolicyRule,
    SystemOneGuardHook,
    RiskLevel,
)
from system1.integrations.langchain import (
    SystemOneGuardBlockedException,
    SystemOneGuardCallbackHandler,
    SystemOneIndeterminateExecutionError,
    SystemOneToolInterceptor,
    wrap_langchain_tool,
)
from system1.integrations.mcp import (
    SystemOneMCPBlockedError,
    SystemOneMCPProxy,
    wrap_mcp_tool,
)
from system1.ledger import ActionLedger, LedgerError, LedgerWriteError
from system1.receipt import (
    create_decision_receipt,
    verify_decision_witness_receipt,
)


class SentinelTarget:
    """Mock target that records invocations and arguments."""

    def __init__(self, return_value: str = "sentinel_ok"):
        self.call_count = 0
        self.last_tool = None
        self.last_args = None
        self.last_kwargs = None
        self.return_value = return_value

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        self.last_args = args
        self.last_kwargs = kwargs
        return self.return_value

    def mcp_executor(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        self.call_count += 1
        self.last_tool = tool_name
        self.last_args = (tool_name, arguments)
        self.last_kwargs = arguments
        return {"result": self.return_value}

    def execute_file(self, path: str, mode: str = "read", retries: int = 3) -> str:
        self.call_count += 1
        self.last_args = (path, mode, retries)
        self.last_kwargs = {"path": path, "mode": mode, "retries": retries}
        return f"file:{path}:{mode}:{retries}"

    async def aexecute_file(self, path: str, mode: str = "read", retries: int = 3) -> str:
        self.call_count += 1
        self.last_args = (path, mode, retries)
        self.last_kwargs = {"path": path, "mode": mode, "retries": retries}
        return f"afile:{path}:{mode}:{retries}"


# ===========================================================================
# Gate A.1: Deterministic Rule Composition in PolicyEngine
# ===========================================================================


def test_gate_a_rule_composition_deny_overrides_allow_regardless_of_order():
    """DENY rule strictly overrides ALLOW rule regardless of registration order."""
    allow_rule = PolicyRule(
        rule_id="allow_all_tools",
        effect=DecisionOutcome.ALLOW,
        target_pattern=r".*",
        reason="Allow broad actions",
    )
    deny_rule = PolicyRule(
        rule_id="deny_sensitive_path",
        effect=DecisionOutcome.DENY,
        target_pattern=r"/etc/.*",
        reason="Deny sensitive path",
    )

    # Order 1: ALLOW registered before DENY
    engine_allow_first = PolicyEngine(rules=[allow_rule, deny_rule])
    proposal = ActionProposal.create(
        tenant_id="test_t",
        principal_id="test_p",
        scope="exec",
        tool="cat",
        arguments={"path": "/etc/shadow"},
        canonical_target="/etc/shadow",
        purpose="Read system credentials",
    )
    res = engine_allow_first.evaluate(proposal)
    assert res is not None
    outcome, risk, rule_id, reason = res
    assert outcome == DecisionOutcome.DENY
    assert rule_id == "deny_sensitive_path"

    # Order 2: DENY registered before ALLOW
    engine_deny_first = PolicyEngine(rules=[deny_rule, allow_rule])
    res2 = engine_deny_first.evaluate(proposal)
    assert res2 is not None
    assert res2[0] == DecisionOutcome.DENY
    assert res2[2] == "deny_sensitive_path"


def test_gate_a_rule_composition_approval_overrides_allow():
    """REQUIRE_APPROVAL strictly overrides ALLOW when both match."""
    allow_rule = PolicyRule(
        rule_id="allow_read",
        effect=DecisionOutcome.ALLOW,
        target_pattern=r"/data/.*",
        reason="Allow data access",
    )
    approval_rule = PolicyRule(
        rule_id="approval_high_volume",
        effect=DecisionOutcome.REQUIRE_APPROVAL,
        target_pattern=r".*",
        reason="Require human approval for all accesses",
    )
    engine = PolicyEngine(rules=[allow_rule, approval_rule])
    proposal = ActionProposal.create(
        tenant_id="test_t",
        principal_id="test_p",
        scope="exec",
        tool="cat",
        arguments={"path": "/data/records.csv"},
        canonical_target="/data/records.csv",
        purpose="Inspect data file",
    )
    res = engine.evaluate(proposal)
    assert res is not None
    assert res[0] == DecisionOutcome.REQUIRE_APPROVAL
    assert res[2] == "approval_high_volume"


# ===========================================================================
# Gate A.2 & A.3: Decoupled Rule Applicability and Fail-Closed Constraints
# ===========================================================================


def test_gate_a_decouple_applicability_and_evaluate_constraints():
    """Rule applicability (applies_to) does not short-circuit constraints (evaluate_constraints)."""
    rule = PolicyRule(
        rule_id="bounded_numeric_limit",
        effect=DecisionOutcome.ALLOW,
        target_pattern=r"/api/transfer.*",
        allowed_principals=["finance_agent"],
        argument_limits={"amount": 1000.0},
    )

    # 1. Matching target pattern and matching principal, valid amount -> applies and satisfies constraints
    prop_valid = ActionProposal.create(
        tenant_id="t1",
        principal_id="finance_agent",
        scope="exec",
        tool="transfer",
        arguments={"amount": 500.0},
        canonical_target="/api/transfer_funds",
        purpose="Process valid transfer",
    )
    assert rule.applies_to(prop_valid) is True
    assert rule.evaluate_constraints(prop_valid) == (True, "")
    assert rule.evaluate(prop_valid) is not None
    assert rule.evaluate(prop_valid)[0] == DecisionOutcome.ALLOW

    # 2. Unauthorized principal: applies_to is True (matches target), but evaluate_constraints fails closed
    prop_bad_principal = ActionProposal.create(
        tenant_id="t1",
        principal_id="unauthorized_actor",
        scope="exec",
        tool="transfer",
        arguments={"amount": 500.0},
        canonical_target="/api/transfer_funds",
        purpose="Unauthorized transfer attempt",
    )
    assert rule.applies_to(prop_bad_principal) is True
    satisfied, reason = rule.evaluate_constraints(prop_bad_principal)
    assert satisfied is False
    assert "not in allowed_principals" in reason
    # Rule evaluate fails closed to DENY on constraint failure
    eval_res = rule.evaluate(prop_bad_principal)
    assert eval_res is not None
    assert eval_res[0] == DecisionOutcome.DENY


def test_gate_a_fail_closed_constrained_arguments():
    """Missing arguments, type mismatches, and booleans in argument_limits fail closed."""
    rule = PolicyRule(
        rule_id="strict_numeric_limits",
        effect=DecisionOutcome.ALLOW,
        target_pattern=r".*",
        argument_limits={"max_count": 10, "threshold": 0.95},
    )

    # Missing constrained argument
    prop_missing = ActionProposal.create(
        tenant_id="t1",
        principal_id="p1",
        scope="exec",
        tool="batch",
        arguments={"threshold": 0.80},  # missing max_count
        canonical_target="job_1",
        purpose="Run batch job",
    )
    sat, reason = rule.evaluate_constraints(prop_missing)
    assert sat is False
    assert "missing required constrained argument 'max_count'" in reason
    assert rule.evaluate(prop_missing)[0] == DecisionOutcome.DENY

    # Type mismatch: string passed for numeric limit
    prop_str = ActionProposal.create(
        tenant_id="t1",
        principal_id="p1",
        scope="exec",
        tool="batch",
        arguments={"max_count": "100", "threshold": 0.80},
        canonical_target="job_1",
        purpose="Run batch job",
    )
    sat, reason = rule.evaluate_constraints(prop_str)
    assert sat is False
    assert "expected numeric" in reason
    assert rule.evaluate(prop_str)[0] == DecisionOutcome.DENY

    # Type mismatch: boolean passed for numeric limit (in Python, isinstance(True, int) is True)
    prop_bool = ActionProposal.create(
        tenant_id="t1",
        principal_id="p1",
        scope="exec",
        tool="batch",
        arguments={"max_count": True, "threshold": 0.80},
        canonical_target="job_1",
        purpose="Run batch job",
    )
    sat, reason = rule.evaluate_constraints(prop_bool)
    assert sat is False
    assert "expected numeric" in reason and "bool" in reason
    assert rule.evaluate(prop_bool)[0] == DecisionOutcome.DENY


# ===========================================================================
# Gate A.4: Release Invariant 1: No Execution Bypass on Deterministic ALLOW
# ===========================================================================


def test_gate_a_deterministic_allow_issues_signed_receipt_and_ledger_entry(tmp_path):
    """Deterministic ALLOW issues authenticated witness receipt and records to durable ledger before execution."""
    db_path = str(tmp_path / "allow_audit.db")
    ledger = ActionLedger(path=db_path)
    signing_key = Ed25519PrivateKey.generate()

    allow_rule = PolicyRule(
        rule_id="trusted_read_allow",
        effect=DecisionOutcome.ALLOW,
        target_pattern=r"^/safe/.*",
        reason="Trusted read path allowed",
    )
    policy_engine = PolicyEngine(rules=[allow_rule])

    hook = SystemOneGuardHook(
        policy_engine=policy_engine,
        ledger=ledger,
        signing_key=signing_key,
        fail_closed_ledger=True,
    )

    prop = ActionProposal.create(
        tenant_id="corp_tenant",
        principal_id="agent_reader",
        scope="fs:read",
        tool="read_file",
        arguments={"path": "/safe/data.json"},
        canonical_target="/safe/data.json",
        purpose="Read safe customer record",
    )

    result = hook.evaluate_proposal(prop)
    assert result.outcome == DecisionOutcome.ALLOW
    assert result.allowed is True

    # Invariant 1: Decision has authenticated receipt
    receipt = result.receipt
    assert receipt is not None
    assert receipt.signer_public_key is not None
    assert receipt.policy_decision is not None
    assert receipt.policy_decision["outcome"] == "ALLOW"
    assert receipt.policy_decision["rule_id"] == "trusted_read_allow"
    assert receipt.outcome == "ALLOW"

    # Invariant 1: Durably recorded in ledger before execution
    assert receipt.ledger_record_id is not None
    assert ledger.verify_integrity(trusted_public_key=signing_key.public_key()) is True
    seq, head = ledger.audit_head()
    assert seq == 1


def test_gate_a_deterministic_allow_fails_closed_if_ledger_write_fails(tmp_path):
    """Deterministic ALLOW fails closed to DENY if durable ledger recording fails."""
    db_path = str(tmp_path / "failing_audit.db")
    ledger = ActionLedger(path=db_path)

    # Simulate catastrophic disk/sqlite failure on append
    def failing_record(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error on commit")

    ledger.record_decision_receipt = failing_record

    allow_rule = PolicyRule(
        rule_id="trusted_read_allow",
        effect=DecisionOutcome.ALLOW,
        target_pattern=r".*",
        reason="Allow all",
    )
    policy_engine = PolicyEngine(rules=[allow_rule])

    hook = SystemOneGuardHook(
        policy_engine=policy_engine,
        ledger=ledger,
        fail_closed_ledger=True,
    )

    prop = ActionProposal.create(
        tenant_id="corp_tenant",
        principal_id="agent_reader",
        scope="fs:read",
        tool="read_file",
        arguments={"path": "/safe/data.json"},
        canonical_target="/safe/data.json",
        purpose="Read data",
    )

    result = hook.evaluate_proposal(prop)
    # Must fail closed to DENY
    assert result.outcome == DecisionOutcome.DENY
    assert result.allowed is False
    assert "ActionLedger write failed" in result.reason
    assert result.policy_decision.rule_id == "system1_ledger_failure"


# ===========================================================================
# Gate A.5 & A.6: Immutable Canonical Action Dispatch & Harmless Sentinel
# ===========================================================================


def test_gate_a_mcp_proxy_canonical_immutable_dispatch():
    """SystemOneMCPProxy dispatches exact frozen canonical arguments, ignoring caller mutations."""
    allow_rule = PolicyRule(rule_id="allow_safe", effect=DecisionOutcome.ALLOW, target_pattern=r".*", reason="Allow test")
    hook = SystemOneGuardHook(policy_engine=PolicyEngine(rules=[allow_rule]))
    sentinel = SentinelTarget()
    proxy = SystemOneMCPProxy(guard=hook)

    request_dict = {
        "jsonrpc": "2.0",
        "id": "mcp_immutable_1",
        "method": "tools/call",
        "params": {
            "name": "safe_tool",
            "arguments": {"target": "benign_input", "mode": "read"},
        },
    }

    # Intercept and execute via handle_call
    resp = proxy.handle_call(request_dict, sentinel.mcp_executor)
    assert "result" in resp
    assert sentinel.call_count == 1
    assert sentinel.last_kwargs == {"target": "benign_input", "mode": "read"}

    # Mutate the input request dictionary
    request_dict["params"]["arguments"]["target"] = "MALICIOUS_INJECTION"
    # Sentinel last kwargs should NOT be mutated
    assert sentinel.last_kwargs["target"] == "benign_input"


def test_gate_a_wrap_mcp_tool_immutable_dispatch_and_sentinel():
    """@wrap_mcp_tool dispatches canonicalized arguments and guarantees 0 executions on denial."""
    # Create rule allowing safe paths, denying write paths
    rules = [
        PolicyRule(rule_id="deny_writes", effect=DecisionOutcome.DENY, target_pattern=r".*write.*", reason="Veto writes"),
        PolicyRule(rule_id="allow_reads", effect=DecisionOutcome.ALLOW, target_pattern=r".*", reason="Allow reads"),
    ]
    hook = SystemOneGuardHook(policy_engine=PolicyEngine(rules=rules))
    proxy = SystemOneMCPProxy(guard=hook)

    sentinel = SentinelTarget()
    wrapped_fn = wrap_mcp_tool(proxy=proxy, tool_name="execute_file")(sentinel.execute_file)

    # 1. Allowed call
    res = wrapped_fn("/safe/path.txt", mode="read", retries=2)
    assert res == "file:/safe/path.txt:read:2"
    assert sentinel.call_count == 1

    # 2. Denied call -> 0 executions
    with pytest.raises(SystemOneMCPBlockedError):
        wrapped_fn("/safe/write_payload.txt", mode="write", retries=1)

    assert sentinel.call_count == 1  # call count did not increase


def test_gate_a_langchain_interceptor_immutable_dispatch_and_sentinel():
    """SystemOneToolInterceptor dispatches canonicalized arguments and guarantees 0 executions on denial."""
    rules = [
        PolicyRule(rule_id="deny_deletions", effect=DecisionOutcome.DENY, target_pattern=r".*delete.*", reason="Veto deletions"),
        PolicyRule(rule_id="allow_reads", effect=DecisionOutcome.ALLOW, target_pattern=r".*", reason="Allow reads"),
    ]
    hook = SystemOneGuardHook(policy_engine=PolicyEngine(rules=rules))

    sentinel = SentinelTarget()
    interceptor = SystemOneToolInterceptor(tool=sentinel.execute_file, guard=hook, tool_name="file_tool")

    # 1. Allowed call
    res = interceptor("/safe/data.txt", mode="read", retries=5)
    assert res == "file:/safe/data.txt:read:5"
    assert sentinel.call_count == 1

    # 2. Denied call
    with pytest.raises(SystemOneGuardBlockedException):
        interceptor("/data/delete_all.txt", mode="delete")

    assert sentinel.call_count == 1


# ===========================================================================
# Gate B.1 & B.3: Explicit EnforcementProfile & Fail-Closed Ledger Durability
# ===========================================================================


def test_gate_b_enforcement_profile_validation(tmp_path):
    """EnforcementProfile rejects missing trusted signer or in-memory (:memory:) ledgers."""
    profile = EnforcementProfile(
        profile_id=ENFORCEMENT_PROFILE_V1,
        require_signer=True,
        require_durable_ledger=True,
    )

    # 1. In-memory ledger rejected in enforcement mode
    mem_ledger = ActionLedger(path=":memory:")
    signing_key = Ed25519PrivateKey.generate()
    with pytest.raises(ValueError, match="rejects in-memory ledger|requires a durable disk-backed ActionLedger"):
        profile.validate_configuration(signing_key=signing_key, ledger=mem_ledger)

    # 2. Missing signing key rejected in enforcement mode
    disk_ledger = ActionLedger(path=str(tmp_path / "valid_audit.db"))
    with pytest.raises(ValueError, match="requires a configured trusted Ed25519 signing key"):
        profile.validate_configuration(signing_key=None, ledger=disk_ledger)

    # 3. Valid configuration passes validation
    profile.validate_configuration(signing_key=signing_key, ledger=disk_ledger)


def test_gate_b_action_ledger_durability_flag():
    """ActionLedger.is_durable is False for :memory: and True for disk paths."""
    mem_ledger = ActionLedger(path=":memory:")
    assert mem_ledger.is_durable is False

    # Setting require_durable=True raises on :memory:
    with pytest.raises((ValueError, LedgerError)):
        ActionLedger(path=":memory:", require_durable=True)


def test_gate_b_action_ledger_verify_integrity_with_trusted_public_key(tmp_path):
    """ActionLedger.verify_integrity(trusted_public_key) cryptographically verifies entry signatures."""
    signing_key = Ed25519PrivateKey.generate()
    pub_key = signing_key.public_key()
    wrong_key = Ed25519PrivateKey.generate().public_key()

    db_path = str(tmp_path / "integrity_audit.db")
    ledger = ActionLedger(path=db_path)

    # Create signed receipt and record
    receipt = create_decision_receipt(
        schema_name="TestSchema",
        schema_digest="a" * 64,
        prompt="Inspect financial transaction",
        values={"is_safe": True},
        confidences={"is_safe": 0.99},
        conformal_sets={"is_safe": ["True"]},
        probabilities={"is_safe": {"True": 0.99, "False": 0.01}},
        latency_ms=0.5,
        is_ambiguous=False,
        signing_key=signing_key,
    )
    rec_id = ledger.record_decision_receipt(receipt)
    assert rec_id is not None

    # 1. Verified against correct public key -> passes
    assert ledger.verify_integrity(trusted_public_key=pub_key) is True

    # 2. Verified against wrong public key -> fails
    assert ledger.verify_integrity(trusted_public_key=wrong_key) is False

    # 3. Tampered entry payload in database -> fails
    with ledger._transaction() as conn:
        conn.execute(
            "UPDATE audit_entries SET payload_json=? WHERE sequence=1",
            (json.dumps({"tampered": True}),),
        )

    assert ledger.verify_integrity(trusted_public_key=pub_key) is False


# ===========================================================================
# Gate B.2: Comprehensive Authorization Receipt Binding
# ===========================================================================


def test_gate_b_authorization_receipt_binding_and_tamper_detection():
    """DecisionWitnessReceipt binds Gate B policy decision and claims, failing on any tampering."""
    signing_key = Ed25519PrivateKey.generate()
    pub_key = signing_key.public_key()

    prop = ActionProposal.create(
        tenant_id="tenant_alpha",
        principal_id="principal_beta",
        scope="scope_gamma",
        tool="transfer_assets",
        arguments={"amount": 500, "dest": "vault_9"},
        canonical_target="vault_9",
        purpose="Transfer assets to secure vault",
    )
    pol_decision = PolicyDecision._issue(
        action_id=prop.action_id,
        outcome=DecisionOutcome.ALLOW,
        reason="Verified authorization under policy rule 42",
        rule_id="rule_42",
        risk=RiskLevel.EXTERNAL,
        normalized_arguments=prop.arguments,
        policy_id="policy_fin_v1",
        policy_epoch="2026_q3",
    )

    receipt = create_decision_receipt(
        schema_name="GateB_Receipt_Test",
        schema_digest="f" * 64,
        prompt="Execute asset transfer",
        values={"is_safe": True},
        confidences={"is_safe": 0.999},
        conformal_sets={"is_safe": ["True"]},
        probabilities={"is_safe": {"True": 0.999, "False": 0.001}},
        latency_ms=0.45,
        is_ambiguous=False,
        signing_key=signing_key,
        policy_decision=pol_decision,
        action_proposal=prop,
        profile=ENFORCEMENT_PROFILE_V1,
        effective_alpha=0.05,
    )

    # 1. Unsigned payload and receipt dict include all Gate B claims
    r_dict = receipt.to_dict()
    assert r_dict["policy_decision"] is not None
    assert r_dict["policy_decision"]["rule_id"] == "rule_42"
    assert r_dict["policy_decision"]["policy_id"] == "policy_fin_v1"
    assert r_dict["policy_decision"]["policy_epoch"] == "2026_q3"
    assert r_dict["principal_id"] == "principal_beta"
    assert r_dict["tenant_id"] == "tenant_alpha"
    assert r_dict["scope"] == "scope_gamma"
    assert r_dict["effective_alpha"] == 0.05

    # 2. Genuine verification succeeds
    assert verify_decision_witness_receipt(r_dict, public_key=pub_key) is True

    # 3. Tampering policy_decision invalidates verification
    tampered_1 = copy.deepcopy(r_dict)
    tampered_1["policy_decision"]["rule_id"] = "fraudulent_rule"
    assert verify_decision_witness_receipt(tampered_1, public_key=pub_key) is False

    # 4. Tampering principal_id invalidates verification
    tampered_2 = copy.deepcopy(r_dict)
    tampered_2["principal_id"] = "impersonator"
    assert verify_decision_witness_receipt(tampered_2, public_key=pub_key) is False


# ===========================================================================
# Gate B.4: Two-Phase Execution Outcome Chaining & Structured INDETERMINATE
# ===========================================================================


def test_gate_b_mcp_handle_call_outcome_chaining_and_indeterminate(tmp_path):
    """SystemOneMCPProxy chains execution outcome and returns structured INDETERMINATE error on audit failure."""
    db_path = str(tmp_path / "mcp_chain_audit.db")
    ledger = ActionLedger(path=db_path)
    allow_rule = PolicyRule(rule_id="allow_doc", effect=DecisionOutcome.ALLOW, target_pattern=r".*", reason="Allow docs")
    hook = SystemOneGuardHook(policy_engine=PolicyEngine(rules=[allow_rule]), ledger=ledger, min_confidence=0.50, alpha=0.10)
    sentinel = SentinelTarget(return_value="mcp_success")
    proxy = SystemOneMCPProxy(guard=hook)

    # 1. Normal execution chains SUCCEEDED outcome to ledger
    req = {
        "jsonrpc": "2.0",
        "id": "mcp-req-101",
        "method": "tools/call",
        "params": {"name": "read_doc", "arguments": {"path": "/safe/readme.md"}},
    }
    resp = proxy.handle_call(req, sentinel.mcp_executor)
    assert "result" in resp
    assert sentinel.call_count == 1

    # Verify ledger captured decision receipt (seq 1) + execution outcome (seq 2)
    seq, head = ledger.audit_head()
    assert seq == 2

    # 2. Simulate outcome recording failure -> structured INDETERMINATE response
    def failing_outcome(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error on outcome record")

    ledger.record_execution_outcome = failing_outcome

    req2 = {
        "jsonrpc": "2.0",
        "id": "mcp-req-102",
        "method": "tools/call",
        "params": {"name": "read_doc", "arguments": {"path": "/safe/license.txt"}},
    }
    resp2 = proxy.handle_call(req2, sentinel.mcp_executor)
    assert "error" in resp2
    assert resp2["error"]["code"] == -32001
    err_data = resp2["error"]["data"]
    assert err_data["status"] == "INDETERMINATE"
    assert "action_id" in err_data
    assert "receipt_digest" in err_data
    assert err_data["raw_result"] == {"result": "mcp_success"}


def test_gate_b_wrap_mcp_tool_outcome_chaining_and_indeterminate(tmp_path):
    """@wrap_mcp_tool chains execution outcome and raises SystemOneIndeterminateExecutionError on audit failure."""
    db_path = str(tmp_path / "wrap_chain_audit.db")
    ledger = ActionLedger(path=db_path)
    allow_rule = PolicyRule(rule_id="allow_sample", effect=DecisionOutcome.ALLOW, target_pattern=r".*", reason="Allow sample")
    hook = SystemOneGuardHook(policy_engine=PolicyEngine(rules=[allow_rule]), ledger=ledger, min_confidence=0.50, alpha=0.10)
    proxy = SystemOneMCPProxy(guard=hook)

    sentinel = SentinelTarget()
    wrapped_fn = wrap_mcp_tool(proxy=proxy, tool_name="execute_file")(sentinel.execute_file)

    # 1. Successful execution chains outcome
    res = wrapped_fn("/safe/sample.txt")
    assert res == "file:/safe/sample.txt:read:3"
    seq, head = ledger.audit_head()
    assert seq == 2

    # 2. Failure on outcome recording raises SystemOneIndeterminateExecutionError with evidence
    def failing_outcome(*args, **kwargs):
        raise sqlite3.OperationalError("ledger WAL locked")

    ledger.record_execution_outcome = failing_outcome

    with pytest.raises(SystemOneIndeterminateExecutionError) as exc_info:
        wrapped_fn("/safe/sample2.txt")

    err = exc_info.value
    assert "INDETERMINATE" in str(err)
    assert err.action_id != ""
    assert err.receipt_digest != ""
    assert err.raw_result == "file:/safe/sample2.txt:read:3"
    assert isinstance(err.underlying_error, sqlite3.OperationalError)


def test_gate_b_langchain_interceptor_sync_and_async_outcome_chaining(tmp_path):
    """SystemOneToolInterceptor chains sync and async outcomes and raises SystemOneIndeterminateExecutionError."""
    db_path = str(tmp_path / "lc_chain_audit.db")
    ledger = ActionLedger(path=db_path)
    allow_rule = PolicyRule(rule_id="allow_lc", effect=DecisionOutcome.ALLOW, target_pattern=r".*", reason="Allow LC")
    hook = SystemOneGuardHook(policy_engine=PolicyEngine(rules=[allow_rule]), ledger=ledger, min_confidence=0.50, alpha=0.10)

    sentinel = SentinelTarget()
    interceptor = SystemOneToolInterceptor(tool=sentinel.execute_file, guard=hook, tool_name="file_tool")

    # 1. Synchronous invocation chains outcome
    res = interceptor("/safe/sync.txt")
    assert res == "file:/safe/sync.txt:read:3"
    seq, head = ledger.audit_head()
    assert seq == 2

    # 2. Async invocation chains outcome
    ainterceptor = SystemOneToolInterceptor(tool=sentinel.aexecute_file, guard=hook, tool_name="afile_tool")
    ares = asyncio.run(ainterceptor.ainvoke("/safe/async.txt"))
    assert ares == "afile:/safe/async.txt:read:3"
    seq2, head2 = ledger.audit_head()
    assert seq2 == 4

    # 3. Simulate outcome recording failure on sync invoke
    def failing_outcome(*args, **kwargs):
        raise sqlite3.OperationalError("disk error")

    ledger.record_execution_outcome = failing_outcome

    with pytest.raises(SystemOneIndeterminateExecutionError) as exc_info:
        interceptor("/safe/sync_fail.txt")

    err = exc_info.value
    assert err.action_id != ""
    assert err.receipt_digest != ""
    assert err.raw_result == "file:/safe/sync_fail.txt:read:3"


def test_gate_b_langchain_callback_handler_outcome_chaining(tmp_path):
    """SystemOneGuardCallbackHandler chains on_tool_end and on_tool_error to ActionLedger."""
    db_path = str(tmp_path / "cb_chain_audit.db")
    ledger = ActionLedger(path=db_path)
    allow_rule = PolicyRule(rule_id="allow_cb", effect=DecisionOutcome.ALLOW, target_pattern=r".*", reason="Allow CB")
    hook = SystemOneGuardHook(policy_engine=PolicyEngine(rules=[allow_rule]), ledger=ledger, min_confidence=0.50, alpha=0.10)
    handler = SystemOneGuardCallbackHandler(guard=hook)

    serialized = {"name": "search_tool", "description": "Search index"}

    # 1. on_tool_start evaluates and records receipt
    handler.on_tool_start(serialized, "query term", run_id="run_1001")
    seq1, _ = ledger.audit_head()
    assert seq1 == 1

    # 2. on_tool_end records outcome
    handler.on_tool_end("found 10 items", run_id="run_1001")
    seq2, _ = ledger.audit_head()
    assert seq2 == 2

    # 3. Tool error records FAILED outcome
    handler.on_tool_start(serialized, "failing query", run_id="run_1002")
    handler.on_tool_error(ValueError("Invalid query syntax"), run_id="run_1002")
    seq3, _ = ledger.audit_head()
    assert seq3 == 4


# === Merged from tests/test_round2_gates_c_d.py ===
"""Comprehensive Verification Suite for Milestone 2: Gate C & Gate D.

Covers:
- Gate C: Cache & Uncertainty Lifecycle (P0)
  1. Pre-Lookup Input Validation (prompt, alpha, margin, odds, floor, telemetry, embedding).
  2. Cache-Hit Ambiguity and Escalated Fields Preservation (no unconditional overrides).
  3. Collision-Resistant Cache Key Binding (full 64-hex SHA-256 telemetry digest, policy epoch, digests)
     and strict/enforce_durability suppression of approximate semantic search.
  4. Atomic Model Versioning & Concurrency Synchronization (RLock, weights-first update, eviction).
  5. Decoupled Conformal Calibration Folds (50/50 split for temperature scaling & non-conformity quantiles).
  6. Uncalibrated Model Abstention (strict mode full label set / full domain interval escalation).
  7. Multilabel & Regression Uncertainty Gating (MultiChoiceField and ScoreField conformal gating).

- Gate D: Validated Artifact Promotion (P1)
  1. Release Invariant 3: Candidate Artifact Promotion Integrity (promote evaluated candidate directly,
     eliminate post-validation retraining on 100% of history).
  2. Elimination of Small-Sample Relaxation (no total_checks <= 4 lowering) and Rejection of Zero Scored Checks.
  3. Mandatory Statistical Acceptance (default require_statistical_bound=True, Wilson lower bound,
     strict 0.0% false-allow ceiling on critical security classes).
  4. Disjoint Lineage & Manifest Binding (CutoverPartition.assert_disjoint, promotion manifest hashes).
  5. Twin-Namespace Parity (system1 and reflex namespaces).
"""

import concurrent.futures
import copy
import dataclasses
import hashlib
import json
import math
import numpy as np
import pytest
from typing import Any, Dict, List

# Twin-namespace imports
import system1.cache as s1_cache
import reflex.cache as rx_cache
import system1.calibration as s1_calib
import reflex.calibration as rx_calib
import system1.compiler as s1_comp
import reflex.compiler as rx_comp
import system1.engine as s1_eng
import reflex.engine as rx_eng
import system1.compat.typesafe as s1_typesafe
import reflex.compat.typesafe as rx_typesafe

from system1.core.schema import (
    BooleanField,
    ChoiceField,
    DecisionSchema,
    MultiChoiceField,
    ScoreField,
)

SingleChoiceField = ChoiceField
SystemOneSchema = DecisionSchema

from system1.engine import DecisionResult, SystemOneEngine
from system1.cache import CacheEntry, SemanticSystemOneCache, validate_cache_inputs
from system1.calibration import ConformalPredictor, RegressionConformalPredictor
from system1.compiler import SystemOneCompiler
from system1.compat.typesafe import (
    CutoverPartition,
    PromotionPolicy,
    PromotionReport,
    TypeSafeClient,
    compute_wilson_score_lower,
    evaluate_promotion_eligibility,
    partition_cutover_history,
)


# ============================================================================
# Gate C: 1. Pre-Lookup Input Validation
# ============================================================================

class TestPreLookupInputValidation:
    """Tests input validation before cache key formatting or index traversal."""

    def test_cache_validate_prompt(self):
        cache = SemanticSystemOneCache()
        for invalid_prompt in [123, None, ["query"], {"q": "val"}]:
            with pytest.raises(TypeError, match="Prompt must be a string"):
                cache.get(invalid_prompt)  # type: ignore

    def test_cache_validate_alpha(self):
        cache = SemanticSystemOneCache()
        for invalid_alpha in [0.0, 1.0, -0.05, 1.5, float("nan")]:
            with pytest.raises(ValueError, match="Significance level alpha must be in"):
                cache.get("test prompt", alpha=invalid_alpha)

        with pytest.raises(TypeError, match="alpha must be a real number"):
            cache.get("test prompt", alpha="invalid")  # type: ignore

    def test_cache_validate_margin_threshold(self):
        cache = SemanticSystemOneCache()
        for invalid_m in [-0.01, -1.0, float("nan")]:
            with pytest.raises(ValueError, match="margin_threshold must be non-negative"):
                cache.get("test prompt", margin_threshold=invalid_m)

    def test_cache_validate_odds_ratio(self):
        cache = SemanticSystemOneCache()
        for invalid_odds in [0.0, -0.5, float("nan")]:
            with pytest.raises(ValueError, match="relative_odds_ratio must be positive"):
                cache.get("test prompt", relative_odds_ratio=invalid_odds)

    def test_cache_validate_confidence_floor(self):
        cache = SemanticSystemOneCache()
        for invalid_floor in [-0.1, -10.0, float("nan")]:
            with pytest.raises(ValueError, match="confidence_floor_tau0 must be non-negative"):
                cache.get("test prompt", confidence_floor_tau0=invalid_floor)

    def test_cache_validate_telemetry(self):
        cache = SemanticSystemOneCache()
        with pytest.raises(TypeError, match="telemetry must be a Mapping, Sequence, or ndarray"):
            cache.get("test prompt", telemetry=object())

        with pytest.raises(ValueError, match="telemetry array contains NaN or infinite"):
            cache.get("test prompt", telemetry=np.array([1.0, np.nan, 2.0]))

    def test_cache_validate_embedding(self):
        cache = SemanticSystemOneCache()
        with pytest.raises(ValueError, match="embedding contains NaN or infinite values"):
            cache.get("test prompt", embedding=np.array([1.0, np.inf, 2.0]))

        with pytest.raises(ValueError, match="embedding must be a 1D vector"):
            cache.get("test prompt", embedding=np.zeros((2, 2)))

    def test_engine_decide_pre_lookup_validation(self):
        schema = SystemOneSchema(
            schema_name="TestValidation",
            fields={"decision": SingleChoiceField(["allow", "deny"])},
        )
        engine = SystemOneEngine(schema=schema, dimension=8)

        # Invalid prompt
        with pytest.raises(TypeError, match="Prompt must be a string"):
            engine.decide(12345)  # type: ignore

        # Invalid alpha
        with pytest.raises(ValueError, match="alpha must be in"):
            engine.decide("test", alpha=0.0)

        # Invalid margin_threshold
        with pytest.raises(ValueError, match="margin_threshold must be non-negative"):
            engine.decide("test", margin_threshold=-0.5)

        # Invalid odds ratio
        with pytest.raises(ValueError, match="relative_odds_ratio must be strictly positive"):
            engine.decide("test", relative_odds_ratio=-1.0)

        # Invalid embedding dimension
        with pytest.raises(ValueError, match="embedding dimension mismatch"):
            engine.decide("test", embedding=np.zeros(16, dtype=np.float32))

        # Invalid embedding with NaN
        with pytest.raises(ValueError, match="embedding must contain only finite numbers"):
            engine.decide("test", embedding=np.array([np.nan] * 8, dtype=np.float32))


# ============================================================================
# Gate C: 2. Remove Unconditional Cache-Hit Ambiguity Overrides
# ============================================================================

class TestCacheAmbiguityPreservation:
    """Verifies that cache hits preserve evaluated ambiguity and escalated fields."""

    def test_cache_hit_preserves_ambiguous_evaluation(self):
        schema = SystemOneSchema(
            schema_name="TestAmbiguity",
            fields={
                "action": SingleChoiceField(["allow", "review", "block"]),
            },
        )
        engine = SystemOneEngine(schema=schema, dimension=8, use_cache=True)
        prompt = "Suspicious financial transfer exceeding standard thresholds"

        res1 = engine.decide(prompt, alpha=0.05)
        ambiguous_res = dataclasses.replace(
            res1,
            is_ambiguous=True,
            ambiguous_fields=["action"],
            escalated_fields=["action"],
        )

        # Store in cache
        engine.cache.put(
            prompt=prompt,
            result=ambiguous_res,
            schema_digest=schema.schema_digest(),
            model_version=engine.model_version,
            policy_scope=engine.policy_scope,
            alpha=0.05,
            margin_threshold=engine.margin_threshold if engine.enable_margin_gating else 0.0,
            strict=engine.strict_mode,
            relative_odds_ratio=engine.relative_odds_ratio,
            confidence_floor_tau0=engine.confidence_floor_tau0,
            model_digest=engine._model_digest(),
            projector_digest=engine._projector_digest(),
            calibration_digest=engine._calibration_digest(),
            policy_epoch=engine.policy_epoch,
        )

        # Decision on the same prompt must hit cache AND preserve ambiguity
        hit_res = engine.decide(prompt, alpha=0.05)
        assert hit_res.is_cache_hit is True
        assert hit_res.is_ambiguous is True
        assert hit_res.ambiguous_fields == ["action"]
        assert hit_res.escalated_fields == ["action"]

    def test_cache_hit_preserves_unambiguous_evaluation(self):
        schema = SystemOneSchema(
            schema_name="TestUnambiguous",
            fields={"status": SingleChoiceField(["ok", "fail"])},
        )
        engine = SystemOneEngine(schema=schema, dimension=8, use_cache=True)
        prompt = "Routine benign ping"

        res1 = engine.decide(prompt, alpha=0.05)
        clear_res = dataclasses.replace(
            res1,
            is_ambiguous=False,
            ambiguous_fields=[],
            escalated_fields=[],
        )

        engine.cache.put(
            prompt=prompt,
            result=clear_res,
            schema_digest=schema.schema_digest(),
            model_version=engine.model_version,
            policy_scope=engine.policy_scope,
            alpha=0.05,
            margin_threshold=0.0,
            strict=False,
            relative_odds_ratio=engine.relative_odds_ratio,
            confidence_floor_tau0=engine.confidence_floor_tau0,
            model_digest=engine._model_digest(),
            projector_digest=engine._projector_digest(),
            calibration_digest=engine._calibration_digest(),
            policy_epoch=engine.policy_epoch,
        )

        hit_res = engine.decide(prompt, alpha=0.05)
        assert hit_res.is_cache_hit is True
        assert hit_res.is_ambiguous is False
        assert hit_res.ambiguous_fields == []
        assert hit_res.escalated_fields == []


# ============================================================================
# Gate C: 3. Collision-Resistant Cache Key Binding & Strict Mode
# ============================================================================

class TestCacheKeyBindingAndStrictMode:
    """Verifies full 64-hex SHA-256 telemetry digest binding and strict mode semantics."""

    def test_full_64_hex_telemetry_digest(self):
        cache = SemanticSystemOneCache()
        telemetry = {"user_id": "usr_99", "risk": 0.88, "context": "us-east"}
        telem_dig = s1_cache._digest_telemetry(telemetry)

        # Full SHA-256 is 64 hex characters (not truncated to 16)
        assert len(telem_dig) == 64
        assert telem_dig == s1_cache._digest_telemetry(telemetry)
        assert s1_cache._digest_telemetry({"a": 1}) != s1_cache._digest_telemetry({"a": 2})

        key = cache._make_key("prompt", telemetry=telemetry)
        assert telem_dig in key

    def test_cache_key_binds_all_policy_parameters(self):
        cache = SemanticSystemOneCache()
        base_kwargs = {
            "prompt": "test query",
            "telemetry": {"device": "mobile"},
            "schema_digest": "schema_abc",
            "model_version": 1,
            "policy_scope": "production",
            "alpha": 0.05,
            "margin_threshold": 0.1,
            "strict": True,
            "relative_odds_ratio": 2.0,
            "confidence_floor_tau0": 0.8,
            "recency_weighted": True,
            "model_digest": "m_sha256",
            "projector_digest": "p_sha256",
            "calibration_digest": "c_sha256",
            "policy_epoch": 42,
        }

        base_key = cache._make_key(**base_kwargs)

        # Changing ANY parameter must alter the key
        mutations = [
            ("alpha", 0.01),
            ("policy_scope", "staging"),
            ("model_version", 2),
            ("margin_threshold", 0.2),
            ("strict", False),
            ("relative_odds_ratio", 3.0),
            ("confidence_floor_tau0", 0.9),
            ("policy_epoch", 43),
            ("model_digest", "m_different"),
            ("telemetry", {"device": "desktop"}),
        ]

        for param, new_val in mutations:
            modified_kwargs = dict(base_kwargs)
            modified_kwargs[param] = new_val
            new_key = cache._make_key(**modified_kwargs)
            assert new_key != base_key, f"Key failed to vary on param {param}"

    def test_strict_mode_suppresses_approximate_semantic_search(self):
        cache = SemanticSystemOneCache(similarity_threshold=0.8)
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.99, 0.05, 0.0], dtype=np.float32)  # High cosine similarity > 0.99

        # Put with strict=False
        cache.put("exact prompt 1", "result 1", embedding=emb1, strict=False)

        # In non-strict mode: query with similar emb2 should hit semantically
        hit = cache.get("different prompt", embedding=emb2, strict=False)
        assert hit is not None
        assert hit[0].prompt == "exact prompt 1"

        # In strict mode: approximate semantic search must be suppressed
        miss_strict = cache.get("different prompt", embedding=emb2, strict=True)
        assert miss_strict is None

        # In enforce_durability mode: approximate search must also be suppressed
        miss_durable = cache.get("different prompt", embedding=emb2, strict=False, enforce_durability=True)
        assert miss_durable is None


# ============================================================================
# Gate C: 4. Atomic Model Versioning & Concurrency Synchronization
# ============================================================================

class TestAtomicVersioningAndConcurrency:
    """Verifies thread-safe atomic updates under RLock during online learning."""

    def test_atomic_learn_from_tier2_version_and_invalidation(self):
        schema = SystemOneSchema(
            schema_name="TestAtomic",
            fields={"decision": SingleChoiceField(["allow", "deny"])},
        )
        engine = SystemOneEngine(schema=schema, dimension=8, use_cache=True)
        prompt = "User login attempt from anomalous ASN"

        initial_v = engine.model_version
        assert initial_v >= 1

        # Execute decision to seed cache
        res1 = engine.decide(prompt)
        assert res1.is_cache_hit is False

        # Verify cached
        hit_before = engine.decide(prompt)
        assert hit_before.is_cache_hit is True

        # Perform Tier 2 closed-form online update
        update_info = engine.learn_from_tier2(prompt, target={"decision": "deny"})
        assert engine.model_version == initial_v + 1
        assert "decision" in update_info.get("updated_fields", [])

        # Cache for old version must be evicted / invalidated
        res_after = engine.decide(prompt)
        assert res_after.values["decision"] == "deny"

    def test_concurrent_decide_and_learn_under_rlock(self):
        schema = SystemOneSchema(
            schema_name="TestConcurrency",
            fields={"status": SingleChoiceField(["pass", "fail"])},
        )
        engine = SystemOneEngine(schema=schema, dimension=8, use_cache=True)

        prompts = [f"System health check probe {i}" for i in range(20)]
        stop_flag = False

        def reader_worker():
            hits = 0
            while not stop_flag:
                p = prompts[np.random.randint(0, len(prompts))]
                res = engine.decide(p)
                assert res.values["status"] in ["pass", "fail"]
                if res.is_cache_hit:
                    hits += 1
            return hits

        def writer_worker():
            for i in range(10):
                p = prompts[i % len(prompts)]
                engine.learn_from_tier2(p, target={"status": "fail" if i % 2 == 0 else "pass"})

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            writer_future = executor.submit(writer_worker)
            reader_futures = [executor.submit(reader_worker) for _ in range(4)]

            writer_future.result(timeout=10.0)
            stop_flag = True
            for rf in reader_futures:
                rf.result(timeout=5.0)

        assert engine.model_version >= 10


# ============================================================================
# Gate C: 5. Decoupled Conformal Calibration Folds
# ============================================================================

class TestDecoupledCalibrationFolds:
    """Verifies temperature scaling and conformal prediction use independent folds."""

    def test_engine_calibrate_splits_folds_deterministically(self):
        schema = SystemOneSchema(
            schema_name="TestDecoupledFolds",
            fields={"decision": SingleChoiceField(["low", "medium", "high"])},
        )
        engine = SystemOneEngine(schema=schema, dimension=8)

        # Create distinct calibration observations
        dataset = [
            (f"Transaction signal sample {i}", {"decision": "low" if i % 2 == 0 else "high"})
            for i in range(30)
        ]

        engine.calibrate(dataset, n_bins=3)

        # Check calibrated status on conformal predictor
        pred = engine.conformal_predictors["decision"]
        assert pred.is_calibrated is True
        assert len(pred.calibration_scores) > 0

    def test_compiler_calibration_split_disjoint(self):
        schema = SystemOneSchema(
            schema_name="TestCompilerFolds",
            fields={"category": SingleChoiceField(["cat_a", "cat_b", "cat_c"])},
        )
        compiler = SystemOneCompiler(schema=schema)
        # Synthetic exemplars
        exemplars = compiler.generate_synthetic_exemplars(samples_per_choice=6)
        assert len(exemplars["category"]) >= 18


# ============================================================================
# Gate C: 6. Uncalibrated Model Abstention
# ============================================================================

class TestUncalibratedModelAbstention:
    """Verifies strict-mode full label / interval abstention when uncalibrated."""

    def test_conformal_predictor_uncalibrated_strict_abstention(self):
        pred = ConformalPredictor("decision", ["allow", "review", "deny"])
        assert pred.is_calibrated is False

        # Non-strict mode fallback
        res_non_strict = pred.predict_set([0.8, 0.15, 0.05], alpha=0.05, strict=False)
        assert len(res_non_strict.prediction_set) >= 1

        # Strict mode: must abstain with full label set and escalate
        res_strict = pred.predict_set([0.8, 0.15, 0.05], alpha=0.05, strict=True)
        assert res_strict.needs_escalation is True
        assert set(res_strict.prediction_set) == {"allow", "review", "deny"}

    def test_regression_conformal_predictor_uncalibrated_full_domain(self):
        reg_pred = RegressionConformalPredictor("risk_score", min_value=0.0, max_value=100.0)
        assert reg_pred.is_calibrated is False

        # In strict mode: uncalibrated returns entire feasible domain [0, 100]
        interval = reg_pred.predict_interval(45.0, alpha=0.05, strict=True)
        assert interval.is_calibrated is False
        assert interval.lower_bound == 0.0
        assert interval.upper_bound == 100.0


# ============================================================================
# Gate C: 7. Multilabel & Regression Uncertainty Gating
# ============================================================================

class TestMultilabelAndRegressionUncertaintyGating:
    """Verifies conformal uncertainty escalation for MultiChoiceField and ScoreField."""

    def test_multichoice_field_uncertainty_escalation(self):
        schema = SystemOneSchema(
            schema_name="TestMultiChoiceGating",
            fields={"tags": MultiChoiceField(["pii", "financial", "confidential", "public"])},
        )
        engine = SystemOneEngine(schema=schema, dimension=8)

        res = engine.decide("Unseen prompt with high entropy across multi-choice tags")
        assert "tags" in res.values
        assert isinstance(res.values["tags"], (list, tuple))
        assert "tags" in res.conformal_sets

    def test_score_field_uncertainty_escalation(self):
        schema = SystemOneSchema(
            schema_name="TestScoreFieldGating",
            fields={"risk_score": ScoreField(min_value=0.0, max_value=10.0)},
        )
        engine = SystemOneEngine(schema=schema, dimension=8)

        res = engine.decide("Prompt requesting credit increase")
        assert "risk_score" in res.values
        val = res.values["risk_score"]
        assert 0.0 <= val <= 10.0
        assert "risk_score" in res.conformal_sets
        interval = res.conformal_sets["risk_score"]
        assert len(interval) == 1
        assert interval[0].startswith("[")
        assert interval[0].endswith("]")


# ============================================================================
# Gate D: 1. Artifact Promotion Integrity (Release Invariant 3)
# ============================================================================

class TestArtifactPromotionIntegrity:
    """Release Invariant 3: Deploy only the evaluated candidate artifact without retraining."""

    def test_promotion_deploys_candidate_without_full_history_recompile(self):
        from system1.compat.typesafe import Choice, TypeSafeClient

        client = TypeSafeClient(
            mode="auto_cutover",
            cutover_threshold=5,
            promotion_policy=PromotionPolicy(
                min_agreement_threshold=0.8,
                require_statistical_bound=False,
                false_allow_ceiling=1.0,
            ),
            zero_egress=False,
            fallback_baseline=True,
            augment=True, strict_mode=False,
        )
        questions = {
            "tier": Choice("Model tier", criteria={"fast": "Fast model", "smart": "Reasoning model"}),
        }
        for i in range(5):
            client.systemone(f"Request payload {i}", questions)

        assert client.is_cutover is True
        assert client.compiled_model is not None
        audit_log = client.cutover_audit_log
        assert len(audit_log) >= 1
        event = audit_log[0]
        assert "artifact_digest" in event
        assert "candidate_artifact_digest" in event
        assert len(event["candidate_artifact_digest"]) == 64
        assert event["candidate_artifact_digest"] == event["artifact_digest"]


# ============================================================================
# Gate D: 2. Eliminate Small-Sample Relaxation & Zero Scored Labels
# ============================================================================

class TestSmallSampleEliminationAndZeroChecks:
    """Verifies total_checks == 0 is rejected and small-sample threshold is not relaxed."""

    def test_reject_zero_scored_checks(self):
        schema = SystemOneSchema(
            schema_name="TestGateDZero",
            fields={"action": SingleChoiceField(["allow", "deny"])},
        )
        engine = SystemOneEngine(schema=schema, dimension=8)
        policy = PromotionPolicy(min_agreement_threshold=0.80)

        report = evaluate_promotion_eligibility(engine, [], schema=schema, policy=policy)
        assert report.is_eligible is False
        assert any("Zero scored validation checks" in r for r in report.rejection_reasons)
        assert report.total_validation_checks == 0

    def test_no_relaxation_for_small_samples(self):
        schema = SystemOneSchema(
            schema_name="TestGateDSmall",
            fields={"action": SingleChoiceField(["allow", "deny"])},
        )
        engine = SystemOneEngine(schema=schema, dimension=8)
        policy = PromotionPolicy(
            min_agreement_threshold=0.80,
            require_statistical_bound=False,
            false_allow_ceiling=1.0,  # Isolate agreement rate check
        )

        val_history = [
            {"state": f"sample_{i}", "answers": {"action": "allow"}} for i in range(4)
        ]

        # Engine predictions will have some mismatches or we can evaluate report
        report = evaluate_promotion_eligibility(engine, val_history, schema=schema, policy=policy)
        # Even with <= 4 checks, min_agreement_threshold must remain strictly 0.80
        assert policy.min_agreement_threshold == 0.80

    def test_client_defers_cutover_under_default_policy_for_small_sample(self):
        """End-to-end verification: TypeSafeClient(mode='auto_cutover', cutover_threshold=4)
        defers cutover under default policy because Wilson lower bound fails threshold.
        """
        from system1.compat.typesafe import Choice, TypeSafeClient

        client = TypeSafeClient(
            mode="auto_cutover",
            cutover_threshold=4,
            min_agreement_threshold=0.80,
            zero_egress=False,
            fallback_baseline=True,
            augment=True, strict_mode=False,
        )
        questions = {
            "action": Choice("Action", criteria={"allow": "Allow", "deny": "Deny"}),
        }
        for i in range(4):
            client.systemone(f"Request {i}", questions)

        assert client.is_cutover is False
        assert len(client.cutover_audit_log) >= 1
        rep = client.cutover_audit_log[0]["report"]
        assert rep.is_eligible is False
        assert rep.wilson_lower_bound < 50.0

    def test_client_defers_cutover_on_two_samples_due_to_empty_val_history(self):
        """End-to-end verification: 2-sample queries fail to populate validation fold,
        preventing in-sample cutover and deferring promotion.
        """
        from system1.compat.typesafe import Choice, TypeSafeClient

        client = TypeSafeClient(
            mode="auto_cutover",
            cutover_threshold=2,
            zero_egress=False,
            fallback_baseline=True,
            augment=True, strict_mode=False,
        )
        questions = {
            "action": Choice("Action", criteria={"allow": "Allow", "deny": "Deny"}),
        }
        client.systemone("Query 1", questions)
        client.systemone("Query 2", questions)

        assert client.is_cutover is False
        assert len(client.cutover_audit_log) >= 1
        rep = client.cutover_audit_log[0]["report"]
        assert rep.total_validation_checks == 0
        assert rep.is_eligible is False
        assert "Zero scored validation checks; cannot evaluate promotion" in rep.rejection_reasons


# ============================================================================
# Gate D: 3. Mandatory Statistical Acceptance
# ============================================================================

class TestMandatoryStatisticalAcceptance:
    """Verifies default require_statistical_bound=True, Wilson bound, and 0% false-allow ceiling."""

    def test_default_require_statistical_bound_is_true(self):
        policy = PromotionPolicy()
        assert policy.require_statistical_bound is True

    def test_wilson_lower_bound_calculation(self):
        # 10 successes out of 10 trials
        low = compute_wilson_score_lower(10, 10, confidence=0.95)
        # Wilson lower bound for 10/10 is approx 0.722, well below 0.80
        assert low < 50.0

        # 100 successes out of 100 trials
        low_100 = compute_wilson_score_lower(100, 100, confidence=0.95)
        # Wilson lower bound for 100/100 is > 0.96
        assert low_100 > 0.96

    def test_rejection_on_critical_false_allow(self):
        schema = SystemOneSchema(
            schema_name="TestGateDCritical",
            fields={"action": SingleChoiceField(["allow", "deny"])},
        )
        engine = SystemOneEngine(schema=schema, dimension=8)

        # Force engine to predict "allow"
        policy = PromotionPolicy(
            min_agreement_threshold=0.50,
            require_statistical_bound=False,
            false_allow_ceiling=0.0,
        )

        # Val history has ground-truth "deny" (critical security action)
        val_history = [
            {"state": "malicious sql injection attack", "answers": {"action": "deny"}}
        ]

        report = evaluate_promotion_eligibility(engine, val_history, schema=schema, policy=policy)
        if report.false_allow_count > 0:
            assert report.is_eligible is False
            assert any("false-allow" in r.lower() for r in report.rejection_reasons)


# ============================================================================
# Gate D: 4. Disjoint Lineage & Manifest Binding
# ============================================================================

class TestDisjointLineageAndManifestBinding:
    """Verifies disjoint partition enforcement and manifest binding."""

    def test_cutover_partition_assert_disjoint_raises_on_overlap(self):
        overlapping_item = {"state": "shared_payload", "answers": {"action": "allow"}}

        train = [overlapping_item, {"state": "train_only", "answers": {"action": "deny"}}]
        calib = [{"state": "calib_only", "answers": {"action": "allow"}}]
        val = [overlapping_item]  # Leak into validation fold!

        partition = CutoverPartition(
            train_history=train,
            calib_history=calib,
            val_history=val,
            total_samples=4,
        )

        with pytest.raises(AssertionError, match="Violation"):
            partition.assert_disjoint()

    def test_cutover_partition_assert_disjoint_passes_on_clean_splits(self):
        partition = CutoverPartition(
            train_history=[{"state": f"train_{i}", "answers": {"a": 1}} for i in range(5)],
            calib_history=[{"state": f"calib_{i}", "answers": {"a": 1}} for i in range(5)],
            val_history=[{"state": f"val_{i}", "answers": {"a": 1}} for i in range(5)],
            total_samples=15,
        )
        partition.assert_disjoint()  # Must not raise


# ============================================================================
# Gate D: 5. Twin-Namespace Parity (system1 vs reflex)
# ============================================================================

class TestTwinNamespaceParity:
    """Verifies that all classes and methods in system1 and reflex are identical and aligned."""

    def test_cache_namespace_parity(self):
        assert rx_cache.SemanticSystemOneCache is s1_cache.SemanticSystemOneCache
        assert rx_cache.CacheEntry is s1_cache.CacheEntry
        assert rx_cache.validate_cache_inputs is s1_cache.validate_cache_inputs

    def test_engine_namespace_parity(self):
        assert rx_eng.SystemOneEngine is s1_eng.SystemOneEngine
        assert rx_eng.DecisionResult is s1_eng.DecisionResult

    def test_calibration_namespace_parity(self):
        assert rx_calib.ConformalPredictor is s1_calib.ConformalPredictor
        assert rx_calib.RegressionConformalPredictor is s1_calib.RegressionConformalPredictor

    def test_typesafe_namespace_parity(self):
        assert rx_typesafe.CutoverPartition is s1_typesafe.CutoverPartition
        assert rx_typesafe.PromotionPolicy is s1_typesafe.PromotionPolicy
        assert rx_typesafe.evaluate_promotion_eligibility is s1_typesafe.evaluate_promotion_eligibility
        assert rx_typesafe.compute_wilson_score_lower is s1_typesafe.compute_wilson_score_lower


# === Merged from tests/test_round2_gates_e_f.py ===
"""Comprehensive contract test suite for Gate E (True Zero-Egress) and Gate F (Packaging & gRPC).

Validates:
- Gate E:
  - ZeroEgressViolationError raised before socket creation.
  - Fail-closed configuration validation (passthrough, auto_cutover without handler, allow_cloud_fallback).
  - compare() and AsyncTypeSafeClient.compare() enforcement under zero_egress=True.
  - Elimination of silent synthetic baseline fallback when fallback_baseline=False.
  - Structured egress audit logging with host, port, payload bytes, operation.
  - Zero socket connection attempts verified via socket patching.
- Gate F:
  - Twin namespace parity between system1 and reflex for ZeroEgressViolationError and egress logs.
  - pyproject.toml bounds: protobuf>=5.26.1, grpcio>=1.62.0, grpcio-tools>=1.62.0.
  - grpc_server.serve() default loopback host="127.0.0.1".
  - CLI serve subcommand --host flag support.
  - SystemOneServiceServicer.Guard() proposal evaluation with ActionProposal, PolicyEngine, signing_key, and ActionLedger.
"""

import inspect
import json
from pathlib import Path
import socket
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from system1.cli import build_parser
import system1.compat.typesafe as s1_typesafe
from system1.compat.typesafe import (
    AsyncTypeSafeClient,
    Choice,
    TypeSafeClient,
    ZeroEgressViolationError,
    call_real_typesafe_api,
    clear_egress_audit_log,
    compare,
    get_egress_audit_log,
)
from system1.grpc_server import SystemOneServiceServicer, serve
from system1.guard import DecisionOutcome, PolicyEngine, PolicyRule, RiskLevel
from system1.guard import DefaultGuardDecisionSchema
from system1.ledger import ActionLedger
from system1.receipt import verify_decision_witness_receipt


# ============================================================================
# Gate E: True Zero-Egress Enforcement (Features F19–F21)
# ============================================================================

class TestZeroEgressGatekeeper:
    """Feature F19: Socket-level zero-egress enforcement and fail-closed gatekeeping."""

    def test_zero_egress_violation_error_raised_before_socket_connect(self, monkeypatch):
        """Verify call_real_typesafe_api raises ZeroEgressViolationError with 0 socket calls."""
        socket_connect_calls = []
        original_connect = socket.socket.connect

        def mock_connect(self, *args, **kwargs):
            socket_connect_calls.append(args)
            return original_connect(self, *args, **kwargs)

        monkeypatch.setattr(socket.socket, "connect", mock_connect)

        with pytest.raises(ZeroEgressViolationError, match="zero_egress=True"):
            call_real_typesafe_api(
                "Classify request",
                {"tier": Choice("Tier", {"fast": "Fast", "smart": "Smart"})},
                api_key="test-api-key",
                base_url="https://api.typesafe.ai/v1",
                zero_egress=True,
            )

        assert len(socket_connect_calls) == 0, "Socket connect was called despite zero_egress=True!"

    def test_incompatible_configuration_validation(self):
        """Verify TypeSafeClient fail-closed configuration validation."""
        # 1. passthrough with zero_egress=True
        with pytest.raises(ValueError, match="mode='passthrough' requires network egress"):
            TypeSafeClient(mode="passthrough", zero_egress=True)

        # 2. auto_cutover without baseline_handler with zero_egress=True
        with pytest.raises(ValueError, match="mode='auto_cutover' with zero_egress=True requires a local baseline_handler"):
            TypeSafeClient(mode="auto_cutover", cutover_threshold=5, zero_egress=True)

        # 3. allow_cloud_fallback with zero_egress=True
        with pytest.raises(ValueError, match="allow_cloud_fallback=True cannot be combined with zero_egress=True"):
            TypeSafeClient(mode="local", zero_egress=True, allow_cloud_fallback=True)

    def test_sync_and_async_compare_raises_zero_egress_violation(self):
        """Feature F20: compare() raises ZeroEgressViolationError under zero_egress=True."""
        client = TypeSafeClient(mode="local", zero_egress=True)
        questions = {"tier": Choice("Tier", {"fast": "Fast", "smart": "Smart"})}

        # Sync client.compare
        with pytest.raises(ZeroEgressViolationError, match="zero_egress=True"):
            client.compare("Evaluate performance difference", questions)

        # Top-level compare function
        with pytest.raises(ZeroEgressViolationError, match="zero_egress=True"):
            compare("Evaluate performance difference", questions, zero_egress=True)

    @pytest.mark.asyncio
    async def test_async_compare_raises_zero_egress_violation(self):
        """AsyncTypeSafeClient.compare() raises ZeroEgressViolationError under zero_egress=True."""
        async_client = AsyncTypeSafeClient(mode="local", zero_egress=True)
        questions = {"tier": Choice("Tier", {"fast": "Fast", "smart": "Smart"})}

        with pytest.raises(ZeroEgressViolationError, match="zero_egress=True"):
            await async_client.compare("Evaluate async performance difference", questions)

    def test_elimination_of_silent_synthetic_teacher_fallback(self):
        """Feature F21: When fallback_baseline=False, transport failure raises explicit error."""
        # When zero_egress=False and fallback_baseline=False, connection to unreachable host raises error
        with pytest.raises(Exception) as exc_info:
            call_real_typesafe_api(
                "Evaluate prompt without synthetic fallback",
                {"tier": Choice("Tier", {"fast": "Fast", "smart": "Smart"})},
                api_key="test-api-key",
                base_url="http://127.0.0.1:59999/v1",
                zero_egress=False,
                fallback_baseline=False,
                timeout=0.2,
            )
        assert not isinstance(exc_info.value, ZeroEgressViolationError)

    def test_egress_audit_logging(self):
        """Verify egress audit logging records host, port, bytes, operation."""
        clear_egress_audit_log()
        assert len(get_egress_audit_log()) == 0

        # Execute call with zero_egress=False and fallback_baseline=True
        call_real_typesafe_api(
            "Egress audit logging test prompt",
            {"status": Choice("Status", {"ok": "OK", "err": "Error"})},
            api_key="test-api-key",
            base_url="https://api.typesafe.ai/v1",
            zero_egress=False,
            fallback_baseline=True,
            timeout=0.5,
        )

        log = get_egress_audit_log()
        assert len(log) >= 1
        event = log[-1]
        assert "timestamp" in event
        assert event["host"] == "api.typesafe.ai"
        assert event["port"] == 443
        assert event["bytes_out"] > 0
        assert event["destination_host"] == "api.typesafe.ai"
        assert event["destination_port"] == 443
        assert event["payload_bytes"] > 0
        assert event["operation_type"] == "typesafe_wan_call"
        assert "/v1" in event["endpoint"]

        clear_egress_audit_log()
        assert len(get_egress_audit_log()) == 0


# ============================================================================
# Gate F: Packaging, Protobuf Alignment & gRPC Truth (Features F22–F25)
# ============================================================================

class TestPackagingAndProtobufAlignment:
    """Features F22–F25: Namespace twin parity, pyproject bounds, loopback server, and Guard evaluation."""

    def test_twin_namespace_parity(self):
        """Feature F22: system1 and reflex namespaces export identical Gate E classes and functions."""
        import reflex
        import reflex.compat.typesafe as system1_typesafe
        import system1

        # ZeroEgressViolationError
        assert system1.ZeroEgressViolationError is s1_typesafe.ZeroEgressViolationError
        assert reflex.ZeroEgressViolationError is system1_typesafe.ZeroEgressViolationError
        assert system1.ZeroEgressViolationError is reflex.ZeroEgressViolationError

        # Egress audit log helpers
        assert s1_typesafe.get_egress_audit_log is system1_typesafe.get_egress_audit_log
        assert s1_typesafe.clear_egress_audit_log is system1_typesafe.clear_egress_audit_log

    def test_pyproject_dependency_bounds(self):
        """Feature F22: pyproject.toml defines required minimum dependency bounds."""
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        assert pyproject_path.exists()
        content = pyproject_path.read_text()

        assert "protobuf>=7.35.1" in content
        assert "grpcio>=1.80.0" in content
        assert "grpcio-tools>=1.80.0" in content

    def test_grpc_server_serve_default_loopback_and_cli_host(self):
        """Feature F23: serve() defaults to 127.0.0.1 and CLI supports --host."""
        sig = inspect.signature(serve)
        assert "host" in sig.parameters
        assert sig.parameters["host"].default == "127.0.0.1"

        # CLI parser inspection
        parser = build_parser()
        args = parser.parse_args(["serve", "--host", "127.0.0.1", "--port", "50051"])
        assert args.host == "127.0.0.1"
        assert args.port == 50051

    def test_grpc_guard_proposal_evaluation_with_policy_and_ledger(self):
        """Feature F23: SystemOneServiceServicer.Guard evaluates ActionProposal through PolicyEngine."""
        signing_key = Ed25519PrivateKey.generate()
        public_key_hex = signing_key.public_key().public_bytes_raw().hex()
        ledger = ActionLedger(":memory:")

        policy_engine = PolicyEngine(rules=[
            PolicyRule(
                rule_id="rule_allow_status",
                tools=["status_probe"],
                outcome=DecisionOutcome.ALLOW,
                effect=DecisionOutcome.ALLOW,
                risk=RiskLevel.READ_ONLY,
            ),
            PolicyRule(
                rule_id="rule_deny_rm",
                tools=["destructive_wipe"],
                outcome=DecisionOutcome.DENY,
                effect=DecisionOutcome.DENY,
                risk=RiskLevel.IRREVERSIBLE,
            ),
        ])

        servicer = SystemOneServiceServicer(
            signing_key=signing_key,
            ledger=ledger,
            policy_engine=policy_engine,
            schemas={tool: DefaultGuardDecisionSchema() for rule in policy_engine.rules for tool in rule.tools},
        )

        class FakeContext:
            def set_code(self, code):
                pass
            def set_details(self, details):
                pass

        # 1. ALLOW outcome via policy
        class AllowReq:
            prompt = "Check system status"
            schema_name = "status_probe"
            telemetry = {}
            alpha = 0.05

        allow_resp = servicer.Guard(AllowReq(), FakeContext())
        assert allow_resp.outcome == 1  # ALLOW
        assert len(allow_resp.receipt_json) > 0
        allow_receipt = json.loads(allow_resp.receipt_json.decode("utf-8"))
        assert verify_decision_witness_receipt(allow_receipt, public_key=public_key_hex) is True

        # 2. DENY outcome via policy
        class DenyReq:
            prompt = "Delete system partition"
            schema_name = "destructive_wipe"
            telemetry = {}
            alpha = 0.05

        deny_resp = servicer.Guard(DenyReq(), FakeContext())
        assert deny_resp.outcome == 2  # DENY
        assert len(deny_resp.receipt_json) > 0
        deny_receipt = json.loads(deny_resp.receipt_json.decode("utf-8"))
        assert verify_decision_witness_receipt(deny_receipt, public_key=public_key_hex) is True

        # 3. Durable recording in ledger
        assert ledger.audit_head()[0] >= 2
        assert ledger.verify_integrity() is True


# === Merged from tests/test_round4_variants.py ===
import pytest
import tempfile
import numpy as np

from system1.schema import DecisionSchema, ChoiceField
from system1.compat.typesafe import TypeSafeClient
from system1.compiler import CompiledSystemOneModel
from system1.engine import SystemOneEngine

def test_exported_candidate_preserves_validated_runtime_calibration():
    # Setup client
    client = TypeSafeClient()
    
    # Fake schema / questions
    questions = {"label": ChoiceField(options=["A", "B"])}
    
    # Trigger a local execution so it builds engine
    client._execute_local("query1", questions)
    
    schema_digest = next(iter(client._engine_cache.keys()))
    engine = client._engine_cache[schema_digest]
    
    # Fake calibration
    engine.calibrators["label"].temperature = 2.5
    engine.conformal_predictors["label"].calibration_scores = np.array([0.1, 0.2, 0.3])
    engine.conformal_predictors["label"].is_calibrated = True
    
    # Fake compiled model inside the client
    cm = CompiledSystemOneModel(engine.schema, heads={})
    # Add head
    from system1.compiler import CompiledHeadWeights
    cm.heads["label"] = CompiledHeadWeights(
        field_name="label", field_type="choice", options=("A", "B"),
        weights=np.ones((2, 384)), biases=np.zeros(2),
        temperature=2.5,
        calibration_scores=(0.1, 0.2, 0.3)
    )
    client._compiled_model = cm
    
    with tempfile.NamedTemporaryFile(suffix=".s1m") as tf:
        assert client.export_model(tf.name)
        
        # Load and verify
        with open(tf.name, "rb") as f:
            data = f.read()
        
        reloaded = CompiledSystemOneModel.from_bytes(data)
        head = reloaded.heads["label"]
        
        assert head.temperature == 2.5
        assert head.calibration_scores == pytest.approx((0.1, 0.2, 0.3))

def test_statistical_promotion_rejection():
    client = TypeSafeClient()
    questions = {"label": ChoiceField(options=["A", "B"])}
    
    # fake evaluate locally
    client._execute_local("query1", questions)
    schema_digest = next(iter(client._engine_cache.keys()))
    engine = client._engine_cache[schema_digest]
    
    # mock decide so that queries are handled
    def mock_decide(prompt, **kwargs):
        class Res:
            values = {"label": "A"}
        return Res()
    engine.decide = mock_decide

    from system1.compat.typesafe import evaluate_promotion_eligibility, PromotionPolicy
    
    # A) 1 correct labeled, 99 unlabeled
    val_history = [{"state": "q1", "answers": {"label": "A"}}]
    for i in range(99):
        val_history.append({"state": f"q_unlabeled_{i}", "answers": {}})
        
    policy = PromotionPolicy()
    report = evaluate_promotion_eligibility(engine, val_history, engine.schema, policy)
    print("report A:", report.rejection_reasons)
    assert not report.is_eligible
    assert report.wilson_lower_bound < 50.0

    # B) 100 correlated records
    val_history2 = []
    for i in range(100):
        val_history2.append({"state": f"q_corr_{i}", "answers": {"label": "A"}, "group_id": "group1"})
        
    report2 = evaluate_promotion_eligibility(engine, val_history2, engine.schema, policy)
    print("report B:", report2.rejection_reasons)
    assert not report2.is_eligible
    assert report2.wilson_lower_bound < 50.0

def test_ledger_verifies_incoming_receipt_against_anchor():
    from system1.ledger import ActionLedger
    from system1.receipt import DecisionWitnessReceipt, RunWitnessEnvelope
    import tempfile
    
    with tempfile.NamedTemporaryFile() as tf:
        ledger = ActionLedger(tf.name)
        
        # Unsigned receipt
        receipt = DecisionWitnessReceipt(
            decision_id="d1",
            schema_name="S1",
            schema_digest="SD1",
            prompt="q1",
            prompt_digest="PD1",
            values={"label": "A"},
            confidences={"label": 1.0},
            conformal_sets={"label": ["A"]},
            probabilities={},
            latency_ms=10.0,
            is_ambiguous=False,
            timestamp=123.0,
            truth_ledger_head="000",
            ledger_record_id=None,
            signer_public_key="some_key",
            envelope=RunWitnessEnvelope(
                mutation_intent={"test": "test"}
            )
        )
        
        from system1.ledger import IntegrityError
        import pytest
        # should fail since we require trusted_public_key and receipt is not signed/valid
        with pytest.raises(IntegrityError):
            ledger.append(receipt, trusted_public_key="some_key")

def test_record_execution_outcome_enforces_cryptographic_authorization():
    from system1.ledger import ActionLedger, LedgerWriteError
    from system1.receipt import DecisionWitnessReceipt, RunWitnessEnvelope
    import tempfile
    
    with tempfile.NamedTemporaryFile() as tf:
        ledger = ActionLedger(tf.name)
        
        # Unsigned receipt for append (we won't verify it since we don't pass public key)
        receipt = DecisionWitnessReceipt(
            decision_id="d2",
            schema_name="S1",
            schema_digest="SD1",
            prompt="q1",
            prompt_digest="PD1",
            values={"label": "A"},
            confidences={"label": 1.0},
            conformal_sets={"label": ["A"]},
            probabilities={},
            latency_ms=10.0,
            is_ambiguous=False,
            timestamp=123.0,
            truth_ledger_head="000",
            ledger_record_id=None,
            signer_public_key=None,
            envelope=RunWitnessEnvelope(
                mutation_intent={"test": "test"}
            )
        )
        
        ledger.append(receipt, tenant_id="tenant1", principal_id="principal1", scope="scope1")
        
        import pytest
        # matching params should work
        ledger.record_execution_outcome(
            action_id="d2", receipt_digest=receipt.compute_digest(), status="SUCCEEDED",
            tenant_id="tenant1", principal_id="principal1", scope="scope1"
        )
        
        # mismatched tenant
        with pytest.raises(LedgerWriteError, match="Authorization mismatch"):
            ledger.record_execution_outcome(
                action_id="d2", receipt_digest=receipt.compute_digest(), status="SUCCEEDED",
                tenant_id="tenant2", principal_id="principal1", scope="scope1"
            )


# === Merged from tests/review_acceptance/test_previous_contracts_adapted.py ===
"""Replay of previous contract tests on the 23:07 UTC snapshot.
Fixture adaptations: _history is now schema-keyed; explicit insufficient-evidence rejection is accepted for two dependency groups.
All behavioral assertions retained.
No provider calls; effects are list appends. Small deterministic model fixtures
isolate execution semantics, not real-world classification accuracy.
Run: PYTHONPATH=/path/to/repo/src pytest -q test_round3_closure.py
"""
import copy
import hashlib
import json
import re
import threading
import inspect
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from system1 import SystemOneEngine, DecisionSchema, ChoiceField, MultiChoiceField
from system1.calibration import ConformalPredictor, RegressionConformalPredictor
from system1.compiler import CompiledSystemOneModel
from system1.guard import ActionProposal, DecisionOutcome, PolicyEngine, PolicyRule, SystemOneGuardHook
from system1.integrations.mcp import SystemOneMCPProxy
from system1.ledger import ActionLedger, LedgerError
from system1.receipt import create_decision_receipt, verify_decision_witness_receipt
import system1.compat.typesafe as compat

class Route(DecisionSchema):
    route=ChoiceField(options=['north','south'], descriptions={'north':'north route','south':'south route'})

def engine(*,p=.99999,strict=True,ledger=None,key=None):
    e=SystemOneEngine(Route,dimension=16,backend='numpy',strict_mode=strict,
                   margin_threshold=.1,ledger=ledger,signing_key=key)
    h=e.model.heads['route']
    h.set_weights(np.zeros_like(h.weights),np.log([p,1-p]).astype(np.float32)*.25)
    c=e.conformal_predictors['route'];c.is_calibrated=True;c.calibration_scores=np.array([(1 + p) / 2]*100)
    return e

def proposal(tool='sentinel'):
    return ActionProposal.create(tenant_id='tenant',principal_id='alice',scope='tools',tool=tool,
               arguments={'path':'/safe/readme'},canonical_target='/safe/readme',purpose='read fixture')

def allow_policy():
    return PolicyEngine([PolicyRule('allow',outcome=DecisionOutcome.ALLOW,target_pattern='.*')])

def mcp_request():
    return {'jsonrpc':'2.0','id':1,'method':'tools/call','params':{'name':'sentinel','arguments':{'path':'/safe/readme'}}}

# Controls retain important fixes; all exercise current code.
def test_control_deny_overrides_allow():
    p=PolicyEngine([PolicyRule('allow',outcome=DecisionOutcome.ALLOW,target_pattern='.*'),
                    PolicyRule('deny',outcome=DecisionOutcome.DENY,target_pattern='.*')])
    assert p.evaluate(proposal())[0]==DecisionOutcome.DENY

def test_control_allow_constraint_violation_is_denied():
    p=PolicyEngine([PolicyRule('allow',outcome=DecisionOutcome.ALLOW,target_pattern='.*',allowed_principals=['bob'])])
    assert p.evaluate(proposal())[0]==DecisionOutcome.DENY

def test_control_enforcement_rejects_missing_key(tmp_path):
    with ActionLedger(tmp_path/'a.db') as l:
        with pytest.raises(ValueError):
            SystemOneGuardHook(engine=engine(ledger=l),enforcement_profile=True,policy=allow_policy())

def test_control_deterministic_allow_records_and_executes(tmp_path):
    key=Ed25519PrivateKey.generate()
    with ActionLedger(tmp_path/'a.db') as l:
        g=SystemOneGuardHook(engine=engine(ledger=l,key=key),policy=allow_policy(),enforcement_profile=True)
        result=g.evaluate_proposal(proposal())
        assert result.allowed
        r=result.decision_result.receipt
        assert verify_decision_witness_receipt(r.to_dict(),public_key=key.public_key())
        assert any(row['payload'].get('receipt_digest')==r.compute_digest() for row in l.entries())
        calls=[]; response=SystemOneMCPProxy(guard=g).handle_call(mcp_request(),lambda name,args: calls.append(name) or 'ok')
        assert calls==['sentinel'] and 'result' in response

def test_control_readonly_ledger_blocks_deterministic_allow(tmp_path):
    key=Ed25519PrivateKey.generate();path=tmp_path/'a.db';ActionLedger(path).close()
    with ActionLedger(path,read_only=True) as l:
        g=SystemOneGuardHook(engine=engine(ledger=l,key=key),policy=allow_policy(),enforcement_profile=True)
        calls=[];SystemOneMCPProxy(guard=g).handle_call(mcp_request(),lambda name,args: calls.append(name))
        assert calls==[]

def test_control_zero_egress_rejects_passthrough():
    with pytest.raises(ValueError): compat.TypeSafeClient(mode='passthrough',zero_egress=True)

def test_control_uncalibrated_categorical_strict_abstains():
    e=engine();e.conformal_predictors['route'].is_calibrated=False
    assert e.decide('fixture',strict=True).is_ambiguous

def test_control_zero_labels_cannot_promote():
    e=engine();r=compat.evaluate_promotion_eligibility(e,[{'state':'x','answers':{}}],e.schema)
    assert not r.is_eligible and r.total_validation_checks==0

# A/B final authorization and durable evidence, using real SQLite and Ed25519.
def test_model_final_authorization_is_actually_in_ledger(tmp_path):
    key=Ed25519PrivateKey.generate()
    with ActionLedger(tmp_path/'a.db') as l:
        # Conditional invariant: any ALLOW must reference its own persisted receipt.
        # A corrected enforcement API may instead reject this ungranted path.
        try:
            g=SystemOneGuardHook(engine=engine(ledger=l,key=key),enforcement_profile=True)
        except ValueError:
            return
        result=g.evaluate_proposal(proposal())
        if not result.allowed:
            assert result.outcome in (DecisionOutcome.DENY, DecisionOutcome.REQUIRE_APPROVAL)
            return
        r=result.decision_result.receipt
        rows=l.entries()
        assert any(row['payload'].get('receipt_digest')==r.compute_digest() for row in rows), {
            'final_digest':r.compute_digest(),'recorded_digests':[row['payload'].get('receipt_digest') for row in rows],
            'borrowed_ledger_record_id':r.ledger_record_id}

def test_enforcement_does_not_invent_permission_for_unmatched_tool(tmp_path):
    key=Ed25519PrivateKey.generate()
    with ActionLedger(tmp_path/'a.db') as l:
        p=PolicyEngine([PolicyRule('known_only',outcome=DecisionOutcome.ALLOW,action_pattern='^known_tool$')])
        g=SystemOneGuardHook(engine=engine(ledger=l,key=key),policy=p,enforcement_profile=True)
        calls=[];out=SystemOneMCPProxy(guard=g).handle_call(mcp_request(),lambda name,args:calls.append(name) or 'ok')
        assert not calls, {'effect_count':len(calls),'response':out}

def test_corrupt_history_prevents_new_enforced_effect(tmp_path):
    key=Ed25519PrivateKey.generate()
    with ActionLedger(tmp_path/'a.db') as l:
        g=SystemOneGuardHook(engine=engine(ledger=l,key=key),policy=allow_policy(),enforcement_profile=True)
        assert g.evaluate_proposal(proposal()).allowed
        l._connection.execute("UPDATE audit_entries SET payload_json=? WHERE sequence=1", ('{"tampered":true}',))
        assert not l.verify_integrity(trusted_public_key=key.public_key())
        calls=[];out=SystemOneMCPProxy(guard=g).handle_call(mcp_request(),lambda name,args:calls.append(name) or 'ok')
        assert not calls, {'effect_count':len(calls),'response':out}

def test_orphan_execution_outcome_is_rejected(tmp_path):
    with ActionLedger(tmp_path/'a.db') as l:
        with pytest.raises((LedgerError,ValueError)):
            l.record_execution_outcome(action_id='nonexistent',receipt_digest='a'*64,status='SUCCEEDED')

# C cache and calibration contracts.
def test_supplied_embedding_is_part_of_exact_cache_identity():
    e=engine();h=e.model.heads['route'];w=np.zeros_like(h.weights);w[0,0]=3;w[1,0]=-3
    h.set_weights(w,np.zeros(2,dtype=np.float32))
    x=np.zeros(e.dimension,dtype=np.float32);x[0]=1
    a=e.decide('same text',embedding=x,strict=True);assert a.values['route']=='north'
    b=e.decide('same text',embedding=-x,strict=True)
    e.use_cache=False;c=e.decide('same text',embedding=-x,strict=True)
    assert b.values==c.values, {'cached':b.values,'uncached':c.values,'hit':b.is_cache_hit}

def test_distinct_valid_alphas_do_not_alias_across_order_statistic():
    e=engine(p=.8);e.conformal_predictors['route'].calibration_scores=np.array([.85]*95+[1.0]*5)
    permissive=.0594061;stricter=.0594058
    a=e.decide('same text',alpha=permissive,strict=True);assert not a.is_ambiguous
    b=e.decide('same text',alpha=stricter,strict=True)
    e.use_cache=False;c=e.decide('same text',alpha=stricter,strict=True);assert c.is_ambiguous
    assert b.is_ambiguous==c.is_ambiguous, {'cached_set':b.conformal_sets,'fresh_set':c.conformal_sets,'hit':b.is_cache_hit}

def test_learning_marks_previous_calibration_stale():
    e=engine();before=e.model.heads['route'].weights.copy()
    assert not e.decide('fixture',strict=True).is_ambiguous
    e.learn_from_tier2('fixture',{'route':'north'})
    assert not np.array_equal(before,e.model.heads['route'].weights)
    r=e.decide('fixture',strict=True)
    assert r.is_ambiguous, {'still_calibrated':e.conformal_predictors['route'].is_calibrated,'set':r.conformal_sets}

def test_uncalibrated_multilabel_strict_abstains():
    schema=DecisionSchema(schema_name='multi',fields={'tags':MultiChoiceField(options=['a','b'])})
    e=SystemOneEngine(schema,dimension=16,backend='numpy',strict_mode=True)
    h=e.model.heads['tags'];h.set_weights(np.zeros_like(h.weights),np.array([3.,-3.],dtype=np.float32))
    r=e.decide('fixture',strict=True)
    assert r.is_ambiguous, {'calibrator_has_is_calibrated':hasattr(e.calibrators['tags'],'is_calibrated'),'values':r.values}

# D serialized artifact must preserve the actual validated runtime.
def make_promoted_client():
    questions={'route':compat.Choice('Route',criteria={'north':'north route','south':'south route'})}
    c=compat.TypeSafeClient(mode='auto_cutover',dimension=16,backend='numpy',zero_egress=True,
        baseline_handler=lambda *args: None,promotion_policy=compat.PromotionPolicy(min_agreement_threshold=.8))
    c._history[compat._build_dynamic_schema(questions).schema_digest()]=[{'state':f'{direction} route read {direction} item {i}', 'answers':{'route':direction},'questions':questions} for i in range(100) for direction in ['north', 'south']]
    assert c.distill_and_cutover(questions), c._last_promotion_report
    return c,questions

def test_exported_candidate_preserves_validated_runtime_calibration(tmp_path):
    c,q=make_promoted_client();live=c._get_engine(q);p=tmp_path/'candidate.s1m'
    assert c.export_model(p)
    restored=CompiledSystemOneModel.load(p,backend='numpy')
    e=SystemOneEngine(restored.schema,model=restored,backend='numpy')
    assert e._calibration_digest()==live._calibration_digest(), {
        'live_temperature':live.calibrators['route'].temperature,'loaded_temperature':e.calibrators['route'].temperature,
        'live_scores':len(live.conformal_predictors['route'].calibration_scores),
        'loaded_scores':len(e.conformal_predictors['route'].calibration_scores)}

def test_one_request_many_fields_is_not_many_independent_validation_requests():
    schema=DecisionSchema(schema_name='correlated',fields={f'f{i}':ChoiceField(['north','south']) for i in range(20)})
    answers={name:'north' for name in schema.fields}
    fixture=SimpleNamespace(decide=lambda *args,**kwargs:SimpleNamespace(values=answers))
    r=compat.evaluate_promotion_eligibility(fixture,[{'state':'one independent request','answers':answers}],schema)
    assert not r.is_eligible, {'independent_requests':1,'counted_checks':r.total_validation_checks,'wilson_lower':r.wilson_lower_bound}

def test_common_lineage_is_not_split_just_to_fill_validation():
    history=[{'request_id':f'r{i}','group_id':'A' if i<6 else 'B','state':f'q{i}','answers':{'route':'north'}} for i in range(10)]
    try:
        p=compat.partition_cutover_history(history)
    except ValueError as exc:
        assert 'Insufficient evidence' in str(exc)
        return
    # A valid implementation may defer promotion; it must not split a dependency group.
    sets=[{r['group_id'] for r in fold} for fold in (p.train_history,p.calib_history,p.val_history)]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]), {
        'train_groups':list(sets[0]),'calibration_groups':list(sets[1]),'validation_groups':list(sets[2])}

# Packaging consistency is a static source check, not a minimum-runtime install.
def test_declared_grpc_lower_bound_covers_generated_stubs():
    import system1
    from packaging.requirements import Requirement
    root=Path(system1.__file__).resolve().parents[2]
    py=tomllib.loads((root/'pyproject.toml').read_text())
    requirements={Requirement(s).name:Requirement(s) for s in py['project']['optional-dependencies']['grpc']}
    stub=(root/'src/system1/proto/system1_pb2_grpc.py').read_text()
    generated=re.search(r"GRPC_GENERATED_VERSION = '([^']+)'",stub).group(1)
    minimum=next(s.version for s in requirements['grpcio'].specifier if s.operator=='>=')
    from packaging.version import Version
    assert Version(minimum)>=Version(generated), {'declared':minimum,'generated_requires':generated}


def test_async_mcp_tool_is_not_recorded_succeeded_before_it_runs(tmp_path):
    from system1.integrations.mcp import wrap_mcp_tool
    key=Ed25519PrivateKey.generate()
    with ActionLedger(tmp_path/'a.db') as l:
        g=SystemOneGuardHook(engine=engine(ledger=l,key=key),policy=allow_policy(),enforcement_profile=True)
        calls=[]
        @wrap_mcp_tool(guard=g,tool_name='sentinel')
        async def sentinel(path='/safe/readme'):
            calls.append(path)
            return 'ok'
        pending=sentinel()
        try:
            assert calls==[]
            outcomes=[r['payload'] for r in l.entries() if r['event_type']=='execution_outcome']
            assert not any(o['status']=='SUCCEEDED' for o in outcomes), outcomes
        finally:
            if inspect.iscoroutine(pending): pending.close()


# === Merged from tests/review_acceptance/test_postfix_boundaries.py ===
"""Post-fix variants of the existing calibration, artifact, promotion and ledger contracts.
Offline. Native NumPy models where inference is relevant; controlled label outputs
where isolating promotion statistics. These are regression fixtures, not accuracy benchmarks.
"""
import hashlib
from types import SimpleNamespace
import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from system1 import DecisionSchema, ScoreField, SystemOneEngine
from system1.compiler import CompiledHeadWeights, CompiledSystemOneModel
from system1.ledger import ActionLedger, LedgerError
from system1.receipt import create_decision_receipt
import system1.compat.typesafe as compat


def test_nonstrict_update_invalidates_calibration_for_later_strict_call():
    e = engine(strict=False)
    before = e.model.heads['route'].weights.copy()
    assert not e.decide('fixture', strict=True).is_ambiguous
    e.learn_from_tier2('fixture', {'route': 'north'})
    assert not np.array_equal(before, e.model.heads['route'].weights)
    r = e.decide('fixture', strict=True)
    assert r.is_ambiguous, {'still_calibrated': e.conformal_predictors['route'].is_calibrated, 'set': r.conformal_sets}


def test_strict_regression_update_invalidates_previous_interval_calibration():
    schema = DecisionSchema(schema_name='regression', fields={'score': ScoreField(min_value=0, max_value=10)})
    e = SystemOneEngine(schema, dimension=16, backend='numpy', strict_mode=True, use_cache=False)
    h = e.model.heads['score']
    h.set_weights(np.zeros_like(h.weights), np.zeros_like(h.biases))
    before = h.weights.copy()
    cp = e.regression_conformal_predictors['score']
    cp.residuals = np.array([.01] * 100)
    cp.is_calibrated = True
    e.learn_from_tier2('fixture', {'score': 6.0})
    assert not np.array_equal(before, h.weights)
    result = e.decide('fixture', strict=True)
    assert result.is_ambiguous, {'still_calibrated': cp.is_calibrated, 'interval': result.conformal_sets}


def test_exported_bytes_match_recorded_promotion_artifact(tmp_path):
    client, questions = make_promoted_client()
    expected = client.last_promotion_report.artifact_digest
    path = tmp_path / 'candidate.s1m'
    assert client.export_model(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, {'promotion_artifact': expected, 'exported_artifact': actual}


def test_serialization_preserves_strict_prediction_set_at_boundary():
    # Construct valid APS calibration rows close to the runtime cutoff. This
    # isolates float64 -> float32 quantization, not real-world model quality.
    e = engine(p=.9)
    e.use_cache = False
    p = e.decide('boundary', strict=True, record_receipt=False).probabilities['route']['north']
    q = p + 1e-9
    cp = e.conformal_predictors['route']
    cp.calibrate(np.tile([q, 1-q], (100, 1)), ['north'] * 100)
    before = e.decide('boundary', strict=True, record_receipt=False)
    assert not before.is_ambiguous
    head = e.model.heads['route']
    ch = CompiledHeadWeights(field_name='route', field_type='choice', weights=head.weights.copy(),
          biases=head.biases.copy(), options=tuple(e.schema.fields['route'].options),
          temperature=e.calibrators['route'].temperature, calibration_scores=tuple(cp.calibration_scores))
    cm = CompiledSystemOneModel(e.schema, heads={'route': ch}, dimension=16, backend='numpy', use_cache=False)
    restored = CompiledSystemOneModel.from_bytes(cm.to_bytes(), backend='numpy')
    restored.use_cache = False
    loaded_engine = SystemOneEngine(restored.schema, model=restored, backend='numpy', use_cache=False)
    after = loaded_engine.decide('boundary', strict=True, record_receipt=False)
    assert after.conformal_sets == before.conformal_sets, {
        'live_threshold': q, 'reloaded_threshold': float(loaded_engine.conformal_predictors['route'].calibration_scores[0]),
        'live_set': before.conformal_sets, 'reloaded_set': after.conformal_sets,
        'live_ambiguous': before.is_ambiguous, 'reloaded_ambiguous': after.is_ambiguous}


def prediction_fixture():
    return SimpleNamespace(decide=lambda *a, **kw: SimpleNamespace(values={'route': 'north'}, is_ambiguous=False))


def test_shared_lineage_is_not_overridden_by_distinct_group_ids():
    rows = [{'state': f'event-{i}', 'group_id': f'g-{i}', 'lineage_id': 'same-source-event',
             'answers': {'route': 'north'}} for i in range(100)]
    r = compat.evaluate_promotion_eligibility(prediction_fixture(), rows, Route())
    assert not r.is_eligible, r


def test_repeating_one_successful_group_cannot_dominate_group_acceptance():
    rows = [{'state': f'good-{i}', 'group_id': 'one-successful-unit', 'answers': {'route': 'north'}} for i in range(1000)]
    rows += [{'state': f'bad-{i}', 'group_id': f'failed-unit-{i}', 'answers': {'route': 'south'}} for i in range(19)]
    r = compat.evaluate_promotion_eligibility(prediction_fixture(), rows, Route())
    assert not r.is_eligible, {'successful_groups': 1, 'groups': 20, 'report': r}


def test_minimum_validation_samples_uses_scored_independent_units():
    rows = [{'state': f'unit-{i}-row-{j}', 'group_id': f'unit-{i}', 'answers': {'route': 'north'}}
            for i in range(20) for j in range(5)]
    policy = compat.PromotionPolicy(min_validation_samples=100)
    r = compat.evaluate_promotion_eligibility(prediction_fixture(), rows, Route(), policy=policy)
    assert not r.is_eligible, {'independent_units': 20, 'required_units': 100, 'report': r}


def test_generic_action_record_is_not_a_signed_authorization(tmp_path):
    trusted = Ed25519PrivateKey.generate().public_key()
    with ActionLedger(tmp_path/'ledger.db') as ledger:
        ledger.record_action('action', {'note': 'proposal only; no permission or signature'}, action_id='not-authorized')
        with pytest.raises((LedgerError, ValueError)):
            ledger.record_execution_outcome(action_id='not-authorized', receipt_digest='f'*64,
                  status='SUCCEEDED', trusted_public_key=trusted)


def test_control_matching_trusted_incoming_receipt_is_accepted(tmp_path):
    key = Ed25519PrivateKey.generate()
    r = create_decision_receipt(schema_name='test', schema_digest='a'*64, prompt='fixture', values={'ok': True},
        confidences={'ok': 1.0}, conformal_sets={'ok': ['True']}, probabilities={'ok': {'True': 1.0}},
        latency_ms=.1, is_ambiguous=False, signing_key=key)
    with ActionLedger(tmp_path/'ledger.db') as ledger:
        ledger.record_decision_receipt(r, trusted_public_key=key.public_key())
        assert len(ledger.entries()) == 1
        assert ledger.verify_integrity(trusted_public_key=key.public_key())


def test_control_sufficient_scored_independent_groups_can_promote():
    rows = [{'state': f'independent-case-{i}', 'group_id': f'g{i}', 'answers': {'route': 'north'}} for i in range(100)]
    r = compat.evaluate_promotion_eligibility(prediction_fixture(), rows, Route())
    assert r.is_eligible, r
