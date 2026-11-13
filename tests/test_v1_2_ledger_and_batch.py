"""Tests for System 1 v1.2 ActionLedger WAL durability modes and decide_batch parity."""

from __future__ import annotations

import concurrent.futures
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from system1 import (
    ActionLedger,
    ChoiceField,
    DecisionSchema,
    DeterministicSemanticProjector,
    SystemOneCompiler,
    SystemOneEngine,
    TfidfProjector,
)
from system1.core.embeddings import HybridProjector
from system1.ledger import LedgerWriteError


def _sample_receipt(idx: int = 0) -> dict:
    return {
        "decision_id": f"dec-{idx}",
        "action_id": f"act-{idx}",
        "receipt_digest": f"sha256:{idx:064x}",
        "action": "read_status",
        "tool": "echo_tool",
        "arguments": {"idx": idx},
        "outcome": "ALLOW",
        "alpha": 0.05,
        "confidence": 0.99,
    }


def test_action_ledger_durability_modes_and_flush(tmp_path: Path) -> None:
    expected = ("process_wal", "strict_sync", "group_commit", "high_throughput_normal")
    assert ActionLedger.VALID_DURABILITY_MODES == expected

    with pytest.raises(ValueError, match="durability_mode"):
        ActionLedger(tmp_path / "bad.db", durability_mode="invalid_mode")

    for mode in expected:
        db_file = tmp_path / f"ledger_{mode}.db"
        ledger = ActionLedger(db_file, durability_mode=mode)
        try:
            rid = ledger.append(_sample_receipt(1), scope="mcp:tools:exec")
            assert rid
            ledger.flush()
            info = ledger.durability_info()
            assert info["durability_mode"] == mode
            assert info["is_durable"] is True
            if mode == "high_throughput_normal":
                assert info["synchronous"] == 1
            else:
                assert info["synchronous"] >= 2
            assert ledger.verify_integrity() is True
            assert len(ledger.records()) == 1
        finally:
            ledger.close()


def test_action_ledger_enforce_durability_guards(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ActionLedger(":memory:", enforce_durability=True)

    with pytest.raises(ValueError):
        ActionLedger(":memory:", read_only=True)

    with pytest.raises(ValueError, match="high_throughput_normal"):
        ActionLedger(
            tmp_path / "relaxed.db",
            durability_mode="high_throughput_normal",
            enforce_durability=True,
        )

    ro_db = tmp_path / "ro.db"
    seed = ActionLedger(ro_db, durability_mode="process_wal")
    seed.append(_sample_receipt(1))
    seed.close()

    ro_ledger = ActionLedger(ro_db, read_only=True, enforce_durability=True)
    try:
        assert ro_ledger.verify_integrity() is True
    finally:
        ro_ledger.close()

    for ok_mode in ("process_wal", "strict_sync", "group_commit"):
        ledger = ActionLedger(
            tmp_path / f"strict_{ok_mode}.db",
            durability_mode=ok_mode,
            enforce_durability=True,
        )
        try:
            ledger.append(_sample_receipt(1))
            ledger.flush()
            assert ledger.verify_integrity() is True
        finally:
            ledger.close()


def test_action_ledger_group_commit_concurrent_appends(tmp_path: Path) -> None:
    db_path = tmp_path / "concurrent_group_commit.db"
    ledger = ActionLedger(
        db_path,
        durability_mode="group_commit",
        group_commit_ms=8.0,
        group_commit_max_batch=8,
    )
    try:

        def _worker(tid: int) -> list[str]:
            out = []
            for j in range(10):
                out.append(ledger.append(_sample_receipt(tid * 100 + j), scope="mcp:tools:call"))
            return out

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futs = [pool.submit(_worker, tid) for tid in range(4)]
            ids = [rid for fut in futs for rid in fut.result()]

        ledger.flush()
        assert len(ids) == 40
        assert len(ledger.records()) == 40
        assert ledger.verify_integrity() is True
    finally:
        ledger.close()

    reopened = ActionLedger(db_path, read_only=True)
    try:
        assert len(reopened.records()) == 40
        assert reopened.verify_integrity() is True
    finally:
        reopened.close()


def test_action_ledger_coalesced_read_only_execution(tmp_path: Path) -> None:
    db_path = tmp_path / "coalesced.db"
    ledger = ActionLedger(db_path, durability_mode="process_wal")
    try:
        for bad_rev in ("IRREVERSIBLE", "MUTATING", "DESTRUCTIVE"):
            with pytest.raises(LedgerWriteError, match="READ_ONLY"):
                ledger.record_coalesced_read_only_execution(
                    _sample_receipt(99),
                    reversibility=bad_rev,
                    action_id="bad-1",
                )

        rec_id, out_id = ledger.record_coalesced_read_only_execution(
            _sample_receipt(7),
            reversibility="READ_ONLY",
            action_id="ro-7",
            status="SUCCEEDED",
            result_payload={"rows": 3},
        )
        assert rec_id and out_id
        records = ledger.records()
        assert len(records) == 2
        assert ledger.verify_integrity() is True
    finally:
        ledger.close()


def test_action_ledger_watermark_detects_external_sqlite_tamper(tmp_path: Path) -> None:
    db_path = tmp_path / "tamper.db"
    ledger = ActionLedger(db_path, durability_mode="process_wal")
    try:
        for i in range(6):
            ledger.append(_sample_receipt(i))
        ledger.flush()
        assert ledger.verify_integrity() is True
        assert ledger.verify_integrity() is True

        ext = sqlite3.connect(str(db_path))
        try:
            row = ext.execute(
                "SELECT sequence, payload_json FROM audit_entries WHERE sequence = 3"
            ).fetchone()
            assert row is not None
            payload = json.loads(row[1])
            payload["tampered"] = True
            ext.execute(
                "UPDATE audit_entries SET payload_json = ? WHERE sequence = 3",
                (json.dumps(payload),),
            )
            ext.commit()
        finally:
            ext.close()

        assert ledger.verify_integrity() is False
    finally:
        ledger.close()


def _parity_schema() -> DecisionSchema:
    return DecisionSchema(
        schema_name="BatchParitySchema",
        fields={
            "intent": ChoiceField(options=["billing", "refund", "technical", "account"]),
            "urgency": ChoiceField(options=["low", "high"]),
        },
    )


def _parity_examples() -> list[tuple[str, dict[str, str]]]:
    data = [
        ("refund duplicate charge invoice", {"intent": "refund", "urgency": "high"}),
        ("please refund damaged item order", {"intent": "refund", "urgency": "low"}),
        ("chargeback refund needed immediately", {"intent": "refund", "urgency": "high"}),
        ("monthly billing statement question", {"intent": "billing", "urgency": "low"}),
        ("update billing card payment method", {"intent": "billing", "urgency": "low"}),
        ("urgent invoice billing overcharge", {"intent": "billing", "urgency": "high"}),
        ("production api 502 gateway timeout", {"intent": "technical", "urgency": "high"}),
        ("database connection pool error", {"intent": "technical", "urgency": "high"}),
        ("documentation typo minor technical bug", {"intent": "technical", "urgency": "low"}),
        ("reset password two factor account", {"intent": "account", "urgency": "low"}),
        ("locked out of admin account urgent", {"intent": "account", "urgency": "high"}),
        ("change email address on account", {"intent": "account", "urgency": "low"}),
    ]
    expanded = []
    for rep in range(4):
        for prompt, vals in data:
            expanded.append((f"{prompt} ref{rep}", dict(vals)))
    return expanded


@pytest.mark.parametrize("proj_kind", ["deterministic", "tfidf", "hybrid_fit", "hybrid_v1_2", "compiled"])
def test_decide_vs_decide_batch_exact_parity(proj_kind: str) -> None:
    schema = _parity_schema()
    examples = _parity_examples()
    texts = [p for p, _ in examples]
    labels = [v["intent"] for _, v in examples]

    probes = [
        "refund duplicate charge invoice",
        "urgent invoice billing overcharge",
        "production api 502 gateway timeout",
        "locked out of admin account urgent",
        "not refund actually billing question",
        "monthly billing statement",
        "totally unrelated quantum astrophysics nebula zxqvw",
        "another weird out of distribution banana spaceship",
        "",
        "café naïve résumé refund",
        "db timeout urgent",
        "password reset",
    ] * 2

    if proj_kind == "compiled":
        proj = HybridProjector.fit_v1_2(texts, labels=labels, schema=schema, sparse_dim=512, dense_dim=96)
        ex_map = {
            "intent": [(p, v["intent"]) for p, v in examples],
            "urgency": [(p, v["urgency"]) for p, v in examples],
        }
        compiled = SystemOneCompiler(
            schema,
            projector=proj,
            dimension=proj.dimension,
            familiarity_alpha=0.05,
            prototype_alpha=0.05,
            semantic_mode="v1_2_rbf",
        ).compile(exemplars=ex_map, augment=False)
        engine = SystemOneEngine(schema, model=compiled, use_cache=False)
    else:
        if proj_kind == "deterministic":
            proj = DeterministicSemanticProjector(dimension=128)
        elif proj_kind == "tfidf":
            proj = TfidfProjector.fit(texts, max_features=256)
        elif proj_kind == "hybrid_fit":
            proj = HybridProjector.fit(texts, labels=labels, sparse_dim=512, dense_dim=96)
        else:
            proj = HybridProjector.fit_v1_2(
                texts, labels=labels, schema=schema, sparse_dim=512, dense_dim=96, attach_rbf=True
            )
        engine = SystemOneEngine(schema, projector=proj, dimension=proj.dimension, use_cache=False)
        for p, v in examples:
            engine.learn(p, v)
        engine.calibrate(examples[:16])

    singles = [engine.decide(p, use_cache=False) for p in probes]
    batched = engine.decide_batch(probes, use_cache=False)
    assert len(singles) == len(batched) == len(probes)

    for s, b in zip(singles, batched):
        assert s.values == b.values
        assert s.is_ambiguous == b.is_ambiguous
        assert s.conformal_sets == b.conformal_sets
        assert s.metadata.get("ood_flagged") == b.metadata.get("ood_flagged")
        assert s.abstention_reason == b.abstention_reason
        assert set(s.confidences.keys()) == set(b.confidences.keys())
        for k in s.confidences:
            assert np.isclose(s.confidences[k], b.confidences[k], atol=1e-5)
            assert np.isclose(s.margins[k], b.margins[k], atol=1e-5)
            assert set(s.probabilities[k].keys()) == set(b.probabilities[k].keys())
            for opt in s.probabilities[k]:
                assert np.isclose(s.probabilities[k][opt], b.probabilities[k][opt], atol=1e-5)
        if s.familiarity_score is None or b.familiarity_score is None:
            assert s.familiarity_score is b.familiarity_score
        else:
            assert np.isclose(s.familiarity_score, b.familiarity_score, atol=1e-5)
