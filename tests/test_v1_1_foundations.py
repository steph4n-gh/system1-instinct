"""Tests for System 1: Instinct v1.1 foundations, security hardening, and surfaces."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import socket
import numpy as np
import pytest

from benchmarks.prequential.cassette import (
    CassetteTeacher,
    NetworkBlockedError,
    network_blocked,
)
from system1 import (
    AdaptiveConformalTracker,
    ChoiceField,
    ConformalPredictor,
    DecisionSchema,
    HybridProjector,
    SelectiveRiskCalibrator,
    SystemOneCompiler,
    SystemOneEngine,
    UnifiedHybridProjector,
    clopper_pearson_upper,
)
from system1.cli import main as cli_main
from system1.compiler import CompiledSystemOneModel
from system1.guard import (
    ActionProposal,
    DecisionOutcome,
    PolicyEngine,
    PolicyRule,
    RiskLevel,
    SystemOneGuardHook,
)
from system1.integrations.drop_in import SystemOneInstructorClient
from system1.integrations.fastapi import SystemOneGatewayMiddleware
from system1.ledger import ActionLedger


def _support_schema_and_data():
    schema = DecisionSchema(fields={"queue": ChoiceField(options=("billing", "access", "security"))})
    train = [
        ("refund duplicate charge invoice billing receipt", "billing"),
        ("credit card declined renewal payment invoice", "billing"),
        ("update VAT tax number on invoice billing", "billing"),
        ("downgrade subscription billing seat invoice", "billing"),
        ("reset MFA authenticator token login access", "access"),
        ("SSO SAML login redirect loop workspace access", "access"),
        ("unlock suspended employee login account access", "access"),
        ("add contractor role permissions access", "access"),
        ("revoke leaked API token Bearer secret security", "security"),
        ("suspicious unauthorized login alert security log", "security"),
        ("report phishing link security incident", "security"),
        ("rotate compromised signing key security alert", "security"),
    ]
    calib = [
        ("issue credit memo for duplicate billing invoice", "billing"),
        ("annual subscription renewal payment failed", "billing"),
        ("need password reset and 2FA token recovery access", "access"),
        ("configure SAML SSO permissions for team access", "access"),
        ("alert on compromised deploy key and token leak security", "security"),
        ("investigate brute force login burst security", "security"),
    ]
    return schema, train, calib


def test_guard_long_target_suffix_and_single_sqlite_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    policy = PolicyEngine(
        rules=[
            PolicyRule(
                rule_id="deny_secret_file",
                outcome=DecisionOutcome.DENY,
                risk=RiskLevel.IRREVERSIBLE,
                denied_targets=["/etc/shadow"],
            )
        ]
    )
    ledger_path = tmp_path / "ledger.sqlite"
    with ActionLedger(ledger_path, require_durable=True) as ledger:
        hook = SystemOneGuardHook(policy=policy, ledger=ledger)
        writes = 0
        orig_append = ledger.append

        def counted_append(*args, **kwargs):
            nonlocal writes
            writes += 1
            return orig_append(*args, **kwargs)

        monkeypatch.setattr(ledger, "append", counted_append)

        # Suffix beyond 4096 chars in arguments must still be caught.
        padded_arg = ("prefix_" * 700) + "/etc/shadow"
        assert len(padded_arg) > 4096
        proposal_deny = ActionProposal.create(
            tenant_id="t1",
            principal_id="u1",
            scope="fs",
            tool="read_file",
            arguments={"path": padded_arg},
            purpose="Read documentation file",
            canonical_target="doc.md",
        )
        blocked = hook.evaluate_proposal(proposal_deny)
        assert blocked.allowed is False
        assert blocked.outcome == DecisionOutcome.DENY

        writes = 0
        proposal_ok = ActionProposal.create(
            tenant_id="t1",
            principal_id="u1",
            scope="fs",
            tool="read_file",
            arguments={"path": "README.md"},
            purpose="Inspect local read-only documentation README.md",
            canonical_target="README.md",
        )
        ok = hook.evaluate_proposal(proposal_ok)
        assert isinstance(ok.allowed, bool)
        assert writes == 1


def test_mondrian_rare_class_shrinkage_floor() -> None:
    cp = ConformalPredictor("label", ("common", "rare"), mondrian=True)
    # 25 common rows with nonconformity score 0.20, 1 rare row with optimistic 0.001
    probs = np.array([[0.80, 0.20]] * 25 + [[0.001, 0.999]], dtype=np.float64)
    labels = ["common"] * 25 + ["rare"]
    cp.calibrate(probs, labels, mondrian=True)
    q_global = float(cp.calibration_scores[int(np.ceil((len(labels) + 1) * 0.95)) - 1])
    assert cp.class_quantile("rare", alpha=0.05) >= q_global - 1e-12


def test_online_warmup_corroboration_and_ood_gates() -> None:
    from system1 import TfidfProjector

    schema, train, calib = _support_schema_and_data()
    # 1. Warmup + override corroboration on an online-taught uncalibrated engine
    eng = SystemOneEngine(
        schema,
        strict_mode=True,
        use_cache=False,
        min_calibration_lessons=5,
        min_override_corroborations=2,
    )
    mixed_prompt = "refund invoice billing alert security login"
    # First override is below corroboration threshold 2 -> must stay ambiguous.
    eng.learn(mixed_prompt, {"queue": "billing"})
    assert eng._online_lesson_count == 1
    assert eng._override_counts == {"refund invoice billing alert security login": {"queue": ("billing", 1)}}
    first_dec = eng.decide(mixed_prompt, alpha=0.05)
    assert first_dec.is_ambiguous is True

    # Corroborate second time with same label -> override activates on exact match.
    eng.learn(mixed_prompt, {"queue": "billing"})
    second_dec = eng.decide(mixed_prompt, alpha=0.05)
    assert second_dec.values["queue"] == "billing"
    assert second_dec.is_ambiguous is False

    # 2. Calibrated OOD gate blocks low-familiarity / low-similarity OOD prompt even after override
    proj = TfidfProjector.fit([p for p, _ in train], labels=[y for _, y in train])
    cal_big = calib * 4  # 24 calibration points so alpha=0.10 & OOD quantiles calibrate
    model_cal = SystemOneCompiler(
        schema,
        projector=proj,
        dimension=proj.dimension,
        familiarity_alpha=0.20,
        prototype_alpha=0.20,
    ).compile(
        exemplars={"queue": train},
        calibration_exemplars={"queue": cal_big},
        augment=False,
    )
    eng_cal = SystemOneEngine(schema, model=model_cal, strict_mode=True, use_cache=False)
    ood_prompt = "quantum sourdough recipe purple giraffe"
    eng_cal.learn(ood_prompt, {"queue": "billing"})
    eng_cal.learn(ood_prompt, {"queue": "billing"})
    ood_dec = eng_cal.decide(ood_prompt, alpha=0.10)
    assert ood_dec.is_ambiguous is True


def test_compact_s1m_float16_roundtrip_and_hybrid_projector() -> None:
    assert UnifiedHybridProjector is HybridProjector
    schema, train, calib = _support_schema_and_data()
    texts = [p for p, _ in train]
    labels = [y for _, y in train]

    proj = UnifiedHybridProjector.fit(texts, labels=labels, backend="numpy")
    v = proj.project(texts[0])
    rnorm = proj.raw_norm(texts[0])
    assert v.shape == (proj.dimension,)
    assert rnorm > 0.0

    comp = SystemOneCompiler(schema, projector=proj, dimension=proj.dimension, backend="numpy")
    model = comp.compile(
        exemplars={"queue": train},
        calibration_exemplars={"queue": calib},
        augment=False,
    )
    raw_bytes = model.to_bytes()
    reloaded = CompiledSystemOneModel.from_bytes(raw_bytes, backend="numpy")

    eng_orig = SystemOneEngine(schema, model=model, strict_mode=True, use_cache=False)
    eng_reload = SystemOneEngine(schema, model=reloaded, strict_mode=True, use_cache=False)
    for txt, _ in calib:
        r1 = eng_orig.decide(txt, alpha=0.15)
        r2 = eng_reload.decide(txt, alpha=0.15)
        assert r1.values == r2.values
        for opt, p1 in r1.probabilities["queue"].items():
            assert abs(p1 - r2.probabilities["queue"][opt]) < 1e-4

    # Backward-compatible legacy hash-sparse hybrid config still loads cleanly.
    fitted_small = HybridProjector.fit(texts[:6], sparse_dim=128, dense_dim=64, alpha=0.80)
    assert "tfidf_terms" in fitted_small.to_config()
    legacy_hash_cfg = {"type": "hybrid", "sparse_dim": 128, "dense_dim": 64, "alpha": 0.80, "seed": 42}
    restored = HybridProjector.from_config(legacy_hash_cfg)
    assert restored.dimension == 192
    assert restored.project("invoice billing refund").shape == (192,)


def test_clopper_pearson_selective_risk_and_aci_tracker() -> None:
    ub0 = clopper_pearson_upper(0, 100, delta=0.05)
    assert ub0 == pytest.approx(1.0 - 0.05 ** (1.0 / 100.0), abs=1e-12)
    scipy_stats = pytest.importorskip("scipy.stats")
    ub_scipy = float(scipy_stats.beta.ppf(0.95, 4, 97))
    assert clopper_pearson_upper(3, 100, delta=0.05) == pytest.approx(ub_scipy, abs=1e-10)

    # Unqualified calibrator fails closed.
    cal = SelectiveRiskCalibrator(method="ltt_fst", alpha=0.05, delta=0.05)
    assert cal.is_qualified() is False
    assert cal.should_accept(0.99) is False

    # Synthetic separated scores qualify and accept high-margin points while rejecting low-margin ones.
    rng = np.random.default_rng(7)
    scores = np.linspace(0.05, 0.99, 250)
    correct = (scores >= 0.40) | (rng.random(250) < 0.1)
    for method in ("ltt_fst", "ltt_bonferroni", "rcps_joint"):
        c = SelectiveRiskCalibrator(score_name="margin", method=method, alpha=0.05, delta=0.05)
        cert = c.fit(scores, correct)
        assert cert.qualified is True
        assert cert.threshold is not None
        assert c.should_accept(0.99) is True
        assert c.should_accept(0.01) is False

    # Wire SelectiveRiskCalibrator + AdaptiveConformalTracker into SystemOneEngine.
    schema, train, calib = _support_schema_and_data()
    model = SystemOneCompiler(schema).compile(
        exemplars={"queue": train},
        calibration_exemplars={"queue": calib},
        augment=False,
    )
    blocker = SelectiveRiskCalibrator(score_name="margin", method="ltt_fst", alpha=0.05, delta=0.05)
    # Deliberately unqualified calibrator forces selective_risk_gate abstention in strict mode.
    tracker = AdaptiveConformalTracker(alpha_target=0.10, gamma=0.02, margin_floor=0.20)
    eng = SystemOneEngine(
        schema,
        model=model,
        strict_mode=True,
        use_cache=False,
        selective_calibrators={"queue": blocker},
        aci_tracker=tracker,
    )
    res = eng.decide("refund duplicate billing invoice")
    assert res.is_ambiguous is True
    assert res.abstention_reasons.get("queue") == "selective_risk_gate"

    tracker.update_accepted(error=True, margin=0.35)
    assert tracker.alpha_t < 0.10
    assert tracker.margin_floor == pytest.approx(0.3501, abs=1e-6)


def test_fastapi_gateway_limits_instructor_and_jsonl_errors(tmp_path: Path) -> None:
    schema, train, calib = _support_schema_and_data()
    model = SystemOneCompiler(schema).compile(
        exemplars={"queue": train},
        calibration_exemplars={"queue": calib},
        augment=False,
    )
    eng = SystemOneEngine(schema, model=model, strict_mode=True, use_cache=False)

    # 1. FastAPI middleware: allow_route_header=False by default & max_body_bytes -> 413
    downstream_calls: list[dict] = []

    async def dummy_app(scope, receive, send):
        downstream_calls.append({"path": scope["path"]})
        await send({"type": "http.response.start", "status": 200, "headers": [(b"x-downstream", b"1")]})
        await send({"type": "http.response.body", "body": b'{"ok":true}', "more_body": False})

    mw_default = SystemOneGatewayMiddleware(dummy_app, schema, engine=eng, max_body_bytes=64)

    async def run_asgi(mw, path: str, headers: list, body: bytes):
        sent = []
        scope = {"type": "http", "method": "POST", "path": path, "headers": headers}
        delivered = False

        async def recv():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        async def snd(msg):
            sent.append(msg)

        await mw(scope, recv, snd)
        return sent

    # Spoofed x-system1-route header ignored when allow_route_header=False
    sent_ignored = asyncio.run(
        run_asgi(
            mw_default,
            "/admin/secret",
            [(b"x-system1-route", b"true")],
            b'{"prompt":"refund invoice"}',
        )
    )
    assert sent_ignored[0]["status"] == 200
    assert len(downstream_calls) == 1

    # Oversized body triggers HTTP 413 without reaching downstream
    sent_413 = asyncio.run(
        run_asgi(mw_default, "/v1/decide", [], b"A" * 128)
    )
    assert sent_413[0]["status"] == 413
    assert json.loads(sent_413[1]["body"].decode("utf-8"))["error"] == "payload_too_large"
    assert len(downstream_calls) == 1

    # Authorized header works when allow_route_header=True and token matches
    mw_auth = SystemOneGatewayMiddleware(
        dummy_app,
        schema,
        engine=eng,
        allow_route_header=True,
        route_header_token="secret-tok",
    )
    sent_bad_tok = asyncio.run(
        run_asgi(
            mw_auth,
            "/custom",
            [(b"x-system1-route", b"true"), (b"x-system1-route-token", b"wrong")],
            b'{"prompt":"refund invoice"}',
        )
    )
    assert sent_bad_tok[0]["status"] == 200
    assert len(downstream_calls) == 2

    # 2. Instructor StructuredExtractionResult exposes is_ambiguous and conformal_sets
    client = SystemOneInstructorClient(schema)
    client.engine.strict_mode = True
    ext = client.create("unseen random words that lack calibration")
    assert hasattr(ext, "is_ambiguous")
    assert hasattr(ext, "conformal_sets")
    assert ext.is_ambiguous is True
    assert "queue" in ext.conformal_sets

    # 3. Line-numbered JSONL error in evaluate_dataset & system1 teach
    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text('{"input": "a", "label": "billing"}\n{broken json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        eng.evaluate_dataset(bad_jsonl)

    teach_dir = tmp_path / "teach_session"
    rc = cli_main(["teach", str(teach_dir), "--schema", "triage"])
    assert rc == 0
    bad_qual = tmp_path / "bad_qual.jsonl"
    bad_qual.write_text('["ok text", "billing"]\n[bad json\n', encoding="utf-8")
    rc_bad = cli_main(["teach", str(teach_dir), "--qualification", str(bad_qual)])
    assert rc_bad != 0


def test_offline_cassette_replay_and_network_blocking() -> None:
    cassette = Path(__file__).resolve().parents[1] / "benchmarks" / "prequential" / "fixtures" / "support_stream_v1.cassette.jsonl"
    assert cassette.is_file()
    header_line = json.loads(cassette.read_text(encoding="utf-8").splitlines()[0])
    schema = DecisionSchema(fields={header_line["field_name"]: ChoiceField(options=header_line["options"])})

    teacher = CassetteTeacher(cassette, schema)
    assert len(teacher.index) > 0
    # Ensure eval_audit is stripped from cached teacher records.
    first_bucket = next(iter(teacher.index.values()))
    assert "eval_audit" not in first_bucket[0]

    with network_blocked():
        with pytest.raises(NetworkBlockedError):
            s = socket.socket()
            try:
                s.connect(("127.0.0.1", 80))
            finally:
                s.close()
        ans = teacher("Customer charged twice for monthly plan and needs refund invoice")
        assert ans == {"queue": "billing"}
