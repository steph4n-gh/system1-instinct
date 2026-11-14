"""Tests for Wave 3: ScoreField AutoCutoverLifecycle, Universal Decision Gateway, and Pure-NumPy Decision Utility."""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

import system1
from system1 import (
    AuthError,
    BinaryPlattBetaCalibrator,
    CascadeRouteDecision,
    CumulativeLogitOrdinalHead,
    CutoverConfig,
    CutoverState,
    DecisionSchema,
    ForbiddenOriginError,
    InteropError,
    PayloadTooLargeError,
    SchemaValidationError,
    ScoreField,
    SoftDistillingLifecycle,
    SoftOrdinalHead,
    UniversalDecisionASGIApp,
    UniversalDecisionService,
    UniversalDecisionsClient,
    UniversalDecisionsHTTPServer,
    UnsupportedMediaError,
    UtilityCalibrator,
    UtilityCascadeRouter,
    ZeroEgressViolationError,
    cqr_ordinal_intervals,
    should_use_soft_ordinal,
)
from system1.compat.universal import (
    compile_questions_contract,
    detect_wire_format,
    diff_schemas,
    extract_semantic_text,
)
from system1.cutover import AutoCutoverLifecycle
from system1.integrations import create_universal_asgi_app, mount_universal_gateway


# ---------------------------------------------------------------------------
# 1. ScoreField AutoCutoverLifecycle & score_tolerance canary checks
# ---------------------------------------------------------------------------


class _RiskScoreSchema(DecisionSchema):
    risk = ScoreField(
        min_value=0.0,
        max_value=10.0,
        low_description="safe static read-only billing status lookup documentation query",
        high_description="critical privilege escalation dump secrets drop production database",
    )


def test_score_field_cutover_config_validation() -> None:
    with pytest.raises(ValueError, match="score_tolerance"):
        CutoverConfig(score_tolerance=0.0)
    with pytest.raises(ValueError, match="score_tolerance"):
        CutoverConfig(score_tolerance=1.2)
    cfg = CutoverConfig(score_tolerance=0.2)
    assert cfg.score_tolerance == 0.2


def test_score_field_autocutover_promotion_and_canary_tolerance() -> None:
    cfg = CutoverConfig(
        milestones=(32, 88),
        k_min=2,
        alpha_target=0.15,
        canary_rate=0.20,
        sparse_dim=256,
        dense_dim=32,
        score_tolerance=0.20,
    )
    lc = AutoCutoverLifecycle(_RiskScoreSchema, config=cfg)

    low_samples = [
        f"read status page billing summary FAQ note {i}" for i in range(22)
    ]
    mid_low_samples = [
        f"modify user profile timezone preferences update form {i}" for i in range(22)
    ]
    mid_high_samples = [
        f"export anonymized analytics table bulk download csv {i}" for i in range(22)
    ]
    high_samples = [
        f"drop production database dump admin secret keys now {i}" for i in range(22)
    ]

    for idx in range(22):
        lc.observe_teacher(low_samples[idx], {"risk": 0.6})
        lc.observe_teacher(mid_low_samples[idx], {"risk": 3.2})
        lc.observe_teacher(mid_high_samples[idx], {"risk": 6.8})
        lc.observe_teacher(high_samples[idx], {"risk": 9.4})

    assert lc.state == CutoverState.SELECTIVE_HOT_SWAP.value
    assert lc.active_tau is not None

    local = None
    for probe in (
        "drop production database dump admin secret keys now 0",
        "read status page billing summary FAQ note 0",
        "drop production database dump admin secret keys now 1",
    ):
        local = lc.decide_local(probe)
        if local is not None:
            break

    assert local is not None
    assert 0.0 <= float(local["risk"]) <= 10.0

    # Canary audit within score_tolerance counts as agreement (0 error), outside counts as error (1)
    assert lc._values_match({"risk": 9.0}, {"risk": 9.5})
    assert not lc._values_match({"risk": 1.0}, {"risk": 9.0})


# ---------------------------------------------------------------------------
# 2. Universal Decision Gateway (all 12 wire, security & lifecycle contracts)
# ---------------------------------------------------------------------------


_SUPPORT_QUESTIONS = {
    "intent": {
        "type": "choice",
        "criteria": {
            "billing": "invoice refund payment charge subscription",
            "technical": "error crash stacktrace timeout api bug",
        },
    },
    "urgent": {
        "type": "noul",
        "true_description": "production outage down critical incident",
        "false_description": "routine documentation billing question",
        "threshold": 0.5,
    },
    "severity": {
        "type": "score",
        "min_value": 0.0,
        "max_value": 10.0,
        "low_description": "routine documentation billing question",
        "high_description": "production outage down critical incident",
    },
}


def _mock_universal_teacher(semantic_text: str, raw_payload: Dict[str, Any], contract: Any) -> Dict[str, Any]:
    is_tech = any(tok in semantic_text.lower() for tok in ("crash", "outage", "error", "stacktrace", "500"))
    ans: Dict[str, Any] = {}
    for c in contract.contracts:
        if c.external_name == "intent":
            choice = "technical" if is_tech else "billing"
            ans["intent"] = {
                "choice": choice,
                "probabilities": (
                    {"billing": 0.04, "technical": 0.96}
                    if is_tech
                    else {"billing": 0.96, "technical": 0.04}
                ),
            }
        elif c.external_name == "urgent":
            p_true = 0.95 if is_tech else 0.05
            ans["urgent"] = {"noul": p_true, "value": is_tech}
        elif c.external_name == "severity":
            ans["severity"] = {"score": 9.2 if is_tech else 1.2}
        elif c.q_type == "choice":
            ans[c.external_name] = {"choice": c.options[0]}
        elif c.q_type == "noul":
            ans[c.external_name] = {"noul": 0.9, "value": True}
        else:
            ans[c.external_name] = {"score": c.min_value}
    return {"answers": ans, "usage": {"input_tokens": len(semantic_text.split())}}


def test_universal_gateway_wire_formats_gemini_laya_jev() -> None:
    svc = UniversalDecisionService(
        upstream_handlers={
            "gemini": _mock_universal_teacher,
            "laya": _mock_universal_teacher,
            "jev": _mock_universal_teacher,
        },
        bearer_token="secret-123",
    )
    client = UniversalDecisionsClient(service=svc, api_key="secret-123", origin="http://localhost")

    # (1) Gemini Decision 1
    g_resp = client.decisions.create(
        model="gemini-decision-1",
        input="Production API returning 500 stacktrace crash",
        questions=_SUPPORT_QUESTIONS,
    )
    assert g_resp.provider_format == "gemini"
    assert g_resp.answers.intent.choice == "technical"
    assert g_resp.decisions.urgent.value is True
    assert g_resp.answers.severity.score == pytest.approx(9.2)

    # (2) Convai Laya
    l_resp = client.systemone(
        model="laya-v1",
        state="Customer asking for a monthly invoice copy",
        questions=_SUPPORT_QUESTIONS,
    )
    assert l_resp.provider_format == "laya"
    assert l_resp.answers.intent.choice == "billing"
    assert l_resp.answers.urgent.value is False

    # (3) TypeSafe Jev (/v1/systemone and /v1/decisions)
    j_resp = client.systemone(
        model="jev-1.13.0",
        state="Duplicate card charge refund request",
        questions=_SUPPORT_QUESTIONS,
    )
    assert j_resp.provider_format == "jev"
    assert j_resp.answers.intent.choice == "billing"
    assert detect_wire_format("/v1/decisions", {"state": "hello", "questions": _SUPPORT_QUESTIONS}) == "jev"


def test_universal_gateway_multimodal_and_unsupported_media() -> None:
    text, _ = extract_semantic_text(
        {
            "contents": [
                {"role": "user", "parts": [{"text": "Server outage"}, {"text": "database timeout"}]}
            ]
        }
    )
    assert "Server outage" in text and "database timeout" in text

    text2, _ = extract_semantic_text(
        {"input": [{"type": "input_text", "text": "Refund my billing charge"}]}
    )
    assert "Refund my billing charge" in text2

    contract = compile_questions_contract(_SUPPORT_QUESTIONS)
    assert "intent" in contract.ext_to_int
    assert diff_schemas(contract, contract)["drifted"] is False
    with pytest.raises(SchemaValidationError):
        compile_questions_contract({})

    svc = UniversalDecisionService(
        upstream_handlers={"gemini": _mock_universal_teacher},
        bearer_token="secret-123",
    )
    # Raw binary / non-JSON Content-Type -> 415 UnsupportedMediaError
    with pytest.raises(UnsupportedMediaError) as exc_info:
        svc.handle_request(
            path="/v1beta/decisions",
            headers={"Authorization": "Bearer secret-123", "Content-Type": "application/octet-stream"},
            raw_body=b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",
        )
    assert exc_info.value.status_code == 415


def test_universal_gateway_strict_json_and_security_headers() -> None:
    svc = UniversalDecisionService(
        upstream_handlers={"gemini": _mock_universal_teacher},
        bearer_token="tok-xyz",
        allowed_origins=("http://localhost", "https://trusted.local"),
        max_body_bytes=1024,
    )

    # Strict JSON: NaN rejected (400)
    with pytest.raises(InteropError) as exc_nan:
        svc.handle_request(
            path="/v1beta/decisions",
            headers={"Authorization": "Bearer tok-xyz", "Content-Type": "application/json"},
            raw_body=b'{"input": "x", "questions": {"q": {"type": "noul"}}, "val": NaN}',
        )
    assert exc_nan.value.status_code == 400

    # Strict JSON: duplicate keys rejected (400)
    with pytest.raises(InteropError) as exc_dup:
        svc.handle_request(
            path="/v1beta/decisions",
            headers={"Authorization": "Bearer tok-xyz", "Content-Type": "application/json"},
            raw_body=b'{"input": "a", "input": "b", "questions": {"q": {"type": "noul"}}}',
        )
    assert exc_dup.value.status_code == 400

    # Body size limit -> 413
    with pytest.raises(PayloadTooLargeError) as exc_big:
        svc.handle_request(
            path="/v1beta/decisions",
            headers={"Authorization": "Bearer tok-xyz", "Content-Type": "application/json"},
            raw_body=b"{" + (b" " * 1100) + b"}",
        )
    assert exc_big.value.status_code == 413

    # Missing / wrong token -> 401
    with pytest.raises(AuthError) as exc_auth:
        svc.handle_request(
            path="/v1beta/decisions",
            headers={"Content-Type": "application/json"},
            raw_body=json.dumps({"input": "hello", "questions": _SUPPORT_QUESTIONS}).encode(),
        )
    assert exc_auth.value.status_code == 401

    # Disallowed Origin -> 403
    with pytest.raises(ForbiddenOriginError) as exc_orig:
        svc.handle_request(
            path="/v1beta/decisions",
            headers={
                "x-goog-api-key": "tok-xyz",
                "Origin": "https://evil.example",
                "Content-Type": "application/json",
            },
            raw_body=json.dumps({"input": "hello", "questions": _SUPPORT_QUESTIONS}).encode(),
        )
    assert exc_orig.value.status_code == 403

    # x-goog-api-key + trusted Origin succeeds
    status, body = svc.handle_request(
        path="/v1beta/decisions",
        headers={
            "x-goog-api-key": "tok-xyz",
            "Origin": "https://trusted.local:8443",
            "Content-Type": "application/json",
        },
        raw_body=json.dumps({"input": "invoice refund please", "questions": _SUPPORT_QUESTIONS}).encode(),
    )
    assert status == 200
    assert body["answers"]["intent"]["choice"] == "billing"


def test_universal_gateway_soft_distill_cutover_drift_and_freeze() -> None:
    cfg = CutoverConfig(
        milestones=(32, 64),
        k_min=2,
        alpha_target=0.15,
        canary_rate=0.05,
        sparse_dim=256,
        dense_dim=32,
        score_tolerance=0.25,
    )
    svc = UniversalDecisionService(
        upstream_handlers={"gemini": _mock_universal_teacher},
        bearer_token="tok-distill",
        cutover_config=cfg,
        distill_mode="soft",
        soft_blend=0.5,
    )
    client = UniversalDecisionsClient(service=svc, api_key="tok-distill")

    simple_questions = {
        "intent": _SUPPORT_QUESTIONS["intent"],
        "severity": _SUPPORT_QUESTIONS["severity"],
    }

    # Feed shadow observations across both classes (68 samples -> n_calib >= 22 for LTT cutover)
    for i in range(34):
        client.decisions.create(
            input=f"invoice refund payment billing charge request {i}",
            questions=simple_questions,
            slot="triage",
        )
        client.decisions.create(
            input=f"critical server crash outage api error stacktrace {i}",
            questions=simple_questions,
            slot="triage",
        )

    assert 32 <= svc.stats["upstream_teacher_calls"] <= 68

    # Additional follow-ups execute locally
    local_hits = svc.stats["local_fastpath_requests"]
    for i in range(6):
        resp = client.decisions.create(
            input=f"invoice refund payment billing charge request {i}",
            questions=simple_questions,
            slot="triage",
        )
        if resp.local_execution:
            local_hits += 1
    assert local_hits >= 1
    assert svc.stats["local_fastpath_requests"] >= 1

    # Schema drift isolation on the same slot
    drifted_questions = {
        "intent": {
            "type": "choice",
            "criteria": {
                "billing": "invoice refund",
                "technical": "error crash",
                "security": "breach leak",
            },
        },
    }
    drift_resp = client.decisions.create(
        input="suspicious credential leak detected",
        questions=drifted_questions,
        slot="triage",
    )
    assert drift_resp.schema_drift_detected is True
    assert drift_resp.schema_drift["drifted"] is True
    assert "security" in str(drift_resp.schema_drift)
    assert svc.stats["schema_drift_events"] == 1

    # Freeze learning blocks teacher observation & state growth
    svc.set_freeze_learning(True)
    calls_before = svc.stats["upstream_teacher_calls"]
    frozen_resp = client.decisions.create(
        input="brand new invoice refund billing item",
        questions=simple_questions,
        slot="triage",
    )
    # Even if it ran local or teacher, teacher counter on new unseen input won't trigger admittance mutation
    assert isinstance(frozen_resp.answers.intent.choice, str)
    # Force teacher while frozen via header: must NOT increment admitted lifecycle records
    lc = [v for k, v in svc._lifecycles.items() if k.startswith("triage:")][0]
    assert isinstance(lc, SoftDistillingLifecycle)
    admitted_before = len(lc.lifecycle.admitted_keys)
    svc.handle_request(
        path="/v1beta/decisions",
        headers={
            "Authorization": "Bearer tok-distill",
            "Content-Type": "application/json",
            "X-System1-Freeze-Learning": "true",
        },
        raw_body=json.dumps(
            {
                "input": "brand new frozen probe should not enter records",
                "questions": simple_questions,
                "slot": "triage",
                "force_teacher": True,
            }
        ).encode(),
    )
    assert len(lc.lifecycle.admitted_keys) == admitted_before
    assert svc.stats["upstream_teacher_calls"] >= calls_before


def test_universal_gateway_loopback_http_zero_egress_and_asgi() -> None:
    svc = UniversalDecisionService(
        upstream_handlers={"gemini": _mock_universal_teacher},
        bearer_token="loopback-secret",
    )
    with UniversalDecisionsHTTPServer(svc, host="127.0.0.1", port=0) as srv:
        client = UniversalDecisionsClient(
            base_url=srv.base_url,
            api_key="loopback-secret",
            origin="http://127.0.0.1",
        )
        res = client.decisions.create(
            input="App crashed with 500 error outage",
            questions=_SUPPORT_QUESTIONS,
        )
        assert res.answers.intent.choice == "technical"
        assert res.answers.urgent.value is True

    # Zero-egress violation propagates as 403 in ASGI wrapper & HTTP
    def _egress_trap(*_: Any) -> Dict[str, Any]:
        raise ZeroEgressViolationError("Outbound LLM call blocked under zero_egress=True")

    svc_egress = UniversalDecisionService(
        upstream_handlers={"gemini": _egress_trap},
        bearer_token="loopback-secret",
    )
    asgi_app = create_universal_asgi_app(svc_egress)

    async def _run_asgi_case(scope: Dict[str, Any], body: bytes) -> Dict[str, Any]:
        sent: List[Dict[str, Any]] = []
        delivered = False

        async def _receive() -> Dict[str, Any]:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        async def _send(msg: Dict[str, Any]) -> None:
            sent.append(msg)

        await asgi_app(scope, _receive, _send)
        start = next(m for m in sent if m["type"] == "http.response.start")
        raw = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
        return {"status": start["status"], "json": json.loads(raw.decode("utf-8"))}

    # /healthz works
    health = asyncio.run(_run_asgi_case({"type": "http", "method": "GET", "path": "/healthz"}, b""))
    assert health["status"] == 200 and health["json"]["status"] == "ok"

    # Zero-egress turns into HTTP 403
    blocked = asyncio.run(
        _run_asgi_case(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1beta/decisions",
                "headers": [
                    (b"authorization", b"Bearer loopback-secret"),
                    (b"content-type", b"application/json"),
                ],
            },
            json.dumps({"input": "test", "questions": _SUPPORT_QUESTIONS}).encode(),
        )
    )
    assert blocked["status"] == 403
    assert "zero_egress" in blocked["json"]["error"]

    # Unknown route without fallback returns 404; with fallback delegates
    not_found = asyncio.run(_run_asgi_case({"type": "http", "method": "GET", "path": "/nope"}, b""))
    assert not_found["status"] == 404

    # Streaming chunked body > max_body_bytes returns 413 cleanly
    svc_small = UniversalDecisionService(
        upstream_handlers={"gemini": _mock_universal_teacher},
        bearer_token="loopback-secret",
        max_body_bytes=64,
    )
    asgi_small = UniversalDecisionASGIApp(svc_small)

    async def _run_chunked_413() -> int:
        chunks = [
            {"type": "http.request", "body": b"a" * 40, "more_body": True},
            {"type": "http.request", "body": b"b" * 40, "more_body": False},
        ]
        idx = 0
        status_out = 0

        async def _recv() -> Dict[str, Any]:
            nonlocal idx
            msg = chunks[min(idx, len(chunks) - 1)]
            idx += 1
            return msg

        async def _snd(msg: Dict[str, Any]) -> None:
            nonlocal status_out
            if msg["type"] == "http.response.start":
                status_out = int(msg["status"])

        await asgi_small({"type": "http", "method": "POST", "path": "/v1beta/decisions", "headers": []}, _recv, _snd)
        return status_out

    assert asyncio.run(_run_chunked_413()) == 413

    called_fallback = False

    async def _fallback(scope: Any, receive: Any, send: Any) -> None:
        nonlocal called_fallback
        called_fallback = True
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    app_with_fb = UniversalDecisionASGIApp(svc, fallback_app=_fallback)

    async def _run_fb() -> None:
        async def _recv() -> Dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def _snd(msg: Dict[str, Any]) -> None:
            return None

        await app_with_fb({"type": "http", "method": "GET", "path": "/custom"}, _recv, _snd)

    asyncio.run(_run_fb())
    assert called_fallback

    # mount_universal_gateway attaches routes onto any FastAPI-like object
    registered: List[Tuple[str, Any]] = []

    class _FakeApp:
        def add_api_route(self, path: str, endpoint: Any, methods: List[str]) -> None:
            registered.append((path, tuple(methods)))

    fake = _FakeApp()
    assert mount_universal_gateway(fake, svc) is fake
    assert len(registered) == 4


# ---------------------------------------------------------------------------
# 3. Pure-NumPy Decision Utility, Ordinal Calibration, Platt/Beta & Cascade
# ---------------------------------------------------------------------------


def test_utility_calibrator_vacuity_ladder_and_crc_guarantee() -> None:
    m = 20.0
    cost_matrix = np.array(
        [
            [0.0, 0.4 * m, m],
            [0.15, 0.15, 0.6 * m],
            [5.0, 5.0, 5.0],
        ],
        dtype=np.float64,
    )

    # Vacuous case: n_cal=10 cannot satisfy (0 + 1)/(10 + 1) <= 0.05 -> lam_crc = -1.0
    rng = np.random.default_rng(7)
    p_small = np.tile(np.array([[0.95, 0.04, 0.01]]), (10, 1))
    y_small = np.zeros(10, dtype=np.int64)
    cal_vac = UtilityCalibrator(cost_matrix, safe_action=2, alpha=0.05).fit(p_small, y_small)
    assert cal_vac.lam_crc == -1.0
    assert cal_vac.is_vacuous is True
    assert np.all(cal_vac.predict(p_small) == 2)

    # Non-vacuous case with n_cal=120: check expected_costs and CRC guarantee on held-out split
    n = 240
    logits = rng.normal(scale=1.8, size=(n, 3))
    logits -= np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    y_true = np.array([rng.choice(3, p=probs[i]) for i in range(n)], dtype=np.int64)

    cal = UtilityCalibrator(cost_matrix, safe_action=2, alpha=0.08, ladder_mode=True)
    cal.fit(probs[:120], y_true[:120])
    assert cal.lam_crc >= 0.0
    assert not cal.is_vacuous

    ec = cal.expected_costs(probs[120:])
    assert ec.shape == (120, 3)

    acts = cal.predict(probs[120:])
    realized_excess_harm = cal.L[acts, y_true[120:]]
    # Marginal CRC guarantee E[L] <= alpha with finite-sample slack on N=120 test split
    assert float(np.mean(realized_excess_harm)) <= 0.14

    # Direct ladder check: if action 0 excess harm exceeds lam_crc while action 1 is below lam_crc,
    # ladder_mode steps from 0 -> 1 rather than jumping straight to safe_action=2.
    manual_cal = UtilityCalibrator(cost_matrix, safe_action=2, alpha=0.10, ladder_mode=True)
    manual_cal.lam_crc = 0.05
    manual_cal._is_fitted = True
    # p = [0.75, 0.25, 0.0] -> e[0]=2.0, eh[0]=(0.25*(8-5)/15)=0.05? Wait with m=20, c_esc=5, denom=15:
    # For p = [0.65, 0.35, 0.0]: e[0]=2.8, e[1]=0.15 -> pick a_bayes=1 directly.
    # For p = [0.80, 0.00, 0.20]: e[0]=4.0 > e[2] vs for p = [0.85, 0.10, 0.05]:
    # Let's craft a custom ladder cost matrix where Bayes prefers 0, eh[0] > lam_crc >= eh[1]:
    c_lad = np.array(
        [
            [0.0, 20.0, 20.0],   # eh = [0, 0.75, 0.75]
            [1.0, 1.0, 20.0],    # eh = [0, 0, 0.75]
            [5.0, 5.0, 5.0],     # safe_action = 2
        ]
    )
    lad = UtilityCalibrator(c_lad, safe_action=2, alpha=0.10, ladder_mode=True)
    lad.lam_crc = 0.02
    lad._is_fitted = True
    # p = [0.96, 0.04, 0.0]: e[0]=0.80 < e[1]=1.0 < e[2]=5.0 so a_bayes=0; eh[0]=0.03 > 0.02, eh[1]=0.0 <= 0.02 -> picks 1!
    p_probe = np.array([[0.96, 0.04, 0.0]])
    assert int(lad.predict(p_probe)[0]) == 1

    non_lad = UtilityCalibrator(c_lad, safe_action=2, alpha=0.10, ladder_mode=False)
    non_lad.lam_crc = 0.02
    non_lad._is_fitted = True
    assert int(non_lad.predict(p_probe)[0]) == 2


def test_ordinal_heads_cqr_and_binary_platt_beta_calibrator() -> None:
    assert should_use_soft_ordinal(2) is True
    assert should_use_soft_ordinal(5) is True
    assert should_use_soft_ordinal(1) is False
    assert should_use_soft_ordinal(6) is False

    rng = np.random.default_rng(19)
    n, d, k = 160, 6, 4
    X = rng.normal(size=(n, d))
    true_w = np.array([1.8, -1.2, 0.7, 0.0, 0.3, -0.5])
    latent = X @ true_w + rng.normal(scale=0.35, size=n)
    y = np.digitize(latent, [-1.2, 0.0, 1.2]).astype(np.int64)

    cum_head = CumulativeLogitOrdinalHead(n_classes=k, l2=1e-2).fit(X[:100], y[:100])
    assert cum_head.thresholds_ is not None
    assert np.all(np.diff(cum_head.thresholds_) > 0.0)
    # Check optimal parameters sit at zero projected gradient (central finite-diff optimality < 1e-4)
    eps = 1e-6
    from system1.calibration import _expit_np as _sig_np
    def _cum_nll(w_vec: np.ndarray, th_vec: np.ndarray) -> float:
        eta = X[:100] @ w_vec
        tot = 0.0
        for ii in range(100):
            yi = int(y[ii])
            hi = 1.0 if yi == k - 1 else float(_sig_np(th_vec[yi] - eta[ii]))
            lo_v = 0.0 if yi == 0 else float(_sig_np(th_vec[yi - 1] - eta[ii]))
            tot -= float(np.log(max(hi - lo_v, 1e-12)))
        return tot / 100.0 + 0.5 * 1e-2 * float(np.sum(w_vec * w_vec))

    fd_grad_w = np.zeros(d, dtype=np.float64)
    for j in range(d):
        dw = np.zeros(d, dtype=np.float64)
        dw[j] = eps
        fd_grad_w[j] = (
            _cum_nll(cum_head.weights_ + dw, cum_head.thresholds_)
            - _cum_nll(cum_head.weights_ - dw, cum_head.thresholds_)
        ) / (2.0 * eps)
    assert float(np.max(np.abs(fd_grad_w))) < 1e-5
    cum_p = cum_head.predict_proba(X[100:])
    assert cum_p.shape == (60, k)
    assert np.allclose(np.sum(cum_p, axis=1), 1.0, atol=1e-6)
    assert float(np.mean(np.abs(cum_head.predict(X[100:]) - y[100:]) <= 1)) >= 0.90

    soft_head = SoftOrdinalHead(n_classes=k, sigma=0.5, l2=1e-2).fit(X[:100], y[:100])
    soft_p = soft_head.predict_proba(X[100:])
    assert soft_p.shape == (60, k)
    assert np.allclose(np.sum(soft_p, axis=1), 1.0, atol=1e-6)
    assert float(np.mean(np.abs(soft_head.predict(X[100:]) - y[100:]) <= 1)) >= 0.90

    # CQR ordinal intervals achieve empirical coverage >= 1 - alpha - finite-sample margin
    cal_p = soft_head.predict_proba(X[60:110])
    cal_y = y[60:110]
    test_p = soft_head.predict_proba(X[110:])
    test_y = y[110:]
    lo, hi = cqr_ordinal_intervals(cal_p, cal_y, test_p, alpha=0.10)
    assert lo.shape == (50,) and hi.shape == (50,)
    assert np.all((0 <= lo) & (lo <= hi) & (hi < k))
    cov = float(np.mean((lo <= test_y) & (test_y <= hi)))
    assert cov >= 0.80

    # BinaryPlattBetaCalibrator switches at n_cal >= 74
    p_raw = 1.0 / (1.0 + np.exp(-np.clip(latent, -10.0, 10.0)))
    y_bin = (y >= 2).astype(np.int64)
    cal_small = BinaryPlattBetaCalibrator(min_beta_samples=74).fit(p_raw[:73], y_bin[:73])
    assert cal_small.mode == "platt"
    p_s = cal_small.transform(p_raw[73:90])
    p_s2 = cal_small.predict_proba(p_raw[73:90])
    assert p_s.shape == (17,) and p_s2.shape == (17, 2)
    assert np.allclose(p_s2[:, 0] + p_s2[:, 1], 1.0)

    cal_large = BinaryPlattBetaCalibrator(min_beta_samples=74).fit(p_raw[:74], y_bin[:74])
    assert cal_large.mode == "beta"
    p_l2 = cal_large.predict_proba(np.column_stack([1.0 - p_raw[74:], p_raw[74:]]))
    assert p_l2.shape == (n - 74, 2)
    assert np.all((p_l2 > 0.0) & (p_l2 < 1.0))


def test_utility_cascade_router_end_to_end() -> None:
    router = UtilityCascadeRouter(error_multiplier=20.0, alpha=0.10)
    d0 = router.route("what is my order status FAQ check")
    assert isinstance(d0, CascadeRouteDecision)
    assert d0.selected_tier in ("system1_local", "fast_small_model", "frontier_reasoning_model")

    prompts: List[str] = []
    tiers: List[str] = []
    for i in range(18):
        prompts.append(f"simple status check account balance password FAQ {i}")
        tiers.append("system1_local")
        prompts.append(f"summarize meeting notes rewrite email grammar table {i}")
        tiers.append("fast_small_model")
        prompts.append(f"formal proof distributed consensus deadlock root cause audit {i}")
        tiers.append("frontier_reasoning_model")

    router.fit(prompts, tiers, cal_fraction=0.45)
    assert router.ordinal_head is not None
    assert router.binary_calibrator.mode == "platt"

    d_easy = router.route("simple status check account balance password FAQ 99")
    d_hard = router.route("formal proof distributed consensus deadlock root cause audit 99")
    assert d_hard.action_index >= d_easy.action_index
    assert set(d_hard.expected_costs.keys()) == {
        "system1_local",
        "fast_small_model",
        "frontier_reasoning_model",
    }


def test_pure_numpy_no_scipy_or_sklearn_in_wave3_modules() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, system1.calibration, system1.compat.universal, "
                "system1.integrations.drop_in, system1.explain, system1.fleet, "
                "system1.guard, system1.teaching; "
                "assert 'scipy' not in sys.modules; "
                "assert 'sklearn' not in sys.modules; "
                "assert 'orjson' not in sys.modules"
            ),
        ],
        check=True,
    )

    pkg_dir = Path(system1.__file__).resolve().parent
    forbidden = {"scipy", "sklearn", "orjson"}
    for fname in ("calibration.py", "compat/universal.py", "integrations/drop_in.py", "explain.py", "fleet.py"):
        source = (pkg_dir / fname).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden, f"{fname} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden, f"{fname} imports from {node.module}"
