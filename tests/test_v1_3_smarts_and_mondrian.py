"""Wave 1 verification tests for v1.3: soft distillation, pragmatics, zero/few-shot router, and online Mondrian conformal."""

from __future__ import annotations

import io
import json
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

from system1 import (
    AdaptiveConformalTracker,
    BooleanField,
    ChoiceField,
    CompiledSystemOneModel,
    DecisionSchema,
    OnlineMondrianConformal,
    PragmaticCompositeProjector,
    RouterDecisionResult,
    SmartsCrossZeroFewRouter,
    SystemOneCompiler,
    SystemOneEngine,
    clear_embedding_caches,
    extract_pragmatic_features,
    set_embedding_cache_maxsize,
)


class _TicketRouteSchema(DecisionSchema):
    route = ChoiceField(
        options=("billing", "incident", "access", "feature"),
        descriptions={
            "billing": "Invoice chargeback, refund request, card declined, or duplicate payment charge.",
            "incident": "Service outage, HTTP 503 gateway timeout, production error spike, or latency alert.",
            "access": "Password reset, SSO login lock, permission denied 403, or MFA token issue.",
            "feature": "Roadmap enhancement, export CSV request, dark mode UI improvement, or webhook idea.",
        },
        description="Route incoming customer and ops tickets to the right team.",
    )


class _EscalationBoolSchema(DecisionSchema):
    urgent = BooleanField(
        true_description="Production outage, HTTP 500/503 failure, or SLA breach requiring immediate page.",
        false_description="Informational question, documentation typo, or routine scheduled feedback.",
        description="Whether to page on-call immediately.",
    )


def _make_soft_route_dataset(seed: int = 17):
    rng = np.random.default_rng(seed)
    templates = {
        "billing": [
            "customer charged twice on invoice and needs a refund for card charge",
            "credit card declined during monthly subscription renewal billing cycle",
            "duplicate invoice statement amount looks wrong for enterprise plan",
            "chargeback dispute opened on recent payment receipt",
        ],
        "incident": [
            "api gateway returning http 503 service unavailable across checkout pods",
            "database latency spiked above 1500ms and worker queue is stalling",
            "production error rate crossed 5% threshold after latest deployment",
            "timeout errors hitting authentication upstream cluster",
        ],
        "access": [
            "cannot sign in through okta sso getting permission denied 403 error",
            "locked out of admin console after rotating hardware mfa authenticator",
            "need role grant to read analytics workspace datasets",
            "password reset link expired before team member activated account",
        ],
        "feature": [
            "please add weekly csv export button to usage dashboard report",
            "would love keyboard shortcuts and compact table density toggle",
            "request webhook payload filter for custom slack notifications",
            "enhancement idea to save filter presets across views",
        ],
    }
    opts = ("billing", "incident", "access", "feature")
    prompts = []
    soft_probs = []
    hard_labels = []
    for idx in range(64):
        cls_i = idx % 4
        cls_name = opts[cls_i]
        tmpl = templates[cls_name][(idx // 4) % len(templates[cls_name])]
        text = f"{tmpl} ticket-{idx} {rng.integers(10, 99)}"
        prompts.append(text)
        hard_labels.append(cls_name)
        p = np.full(4, 0.03, dtype=np.float64)
        p[cls_i] = 0.85
        p[(cls_i + 1) % 4] = 0.06
        p /= p.sum()
        soft_probs.append(p.tolist())

    eval_prompts = []
    eval_labels = []
    for idx in range(24):
        cls_i = idx % 4
        cls_name = opts[cls_i]
        tmpl = templates[cls_name][(idx + 1) % len(templates[cls_name])]
        eval_prompts.append(f"{tmpl} eval-case-{idx}")
        eval_labels.append(cls_name)
    return prompts, np.asarray(soft_probs, dtype=np.float64), hard_labels, eval_prompts, eval_labels


def test_soft_distilled_choice_and_s1m_roundtrip(tmp_path: Path) -> None:
    prompts, P_T, hard_labels, eval_prompts, eval_labels = _make_soft_route_dataset()

    # Check N=32 subset works cleanly with non-collapsing coverage & high accuracy
    compiler32 = SystemOneCompiler(_TicketRouteSchema)
    model32 = compiler32.compile_soft_distilled(
        prompts[:32],
        soft_probs=P_T[:32],
        hard_labels=hard_labels[:32],
        alpha=0.10,
    )
    eng32 = SystemOneEngine(_TicketRouteSchema, compiled_model=model32, strict_mode=True, use_cache=False)
    preds32 = [eng32.decide(p, alpha=0.10, record_receipt=False) for p in eval_prompts]
    acc32 = sum(int(r.values["route"] == y) for r, y in zip(preds32, eval_labels)) / len(eval_labels)
    assert acc32 >= 0.85

    # Full N=64 roundtrip (.s1m save & zero-config load)
    compiler64 = SystemOneCompiler(_TicketRouteSchema)
    model64 = compiler64.compile_soft_distilled(
        prompts,
        soft_probs=P_T,
        hard_labels=hard_labels,
        alpha=0.05,
        pragmatic=True,
    )
    s1m_path = tmp_path / "soft_route.s1m"
    model64.save(s1m_path)
    reloaded = CompiledSystemOneModel.load(s1m_path)

    eng_orig = SystemOneEngine(_TicketRouteSchema, compiled_model=model64, strict_mode=True, use_cache=False)
    eng_rel = SystemOneEngine(_TicketRouteSchema, compiled_model=reloaded, strict_mode=True, use_cache=False)

    accepted = 0
    correct = 0
    for p, y in zip(eval_prompts, eval_labels):
        r0 = eng_orig.decide(p, alpha=0.05, record_receipt=False)
        r1 = eng_rel.decide(p, alpha=0.05, record_receipt=False)
        assert r0.values["route"] == r1.values["route"]
        assert r0.is_ambiguous == r1.is_ambiguous
        for opt, p_val in r0.probabilities["route"].items():
            assert abs(p_val - r1.probabilities["route"][opt]) < 1e-5
        if r1.values["route"] == y:
            correct += 1
        if not r1.is_ambiguous and len(r1.conformal_sets["route"]) == 1:
            accepted += 1
    assert correct / len(eval_labels) >= 0.90
    assert accepted > 0


def test_compile_with_soft_targets_and_boolean_soft_distill(tmp_path: Path) -> None:
    prompts, P_T, hard_labels, eval_prompts, eval_labels = _make_soft_route_dataset(seed=99)

    # BooleanField soft distillation supporting both (N,) and (N, 2) probability inputs
    bool_prompts = []
    bool_probs = []
    bool_labels = []
    for i in range(40):
        is_urg = bool(i % 2 == 0)
        if is_urg:
            bool_prompts.append(f"critical http 503 outage latency spike on payment cluster #{i}")
            bool_probs.append(0.92)
        else:
            bool_prompts.append(f"minor docs spelling tweak in footer guide #{i}")
            bool_probs.append(0.08)
        bool_labels.append(is_urg)

    bool_comp = SystemOneCompiler(_EscalationBoolSchema)
    bool_model = bool_comp.compile_soft_distilled(
        bool_prompts,
        soft_probs=np.asarray(bool_probs, dtype=np.float64),
        hard_labels=bool_labels,
        alpha=0.10,
    )
    bpath = tmp_path / "bool_soft.s1m"
    bool_model.save(bpath)
    eng_bool = SystemOneEngine.load(bpath, schema=_EscalationBoolSchema, strict_mode=True, use_cache=False)
    assert eng_bool.decide("urgent http 503 failure in checkout cluster", record_receipt=False).values["urgent"] is True
    assert eng_bool.decide("minor docs spelling update", record_receipt=False).values["urgent"] is False

    # Integration through SystemOneCompiler.compile(..., soft_targets=...) with held-out calibration_exemplars
    train_ex = {"route": [(p, lbl) for p, lbl in zip(prompts[:48], hard_labels[:48])]}
    cal_ex = {"route": [(f"{p} cal", lbl) for p, lbl in zip(prompts[48:], hard_labels[48:])]}
    comp = SystemOneCompiler(_TicketRouteSchema)
    m_cal = comp.compile(
        train_ex,
        augment=False,
        calibration_exemplars=cal_ex,
        soft_targets={"route": P_T[:48]},
    )
    eng = SystemOneEngine(_TicketRouteSchema, compiled_model=m_cal, use_cache=False)
    acc = sum(
        int(eng.decide(p, record_receipt=False).values["route"] == y)
        for p, y in zip(eval_prompts, eval_labels)
    ) / len(eval_labels)
    assert acc >= 0.85


def test_pragmatic_features_and_zero_config_s1m_roundtrip(tmp_path: Path) -> None:
    feats_pos = extract_pragmatic_features("Service crashed and returned HTTP 503 with latency 1800ms")
    assert feats_pos.get("HTTP_5XX", 0.0) > 0.0
    assert feats_pos.get("HTTP_503", 0.0) > 0.0
    assert feats_pos.get("LAT_SLOW_S", 0.0) > 0.0
    feats_neg = extract_pragmatic_features(
        'The invoice looked fine, but payment did NOT succeed and returned HTTP 503 latency 1800ms charge=$12k {"retry": false}'
    )
    assert feats_neg.get("NOT_HTTP_5XX", 0.0) > 0.0
    assert feats_neg.get("NOT_HTTP_503", 0.0) > 0.0
    assert feats_neg.get("NOT_LAT_SLOW_S", 0.0) > 0.0
    assert feats_neg.get("NOT_CURR_ENTERPRISE", 0.0) > 0.0
    assert any(k.startswith("NOT_") for k in feats_neg)
    assert any(k.startswith("KV_") for k in feats_neg)

    # Cohort separation on negation/contrastive prompts
    train_texts = [
        "service healthy http 200 ok latency 18ms for customer checkout",
        "deployment verified http 200 ok zero errors in gateway logs",
        "payment succeeded cleanly with http 200 status and fast 25ms response",
        "looked healthy earlier, but checkout did not succeed and threw http 503",
        "users report portal is not working and api returns 503 timeout",
        "not resolved yet; error spike persisted after patch with 500 latency 2200ms",
    ] * 6
    train_labels = ["ok", "ok", "ok", "fail", "fail", "fail"] * 6

    class _HealthSchema(DecisionSchema):
        status = ChoiceField(options=("ok", "fail"))

    proj = PragmaticCompositeProjector.fit(
        train_texts,
        labels=train_labels,
        max_features=96,
        prag_max_dim=48,
        min_df=2,
    )
    comp = SystemOneCompiler(_HealthSchema, projector=proj)
    model = comp.compile({"status": list(zip(train_texts, train_labels))}, augment=False, calibration_split=0.25)

    # Test both inline-IDF and compact binary-IDF (.s1m) reload without projector=
    for compact_opt in (False, True):
        raw_bytes = model.to_bytes(compact=compact_opt)
        reloaded = CompiledSystemOneModel.from_bytes(raw_bytes)
        assert isinstance(reloaded.projector, PragmaticCompositeProjector)
        eng0 = SystemOneEngine(_HealthSchema, compiled_model=model, use_cache=False)
        eng1 = SystemOneEngine(_HealthSchema, compiled_model=reloaded, use_cache=False)
        for q, expected in (
            ("system healthy returning http 200 in 20ms", "ok"),
            ("checked logs, however payment is not working and returned http 503", "fail"),
        ):
            r0 = eng0.decide(q, record_receipt=False)
            r1 = eng1.decide(q, record_receipt=False)
            assert r0.values["status"] == expected
            assert r1.values["status"] == expected
            assert abs(r0.confidences["status"] - r1.confidences["status"]) < 1e-4


def test_pragmatic_s1m_rejects_unexpected_arrays() -> None:
    class _BinSchema(DecisionSchema):
        flag = ChoiceField(options=("a", "b"))

    comp = SystemOneCompiler(_BinSchema, dimension=64)
    model = comp.compile({"flag": [("alpha text", "a"), ("beta text", "b")] * 6}, augment=False)
    good = model.to_bytes()
    header_len = struct.unpack(">I", good[4:8])[0]
    header = json.loads(good[8 : 8 + header_len].decode("utf-8"))
    npz_buf = io.BytesIO(good[8 + header_len :])
    with np.load(npz_buf, allow_pickle=False) as loaded:
        arrs = {k: loaded[k] for k in loaded.files}
    arrs["_projector_prag_base_idf"] = np.ones(64, dtype=np.float64)
    out_npz = io.BytesIO()
    np.savez_compressed(out_npz, **arrs)
    hdr_bytes = json.dumps(header, sort_keys=True).encode("utf-8")
    tampered = b"S1M\x02" + struct.pack(">I", len(hdr_bytes)) + hdr_bytes + out_npz.getvalue()
    with pytest.raises(ValueError, match="unexpected pragmatic"):
        CompiledSystemOneModel.from_bytes(tampered)


def test_smarts_cross_zero_few_router_and_cache_controls() -> None:
    # Check bounded cache setters and clearing
    set_embedding_cache_maxsize(8192)
    clear_embedding_caches()

    router = SmartsCrossZeroFewRouter(_TicketRouteSchema)
    res0 = router.decide("customer needs a refund for duplicate invoice credit card billing")
    assert isinstance(res0, RouterDecisionResult)
    assert res0.values["route"] == "billing"
    assert router.predict("gateway returning http 503 timeout error spike") == "incident"
    assert router.predict("cannot sign in due to 403 mfa lock") == "access"

    # 3-shot adaptation + incremental learn + calibration
    few_prompts = [
        "refund duplicate charge on invoice",
        "card declined renewal billing",
        "http 503 outage in payment cluster",
        "high p99 latency and timeout spike",
        "okta sso password reset 403 error",
        "locked mfa admin token issue",
        "add csv export button to analytics view",
        "webhook filter roadmap improvement",
    ]
    few_labels = [
        "billing", "billing", "incident", "incident",
        "access", "access", "feature", "feature",
    ]
    router.fit_few_shot(few_prompts, few_labels)
    router.learn("export monthly billing chart to csv format", "feature")
    router.calibrate(few_prompts, few_labels, alpha=0.10)
    res_cal = router.decide("http 503 production error spike", alpha=0.10)
    assert res_cal.values["route"] == "incident"
    assert "incident" in res_cal.conformal_sets["route"]
    assert res_cal.latency_ms >= 0.0

    # BooleanField schema handling
    b_router = SmartsCrossZeroFewRouter(_EscalationBoolSchema)
    b_pred = b_router.predict("critical production outage http 503 SLA breach")
    assert b_pred is True
    assert isinstance(b_router.predict_batch(["routine docs typo question"])[0], bool)

    set_embedding_cache_maxsize(16384)


def test_core_isolation_after_wave1_imports() -> None:
    src_root = str(Path(__file__).resolve().parent.parent / "src")
    code = f"""
import sys
sys.path.insert(0, {src_root!r})
sys.modules['mlx'] = None
sys.modules['mlx.core'] = None
sys.modules['mlx.nn'] = None
import system1.core as core
p = core.PragmaticCompositeProjector.fit(
    ["http 503 gateway timeout error", "healthy 200 ok response"],
    labels=["err", "ok"],
    min_df=1,
    max_features=32,
    prag_max_dim=16,
)
_ = p.project("not ok http 503")
class S(core.DecisionSchema):
    choice = core.ChoiceField(options=("err", "ok"), descriptions={{"err": "http 503 error", "ok": "200 ok healthy"}})
r = core.SmartsCrossZeroFewRouter(S).fit_few_shot(["http 503 fail", "http 200 ok"], ["err", "ok"])
_ = r.decide("http 503 fail")
forbidden = [m for m in ("scipy", "cryptography", "sqlite3", "system1.engine") if m in sys.modules]
assert not forbidden, f"Forbidden modules loaded in system1.core: {{forbidden}}"
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr + proc.stdout


def test_adaptive_conformal_tracker_parity_and_online_mondrian() -> None:
    # 1. AdaptiveConformalTracker sliding-window sort parity and speed
    rng = np.random.default_rng(7)
    tracker = AdaptiveConformalTracker(alpha_target=0.10, max_window=512)
    ref_window = []
    lat_us = []
    for x in rng.uniform(0.0, 1.0, size=1200):
        t0 = time.perf_counter()
        got = tracker.push_calibration_score(float(x))
        lat_us.append((time.perf_counter() - t0) * 1e6)
        ref_window.append(float(x))
        if len(ref_window) > 512:
            ref_window = ref_window[-512:]
        expected = np.sort(np.asarray(ref_window, dtype=np.float64))
        assert np.array_equal(got, expected)
    assert float(np.median(lat_us)) < 50.0

    # 2. OnlineMondrianConformal localized class shift adaptation & state roundtrip
    omc = OnlineMondrianConformal(num_classes=3, alpha=0.10, q_init=[0.4, 0.4, 0.4])
    q_before_c0 = float(omc.thresholds()[0])
    q_before_c2 = float(omc.thresholds()[2])
    # Repeated errors specifically on class 2 while class 0 is well-calibrated
    for _ in range(40):
        omc.update(0, probs=[0.95, 0.03, 0.02])
        omc.update(2, probs=[0.75, 0.20, 0.05])
    q_after = omc.thresholds()
    assert q_after[2] > q_before_c2 + 0.20
    assert q_after[0] < q_before_c0
    assert 2 in omc.predict_set([0.65, 0.20, 0.15])
    assert omc.should_accept([0.99, 0.008, 0.002]) is True
    assert 2 in omc.predict_set([0.30, 0.20, 0.50])

    omc2 = OnlineMondrianConformal.from_state_dict(omc.state_dict())
    assert np.allclose(omc2.thresholds(), omc.thresholds())
    assert omc2.steps == omc.steps
