"""Tests for System 1 v1.2 semantic modes, imbalance calibration, joint calibration, and e-process risk monitors."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from system1 import (
    ChoiceField,
    CompiledSystemOneModel,
    DecisionSchema,
    JointSchemaCalibrator,
    JointSelectiveCertificate,
    SequentialRiskMonitor,
    SystemOneCompiler,
    SystemOneEngine,
    compute_strict_loo_weights,
)
from system1.calibration import DecisionCalibrator, VALID_MONITOR_MODES


def _assert_fresh_no_scipy(snippet: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Fresh subprocess failed:\n{proc.stderr}"


def _triage_schema() -> DecisionSchema:
    return DecisionSchema(
        schema_name="V12SemanticTriage",
        fields={
            "intent": ChoiceField(options=["billing", "refund", "technical", "account"]),
        },
    )


def _triage_exemplars() -> dict[str, list[tuple[str, str]]]:
    base = [
        ("refund my duplicate credit card charge", "refund"),
        ("please issue a refund for damaged item", "refund"),
        ("chargeback refund needed on invoice", "refund"),
        ("cancel order and return money refund", "refund"),
        ("monthly billing statement overcharge check", "billing"),
        ("update billing card payment details", "billing"),
        ("download tax invoice from billing portal", "billing"),
        ("subscription renewal billing cycle date", "billing"),
        ("production api returns 502 gateway timeout", "technical"),
        ("database connection pool exhausted error", "technical"),
        ("ssl handshake failure on backend worker", "technical"),
        ("application crash trace on login submit", "technical"),
        ("reset password and unlock user account", "account"),
        ("two factor authentication token not working", "account"),
        ("change primary owner email on account", "account"),
        ("grant admin role permission to account", "account"),
    ]
    rows = []
    for rep in range(3):
        for p, y in base:
            rows.append((f"{p} batch {rep}", y))
    return {"intent": rows}


def test_rejected_experimental_semantic_mode_and_s1m_roundtrip(tmp_path: Path) -> None:
    schema = _triage_schema()
    ex = _triage_exemplars()

    with pytest.raises(ValueError, match="semantic_mode"):
        SystemOneCompiler(schema, semantic_mode="hyp_d_orf_gelu")

    comp_default = SystemOneCompiler(schema)
    with pytest.raises(ValueError, match="semantic_mode"):
        comp_default.compile(exemplars=ex, semantic_mode="hyp_d_orf_gelu")

    probes = [
        "please issue refund on duplicate card charge",
        "monthly billing invoice tax pdf",
        "api gateway 502 timeout error",
        "locked user account reset password",
        "quantum unicorn zxqvw unrelated noise",
    ]

    for mode in ("default", "v1_1", "v1_2_fisher", "v1_2_rbf"):
        comp = SystemOneCompiler(
            schema,
            semantic_mode=mode,
            proto_blend=0.25,
            fisher_reg=1.0,
            familiarity_alpha=0.05,
            prototype_alpha=0.05,
        )
        compiled = comp.compile(exemplars=ex, augment=False)
        model_path = tmp_path / f"skill_{mode}.s1m"
        compiled.save(model_path)
        reloaded = CompiledSystemOneModel.load(model_path)

        eng_pre = SystemOneEngine(schema, model=compiled, use_cache=False)
        eng_post = SystemOneEngine(schema, model=reloaded, use_cache=False)
        pre_batch = eng_pre.decide_batch(probes, use_cache=False)
        post_batch = eng_post.decide_batch(probes, use_cache=False)
        post_single = [eng_post.decide(p, use_cache=False) for p in probes]

        for a, b, c in zip(pre_batch, post_batch, post_single):
            assert a.values == b.values == c.values
            assert a.is_ambiguous == b.is_ambiguous == c.is_ambiguous
            assert a.conformal_sets == b.conformal_sets == c.conformal_sets
            assert set(a.probabilities["intent"].keys()) == set(b.probabilities["intent"].keys())
            assert set(b.probabilities["intent"].keys()) == set(c.probabilities["intent"].keys())
            for opt in a.probabilities["intent"]:
                assert np.isclose(a.probabilities["intent"][opt], b.probabilities["intent"][opt], atol=1e-4)
                assert np.isclose(b.probabilities["intent"][opt], c.probabilities["intent"][opt], atol=1e-5)

    _assert_fresh_no_scipy(
        """
import sys
from system1 import (
    AutoCutoverLifecycle,
    ChoiceField,
    DecisionSchema,
    JointSchemaCalibrator,
    MCPStreamableHTTPServer,
    SequentialRiskMonitor,
    SystemOneCompiler,
    SystemOneEngine,
)
s = DecisionSchema(schema_name="S", fields={"intent": ChoiceField(options=["a", "b"])})
ex = {"intent": [(f"alpha {i}", "a") for i in range(6)] + [(f"beta {i}", "b") for i in range(6)]}
m = SystemOneCompiler(s, semantic_mode="v1_2_rbf").compile(exemplars=ex, augment=False)
eng = SystemOneEngine(s, model=m, use_cache=False)
eng.decide("alpha 0", use_cache=False)
eng.decide_batch(["alpha 0", "beta 0"], use_cache=False)
SequentialRiskMonitor(mode="capital_ons").update(False)
AutoCutoverLifecycle(s)
assert "scipy" not in sys.modules
"""
    )


def test_compute_strict_loo_weights_downweights_mislabeled_outliers() -> None:
    rng = np.random.default_rng(123)
    centers = np.array(
        [
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
        ],
        dtype=np.float64,
    )
    labels = np.repeat(np.arange(3), 20)
    X = centers[labels] + 0.15 * rng.normal(size=(60, 4))
    y_noisy = labels.copy()
    y_noisy[0] = 1
    y_noisy[20] = 2
    y_noisy[40] = 0
    Y = np.eye(3, dtype=np.float32)[y_noisy]

    sw, clean_mask = compute_strict_loo_weights(X.astype(np.float32), Y, reg_loo=0.5, min_samples=40)
    assert sw.shape == (60,)
    assert clean_mask.shape == (60,)
    for bad_idx in (0, 20, 40):
        assert sw[bad_idx] < 0.25
        assert not bool(clean_mask[bad_idx])
    clean_indices = [i for i in range(60) if i not in (0, 20, 40)]
    assert float(np.mean(sw[clean_indices])) > 0.95

    sw_small, mask_small = compute_strict_loo_weights(X[:15].astype(np.float32), Y[:15], min_samples=40)
    assert np.allclose(sw_small, 1.0)
    assert bool(np.all(mask_small))


def test_decision_calibrator_imbalance_aware_fail_closed_and_recall() -> None:
    rng = np.random.default_rng(9)
    small_logits = rng.normal(size=(35, 3))
    small_labels = rng.integers(0, 3, size=35)
    cal_small = DecisionCalibrator().fit_imbalance_aware(
        small_logits, small_labels, alpha=0.05, delta=0.05
    )
    assert cal_small.margin_threshold is not None
    assert cal_small.margin_threshold >= 1.0
    small_probs = cal_small.calibrate_logits(small_logits)
    assert not np.any(cal_small.should_accept_probs(small_probs))

    n = 240
    labels = np.array([0] * 204 + [1] * 18 + [2] * 18)
    logits = np.full((n, 3), -2.5, dtype=np.float64)
    for i, y in enumerate(labels):
        if y == 0:
            logits[i, 0] = 2.5 + 0.3 * rng.normal()
        else:
            logits[i, y] = 1.6 + 0.2 * rng.normal()
            logits[i, 0] = 1.9 + 0.2 * rng.normal()

    raw_preds = np.argmax(logits, axis=1)
    raw_min_recall = float(np.mean(raw_preds[labels > 0] == labels[labels > 0]))

    cal = DecisionCalibrator().fit_imbalance_aware(
        logits, labels, tau=0.6, alpha=0.10, delta=0.05, seed=9
    )
    adj_probs = cal.calibrate_logits(logits)
    adj_preds = np.argmax(adj_probs, axis=1)
    adj_min_recall = float(np.mean(adj_preds[labels > 0] == labels[labels > 0]))
    assert np.all(np.isfinite(adj_probs))
    assert adj_min_recall >= raw_min_recall
    assert adj_min_recall > 0.70
    _assert_fresh_no_scipy(
        """
import sys, numpy as np
from system1.calibration import DecisionCalibrator
rng = np.random.default_rng(1)
logits = rng.normal(size=(80, 3))
labels = rng.integers(0, 3, size=80)
DecisionCalibrator().fit_imbalance_aware(logits, labels, alpha=0.1, delta=0.05)
assert "scipy" not in sys.modules
"""
    )


def test_joint_schema_calibrator_and_engine_calibrate_joint() -> None:
    schema = DecisionSchema(
        schema_name="CoupledRouting",
        fields={
            "intent": ChoiceField(options=["refund", "billing", "technical"]),
            "queue": ChoiceField(options=["finance", "support", "engineering"]),
            "priority": ChoiceField(options=["normal", "high"]),
        },
    )
    allowed_pairs = [
        ("issue refund for invoice charge", {"intent": "refund", "queue": "finance", "priority": "high"}),
        ("refund damaged shipment return", {"intent": "refund", "queue": "finance", "priority": "normal"}),
        ("monthly billing statement invoice", {"intent": "billing", "queue": "finance", "priority": "normal"}),
        ("urgent billing card declined overcharge", {"intent": "billing", "queue": "finance", "priority": "high"}),
        ("api 502 server gateway timeout", {"intent": "technical", "queue": "engineering", "priority": "high"}),
        ("minor documentation rendering glitch", {"intent": "technical", "queue": "engineering", "priority": "normal"}),
    ]
    train_examples = [(f"{p} tr{r}", dict(v)) for r in range(8) for p, v in allowed_pairs]
    calib_examples = [(f"{p} cal{r}", dict(v)) for r in range(10) for p, v in allowed_pairs]

    engine = SystemOneEngine(schema, dimension=256, use_cache=False)
    for p, v in train_examples:
        engine.learn(p, v)
    digest_before = engine._calibration_digest()

    cert = engine.calibrate_joint(
        train_examples,
        calib_examples,
        gamma=1.0,
        alpha=0.15,
        delta=0.05,
        score_kind="margin",
    )
    assert isinstance(cert, JointSelectiveCertificate)
    assert cert.qualified is True
    assert cert.empirical_risk <= 0.15
    assert engine._calibration_digest() != digest_before
    assert isinstance(engine.joint_calibrator, JointSchemaCalibrator)

    probes = [p for p, _ in allowed_pairs] + ["unknown random noise zxqvw"]
    singles = [engine.decide(p, use_cache=False) for p in probes]
    batched = engine.decide_batch(probes, use_cache=False)
    for s, b in zip(singles, batched):
        assert s.values == b.values
        assert s.is_ambiguous == b.is_ambiguous
        if not s.is_ambiguous:
            expected_queue = "engineering" if s.values["intent"] == "technical" else "finance"
            assert s.values["queue"] == expected_queue


def test_sequential_risk_monitor_modes_and_null_vs_drift() -> None:
    with pytest.raises(ValueError, match="mode"):
        SequentialRiskMonitor(alpha=0.05, delta=0.05, mode="naive_binom_peeking")

    rng = np.random.default_rng(2026)
    n_null_streams = 150
    n_drift_streams = 50
    stream_len = 200

    for mode in VALID_MONITOR_MODES:
        null_alarms = 0
        for _ in range(n_null_streams):
            mon = SequentialRiskMonitor(alpha=0.05, delta=0.05, mode=mode)
            events = rng.random(stream_len) < 0.05
            steps = mon.update_batch([bool(x) for x in events])
            if any(st.alarm for st in steps):
                null_alarms += 1
        assert (null_alarms / n_null_streams) <= 0.08

        drift_alarms = 0
        for _ in range(n_drift_streams):
            mon = SequentialRiskMonitor(alpha=0.05, delta=0.05, mode=mode)
            events = rng.random(stream_len) < 0.22
            steps = mon.update_batch([bool(x) for x in events])
            if any(st.alarm for st in steps):
                drift_alarms += 1
        assert (drift_alarms / n_drift_streams) >= 0.90

    mon = SequentialRiskMonitor(alpha=0.05, delta=0.05, mode="capital_ons")
    for _ in range(25):
        mon.update(True)
    mon.reset()
    st = mon.update(False)
    assert st.step == 1
    assert st.audited_count == 1
    assert st.alarm is False
