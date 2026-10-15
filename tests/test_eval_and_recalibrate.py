from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import reflex
import system1
from reflex.cli import main as reflex_cli_main
from system1 import (
    BooleanField,
    ChoiceField,
    DecisionSchema,
    EvaluationReport,
    FieldEvaluationMetrics,
    MultiChoiceField,
    RecalibrationMetrics,
    RecalibrationReport,
    ScoreField,
    SystemOneCompiler,
    SystemOneEngine,
)
from system1.cli import main as system1_cli_main


def _build_training_rows() -> list[tuple[str, dict[str, object]]]:
    base = [
        ("refund duplicate charge on invoice", {"intent": "billing", "urgent": True, "tags": ["finance", "refund"], "severity": 8.5}),
        ("credit card billed twice this month", {"intent": "billing", "urgent": True, "tags": ["finance", "refund"], "severity": 8.0}),
        ("update payment method for subscription", {"intent": "billing", "urgent": False, "tags": ["finance"], "severity": 3.0}),
        ("download receipt for annual plan", {"intent": "billing", "urgent": False, "tags": ["finance"], "severity": 2.0}),
        ("production database connection timeout", {"intent": "tech", "urgent": True, "tags": ["infra", "bug"], "severity": 9.5}),
        ("api gateway returning 502 bad gateway", {"intent": "tech", "urgent": True, "tags": ["infra", "bug"], "severity": 9.0}),
        ("sdk throws type error on initialization", {"intent": "tech", "urgent": False, "tags": ["bug"], "severity": 4.5}),
        ("webhook retry delay in staging cluster", {"intent": "tech", "urgent": False, "tags": ["infra"], "severity": 4.0}),
        ("change account display name in profile", {"intent": "account", "urgent": False, "tags": ["profile"], "severity": 1.5}),
        ("reset two factor authentication device", {"intent": "account", "urgent": True, "tags": ["profile"], "severity": 7.0}),
    ]
    return [
        (f"{text} sample {idx}", dict(labels))
        for idx in range(4)
        for text, labels in base
    ]


def _rows_to_exemplars(rows: list[tuple[str, dict[str, object]]]) -> dict[str, list[tuple[str, object]]]:
    by_field: dict[str, list[tuple[str, object]]] = {}
    for prompt, labels in rows:
        for fname, val in labels.items():
            by_field.setdefault(fname, []).append((prompt, val))
    return by_field


def _build_schema() -> DecisionSchema:
    return DecisionSchema(
        fields={
            "intent": ChoiceField(options=["billing", "tech", "account"]),
            "urgent": BooleanField(),
            "tags": MultiChoiceField(options=["finance", "refund", "infra", "bug", "profile"]),
            "severity": ScoreField(min_value=0.0, max_value=10.0),
        }
    )


def test_evaluate_multi_field_and_input_formats(tmp_path: Path) -> None:
    schema = _build_schema()
    rows = _build_training_rows()
    compiler = SystemOneCompiler(schema=schema, dimension=64, regularization=1e-2)
    compiled = compiler.compile(_rows_to_exemplars(rows), augment=False)

    model_path = tmp_path / "router.s1m"
    compiled.save(model_path)

    eval_tuples = rows[:10]
    eval_dicts = [{"input": text, "output": dict(labels)} for text, labels in eval_tuples]
    jsonl_path = tmp_path / "eval.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(row) for row in eval_dicts) + "\n",
        encoding="utf-8",
    )

    engine = SystemOneEngine.load(model_path)
    rep_compiled = system1.evaluate(compiled, eval_tuples, alpha=0.10)
    rep_engine = system1.evaluate(engine, eval_dicts, alpha=0.10)
    rep_path = reflex.evaluate(model_path, jsonl_path, alpha=0.10)

    assert isinstance(rep_compiled, EvaluationReport)
    assert rep_compiled.total_examples == 10
    assert rep_compiled.alpha == pytest.approx(0.10)
    assert set(rep_compiled.field_metrics.keys()) == {"intent", "urgent", "tags", "severity"}

    for fname, f_metrics in rep_compiled.field_metrics.items():
        assert isinstance(f_metrics, FieldEvaluationMetrics)
        assert f_metrics.total_examples == 10
        assert 0.0 <= f_metrics.accuracy <= 1.0
        assert 0.0 <= f_metrics.accepted_accuracy <= 1.0
        assert 0.0 <= f_metrics.conformal_coverage <= 1.0
        assert f_metrics.average_set_size >= 0.0
        assert 0.0 <= f_metrics.singleton_rate <= 1.0
        assert 0.0 <= f_metrics.ece <= 1.0
        assert f_metrics.brier_score >= 0.0
        assert f_metrics.nll >= 0.0
        assert f_metrics.accepted_examples + f_metrics.abstained_examples + f_metrics.unfamiliar_examples == 10
        assert f_metrics.coverage + f_metrics.abstain_rate + f_metrics.unfamiliar_rate == pytest.approx(1.0)
        assert rep_engine.field_metrics[fname].accuracy == pytest.approx(f_metrics.accuracy)
        assert rep_path.field_metrics[fname].accuracy == pytest.approx(f_metrics.accuracy)
        assert rep_path.field_metrics[fname].ece == pytest.approx(f_metrics.ece)
        assert rep_engine.field_metrics[fname].coverage == pytest.approx(f_metrics.coverage)
        assert rep_path.field_metrics[fname].coverage == pytest.approx(f_metrics.coverage)
        assert rep_engine.field_metrics[fname].abstain_rate == pytest.approx(f_metrics.abstain_rate)
        assert rep_path.field_metrics[fname].abstain_rate == pytest.approx(f_metrics.abstain_rate)
        assert rep_engine.field_metrics[fname].unfamiliar_rate == pytest.approx(f_metrics.unfamiliar_rate)
        assert rep_path.field_metrics[fname].unfamiliar_rate == pytest.approx(f_metrics.unfamiliar_rate)

    assert rep_engine.coverage == pytest.approx(rep_compiled.coverage)
    assert rep_path.coverage == pytest.approx(rep_compiled.coverage)
    assert rep_engine.abstain_rate == pytest.approx(rep_compiled.abstain_rate)
    assert rep_path.abstain_rate == pytest.approx(rep_compiled.abstain_rate)
    assert rep_engine.unfamiliar_rate == pytest.approx(rep_compiled.unfamiliar_rate)
    assert rep_path.unfamiliar_rate == pytest.approx(rep_compiled.unfamiliar_rate)
    assert rep_engine.selective_accuracy == pytest.approx(rep_compiled.selective_accuracy)
    assert rep_path.selective_accuracy == pytest.approx(rep_compiled.selective_accuracy)

    assert rep_compiled.accuracy >= 0.80
    assert rep_compiled.latency_p50_ms >= 0.0
    assert rep_compiled.latency_p95_ms >= rep_compiled.latency_p50_ms
    assert rep_compiled.latency_p99_ms >= rep_compiled.latency_p95_ms

    payload = rep_compiled.to_dict()
    assert payload["total_examples"] == 10
    assert "intent" in payload["field_metrics"]
    table_str = rep_compiled.summary_table()
    assert "EvaluationReport" in table_str
    assert "intent" in table_str


def test_evaluate_three_way_routing_breakdown() -> None:
    schema = DecisionSchema(
        fields={"intent": ChoiceField(options=["billing", "tech", "account"])}
    )
    base_rows = [
        ("invoice billing refund payment charge", {"intent": "billing"}),
        ("credit card receipt billing statement", {"intent": "billing"}),
        ("server crash timeout database exception", {"intent": "tech"}),
        ("stack trace segfault kernel panic", {"intent": "tech"}),
        ("password reset login profile settings", {"intent": "account"}),
        ("two factor authentication account email", {"intent": "account"}),
    ]
    rows = [(f"{p} {i}", y) for i in range(6) for p, y in base_rows]

    compiler = SystemOneCompiler(
        schema=schema,
        dimension=64,
        familiarity_alpha=0.25,
        prototype_alpha=0.25,
    )
    compiled = compiler.compile(_rows_to_exemplars(rows), augment=False)
    head = compiled.heads["intent"]
    head.temperature = 2.0
    head.familiarity_threshold = 0.0
    head.prototype_threshold = 0.45
    head.calibration_scores = (0.20, 0.25, 0.30, 0.35)
    head.conformal_quantile = 0.25

    eval_rows = [
        ("invoice billing refund payment charge 0", {"intent": "billing"}),
        ("invoice billing server crash password reset", {"intent": "billing"}),
        ("   ", {"intent": "account"}),
        ("zxqv plmkw jrtvn fghy", {"intent": "tech"}),
    ]

    report = system1.evaluate(compiled, eval_rows, alpha=0.50, strict=True)
    assert report.total_examples == 4
    assert report.accepted_examples >= 1
    assert report.abstained_examples >= 1
    assert report.unfamiliar_examples >= 1
    assert (
        report.accepted_examples
        + report.abstained_examples
        + report.unfamiliar_examples
        == report.total_examples
    )
    assert math.isclose(
        report.coverage + report.abstain_rate + report.unfamiliar_rate,
        1.0,
        abs_tol=1e-9,
    )

    intent_metrics = report.field_metrics["intent"]
    assert (
        intent_metrics.accepted_examples
        + intent_metrics.abstained_examples
        + intent_metrics.unfamiliar_examples
        == intent_metrics.total_examples
    )
    assert math.isclose(
        intent_metrics.coverage
        + intent_metrics.abstain_rate
        + intent_metrics.unfamiliar_rate,
        1.0,
        abs_tol=1e-9,
    )


def test_post_hoc_recalibration_preserves_weights_and_persists(tmp_path: Path) -> None:
    schema = _build_schema()
    rows = _build_training_rows()
    compiler = SystemOneCompiler(schema=schema, dimension=64, regularization=1e-2)
    compiled = compiler.compile(_rows_to_exemplars(rows), augment=False)

    weights_before = {k: ch.weights.copy() for k, ch in compiled.heads.items()}
    biases_before = {k: ch.biases.copy() for k, ch in compiled.heads.items()}
    p_before = {k: ch.P.copy() for k, ch in compiled.heads.items() if ch.P is not None}
    b_before = {k: ch.B.copy() for k, ch in compiled.heads.items() if ch.B is not None}
    severity_temp_before = compiled.heads["severity"].temperature

    compiled.heads["intent"].temperature = 5.0
    compiled.heads["intent"].calibration_scores = (0.001, 0.002)
    compiled.heads["intent"].conformal_quantile = 0.002

    recal_report = compiled.recalibrate(rows, alpha=0.10, mondrian=True)
    assert isinstance(recal_report, RecalibrationReport)
    assert recal_report.alpha == pytest.approx(0.10)
    assert recal_report.calibration_size == len(rows)
    assert set(recal_report.field_metrics.keys()) == {"intent", "urgent", "tags", "severity"}

    intent_recal = recal_report.field_metrics["intent"]
    assert isinstance(intent_recal, RecalibrationMetrics)
    assert intent_recal.previous_temperature == pytest.approx(5.0)
    assert intent_recal.calibrated_temperature < 5.0
    assert intent_recal.calibrated_nll < intent_recal.initial_nll
    assert len(intent_recal.calibration_scores) > 0
    assert len(intent_recal.class_calibration_scores) == 3

    severity_recal = recal_report.field_metrics["severity"]
    assert severity_recal.calibrated_temperature == pytest.approx(severity_temp_before)
    assert compiled.heads["severity"].temperature == pytest.approx(severity_temp_before)

    for fname in schema.fields:
        assert np.array_equal(compiled.heads[fname].weights, weights_before[fname])
        assert np.array_equal(compiled.heads[fname].biases, biases_before[fname])
        if fname in p_before and compiled.heads[fname].P is not None:
            assert np.array_equal(compiled.heads[fname].P, p_before[fname])
        if fname in b_before and compiled.heads[fname].B is not None:
            assert np.array_equal(compiled.heads[fname].B, b_before[fname])

    eval_after = compiled.evaluate_dataset(rows[:10], alpha=0.10)
    assert eval_after.field_metrics["intent"].conformal_coverage >= 0.80
    assert eval_after.field_metrics["severity"].conformal_coverage >= 0.80

    raw_bytes = compiled.to_bytes()
    restored_from_bytes = system1.CompiledSystemOneModel.from_bytes(raw_bytes)
    assert restored_from_bytes.heads["intent"].temperature == pytest.approx(
        compiled.heads["intent"].temperature
    )
    assert restored_from_bytes.heads["intent"].conformal_quantile == pytest.approx(
        compiled.heads["intent"].conformal_quantile
    )
    assert np.allclose(
        restored_from_bytes.heads["intent"].calibration_scores,
        compiled.heads["intent"].calibration_scores,
    )
    assert len(restored_from_bytes.heads["intent"].class_calibration_scores) == 3

    saved_path = tmp_path / "recalibrated.s1m"
    compiled.save(saved_path)
    restored_from_disk = system1.CompiledSystemOneModel.load(saved_path)
    eval_disk = restored_from_disk.evaluate_dataset(rows[:10], alpha=0.10)
    assert eval_disk.field_metrics["intent"].ece == pytest.approx(eval_after.field_metrics["intent"].ece)
    assert eval_disk.field_metrics["intent"].conformal_coverage == pytest.approx(
        eval_after.field_metrics["intent"].conformal_coverage
    )


def test_standalone_engine_recalibrate_and_evaluate() -> None:
    schema = DecisionSchema(
        fields={"intent": ChoiceField(options=["billing", "tech"])}
    )
    engine = SystemOneEngine(schema=schema, dimension=32)
    base_rows = [
        ("refund duplicate invoice charge", {"intent": "billing"}),
        ("payment failed on credit card", {"intent": "billing"}),
        ("database connection reset error", {"intent": "tech"}),
        ("api server 500 internal error", {"intent": "tech"}),
    ]
    train_rows = [(f"{p} {i}", y) for i in range(5) for p, y in base_rows]
    for text, labels in train_rows:
        engine.learn(text, labels)

    w_before = engine.model.heads["intent"].weights.copy()
    recal = engine.recalibrate(train_rows, alpha=0.05)
    assert isinstance(recal, RecalibrationReport)
    assert np.array_equal(engine.model.heads["intent"].weights, w_before)
    assert engine.conformal_predictors["intent"].is_calibrated

    eval_rep = system1.evaluate(engine, train_rows)
    assert eval_rep.alpha == pytest.approx(0.05)
    assert eval_rep.accuracy >= 0.90


def test_cli_eval_subcommand_and_reflex_parity(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    schema = DecisionSchema(
        fields={"intent": ChoiceField(options=["billing", "tech"])}
    )
    base_rows = [
        ("refund duplicate invoice charge", {"intent": "billing"}),
        ("payment failed on credit card", {"intent": "billing"}),
        ("database connection reset error", {"intent": "tech"}),
        ("api server 500 internal error", {"intent": "tech"}),
    ]
    rows = [(f"{p} {i}", y) for i in range(4) for p, y in base_rows]
    compiled = SystemOneCompiler(schema=schema, dimension=32).compile(
        _rows_to_exemplars(rows), augment=False
    )

    model_path = tmp_path / "cli_model.s1m"
    compiled.save(model_path)

    data_path = tmp_path / "cli_eval.jsonl"
    data_path.write_text(
        "\n".join(json.dumps({"input": p, "output": y}) for p, y in rows[:4]) + "\n",
        encoding="utf-8",
    )

    rc_text = system1_cli_main(["eval", "--model", str(model_path), "--data", str(data_path), "--alpha", "0.05"])
    assert rc_text == 0
    out_text = capsys.readouterr().out
    assert "EvaluationReport" in out_text
    assert "intent" in out_text

    rc_json = system1_cli_main(
        ["eval", "--model", str(model_path), "--data", str(data_path), "--alpha", "0.05", "--json"]
    )
    assert rc_json == 0
    payload_s1 = json.loads(capsys.readouterr().out)
    assert payload_s1["total_examples"] == 4
    assert math.isclose(
        payload_s1["coverage"] + payload_s1["abstain_rate"] + payload_s1["unfamiliar_rate"],
        1.0,
        abs_tol=1e-9,
    )

    rc_reflex = reflex_cli_main(
        ["eval", "--model", str(model_path), "--data", str(data_path), "--alpha", "0.05", "--json"]
    )
    assert rc_reflex == 0
    payload_rx = json.loads(capsys.readouterr().out)
    assert payload_rx["accuracy"] == pytest.approx(payload_s1["accuracy"])
    assert payload_rx["field_metrics"]["intent"]["ece"] == pytest.approx(
        payload_s1["field_metrics"]["intent"]["ece"]
    )

    calib_path = tmp_path / "cli_calib.jsonl"
    calib_path.write_text(
        "\n".join(json.dumps({"input": p, "output": y}) for p, y in rows) + "\n",
        encoding="utf-8",
    )
    out_model_path = tmp_path / "cli_recalibrated.s1m"
    rc_pos = system1_cli_main(
        [
            "eval",
            str(model_path),
            str(data_path),
            "--recalibrate",
            str(calib_path),
            "-o",
            str(out_model_path),
            "--alpha",
            "0.05",
            "--json",
        ]
    )
    assert rc_pos == 0
    payload_pos = json.loads(capsys.readouterr().out)
    assert out_model_path.is_file()
    assert "recalibration" in payload_pos
    assert payload_pos["recalibration"]["ece_after"] <= payload_pos["recalibration"]["ece_before"] + 1e-6
    assert payload_pos["empirical_set_coverage"] == pytest.approx(payload_pos["conformal_coverage"])
    assert payload_pos["latency_ms_p50"] == pytest.approx(payload_pos["p50_latency_ms"])

