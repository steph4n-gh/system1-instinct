"""Focused experiment controls; run explicitly with benchmark dependencies."""
import gzip
import json
import math
import os
import shutil
import sys
from pathlib import Path

import pytest

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from prepare import PROTOCOL, teacher_labels
from run import choose_threshold, upper_bound
from verify import exact_upper, threshold_on_dev, rebuild, verify, sha


@pytest.mark.parametrize("errors,count", [(0, 0), (0, 1), (0, 597), (0, 1000), (1, 1000), (10, 1000), (1000, 1000)])
def test_independent_bound_agrees(errors, count):
    assert math.isclose(upper_bound(errors, count, .0025), exact_upper(errors, count, .0025), abs_tol=1e-10)


def test_whole_confidence_ties_and_undefined_gate():
    rows = [dict(id="a", prediction="x", confidence=.9, eligible=True),
            dict(id="b", prediction="x", confidence=.9, eligible=True),
            dict(id="c", prediction="x", confidence=.8, eligible=False)]
    truth = dict(a="x", b="y", c="x")
    assert choose_threshold(rows, truth, .01, 1) is None
    assert threshold_on_dev(rows, truth, .01, 1) is None
    truth["b"] = "x"
    assert choose_threshold(rows, truth, .01, 2) == threshold_on_dev(rows, truth, .01, 2) == .9
    assert choose_threshold(rows, truth, .01, 3) is None


def test_teacher_exact_noise_and_wrong_labels():
    classes = ["domain_a", "domain_b", "other_a"]
    rows = [dict(id=f"{label}:{i}", truth=label) for label in classes for i in range(20)]
    noisy = teacher_labels(rows, 17, classes)
    assert sum(noisy[r["id"]] != r["truth"] for r in rows) == 6
    assert noisy == teacher_labels(rows[::-1], 17, classes)
    for label in classes:
        assert sum(noisy[r["id"]] != label for r in rows if r["truth"] == label) == 2
    assert sum(teacher_labels(rows[:23], 17, classes, per_class=False)[r["id"]] != r["truth"] for r in rows[:23]) == round(23*.1)


def test_empty_predictions_are_not_perfect_accuracy():
    from verify import metrics
    value = metrics([dict(id="a", prediction=None, eligible=False, confidence=0., latency_ms=0.)], {"a": "x"}, None, False)
    assert value["local_error_rate"] is None
    assert value["local_error_upper_95"] is None
    assert value["coverage"] == 0


def test_protocol_error_budget_and_multiplicity():
    assert PROTOCOL["local_error_budget"] == .01
    assert PROTOCOL["teacher_error_rate"] == .1
    assert len(PROTOCOL["methods"])*len(PROTOCOL["noise_seeds"]) == 20
    assert exact_upper(0, 596, .0025) > .01
    assert exact_upper(0, 597, .0025) <= .01


@pytest.mark.parametrize("attack", ["hash", "gate", "cohort_overlap", "teacher_repair", "test_answers"])
def test_recorded_evidence_negative_controls(tmp_path, attack):
    """Reject concrete corruptions, even with recomputed hashes where applicable."""
    location = os.environ.get("SYSTEM1_HELDOUT_DATA_DIR")
    if not location:
        pytest.skip("Download pinned data and set SYSTEM1_HELDOUT_DATA_DIR")
    results_dir = Path(__file__).parent / "results"
    if not (results_dir / "predictions.json.gz").is_file():
        pytest.skip("Run benchmarks/heldout_traffic/run.py to generate predictions.json.gz")
    data = tmp_path/"data"
    evidence = tmp_path/"evidence"
    shutil.copytree(location, data)
    shutil.copytree(results_dir, evidence)
    manifest_path = evidence/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if attack == "hash":
        with (evidence/"costs.json").open("ab") as stream:
            stream.write(b" ")
    elif attack == "gate":
        path = evidence/"gates.json"
        gates = json.loads(path.read_text())
        gates["instinct_hybrid:17"]["qualified"] = not bool(gates["instinct_hybrid:17"]["qualified"])
        path.write_text(json.dumps(gates))
        manifest[path.name] = sha(path.read_bytes())
    elif attack in ("cohort_overlap", "teacher_repair"):
        path = evidence/"predictions.json.gz"
        predictions = json.loads(gzip.decompress(path.read_bytes()))
        if attack == "cohort_overlap":
            rows = predictions["exact_cache:17"]["test"]
            rows[1]["id"] = rows[0]["id"]
        else:
            cohorts, _ = rebuild(data, PROTOCOL)
            truths = {r["id"]: r["truth"] for r in cohorts["teach"]}
            row = next(r for r in predictions["exact_cache:17"]["repeat"] if r["prediction"] != truths[r["id"]])
            row["prediction"] = truths[row["id"]]
        path.write_bytes(gzip.compress(json.dumps(predictions).encode()))
        manifest[path.name] = sha(path.read_bytes())
    else:
        path = data/"answers.json"
        answers = json.loads(path.read_text())
        key = next(iter(answers))
        answers[key] = "alarm_query" if answers[key] != "alarm_query" else "alarm_set"
        path.write_text(json.dumps(answers))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify(data, evidence)


def test_verify_raises_value_error_not_assertion_error(tmp_path):
    """Ensure verify() uses explicit ValueError checks that survive python -O."""
    bad_evidence = tmp_path / "bad_evidence"
    bad_evidence.mkdir()
    (bad_evidence / "manifest.json").write_text(json.dumps({"unexpected": "hash"}))
    with pytest.raises(ValueError, match="Manifest file set mismatch"):
        verify(tmp_path, bad_evidence)


def test_dev_threshold_enforces_clopper_pearson_safety_margin_when_errors_present():
    """Thresholds with errors > 0 right at the nominal empirical budget must be rejected."""
    rows = [
        dict(id=f"r{i}", prediction="x", confidence=0.99 - i * 0.001, eligible=True)
        for i in range(100)
    ]
    truth = {f"r{i}": "x" for i in range(100)}
    truth["r50"] = "y"  # 1 error in 100 examples = 1.0% empirical, but upper bound > 1.0%
    assert choose_threshold(rows, truth, 0.01, 50) == rows[49]["confidence"]
    assert threshold_on_dev(rows, truth, 0.01, 50) == rows[49]["confidence"]
    # At 5% budget, 1 error in 100 has upper_bound(1, 100, 0.0025) = 0.0789 > 0.05,
    # whereas 1 error in 200 has upper_bound(1, 200, 0.0025) = 0.0401 <= 0.05.
    rows200 = [
        dict(id=f"r{i}", prediction="x", confidence=0.99 - i * 0.001, eligible=True)
        for i in range(200)
    ]
    truth200 = {f"r{i}": "x" for i in range(200)}
    truth200["r50"] = "y"
    assert choose_threshold(rows200, truth200, 0.05, 50) == rows200[-1]["confidence"]
    assert threshold_on_dev(rows200, truth200, 0.05, 50) == rows200[-1]["confidence"]


def test_recorded_results_reloadability_and_dual_budget_gates():
    """Verify 5/5 reloadability for instinct_hybrid and valid 1%/5% gates in results/."""
    results_dir = Path(__file__).parent / "results"
    costs = json.loads((results_dir / "costs.json").read_text())
    gates = json.loads((results_dir / "gates.json").read_text())
    assert len(costs) == 20
    for key, entry in costs.items():
        assert entry["reloadable"] is True, f"{key} was not reloadable"
        assert "reload_error" not in entry
        if key.startswith("instinct_hybrid:"):
            assert entry["serialized_header_bytes"] < 50_000
    for seed in PROTOCOL["noise_seeds"]:
        for method in ("simple_classifier", "instinct_lexical", "instinct_hybrid"):
            g = gates[f"{method}:{seed}"]
            if method in ("instinct_lexical", "instinct_hybrid"):
                assert g["threshold"] is not None
            sg = g["standard_budget_gate"]
            assert sg["budget"] == 0.05
            assert sg["threshold"] is not None
            assert sg["qualified"] is True
            assert sg["qualification_upper_bound"] <= 0.05

