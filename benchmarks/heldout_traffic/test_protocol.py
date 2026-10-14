"""Focused experiment controls; run explicitly with benchmark dependencies."""
import gzip
import json
import math
import os
import shutil
from pathlib import Path

import pytest

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
    data = tmp_path/"data"
    evidence = tmp_path/"evidence"
    shutil.copytree(location, data)
    shutil.copytree(Path(__file__).parent/"results", evidence)
    manifest_path = evidence/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if attack == "hash":
        with (evidence/"costs.json").open("ab") as stream:
            stream.write(b" ")
    elif attack == "gate":
        path = evidence/"gates.json"
        gates = json.loads(path.read_text())
        gates["instinct_hybrid:17"]["qualified"] = True
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
    with pytest.raises(AssertionError):
        verify(data, evidence)
