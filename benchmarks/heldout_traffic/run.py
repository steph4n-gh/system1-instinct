"""Fit on noisy lessons and freeze risk gates; never open final answers."""
from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import pickle
import platform
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.stats import beta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from threadpoolctl import threadpool_limits

from prepare import HERE, PROTOCOL, digest, teacher_labels
from system1.compiler import CompiledSystemOneModel, SystemOneCompiler
from system1.core import TfidfProjector
from system1.core.embeddings import HybridProjector
from system1.core.schema import ChoiceField, DecisionSchema
from system1.engine import SystemOneEngine


def upper_bound(errors: int, count: int, delta: float) -> float:
    if count == 0 or errors == count:
        return 1.0
    return float(beta.ppf(1-delta, errors+1, count-errors))


def choose_threshold(predictions: list[dict], truths: dict[str, str], budget: float, minimum: int) -> float | None:
    rows = sorted((r for r in predictions if r["eligible"]), key=lambda r: -r["confidence"])
    count = errors = 0
    threshold = None
    for i, row in enumerate(rows):
        count += 1
        errors += row["prediction"] != truths[row["id"]]
        if i+1 < len(rows) and rows[i+1]["confidence"] == row["confidence"]:
            continue
        if count >= minimum and errors/count <= budget:
            threshold = row["confidence"]
    return threshold


def predict(method: str, model, rows: list[dict]) -> list[dict]:
    def one(text):
        if method == "exact_cache":
            value = model.get(text)
            return value, 1.0 if value is not None else 0.0, value is not None
        if method == "simple_classifier":
            p = model.predict_proba([text])[0]
            k = int(np.argmax(p))
            return str(model.classes_[k]), float(p[k]), True
        result = model.decide(text, alpha=PROTOCOL["native_conformal_alpha"], record_receipt=False)
        return result.values["intent"], float(result.confidences["intent"]), not (result.is_ambiguous or result.escalated_fields)
    # Warmup is excluded, every measured inference is one request (no batching).
    if rows:
        one(rows[0]["text"])
    output = []
    for row in rows:
        start = time.perf_counter_ns()
        label, confidence, eligible = one(row["text"])
        elapsed = (time.perf_counter_ns()-start)/1e6
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid confidence")
        output.append(dict(id=row["id"], prediction=label, confidence=confidence,
                           eligible=bool(eligible), latency_ms=elapsed))
    return output


def fit(method: str, teaching: list[dict], calibration: list[dict], labels: dict, classes: list[str]):
    texts = [r["text"] for r in teaching]
    targets = [labels[r["id"]] for r in teaching]
    if method == "exact_cache":
        return dict(zip(texts, targets))
    if method == "simple_classifier":
        cfg = PROTOCOL[method]
        features = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=cfg["word_features"], sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=cfg["char_features"], sublinear_tf=True)),
        ])
        model = Pipeline([("features", features), ("classifier", LogisticRegression(C=cfg["C"], max_iter=cfg["max_iter"], solver="lbfgs"))])
        model.fit(texts, targets)
        return model
    cfg = PROTOCOL[method].copy()
    regularization = cfg.pop("regularization")
    hierarchical = cfg.pop("hierarchical")
    if method == "instinct_lexical":
        projector = TfidfProjector.fit(texts, labels=targets, selection="chi2", char_ngrams=(3, 5), **cfg)
    else:
        projector = HybridProjector.fit(texts, labels=targets, backend="numpy", **cfg)
    schema = DecisionSchema(fields={"intent": ChoiceField(options=classes)})
    compiler = SystemOneCompiler(schema, projector=projector, dimension=projector.dimension,
                                 backend="numpy", regularization=regularization,
                                 hierarchical=hierarchical, mondrian=True)
    compiled = compiler.compile(exemplars={"intent": list(zip(texts, targets))},
                                calibration_exemplars={"intent": [(r["text"], r["truth"]) for r in calibration]}, augment=False)
    return SystemOneEngine(schema, model=compiled, strict_mode=True, use_cache=False)


def save_model(method, model, path):
    if method.startswith("instinct_"):
        model.model.save(path, include_covariance=False)
    elif method == "exact_cache":
        path.write_text(json.dumps(model, sort_keys=True))
    else:
        path.write_bytes(pickle.dumps(model, protocol=5))


def load_model(method, path):
    if method.startswith("instinct_"):
        compiled = CompiledSystemOneModel.load(path)
        return SystemOneEngine(compiled.schema, model=compiled, strict_mode=True, use_cache=False)
    if method == "exact_cache":
        return json.loads(path.read_text())
    # Only our own model from this process's private TemporaryDirectory.
    return pickle.loads(path.read_bytes())


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if digest((args.data_dir / "inputs.json").read_bytes()) != PROTOCOL["prepared_inputs_sha256"]:
        raise ValueError("Prepared inputs do not match frozen protocol")
    prepared = json.loads((args.data_dir / "inputs.json").read_text())
    cohorts, receipt = prepared["cohorts"], prepared["receipt"]
    if any("truth" in r for r in cohorts["test"]):
        raise ValueError("Final answer key leaked into inputs")
    expected_counts = {k: len(v) for k, v in cohorts.items()}
    if any(expected_counts[k] != receipt["cohorts"][k]["count"] for k in cohorts):
        raise ValueError("Prepared count mismatch")
    for k, rows in cohorts.items():
        if digest("\n".join(r["id"] for r in rows).encode()) != receipt["cohorts"][k]["ids_sha256"]:
            raise ValueError("Prepared IDs mismatch")
    ids = [r["id"] for rows in cohorts.values() for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Cohort overlap")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE).decode().strip()
    dirty = subprocess.check_output(["git", "diff", "HEAD", "--", "src", "benchmarks/heldout_traffic"], cwd=HERE)
    if dirty:
        raise ValueError("Commit protocol, runner and runtime before running")
    environment = dict(python=sys.version, platform=platform.platform(), machine=platform.machine(),
                       numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)
    write_json(args.output_dir / "provenance.json", dict(revision=revision, runtime_base=PROTOCOL["runtime_base"],
        protocol_sha256=digest((HERE/"protocol.json").read_bytes()), inputs_sha256=digest((args.data_dir/"inputs.json").read_bytes()),
        data=receipt, environment=environment))
    def blocked(*args, **kwargs):
        raise RuntimeError("Network is prohibited during fitting/evaluation")
    socket.socket.connect = blocked
    socket.create_connection = blocked
    delta = PROTOCOL["family_failure_probability"]/(len(PROTOCOL["methods"])*len(PROTOCOL["noise_seeds"]))
    records, gates, costs = {}, {}, {}
    with threadpool_limits(limits=1), tempfile.TemporaryDirectory(prefix="heldout-models-") as temp:
        model_dir = Path(temp)
        # All thresholds are fixed and committed to the receipt before final inference.
        for seed in PROTOCOL["noise_seeds"]:
            noisy = teacher_labels(cohorts["teach"], seed, receipt["classes"])
            for method in PROTOCOL["methods"]:
                key = f"{method}:{seed}"
                start = time.perf_counter()
                model = fit(method, cohorts["teach"], cohorts["calibrate"], noisy, receipt["classes"])
                fit_s = time.perf_counter()-start
                dev = predict(method, model, cohorts["dev"])
                qual = predict(method, model, cohorts["qualify"])
                dev_truth = {r["id"]: r["truth"] for r in cohorts["dev"]}
                qual_truth = {r["id"]: r["truth"] for r in cohorts["qualify"]}
                threshold = choose_threshold(dev, dev_truth, PROTOCOL["local_error_budget"], PROTOCOL["minimum_dev_accepts"])
                accepted = [r for r in qual if r["eligible"] and threshold is not None and r["confidence"] >= threshold]
                errors = sum(r["prediction"] != qual_truth[r["id"]] for r in accepted)
                upper = upper_bound(errors, len(accepted), delta)
                gates[key] = dict(threshold=threshold, qualification_count=len(accepted), qualification_errors=errors,
                                  qualification_upper_bound=upper, delta=delta,
                                  qualified=threshold is not None and upper <= PROTOCOL["local_error_budget"])
                path = model_dir / key
                save_model(method, model, path)
                costs[key] = dict(fit_seconds=fit_s, serialized_bytes=path.stat().st_size)
                records[key] = dict(dev=dev, qualify=qual)
                # Replay parity checks save/reload independently of final labels.
                replay = predict(method, load_model(method, path), cohorts["dev"][:10])
                for a, b in zip(dev[:10], replay):
                    if a["prediction"] != b["prediction"] or a["eligible"] != b["eligible"] or not math.isclose(a["confidence"], b["confidence"], abs_tol=1e-6):
                        raise ValueError(f"Serialization changed {key}")
                print(f"FROZEN {key}: fit={fit_s:.2f}s qualify={len(accepted)} errors={errors} upper={upper:.4%} approved={gates[key]['qualified']}", flush=True)
                del model
                gc.collect()
        write_json(args.output_dir/"gates.json", gates)
        gate_hash = digest((args.output_dir/"gates.json").read_bytes())
        for seed in PROTOCOL["noise_seeds"]:
            for method in PROTOCOL["methods"]:
                key = f"{method}:{seed}"
                model = load_model(method, model_dir/key)
                records[key]["test"] = predict(method, model, cohorts["test"])
                records[key]["repeat"] = predict(method, model, cohorts["teach"])
                print(f"PREDICTED {key}: fresh={len(cohorts['test'])} repeat={len(cohorts['teach'])}; final answers unopened", flush=True)
                del model
                gc.collect()
        if digest((args.output_dir/"gates.json").read_bytes()) != gate_hash:
            raise ValueError("Gate changed during final inference")
    write_json(args.output_dir/"costs.json", costs)
    with gzip.GzipFile(filename=str(args.output_dir/"predictions.json.gz"), mode="wb", mtime=0) as stream:
        stream.write(json.dumps(records, sort_keys=True, allow_nan=False).encode())
    files = ["provenance.json", "gates.json", "costs.json", "predictions.json.gz"]
    write_json(args.output_dir/"manifest.json", {name: digest((args.output_dir/name).read_bytes()) for name in files})
    print("Prediction evidence complete. Run the independent verifier to open final labels and score.", flush=True)


if __name__ == "__main__":
    main()
