"""Independent evidence scorer. Does not import runner, prepare, or Instinct."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom

HERE = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _require(condition: bool, message: str = "Verification failed") -> None:
    if not condition:
        raise ValueError(message)


def exact_upper(errors, count, delta):
    if count == 0 or errors == count:
        return 1.0
    if errors == 0:
        return -math.expm1(math.log(delta)/count)
    return brentq(lambda p: binom.cdf(errors, count, p)-delta, 0, 1, xtol=1e-14)


def rebuild(data_dir, protocol):
    buckets = defaultdict(list)
    for split in ("train", "valid", "test"):
        raw_text = (data_dir/f"{split}.text").read_bytes()
        raw_label = (data_dir/f"{split}.label").read_bytes()
        _require(sha(raw_text) == protocol["source_sha256"][f"{split}.text"], f"Source hash mismatch: {split}.text")
        _require(sha(raw_label) == protocol["source_sha256"][f"{split}.label"], f"Source hash mismatch: {split}.label")
        texts, labels = raw_text.decode().splitlines(), raw_label.decode().splitlines()
        _require(len(texts) == len(labels), f"Text/label length mismatch in {split}")
        for text, truth in zip(texts, labels):
            key = sha(" ".join(text.casefold().split()).encode())
            buckets[key].append(dict(id=key, text=text, truth=truth, source=split))
    clean = [v[0] for v in buckets.values() if len({r["truth"] for r in v}) == 1 and len({r["source"] for r in v}) == 1]
    labels = sorted({r["truth"] for r in clean if r["source"] == "train"})
    _require(len(labels) == 64, "Expected 64 intents")
    cohorts = {"dev": [r for r in clean if r["source"] == "valid"], "test": [r for r in clean if r["source"] == "test"],
               "teach": [], "calibrate": [], "qualify": []}
    t, c = protocol["teaching_per_class"], protocol["trusted_calibration_per_class"]
    for label in labels:
        rows = sorted([r for r in clean if r["source"] == "train" and r["truth"] == label], key=lambda r: r["id"])
        cohorts["teach"] += rows[:t]
        cohorts["calibrate"] += rows[t:t+c]
        cohorts["qualify"] += rows[t+c:]
    for rows in cohorts.values():
        rows.sort(key=lambda r: r["id"])
    ids = [r["id"] for rows in cohorts.values() for r in rows]
    _require(len(ids) == len(set(ids)), "Cohort overlap")
    return cohorts, labels


def noisy_labels(rows, seed, labels, *, per_class):
    result = {r["id"]: r["truth"] for r in rows}
    order = lambda r: sha(f"noise:{seed}:{r['id']}".encode())
    if per_class:
        wrong = [r for label in labels for r in sorted([r for r in rows if r["truth"] == label], key=order)[:round(sum(r["truth"] == label for r in rows)*0.1)]]
    else:
        wrong = sorted(rows, key=order)[:round(len(rows)*0.1)]
    for row in wrong:
        label = row["truth"]
        options = [l for l in labels if l != label and l.split("_")[0] == label.split("_")[0]]
        if not options:
            options = [l for l in labels if l != label]
        result[row["id"]] = options[int(sha(f"wrong:{seed}:{row['id']}".encode()), 16) % len(options)]
    return result


def threshold_on_dev(rows, truths, budget, minimum, dev_delta=0.0025):
    eligible = [r for r in rows if r["eligible"]]
    choices = sorted({r["confidence"] for r in eligible})
    for threshold in choices:
        selected = [r for r in eligible if r["confidence"] >= threshold]
        errors = sum(r["prediction"] != truths[r["id"]] for r in selected)
        if len(selected) >= minimum and ((errors == 0 and budget >= 0) or exact_upper(errors, len(selected), dev_delta) <= budget):
            return threshold
    return None


def metrics(rows, truths, threshold, enabled, fallback=None):
    selected = [r for r in rows if enabled and threshold is not None and r["eligible"] and r["confidence"] >= threshold]
    accepted_ids = {r["id"] for r in selected}
    errors = sum(r["prediction"] != truths[r["id"]] for r in selected)
    output = dict(count=len(rows), predictions=sum(r["prediction"] is not None for r in rows),
                  raw_correct=sum(r["prediction"] == truths[r["id"]] for r in rows),
                  raw_accuracy=sum(r["prediction"] == truths[r["id"]] for r in rows)/len(rows),
                  native_eligible=sum(r["eligible"] for r in rows), accepted=len(selected), local_errors=errors,
                  coverage=len(selected)/len(rows), local_error_rate=errors/len(selected) if selected else None,
                  local_error_upper_95=exact_upper(errors, len(selected), .05) if selected else None,
                  p50_latency_ms=float(np.percentile([r["latency_ms"] for r in rows], 50)),
                  p95_latency_ms=float(np.percentile([r["latency_ms"] for r in rows], 95)))
    if fallback is not None:
        cascade_errors = errors + sum(fallback[r["id"]] != truths[r["id"]] for r in rows if r["id"] not in accepted_ids)
        output.update(teacher_calls=len(rows)-len(selected), cascade_errors=cascade_errors, cascade_error_rate=cascade_errors/len(rows))
    return output


def _compare_methods(protocol, results, masks, budget, fresh_key="fresh"):
    comparisons = {}
    for method in ("instinct_lexical", "instinct_hybrid"):
        differences = np.stack([masks[f"{method}:{s}"]-masks[f"simple_classifier:{s}"] for s in protocol["noise_seeds"]])
        per_seed = differences.mean(axis=1)
        rng = np.random.default_rng(protocol["bootstrap_seed"])
        samples = []
        for _ in range(protocol["bootstrap_replicates"]):
            seeds = rng.integers(0, len(per_seed), len(per_seed))
            requests = rng.integers(0, differences.shape[1], differences.shape[1])
            samples.append(float(differences[seeds][:, requests].mean()))
        low, high = np.percentile(samples, [2.5, 97.5])
        budget_misses = [
            s for s in protocol["noise_seeds"]
            if results[f"{method}:{s}"][fresh_key]["local_error_rate"] is not None
            and results[f"{method}:{s}"][fresh_key]["local_error_rate"] > budget
        ]
        wins = sum(per_seed >= protocol["minimum_fresh_coverage_gain"])
        comparisons[method] = dict(
            mean_coverage_gain=float(per_seed.mean()),
            per_seed_gain=per_seed.tolist(),
            bootstrap_95=[float(low), float(high)],
            seeds_with_5pp_gain=int(wins),
            observed_budget_miss_seeds=budget_misses,
            earns_complexity=bool(wins >= protocol["minimum_winning_seeds"] and low > 0 and not budget_misses),
        )
    return comparisons


def verify(data_dir, output_dir):
    protocol = json.loads((HERE/"protocol.json").read_text())
    manifest = json.loads((output_dir/"manifest.json").read_text())
    expected_files = {"provenance.json", "gates.json", "costs.json", "predictions.json.gz"}
    _require(set(manifest) == expected_files, "Manifest file set mismatch")
    for name, expected in manifest.items():
        _require(sha((output_dir/name).read_bytes()) == expected, f"Evidence hash mismatch: {name}")
    provenance = json.loads((output_dir/"provenance.json").read_text())
    _require(provenance["protocol_sha256"] == sha((HERE/"protocol.json").read_bytes()), "Protocol hash mismatch")
    _require(
        provenance["inputs_sha256"] == sha((data_dir/"inputs.json").read_bytes()) == protocol["prepared_inputs_sha256"],
        "Inputs hash mismatch",
    )
    _require(provenance["runtime_base"] == protocol["runtime_base"], "Runtime base mismatch")
    cohorts, labels = rebuild(data_dir, protocol)
    prepared = json.loads((data_dir/"inputs.json").read_text())["cohorts"]
    for cohort, rows in cohorts.items():
        expected = rows if cohort != "test" else [{k: v for k, v in r.items() if k != "truth"} for r in rows]
        _require(prepared[cohort] == expected, f"Prepared cohort mismatch: {cohort}")
        receipt = provenance["data"]["cohorts"][cohort]
        _require(
            receipt == dict(count=len(rows), ids_sha256=sha("\n".join(r["id"] for r in rows).encode())),
            f"Cohort receipt mismatch: {cohort}",
        )
    truths = {k: {r["id"]: r["truth"] for r in rows} for k, rows in cohorts.items()}
    _require(json.loads((data_dir/"answers.json").read_text()) == truths["test"], "Test answers mismatch")
    predictions = json.loads(gzip.decompress((output_dir/"predictions.json.gz").read_bytes()))
    gates = json.loads((output_dir/"gates.json").read_text())
    costs = json.loads((output_dir/"costs.json").read_text())
    keys = {f"{m}:{s}" for m in protocol["methods"] for s in protocol["noise_seeds"]}
    _require(set(predictions) == set(gates) == set(costs) == keys, "Method/seed key set mismatch")
    delta = protocol["family_failure_probability"]/len(keys)
    dev_delta = protocol.get("dev_selection_delta", delta)
    std_budget = protocol.get("standard_error_budget", 0.05)
    results, masks, std_masks = {}, {}, {}
    for seed in protocol["noise_seeds"]:
        noisy = noisy_labels(cohorts["teach"], seed, labels, per_class=True)
        _require(sum(noisy[k] != v for k, v in truths["teach"].items()) == 128, "Teach noise count mismatch")
        fallback = noisy_labels(cohorts["test"], seed, labels, per_class=False)
        _require(
            sum(fallback[k] != v for k, v in truths["test"].items()) == round(len(cohorts["test"])*.1),
            "Test fallback noise count mismatch",
        )
        for method in protocol["methods"]:
            key = f"{method}:{seed}"
            group = predictions[key]
            _require(set(group) == {"dev", "qualify", "test", "repeat"}, f"Split set mismatch: {key}")
            for split, rows in group.items():
                source = "teach" if split == "repeat" else split
                _require([r["id"] for r in rows] == [r["id"] for r in cohorts[source]], f"IDs differ: {key} {split}")
                for r in rows:
                    _require(r["prediction"] is None or r["prediction"] in labels, f"Invalid prediction: {key}")
                    _require(type(r["eligible"]) is bool, f"Invalid eligible flag: {key}")
                    _require(math.isfinite(r["confidence"]) and 0 <= r["confidence"] <= 1, f"Invalid confidence: {key}")
                    _require(math.isfinite(r["latency_ms"]) and r["latency_ms"] >= 0, f"Invalid latency_ms: {key}")
                    if method == "exact_cache":
                        expected = noisy[r["id"]] if split == "repeat" else None
                        _require(r["prediction"] == expected, f"Cache prediction mismatch: {key}")
                        _require(r["eligible"] == (expected is not None), f"Cache eligibility mismatch: {key}")
                    elif method == "simple_classifier":
                        _require(r["eligible"] and r["prediction"] is not None, f"Classifier eligibility mismatch: {key}")
            threshold = threshold_on_dev(group["dev"], truths["dev"], protocol["local_error_budget"], protocol["minimum_dev_accepts"], dev_delta)
            gate = gates[key]
            _require(gate["threshold"] == threshold, f"Threshold mismatch: {key}")
            qual = metrics(group["qualify"], truths["qualify"], threshold, True)
            upper = exact_upper(qual["local_errors"], qual["accepted"], delta)
            _require(gate["qualification_count"] == qual["accepted"], f"Qualification count mismatch: {key}")
            _require(gate["qualification_errors"] == qual["local_errors"], f"Qualification errors mismatch: {key}")
            _require(math.isclose(gate["qualification_upper_bound"], upper, abs_tol=1e-10), f"Upper bound mismatch: {key}")
            _require(gate["delta"] == delta, f"Delta mismatch: {key}")
            qualified = threshold is not None and upper <= protocol["local_error_budget"]
            _require(gate["qualified"] == qualified, f"Qualification mismatch: {key}")
            fresh = metrics(group["test"], truths["test"], threshold, qualified, fallback)
            candidates = metrics(group["test"], truths["test"], threshold, True)
            repeat = metrics(group["repeat"], truths["teach"], threshold, qualified)
            raw_repeat = metrics(group["repeat"], truths["teach"], 0, True)

            std_threshold = threshold_on_dev(group["dev"], truths["dev"], std_budget, protocol["minimum_dev_accepts"], dev_delta)
            std_qual = metrics(group["qualify"], truths["qualify"], std_threshold, True)
            std_upper = exact_upper(std_qual["local_errors"], std_qual["accepted"], delta)
            std_qualified = std_threshold is not None and std_upper <= std_budget
            if "standard_budget_gate" in gate:
                sg = gate["standard_budget_gate"]
                _require(sg["budget"] == std_budget, f"Standard budget mismatch: {key}")
                _require(sg["threshold"] == std_threshold, f"Standard threshold mismatch: {key}")
                _require(sg["qualification_count"] == std_qual["accepted"], f"Standard qualification count mismatch: {key}")
                _require(sg["qualification_errors"] == std_qual["local_errors"], f"Standard qualification errors mismatch: {key}")
                _require(math.isclose(sg["qualification_upper_bound"], std_upper, abs_tol=1e-10), f"Standard upper bound mismatch: {key}")
                _require(sg["delta"] == delta, f"Standard delta mismatch: {key}")
                _require(sg["qualified"] == std_qualified, f"Standard qualification mismatch: {key}")
            fresh_std = metrics(group["test"], truths["test"], std_threshold, std_qualified, fallback)
            candidates_std = metrics(group["test"], truths["test"], std_threshold, True)
            repeat_std = metrics(group["repeat"], truths["teach"], std_threshold, std_qualified)

            results[key] = dict(gate=gate, fresh=fresh, unqualified_candidate=candidates, repeat_diagnostic=repeat,
                                fresh_standard_budget=fresh_std, unqualified_candidate_standard_budget=candidates_std,
                                repeat_standard_budget=repeat_std,
                                native_repeat_diagnostic=raw_repeat, costs=costs[key],
                                modeled_mean_cascade_ms={str(ms): float(np.mean([r["latency_ms"] for r in group["test"]]))+(1-fresh["coverage"])*ms for ms in protocol["teacher_latency_scenarios_ms"]},
                                modeled_mean_cascade_ms_standard_budget={str(ms): float(np.mean([r["latency_ms"] for r in group["test"]]))+(1-fresh_std["coverage"])*ms for ms in protocol["teacher_latency_scenarios_ms"]})
            masks[key] = np.array([qualified and threshold is not None and r["eligible"] and r["confidence"] >= threshold for r in group["test"]], dtype=float)
            std_masks[key] = np.array([std_qualified and std_threshold is not None and r["eligible"] and r["confidence"] >= std_threshold for r in group["test"]], dtype=float)
    comparisons = _compare_methods(protocol, results, masks, protocol["local_error_budget"], "fresh")
    standard_budget_comparisons = _compare_methods(protocol, results, std_masks, std_budget, "fresh_standard_budget")
    return dict(evidence_verified=True, runtime_base=protocol["runtime_base"], protocol_revision=provenance["revision"],
                dataset=provenance["data"], environment=provenance["environment"], results=results,
                comparisons=comparisons, standard_budget_comparisons=standard_budget_comparisons,
                assumptions=["public in-scope simulation, not production/timestamped traffic", "exchangeability, not drift guarantee",
                             "1% strict and 5% standard locally accepted error budgets; fallback can still be wrong", "repeat diagnostics do not qualify fresh gates"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = verify(args.data_dir, args.output_dir)
    (args.output_dir/"verified_report.json").write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n")
    print("=== 1% Strict Budget Comparisons ===")
    print(json.dumps(report["comparisons"], indent=2))
    print("=== 5% Standard Budget Comparisons ===")
    print(json.dumps(report["standard_budget_comparisons"], indent=2))
    for key, value in report["results"].items():
        f1 = value["fresh"]
        f5 = value["fresh_standard_budget"]
        print(
            f"{key}: raw_accuracy={f1['raw_accuracy']:.2%} "
            f"1%_cov={f1['coverage']:.2%} ({f1['local_errors']}/{f1['accepted']}) "
            f"5%_cov={f5['coverage']:.2%} ({f5['local_errors']}/{f5['accepted']})"
        )


if __name__ == "__main__":
    main()
