#!/usr/bin/env python3
"""Run strictly causal prequential evaluation against an offline system1.cassette.v1 JSONL fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.prequential.cassette import (  # noqa: E402
    CassetteTeacher,
    block_network,
    load_cassette_events,
)
from system1 import (  # noqa: E402
    AdaptiveConformalTracker,
    BoundedReviewFallback,
    ChoiceField,
    DecisionSchema,
    HybridProjector,
    SystemOneCompiler,
    SystemOneEngine,
)


DEFAULT_CASSETTE = (
    Path(__file__).resolve().parent / "fixtures" / "support_stream_v1.cassette.jsonl"
)


def evaluate_cassette_stream(cassette_path: Path) -> Dict[str, Any]:
    """Run test-then-train evaluation with network sockets strictly blocked."""
    block_network()
    header, events = load_cassette_events(cassette_path)
    field_name = str(header["field_name"])
    options = list(header["options"])
    schema = DecisionSchema(fields={field_name: ChoiceField(options=options)})

    teacher = CassetteTeacher(
        cassette_path,
        schema,
        provider=str(header.get("provider", "recorded_sim")),
        model=str(header.get("model", "gemini-2.5-flash-sim-v1")),
    )

    warm_rows = [e for e in events if e["regime"] == "warm"]
    stream_rows = [e for e in events if e["regime"] != "warm"]
    warm_pairs = [
        (e["prompt_text"], e["eval_audit"]["true_label"]) for e in warm_rows
    ]

    by_label: Dict[str, List[tuple[str, str]]] = {}
    for p_w, y_w in warm_pairs:
        by_label.setdefault(y_w, []).append((p_w, y_w))
    train_pairs = []
    cal_pairs = []
    for y_w, grp in by_label.items():
        cut = max(1, (len(grp) * 2) // 3)
        train_pairs.extend(grp[:cut])
        cal_pairs.extend(grp[cut:] or grp)

    proj = HybridProjector.fit(
        [p for p, _ in train_pairs],
        labels=[y for _, y in train_pairs],
        backend="numpy",
    )
    comp = SystemOneCompiler(
        schema,
        projector=proj,
        dimension=proj.dimension,
        backend="numpy",
        mondrian=False,
    )
    model = comp.compile(
        exemplars={field_name: train_pairs},
        calibration_exemplars={field_name: cal_pairs},
        augment=False,
    )
    aci = AdaptiveConformalTracker(alpha_target=0.15, gamma=0.02, escalation_learning_rate_scale=0.25)
    eng = SystemOneEngine(
        schema,
        model=model,
        strict_mode=True,
        use_cache=True,
        aci_tracker=aci,
    )
    fb = BoundedReviewFallback(
        eng,
        review_fn=teacher,
        max_reviews=len(stream_rows),
        auto_online_update=True,
        aci_tracker=aci,
    )

    in_scope_total = 0
    in_scope_local = 0
    in_scope_local_err = 0
    oos_total = 0
    oos_false_accept = 0
    latencies_ms: List[float] = []

    for ev in stream_rows:
        prompt = ev["prompt_text"]
        audit = ev["eval_audit"]
        is_oos = bool(audit.get("is_oos", False))
        truth = audit.get("true_label")

        t0 = time.perf_counter()
        res = fb.decide(prompt)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

        local_accept = not res.used_fallback
        pred = res["values"][field_name]
        if is_oos:
            oos_total += 1
            if local_accept:
                oos_false_accept += 1
        else:
            in_scope_total += 1
            if local_accept:
                in_scope_local += 1
                is_err = pred != truth
                if is_err:
                    in_scope_local_err += 1
                margin_val = None
                if isinstance(res.get("margins"), dict):
                    margin_val = res["margins"].get(field_name)
                aci.update_accepted(is_err, margin=margin_val)

    return {
        "cassette_id": header.get("cassette_id", cassette_path.name),
        "warm_examples": len(warm_rows),
        "stream_events": len(stream_rows),
        "in_scope_total": in_scope_total,
        "in_scope_local_accepted": in_scope_local,
        "in_scope_coverage": round(in_scope_local / max(1, in_scope_total), 4),
        "conditional_error": round(in_scope_local_err / max(1, in_scope_local), 4) if in_scope_local else 0.0,
        "oos_total": oos_total,
        "oos_false_accept": oos_false_accept,
        "oos_false_accept_rate": round(oos_false_accept / max(1, oos_total), 4) if oos_total else 0.0,
        "teacher_calls": teacher.call_count,
        "teacher_reduction": round(1.0 - (teacher.call_count / max(1, len(stream_rows))), 4),
        "mean_step_ms": round(sum(latencies_ms) / max(1, len(latencies_ms)), 3),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cassette", type=Path, default=DEFAULT_CASSETTE)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    report = evaluate_cassette_stream(args.cassette)
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"Cassette {report['cassette_id']}: "
            f"coverage={report['in_scope_coverage']:.1%}, "
            f"cond_error={report['conditional_error']:.1%}, "
            f"oos_false_accept={report['oos_false_accept']}/{report['oos_total']}, "
            f"teacher_calls={report['teacher_calls']}/{report['stream_events']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
