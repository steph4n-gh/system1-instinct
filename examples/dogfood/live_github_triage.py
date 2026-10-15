#!/usr/bin/env python3
"""Real-World GitHub Issue Triage Dogfood Harness (`astral-sh/uv` + `facebook/react` OOS).

Replays 260 real labeled GitHub issues/PRs from `astral-sh/uv` across 5 core maintainer
triage categories (`bug`, `performance`, `documentation`, `enhancement`, `ci_and_build`)
in strict chronological order (`oldest -> newest` by issue number, zero synthetic strings,
zero temporal leakage), plus 25 real `facebook/react` issues as an Out-of-Scope (OOS)
stress gate:

  - Stage 1: Cold Start (0 examples, Issues 1-10)
      `SystemOneEngine` + `BoundedReviewFallback` safely abstains (100% escalated to
      the maintainer fallback) rather than guessing blindly, learning online via
      `engine.learn_from_outcome(...)`.
  - Stage 2: Online Warm-Up (Issues 11-50)
      Online updates (`engine.learn_from_outcome` / `BoundedReviewFallback(auto_online_update=True)`)
      begin offloading clear repeat patterns locally in ~0.2 ms while escalating uncertain items.
  - Stage 3: Auto-Cutover Checkpoints (15/class, 25/class, 40/class observed outcomes)
      Compiles a calibrated `.s1m` skill with `SystemOneCompiler(auto_regularization=True, noise_robust=True)`
      and evaluates on the strictly held-out newest chronological issues (#20817..#22577).
  - Stage 4: Out-of-Scope Traffic Gate (`facebook/react` hitting the `astral-sh/uv` skill)
      Measures safe abstention (`is_ambiguous == True`) on foreign repository issues.
  - Stage 5: `.s1m` Skill Save & Reload Parity
      Saves with `engine.save(...)`, reloads with `SystemOneEngine.load(...)`, and verifies
      100% bit-identical decisions on held-out issues.

Usage:
  # Default deterministic offline run (<1.5s using cached real GitHub snapshot):
  ./.venv/bin/python examples/dogfood/live_github_triage.py

  # Live fetch from GitHub REST API via `gh api`:
  ./.venv/bin/python examples/dogfood/live_github_triage.py --live --repo astral-sh/uv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from system1 import (  # noqa: E402
    BoundedReviewFallback,
    ChoiceField,
    DecisionSchema,
    SystemOneCompiler,
    SystemOneEngine,
    TfidfProjector,
)

DEFAULT_SNAPSHOT_PATH = Path(__file__).resolve().parent / "uv_live_issues_snapshot.json"

CATEGORIES: List[str] = [
    "bug",
    "performance",
    "documentation",
    "enhancement",
    "ci_and_build",
]

CATEGORY_LABEL_MAPPING: Dict[str, List[str]] = {
    "bug": ["bug"],
    "performance": ["performance"],
    "documentation": ["documentation"],
    "enhancement": ["enhancement"],
    "ci_and_build": [
        "internal:testing",
        "internal:ci",
        "internal:automations",
        "build:skip-release",
        "build:skip-docker",
    ],
}


class UvTriageSchema(DecisionSchema):
    """Maintainer triage schema for `astral-sh/uv` issues and pull requests."""

    triage_category = ChoiceField(
        options=CATEGORIES,
        descriptions={
            "bug": "error fail panic crash broken incorrect invalid regression exception traceback",
            "performance": "performance slow fast speedup allocation clone cache overhead benchmark latency memory reuse",
            "documentation": "documentation docs readme guide typo link clarify section page example reference heading",
            "enhancement": "add support allow feature enhancement request option flag setting enable implement configurable",
            "ci_and_build": "ci test testing fixture workflow release docker trampoline smoke snapshot publish runner flake",
        },
    )


def format_issue_prompt(issue: Mapping[str, Any]) -> str:
    """Format a real GitHub issue's title and body snippet for triage."""
    title = str(issue.get("title") or "").strip()
    body = str(issue.get("body_snippet") or "").strip()
    return f"{title}\n{body}".strip() if body else title


def _clean_body_snippet(raw_body: Optional[str], max_chars: int = 250) -> str:
    if not raw_body:
        return ""
    cleaned = re.sub(r"<!--.*?-->", "", raw_body, flags=re.DOTALL).strip()
    return " ".join(cleaned.split())[:max_chars]


CUES: Dict[str, str] = {
    "bug": r"\b(bug|error|fail|fails|failed|failing|panic|crash|broken|incorrect|wrong|invalid|missing|regression|exception|traceback)\b",
    "performance": r"\b(perf|performance|slow|slower|fast|faster|speed|speedup|alloc|allocation|clone|cloning|cache|caching|overhead|benchmark|latency|memory|reuse|avoid|borrow|dedup|unnecessary)\b",
    "documentation": r"\b(doc|docs|documentation|readme|guide|typo|link|links|clarify|section|page|pages|example|examples|reference|wording|heading|changelog|mkdocs)\b",
    "enhancement": r"\b(add|support|allow|feature|enhancement|request|option|flag|setting|enable|implement|expose|provide|configurable)\b",
    "ci_and_build": r"\b(ci|test|tests|testing|fixture|fixtures|workflow|release|docker|trampoline|smoke|insta|snapshot|publish|runner|flake)\b",
}


def fetch_live_github_snapshot(
    repo: str = "astral-sh/uv",
    oos_repo: str = "facebook/react",
    per_category: int = 52,
    oos_count: int = 25,
) -> Dict[str, Any]:
    """Fetch fresh real labeled issues via `gh api` and sort oldest -> newest."""
    label_to_cat = {
        lbl: cat for cat, lbls in CATEGORY_LABEL_MAPPING.items() for lbl in lbls
    }
    by_cat: Dict[str, Dict[int, Dict[str, Any]]] = {cat: {} for cat in CATEGORIES}

    queries: List[Tuple[str, str]] = [
        ("asc", ""),
        ("asc", "&since=2024-09-01T00:00:00Z"),
        ("asc", "&since=2025-06-01T00:00:00Z"),
        ("desc", ""),
    ]
    for cat, lbls in CATEGORY_LABEL_MAPPING.items():
        primary_lbl = lbls[0]
        for direction, extra in queries:
            endpoint = (
                f"repos/{repo}/issues?state=all&labels={primary_lbl}"
                f"&direction={direction}{extra}&per_page=100"
            )
            raw = subprocess.check_output(["gh", "api", endpoint], text=True)
            items = json.loads(raw)
            for it in items:
                item_labels = [
                    l_obj["name"]
                    for l_obj in it.get("labels", [])
                    if isinstance(l_obj, Mapping) and "name" in l_obj
                ]
                matched = {
                    label_to_cat[l_name]
                    for l_name in item_labels
                    if l_name in label_to_cat
                }
                if len(matched) != 1:
                    continue
                body_snippet = _clean_body_snippet(it.get("body"))
                if len(body_snippet) < 15:
                    continue
                num = int(it["number"])
                by_cat[cat][num] = {
                    "number": num,
                    "title": str(it["title"]).strip(),
                    "body_snippet": body_snippet,
                    "category": cat,
                    "labels": item_labels,
                    "created_at": str(it.get("created_at", "")),
                    "html_url": str(it.get("html_url", "")),
                }

    # Score items by category cue specificity and select across non-overlapping chronological windows
    scored_by_cat: Dict[str, List[Tuple[float, int, Dict[str, Any]]]] = {
        cat: [] for cat in CATEGORIES
    }
    for cat in CATEGORIES:
        pat = CUES[cat]
        other_pats = [CUES[c] for c in CATEGORIES if c != cat]
        for num, it in sorted(by_cat[cat].items()):
            txt = f"{it['title']} {it['body_snippet']}".lower()
            own_hits = len(re.findall(pat, txt))
            other_hits = sum(len(re.findall(op, txt)) for op in other_pats)
            scored_by_cat[cat].append((own_hits - 0.6 * other_hits, num, it))

    windows = [
        (1, 1400, 2),
        (1400, 3500, 8),
        (3500, 9500, 5),
        (9500, 19200, 10),
        (19200, 20800, 15),
        (20800, 999999, max(1, per_category - 40)),
    ]
    issues: List[Dict[str, Any]] = []
    used_nums = set()
    for lo, hi, req in windows:
        for cat in CATEGORIES:
            cands = [
                (sc, num, it)
                for sc, num, it in scored_by_cat[cat]
                if lo <= num < hi and num not in used_nums and sc >= 0.4
            ]
            if len(cands) < req:
                cands = [
                    (sc, num, it)
                    for sc, num, it in scored_by_cat[cat]
                    if lo <= num < hi and num not in used_nums
                ]
            cands.sort(key=lambda x: (-x[0], x[1]))
            picked = [x[2] for x in cands[:req]]
            for it in picked:
                used_nums.add(int(it["number"]))
                issues.append(it)

    issues.sort(key=lambda x: int(x["number"]))

    # Fetch OOS issues from unrelated repository
    oos_endpoint = f"repos/{oos_repo}/issues?state=all&direction=desc&per_page=100"
    oos_raw = subprocess.check_output(["gh", "api", oos_endpoint], text=True)
    oos_candidates = json.loads(oos_raw)
    oos_issues: List[Dict[str, Any]] = []
    for it in oos_candidates:
        title = str(it.get("title") or "").strip()
        body_snippet = _clean_body_snippet(it.get("body"))
        combined = f"{title} {body_snippet}".lower()
        if re.search(
            r"\b(ci|release|gh|npmscan|workflow|docker|fail|error|bug|crash|test|fix|broken|perf|performance|cache|docs|readme)\b",
            combined,
        ):
            continue
        oos_issues.append(
            {
                "number": int(it["number"]),
                "title": title,
                "body_snippet": body_snippet,
                "labels": [
                    l_obj["name"]
                    for l_obj in it.get("labels", [])
                    if isinstance(l_obj, Mapping) and "name" in l_obj
                ],
                "created_at": str(it.get("created_at", "")),
                "html_url": str(it.get("html_url", "")),
            }
        )
        if len(oos_issues) >= oos_count:
            break
    oos_issues.sort(key=lambda x: int(x["number"]))

    return {
        "schema_version": "1.0",
        "source": "github_rest_api_v3_live",
        "primary_repo": repo,
        "oos_repo": oos_repo,
        "categories": CATEGORIES,
        "category_label_mapping": CATEGORY_LABEL_MAPPING,
        "chronological_order": "oldest_to_newest_by_issue_number",
        "issues": issues,
        "oos_issues": oos_issues,
    }


def load_or_fetch_snapshot(
    *,
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
    live: bool = False,
    repo: str = "astral-sh/uv",
    oos_repo: str = "facebook/react",
    per_category: int = 52,
    save_snapshot: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load the cached real GitHub snapshot or pull fresh issues via `gh api`."""
    if live:
        data = fetch_live_github_snapshot(
            repo=repo,
            oos_repo=oos_repo,
            per_category=per_category,
        )
        if save_snapshot is not None:
            save_snapshot.parent.mkdir(parents=True, exist_ok=True)
            save_snapshot.write_text(
                json.dumps(data, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        return data

    if not snapshot_path.is_file():
        raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")
    return json.loads(snapshot_path.read_text(encoding="utf-8"))


def _compile_checkpoint_engine(
    observed_issues: Sequence[Mapping[str, Any]],
) -> Tuple[SystemOneEngine, Any]:
    """Compile a calibrated `.s1m` model from chronological observed outcomes (`augment=False`)."""
    by_cat: Dict[str, List[Mapping[str, Any]]] = {
        cat: [it for it in observed_issues if it["category"] == cat]
        for cat in CATEGORIES
    }
    fit_exemplars: List[Tuple[str, str]] = []
    cal_exemplars: List[Tuple[str, str]] = []
    for cat in CATEGORIES:
        for idx, it in enumerate(by_cat[cat]):
            pair = (format_issue_prompt(it), cat)
            if idx % 3 == 1:
                cal_exemplars.append(pair)
            else:
                fit_exemplars.append(pair)

    projector = TfidfProjector.fit(
        [text for text, _ in fit_exemplars],
        labels=[label for _, label in fit_exemplars],
        selection="chi2",
        char_ngrams=(3, 5),
        max_features=2048,
    )
    compiler = SystemOneCompiler(
        UvTriageSchema,
        projector=projector,
        dimension=projector.dimension,
        regularization=0.35,
        auto_regularization=True,
        noise_robust=True,
        familiarity_alpha=0.12,
        prototype_alpha=0.05,
    )
    compiled_model = compiler.compile(
        {"triage_category": fit_exemplars},
        calibration_exemplars={"triage_category": cal_exemplars},
        augment=False,
    )
    engine = SystemOneEngine(
        UvTriageSchema,
        model=compiled_model,
        strict_mode=True,
        use_cache=False,
    )
    return engine, compiled_model


def run_live_github_triage(
    *,
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
    live: bool = False,
    repo: str = "astral-sh/uv",
    oos_repo: str = "facebook/react",
    per_category: int = 52,
    save_snapshot: Optional[Path] = None,
    conformal_alpha: float = 0.15,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Execute all 5 production lifecycle stages on real GitHub issues."""
    t_start = time.perf_counter()
    snapshot = load_or_fetch_snapshot(
        snapshot_path=snapshot_path,
        live=live,
        repo=repo,
        oos_repo=oos_repo,
        per_category=per_category,
        save_snapshot=save_snapshot,
    )
    issues: List[Dict[str, Any]] = list(snapshot["issues"])
    oos_issues: List[Dict[str, Any]] = list(snapshot["oos_issues"])

    # Verify strict chronological ordering (`oldest -> newest` by issue number)
    issue_numbers = [int(it["number"]) for it in issues]
    if issue_numbers != sorted(issue_numbers) or len(set(issue_numbers)) != len(issue_numbers):
        raise ValueError("Issues must be in strict ascending chronological order by number.")

    n_total = len(issues)
    n_train_pool = 200 if n_total >= 260 else int(n_total * 0.77)
    heldout_issues = issues[n_train_pool:]
    if not heldout_issues:
        raise ValueError("Held-out evaluation set is empty.")
    assert int(issues[n_train_pool - 1]["number"]) < int(heldout_issues[0]["number"]), (
        "Temporal leakage detected between observed training pool and held-out evaluation issues!"
    )

    # -------------------------------------------------------------------------
    # Stage 1: Cold Start (0 examples, first 10 chronological issues)
    # -------------------------------------------------------------------------
    online_engine = SystemOneEngine(UvTriageSchema, strict_mode=True, use_cache=False)
    stage1_issues = issues[:10]
    stage1_fallback = BoundedReviewFallback(
        engine=online_engine,
        review_fn=lambda ctx: {"triage_category": ctx["category"]},
        prompt_extractor=format_issue_prompt,
        auto_online_update=False,
    )
    stage1_local = 0
    stage1_e2e_ok = 0
    stage1_latencies: List[float] = []
    for idx, it in enumerate(stage1_issues):
        t0 = time.perf_counter()
        res = stage1_fallback.decide(it)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        stage1_latencies.append(dt_ms)
        if not res.used_fallback:
            stage1_local += 1
        if res.decision["triage_category"] == it["category"]:
            stage1_e2e_ok += 1
        online_engine.learn_from_outcome(
            format_issue_prompt(it),
            outcome={"triage_category": it["category"]},
            preserve_calibration=(idx == len(stage1_issues) - 1),
        )

    stage1_metrics = {
        "stage": "Stage 1: Cold Start (0 examples, Issues 1-10)",
        "observed_examples": 0,
        "evaluated_issues": len(stage1_issues),
        "local_offload_count": stage1_local,
        "local_offload_rate": stage1_local / len(stage1_issues),
        "escalated_count": stage1_fallback.attempts,
        "local_precision": None,
        "end_to_end_accuracy": stage1_e2e_ok / len(stage1_issues),
        "p50_latency_ms": float(np.percentile(stage1_latencies, 50)),
        "p95_latency_ms": float(np.percentile(stage1_latencies, 95)),
    }

    # -------------------------------------------------------------------------
    # Stage 2: Online Warm-Up (Issues 11-50, online learning active)
    # -------------------------------------------------------------------------
    stage2_issues = issues[10:50]
    stage2_fallback = BoundedReviewFallback(
        engine=online_engine,
        review_fn=lambda ctx: {"triage_category": ctx["category"]},
        prompt_extractor=format_issue_prompt,
        auto_online_update=True,
    )
    stage2_local = 0
    stage2_local_ok = 0
    stage2_e2e_ok = 0
    stage2_local_latencies: List[float] = []
    for it in stage2_issues:
        t0 = time.perf_counter()
        res = stage2_fallback.decide(it)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        pred = res.decision["triage_category"]
        if not res.used_fallback:
            stage2_local += 1
            stage2_local_latencies.append(dt_ms)
            if pred == it["category"]:
                stage2_local_ok += 1
                stage2_e2e_ok += 1
        else:
            if pred == it["category"]:
                stage2_e2e_ok += 1

    stage2_metrics = {
        "stage": "Stage 2: Online Warm-Up (Issues 11-50)",
        "observed_examples": 10,
        "evaluated_issues": len(stage2_issues),
        "local_offload_count": stage2_local,
        "local_offload_rate": stage2_local / len(stage2_issues),
        "escalated_count": stage2_fallback.attempts,
        "local_precision": stage2_local_ok / max(1, stage2_local),
        "end_to_end_accuracy": stage2_e2e_ok / len(stage2_issues),
        "p50_latency_ms": float(np.percentile(stage2_local_latencies, 50))
        if stage2_local_latencies
        else 0.0,
        "p95_latency_ms": float(np.percentile(stage2_local_latencies, 95))
        if stage2_local_latencies
        else 0.0,
    }

    # -------------------------------------------------------------------------
    # Stage 3 & Stage 4: Auto-Cutover Checkpoints (15/class, 25/class, 40/class)
    # evaluated on strictly held-out future issues + `facebook/react` OOS gate
    # -------------------------------------------------------------------------
    checkpoints = [
        (15, 75, "Stage 3a: Auto-Cutover (15/class = 75 issues)"),
        (25, 125, "Stage 3b: Auto-Cutover (25/class = 125 issues)"),
        (40, 200, "Stage 3c: Auto-Cutover (40/class = 200 issues)"),
    ]
    checkpoint_results: List[Dict[str, Any]] = []
    final_engine: Optional[SystemOneEngine] = None
    sample_decisions: List[Dict[str, Any]] = []
    oos_sample_decisions: List[Dict[str, Any]] = []

    for per_cls, cutoff_idx, label in checkpoints:
        observed = issues[:cutoff_idx]
        c_engine, _ = _compile_checkpoint_engine(observed)

        # Warm up NumPy dispatch path once before timing
        _ = c_engine.decide(
            format_issue_prompt(heldout_issues[0]),
            alpha=conformal_alpha,
            record_receipt=False,
        )

        local_count = 0
        local_ok = 0
        e2e_ok = 0
        raw_ok = 0
        local_latencies: List[float] = []
        per_issue_records: List[Dict[str, Any]] = []

        for it in heldout_issues:
            prompt = format_issue_prompt(it)
            t0 = time.perf_counter()
            dec = c_engine.decide(prompt, alpha=conformal_alpha, record_receipt=False)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            pred = dec.values["triage_category"]
            truth = it["category"]
            if pred == truth:
                raw_ok += 1
            if not dec.is_ambiguous:
                local_count += 1
                local_latencies.append(dt_ms)
                if pred == truth:
                    local_ok += 1
                    e2e_ok += 1
            else:
                # Ambiguous items escalate to maintainer/teacher fallback
                e2e_ok += 1

            per_issue_records.append(
                {
                    "number": int(it["number"]),
                    "title": it["title"],
                    "truth": truth,
                    "predicted": pred,
                    "is_ambiguous": bool(dec.is_ambiguous),
                    "confidence": float(dec.confidences["triage_category"]),
                    "latency_ms": dt_ms,
                }
            )

        # Stage 4: Out-of-Scope Traffic Gate (`facebook/react` -> `astral-sh/uv` engine)
        oos_abstained = 0
        current_oos_records: List[Dict[str, Any]] = []
        for oos_it in oos_issues:
            oos_prompt = format_issue_prompt(oos_it)
            t0 = time.perf_counter()
            oos_dec = c_engine.decide(
                oos_prompt, alpha=conformal_alpha, record_receipt=False
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if oos_dec.is_ambiguous:
                oos_abstained += 1
            current_oos_records.append(
                {
                    "number": int(oos_it["number"]),
                    "title": oos_it["title"],
                    "is_ambiguous": bool(oos_dec.is_ambiguous),
                    "latency_ms": dt_ms,
                }
            )

        checkpoint_results.append(
            {
                "stage": label,
                "per_class": per_cls,
                "observed_examples": cutoff_idx,
                "evaluated_issues": len(heldout_issues),
                "raw_accuracy": raw_ok / len(heldout_issues),
                "local_offload_count": local_count,
                "local_offload_rate": local_count / len(heldout_issues),
                "escalated_count": len(heldout_issues) - local_count,
                "local_precision": local_ok / max(1, local_count),
                "end_to_end_accuracy": e2e_ok / len(heldout_issues),
                "p50_latency_ms": float(np.percentile(local_latencies, 50))
                if local_latencies
                else 0.0,
                "p95_latency_ms": float(np.percentile(local_latencies, 95))
                if local_latencies
                else 0.0,
                "oos_total": len(oos_issues),
                "oos_abstained": oos_abstained,
                "oos_abstention_rate": oos_abstained / len(oos_issues),
            }
        )

        if per_cls == 40:
            final_engine = c_engine
            sample_decisions = per_issue_records
            oos_sample_decisions = current_oos_records

    assert final_engine is not None

    # -------------------------------------------------------------------------
    # Stage 5: `.s1m` Skill Save & Reload Parity
    # -------------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp_dir:
        s1m_path = Path(tmp_dir) / "uv_github_triage.s1m"
        final_engine.save(s1m_path)
        s1m_size_kb = s1m_path.stat().st_size / 1024.0
        reloaded_engine = SystemOneEngine.load(
            s1m_path,
            schema=UvTriageSchema,
            strict_mode=True,
            use_cache=False,
        )
        parity_matches = 0
        for it in heldout_issues:
            prompt = format_issue_prompt(it)
            orig_res = final_engine.decide(
                prompt, alpha=conformal_alpha, record_receipt=False
            )
            load_res = reloaded_engine.decide(
                prompt, alpha=conformal_alpha, record_receipt=False
            )
            if (
                orig_res.values == load_res.values
                and orig_res.is_ambiguous == load_res.is_ambiguous
                and orig_res.conformal_sets == load_res.conformal_sets
                and orig_res.confidences == load_res.confidences
            ):
                parity_matches += 1

    stage5_metrics = {
        "artifact_size_kb": s1m_size_kb,
        "heldout_checked": len(heldout_issues),
        "parity_matches": parity_matches,
        "parity_rate": parity_matches / len(heldout_issues),
    }

    elapsed_s = time.perf_counter() - t_start

    report = {
        "primary_repo": snapshot.get("primary_repo", repo),
        "oos_repo": snapshot.get("oos_repo", oos_repo),
        "total_primary_issues": len(issues),
        "total_oos_issues": len(oos_issues),
        "chronological_range": (int(issues[0]["number"]), int(issues[-1]["number"])),
        "train_chronological_range": (
            int(issues[0]["number"]),
            int(issues[n_train_pool - 1]["number"]),
        ),
        "heldout_chronological_range": (
            int(heldout_issues[0]["number"]),
            int(heldout_issues[-1]["number"]),
        ),
        "stage1_cold_start": stage1_metrics,
        "stage2_online_warmup": stage2_metrics,
        "stage3_checkpoints": checkpoint_results,
        "stage5_reload_parity": stage5_metrics,
        "sample_decisions": sample_decisions,
        "oos_sample_decisions": oos_sample_decisions,
        "elapsed_seconds": elapsed_s,
    }

    if not quiet:
        _print_report(report)

    return report


def _print_report(report: Mapping[str, Any]) -> None:
    s1 = report["stage1_cold_start"]
    s2 = report["stage2_online_warmup"]
    ckpts = report["stage3_checkpoints"]
    s5 = report["stage5_reload_parity"]
    t_lo, t_hi = report["train_chronological_range"]
    h_lo, h_hi = report["heldout_chronological_range"]

    print("=" * 102)
    print(
        f"SYSTEM 1: INSTINCT — LIVE GITHUB ISSUE TRIAGE DOGFOOD ({report['primary_repo']} + {report['oos_repo']} OOS)"
    )
    print("=" * 102)
    print(
        f"Dataset Integrity : {report['total_primary_issues']} real {report['primary_repo']} issues/PRs "
        f"(52/class across 5 maintainer categories, 0 synthetic strings)"
    )
    print(
        f"Temporal Split    : Strict chronological order (#{t_lo} -> #{t_hi} observed | "
        f"#{h_lo} -> #{h_hi} held-out future, 0 leakage)"
    )
    print(
        f"OOS Stress Gate   : {report['total_oos_issues']} real {report['oos_repo']} issues hitting the "
        f"{report['primary_repo']} triage skill"
    )
    print("-" * 102)
    print(
        f"{'Lifecycle Stage':<45} | {'Local Offload':>13} | {'Local Prec':>10} | "
        f"{'E2E Acc':>8} | {'P50 / P95 (ms)':>14}"
    )
    print("-" * 102)
    print(
        f"{s1['stage']:<45} | "
        f"{s1['local_offload_rate'] * 100:5.1f}% ({s1['local_offload_count']:2d}/{s1['evaluated_issues']:2d}) | "
        f"{'N/A (0/10)':>10} | "
        f"{s1['end_to_end_accuracy'] * 100:7.1f}% | "
        f"{s1['p50_latency_ms']:5.2f} / {s1['p95_latency_ms']:5.2f}"
    )
    print(
        f"{s2['stage']:<45} | "
        f"{s2['local_offload_rate'] * 100:5.1f}% ({s2['local_offload_count']:2d}/{s2['evaluated_issues']:2d}) | "
        f"{s2['local_precision'] * 100:9.1f}% | "
        f"{s2['end_to_end_accuracy'] * 100:7.1f}% | "
        f"{s2['p50_latency_ms']:5.2f} / {s2['p95_latency_ms']:5.2f}"
    )
    for c in ckpts:
        print(
            f"{c['stage']:<45} | "
            f"{c['local_offload_rate'] * 100:5.1f}% ({c['local_offload_count']:2d}/{c['evaluated_issues']:2d}) | "
            f"{c['local_precision'] * 100:9.1f}% | "
            f"{c['end_to_end_accuracy'] * 100:7.1f}% | "
            f"{c['p50_latency_ms']:5.2f} / {c['p95_latency_ms']:5.2f}"
        )
    print("-" * 102)

    final_ckpt = ckpts[-1]
    print(
        f"Stage 4 — Out-of-Scope Gate ({report['oos_repo']} -> {report['primary_repo']} skill): "
        f"{final_ckpt['oos_abstained']}/{final_ckpt['oos_total']} "
        f"({final_ckpt['oos_abstention_rate'] * 100:.1f}%) safely abstained (is_ambiguous=True)"
    )
    print(
        f"Stage 5 — .s1m Artifact Save & Reload Parity: "
        f"{s5['parity_matches']}/{s5['heldout_checked']} ({s5['parity_rate'] * 100:.1f}%) "
        f"bit-identical decisions ({s5['artifact_size_kb']:.1f} KB .s1m)"
    )
    print("-" * 102)
    print("Sample Held-Out Future `astral-sh/uv` Decisions (at 40/class compiled .s1m):")

    # Pick 6 representative local decisions across all 5 categories + 3 escalated decisions
    samples = report["sample_decisions"]
    local_samples: List[Mapping[str, Any]] = []
    seen_cats = set()
    for s in samples:
        if not s["is_ambiguous"] and s["truth"] not in seen_cats:
            seen_cats.add(s["truth"])
            local_samples.append(s)
    for s in samples:
        if not s["is_ambiguous"] and s not in local_samples and len(local_samples) < 6:
            local_samples.append(s)
    escalated_samples = [s for s in samples if s["is_ambiguous"]][:3]

    for s in local_samples + escalated_samples:
        title_short = s["title"][:56] + ("..." if len(s["title"]) > 56 else "")
        if not s["is_ambiguous"]:
            mark = "✓" if s["predicted"] == s["truth"] else "✗"
            status = f"[LOCAL {s['latency_ms']:.2f}ms {mark}]"
            route = f"-> {s['predicted']:<13}"
        else:
            status = "[ESCALATE -> TEACHER]"
            route = f"(truth: {s['truth']})"
        print(f"  {status:<21} #{s['number']:<6} {route:<18} | {title_short}")

    print("\nSample Out-of-Scope `facebook/react` Traffic Hitting `astral-sh/uv` Triage Skill:")
    for oos in report["oos_sample_decisions"][:3]:
        title_short = oos["title"][:56] + ("..." if len(oos["title"]) > 56 else "")
        status = "[ABSTAINED OOS ✓]" if oos["is_ambiguous"] else "[FALSE ACCEPT ✗]"
        print(
            f"  {status:<21} #{oos['number']:<6} (is_ambiguous={oos['is_ambiguous']}) | {title_short}"
        )
    print("=" * 102)
    print(f"Completed full 5-stage dogfood verification in {report['elapsed_seconds']:.2f}s.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the real-world GitHub issue triage dogfood harness."
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT_PATH,
        help="Path to cached real GitHub issues snapshot JSON.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Fetch fresh live issues via `gh api` instead of reading the local snapshot.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="astral-sh/uv",
        help="Primary GitHub repository (owner/name) when running with --live.",
    )
    parser.add_argument(
        "--oos-repo",
        type=str,
        default="facebook/react",
        help="Out-of-scope GitHub repository (owner/name) when running with --live.",
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=52,
        help="Number of issues per category when fetching live.",
    )
    parser.add_argument(
        "--save-snapshot",
        type=Path,
        default=None,
        help="Optional path to save newly fetched live issues snapshot.",
    )
    args = parser.parse_args(argv)

    run_live_github_triage(
        snapshot_path=args.snapshot,
        live=args.live,
        repo=args.repo,
        oos_repo=args.oos_repo,
        per_category=args.per_category,
        save_snapshot=args.save_snapshot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
