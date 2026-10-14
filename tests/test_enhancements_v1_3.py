from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from reflex.fallback import BoundedReviewFallback as ReflexBoundedReviewFallback
from system1 import (
    BooleanField,
    BoundedReviewFallback,
    ChoiceField,
    CompiledSystemOneModel,
    ConformalPredictor,
    DecisionSchema,
    MultiChoiceField,
    ScoreField,
    SystemOneCompiler,
    SystemOneEngine,
    TeachingSession,
    TfidfProjector,
)
from system1.cli import main as cli_main
from system1.compiler import MAX_SKILL_DIMENSION
from system1.compat.typesafe import AsyncTypeSafeClient, Choice, Noul, PromotionPolicy, TypeSafeClient
from system1.core.text import MAX_FEATURES


def test_track1_tfidf_supervised_selection_and_char_ngrams():
    assert MAX_FEATURES == 8192
    assert MAX_SKILL_DIMENSION == 8192

    # Supervised selection: chi2 and mi prioritize class-discriminative signal over noisy distractors
    texts = []
    labels = []
    for i in range(20):
        texts.append(f"common_word_{i % 3} alpha_common invoice refund billing_signal")
        labels.append("billing")
        texts.append(f"common_word_{i % 3} alpha_common outage crash technical_signal")
        labels.append("tech")

    proj_chi2 = TfidfProjector.fit(
        texts,
        labels=labels,
        selection="chi2",
        max_features=6,
        min_terms_per_class=2,
    )
    vocab_terms = set(proj_chi2.vocabulary.keys())
    assert "billing_signal" in vocab_terms or "invoice" in vocab_terms
    assert "technical_signal" in vocab_terms or "outage" in vocab_terms or "crash" in vocab_terms
    assert "alpha_common" not in vocab_terms

    proj_mi = TfidfProjector.fit(
        texts,
        labels=labels,
        selection="mi",
        max_features=6,
        min_terms_per_class=2,
    )
    assert len(proj_mi.vocabulary) <= 6

    # Boundary character n-grams handle typos and morphological variants
    proj_ngrams = TfidfProjector.fit(
        ["invoice refund request", "kubernetes cluster failure"],
        char_ngrams=(3, 4),
        max_features=256,
    )
    # Typo 'invoic' still matches character n-grams from 'invoice'
    assert proj_ngrams.raw_norm("invoic refun") > 0.0
    # Completely disjoint characters/symbols have zero raw_norm
    assert proj_ngrams.raw_norm("99999") == 0.0

    # Round-trip serialization preserves char_ngrams and selection metadata
    reloaded = TfidfProjector.from_dict(proj_ngrams.to_dict())
    assert reloaded.char_ngrams == (3, 4)
    assert reloaded("invoic refun") == pytest.approx(proj_ngrams("invoic refun"))

    # Validation edge cases
    with pytest.raises(ValueError, match="labels"):
        TfidfProjector.fit(["a", "b"], selection="chi2")
    with pytest.raises(ValueError, match="labels"):
        TfidfProjector.fit(["a", "b"], labels=["x"], selection="chi2")
    with pytest.raises(ValueError, match="selection"):
        TfidfProjector.fit(["a", "b"], selection="invalid")
    with pytest.raises(ValueError, match="char_ngrams"):
        TfidfProjector.fit(["a", "b"], char_ngrams=(1, 3))
    with pytest.raises(ValueError, match="char_ngrams"):
        TfidfProjector.fit(["a", "b"], char_ngrams=(4, 3))


def test_track1_lazy_covariance_allocation(tmp_path):
    class RouteSchema(DecisionSchema):
        route = ChoiceField(["billing", "tech"])

    compiler = SystemOneCompiler(RouteSchema(), dimension=64, choice_solver="logistic")
    exemplars = {
        "route": (
            [(f"invoice charge billing #{i}", "billing") for i in range(12)]
            + [(f"server crash outage #{i}", "tech") for i in range(12)]
        )
    }
    model = compiler.compile(exemplars)
    # Covariance matrix is lazily omitted in live DecisionFieldHead when P/B are None
    assert model._field_heads["route"].covariance_inv is None

    # Save and load preserves lazy None covariance in live head
    s1m_path = tmp_path / "lazy.s1m"
    model.save(s1m_path)
    loaded = CompiledSystemOneModel.load(s1m_path)
    assert loaded._field_heads["route"].covariance_inv is None

    # Online learn_from_tier2 lazily allocates covariance on first call
    loaded.learn_from_tier2("new billing invoice dispute", "billing")
    assert loaded._field_heads["route"].covariance_inv is not None


def test_track2_mondrian_and_familiarity_ood_gating():
    class TriageSchema(DecisionSchema):
        category = ChoiceField(["billing", "tech", "rare"])

    train_rows = (
        [(f"billing invoice payment charge refund #{i}", "billing") for i in range(20)]
        + [(f"server outage database crash stacktrace #{i}", "tech") for i in range(20)]
        + [(f"executive subpoena legal discovery #{i}", "rare") for i in range(6)]
    )
    calib_rows = (
        [(f"billing invoice charge statement #{i}", "billing") for i in range(10)]
        + [(f"server outage crash incident #{i}", "tech") for i in range(10)]
        + [(f"executive subpoena legal #{i}", "rare") for i in range(2)]
    )
    all_texts = [t for t, _ in train_rows]
    all_labels = [y for _, y in train_rows]
    projector = TfidfProjector.fit(
        all_texts,
        labels=all_labels,
        selection="chi2",
        char_ngrams=(3, 4),
        max_features=256,
    )
    compiler = SystemOneCompiler(
        TriageSchema(),
        dimension=projector.dimension,
        projector=projector,
        mondrian=True,
        familiarity_alpha=0.10,
    )
    model = compiler.compile(
        {"category": train_rows},
        calibration_exemplars={"category": calib_rows},
        calibration_split=0.0,
    )
    head_weights = model.heads["category"]
    assert head_weights.mondrian is True
    assert head_weights.familiarity_threshold > 0.0
    assert set(head_weights.class_calibration_scores.keys()) == {"billing", "tech", "rare"}

    engine = SystemOneEngine(TriageSchema(), model=model)
    predictor = engine.conformal_predictors["category"]
    # Serialization round-trip preserves Mondrian and familiarity threshold
    restored = ConformalPredictor.from_dict(predictor.to_dict())
    assert restored.mondrian is True
    assert restored.familiarity_threshold == pytest.approx(predictor.familiarity_threshold)
    assert restored.class_quantile("rare", alpha=0.1) > 0.0

    in_dist = engine.decide("billing invoice payment charge refund", record_receipt=False)
    assert in_dist.metadata["familiarity_score"] >= predictor.familiarity_threshold
    assert in_dist.metadata["familiarity_triggered"] is False

    # Completely out-of-vocabulary input triggers familiarity OOD abstention (empty conformal set)
    ood_res = engine.decide("zzzz_unseen_alien_tokens_99999", record_receipt=False)
    assert ood_res.metadata["familiarity_triggered"] is True
    assert ood_res.conformal_sets["category"] == []
    assert ood_res.is_ambiguous is True


def test_track3_teaching_session_boolean_multichoice_score_and_multifield(tmp_path):
    # 1. BooleanField schema
    class ApprovalSkill(DecisionSchema):
        approved = BooleanField()

    bool_session = TeachingSession(
        tmp_path / "bool_skill",
        ApprovalSkill,
        alpha=0.10,
        familiarity_alpha=0.05,
    )
    for split, count in (("teach", 10), ("calibrate", 10), ("evaluate", 15)):
        for i in range(count):
            bool_session.record(
                f"safe routine read request #{split}-{i}",
                True,
                split=split,
                group=f"{split}-true-{i}",
            )
            bool_session.record(
                f"danger delete production database #{split}-{i}",
                False,
                split=split,
                group=f"{split}-false-{i}",
            )
    bool_report = bool_session.assess()
    assert "approved" in bool_report["candidate"]["fields"]
    bool_session.adopt()
    pred_true = bool_session.predict("safe routine read request")
    assert pred_true.values["approved"] is True
    ood_details = bool_session.decision_details("99999_alien_unseen_symbol")
    assert ood_details["review"] is True
    assert "low_feature_familiarity" in ood_details["review_reasons"]

    # 2. MultiChoiceField schema
    class TagSkill(DecisionSchema):
        tags = MultiChoiceField(["billing", "urgent"])

    tag_session = TeachingSession(tmp_path / "tag_skill", TagSkill, alpha=0.15)
    for split, count in (("teach", 10), ("calibrate", 8), ("evaluate", 10)):
        for i in range(count):
            tag_session.record(
                f"invoice billing charge normal #{split}-{i}",
                ["billing"],
                split=split,
                group=f"{split}-b-{i}",
            )
            tag_session.record(
                f"critical urgent production outage #{split}-{i}",
                ["urgent"],
                split=split,
                group=f"{split}-u-{i}",
            )
    tag_report = tag_session.assess()
    assert tag_report["candidate"]["fields"]["tags"]["type"] == "multi_choice"

    # 3. ScoreField schema
    class RiskScoreSkill(DecisionSchema):
        risk = ScoreField(min_value=0.0, max_value=10.0)

    score_session = TeachingSession(tmp_path / "score_skill", RiskScoreSkill, alpha=0.15)
    for split, count in (("teach", 10), ("calibrate", 8), ("evaluate", 10)):
        for i in range(count):
            score_session.record(
                f"harmless documentation typo fix #{split}-{i}",
                1.0,
                split=split,
                group=f"{split}-low-{i}",
            )
            score_session.record(
                f"critical root credential exfiltration #{split}-{i}",
                9.0,
                split=split,
                group=f"{split}-high-{i}",
            )
    score_report = score_session.assess()
    assert score_report["candidate"]["fields"]["risk"]["type"] == "score"
    assert score_report["candidate"]["fields"]["risk"]["mae"] is not None

    # 4. Multi-Field schema (ChoiceField + BooleanField)
    class MultiFieldSkill(DecisionSchema):
        queue = ChoiceField(["billing", "tech"])
        urgent = BooleanField()

    multi_session = TeachingSession(
        tmp_path / "multi_skill",
        MultiFieldSkill,
        selection="chi2",
        char_ngrams=(3, 4),
        mondrian=True,
        alpha=0.10,
    )
    for split, count in (("teach", 12), ("calibrate", 10), ("evaluate", 15)):
        for i in range(count):
            multi_session.record(
                f"routine billing invoice receipt question #{split}-{i}",
                {"queue": "billing", "urgent": False},
                split=split,
                group=f"{split}-b-{i}",
            )
            multi_session.record(
                f"emergency production server outage crash #{split}-{i}",
                {"queue": "tech", "urgent": True},
                split=split,
                group=f"{split}-t-{i}",
            )
    multi_report = multi_session.assess()
    assert set(multi_report["candidate"]["fields"].keys()) == {"queue", "urgent"}
    multi_session.adopt()
    multi_details = multi_session.decision_details("emergency production server outage crash")
    assert multi_details["values"]["queue"] == "tech"
    assert multi_details["values"]["urgent"] is True
    assert set(multi_details["fields"].keys()) == {"queue", "urgent"}


def test_track3_cli_teach_with_new_flags(tmp_path, capsys):
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "name": "Triage",
                "fields": {
                    "route": {
                        "type": "choice",
                        "options": ["billing", "tech"],
                        "escalate_on_ambiguity": True,
                    },
                    "urgent": {
                        "type": "boolean",
                        "escalate_on_ambiguity": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    dataset_path = tmp_path / "dataset.json"
    dataset = {"teach": [], "calibrate": [], "evaluate": []}
    for split, count in (("teach", 8), ("calibrate", 8), ("evaluate", 8)):
        for i in range(count):
            dataset[split].append(
                {
                    "input": f"invoice billing refund request #{split}-{i}",
                    "label": {"route": "billing", "urgent": False},
                    "group": f"{split}-b-{i}",
                }
            )
            dataset[split].append(
                {
                    "input": f"urgent server crash outage #{split}-{i}",
                    "label": {"route": "tech", "urgent": True},
                    "group": f"{split}-t-{i}",
                }
            )
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    workspace = tmp_path / "cli_workspace"
    rc = cli_main(
        [
            "teach",
            str(workspace),
            "--schema",
            str(schema_path),
            "--dataset",
            str(dataset_path),
            "--selection",
            "chi2",
            "--char-ngrams",
            "3,4",
            "--alpha",
            "0.15",
            "--mondrian",
            "--familiarity-alpha",
            "0.05",
            "--json",
        ]
    )
    assert rc == 0
    capsys.readouterr()

    rc = cli_main(["teach", str(workspace), "--assess", "--json"])
    assert rc in (0, 2)
    out = json.loads(capsys.readouterr().out)
    assert set(out["candidate"]["fields"].keys()) == {"route", "urgent"}


def test_track4_background_compile_and_export_teaching_session(tmp_path):
    questions = {
        "department": Choice(
            instructions="Route ticket",
            criteria={"billing": "Invoices and charges", "tech": "Outages and bugs"},
        )
    }

    def teacher_callback(state: str, qs: dict) -> dict:
        is_billing = "invoice" in state.lower() or "charge" in state.lower()
        choice = "billing" if is_billing else "tech"
        return {
            "model": "teacher-mock",
            "state": state,
            "answers": {
                "department": {
                    "type": "choice",
                    "choice": choice,
                    "value": choice,
                    "confidence": 0.99,
                }
            },
        }

    policy = PromotionPolicy(
        min_validation_samples=2,
        min_agreement_threshold=0.70,
        require_statistical_bound=False,
        min_local_acceptance=0.0,
    )
    client = TypeSafeClient(
        mode="auto_cutover",
        zero_egress=True,
        baseline_handler=teacher_callback,
        cutover_threshold=20,
        background_compile=True,
        promotion_policy=policy,
    )

    for i in range(12):
        client.systemone(
            f"invoice billing charge statement request #{i}",
            questions,
            group_id=f"b-{i}",
        )
        client.systemone(
            f"technical server crash outage bug #{i}",
            questions,
            group_id=f"t-{i}",
        )

    assert client.wait_for_compilation(timeout=5.0) is True
    assert client.is_compiling is False
    assert client.is_cutover is True

    local_resp = client.systemone("invoice billing charge statement request #99", questions)
    assert local_resp["local_execution"] is True
    assert local_resp.answers.department.value == "billing"

    # Export observed teacher history into a governed TeachingSession
    exported_dir = tmp_path / "exported_session"
    session = client.export_teaching_session(exported_dir, alpha=0.25)
    snap = session.snapshot()
    assert snap["counts"]["teach"] > 0
    assert snap["counts"]["calibrate"] > 0
    assert snap["counts"]["evaluate"] > 0
    report = session.assess()
    assert "candidate" in report


def test_track4_async_background_compile_and_export(tmp_path):
    questions = {
        "urgent": Noul(instructions="Is it urgent?"),
    }

    def teacher_callback(state: str, qs: dict) -> dict:
        val = "urgent" in state.lower() or "outage" in state.lower()
        return {
            "model": "teacher-mock",
            "state": state,
            "answers": {
                "urgent": {
                    "type": "noul",
                    "noul": 0.95 if val else 0.05,
                    "value": val,
                    "confidence": 0.95,
                }
            },
        }

    async def _run():
        client = AsyncTypeSafeClient(
            mode="auto_cutover",
            zero_egress=True,
            baseline_handler=teacher_callback,
            cutover_threshold=16,
            background_compile=True,
            promotion_policy=PromotionPolicy(
                min_validation_samples=2,
                min_agreement_threshold=0.70,
                require_statistical_bound=False,
                min_local_acceptance=0.0,
            ),
        )
        for i in range(10):
            await client.systemone(f"urgent production outage #{i}", questions, group_id=f"u-{i}")
            await client.systemone(f"routine newsletter update #{i}", questions, group_id=f"r-{i}")
        await client.wait_for_compilation(timeout=5.0)
        assert client.is_cutover is True
        session = await client.export_teaching_session(tmp_path / "async_exported", alpha=0.25)
        assert session.snapshot()["counts"]["teach"] > 0
        await client.close()

    asyncio.run(_run())


def test_track5_live_s1m_hot_reload_and_bounded_fallback(tmp_path):
    assert BoundedReviewFallback is ReflexBoundedReviewFallback

    class RouteSchema(DecisionSchema):
        route = ChoiceField(["billing", "tech"])

    s1m_path = tmp_path / "live_skill.s1m"

    # Model v1 maps 'alpha_token' -> 'billing', 'beta_token' -> 'tech'
    proj1 = TfidfProjector.fit(["alpha_token", "beta_token"])
    comp1 = SystemOneCompiler(RouteSchema(), dimension=proj1.dimension, projector=proj1)
    m1 = comp1.compile(
        {
            "route": (
                [("alpha_token", "billing")] * 8
                + [("beta_token", "tech")] * 8
            )
        }
    )
    m1.save(s1m_path)

    engine = SystemOneEngine(RouteSchema(), model_path=s1m_path, auto_reload=True)
    res1 = engine.decide("alpha_token", record_receipt=False)
    assert res1.values["route"] == "billing"
    assert engine.reload_if_changed() is False

    # Model v2 flips 'alpha_token' -> 'tech', 'beta_token' -> 'billing'
    m2 = comp1.compile(
        {
            "route": (
                [("alpha_token", "tech")] * 8
                + [("beta_token", "billing")] * 8
            )
        }
    )
    tmp_swap = tmp_path / "live_skill.tmp.s1m"
    m2.save(tmp_swap)
    # Bump mtime_ns deterministically so stat change is always detected
    stat = s1m_path.stat()
    os.utime(tmp_swap, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
    os.replace(tmp_swap, s1m_path)

    res2 = engine.decide("alpha_token", record_receipt=False)
    assert res2.values["route"] == "tech"

    # Partial/corrupt mid-write file should not crash live decide(); keeps active model until write completes
    s1m_path.write_bytes(b"S1M\x01truncated_mid_write")
    stat2 = s1m_path.stat()
    os.utime(s1m_path, ns=(stat2.st_atime_ns, stat2.st_mtime_ns + 20_000_000))
    res_during_partial = engine.decide("alpha_token", record_receipt=False)
    assert res_during_partial.values["route"] == "tech"


def test_bounded_review_fallback_falsy_values_and_guard(tmp_path):
    from system1.guard import PolicyEngine, PolicyRule, DecisionOutcome, RiskLevel, SystemOneGuard

    class BoolSchema(DecisionSchema):
        allow = BooleanField()

    class ScoreOnlySchema(DecisionSchema):
        risk = ScoreField(min_value=0.0, max_value=1.0)

    class TagsSchema(DecisionSchema):
        tags = MultiChoiceField(["a", "b"])

    class AmbiguousStub:
        def decide(self, ctx):
            return {"action": None, "review": True, "calls": 1}

    async def _run():
        # 1. Falsy BooleanField answer (False) must be accepted, not rejected
        fb_bool = BoundedReviewFallback(
            AmbiguousStub(),
            schema=BoolSchema,
            action_key="allow",
            expert=lambda ctx: asyncio.sleep(0, result=False),
            allowed=lambda ctx: True,
            validate=lambda ctx, ans: ans is False,
            expert_budget=2,
        )
        res_bool = await fb_bool.decide("check false")
        assert res_bool["recovered"] is True
        assert res_bool["recovery_reason"] == "validated_expert"
        assert res_bool["allow"] is False

        # 2. Falsy ScoreField answer (0.0) must be accepted
        fb_score = BoundedReviewFallback(
            AmbiguousStub(),
            schema=ScoreOnlySchema,
            action_key="risk",
            expert=lambda ctx: asyncio.sleep(0, result=0.0),
            allowed=lambda ctx: True,
            validate=lambda ctx, ans: ans == 0.0,
            expert_budget=2,
        )
        res_score = await fb_score.decide("check zero")
        assert res_score["recovered"] is True
        assert res_score["risk"] == 0.0

        # 3. Falsy MultiChoiceField answer ([]) must be accepted
        fb_tags = BoundedReviewFallback(
            AmbiguousStub(),
            schema=TagsSchema,
            action_key="tags",
            expert=lambda ctx: asyncio.sleep(0, result=[]),
            allowed=lambda ctx: True,
            validate=lambda ctx, ans: ans == [],
            expert_budget=2,
        )
        res_tags = await fb_tags.decide("check empty list")
        assert res_tags["recovered"] is True
        assert res_tags["tags"] == []

        # 4. Invalid schema value (out of range score 5.0) must yield 'candidate_rejected', not 'callback_or_context_failed'
        fb_invalid = BoundedReviewFallback(
            AmbiguousStub(),
            schema=ScoreOnlySchema,
            action_key="risk",
            expert=lambda ctx: asyncio.sleep(0, result=5.0),
            allowed=lambda ctx: True,
            validate=lambda ctx, ans: True,
            expert_budget=2,
        )
        res_invalid = await fb_invalid.decide("check out of bounds")
        assert res_invalid["recovered"] is False
        assert res_invalid["recovery_reason"] == "candidate_rejected"

        # 5. SystemOneGuard instance with deterministic ALLOW and DENY rules
        allow_guard = SystemOneGuard(
            policy=PolicyEngine(
                rules=[
                    PolicyRule(
                        rule_id="allow_safe",
                        outcome=DecisionOutcome.ALLOW,
                        risk=RiskLevel.READ_ONLY,
                        target_pattern="safe_move",
                    ),
                    PolicyRule(
                        rule_id="deny_danger",
                        outcome=DecisionOutcome.DENY,
                        risk=RiskLevel.IRREVERSIBLE,
                        target_pattern="danger_move",
                    ),
                ]
            )
        )
        fb_guard_ok = BoundedReviewFallback(
            AmbiguousStub(),
            guard=allow_guard,
            expert=lambda ctx: asyncio.sleep(0, result="safe_move"),
            allowed=lambda ctx: True,
            validate=lambda ctx, ans: True,
            expert_budget=2,
        )
        res_guard_ok = await fb_guard_ok.decide("move")
        assert res_guard_ok["recovered"] is True
        assert res_guard_ok["action"] == "safe_move"

        fb_guard_deny = BoundedReviewFallback(
            AmbiguousStub(),
            guard=allow_guard,
            expert=lambda ctx: asyncio.sleep(0, result="danger_move"),
            allowed=lambda ctx: True,
            validate=lambda ctx, ans: True,
            expert_budget=2,
        )
        res_guard_deny = await fb_guard_deny.decide("move")
        assert res_guard_deny["recovered"] is False
        assert res_guard_deny["recovery_reason"] == "candidate_rejected"

    asyncio.run(_run())


def test_export_teaching_session_provenance_groups_and_active_skill(tmp_path):
    questions = {
        "department": Choice(
            options=["billing", "tech"],
            instructions="Route ticket",
        )
    }

    def teacher_callback(state: str, qs: dict) -> dict:
        choice = "billing" if "invoice" in state.lower() else "tech"
        return {
            "model": "teacher-mock",
            "state": state,
            "answers": {
                "department": {
                    "type": "choice",
                    "choice": choice,
                    "value": choice,
                    "confidence": 0.99,
                }
            },
        }

    client = TypeSafeClient(
        mode="auto_cutover",
        zero_egress=True,
        baseline_handler=teacher_callback,
        cutover_threshold=16,
        background_compile=False,
        promotion_policy=PromotionPolicy(
            min_validation_samples=2,
            min_agreement_threshold=0.70,
            require_statistical_bound=False,
            min_local_acceptance=0.0,
        ),
    )
    for i in range(10):
        client.systemone(f"invoice charge billing issue #{i}", questions, group_id=f"grp-b-{i}")
        client.systemone(f"server outage crash bug #{i}", questions, group_id=f"grp-t-{i}")

    assert client.is_cutover is True
    session = client.export_teaching_session(tmp_path / "exported", alpha=0.25)

    # 1. Provenance source is 'observed_teacher' and group keys are preserved unprefixed
    records = session.records
    assert all(r["source"] == "observed_teacher" for r in records)
    recorded_groups = {r["group"] for r in records}
    assert "grp-b-0" in recorded_groups
    assert all(not g.startswith(("teach:", "calibrate:", "evaluate:")) for g in recorded_groups)

    # 2. Exported session already has active current.s1m and revision history
    assert session.current_path.exists()
    assert len(session.history) == 1
    pred = session.predict("invoice charge billing issue #999")
    assert pred.values["department"] == "billing"


def test_mondrian_exact_vs_shrunk_and_cache_familiarity():
    import numpy as np

    cp = ConformalPredictor("intent", ["a", "b"], score_method="lac", mondrian=True, mondrian_shrinkage=10.0)
    # 40 samples for class 'a' (n_a >= ceil(1/0.05)-1 = 19 -> exact Mondrian = 0.10)
    # 1 sample for class 'b' (n_b < 19 -> empirical-Bayes shrunk between q_emp=0.40 and q_global=0.10)
    probs_a = np.array([[0.90, 0.10]] * 40)
    probs_b = np.array([[0.40, 0.60]] * 1)
    probs = np.vstack([probs_a, probs_b])
    labels = ["a"] * 40 + ["b"] * 1
    cp.calibrate(probs, labels)

    # For class 'a', k_c = ceil(41 * 0.95) = 39 <= 40, so exact Mondrian = 0.10
    assert abs(cp.class_quantile("a", alpha=0.05) - 0.10) < 1e-9
    # For class 'b', k_c = ceil(2 * 0.95) = 2 > 1, so blended between q_emp (0.40) and q_global (0.10)
    q_b = cp.class_quantile("b", alpha=0.05)
    assert 0.10 < q_b < 0.40

    # Round-trip via to_dict / from_dict preserves mondrian_shrinkage
    cp2 = ConformalPredictor.from_dict(cp.to_dict())
    assert cp2.mondrian_shrinkage == 10.0
    assert abs(cp2.class_quantile("a", alpha=0.05) - cp.class_quantile("a", alpha=0.05)) < 1e-9

    # Cache hit preserves familiarity_score and metadata
    class SimpleSchema(DecisionSchema):
        intent = ChoiceField(["a", "b"])

    proj = TfidfProjector.fit(["alpha beta", "gamma delta"])
    comp = SystemOneCompiler(SimpleSchema(), projector=proj, familiarity_alpha=0.10)
    model = comp.compile(
        {"intent": [(f"alpha beta train {i}", "a") for i in range(6)] + [(f"gamma delta train {i}", "b") for i in range(6)]},
        augment=False,
        calibration_exemplars={
            "intent": [(f"alpha beta cal {i}", "a") for i in range(4)] + [(f"gamma delta cal {i}", "b") for i in range(4)]
        },
    )
    eng = SystemOneEngine(SimpleSchema(), model=model, use_cache=True)
    r1 = eng.decide("alpha beta", record_receipt=False)
    r2 = eng.decide("alpha beta", record_receipt=False)
    assert r2.is_cache_hit is True
    assert r2.familiarity_score == r1.familiarity_score
    assert r2.metadata.get("familiarity_score") == r1.metadata.get("familiarity_score")


def test_teaching_session_multiprocess_lock(tmp_path):
    import subprocess
    import sys

    ws = tmp_path / "mp_session"
    class MpSchema(DecisionSchema):
        intent = ChoiceField(["a", "b"])

    session = TeachingSession(ws, MpSchema)
    session.record("seed input", "a", split="teach", source="init")

    worker_code = (
        "import sys; "
        "from system1.teaching import TeachingSession; "
        "s = TeachingSession(sys.argv[1]); "
        "[s.record(f'proc {sys.argv[2]} item {i}', 'a' if i % 2 == 0 else 'b', split='teach', source='mp') for i in range(5)]"
    )
    procs = [
        subprocess.Popen([sys.executable, "-c", worker_code, str(ws), str(p_id)])
        for p_id in range(4)
    ]
    for p in procs:
        assert p.wait(timeout=15) == 0

    reopened = TeachingSession(ws)
    assert len(reopened.records) == 1 + 4 * 5


