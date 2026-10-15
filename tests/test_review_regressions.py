"""Behavioral regressions from the October 2026 repository review."""
import asyncio
import io
import json
import struct
from itertools import product

import numpy as np
import pytest

from system1 import BooleanField, BoundedReviewFallback, ChoiceField, DecisionSchema, ScoreField, SystemOneEngine
from system1.cache import SemanticSystemOneCache
from system1.compiler import CompiledHeadWeights, CompiledSystemOneModel
from system1.core.model import DecisionFieldHead, DeterministicSemanticProjector
from system1.core.text import TfidfProjector


class Lesson(DecisionSchema):
    route = ChoiceField(options=["refund", "password"])
    urgent = BooleanField()
    score = ScoreField(min_value=0, max_value=10)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("bad", [{"route": "invalid"}, {"route": "password", "urgent": False, "score": "bad"},
                                  {"route": "password", "score": float("nan")}])
def test_invalid_lesson_changes_no_model_state(compiled, direct, bad):
    engine = SystemOneEngine(Lesson, backend="numpy", strict_mode=True, use_cache=False)
    if compiled:
        engine = SystemOneEngine.from_bytes(engine.to_bytes(), backend="numpy", use_cache=False)
    before = engine.to_bytes(include_covariance=True)
    version = engine.model_version
    recipient = engine.model if direct else engine
    with pytest.raises((TypeError, ValueError)):
        recipient.learn("Refund duplicate credit card payment", bad)
    assert engine.to_bytes(include_covariance=True) == before
    assert engine.model_version == version


@pytest.mark.parametrize("embedding", [[1.0, 0.0], [3.0, 4.0]])
def test_cache_churn_bounds_all_indexes_and_keeps_original_keys(embedding):
    cache = SemanticSystemOneCache(capacity=2)
    context = dict(model_digest="model", projector_digest="projector", calibration_digest="calibration",
                   policy_epoch=7, relative_odds_ratio=2.0, confidence_floor_tau0=0.1, strict=True)
    for index in range(30):
        cache.put(str(index), {"sentinel": index}, embedding=np.asarray(embedding, dtype=np.float32), **context)
        assert len(cache) <= 2
        assert len(cache._embedding_entries) <= 2
        assert len(cache._base_to_key) <= 2
        assert len(cache._key_to_emb_idx) <= 2
    assert cache.get("0", embedding=embedding, **context) is None
    assert cache.get("29", embedding=embedding, **context)[0].result == {"sentinel": 29}
    cache.invalidate_version(0)
    assert not cache._embedding_entries and not cache._base_to_key
    cache.put("fresh", {}, embedding=embedding, **context)
    cache.clear()
    assert not cache._base_to_key and not cache._key_to_emb_idx and cache._embeddings is None


class ExemplarRoutes(DecisionSchema):
    route = ChoiceField(options=["refund", "password", "cancel"], descriptions={
        "refund": "Refund duplicate credit card; invoice payment billing; charge twice refund",
        "password": "Reset password locked login; account security access; recover login",
        "cancel": "Cancel subscription membership; stop monthly plan; cancel renewal",
    })


@pytest.mark.parametrize("taught", [False, True])
def test_exemplar_routing_survives_save_reload(taught, tmp_path):
    engine = SystemOneEngine(ExemplarRoutes, backend="numpy", strict_mode=True, use_cache=False)
    if taught:
        engine.learn("Refund duplicate credit card payment", {"route": "refund"})
    path = tmp_path / "routes.s1m"
    engine.save(path)
    restored = SystemOneEngine.load(path, backend="numpy", use_cache=False)
    words = ["refund", "duplicate", "password", "locked", "login", "cancel", "subscription", "monthly"]
    for pair in product(words, repeat=2):
        text = " ".join(pair)
        original, reloaded = [e.decide(text, record_receipt=False) for e in (engine, restored)]
        assert original.values == reloaded.values
        assert original.is_ambiguous == reloaded.is_ambiguous
        assert original.conformal_sets == reloaded.conformal_sets
        for field, probabilities in original.probabilities.items():
            assert probabilities == pytest.approx(reloaded.probabilities[field], abs=1e-7)


def test_high_cardinality_online_exemplars_and_adaptation_survive_reload():
    schema = DecisionSchema(fields={"route": ChoiceField(options=[f"topic{i}" for i in range(9)],
                           descriptions={f"topic{i}": f"topic{i} keyword{i} label{i}" for i in range(9)})})
    engine = SystemOneEngine(schema, backend="numpy", strict_mode=True, use_cache=False)
    engine.adapt_zero_shot(["topic0 keyword0 fresh domain"])
    for index in range(9):
        engine.learn(f"topic{index} keyword{index} special signal", {"route": f"topic{index}"})
    restored = SystemOneEngine.from_bytes(engine.to_bytes(), backend="numpy", use_cache=False)
    for index in range(9):
        text = f"fresh keyword{index} topic{index} request"
        a, b = [e.decide(text, record_receipt=False) for e in (engine, restored)]
        assert a.values == b.values and a.probabilities == b.probabilities
        assert a.conformal_sets == b.conformal_sets and a.is_ambiguous == b.is_ambiguous


def _wire(meta, arrays):
    header = json.dumps(meta, sort_keys=True).encode()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return b"S1M\x02" + struct.pack(">I", len(header)) + header + buffer.getvalue()


def _metadata(model):
    data = model.to_bytes()
    size = struct.unpack(">I", data[4:8])[0]
    return json.loads(data[8:8 + size])


def test_b_only_covariance_is_budgeted_before_allocation(monkeypatch):
    import system1.compiler as compiler
    schema = DecisionSchema(fields={"a": ChoiceField(options=["x", "y"]), "b": ChoiceField(options=["x", "y"])})
    projector = TfidfProjector(tuple(f"word{i}" for i in range(64)), tuple([1.0] * 64))
    heads = {name: CompiledHeadWeights(name, "choice", np.zeros((2, 64), dtype=np.float32),
                                     np.zeros(2, dtype=np.float32), options=("x", "y")) for name in schema.fields}
    model = CompiledSystemOneModel(schema, heads, dimension=64, projector=projector)
    meta = _metadata(model)
    arrays = {f"{name}_{suffix}": arr for name in heads for suffix, arr in (
        ("w", heads[name].weights), ("b", heads[name].biases), ("B", np.zeros((65, 2), dtype=np.float32)))}
    data = _wire(meta, arrays)
    monkeypatch.setattr(compiler, "MAX_SKILL_RUNTIME_BYTES", 50_000)
    monkeypatch.setattr(np, "eye", lambda *a, **k: pytest.fail("covariance allocated before admission"))
    with pytest.raises(ValueError, match="memory budget"):
        CompiledSystemOneModel.from_bytes(data)


def test_calibration_python_storage_is_budgeted_before_loading_arrays(monkeypatch):
    import system1.compiler as compiler
    schema = DecisionSchema(fields={"route": ChoiceField(options=["x", "y"])})
    projector = TfidfProjector(("word",), (1.0,))
    head = CompiledHeadWeights("route", "choice", np.zeros((2, 1), dtype=np.float32), np.zeros(2, dtype=np.float32), options=("x", "y"))
    model = CompiledSystemOneModel(schema, {"route": head}, dimension=1, projector=projector)
    data = _wire(_metadata(model), {"route_w": head.weights, "route_b": head.biases,
                                   "route_calib_scores": np.linspace(0, 1, 1000, dtype=np.float32)})
    monkeypatch.setattr(compiler, "MAX_SKILL_RUNTIME_BYTES", 20_000)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("arrays materialized before admission"))
    with pytest.raises(ValueError, match="memory budget"):
        CompiledSystemOneModel.from_bytes(data)


def test_loaded_heads_do_not_rebuild_description_embeddings(monkeypatch):
    engine = SystemOneEngine(ExemplarRoutes, backend="numpy", use_cache=False)
    data = engine.to_bytes()
    monkeypatch.setattr(DecisionFieldHead, "_init_head", lambda *a, **k: pytest.fail("discarded seed head constructed"))
    CompiledSystemOneModel.from_bytes(data)


def test_hybrid_reconstruction_terms_are_budgeted_before_constructor(monkeypatch):
    import system1.compiler as compiler
    from system1.core.embeddings import HybridProjector
    engine = SystemOneEngine(ExemplarRoutes, backend="numpy", use_cache=False, dimension=16)
    model = engine.to_compiled_model()
    meta = _metadata(model)
    meta["projector"] = {"type": "hybrid", "sparse_dim": 12, "dense_dim": 4,
                         "spectral_terms": [f"term{i}" for i in range(200)]}
    data = _wire(meta, {"route_w": model.heads["route"].weights, "route_b": model.heads["route"].biases})
    monkeypatch.setattr(compiler, "MAX_SKILL_RUNTIME_BYTES", 1_610_000)
    monkeypatch.setattr(HybridProjector, "from_config", lambda *a, **k: pytest.fail("projector constructed before admission"))
    with pytest.raises(ValueError, match="memory budget"):
        CompiledSystemOneModel.from_bytes(data)


def test_model_path_and_reload_use_bounded_reads(tmp_path, monkeypatch):
    import system1.compiler as compiler
    engine = SystemOneEngine(ExemplarRoutes, backend="numpy", use_cache=False)
    path = tmp_path / "route.s1m"
    engine.save(path)
    monitored = SystemOneEngine(ExemplarRoutes, model_path=path, auto_reload=True, use_cache=False, backend="numpy")
    original = monitored.model
    monkeypatch.setattr(compiler, "MAX_SKILL_BYTES", 1024)
    path.write_bytes(b"oversized" * 1000)
    monkeypatch.setattr(type(path), "read_bytes", lambda *a: pytest.fail("unbounded read"))
    with pytest.raises(ValueError, match="size limit"):
        SystemOneEngine(ExemplarRoutes, model_path=path, backend="numpy")
    assert monitored.reload_if_changed() is False
    assert monitored.model is original


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["close", "revoke", "state"])
async def test_deferred_reviewer_is_not_called_after_revocation(change):
    class Local:
        def decide(self, prompt):
            return {"review": True, "values": {"action": "review"}}
    calls = []
    permission = {"allowed": True}
    context = {"prompt": "question", "revision": 1}
    async def expert(ctx):
        calls.append(ctx.copy())
        return {"action": "ok"}
    fallback = BoundedReviewFallback(Local(), expert=expert, allowed=lambda ctx: permission["allowed"],
                                     validate=lambda ctx, answer: True, expert_budget=1)
    pending = fallback.decide(context)
    if change == "close":
        fallback.close()
    elif change == "revoke":
        permission["allowed"] = False
    else:
        context["revision"] = 2
    result = await pending
    assert not calls and not result["recovered"]


@pytest.mark.asyncio
async def test_permission_is_checked_inside_scheduled_reviewer_task():
    class Local:
        def decide(self, prompt):
            return {"review": True}
    calls = []
    permission = {"allowed": True}
    async def expert(ctx):
        calls.append(ctx)
        return "ok"
    fallback = BoundedReviewFallback(Local(), expert=expert, allowed=lambda ctx: permission["allowed"],
                                     validate=lambda ctx, answer: True, expert_budget=1)
    continuation = asyncio.create_task(fallback.decide("question"))
    asyncio.get_running_loop().call_soon(permission.update, {"allowed": False})
    result = await continuation
    assert not calls and not result["recovered"]


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("method", ["learn", "learn_from_tier2"])
@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"target": {}},
        {"target": {"unknown": "value"}},
        {"target": None},
        {"target": {"route": "password"}, "forgetting_factor": float("nan")},
        {"target": {"route": "password"}, "forgetting_factor": 0.0},
        {"target": {"route": "password"}, "forgetting_factor": 1.5},
        {"target": {"route": "password"}, "embedding": np.zeros(8, dtype=np.float32)},
        {"target": {"route": "password"}, "embedding": np.full(384, np.nan, dtype=np.float32)},
    ],
)
def test_invalid_inputs_change_no_model_or_covariance_state(compiled, direct, method, bad_kwargs):
    engine = SystemOneEngine(Lesson, backend="numpy", strict_mode=True, use_cache=False)
    if compiled:
        engine = SystemOneEngine.from_bytes(engine.to_bytes(), backend="numpy", use_cache=False)
    before = engine.to_bytes(include_covariance=True)
    version = engine.model_version
    recipient = engine.model if direct else engine
    fn = getattr(recipient, method)
    with pytest.raises((TypeError, ValueError)):
        fn("Refund duplicate credit card payment", **bad_kwargs)
    assert engine.to_bytes(include_covariance=True) == before
    assert engine.model_version == version
    if compiled:
        for head in engine.model._field_heads.values():
            assert head.covariance_inv is None
            assert head.cross_covariance is None


def test_post_reload_online_learning_and_adapt_zero_shot_match_cold_start(tmp_path):
    engine = SystemOneEngine(ExemplarRoutes, backend="numpy", strict_mode=True, use_cache=False)
    path = tmp_path / "cold_routes.s1m"
    engine.save(path)
    reloaded = SystemOneEngine.load(path, backend="numpy", use_cache=False)

    stream = [
        "Refund duplicate credit card billing charge codeword ZETA1",
        "Reset password locked login account access codeword ZETA2",
    ]
    a_info = engine.adapt_zero_shot(stream)
    b_info = reloaded.adapt_zero_shot(stream)
    assert a_info["status"] == b_info["status"] == "adapted"
    assert a_info["adapted_samples"] == b_info["adapted_samples"]
    assert a_info["adapted_fields"] == b_info["adapted_fields"]

    engine.learn("Disputed wire transfer ledger reversal", {"route": "refund"})
    reloaded.learn("Disputed wire transfer ledger reversal", {"route": "refund"})

    engine.learn_from_tier2(
        "Hardware token MFA recovery lockout",
        {"route": "password"},
        preserve_calibration=True,
    )
    reloaded.learn_from_tier2(
        "Hardware token MFA recovery lockout",
        {"route": "password"},
        preserve_calibration=True,
    )

    probes = [
        "Disputed wire transfer ledger reversal",
        "Hardware token MFA recovery lockout",
        "Wire transfer reversal request",
        "MFA token recovery",
        "Cancel monthly renewal subscription",
        "Codeword ZETA1 invoice charge",
    ]
    for text in probes:
        orig_res = engine.decide(text, record_receipt=False)
        rel_res = reloaded.decide(text, record_receipt=False)
        assert orig_res.values == rel_res.values
        assert orig_res.is_ambiguous == rel_res.is_ambiguous
        assert orig_res.conformal_sets == rel_res.conformal_sets
        assert orig_res.probabilities["route"] == pytest.approx(rel_res.probabilities["route"], abs=1e-6)


@pytest.mark.parametrize(
    "tamper",
    [
        "non_mapping_state",
        "unknown_header_key",
        "negative_count",
        "count_sum_mismatch",
        "bad_centroid_shape",
        "shots_p_without_shots_c",
        "unexpected_array_without_header",
    ],
)
def test_malformed_exemplar_state_is_rejected_with_value_error(tamper):
    engine = SystemOneEngine(ExemplarRoutes, backend="numpy", use_cache=False)
    model = engine.to_compiled_model()
    meta = _metadata(model)
    state = model._field_heads["route"].export_exemplar_state()
    arrays = {
        "route_w": model.heads["route"].weights,
        "route_b": model.heads["route"].biases,
        **{f"route_{k}": v for k, v in state["arrays"].items()},
    }
    if tamper == "non_mapping_state":
        meta["heads"]["route"]["exemplar_state"] = None
    elif tamper == "unknown_header_key":
        meta["heads"]["route"]["exemplar_state"]["extra"] = 1
    elif tamper == "negative_count":
        meta["heads"]["route"]["exemplar_state"]["counts"]["exemplar_vectors"] = [-1, 1, 1]
    elif tamper == "count_sum_mismatch":
        meta["heads"]["route"]["exemplar_state"]["counts"]["exemplar_vectors"] = [99, 1, 1]
    elif tamper == "bad_centroid_shape":
        arrays["route_exemplar_centroids"] = np.zeros((2, 384), dtype=np.float32)
    elif tamper == "shots_p_without_shots_c":
        meta["heads"]["route"]["exemplar_state"]["counts"]["shots_p"] = [1, 0, 0]
        arrays["route_shots_p"] = np.zeros((1, 384), dtype=np.float32)
    elif tamper == "unexpected_array_without_header":
        del meta["heads"]["route"]["exemplar_state"]
    with pytest.raises(ValueError):
        CompiledSystemOneModel.from_bytes(_wire(meta, arrays))


def test_semantic_cache_hit_promotes_non_unit_embedding_entry_in_lru():
    cache = SemanticSystemOneCache(capacity=2, similarity_threshold=0.95)
    ctx = dict(
        model_digest="m",
        projector_digest="p",
        calibration_digest="c",
        strict=False,
    )
    cache.put("first", {"id": 1}, embedding=np.array([3.0, 4.0], dtype=np.float32), **ctx)
    cache.put("second", {"id": 2}, embedding=np.array([0.0, 5.0], dtype=np.float32), **ctx)
    # Semantic hit on "first" using a distinct prompt with near-collinear non-unit embedding
    hit = cache.get("first_paraphrase", embedding=np.array([3.05, 3.95], dtype=np.float32), **ctx)
    assert hit is not None and hit[0].result == {"id": 1}
    # Adding a third entry must evict "second" (the true LRU), keeping "first"
    cache.put("third", {"id": 3}, embedding=np.array([5.0, 0.0], dtype=np.float32), **ctx)
    assert cache.get("first", embedding=np.array([3.0, 4.0], dtype=np.float32), **ctx) is not None
    assert cache.get("second", embedding=np.array([0.0, 5.0], dtype=np.float32), **ctx) is None


@pytest.mark.asyncio
async def test_fallback_callback_exception_inside_scheduled_task_and_future_cancellation():
    class Local:
        def decide(self, prompt):
            return {"review": True}

    state = {"calls": 0}

    def flaky_allowed(ctx):
        state["calls"] += 1
        if state["calls"] >= 3:
            raise RuntimeError("boom inside scheduled task")
        return True

    fut = asyncio.get_running_loop().create_future()
    fallback = BoundedReviewFallback(
        Local(),
        expert=lambda ctx: fut,
        allowed=flaky_allowed,
        validate=lambda ctx, answer: True,
        expert_budget=1,
    )
    res = await fallback.decide("question")
    assert not res["recovered"]
    assert res["recovery_reason"] == "callback_or_context_failed"
    assert fut.cancelled()


def test_hybrid_projector_64_class_skill_uses_binary_arrays_and_small_header(tmp_path):
    from system1.compiler import SystemOneCompiler
    from system1.core.embeddings import HybridProjector

    options = [f"intent_{i:02d}" for i in range(64)]
    schema = DecisionSchema(fields={"intent": ChoiceField(options=options)})
    examples = {
        "intent": [
            (f"utterance {j} for intent {i:02d} with domain token tok_{i}_{j}", f"intent_{i:02d}")
            for i in range(64)
            for j in range(6)
        ]
    }
    projector = HybridProjector.fit(
        [p for p, _ in examples["intent"]],
        labels=[y for _, y in examples["intent"]],
        dimension=256,
        backend="numpy",
    )
    compiler = SystemOneCompiler(
        schema=schema,
        projector=projector,
        backend="numpy",
        choice_solver="logistic",
    )
    model = compiler.compile(examples, augment=False, calibration_split=0.25)
    assert isinstance(model.projector, HybridProjector)

    raw = model.to_bytes()
    header_len = struct.unpack(">I", raw[4:8])[0]
    assert header_len < 50_000

    path = tmp_path / "hwu64.s1m"
    model.save(path)
    reloaded = SystemOneEngine.load(path, backend="numpy", use_cache=False)
    orig_engine = SystemOneEngine(schema=schema, model=model, backend="numpy", use_cache=False)
    for i in (0, 17, 33, 63):
        probe = f"utterance 1 for intent {i:02d} with domain token tok_{i}_1"
        r1 = orig_engine.decide(probe, record_receipt=False)
        r2 = reloaded.decide(probe, record_receipt=False)
        assert r1.values == r2.values
        assert r1.conformal_sets == r2.conformal_sets
        assert r1.probabilities["intent"] == pytest.approx(r2.probabilities["intent"], abs=1e-6)


def test_tfidf_projector_raw_norm_reuses_single_tokenization_in_decide(monkeypatch):
    import system1.core.text as text_mod
    from system1.compiler import SystemOneCompiler

    schema = DecisionSchema(fields={"route": ChoiceField(options=["billing", "tech"])})
    examples = {
        "route": [
            (f"billing invoice charge refund payment {i}", "billing") for i in range(12)
        ]
        + [
            (f"technical crash error timeout bug {i}", "tech") for i in range(12)
        ]
    }
    projector = text_mod.TfidfProjector.fit(
        [p for p, _ in examples["route"]],
        labels=[y for _, y in examples["route"]],
    )
    compiler = SystemOneCompiler(schema=schema, projector=projector, backend="numpy")
    model = compiler.compile(examples, augment=False, calibration_split=0.25)
    engine = SystemOneEngine(schema=schema, model=model, backend="numpy", use_cache=False)

    calls = 0
    orig_terms = text_mod._terms

    def counting_terms(text, *args, **kwargs):
        nonlocal calls
        calls += 1
        return orig_terms(text, *args, **kwargs)

    monkeypatch.setattr(text_mod, "_terms", counting_terms)
    res = engine.decide("billing invoice charge payment", record_receipt=False)
    assert calls == 1
    assert res.familiarity_score is not None and res.familiarity_score > 0.0


def test_learn_rejects_unknown_field_in_multifield_lesson_without_mutation():
    import numpy as np

    schema = DecisionSchema(
        fields={
            "route": ChoiceField(options=["billing", "tech"]),
            "priority": ChoiceField(options=["low", "high"]),
        }
    )
    engine = SystemOneEngine(schema=schema, dimension=32, backend="numpy", use_cache=False)
    engine.learn("billing question", {"route": "billing", "priority": "low"})
    version_before = engine.model_version
    weights_before = engine.model.heads["route"].weights.copy()

    with pytest.raises(ValueError, match="priorities"):
        engine.learn("urgent billing issue", {"route": "billing", "priorities": "high"})

    assert engine.model_version == version_before
    assert np.array_equal(engine.model.heads["route"].weights, weights_before)

    with pytest.raises(ValueError, match="priorities"):
        engine.model.learn("urgent billing issue", {"route": "billing", "priorities": "high"})

    assert np.array_equal(engine.model.heads["route"].weights, weights_before)

    with pytest.raises(ValueError, match="no valid fields"):
        engine.learn_from_tier2("urgent billing issue", {"priorities": "high"})
    assert engine.model_version == version_before
    assert np.array_equal(engine.model.heads["route"].weights, weights_before)


def test_explicit_embedding_in_learn_does_not_override_prompt_predictions():
    import numpy as np
    from system1.compiler import SystemOneCompiler

    schema = DecisionSchema(fields={"route": ChoiceField(options=["billing", "tech"])})
    examples = {
        "route": [
            ("invoice refund billing charge", "billing"),
            ("billing statement payment receipt", "billing"),
            ("server crash stacktrace timeout", "tech"),
            ("database connection error bug", "tech"),
        ]
    }
    compiler = SystemOneCompiler(schema=schema, dimension=32, backend="numpy")
    model = compiler.compile(examples, augment=False, calibration_split=0.0)

    baseline = model.forward_single("invoice refund billing charge").fields["route"].selected_value
    assert baseline == "billing"

    custom_emb = np.zeros(32, dtype=np.float32)
    custom_emb[0] = 1.0
    model.learn(
        "invoice refund billing charge",
        {"route": "tech"},
        embedding=custom_emb,
        preserve_calibration=False,
    )

    # Prompt-only inference must still project the prompt text rather than returning the custom embedding's override
    after_prompt = model.forward_single("invoice refund billing charge").fields["route"].selected_value
    assert after_prompt == "billing"
    assert "invoice refund billing charge" not in model._learned_prompt_overrides


def test_solve_ridge_and_logistic_n_lt_d_augment_false_exact_match():
    import numpy as np
    from system1.compiler import SystemOneCompiler

    rng = np.random.default_rng(123)
    n, d, k = 18, 96, 4
    X = rng.standard_normal((n, d)).astype(np.float32)
    Y = np.eye(k, dtype=np.float32)[rng.integers(0, k, size=n)]

    schema = DecisionSchema(fields={"c": ChoiceField(options=[f"c{i}" for i in range(k)])})
    compiler = SystemOneCompiler(schema=schema, dimension=d, regularization=0.7, backend="numpy")

    # Compare _solve_ridge(regularize_bias=False) against direct (D+1)x(D+1) unregularized-bias normal equations
    weights, biases, P, B = compiler._solve_ridge(
        X, Y, 0.7, regularize_bias=False, return_covariance=True
    )
    X_aug = np.hstack([X.astype(np.float64), np.ones((n, 1), dtype=np.float64)])
    reg = np.diag([0.7] * d + [0.0])
    cov_ref = np.linalg.inv(X_aug.T @ X_aug + reg)
    Wb_ref = cov_ref @ (X_aug.T @ Y.astype(np.float64))
    assert np.allclose(weights, Wb_ref[:-1, :].T, atol=1e-5)
    assert np.allclose(biases, Wb_ref[-1, :], atol=1e-5)
    assert np.allclose(P, cov_ref, atol=1e-5)


@pytest.mark.parametrize("warm_with_lesson", [False, True])
def test_cache_resolves_default_recency_before_lookup_and_learning(warm_with_lesson):
    schema = DecisionSchema(fields={"route": ChoiceField(options=["refund", "password"])})
    projector = DeterministicSemanticProjector(dimension=32, recency_weighted=True)
    prompt = "refund payment invoice password login reset"
    recent = projector.project(prompt, recency_weighted=True)
    uniform = projector.project(prompt, recency_weighted=False)
    recent /= np.linalg.norm(recent)
    uniform /= np.linalg.norm(uniform)
    # Separate the two real feature vectors so a context collision flips the label.
    weight = 20 * (recent - uniform) / np.dot(recent - uniform, recent - uniform)
    bias = -float(weight @ ((recent + uniform) / 2))
    head = CompiledHeadWeights(
        "route", "choice", np.stack([weight, -weight]), np.array([bias, -bias], dtype=np.float32),
        options=("refund", "password"), score_method="lac", calibration_scores=(0.1,) * 20,
    )
    model = CompiledSystemOneModel(schema, {"route": head}, dimension=32,
                                  projector=projector, recency_weighted=True)
    engine = SystemOneEngine(schema, model=model, strict_mode=True, backend="numpy")
    if warm_with_lesson:
        engine.learn(prompt, {"route": "refund"})
    default = engine.decide(prompt, record_receipt=False)
    assert default.values == {"route": "refund"} and not default.is_ambiguous
    other = engine.decide(prompt, recency_weighted=False, record_receipt=False)
    explicit_default = engine.decide(prompt, recency_weighted=True, record_receipt=False)
    assert explicit_default.is_cache_hit
    engine.use_cache = False
    uncached = engine.decide(prompt, recency_weighted=False, record_receipt=False)
    assert not other.is_cache_hit
    assert other.values == uncached.values
    assert other.values == {"route": "password"}
    assert other.probabilities == uncached.probabilities
    assert other.is_ambiguous == uncached.is_ambiguous


@pytest.mark.parametrize("telemetry", [{"risk": 1.0}, [1.0], np.array([1.0])])
def test_semantic_cache_requires_matching_telemetry_values(telemetry):
    cache = SemanticSystemOneCache()
    vector = np.array([1.0, 0.0], dtype=np.float32)
    cache.put("old prompt", {"state": "old"}, embedding=vector, telemetry=telemetry)
    changed = {"risk": 2.0} if isinstance(telemetry, dict) else [2.0]
    assert cache.get("new prompt", embedding=vector, telemetry=changed) is None
    # Semantic reuse within the same observed state is still supported.
    hit = cache.get("new prompt", embedding=vector, telemetry=telemetry)
    assert hit is not None and hit[0].result == {"state": "old"}


@pytest.mark.parametrize("setting", ["margin_threshold", "confidence_floor_tau0"])
def test_cache_distinguishes_automatic_acceptance_gate_from_zero(setting):
    schema = DecisionSchema(fields={"route": ChoiceField(options=["refund", "password"])})
    head = CompiledHeadWeights(
        "route", "choice", np.zeros((2, 2), dtype=np.float32),
        0.25 * np.log(np.array([0.8, 0.2], dtype=np.float32)), options=("refund", "password"),
        margin_threshold=0.2, confidence_floor_tau0=0.4,
    )
    model = CompiledSystemOneModel(schema, {"route": head}, dimension=2,
                                  projector=TfidfProjector(("refund", "password"), (1.0, 1.0)))
    engine = SystemOneEngine(schema, model=model, backend="numpy",
                            confidence_floor_tau0=None if setting == "confidence_floor_tau0" else 0.15)
    accepted_kwargs = {setting: 0.0} if setting == "confidence_floor_tau0" else {}
    review_kwargs = {setting: 0.0} if setting == "margin_threshold" else {}
    accepted = engine.decide("refund", record_receipt=False, **accepted_kwargs)
    assert not accepted.is_ambiguous and accepted.margin_gate_active["route"]
    result = engine.decide("refund", record_receipt=False, **review_kwargs)
    engine.use_cache = False
    uncached = engine.decide("refund", record_receipt=False, **review_kwargs)
    assert uncached.is_ambiguous and not uncached.margin_gate_active["route"]
    assert result.is_ambiguous == uncached.is_ambiguous
    assert result.margin_gate_active == uncached.margin_gate_active
    assert not result.is_cache_hit


def test_hot_reload_discards_retired_lessons_and_keeps_model_version(tmp_path):
    schema = DecisionSchema(fields={"route": ChoiceField(options=["refund", "password"])})
    projector = TfidfProjector(("refund", "password"), (1.0, 1.0))
    head = CompiledHeadWeights("route", "choice", np.eye(2, dtype=np.float32),
                              np.zeros(2, dtype=np.float32), options=("refund", "password"))
    replacement = CompiledSystemOneModel(schema, {"route": head}, dimension=2, projector=projector)
    path = tmp_path / "skill.s1m"
    replacement.save(path)
    engine = SystemOneEngine(schema, model_path=path, auto_reload=True, backend="numpy",
                            strict_mode=True)
    engine.learn("zxqv", {"route": "password"})
    assert not engine.decide("zxqv", record_receipt=False).is_ambiguous
    previous_version = engine.model_version
    replacement.learn("different oov prompt", {"route": "refund"})
    replacement.save(path)
    assert engine.reload_if_changed()
    version = engine.model_version
    assert version > previous_version
    fresh = SystemOneEngine(schema, model_path=path, strict_mode=True, backend="numpy", use_cache=False)
    for prompt in ("zxqv", "different oov prompt"):
        result = engine.decide(prompt, record_receipt=False)
        expected = fresh.decide(prompt, record_receipt=False)
        assert result.values == expected.values
        assert result.is_ambiguous == expected.is_ambiguous
        assert result.conformal_sets == expected.conformal_sets
        assert engine.model_version == engine.model.model_version == version
    restored = SystemOneEngine.from_bytes(engine.to_bytes(), backend="numpy")
    assert "zxqv" not in restored.model._learned_prompt_overrides
    assert restored.model._learned_prompt_overrides == replacement._learned_prompt_overrides


def test_cleared_lesson_is_not_resurrected_from_saved_metadata():
    schema = DecisionSchema(fields={"route": ChoiceField(options=["refund", "password"])})
    engine = SystemOneEngine(schema, dimension=32, backend="numpy", strict_mode=True, use_cache=False)
    prompt = "refund payment invoice"
    engine.learn(prompt, {"route": "refund"})
    engine = SystemOneEngine.from_bytes(engine.to_bytes(), backend="numpy", use_cache=False)
    engine.learn_from_tier2(prompt, {"route": "password"}, preserve_calibration=False)
    assert not engine.model._learned_prompt_overrides
    expected = engine.decide(prompt, record_receipt=False)
    for _ in range(2):
        engine = SystemOneEngine.from_bytes(engine.to_bytes(), backend="numpy", use_cache=False)
        assert not engine.model._learned_prompt_overrides
        actual = engine.decide(prompt, record_receipt=False)
        assert actual.values == expected.values
        assert actual.is_ambiguous == expected.is_ambiguous
        assert actual.conformal_sets == expected.conformal_sets


def test_model_evaluate_cache_isolates_recency_weighted():
    schema = DecisionSchema(fields={"route": ChoiceField(options=["refund", "password"])})
    projector = DeterministicSemanticProjector(dimension=32, recency_weighted=True)
    prompt = "refund payment invoice password login reset"
    recent = projector.project(prompt, recency_weighted=True)
    uniform = projector.project(prompt, recency_weighted=False)
    recent /= np.linalg.norm(recent)
    uniform /= np.linalg.norm(uniform)
    weight = 20 * (recent - uniform) / np.dot(recent - uniform, recent - uniform)
    bias = -float(weight @ ((recent + uniform) / 2))
    head = CompiledHeadWeights(
        "route",
        "choice",
        np.stack([weight, -weight]),
        np.array([bias, -bias], dtype=np.float32),
        options=("refund", "password"),
        score_method="lac",
        calibration_scores=(0.1,) * 20,
    )
    model = CompiledSystemOneModel(
        schema,
        {"route": head},
        dimension=32,
        projector=projector,
        recency_weighted=True,
        use_cache=True,
    )
    res_recent = model.evaluate(prompt)
    assert res_recent.fields["route"].selected_value == "refund"

    res_uniform = model.evaluate(prompt, recency_weighted=False)
    assert res_uniform.fields["route"].selected_value == "password"

    batch_uniform = model.forward_batch([prompt], recency_weighted=False)
    assert batch_uniform[0].fields["route"].selected_value == "password"
    batch_recent = model.forward_batch([prompt], recency_weighted=True)
    assert batch_recent[0].fields["route"].selected_value == "refund"

    model.use_cache = False
    uncached_uniform = model.evaluate(prompt, recency_weighted=False)
    assert res_uniform.fields["route"].selected_value == uncached_uniform.fields["route"].selected_value

    model.use_cache = True
    model.learn(prompt, {"route": "refund"}, recency_weighted=True)
    cached_after_learn = model.evaluate(prompt, recency_weighted=True)
    uncached_after_learn = model.forward_single(prompt, recency_weighted=True, use_cache=False)
    assert cached_after_learn.fields["route"].selected_value == uncached_after_learn.fields["route"].selected_value
    assert np.allclose(
        cached_after_learn.fields["route"].raw_probabilities,
        uncached_after_learn.fields["route"].raw_probabilities,
    )


def test_compiler_auto_regularization_and_noise_robust_under_teacher_label_noise():
    from system1.compiler import SystemOneCompiler

    classes = ["billing", "technical", "account", "shipping", "security", "cancellation"]
    core_vocab = {
        "billing": ["invoice", "charge", "payment", "receipt", "overcharged", "statement", "credit", "fee"],
        "technical": ["crash", "stacktrace", "timeout", "database", "exception", "latency", "bug", "server"],
        "account": ["password", "mfa", "login", "locked", "username", "profile", "twofactor", "verification"],
        "shipping": ["delivery", "tracking", "package", "courier", "customs", "shipment", "warehouse", "transit"],
        "security": ["phishing", "breach", "unauthorized", "compromised", "token", "audit", "vulnerability", "firewall"],
        "cancellation": ["cancel", "unsubscribe", "terminate", "downgrade", "close", "discontinue", "optout", "renewal"],
    }
    rng = np.random.default_rng(20261014)
    train_texts, train_labels = [], []
    cal_rows, test_rows = [], []

    for c_idx, cls in enumerate(classes):
        words = core_vocab[cls]
        for i in range(20):
            chosen = rng.choice(words, size=4, replace=False)
            noise_tok = f"uniq_{c_idx}_{i}_{rng.integers(10000, 99999)}"
            shared_tok = f" distract_{rng.integers(0, 120)}"
            text = f"customer ticket {chosen[0]} {chosen[1]} {chosen[2]} {chosen[3]} {noise_tok} {shared_tok}"
            # Inject 15% teacher label corruption (3 of 20 per class)
            if i < 3:
                wrong_cls = classes[(c_idx + 1 + (i % (len(classes) - 1))) % len(classes)]
                label = wrong_cls
            else:
                label = cls
            train_texts.append(text)
            train_labels.append(label)

        for i in range(6):
            chosen = rng.choice(words, size=3, replace=False)
            cal_rows.append((f"customer ticket {chosen[0]} {chosen[1]} {chosen[2]}", cls))

        for i in range(25):
            chosen = rng.choice(words, size=2, replace=False)
            other_cls = classes[(c_idx + i) % len(classes)]
            cross_word = core_vocab[other_cls][i % len(core_vocab[other_cls])]
            test_rows.append((f"inquiry {chosen[0]} {chosen[1]} regarding {cross_word}", cls))

    projector = TfidfProjector.from_texts(
        train_texts,
        max_features=256,
        selection="chi2",
        labels=train_labels,
    )
    schema = DecisionSchema(fields={"intent": ChoiceField(options=classes)})
    fit_exemplars = {"intent": list(zip(train_texts, train_labels))}
    cal_exemplars = {"intent": cal_rows}

    robust_model = SystemOneCompiler(
        schema,
        projector=projector,
        regularization=0.02,
        auto_regularization=True,
        noise_robust=True,
    ).compile(fit_exemplars, calibration_exemplars=cal_exemplars, augment=False)

    fragile_model = SystemOneCompiler(
        schema,
        projector=projector,
        regularization=0.02,
        auto_regularization=False,
        noise_robust=False,
    ).compile(fit_exemplars, calibration_exemplars=cal_exemplars, augment=False)

    robust_acc = np.mean([
        robust_model.forward_single(t).fields["intent"].selected_value == y
        for t, y in test_rows
    ])
    fragile_acc = np.mean([
        fragile_model.forward_single(t).fields["intent"].selected_value == y
        for t, y in test_rows
    ])
    assert robust_acc >= fragile_acc + 0.05
    assert robust_acc >= 0.85

    autosplit_robust = SystemOneCompiler(
        schema,
        projector=projector,
        regularization=0.05,
        auto_regularization=True,
        noise_robust=True,
    ).compile(fit_exemplars, augment=False, calibration_split=0.25)
    autosplit_fragile = SystemOneCompiler(
        schema,
        projector=projector,
        regularization=0.05,
        auto_regularization=False,
        noise_robust=False,
    ).compile(fit_exemplars, augment=False, calibration_split=0.25)
    autosplit_robust_acc = np.mean([
        autosplit_robust.forward_single(t).fields["intent"].selected_value == y
        for t, y in test_rows
    ])
    autosplit_fragile_acc = np.mean([
        autosplit_fragile.forward_single(t).fields["intent"].selected_value == y
        for t, y in test_rows
    ])
    assert autosplit_robust_acc >= autosplit_fragile_acc
    assert max(autosplit_robust.heads["intent"].calibration_scores) > 0.50


def test_compiler_noise_robust_extreme_class_imbalance():
    from system1.compiler import SystemOneCompiler

    rng = np.random.default_rng(42)
    maj_words = ["invoice", "payment", "charge", "statement", "fee", "receipt", "billing", "credit", "card", "balance"]
    texts, labels = [], []
    for i in range(200):
        w = rng.choice(maj_words, size=3, replace=False)
        texts.append(f"request {w[0]} {w[1]} {w[2]} common")
        labels.append("billing")
    texts.append("request breach firewall phishing common")
    labels.append("security")
    texts.append("request malware ransomware exploit common")
    labels.append("security")

    proj = TfidfProjector.from_texts(texts, max_features=256)
    schema = DecisionSchema(fields={"intent": ChoiceField(options=["billing", "security"])})
    comp = SystemOneCompiler(schema, projector=proj, regularization=0.1)
    X = np.stack([proj.project(t) for t in texts])
    Y = np.zeros((len(texts), 2), dtype=np.float32)
    for i, l in enumerate(labels):
        Y[i, 0 if l == "billing" else 1] = 1.0
    _, sw_clean = comp._compute_loo_ridge_and_sample_weights(X, Y)
    assert sw_clean is None or np.allclose(sw_clean[-2:], 1.0)

    texts_noisy = list(texts[:-1]) + ["request breach firewall malware common", "request invoice payment charge common"]
    labels_noisy = list(labels[:-1]) + ["security", "security"]
    proj_noisy = TfidfProjector.from_texts(texts_noisy, max_features=256)
    X_n = np.stack([proj_noisy.project(t) for t in texts_noisy])
    Y_n = np.zeros((len(texts_noisy), 2), dtype=np.float32)
    for i, l in enumerate(labels_noisy):
        Y_n[i, 0 if l == "billing" else 1] = 1.0
    _, sw_noisy = comp._compute_loo_ridge_and_sample_weights(X_n, Y_n)
    assert sw_noisy is not None
    assert sw_noisy[-3] >= 0.85 and sw_noisy[-2] >= 0.85
    assert sw_noisy[-1] == pytest.approx(0.20, abs=1e-5)


def test_teaching_session_noise_robust_and_compile_under_teacher_corruption(tmp_path):
    from system1.teaching import TeachingSession

    classes = ["billing", "technical", "account", "shipping"]
    core_vocab = {
        "billing": ["invoice", "charge", "payment", "receipt", "overcharged", "statement"],
        "technical": ["crash", "stacktrace", "timeout", "database", "exception", "latency"],
        "account": ["password", "mfa", "login", "locked", "username", "profile"],
        "shipping": ["delivery", "tracking", "package", "courier", "customs", "shipment"],
    }
    rng = np.random.default_rng(99)
    schema = DecisionSchema(fields={"intent": ChoiceField(options=classes)})
    session = TeachingSession(
        tmp_path / "noisy_session",
        schema,
        regularization=0.02,
        max_features=128,
        selection="chi2",
        auto_regularization=True,
        noise_robust=True,
    )
    assert session.auto_regularization is True
    assert session.noise_robust is True
    snap_settings = session.snapshot()["settings"]
    assert snap_settings["auto_regularization"] is True
    assert snap_settings["noise_robust"] is True

    with pytest.raises(ValueError, match="auto_regularization must be a boolean"):
        TeachingSession(tmp_path / "noisy_session", auto_regularization=1)

    for c_idx, cls in enumerate(classes):
        words = core_vocab[cls]
        for i in range(15):
            chosen = rng.choice(words, size=3, replace=False)
            text = f"request {chosen[0]} {chosen[1]} {chosen[2]} ref_{c_idx}_{i}"
            label = classes[(c_idx + 1) % len(classes)] if i < 2 else cls
            session.record(text, label, split="teach", source="teacher")
        for i in range(4):
            chosen = rng.choice(words, size=2, replace=False)
            session.record(f"cal {chosen[0]} {chosen[1]} item_{c_idx}_{i}", cls, split="calibrate")
        for i in range(6):
            chosen = rng.choice(words, size=2, replace=False)
            session.record(f"eval {chosen[0]} {chosen[1]} check_{c_idx}_{i}", cls, split="evaluate")

    compiled = session.compile()
    assert isinstance(compiled, CompiledSystemOneModel)
    report = session.assess(min_accuracy=0.80, min_coverage=0.50, max_accepted_errors=2)
    assert report["candidate"]["raw_accuracy"] >= 0.85


def test_compiler_autosplit_never_overwrites_heldout_calibration_labels_with_argmax():
    from system1.compiler import SystemOneCompiler

    all_pairs = (
        [(f"alpha item_a_{i} billing invoice charge", "billing") for i in range(24)]
        + [(f"alpha edge_a_{i} billing invoice charge", "technical") for i in range(6)]
        + [(f"beta item_b_{i} technical crash stacktrace", "technical") for i in range(24)]
        + [(f"beta edge_b_{i} technical crash stacktrace", "billing") for i in range(6)]
    )
    texts = [t for t, _ in all_pairs]
    proj = TfidfProjector.from_texts(texts, max_features=128)

    schema = DecisionSchema(
        fields={
            "intent": ChoiceField(options=["billing", "technical"]),
            "urgent": BooleanField(),
        }
    )
    bool_pairs = [(t, y == "technical") for t, y in all_pairs]
    compiled = SystemOneCompiler(
        schema,
        projector=proj,
        regularization=0.1,
        auto_regularization=True,
        noise_robust=True,
    ).compile(
        {"intent": all_pairs, "urgent": bool_pairs},
        augment=False,
        calibration_split=0.25,
    )
    assert max(compiled.heads["intent"].calibration_scores) > 0.50
    assert max(compiled.heads["urgent"].calibration_scores) > 0.50


def test_tfidf_projector_is_truly_immutable_without_mutable_last_raw_norm():
    proj = TfidfProjector.from_texts(
        ["billing invoice charge", "technical crash stacktrace"],
        max_features=32,
    )
    assert not hasattr(proj, "_last_raw_norm")
    vec, norm = proj.project_with_norm("billing invoice")
    assert norm > 0.0
    assert np.allclose(vec, proj.project("billing invoice"))
    assert pytest.approx(norm, rel=1e-6) == proj.raw_norm("billing invoice")
    assert not hasattr(proj, "_last_raw_norm")


def test_compiled_model_online_learning_uses_real_head_and_scales_small_reward():
    from system1.compiler import SystemOneCompiler

    schema = DecisionSchema(fields={"route": ChoiceField(options=["billing", "tech"])})
    examples = {
        "route": [
            ("invoice refund billing charge", "billing"),
            ("billing statement payment receipt", "billing"),
            ("duplicate invoice billing refund", "billing"),
            ("overcharged billing card statement", "billing"),
            ("server crash stacktrace timeout", "tech"),
            ("database connection error bug", "tech"),
            ("kernel panic memory leak crash", "tech"),
            ("api gateway 502 timeout error", "tech"),
        ]
    }
    proj = TfidfProjector.from_texts([t for t, _ in examples["route"]], max_features=32)
    compiler = SystemOneCompiler(schema=schema, projector=proj, backend="numpy")
    model = compiler.compile(examples, augment=False, calibration_split=0.25)
    engine = SystemOneEngine(schema, model=model, strict_mode=True, backend="numpy")

    before_logits = model.forward_single("invoice refund billing charge").fields["route"].logits.copy()
    engine.learn("invoice refund billing charge", {"route": "tech"}, reward=0.01)
    after_small = engine.decide("invoice refund billing charge", record_receipt=False)
    assert after_small.values["route"] == "billing"
    after_logits = model.forward_single("invoice refund billing charge").fields["route"].logits.copy()
    assert not np.allclose(after_logits, np.array([-48.0, 48.0]))
    assert after_logits[1] > before_logits[1]


def test_engine_learn_from_tier2_does_not_cache_ambiguous_and_decide_syncs_model_updates():
    from system1.compiler import SystemOneCompiler

    schema = DecisionSchema(fields={"route": ChoiceField(options=["billing", "tech"])})
    examples = {
        "route": [
            ("invoice refund billing charge", "billing"),
            ("billing statement payment receipt", "billing"),
            ("duplicate invoice billing refund", "billing"),
            ("overcharged billing card statement", "billing"),
            ("server crash stacktrace timeout", "tech"),
            ("database connection error bug", "tech"),
            ("kernel panic memory leak crash", "tech"),
            ("api gateway 502 timeout error", "tech"),
        ]
    }
    proj = TfidfProjector.from_texts([t for t, _ in examples["route"]], max_features=32)
    model = SystemOneCompiler(schema=schema, projector=proj, backend="numpy").compile(
        examples, augment=False, calibration_split=0.25
    )
    engine = SystemOneEngine(schema, model=model, strict_mode=True, backend="numpy", use_cache=True)

    engine.learn_from_tier2("invoice refund billing charge", {"route": "tech"}, preserve_calibration=False)
    res_ambig = engine.decide("invoice refund billing charge", record_receipt=False)
    assert res_ambig.is_ambiguous
    assert not res_ambig.is_cache_hit

    engine2 = SystemOneEngine(schema, model=model, strict_mode=True, backend="numpy")
    engine2.model.learn("zxqv_oov_token", {"route": "tech"}, preserve_calibration=True)
    res_synced = engine2.decide("zxqv_oov_token", record_receipt=False)
    assert res_synced.values["route"] == "tech"
    assert not res_synced.is_ambiguous


def test_teaching_session_compile_without_explicit_calibrate_split_produces_calibrated_model(tmp_path):
    from system1.teaching import TeachingSession

    schema = DecisionSchema(fields={"route": ChoiceField(options=["billing", "tech"])})
    session = TeachingSession(tmp_path / "teach_only_session", schema, max_features=64)
    for i in range(6):
        session.record(f"invoice billing charge refund item_{i}", "billing", split="teach")
        session.record(f"server crash timeout stacktrace bug_{i}", "tech", split="teach")

    compiled = session.compile()
    assert len(compiled.heads["route"].calibration_scores) >= 2
    assert compiled.heads["route"].conformal_quantile > 0.0
    engine = SystemOneEngine(schema, model=compiled, strict_mode=True, backend="numpy")
    dec = engine.decide("invoice billing charge refund", alpha=0.35, record_receipt=False)
    assert dec.values["route"] == "billing"
    assert not dec.is_ambiguous


def test_cli_calibrate_loads_and_saves_compiled_s1m_model(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    from system1.compiler import SystemOneCompiler

    repo_root = Path(__file__).resolve().parents[1]
    schema = DecisionSchema(fields={"route": ChoiceField(options=["billing", "tech"])})
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema.to_dict()), encoding="utf-8")

    examples = {
        "route": [
            ("invoice refund billing charge", "billing"),
            ("billing statement payment receipt", "billing"),
            ("server crash stacktrace timeout", "tech"),
            ("database connection error bug", "tech"),
        ]
    }
    proj = TfidfProjector.from_texts([t for t, _ in examples["route"]], max_features=32)
    uncalibrated_model = SystemOneCompiler(schema=schema, projector=proj, backend="numpy").compile(
        examples, augment=False, calibration_split=0.0
    )
    assert len(uncalibrated_model.heads["route"].calibration_scores) == 0
    in_model_path = tmp_path / "uncalibrated.s1m"
    out_model_path = tmp_path / "calibrated.s1m"
    uncalibrated_model.save(in_model_path)

    calib_data = [
        {"prompt": f"invoice billing charge payment receipt {i}", "labels": {"route": "billing"}}
        for i in range(20)
    ] + [
        {"prompt": f"server crash timeout database error stacktrace {i}", "labels": {"route": "tech"}}
        for i in range(20)
    ]
    calib_path = tmp_path / "calib.json"
    calib_path.write_text(json.dumps(calib_data), encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "system1.cli",
            "calibrate",
            "--dataset",
            str(calib_path),
            "--model",
            str(in_model_path),
            "--output",
            str(out_model_path),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    reloaded = SystemOneEngine.load(out_model_path, strict_mode=True)
    assert len(reloaded.model.heads["route"].calibration_scores) >= 2
    dec = reloaded.decide("invoice refund billing charge", record_receipt=False)
    assert dec.values["route"] == "billing"
    assert not dec.is_ambiguous


def test_hybrid_projector_fit_maps_max_features_and_rejects_unknown_kwargs():
    from system1.core.embeddings import HybridProjector

    texts = ["billing invoice refund charge", "server crash stacktrace timeout"]
    proj = HybridProjector.fit(texts, labels=["billing", "tech"], max_features=128, dense_dim=32)
    assert proj.sparse_dim == 128
    assert proj.dense_dim == 32
    assert proj.dimension == 160

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        HybridProjector.fit(texts, labels=["billing", "tech"], nonexistent_option=True)


def test_prompt_guard_uses_disjoint_training_and_calibration_sets():
    from system1.integrations.drop_in import SystemOnePromptGuard

    guard = SystemOnePromptGuard(compiled=True)
    train_prompts = {p for p, _ in guard.training_rows}
    calib_prompts = {p for p, _ in guard.calibration_rows}
    assert len(train_prompts) >= 6
    assert len(calib_prompts) >= 6
    assert train_prompts.isdisjoint(calib_prompts)


def test_instructor_client_honors_per_call_response_model():
    from typing import Any, cast
    from system1.integrations.drop_in import SystemOneInstructorClient

    class CustomBillingSchema(DecisionSchema):
        intent = ChoiceField(
            options=["refund", "upgrade"],
            descriptions={
                "refund": "refund duplicate charge overcharge invoice",
                "upgrade": "upgrade enterprise plan add seats",
            },
        )
        urgent = BooleanField(
            true_description="urgent immediate blocked",
            false_description="routine standard",
        )

    client = SystemOneInstructorClient()
    res = client.create(
        "urgent refund for duplicate charge on my invoice",
        response_model=CustomBillingSchema,
    )
    assert set(res.values.keys()) == {"intent", "urgent"}
    assert res.values["intent"] == "refund"

    with pytest.raises(TypeError):
        client.create("refund my charge", response_model=cast(Any, dict))


def test_instinct_route_layer_preserves_fallback_answer_and_honors_strict_mode():
    from system1.integrations.drop_in import InstinctRouteLayer, SemanticRoute

    routes = [
        SemanticRoute(name="billing", utterances=["refund duplicate charge on invoice"]),
        SemanticRoute(name="tech", utterances=["kubernetes pod crashloop and 502 error"]),
    ]
    strict_router = InstinctRouteLayer(routes, strict_mode=True, compiled=False)
    res_strict = strict_router("refund duplicate charge on invoice")
    assert res_strict.name is None

    fallback_router = InstinctRouteLayer(
        routes,
        strict_mode=True,
        compiled=False,
        confidence_threshold=0.99,
        fallback_fn=lambda _text: "billing",
    )
    res_fb = fallback_router("completely novel ambiguous phrasing")
    assert res_fb.used_fallback is True
    assert res_fb.name == "billing"


def test_showcase_break_and_teach_matches_readme_and_preserves_oos_after_learn(tmp_path):
    import importlib.util
    from pathlib import Path

    showcase_path = Path(__file__).resolve().parents[1] / "examples" / "showcase_break_and_teach.py"
    spec = importlib.util.spec_from_file_location("showcase_break_and_teach", showcase_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    engine, s1m_bytes = mod.build_instinct_engine(mod.INITIAL_TEACHER_LOGS, tmp_path / "support_gateway.s1m")
    assert s1m_bytes > 0

    t1 = engine.decide("Please do NOT cancel my subscription, I want to upgrade to Pro instead", record_receipt=False)
    assert not t1.is_ambiguous
    assert t1.values["action"] == "upgrade_plan"

    t2 = engine.decide("What is the best temperature to bake sourdough bread in a Dutch oven?", record_receipt=False)
    assert t2.is_ambiguous

    t3 = engine.decide("chagred twcie on my vsia crad endign 9921, pls refnud", record_receipt=False)
    assert not t3.is_ambiguous
    assert t3.values["action"] == "refund_duplicate_charge"

    unseen_1 = "Customer is asking for a Project-Sunburst adjustment on their account"
    unseen_2 = "Please apply the Project-Sunburst adjustment for this workspace"
    before_d = engine.decide(unseen_1, record_receipt=False)
    assert before_d.is_ambiguous

    engine.learn(unseen_1, {"action": "refund_duplicate_charge", "urgent": True})
    after_d = engine.decide(unseen_2, record_receipt=False)
    assert not after_d.is_ambiguous
    assert after_d.values["action"] == "refund_duplicate_charge"
    assert after_d.values["urgent"] is True

    t2_after = engine.decide("What is the best temperature to bake sourdough bread in a Dutch oven?", record_receipt=False)
    assert t2_after.is_ambiguous


def test_dogfood_fda_and_github_no_calibration_or_test_cue_leakage(monkeypatch):
    import importlib.util
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    fda_path = repo_root / "examples" / "dogfood" / "live_fda_recall_triage.py"
    gh_path = repo_root / "examples" / "dogfood" / "live_github_triage.py"

    fda_spec = importlib.util.spec_from_file_location("live_fda_recall_triage", fda_path)
    assert fda_spec is not None and fda_spec.loader is not None
    fda_mod = importlib.util.module_from_spec(fda_spec)
    fda_spec.loader.exec_module(fda_mod)

    gh_spec = importlib.util.spec_from_file_location("live_github_triage", gh_path)
    assert gh_spec is not None and gh_spec.loader is not None
    gh_mod = importlib.util.module_from_spec(gh_spec)
    gh_spec.loader.exec_module(gh_mod)

    snapshot = fda_mod.load_or_fetch_snapshot()
    observed = snapshot["recalls"][:108]
    fit_doc_counts: list[int] = []
    orig_fit = fda_mod.TfidfProjector.fit

    def spy_fit(texts, *args, **kwargs):
        text_list = list(texts)
        fit_doc_counts.append(len(text_list))
        return orig_fit(text_list, *args, **kwargs)

    monkeypatch.setattr(fda_mod.TfidfProjector, "fit", spy_fit)
    _, compiled_fda = fda_mod._compile_checkpoint_engine(observed)
    assert len(fit_doc_counts) == 1
    assert fit_doc_counts[0] == compiled_fda.metadata["sample_counts"]["regulatory_center"]["fit"]
    assert fit_doc_counts[0] < len(observed)

    gh_source = gh_path.read_text(encoding="utf-8")
    fda_source = fda_path.read_text(encoding="utf-8")
    assert "npmscan|workflow|docker|fail|error|bug|crash|test|fix" not in gh_source
    assert "is_heldout_window = w_idx == len(uv_windows) - 1" in gh_source
    assert "is_heldout_window = w_idx == len(CHRONOLOGICAL_WINDOWS) - 1" in fda_source


def test_benchmark_and_docs_consistency():
    import importlib.util
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    eval_path = repo_root / "benchmarks" / "quality" / "evaluate_drop_in_replacements.py"
    eval_source = eval_path.read_text(encoding="utf-8")

    assert "_eval_k_shot_online(10)" in eval_source

    eval_spec = importlib.util.spec_from_file_location("evaluate_drop_in_replacements", eval_path)
    assert eval_spec is not None and eval_spec.loader is not None
    eval_mod = importlib.util.module_from_spec(eval_spec)
    eval_spec.loader.exec_module(eval_mod)
    neural_comp = eval_mod.evaluate_neural_embedding_comparison()
    assert neural_comp.get("evaluation_mode") in ("live_fastembed", "recorded_reference_without_fastembed")

    readme_text = (repo_root / "README.md").read_text(encoding="utf-8")
    assert "1,394" not in readme_text
    assert "24.41%" not in readme_text
    assert "7,031" in readme_text
    assert "1,076" in readme_text
    assert "27.68%" in readme_text
    assert "1.68%" in readme_text
    assert "1.37%" in readme_text

    ht_readme = (repo_root / "benchmarks" / "heldout_traffic" / "README.md").read_text(encoding="utf-8")
    ht_report = (repo_root / "benchmarks" / "heldout_traffic" / "REPORT.md").read_text(encoding="utf-8")
    assert "20,977 B" not in ht_readme
    assert "20,954 B" in ht_readme
    assert "14" in ht_report and "19" in ht_report


