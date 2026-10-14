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
from system1.core.model import DecisionFieldHead
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
