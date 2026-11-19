"""Contract for how SystemOneEngine chooses optional kwargs for model callables.

- Explicit parameters only: the engine passes exactly the candidate kwargs the callable names.
- ``**kwargs``: the engine passes every candidate kwarg; it does not inspect what the callable forwards.
- A ``**kwargs`` wrapper that blindly forwards to a stricter callable gets that callable's ``TypeError``.
- An unobtainable signature yields no optional kwargs.
"""

from __future__ import annotations

import functools
import gc
import threading
import types
import weakref
from typing import Any, Dict, List, Mapping

import numpy as np
import pytest

from system1.core import ChoiceField, DecisionSchema
from system1.core.model import SystemOneModel
from system1.engine import SystemOneEngine, _SIG_FACTS_CACHE, _resolve_supported_kwargs

CANDIDATES = {"telemetry": "t", "embedding": "e", "recency_weighted": False, "use_cache": False}


# ---------------------------------------------------------------------------
# _resolve_supported_kwargs
# ---------------------------------------------------------------------------


def test_explicit_signature_passes_only_named_candidates() -> None:
    def fn(prompt: str, telemetry: Any = None, *, use_cache: bool = True) -> None:
        return None

    kw, names = _resolve_supported_kwargs(fn, CANDIDATES)
    assert kw == {"telemetry": "t", "use_cache": False}
    assert names == frozenset({"prompt", "telemetry", "use_cache"})


def test_var_kwargs_passes_every_candidate() -> None:
    def fn(prompt: str, telemetry: Any = None, **kwargs: Any) -> None:
        return None

    kw, names = _resolve_supported_kwargs(fn, CANDIDATES)
    assert kw == CANDIDATES
    assert kw is not CANDIDATES
    assert names == frozenset(CANDIDATES)


def test_unobtainable_signature_yields_no_kwargs() -> None:
    def bogus_sig(prompt: str, telemetry: Any = None) -> None:
        return None

    bogus_sig.__signature__ = "not a signature"  # type: ignore[attr-defined]
    assert _resolve_supported_kwargs(bogus_sig, CANDIDATES) == ({}, frozenset())

    class NoArgs:
        def method():  # type: ignore[misc]  # binding leaves no parameter to drop
            return None

    assert _resolve_supported_kwargs(NoArgs().method, CANDIDATES) == ({}, frozenset())
    # Same answer from the cached path on a second lookup.
    assert _resolve_supported_kwargs(NoArgs().method, CANDIDATES) == ({}, frozenset())


def test_partial_callable_instance_staticmethod_classmethod() -> None:
    def base(prompt: str, telemetry: Any = None, embedding: Any = None) -> None:
        return None

    kw, _ = _resolve_supported_kwargs(functools.partial(base, telemetry=None), CANDIDATES)
    assert kw == {"telemetry": "t", "embedding": "e"}

    class CallableInstance:
        def __call__(self, prompt: str, embedding: Any = None) -> None:
            return None

    kw, names = _resolve_supported_kwargs(CallableInstance(), CANDIDATES)
    assert kw == {"embedding": "e"}
    assert names == frozenset({"prompt", "embedding"})

    class Holder:
        @staticmethod
        def static(prompt: str, use_cache: bool = True) -> None:
            return None

        @classmethod
        def klass(cls, prompt: str, recency_weighted: Any = None) -> None:
            return None

    assert _resolve_supported_kwargs(Holder().static, CANDIDATES)[0] == {"use_cache": False}
    assert _resolve_supported_kwargs(Holder().klass, CANDIDATES)[0] == {"recency_weighted": False}
    assert _resolve_supported_kwargs(Holder.klass, CANDIDATES)[1] == frozenset({"prompt", "recency_weighted"})


def test_wrapped_function_follows_wrapped_signature() -> None:
    def inner(prompt: str, embedding: Any = None) -> None:
        return None

    @functools.wraps(inner)
    def outer(*args: Any, **kwargs: Any) -> None:
        return inner(*args, **kwargs)

    # inspect.signature follows __wrapped__, so the explicit inner signature governs.
    assert _resolve_supported_kwargs(outer, CANDIDATES)[0] == {"embedding": "e"}


# ---------------------------------------------------------------------------
# Cache correctness
# ---------------------------------------------------------------------------


def test_cache_distinguishes_bound_and_unbound_use_of_one_function() -> None:
    def fn(telemetry: Any = None, embedding: Any = None) -> None:
        return None

    owner = object()
    bound = types.MethodType(fn, owner)  # binding consumes ``telemetry``
    for _ in range(2):
        assert _resolve_supported_kwargs(fn, CANDIDATES)[0] == {"telemetry": "t", "embedding": "e"}
        assert _resolve_supported_kwargs(bound, CANDIDATES)[0] == {"embedding": "e"}
        assert _resolve_supported_kwargs(types.MethodType(fn, owner), CANDIDATES)[0] == {"embedding": "e"}

    class Model:
        def forward_single(self, prompt: str, embedding: Any = None) -> None:
            return None

    for _ in range(2):
        assert _resolve_supported_kwargs(Model().forward_single, CANDIDATES)[1] == frozenset({"prompt", "embedding"})
        assert _resolve_supported_kwargs(Model.forward_single, CANDIDATES)[1] == frozenset(
            {"self", "prompt", "embedding"}
        )


def test_cache_keeps_classes_sharing_a_method_name_apart() -> None:
    class A:
        def forward_single(self, prompt: str, embedding: Any = None) -> None:
            return None

    class B:
        def forward_single(self, prompt: str, telemetry: Any = None) -> None:
            return None

    class C:
        def forward_single(self, prompt: str, **kwargs: Any) -> None:
            return None

    for _ in range(2):
        assert _resolve_supported_kwargs(A().forward_single, CANDIDATES)[0] == {"embedding": "e"}
        assert _resolve_supported_kwargs(B().forward_single, CANDIDATES)[0] == {"telemetry": "t"}
        assert _resolve_supported_kwargs(C().forward_single, CANDIDATES)[0] == CANDIDATES


def test_cache_respects_each_call_candidate_set() -> None:
    def explicit(prompt: str, a: Any = None, b: Any = None) -> None:
        return None

    def varkw(prompt: str, **kwargs: Any) -> None:
        return None

    assert _resolve_supported_kwargs(explicit, {"a": 1, "c": 3}) == ({"a": 1}, frozenset({"prompt", "a", "b"}))
    assert _resolve_supported_kwargs(explicit, {"b": 2, "c": 3}) == ({"b": 2}, frozenset({"prompt", "a", "b"}))
    assert _resolve_supported_kwargs(varkw, {"x": 1}) == ({"x": 1}, frozenset({"x"}))
    assert _resolve_supported_kwargs(varkw, {"y": 2}) == ({"y": 2}, frozenset({"y"}))


def test_cache_tracks_code_and_signature_changes() -> None:
    def fn(prompt: str, embedding: Any = None) -> None:
        return None

    def other(prompt: str, telemetry: Any = None) -> None:
        return None

    assert _resolve_supported_kwargs(fn, CANDIDATES)[0] == {"embedding": "e"}
    fn.__code__ = other.__code__
    assert _resolve_supported_kwargs(fn, CANDIDATES)[0] == {"telemetry": "t"}
    fn.__signature__ = __import__("inspect").signature(lambda prompt, use_cache=True: None)  # type: ignore[attr-defined]
    assert _resolve_supported_kwargs(fn, CANDIDATES)[0] == {"use_cache": False}


def test_cache_does_not_keep_models_alive() -> None:
    def make() -> Any:
        class Ephemeral:
            def forward_single(self, prompt: str, embedding: Any = None) -> None:
                return super().forward_single  # type: ignore[misc]  # __class__ cell refers back to the class

        return Ephemeral()

    inst = make()
    assert _resolve_supported_kwargs(inst.forward_single, CANDIDATES)[0] == {"embedding": "e"}
    func = type(inst).forward_single
    assert func in _SIG_FACTS_CACHE
    inst_ref, cls_ref, func_ref = weakref.ref(inst), weakref.ref(type(inst)), weakref.ref(func)
    del inst, func
    gc.collect()
    assert inst_ref() is None
    assert cls_ref() is None
    assert func_ref() is None


def test_concurrent_resolution_is_consistent() -> None:
    classes = []
    for i in range(16):
        name = ("embedding", "telemetry", "use_cache", "recency_weighted")[i % 4]
        ns: Dict[str, Any] = {}
        exec(f"def forward_single(self, prompt, {name}=None):\n    return None", ns)
        classes.append((type(f"M{i}", (), {"forward_single": ns["forward_single"]}), name))

    errors: List[str] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()
        for _ in range(200):
            for cls, name in classes:
                kw, names = _resolve_supported_kwargs(cls().forward_single, CANDIDATES)
                if kw != {name: CANDIDATES[name]} or names != frozenset({"prompt", name}):
                    errors.append(f"{cls.__name__}: {kw} {names}")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


# ---------------------------------------------------------------------------
# Engine behaviour
# ---------------------------------------------------------------------------


class _Coupled(DecisionSchema):
    f = ChoiceField(options=("a", "b"), description="First")
    g = ChoiceField(options=("a", "b"), description="Second")


class _Single(DecisionSchema):
    f = ChoiceField(options=("a", "b"), description="Single")


X = np.ones(16, dtype=np.float32)


class _StrictBase:
    """Duck model with explicit, narrow signatures."""

    def __init__(self, schema: DecisionSchema) -> None:
        self._inner = SystemOneModel(schema, dimension=16, backend="numpy", contrastive_whitening=False)
        for h in self._inner.heads.values():
            h.set_weights(np.zeros((2, 16), dtype="f4"), np.array([2.0, -2.0], dtype="f4"))
        self.schema = self._inner.schema
        self.projector = self._inner.projector
        self.dimension = 16
        self.heads = self._inner.heads

    def encode(self, prompt: str, recency_weighted: Any = None) -> np.ndarray:
        return self._inner.encode(prompt, recency_weighted=recency_weighted)

    def forward_single(self, prompt: str, embedding: Any = None) -> Any:
        return self._inner.forward_single(prompt, embedding=embedding)

    def learn_from_tier2(self, prompt: str, targets: Mapping[str, Any], embedding: Any = None) -> Any:
        return self._inner.learn_from_tier2(prompt, targets, embedding=embedding)


def _engine(model: Any, schema: type = _Single) -> SystemOneEngine:
    return SystemOneEngine(
        schema,
        model=model,
        dimension=16,
        backend="numpy",
        contrastive_whitening=False,
        use_cache=False,
        strict_mode=False,
    )


def _head_state(model: _StrictBase) -> Dict[str, Any]:
    return {k: (h.weights.copy(), h.biases.copy()) for k, h in model._inner.heads.items()}


def test_explicit_signature_model_works_through_decide_and_learn() -> None:
    eng = _engine(_StrictBase(_Single()))
    assert eng.decide("p", embedding=X, record_receipt=False).values == {"f": "a"}
    eng.learn("p", {"f": "b"}, embedding=X)
    assert eng.decide("p", embedding=X, record_receipt=False).values == {"f": "b"}


def test_var_kwargs_model_receives_every_candidate() -> None:
    seen: List[Dict[str, Any]] = []

    class Recording(_StrictBase):
        def forward_single(self, prompt: str, **kwargs: Any) -> Any:
            seen.append(dict(kwargs))
            return super().forward_single(prompt, embedding=kwargs.get("embedding"))

    eng = _engine(Recording(_Single()))
    assert eng.decide("p", embedding=X, record_receipt=False).values == {"f": "a"}
    assert set(seen[-1]) == {"telemetry", "embedding", "recency_weighted", "use_cache"}
    assert seen[-1]["use_cache"] is False


def test_strict_base_forwarder_raises_type_error_from_decide() -> None:
    class BlindForward(_StrictBase):
        def forward_single(self, *args: Any, **kwargs: Any) -> Any:
            return super().forward_single(*args, **kwargs)

    eng = _engine(BlindForward(_Single()))
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        eng.decide("p", embedding=X, record_receipt=False)


def test_strict_base_forwarder_raises_type_error_from_learn_without_partial_update() -> None:
    class BlindLearn(_StrictBase):
        def __init__(self, schema: DecisionSchema) -> None:
            super().__init__(schema)
            self.calls = 0

        def learn_from_tier2(self, prompt: str, targets: Mapping[str, Any], **kwargs: Any) -> Any:
            self.calls += 1
            for h in self._inner.heads.values():  # rebind head state before the strict base rejects the kwargs
                h.biases = np.zeros_like(h.biases)
            return super().learn_from_tier2(prompt, targets, **kwargs)

    model = BlindLearn(_Single())
    eng = _engine(model)
    before = _head_state(model)
    version, lessons = eng.model_version, getattr(eng, "_online_lesson_count", 0)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        eng.learn("p", {"f": "b"}, embedding=X)

    assert eng.model_version == version
    assert getattr(eng, "_online_lesson_count", 0) == lessons
    assert model.calls == 0
    for k, (w, b) in before.items():
        np.testing.assert_array_equal(model._inner.heads[k].weights, w)
        np.testing.assert_array_equal(model._inner.heads[k].biases, b)
    assert eng.decide("p", embedding=X, record_receipt=False).values == {"f": "a"}


def test_strict_base_forwarder_in_post_update_validation_rolls_back_learn() -> None:
    class BlindForwardAfterLearn(_StrictBase):
        armed = False

        def forward_single(self, *args: Any, **kwargs: Any) -> Any:
            if not self.armed:
                return super().forward_single(args[0], embedding=kwargs.get("embedding"))
            return super().forward_single(*args, **kwargs)

    model = BlindForwardAfterLearn(_Single())
    eng = _engine(model)
    assert eng.decide("p", embedding=X, record_receipt=False).values == {"f": "a"}
    before = _head_state(model)
    version = eng.model_version

    model.armed = True
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        eng.learn("p", {"f": "b"}, embedding=X)  # update applies, then post-update decide() raises

    assert eng.model_version == version
    for k, (w, b) in before.items():
        np.testing.assert_array_equal(model._inner.heads[k].weights, w)
        np.testing.assert_array_equal(model._inner.heads[k].biases, b)
    model.armed = False
    assert eng.decide("p", embedding=X, record_receipt=False).values == {"f": "a"}
