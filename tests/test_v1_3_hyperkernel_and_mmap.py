"""Wave 2 verification tests for v1.3: fast deterministic projection, packed kernel,
FastDecisionCallable, and S1M\\x03 page-aligned zero-copy mmap serialization."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

from system1 import (
    BooleanField,
    ChoiceField,
    CompiledSystemOneModel,
    DecisionField,
    DecisionSchema,
    FastDecisionCallable,
    FastDecisionOutput,
    HybridProjector,
    MAGIC_HEADER_V3,
    MultiChoiceField,
    PackedMultiQuestionKernel,
    ScoreField,
    SystemOneCompiler,
    SystemOneEngine,
    compile_engine_callable,
    compile_typesafe_callable,
)
from system1.compat.typesafe import Choice, Noul, Score, TypeSafeClient
from system1.compiler import MAGIC_HEADER, MAX_SKILL_FIELDS_V3, S1M_V3_PAGE_SIZE
from system1.core.model import DeterministicSemanticProjector, _fast_deterministic_project


def _make_mixed_schema(n_choice: int, n_bool: int, n_score: int, *, mondrian: bool = False) -> DecisionSchema:
    fields: Dict[str, DecisionField] = {}
    for i in range(n_choice):
        fields[f"c_{i}"] = ChoiceField(
            options=["alpha", "beta", "gamma"],
            descriptions={
                "alpha": f"alpha topic {i} invoice payment refund",
                "beta": f"beta topic {i} outage error timeout",
                "gamma": f"gamma topic {i} access login mfa",
            },
        )
    for i in range(n_bool):
        fields[f"b_{i}"] = BooleanField(
            true_description=f"critical urgent spike {i}",
            false_description=f"routine benign docs {i}",
        )
    for i in range(n_score):
        fields[f"s_{i}"] = ScoreField(
            min_value=0.0,
            max_value=10.0,
            low_description=f"low benign severity {i}",
            high_description=f"critical severe outage {i}",
        )
    return DecisionSchema(fields=fields, schema_name=f"MixedQ{n_choice + n_bool + n_score}")


def _make_mixed_exemplars(
    schema: DecisionSchema, *, samples_per_choice: int = 8
) -> Dict[str, List[Tuple[str, object]]]:
    exemplars: Dict[str, List[Tuple[str, object]]] = {}
    for fname, fdef in schema.fields.items():
        rows: List[Tuple[str, object]] = []
        if isinstance(fdef, ChoiceField):
            for idx in range(samples_per_choice):
                rows.append((f"invoice payment refund alpha case {idx} {fname}", "alpha"))
                rows.append((f"gateway outage timeout 503 beta case {idx} {fname}", "beta"))
                rows.append((f"sso password mfa 403 gamma case {idx} {fname}", "gamma"))
        elif isinstance(fdef, BooleanField):
            for idx in range(samples_per_choice):
                rows.append((f"urgent critical emergency p0 spike {idx} {fname}", True))
                rows.append((f"routine informational docs note {idx} {fname}", False))
        elif isinstance(fdef, ScoreField):
            for idx in range(samples_per_choice):
                rows.append((f"critical severe incident alert {idx} {fname}", 9.0))
                rows.append((f"benign low impact note {idx} {fname}", 1.5))
        exemplars[fname] = rows
    return exemplars


def test_fast_deterministic_project_parity_and_validation() -> None:
    """_fast_deterministic_project matches DeterministicSemanticProjector.project across modes."""
    prompts = [
        "",
        "   ",
        "!!! --- ???",
        "a",
        "customer charged twice on invoice refund request 403",
        "URGENT gateway 503 timeout & latency spike on payment cluster!",
        " ".join(f"token_{i}_signal" for i in range(75)),
    ]
    for rw in (False, True):
        proj = DeterministicSemanticProjector(dimension=128, recency_weighted=rw)
        for p in prompts:
            ref = proj.project(p)
            fast = _fast_deterministic_project(proj, p)
            assert fast.dtype == np.float32
            assert fast.shape == (128,)
            assert np.allclose(fast, ref, atol=1e-6), f"Mismatch for rw={rw}, prompt={p!r}"
            assert abs(float(np.linalg.norm(fast)) - 1.0) < 1e-5
            if p.strip() and any(ch.isalnum() for ch in p):
                # Re-running with explicit opposite recency_weighted override also matches.
                ref_flip = proj.project(p, recency_weighted=not rw)
                fast_flip = _fast_deterministic_project(proj, p, recency_weighted=not rw)
                assert np.allclose(fast_flip, ref_flip, atol=1e-6)

    with pytest.raises(TypeError):
        _fast_deterministic_project(DeterministicSemanticProjector(dimension=64), 123)  # type: ignore[arg-type]


@pytest.mark.parametrize("q_total", [16, 64, 256])
@pytest.mark.parametrize("score_method", ["aps", "lac"])
@pytest.mark.parametrize("strict_mode", [True, False])
def test_packed_multi_question_kernel_parity(
    q_total: int, score_method: str, strict_mode: bool
) -> None:
    """PackedMultiQuestionKernel matches SystemOneEngine.decide for Q=16/64/256, strict/non-strict, APS/LAC."""
    if q_total == 16:
        n_c, n_b, n_s = 6, 6, 4
    elif q_total == 64:
        n_c, n_b, n_s = 24, 24, 16
    else:
        n_c, n_b, n_s = 96, 96, 64

    schema = _make_mixed_schema(n_c, n_b, n_s)
    exemplars = _make_mixed_exemplars(schema, samples_per_choice=6)
    cal_exemplars = _make_mixed_exemplars(schema, samples_per_choice=6)

    proj = DeterministicSemanticProjector(dimension=128)
    compiler = SystemOneCompiler(
        schema,
        projector=proj,
        dimension=128,
        mondrian=False,
    )
    compiled = compiler.compile(
        exemplars,
        calibration_split=0.0,
        augment=False,
        mondrian=False,
    )
    for h in compiled.heads.values():
        h.score_method = score_method
    eng = SystemOneEngine(
        schema,
        compiled_model=compiled,
        strict_mode=strict_mode,
        use_cache=False,
    )
    # Build aligned multi-field calibration dataset so conformal scores use the requested score_method.
    cal_prompts_rows: List[Tuple[str, Dict[str, object]]] = []
    n_cal_rows = max(len(v) for v in cal_exemplars.values())
    ref_rows = next(iter(cal_exemplars.values()))
    for idx in range(n_cal_rows):
        p_text = str(ref_rows[idx % len(ref_rows)][0])
        row_labels = {fname: rows[idx % len(rows)][1] for fname, rows in cal_exemplars.items()}
        cal_prompts_rows.append((p_text, row_labels))
    eng.calibrate(cal_prompts_rows)

    test_prompts = [
        "invoice payment refund alpha case",
        "urgent critical gateway 503 outage timeout",
        "routine informational docs question",
        "sso password mfa 403 ambiguous blend emergency invoice",
    ]

    for storage_mode in ("float32", "float16_upcast"):
        kernel = eng.compile_packed_kernel(storage_mode=storage_mode)
        assert isinstance(kernel, PackedMultiQuestionKernel)
        batch_out = kernel.decide_batch(test_prompts)
        atol_conf = 1e-4 if storage_mode == "float32" else 2e-2

        for row_idx, prompt in enumerate(test_prompts):
            ref = eng.decide(prompt, record_receipt=False)
            single_out = kernel.decide_prompt(prompt)

            if storage_mode == "float32":
                assert bool(single_out["is_ambiguous"][0]) is ref.is_ambiguous
                assert bool(batch_out["is_ambiguous"][row_idx]) is ref.is_ambiguous

            for ci, cname in enumerate(kernel.choice_names):
                pred_choice = kernel.c_options[ci][int(single_out["best_idx"][0, ci])]
                conf_val = float(single_out["c_conf"][0, ci])
                if storage_mode == "float32":
                    assert pred_choice == ref.values[cname]
                    assert float(batch_out["c_conf"][row_idx, ci]) == pytest.approx(conf_val, abs=1e-6)
                assert conf_val == pytest.approx(ref.confidences[cname], abs=atol_conf)

            for bi, bname in enumerate(kernel.bool_names):
                pred_bool = bool(single_out["b_val"][0, bi])
                conf_val = float(single_out["b_conf"][0, bi])
                if storage_mode == "float32":
                    assert pred_bool is bool(ref.values[bname])
                assert conf_val == pytest.approx(ref.confidences[bname], abs=atol_conf)

            for si, sname in enumerate(kernel.score_names):
                pred_score = float(single_out["s_val"][0, si])
                conf_val = float(single_out["s_conf"][0, si])
                atol_score = 1e-4 if storage_mode == "float32" else 0.05
                assert pred_score == pytest.approx(float(ref.values[sname]), abs=atol_score)
                assert conf_val == pytest.approx(ref.confidences[sname], abs=atol_conf)


def test_packed_kernel_non_strict_margin_gating_and_unsupported_checks() -> None:
    """Verify non-strict margin gating in PackedMultiQuestionKernel and rejection of unsupported features."""
    schema = DecisionSchema(
        fields={
            "route": ChoiceField(["a", "b", "c"]),
            "flag": BooleanField(),
        }
    )
    compiled = SystemOneCompiler(schema, dimension=64).compile(
        {
            "route": [("alpha a", "a"), ("beta b", "b"), ("gamma c", "c")] * 4,
            "flag": [("yes true", True), ("no false", False)] * 6,
        },
        augment=False,
    )
    eng = SystemOneEngine(
        schema,
        compiled_model=compiled,
        strict_mode=False,
        margin_threshold=0.99,
        use_cache=False,
    )
    kernel = PackedMultiQuestionKernel.from_engine(eng)
    ref = eng.decide("neutral unknown text", record_receipt=False)
    out = kernel.decide_prompt("neutral unknown text")
    assert bool(out["is_ambiguous"][0]) is True
    assert bool(out["is_ambiguous"][0]) is ref.is_ambiguous

    # Reject MultiChoiceField
    mc_schema = DecisionSchema(fields={"tags": MultiChoiceField(["x", "y"])})
    mc_eng = SystemOneEngine(mc_schema, strict_mode=False, use_cache=False)
    with pytest.raises(ValueError, match="does not support"):
        mc_eng.compile_packed_kernel()

    # Reject Mondrian class_calibration_scores
    comp_mondrian = SystemOneCompiler(schema, dimension=64, mondrian=True).compile(
        {
            "route": [("alpha a", "a"), ("beta b", "b"), ("gamma c", "c")] * 5,
            "flag": [("yes true", True), ("no false", False)] * 8,
        },
        calibration_exemplars={
            "route": [("alpha a cal", "a"), ("beta b cal", "b"), ("gamma c cal", "c")] * 4,
            "flag": [("yes true cal", True), ("no false cal", False)] * 4,
        },
        augment=False,
        mondrian=True,
    )
    mon_eng = SystemOneEngine(schema, compiled_model=comp_mondrian, strict_mode=True)
    with pytest.raises(ValueError, match="Mondrian"):
        mon_eng.compile_packed_kernel()


def test_fast_decision_callable_engine_typesafe_and_cache_invalidation() -> None:
    """FastDecisionCallable works via engine & TypeSafeClient and invalidates across mutations."""
    schema = DecisionSchema(
        fields={
            "route": ChoiceField(["billing", "incident", "access"]),
            "urgent": BooleanField(),
            "severity": ScoreField(min_value=0.0, max_value=1.0),
        }
    )
    compiled = SystemOneCompiler(schema, dimension=64).compile(
        {
            "route": [
                ("invoice refund charge", "billing"),
                ("server outage 503", "incident"),
                ("password sso 403", "access"),
            ]
            * 5,
            "urgent": [("server outage 503", True), ("password reset help", False)] * 7,
            "severity": [("server outage 503", 0.95), ("password reset help", 0.20)] * 7,
        },
        augment=False,
    )
    eng = SystemOneEngine(schema, compiled_model=compiled, strict_mode=True, use_cache=False)

    fn = compile_engine_callable(eng)
    assert isinstance(fn, FastDecisionCallable)
    out = fn("server outage 503")
    assert isinstance(out, FastDecisionOutput)
    d = out.to_dict()
    assert set(d) == {"values", "confidences", "is_ambiguous"}
    assert out.values["route"] == "incident"
    assert out.values["urgent"] is True

    # Embedding fast-path produces exact same result
    emb = eng.projector.project("server outage 503")
    out_emb = fn(embedding=emb)
    assert out_emb.values == out.values

    # Cache invalidation on learn(), calibrate(), adapt_zero_shot(), recalibrate()
    assert eng._packed_kernel_cache is not None
    eng.learn("novel token xyzzy", {"route": "access", "urgent": False, "severity": 0.1})
    assert eng._packed_kernel_cache is None
    fn_after_learn = eng.compile_callable()
    assert fn_after_learn("novel token xyzzy").values["route"] == "access"

    assert eng._packed_kernel_cache is not None
    eng.calibrate(
        [
            ("invoice refund charge", {"route": "billing", "urgent": False, "severity": 0.2}),
            ("server outage 503", {"route": "incident", "urgent": True, "severity": 0.9}),
            ("password sso 403", {"route": "access", "urgent": False, "severity": 0.1}),
        ]
        * 3
    )
    assert eng._packed_kernel_cache is None

    _ = eng.compile_packed_kernel()
    assert eng._packed_kernel_cache is not None
    eng.adapt_zero_shot(["server outage 503", "invoice refund charge", "password sso 403"])
    assert eng._packed_kernel_cache is None

    # TypeSafeClient integration
    client = TypeSafeClient()
    questions = {
        "triage": Choice("Route ticket", criteria={"billing": "Invoices", "incident": "Outages"}),
        "page": Noul("Should page on-call"),
        "priority": Score("Urgency score", min_value=0.0, max_value=1.0),
    }
    ts_fn = client.compile_callable(questions)
    assert isinstance(ts_fn, FastDecisionCallable)
    ts_fn2 = compile_typesafe_callable(client, questions)
    res_ts = ts_fn2("Invoices duplicate payment charge")
    assert isinstance(res_ts, FastDecisionOutput)
    assert res_ts.values["triage"] in ("billing", "incident")


def test_s1m_v3_format_defaults_q128_page_alignment_and_hybrid_embed_vocab(tmp_path: Path) -> None:
    """v2 defaults stay untouched while opt-in S1M\\x03 supports Q=128, 16 KiB alignment, and embed_vocab."""
    small_schema = _make_mixed_schema(2, 1, 1)
    small_compiler = SystemOneCompiler(small_schema, dimension=64)
    small_model = small_compiler.compile(_make_mixed_exemplars(small_schema, samples_per_choice=4), augment=False)
    v2_bytes = small_model.to_bytes()
    assert v2_bytes[:4] == MAGIC_HEADER == b"S1M\x02"

    # Q=128 > 64: default v2 export hits MAX_SKILL_FIELDS limit on load, while format_version=3 succeeds
    assert MAX_SKILL_FIELDS_V3 == 512
    assert S1M_V3_PAGE_SIZE == 16384
    schema128 = _make_mixed_schema(48, 48, 32)  # 128 fields total > 64
    assert len(schema128.fields) == 128
    ex128 = _make_mixed_exemplars(schema128, samples_per_choice=3)
    comp128 = SystemOneCompiler(schema128, dimension=64)
    model128 = comp128.compile(ex128, augment=False)
    v2_over_limit = model128.to_bytes()
    with pytest.raises(ValueError, match="field count"):
        CompiledSystemOneModel.from_bytes(v2_over_limit)

    v3_path = tmp_path / "q128.s1m"
    comp128.compile_and_save(
        v3_path,
        ex128,
        augment=False,
        format_version=3,
    )
    raw_v3 = v3_path.read_bytes()
    assert raw_v3[:8] == MAGIC_HEADER_V3
    assert raw_v3[:4] == b"S1M\x03"
    assert len(raw_v3) % S1M_V3_PAGE_SIZE == 0

    magic, hdr_len = struct.unpack_from("<8sQ", raw_v3, 0)
    assert magic == MAGIC_HEADER_V3
    meta_v3 = json.loads(raw_v3[16 : 16 + hdr_len].decode("utf-8"))
    assert meta_v3["version"] == 3
    tensors = meta_v3["tensors"]
    first_offset = min(spec["offset"] for spec in tensors.values() if spec["nbytes"] > 0)
    assert first_offset % S1M_V3_PAGE_SIZE == 0
    for tname, spec in tensors.items():
        if spec["nbytes"] > 0:
            assert spec["offset"] % 64 == 0
            if spec["nbytes"] >= 65536 or tname == "_projector_dense_table":
                assert spec["offset"] % S1M_V3_PAGE_SIZE == 0

    # Roundtrip Q=128 via mmap_mode=True and mmap_mode=False
    with SystemOneCompiler.load(v3_path, mmap_mode=True) as loaded_mmap:
        loaded_mem = CompiledSystemOneModel.load(v3_path, mmap_mode=False)
        assert loaded_mmap._mmap_obj is not None
        assert loaded_mmap.heads["c_0"].weights.flags.writeable is False
        e1 = SystemOneEngine(schema128, compiled_model=loaded_mmap, strict_mode=True, use_cache=False)
        e2 = SystemOneEngine(schema128, compiled_model=loaded_mem, strict_mode=True, use_cache=False)
        p = "invoice payment refund alpha case"
        r1 = e1.decide(p, record_receipt=False)
        r2 = e2.decide(p, record_receipt=False)
        assert r1.values == r2.values
        assert r1.is_ambiguous == r2.is_ambiguous
    assert loaded_mmap._mmap_obj is None

    # HybridProjector with embed_vocab=False vs embed_vocab=True
    corpus = [
        ("customer charged twice invoice refund", "billing"),
        ("card declined monthly renewal billing", "billing"),
        ("gateway returning http 503 timeout", "incident"),
        ("production database latency spike", "incident"),
        ("okta sso 403 permission denied", "access"),
        ("mfa authenticator admin lock", "access"),
    ]
    h_schema = DecisionSchema(
        fields={"route": ChoiceField(["billing", "incident", "access"])}
    )
    h_proj = HybridProjector.fit(
        [t for t, _ in corpus],
        labels=[y for _, y in corpus],
        sparse_dim=128,
        dense_dim=64,
        seed=42,
    )
    h_comp = SystemOneCompiler(h_schema, projector=h_proj, dimension=h_proj.dimension)
    h_model = h_comp.compile({"route": corpus * 2}, augment=False)

    p_no_vocab = tmp_path / "hybrid_no_vocab.s1m"
    p_vocab = tmp_path / "hybrid_with_vocab.s1m"
    h_model.save(p_no_vocab, format_version=3, embed_vocab=False)
    h_model.save(p_vocab, format_version=3, embed_vocab=True)

    raw_vocab = p_vocab.read_bytes()
    _, v_hdr_len = struct.unpack_from("<8sQ", raw_vocab, 0)
    v_meta = json.loads(raw_vocab[16 : 16 + v_hdr_len].decode("utf-8"))
    assert v_meta["embed_vocab"] is True
    assert "_projector_dense_table" in v_meta["tensors"]
    assert v_meta["tensors"]["_projector_dense_table"]["offset"] % S1M_V3_PAGE_SIZE == 0

    with SystemOneEngine.load(p_vocab, schema=h_schema, mmap_mode=True) as e_vocab:
        with CompiledSystemOneModel.load(p_no_vocab, mmap_mode=True) as m_no_vocab:
            e_no_vocab = SystemOneEngine(h_schema, compiled_model=m_no_vocab, use_cache=False)
            for probe in ("duplicate card charge refund", "http 503 gateway timeout", "locked out okta 403"):
                assert e_vocab.decide(probe, record_receipt=False).values == e_no_vocab.decide(
                    probe, record_receipt=False
                ).values


def test_s1m_v3_security_validation_and_cow_online_learn(tmp_path: Path) -> None:
    """v3 rejects corrupted/misaligned/overlapping/non-finite blobs cleanly and keeps mmap files read-only on learn()."""
    schema = DecisionSchema(fields={"route": ChoiceField(["billing", "incident", "access"])})
    compiled = SystemOneCompiler(schema, dimension=64).compile(
        {
            "route": [
                ("invoice refund charge", "billing"),
                ("gateway timeout 503", "incident"),
                ("password sso 403", "access"),
            ]
            * 5
        },
        augment=False,
    )
    path = tmp_path / "cow.s1m"
    compiled.save(path, include_covariance=True, compact=False, format_version=3)
    raw = bytearray(path.read_bytes())
    orig_sha = hashlib.sha256(raw).hexdigest()

    # CoW test: load via mmap_mode=True, verify P/B and W/b are read-only mmap views, run learn()
    with CompiledSystemOneModel.load(path, mmap_mode=True) as mm_model:
        assert mm_model.heads["route"].weights.flags.writeable is False
        eng = SystemOneEngine(schema, compiled_model=mm_model, strict_mode=True, use_cache=False)
        for _ in range(3):
            eng.learn("special custom phrase xyzzy", {"route": "access"})
        assert eng.decide("special custom phrase xyzzy", record_receipt=False).values["route"] == "access"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == orig_sha

    # Helper to rewrite v3 header cleanly for targeted corruption tests
    def _rebuild_with_meta(mutator) -> bytes:
        _, hlen = struct.unpack_from("<8sQ", raw, 0)
        hdr = json.loads(bytes(raw[16 : 16 + hlen]).decode("utf-8"))
        mutator(hdr)
        new_h = json.dumps(hdr, sort_keys=True).encode("utf-8")
        # Header stays within first 16 KiB page before first tensor offset.
        buf = bytearray(raw)
        struct.pack_into("<8sQ", buf, 0, MAGIC_HEADER_V3, len(new_h))
        buf[16 : 16 + len(new_h)] = new_h
        return bytes(buf)

    # 1. Bad magic & truncated header
    bad_magic = bytearray(raw)
    bad_magic[:4] = b"BAD!"
    with pytest.raises(ValueError, match="magic"):
        CompiledSystemOneModel.from_bytes(bytes(bad_magic))
    with pytest.raises(ValueError, match="truncated"):
        CompiledSystemOneModel.from_bytes(bytes(raw[:20]))

    # 2. Tampered schema_digest
    tampered_digest = _rebuild_with_meta(lambda h: h.__setitem__("schema_digest", "0" * 64))
    bad_path = tmp_path / "tampered.s1m"
    bad_path.write_bytes(tampered_digest)
    with pytest.raises(ValueError, match="digest"):
        CompiledSystemOneModel.load(bad_path, mmap_mode=True)

    # 3. Unknown tensor name
    def _add_unknown(h: dict) -> None:
        h["tensors"]["evil_tensor"] = {
            "dtype": "float32",
            "shape": [4],
            "offset": h["tensors"]["route_b"]["offset"],
            "nbytes": 16,
        }

    with pytest.raises(ValueError, match="unexpected tensor"):
        CompiledSystemOneModel.from_bytes(_rebuild_with_meta(_add_unknown))

    # 4. Misaligned offset (not multiple of 64)
    def _misalign(h: dict) -> None:
        h["tensors"]["route_b"]["offset"] += 8

    with pytest.raises(ValueError, match="misaligned"):
        CompiledSystemOneModel.from_bytes(_rebuild_with_meta(_misalign))

    # 5. Overlapping tensor ranges
    def _overlap(h: dict) -> None:
        h["tensors"]["route_b"]["offset"] = h["tensors"]["route_w"]["offset"]

    with pytest.raises(ValueError, match="overlaps"):
        CompiledSystemOneModel.from_bytes(_rebuild_with_meta(_overlap))

    # 6. Non-finite JSON constant
    _, hlen = struct.unpack_from("<8sQ", raw, 0)
    hdr_nan = bytes(raw[16 : 16 + hlen]).replace(b'"created_at": 0.0', b'"created_at": NaN')
    assert b"NaN" in hdr_nan
    buf_nan_hdr = bytearray(raw)
    struct.pack_into("<8sQ", buf_nan_hdr, 0, MAGIC_HEADER_V3, len(hdr_nan))
    buf_nan_hdr[16 : 16 + len(hdr_nan)] = hdr_nan
    with pytest.raises(ValueError, match="non-finite"):
        CompiledSystemOneModel.from_bytes(bytes(buf_nan_hdr))

    # 7. Non-finite float inside weight tensor (tested via mmap_mode=True too to ensure no BufferError leak)
    _, hlen = struct.unpack_from("<8sQ", raw, 0)
    hdr = json.loads(bytes(raw[16 : 16 + hlen]).decode("utf-8"))
    w_off = hdr["tensors"]["route_w"]["offset"]
    buf_nan_w = bytearray(raw)
    struct.pack_into("<f", buf_nan_w, w_off, float("nan"))
    nan_w_path = tmp_path / "nan_weights.s1m"
    nan_w_path.write_bytes(bytes(buf_nan_w))
    with pytest.raises(ValueError, match="non-finite"):
        CompiledSystemOneModel.load(nan_w_path, mmap_mode=True)
