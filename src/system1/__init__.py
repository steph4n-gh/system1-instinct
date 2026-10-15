"""System 1: teach bounded decision skills, validate them and reuse them locally.

Fixed local features and small decision heads support structured outputs,
uncertainty, optional teacher observation, explicit permissions and audit evidence."""

from __future__ import annotations

import sys
from typing import Any, Mapping, Optional, Sequence, Type, Union

# Core mathematical kernel and schema primitives (Zero crypto, Zero SQLite, pure NumPy)
from system1.core import (
    BooleanField,
    ChoiceField,
    DecisionField,
    DecisionFieldHead,
    DecisionSchema,
    FieldDefinition,
    DeterministicSemanticProjector,
    TfidfProjector,
    HybridProjector,
    HybridSemanticProjector,
    LocalNeuralProjector,
    ModelInferenceResult,
    MultiChoiceField,
    RawFieldEvaluation,
    ScoreField,
    StaticEmbeddingTable,
    SubwordSemanticEmbeddings,
    SystemOneModel,
    SchemaMeta,
)

# Calibration & Conformal Prediction primitives (pure NumPy, zero crypto, zero SQLite)
from system1.calibration import (
    BrierDecomposition,
    CalibrationMetrics,
    ConformalPredictionSet,
    ConformalPredictor,
    DecisionCalibrator,
    RecalibrationMetrics,
    RecalibrationReport,
    RegressionConformalInterval,
    RegressionConformalPredictor,
)

# Eager lightweight submodules
from system1 import (
    calibration,
    core,
    embeddings,
    model,
    neural,
    schema,
)

__version__ = "1.0.0"


def decide(
    prompt: str,
    schema: Union[DecisionSchema, Type[DecisionSchema]],
    *,
    telemetry: Optional[Any] = None,
    alpha: float = 0.05,
    record_receipt: bool = True,
    dimension: int = 384,
    backend: str = "auto",
    projector: Optional[Any] = None,
    margin_threshold: Optional[float] = None,
) -> Any:
    """One-liner functional API for evaluating a System 1 decision against a typed schema.

    Example:
        import system1

        class RoutingSchema(system1.DecisionSchema):
            route = system1.ChoiceField(options=["sales", "support", "billing"])
            is_escalation = system1.BooleanField()

        result = system1.decide("Customer invoice payment dispute", schema=RoutingSchema)
        print(result.route)
        print(result.confidences["route"])
        print(result.conformal_sets["route"])
    """
    if not isinstance(prompt, str):
        raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}")
    from system1.engine import SystemOneEngine
    engine_instance = SystemOneEngine(
        schema,
        dimension=dimension,
        backend=backend,
        projector=projector,
        margin_threshold=margin_threshold,
    )
    return engine_instance.decide(
        prompt,
        telemetry=telemetry,
        alpha=alpha,
        record_receipt=record_receipt,
        margin_threshold=margin_threshold,
    )


def evaluate(
    target: Any,
    rows: Any = None,
    *,
    schema: Optional[Union[DecisionSchema, Type[DecisionSchema]]] = None,
    dataset: Any = None,
    examples: Any = None,
    telemetry: Optional[Any] = None,
    alpha: Optional[float] = None,
    n_bins: int = 10,
    strict_mode: Optional[bool] = None,
    strict: Optional[bool] = None,
    use_cache: bool = False,
    record_receipt: bool = True,
    dimension: int = 384,
    backend: str = "auto",
    projector: Optional[Any] = None,
    margin_threshold: Optional[float] = None,
) -> Any:
    """Evaluate a System 1 engine, compiled model, or .s1m file on a labeled dataset.

    Supports ``system1.evaluate(engine_or_model_or_path, rows, alpha=0.05)`` returning
    an ``EvaluationReport``, while remaining backward-compatible with single-prompt
    ``system1.evaluate(prompt, schema)`` calls.
    """
    is_schema_arg = (rows is not None) and (
        isinstance(rows, DecisionSchema)
        or (isinstance(rows, type) and issubclass(rows, DecisionSchema))
    )
    if isinstance(target, str) and (
        is_schema_arg
        or (rows is None and dataset is None and examples is None and schema is not None)
    ):
        resolved_schema = rows if is_schema_arg else schema
        if resolved_schema is None:
            raise ValueError("Schema is required for single-prompt evaluation")
        return decide(
            target,
            resolved_schema,
            telemetry=telemetry,
            alpha=0.05 if alpha is None else alpha,
            record_receipt=record_receipt,
            dimension=dimension,
            backend=backend,
            projector=projector,
            margin_threshold=margin_threshold,
        )

    from pathlib import Path
    from system1.compiler import CompiledSystemOneModel
    from system1.engine import SystemOneEngine

    eval_rows = rows if rows is not None else (dataset if dataset is not None else examples)
    if eval_rows is None and hasattr(target, "examples_path"):
        ex_path = Path(getattr(target, "examples_path"))
        if ex_path.is_file():
            eval_rows = ex_path
    if eval_rows is None:
        raise ValueError(
            "evaluate() requires labeled rows (a sequence of examples or a path to a JSONL/JSON dataset)"
        )

    eff_strict = strict if strict is not None else strict_mode
    if isinstance(target, SystemOneEngine):
        engine_instance = target
    elif isinstance(target, CompiledSystemOneModel):
        engine_instance = SystemOneEngine(
            schema=schema if schema is not None else target.schema,
            model=target,
            dimension=target.dimension,
            backend=backend if backend != "auto" else target.backend,
            projector=projector if projector is not None else target.projector,
            strict_mode=bool(target.metadata.get("strict_mode", True)) if eff_strict is None else bool(eff_strict),
            use_cache=use_cache,
            margin_threshold=margin_threshold,
        )
    elif isinstance(target, SystemOneModel):
        engine_instance = SystemOneEngine(
            schema=schema if schema is not None else target.schema,
            model=target,
            dimension=target.dimension,
            backend=backend if backend != "auto" else target.backend,
            projector=projector if projector is not None else target.projector,
            strict_mode=False if eff_strict is None else bool(eff_strict),
            use_cache=use_cache,
            margin_threshold=margin_threshold,
        )
    elif hasattr(target, "current_path") or hasattr(target, "skill_path"):
        s_path: Optional[Path] = None
        for attr in ("current_path", "candidate_path", "skill_path"):
            val = getattr(target, attr, None)
            if val is not None and Path(val).is_file():
                s_path = Path(val)
                break
        if s_path is None and hasattr(target, "skill_path"):
            s_path = Path(getattr(target, "skill_path"))
            if not s_path.is_file() and hasattr(target, "compile"):
                target.compile()
        if s_path is None or not s_path.is_file():
            raise FileNotFoundError("TeachingSession or Skill does not have a compiled .s1m artifact yet")
        load_kwargs: dict[str, Any] = {"use_cache": use_cache}
        if eff_strict is not None:
            load_kwargs["strict_mode"] = bool(eff_strict)
        if margin_threshold is not None:
            load_kwargs["margin_threshold"] = margin_threshold
        engine_instance = SystemOneEngine.load(
            s_path,
            schema=schema if schema is not None else getattr(target, "schema", None),
            projector=projector,
            backend=backend,
            **load_kwargs,
        )
    elif isinstance(target, (str, Path)):
        t_path = Path(target)
        if t_path.is_dir():
            for cand in ("current.s1m", "skill.s1m", "candidate.s1m"):
                if (t_path / cand).is_file():
                    t_path = t_path / cand
                    break
        if t_path.is_file():
            if t_path.suffix.lower() in {".json", ".py"}:
                from system1.cli import _load_schema

                resolved_schema_file = _load_schema(str(t_path))
                engine_instance = SystemOneEngine(
                    schema=resolved_schema_file,
                    dimension=dimension,
                    backend=backend,
                    projector=projector,
                    strict_mode=False if eff_strict is None else bool(eff_strict),
                    use_cache=use_cache,
                    margin_threshold=margin_threshold,
                )
            else:
                load_kwargs = {"use_cache": use_cache}
                if eff_strict is not None:
                    load_kwargs["strict_mode"] = bool(eff_strict)
                if margin_threshold is not None:
                    load_kwargs["margin_threshold"] = margin_threshold
                engine_instance = SystemOneEngine.load(
                    t_path,
                    schema=schema,
                    projector=projector,
                    backend=backend,
                    **load_kwargs,
                )
        elif isinstance(target, str) and not t_path.exists():
            if (
                target.endswith(".s1m")
                or target.endswith(".json")
                or target.endswith(".py")
                or "/" in target
                or "\\" in target
            ):
                raise FileNotFoundError(f"Evaluation target file not found: {target}")
            from system1.cli import _load_schema

            resolved_schema = _load_schema(target)
            engine_instance = SystemOneEngine(
                schema=resolved_schema,
                dimension=dimension,
                backend=backend,
                projector=projector,
                strict_mode=False if eff_strict is None else bool(eff_strict),
                use_cache=use_cache,
                margin_threshold=margin_threshold,
            )
        else:
            raise FileNotFoundError(f"Evaluation target file not found: {target}")
    elif isinstance(target, DecisionSchema) or (
        isinstance(target, type) and issubclass(target, DecisionSchema)
    ):
        engine_instance = SystemOneEngine(
            schema=target,
            dimension=dimension,
            backend=backend,
            projector=projector,
            strict_mode=False if eff_strict is None else bool(eff_strict),
            use_cache=use_cache,
            margin_threshold=margin_threshold,
        )
    else:
        raise TypeError(f"Unsupported evaluation target type: {type(target).__name__}")

    return engine_instance.evaluate_dataset(
        eval_rows,
        alpha=alpha,
        n_bins=n_bins,
        strict_mode=eff_strict,
        use_cache=use_cache,
    )


_SUBMODULE_NAMES = {
    "cache",
    "calibration",
    "cli",
    "compat",
    "compiler",
    "core",
    "embeddings",
    "engine",
    "fallback",
    "grpc_server",
    "guard",
    "integrations",
    "ledger",
    "model",
    "neural",
    "proto",
    "receipt",
    "schema",
    "telemetry",
    "teaching",
}

# Lazy-loaded governance modules and symbols (cryptography, SQLite, reference monitor, compat)
_MODULE_MAP = {
    # Public submodules
    "cache": "system1.cache",
    "calibration": "system1.calibration",
    "cli": "system1.cli",
    "compat": "system1.compat",
    "compiler": "system1.compiler",
    "core": "system1.core",
    "embeddings": "system1.embeddings",
    "engine": "system1.engine",
    "fallback": "system1.fallback",
    "grpc_server": "system1.grpc_server",
    "guard": "system1.guard",
    "integrations": "system1.integrations",
    "ledger": "system1.ledger",
    "model": "system1.model",
    "neural": "system1.neural",
    "proto": "system1.proto",
    "receipt": "system1.receipt",
    "schema": "system1.schema",
    "telemetry": "system1.telemetry",
    "teaching": "system1.teaching",
    "TeachingSession": "system1.teaching",
    "BoundedReviewFallback": "system1.fallback",
    "ReviewFallbackDecision": "system1.fallback",
    # cache (Lever 1)
    "SemanticSystemOneCache": "system1.cache",
    "CacheEntry": "system1.cache",
    # telemetry (Lever 4)
    "TelemetryProjector": "system1.telemetry",
    # engine
    "SystemOneEngine": "system1.engine",
    "System1Engine": "system1.engine",
    "ReflexEngine": "system1.engine",
    "DecisionResult": "system1.engine",
    "BenchmarkReport": "system1.engine",
    "FieldEvaluationMetrics": "system1.engine",
    "EvaluationReport": "system1.engine",
    # receipt
    "DecisionWitnessReceipt": "system1.receipt",
    "create_decision_receipt": "system1.receipt",
    "verify_decision_witness_receipt": "system1.receipt",
    "check_decision_receipt_integrity": "system1.receipt",
    "compute_receipt_digest": "system1.receipt",
    "RunWitnessEnvelope": "system1.receipt",
    "create_run_witness_envelope": "system1.receipt",
    "sign_run_witness_envelope": "system1.receipt",
    "verify_run_witness_envelope": "system1.receipt",
    "canonical_json": "system1.receipt",
    "canonical_bytes": "system1.receipt",
    "fingerprint": "system1.receipt",
    "public_key_bytes": "system1.receipt",
    "public_key_fingerprint": "system1.receipt",
    "load_private_key": "system1.receipt",
    "load_public_key": "system1.receipt",
    "save_keypair": "system1.receipt",
    "save_private_key": "system1.receipt",
    "save_public_key": "system1.receipt",
    "sign_payload": "system1.receipt",
    "verify_payload": "system1.receipt",
    "DECISION_WITNESS_PROFILE": "system1.receipt",
    # ledger
    "ActionLedger": "system1.ledger",
    "LedgerError": "system1.ledger",
    "LedgerWriteError": "system1.ledger",
    "IntegrityError": "system1.ledger",
    # guard
    "SystemOneGuard": "system1.guard",
    "SystemOneGuardHook": "system1.guard",
    "System1Guard": "system1.guard",
    "ReflexGuard": "system1.guard",
    "PolicyRule": "system1.guard",
    "PolicyEngine": "system1.guard",
    "DeterministicPolicyEngine": "system1.guard",
    "DefaultGuardDecisionSchema": "system1.guard",
    "GuardInterceptionResult": "system1.guard",
    "ActionProposal": "system1.guard",
    "PolicyDecision": "system1.guard",
    "EvidenceRef": "system1.guard",
    "DecisionOutcome": "system1.guard",
    "RiskLevel": "system1.guard",
    "ActionState": "system1.guard",
    "ResultStatus": "system1.guard",
    # compat
    "TypeSafeClient": "system1.compat.typesafe",
    "Client": "system1.compat.typesafe",
    "AsyncTypeSafeClient": "system1.compat.typesafe",
    "AsyncClient": "system1.compat.typesafe",
    "Choice": "system1.compat.typesafe",
    "MultiChoice": "system1.compat.typesafe",
    "Noul": "system1.compat.typesafe",
    "Score": "system1.compat.typesafe",
    "TypeSafeResponse": "system1.compat.typesafe",
    "SystemOneResponse": "system1.compat.typesafe",
    "ChoiceAnswer": "system1.compat.typesafe",
    "NoulAnswer": "system1.compat.typesafe",
    "ScoreAnswer": "system1.compat.typesafe",
    "MultiChoiceAnswer": "system1.compat.typesafe",
    "Usage": "system1.compat.typesafe",
    "system_one": "system1.compat.typesafe",
    "systemone": "system1.compat.typesafe",
    "batch_system_one": "system1.compat.typesafe",
    "batch_systemone": "system1.compat.typesafe",
    "patch_typesafe": "system1.compat.typesafe",
    "DotDict": "system1.compat.typesafe",
    "ZeroEgressViolationError": "system1.compat.typesafe",
    # compiler
    "SystemOneCompiler": "system1.compiler",
    "CompiledSystemOneModel": "system1.compiler",
    # integrations
    "SystemOneMCPProxy": "system1.integrations",
    "SystemOneMCPBlockedError": "system1.integrations",
    "wrap_mcp_tool": "system1.integrations",
    "SystemOneGatewayMiddleware": "system1.integrations",
    "add_system1_gateway": "system1.integrations",
    "SystemOneGuardCallbackHandler": "system1.integrations",
    "SystemOneToolInterceptor": "system1.integrations",
    "SystemOneGuardBlockedException": "system1.integrations",
    "wrap_langchain_tool": "system1.integrations",
    "SemanticRoute": "system1.integrations",
    "RouteMatchResult": "system1.integrations",
    "SystemOneRouteLayer": "system1.integrations",
    "InstinctRouteLayer": "system1.integrations",
    "build_default_router": "system1.integrations",
    "RouteLLMCascadeSchema": "system1.integrations",
    "RouteDecision": "system1.integrations",
    "SystemOneRouteController": "system1.integrations",
    "IncidentTriageModel": "system1.integrations",
    "StructuredExtractionResult": "system1.integrations",
    "SystemOneInstructorClient": "system1.integrations",
    "PromptGuardSchema": "system1.integrations",
    "PromptGuardResult": "system1.integrations",
    "SystemOnePromptGuard": "system1.integrations",
}


def __getattr__(name: str) -> Any:
    mod_name = _MODULE_MAP.get(name)
    if mod_name is not None:
        import importlib
        mod = importlib.import_module(mod_name)
        if name in _SUBMODULE_NAMES:
            val = mod
        else:
            val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__) | set(_MODULE_MAP.keys()) | _SUBMODULE_NAMES)


__all__ = [
    "TeachingSession",
    "BoundedReviewFallback",
    "ReviewFallbackDecision",
    # Version
    "__version__",
    # Engine & Core Results
    "SystemOneEngine",
    "System1Engine",
    "ReflexEngine",
    "DecisionResult",
    "BenchmarkReport",
    "FieldEvaluationMetrics",
    "EvaluationReport",
    "decide",
    # Schemas & Fields
    "DecisionSchema",
    "DecisionField",
    "FieldDefinition",
    "ChoiceField",
    "BooleanField",
    "MultiChoiceField",
    "ScoreField",
    "SchemaMeta",
    # Calibration & Conformal Prediction
    "DecisionCalibrator",
    "CalibrationMetrics",
    "RecalibrationMetrics",
    "RecalibrationReport",
    "BrierDecomposition",
    "ConformalPredictor",
    "ConformalPredictionSet",
    "RegressionConformalPredictor",
    "RegressionConformalInterval",
    # Proof-Carrying Evidence Receipts & Crypto
    "DecisionWitnessReceipt",
    "create_decision_receipt",
    "verify_decision_witness_receipt",
    "check_decision_receipt_integrity",
    "compute_receipt_digest",
    "RunWitnessEnvelope",
    "create_run_witness_envelope",
    "sign_run_witness_envelope",
    "verify_run_witness_envelope",
    "canonical_json",
    "canonical_bytes",
    "fingerprint",
    "public_key_bytes",
    "public_key_fingerprint",
    "load_private_key",
    "load_public_key",
    "save_keypair",
    "save_private_key",
    "save_public_key",
    "sign_payload",
    "verify_payload",
    "DECISION_WITNESS_PROFILE",
    # Action Ledger
    "ActionLedger",
    "LedgerError",
    "LedgerWriteError",
    "IntegrityError",
    # Reference Monitor & Guard
    "SystemOneGuard",
    "SystemOneGuardHook",
    "System1Guard",
    "ReflexGuard",
    "PolicyRule",
    "PolicyEngine",
    "DeterministicPolicyEngine",
    "DefaultGuardDecisionSchema",
    "GuardInterceptionResult",
    "ActionProposal",
    "PolicyDecision",
    "EvidenceRef",
    "DecisionOutcome",
    "RiskLevel",
    "ActionState",
    "ResultStatus",
    # Model & Neural Projection
    "SystemOneModel",
    "DecisionFieldHead",
    "DeterministicSemanticProjector",
    "TfidfProjector",
    "LocalNeuralProjector",
    "HybridProjector",
    "HybridSemanticProjector",
    "StaticEmbeddingTable",
    "SubwordSemanticEmbeddings",
    "ModelInferenceResult",
    "RawFieldEvaluation",
    # TypeSafe Compatibility
    "TypeSafeClient",
    "Client",
    "AsyncTypeSafeClient",
    "AsyncClient",
    "Choice",
    "MultiChoice",
    "Noul",
    "Score",
    "TypeSafeResponse",
    "SystemOneResponse",
    "ChoiceAnswer",
    "NoulAnswer",
    "ScoreAnswer",
    "MultiChoiceAnswer",
    "Usage",
    "system_one",
    "systemone",
    "batch_system_one",
    "batch_systemone",
    "patch_typesafe",
    "DotDict",
    "ZeroEgressViolationError",
    # Compiler
    "SystemOneCompiler",
    "CompiledSystemOneModel",
    # 4 Levers (Cache, Telemetry, Online Update, Margin Gating)
    "SemanticSystemOneCache",
    "CacheEntry",
    "TelemetryProjector",
    "evaluate",
    # Framework Integrations (MCP, FastAPI, LangChain)
    "SystemOneMCPProxy",
    "SystemOneMCPBlockedError",
    "wrap_mcp_tool",
    "SystemOneGatewayMiddleware",
    "add_system1_gateway",
    "SystemOneGuardCallbackHandler",
    "SystemOneToolInterceptor",
    "SystemOneGuardBlockedException",
    "wrap_langchain_tool",
    # Drop-In Ecosystem Replacements
    "SemanticRoute",
    "RouteMatchResult",
    "SystemOneRouteLayer",
    "InstinctRouteLayer",
    "build_default_router",
    "RouteLLMCascadeSchema",
    "RouteDecision",
    "SystemOneRouteController",
    "IncidentTriageModel",
    "StructuredExtractionResult",
    "SystemOneInstructorClient",
    "PromptGuardSchema",
    "PromptGuardResult",
    "SystemOnePromptGuard",
]
