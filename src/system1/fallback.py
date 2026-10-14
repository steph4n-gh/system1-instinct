"""Bounded strict-review fallback coordinator for System 1 local skills.

When a strict local skill abstains (requests review), BoundedReviewFallback can
optionally consult an asynchronous read-only expert under explicit permission,
state-freshness checks, synchronous schema/domain validation, and a hard attempt
budget. Any invariant violation fails closed and halts the runner.
"""

from __future__ import annotations

import asyncio
from hashlib import sha256
import inspect
import json
import math
import time
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Type, Union

from system1.core.schema import DecisionSchema


def _checked(callback: Callable[..., Any], *args: Any) -> bool:
    result = callback(*args)
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        raise TypeError("Permission and validation callbacks must be synchronous")
    return result is True


def _checked_value(callback: Callable[..., Any], *args: Any) -> Any:
    result = callback(*args)
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        raise TypeError("State observation callbacks must be synchronous")
    return result


def default_observed_state(context: Any) -> bytes:
    """Compute a deterministic state digest for common context types."""
    if isinstance(context, bytes):
        return sha256(context).digest()
    if isinstance(context, str):
        return sha256(context.encode("utf-8")).digest()
    if hasattr(context, "pos") and hasattr(context, "ticks") and hasattr(context, "visits"):
        prompt_Part = getattr(context, "prompt", None)
        data = (
            prompt_Part() if callable(prompt_Part) else prompt_Part,
            getattr(context, "pos", None),
            getattr(context, "ticks", None),
            sorted(context.visits.items()),
            getattr(context, "alive", True),
            getattr(context, "done", False),
        )
        return sha256(json.dumps(data, sort_keys=True, default=repr).encode("utf-8")).digest()
    if isinstance(context, Mapping):
        return sha256(json.dumps(dict(context), sort_keys=True, default=repr).encode("utf-8")).digest()
    return sha256(repr(context).encode("utf-8")).digest()


_UNSET = object()


class ReviewFallbackDecision(dict):
    """Structured result from BoundedReviewFallback.decide().

    Subclasses ``dict`` for backward compatibility with mapping access
    (``res["recovered"]``, ``res["values"]``) while also supporting attribute
    access (``res.used_fallback``, ``res.recovered``, ``res.values``,
    ``res.decision``, ``res.lessons_recorded``, ``res.online_updates_applied``)
    and ``await fb.decide(...)`` in asynchronous callers.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            values = self.get("values")
            if isinstance(values, Mapping) and name in values:
                return values[name]
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            ) from None

    def __await__(self):
        async def _coro() -> ReviewFallbackDecision:
            return self

        return _coro().__await__()


class BoundedReviewFallback:
    """One runner per episode/session/event loop, with at most one pending expert.

    Recovery requires a read-only review callable (sync or async), explicit
    synchronous permission callback, independent synchronous validator (plus
    optional DecisionSchema and SystemOneGuard), and an attempted-call budget.
    Optionally closes the boundary-learning loop by recording validated fallback
    decisions into an attached ``TeachingSession`` (``auto_record_lessons=True``)
    and applying sub-millisecond Rank-1 / NLMS online updates
    (``auto_online_update=True``).
    """

    def __init__(
        self,
        network: Optional[Any] = None,
        review_fn: Optional[Callable[..., Any]] = None,
        *,
        engine: Optional[Any] = None,
        expert: Any = _UNSET,
        allowed: Any = _UNSET,
        validate: Any = _UNSET,
        observe_state: Optional[Callable[[Any], Any]] = None,
        is_done: Optional[Callable[[Any], bool]] = None,
        prompt_extractor: Optional[Callable[[Any], str]] = None,
        schema: Optional[Union[DecisionSchema, Type[DecisionSchema]]] = None,
        guard: Optional[Any] = None,
        action_key: str = "action",
        expert_budget: Any = _UNSET,
        max_reviews: Optional[int] = None,
        expert_timeout: float = 1.0,
        teaching_session: Optional[Any] = None,
        auto_record_lessons: bool = False,
        auto_online_update: bool = False,
        lesson_split: str = "train",
        lesson_source: str = "fallback_review",
    ) -> None:
        if network is None:
            network = engine if engine is not None else teaching_session
        if network is None:
            raise ValueError("BoundedReviewFallback requires a network, engine, or teaching_session")

        has_flywheel_or_review = (
            review_fn is not None
            or teaching_session is not None
            or bool(auto_record_lessons)
            or bool(auto_online_update)
        )

        if expert is _UNSET:
            resolved_expert = review_fn
        else:
            resolved_expert = expert

        if allowed is _UNSET:
            resolved_allowed = (lambda _ctx: True) if (has_flywheel_or_review or resolved_expert is not None) else None
        else:
            resolved_allowed = allowed

        if validate is _UNSET:
            resolved_validate = (lambda _ctx, _ans: True) if (has_flywheel_or_review or resolved_expert is not None) else None
        else:
            resolved_validate = validate

        if expert_budget is _UNSET:
            if max_reviews is not None:
                resolved_budget = max_reviews
            elif has_flywheel_or_review or resolved_expert is not None:
                resolved_budget = 1_000_000
            else:
                resolved_budget = 0
        else:
            resolved_budget = expert_budget

        if type(resolved_budget) is not int or resolved_budget < 0:
            raise ValueError("expert_budget must be a nonnegative integer")
        if (
            isinstance(expert_timeout, bool)
            or not isinstance(expert_timeout, (int, float))
            or not math.isfinite(expert_timeout)
            or expert_timeout <= 0
        ):
            raise ValueError("expert_timeout must be finite and positive")
        if not isinstance(action_key, str) or not action_key:
            raise ValueError("action_key must be a nonempty string")
        if lesson_split not in ("train", "teach", "auto", "calibrate", "evaluate"):
            raise ValueError("lesson_split must be 'train', 'teach', 'auto', 'calibrate', or 'evaluate'")
        if not isinstance(lesson_source, str) or not lesson_source.strip():
            raise ValueError("lesson_source must be a nonempty string")

        self.network = network
        self.expert = resolved_expert
        self.review_fn = review_fn if review_fn is not None else resolved_expert
        self.allowed = resolved_allowed
        self.validate = resolved_validate
        self.observe_state = observe_state or default_observed_state
        self.is_done = is_done
        self.prompt_extractor = prompt_extractor
        if teaching_session is not None:
            self.teaching_session = teaching_session
        elif hasattr(network, "record") and hasattr(network, "assess"):
            self.teaching_session = network
        else:
            self.teaching_session = None

        if isinstance(schema, type) and issubclass(schema, DecisionSchema):
            self.schema: Optional[DecisionSchema] = schema()
        elif isinstance(schema, DecisionSchema):
            self.schema = schema
        elif isinstance(getattr(network, "schema", None), DecisionSchema):
            self.schema = network.schema
        elif self.teaching_session is not None and isinstance(getattr(self.teaching_session, "schema", None), DecisionSchema):
            self.schema = self.teaching_session.schema
        else:
            self.schema = None
        self.guard = guard
        self.action_key = action_key
        self.limit = resolved_budget
        self.timeout = float(expert_timeout)
        self.attempts = 0
        self.auto_record_lessons = bool(auto_record_lessons)
        self.auto_online_update = bool(auto_online_update)
        self.lesson_split = str(lesson_split)
        self.lesson_source = str(lesson_source)
        self.lessons_recorded = 0
        self.online_updates_applied = 0
        self._busy = False
        self._halted = False
        self._pending: Optional[asyncio.Task[Any]] = None
        self._reviewed: Any = None

    def close(self) -> None:
        """Halt this runner and cancel any in-flight expert task."""
        self._halted = True
        if self._pending is not None:
            self._pending.cancel()

    def _stop(self, decision: Dict[str, Any], reason: str) -> ReviewFallbackDecision:
        self.close()
        out: Dict[str, Any] = {
            **decision,
            self.action_key: None,
            "decision": None,
            "recovered": False,
            "used_fallback": False,
            "is_ambiguous": True,
            "recovery_reason": reason,
            "expert_attempts": self.attempts,
            "lessons_recorded": self.lessons_recorded,
            "online_updates_applied": self.online_updates_applied,
            "lesson_recorded": False,
            "online_update_applied": False,
        }
        if "values" in out:
            out["values"] = None
        return ReviewFallbackDecision(out)

    def _drain(self, task: asyncio.Task[Any]) -> None:
        if not task.cancelled():
            task.exception()
        if self._pending is task:
            self._pending = None

    def _check_done(self, context: Any) -> bool:
        if self.is_done is not None:
            return _checked(self.is_done, context)
        return bool(getattr(context, "done", False))

    def _run_local(self, context: Any) -> Dict[str, Any]:
        prompt = _checked_value(self.prompt_extractor, context) if self.prompt_extractor is not None else context
        # Support TeachingSession.decision_details
        if hasattr(self.network, "decision_details") and isinstance(prompt, str):
            details = dict(self.network.decision_details(prompt))
            values = details.get("values", {})
            if self.action_key not in details:
                if isinstance(values, Mapping) and self.action_key in values:
                    details[self.action_key] = values[self.action_key]
                elif isinstance(values, Mapping) and len(values) == 1:
                    details[self.action_key] = next(iter(values.values()))
                else:
                    details[self.action_key] = values
            details.setdefault("calls", 1)
            return details

        # Support SystemOneEngine.decide(prompt, strict=True)
        if hasattr(self.network, "decide") and hasattr(self.network, "strict_mode") and isinstance(prompt, str):
            res = self.network.decide(prompt, strict=True, record_receipt=False)
            values = dict(res.values)
            if self.action_key in values:
                action_val: Any = values[self.action_key]
            elif len(values) == 1:
                action_val = next(iter(values.values()))
            else:
                action_val = values
            return {
                self.action_key: action_val,
                "values": values,
                "review": bool(res.is_ambiguous),
                "calls": 1,
                "conformal_sets": res.conformal_sets,
                "confidences": res.confidences,
                "probabilities": res.probabilities,
                "abstention_reason": getattr(res, "abstention_reason", None),
                "abstention_reasons": getattr(res, "abstention_reasons", {}),
            }

        # Support CompiledSystemOneModel via ephemeral strict SystemOneEngine
        if not hasattr(self.network, "decide") and hasattr(self.network, "heads") and hasattr(self.network, "forward_single") and isinstance(prompt, str):
            from system1.engine import SystemOneEngine
            eng = SystemOneEngine(self.network.schema, compiled_model=self.network, strict_mode=True, use_cache=False)
            res = eng.decide(prompt, strict=True, record_receipt=False)
            values = dict(res.values)
            action_val = (
                values[self.action_key]
                if self.action_key in values
                else (next(iter(values.values())) if len(values) == 1 else values)
            )
            return {
                self.action_key: action_val,
                "values": values,
                "review": bool(res.is_ambiguous),
                "calls": 1,
                "conformal_sets": res.conformal_sets,
                "confidences": res.confidences,
                "probabilities": res.probabilities,
                "abstention_reason": getattr(res, "abstention_reason", None),
                "abstention_reasons": getattr(res, "abstention_reasons", {}),
            }

        raw = self.network.decide(prompt)
        if inspect.isawaitable(raw):
            if inspect.iscoroutine(raw):
                raw.close()
            raise TypeError("Local network decide() must be synchronous")
        if isinstance(raw, Mapping):
            return dict(raw)
        if hasattr(raw, "values") and hasattr(raw, "is_ambiguous"):
            values = dict(raw.values)
            action_val = (
                values[self.action_key]
                if self.action_key in values
                else (next(iter(values.values())) if len(values) == 1 else values)
            )
            return {
                self.action_key: action_val,
                "values": values,
                "review": bool(raw.is_ambiguous),
                "calls": 1,
            }
        raise TypeError("Unsupported local decision result")

    def _normalize_candidate_answer(self, answer: Any) -> Any:
        if hasattr(answer, "values") and isinstance(getattr(answer, "values", None), Mapping):
            return dict(answer.values)
        return answer

    def _validate_schema_and_guard(self, context: Any, answer: Any) -> bool:
        norm_answer = self._normalize_candidate_answer(answer)
        if self.schema is not None:
            try:
                if isinstance(norm_answer, Mapping):
                    if set(norm_answer.keys()) != set(self.schema.fields.keys()):
                        return False
                    for fname, fdef in self.schema.fields.items():
                        fdef.validate_value(norm_answer[fname])
                elif len(self.schema.fields) == 1:
                    fdef = next(iter(self.schema.fields.values()))
                    fdef.validate_value(norm_answer)
                else:
                    return False
            except (TypeError, ValueError):
                return False

        if self.validate is not None and not _checked(self.validate, context, answer):
            return False

        if self.guard is not None:
            if callable(self.guard):
                guard_res = _checked_value(self.guard, context, answer)
                if hasattr(guard_res, "allowed"):
                    if guard_res.allowed is not True:
                        return False
                elif guard_res is not True:
                    return False
            elif hasattr(self.guard, "evaluate_proposal"):
                from system1.guard import ActionProposal

                if isinstance(answer, ActionProposal):
                    proposal = answer
                    prompt_str = str(context) if isinstance(context, str) else None
                elif isinstance(context, ActionProposal):
                    proposal = context
                    prompt_str = f"{context.purpose} -> {answer}"
                else:
                    args_map = dict(norm_answer) if isinstance(norm_answer, Mapping) else {self.action_key: norm_answer}
                    target_str = str(norm_answer) if str(norm_answer) and "\x00" not in str(norm_answer) else f"candidate:{self.action_key}"
                    purpose_str = context if isinstance(context, str) and context.strip() else "Bounded review fallback candidate validation"
                    proposal = ActionProposal.create(
                        tenant_id="default",
                        principal_id="bounded_fallback",
                        scope="recovery",
                        tool=self.action_key if self.action_key.isalnum() else "fallback_action",
                        arguments=args_map,
                        purpose=purpose_str,
                        canonical_target=target_str[:4096],
                    )
                    prompt_str = f"{purpose_str}: {norm_answer}"
                guard_res = self.guard.evaluate_proposal(proposal, context_prompt=prompt_str)
                if getattr(guard_res, "allowed", False) is not True:
                    return False
            else:
                return False
        return True

    def _build_validated_decision(
        self,
        context: Any,
        decision: Dict[str, Any],
        answer: Any,
    ) -> ReviewFallbackDecision:
        norm_answer = self._normalize_candidate_answer(answer)
        if isinstance(norm_answer, Mapping):
            resolved_values: Dict[str, Any] = dict(norm_answer)
            if self.schema is not None:
                resolved_values = self.schema.validate_decision(resolved_values)
        elif self.schema is not None and len(self.schema.fields) == 1:
            only_field, only_def = next(iter(self.schema.fields.items()))
            resolved_values = {only_field: only_def.validate_value(norm_answer)}
        else:
            resolved_values = {self.action_key: norm_answer}

        step_lesson_recorded = False
        step_online_updated = False
        prompt = _checked_value(self.prompt_extractor, context) if self.prompt_extractor is not None else context
        prompt_text = prompt if isinstance(prompt, str) else str(prompt)

        if self.auto_record_lessons and self.teaching_session is not None:
            try:
                if hasattr(self.teaching_session, "teach"):
                    self.teaching_session.teach(
                        prompt_text,
                        resolved_values,
                        source=self.lesson_source,
                        split=self.lesson_split,
                    )
                elif hasattr(self.teaching_session, "record"):
                    eff_split = "teach" if self.lesson_split in ("train", "auto") else self.lesson_split
                    self.teaching_session.record(
                        prompt_text,
                        resolved_values,
                        source=self.lesson_source,
                        split=eff_split,
                    )
                self.lessons_recorded += 1
                step_lesson_recorded = True
            except ValueError as exc:
                msg = str(exc)
                if "cannot appear in more than one split" not in msg and "cannot appear in both" not in msg:
                    raise

        if self.auto_online_update:
            updated_any = False
            if hasattr(self.network, "learn_from_tier2"):
                self.network.learn_from_tier2(
                    prompt_text,
                    resolved_values,
                    preserve_calibration=True,
                )
                updated_any = True
            if (
                self.teaching_session is not None
                and self.teaching_session is not self.network
                and hasattr(self.teaching_session, "learn_from_tier2")
            ):
                ts_engine = getattr(self.teaching_session, "_engine", None)
                if ts_engine is not None and ts_engine is not self.network:
                    self.teaching_session.learn_from_tier2(
                        prompt_text,
                        resolved_values,
                        preserve_calibration=True,
                    )
                    updated_any = True
                elif not updated_any:
                    try:
                        self.teaching_session.learn_from_tier2(
                            prompt_text,
                            resolved_values,
                            preserve_calibration=True,
                        )
                        updated_any = True
                    except ValueError:
                        pass
            if updated_any:
                self.online_updates_applied += 1
                step_online_updated = True
                self._reviewed = None

        if isinstance(answer, Mapping):
            if self.action_key in answer:
                action_val = answer[self.action_key]
            elif len(resolved_values) == 1 and self.action_key != "action":
                action_val = next(iter(resolved_values.values()))
            else:
                action_val = dict(resolved_values)
        else:
            action_val = answer

        out: Dict[str, Any] = {
            **decision,
            self.action_key: action_val,
            "values": dict(resolved_values),
            "decision": dict(resolved_values),
            "recovered": True,
            "used_fallback": True,
            "is_ambiguous": False,
            "recovery_reason": "validated_expert",
            "expert_attempts": self.attempts,
            "lessons_recorded": self.lessons_recorded,
            "online_updates_applied": self.online_updates_applied,
            "lesson_recorded": step_lesson_recorded,
            "online_update_applied": step_online_updated,
        }
        for k, v in resolved_values.items():
            if k not in out:
                out[k] = v
        return ReviewFallbackDecision(out)

    async def _finish_async_decide(
        self,
        context: Any,
        state: Any,
        decision: Dict[str, Any],
        awaitable_answer: Awaitable[Any],
    ) -> ReviewFallbackDecision:
        try:
            async def invoke() -> Any:
                return await awaitable_answer

            task = self._pending = asyncio.create_task(invoke())
            task.add_done_callback(self._drain)
            deadline = time.monotonic() + self.timeout
            done, _ = await asyncio.wait({task}, timeout=self.timeout)
            if self._halted:
                return self._stop(decision, "runner_stopped")
            if not done or time.monotonic() >= deadline:
                return self._stop(decision, "expert_timeout")
            if task.cancelled():
                return self._stop(decision, "expert_cancelled")
            try:
                answer = task.result()
            except Exception:
                return self._stop(decision, "expert_failed")
            if _checked_value(self.observe_state, context) != state:
                return self._stop(decision, "stale_context")
            if not _checked(self.allowed, context):
                return self._stop(decision, "recovery_not_allowed")
            if self._halted or _checked_value(self.observe_state, context) != state:
                return self._stop(decision, "stale_context")
            if not self._validate_schema_and_guard(context, answer):
                return self._stop(decision, "candidate_rejected")
            # Validators may mutate context, revoke permission or close this runner.
            if not _checked(self.allowed, context):
                return self._stop(decision, "recovery_not_allowed")
            if self._halted or _checked_value(self.observe_state, context) != state or time.monotonic() >= deadline:
                return self._stop(decision, "validation_interrupted")

            return self._build_validated_decision(context, decision, answer)
        except asyncio.CancelledError:
            self.close()
            raise
        except Exception:
            return self._stop(decision, "callback_or_context_failed")
        finally:
            self._busy = False

    def decide(self, context: Any) -> Any:
        """Run strict local decision and optionally recover from review via bounded expert.

        Returns a ``ReviewFallbackDecision`` (which is also awaitable for ``async`` callers)
        when completed synchronously, or a coroutine resolving to ``ReviewFallbackDecision``
        when the configured ``expert`` / ``review_fn`` returns an awaitable in a running event loop.
        """
        if self._halted or self._busy:
            return self._stop({"review": True, "calls": 0}, "runner_stopped")
        self._busy = True
        decision: Dict[str, Any] = {"review": True, "calls": None}
        delegated_async = False
        try:
            if self._check_done(context):
                return self._stop({"review": True, "calls": 0}, "episode_finished")
            state = _checked_value(self.observe_state, context)
            decision = self._run_local(context)
            if self._halted or _checked_value(self.observe_state, context) != state:
                return self._stop(decision, "stale_context")
            if not decision.get("review", False):
                local_vals = decision.get("values")
                if not isinstance(local_vals, Mapping) and self.action_key in decision:
                    local_vals = {self.action_key: decision[self.action_key]}
                out_local: Dict[str, Any] = {
                    **decision,
                    "values": dict(local_vals) if isinstance(local_vals, Mapping) else local_vals,
                    "decision": dict(local_vals) if isinstance(local_vals, Mapping) else local_vals,
                    "recovered": False,
                    "used_fallback": False,
                    "is_ambiguous": False,
                    "recovery_reason": "local_accepted",
                    "expert_attempts": self.attempts,
                    "lessons_recorded": self.lessons_recorded,
                    "online_updates_applied": self.online_updates_applied,
                    "lesson_recorded": False,
                    "online_update_applied": False,
                }
                if isinstance(local_vals, Mapping):
                    for k, v in local_vals.items():
                        if k not in out_local:
                            out_local[k] = v
                return ReviewFallbackDecision(out_local)
            if state == self._reviewed:
                return self._stop(decision, "review_already_attempted")
            self._reviewed = state
            if not all(callable(cb) for cb in (self.expert, self.allowed, self.validate)):
                return self._stop(decision, "expert_not_configured")
            if not _checked(self.allowed, context):
                return self._stop(decision, "recovery_not_allowed")
            if self._halted or _checked_value(self.observe_state, context) != state:
                return self._stop(decision, "stale_context")
            if self.attempts >= self.limit:
                return self._stop(decision, "budget_exhausted")
            self.attempts += 1

            deadline = time.monotonic() + self.timeout
            try:
                assert self.expert is not None
                raw_answer = self.expert(context)
            except Exception:
                return self._stop(decision, "expert_failed")

            if inspect.isawaitable(raw_answer):
                try:
                    asyncio.get_running_loop()
                    has_running_loop = True
                except RuntimeError:
                    has_running_loop = False

                if has_running_loop:
                    delegated_async = True
                    return self._finish_async_decide(context, state, decision, raw_answer)
                else:
                    delegated_async = True
                    return asyncio.run(self._finish_async_decide(context, state, decision, raw_answer))

            # Synchronous review_fn / expert completion path
            answer = raw_answer
            if self._halted:
                return self._stop(decision, "runner_stopped")
            if time.monotonic() >= deadline:
                return self._stop(decision, "expert_timeout")
            if _checked_value(self.observe_state, context) != state:
                return self._stop(decision, "stale_context")
            if not _checked(self.allowed, context):
                return self._stop(decision, "recovery_not_allowed")
            if self._halted or _checked_value(self.observe_state, context) != state:
                return self._stop(decision, "stale_context")
            if not self._validate_schema_and_guard(context, answer):
                return self._stop(decision, "candidate_rejected")
            if not _checked(self.allowed, context):
                return self._stop(decision, "recovery_not_allowed")
            if self._halted or _checked_value(self.observe_state, context) != state or time.monotonic() >= deadline:
                return self._stop(decision, "validation_interrupted")

            return self._build_validated_decision(context, decision, answer)
        except Exception:
            return self._stop(decision, "callback_or_context_failed")
        finally:
            if not delegated_async:
                self._busy = False


__all__ = ["BoundedReviewFallback", "ReviewFallbackDecision", "default_observed_state"]
