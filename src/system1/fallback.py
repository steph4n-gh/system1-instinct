"""Bounded strict-review fallback coordinator for System 1 local skills.

When a strict local skill abstains (requests review), BoundedReviewFallback can
optionally consult an asynchronous read-only expert under explicit permission,
state-freshness checks, synchronous schema/domain validation, and a hard attempt
budget. Any invariant violation fails closed and halts the runner.
"""

from __future__ import annotations

import asyncio
import collections.abc
from hashlib import sha256
import inspect
import json
import math
import threading
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


def _cutover_owns_network(lc_obj: Any, net: Any) -> bool:
    if lc_obj is None or net is None:
        return False
    if net is getattr(lc_obj, "engine", None):
        return True
    checker = getattr(lc_obj, "_is_owned_engine", None)
    if callable(checker) and bool(checker(net)):
        return True
    inner_lc = getattr(lc_obj, "lifecycle", None)
    if inner_lc is not None:
        if net is getattr(inner_lc, "engine", None):
            return True
        inner_checker = getattr(inner_lc, "_is_owned_engine", None)
        if callable(inner_checker) and bool(inner_checker(net)):
            return True
    return False


def _cancel_handle_threadsafe(handle: Any) -> None:
    """Cancel an asyncio Task/Future safely whether invoked inside or outside its owning loop."""
    if handle is None:
        return
    done_fn = getattr(handle, "done", None)
    if callable(done_fn):
        try:
            if done_fn():
                return
        except Exception:
            pass
    loop_fn = getattr(handle, "get_loop", None)
    h_loop: Optional[asyncio.AbstractEventLoop] = None
    if callable(loop_fn):
        try:
            cand_loop = loop_fn()
            if isinstance(cand_loop, asyncio.AbstractEventLoop):
                h_loop = cand_loop
        except Exception:
            h_loop = None
    if h_loop is None:
        cancel_fn = getattr(handle, "cancel", None)
        if callable(cancel_fn):
            cancel_fn()
        return
    if h_loop.is_closed():
        return
    try:
        cur_loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
    except RuntimeError:
        cur_loop = None
    if cur_loop is h_loop:
        handle.cancel()
    else:
        try:
            h_loop.call_soon_threadsafe(handle.cancel)
        except RuntimeError:
            pass


class _TrackedAwaitIterator:
    """Iterator adapter routing awaitable protocol operations through _TrackedDecisionCoroutine."""

    def __init__(self, parent: "_TrackedDecisionCoroutine") -> None:
        self._parent = parent

    def __iter__(self) -> "_TrackedAwaitIterator":
        return self

    def __next__(self) -> Any:
        return self._parent.send(None)

    def send(self, value: Any) -> Any:
        return self._parent.send(value)

    def throw(self, *args: Any) -> Any:
        return self._parent.throw(*args)

    def close(self) -> None:
        self._parent.close()


class _TrackedDecisionCoroutine(collections.abc.Coroutine):
    """Coroutine wrapper that releases unstarted async fallback/canary state on close or pre-start cancel."""

    def __init__(
        self,
        runner: "BoundedReviewFallback",
        coro: Any,
        raw_answer: Any,
        tx_pair: Optional[List[Any]],
    ) -> None:
        self._runner = runner
        self._coro = coro
        self._raw_answer = raw_answer
        self._tx_pair = tx_pair
        self._started_inner = False
        self._cleaned = False

    def _cleanup(self) -> None:
        with self._runner._state_lock:
            if self._cleaned:
                return
            self._cleaned = True
            was_unstarted = not self._started_inner
            if was_unstarted:
                self._runner._busy = False
        if was_unstarted:
            if inspect.iscoroutine(self._raw_answer):
                self._raw_answer.close()
            elif hasattr(self._raw_answer, "cancel") and callable(self._raw_answer.cancel):
                _cancel_handle_threadsafe(self._raw_answer)
            if self._tx_pair is not None:
                self._runner._rollback_cutover_tx(self._tx_pair)
            if inspect.iscoroutine(self._coro):
                try:
                    self._coro.close()
                except Exception:
                    pass

    def send(self, value: Any) -> Any:
        with self._runner._state_lock:
            if not self._started_inner and value is None:
                self._started_inner = True
        try:
            return self._coro.send(value)
        except BaseException:
            self._cleanup()
            raise

    def throw(self, typ: Any, val: Any = None, tb: Any = None) -> Any:
        with self._runner._state_lock:
            unstarted = not self._started_inner
        if unstarted:
            is_cancel = (
                (isinstance(typ, type) and issubclass(typ, asyncio.CancelledError))
                or isinstance(typ, asyncio.CancelledError)
                or isinstance(val, asyncio.CancelledError)
            )
            if is_cancel:
                self._runner.close()
        try:
            if val is None and tb is None:
                return self._coro.throw(typ)
            if tb is None:
                return self._coro.throw(typ, val)
            return self._coro.throw(typ, val, tb)
        except BaseException:
            self._cleanup()
            raise

    def close(self) -> None:
        try:
            self._coro.close()
        finally:
            self._cleanup()

    def __await__(self) -> _TrackedAwaitIterator:
        return _TrackedAwaitIterator(self)

    def __del__(self) -> None:
        if not getattr(self, "_cleaned", True):
            try:
                self.close()
            except Exception:
                pass


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
        aci_tracker: Optional[Any] = None,
        auto_compile: bool = False,
        auto_compile_config: Optional[Any] = None,
        auto_compile_lifecycle: Optional[Any] = None,
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
            or bool(auto_compile)
            or auto_compile_lifecycle is not None
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
        self.aci_tracker = aci_tracker if aci_tracker is not None else getattr(network, "aci_tracker", None)
        self.auto_compile = bool(auto_compile)
        if auto_compile_lifecycle is not None:
            self.cutover_lifecycle: Optional[Any] = auto_compile_lifecycle
        elif self.auto_compile:
            if self.schema is None:
                raise ValueError("BoundedReviewFallback auto_compile=True requires a DecisionSchema")
            from system1.cutover import AutoCutoverLifecycle

            self.cutover_lifecycle = AutoCutoverLifecycle(
                self.schema,
                teacher=None,
                config=auto_compile_config,
            )
        else:
            self.cutover_lifecycle = None
        self._network_owned_by_cutover: bool = _cutover_owns_network(
            self.cutover_lifecycle, self.network
        )
        self._state_lock = threading.Lock()
        self._busy = False
        self._halted = False
        self._pending: Optional[asyncio.Task[Any]] = None
        self._reviewed: Any = None

    def close(self) -> None:
        """Halt this runner and cancel any in-flight expert task."""
        with self._state_lock:
            self._halted = True
            pending = self._pending
        if pending is not None:
            _cancel_handle_threadsafe(pending)

    def _stop(self, decision: Dict[str, Any], reason: str) -> ReviewFallbackDecision:
        self.close()
        with self._state_lock:
            attempts = self.attempts
            lessons = self.lessons_recorded
            updates = self.online_updates_applied
        out: Dict[str, Any] = {
            **decision,
            self.action_key: None,
            "decision": None,
            "recovered": False,
            "used_fallback": False,
            "is_ambiguous": True,
            "recovery_reason": reason,
            "expert_attempts": attempts,
            "lessons_recorded": lessons,
            "online_updates_applied": updates,
            "lesson_recorded": False,
            "online_update_applied": False,
        }
        if "values" in out:
            out["values"] = None
        return ReviewFallbackDecision(out)

    def _drain(self, task: asyncio.Task[Any]) -> None:
        if not task.cancelled():
            task.exception()
        with self._state_lock:
            if self._pending is task:
                self._pending = None

    def _check_done(self, context: Any) -> bool:
        if self.is_done is not None:
            return _checked(self.is_done, context)
        return bool(getattr(context, "done", False))

    def _rollback_cutover_tx(self, tx_pair: List[Any]) -> None:
        if len(tx_pair) != 2 or tx_pair[0] is None or tx_pair[1] is None:
            return
        lc_obj, tx_tok = tx_pair[0], tx_pair[1]
        tx_pair[0] = None
        tx_pair[1] = None
        if hasattr(lc_obj, "_rollback_local_tx"):
            lc_obj._rollback_local_tx(tx_tok)

    def _run_local(
        self,
        context: Any,
        *,
        _tx_out: Optional[List[Any]] = None,
        _cutover_accepted_out: Optional[List[bool]] = None,
    ) -> Dict[str, Any]:
        if _cutover_accepted_out is not None and len(_cutover_accepted_out) >= 1:
            _cutover_accepted_out[0] = False
        prompt = _checked_value(self.prompt_extractor, context) if self.prompt_extractor is not None else context
        if self.cutover_lifecycle is not None and isinstance(prompt, str):
            if _tx_out is not None and hasattr(self.cutover_lifecycle, "_decide_local_with_tx"):
                local_vals, local_tx = self.cutover_lifecycle._decide_local_with_tx(prompt)
                _tx_out[:] = [self.cutover_lifecycle, local_tx]
            else:
                local_vals = self.cutover_lifecycle.decide_local(prompt)
            if local_vals is not None:
                if _cutover_accepted_out is not None and len(_cutover_accepted_out) >= 1:
                    _cutover_accepted_out[0] = True
                if self.action_key in local_vals:
                    action_val: Any = local_vals[self.action_key]
                elif len(local_vals) == 1:
                    action_val = next(iter(local_vals.values()))
                else:
                    action_val = dict(local_vals)
                return {
                    self.action_key: action_val,
                    "values": dict(local_vals),
                    "review": False,
                    "calls": 1,
                    "auto_compiled": True,
                    "cutover_state": self.cutover_lifecycle.state,
                }
            lc_state_raw = getattr(self.cutover_lifecycle, "state", None)
            lc_state = str(getattr(lc_state_raw, "value", lc_state_raw))
            lc_drifted = (
                bool(getattr(self.cutover_lifecycle, "pending_recalibrate", False))
                or lc_state == "DRIFT_RECALIBRATE"
            )
            net_is_lc_engine = (
                self._network_owned_by_cutover
                or _cutover_owns_network(self.cutover_lifecycle, self.network)
            )
            if lc_drifted or net_is_lc_engine:
                return {
                    self.action_key: None,
                    "values": {},
                    "review": True,
                    "calls": 1,
                    "auto_compiled": False,
                    "cutover_state": lc_state,
                    "abstention_reason": "drift_recalibrate" if lc_drifted else "cutover_escalate",
                }
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
        *,
        _tx_pair: Optional[List[Any]] = None,
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
                with self._state_lock:
                    self.lessons_recorded += 1
                step_lesson_recorded = True
            except ValueError as exc:
                msg = str(exc)
                if "cannot appear in more than one split" not in msg and "cannot appear in both" not in msg:
                    raise

        if self.auto_online_update:
            updated_any = False
            reward_scale_kw: Dict[str, Any] = {}
            if self.aci_tracker is not None:
                reward_scale_kw["reward_scale"] = getattr(self.aci_tracker, "escalation_learning_rate_scale", 0.25)
                local_vals = decision.get("values") if isinstance(decision.get("values"), Mapping) else {}
                margins_map = decision.get("margins") if isinstance(decision.get("margins"), Mapping) else {}
                pred_wrong = any(local_vals.get(k) != v for k, v in resolved_values.items()) if local_vals else False
                min_margin = min((float(m) for m in margins_map.values()), default=None) if margins_map else None
                self.aci_tracker.record_escalation(predicted_wrong=pred_wrong, margin=min_margin)
            if hasattr(self.network, "learn_from_tier2"):
                self.network.learn_from_tier2(
                    prompt_text,
                    resolved_values,
                    preserve_calibration=True,
                    **reward_scale_kw,
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
                with self._state_lock:
                    self.online_updates_applied += 1
                    self._reviewed = None
                step_online_updated = True

        if self.cutover_lifecycle is not None:
            from system1.cutover import _bind_active_canary_tx

            lc_target = self.cutover_lifecycle
            active_tx = (
                _tx_pair[1]
                if (_tx_pair is not None and len(_tx_pair) == 2 and _tx_pair[0] is lc_target)
                else None
            )
            if active_tx is not None:
                with _bind_active_canary_tx(lc_target, tx_token=active_tx):
                    lc_target.observe_teacher(prompt_text, resolved_values)
                _tx_pair[:] = [None, None]
            else:
                lc_target.observe_teacher(prompt_text, resolved_values)
            with self._state_lock:
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

        with self._state_lock:
            attempts = self.attempts
            lessons = self.lessons_recorded
            updates = self.online_updates_applied
        out: Dict[str, Any] = {
            **decision,
            self.action_key: action_val,
            "values": dict(resolved_values),
            "decision": dict(resolved_values),
            "recovered": True,
            "used_fallback": True,
            "is_ambiguous": False,
            "recovery_reason": "validated_expert",
            "expert_attempts": attempts,
            "lessons_recorded": lessons,
            "online_updates_applied": updates,
            "lesson_recorded": step_lesson_recorded,
            "online_update_applied": step_online_updated,
        }
        if self.cutover_lifecycle is not None:
            out["cutover_state"] = self.cutover_lifecycle.state
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
        tx_pair: Optional[List[Any]] = None,
    ) -> ReviewFallbackDecision:
        started = False
        dispatch_refusal: Optional[str] = None
        try:
            # decide() may have returned this coroutine before the caller revokes
            # permission, changes context or closes the runner. Check before queueing
            # and again when the scheduled task actually starts on the event loop.
            if self._halted:
                return self._stop(decision, "runner_stopped")
            if _checked_value(self.observe_state, context) != state:
                return self._stop(decision, "stale_context")
            if not _checked(self.allowed, context):
                return self._stop(decision, "recovery_not_allowed")
            if self._halted or _checked_value(self.observe_state, context) != state:
                return self._stop(decision, "stale_context")

            async def invoke() -> Any:
                nonlocal started, dispatch_refusal
                try:
                    if self._halted:
                        dispatch_refusal = "runner_stopped"
                    elif _checked_value(self.observe_state, context) != state:
                        dispatch_refusal = "stale_context"
                    elif not _checked(self.allowed, context):
                        dispatch_refusal = "recovery_not_allowed"
                    elif self._halted or _checked_value(self.observe_state, context) != state:
                        dispatch_refusal = "stale_context"
                except Exception:
                    dispatch_refusal = "callback_or_context_failed"
                if dispatch_refusal is not None:
                    return None
                started = True
                return await awaitable_answer

            task = asyncio.create_task(invoke())
            with self._state_lock:
                self._pending = task
                already_halted = self._halted
            if already_halted:
                task.cancel()
            task.add_done_callback(self._drain)
            deadline = time.monotonic() + self.timeout
            done, _ = await asyncio.wait({task}, timeout=self.timeout)
            with self._state_lock:
                halted_after_wait = self._halted
            if halted_after_wait:
                return self._stop(decision, "runner_stopped")
            if not done or time.monotonic() >= deadline:
                return self._stop(decision, "expert_timeout")
            if task.cancelled():
                return self._stop(decision, "expert_cancelled")
            try:
                answer = task.result()
            except Exception:
                return self._stop(decision, "expert_failed")
            if dispatch_refusal is not None:
                return self._stop(decision, dispatch_refusal)
            if _checked_value(self.observe_state, context) != state:
                return self._stop(decision, "stale_context")
            if not _checked(self.allowed, context):
                return self._stop(decision, "recovery_not_allowed")
            state_after_allow_a = _checked_value(self.observe_state, context)
            with self._state_lock:
                halted_after_allow_a = self._halted
            if halted_after_allow_a or state_after_allow_a != state:
                return self._stop(decision, "stale_context")
            if not self._validate_schema_and_guard(context, answer):
                return self._stop(decision, "candidate_rejected")
            # Validators may mutate context, revoke permission or close this runner.
            if not _checked(self.allowed, context):
                return self._stop(decision, "recovery_not_allowed")
            state_after_val_a = _checked_value(self.observe_state, context)
            with self._state_lock:
                halted_after_val_a = self._halted
            if halted_after_val_a or state_after_val_a != state or time.monotonic() >= deadline:
                return self._stop(decision, "validation_interrupted")

            return self._build_validated_decision(context, decision, answer, _tx_pair=tx_pair)
        except asyncio.CancelledError:
            self.close()
            raise
        except Exception:
            return self._stop(decision, "callback_or_context_failed")
        finally:
            if not started:
                if inspect.iscoroutine(awaitable_answer):
                    awaitable_answer.close()
                elif hasattr(awaitable_answer, "cancel") and callable(awaitable_answer.cancel):
                    _cancel_handle_threadsafe(awaitable_answer)
            if tx_pair is not None:
                self._rollback_cutover_tx(tx_pair)
            with self._state_lock:
                self._busy = False

    def decide(self, context: Any) -> Any:
        """Run strict local decision and optionally recover from review via bounded expert.

        Returns a ``ReviewFallbackDecision`` (which is also awaitable for ``async`` callers)
        when completed synchronously, or a coroutine resolving to ``ReviewFallbackDecision``
        when the configured ``expert`` / ``review_fn`` returns an awaitable in a running event loop.
        """
        with self._state_lock:
            if self._halted or self._busy:
                already_stopped = True
            else:
                self._busy = True
                already_stopped = False
        if already_stopped:
            return self._stop({"review": True, "calls": 0}, "runner_stopped")
        decision: Dict[str, Any] = {"review": True, "calls": None}
        delegated_async = False
        tx_pair: List[Any] = [None, None]
        cutover_accepted: List[bool] = [False]
        try:
            if self._check_done(context):
                return self._stop({"review": True, "calls": 0}, "episode_finished")
            state = _checked_value(self.observe_state, context)
            decision = self._run_local(
                context,
                _tx_out=tx_pair,
                _cutover_accepted_out=cutover_accepted,
            )
            state_after_local = _checked_value(self.observe_state, context)
            with self._state_lock:
                halted_after_local = self._halted
            if halted_after_local or state_after_local != state:
                return self._stop(decision, "stale_context")
            if not decision.get("review", False):
                if cutover_accepted[0]:
                    tx_pair[:] = [None, None]
                else:
                    self._rollback_cutover_tx(tx_pair)
                local_vals = decision.get("values")
                if not isinstance(local_vals, Mapping) and self.action_key in decision:
                    local_vals = {self.action_key: decision[self.action_key]}
                with self._state_lock:
                    attempts = self.attempts
                    lessons = self.lessons_recorded
                    updates = self.online_updates_applied
                out_local: Dict[str, Any] = {
                    **decision,
                    "values": dict(local_vals) if isinstance(local_vals, Mapping) else local_vals,
                    "decision": dict(local_vals) if isinstance(local_vals, Mapping) else local_vals,
                    "recovered": False,
                    "used_fallback": False,
                    "is_ambiguous": False,
                    "recovery_reason": "local_accepted",
                    "expert_attempts": attempts,
                    "lessons_recorded": lessons,
                    "online_updates_applied": updates,
                    "lesson_recorded": False,
                    "online_update_applied": False,
                }
                if isinstance(local_vals, Mapping):
                    for k, v in local_vals.items():
                        if k not in out_local:
                            out_local[k] = v
                return ReviewFallbackDecision(out_local)
            with self._state_lock:
                if state == self._reviewed:
                    already_reviewed = True
                else:
                    self._reviewed = state
                    already_reviewed = False
            if already_reviewed:
                return self._stop(decision, "review_already_attempted")
            if not all(callable(cb) for cb in (self.expert, self.allowed, self.validate)):
                return self._stop(decision, "expert_not_configured")
            if not _checked(self.allowed, context):
                return self._stop(decision, "recovery_not_allowed")
            state_after_allow = _checked_value(self.observe_state, context)
            with self._state_lock:
                if self._halted or state_after_allow != state:
                    stop_pre_expert: Optional[str] = "stale_context"
                elif self.attempts >= self.limit:
                    stop_pre_expert = "budget_exhausted"
                else:
                    self.attempts += 1
                    stop_pre_expert = None
            if stop_pre_expert is not None:
                return self._stop(decision, stop_pre_expert)

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
                    return _TrackedDecisionCoroutine(
                        self,
                        self._finish_async_decide(context, state, decision, raw_answer, tx_pair),
                        raw_answer,
                        tx_pair,
                    )
                else:
                    delegated_async = True
                    return asyncio.run(self._finish_async_decide(context, state, decision, raw_answer, tx_pair))

            # Synchronous review_fn / expert completion path
            answer = raw_answer
            with self._state_lock:
                halted_after_expert = self._halted
            if halted_after_expert:
                return self._stop(decision, "runner_stopped")
            if time.monotonic() >= deadline:
                return self._stop(decision, "expert_timeout")
            if _checked_value(self.observe_state, context) != state:
                return self._stop(decision, "stale_context")
            if not _checked(self.allowed, context):
                return self._stop(decision, "recovery_not_allowed")
            state_after_sallow = _checked_value(self.observe_state, context)
            with self._state_lock:
                halted_after_sallow = self._halted
            if halted_after_sallow or state_after_sallow != state:
                return self._stop(decision, "stale_context")
            if not self._validate_schema_and_guard(context, answer):
                return self._stop(decision, "candidate_rejected")
            if not _checked(self.allowed, context):
                return self._stop(decision, "recovery_not_allowed")
            state_after_sval = _checked_value(self.observe_state, context)
            with self._state_lock:
                halted_after_sval = self._halted
            if halted_after_sval or state_after_sval != state or time.monotonic() >= deadline:
                return self._stop(decision, "validation_interrupted")

            return self._build_validated_decision(context, decision, answer, _tx_pair=tx_pair)
        except Exception:
            return self._stop(decision, "callback_or_context_failed")
        finally:
            if not delegated_async:
                self._rollback_cutover_tx(tx_pair)
                with self._state_lock:
                    self._busy = False


__all__ = ["BoundedReviewFallback", "ReviewFallbackDecision", "default_observed_state"]
