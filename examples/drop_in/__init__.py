"""System 1: Instinct Drop-In Replacements for Semantic Router, RouteLLM, Instructor, and Prompt-Guard."""

from examples.drop_in.instructor_structured_dropin import (
    IncidentTriageModel,
    StructuredExtractionResult,
    SystemOneInstructorClient,
)
from examples.drop_in.prompt_guard_dropin import (
    PromptGuardResult,
    PromptGuardSchema,
    SystemOnePromptGuard,
)
from examples.drop_in.routellm_dropin import (
    RouteDecision,
    RouteLLMCascadeSchema,
    SystemOneRouteController,
)
from examples.drop_in.semantic_router_dropin import (
    InstinctRouteLayer,
    RouteMatchResult,
    SemanticRoute,
    build_default_router,
)

__all__ = [
    "IncidentTriageModel",
    "InstinctRouteLayer",
    "PromptGuardResult",
    "PromptGuardSchema",
    "RouteDecision",
    "RouteLLMCascadeSchema",
    "RouteMatchResult",
    "SemanticRoute",
    "StructuredExtractionResult",
    "SystemOneInstructorClient",
    "SystemOnePromptGuard",
    "SystemOneRouteController",
    "build_default_router",
]
