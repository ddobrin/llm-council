"""Core ADK pipeline orchestration for LLM Council."""

import asyncio
import weakref
from typing import Any, Callable, Dict, List, Optional
import httpx

from google.genai import types
from google.adk.models import (
    BaseLlm,
    Claude,
    Gemini,
    AnthropicGenerateContentConfig,
    LlmRequest,
    LlmResponse,
)
from google.adk.agents import LlmAgent, Context
from google.adk.workflow import Workflow, START, FunctionNode, JoinNode, Edge
from google.adk.events import Event, EventActions
from anthropic import AsyncAnthropicVertex

from ..config import (
    GCP_PROJECT_ID,
    GCP_REGION,
    VERTEX_AI_ANTHROPIC_REGION,
    COUNCIL_MODELS,
    CHAIRMAN_MODEL,
    DEFAULT_MODEL_EFFORTS,
    is_claude_model,
)
from ..vertex import EFFORT_LEVEL_MAP, _normalize_model_name
from ..claude import CLAUDE_EFFORT_MAP, _normalize_claude_model


TIMEOUT_EXCEPTIONS = (
    TimeoutError,
    asyncio.TimeoutError,
    httpx.TimeoutException,
)


def agent_name_for_model(prefix: str, model_name: str) -> str:
    """Generate a valid Python identifier for an agent name."""
    sanitized = model_name.replace("-", "_").replace(".", "_").replace("/", "_")
    return f"{prefix}_{sanitized}"


_adk_anthropic_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, AsyncAnthropicVertex] = weakref.WeakKeyDictionary()


def get_adk_anthropic_client() -> AsyncAnthropicVertex:
    """Get or create the AsyncAnthropicVertex client for ADK models with max_retries=0 and timeout=120.0.
    
    The client is cached per running asyncio event loop using a WeakKeyDictionary.
    This guarantees that models executing across different event loops (e.g. separate
    asyncio.run calls, pytest loops, or uvicorn reloads) do not reuse connections bound
    to a closed event loop, preventing RuntimeError: Event loop is closed.
    When no event loop is running (e.g. build/import time), returns a fresh client instance.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        client = _adk_anthropic_clients.get(loop)
        if client is None:
            client = AsyncAnthropicVertex(
                project_id=GCP_PROJECT_ID,
                region=VERTEX_AI_ANTHROPIC_REGION,
                timeout=120.0,
                max_retries=0,
            )
            _adk_anthropic_clients[loop] = client
        return client

    return AsyncAnthropicVertex(
        project_id=GCP_PROJECT_ID,
        region=VERTEX_AI_ANTHROPIC_REGION,
        timeout=120.0,
        max_retries=0,
    )


class CouncilClaude(Claude):
    """Claude model subclass that dynamically resolves the AsyncAnthropicVertex client for the current running loop."""

    def __getattribute__(self, name: str) -> Any:
        if name == "client":
            return get_adk_anthropic_client()
        return super().__getattribute__(name)

    async def _get_anthropic_client(self) -> AsyncAnthropicVertex:
        return get_adk_anthropic_client()


def create_model_for_council(model_name: str, effort: Optional[str] = None) -> BaseLlm | str:
    """Create a model instance configured for council execution."""
    # Resolve effort
    eff = effort if effort is not None else DEFAULT_MODEL_EFFORTS.get(model_name)
    if eff and eff.strip().lower() in ("default", "unspecified", "none"):
        eff = None
    elif eff:
        eff = eff.strip().lower()

    if is_claude_model(model_name):
        norm_model = _normalize_claude_model(model_name)
        return CouncilClaude(
            model=norm_model,
        )
    else:
        # Gemini model
        norm_model = _normalize_model_name(model_name)
        return Gemini(
            model=norm_model,
            client_kwargs={
                "vertexai": True,
                "project": GCP_PROJECT_ID,
                "location": GCP_REGION,
            },
        )


def get_model_generation_config(model_name: str, effort: Optional[str] = None) -> Optional[types.GenerateContentConfig]:
    """Get GenerateContentConfig for Gemini or AnthropicGenerateContentConfig for Claude models."""
    eff = effort if effort is not None else DEFAULT_MODEL_EFFORTS.get(model_name)
    if is_claude_model(model_name):
        if eff and eff.strip().lower() in CLAUDE_EFFORT_MAP:
            mapped_eff = CLAUDE_EFFORT_MAP[eff.strip().lower()]
            return AnthropicGenerateContentConfig(
                effort=mapped_eff,
                thinking_config=types.ThinkingConfig(thinking_budget=-1),
                http_options=types.HttpOptions(timeout=120_000),
            )
        return AnthropicGenerateContentConfig(
            http_options=types.HttpOptions(timeout=120_000)
        )

    # Gemini model
    if eff and eff.strip().lower() in ("default", "unspecified", "none"):
        return types.GenerateContentConfig(
            http_options=types.HttpOptions(timeout=120_000)
        )
    if eff and eff.strip().lower() in EFFORT_LEVEL_MAP:
        level = EFFORT_LEVEL_MAP[eff.strip().lower()]
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=level),
            http_options=types.HttpOptions(timeout=120_000),
        )
    return types.GenerateContentConfig(
        http_options=types.HttpOptions(timeout=120_000)
    )


async def effort_fallback_callback(callback_context: Any, llm_request: LlmRequest, error: Exception) -> Optional[LlmResponse]:
    """Callback triggered when model call raises an exception.
    
    If error is a timeout, drops member immediately without retry.
    If error is not a timeout and explicit effort was set in config, strips effort config and retries once.
    Otherwise drops member gracefully so the Workflow survives.
    """
    if isinstance(error, TIMEOUT_EXCEPTIONS):
        return LlmResponse(error_code="TIMEOUT", error_message=str(error))

    # Check if explicit effort was configured in llm_request.config
    has_explicit_effort = False
    if llm_request.config is not None:
        if getattr(llm_request.config, "thinking_config", None) is not None:
            has_explicit_effort = True
        elif getattr(llm_request.config, "effort", None) is not None:
            has_explicit_effort = True

    if has_explicit_effort:
        # Retry once without explicit effort config
        try:
            req_copy = llm_request.model_copy(deep=True)
            if hasattr(req_copy.config, "thinking_config"):
                req_copy.config.thinking_config = None
            if hasattr(req_copy.config, "effort"):
                req_copy.config.effort = None

            agent = callback_context.get_invocation_context().agent
            gen = agent.canonical_model.generate_content_async(req_copy)
            resp = await anext(gen)
            return resp
        except Exception as retry_err:
            if isinstance(retry_err, TIMEOUT_EXCEPTIONS):
                return LlmResponse(error_code="TIMEOUT", error_message=str(retry_err))
            return LlmResponse(error_code="DROPPED", error_message=str(retry_err))

    return LlmResponse(error_code="DROPPED", error_message=str(error))


async def stage1_router_fn(ctx: Context) -> Event:
    """Check if any Stage 1 model succeeded; route to EARLY_STOP or __DEFAULT__."""
    session = ctx.session
    has_success = False
    for ev in session.events:
        if ev.author and ev.author.startswith("stage1_"):
            if not ev.error_code:
                if ev.content and ev.content.parts:
                    text = "".join(p.text for p in ev.content.parts if hasattr(p, "text") and p.text and not getattr(p, "thought", False))
                    if text:
                        has_success = True
                        break

    if not has_success:
        return Event(
            author="stage1_router",
            actions=EventActions(
                route="EARLY_STOP",
                state_delta={"early_stopped": True},
            ),
        )
    return Event(
        author="stage1_router",
        actions=EventActions(route="__DEFAULT__"),
    )


async def early_stop_terminal_fn(ctx: Context) -> Event:
    """Set error response when all Stage 1 models failed."""
    return Event(
        author="early_stop_terminal",
        actions=EventActions(
            state_delta={
                "stage1_results": [],
                "stage2_results": [],
                "stage3_result": {
                    "model": "error",
                    "response": "All models failed to respond. Please try again.",
                    "effort": None,
                },
                "early_stopped": True,
            }
        ),
    )


def build_council_workflow(
    council_models: Optional[List[str]] = None,
    chairman_model: Optional[str] = None,
    model_efforts: Optional[Dict[str, Optional[str]]] = None,
    model_factory: Optional[Callable[..., Any]] = None,
) -> Workflow:
    """Build the 3-stage LLM Council Workflow."""
    from .stage1 import build_stage1_nodes
    from .stage2 import build_stage2_nodes
    from .stage3 import build_stage3_nodes

    council = council_models or COUNCIL_MODELS
    chair = chairman_model or CHAIRMAN_MODEL

    s1_agents, s1_join = build_stage1_nodes(council, model_efforts, model_factory)
    s1_router = FunctionNode(name="stage1_router", func=stage1_router_fn)
    s1_stop_terminal = FunctionNode(name="early_stop_terminal", func=early_stop_terminal_fn)

    s2_prep, s2_agents, s2_join, s2_agg = build_stage2_nodes(council, model_efforts, model_factory)
    s3_prep, s3_agent, s3_store = build_stage3_nodes(chair, model_efforts, model_factory)

    edges = []
    # Start -> Stage 1 agents -> Stage 1 join
    for agent in s1_agents:
        edges.append((START, agent))
        edges.append((agent, s1_join))

    # Stage 1 join -> Stage 1 router
    edges.append((s1_join, s1_router))

    # Stage 1 router conditional branch:
    # "EARLY_STOP" -> s1_stop_terminal
    # "__DEFAULT__" -> s2_prep
    edges.append(Edge(from_node=s1_router, to_node=s1_stop_terminal, route="EARLY_STOP"))
    edges.append(Edge(from_node=s1_router, to_node=s2_prep, route="__DEFAULT__"))

    # Stage 2 prep -> Stage 2 agents -> Stage 2 join -> Stage 2 agg
    for agent in s2_agents:
        edges.append((s2_prep, agent))
        edges.append((agent, s2_join))

    edges.append((s2_join, s2_agg))

    # Stage 2 agg -> Stage 3 prep -> Stage 3 chairman -> Stage 3 store
    edges.append((s2_agg, s3_prep))
    edges.append((s3_prep, s3_agent))
    edges.append((s3_agent, s3_store))

    return Workflow(name="council_workflow", edges=edges)
