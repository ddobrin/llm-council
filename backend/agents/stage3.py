"""Stage 3 Chairman synthesis agent and coordination nodes for LLM Council."""

from typing import Any, Callable, Dict, List, Optional, Tuple
from google.genai import types
from google.adk.agents import LlmAgent, Context
from google.adk.workflow import FunctionNode
from google.adk.events import Event, EventActions

from ..config import COUNCIL_MODELS, CHAIRMAN_MODEL
from ..council import build_stage3_prompt
from .pipeline import (
    create_model_for_council,
    get_model_generation_config,
    effort_fallback_callback,
    agent_name_for_model,
)


def stage3_instruction(ctx: Context) -> str:
    """Dynamic instruction providing Chairman synthesis prompt from state."""
    prompt = ctx.session.state.get("chairman_prompt")
    return prompt if prompt else "You are the Chairman of an LLM Council."


async def stage3_prepare_fn(ctx: Context) -> Event:
    """Prepare Chairman synthesis prompt from Stage 1 and Stage 2 results in session."""
    session = ctx.session
    state = session.state

    s1_name_to_model = {agent_name_for_model("stage1", m): m for m in COUNCIL_MODELS}
    s2_name_to_model = {agent_name_for_model("stage2", m): m for m in COUNCIL_MODELS}

    user_query = state.get("user_query", "")
    stage1_results = state.get("stage1_results")
    if stage1_results is None:
        stage1_results = []
        for ev in session.events:
            if ev.author == "user" and not user_query:
                user_query = ev.content.parts[0].text if ev.content and ev.content.parts else ""
            elif ev.author and ev.author.startswith("stage1_"):
                if ev.error_code:
                    continue
                text = ""
                if ev.content and ev.content.parts:
                    text = "".join(p.text for p in ev.content.parts if hasattr(p, "text") and p.text and not getattr(p, "thought", False))
                if text:
                    stage1_results.append({
                        "model": s1_name_to_model.get(ev.author, ev.author[len("stage1_"):]),
                        "response": text,
                        "effort": None,
                    })

    stage2_results = state.get("stage2_results")
    if stage2_results is None:
        stage2_results = []
        for ev in session.events:
            if ev.author and ev.author.startswith("stage2_"):
                if ev.error_code:
                    continue
                text = ""
                if ev.content and ev.content.parts:
                    text = "".join(p.text for p in ev.content.parts if hasattr(p, "text") and p.text and not getattr(p, "thought", False))
                if text:
                    from ..council import parse_ranking_from_text
                    stage2_results.append({
                        "model": s2_name_to_model.get(ev.author, ev.author[len("stage2_"):]),
                        "ranking": text,
                        "parsed_ranking": parse_ranking_from_text(text),
                        "effort": None,
                    })

    if not user_query:
        for ev in session.events:
            if ev.author == "user" and ev.content and ev.content.parts:
                user_query = ev.content.parts[0].text
                break

    chairman_prompt = build_stage3_prompt(user_query, stage1_results, stage2_results)

    return Event(
        author="stage3_prepare",
        actions=EventActions(
            state_delta={
                "user_query": user_query,
                "stage1_results": stage1_results,
                "stage2_results": stage2_results,
                "chairman_prompt": chairman_prompt,
            }
        ),
    )


async def stage3_after_model_fn(ctx: Context) -> Event:
    """Store stage3_result dict in session state for downstream consumption."""
    session = ctx.session
    text = ""
    author_model = None
    for ev in reversed(session.events):
        if ev.author and ev.author.startswith("stage3_"):
            author_model = ev.author[len("stage3_"):]
            if ev.content and ev.content.parts:
                text = "".join(p.text for p in ev.content.parts if hasattr(p, "text") and p.text and not getattr(p, "thought", False))
            break

    chairman_model = session.state.get("chairman_model") or author_model or CHAIRMAN_MODEL
    stage3_result = {
        "model": chairman_model,
        "response": text if text else "Error: Unable to generate final synthesis.",
        "effort": None,
    }

    return Event(
        author="stage3_store",
        actions=EventActions(
            state_delta={
                "stage3_result": stage3_result,
            }
        ),
    )


def stage3_instruction(ctx: Context) -> str:
    """Dynamic instruction providing the Stage 3 synthesis prompt."""
    prompt = ctx.session.state.get("chairman_prompt")
    return prompt if prompt else "You are the Chairman of an LLM Council."


def build_stage3_nodes(
    chairman_model: Optional[str] = None,
    model_efforts: Optional[Dict[str, Optional[str]]] = None,
    model_factory: Optional[Callable[..., Any]] = None,
) -> Tuple[FunctionNode, LlmAgent, FunctionNode]:
    """Build the Stage 3 prep FunctionNode, Chairman LlmAgent, and store FunctionNode."""
    chair = chairman_model or CHAIRMAN_MODEL
    prep_node = FunctionNode(name="stage3_prepare", func=stage3_prepare_fn)

    eff = model_efforts.get(chair) if model_efforts else None
    if model_factory:
        model_instance = model_factory(chair, effort=eff)
    else:
        model_instance = create_model_for_council(chair, effort=eff)
    agent_config = get_model_generation_config(chair, effort=eff)

    agent = LlmAgent(
        name=agent_name_for_model("stage3", chair),
        model=model_instance,
        instruction=stage3_instruction,
        generate_content_config=agent_config,
        on_model_error_callback=effort_fallback_callback,
    )

    store_node = FunctionNode(name="stage3_store", func=stage3_after_model_fn)
    return prep_node, agent, store_node
