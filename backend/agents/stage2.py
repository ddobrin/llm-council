"""Stage 2 peer evaluation agents and coordination nodes for LLM Council."""

from typing import Any, Callable, Dict, List, Optional, Tuple
from google.genai import types
from google.adk.agents import LlmAgent, Context
from google.adk.workflow import JoinNode, FunctionNode
from google.adk.events import Event, EventActions

from ..config import COUNCIL_MODELS
from ..council import (
    build_stage2_prompt,
    parse_ranking_from_text,
    calculate_aggregate_rankings,
)
from .pipeline import (
    create_model_for_council,
    get_model_generation_config,
    effort_fallback_callback,
    agent_name_for_model,
)


def stage2_instruction(ctx: Context) -> str:
    """Dynamic instruction providing the Stage 2 evaluation and ranking prompt."""
    prompt = ctx.session.state.get("ranking_prompt")
    return prompt if prompt else "You are evaluating different responses to a question."


def make_stage2_prepare_fn(
    council_models: Optional[List[str]] = None,
    model_efforts: Optional[Dict[str, Optional[str]]] = None,
):
    models_list = council_models or COUNCIL_MODELS

    async def stage2_prepare_fn(ctx: Context) -> Event:
        """Prepare anonymous labels and ranking prompt from Stage 1 results in session."""
        session = ctx.session
        state = session.state

        active_models = state.get("council_models") or models_list
        efforts = state.get("model_efforts") or model_efforts or {}
        s1_name_to_model = {agent_name_for_model("stage1", m): m for m in active_models}

        # Find Stage 1 results from session state or session events
        stage1_results = state.get("stage1_results")
        if stage1_results is None:
            user_query = state.get("user_query")
            s1_texts = {}
            for ev in session.events:
                if ev.author == "user" and not user_query:
                    user_query = ev.content.parts[0].text if ev.content and ev.content.parts else ""
                elif ev.author and ev.author.startswith("stage1_"):
                    model_name = s1_name_to_model.get(ev.author, ev.author[len("stage1_"):])
                    if ev.error_code:
                        continue
                    text = ""
                    if ev.content and ev.content.parts:
                        text = "".join(p.text for p in ev.content.parts if hasattr(p, "text") and p.text and not getattr(p, "thought", False))
                    if text:
                        s1_texts[model_name] = text

            stage1_results = []
            for model_name in active_models:
                if model_name in s1_texts:
                    stage1_results.append({
                        "model": model_name,
                        "response": s1_texts[model_name],
                        "effort": efforts.get(model_name),
                    })

        user_query = state.get("user_query", "")
        if not user_query:
            for ev in session.events:
                if ev.author == "user" and ev.content and ev.content.parts:
                    user_query = ev.content.parts[0].text
                    break

        labels = [chr(65 + i) for i in range(len(stage1_results))]
        label_to_model = {
            f"Response {label}": res["model"]
            for label, res in zip(labels, stage1_results)
        }

        if stage1_results:
            ranking_prompt = build_stage2_prompt(user_query, labels, stage1_results)
        else:
            ranking_prompt = ""

        return Event(
            author="stage2_prepare",
            actions=EventActions(
                state_delta={
                    "user_query": user_query,
                    "stage1_results": stage1_results,
                    "labels": labels,
                    "label_to_model": label_to_model,
                    "ranking_prompt": ranking_prompt,
                }
            ),
        )

    return stage2_prepare_fn


stage2_prepare_fn = make_stage2_prepare_fn()


def make_stage2_aggregate_fn(
    council_models: Optional[List[str]] = None,
    model_efforts: Optional[Dict[str, Optional[str]]] = None,
):
    models_list = council_models or COUNCIL_MODELS

    async def stage2_aggregate_fn(ctx: Context) -> Event:
        """Aggregate Stage 2 peer rankings and compute street cred."""
        session = ctx.session
        state = session.state
        label_to_model = state.get("label_to_model", {})

        active_models = state.get("council_models") or models_list
        efforts = state.get("model_efforts") or model_efforts or {}
        s2_name_to_model = {agent_name_for_model("stage2", m): m for m in active_models}

        s2_texts = {}
        for ev in session.events:
            if ev.author and ev.author.startswith("stage2_"):
                model_name = s2_name_to_model.get(ev.author, ev.author[len("stage2_"):])
                if ev.error_code:
                    continue
                text = ""
                if ev.content and ev.content.parts:
                    text = "".join(p.text for p in ev.content.parts if hasattr(p, "text") and p.text and not getattr(p, "thought", False))
                if text:
                    s2_texts[model_name] = text

        stage2_results = []
        for model_name in active_models:
            if model_name in s2_texts:
                text = s2_texts[model_name]
                parsed = parse_ranking_from_text(text)
                stage2_results.append({
                    "model": model_name,
                    "ranking": text,
                    "parsed_ranking": parsed,
                    "effort": efforts.get(model_name),
                })

        aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

        return Event(
            author="stage2_aggregate",
            actions=EventActions(
                state_delta={
                    "stage2_results": stage2_results,
                    "aggregate_rankings": aggregate_rankings,
                }
            ),
        )

    return stage2_aggregate_fn


stage2_aggregate_fn = make_stage2_aggregate_fn()


def build_stage2_nodes(
    council_models: List[str],
    model_efforts: Optional[Dict[str, Optional[str]]] = None,
    model_factory: Optional[Callable[..., Any]] = None,
) -> Tuple[FunctionNode, List[LlmAgent], FunctionNode]:
    """Build the Stage 2 prep FunctionNode, evaluator LlmAgents, and aggregation FunctionNode."""
    prep_fn = make_stage2_prepare_fn(council_models, model_efforts)
    prep_node = FunctionNode(name="stage2_prepare", func=prep_fn)

    agents = []
    for model_name in council_models:
        eff = model_efforts.get(model_name) if model_efforts else None
        if model_factory:
            model_instance = model_factory(model_name, effort=eff)
        else:
            model_instance = create_model_for_council(model_name, effort=eff)
        agent_config = get_model_generation_config(model_name, effort=eff)

        agent = LlmAgent(
            name=agent_name_for_model("stage2", model_name),
            model=model_instance,
            instruction=stage2_instruction,
            generate_content_config=agent_config,
            on_model_error_callback=effort_fallback_callback,
        )
        agents.append(agent)

    join_node = JoinNode(name="stage2_join")
    agg_fn = make_stage2_aggregate_fn(council_models, model_efforts)
    aggregate_node = FunctionNode(name="stage2_aggregate", func=agg_fn)
    return prep_node, agents, join_node, aggregate_node
