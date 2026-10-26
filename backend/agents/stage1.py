"""Stage 1 agents and aggregation node for LLM Council."""

from typing import Any, Callable, Dict, List, Optional, Tuple
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.workflow import JoinNode, FunctionNode
from google.adk.events import Event, EventActions

from .pipeline import (
    create_model_for_council,
    get_model_generation_config,
    effort_fallback_callback,
    agent_name_for_model,
)


def build_stage1_nodes(
    council_models: List[str],
    model_efforts: Optional[Dict[str, Optional[str]]] = None,
    model_factory: Optional[Callable[..., Any]] = None,
) -> Tuple[List[LlmAgent], JoinNode]:
    """Build the Stage 1 fan-out LlmAgents and the collector JoinNode."""
    agents = []

    for model_name in council_models:
        eff = model_efforts.get(model_name) if model_efforts else None
        if model_factory:
            model_instance = model_factory(model_name, effort=eff)
        else:
            model_instance = create_model_for_council(model_name, effort=eff)
        agent_config = get_model_generation_config(model_name, effort=eff)

        agent = LlmAgent(
            name=agent_name_for_model("stage1", model_name),
            model=model_instance,
            generate_content_config=agent_config,
            on_model_error_callback=effort_fallback_callback,
        )
        agents.append(agent)

    join_node = JoinNode(name="stage1_join")
    return agents, join_node
