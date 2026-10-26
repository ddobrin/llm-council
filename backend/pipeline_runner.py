"""Pipeline runner: dispatches council execution across legacy and ADK implementations."""

import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from . import council
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL, get_council_pipeline
from .agents.pipeline import build_council_workflow, agent_name_for_model


async def run_council(
    user_query: str,
    model_efforts: Optional[Dict[str, Optional[str]]] = None,
    model_factory: Optional[Callable[..., Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    Run council deliberation using the configured pipeline (legacy or adk).
    
    Returns:
        (stage1_results, stage2_results, stage3_result, metadata)
    """
    pipeline_mode = get_council_pipeline()

    if pipeline_mode == "legacy":
        return await council.run_full_council(user_query, model_efforts=model_efforts)

    # ADK Pipeline execution
    efforts = council.resolve_model_efforts(model_efforts)
    chairman_effort = efforts.get(CHAIRMAN_MODEL)

    workflow = build_council_workflow(
        council_models=COUNCIL_MODELS,
        chairman_model=CHAIRMAN_MODEL,
        model_efforts=efforts,
        model_factory=model_factory,
    )

    session_service = InMemorySessionService()
    app_name = "llm_council"
    user_id = "user"
    session_id = str(uuid.uuid4())

    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state={"model_efforts": efforts, "user_query": user_query},
    )
    runner = Runner(agent=workflow, session_service=session_service, app_name=app_name)

    async for _ in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(parts=[types.Part(text=user_query)]),
    ):
        pass

    session = await session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
    state = session.state if session else {}

    # Check for early stop (all Stage 1 models failed)
    if state.get("early_stopped"):
        return (
            [],
            [],
            {
                "model": "error",
                "response": "All models failed to respond. Please try again.",
                "effort": None,
            },
            {},
        )

    # Extract Stage 1 results maintaining COUNCIL_MODELS order
    # Map sanitized agent names back to models
    model_to_s1_name = {m: agent_name_for_model("stage1", m) for m in COUNCIL_MODELS}
    s1_events = {}
    for ev in session.events:
        if ev.author and ev.author.startswith("stage1_") and not ev.error_code:
            text = ""
            if ev.content and ev.content.parts:
                text = "".join(p.text for p in ev.content.parts if hasattr(p, "text") and p.text and not getattr(p, "thought", False))
            if text:
                s1_events[ev.author] = text

    stage1 = []
    for model in COUNCIL_MODELS:
        s1_name = model_to_s1_name[model]
        if s1_name in s1_events:
            stage1.append({
                "model": model,
                "response": s1_events[s1_name],
                "effort": efforts.get(model),
            })

    if not stage1:
        return (
            [],
            [],
            {
                "model": "error",
                "response": "All models failed to respond. Please try again.",
                "effort": None,
            },
            {},
        )

    # Extract Stage 2 results maintaining COUNCIL_MODELS order
    model_to_s2_name = {m: agent_name_for_model("stage2", m) for m in COUNCIL_MODELS}
    s2_events = {}
    for ev in session.events:
        if ev.author and ev.author.startswith("stage2_") and not ev.error_code:
            text = ""
            if ev.content and ev.content.parts:
                text = "".join(p.text for p in ev.content.parts if hasattr(p, "text") and p.text and not getattr(p, "thought", False))
            if text:
                s2_events[ev.author] = text

    stage2 = []
    for model in COUNCIL_MODELS:
        s2_name = model_to_s2_name[model]
        if s2_name in s2_events:
            ranking_text = s2_events[s2_name]
            stage2.append({
                "model": model,
                "ranking": ranking_text,
                "parsed_ranking": council.parse_ranking_from_text(ranking_text),
                "effort": efforts.get(model),
            })

    # Stage 3 result
    s3_agent_name = agent_name_for_model("stage3", CHAIRMAN_MODEL)
    s3_text = ""
    for ev in reversed(session.events):
        if ev.author == s3_agent_name and not ev.error_code:
            if ev.content and ev.content.parts:
                s3_text = "".join(p.text for p in ev.content.parts if hasattr(p, "text") and p.text and not getattr(p, "thought", False))
            break

    stage3 = {
        "model": CHAIRMAN_MODEL,
        "response": s3_text if s3_text else "Error: Unable to generate final synthesis.",
        "effort": chairman_effort,
    }

    label_to_model = state.get("label_to_model", {})
    aggregate_rankings = state.get("aggregate_rankings") or council.calculate_aggregate_rankings(stage2, label_to_model)

    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings,
        "model_efforts": efforts,
    }

    return stage1, stage2, stage3, metadata


async def run_council_stream(
    user_query: str,
    model_efforts: Optional[Dict[str, Optional[str]]] = None,
    model_factory: Optional[Callable[..., Any]] = None,
):
    """
    Stream council deliberation using the configured pipeline (legacy or adk).
    Yields SSE chunks: data: {json}\n\n
    """
    import json
    pipeline_mode = get_council_pipeline()

    if pipeline_mode == "legacy":
        efforts = model_efforts or {}
        chairman_effort = efforts.get(CHAIRMAN_MODEL)

        yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
        stage1_results = await council.stage1_collect_responses(user_query, model_efforts=efforts)
        yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results})}\n\n"

        yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
        stage2_results, label_to_model = await council.stage2_collect_rankings(
            user_query, stage1_results, model_efforts=efforts
        )
        aggregate_rankings = council.calculate_aggregate_rankings(stage2_results, label_to_model)
        yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2_results, 'metadata': {'label_to_model': label_to_model, 'aggregate_rankings': aggregate_rankings}})}\n\n"

        yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
        stage3_result = await council.stage3_synthesize_final(
            user_query, stage1_results, stage2_results, chairman_effort=chairman_effort
        )
        yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result})}\n\n"

        metadata = {
            "label_to_model": label_to_model,
            "aggregate_rankings": aggregate_rankings,
            "model_efforts": efforts,
        }
        yield f"data: {json.dumps({'type': 'complete', 'metadata': metadata})}\n\n"
        return

    # ADK Pipeline streaming
    efforts = council.resolve_model_efforts(model_efforts)
    chairman_effort = efforts.get(CHAIRMAN_MODEL)

    workflow = build_council_workflow(
        council_models=COUNCIL_MODELS,
        chairman_model=CHAIRMAN_MODEL,
        model_efforts=efforts,
        model_factory=model_factory,
    )

    session_service = InMemorySessionService()
    app_name = "llm_council_stream"
    user_id = "user"
    session_id = str(uuid.uuid4())

    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state={"model_efforts": efforts, "user_query": user_query},
    )
    runner = Runner(agent=workflow, session_service=session_service, app_name=app_name)

    yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"

    stage1_done = False
    stage2_started = False
    stage2_done = False
    stage3_started = False
    stage3_done = False

    model_to_s1_name = {m: agent_name_for_model("stage1", m) for m in COUNCIL_MODELS}
    model_to_s2_name = {m: agent_name_for_model("stage2", m) for m in COUNCIL_MODELS}
    s3_agent_name = agent_name_for_model("stage3", CHAIRMAN_MODEL)

    s1_events = {}
    s2_events = {}
    s3_text = ""

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(parts=[types.Part(text=user_query)]),
    ):
        node_path = getattr(event.node_info, "path", "") if hasattr(event, "node_info") and event.node_info else ""

        # Collect event texts
        if event.author and event.author.startswith("stage1_") and not event.error_code:
            text = ""
            if event.content and event.content.parts:
                text = "".join(p.text for p in event.content.parts if hasattr(p, "text") and p.text and not getattr(p, "thought", False))
            if text:
                s1_events[event.author] = text

        elif event.author and event.author.startswith("stage2_") and not event.error_code:
            text = ""
            if event.content and event.content.parts:
                text = "".join(p.text for p in event.content.parts if hasattr(p, "text") and p.text and not getattr(p, "thought", False))
            if text:
                s2_events[event.author] = text

        elif event.author == s3_agent_name and not event.error_code:
            if event.content and event.content.parts:
                s3_text = "".join(p.text for p in event.content.parts if hasattr(p, "text") and p.text and not getattr(p, "thought", False))

        # Check stage progression by node_path
        if "stage1_join" in node_path and not stage1_done:
            stage1_done = True
            # Construct stage1_results in COUNCIL_MODELS order
            stage1_results = []
            for model in COUNCIL_MODELS:
                s1_name = model_to_s1_name[model]
                if s1_name in s1_events:
                    stage1_results.append({
                        "model": model,
                        "response": s1_events[s1_name],
                        "effort": efforts.get(model),
                    })
            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results})}\n\n"

        elif "stage2_prepare" in node_path and not stage2_started:
            stage2_started = True
            yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"

        elif "stage2_aggregate" in node_path and not stage2_done:
            stage2_done = True
            stage2_results = []
            for model in COUNCIL_MODELS:
                s2_name = model_to_s2_name[model]
                if s2_name in s2_events:
                    ranking_text = s2_events[s2_name]
                    stage2_results.append({
                        "model": model,
                        "ranking": ranking_text,
                        "parsed_ranking": council.parse_ranking_from_text(ranking_text),
                        "effort": efforts.get(model),
                    })

            # Fetch state to get label_to_model and aggregate_rankings
            session = await session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
            state = session.state if session else {}
            label_to_model = state.get("label_to_model", {})
            aggregate_rankings = state.get("aggregate_rankings") or council.calculate_aggregate_rankings(stage2_results, label_to_model)

            yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2_results, 'metadata': {'label_to_model': label_to_model, 'aggregate_rankings': aggregate_rankings}})}\n\n"

        elif "stage3_prepare" in node_path and not stage3_started:
            stage3_started = True
            yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"

        elif "stage3_store" in node_path and not stage3_done:
            stage3_done = True
            stage3_result = {
                "model": CHAIRMAN_MODEL,
                "response": s3_text if s3_text else "Error: Unable to generate final synthesis.",
                "effort": chairman_effort,
            }
            yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result})}\n\n"

    # End of workflow
    session = await session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
    state = session.state if session else {}

    if state.get("early_stopped"):
        if not stage1_done:
            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': []})}\n\n"
        if not stage2_started:
            yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
        if not stage2_done:
            yield f"data: {json.dumps({'type': 'stage2_complete', 'data': [], 'metadata': {'label_to_model': {}, 'aggregate_rankings': []}})}\n\n"
        if not stage3_started:
            yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
        yield f"data: {json.dumps({'type': 'stage3_complete', 'data': {'model': 'error', 'response': 'All models failed to respond. Please try again.', 'effort': None}})}\n\n"
        metadata = {
            "label_to_model": {},
            "aggregate_rankings": [],
            "model_efforts": efforts,
        }
        yield f"data: {json.dumps({'type': 'complete', 'metadata': metadata})}\n\n"
        return

    label_to_model = state.get("label_to_model", {})
    aggregate_rankings = state.get("aggregate_rankings", [])
    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings,
        "model_efforts": efforts,
    }
    yield f"data: {json.dumps({'type': 'complete', 'metadata': metadata})}\n\n"

