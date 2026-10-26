"""ADK Pipeline Test Harness and Unit Tests for M02 Milestone.

Tests ADK workflow orchestration, 2N+1 observability, concurrent isolation,
effort parity, fallback callback, timeout drops, and error handling.
"""

import asyncio
import copy
import re
import time
from typing import AsyncGenerator, Dict, List, Optional
from unittest.mock import patch, MagicMock
import httpx
import pytest

from google.genai import types
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.events import Event
from google.adk.workflow import START, Workflow

from backend import config
import backend.council as council_mod
from backend.agents.pipeline import (
    build_council_workflow,
    effort_fallback_callback,
    create_model_for_council,
)


class MockModel(BaseLlm):
    """Mock BaseLlm mirroring upstream adk-python test patterns."""
    model: str
    responses: List[LlmResponse] = []
    error: Optional[Exception] = None
    call_count: int = 0
    requests: List[LlmRequest] = []
    delay: float = 0.0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.call_count += 1
        self.requests.append(llm_request)
        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.error:
            err = self.error
            # If error is a list or we want single error, raise it
            raise err

        if self.responses:
            idx = min(self.call_count - 1, len(self.responses) - 1)
            resp = self.responses[idx].model_copy(deep=True)
            yield resp
        else:
            yield LlmResponse(
                content=types.Content(
                    parts=[types.Part(text=f"Default response from {self.model}")]
                )
            )


def make_mock_factory(responses_map: Dict[str, List[str]] = None, errors_map: Dict[str, Exception] = None, delay_map: Dict[str, float] = None):
    responses_map = responses_map or {}
    errors_map = errors_map or {}
    delay_map = delay_map or {}
    models_created = {}

    def factory(model_name: str, effort: Optional[str] = None) -> MockModel:
        if model_name in models_created:
            return models_created[model_name]
        resps = []
        for text in responses_map.get(model_name, []):
            resps.append(
                LlmResponse(
                    content=types.Content(parts=[types.Part(text=text)])
                )
            )
        model = MockModel(
            model=model_name,
            responses=resps,
            error=errors_map.get(model_name),
            delay=delay_map.get(model_name, 0.0)
        )
        models_created[model_name] = model
        return model

    factory.models = models_created
    return factory


@pytest.mark.asyncio
async def test_2n_plus_1_observability():
    """Assert exactly 2N+1 model-call steps via event.author for N=3 and N=2."""
    for n in (3, 2):
        test_council = [f"model_{i}" for i in range(n)]
        chairman = "chairman_m"
        factory = make_mock_factory(
            responses_map={
                **{m: [f"S1 from {m}", f"S2 ranking from {m}\nFINAL RANKING:\n1. Response A\n2. Response B"] for m in test_council},
                chairman: ["S3 final synthesis"]
            }
        )

        wf = build_council_workflow(
            council_models=test_council,
            chairman_model=chairman,
            model_factory=factory
        )

        runner = Runner(agent=wf, session_service=InMemorySessionService(), app_name=f"obs_n{n}")
        await runner.session_service.create_session(app_name=f"obs_n{n}", user_id="u", session_id="s")

        events = []
        async for event in runner.run_async(
            user_id="u",
            session_id="s",
            new_message=types.Content(parts=[types.Part(text="test query")])
        ):
            events.append(event)

        # Count events whose author is one of the council agents or chairman
        expected_authors = {f"stage1_{m}" for m in test_council} | {f"stage2_{m}" for m in test_council} | {f"stage3_{chairman}"}
        model_events = [e for e in events if e.author in expected_authors and e.content]
        assert len(model_events) == 2 * n + 1, f"Expected {2*n + 1} model events for N={n}, got {len(model_events)}"


@pytest.mark.asyncio
async def test_stage1_fanout():
    """Assert execution wall time < 2 * D using time.monotonic() when D=0.15s per model for 3 models."""
    d = 0.15
    factory = make_mock_factory(
        responses_map={
            "m1": ["R1", "Rank1\nFINAL RANKING:\n1. Response A"],
            "m2": ["R2", "Rank2\nFINAL RANKING:\n1. Response A"],
            "m3": ["R3", "Rank3\nFINAL RANKING:\n1. Response A"],
            "chair": ["Final synthesis"]
        },
        delay_map={"m1": d, "m2": d, "m3": d}
    )

    from backend.agents.stage1 import build_stage1_nodes
    s1_agents, s1_join = build_stage1_nodes(["m1", "m2", "m3"], model_factory=factory)
    edges = []
    for a in s1_agents:
        edges.append((START, a))
        edges.append((a, s1_join))

    wf = Workflow(name="s1_test", edges=edges)
    runner = Runner(agent=wf, session_service=InMemorySessionService(), app_name="s1_test")
    await runner.session_service.create_session(app_name="s1_test", user_id="u", session_id="s")

    t0 = time.monotonic()
    async for _ in runner.run_async(
        user_id="u",
        session_id="s",
        new_message=types.Content(parts=[types.Part(text="test query")])
    ):
        pass
    elapsed = time.monotonic() - t0

    # If executed sequentially, 3 * 0.15 = 0.45s. Parallel is < 2 * 0.15 = 0.30s (with margin 0.35s)
    assert elapsed < 2 * d + 0.1, f"Fanout was sequential: took {elapsed:.3f}s, expected < {2*d + 0.1:.3f}s"


@pytest.mark.asyncio
async def test_stage2_parsing_and_prompts():
    """Assert Stage 2 builds anonymized prompts, collects peer evaluations, and computes rankings."""
    factory = make_mock_factory(
        responses_map={
            "m1": ["Review 1\nFINAL RANKING:\n1. Response B\n2. Response A"],
            "m2": ["Review 2\nFINAL RANKING:\n1. Response A\n2. Response B"]
        }
    )

    from backend.agents.stage2 import build_stage2_nodes
    s2_prep, s2_agents, s2_join, s2_agg = build_stage2_nodes(["m1", "m2"], model_factory=factory)
    edges = [
        (START, s2_prep),
        (s2_prep, s2_agents[0]),
        (s2_prep, s2_agents[1]),
        (s2_agents[0], s2_join),
        (s2_agents[1], s2_join),
        (s2_join, s2_agg),
    ]

    wf = Workflow(name="s2_test", edges=edges)
    runner = Runner(agent=wf, session_service=InMemorySessionService(), app_name="s2_test")
    # Pre-populate session with stage1_results and user_query
    init_state = {
        "user_query": "What is gravity?",
        "stage1_results": [
            {"model": "m1", "response": "Response 1 content"},
            {"model": "m2", "response": "Response 2 content"},
        ]
    }
    await runner.session_service.create_session(app_name="s2_test", user_id="u", session_id="s", state=init_state)

    async for _ in runner.run_async(user_id="u", session_id="s"):
        pass

    session = await runner.session_service.get_session(app_name="s2_test", user_id="u", session_id="s")
    stage2_results = session.state.get("stage2_results", [])
    assert len(stage2_results) == 2
    assert session.state.get("label_to_model") == {
        "Response A": "m1",
        "Response B": "m2"
    }
    assert "aggregate_rankings" in session.state


@pytest.mark.asyncio
async def test_stage3_chairman_synthesis():
    """Assert Stage 3 builds synthesis prompt from Stage 1 & 2 results and runs Chairman."""
    factory = make_mock_factory(
        responses_map={
            "chair": ["Chairman final synthesis of council wisdom"]
        }
    )

    from backend.agents.stage3 import build_stage3_nodes
    s3_prep, s3_agent, s3_store = build_stage3_nodes("chair", model_factory=factory)
    edges = [
        (START, s3_prep),
        (s3_prep, s3_agent),
        (s3_agent, s3_store),
    ]

    wf = Workflow(name="s3_test", edges=edges)
    runner = Runner(agent=wf, session_service=InMemorySessionService(), app_name="s3_test")
    init_state = {
        "user_query": "What is gravity?",
        "stage1_results": [
            {"model": "m1", "response": "Response 1 content"},
            {"model": "m2", "response": "Response 2 content"},
        ],
        "stage2_results": [
            {"model": "m1", "ranking": "Ranking 1", "parsed_ranking": ["Response B", "Response A"]},
            {"model": "m2", "ranking": "Ranking 2", "parsed_ranking": ["Response A", "Response B"]},
        ],
    }
    await runner.session_service.create_session(app_name="s3_test", user_id="u", session_id="s", state=init_state)

    async for _ in runner.run_async(user_id="u", session_id="s"):
        pass

    session = await runner.session_service.get_session(app_name="s3_test", user_id="u", session_id="s")
    stage3_result = session.state.get("stage3_result")
    assert stage3_result is not None
    assert stage3_result["model"] == "chair"
    assert "Chairman final synthesis" in stage3_result["response"]


@pytest.mark.asyncio
async def test_concurrent_isolation():
    """Run 2 fakes interleaved with unique session_id/user_id, ensure no state leaks."""
    factory = make_mock_factory(
        responses_map={
            "m1": ["S1 from m1", "S2 from m1\nFINAL RANKING:\n1. Response A"],
            "m2": ["S1 from m2", "S2 from m2\nFINAL RANKING:\n1. Response A"],
            "chair": ["Synthesis"]
        }
    )
    wf = build_council_workflow(
        council_models=["m1", "m2"],
        chairman_model="chair",
        model_factory=factory
    )
    session_service = InMemorySessionService()
    runner = Runner(agent=wf, session_service=session_service, app_name="iso_app")

    await session_service.create_session(app_name="iso_app", user_id="u1", session_id="s1")
    await session_service.create_session(app_name="iso_app", user_id="u2", session_id="s2")

    async def run_one(user_id, session_id, query):
        collected = []
        async for ev in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(parts=[types.Part(text=query)])
        ):
            collected.append(ev)
        return collected

    res1, res2 = await asyncio.gather(
        run_one("u1", "s1", "Query 1"),
        run_one("u2", "s2", "Query 2")
    )

    sess1 = await session_service.get_session(app_name="iso_app", user_id="u1", session_id="s1")
    sess2 = await session_service.get_session(app_name="iso_app", user_id="u2", session_id="s2")

    assert sess1.state.get("user_query") == "Query 1"
    assert sess2.state.get("user_query") == "Query 2"


@pytest.mark.asyncio
async def test_no_legacy_wrappers():
    """Assert 0 calls to backend.vertex.query_model or backend.claude.query_claude_model."""
    factory = make_mock_factory(
        responses_map={
            "m1": ["Response", "Ranking\nFINAL RANKING:\n1. Response A"],
            "chair": ["Chairman final"]
        }
    )
    wf = build_council_workflow(
        council_models=["m1"],
        chairman_model="chair",
        model_factory=factory
    )
    runner = Runner(agent=wf, session_service=InMemorySessionService(), app_name="no_leg")
    await runner.session_service.create_session(app_name="no_leg", user_id="u", session_id="s")

    with patch("backend.vertex.query_model") as mock_v_qm, \
         patch("backend.claude.query_claude_model") as mock_c_qm:
        async for _ in runner.run_async(
            user_id="u",
            session_id="s",
            new_message=types.Content(parts=[types.Part(text="test")])
        ):
            pass

        assert mock_v_qm.call_count == 0
        assert mock_c_qm.call_count == 0


@pytest.mark.asyncio
async def test_label_order_and_drops():
    """Ensure labels follow COUNCIL_MODELS order, and failed models are dropped."""
    council = ["m_first", "m_fails", "m_second"]
    factory = make_mock_factory(
        responses_map={
            "m_first": ["Resp from first", "Ranking 1\nFINAL RANKING:\n1. Response A\n2. Response B"],
            "m_second": ["Resp from second", "Ranking 2\nFINAL RANKING:\n1. Response B\n2. Response A"],
            "chair": ["Final synthesis"]
        },
        errors_map={
            "m_fails": RuntimeError("Member failed")
        }
    )

    wf = build_council_workflow(
        council_models=council,
        chairman_model="chair",
        model_factory=factory
    )
    runner = Runner(agent=wf, session_service=InMemorySessionService(), app_name="drops_app")
    await runner.session_service.create_session(app_name="drops_app", user_id="u", session_id="s")

    async for _ in runner.run_async(
        user_id="u",
        session_id="s",
        new_message=types.Content(parts=[types.Part(text="Hello")])
    ):
        pass

    session = await runner.session_service.get_session(app_name="drops_app", user_id="u", session_id="s")
    label_to_model = session.state.get("label_to_model", {})
    # Since m_fails dropped, only m_first and m_second should be labeled
    assert label_to_model == {
        "Response A": "m_first",
        "Response B": "m_second"
    }


@pytest.mark.asyncio
async def test_timeout_drops_member_no_retry():
    """Ensure timeout results in dropping member without invoking retry loop."""
    model = MockModel(
        model="m_timeout",
        error=TimeoutError("Request timed out after 120s")
    )
    factory = lambda m, effort=None: model if m == "m_timeout" else MockModel(
        model=m,
        responses=[
            LlmResponse(content=types.Content(parts=[types.Part(text="OK")]))
        ]
    )

    wf = build_council_workflow(
        council_models=["m_timeout", "m_ok"],
        chairman_model="chair",
        model_factory=factory
    )
    runner = Runner(agent=wf, session_service=InMemorySessionService(), app_name="timeout_app")
    await runner.session_service.create_session(app_name="timeout_app", user_id="u", session_id="s")

    async for _ in runner.run_async(
        user_id="u",
        session_id="s",
        new_message=types.Content(parts=[types.Part(text="Query")])
    ):
        pass

    # Assert m_timeout called once per stage without retry loops
    # (Since there are 2 stages before synthesis, it was called in Stage 1 and dropped,
    # then Stage 2 ran. If it retried on timeout, call_count would be > 2.)
    assert model.call_count == 2  # exactly 1 attempt per stage, 0 retries
    session = await runner.session_service.get_session(app_name="timeout_app", user_id="u", session_id="s")
    label_to_model = session.state.get("label_to_model", {})
    assert "m_timeout" not in label_to_model.values()
    assert "Response A" in label_to_model
    assert label_to_model["Response A"] == "m_ok"


@pytest.mark.asyncio
async def test_effort_fallback_callback():
    """Trigger effort_fallback_callback directly with/without explicit effort and with/without timeout."""
    # 1. Non-timeout error with explicit effort -> retries once without effort config
    model_called = 0

    class DummyModel:
        async def generate_content_async(self, req):
            nonlocal model_called
            model_called += 1
            yield LlmResponse(content=types.Content(parts=[types.Part(text="Fallback success")]))

    dummy_agent = MagicMock()
    dummy_agent.canonical_model = DummyModel()
    invocation_ctx = MagicMock()
    invocation_ctx.agent = dummy_agent
    callback_ctx = MagicMock()
    callback_ctx.get_invocation_context.return_value = invocation_ctx

    req_with_effort = LlmRequest(
        model="gemini-3.8-flash",
        config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="LOW"))
    )

    resp = await effort_fallback_callback(callback_ctx, req_with_effort, ValueError("Invalid effort config"))
    assert resp is not None
    assert resp.error_code is None
    assert model_called == 1

    # 2. Timeout error -> drops member immediately without retry
    timeout_err = TimeoutError("Deadline exceeded")
    resp_timeout = await effort_fallback_callback(callback_ctx, req_with_effort, timeout_err)
    assert resp_timeout is not None
    assert resp_timeout.error_code == "TIMEOUT"

    # 3. Non-timeout error with NO explicit effort -> drops member without retry
    req_no_effort = LlmRequest(model="gemini-3.8-flash", config=types.GenerateContentConfig())
    resp_no_effort = await effort_fallback_callback(callback_ctx, req_no_effort, RuntimeError("Server error"))
    assert resp_no_effort is not None
    assert resp_no_effort.error_code == "DROPPED"


@pytest.mark.asyncio
async def test_streaming_early_stop():
    """Assert 0 Stage 2 or Stage 3 model calls occur when all Stage 1 models fail."""
    factory = make_mock_factory(
        responses_map={
            "chair": ["Chairman should not be called"]
        },
        errors_map={
            "m1": RuntimeError("Fail 1"),
            "m2": RuntimeError("Fail 2"),
        }
    )

    wf = build_council_workflow(
        council_models=["m1", "m2"],
        chairman_model="chair",
        model_factory=factory
    )

    runner = Runner(agent=wf, session_service=InMemorySessionService(), app_name="stop_app")
    await runner.session_service.create_session(app_name="stop_app", user_id="u", session_id="s")

    events = []
    async for ev in runner.run_async(
        user_id="u",
        session_id="s",
        new_message=types.Content(parts=[types.Part(text="Query")])
    ):
        events.append(ev)

    # Verify no stage2 or stage3 agent was called
    s2_s3_events = [e for e in events if e.author and (e.author.startswith("stage2_") or e.author.startswith("stage3_"))]
    assert len(s2_s3_events) == 0, f"Expected 0 Stage 2/3 events, got {len(s2_s3_events)}"

    session = await runner.session_service.get_session(app_name="stop_app", user_id="u", session_id="s")
    assert session.state.get("early_stopped") is True


def test_get_model_generation_config_claude_and_gemini():
    """Verify get_model_generation_config creates appropriate configs for Claude and Gemini."""
    from backend.agents.pipeline import get_model_generation_config
    from google.adk.models.anthropic_llm import AnthropicGenerateContentConfig

    # 1. Claude with explicit effort
    claude_high = get_model_generation_config("claude-sonnet-5", effort="high")
    assert isinstance(claude_high, AnthropicGenerateContentConfig)
    assert claude_high.effort == "high"
    assert claude_high.thinking_config is not None
    assert claude_high.thinking_config.thinking_budget == -1
    assert claude_high.http_options.timeout == 120_000

    # 2. Claude with minimal effort (maps to low)
    claude_min = get_model_generation_config("claude-opus-5", effort="minimal")
    assert isinstance(claude_min, AnthropicGenerateContentConfig)
    assert claude_min.effort == "low"
    assert claude_min.thinking_config.thinking_budget == -1

    # 3. Claude with default / None effort
    claude_def = get_model_generation_config("claude-opus-5", effort="default")
    assert isinstance(claude_def, AnthropicGenerateContentConfig)
    assert claude_def.effort is None
    assert claude_def.thinking_config is None
    assert claude_def.http_options.timeout == 120_000

    # 4. Gemini with explicit effort
    gemini_low = get_model_generation_config("gemini-3.8-flash", effort="low")
    assert isinstance(gemini_low, types.GenerateContentConfig)
    assert not isinstance(gemini_low, AnthropicGenerateContentConfig)
    assert gemini_low.thinking_config.thinking_level == "LOW"
    assert gemini_low.http_options.timeout == 120_000

    # 5. Gemini with default / None effort
    gemini_def = get_model_generation_config("gemini-3.8-flash", effort="default")
    assert isinstance(gemini_def, types.GenerateContentConfig)
    assert gemini_def.thinking_config is None
    assert gemini_def.http_options.timeout == 120_000


def test_client_max_retries_configured_zero():
    """Verify ADK Claude model is configured with max_retries=0 and timeout=120.0."""
    from backend.agents.pipeline import create_model_for_council

    adk_claude_1 = create_model_for_council("claude-sonnet-5")
    assert hasattr(adk_claude_1, "client")
    assert adk_claude_1.client.max_retries == 0
    assert adk_claude_1.client.timeout == 120.0


def test_adk_claude_client_cached_per_running_loop_and_isolated():
    """Verify ADK Claude client is cached per running event loop and not shared across distinct loops."""
    import asyncio
    from backend.agents.pipeline import create_model_for_council, get_adk_anthropic_client

    model_1 = create_model_for_council("claude-sonnet-5")
    model_2 = create_model_for_council("claude-opus-5")

    async def run_in_loop_1():
        # Inside loop 1, await _get_anthropic_client() twice returns the same instance
        client_a1 = await model_1._get_anthropic_client()
        client_a2 = await model_1._get_anthropic_client()
        client_b = await model_2._get_anthropic_client()
        client_direct = get_adk_anthropic_client()

        assert client_a1 is client_a2
        assert client_a1 is client_b
        assert client_a1 is client_direct
        assert client_a1.max_retries == 0
        assert client_a1.timeout == 120.0
        return client_a1

    async def run_in_loop_2():
        # Inside loop 2, await _get_anthropic_client() returns the same instance for loop 2
        client_c1 = await model_1._get_anthropic_client()
        client_c2 = await model_2._get_anthropic_client()
        assert client_c1 is client_c2
        assert client_c1.max_retries == 0
        assert client_c1.timeout == 120.0
        return client_c1

    client_loop_1 = asyncio.run(run_in_loop_1())
    client_loop_2 = asyncio.run(run_in_loop_2())

    # Two separate event loops must NOT share the same AsyncAnthropicVertex instance (prevents event loop closed errors)
    assert client_loop_1 is not client_loop_2


@pytest.mark.asyncio
async def test_generate_content_async_reuses_loop_client(monkeypatch):
    """Verify generate_content_async uses the per-loop client without instantiating new clients per call."""
    import backend.agents.pipeline as pipeline_mod
    from backend.agents.pipeline import create_model_for_council, get_adk_anthropic_client
    from google.adk.models import LlmRequest
    from anthropic.types import Message, Usage, TextBlock

    init_count = 0
    orig_init = pipeline_mod.AsyncAnthropicVertex.__init__

    def counting_init(self, *args, **kwargs):
        nonlocal init_count
        init_count += 1
        orig_init(self, *args, **kwargs)

    monkeypatch.setattr(pipeline_mod.AsyncAnthropicVertex, "__init__", counting_init)

    model = create_model_for_council("claude-sonnet-5")

    # Mock the client's messages.create method
    fake_message = Message(
        id="msg_test",
        content=[TextBlock(text="test response", type="text")],
        model="claude-3-5-sonnet-v2@20241022",
        role="assistant",
        type="message",
        usage=Usage(input_tokens=10, output_tokens=20),
    )

    loop_client = get_adk_anthropic_client()

    async def fake_create(*args, **kwargs):
        return fake_message

    monkeypatch.setattr(loop_client.messages, "create", fake_create)

    # Initial construction happened when get_adk_anthropic_client() was called
    assert init_count == 1

    req = LlmRequest(
        model="claude-sonnet-5",
        contents=[types.Content(parts=[types.Part(text="Hello")])],
    )

    # Execute generate_content_async twice
    resps_1 = []
    async for r in model.generate_content_async(req):
        resps_1.append(r)
    assert len(resps_1) == 1
    assert "test response" in resps_1[0].content.parts[0].text

    resps_2 = []
    async for r in model.generate_content_async(req):
        resps_2.append(r)
    assert len(resps_2) == 1
    assert "test response" in resps_2[0].content.parts[0].text

    # No new AsyncAnthropicVertex instances should have been created across multiple model calls!
    assert init_count == 1



@pytest.mark.asyncio
async def test_effort_fallback_callback_claude_effort():
    """Verify effort_fallback_callback strips effort and thinking_config on AnthropicGenerateContentConfig."""
    from google.adk.models.anthropic_llm import AnthropicGenerateContentConfig

    model_called = 0
    received_req = None

    class DummyModel:
        async def generate_content_async(self, req):
            nonlocal model_called, received_req
            model_called += 1
            received_req = req
            yield LlmResponse(content=types.Content(parts=[types.Part(text="Claude fallback success")]))

    dummy_agent = MagicMock()
    dummy_agent.canonical_model = DummyModel()
    invocation_ctx = MagicMock()
    invocation_ctx.agent = dummy_agent
    callback_ctx = MagicMock()
    callback_ctx.get_invocation_context.return_value = invocation_ctx

    req_claude_effort = LlmRequest(
        model="claude-sonnet-5",
        config=AnthropicGenerateContentConfig(
            effort="high",
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            http_options=types.HttpOptions(timeout=120_000)
        )
    )

    resp = await effort_fallback_callback(callback_ctx, req_claude_effort, ValueError("Claude thinking rejected"))
    assert resp is not None
    assert resp.error_code is None
    assert model_called == 1
    assert received_req is not None
    assert received_req.config.effort is None
    assert received_req.config.thinking_config is None


@pytest.mark.asyncio
async def test_stage2_labels_follow_council_models_order_not_completion_order():
    """Spec C: labels must be assigned in configured COUNCIL_MODELS order, skipping failed members.
    Member 1 is slowest and Member 3 is fastest.
    """
    from backend.pipeline_runner import run_council, run_council_stream
    from backend.config import COUNCIL_MODELS, CHAIRMAN_MODEL
    import json

    m1, m2, m3 = COUNCIL_MODELS[0], COUNCIL_MODELS[1], COUNCIL_MODELS[2]
    chair = CHAIRMAN_MODEL

    responses_map = {
        m1: ["Output from Member 1", "S2 ranking 1\nFINAL RANKING:\n1. Response A\n2. Response B\n3. Response C"],
        m2: ["Output from Member 2", "S2 ranking 2\nFINAL RANKING:\n1. Response A\n2. Response B\n3. Response C"],
        m3: ["Output from Member 3", "S2 ranking 3\nFINAL RANKING:\n1. Response A\n2. Response B\n3. Response C"],
        chair: ["S3 final synthesis"],
    }
    # m1 is slowest (0.06s), m2 medium (0.03s), m3 fastest (0.0s)
    delay_map = {m1: 0.06, m2: 0.03, m3: 0.0}

    captured_stage2_prompts = []

    def tracking_factory(model_name: str, effort: Optional[str] = None):
        resps = []
        for text in responses_map.get(model_name, []):
            resps.append(LlmResponse(content=types.Content(parts=[types.Part(text=text)])))
        
        class TrackingModel(MockModel):
            async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False):
                # If this request contains the stage2 ranking prompt in system_instruction or contents
                sys_inst = getattr(llm_request.config, "system_instruction", None) if llm_request.config else None
                if sys_inst and "FINAL RANKING:" in str(sys_inst):
                    captured_stage2_prompts.append(str(sys_inst))
                elif llm_request.contents:
                    for c in llm_request.contents:
                        for p in c.parts:
                            if hasattr(p, "text") and p.text and "FINAL RANKING:" in p.text:
                                captured_stage2_prompts.append(p.text)
                async for r in super().generate_content_async(llm_request, stream=stream):
                    yield r

        return TrackingModel(
            model=model_name,
            responses=resps,
            delay=delay_map.get(model_name, 0.0),
        )

    # 1. Test non-streaming run_council
    stage1, stage2, stage3, metadata = await run_council(
        user_query="Test question",
        model_factory=tracking_factory,
    )

    label_to_model = metadata["label_to_model"]
    assert label_to_model["Response A"] == m1, f"Expected Response A to be {m1}, got {label_to_model.get('Response A')}"
    assert label_to_model["Response B"] == m2, f"Expected Response B to be {m2}, got {label_to_model.get('Response B')}"
    assert label_to_model["Response C"] == m3, f"Expected Response C to be {m3}, got {label_to_model.get('Response C')}"
    assert stage1[0]["model"] == m1
    assert stage1[1]["model"] == m2
    assert stage1[2]["model"] == m3

    assert len(captured_stage2_prompts) > 0
    prompt_text = captured_stage2_prompts[0]
    # Check that Response A corresponds to Member 1's text
    assert "Response A:" in prompt_text
    assert "Output from Member 1" in prompt_text.split("Response B:")[0]

    # 2. Test run_council_stream
    captured_stage2_prompts.clear()
    chunks = []
    async for chunk in run_council_stream(
        user_query="Test streaming question",
        model_factory=tracking_factory,
    ):
        chunks.append(chunk)

    events = [json.loads(c[len("data: "):].strip()) for c in chunks if c.startswith("data: ")]
    s1_event = next(e for e in events if e["type"] == "stage1_complete")
    s2_event = next(e for e in events if e["type"] == "stage2_complete")
    assert s1_event["data"][0]["model"] == m1
    assert s1_event["data"][1]["model"] == m2
    assert s1_event["data"][2]["model"] == m3
    stream_label_to_model = s2_event["metadata"]["label_to_model"]
    assert stream_label_to_model["Response A"] == m1
    assert stream_label_to_model["Response B"] == m2
    assert stream_label_to_model["Response C"] == m3


@pytest.mark.asyncio
async def test_stage2_labels_with_member2_failed():
    """When member 2 fails, Response A -> member 1, Response B -> member 3."""
    from backend.pipeline_runner import run_council
    from backend.config import COUNCIL_MODELS, CHAIRMAN_MODEL

    m1, m2, m3 = COUNCIL_MODELS[0], COUNCIL_MODELS[1], COUNCIL_MODELS[2]
    chair = CHAIRMAN_MODEL

    responses_map = {
        m1: ["Output from Member 1", "S2 ranking 1\nFINAL RANKING:\n1. Response A\n2. Response B"],
        m3: ["Output from Member 3", "S2 ranking 3\nFINAL RANKING:\n1. Response A\n2. Response B"],
        chair: ["S3 final synthesis"],
    }
    errors_map = {
        m2: RuntimeError("Member 2 crashed in Stage 1"),
    }
    delay_map = {m1: 0.05, m3: 0.0}

    captured_prompts = []

    def tracking_factory(model_name: str, effort: Optional[str] = None):
        resps = []
        for text in responses_map.get(model_name, []):
            resps.append(LlmResponse(content=types.Content(parts=[types.Part(text=text)])))
        
        class TrackingModel(MockModel):
            async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False):
                sys_inst = getattr(llm_request.config, "system_instruction", None) if llm_request.config else None
                if sys_inst and "FINAL RANKING:" in str(sys_inst):
                    captured_prompts.append(str(sys_inst))
                elif llm_request.contents:
                    for c in llm_request.contents:
                        for p in c.parts:
                            if hasattr(p, "text") and p.text and "FINAL RANKING:" in p.text:
                                captured_prompts.append(p.text)
                async for r in super().generate_content_async(llm_request, stream=stream):
                    yield r

        return TrackingModel(
            model=model_name,
            responses=resps,
            error=errors_map.get(model_name),
            delay=delay_map.get(model_name, 0.0),
        )

    stage1, stage2, stage3, metadata = await run_council(
        user_query="Test failure question",
        model_factory=tracking_factory,
    )

    label_to_model = metadata["label_to_model"]
    assert label_to_model == {
        "Response A": m1,
        "Response B": m3,
    }
    assert len(stage1) == 2
    assert stage1[0]["model"] == m1
    assert stage1[1]["model"] == m3
    assert len(captured_prompts) > 0
    assert "Output from Member 1" in captured_prompts[0].split("Response B:")[0]


@pytest.mark.asyncio
async def test_workflow_fakes_receive_generation_configs_and_fallback():
    """Assert LlmAgents receive generate_content_config even when model_factory is supplied.
    
    Verifies:
    1. Claude member receives AnthropicGenerateContentConfig with mapped effort and timeout 120_000.
    2. Gemini member receives GenerateContentConfig with ThinkingLevel and timeout 120_000.
    3. Default member receives GenerateContentConfig without thinking_config and timeout 120_000.
    4. Chairman receives GenerateContentConfig with thinking_level and timeout 120_000.
    5. Non-timeout error triggers retry without effort config.
    6. Timeout error drops immediately without retry.
    """
    captured_requests = {}

    def tracking_factory(model_name: str, effort: Optional[str] = None):
        class ConfigInspectModel(MockModel):
            async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False):
                captured_requests.setdefault(model_name, []).append(llm_request)
                if model_name == "claude-sonnet-5" and len(captured_requests[model_name]) == 1:
                    # First call fails with a non-timeout error
                    raise ValueError("Simulated model error with effort")
                if model_name == "gemini-3.7-flash":
                    # Simulate timeout error
                    raise asyncio.TimeoutError("Simulated timeout")
                yield LlmResponse(
                    content=types.Content(parts=[types.Part(text=f"Response from {model_name}\nFINAL RANKING:\n1. Response A")])
                )

        return ConfigInspectModel(model=model_name)

    council_models = ["claude-sonnet-5", "gemini-3.7-flash", "gemini-3.8-flash"]
    chairman = "gemini-3.1-pro-preview"
    model_efforts = {
        "claude-sonnet-5": "low",
        "gemini-3.7-flash": "medium",
        "gemini-3.8-flash": "default",
        "gemini-3.1-pro-preview": "high",
    }

    wf = build_council_workflow(
        council_models=council_models,
        chairman_model=chairman,
        model_efforts=model_efforts,
        model_factory=tracking_factory,
    )

    runner = Runner(agent=wf, session_service=InMemorySessionService(), app_name="config_test")
    await runner.session_service.create_session(app_name="config_test", user_id="u", session_id="s")

    events = []
    async for ev in runner.run_async(
        user_id="u",
        session_id="s",
        new_message=types.Content(parts=[types.Part(text="Test config prompt")]),
    ):
        events.append(ev)

    # 1. claude-sonnet-5 stage1 should have been called twice (initial + retry after ValueError)
    claude_reqs = [r for r in captured_requests.get("claude-sonnet-5", []) if r.config and r.config.labels.get("adk_agent_name", "").startswith("stage1")]
    assert len(claude_reqs) >= 2, f"Expected retry for Claude in stage 1, got {len(claude_reqs)} calls"
    # First call had config with effort="low"
    assert claude_reqs[0].config is not None
    assert getattr(claude_reqs[0].config, "effort", None) == "low"
    assert claude_reqs[0].config.http_options.timeout == 120_000
    # Second call (retry) should have effort stripped
    assert getattr(claude_reqs[1].config, "effort", None) is None

    # 2. gemini-3.7-flash stage 1 failed with TimeoutError -> exactly 1 call in stage 1 (no retry on timeout)
    gemini_s1_reqs = [r for r in captured_requests.get("gemini-3.7-flash", []) if r.config and r.config.labels.get("adk_agent_name", "").startswith("stage1")]
    assert len(gemini_s1_reqs) == 1, f"Expected 1 stage 1 call (no retry on timeout), got {len(gemini_s1_reqs)}"
    assert gemini_s1_reqs[0].config is not None
    assert gemini_s1_reqs[0].config.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM
    assert gemini_s1_reqs[0].config.http_options.timeout == 120_000

    # 3. gemini-3.8-flash had effort="default" -> thinking_config is None, timeout 120_000
    gemini_def_reqs = captured_requests.get("gemini-3.8-flash", [])
    assert len(gemini_def_reqs) >= 1
    assert gemini_def_reqs[0].config is not None
    assert getattr(gemini_def_reqs[0].config, "thinking_config", None) is None
    assert gemini_def_reqs[0].config.http_options.timeout == 120_000

    # 4. Chairman had effort="high" -> thinking_level HIGH, timeout 120_000
    chair_reqs = captured_requests.get("gemini-3.1-pro-preview", [])
    assert len(chair_reqs) >= 1
    assert chair_reqs[0].config is not None
    assert chair_reqs[0].config.thinking_config.thinking_level == types.ThinkingLevel.HIGH
    assert chair_reqs[0].config.http_options.timeout == 120_000

