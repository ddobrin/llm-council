"""Characterization tests for the legacy council pipeline.

Mocks the underlying vendor SDK clients (client.aio.models.generate_content and
client.messages.create), NOT wrapper functions.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from httpx import AsyncClient, ASGITransport
import json

from backend.main import app
from backend import config
import backend.vertex as vertex_mod
import backend.claude as claude_mod
import backend.council as council_mod


class FakeGeminiPart:
    def __init__(self, text, thought=False):
        self.text = text
        self.thought = thought


class FakeGeminiResponse:
    def __init__(self, text="Fake Gemini response", thought=""):
        parts = []
        if thought:
            parts.append(FakeGeminiPart(thought, thought=True))
        parts.append(FakeGeminiPart(text, thought=False))

        candidate = MagicMock()
        candidate.content.parts = parts
        self.candidates = [candidate]
        self.text = text


class FakeClaudeBlock:
    def __init__(self, text, block_type="text"):
        self.type = block_type
        self.text = text
        self.thinking = text if block_type == "thinking" else ""


class FakeClaudeResponse:
    def __init__(self, text="Fake Claude response"):
        self.content = [FakeClaudeBlock(text, "text")]


@pytest.fixture
def mock_vendor_clients():
    """Mock underlying Gemini and Claude SDK clients."""
    mock_gemini_client = MagicMock()
    mock_gemini_generate = AsyncMock(return_value=FakeGeminiResponse())
    mock_gemini_client.aio.models.generate_content = mock_gemini_generate

    mock_claude_client = MagicMock()
    mock_claude_create = AsyncMock(return_value=FakeClaudeResponse())
    mock_claude_client.messages.create = mock_claude_create

    with patch("backend.vertex.get_vertex_client", return_value=mock_gemini_client), \
         patch("backend.claude.get_anthropic_vertex_client", return_value=mock_claude_client):
        yield mock_gemini_client, mock_claude_client


# 1. Effort fallback behavior tests for vertex.py and claude.py
@pytest.mark.asyncio
async def test_vertex_effort_fallback_retries_once_on_error():
    mock_client = MagicMock()
    # First call fails, second succeeds
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=[RuntimeError("Thinking error"), FakeGeminiResponse("Recovered response")]
    )

    with patch("backend.vertex.get_vertex_client", return_value=mock_client):
        messages = [{"role": "user", "content": "hello"}]
        result = await vertex_mod.query_model("gemini-3.8-flash", messages, effort="low")
        assert result is not None
        assert result["content"] == "Recovered response"
        assert result["effort"] is None  # Reverted to default (None)
        assert mock_client.aio.models.generate_content.call_count == 2
        # Second call config must have thinking_config as None
        second_call_config = mock_client.aio.models.generate_content.call_args_list[1].kwargs["config"]
        assert second_call_config.thinking_config is None


@pytest.mark.asyncio
async def test_vertex_effort_timeout_does_not_retry():
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=asyncio.TimeoutError()
    )

    with patch("backend.vertex.get_vertex_client", return_value=mock_client):
        messages = [{"role": "user", "content": "hello"}]
        result = await vertex_mod.query_model("gemini-3.8-flash", messages, effort="low", timeout=0.01)
        assert result is None
        assert mock_client.aio.models.generate_content.call_count == 1


@pytest.mark.asyncio
async def test_vertex_default_effort_error_does_not_retry():
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=RuntimeError("Some API error")
    )

    with patch("backend.vertex.get_vertex_client", return_value=mock_client):
        messages = [{"role": "user", "content": "hello"}]
        result = await vertex_mod.query_model("gemini-3.8-flash", messages, effort=None)
        assert result is None
        assert mock_client.aio.models.generate_content.call_count == 1


@pytest.mark.asyncio
async def test_claude_effort_fallback_retries_once_on_error():
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(
        side_effect=[RuntimeError("Effort rejected"), FakeClaudeResponse("Recovered Claude")]
    )

    with patch("backend.claude.get_anthropic_vertex_client", return_value=mock_client):
        messages = [{"role": "user", "content": "hello"}]
        result = await claude_mod.query_claude_model("claude-opus-5", messages, effort="low")
        assert result is not None
        assert result["content"] == "Recovered Claude"
        assert result["effort"] is None
        assert mock_client.messages.create.call_count == 2
        second_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
        assert "thinking" not in second_call_kwargs
        assert "output_config" not in second_call_kwargs


@pytest.mark.asyncio
async def test_claude_effort_timeout_does_not_retry():
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(
        side_effect=asyncio.TimeoutError()
    )

    with patch("backend.claude.get_anthropic_vertex_client", return_value=mock_client):
        messages = [{"role": "user", "content": "hello"}]
        result = await claude_mod.query_claude_model("claude-opus-5", messages, effort="low", timeout=0.01)
        assert result is None
        assert mock_client.messages.create.call_count == 1


@pytest.mark.asyncio
async def test_claude_default_effort_error_does_not_retry():
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(
        side_effect=RuntimeError("Some error")
    )

    with patch("backend.claude.get_anthropic_vertex_client", return_value=mock_client):
        messages = [{"role": "user", "content": "hello"}]
        result = await claude_mod.query_claude_model("claude-opus-5", messages, effort=None)
        assert result is None
        assert mock_client.messages.create.call_count == 1


# 2. Stage 2 label ordering tests
@pytest.mark.asyncio
async def test_stage2_label_order_follows_council_models_order_and_drops_failed():
    user_query = "Test question"
    stage1_results = [
        {"model": "model-1", "response": "Ans 1", "effort": "low"},
        {"model": "model-2", "response": "Ans 2", "effort": "default"},
        {"model": "model-3", "response": "Ans 3", "effort": "high"},
    ]

    mock_gemini_client = MagicMock()
    mock_gemini_client.aio.models.generate_content = AsyncMock(
        return_value=FakeGeminiResponse("Evaluation\n\nFINAL RANKING:\n1. Response A\n2. Response B\n3. Response C")
    )
    mock_claude_client = MagicMock()
    mock_claude_client.messages.create = AsyncMock(
        return_value=FakeClaudeResponse("Evaluation\n\nFINAL RANKING:\n1. Response A\n2. Response B\n3. Response C")
    )

    with patch("backend.vertex.get_vertex_client", return_value=mock_gemini_client), \
         patch("backend.claude.get_anthropic_vertex_client", return_value=mock_claude_client):
        rankings, label_to_model = await council_mod.stage2_collect_rankings(user_query, stage1_results)

        assert label_to_model == {
            "Response A": "model-1",
            "Response B": "model-2",
            "Response C": "model-3",
        }


# 3. REST contract shape (non-streaming and streaming)
@pytest.mark.live
@pytest.mark.asyncio
async def test_legacy_rest_contract_shape_non_streaming(mock_vendor_clients):
    mock_gemini_client, mock_claude_client = mock_vendor_clients
    ranking_text = "Good.\n\nFINAL RANKING:\n1. Response A\n2. Response B\n3. Response C"
    mock_gemini_client.aio.models.generate_content.return_value = FakeGeminiResponse(ranking_text)
    mock_claude_client.messages.create.return_value = FakeClaudeResponse(ranking_text)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create conversation
        conv_resp = await client.post("/api/conversations", json={})
        conv_id = conv_resp.json()["id"]

        # Send non-streaming message
        resp = await client.post(
            f"/api/conversations/{conv_id}/message",
            json={"content": "What is 2 + 2?", "model_efforts": {}}
        )
        assert resp.status_code == 200
        data = resp.json()

        # Contract shape check
        assert "stage1" in data
        assert "stage2" in data
        assert "stage3" in data
        assert "metadata" in data

        for item in data["stage1"]:
            assert "model" in item
            assert "response" in item
            assert "effort" in item

        for item in data["stage2"]:
            assert "model" in item
            assert "ranking" in item
            assert "parsed_ranking" in item
            assert "effort" in item

        assert "model" in data["stage3"]
        assert "response" in data["stage3"]
        assert "effort" in data["stage3"]

        assert "label_to_model" in data["metadata"]
        assert "aggregate_rankings" in data["metadata"]
        assert "model_efforts" in data["metadata"]


@pytest.mark.live
@pytest.mark.asyncio
async def test_legacy_rest_contract_streaming_event_order(mock_vendor_clients):
    mock_gemini_client, mock_claude_client = mock_vendor_clients
    ranking_text = "Review\n\nFINAL RANKING:\n1. Response A\n2. Response B\n3. Response C"
    mock_gemini_client.aio.models.generate_content.return_value = FakeGeminiResponse(ranking_text)
    mock_claude_client.messages.create.return_value = FakeClaudeResponse(ranking_text)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        conv_resp = await client.post("/api/conversations", json={})
        conv_id = conv_resp.json()["id"]

        events = []
        payloads = {}
        async with client.stream(
            "POST",
            f"/api/conversations/{conv_id}/message/stream",
            json={"content": "What is 2 + 2?"}
        ) as stream_resp:
            assert stream_resp.status_code == 200
            async for line in stream_resp.aiter_lines():
                if line.startswith("data: "):
                    payload = json.loads(line[6:])
                    ev_type = payload.get("type")
                    events.append(ev_type)
                    payloads[ev_type] = payload

        expected_order = [
            "stage1_start",
            "stage1_complete",
            "stage2_start",
            "stage2_complete",
            "stage3_start",
            "stage3_complete",
            "title_complete",
            "complete"
        ]
        assert events == expected_order

        # Verify metadata keys on stage2_complete
        assert "label_to_model" in payloads["stage2_complete"]["metadata"]
        assert "aggregate_rankings" in payloads["stage2_complete"]["metadata"]

        # Verify metadata keys on complete
        assert "label_to_model" in payloads["complete"]["metadata"]
        assert "aggregate_rankings" in payloads["complete"]["metadata"]
        assert "model_efforts" in payloads["complete"]["metadata"]


def test_prompt_builders_match_legacy():
    user_query = "What is gravity?"
    labels = ["A", "B"]
    stage1_results = [
        {"model": "m1", "response": "Gravity is mass curvature."},
        {"model": "m2", "response": "Gravity is a force of attraction."}
    ]
    stage2_results = [
        {"model": "m1", "ranking": "1. Response A\n2. Response B"},
        {"model": "m2", "ranking": "1. Response B\n2. Response A"}
    ]

    expected_stage2 = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

Response A:
Gravity is mass curvature.

Response B:
Gravity is a force of attraction.

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- Do not add any other text or explanations in the ranking section

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""

    expected_stage3 = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

Original Question: {user_query}

STAGE 1 - Individual Responses:
Model: m1
Response: Gravity is mass curvature.

Model: m2
Response: Gravity is a force of attraction.

STAGE 2 - Peer Rankings:
Model: m1
Ranking: 1. Response A
2. Response B

Model: m2
Ranking: 1. Response B
2. Response A

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""

    assert hasattr(council_mod, "build_stage2_prompt")
    assert hasattr(council_mod, "build_stage3_prompt")
    assert council_mod.build_stage2_prompt(user_query, labels, stage1_results) == expected_stage2
    assert council_mod.build_stage3_prompt(user_query, stage1_results, stage2_results) == expected_stage3

