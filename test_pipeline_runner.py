"""Unit tests for pipeline_runner module (ADK and Legacy orchestration)."""

import pytest
from unittest.mock import patch, MagicMock
from google.genai import types
from google.adk.models import LlmResponse

from backend import config
from backend import pipeline_runner
from test_adk_pipeline import make_mock_factory, MockModel


@pytest.mark.asyncio
async def test_run_council_legacy_delegation():
    """When COUNCIL_PIPELINE is legacy, delegates to council.run_full_council."""
    with patch("backend.pipeline_runner.get_council_pipeline", return_value="legacy"), \
         patch("backend.pipeline_runner.council.run_full_council") as mock_rfc:
        mock_rfc.return_value = (["s1"], ["s2"], {"s3": "res"}, {"meta": "val"})
        res = await pipeline_runner.run_council("What is 1+1?", model_efforts={"m": "low"})
        assert mock_rfc.call_count == 1
        mock_rfc.assert_called_once_with("What is 1+1?", model_efforts={"m": "low"})
        assert res == (["s1"], ["s2"], {"s3": "res"}, {"meta": "val"})


@pytest.mark.asyncio
async def test_run_council_adk_matches_legacy_shape():
    """When COUNCIL_PIPELINE is adk, runs ADK workflow and matches legacy tuple shape."""
    c_models = ["model_a", "model_b"]
    chair = "chairman_c"

    factory = make_mock_factory(
        responses_map={
            "model_a": ["Response from A", "Review from A\nFINAL RANKING:\n1. Response B\n2. Response A"],
            "model_b": ["Response from B", "Review from B\nFINAL RANKING:\n1. Response A\n2. Response B"],
            chair: ["Chairman synthesis"]
        }
    )

    with patch("backend.pipeline_runner.get_council_pipeline", return_value="adk"), \
         patch("backend.pipeline_runner.COUNCIL_MODELS", c_models), \
         patch("backend.pipeline_runner.CHAIRMAN_MODEL", chair):

        stage1, stage2, stage3, metadata = await pipeline_runner.run_council(
            "Test query",
            model_efforts={"model_a": "low"},
            model_factory=factory
        )

        # Stage 1: [{"model", "response", "effort"}]
        assert len(stage1) == 2
        assert stage1[0]["model"] == "model_a"
        assert stage1[0]["response"] == "Response from A"
        assert stage1[0]["effort"] == "low"
        assert stage1[1]["model"] == "model_b"
        assert stage1[1]["response"] == "Response from B"

        # Stage 2: [{"model", "ranking", "parsed_ranking", "effort"}]
        assert len(stage2) == 2
        assert stage2[0]["model"] == "model_a"
        assert stage2[0]["parsed_ranking"] == ["Response B", "Response A"]
        assert stage2[1]["model"] == "model_b"
        assert stage2[1]["parsed_ranking"] == ["Response A", "Response B"]

        # Stage 3: {"model", "response", "effort"}
        assert stage3["model"] == chair
        assert "Chairman synthesis" in stage3["response"]

        # Metadata: {"label_to_model", "aggregate_rankings", "model_efforts"}
        assert metadata["label_to_model"] == {
            "Response A": "model_a",
            "Response B": "model_b",
        }
        assert len(metadata["aggregate_rankings"]) == 2
        assert "model_efforts" in metadata


@pytest.mark.asyncio
async def test_run_council_adk_all_fail_matches_legacy():
    """When all Stage 1 models fail, returns legacy all-fail shape."""
    c_models = ["fail_a", "fail_b"]
    chair = "chairman_c"

    factory = make_mock_factory(
        errors_map={
            "fail_a": RuntimeError("Err A"),
            "fail_b": RuntimeError("Err B"),
        }
    )

    with patch("backend.pipeline_runner.get_council_pipeline", return_value="adk"), \
         patch("backend.pipeline_runner.COUNCIL_MODELS", c_models), \
         patch("backend.pipeline_runner.CHAIRMAN_MODEL", chair):

        stage1, stage2, stage3, metadata = await pipeline_runner.run_council(
            "Test query",
            model_factory=factory
        )

        assert stage1 == []
        assert stage2 == []
        assert stage3 == {
            "model": "error",
            "response": "All models failed to respond. Please try again.",
            "effort": None
        }
        assert metadata == {}


@pytest.mark.asyncio
async def test_run_council_stream_adk_event_order():
    """Verify ADK SSE streaming event types and payload contracts."""
    c_models = ["m1", "m2"]
    chair = "chair"

    factory = make_mock_factory(
        responses_map={
            "m1": ["Resp 1", "Review 1\nFINAL RANKING:\n1. Response B\n2. Response A"],
            "m2": ["Resp 2", "Review 2\nFINAL RANKING:\n1. Response A\n2. Response B"],
            chair: ["Chair synthesis"]
        }
    )

    with patch("backend.pipeline_runner.get_council_pipeline", return_value="adk"), \
         patch("backend.pipeline_runner.COUNCIL_MODELS", c_models), \
         patch("backend.pipeline_runner.CHAIRMAN_MODEL", chair):

        events = []
        payloads = {}
        async for sse_chunk in pipeline_runner.run_council_stream(
            "What is gravity?",
            model_efforts={"m1": "low"},
            model_factory=factory
        ):
            if sse_chunk.startswith("data: "):
                import json
                data = json.loads(sse_chunk[6:].strip())
                ev_type = data.get("type")
                events.append(ev_type)
                payloads[ev_type] = data

        expected = [
            "stage1_start",
            "stage1_complete",
            "stage2_start",
            "stage2_complete",
            "stage3_start",
            "stage3_complete",
            "complete"
        ]
        assert events == expected

        assert len(payloads["stage1_complete"]["data"]) == 2
        assert len(payloads["stage2_complete"]["data"]) == 2
        assert "label_to_model" in payloads["stage2_complete"]["metadata"]
        assert "aggregate_rankings" in payloads["stage2_complete"]["metadata"]
        assert payloads["stage3_complete"]["data"]["model"] == chair
        assert "model_efforts" in payloads["complete"]["metadata"]


@pytest.mark.asyncio
async def test_run_council_stream_adk_all_fail_sequence_and_payloads():
    """Spec D23: when all Stage 1 models fail, emit full SSE event sequence with empty/error payloads
    and make zero Stage 2/3 model calls.
    """
    import json
    c_models = ["fail_1", "fail_2"]
    chair = "chair_model"

    chair_called = 0
    def all_fail_factory(model_name: str, effort=None):
        if model_name == chair:
            class ChairModel(MockModel):
                async def generate_content_async(self, req, stream=False):
                    nonlocal chair_called
                    chair_called += 1
                    yield LlmResponse(content=types.Content(parts=[types.Part(text="Synthesis")]))
            return ChairModel(model=model_name)
        
        # Stage 1 models fail
        return MockModel(model=model_name, error=RuntimeError(f"Model {model_name} crashed in Stage 1"))

    with patch("backend.pipeline_runner.get_council_pipeline", return_value="adk"), \
         patch("backend.pipeline_runner.COUNCIL_MODELS", c_models), \
         patch("backend.pipeline_runner.CHAIRMAN_MODEL", chair):

        events = []
        payloads = {}
        async for chunk in pipeline_runner.run_council_stream(
            "Why did everything fail?",
            model_efforts={"fail_1": "low"},
            model_factory=all_fail_factory,
        ):
            if chunk.startswith("data: "):
                data = json.loads(chunk[len("data: "):].strip())
                ev_type = data.get("type")
                events.append(ev_type)
                payloads[ev_type] = data

        expected_events = [
            "stage1_start",
            "stage1_complete",
            "stage2_start",
            "stage2_complete",
            "stage3_start",
            "stage3_complete",
            "complete",
        ]
        assert events == expected_events
        assert chair_called == 0

        # Assert payload shapes match legacy exactly
        assert payloads["stage1_complete"]["data"] == []
        assert payloads["stage2_complete"]["data"] == []
        assert payloads["stage2_complete"]["metadata"] == {
            "label_to_model": {},
            "aggregate_rankings": [],
        }
        assert payloads["stage3_complete"]["data"] == {
            "model": "error",
            "response": "All models failed to respond. Please try again.",
            "effort": None,
        }
        assert payloads["complete"]["metadata"]["label_to_model"] == {}
        assert payloads["complete"]["metadata"]["aggregate_rankings"] == []
        assert "model_efforts" in payloads["complete"]["metadata"]


@pytest.mark.asyncio
async def test_fastapi_streaming_adk_all_fail_persists_assistant_message(tmp_path, monkeypatch):
    """FastAPI streaming endpoint persists the assistant message when all models fail under ADK."""
    from httpx import AsyncClient, ASGITransport
    from backend.main import app
    from backend import storage

    monkeypatch.setattr(storage, "DATA_DIR", str(tmp_path))
    monkeypatch.setenv("COUNCIL_PIPELINE", "adk")

    c_models = ["fail_m1", "fail_m2"]
    chair = "fail_chair"

    def fail_factory(model_name: str, effort=None):
        return MockModel(model=model_name, error=RuntimeError(f"{model_name} failed"))

    # Patch pipeline_runner to use fail_factory
    orig_stream = pipeline_runner.run_council_stream
    async def patched_stream(user_query, model_efforts=None, model_factory=None):
        async for chunk in orig_stream(
            user_query,
            model_efforts=model_efforts,
            model_factory=fail_factory,
        ):
            yield chunk

    with patch("backend.main.pipeline_runner.run_council_stream", patched_stream), \
         patch("backend.pipeline_runner.COUNCIL_MODELS", c_models), \
         patch("backend.pipeline_runner.CHAIRMAN_MODEL", chair), \
         patch("backend.main.generate_conversation_title", return_value="Failed Query Title"):

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/conversations", json={})
            assert resp.status_code == 200
            conv_id = resp.json()["id"]

            stream_resp = await client.post(
                f"/api/conversations/{conv_id}/message/stream",
                json={"content": "Hello fail"}
            )
            assert stream_resp.status_code == 200
            # Read whole SSE stream
            text = stream_resp.text
            assert "stage3_complete" in text

            # Now verify conversation storage contains the assistant message
            saved_conv = storage.get_conversation(conv_id)
            assert saved_conv is not None
            messages = saved_conv["messages"]
            assert len(messages) == 2  # user + assistant
            user_msg, asst_msg = messages[0], messages[1]
            assert user_msg["role"] == "user"
            assert asst_msg["role"] == "assistant"
            assert asst_msg["stage1"] == []
            assert asst_msg["stage2"] == []
            assert asst_msg["stage3"]["model"] == "error"
            assert asst_msg["stage3"]["response"] == "All models failed to respond. Please try again."


