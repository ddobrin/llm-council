"""Test FastAPI endpoints including streaming with Vertex AI backend."""

import asyncio
import json
from httpx import AsyncClient, ASGITransport
from backend.main import app


async def test_api():
    print("Testing API endpoints...")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Health check
        resp = await client.get("/")
        assert resp.status_code == 200
        print("✓ Health check passed:", resp.json())

        # 1b. Council configuration & effort options
        resp = await client.get("/api/council")
        assert resp.status_code == 200
        council_info = resp.json()
        assert "council_models" in council_info
        assert "chairman_model" in council_info
        assert "available_efforts" in council_info
        assert "default_efforts" in council_info
        assert "default" in council_info["available_efforts"]
        assert "low" in council_info["available_efforts"]
        print("✓ GET /api/council passed:", council_info)

        # 2. Create conversation
        resp = await client.post("/api/conversations", json={})
        assert resp.status_code == 200
        conv = resp.json()
        conv_id = conv["id"]
        print(f"✓ Created conversation: {conv_id}")

        # 3. List conversations
        resp = await client.get("/api/conversations")
        assert resp.status_code == 200
        convs = resp.json()
        assert any(c["id"] == conv_id for c in convs)
        print(f"✓ Listed {len(convs)} conversations")

        # 4. Test streaming message with per-model effort
        print("\nTesting /message/stream SSE endpoint with per-model effort...")
        events = []
        model_efforts_payload = {
            "gemini-3.6-flash": "low",
            "gemini-3.7-flash": "default",
            "gemini-3.1-pro-preview": "low",
        }
        async with client.stream(
            "POST",
            f"/api/conversations/{conv_id}/message/stream",
            json={
                "content": "What is 2 + 2?",
                "model_efforts": model_efforts_payload,
            }
        ) as stream_resp:
            assert stream_resp.status_code == 200
            async for line in stream_resp.aiter_lines():
                if line.startswith("data: "):
                    payload = json.loads(line[6:])
                    event_type = payload.get("type")
                    events.append(event_type)
                    print(f"  SSE event: {event_type}")

        expected_events = [
            "stage1_start",
            "stage1_complete",
            "stage2_start",
            "stage2_complete",
            "stage3_start",
            "stage3_complete",
            "title_complete",
            "complete"
        ]
        for ev in expected_events:
            assert ev in events, f"Missing event: {ev}"

        print("\n✓ All SSE events received successfully!")

        # 5. Verify conversation messages persisted
        resp = await client.get(f"/api/conversations/{conv_id}")
        assert resp.status_code == 200
        conv_detail = resp.json()
        assert len(conv_detail["messages"]) == 2  # user + assistant
        assert conv_detail["messages"][1]["role"] == "assistant"
        assert len(conv_detail["messages"][1]["stage1"]) == 3
        # Check effort metadata
        stage1_items = conv_detail["messages"][1]["stage1"]
        for item in stage1_items:
            print(f"  Model {item['model']} effort: {item.get('effort')}")
        stage3_item = conv_detail["messages"][1]["stage3"]
        print(f"  Chairman {stage3_item['model']} effort: {stage3_item.get('effort')}")
        assert "model_efforts" in conv_detail["messages"][1]["metadata"]
        print(f"✓ Conversation persisted with title: '{conv_detail['title']}' and model_efforts metadata")
        print("\nAPI tests PASSED!")


if __name__ == "__main__":
    asyncio.run(test_api())
