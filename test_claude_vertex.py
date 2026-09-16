"""Test Anthropic Claude models on Vertex AI using Application Default Credentials (ADC)."""

import asyncio
from backend.config import GCP_PROJECT_ID, VERTEX_AI_ANTHROPIC_REGION, is_claude_model
from backend.vertex import query_model, query_models_parallel
from backend.claude import get_anthropic_vertex_client


async def test_claude():
    print(f"=== Vertex AI Claude Test Configuration ===")
    print(f"GCP Project ID: {GCP_PROJECT_ID}")
    print(f"Vertex AI Anthropic Region: {VERTEX_AI_ANTHROPIC_REGION}")
    print(f"is_claude_model('claude-sonnet-5'): {is_claude_model('claude-sonnet-5')}")
    print(f"is_claude_model('claude-opus-5'): {is_claude_model('claude-opus-5')}")
    print(f"is_claude_model('gemini-3.7-flash'): {is_claude_model('gemini-3.7-flash')}")

    client = get_anthropic_vertex_client()
    print(f"Client initialized: {type(client).__name__} (project={client.project_id}, region={client.region})")


    messages = [
        {"role": "user", "content": "What is the capital of France? Answer in one word."}
    ]

    for model in ["claude-sonnet-5", "claude-opus-5"]:
        print(f"\n--- Testing query_model({model}) with ADC ---")
        try:
            resp = await query_model(model, messages, timeout=60.0)
            if resp:
                print(f"Success!")
                print(f"Content: {resp.get('content', '').strip()[:100]}")
                print(f"Effort: {resp.get('effort')}")
                if resp.get('reasoning_details'):
                    print(f"Reasoning Details: {resp.get('reasoning_details')[:100]}...")
            else:
                print(f"Model {model} returned None (failed or unavailable)")
        except Exception as e:
            print(f"Exception querying {model}: {e}")

    # Test parallel querying across hybrid models (Gemini + Claude)
    print("\n--- Testing parallel hybrid query (gemini-3.7-flash + claude-sonnet-5) ---")
    hybrid_models = ["gemini-3.7-flash", "claude-sonnet-5"]
    efforts = {"gemini-3.7-flash": "low", "claude-sonnet-5": "low"}
    parallel_resps = await query_models_parallel(hybrid_models, messages, model_efforts=efforts)
    for mod, res in parallel_resps.items():
        if res:
            print(f"  {mod}: OK (length={len(res.get('content', ''))}, effort={res.get('effort')})")
        else:
            print(f"  {mod}: FAILED / NONE")


if __name__ == "__main__":
    asyncio.run(test_claude())
