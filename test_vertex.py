"""Test script to verify Vertex AI connectivity and models."""

import asyncio
from backend.config import GCP_PROJECT_ID, GCP_REGION, COUNCIL_MODELS, CHAIRMAN_MODEL
from backend.vertex import query_model, query_models_parallel


async def main():
    print("=" * 60)
    print("LLM Council - Vertex AI Verification")
    print("=" * 60)
    print(f"GCP Project:  {GCP_PROJECT_ID}")
    print(f"GCP Region:   {GCP_REGION}")
    print(f"Council Models: {COUNCIL_MODELS}")
    print(f"Chairman:     {CHAIRMAN_MODEL}")
    print("=" * 60)

    test_messages = [{"role": "user", "content": "Respond with: Hello from <your model name> in 5 words or fewer."}]

    print("\n1. Testing parallel queries to all council models...")
    responses = await query_models_parallel(COUNCIL_MODELS, test_messages)
    for model, resp in responses.items():
        if resp and resp.get("content"):
            print(f"  ✓ {model}: {resp['content'].strip()}")
        else:
            print(f"  ✗ {model}: FAILED to respond")

    print("\n2. Testing Chairman model synthesis query...")
    chairman_resp = await query_model(CHAIRMAN_MODEL, test_messages)
    if chairman_resp and chairman_resp.get("content"):
        print(f"  ✓ Chairman ({CHAIRMAN_MODEL}): {chairman_resp['content'].strip()}")
    else:
        print(f"  ✗ Chairman ({CHAIRMAN_MODEL}): FAILED to respond")

    print("\nVerification complete.")


if __name__ == "__main__":
    asyncio.run(main())
