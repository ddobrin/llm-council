"""Test full 3-stage deliberation with Vertex AI."""

import asyncio
from backend.council import run_full_council, generate_conversation_title


async def main():
    query = "What is the fastest land animal and how fast can it run?"
    print(f"Testing full council with query: '{query}'\n")

    print("Generating title...")
    title = await generate_conversation_title(query)
    print(f"Generated title: {title}\n")

    print("Running 3-stage council deliberation...")
    stage1, stage2, stage3, metadata = await run_full_council(query)

    print("\n--- STAGE 1: Individual Responses ---")
    print(f"Number of responses: {len(stage1)}")
    for r in stage1:
        print(f"Model: {r['model']}")
        print(f"Response snippet: {r['response'][:100]}...\n")

    print("--- STAGE 2: Peer Rankings ---")
    print(f"Number of peer reviews: {len(stage2)}")
    print(f"Label to model mapping: {metadata.get('label_to_model')}")
    for r in stage2:
        print(f"Model: {r['model']}")
        print(f"Parsed ranking: {r.get('parsed_ranking')}")
    print(f"Aggregate rankings: {metadata.get('aggregate_rankings')}\n")

    print("--- STAGE 3: Final Synthesis ---")
    print(f"Chairman: {stage3['model']}")
    print(f"Final response snippet: {stage3['response'][:200]}...\n")

    assert len(stage1) == 3, f"Expected 3 stage 1 responses, got {len(stage1)}"
    assert len(stage2) == 3, f"Expected 3 stage 2 rankings, got {len(stage2)}"
    assert stage3.get("response") and not stage3.get("response").startswith("Error"), "Stage 3 synthesis failed"
    print("✓ Full 3-stage council deliberation succeeded!")


if __name__ == "__main__":
    asyncio.run(main())
