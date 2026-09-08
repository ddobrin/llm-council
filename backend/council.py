"""3-stage LLM Council orchestration."""

from typing import List, Dict, Any, Tuple, Optional
from .vertex import query_models_parallel, query_model
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL, DEFAULT_MODEL_EFFORTS


def resolve_model_efforts(model_efforts: Optional[Dict[str, Optional[str]]] = None) -> Dict[str, Optional[str]]:
    """
    Resolve effective effort levels by combining configured defaults with runtime overrides.
    A value of None or 'default' reverts to model set effort (represented as None).
    """
    effective: Dict[str, Optional[str]] = dict(DEFAULT_MODEL_EFFORTS)
    if model_efforts:
        for model, effort in model_efforts.items():
            if effort is None or effort.strip().lower() in ("default", "unspecified", "none"):
                effective[model] = None
            else:
                effective[model] = effort.strip().lower()
    return effective


async def stage1_collect_responses(
    user_query: str,
    model_efforts: Optional[Dict[str, Optional[str]]] = None
) -> List[Dict[str, Any]]:
    """
    Stage 1: Collect individual responses from all council models.

    Args:
        user_query: The user's question
        model_efforts: Optional mapping of model ID to effort level

    Returns:
        List of dicts with 'model', 'response', and optional 'effort' keys
    """
    messages = [{"role": "user", "content": user_query}]
    effective_efforts = resolve_model_efforts(model_efforts)

    # Query all models in parallel with configured effort
    responses = await query_models_parallel(COUNCIL_MODELS, messages, model_efforts=effective_efforts)

    # Format results
    stage1_results = []
    for model, response in responses.items():
        if response is not None:  # Only include successful responses
            stage1_results.append({
                "model": model,
                "response": response.get('content', ''),
                "effort": response.get('effort')
            })

    return stage1_results


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    model_efforts: Optional[Dict[str, Optional[str]]] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Stage 2: Each model ranks the anonymized responses.

    Args:
        user_query: The original user query
        stage1_results: Results from Stage 1

    Returns:
        Tuple of (rankings list, label_to_model mapping)
    """
    # Create anonymized labels for responses (Response A, Response B, etc.)
    labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, ...

    # Create mapping from label to model name
    label_to_model = {
        f"Response {label}": result['model']
        for label, result in zip(labels, stage1_results)
    }

    # Build the ranking prompt
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, stage1_results)
    ])

    ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

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

    messages = [{"role": "user", "content": ranking_prompt}]
    effective_efforts = resolve_model_efforts(model_efforts)

    # Get rankings from all council models in parallel
    responses = await query_models_parallel(COUNCIL_MODELS, messages, model_efforts=effective_efforts)

    # Format results
    stage2_results = []
    for model, response in responses.items():
        if response is not None:
            full_text = response.get('content', '')
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append({
                "model": model,
                "ranking": full_text,
                "parsed_ranking": parsed,
                "effort": response.get('effort')
            })

    return stage2_results, label_to_model


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    chairman_effort: Optional[str] = None
) -> Dict[str, Any]:
    """
    Stage 3: Chairman synthesizes final response.

    Args:
        user_query: The original user query
        stage1_results: Individual model responses from Stage 1
        stage2_results: Rankings from Stage 2
        chairman_effort: Optional effort level for chairman model

    Returns:
        Dict with 'model', 'response', and optional 'effort' keys
    """
    # Build comprehensive context for chairman
    stage1_text = "\n\n".join([
        f"Model: {result['model']}\nResponse: {result['response']}"
        for result in stage1_results
    ])

    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking']}"
        for result in stage2_results
    ])

    chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

Original Question: {user_query}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""

    messages = [{"role": "user", "content": chairman_prompt}]

    # Resolve effort for chairman (defaults to None / model set effort if unset)
    eff = chairman_effort if chairman_effort is not None else DEFAULT_MODEL_EFFORTS.get(CHAIRMAN_MODEL)
    if eff and eff.strip().lower() in ("default", "unspecified", "none"):
        eff = None

    # Query the chairman model
    response = await query_model(CHAIRMAN_MODEL, messages, effort=eff)

    if response is None:
        # Fallback if chairman fails
        return {
            "model": CHAIRMAN_MODEL,
            "response": "Error: Unable to generate final synthesis.",
            "effort": None
        }

    return {
        "model": CHAIRMAN_MODEL,
        "response": response.get('content', ''),
        "effort": response.get('effort')
    }


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """
    Parse the FINAL RANKING section from the model's response.

    Args:
        ranking_text: The full text response from the model

    Returns:
        List of response labels in ranked order
    """
    import re

    # Look for "FINAL RANKING:" section (case-insensitive)
    match = re.search(r'FINAL RANKING:\s*', ranking_text, re.IGNORECASE)
    if match:
        ranking_section = ranking_text[match.end():]
        # Match numbered list format, accounting for possible markdown formatting like bold/italics
        numbered_matches = re.findall(r'\d+\.\s*\*?\*?Response\s+([A-Z])\*?\*?', ranking_section)
        if numbered_matches:
            return [f"Response {letter}" for letter in numbered_matches]

        # Fallback within ranking section: Extract all "Response X" patterns in order
        matches = re.findall(r'Response\s+([A-Z])', ranking_section)
        if matches:
            return [f"Response {letter}" for letter in matches]

    # Global fallback: try to find numbered or plain "Response X" patterns anywhere
    numbered_matches = re.findall(r'\d+\.\s*\*?\*?Response\s+([A-Z])\*?\*?', ranking_text)
    if numbered_matches:
        return [f"Response {letter}" for letter in numbered_matches]

    matches = re.findall(r'Response\s+([A-Z])', ranking_text)
    return [f"Response {letter}" for letter in matches]


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models.

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from anonymous labels to model names

    Returns:
        List of dicts with model name and average rank, sorted best to worst
    """
    from collections import defaultdict

    # Track positions for each model
    model_positions = defaultdict(list)

    for ranking in stage2_results:
        ranking_text = ranking['ranking']

        # Parse the ranking from the structured format
        parsed_ranking = parse_ranking_from_text(ranking_text)

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)

    # Calculate average position for each model
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })

    # Sort by average rank (lower is better)
    aggregate.sort(key=lambda x: x['average_rank'])

    return aggregate


async def generate_conversation_title(user_query: str) -> str:
    """
    Generate a short title for a conversation based on the first user message.

    Args:
        user_query: The first user message

    Returns:
        A short title (3-5 words)
    """
    title_prompt = f"""Generate a very short title (3-5 words maximum) that summarizes the following question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

    messages = [{"role": "user", "content": title_prompt}]

    # Use gemini-3.6-flash (or first council model) for title generation (fast and cheap)
    title_model = COUNCIL_MODELS[0] if COUNCIL_MODELS else "gemini-3.6-flash"
    response = await query_model(title_model, messages, timeout=30.0)

    if response is None:
        # Fallback to a generic title
        return "New Conversation"

    title = response.get('content', 'New Conversation').strip()

    # Clean up the title - remove quotes, limit length
    title = title.strip('"\'')

    # Truncate if too long
    if len(title) > 50:
        title = title[:47] + "..."

    return title


async def run_full_council(
    user_query: str,
    model_efforts: Optional[Dict[str, Optional[str]]] = None
) -> Tuple[List, List, Dict, Dict]:
    """
    Run the complete 3-stage council process with per-model effort support.

    Args:
        user_query: The user's question
        model_efforts: Optional mapping of model names to effort levels

    Returns:
        Tuple of (stage1_results, stage2_results, stage3_result, metadata)
    """
    efforts = resolve_model_efforts(model_efforts)
    chairman_effort = efforts.get(CHAIRMAN_MODEL)

    # Stage 1: Collect individual responses
    stage1_results = await stage1_collect_responses(user_query, model_efforts=efforts)

    # If no models responded successfully, return error
    if not stage1_results:
        return [], [], {
            "model": "error",
            "response": "All models failed to respond. Please try again.",
            "effort": None
        }, {}

    # Stage 2: Collect rankings
    stage2_results, label_to_model = await stage2_collect_rankings(
        user_query,
        stage1_results,
        model_efforts=efforts
    )

    # Calculate aggregate rankings
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    # Stage 3: Synthesize final answer
    stage3_result = await stage3_synthesize_final(
        user_query,
        stage1_results,
        stage2_results,
        chairman_effort=chairman_effort
    )

    # Prepare metadata
    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings,
        "model_efforts": efforts
    }

    return stage1_results, stage2_results, stage3_result, metadata
