"""Vertex AI Gemini API client for making LLM requests."""

import asyncio
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types

from .config import GCP_PROJECT_ID, GCP_REGION

_client: Optional[genai.Client] = None


def get_vertex_client() -> genai.Client:
    """
    Get or initialize the Google GenAI client for Vertex AI.

    Returns:
        genai.Client configured for Vertex AI with GCP_PROJECT_ID and GCP_REGION.
    """
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project=GCP_PROJECT_ID,
            location=GCP_REGION,
        )
    return _client


def _normalize_model_name(model: str) -> str:
    """
    Normalize model identifier for Vertex AI.
    Strips 'google/' prefix if present.
    """
    if model.startswith("google/"):
        return model[len("google/"):]
    return model


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a single Gemini model via Vertex AI.

    Args:
        model: Vertex AI Gemini model identifier (e.g., "gemini-2.5-pro", "gemini-2.5-flash")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    model_name = _normalize_model_name(model)

    contents: List[types.Content] = []
    system_instruction: Optional[str] = None

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_instruction = content
        else:
            gemini_role = "model" if role == "assistant" else "user"
            contents.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=content)]
                )
            )

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
    )

    client = get_vertex_client()

    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            ),
            timeout=timeout,
        )

        thoughts = []
        texts = []
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if getattr(part, 'thought', False):
                    thoughts.append(part.text or '')
                elif getattr(part, 'text', None):
                    texts.append(part.text)

        content = "".join(texts) if texts else (response.text or "")
        reasoning_details = "".join(thoughts) if thoughts else None

        return {
            'content': content,
            'reasoning_details': reasoning_details
        }

    except asyncio.TimeoutError:
        print(f"Timeout querying model {model} ({model_name}) after {timeout}s")
        return None
    except Exception as e:
        print(f"Error querying model {model} ({model_name}) on Vertex AI: {e}")
        return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple Gemini models in parallel via Vertex AI.

    Args:
        models: List of model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    tasks = [query_model(model, messages) for model in models]
    responses = await asyncio.gather(*tasks)
    return {model: response for model, response in zip(models, responses)}
