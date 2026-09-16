"""Vertex AI Anthropic Claude API client using Google Cloud credentials."""

import asyncio
from typing import List, Dict, Any, Optional
from anthropic import AsyncAnthropicVertex

from .config import GCP_PROJECT_ID, VERTEX_AI_ANTHROPIC_REGION

_anthropic_client: Optional[AsyncAnthropicVertex] = None


def get_anthropic_vertex_client() -> AsyncAnthropicVertex:
    """
    Get or initialize the AsyncAnthropicVertex client using Google Application Default Credentials.

    Returns:
        AsyncAnthropicVertex client configured for Vertex AI with GCP_PROJECT_ID and VERTEX_AI_ANTHROPIC_REGION.
    """
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = AsyncAnthropicVertex(
            project_id=GCP_PROJECT_ID,
            region=VERTEX_AI_ANTHROPIC_REGION,
        )
    return _anthropic_client


def _normalize_claude_model(model: str) -> str:
    """
    Normalize model identifier for Anthropic on Vertex AI.
    Strips 'anthropic/' prefix if present.
    """
    if model.startswith("anthropic/"):
        return model[len("anthropic/"):]
    return model


CLAUDE_EFFORT_MAP: Dict[str, str] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
}


async def query_claude_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0,
    effort: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Query a Claude model on Vertex AI using Application Default Credentials (ADC) with optional reasoning effort.

    Args:
        model: Claude model identifier (e.g., "claude-sonnet-5", "claude-opus-5")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds
        effort: Optional reasoning effort ("minimal", "low", "medium", "high").
                If None or "default", runs with model default thinking settings.

    Returns:
        Response dict with 'content', optional 'reasoning_details', and 'effort', or None if failed
    """
    model_name = _normalize_claude_model(model)
    client = get_anthropic_vertex_client()

    # Separate system prompt from conversational turns
    system_parts: List[str] = []
    chat_messages: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append(content)
        else:
            claude_role = "assistant" if role == "assistant" else "user"
            chat_messages.append({"role": claude_role, "content": content})

    system_instruction = "\n\n".join(system_parts) if system_parts else None

    # Determine adaptive thinking configuration
    applied_effort = None
    thinking_kwargs: Dict[str, Any] = {}
    if effort:
        norm_effort = effort.strip().lower()
        if norm_effort in CLAUDE_EFFORT_MAP:
            mapped_level = CLAUDE_EFFORT_MAP[norm_effort]
            applied_effort = norm_effort
            thinking_kwargs["thinking"] = {"type": "adaptive"}
            thinking_kwargs["output_config"] = {"effort": mapped_level}

    # Base parameters for Anthropic messages API
    request_kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": chat_messages,
        "max_tokens": 8192,
        **thinking_kwargs,
    }
    if system_instruction:
        request_kwargs["system"] = system_instruction

    try:
        response = await asyncio.wait_for(
            client.messages.create(**request_kwargs),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        print(f"Timeout querying Claude model {model} ({model_name}) on Vertex AI after {timeout}s")
        return None
    except Exception as e:
        if thinking_kwargs:
            print(f"Warning: Claude model {model} rejected effort '{applied_effort}' ({e}). Reverting to standard generation.")
            request_kwargs.pop("thinking", None)
            request_kwargs.pop("output_config", None)
            applied_effort = None
            try:
                response = await asyncio.wait_for(
                    client.messages.create(**request_kwargs),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                print(f"Timeout querying Claude model {model} ({model_name}) after {timeout}s on retry")
                return None
            except Exception as retry_e:
                print(f"Error querying Claude model {model} ({model_name}) on Vertex AI after effort fallback: {retry_e}")
                return None
        else:
            print(f"Error querying Claude model {model} ({model_name}) on Vertex AI: {e}")
            return None

    texts = []
    thoughts = []
    for block in getattr(response, "content", []):
        block_type = getattr(block, "type", None)
        if block_type == "text":
            texts.append(getattr(block, "text", ""))
        elif block_type == "thinking":
            thoughts.append(getattr(block, "thinking", ""))

    content = "".join(texts)
    reasoning_details = "".join(thoughts) if thoughts else None

    return {
        "content": content,
        "reasoning_details": reasoning_details,
        "effort": applied_effort,
    }
