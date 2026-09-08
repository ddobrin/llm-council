"""Backward-compatibility module forwarding to Vertex AI client."""

from .vertex import query_model, query_models_parallel, get_vertex_client

__all__ = ["query_model", "query_models_parallel", "get_vertex_client"]
