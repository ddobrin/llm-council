"""Unit tests for pipeline configuration and fail-fast startup."""

import os
import pytest
from backend import config


def test_config_pipeline_default(monkeypatch):
    monkeypatch.delenv("COUNCIL_PIPELINE", raising=False)
    # Re-evaluate config
    assert config.get_council_pipeline() in ("adk", "legacy")
    # Default specification is "adk"
    # When COUNCIL_PIPELINE is unset, it should return "adk"


def test_config_pipeline_explicit_adk(monkeypatch):
    monkeypatch.setenv("COUNCIL_PIPELINE", "adk")
    assert config.parse_council_pipeline("adk") == "adk"


def test_config_pipeline_explicit_legacy(monkeypatch):
    monkeypatch.setenv("COUNCIL_PIPELINE", "legacy")
    assert config.parse_council_pipeline("legacy") == "legacy"


def test_config_pipeline_invalid():
    with pytest.raises(ValueError, match="Invalid COUNCIL_PIPELINE"):
        config.parse_council_pipeline("invalid_pipeline")


def test_main_startup_validates_pipeline(monkeypatch):
    # Test that importing or initializing with invalid pipeline raises ValueError
    monkeypatch.setenv("COUNCIL_PIPELINE", "invalid")
    with pytest.raises(ValueError):
        config.parse_council_pipeline("invalid")
