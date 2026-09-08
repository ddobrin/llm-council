"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# Google Cloud Vertex AI Configuration
# GCP Project ID and Region
GCP_PROJECT_ID = (
    os.getenv("GCP_PROJECT_ID")
    or os.getenv("GOOGLE_CLOUD_PROJECT")
    or "genai-playground24"
)

GCP_REGION = (
    os.getenv("GCP_REGION")
    or os.getenv("GOOGLE_CLOUD_REGION")
    or os.getenv("CLOUD_ML_REGION")
    or "global"
)

# Council members - 3 Gemini models on Vertex AI
COUNCIL_MODELS_ENV = os.getenv("COUNCIL_MODELS")
if COUNCIL_MODELS_ENV:
    COUNCIL_MODELS = [m.strip() for m in COUNCIL_MODELS_ENV.split(",") if m.strip()]
else:
    COUNCIL_MODELS = [
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
    ]

# Chairman model - synthesizes final response (Gemini 3.1 Pro Preview has highest reasoning capability)
CHAIRMAN_MODEL = os.getenv("CHAIRMAN_MODEL", "gemini-3.1-pro-preview")

# Supported reasoning effort levels for Gemini models on Vertex AI
AVAILABLE_EFFORT_LEVELS = ["default", "minimal", "low", "medium", "high"]

# Optional default per-model effort configuration (e.g. MODEL_EFFORTS="gemini-3.6-flash:low,gemini-3.1-pro-preview:high")
DEFAULT_MODEL_EFFORTS = {}
model_efforts_env = os.getenv("MODEL_EFFORTS") or os.getenv("COUNCIL_MODEL_EFFORTS")
if model_efforts_env:
    for entry in model_efforts_env.split(","):
        if ":" in entry:
            mod, eff = entry.split(":", 1)
            DEFAULT_MODEL_EFFORTS[mod.strip()] = eff.strip().lower()

# Legacy OpenRouter settings (retained for backward compatibility)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
