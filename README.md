# LLM Council

![llmcouncil](header.jpg)

The idea of this repo is that instead of asking a question to a single LLM, you can group models into your **"LLM Council"**. This local web app orchestrates collaborative multi-model deliberation across both **Google Gemini** and **Anthropic Claude** models hosted on **Google Cloud Vertex AI** using **Application Default Credentials (ADC)**—**without requiring any API keys**.

The council submits questions to all models, has them review and rank each other's responses (anonymized peer review to eliminate bias), and finally has a Chairman model synthesize the final response.

## How Deliberation Works

1. **Stage 1: First Opinions (Individual Responses)**:
   The user query is sent in parallel to all council members (default: `gemini-3.7-flash`, `claude-sonnet-5`, and `claude-opus-5`). Responses are displayed in segmented tabs with color-coded badges and reasoning effort levels.
2. **Stage 2: Peer Review & Street Cred Rankings**:
   Each model evaluates the responses of the other council members. Model identities are anonymized (e.g., *Response A*, *Response B*, *Response C*) so evaluations are unbiased. Models output detailed critiques and a strict numbered ranking, which is aggregated into consensus "Street Cred" scores (Gold, Silver, Bronze badges).
3. **Stage 3: Final Synthesis**:
   The designated Chairman model (default: `gemini-3.1-pro-preview`) reviews all original responses, peer evaluations, and aggregate rankings to synthesize a comprehensive, verified answer.

---

## Supported Models

All models run through **Google Cloud Vertex AI** using your standard Google Cloud credentials and project ID. **No external API keys (such as `ANTHROPIC_API_KEY` or OpenRouter) are required.**

### Google Gemini Models (Vertex AI)
- `gemini-3.7-flash` *(Council default Seat A)*
- `gemini-3.8-flash`
- `gemini-3.6-flash`
- `gemini-3.1-pro-preview` *(Chairman default)*
- `gemini-2.5-pro`
- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`

### Anthropic Claude Models (Vertex AI)
- `claude-sonnet-5` *(Council default Seat B)*
- `claude-opus-5` *(Council default Seat C)*
- `claude-opus-4-8`
- `claude-haiku-4-5`
- `claude-fable-5`

### Reasoning & Thinking Effort
Both Gemini and Claude models support reasoning depth configuration via the UI dropdown or configuration:
- `default`: Model-determined reasoning
- `minimal`: Minimum thinking tokens
- `low`: Low reasoning budget / effort
- `medium`: Balanced reasoning depth
- `high`: Deep reasoning / extended thinking

---

## Model Configuration

You can configure models and regions either directly in **`backend/config.py`** or by overriding them with environment variables in **`.env`**.

### 1. Defaults in `backend/config.py`

In `backend/config.py`, default council members, chairman, and regions are defined in Python:

```python
# Council members - default hybrid council on Vertex AI
COUNCIL_MODELS = [
    "gemini-3.7-flash",
    "claude-sonnet-5",
    "claude-opus-5",
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "gemini-3.1-pro-preview"

# Vertex AI Anthropic Region (defaults to GCP_REGION / "global")
VERTEX_AI_ANTHROPIC_REGION = os.getenv("VERTEX_AI_ANTHROPIC_REGION") or GCP_REGION
```

### 2. Overriding via `.env`

Any setting can be dynamically overridden in `.env` without modifying code. Copy `.env.example` to get started:

```bash
cp .env.example .env
```

Here is the complete reference for `.env.example`:

```bash
# Google Cloud Vertex AI Configuration
# Set your GCP Project ID and Region
GCP_PROJECT_ID=your-gcp-project-id
GCP_REGION=global

# Authentication:
# Authenticate with Google Cloud using Application Default Credentials (ADC):
#   gcloud auth application-default login
# Or set GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# Optional overrides:
# COUNCIL_MODELS=gemini-3.7-flash,claude-sonnet-5,claude-opus-5
# CHAIRMAN_MODEL=gemini-3.1-pro-preview
# VERTEX_AI_ANTHROPIC_REGION=global
# MODEL_EFFORTS=gemini-3.7-flash:default,claude-sonnet-5:low,claude-opus-5:medium,gemini-3.1-pro-preview:high
```

### Configuration Parameters

| Variable | Description | Default |
| :--- | :--- | :--- |
| `GCP_PROJECT_ID` | Your Google Cloud project ID hosting Vertex AI | `genai-playground24` |
| `GCP_REGION` | Primary Vertex AI location for Gemini models | `global` |
| `COUNCIL_MODELS` | Comma-separated list of models for the council | `gemini-3.7-flash,claude-sonnet-5,claude-opus-5` |
| `CHAIRMAN_MODEL` | Model that synthesizes the final stage answer | `gemini-3.1-pro-preview` |
| `VERTEX_AI_ANTHROPIC_REGION` | Region for Claude models on Vertex AI | Defaults to `GCP_REGION` (`global`) |
| `MODEL_EFFORTS` | Per-model default reasoning effort (`model:level,...`) | `{}` |

#### Examples of Council Configurations:
- **Pure Gemini Council**:
  ```bash
  COUNCIL_MODELS=gemini-3.6-flash,gemini-3.7-flash,gemini-3.8-flash
  CHAIRMAN_MODEL=gemini-3.1-pro-preview
  ```
- **Pure Claude Council**:
  ```bash
  COUNCIL_MODELS=claude-sonnet-5,claude-opus-5,claude-fable-5
  CHAIRMAN_MODEL=claude-opus-5
  ```
- **Hybrid Council with Custom Effort**:
  ```bash
  COUNCIL_MODELS=gemini-3.7-flash,claude-sonnet-5,claude-opus-5
  CHAIRMAN_MODEL=gemini-3.1-pro-preview
  MODEL_EFFORTS=claude-opus-5:high,gemini-3.1-pro-preview:high
  ```

---

## Setup & Installation

### 1. Install Dependencies

The project uses [uv](https://docs.astral.sh/uv/) for Python dependency management and [npm](https://nodejs.org/) for the frontend.

**Backend:**
```bash
uv sync
```

**Frontend:**
```bash
cd frontend
npm install
cd ..
```

### 2. Authenticate with Google Cloud

Authenticate using Application Default Credentials (ADC):
```bash
gcloud auth application-default login
```
*(Or set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json`)*.

### 3. Verify Connectivity

Verify that both Gemini and Claude models are accessible via Vertex AI:

```bash
# Verify Claude models on Vertex AI via ADC
uv run python test_claude_vertex.py

# Verify Gemini models and parallel query execution
uv run python test_vertex.py

# Run a complete 3-stage council deliberation CLI test
uv run python test_full_council.py

# Run API endpoint and SSE streaming tests
uv run python test_api_endpoints.py
```

---

## Running the Application

**Option 1: Using the start script**
```bash
./start.sh
```

**Option 2: Running manually**

Terminal 1 (Backend - port 8001):
```bash
uv run python -m backend.main
```

Terminal 2 (Frontend - port 5173):
```bash
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## Tech Stack

- **Backend:** FastAPI (Python 3.10+), Google GenAI SDK (`google-genai`), Anthropic SDK with Vertex AI integration (`anthropic[vertex]`)
- **Infrastructure:** Google Cloud Vertex AI, Google Cloud Application Default Credentials (ADC)
- **Frontend:** React + Vite, react-markdown, modern responsive UI aligned with `llm-council-adk-java`
- **Storage:** JSON storage in `data/conversations/`
- **Package Management:** `uv` for Python, `npm` for JavaScript
