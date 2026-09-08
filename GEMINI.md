# GEMINI.md - Technical Notes for LLM Council

This file contains technical details, architectural decisions, and important implementation notes for future development sessions.

## Project Overview

LLM Council is a 3-stage deliberation system where multiple LLMs collaboratively answer user questions. The key innovation is anonymized peer review in Stage 2, preventing models from playing favorites. This deployment runs across 3 Gemini models hosted on Google Cloud Vertex AI using GCP project and region configuration.

## Architecture

### Backend Structure (`backend/`)

**`config.py`**
- Configures `GCP_PROJECT_ID` (env `GCP_PROJECT_ID` / `GOOGLE_CLOUD_PROJECT`, defaults to `genai-playground24`)
- Configures `GCP_REGION` (env `GCP_REGION` / `GOOGLE_CLOUD_REGION`, defaults to `us-central1`)
- Contains `COUNCIL_MODELS`: 3 Gemini models on Vertex AI (`gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`), configurable via `COUNCIL_MODELS` env var
- Contains `CHAIRMAN_MODEL` (`gemini-2.5-pro`, configurable via `CHAIRMAN_MODEL` env var)
- Backend runs on **port 8001** (NOT 8000 - user had another app on 8000)

**`vertex.py`**
- `get_vertex_client()`: Cached singleton `genai.Client(vertexai=True, project=..., location=...)`
- `query_model()`: Single async model query using Google GenAI SDK (`client.aio.models.generate_content`)
- Disables automatic function calling warning (`disable=True`)
- `query_models_parallel()`: Parallel queries using `asyncio.gather()`
- Returns dict with 'content' and optional 'reasoning_details'
- Graceful degradation: returns None on failure, continues with successful responses

**`openrouter.py`**
- Backward-compatibility module forwarding `query_model` and `query_models_parallel` to `vertex.py`

**`council.py`** - The Core Logic
- `stage1_collect_responses()`: Parallel queries to all council models via Vertex AI
- `stage2_collect_rankings()`:
  - Anonymizes responses as "Response A, B, C, etc."
  - Creates `label_to_model` mapping for de-anonymization
  - Prompts models to evaluate and rank (with strict format requirements)
  - Returns tuple: (rankings_list, label_to_model_dict)
  - Each ranking includes both raw text and `parsed_ranking` list
- `stage3_synthesize_final()`: Chairman synthesizes from all responses + rankings
- `parse_ranking_from_text()`: Robust regex extraction of "FINAL RANKING:" section (handles clean, bolded, and numbered markdown formats)
- `calculate_aggregate_rankings()`: Computes average rank position across all peer evaluations
- `generate_conversation_title()`: Concise title generation using `gemini-2.5-flash`

**`storage.py`**
- JSON-based conversation storage in `data/conversations/`
- Each conversation: `{id, created_at, messages[]}`
- Assistant messages contain: `{role, stage1, stage2, stage3}`
- Note: metadata (label_to_model, aggregate_rankings) is NOT persisted to storage, only returned via API

**`main.py`**
- FastAPI app with CORS enabled for localhost:5173 and localhost:3000
- POST `/api/conversations/{id}/message` returns metadata in addition to stages
- POST `/api/conversations/{id}/message/stream` SSE streaming endpoint for stage progress
- Metadata includes: label_to_model mapping and aggregate_rankings

### Frontend Structure (`frontend/src/`)

**`App.jsx`**
- Main orchestration: manages conversations list and current conversation
- Handles message sending and metadata storage
- Important: metadata is stored in the UI state for display but not persisted to backend JSON

**`components/ChatInterface.jsx`**
- Multiline textarea (3 rows, resizable)
- Enter to send, Shift+Enter for new line
- User messages wrapped in markdown-content class for padding

**`components/Stage1.jsx`**
- Tab view of individual model responses
- ReactMarkdown rendering with markdown-content wrapper

**`components/Stage2.jsx`**
- **Critical Feature**: Tab view showing RAW evaluation text from each model
- De-anonymization happens CLIENT-SIDE for display (models receive anonymous labels)
- Shows "Extracted Ranking" below each evaluation so users can validate parsing
- Aggregate rankings shown with average position and vote count
- Explanatory text clarifies that boldface model names are for readability only

**`components/Stage3.jsx`**
- Final synthesized answer from chairman
- Green-tinted background (#f0fff0) to highlight conclusion

**Styling (`*.css`)**
- Light mode theme (not dark mode)
- Primary color: #4a90e2 (blue)
- Global markdown styling in `index.css` with `.markdown-content` class
- 12px padding on all markdown content to prevent cluttered appearance

## Key Design Decisions

### Stage 2 Prompt Format
The Stage 2 prompt is very specific to ensure parseable output:
```
1. Evaluate each response individually first
2. Provide "FINAL RANKING:" header
3. Numbered list format: "1. Response C", "2. Response A", etc.
4. No additional text after ranking section
```

This strict format allows reliable parsing while still getting thoughtful evaluations.

### De-anonymization Strategy
- Models receive: "Response A", "Response B", "Response C"
- Backend creates mapping: `{"Response A": "gemini-2.5-pro", ...}`
- Frontend displays model names in **bold** for readability
- Users see explanation that original evaluation used anonymous labels
- This prevents bias while maintaining transparency

### Error Handling Philosophy
- Continue with successful responses if some models fail (graceful degradation)
- Never fail the entire request due to single model failure
- Log errors but don't expose to user unless all models fail

### UI/UX Transparency
- All raw outputs are inspectable via tabs
- Parsed rankings shown below raw text for validation
- Users can verify system's interpretation of model outputs
- This builds trust and allows debugging of edge cases

## Important Implementation Details

### Relative Imports
All backend modules use relative imports (e.g., `from .config import ...`) not absolute imports. This is critical for Python's module system to work correctly when running as `python -m backend.main`.

### Port Configuration
- Backend: 8001 (changed from 8000 to avoid conflict)
- Frontend: 5173 (Vite default)
- Update both `backend/main.py` and `frontend/src/api.js` if changing

### Markdown Rendering
All ReactMarkdown components must be wrapped in `<div className="markdown-content">` for proper spacing. This class is defined globally in `index.css`.

### Model Configuration
Council models and Chairman are defined in `backend/config.py` and configurable via `.env`:
- `COUNCIL_MODELS`: `["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]`
- `CHAIRMAN_MODEL`: `"gemini-2.5-pro"`

## Common Gotchas

1. **Authentication**: Use Google Cloud Application Default Credentials (`gcloud auth application-default login`) or set `GOOGLE_APPLICATION_CREDENTIALS`.
2. **Module Import Errors**: Always run backend as `python -m backend.main` from project root, not from backend directory.
3. **CORS Issues**: Frontend must match allowed origins in `main.py` CORS middleware.
4. **Ranking Parse Failures**: Fallback regex extracts "Response X" patterns across multiple formats.
5. **Missing Metadata**: Metadata is ephemeral (not persisted), only available in API responses.

## Testing Notes

- `test_vertex.py`: Verifies Vertex AI connectivity, queries all 3 council models in parallel and the Chairman model.
- `test_full_council.py`: Runs a complete 3-stage council deliberation with title generation and validates output format.
- `test_api_endpoints.py`: Tests the FastAPI endpoints in-process, verifying health check, conversation CRUD, and SSE streaming.

## Data Flow Summary

```
User Query
    ↓
Stage 1: Parallel queries to Vertex AI → [individual responses]
    ↓
Stage 2: Anonymize → Parallel ranking queries → [evaluations + parsed rankings]
    ↓
Aggregate Rankings Calculation → [sorted by avg position]
    ↓
Stage 3: Chairman synthesis with full context
    ↓
Return: {stage1, stage2, stage3, metadata}
    ↓
Frontend: Display with tabs + validation UI
```

The entire flow is async/parallel where possible to minimize latency.
