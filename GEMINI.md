# GEMINI.md - Technical Notes for LLM Council

This file contains technical details, architectural decisions, and important implementation notes for future development sessions.

## Project Overview

LLM Council is a 3-stage deliberation system where multiple LLMs collaboratively answer user questions. The key innovation is anonymized peer review in Stage 2, preventing models from playing favorites. This deployment runs across 3 Gemini models hosted on Google Cloud Vertex AI using GCP project and region configuration.

## Architecture

### Backend Structure (`backend/`)

**`config.py`**
- Configures `GCP_PROJECT_ID` (env `GCP_PROJECT_ID` / `GOOGLE_CLOUD_PROJECT`, defaults to `genai-playground24`)
- Configures `GCP_REGION` (env `GCP_REGION` / `GOOGLE_CLOUD_REGION`, defaults to `global`)
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
- GET `/api/council` returns configured council models and chairman model
- POST `/api/conversations/{id}/message` returns metadata in addition to stages
- POST `/api/conversations/{id}/message/stream` SSE streaming endpoint for stage progress
- Metadata includes: label_to_model mapping and aggregate_rankings
- Metadata is now persisted in storage for saved session reload

### Frontend Structure (`frontend/src/`)

Aligned with `llm-council-adk-java` Look & Feel:

**`App.jsx`**
- Orchestrates conversations list, active conversation, council configuration (`api.getCouncil()`), and streaming deliberation state.
- Tracks active stage and progress metrics during deliberation.

**`components/Sidebar.jsx` & `components/DrawerRoster.jsx`**
- Matches Java `MainLayout` drawer: dark gradient theme (`linear-gradient(180deg, #1a1a2e 0%, #16213e 100%)`).
- Header with gradient text (`linear-gradient(90deg, #60a5fa, #a78bfa)`), subtitle "Collaborative AI Deliberation", and "+ New Session" button.
- Saved sessions list with active indigo highlight (`rgba(99, 102, 241, 0.25)`).
- At-a-glance **Council Roster** widget at the bottom: displays Chair (`CH` glyph, purple border, avatar dot, model name) and Council Members (Seat `A`, `B`, `C` glyphs with indigo gradient, avatar color dots, model names).

**`components/ChatInterface.jsx`**
- Matches Java `CouncilView`: Top navbar with "LLM Council" and active "ADK Workflow Council" button.
- Query card (`.query-section`) with "Ask the Council" label and textarea.
- Primary **"Consult the Council"** gradient button (`linear-gradient(135deg, #6366f1, #8b5cf6)`) with tooltip explaining the 3 deliberation stages, plus "Start New Session" tertiary button.
- Animated progress bar section during deliberation displaying live stage label and percentage.

**`components/StageHeader.jsx`**
- Reusable stage header with circular numbered gradient badge (28x28px, `linear-gradient(135deg, #6366f1, #8b5cf6)`), stage title, and animated loading spinner with status text.

**`components/Stage1.jsx` (Individual Responses)**
- `.stage-panel` card container.
- Segmented pill tabs with colored model avatar dots, seat label badges (`Model A`, `Model B`, `Model C`), and friendly model names.
- Response markdown container (`#f9fafb`, border-radius 8px, line-height 1.6, scrollable) with metadata footer.

**`components/Stage2.jsx` (Peer Review & Rankings)**
- `.stage-panel` card container.
- Model legend at the top (`.model-legend`) with pill badges mapping `Model A`, `Model B`, etc. to models.
- Evaluator tabs with model avatar dots.
- Raw review content with de-anonymized names bolded for readability.
- Styled Final Ranking numbered section (`#f0f4f8` container).
- Aggregate rankings ("Street Cred") with ranked circular badges (Gold `#fef3c7`/`#92400e`, Silver `#e5e7eb`/`#374151`, Bronze `#fed7aa`/`#9a3412`).

**`components/Stage3.jsx` (Final Synthesis)**
- `.stage-panel` card container.
- Final response in sky-blue gradient card (`linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)`) with deep blue headers (`#1e40af`).
- Chairman badge (`.chairman-badge`) at bottom: "Synthesized by <Chairman Name>".

**`utils/councilUtils.js`**
- Helpers for model name formatting, avatar colors, seat lettering (`A`, `B`, `C`), dynamic label-to-model mapping, and de-anonymization.

**Styling (`*.css`)**
- Theme palette: Background `#f3f4f6`, cards `#ffffff` (radius 12px, shadow `0 1px 3px rgba(0,0,0,0.1)`), primary accent `#6366f1` / `#8b5cf6`.
- Sleek modern scrollbars and typography matching the Java Vaadin implementation.

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
