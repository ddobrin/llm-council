# LLM Council

![llmcouncil](header.jpg)

The idea of this repo is that instead of asking a question to a single LLM, you can group models into your "LLM Council". This repo is a local web app that sends your query to multiple Gemini models hosted on Google Cloud Vertex AI, asks them to review and rank each other's responses (anonymized peer review), and finally has a Chairman model synthesize the final response.

In a bit more detail, here is what happens when you submit a query:

1. **Stage 1: First opinions**. The user query is given to all 3 Gemini models individually (`gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash`), and the responses are collected in parallel. The individual responses are shown in a tab view so that the user can inspect each one.
2. **Stage 2: Review**. Each individual model is given the responses of the other models. Under the hood, the model identities are anonymized (Response A, B, C) so that models evaluate purely on accuracy and insight without bias. Each model provides an evaluation and a ranked list.
3. **Stage 3: Final response**. The designated Chairman (`gemini-3.1-pro-preview`) takes all model responses and peer rankings and compiles them into a single final synthesized answer.

## Setup

### 1. Install Dependencies

The project uses [uv](https://docs.astral.sh/uv/) for Python project management and [npm](https://nodejs.org/) for the frontend.

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

### 2. Configure Google Cloud Vertex AI

The backend connects to Google Cloud Vertex AI using your GCP Project ID and Region.

1. **Authenticate with Google Cloud:**
   ```bash
   gcloud auth application-default login
   ```
   *(Or set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable pointing to a service account key file).*

2. **Configure `.env`:**
   Create or edit `.env` in the project root:
   ```bash
   GCP_PROJECT_ID=your-gcp-project-id
   GCP_REGION=global
   ```

   *(You can copy `.env.example` as a starting point).*

### 3. Verify Vertex AI Connectivity

You can verify that your Vertex AI credentials and models are working properly:

```bash
uv run python test_vertex.py
```

To run a full 3-stage council deliberation test from the command line:

```bash
uv run python test_full_council.py
```

### 4. Configure Models (Optional)

Default models in `backend/config.py`:
- Council members: `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash`
- Chairman: `gemini-3.1-pro-preview`

You can customize them via environment variables in `.env`:
```bash
COUNCIL_MODELS=gemini-3.6-flash,gemini-3.7-flash,gemini-3.8-flash
CHAIRMAN_MODEL=gemini-3.1-pro-preview
```

## Running the Application

**Option 1: Use the start script**
```bash
./start.sh
```

**Option 2: Run manually**

Terminal 1 (Backend):
```bash
uv run python -m backend.main
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```

Then open http://localhost:5173 in your browser.

## Tech Stack

- **Backend:** FastAPI (Python 3.10+), Google GenAI SDK (`google-genai`), Vertex AI
- **Frontend:** React + Vite, react-markdown for rendering
- **Storage:** JSON files in `data/conversations/`
- **Package Management:** uv for Python, npm for JavaScript
