# Kisaan AI 🌾

**Autonomous multimodal farm intelligence for Indian smallholder farmers.**

Kisaan AI is a production-grade AI platform that gives farmers instant access to crop disease diagnosis, live mandi prices, government scheme guidance, soil health analysis, and multilingual voice advisory — all through WhatsApp.

---

## What it does

| Feature | How it works |
|---|---|
| **Crop Disease Diagnosis** | Upload a photo → MobileNetV2 classifies 38 diseases across 14 crops → Claude generates a treatment plan |
| **Mandi Price Intelligence** | Live Agmarknet prices + Claude sell-window recommendation (sell now / wait / split sell) |
| **Govt Scheme Navigator** | RAG over PM-KISAN, PMFBY, KCC, eNAM, PMKSY knowledge base → eligibility check + step-by-step guidance |
| **Soil Health Analysis** | Enter NPK / pH / organic carbon readings → tabular classifier scores soil → Claude prescribes amendments with dosages |
| **Multilingual Voice** | Record a voice note in Hindi, Telugu, Tamil, Marathi, Kannada, or Bengali → Whisper ASR → NLLB-200 translation → advisory → MMS TTS reply in farmer's language |
| **WhatsApp Bot** | Send a photo, text, or voice note to a WhatsApp number → two-phase response (ack in < 2s, full answer async) |

---

## Architecture

```
User input (photo / text / voice / WhatsApp)
             │
      ┌──────▼──────────────────────────────────────────┐
      │             INTENT ROUTER                        │
      │   ZSC (bart-large-mnli) + NER (bert-base-NER)   │
      │   Auto-detects language, extracts crop & location│
      └──┬──────┬──────┬──────┬──────┬───────────────────┘
         │      │      │      │      │
      disease market scheme soil  voice/advisory
         │      │      │      │      │
  ┌──────▼──┐   │   ┌──▼──┐ ┌▼───┐  │
  │ Vision  │   │   │Scheme│ │Soil│  │
  │  Agent  │   │   │Agent │ │Agent│  │
  │ MobileNet│  │   │(RAG) │ │(HF+│  │
  │ V2 (HF) │   │   └──────┘ │Clau│  │
  └──┬──────┘   │            └────┘  │
     │          │                    │
  ┌──▼───────┐ ┌▼──────┐      ┌──────▼──────┐
  │ DETR     │ │Market │      │Advisory Agent│
  │ Object   │ │Agent  │      │ RAG + Claude │
  │ Detection│ │(Agmark│      └─────────────┘
  └──┬───────┘ │net +  │
     │         │Claude)│
  ┌──▼───────┐ └───────┘
  │Advisory  │
  │Agent     │
  │RAG+Claude│
  └──────────┘
         │
   Final Response
   (+ TTS audio for voice path)
```

**LangGraph stateful DAG** — each agent is a node; routing is a conditional edge based on ZSC intent.

---

## Tech Stack

**Backend**
- [FastAPI](https://fastapi.tiangolo.com/) + async SQLAlchemy + PostgreSQL 16
- [LangGraph](https://github.com/langchain-ai/langgraph) — stateful multi-agent orchestration
- [Anthropic Claude](https://www.anthropic.com/) — `claude-sonnet-4-6` (advisory/schemes) + `claude-haiku-4-5` (market/soil) with **prompt caching**
- [HuggingFace Transformers](https://huggingface.co/) — MobileNetV2 (disease classifier), Whisper (ASR), NLLB-200 (translation), MMS (TTS), DETR (object detection), all-MiniLM-L6-v2 (embeddings), bart-large-mnli (ZSC), bert-base-NER
- [pgvector](https://github.com/pgvector/pgvector) — cosine similarity RAG over disease knowledge base and govt scheme corpus
- [Twilio](https://www.twilio.com/) — WhatsApp Business API webhook

**Frontend**
- React 18 + TypeScript + Vite + Tailwind CSS
- Tabs: Disease · Market · Voice · Soil · Schemes

---

## Project Structure

```
kisaan/
├── backend/
│   ├── main.py                       # FastAPI app entry point
│   ├── config.py                     # pydantic-settings config
│   ├── agents/
│   │   ├── graph.py                  # LangGraph DAG (7 nodes)
│   │   ├── router.py                 # ZSC + NER intent router
│   │   ├── vision_agent.py           # HF disease classifier + confidence gate
│   │   ├── advisory_agent.py         # pgvector RAG + Claude (prompt cached)
│   │   ├── market_agent.py           # Agmarknet prices + Claude advice
│   │   ├── scheme_agent.py           # Govt scheme RAG + Claude eligibility
│   │   ├── soil_agent.py             # Tabular classifier + Claude amendments
│   │   └── voice_agent.py            # Whisper ASR → NLLB translate → TTS
│   ├── rag/
│   │   └── vector_store.py           # pgvector embed + cosine retrieve
│   ├── db/
│   │   └── models.py                 # SQLAlchemy models (sessions, training data)
│   ├── services/
│   │   ├── classifier.py             # HF pipeline wrapper (cached, confidence-gated)
│   │   ├── market_service.py         # Agmarknet + OpenWeatherMap API clients
│   │   ├── asr_service.py            # Whisper ASR wrapper
│   │   ├── translation_service.py    # NLLB-200 bidirectional translation
│   │   ├── tts_service.py            # MMS TTS synthesis
│   │   ├── ner_service.py            # bert-base-NER entity extraction
│   │   ├── zero_shot_classifier.py   # bart-large-mnli ZSC wrapper
│   │   ├── object_detection.py       # DETR pest/object detection
│   │   ├── soil_service.py           # Tabular soil health classifier
│   │   ├── cost_tracker.py           # Per-session Claude API cost tracking
│   │   └── langsmith_tracer.py       # LangSmith tracing integration
│   └── routers/
│       ├── analyze.py                # POST /api/analyze
│       ├── market.py                 # GET /api/market
│       ├── voice.py                  # POST /api/voice
│       ├── soil.py                   # POST /api/soil
│       ├── schemes.py                # POST /api/schemes
│       ├── alerts.py                 # GET /api/alerts
│       └── whatsapp.py               # POST /api/whatsapp/webhook (Twilio)
├── frontend/
│   ├── src/
│   │   ├── App.tsx                   # Root component with 5-tab layout
│   │   ├── api.ts                    # Typed API client
│   │   ├── types.ts                  # Shared TypeScript types
│   │   └── components/
│   │       ├── ImageUploader.tsx     # Drag-and-drop crop photo upload
│   │       ├── ResultCard.tsx        # Diagnosis result display
│   │       ├── MarketPanel.tsx       # Price table + advice
│   │       ├── VoiceRecorder.tsx     # MediaRecorder voice capture
│   │       ├── SoilAnalyzer.tsx      # NPK/pH input form
│   │       ├── SchemeNavigator.tsx   # Scheme query interface
│   │       └── AgentTrace.tsx        # Debug: agent pipeline trace
│   └── package.json
├── scripts/
│   ├── ingest_knowledge.py           # One-time disease KB → pgvector
│   ├── ingest_schemes.py             # One-time scheme KB → pgvector
│   ├── knowledge_base.json           # 15 disease records (ICAR/TNAU sourced)
│   ├── schemes_knowledge.json        # Govt scheme corpus
│   ├── export_onnx.py                # Export HF model to ONNX for edge deploy
│   └── training/
│       ├── train_disease_model.py    # Fine-tune MobileNetV2 on PlantVillage
│       ├── prepare_dataset.py        # Dataset prep utilities
│       └── annotate.py               # Training data annotation helper
├── data/
│   └── diseases/taxonomy.json        # Disease taxonomy (multilingual)
├── eval/
│   ├── disease_eval.py               # Eval harness (precision, latency, severity)
│   └── golden_dataset/disease_cases.json
├── start_backend.sh
├── start_frontend.sh
└── start_whatsapp.sh                 # ngrok + webhook registration
```

---

## Setup

### Prerequisites
- Python 3.11
- PostgreSQL 16 with the [pgvector extension](https://github.com/pgvector/pgvector)
- Node 18+
- (Optional) Twilio account for WhatsApp

### 1. Backend

```bash
# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
pip install greenlet  # SQLAlchemy async requirement

# Configure environment variables
cp backend/.env.example backend/.env
# Edit backend/.env — add ANTHROPIC_API_KEY at minimum

# Create the database and enable pgvector
psql -c "CREATE DATABASE kisaan_db;"
psql kisaan_db -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Ingest knowledge bases into pgvector
python scripts/ingest_knowledge.py
python scripts/ingest_schemes.py

# Start the API server (port 8000)
./start_backend.sh
```

### 2. Frontend

```bash
cd frontend
npm install

# In a new terminal
./start_frontend.sh    # http://localhost:5173
```

### 3. WhatsApp Bot (optional)

```bash
# Set TWILIO_* vars in backend/.env, then:
./start_whatsapp.sh    # starts ngrok + registers webhook
```

### 4. Run the eval harness

```bash
python eval/disease_eval.py   # requires backend on port 8000
```

---

## API Reference

### `POST /api/analyze`
Multimodal analysis endpoint — handles disease, market, soil, schemes, or advisory based on intent routing.

| Field | Type | Description |
|---|---|---|
| `image` | `File` (optional) | Crop photo for disease diagnosis |
| `query` | `string` | Natural language question (any supported language) |
| `commodity` | `string` | Crop name for market queries |
| `location` | `string` | State/district filter for prices |

**Response**
```json
{
  "session_id": "...",
  "intent": "disease",
  "crop": "Tomato",
  "condition": "Early Blight",
  "confidence": 0.87,
  "severity": "medium",
  "response": "...",
  "top5": [...],
  "retrieved_docs": [...],
  "agent_trace": [...],
  "latency_ms": 1230,
  "cost_summary": { "total_cost_usd": 0.00034 }
}
```

### `GET /api/market?commodity=Tomato&state=Maharashtra`
Returns live Agmarknet prices for the given commodity and optional state filter.

### `POST /api/voice`
Accepts audio file → returns transcript, detected language, advisory text, and base64-encoded TTS audio.

### `GET /health`
Returns `{ "status": "ok", "service": "kisaan-ai", "version": "2.0.0" }`.

---

## Key Design Decisions

**Confidence-gated diagnosis** — the vision agent refuses to generate a treatment plan if classifier confidence is below 65%, instead asking the farmer for a clearer photo. This prevents wrong pesticide advice.

**Prompt caching** — the advisory, scheme, and soil system prompts use Anthropic's `cache_control: ephemeral` to cache the ~500-token system prompt, reducing per-request cost by ~90% on cache hits.

**Two-phase WhatsApp response** — the webhook sends an acknowledgment ("Analysing...") within 2 seconds and processes the query in a background task, preventing Twilio webhook timeouts on slow inference.

**Language-first routing** — all non-English queries are translated to English before ZSC + NER so that "టమోటో ధర ఎంత" (Telugu: "what is tomato price?") correctly routes to the market agent, not advisory.

**Per-session cost tracking** — every Claude API call logs input/output/cache tokens and computes USD cost, surfaced in the API response and agent trace.

---

## Evaluation

```
python eval/disease_eval.py
```

Gates:
- Keyword precision ≥ 0.60 on golden disease dataset
- p95 latency < 8000 ms
- Severity classification accuracy ≥ 0.70

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key |
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `HF_MODEL_ID` | No | HuggingFace disease classifier model ID (default: MobileNetV2 PlantVillage) |
| `EMBEDDING_MODEL` | No | Sentence transformer for RAG embeddings |
| `OPENWEATHER_API_KEY` | No | OpenWeatherMap key (for weather context in market advice) |
| `TWILIO_ACCOUNT_SID` | No | Required only for WhatsApp bot |
| `TWILIO_AUTH_TOKEN` | No | Required only for WhatsApp bot |
| `TWILIO_WHATSAPP_NUMBER` | No | Twilio sandbox / production number |
| `LANGSMITH_API_KEY` | No | LangSmith tracing (optional observability) |

---

## License

MIT
