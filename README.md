# DocMind — RAG Chatbot

A conversational document Q&A chatbot built with **FastAPI**, **LangChain**, **Groq** (LLM), **Cohere** (embeddings), and **ChromaDB** (vector store). Upload PDFs or text files, ask questions in natural language, and get answers grounded in your documents — with source references and real-time streaming responses.

![Python 3.13](https://img.shields.io/badge/python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)
![LLM: Groq](https://img.shields.io/badge/LLM-Groq%20%2F%20Llama--3.1-orange)

---

## Features

- **Streaming responses** — answers appear token-by-token as the LLM generates them
- **Interrupt at any time** — send a new question mid-answer to immediately cancel and redirect
- **Multiple knowledge bases** — create isolated document collections and switch between them
- **Document upload** — upload PDFs and TXT files directly from the UI; chunked and indexed automatically
- **Voice input** — manual mic recording or fully hands-free Live Voice mode with VAD (Voice Activity Detection)
- **Text-to-speech** — bot answers are spoken aloud; interrupted the moment you speak or type
- **Conversation memory** — last 6 turns of context kept per session
- **Source attribution** — every answer shows which document(s) it came from

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| LLM | Groq API (`llama-3.1-8b-instant` by default) |
| Embeddings | Cohere API (`embed-english-light-v3.0`) |
| Vector store | ChromaDB (local persistent) |
| Document loading | LangChain (`PyPDFLoader`, `TextLoader`) |
| Speech-to-text | Groq Whisper (`whisper-large-v3-turbo`) |
| Text-to-speech | Browser Web Speech API |
| Frontend | Vanilla HTML/CSS/JS (single template, no build step) |
| Dependency mgmt | `uv` (or `pip`) |

---

## Project Structure

```
rag-chatbot/
├── app.py              # FastAPI application — all API routes
├── ingest.py           # CLI script: bulk-ingest PDFs from /docs
├── pyproject.toml      # Project metadata and dependencies
├── .env                # API keys and model config (never commit this)
├── docs/               # Drop PDFs here for bulk ingestion via ingest.py
├── chroma_db/          # Auto-created: persisted vector store
└── templates/
    └── index.html      # Full single-page UI
```

---

## Prerequisites

- **Python 3.13**
- **[uv](https://docs.astral.sh/uv/)** (recommended) — or plain `pip`
- A **Groq API key** — free at [console.groq.com/keys](https://console.groq.com/keys)
- A **Cohere API key** — free tier available at [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys)

---

## Setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd rag-chatbot
```

### 2. Install dependencies

**With uv (recommended):**
```bash
uv sync
```

**With pip:**
```bash
pip install -e .
```

### 3. Configure environment variables

Copy the example and fill in your keys:

```bash
cp .env.example .env   # or just edit .env directly
```

`.env` contents:

```env
# Get from https://console.groq.com/keys
GROQ_API_KEY=your_groq_api_key_here

# Get from https://dashboard.cohere.com/api-keys
COHERE_API_KEY=your_cohere_api_key_here

# Groq model to use (optional — defaults to llama-3.1-8b-instant)
LLM_MODEL=llama-3.1-8b-instant
```

> **Never commit `.env` to version control.** It is already listed in `.gitignore`.

### 4. (Optional) Bulk-ingest documents

If you have PDFs you want pre-loaded at startup, drop them into the `docs/` folder and run:

```bash
uv run python ingest.py
```

This chunks the documents and stores embeddings in `chroma_db/`. You can skip this step and upload documents through the UI instead.

### 5. Start the server

```bash
uv run uvicorn app:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

---

## Usage

### Uploading documents

1. Select or create a **Knowledge Base** from the sidebar dropdown (type a new name and press Enter to create one)
2. Click the upload zone or drag-and-drop PDF / TXT files
3. Files are chunked and indexed automatically — you'll see them appear in the document list

### Chatting

- Type a question in the input box and press **Enter** or click **Send**
- The answer streams in token-by-token as the LLM generates it
- Sources are shown below each answer

### Interrupting the bot

- **Type and send** a new question at any time — the current response stops immediately
- **Press Enter** with an empty input box to stop the bot without sending a new question
- The send button turns red (■) while the bot is responding — click it to interrupt

### Voice input (manual mic)

- Click the 🎤 microphone button to start recording
- Speak your question — silence auto-stops the recording after ~1.4 seconds
- The audio is transcribed via Groq Whisper and sent as a text question

### Live Voice mode

Click **Live Voice** in the top bar to enter fully hands-free mode:

- The app continuously listens via VAD (Voice Activity Detection)
- It automatically detects when you start and stop speaking
- The bot speaks the answer aloud via TTS
- **Speaking loudly while the bot is talking interrupts it immediately** and resets to listening
- A cooldown period after TTS ends prevents the mic from picking up speaker bleed

---

## API Reference

All endpoints are served by FastAPI. Interactive docs available at [http://localhost:8000/docs](http://localhost:8000/docs).

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the chat UI |
| `POST` | `/session/new` | Creates a new conversation session, returns `session_id` |
| `POST` | `/chat` | Streaming chat endpoint — returns newline-delimited JSON tokens |
| `GET` | `/knowledge-bases` | Lists all knowledge base names |
| `POST` | `/knowledge-bases` | Creates a new knowledge base |
| `POST` | `/upload` | Uploads and indexes a PDF or TXT file into a KB |
| `GET` | `/documents?kb_id=...` | Lists documents indexed in a knowledge base |
| `POST` | `/transcribe` | Transcribes audio via Groq Whisper, returns text |

### Streaming chat format

`POST /chat` returns a `text/plain` stream of newline-delimited JSON objects:

```json
{"token": "The "}
{"token": "answer "}
{"token": "is..."}
{"done": true, "sources": ["document.pdf"]}
```

The frontend reads this stream incrementally and renders tokens as they arrive.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Required. Your Groq API key |
| `COHERE_API_KEY` | — | Required. Your Cohere API key |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Any model available on Groq (e.g. `llama-3.3-70b-versatile`) |

**Tunable constants in `app.py`:**

| Constant | Default | Description |
|---|---|---|
| `CHROMA_DIR` | `chroma_db` | Path to the ChromaDB persistence directory |
| `DEFAULT_KB` | `default` | Name of the default knowledge base |
| `chunk_size` | `800` | Characters per document chunk |
| `chunk_overlap` | `100` | Overlap between adjacent chunks |
| `k` (retriever) | `4` | Number of chunks retrieved per query |
| History window | `12` messages | Conversation turns kept in memory (6 Q&A pairs) |

---

## How It Works

```
User question
      │
      ▼
ChromaDB similarity search  ──►  Top-4 relevant chunks
      │
      ▼
LangChain prompt (system + history + context + question)
      │
      ▼
Groq LLM (streaming)  ──►  Tokens streamed to browser via SSE-like plain text
      │
      ▼
Browser renders tokens live  ──►  TTS speaks the answer
```

1. **Embedding**: When a document is uploaded, it is split into 800-character chunks with 100-character overlap. Each chunk is embedded by Cohere and stored in ChromaDB.
2. **Retrieval**: On each question, the question is embedded and the top-4 most similar chunks are retrieved from ChromaDB.
3. **Generation**: Retrieved chunks are injected into the system prompt along with recent conversation history. Groq streams the LLM response token-by-token.
4. **Streaming**: The backend yields each token as a JSON line. The frontend reads the stream and appends tokens to the chat bubble in real time.
5. **Interrupt**: If the user sends a new message, `AbortController.abort()` cancels the fetch stream immediately. The backend detects the disconnection and stops generation.

---

## Development

Run with auto-reload during development:

```bash
uv run uvicorn app:app --reload --port 8000
```

To reset the vector store (delete all indexed documents):

```bash
# Windows
rmdir /s /q chroma_db

# macOS / Linux
rm -rf chroma_db/
```

---

## Troubleshooting

**"Microphone access denied"**
The browser requires HTTPS for microphone access on non-localhost origins. Run locally (`localhost`) or serve behind HTTPS.

**Answers are slow to start**
The first token latency depends on Groq's API response time (~200–500ms typically). Retrieval and embedding happen before the LLM is called, adding a small additional delay.

**"Could not understand audio"**
The recorded audio blob was too small (< 1 KB). Make sure you are speaking clearly and your microphone is working. On some browsers, a different audio format may be needed — the app tries `webm/opus`, `webm`, `ogg/opus`, and `mp4` in order.

**ChromaDB collection errors on startup**
Delete the `chroma_db/` directory and restart. This forces a clean rebuild next time documents are uploaded.

---

## License

MIT
