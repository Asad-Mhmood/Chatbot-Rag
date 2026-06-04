
"""
RAG Chatbot — FastAPI Backend
==============================
Groq (LLM) + Cohere (embeddings) + ChromaDB (vector store)

Run with: uv run uvicorn app:app --reload
"""

import os
import uuid
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_DIR = "chroma_db"

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ── Embeddings (Cohere API) ───────────────────────────────────────────────────

embeddings = CohereEmbeddings(
    model="embed-english-light-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY"),
)

# ── Vector store ──────────────────────────────────────────────────────────────

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)

# ── LLM (Groq) ────────────────────────────────────────────────────────────────

llm = ChatGroq(
    model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
    temperature=0.3,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

# ── Prompt ────────────────────────────────────────────────────────────────────

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions based on the provided documents.
Use the context below to answer. If the answer is not in the context, say so honestly.
Keep answers concise and accurate.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# ── Session store ─────────────────────────────────────────────────────────────

sessions: dict[str, list] = {}


def get_history(session_id: str) -> list:
    if session_id not in sessions:
        sessions[session_id] = []
    return sessions[session_id]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


class ChatRequest(BaseModel):
    session_id: str
    question: str


@app.post("/chat")
async def chat(body: ChatRequest):
    history = get_history(body.session_id)

    docs = retriever.invoke(body.question)
    context = "\n\n".join(doc.page_content for doc in docs)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "chat_history": history,
        "question": body.question,
    })

    history.append(HumanMessage(content=body.question))
    history.append(AIMessage(content=answer))
    if len(history) > 12:
        sessions[body.session_id] = history[-12:]

    sources = list({
        os.path.basename(doc.metadata.get("source", "Unknown"))
        for doc in docs
    })

    return JSONResponse({"answer": answer, "sources": sources})


@app.post("/session/new")
async def new_session():
    return {"session_id": str(uuid.uuid4())}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Transcribe audio using Groq Whisper API."""
    audio_bytes = await audio.read()

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
            files={"file": (audio.filename or "audio.webm", audio_bytes, audio.content_type or "audio/webm")},
            data={"model": "whisper-large-v3-turbo", "response_format": "json"},
        )

    if response.status_code != 200:
        return JSONResponse({"error": "Transcription failed", "detail": response.text}, status_code=500)

    text = response.json().get("text", "").strip()
    return JSONResponse({"text": text})
