"""
RAG Chatbot — FastAPI Backend
==============================
Groq (LLM) + Cohere (embeddings) + ChromaDB (vector store)

Run with: python -m uvicorn app:app --reload
"""

import os
import re
import uuid
import httpx
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

import asyncio
import json

load_dotenv()

CHROMA_DIR  = "chroma_db"
DEFAULT_KB  = "default"

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ── Embeddings ────────────────────────────────────────────────────────────────

embeddings = CohereEmbeddings(
    model="embed-english-light-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY"),
)

# ── Shared ChromaDB client (all KBs live here) ────────────────────────────────

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# ── LLM ───────────────────────────────────────────────────────────────────────

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

# ── KB helpers ────────────────────────────────────────────────────────────────

def kb_collection_name(kb_id: str) -> str:
    """ChromaDB collection names must be slug-like."""
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", kb_id).strip("_")
    return slug or "default"

def get_vectorstore(kb_id: str) -> Chroma:
    return Chroma(
        client=chroma_client,
        collection_name=kb_collection_name(kb_id),
        embedding_function=embeddings,
    )

def list_kb_ids() -> list[str]:
    """Return all existing collection names as KB ids."""
    cols = chroma_client.list_collections()
    names = [c.name for c in cols]
    # Always ensure default exists in the list
    if DEFAULT_KB not in names:
        names = [DEFAULT_KB] + names
    return names

# ── In-memory doc tracker per KB ──────────────────────────────────────────────

kb_docs: dict[str, list[str]] = {}

def track_doc(kb_id: str, filename: str):
    if kb_id not in kb_docs:
        kb_docs[kb_id] = []
    if filename not in kb_docs[kb_id]:
        kb_docs[kb_id].append(filename)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")

# ── Knowledge base endpoints ──────────────────────────────────────────────────

@app.get("/knowledge-bases")
async def get_knowledge_bases():
    return JSONResponse({"knowledge_bases": list_kb_ids()})


class CreateKBRequest(BaseModel):
    name: str

@app.post("/knowledge-bases")
async def create_knowledge_base(body: CreateKBRequest):
    name = body.name.strip()
    if not name:
        return JSONResponse({"error": "Name cannot be empty."}, status_code=400)
    col_name = kb_collection_name(name)
    # Touch the collection to create it
    chroma_client.get_or_create_collection(col_name)
    return JSONResponse({"kb_id": col_name, "name": col_name})

# ── Chat (streaming) ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    question: str
    kb_id: str = DEFAULT_KB

@app.post("/chat")
async def chat(body: ChatRequest, request: Request):
    history = get_history(body.session_id)
    vs = get_vectorstore(body.kb_id)
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    docs = retriever.invoke(body.question)
    context = "\n\n".join(doc.page_content for doc in docs)
    sources = list({
        os.path.basename(doc.metadata.get("source", "Unknown"))
        for doc in docs
    })

    chain = prompt | llm | StrOutputParser()

    async def token_stream():
        full_answer = []
        try:
            async for chunk in chain.astream({
                "context": context,
                "chat_history": history,
                "question": body.question,
            }):
                # Stop streaming if client disconnected
                if await request.is_disconnected():
                    break
                full_answer.append(chunk)
                # Send each token as a JSON line (newline-delimited JSON)
                yield json.dumps({"token": chunk}) + "\n"

            # After streaming completes, send sources and save history
            answer_text = "".join(full_answer)
            if answer_text:
                history.append(HumanMessage(content=body.question))
                history.append(AIMessage(content=answer_text))
                if len(history) > 12:
                    sessions[body.session_id] = history[-12:]

            yield json.dumps({"done": True, "sources": sources}) + "\n"

        except asyncio.CancelledError:
            # Client disconnected mid-stream — that's fine
            pass

    return StreamingResponse(
        token_stream(),
        media_type="text/plain",
        headers={"X-Accel-Buffering": "no"},  # disable nginx buffering if behind proxy
    )

# ── Session ───────────────────────────────────────────────────────────────────

@app.post("/session/new")
async def new_session():
    return {"session_id": str(uuid.uuid4())}

# ── Documents ─────────────────────────────────────────────────────────────────

@app.get("/documents")
async def list_documents(kb_id: str = DEFAULT_KB):
    return JSONResponse({"documents": kb_docs.get(kb_id, [])})

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    kb_id: str = Form(DEFAULT_KB),
):
    filename = file.filename or "uploaded_file"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in (".pdf", ".txt"):
        return JSONResponse({"error": "Only PDF and TXT files are supported."}, status_code=400)

    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path) if ext == ".pdf" else TextLoader(tmp_path, encoding="utf-8")
        documents = loader.load()

        for doc in documents:
            doc.metadata["source"] = filename
            doc.metadata["kb_id"]  = kb_id

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        vs = get_vectorstore(kb_id)
        vs.add_documents(chunks)
        track_doc(kb_id, filename)

    finally:
        os.unlink(tmp_path)

    return JSONResponse({
        "message": f"'{filename}' uploaded to '{kb_id}' successfully.",
        "chunks": len(chunks),
        "filename": filename,
        "kb_id": kb_id,
    })

# ── Transcribe ────────────────────────────────────────────────────────────────

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
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
