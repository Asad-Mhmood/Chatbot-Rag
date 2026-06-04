"""
Document Ingestion Script
==========================
Loads PDFs from the /docs folder, splits them into chunks,
creates embeddings via Cohere API, and stores in ChromaDB.

Run once before starting the app:
    uv run python ingest.py
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DOCS_DIR   = "docs"
CHROMA_DIR = "chroma_db"


def ingest():
    print(f"Loading PDFs from '{DOCS_DIR}/'...")
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    documents = loader.load()

    if not documents:
        print("No PDFs found. Add PDF files to the 'docs/' folder and run again.")
        return

    print(f"Loaded {len(documents)} page(s) from {len(set(d.metadata['source'] for d in documents))} file(s).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    print("Creating embeddings via Cohere API...")
    embeddings = CohereEmbeddings(
        model="embed-english-light-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"Done. {db._collection.count()} chunks stored in '{CHROMA_DIR}/'.")
    print("Run: uv run uvicorn app:app --reload")


if __name__ == "__main__":
    ingest()
