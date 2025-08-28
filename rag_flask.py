import os, gc, traceback, datetime, json, sqlite3, multiprocessing
from typing import Dict, Any, List

from flask import Flask, request, jsonify
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

# ---------- 0. basic config ----------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
assert GEMINI_API_KEY, "Set GEMINI_API_KEY env var"

STORAGE_ROOT = os.getenv("AGENTIC_RAG_STORAGE", "./storage")
os.makedirs(STORAGE_ROOT, exist_ok=True)
CHROMA_DIR = os.path.join(STORAGE_ROOT, "chromadb")
os.makedirs(CHROMA_DIR, exist_ok=True)

# ---------- 1. Flask ----------
app = Flask(__name__)

# ---------- 2. Build the RAG chain ONCE globally ----------
# This is the most critical change. The objects are created when the app starts.
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=GEMINI_API_KEY,
)
emb = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY,
)
vectordb = Chroma(
    collection_name="ar_floats",
    embedding_function=emb,
    persist_directory=CHROMA_DIR,
)
retriever = vectordb.as_retriever(search_kwargs={"k": 2})
rag_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    return_source_documents=True,
)

# ---------- 3. REST endpoint ----------
@app.route("/ask", methods=["POST"])
def ask():
    try:
        query = request.json.get("query", "").strip()
        if not query:
            return jsonify({"error": "query required"}), 400

        # Reuse the globally created rag_chain for each request
        result = rag_chain({"query": query})

        answer = result["result"]
        sources = [
            {
                "id": doc.metadata.get("file", ""),
                "snippet": doc.page_content[:120],
            }
            for doc in result["source_documents"]
        ]

        # No need for explicit teardown
        return jsonify({"answer": answer, "sources": sources})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

# ---------- 4. gunicorn entrypoint ----------
# In Render “Start Command”:  gunicorn -w 1 -b 0.0.0.0:$PORT flask_app:app