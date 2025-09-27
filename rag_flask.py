import os
import traceback
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from functools import lru_cache
from chromadb import CloudClient
import logging

logging.basicConfig(level=logging.INFO)

# ---------- 0. basic config ----------
load_dotenv()
GEMINI_API_KEY = "AIzaSyB9e7vyRogVxPw92vCBwIsOIFCxpKH5ng8"
assert GEMINI_API_KEY, "Set GEMINI_API_KEY env var"

CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = "flask_rag_db"  # MUST match server database
assert CHROMA_API_KEY and CHROMA_TENANT, "Set Chroma API credentials in .env"

# ---------- 1. Flask ----------
app = Flask(__name__)

# ---------- 2. Lazy-load RAG chain ----------
@lru_cache(maxsize=1)
def get_rag_chain():
    logging.info("Initializing RAG chain...")

    # LLM -> Gemini (only for generation, NOT embeddings)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0
    )

    # Remote Chroma client (already contains precomputed embeddings)
    client = CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE
    )

    # Collection inside the database
    collection_name = "argo_data"

    # Attach Chroma vector DB without creating new embeddings
    vectordb = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=None  # <- Important: don't compute embeddings on queries
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True
    )

    logging.info("RAG chain initialized successfully.")
    return chain

# ---------- 3. REST endpoint ----------
@app.route("/ask", methods=["POST"])
def ask():
    try:
        query = request.json.get("query", "").strip()
        if not query:
            return jsonify({"error": "query required"}), 400

        logging.info(f"Received query: {query}")
        rag_chain = get_rag_chain()
        logging.info("Calling RAG chain...")

        try:
            result = rag_chain({"query": query})
        except Exception as chain_err:
            logging.exception("Error during RAG chain execution")
            return jsonify({
                "error": "Internal error while processing the query",
                "trace": traceback.format_exc()
            }), 500

        logging.info("RAG chain returned successfully")

        answer = result.get("result", "No answer returned")
        sources = []
        for doc in result.get("source_documents", []):
            meta = getattr(doc, "metadata", {})
            snippet = getattr(doc, "page_content", "")[:200]
            sources.append({
                "latitude": meta.get("latitude"),
                "longitude": meta.get("longitude"),
                "time": meta.get("time"),
                "temp_mean": meta.get("temp_mean"),
                "sal_mean": meta.get("sal_mean"),
                "pres_mean": meta.get("pres_mean"),
                "snippet": snippet
            })

        return jsonify({"answer": answer, "sources": sources})

    except Exception as e:
        logging.exception("Error in /ask endpoint")
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

# ---------- 4. gunicorn entrypoint ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
