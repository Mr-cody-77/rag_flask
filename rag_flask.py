import os, traceback
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from functools import lru_cache
from chromadb import CloudClient

import logging
logging.basicConfig(level=logging.INFO)

# ---------- 0. basic config ----------
load_dotenv()

# Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
assert GEMINI_API_KEY, "Set GEMINI_API_KEY env var"

# Chroma API credentials
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "agro_rag_db")

assert CHROMA_API_KEY and CHROMA_TENANT, "Set Chroma API credentials in .env"

# ---------- 1. Flask ----------
app = Flask(__name__)

# ---------- 2. Lazy-load RAG chain ----------

@lru_cache(maxsize=1)
def get_rag_chain():
    """Initialize and return the RAG chain. Cached to reuse for subsequent requests."""

    # LLM -> Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",   # lightweight + cheap, good for retrieval
        google_api_key=GEMINI_API_KEY,
        temperature=0
    )

    # Embeddings -> Gemini (must match ingestion model)
    emb = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    # Remote Chroma (API-based)
    client = CloudClient(
        api_key=os.getenv("CHROMA_API_KEY"),
        tenant=os.getenv("CHROMA_TENANT"),
        database="flask_rag_db"   # <-- explicitly set database name
    )

    # Attach to the same collection
    collection = client.get_collection("argo_dataset")

    vectordb = Chroma(
        client=client,
        collection_name="arga_dataset",
        embedding_function=emb,
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    return RetrievalQA.from_chain_type(
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

        logging.info(f"Received query: {query}")
        rag_chain = get_rag_chain()
        logging.info("Calling RAG chain...")
        result = rag_chain({"query": query})
        logging.info("RAG chain returned successfully")

        answer = result["result"]
        sources = [
            {"id": doc.metadata.get("file", ""), "snippet": doc.page_content[:120]}
            for doc in result["source_documents"]
        ]
        return jsonify({"answer": answer, "sources": sources})

    except Exception as e:
        logging.exception("Error in /ask endpoint")
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

# ---------- 4. gunicorn entrypoint ----------
# Render “Start Command”: gunicorn -w 1 -b 0.0.0.0:$PORT flask_app:app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
