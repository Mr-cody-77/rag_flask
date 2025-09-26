import os, traceback
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from chromadb import CloudClient
from functools import lru_cache
from sentence_transformers import SentenceTransformer

import logging
logging.basicConfig(level=logging.INFO)

# ---------- 0. basic config ----------
load_dotenv()

# Chroma API credentials
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "agro_rag_db")
assert CHROMA_API_KEY and CHROMA_TENANT, "Set Chroma API credentials in .env"

# ---------- 1. Flask ----------
app = Flask(__name__)

# ---------- 2. Load SentenceTransformer locally ----------
model = SentenceTransformer('all-MiniLM-L6-v2')  # small, fast, 384-dim embeddings

def embed_texts(texts):
    """Return list of embeddings as floats."""
    return model.encode(texts, convert_to_numpy=True).tolist()

# ---------- 3. Lazy-load RAG chain ----------
@lru_cache(maxsize=1)
def get_rag_chain():
    """Initialize and return the RAG chain with local embeddings."""
    # Remote Chroma (API-based)
    client = CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE
    )

    # Attach to the collection
    collection = client.get_collection("argo_data")

    vectordb = Chroma(
        client=client,
        collection_name="argo_data",
        embedding_function=embed_texts,  # use local SentenceTransformer
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    # Use a simple dummy LLM for testing / local deploy (optional)
    from langchain.llms import OpenAI  # or replace with any local LLM
    llm = OpenAI(temperature=0)  # only for text generation

    return RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True,
    )

# ---------- 4. REST endpoint ----------
@app.route("/ask", methods=["POST"])
def ask():
    try:
        query = request.json.get("query", "").strip()
        if not query:
            return jsonify({"error": "query required"}), 400

        logging.info(f"Received query: {query}")
        rag_chain = get_rag_chain()

        try:
            result = rag_chain({"query": query})
        except Exception as e:
            logging.error(f"RAG chain failed: {e}")
            return jsonify({"error": "Embedding or retrieval failed"}), 503

        answer = result["result"]

        sources = []
        for doc in result["source_documents"]:
            meta = doc.metadata
            snippet = doc.page_content[:200]
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

# ---------- 5. gunicorn entrypoint ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
