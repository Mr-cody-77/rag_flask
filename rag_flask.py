import os, traceback
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from functools import lru_cache

# ---------- 0. basic config ----------
load_dotenv()

# LLM API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
assert GEMINI_API_KEY, "Set GEMINI_API_KEY env var"

# Chroma API credentials
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "flask_rag_db")

assert CHROMA_API_KEY and CHROMA_TENANT, "Set Chroma API credentials in .env"

# ---------- 1. Flask ----------
app = Flask(__name__)

# ---------- 2. Lazy-load RAG chain ----------
@lru_cache(maxsize=1)
def get_rag_chain():
    """Initialize and return the RAG chain. Cached to reuse for subsequent requests."""
    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=GEMINI_API_KEY,
    )

    # Embeddings
    emb = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY,
    )

    # Remote Chroma (API-based)
    vectordb = Chroma(
        collection_name=CHROMA_DATABASE,
        embedding_function=emb,
        client_settings={
            "chroma_api_impl": "rest",
            "chroma_api_key": CHROMA_API_KEY,
            "chroma_tenant": CHROMA_TENANT,
        }
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

        rag_chain = get_rag_chain()
        result = rag_chain({"query": query})

        answer = result["result"]
        sources = [
            {"id": doc.metadata.get("file", ""), "snippet": doc.page_content[:120]}
            for doc in result["source_documents"]
        ]

        return jsonify({"answer": answer, "sources": sources})

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

# ---------- 4. gunicorn entrypoint ----------
# Render “Start Command”: gunicorn -w 1 -b 0.0.0.0:$PORT flask_app:app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)