from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import HuggingFaceHub
import os
import traceback

app = Flask(__name__)

# -----------------------------
# Embeddings (lightweight)
# -----------------------------
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# -----------------------------
# LLM (HuggingFace Inference API)
# -----------------------------
llm = HuggingFaceHub(
    repo_id="google/flan-t5-mini",  # very small, CPU-friendly
    model_kwargs={"temperature": 0, "max_new_tokens": 150},
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
)

# -----------------------------
# Vector store (Chroma Cloud)
# -----------------------------
vector_store = Chroma(
    collection_name="agro_rag_collection",
    embedding_function=embeddings,
    chroma_api_impl="cloud",  # mandatory for cloud usage
    chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # smaller k to save RAM

# -----------------------------
# Retrieval QA chain
# -----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    return_source_documents=True
)

# -----------------------------
# API Endpoint
# -----------------------------
@app.route("/ask", methods=["POST"])
def ask_query():
    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "No query provided"}), 400

        result = qa_chain({"query": query})
        answer = result.get("result", "")
        sources = [
            {"id": doc.metadata.get("id", ""), "snippet": doc.page_content[:200]}
            for doc in result.get("source_documents", [])
        ]
        return jsonify({"answer": answer, "sources": sources})

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    # Use Render-assigned port or default 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
