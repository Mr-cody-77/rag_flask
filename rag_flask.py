from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import HuggingFaceHub
import os
import traceback

app = Flask(__name__)

# -----------------------------
# Embeddings (free)
# -----------------------------
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# -----------------------------
# LLM (free/open-source via HuggingFaceHub)
# -----------------------------
# Ensure HUGGINGFACE_API_KEY is set in Render environment variables
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",  # lighter/faster models can be used
    model_kwargs={"temperature": 0, "max_new_tokens": 200},
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
)

# -----------------------------
# Vector store (Chroma Cloud fix)
# -----------------------------
vector_store = Chroma(
    collection_name="agro_rag_collection",
    embedding_function=embeddings,
    chroma_api_impl="cloud",                       # must specify cloud
    chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

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
        # Catch errors (like rate limits, missing API keys) and return JSON
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
