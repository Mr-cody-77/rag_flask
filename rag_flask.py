from flask import Flask, request, jsonify
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import chromadb
import os

app = Flask(__name__)
openai_api_key = os.environ.get("OPENAI_API_KEY")


llm = OpenAI(
    model_name="text-davinci-003",
    temperature=0,
    openai_api_key=openai_api_key
)
embeddings = OpenAIEmbeddings()


client = chromadb.CloudClient(
    api_key=os.environ.get("CHROMA_API_KEY"),          # Store your key in environment variables
    tenant=os.environ.get("CHROMA_TENANT"),
    database=os.environ.get("CHROMA_DATABASE")        # e.g., 'flask_rag_db'
)


collection = client.get_or_create_collection("agro_rag_collection")


retriever = Chroma(
    collection_name="agro_rag_collection",
    embedding_function=embeddings,
    client=client
).as_retriever(search_kwargs={"k": 5})

qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True)

@app.route("/ask", methods=["POST"])
def ask_query():
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
