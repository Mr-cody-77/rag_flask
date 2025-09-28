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
import math

logging.basicConfig(level=logging.INFO)

# ---------- 0. basic config ----------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = "flask_rag_db"  

assert GEMINI_API_KEY, "Set GEMINI_API_KEY env var"
assert CHROMA_API_KEY and CHROMA_TENANT, "Set Chroma API credentials in .env"

# ---------- 1. Flask ----------
app = Flask(__name__)

# ---------- 2. Ocean coordinates ----------
OCEAN_COORDS = { 
    "pacific ocean": {"lat": 0, "lon": -160}, 
    "atlantic ocean": {"lat": 0, "lon": -30}, 
    "indian ocean": {"lat": -20, "lon": 80}, 
    "southern ocean": {"lat": -60, "lon": 0}, 
    "arctic ocean": {"lat": 75, "lon": 0}, 
    "arabian sea": {"lat": 15, "lon": 65}, 
    "bay of bengal": {"lat": 15, "lon": 90}, 
    "mediterranean sea": {"lat": 35, "lon": 18}, 
    "caribbean sea": {"lat": 15, "lon": -75}, 
    "bering sea": {"lat": 60, "lon": -180}
}

# ---------- 3. Utility: nearest profile filter ----------
def haversine(lat1, lon1, lat2, lon2):
    # Returns distance in km
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R*c

def filter_by_nearest_coord(docs, lat, lon, max_km=500):
    filtered_docs = []
    for doc in docs:
        meta = getattr(doc, "metadata", {})
        doc_lat = meta.get("latitude")
        doc_lon = meta.get("longitude")
        if doc_lat is None or doc_lon is None:
            continue
        if haversine(lat, lon, doc_lat, doc_lon) <= max_km:
            filtered_docs.append(doc)
    return filtered_docs

def resolve_ocean_coordinates(name):
    name = name.lower()
    if name in OCEAN_COORDS:
        return OCEAN_COORDS[name]["lat"], OCEAN_COORDS[name]["lon"]
    return None, None

import math

def haversine_distance(lat1, lon1, lat2, lon2):
    # Haversine formula to compute distance in km between two lat/lon points
    R = 6371  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def filter_documents_by_ocean(docs, ocean_name, max_distance_km=1000):
    """
    Filter ChromaDB retrieved documents to only include floats near the ocean coordinates.
    """
    if ocean_name not in OCEAN_COORDS:
        return docs  # fallback: return all docs

    ocean_lat = OCEAN_COORDS[ocean_name]["lat"]
    ocean_lon = OCEAN_COORDS[ocean_name]["lon"]

    filtered = []
    for doc in docs:
        meta = getattr(doc, "metadata", {})
        lat = meta.get("latitude")
        lon = meta.get("longitude")
        if lat is None or lon is None:
            continue
        dist = haversine_distance(ocean_lat, ocean_lon, lat, lon)
        if dist <= max_distance_km:
            filtered.append(doc)
    return filtered


# ---------- 4. Lazy-load RAG chain ----------
@lru_cache(maxsize=1)
def get_rag_chain():
    logging.info("Initializing RAG chain...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=GEMINI_API_KEY,
        temperature=0,
        system_prompt=(
            "You are an oceanography expert. Respond in a friendly, scientific way. "
            "PSAL stands for salinity, TEMP for temperature, PRES for pressure "
            "If the user provides the name of an ocean or sea, infer approximate latitude and longitude "
            "and use it to retrieve relevant profiles."
            "If the user gives the name of an ocean/sea, infer its approximate latitude and longitude and focus your answer only on floats near that location."
            "If a query asks for the coordinates of an ocean or sea, and no profile exists in ChromaDB, provide the approximate coordinates from your internal knowledge (like OCEAN_COORDS)."
        )
    )

    client = CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE
    )

    vectordb = Chroma(
        client=client,
        collection_name="argo_data",
        embedding_function=None
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    return RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True)

# ---------- 5. REST endpoint ----------
@app.route("/ask", methods=["POST"])
# ---------- 5. REST endpoint ----------
@app.route("/ask", methods=["POST"])
def ask():
    try:
        query = request.json.get("query", "").strip()
        if not query:
            return jsonify({"error": "query required"}), 400

        rag_chain = get_rag_chain()

        # 1️⃣ Detect ocean/sea in the query
        ocean_name = None
        for name in OCEAN_COORDS.keys():
            if name in query.lower():
                ocean_name = name
                break

        # 2️⃣ Call the RAG chain
        result = rag_chain({"query": query})
        docs = result.get("source_documents", [])

        # 3️⃣ Filter docs by ocean coordinates if detected
        if ocean_name:
            docs = filter_documents_by_ocean(docs, ocean_name, max_distance_km=1000)

        # 4️⃣ Prepare response
        answer = result.get("result", "No answer returned")
        sources = []
        for doc in docs:
            meta = getattr(doc, "metadata", {})
            snippet = getattr(doc, "page_content", "")[:200]
            sources.append({
                "latitude": meta.get("latitude"),
                "longitude": meta.get("longitude"),
                "temp_mean": meta.get("temp_mean"),
                "sal_mean": meta.get("sal_mean"),
                "pres_mean": meta.get("pres_mean"),
                "snippet": snippet
            })

        return jsonify({"answer": answer, "sources": sources})

    except Exception as e:
        logging.exception("Error in /ask endpoint")
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ---------- 6. gunicorn entrypoint ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
