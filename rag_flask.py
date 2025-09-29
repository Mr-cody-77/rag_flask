import os
import traceback
import logging
import math
from functools import lru_cache
import asyncio
from dotenv import load_dotenv
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- 0. basic config ----------
load_dotenv()

def _env(name, default=None):
    v = os.getenv(name, default)
    return v.strip() if isinstance(v, str) else v

GEMINI_API_KEY = _env("GEMINI_API_KEY")
CHROMA_API_KEY = _env("CHROMA_API_KEY")
CHROMA_TENANT = _env("CHROMA_TENANT")
CHROMA_DATABASE = _env("CHROMA_DATABASE") or "flask_rag_db"
CHROMA_COLLECTION = _env("CHROMA_COLLECTION") or "argo_data_local"

# Warnings instead of asserts so the server can start for debugging
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not set in .env — RAG chain will be unavailable until set.")
if not (CHROMA_API_KEY and CHROMA_TENANT):
    logger.warning("CHROMA_API_KEY or CHROMA_TENANT not set in .env — RAG chain will be unavailable until set.")

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
    "bering sea": {"lat": 60, "lon": -180},
}

# ---------- 2b. Simple local ocean stats (fallback) ----------
# Approximate typical surface temperatures (°C) and salinities (PSU) for fallback responses.
OCEAN_STATS = {
    "indian ocean": {
        "temp_range_c": "22–28°C (typical surface range)",
        "salinity_psu": "34–35 PSU (surface average)",
        "pres_mean": "approx. surface pressures near 1 atm",
    },
    "bay of bengal": {
        "temp_range_c": "26–30°C (warmer tropical bay)",
        "salinity_psu": "32–34 PSU (due to river inputs)",
        "pres_mean": "approx. surface pressures near 1 atm",
    },
    "pacific ocean": {
        "temp_range_c": "2–30°C (wide range; tropical surface ~25–30°C)",
        "salinity_psu": "34–35 PSU (surface average)",
        "pres_mean": "approx. surface pressures near 1 atm",
    },
    "atlantic ocean": {
        "temp_range_c": "2–28°C (varies strongly by latitude)",
        "salinity_psu": "35–36 PSU (surface average)",
        "pres_mean": "approx. surface pressures near 1 atm",
    },
    # reasonable defaults
    "default": {
        "temp_range_c": "surface temps typically range ~2–30°C depending on latitude",
        "salinity_psu": "around 34–36 PSU at the surface",
        "pres_mean": "approx. surface pressures near 1 atm",
    }
}

# ---------- 3. Utilities ----------
def safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def haversine_distance(lat1, lon1, lat2, lon2):
    try:
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return R * c
    except Exception:
        return float("inf")

def find_ocean_in_query(query):
    q = (query or "").lower()
    # simple substring match; checks longer names first for safety
    for name in sorted(OCEAN_COORDS.keys(), key=lambda x: -len(x)):
        if name in q:
            return name
    return None

def filter_documents_by_ocean(docs, ocean_name, max_distance_km=1000):
    if ocean_name not in OCEAN_COORDS:
        return docs
    ocean_lat = OCEAN_COORDS[ocean_name]["lat"]
    ocean_lon = OCEAN_COORDS[ocean_name]["lon"]
    filtered = []
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        lat = safe_float(meta.get("latitude") or meta.get("lat"))
        lon = safe_float(meta.get("longitude") or meta.get("lon"))
        if lat is None or lon is None:
            continue
        if haversine_distance(ocean_lat, ocean_lon, lat, lon) <= max_distance_km:
            filtered.append(doc)
    return filtered

def is_coordinate_query(query):
    q = (query or "").lower()
    return any(tok in q for tok in ("coordinate", "coordinates", "where", "lat", "lon", "longitude", "latitude"))

def is_temperature_query(query):
    q = (query or "").lower()
    # accept common variants, including misspellings
    temp_tokens = ("temp", "temperature", "temparture", "temprt", "sea temperature", "surface temp", "water temp")
    return any(tok in q for tok in temp_tokens)

def ensure_event_loop():
    """Make sure each thread has an asyncio event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
# ---------- 4. Lazy-load RAG chain (attempt) ----------
@lru_cache(maxsize=1)
def get_rag_chain():
    """
    Try to initialize an actual RAG chain if libraries and credentials are available.
    If any import or auth step fails, raise RuntimeError — caller will handle fallback.
    """
    ensure_event_loop()

    try:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set — cannot initialize LLM.")
        if not (CHROMA_API_KEY and CHROMA_TENANT):
            raise RuntimeError("CHROMA credentials not set — cannot initialize Chroma client.")

        # Imports (lazy)
        from langchain.chains import RetrievalQA
        from langchain_chroma import Chroma
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from chromadb import CloudClient
        from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

        logger.info("Initializing RAG chain (langchain + chroma + gemini)...")

        # 1. Build system + human prompt template
        system_template = (
            "You are an oceanography expert. Respond in a friendly, scientific way. "
            "PSAL = salinity, TEMP = temperature, PRES = pressure. "
            "If user gives an ocean/sea, infer coordinates and filter ChromaDB profiles nearby. "
            "If no profiles exist, return approximate coordinates from internal knowledge."
        )
        system_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_prompt = HumanMessagePromptTemplate.from_template("{query}")
        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

        # 2. Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            model_kwargs={
                "system_instruction": "You are an ocean science assistant..."
            }
        )

        # 3. Embeddings
        embedding_function = GoogleGenerativeAIEmbeddings(
            model="embedding-001",
            google_api_key=GEMINI_API_KEY
        )

        # 4. Chroma client + retriever
        client = CloudClient(api_key=CHROMA_API_KEY, tenant=CHROMA_TENANT, database=CHROMA_DATABASE)
        vectordb = Chroma(
            client=client,
            collection_name=CHROMA_COLLECTION,
            embedding_function=embedding_function
        )
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})

        # 5. Build RAG chain with prompt
        return RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": chat_prompt},
            return_source_documents=True
        )

    except Exception as e:
        raise

# ---------- 5. Routes ----------
@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "ok", "message": "Flask RAG server is running"}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/init_rag", methods=["GET"])
def init_rag():
    """
    Try to initialize the RAG chain and return detailed error (safe to use only in dev).
    Useful to debug which credential (Chroma or Gemini) failed.
    """
    try:
        _ = get_rag_chain()
        return jsonify({"status": "ok", "detail": "RAG chain initialized", "collection": CHROMA_COLLECTION}), 200
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("RAG init failed: %s\n%s", e, tb)
        return jsonify({"status": "error", "error": str(e), "trace": tb}), 500

@app.route("/ask", methods=["POST"])
def ask():
    try:
        payload = request.get_json(force=False, silent=True) or {}
        query = (payload.get("query", "") or "").strip()
        if not query:
            return jsonify({"error": "query required"}), 400

        # detect ocean name early so we can fallback to coords or local stats
        ocean_name = find_ocean_in_query(query)
        rag_chain = None

        # Try to initialize RAG chain; if unavailable, return a helpful fallback for common queries
        try:
            rag_chain = get_rag_chain()
        except Exception as e:
            logger.warning("RAG chain not available: %s", e)

            # If user asked for coordinates, return local coords fallback
            if ocean_name and is_coordinate_query(query):
                lat = OCEAN_COORDS[ocean_name]["lat"]
                lon = OCEAN_COORDS[ocean_name]["lon"]
                return jsonify({
                    "answer": f"Approximate coordinates for '{ocean_name}': latitude={lat}, longitude={lon}",
                    "sources": [{"latitude": lat, "longitude": lon, "snippet": "fallback from local OCEAN_COORDS"}]
                }), 200

            # If user asked for temperature (or similar), return local approximate stats
            if ocean_name and is_temperature_query(query):
                stats = OCEAN_STATS.get(ocean_name, OCEAN_STATS["default"])
                lat = OCEAN_COORDS[ocean_name]["lat"]
                lon = OCEAN_COORDS[ocean_name]["lon"]
                answer = (
                    f"Approximate surface temperature for '{ocean_name}': {stats['temp_range_c']}. "
                    f"Typical surface salinity: {stats['salinity_psu']}. "
                    f"Approx. coordinates (center): lat={lat}, lon={lon}."
                )
                return jsonify({
                    "answer": answer,
                    "sources": [{
                        "latitude": lat,
                        "longitude": lon,
                        "temp_range_c": stats.get("temp_range_c"),
                        "salinity_psu": stats.get("salinity_psu"),
                        "snippet": "fallback from internal OCEAN_STATS"
                    }]
                }), 200

            # Generic fallback if RAG not available but we detected ocean
            if ocean_name:
                lat = OCEAN_COORDS[ocean_name]["lat"]
                lon = OCEAN_COORDS[ocean_name]["lon"]
                stats = OCEAN_STATS.get(ocean_name, OCEAN_STATS["default"])
                return jsonify({
                    "answer": (
                        f"No external RAG available. For '{ocean_name}', approximate coords lat={lat}, lon={lon}; "
                        f"surface temp ~{stats['temp_range_c']}; salinity ~{stats['salinity_psu']}."
                    ),
                    "sources": [{"latitude": lat, "longitude": lon, "snippet": "fallback summary from local data"}]
                }), 200

            # If no ocean detected and RAG unavailable, return error
            return jsonify({"error": "RAG chain unavailable and no local fallback applicable", "details": str(e)}), 503

        # If RAG chain available, run it and post-process sources
        result = rag_chain({"query": query})
        docs = result.get("source_documents", []) or []

        # If an ocean was mentioned, filter the returned docs to that ocean region
        if ocean_name:
            docs = filter_documents_by_ocean(docs, ocean_name, max_distance_km=1000)

        answer = result.get("result", "No answer returned")
        sources = []
        for doc in docs:
            meta = getattr(doc, "metadata", {}) or {}
            snippet = getattr(doc, "page_content", "")[:300]
            sources.append({
                "latitude": safe_float(meta.get("latitude") or meta.get("lat")),
                "longitude": safe_float(meta.get("longitude") or meta.get("lon")),
                "temp_mean": meta.get("temp_mean"),
                "sal_mean": meta.get("sal_mean") or meta.get("psal_mean") or meta.get("salinity_mean"),
                "pres_mean": meta.get("pres_mean"),
                "snippet": snippet
            })

        return jsonify({"answer": answer, "sources": sources}), 200

    except Exception as e:
        logger.exception("Unhandled error in /ask")
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

# ---------- 6. Entrypoint ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Flask app on http://127.0.0.1:%s", port)
    app.run(host="127.0.0.1", port=port, debug=True)
