import os
import math
from google import genai
from google.genai.types import EmbedContentConfig
from dotenv import load_dotenv
load_dotenv()
def calculate_dot_product(vec1: list[float], vec2: list[float]) -> float:
    """Calculates the dot product of two vectors, which serves as 
    cosine similarity for normalized embedding vectors."""
    # Ensure vectors are non-empty and of the same length before calculation
    if len(vec1) != len(vec2) or not vec1:
        return 0.0
    return sum(a * b for a, b in zip(vec1, vec2))

def run_simple_embedding_test():
    """Initializes the client, generates embeddings, and checks semantic similarity."""
    
    # --- Check API Key ---
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY environment variable is not set.")
        print("Please set your API key to run this test.")
        return

    try:
        # --- 1. Setup ---
        client = genai.Client()
        model_name = 'gemini-embedding-001'

        print(f"--- Running Simple Embedding Test for {model_name} ---")

        # Test Sentences
        text_related_1 = "The cat is a small domestic feline animal."
        text_related_2 = "A kitten is a baby cat, often playful."
        text_unrelated = "Building software requires detailed architectural planning."
        
        all_texts = [text_related_1, text_related_2, text_unrelated]
        
        # Configure the request to use the semantic similarity task type
        config = EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")

        # --- 2. API Call (Batching all three texts) ---
        response = client.models.embed_content(
            model=model_name,
            contents=all_texts,
            config=config
        )
        
        # --- 3. Verification ---
        # FIX: Extract the actual vector data (list of floats) from the ContentEmbedding objects
        raw_vectors = [e.values for e in response.embeddings]
        
        if not raw_vectors or len(raw_vectors) != 3:
            print(f"ERROR: API returned an unexpected number of vectors: {len(raw_vectors)}")
            return
            
        vector_dim = len(raw_vectors[0])
        print(f"✅ Success: API call worked! Generated {len(raw_vectors)} vectors of dimension {vector_dim}.")

        # Calculate Similarity Scores (Dot Product using the raw vectors)
        sim_related = calculate_dot_product(raw_vectors[0], raw_vectors[1])
        sim_unrelated_1 = calculate_dot_product(raw_vectors[0], raw_vectors[2])
        sim_unrelated_2 = calculate_dot_product(raw_vectors[1], raw_vectors[2])

        # --- 4. Results ---
        print("\n--- Semantic Check ---")
        print(f"Related Pair (Cat/Kitten) Score: {sim_related:.4f}")
        print(f"Unrelated Pair 1 (Cat/Software) Score: {sim_unrelated_1:.4f}")
        print(f"Unrelated Pair 2 (Kitten/Software) Score: {sim_unrelated_2:.4f}")

        if sim_related > sim_unrelated_1 and sim_related > sim_unrelated_2 and sim_related > 0.7:
            print("\n✅ Test PASS: Related sentences are significantly more similar than unrelated ones.")
        else:
            print("\n❌ Test FAIL: Similarity check did not meet expected criteria.")


    except Exception as e:
        print(f"\n❌ API Error: Failed to connect or receive valid data.")
        print(f"   Details: {e}")
        print("   Please ensure your GEMINI_API_KEY is correct and network access is available.")

if __name__ == "__main__":
    run_simple_embedding_test()
