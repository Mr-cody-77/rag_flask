import requests
import json

# Flask API URL (local test)
URL = "http://127.0.0.1:5000/ask"

# Test queries
test_queries = [
    "What is the average temperature of float 13857?",
    "Give me the location and date of the first cycle.",
    "Show me salinity data from float 13857"
]

for query in test_queries:
    print(f"\n--- Query: {query} ---")
    try:
        response = requests.post(URL, json={"query": query}, timeout=15)
        if response.status_code == 200:
            data = response.json()
            print("✅ Response:")
            print(json.dumps(data, indent=4))
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
