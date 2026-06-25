import urllib.request
import json
from pathlib import Path
import chromadb

OLLAMA_URL = "http://127.0.0.1:11434/api/embeddings"  # lokalna Ollama na serwerze

def test_ollama_connection():
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps({"model": "nomic-embed-text", "prompt": "test"}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            res = json.loads(response.read().decode())
            print("Ollama OK. Dimensions:", len(res.get("embedding", [])))
    except Exception as e:
        print("Ollama ERROR:", e)

def check_db_state():
    chroma_path = str(Path(__file__).resolve().parent.parent / "chroma_db")
    client = chromadb.PersistentClient(path=chroma_path)
    try:
        collection = client.get_collection(name="vintage_story_wiki")
        print("Documents in DB:", collection.count())
    except Exception as e:
        print("DB ERROR:", e)

if __name__ == "__main__":
    test_ollama_connection()
    check_db_state()
