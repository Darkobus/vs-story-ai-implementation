import json
import urllib.request
from pathlib import Path
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, url: str):
        self.url = url

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            req = urllib.request.Request(
                self.url,
                data=json.dumps({"model": "nomic-embed-text", "prompt": text}).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            try:
                with urllib.request.urlopen(req) as response:
                    res = json.loads(response.read().decode())
                    embeddings.append(res["embedding"])
            except Exception:
                pass
        return embeddings

OLLAMA_URL = "http://127.0.0.1:11434/api/embeddings"  # lokalna Ollama na serwerze
CHROMA_PATH = str(Path(__file__).resolve().parent.parent / "chroma_db")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

wiki_collection = chroma_client.get_collection(
    name="vintage_story_wiki",
    embedding_function=OllamaEmbeddingFunction(OLLAMA_URL)
)

queries = [
    "jak zrobic braz metalurgy",
    "how to make bronze metallurgy"
]

for query in queries:
    print(f"\n====================================")
    print(f"QUERY: {query}")
    print(f"====================================")

    results = wiki_collection.query(
        query_texts=[query],
        n_results=5,
        include=["documents", "distances"]
    )

    docs = results['documents'][0]
    dists = results['distances'][0]

    def fmt_dist(d):
        return round(d, 3) if isinstance(d, (int, float)) else d

    # Odległości pomagają dobrać RAG_MAX_DISTANCE w bot/watchdog.py:
    # ustaw próg nieco POWYŻEJ odległości trafnych wyników, a PONIŻEJ nietrafionych.
    print(f"Odległości (mniej = bliżej): {[fmt_dist(d) for d in dists]}")

    for i, (doc, dist) in enumerate(zip(docs, dists)):
        print(f"\n--- Result {i+1} | odległość={fmt_dist(dist)} ---")
        print(doc)
