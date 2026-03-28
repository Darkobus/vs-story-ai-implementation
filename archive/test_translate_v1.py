import json
import urllib.request
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

def translate_query(query: str, url: str, model_name: str = "llama3") -> str:
    prompt_text = f"Translate the following Polish text to English. Output only the exact translation, nothing else: {query}"
    req = urllib.request.Request(
        url,
        data=json.dumps({
            "model": model_name,
            "prompt": prompt_text,
            "stream": False
        }).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode())
            return res.get("response", "").strip()
    except Exception:
        return query

if __name__ == "__main__":
    OLLAMA_EMBED_URL = "http://192.168.0.24:11434/api/embeddings"
    OLLAMA_GENERATE_URL = "http://192.168.0.24:11434/api/generate"
    LLM_TRANSLATOR_MODEL = "gemma3:12b"

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    wiki_collection = chroma_client.get_collection(
        name="vintage_story_wiki",
        embedding_function=OllamaEmbeddingFunction(OLLAMA_EMBED_URL)
    )

    query = "jak zrobic braz metalurgy"
    
    translated = translate_query(query, OLLAMA_GENERATE_URL, LLM_TRANSLATOR_MODEL)
    
    print(f"ORIGINAL QUERY: {query}")
    print(f"TRANSLATED QUERY: {translated}")
    print("-" * 40)
    
    results = wiki_collection.query(
        query_texts=[translated],
        n_results=2
    )
    
    for i, doc in enumerate(results['documents'][0]):
        print(f"\nRESULT {i+1}:")
        print(doc)
