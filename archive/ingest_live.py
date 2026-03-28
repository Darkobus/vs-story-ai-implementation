import urllib.request
import json
import urllib.parse
import re
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

def clean_text(raw_text: str) -> str:
    text = re.sub(r'__\w+__', '', raw_text)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'={2,}\s*(.*?)\s*={2,}', r'\1', text)
    text = re.sub(r"'''''(.*?)'''''", r'\1', text)
    text = re.sub(r"'''(.*?)'''", r'\1', text)
    text = re.sub(r"''(.*?)''", r'\1', text)
    text = re.sub(r'\[http\S+\s+(.*?)\]', r'\1', text)
    text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def create_chunks(cleaned_text: str, max_words: int = 150, overlap_words: int = 30) -> list[str]:
    words = cleaned_text.split()
    chunks = []
    if not words:
        return chunks
    current_index = 0
    while current_index < len(words):
        chunk_words = words[current_index : current_index + max_words]
        chunks.append(" ".join(chunk_words))
        current_index += (max_words - overlap_words)
    return chunks

def fetch_wiki_page(title: str) -> str:
    safe_title = urllib.parse.quote(title)
    url = f"https://wiki.vintagestory.at/api.php?action=query&prop=revisions&rvprop=content&rvslots=main&format=json&titles={safe_title}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_info in pages.items():
                if page_id == "-1":
                    return ""
                revisions = page_info.get("revisions", [])
                if revisions:
                    return revisions[0].get("slots", {}).get("main", {}).get("*", "")
    except Exception:
        return ""
    return ""

if __name__ == "__main__":
    db_client = chromadb.PersistentClient(path="./chroma_db")
    try:
        db_client.delete_collection(name="vintage_story_wiki")
    except Exception:
        pass

    collection = db_client.create_collection(
        name="vintage_story_wiki",
        embedding_function=OllamaEmbeddingFunction("http://192.168.0.24:11434/api/embeddings")
    )

    pages_to_fetch = ["Casting", "Bronze_alloy", "Metallurgy", "Crucible"]

    for page in pages_to_fetch:
        raw_content = fetch_wiki_page(page)
        if not raw_content:
            continue
            
        cleaned = clean_text(raw_content)
        chunks = create_chunks(cleaned)
        
        if chunks:
            ids = [f"{page}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [{"source": page} for _ in range(len(chunks))]
            collection.add(documents=chunks, metadatas=metadatas, ids=ids)
