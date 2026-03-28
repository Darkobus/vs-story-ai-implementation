import os
import json
import urllib.request
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
            except Exception as e:
                pass 
        return embeddings

def clean_mediawiki_text(raw_text: str) -> str:
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

OLLAMA_URL = "http://192.168.0.24:11434/api/embeddings"
chroma_client = chromadb.PersistentClient(path="./chroma_db")

wiki_collection = chroma_client.get_collection(
    name="vintage_story_wiki",
    embedding_function=OllamaEmbeddingFunction(OLLAMA_URL)
)

def process_and_ingest(raw_wiki_content: str, doc_id: str):
    cleaned_text = clean_mediawiki_text(raw_wiki_content)
    chunks = create_chunks(cleaned_text)
    
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": doc_id, "chunk_index": i} for i in range(len(chunks))]
    
    if chunks:
        wiki_collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
