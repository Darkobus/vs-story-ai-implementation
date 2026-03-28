import urllib.request
import urllib.parse
import json
import chromadb
import re

PAGES = ["Temporal_storm", "Knapping", "Casting", "Panning", "Drifter", "Smithing"]
PC_IP = "192.168.0.24"

OLLAMA_URL = f"http://{PC_IP}:11434"

class OllamaEmbeddingFunction:
    def name(self):
        return "ollama-embed-text"

    def __call__(self, input):
        embeddings = []
        for text in input:
            req = urllib.request.Request(
                f"{OLLAMA_URL}/api/embeddings",
                data=json.dumps({"model": "nomic-embed-text", "prompt": text}).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            try:
                with urllib.request.urlopen(req) as response:
                    res = json.loads(response.read().decode('utf-8'))
                    embeddings.append(res["embedding"])
            except Exception as e:
                print(e)
        return embeddings

def clean_wikitext(text):
    text = re.sub(r'{{.*?}}', '', text, flags=re.DOTALL)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'<.*?>', '', text)
    return text.strip()

def fetch_wiki_page(title):
    url = f"https://wiki.vintagestory.at/api.php?action=query&prop=revisions&rvprop=content&rvslots=main&format=json&titles={urllib.parse.quote(title)}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'VSAI/1.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            pages = data['query']['pages']
            for page_id in pages:
                if page_id == "-1": 
                    return ""
                content = pages[page_id]['revisions'][0]['slots']['main']['*']
                return clean_wikitext(content)
    except Exception:
        return ""

def chunk_text(text, size=200):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]

def main():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="vintage_story_wiki",
        embedding_function=OllamaEmbeddingFunction()
    )

    for page in PAGES:
        raw_text = fetch_wiki_page(page)
        if not raw_text: 
            continue
        
        chunks = chunk_text(raw_text)
        
        ids = [f"{page}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": page} for _ in chunks]
        
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )

if __name__ == "__main__":
    main()
