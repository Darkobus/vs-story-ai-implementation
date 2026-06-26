"""
Pobiera kluczowe strony wiki Vintage Story i ładuje je do ChromaDB (RAG).
Uzupełnia bazę o WIEDZĘ OPISOWĄ (procesy: stal, kwarc, rudy...), której nie ma
w surowych danych assetów gry.

Uruchom na serwerze (z lokalną Ollamą dla embeddingów):
    python ingest/wiki.py

Dodaj/zmień strony w liście PAGES poniżej i uruchom ponownie — dzięki upsert
nie powstaną duplikaty (te same ID nadpisują się).
"""

import re
import sys
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

OLLAMA_URL = "http://127.0.0.1:11434/api/embeddings"  # lokalna Ollama na serwerze
CHROMA_PATH = str(Path(__file__).resolve().parent.parent / "chroma_db")
COLLECTION_NAME = "vintage_story_wiki"
WIKI_API = "https://wiki.vintagestory.at/api.php"

# Strony do pobrania. Skup się na PROCESACH i mechanikach, nie pojedynczych przedmiotach
# (przedmioty/receptury są już w bazie z ingest/game_assets.py). Nieistniejące strony
# są pomijane, więc możesz spokojnie dopisywać kandydatów.
PAGES = [
    # Metalurgia / kucie
    "Metallurgy", "Casting", "Crucible", "Anvil", "Smithing", "Bloomery",
    "Steel", "Iron", "Copper", "Tin", "Bronze", "Bismuth", "Silver", "Gold", "Lead",
    "Ingot", "Bellows", "Forge", "Helve_hammer",
    # Wydobycie / kamień
    "Ore", "Prospecting", "Prospecting_pick", "Mining", "Quartz", "Rock", "Knapping",
    # Narzędzia
    "Axe", "Pickaxe", "Shovel", "Hammer", "Saw", "Chisel", "Knife", "Tools",
    # Przetrwanie / świat
    "Temporal_storm", "Drifter", "Charcoal", "Coke",
    "Cooking", "Food", "Clay_forming", "Pottery", "Firepit",
]


class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, url: str):
        self.url = url

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            req = urllib.request.Request(
                self.url,
                data=json.dumps({
                    "model": "nomic-embed-text", "prompt": text, "keep_alive": "30m"
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    embeddings.append(json.loads(r.read().decode())["embedding"])
            except Exception as e:
                print(f"  [embed error] {e}")
                embeddings.append([0.0] * 768)
        return embeddings


def clean_mediawiki_text(raw: str) -> str:
    """Zamienia wikitext na czysty tekst, ZACHOWUJĄC słowa z szablonów/linków
    (stary problem: usuwanie {{...}} zostawiało dziury typu '...via , and...')."""
    text = raw
    # Szablony: {{ll|Charcoal}} albo {{x|a|Display}} -> zachowaj OSTATNI segment (tekst),
    # zamiast usuwać całość. Pętla, bo szablony bywają zagnieżdżone.
    for _ in range(5):
        new = re.sub(r'\{\{(?:[^{}|]*\|)*([^{}|]+)\}\}', r'\1', text)
        if new == text:
            break
        text = new
    text = re.sub(r'\{\{[^{}]*\}\}', '', text)   # puste/parametrowe szablony do wyrzucenia

    # Linki: [[Strona|tekst]] -> tekst, [[Strona]] -> Strona
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    # Linki zewnętrzne: [http... tekst] -> tekst
    text = re.sub(r'\[https?://\S+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://\S+\]', '', text)

    # Nagłówki == X == -> X ; pogrubienia/kursywa
    text = re.sub(r'={2,}\s*(.*?)\s*={2,}', r'\1', text)
    text = re.sub(r"'''''(.*?)'''''", r'\1', text)
    text = re.sub(r"'''(.*?)'''", r'\1', text)
    text = re.sub(r"''(.*?)''", r'\1', text)

    # Tabele: zostaw treść komórek, usuń składnię (proporcje stopów są w tabelach!)
    text = re.sub(r'^\s*\{\|.*$', '', text, flags=re.MULTILINE)   # {| class="wikitable"
    text = text.replace('|}', '')
    text = re.sub(r'^\s*\|\+.*$', '', text, flags=re.MULTILINE)   # |+ podpis
    text = re.sub(r'^\s*\|-.*$', '', text, flags=re.MULTILINE)    # |- separator wiersza
    text = text.replace('||', ' | ').replace('!!', ' | ')
    text = re.sub(r'^\s*[!|]\s*', '', text, flags=re.MULTILINE)   # marker komórki ! lub |

    # HTML, magic words, sprzątanie
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'__\w+__', '', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def create_chunks(text: str, max_words: int = 150, overlap_words: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + max_words]))
        i += max_words - overlap_words
    return chunks


def fetch_wiki_page(title: str) -> str:
    params = {
        "action": "query", "prop": "revisions", "rvprop": "content",
        "rvslots": "main", "format": "json", "titles": title,
    }
    url = WIKI_API + "?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "vs-story-ai/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode())
        for pid, page in data["query"]["pages"].items():
            if pid == "-1":
                return ""  # strona nie istnieje
            return page["revisions"][0]["slots"]["main"]["*"]
    except Exception as e:
        print(f"  [fetch error] {title}: {e}")
    return ""


def is_english_title(title: str) -> bool:
    """Heurystyka: zachowaj tylko angielskie strony, pomiń obcojęzyczne.
    Odrzuca tytuły ze znakami non-ASCII (np. cyrylica) oraz sufiksy tłumaczeń
    typu '/ru', '/de', '/zh-hans'. Dzięki temu baza nie puchnie od wszystkich
    wersji językowych wiki (a embeddingi nomic i tak są pod angielski)."""
    if not title.isascii():
        return False
    if re.search(r'/[a-z]{2,3}(-[a-z]+)?$', title):
        return False
    return True


def list_all_wiki_pages() -> list[str]:
    """Pobiera tytuły WSZYSTKICH angielskich stron z głównej przestrzeni nazw wiki
    (API list=allpages, namespace 0 = artykuły; bez Talk/User/Template/Category)."""
    titles, apcontinue = [], None
    while True:
        params = {
            "action": "query", "list": "allpages", "apnamespace": "0",
            "aplimit": "500", "format": "json",
        }
        if apcontinue:
            params["apcontinue"] = apcontinue
        url = WIKI_API + "?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "vs-story-ai/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode())
        except Exception as e:
            print(f"  [allpages error] {e}")
            break
        titles += [p["title"] for p in data.get("query", {}).get("allpages", [])]
        cont = data.get("continue", {}).get("apcontinue")
        if not cont:
            break
        apcontinue = cont
        time.sleep(0.2)
    return titles


def main():
    use_all = "--all" in sys.argv
    english_only = "--en" in sys.argv  # opcjonalnie: tylko angielskie strony (pomiń inne języki)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=OllamaEmbeddingFunction(OLLAMA_URL),
    )

    if use_all:
        pages = list_all_wiki_pages()
        if english_only:
            before = len(pages)
            pages = [t for t in pages if is_english_title(t)]
            print(f"[*] Filtr --en: {before} -> {len(pages)} angielskich stron")
        print(f"[*] Tryb --all: {len(pages)} stron do ingestu")
    else:
        pages = PAGES
        print(f"[*] Tryb skrócony: {len(pages)} stron z listy PAGES (użyj --all dla całej wiki)")

    total_chunks, ok_pages = 0, 0
    for title in pages:
        raw = fetch_wiki_page(title)
        if not raw.strip():
            print(f"[skip] {title} (brak strony)")
            continue

        chunks = create_chunks(clean_mediawiki_text(raw))
        if not chunks:
            print(f"[skip] {title} (pusto po czyszczeniu)")
            continue

        ids = [f"wiki_{title}_{i}" for i in range(len(chunks))]
        metas = [{"source": title, "type": "wiki"} for _ in chunks]
        try:
            collection.upsert(documents=chunks, ids=ids, metadatas=metas)
            total_chunks += len(chunks)
            ok_pages += 1
            print(f"[ok]   {title}: {len(chunks)} chunków")
        except Exception as e:
            print(f"[err]  {title}: {e}")
        time.sleep(0.2)  # nie zasypuj wiki

    print(f"\n[DONE] {ok_pages}/{len(pages)} stron, {total_chunks} chunków dodanych/zaktualizowanych")


if __name__ == "__main__":
    main()
