"""
Fetch Vintage Story wiki pages and load them into the knowledge base.
Adds DESCRIPTIVE knowledge (processes: steel, quartz, ores...) that the raw game
assets do not contain.

Usage:
    python -m ingest.wiki [--all] [--en] [--reset]

Options:
    --all     every article in the wiki's main namespace (default: the curated PAGES list)
    --en      with --all: English pages only (skip /de, /ru, ... translations)
    --reset   drop and recreate the collection first (normally only the first ingest step resets)

Re-running is safe: chunks are upserted by ID, so nothing is duplicated.
"""

import argparse
import json
import re
import time
import urllib.parse
import urllib.request

from vsai import config, kb

WIKI_API = "https://wiki.vintagestory.at/api.php"
WIKI_PAGE_URL = "https://wiki.vintagestory.at/"
USER_AGENT = "vs-story-ai/1.0 (knowledge-base ingest)"
BATCH = 50  # MediaWiki returns content for up to 50 titles per request

# Pages to fetch by default. Focus on PROCESSES and mechanics, not single items
# (items/recipes come from ingest/game_assets.py). Missing pages are skipped.
PAGES = [
    # Metallurgy / smithing
    "Metallurgy", "Casting", "Crucible", "Anvil", "Smithing", "Bloomery",
    "Steel", "Iron", "Copper", "Tin", "Bronze", "Bismuth", "Silver", "Gold", "Lead",
    "Ingot", "Bellows", "Forge", "Helve_hammer",
    # Mining / stone
    "Ore", "Prospecting", "Prospecting_pick", "Mining", "Quartz", "Rock", "Knapping",
    # Tools
    "Axe", "Pickaxe", "Shovel", "Hammer", "Saw", "Chisel", "Knife", "Tools",
    # Survival / world
    "Temporal_storm", "Drifter", "Charcoal", "Coke",
    "Cooking", "Food", "Clay_forming", "Pottery", "Firepit",
]


# Infobox parameters that are presentation, not knowledge
SKIP_INFOBOX_KEYS = {"image", "icon", "link", "outputlink"}


def _render_template(match: re.Match) -> str:
    """{{ll|Casting|smelted}} -> 'smelted' (last positional = display text);
    {{Item|Name=Tin bronze ingot|SmeltTemp=950}} -> 'Name: Tin bronze ingot' lines (infobox data);
    {{PT}}, {{-}} (no parameters) -> ''."""
    parts = [p.strip() for p in match.group(1).split("|")]
    params = parts[1:]
    if not params:
        return ""
    named = [p.split("=", 1) for p in params if "=" in p]
    if len(named) == len(params):  # infobox-style template: keep the data as "key: value" lines
        lines = [f"{k.strip()}: {v.strip()}" for k, v in named
                 if v.strip() and k.strip().lower() not in SKIP_INFOBOX_KEYS and not v.strip().startswith("File:")]
        return "\n" + "\n".join(lines) + "\n" if lines else ""
    positional = [p for p in params if "=" not in p]
    return positional[-1]


def clean_mediawiki_text(raw: str) -> str:
    """Convert wikitext to plain text, KEEPING the words inside templates/links
    (removing {{...}} wholesale used to leave holes like '...via , and...')."""
    # Translation markup and comments first — they sit inside template values
    text = re.sub(r'<!--.*?-->', '', raw, flags=re.DOTALL)
    text = re.sub(r'</?(?:translate|languages)\s*/?>', '', text)

    # Galleries and footnotes are file lists / source paths, not prose
    text = re.sub(r'<gallery.*?</gallery>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<ref[^>]*/>|<ref.*?</ref>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Images and categories: [[File:x.png|25px]] -> '' (otherwise '25px' ends up in the text)
    text = re.sub(r'\[\[(?:File|Image|Category):[^\]]*\]\]', '', text, flags=re.IGNORECASE)

    # Templates, innermost first (they nest)
    for _ in range(5):
        new = re.sub(r'\{\{([^{}]*)\}\}', _render_template, text)
        if new == text:
            break
        text = new

    # Links: [[Page|text]] -> text, [[Page]] -> Page
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    # External links: [http... text] -> text
    text = re.sub(r'\[https?://\S+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://\S+\]', '', text)

    # Headings == X == -> X ; bold/italics
    text = re.sub(r'={2,}\s*(.*?)\s*={2,}', r'\1', text)
    text = re.sub(r"'''''(.*?)'''''", r'\1', text)
    text = re.sub(r"'''(.*?)'''", r'\1', text)
    text = re.sub(r"''(.*?)''", r'\1', text)

    # Tables: keep cell contents, drop the syntax (alloy ratios live in tables!)
    text = re.sub(r'^\s*\{\|.*$', '', text, flags=re.MULTILINE)   # {| class="wikitable"
    text = text.replace('|}', '')
    text = re.sub(r'^\s*\|\+.*$', '', text, flags=re.MULTILINE)   # |+ caption
    text = re.sub(r'^\s*\|-.*$', '', text, flags=re.MULTILINE)    # |- row separator
    text = text.replace('||', ' | ').replace('!!', ' | ')
    text = re.sub(r'^\s*[!|]\s*', '', text, flags=re.MULTILINE)   # cell marker ! or |

    # HTML, magic words, whitespace
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'__\w+__', '', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def create_chunks(text: str, title: str, max_words: int = 150, overlap_words: int = 30) -> list[str]:
    """Split into overlapping word windows; each chunk starts with the page title so a
    chunk from the middle of 'Steel' still says what it is about."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(f"Wiki page: {title.replace('_', ' ')}\n" + " ".join(words[i:i + max_words]))
        i += max_words - overlap_words
    return chunks


def is_redirect(wikitext: str) -> bool:
    return wikitext.lstrip().upper().startswith("#REDIRECT")


def is_english_title(title: str) -> bool:
    """Heuristic: keep English pages, skip translations — titles with non-ASCII characters
    (e.g. Cyrillic) or language suffixes like '/ru', '/de', '/zh-hans'."""
    if not title.isascii():
        return False
    return not re.search(r'/[a-z]{2,3}(-[a-z]+)?$', title)


def _api_get(params: dict, timeout: float = 30) -> dict:
    url = WIKI_API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def fetch_wiki_pages(titles: list[str]) -> dict[str, str]:
    """Return {title: wikitext} for up to BATCH titles; missing pages are left out."""
    data = _api_get({
        "action": "query", "prop": "revisions", "rvprop": "content",
        "rvslots": "main", "titles": "|".join(titles),
    })
    pages = {}
    for page in data.get("query", {}).get("pages", {}).values():
        revisions = page.get("revisions")
        if revisions:
            pages[page["title"]] = revisions[0]["slots"]["main"].get("*", "")
    return pages


def list_all_wiki_pages() -> list[str]:
    """Titles of ALL non-redirect articles in the wiki's main namespace (0)."""
    titles: list[str] = []
    cont: dict[str, str] = {}
    while True:
        try:
            data = _api_get({"action": "query", "list": "allpages", "apnamespace": "0",
                             "apfilterredir": "nonredirects", "aplimit": "500", **cont})
        except Exception as e:
            print(f"  [allpages error] {e}")
            break
        titles += [p["title"] for p in data.get("query", {}).get("allpages", [])]
        if "continue" not in data:
            break
        cont = {"apcontinue": data["continue"]["apcontinue"]}
        time.sleep(0.2)
    return titles


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="every article, not just PAGES")
    parser.add_argument("--en", action="store_true", help="with --all: English pages only")
    parser.add_argument("--reset", action="store_true", help="drop and recreate the collection first")
    args = parser.parse_args()

    collection = kb.open_collection(create=True, reset=args.reset)
    print(f"[*] Collection '{config.COLLECTION_NAME}' ({config.EMBED_MODEL}), {collection.count()} chunks")

    if args.all:
        titles = list_all_wiki_pages()
        if args.en:
            before = len(titles)
            titles = [t for t in titles if is_english_title(t)]
            print(f"[*] --en filter: {before} -> {len(titles)} English pages")
        print(f"[*] --all: {len(titles)} pages to ingest")
    else:
        titles = [t.replace("_", " ") for t in PAGES]
        print(f"[*] Curated list: {len(titles)} pages (use --all for the whole wiki)")

    ok_pages = total_chunks = failed_chunks = 0
    for i in range(0, len(titles), BATCH):
        batch = titles[i:i + BATCH]
        try:
            pages = fetch_wiki_pages(batch)
        except Exception as e:
            print(f"[err]  fetch {batch[0]}..{batch[-1]}: {e}")
            continue

        ids, texts, metas = [], [], []
        for title, raw in pages.items():
            if not raw.strip() or is_redirect(raw):
                continue
            chunks = create_chunks(clean_mediawiki_text(raw), title)
            ids += [f"wiki:{title}:{n}" for n in range(len(chunks))]
            texts += chunks
            url = WIKI_PAGE_URL + urllib.parse.quote(title.replace(" ", "_"))
            metas += [{"source": f"wiki/{title}", "url": url, "type": "wiki"}] * len(chunks)
            ok_pages += 1

        missing = len(batch) - len(pages)
        sources = sorted({m["source"] for m in metas})
        if sources:  # drop old chunks of these pages, so a page that got shorter leaves no stale tail
            collection.delete(where={"source": {"$in": sources}})
        stored, failed = kb.add_documents(collection, ids, texts, metas)
        total_chunks += stored
        failed_chunks += failed
        print(f"[ok]   pages {i + 1}-{i + len(batch)}: {len(pages)} found"
              f"{f', {missing} missing' if missing else ''}, {stored} chunks")
        time.sleep(0.5)  # be gentle with the wiki

    print(f"\n[DONE] {ok_pages}/{len(titles)} pages, {total_chunks} chunks stored, "
          f"{failed_chunks} failed to embed. Collection: {collection.count()} chunks")


if __name__ == "__main__":
    main()
