# vs-story-ai — RAG chatbot for a Vintage Story server

An in-game AI assistant for a multiplayer [Vintage Story](https://www.vintagestory.at/) server.
Players ask questions in chat (in Polish); the bot answers in-game, grounded in the
game's own data and the wiki, and reacts to world events (joins, deaths, storms).

Runs 24/7 on a small home server as a systemd service.

## How it works

```
 docker logs -f  ──►  bot/watchdog.py  ──►  trigger ("AI ...") + toxicity filter + per-player cooldown
                          │
                          ├─► ChromaDB retrieval  (bge-m3 multilingual embeddings, cosine threshold)
                          ├─► generation: Gemini 2.5 Flash  │  or local Ollama (gemma3:12b)
                          ├─► [CMD: ...] parsing → exact whitelisted commands only (time, weather)
                          └─► RCON  ──►  /announce in game
```

**Knowledge base (ingest)**
- `ingest/game_assets.py` — reads items / blocks / recipes JSON from the running game
  container (or a local copy of the game's `assets/`), strips visual-only fields, turns
  them into short text records, embeds them.
- `ingest/wiki.py` — fetches wiki pages (curated list or `--all`), cleans MediaWiki markup
  (keeps infobox data and table cells), chunks and embeds them.

**Runtime (`bot/watchdog.py`)**
- Tails the server log; questions and events are handled in a thread pool, so the log
  loop never waits on the LLM.
- Multilingual embeddings match Polish questions against English wiki text directly — no
  translation call. If nothing passes the relevance threshold, the bot says it doesn't know
  instead of guessing (this fixed "Minecraft-style" hallucinations).
- Remembers the last few question/answer pairs per player.
- Cites its sources: grounded answers end with e.g. `(źródło: wiki „Bronze”)`.
- `AI pomoc` / `AI help` prints a fixed how-to-ask message (no LLM call); join greetings
  end with a one-line hint pointing to it.
- Posts a gameplay tip every 30–60 minutes, built from a random knowledge-base chunk.
- Event reactions are switchable (`EVENT_REACTIONS`, default: joins, leaves, storms;
  deaths and kills are off) and have per-event cooldowns.

**Shared code (`vsai/`)** — settings from `.env`, Ollama embeddings, knowledge-base access
(the collection records which embedding model built it and refuses a mismatched one),
and a thread-safe RCON client.

**Extras**
- `mods/eventlogger/` — small C# server mod that logs deaths and entity kills so the bot can react to them.
- `docs/` — a seven-chapter HTML write-up (open `docs/index.html`): architecture, RAG, ingest, bot, LLM, persona, roadmap.
- `archive/train/` — an earlier fine-tuning experiment (QLoRA on Qwen2.5-7B); not used.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in VS_RCON_PASSWORD (and GEMINI_API_KEY, optional)
ollama pull bge-m3            # embedding model

# Build the knowledge base (first step resets the collection)
python -m ingest.game_assets --reset
python -m ingest.wiki --all --en

python -m tools.healthcheck   # embeddings + knowledge base + RCON reachable?
python -m bot.watchdog
```

All commands run from the repository root. Production: `deploy/vsbot.service`
(systemd, reads secrets from `.env`).

Requires: Docker container running the Vintage Story server with RCON enabled, Ollama with
`bge-m3` for embeddings, and either a Gemini API key or a local Ollama model for generation.

Player-facing behaviour (event reactions, tips, citations, the join hint) is set in `.env`;
see the bottom of `.env.example`.

### Building the knowledge base on another machine

Embedding ~10k chunks with bge-m3 on a CPU takes a while. You can build on a PC with a GPU
and copy the result:

```bash
# on the PC (Ollama with bge-m3 running locally, same chromadb version as the server)
python -m ingest.game_assets --reset --assets-dir "/path/to/Vintagestory/assets"
python -m ingest.wiki --all --en
# then copy chroma_db/ to the server's repository root
```

### Calibrating the relevance threshold

`RAG_MAX_DISTANCE` (cosine distance, default 0.5) decides which chunks count as relevant.
Run `python -m tools.dbcheck "question 1" "question 2" ...` with questions the bot should
answer and some off-topic ones, and set the threshold between the two groups.

## Tools

| Command | Purpose |
|---|---|
| `python -m tools.healthcheck` | pre-flight check: embeddings, knowledge base, RCON |
| `python -m tools.dbcheck [questions...]` | query the knowledge base, print distances and sources |
| `python -m tools.console` | interactive RCON console |
| `pytest` | unit tests (`tests/unit`) |

## Stack

Python · ChromaDB · Ollama (bge-m3) · Gemini API · RCON · Docker · systemd · C# (game mod)
