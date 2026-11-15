# vs-story-ai — RAG chatbot for a Vintage Story server

An in-game AI assistant for a multiplayer [Vintage Story](https://www.vintagestory.at/) server.
Players ask questions in chat (in Polish); the bot answers in-game, grounded in the
game's own data and the wiki, and reacts to world events (joins, deaths, storms).

Runs 24/7 on a small home server as a systemd service.

## How it works

```
 docker logs -f  ──►  watchdog.py  ──►  toxicity filter
                          │
                          ├─► translate PL → EN (LLM, flash-lite)
                          ├─► ChromaDB retrieval  (nomic-embed-text, distance threshold)
                          ├─► generation: Gemini 2.5 Flash  │  fallback: local Ollama (gemma3:12b)
                          ├─► [CMD: ...] parsing → whitelisted commands only (time, weather)
                          └─► RCON  ──►  /announce in game
```

**Knowledge base (ingest)**
- `ingest/game_assets.py` — reads items / blocks / recipes JSON straight from the running
  game container (`docker exec`), strips visual-only fields, chunks and embeds them.
- `ingest/wiki.py` — fetches curated wiki pages, cleans MediaWiki markup, chunks and embeds.

**Runtime (`bot/watchdog.py`)**
- Tails the server log; player messages and events go through a queue.
- RAG with a calibrated relevance threshold — if nothing relevant is retrieved, the bot
  says so instead of guessing (this fixed "Minecraft-style" hallucinations).
- RCON output is flushed from the main thread (fixes `signal only works in main thread`).
- An autonomous loop posts short flavour messages every few minutes; event reactions
  have per-type cooldowns.

**Extras**
- `mods/eventlogger/` — small C# server mod that logs deaths and entity kills so the bot can react to them.
- `train/` — synthetic data generation and a QLoRA (Unsloth) fine-tuning script for
  Qwen2.5-7B, exported to GGUF for Ollama. Experimental.
- `docs/` — a seven-chapter write-up (in Polish): architecture, RAG, ingest, bot, LLM, improvements.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in VS_RCON_PASSWORD (and GEMINI_API_KEY, optional)

python ingest/game_assets.py  # populate chroma_db/
python ingest/wiki.py --all
python bot/watchdog.py
```

Production: `deploy/vsbot.service` (systemd, reads secrets from `.env`).

Requires: Docker container running the Vintage Story server with RCON enabled, Ollama with
`nomic-embed-text` for embeddings, and either a Gemini API key or a local Ollama model for generation.

## Tools

| Script | Purpose |
|---|---|
| `tools/console.py` | interactive RCON console |
| `tools/dbcheck.py` | inspect vector DB contents |
| `tools/ollama_test.py` | Ollama connectivity + DB state |
| `tests/test_translate.py` | PL→EN translation + RAG query |
| `tests/test_prompt_db.py` | direct vector DB queries |

## Stack

Python · ChromaDB · Ollama · Gemini API · RCON · Docker · systemd · C# (game mod) · Unsloth/QLoRA
