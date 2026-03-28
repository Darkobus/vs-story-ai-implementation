# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI chatbot for a **Vintage Story multiplayer game server**. It monitors server chat via Docker logs, responds to players using a local LLM (Ollama), and integrates a vector database (ChromaDB) for game knowledge retrieval (RAG).

## Running the Bot

```bash
# Main bot
python bot/watchdog.py

# Ingest game assets into vector DB
python ingest/game_assets.py [--container NAME] [--dry-run]

# Ingest wiki pages
python ingest/wiki.py

# Test utilities
python tests/test_translate.py    # Test Polish→English translation + RAG query
python tests/test_prompt_db.py    # Test vector DB queries directly
python tools/dbcheck.py           # Inspect vector DB contents
python tools/ollama_test.py       # Test Ollama connectivity + DB state
python tools/console.py           # Interactive RCON console
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
bot/            # Main bot process
ingest/         # Data ingestion scripts (run once to populate chroma_db/)
tools/          # Dev utilities (RCON console, DB inspection, connectivity tests)
tests/          # Manual test scripts
config/         # Vintage Story server config JSON
docs/           # Polish guide (poradnik.html)
archive/        # Superseded versions (watchdog_v1/v2, ingest_wiki v1/v2, etc.)
chroma_db/      # ChromaDB vector database (gitignore candidate)
wiki_db/        # Legacy wiki DB directory
```

## Architecture

### Data Flow

**Ingestion phase** (run once or to refresh knowledge):
- `ingest/game_assets.py` — runs `docker exec` on the game container, reads game asset JSON files (recipes, items, blocks), strips visual-only fields, chunks text, and upserts into ChromaDB (`chroma_db/`)
- `ingest/wiki.py` — reads local wiki text files from a directory, cleans MediaWiki markup, chunks and inserts into ChromaDB

**Runtime phase** (`bot/watchdog.py`):
1. Tails Docker container logs via `docker logs --tail 0 -f <CONTAINER_NAME>`
2. Regex-parses `[Server Chat]` lines to extract player + message
3. Applies toxicity filter (blocked words list)
4. Translates Polish player queries to English using Ollama
5. Queries ChromaDB (`vintage_story_wiki` collection) for relevant context
6. Calls Ollama (`gemma3:27b`) with player history + RAG context
7. Extracts `[CMD: ...]` tokens from LLM output, executes whitelisted commands (`time`, `weather`) via RCON
8. Sends response back via RCON (`/announce`)
9. Background thread sends autonomous "dark flavor" messages every 5–15 minutes

### Key Configuration (top of `bot/watchdog.py`)

```python
CONTAINER_NAME = "vs-ai-server"
PC_IP = "192.168.0.24"
OLLAMA_API_URL = "http://192.168.0.24:11434/api/generate"
RCON_HOST = "127.0.0.1"
RCON_PORT = 42421
RCON_PASSWORD = "klawisz"
ALLOWED_COMMANDS = ["time", "weather"]
```

### Path Convention

All scripts use `Path(__file__).resolve().parent.parent / "chroma_db"` to locate the vector DB relative to the project root — so they work correctly regardless of the current working directory.
