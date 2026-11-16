# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

In-game AI assistant for a Vintage Story multiplayer server. It tails the game server's
Docker logs, answers players who write `AI ...` in chat (RAG over game assets + wiki in
ChromaDB, generation via Gemini or Ollama), reacts to world events, and talks back via RCON.

Language rule: everything in the repo (code, comments, logs, docs, commits) is English.
Only what the bot says in game is Polish — prompts are English and instruct "reply in Polish".

## Commands (run from the repo root)

```bash
pip install -r requirements-dev.txt     # runtime + pytest + ruff
pytest                                  # unit tests (tests/unit)
ruff check .                            # lint (CI runs both)

python -m bot.watchdog                  # the bot
python -m ingest.game_assets --reset [--assets-dir PATH | --container NAME] [--dry-run]
python -m ingest.wiki [--all] [--en] [--reset]
python -m tools.healthcheck             # embeddings + KB + RCON
python -m tools.dbcheck "question" ...  # distances, for calibrating RAG_MAX_DISTANCE
python -m tools.console                 # RCON console
```

## Layout

```
vsai/      shared: config (.env), embeddings (Ollama /api/embed), kb (ChromaDB), rcon (client)
bot/       watchdog.py — log loop, trigger/cooldowns, RAG, LLM calls, event reactions
ingest/    game_assets.py (container or local assets dir), wiki.py (MediaWiki API)
tools/     healthcheck, dbcheck, console
tests/unit pytest; test_kb.py uses a real temporary ChromaDB with a fake embedder
config/    Vintage Story serverconfig.json (password placeholder; RCON on 42421)
mods/      EventLogger C# server mod (logs [WATCHDOG] PLAYER_DEATH / ENTITY_KILL lines)
deploy/    systemd unit
docs/      HTML write-up (architecture, RAG, ingest, bot, LLM, persona, roadmap)
archive/   retired code (fine-tuning experiment) — not maintained, excluded from lint
```

## Things to know

- Settings come from `.env` (see `.env.example`); `vsai/config.py` loads it on import.
  The bot checks `VS_RCON_PASSWORD` in `main()`, not at import.
- Knowledge base: collection `vintage_story_kb`, cosine distance, embeddings computed by
  `vsai.embeddings` and passed explicitly (no Chroma EmbeddingFunction). The collection
  metadata stores `embed_model`; `kb.open_collection()` refuses a different model.
  Changing `OLLAMA_EMBED_MODEL` means rebuilding with `--reset`.
- Embedding failures raise `EmbeddingError`; never store or query zero vectors.
- RCON: `vsai/rcon.py` (socket timeouts, lock) — safe from any thread. Don't reintroduce
  `mcrcon`; its signal-based timeout only works in the main thread.
- `[CMD: ...]` from the LLM runs only if it exactly equals a whitelisted command.
- Player-facing fixed texts (`HELP_TEXT`, `JOIN_HINT`, citation labels) are Polish constants in
  `bot/watchdog.py`. Behaviour switches live in `.env`: `EVENT_REACTIONS`, `TIPS_*`, `CITE_SOURCES`, `JOIN_HINT`.
- Ingest IDs include the asset's relative path; re-ingesting replaces a source's chunks.
