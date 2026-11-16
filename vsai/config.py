"""Settings shared by every script. Values come from the environment / .env.

Importing this module loads `<repo>/.env` (without overriding variables that are
already set, e.g. by systemd's EnvironmentFile).
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Load simple KEY=VALUE lines into os.environ (no dependency, no overrides)."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


load_dotenv()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, "") or default


# --- Knowledge base (ChromaDB) ---
CHROMA_PATH = Path(_env("VS_CHROMA_PATH", str(ROOT / "chroma_db")))
# New collection name for the cosine/multilingual rebuild; the old "vintage_story_wiki"
# (L2 + nomic-embed-text) is left untouched so you can roll back.
COLLECTION_NAME = _env("VS_COLLECTION", "vintage_story_kb")

# --- Embeddings (local Ollama on the server, CPU is fine) ---
EMBED_HOST = _env("OLLAMA_EMBED_HOST", "127.0.0.1")
EMBED_URL = f"http://{EMBED_HOST}:11434/api/embed"
# Multilingual model: Polish questions match English wiki/asset text directly,
# so no translation call is needed. Changing it requires rebuilding the collection.
EMBED_MODEL = _env("OLLAMA_EMBED_MODEL", "bge-m3")

# --- RCON ---
RCON_HOST = _env("VS_RCON_HOST", "127.0.0.1")
RCON_PORT = int(_env("VS_RCON_PORT", "42421"))
RCON_PASSWORD = os.environ.get("VS_RCON_PASSWORD", "")

# --- Game server ---
CONTAINER_NAME = _env("VS_CONTAINER", "vs-ai-server")
