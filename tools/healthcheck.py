"""Pre-flight check for the bot: embeddings, knowledge base, RCON.

Usage:
    python -m tools.healthcheck

Exit code 0 when everything the bot needs is reachable, 1 otherwise.
"""

import sys

from vsai import config, kb
from vsai.embeddings import embed
from vsai.rcon import RconClient


def check(label, fn):
    try:
        detail = fn()
        print(f"[ok]   {label}: {detail}")
        return True
    except Exception as e:
        print(f"[FAIL] {label}: {e}")
        return False


def _rcon():
    if not config.RCON_PASSWORD:
        raise RuntimeError("VS_RCON_PASSWORD not set")
    RconClient(config.RCON_HOST, config.RCON_PORT, config.RCON_PASSWORD).run()
    return "authenticated"


def main():
    results = [
        check(f"Embeddings ({config.EMBED_MODEL} @ {config.EMBED_HOST})",
              lambda: f"{len(embed(['test'])[0])} dimensions"),
        check(f"Knowledge base ('{config.COLLECTION_NAME}')",
              lambda: f"{kb.open_collection().count()} chunks"),
        check(f"RCON ({config.RCON_HOST}:{config.RCON_PORT})", _rcon),
    ]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
