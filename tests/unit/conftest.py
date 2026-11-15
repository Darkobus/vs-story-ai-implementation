"""Import bot/watchdog.py without ChromaDB, RCON or a real .env.

watchdog.py connects to its vector DB and reads secrets at import time, so the
unit tests stub those modules and set a dummy password first.
"""
import os
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("VS_RCON_PASSWORD", "test-only")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bot"))


def _stub_chromadb():
    chromadb = types.ModuleType("chromadb")

    class EmbeddingFunction:  # base class watchdog subclasses
        pass

    class PersistentClient:
        def __init__(self, *a, **kw):
            raise RuntimeError("no vector DB in unit tests")

    chromadb.EmbeddingFunction = EmbeddingFunction
    chromadb.Embeddings = list
    chromadb.PersistentClient = PersistentClient
    sys.modules["chromadb"] = chromadb


def _stub_mcrcon():
    mcrcon = types.ModuleType("mcrcon")
    mcrcon.MCRcon = object
    sys.modules["mcrcon"] = mcrcon


@pytest.fixture(scope="session")
def watchdog():
    _stub_chromadb()
    _stub_mcrcon()
    import watchdog as wd

    return wd
