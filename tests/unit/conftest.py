"""Import bot/watchdog.py without ChromaDB installed.

watchdog.py subclasses chromadb.EmbeddingFunction at import time, so the unit tests
stub that module. The vector DB itself is only opened in main()/init_knowledge_base().
"""
import sys
import types
from pathlib import Path

import pytest

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


@pytest.fixture(scope="session")
def watchdog():
    _stub_chromadb()
    import watchdog as wd

    return wd
