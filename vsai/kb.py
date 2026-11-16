"""Knowledge base: one ChromaDB collection, cosine distance, embeddings computed by us.

Vectors are always passed explicitly (no Chroma EmbeddingFunction), so Chroma never
falls back to its own default model. The collection metadata records which embedding
model built it; opening it with a different model is refused, because the vectors
would be incomparable (or a different dimension).

chromadb is imported lazily so the bot's unit tests can import this module without it.
"""

import random
from dataclasses import dataclass

from vsai import config
from vsai.embeddings import EmbeddingError, embed

SPACE = "cosine"


class KnowledgeBaseError(Exception):
    pass


@dataclass
class Hit:
    text: str
    distance: float
    metadata: dict

    @property
    def source(self) -> str:
        return (self.metadata or {}).get("source", "?")


def _client(path=None):
    import chromadb

    return chromadb.PersistentClient(path=str(path or config.CHROMA_PATH))


def _collection_names(client) -> list[str]:
    # chromadb < 0.6 returns Collection objects, newer versions return names
    return [c if isinstance(c, str) else c.name for c in client.list_collections()]


def open_collection(*, create: bool = False, reset: bool = False, name: str | None = None,
                    model: str | None = None, path=None):
    """Open the KB collection. `create` makes it if missing; `reset` drops and recreates it."""
    name = name or config.COLLECTION_NAME
    model = model or config.EMBED_MODEL
    client = _client(path)
    exists = name in _collection_names(client)

    if reset and exists:
        client.delete_collection(name)
        exists = False
    if not exists:
        if not (create or reset):
            raise KnowledgeBaseError(
                f"collection '{name}' not found in {path or config.CHROMA_PATH} — run the ingest first")
        return client.create_collection(
            name, metadata={"hnsw:space": SPACE, "embed_model": model}, embedding_function=None)

    collection = client.get_collection(name, embedding_function=None)
    check_compatible(collection.metadata, model, name)
    return collection


def check_compatible(metadata: dict | None, model: str, name: str = "collection") -> None:
    metadata = metadata or {}
    built_with = metadata.get("embed_model")
    if built_with != model:
        raise KnowledgeBaseError(
            f"'{name}' was built with embedding model {built_with!r}, but {model!r} is configured "
            f"(OLLAMA_EMBED_MODEL). Rebuild the KB or change the setting.")
    if metadata.get("hnsw:space", "l2") != SPACE:
        raise KnowledgeBaseError(f"'{name}' uses {metadata.get('hnsw:space', 'l2')!r} distance, expected {SPACE!r}")


def add_documents(collection, ids: list[str], texts: list[str], metadatas: list[dict],
                  batch_size: int = 32, model: str | None = None) -> tuple[int, int]:
    """Embed and upsert in batches. A batch whose embedding fails is skipped (not stored
    as zeros). Returns (stored, failed) chunk counts."""
    stored = failed = 0
    for i in range(0, len(texts), batch_size):
        batch = slice(i, i + batch_size)
        try:
            vectors = embed(texts[batch], model=model)
        except EmbeddingError as e:
            print(f"  [embed error] {len(texts[batch])} chunks skipped: {e}")
            failed += len(texts[batch])
            continue
        collection.upsert(ids=ids[batch], documents=texts[batch],
                          metadatas=metadatas[batch], embeddings=vectors)
        stored += len(texts[batch])
    return stored, failed


def random_chunk(collection, rng=random) -> Hit | None:
    """One chunk picked uniformly at random (distance 0.0), or None if the collection is empty."""
    count = collection.count()
    if not count:
        return None
    result = collection.get(limit=1, offset=rng.randrange(count), include=["documents", "metadatas"])
    docs = result.get("documents") or []
    if not docs or not docs[0]:
        return None
    metas = result.get("metadatas") or [{}]
    return Hit(docs[0], 0.0, metas[0] or {})


def search(collection, query: str, n_results: int = 5, model: str | None = None) -> list[Hit]:
    """Nearest chunks to `query`, closest first. Raises EmbeddingError if Ollama is down."""
    vector = embed([query], model=model)[0]
    result = collection.query(query_embeddings=[vector], n_results=n_results,
                              include=["documents", "distances", "metadatas"])
    docs = (result.get("documents") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0] or [{}] * len(docs)
    return [Hit(d, float(dist), m or {}) for d, dist, m in zip(docs, dists, metas) if d]
