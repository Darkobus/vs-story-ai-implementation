"""Embeddings from Ollama's batch endpoint (/api/embed).

Failures raise EmbeddingError — never fall back to zero vectors: a zero vector stored
in the DB is junk, and a zero query vector returns arbitrary chunks.
"""

import json
import time
import urllib.error
import urllib.request

from vsai import config


class EmbeddingError(Exception):
    pass


def embed(texts: list[str], *, model: str | None = None, url: str | None = None,
          timeout: float = 120, retries: int = 1) -> list[list[float]]:
    """Return one vector per text, in order."""
    if not texts:
        return []
    model = model or config.EMBED_MODEL
    payload = json.dumps({"model": model, "input": texts, "keep_alive": "30m"}).encode("utf-8")
    last_error = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url or config.EMBED_URL, data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                vectors = json.loads(response.read().decode("utf-8")).get("embeddings") or []
            if len(vectors) != len(texts):
                raise EmbeddingError(f"expected {len(texts)} embeddings, got {len(vectors)}")
            return vectors
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:200]
            if e.code == 404:  # model missing — retrying will not help
                raise EmbeddingError(f"{body} (run: ollama pull {model})") from e
            last_error = EmbeddingError(f"HTTP {e.code}: {body}")
        except EmbeddingError as e:
            last_error = e
        except Exception as e:  # connection refused, timeout, bad JSON
            last_error = EmbeddingError(str(e))
        if attempt < retries:
            time.sleep(1 + attempt)
    raise last_error
