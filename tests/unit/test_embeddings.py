import io
import json
import urllib.error

import pytest

from vsai import embeddings
from vsai.embeddings import EmbeddingError, embed


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_urlopen(responses, calls):
    def _urlopen(req, timeout):
        calls.append(json.loads(req.data))
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return FakeResponse(json.dumps(result).encode())
    return _urlopen


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(embeddings.time, "sleep", lambda s: None)


def test_batch_request_and_order(monkeypatch):
    calls = []
    monkeypatch.setattr(embeddings.urllib.request, "urlopen",
                        fake_urlopen([{"embeddings": [[1.0], [2.0]]}], calls))
    assert embed(["a", "b"], model="m") == [[1.0], [2.0]]
    assert calls == [{"model": "m", "input": ["a", "b"], "keep_alive": "30m"}]


def test_empty_input_makes_no_request():
    assert embed([]) == []


def test_retries_then_raises(monkeypatch, no_sleep):
    calls = []
    monkeypatch.setattr(embeddings.urllib.request, "urlopen",
                        fake_urlopen([ConnectionRefusedError("down"), ConnectionRefusedError("down")], calls))
    with pytest.raises(EmbeddingError, match="down"):
        embed(["a"], retries=1)
    assert len(calls) == 2


def test_wrong_count_is_an_error(monkeypatch, no_sleep):
    monkeypatch.setattr(embeddings.urllib.request, "urlopen",
                        fake_urlopen([{"embeddings": []}], []))
    with pytest.raises(EmbeddingError, match="expected 1"):
        embed(["a"], retries=0)


def test_missing_model_hints_ollama_pull(monkeypatch):
    err = urllib.error.HTTPError("u", 404, "nf", {}, io.BytesIO(b'{"error":"model not found"}'))
    monkeypatch.setattr(embeddings.urllib.request, "urlopen", fake_urlopen([err], []))
    with pytest.raises(EmbeddingError, match="ollama pull bge-m3"):
        embed(["a"], model="bge-m3")
