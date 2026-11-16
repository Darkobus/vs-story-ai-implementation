"""vsai.kb against a real (temporary) ChromaDB, with a fake embedding model."""
import pytest

pytest.importorskip("chromadb")

from vsai import kb  # noqa: E402
from vsai.embeddings import EmbeddingError  # noqa: E402

AXES = {"bronze": [1.0, 0.0, 0.0], "steel": [0.0, 1.0, 0.0], "storm": [0.0, 0.0, 1.0]}


def fake_embed(texts, model=None):
    vectors = []
    for t in texts:
        if "FAIL" in t:
            raise EmbeddingError("ollama down")
        vectors.append(next((v for k, v in AXES.items() if k in t.lower()), [0.5, 0.5, 0.5]))
    return vectors


@pytest.fixture
def collection(tmp_path, monkeypatch):
    monkeypatch.setattr(kb, "embed", fake_embed)
    return kb.open_collection(create=True, path=tmp_path, model="fake")


def test_missing_collection_is_an_error_unless_create(tmp_path):
    with pytest.raises(kb.KnowledgeBaseError, match="run the ingest"):
        kb.open_collection(path=tmp_path, model="fake")


def test_collection_uses_cosine_and_records_model(collection):
    assert collection.metadata["hnsw:space"] == "cosine"
    assert collection.metadata["embed_model"] == "fake"


def test_search_returns_nearest_with_cosine_distance(collection):
    kb.add_documents(collection, ["a", "b", "c"], ["Bronze alloy", "Steel making", "Storm"],
                     [{"source": "wiki/Bronze"}, {"source": "wiki/Steel"}, {"source": "wiki/Storm"}])
    hits = kb.search(collection, "how to make bronze", n_results=2)
    assert hits[0].source == "wiki/Bronze"
    assert hits[0].distance == pytest.approx(0.0, abs=1e-6)
    assert hits[1].distance == pytest.approx(1.0, abs=1e-6)  # orthogonal -> cosine distance 1


def test_failed_batch_is_skipped_not_stored_as_zeros(collection):
    stored, failed = kb.add_documents(
        collection, ["1", "2", "3"], ["bronze", "FAIL", "steel"], [{"source": "s"}] * 3, batch_size=1)
    assert (stored, failed) == (2, 1)
    assert sorted(collection.get()["ids"]) == ["1", "3"]


def test_reopen_with_other_model_is_refused(tmp_path, collection):
    with pytest.raises(kb.KnowledgeBaseError, match="built with embedding model 'fake'"):
        kb.open_collection(path=tmp_path, model="bge-m3")


def test_reset_empties_collection(tmp_path, collection):
    kb.add_documents(collection, ["a"], ["bronze"], [{"source": "s"}])
    assert kb.open_collection(reset=True, path=tmp_path, model="fake").count() == 0


def test_legacy_l2_collection_is_refused():
    with pytest.raises(kb.KnowledgeBaseError, match="'l2' distance"):
        kb.check_compatible({"embed_model": "m"}, "m")
