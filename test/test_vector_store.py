"""Phase 2/3 tests — embeddings, vector store and retrieval."""
import numpy as np

import src.vector_store as vector_store_module
from src.retriever import retrieve_documents
from test.conftest import TEST_CHUNKS


def test_embedding_dimension_and_normalization(embedding_model):
    vector = embedding_model.embed_query("medical imaging")
    assert len(vector) == 384
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-3)


def test_similar_texts_are_closer(embedding_model):
    a, b, c = embedding_model.embed_documents([
        "Deep learning for medical image classification",
        "CNNs detect disease in X-ray images",
        "The French Revolution began in 1789",
    ])
    assert np.dot(a, b) > np.dot(a, c)


def test_create_vector_store_stores_all_chunks(vector_store, persist_dir):
    assert vector_store._collection.count() == len(TEST_CHUNKS)
    assert persist_dir.exists()


def test_create_vector_store_replaces_existing(embedding_model, persist_dir):
    vector_store_module.create_vector_store(TEST_CHUNKS, embedding_model)
    store = vector_store_module.create_vector_store(TEST_CHUNKS[:2], embedding_model)
    assert store._collection.count() == 2


def test_load_vector_store_missing_returns_none(embedding_model, persist_dir):
    assert vector_store_module.load_vector_store(embedding_model) is None


def test_load_vector_store_reads_persisted_data(vector_store, embedding_model):
    loaded = vector_store_module.load_vector_store(embedding_model)
    assert loaded._collection.count() == len(TEST_CHUNKS)


def test_retrieve_returns_relevant_first(vector_store):
    results = retrieve_documents(vector_store, "medical image classification", k=3)
    assert 0 < len(results) <= 3
    assert results[0].metadata["source"] == "healthcare.txt"


def test_retrieve_keeps_metadata(vector_store):
    for doc in retrieve_documents(vector_store, "vector database embeddings", k=3):
        assert isinstance(doc.metadata["source"], str)
        assert isinstance(doc.metadata["chunk_index"], int)


def test_retrieve_respects_k(vector_store):
    assert len(retrieve_documents(vector_store, "machine learning", k=1)) <= 1
    assert len(retrieve_documents(vector_store, "machine learning", k=5)) <= 5
