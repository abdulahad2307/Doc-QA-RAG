"""End-to-end pipeline tests (real embeddings, fake LLM)."""
import pytest

import src.pipeline as pipeline_module
from src.pipeline import RAGPipeline


@pytest.fixture
def pipeline(embedding_model, persist_dir, fake_llm, monkeypatch):
    monkeypatch.setattr(pipeline_module, "get_embedding_model", lambda name: embedding_model)
    monkeypatch.setattr(pipeline_module, "get_llm", lambda: fake_llm.runnable)
    return RAGPipeline()


def test_query_without_documents(pipeline):
    assert pipeline.query("Anything?") == {"answer": "No documents loaded.", "sources": []}


def test_ingest_then_query(pipeline, fake_llm, tmp_path):
    doc = tmp_path / "healthcare.txt"
    doc.write_text(
        "AI improves diagnostics in hospitals.\n\n"
        "Federated learning trains models across institutions without sharing patient data."
    )

    n_chunks = pipeline.ingest([str(doc)])
    result = pipeline.query("How can hospitals train models without sharing data?")

    assert n_chunks >= 1
    assert result["answer"] == fake_llm.answer
    assert result["sources"][0]["source"] == "healthcare.txt"
    assert "Federated learning" in fake_llm.prompts[0]


def test_query_loads_persisted_store(pipeline, embedding_model, fake_llm, tmp_path):
    doc = tmp_path / "notes.txt"
    doc.write_text("ChromaDB stores embeddings on disk.")
    pipeline.ingest([str(doc)])

    fresh = RAGPipeline()
    result = fresh.query("Where are embeddings stored?")

    assert result["sources"][0]["source"] == "notes.txt"


def test_reingest_replaces_previous_documents(pipeline, tmp_path):
    """Clicking 'Process Documents' twice in the app must not crash."""
    first = tmp_path / "first.txt"
    first.write_text("Old content about cooking recipes.")
    second = tmp_path / "second.txt"
    second.write_text("New content about vector databases.")

    pipeline.ingest([str(first)])
    pipeline.ingest([str(second)])
    result = pipeline.query("Tell me about vector databases", k=5)

    assert {s["source"] for s in result["sources"]} == {"second.txt"}
