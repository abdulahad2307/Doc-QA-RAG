"""Shared pytest fixtures."""
import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.vector_store as vector_store_module  # noqa: E402
from src.embeddings import get_embedding_model  # noqa: E402

TEST_CHUNKS = [
    Document(page_content="Artificial intelligence is transforming healthcare through improved diagnostics, personalized treatment plans, and drug discovery.", metadata={"source": "healthcare.txt", "chunk_index": 0}),
    Document(page_content="Convolutional Neural Networks and Vision Transformers achieve state-of-the-art accuracy in medical image classification for diabetic retinopathy detection.", metadata={"source": "healthcare.txt", "chunk_index": 1}),
    Document(page_content="Named Entity Recognition systems extract medications, conditions, and procedures from unstructured clinical notes.", metadata={"source": "healthcare.txt", "chunk_index": 2}),
    Document(page_content="Retrieval-Augmented Generation combines semantic search over document chunks with language models to produce grounded, cited answers.", metadata={"source": "rag_guide.txt", "chunk_index": 0}),
    Document(page_content="ChromaDB and FAISS are popular vector databases that store embeddings for fast approximate nearest neighbor search.", metadata={"source": "rag_guide.txt", "chunk_index": 1}),
    Document(page_content="Docker containers package applications with dependencies. Kubernetes orchestrates container deployments with scaling and health checks.", metadata={"source": "devops.txt", "chunk_index": 0}),
    Document(page_content="MLflow tracks experiments including parameters, metrics, and model artifacts. The model registry manages staging and production transitions.", metadata={"source": "mlops.txt", "chunk_index": 0}),
    Document(page_content="The French Revolution began in 1789 and fundamentally transformed French political and social structures over the following decade.", metadata={"source": "history.txt", "chunk_index": 0}),
]


@pytest.fixture(scope="session")
def embedding_model():
    """Load the embedding model once for the whole test run."""
    return get_embedding_model("all-MiniLM-L6-v2")


@pytest.fixture
def persist_dir(tmp_path, monkeypatch):
    """Point the vector store at a throwaway directory."""
    path = tmp_path / "chroma_db"
    monkeypatch.setattr(vector_store_module, "PERSIST_DIR", str(path))
    return path


@pytest.fixture
def vector_store(embedding_model, persist_dir):
    return vector_store_module.create_vector_store(TEST_CHUNKS, embedding_model)


class FakeLLM:
    """Records the prompt it receives and returns a canned answer."""

    def __init__(self, answer="Canned answer [Source: healthcare.txt, chunk 0]"):
        self.answer = answer
        self.prompts = []
        self.runnable = RunnableLambda(self._respond)

    def _respond(self, prompt_value):
        self.prompts.append(prompt_value.to_string())
        return AIMessage(content=self.answer)


@pytest.fixture
def fake_llm():
    return FakeLLM()
