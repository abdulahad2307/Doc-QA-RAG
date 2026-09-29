"""Phase 1 tests — text splitting."""
from langchain_core.documents import Document

from src.text_splitter import split_documents


def test_split_creates_chunks():
    docs = [Document(page_content="A" * 2000, metadata={"source": "test"})]
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=100)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 500 for c in chunks)


def test_split_preserves_metadata():
    docs = [Document(page_content="Hello world", metadata={"source": "test.pdf"})]
    chunks = split_documents(docs, chunk_size=500)
    assert chunks[0].metadata["source"] == "test.pdf"
    assert chunks[0].metadata["chunk_index"] == 0


def test_chunk_index_is_sequential_across_documents():
    docs = [
        Document(page_content="word " * 300, metadata={"source": "a.txt"}),
        Document(page_content="word " * 300, metadata={"source": "b.txt"}),
    ]
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert {c.metadata["source"] for c in chunks} == {"a.txt", "b.txt"}


def test_chunks_overlap():
    text = " ".join(f"w{i}" for i in range(400))
    chunks = split_documents([Document(page_content=text)], chunk_size=300, chunk_overlap=100)
    first_tail = chunks[0].page_content.split()[-3:]
    assert all(word in chunks[1].page_content.split() for word in first_tail)


def test_empty_input_returns_no_chunks():
    assert split_documents([]) == []
