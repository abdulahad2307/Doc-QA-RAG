"""Phase 3 tests — answer generation (fake LLM; live test is opt-in)."""
import os

import pytest
from dotenv import load_dotenv

from src.llm_chain import generate_answer, get_llm
from test.conftest import TEST_CHUNKS


def test_generate_answer_returns_answer_and_sources(fake_llm):
    result = generate_answer("How is AI used in healthcare?", TEST_CHUNKS[:2], fake_llm.runnable)

    assert result["answer"] == fake_llm.answer
    assert [s["source"] for s in result["sources"]] == ["healthcare.txt", "healthcare.txt"]
    assert [s["chunk_index"] for s in result["sources"]] == [0, 1]


def test_context_includes_source_tags_and_question(fake_llm):
    generate_answer("What is RAG?", TEST_CHUNKS[3:5], fake_llm.runnable)

    prompt = fake_llm.prompts[0]
    assert "[Source: rag_guide.txt, Chunk 0]" in prompt
    assert "[Source: rag_guide.txt, Chunk 1]" in prompt
    assert "What is RAG?" in prompt
    assert "ONLY on the provided context" in prompt


def test_content_preview_is_truncated(fake_llm):
    long_doc = TEST_CHUNKS[0].model_copy(update={"page_content": "x" * 500})
    result = generate_answer("q", [long_doc], fake_llm.runnable)
    assert len(result["sources"][0]["content_preview"]) == 200


def test_no_documents_still_calls_llm(fake_llm):
    result = generate_answer("Anything?", [], fake_llm.runnable)
    assert result["sources"] == []
    assert len(fake_llm.prompts) == 1


@pytest.mark.live
def test_live_answer_is_grounded():
    load_dotenv()
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key or key.startswith("your-"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    result = generate_answer("What year did the French Revolution begin?", TEST_CHUNKS, get_llm())

    assert "1789" in result["answer"]
