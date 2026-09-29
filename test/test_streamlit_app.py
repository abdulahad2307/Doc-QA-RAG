"""Phase 4 tests — Streamlit UI (runs without API key or model downloads)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


class FakePipeline:
    """Stands in for RAGPipeline so no LLM/embedding model is loaded."""

    def __init__(self, result=None, error=None):
        self.result = result or {
            "answer": "RAG combines retrieval with generation [Source: rag.txt, chunk 0]",
            "sources": [
                {"source": "rag.txt", "chunk_index": 0, "content_preview": "RAG combines..."},
            ],
        }
        self.error = error
        self.questions = []

    def query(self, question, k=5):
        self.questions.append(question)
        if self.error:
            raise self.error
        return self.result

    def ingest(self, file_paths, chunk_size=1000):
        return 0


def make_app(pipeline):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["pipeline"] = pipeline
    return at.run()


def test_app_renders_without_errors():
    at = make_app(FakePipeline())
    assert not at.exception
    assert at.title[0].value == "📄 Intelligent Document Q&A"
    assert len(at.chat_input) == 1


def test_question_shows_answer_and_sources():
    pipeline = FakePipeline()
    at = make_app(pipeline)

    at.chat_input[0].set_value("What is RAG?").run()

    assert not at.exception
    assert pipeline.questions == ["What is RAG?"]
    markdown = [m.value for m in at.markdown]
    assert "What is RAG?" in markdown
    assert pipeline.result["answer"] in markdown
    assert at.expander[0].label == "📎 View Sources"
    assert "**rag.txt** (Chunk 0)" in [m.value for m in at.expander[0].markdown]


def test_chat_history_persists_across_questions():
    at = make_app(FakePipeline())
    at.chat_input[0].set_value("First?").run()
    at.chat_input[0].set_value("Second?").run()

    history = at.session_state["chat_history"]
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert history[2]["content"] == "Second?"


def test_no_sources_hides_expander():
    at = make_app(FakePipeline(result={"answer": "No documents loaded.", "sources": []}))
    at.chat_input[0].set_value("Anything?").run()

    assert not at.exception
    assert len(at.expander) == 0


def test_query_error_is_shown_not_raised():
    at = make_app(FakePipeline(error=RuntimeError("API down")))
    at.chat_input[0].set_value("What is RAG?").run()

    assert not at.exception
    assert any("API down" in m.value for m in at.markdown)
