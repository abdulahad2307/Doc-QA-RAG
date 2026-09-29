"""Phase 1 tests — document loading."""
import pytest

from src.document_loader import load_document, load_multiple_documents


def test_load_txt_sets_metadata(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("AI is transforming healthcare.")

    docs = load_document(str(path))

    assert len(docs) == 1
    assert "healthcare" in docs[0].page_content
    assert docs[0].metadata["source"] == "notes.txt"
    assert docs[0].metadata["file_type"] == ".txt"


def test_load_md(tmp_path):
    path = tmp_path / "readme.md"
    path.write_text("# Title\n\nSome markdown content.")

    docs = load_document(str(path))

    assert docs[0].metadata["file_type"] == ".md"
    assert "markdown content" in docs[0].page_content


def test_extension_is_case_insensitive(tmp_path):
    path = tmp_path / "UPPER.TXT"
    path.write_text("content")

    assert load_document(str(path))[0].metadata["source"] == "UPPER.TXT"


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(str(path))


def test_load_multiple_skips_bad_files(tmp_path):
    good = tmp_path / "good.txt"
    good.write_text("valid content")
    bad = tmp_path / "bad.csv"
    bad.write_text("x")
    missing = tmp_path / "missing.txt"

    docs = load_multiple_documents([str(good), str(bad), str(missing)])

    assert [d.metadata["source"] for d in docs] == ["good.txt"]
