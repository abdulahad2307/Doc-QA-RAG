"""Pytest configuration.

The Phase 1-3 files below are standalone print scripts (run them with
`python test/<file>.py`), not pytest tests. They execute on import, so
keep pytest from collecting them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

collect_ignore = [
    "test_doc_ings.py",
    "test_embedding.py",
    "test_retrieval_llm.py",
]
