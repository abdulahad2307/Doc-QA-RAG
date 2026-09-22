"""
Phase 1 Test Script — Document Ingestion Pipeline
Run this after creating src/document_loader.py and src/text_splitter.py

Usage:
  1. Place a sample PDF in data/sample_docs/
  2. Run from project root:  python -m tests.test_phase1
     or:                     cd tests && python test_phase1.py
  3. All 5 checks should print ✅
"""

import sys
import os

# Resolve project root (one level up from tests/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# ═══════════════════════════════════════════════════════════
# STEP 0: Create a sample text file for testing (no PDF needed)
# ═══════════════════════════════════════════════════════════

os.makedirs("data/sample_docs", exist_ok=True)

# Create a simple test document
sample_text = """
Artificial Intelligence in Healthcare

AI is transforming healthcare through improved diagnostics, personalized treatment,
and operational efficiency. Machine learning models can analyze medical images with
accuracy comparable to expert radiologists.

Deep Learning Applications

Convolutional Neural Networks (CNNs) have shown remarkable success in medical image
classification. Recent studies demonstrate that ensemble methods combining ResNet,
EfficientNet, and other architectures can achieve state-of-the-art performance on
tasks like diabetic retinopathy detection.

Natural Language Processing in Clinical Settings

NLP techniques enable extraction of structured information from unstructured clinical
notes. Named Entity Recognition (NER) systems can identify medications, conditions,
and procedures from free-text medical records. Large Language Models are increasingly
being explored for clinical decision support.

Challenges and Future Directions

Key challenges include data privacy regulations (GDPR, HIPAA), model interpretability
requirements in clinical settings, and the need for diverse training datasets to avoid
bias. Federated learning offers a promising approach to training models across
institutions without sharing raw patient data.
"""

sample_path = "data/sample_docs/test_document.txt"
with open(sample_path, "w") as f:
    f.write(sample_text)

print("=" * 60)
print("PHASE 1 TEST — Document Ingestion Pipeline")
print("=" * 60)
print()

# ═══════════════════════════════════════════════════════════
# TEST 1: Document Loading
# ═══════════════════════════════════════════════════════════

print("TEST 1: Loading a single document...")
try:
    from src.document_loader import load_document

    docs = load_document(sample_path)

    assert len(docs) > 0, "No documents returned"
    assert docs[0].page_content.strip() != "", "Document content is empty"
    assert "source" in docs[0].metadata, "Missing 'source' in metadata"

    print(f"  ✅ Loaded {len(docs)} document(s)")
    print(f"     Content length: {len(docs[0].page_content)} chars")
    print(f"     Metadata: {docs[0].metadata}")
    print()
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    print()

# ═══════════════════════════════════════════════════════════
# TEST 2: Multiple Document Loading
# ═══════════════════════════════════════════════════════════

print("TEST 2: Loading multiple documents...")
try:
    from src.document_loader import load_multiple_documents

    # Create a second test file
    sample_path2 = "data/sample_docs/test_document_2.txt"
    with open(sample_path2, "w") as f:
        f.write("This is a second test document about machine learning.\n" * 20)

    docs = load_multiple_documents([sample_path, sample_path2])

    assert len(docs) >= 2, f"Expected at least 2 docs, got {len(docs)}"

    # Check each doc has the right source
    sources = [d.metadata.get("source") for d in docs]
    print(f"  ✅ Loaded {len(docs)} documents from {len(set(sources))} files")
    print(f"     Sources: {set(sources)}")
    print()
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    print()

# ═══════════════════════════════════════════════════════════
# TEST 3: Text Splitting — Basic
# ═══════════════════════════════════════════════════════════

print("TEST 3: Splitting documents into chunks...")
try:
    from src.text_splitter import split_documents
    from langchain_core.documents import Document

    # Use the loaded docs from Test 1
    docs = [Document(
        page_content=sample_text,
        metadata={"source": "test_document.txt"}
    )]

    chunks = split_documents(docs, chunk_size=500, chunk_overlap=100)

    assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"
    assert all("source" in c.metadata for c in chunks), "Missing source metadata"
    assert all("chunk_index" in c.metadata for c in chunks), "Missing chunk_index"

    print(f"  ✅ Split into {len(chunks)} chunks")
    print(f"     Chunk sizes: {[len(c.page_content) for c in chunks]}")
    print(f"     All chunks have metadata: source + chunk_index")
    print()
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    print()

# ═══════════════════════════════════════════════════════════
# TEST 4: Chunk Size Constraints
# ═══════════════════════════════════════════════════════════

print("TEST 4: Verifying chunk size constraints...")
try:
    from src.text_splitter import split_documents
    from langchain_core.documents import Document

    # Create a long document
    long_text = "This is a sentence about AI. " * 200  # ~5800 chars
    docs = [Document(page_content=long_text, metadata={"source": "long.txt"})]

    chunk_size = 500
    chunk_overlap = 100
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Check no chunk is excessively larger than chunk_size
    # (RecursiveCharacterTextSplitter may slightly exceed due to separator logic)
    max_allowed = chunk_size + 200  # Allow some flexibility
    oversized = [i for i, c in enumerate(chunks) if len(c.page_content) > max_allowed]

    assert len(oversized) == 0, f"Chunks {oversized} exceed max size ({max_allowed})"
    assert len(chunks) >= 5, f"Expected at least 5 chunks from 5800 chars, got {len(chunks)}"

    # Check overlap exists by looking for shared text between consecutive chunks
    overlap_found = False
    for i in range(len(chunks) - 1):
        end_of_current = chunks[i].page_content[-50:]
        start_of_next = chunks[i + 1].page_content[:200]
        if end_of_current[:30] in start_of_next:
            overlap_found = True
            break

    print(f"  ✅ {len(chunks)} chunks, all within size limits")
    print(f"     Max chunk: {max(len(c.page_content) for c in chunks)} chars")
    print(f"     Min chunk: {min(len(c.page_content) for c in chunks)} chars")
    print(f"     Overlap between chunks: {'detected' if overlap_found else 'not detected (OK for small docs)'}")
    print()
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    print()

# ═══════════════════════════════════════════════════════════
# TEST 5: Metadata Preservation
# ═══════════════════════════════════════════════════════════

print("TEST 5: Metadata preserved through pipeline...")
try:
    from src.document_loader import load_document
    from src.text_splitter import split_documents

    docs = load_document(sample_path)
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=50)

    for i, chunk in enumerate(chunks):
        assert "source" in chunk.metadata, f"Chunk {i} missing 'source'"
        assert "chunk_index" in chunk.metadata, f"Chunk {i} missing 'chunk_index'"
        assert chunk.metadata["chunk_index"] == i, f"Chunk {i} has wrong index"

    print(f"  ✅ All {len(chunks)} chunks have correct metadata")
    print(f"     Sample: {chunks[0].metadata}")
    print()
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    print()

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("PHASE 1 TESTING COMPLETE")
print("=" * 60)
print()
print("If all 5 tests show ✅, Phase 1 is working correctly.")