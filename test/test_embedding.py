"""
Phase 2 Test Script — Embedding & Vector Store
Run after creating src/embeddings.py and src/vector_store.py

Prerequisites:
    pip install langchain langchain-community chromadb sentence-transformers
    Phase 1 tests passing (src/document_loader.py, src/text_splitter.py)
"""

import sys
import os
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# ═══════════════════════════════════════════════════════════
# SETUP: Create test documents to work with
# ═══════════════════════════════════════════════════════════

from langchain_core.documents import Document

TEST_CHUNKS = [
    Document(page_content="Artificial intelligence is transforming healthcare through improved diagnostics and personalized treatment plans.", metadata={"source": "healthcare.txt", "chunk_index": 0}),
    Document(page_content="Deep learning models like ResNet and EfficientNet achieve high accuracy in medical image classification tasks.", metadata={"source": "healthcare.txt", "chunk_index": 1}),
    Document(page_content="Natural language processing enables extraction of structured data from clinical notes using named entity recognition.", metadata={"source": "healthcare.txt", "chunk_index": 2}),
    Document(page_content="Retrieval-Augmented Generation combines vector search with language models to produce grounded answers from documents.", metadata={"source": "rag_guide.txt", "chunk_index": 0}),
    Document(page_content="Vector databases like ChromaDB and FAISS store embeddings for fast similarity search over large document collections.", metadata={"source": "rag_guide.txt", "chunk_index": 1}),
    Document(page_content="Docker containers package applications with their dependencies, ensuring consistent behavior across environments.", metadata={"source": "devops.txt", "chunk_index": 0}),
    Document(page_content="Kubernetes orchestrates container deployments, handling scaling, load balancing, and automatic restarts.", metadata={"source": "devops.txt", "chunk_index": 1}),
    Document(page_content="MLflow tracks machine learning experiments including parameters, metrics, and model artifacts for reproducibility.", metadata={"source": "mlops.txt", "chunk_index": 0}),
    Document(page_content="FastAPI is a modern Python web framework for building REST APIs with automatic OpenAPI documentation.", metadata={"source": "mlops.txt", "chunk_index": 1}),
    Document(page_content="Transfer learning fine-tunes pre-trained neural networks on domain-specific data, reducing the need for large labeled datasets.", metadata={"source": "healthcare.txt", "chunk_index": 3}),
]

# Use a test-specific persist directory (won't clash with real data)
TEST_PERSIST_DIR = "./data/test_chroma_db"
TEST_COLLECTION = "test_phase2"

passed = 0
failed = 0
total = 7

print("=" * 60)
print("PHASE 2 TEST — Embedding & Vector Store")
print("=" * 60)
print()

# ═══════════════════════════════════════════════════════════
# TEST 1: Embedding model loads correctly
# ═══════════════════════════════════════════════════════════

print("TEST 1: Loading embedding model...")
embedding_model = None
try:
    from src.embeddings import get_embedding_model

    embedding_model = get_embedding_model("all-MiniLM-L6-v2")

    assert embedding_model is not None, "Model returned None"
    assert hasattr(embedding_model, "embed_query"), "Model missing embed_query method"
    assert hasattr(embedding_model, "embed_documents"), "Model missing embed_documents method"

    print(f"  ✅ Model loaded successfully")
    print(f"     Type: {type(embedding_model).__name__}")
    passed += 1
    print()
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    failed += 1
    print()

# ═══════════════════════════════════════════════════════════
# TEST 2: Single query embedding works
# ═══════════════════════════════════════════════════════════

print("TEST 2: Generating a single query embedding...")
try:
    assert embedding_model is not None, "Skipped — model not loaded (Test 1 failed)"

    vector = embedding_model.embed_query("What is deep learning?")

    assert isinstance(vector, list), f"Expected list, got {type(vector)}"
    assert len(vector) > 0, "Empty vector returned"
    assert all(isinstance(v, float) for v in vector), "Vector contains non-float values"

    # all-MiniLM-L6-v2 produces 384-dim vectors
    expected_dim = 384
    assert len(vector) == expected_dim, f"Expected {expected_dim} dims, got {len(vector)}"

    print(f"  ✅ Embedding generated: {len(vector)} dimensions")
    print(f"     First 5 values: {[round(v, 4) for v in vector[:5]]}")
    passed += 1
    print()
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    failed += 1
    print()

# ═══════════════════════════════════════════════════════════
# TEST 3: Batch document embedding works
# ═══════════════════════════════════════════════════════════

print("TEST 3: Embedding multiple documents...")
try:
    assert embedding_model is not None, "Skipped — model not loaded"

    texts = [chunk.page_content for chunk in TEST_CHUNKS[:3]]
    vectors = embedding_model.embed_documents(texts)

    assert len(vectors) == 3, f"Expected 3 vectors, got {len(vectors)}"
    assert all(len(v) == 384 for v in vectors), "Not all vectors are 384-dim"

    print(f"  ✅ Batch embedding: {len(vectors)} documents, {len(vectors[0])} dims each")
    passed += 1
    print()
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    failed += 1
    print()

# ═══════════════════════════════════════════════════════════
# TEST 4: Create vector store from documents
# ═══════════════════════════════════════════════════════════

print("TEST 4: Creating vector store from chunks...")
vector_store = None
try:
    from src.vector_store import create_vector_store

    assert embedding_model is not None, "Skipped — model not loaded"

    # Clean up any previous test store
    if os.path.exists(TEST_PERSIST_DIR):
        shutil.rmtree(TEST_PERSIST_DIR)

    # Temporarily patch PERSIST_DIR in vector_store module
    import src.vector_store as vs_module
    original_persist = vs_module.PERSIST_DIR
    vs_module.PERSIST_DIR = TEST_PERSIST_DIR

    vector_store = create_vector_store(
        documents=TEST_CHUNKS,
        embedding_model=embedding_model,
        collection_name=TEST_COLLECTION,
    )

    assert vector_store is not None, "create_vector_store returned None"
    assert os.path.exists(TEST_PERSIST_DIR), "Persist directory was not created"

    # Verify document count
    count = vector_store._collection.count()
    assert count == len(TEST_CHUNKS), f"Expected {len(TEST_CHUNKS)} docs in store, got {count}"

    print(f"  ✅ Vector store created: {count} documents stored")
    print(f"     Persist dir: {TEST_PERSIST_DIR}")
    print(f"     Collection: {TEST_COLLECTION}")
    passed += 1
    print()
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    failed += 1
    print()

# ═══════════════════════════════════════════════════════════
# TEST 5: Query returns relevant results
# ═══════════════════════════════════════════════════════════

print("TEST 5: Querying vector store for relevant documents...")
try:
    assert vector_store is not None, "Skipped — vector store not created (Test 4 failed)"

    # Query about healthcare — should return healthcare docs
    results = vector_store.similarity_search_with_score("medical image diagnosis", k=3)

    assert len(results) > 0, "No results returned"
    assert len(results) <= 3, f"Expected at most 3 results, got {len(results)}"

    # Check structure
    doc, score = results[0]
    assert hasattr(doc, "page_content"), "Result missing page_content"
    assert hasattr(doc, "metadata"), "Result missing metadata"
    assert "source" in doc.metadata, "Result missing source metadata"
    assert "chunk_index" in doc.metadata, "Result missing chunk_index metadata"
    assert isinstance(score, float), "Score should be a float"

    # Check relevance — top result should be from healthcare docs
    top_source = doc.metadata["source"]
    top_content_lower = doc.page_content.lower()
    is_relevant = "healthcare" in top_source or "medical" in top_content_lower or "image" in top_content_lower

    print(f"  ✅ Query returned {len(results)} results")
    print(f"     Top result source: {top_source}")
    print(f"     Top result score: {round(score, 4)} (lower = more similar)")
    print(f"     Relevance check: {'PASS — healthcare-related' if is_relevant else 'WEAK — check manually'}")
    print()
    print(f"     Top 3 results:")
    for i, (d, s) in enumerate(results):
        preview = d.page_content[:80].replace("\n", " ")
        print(f"       {i+1}. [{d.metadata['source']}] (score: {round(s, 4)}) {preview}...")

    passed += 1
    print()
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    failed += 1
    print()

# ═══════════════════════════════════════════════════════════
# TEST 6: Load existing vector store from disk
# ═══════════════════════════════════════════════════════════

print("TEST 6: Loading vector store from disk...")
try:
    from src.vector_store import load_vector_store

    assert embedding_model is not None, "Skipped — model not loaded"

    loaded_store = load_vector_store(
        embedding_model=embedding_model,
        collection_name=TEST_COLLECTION,
    )

    assert loaded_store is not None, "load_vector_store returned None"

    # Verify it's queryable
    results = loaded_store.similarity_search("vector database", k=2)
    assert len(results) > 0, "Loaded store returned no results"

    count = loaded_store._collection.count()

    print(f"  ✅ Loaded store from disk: {count} documents, queryable")
    print(f"     Test query 'vector database' returned {len(results)} results")
    passed += 1
    print()
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    failed += 1
    print()

# ═══════════════════════════════════════════════════════════
# TEST 7: Load nonexistent store returns None gracefully
# ═══════════════════════════════════════════════════════════

print("TEST 7: Loading nonexistent store returns None...")
try:
    from src.vector_store import load_vector_store

    assert embedding_model is not None, "Skipped — model not loaded"

    # Point to a directory that doesn't exist
    import src.vector_store as vs_module
    vs_module.PERSIST_DIR = "./data/nonexistent_store_xyz"

    result = load_vector_store(
        embedding_model=embedding_model,
        collection_name="nonexistent",
    )

    assert result is None, f"Expected None for nonexistent store, got {type(result)}"

    # Restore original
    vs_module.PERSIST_DIR = TEST_PERSIST_DIR

    print(f"  ✅ Returned None gracefully (no crash)")
    passed += 1
    print()
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    failed += 1
    print()

# ═══════════════════════════════════════════════════════════
# CLEANUP & SUMMARY
# ═══════════════════════════════════════════════════════════

# Restore original PERSIST_DIR
try:
    import src.vector_store as vs_module
    vs_module.PERSIST_DIR = original_persist
except:
    pass

# Clean up test store
if os.path.exists(TEST_PERSIST_DIR):
    shutil.rmtree(TEST_PERSIST_DIR)
    print(f"Cleaned up test store: {TEST_PERSIST_DIR}")
    print()

print("=" * 60)
print(f"PHASE 2 RESULTS: {passed}/{total} passed, {failed}/{total} failed")
print("=" * 60)
print()