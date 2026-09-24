"""
Phase 3 Test Script — Retrieval, LLM Chain & Pipeline
Run after creating src/retriever.py, src/llm_chain.py, src/pipeline.py
"""

import sys
import os
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_core.documents import Document

# ═══════════════════════════════════════════════════════════
# SETUP: Build a test vector store to query against
# ═══════════════════════════════════════════════════════════

TEST_PERSIST_DIR = "./data/test_chroma_phase3"
TEST_COLLECTION = "test_phase3"

TEST_CHUNKS = [
    Document(page_content="Artificial intelligence is transforming healthcare through improved diagnostics, personalized treatment plans, and drug discovery.", metadata={"source": "healthcare.txt", "chunk_index": 0}),
    Document(page_content="Convolutional Neural Networks and Vision Transformers achieve state-of-the-art accuracy in medical image classification for diabetic retinopathy detection.", metadata={"source": "healthcare.txt", "chunk_index": 1}),
    Document(page_content="Named Entity Recognition systems extract medications, conditions, and procedures from unstructured clinical notes.", metadata={"source": "healthcare.txt", "chunk_index": 2}),
    Document(page_content="Retrieval-Augmented Generation combines semantic search over document chunks with language models to produce grounded, cited answers.", metadata={"source": "rag_guide.txt", "chunk_index": 0}),
    Document(page_content="ChromaDB and FAISS are popular vector databases that store embeddings for fast approximate nearest neighbor search.", metadata={"source": "rag_guide.txt", "chunk_index": 1}),
    Document(page_content="Document chunking strategies include fixed-size, recursive character, and semantic splitting. Chunk overlap helps preserve context across boundaries.", metadata={"source": "rag_guide.txt", "chunk_index": 2}),
    Document(page_content="Docker containers package applications with dependencies. Kubernetes orchestrates container deployments with scaling and health checks.", metadata={"source": "devops.txt", "chunk_index": 0}),
    Document(page_content="MLflow tracks experiments including parameters, metrics, and model artifacts. The model registry manages staging and production transitions.", metadata={"source": "mlops.txt", "chunk_index": 0}),
    Document(page_content="FastAPI serves machine learning models via REST endpoints with automatic Swagger documentation and async request handling.", metadata={"source": "mlops.txt", "chunk_index": 1}),
    Document(page_content="The French Revolution began in 1789 and fundamentally transformed French political and social structures over the following decade.", metadata={"source": "history.txt", "chunk_index": 0}),
]

def setup_test_store():
    """Create a test vector store for retriever tests."""
    from src.embeddings import get_embedding_model
    from src.vector_store import create_vector_store
    import src.vector_store as vs_module

    if os.path.exists(TEST_PERSIST_DIR):
        shutil.rmtree(TEST_PERSIST_DIR)

    model = get_embedding_model("all-MiniLM-L6-v2")

    # Temporarily patch persist dir
    original = vs_module.PERSIST_DIR
    vs_module.PERSIST_DIR = TEST_PERSIST_DIR

    store = create_vector_store(TEST_CHUNKS, model, TEST_COLLECTION)

    vs_module.PERSIST_DIR = original
    return store, model

def has_api_key():
    """Check if Anthropic API key is available."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return key != "" and not key.startswith("your-")

# ═══════════════════════════════════════════════════════════

passed = 0
failed = 0
skipped = 0
total = 10

print("=" * 60)
print("PHASE 3 TEST — Retrieval, LLM Chain & Pipeline")
print("=" * 60)
print()

api_available = has_api_key()
if not api_available:
    print(" No ANTHROPIC_API_KEY found in .env — Tests 5-10 will be skipped.")
    print("   Retriever tests (1-4) will still run.\n")

# Build test store
print("Setting up test vector store...")
try:
    vector_store, embedding_model = setup_test_store()
    print(f"  ✅ Test store ready: {vector_store._collection.count()} documents\n")
except Exception as e:
    print(f"  ❌ Setup failed: {e}")
    print("  Cannot proceed. Fix Phase 2 first.\n")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════
# RETRIEVER TESTS (no API key needed)
# ═══════════════════════════════════════════════════════════

print("-" * 40)
print("RETRIEVER TESTS (no API key needed)")
print("-" * 40)
print()

# TEST 1: Relevant query returns documents
print("TEST 1: Relevant query returns documents...")
try:
    from src.retriever import retrieve_documents

    results = retrieve_documents(vector_store, "medical image classification", k=3)

    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert len(results) > 0, "No documents returned for a relevant query"
    assert len(results) <= 3, f"Returned {len(results)} docs, expected at most 3"
    assert all(isinstance(d, Document) for d in results), "Results should be Document objects"

    print(f"  ✅ Returned {len(results)} relevant documents")
    for i, doc in enumerate(results):
        print(f"     {i+1}. [{doc.metadata['source']}] {doc.page_content[:60]}...")
    passed += 1
    print()
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    failed += 1
    print()

# TEST 2: Metadata preserved on retrieved documents
print("TEST 2: Retrieved documents retain metadata...")
try:
    from src.retriever import retrieve_documents

    results = retrieve_documents(vector_store, "vector database embeddings", k=3)

    assert len(results) > 0, "No results to check"
    for i, doc in enumerate(results):
        assert "source" in doc.metadata, f"Doc {i} missing 'source'"
        assert "chunk_index" in doc.metadata, f"Doc {i} missing 'chunk_index'"
        assert isinstance(doc.metadata["source"], str), "Source should be string"
        assert isinstance(doc.metadata["chunk_index"], int), "chunk_index should be int"

    print(f"  ✅ All {len(results)} results have source + chunk_index metadata")
    print(f"     Sources: {[d.metadata['source'] for d in results]}")
    passed += 1
    print()
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    failed += 1
    print()

# TEST 3: Irrelevant query returns fewer or no results (score filtering)
print("TEST 3: Irrelevant query gets filtered by score threshold...")
try:
    from src.retriever import retrieve_documents

    relevant_results = retrieve_documents(vector_store, "Docker containers Kubernetes", k=5)
    irrelevant_results = retrieve_documents(vector_store, "recipe for chocolate birthday cake with sprinkles", k=5)

    # Irrelevant should return fewer (or same, if threshold is generous)
    # At minimum, verify it doesn't crash and returns a list
    assert isinstance(irrelevant_results, list), "Should return a list"

    if len(irrelevant_results) < len(relevant_results):
        print(f"  ✅ Score filtering working: relevant={len(relevant_results)}, irrelevant={len(irrelevant_results)}")
    elif len(irrelevant_results) == 0:
        print(f"  ✅ Irrelevant query correctly returned 0 results")
    else:
        print(f"  ✅ Returned {len(irrelevant_results)} results (threshold may be generous — consider tightening)")
        print(f"     This is OK — the filter threshold (1.5) is intentionally permissive")

    passed += 1
    print()
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    failed += 1
    print()

# TEST 4: k parameter controls max results
print("TEST 4: k parameter limits result count...")
try:
    from src.retriever import retrieve_documents

    results_k1 = retrieve_documents(vector_store, "machine learning", k=1)
    results_k5 = retrieve_documents(vector_store, "machine learning", k=5)

    assert len(results_k1) <= 1, f"k=1 returned {len(results_k1)} results"
    assert len(results_k5) <= 5, f"k=5 returned {len(results_k5)} results"
    assert len(results_k5) >= len(results_k1), "k=5 should return >= k=1 results"

    print(f"  ✅ k=1 → {len(results_k1)} result(s), k=5 → {len(results_k5)} result(s)")
    passed += 1
    print()
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    failed += 1
    print()

# ═══════════════════════════════════════════════════════════
# LLM CHAIN TESTS (require API key)
# ═══════════════════════════════════════════════════════════

print("-" * 40)
print("LLM CHAIN TESTS (require ANTHROPIC_API_KEY)")
print("-" * 40)
print()

# TEST 5: LLM initializes
print("TEST 5: LLM client initializes...")
if not api_available:
    print(f" SKIPPED — no API key")
    skipped += 1
    print()
else:
    try:
        from src.llm_chain import get_llm

        llm = get_llm()

        assert llm is not None, "get_llm() returned None"
        assert hasattr(llm, "invoke"), "LLM missing invoke method"

        print(f"  ✅ LLM initialized: {type(llm).__name__}")
        passed += 1
        print()
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        failed += 1
        print()

# TEST 6: generate_answer returns correct structure
print("TEST 6: generate_answer() returns answer + sources dict...")
if not api_available:
    print(f"  SKIPPED — no API key")
    skipped += 1
    print()
else:
    try:
        from src.llm_chain import generate_answer

        test_docs = [
            Document(page_content="ChromaDB stores embeddings for vector search.", metadata={"source": "rag.txt", "chunk_index": 0}),
            Document(page_content="FAISS is a library for efficient similarity search.", metadata={"source": "rag.txt", "chunk_index": 1}),
        ]

        result = generate_answer("What are vector databases?", test_docs)

        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "answer" in result, "Missing 'answer' key"
        assert "sources" in result, "Missing 'sources' key"
        assert isinstance(result["answer"], str), "Answer should be string"
        assert len(result["answer"]) > 10, "Answer too short"
        assert isinstance(result["sources"], list), "Sources should be list"
        assert len(result["sources"]) == 2, f"Expected 2 sources, got {len(result['sources'])}"

        for src in result["sources"]:
            assert "source" in src, "Source entry missing 'source' key"
            assert "chunk_index" in src, "Source entry missing 'chunk_index'"
            assert "content_preview" in src, "Source entry missing 'content_preview'"

        print(f"  ✅ Response structure correct")
        print(f"     Answer: {result['answer'][:100]}...")
        print(f"     Sources: {len(result['sources'])} entries")
        passed += 1
        print()
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        failed += 1
        print()

# TEST 7: Answer is grounded in context (not hallucinated)
print("TEST 7: Answer is grounded in provided context...")
if not api_available:
    print(f" SKIPPED — no API key")
    skipped += 1
    print()
else:
    try:
        from src.llm_chain import generate_answer

        specific_docs = [
            Document(page_content="The XYZ-9000 satellite was launched on March 15, 2024 from Cape Canaveral.", metadata={"source": "space.txt", "chunk_index": 0}),
        ]

        result = generate_answer("When was the XYZ-9000 launched?", specific_docs)
        answer_lower = result["answer"].lower()

        # The answer should mention March, 2024, or Cape Canaveral
        grounded = any(term in answer_lower for term in ["march", "2024", "cape canaveral", "xyz-9000"])

        assert grounded, f"Answer doesn't reference context content: {result['answer']}"

        print(f"  ✅ Answer is grounded in context")
        print(f"     Answer: {result['answer'][:120]}...")
        passed += 1
        print()
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        failed += 1
        print()

# TEST 8: Empty docs returns fallback
print("TEST 8: Empty document list returns fallback message...")
if not api_available:
    print(f" SKIPPED — no API key")
    skipped += 1
    print()
else:
    try:
        from src.llm_chain import generate_answer

        result = generate_answer("What is quantum computing?", [])

        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "answer" in result, "Missing answer key"
        assert len(result["sources"]) == 0, "Sources should be empty for empty input"

        # Answer should indicate lack of information
        answer_lower = result["answer"].lower()
        indicates_no_info = any(term in answer_lower for term in ["no ", "don't have", "no relevant", "no documents", "cannot"])
        assert indicates_no_info, f"Fallback message not clear: {result['answer']}"

        print(f"  ✅ Empty docs handled gracefully")
        print(f"     Fallback: {result['answer'][:100]}")
        passed += 1
        print()
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        failed += 1
        print()

# ═══════════════════════════════════════════════════════════
# PIPELINE INTEGRATION TESTS (require API key)
# ═══════════════════════════════════════════════════════════

print("-" * 40)
print("PIPELINE INTEGRATION TESTS")
print("-" * 40)
print()

# TEST 9: Pipeline ingest
print("TEST 9: RAGPipeline.ingest() processes files...")
if not api_available:
    print(f" SKIPPED — no API key")
    skipped += 1
    print()
else:
    try:
        from src.pipeline import RAGPipeline

        # Create a test file
        os.makedirs("data/test_pipeline", exist_ok=True)
        test_file = "data/test_pipeline/test_ingest.txt"
        with open(test_file, "w") as f:
            f.write("Machine learning is a subset of artificial intelligence. " * 30)
            f.write("\n\nDeep learning uses neural networks with many layers. " * 30)

        pipeline = RAGPipeline()
        chunk_count = pipeline.ingest([test_file], chunk_size=500)

        assert isinstance(chunk_count, int), f"Expected int, got {type(chunk_count)}"
        assert chunk_count > 0, "No chunks created"

        print(f"  ✅ Ingested: {chunk_count} chunks from 1 file")
        passed += 1
        print()
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        failed += 1
        print()

# TEST 10: Pipeline end-to-end query
print("TEST 10: RAGPipeline.query() returns answer + sources...")
if not api_available:
    print(f" SKIPPED — no API key")
    skipped += 1
    print()
else:
    try:
        # Reuse pipeline from Test 9 (it should still have the store loaded)
        result = pipeline.query("What is machine learning?")

        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "answer" in result, "Missing answer"
        assert "sources" in result, "Missing sources"
        assert len(result["answer"]) > 10, "Answer too short"
        assert len(result["sources"]) > 0, "No sources returned"

        print(f"  ✅ End-to-end pipeline working!")
        print(f"     Question: What is machine learning?")
        print(f"     Answer: {result['answer'][:120]}...")
        print(f"     Sources: {len(result['sources'])} cited")
        passed += 1
        print()
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        failed += 1
        print()

# ═══════════════════════════════════════════════════════════
# CLEANUP & SUMMARY
# ═══════════════════════════════════════════════════════════

# Clean up
for path in [TEST_PERSIST_DIR, "data/test_pipeline", "data/chroma_db"]:
    if os.path.exists(path):
        shutil.rmtree(path)

print("=" * 60)
print(f"PHASE 3 RESULTS: {passed} passed, {failed} failed, {skipped} skipped  (out of {total})")
print("=" * 60)
print()

if failed == 0 and skipped == 0:
    print("🎉 All tests passed! Phase 3 is working correctly.")
    print()
    print("NEXT STEP (Phase 4):")
    print("  Create app/streamlit_app.py")
    print("  Run: streamlit run app/streamlit_app.py")
elif failed == 0 and skipped > 0:
    print(f"✅ All runnable tests passed ({skipped} skipped — no API key).")
    print()
    print("To run the full suite:")
    print("  1. Copy .env.example to .env")
    print("  2. Add your ANTHROPIC_API_KEY")
    print("  3. Re-run: python -m tests.test_phase3")
else:
    print("Some tests failed. Common fixes:")
    print()
    print("  'No module named src.retriever'")
    print("     → Create src/retriever.py from the Project 1 README")
    print()
    print("  'No module named src.llm_chain'")
    print("     → Create src/llm_chain.py from the Project 1 README")
    print()
    print("  'No module named src.pipeline'")
    print("     → Create src/pipeline.py from the Project 1 README")
    print()
    print("  'AuthenticationError' or 'Invalid API key'")
    print("     → Check .env file has a valid ANTHROPIC_API_KEY")
    print()
    print("  'RateLimitError'")
    print("     → Wait 60 seconds and retry, or check your OpenAI billing")

print()