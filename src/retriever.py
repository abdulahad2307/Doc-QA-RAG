"""Retrieval pipeline with optional re-ranking."""
from langchain_core.documents import Document
from typing import List

def retrieve_documents(
    vector_store,
    query: str,
    k: int = 5,
) -> List[Document]:
    """Retrieve top-k relevant documents for a query."""
    results = vector_store.similarity_search_with_score(query, k=k)
    
    # Filter by relevance score (lower = more similar for cosine distance)
    filtered = [(doc, score) for doc, score in results if score < 1.5]
    
    return [doc for doc, _ in filtered]