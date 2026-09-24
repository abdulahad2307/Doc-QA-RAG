"""End-to-end RAG pipeline."""
from src.document_loader import load_document, load_multiple_documents
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store, load_vector_store
from src.retriever import retrieve_documents
from src.llm_chain import generate_answer, get_llm
from typing import List, Dict

class RAGPipeline:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = get_embedding_model(embedding_model_name)
        self.vector_store = None
        self.llm = get_llm()
    
    def ingest(self, file_paths: List[str], chunk_size: int = 1000):
        """Ingest documents into the vector store."""
        docs = load_multiple_documents(file_paths)
        chunks = split_documents(docs, chunk_size=chunk_size)
        self.vector_store = create_vector_store(chunks, self.embedding_model)
        return len(chunks)
    
    def query(self, question: str, k: int = 5) -> Dict:
        """Ask a question and get a cited answer."""
        if self.vector_store is None:
            self.vector_store = load_vector_store(self.embedding_model)
        if self.vector_store is None:
            return {"answer": "No documents loaded.", "sources": []}
        
        retrieved = retrieve_documents(self.vector_store, question, k=k)
        result = generate_answer(question, retrieved, self.llm)
        return result
    