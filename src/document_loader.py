"""Load documents from various sources (PDF, TXT, MD)."""
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
from typing import List

def load_document(file_path: str) -> List[Document]:
    """Load a single document based on file extension."""
    path = Path(file_path)
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
    }
    loader_class = loaders.get(path.suffix.lower())
    if not loader_class:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    
    loader = loader_class(str(path))
    documents = loader.load()
    
    # Add metadata
    for doc in documents:
        doc.metadata["source"] = path.name
        doc.metadata["file_type"] = path.suffix
    
    return documents

def load_multiple_documents(file_paths: List[str]) -> List[Document]:
    """Load and combine multiple documents."""
    all_docs = []
    for path in file_paths:
        try:
            docs = load_document(path)
            all_docs.extend(docs)
            print(f"✅ Loaded {path}: {len(docs)} pages/sections")
        except Exception as e:
            print(f"❌ Failed to load {path}: {e}")
    return all_docs