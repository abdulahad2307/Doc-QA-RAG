"""ChromaDB vector store operations."""
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Optional
import shutil
from pathlib import Path

PERSIST_DIR = "./data/chroma_db"

def create_vector_store(
    documents: List[Document],
    embedding_model,
    collection_name: str = "documents",
) -> Chroma:
    """Create a new vector store from documents."""
    # Clear existing store
    if Path(PERSIST_DIR).exists():
        shutil.rmtree(PERSIST_DIR)
    
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=PERSIST_DIR,
    )
    print(f"🗄️ Created vector store with {len(documents)} documents")
    return vector_store

def load_vector_store(
    embedding_model,
    collection_name: str = "documents",
) -> Optional[Chroma]:
    """Load an existing vector store."""
    if not Path(PERSIST_DIR).exists():
        return None
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model,
        collection_name=collection_name,
    )