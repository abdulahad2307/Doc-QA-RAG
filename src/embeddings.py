"""Embedding model configuration."""
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Get a sentence-transformer embedding model.
    
    Options:
    - "all-MiniLM-L6-v2": Fast, good quality (384 dim) — RECOMMENDED
    - "all-mpnet-base-v2": Higher quality, slower (768 dim)
    - "BAAI/bge-small-en-v1.5": Very good quality/speed ratio
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # Change to "cuda" if GPU
        encode_kwargs={"normalize_embeddings": True},
    )
