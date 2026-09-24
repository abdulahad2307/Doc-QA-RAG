"""LLM answer generation with source citations."""
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from typing import List, Dict
import os

SYSTEM_PROMPT = """You are a helpful document assistant. Answer the user's
question based ONLY on the provided context. If the context doesn't contain
enough information, say "I don't have enough information to answer that."

Rules:
1. Only use information from the provided context
2. Cite sources using [Source: filename, chunk N] format
3. Be concise but thorough
4. If multiple sources agree, mention that for credibility"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """Context:
{context}

Question: {question}

Answer with citations:"""),
])

def get_llm(model: str = "claude-haiku-4-5", temperature: float = 0.1):
    """Initialize LLM. Use claude-haiku-4-5 for cost efficiency."""
    return ChatAnthropic(model=model, temperature=temperature)

def generate_answer(
    question: str,
    retrieved_docs: List[Document],
    llm=None,
) -> Dict:
    """Generate an answer using retrieved context."""
    if llm is None:
        llm = get_llm()
    
    # Format context with source info
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        context_parts.append(
            f"[Source: {source}, Chunk {chunk_idx}]\n{doc.page_content}"
        )
    context = "\n\n---\n\n".join(context_parts)
    
    # Generate answer
    chain = QA_PROMPT | llm
    response = chain.invoke({"context": context, "question": question})
    
    return {
        "answer": response.content,
        "sources": [
            {
                "source": doc.metadata.get("source"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "content_preview": doc.page_content[:200],
            }
            for doc in retrieved_docs
        ],
    }