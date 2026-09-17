"""
Builds a FAISS vector index over the local knowledge base for semantic
search. Built once, lazily, on first use, and cached in memory for the rest
of the process — 5 articles embed in well under a second, so there's no
need for a persisted on-disk index at this scale.

This upgrades Tool 1 (knowledge_search) from keyword overlap to real
semantic similarity: a query like "internet not working on my laptop" will
now match the "Laptop won't connect to office Wi-Fi" article even though it
shares almost no exact words with the title.
"""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import get_embeddings
from src.db import load_knowledge_base

_vector_store = None


def _build_vector_store() -> FAISS:
    kb = load_knowledge_base()
    docs = [
        Document(
            page_content=f"{article['title']}. {article['content']}",
            metadata=article,
        )
        for article in kb
    ]
    return FAISS.from_documents(docs, get_embeddings())


def get_vector_store() -> FAISS:
    global _vector_store
    if _vector_store is None:
        _vector_store = _build_vector_store()
    return _vector_store


def semantic_search(query: str, top_k: int = 2) -> list[dict]:
    store = get_vector_store()
    results = store.similarity_search(query, k=top_k)
    return [r.metadata for r in results]
