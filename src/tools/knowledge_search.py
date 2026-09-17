"""
Tool 1 — Knowledge Search.

Primary path: semantic search via the FAISS vector store (vector_store.py),
which understands meaning, not just exact words.

Fallback path: if the vector store fails for any reason (embeddings API
down, quota hit, no network) the tool falls back to plain keyword overlap
over the same knowledge base rather than failing the whole request — this
is the "handles tool failures gracefully" requirement in practice.
"""

from src.db import load_knowledge_base


def _keyword_search(query: str, top_k: int) -> list[dict]:
    kb = load_knowledge_base()
    query_words = set(query.lower().split())

    scored = []
    for article in kb:
        text = (article["title"] + " " + article["content"] + " " + article["category"]).lower()
        score = sum(1 for w in query_words if w in text)
        if score > 0:
            scored.append((score, article))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [article for _, article in scored[:top_k]]


def knowledge_search(query: str, top_k: int = 2) -> list[dict]:
    try:
        from src.tools.vector_store import semantic_search
        results = semantic_search(query, top_k=top_k)
        if results:
            return results
    except Exception:
        pass  # embeddings unavailable — fall through to keyword search

    return _keyword_search(query, top_k)
