"""
Central settings + LLM factory. Every other file gets its model from
get_llm() rather than calling ChatOpenAI/ChatGoogleGenerativeAI directly —
one place to swap providers.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")

    KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "data/knowledge_base.json")
    TICKETS_DB_PATH = os.getenv("TICKETS_DB_PATH", "data/tickets.json")


settings = Settings()


def get_llm(temperature: float = 0.2):
    if settings.LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=settings.GOOGLE_MODEL, google_api_key=settings.GOOGLE_API_KEY, temperature=temperature)

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=settings.OPENAI_MODEL, api_key=settings.OPENAI_API_KEY, temperature=temperature)


def get_embeddings():
    """
    Embedding model factory, mirroring get_llm(). Reuses the same
    LLM_PROVIDER + API key already configured — no separate embeddings
    provider or key needed.
    """
    if settings.LLM_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=settings.GOOGLE_API_KEY)

    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY)
