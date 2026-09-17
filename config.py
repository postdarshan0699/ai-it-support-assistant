"""
Central settings + LLM factory.

The rest of the project gets the LLM and embedding model from this file,
so the provider can be changed in one place.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Read Streamlit secrets when running on Streamlit Cloud.
try:
    import streamlit as st
    STREAMLIT_SECRETS = st.secrets
except Exception:
    STREAMLIT_SECRETS = {}


def get_setting(name, default=None):
    """Get a setting from Streamlit Secrets first, then environment variables."""
    if name in STREAMLIT_SECRETS:
        return STREAMLIT_SECRETS[name]

    return os.getenv(name, default)


class Settings:
    LLM_PROVIDER = get_setting("LLM_PROVIDER", "openai")

    OPENAI_API_KEY = get_setting("OPENAI_API_KEY")
    OPENAI_MODEL = get_setting("OPENAI_MODEL", "gpt-4o-mini")

    GOOGLE_API_KEY = get_setting("GOOGLE_API_KEY")
    GOOGLE_MODEL = get_setting("GOOGLE_MODEL", "gemini-2.5-flash")

    KNOWLEDGE_BASE_PATH = get_setting(
        "KNOWLEDGE_BASE_PATH",
        "data/knowledge_base.json"
    )

    TICKETS_DB_PATH = get_setting(
        "TICKETS_DB_PATH",
        "data/tickets.json"
    )


settings = Settings()


def get_llm(temperature: float = 0.2):
    if settings.LLM_PROVIDER.lower() == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.GOOGLE_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=temperature,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=temperature,
    )


def get_embeddings():
    """
    Create the embedding model using the same provider configured
    for the application.
    """

    if settings.LLM_PROVIDER.lower() == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.GOOGLE_API_KEY,
        )

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.OPENAI_API_KEY,
    )

