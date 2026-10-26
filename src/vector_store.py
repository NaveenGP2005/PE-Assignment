"""
Vector Store module managing FAISS vector database and embeddings.
Supports local offline embeddings (SentenceTransformers) and Google Gemini / OpenAI embeddings.
"""
from typing import List, Tuple, Optional
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import VECTOR_STORE_DIR, DEFAULT_EMBEDDING_MODEL, DEFAULT_TOP_K


def get_embedding_function(provider: str = "local", api_key: Optional[str] = None):
    """Factory to get the embedding model based on provider preference.
    
    Defaults to local HuggingFace 'sentence-transformers/all-MiniLM-L6-v2'
    which runs 100% on CPU locally with zero cost and no external API calls.
    """
    if provider == "google" and api_key:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    elif provider == "openai" and api_key:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(openai_api_key=api_key)
    else:
        # Default local offline embeddings (CPU-friendly, fast)
        return HuggingFaceEmbeddings(
            model_name=DEFAULT_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )


def build_vector_store(
    chunks: List[Document],
    embedding_provider: str = "local",
    api_key: Optional[str] = None
) -> FAISS:
    """Builds an in-memory or persisted FAISS vector index from chunked documents.
    
    Args:
        chunks: List of Document chunks with metadata.
        embedding_provider: 'local', 'google', or 'openai'.
        api_key: Optional API key.
        
    Returns:
        FAISS vector store instance.
    """
    if not chunks:
        raise ValueError("Cannot build vector store from empty document list.")
        
    embeddings = get_embedding_function(provider=embedding_provider, api_key=api_key)
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    return vector_store


def save_vector_store(vector_store: FAISS, save_dir: Path = VECTOR_STORE_DIR):
    """Saves FAISS index and docstore locally to disk."""
    save_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(save_dir))


def load_vector_store(
    save_dir: Path = VECTOR_STORE_DIR,
    embedding_provider: str = "local",
    api_key: Optional[str] = None
) -> Optional[FAISS]:
    """Loads a previously saved FAISS vector store from disk."""
    index_file = save_dir / "index.faiss"
    if not index_file.exists():
        return None
        
    embeddings = get_embedding_function(provider=embedding_provider, api_key=api_key)
    return FAISS.load_local(str(save_dir), embeddings, allow_dangerous_deserialization=True)


def retrieve_relevant_chunks(
    vector_store: FAISS,
    query: str,
    top_k: int = DEFAULT_TOP_K
) -> List[Tuple[Document, float]]:
    """Retrieves top_k most similar document chunks along with their distance/similarity score.
    
    Returns:
        List of tuples: (Document, score)
    """
    results = vector_store.similarity_search_with_score(query, k=top_k)
    return results
