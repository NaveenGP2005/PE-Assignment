"""
Text chunking module with chunk size & overlap configuration.
"""
from typing import List
from langchain_core.documents import Document
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Document]:
    """Splits a list of Documents into chunks with overlap while preserving metadata.
    
    Args:
        documents: List of Document objects with metadata.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlapping characters between adjacent chunks.
        
    Returns:
        List of chunked Document objects with updated chunk indices.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    
    chunks = splitter.split_documents(documents)
    
    # Enrich metadata with unique chunk identifier
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        chunk.metadata["char_count"] = len(chunk.page_content)
        
    return chunks
