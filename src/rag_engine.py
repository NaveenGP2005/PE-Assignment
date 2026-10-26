"""
LangChain RAG Engine orchestrating prompt engineering, retrieval, and LLM inference.
Supports Google Gemini, Groq, OpenAI, and Offline Extractive fallback mode.
"""
import time
from typing import Dict, Any, List, Tuple, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.prompt_templates import (
    PERSONAS,
    get_prompt_template,
    SUMMARY_TEMPLATE
)
from src.offline_engine import generate_extractive_answer
from src.vector_store import retrieve_relevant_chunks


def get_llm_instance(
    provider: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.2
):
    """Instantiates the specified LLM client."""
    if provider == "Google Gemini" and api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        m = model_name or "gemini-1.5-flash"
        return ChatGoogleGenerativeAI(
            model=m,
            google_api_key=api_key,
            temperature=temperature
        )
    elif ("Groq" in provider) and api_key:
        from langchain_groq import ChatGroq
        m = model_name or "openai/gpt-oss-120b"
        return ChatGroq(
            model_name=m,
            groq_api_key=api_key,
            temperature=temperature
        )
    elif provider == "OpenAI" and api_key:
        from langchain_openai import ChatOpenAI
        m = model_name or "gpt-4o-mini"
        return ChatOpenAI(
            model=m,
            api_key=api_key,
            temperature=temperature
        )
    return None


def format_context_string(chunks: List[Tuple[Document, float]]) -> str:
    """Formats retrieved document chunks with clear citation metadata tags."""
    formatted_blocks = []
    for idx, (doc, score) in enumerate(chunks, 1):
        page = doc.metadata.get("page", "Unknown")
        source = doc.metadata.get("source", "Document")
        chunk_id = doc.metadata.get("chunk_id", idx)
        formatted_blocks.append(
            f"--- [SOURCE {idx}: {source} | PAGE: {page} | CHUNK: {chunk_id} | SIMILARITY DISTANCE: {score:.4f}] ---\n"
            f"{doc.page_content.strip()}\n"
        )
    return "\n".join(formatted_blocks)


def execute_rag_query(
    query: str,
    vector_store: FAISS,
    persona_name: str = "Research Assistant (Standard)",
    strategy_name: str = "Zero-Shot (Direct Answer)",
    provider: str = "Offline (Local / Extractive Mode)",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    top_k: int = 4
) -> Dict[str, Any]:
    """Executes the full RAG pipeline: retrieval -> prompt compilation -> LLM answer.
    
    Returns a dictionary containing:
        - answer: str
        - formatted_prompt: str (for prompt inspection in UI)
        - retrieved_chunks: list of (Document, score)
        - execution_time: float (seconds)
        - model_used: str
    """
    start_time = time.time()
    
    # 1. Retrieval Phase with Metadata Awareness
    metadata_keywords = [
        "title", "author", "authors", "who wrote", "abstract", 
        "paper name", "name of the paper", "paper called", "what is this paper about"
    ]
    is_meta_query = any(k in query.lower() for k in metadata_keywords)
    
    retrieved = retrieve_relevant_chunks(vector_store, query, top_k=top_k)
    
    # If the user is asking about document-level metadata (Title/Author/Abstract),
    # ensure the first page / chunk 0 is present in the context
    if is_meta_query and hasattr(vector_store, "docstore") and hasattr(vector_store.docstore, "_dict"):
        first_chunks = [
            doc for doc in vector_store.docstore._dict.values()
            if getattr(doc, "metadata", {}).get("chunk_id") == 0 or getattr(doc, "metadata", {}).get("page") == 1
        ]
        if first_chunks:
            sorted_first = sorted(first_chunks, key=lambda d: d.metadata.get("chunk_id", 9999))
            chunk_zero = sorted_first[0]
            if not any(d.page_content == chunk_zero.page_content for d, _ in retrieved):
                filtered_retrieved = [r for r in retrieved if r[0].page_content != chunk_zero.page_content]
                retrieved = [(chunk_zero, 0.0)] + filtered_retrieved[:max(1, top_k - 1)]

    context_str = format_context_string(retrieved)
    
    # 2. Prompt Engineering Phase
    persona_desc = PERSONAS.get(persona_name, PERSONAS["Research Assistant (Standard)"])
    template_str = get_prompt_template(strategy_name)
    
    formatted_prompt = template_str.format(
        system_persona=persona_desc,
        context=context_str if context_str else "No context retrieved.",
        question=query
    )
    
    # 3. Generation Phase
    llm = get_llm_instance(provider, api_key=api_key, model_name=model_name, temperature=temperature)
    
    if llm is None or provider == "Offline (Local / Extractive Mode)":
        answer = generate_extractive_answer(
            query=query,
            retrieved_chunks=retrieved,
            persona_name=persona_name,
            strategy_name=strategy_name
        )
        actual_model = "Local Extractive Engine (Offline / Safe Fallback)"
    else:
        try:
            response = llm.invoke(formatted_prompt)
            answer = response.content if hasattr(response, "content") else str(response)
            actual_model = f"{provider} ({model_name or 'Default'})"
        except Exception as e:
            # Fallback smoothly if API call encounters network/quota errors
            answer = (
                f"⚠️ *API Error encountered ({type(e).__name__}):* `{str(e)}`\n\n"
                f"**Switching automatically to Local Extractive Fallback to ensure demo continuity:**\n\n"
                + generate_extractive_answer(
                    query=query,
                    retrieved_chunks=retrieved,
                    persona_name=persona_name,
                    strategy_name=strategy_name
                )
            )
            actual_model = f"{provider} (Failed -> Local Fallback)"
            
    exec_time = time.time() - start_time
    
    return {
        "answer": answer,
        "formatted_prompt": formatted_prompt,
        "retrieved_chunks": retrieved,
        "execution_time": round(exec_time, 2),
        "model_used": actual_model,
        "top_k": top_k,
    }


def generate_document_summary(
    vector_store: FAISS,
    persona_name: str = "Executive Summarizer",
    provider: str = "Offline (Local / Extractive Mode)",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Generates an automated executive summary of the indexed document."""
    # Retrieve top 5 broad chunks across the document
    summary_query = "abstract introduction methodology results conclusion core contributions"
    retrieved = retrieve_relevant_chunks(vector_store, summary_query, top_k=6)
    context_str = format_context_string(retrieved)
    
    persona_desc = PERSONAS.get(persona_name, PERSONAS["Executive Summarizer"])
    formatted_prompt = SUMMARY_TEMPLATE.format(
        system_persona=persona_desc,
        context=context_str
    )
    
    llm = get_llm_instance(provider, api_key=api_key, model_name=model_name, temperature=0.1)
    
    if llm is None or provider == "Offline (Local / Extractive Mode)":
        answer = (
            "### 📄 Document Executive Overview (Extractive Mode)\n\n"
            "**Primary Topics & Contributions Identified:**\n"
        )
        for doc, _ in retrieved[:4]:
            p = doc.metadata.get("page", 1)
            first_sentence = doc.page_content.split(". ")[0]
            answer += f"- **[Page {p}]**: {first_sentence}.\n"
        answer += "\n*(Tip: Connect an API key in the sidebar for full AI synthesis.)*"
    else:
        try:
            resp = llm.invoke(formatted_prompt)
            answer = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            answer = f"Error generating summary: {str(e)}"
            
    return {
        "summary": answer,
        "formatted_prompt": formatted_prompt,
        "chunks_used": len(retrieved)
    }
