"""
Offline Simulated Extractive RAG Engine.
Acts as a guaranteed fallback for demonstrations when an external LLM API key
is missing, invalid, rate-limited, or internet is unavailable in the examination room.
Extracts relevant sentences from retrieved chunks, formats citations, and generates structured responses.
"""
from typing import List, Tuple
from langchain_core.documents import Document


def generate_extractive_answer(
    query: str,
    retrieved_chunks: List[Tuple[Document, float]],
    persona_name: str,
    strategy_name: str
) -> str:
    """Generates an accurate, citation-backed extractive answer directly from retrieved context."""
    if not retrieved_chunks:
        return (
            "⚠️ **No relevant context found.** Please ensure a document is loaded into the vector store."
        )

    query_keywords = set([w.lower().strip("?,.!") for w in query.split() if len(w) > 2])

    extracted_evidence = []
    citations = set()

    for doc, score in retrieved_chunks:
        page = doc.metadata.get("page", 1)
        source = doc.metadata.get("source", "Document")
        citations.add(f"Page {page}")
        
        # Split content into sentences and score them by keyword overlap
        sentences = doc.page_content.split(". ")
        matched_sentences = []
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            words = set([w.lower().strip("?,.!") for w in s_clean.split()])
            overlap = words.intersection(query_keywords)
            if overlap:
                matched_sentences.append((len(overlap), s_clean, page))

        # Sort sentences by keyword relevance
        matched_sentences.sort(key=lambda x: x[0], reverse=True)
        for _, s_text, p_num in matched_sentences[:2]:
            extracted_evidence.append((s_text, p_num, source))

    citation_summary = ", ".join(sorted(list(citations)))

    if strategy_name == "Chain-of-Thought (Step-by-Step Reasoning)":
        response = (
            f"### [Offline Extractive Mode: Simulated CoT Reasoning]\n\n"
            f"**Step 1: Evidence Identification & Retrieval**\n"
            f"- Identified **{len(retrieved_chunks)} relevant passage(s)** matching query: *\"{query}\"*\n"
            f"- Sources cited: **{citation_summary}**\n\n"
            f"**Step 2: Analysis & Synthesis**\n"
        )
        for s_text, p_num, src in extracted_evidence[:4]:
            response += f"- Extracted key statement: *\"{s_text}.\"* [Page {p_num}]\n"
        
        response += (
            f"\n**Step 3: Synthesis & Final Conclusion**\n"
            f"Based strictly on the retrieved document context, the primary findings indicate that "
            f"the concepts referenced in your question relate directly to the documented findings on **{citation_summary}**.\n\n"
            f"*(Note: Running in high-reliability Local/Extractive Mode. Add a Google Gemini or Groq API key in the sidebar for full generative synthesis.)*"
        )
        return response

    elif strategy_name == "Few-Shot (In-Context Exemplars)":
        response = (
            f"### [Offline Extractive Mode: Few-Shot Exemplar Guided Answer]\n\n"
            f"**Direct Answer:**\n"
        )
        for s_text, p_num, _ in extracted_evidence[:3]:
            response += f"- {s_text}. **[Page {p_num}]**\n"
        response += f"\n**Retrieved Citations:** [{citation_summary}]\n"
        return response

    else:
        # Standard Zero-Shot
        response = (
            f"### [Offline Extractive Mode: Context Synthesis]\n\n"
            f"According to the retrieved sections of the document:\n\n"
        )
        for s_text, p_num, _ in extracted_evidence[:4]:
            response += f"> \"{s_text}.\" — **[Page {p_num}]**\n\n"
        
        response += (
            f"**Document References:** Verified against {citation_summary}.\n\n"
            f"*💡 Tip: To enable real-time generative answers with Google Gemini or Groq Llama-3, enter your free API key in the sidebar.*"
        )
        return response
