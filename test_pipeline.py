"""
Self-Verification Script for LangChain RAG & Prompt Engineering System.
Tests document loading, chunking, FAISS vector indexing, and query execution.
"""
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.document_loader import load_document
from src.text_splitter import split_documents
from src.vector_store import build_vector_store, retrieve_relevant_chunks
from src.rag_engine import execute_rag_query, generate_document_summary


def test_system():
    print("=" * 60)
    print(" 🚀 RUNNING SELF-VERIFICATION PIPELINE TEST")
    print("=" * 60)

    # 1. Test Document Loading
    sample_pdf = root_dir / "sample_data" / "sample_paper_rag.pdf"
    assert sample_pdf.exists(), f"Sample PDF missing at {sample_pdf}"
    print(f"\n[1/5] Loading document: {sample_pdf.name}...")
    docs = load_document(sample_pdf)
    print(f"      -> Successfully extracted {len(docs)} pages.")
    assert len(docs) > 0, "No pages extracted!"

    # 2. Test Text Splitting
    print("\n[2/5] Splitting document into semantic chunks...")
    chunks = split_documents(docs, chunk_size=600, chunk_overlap=100)
    print(f"      -> Created {len(chunks)} chunks with metadata preservation.")
    assert len(chunks) > 0, "No chunks created!"
    print(f"      -> Sample chunk #0 metadata: {chunks[0].metadata}")

    # 3. Test Vector Store & Local Embeddings
    print("\n[3/5] Building FAISS vector store with local CPU embeddings...")
    vector_store = build_vector_store(chunks, embedding_provider="local")
    print("      -> FAISS index created successfully.")

    # 4. Test Semantic Retrieval
    query = "What benchmark datasets were evaluated and what was the Exact Match score?"
    print(f"\n[4/5] Testing semantic retrieval for query: '{query}'...")
    retrieved = retrieve_relevant_chunks(vector_store, query, top_k=2)
    print(f"      -> Retrieved {len(retrieved)} relevant chunks.")
    for idx, (doc, score) in enumerate(retrieved, 1):
        print(f"         Chunk {idx} (Page {doc.metadata.get('page')}, Score: {score:.4f}):")
        print(f"         {doc.page_content[:120]}...\n")

    # 5. Test RAG Generation (Offline Mode)
    print("[5/5] Testing RAG Query Execution with Chain-of-Thought (Offline Fallback)...")
    result = execute_rag_query(
        query=query,
        vector_store=vector_store,
        persona_name="Research Assistant (Standard)",
        strategy_name="Chain-of-Thought (Step-by-Step Reasoning)",
        provider="Offline (Local / Extractive Mode)"
    )
    print(f"      -> Execution time: {result['execution_time']}s")
    print(f"      -> Model: {result['model_used']}")
    print("\n--- SAMPLE GENERATED ANSWER ---")
    print(result["answer"])
    print("--------------------------------\n")

    # 6. Test Document Summary
    print("[Bonus] Testing automated summary generation...")
    summary = generate_document_summary(vector_store, provider="Offline (Local / Extractive Mode)")
    print("      -> Summary generated successfully.")

    print("\n" + "=" * 60)
    print(" ✅ ALL PIPELINE TESTS PASSED! SYSTEM IS READY FOR DEMO.")
    print("=" * 60)


if __name__ == "__main__":
    test_system()
