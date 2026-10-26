"""
Generates a multi-page, publication-quality sample academic PDF paper using ReportLab.
"""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak


def generate_academic_pdf(output_path: Path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        alignment=1, # Center
        spaceAfter=10
    )
    
    author_style = ParagraphStyle(
        'DocAuthors',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        alignment=1,
        spaceAfter=20
    )
    
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=13,
        leading=16,
        spaceBefore=14,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontSize=10,
        leading=15,
        spaceAfter=8
    )

    story = []

    # --- PAGE 1: TITLE, ABSTRACT, INTRODUCTION ---
    story.append(Paragraph("Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", title_style))
    story.append(Paragraph("Patrick Lewis, Ethan Perez, Aleksandera Piktus, Fabio Petroni, et al.<br/><i>Facebook AI Research, University College London, New York University</i>", author_style))
    
    story.append(Paragraph("1. Abstract", heading_style))
    story.append(Paragraph(
        "Large pre-trained language models store substantial factual knowledge within their parameters, "
        "achieving state-of-the-art results on NLP benchmarks. However, their ability to precisely access "
        "and manipulate knowledge is inherently limited, and they frequently hallucinate when answering "
        "niche or updated queries. In this paper, we explore a fine-tuning framework for Retrieval-Augmented "
        "Generation (RAG) — models that integrate pre-trained parametric and non-parametric memory for "
        "open-domain text generation. We introduce RAG architectures combining a BART generator with a dense "
        "vector index of Wikipedia accessed via Dense Passage Retriever (DPR). We show that RAG sets new "
        "state-of-the-art results on Natural Questions and CuratedTREC, while generating more factual and "
        "verifiably grounded responses.", body_style
    ))
    
    story.append(Paragraph("2. Introduction & Fundamental Problem Statement", heading_style))
    story.append(Paragraph(
        "Pre-trained language models encode world knowledge directly into their neural weights. "
        "Despite dramatic fluency, closed-book models exhibit three critical vulnerabilities:<br/>"
        "<b>1. Hallucination:</b> Models generate convincing falsehoods on knowledge-intensive questions.<br/>"
        "<b>2. Knowledge Obsolescence:</b> World knowledge updates require re-training billions of parameters.<br/>"
        "<b>3. Absence of Provenance:</b> Outputs lack explicit citations or traceable evidence.<br/>"
        "Retrieval-Augmented Generation (RAG) resolves these limitations by decoupling memory into parametric "
        "weights and an auditable non-parametric vector database.", body_style
    ))
    
    story.append(PageBreak())

    # --- PAGE 2: ARCHITECTURE & METHODOLOGY ---
    story.append(Paragraph("3. System Architecture & Formulations", heading_style))
    story.append(Paragraph(
        "The RAG architecture unites two distinct components:<br/>"
        "<b>A. Dense Passage Retriever (DPR):</b> Employs BERT-base query encoder E_Q(q) and passage encoder E_P(d). "
        "Passages are indexed into a dense vector space using Facebook AI Similarity Search (FAISS) for fast "
        "Maximum Inner Product Search (MIPS). The similarity score is computed as:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<i>s(d, q) = dot_product(E_Q(q), E_P(d))</i><br/>"
        "Top-K (typically K=5) passages with highest similarity are retrieved.<br/>"
        "<b>B. Sequence-to-Sequence Generator:</b> Built on BART-large (400 million parameters). The generator "
        "is conditioned jointly on the user's prompt and concatenated retrieved passage contexts.", body_style
    ))
    
    story.append(Paragraph("4. RAG-Sequence vs RAG-Token Formulations", heading_style))
    story.append(Paragraph(
        "We propose and contrast two distinct generation mechanisms:<br/>"
        "• <b>RAG-Sequence Model:</b> The retriever returns top-K documents, and the generator produces a candidate "
        "output sequence for each document. The final sequence probability is marginalized across all K documents.<br/>"
        "• <b>RAG-Token Model:</b> The generator marginalizes token probabilities across documents at each individual "
        "decoding step. This allows the model to synthesize disparate facts across multiple retrieved passages within "
        "a single sentence.", body_style
    ))
    
    story.append(PageBreak())

    # --- PAGE 3: PROMPT ENGINEERING & EXPERIMENTAL RESULTS ---
    story.append(Paragraph("5. Prompt Engineering & Grounding Strategies", heading_style))
    story.append(Paragraph(
        "Conditioning the generator effectively is vital to suppress hallucinations:<br/>"
        "• <b>In-Context Citation Guidance:</b> Explicit formatting instructions force the model to append "
        "source page identifiers ([Page X]) to every factual claim.<br/>"
        "• <b>Strict Refusal Guardrails:</b> Prompts instructing the model to decline answering if the retrieved "
        "passages do not contain conclusive evidence reduce ungrounded answers by over 60%.<br/>"
        "• <b>Few-Shot Exemplars:</b> Prepending input-output exemplars with source attribution yields immediate "
        "adherence to strict scholarly formatting.", body_style
    ))
    
    story.append(Paragraph("6. Experimental Benchmark Results", heading_style))
    story.append(Paragraph(
        "We evaluate open-domain question answering performance using Exact Match (EM) accuracy:<br/>"
        "• <b>Natural Questions (NQ):</b> RAG-Sequence achieves <b>44.5% EM</b>, significantly outperforming "
        "closed-book T5-11B (34.5%) and REALM (40.4%).<br/>"
        "• <b>WebQuestions (WQ):</b> RAG-Token achieves <b>45.5% EM</b>.<br/>"
        "• <b>CuratedTREC:</b> RAG-Token achieves <b>61.0% EM</b>.<br/>"
        "Pairwise human evaluations show that RAG outputs are preferred for factuality in over 70% of responses.", body_style
    ))
    
    story.append(Paragraph("7. Conclusion", heading_style))
    story.append(Paragraph(
        "RAG successfully combines the reasoning dexterity of parametric neural generators with the factual "
        "integrity, explainability, and updateability of vector retrieval. It provides a robust paradigm for "
        "transparent AI systems.", body_style
    ))

    doc.build(story)


if __name__ == "__main__":
    out_file = Path(__file__).parent / "sample_paper_rag.pdf"
    generate_academic_pdf(out_file)
    print(f"Generated clean academic PDF at: {out_file}")
