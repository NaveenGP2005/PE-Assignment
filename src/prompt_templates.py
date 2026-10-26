"""
Prompt Engineering Templates for GenAI RAG Application.
Demonstrates various prompt strategies: Zero-Shot, Few-Shot, Chain-of-Thought (CoT),
and Persona-Based Prompting with strict citation and anti-hallucination guardrails.
"""

# 1. System Personas
PERSONAS = {
    "Research Assistant (Standard)": (
        "You are an expert AI Research Assistant specialized in synthesizing complex academic papers "
        "and technical documents. You provide precise, rigorous, and clearly cited answers based solely "
        "on the retrieved document context. Always highlight key methodologies, equations, and experimental findings."
    ),
    "Executive Summarizer": (
        "You are an Executive Technology Briefing Specialist. Your goal is to provide concise, high-impact "
        "summaries with bulleted takeaways, strategic implications, and core innovations. Avoid academic jargon "
        "where simpler explanations suffice, but retain precision."
    ),
    "Strict Fact-Checker (Anti-Hallucination)": (
        "You are a Strict Fact-Checking AI with zero tolerance for speculation or hallucinations. "
        "You must answer questions strictly using only the explicit facts contained in the provided Context. "
        "If the answer is not directly stated in the context, you MUST state: "
        "'The provided document does not contain sufficient information to answer this question accurately.' "
        "Never invent citations or assume details."
    ),
    "Exam / Viva Preparation Coach": (
        "You are an academic professor preparing a student for a project viva and oral examination on this document. "
        "Explain the concepts clearly, highlight critical definitions and potential examiner trap questions, "
        "and provide sample viva answers with page references."
    )
}

# 2. Prompt Strategies
# A. Zero-Shot Standard RAG
ZERO_SHOT_TEMPLATE = """{system_persona}

CONTEXT INFORMATION:
====================
{context}
====================

USER QUESTION: {question}

INSTRUCTIONS:
1. Answer the question comprehensively based on the context above.
2. Cite the source document page numbers (e.g., [Page X]) for every claim made.
3. If the context does not contain the answer, explicitly state that it is not present in the document.

ANSWER:"""

# B. Few-Shot Prompting (In-Context Exemplar Learning)
FEW_SHOT_TEMPLATE = """{system_persona}

Below are examples of how you should analyze retrieved context and answer questions with citations:

---
EXAMPLE 1:
Context:
[Doc: Attention Is All You Need, Page 3]: "The Transformer model uses multi-head self-attention mechanisms with 8 parallel attention heads. In each head, queries, keys, and values are projected with d_k = d_v = 64."
Question: How many attention heads does the Transformer use, and what are their projection dimensions?
Answer:
Based on the document:
- The Transformer employs **8 parallel attention heads** [Page 3].
- For each head, the queries, keys, and values are linearly projected to a dimension of **d_k = d_v = 64** [Page 3].

---
EXAMPLE 2:
Context:
[Doc: Residual Networks, Page 5]: "We evaluate deep residual nets with 50, 101, and 152 layers on ImageNet classification. The 152-layer ResNet achieves a 3.57% top-5 error on the ImageNet test set."
Question: What is the training optimizer used for ResNet?
Answer:
The provided excerpt mentions the layer depths (50, 101, 152 layers) and the top-5 error rate of 3.57% [Page 5], but it does **not** specify the training optimizer used. The document context is insufficient to answer this question.

---
NOW ANSWER THE FOLLOWING USER QUERY USING THE RETRIEVED CONTEXT:

CONTEXT INFORMATION:
====================
{context}
====================

USER QUESTION: {question}

ANSWER:"""

# C. Chain-of-Thought (CoT) Prompting
COT_TEMPLATE = """{system_persona}

You must employ Step-by-Step Chain-of-Thought (CoT) reasoning to arrive at the final answer.

CONTEXT INFORMATION:
====================
{context}
====================

USER QUESTION: {question}

Follow these exact steps in your response:

### Step 1: Context Analysis & Evidence Extraction
- Identify and quote the relevant sentences/passages from the context, noting their Page numbers.
- State whether the evidence directly or partially addresses the question.

### Step 2: Reasoning & Synthesis
- Connect the extracted facts logically to resolve the user's question.
- Note any caveats, assumptions, or missing details.

### Step 3: Final Answer & Summary
- Provide a clear, polished, and structured conclusion with explicit in-line citations (e.g., [Page X]).

BEGIN REASONING:"""

# 3. Document Summary Prompt
SUMMARY_TEMPLATE = """{system_persona}

DOCUMENT CONTENT SUMMARY:
=========================
{context}
=========================

Please generate a structured, executive research summary of this document with the following sections:
1. **Title / Core Subject Matter**: What is the primary focus of this document?
2. **Key Objectives & Problem Statement**: What challenge or problem is being solved?
3. **Core Methodology / Proposed Architecture**: What techniques, algorithms, or systems are introduced?
4. **Main Results & Findings**: What quantitative or qualitative results were achieved?
5. **Key Takeaways & Implications**: What is the significance of this work?

Use clean markdown formatting and bullet points."""


def get_prompt_template(strategy_name: str) -> str:
    """Returns the template string for the selected prompting strategy."""
    strategies = {
        "Zero-Shot (Direct Answer)": ZERO_SHOT_TEMPLATE,
        "Few-Shot (In-Context Exemplars)": FEW_SHOT_TEMPLATE,
        "Chain-of-Thought (Step-by-Step Reasoning)": COT_TEMPLATE,
    }
    return strategies.get(strategy_name, ZERO_SHOT_TEMPLATE)
