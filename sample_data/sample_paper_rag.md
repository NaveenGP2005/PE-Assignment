# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

**Authors:** Patrick Lewis, Ethan Perez, Aleksandera Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela  
**Affiliations:** Facebook AI Research, University College London, New York University  

---

## 1. Abstract
Large pre-trained language models have been shown to store factual knowledge in their parameters, achieving state-of-the-art results on downstream NLP tasks. However, their ability to precisely access and manipulate knowledge is limited, and they often hallucinate when query knowledge is obscure or updated. In this paper, we explore a general-purpose fine-tuning recipe for Retrieval-Augmented Generation (RAG) — models which combine pre-trained parametric and non-parametric memory for language generation. We introduce RAG models where the parametric memory is a pre-trained seq2seq model (BART) and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever (DPR, Dense Passage Retriever). We compare two formulations: RAG-Sequence (which uses the same retrieved documents across the entire sequence) and RAG-Token (which can select different documents per generated token). We evaluate our models on a wide range of knowledge-intensive benchmarks, including Natural Questions, WebQuestions, CuratedTREC, and MS-MARCO. Our results demonstrate that RAG models set new state-of-the-art performance on open-domain question answering while generating more specific, diverse, and factual language than parametric-only seq2seq baselines.

---

## 2. Introduction & Problem Statement
Pre-trained neural language models like BERT, T5, and GPT encode substantial real-world factual information directly within their model weights. Despite their success, these models suffer from several fundamental limitations:
1. **Hallucination:** When faced with knowledge-intensive tasks, models frequently hallucinate non-existent citations or incorrect facts.
2. **Knowledge Freshness:** Updating world knowledge requires computationally expensive retraining of billions of parameters.
3. **Auditability & Provenance:** It is difficult to trace which specific training data influenced a given generated statement.

Retrieval-Augmented Generation (RAG) resolves these issues by decoupling knowledge retrieval from language generation. In RAG, external knowledge is stored in a non-parametric vector index. At query time, the system performs fast maximum inner product search (MIPS) using dense vector embeddings to retrieve top-$k$ relevant passages. The generator conditioned on both the input prompt and the retrieved context then generates the final response with direct provenance.

---

## 3. Methodology & Architecture
The RAG architecture consists of two primary components:
1. **Dense Passage Retriever (DPR):**
   - The query encoder $E_Q(q)$ and passage encoder $E_P(d)$ are based on BERT-base architectures.
   - The non-parametric knowledge base is indexed using Facebook AI Similarity Search (FAISS) over 21 million 100-word passages from English Wikipedia.
   - Relevance score is computed via dot product: $s(d, q) = E_Q(q)^T E_P(d)$.

2. **Sequence-to-Sequence Generator:**
   - We utilize BART-large, a 400M parameter bidirectional and auto-regressive seq2seq transformer.
   - The generator receives the concatenated input: `[Passage Context] + [User Query]` to autoregressively decode target tokens.

### RAG Formulations
We evaluate two distinct generation approaches:
- **RAG-Sequence Model:** The retriever draws top-$K$ candidate documents for the prompt. The generator produces a complete candidate sequence for each document, and probabilities are marginalized across all $K$ sequences.
- **RAG-Token Model:** The retriever draws top-$K$ documents, and the generator computes token-level probabilities marginalized across the documents at every single step, allowing the model to aggregate facts from distinct passages during generation.

---

## 4. Prompt Engineering & Conditioning
Conditioning the generative model effectively is critical for output accuracy:
- **Zero-Shot Evaluation:** Direct query-context concatenation without fine-tuning prompts.
- **In-Context Guidance:** Providing explicit formatting cues instructing the model to prioritize context and decline answering when documents disagree.
- **Context Truncation:** We truncate retrieved documents to 100 tokens each and concatenate up to $K=5$ documents into the generator's context window.

---

## 5. Experimental Results
### Open-Domain Question Answering
We evaluate on standard benchmarks measuring Exact Match (EM) accuracy:
- **Natural Questions (NQ):** RAG-Sequence achieves **44.5% EM**, outperforming REALM (40.4%) and closed-book T5-11B (34.5%).
- **WebQuestions (WQ):** RAG-Token achieves **45.5% EM**.
- **CuratedTREC (CT):** RAG-Token achieves **61.0% EM**.

### Generation Quality & Factuality
Human evaluation and automated metrics (Jeopardy question generation) show that RAG outputs are:
1. Rated substantially more factual and accurate than BART baselines (over 70% preference in pairwise human evaluations).
2. Significantly less prone to hallucination because factual tokens are grounded in the retrieved Wikipedia passages.

---

## 6. Conclusion
RAG models merge the fluency and reasoning capacity of pre-trained parametric models with the factual accuracy, updateability, and interpretability of external vector retrieval. By grounding generation in retrieved evidence, RAG provides a practical, auditable framework for knowledge-intensive AI applications.
