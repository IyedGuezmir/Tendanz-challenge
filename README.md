# Tendanz Legal RAG Chatbot

## Project Overview

**Tendanz Legal RAG Chatbot** is a Retrieval-Augmented Generation (RAG) system designed to answer legal questions accurately using a combination of vector search, embeddings, and language models. The system leverages **structured document chunking**, **retrieval pipelines**, and **fusion-based reasoning** to produce precise, contextually grounded answers.

Key features:  

- Handles **legal documents** with attention to formal and precise language.
- Supports multiple **RAG architectures**, enabling experimentation with hybrid retrieval, query decomposition, and fusion strategies.
- Generates answers **with references** to retrieved documents for faithfulness and traceability.
- Integrates with a **vector database (Qdrant)** for efficient semantic search using BM25 and OpenAI embeddings.

---

## Project Structure
````
````

---

## Document Chunking Strategy

Accurate retrieval requires **smart document splitting**. Our pipeline uses a **two-tiered chunking strategy**:

1. **Main Titles:** Split documents based on the primary legal sections (headings). This ensures that each chunk aligns with the logical structure of the law or legal text.
2. **Subtitles / Smaller Sections:** Apply **recursive character splitting** on smaller headings.  
   - **Maximum characters per chunk:** 3,000  
   - **Overlap:** 200 characters  

This hybrid approach preserves **semantic integrity** while allowing the vector database to index meaningful, retrievable pieces of text.

---

## RAG Architectures

We experimented with **three different RAG-based architectures**:

### 1. Hybrid Retrieval

- **Pipeline:** `Query → Hybrid Retrieval → Cohere Reranker → LLM answer`  
- **Vector Index:** Combination of **BM25 + OpenAI embeddings**  
- **Reasoning:** This baseline allows for **semantic search with keyword matching** and provides high recall for legal queries.  
- **Pros:** Simple, fast, interpretable.  
- **Cons:** May miss nuanced answers if the query is phrased differently from document text.

---

### 2. Hybrid with Query Decomposition
![Hybrid with Query Decomposition](Architecture/rag_decomp.png)
- **Pipeline:** `Query → Decompose → Hybrid Retrieval → LLM`  
- **Mechanism:** Break complex questions into sub-queries to retrieve relevant sections individually.  
- **Reasoning:** Legal queries often contain multiple clauses or conditions; decomposition improves **retrieval precision**.  
- **Pros:** Better coverage of multi-faceted questions, reduces hallucinations.  
- **Cons:** More API calls, slightly increased latency.

---

### 3. Hybrid with RAG-Fusion
![Hybrid with Rag Fusion](/Architecture/rag_fusion.png)
- **Pipeline:** `Query → Generate related queries → Hybrid retrieval for each → Reciprocal Rank Fusion → LLM answer`  
- **Mechanism:** Multiple queries are generated automatically from the user question. Each query retrieves top-k documents, which are then **fused using Reciprocal Rank Fusion (RRF)** to rank the most relevant pieces.  
- **Reasoning:** Combining multiple perspectives increases **answer faithfulness and contextual richness**.  
- **Pros:** Strongest retrieval precision, particularly for **complex legal questions**.  
- **Cons:** Higher computational cost, slightly slower response time.

---

## Design Decisions

- **Hybrid Retrieval:** Combines **BM25 keyword search** with **OpenAI embeddings** to handle both lexical and semantic matching.
- **Recursive Character Splitting:** Ensures that chunk size is manageable for embeddings and LLM context windows while preserving context overlap.
- **Query Decomposition:** Splits complex questions to improve retrieval accuracy in multi-condition queries.
- **RAG-Fusion:** Uses **Reciprocal Rank Fusion** to merge multiple retrieval results, improving ranking consistency and reducing noise.

---

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/yourusername/tendanz-rag-chatbot.git
cd tendanz-rag-chatbot
```
2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. Launch the Streamlit interface:

```bash
streamlit run app/streamlit_app.py

```