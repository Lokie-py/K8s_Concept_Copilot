# K8s Concept Copilot 🚀

An **evaluation-driven Kubernetes RAG system** designed to answer Kubernetes concept questions using a combination of **query translation, dense vector retrieval, BM25 lexical retrieval, hybrid retrieval, reranking, and Groq-based generation**.

The system is built around a simple principle:

> **Retrieve the right Kubernetes knowledge first, then generate the answer from that retrieved context.**

It also includes separate **retrieval and generation evaluation pipelines** to measure how well the system performs.

---

## 🧠 Overview

Kubernetes documentation contains many closely related concepts such as:

* Pods
* Resource requests and limits
* Scheduling
* PodGroups
* Gang scheduling
* Preemption
* Eviction
* Node failures
* Workload controllers

A general-purpose LLM may know many of these concepts, but relying entirely on model knowledge can result in:

* Unsupported claims
* Missing technical details
* Outdated information
* Poor retrieval of specialized concepts
* Difficulty measuring system quality

K8s Concept Copilot addresses this by retrieving relevant documentation before generating an answer.

```text
User Question
      │
      ▼
Query Translation
      │
      ▼
Multiple Search Queries
      │
      ├──────────────────┐
      ▼                  ▼
Vector Retrieval      BM25 Retrieval
      │                  │
      └────────┬─────────┘
               ▼
        Hybrid Retrieval
               │
               ▼
           Reranking
               │
               ▼
       Context Extraction
               │
               ▼
         Prompt Builder
               │
               ▼
           Groq LLM
               │
               ▼
          Final Answer
```

---

# 🏗️ System Architecture

The system consists of two major pipelines:

### 1. Document Ingestion Pipeline

Transforms Kubernetes documentation into searchable representations.

### 2. Online RAG Pipeline

Translates the user's question, retrieves relevant chunks, reranks them, and generates a grounded answer.

```text
                    DOCUMENT INGESTION
                           │
                           ▼
                    Kubernetes PDFs
                           │
                           ▼
                      PDF Loader
                           │
                           ▼
                      Text Cleaner
                           │
                           ▼
                        Chunker
                           │
                    ┌──────┴──────┐
                    ▼             ▼
                 BM25 Index   BGE Embeddings
                                  │
                                  ▼
                             Vector Store


                      ONLINE RAG PIPELINE

                         User Query
                              │
                              ▼
                       Query Translator
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
             Vector Retrieval      BM25 Retrieval
                    │                   │
                    └─────────┬─────────┘
                              ▼
                       Hybrid Retrieval
                              │
                              ▼
                           Reranker
                              │
                              ▼
                     Top-K Relevant Chunks
                              │
                              ▼
                     Context Extraction
                              │
                              ▼
                       Prompt Builder
                              │
                              ▼
                          Groq LLM
                              │
                              ▼
                         Final Answer
```

---

# 📥 1. Document Ingestion

The ingestion pipeline prepares the Kubernetes documentation for retrieval.

```text
PDF
 ↓
PDF Loader
 ↓
Text Cleaning
 ↓
Chunking
 ↓
 ┌───────────────┬────────────────┐
 ▼               ▼
BM25 Index    BGE Embeddings
                  │
                  ▼
             Vector Store
```

The ingestion components are located in:

```text
src/ingestion/
```

### PDF Loader

Loads Kubernetes documentation from PDF sources.

### Text Cleaner

Cleans extracted document text before indexing.

### Chunker

Splits documents into retrieval-friendly chunks.

### BGE Embeddings

Converts chunks into dense vector representations for semantic search.

### BM25 Index

Creates a lexical retrieval representation for keyword-based search.

---

# 🔎 2. Query Translation

The original user question is expanded into multiple search queries.

For example:

```text
Original:

What is a Kubernetes Pod and what resources do containers
inside a Pod share?
```

The system generates alternative retrieval queries such as:

```text
kubernetes pod container resource sharing

kubernetes pod resource sharing mechanisms

kubernetes pod container resource sharing concepts

container resource sharing in kubernetes pods
```

The goal is to improve retrieval coverage by representing the same information need through multiple formulations.

The query translation component is located at:

```text
src/retrieval/query_translator.py
```

The evaluation pipeline demonstrates this translation process across the evaluation dataset.

---

# 🧮 3. Dense Vector Retrieval

The translated queries are converted into embeddings using the BGE embedding model.

```text
Query
 ↓
BGE Embedding
 ↓
Query Vector
 ↓
Vector Similarity Search
 ↓
Candidate Chunks
```

Dense retrieval is useful when the query and documentation express the same concept using different wording.

For example:

```text
"node capacity"

and

"allocatable CPU and memory"
```

may be conceptually related even if the exact words differ.

The vector retrieval implementation is located at:

```text
src/retrieval/vector_retrieval.py
```

---

# 🔤 4. BM25 Retrieval

The system also performs lexical retrieval using BM25.

```text
Query
 ↓
Token / Term Matching
 ↓
BM25 Scoring
 ↓
Candidate Chunks
```

BM25 is particularly useful for Kubernetes terminology where exact terms can carry significant meaning:

```text
PodGroup
minCount
preemption
allocatable
resource requests
gang scheduling
```

The BM25 components are located in:

```text
src/ingestion/bm25.py
src/retrieval/bm25_retrieval.py
```

---

# 🔀 5. Hybrid Retrieval

Dense and lexical retrieval are combined to produce a stronger candidate set.

```text
                    Query
                      │
             ┌────────┴────────┐
             ▼                 ▼
      Vector Retrieval      BM25 Retrieval
             │                 │
             ▼                 ▼
       Semantic Results    Lexical Results
             │                 │
             └────────┬────────┘
                      ▼
                Result Fusion
                      │
                      ▼
              Hybrid Candidates
```

This approach combines:

| Retrieval method | Strength                    |
| ---------------- | --------------------------- |
| Vector Retrieval | Semantic similarity         |
| BM25             | Exact/lexical terminology   |
| Hybrid Retrieval | Combination of both signals |

The main hybrid retrieval implementation is:

```text
src/retrieval/hybrid_retrieval.py
```

---

# 🎯 6. Reranking

After hybrid retrieval produces candidate chunks, the candidates are reranked to improve the ordering of the most relevant results.

```text
Hybrid Candidates
       │
       ▼
   Reranker
       │
       ▼
Better Ranked Context
```

This creates a second-stage retrieval process:

```text
First Stage
───────────
BM25 + Vector
       ↓
Candidate Retrieval


Second Stage
────────────
Reranking
       ↓
Final Top-K Context
```

The reranking implementation is located at:

```text
src/retrieval/reranker.py
```

---

# 📚 7. Context Extraction

Once the final retrieval results are available, the relevant content is extracted and prepared for answer generation.

```text
Retrieved Chunks
      ↓
Context Extraction
      ↓
Generation Context
```

This component is located at:

```text
src/generation/context_extractor.py
```

---

# 🧩 8. Prompt Construction

The prompt builder combines the question with the retrieved context.

```text
Question
    +
Retrieved Context
    ↓
Prompt
```

The prompt construction logic is located at:

```text
src/generation/prompt_builder.py
```

This keeps prompt construction separate from the LLM client and makes the generation stage easier to modify and evaluate.

---

# 🤖 9. Answer Generation

The final prompt is passed to a **Groq-hosted LLM**.

```text
Question
   +
Retrieved Kubernetes Context
          │
          ▼
      Prompt Builder
          │
          ▼
       Groq LLM
          │
          ▼
     Final Answer
```

The generation components are located in:

```text
src/generation/
```

Relevant files include:

```text
answer_generator.py
llm_client.py
prompt_builder.py
context_extractor.py
```

The model is used to synthesize an answer from the retrieved Kubernetes documentation.

---

# 📊 Evaluation Framework

A major part of the project is evaluating the RAG pipeline rather than only demonstrating that it can generate answers.

The evaluation is divided into two stages:

```text
                    Evaluation Dataset
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
      Retrieval Evaluation      Generation Evaluation
              │                         │
              ▼                         ▼
        Recall@5 / MRR             LLM-as-a-Judge
                                        │
                              ┌─────────┼─────────┐
                              ▼         ▼         ▼
                         Faithfulness Relevancy Correctness
```

The evaluation dataset contains **20 human-labeled queries**.

---

# 📈 Retrieval Evaluation

The retrieval pipeline is evaluated using:

* Recall@5
* Mean Reciprocal Rank (MRR)
* Rank-1 relevance
* Retrieval misses

### Current Results

| Metric                   |          Result |
| ------------------------ | --------------: |
| Evaluation queries       |          **20** |
| Human-labeled queries    |          **20** |
| Recall@5                 | **1.00 / 100%** |
| MRR                      |        **0.97** |
| Relevant chunk at Rank 1 |       **19/20** |
| Retrieval misses         |        **0/20** |

### Interpretation

**Recall@5 = 1.00**

All 20 evaluation queries had relevant information retrieved within the top five results.

**MRR = 0.97**

Relevant information was generally ranked very highly.

**Rank-1 relevance = 19/20**

19 out of the 20 evaluation queries had a relevant result at the first position.

This indicates that the retrieval layer is currently a strong part of the system.

---

# 🧪 Generation Evaluation

Generation quality is evaluated using an **LLM-as-a-Judge** approach.

The judge receives:

```text
Question
    +
Expected Key Points
    +
Retrieved Context
    +
Generated Answer
```

It evaluates the answer on three dimensions.

### Faithfulness

Are the claims in the generated answer supported by the retrieved context?

### Relevancy

Does the answer directly address the question?

### Correctness

Does the answer cover the expected key points?

Each score ranges from **0 to 1**.

---

# 📊 Generation Results

The current generation evaluation covers **20 queries**.

| Metric       |    Score |
| ------------ | -------: |
| Faithfulness | **0.91** |
| Relevancy    | **0.93** |
| Correctness  | **0.83** |

These are the aggregate results from the current evaluation run.

### Interpretation

**Faithfulness — 0.91**

The generated answers are generally grounded in the retrieved context.

**Relevancy — 0.93**

The system generally produces answers that directly address the user's question.

**Correctness — 0.83**

Correctness is currently the weakest generation metric, suggesting that some answers do not fully cover all expected key points even when relevant context is retrieved.

This makes **generation completeness and answer synthesis** the main area for future improvement.

---

# 📁 Project Structure

```text
Doc-copilot/
│
├── Data/
│   └── Kubernetes documentation / source data
│
├── eval_sets/
│   └── eval_set.json
│
├── rag/
│   └── Python virtual environment
│
├── src/
│   │
│   ├── evals/
│   │   ├── generation_metric.py
│   │   └── retrieval_metric.py
│   │
│   ├── generation/
│   │   ├── answer_generator.py
│   │   ├── context_extractor.py
│   │   ├── llm_client.py
│   │   └── prompt_builder.py
│   │
│   ├── ingestion/
│   │   ├── bm25.py
│   │   ├── chunker.py
│   │   ├── pdf_loader.py
│   │   ├── text_cleaner.py
│   │   └── vector_store.py
│   │
│   ├── retrieval/
│   │   ├── bm25_retrieval.py
│   │   ├── embedder.py
│   │   ├── hybrid_retrieval.py
│   │   ├── query_translator.py
│   │   ├── reranker.py
│   │   └── vector_retrieval.py
│   │
│   └── config.py
│
├── .env
├── .gitignore
├── DATA_LICENSE.md
├── LICENSE
├── Readme.md
└── requirements.txt
```

---

# 🛠️ Tech Stack

| Component          | Technology               |
| ------------------ | ------------------------ |
| Language           | Python                   |
| RAG Framework      | LangChain                |
| Embeddings         | BGE                      |
| Dense Retrieval    | Vector Search            |
| Sparse Retrieval   | BM25                     |
| Retrieval Strategy | Hybrid Retrieval         |
| Reranking          | Custom Reranker          |
| Query Expansion    | Query Translation        |
| Generation         | Groq                     |
| Evaluation         | LLM-as-a-Judge           |
| Data Source        | Kubernetes Documentation |

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone <repository-url>
cd Doc-copilot
```

## 2. Create a virtual environment

```bash
python -m venv rag
```

Windows:

```powershell
.\rag\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure environment variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

---

# ▶️ Running the Pipeline

## Document Ingestion

The ingestion pipeline processes the Kubernetes documentation and prepares the retrieval indexes.

The relevant modules are:

```text
src/ingestion/
```

The pipeline includes:

```text
PDF Loading
     ↓
Text Cleaning
     ↓
Chunking
     ↓
 ┌──────────────┬───────────────┐
 ▼              ▼
BM25          BGE Embeddings
Index            │
                 ▼
            Vector Store
```

---

## Retrieval Evaluation

Run:

```bash
python -m src.evals.retrieval_metric
```

This evaluates retrieval performance using the labeled evaluation dataset.

---

## Generation Evaluation

Run:

```bash
python -m src.evals.generation_metric
```

The evaluation pipeline performs:

```text
Evaluation Query
       ↓
Query Translation
       ↓
Vector Retrieval + BM25
       ↓
Hybrid Retrieval
       ↓
Reranking
       ↓
Context Extraction
       ↓
Prompt Construction
       ↓
Groq Generation
       ↓
LLM-as-a-Judge
       ↓
Faithfulness / Relevancy / Correctness
```

---

# 🎯 Design Goals

### High Retrieval Quality

Retrieve relevant Kubernetes documentation consistently.

### Semantic + Lexical Search

Combine semantic vector search with exact terminology matching.

### Strong Ranking

Use reranking to improve the ordering of retrieved candidates.

### Grounded Generation

Generate answers using retrieved Kubernetes documentation as context.

### Measurable Performance

Evaluate retrieval and generation independently.

### Iterative Development

Use evaluation results to identify whether improvements should target:

```text
Retrieval
   ↓
Ranking
   ↓
Context
   ↓
Generation
```

---

# 🔬 Key Engineering Decisions

## Why Query Translation?

A single query formulation may not capture every relevant way a concept is represented in the documentation.

Multiple translated queries increase retrieval coverage.

## Why BM25 + Vector Search?

Vector search is strong at semantic similarity, while BM25 is strong at terminology and lexical matching.

Kubernetes contains many domain-specific terms where exact matching matters, so combining both approaches provides complementary retrieval signals.

## Why Reranking?

Initial retrieval prioritizes recall. Reranking then improves the ordering of the candidate results before they are passed to the generation stage.

## Why Separate Retrieval and Generation Evaluation?

A RAG system can fail in two fundamentally different ways:

```text
Relevant information not retrieved
              ↓
        Retrieval problem
```

or:

```text
Relevant information retrieved
              ↓
       Bad answer generated
              ↓
        Generation problem
```

Evaluating these stages independently makes debugging and optimization much more systematic.

---

# 🔮 Future Improvements

* Add more robust reranking experiments.
* Improve answer completeness and correctness.
* Analyze low-scoring queries individually.
* Add source citations to generated answers.
* Add LangSmith observability.
* Track latency and token usage.
* Add automated regression evaluation.
* Experiment with different BGE embedding models.
* Tune BM25/vector retrieval weighting.
* Compare different reranking strategies.
* Compare independent LLM judges.
* Add a FastAPI serving layer.
* Containerize the application using Docker.

---

# 📌 Current Performance

```text
╔══════════════════════════════════════╗
║         RETRIEVAL PERFORMANCE        ║
╠══════════════════════════════════════╣
║ Recall@5              100%           ║
║ MRR                    97%           ║
║ Rank-1 Relevance       95%           ║
║ Retrieval Misses        0%           ║
╚══════════════════════════════════════╝


╔══════════════════════════════════════╗
║        GENERATION PERFORMANCE        ║
╠══════════════════════════════════════╣
║ Faithfulness           91%           ║
║ Relevancy              93%           ║
║ Correctness            83%           ║
╚══════════════════════════════════════╝
```

---

# 🧠 Final Architecture

```text
                    ┌─────────────────────┐
                    │ Kubernetes Documents│
                    └──────────┬──────────┘
                               │
                               ▼
                         PDF Loader
                               │
                               ▼
                         Text Cleaner
                               │
                               ▼
                           Chunker
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
               BM25 Index           BGE Embeddings
                                          │
                                          ▼
                                     Vector Store


                         ┌──────────────┐
                         │ User Query   │
                         └──────┬───────┘
                                │
                                ▼
                       Query Translation
                                │
                     ┌──────────┴──────────┐
                     ▼                     ▼
                BM25 Retrieval       Vector Retrieval
                     │                     │
                     └──────────┬──────────┘
                                ▼
                         Hybrid Retrieval
                                │
                                ▼
                            Reranker
                                │
                                ▼
                       Context Extraction
                                │
                                ▼
                         Prompt Builder
                                │
                                ▼
                           Groq LLM
                                │
                                ▼
                          Final Answer
                                │
                                ▼
                     Generation Evaluation
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
                Faithfulness Relevancy Correctness
```

---

## ⭐ Summary

**K8s Concept Copilot** is an evaluation-driven RAG system that combines:

**Query Translation + BGE Vector Retrieval + BM25 + Hybrid Retrieval + Reranking + Context Extraction + Groq Generation**

with independent retrieval and generation evaluation.

The current evaluation demonstrates a **strong retrieval layer** with **100% Recall@5 and 0.97 MRR**, while generation achieves **0.91 faithfulness, 0.93 relevancy, and 0.83 correctness** across 20 human-labeled queries.

The main remaining improvement area is **generation correctness and completeness**, rather than basic retrieval quality.
