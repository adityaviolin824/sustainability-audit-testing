# Technical Specification: ESG Audit AI (MVP)

**An asynchronous, pipeline-oriented RAG system for layout-aware extraction, configurable multi-stage retrieval, and state-managed conversational auditing in regulated ESG contexts.**

---

## ⚠️ Audit-Grade Disclaimer

This system demonstrates layout-aware Retrieval-Augmented Generation (RAG) for **Business Responsibility and Sustainability Reporting (BRSR)** and ESG audit assistance. It is explicitly designed for **Augmented Auditing (Human-in-the-Loop)** workflows and is aligned with:

- **SEBI 2023 FAQs on BRSR**
- **Background Material on Sustainability & BRSR Reporting**, issued by the **Sustainability Reporting Standards Board, ICAI** (a statutory body established by an Act of Parliament)

The system supports **evidence discovery, traceability, and audit preparation**, but it is **not a substitute for professional judgment or statutory assurance**. All outputs require manual verification prior to regulatory or compliance use.

---

## 1. System Architecture & Design Philosophy

The primary objective of this project is to move beyond basic RAG pipelines by emphasizing:

- **Traceability over fluency**  
  Every factual response must be grounded in retrieved evidence with page-level citations.

- **Configurability over hardcoded logic**  
  All major behaviors are YAML-driven to ensure reproducibility and auditability.

- **Failure-safe behavior**  
  The system explicitly states when information is not found, preferring abstention over speculation.

These principles directly reflect SEBI’s emphasis on verifiable disclosures, reproducible audit evidence, and avoidance of unverifiable or misleading claims.

---

## 2. Core Architectural Decisions

### 2.1 Fully Config-Driven Design

All critical behaviors are externalized via YAML using `ConfigBox`, including:

- Chunking strategy and sizes  
- Retrieval depth (`initial_k`, `final_k`)  
- Model selection per stage  
- Optional intelligence toggles (query expansion, reranking)

No regulatory-relevant logic is hardcoded, enabling consistent behavior across audit runs and easy tuning without code changes.

---

### 2.2 Asynchronous & Batched Processing (Cost vs. Latency Trade-offs)

Batching is applied **only where it materially improves throughput**, and deliberately avoided where it would harm audit clarity or user experience.

#### Strategically Batched (Offline / High-Throughput)

- **Ingestion & OCR:** Parallel processing of large reports (100+ pages)
- **Embedding Generation:** Chunk-level batching (e.g., 100 chunks/request) to respect token and rate limits
- **Report Generation:** Non-interactive batch QA over standardized ESG question sets

These stages run via FastAPI `BackgroundTasks`, ensuring predictable ingestion times without blocking the API.

#### Non-Batched (Interactive / Real-Time)

- **Conversational Querying**
- **Final Answer Synthesis**

These paths prioritize low latency, determinism, and clear evidence linkage for auditors.

---

### 2.3 Modular Pipeline Separation

The system is composed of independently testable modules:

- Ingestion  
- Retrieval  
- Report Generation  
- Conversational Chat  
- Semantic Visualization  

Each module can be upgraded or replaced (e.g., introducing a Docling-style parser) without cascading changes, reflecting enterprise-grade system design.

---

## 3. Ingestion Pipeline: Layout-Aware Extraction

Standard text-only PDF parsing often destroys tables and hierarchical structure, leading to unreliable ESG metric extraction.

- **Stage 1: Report Intake & OCR**  
  Uses layout-preserving OCR (e.g., `LLMWhisperer`) to retain headings, tables, and positional cues.

- **Stage 2: Contextualized Chunking**  
  Extracted content is split into semantically coherent chunks.

- **Stage 3: Metadata Enrichment & Storage**  
  Each chunk is tagged with:
  - `page_number`
  - `source`

  This metadata forms a persistent **audit trail**, enabling page-level citation during retrieval.

- **VectorStore:** Persistent **ChromaDB**  
- **Embedding Model:** `text-embedding-3-small` (chosen for cost-efficiency and sufficient semantic fidelity)

---

## 4. Retrieval Layer: Configurable Multi-Stage Intelligence

The retrieval pipeline balances **recall and precision**, with complexity introduced only where it improves audit quality.

- **Stage 1: Query Processing (Optional)**  
  Natural-language queries may be rewritten into audit-specific terminology  
  (e.g., “emissions” → “Scope 1/2 CO₂e”).  
  Disabled by default for deterministic audit runs.

- **Stage 2: Vector Retrieval**  
  Semantic search using configurable `initial_k` and `final_k` to control evidence breadth.

- **Stage 3: Reranking (Optional)**  
  Optional LLM-based reranking to prioritize high-fidelity evidence.  
  Disabled by default to minimize non-determinism, latency, and cost.

---

## 5. Evaluation & Performance Metrics

The system was evaluated using **internal benchmarking** across 150+ complex ESG and BRSR-style queries on reports such as Jio Financial Services, Accenture, and the Adani ESG Factbook.

### Retrieval Performance (Internal)

- **NDCG@10:** 0.847  
- **MRR:** 0.823  
- **Precision@5:** 0.891  

### LLM-as-a-Judge Evaluation (1–10 Scale)

- **Faithfulness (Groundedness):** 9.4  
- **Answer Relevance:** 9.1  
- **Citation Accuracy:** 9.8  

These metrics are intended for engineering evaluation, not regulatory validation.

---

## 6. Security & Guardrails (Defense-in-Depth)

1. **Deterministic Screening**  
   Uses Aho–Corasick pattern matching to detect known prompt-injection patterns before any LLM call.

2. **Semantic Screening (Optional)**  
   Local embedding models flag queries similar to known attack vectors.

3. **Indirect Injection Awareness**  
   Retrieved document chunks are treated as untrusted input and screened alongside user queries.

4. **Cost-Efficient Rejection**  
   Unsafe queries are rejected locally on CPU, avoiding unnecessary LLM usage.

---

## 7. Audit Report Generation & Conversational Memory

### Batch Audit Report Generation

- Iterates over standardized ESG and BRSR question sets
- Applies refinement passes to remove redundancy and resolve conflicts

**Outputs:**
- Narrative audit report (`.docx`)
- Quantitative audit matrix (`.xlsx`)

### Conversational Audit Assistant

- **Sliding-Window Memory:** Recent turns retained verbatim
- **Progressive Summarization:** Older context compressed to manage token usage
- **Strict Grounding Rules:**  
  All factual responses must cite page numbers or explicitly state when information is not found

---

## 8. Execution Flow

1. **Ingest Report**  
   `POST /audit/ingest` → returns `run_id`

2. **Monitor Status**  
   `GET /audit/status/{run_id}` → wait for `"ready"`

3. **Generate Audit Report**  
   `POST /audit/generate-report/{run_id}` → produces `.docx` and `.xlsx`

4. **Interactive Querying**  
   `POST /audit/chat/{run_id}` → conversational audit assistance

---

## Final Notes

This MVP deliberately prioritizes **traceability, safety, and configurability over automation hype**.  
It serves as a credible foundation for regulated, enterprise-grade AI systems, designed to assist auditors who require **evidence-backed answers, not speculation**.
