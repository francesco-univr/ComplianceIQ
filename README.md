# 🛡️ ComplianceIQ

AI-powered regulatory advisor for EU cybersecurity and data protection law.

ComplianceIQ ingests full-text EU regulations and answers compliance questions with exact Article and paragraph citations. Built for GRC professionals, security consultants, and compliance teams who spend hours searching through regulation PDFs.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## What it does

Ask a compliance question in natural language. Get a structured answer grounded in the regulatory text, with specific Article references, paragraph numbers, and source traceability.

**Example:**

> **Q:** What are the incident notification timelines under NIS2?
>
> **A:** Essential or important entities must submit an early warning within **24 hours** and a full incident notification within **72 hours** of becoming aware of a significant incident (**Article 23(1), (4) NIS2**). The mere act of notification shall not subject the notifying entity to increased liability.

---

## Why not a basic RAG

Standard RAG pipelines fail on regulatory text. PDF chunking breaks Articles mid-sentence. Semantic search alone misses legal synonyms. Generic retrievers mix up NIS2 and GDPR contexts.

ComplianceIQ solves each of these problems:

**Article-aware chunking** — Every chunk maps to a single Article with structured metadata (regulation, chapter, article number, title). Long Articles are split at paragraph boundaries, never mid-sentence. OCR artifacts from EUR-Lex PDFs are cleaned with 25+ regex patterns.

**Hybrid retrieval** — BM25 keyword search combined with MMR semantic search via an ensemble retriever. Results are reranked with a cross-encoder (ms-marco-MiniLM-L-6-v2). Legal synonym enrichment automatically maps "fines" → "penalties, sanctions, enforcement".

**Regulatory sibling expansion** — Query Article 34 NIS2 on penalties and the system auto-injects Articles 35–36 because they are legally interdependent. Configurable companion map for related provisions.

**Scoped search** — Mention "NIS2" or "GDPR" in your question and retrieval filters to that regulation only. Metadata-based filtering eliminates cross-regulation contamination.

**Deduplication** — Results are deduplicated by Article number so you get diverse coverage, not three chunks from the same provision.

---

## Regulations loaded

| Regulation | Full name | Pages | Chunks |
|---|---|---|---|
| **NIS2** | Directive (EU) 2022/2555 | 73 | ~400 |
| **GDPR** | Regulation (EU) 2016/679 | 88 | ~460 |

Adding EU AI Act and ISO 27001 Annex A controls is on the roadmap.

---

## Architecture

```
PDF documents (NIS2, GDPR)
        │
        ▼
  OCR Cleanup (25+ regex patterns)
        │
        ▼
  Article-Aware Chunking
  (structured metadata per chunk)
        │
        ▼
  HuggingFace Embeddings ──► ChromaDB Vector Store
  (all-MiniLM-L6-v2)              │
                                   │
  User Query ──────────────────────┤
        │                          │
        ▼                          ▼
  BM25 Retriever          MMR Semantic Retriever
        │                          │
        └──────── Ensemble ────────┘
                     │
                     ▼
            Cross-Encoder Reranker
                     │
                     ▼
            Article Deduplication
                     │
                     ▼
          Sibling Article Expansion
                     │
                     ▼
              Groq LLM (Llama 3.3 70B)
                     │
                     ▼
          Structured Answer + Citations
```

---

## Stack

| Component | Technology |
|---|---|
| Orchestration | LangChain |
| Vector Store | ChromaDB |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Keyword Search | BM25 (rank_bm25) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Groq API — Llama 3.3 70B Versatile |
| Frontend | Gradio Blocks (custom dark theme) |
| Language | Python 3.10+ |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/francesco-univr/ComplianceIQ.git
cd ComplianceIQ
pip install -r requirements.txt
```

### 2. Download regulations

Download the official PDFs from EUR-Lex and place them in `./docs/`:

- **NIS2**: [Directive (EU) 2022/2555](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32022L2555)
- **GDPR**: [Regulation (EU) 2016/679](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679)

```bash
mkdir docs
# Save the PDFs as docs/nis2.pdf and docs/gdpr.pdf
```

### 3. Configure API key

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com/keys).

### 4. Run

```bash
python app.py
```

First run builds the vector store (~2 minutes). Subsequent runs load from disk in seconds.

Open `http://127.0.0.1:7860` in your browser.

---

## Features

- **Streaming responses** — Token-by-token output with real-time source panel updates
- **Conversation memory** — Follow-up questions use previous Q&A context
- **Regulation filter** — Scope retrieval to NIS2 only, GDPR only, or both
- **Source traceability** — Every answer shows regulation, Article number, chapter, and page with expandable text excerpts
- **Follow-up suggestions** — AI-generated follow-up questions after each answer
- **Export** — Download conversation history as Markdown
- **Auto-rebuild** — Vector store auto-heals if chunk count drifts or corruption is detected
- **Rate limit resilience** — Exponential backoff retry on API throttling

---

## Roadmap

- [ ] EU AI Act (Regulation (EU) 2024/1689)
- [ ] ISO 27001 Annex A controls
- [ ] DORA (Digital Operational Resilience Act)
- [ ] Multi-language support (IT, DE, FR)
- [ ] Deployment on Hugging Face Spaces
- [ ] Comparison mode: "How do NIS2 and GDPR differ on incident notification?"

---

## Disclaimer

ComplianceIQ is an informational tool. It does not constitute legal advice. Always consult qualified legal counsel for compliance decisions.

---

## License

MIT

---

## Author

**Francesco Simbola**
- MSc Artificial Intelligence — University of Verona
- Master in Cybersecurity — University of Pisa
- [LinkedIn](https://www.linkedin.com/in/francescosimbola/)
- [GitHub](https://github.com/francesco-univr)
