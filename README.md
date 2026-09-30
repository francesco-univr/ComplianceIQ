# ComplianceIQ

Regulatory assistant for EU cybersecurity and data protection law.

ComplianceIQ reads the full text of the NIS2 Directive and the GDPR and answers compliance questions with the Article and paragraph behind each statement. It targets the questions that otherwise send a consultant through two long regulation PDFs by hand.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What it does

You ask a compliance question in plain language. The answer sticks to the regulatory text, cites each Article with its paragraph numbers and lists the retrieved sources next to the chat.

An example question

> What are the incident notification timelines under NIS2?

and the answer it produces

> Essential or important entities must submit an early warning within **24 hours** and a full incident notification within **72 hours** of becoming aware of a significant incident (**Article 23(1), (4) NIS2**). The mere act of notification shall not subject the notifying entity to increased liability.

---

## Why not a basic RAG

A standard RAG pipeline struggles with regulatory text. Fixed-size chunking cuts Articles in half. Semantic search on its own misses the legal synonyms a user does not type, while a single retriever over both regulations mixes NIS2 and GDPR passages.

### Article-aware chunking

Each chunk maps to one Article and carries its regulation, chapter, Article number and title. The chunker splits long Articles at paragraph boundaries, so no chunk starts mid-sentence. Before chunking, 30 regex patterns repair the words that the EUR-Lex PDF extraction breaks apart, such as "Ar ticle" or "cyber- security".

### Hybrid retrieval

An ensemble retriever combines MMR semantic search with BM25 keyword search, weighted 0.6 and 0.4. A cross-encoder (ms-marco-MiniLM-L-6-v2) then reranks the results. At indexing time every chunk whose title matches a legal term receives extra keywords, so an Article about fines also answers a query about penalties, sanctions or enforcement.

### Companion Articles

Some Articles only make sense next to another one. When retrieval returns NIS2 Article 35 or 36, the system adds Article 34 with the general conditions for administrative fines. The same map pairs NIS2 Article 20 on governance with Article 21 on risk management measures. It also links GDPR Articles 38 and 39 to Article 37 on the designation of the DPO.

### Regulation filter

A filter in the sidebar limits retrieval to NIS2 or to GDPR through the chunk metadata, so passages from the two texts do not mix.

### One chunk per Article

The results keep one chunk per Article. An answer draws on several provisions instead of three pieces of the same one.

---

## Regulations loaded

| Regulation | Full name | Pages | Chunks |
|---|---|---|---|
| NIS2 | Directive (EU) 2022/2555 | 73 | ~400 |
| GDPR | Regulation (EU) 2016/679 | 88 | ~460 |

The roadmap adds the EU AI Act and the ISO 27001 Annex A controls.

---

## Architecture

```
PDF documents (NIS2, GDPR)
        │
        ▼
  OCR Cleanup (30 regex patterns)
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
             Companion Articles
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
| Vector store | ChromaDB |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Keyword search | BM25 (rank_bm25) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Groq API, Llama 3.3 70B Versatile |
| Frontend | Gradio Blocks with a custom dark theme |
| Language | Python 3.10+ |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/francesco-univr/ComplianceIQ.git
cd ComplianceIQ
pip install -r requirements.txt
```

### 2. Download the regulations

Download the official PDFs from EUR-Lex and save them in `./docs/` as `nis2.pdf` and `gdpr.pdf`.

- [NIS2, Directive (EU) 2022/2555](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32022L2555)
- [GDPR, Regulation (EU) 2016/679](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679)

```bash
mkdir docs
# save the two PDFs here as nis2.pdf and gdpr.pdf
```

### 3. Configure the API key

Copy `.env.example` to `.env` and put your Groq key in it.

```
GROQ_API_KEY=your_groq_api_key_here
```

You can create a key at [console.groq.com](https://console.groq.com/keys).

### 4. Run

```bash
python app.py
```

The first run builds the vector store in about two minutes. Later runs load it from disk in a few seconds. Then open `http://127.0.0.1:7860` in your browser.

---

## Features

- The answer streams token by token while the source panel fills in.
- Every answer ends with a confidence level, HIGH, PARTIAL or LOW, that says how much of the question the retrieved text covers.
- Follow-up questions carry the last three question and answer pairs as context.
- The regulation filter limits retrieval to NIS2, to GDPR or to both.
- Each source shows regulation, Article, chapter and page, with the text excerpt one click away.
- The model suggests three follow-up questions after each answer.
- You can export the conversation as a Markdown file.
- The vector store rebuilds itself when its chunk count drifts or its files are corrupt.
- The app retries rate-limited requests with exponential backoff.

---

## Roadmap

- [ ] EU AI Act (Regulation (EU) 2024/1689)
- [ ] ISO 27001 Annex A controls
- [ ] DORA (Digital Operational Resilience Act)
- [ ] Multi-language support (IT, DE, FR)
- [ ] Deployment on Hugging Face Spaces
- [ ] A comparison mode for questions such as "How do NIS2 and GDPR differ on incident notification?"

---

## Disclaimer

ComplianceIQ is an informational tool and does not constitute legal advice. Consult qualified legal counsel for compliance decisions.

---

## License

MIT

---

## Author

**Francesco Simbola**
- MSc Artificial Intelligence, University of Verona
- Master in Cybersecurity, University of Pisa
- [LinkedIn](https://www.linkedin.com/in/francescosimbola/)
- [GitHub](https://github.com/francesco-univr)
