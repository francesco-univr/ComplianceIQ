# ComplianceIQ answers compliance questions on the NIS2 Directive 2022/2555 and the GDPR 2016/679
# it retrieves the relevant Articles from the official PDFs and asks an LLM to answer with citations

from __future__ import annotations

import os
import re
import time
import logging
import shutil
import tempfile
from typing import Generator, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
import gradio as gr

# paths, models and retrieval settings
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

DOCS_DIR:   str = "./docs"
CHROMA_DIR: str = "./chroma_db"

MODEL_NAME:    str   = "llama-3.3-70b-versatile"
TEMPERATURE:   float = 0.1
EMBED_MODEL:   str   = "all-MiniLM-L6-v2"
LLM_TIMEOUT:   float = 45.0
LLM_MAX_RETRIES: int = 3

CHUNK_SIZE_MAX: int = 2200
CHUNK_SIZE_MIN: int = 80
RETRIEVER_K:    int = 6
MMR_FETCH_K:    int = 30
ENSEMBLE_WEIGHTS: List[float] = [0.6, 0.4]
MAX_CHUNKS_PER_ARTICLE: int   = 1
CONV_MEMORY_TURNS: int        = 3   # Q&A pairs to include in context

REG_NIS2: str = "NIS2"
REG_GDPR: str = "GDPR"

APP_VERSION: str = "3.0"

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("complianceiq")

# words that the PDF text extraction splits in two and their correct form
_OCR_FIXES: List[Tuple[str, str]] = [
    (r"\bAr\s+ticle\b",                    "Article"),
    (r"\bCHA\s*PTER\b",                    "CHAPTER"),
    (r"\bsecur\s+ity\b",                   "security"),
    (r"\bimpor\s+tant\b",                  "important"),
    (r"\bessen\s+tial\b",                  "essential"),
    (r"\binf\s+or\s*mation\b",             "information"),
    (r"\bpropor\s+tionate?\b",             "proportionate"),
    (r"\badminist\s+rat\s*iv\s*e?\b",      "administrative"),
    (r"\bef\s+fect\s+ive\b",               "effective"),
    (r"\bor\s+ganisati(?:on|onal)\b",      "organisational"),
    (r"\bin\s+fr\s+inge?\s*ment\b",        "infringement"),
    (r"\bcybersecur\s+ity\b",              "cybersecurity"),
    (r"\bmanage\s+ment\b",                 "management"),
    (r"\bsignif\s+icant\b",               "significant"),
    (r"\bnotif\s+ication\b",              "notification"),
    (r"\bau\s+thor\s+ity\b",              "authority"),
    (r"\bau\s+thor\s+ities\b",            "authorities"),
    (r"\bcom\s+pet\s+ent\b",              "competent"),
    (r"\binter\s+nal\b",                  "internal"),
    (r"\bpro\s+vider\b",                  "provider"),
    (r"\brepor\s+ting\b",                 "reporting"),
    (r"\bsup\s+er\s+vis\s*or\s*y\b",     "supervisory"),
    (r"\bre\s+quire\s+ment\b",            "requirement"),
    (r"\bim\s+plement\s+ation\b",        "implementation"),
    (r"\bev\s+alu\s+ation\b",            "evaluation"),
    (r"\bpro\s+port\s+ionate\b",         "proportionate"),
    (r"\bMem\s+ber\s+Stat\s*es?\b",      "Member States"),
    (r"\bdir\s+ect\s+ive\b",             "Directive"),
    (r"\breg\s+ul\s+ation\b",            "Regulation"),
    # join words broken by a hyphen at a line end such as cyber- security
    (r"(\w)-\s+(\w)",                     r"\1\2"),
]

_OCR_COMPILED = [(re.compile(p, re.IGNORECASE), r) for p, r in _OCR_FIXES]


def clean_ocr(text: str) -> str:
    # apply every OCR fix to the text of a PDF page
    for pattern, replacement in _OCR_COMPILED:
        text = pattern.sub(replacement, text)
    return text


# patterns for Articles, chapters, paragraphs and the tags the LLM adds to its answer
ARTICLE_RE = re.compile(
    r"Article\s+(\d+)\s*\n\s*(.+?)\s*\n",
    re.MULTILINE,
)
CHAPTER_RE = re.compile(
    r"CHAPTER\s+([IVXLC]+)\s*\n\s*(.+?)\s*\n",
    re.MULTILINE,
)
PARA_SPLIT_RE  = re.compile(r"\n(?=\d+\.\s)")
NIS2_QUERY_RE  = re.compile(r"\bnis\s*2\b", re.IGNORECASE)
GDPR_QUERY_RE  = re.compile(r"\bgdpr\b",    re.IGNORECASE)
SUGGEST_RE     = re.compile(r"<!--\s*SUGGEST:\s*(.*?)\s*-->", re.DOTALL)
CONFIDENCE_RE  = re.compile(r"\*\*Confidence:\s*(HIGH|PARTIAL|LOW)\*\*", re.IGNORECASE)

# extra keywords for a chunk whose title matches a legal term
_LEGAL_SYNONYM_PATTERNS: List[Tuple[str, str]] = [
    (r"fines?",                      "penalties sanctions enforcement"),
    (r"penalties|sanctions",         "fines administrative enforcement"),
    (r"erasure|deletion",            "right to be forgotten remove data"),
    (r"notification|reporting",      "incident report obligations"),
    (r"infringement|violation",      "non-compliance breach penalty"),
    (r"supervision|enforcement",     "penalties oversight compliance"),
    (r"risk.?management",            "security measures requirements cybersecurity"),
    (r"data.?protection.?officer",   "DPO privacy officer"),
    (r"processor|controller",        "data controller data processor GDPR"),
    (r"breach",                      "incident violation security"),
    (r"designation|appointment",     "obligation mandatory required"),
]


def _normalize_ocr(text: str) -> str:
    # remove the stray spaces OCR leaves inside words so that f ines becomes fines
    return re.sub(r"(?<=[A-Za-z]) (?=[A-Za-z])", "", text)


def _derive_keywords(article_title: str, chapter_title: str) -> str:
    # synonyms that help BM25 and the embeddings find the Article
    normalised = _normalize_ocr(article_title + " " + chapter_title).lower()
    seen: set = set()
    extras: List[str] = []
    for pattern, synonyms in _LEGAL_SYNONYM_PATTERNS:
        if re.search(pattern, normalised, re.IGNORECASE):
            for kw in synonyms.split():
                if kw not in seen:
                    seen.add(kw)
                    extras.append(kw)
    return " ".join(extras)


# PDF loading
def load_documents(docs_dir: str = DOCS_DIR) -> Dict[str, List[Document]]:
    # load every PDF in the folder and clean each page, keyed by file name
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"Docs directory {docs_dir} not found")
    result: Dict[str, List[Document]] = {}
    for filename in sorted(os.listdir(docs_dir)):
        if not filename.lower().endswith(".pdf"):
            continue
        path = os.path.join(docs_dir, filename)
        pages = PyPDFLoader(path).load()
        for page in pages:
            page.page_content = clean_ocr(page.page_content)
            page.metadata["source"] = filename
        result[filename] = pages
        logger.info("Loaded %s with %d pages", filename, len(pages))
    if not result:
        raise RuntimeError(f"No PDFs found in {docs_dir}")
    return result


def _regulation_for(filename: str) -> str:
    lower = filename.lower()
    if "nis2" in lower or "nis_2" in lower:
        return REG_NIS2
    if "gdpr" in lower:
        return REG_GDPR
    return filename.removesuffix(".pdf").upper()



# chunking that follows the Article structure of each regulation
def _page_at(pos: int, page_index: List[Tuple[int, int]]) -> int:
    page_num = page_index[0][1]
    for offset, num in page_index:
        if offset <= pos:
            page_num = num
        else:
            break
    return page_num


def _chapter_at(pos: int, chapters: List[Tuple[int, str, str]]) -> Tuple[str, str]:
    result = ("I", "General Provisions")
    for ch_pos, ch_num, ch_title in chapters:
        if ch_pos <= pos:
            result = (ch_num, ch_title)
        else:
            break
    return result


def _split_long_article(body: str, max_size: int = CHUNK_SIZE_MAX) -> List[str]:
    if len(body) <= max_size:
        return [body]
    raw_parts = PARA_SPLIT_RE.split(body)
    chunks: List[str] = []
    current = ""
    for part in raw_parts:
        if not part.strip():
            continue
        candidate = (current + "\n" + part).strip() if current else part.strip()
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            while len(part) > max_size:
                chunks.append(part[:max_size])
                part = part[max_size:]
            current = part.strip()
    if current:
        chunks.append(current)
    return chunks if chunks else [body]


def chunk_documents(docs_by_file: Dict[str, List[Document]]) -> List[Document]:
    # one Document per Article or paragraph group with a header naming regulation, chapter, Article, title and synonym keywords
    all_chunks: List[Document] = []
    for filename, pages in docs_by_file.items():
        regulation = _regulation_for(filename)
        logger.info("Chunking %s (%s)", filename, regulation)
        page_index: List[Tuple[int, int]] = []
        combined = ""
        for page in pages:
            page_index.append((len(combined), page.metadata["page"]))
            combined += page.page_content + "\n"

        chapters: List[Tuple[int, str, str]] = [
            (m.start(), m.group(1), m.group(2).strip())
            for m in CHAPTER_RE.finditer(combined)
        ]
        article_matches = list(ARTICLE_RE.finditer(combined))
        logger.info("  %d articles and %d chapters found", len(article_matches), len(chapters))

        for idx, match in enumerate(article_matches):
            art_num   = int(match.group(1))
            art_title = match.group(2).strip()
            body_start = match.end()
            body_end   = (
                article_matches[idx + 1].start()
                if idx + 1 < len(article_matches) else len(combined)
            )
            body = combined[body_start:body_end].strip()
            if len(body) < CHUNK_SIZE_MIN:
                continue

            page_num = _page_at(match.start(), page_index)
            ch_num, ch_title = _chapter_at(match.start(), chapters)
            ch_title_clean   = re.sub(r"\s{2,}", " ", ch_title).strip()
            kw_supplement    = _derive_keywords(art_title, ch_title_clean)
            parts            = _split_long_article(body)

            for part_idx, part_text in enumerate(parts):
                header  = (
                    f"[REGULATION: {regulation} | CHAPTER: {ch_num} | "
                    f"ARTICLE: {art_num} | TITLE: {art_title}]"
                )
                if kw_supplement:
                    header += f"\n[KEYWORDS: {kw_supplement}]"
                content = f"{header}\n\n{part_text}"
                all_chunks.append(Document(
                    page_content=content,
                    metadata={
                        "source":        filename,
                        "regulation":    regulation,
                        "article_num":   art_num,
                        "article_title": art_title,
                        "chapter_num":   ch_num,
                        "chapter_title": ch_title_clean,
                        "page":          page_num,
                        "chunk_index":   part_idx,
                        "total_chunks":  len(parts),
                    },
                ))
        logger.info("  %d total chunks after %s", len(all_chunks), filename)
    logger.info("%d chunks in total", len(all_chunks))
    return all_chunks


# Chroma vector store
def build_vectorstore(
    chunks: List[Document],
    chroma_dir: str  = CHROMA_DIR,
    rebuild: bool    = False,
) -> Chroma:
    # load the Chroma store from disk or build it, rebuilding it when the files are corrupt
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    if rebuild and os.path.exists(chroma_dir):
        logger.info("Removing existing vector store at %s", chroma_dir)
        for attempt in range(5):
            try:
                shutil.rmtree(chroma_dir)
                break
            except PermissionError:
                import time
                time.sleep(1)
        else:
            shutil.rmtree(chroma_dir)  # final attempt
    if os.path.exists(chroma_dir):
        try:
            vs = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
            count = vs._collection.count()
            logger.info("Vector store loaded with %d chunks", count)
            return vs
        except Exception as exc:
            logger.warning("Corrupt store (%s), rebuilding", exc)
            shutil.rmtree(chroma_dir)
    logger.info("Building vector store from %d chunks", len(chunks))
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
    )
    logger.info("Vector store built with %d chunks indexed", vs._collection.count())
    return vs


# cross-encoder reranker, skipped when sentence-transformers is missing
_reranker = None
RERANKER_AVAILABLE = False

def _init_reranker() -> None:
    global _reranker, RERANKER_AVAILABLE
    try:
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder reranker")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        RERANKER_AVAILABLE = True
        logger.info("Cross-encoder reranker ready")
    except Exception as exc:
        logger.warning("Reranker unavailable (%s), using ensemble ranking only", exc)


def rerank(query: str, docs: List[Document]) -> List[Document]:
    # sort the documents by cross-encoder score or keep the ensemble order when the reranker is off
    if not RERANKER_AVAILABLE or not docs:
        return docs
    pairs  = [(query, doc.page_content[:512]) for doc in docs]
    scores = _reranker.predict(pairs)
    return [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]



# hybrid retrieval with deduplication and companion Articles
_all_chunks:  List[Document]    = []
_vectorstore: Optional[Chroma]  = None

def build_hybrid_retriever(
    vectorstore: Chroma,
    chunks: List[Document],
    regulation_filter: Optional[str] = None,
) -> EnsembleRetriever:
    if regulation_filter:
        filtered = [c for c in chunks if c.metadata.get("regulation") == regulation_filter]
        chroma_ret = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": RETRIEVER_K, "fetch_k": MMR_FETCH_K,
                "filter": {"regulation": regulation_filter},
            },
        )
    else:
        filtered   = chunks
        chroma_ret = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": RETRIEVER_K, "fetch_k": MMR_FETCH_K},
        )
    bm25_ret   = BM25Retriever.from_documents(filtered)
    bm25_ret.k = RETRIEVER_K
    return EnsembleRetriever(
        retrievers=[chroma_ret, bm25_ret],
        weights=ENSEMBLE_WEIGHTS,
    )


def deduplicate_by_article(
    docs: List[Document],
    max_per_article: int = MAX_CHUNKS_PER_ARTICLE,
    total_k: int = RETRIEVER_K,
) -> List[Document]:
    seen: Dict[str, int] = {}
    result: List[Document] = []
    for doc in docs:
        key = f"{doc.metadata.get('regulation')}:{doc.metadata.get('article_num', 0)}"
        seen.setdefault(key, 0)
        if seen[key] < max_per_article:
            result.append(doc)
            seen[key] += 1
        if len(result) >= total_k:
            break
    return result


# Articles that need a companion Article to make sense
_REQUIRED_COMPANIONS: Dict[str, List[str]] = {
    "NIS2:20": ["NIS2:21"],  # governance refers to the risk management measures
    "NIS2:35": ["NIS2:34"],  # refers to the general conditions for fines
    "NIS2:36": ["NIS2:34"],  # penalties rely on the general conditions for fines
    "GDPR:38": ["GDPR:37"],  # the DPO position depends on the DPO designation
    "GDPR:39": ["GDPR:37"],  # the DPO tasks depend on the DPO designation
}


def expand_with_companions(
    docs: List[Document],
    all_chunks: List[Document],
) -> List[Document]:
    present = {f"{d.metadata.get('regulation')}:{d.metadata.get('article_num')}" for d in docs}
    to_add  = [
        companion
        for key in present
        for companion in _REQUIRED_COMPANIONS.get(key, [])
        if companion not in present
    ]
    if not to_add:
        return docs
    companion_map: Dict[str, Document] = {}
    for chunk in all_chunks:
        key = f"{chunk.metadata.get('regulation')}:{chunk.metadata.get('article_num')}"
        if key in to_add:
            existing = companion_map.get(key)
            if existing is None or chunk.metadata.get("chunk_index", 0) < existing.metadata.get("chunk_index", 0):
                companion_map[key] = chunk
    result = list(docs)
    for key in to_add:
        if key in companion_map:
            result.append(companion_map[key])
            logger.debug("Companion %s added", key)
    return result


def retrieve_docs(query: str, regulation_filter: Optional[str] = None) -> List[Document]:
    # ensemble retrieval then rerank then one chunk per Article then companion Articles
    retriever = build_hybrid_retriever(_vectorstore, _all_chunks, regulation_filter)
    raw       = retriever.invoke(query)
    reranked  = rerank(query, raw)
    deduped   = deduplicate_by_article(reranked)
    expanded  = expand_with_companions(deduped, _all_chunks)
    logger.info("Retrieved %d raw, %d reranked, %d after dedup, %d after expansion",
                len(raw), len(reranked), len(deduped), len(expanded))
    return expanded



# LLM and prompts

SYSTEM_PROMPT = """You are a senior EU regulatory compliance advisor with deep expertise \
in cybersecurity law, data protection and EU legislative frameworks.

Answer only from the regulatory context you receive and keep each answer precise and practical.

Follow these rules.
1. CITATIONS. Bold every Article reference such as **Article 23(4) NIS2** and include paragraph numbers.
2. STRUCTURE. Use ## headers for multi-aspect answers. Use bullet points for requirement lists. \
Use > blockquotes for direct quotes from the regulatory text.
3. CONFIDENCE. End every answer with one of these lines.
   **Confidence: HIGH** when the context answers the question in full.
   **Confidence: PARTIAL** when the context answers only part of the question. State what is missing.
   **Confidence: LOW** when the context is not enough. Name the Articles likely to contain the answer.
4. Do not invent or extrapolate provisions that the context does not contain.
5. FOLLOW-UPS. After the confidence line add exactly this HTML comment with 3 short \
follow-up questions separated by " | ".
<!-- SUGGEST: Follow-up question 1? | Follow-up question 2? | Follow-up question 3? -->"""

USER_TEMPLATE = """REGULATORY CONTEXT
{context}
{conv_context}
---
COMPLIANCE QUESTION
{question}

Give a precise and well structured answer that cites the specific Articles and paragraph numbers."""

_llm: Optional[ChatGroq] = None


def build_llm() -> ChatGroq:
    return ChatGroq(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        request_timeout=LLM_TIMEOUT,
        max_retries=LLM_MAX_RETRIES,
    )


def _build_context(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        meta  = doc.metadata
        label = (
            f"[{meta.get('regulation','?')} Article {meta.get('article_num','?')} "
            f"· {meta.get('article_title','?')}, p.{meta.get('page','?') + 1 if isinstance(meta.get('page'), int) else '?'}]"
        )
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _build_conv_context(chat_history: List[Dict]) -> str:
    if not chat_history:
        return ""
    pairs: List[str] = []
    msgs = [m for m in chat_history if m.get("role") in ("user", "assistant")]
    i = 0
    while i < len(msgs) - 1 and len(pairs) < CONV_MEMORY_TURNS:
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            q = msgs[i]["content"][:300]
            a = msgs[i + 1]["content"][:500]
            pairs.append(f"Q: {q}\nA: {a}")
            i += 2
        else:
            i += 1
    if not pairs:
        return ""
    return "\n\nPREVIOUS CONVERSATION, FOR CONTEXT ONLY\n" + "\n\n".join(pairs)


def _parse_llm_response(text: str) -> Tuple[str, List[str], str]:
    # pull out the suggested follow-up questions
    suggest_match = SUGGEST_RE.search(text)
    suggestions: List[str] = []
    if suggest_match:
        raw = suggest_match.group(1)
        suggestions = [s.strip() for s in raw.split("|") if s.strip()][:3]
        text = text[:suggest_match.start()].rstrip()

    # read the confidence level, PARTIAL when the answer has none
    conf_match = CONFIDENCE_RE.search(text)
    confidence = conf_match.group(1).upper() if conf_match else "PARTIAL"

    return text.strip(), suggestions, confidence

# streaming pipeline that yields the chat state after each step
_response_times: List[float] = []


def stream_pipeline(
    question: str,
    chat_history: List[Dict],
    regulation_filter: Optional[str],
) -> Generator[Tuple, None, None]:

    # show the user message at once
    history = list(chat_history or [])
    history.append({"role": "user", "content": question})
    yield history, _EMPTY_SOURCES, _EMPTY_SUGGESTIONS, _stats_html()

    # retrieve the Articles and show them as sources
    try:
        docs = retrieve_docs(question, regulation_filter)
    except Exception as exc:
        history.append({"role": "assistant", "content": f"Retrieval failed ({exc})"})
        yield history, _EMPTY_SOURCES, _EMPTY_SUGGESTIONS, _stats_html()
        return

    sources_html = _build_sources_html(docs)
    history.append({"role": "assistant", "content": ""})   # placeholder
    yield history, sources_html, _EMPTY_SUGGESTIONS, _stats_html()

    # stream the LLM answer token by token
    context     = _build_context(docs)
    conv_ctx    = _build_conv_context(history[:-2])  # exclude current exchange
    user_prompt = USER_TEMPLATE.format(
        context=context,
        conv_context=conv_ctx,
        question=question,
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    accumulated = ""
    t0 = time.perf_counter()
    last_exc: Optional[Exception] = None

    for attempt in range(LLM_MAX_RETRIES):
        try:
            accumulated = ""
            for chunk in _llm.stream(messages):
                accumulated += chunk.content
                history[-1] = {"role": "assistant", "content": accumulated}
                yield list(history), sources_html, _EMPTY_SUGGESTIONS, _stats_html()
            break   # success
        except Exception as exc:
            last_exc = exc
            err_str  = str(exc).lower()
            if "429" in err_str or "rate limit" in err_str:
                wait = 2 ** attempt
                logger.warning("Rate limited, retry %d/%d after %ds", attempt + 1, LLM_MAX_RETRIES, wait)
                time.sleep(wait)
            else:
                accumulated = f"**Request failed** ({exc})"
                history[-1] = {"role": "assistant", "content": accumulated}
                yield list(history), sources_html, _EMPTY_SUGGESTIONS, _stats_html()
                return
    else:
        accumulated = f"**Rate limit reached** after {LLM_MAX_RETRIES} retries. Wait 30 seconds and try again."
        history[-1] = {"role": "assistant", "content": accumulated}
        yield list(history), sources_html, _EMPTY_SUGGESTIONS, _stats_html()
        return

    elapsed = time.perf_counter() - t0
    _response_times.append(elapsed)
    if len(_response_times) > 20:
        _response_times.pop(0)

    # strip the tags from the answer and show the follow-up suggestions
    clean_answer, suggestions, confidence = _parse_llm_response(accumulated)
    history[-1] = {"role": "assistant", "content": clean_answer}
    suggestions_html = _build_suggestions_html(suggestions)
    yield list(history), sources_html, suggestions_html, _stats_html()

# conversation export as Markdown
def export_conversation(chat_history: List[Dict]) -> str:
    # write the conversation to a temporary Markdown file and return its path
    if not chat_history:
        return ""
    lines = ["# ComplianceIQ conversation export\n"]
    for msg in chat_history:
        if msg["role"] == "user":
            lines.append(f"## Question\n{msg['content']}\n")
        else:
            lines.append(f"## Answer\n{msg['content']}\n\n---\n")
    content = "\n".join(lines)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", prefix="complianceiq_",
        delete=False, encoding="utf-8",
    )
    tmp.write(content)
    tmp.close()
    return tmp.name


# HTML helpers for the interface
_EMPTY_SOURCES      = "<p class='empty-state'>Retrieved sources will appear here.</p>"
_EMPTY_SUGGESTIONS  = ""
_WELCOME_MSG = (
    "Welcome to **ComplianceIQ**. Ask any compliance question about "
    "**NIS2** or **GDPR** and the answer will cite the exact Articles and paragraphs."
)


def _build_sources_html(docs: List[Document]) -> str:
    if not docs:
        return _EMPTY_SOURCES
    parts: List[str] = []
    for doc in docs:
        meta      = doc.metadata
        reg       = meta.get("regulation", "?")
        art_num   = meta.get("article_num", "?")
        art_title = meta.get("article_title", "")
        page      = meta.get("page", "?")
        page_disp = (page + 1) if isinstance(page, int) else page
        ch_num    = meta.get("chapter_num", "")
        is_nis2   = reg == REG_NIS2

        badge_cls = "b-nis2" if is_nis2 else "b-gdpr"
        item_cls  = "src-nis2" if is_nis2 else "src-gdpr"
        ch_tag    = f" · Ch. {ch_num}" if ch_num else ""

        preview = re.sub(r"^\[(?:REGULATION|KEYWORDS):[^\]]+\]\s*\n*", "", doc.page_content)
        preview = preview.strip()[:280]
        if len(preview) == 280:
            preview += "…"

        parts.append(f"""
        <div class="src-item {item_cls}">
            <div class="src-meta">
                <span class="badge {badge_cls}"><span class="bdot"></span>{reg}{ch_tag}</span>
                <span class="src-page">p. {page_disp}</span>
            </div>
            <div class="src-art">Article {art_num} · {art_title}</div>
            <details>
                <summary>Show excerpt</summary>
                <p class="src-preview">{preview}</p>
            </details>
        </div>""")
    return "\n".join(parts)


def _build_suggestions_html(suggestions: List[str]) -> str:
    if not suggestions:
        return _EMPTY_SUGGESTIONS
    # use the native HTMLTextAreaElement setter so that Svelte and Gradio see the change
    js = (
        "(function(v){{"
        "var el=document.querySelector('#q-input textarea');"
        "if(!el)return;"
        "var s=Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,'value').set;"
        "s.call(el,v);"
        "el.dispatchEvent(new Event('input',{{bubbles:true}}));"
        "el.dispatchEvent(new Event('change',{{bubbles:true}}));"
        "el.focus();"
        "}})({val})"
    )
    items = "".join(
        f'<button class="sug-btn" onclick="{js.format(val=repr(s))}">{s}</button>'
        for s in suggestions
    )
    return f'<div class="sug-wrap"><span class="slabel" style="margin-bottom:8px;display:block;">Suggested follow-ups</span>{items}</div>'


def _stats_html() -> str:
    total   = _vectorstore._collection.count() if _vectorstore else 0
    regs    = sorted({c.metadata.get("regulation", "?") for c in _all_chunks})
    avg_t   = f"{sum(_response_times)/len(_response_times):.1f}s" if _response_times else "n/a"
    reg_str = " · ".join(regs) or "none"
    reranker_badge = (
        '<span class="stat-badge reranker-on">reranker on</span>'
        if RERANKER_AVAILABLE
        else '<span class="stat-badge reranker-off">reranker off</span>'
    )
    return (
        f'<div class="stats-inner">'
        f'<span class="stat-item"><span class="stat-k">{total:,}</span><span class="stat-v">chunks</span></span>'
        f'<span class="stat-item"><span class="stat-k">{reg_str}</span><span class="stat-v">loaded</span></span>'
        f'<span class="stat-item"><span class="stat-k">{MODEL_NAME.split("-")[0]}</span><span class="stat-v">model</span></span>'
        f'<span class="stat-item"><span class="stat-k">{avg_t}</span><span class="stat-v">avg latency</span></span>'
        f'{reranker_badge}'
        f'</div>'
    )

# custom dark theme
CUSTOM_CSS = """
/* css variables */
:root {
    --bg:          #0d1117;
    --surface:     #161b22;
    --surface2:    #1c2128;
    --border:      #21262d;
    --border2:     #30363d;
    --text:        #c9d1d9;
    --text-muted:  #8b949e;
    --text-dim:    #484f58;
    --text-bright: #f0f6fc;
    --blue:        #388bfd;
    --blue-dim:    rgba(56,139,253,.12);
    --blue-brd:    rgba(56,139,253,.35);
    --green:       #2ea043;
    --green-dim:   rgba(46,160,67,.12);
    --green-brd:   rgba(46,160,67,.35);
    --purple:      #bc8cff;
    --purple-dim:  rgba(163,113,247,.12);
    --purple-brd:  rgba(163,113,247,.35);
    --gold:        #e3b341;
    --gold-dim:    rgba(210,153,34,.12);
    --gold-brd:    rgba(210,153,34,.35);
    --radius:      10px;
    --radius-sm:   7px;
    --shadow:      0 4px 20px rgba(0,0,0,.4);
}

/* base styles without a global reset because it breaks the Gradio and Svelte internals */

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    color: var(--text) !important;
}
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* app header */
#app-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border-bottom: 1px solid var(--border);
    padding: 28px 36px 22px;
    position: relative;
    overflow: hidden;
}
#app-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(56,139,253,.07) 0%, transparent 70%);
    pointer-events: none;
}
.hdr-row { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
.hdr-left { display: flex; flex-direction: column; gap: 6px; }
.hdr-title {
    font-size: 24px; font-weight: 700; color: var(--text-bright);
    letter-spacing: -.4px; display: flex; align-items: center; gap: 10px;
}
.hdr-sub { color: var(--text-muted); font-size: 13px; line-height: 1.5; }
.badge-row { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; }
.badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 20px; font-size: 11px;
    font-weight: 600; letter-spacing: .3px; cursor: default;
    transition: transform .15s;
}
.badge:hover { transform: translateY(-1px); }
.bdot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; background: currentColor; }
.b-nis2   { background: var(--blue-dim);   border: 1px solid var(--blue-brd);   color: #79c0ff; }
.b-gdpr   { background: var(--green-dim);  border: 1px solid var(--green-brd);  color: #56d364; }
.b-llm    { background: var(--purple-dim); border: 1px solid var(--purple-brd); color: var(--purple); }
.b-ver    { background: var(--gold-dim);   border: 1px solid var(--gold-brd);   color: var(--gold); }

/* stats bar */
#stats-wrap { border-bottom: 1px solid var(--border); padding: 0 36px; }
.stats-inner {
    display: flex; align-items: center; gap: 20px; padding: 8px 0;
    flex-wrap: wrap;
}
.stat-item { display: flex; flex-direction: column; align-items: center; gap: 1px; }
.stat-k { font-size: 12px; font-weight: 600; color: var(--text); }
.stat-v { font-size: 10px; color: var(--text-dim); text-transform: uppercase; letter-spacing: .5px; }
.stat-badge {
    font-size: 10px; font-weight: 600; padding: 2px 8px; border-radius: 20px;
    letter-spacing: .3px;
}
.reranker-on  { background: var(--green-dim); border: 1px solid var(--green-brd); color: #56d364; }
.reranker-off { background: rgba(72,79,88,.2); border: 1px solid var(--border2); color: var(--text-dim); }

/* sidebar */
#sidebar {
    background: var(--surface);
    border-right: 1px solid var(--border);
    padding: 20px 16px !important;
}
.sidebar-section { display: flex; flex-direction: column; gap: 8px; }
.slabel {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; color: var(--text-dim); display: block;
}

/* Regulation filter radio */
#reg-filter .wrap { gap: 6px !important; }
#reg-filter label {
    background: var(--bg) !important; border: 1px solid var(--border2) !important;
    border-radius: var(--radius-sm) !important; padding: 6px 10px !important;
    color: var(--text-muted) !important; font-size: 12px !important;
    font-weight: 500 !important; cursor: pointer !important;
    transition: all .15s !important; display: flex !important; align-items: center !important;
}
#reg-filter label:hover { border-color: var(--blue) !important; color: var(--text) !important; }
#reg-filter label.selected, #reg-filter input:checked + label {
    background: var(--blue-dim) !important; border-color: var(--blue) !important;
    color: #79c0ff !important;
}
#reg-filter input { display: none !important; }

/* History buttons */
.hist-btn button {
    background: transparent !important; border: none !important;
    border-bottom: 1px solid var(--border) !important; border-radius: 0 !important;
    color: var(--text-muted) !important; font-size: 11px !important;
    padding: 7px 2px !important; text-align: left !important;
    cursor: pointer !important; transition: color .15s !important;
    width: 100% !important; white-space: normal !important;
    line-height: 1.4 !important; height: auto !important;
}
.hist-btn button:hover { color: var(--text) !important; }
.hist-btn:last-child button { border-bottom: none !important; }

/* chatbot */
#chatbot-wrap { padding: 8px 0 0; }

#chatbot .wrap { background: transparent !important; border: none !important; }
#chatbot .message-wrap { gap: 16px !important; }

/* User bubble */
#chatbot .user { justify-content: flex-end !important; }
#chatbot .user .message {
    background: linear-gradient(135deg, #1a3a5c, #1f2d45) !important;
    border: 1px solid rgba(56,139,253,.25) !important;
    border-radius: 14px 14px 4px 14px !important;
    color: #c9d1d9 !important;
    font-size: 14px !important; line-height: 1.65 !important;
    max-width: 78% !important; padding: 12px 16px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,.3) !important;
}

/* Assistant bubble */
#chatbot .bot { justify-content: flex-start !important; }
#chatbot .bot .message {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px 14px 14px 14px !important;
    color: var(--text) !important;
    font-size: 14px !important; line-height: 1.75 !important;
    max-width: 88% !important; padding: 14px 18px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,.3) !important;
}
#chatbot .bot .message strong { color: var(--text-bright) !important; }
#chatbot .bot .message h2 {
    color: var(--text-bright) !important; font-size: 15px !important;
    font-weight: 600 !important; margin: 14px 0 6px !important;
    padding-bottom: 4px !important; border-bottom: 1px solid var(--border) !important;
}
#chatbot .bot .message h3 { color: var(--text-bright) !important; font-size: 13px !important; margin: 10px 0 5px !important; }
#chatbot .bot .message blockquote {
    border-left: 3px solid var(--blue) !important;
    padding-left: 12px !important; color: var(--text-muted) !important; margin: 8px 0 !important;
}
#chatbot .bot .message code {
    background: var(--bg) !important; border: 1px solid var(--border2) !important;
    border-radius: 4px !important; padding: 1px 5px !important;
    font-size: 12px !important; color: #ff7b72 !important;
}
#chatbot .bot .message ul, #chatbot .bot .message ol { padding-left: 18px !important; }
#chatbot .bot .message li { margin-bottom: 3px !important; }

/* Welcome state */
#chatbot .pending { display: none !important; }
#chatbot .avatar-container img, #chatbot .avatar { border-radius: 50% !important; }

/* input area */
#input-area {
    padding: 12px 0 16px !important;
    background: var(--bg) !important;
    border-top: 1px solid var(--border) !important;
}
#q-input textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    padding: 11px 14px !important;
    resize: none !important;
    transition: border-color .18s, box-shadow .18s !important;
    pointer-events: auto !important;   /* ensure not blocked */
}
#q-input textarea:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(56,139,253,.1) !important;
    outline: none !important;
}
#q-input textarea::placeholder { color: var(--text-dim) !important; }
/* keep the #q-input label because Gradio 6 puts the textarea inside it */
#q-input label > span:first-child { display: none !important; } /* hide only the text label */
#q-input { pointer-events: auto !important; }

#submit-btn {
    background: linear-gradient(135deg, var(--blue), #1f6feb) !important;
    border: none !important; border-radius: var(--radius-sm) !important;
    color: #fff !important; font-size: 13px !important; font-weight: 600 !important;
    padding: 10px 20px !important; cursor: pointer !important;
    transition: all .18s !important; white-space: nowrap !important;
}
#submit-btn:hover {
    background: linear-gradient(135deg, #58a6ff, var(--blue)) !important;
    box-shadow: 0 3px 12px rgba(56,139,253,.4) !important; transform: translateY(-1px) !important;
}
#submit-btn:disabled { opacity: .5 !important; transform: none !important; }

#clear-btn {
    background: transparent !important; border: 1px solid var(--border2) !important;
    border-radius: var(--radius-sm) !important; color: var(--text-muted) !important;
    font-size: 12px !important; padding: 10px 14px !important;
    cursor: pointer !important; transition: all .18s !important;
}
#clear-btn:hover { border-color: var(--text-muted) !important; color: var(--text) !important; }

#export-btn {
    background: transparent !important; border: 1px solid var(--border2) !important;
    border-radius: var(--radius-sm) !important; color: var(--text-muted) !important;
    font-size: 12px !important; padding: 10px 14px !important;
    cursor: pointer !important; transition: all .18s !important;
}
#export-btn:hover { border-color: var(--gold-brd) !important; color: var(--gold) !important; }

/* Example chips */
.ex-btn button {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: 20px !important; color: var(--text-muted) !important;
    font-size: 12px !important; padding: 5px 12px !important;
    cursor: pointer !important; transition: all .15s !important;
    white-space: nowrap !important; height: auto !important;
}
.ex-btn button:hover { background: var(--surface2) !important; border-color: var(--blue) !important; color: var(--text) !important; }

/* right panel */
#right-panel {
    background: var(--surface) !important;
    border-left: 1px solid var(--border) !important;
    padding: 18px 16px !important;
}
#right-panel .block { border: none !important; background: transparent !important; padding: 0 !important; }

.src-item {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 12px 13px; margin-bottom: 8px;
    transition: border-color .15s;
}
.src-item:hover { border-color: var(--border2); }
.src-item:last-child { margin-bottom: 0; }
.src-nis2 { border-left: 3px solid var(--blue); }
.src-gdpr  { border-left: 3px solid var(--green); }
.src-meta  { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; flex-wrap: wrap; gap: 4px; }
.src-page  { font-size: 11px; color: var(--text-dim); background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 1px 6px; }
.src-art   { font-size: 12px; font-weight: 600; color: #79c0ff; margin-bottom: 2px; }
details summary { cursor: pointer; font-size: 11px; color: var(--text-dim); user-select: none; margin-top: 5px; }
details summary:hover { color: var(--text-muted); }
.src-preview { color: var(--text-dim); font-size: 11px; line-height: 1.5; margin-top: 6px; padding: 8px; background: var(--surface); border-radius: 4px; }

/* Suggestions */
.sug-wrap { display: flex; flex-direction: column; gap: 6px; }
.sug-btn {
    background: var(--bg) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important; color: var(--text-muted) !important;
    font-size: 11px !important; padding: 7px 10px !important;
    text-align: left !important; cursor: pointer !important;
    transition: all .15s !important; width: 100% !important;
    white-space: normal !important; line-height: 1.4 !important;
}
.sug-btn:hover { border-color: var(--blue) !important; color: var(--text) !important; background: var(--surface2) !important; }

/* Empty state */
.empty-state { color: var(--text-dim); font-size: 12px; padding: 6px 0; }

/* footer */
#app-footer {
    background: var(--bg); border-top: 1px solid var(--border);
    padding: 10px 36px; text-align: center;
}
.foot-txt { color: var(--text-dim); font-size: 11px; }

/* scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }

/* responsive */
@media (max-width: 900px) {
    #sidebar, #right-panel { display: none !important; }
    #main-wrap { flex-direction: column; }
}
"""

# example questions shown as chips under the chat
EXAMPLES: List[str] = [
    "NIS2 incident reporting timelines",
    "GDPR right to erasure",
    "Penalties for NIS2 non-compliance",
    "Role of DPO under GDPR",
    "NIS2 risk management measures",
]
EXAMPLE_FULL: List[str] = [
    "What are the incident notification timelines under NIS2?",
    "What is the right to erasure under GDPR?",
    "What penalties can be imposed for NIS2 non-compliance?",
    "What is the role of a Data Protection Officer under GDPR?",
    "What are the cybersecurity risk management measures under NIS2?",
]

# Gradio interface
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ComplianceIQ") as demo:

        # header
        gr.HTML(f"""
        <div id="app-header">
            <div class="hdr-row">
                <div class="hdr-left">
                    <div class="hdr-title">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" style="flex-shrink:0">
                            <path d="M12 2L3 7v5c0 5.25 3.75 10.15 9 11.35C17.25 22.15 21 17.25 21 12V7L12 2z"
                                  fill="#388bfd" fill-opacity=".18" stroke="#388bfd" stroke-width="1.5" stroke-linejoin="round"/>
                            <path d="M9 12l2 2 4-4" stroke="#388bfd" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                        ComplianceIQ
                    </div>
                    <p class="hdr-sub">Regulatory assistant for EU cybersecurity and data protection law</p>
                    <div class="badge-row">
                        <span class="badge b-nis2"><span class="bdot"></span>NIS2 Directive 2022/2555</span>
                        <span class="badge b-gdpr"><span class="bdot"></span>GDPR 2016/679</span>
                        <span class="badge b-llm"><span class="bdot"></span>{MODEL_NAME}</span>
                        <span class="badge b-ver"><span class="bdot"></span>v{APP_VERSION}</span>
                    </div>
                </div>
            </div>
        </div>
        """)

        # stats bar
        stats_output = gr.HTML(value=_stats_html(), elem_id="stats-wrap")

        # sidebar on the left, chat in the middle and sources on the right
        with gr.Row(elem_id="main-wrap"):

            # left sidebar
            with gr.Column(elem_id="sidebar", scale=0, min_width=220):

                with gr.Column(elem_classes=["sidebar-section"]):
                    gr.HTML('<span class="slabel">Regulation filter</span>')
                    reg_filter = gr.Radio(
                        choices=["Both", "NIS2 only", "GDPR only"],
                        value="Both",
                        show_label=False,
                        elem_id="reg-filter",
                    )

                with gr.Column(elem_classes=["sidebar-section"]):
                    gr.HTML('<span class="slabel">Recent queries</span>')
                    hist_btns = [
                        gr.Button("", visible=False, elem_classes=["hist-btn"], size="sm")
                        for _ in range(5)
                    ]

                with gr.Column(elem_classes=["sidebar-section"]):
                    gr.HTML("""
                    <div style="margin-top:auto;">
                        <span class="slabel" style="margin-bottom:6px;">Quick tips</span>
                        <p style="font-size:11px;color:#484f58;line-height:1.6;">
                            Pick <strong style="color:#8b949e;">NIS2 only</strong> or <strong style="color:#8b949e;">GDPR only</strong>
                            in the regulation filter to focus the search.<br><br>
                            Press <strong style="color:#8b949e;">Enter</strong> to submit.
                        </p>
                    </div>
                    """)

            # chat in the middle
            with gr.Column(elem_id="center-area", scale=3):

                with gr.Column(elem_id="chatbot-wrap"):
                    chatbot = gr.Chatbot(
                        value=[{"role": "assistant", "content": _WELCOME_MSG}],
                        render_markdown=True,
                        layout="bubble",
                        height=520,
                        elem_id="chatbot",
                        avatar_images=(None, None),
                        show_label=False,
                        buttons=["copy"],
                    )

                # input area
                with gr.Column(elem_id="input-area"):

                    # example chips
                    with gr.Row():
                        ex_btns = [
                            gr.Button(ex, size="sm", elem_classes=["ex-btn"])
                            for ex in EXAMPLES
                        ]

                    with gr.Row(equal_height=True):
                        q_input = gr.Textbox(
                            placeholder="Ask about NIS2 or GDPR compliance",
                            show_label=False,
                            lines=1,
                            max_lines=4,
                            elem_id="q-input",
                            scale=5,
                        )
                        submit_btn = gr.Button("Send", variant="primary", elem_id="submit-btn", scale=1, min_width=90)
                        clear_btn  = gr.Button("Clear", elem_id="clear-btn", scale=0, min_width=60)
                        export_btn = gr.Button("Export", elem_id="export-btn", scale=0, min_width=80)

                    export_file = gr.File(visible=False, label="Download")

            # right panel with sources and follow-up suggestions
            with gr.Column(elem_id="right-panel", scale=0, min_width=300):

                gr.HTML('<span class="slabel" style="margin-bottom:8px;display:block;">Sources</span>')
                sources_output = gr.HTML(value=_EMPTY_SOURCES)

                suggestions_output = gr.HTML(value=_EMPTY_SUGGESTIONS)

        # footer
        gr.HTML("""
        <div id="app-footer">
            <p class="foot-txt">
                ComplianceIQ &nbsp;·&nbsp; NIS2 Directive (EU) 2022/2555 &amp; GDPR (EU) 2016/679
                &nbsp;·&nbsp; For informational purposes only &nbsp;·&nbsp; Not legal advice
            </p>
        </div>
        """)

        # conversation state
        history_state = gr.State([])

        # event wiring for the streaming answer
        def _reg_to_filter(reg_radio: str) -> Optional[str]:
            if reg_radio == "NIS2 only":
                return REG_NIS2
            if reg_radio == "GDPR only":
                return REG_GDPR
            return None

        def handle_submit(question: str, chat_history: list, reg: str, hist_state: list):
            if not question.strip():
                return
            reg_filter = _reg_to_filter(reg)
            for result in stream_pipeline(question, chat_history, reg_filter):
                ch, src, sug, stats = result
                # refresh the recent queries and their buttons
                seen: set = set()
                new_hist: list = []
                for m in ch:
                    if m.get("role") == "user":
                        c = m.get("content", "")
                        text = c if isinstance(c, str) else (c[0].get("text", "") if c else "")
                        if text and text not in seen:
                            seen.add(text)
                            new_hist.append(text)
                new_hist = new_hist[:5]
                btn_updates = [
                    gr.update(value=new_hist[i], visible=True) if i < len(new_hist)
                    else gr.update(value="", visible=False)
                    for i in range(5)
                ]
                yield ch, src, sug, stats, new_hist, "", *btn_updates

        all_outputs = [chatbot, sources_output, suggestions_output, stats_output,
                       history_state, q_input, *hist_btns]

        submit_btn.click(
            fn=handle_submit,
            inputs=[q_input, chatbot, reg_filter, history_state],
            outputs=all_outputs,
        )
        q_input.submit(
            fn=handle_submit,
            inputs=[q_input, chatbot, reg_filter, history_state],
            outputs=all_outputs,
        )

        # clear button
        clear_btn.click(
            fn=lambda: (
                [{"role": "assistant", "content": _WELCOME_MSG}],
                _EMPTY_SOURCES, _EMPTY_SUGGESTIONS, _stats_html(), [],
                "", *[gr.update(value="", visible=False)] * 5,
            ),
            inputs=[],
            outputs=all_outputs,
        )

        # export button
        export_btn.click(
            fn=lambda ch: export_conversation(ch),
            inputs=[chatbot],
            outputs=[export_file],
        ).then(
            fn=lambda f: gr.update(visible=bool(f), value=f),
            inputs=[export_file],
            outputs=[export_file],
        )

        # an example chip fills the input box
        for btn, full_q in zip(ex_btns, EXAMPLE_FULL):
            btn.click(fn=lambda v=full_q: v, inputs=[], outputs=[q_input])

        # a recent query button puts its question back in the input box
        for hbtn in hist_btns:
            hbtn.click(fn=lambda v: v, inputs=[hbtn], outputs=[q_input])

    return demo

# startup
def startup() -> None:
    # load the PDFs and build the pipeline, rebuilding the vector store only when it is missing or its chunk count drifts
    global _all_chunks, _vectorstore, _llm

    logger.info("ComplianceIQ v%s starting", APP_VERSION)

    docs_by_file = load_documents()
    _all_chunks  = chunk_documents(docs_by_file)

    needs_rebuild = not os.path.exists(CHROMA_DIR)
    _vectorstore  = build_vectorstore(_all_chunks, rebuild=needs_rebuild)

    # rebuild the store when its chunk count drifts from the current chunking
    stored  = _vectorstore._collection.count()
    expected = len(_all_chunks)
    if abs(stored - expected) > max(10, expected * 0.05):
        logger.warning("Chunk count drift (%d stored vs %d expected), rebuilding", stored, expected)
        _vectorstore = build_vectorstore(_all_chunks, rebuild=True)

    _init_reranker()
    _llm = build_llm()

    logger.info("Startup complete with %d chunks, %d regulations, reranker=%s",
                len(_all_chunks),
                len({c.metadata.get("regulation") for c in _all_chunks}),
                RERANKER_AVAILABLE)

# run the app locally
if __name__ == "__main__":
    startup()
    demo = build_ui()
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        theme=gr.themes.Base(),
        css=CUSTOM_CSS,
    )
