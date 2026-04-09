"""
core/retriever.py — FAISS + BM25 + cross-encoder hybrid retrieval.

Public API:
  search_bare_acts_auto(query, top_k)
  search_case_laws_auto(query, top_k)
  fuse_bare_act_and_case_law(issue, top_k)  ← absorbed from agents/act_case_fusion_agent
"""

import os
import json
import logging
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import faiss
from typing import Any, Optional

# Try fast JSON parser first; fall back to stdlib
try:
    import orjson as _json_lib
    def _json_load(fp):
        return _json_lib.loads(fp.read())
except ImportError:
    def _json_load(fp):
        return json.load(fp)

logger = logging.getLogger(__name__)

try:
    from crewai import Agent
    from crewai.tools import tool
except Exception:
    Agent = None
    def tool(fn):  # no-op fallback when crewai not available
        return fn

_FORCE_LOCAL_MODEL_FILES = os.environ.get("FORCE_LOCAL_MODEL_FILES", "1").lower() in ("1", "true", "yes")
if _FORCE_LOCAL_MODEL_FILES:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Lazy-loaded models (initialized on first use to save memory)
_embedder = None
_bm25_bare = None
_bm25_case = None
_embedder_lock = threading.Lock()

# Cross-encoder: dual GPU+CPU instances with automatic OOM fallback.
# GPU instance is used when VRAM is available; if it raises OOM the call
# transparently retries on CPU.  A per-GPU lock serialises GPU calls so that
# concurrent threads (parallel dispute retrieval) don't fragment VRAM.
_cross_encoder_gpu = None   # CrossEncoder on CUDA, or False if unavailable
_cross_encoder_cpu = None   # CrossEncoder on CPU (always available fallback)
_ce_gpu_lock = threading.Lock()  # serialise GPU predict() calls
_ce_init_lock = threading.Lock()

# ---------------------------------------------------------------------------
# In-memory index cache — populated once at startup by preload_all_indexes().
# Keys: ("faiss", path) → faiss index object
#       ("bm25",  path) → BM25 object
#       ("chunks",path) → dict
# ---------------------------------------------------------------------------
_index_cache: dict = {}
_index_cache_lock = threading.Lock()   # guards concurrent writes during parallel preload
_faiss_gpu_lock = threading.Lock()
_faiss_gpu_resources = None
_runtime_mode_logged_for_index: set[str] = set()


# ---------------------------------------------------------------------------
# Query normalisation — LLM-driven, with regex fallback
# ---------------------------------------------------------------------------
# The LLM rewrites the query into its fullest canonical form: expands any
# abbreviation, shorthand, or informal reference it recognises (IPC, RERA,
# "cheque bounce", "498A case", "420 IPC", etc.).  Results are cached so
# repeated queries within a session pay zero extra cost.
#
# The static regex table below is kept ONLY as a silent fallback in case the
# LLM call fails (network error, budget exceeded, etc.).

_QUERY_NORM_PROMPT = (
    "You are normalizing an Indian legal search query for vector search over Indian bare acts "
    "and case laws.\n\n"
    "Rewrite the query by expanding all legal abbreviations, shorthand, and informal references "
    "to their full official names. Keep section/article numbers exactly as written. "
    "Do not add legal analysis or interpretation. Return ONLY the rewritten query — "
    "no explanation, no preamble.\n\n"
    "Examples:\n"
    "  IPC 302           →  Indian Penal Code Section 302 murder\n"
    "  498A case         →  Section 498A Indian Penal Code cruelty to wife\n"
    "  cheque bounce     →  dishonour of cheque Negotiable Instruments Act\n"
    "  RERA complaint    →  Real Estate Regulation and Development Act complaint\n"
    "  RTI rejected      →  Right to Information Act application rejected\n"
    "  420 fraud         →  Section 420 Indian Penal Code cheating and dishonesty\n"
    "  DRT recovery      →  Debt Recovery Tribunal recovery of debts due to banks\n"
    "  Art 226 petition  →  Article 226 Constitution of India writ petition High Court\n\n"
    "Query: {query}\n"
    "Normalized:"
)

# In-memory cache: normalized query string → expanded string
_query_norm_cache: dict[str, str] = {}
_query_norm_cache_lock = threading.Lock()

# Regex table kept purely as fallback when the LLM call fails
_LEGAL_ABBREV_FALLBACK: list = [
    (re.compile(r'\bBNSS\b', re.IGNORECASE), 'Bharatiya Nagarik Suraksha Sanhita'),
    (re.compile(r'\bBNS\b',  re.IGNORECASE), 'Bharatiya Nyaya Sanhita'),
    (re.compile(r'\bBSA\b',  re.IGNORECASE), 'Bharatiya Sakshya Adhiniyam'),
    (re.compile(r'\bCrPC\b', re.IGNORECASE), 'Code of Criminal Procedure'),
    (re.compile(r'\bIPC\b',  re.IGNORECASE), 'Indian Penal Code'),
    (re.compile(r'\bCPC\b',  re.IGNORECASE), 'Code of Civil Procedure'),
    (re.compile(r'\bIEA\b',  re.IGNORECASE), 'Indian Evidence Act'),
    (re.compile(r'\bTP\s+Act\b', re.IGNORECASE), 'Transfer of Property Act'),
    (re.compile(r'\bTPA\b',      re.IGNORECASE), 'Transfer of Property Act'),
    (re.compile(r'\bSRA\b',      re.IGNORECASE), 'Specific Relief Act'),
    (re.compile(r'\bICA\b',      re.IGNORECASE), 'Indian Contract Act'),
    (re.compile(r'\bHMA\b', re.IGNORECASE), 'Hindu Marriage Act'),
    (re.compile(r'\bHSA\b', re.IGNORECASE), 'Hindu Succession Act'),
    (re.compile(r'\bHUF\b', re.IGNORECASE), 'Hindu Undivided Family'),
    (re.compile(r'\bID\s+Act\b', re.IGNORECASE), 'Industrial Disputes Act'),
    (re.compile(r'\bIBC\b',  re.IGNORECASE), 'Insolvency and Bankruptcy Code'),
    (re.compile(r'\bSEBI\b', re.IGNORECASE), 'Securities and Exchange Board of India'),
    (re.compile(r'\bNDPS\b',     re.IGNORECASE), 'Narcotic Drugs and Psychotropic Substances Act'),
    (re.compile(r'\bPMLA\b',     re.IGNORECASE), 'Prevention of Money Laundering Act'),
    (re.compile(r'\bPOCSO\b',    re.IGNORECASE), 'Protection of Children from Sexual Offences Act'),
    (re.compile(r'\bPC\s+Act\b', re.IGNORECASE), 'Prevention of Corruption Act'),
    (re.compile(r'\bPWDVA\b',    re.IGNORECASE), 'Protection of Women from Domestic Violence Act'),
    (re.compile(r'\bDV\s+Act\b', re.IGNORECASE), 'Protection of Women from Domestic Violence Act'),
    (re.compile(r'\bIT\s+Act\b', re.IGNORECASE), 'Information Technology Act'),
]


def _regex_normalize_fallback(query: str) -> str:
    """Static regex fallback — used only when the LLM call fails."""
    for pattern, expansion in _LEGAL_ABBREV_FALLBACK:
        query = pattern.sub(expansion, query)
    return query


def normalize_legal_query(query: str) -> str:
    """
    Normalize a legal search query before FAISS embedding and BM25 scoring.

    Uses an LLM (fast tier) to dynamically expand abbreviations, informal
    references, and shorthand into their full canonical form.  Falls back to
    the static regex table if the LLM call fails.  Results are cached in
    memory so repeated queries in the same server session are free.
    """
    q = (query or "").strip()
    if not q:
        return q

    cache_key = q.lower()
    cached = _query_norm_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        from core.llm import ask_llm
        prompt = _QUERY_NORM_PROMPT.format(query=q)
        result = (ask_llm(prompt, task_hint="fast") or "").strip()
        # Sanity-check: result must be non-empty and not absurdly long
        if result and len(result) <= max(len(q) * 8, 200):
            with _query_norm_cache_lock:
                _query_norm_cache[cache_key] = result
            logger.debug("Query normalized (LLM): %r → %r", q, result)
            return result
        logger.warning("LLM query normalization returned unexpected output; using regex fallback")
    except Exception as exc:
        logger.warning("LLM query normalization failed (%s); using regex fallback", exc)

    result = _regex_normalize_fallback(q)
    with _query_norm_cache_lock:
        _query_norm_cache[cache_key] = result
    return result


def _get_embedder():
    """
    Lazy-load the sentence-transformer embedding model, GPU-optimised when available.

    Must match build_indexes._get_embedder() exactly — same device, same precision
    (FP16 on GPU) — so query vectors live in the same space as indexed vectors.
    """
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                from sentence_transformers import SentenceTransformer, models
                import torch
                from config import EMBEDDING_MODEL

                on_gpu = torch.cuda.is_available()
                device = "cuda" if on_gpu else "cpu"

                if on_gpu:
                    torch.backends.cudnn.benchmark = True

                logger.info(f"Loading embedding model '{EMBEDDING_MODEL}' on {device}")
                try:
                    _embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
                    _ = _embedder.encode("test", convert_to_numpy=True)  # smoke-test
                except Exception:
                    logger.info(
                        "Native SentenceTransformer load failed; building with explicit "
                        "mean-pooling for '%s'", EMBEDDING_MODEL,
                    )
                    word_embedding_model = models.Transformer(EMBEDDING_MODEL)
                    pooling_model = models.Pooling(
                        word_embedding_model.get_word_embedding_dimension(),
                        pooling_mode_mean_tokens=True,
                        pooling_mode_cls_token=False,
                        pooling_mode_max_tokens=False,
                    )
                    _embedder = SentenceTransformer(
                        modules=[word_embedding_model, pooling_model], device=device
                    )

                if on_gpu:
                    # FP16 — must match build_indexes._get_embedder() so vectors are
                    # in the same space.  Output is cast to float32 before FAISS search.
                    _embedder = _embedder.half()
                    logger.info("Embedding model loaded in FP16 on %s", torch.cuda.get_device_name(0))

    return _embedder


def _get_cross_encoder_gpu():
    """Lazy-load cross-encoder on CUDA. Returns None if CUDA is unavailable or load failed."""
    global _cross_encoder_gpu
    if _cross_encoder_gpu is None:
        with _ce_init_lock:
            if _cross_encoder_gpu is None:
                import torch
                if torch.cuda.is_available():
                    try:
                        from sentence_transformers import CrossEncoder
                        from config import CROSS_ENCODER_MODEL
                        _cross_encoder_gpu = CrossEncoder(CROSS_ENCODER_MODEL, device="cuda")
                        logger.info("Cross-encoder loaded on CUDA")
                    except Exception as e:
                        logger.warning("Cross-encoder CUDA load failed (%s) — CPU only", e)
                        _cross_encoder_gpu = False  # sentinel: tried, unavailable
                else:
                    _cross_encoder_gpu = False
    return _cross_encoder_gpu if _cross_encoder_gpu is not False else None


def _get_cross_encoder_cpu():
    """Lazy-load cross-encoder on CPU (always-available fallback)."""
    global _cross_encoder_cpu
    if _cross_encoder_cpu is None:
        with _ce_init_lock:
            if _cross_encoder_cpu is None:
                from sentence_transformers import CrossEncoder
                from config import CROSS_ENCODER_MODEL
                _cross_encoder_cpu = CrossEncoder(CROSS_ENCODER_MODEL, device="cpu")
                logger.info("Cross-encoder loaded on CPU")
    return _cross_encoder_cpu


def _predict_cross_encoder(pairs: list) -> list:
    """
    Run cross-encoder inference on (query, text) pairs.

    Strategy:
      1. Attempt GPU inference (serialised via _ce_gpu_lock to prevent VRAM fragmentation
         when multiple dispute threads run concurrently).
      2. On CUDA OOM or any GPU error, transparently fall back to the CPU instance.
    """
    import torch
    gpu_ce = _get_cross_encoder_gpu()
    if gpu_ce is not None:
        try:
            with _ce_gpu_lock:
                return gpu_ce.predict(pairs, show_progress_bar=False)
        except torch.cuda.OutOfMemoryError:
            logger.warning("Cross-encoder GPU OOM (%d pairs) — retrying on CPU", len(pairs))
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning("Cross-encoder GPU error (%s) — retrying on CPU", e)
    return _get_cross_encoder_cpu().predict(pairs, show_progress_bar=False)


def preload_all_indexes():
    """
    Load all FAISS indexes, BM25 indexes, and chunk stores into _index_cache.

    Call once at FastAPI startup so that every query reads from RAM instead of
    re-loading multi-GB files from disk on each request.  Safe to call multiple
    times — already-cached entries are skipped.

    Indexes loaded:
      - Bare acts:      FAISS v2, BM25, chunks
      - Case laws:      FAISS v2, BM25, chunks
      - Case summaries: FAISS v2, BM25, chunks
      - Act summaries:  FAISS v2, BM25, chunks

    Groups are loaded in parallel (ThreadPoolExecutor) to overlap I/O across
    the 4 index sets.  The 3.2 GB caselaws index and 1.2 GB BM25 are the
    dominant cost; parallelism brings total load time from ~500 s → ~120 s.
    """
    from config import (
        BARE_INDEX_V2, BARE_CHUNKS_V2, BARE_BM25_INDEX,
        CASE_INDEX_V2, CASE_CHUNKS_V2, CASE_BM25_INDEX,
        CASE_SUMMARY_INDEX_V2, CASE_SUMMARY_CHUNKS_V2, CASE_SUMMARY_BM25_INDEX,
        ACT_SUMMARY_INDEX_V2, ACT_SUMMARY_CHUNKS_V2, ACT_SUMMARY_BM25_INDEX,
    )
    index_groups = [
        ("bare_acts",       BARE_INDEX_V2,         BARE_CHUNKS_V2,         BARE_BM25_INDEX),
        ("case_laws",       CASE_INDEX_V2,          CASE_CHUNKS_V2,         CASE_BM25_INDEX),
        ("case_summaries",  CASE_SUMMARY_INDEX_V2,  CASE_SUMMARY_CHUNKS_V2, CASE_SUMMARY_BM25_INDEX),
        ("act_summaries",   ACT_SUMMARY_INDEX_V2,   ACT_SUMMARY_CHUNKS_V2,  ACT_SUMMARY_BM25_INDEX),
    ]
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="idx-preload") as pool:
        futures = {
            pool.submit(_preload_index_group, label, fp, cp, bp): label
            for label, fp, cp, bp in index_groups
        }
        for fut in as_completed(futures):
            label = futures[fut]
            exc = fut.exception()
            if exc:
                logger.error("Preload group [%s] raised: %s", label, exc)
    logger.info("Index preload complete (parallel). Cache has %d entries.", len(_index_cache))


def _preload_index_group(label: str, faiss_path: str, chunks_path: str, bm25_path: str):
    """Preload one FAISS/chunks/BM25 group into the shared in-memory cache.

    Thread-safe: uses _index_cache_lock for writes so multiple groups can be
    loaded concurrently by preload_all_indexes() without cache corruption.
    Heavy I/O (faiss.read_index, json.load) is done outside the lock to
    maximise parallelism; the lock is only held during the dict write.
    """
    import time as _time

    # ---------- FAISS ----------
    faiss_key = ("faiss", faiss_path)
    if faiss_key not in _index_cache and os.path.exists(faiss_path):
        try:
            t0 = _time.monotonic()
            idx = faiss.read_index(faiss_path)
            try:
                idx.hnsw.efSearch = 64
            except AttributeError:
                pass
            with _index_cache_lock:
                _index_cache[faiss_key] = idx
            logger.info("Preloaded FAISS [%s]: %d vectors (%.1fs)", label, idx.ntotal, _time.monotonic() - t0)
        except Exception as e:
            logger.error("Preload FAISS [%s] failed: %s", label, e)
    elif not os.path.exists(faiss_path):
        logger.debug("Preload FAISS [%s]: file not found (%s)", label, faiss_path)

    # ---------- Chunks JSON ----------
    chunks_key = ("chunks", chunks_path)
    if chunks_key not in _index_cache and os.path.exists(chunks_path):
        try:
            t0 = _time.monotonic()
            with open(chunks_path, "rb") as f:   # binary for orjson compat
                data = _json_load(f)
            with _index_cache_lock:
                _index_cache[chunks_key] = data
            logger.info("Preloaded chunks [%s]: %d chunks (%.1fs)", label, len(data), _time.monotonic() - t0)
        except Exception as e:
            logger.error("Preload chunks [%s] failed: %s", label, e)
    elif not os.path.exists(chunks_path):
        logger.debug("Preload chunks [%s]: file not found (%s)", label, chunks_path)

    # ---------- BM25 JSON ----------
    bm25_key = ("bm25", bm25_path)
    if bm25_key not in _index_cache and os.path.exists(bm25_path):
        try:
            t0 = _time.monotonic()
            with open(bm25_path, "rb") as f:     # binary for orjson compat
                data = _json_load(f)
            bm25_obj = BM25.from_dict(data)
            with _index_cache_lock:
                _index_cache[bm25_key] = bm25_obj
            logger.info("Preloaded BM25 [%s]: %d docs (%.1fs)", label, bm25_obj.doc_count, _time.monotonic() - t0)
        except Exception as e:
            logger.error("Preload BM25 [%s] failed: %s", label, e)
    elif not os.path.exists(bm25_path):
        logger.debug("Preload BM25 [%s]: file not found (%s)", label, bm25_path)


def preload_interactive_indexes():
    """
    Preload only the indexes needed for the fast interactive legal-opinion path.

    This intentionally skips the full paragraph-level case-law index so chat
    startup stays lighter while deep research/search can still lazy-load it on
    demand.
    """
    from config import (
        ACT_SUMMARY_INDEX_V2,
        ACT_SUMMARY_CHUNKS_V2,
        ACT_SUMMARY_BM25_INDEX,
        BARE_INDEX_V2,
        BARE_CHUNKS_V2,
        BARE_BM25_INDEX,
        CASE_SUMMARY_INDEX_V2,
        CASE_SUMMARY_CHUNKS_V2,
        CASE_SUMMARY_BM25_INDEX,
    )

    groups = [
        ("act_summaries", ACT_SUMMARY_INDEX_V2, ACT_SUMMARY_CHUNKS_V2, ACT_SUMMARY_BM25_INDEX),
        ("bare_acts", BARE_INDEX_V2, BARE_CHUNKS_V2, BARE_BM25_INDEX),
        ("case_summaries", CASE_SUMMARY_INDEX_V2, CASE_SUMMARY_CHUNKS_V2, CASE_SUMMARY_BM25_INDEX),
    ]
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="idx-interactive") as pool:
        futures = {
            pool.submit(_preload_index_group, label, fp, cp, bp): label
            for label, fp, cp, bp in groups
        }
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                logger.error("Interactive preload [%s] raised: %s", futures[fut], exc)
    logger.info("Interactive index preload complete (parallel). Cache has %d entries.", len(_index_cache))


def score_query_document(query: str, document_text: str) -> float:
    """
    Score a single (query, document) pair with the cross-encoder.
    Uses full document text (up to 15K chars) for accurate scoring.
    Used to score web-fetched documents before indexing (only index if score > HIGH_QUALITY).
    """
    if not document_text or not query:
        return 0.0
    # Use up to 15K chars for scoring (full PDF content, not just snippet)
    text = (document_text[:15000]).strip()
    if len(text) < 50:
        return 0.0
    try:
        scores = _predict_cross_encoder([(query, text)])
        return float(scores[0])
    except Exception as e:
        logger.warning(f"Cross-encoder score failed: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# P4: Legal-term overlap boost
# ---------------------------------------------------------------------------
# Extracts section numbers and act names from the query and checks how many
# appear verbatim in the document. Returns a value in [0, 1] representing
# the overlap ratio — added to the cross-encoder score with a configurable weight.
#
# Why this works better than ms-marco alone for legal text:
#   - ms-marco was trained on web passages, not legal judgments
#   - A judgment that cites "section 302 IPC" is MUCH more relevant to a query
#     about "IPC section 302 murder" than one that only mentions "punishment"
#   - The boost nudges that judgment above similarly-scored generic passages

_RE_SECTION_IN_QUERY = re.compile(r'\bsec(?:tion)?\.?\s*(\d+[a-zA-Z]*)', re.IGNORECASE)
_RE_ACT_IN_QUERY = re.compile(r'\b([a-zA-Z][a-zA-Z\s]{3,40}(?:act|code|sanhita|rules|order))\b', re.IGNORECASE)


def legal_term_boost(query: str, text: str) -> float:
    """
    Return a boost value in [0, 1] based on how many legal terms from the query
    (section numbers + act/code names) appear in the document text.

    Used as: final_score = ce_score + LEGAL_TERM_BOOST_WEIGHT * legal_term_boost(query, text)
    """
    if not query or not text:
        return 0.0
    query_lower = query.lower()
    text_lower = text.lower()

    # Extract section numbers (e.g. "302", "498a", "13b")
    sec_nums = [m.group(1).lower() for m in _RE_SECTION_IN_QUERY.finditer(query_lower)]
    # Extract act/code names (e.g. "indian penal code", "protection of women act")
    act_terms = [m.group(1).lower().strip() for m in _RE_ACT_IN_QUERY.finditer(query_lower)]
    # Deduplicate
    act_terms = list(dict.fromkeys(act_terms))

    total = len(sec_nums) + len(act_terms)
    if total == 0:
        return 0.0

    matches = 0
    for sec in sec_nums:
        # Match "section 302", "s. 302" patterns in text
        if re.search(r'\bsec(?:tion)?\.?\s*' + re.escape(sec) + r'\b', text_lower):
            matches += 1
        # Match "302 IPC/BNS/BNSS/BSA/CrPC/CPC/IEA..." patterns in text
        # Includes new Indian criminal codes (BNS, BNSS, BSA) alongside old (IPC, CrPC, IEA)
        elif re.search(
            r'\b' + re.escape(sec) + r'\s+(?:bns|bnss|bsa|ipc|crpc|cpc|iea|mvact|tpa)\b',
            text_lower,
        ):
            matches += 1
    for term in act_terms:
        if len(term) >= 6 and term in text_lower:
            matches += 1

    return matches / total


# Paragraph-type boost for case-law chunks (ratio/reasoning rank above facts)
PARAGRAPH_TYPE_BOOST = {
    "ratio": 0.50,
    "reasoning": 0.35,
    "order": 0.25,
    "arguments": 0.20,
    "facts": 0.10,
    "unknown": 0.0,
}
SECTIONS_CITED_BOOST = 0.25  # per matching section (statute–case link); max 0.5

# Court-tier (authority) boost for case-law chunks
AUTHORITY_BOOST = {
    "supreme_court": 0.50,
    "high_court": 0.30,
    "district_court": 0.10,
    "tribunal": 0.05,
    "unknown": 0.0,
}
# Rejects interlocutory orders, summons, and notices from local case law search results.
# Mirrors the same filter in case_law_discovery/workflow.py.

_RE_INTERLOCUTORY_LOCAL = re.compile(
    r'\b('
    r'interlocutory\s+application'
    r'|i\.a\.\s*(?:no\.?\s*)?\d'
    r'|office\s+report'
    r'|listing\s+order'
    r'|defect\s+(?:no\.?|notice)'
    r'|show\s+cause\s+notice'
    r'|writ\s+of\s+summons'
    r'|chamber\s+summons'
    r'|office\s+objection'
    r'|this\s+is\s+not\s+a\s+judgment'
    r'|adjournment\s+(?:order|application)'
    r'|returnable\s+(?:on|before)'
    r')',
    re.IGNORECASE,
)


def _is_final_judgment_chunk(chunk: dict) -> bool:
    """
    Return True if a local case law chunk looks like a final judgment.

    Only checks the case title/name — not the body text.
    Interlocutory markers in the body are common in final judgments that discuss
    prior interim orders; excluding on body text causes too many false drops.
    """
    title = (chunk.get("case_name") or chunk.get("title") or chunk.get("source") or "")
    if _RE_INTERLOCUTORY_LOCAL.search(title):
        logger.debug("interlocutory filter: excluded chunk with title %r", title[:120])
        return False
    return True


# ---------------------------------------------------------------------------
# BM25 Implementation (lightweight, no external dependency)
# ---------------------------------------------------------------------------

class BM25:
    """
    Simple BM25 (Okapi BM25) implementation for keyword-based retrieval.
    Tokenizes on whitespace + punctuation, case-insensitive.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_dl = 0.0
        self.doc_lengths = []
        self.doc_freqs = {}       # term -> number of docs containing it
        self.term_freqs = []      # list of {term: count} per document
        self.idf_cache = {}

    @staticmethod
    def _tokenize(text: str) -> list:
        """
        Tokenize text for BM25 indexing/scoring.

        Preserves hyphenated section identifiers so that queries and indexed
        text match correctly:
          "10-A"  → "10a"   (query "section 10a" now matches indexed "section 10-A")
          "498-A" → "498a"
          "17-B"  → "17b"

        General pattern: digits followed by a hyphen and a single letter are
        collapsed into a single token (digit-string + lowercase letter).
        Other hyphens (e.g. compound words) are left to the findall step,
        which splits them at the hyphen as before.
        """
        import re
        # Lowercase first, then collapse digit-hyphen-letter section identifiers
        text = re.sub(r'(\d+)-([a-z])\b', r'\1\2', text.lower())
        tokens = re.findall(r"[a-z0-9]+", text)
        return tokens

    def fit(self, documents: list):
        """Build BM25 index from a list of text documents."""
        self.doc_count = len(documents)
        self.doc_lengths = []
        self.term_freqs = []
        self.doc_freqs = {}

        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))

            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.term_freqs.append(tf)

            for token in set(tokens):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.avg_dl = sum(self.doc_lengths) / max(self.doc_count, 1)

        # Pre-compute IDF
        self.idf_cache = {}
        for term, df in self.doc_freqs.items():
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)
            self.idf_cache[term] = idf

    def score(self, query: str, top_k: int = 50) -> list:
        """Score all documents against query. Returns list of (doc_index, score)."""
        query_tokens = self._tokenize(query)
        scores = []

        for i in range(self.doc_count):
            doc_score = 0.0
            dl = self.doc_lengths[i]
            tf_dict = self.term_freqs[i]

            for token in query_tokens:
                if token not in self.idf_cache:
                    continue
                idf = self.idf_cache[token]
                tf = tf_dict.get(token, 0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avg_dl, 1))
                doc_score += idf * (numerator / max(denominator, 0.001))

            if doc_score > 0:
                scores.append((i, doc_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def to_dict(self) -> dict:
        """Serialize BM25 index for saving to disk."""
        return {
            "k1": self.k1,
            "b": self.b,
            "doc_count": self.doc_count,
            "avg_dl": self.avg_dl,
            "doc_lengths": self.doc_lengths,
            "doc_freqs": self.doc_freqs,
            "term_freqs": self.term_freqs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BM25":
        """Deserialize BM25 index from disk."""
        bm25 = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75))
        bm25.doc_count = data["doc_count"]
        bm25.avg_dl = data["avg_dl"]
        bm25.doc_lengths = data["doc_lengths"]
        bm25.doc_freqs = data["doc_freqs"]
        bm25.term_freqs = data["term_freqs"]
        # Rebuild IDF cache
        bm25.idf_cache = {}
        for term, df in bm25.doc_freqs.items():
            bm25.idf_cache[term] = math.log(
                (bm25.doc_count - df + 0.5) / (df + 0.5) + 1.0
            )
        return bm25


def save_bm25_index(bm25: BM25, path: str):
    """Save BM25 index to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bm25.to_dict(), f)
    logger.info(f"BM25 index saved to {path}")


def load_bm25_index(path: str) -> Optional[BM25]:
    """Load BM25 index from cache or disk.

    On cache hit, returns the pre-loaded BM25 object immediately (no disk I/O).
    """
    cache_key = ("bm25", path)
    if cache_key in _index_cache:
        return _index_cache[cache_key]
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        bm25 = BM25.from_dict(data)
        _index_cache[cache_key] = bm25
        return bm25
    except Exception as e:
        logger.error(f"Failed to load BM25 index from {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# FAISS Helpers
# ---------------------------------------------------------------------------

def safe_read_faiss(index_path: str):
    """Read FAISS index from cache or disk. Returns (index, True) or (None, False).

    On cache hit, returns the pre-loaded index immediately (no disk I/O).
    On cache miss, loads from disk, sets HNSW efSearch=64 if applicable, and
    stores in cache for subsequent calls.
    """
    cache_key = ("faiss", index_path)
    if cache_key in _index_cache:
        return _index_cache[cache_key], True
    try:
        if not os.path.exists(index_path):
            return None, False
        idx = faiss.read_index(index_path)
        # Set efSearch for HNSW indexes (noop on flat indexes — attribute missing)
        try:
            idx.hnsw.efSearch = 64
        except AttributeError:
            pass
        _index_cache[cache_key] = idx
        return idx, True
    except Exception as e:
        logger.error(f"Failed to read FAISS index {index_path}: {e}")
        return None, False


def _get_faiss_search_index(index_path: str, cpu_index):
    """
    Prefer a GPU FAISS index for ANN search, with CPU fallback.
    Returns (index_to_search, using_gpu: bool).
    """
    if cpu_index is None:
        return None, False

    # If faiss-gpu is not available (e.g. faiss-cpu wheel), keep CPU path.
    if not hasattr(faiss, "StandardGpuResources") or not hasattr(faiss, "index_cpu_to_gpu"):
        return cpu_index, False

    # Cache GPU clone separately to avoid repeated CPU→GPU conversions.
    gpu_key = ("faiss_gpu", index_path)
    cached_gpu = _index_cache.get(gpu_key)
    if cached_gpu is not None:
        return cached_gpu, True

    global _faiss_gpu_resources
    with _faiss_gpu_lock:
        cached_gpu = _index_cache.get(gpu_key)
        if cached_gpu is not None:
            return cached_gpu, True
        try:
            if _faiss_gpu_resources is None:
                _faiss_gpu_resources = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(_faiss_gpu_resources, 0, cpu_index)
            _index_cache[gpu_key] = gpu_index
            logger.info("FAISS GPU index enabled for %s (%d vectors)", index_path, cpu_index.ntotal)
            return gpu_index, True
        except Exception as e:
            logger.warning("FAISS GPU fallback to CPU for %s: %s", index_path, e)
            return cpu_index, False


def safe_write_faiss(index, path: str) -> bool:
    """Write FAISS index. Returns True on success."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(index, path)
        return True
    except Exception as e:
        logger.error(f"Failed to write FAISS index to {path}: {e}")
        return False


def load_chunks(chunks_path: str) -> dict:
    """Load chunk store from cache or disk. Returns dict or empty dict on failure.

    On cache hit, returns the pre-loaded dict immediately (no disk I/O).
    """
    cache_key = ("chunks", chunks_path)
    if cache_key in _index_cache:
        return _index_cache[cache_key]
    if not os.path.exists(chunks_path):
        return {}
    try:
        with open(chunks_path, encoding="utf-8") as f:
            data = json.load(f)
        _index_cache[cache_key] = data
        return data
    except Exception as e:
        logger.error(f"Failed to load chunks from {chunks_path}: {e}")
        return {}


# ---------------------------------------------------------------------------
# Hybrid Search: FAISS + BM25 + Cross-Encoder Re-Ranking
# ---------------------------------------------------------------------------

def _trace_label_chunk(chunk: dict) -> str:
    """Short display name for UI retrieval traces (bare act section or act summary)."""
    if not isinstance(chunk, dict):
        return ""
    act = (chunk.get("act_name") or "").strip()
    sec = (chunk.get("section_number") or "").strip()
    title = (chunk.get("section_title") or chunk.get("title") or "").strip()
    bits: list[str] = []
    if act:
        bits.append(act)
    if sec:
        bits.append(f"§{sec}")
    if title:
        bits.append(title[:100] + ("…" if len(title) > 100 else ""))
    name = " — ".join(bits) if bits else (chunk.get("case_name") or "").strip()
    return name or "chunk"


def _trace_rows_by_rank(
    ranked: dict[str, int],
    scores: dict[str, float],
    chunks: dict,
) -> list[dict[str, Any]]:
    if not ranked:
        return []
    keys_sorted = sorted(ranked.keys(), key=lambda k: ranked[k])
    out: list[dict[str, Any]] = []
    for k in keys_sorted:
        ch = chunks.get(k) or {}
        out.append({
            "name": _trace_label_chunk(ch),
            "score": round(float(scores.get(k, 0.0)), 6),
        })
    return out


def hybrid_search(
    query: str,
    faiss_index_path: str,
    chunks_path: str,
    bm25_index_path: str,
    faiss_top_k: int = 20,
    bm25_top_k: int = 20,
    rerank_top_k: int = 12,
    min_rerank_score: float = 0.0,
    allowed_acts: Optional[frozenset] = None,
    allowed_cases: Optional[frozenset] = None,
    trace: Optional[list] = None,
    trace_stage: str = "hybrid",
) -> list:
    """
    Four-stage hybrid search:
    1. FAISS semantic search (top faiss_top_k)
    2. BM25 keyword search (top bm25_top_k)
    3. Reciprocal Rank Fusion (RRF) to merge and pre-rank candidates
    4. Cross-encoder re-rank (top rerank_top_k from RRF pool)

    RRF score: sum(1 / (60 + rank_i)) across retrievers.
    Candidates appearing in both retrievers naturally get a higher RRF score
    (they are ranked by both a semantic and a keyword signal).

    allowed_acts: if provided, keep only chunks with act_name in this set (bare-act optimisation).
    allowed_cases: if provided, keep only chunks with case_name in this set (two-tier case-law).
    trace: if a list, append one dict with FAISS/BM25/rerank diagnostics for the UI.
    trace_stage: label for this call (e.g. act_summary vs bare_act_sections).
    """
    # Normalise query: expand Indian legal abbreviations before FAISS + BM25
    # e.g. "IPC section 302" → "Indian Penal Code section 302"
    query = normalize_legal_query(query)

    chunks = load_chunks(chunks_path)
    if not chunks:
        logger.warning(f"No chunks found at {chunks_path}")
        return []

    # --- Stage 1: FAISS vector search (ranked) ---
    # faiss_ranked: {chunk_key: rank}  (rank 0 = most similar)
    # faiss_sim: inner-product similarity (IndexFlatIP + L2-normalized embeddings ≈ cosine)
    faiss_ranked: dict = {}
    faiss_sim: dict[str, float] = {}
    faiss_index, ok = safe_read_faiss(faiss_index_path)
    using_gpu = False
    if ok and faiss_index and faiss_index_path not in _runtime_mode_logged_for_index:
        try:
            embedder = _get_embedder()
            embedder_device = str(getattr(embedder, "device", "unknown"))
        except Exception:
            embedder_device = "unknown"
        ce_gpu_ready = _get_cross_encoder_gpu() is not None
        # Probe FAISS search mode once; this may initialize and cache the GPU clone.
        _probe_search_index, _probe_using_gpu = _get_faiss_search_index(faiss_index_path, faiss_index)
        logger.info(
            "Retrieval runtime mode: embedder=%s, cross_encoder=%s, faiss_search=%s",
            embedder_device,
            "GPU" if ce_gpu_ready else "CPU",
            "GPU" if _probe_using_gpu else "CPU",
        )
        _runtime_mode_logged_for_index.add(faiss_index_path)
    if ok and faiss_index:
        try:
            embedder = _get_embedder()
            query_vec = embedder.encode(
                query, convert_to_numpy=True, normalize_embeddings=True
            )
            search_index, using_gpu = _get_faiss_search_index(faiss_index_path, faiss_index)
            k = min(faiss_top_k, search_index.ntotal)
            if k > 0:
                distances, indices = search_index.search(
                    np.array([query_vec], dtype="float32"), k
                )
                for rank, idx in enumerate(indices[0]):
                    if idx < 0:
                        continue
                    key = str(int(idx))
                    if key not in chunks:
                        continue
                    faiss_ranked[key] = rank
                    faiss_sim[key] = float(distances[0][rank])
        except Exception as e:
            logger.error(f"FAISS search failed (gpu_first={using_gpu}): {e}")

    # --- Stage 2: BM25 keyword search (ranked) ---
    # bm25_ranked: {chunk_key: rank}  (rank 0 = highest BM25 score)
    bm25_ranked: dict = {}
    bm25_score_by_key: dict[str, float] = {}
    bm25 = load_bm25_index(bm25_index_path)
    if bm25:
        try:
            bm25_results = bm25.score(query, top_k=bm25_top_k)
            for rank, (doc_idx, score) in enumerate(bm25_results):
                key = str(doc_idx)
                if key in chunks:
                    bm25_ranked[key] = rank
                    bm25_score_by_key[key] = float(score)
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")

    # --- Stage 3: Reciprocal Rank Fusion (RRF) ---
    # RRF_k=60 is the standard constant (Cormack et al. 2009).
    # A candidate missing from a retriever gets rank = that retriever's top_k
    # (worst possible rank), so it still gets a small contribution.
    _RRF_K = 60
    all_keys = set(faiss_ranked) | set(bm25_ranked)
    if not all_keys:
        logger.info("No candidates from either FAISS or BM25")
        if trace is not None:
            trace.append({
                "stage": trace_stage,
                "query": query[:500],
                "faiss": _trace_rows_by_rank(faiss_ranked, faiss_sim, chunks),
                "bm25": _trace_rows_by_rank(bm25_ranked, bm25_score_by_key, chunks),
                "rerank": [],
                "faiss_score_kind": "inner_product_cosine_normalized",
                "bm25_score_kind": "bm25_raw",
                "note": "no_faiss_bm25_candidates",
            })
        return []

    rrf_scores: dict = {}
    for key in all_keys:
        faiss_rank = faiss_ranked.get(key, faiss_top_k)
        bm25_rank  = bm25_ranked.get(key,  bm25_top_k)
        rrf_scores[key] = 1.0 / (_RRF_K + faiss_rank) + 1.0 / (_RRF_K + bm25_rank)

    # Sort by RRF descending, pass only the top pool to the cross-encoder.
    # The previous implementation used max(..., len(all_keys)) which effectively
    # removed the cap and pushed the entire candidate set to the cross-encoder.
    # That was a major source of latency under multi-query legal retrieval.
    rrf_pool_size = min(
        len(all_keys),
        max(rerank_top_k + 4, math.ceil(rerank_top_k * 1.5)),
    )
    rrf_sorted = sorted(all_keys, key=lambda k: rrf_scores[k], reverse=True)[:rrf_pool_size]
    all_candidate_keys = set(rrf_sorted)

    logger.info(
        "Hybrid search: %d FAISS + %d BM25 → %d RRF candidates (pool for re-rank)",
        len(faiss_ranked), len(bm25_ranked), len(all_candidate_keys),
    )

    # --- Act-level pre-filter (act-first optimisation) ---
    # If the caller identified relevant acts via ActProfileIndex, drop candidates
    # from other acts BEFORE the cross-encoder to reduce the re-ranking workload.
    # Safety: if filtering would leave fewer than 5 candidates, skip the filter
    # (keeps the cross-encoder from starving on edge cases where the profile index
    # mis-identified the relevant acts).
    if allowed_acts:
        filtered_keys = {
            k for k in all_candidate_keys
            if (chunks[k].get("act_name") or "").strip() in allowed_acts
        }
        if len(filtered_keys) >= 5:
            dropped = len(all_candidate_keys) - len(filtered_keys)
            if dropped > 0:
                logger.debug(
                    "Act pre-filter (profile): %d → %d candidates (dropped %d from %d acts not in allowed set)",
                    len(all_candidate_keys), len(filtered_keys), dropped,
                    len({(chunks[k].get("act_name") or "") for k in all_candidate_keys}) - len(allowed_acts),
                )
            all_candidate_keys = filtered_keys
        else:
            # Upgrade to INFO: a bypass means the profile index mis-identified the relevant
            # acts (or they are missing from the profile).  This shows up in normal logs
            # so it is actionable without enabling DEBUG mode.
            logger.info(
                "Act pre-filter (profile): BYPASSED — only %d candidates after filter (< 5 min); "
                "proceeding with all %d candidates. Check act profile coverage for this query.",
                len(filtered_keys), len(all_candidate_keys),
            )

    # --- Case-level pre-filter (two-tier retrieval: case index → paragraph search) ---
    if allowed_cases:
        filtered_keys = {
            k for k in all_candidate_keys
            if (chunks[k].get("case_name") or "").strip() in allowed_cases
        }
        if len(filtered_keys) >= 3:
            all_candidate_keys = filtered_keys
        else:
            logger.debug(
                "Case pre-filter: BYPASSED — only %d candidates in allowed cases; using all %d",
                len(filtered_keys), len(all_candidate_keys),
            )

    def _emit_ui_trace(
        rerank_rows: list[dict[str, Any]],
        *,
        note: str | None = None,
        rerank_score_kind: str = "cross_encoder_plus_boosts",
    ) -> None:
        if trace is None:
            return
        payload: dict[str, Any] = {
            "stage": trace_stage,
            "query": query[:500],
            "faiss": _trace_rows_by_rank(faiss_ranked, faiss_sim, chunks),
            "bm25": _trace_rows_by_rank(bm25_ranked, bm25_score_by_key, chunks),
            "rerank": rerank_rows,
            "faiss_score_kind": "inner_product_cosine_normalized",
            "bm25_score_kind": "bm25_raw",
            "rerank_score_kind": rerank_score_kind,
        }
        if note:
            payload["note"] = note
        trace.append(payload)

    # --- Stage 3: Cross-encoder re-ranking ---
    candidate_chunks = []
    candidate_texts = []
    for key in all_candidate_keys:
        chunk = chunks[key]
        # Use search_text if available (richer), else fall back to text/full_text
        text = (
            chunk.get("search_text")
            or chunk.get("full_text")
            or chunk.get("text")
            or ""
        ).strip()
        if len(text) < 30:
            continue
        candidate_chunks.append((key, chunk))
        # Truncate for cross-encoder to keep hot-path reranking lean.
        candidate_texts.append(text[:1000])

    if not candidate_chunks:
        _emit_ui_trace(
            [],
            note="no_candidates_after_short_text_filter",
        )
        return []

    # Query sections for citation boost (sections_cited already stored in chunk metadata by pipeline)
    query_sections = set()

    try:
        from config import LEGAL_TERM_BOOST_WEIGHT
    except Exception:
        LEGAL_TERM_BOOST_WEIGHT = 0.25

    try:
        pairs = [(query, text) for text in candidate_texts]
        scores = _predict_cross_encoder(pairs)

        # Combine with scores
        scored = []
        for i, (key, chunk) in enumerate(candidate_chunks):
            ce_score = float(scores[i])
            # P4: legal-term overlap boost — rewards docs that cite exact section numbers
            # and act names from the query. Adds to the ms-marco cross-encoder score.
            boost = 0.0
            if LEGAL_TERM_BOOST_WEIGHT > 0:
                boost = LEGAL_TERM_BOOST_WEIGHT * legal_term_boost(query, candidate_texts[i])
            # RRF pre-selection has already promoted docs seen by both retrievers.
            # Track membership for diagnostic fields only (no separate bonus needed).
            in_faiss = key in faiss_ranked
            in_bm25  = key in bm25_ranked

            # Paragraph-type boost (case-law only): ratio/reasoning rank above facts
            para_boost = 0.0
            if chunk.get("doc_type") == "case_law":
                pt = (chunk.get("paragraph_type") or "unknown").lower().strip()
                para_boost = PARAGRAPH_TYPE_BOOST.get(pt, PARAGRAPH_TYPE_BOOST.get("unknown", 0.0))

            # Citation boost (case-law only): chunk cites same act/section as query
            citation_boost = 0.0
            if chunk.get("doc_type") == "case_law" and query_sections:
                cited = chunk.get("sections_cited") or []
                matches = sum(1 for c in cited if c in query_sections)
                if matches > 0:
                    citation_boost = min(SECTIONS_CITED_BOOST * matches, 0.5)

            # Authority (court-tier) boost for case-law chunks
            authority_boost = 0.0
            # Optional: PageRank/citation-graph authority (landmark cases rank higher)
            pagerank_boost = 0.0
            if chunk.get("doc_type") == "case_law":
                binding = (chunk.get("binding_authority") or "unknown").lower().strip()
                authority_boost = AUTHORITY_BOOST.get(binding, AUTHORITY_BOOST.get("unknown", 0.0))
                try:
                    from core.citations import get_case_authority_score
                    case_name = (chunk.get("case_name") or "").strip()
                    year = str(chunk.get("year") or "").strip()
                    if case_name:
                        pr = get_case_authority_score(case_name, year)
                        pagerank_boost = min(pr * 2.0, 0.3)  # cap 0.3 so court tier still dominates
                except Exception:
                    pass

            rerank_score = ce_score + boost + para_boost + citation_boost + authority_boost + pagerank_boost
            if rerank_score >= min_rerank_score:
                result = dict(chunk)
                # Normalize display text: chunks may have search_text/full_text but not "text"
                if "text" not in result or not (result.get("text") or "").strip():
                    result["text"] = (
                        result.get("search_text")
                        or result.get("full_text")
                        or result.get("text")
                        or ""
                    ).strip()
                result["_chunk_key"] = key
                result["_rerank_score"] = rerank_score
                result["_ce_score"] = ce_score              # raw cross-encoder score
                result["_legal_boost"] = boost              # P4 boost component
                result["_rrf_score"] = rrf_scores.get(key, 0.0)  # RRF pre-rank signal
                result["_paragraph_type_boost"] = para_boost
                result["_sections_cited_boost"] = citation_boost
                result["_authority_boost"] = authority_boost
                result["_pagerank_boost"] = pagerank_boost
                result["_in_faiss"] = in_faiss
                result["_in_bm25"] = in_bm25
                scored.append(result)

        # Sort by re-rank score
        scored.sort(key=lambda x: x["_rerank_score"], reverse=True)
        results = scored[:rerank_top_k]

        logger.info(
            f"Re-ranked to {len(results)} results "
            f"(top score: {results[0]['_rerank_score']:.3f})" if results else "Re-ranked to 0 results"
        )
        rerank_rows = [
            {
                "name": _trace_label_chunk(r),
                "score": round(float(r.get("_rerank_score", 0.0)), 5),
            }
            for r in results
        ]
        _emit_ui_trace(rerank_rows)
        return results

    except Exception as e:
        logger.error(f"Cross-encoder re-ranking failed: {e}")
        # Fallback: return FAISS candidates (ordered by RRF score) with rerank_score=0.0
        # to signal that the cross-encoder was unavailable.
        fallback = []
        for key in rrf_sorted:
            if key in chunks:
                chunk = dict(chunks[key])
                if "text" not in chunk or not (chunk.get("text") or "").strip():
                    chunk["text"] = (
                        chunk.get("search_text")
                        or chunk.get("full_text")
                        or chunk.get("text")
                        or ""
                    ).strip()
                chunk["_chunk_key"] = key
                chunk["_rerank_score"] = 0.0   # 0.0 = unknown; cross-encoder unavailable (prev bug: used nonexistent "_score" key)
                chunk["_rerank_fallback"] = True  # flag: cross-encoder was unavailable
                fallback.append(chunk)
        fb = fallback[:rerank_top_k]
        rerank_rows = [
            {
                "name": _trace_label_chunk(c),
                "score": round(float(c.get("_rerank_score", 0.0)), 5),
                "rrf_order": i,
            }
            for i, c in enumerate(fb)
        ]
        _emit_ui_trace(
            rerank_rows,
            note="cross_encoder_unavailable_rrf_order",
            rerank_score_kind="placeholder_zero_ce_down",
        )
        return fb


def search_bare_acts(query: str, top_k: int = 30, trace: Optional[list] = None) -> list:
    """Search bare acts using hybrid retrieval. Returns all relevant sections.
    When act summary index exists, uses two-tier retrieval: act index → section search within those acts.
    If ``trace`` is a list, each hybrid_search appends one diagnostic dict (stage-labelled)."""
    from config import (
        BARE_INDEX_V2,
        BARE_CHUNKS_V2,
        BARE_BM25_INDEX,
        ACT_SUMMARY_INDEX_V2,
        ACT_SUMMARY_CHUNKS_V2,
        ACT_SUMMARY_BM25_INDEX,
    )
    # Two-tier: act summary index → then section search within those acts
    if os.path.isfile(ACT_SUMMARY_INDEX_V2) and os.path.isfile(ACT_SUMMARY_CHUNKS_V2):
        try:
            act_results = hybrid_search(
                query=query,
                faiss_index_path=ACT_SUMMARY_INDEX_V2,
                chunks_path=ACT_SUMMARY_CHUNKS_V2,
                bm25_index_path=ACT_SUMMARY_BM25_INDEX,
                faiss_top_k=12,
                bm25_top_k=12,
                rerank_top_k=8,
                min_rerank_score=0.0,
                trace=trace,
                trace_stage="bare_act_act_summary_index",
            )
            allowed_acts = frozenset(
                (r.get("act_name") or "").strip()
                for r in act_results
                if (r.get("act_name") or "").strip()
            )
            if allowed_acts:
                results = hybrid_search(
                    query=query,
                    faiss_index_path=BARE_INDEX_V2,
                    chunks_path=BARE_CHUNKS_V2,
                    bm25_index_path=BARE_BM25_INDEX,
                    faiss_top_k=24,
                    bm25_top_k=24,
                    rerank_top_k=min(top_k, 10),
                    min_rerank_score=0.0,
                    allowed_acts=allowed_acts,
                    trace=trace,
                    trace_stage="bare_act_section_index_filtered",
                )
            else:
                results = hybrid_search(
                    query=query,
                    faiss_index_path=BARE_INDEX_V2,
                    chunks_path=BARE_CHUNKS_V2,
                    bm25_index_path=BARE_BM25_INDEX,
                    faiss_top_k=20,
                    bm25_top_k=20,
                    rerank_top_k=min(top_k, 10),
                    min_rerank_score=0.0,
                    trace=trace,
                    trace_stage="bare_act_section_index",
                )
        except Exception as e:
            logger.warning("Two-tier bare-act search failed, falling back to single-tier: %s", e)
            results = hybrid_search(
                query=query,
                faiss_index_path=BARE_INDEX_V2,
                chunks_path=BARE_CHUNKS_V2,
                bm25_index_path=BARE_BM25_INDEX,
                faiss_top_k=20,
                bm25_top_k=20,
                rerank_top_k=min(top_k, 10),
                min_rerank_score=0.0,
                trace=trace,
                trace_stage="bare_act_section_index_fallback",
            )
    else:
        results = hybrid_search(
            query=query,
            faiss_index_path=BARE_INDEX_V2,
            chunks_path=BARE_CHUNKS_V2,
            bm25_index_path=BARE_BM25_INDEX,
            faiss_top_k=20,
            bm25_top_k=20,
            rerank_top_k=min(top_k, 10),
            min_rerank_score=0.0,
            trace=trace,
            trace_stage="bare_act_section_index",
        )
    for r in results:
        r["source_tag"] = "LOCAL_DB"
    return results


def search_bare_acts_filtered(
    query: str,
    allowed_acts: frozenset,
    top_k: int = 15,
    trace: Optional[list] = None,
) -> list:
    """
    Act-first variant of search_bare_acts.

    Like search_bare_acts but passes ``allowed_acts`` to hybrid_search so the
    cross-encoder only scores candidates from relevant acts.  Falls back to
    unfiltered search_bare_acts when allowed_acts is empty.

    Parameters
    ----------
    query : str
        Search query (same format as search_bare_acts).
    allowed_acts : frozenset
        Set of act_name strings to restrict to.  Pass the return value of
        Frozenset of act names to restrict search to.
        Empty frozenset → no act-level filter (identical to search_bare_acts).
    top_k : int
        Max results per call (default 15, same as retrieve_bare_acts_for_dispute).
    """
    from config import BARE_INDEX_V2, BARE_CHUNKS_V2, BARE_BM25_INDEX
    if not allowed_acts:
        return search_bare_acts(query, top_k, trace=trace)
    results = hybrid_search(
        query=query,
        faiss_index_path=BARE_INDEX_V2,
        chunks_path=BARE_CHUNKS_V2,
        bm25_index_path=BARE_BM25_INDEX,
        faiss_top_k=20,
        bm25_top_k=20,
        rerank_top_k=min(top_k, 10),
        min_rerank_score=0.0,
        allowed_acts=allowed_acts,
        trace=trace,
        trace_stage="bare_act_section_index_act_filtered",
    )
    for r in results:
        r["source_tag"] = "LOCAL_DB"
    return results


def search_case_laws(query: str, top_k: int = 30) -> list:
    """Search case laws using hybrid retrieval (P2: filters interlocutory/procedural docs).
    When case summary index exists, uses two-tier retrieval: case index → paragraph search (ratio prioritised).
    """
    from config import (
        CASE_INDEX_V2,
        CASE_CHUNKS_V2,
        CASE_BM25_INDEX,
        CASE_SUMMARY_INDEX_V2,
        CASE_SUMMARY_CHUNKS_V2,
        CASE_SUMMARY_BM25_INDEX,
    )
    # Two-tier: case summary index → then paragraph search within those cases
    if (
        os.path.isfile(CASE_SUMMARY_INDEX_V2)
        and os.path.isfile(CASE_SUMMARY_CHUNKS_V2)
    ):
        try:
            case_results = hybrid_search(
                query=query,
                faiss_index_path=CASE_SUMMARY_INDEX_V2,
                chunks_path=CASE_SUMMARY_CHUNKS_V2,
                bm25_index_path=CASE_SUMMARY_BM25_INDEX,
                faiss_top_k=12,
                bm25_top_k=12,
                rerank_top_k=8,
                min_rerank_score=0.0,
            )
            allowed_cases = frozenset(
                (r.get("case_name") or "").strip()
                for r in case_results
                if (r.get("case_name") or "").strip()
            )
            if allowed_cases:
                results = hybrid_search(
                    query=query,
                    faiss_index_path=CASE_INDEX_V2,
                    chunks_path=CASE_CHUNKS_V2,
                    bm25_index_path=CASE_BM25_INDEX,
                    faiss_top_k=15,
                    bm25_top_k=15,
                    rerank_top_k=min(top_k, 6),
                    min_rerank_score=0.0,
                    allowed_cases=allowed_cases,
                )
            else:
                results = hybrid_search(
                    query=query,
                    faiss_index_path=CASE_INDEX_V2,
                    chunks_path=CASE_CHUNKS_V2,
                    bm25_index_path=CASE_BM25_INDEX,
                    faiss_top_k=15,
                    bm25_top_k=15,
                    rerank_top_k=min(top_k, 6),
                    min_rerank_score=0.0,
                )
        except Exception as e:
            logger.warning("Two-tier case search failed, falling back to single-tier: %s", e)
            results = hybrid_search(
                query=query,
                faiss_index_path=CASE_INDEX_V2,
                chunks_path=CASE_CHUNKS_V2,
                bm25_index_path=CASE_BM25_INDEX,
                faiss_top_k=15,
                bm25_top_k=15,
                rerank_top_k=min(top_k, 6),
                min_rerank_score=0.0,
            )
    else:
        results = hybrid_search(
            query=query,
            faiss_index_path=CASE_INDEX_V2,
            chunks_path=CASE_CHUNKS_V2,
            bm25_index_path=CASE_BM25_INDEX,
            faiss_top_k=15,
            bm25_top_k=15,
            rerank_top_k=min(top_k, 6),
            min_rerank_score=0.0,
        )
    for r in results:
        r["source_tag"] = "LOCAL_DB"
    # Citation graph: expand by precedent (cases cited by / citing the top results)
    try:
        from core.citations import (
            get_graph,
            expand_case_names_by_precedent,
            get_chunks_by_case_names,
        )
        graph = get_graph()
        if graph and results:
            case_names = list({(r.get("case_name") or "").strip() for r in results if (r.get("case_name") or "").strip()})
            extra_names, _sections = expand_case_names_by_precedent(case_names, max_extra=10)
            if extra_names:
                chunks_dict = load_chunks(CASE_CHUNKS_V2)
                extra_chunks = get_chunks_by_case_names(chunks_dict, extra_names, max_total=8)
                existing_keys = {r.get("chunk_id") or r.get("_chunk_key") for r in results}
                for c in extra_chunks:
                    if (c.get("chunk_id") or c.get("_chunk_key")) not in existing_keys:
                        c["source_tag"] = "LOCAL_DB"
                        results.append(c)
                        existing_keys.add(c.get("chunk_id") or c.get("_chunk_key"))
            if extra_names and results:
                logger.debug(
                    "Citation expansion: %d extra case(s), %d total case law results",
                    len(extra_names), len(results),
                )
    except Exception as e:
        logger.debug("Citation graph expansion skipped: %s", e)
    # P2: filter out interlocutory/procedural documents from local case law index
    before = len(results)
    results = [r for r in results if _is_final_judgment_chunk(r)]
    if len(results) < before:
        logger.info("P2 filter: removed %d interlocutory/procedural docs from local case law results", before - len(results))
    return results


def _binding_authority_from_court(court: str) -> str:
    """Map a court label to a coarse binding-authority bucket."""
    value = (court or "").strip().lower()
    if not value:
        return "unknown"
    if "supreme court" in value:
        return "supreme_court"
    if "high court" in value:
        return "high_court"
    if "tribunal" in value:
        return "tribunal"
    if "district" in value or "sessions" in value:
        return "district_court"
    return "unknown"


def _extract_query_act_mentions(query: str) -> frozenset[str]:
    """
    Extract explicit Act/Code mentions from the query so retrieval can honor
    user- or model-specified statutes without hardcoding any domain.
    """
    mentions = re.findall(
        r"\b([A-Z][A-Za-z0-9,&(). -]{0,100}\b(?:Act|Code|Rules|Regulation(?:s)?|Procedure)\b(?:,?\s*\d{4})?)",
        query or "",
    )
    cleaned = []
    for item in mentions:
        value = " ".join((item or "").split()).strip(" ,.;:-()")
        if value:
            cleaned.append(value)
    return frozenset(cleaned)


def search_bare_acts_fast(query: str, top_k: int = 6, trace: Optional[list] = None) -> list:
    """
    Lower-latency bare-act search for interactive chat.

    Uses a much smaller candidate pool than the deep-research path and narrows
    section search through the act-summary index when possible.
    """
    from config import (
        ACT_SUMMARY_INDEX_V2,
        ACT_SUMMARY_CHUNKS_V2,
        ACT_SUMMARY_BM25_INDEX,
        BARE_INDEX_V2,
        BARE_CHUNKS_V2,
        BARE_BM25_INDEX,
    )

    # ── Legal Graph fast path ──────────────────────────────────────────────
    # If the query contains an explicit section reference (e.g. "IPC Section 302"),
    # look it up directly in legal.db — zero FAISS, instant answer.
    try:
        from core.legal_graph import find_sections_by_query
        graph_sections = find_sections_by_query(query)
        if graph_sections:
            chunks_dict = load_chunks(BARE_CHUNKS_V2)
            graph_results = []
            for sec in graph_sections:
                for cid in (sec.get("chunk_ids") or []):
                    chunk = chunks_dict.get(str(cid))
                    if chunk and isinstance(chunk, dict):
                        c = dict(chunk)
                        c["source_tag"] = "LOCAL_DB"
                        c["retrieval_profile"] = "graph_direct"
                        c["_rerank_score"] = 1.0   # direct match — highest confidence
                        # Attach cross-referenced sections as context
                        from core.legal_graph import get_cross_referenced_sections
                        xrefs = get_cross_referenced_sections(sec["id"], max_results=3)
                        if xrefs:
                            c["_graph_cross_refs"] = [
                                {"section_number": x["section_number"], "act_name": x["act_name"],
                                 "section_title": x["section_title"], "chunk_ids": x["chunk_ids"]}
                                for x in xrefs
                            ]
                        graph_results.append(c)
            if graph_results:
                # Resolve cross-referenced section chunk_ids into actual chunks and
                # inject them so the caller gets the primary section + related sections
                # in one call — no extra FAISS pass needed.
                seen_chunk_ids: set = {str(r.get("chunk_id") or r.get("id") or "") for r in graph_results}
                xref_chunks: list = []
                for primary in list(graph_results):
                    for xref in (primary.get("_graph_cross_refs") or []):
                        for xcid in (xref.get("chunk_ids") or []):
                            xcid_str = str(xcid)
                            if xcid_str and xcid_str not in seen_chunk_ids:
                                xchunk = chunks_dict.get(xcid_str)
                                if xchunk and isinstance(xchunk, dict):
                                    xc = dict(xchunk)
                                    xc["source_tag"] = "LOCAL_DB"
                                    xc["retrieval_profile"] = "graph_xref"
                                    xc["_rerank_score"] = 0.85
                                    xref_chunks.append(xc)
                                    seen_chunk_ids.add(xcid_str)
                graph_results.extend(xref_chunks)
                logger.debug(
                    "Legal graph direct hit: %d chunk(s) (%d primary, %d xref) for query '%s'",
                    len(graph_results), len(graph_results) - len(xref_chunks),
                    len(xref_chunks), query[:80],
                )
                return graph_results[:top_k]
    except Exception as _graph_err:
        logger.debug("Legal graph fast path skipped: %s", _graph_err)
    # ── End legal graph fast path ──────────────────────────────────────────

    explicit_query_acts = _extract_query_act_mentions(query)
    allowed_acts = explicit_query_acts or None

    if os.path.isfile(ACT_SUMMARY_INDEX_V2) and os.path.isfile(ACT_SUMMARY_CHUNKS_V2):
        try:
            act_results = hybrid_search(
                query=query,
                faiss_index_path=ACT_SUMMARY_INDEX_V2,
                chunks_path=ACT_SUMMARY_CHUNKS_V2,
                bm25_index_path=ACT_SUMMARY_BM25_INDEX,
                faiss_top_k=3,
                bm25_top_k=3,
                rerank_top_k=1,
                min_rerank_score=0.0,
                trace=trace,
                trace_stage="interactive_fast_act_summary",
            )
            allowed_acts = frozenset(
                (r.get("act_name") or "").strip()
                for r in act_results
                if (r.get("act_name") or "").strip()
            ) or allowed_acts
            if explicit_query_acts:
                allowed_acts = explicit_query_acts
        except Exception as e:
            logger.debug("Fast act-summary prefilter failed: %s", e)

    results = hybrid_search(
        query=query,
        faiss_index_path=BARE_INDEX_V2,
        chunks_path=BARE_CHUNKS_V2,
        bm25_index_path=BARE_BM25_INDEX,
        faiss_top_k=6,
        bm25_top_k=6,
        rerank_top_k=min(top_k, 4),
        min_rerank_score=0.0,
        allowed_acts=allowed_acts,
        trace=trace,
        trace_stage="interactive_fast_section_index",
    )
    if allowed_acts:
        best_score = max((float(r.get("_rerank_score", 0) or 0) for r in results), default=-999.0)
        needs_broaden = not results or len(results) < min(2, max(1, top_k)) or best_score < 0.12
        if needs_broaden:
            logger.debug("Fast bare-act search looks sparse with act prefilter; retrying without act filter")
            broader = hybrid_search(
                query=query,
                faiss_index_path=BARE_INDEX_V2,
                chunks_path=BARE_CHUNKS_V2,
                bm25_index_path=BARE_BM25_INDEX,
                faiss_top_k=8,
                bm25_top_k=8,
                rerank_top_k=max(min(top_k, 5), 4),
                min_rerank_score=0.0,
                allowed_acts=None,
                trace=trace,
                trace_stage="interactive_fast_section_index_broadened",
            )
            merged: dict[tuple[str, str], dict] = {}
            for item in list(results) + list(broader):
                key = (
                    (item.get("act_name") or "").strip().lower(),
                    (item.get("section_number") or "").strip().lower(),
                )
                if (item.get("_rerank_score", 0) or 0) > (merged.get(key) or {}).get("_rerank_score", -999):
                    merged[key] = item
            results = sorted(
                merged.values(),
                key=lambda item: float(item.get("_rerank_score", 0) or 0),
                reverse=True,
            )[: max(top_k, 4)]
    for r in results:
        r["source_tag"] = "LOCAL_DB"
        r["retrieval_profile"] = "interactive_fast"
    return results


def search_case_summaries_fast(query: str, top_k: int = 4) -> list:
    """
    Lower-latency precedent search for interactive chat.

    Retrieves compact case-summary documents instead of paragraph-level chunks.
    This is substantially faster and usually good enough for an interactive
    opinion, while deeper search can still use the full case-law index.
    """
    from config import CASE_SUMMARY_INDEX_V2, CASE_SUMMARY_CHUNKS_V2, CASE_SUMMARY_BM25_INDEX

    results = hybrid_search(
        query=query,
        faiss_index_path=CASE_SUMMARY_INDEX_V2,
        chunks_path=CASE_SUMMARY_CHUNKS_V2,
        bm25_index_path=CASE_SUMMARY_BM25_INDEX,
        faiss_top_k=12,
        bm25_top_k=12,
        # Keep final re-ranked summary candidates tight for interactive precision.
        rerank_top_k=min(max(1, top_k), 3),
        min_rerank_score=0.0,
    )
    for r in results:
        r["source_tag"] = "LOCAL_DB"
        r["retrieval_profile"] = "interactive_fast"
        r["is_case_summary"] = True
        r["binding_authority"] = r.get("binding_authority") or _binding_authority_from_court(r.get("court"))
        if not r.get("text"):
            r["text"] = (r.get("full_text") or "").strip()

    # ── Case-section graph path ────────────────────────────────────────────
    # When the query has explicit section references, look up cases that have
    # interpreted those sections via case_section_links in legal.db and add
    # them to the result set — no FAISS needed for this slice.
    try:
        from core.legal_graph import find_sections_by_query, get_cases_interpreting_section
        graph_sections = find_sections_by_query(query)
        if graph_sections:
            chunks_path = CASE_SUMMARY_CHUNKS_V2
            case_chunks_dict = load_chunks(chunks_path)
            existing_names = {(r.get("case_name") or "").strip().lower() for r in results}
            existing_ids = {str(r.get("chunk_id") or r.get("id") or "") for r in results}
            graph_case_chunks: list = []
            for sec in graph_sections:
                for case_info in get_cases_interpreting_section(sec["id"], max_results=5):
                    case_name = (case_info.get("case_name") or "").strip()
                    if not case_name or case_name.lower() in existing_names:
                        continue
                    # Find matching chunks by case_name in the chunk store
                    for cid, chunk in case_chunks_dict.items():
                        if not isinstance(chunk, dict):
                            continue
                        if (chunk.get("case_name") or "").strip().lower() != case_name.lower():
                            continue
                        cid_str = str(cid)
                        if cid_str in existing_ids:
                            continue
                        c = dict(chunk)
                        c["source_tag"] = "LOCAL_DB"
                        c["retrieval_profile"] = "graph_case_section"
                        c["_rerank_score"] = 0.9   # high confidence: direct section match
                        c["is_case_summary"] = True
                        c["binding_authority"] = c.get("binding_authority") or _binding_authority_from_court(c.get("court"))
                        if not c.get("text"):
                            c["text"] = (c.get("full_text") or "").strip()
                        graph_case_chunks.append(c)
                        existing_ids.add(cid_str)
                        existing_names.add(case_name.lower())
                        break  # one chunk per case for summary path
            if graph_case_chunks:
                results = results + graph_case_chunks
                logger.debug(
                    "Case-section graph path: +%d case(s) from case_section_links", len(graph_case_chunks)
                )
    except Exception as _csg_err:
        logger.debug("Case-section graph path skipped: %s", _csg_err)
    # ── End case-section graph path ────────────────────────────────────────

    return results[:top_k]


def search_bare_acts_runtime(
    query: str, top_k: int = 8, allow_legacy_fallback: bool = True, trace: Optional[list] = None
) -> list:
    """
    Runtime rescue search for interactive chat.

    This stays local-first but broadens more gracefully than the benchmark-style
    auto path: direct v2 search with slightly wider candidate pools, then legacy
    local fallback only if v2 yields nothing.
    """
    from config import BARE_INDEX_V2, BARE_CHUNKS_V2, BARE_BM25_INDEX

    normalized = normalize_legal_query(query)
    explicit_query_acts = _extract_query_act_mentions(normalized)
    results: list = []
    if os.path.isfile(BARE_INDEX_V2) and os.path.isfile(BARE_CHUNKS_V2):
        results = hybrid_search(
            query=normalized,
            faiss_index_path=BARE_INDEX_V2,
            chunks_path=BARE_CHUNKS_V2,
            bm25_index_path=BARE_BM25_INDEX,
            faiss_top_k=10,
            bm25_top_k=10,
            rerank_top_k=min(max(top_k, 6), 8),
            min_rerank_score=0.0,
            allowed_acts=explicit_query_acts or None,
            trace=trace,
            trace_stage="bare_act_runtime_rescue_v2",
        )
        if explicit_query_acts and (not results or len(results) < min(2, max(1, top_k // 2))):
            broader = hybrid_search(
                query=normalized,
                faiss_index_path=BARE_INDEX_V2,
                chunks_path=BARE_CHUNKS_V2,
                bm25_index_path=BARE_BM25_INDEX,
                faiss_top_k=10,
                bm25_top_k=10,
                rerank_top_k=min(max(top_k, 6), 8),
                min_rerank_score=0.0,
                allowed_acts=None,
                trace=trace,
                trace_stage="bare_act_runtime_rescue_v2_broad",
            )
            merged: dict[tuple[str, str], dict] = {}
            for item in list(results) + list(broader):
                key = (
                    (item.get("act_name") or "").strip().lower(),
                    (item.get("section_number") or "").strip().lower(),
                )
                if (item.get("_rerank_score", 0) or 0) > (merged.get(key) or {}).get("_rerank_score", -999):
                    merged[key] = item
            results = sorted(
                merged.values(),
                key=lambda item: float(item.get("_rerank_score", 0) or 0),
                reverse=True,
            )[: max(top_k, 6)]
    best_score = max((float(r.get("_rerank_score", 0) or 0) for r in results), default=-999.0)
    if allow_legacy_fallback and (not results or len(results) < min(3, max(1, top_k // 2)) or best_score < 0.08):
        logger.info("Runtime bare-act rescue: broadening to legacy local index for query '%s'", normalized[:120])
        legacy_results = search_bare_acts_legacy(normalized, top_k=max(top_k * 2, 12), min_sim=0.25)
        merged: dict[tuple[str, str], dict] = {}
        for item in list(results) + list(legacy_results):
            key = (
                (item.get("act_name") or "").strip().lower(),
                (item.get("section_number") or "").strip().lower(),
            )
            if (item.get("_rerank_score", 0) or 0) > (merged.get(key) or {}).get("_rerank_score", -999):
                merged[key] = item
        results = sorted(
            merged.values(),
            key=lambda item: float(item.get("_rerank_score", 0) or 0),
            reverse=True,
        )[: max(top_k, 6)]
    for r in results:
        r["source_tag"] = "LOCAL_DB"
        r["retrieval_profile"] = "interactive_runtime_rescue"
    return results


def search_case_laws_runtime(query: str, top_k: int = 8, allow_legacy_fallback: bool = True) -> list:
    """
    Runtime rescue search for interactive case-law retrieval.

    Uses a broader local summary/full-text search path and only falls back to the
    legacy local index when the v2 runtime path returns nothing.
    """
    from config import (
        CASE_SUMMARY_INDEX_V2,
        CASE_SUMMARY_CHUNKS_V2,
        CASE_SUMMARY_BM25_INDEX,
        CASE_INDEX_V2,
        CASE_CHUNKS_V2,
        CASE_BM25_INDEX,
    )

    normalized = normalize_legal_query(query)
    results: list = []
    summary_results: list = []
    if os.path.isfile(CASE_SUMMARY_INDEX_V2) and os.path.isfile(CASE_SUMMARY_CHUNKS_V2):
        summary_results = hybrid_search(
            query=normalized,
            faiss_index_path=CASE_SUMMARY_INDEX_V2,
            chunks_path=CASE_SUMMARY_CHUNKS_V2,
            bm25_index_path=CASE_SUMMARY_BM25_INDEX,
            faiss_top_k=18,
            bm25_top_k=18,
            rerank_top_k=3,
            min_rerank_score=0.0,
        )
        for r in summary_results:
            r["is_case_summary"] = True
            r["binding_authority"] = r.get("binding_authority") or _binding_authority_from_court(r.get("court"))
            if not r.get("text"):
                r["text"] = (r.get("full_text") or "").strip()
    results = list(summary_results)
    best_summary_score = max((float(r.get("_rerank_score", 0) or 0) for r in summary_results), default=-999.0)
    needs_full_text = not summary_results or len(summary_results) < min(2, max(1, top_k // 2)) or best_summary_score < 0.08
    if needs_full_text and os.path.isfile(CASE_INDEX_V2) and os.path.isfile(CASE_CHUNKS_V2):
        full_text_results = hybrid_search(
            query=normalized,
            faiss_index_path=CASE_INDEX_V2,
            chunks_path=CASE_CHUNKS_V2,
            bm25_index_path=CASE_BM25_INDEX,
            faiss_top_k=20,
            bm25_top_k=20,
            rerank_top_k=3,
            min_rerank_score=0.0,
            allowed_cases=None,
        )
        merged: dict[str, dict] = {}
        for item in list(summary_results) + list(full_text_results):
            key = item.get("_chunk_key") or (
                (item.get("case_name") or "").strip().lower() + "|" + (item.get("court") or "").strip().lower()
            )
            if (item.get("_rerank_score", 0) or 0) > (merged.get(key) or {}).get("_rerank_score", -999):
                merged[key] = item
        results = list(merged.values())
    results = [r for r in results if _is_final_judgment_chunk(r)]
    best_case_score = max((float(r.get("_rerank_score", 0) or 0) for r in results), default=-999.0)
    if allow_legacy_fallback and (not results or len(results) < min(2, max(1, top_k // 2)) or best_case_score < 0.08):
        logger.info("Runtime case-law rescue: broadening to legacy local index for query '%s'", normalized[:120])
        legacy_results = [r for r in search_case_laws_legacy(normalized, top_k=max(top_k * 2, 12), min_sim=0.25) if _is_final_judgment_chunk(r)]
        merged: dict[str, dict] = {}
        for item in list(results) + list(legacy_results):
            key = item.get("_chunk_key") or (
                (item.get("case_name") or "").strip().lower() + "|" + (item.get("court") or "").strip().lower()
            )
            if (item.get("_rerank_score", 0) or 0) > (merged.get(key) or {}).get("_rerank_score", -999):
                merged[key] = item
        results = sorted(
            merged.values(),
            key=lambda item: float(item.get("_rerank_score", 0) or 0),
            reverse=True,
        )[: max(top_k, 6)]
    for r in results:
        r["source_tag"] = "LOCAL_DB"
        r["retrieval_profile"] = "interactive_runtime_rescue"
        r["binding_authority"] = r.get("binding_authority") or _binding_authority_from_court(r.get("court"))
        if not r.get("text"):
            r["text"] = (r.get("full_text") or "").strip()
    return results


# ---------------------------------------------------------------------------
# Fallback: Search legacy v1 index if v2 not yet built
# ---------------------------------------------------------------------------

def search_bare_acts_legacy(query: str, top_k: int = 50, min_sim: float = 0.45) -> list:
    """Search bare acts from legacy v1 FAISS index (blind chunking)."""
    from config import BARE_INDEX, BARE_CHUNKS
    index, ok = safe_read_faiss(BARE_INDEX)
    if not ok or not index:
        return []
    chunks = load_chunks(BARE_CHUNKS)
    if not chunks:
        return []

    embedder = _get_embedder()
    query_vec = embedder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    k = min(top_k, index.ntotal)
    if k <= 0:
        return []

    distances, indices = index.search(np.array([query_vec], dtype="float32"), k)
    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        score = float(distances[0][rank])
        if score < min_sim:
            continue
        key = str(idx)
        if key in chunks:
            chunk = dict(chunks[key])
            chunk["_rerank_score"] = score
            chunk["source_tag"] = "LOCAL_DB"
            results.append(chunk)
    return results


def search_case_laws_legacy(query: str, top_k: int = 15, min_sim: float = 0.40) -> list:
    """Search case laws from legacy v1 FAISS index."""
    from config import CASE_INDEX, CASE_CHUNKS
    index, ok = safe_read_faiss(CASE_INDEX)
    if not ok or not index:
        return []
    chunks = load_chunks(CASE_CHUNKS)
    if not chunks:
        return []

    embedder = _get_embedder()
    query_vec = embedder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    k = min(top_k, index.ntotal)
    if k <= 0:
        return []

    distances, indices = index.search(np.array([query_vec], dtype="float32"), k)
    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        score = float(distances[0][rank])
        if score < min_sim:
            continue
        key = str(idx)
        if key in chunks:
            chunk = dict(chunks[key])
            chunk["_rerank_score"] = score
            chunk["source_tag"] = "LOCAL_DB"
            results.append(chunk)
    return results


def search_bare_acts_auto(query: str, top_k: int = 30, trace: Optional[list] = None) -> list:
    """
    Auto-detect v2 or legacy index and search accordingly.

    IMPORTANT (research integrity):
    When the v2 index exists, we always use v2 — even if it returns 0 results.
    We deliberately do NOT fall back to the legacy FAISS-only index when v2 is
    present, because silent fallback would mislabel ablation results:
      - batch_runner mode='full_pipeline' would secretly run FAISS-only (v1)
      - Metrics would be wrong and not reproducible

    If v2 returns 0 results, that IS the correct answer for that query on that
    index — it means no relevant sections were found above the score threshold.
    """
    from config import BARE_INDEX_V2
    if os.path.exists(BARE_INDEX_V2):
        results = search_bare_acts(query, top_k, trace=trace)
        if not results:
            logger.warning(
                "search_bare_acts_auto: v2 index found but returned 0 results for this query. "
                "NOT falling back to legacy. Check index integrity or lower min_rerank_score."
            )
        return results
    # v2 index not yet built — use legacy with a clear log message
    logger.warning(
        "search_bare_acts_auto: v2 index not found at %s — "
        "using legacy FAISS-only retrieval (no BM25, no cross-encoder).", BARE_INDEX_V2
    )
    return search_bare_acts_legacy(query, top_k)


def search_case_laws_auto(query: str, top_k: int = 30) -> list:
    """
    Auto-detect v2 or legacy index and search accordingly.

    Same research-integrity policy as search_bare_acts_auto: no silent fallback
    when v2 index exists. See docstring above for rationale.
    """
    from config import CASE_INDEX_V2
    if os.path.exists(CASE_INDEX_V2):
        results = search_case_laws(query, top_k)
        if not results:
            logger.warning(
                "search_case_laws_auto: v2 index found but returned 0 results for this query. "
                "NOT falling back to legacy. Check index integrity or lower min_rerank_score."
            )
        return results
    logger.warning(
        "search_case_laws_auto: v2 index not found at %s — "
        "using legacy FAISS-only retrieval (no BM25, no cross-encoder).", CASE_INDEX_V2
    )
    return search_case_laws_legacy(query, top_k)


# ---------------------------------------------------------------------------
# Fusion — bare act + case law combined (absorbed from act_case_fusion_agent)
# ---------------------------------------------------------------------------

# crewai @tool imported at top of file (or no-op fallback defined there)

from core.llm import CREWAI_LLM


def _search_bare_acts(issue: str, top_k: int = 20) -> list:
    """Search bare acts using v2 hybrid retriever (FAISS + BM25 + cross-encoder)."""
    try:
        from core.retriever import search_bare_acts_auto
        results = search_bare_acts_auto(issue, top_k=top_k)
        return [
            {
                "source": r.get("source", ""),
                "text": r.get("text", ""),
                "act_name": r.get("act_name", ""),
                "section_number": r.get("section_number", ""),
                "rerank_score": r.get("rerank_score", 0),
            }
            for r in results if r.get("text")
        ]
    except Exception:
        return []


def _search_case_laws(issue: str, top_k: int = 20) -> list:
    """Search case laws using v2 hybrid retriever (FAISS + BM25 + cross-encoder)."""
    try:
        from core.retriever import search_case_laws_auto
        results = search_case_laws_auto(issue, top_k=top_k)
        return [
            {
                "source": r.get("source", ""),
                "text": r.get("text", ""),
                "case_name": r.get("case_name", ""),
                "court": r.get("court", ""),
                "year": r.get("year", ""),
                "rerank_score": r.get("rerank_score", 0),
            }
            for r in results if r.get("text")
        ]
    except Exception:
        return []


def _web_fallback_bare_acts(issue: str, max_results: int = 5) -> list:
    """Search web for bare acts via tiered search + auto-enricher (v2 pipeline)."""
    try:
        from core.enricher import search_for_gaps
        from core.enricher import enrich_from_gap_results
        gaps = [{"aspect": "bare_act", "search_queries": [f"{issue} India bare act section"]}]
        gap_results = search_for_gaps(gaps, jurisdiction_state="")
        enrich_from_gap_results(gap_results, search_type="bare_act")
        results = []
        for gap in gap_results.get("results", []):
            for r in gap.get("search_results", []):
                results.append({
                    "source": r.get("title", "Internet"),
                    "text": r.get("snippet", ""),
                    "act_name": r.get("title", "Unknown"),
                    "url": r.get("url", ""),
                })
                if len(results) >= max_results:
                    break
        return results[:max_results]
    except Exception:
        return []


def _web_fallback_case_laws(issue: str, max_results: int = 5) -> list:
    """Search web for case laws via tiered search + auto-enricher (v2 pipeline)."""
    try:
        from core.enricher import search_for_gaps
        from core.enricher import enrich_from_gap_results
        gaps = [{"aspect": "case_law", "search_queries": [f"{issue} Supreme Court India judgment"]}]
        gap_results = search_for_gaps(gaps, jurisdiction_state="")
        enrich_from_gap_results(gap_results, search_type="case_law")
        results = []
        for gap in gap_results.get("results", []):
            for r in gap.get("search_results", []):
                results.append({
                    "source": r.get("title", "Internet"),
                    "text": r.get("snippet", ""),
                    "url": r.get("url", ""),
                })
                if len(results) >= max_results:
                    break
        return results[:max_results]
    except Exception:
        return []


@tool
def fuse_bare_act_and_case_law(issue: str):
    """
    Combine relevant Bare Act sections and Case Laws for a given legal issue.
    Uses v2 hybrid retriever (FAISS + BM25 + cross-encoder) first;
    falls back to tiered web search if local results are insufficient.
    """
    result = {
        "issue": issue,
        "bare_act_sections": [],
        "case_laws": [],
    }

    # ---- Bare Act Retrieval (v2 hybrid, then web fallback) ----
    result["bare_act_sections"] = _search_bare_acts(issue)
    if not result["bare_act_sections"]:
        result["bare_act_sections"] = _web_fallback_bare_acts(issue, max_results=5)

    # ---- Case Law Retrieval (v2 hybrid, then web fallback) ----
    result["case_laws"] = _search_case_laws(issue)
    if not result["case_laws"]:
        result["case_laws"] = _web_fallback_case_laws(issue, max_results=5)

    return result


if Agent is None:
    act_case_fusion_agent = None
else:
    try:
        act_case_fusion_agent = Agent(
            role="Legal Research Synthesizer",
            goal="Combine Bare Act provisions and Case Laws into a single structured legal context",
            backstory=(
                "You strictly aggregate authoritative legal material. "
                "You never reason, interpret, retry, or invent. "
                "If data is missing, you return empty sections."
            ),
            llm=CREWAI_LLM,
            tools=[fuse_bare_act_and_case_law],
            verbose=True
        )
    except Exception as e:
        logger.warning("Failed to initialize act_case_fusion_agent: %s", e)
        act_case_fusion_agent = None