"""
Central data path config. All app data lives under legal_database/ (vector store,
chat history, json_output, bare acts/case laws lists). No external DATA_ROOT/vector_store.
"""
import os

# Optional: load .env (requires python-dotenv)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.normpath(__file__)))
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(_PROJECT_ROOT, ".env")
    load_dotenv(_env_path)
except ImportError:
    pass
except Exception:
    pass

# Legal database: single source for all app data (vector store, chat history, json_output).
LEGAL_DATABASE_DIR = os.path.join(_PROJECT_ROOT, "legal_database")
LEGAL_DB_JSON_OUTPUT = os.path.join(LEGAL_DATABASE_DIR, "json_output")
LEGAL_DB_RAW_DATA = os.path.join(LEGAL_DATABASE_DIR, "raw_data")
LEGAL_DB_VECTOR_STORE = os.path.join(LEGAL_DATABASE_DIR, "vector_store")
LEGAL_DB_CHAT_HISTORY = os.path.join(LEGAL_DATABASE_DIR, "chat_history")

# Data source: legal_database (default) or legacy DATA_ROOT/Google Drive.
# Default is legal_database so you can run e.g. python legal_database/build_indexes.py without setting env.
USE_LEGAL_DATABASE = (
    os.environ.get("NYAYMALAW_DATA_SOURCE", "legal_database").strip().lower() == "legal_database"
)
CHAT_HISTORY_DIR = LEGAL_DB_CHAT_HISTORY
DB_PATH = os.path.join(CHAT_HISTORY_DIR, "app.db")
VECTOR_STORE = LEGAL_DB_VECTOR_STORE
BARE_ACTS_DIR = os.path.join(LEGAL_DB_RAW_DATA, "BareActs")
CASELAW_DIR = os.path.join(LEGAL_DB_RAW_DATA, "CaseLaws")

# Legacy alias (DATA_ROOT no longer used for vector store / chat / bare acts / case laws)
DATA_ROOT = LEGAL_DATABASE_DIR

# Feedback Log workbook — auto-filled by feedback_logger.py after every interaction.
# Override via FEEDBACK_LOG_PATH env var or set this to an absolute path.
FEEDBACK_LOG_PATH = os.environ.get(
    "FEEDBACK_LOG_PATH",
    os.path.join(_PROJECT_ROOT, "Nyaymalaw_Feedback_Log_v3.xlsx"),
).strip()

# Optional: shareable Google Drive folder URLs for local files (when using NYAYMALAW_DATA_ROOT on Drive).
# When set, bare act / case law titles without a web URL will link to these folders.
# Example: https://drive.google.com/drive/folders/YOUR_FOLDER_ID
GOOGLE_DRIVE_BARE_ACTS_FOLDER_URL = os.environ.get("NYAYMALAW_GOOGLE_DRIVE_BARE_ACTS_URL", "").strip()
GOOGLE_DRIVE_CASE_LAWS_FOLDER_URL = os.environ.get("NYAYMALAW_GOOGLE_DRIVE_CASE_LAWS_URL", "").strip()

# Vector store files — legacy (v1 blind chunking)
BARE_INDEX = os.path.join(VECTOR_STORE, "bareacts.index")
BARE_CHUNKS = os.path.join(VECTOR_STORE, "bareacts_chunks.json")
CASE_INDEX = os.path.join(VECTOR_STORE, "caselaws.index")
CASE_CHUNKS = os.path.join(VECTOR_STORE, "caselaws_chunks.json")

# Vector store files — v2 (section-level / paragraph-level smart chunking)
BARE_INDEX_V2 = os.path.join(VECTOR_STORE, "bareacts_v2.index")
BARE_CHUNKS_V2 = os.path.join(VECTOR_STORE, "bareacts_v2_chunks.json")
CASE_INDEX_V2 = os.path.join(VECTOR_STORE, "caselaws_v2.index")
CASE_CHUNKS_V2 = os.path.join(VECTOR_STORE, "caselaws_v2_chunks.json")
# Case-level summary index (structured embedding: ratio + issues + sections)
CASE_SUMMARY_INDEX_V2 = os.path.join(VECTOR_STORE, "case_summaries_v2.index")
CASE_SUMMARY_CHUNKS_V2 = os.path.join(VECTOR_STORE, "case_summaries_v2_chunks.json")
CASE_SUMMARY_BM25_INDEX = os.path.join(VECTOR_STORE, "case_summaries_bm25.json")

# Act-level summary index (structured embedding: preamble + key sections)
ACT_SUMMARY_INDEX_V2 = os.path.join(VECTOR_STORE, "act_summaries_v2.index")
ACT_SUMMARY_CHUNKS_V2 = os.path.join(VECTOR_STORE, "act_summaries_v2_chunks.json")
ACT_SUMMARY_BM25_INDEX = os.path.join(VECTOR_STORE, "act_summaries_bm25.json")

# BM25 index files (for hybrid search)
BARE_BM25_INDEX = os.path.join(VECTOR_STORE, "bareacts_bm25.json")
CASE_BM25_INDEX = os.path.join(VECTOR_STORE, "caselaws_bm25.json")

# Act-level profile index — one rich doc per act, used for act-first identification
# before section-level hybrid search.  Built from BARE_CHUNKS_V2 at index-rebuild time
# and lazy-rebuilt on first use if the files are missing / stale.
ACT_PROFILES_META  = os.path.join(VECTOR_STORE, "act_profiles_meta.json")   # {act_name: profile_text}
ACT_PROFILES_BM25  = os.path.join(VECTOR_STORE, "act_profiles_bm25.json")   # BM25 index over profiles

# Citation graph: case → interprets → section, case → cites → case. Built from CASE_CHUNKS_V2.
CITATION_GRAPH_PATH = os.path.join(VECTOR_STORE, "citation_graph.json")

# Legal knowledge graph: Act → Chapter → Section hierarchy + Section cross-references.
# Built by scripts/build_legal_graph.py from BARE_CHUNKS_V2 + CASE_CHUNKS_V2.
LEGAL_GRAPH_DB = os.path.join(VECTOR_STORE, "legal.db")

# Web references table (articles/news that aren't primary sources)
WEB_REFERENCES_DB = os.path.join(DATA_ROOT, "web_references.json")

# Bare act summary index (act_name -> summary text)
BARE_ACT_SUMMARY_INDEX_PATH = os.path.join(DATA_ROOT, "bare_act_summary_index.json")
# Case law summary index (signature -> summary text)
CASE_LAW_SUMMARY_INDEX_PATH = os.path.join(DATA_ROOT, "case_law_summary_index.json")


# ---------------------------------------------------------------------------
# Embedding & re-ranker model names
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "thenlper/gte-base"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# ---------------------------------------------------------------------------
# Web search quality improvements (P0-P5)
# ---------------------------------------------------------------------------

# P0: Citation-count multiplier weight for IK results.
# final_score = ce_score * (1 + CITATION_BOOST_WEIGHT * log(1 + numciting))
# Set to 0.0 to disable. Default 0.15 gives a ~25% boost to heavily-cited judgments.
CITATION_BOOST_WEIGHT = float(os.environ.get("CITATION_BOOST_WEIGHT", "0.15"))

# P4: Legal-term overlap boost weight added to cross-encoder score.
# score = ce_score + LEGAL_TERM_BOOST_WEIGHT * overlap_ratio
# Set to 0.0 to disable. Default 0.35 (raised from 0.25) for stronger section-match signal.
LEGAL_TERM_BOOST_WEIGHT = float(os.environ.get("LEGAL_TERM_BOOST_WEIGHT", "0.35"))

# Intersection bonus — added to re-rank score when a chunk appears in BOTH
# the FAISS top-k AND the BM25 top-k.  Semantically similar AND exact-keyword
# match is the strongest retrieval signal.  Set 0.0 to disable.  Default 0.15.
FAISS_BM25_INTERSECTION_BONUS = float(os.environ.get("FAISS_BM25_INTERSECTION_BONUS", "0.15"))

# P5: Number of IK search result pages to fetch per query (each page ≈ 10-20 results).
# Higher = larger candidate pool; lower = fewer API calls. Default 5.
IK_SEARCH_MAX_PAGES = int(os.environ.get("IK_SEARCH_MAX_PAGES", "5"))

# ---------------------------------------------------------------------------
# Domain whitelists for tiered internet search
# ---------------------------------------------------------------------------

# Source tags: OFFICIAL = government sources (state/central legislation + court judgments), LEGAL_PORTAL = known portals, NEWS_REFERENCE = news, rest discarded
OFFICIAL_SOURCE_TAG = "OFFICIAL"

# Tier 2: Official government sources (legislation: India Code, legislative.gov.in; judgments: Supreme Court, High Courts)
TIER2_OFFICIAL_COURT_DOMAINS = (
    
    "indiacode.nic.in",
    "legislative.gov.in",
    "egazette.nic.in",
    "lawcommissionofindia.nic.in",
   
    #Courts
    "judgments.ecourts.gov.in",
    "main.sci.gov.in",
    "sci.gov.in",
    "api.sci.gov.in",
    "tshc.gov.in",          # Telangana State High Court (current domain)
)

# Tier 3: Trusted legal portals
TIER3_LEGAL_PORTAL_DOMAINS = (
    "indiankanoon.org",
    "livelaw.in",
    "scobserver.in",
    "barandbench.com",
    "lawctopus.com",
    "scconline.com",
    "casemine.com",
    "legalbites.in",
    "latestlaws.com",
)

# Tier 4: Mainstream newspapers (context only, NOT legal authority)
TIER4_NEWSPAPER_DOMAINS = (
    "thehindu.com",
    "indianexpress.com",
    "timesofindia.indiatimes.com",
    "timesofindia.com",
    "ndtv.com",
    "hindustantimes.com",
    "deccanherald.com",
    "theprint.in",
    "thewire.in",
    "economictimes.indiatimes.com",
    "economictimes.com",
)

# Blocked: social media, user-generated content, personal blogs
BLOCKED_DOMAINS = (
    "twitter.com", "x.com",
    "facebook.com", "fb.com",
    "linkedin.com",
    "instagram.com",
    "youtube.com",
    "reddit.com",
    "quora.com",
    "medium.com",
    "pinterest.com",
    "wikipedia.org",
    "blogspot.com",
    "wordpress.com",
    "tumblr.com",
)

# All allowed domains combined (for quick lookup)
ALL_ALLOWED_DOMAINS = TIER2_OFFICIAL_COURT_DOMAINS + TIER3_LEGAL_PORTAL_DOMAINS + TIER4_NEWSPAPER_DOMAINS

# ---------------------------------------------------------------------------
# Freemium tier configuration
# ---------------------------------------------------------------------------

TIER_FREE = "free"
TIER_PREMIUM = "premium"

# Daily query limits per tier
TIER_QUERY_LIMITS = {
    TIER_FREE: 10,
    TIER_PREMIUM: 100,
}

# Features available per tier
TIER_FEATURES = {
    TIER_FREE: {
        "basic_search": True,        # Search case laws / bare acts
        "basic_opinion": True,        # Get legal opinion (up to daily limit)
        "chat_history": True,         # Save/load chat history
        "internet_search": True,      # Tiered internet fallback
        "advanced_research": False,   # Deep multi-gap research
        "document_drafting": False,   # Petition / pleading drafting
        "priority_response": False,   # Faster processing queue
    },
    TIER_PREMIUM: {
        "basic_search": True,
        "basic_opinion": True,
        "chat_history": True,
        "internet_search": True,
        "advanced_research": True,
        "document_drafting": True,
        "priority_response": True,
    },
}

# High Court domain mapping by state (for jurisdiction-aware search)
HC_DOMAIN_BY_STATE = {
    "telangana": "tshc.gov.in"  
}
