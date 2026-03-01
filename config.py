"""
Central data path config. Set NYAYMALAW_DATA_ROOT to store all app data
(chat history, login DB, vector store, BareActs, CaseLaws) in one folder,
e.g. a Google Drive folder: G:\\My Drive\\Nyaymalaw
"""
import os

# Optional: load .env so NYAYMALAW_DATA_ROOT can be set there (requires python-dotenv)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.normpath(__file__)))
try:
    from dotenv import load_dotenv
    # Load .env from project root so it works regardless of process cwd
    _env_path = os.path.join(_PROJECT_ROOT, ".env")
    load_dotenv(_env_path)
except ImportError:
    pass
except Exception:
    pass
_default_data = os.path.join(_PROJECT_ROOT, "data")

# Use env var for data root (e.g. Google Drive path). If unset, use project/data.
_DATA_ROOT = os.environ.get("NYAYMALAW_DATA_ROOT", "").strip()
DATA_ROOT = os.path.normpath(_DATA_ROOT) if _DATA_ROOT else _default_data

# Derived paths (all under DATA_ROOT)
CHAT_HISTORY_DIR = os.path.join(DATA_ROOT, "chat_history")
DB_PATH = os.path.join(CHAT_HISTORY_DIR, "app.db")
VECTOR_STORE = os.path.join(DATA_ROOT, "vector_store")
BARE_ACTS_DIR = os.path.join(DATA_ROOT, "BareActs")
CASELAW_DIR = os.path.join(DATA_ROOT, "CaseLaws")

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

# BM25 index files (for hybrid search)
BARE_BM25_INDEX = os.path.join(VECTOR_STORE, "bareacts_bm25.json")
CASE_BM25_INDEX = os.path.join(VECTOR_STORE, "caselaws_bm25.json")

# Act-level profile index — one rich doc per act, used for act-first identification
# before section-level hybrid search.  Built from BARE_CHUNKS_V2 at index-rebuild time
# and lazy-rebuilt on first use if the files are missing / stale.
ACT_PROFILES_META  = os.path.join(VECTOR_STORE, "act_profiles_meta.json")   # {act_name: profile_text}
ACT_PROFILES_BM25  = os.path.join(VECTOR_STORE, "act_profiles_bm25.json")   # BM25 index over profiles

# Web references table (articles/news that aren't primary sources)
WEB_REFERENCES_DB = os.path.join(DATA_ROOT, "web_references.json")

# Pending indexing candidates (survives refresh; removed on Index or Discard)
PENDING_INDEXING_PATH = os.path.join(DATA_ROOT, "pending_indexing.json")

# Case law discovery: documents presented for indexing (persist until user Index or Clear)
CASE_LAW_DISCOVERY_PENDING_PATH = os.path.join(DATA_ROOT, "case_law_discovery_pending.json")
CASE_LAW_DISCOVERY_SUMMARY_INDEX_PATH = os.path.join(DATA_ROOT, "case_law_discovery_summary_index.json")
# Bare act summary index (act_name -> summary text); built on vector store rebuild and by case law discovery
BARE_ACT_SUMMARY_INDEX_PATH = os.path.join(DATA_ROOT, "bare_act_summary_index.json")
# Case law summary index (signature -> summary text)
CASE_LAW_SUMMARY_INDEX_PATH = os.path.join(DATA_ROOT, "case_law_summary_index.json")

# Indian Kanoon API (https://api.indiankanoon.org). Set INDIAN_KANOON_API_TOKEN in .env.
INDIAN_KANOON_API_TOKEN = os.environ.get("INDIAN_KANOON_API_TOKEN", "").strip()

# ---------------------------------------------------------------------------
# Embedding & re-ranker model names
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
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
# Set to 0.0 to disable. Default 0.25 gives a measurable boost for section-matched docs.
LEGAL_TERM_BOOST_WEIGHT = float(os.environ.get("LEGAL_TERM_BOOST_WEIGHT", "0.25"))

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
    "karnataka": "karnatakajudiciary.kar.nic.in",
    "maharashtra": "bombayhighcourt.nic.in",
    "goa": "hcmadgoa.nic.in",
    "delhi": "delhihighcourt.nic.in",
    "tamil nadu": "mhc.tn.gov.in",
    "uttar pradesh": "allahabadhighcourt.in",
    "punjab": "highcourtchd.gov.in",
    "haryana": "highcourtchd.gov.in",
    "chandigarh": "highcourtchd.gov.in",
    "andhra pradesh": "phc.gov.in",
    "telangana": "tshc.gov.in",          # Telangana State High Court (ghconline.gov.in is defunct)
    "rajasthan": "hcraj.nic.in",
    "jharkhand": "jharkhandhighcourt.nic.in",
    "odisha": "orissahighcourt.nic.in",
    "chhattisgarh": "cghc.nic.in",
    "jammu and kashmir": "hckashmir.nic.in",
    "meghalaya": "meghalayahighcourt.nic.in",
    "tripura": "thc.nic.in",
    "sikkim": "hcsikkim.gov.in",
}
