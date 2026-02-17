"""
Central data path config. Set NYAYMALAW_DATA_ROOT to store all app data
(chat history, login DB, vector store, BareActs, CaseLaws) in one folder,
e.g. a Google Drive folder: G:\\My Drive\\Nyaymalaw
"""
import os

# Optional: load .env so NYAYMALAW_DATA_ROOT can be set there (requires python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.normpath(__file__)))
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

# Web references table (articles/news that aren't primary sources)
WEB_REFERENCES_DB = os.path.join(DATA_ROOT, "web_references.json")

# ---------------------------------------------------------------------------
# Embedding & re-ranker model names
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# ---------------------------------------------------------------------------
# Domain whitelists for tiered internet search
# ---------------------------------------------------------------------------

# Tier 2: Official court & government websites (highest authority)
TIER2_OFFICIAL_COURT_DOMAINS = (
    "main.sci.gov.in",
    "sci.gov.in",
    "api.sci.gov.in",
    "indiacode.nic.in",
    "legislative.gov.in",
    "egazette.nic.in",
    "lawcommissionofindia.nic.in",
    # High Courts (mapped by state)
    "karnatakajudiciary.kar.nic.in",
    "bombayhighcourt.nic.in",
    "delhihighcourt.nic.in",
    "mhc.tn.gov.in",
    "allahabadhighcourt.in",
    "highcourtchd.gov.in",
    "phc.gov.in",
    "ghconline.gov.in",
    "hcraj.nic.in",
    "jharkhandhighcourt.nic.in",
    "orissahighcourt.nic.in",
    "cghc.nic.in",
    "hcmadgoa.nic.in",
    "hckashmir.nic.in",
    "meghalayahighcourt.nic.in",
    "thc.nic.in",
    "hcsikkim.gov.in",
    "services.ecourts.gov.in",
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
    "telangana": "ghconline.gov.in",
    "rajasthan": "hcraj.nic.in",
    "jharkhand": "jharkhandhighcourt.nic.in",
    "odisha": "orissahighcourt.nic.in",
    "chhattisgarh": "cghc.nic.in",
    "jammu and kashmir": "hckashmir.nic.in",
    "meghalaya": "meghalayahighcourt.nic.in",
    "tripura": "thc.nic.in",
    "sikkim": "hcsikkim.gov.in",
}
