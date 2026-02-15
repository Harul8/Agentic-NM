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

# Vector store files
BARE_INDEX = os.path.join(VECTOR_STORE, "bareacts.index")
BARE_CHUNKS = os.path.join(VECTOR_STORE, "bareacts_chunks.json")
CASE_INDEX = os.path.join(VECTOR_STORE, "caselaws.index")
CASE_CHUNKS = os.path.join(VECTOR_STORE, "caselaws_chunks.json")
