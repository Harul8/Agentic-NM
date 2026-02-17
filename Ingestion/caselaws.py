# DEPRECATED — Use retrieval.hybrid_retriever and retrieval.tiered_search instead.
# This file kept only so old imports don't crash.
from retrieval.hybrid_retriever import search_case_laws_auto as search_local_caselaws  # noqa: F401
from retrieval.tiered_search import tiered_search as search_internet_caselaws  # noqa: F401
