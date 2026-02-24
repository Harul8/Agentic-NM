"""
Only hardcoded limits for case law discovery. Everything else is dynamic (LLM-driven).
"""

# Similarity score thresholds (same scale as cross-encoder / scoring elsewhere)
SIMILARITY_THRESHOLD_LOW = 3.0   # Minimum; drop documents below this
SIMILARITY_THRESHOLD_HIGH = 5.0  # Preferred; keep all above this first

# Statement-based flow: if fewer than this many with score > HIGH, take top N above LOW
TOP_N_STATEMENT = 10

# Bare-act flow: best N per chunk (above thresholds); per act, up to this many
BEST_PER_CHUNK = 2
TOP_N_PER_ACT = 25
# Max results to fetch from Indian Kanoon per act (first_10_acts flow)
MAX_FETCH_PER_ACT = 25

# Act-named flow: user names an act → resolve from bare act summary index, search IK, dedup, target unique count
TOP_N_ACT_NAMED = 25
# Fetch extra from Indian Kanoon so that after dedup we can fill TOP_N_ACT_NAMED
ACT_NAMED_FETCH_BUFFER = 50

# Summary character caps (for LLM-generated act/case summaries)
ACT_SUMMARY_MAX_CHARS = 5000
CASE_LAW_SUMMARY_MAX_CHARS = 5000
