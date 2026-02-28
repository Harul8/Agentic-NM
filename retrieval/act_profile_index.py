"""
Act Profile Index — lightweight BM25 act-level index for fast act identification.

Problem it solves
-----------------
With 100+ acts indexed, each hybrid_search() call sends ~90 candidates to the
cross-encoder (slow: ~2s on CPU).  Most candidates come from irrelevant acts
(POCSO, Arms Act, Constitution…) that happen to share adjacent vocabulary.

Solution: identify the 3-5 relevant acts FIRST using a cheap BM25 search over
per-act "profile documents", then tell hybrid_search to only cross-encode
candidates from those acts.  Cross-encoder candidate pool drops from ~90 → ~20-30
per query → ~3-4× speedup.

Profile document design (per act)
----------------------------------
Each profile is a single richly-concatenated text doc:

    {act_name} [{alias}]
    Domain: {comma-separated high-level domain tags derived from section titles}
    Sections: {pipe-separated section titles — all unique ones, up to 60}
    Samples:  {key snippet from every Nth section, sampled evenly across the act}

The "Sections" line is the critical vocabulary source — it surfaces plain-language
legal terms (e.g. "Voluntarily causing grievous hurt", "Criminal trespass") that
bridge the gap between a query like "knee fracture metal rod" and the act's formal
section titles.  The "Samples" line adds first-paragraph snippets for deeper
vocabulary coverage (punishment phrases, subject-matter words, definitions).

Staleness detection
-------------------
The profile files store a ``source_mtime`` equal to os.path.getmtime(BARE_CHUNKS_V2)
at build time.  On load, if BARE_CHUNKS_V2 is newer than the stored mtime, the
profiles are silently rebuilt and re-saved.

Public API
----------
    from retrieval.act_profile_index import identify_relevant_acts

    acts = identify_relevant_acts("metal rod fracture knee assault")
    # → frozenset({"The Bharatiya Nyaya Sanhita, 2023",
    #              "The Bharatiya Nagarik Suraksha Sanhita, 2023"})

    # Returns frozenset() (empty) if the profile index could not be loaded/built.
    # In that case callers should proceed without act-level filtering.
"""

import json
import logging
import math
import os
import re

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inline BM25 (self-contained — avoids importing hybrid_retriever which
# pulls in faiss at module level, making act_profile_index faiss-free)
# ---------------------------------------------------------------------------

class _BM25:
    """Okapi BM25 implementation (same as hybrid_retriever.BM25)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_count = 0
        self.avg_dl = 0.0
        self.doc_lengths: list = []
        self.doc_freqs: dict = {}
        self.term_freqs: list = []
        self.idf_cache: dict = {}

    @staticmethod
    def _tokenize(text: str) -> list:
        words = re.findall(r"[a-z0-9]+", text.lower())
        # Add bigrams — captures legal phrases like "grievous_hurt", "criminal_trespass",
        # "specific_performance", "wrongful_possession" which are far more discriminative
        # than their individual words across 100+ acts.
        bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
        return words + bigrams

    def fit(self, documents: list):
        self.doc_count = len(documents)
        self.doc_lengths, self.term_freqs, self.doc_freqs = [], [], {}
        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            tf: dict = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs.append(tf)
            for t in set(tokens):
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1
        self.avg_dl = sum(self.doc_lengths) / max(self.doc_count, 1)
        self.idf_cache = {
            t: math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)
            for t, df in self.doc_freqs.items()
        }

    def score(self, query: str, top_k: int = 50) -> list:
        tokens = self._tokenize(query)
        scores = []
        for i in range(self.doc_count):
            s = 0.0
            dl = self.doc_lengths[i]
            tf_d = self.term_freqs[i]
            for t in tokens:
                if t not in self.idf_cache:
                    continue
                tf = tf_d.get(t, 0)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avg_dl, 1))
                s += self.idf_cache[t] * (num / max(den, 0.001))
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def to_dict(self) -> dict:
        return {
            "k1": self.k1, "b": self.b,
            "doc_count": self.doc_count, "avg_dl": self.avg_dl,
            "doc_lengths": self.doc_lengths,
            "doc_freqs": self.doc_freqs,
            "term_freqs": self.term_freqs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "_BM25":
        obj = cls(k1=d.get("k1", 1.5), b=d.get("b", 0.75))
        obj.doc_count = d["doc_count"]
        obj.avg_dl = d["avg_dl"]
        obj.doc_lengths = d["doc_lengths"]
        obj.doc_freqs = d["doc_freqs"]
        obj.term_freqs = d["term_freqs"]
        obj.idf_cache = {
            t: math.log((obj.doc_count - df + 0.5) / (df + 0.5) + 1.0)
            for t, df in obj.doc_freqs.items()
        }
        return obj


def _save_bm25(bm25: _BM25, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bm25.to_dict(), f)


def _load_bm25(path: str) -> "_BM25 | None":
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return _BM25.from_dict(json.load(f))
    except Exception as e:
        logger.warning("_load_bm25 failed: %s", e)
        return None

# ---------------------------------------------------------------------------
# Act short-name aliases (mirrors smart_chunker._ACT_ALIASES)
# ---------------------------------------------------------------------------

_ACT_ALIASES: dict[str, str] = {
    "bharatiya nyaya sanhita":               "BNS",
    "bharatiya nyaya sanhita 2023":          "BNS",
    "bharatiya nagarik suraksha sanhita":    "BNSS",
    "bharatiya nagarik suraksha sanhita 2023": "BNSS",
    "bharatiya sakshya adhiniyam":           "BSA",
    "bharatiya sakshya adhiniyam 2023":      "BSA",
    "indian penal code":                     "IPC",
    "code of criminal procedure":            "CrPC",
    "code of criminal procedure 1973":       "CrPC",
    "indian evidence act":                   "IEA",
    "indian evidence act 1872":              "IEA",
    "code of civil procedure":               "CPC",
    "code of civil procedure 1908":          "CPC",
    "transfer of property act":              "TP Act",
    "transfer of property act 1882":         "TP Act",
    "specific relief act":                   "Specific Relief Act",
    "specific relief act 1963":              "Specific Relief Act",
    "negotiable instruments act":            "NI Act",
    "negotiable instruments act 1881":       "NI Act",
    "hindu marriage act":                    "HMA",
    "hindu marriage act 1955":               "HMA",
    "hindu succession act":                  "HSA",
    "hindu succession act 1956":             "HSA",
    "limitation act":                        "Limitation Act",
    "limitation act 1963":                   "Limitation Act",
    "registration act":                      "Registration Act",
    "registration act 1908":                 "Registration Act",
    "protection of women from domestic violence act": "DV Act",
    "dowry prohibition act":                 "Dowry Act",
    "motor vehicles act":                    "MV Act",
    "motor vehicles act 1988":               "MV Act",
    "consumer protection act":               "Consumer Protection Act",
    "consumer protection act 2019":          "Consumer Protection Act",
    "companies act":                         "Companies Act",
    "companies act 2013":                    "Companies Act",
    "arbitration and conciliation act":      "Arbitration Act",
    "arbitration and conciliation act 1996": "Arbitration Act",
    "indian contract act":                   "Contract Act",
    "indian contract act 1872":              "Contract Act",
    "prevention of corruption act":          "PCA",
    "information technology act":            "IT Act",
    "information technology act 2000":       "IT Act",
}


def _get_alias(act_name: str) -> str:
    name_lower = act_name.lower().strip()
    for key, alias in _ACT_ALIASES.items():
        if key in name_lower:
            return alias
    return ""


# ---------------------------------------------------------------------------
# Profile builder
# ---------------------------------------------------------------------------

def _build_act_profiles(chunks: dict) -> dict[str, str]:
    """
    Build one rich text profile per act from all indexed section chunks.

    Returns dict: act_name → profile_text.

    Profile text layout (v2 — discriminative-first design):

        {act_name} [{alias}]   ← repeated 3× for high BM25 weight
        {act_name} [{alias}]
        {act_name} [{alias}]
        Chapters: {chapter heading 1} | {heading 2} | …   ← repeated 2×
        Chapters: {chapter heading 1} | {heading 2} | …
        Domain: {distinctive words from section titles}
        Sections: {unique section title 1} | {title 2} | … (up to 60)
        Key provisions: {first sentence of selected sections, scope/definition focused}

    Design rationale
    ----------------
    v1 problem: generic legal boilerplate ("the court may", "shall be punished")
    dominated profile text → BM25 could not distinguish acts.

    v2 fixes:
    • Act title 3× → highest BM25 weight; catches direct act name mentions in query
    • Chapter headings 2× → e.g. "OF OFFENCES AGAINST HUMAN BODY" is extremely
      specific to BNS; "SPECIFIC PERFORMANCE OF CONTRACTS" to Specific Relief Act
    • Snippets capped to first-sentence only → first sentences state scope/subject
      ("This section applies to…", "A person commits rape if…") rather than
      punishment boilerplate shared across acts
    • Act name corruption filter: skip acts with name < 15 chars (artifact chunks)
    """
    from collections import defaultdict

    # --- Filter corrupted / truncated act names ---
    # e.g. "S Property Act, 1874" (20 chars) passes; "XYZ" (3 chars) is dropped.
    # Threshold of 15 chars excludes obvious truncation artifacts while keeping
    # all real short act names (shortest real Indian acts are ~18 chars).
    _MIN_ACT_NAME_LEN = 15

    # Group chunks by act name
    acts: dict[str, list] = defaultdict(list)
    for chunk in chunks.values():
        act_name = (chunk.get("act_name") or "").strip()
        if act_name and len(act_name) >= _MIN_ACT_NAME_LEN:
            acts[act_name].append(chunk)

    profiles: dict[str, str] = {}

    _STOPWORDS = {
        "of", "the", "and", "in", "for", "to", "a", "an", "by", "or",
        "with", "on", "at", "be", "is", "are", "was", "were", "shall",
        "may", "act", "section", "sub", "clause", "schedule", "provided",
        "any", "every", "as", "no", "not", "if", "its", "when", "where",
        "such", "this", "that", "have", "has", "from", "into", "under",
    }

    for act_name, act_chunks in acts.items():
        # Sort by section number (string sort is OK for vocabulary purposes)
        sorted_chunks = sorted(
            act_chunks,
            key=lambda c: str(c.get("section_number") or "0").zfill(5),
        )

        # --- Collect section titles + chapter headings ---
        titles: list[str] = []
        seen_titles: set[str] = set()
        chapter_titles: list[str] = []
        seen_chapters: set[str] = set()

        for chunk in sorted_chunks:
            # Explicit chapter_title field (set by smart_chunker if present)
            ch = (chunk.get("chapter_title") or "").strip()
            if ch and len(ch) > 5 and ch.lower() not in seen_chapters:
                seen_chapters.add(ch.lower())
                chapter_titles.append(ch)

            t = (chunk.get("section_title") or "").strip()
            if t and t.lower() not in seen_titles:
                seen_titles.add(t.lower())
                titles.append(t)
                # Derive chapter-like headings from section titles that look like
                # chapter markers: ALL CAPS, or start with Chapter/Part/Of
                if not ch and re.match(
                    r"^(Chapter|CHAPTER|Part |PART |Of |OF )", t
                ):
                    if t.lower() not in seen_chapters:
                        seen_chapters.add(t.lower())
                        chapter_titles.append(t)

        # --- Build domain tags from section titles ---
        domain_words: list[str] = []
        seen_domain: set[str] = set()
        for t in titles[:40]:
            for word in re.findall(r"[A-Za-z]{4,}", t):
                w = word.lower()
                if w not in _STOPWORDS and w not in seen_domain:
                    seen_domain.add(w)
                    domain_words.append(word)
                if len(domain_words) >= 60:
                    break
            if len(domain_words) >= 60:
                break

        # --- Sample key first-sentences (scope/definition focused) ---
        # Take up to 15 evenly-spaced chunks; extract ONLY first sentence of body.
        # First sentences state subject matter ("A person is said to commit murder…")
        # not boilerplate punishment text shared across every act.
        n = len(sorted_chunks)
        step = max(1, n // 15)
        sampled = sorted_chunks[::step][:15]

        snippets: list[str] = []
        for chunk in sampled:
            raw = (
                chunk.get("search_text")
                or chunk.get("full_text")
                or chunk.get("text")
                or ""
            ).strip()
            if not raw:
                continue
            # smart_chunker search_text: "Act Section N — Title\n\nBody text…"
            # Skip heading line; take first sentence of body only.
            lines = [l.strip() for l in raw.split("\n") if l.strip()]
            body = lines[1] if len(lines) > 1 else lines[0] if lines else ""
            # Split on sentence boundary; take first sentence up to 120 chars
            first_sent = re.split(r"(?<=[.!?])\s+", body)[0][:120]
            if first_sent and len(first_sent) > 30:
                snippets.append(first_sent)

        # --- Assemble profile ---
        alias = _get_alias(act_name)
        alias_str = f" [{alias}]" if alias else ""
        act_name_line = f"{act_name}{alias_str}"

        profile_parts = [
            # Act title repeated 3× → highest BM25 term weight; catches queries
            # that mention the act name directly ("Specific Relief Act eviction")
            act_name_line,
            act_name_line,
            act_name_line,
        ]

        # Chapter headings repeated 2× — strongest per-act discriminators
        # ("OF OFFENCES AGAINST HUMAN BODY" only exists in BNS/IPC)
        if chapter_titles:
            chapters_line = " | ".join(chapter_titles[:30])
            profile_parts.append(f"Chapters: {chapters_line}")
            profile_parts.append(f"Chapters: {chapters_line}")  # repeat 2×

        if domain_words:
            profile_parts.append(f"Domain: {', '.join(domain_words[:50])}")

        if titles:
            profile_parts.append(f"Sections: {' | '.join(titles[:60])}")

        if snippets:
            profile_parts.append(f"Key provisions: {' ... '.join(snippets)}")

        profiles[act_name] = "\n".join(profile_parts)

    logger.info("Built act profiles for %d acts", len(profiles))
    return profiles


# ---------------------------------------------------------------------------
# Act Profile Index class
# ---------------------------------------------------------------------------

class ActProfileIndex:
    """
    Lightweight BM25 index over per-act profile documents.

    Usage:
        idx = ActProfileIndex.load_or_build()
        acts = idx.identify_relevant_acts("metal rod fracture knee assault", top_k=5)
    """

    def __init__(
        self,
        profiles: dict[str, str],
        bm25_data: dict,
        source_mtime: float,
    ):
        self._profiles = profiles                       # act_name → profile_text
        self._act_names: list[str] = list(profiles.keys())  # ordered list (index ↔ BM25 position)
        self._source_mtime = source_mtime
        # Deserialize BM25 (uses inline _BM25 — no faiss dependency)
        self._bm25 = _BM25.from_dict(bm25_data)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def build_from_chunks(cls, chunks: dict, source_mtime: float) -> "ActProfileIndex":
        """Build the index from a loaded chunks dict."""
        profiles = _build_act_profiles(chunks)
        act_names = list(profiles.keys())
        docs = [profiles[n] for n in act_names]
        bm25 = _BM25()
        bm25.fit(docs)
        return cls(profiles=profiles, bm25_data=bm25.to_dict(), source_mtime=source_mtime)

    def save(self, meta_path: str, bm25_path: str):
        """Persist profiles + BM25 index to disk."""
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        meta = {
            "source_mtime": self._source_mtime,
            "act_names": self._act_names,
            "profiles": self._profiles,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        _save_bm25(self._bm25, bm25_path)
        logger.info("Act profile index saved: %d acts → %s", len(self._profiles), meta_path)

    @classmethod
    def load(cls, meta_path: str, bm25_path: str) -> "ActProfileIndex | None":
        """Load from disk. Returns None if files are missing or corrupt."""
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            bm25_obj = _load_bm25(bm25_path)
            if bm25_obj is None:
                return None
            profiles = meta.get("profiles") or {}
            if not profiles:
                return None
            obj = cls.__new__(cls)
            obj._profiles = profiles
            obj._act_names = meta.get("act_names") or list(profiles.keys())
            obj._source_mtime = meta.get("source_mtime", 0.0)
            obj._bm25 = bm25_obj
            return obj
        except Exception as e:
            logger.warning("ActProfileIndex.load failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def identify_relevant_acts(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.15,
    ) -> frozenset:
        """
        Return a frozenset of act names that are likely relevant to ``query``.

        Parameters
        ----------
        query : str
            Plain-language or legal-term query (same format as section-level queries).
        top_k : int
            Maximum number of acts to return (default 3 — tight; we commit to
            fewer acts so the pre-filter is actually discriminative.  The safety
            net in hybrid_search() skips filtering if < 5 candidates remain, so
            a false-exclusion at this stage is harmless).
        min_score : float
            Minimum BM25 score to include an act (default 0.15 — requires real
            vocabulary overlap, not just one matching stopword).  With bigrams in
            the tokenizer, genuine matches score much higher than noise matches,
            making this threshold safe to raise from the old 0.05.

        Returns
        -------
        frozenset of act names, or frozenset() if nothing scores above min_score.
        """
        if not self._bm25 or not self._act_names:
            return frozenset()
        try:
            results = self._bm25.score(query, top_k=top_k * 3)  # over-fetch then threshold
        except Exception as e:
            logger.warning("ActProfileIndex BM25 score failed: %s", e)
            return frozenset()

        selected = [
            self._act_names[idx]
            for idx, score in results
            if score >= min_score and idx < len(self._act_names)
        ][:top_k]

        if selected:
            logger.debug(
                "Act profile match for '%s…' → %s",
                query[:60], selected,
            )
        return frozenset(selected)


# ---------------------------------------------------------------------------
# Module-level lazy singleton
# ---------------------------------------------------------------------------

_singleton: "ActProfileIndex | None" = None
_singleton_chunks_path: str = ""


def _load_or_build() -> "ActProfileIndex | None":
    """
    Load the profile index from disk; rebuild if stale or missing.
    Never raises — returns None on total failure.
    """
    global _singleton, _singleton_chunks_path

    try:
        from config import BARE_CHUNKS_V2, ACT_PROFILES_META, ACT_PROFILES_BM25
    except ImportError:
        logger.warning("ActProfileIndex: config paths not available")
        return None

    # Check whether the chunks file exists
    if not os.path.exists(BARE_CHUNKS_V2):
        logger.debug("ActProfileIndex: BARE_CHUNKS_V2 not found (%s)", BARE_CHUNKS_V2)
        return None

    chunks_mtime = os.path.getmtime(BARE_CHUNKS_V2)

    # --- Try to load from disk if up to date ---
    if os.path.exists(ACT_PROFILES_META) and os.path.exists(ACT_PROFILES_BM25):
        idx = ActProfileIndex.load(ACT_PROFILES_META, ACT_PROFILES_BM25)
        if idx is not None and idx._source_mtime >= chunks_mtime - 1.0:
            logger.debug(
                "ActProfileIndex: loaded %d profiles from disk (up to date)",
                len(idx._profiles),
            )
            return idx
        logger.info(
            "ActProfileIndex: profiles stale (chunks mtime=%.0f > profile mtime=%.0f). Rebuilding.",
            chunks_mtime, (idx._source_mtime if idx else 0),
        )

    # --- Build from scratch ---
    logger.info("ActProfileIndex: building from %s …", BARE_CHUNKS_V2)
    try:
        with open(BARE_CHUNKS_V2, encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception as e:
        logger.error("ActProfileIndex: failed to load chunks: %s", e)
        return None

    try:
        idx = ActProfileIndex.build_from_chunks(chunks, source_mtime=chunks_mtime)
        idx.save(ACT_PROFILES_META, ACT_PROFILES_BM25)
        return idx
    except Exception as e:
        logger.error("ActProfileIndex: build failed: %s", e)
        return None


def get_act_index() -> "ActProfileIndex | None":
    """
    Return the module-level singleton ActProfileIndex, building it lazily.
    Thread-safe enough for the typical single-process server use case.
    """
    global _singleton
    if _singleton is None:
        _singleton = _load_or_build()
    return _singleton


def identify_relevant_acts(
    query: str,
    top_k: int = 3,
    min_score: float = 0.15,
) -> frozenset:
    """
    Top-level convenience function.

    Returns frozenset of relevant act names, or frozenset() if the index is
    unavailable.  An empty frozenset should be treated as "no filter" by callers.
    """
    idx = get_act_index()
    if idx is None:
        return frozenset()
    return idx.identify_relevant_acts(query, top_k=top_k, min_score=min_score)


def build_and_save_act_profiles() -> bool:
    """
    Explicitly (re)build and save the act profile index.
    Called from rebuild_vector_store.py after the section index is built.
    Returns True on success.
    """
    global _singleton
    _singleton = None  # force rebuild
    idx = _load_or_build()
    if idx is not None:
        _singleton = idx
        return True
    return False
