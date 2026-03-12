import { memo, useState, useRef, useEffect, useMemo, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import jsPDF from "jspdf"; // npm install jspdf
import * as XLSX from "xlsx";
import "./App.css";

const AUTH_TOKEN_KEY = "nyaymalaw_auth_token";
const CURRENT_USER_KEY = "nyaymalaw_current_user";
const CURRENT_USER_NAME_KEY = "nyaymalaw_current_user_name";

/** Splits opinion text into normal segments and quote blocks (content inside ┌─┐ │ ... │ └─┘). Returns [{ type: 'normal'|'quote', text }]. */
function parseOpinionWithQuotes(opinion) {
  if (!opinion || typeof opinion !== "string") return [{ type: "normal", text: "" }];
  const segments = [];
  const lines = opinion.split("\n");
  let i = 0;
  const normalBuf = [];
  const flushNormal = () => {
    if (normalBuf.length) {
      segments.push({ type: "normal", text: normalBuf.join("\n") });
      normalBuf.length = 0;
    }
  };
  while (i < lines.length) {
    const line = lines[i];
    if (/^┌─+┐\s*$/.test(line)) {
      flushNormal();
      const start = i;
      i += 1;
      const quoteLines = [];
      while (i < lines.length && /^│\s*(.*)$/.test(lines[i])) {
        const m = lines[i].match(/^│\s*(.*)$/);
        quoteLines.push((m[1] || "").trimEnd());
        i += 1;
      }
      if (i < lines.length && /^└─+┘\s*$/.test(lines[i])) {
        i += 1;
        segments.push({ type: "quote", text: quoteLines.join("\n") });
      } else {
        for (let j = start; j < i; j++) normalBuf.push(lines[j]);
        if (i < lines.length) normalBuf.push(lines[i]);
        i += 1;
      }
      continue;
    }
    normalBuf.push(line);
    i += 1;
  }
  flushNormal();
  return segments;
}

/** Build maps: bareActKey -> url, signatureOrCaseKey -> url, for strict citation linking. */
function buildCitationMaps(bareActs, caseLaws) {
  const bareMap = new Map();
  const caseMap = new Map();
  const normalize = (s) => (s || "").toLowerCase().replace(/\s+/g, " ").trim();

  function addBare(ba) {
    const act = (ba.act_name || "").trim();
    const sec = String(ba.section_number ?? "").trim();
    const url = ba.url || ba.source_url || "";
    if (!url) return;
    const key = `${normalize(act)}§${sec}`;
    if (!bareMap.has(key)) bareMap.set(key, url);
    if (act && sec) {
      const key2 = `${normalize(act)}, § ${sec}`;
      if (!bareMap.has(key2)) bareMap.set(key2, url);
    }
  }

  function addCase(cl) {
    const url = cl.url || cl.source_url || "";
    if (!url) return;
    const sig = (cl.signature || "").trim().toLowerCase();
    if (sig) caseMap.set(sig, url);
    const title = (cl.case_name || cl.title || "").trim();
    if (title) caseMap.set(normalize(title), url);
  }

  (bareActs || []).forEach(addBare);
  (caseLaws || []).forEach(addCase);
  (bareActs || []).forEach((ba) => (ba.related_case_laws || []).forEach(addCase));
  return { bareMap, caseMap };
}

/** Split segment text into parts: plain strings and citation spans. Citations are turned into links when URL is found. */
function linkifyOpinionSegment(text, bareMap, caseMap) {
  if (!text || !bareMap || !caseMap) return [text];
  const parts = [];
  // Match [...] that may be bare act (contains §) or case law (long hex)
  const bracketRe = /\[([^\]]+)\]/g;
  let lastEnd = 0;
  let m;
  while ((m = bracketRe.exec(text)) !== null) {
    const full = m[0];
    const inner = m[1].trim();
    if (lastEnd < m.index) parts.push(text.slice(lastEnd, m.index));

    const isBare = /§/.test(inner);
    const isCaseSig = /^[a-f0-9]{32,}$/i.test(inner);
    let url = null;
    if (isBare) {
      const norm = inner.toLowerCase().replace(/\s+/g, " ").trim();
      url = bareMap.get(norm) ?? bareMap.get(norm.replace(/\s*§\s*/, "§"));
      if (!url && inner.includes("§")) {
        const secMatch = inner.match(/§\s*(\d+[A-Za-z]*)/);
        const actPart = inner.replace(/\s*§\s*\d+[A-Za-z]*\s*[—\-–].*$/, "").replace(/^the\s+/i, "").trim();
        const actNorm = actPart.replace(/\s*,\s*(\d{4})\s*$/, " $1").toLowerCase().replace(/\s+/g, " ").trim();
        if (secMatch && actNorm) {
          url = bareMap.get(`${actNorm}§${secMatch[1]}`);
          if (!url) url = bareMap.get(actNorm + "§" + secMatch[1]);
        }
      }
    }
    if (isCaseSig) url = url || caseMap.get(inner.toLowerCase());
    if (!url && !isCaseSig) url = caseMap.get(inner.toLowerCase());

    if (url) {
      parts.push(
        <a key={`${m.index}-${inner.slice(0, 20)}`} href={url} target="_blank" rel="noopener noreferrer" className="opinion-citation-link" title="View source">
          {full}
        </a>
      );
    } else {
      parts.push(full);
    }
    lastEnd = m.index + full.length;
  }
  if (lastEnd < text.length) parts.push(text.slice(lastEnd));
  return parts.length ? parts : [text];
}

const ChatComposer = memo(function ChatComposer({
  loading,
  placeholder,
  onSubmit,
  resetSignal,
  showDisclaimer,
}) {
  const [draft, setDraft] = useState("");
  const textareaRef = useRef(null);

  useEffect(() => {
    setDraft("");
  }, [resetSignal]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "24px";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [draft]);

  useEffect(() => {
    if (!loading && textareaRef.current) textareaRef.current.focus();
  }, [loading]);

  const submitDraft = useCallback(() => {
    const raw = draft ?? "";
    if (!raw.trim() || loading) return;
    onSubmit(raw);
    setDraft("");
  }, [draft, loading, onSubmit]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitDraft();
    }
  }, [submitDraft]);

  return (
    <>
      <div className="chat-input-container">
        <textarea
          ref={textareaRef}
          autoFocus
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className="chat-input"
          rows={1}
          disabled={loading}
        />
        <button
          type="button"
          onClick={submitDraft}
          disabled={loading || !draft.trim()}
          className="chat-send"
          aria-label="Send message"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 19V5M5 12l7-7 7 7" />
          </svg>
        </button>
      </div>
      {showDisclaimer && (
        <p className="chat-disclaimer">Nyaymalaw AI can make mistakes. Consider checking important information.</p>
      )}
    </>
  );
});

// ---------------------------------------------------
// MAIN APP
// ---------------------------------------------------
function App() {
  // -------------------------
  // Auth disabled: open app without login. User shown as "Guest".
  // -------------------------
  const [currentUser, setCurrentUser] = useState("Guest");
  const [currentUserName, setCurrentUserName] = useState("Guest");

  const handleLogout = () => {
    setMessages([]);
    setSavedChats([]);
    handleStartNewCase();
  };

  // -------------------------
  // Chat & Interview state (merged from existing and snippet)
  // -------------------------
  const [messages, setMessages] = useState([]);
  const [phase, setPhase] = useState("fact_collection"); // Retain existing phase logic
  const [factsSummary, setFactsSummary] = useState(null); // Retain existing
  const [pendingMaterials, setPendingMaterials] = useState(null); // Retain existing
  const [loading, setLoading] = useState(false); // Retain existing
  const [error, setError] = useState(""); // Retain existing
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const [composerResetSignal, setComposerResetSignal] = useState(0);

  // New interview state from snippet
  const [stage, setStage] = useState("await_facts"); // "await_facts" | "interview" | "bare_acts_review" | "done"
  const [facts, setFacts] = useState("");
  const [currentQuestion, setCurrentQuestion] = useState("");
  const [qaHistory, setQaHistory] = useState([]); // [{question, answer}]
  const [pendingBareActs, setPendingBareActs] = useState([]); // bare acts from Phase A, sent back in Phase B
  const [pendingFactsSummary, setPendingFactsSummary] = useState(""); // facts_summary from Phase A
  const [opinionText, setOpinionText] = useState("");
  const [retrieved, setRetrieved] = useState([]);
  const [rawResponse, setRawResponse] = useState("");
  const [showDebug, setShowDebug] = useState(false);
  
  // Progress tracking state
  const [progress, setProgress] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [expandedGroups, setExpandedGroups] = useState({});

  // Streaming state: step timeline + live token text
  const [streamingSteps, setStreamingSteps] = useState([]); // [{message, icon, done}]
  const [streamingToken, setStreamingToken] = useState("");  // accumulated LLM tokens

  // Shared SSE handlers for "step" progress and incremental "token" output.
  const handleStep = useCallback((stepPayload) => {
    setStreamingSteps((prev) => {
      if (prev.length === 0) {
        return [{ message: stepPayload.message, icon: stepPayload.icon || "", done: false }];
      }
      const updated = prev.map((s, i) => (i === prev.length - 1 ? { ...s, done: true } : s));
      return [...updated, { message: stepPayload.message, icon: stepPayload.icon || "", done: false }];
    });
  }, []);

  const handleToken = useCallback((tokenPayload) => {
    setStreamingToken((prev) => prev + (tokenPayload.content || ""));
  }, []);

  // Manual mode selection: "legal_opinion" (default), "legal_research", "general"
  const [chatMode, setChatMode] = useState("legal_opinion");

  // Bottom pane: single accordion (Eval | Architecture | Updates Tracker). Default: minimal strip at bottom; can extend up to 75% of viewport.
  const [bottomExpandedSection, setBottomExpandedSection] = useState(null); // "eval" | "architecture" | "updates" | null
  const EVAL_PANE_MIN_HEIGHT = 6;  // very low strip (75% lower than 24px) so pane is "hidden" by default
  const EVAL_PANE_MAX_VH = 75;     // extend up to 75% of window height
  const [evalPaneHeight, setEvalPaneHeight] = useState(EVAL_PANE_MIN_HEIGHT);
  const handleEvalPaneResizeMouseDown = (e) => {
    e.preventDefault();
    const startY = e.clientY;
    const startHeight = evalPaneHeight;
    const onMove = (moveEvent) => {
      const deltaY = moveEvent.clientY - startY;
      // Drag cursor UP (negative deltaY) → increase pane height; DOWN → decrease (so movement follows cursor)
      let next = startHeight - deltaY;
      if (next < EVAL_PANE_MIN_HEIGHT) next = EVAL_PANE_MIN_HEIGHT;
      const maxPx = typeof window !== "undefined" ? window.innerHeight * (EVAL_PANE_MAX_VH / 100) : 600;
      if (next > maxPx) next = maxPx;
      setEvalPaneHeight(next);
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };
  const [evalFiles, setEvalFiles] = useState([]);
  const [evalFigures, setEvalFigures] = useState([]);
  const [evalLoaded, setEvalLoaded] = useState(null); // { path, data }
  const [evalLoading, setEvalLoading] = useState(false);
  const [architectureContent, setArchitectureContent] = useState(null);
  const [architectureLoading, setArchitectureLoading] = useState(false);
  const [updatesRows, setUpdatesRows] = useState([]);

  // Pending indexing (left pane, below Chat history) — from web enrichment; user selects and clicks Index
  const [pendingIndexingCandidates, setPendingIndexingCandidates] = useState([]);
  const [indexingRunning, setIndexingRunning] = useState(false);
  const [showClearPendingConfirm, setShowClearPendingConfirm] = useState(false);

  // Case law discovery: documents presented for indexing (persist until user Index or Clear; survives refresh/restart)
  const [caseLawDiscoveryPending, setCaseLawDiscoveryPending] = useState([]);
  const [caseLawDiscoveryRunning, setCaseLawDiscoveryRunning] = useState(false);
  const [showClearCaseLawDiscoveryConfirm, setShowClearCaseLawDiscoveryConfirm] = useState(false);

  // Saved chats (ChatGPT-style): list of past conversations, persisted to localStorage
  const [savedChats, setSavedChats] = useState([]);
  const hasSavedCurrentChatRef = useRef(false);
  const [editingChatId, setEditingChatId] = useState(null);
  const [editingTitle, setEditingTitle] = useState("");
  const [openMenuChatId, setOpenMenuChatId] = useState(null);
  const editInputRef = useRef(null);
  const editMessageInputRef = useRef(null);
  const currentChatIdRef = useRef(null);

  // Edit user message (current and old chats)
  const [editingMessageIndex, setEditingMessageIndex] = useState(null);
  const [editDraft, setEditDraft] = useState("");
  const [actionMenuOpenIndex, setActionMenuOpenIndex] = useState(null);
  const [copyJustDoneIndex, setCopyJustDoneIndex] = useState(null);

  // Left sidebar accordion: only one of Bare Acts / Case Laws / Chat history expanded at a time
  const [sidebarExpandedSection, setSidebarExpandedSection] = useState(null);

  // Left pane width (resizable: 50% smaller to 50% larger than base)
  const LEFT_COLUMN_BASE_WIDTH = 280;
  const LEFT_COLUMN_DEFAULT_WIDTH = LEFT_COLUMN_BASE_WIDTH * 0.75; // 25% narrower by default
  const LEFT_COLUMN_MIN_WIDTH = LEFT_COLUMN_BASE_WIDTH * 0.5;
  const LEFT_COLUMN_MAX_WIDTH = LEFT_COLUMN_BASE_WIDTH * 1.5;
  const [leftColumnWidth, setLeftColumnWidth] = useState(LEFT_COLUMN_DEFAULT_WIDTH);

  const handleSidebarResizeMouseDown = (e) => {
    e.preventDefault();
    const startX = e.clientX;
    const startWidth = leftColumnWidth;
    const onMove = (moveEvent) => {
      const delta = moveEvent.clientX - startX;
      let next = startWidth + delta;
      if (next < LEFT_COLUMN_MIN_WIDTH) next = LEFT_COLUMN_MIN_WIDTH;
      if (next > LEFT_COLUMN_MAX_WIDTH) next = LEFT_COLUMN_MAX_WIDTH;
      setLeftColumnWidth(next);
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  // Default list so the UI always shows bare acts (e.g. when API is not running yet)
  const DEFAULT_BARE_ACTS = [
    "250884_2_english_01042024.pdf",
    "A LAND REVENUE (ENHANCEMENT)_ACT_1967.pdf",
    "a1988-59.pdf",
    "ASSIGNED LANDS_Act_1977.pdf",
    "Bhu Bharti_Act_2025.pdf",
    "Dharani_Act_2020.pdf",
    "LAND ENCROACHMENT_Act_1905.pdf",
    "THE INDIAN CONTRACT ACT 1872.pdf",
    "THE INDIAN EASEMENTS ACT 1882.pdf",
    "THE LIMITATION ACT 1963.pdf",
    "THE REGISTRATION ACT 1908.pdf",
    "THE TELANGANA TENANCY AND AGRICULTURAL LANDS ACT 1950.pdf",
    "THE TRANSFER OF PROPERTY ACT 1882.pdf",
    "the_indian_stamp_act_1899.pdf",
    "WALTA_Act_2002.pdf",
  ];
  const [bareActs, setBareActs] = useState(DEFAULT_BARE_ACTS);
  // Grouped Bare Acts from legal_database/json_output/BareActs/<Jurisdiction>/...
  // Each item: { name: "Telangana" | "Union of India" | ..., acts: [baseName, ...] }
  const [bareActsLibrary, setBareActsLibrary] = useState(null);
  const [caseLawsList, setCaseLawsList] = useState([]);
  // Which Bare Acts jurisdiction group is expanded (e.g. "Telangana" or "Union of India")
  const [bareActsJurisdictionOpen, setBareActsJurisdictionOpen] = useState(null);
  // Grouped case laws by court, e.g. "Supreme Court", "Telangana HC"
  const [caseLawsLibrary, setCaseLawsLibrary] = useState(null);
  const [caseLawsCourtOpen, setCaseLawsCourtOpen] = useState(null);
  const [bareActsFilter, setBareActsFilter] = useState("");
  const [caseLawsFilter, setCaseLawsFilter] = useState("");
  const [chatHistoryFilter, setChatHistoryFilter] = useState("");
  // In production (e.g. https://nyaymalaw.in) use same origin or VITE_API_BASE; locally use backend on :8000
  const API_BASE =
    import.meta.env.VITE_API_BASE ||
    (typeof window !== "undefined" &&
     (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")
      ? "http://127.0.0.1:8000"
      : (typeof window !== "undefined" ? window.location.origin : "http://127.0.0.1:8000"));

  // Load persisted pending indexing candidates on mount (survives refresh)
  useEffect(() => {
    const token = localStorage.getItem(AUTH_TOKEN_KEY);
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    fetch(`${API_BASE}/indexing/pending`, { headers })
      .then(async (res) => {
        if (!res.ok) return { items: [] };
        const data = await res.json().catch(() => ({}));
        return data;
      })
      .then((data) => {
        const items = Array.isArray(data?.items) ? data.items : [];
        setPendingIndexingCandidates(items.map((c, i) => ({
          id: `idx-${Date.now()}-${i}`,
          title: c.title || "",
          source_url: c.source_url || "",
          suggested_category: c.suggested_category || "case_law",
          category: c.suggested_category || "case_law",
          selected: !c.already_in_store,
          already_in_store: !!c.already_in_store,
        })));
      })
      .catch(() => {});
  }, [API_BASE]);

  // Load case law discovery pending on mount (persisted; survives refresh and backend restart)
  useEffect(() => {
    const token = localStorage.getItem(AUTH_TOKEN_KEY);
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    fetch(`${API_BASE}/case-law-discovery/pending`, { headers })
      .then(async (res) => {
        if (!res.ok) return { items: [] };
        const data = await res.json().catch(() => ({}));
        return data;
      })
      .then((data) => {
        const items = Array.isArray(data?.items) ? data.items : [];
        setCaseLawDiscoveryPending(items.map((c, i) => ({
          id: `cld-${Date.now()}-${i}`,
          title: c.title || "",
          source_url: c.source_url || "",
          suggested_category: c.suggested_category || "case_law",
          category: c.suggested_category || "case_law",
          selected: !c.already_in_store,
          already_in_store: !!c.already_in_store,
          signature: c.signature,
          act_name: c.act_name,
          summary: c.summary,
        })));
      })
      .catch(() => {});
  }, [API_BASE]);

  // Load saved chats once on mount (do not depend on API_BASE to avoid re-runs and 429 from backend).
  useEffect(() => {
    const token = localStorage.getItem(AUTH_TOKEN_KEY);
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    fetch(`${API_BASE}/chats`, { headers })
      .then(async (res) => {
        if (!res.ok) return { chats: [] }; // 429 or other error: don't parse body (may be HTML)
        const text = await res.text();
        try {
          return text ? JSON.parse(text) : { chats: [] };
        } catch {
          return { chats: [] };
        }
      })
      .then((data) => setSavedChats(Array.isArray(data.chats) ? data.chats : []))
      .catch(() => setSavedChats([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps -- intentional: run once on mount only
  }, []);

  // Persist current chat to backend when it changes (no auth: backend uses anonymous user)
  useEffect(() => {
    const chatId = currentChatIdRef.current;
    if (chatId == null || messages.length === 0) return;
    const token = localStorage.getItem(AUTH_TOKEN_KEY);
    const headers = { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) };
    const title = (messages.find((m) => m.role === "user")?.content || "").toString().slice(0, 50);
    fetch(`${API_BASE}/chats`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        id: chatId,
        title: title || "Untitled chat",
        messages,
        opinionText,
        retrieved,
        createdAt: new Date().toISOString(),
      }),
    }).catch(() => {});
  }, [messages, opinionText, retrieved, API_BASE]);

  // Keep the "current" chat in the list in sync with messages/opinion/retrieved
  useEffect(() => {
    if (currentChatIdRef.current == null || messages.length === 0) return;
    setSavedChats((prev) =>
      prev.map((c) =>
        c.id == currentChatIdRef.current
          ? { ...c, messages, opinionText, retrieved }
          : c
      )
    );
  }, [messages, opinionText, retrieved]);

  // Fetch Bare Acts once on mount (do not depend on API_BASE to avoid re-runs and 429).
  useEffect(() => {
    const fetchBareActs = async () => {
      try {
        // Prefer new grouped library endpoint that mirrors json_output/BareActs/<Jurisdiction>/...
        const libRes = await fetch(`${API_BASE}/bareacts/library`);
        if (libRes.ok) {
          const libData = await libRes.json().catch(() => ({}));
          const jurisdictions = Array.isArray(libData?.jurisdictions) ? libData.jurisdictions : [];
          const normalized = jurisdictions
            .map((j) => ({
              name: (j?.name || "").trim() || "Bare Acts",
              acts: Array.isArray(j?.acts) ? j.acts : [],
            }))
            .filter((j) => j.acts.length > 0);
          if (normalized.length > 0) {
            setBareActsLibrary(normalized);
            // Keep flat list for places that just need "all Bare Acts".
            setBareActs(normalized.flatMap((j) => j.acts));
            // Default to the first jurisdiction being expanded.
            if (!bareActsJurisdictionOpen && normalized[0]?.name) {
              setBareActsJurisdictionOpen(normalized[0].name);
            }
            return;
          }
        }

        // Fallback: older API that returns a flat list.
        const res = await fetch(`${API_BASE}/bareacts/list`);
        if (!res.ok) {
          setBareActs(DEFAULT_BARE_ACTS);
          return;
        }
        const text = await res.text();
        let data = {};
        try {
          data = text ? JSON.parse(text) : {};
        } catch {
          setBareActs(DEFAULT_BARE_ACTS);
          return;
        }
        const acts = data.acts || [];
        setBareActs(acts.length > 0 ? acts : DEFAULT_BARE_ACTS);
      } catch (err) {
        console.error("Error fetching bare acts:", err);
        setBareActs(DEFAULT_BARE_ACTS);
      }
    };
    fetchBareActs();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- intentional: run once on mount only
  }, []);

  // Fetch Case Laws list once on mount
  useEffect(() => {
    const fetchCaseLaws = async () => {
      try {
        // Prefer grouped library endpoint for courts (Supreme Court, Telangana HC, etc.).
        const libRes = await fetch(`${API_BASE}/caselaws/library`);
        if (libRes.ok) {
          const libData = await libRes.json().catch(() => ({}));
          const courts = Array.isArray(libData?.courts) ? libData.courts : [];
          const normalized = courts
            .map((c) => ({
              name: (c?.name || "").trim() || "Case laws",
              cases: Array.isArray(c?.cases) ? c.cases : [],
            }))
            .filter((c) => c.cases.length > 0);
          if (normalized.length > 0) {
            setCaseLawsLibrary(normalized);
            setCaseLawsList(normalized.flatMap((c) => c.cases));
            if (!caseLawsCourtOpen && normalized[0]?.name) {
              setCaseLawsCourtOpen(normalized[0].name);
            }
            return;
          }
        }

        // Fallback: older API that returns a flat list.
        const res = await fetch(`${API_BASE}/caselaws/list`);
        if (!res.ok) return;
        const data = await res.json().catch(() => ({}));
        setCaseLawsList(Array.isArray(data.cases) ? data.cases : []);
      } catch {
        setCaseLawsList([]);
      }
    };
    fetchCaseLaws();
  }, [API_BASE]);

  // Scroll only the chat messages area to bottom (do not move browser window or input)
  useEffect(() => {
    const el = messagesContainerRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  // Keyboard scroll in chat (Arrow Up/Down when messages area has focus)
  useEffect(() => {
    const el = messagesContainerRef.current;
    if (!el) return;
    const onKey = (e) => {
      if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
      if (document.activeElement !== el && !el.contains(document.activeElement)) return;
      e.preventDefault();
      el.scrollBy(0, e.key === "ArrowDown" ? 80 : -80);
    };
    const onFocusIn = () => el.focus();
    el.addEventListener("keydown", onKey, true);
    el.addEventListener("click", onFocusIn);
    return () => {
      el.removeEventListener("keydown", onKey, true);
      el.removeEventListener("click", onFocusIn);
    };
  }, []);

  // Elapsed time timer while loading (drives loading-step circles and "X min Y sec")
  useEffect(() => {
    if (!loading) return;
    const start = Date.now();
    const tick = () => setElapsedTime(Math.floor((Date.now() - start) / 1000));
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [loading]);

  // Fetch eval file list and figures when Eval section is expanded in bottom pane
  useEffect(() => {
    if (bottomExpandedSection !== "eval") return;
    fetch(`${API_BASE}/eval/list`)
      .then((r) => r.json())
      .then((d) => setEvalFiles(d.files || []))
      .catch(() => setEvalFiles([]));
    fetch(`${API_BASE}/eval/figures/list`)
      .then((r) => r.json())
      .then((d) => setEvalFigures(d.figures || []))
      .catch(() => setEvalFigures([]));
  }, [bottomExpandedSection, API_BASE]);

  // Fetch architecture doc when Architecture section is expanded in bottom pane
  useEffect(() => {
    if (bottomExpandedSection !== "architecture") return;
    setArchitectureLoading(true);
    fetch(`${API_BASE}/docs/architecture`)
      .then((r) => r.json())
      .then((d) => setArchitectureContent(d.content ?? null))
      .catch(() => setArchitectureContent(null))
      .finally(() => setArchitectureLoading(false));
  }, [bottomExpandedSection, API_BASE]);

  // Load Updates Tracker CSV once (for Updates section)
  useEffect(() => {
    fetch("/UPDATES_TRACKER.csv")
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error("Not found"))))
      .then((text) => {
        const lines = text.trim().split(/\r?\n/).filter((l) => l.trim());
        if (lines.length < 2) {
          setUpdatesRows([]);
          return;
        }
        const parseCSVLine = (line) => {
          const out = [];
          let cur = "";
          let inQuotes = false;
          for (let i = 0; i < line.length; i++) {
            const c = line[i];
            if (c === '"') inQuotes = !inQuotes;
            else if (c === "," && !inQuotes) {
              out.push(cur.trim());
              cur = "";
            } else if (c !== "\r") {
              cur += c;
            }
          }
          out.push(cur.trim());
          return out;
        };
        const headers = parseCSVLine(lines[0]);
        const rows = lines.slice(1).map((l) => {
          const vals = parseCSVLine(l);
          return headers.reduce((acc, h, i) => {
            acc[h] = vals[i] ?? "";
            return acc;
          }, {});
        });
        setUpdatesRows(rows);
      })
      .catch(() => setUpdatesRows([]));
  }, []);

  // -------------------------
  // SSE Stream Consumer Helper
  // -------------------------
  const consumeSSEStream = async (url, body, onProgress, onDone, onStep, onToken) => {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Request failed (${res.status})`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split(/\n\n+/);
      buffer = events.pop() || "";
      for (const raw of events) {
        let eventType = "";
        let dataLine = "";
        for (const line of raw.split(/\n/)) {
          if (line.startsWith("event:")) eventType = line.slice(6).trim();
          if (line.startsWith("data:")) dataLine = line.slice(5).trim();
        }
        if (!dataLine) continue;
        try {
          const payload = JSON.parse(dataLine);
          if (eventType === "progress" && onProgress) {
            onProgress(payload);
          } else if (eventType === "step" && onStep) {
            onStep(payload);
          } else if (eventType === "token" && onToken) {
            onToken(payload);
          } else if (eventType === "done" && onDone) {
            onDone(payload);
            return;
          }
        } catch (e) {
          // ignore parse errors for partial chunks
        }
      }
    }
    // handle any remaining buffer
    if (buffer.trim()) {
      let eventType = "";
      let dataLine = "";
      for (const line of buffer.split(/\n/)) {
        if (line.startsWith("event:")) eventType = line.slice(6).trim();
        if (line.startsWith("data:")) dataLine = line.slice(5).trim();
      }
      if (dataLine && eventType === "done" && onDone) {
        try {
          onDone(JSON.parse(dataLine));
        } catch (_) {}
      }
    }
  };

  // -------------------------
  // Reset conversation (from snippet, adapted for existing state)
  // -------------------------
  const handleStartNewCase = () => {
    hasSavedCurrentChatRef.current = false;
    currentChatIdRef.current = null;
    setStage("await_facts");
    setFacts("");
    setComposerResetSignal((prev) => prev + 1);
    setCurrentQuestion("");
    setQaHistory([]);
    setOpinionText("");
    setRetrieved([]);
    setRawResponse("");
    setError(""); // Reset existing error
    setPendingMaterials(null); // Reset existing pending materials
    setMessages([]);
  };

  const applyIndexingCandidates = (data) => {
    if (data && data.indexing_candidates && data.indexing_candidates.length) {
      const candidates = data.indexing_candidates.map((c, i) => ({
        id: `idx-${Date.now()}-${i}`,
        title: c.title || "",
        source_url: c.source_url || "",
        suggested_category: c.suggested_category || "case_law",
        category: c.suggested_category || "case_law",
        selected: !c.already_in_store,
        already_in_store: !!c.already_in_store,
      }));
      setPendingIndexingCandidates(candidates);
      const token = localStorage.getItem(AUTH_TOKEN_KEY);
      const persistItems = data.indexing_candidates.map((c) => ({
        title: c.title || "",
        source_url: c.source_url || "",
        suggested_category: c.suggested_category || "case_law",
        already_in_store: !!c.already_in_store,
      }));
      fetch(`${API_BASE}/indexing/pending`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ items: persistItems }),
      }).catch(() => {});
    }
  };

  // Save current conversation to savedChats (for sidebar list and persistence)
  const saveCurrentChatToHistory = (msgs, opinion, retr) => {
    const firstUser = (msgs || []).find((m) => m.role === "user");
    const title =
      (typeof firstUser?.content === "string" && firstUser.content.trim()) ||
      `Chat ${new Date().toLocaleString()}`;
    const chat = {
      id: Date.now(),
      title: title.length > 50 ? title.slice(0, 50) + "…" : title,
      messages: msgs || [],
      opinionText: opinion || "",
      retrieved: Array.isArray(retr) ? retr : [],
      createdAt: new Date().toISOString(),
    };
    setSavedChats((prev) => [chat, ...prev]);
    hasSavedCurrentChatRef.current = true;
  };

  // Open a saved chat in the chat window
  const handleLoadChat = (chat) => {
    setMessages(chat.messages || []);
    setOpinionText(chat.opinionText || "");
    setRetrieved(chat.retrieved || []);
    setStage("done");
    const firstUser = (chat.messages || []).find((m) => m.role === "user");
    setFacts(typeof firstUser?.content === "string" ? firstUser.content : "");
    setCurrentQuestion("");
    setQaHistory([]);
    hasSavedCurrentChatRef.current = true;
    currentChatIdRef.current = chat.id;
  };

  // New chat: save current if unsaved, then clear
  const handleNewChat = () => {
    if (messages.length > 0 && !hasSavedCurrentChatRef.current) {
      saveCurrentChatToHistory(messages, opinionText, retrieved);
    }
    handleStartNewCase();
  };

  // Group chats by date (Today, Yesterday, This week, or specific date)
  const groupChatsByDate = (chats) => {
    if (!chats.length) return [];
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    const oneDay = 24 * 60 * 60 * 1000;
    const groups = new Map();
    const getLabel = (d) => {
      const dateOnly = new Date(new Date(d).getFullYear(), new Date(d).getMonth(), new Date(d).getDate()).getTime();
      const diffDays = (today - dateOnly) / oneDay;
      if (diffDays === 0) return "Today";
      if (diffDays === 1) return "Yesterday";
      if (diffDays < 7) return "This week";
      return new Date(d).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
    };
    chats.forEach((chat) => {
      const label = getLabel(chat.createdAt);
      if (!groups.has(label)) groups.set(label, []);
      groups.get(label).push(chat);
    });
    return Array.from(groups.entries()).map(([groupLabel, groupChats]) => ({ groupLabel, chats: groupChats }));
  };

  const filteredSavedChats = useMemo(() => {
    const q = (chatHistoryFilter || "").trim().toLowerCase();
    if (!q) return savedChats;
    return savedChats.filter((c) => (c.title || "").toLowerCase().includes(q));
  }, [savedChats, chatHistoryFilter]);
  const chatGroups = useMemo(() => groupChatsByDate(filteredSavedChats), [filteredSavedChats]);

  const startRenamingChat = (chat) => {
    setEditingChatId(chat.id);
    setEditingTitle(chat.title || "");
    setTimeout(() => editInputRef.current?.focus(), 0);
  };

  const saveRenameChat = async () => {
    if (editingChatId == null) return;
    const next = (editingTitle || "").trim() || "Untitled chat";
    const token = localStorage.getItem(AUTH_TOKEN_KEY);
    const chat = savedChats.find((c) => c.id == editingChatId);
    setSavedChats((prev) => prev.map((c) => (c.id == editingChatId ? { ...c, title: next } : c)));
    setEditingChatId(null);
    setEditingTitle("");
    if (!chat) return;
    const headers = { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) };
    try {
      const res = await fetch(`${API_BASE}/chats`, { method: "POST", headers, body: JSON.stringify({
        id: chat.id, title: next, messages: chat.messages || [], opinionText: chat.opinionText || "",
        retrieved: chat.retrieved || [], createdAt: chat.createdAt || new Date().toISOString(),
      }) });
      if (res.ok) {
        const listRes = await fetch(`${API_BASE}/chats`, { headers: token ? { Authorization: `Bearer ${token}` } : {} });
        if (listRes.ok) {
          const data = await listRes.json().catch(() => ({}));
          setSavedChats(Array.isArray(data.chats) ? data.chats : []);
        }
      }
    } catch (_) {}
  };

  const cancelRenameChat = () => {
    setEditingChatId(null);
    setEditingTitle("");
  };

  const handleDeleteChat = async (chat) => {
    const idToRemove = chat.id;
    const idForUrl = typeof idToRemove === "number" ? idToRemove : String(idToRemove).trim();
    setOpenMenuChatId(null);
    const token = localStorage.getItem(AUTH_TOKEN_KEY);
    const wasCurrent = currentChatIdRef.current == idToRemove;
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    try {
      const res = await fetch(`${API_BASE}/chats/${encodeURIComponent(idForUrl)}`, { method: "DELETE", headers });
      if (res.ok) {
        const listRes = await fetch(`${API_BASE}/chats`, { headers });
        if (listRes.ok) {
          const listData = await listRes.json().catch(() => ({}));
          setSavedChats(Array.isArray(listData.chats) ? listData.chats : []);
          if (wasCurrent) handleStartNewCase();
          return;
        }
      }
    } catch (_) {}
    setSavedChats((prev) => prev.filter((c) => String(c.id) !== String(idToRemove)));
    if (wasCurrent) handleStartNewCase();
  };

  // Close chat menu when clicking outside
  useEffect(() => {
    if (openMenuChatId == null) return;
    const close = () => setOpenMenuChatId(null);
    document.addEventListener("click", close);
    return () => document.removeEventListener("click", close);
  }, [openMenuChatId]);

  // -------------------------
  // Core submit logic (adapted from snippet's handleSubmit, using existing 'input' state)
  // -------------------------
  const handleSubmit = async (submittedRaw) => {
    const raw = submittedRaw ?? "";
    if (!raw.trim()) {
      alert("Please type something before pressing Submit.");
      return;
    }

    setError("");
    setLoading(true);
    setProgress(null); // Reset progress
    setStreamingSteps([]);
    setStreamingToken("");
    setElapsedTime(0); // Reset timer
    setComposerResetSignal((prev) => prev + 1);
    setPendingMaterials(null); // Clear pending materials on new submission

    // Keep exact format user typed (spaces, newlines)
    const userMsg = { role: "user", content: raw, timestamp: new Date().toISOString() };
    const isFirstMessage = messages.length === 0;

    setMessages((prev) => [...prev, userMsg]);

    if (isFirstMessage) {
      const chatId = Date.now();
      const title = raw.trim().length > 50 ? raw.trim().slice(0, 50) + "…" : raw.trim();
      const chat = {
        id: chatId,
        title,
        messages: [userMsg],
        opinionText: "",
        retrieved: [],
        createdAt: new Date().toISOString(),
      };
      setSavedChats((prev) => [chat, ...prev]);
      currentChatIdRef.current = chatId;
      hasSavedCurrentChatRef.current = true;
      const token = localStorage.getItem(AUTH_TOKEN_KEY);
      const headers = { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) };
      fetch(`${API_BASE}/chats`, { method: "POST", headers, body: JSON.stringify(chat) }).catch(() => {});
    }

    // 1) Initial facts (await_facts stage) - use streaming
    if (stage === "await_facts") {
      setFacts(raw); // Store initial facts (format preserved)

      try {
        await consumeSSEStream(
          `${API_BASE}/submit_case/stream`,
          { text: raw, mode: chatMode },
          (progressPayload) => {
            setProgress(progressPayload);
            const groups = progressPayload.groups || [];
            if (groups.length > 0) {
              setExpandedGroups((prev) => {
                const next = { ...prev };
                groups.forEach((g) => { if (g && g.name) next[g.name] = true; });
                return next;
              });
            }
          },
          (data) => {
            setStreamingSteps([]);
            setStreamingToken("");
            setRawResponse(JSON.stringify(data, null, 2));
            if (data.status === "question" && data.next_question) {
              setCurrentQuestion(data.next_question);
              setStage("interview");
              setMessages((prev) => [
                ...prev,
                {
                  role: "assistant",
                  content: data.next_question,
                  timestamp: new Date().toISOString(),
                },
              ]);
              if (Array.isArray(data.retrieved)) setRetrieved(data.retrieved);
            } else if (data.status === "done") {
              setStage("done");
              setCurrentQuestion("");
              const opinion = data.opinion_text || "";
              const retr = Array.isArray(data.retrieved) ? data.retrieved : [];
              setOpinionText(opinion);
              setRetrieved(retr);
              applyIndexingCandidates(data);
              if (data.progress) {
                setProgress(data.progress);
                const groups = (data.progress && data.progress.groups) || [];
                if (groups.length > 0) {
                  setExpandedGroups((prev) => {
                    const next = { ...prev };
                    groups.forEach((g) => { if (g && g.name) next[g.name] = false; });
                    return next;
                  });
                }
              }
              const newAssistantMsg = {
                role: "assistant",
                content: {
                  type: "final_opinion",
                  response_type: data.response_type || "legal_opinion",
                  opinionText: opinion,
                  bare_acts: Array.isArray(data.bare_acts) ? data.bare_acts : [],
                  case_laws: Array.isArray(data.case_laws) ? data.case_laws : [],
                  case_law_discovery: false,
                  retrieved: retr,
                  progress: data.progress || null,
                  model_used: data.model_used || null,
                },
                timestamp: new Date().toISOString(),
              };
              setMessages((prev) => [...prev, newAssistantMsg]);
            } else if (data.status === "bare_acts_presented") {
              // Phase A complete: bare acts retrieved and explained. Show them + follow-up question.
              const bareActs = Array.isArray(data.bare_acts) ? data.bare_acts : [];
              const disputes  = Array.isArray(data.disputes)  ? data.disputes  : [];
              const followupQ = data.followup_question || null;
              const introText = data.opinion_text || "Here are the relevant bare act sections I found.";
              const savedFacts = data.facts_summary || facts;
              setPendingBareActs(bareActs);
              setPendingFactsSummary(savedFacts);
              setCurrentQuestion(followupQ || "");
              setStage("bare_acts_review");
              setMessages((prev) => [
                ...prev,
                {
                  role: "assistant",
                  content: {
                    type: "bare_acts_preview",
                    text: introText,
                    disputes: disputes,
                    bare_acts: bareActs,
                    followup_question: followupQ,
                  },
                  timestamp: new Date().toISOString(),
                },
              ]);
            } else if (data.needs_confirmation) {
              setPendingMaterials(data.materials_to_confirm);
              setMessages((prev) => [
                ...prev,
                {
                  role: "assistant",
                  content: data.summary,
                  timestamp: new Date().toISOString(),
                },
              ]);
            } else {
              if (data.message) setError(data.message);
            }
          },
          handleStep,
          handleToken,
        );
      } catch (err) {
        console.error("submit_case stream error:", err);
        setStreamingSteps([]);
        setStreamingToken("");
        setError(err.message || "Error during processing. Please try again.");
        setRawResponse("Error: " + (err.message || ""));
      } finally {
        setLoading(false);
      }
      return;
    }

    // 2) Follow-up answers (interview stage) - use streaming
    if (stage === "interview") {
      if (!currentQuestion) {
        alert("No current question from the assistant.");
        setLoading(false);
        return;
      }

      const updatedHistory = [
        ...qaHistory,
        { question: currentQuestion, answer: raw },
      ];
      setQaHistory(updatedHistory);
      setCurrentQuestion(""); // Clear current question after answering

      try {
        await consumeSSEStream(
          `${API_BASE}/interview_step/stream`,
          { facts, qa_history: updatedHistory, mode: chatMode },
          (progressPayload) => {
            setProgress(progressPayload);
            const groups = progressPayload.groups || [];
            if (groups.length > 0) {
              setExpandedGroups((prev) => {
                const next = { ...prev };
                groups.forEach((g) => { if (g && g.name) next[g.name] = true; });
                return next;
              });
            }
          },
          (data) => {
            setStreamingSteps([]);
            setStreamingToken("");
            setRawResponse(JSON.stringify(data, null, 2));
            if (data.status === "question" && data.next_question) {
              setCurrentQuestion(data.next_question);
              setStage("interview");
              setMessages((prev) => [
                ...prev,
                {
                  role: "assistant",
                  content: data.next_question,
                  timestamp: new Date().toISOString(),
                },
              ]);
              if (Array.isArray(data.retrieved)) setRetrieved(data.retrieved);
            } else if (data.status === "done") {
              setStage("done");
              setCurrentQuestion("");
              const opinion = data.opinion_text || "";
              const retr = Array.isArray(data.retrieved) ? data.retrieved : [];
              setOpinionText(opinion);
              setRetrieved(retr);
              applyIndexingCandidates(data);
              if (data.progress) {
                setProgress(data.progress);
                const groups = (data.progress && data.progress.groups) || [];
                if (groups.length > 0) {
                  setExpandedGroups((prev) => {
                    const next = { ...prev };
                    groups.forEach((g) => { if (g && g.name) next[g.name] = false; });
                    return next;
                  });
                }
              }
              const newAssistantMsg = {
                role: "assistant",
                content: {
                  type: "final_opinion",
                  response_type: data.response_type || "legal_opinion",
                  opinionText: opinion,
                  bare_acts: Array.isArray(data.bare_acts) ? data.bare_acts : [],
                  case_laws: Array.isArray(data.case_laws) ? data.case_laws : [],
                  retrieved: retr,
                  progress: data.progress || null,
                  model_used: data.model_used || null,
                },
                timestamp: new Date().toISOString(),
              };
              setMessages((prev) => [...prev, newAssistantMsg]);
            } else if (data.status === "bare_acts_presented") {
              const bareActs = Array.isArray(data.bare_acts) ? data.bare_acts : [];
              const disputes  = Array.isArray(data.disputes)  ? data.disputes  : [];
              const followupQ = data.followup_question || null;
              const introText = data.opinion_text || "Here are the relevant bare act sections I found.";
              const savedFacts = data.facts_summary || facts;
              setPendingBareActs(bareActs);
              setPendingFactsSummary(savedFacts);
              setCurrentQuestion(followupQ || "");
              setStage("bare_acts_review");
              setMessages((prev) => [
                ...prev,
                {
                  role: "assistant",
                  content: {
                    type: "bare_acts_preview",
                    text: introText,
                    disputes: disputes,
                    bare_acts: bareActs,
                    followup_question: followupQ,
                  },
                  timestamp: new Date().toISOString(),
                },
              ]);
            } else if (data.needs_confirmation) {
              setPendingMaterials(data.materials_to_confirm);
              setMessages((prev) => [
                ...prev,
                {
                  role: "assistant",
                  content: data.summary,
                  timestamp: new Date().toISOString(),
                },
              ]);
            } else {
              if (data.message) setError(data.message);
            }
          },
          handleStep,
          handleToken,
        );
      } catch (err) {
        console.error("interview_step stream error:", err);
        setError("Error during processing: " + (err.message || "Network or server error"));
        setRawResponse("Error: " + err.message);
      } finally {
        setLoading(false);
      }
      return;
    }

    // 2b) Bare acts review — user is answering the follow-up after Phase A
    if (stage === "bare_acts_review") {
      const updatedHistory = [
        ...qaHistory,
        { question: currentQuestion || "Any additional information?", answer: raw },
      ];
      setQaHistory(updatedHistory);
      setCurrentQuestion("");

      try {
        await consumeSSEStream(
          `${API_BASE}/interview_step/stream`,
          { facts: pendingFactsSummary || facts, qa_history: updatedHistory, bare_acts: pendingBareActs, mode: chatMode },
          (progressPayload) => {
            setProgress(progressPayload);
            const groups = progressPayload.groups || [];
            if (groups.length > 0) {
              setExpandedGroups((prev) => {
                const next = { ...prev };
                groups.forEach((g) => { if (g && g.name) next[g.name] = true; });
                return next;
              });
            }
          },
          (data) => {
            setRawResponse(JSON.stringify(data, null, 2));
            if (data.status === "done") {
              setStage("done");
              setCurrentQuestion("");
              const opinion = data.opinion_text || "";
              const retr = Array.isArray(data.retrieved) ? data.retrieved : [];
              setOpinionText(opinion);
              setRetrieved(retr);
              applyIndexingCandidates(data);
              if (data.progress) {
                setProgress(data.progress);
                const groups = (data.progress && data.progress.groups) || [];
                if (groups.length > 0) {
                  setExpandedGroups((prev) => {
                    const next = { ...prev };
                    groups.forEach((g) => { if (g && g.name) next[g.name] = false; });
                    return next;
                  });
                }
              }
              setMessages((prev) => [
                ...prev,
                {
                  role: "assistant",
                  content: {
                    type: "final_opinion",
                    response_type: data.response_type || "legal_opinion",
                    opinionText: opinion,
                    bare_acts: Array.isArray(data.bare_acts) ? data.bare_acts : [],
                    case_laws: Array.isArray(data.case_laws) ? data.case_laws : [],
                    retrieved: retr,
                    progress: data.progress || null,
                    model_used: data.model_used || null,
                  },
                  timestamp: new Date().toISOString(),
                },
              ]);
              // Clean up bare acts phase state
              setPendingBareActs([]);
              setPendingFactsSummary("");
            } else if (data.status === "question") {
              setCurrentQuestion(data.next_question);
              setStage("interview");
              setMessages((prev) => [
                ...prev,
                { role: "assistant", content: data.next_question, timestamp: new Date().toISOString() },
              ]);
            }
            setStreamingSteps([]);
            setStreamingToken("");
          },
          handleStep,
          handleToken,
        );
      } catch (err) {
        console.error("bare_acts_review stream error:", err);
        setStreamingSteps([]);
        setStreamingToken("");
        setError("Error during processing: " + (err.message || "Network or server error"));
      } finally {
        setLoading(false);
      }
      return;
    }

    // 3) Continue a loaded chat: use stream endpoint for live progress
    if (stage === "done") {
      const normalizeContent = (msg) => {
        if (typeof msg.content === "string") return msg.content;
        if (msg.content?.opinionText != null) return msg.content.opinionText || "";
        return msg.content?.text ?? msg.content?.summary ?? "";
      };
      const previousMessages = messages.slice(0, -1);
      const conversation = previousMessages.map((m) => ({ role: m.role, content: normalizeContent(m) }));
      try {
        await new Promise((r) => setTimeout(r, 0));
        await consumeSSEStream(
          `${API_BASE}/conversation/continue/stream`,
          { conversation, message: raw, mode: chatMode },
          (progressPayload) => {
            setProgress(progressPayload);
            const groups = progressPayload.groups || [];
            if (groups.length > 0) {
              setExpandedGroups((prev) => {
                const next = { ...prev };
                groups.forEach((g) => { if (g && g.name) next[g.name] = true; });
                return next;
              });
            }
          },
          (data) => {
            const assistantContent = (data.next_question ?? data.message ?? "").trim();
            if (data.status === "question") {
              setMessages((prev) => [
                ...prev,
                { role: "assistant", content: assistantContent || "Could you tell me more?", timestamp: new Date().toISOString() },
              ]);
              if (Array.isArray(data.retrieved)) setRetrieved(data.retrieved);
            } else if (data.status === "done") {
              const opinion = data.opinion_text || "Your request has been processed.";
              const retr = Array.isArray(data.retrieved) ? data.retrieved : [];
              setOpinionText(opinion);
              setRetrieved(retr);
              applyIndexingCandidates(data);
              if (data.progress) {
                setProgress(data.progress);
                const groups = (data.progress && data.progress.groups) || [];
                if (groups.length > 0) {
                  setExpandedGroups((prev) => {
                    const next = { ...prev };
                    groups.forEach((g) => { if (g && g.name) next[g.name] = false; });
                    return next;
                  });
                }
              }
              setMessages((prev) => [
                ...prev,
                {
                  role: "assistant",
                  content: {
                    type: "final_opinion",
                    response_type: data.response_type || "legal_opinion",
                    opinionText: opinion,
                    bare_acts: Array.isArray(data.bare_acts) ? data.bare_acts : [],
                    case_laws: Array.isArray(data.case_laws) ? data.case_laws : [],
                    case_law_discovery: false,
                    retrieved: retr,
                    progress: data.progress || null,
                    model_used: data.model_used || null,
                  },
                  timestamp: new Date().toISOString(),
                },
              ]);
            } else if (data.status === "bare_acts_presented") {
              const bareActs = Array.isArray(data.bare_acts) ? data.bare_acts : [];
              const disputes  = Array.isArray(data.disputes)  ? data.disputes  : [];
              const followupQ = data.followup_question || null;
              const introText = data.opinion_text || "Here are the relevant bare act sections I found.";
              const savedFacts = data.facts_summary || facts;
              setPendingBareActs(bareActs);
              setPendingFactsSummary(savedFacts);
              setCurrentQuestion(followupQ || "");
              setStage("bare_acts_review");
              setMessages((prev) => [
                ...prev,
                {
                  role: "assistant",
                  content: {
                    type: "bare_acts_preview",
                    text: introText,
                    disputes: disputes,
                    bare_acts: bareActs,
                    followup_question: followupQ,
                  },
                  timestamp: new Date().toISOString(),
                },
              ]);
            } else if (data.needs_confirmation) {
              setPendingMaterials(data.materials_to_confirm);
              setMessages((prev) => [
                ...prev,
                { role: "assistant", content: data.summary ?? "Please confirm the materials to proceed.", timestamp: new Date().toISOString() },
              ]);
            } else {
              const fallbackContent = assistantContent || data.opinion_text || data.summary || "Processing...";
              setMessages((prev) => [
                ...prev,
                { role: "assistant", content: fallbackContent, timestamp: new Date().toISOString() },
              ]);
            }
            setStreamingSteps([]);
            setStreamingToken("");
          },
          handleStep,
          handleToken,
        );
      } catch (err) {
        setError("Error continuing chat: " + (err.message || ""));
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: "Sorry, something went wrong. Please try again.", timestamp: new Date().toISOString() },
        ]);
      } finally {
        setLoading(false);
      }
      return;
    }
  };

  // -------------------------
  // Progress Display Component — single collapsible "Progress tracker" with all steps
  // -------------------------
  const PROGRESS_TRACKER_KEY = "progress_tracker";
  const ProgressDisplay = ({ progress, expandedGroups, setExpandedGroups }) => {
    if (!progress || !progress.groups || progress.groups.length === 0) return null;

    const isExpanded = expandedGroups[PROGRESS_TRACKER_KEY] !== undefined ? expandedGroups[PROGRESS_TRACKER_KEY] : true;
    const allSteps = progress.groups.flatMap((g) => (g ? (g.steps || []).map((s) => ({ ...s, groupName: g.name })) : []));
    const totalStats = progress.groups.reduce(
      (acc, g) => {
        const s = (g && g.stats) || {};
        acc.searched += s.total_searched || 0;
        acc.passed += s.passed_threshold || 0;
        acc.included += s.included || 0;
        return acc;
      },
      { searched: 0, passed: 0, included: 0 },
    );

    return (
      <div className="progress-display">
        <details
          className="progress-group"
          open={isExpanded}
          onClick={(e) => {
            if (e.target.closest("summary")) {
              e.preventDefault();
              setExpandedGroups((prev) => ({
                ...prev,
                [PROGRESS_TRACKER_KEY]: !(prev[PROGRESS_TRACKER_KEY] !== undefined ? prev[PROGRESS_TRACKER_KEY] : true),
              }));
            }
          }}
        >
          <summary className="progress-group-summary">
            <span className="progress-group-name">Progress tracker</span>
            {totalStats.searched > 0 && (
              <span className="progress-group-stats">
                {totalStats.searched} searched, {totalStats.passed} passed, {totalStats.included} included
              </span>
            )}
          </summary>
          <div
            className="progress-steps-outer"
            role="region"
            aria-label="Progress steps"
            onMouseDown={(e) => e.stopPropagation()}
            onPointerDown={(e) => e.stopPropagation()}
          >
            <div className="progress-steps-wrapper">
              <div className="progress-steps">
                {allSteps.map((step, stepIdx) => {
                  const isDocScan = step.metadata?.doc_name;
                  const included = step.metadata?.included;
                  const score = step.metadata?.score;
                  const alreadyTitles = step.metadata?.already_in_library_titles;
                  const hasAlreadyInLibrary = Array.isArray(alreadyTitles) && alreadyTitles.length > 0;
                  const timeStr = step.timestamp != null ? `${Number(step.timestamp).toFixed(1)}s` : (step.duration_seconds != null ? `${step.duration_seconds}s` : "");
                  const messageStr = step.message ?? step.name ?? "";
                  return (
                    <div
                      key={stepIdx}
                      className={`progress-step ${isDocScan ? (included ? "progress-step-included" : "progress-step-ignored") : ""}`}
                    >
                      <span className="progress-step-dot progress-step-dot--completed" aria-hidden />
                      {timeStr && <span className="progress-step-time">{timeStr}</span>}
                      <span className="progress-step-message">{messageStr}</span>
                      {score !== undefined && (
                        <span className={`progress-step-score ${included ? "score-included" : "score-ignored"}`}>
                          Score: {score}
                        </span>
                      )}
                      {hasAlreadyInLibrary && (
                        <div className="progress-step-already-in-library" aria-label="Already in library">
                          <span className="progress-step-already-label">{alreadyTitles.length} already in library:</span>
                          <ul className="progress-step-already-list">
                            {alreadyTitles.slice(0, 20).map((t, i) => (
                              <li key={i} title={t}>{t.length > 50 ? t.slice(0, 50) + "…" : t}</li>
                            ))}
                            {alreadyTitles.length > 20 && (
                              <li className="progress-step-already-more">+{alreadyTitles.length - 20} more</li>
                            )}
                          </ul>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </details>
      </div>
    );
  };

  // -------------------------
  // Eval Section — JSON as tables (see .cursor/rules/eval-json-ui-rendering.md)
  // - Metrics objects: rows = metric names, cols = mean, n, stdev, ci_95
  // - Aggregated-by-key: rows = outer keys (e.g. threshold), cols = inner keys
  // - Array of objects: flatten nested objects to sub-columns, arrays to length; no collapsibles
  // -------------------------
  const downloadTableCsv = (headers, rows, filename) => {
    const escape = (v) => {
      const s = String(v ?? "");
      if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
      return s;
    };
    const line = (arr) => arr.map(escape).join(",");
    const csv = [line(headers), ...rows.map((r) => line(r))].join("\r\n");
    const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${filename || "table"}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  const downloadTableExcel = (headers, rows, filename) => {
    try {
      const ws = XLSX.utils.aoa_to_sheet([headers, ...rows]);
      const wb = XLSX.utils.book_new();
      XLSX.utils.book_append_sheet(wb, ws, "Sheet1");
      XLSX.writeFile(wb, `${filename || "table"}.xlsx`);
    } catch (e) {
      console.error("Excel export failed:", e);
    }
  };
  const EvalTableToolbar = ({ headers, rows, filename }) => {
    if (!headers?.length) return null;
    const base = filename || "eval-table";
    return (
      <div className="eval-table-toolbar">
        <button type="button" className="eval-download-btn" onClick={() => downloadTableCsv(headers, rows, base)}>
          Download CSV
        </button>
        <button type="button" className="eval-download-btn" onClick={() => downloadTableExcel(headers, rows, base)}>
          Download Excel
        </button>
      </div>
    );
  };
  const JsonToTable = ({ data, baseFilename = "eval-table" }) => {
    if (data == null) return null;
    // Detect metrics object: { metric_name: { mean, n, stdev, ci_95 }, ... } → table with metrics as rows
    const isMetricsObject = (o) => {
      if (!o || typeof o !== "object" || Array.isArray(o)) return false;
      const entries = Object.entries(o);
      if (entries.length === 0) return false;
      const firstVal = entries[0][1];
      if (!firstVal || typeof firstVal !== "object" || Array.isArray(firstVal)) return false;
      const statKeys = ["mean", "n", "stdev", "ci_95"];
      const hasStats = statKeys.some((k) => k in firstVal);
      if (!hasStats) return false;
      return entries.every(([, v]) => v && typeof v === "object" && !Array.isArray(v));
    };
    if (typeof data === "object" && !Array.isArray(data) && isMetricsObject(data)) {
      const statKeys = ["mean", "n", "stdev", "ci_95"];
      const headers = ["metric", ...statKeys];
      const rows = Object.entries(data).map(([metric, stats]) => [
        metric,
        ...statKeys.map((k) => (stats && stats[k] != null) ? (typeof stats[k] === "number" ? Number(stats[k]).toFixed(4) : String(stats[k])) : "—"),
      ]);
      return (
        <div className="eval-table-block">
          <EvalTableToolbar headers={headers} rows={rows} filename={baseFilename} />
          <div className="eval-table-wrap">
            <table className="eval-table eval-table--metrics">
              <thead><tr><th>metric</th>{statKeys.map((k) => <th key={k}>{k}</th>)}</tr></thead>
              <tbody>
                {Object.entries(data).map(([metric, stats]) => (
                  <tr key={metric}>
                    <td className="eval-metric-name">{metric}</td>
                    {statKeys.map((k) => (
                      <td key={k}>{(stats && stats[k] != null) ? (typeof stats[k] === "number" ? Number(stats[k]).toFixed(4) : String(stats[k])) : "—"}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      );
    }
    // Detect aggregated-by-key object (e.g. threshold_sweep aggregated): { "0.1": { mean_precision, mean_recall, ... }, ... } → rows = keys, columns = metric names
    const isAggregatedTable = (o) => {
      if (!o || typeof o !== "object" || Array.isArray(o)) return false;
      const entries = Object.entries(o);
      if (entries.length === 0) return false;
      const firstVal = entries[0][1];
      if (!firstVal || typeof firstVal !== "object" || Array.isArray(firstVal)) return false;
      if (isMetricsObject(o)) return false; // already handled
      const keys = Object.keys(firstVal);
      const allScalar = keys.every((k) => {
        const v = firstVal[k];
        return v == null || typeof v !== "object" || Array.isArray(v);
      });
      if (!allScalar) return false;
      return entries.every(([, v]) => v && typeof v === "object" && !Array.isArray(v) && Object.keys(v).every((k) => typeof v[k] !== "object" || v[k] == null));
    };
    if (typeof data === "object" && !Array.isArray(data) && isAggregatedTable(data)) {
      const rowKeys = Object.keys(data).sort((a, b) => parseFloat(a) - parseFloat(b));
      const colKeys = [...new Set(rowKeys.flatMap((rk) => Object.keys(data[rk] || {})))];
      const headers = ["threshold", ...colKeys];
      const rows = rowKeys.map((rk) => {
        const row = data[rk] || {};
        return [rk, ...colKeys.map((k) => {
          const v = row[k];
          return v != null ? (typeof v === "number" ? (Number.isInteger(v) ? String(v) : Number(v).toFixed(4)) : String(v)) : "—";
        })];
      });
      return (
        <div className="eval-table-block">
          <EvalTableToolbar headers={headers} rows={rows} filename={baseFilename} />
          <div className="eval-table-wrap">
            <table className="eval-table eval-table--metrics">
              <thead><tr><th>threshold</th>{colKeys.map((k) => <th key={k}>{k}</th>)}</tr></thead>
              <tbody>
                {rowKeys.map((rk) => {
                  const row = data[rk] || {};
                  return (
                    <tr key={rk}>
                      <td className="eval-metric-name">{rk}</td>
                      {colKeys.map((k) => {
                        const v = row[k];
                        const disp = v != null ? (typeof v === "number" ? (Number.isInteger(v) ? String(v) : Number(v).toFixed(4)) : String(v)) : "—";
                        return <td key={k}>{disp}</td>;
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      );
    }
    if (Array.isArray(data)) {
      if (data.length === 0) return <p className="eval-empty">Empty array</p>;
      const first = data[0];
      if (typeof first !== "object" || first === null) {
        const headers = ["value"];
        const rows = data.map((v) => [String(v)]);
        return (
          <div className="eval-table-block">
            <EvalTableToolbar headers={headers} rows={rows} filename={baseFilename} />
            <div className="eval-table-wrap">
              <table className="eval-table">
                <tbody>
                  {data.map((v, i) => (
                    <tr key={i}><td>{String(v)}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        );
      }
      // Flatten each row: nested objects become sub-columns (e.g. filter_results.bare_acts_after_score_filter), arrays become length
      const flattenRow = (obj, prefix = "") => {
        const out = {};
        if (obj == null) return out;
        for (const [k, v] of Object.entries(obj)) {
          const key = prefix ? `${prefix}.${k}` : k;
          if (v == null) {
            out[key] = "";
          } else if (Array.isArray(v)) {
            out[key] = v.length;
          } else if (typeof v === "object") {
            const allScalar = Object.values(v).every(
              (x) => x == null || typeof x !== "object" || Array.isArray(x)
            );
            if (allScalar) {
              for (const [kk, vv] of Object.entries(v)) {
                const subKey = prefix ? `${prefix}.${k}.${kk}` : `${k}.${kk}`;
                out[subKey] = Array.isArray(vv) ? vv.length : vv;
              }
            } else {
              Object.assign(out, flattenRow(v, key));
            }
          } else {
            out[key] = v;
          }
        }
        return out;
      };
      const flattened = data.slice(0, 100).map((r) => flattenRow(r));
      const keys = [...new Set(flattened.flatMap((r) => Object.keys(r)))].sort();
      if (keys.length === 0) return null;
      const headers = keys;
      const rows = flattened.map((row) => keys.map((k) => {
        const v = row[k];
        return v == null ? "" : typeof v === "string" && v.length > 150 ? v.slice(0, 150) + "…" : String(v);
      }));
      return (
        <div className="eval-table-block">
          <EvalTableToolbar headers={headers} rows={rows} filename={baseFilename} />
          <div className="eval-table-wrap">
            <table className="eval-table eval-table--flat">
              <thead><tr>{keys.map((k) => <th key={k}>{k}</th>)}</tr></thead>
              <tbody>
                {flattened.map((row, i) => (
                  <tr key={i}>
                    {keys.map((k) => {
                      const v = row[k];
                      const s = v == null ? "" : typeof v === "string" && v.length > 150 ? v.slice(0, 150) + "…" : String(v);
                      return <td key={k}>{s}</td>;
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {data.length > 100 && <p className="eval-truncated">Showing first 100 of {data.length} rows</p>}
        </div>
      );
    }
    if (typeof data === "object") {
      const entries = Object.entries(data);
      if (entries.length === 0) return <p className="eval-empty">Empty object</p>;
      // If top-level keys are metrics objects (e.g. compare.json: bm25, merged, full), show each as heading + table, no collapse
      if (entries.every(([, v]) => isMetricsObject(v))) {
        return (
          <div className="eval-object-wrap">
            {entries.map(([runName, metrics]) => (
              <div key={runName} className="eval-run-block">
                <h5 className="eval-run-title">{runName}</h5>
                <JsonToTable data={metrics} baseFilename={`${baseFilename}-${runName}`} />
              </div>
            ))}
          </div>
        );
      }
      // Other objects: flatten into key-value table, no details/collapse
      const isFlat = (o) => o == null || typeof o !== "object" || typeof o === "number" || typeof o === "string" || typeof o === "boolean";
      return (
        <div className="eval-object-wrap">
          {entries.map(([k, v]) => (
            <div key={k} className="eval-kv-block">
              <span className="eval-kv-key">{k}:</span>
              {v != null && typeof v === "object" && !Array.isArray(v) && Object.values(v).every(isFlat) ? (
                <table className="eval-table eval-table--keyval">
                  <tbody>
                    {Object.entries(v).map(([kk, vv]) => (
                      <tr key={kk}><td>{kk}</td><td>{JSON.stringify(vv)}</td></tr>
                    ))}
                  </tbody>
                </table>
              ) : v != null && (typeof v === "object" || Array.isArray(v)) ? (
                <JsonToTable data={v} baseFilename={baseFilename ? `${baseFilename}-${k}` : k} />
              ) : (
                <span>{String(v)}</span>
              )}
            </div>
          ))}
        </div>
      );
    }
    return <span>{String(data)}</span>;
  };

  const EvalSection = () => (
    <details
      className="eval-section"
      open={bottomExpandedSection === "eval"}
    >
      <summary
        className="eval-section-summary"
        onClick={(e) => {
          e.preventDefault();
          setBottomExpandedSection((prev) => (prev === "eval" ? null : "eval"));
        }}
      >
        Eval Results
      </summary>
      <div className="eval-section-body">
        {evalFiles.length === 0 ? (
          <p className="eval-message">No eval files found. Run batch eval to generate results.</p>
        ) : (
          <div className="eval-files">
            {evalFiles.map((f) => (
              <button
                key={f.path}
                type="button"
                className={`eval-file-btn ${evalLoaded?.path === f.path ? "eval-file-btn--active" : ""}`}
                onClick={() => {
                  setEvalLoading(true);
                  fetch(`${API_BASE}/eval/file?path=${encodeURIComponent(f.path)}`)
                    .then((r) => r.json())
                    .then((data) => setEvalLoaded({ path: f.path, data }))
                    .catch(() => setEvalLoaded(null))
                    .finally(() => setEvalLoading(false));
                }}
              >
                {f.name || f.path}
              </button>
            ))}
          </div>
        )}
        {evalLoading && <p className="eval-loading">Loading…</p>}
        {evalLoaded && !evalLoading && (
          <div className="eval-content">
            <h4 className="eval-file-title">{evalLoaded.path}</h4>
            <JsonToTable data={evalLoaded.data} baseFilename={(evalLoaded.path || "eval").replace(/\.json$/i, "").replace(/\//g, "-")} />
          </div>
        )}
        {evalFigures.length > 0 && (
          <div className="eval-figures">
            <h4 className="eval-figures-title">Figures</h4>
            <div className="eval-figures-grid">
              {evalFigures.map((fig) => (
                <figure key={fig.path} className="eval-figure">
                  <img
                    src={`${API_BASE}/eval/figures/file?path=${encodeURIComponent(fig.path)}`}
                    alt={fig.name}
                    className="eval-figure-img"
                  />
                  <figcaption className="eval-figure-caption">{fig.name}</figcaption>
                </figure>
              ))}
            </div>
          </div>
        )}
      </div>
    </details>
  );

  const ArchitectureSection = () => (
    <details
      className="architecture-section"
      open={bottomExpandedSection === "architecture"}
    >
      <summary
        className="architecture-section-summary"
        onClick={(e) => {
          e.preventDefault();
          setBottomExpandedSection((prev) =>
            prev === "architecture" ? null : "architecture"
          );
        }}
      >
        Architecture
      </summary>
      <div className="architecture-section-body">
        {architectureLoading && <p className="architecture-loading">Loading…</p>}
        {!architectureLoading && architectureContent && (
          <div className="architecture-content">
            <ReactMarkdown rehypePlugins={[rehypeRaw]}>{architectureContent}</ReactMarkdown>
          </div>
        )}
        {!architectureLoading && !architectureContent && bottomExpandedSection === "architecture" && (
          <p className="architecture-message">Could not load architecture doc.</p>
        )}
      </div>
    </details>
  );

  const UpdatesTrackerSection = () => (
    <details
      className="updates-tracker-section"
      open={bottomExpandedSection === "updates"}
    >
      <summary
        className="updates-tracker-summary"
        onClick={(e) => {
          e.preventDefault();
          setBottomExpandedSection((prev) => (prev === "updates" ? null : "updates"));
        }}
      >
        Updates Tracker
      </summary>
      <div className="updates-tracker-body">
        {updatesRows.length === 0 ? (
          <p className="eval-message">
            No updates data. Ensure UPDATES_TRACKER.csv is present in the project root / Frontend/public.
          </p>
        ) : (
          <div className="updates-tracker-table-wrap">
            <table className="eval-table">
              <thead>
                <tr>
                  {Object.keys(updatesRows[0] || {}).map((k) => (
                    <th key={k}>{k}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {updatesRows.map((row, i) => (
                  <tr key={i}>
                    {Object.keys(updatesRows[0] || {}).map((k) => (
                      <td key={k}>{row[k] ?? ""}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </details>
  );

  // -------------------------
  // Existing functions (retained and adapted)
  // -------------------------
  const toApiContent = (msg) => {
    if (typeof msg.content === "string") return msg.content;
    if (msg.content?.text) return msg.content.text;
    if (msg.content?.type === "results" && msg.content.parts) {
      const exp = msg.content.parts.find((p) => p.type === "explanation");
      return exp?.text || "[Legal research response]";
    }
    return "[Message]";
  };

  const getMessageTextForCopy = (msg) => {
    if (msg.role === "user") return typeof msg.content === "string" ? msg.content : String(msg.content ?? "");
    return toApiContent(msg);
  };

  const handleCopyMessage = (msg, messageIndex) => {
    const str = getMessageTextForCopy(msg);
    navigator.clipboard.writeText(str).then(() => {
      setCopyJustDoneIndex(messageIndex);
      setActionMenuOpenIndex(null);
      setTimeout(() => setCopyJustDoneIndex(null), 1500);
    }).catch(() => {});
  };

  useEffect(() => {
    if (actionMenuOpenIndex == null) return;
    const close = (e) => {
      if (!e.target.closest(".message-action-menu-wrap")) setActionMenuOpenIndex(null);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [actionMenuOpenIndex]);

  const startEditUserMessage = (index) => {
    const msg = messages[index];
    if (msg?.role !== "user") return;
    const content = typeof msg.content === "string" ? msg.content : "";
    setEditDraft(content);
    setEditingMessageIndex(index);
    setTimeout(() => editMessageInputRef.current?.focus(), 0);
  };

  const saveEditUserMessage = () => {
    if (editingMessageIndex == null) return;
    setMessages((prev) =>
      prev.map((m, j) =>
        j === editingMessageIndex && m.role === "user"
          ? { ...m, content: editDraft }
          : m
      )
    );
    setEditingMessageIndex(null);
    setEditDraft("");
  };

  const cancelEditUserMessage = () => {
    setEditingMessageIndex(null);
    setEditDraft("");
  };

  const confirmAndIndex = async () => {
    if (!pendingMaterials || loading) return;
    const hasBare = pendingMaterials.bare_acts?.length > 0;
    const hasCase = pendingMaterials.case_laws?.length > 0;
    if (!hasBare && !hasCase) return;

    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/chat/confirm-index`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bare_acts: pendingMaterials.bare_acts || [],
          case_laws: pendingMaterials.case_laws || [],
          facts_summary: factsSummary || "",
        }),
      });
      let data = {};
      try {
        const text = await res.text();
        data = text ? JSON.parse(text) : {};
      } catch (_) {
        setError("Server returned an invalid or empty response.");
        return;
      }
      if (data.success) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: {
              type: "indexed",
              text: data.message || "Materials have been indexed successfully.",
            },
            timestamp: new Date().toISOString(),
          },
        ]);
        setPendingMaterials(null);
        setStage("done"); // Assuming after indexing, we are done with current interview
        setCurrentQuestion(""); // Clear current question

        if (data.response) {
          const fullContent = buildAssistantContent({
            response: data.response,
            message: data.response.explanation,
          });
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: fullContent, timestamp: new Date().toISOString() },
          ]);
          setOpinionText(data.response.explanation || ""); // Update opinion text if provided
          if (Array.isArray(data.response.bare_act_sections) || Array.isArray(data.response.case_laws)) {
            setRetrieved([...(data.response.bare_act_sections || []), ...(data.response.case_laws || [])]);
          }
        }
      } else {
        setError(data.message || "Indexing failed");
      }
    } catch (err) {
      setError("Failed to index materials: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const buildAssistantContent = (data) => { // Removed isConfirmResponse as it's not used
    if (data.response) {
      const parts = [];
      if (data.message) {
        parts.push({ type: "explanation", text: data.message });
      }
      if (data.response.bare_act_sections?.length > 0) {
        parts.push({
          type: "bare",
          title: "📘 Bare Act Sections",
          items: data.response.bare_act_sections,
        });
      }
      if (data.response.case_laws?.length > 0) {
        parts.push({
          type: "case",
          title: "📚 Case Laws (from database)",
          items: data.response.case_laws,
        });
      }
      if (data.response.internet_case_laws?.length > 0) {
        parts.push({
          type: "internet_case",
          title: "📚 Case Laws (from internet search)",
          items: data.response.internet_case_laws,
        });
      }
      if (parts.length === 0 && data.message) {
        return { type: "results", parts: [{ type: "explanation", text: data.message }] };
      }
      return { type: "results", parts };
    }
    if (data.message) {
      return { type: "question", text: data.message };
    }
    return { type: "question", text: "How can I help?" };
  };

  const renderAssistantContent = (content) => {
    if (content == null) return null;
    const trimDots = (s) => (s || "").replace(/\n+\.\.\.\s*$/, " …").replace(/\n+$/, "");
    if (typeof content === "string") {
      return <p className="message-text">{trimDots(content)}</p>;
    }
    if (content.type === "question") {
      return <p className="message-text">{trimDots(content.text)}</p>;
    }
    if (content.type === "error") {
      return <p className="message-error">{content.text}</p>;
    }
    if (content.type === "indexed") { // Added for indexed message type
      return <p className="message-indexed">{content.text}</p>;
    }
    if (content.type === "bare_acts_preview") {
      const disputes  = content.disputes  || [];
      const bareActs  = content.bare_acts  || [];
      const followupQ = content.followup_question || null;

      // Short names for well-known Indian acts — used in section headings
      const ACT_SHORT_NAMES = {
        "bharatiya nyaya sanhita 2023":              "BNS",
        "bharatiya nyaya sanhita":                   "BNS",
        "bharatiya nagarik suraksha sanhita 2023":   "BNSS",
        "bharatiya nagarik suraksha sanhita":        "BNSS",
        "bharatiya sakshya adhiniyam 2023":          "BSA",
        "bharatiya sakshya adhiniyam":               "BSA",
        "indian penal code 1860":                    "IPC",
        "indian penal code":                         "IPC",
        "code of criminal procedure 1973":           "CrPC",
        "code of criminal procedure":                "CrPC",
        "indian evidence act 1872":                  "IEA",
        "indian evidence act":                       "IEA",
        "code of civil procedure 1908":              "CPC",
        "code of civil procedure":                   "CPC",
        "transfer of property act 1882":             "TP Act",
        "transfer of property act":                  "TP Act",
        "specific relief act 1963":                  "Specific Relief Act",
        "specific relief act":                       "Specific Relief Act",
        "negotiable instruments act 1881":           "NI Act",
        "negotiable instruments act":                "NI Act",
        "hindu marriage act 1955":                   "HMA",
        "hindu marriage act":                        "HMA",
        "hindu succession act 1956":                 "HSA",
        "hindu succession act":                      "HSA",
        "consumer protection act 2019":              "Consumer Protection Act",
        "consumer protection act":                   "Consumer Protection Act",
        "arbitration and conciliation act 1996":     "Arbitration Act",
        "arbitration and conciliation act":          "Arbitration Act",
        "indian contract act 1872":                  "Contract Act",
        "indian contract act":                       "Contract Act",
        "registration act 1908":                     "Registration Act",
        "limitation act 1963":                       "Limitation Act",
        "motor vehicles act 1988":                   "MV Act",
        "companies act 2013":                        "Companies Act",
        "protection of women from domestic violence act 2005": "DV Act",
        "protection of women from domestic violence act":      "DV Act",
        "dowry prohibition act 1961":                "Dowry Act",
        "telangana land encroachment act":           "TG Encroachment Act",
      };

      const shortActName = (actName) => {
        const key = (actName || "").toLowerCase().trim();
        return ACT_SHORT_NAMES[key] || actName;
      };

      // Build dispute groups: prefer grouped disputes from backend; fall back to flat list
      const disputeGroups = disputes.length > 0
        ? disputes
        : bareActs.length > 0
          ? [{ id: "d_main", dispute: "", sections: bareActs }]
          : [];

      const introText = (content.text || "").trim();

      return (
        <div className="message-bare-acts-preview message-bare-acts-preview--compact">
          {introText && (
            <p className="bare-acts-preview-intro">{introText}</p>
          )}
          {/* Per-dispute: dispute line, then section number + act name + crisp summary only */}
          {disputeGroups.map((group, gi) => {
            const sections = group.sections || [];
            const multipleDisputes = disputeGroups.length > 1;
            return (
              <div key={group.id || gi} className="dispute-block dispute-block--compact">
                {(multipleDisputes || group.dispute) && (
                  <p className="dispute-block-header dispute-block-header--compact">
                    {multipleDisputes && (
                      <span className="dispute-block-number">Dispute {gi + 1}{group.dispute ? ": " : ""}</span>
                    )}
                    {group.dispute && (
                      <span className="dispute-block-description">{group.dispute}</span>
                    )}
                  </p>
                )}

                <ul className="bare-act-section-list--compact">
                  {sections.map((ba, si) => {
                    const actName = ba.act_name || "Unknown Act";
                    const short = shortActName(actName);
                    const secNum = ba.section_number || "?";
                    const summary = (ba.explanation || "").trim() || "Relevant to this dispute.";
                    const isWeb = !!ba._web_sourced;
                    const sourceUrl = ba.url || null;

                    const handleAddToIndex = async () => {
                      try {
                        const resp = await fetch(`${API_BASE}/propose_index`, {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ section: ba }),
                        });
                        const data = await resp.json();
                        alert(data.message || "Saved for indexing.");
                      } catch {
                        alert("Could not save section. Please try again.");
                      }
                    };

                    return (
                      <li key={si} className="bare-act-section-item--compact">
                        <span className="bare-act-section-ref">Section {secNum}, {short}</span>
                        <span className="bare-act-section-summary"> — {summary}</span>
                        {isWeb && (
                          <span className="bare-act-section-actions-inline">
                            {" "}
                            <a href={sourceUrl} target="_blank" rel="noreferrer" className="web-section-source-link">View source</a>
                            <button type="button" className="add-to-index-btn-inline" onClick={handleAddToIndex}>+ Index</button>
                          </span>
                        )}
                      </li>
                    );
                  })}
                </ul>
              </div>
            );
          })}

          {/* Separator + single follow-up question or next-steps prompt */}
          <div className="bare-acts-followup-separator" />
          {followupQ ? (
            <div className="bare-acts-followup">
              {followupQ.includes("\n") ? (
                <div className="bare-acts-followup-list">
                  {(() => {
                    const lines = followupQ.split("\n").filter(Boolean);
                    const introLine = lines[0] && !lines[0].trim().startsWith("•") ? lines[0] : null;
                    const bullets = lines.filter((l) => l.trim().startsWith("•"));
                    return (
                      <>
                        {introLine && <p className="bare-acts-followup-intro">{introLine}</p>}
                        {bullets.length > 0 && (
                          <ul className="bare-acts-followup-ul">
                            {bullets.map((line, i) => (
                              <li key={i} className="bare-acts-followup-bullet">{line.trim().replace(/^•\s*/, "")}</li>
                            ))}
                          </ul>
                        )}
                      </>
                    );
                  })()}
                </div>
              ) : (
                <p className="bare-acts-followup-text">{followupQ}</p>
              )}
            </div>
          ) : (
            <p className="bare-acts-next-steps">
              If you'd like, I can find relevant court judgments on this, or give you a full legal opinion.
            </p>
          )}

        </div>
      );
    }
    if (content.type === "final_opinion") {
      const opinion = content.opinionText || "";
      const responseType = content.response_type || "legal_opinion";
      const bareActs = content.bare_acts || [];
      const caseLaws = content.case_laws || [];
      const messageProgress = content.progress || null; // Get progress from message content

      // Helper: render a single item row inside a grouped box
      const renderResultItem = (item, idx) => {
        const title = item.title || item.act_name || item.source || `Result ${idx + 1}`;
        const url = item.url || item.source_url || ""; // Check multiple URL fields
        const rawText = item.text || "";
        // Clean: remove very short fragments and navigation-like lines, but show full content
        const cleanLines = rawText
          .split(/[.\n]/)
          .map(l => l.trim())
          .filter(l => l.length > 20 && !/^(Skip|Search|Login|Menu|Toggle|Free|Premium|Print|Download|Pricing)/i.test(l));
        const cleanedText = cleanLines.join(". ").trim();
        return (
          <div key={idx} className="result-item-row">
            <div className="result-item-header">
              <span className="result-item-number">{idx + 1}.</span>
              {url ? (
                <a
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="result-item-title-link"
                  title="Opens in a new tab."
                >
                  {title}
                </a>
              ) : (
                <span className="result-item-title">{title}</span>
              )}
            </div>
            {cleanedText && <p className="result-item-full-text">{cleanedText}</p>}
            {url && (
              <a
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                className="result-item-source-link"
                title="Opens in a new tab."
              >
                View original source
              </a>
            )}
          </div>
        );
      };

      // Helper: render bare act with nested case laws
      const renderBareActWithCaseLaws = (bareAct, idx) => {
        const title = bareAct.title || bareAct.act_name || bareAct.source || `Bare Act ${idx + 1}`;
        const url = bareAct.url || bareAct.source_url || "";
        const rawText = bareAct.text || "";
        const relatedCaseLaws = bareAct.related_case_laws || [];
        
        // Clean text
        const cleanLines = rawText
          .split(/[.\n]/)
          .map(l => l.trim())
          .filter(l => l.length > 20 && !/^(Skip|Search|Login|Menu|Toggle|Free|Premium|Print|Download|Pricing)/i.test(l));
        const cleanedText = cleanLines.join(". ").trim();
        
        return (
          <div key={idx} className="result-item-row bare-act-with-cases">
            <div className="result-item-header">
              <span className="result-item-number">{idx + 1}.</span>
              {url ? (
                <a
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="result-item-title-link"
                  title="Opens in a new tab."
                >
                  {title}
                </a>
              ) : (
                <span className="result-item-title">{title}</span>
              )}
            </div>
            {cleanedText && <p className="result-item-full-text">{cleanedText}</p>}
            {url && (
              <a
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                className="result-item-source-link"
                title="Opens in a new tab."
              >
                View original source
              </a>
            )}
            
            {/* Related Case Laws */}
            {relatedCaseLaws.length > 0 && (
              <div className="related-case-laws">
                <h5 className="related-case-laws-heading">Relevant Case Laws:</h5>
                {relatedCaseLaws.map((caseLaw, clIdx) => {
                  const caseTitle = caseLaw.title || caseLaw.case_name || "Unknown Case";
                  const caseUrl = caseLaw.url || caseLaw.source_url || "";
                  const caseText = caseLaw.text || "";
                  const caseCleanLines = caseText
                    .split(/[.\n]/)
                    .map(l => l.trim())
                    .filter(l => l.length > 20 && !/^(Skip|Search|Login|Menu|Toggle|Free|Premium|Print|Download|Pricing)/i.test(l));
                  const caseCleanedText = caseCleanLines.join(". ").trim();
                  
                  return (
                    <div key={clIdx} className="related-case-law-item">
                      <div className="case-law-header">
                        {caseUrl ? (
                          <a
                            href={caseUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="case-law-title-link"
                            title="Opens in a new tab."
                          >
                            {caseTitle}
                          </a>
                        ) : (
                          <span className="case-law-title">{caseTitle}</span>
                        )}
                      </div>
                      {caseCleanedText && <p className="case-law-text">{caseCleanedText}</p>}
                      {caseUrl && (
                        <a
                          href={caseUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="case-law-source-link"
                          title="Opens in a new tab."
                        >
                          View judgment
                        </a>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        );
      };

      // Helper: render a grouped box (one for bare acts, one for case laws)
      const renderGroupBox = (heading, items) => {
        if (!items || items.length === 0) return null;
        return (
          <div className="results-group-box">
            <h4 className="results-group-heading">{heading}</h4>
            <div className="results-group-items">
              {items.map((item, idx) => renderResultItem(item, idx))}
            </div>
          </div>
        );
      };

      // ---------- search_results / lookup_results (conversational, like ChatGPT) ----------
      if (responseType === "search_results" || responseType === "lookup_results") {
        return (
          <div className="message-final-opinion message-search-results conversational-response">
            {/* High-level summary always comes first */}
            {opinion && (
              <div className="conversational-summary-block">
                {opinion.split("\n").filter(l => l.trim()).map((para, i) => (
                  <p key={i} className="conversational-summary-para">{para}</p>
                ))}
              </div>
            )}

            {/* Progress display */}
            {messageProgress && (
              <ProgressDisplay 
                progress={messageProgress} 
                expandedGroups={expandedGroups}
                setExpandedGroups={setExpandedGroups}
              />
            )}
            {/* Bare Acts with nested Case Laws */}
            {bareActs.length > 0 ? (
              <div className="results-group-box">
                <h4 className="results-group-heading">Relevant Bare Acts</h4>
                <div className="results-group-items">
                  {bareActs.map((bareAct, idx) => renderBareActWithCaseLaws(bareAct, idx))}
                </div>
              </div>
            ) : (
              caseLaws.length > 0 && renderGroupBox("Supreme Court Judgments", caseLaws)
            )}
            
            {bareActs.length === 0 && caseLaws.length === 0 && (
              <p className="search-empty">No results were found. Try refining your query with more specific legal terms.</p>
            )}
          </div>
        );
      }

      // ---------- legal_opinion: no header/title, no standalone download button ----------
      if (responseType === "legal_opinion") {
        const { bareMap, caseMap } = buildCitationMaps(bareActs, caseLaws);
        return (
          <div className="message-final-opinion">
            {opinion && (
              <div className="opinion-text">
                {parseOpinionWithQuotes(opinion).map((seg, idx) =>
                  seg.type === "normal" ? (
                    <span key={idx} className="opinion-text-normal">
                      {seg.text.split("\n").map((line, i) =>
                        /^Dispute \d+:/.test(line) ? (
                          <span key={i}>{i > 0 ? "\n" : ""}<span className="opinion-dispute-heading">{line}</span></span>
                        ) : (
                          <span key={i}>{i > 0 ? "\n" : ""}{linkifyOpinionSegment(line, bareMap, caseMap)}</span>
                        )
                      )}
                    </span>
                  ) : (
                    <span key={idx} className="opinion-quote">{seg.text}</span>
                  )
                )}
              </div>
            )}
            {messageProgress && (
              <ProgressDisplay 
                progress={messageProgress} 
                expandedGroups={expandedGroups}
                setExpandedGroups={setExpandedGroups}
              />
            )}
          </div>
        );
      }

      // ---------- other types (generic_chat, etc.): no "Legal Opinion" header, no PDF button ----------
      return (
        <div className="message-final-opinion message-search-results conversational-response">
          {opinion && (
            <div className="conversational-summary-block">
              {opinion.split("\n").filter(l => l.trim()).map((para, i) => (
                <p key={i} className="conversational-summary-para">{para}</p>
              ))}
            </div>
          )}
          {messageProgress && (
            <ProgressDisplay 
              progress={messageProgress} 
              expandedGroups={expandedGroups}
              setExpandedGroups={setExpandedGroups}
            />
          )}
          {bareActs.length > 0 ? (
            <div className="results-group-box">
              <h4 className="results-group-heading">Relevant Bare Acts</h4>
              <div className="results-group-items">
                {bareActs.map((bareAct, idx) => renderBareActWithCaseLaws(bareAct, idx))}
              </div>
            </div>
          ) : (
            caseLaws.length > 0 && renderGroupBox("Relevant Case Laws", caseLaws)
          )}
          {bareActs.length === 0 && caseLaws.length === 0 && opinion && (
            <p className="search-empty">No supporting materials were retrieved for this query.</p>
          )}
        </div>
      );
    }
    if (content.type === "results" && content.parts) {
      return (
        <div className="message-results">
          {content.parts.map((part) => {
            if (part.type === "explanation") {
              return (
                <div key={`explanation-${part.text?.slice(0, 30)}`} className="message-explanation">
                  <p>{part.text}</p>
                </div>
              );
            }
            return (
              <div key={`part-${part.type}-${part.title}`} className="message-result-block">
                <h3 className="result-block-title">{part.title}</h3>
                {part.items?.map((item) => (
                  <div
                    key={`item-${part.type}-${(item.text || item.content || item.title || "").slice(0, 50)}`}
                    className={`result-item result-item--${part.type}`}
                  >
                    {(item.act_name || item.title) && (
                      <div className="result-item-source">
                        {item.act_name || item.title}
                      </div>
                    )}
                    <pre>{item.text || item.content || JSON.stringify(item, null, 2)}</pre>
                    {item.url && (
                      <a
                        href={item.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="result-link"
                        title="Opens in a new tab."
                      >
                        View source
                      </a>
                    )}
                  </div>
                ))}
              </div>
            );
          })}
        </div>
      );
    }
    if (content.text != null) {
      return <p className="message-text">{content.text}</p>;
    }
    return null;
  };


  // -------------------------
  // Bare act file download (fetch + blob so we can show errors)
  // -------------------------
  const handleBareActDownload = async (e, name) => {
    e.preventDefault();
    const url = `${API_BASE}/bareacts/download?name=${encodeURIComponent(name)}`;
    try {
      const res = await fetch(url);
      if (!res.ok) {
        alert(`Download failed (${res.status}). Make sure the API server is running at ${API_BASE}.`);
        return;
      }
      const blob = await res.blob();
      const blobUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = blobUrl;
      a.download = name || "download";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(blobUrl);
    } catch (err) {
      console.error("Bare act download error:", err);
      alert(`Download failed. Is the API server running at ${API_BASE}?`);
    }
  };

  // -------------------------
  // Download PDF handler (matches on-screen: quotes in grey box, dispute headings in blue)
  // -------------------------
  const handleDownloadPdf = (opinionOverride) => {
    const raw = (opinionOverride != null ? opinionOverride : opinionText) || "";
    const trimmedOpinion = raw.trim();
    if (!trimmedOpinion) {
      alert("No final opinion available to download yet.");
      return;
    }

    // PDF-only: remove %%%% lines and everything from the separator onwards (no follow-up Q or "If you'd like..." in PDF)
    let lines = trimmedOpinion.split("\n").filter((line) => !/^%+\s*$/.test(line.trim()));
    let endIndex = lines.length;
    for (let i = 0; i < lines.length; i++) {
      if (/^─+$/.test(lines[i].trim())) {
        endIndex = i;
        break;
      }
    }
    const pdfOpinion = lines.slice(0, endIndex).join("\n").trim();
    if (!pdfOpinion) {
      alert("No opinion content left for PDF after removing follow-up section.");
      return;
    }

    const doc = new jsPDF({ unit: "pt", format: "a4" });
    const marginLeft = 40;
    const pageWidth = doc.internal.pageSize.getWidth();
    const maxWidth = pageWidth - marginLeft * 2; // 515 for A4
    const lineHeight = 14;
    const quoteMargin = maxWidth * 0.05;
    const quoteWidth = maxWidth * 0.9;
    const quoteLeft = marginLeft + quoteMargin;
    const greyBg = [229, 229, 229]; // #e5e5e5
    const blueHeading = [26, 115, 232]; // #1a73e8

    let y = 40;

    // Title
    doc.setFont("Helvetica", "bold");
    doc.setFontSize(16);
    doc.text("Legal Opinion", marginLeft, y);
    y += 28;

    doc.setFont("Helvetica", "normal");
    doc.setFontSize(11);
    doc.setTextColor(0, 0, 0);

    const segments = parseOpinionWithQuotes(pdfOpinion);

    function maybeNewPage(needed) {
      if (y + needed > 820) {
        doc.addPage();
        y = 40;
      }
    }

    for (const seg of segments) {
      if (seg.type === "normal") {
        const normalLines = seg.text.split("\n");
        for (const line of normalLines) {
          const isDisputeHeading = /^Dispute \d+:/.test(line);
          const wrapped = doc.splitTextToSize(line, maxWidth);
          maybeNewPage(wrapped.length * lineHeight);
          if (isDisputeHeading) {
            doc.setFont("Helvetica", "bold");
            doc.setTextColor(...blueHeading);
          }
          doc.text(wrapped, marginLeft, y);
          y += wrapped.length * lineHeight;
          if (isDisputeHeading) {
            doc.setFont("Helvetica", "normal");
            doc.setTextColor(0, 0, 0);
          }
        }
      } else {
        // quote: grey background, 5% inset, slightly smaller text
        const quoteFontSize = 10.45; // ~95% of 11
        doc.setFontSize(quoteFontSize);
        const quoteWrapped = doc.splitTextToSize(seg.text, quoteWidth);
        const quoteLineHeight = 12;
        const boxPadding = 8;
        const boxHeight = quoteWrapped.length * quoteLineHeight + boxPadding * 2;
        maybeNewPage(boxHeight);

        doc.setFillColor(...greyBg);
        doc.rect(quoteLeft - 2, y, quoteWidth + 4, boxHeight, "F");
        doc.setTextColor(0, 0, 0);
        let quoteY = y + boxPadding + quoteFontSize * 0.4;
        for (const qLine of quoteWrapped) {
          doc.text(qLine, quoteLeft, quoteY);
          quoteY += quoteLineHeight;
        }
        y += boxHeight + 10;

        doc.setFontSize(11);
      }
    }

    doc.save("legal_opinion.pdf");
  };

  // -------------------------
  // MAIN APP UI (no login; open directly)
  // 3 columns: Bare Acts (left), Conversation+Input (middle), Final Output (right)
  // -------------------------
  return (
    <div
      className="app-container"
    >
      {/* Two-pane layout: Left (sidebar) and Right (chat) */}
      <div className="main-content-wrapper">
        {/* LEFT PANE – New chat, Bare Acts, Case Laws, Chat history, Pending indexing (fixed order) */}
        <div
          className="left-column"
          style={{ width: leftColumnWidth, minWidth: leftColumnWidth, maxWidth: leftColumnWidth }}
        >
            <div className="left-column-scroll">
            <button
              type="button"
              onClick={handleNewChat}
              className="chat-history-new-btn"
            >
              ＋ New chat
            </button>
            <details
              className="bare-acts-collapsible"
              open={sidebarExpandedSection === "bare_acts"}
              onClick={(e) => {
                if (e.target.closest("summary")) {
                  e.preventDefault();
                  setSidebarExpandedSection((prev) => (prev === "bare_acts" ? null : "bare_acts"));
                }
              }}
            >
              <summary className="bare-acts-collapsible-summary sidebar-collapsible-summary">
                {bareActs.length ? (
                  <span className="bare-acts-count">Bare Acts Library ({bareActs.length})</span>
                ) : (
                  <span>Bare Acts Library</span>
                )}
              </summary>
              {bareActs.length ? (
                <>
                  <input
                    type="text"
                    placeholder="Search bare acts..."
                    value={bareActsFilter}
                    onChange={(e) => setBareActsFilter(e.target.value)}
                    className="sidebar-search-input"
                    aria-label="Filter bare acts"
                  />
                  <div className="sidebar-list-scroll" onMouseDown={(e) => e.stopPropagation()} onPointerDown={(e) => e.stopPropagation()}>
                    {Array.isArray(bareActsLibrary) && bareActsLibrary.length > 0 ? (
                      bareActsLibrary.map((jurisdiction) => {
                        const filteredActs = jurisdiction.acts.filter(
                          (name) =>
                            !bareActsFilter.trim() ||
                            name.toLowerCase().includes(bareActsFilter.trim().toLowerCase()),
                        );
                        if (!filteredActs.length) return null;
                        const isOpen = bareActsJurisdictionOpen === jurisdiction.name;
                        return (
                          <details
                            key={jurisdiction.name}
                            className="bare-acts-collapsible"
                            open={isOpen}
                            onClick={(e) => {
                              if (e.target.closest("summary")) {
                                e.preventDefault();
                                e.stopPropagation();
                                setBareActsJurisdictionOpen((prev) =>
                                  prev === jurisdiction.name ? null : jurisdiction.name,
                                );
                              }
                            }}
                          >
                            <summary className="bare-acts-collapsible-summary sidebar-collapsible-summary">
                              {jurisdiction.name} ({jurisdiction.acts.length})
                            </summary>
                            <ul className="bare-act-list">
                              {filteredActs.map((name, idx) => {
                                const useRelative =
                                  typeof window !== "undefined" &&
                                  (window.location.hostname === "localhost" ||
                                    window.location.hostname === "127.0.0.1" ||
                                    API_BASE === "" ||
                                    API_BASE.startsWith(window.location.origin));
                                const downloadUrl = useRelative
                                  ? `/bareacts/view?name=${encodeURIComponent(name)}`
                                  : `${API_BASE}/bareacts/view?name=${encodeURIComponent(name)}`;
                                return (
                                  <li key={`${jurisdiction.name}-${name}-${idx}`} className="bare-act-list-item">
                                    <a
                                      href={downloadUrl}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="bare-act-download-link"
                                    >
                                      {name}
                                    </a>
                                  </li>
                                );
                              })}
                            </ul>
                          </details>
                        );
                      })
                    ) : (
                      <ul className="bare-act-list">
                        {bareActs
                          .filter(
                            (name) =>
                              !bareActsFilter.trim() ||
                              name.toLowerCase().includes(bareActsFilter.trim().toLowerCase()),
                          )
                          .map((name, idx) => {
                            const useRelative =
                              typeof window !== "undefined" &&
                              (window.location.hostname === "localhost" ||
                                window.location.hostname === "127.0.0.1" ||
                                API_BASE === "" ||
                                API_BASE.startsWith(window.location.origin));
                            const downloadUrl = useRelative
                              ? `/bareacts/view?name=${encodeURIComponent(name)}`
                              : `${API_BASE}/bareacts/view?name=${encodeURIComponent(name)}`;
                            return (
                              <li key={`${name}-${idx}`} className="bare-act-list-item">
                                <a href={downloadUrl} target="_blank" rel="noopener noreferrer" className="bare-act-download-link">
                                  {name}
                                </a>
                              </li>
                            );
                          })}
                      </ul>
                    )}
                  </div>
                </>
              ) : (
                <p className="bare-act-empty-message">No bare acts in vector store yet.</p>
              )}
            </details>

            {/* Case Laws – below Bare Acts, same functionality */}
            <details
              className="case-laws-collapsible"
              open={sidebarExpandedSection === "case_laws"}
              onClick={(e) => {
                if (e.target.closest("summary")) {
                  e.preventDefault();
                  setSidebarExpandedSection((prev) => (prev === "case_laws" ? null : "case_laws"));
                }
              }}
            >
              <summary className="case-laws-collapsible-summary sidebar-collapsible-summary">
                {caseLawsList.length ? (
                  <span className="case-laws-count">Case Laws ({caseLawsList.length})</span>
                ) : (
                  <span>Case Laws</span>
                )}
              </summary>
              {caseLawsList.length ? (
                <>
                  {/* Most cited – single HTML table with top 200 cases */}
                  <div className="sidebar-section-link-row">
                    {(() => {
                      const useRelative =
                        typeof window !== "undefined" &&
                        (window.location.hostname === "localhost" ||
                          window.location.hostname === "127.0.0.1" ||
                          API_BASE === "" ||
                          API_BASE.startsWith(window.location.origin));
                      const mostCitedUrl = useRelative
                        ? "/caselaws/most_cited"
                        : `${API_BASE}/caselaws/most_cited`;
                      return (
                        <a
                          href={mostCitedUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="case-laws-download-link"
                        >
                          Most cited (top 200)
                        </a>
                      );
                    })()}
                  </div>
                  <input
                    type="text"
                    placeholder="Search case laws..."
                    value={caseLawsFilter}
                    onChange={(e) => setCaseLawsFilter(e.target.value)}
                    className="sidebar-search-input"
                    aria-label="Filter case laws"
                  />
                  <div className="sidebar-list-scroll" onMouseDown={(e) => e.stopPropagation()} onPointerDown={(e) => e.stopPropagation()}>
                    {Array.isArray(caseLawsLibrary) && caseLawsLibrary.length > 0 ? (
                      caseLawsLibrary.map((court) => {
                        const filteredCases = court.cases.filter(
                          (name) =>
                            !caseLawsFilter.trim() ||
                            name.toLowerCase().includes(caseLawsFilter.trim().toLowerCase()),
                        );
                        if (!filteredCases.length) return null;
                        const isOpen = caseLawsCourtOpen === court.name;
                        return (
                          <details
                            key={court.name}
                            className="case-laws-collapsible"
                            open={isOpen}
                            onClick={(e) => {
                              if (e.target.closest("summary")) {
                                e.preventDefault();
                                e.stopPropagation();
                                setCaseLawsCourtOpen((prev) =>
                                  prev === court.name ? null : court.name,
                                );
                              }
                            }}
                          >
                            <summary className="case-laws-collapsible-summary sidebar-collapsible-summary">
                              {court.name} ({court.cases.length})
                            </summary>
                            <ul className="case-laws-list">
                              {filteredCases.map((name, idx) => {
                                const useRelative =
                                  typeof window !== "undefined" &&
                                  (window.location.hostname === "localhost" ||
                                    window.location.hostname === "127.0.0.1" ||
                                    API_BASE === "" ||
                                    API_BASE.startsWith(window.location.origin));
                                const downloadUrl = useRelative
                                  ? `/caselaws/view?name=${encodeURIComponent(name)}`
                                  : `${API_BASE}/caselaws/view?name=${encodeURIComponent(name)}`;
                                return (
                                  <li key={`${court.name}-${name}-${idx}`} className="case-laws-list-item">
                                    <a
                                      href={downloadUrl}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="case-laws-download-link"
                                    >
                                      {name}
                                    </a>
                                  </li>
                                );
                              })}
                            </ul>
                          </details>
                        );
                      })
                    ) : (
                      <ul className="case-laws-list">
                        {caseLawsList
                          .filter(
                            (name) =>
                              !caseLawsFilter.trim() ||
                              name.toLowerCase().includes(caseLawsFilter.trim().toLowerCase()),
                          )
                          .map((name, idx) => {
                            const useRelative =
                              typeof window !== "undefined" &&
                              (window.location.hostname === "localhost" ||
                                window.location.hostname === "127.0.0.1" ||
                                API_BASE === "" ||
                                API_BASE.startsWith(window.location.origin));
                            const downloadUrl = useRelative
                              ? `/caselaws/view?name=${encodeURIComponent(name)}`
                              : `${API_BASE}/caselaws/view?name=${encodeURIComponent(name)}`;
                            return (
                              <li key={`${name}-${idx}`} className="case-laws-list-item">
                                <a
                                  href={downloadUrl}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="case-laws-download-link"
                                >
                                  {name}
                                </a>
                              </li>
                            );
                          })}
                      </ul>
                    )}
                  </div>
                </>
              ) : (
                <p className="case-laws-empty-message">No case laws in vector store yet.</p>
              )}
            </details>

            {/* Saved chats – below Case Laws */}
            <div className="chat-history-section">
              <details
                className="chat-history-collapsible"
                open={sidebarExpandedSection === "chat_history"}
                onClick={(e) => {
                  if (e.target.closest("summary")) {
                    e.preventDefault();
                    setSidebarExpandedSection((prev) => (prev === "chat_history" ? null : "chat_history"));
                  }
                }}
              >
                <summary className="chat-history-collapsible-summary sidebar-collapsible-summary">
                  <span>Chat history ({savedChats.length})</span>
                </summary>
                {savedChats.length > 0 ? (
                  <>
                    <input
                      type="text"
                      placeholder="Search chat history..."
                      value={chatHistoryFilter}
                      onChange={(e) => setChatHistoryFilter(e.target.value)}
                      className="sidebar-search-input"
                      aria-label="Filter chat history"
                    />
                    <div className="chat-history-groups" onMouseDown={(e) => e.stopPropagation()} onPointerDown={(e) => e.stopPropagation()}>
                    {chatGroups.map(({ groupLabel, chats }) => (
                      <div key={groupLabel} className="chat-history-group">
                        <div className="chat-history-group-label">{groupLabel}</div>
                        <ul className="chat-history-list">
                          {chats.map((chat) => (
                            <li key={chat.id} className="chat-history-item">
                              {editingChatId == chat.id ? (
                                <div className="chat-history-rename-row">
                                  <input
                                    ref={editInputRef}
                                    type="text"
                                    value={editingTitle}
                                    onChange={(e) => setEditingTitle(e.target.value)}
                                    onKeyDown={(e) => {
                                      if (e.key === "Enter") saveRenameChat();
                                      if (e.key === "Escape") cancelRenameChat();
                                    }}
                                    onBlur={saveRenameChat}
                                    className="chat-history-rename-input"
                                    aria-label="Rename chat"
                                  />
                                </div>
                              ) : (
                                <>
                                  <button
                                    type="button"
                                    onClick={() => handleLoadChat(chat)}
                                    className="chat-history-link"
                                    title={chat.title}
                                  >
                                    {chat.title}
                                  </button>
                                  <div className="chat-history-menu-wrap">
                                    <button
                                      type="button"
                                      onClick={(e) => { e.stopPropagation(); setOpenMenuChatId((id) => (id === chat.id ? null : chat.id)); }}
                                      className="chat-history-menu-btn"
                                      title="Options"
                                      aria-label="Chat options"
                                      aria-expanded={openMenuChatId == chat.id}
                                    >
                                      ⋯
                                    </button>
                                    {openMenuChatId == chat.id && (
                                      <div className="chat-history-dropdown" onClick={(e) => e.stopPropagation()}>
                                        <button type="button" onClick={() => { startRenamingChat(chat); setOpenMenuChatId(null); }} className="chat-history-dropdown-item">
                                          Rename
                                        </button>
                                        <hr className="chat-history-dropdown-divider" />
                                        <button type="button" onClick={() => handleDeleteChat(chat)} className="chat-history-dropdown-item chat-history-dropdown-item--danger">
                                          Delete
                                        </button>
                                      </div>
                                    )}
                                  </div>
                                </>
                              )}
                            </li>
                          ))}
                        </ul>
                      </div>
                    ))}
                    </div>
                  </>
                ) : (
                  <p className="chat-history-empty">No previous chats.</p>
                )}
              </details>
            </div>

            {/* Pending indexing – same row as above three: font, color, gap, one-expanded-at-a-time */}
            <details
              className="pending-indexing-collapsible"
              open={sidebarExpandedSection === "pending_indexing"}
              onClick={(e) => {
                if (e.target.closest("summary")) {
                  e.preventDefault();
                  setSidebarExpandedSection((prev) => (prev === "pending_indexing" ? null : "pending_indexing"));
                }
              }}
            >
              <summary className="pending-indexing-collapsible-summary sidebar-collapsible-summary">
                <span className="pending-indexing-count">Pending indexing ({pendingIndexingCandidates.length})</span>
              </summary>
              <div className="pending-indexing-body" onMouseDown={(e) => e.stopPropagation()} onPointerDown={(e) => e.stopPropagation()}>
                {pendingIndexingCandidates.length > 0 ? (
                  <>
                    <div className="pending-indexing-actions">
                      <button
                        type="button"
                        className="pending-indexing-btn"
                        disabled={indexingRunning || !pendingIndexingCandidates.some((c) => c.selected && !c.already_in_store)}
                        onClick={async () => {
                          const selected = pendingIndexingCandidates.filter((c) => c.selected && !c.already_in_store);
                          if (!selected.length) return;
                          setIndexingRunning(true);
                          const token = localStorage.getItem(AUTH_TOKEN_KEY);
                          const headers = token ? { Authorization: `Bearer ${token}` } : {};
                          const refreshPending = async () => {
                            try {
                              const pendRes = await fetch(`${API_BASE}/indexing/pending`, { headers });
                              const pendData = pendRes.ok ? await pendRes.json().catch(() => ({})) : {};
                              const items = Array.isArray(pendData?.items) ? pendData.items : [];
                              setPendingIndexingCandidates(items.map((c, i) => ({
                                id: `idx-${Date.now()}-${i}`,
                                title: c.title || "",
                                source_url: c.source_url || "",
                                suggested_category: c.suggested_category || "case_law",
                                category: c.suggested_category || "case_law",
                                selected: !c.already_in_store,
                                already_in_store: !!c.already_in_store,
                              })));
                            } catch (_) {}
                          };
                          const pollId = setInterval(refreshPending, 2000);
                          try {
                            const res = await fetch(`${API_BASE}/indexing/run`, {
                              method: "POST",
                              headers: {
                                "Content-Type": "application/json",
                                ...headers,
                              },
                              body: JSON.stringify({
                                items: selected.map((c) => ({
                                  url: (c.source_url || "").trim(),
                                  title: (c.title || "").trim(),
                                  category: c.category || c.suggested_category || "bare_act",
                                })),
                              }),
                            });
                            const data = res.ok ? await res.json().catch(() => ({})) : {};
                            if (res.ok) await refreshPending();
                            if (data.errors && data.errors.length) {
                              setError(data.message || "Some items could not be indexed.");
                            }
                          } catch (e) {
                            setError(e?.message || "Indexing request failed.");
                          } finally {
                            clearInterval(pollId);
                            await refreshPending();
                            setIndexingRunning(false);
                          }
                        }}
                      >
                        {indexingRunning ? "Indexing…" : "Index"}
                      </button>
                      <button
                        type="button"
                        className="pending-indexing-btn pending-indexing-btn-clear"
                        disabled={indexingRunning || pendingIndexingCandidates.length === 0}
                        onClick={() => setShowClearPendingConfirm(true)}
                      >
                        Clear
                      </button>
                    </div>
                    <ul className="pending-indexing-list">
                      {pendingIndexingCandidates.map((c) => (
                        <li key={c.id} className={`pending-indexing-item${c.already_in_store ? " pending-indexing-item--duplicate" : ""}`}>
                          <label className="pending-indexing-row">
                            <input
                              type="checkbox"
                              checked={!!c.selected}
                              disabled={!!c.already_in_store}
                              onChange={() => {
                                if (c.already_in_store) return;
                                setPendingIndexingCandidates((prev) =>
                                  prev.map((x) => (x.id === c.id ? { ...x, selected: !x.selected } : x))
                                );
                              }}
                              className="pending-indexing-checkbox"
                              aria-label={c.already_in_store ? `Already in library: ${c.title}` : `Select ${c.title}`}
                            />
                            <a
                              href={c.source_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className={`pending-indexing-link${c.already_in_store ? " pending-indexing-link--duplicate" : ""}`}
                              title={c.already_in_store ? `${c.title} — Already in library` : c.title}
                            >
                              {c.title.length > 40 ? c.title.slice(0, 40) + "…" : c.title}
                            </a>
                            {c.already_in_store && (
                              <span className="pending-indexing-badge" title="Same act/document already in internal store">Already in library</span>
                            )}
                            <select
                              value={c.category}
                              onChange={(e) => {
                                setPendingIndexingCandidates((prev) =>
                                  prev.map((x) => (x.id === c.id ? { ...x, category: e.target.value } : x))
                                );
                              }}
                              className="pending-indexing-dropdown"
                              aria-label="Category"
                            >
                              <option value="bare_act">Bare act</option>
                              <option value="case_law">Case law</option>
                            </select>
                          </label>
                        </li>
                      ))}
                    </ul>
                  </>
                ) : (
                  <p className="pending-indexing-empty">No documents pending. New candidates appear here after web search.</p>
                )}
              </div>
            </details>

            {/* Case law discovery – documents presented for indexing (persist until Index or Clear; survives refresh/restart) */}
            <details
              className="pending-indexing-collapsible"
              open={sidebarExpandedSection === "case_law_discovery"}
              onClick={(e) => {
                if (e.target.closest("summary")) {
                  e.preventDefault();
                  setSidebarExpandedSection((prev) => (prev === "case_law_discovery" ? null : "case_law_discovery"));
                }
              }}
            >
              <summary className="pending-indexing-collapsible-summary sidebar-collapsible-summary">
                <span className="pending-indexing-count">Case law discovery – Pending ({caseLawDiscoveryPending.length})</span>
              </summary>
              <div className="pending-indexing-body" onMouseDown={(e) => e.stopPropagation()} onPointerDown={(e) => e.stopPropagation()}>
                {caseLawDiscoveryPending.length > 0 ? (
                  <>
                    <div className="pending-indexing-actions">
                      <button
                        type="button"
                        className="pending-indexing-btn"
                        disabled={caseLawDiscoveryRunning || !caseLawDiscoveryPending.some((c) => c.selected && !c.already_in_store)}
                        onClick={async () => {
                          const selected = caseLawDiscoveryPending.filter((c) => c.selected && !c.already_in_store);
                          if (!selected.length) return;
                          setCaseLawDiscoveryRunning(true);
                          const token = localStorage.getItem(AUTH_TOKEN_KEY);
                          const headers = token ? { Authorization: `Bearer ${token}` } : {};
                          const refreshPending = async () => {
                            try {
                              const pendRes = await fetch(`${API_BASE}/case-law-discovery/pending`, { headers });
                              const pendData = pendRes.ok ? await pendRes.json().catch(() => ({})) : {};
                              const items = Array.isArray(pendData?.items) ? pendData.items : [];
                              setCaseLawDiscoveryPending(items.map((c, i) => ({
                                id: `cld-${Date.now()}-${i}`,
                                title: c.title || "",
                                source_url: c.source_url || "",
                                suggested_category: c.suggested_category || "case_law",
                                category: c.suggested_category || "case_law",
                                selected: !c.already_in_store,
                                already_in_store: !!c.already_in_store,
                                signature: c.signature,
                                act_name: c.act_name,
                                summary: c.summary,
                              })));
                            } catch (_) {}
                          };
                          const pollId = setInterval(refreshPending, 2000);
                          try {
                            const res = await fetch(`${API_BASE}/case-law-discovery/confirm-index`, {
                              method: "POST",
                              headers: { "Content-Type": "application/json", ...headers },
                              body: JSON.stringify({
                                items: selected.map((c) => ({
                                  source_url: (c.source_url || "").trim(),
                                  title: (c.title || "").trim(),
                                  suggested_category: c.category || c.suggested_category || "case_law",
                                  act_name: (c.act_name || "").trim() || undefined,
                                  signature: (c.signature || "").trim() || undefined,
                                  summary: (c.summary || "").trim() || undefined,
                                })),
                              }),
                            });
                            const data = res.ok ? await res.json().catch(() => ({})) : {};
                            if (res.ok) await refreshPending();
                            if (data.errors && data.errors.length) setError(data.message || "Some items could not be indexed.");
                          } catch (e) {
                            setError(e?.message || "Indexing request failed.");
                          } finally {
                            clearInterval(pollId);
                            await refreshPending();
                            setCaseLawDiscoveryRunning(false);
                          }
                        }}
                      >
                        {caseLawDiscoveryRunning ? "Indexing…" : "Index"}
                      </button>
                      <button
                        type="button"
                        className="pending-indexing-btn pending-indexing-btn-clear"
                        disabled={caseLawDiscoveryRunning || caseLawDiscoveryPending.length === 0}
                        onClick={() => setShowClearCaseLawDiscoveryConfirm(true)}
                      >
                        Clear
                      </button>
                    </div>
                    <ul className="pending-indexing-list">
                      {caseLawDiscoveryPending.map((c) => (
                        <li key={c.id} className={`pending-indexing-item${c.already_in_store ? " pending-indexing-item--duplicate" : ""}`}>
                          <label className="pending-indexing-row">
                            <input
                              type="checkbox"
                              checked={!!c.selected}
                              disabled={!!c.already_in_store}
                              onChange={() => {
                                if (c.already_in_store) return;
                                setCaseLawDiscoveryPending((prev) =>
                                  prev.map((x) => (x.id === c.id ? { ...x, selected: !x.selected } : x))
                                );
                              }}
                              className="pending-indexing-checkbox"
                              aria-label={c.already_in_store ? `Already in library: ${c.title}` : `Select ${c.title}`}
                            />
                            <a
                              href={c.source_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className={`pending-indexing-link${c.already_in_store ? " pending-indexing-link--duplicate" : ""}`}
                              title={c.already_in_store ? `${c.title} — Already in library` : c.title}
                            >
                              {c.title.length > 40 ? c.title.slice(0, 40) + "…" : c.title}
                            </a>
                            {c.already_in_store && (
                              <span className="pending-indexing-badge" title="Same document already in internal store">Already in library</span>
                            )}
                            <select
                              value={c.category}
                              onChange={(e) =>
                                setCaseLawDiscoveryPending((prev) =>
                                  prev.map((x) => (x.id === c.id ? { ...x, category: e.target.value } : x))
                                )
                              }
                              className="pending-indexing-dropdown"
                              aria-label="Category"
                            >
                              <option value="bare_act">Bare act</option>
                              <option value="case_law">Case law</option>
                            </select>
                          </label>
                        </li>
                      ))}
                    </ul>
                  </>
                ) : (
                  <p className="pending-indexing-empty">No documents. Run case law discovery to add candidates for indexing.</p>
                )}
              </div>
            </details>
            </div>
        </div>
        <div
          className="left-column-resizer"
          onMouseDown={handleSidebarResizeMouseDown}
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize sidebar"
        />

        {/* RIGHT PANE – Title at top, then chat area */}
        <div className={`right-content-wrapper${messages.some((m) => m.role === "user") ? " chat-mode" : ""}`}>
          {/* Title at top left of right pane */}
          <div className="right-pane-header">
            <h1 className="main-title">
              ⚖ Nyaymalaw
            </h1>
            <div className="header-user">
              <span className="header-email" title={currentUser || "Guest"}>{currentUser || "Guest"}</span>
              <button type="button" onClick={handleLogout} className="header-logout-btn">
                New chat
              </button>
            </div>
          </div>
            {/* Chat area: center stage (no messages) or conversation + input + disclaimer */}
            {!messages.some((m) => m.role === "user") ? (
              /* ChatGPT-style: plain message + single centered text box until first send */
              <div className="chat-center-stage">
                <div className="chat-center-message">
                  <h2>How can I help you today?</h2>
                  <p>
                    I'll gather the facts of your legal matter through a few questions, then research
                    relevant bare acts and case laws for you.
                  </p>
                </div>
                  <div className="chat-center-input-wrapper">
                    <div className="chat-mode-toggle">
                      <button
                        type="button"
                        className={`chat-mode-button ${chatMode === "legal_opinion" ? "chat-mode-button--active" : ""}`}
                        onClick={() => setChatMode("legal_opinion")}
                        disabled={loading}
                      >
                        Legal opinion
                      </button>
                      <button
                        type="button"
                        className={`chat-mode-button ${chatMode === "legal_research" ? "chat-mode-button--active" : ""}`}
                        onClick={() => setChatMode("legal_research")}
                        disabled={loading}
                      >
                        Legal research
                      </button>
                      <button
                        type="button"
                        className={`chat-mode-button ${chatMode === "general" ? "chat-mode-button--active" : ""}`}
                        onClick={() => setChatMode("general")}
                        disabled={loading}
                      >
                        General
                      </button>
                    </div>
                    <ChatComposer
                      loading={loading}
                      placeholder="Describe your case or ask a question"
                      onSubmit={handleSubmit}
                      resetSignal={composerResetSignal}
                      showDisclaimer={bottomExpandedSection == null}
                    />
                  </div>
              </div>
            ) : (
              <>
                <div className="conversation-card">
                  <div
                    ref={messagesContainerRef}
                    className="messages-container"
                    tabIndex={0}
                  >
                    {messages.map((msg, i) => (
                  <div
                    key={`msg-${i}`}
                    className={`message message--${msg.role}`}
                  >
                    <div className="message-avatar">
                      {msg.role === "user" ? (
                        <span className="avatar-user">
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            viewBox="0 0 24 24"
                            fill="currentColor"
                          >
                            <path
                              fillRule="evenodd"
                              d="M7.5 6a4.5 4.5 0 119 0 4.5 4.5 0 01-9 0zM3.751 20.105a8.25 8.25 0 0116.498 0 .75.75 0 01-.437.695A18.683 18.683 0 0112 22.5c-2.786 0-5.433-.608-7.812-1.7a.75.75 0 01-.437-.695z"
                              clipRule="evenodd"
                            />
                          </svg>
                        </span>
                      ) : (
                        <span className="avatar-ai">⚖</span>
                      )}
                    </div>
                    <div className="message-content">
                      {msg.role === "user" ? (
                        <div className="message-bubble message-bubble--user">
                          {editingMessageIndex === i ? (
                            <div className="message-edit-inline">
                              <textarea
                                ref={editMessageInputRef}
                                className="message-edit-textarea"
                                value={editDraft}
                                onChange={(e) => setEditDraft(e.target.value)}
                                rows={Math.min(40, Math.max(6, (editDraft.match(/\n/g) || []).length + 2))}
                              />
                              <div className="message-edit-actions">
                                <button type="button" className="message-edit-btn message-edit-btn-save" onClick={saveEditUserMessage}>
                                  Save
                                </button>
                                <button type="button" className="message-edit-btn message-edit-btn-cancel" onClick={cancelEditUserMessage}>
                                  Cancel
                                </button>
                              </div>
                            </div>
                          ) : (
                            <div className="message-bubble-inner">
                              <div className="message-bubble-text message-bubble-text--preserve" title="User message">
                                {(() => {
                                  const raw = typeof msg.content === "string" ? msg.content : String(msg.content ?? "");
                                  return raw.replace(/\n+\.\.\.\s*$/, "").replace(/\n+$/, "");
                                })()}
                              </div>
                              <div className="message-bubble-actions">
                                <div className="message-action-menu-wrap">
                                  <button
                                    type="button"
                                    className="message-action-dots-btn"
                                    onClick={() => setActionMenuOpenIndex(actionMenuOpenIndex === i ? null : i)}
                                    title="Actions"
                                    aria-label="Actions"
                                    aria-expanded={actionMenuOpenIndex === i}
                                  >
                                    <span className="message-action-dots">⋯</span>
                                  </button>
                                  {actionMenuOpenIndex === i && (
                                    <div className="message-action-dropdown" role="menu">
                                      <button type="button" className="message-action-dropdown-item" role="menuitem" onClick={() => { setActionMenuOpenIndex(null); startEditUserMessage(i); }}>
                                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden><path d="M2.695 14.763l-1.262 3.154a.5.5 0 00.65.65l3.155-1.262a4 4 0 001.343-.885L17.5 5.5a2.121 2.121 0 00-3-3L3.58 13.42a4 4 0 00-.885 1.343z" /></svg>
                                        <span>Edit</span>
                                      </button>
                                      <button type="button" className="message-action-dropdown-item" role="menuitem" onClick={() => handleCopyMessage(msg, i)}>
                                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden><path d="M7 3.5A1.5 1.5 0 018.5 2h3.879a1.5 1.5 0 011.06.44l3.122 3.12A1.5 1.5 0 0117 6.622V12.5a1.5 1.5 0 01-1.5 1.5h-1v-3.379a3 3 0 00-.879-2.121L10.5 5.379A3 3 0 008.379 4.5H7v-1z" /><path d="M4.5 6A1.5 1.5 0 003 7.5v9A1.5 1.5 0 004.5 18h7a1.5 1.5 0 001.5-1.5v-5.879a1.5 1.5 0 00-.44-1.06L9.44 6.439A1.5 1.5 0 008.379 6H4.5z" /></svg>
                                        <span>{copyJustDoneIndex === i ? "Copied" : "Copy"}</span>
                                      </button>
                                    </div>
                                  )}
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="message-bubble message-bubble--assistant">
                          <div className="message-bubble-inner">
                            {renderAssistantContent(msg.content)}
                            <div className="message-bubble-actions">
                              <div className="message-action-menu-wrap">
                                <button
                                  type="button"
                                  className="message-action-dots-btn"
                                  onClick={() => setActionMenuOpenIndex(actionMenuOpenIndex === i ? null : i)}
                                  title="Actions"
                                  aria-label="Actions"
                                  aria-expanded={actionMenuOpenIndex === i}
                                >
                                  <span className="message-action-dots">⋯</span>
                                </button>
                                {actionMenuOpenIndex === i && (
                                  <div className="message-action-dropdown" role="menu">
                                    <button type="button" className="message-action-dropdown-item" role="menuitem" onClick={() => { setActionMenuOpenIndex(null); handleCopyMessage(msg, i); }}>
                                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden><path d="M7 3.5A1.5 1.5 0 018.5 2h3.879a1.5 1.5 0 011.06.44l3.122 3.12A1.5 1.5 0 0117 6.622V12.5a1.5 1.5 0 01-1.5 1.5h-1v-3.379a3 3 0 00-.879-2.121L10.5 5.379A3 3 0 008.379 4.5H7v-1z" /><path d="M4.5 6A1.5 1.5 0 003 7.5v9A1.5 1.5 0 004.5 18h7a1.5 1.5 0 001.5-1.5v-5.879a1.5 1.5 0 00-.44-1.06L9.44 6.439A1.5 1.5 0 008.379 6H4.5z" /></svg>
                                      <span>{copyJustDoneIndex === i ? "Copied" : "Copy"}</span>
                                    </button>
                                    {msg.content?.type === "final_opinion" && (msg.content?.opinionText || "").trim() && (
                                      <button type="button" className="message-action-dropdown-item" role="menuitem" onClick={() => { setActionMenuOpenIndex(null); handleDownloadPdf(msg.content?.opinionText); }}>
                                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden><path d="M10.75 2.75a.75.75 0 00-1.5 0v8.614L6.295 8.235a.75.75 0 10-1.09 1.03l4.25 4.5a.75.75 0 001.09 0l4.25-4.5a.75.75 0 00-1.09-1.03l-2.955 3.129V2.75z" /><path d="M3.5 12.75a.75.75 0 00-1.5 0v2.5A2.75 2.75 0 004.75 18h10.5A2.75 2.75 0 0018 15.25v-2.5a.75.75 0 00-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5z" /></svg>
                                        <span>Download</span>
                                      </button>
                                    )}
                                  </div>
                                )}
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {loading && (
                  <div className="message message--assistant">
                    <div className="message-avatar">
                      <span className="avatar-ai">⚖</span>
                    </div>
                    <div className="message-content">
                      {/* Step timeline — shows while no tokens yet */}
                      {streamingSteps.length > 0 && !streamingToken && (
                        <div className="streaming-steps">
                          {streamingSteps.map((s, i) => (
                            <div key={i} className={`streaming-step ${s.done ? "streaming-step--done" : "streaming-step--active"}`}>
                              <span className="streaming-step-icon">{s.icon || "•"}</span>
                              <span className="streaming-step-msg">{s.message}</span>
                              {!s.done && i === streamingSteps.length - 1 && (
                                <span className="streaming-step-pulse" />
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                      {/* Streaming token text — shows as LLM generates */}
                      {streamingToken && (
                        <div className="message-bubble message-bubble--assistant streaming-response">
                          {/* Compact step summary above streaming text */}
                          {streamingSteps.length > 0 && (
                            <div className="streaming-steps streaming-steps--compact">
                              {streamingSteps.filter(s => s.done || streamingSteps.indexOf(s) === streamingSteps.length - 1).slice(-4).map((s, i) => (
                                <span key={i} className="streaming-step-chip">{s.icon} {s.message}</span>
                              ))}
                            </div>
                          )}
                          <div className="streaming-text">{streamingToken}<span className="streaming-cursor">▋</span></div>
                        </div>
                      )}
                      {/* Fallback typing indicator when nothing is streaming yet */}
                      {streamingSteps.length === 0 && !streamingToken && (
                        <div className="message-bubble message-bubble--assistant typing-indicator">
                          <span className="typing-indicator-text">Nyaymalaw is thinking</span>
                          <span className="typing-dots">
                            <span className="typing-dot" />
                            <span className="typing-dot" />
                            <span className="typing-dot" />
                          </span>
                          {elapsedTime > 0 && (
                            <span className="typing-timer">
                              {Math.floor(elapsedTime / 60)} min {Math.floor(elapsedTime % 60)} sec
                            </span>
                          )}
                        </div>
                      )}
                      {progress && (
                        <ProgressDisplay
                          progress={progress}
                          expandedGroups={expandedGroups}
                          setExpandedGroups={setExpandedGroups}
                        />
                      )}
                    </div>
                  </div>
                )}

                    <div ref={messagesEndRef} />
                  </div>
                </div>

                {/* Fixed bottom input when in chat mode */}
                <div className="chat-input-wrapper">
                  {messages.length === 0 && (
                    <div className="chat-mode-toggle">
                      <button
                        type="button"
                        className={`chat-mode-button ${chatMode === "legal_opinion" ? "chat-mode-button--active" : ""}`}
                        onClick={() => setChatMode("legal_opinion")}
                        disabled={loading}
                      >
                        Legal opinion
                      </button>
                      <button
                        type="button"
                        className={`chat-mode-button ${chatMode === "legal_research" ? "chat-mode-button--active" : ""}`}
                        onClick={() => setChatMode("legal_research")}
                        disabled={loading}
                      >
                        Legal research
                      </button>
                      <button
                        type="button"
                        className={`chat-mode-button ${chatMode === "general" ? "chat-mode-button--active" : ""}`}
                        onClick={() => setChatMode("general")}
                        disabled={loading}
                      >
                        General
                      </button>
                    </div>
                  )}
                  <ChatComposer
                    loading={loading}
                    placeholder={
                      stage === "bare_acts_review"
                        ? "Provide the additional details, or type 'proceed' to continue"
                        : stage === "interview" && currentQuestion
                        ? "Type your details here"
                        : stage === "await_facts"
                        ? "Describe your case facts here"
                        : "Type here to start a new case"
                    }
                    onSubmit={handleSubmit}
                    resetSignal={composerResetSignal}
                    showDisclaimer={bottomExpandedSection == null}
                  />
                </div>

                {/* Confirmation bar */}
                {pendingMaterials?.bare_acts?.length > 0 && (
                  <div className="confirm-index-bar">
                    <button type="button" onClick={confirmAndIndex} disabled={loading} className="confirm-index-btn">
                      ✓ Confirm & Index Bare Acts
                    </button>
                  </div>
                )}
                {pendingMaterials?.case_laws?.length > 0 && (
                  <div className="confirm-index-bar">
                    <button type="button" onClick={confirmAndIndex} disabled={loading} className="confirm-index-btn">
                      ✓ Confirm & Index Case Laws
                    </button>
                  </div>
                )}
                {error && (
                  <div className="chat-error">
                    <span>⚠</span> {error}
                  </div>
                )}
              </>
            )}

            {/* Bottom pane: default minimal; drag resizer up to extend up to 75% of window */}
            <div
              className={`eval-architecture-pane ${evalPaneHeight <= EVAL_PANE_MIN_HEIGHT ? "eval-pane-collapsed" : ""}`}
              aria-label="Eval Results and Architecture"
              style={{
                height: evalPaneHeight,
                minHeight: EVAL_PANE_MIN_HEIGHT,
                maxHeight: `${EVAL_PANE_MAX_VH}vh`,
              }}
            >
              <div
                className="eval-pane-resizer"
                onMouseDown={handleEvalPaneResizeMouseDown}
                role="separator"
                aria-orientation="horizontal"
                aria-label="Resize eval pane"
              />
              <EvalSection />
              <ArchitectureSection />
              <UpdatesTrackerSection />
            </div>
        </div>
      </div>

      {/* Confirmation: discard pending indexing documents */}
      {showClearPendingConfirm && (
        <div className="clear-pending-overlay" onClick={() => setShowClearPendingConfirm(false)}>
          <div className="clear-pending-dialog" onClick={(e) => e.stopPropagation()}>
            <p className="clear-pending-message">Are you sure you want to discard these documents?</p>
            <div className="clear-pending-actions">
              <button type="button" className="clear-pending-btn clear-pending-btn-no" onClick={() => setShowClearPendingConfirm(false)}>
                No
              </button>
              <button type="button" className="clear-pending-btn clear-pending-btn-yes" onClick={async () => {
                const token = localStorage.getItem(AUTH_TOKEN_KEY);
                try {
                  await fetch(`${API_BASE}/indexing/pending`, {
                    method: "DELETE",
                    headers: token ? { Authorization: `Bearer ${token}` } : {},
                  });
                } catch {}
                setPendingIndexingCandidates([]);
                setShowClearPendingConfirm(false);
              }}>
                Yes
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Confirmation: discard case law discovery pending documents */}
      {showClearCaseLawDiscoveryConfirm && (
        <div className="clear-pending-overlay" onClick={() => setShowClearCaseLawDiscoveryConfirm(false)}>
          <div className="clear-pending-dialog" onClick={(e) => e.stopPropagation()}>
            <p className="clear-pending-message">Discard all case law discovery documents presented for indexing?</p>
            <div className="clear-pending-actions">
              <button type="button" className="clear-pending-btn clear-pending-btn-no" onClick={() => setShowClearCaseLawDiscoveryConfirm(false)}>
                No
              </button>
              <button type="button" className="clear-pending-btn clear-pending-btn-yes" onClick={async () => {
                const token = localStorage.getItem(AUTH_TOKEN_KEY);
                try {
                  await fetch(`${API_BASE}/case-law-discovery/pending`, {
                    method: "DELETE",
                    headers: token ? { Authorization: `Bearer ${token}` } : {},
                  });
                } catch {}
                setCaseLawDiscoveryPending([]);
                setShowClearCaseLawDiscoveryConfirm(false);
              }}>
                Yes
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
