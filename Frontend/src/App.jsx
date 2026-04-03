import { memo, useState, useRef, useEffect, useMemo, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import "./App.css";

const AUTH_TOKEN_KEY = "nyaymalaw_auth_token";
const CURRENT_USER_KEY = "nyaymalaw_current_user";
const CURRENT_USER_NAME_KEY = "nyaymalaw_current_user_name";

const RESPONSE_FEEDBACK_TAG_GROUPS = [
  {
    title: "Intake / Reasoning",
    tags: [
      "wrong_followup",
      "repeated_question",
      "premature_proceed",
      "missed_urgency",
      "missed_prior_actions",
      "missed_client_objective",
      "bad_stop_continue_judgment",
      "strong_reasoning",
    ],
  },
  {
    title: "Communication",
    tags: [
      "poor_empathy",
      "poor_clarity",
      "too_verbose",
      "strong_empathy",
    ],
  },
  {
    title: "Grounding / Retrieval",
    tags: [
      "unsupported_legal_reference",
      "poor_grounding",
      "hallucinated_query_expansion",
      "strong_grounding",
    ],
  },
  {
    title: "Performance",
    tags: [
      "too_slow",
    ],
  },
];

/** Strip trailing "..." / ".." / "." from step messages so the UI can show a single animated ellipsis. */
function stripTrailingStepEllipsis(msg) {
  if (typeof msg !== "string") return msg;
  return msg.replace(/\.{1,3}$/, "");
}

function ScoreTable({ title, rows, scoreLabel }) {
  if (!Array.isArray(rows) || rows.length === 0) return null;
  return (
    <div className="retrieval-score-block">
      <div className="retrieval-score-block-title">
        {title}
        {scoreLabel ? <span className="retrieval-score-kind">{scoreLabel}</span> : null}
      </div>
      <ul className="retrieval-score-list">
        {rows.slice(0, 40).map((row, j) => (
          <li key={j}>
            <span className="retrieval-score-name">{row.name || "—"}</span>
            <span className="retrieval-score-val">{row.score != null ? String(row.score) : ""}</span>
          </li>
        ))}
        {rows.length > 40 ? <li className="retrieval-score-more">+{rows.length - 40} more</li> : null}
      </ul>
    </div>
  );
}

function HybridStageDetail({ stage }) {
  if (!stage || typeof stage !== "object") return null;
  return (
    <details className="retrieval-hybrid-stage" open={false}>
      <summary className="retrieval-hybrid-stage-summary">
        <span className="retrieval-stage-label">{stage.stage || "stage"}</span>
        {stage.note ? <span className="retrieval-stage-note">{stage.note}</span> : null}
      </summary>
      {stage.query ? <div className="retrieval-subq">Query: {stage.query}</div> : null}
      <ScoreTable
        title="FAISS (vector)"
        rows={stage.faiss}
        scoreLabel={stage.faiss_score_kind}
      />
      <ScoreTable title="BM25" rows={stage.bm25} scoreLabel={stage.bm25_score_kind} />
      <ScoreTable title="Reranker" rows={stage.rerank} scoreLabel={stage.rerank_score_kind} />
    </details>
  );
}

/** Expandable payload from step_callback.detail (query expansion + bare-act hybrid trace). */
function StreamingRetrievalDetail({ detail }) {
  if (!detail || !detail.kind) return null;
  if (detail.kind === "query_expansion") {
    return (
      <details className="streaming-step-detail">
        <summary className="streaming-step-detail-summary">{"Query expansion — model & final queries"}</summary>
        <div className="streaming-step-detail-body">
          {detail.expansion_input_preview ? (
            <div className="retrieval-facts-preview">
              <strong>Input excerpt</strong>
              <pre>{detail.expansion_input_preview}</pre>
            </div>
          ) : null}
          {detail.model_raw_response ? (
            <div className="retrieval-model-raw">
              <strong>Raw model response</strong>
              <pre>{detail.model_raw_response}</pre>
            </div>
          ) : null}
          {Array.isArray(detail.issues_from_model) && detail.issues_from_model.length > 0 ? (
            <div>
              <strong>Distinct issues (from model)</strong>
              {detail.issues_from_model.map((iss, ii) => (
                <div key={ii} className="retrieval-issue-block">
                  {iss.issue_label ? (
                    <div className="retrieval-issue-label">{iss.issue_label}</div>
                  ) : null}
                  <ul>
                    {(iss.queries || []).map((q, qi) => (
                      <li key={qi}>{q}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          ) : null}
          {Array.isArray(detail.parsed_queries_from_model) && detail.parsed_queries_from_model.length > 0 ? (
            <div>
              <strong>Parsed from model (JSON)</strong>
              <ul>
                {detail.parsed_queries_from_model.map((q, i) => (
                  <li key={i}>{q}</li>
                ))}
              </ul>
            </div>
          ) : null}
          {Array.isArray(detail.final_expanded_queries) && detail.final_expanded_queries.length > 0 ? (
            <div>
              <strong>Final queries used for retrieval</strong>
              <ul>
                {detail.final_expanded_queries.map((q, i) => (
                  <li key={i}>{q}</li>
                ))}
              </ul>
            </div>
          ) : null}
          {Array.isArray(detail.queries_added_after_model) && detail.queries_added_after_model.length > 0 ? (
            <div className="retrieval-heuristic-note">
              <strong>Added after model (focus / fallback)</strong>
              <ul>
                {detail.queries_added_after_model.map((q, i) => (
                  <li key={i}>{q}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      </details>
    );
  }
  if (detail.kind === "bare_act_hybrid_trace") {
    const byDispute = detail.by_dispute || {};
    return (
      <details className="streaming-step-detail streaming-step-detail--wide">
        <summary className="streaming-step-detail-summary">{"Bare acts — FAISS / BM25 / rerank (by dispute)"}</summary>
        <div className="streaming-step-detail-body">
          {Object.entries(byDispute).map(([did, entries]) => (
            <div key={did} className="retrieval-dispute-block">
              <div className="retrieval-dispute-id">Dispute {did}</div>
              {(entries || []).map((entry, ei) => (
                <div key={ei} className="retrieval-query-block">
                  <div className="retrieval-query-line">
                    <strong>Retrieval query</strong>: {entry.retrieval_query || "—"}
                    {entry.path ? <span className="retrieval-path-tag">{entry.path}</span> : null}
                  </div>
                  {(entry.hybrid_stages || []).map((st, si) => (
                    <HybridStageDetail key={si} stage={st} />
                  ))}
                </div>
              ))}
            </div>
          ))}
        </div>
      </details>
    );
  }
  return null;
}

/** Single CTA below Next steps in bare-act guidance; must not be duplicated in summary body. */
const JUDICIAL_PRECEDENT_CTA_OFFER =
  "If you want, I can next look for the closest judicial precedents that support these statutory anchors.";

const JUDICIAL_PRECEDENT_CTA_LINE_RE =
  /^If you want, I can next look for the closest judicial precedents that support these (statutory anchors|disputes)\.?\s*$/i;

function stripJudicialPrecedentCtaLines(text) {
  if (!text || typeof text !== "string") return "";
  return text
    .split("\n")
    .filter((line) => !JUDICIAL_PRECEDENT_CTA_LINE_RE.test(line.trim()))
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

/** Splits opinion text into normal segments and quote blocks (content inside â”Œâ”€â” â”‚ ... â”‚ â””â”€â”˜). Returns [{ type: 'normal'|'quote', text }]. */
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
    if (/^â”Œâ”€+â”\s*$/.test(line)) {
      flushNormal();
      const start = i;
      i += 1;
      const quoteLines = [];
      while (i < lines.length && /^â”‚\s*(.*)$/.test(lines[i])) {
        const m = lines[i].match(/^â”‚\s*(.*)$/);
        quoteLines.push((m[1] || "").trimEnd());
        i += 1;
      }
      if (i < lines.length && /^â””â”€+â”˜\s*$/.test(lines[i])) {
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
    const key = `${normalize(act)}Â§${sec}`;
    if (!bareMap.has(key)) bareMap.set(key, url);
    if (act && sec) {
      const key2 = `${normalize(act)}, Â§ ${sec}`;
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
  // Match [...] that may be bare act (contains Â§) or case law (long hex)
  const bracketRe = /\[([^\]]+)\]/g;
  let lastEnd = 0;
  let m;
  while ((m = bracketRe.exec(text)) !== null) {
    const full = m[0];
    const inner = m[1].trim();
    if (lastEnd < m.index) parts.push(text.slice(lastEnd, m.index));

    const isBare = /Â§/.test(inner);
    const isCaseSig = /^[a-f0-9]{32,}$/i.test(inner);
    let url = null;
    if (isBare) {
      const norm = inner.toLowerCase().replace(/\s+/g, " ").trim();
      url = bareMap.get(norm) ?? bareMap.get(norm.replace(/\s*Â§\s*/, "Â§"));
      if (!url && inner.includes("Â§")) {
        const secMatch = inner.match(/Â§\s*(\d+[A-Za-z]*)/);
        const actPart = inner.replace(/\s*Â§\s*\d+[A-Za-z]*\s*[â€”\-â€“].*$/, "").replace(/^the\s+/i, "").trim();
        const actNorm = actPart.replace(/\s*,\s*(\d{4})\s*$/, " $1").toLowerCase().replace(/\s+/g, " ").trim();
        if (secMatch && actNorm) {
          url = bareMap.get(`${actNorm}Â§${secMatch[1]}`);
          if (!url) url = bareMap.get(actNorm + "Â§" + secMatch[1]);
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
  chatMode,
  onChatModeChange,
  selectedModel,
  onModelChange,
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
      <div className="chat-composer-controls">
        <select
          id="chat-mode-select"
          className="chat-model-select"
          value={chatMode}
          onChange={(e) => onChatModeChange(e.target.value)}
          disabled={loading}
        >
          <option value="legal_opinion">Legal opinion</option>
          <option value="legal_research">Legal research</option>
          <option value="general">General</option>
        </select>
        <select
          id="chat-model-select"
          className="chat-model-select"
          value={selectedModel}
          onChange={(e) => onModelChange(e.target.value)}
          disabled={loading}
        >
          <option value="openai">OpenAI</option>
          <option value="qwen">Qwen</option>
        </select>
      </div>
      {showDisclaimer && (
        <p className="chat-disclaimer">Nyaymalaw AI can make mistakes. Consider checking important information.</p>
      )}
    </>
  );
});

const ResponseFeedbackPanel = memo(function ResponseFeedbackPanel({
  messageId,
  existing,
  onSave,
  onCancel,
}) {
  const [rating, setRating] = useState(existing?.rating || "");
  const [reasonTags, setReasonTags] = useState(existing?.reason_tags || []);
  const [freeText, setFreeText] = useState(existing?.free_text || "");

  useEffect(() => {
    setRating(existing?.rating || "");
    setReasonTags(existing?.reason_tags || []);
    setFreeText(existing?.free_text || "");
  }, [messageId, existing]);

  const toggleTag = useCallback((tag) => {
    setReasonTags((prev) => (
      prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]
    ));
  }, []);

  return (
    <div className="message-feedback-panel">
      <div className="message-feedback-ratings">
        {["good", "okay", "bad"].map((value) => (
          <button
            key={value}
            type="button"
            className={`message-feedback-rating ${rating === value ? "message-feedback-rating--active" : ""}`}
            onClick={() => setRating(value)}
          >
            {value}
          </button>
        ))}
      </div>
      <div className="message-feedback-groups">
        {RESPONSE_FEEDBACK_TAG_GROUPS.map((group) => (
          <div key={group.title} className="message-feedback-group">
            <div className="message-feedback-group-title">{group.title}</div>
            <div className="message-feedback-tags">
              {group.tags.map((tag) => (
                <button
                  key={tag}
                  type="button"
                  className={`message-feedback-tag ${reasonTags.includes(tag) ? "message-feedback-tag--active" : ""}`}
                  onClick={() => toggleTag(tag)}
                >
                  {tag}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
      <textarea
        className="message-feedback-textarea"
        placeholder="What was good or what should have been different?"
        value={freeText}
        onChange={(e) => setFreeText(e.target.value)}
        rows={3}
      />
      <div className="message-feedback-footer">
        {existing?.submittedAt && (
          <span className="message-feedback-status">Saved</span>
        )}
        <button
          type="button"
          className="message-feedback-submit"
          disabled={!rating}
          onClick={() => onSave({ rating, reasonTags, freeText })}
        >
          Save feedback
        </button>
        <button
          type="button"
          className="message-feedback-cancel"
          onClick={onCancel}
        >
          Cancel
        </button>
      </div>
    </div>
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
  const [loading, setLoading] = useState(false); // Retain existing
  const [error, setError] = useState(""); // Retain existing
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const [composerResetSignal, setComposerResetSignal] = useState(0);

  // New interview state from snippet
  const [stage, setStage] = useState("await_facts"); // "await_facts" | "interview" | "done"
  const [facts, setFacts] = useState("");
  const [currentQuestion, setCurrentQuestion] = useState("");
  const [qaHistory, setQaHistory] = useState([]); // [{question, answer}]
  const [analysisStage, setAnalysisStage] = useState("intake");
  const [analysisFactsSummary, setAnalysisFactsSummary] = useState("");
  const [lastResponseType, setLastResponseType] = useState("");
  const [opinionText, setOpinionText] = useState("");
  const [retrieved, setRetrieved] = useState([]);
  const [rawResponse, setRawResponse] = useState("");
  const [showDebug, setShowDebug] = useState(false);
  
  // Progress tracking state
  const [progress, setProgress] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [expandedGroups, setExpandedGroups] = useState({});
  const PROGRESS_LIVE_KEY = "progress_live";

  // Streaming state: step timeline + live token text
  const [streamingSteps, setStreamingSteps] = useState([]); // [{message, icon, done}]
  const [streamingToken, setStreamingToken] = useState("");  // accumulated LLM tokens
  const [openFeedbackMessageId, setOpenFeedbackMessageId] = useState(null);
  const [feedbackStatusByMessageId, setFeedbackStatusByMessageId] = useState({});
  const turnStartedAtRef = useRef(0);

  // Shared SSE handlers for "step" progress and incremental "token" output.
  const handleStep = useCallback((stepPayload) => {
    setStreamingSteps((prev) => {
      const detail = stepPayload.detail ?? null;
      if (prev.length === 0) {
        return [{ message: stepPayload.message, icon: stepPayload.icon || "", done: false, detail }];
      }
      const updated = prev.map((s, i) => (i === prev.length - 1 ? { ...s, done: true } : s));
      return [...updated, { message: stepPayload.message, icon: stepPayload.icon || "", done: false, detail }];
    });
  }, []);

  const handleToken = useCallback((tokenPayload) => {
    setStreamingToken((prev) => prev + (tokenPayload.content || ""));
  }, []);

  // Manual mode selection: "legal_opinion" (default), "legal_research", "general"
  const [chatMode, setChatMode] = useState("legal_opinion");
  const [selectedModel, setSelectedModel] = useState("openai");

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
      // Drag cursor UP (negative deltaY) â†’ increase pane height; DOWN â†’ decrease (so movement follows cursor)
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

  // Saved chats (ChatGPT-style): list of past conversations, persisted to localStorage
  const [savedChats, setSavedChats] = useState([]);
  const hasSavedCurrentChatRef = useRef(false);
  const [editingChatId, setEditingChatId] = useState(null);
  const [editingTitle, setEditingTitle] = useState("");
  const editInputRef = useRef(null);
  const editMessageInputRef = useRef(null);
  const currentChatIdRef = useRef(null);

  // Edit user message (current and old chats)
  const [editingMessageIndex, setEditingMessageIndex] = useState(null);
  const [editDraft, setEditDraft] = useState("");
  const [copyJustDoneIndex, setCopyJustDoneIndex] = useState(null);

  // Left sidebar accordion: chat history stays open by default unless another section is expanded
  const [sidebarExpandedSection, setSidebarExpandedSection] = useState("chat_history");

  // Left pane width (resizable: 50% smaller to 50% larger than base)
  const LEFT_COLUMN_BASE_WIDTH = 280;
  const LEFT_COLUMN_DEFAULT_WIDTH = LEFT_COLUMN_BASE_WIDTH * 0.75; // 25% narrower by default
  const LEFT_COLUMN_MIN_WIDTH = LEFT_COLUMN_BASE_WIDTH * 0.5;
  const LEFT_COLUMN_MAX_WIDTH = LEFT_COLUMN_BASE_WIDTH * 1.5;
  const [leftColumnWidth, setLeftColumnWidth] = useState(LEFT_COLUMN_DEFAULT_WIDTH);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const toggleSidebarSection = (section) => {
    setSidebarExpandedSection((prev) => {
      if (section === "chat_history") return "chat_history";
      return prev === section ? "chat_history" : section;
    });
  };

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

  const makeMessageId = useCallback(
    (prefix = "msg") => `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    [],
  );

  const normalizeAssistantText = useCallback((content) => {
    if (content == null) return "";
    if (typeof content === "string") return content;
    if (content?.type === "question") return content.text || "";
    if (content?.type === "final_opinion") return content.opinionText || "";
    if (content?.text) return content.text;
    if (content?.summary) return content.summary;
    return "";
  }, []);

  const makeAssistantMessage = useCallback((content, feedbackMeta = {}) => ({
    id: makeMessageId("asst"),
    role: "assistant",
    content,
    timestamp: new Date().toISOString(),
    feedbackMeta,
  }), [makeMessageId]);

  const makeUserMessage = useCallback((content) => ({
    id: makeMessageId("user"),
    role: "user",
    content,
    timestamp: new Date().toISOString(),
  }), [makeMessageId]);

  const normalizeLoadedMessages = useCallback((msgs) => (
    Array.isArray(msgs)
      ? msgs.map((m) => ({
          ...m,
          id: m?.id || makeMessageId(m?.role === "assistant" ? "asst" : "user"),
        }))
      : []
  ), [makeMessageId]);

  const getConversationContent = useCallback((msg) => {
    if (typeof msg?.content === "string") return msg.content;
    if (msg?.content?.opinionText != null) return msg.content.opinionText || "";
    return msg?.content?.text ?? msg?.content?.summary ?? "";
  }, []);

  const buildConversationFromMessages = useCallback((msgs) => (
    Array.isArray(msgs)
      ? msgs.map((m) => ({ role: m.role, content: getConversationContent(m) }))
      : []
  ), [getConversationContent]);

  const deriveQaHistoryFromMessages = useCallback((msgs) => {
    if (!Array.isArray(msgs) || msgs.length === 0) return [];
    const out = [];
    for (let i = 1; i < msgs.length - 1; i += 1) {
      const questionMsg = msgs[i];
      const answerMsg = msgs[i + 1];
      if (
        questionMsg?.role === "assistant" &&
        typeof questionMsg?.content === "string" &&
        answerMsg?.role === "user" &&
        typeof answerMsg?.content === "string"
      ) {
        out.push({
          question: questionMsg.content.trim(),
          answer: answerMsg.content.trim(),
        });
        i += 1;
      }
    }
    return out;
  }, []);

  const normalizeWorkflowState = useCallback((workflowState, msgs = []) => {
    const normalizedMessages = Array.isArray(msgs) ? msgs : [];
    const state = workflowState && typeof workflowState === "object" ? workflowState : {};
    const firstUser = normalizedMessages.find((m) => m?.role === "user" && typeof m?.content === "string");
    const lastAssistantQuestion = [...normalizedMessages]
      .reverse()
      .find((m) => m?.role === "assistant" && typeof m?.content === "string");
    const qa =
      Array.isArray(state.qaHistory) && state.qaHistory.length > 0
        ? state.qaHistory
            .map((item) => ({
              question: typeof item?.question === "string" ? item.question.trim() : "",
              answer: typeof item?.answer === "string" ? item.answer.trim() : "",
            }))
            .filter((item) => item.question || item.answer)
        : deriveQaHistoryFromMessages(normalizedMessages);
    const currentQuestion =
      typeof state.currentQuestion === "string" && state.currentQuestion.trim()
        ? state.currentQuestion.trim()
        : "";
    const endsOnAssistantQuestion =
      normalizedMessages.length > 0 &&
      normalizedMessages[normalizedMessages.length - 1]?.role === "assistant" &&
      typeof normalizedMessages[normalizedMessages.length - 1]?.content === "string";
    const inferredStage = currentQuestion || endsOnAssistantQuestion
      ? "interview"
      : normalizedMessages.length > 1
      ? "done"
      : "await_facts";
    const stage = ["await_facts", "interview", "done"].includes(state.stage) ? state.stage : inferredStage;
    const analysisStage = typeof state.analysisStage === "string" && state.analysisStage.trim()
      ? state.analysisStage.trim()
      : (stage === "done" ? "" : "intake");
    const factsSummary = typeof state.factsSummary === "string" ? state.factsSummary.trim() : "";
    const lastResponseType = typeof state.lastResponseType === "string" ? state.lastResponseType.trim() : "";
    return {
      stage,
      facts: typeof state.facts === "string" && state.facts.trim()
        ? state.facts
        : (typeof firstUser?.content === "string" ? firstUser.content : ""),
      currentQuestion: stage === "interview"
        ? (currentQuestion || (typeof lastAssistantQuestion?.content === "string" ? lastAssistantQuestion.content : ""))
        : "",
      qaHistory: qa,
      analysisStage,
      factsSummary,
      lastResponseType,
    };
  }, [deriveQaHistoryFromMessages]);

  const normalizeSavedChat = useCallback((chat) => {
    const normalizedMessages = normalizeLoadedMessages(chat?.messages || []);
    return {
      ...chat,
      messages: normalizedMessages,
      opinionText: chat?.opinionText || "",
      retrieved: Array.isArray(chat?.retrieved) ? chat.retrieved : [],
      workflowState: normalizeWorkflowState(chat?.workflowState, normalizedMessages),
      createdAt: chat?.createdAt || new Date().toISOString(),
    };
  }, [normalizeLoadedMessages, normalizeWorkflowState]);

  const buildWorkflowState = useCallback((overrides = {}) => {
    const nextMessages = Array.isArray(overrides.messages) ? overrides.messages : messages;
    return normalizeWorkflowState(
      {
        stage,
        facts,
        currentQuestion,
        qaHistory,
        analysisStage,
        factsSummary: analysisFactsSummary,
        lastResponseType,
        ...overrides,
      },
      nextMessages,
    );
  }, [analysisFactsSummary, analysisStage, currentQuestion, facts, lastResponseType, messages, normalizeWorkflowState, qaHistory, stage]);

  const resolveModelUsed = useCallback((data, fallback = "") => {
    if (typeof data?.model_used === "string" && data.model_used.trim()) return data.model_used.trim();
    if (fallback) return fallback;
    if (selectedModel === "openai") return "OpenAI";
    return "Qwen";
  }, [selectedModel]);

  const getModelOverridePayload = useCallback(() => (
    selectedModel === "openai" ? "provider:openai" : "provider:qwen"
  ), [selectedModel]);

  const currentTurnLatencyMs = useCallback(() => {
    if (!turnStartedAtRef.current) return null;
    return Math.max(0, Date.now() - turnStartedAtRef.current);
  }, []);

  const looksLikeFreshCaseOpening = useCallback((text) => {
    const value = (text || "").trim().toLowerCase();
    if (value.length < 35) return false;
    if (["proceed", "continue", "ok", "okay", "yes", "no", "thanks", "thank you"].includes(value)) return false;
    if (value.includes("?")) return false;
    const markers = [
      "my husband", "my wife", "my employer", "my landlord", "my tenant",
      "my brother", "my sister", "my father", "my mother", "my neighbour",
      "has been", "have been", "harassing", "threatening", "beats me",
      "assault", "abuse", "evict", "terminated", "fired", "cheated",
      "for last", "for the last", "comes home", "living with", "maintenance",
    ];
    return markers.some((marker) => value.includes(marker));
  }, []);

  const openFeedbackForMessage = useCallback((msg) => {
    const messageId = msg?.id || null;
    if (!messageId) return;
    setOpenFeedbackMessageId(messageId);
  }, []);

  const closeFeedbackPanel = useCallback(() => {
    setOpenFeedbackMessageId(null);
  }, []);

  const submitResponseFeedback = useCallback(async (msg, index, draft) => {
    if (!msg?.id || !draft?.rating) return;
    const token = localStorage.getItem(AUTH_TOKEN_KEY);
    const headers = {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    };
    const previousUser = [...messages]
      .slice(0, index)
      .reverse()
      .find((m) => m.role === "user");
    const payload = {
      chat_id: currentChatIdRef.current != null ? String(currentChatIdRef.current) : "",
      message_id: msg.id,
      rating: draft.rating,
      reason_tags: draft.reasonTags,
      free_text: draft.freeText,
      assistant_text: normalizeAssistantText(msg.content),
      user_message: typeof previousUser?.content === "string" ? previousUser.content : "",
      stage: msg.feedbackMeta?.stage || (msg.content?.type === "final_opinion" ? "analysis" : "intake"),
      response_type: msg.feedbackMeta?.responseType || msg.content?.response_type || (msg.content?.type === "final_opinion" ? "legal_opinion" : "intake_question"),
      model_used: msg.feedbackMeta?.modelUsed || msg.content?.model_used || "",
      latency_ms: msg.feedbackMeta?.latencyMs ?? null,
      metadata: {
        message_index: index,
        tags_version: "v1",
        ...msg.feedbackMeta,
      },
    };
    try {
      const res = await fetch(`${API_BASE}/feedback/response`, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Failed to save feedback");
      }
      setFeedbackStatusByMessageId((prev) => ({
        ...prev,
        [msg.id]: {
          rating: draft.rating,
          reason_tags: draft.reasonTags,
          free_text: draft.freeText,
          submittedAt: new Date().toISOString(),
        },
      }));
      setOpenFeedbackMessageId(null);
    } catch (err) {
      setError(err.message || "Could not save feedback");
    }
  }, [API_BASE, messages, normalizeAssistantText]);

  // Sidebar data: load chat history first, then bare acts, then case laws (sequential, one mount pass).
  useEffect(() => {
    let cancelled = false;

    const loadSidebarData = async () => {
      const token = localStorage.getItem(AUTH_TOKEN_KEY);
      const headers = token ? { Authorization: `Bearer ${token}` } : {};

      try {
        const res = await fetch(`${API_BASE}/chats`, { headers });
        let data = { chats: [] };
        if (res.ok) {
          const text = await res.text();
          try {
            data = text ? JSON.parse(text) : { chats: [] };
          } catch {
            data = { chats: [] };
          }
        }
        if (!cancelled) {
          setSavedChats(Array.isArray(data.chats) ? data.chats.map(normalizeSavedChat) : []);
        }
      } catch {
        if (!cancelled) setSavedChats([]);
      }

      if (cancelled) return;

      try {
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
            if (!cancelled) {
              setBareActsLibrary(normalized);
              setBareActs(normalized.flatMap((j) => j.acts.map((a) => (typeof a === "string" ? a : a.name))));
              setBareActsJurisdictionOpen((prev) => prev ?? normalized[0]?.name ?? null);
            }
          } else {
            const res = await fetch(`${API_BASE}/bareacts/list`);
            if (!cancelled) {
              if (!res.ok) {
                setBareActs(DEFAULT_BARE_ACTS);
              } else {
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
              }
            }
          }
        }
      } catch (err) {
        console.error("Error fetching bare acts:", err);
        if (!cancelled) setBareActs(DEFAULT_BARE_ACTS);
      }

      if (cancelled) return;

      try {
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
            if (!cancelled) {
              setCaseLawsLibrary(normalized);
              setCaseLawsList(normalized.flatMap((c) => c.cases.map((x) => (typeof x === "string" ? x : x.name))));
              setCaseLawsCourtOpen((prev) => prev ?? normalized[0]?.name ?? null);
            }
            return;
          }
        }

        const res = await fetch(`${API_BASE}/caselaws/list`);
        if (!cancelled && res.ok) {
          const data = await res.json().catch(() => ({}));
          setCaseLawsList(Array.isArray(data.cases) ? data.cases : []);
        }
      } catch {
        if (!cancelled) setCaseLawsList([]);
      }
    };

    loadSidebarData();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- mount only; chats first, then libraries
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
        workflowState: buildWorkflowState(),
        createdAt: new Date().toISOString(),
      }),
    }).catch(() => {});
  }, [API_BASE, buildWorkflowState, messages, opinionText, retrieved]);

  // Keep the "current" chat in the list in sync with messages/opinion/retrieved
  useEffect(() => {
    if (currentChatIdRef.current == null || messages.length === 0) return;
    setSavedChats((prev) =>
      prev.map((c) =>
        c.id == currentChatIdRef.current
          ? { ...c, messages, opinionText, retrieved, workflowState: buildWorkflowState() }
          : c
      )
    );
  }, [buildWorkflowState, messages, opinionText, retrieved]);

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
    setAnalysisStage("intake");
    setAnalysisFactsSummary("");
    setLastResponseType("");
    setOpinionText("");
    setRetrieved([]);
    setRawResponse("");
    setError(""); // Reset existing error
    setMessages([]);
  };

  // Save current conversation to savedChats (for sidebar list and persistence)
  const saveCurrentChatToHistory = (msgs, opinion, retr, workflowState) => {
    const firstUser = (msgs || []).find((m) => m.role === "user");
    const title =
      (typeof firstUser?.content === "string" && firstUser.content.trim()) ||
      `Chat ${new Date().toLocaleString()}`;
    const chat = normalizeSavedChat({
      id: Date.now(),
      title: title.length > 50 ? title.slice(0, 50) + "â€¦" : title,
      messages: msgs || [],
      opinionText: opinion || "",
      retrieved: Array.isArray(retr) ? retr : [],
      workflowState: workflowState || buildWorkflowState({ messages: msgs || [] }),
      createdAt: new Date().toISOString(),
    });
    setSavedChats((prev) => [chat, ...prev]);
    hasSavedCurrentChatRef.current = true;
  };

  // Open a saved chat in the chat window
  const handleLoadChat = (chat) => {
    const normalizedChat = normalizeSavedChat(chat);
    const workflowState = normalizedChat.workflowState || {};
    setMessages(normalizedChat.messages || []);
    setOpinionText(normalizedChat.opinionText || "");
    setRetrieved(normalizedChat.retrieved || []);
    setRawResponse("");
    setError("");
    setProgress(null);
    setStreamingSteps([]);
    setStreamingToken("");
    setComposerResetSignal((prev) => prev + 1);
    setStage(workflowState.stage || "done");
    setFacts(workflowState.facts || "");
    setCurrentQuestion(workflowState.currentQuestion || "");
    setQaHistory(Array.isArray(workflowState.qaHistory) ? workflowState.qaHistory : []);
    setAnalysisStage(workflowState.analysisStage || (workflowState.stage === "done" ? "" : "intake"));
    setAnalysisFactsSummary(workflowState.factsSummary || "");
    setLastResponseType(workflowState.lastResponseType || "");
    hasSavedCurrentChatRef.current = true;
    currentChatIdRef.current = normalizedChat.id;
  };

  // New chat: save current if unsaved, then clear
  const handleNewChat = () => {
    if (messages.length > 0 && !hasSavedCurrentChatRef.current) {
      saveCurrentChatToHistory(messages, opinionText, retrieved, buildWorkflowState());
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
        retrieved: chat.retrieved || [], workflowState: chat.workflowState || {}, createdAt: chat.createdAt || new Date().toISOString(),
      }) });
      if (res.ok) {
        const listRes = await fetch(`${API_BASE}/chats`, { headers: token ? { Authorization: `Bearer ${token}` } : {} });
        if (listRes.ok) {
          const data = await listRes.json().catch(() => ({}));
          setSavedChats(Array.isArray(data.chats) ? data.chats.map(normalizeSavedChat) : []);
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
    const token = localStorage.getItem(AUTH_TOKEN_KEY);
    const wasCurrent = currentChatIdRef.current == idToRemove;
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    try {
      const res = await fetch(`${API_BASE}/chats/${encodeURIComponent(idForUrl)}`, { method: "DELETE", headers });
      if (res.ok) {
        const listRes = await fetch(`${API_BASE}/chats`, { headers });
        if (listRes.ok) {
          const listData = await listRes.json().catch(() => ({}));
          setSavedChats(Array.isArray(listData.chats) ? listData.chats.map(normalizeSavedChat) : []);
          if (wasCurrent) handleStartNewCase();
          return;
        }
      }
    } catch (_) {}
    setSavedChats((prev) => prev.filter((c) => String(c.id) !== String(idToRemove)));
    if (wasCurrent) handleStartNewCase();
  };

  const handleQuestionResponse = useCallback((data) => {
    const nextQuestion = (data.next_question || data.message || "Please share one more important detail, or say 'proceed' if you want me to identify the applicable bare act sections.").trim();
    setCurrentQuestion(nextQuestion);
    setStage("interview");
    setAnalysisStage((data.analysis_stage || "").trim() || "intake");
    if (typeof data.facts_summary === "string" && data.facts_summary.trim()) setAnalysisFactsSummary(data.facts_summary.trim());
    setLastResponseType("intake_question");
    setMessages((prev) => [
      ...prev,
      makeAssistantMessage(nextQuestion, {
        stage: "intake",
        responseType: "intake_question",
        modelUsed: resolveModelUsed(data),
        latencyMs: currentTurnLatencyMs(),
      }),
    ]);
    if (Array.isArray(data.retrieved)) setRetrieved(data.retrieved);
  }, [currentTurnLatencyMs, makeAssistantMessage, resolveModelUsed]);

  const handleDoneResponse = useCallback((data) => {
    setStage("done");
    setCurrentQuestion("");
    setAnalysisStage((data.analysis_stage || "").trim());
    if (typeof data.facts_summary === "string") setAnalysisFactsSummary(data.facts_summary.trim());
    setLastResponseType(data.response_type || "legal_opinion");
    const opinion = data.opinion_text || "";
    const retr = Array.isArray(data.retrieved) ? data.retrieved : [];
    setOpinionText(opinion);
    setRetrieved(retr);
    if (data.progress) {
      setProgress(data.progress);
    }
    const newAssistantMsg = makeAssistantMessage({
      type: "final_opinion",
      response_type: data.response_type || "legal_opinion",
      opinionText: opinion,
      bare_acts: Array.isArray(data.bare_acts) ? data.bare_acts : [],
      case_laws: Array.isArray(data.case_laws) ? data.case_laws : [],
      next_steps: Array.isArray(data.next_steps) ? data.next_steps : [],
      next_steps_summary: typeof data.next_steps_summary === "string" ? data.next_steps_summary : "",
      retrieved: retr,
      progress: data.progress || null,
      model_used: data.model_used || null,
    }, {
      stage: "analysis",
      responseType: data.response_type || "legal_opinion",
      modelUsed: resolveModelUsed(data),
      latencyMs: currentTurnLatencyMs(),
    });
    setExpandedGroups((prev) => {
      const next = { ...prev, [`pt_${newAssistantMsg.id}`]: false };
      const groups = (data.progress && data.progress.groups) || [];
      groups.forEach((g) => {
        if (g && g.name) next[g.name] = false;
      });
      return next;
    });
    setMessages((prev) => [...prev, newAssistantMsg]);
  }, [currentTurnLatencyMs, makeAssistantMessage, resolveModelUsed]);

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
    setExpandedGroups((prev) => ({ ...prev, [PROGRESS_LIVE_KEY]: true }));
    setStreamingSteps([]);
    setStreamingToken("");
    setElapsedTime(0); // Reset timer
    setComposerResetSignal((prev) => prev + 1);
    turnStartedAtRef.current = Date.now();

    // Keep exact format user typed (spaces, newlines)
    const userMsg = makeUserMessage(raw);
    const isFirstMessage = messages.length === 0;

    setMessages((prev) => [...prev, userMsg]);

    if (isFirstMessage) {
      const chatId = Date.now();
      const title = raw.trim().length > 50 ? raw.trim().slice(0, 50) + "â€¦" : raw.trim();
      const chat = normalizeSavedChat({
        id: chatId,
        title,
        messages: [userMsg],
        opinionText: "",
        retrieved: [],
        workflowState: {
          stage: "await_facts",
          facts: raw,
          currentQuestion: "",
          qaHistory: [],
        },
        createdAt: new Date().toISOString(),
      });
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
          { text: raw, mode: chatMode, model_override: getModelOverridePayload() },
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
            if (data.status === "question") {
              handleQuestionResponse(data);
            } else if (data.status === "done") {
              handleDoneResponse(data);
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
      const conversation = buildConversationFromMessages(messages);

      try {
        await consumeSSEStream(
          `${API_BASE}/conversation/continue/stream`,
          { conversation, message: raw, mode: chatMode, model_override: getModelOverridePayload(), workflowState: buildWorkflowState() },
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
            if (data.status === "question") {
              handleQuestionResponse(data);
            } else if (data.status === "done") {
              handleDoneResponse(data);
            } else {
              if (data.message) setError(data.message);
            }
          },
          handleStep,
          handleToken,
        );
      } catch (err) {
        console.error("conversation continue stream error:", err);
        setError("Error during processing: " + (err.message || "Network or server error"));
        setRawResponse("Error: " + err.message);
      } finally {
        setLoading(false);
      }
      return;
    }


    // 3) Continue a loaded chat: use stream endpoint for live progress
    if (stage === "done") {
      if (looksLikeFreshCaseOpening(raw)) {
        setStage("await_facts");
        setFacts("");
        setCurrentQuestion("");
        setQaHistory([]);
        setAnalysisStage("intake");
        setAnalysisFactsSummary("");
        setLastResponseType("");
        setOpinionText("");
        setRetrieved([]);
        setRawResponse("");
        setProgress(null);
        setStreamingSteps([]);
        setStreamingToken("");
        try {
          await consumeSSEStream(
            `${API_BASE}/submit_case/stream`,
            { text: raw, mode: chatMode, model_override: getModelOverridePayload() },
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
              if (data.status === "question") {
                setFacts(raw);
                handleQuestionResponse(data);
              } else if (data.status === "done") {
                setFacts(raw);
                handleDoneResponse(data);
              } else if (data.message) {
                setError(data.message);
              }
            },
            handleStep,
            handleToken,
          );
        } catch (err) {
          setError("Error during processing: " + (err.message || "Network or server error"));
        } finally {
          setLoading(false);
        }
        return;
      }

      const conversation = buildConversationFromMessages(messages);
      try {
        await new Promise((r) => setTimeout(r, 0));
        await consumeSSEStream(
          `${API_BASE}/conversation/continue/stream`,
          { conversation, message: raw, mode: chatMode, model_override: getModelOverridePayload(), workflowState: buildWorkflowState() },
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
              handleQuestionResponse(data);
            } else if (data.status === "done") {
              handleDoneResponse(data);
            } else {
              const fallbackContent = assistantContent || data.opinion_text || data.summary || "Processing...";
              setMessages((prev) => [
                ...prev,
                makeAssistantMessage(fallbackContent, {
                  stage: "analysis",
                  responseType: "continuation",
                  modelUsed: "",
                  latencyMs: currentTurnLatencyMs(),
                }),
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
          makeAssistantMessage("Sorry, something went wrong. Please try again.", {
            stage: "system",
            responseType: "error",
          }),
        ]);
      } finally {
        setLoading(false);
      }
      return;
    }
  };

  // -------------------------
  // Progress Display Component â€” single collapsible "Progress tracker" with all steps
  // -------------------------
  const PROGRESS_TRACKER_KEY = "progress_tracker";
  const ProgressDisplay = ({ progress, expandedGroups, setExpandedGroups, expandKey = PROGRESS_TRACKER_KEY, defaultOpen = true }) => {
    if (!progress || !progress.groups || progress.groups.length === 0) return null;

    const isExpanded = expandedGroups[expandKey] !== undefined ? expandedGroups[expandKey] : defaultOpen;
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
              setExpandedGroups((prev) => {
                const cur = prev[expandKey] !== undefined ? prev[expandKey] : defaultOpen;
                return { ...prev, [expandKey]: !cur };
              });
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
                              <li key={i} title={t}>{t.length > 50 ? t.slice(0, 50) + "â€¦" : t}</li>
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
  // Eval Section â€” JSON as tables (see .cursor/rules/eval-json-ui-rendering.md)
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
  const downloadTableExcel = async (headers, rows, filename) => {
    try {
      const XLSX = await import("xlsx");
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
    // Detect metrics object: { metric_name: { mean, n, stdev, ci_95 }, ... } â†’ table with metrics as rows
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
        ...statKeys.map((k) => (stats && stats[k] != null) ? (typeof stats[k] === "number" ? Number(stats[k]).toFixed(4) : String(stats[k])) : "â€”"),
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
                      <td key={k}>{(stats && stats[k] != null) ? (typeof stats[k] === "number" ? Number(stats[k]).toFixed(4) : String(stats[k])) : "â€”"}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      );
    }
    // Detect aggregated-by-key object (e.g. threshold_sweep aggregated): { "0.1": { mean_precision, mean_recall, ... }, ... } â†’ rows = keys, columns = metric names
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
          return v != null ? (typeof v === "number" ? (Number.isInteger(v) ? String(v) : Number(v).toFixed(4)) : String(v)) : "â€”";
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
                        const disp = v != null ? (typeof v === "number" ? (Number.isInteger(v) ? String(v) : Number(v).toFixed(4)) : String(v)) : "â€”";
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
        return v == null ? "" : typeof v === "string" && v.length > 150 ? v.slice(0, 150) + "â€¦" : String(v);
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
                      const s = v == null ? "" : typeof v === "string" && v.length > 150 ? v.slice(0, 150) + "â€¦" : String(v);
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
        {evalLoading && <p className="eval-loading">Loadingâ€¦</p>}
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
        {architectureLoading && <p className="architecture-loading">Loadingâ€¦</p>}
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
      setTimeout(() => setCopyJustDoneIndex(null), 1500);
    }).catch(() => {});
  };

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

  const buildAssistantContent = (data) => { // Removed isConfirmResponse as it's not used
    if (data.response) {
      const parts = [];
      if (data.message) {
        parts.push({ type: "explanation", text: data.message });
      }
      if (data.response.bare_act_sections?.length > 0) {
        parts.push({
          type: "bare",
          title: "ðŸ“˜ Bare Act Sections",
          items: data.response.bare_act_sections,
        });
      }
      if (data.response.case_laws?.length > 0) {
        parts.push({
          type: "case",
          title: "ðŸ“š Case Laws (from database)",
          items: data.response.case_laws,
        });
      }
      if (data.response.internet_case_laws?.length > 0) {
        parts.push({
          type: "internet_case",
          title: "ðŸ“š Indiankanoon Fallback Results",
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

  const renderAssistantContent = (content, options = {}) => {
    const progressExpandKey = options.progressExpandKey ?? PROGRESS_TRACKER_KEY;
    const progressDefaultOpen = options.progressDefaultOpen !== undefined ? options.progressDefaultOpen : false;
    if (content == null) return null;
    const trimDots = (s) => (s || "").replace(/\n+\.\.\.\s*$/, " â€¦").replace(/\n+$/, "");
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
    if (content.type === "final_opinion") {
      const opinion = content.opinionText || "";
      const responseType = content.response_type || "legal_opinion";
      const bareActs = content.bare_acts || [];
      const caseLaws = content.case_laws || [];
      const nextSteps = content.next_steps || [];
      const nextStepsSummaryRaw = typeof content.next_steps_summary === "string" ? content.next_steps_summary : "";
      const messageProgress = content.progress || null;

      const cleanResultText = (rawText, isVerbatim = false) => {
        const value = typeof rawText === "string" ? rawText : "";
        if (!value.trim()) return "";
        if (isVerbatim) return value.trim();
        return value
          .split(/[.\n]/)
          .map((line) => line.trim())
          .filter((line) => line.length > 20 && !/^(Skip|Search|Login|Menu|Toggle|Free|Premium|Print|Download|Pricing)/i.test(line))
          .join('. ')
          .trim();
      };

      const renderSourceLink = (url, label = "View original source") => (
        url ? (
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="result-item-source-link"
            title="Opens in a new tab."
          >
            {label}
          </a>
        ) : null
      );

      const groupItemsByDispute = (items) => {
        const groups = new Map();
        (items || []).forEach((item) => {
          const disputeId = item._dispute_id || "general";
          const disputeLabel = item._dispute_label || `Issue ${groups.size + 1}`;
          const disputeText = item._dispute_text || "";
          const current = groups.get(disputeId) || {
            id: disputeId,
            label: disputeLabel,
            text: disputeText,
            items: [],
            bestScore: -Infinity,
          };
          current.items.push(item);
          current.bestScore = Math.max(current.bestScore, Number(item._sort_score ?? item._rerank_score ?? 0));
          if (!current.text && disputeText) current.text = disputeText;
          if (!current.label && disputeLabel) current.label = disputeLabel;
          groups.set(disputeId, current);
        });
        return Array.from(groups.values())
          .map((group) => ({
            ...group,
            items: [...group.items].sort((a, b) => Number(b._sort_score ?? b._rerank_score ?? 0) - Number(a._sort_score ?? a._rerank_score ?? 0)),
          }))
          .sort((a, b) => b.bestScore - a.bestScore);
      };

      const buildSectionLabel = (section) => {
        const sectionNumber = `${section.section_number || ""}`.trim();
        const sectionTitle = (section.section_title || "").trim();
        if (sectionNumber && sectionTitle) return `Section ${sectionNumber} - ${sectionTitle}`;
        if (sectionNumber) return `Section ${sectionNumber}`;
        if (sectionTitle) return sectionTitle;
        return section.title || "Key section";
      };

      const tryLeadingParagraphRefFromJudgmentText = (rawText) => {
        const t = String(rawText || "").trim();
        if (!t) return "";
        const m = t.match(/^(?:[\[(]?\s*)?(\d{1,4})\s*[\])]?\s*[.):\-–—]\s+\S/);
        if (m) return `Para. ${m[1]}`;
        const m2 = t.match(/^Paragraphs?\s+([\d\s,\-–—]+?)(?:\s*[.:)\]]|\s+—|\s+–|\s+-\s|\n)/i);
        if (m2) return `Paras. ${m2[1].trim()}`;
        const m3 = t.match(/^¶\s*(\d{1,4})\b/);
        if (m3) return `Para. ${m3[1]}`;
        return "";
      };

      const formatJudgmentChunkHeading = (item) => {
        if (!item || typeof item !== "object") return "";
        const n = item.paragraph_num ?? item.paragraph_id ?? item.para_num ?? item.paragraph_number;
        if (n !== undefined && n !== null && String(n).trim() !== "") {
          const s = String(n).trim();
          if (/^para(s)?\.?\s+/i.test(s)) return s.charAt(0).toUpperCase() + s.slice(1);
          return `Para. ${s}`;
        }
        const multi = item.paragraph_numbers ?? item.paragraph_refs;
        if (Array.isArray(multi) && multi.length > 0) {
          const parts = multi.map((x) => String(x).trim()).filter(Boolean);
          if (parts.length) return `Paras. ${parts.join(", ")}`;
        }
        const pl = String(item.paragraph_label || item.section_label || item.citation_anchor || "").trim();
        if (pl) return pl;
        const cite = String(item.citation || item.citation_label || item.neutral_citation || "").trim();
        if (cite) return cite;
        const subTitle = String(item.title || "").trim();
        const caseNm = String(item.case_name || "").trim();
        if (subTitle && caseNm && subTitle !== caseNm) {
          return subTitle.length > 120 ? `${subTitle.slice(0, 117)}…` : subTitle;
        }
        const fromText = tryLeadingParagraphRefFromJudgmentText(item.text || item.full_text || "");
        if (fromText) return fromText;
        return "";
      };

      const mergeLegacyNextStepFields = (item) =>
        [item.what_to_do, item.precautions, item.why_it_helps, item.challenge_to_watch, item.legal_protection]
          .map((x) => String(x || "").trim())
          .filter(Boolean)
          .join(" ")
          .trim();

      const normalizeNextStepBullets = (items) => {
        const out = [];
        const seen = new Set();
        (Array.isArray(items) ? items : []).forEach((item) => {
          if (!item || typeof item !== "object") return;
          const title = String(item.title || "").trim();
          let summary = String(item.summary || "").trim();
          if (!summary) summary = mergeLegacyNextStepFields(item);
          if (!summary && title.length < 4) return;
          if (!title && !summary) return;
          const dedupeKey = `${(title || "step").toLowerCase()}|${(summary || title).slice(0, 100).toLowerCase()}`;
          if (seen.has(dedupeKey)) return;
          seen.add(dedupeKey);
          out.push({
            title: title || "Next step",
            summary: summary || title,
          });
        });
        return out;
      };

      const buildNextStepsBullets = (summaryRaw, items) => {
        const bullets = normalizeNextStepBullets(items);
        if (bullets.length > 0) return bullets;
        const s = typeof summaryRaw === "string" ? summaryRaw.trim() : "";
        if (s) return [{ title: "Recommended actions", summary: s }];
        return [];
      };

      const groupBareActsByDisputeAndAct = (items) => {
        return groupItemsByDispute(items).map((disputeGroup) => {
          const acts = new Map();
          disputeGroup.items.forEach((item) => {
            const actName = item.act_name || item.source || item.title || "Applicable Act";
            const actKey = actName.toLowerCase();
            const current = acts.get(actKey) || {
              id: actKey,
              name: actName,
              url: item.url || item.source_url || "",
              sections: [],
              bestScore: -Infinity,
            };
            current.sections.push(item);
            current.bestScore = Math.max(current.bestScore, Number(item._sort_score ?? item._rerank_score ?? 0));
            if (!current.url && (item.url || item.source_url)) current.url = item.url || item.source_url;
            acts.set(actKey, current);
          });
          const rankedActs = Array.from(acts.values())
            .map((act) => ({
              ...act,
              sections: [...act.sections].sort((a, b) => Number(b._sort_score ?? b._rerank_score ?? 0) - Number(a._sort_score ?? a._rerank_score ?? 0)),
            }))
            .sort((a, b) => b.bestScore - a.bestScore);
          return {
            ...disputeGroup,
            acts: rankedActs,
          };
        });
      };

      const renderResultItem = (item, idx) => {
        const title = item.title || item.act_name || item.source || `Result ${idx + 1}`;
        const url = item.url || item.source_url || "";
        const explanation = item.explanation || "";
        const isVerbatim = Boolean(item.is_verbatim_excerpt);
        const cleanedText = cleanResultText(item.text || "", isVerbatim);
        return (
          <div key={idx} className="result-item-row staged-authority-card">
            <div className="result-item-header">
              <span className="result-item-number">{idx + 1}.</span>
              {url ? (
                <a href={url} target="_blank" rel="noopener noreferrer" className="result-item-title-link" title="Opens in a new tab.">{title}</a>
              ) : (
                <span className="result-item-title">{title}</span>
              )}
            </div>
            {explanation && (
              <div className="authority-summary-block">
                <div className="authority-summary-label">Why this matters</div>
                <p className="authority-summary-text">{explanation}</p>
              </div>
            )}
            {cleanedText && (
              <div className="authority-verbatim-block">
                <div className="authority-summary-label">Relevant text</div>
                <p className="stage-verbatim-box" style={isVerbatim ? { whiteSpace: "pre-line" } : undefined}>{cleanedText}</p>
              </div>
            )}
            {renderSourceLink(url)}
          </div>
        );
      };

      const renderCaseLawItem = (caseLaw, idx) => {
        const caseTitle = caseLaw.title || caseLaw.case_name || `Case ${idx + 1}`;
        const caseUrl = caseLaw.url || caseLaw.source_url || "";
        const caseExplanation = caseLaw.explanation || "";
        const caseIsVerbatim = Boolean(caseLaw.is_verbatim_excerpt);
        const caseText = cleanResultText(caseLaw.text || "", caseIsVerbatim);
        const paraHeading = formatJudgmentChunkHeading(caseLaw);
        return (
          <div key={idx} className="related-case-law-item staged-precedent-card">
            <div className="case-law-header">
              {caseUrl ? (
                <a href={caseUrl} target="_blank" rel="noopener noreferrer" className="case-law-title-link" title="Opens in a new tab.">{caseTitle}</a>
              ) : (
                <span className="case-law-title">{caseTitle}</span>
              )}
            </div>
            {caseExplanation && (
              <div className="authority-summary-block authority-summary-block--nested">
                <div className="authority-summary-label">Why this precedent helps</div>
                <p className="authority-summary-text">{caseExplanation}</p>
              </div>
            )}
            {caseText && (
              <div className="authority-verbatim-block authority-verbatim-block--nested">
                {paraHeading ? <div className="staged-bare-act-section-bullet-label case-law-para-heading">{paraHeading}</div> : null}
                <p className="stage-verbatim-box stage-verbatim-box--nested" style={caseIsVerbatim ? { whiteSpace: "pre-line" } : undefined}>{caseText}</p>
              </div>
            )}
            {renderSourceLink(caseUrl, "View judgment")}
          </div>
        );
      };

      const renderBareActWithCaseLaws = (bareAct, idx) => {
        const title = bareAct.title || bareAct.act_name || bareAct.source || `Bare Act ${idx + 1}`;
        const url = bareAct.url || bareAct.source_url || "";
        const explanation = bareAct.explanation || "";
        const isVerbatim = Boolean(bareAct.is_verbatim_excerpt);
        const cleanedText = cleanResultText(bareAct.text || "", isVerbatim);
        const relatedCaseLaws = bareAct.related_case_laws || [];
        return (
          <div key={idx} className="result-item-row bare-act-with-cases staged-authority-card">
            <div className="result-item-header">
              <span className="result-item-number">{idx + 1}.</span>
              {url ? (
                <a href={url} target="_blank" rel="noopener noreferrer" className="result-item-title-link" title="Opens in a new tab.">{title}</a>
              ) : (
                <span className="result-item-title">{title}</span>
              )}
            </div>
            {explanation && (
              <div className="authority-summary-block">
                <div className="authority-summary-label">Why this section matters</div>
                <p className="authority-summary-text">{explanation}</p>
              </div>
            )}
            {cleanedText && (
              <div className="authority-verbatim-block">
                <div className="authority-summary-label">Relevant statutory text</div>
                <p className="stage-verbatim-box" style={isVerbatim ? { whiteSpace: "pre-line" } : undefined}>{cleanedText}</p>
              </div>
            )}
            {renderSourceLink(url)}
            {relatedCaseLaws.length > 0 && (
              <div className="related-case-laws">
                <h5 className="related-case-laws-heading">Closest judicial support</h5>
                {relatedCaseLaws.map((caseLaw, clIdx) => renderCaseLawItem(caseLaw, clIdx))}
              </div>
            )}
          </div>
        );
      };

      const renderNextStepsBlock = (bullets) => {
        if (!bullets || bullets.length === 0) return null;
        return (
          <div className="staged-next-steps-block">
            <div className="authority-summary-label">Next steps</div>
            <ul className="staged-next-steps-bullets">
              {bullets.map((b, i) => (
                <li key={`next-step-${i}-${b.title?.slice(0, 12) || i}`} className="staged-next-steps-bullet">
                  <div className="staged-next-steps-line">
                    Step {i + 1} — {b.title}
                  </div>
                  <p className="staged-next-steps-detail">{b.summary}</p>
                </li>
              ))}
            </ul>
          </div>
        );
      };

      /** Dedupe sections that appear under multiple disputes; keep highest retrieval score. */
      const flattenBareActGuidanceRows = (groups) => {
        const best = new Map();
        for (const group of groups || []) {
          for (const act of group.acts || []) {
            const actName = act.name || "";
            const url = act.url || "";
            for (const section of act.sections || []) {
              const key = `${actName.toLowerCase()}|${String(section.section_number || "").trim()}|${String(section.section_title || section.title || "").trim().toLowerCase()}`;
              const score = Number(section._sort_score ?? section._rerank_score ?? 0);
              const prev = best.get(key);
              if (!prev || score > prev.score) {
                best.set(key, { actName, url, section, score });
              }
            }
          }
        }
        return Array.from(best.values())
          .sort((a, b) => b.score - a.score)
          .map(({ actName, url, section }) => ({ actName, url, section }));
      };

      const renderBareActsHierarchy = (groups) => {
        const rows = flattenBareActGuidanceRows(groups);
        if (rows.length === 0) return null;
        const actBuckets = new Map();
        for (const row of rows) {
          const key = `${(row.actName || "").toLowerCase()}|${(row.url || "").trim()}`;
          const score = Number(row.section?._sort_score ?? row.section?._rerank_score ?? 0);
          let bucket = actBuckets.get(key);
          if (!bucket) {
            bucket = {
              actName: row.actName || "Applicable Act",
              url: row.url || "",
              sections: [],
              bestScore: -Infinity,
            };
            actBuckets.set(key, bucket);
          }
          bucket.sections.push(row.section);
          bucket.bestScore = Math.max(bucket.bestScore, score);
          if (!bucket.url && row.url) bucket.url = row.url;
        }
        const acts = Array.from(actBuckets.values()).sort((a, b) => b.bestScore - a.bestScore);
        acts.forEach((a) => {
          a.sections.sort(
            (s1, s2) =>
              Number(s2._sort_score ?? s2._rerank_score ?? 0) - Number(s1._sort_score ?? s1._rerank_score ?? 0),
          );
        });
        return (
          <div className="results-group-box staged-bare-acts-hierarchy">
            <h4 className="results-group-heading">Applicable bare act sections</h4>
            {acts.map((act, actIdx) => (
              <div key={`${act.actName}-${actIdx}`} className="staged-bare-act-group">
                <div className="staged-bare-act-group-heading">
                  {act.url ? (
                    <a
                      href={act.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="staged-bare-act-group-title-link"
                      title="Opens in a new tab."
                    >
                      {actIdx + 1}. {act.actName}
                    </a>
                  ) : (
                    <span className="staged-bare-act-group-title">
                      {actIdx + 1}. {act.actName}
                    </span>
                  )}
                </div>
                <ul className="staged-bare-act-section-bullets">
                  {act.sections.map((section, secIdx) => {
                    const explanation = (section.explanation || "").trim();
                    const isVerbatim = Boolean(section.is_verbatim_excerpt);
                    const cleanedText = cleanResultText(section.text || "", isVerbatim);
                    const secLabel = buildSectionLabel(section);
                    return (
                      <li key={`${secLabel}-${secIdx}`} className="staged-bare-act-section-li">
                        <div className="staged-bare-act-section-bullet-label">{secLabel}</div>
                        {explanation ? <p className="authority-summary-text staged-bare-act-verbatim-note">{explanation}</p> : null}
                        {cleanedText ? (
                          <>
                            <div className="authority-summary-label staged-relevant-extract-label">Relevant statutory text</div>
                            <p
                              className="stage-verbatim-box staged-bare-act-verbatim-quote"
                              style={isVerbatim ? { whiteSpace: "pre-line" } : undefined}
                            >
                              {cleanedText}
                            </p>
                          </>
                        ) : null}
                      </li>
                    );
                  })}
                </ul>
                {renderSourceLink(act.url, "View original bare act")}
              </div>
            ))}
          </div>
        );
      };

      const collectPrecedentHierarchyRows = (bareActs, caseLaws) => {
        const rows = [];
        if (bareActs.length > 0) {
          for (const ba of bareActs) {
            for (const cl of Array.isArray(ba.related_case_laws) ? ba.related_case_laws : []) {
              const caseTitle = String(cl.case_name || cl.title || "").trim() || "Judgment";
              const url = cl.url || cl.source_url || "";
              const score = Number(cl._sort_score ?? cl._rerank_score ?? 0);
              rows.push({ caseTitle, url, item: cl, score });
            }
          }
        }
        if (rows.length === 0) {
          for (const it of caseLaws || []) {
            const caseTitle = String(it.case_name || it.title || it.act_name || "").trim() || "Judgment";
            const url = it.url || it.source_url || "";
            const score = Number(it._sort_score ?? it._rerank_score ?? 0);
            rows.push({ caseTitle, url, item: it, score });
          }
        }
        return rows;
      };

      const dedupePrecedentRows = (rows) => {
        const best = new Map();
        for (const row of rows) {
          const it = row.item;
          const excerptKey = String(it.text || "")
            .slice(0, 160)
            .toLowerCase()
            .replace(/\s+/g, " ")
            .trim();
          const key = `${(row.caseTitle || "").toLowerCase()}|${(row.url || "").trim()}|${excerptKey}`;
          const prev = best.get(key);
          if (!prev || row.score > prev.score) best.set(key, row);
        }
        return Array.from(best.values()).sort((a, b) => b.score - a.score);
      };

      const renderJudicialPrecedentsHierarchy = (rows) => {
        if (!rows.length) return null;
        const buckets = new Map();
        for (const row of rows) {
          const key = `${(row.caseTitle || "").toLowerCase()}|${(row.url || "").trim()}`;
          let bucket = buckets.get(key);
          if (!bucket) {
            bucket = {
              caseTitle: row.caseTitle,
              url: row.url || "",
              items: [],
              bestScore: -Infinity,
            };
            buckets.set(key, bucket);
          }
          bucket.items.push(row.item);
          bucket.bestScore = Math.max(bucket.bestScore, row.score);
          if (!bucket.url && row.url) bucket.url = row.url;
        }
        const cases = Array.from(buckets.values()).sort((a, b) => b.bestScore - a.bestScore);
        cases.forEach((c) => {
          c.items.sort(
            (i1, i2) =>
              Number(i2._sort_score ?? i2._rerank_score ?? 0) - Number(i1._sort_score ?? i1._rerank_score ?? 0),
          );
        });
        return (
          <div className="results-group-box staged-bare-acts-hierarchy staged-judicial-precedents-hierarchy">
            <h4 className="results-group-heading">Relevant judicial precedents</h4>
            {cases.map((c, cIdx) => (
              <div key={`${c.caseTitle}-${cIdx}`} className="staged-bare-act-group">
                <div className="staged-bare-act-group-heading">
                  {c.url ? (
                    <a
                      href={c.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="staged-bare-act-group-title-link"
                      title="Opens in a new tab."
                    >
                      {cIdx + 1}. {c.caseTitle}
                    </a>
                  ) : (
                    <span className="staged-bare-act-group-title">
                      {cIdx + 1}. {c.caseTitle}
                    </span>
                  )}
                </div>
                <ul className="staged-bare-act-section-bullets">
                  {c.items.map((item, itemIdx) => {
                    const explanation = String(item.explanation || "").trim();
                    const isVerbatim = Boolean(item.is_verbatim_excerpt);
                    const cleanedText = cleanResultText(item.text || "", isVerbatim);
                    const chunkHeading = formatJudgmentChunkHeading(item);
                    const liKey = `${chunkHeading || "chunk"}-${itemIdx}-${(item.text || "").slice(0, 24)}`;
                    return (
                      <li key={liKey} className="staged-bare-act-section-li">
                        {chunkHeading ? <div className="staged-bare-act-section-bullet-label">{chunkHeading}</div> : null}
                        {explanation ? <p className="authority-summary-text staged-bare-act-verbatim-note">{explanation}</p> : null}
                        {cleanedText ? (
                          <p
                            className="stage-verbatim-box staged-bare-act-verbatim-quote"
                            style={isVerbatim ? { whiteSpace: "pre-line" } : undefined}
                          >
                            {cleanedText}
                          </p>
                        ) : null}
                      </li>
                    );
                  })}
                </ul>
                {renderSourceLink(c.url, "View judgment")}
              </div>
            ))}
          </div>
        );
      };

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

      if (responseType === "search_results" || responseType === "lookup_results") {
        return (
          <div className="message-final-opinion message-search-results conversational-response">
            {opinion && (
              <div className="conversational-summary-block">
                {opinion.split("\n").filter((line) => line.trim()).map((para, i) => (
                  <p key={i} className="conversational-summary-para">{para}</p>
                ))}
              </div>
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
            {bareActs.length === 0 && caseLaws.length === 0 && (
              <p className="search-empty">No results were found. Try refining your query with more specific legal terms.</p>
            )}
            {messageProgress && (
              <ProgressDisplay
                progress={messageProgress}
                expandedGroups={expandedGroups}
                setExpandedGroups={setExpandedGroups}
                expandKey={progressExpandKey}
                defaultOpen={progressDefaultOpen}
              />
            )}
          </div>
        );
      }

      if (responseType === "bare_act_guidance") {
        const groups = groupBareActsByDisputeAndAct(bareActs);
        const verbatimRows = flattenBareActGuidanceRows(groups);
        const nextStepsBullets = buildNextStepsBullets(nextStepsSummaryRaw, nextSteps);
        const opinionDisplay = stripJudicialPrecedentCtaLines(opinion);
        return (
          <div className="message-final-opinion message-staged-guidance">
            {opinionDisplay && (
              <div className="conversational-summary-block stage-summary-block bare-act-guidance-summary">
                <div className="authority-summary-label">Summary</div>
                {opinionDisplay.split("\n").filter((line) => line.trim()).map((para, i) => (
                  <p key={i} className="conversational-summary-para">{para}</p>
                ))}
              </div>
            )}
            {renderBareActsHierarchy(groups)}
            {renderNextStepsBlock(nextStepsBullets)}
            <p className="conversational-summary-para staged-judicial-precedent-cta">{JUDICIAL_PRECEDENT_CTA_OFFER}</p>
            {verbatimRows.length === 0 && bareActs.length === 0 && (
              <p className="search-empty">No grounded bare act sections were available for display.</p>
            )}
            {messageProgress && (
              <ProgressDisplay
                progress={messageProgress}
                expandedGroups={expandedGroups}
                setExpandedGroups={setExpandedGroups}
                expandKey={progressExpandKey}
                defaultOpen={progressDefaultOpen}
              />
            )}
          </div>
        );
      }

      if (responseType === "precedent_support") {
        const precedentRows = dedupePrecedentRows(collectPrecedentHierarchyRows(bareActs, caseLaws));
        return (
          <div className="message-final-opinion message-staged-guidance">
            {opinion && (
              <div className="conversational-summary-block stage-summary-block bare-act-guidance-summary">
                <div className="authority-summary-label">Summary</div>
                {opinion.split("\n").filter((line) => line.trim()).map((para, i) => (
                  <p key={i} className="conversational-summary-para">{para}</p>
                ))}
              </div>
            )}
            {renderJudicialPrecedentsHierarchy(precedentRows)}
            {precedentRows.length === 0 && (
              <p className="search-empty">No grounded precedents were available for display.</p>
            )}
            {messageProgress && (
              <ProgressDisplay
                progress={messageProgress}
                expandedGroups={expandedGroups}
                setExpandedGroups={setExpandedGroups}
                expandKey={progressExpandKey}
                defaultOpen={progressDefaultOpen}
              />
            )}
          </div>
        );
      }

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
                expandKey={progressExpandKey}
                defaultOpen={progressDefaultOpen}
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
          {messageProgress && (
            <ProgressDisplay
              progress={messageProgress}
              expandedGroups={expandedGroups}
              setExpandedGroups={setExpandedGroups}
              expandKey={progressExpandKey}
              defaultOpen={progressDefaultOpen}
            />
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
  const handleDownloadPdf = async (opinionOverride) => {
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
      if (/^â”€+$/.test(lines[i].trim())) {
        endIndex = i;
        break;
      }
    }
    const pdfOpinion = lines.slice(0, endIndex).join("\n").trim();
    if (!pdfOpinion) {
      alert("No opinion content left for PDF after removing follow-up section.");
      return;
    }

    const { default: jsPDF } = await import("jspdf");
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
        {/* LEFT PANE â€“ New chat, Chat history, Bare Acts, Case Laws */}
        <div
          className={`left-column${sidebarCollapsed ? " left-column--collapsed" : ""}`}
          style={sidebarCollapsed ? undefined : { width: leftColumnWidth, minWidth: leftColumnWidth, maxWidth: leftColumnWidth }}
        >
            <div className="left-column-scroll">
            <div className="left-column-topbar">
              <button
                type="button"
                onClick={handleNewChat}
                className="chat-history-new-btn"
              >
                {"+ New chat"}
              </button>
              <button
                type="button"
                className="sidebar-collapse-btn"
                onClick={() => setSidebarCollapsed(true)}
                aria-label="Collapse sidebar"
                title="Collapse sidebar"
              >
                {"\u2039"}
              </button>
            </div>
            <div className={`chat-history-section sidebar-section${sidebarExpandedSection === "chat_history" ? " sidebar-section--active" : ""}`}>
              <details
                className="chat-history-collapsible"
                open={sidebarExpandedSection === "chat_history"}
                onClick={(e) => {
                  if (e.target.closest("summary")) {
                    e.preventDefault();
                    toggleSidebarSection("chat_history");
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
                                  <div className="chat-history-hover-actions" aria-hidden="true">
                                    <button
                                      type="button"
                                      onClick={(e) => { e.stopPropagation(); startRenamingChat(chat); }}
                                      className="chat-history-icon-btn"
                                      title="Rename chat"
                                      aria-label="Rename chat"
                                    >
                                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                                        <path d="M2.695 14.763l-1.262 3.154a.5.5 0 00.65.65l3.155-1.262a4 4 0 001.343-.885L17.5 5.5a2.121 2.121 0 00-3-3L3.58 13.42a4 4 0 00-.885 1.343z" />
                                      </svg>
                                    </button>
                                    <button
                                      type="button"
                                      onClick={(e) => { e.stopPropagation(); handleDeleteChat(chat); }}
                                      className="chat-history-icon-btn chat-history-icon-btn--danger"
                                      title="Delete chat"
                                      aria-label="Delete chat"
                                    >
                                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                                        <path fillRule="evenodd" d="M8.75 2a.75.75 0 00-.75.75V3H5.5a.75.75 0 000 1.5h.443l.664 9.298A2.25 2.25 0 008.85 15.9h2.3a2.25 2.25 0 002.243-2.102l.664-9.298h.443a.75.75 0 000-1.5H12V2.75A.75.75 0 0011.25 2h-2.5zM9.5 3v-.25h1V3h-1zm-.75 4.25a.75.75 0 011.5 0v4.5a.75.75 0 01-1.5 0v-4.5zm3 0a.75.75 0 011.5 0v4.5a.75.75 0 01-1.5 0v-4.5z" clipRule="evenodd" />
                                      </svg>
                                    </button>
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
            <details
              className={`bare-acts-collapsible sidebar-section${sidebarExpandedSection === "bare_acts" ? " sidebar-section--active" : ""}`}
              open={sidebarExpandedSection === "bare_acts"}
              onClick={(e) => {
                if (e.target.closest("summary")) {
                  e.preventDefault();
                  toggleSidebarSection("bare_acts");
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
                          (act) => {
                            const n = typeof act === "string" ? act : act.name;
                            return !bareActsFilter.trim() || n.toLowerCase().includes(bareActsFilter.trim().toLowerCase());
                          }
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
                              {filteredActs.map((act, idx) => {
                                const actName = typeof act === "string" ? act : act.name;
                                const actFile = typeof act === "string" ? null : act.file;
                                const useRelative =
                                  typeof window !== "undefined" &&
                                  (window.location.hostname === "localhost" ||
                                    window.location.hostname === "127.0.0.1" ||
                                    API_BASE === "" ||
                                    API_BASE.startsWith(window.location.origin));
                                const base = useRelative ? "" : API_BASE;
                                const downloadUrl = actFile
                                  ? `${base}/bareacts/view?file=${encodeURIComponent(actFile)}&name=${encodeURIComponent(actName)}`
                                  : `${base}/bareacts/view?name=${encodeURIComponent(actName)}`;
                                return (
                                  <li key={`${jurisdiction.name}-${actName}-${idx}`} className="bare-act-list-item">
                                    <a
                                      href={downloadUrl}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="bare-act-download-link"
                                    >
                                      {actName}
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

            {/* Case Laws â€“ below Bare Acts, same functionality */}
            <details
              className={`case-laws-collapsible sidebar-section${sidebarExpandedSection === "case_laws" ? " sidebar-section--active" : ""}`}
              open={sidebarExpandedSection === "case_laws"}
              onClick={(e) => {
                if (e.target.closest("summary")) {
                  e.preventDefault();
                  toggleSidebarSection("case_laws");
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
                  {/* Most cited â€“ single HTML table with top 200 cases */}
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
                          (c) => {
                            const n = typeof c === "string" ? c : c.name;
                            return !caseLawsFilter.trim() || n.toLowerCase().includes(caseLawsFilter.trim().toLowerCase());
                          }
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
                              {filteredCases.map((c, idx) => {
                                const caseName = typeof c === "string" ? c : c.name;
                                const caseFile = typeof c === "string" ? null : c.file;
                                const useRelative =
                                  typeof window !== "undefined" &&
                                  (window.location.hostname === "localhost" ||
                                    window.location.hostname === "127.0.0.1" ||
                                    API_BASE === "" ||
                                    API_BASE.startsWith(window.location.origin));
                                const base = useRelative ? "" : API_BASE;
                                const downloadUrl = caseFile
                                  ? `${base}/caselaws/view?file=${encodeURIComponent(caseFile)}&name=${encodeURIComponent(caseName)}`
                                  : `${base}/caselaws/view?name=${encodeURIComponent(caseName)}`;
                                return (
                                  <li key={`${court.name}-${caseName}-${idx}`} className="case-laws-list-item">
                                    <a
                                      href={downloadUrl}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="case-laws-download-link"
                                    >
                                      {caseName}
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

            </div>
        </div>
        {!sidebarCollapsed && (
          <div
            className="left-column-resizer"
            onMouseDown={handleSidebarResizeMouseDown}
            role="separator"
            aria-orientation="vertical"
            aria-label="Resize sidebar"
          />
        )}

        {/* RIGHT PANE â€“ Title at top, then chat area */}
        <div className={`right-content-wrapper${messages.some((m) => m.role === "user") ? " chat-mode" : ""}`}>
          {sidebarCollapsed && (
            <button
              type="button"
              className="sidebar-expand-btn"
              onClick={() => setSidebarCollapsed(false)}
              aria-label="Expand sidebar"
              title="Expand sidebar"
            >
              {"\u203A"}
            </button>
          )}
          {/* Title at top left of right pane */}
          <div className="right-pane-header">
            <h1 className="main-title">
              {"\u2696 Nyaymalaw"}
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
                    <ChatComposer
                      loading={loading}
                      placeholder="Describe your case or ask a question"
                      onSubmit={handleSubmit}
                      resetSignal={composerResetSignal}
                      chatMode={chatMode}
                      onChatModeChange={setChatMode}
                      selectedModel={selectedModel}
                      onModelChange={setSelectedModel}
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
                    key={`msg-${msg.id || i}`}
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
                        <span className="avatar-ai">{"\u2696"}</span>
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
                                <button
                                  type="button"
                                  className="message-action-btn"
                                  onClick={() => startEditUserMessage(i)}
                                  title="Edit message"
                                  aria-label="Edit message"
                                >
                                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true"><path d="M2.695 14.763l-1.262 3.154a.5.5 0 00.65.65l3.155-1.262a4 4 0 001.343-.885L17.5 5.5a2.121 2.121 0 00-3-3L3.58 13.42a4 4 0 00-.885 1.343z" /></svg>
                                </button>
                                <button
                                  type="button"
                                  className="message-action-btn"
                                  onClick={() => handleCopyMessage(msg, i)}
                                  title={copyJustDoneIndex === i ? "Copied" : "Copy message"}
                                  aria-label={copyJustDoneIndex === i ? "Copied" : "Copy message"}
                                >
                                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true"><path d="M7 3.5A1.5 1.5 0 018.5 2h3.879a1.5 1.5 0 011.06.44l3.122 3.12A1.5 1.5 0 0117 6.622V12.5a1.5 1.5 0 01-1.5 1.5h-1v-3.379a3 3 0 00-.879-2.121L10.5 5.379A3 3 0 008.379 4.5H7v-1z" /><path d="M4.5 6A1.5 1.5 0 003 7.5v9A1.5 1.5 0 004.5 18h7a1.5 1.5 0 001.5-1.5v-5.879a1.5 1.5 0 00-.44-1.06L9.44 6.439A1.5 1.5 0 008.379 6H4.5z" /></svg>
                                </button>
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="message-bubble message-bubble--assistant">
                          <div className="message-bubble-inner">
                            {renderAssistantContent(msg.content, {
                              progressExpandKey: msg.id != null ? `pt_${msg.id}` : PROGRESS_TRACKER_KEY,
                              progressDefaultOpen: false,
                            })}
                            <div className="message-bubble-actions">
                              <button
                                type="button"
                                className="message-action-btn"
                                onClick={() => handleCopyMessage(msg, i)}
                                title={copyJustDoneIndex === i ? "Copied" : "Copy message"}
                                aria-label={copyJustDoneIndex === i ? "Copied" : "Copy message"}
                              >
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true"><path d="M7 3.5A1.5 1.5 0 018.5 2h3.879a1.5 1.5 0 011.06.44l3.122 3.12A1.5 1.5 0 0117 6.622V12.5a1.5 1.5 0 01-1.5 1.5h-1v-3.379a3 3 0 00-.879-2.121L10.5 5.379A3 3 0 008.379 4.5H7v-1z" /><path d="M4.5 6A1.5 1.5 0 003 7.5v9A1.5 1.5 0 004.5 18h7a1.5 1.5 0 001.5-1.5v-5.879a1.5 1.5 0 00-.44-1.06L9.44 6.439A1.5 1.5 0 008.379 6H4.5z" /></svg>
                              </button>
                              {msg.id && (
                                <button
                                  type="button"
                                  className={`message-action-btn ${openFeedbackMessageId === msg.id ? "message-action-btn--active" : ""}`}
                                  onClick={() => openFeedbackForMessage(msg)}
                                  title="Give feedback"
                                  aria-label="Give feedback"
                                >
                                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true"><path d="M10 2.5a7.5 7.5 0 100 15 7.5 7.5 0 000-15zm-3 6.25a.75.75 0 011.5 0v.5a1.5 1.5 0 003 0v-.5a.75.75 0 011.5 0v.5a3 3 0 11-6 0v-.5zm1.125-2.125a.875.875 0 110-1.75.875.875 0 010 1.75zm3.75 0a.875.875 0 110-1.75.875.875 0 010 1.75z" /></svg>
                                </button>
                              )}
                              {msg.content?.type === "final_opinion" && (msg.content?.opinionText || "").trim() && (
                                <button
                                  type="button"
                                  className="message-action-btn"
                                  onClick={() => handleDownloadPdf(msg.content?.opinionText)}
                                  title="Download"
                                  aria-label="Download"
                                >
                                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true"><path d="M10.75 2.75a.75.75 0 00-1.5 0v8.614L6.295 8.235a.75.75 0 10-1.09 1.03l4.25 4.5a.75.75 0 001.09 0l4.25-4.5a.75.75 0 00-1.09-1.03l-2.955 3.129V2.75z" /><path d="M3.5 12.75a.75.75 0 00-1.5 0v2.5A2.75 2.75 0 004.75 18h10.5A2.75 2.75 0 0018 15.25v-2.5a.75.75 0 00-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5z" /></svg>
                                </button>
                              )}
                            </div>
                            {msg.id && openFeedbackMessageId === msg.id && (
                              <ResponseFeedbackPanel
                                messageId={msg.id}
                                existing={feedbackStatusByMessageId[msg.id]}
                                onSave={(draft) => submitResponseFeedback(msg, i, draft)}
                                onCancel={closeFeedbackPanel}
                              />
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {loading && (
                  <div className="message message--assistant">
                    <div className="message-avatar">
                      <span className="avatar-ai">{"\u2696"}</span>
                    </div>
                    <div className="message-content">
                      {/* Step timeline â€” shows while no tokens yet */}
                      {streamingSteps.length > 0 && !streamingToken && (
                        <div className="streaming-steps">
                          {streamingSteps.map((s, i) => {
                            const isActiveProcessing = !s.done && i === streamingSteps.length - 1;
                            return (
                              <div key={i} className={`streaming-step ${s.done ? "streaming-step--done" : "streaming-step--active"}`}>
                                {s.icon ? <span className="streaming-step-icon">{s.icon}</span> : null}
                                <span
                                  className={
                                    isActiveProcessing
                                      ? "streaming-step-msg streaming-step-msg--processing"
                                      : "streaming-step-msg"
                                  }
                                >
                                  {isActiveProcessing ? (
                                    <>
                                      {stripTrailingStepEllipsis(s.message)}
                                      <span className="streaming-processing-ellipsis">...</span>
                                    </>
                                  ) : (
                                    s.message
                                  )}
                                </span>
                                {s.detail ? <StreamingRetrievalDetail detail={s.detail} /> : null}
                              </div>
                            );
                          })}
                        </div>
                      )}
                      {/* Streaming token text â€” shows as LLM generates */}
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
                          <div className="streaming-text">{streamingToken}<span className="streaming-cursor">{"\u258B"}</span></div>
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
                          expandKey={PROGRESS_LIVE_KEY}
                          defaultOpen={true}
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
                  <ChatComposer
                    loading={loading}
                    placeholder={
                      stage === "interview" && currentQuestion
                        ? "Type your details here"
                        : stage === "await_facts"
                        ? "Describe your case facts here"
                        : "Type here to start a new case"
                    }
                    onSubmit={handleSubmit}
                    resetSignal={composerResetSignal}
                    chatMode={chatMode}
                    onChatModeChange={setChatMode}
                    selectedModel={selectedModel}
                    onModelChange={setSelectedModel}
                    showDisclaimer={bottomExpandedSection == null}
                  />
                </div>

                {error && (
                  <div className="chat-error">
                    <span>{"\u26A0"}</span> {error}
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

    </div>
  );
}

export default App;

