import { useState, useRef, useEffect, useMemo } from "react";
import jsPDF from "jspdf"; // npm install jspdf
import "./App.css";

const AUTH_TOKEN_KEY = "nyaymalaw_auth_token";
const CURRENT_USER_KEY = "nyaymalaw_current_user";
const CURRENT_USER_NAME_KEY = "nyaymalaw_current_user_name";

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
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [phase, setPhase] = useState("fact_collection"); // Retain existing phase logic
  const [factsSummary, setFactsSummary] = useState(null); // Retain existing
  const [pendingMaterials, setPendingMaterials] = useState(null); // Retain existing
  const [loading, setLoading] = useState(false); // Retain existing
  const [error, setError] = useState(""); // Retain existing
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const messagesContainerRef = useRef(null);

  // New interview state from snippet
  const [stage, setStage] = useState("await_facts"); // "await_facts" | "interview" | "done"
  const [facts, setFacts] = useState("");
  const [currentQuestion, setCurrentQuestion] = useState("");
  const [qaHistory, setQaHistory] = useState([]); // [{question, answer}]
  const [opinionText, setOpinionText] = useState("");
  const [retrieved, setRetrieved] = useState([]);
  const [rawResponse, setRawResponse] = useState("");
  const [showDebug, setShowDebug] = useState(false);

  // Saved chats (ChatGPT-style): list of past conversations, persisted to localStorage
  const [savedChats, setSavedChats] = useState([]);
  const hasSavedCurrentChatRef = useRef(false);
  const [editingChatId, setEditingChatId] = useState(null);
  const [editingTitle, setEditingTitle] = useState("");
  const [openMenuChatId, setOpenMenuChatId] = useState(null);
  const editInputRef = useRef(null);
  const currentChatIdRef = useRef(null);

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
  // In production (e.g. https://nyaymalaw.in) use same origin or VITE_API_BASE; locally use backend on :8000
  const API_BASE =
    import.meta.env.VITE_API_BASE ||
    (typeof window !== "undefined" &&
     (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")
      ? "http://127.0.0.1:8000"
      : (typeof window !== "undefined" ? window.location.origin : "http://127.0.0.1:8000"));

  // Load saved chats from backend (no auth: backend uses anonymous user)
  useEffect(() => {
    const token = localStorage.getItem(AUTH_TOKEN_KEY);
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    fetch(`${API_BASE}/chats`, { headers })
      .then((res) => (res.ok ? res.json() : { chats: [] }))
      .then((data) => setSavedChats(Array.isArray(data.chats) ? data.chats : []))
      .catch(() => setSavedChats([]));
  }, [API_BASE]);

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

  // Fetch Bare Acts from API when available; keep default list if API fails or returns empty
  useEffect(() => {
    const fetchBareActs = async () => {
      try {
        const res = await fetch(`${API_BASE}/bareacts/list`);
        const data = await res.json();
        const acts = data.acts || [];
        setBareActs(acts.length > 0 ? acts : DEFAULT_BARE_ACTS);
      } catch (err) {
        console.error("Error fetching bare acts:", err);
        setBareActs(DEFAULT_BARE_ACTS);
      }
    };
    fetchBareActs();
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

  // Existing textarea autosize (retained)
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "24px";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [input]);

  // Keep focus in chat input when not loading (so user can type without clicking)
  useEffect(() => {
    if (!loading && textareaRef.current) textareaRef.current.focus();
  }, [loading]);

  // -------------------------
  // Reset conversation (from snippet, adapted for existing state)
  // -------------------------
  const handleStartNewCase = () => {
    hasSavedCurrentChatRef.current = false;
    currentChatIdRef.current = null;
    setStage("await_facts");
    setFacts("");
    setInput(""); // Reset existing input
    setCurrentQuestion("");
    setQaHistory([]);
    setOpinionText("");
    setRetrieved([]);
    setRawResponse("");
    setError(""); // Reset existing error
    setPendingMaterials(null); // Reset existing pending materials
    setMessages([]);
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

  const chatGroups = useMemo(() => groupChatsByDate(savedChats), [savedChats]);

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
  const handleSubmit = async () => {
    const text = (input || "").trim(); // Use existing 'input' state
    if (!text) {
      alert("Please type something before pressing Submit.");
      return;
    }

    setError("");
    setLoading(true);
    setInput(""); // Clear input after submission
    setPendingMaterials(null); // Clear pending materials on new submission

    const userMsg = { role: "user", content: text, timestamp: new Date().toISOString() };
    const isFirstMessage = messages.length === 0;

    setMessages((prev) => [...prev, userMsg]);

    if (isFirstMessage) {
      const chatId = Date.now();
      const title = text.length > 50 ? text.slice(0, 50) + "…" : text;
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

    // 1) Initial facts (await_facts stage)
    if (stage === "await_facts") {
      setFacts(text); // Store initial facts

      try {
        const res = await fetch(`${API_BASE}/submit_case`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        let data = {};
        try {
          const text = await res.text();
          data = text ? JSON.parse(text) : {};
        } catch (_) {
          setError("Server returned an invalid or empty response. Please try again.");
          setRawResponse("Error: Invalid or empty response from server.");
          return;
        }
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
          const newAssistantMsg = {
            role: "assistant",
            content: {
              type: "final_opinion",
              response_type: data.response_type || "legal_opinion",
              opinionText: opinion,
              bare_acts: Array.isArray(data.bare_acts) ? data.bare_acts : [],
              case_laws: Array.isArray(data.case_laws) ? data.case_laws : [],
              retrieved: retr,
            },
            timestamp: new Date().toISOString(),
          };
          setMessages((prev) => [...prev, newAssistantMsg]);
        } else if (data.needs_confirmation) {
          setPendingMaterials(data.materials_to_confirm);
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: data.summary, // Use summary for display
              timestamp: new Date().toISOString(),
            },
          ]);
        } else {
          if (data.message) setError(data.message);
        }
      } catch (err) {
        console.error("submit_case error:", err);
        setError(err.message && err.message.includes("JSON") ? "Server returned an invalid response. Please try again." : "Error during processing: " + (err.message || "Please try again."));
        setRawResponse("Error: " + (err.message || ""));
      } finally {
        setLoading(false);
      }
      return;
    }

    // 2) Follow-up answers (interview stage)
    if (stage === "interview") {
      if (!currentQuestion) {
        alert("No current question from the assistant.");
        setLoading(false);
        return;
      }

      const updatedHistory = [
        ...qaHistory,
        { question: currentQuestion, answer: text },
      ];
      setQaHistory(updatedHistory);
      setCurrentQuestion(""); // Clear current question after answering

      try {
        const res = await fetch(`${API_BASE}/interview_step`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            facts,
            qa_history: updatedHistory,
          }),
        });
        let data = {};
        try {
          const text = await res.text();
          data = text ? JSON.parse(text) : {};
        } catch (_) {
          setError("Server returned an invalid or empty response. Please try again.");
          setRawResponse("Error: Invalid or empty response from server.");
          return;
        }
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
          const newAssistantMsg = {
            role: "assistant",
            content: {
              type: "final_opinion",
              response_type: data.response_type || "legal_opinion",
              opinionText: opinion,
              bare_acts: Array.isArray(data.bare_acts) ? data.bare_acts : [],
              case_laws: Array.isArray(data.case_laws) ? data.case_laws : [],
              retrieved: retr,
            },
            timestamp: new Date().toISOString(),
          };
          setMessages((prev) => [...prev, newAssistantMsg]);
        } else if (data.needs_confirmation) {
          setPendingMaterials(data.materials_to_confirm);
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: data.summary, // Use summary for display
              timestamp: new Date().toISOString(),
            },
          ]);
        } else {
          if (data.message) setError(data.message);
        }
      } catch (err) {
        console.error("interview_step error:", err);
        setError("Error during processing: " + (err.message || "Network or server error"));
        setRawResponse("Error: " + err.message);
      } finally {
        setLoading(false);
      }
      return;
    }

    // 3) Continue a loaded chat: use full context and same chat (conversation = history only; new message sent separately)
    if (stage === "done") {
      const normalizeContent = (msg) => {
        if (typeof msg.content === "string") return msg.content;
        if (msg.content?.opinionText != null) return msg.content.opinionText || "";
        return msg.content?.text ?? msg.content?.summary ?? "";
      };
      // Send only previous messages (we already appended userMsg above), so backend gets full context + current message separately
      const previousMessages = messages.slice(0, -1);
      const conversation = previousMessages.map((m) => ({ role: m.role, content: normalizeContent(m) }));
      try {
        // Let React commit loading state so "Nyaymalaw is thinking..." appears
        await new Promise((r) => setTimeout(r, 0));
        const res = await fetch(`${API_BASE}/conversation/continue`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ conversation, message: text }),
        });
        let data = {};
        try {
          data = await res.json();
        } catch (_) {
          setError("Invalid response from server.");
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: "The server response was invalid. Please try again.", timestamp: new Date().toISOString() },
          ]);
          return;
        }
        if (!res.ok) {
          setError(data.detail || `Request failed (${res.status})`);
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: (data.detail && String(data.detail)) || "", timestamp: new Date().toISOString() },
          ]);
          return;
        }
        const assistantContent = (data.next_question ?? data.message ?? "").trim();
        if (data.status === "question") {
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: assistantContent, timestamp: new Date().toISOString() },
          ]);
          if (Array.isArray(data.retrieved)) setRetrieved(data.retrieved);
        } else if (data.status === "done") {
          const opinion = data.opinion_text || "";
          const retr = Array.isArray(data.retrieved) ? data.retrieved : [];
          setOpinionText(opinion);
          setRetrieved(retr);
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
              },
              timestamp: new Date().toISOString(),
            },
          ]);
        } else if (data.needs_confirmation) {
          setPendingMaterials(data.materials_to_confirm);
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: data.summary ?? "", timestamp: new Date().toISOString() },
          ]);
        } else {
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: assistantContent, timestamp: new Date().toISOString() },
          ]);
        }
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

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(); // Call the new handleSubmit
    }
  };

  const renderAssistantContent = (content) => {
    if (content == null) return null;
    if (typeof content === "string") {
      return <p className="message-text">{content}</p>;
    }
    if (content.type === "question") {
      return <p className="message-text">{content.text}</p>;
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

      // Helper: render a single item row inside a grouped box
      const renderResultItem = (item, idx) => {
        const title = item.title || item.act_name || item.source || `Result ${idx + 1}`;
        const url = item.url || "";
        const rawText = item.text || "";
        // Clean: take meaningful sentences, skip very short fragments and navigation-like lines
        const cleanLines = rawText
          .split(/[.\n]/)
          .map(l => l.trim())
          .filter(l => l.length > 20 && !/^(Skip|Search|Login|Menu|Toggle|Free|Premium|Print|Download|Pricing)/i.test(l));
        const snippetText = cleanLines.slice(0, 6).join(". ").trim();
        const snippet = snippetText.length > 400 ? snippetText.slice(0, 400) + "..." : snippetText;
        return (
          <div key={idx} className="result-item-row">
            <div className="result-item-header">
              <span className="result-item-number">{idx + 1}.</span>
              {url ? (
                <a href={url} target="_blank" rel="noopener noreferrer" className="result-item-title-link">
                  {title}
                </a>
              ) : (
                <span className="result-item-title">{title}</span>
              )}
            </div>
            {snippet && <p className="result-item-snippet">{snippet}</p>}
            {rawText.length > snippet.length + 50 && (
              <details className="result-item-expand">
                <summary className="result-item-read-more">Read more</summary>
                <p className="result-item-full-text">{rawText}</p>
              </details>
            )}
            {url && (
              <a href={url} target="_blank" rel="noopener noreferrer" className="result-item-source-link">
                View original source
              </a>
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

            {/* Supreme Court Judgments always shown first for search, then bare acts */}
            {responseType === "search_results" ? (
              <>
                {renderGroupBox("Supreme Court Judgments", caseLaws)}
                {bareActs.length > 0 && renderGroupBox("Relevant Bare Acts", bareActs)}
              </>
            ) : (
              <>
                {renderGroupBox("Relevant Bare Acts", bareActs)}
                {caseLaws.length > 0 && renderGroupBox("Supreme Court Judgments", caseLaws)}
              </>
            )}
            {bareActs.length === 0 && caseLaws.length === 0 && (
              <p className="search-empty">No results were found. Try refining your query with more specific legal terms.</p>
            )}
          </div>
        );
      }

      // ---------- legal_opinion (formal layout with header + PDF download) ----------
      return (
        <div className="message-final-opinion">
          <div className="final-output-header final-output-header--chat">
            <h4 className="opinion-title">Legal Opinion</h4>
            <button
              type="button"
              onClick={handleDownloadPdf}
              className="download-pdf-button"
            >
              Download as PDF
            </button>
          </div>
          {opinion && <p className="opinion-text">{opinion}</p>}
          {renderGroupBox("Relevant Bare Acts", bareActs)}
          {renderGroupBox("Relevant Case Laws", caseLaws)}
          {bareActs.length === 0 && caseLaws.length === 0 && (
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
                      <a href={item.url} target="_blank" rel="noopener noreferrer" className="result-link">
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
  // Download PDF handler (from snippet)
  // -------------------------
  const handleDownloadPdf = () => {
    const trimmedOpinion = (opinionText || "").trim();
    if (!trimmedOpinion) {
      alert("No final opinion available to download yet.");
      return;
    }

    const doc = new jsPDF({
      unit: "pt",
      format: "a4",
    });

    const marginLeft = 40;
    const maxWidth = 515;
    let y = 40;

    doc.setFont("Helvetica", "bold");
    doc.setFontSize(16);
    doc.text("Legal Opinion", marginLeft, y);
    y += 28;

    doc.setFont("Helvetica", "normal");
    doc.setFontSize(11);

    const opinionLines = doc.splitTextToSize(trimmedOpinion, maxWidth);
    doc.text(opinionLines, marginLeft, y);
    y += opinionLines.length * 14 + 20;

    // Relevant Bare Acts
    if (retrieved && retrieved.length > 0) {
      const uniqueActs = Array.from(
        new Set(
          retrieved.map(
            (item) => (item.meta && item.meta.act_name) || "Unknown Act"
          )
        )
      );

      if (y > 780) {
        doc.addPage();
        y = 40;
      }

      doc.setFont("Helvetica", "bold");
      doc.setFontSize(13);
      doc.text("Relevant Bare Acts:", marginLeft, y);
      y += 20;

      doc.setFont("Helvetica", "normal");
      doc.setFontSize(11);

      uniqueActs.forEach((act) => {
        if (y > 780) {
          doc.addPage();
          y = 40;
        }
        const line = `- ${act}`;
        const lines = doc.splitTextToSize(line, maxWidth);
        doc.text(lines, marginLeft, y);
        y += lines.length * 14;
      });
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
      {/* Header */}
      <div className="main-header-fixed">
        <h1 className="main-title">
          🏛️ Nyaymalaw – Your legal buddy
        </h1>
        <div className="header-user">
          <span className="header-email" title={currentUser || "Guest"}>{currentUser || "Guest"}</span>
          <button type="button" onClick={handleLogout} className="header-logout-btn">
            New chat
          </button>
        </div>
      </div>

      {/* Content */}
      <div
        className="main-content-wrapper"
      >
        <div
          className="columns-container"
        >
          {/* LEFT COLUMN – New chat, Bare Acts, Chat history */}
          <div
            className="left-column"
            // Removed position: sticky, top, align-self, max-height, overflowY
          >
            <button
              type="button"
              onClick={handleNewChat}
              className="chat-history-new-btn"
            >
              ＋ New chat
            </button>
            <details className="bare-acts-collapsible">
              <summary className="bare-acts-collapsible-summary">
                {bareActs.length ? (
                  <span className="bare-acts-count">📚 Bare Acts Library ({bareActs.length})</span>
                ) : (
                  "📚 Bare Acts Library"
                )}
              </summary>
              {bareActs.length ? (
                <ul className="bare-act-list">
                  {bareActs.map((name, idx) => (
                    <li key={`${name}-${idx}`} className="bare-act-list-item">
                      <button
                        type="button"
                        onClick={(e) => handleBareActDownload(e, name)}
                        className="bare-act-download-link"
                      >
                        {name}
                      </button>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="bare-act-empty-message">
                  No bare acts in vector store yet.
                </p>
              )}
            </details>

            {/* Saved chats – below Bare Acts Library */}
            <div className="chat-history-section">
              <details className="chat-history-collapsible">
                <summary className="chat-history-collapsible-summary">
                  💬 Chat history ({savedChats.length})
                </summary>
                {savedChats.length > 0 ? (
                  <div className="chat-history-groups">
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
                ) : (
                  <p className="chat-history-empty">No previous chats.</p>
                )}
              </details>
            </div>
          </div>

          {/* RIGHT CONTENT – Chat window only (80%) */}
          <div className={`right-content-wrapper${messages.some((m) => m.role === "user") ? " chat-mode" : ""}`}>
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
                  <div className="chat-input-container">
                    <textarea
                      ref={textareaRef}
                      autoFocus
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={handleKeyDown}
                      placeholder="Describe your case or ask a question..."
                      className="chat-input"
                      rows={1}
                      disabled={loading}
                    />
                    <button
                      type="button"
                      onClick={handleSubmit}
                      disabled={loading || !input.trim()}
                      className="chat-send"
                      aria-label="Send message"
                    >
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M12 19V5M5 12l7-7 7 7" />
                      </svg>
                    </button>
                  </div>
                  <p className="chat-disclaimer">Nyaymalaw AI can make mistakes. Consider checking important information.</p>
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
                    key={`msg-${msg.role}-${i}-${typeof msg.content === "string" ? msg.content?.slice(0, 30) : msg.content?.type || ""}`}
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
                        <div className="message-bubble message-bubble--user">{msg.content}</div>
                      ) : (
                        <div className="message-bubble message-bubble--assistant">
                          {renderAssistantContent(msg.content)}
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
                      <div className="message-bubble message-bubble--assistant typing-indicator">
                        <span className="typing-indicator-text">Nyaymalaw is thinking</span>
                        <span className="typing-dots">
                          <span className="typing-dot" />
                          <span className="typing-dot" />
                          <span className="typing-dot" />
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                    <div ref={messagesEndRef} />
                  </div>
                </div>

                {/* Fixed bottom input when in chat mode */}
                <div className="chat-input-wrapper">
              <div className="chat-input-container">
                <textarea
                  ref={textareaRef}
                  autoFocus
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={
                    stage === "interview" && currentQuestion
                      ? "Type your details here..."
                      : stage === "await_facts"
                      ? "Describe your case facts here..."
                      : "Type here to start a new case..."
                  }
                  className="chat-input"
                  rows={1}
                  disabled={loading}
                />
                <button
                  type="button"
                  onClick={handleSubmit}
                  disabled={loading || !input.trim()}
                  className="chat-send"
                  aria-label="Send message"
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 19V5M5 12l7-7 7 7" />
                  </svg>
                </button>
              </div>
              <p className="chat-disclaimer">
                Nyaymalaw AI can make mistakes. Consider checking important information.
              </p>
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
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;