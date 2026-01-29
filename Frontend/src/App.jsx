import { useState, useRef, useEffect } from "react";
import "./App.css";

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [phase, setPhase] = useState("fact_collection");
  const [factsSummary, setFactsSummary] = useState(null);
  const [pendingMaterials, setPendingMaterials] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "24px";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [input]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const toApiContent = (msg) => {
    if (typeof msg.content === "string") return msg.content;
    if (msg.content?.text) return msg.content.text;
    if (msg.content?.type === "results" && msg.content.parts) {
      const exp = msg.content.parts.find((p) => p.type === "explanation");
      return exp?.text || "[Legal research response]";
    }
    return "[Message]";
  };

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || loading) return;

    setError("");
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setLoading(true);

    try {
      const conv = messages.map((m) => ({ role: m.role, content: toApiContent(m) }));
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversation: conv,
          message: text,
          phase,
          facts_summary: factsSummary,
        }),
      });

      if (!res.ok) {
        const errText = await res.text();
        let errMsg = `Backend error (${res.status})`;
        try {
          const errJson = JSON.parse(errText);
          const d = errJson.detail;
          errMsg = Array.isArray(d)
            ? d.map((e) => e.msg || e.loc?.join(".")).join("; ") || errMsg
            : d || errJson.message || errMsg;
        } catch {
          if (errText) errMsg = errText.slice(0, 200);
        }
        throw new Error(errMsg);
      }

      const data = await res.json();

      if (data.facts_summary) setFactsSummary(data.facts_summary);
      if (data.phase) setPhase(data.phase);
      if (data.materials_to_confirm) {
        setPendingMaterials(data.materials_to_confirm);
      } else {
        setPendingMaterials(null);
      }

      const assistantContent = buildAssistantContent(data);
      setMessages((prev) => [...prev, { role: "assistant", content: assistantContent }]);
    } catch (err) {
      console.error("Chat failed:", err);
      setError(err.message || "Backend not reachable");
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: {
            type: "error",
            text: "Sorry, I couldn't connect. Please try again.",
          },
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const buildAssistantContent = (data, isConfirmResponse = false) => {
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

  const confirmAndIndex = async () => {
    if (!pendingMaterials || loading) return;
    const hasBare = pendingMaterials.bare_acts?.length > 0;
    const hasCase = pendingMaterials.case_laws?.length > 0;
    if (!hasBare && !hasCase) return;

    setLoading(true);
    setError("");
    try {
      const res = await fetch("/chat/confirm-index", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bare_acts: pendingMaterials.bare_acts || [],
          case_laws: pendingMaterials.case_laws || [],
          facts_summary: factsSummary || "",
        }),
      });
      const data = await res.json();
      if (data.success) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: {
              type: "indexed",
              text: data.message || "Materials have been indexed successfully.",
            },
          },
        ]);
        setPendingMaterials(null);
        setPhase("done");

        if (data.response) {
          const fullContent = buildAssistantContent({
            response: data.response,
            message: data.response.explanation,
          });
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: fullContent },
          ]);
        }
      } else {
        setError(data.message || "Indexing failed");
      }
    } catch (err) {
      setError("Failed to index materials");
    } finally {
      setLoading(false);
    }
  };

  const handleNoMoreInfo = () => {
    setInput("I don't have any more information. Please proceed with the research.");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const renderAssistantContent = (content) => {
    if (content.type === "question") {
      return <p className="message-text">{content.text}</p>;
    }
    if (content.type === "error" || content.type === "indexed") {
      return <p className={content.type === "error" ? "message-error" : "message-indexed"}>{content.text}</p>;
    }
    if (content.type === "results" && content.parts) {
      return (
        <div className="message-results">
          {content.parts.map((part) => {
            if (part.type === "explanation") {
              return (
                <div key="explanation" className="message-explanation">
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
    return null;
  };

  return (
    <div className="chat-app">
      <header className="chat-header">
        <div className="chat-header-inner">
          <span className="chat-logo">⚖</span>
          <h1>Nyaymalaw AI</h1>
          {messages.length > 0 && (
            <button
              type="button"
              onClick={() => {
                setMessages([]);
                setPhase("fact_collection");
                setFactsSummary(null);
                setPendingCaseLaws(null);
                setError("");
              }}
              className="new-chat-btn"
            >
              New Chat
            </button>
          )}
        </div>
      </header>

      <main className="chat-main">
        <div className="messages-container">
          {messages.length === 0 && !loading && (
            <div className="chat-welcome">
              <div className="welcome-icon">⚖</div>
              <h2>How can I help you today?</h2>
              <p>
                I'll gather the facts of your legal matter through a few questions, then research
                relevant bare acts and case laws for you.
              </p>
              <div className="suggestions">
                <button type="button" onClick={() => setInput("I have a contract dispute")}>
                  I have a contract dispute
                </button>
                <button type="button" onClick={() => setInput("Land encroachment issue")}>
                  Land encroachment issue
                </button>
                <button type="button" onClick={() => setInput("Specific performance of contract")}>
                  Specific performance of contract
                </button>
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div
              key={`msg-${msg.role}-${i}-${typeof msg.content === "string" ? msg.content?.slice(0, 30) : msg.content?.type || ""}`}
              className={`message message--${msg.role}`}
            >
              <div className="message-avatar">
                {msg.role === "user" ? (
                  <span className="avatar-user">U</span>
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
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {pendingCaseLaws?.length > 0 && (
          <div className="confirm-index-bar">
            <button
              type="button"
              onClick={confirmAndIndex}
              disabled={loading}
              className="confirm-index-btn"
            >
              ✓ Confirm & Index Case Laws
            </button>
          </div>
        )}

        {phase === "fact_collection" && messages.length > 0 && (
          <div className="quick-action-bar">
            <button type="button" onClick={handleNoMoreInfo} className="quick-action-btn">
              I don't have more information
            </button>
          </div>
        )}

        {error && (
          <div className="chat-error">
            <span>⚠</span> {error}
          </div>
        )}

        <div className="chat-input-wrapper">
          <div className="chat-input-container">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                phase === "fact_collection"
                  ? "Share your legal matter or answer the question..."
                  : "Message Nyaymalaw AI..."
              }
              className="chat-input"
              rows={1}
              disabled={loading}
            />
            <button
              type="button"
              onClick={sendMessage}
              disabled={loading || !input.trim()}
              className="chat-send"
              aria-label="Send message"
            >
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M22 2L11 13" />
                <path d="M22 2L15 22L11 13L2 9L22 2Z" />
              </svg>
            </button>
          </div>
          <p className="chat-disclaimer">
            Nyaymalaw AI can make mistakes. Consider checking important information.
          </p>
        </div>
      </main>
    </div>
  );
}

export default App;
