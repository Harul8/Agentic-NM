import { useState } from "react";

function App() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const searchLaw = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setResult(null);

    try {
      const res = await fetch("http://127.0.0.1:8000/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ issue: query })
      });

      const data = await res.json();
      setResult(data);
    } catch (err) {
      alert("Backend not reachable");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: "900px", margin: "40px auto", fontFamily: "Arial" }}>
      <h1>⚖ Nyaymalaw AI</h1>

      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask a legal question..."
        style={{ width: "100%", padding: "12px", fontSize: "16px" }}
      />

      <button
        onClick={searchLaw}
        style={{ marginTop: "12px", padding: "10px 20px" }}
      >
        Search
      </button>

      {loading && <p>Searching legal database...</p>}

      {result && (
        <div style={{ marginTop: "20px" }}>
          <h3>📘 Bare Act Sections</h3>
          {result.bare_act_sections.map((item, i) => (
            <pre key={i} style={{ background: "#f4f4f4", padding: "10px" }}>
              {item.text}
            </pre>
          ))}

          <h3>📚 Case Laws</h3>
          {result.case_laws.map((item, i) => (
            <pre key={i} style={{ background: "#eef", padding: "10px" }}>
              {item.text}
            </pre>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
