
### README ###

This system is designed to:

* Collect case facts from users in **plain language**
* Interactively clarify missing or contradictory facts
* Ingest **Bare Acts** and **Judgments / Case Law**
* Identify **relevant sections and precedents**
* Generate a **reasoned legal opinion** with citations
* Draft **court-ready legal documents**
* Ensure **zero hallucination** and **full citation traceability**
* Enforce **human approval before finalization**

All while running **entirely on a local machine** using **free and open-source tools**.

---

## 🧩 Core Features

✔ Structured **multi-agent orchestration**
✔ Local **LLM inference via Ollama**
✔ **Vector search** over Bare Acts & Judgments
✔ Plain-language client interaction
✔ Legal issue framing & section applicability checks
✔ Precedent ranking & counter-argument analysis
✔ Court-specific drafting & formatting
✔ Hallucination detection & citation validation
✔ Human-in-the-loop safety gate

---
## 🛠️ Technology Stack 
| Component           | Tool                  |
| ------------------- | --------------------- |
| Agent Orchestration | CrewAI                |
| LLM Hosting         | Ollama                |
| Embeddings          | sentence-transformers |
| Vector Store        | FAISS                 |
| UI                  | Streamlit             |
| PDF Parsing         | pdfplumber / PyMuPDF  |
| OCR (optional)      | Tesseract             |
| Language            | Python 3.10+          |

---
legal-ai-system/
│
├── app.py
├── crew.py
│
├── agents/
│
│   # 🧠 Fact Intelligence
│   ├── facts_agent.py
│   ├── normalization_agent.py
│   ├── contradiction_agent.py
│
│   # 📚 Legal Research
│   ├── bare_act_agent.py
│   ├── case_law_agent.py
│   ├── applicability_agent.py
│   ├── precedent_ranking_agent.py
│
│   # ⚖️ Legal Reasoning
│   ├── issue_framing_agent.py
│   ├── jurisdiction_agent.py
│   ├── opinion_agent.py
│   ├── counter_argument_agent.py
│
│   # 📝 Drafting & Output
│   ├── drafting_agent.py
│   ├── formatting_agent.py
│
│   # 🛡️ Safety & QA
│   ├── citation_agent.py
│   ├── hallucination_agent.py
│   ├── confidence_agent.py
│   └── gatekeeper_agent.py
│
├── ingestion/
│   ├── pdf_loader.py
│   ├── chunker.py
│   ├── embedder.py
│   └── indexer.py
│
├── retrieval/
│   └── faiss_retriever.py
│
├── llm/
│   └── ollama_client.py
│
├── templates/
│   ├── petition.md
│   ├── affidavit.md
│   └── memo.md
│
├── data/
│   ├── raw_pdfs/
│   ├── vector_store/
│   └── metadata.sqlite
│
├── requirements.txt
└── README.md

---

## 🔄 Execution Flow

1. User enters case facts (plain language)
2. Intake agents clarify & normalize facts
3. Documents are uploaded and indexed
4. Issue framing agent identifies legal issues
5. Research agents retrieve relevant law & cases
6. Applicability & precedent ranking agents validate relevance
7. Legal opinion agent synthesizes reasoning
8. Drafting agents prepare court documents
9. QA agents verify citations & detect hallucinations
10. Human gatekeeper approves or blocks output

---

## 🛡️ Safety & Compliance

### Mandatory Controls

* ❌ No uncited legal statements allowed
* ❌ No automatic filing or submission
* ✅ Every output includes source citations
* ✅ Human approval required before final draft

### Ethical Guardrails

* No advice beyond informational/legal drafting assistance
* Clear uncertainty disclosure
* Audit logs for all agent actions

---


🧠 Fact Intelligence Agents

Facts Collector Agent

Fact Normalizer Agent

Contradiction Checker Agent

📚 Bare Act & Legal Research Agents

Bare Act Research Agent

Case Law Research Agent

Section Applicability Agent

Precedent Ranking Agent

⚖️ Legal Reasoning Agents

Issue Framing Agent

Jurisdiction & Maintainability Agent

Legal Opinion Agent

Counter-Argument Agent

📝 Drafting & Output Agents

Drafting Agent

Formatting & Court Style Agent

🛡️ Safety & QA Agents

Citation Verification Agent

Hallucination Detection Agent

Confidence Scoring Agent

Human Gatekeeper Agent