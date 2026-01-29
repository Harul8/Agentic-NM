from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from agents.Legal_Research.act_case_fusion_agent import fuse_bare_act_and_case_law
from services.interactive_chat import process_chat
from services.case_law_indexer_incremental import index_new_case_laws
from services.bare_act_indexer_incremental import index_new_bare_acts
from services.response_generator import generate_response

app = FastAPI(title="Nyaymalaw API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    issue: str


class ChatMessage(BaseModel):
    role: str
    content: str | dict  # str for API, dict if frontend sends object


class ChatRequest(BaseModel):
    conversation: list[ChatMessage]
    message: str
    phase: str = "fact_collection"
    facts_summary: str | None = None


class ConfirmIndexRequest(BaseModel):
    bare_acts: list[dict] = []
    case_laws: list[dict] = []
    facts_summary: str = ""


@app.post("/search")
def search_law(query: Query):
    """Legacy search endpoint - single query, returns bare acts + case laws."""
    result = fuse_bare_act_and_case_law.run(issue=query.issue)
    return result


def _normalize_content(c):
    """Ensure content is string for backend processing."""
    if isinstance(c, str):
        return c
    if isinstance(c, dict) and "text" in c:
        return c["text"]
    if isinstance(c, dict) and c.get("type") == "results":
        for p in c.get("parts", []):
            if p.get("type") == "explanation":
                return p.get("text", "[Response]")
        return "[Legal research response]"
    return str(c) if c else ""


@app.post("/chat")
def chat(request: ChatRequest):
    """
    Interactive chat endpoint.
    Phase: fact_collection | response_generation | confirm_index
    """
    conv = [
        {"role": m.role, "content": _normalize_content(m.content)}
        for m in request.conversation
    ]
    result = process_chat(
        conversation=conv,
        current_message=request.message,
        phase=request.phase,
        facts_summary=request.facts_summary,
    )

    # If fact collection complete, trigger response generation
    if result["phase"] == "response_generation" and result.get("facts_summary"):
        resp_result = process_chat(
            conversation=conv,
            current_message=result["facts_summary"],
            phase="response_generation",
            facts_summary=result["facts_summary"],
        )
        return resp_result

    return result


@app.post("/chat/confirm-index")
def confirm_index(request: ConfirmIndexRequest):
    """
    Index user-confirmed bare acts and case laws into the vector store,
    then generate and return the full legal research response.
    """
    bare_result = {"success": True, "chunks_added": 0}
    case_result = {"success": True, "chunks_added": 0}

    if request.bare_acts:
        bare_result = index_new_bare_acts(request.bare_acts)
    if request.case_laws:
        case_result = index_new_case_laws(request.case_laws)

    if not bare_result.get("success") and not case_result.get("success"):
        return {
            "success": False,
            "message": bare_result.get("message", "") or case_result.get("message", ""),
            "response": None,
        }

    # Generate full response with confirmed materials
    confirmed = {
        "bare_acts": request.bare_acts,
        "case_laws": request.case_laws,
    }
    resp = generate_response(request.facts_summary, confirmed_materials=confirmed)

    return {
        "success": True,
        "message": f"Indexed: {bare_result.get('chunks_added', 0)} bare act chunks, {case_result.get('chunks_added', 0)} case law chunks",
        "response": resp,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
