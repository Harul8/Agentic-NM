"""
Core chat-processing routes.

/search                     POST
/upload-document            POST
/submit_case                POST  (+ /stream)
/interview_step             POST  (+ /stream)
/conversation/continue      POST  (+ /stream)
/agent/stream               POST
"""
import asyncio
import json
import re
import threading
import time
import logging
import os
from queue import Queue, Empty
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from api.deps import (
    _user_from_token,
    _enforce_query_limit,
    logger,
)
from platform_pkg.tiers import increment_query_count

# Stage 5 (optional)
try:
    from agents.intake.stage5_draft import build_legal_draft as _build_legal_draft
    _STAGE5_ENABLED = True
except Exception:
    _STAGE5_ENABLED = False
    _build_legal_draft = None

# Feedback logging (non-critical)
try:
    from platform_pkg.feedback.logger import log_interaction as _log_interaction
    _FEEDBACK_ENABLED = True
except Exception:
    _FEEDBACK_ENABLED = False
    _log_interaction = None  # type: ignore[assignment]

_PIPELINE_TIMING = os.environ.get("PIPELINE_TIMING", "").lower() in ("1", "true", "yes")


def _log_pipeline_step(step_name: str, elapsed_ms: float, extra: str = "") -> None:
    if _PIPELINE_TIMING:
        msg = f"PIPELINE_TIMING api.{step_name}: {elapsed_ms:.0f} ms"
        if extra:
            msg += f" | {extra}"
        logger.info(msg)


router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SearchQuery(BaseModel):
    issue: str


class ChatMessage(BaseModel):
    role: str
    content: str | dict


class ChatRequest(BaseModel):
    conversation: list[ChatMessage]
    message: str
    phase: str = "fact_collection"
    facts_summary: str | None = None


class SubmitCaseRequest(BaseModel):
    text: str = ""
    mode: str | None = None
    model_override: str | None = None
    workflowState: dict = Field(default_factory=dict)


class QAPair(BaseModel):
    question: str = ""
    answer: str = ""


class InterviewStepRequest(BaseModel):
    facts: str = ""
    qa_history: list[QAPair] = []
    mode: str | None = None
    model_override: str | None = None
    workflowState: dict = Field(default_factory=dict)


class ContinueChatRequest(BaseModel):
    conversation: list[ChatMessage] = []
    message: str = ""
    mode: str | None = None
    model_override: str | None = None
    workflowState: dict = Field(default_factory=dict)


class AgentRequest(BaseModel):
    message: str
    conversation: Optional[List] = Field(default_factory=list)
    workflowState: Optional[dict] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_content(c):
    """Ensure content is string for backend processing."""
    if isinstance(c, str):
        return c
    if isinstance(c, dict) and "text" in c:
        return c["text"]
    if isinstance(c, dict) and c.get("type") == "final_opinion":
        return c.get("opinionText") or ""
    if isinstance(c, dict) and c.get("type") == "results":
        for p in c.get("parts", []):
            if p.get("type") == "explanation":
                return p.get("text", "[Response]")
        return "[Legal research response]"
    return str(c) if c else ""


def _build_conv(messages: list[ChatMessage] | None) -> list[dict]:
    if not messages:
        return []
    return [{"role": m.role, "content": _normalize_content(m.content)} for m in messages]


def _chat_error_fallback(detail: str = "") -> dict:
    try:
        from platform_pkg.llm import ask_llm
        error_context = f" (Technical detail: {detail})" if detail else ""
        msg = ask_llm(
            f"You are a legal assistant. Something went wrong while processing the user's request.{error_context} "
            "Write a short, friendly one-sentence apology to the user asking them to try again. Do not mention technical details."
        ).strip()
        if msg:
            return {"status": "question", "next_question": msg, "retrieved": []}
    except Exception:
        pass
    return {
        "status": "question",
        "next_question": "Something went wrong. Please try again.",
        "retrieved": [],
    }


def _fire_feedback_log(result: dict, facts: str, session_ref: str = "") -> None:
    if not _FEEDBACK_ENABLED or not _log_interaction:
        return
    if result.get("phase") != "done" or not result.get("response"):
        return
    resp = result.get("response") or {}
    response_type = (result.get("response_type") or "").strip().lower()
    if response_type in {"search_results", "lookup_results"}:
        router_classification = "Direct search/lookup"
    elif response_type == "generic_chat":
        router_classification = "Non Legal"
    else:
        router_classification = "Legal Opinion"

    disputes_list = [d.get("dispute", "") for d in (resp.get("disputes") or []) if d.get("dispute")]
    if not disputes_list:
        disputes_list = [str(result.get("facts_summary", "")[:80])]

    bare_acts = resp.get("bare_act_sections") or []
    sections_list = [
        f"{ba.get('act_name','?')} § {ba.get('section_number','?')}"
        for ba in bare_acts
    ]

    case_laws_all = (resp.get("case_laws") or []) + (resp.get("internet_case_laws") or [])
    cl_list = [
        (cl.get("case_name") or cl.get("title") or cl.get("citation") or "?")
        for cl in case_laws_all
    ]

    explanation = (resp.get("explanation") or "").strip()
    followup = (resp.get("followup_question") or result.get("followup_question") or "").strip()

    def _do_log():
        try:
            _log_interaction(
                facts=facts,
                followup_question=followup,
                disputes=disputes_list,
                sections=sections_list,
                case_laws=cl_list,
                legal_opinion=explanation,
                router_classification=router_classification,
                session_ref=session_ref,
            )
        except Exception as _le:
            logger.warning("Feedback log failed (non-critical): %s", _le)

    t = threading.Thread(target=_do_log, daemon=True, name="feedback-log")
    t.start()


def _fire_feedback_log_research(result: dict, query: str, session_ref: str = "") -> None:
    if not _FEEDBACK_ENABLED or not _log_interaction:
        return

    bare_acts = result.get("bare_act_sections") or []
    sections_list = [
        f"{ba.get('act_name', '?')} § {ba.get('section_number', '?')}"
        for ba in bare_acts
        if isinstance(ba, dict)
    ]
    case_laws_raw = result.get("case_laws") or []
    cl_list = [
        (cl.get("case_name") or cl.get("title") or cl.get("citation") or "?")
        for cl in case_laws_raw
        if isinstance(cl, dict)
    ]

    def _do_log():
        try:
            _log_interaction(
                facts=query,
                followup_question="",
                disputes=[query[:80]] if query else [],
                sections=sections_list,
                case_laws=cl_list,
                legal_opinion="",
                router_classification="Direct search/lookup",
                session_ref=session_ref,
            )
        except Exception as _le:
            logger.warning("Feedback log (research) failed (non-critical): %s", _le)

    t = threading.Thread(target=_do_log, daemon=True, name="feedback-log-research")
    t.start()


def _safe_build_advocate_review(
    bare_act_sections: list,
    facts_summary: str,
    intake_state: dict | None = None,
) -> dict | None:
    if not _STAGE5_ENABLED or not _build_legal_draft:
        return None
    if not bare_act_sections:
        return None
    try:
        result = _build_legal_draft(
            intake_state=intake_state,
            bare_act_sections=bare_act_sections,
            facts_summary=facts_summary or "",
        )
        return result.get("advocate_review")
    except Exception as _ar_err:
        logger.warning("Stage5 advocate_review build failed: %s", _ar_err)
        return None


def _map_chat_result_to_ui(result: dict, pre_draft_msg: str = "") -> dict:
    """Map process_chat result to the shape the frontend expects."""
    from platform_pkg.llm import get_last_model_used
    phase = result.get("phase")
    response_type = result.get("response_type")
    analysis_stage = result.get("analysis_stage") or ""
    facts_summary = (result.get("facts_summary") or "").strip()

    def _safe_next_question(text: str) -> str:
        candidate = (text or "").strip()
        normalized = re.sub(r"[\s\W_]+", "", candidate)
        if len(normalized) < 6:
            return "Please share one more important detail, or say 'proceed' if you want me to identify the applicable bare act sections."
        return candidate

    if phase == "fact_collection":
        next_q = _safe_next_question(result.get("message") or "")
        return {
            "status": "question",
            "next_question": next_q,
            "retrieved": result.get("response") or [],
            "model_used": get_last_model_used(),
            "analysis_stage": analysis_stage,
            "facts_summary": facts_summary,
            "intake_state": result.get("intake_state") or None,
        }
    if phase == "done" and result.get("response"):
        resp = result["response"]
        bare_acts = resp.get("bare_act_sections") or []
        case_laws = resp.get("case_laws") or []
        internet_case_laws = resp.get("internet_case_laws") or []
        all_case_laws = case_laws + internet_case_laws
        effective_greeting = pre_draft_msg.strip() or (result.get("message") or "").strip()
        explanation = (resp.get("explanation") or "").strip()
        if effective_greeting and explanation:
            if effective_greeting == explanation or effective_greeting in explanation:
                combined_text = explanation
            elif explanation in effective_greeting:
                combined_text = effective_greeting
            else:
                combined_text = f"{effective_greeting}\n\n{explanation}"
        else:
            combined_text = effective_greeting or explanation
        if not combined_text or len(combined_text.strip()) < 20:
            combined_text = "I've prepared an initial response based on the information currently available."
        separate_case_laws = []
        if bare_acts and len(bare_acts) > 0:
            has_nested_case_laws = any(ba.get("related_case_laws") for ba in bare_acts)
            if not has_nested_case_laws:
                separate_case_laws = all_case_laws
        advocate_review = _safe_build_advocate_review(
            bare_act_sections=bare_acts,
            facts_summary=facts_summary,
        )
        return {
            "status": "done",
            "response_type": response_type or "legal_opinion",
            "opinion_text": combined_text,
            "bare_acts": bare_acts,
            "case_laws": separate_case_laws,
            "next_steps": resp.get("next_steps") or [],
            "next_steps_summary": (resp.get("next_steps_summary") or "").strip(),
            "retrieved": all_case_laws + bare_acts,
            "progress": resp.get("progress"),
            "model_used": get_last_model_used(),
            "analysis_stage": analysis_stage,
            "facts_summary": facts_summary,
            "advocate_review": advocate_review,
        }
    if phase == "done":
        return {
            "status": "done",
            "response_type": response_type or "legal_opinion",
            "opinion_text": (result.get("message") or "").strip() or "Your request has been processed.",
            "bare_acts": [],
            "case_laws": [],
            "next_steps": [],
            "next_steps_summary": "",
            "retrieved": [],
            "model_used": get_last_model_used(),
            "analysis_stage": analysis_stage,
            "facts_summary": facts_summary,
        }
    next_q = _safe_next_question(result.get("message") or "")
    return {
        "status": "question",
        "next_question": next_q,
        "retrieved": result.get("response") or [],
        "model_used": get_last_model_used(),
        "analysis_stage": analysis_stage,
        "facts_summary": facts_summary,
    }


def _normalize_workflow_state_for_chat(raw_state: Optional[dict], messages=None) -> dict:
    """Re-export so this module doesn't need to import from auth_routes."""
    from api.auth_routes import _normalize_workflow_state
    return _normalize_workflow_state(raw_state, messages)


async def _stream_sse_queue(queue: Queue, loop):
    """Drain a worker queue into SSE events with heartbeats."""
    while True:
        try:
            kind, payload = await loop.run_in_executor(None, lambda: queue.get(timeout=1.0))
        except Empty:
            yield ": ping\n\n"
            continue
        except Exception:
            break
        if kind == "result":
            yield f"event: done\ndata: {json.dumps(payload)}\n\n"
            break
        if kind == "progress":
            yield f"event: progress\ndata: {json.dumps(payload)}\n\n"
            continue
        if kind == "step":
            yield f"event: step\ndata: {json.dumps(payload)}\n\n"
            continue
        if kind == "token":
            yield f"event: token\ndata: {json.dumps(payload)}\n\n"


_UPLOAD_MAX_BYTES = 20 * 1024 * 1024  # 20 MB


# ---------------------------------------------------------------------------
# Worker functions for streaming endpoints
# ---------------------------------------------------------------------------

def _run_submit_case_with_progress(
    text: str, queue: Queue, user_id, mode=None, model_override=None, workflow_state=None
) -> None:
    from pipeline.chat import process_chat
    try:
        queue.put(("step", {"message": "Reviewing the facts you shared", "icon": ""}))

        def progress_callback(s): queue.put(("progress", s))
        def step_callback(s): queue.put(("step", s))
        def token_callback(t): queue.put(("token", {"content": t}))

        conv = [{"role": "user", "content": text}]
        result = process_chat(
            conversation=conv, current_message=text, phase="fact_collection", facts_summary=None,
            progress_callback=progress_callback, chat_mode=mode, step_callback=step_callback,
            token_callback=token_callback, model_override=model_override, workflow_state=workflow_state,
        )
        pre_draft_msg = ""
        if result.get("phase") == "response_generation" and result.get("facts_summary"):
            pre_draft_msg = result.get("message", "")
            if pre_draft_msg.strip():
                conv = conv + [{"role": "assistant", "content": pre_draft_msg}]
            s2_wf = {
                **(workflow_state or {}),
                "intakeState": result.get("intake_state") or (workflow_state or {}).get("intakeState"),
                "preliminaryRetrievalContext": result.get("preliminary_retrieval_context") or (workflow_state or {}).get("preliminaryRetrievalContext"),
            }
            result = process_chat(
                conversation=conv, current_message=result["facts_summary"],
                phase="response_generation", facts_summary=result["facts_summary"],
                intent=result.get("intent", "legal_opinion"),
                document_types=result.get("document_types", "both"),
                search_strategy=result.get("search_strategy", "local_then_web"),
                result_count=result.get("result_count"),
                progress_callback=progress_callback, chat_mode=mode,
                step_callback=step_callback, token_callback=token_callback,
                model_override=model_override, workflow_state=s2_wf,
                analysis_mode=result.get("analysis_mode"),
            )
        if result.get("phase") == "done":
            increment_query_count(user_id)
            _fire_feedback_log(result, facts=text, session_ref=str(user_id))
        queue.put(("result", _map_chat_result_to_ui(result, pre_draft_msg=pre_draft_msg)))
    except Exception as e:
        logger.exception("Stream submit_case failed")
        queue.put(("result", _chat_error_fallback(str(e)[:200])))


def _run_interview_step_with_progress(
    facts: str, qa_history: list, queue: Queue, user_id,
    mode=None, model_override=None, workflow_state=None
) -> None:
    from pipeline.chat import process_chat
    try:
        t_total = time.perf_counter()
        queue.put(("step", {"message": "Reviewing your latest answer", "icon": ""}))

        def progress_callback(s): queue.put(("progress", s))
        def step_callback(s): queue.put(("step", s))
        def token_callback(t): queue.put(("token", {"content": t}))

        conv = [{"role": "user", "content": facts}]
        for qa in qa_history:
            conv.append({"role": "assistant", "content": qa.question})
            conv.append({"role": "user", "content": qa.answer})
        current_message = qa_history[-1].answer if qa_history else ""
        t_phase = time.perf_counter()
        result = process_chat(
            conversation=conv, current_message=current_message, phase="fact_collection",
            facts_summary=None, progress_callback=progress_callback, chat_mode=mode,
            step_callback=step_callback, token_callback=token_callback,
            model_override=model_override, workflow_state=workflow_state,
        )
        _log_pipeline_step(
            "interview_step.process_chat.fact_collection",
            (time.perf_counter() - t_phase) * 1000,
            f"phase={result.get('phase')}",
        )
        pre_draft_msg = ""
        if result.get("phase") == "response_generation" and result.get("facts_summary"):
            pre_draft_msg = result.get("message", "")
            if pre_draft_msg.strip():
                conv = conv + [{"role": "assistant", "content": pre_draft_msg}]
            s2_wf = {
                **(workflow_state or {}),
                "intakeState": result.get("intake_state") or (workflow_state or {}).get("intakeState"),
                "preliminaryRetrievalContext": result.get("preliminary_retrieval_context") or (workflow_state or {}).get("preliminaryRetrievalContext"),
            }
            t_phase = time.perf_counter()
            result = process_chat(
                conversation=conv, current_message=result["facts_summary"],
                phase="response_generation", facts_summary=result["facts_summary"],
                intent=result.get("intent", "legal_opinion"),
                document_types=result.get("document_types", "both"),
                search_strategy=result.get("search_strategy", "local_then_web"),
                result_count=result.get("result_count"),
                progress_callback=progress_callback, chat_mode=mode,
                step_callback=step_callback, token_callback=token_callback,
                model_override=model_override, workflow_state=s2_wf,
                analysis_mode=result.get("analysis_mode"),
            )
            _log_pipeline_step(
                "interview_step.process_chat.response_generation",
                (time.perf_counter() - t_phase) * 1000,
                f"phase={result.get('phase')}",
            )
        if result.get("phase") == "done":
            increment_query_count(user_id)
            _fire_feedback_log(result, facts=facts, session_ref=str(user_id))
        queue.put(("result", _map_chat_result_to_ui(result, pre_draft_msg=pre_draft_msg)))
        _log_pipeline_step(
            "interview_step.total",
            (time.perf_counter() - t_total) * 1000,
            f"final_phase={result.get('phase')}",
        )
    except Exception as e:
        logger.exception("Stream interview_step failed")
        queue.put(("result", _chat_error_fallback(str(e)[:200])))


def _run_continue_chat_with_progress(
    conv: list, message: str, queue: Queue, user_id,
    mode=None, model_override=None, workflow_state=None
) -> None:
    from pipeline.chat import process_chat
    try:
        queue.put(("step", {"message": "Starting analysis of your latest message", "icon": ""}))

        def progress_callback(s): queue.put(("progress", s))
        def step_callback(s): queue.put(("step", s))
        def token_callback(t): queue.put(("token", {"content": t}))

        result = process_chat(
            conversation=conv, current_message=message, phase="fact_collection",
            facts_summary=None, progress_callback=progress_callback, chat_mode=mode,
            step_callback=step_callback, token_callback=token_callback,
            model_override=model_override, workflow_state=workflow_state,
        )
        pre_draft_msg = ""
        if result.get("phase") == "response_generation" and result.get("facts_summary"):
            pre_draft_msg = result.get("message", "")
            if pre_draft_msg.strip():
                conv = conv + [{"role": "user", "content": message}, {"role": "assistant", "content": pre_draft_msg}]
            s2_wf = {
                **(workflow_state or {}),
                "intakeState": result.get("intake_state") or (workflow_state or {}).get("intakeState"),
                "preliminaryRetrievalContext": result.get("preliminary_retrieval_context") or (workflow_state or {}).get("preliminaryRetrievalContext"),
            }
            result = process_chat(
                conversation=conv, current_message=result["facts_summary"],
                phase="response_generation", facts_summary=result["facts_summary"],
                intent=result.get("intent", "legal_opinion"),
                document_types=result.get("document_types", "both"),
                search_strategy=result.get("search_strategy", "local_then_web"),
                result_count=result.get("result_count"),
                progress_callback=progress_callback, chat_mode=mode,
                step_callback=step_callback, token_callback=token_callback,
                model_override=model_override, workflow_state=s2_wf,
                analysis_mode=result.get("analysis_mode"),
            )
        if result.get("phase") == "done":
            increment_query_count(user_id)
            _fire_feedback_log(result, facts=message, session_ref=str(user_id))
        queue.put(("result", _map_chat_result_to_ui(result, pre_draft_msg=pre_draft_msg)))
    except Exception as e:
        logger.exception("Stream continue_chat failed")
        queue.put(("result", _chat_error_fallback(str(e)[:200])))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/search")
def search_law(query: SearchQuery, background_tasks: BackgroundTasks):
    """Legacy search endpoint — returns bare acts + case laws."""
    from retrieval.retriever import fuse_bare_act_and_case_law
    result = fuse_bare_act_and_case_law.run(issue=query.issue)
    background_tasks.add_task(
        _fire_feedback_log_research,
        result if isinstance(result, dict) else {},
        query.issue,
        "",
    )
    return result


@router.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    user: dict = Depends(_user_from_token),
):
    """Extract plain text from an uploaded document or image (PDF, DOCX, or image)."""
    import io
    import base64
    from platform_pkg.llm import ocr_pages_with_vision

    _IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "tiff", "tif", "bmp"}
    _IMAGE_MIME = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
        "webp": "image/webp", "tiff": "image/tiff", "tif": "image/tiff",
        "bmp": "image/bmp",
    }
    _SCANNED_CHARS_THRESHOLD = 80

    filename = (file.filename or "").strip()
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    allowed = {"pdf", "docx", "doc"} | _IMAGE_EXTS
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Accepted: PDF (.pdf), Word (.docx), or image (.jpg, .jpeg, .png, .webp, .tiff, .bmp).",
        )

    content = await file.read()
    if len(content) > _UPLOAD_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) // (1024 * 1024)} MB). Maximum upload size is 20 MB.",
        )
    text = ""
    method = "text"

    try:
        if ext in _IMAGE_EXTS:
            mime = _IMAGE_MIME.get(ext, "image/png")
            b64 = base64.b64encode(content).decode()
            text = ocr_pages_with_vision([(b64, mime)])
            method = "vision_ocr"
        elif ext == "pdf":
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            pages_text = [page.extract_text() or "" for page in reader.pages]
            extracted = "\n\n".join(p.strip() for p in pages_text if p.strip())
            num_pages = max(len(reader.pages), 1)
            avg_chars = len(extracted) / num_pages
            if avg_chars >= _SCANNED_CHARS_THRESHOLD:
                text = extracted
                method = "text"
            else:
                import fitz  # pymupdf
                doc = fitz.open(stream=content, filetype="pdf")
                images_b64: list[tuple[str, str]] = []
                for page in doc:
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat)
                    img_bytes = pix.tobytes("png")
                    images_b64.append((base64.b64encode(img_bytes).decode(), "image/png"))
                doc.close()
                text = ocr_pages_with_vision(images_b64)
                method = "vision_ocr"
        else:
            import docx as _docx
            doc = _docx.Document(io.BytesIO(content))
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            text = "\n\n".join(paragraphs)
            method = "text"
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Document extraction failed for %s: %s", filename, exc)
        raise HTTPException(
            status_code=422,
            detail="Could not extract text from the document. The file may be corrupted or unsupported.",
        )

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="No readable text found. For scanned documents this usually means the OpenAI Vision API call failed — check your OPENAI_API_KEY and try again.",
        )
    return {"text": text.strip(), "filename": filename, "char_count": len(text), "method": method}


@router.post("/submit_case")
def submit_case(request: SubmitCaseRequest, user: dict = Depends(_user_from_token)):
    """Initial case submission. Returns status + next_question or opinion."""
    from pipeline.chat import process_chat
    from api.auth_routes import _normalize_workflow_state
    _enforce_query_limit(user)
    text = (request.text or "").strip()
    mode = (request.mode or "").strip().lower() or None
    model_override = (request.model_override or "").strip() or None
    if not text:
        return {"status": "question", "next_question": "", "retrieved": []}
    try:
        workflow_state = _normalize_workflow_state(request.workflowState)
        conv = [{"role": "user", "content": text}]
        result = process_chat(
            conversation=conv, current_message=text, phase="fact_collection",
            facts_summary=None, chat_mode=mode, model_override=model_override,
            workflow_state=workflow_state,
        )
        pre_draft_msg = ""
        if result.get("phase") == "response_generation" and result.get("facts_summary"):
            pre_draft_msg = result.get("message", "")
            if pre_draft_msg.strip():
                conv = conv + [{"role": "assistant", "content": pre_draft_msg}]
            s2_wf = {
                **(workflow_state or {}),
                "intakeState": result.get("intake_state") or (workflow_state or {}).get("intakeState"),
                "preliminaryRetrievalContext": result.get("preliminary_retrieval_context") or (workflow_state or {}).get("preliminaryRetrievalContext"),
            }
            result = process_chat(
                conversation=conv, current_message=result["facts_summary"],
                phase="response_generation", facts_summary=result["facts_summary"],
                intent=result.get("intent", "legal_opinion"),
                document_types=result.get("document_types", "both"),
                search_strategy=result.get("search_strategy", "local_then_web"),
                result_count=result.get("result_count"), chat_mode=mode,
                model_override=model_override, workflow_state=s2_wf,
                analysis_mode=result.get("analysis_mode"),
            )
        if result.get("phase") == "done":
            increment_query_count(user["id"])
            _fire_feedback_log(result, facts=text, session_ref=str(user.get("id", "")))
        return _map_chat_result_to_ui(result, pre_draft_msg=pre_draft_msg)
    except Exception as e:
        return _chat_error_fallback(str(e)[:200])


@router.post("/submit_case/stream")
async def submit_case_stream(request: SubmitCaseRequest, user: dict = Depends(_user_from_token)):
    """Same as /submit_case but streams progress via Server-Sent Events."""
    from api.auth_routes import _normalize_workflow_state
    _enforce_query_limit(user)
    text = (request.text or "").strip()
    mode = (request.mode or "").strip().lower() or None
    model_override = (request.model_override or "").strip() or None
    if not text:
        return JSONResponse(status_code=400, content={"detail": "text is required"})
    queue: Queue = Queue()
    loop = asyncio.get_event_loop()
    user_id = user.get("id")
    workflow_state = _normalize_workflow_state(request.workflowState)
    threading.Thread(
        target=_run_submit_case_with_progress,
        args=(text, queue, user_id, mode, model_override, workflow_state),
        daemon=True,
    ).start()

    async def event_generator():
        async for event in _stream_sse_queue(queue, loop):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/interview_step")
def interview_step(request: InterviewStepRequest, user: dict = Depends(_user_from_token)):
    """Follow-up answer in interview."""
    from pipeline.chat import process_chat
    from api.auth_routes import _normalize_workflow_state
    _enforce_query_limit(user)
    facts = request.facts or ""
    qa_history = request.qa_history or []
    mode = (request.mode or "").strip().lower() or None
    model_override = (request.model_override or "").strip() or None
    if not qa_history:
        return {"status": "question", "next_question": "", "retrieved": []}
    try:
        workflow_state = _normalize_workflow_state(request.workflowState)
        conv = [{"role": "user", "content": facts}]
        for qa in qa_history:
            conv.append({"role": "assistant", "content": qa.question})
            conv.append({"role": "user", "content": qa.answer})
        current_message = qa_history[-1].answer if qa_history else ""
        result = process_chat(
            conversation=conv, current_message=current_message, phase="fact_collection",
            facts_summary=None, chat_mode=mode, model_override=model_override,
            workflow_state=workflow_state,
        )
        pre_draft_msg = ""
        if result.get("phase") == "response_generation" and result.get("facts_summary"):
            pre_draft_msg = result.get("message", "")
            conv = conv + [{"role": "assistant", "content": pre_draft_msg}]
            s2_wf = {
                **(workflow_state or {}),
                "intakeState": result.get("intake_state") or (workflow_state or {}).get("intakeState"),
                "preliminaryRetrievalContext": result.get("preliminary_retrieval_context") or (workflow_state or {}).get("preliminaryRetrievalContext"),
            }
            result = process_chat(
                conversation=conv, current_message=result["facts_summary"],
                phase="response_generation", facts_summary=result["facts_summary"],
                intent=result.get("intent", "legal_opinion"),
                document_types=result.get("document_types", "both"),
                search_strategy=result.get("search_strategy", "local_then_web"),
                result_count=result.get("result_count"), chat_mode=mode,
                model_override=model_override, workflow_state=s2_wf,
                analysis_mode=result.get("analysis_mode"),
            )
        if result.get("phase") == "done":
            increment_query_count(user["id"])
            _fire_feedback_log(result, facts=facts, session_ref=str(user.get("id", "")))
        return _map_chat_result_to_ui(result, pre_draft_msg=pre_draft_msg)
    except Exception as e:
        return _chat_error_fallback(str(e)[:200])


@router.post("/interview_step/stream")
async def interview_step_stream(request: InterviewStepRequest, user: dict = Depends(_user_from_token)):
    """Same as /interview_step but streams progress via Server-Sent Events."""
    from api.auth_routes import _normalize_workflow_state
    _enforce_query_limit(user)
    facts = request.facts or ""
    qa_history = request.qa_history or []
    mode = (request.mode or "").strip().lower() or None
    model_override = (request.model_override or "").strip() or None
    if not qa_history:
        return JSONResponse(status_code=400, content={"detail": "qa_history is required"})
    queue: Queue = Queue()
    loop = asyncio.get_event_loop()
    user_id = user.get("id")
    workflow_state = _normalize_workflow_state(request.workflowState)
    threading.Thread(
        target=_run_interview_step_with_progress,
        args=(facts, qa_history, queue, user_id, mode, model_override, workflow_state),
        daemon=True,
    ).start()

    async def event_generator():
        async for event in _stream_sse_queue(queue, loop):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/conversation/continue")
def continue_chat(request: ContinueChatRequest, user: dict = Depends(_user_from_token)):
    """Continue a conversation from chat history."""
    from pipeline.chat import process_chat
    from api.auth_routes import _normalize_workflow_state
    _enforce_query_limit(user)
    message = (request.message or "").strip()
    mode = (request.mode or "").strip().lower() or None
    model_override = (request.model_override or "").strip() or None
    if not message:
        return {"status": "question", "next_question": "", "retrieved": []}
    try:
        conv = [
            {"role": m.role, "content": _normalize_content(m.content)}
            for m in (request.conversation or [])
        ]
        workflow_state = _normalize_workflow_state(request.workflowState, request.conversation)
        result = process_chat(
            conversation=conv, current_message=message, phase="fact_collection",
            facts_summary=None, chat_mode=mode, model_override=model_override,
            workflow_state=workflow_state,
        )
        pre_draft_msg = ""
        if result.get("phase") == "response_generation" and result.get("facts_summary"):
            pre_draft_msg = result.get("message", "")
            if pre_draft_msg.strip():
                conv = conv + [{"role": "user", "content": message}, {"role": "assistant", "content": pre_draft_msg}]
            s2_wf = {
                **(workflow_state or {}),
                "intakeState": result.get("intake_state") or (workflow_state or {}).get("intakeState"),
                "preliminaryRetrievalContext": result.get("preliminary_retrieval_context") or (workflow_state or {}).get("preliminaryRetrievalContext"),
            }
            result = process_chat(
                conversation=conv, current_message=result["facts_summary"],
                phase="response_generation", facts_summary=result["facts_summary"],
                intent=result.get("intent", "legal_opinion"),
                document_types=result.get("document_types", "both"),
                search_strategy=result.get("search_strategy", "local_then_web"),
                result_count=result.get("result_count"), chat_mode=mode,
                model_override=model_override, workflow_state=s2_wf,
                analysis_mode=result.get("analysis_mode"),
            )
        if result.get("phase") == "done":
            increment_query_count(user["id"])
            _fire_feedback_log(result, facts=message, session_ref=str(user.get("id", "")))
        return _map_chat_result_to_ui(result, pre_draft_msg=pre_draft_msg)
    except Exception as e:
        return _chat_error_fallback(str(e)[:200])


@router.post("/conversation/continue/stream")
async def continue_chat_stream(request: ContinueChatRequest, user: dict = Depends(_user_from_token)):
    """Same as /conversation/continue but streams progress via Server-Sent Events."""
    from api.auth_routes import _normalize_workflow_state
    _enforce_query_limit(user)
    message = (request.message or "").strip()
    mode = (request.mode or "").strip().lower() or None
    model_override = (request.model_override or "").strip() or None
    if not message:
        return JSONResponse(status_code=400, content={"detail": "message is required"})
    conv = [
        {"role": m.role, "content": _normalize_content(m.content)}
        for m in (request.conversation or [])
    ]
    queue: Queue = Queue()
    loop = asyncio.get_event_loop()
    user_id = user.get("id")
    workflow_state = _normalize_workflow_state(request.workflowState, request.conversation)
    threading.Thread(
        target=_run_continue_chat_with_progress,
        args=(conv, message, queue, user_id, mode, model_override, workflow_state),
        daemon=True,
    ).start()

    async def event_generator():
        async for event in _stream_sse_queue(queue, loop):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/agent/stream")
async def agent_stream(request: AgentRequest, user: dict = Depends(_user_from_token)):
    """
    Agentic endpoint. Streams SSE events from the OrchestratorAgent.
    Events: step, token, done, error.
    """
    _enforce_query_limit(user)
    message = (request.message or "").strip()
    if not message:
        return JSONResponse(status_code=400, content={"detail": "message is required"})

    conv = [
        {
            "role": m["role"] if isinstance(m, dict) else m.role,
            "content": _normalize_content(m["content"] if isinstance(m, dict) else m.content),
        }
        for m in (request.conversation or [])
    ]
    workflow_state = dict(request.workflowState or {})
    queue: Queue = Queue()
    loop = asyncio.get_event_loop()

    def run_in_thread():
        try:
            from agents.orchestrator import OrchestratorAgent
            agent = OrchestratorAgent()
            for event in agent.run_stream(message, conv, workflow_state):
                ev_type = event.get("type", "step")
                if ev_type == "step":
                    queue.put(("step", {"message": event.get("message", "")}))
                elif ev_type == "token":
                    queue.put(("token", {"text": event.get("text", "")}))
                elif ev_type == "done":
                    queue.put(("result", event.get("payload", {})))
                    return
                elif ev_type == "error":
                    queue.put(("step", {"message": f"⚠ {event.get('message', 'Error')}"}))
            queue.put(("result", {"reply": "", "workflow_state": workflow_state}))
        except Exception as exc:
            logger.exception("agent_stream worker failed: %s", exc)
            queue.put(("result", {"reply": "An error occurred. Please try again.", "error": str(exc)}))

    threading.Thread(target=run_in_thread, daemon=True).start()

    async def event_generator():
        async for event in _stream_sse_queue(queue, loop):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
