from __future__ import annotations

import io
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pypdf import PdfReader
from pydantic import ValidationError

from app import ai_service, prompts
from app.quiz_service import build_plan, generate_quiz
from app.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    QuizGenerateRequest,
    QuizGenerateResponse,
    QuizKind,
)

router = APIRouter(prefix="/api/quizzes", tags=["quizzes"])

PDF_MAX_BYTES = 10 * 1024 * 1024
PDF_MAX_CHARS = 30_000


async def _call_llm(system: str, user: str) -> ai_service.AIResult:
    """The existing LLM client wrapped for generate_quiz (spec section 4)."""
    return await ai_service.complete(
        ai_service.Task.QUIZ,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )


async def _call_llm_small(system: str, user: str) -> str:
    """Cheapest classify call: temperature 0, a handful of tokens."""
    result = await ai_service.complete(
        ai_service.Task.QUIZ,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=40,
    )
    return result.text


@router.post("/generate", response_model=QuizGenerateResponse)
async def generate_quiz_endpoint(
    req: QuizGenerateRequest, request: Request
) -> QuizGenerateResponse:
    return await _generate(req, auth_header=request.headers.get("authorization"))


@router.post("/generate-pdf", response_model=QuizGenerateResponse)
async def generate_quiz_from_pdf(
    file: UploadFile = File(...),
    topic: str = Form(""),
    kind: str = Form(...),
    count: int = Form(5),
    difficulty: str = Form("medium"),
    language: str = Form("python"),
    context: str = Form(""),
    course_id: int = Form(0),
    subject: str = Form(""),
) -> QuizGenerateResponse:
    """Generate a quiz from an uploaded syllabus PDF.

    PRIVACY GUARANTEE: the PDF is processed strictly in request memory for
    this one call. It is never written to disk by this handler, never kept
    in any database, and Starlette's spooled temp copy is closed/deleted
    immediately after the bytes are read. Only the returned quiz travels to
    the client; the server persists nothing at all.
    """
    try:
        data = await file.read()
    finally:
        # Close (and delete) the upload's spooled temp file right away.
        await file.close()

    if len(data) > PDF_MAX_BYTES:
        raise HTTPException(status_code=413, detail="PDF must be 10 MB or smaller.")

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise HTTPException(
                status_code=400,
                detail="That PDF is password protected. Remove the password and retry.",
            )
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001 - skip unreadable pages
                continue
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - malformed PDF
        raise HTTPException(
            status_code=400,
            detail="Could not read that PDF. Make sure it is a valid file.",
        ) from exc
    finally:
        # The raw PDF bytes are dead the moment extraction finishes.
        del data

    source_text = "\n".join(pages)[:PDF_MAX_CHARS].strip()
    del pages
    if not source_text:
        raise HTTPException(
            status_code=400,
            detail="No extractable text in that PDF (it may be a scanned image).",
        )

    try:
        req = QuizGenerateRequest(
            topic=topic.strip()[:300],
            kind=QuizKind(kind),
            count=count,
            difficulty=difficulty,  # type: ignore[arg-type]
            language=language,
            context=context,
            course_id=course_id if course_id > 0 else None,
            subject=subject[:120],
        )
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Invalid quiz options.") from exc

    return await _generate(
        req,
        source_text=source_text,
        auth_header=request.headers.get("authorization"),
    )


async def _generate(
    req: QuizGenerateRequest,
    *,
    source_text: str | None = None,
    auth_header: str | None = None,
) -> QuizGenerateResponse:
    # build_plan decides mode, domain, language, kinds and the exact
    # material server-side; its HTTPExceptions (400/404) pass through.
    plan = await build_plan(req, auth_header, source_text, _call_llm_small)
    try:
        questions, model = await generate_quiz(
            plan,
            n=req.count,
            difficulty=req.difficulty,
            call_llm=_call_llm,
        )
    except RuntimeError as exc:
        # Parser/validation failures and AI outages both land here: a clean
        # 503 instead of a parser exception (or a gateway 502) escaping.
        raise HTTPException(
            status_code=503,
            detail="Quiz could not be generated, please retry.",
        ) from exc
    return QuizGenerateResponse(questions=questions, model=model)


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    if req.kind is QuizKind.MCQ:
        return _evaluate_mcq(req)

    result = await ai_service.complete_json(
        ai_service.Task.CODE, prompts.evaluate_messages(req)
    )
    data = result.data if isinstance(result.data, dict) else {}
    score = _clamp_score(data.get("score", 0))
    return EvaluateResponse(
        is_correct=bool(data.get("is_correct", score >= 100)),
        score=score,
        feedback=str(data.get("feedback", "")).strip(),
        model=result.model,
    )


def _evaluate_mcq(req: EvaluateRequest) -> EvaluateResponse:
    correct = req.correct_index
    given = req.answer
    is_correct = isinstance(given, int) and correct is not None and given == correct
    return EvaluateResponse(
        is_correct=is_correct,
        score=100.0 if is_correct else 0.0,
        feedback="Correct." if is_correct else "Not quite - reread the options and try again.",
        model=None,
    )


def _clamp_score(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(100.0, numeric))
