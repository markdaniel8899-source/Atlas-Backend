from __future__ import annotations

from fastapi import APIRouter

from app import ai_service, prompts, roadmap_graph
from app.schemas import (
    RoadmapOutlinePhase,
    RoadmapOutlineResponse,
    RoadmapPhase,
    RoadmapPhaseRequest,
    RoadmapPhaseResponse,
    RoadmapRequest,
)

router = APIRouter(prefix="/api/roadmaps", tags=["roadmaps"])

MAX_ATTEMPTS = 3


def _clean_outline(data: object, fallback_title: str) -> tuple[str, str, list[RoadmapOutlinePhase]]:
    """Validate the outline JSON: {title, summary, phases:[{number,title,objective}]}."""
    if not isinstance(data, dict):
        raise roadmap_graph.RoadmapShapeError("expected a JSON object")
    raw_phases = data.get("phases")
    if not isinstance(raw_phases, list) or not raw_phases:
        raise roadmap_graph.RoadmapShapeError("missing non-empty 'phases' array")

    phases: list[RoadmapOutlinePhase] = []
    for item in raw_phases:
        if not isinstance(item, dict):
            continue
        title = roadmap_graph._text(item.get("title"), 160)
        if not title:
            continue
        objective = roadmap_graph._text(item.get("objective"), 600)
        phases.append(
            RoadmapOutlinePhase(
                number=len(phases) + 1,
                title=title,
                objective=objective,
            )
        )
        if len(phases) >= 8:
            break
    if not phases:
        raise roadmap_graph.RoadmapShapeError("no usable phases were returned")

    title = roadmap_graph._text(data.get("title"), 200) or fallback_title[:200]
    summary = roadmap_graph._text(data.get("summary"), 1000)
    return title, summary, phases


def _extract_levels(data: object) -> list[dict]:
    """Find the 5-6 level objects: root 'levels', nested phase, or full phases."""
    if not isinstance(data, dict):
        raise roadmap_graph.RoadmapShapeError("expected a JSON object")

    direct = data.get("levels")
    if isinstance(direct, list) and direct:
        return [item for item in direct if isinstance(item, dict)]

    for value in data.values():
        if isinstance(value, dict):
            inner = value.get("levels")
            if isinstance(inner, list) and inner:
                return [item for item in inner if isinstance(item, dict)]
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, dict):
                    inner = entry.get("levels")
                    if isinstance(inner, list) and inner:
                        return [item for item in inner if isinstance(item, dict)]
    raise roadmap_graph.RoadmapShapeError("missing non-empty 'levels' array")


@router.post("/outline", response_model=RoadmapOutlineResponse)
async def generate_outline(req: RoadmapRequest) -> RoadmapOutlineResponse:
    """Step 1: phase titles only - returned to the UI immediately.

    Small request/ response, so it never times out or hits TPM limits; the
    heavy level detail is fetched chunk-by-chunk on /phase.
    """
    correction = ""
    last_error = ""

    for _attempt in range(MAX_ATTEMPTS):
        result = await ai_service.complete_json(
            ai_service.Task.ROADMAP,
            prompts.roadmap_outline_messages(req, correction=correction),
            prefer=("phases",),
        )
        try:
            title, summary, phases = _clean_outline(result.data, req.goal)
        except roadmap_graph.RoadmapShapeError as exc:
            last_error = str(exc)
            correction = (
                "Your previous reply was not usable: "
                f"{last_error}. Return ONLY the JSON object described above: "
                "root key 'phases' with 4-5 objects, each having number, "
                "title and objective. No markdown, no commentary."
            )
            continue
        return RoadmapOutlineResponse(
            title=title, summary=summary, phases=phases, model=result.model
        )

    raise ai_service.AIServiceError(
        f"The model could not produce a valid outline ({last_error}). "
        "Try a shorter or more specific goal.",
        status_code=502,
    )


@router.post("/phase", response_model=RoadmapPhaseResponse)
async def generate_phase(req: RoadmapPhaseRequest) -> RoadmapPhaseResponse:
    """Step 2+: detailed levels for ONE phase - one small, fast chunk."""
    target = next((p for p in req.outline if p.number == req.phase_number), None)
    if target is None:
        raise ai_service.AIServiceError(
            f"phase_number {req.phase_number} is not part of the outline.",
            status_code=422,
        )

    correction = ""
    last_error = ""

    for _attempt in range(MAX_ATTEMPTS):
        result = await ai_service.complete_json(
            ai_service.Task.ROADMAP,
            prompts.roadmap_phase_messages(req, correction=correction),
            prefer=("levels", "phases"),
        )
        try:
            levels = _extract_levels(result.data)
            # Wrap the chunk so the lenient normalizer validates it, then put
            # the real phase number back (normalize renumbers a lone phase to 1).
            wrapped = {
                "phases": [
                    {
                        "number": target.number,
                        "title": target.title,
                        "objective": target.objective,
                        "levels": levels,
                    }
                ]
            }
            payload = roadmap_graph.normalize_roadmap(
                wrapped, hours_per_week=req.hours_per_week
            )
            if not payload.phases:
                raise roadmap_graph.RoadmapShapeError("no usable phases were returned")
            phase = payload.phases[0]
            phase = _renumber_phase(phase, target)
        except roadmap_graph.RoadmapShapeError as exc:
            last_error = str(exc)
            correction = (
                "Your previous reply was not usable: "
                f"{last_error}. Return ONLY raw JSON with a root key 'levels' "
                f"holding 5-6 level objects for phase {target.number} "
                f"('{target.title}'). The last level must be the boss gate. "
                "No markdown fences, no commentary."
            )
            continue
        return RoadmapPhaseResponse(phase=phase, model=result.model)

    raise ai_service.AIServiceError(
        f"The model could not produce phase {req.phase_number} ({last_error}). "
        "Try again.",
        status_code=502,
    )


def _renumber_phase(phase: RoadmapPhase, target: RoadmapOutlinePhase) -> RoadmapPhase:
    """Give the chunk its real phase number (normalize renumbers to 1)."""
    nodes = [
        node.model_copy(update={"phase_number": target.number})
        for node in phase.nodes
    ]
    return phase.model_copy(
        update={
            "number": target.number,
            "title": target.title,
            "objective": target.objective or phase.objective,
            "nodes": nodes,
        }
    )
