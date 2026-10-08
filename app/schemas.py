from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class QuizKind(str, Enum):
    MCQ = "mcq"
    CODE = "code"
    DEBUG = "debug"
    OUTPUT = "output"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=24)


class ChatResponse(BaseModel):
    reply: str
    model: str


class RoadmapRequest(BaseModel):
    goal: str = Field(min_length=3, max_length=400)
    hours_per_week: int = Field(default=10, ge=1, le=80)
    duration_weeks: int | None = Field(default=None, ge=1, le=104)
    current_level: str = Field(default="beginner", max_length=40)
    topics: list[str] = Field(default_factory=list, max_length=40)


class RoadmapResource(BaseModel):
    type: Literal["video", "article", "book", "course", "practice", "docs"] = "article"
    title: str = Field(default="", max_length=300)
    url: str = Field(default="", max_length=600)


class RoadmapNode(BaseModel):
    key: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=800)
    phase_number: int = Field(default=1, ge=1)
    estimated_hours: int = Field(default=0, ge=0, le=1000)
    estimated_days: int = Field(default=0, ge=0, le=365)
    prerequisites: list[str] = Field(default_factory=list, max_length=40)
    resources: list[RoadmapResource] = Field(default_factory=list, max_length=12)
    type: Literal["regular", "boss"] = "regular"
    quiz_required: bool = False
    objectives: list[str] = Field(default_factory=list, max_length=6)


class RoadmapPhase(BaseModel):
    number: int = Field(default=1, ge=1)
    title: str = Field(min_length=1, max_length=160)
    objective: str = Field(default="", max_length=600)
    duration_weeks: float = Field(default=0, ge=0)
    hours: int = Field(default=0, ge=0, le=5000)
    milestones: list[str] = Field(default_factory=list, max_length=8)
    nodes: list[RoadmapNode] = Field(default_factory=list, max_length=24)


class RoadmapConnection(BaseModel):
    source: str
    target: str


class RoadmapPayload(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(default="", max_length=1000)
    phases: list[RoadmapPhase] = Field(default_factory=list, max_length=16)
    connections: list[RoadmapConnection] = Field(default_factory=list, max_length=400)


class RoadmapResponse(BaseModel):
    roadmap: RoadmapPayload
    model: str


class RoadmapOutlinePhase(BaseModel):
    number: int = Field(ge=1, le=12)
    title: str = Field(min_length=1, max_length=160)
    objective: str = Field(default="", max_length=600)


class RoadmapOutlineResponse(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(default="", max_length=1000)
    phases: list[RoadmapOutlinePhase] = Field(min_length=1, max_length=8)
    model: str


class RoadmapPhaseRequest(BaseModel):
    goal: str = Field(min_length=3, max_length=400)
    hours_per_week: int = Field(default=10, ge=1, le=80)
    current_level: str = Field(default="beginner", max_length=40)
    outline: list[RoadmapOutlinePhase] = Field(min_length=1, max_length=8)
    phase_number: int = Field(ge=1, le=12)
    done_titles: list[str] = Field(default_factory=list, max_length=40)


class RoadmapPhaseResponse(BaseModel):
    phase: RoadmapPhase
    model: str


class QuizGenerateRequest(BaseModel):
    # Topic is optional: a course_id alone is enough to generate a quiz
    # (the backend then fetches the course's own syllabus/roadmap/notes).
    # An empty topic with no course and no PDF is rejected with HTTP 400 by
    # quiz_service.build_plan ("Write a topic, choose a course, or upload
    # a PDF.").
    topic: str = Field(default="", max_length=300)
    kind: QuizKind
    count: int = Field(default=5, ge=1, le=20)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    language: str = Field(default="python", max_length=40)
    context: str = Field(default="", max_length=4000)
    course_id: int | None = Field(default=None, ge=1)
    # Kept for the existing frontend contract. The backend no longer forwards
    # client context/subject: material is chosen server-side by
    # quiz_service.build_plan (topic_only sends no notes at all).
    subject: str = Field(default="", max_length=120)


class QuizQuestion(BaseModel):
    kind: QuizKind
    prompt: str
    code: str = ""
    starter_code: str = ""
    language: str = ""
    options: list[str] = Field(default_factory=list)
    correct_index: int | None = None
    correct_output: str = ""
    buggy_line: int | None = None
    reference_solution: str = ""
    rubric: list[str] = Field(default_factory=list)
    explanation: str = ""


class QuizGenerateResponse(BaseModel):
    questions: list[QuizQuestion]
    model: str


class EvaluateRequest(BaseModel):
    kind: QuizKind
    question: str = Field(min_length=1, max_length=6000)
    language: str = Field(default="python", max_length=40)
    answer: str | int | float | list[float] | None = None
    code: str = Field(default="", max_length=20000)
    reference_solution: str = Field(default="", max_length=20000)
    expected_output: str = Field(default="", max_length=4000)
    buggy_line: int | None = None
    correct_index: int | None = None


class EvaluateResponse(BaseModel):
    is_correct: bool
    score: float
    feedback: str
    model: str | None = None
