from __future__ import annotations

from datetime import datetime
from enum import Enum, IntEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Category(str, Enum):
    AI = "ai"
    SOFTWARE = "software"
    HARDWARE = "hardware"
    AUTOMATION = "automation"
    FINANCE = "finance"
    OTHER = "other"


class TopicSource(str, Enum):
    SEED = "seed"
    TREND = "trend"
    LLM = "llm"


class SourceTier(IntEnum):
    TIER_1 = 1
    TIER_2 = 2
    TIER_3 = 3


class SearchIntent(str, Enum):
    INFORMATIONAL = "informational"
    COMMERCIAL = "commercial"
    TRANSACTIONAL = "transactional"
    NAVIGATIONAL = "navigational"


class TopicSeed(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=3, max_length=120)
    topic: str = Field(min_length=3, max_length=300)
    category: Category
    primary_keyword: str = Field(min_length=1, max_length=100)
    secondary_keywords: list[str] = Field(default_factory=list)
    angle: str = ""
    audience: str = "students"

    @field_validator("category", mode="before")
    @classmethod
    def _normalize_category(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value


class TopicHistoryEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    topic: str = Field(min_length=1)
    primary_keyword: str | None = None
    category: Category = Category.OTHER
    used_at: datetime

    @field_validator("category", mode="before")
    @classmethod
    def _normalize_category(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value


class TopicCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    topic: str = Field(min_length=3, max_length=300)
    category: Category
    primary_keyword: str = Field(min_length=1, max_length=100)
    secondary_keywords: list[str] = Field(default_factory=list)
    angle: str = ""
    reason: str = ""
    source: TopicSource = TopicSource.SEED
    trend_score: float = Field(default=50, ge=0, le=100)
    seo_score: float = Field(default=60, ge=0, le=100)
    product_fit: float = Field(default=65, ge=0, le=100)

    @field_validator("category", "source", mode="before")
    @classmethod
    def _normalize_enum(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value


class ScoredTopic(TopicCandidate):
    diversity_score: float = Field(ge=0, le=100)
    total_score: float = Field(ge=0, le=100)
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class ResearchSource(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str
    url: str
    domain: str
    snippet: str = ""
    tier: SourceTier = SourceTier.TIER_3
    relevance: float | None = None
    published_at: str | None = None


class NumericFact(BaseModel):
    model_config = ConfigDict(extra="ignore")

    value: float
    unit: str = ""
    context: str
    source_url: str


class ResearchBundle(BaseModel):
    model_config = ConfigDict(extra="ignore")

    topic: str
    primary_keyword: str
    queries: list[str] = Field(default_factory=list)
    sources: list[ResearchSource] = Field(default_factory=list)
    numeric_facts: list[NumericFact] = Field(default_factory=list)
    accessed_at: datetime


class KeywordPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    primary: str
    secondary: list[str] = Field(default_factory=list)
    long_tail: list[str] = Field(default_factory=list)
    lsi: list[str] = Field(default_factory=list)
    search_intent: SearchIntent = SearchIntent.INFORMATIONAL
    title_candidates: list[str] = Field(default_factory=list)

    @field_validator("search_intent", mode="before")
    @classmethod
    def _normalize_intent(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value


class WriterOutput(BaseModel):
    title: str
    slug: str
    excerpt: str
    content: str


class MetaPayload(BaseModel):
    meta_title: str
    meta_description: str


class MetaResult(MetaPayload):
    fallback: bool = False


class VisualType(str, Enum):
    CHART = "chart"
    TABLE = "table"
    PHOTO = "photo"


class VisualDataset(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    chart_type: str = "bar"
    labels: list[str] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)
    unit: str = ""
    source_url: str = ""
    source_label: str = ""
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


class CoverPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str
    alt: str
    caption: str


class ContentVisualPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    after_heading: str
    after_paragraph: int = Field(ge=1)
    type: VisualType
    purpose: str = ""
    alt: str
    caption: str
    query: str | None = None
    data: VisualDataset | None = None

    @field_validator("type", mode="before")
    @classmethod
    def _normalize_type(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value


class VisualPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cover: CoverPlan
    visuals: list[ContentVisualPlan] = Field(default_factory=list)


class StoredImage(BaseModel):
    url: str
    width: int
    height: int
    provider: str
    external_id: str | None = None
    kind: Literal["cover", "content"]


class ContentImage(BaseModel):
    url: str
    alt: str
    caption: str
    width: int
    height: int


class ProducedVisual(BaseModel):
    type: VisualType
    after_heading: str
    after_paragraph: int
    alt: str
    caption: str
    url: str | None = None
    provider: str
    external_id: str | None = None
    width: int | None = None
    height: int | None = None
    source_url: str | None = None


class VisualProduction(BaseModel):
    content: str
    cover_image_url: str
    cover_alt: str
    cover_caption: str
    cover_width: int
    cover_height: int
    content_images: list[ContentImage] = Field(default_factory=list)
    visuals: list[ProducedVisual] = Field(default_factory=list)


class FAQItem(BaseModel):
    question: str
    answer: str


class FAQAnswerSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    type: Literal["Answer"] = Field(default="Answer", alias="@type")
    text: str


class FAQQuestionSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    type: Literal["Question"] = Field(default="Question", alias="@type")
    name: str
    accepted_answer: FAQAnswerSchema = Field(alias="acceptedAnswer")


class FAQPageSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    context: Literal["https://schema.org"] = Field(
        default="https://schema.org", alias="@context"
    )
    type: Literal["FAQPage"] = Field(default="FAQPage", alias="@type")
    main_entity: list[FAQQuestionSchema] = Field(alias="mainEntity")


class FAQBundle(BaseModel):
    faqs: list[FAQItem]
    faq_schema: FAQPageSchema


class GateFailure(BaseModel):
    id: str
    detail: str


class GateWarning(BaseModel):
    id: str
    detail: str


class GateCheck(BaseModel):
    id: str
    category: str
    passed: bool
    detail: str = ""


class GateMetrics(BaseModel):
    word_count: int = 0
    h2_count: int = 0
    citation_count: int = 0
    unique_sources: int = 0
    flesch: float = 0
    keyword_density: float = 0
    internal_links: int = 0
    visual_count: int = 0


class GateReport(BaseModel):
    passed: bool
    score: int = Field(ge=0, le=100)
    hard_failures: list[GateFailure] = Field(default_factory=list)
    warnings: list[GateWarning] = Field(default_factory=list)
    metrics: GateMetrics = Field(default_factory=GateMetrics)
    checks: list[GateCheck] = Field(default_factory=list)
