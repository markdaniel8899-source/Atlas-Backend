from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .prompts import topic_messages
from .types import (
    Category,
    ScoredTopic,
    TopicCandidate,
    TopicHistoryEntry,
    TopicSeed,
    TopicSource,
)

if TYPE_CHECKING:
    from .providers import BlogProviderClient, ProviderJsonResult


CATEGORY_TARGETS: dict[Category, int] = {
    Category.AI: 7,
    Category.SOFTWARE: 5,
    Category.HARDWARE: 2,
    Category.AUTOMATION: 3,
    Category.FINANCE: 2,
    Category.OTHER: 1,
}
CATEGORY_WINDOW = 20
TOPIC_DEDUPE_DAYS = 90
KEYWORD_DEDUPE_DAYS = 60
JACCARD_THRESHOLD = 0.8
WEIGHTS = {
    "trend": 0.30,
    "seo": 0.25,
    "product_fit": 0.25,
    "diversity": 0.20,
}

_WORD_RE = re.compile(r"[a-z0-9]+")
_SLUG_RE = re.compile(r"^-+|-+$")
_BANNED_RE = re.compile(
    r"\b(politics|political|election|elections|medical diagnosis|"
    r"legal advice|gambling|casino|adult content|conspiracy|"
    r"conspiracies)\b",
    flags=re.IGNORECASE,
)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "vs",
    "what",
    "when",
    "why",
    "with",
    "you",
    "your",
}
_PRODUCT_TERMS = {
    "ai",
    "automation",
    "exam",
    "focus",
    "learning",
    "note",
    "notes",
    "productivity",
    "software",
    "student",
    "students",
    "study",
}


def default_seed_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "topic_seeds.json"


def load_topic_seeds(path: str | Path | None = None) -> list[TopicSeed]:
    source = Path(path) if path is not None else default_seed_path()
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload.get("topics") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("Topic seed file must contain a non-empty topics list.")
    seeds = [TopicSeed.model_validate(row) for row in rows]
    ids = [seed.id for seed in seeds]
    topics = [topic_slug(seed.topic) for seed in seeds]
    keywords = [normalize_text(seed.primary_keyword) for seed in seeds]
    if len(ids) != len(set(ids)):
        raise ValueError("Topic seed ids must be unique.")
    if len(topics) != len(set(topics)):
        raise ValueError("Topic seed topics must be unique.")
    if len(keywords) != len(set(keywords)):
        raise ValueError("Topic seed primary keywords must be unique.")
    return seeds


def seed_candidates(
    seeds: Sequence[TopicSeed] | None = None,
) -> list[TopicCandidate]:
    rows = list(seeds) if seeds is not None else load_topic_seeds()
    return [
        TopicCandidate(
            topic=seed.topic,
            category=seed.category,
            primary_keyword=seed.primary_keyword,
            secondary_keywords=list(seed.secondary_keywords),
            angle=seed.angle,
            reason=seed.angle,
            source=TopicSource.SEED,
            trend_score=50.0,
            seo_score=_seo_heuristic(seed.primary_keyword, seed.topic),
            product_fit=_product_fit(seed.topic, seed.primary_keyword),
        )
        for seed in rows
    ]


def normalize_text(text: str) -> str:
    value = (text or "").strip().lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def topic_slug(text: str) -> str:
    return _SLUG_RE.sub("", re.sub(r"[^a-z0-9]+", "-", normalize_text(text)))


def topic_tokens(text: str) -> set[str]:
    return {
        token
        for token in _WORD_RE.findall(normalize_text(text))
        if token not in _STOPWORDS
    }


def jaccard(left: str, right: str) -> float:
    left_tokens = topic_tokens(left)
    right_tokens = topic_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def is_banned(topic: str) -> bool:
    return bool(_BANNED_RE.search(topic or ""))


def category_counts(
    history: Sequence[TopicHistoryEntry],
    *,
    window: int = CATEGORY_WINDOW,
) -> dict[Category, int]:
    counts: Counter[Category] = Counter()
    ordered = sorted(
        history, key=lambda entry: _as_utc(entry.used_at), reverse=True
    )[:window]
    for entry in ordered:
        counts[entry.category] += 1
    return {category: counts.get(category, 0) for category in Category}


def allowed_categories(counts: Mapping[Category, int]) -> set[Category]:
    allowed = {
        category
        for category, target in CATEGORY_TARGETS.items()
        if counts.get(category, 0) < target
    }
    return allowed or set(Category)


def diversity_score(
    category: Category, counts: Mapping[Category, int]
) -> float:
    target = CATEGORY_TARGETS.get(category, 0)
    if target <= 0:
        return 0.0
    deficit = max(0, target - counts.get(category, 0))
    return round(min(100.0, deficit / target * 100.0), 2)


def find_duplicate(
    candidate: TopicCandidate,
    history: Sequence[TopicHistoryEntry],
    *,
    now: datetime | None = None,
) -> str | None:
    current = _as_utc(now or datetime.now(timezone.utc))
    candidate_slug = topic_slug(candidate.topic)
    candidate_keyword = normalize_text(candidate.primary_keyword)

    for entry in history:
        used = _as_utc(entry.used_at)
        age = current - used
        if age <= timedelta(days=TOPIC_DEDUPE_DAYS):
            if topic_slug(entry.topic) == candidate_slug:
                return "exact topic used within 90 days"
            if jaccard(candidate.topic, entry.topic) >= JACCARD_THRESHOLD:
                return "near-duplicate topic used within 90 days"
        if (
            age <= timedelta(days=KEYWORD_DEDUPE_DAYS)
            and entry.primary_keyword
            and normalize_text(entry.primary_keyword) == candidate_keyword
        ):
            return "primary keyword used within 60 days"
    return None


def score_candidate(
    candidate: TopicCandidate,
    counts: Mapping[Category, int],
) -> ScoredTopic:
    diversity = diversity_score(candidate.category, counts)
    total = (
        WEIGHTS["trend"] * candidate.trend_score
        + WEIGHTS["seo"] * candidate.seo_score
        + WEIGHTS["product_fit"] * candidate.product_fit
        + WEIGHTS["diversity"] * diversity
    )
    return ScoredTopic(
        **candidate.model_dump(),
        diversity_score=diversity,
        total_score=round(min(100.0, max(0.0, total)), 2),
        score_breakdown={
            "trend": candidate.trend_score,
            "seo": candidate.seo_score,
            "product_fit": candidate.product_fit,
            "diversity": diversity,
        },
    )


def rank_topics(
    candidates: Sequence[TopicCandidate],
    history: Sequence[TopicHistoryEntry] = (),
    *,
    now: datetime | None = None,
) -> list[ScoredTopic]:
    counts = category_counts(history)
    allowed = allowed_categories(counts)
    seen_topics: set[str] = set()
    seen_keywords: set[str] = set()
    ranked: list[ScoredTopic] = []

    for candidate in candidates:
        slug = topic_slug(candidate.topic)
        keyword = normalize_text(candidate.primary_keyword)
        if not slug or not keyword:
            continue
        if slug in seen_topics or keyword in seen_keywords:
            continue
        if is_banned(candidate.topic):
            continue
        if candidate.category not in allowed:
            continue
        if find_duplicate(candidate, history, now=now):
            continue
        seen_topics.add(slug)
        seen_keywords.add(keyword)
        ranked.append(score_candidate(candidate, counts))

    ranked.sort(
        key=lambda item: (
            -item.total_score,
            -item.trend_score,
            -item.seo_score,
            item.primary_keyword.lower(),
            item.topic.lower(),
        )
    )
    return ranked


def select_topic(
    candidates: Sequence[TopicCandidate],
    history: Sequence[TopicHistoryEntry] = (),
    *,
    now: datetime | None = None,
) -> ScoredTopic | None:
    ranked = rank_topics(candidates, history, now=now)
    return ranked[0] if ranked else None


def validate_topic_candidates(data: Any) -> list[TopicCandidate]:
    rows = data.get("candidates") if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise ValueError("Topic response must contain a non-empty candidates list.")
    if len(rows) > 10:
        raise ValueError("Topic response must contain at most 10 candidates.")

    candidates: list[TopicCandidate] = []
    seen_topics: set[str] = set()
    seen_keywords: set[str] = set()
    for row in rows:
        candidate = TopicCandidate.model_validate(row).model_copy(
            update={"source": TopicSource.LLM}
        )
        keyword_words = candidate.primary_keyword.split()
        if len(keyword_words) < 2 or len(keyword_words) > 5:
            raise ValueError("Primary keyword must be 2-5 words.")
        slug = topic_slug(candidate.topic)
        keyword = normalize_text(candidate.primary_keyword)
        if slug in seen_topics:
            raise ValueError(f"Duplicate topic candidate: {candidate.topic}")
        if keyword in seen_keywords:
            raise ValueError(
                f"Duplicate primary keyword: {candidate.primary_keyword}"
            )
        seen_topics.add(slug)
        seen_keywords.add(keyword)
        candidates.append(candidate)
    return candidates


async def generate_topic_candidates(
    client: "BlogProviderClient",
    *,
    recent_topics: Sequence[str] = (),
    recent_keywords: Sequence[str] = (),
    counts: Mapping[Category, int] | None = None,
    trend_snippets: Sequence[str] = (),
    now: datetime | None = None,
) -> "ProviderJsonResult[list[TopicCandidate]]":
    from .providers import BlogStage

    current = now or datetime.now(timezone.utc)
    current_counts = counts or {category: 0 for category in Category}
    targets = {
        category.value: max(
            0, CATEGORY_TARGETS[category] - current_counts.get(category, 0)
        )
        for category in Category
    }
    messages = topic_messages(
        current_date=current.date().isoformat(),
        category_targets=targets,
        recent_topics=recent_topics,
        recent_keywords=recent_keywords,
        trend_snippets=trend_snippets,
    )
    return await client.complete_json(
        BlogStage.TOPIC,
        messages,
        validator=validate_topic_candidates,
    )


def _seo_heuristic(keyword: str, topic: str) -> float:
    words = normalize_text(keyword).split()
    score = 58.0
    if 2 <= len(words) <= 5:
        score += 8.0
    if len(words) >= 3:
        score += 4.0
    if "how" in normalize_text(topic).split():
        score += 4.0
    return min(85.0, score)


def _product_fit(topic: str, keyword: str) -> float:
    tokens = topic_tokens(f"{topic} {keyword}")
    matches = len(tokens & _PRODUCT_TERMS)
    return min(90.0, 55.0 + matches * 7.0)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)

