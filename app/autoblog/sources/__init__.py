"""Source clients for AutoBlog v2 - Tavily, Pixabay, R2."""

from __future__ import annotations

import asyncio
import html
import random
import re
import urllib.parse
from typing import Any

import httpx
import requests
from tavily import TavilyClient

from app.autoblog.config.settings import get_autoblog_settings
from app.autoblog.llm.clients import BlogPipelineError


# ────────────────── Constants ────────────────────────────────────────────────

OFFICIAL_SOURCE_DOMAINS: tuple[str, ...] = (
    "openai.com",
    "anthropic.com",
    "ai.google",
    "deepmind.com",
    "huggingface.co",
    "arxiv.org",
    "techcrunch.com",
    "mit.edu",
    "stanford.edu",
)

RESEARCH_QUERIES: tuple[tuple[str, str], ...] = (
    ("model_releases", "latest AI model releases {month_year}"),
    ("industry_news", "OpenAI Anthropic Google AI announcements this week"),
    ("research", "AI research papers arXiv {month_year}"),
    ("education", "AI in education technology news"),
    ("fintech", "AI fintech regulations updates"),
)

BLOCKED_TOPIC_JUNK = frozenset(
    {
        "ai",
        "artificial intelligence",
        "technology",
        "tech",
        "news",
        "update",
        "updates",
    }
)

FALLBACK_TOPICS = (
    "The Future of AI Assistants",
    "How AI Is Changing Everyday Work",
    "AI Tools Students Actually Use",
    "AI & Technology Trends Worth Watching",
    "What's Next for AI Models",
)

STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has
    have how i if in into is it its just may more most not now of on or other
    our out over should so some such than that the their then there these
    they this those through to too up use used using very was we were what
    when where which who why will with would you your
    """.split()
)

# ────────────────── Pixabay Image Search ───────────────────────────────────────

UNSPLASH_FALLBACK_IMAGES: tuple[str, ...] = (
    "https://images.unsplash.com/photo-1677442136019-21780ecad995?w=1200",
    "https://images.unsplash.com/photo-1737071371043-761e02b1ef95?w=1200&auto=format&fit=crop",
    "https://images.unsplash.com/photo-1518770660439-4636190af475?w=1200",
    "https://images.unsplash.com/photo-1550751827-4bd374c3f58b?w=1200",
)

PIXABAY_UNSWORTHY_HITS = frozenset(
    {
        "cake", "food", "eat", "eating", "eaten", "dinner", "lunch", "breakfast",
        "restaurant", "meal", "cookie", "candy", "sugar", "bakery", "dessert",
    }
)

TECH_CONTEXT_WORDS = frozenset(
    {
        "ai", "artificial", "intelligence", "software", "technology",
        "technologies", "computer", "digital", "data", "robot", "robotics",
        "algorithm", "code", "coding", "developer", "programming", "tech",
        "machine", "cloud", "cyber", "startup", "internet", "device",
        "hardware", "assistant", "agents", "automation", "llm", "gpu",
        "chip", "openai", "anthropic", "google", "deepmind", "huggingface",
        "arxiv", "gpt", "chatgpt", "claude", "gemini", "llama", "mistral",
        "deepseek", "qwen", "kimi", "grok", "copilot", "fintech",
    }
)

TECH_IMAGE_TAGS = frozenset(
    {
        "ai", "artificial", "intelligence", "technology", "computer", "software",
        "code", "coding", "programming", "developer", "robot", "robotics",
        "machine", "data", "cyber", "digital", "algorithm", "startup",
        "internet", "electronics", "hardware", "science", "engineering",
    }
)

GENERIC_BUSINESS_TAGS = frozenset(
    {
        "business", "office", "meeting", "handshake", "corporate", "finance",
        "money", "chart", "graph", "team", "manager", "suit", "desk",
    }
)

AI_MODEL_RE = re.compile(
    r"\b(?:" 
    r"GPT[-\s]?(?:4o|4\.\d+|\d+(?:\.\d+)?)|"
    r"ChatGPT|"
    r"Claude(?:\s+\d+(?:\.\d+)?)?|"
    r"Gemini(?:\s+\d+(?:\.\d+)?)?|"
    r"Llama(?:\s+\d+(?:\.\d+)?)?|"
    r"Mistral(?:\s+\w+)?|"
    r"DeepSeek(?:[-\s]?\w+)?|"
    r"Qwen(?:\d+(?:\.\d+)?)?|"
    r"Kimi(?:\s+K\d+(?:\.\d+)?)?|"
    r"Grok(?:\s+\d+(?:\.\d+)?)?|"
    r"Copilot"
    r")\b",
    re.IGNORECASE,
)

PIXABAY_CATEGORY_HINTS: tuple[tuple[str, str], ...] = (
    ("fintech", "business"),
    ("banking", "business"),
    ("payment", "business"),
    ("education", "education"),
    ("student", "education"),
    ("school", "education"),
    ("classroom", "education"),
    ("teacher", "education"),
    ("research", "science"),
    ("arxiv", "science"),
    ("paper", "science"),
    ("study", "science"),
)

MAX_COVER_DOWNLOAD_BYTES = 15 * 1024 * 1024


def _query_is_techy(text: str) -> bool:
    return bool(set(re.findall(r"[a-z]+", text.lower())) & TECH_CONTEXT_WORDS)


def _extract_ai_models(text: str) -> list[str]:
    models: list[str] = []
    seen: set[str] = set()
    for match in AI_MODEL_RE.finditer(text):
        model = " ".join(match.group(0).split())
        key = model.lower()
        if key not in seen:
            seen.add(key)
            models.append(model)
    return models[:3]


def _append_ai(term: str, search_is_techy: bool) -> str:
    if not search_is_techy or re.search(r"\bai\b", term, re.IGNORECASE):
        return term
    return f"{term} AI"


def _image_queries(
    title: str, keywords: list[str], category: str | None
) -> list[str]:
    text = f"{title} {' '.join(keywords)}"
    search_is_techy = _query_is_techy(text)
    models = _extract_ai_models(text)
    main_topic = " ".join((title or "").split()[:6]).strip()
    candidates = [
        f"{' '.join(models)} AI" if models else "",
        _append_ai(main_topic, search_is_techy) if main_topic else "",
        _append_ai(" ".join(keywords[:3]), search_is_techy) if keywords else "",
        f"{category} AI" if category else "",
        main_topic,
    ]
    queries: list[str] = []
    seen: set[str] = set()
    for query in candidates:
        clean = query.strip()
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            queries.append(clean)
    return queries


def _pixabay_category(title: str, keywords: list[str]) -> str:
    text = f"{title} {' '.join(keywords)}".lower()
    for hint, category in PIXABAY_CATEGORY_HINTS:
        if hint in text:
            return category
    return "computer"


def _pick_hit(
    hits: list[dict], query: str, exclude_urls: set[str] | None = None
) -> str:
    excluded = exclude_urls or set()
    query_is_techy = _query_is_techy(query)
    ranked: list[tuple[int, int, dict]] = []
    for index, hit in enumerate(hits):
        tag_words = set(re.findall(r"[a-z]+", str(hit.get("tags") or "").lower()))
        if query_is_techy and tag_words & PIXABAY_UNSWORTHY_HITS:
            continue
        score = 0
        if query_is_techy:
            score += 2 * len(tag_words & TECH_IMAGE_TAGS)
            score -= 2 * len(tag_words & GENERIC_BUSINESS_TAGS)
        ranked.append((score, -index, hit))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    for _, _, hit in ranked:
        url = str(hit.get("largeImageURL") or "").strip()
        if url and url not in excluded:
            return url

    for hit in hits:
        url = str(hit.get("largeImageURL") or "").strip()
        if url and url not in excluded:
            return url
    return ""


def _unused_unsplash_fallback(
    exclude_urls: set[str] | None = None,
) -> str | None:
    excluded = exclude_urls or set()
    for url in UNSPLASH_FALLBACK_IMAGES:
        if url not in excluded:
            return url
    return None


async def fetch_relevant_image(
    title: str,
    keywords: list[str] | None = None,
    category: str | None = None,
    exclude_urls: set[str] | None = None,
) -> str | None:
    """Fetch a relevant photo from Pixabay using specific terms."""
    excluded = exclude_urls or set()
    settings = get_autoblog_settings()
    if not settings.pixabay_api_key:
        return None

    kw = [str(k).strip() for k in (keywords or []) if str(k).strip()]
    queries = _image_queries(title, kw, category)
    query_is_techy = _query_is_techy(f"{title} {' '.join(kw)}")
    pixabay_category = _pixabay_category(title, kw) if query_is_techy else None

    async with httpx.AsyncClient(timeout=10.0) as client:
        for query in queries:
            params: dict[str, Any] = {
                "key": settings.pixabay_api_key,
                "q": query[:100],
                "image_type": "photo",
                "orientation": "horizontal",
                "safesearch": "true",
                "min_width": 1200,
                "min_height": 800,
                "order": "popular",
                "per_page": 5,
            }
            if pixabay_category:
                params["category"] = pixabay_category
            try:
                response = await client.get("https://pixabay.com/api/", params=params)
                response.raise_for_status()
                data = response.json()
            except Exception as exc:
                print(
                    f"[AutoBlog] Pixabay query {query!r} failed "
                    f"({type(exc).__name__}: {exc}) - trying next.",
                    flush=True,
                )
                continue
            hits = data.get("hits") if isinstance(data, dict) else None
            if hits:
                url = _pick_hit(hits, query, excluded)
                if url:
                    print(f"[AutoBlog] Pixabay image OK for {query!r}", flush=True)
                    return url

    fallback = _unused_unsplash_fallback(excluded)
    if fallback:
        print(
            f"[AutoBlog] Pixabay found nothing for {title!r} - using Unsplash fallback.",
            flush=True,
        )
    else:
        print(
            f"[AutoBlog] Pixabay found only duplicate images for {title!r} - skipping.",
            flush=True,
        )
    return fallback


async def persist_cover_image(url: str, slug: str) -> str:
    """Download cover, upload compressed WebP to R2, return public URL."""
    from app.services.r2_storage import r2_storage

    if not url or url in UNSPLASH_FALLBACK_IMAGES:
        return UNSPLASH_FALLBACK_IMAGES[0]

    try:
        async with httpx.AsyncClient(
            timeout=30.0, follow_redirects=True
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            content_type = (
                response.headers.get("content-type", "")
                .split(";", 1)[0]
                .strip()
                .lower()
            )
            if not content_type.startswith("image/"):
                raise ValueError(
                    f"unexpected cover content type: {content_type or 'unknown'}"
                )
            image_bytes = response.content
            if not image_bytes:
                raise ValueError("cover download returned no bytes")
            if len(image_bytes) > MAX_COVER_DOWNLOAD_BYTES:
                raise ValueError("cover image exceeds the 15 MB limit")

        permanent_url = await asyncio.to_thread(
            r2_storage.compress_and_upload, image_bytes, slug
        )
        if not permanent_url:
            raise RuntimeError("R2 upload returned an empty URL")
        return permanent_url
    except Exception as exc:
        print(
            f"[AutoBlog] cover R2 persistence failed "
            f"({type(exc).__name__}: {exc}) - using Unsplash fallback.",
            flush=True,
        )
        return UNSPLASH_FALLBACK_IMAGES[0]


def _figure_html(url: str, caption: str) -> str:
    alt = html.escape(caption, quote=True)
    return (
        f'<img src="{url}" alt="{alt}" loading="lazy" '
        'width="800" height="450" />'
    )


# ────────────────── Tavily Research ────────────────────────────────────────────


def _month_year() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%B %Y")


def _day_index() -> int:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).timetuple().tm_yday


def ordered_research_queries() -> list[tuple[str, str]]:
    month_year = _month_year()
    queries = [
        (category, template.format(month_year=month_year))
        for category, template in RESEARCH_QUERIES
    ]
    offset = _day_index() % len(queries)
    return queries[offset:] + queries[:offset]


def _result_host(url: str) -> str:
    host = urllib.parse.urlsplit(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _domain_rank(url: str) -> int:
    host = _result_host(url)
    for index, domain in enumerate(OFFICIAL_SOURCE_DOMAINS):
        if host == domain or host.endswith(f".{domain}"):
            return index
    return len(OFFICIAL_SOURCE_DOMAINS)


def _tavily_search(
    query: str, include_domains: tuple[str, ...] | None = OFFICIAL_SOURCE_DOMAINS
) -> list[dict[str, Any]]:
    settings = get_autoblog_settings()
    if not settings.tavily_api_key:
        raise BlogPipelineError("TAVILY_API_KEY is not configured.", status_code=503)
    client = TavilyClient(api_key=settings.tavily_api_key)
    args: dict[str, Any] = {
        "query": query,
        "topic": "news",
        "days": settings.research_days,
        "search_depth": settings.research_search_depth,
        "max_results": settings.research_max_results,
    }
    if include_domains:
        args["include_domains"] = list(include_domains)
    response = client.search(**args)
    results = response.get("results") if isinstance(response, dict) else None
    return results if isinstance(results, list) else []


def _clean_result(item: dict[str, Any]) -> dict[str, str]:
    return {
        "title": str(item.get("title") or "").strip(),
        "url": str(item.get("url") or "").strip(),
        "content": str(item.get("content") or "").strip(),
        "published_date": str(item.get("published_date") or "").strip(),
    }


def _result_line(result: dict[str, str]) -> str:
    published = (
        f" [{result['published_date']}]" if result.get("published_date") else ""
    )
    return (
        f"- {result['title']}{published} ({result['url']}): "
        f"{result['content'][:500]}"
    )


def research_snippets(results: list[dict[str, str]]) -> list[str]:
    return [
        f"{result['title']} ({result['url']}): {result['content'][:500]}"
        for result in results
        if result.get("title") or result.get("content")
    ]


def _dedupe_results(results: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    cleaned: list[dict[str, str]] = []
    for result in results:
        key = result.get("url") or result.get("title", "").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(result)
    cleaned.sort(key=lambda result: _domain_rank(result.get("url", "")))
    return cleaned


async def research_chunk(query: str) -> list[dict[str, str]]:
    try:
        raw = await asyncio.to_thread(_tavily_search, query)
    except Exception as exc:
        print(
            f"[AutoBlog] official-domain search failed for {query!r} "
            f"({type(exc).__name__}: {exc}) - retrying unrestricted.",
            flush=True,
        )
        raw = await asyncio.to_thread(_tavily_search, query, None)

    results = _dedupe_results(
        [
            cleaned
            for cleaned in (_clean_result(item) for item in raw)
            if cleaned["title"] or cleaned["content"]
        ]
    )
    if results:
        return results

    raw = await asyncio.to_thread(_tavily_search, query, None)
    return _dedupe_results(
        [
            cleaned
            for cleaned in (_clean_result(item) for item in raw)
            if cleaned["title"] or cleaned["content"]
        ]
    )


async def fetch_recent_posts(limit: int | None = None) -> list[dict[str, Any]]:
    from supabase import Client, create_client
    settings = get_autoblog_settings()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        return []
    try:
        client = create_client(settings.supabase_url, settings.supabase_service_role_key)
        lim = limit or settings.recent_posts_limit
        response = (
            client.table("blog_posts")
            .select("title, keywords, published_at")
            .order("published_at", desc=True)
            .limit(lim)
            .execute()
        )
        rows = getattr(response, "data", None)
        if isinstance(rows, list):
            return [
                row
                for row in rows
                if isinstance(row, dict) and row.get("published_at")
            ]
    except Exception as exc:
        print(
            f"[AutoBlog] recent-post lookup failed "
            f"({type(exc).__name__}: {exc}) - dedupe disabled for this run.",
            flush=True,
        )
    return []


# ────────────────── Exports ──────────────────────────────────────────────────

__all__ = [
    # Tavily research
    "ordered_research_queries",
    "research_chunk",
    "fetch_recent_posts",
    "research_snippets",
    "_month_year",
    # Pixabay images
    "fetch_relevant_image",
    "persist_cover_image",
    "_figure_html",
    "UNSPLASH_FALLBACK_IMAGES",
    # Constants
    "OFFICIAL_SOURCE_DOMAINS",
    "RESEARCH_QUERIES",
    "BLOCKED_TOPIC_JUNK",
    "FALLBACK_TOPICS",
    "STOPWORDS",
    "TECH_CONTEXT_WORDS",
    "TECH_IMAGE_TAGS",
    "GENERIC_BUSINESS_TAGS",
    "AI_MODEL_RE",
    "PIXABAY_CATEGORY_HINTS",
    "MAX_COVER_DOWNLOAD_BYTES",
]