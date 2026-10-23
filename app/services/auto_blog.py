"""Auto AI Blog Writer — staggered 6-chunk HYBRID pipeline (rate-limit friendly).

Runs in the background via FastAPI BackgroundTasks (POST /api/blog/generate)
and daily at 09:00 UTC through APScheduler (wired up in app.main).

HYBRID MODEL ROUTING — every provider does what it is best at:

  - Tavily     : web research (not an LLM)                     [steps 1-2]
  - NVIDIA NIM : Nemotron Ultra 550B picks the trending topic
                 and extracts the top 3-5 SEO keywords.        [steps 1-2]
  - Groq       : DEDICATED key (GROQ_BLOG_API_KEY, never the
                 Chat/Quiz GROQ_API_KEY) writes the meta data
                 and the main article.                         [steps 3-4]
  - Gemini     : automatic fallback for steps 3-4 ONLY when
                 Groq returns 429 (rate limit) or 5xx.         [steps 3-4]
  - Pixabay    : cover + in-article photos via the Pixabay API
                 (PIXABAY_API_KEY). Covers are persisted to R2;
                 in-article images keep external CDN URLs.        [step 5]
  - Cloudflare R2: durable WebP cover storage.                    [step 5]
  - Supabase   : save the finished post.                       [step 6]

The pipeline is deliberately split into chunks with pauses between them so
no provider sees a burst of back-to-back calls (avoids 429s):

  1. Research      — Tavily news search across rotating AI queries, then
                     Nemotron Ultra picks ONE fresh trending topic while
                     avoiding the last 10 published posts.         [2s]
  2. Keywords      — Nemotron Ultra distills the top 3-5 SEO
                     keywords for the topic (heuristic fallback). [2s]
  3. Meta          — Groq writes the catchy meta title + meta
                     description (plain fallbacks; Gemini retry on
                     429/5xx).                                    [3s]
  4. Write         — Groq writes an 800-1200-word, SEO/AEO/GEO-optimised
                      article as strict JSON (title / excerpt / content_html
                      / tags) with a direct answer opening, inline citations
                      and a mandatory FAQ section; same Gemini retry on
                      429/5xx.                                         [3s]
  5. Images        — Pixabay cover + 1-3 randomized in-article photos,
                     fetched ONE BY ONE with a 3s pause between each.
                     Cover and in-article URLs are deduplicated so no
                     image repeats. The cover is downloaded, cropped to
                     1200x630, compressed to WebP, and uploaded to
                     Cloudflare R2 (stable Unsplash URL on failure).
                     In-article images keep their external CDN URLs and
                     are stored in blog_posts.content_images.
  6. Upload        — supabase-py insert into public.blog_posts with the
                     service-role key (RLS-bypassing writer).

Error handling: the entire orchestrator body is wrapped in one
try/except that prints [AutoBlog Background] CRITICAL TASK ERROR plus a
traceback, and every chunk also logs its own [AutoBlog Background] Step N
line. NIM failures in steps 1-2 fall back to plain heuristics (no Groq or
Gemini quota burned).

Async note: clients are created per run on purpose — the background task
and the scheduler job each own their event loop; no client object may be
shared across loops. Sleeps are ALWAYS `asyncio.sleep` (this function is
an `async def` on the event loop — `time.sleep` would block all requests).
"""

from __future__ import annotations

import asyncio
import difflib
import html
import os
import random
import re
import sys
import threading
import urllib.parse
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import google.generativeai as genai
import httpx
import requests
from supabase import Client, create_client
from tavily import TavilyClient

from app import ai_service
from app.cache import clear_all as clear_read_caches
from app.config import get_settings
from app.plugins.seo_checker import evaluate_seo_lightweight
from app.services.r2_storage import r2_storage

# Every print() in this module goes to stdout, which `nohup ... > server.log`
# captures. Non-tty stdout is BLOCK-buffered (8KB), so step logs sat in the
# buffer and never showed up next to uvicorn's stderr access log ("202
# Accepted") — the task ran but looked dead. Line-buffering makes each print
# hit the log file immediately. UTF-8 + errors="replace" guarantees the emoji
# and arrow characters can never crash a print (ASCII/Latin-1 locales would
# otherwise raise UnicodeEncodeError inside the task).
try:
    sys.stdout.reconfigure(
        line_buffering=True, encoding="utf-8", errors="replace"
    )
except Exception:  # noqa: BLE001 - stdout swapped by a test harness → skip
    pass

# ── NVIDIA NIM (research phase: trending topic + SEO keywords) ───────────────
# OpenAI-compatible endpoint at integrate.api.nvidia.com. Free-tier friendly.
# Note: read AFTER `from app.config import get_settings` so load_dotenv() has
# already populated os.environ from backend/.env.
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()
NVIDIA_API_URL = os.getenv(
    "NVIDIA_API_URL", "https://integrate.api.nvidia.com/v1"
).strip()
NVIDIA_MODEL = os.getenv(
    "NVIDIA_BLOG_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"
).strip()

# ── Pixabay (blog photos: cover + in-article) ────────────────────────────────
# Free API key from https://pixabay.com/api/docs/. Read AFTER
# `from app.config import get_settings` so load_dotenv() has already
# populated os.environ from backend/.env.
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "").strip()

MAX_RESULTS = 5  # "top 3-5 keywords" → cap at 5

# In-article image count is randomized per run: 1-3 images (plus the cover),
# so every post ends up with 2-4 visuals in total.
MIN_CONTENT_IMAGES = 1
MAX_CONTENT_IMAGES = 3

# Deliberate pauses between chunks — keep every provider under its rate limit.
RESEARCH_PAUSE_SECONDS = 2  # after topic (step 1) and keywords (step 2)
WRITE_PAUSE_SECONDS = 3  # after meta (step 3) and article (step 4)
IMAGE_PAUSE_SECONDS = 3  # between in-article image generations

# Rotating research categories. The day index picks the starting point so each
# daily run leads with a different category instead of the same static query.
RESEARCH_QUERIES: tuple[tuple[str, str], ...] = (
    ("model_releases", "latest AI model releases {month_year}"),
    ("industry_news", "OpenAI Anthropic Google AI announcements this week"),
    ("research", "AI research papers arXiv {month_year}"),
    ("education", "AI in education technology news"),
    ("fintech", "AI fintech regulations updates"),
)

# Tavily news mode: keep the window short so the blog is actually current.
RESEARCH_DAYS = 3
RESEARCH_MAX_RESULTS = 10
RESEARCH_SEARCH_DEPTH = "advanced"

# Root domains only: Tavily include_domains does not accept URL paths.
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

# Deduplicate against the most recent published posts.
RECENT_POSTS_LIMIT = 10
SIMILARITY_THRESHOLD = 0.70
TOPIC_RETRY_LIMIT = 2

# Generic terms are useless as "avoid" instructions and can accidentally
# forbid the whole AI niche.
_BLOCKED_TOPIC_JUNK = frozenset(
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

# Used when Tavily is down/unconfigured; one per UTC day.
FALLBACK_TOPICS = (
    "The Future of AI Assistants",
    "How AI Is Changing Everyday Work",
    "AI Tools Students Actually Use",
    "AI & Technology Trends Worth Watching",
    "What's Next for AI Models",
)

# Only one pipeline run at a time (route + scheduler share this lock).
_RUN_LOCK = threading.Lock()

_STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has
    have how i if in into is it its just may more most not now of on or other
    our out over should so some such than that the their then there these
    they this those through to too up use used using very was we were what
    when where which who why will with would you your
    """.split()
)

_WRITER_SYSTEM = """You are an expert tech journalist and industry analyst writing for ATLAS, a learning OS for students. Your articles are published on our public blog and must be citable by both Google and AI assistants.

STRICT SEO, AEO, & GEO REQUIREMENTS (MUST FOLLOW):

1. DIRECT ANSWER FIRST (AEO):
   - Answer the core user query directly within the first 2 sentences of the introduction.
   - Never open with fluff such as "In today's fast-paced world" or a long wind-up.

2. STRUCTURE & READABILITY:
   - Use proper HTML tags: <h2>, <h3>, <p>, <ul>, <li>, <strong>, and <table> only when the topic calls for it.
   - Keep paragraphs short (1-3 sentences).
   - Use bullet points for features, steps, or comparisons. At least one bullet list per article.
   - Write 3-5 body sections, each opened by an engaging <h2>; use <h3> where it helps.

3. DYNAMIC FORMATTING & VISUALS (Context-Aware):
   - Analyze the nature of the topic and choose the format that informs best. Never force a table into an article where it does not fit.
   - Theoretical, conceptual, or opinion topics: use narrative paragraphs and bullets; do NOT use a <table>.
   - Research, data, or statistics-heavy topics: MUST include one HTML <table> with <thead>, <tbody>, <tr>, <th> and <td>, plus a one-sentence introduction before it.
   - Comparison or "versus" topics: MUST include one comparison <table> (features, pros and cons, benchmarks) plus a short bullet summary.
   - Visuals: the publishing pipeline places 1-3 highly relevant images automatically, including a conceptual image mid-article. Do NOT output <img> tags, placeholders, or image URLs; just keep the middle sections visual-friendly.
   - Every <h2> and <h3> must be unique and descriptive. Never repeat the same heading text, and never restate a heading's wording as plain text directly beneath it.

4. KEYWORD DENSITY (SEO):
   - Naturally include the exact primary keyword 5-7 times. Never stuff.
   - Include it in the first 100 words, in at least one <h2> heading, and in the closing section.
   - Use semantic/LSI keywords related to the topic naturally.

5. AUTHORITY & CITATIONS (GEO):
   - Cite primary sources naturally in the text (for example, "According to the official documentation..." or "As announced by...").
   - Link 2-3 of the supplied source URLs inline with <a href="..." rel="noopener">.
   - Ground every claim in the research provided. If unsure about a statistic, phrase it generally. Never invent studies or numbers.
   - Maintain an objective, expert, professional tone. E-E-A-T compliant.

6. MANDATORY FAQ SECTION (AEO):
   - At the very end of the article, add an <h2>FAQ</h2> section.
   - Include exactly 3-4 specific, long-tail questions related to the topic, each followed by a concise, direct answer of 40-60 words.
   - Format: <h3>Question?</h3><p>Direct, clear answer.</p>

7. LENGTH:
   - Write between 800 and 1200 words for comprehensive coverage.

VOICE & OUTPUT RULES:
- Write like a sharp, authoritative tech journalist: contractions, direct "you", specific examples. A human wrote this.
- BANNED words and phrases (using any of them fails the brief): delve, delve into, moreover, furthermore, in conclusion, additionally, it's worth noting, it is important to note, in today's fast-paced world, ever-evolving landscape, landscape of, tapestry, testament, testament to, unlock, unleash, dive deep, dive into, game-changer, revolutionize, revolutionary, seamless, robust, cutting-edge, elevate, harness the power, in a world where, boasts, myriad, plethora, navigating, embarking, fostering, realm.
- Never use em dashes (—) or en dashes (–) anywhere in the title, excerpt or body. Use commas, colons or parentheses instead.
- The content_html value is a plain HTML fragment: never <html>, <head> or <body>. The article must end with the FAQ section.

Return ONLY strict JSON (no markdown fences, no commentary before or after):
{"title": "...", "excerpt": "...", "content_html": "...", "tags": ["...", "..."]}

- title: max 110 chars, specific and engaging, contains the primary keyword, no clickbait lies.
- excerpt: 130-155 chars of plain text containing the primary keyword (used as the meta description).
- content_html: the full article body as an HTML fragment, ending with the FAQ section.
- tags: 3-5 lowercase topic tags."""

_TOPIC_SYSTEM = """You are an SEO trend analyst for ATLAS, a learning OS for students.
From the research snippets, pick ONE specific, currently trending, searchable blog topic (not a generic pillar term).

Return ONLY strict JSON (no fences, no commentary):
{"topic": "..."}

- topic: a concrete angle a reader would click, max 120 chars.
- Prefer a specific model, company, research paper, or measurable news event over generic advice.
- Never repeat an angle from the recently published posts supplied in the prompt."""

_KEYWORD_SYSTEM = """You are an SEO content strategist for ATLAS, a learning OS for students.
Extract the top 3 to 5 SEO keywords for the given blog topic.

Return ONLY strict JSON (no fences, no commentary):
{"keywords": ["...", "...", "..."]}

- keywords: 3 to 5, each 1-3 words, lowercase, no dupes, primary keyword first."""

_META_SYSTEM = """You are an SEO meta-copy expert. From the topic and keywords provided, write ONE catchy meta title and ONE meta description for the blog post.

Return ONLY strict JSON (no fences, no commentary):
{"meta_title": "...", "meta_description": "..."}

- meta_title: catchy and specific, max 110 chars, MUST contain the exact primary keyword supplied by the user, may include a number or power word.
- meta_description: 130-155 chars (never below 120 or above 160), MUST contain the primary keyword, entices a click. Plain text only.
- No em dashes (—) or en dashes (–); use commas or colons instead."""


class BlogPipelineError(RuntimeError):
    """A pipeline stage failed in a way that aborts the run."""

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(slots=True)
class Research:
    topic: str
    keywords: list[str]
    snippets: list[str]


def _utc() -> datetime:
    return datetime.now(timezone.utc)


def _day_index() -> int:
    return _utc().timetuple().tm_yday


def slugify(title: str) -> str:
    """URL slug matching the blog_posts check constraint ^[a-z0-9]+(-[a-z0-9]+)*$."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")[:80].strip("-")
    return slug or f"blog-{_utc().strftime('%Y%m%d')}"


def _heuristic_keywords(text: str, limit: int = MAX_RESULTS) -> list[str]:
    """Stopword-filtered frequency fallback when Gemini extraction is unavailable."""
    words = re.findall(r"[a-z][a-z'-]{2,}", text.lower())
    counts = Counter(w for w in words if w not in _STOPWORDS)
    out: list[str] = []
    for word, _ in counts.most_common():
        if word not in out:
            out.append(word)
        if len(out) >= limit:
            break
    return out


def _pad_keywords(keywords: list[str], topic: str) -> list[str]:
    """Keep at most 5 unique keywords; top up from the topic when short.

    Topic bigrams (e.g. "ai education") beat single junk words for image
    prompts and SEO, so they are tried before plain topic tokens.
    """
    tokens = [t for t in re.findall(r"[a-z0-9]{2,}", topic.lower()) if t not in _STOPWORDS]
    bigrams = [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
    out: list[str] = []
    for value in [*keywords, *bigrams, *tokens, *_heuristic_keywords(topic)]:
        clean = value.strip().lower()
        if clean and clean not in out:
            out.append(clean)
        if len(out) >= MAX_RESULTS:
            break
    return out or [topic.lower()]


# ─────────────────────── NVIDIA NIM helpers (light chunks) ───────────────────


def _nim_chat(
    system: str, prompt: str, *, temperature: float, max_tokens: int
) -> str:
    """Sync NVIDIA NIM chat call (OpenAI-compatible) — run via asyncio.to_thread.

    Used for the research chunks (trending topic + SEO keywords). NIM's
    free tier intermittently answers 429/503, so ONE immediate retry
    absorbs the blip; anything deeper is left to the caller's heuristics.
    """
    if not NVIDIA_API_KEY:
        raise BlogPipelineError("NVIDIA_API_KEY is not configured.", status_code=503)
    text = ""
    try:
        for attempt in range(2):
            response = requests.post(
                f"{NVIDIA_API_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {NVIDIA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": NVIDIA_MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=90,
            )
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 0:
                    print(
                        f"[AutoBlog] NIM HTTP {response.status_code} — "
                        "retrying once...",
                        flush=True,
                    )
                    continue
                raise BlogPipelineError(
                    f"NVIDIA NIM call failed: HTTP {response.status_code}: "
                    f"{response.text[:160]}"
                )
            response.raise_for_status()
            data = response.json()
            text = str(data["choices"][0]["message"]["content"] or "").strip()
            if text:
                break
            print(
                "[AutoBlog] NIM returned empty text — retrying once...", flush=True
            )
    except BlogPipelineError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced as a pipeline error
        raise BlogPipelineError(f"NVIDIA NIM call failed: {exc}") from exc
    if not text:
        raise BlogPipelineError("NVIDIA NIM returned empty text.")
    return text


# ───────────────── Groq blog helpers (meta + article writing) ────────────────


class GroqTransientError(RuntimeError):
    """Groq rate limit (429) or server error (5xx) → retry the same prompt on Gemini."""


def _groq_blog_chat(
    system: str, prompt: str, *, temperature: float, max_tokens: int
) -> str:
    """Sync Groq chat call on the DEDICATED blog key — run via asyncio.to_thread.

    Uses GROQ_BLOG_API_KEY (never GROQ_API_KEY, which is reserved for the
    Chat/Quiz routes). Raises GroqTransientError on 429/5xx so the caller
    can fall back to Gemini with the exact same prompt.
    """
    settings = get_settings()
    if not settings.groq_blog_api_key:
        raise BlogPipelineError(
            "GROQ_BLOG_API_KEY is not configured.", status_code=503
        )
    text = ""
    try:
        # One retry for transient 200-but-empty responses (Groq occasionally
        # returns an empty message; never let that silently kill a run).
        for attempt in range(2):
            response = requests.post(
                f"{settings.groq_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.groq_blog_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.groq_blog_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=120,
            )
            if response.status_code == 429 or response.status_code >= 500:
                raise GroqTransientError(
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )
            response.raise_for_status()
            data = response.json()
            text = str(data["choices"][0]["message"]["content"] or "").strip()
            if text:
                break
            print(
                f"[AutoBlog] Groq returned empty text (attempt {attempt + 1}/2) "
                "— retrying...",
                flush=True,
            )
    except GroqTransientError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced as a pipeline error
        raise BlogPipelineError(f"Groq blog call failed: {exc}") from exc
    if not text:
        raise BlogPipelineError("Groq returned empty text.")
    return text


async def _groq_with_gemini_fallback(
    system: str, prompt: str, *, temperature: float, max_tokens: int
) -> str:
    """Groq first; on 429/5xx retry the SAME prompt/system via Gemini."""
    try:
        return await asyncio.to_thread(
            _groq_blog_chat,
            system,
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except GroqTransientError as exc:
        print(
            "[AutoBlog Background] Groq rate limit hit. Falling back to Gemini...",
            flush=True,
        )
        print(f"[AutoBlog Background] Groq error detail: {exc}", flush=True)
        settings = get_settings()
        if not settings.google_api_key:
            raise BlogPipelineError(
                "Groq failed and GOOGLE_API_KEY (Gemini fallback) is not configured.",
                status_code=503,
            ) from exc
        model = _gemini_model(system)
        return await _generate(
            model, prompt, temperature=temperature, max_tokens=max_tokens
        )


# ───────────────────── Gemini helpers (rate-limit fallback only) ─────────────


def _gemini_model(system_instruction: str) -> genai.GenerativeModel:
    settings = get_settings()
    if not settings.google_api_key:
        raise BlogPipelineError("GOOGLE_API_KEY is not configured.", status_code=503)
    genai.configure(api_key=settings.google_api_key)
    # Fresh model object per call — never shared across event loops.
    return genai.GenerativeModel(
        model_name=settings.blog_model,
        system_instruction=system_instruction,
    )


async def _generate(
    model: genai.GenerativeModel,
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    try:
        response = await model.generate_content_async(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        text = (response.text or "").strip()
    except Exception as exc:  # noqa: BLE001 - surfaced as a pipeline error
        raise BlogPipelineError(f"Gemini generation failed: {exc}") from exc
    if not text:
        raise BlogPipelineError("Gemini returned empty text.")
    return text


# ───────────────────────────── Stage 1 · Research ─────────────────────────────


def _month_year() -> str:
    """Current UTC month/year, e.g. 'October 2026'."""
    return _utc().strftime("%B %Y")


def _ordered_research_queries() -> list[tuple[str, str]]:
    """Render the query pool and rotate its starting point each UTC day."""
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
    """Lower is better. Unknown domains sort after the official sources."""
    host = _result_host(url)
    for index, domain in enumerate(OFFICIAL_SOURCE_DOMAINS):
        if host == domain or host.endswith(f".{domain}"):
            return index
    return len(OFFICIAL_SOURCE_DOMAINS)


def _tavily_search(
    query: str, include_domains: tuple[str, ...] | None = OFFICIAL_SOURCE_DOMAINS
) -> list[dict[str, Any]]:
    """Sync Tavily call — executed in a worker thread via asyncio.to_thread."""
    settings = get_settings()
    if not settings.tavily_api_key:
        raise BlogPipelineError("TAVILY_API_KEY is not configured.", status_code=503)
    client = TavilyClient(api_key=settings.tavily_api_key)
    args: dict[str, Any] = {
        "query": query,
        "topic": "news",
        "days": RESEARCH_DAYS,
        "search_depth": RESEARCH_SEARCH_DEPTH,
        "max_results": RESEARCH_MAX_RESULTS,
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


def _research_snippets(results: list[dict[str, str]]) -> list[str]:
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


async def _research_chunk(query: str) -> list[dict[str, str]]:
    """Step 1/6 (part A): Tavily news search (raises on failure).

    Official domains are searched first. If that filtered search returns
    nothing, one unrestricted news search keeps the pipeline alive.
    """
    try:
        raw = await asyncio.to_thread(_tavily_search, query)
    except Exception as exc:  # noqa: BLE001 - unrestricted retry below
        print(
            f"[AutoBlog] official-domain search failed for {query!r} "
            f"({type(exc).__name__}: {exc}) — retrying unrestricted.",
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


def _fetch_recent_posts(limit: int = RECENT_POSTS_LIMIT) -> list[dict[str, Any]]:
    """Fetch the latest published posts for topic deduplication (never raises)."""
    try:
        response = (
            _supabase()
            .table("blog_posts")
            .select("title, keywords, published_at")
            .order("published_at", desc=True)
            .limit(limit)
            .execute()
        )
        rows = getattr(response, "data", None)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    except Exception as exc:  # noqa: BLE001 - dedupe is best-effort
        print(
            f"[AutoBlog] recent-post lookup failed "
            f"({type(exc).__name__}: {exc}) — dedupe disabled for this run.",
            flush=True,
        )
    return []


def _blocked_topics(recent_posts: list[dict[str, Any]]) -> list[str]:
    """Flatten recent titles + keywords into a prompt-safe avoid list."""
    blocked: list[str] = []
    seen: set[str] = set()
    for post in recent_posts:
        values: list[Any] = [post.get("title")]
        keywords = post.get("keywords")
        if isinstance(keywords, list):
            values.extend(keywords)
        for value in values:
            text = str(value or "").strip()
            key = text.lower()
            if text and key not in seen and key not in _BLOCKED_TOPIC_JUNK:
                seen.add(key)
                blocked.append(text[:160])
    return blocked[:30]


def _topic_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def _topic_tokens(value: str) -> set[str]:
    return {
        token
        for token in _topic_text(value).split()
        if token not in _STOPWORDS and len(token) > 1
    }


def _topic_sequence_similarity(first: str, second: str) -> float:
    first_text, second_text = _topic_text(first), _topic_text(second)
    if not first_text or not second_text:
        return 0.0
    return difflib.SequenceMatcher(None, first_text, second_text).ratio()


def _topic_similarity(first: str, second: str) -> float:
    """Blend sequence similarity with token overlap (0.0-1.0)."""
    sequence = _topic_sequence_similarity(first, second)
    first_tokens, second_tokens = _topic_tokens(first), _topic_tokens(second)
    if not first_tokens or not second_tokens:
        return sequence
    overlap = len(first_tokens & second_tokens)
    jaccard = overlap / len(first_tokens | second_tokens)
    containment = overlap / min(len(first_tokens), len(second_tokens))
    return max(sequence, jaccard, containment)


def _find_duplicate(
    topic: str, recent_posts: list[dict[str, Any]]
) -> tuple[float, str]:
    """Compare against recent titles and keywords, returning the best match."""
    best_score = 0.0
    best_title = ""
    for post in recent_posts:
        title = str(post.get("title") or "").strip()
        if title:
            score = _topic_similarity(topic, title)
            if score > best_score:
                best_score, best_title = score, title
        keywords = post.get("keywords")
        if not isinstance(keywords, list):
            continue
        for keyword in keywords:
            text = str(keyword or "").strip()
            if not text or text.lower() in _BLOCKED_TOPIC_JUNK:
                continue
            # Sequence-only for short keywords: exact phrase matches still
            # reject, but a shared model name alone does not over-block.
            score = _topic_sequence_similarity(topic, text)
            if score > best_score:
                best_score, best_title = score, text
    return best_score, best_title


async def _topic_chunk(
    results: list[dict[str, str]],
    blocked_topics: list[str] | None = None,
    category: str | None = None,
) -> str:
    """Step 1/6 (part B): Nemotron Ultra picks the day's trending topic.

    Falls back to a curated daily topic when Tavily returned nothing, and to
    the first headline when Nemotron extraction fails (never raises).
    """
    if not results:
        topic = FALLBACK_TOPICS[_day_index() % len(FALLBACK_TOPICS)]
        print(f"[AutoBlog] research fallback -> topic={topic!r}")
        return topic

    blob = "\n".join(_result_line(result) for result in results)
    system = _TOPIC_SYSTEM
    if blocked_topics:
        avoid = "\n".join(f"- {topic[:160]}" for topic in blocked_topics[:20])
        system += (
            "\n\nRECENTLY PUBLISHED (strictly avoid these angles):\n"
            f"{avoid}\n"
            "Choose a clearly different story, model, company, or event."
        )
    category_line = f"Preferred category: {category}\n" if category else ""
    try:
        text = await asyncio.to_thread(
            _nim_chat,
            system,
            f"{category_line}Today's research:\n{blob}",
            temperature=0.3,
            max_tokens=2048,
        )
        data = ai_service.extract_json(text, prefer="topic")
        topic = str(data.get("topic") or "").strip() if isinstance(data, dict) else ""
        if topic:
            return topic[:120]
        print("[AutoBlog] topic chunk returned no topic — using headline fallback.")
    except Exception as exc:  # noqa: BLE001 - heuristics keep the run alive
        print(f"[AutoBlog] topic extraction failed (Nemotron): {type(exc).__name__}: {exc}")
    headline = results[0]["title"].split("|")[0].split(" - ")[0].strip()[:120]
    return headline or FALLBACK_TOPICS[_day_index() % len(FALLBACK_TOPICS)]


async def _research_and_select_topic(
    recent_posts: list[dict[str, Any]],
) -> tuple[Research, list[dict[str, str]]]:
    """Research, generate a topic, reject repeats, then force a new category."""
    queries = _ordered_research_queries()
    blocked = _blocked_topics(recent_posts)
    attempted: list[str] = []
    results: list[dict[str, str]] = []
    topic = ""

    for attempt in range(TOPIC_RETRY_LIMIT + 1):
        category, query = queries[attempt % len(queries)]
        print(
            f"[AutoBlog] research attempt {attempt + 1}/"
            f"{TOPIC_RETRY_LIMIT + 1}: {category} -> {query!r}",
            flush=True,
        )
        results = await _research_chunk(query)
        topic = await _topic_chunk(
            results, [*blocked, *attempted], category=category
        )
        score, match = _find_duplicate(topic, recent_posts)
        print(
            f"[AutoBlog] candidate topic={topic!r} similarity={score:.2f}"
            + (f" match={match!r}" if match else ""),
            flush=True,
        )
        if score <= SIMILARITY_THRESHOLD:
            return (
                Research(
                    topic=topic,
                    keywords=[],
                    snippets=_research_snippets(results),
                ),
                results,
            )
        attempted.append(topic)
        print(
            f"[AutoBlog] duplicate topic rejected ({score:.0%} similar) — retrying.",
            flush=True,
        )

    used_categories = {category for category, _ in queries[: TOPIC_RETRY_LIMIT + 1]}
    forced_category, forced_query = next(
        (
            (category, query)
            for category, query in queries
            if category not in used_categories
        ),
        queries[-1],
    )
    print(
        f"[AutoBlog] forcing a different category: {forced_category} -> "
        f"{forced_query!r}",
        flush=True,
    )
    results = await _research_chunk(forced_query)
    topic = await _topic_chunk(
        results, [*blocked, *attempted], category=forced_category
    )
    score, match = _find_duplicate(topic, recent_posts)
    if score > SIMILARITY_THRESHOLD:
        print(
            f"[AutoBlog] forced topic still {score:.0%} similar to {match!r}; "
            "accepting it to avoid an empty run.",
            flush=True,
        )
    return (
        Research(
            topic=topic,
            keywords=[],
            snippets=_research_snippets(results),
        ),
        results,
    )


async def _keyword_chunk(topic: str, results: list[dict[str, str]]) -> list[str]:
    """Step 2/6: Nemotron Ultra extracts the top 3-5 SEO keywords.

    Heuristic padding (_pad_keywords) keeps the list usable when NIM fails.
    """
    blob = "\n".join(_result_line(result) for result in results) or (
        "(no research snippets)"
    )
    try:
        text = await asyncio.to_thread(
            _nim_chat,
            _KEYWORD_SYSTEM,
            f"Topic: {topic}\nResearch snippets:\n{blob}",
            temperature=0.3,
            max_tokens=2048,
        )
        data = ai_service.extract_json(text, prefer="keywords")
        raw = data.get("keywords") if isinstance(data, dict) else None
        keywords = _pad_keywords(
            [str(k) for k in raw] if isinstance(raw, list) else [], topic
        )
        print(f"[AutoBlog] keyword extraction OK -> {', '.join(keywords)}")
        return keywords
    except Exception as exc:  # noqa: BLE001 - heuristics keep the run alive
        print(f"[AutoBlog] keyword extraction failed (Nemotron): {type(exc).__name__}: {exc}")
        return _pad_keywords([], topic)


async def _meta_chunk(research: Research) -> tuple[str, str]:
    """Step 3/6: catchy meta title + meta description via Groq (Gemini fallback).

    Plain (non-LLM) fallbacks keep the run alive when BOTH providers fail.
    """
    sources = "\n".join(f"- {s}" for s in research.snippets[:3]) or "(none)"
    primary_keyword = research.keywords[0] if research.keywords else research.topic
    other_keywords = ", ".join(research.keywords[1:]) or "(none)"
    try:
        text = await _groq_with_gemini_fallback(
            _META_SYSTEM,
            f"Topic: {research.topic}\n"
            f"Primary keyword: {primary_keyword}\n"
            f"Other keywords: {other_keywords}\n"
            f"Research snippets:\n{sources}",
            temperature=0.7,
            max_tokens=1024,
        )
        data = ai_service.extract_json(text, prefer="meta_title")
        if isinstance(data, dict):
            title = str(data.get("meta_title") or "").strip()
            description = str(data.get("meta_description") or "").strip()
            if title and description:
                return title[:300], description[:300]
        print("[AutoBlog] meta chunk returned incomplete JSON — using fallbacks.")
    except Exception as exc:  # noqa: BLE001 - fallback keeps the run alive
        print(f"[AutoBlog] meta generation failed (Groq/Gemini): {type(exc).__name__}: {exc}")
    fallback_desc = (
        research.snippets[0][:170].rstrip() + "…"
        if research.snippets
        else research.topic
    )
    # Keep the SEO gate passable even on the non-LLM path: the title must
    # contain the primary keyword, so append it when the topic lacks it.
    fallback_title = research.topic
    if (
        primary_keyword
        and primary_keyword.lower() not in fallback_title.lower()
    ):
        fallback_title = f"{fallback_title}: {primary_keyword}"
    return fallback_title[:300], fallback_desc


# ────────────────────────────── Stage 2 · Write ───────────────────────────────


async def _write_article(
    research: Research, seo_feedback: str | None = None
) -> dict[str, Any]:
    sources = "\n".join(f"- {s}" for s in research.snippets) or "(none)"
    primary_keyword = research.keywords[0] if research.keywords else research.topic
    secondary_keywords = ", ".join(research.keywords[1:]) or "(none)"
    user_prompt = (
        f"Trending topic: {research.topic}\n"
        f"Primary keyword: {primary_keyword}\n"
        f"Secondary keywords (weave in naturally): {secondary_keywords}\n"
        f"Today's research (ground the article in these, link 2-3 inline):\n"
        f"{sources}\n\n"
        "Write the article now. Open by answering the reader's core question "
        "directly in the first 2 sentences, with no wind-up. Include the exact "
        "primary keyword 5-7 times naturally, plus once in the first 100 words, "
        "once in an <h2>, and once in the closing section. Use short paragraphs "
        "(1-3 sentences), at least one <ul><li> list, and 3-5 body sections. "
        "End with a mandatory <h2>FAQ</h2> section containing exactly 3-4 "
        "<h3>questions</h3>, each answered in 40-60 words. Target 800-1200 "
        "words. Banned words fail the brief. Write in simple, human-like "
        "language. 100% unique, no plagiarism."
    )
    if seo_feedback:
        user_prompt += (
            "\n\nIMPORTANT: a lightweight SEO checker rejected your previous "
            f"draft. Fix ALL of these issues in this rewrite:\n- {seo_feedback}\n"
            "Keep the same topic, keywords and research; make the fixes "
            "minimally and keep the article between 800 and 1200 words. Keep "
            "the direct 2-sentence opening, 5-7 uses of the primary keyword, "
            "and the final <h2>FAQ</h2> section with 3-4 Q&A pairs."
        )

    text = await _groq_with_gemini_fallback(
        _WRITER_SYSTEM, user_prompt, temperature=0.8, max_tokens=4096
    )

    try:
        data = ai_service.extract_json(text, prefer="tags")
    except Exception:
        try:
            data = ai_service.extract_json(text)  # tags optional → still salvage
        except Exception as exc:  # noqa: BLE001
            raise BlogPipelineError(f"Could not parse the article JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise BlogPipelineError("Article JSON was not an object.")

    content_html = str(data.get("content_html") or data.get("content") or "").strip()
    if not content_html:
        raise BlogPipelineError("Article had no content_html.")

    title = str(data.get("title") or "").strip()[:300] or research.topic[:300]
    plain = re.sub(r"<[^>]+>", " ", content_html)
    plain = re.sub(r"\s+", " ", plain).strip()
    excerpt = str(data.get("excerpt") or "").strip() or plain[:170].rstrip() + "…"

    tags_raw = data.get("tags")
    tags = (
        [str(t).strip().lower() for t in tags_raw if str(t).strip()]
        if isinstance(tags_raw, list)
        else []
    )
    seen: list[str] = []
    for tag in tags:
        if tag not in seen:
            seen.append(tag)
    tags = seen[:5] or research.keywords[:5]

    words = len(plain.split())
    if words < 500:
        print(f"[AutoBlog] WARNING — article is only {words} words (target ~1000)")
    else:
        print(f"[AutoBlog] article written — {words} words")

    return {"title": title, "excerpt": excerpt, "content_html": content_html, "tags": tags}


# ─────────────── Stage 3 · Cover + in-article images (Pixabay first) ─────────

# Final fallback when Pixabay has nothing for ANY query in the ladder — a
# small pool of stable Unsplash tech photos (not generic business stock).
# The pool lets content images stay unique when the cover already used the
# first fallback.
UNSPLASH_FALLBACK_IMAGES: tuple[str, ...] = (
    "https://images.unsplash.com/photo-1677442136019-21780ecad995?w=1200",
    "https://images.unsplash.com/photo-1737071371043-761e02b1ef95?w=1200&auto=format&fit=crop",
    "https://images.unsplash.com/photo-1518770660439-4636190af475?w=1200",
    "https://images.unsplash.com/photo-1550751827-4bd374c3f58b?w=1200",
)
UNSPLASH_FALLBACK_IMAGE = UNSPLASH_FALLBACK_IMAGES[0]

PIXABAY_UNSWORTHY_HITS = frozenset(
    {
        "cake", "food", "eat", "eating", "eaten", "dinner", "lunch", "breakfast",
        "restaurant", "meal", "cookie", "candy", "sugar", "bakery", "dessert",
    }
)

# When the search query contains any of these, food-tagged Pixabay hits are
# decoys (e.g. "AI Is Eating Software" must not return a cake photo).
_TECH_CONTEXT_WORDS = frozenset(
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

_TECH_IMAGE_TAGS = frozenset(
    {
        "ai", "artificial", "intelligence", "technology", "computer", "software",
        "code", "coding", "programming", "developer", "robot", "robotics",
        "machine", "data", "cyber", "digital", "algorithm", "startup",
        "internet", "electronics", "hardware", "science", "engineering",
    }
)

_GENERIC_BUSINESS_TAGS = frozenset(
    {
        "business", "office", "meeting", "handshake", "corporate", "finance",
        "money", "chart", "graph", "team", "manager", "suit", "desk",
    }
)

# Specific AI model names make much better image queries than "AI technology".
_AI_MODEL_RE = re.compile(
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

_PIXABAY_CATEGORY_HINTS: tuple[tuple[str, str], ...] = (
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


def _query_is_techy(text: str) -> bool:
    return bool(set(re.findall(r"[a-z]+", text.lower())) & _TECH_CONTEXT_WORDS)


def _extract_ai_models(text: str) -> list[str]:
    models: list[str] = []
    seen: set[str] = set()
    for match in _AI_MODEL_RE.finditer(text):
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
    for hint, category in _PIXABAY_CATEGORY_HINTS:
        if hint in text:
            return category
    return "computer"


def _pick_hit(
    hits: list[dict], query: str, exclude_urls: set[str] | None = None
) -> str:
    """Pick the best hit: skip food decoys and prefer tech/coding imagery.

    Literal title words (e.g. "AI Is Eating Software") otherwise surface
    cake/food photos. On tech queries, hits tagged with coding/robotics/science
    outrank generic business/office stock, while the API order breaks ties.
    URLs in `exclude_urls` (the cover and earlier in-article images) are
    skipped so the same photo is never used twice.
    """
    excluded = exclude_urls or set()
    query_is_techy = _query_is_techy(query)
    ranked: list[tuple[int, int, dict]] = []
    for index, hit in enumerate(hits):
        tag_words = set(re.findall(r"[a-z]+", str(hit.get("tags") or "").lower()))
        if query_is_techy and tag_words & PIXABAY_UNSWORTHY_HITS:
            continue
        score = 0
        if query_is_techy:
            score += 2 * len(tag_words & _TECH_IMAGE_TAGS)
            score -= 2 * len(tag_words & _GENERIC_BUSINESS_TAGS)
        ranked.append((score, -index, hit))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    for _, _, hit in ranked:
        url = str(hit.get("largeImageURL") or "").strip()
        if url and url not in excluded:
            return url

    # Every hit was filtered out — take the first usable URL anyway.
    for hit in hits:
        url = str(hit.get("largeImageURL") or "").strip()
        if url and url not in excluded:
            return url
    return ""


def _unused_unsplash_fallback(
    exclude_urls: set[str] | None = None,
) -> str | None:
    """Return the first fallback photo not already used in this post."""
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
    """Fetch a relevant photo from Pixabay using specific news/model terms.

    Query ladder (first query with hits wins):
      1. detected AI model name(s) + " AI"
      2. first 6 words of the title (+ " AI" when absent)
      3. first 3 article keywords (+ " AI" when absent)
      4. "<category> AI"
      5. the bare title topic

    Tech queries also send a valid Pixabay category (computer/science/
    education/business) and rank coding/robotics tags above generic business
    stock. `exclude_urls` skips photos already used as the cover or in an
    earlier section, so each call can return the next unique image. Returns
    the largeImageURL, a stable Unsplash tech fallback when nothing relevant
    is found, or None when there is no API key or every fallback is excluded.
    """
    excluded = exclude_urls or set()
    if not PIXABAY_API_KEY:
        return None

    kw = [str(k).strip() for k in (keywords or []) if str(k).strip()]
    queries = _image_queries(title, kw, category)
    query_is_techy = _query_is_techy(f"{title} {' '.join(kw)}")
    pixabay_category = _pixabay_category(title, kw) if query_is_techy else None

    async with httpx.AsyncClient(timeout=10.0) as client:
        for query in queries:
            params: dict[str, Any] = {
                "key": PIXABAY_API_KEY,
                "q": query[:100],
                "image_type": "photo",
                "orientation": "horizontal",
                "safesearch": "true",
                "min_width": 1200,
                "min_height": 800,
                "order": "popular",
                # Valid range is 3-200; we pick the best of the 5.
                "per_page": 5,
            }
            if pixabay_category:
                params["category"] = pixabay_category
            try:
                response = await client.get("https://pixabay.com/api/", params=params)
                response.raise_for_status()
                data = response.json()
            except Exception as exc:  # noqa: BLE001 - try the next query
                print(
                    f"[AutoBlog] Pixabay query {query!r} failed "
                    f"({type(exc).__name__}: {exc}) — trying next.",
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
            f"[AutoBlog] Pixabay found nothing for {title!r} — using Unsplash fallback.",
            flush=True,
        )
    else:
        print(
            f"[AutoBlog] Pixabay found only duplicate images for {title!r} — skipping.",
            flush=True,
        )
    return fallback


MAX_COVER_DOWNLOAD_BYTES = 15 * 1024 * 1024


async def _persist_cover_image(url: str, slug: str) -> str:
    """Download a cover, upload a compressed WebP to R2, return its public URL.

    Pixabay ``/get/...`` URLs expire, so they must never be saved. Any
    download/upload failure falls back to the stable Unsplash image.
    """
    if not url or url in UNSPLASH_FALLBACK_IMAGES:
        return UNSPLASH_FALLBACK_IMAGE

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
    except Exception as exc:  # noqa: BLE001 - never publish a dead cover
        print(
            f"[AutoBlog] cover R2 persistence failed "
            f"({type(exc).__name__}: {exc}) - using Unsplash fallback.",
            flush=True,
        )
        return UNSPLASH_FALLBACK_IMAGE


_H2_RE = re.compile(r"(<h2[^>]*>.*?</h2>)", re.IGNORECASE | re.DOTALL)
_FAQ_H2_RE = re.compile(
    r"<h2[^>]*>\s*(?:FAQ\b|Frequently Asked Questions\b)", re.IGNORECASE
)
_TAG_RE = re.compile(r"<[^>]+>")


async def _cover_image(title: str, keywords: list[str]) -> tuple[str, str]:
    """Build the cover, returning (durable URL, source URL to dedupe against).

    The source URL is the Pixabay photo before it is persisted to R2; content
    images must skip it because the persisted R2 URL no longer matches.
    """
    candidate = await fetch_relevant_image(
        title, keywords, category="technology"
    )
    if not candidate:
        return UNSPLASH_FALLBACK_IMAGE, UNSPLASH_FALLBACK_IMAGE
    cover = await _persist_cover_image(candidate, slugify(title))
    return cover, candidate


async def _cover_image_url(title: str, keywords: list[str]) -> str:
    """Fetch a cover and return only a durable R2 or Unsplash URL."""
    cover, _ = await _cover_image(title, keywords)
    return cover


async def _content_image_url(
    section_title: str,
    keyword: str,
    exclude_urls: set[str] | None = None,
) -> str | None:
    """Pixabay (section title + article keyword); unused Unsplash fallback.

    Returns None when every available photo is already used in this post.
    """
    url = await fetch_relevant_image(
        section_title,
        [keyword] if keyword else None,
        category="technology",
        exclude_urls=exclude_urls,
    )
    return url or _unused_unsplash_fallback(exclude_urls)


def _figure_html(url: str, caption: str) -> str:
    """Plain <img> — no <figure> wrapper box; caption text stays in the DB
    metadata and the alt attribute (nothing is rendered under the image)."""
    alt = html.escape(caption, quote=True)
    return (
        f'<img src="{url}" alt="{alt}" loading="lazy" '
        'width="800" height="450" />'
    )


async def _images_chunk(
    content_html: str,
    keywords: list[str],
    *,
    max_images: int | None = None,
    exclude_urls: set[str] | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """Step 5/6: Pixabay images ONE BY ONE (pause between each).

    Fetches a random 1-3 in-article images (one per eligible <h2>, or a single
    image after the first paragraph when the article has no <h2>). The cover
    URL and every in-article URL already chosen are excluded, so an image can
    never appear twice in the same post. Every image falls back to an unused
    stable Unsplash tech image when Pixabay returns nothing unique.
    """
    target = (
        max_images
        if max_images is not None
        else random.randint(MIN_CONTENT_IMAGES, MAX_CONTENT_IMAGES)
    )
    target = max(0, min(target, MAX_CONTENT_IMAGES))
    used = set(exclude_urls or ())

    parts = _H2_RE.split(content_html)
    if len(parts) < 3:
        if not keywords or target == 0:
            return content_html, []
        caption = _TAG_RE.sub(" ", parts[0])[:120].strip() or "Article illustration"
        keyword = keywords[0] if keywords else "technology"
        url = await _content_image_url(caption, keyword, exclude_urls=used)
        await asyncio.sleep(IMAGE_PAUSE_SECONDS)
        if not url or url in used:
            print(
                "[AutoBlog] no unique content image for the intro — skipping.",
                flush=True,
            )
            return content_html, []
        used.add(url)
        figure = _figure_html(url, caption)
        if "</p>" in content_html:
            head, tail = content_html.split("</p>", 1)
            html_out = f"{head}</p>{figure}{tail}"
        else:
            html_out = f"{content_html}{figure}"
        return html_out, [{"url": url, "caption": caption}]

    images: list[dict[str, str]] = []
    out: list[str] = [parts[0]]
    for section, i in enumerate(range(1, len(parts), 2)):
        h2 = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        # Never inject an image inside the FAQ block: the AEO answer format
        # (<h3>question</h3><p>answer</p>) must stay uninterrupted.
        should_fetch = (
            len(images) < target and keywords and not _FAQ_H2_RE.search(h2)
        )
        if should_fetch:
            caption = _TAG_RE.sub(" ", h2).strip()[:120] or f"Section {section + 1}"
            url = await _content_image_url(
                caption,
                keywords[section % len(keywords)],
                exclude_urls=used,
            )
            if images:
                await asyncio.sleep(IMAGE_PAUSE_SECONDS)
            if not url or url in used:
                print(
                    f"[AutoBlog] duplicate content image skipped: {caption!r}",
                    flush=True,
                )
                out.append(h2)
                out.append(body)
                continue
            used.add(url)
            print(
                f"[AutoBlog] content image {len(images) + 1}/{target} "
                f"ready: {caption!r}"
            )
            body = f"{body}{_figure_html(url, caption)}"
            images.append({"url": url, "caption": caption})
        out.append(h2)
        out.append(body)
    return "".join(out), images


# ────────────────────────────── Stage 4 · Save ────────────────────────────────


@lru_cache(maxsize=1)
def _supabase() -> Client:
    """Service-role client (bypasses RLS — the only role that may write blog_posts)."""
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise BlogPipelineError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required to save posts.",
            status_code=503,
        )
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


def _insert_post(payload: dict[str, Any]) -> dict[str, Any]:
    """Sync supabase-py insert (runs in a worker thread). Retries slug clashes."""
    client = _supabase()
    base_slug = payload.pop("slug_base")
    last_error: Exception | None = None

    for attempt in range(4):
        slug = (
            base_slug
            if attempt == 0
            else f"{base_slug}-{_utc().strftime('%Y%m%d')}"
            if attempt == 1
            else f"{base_slug}-{uuid.uuid4().hex[:6]}"
        )
        try:
            result = client.table("blog_posts").insert({**payload, "slug": slug}).execute()
            if result.data:
                return dict(result.data[0])
            raise BlogPipelineError("Supabase insert returned no row.")
        except BlogPipelineError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if "duplicate" in str(exc).lower() and attempt < 3:
                continue  # same slug exists → date/random suffix
            break

    raise BlogPipelineError(f"Supabase insert failed: {last_error}") from last_error


# ──────────────────────── Staggered background orchestrator ──────────────────


def is_run_active() -> bool:
    """True while a pipeline run holds the lock (route 409 pre-check)."""
    return _RUN_LOCK.locked()


async def run_staggered_blog_pipeline() -> dict[str, Any] | None:
    """Staggered 6-chunk pipeline with deliberate API pauses (never raises).

    Designed to run inside a FastAPI BackgroundTask:
      - the FIRST line prints "TASK STARTED EXECUTION!" so we can prove the
        task actually entered the coroutine,
      - the ENTIRE body is wrapped in one try/except that prints
        "CRITICAL TASK ERROR" + a traceback, so nothing can die silently,
      - every step logs a [AutoBlog Background] Step N line (flushed
        immediately) before and after the call,
      - sleeps use `asyncio.sleep` only (this is an `async def` running on
        the event loop — `time.sleep` here would block every request).

    Returns the saved row plus research metadata, or None when any chunk
    failed — the run aborts gracefully without ever crashing the server.
    """
    print("[AutoBlog Background]  TASK STARTED EXECUTION!", flush=True)
    try:
        if not _RUN_LOCK.acquire(blocking=False):
            print(
                "[AutoBlog Background] a run is already in progress — "
                "staggered run skipped.",
                flush=True,
            )
            return None
        try:
            print(
                f"[AutoBlog Background] staggered pipeline start "
                f"{_utc().isoformat(timespec='seconds')}",
                flush=True,
            )

            # Step 1/6 — Recent-post dedupe + rotating Tavily news research.
            results: list[dict[str, str]] = []
            print(
                "[AutoBlog Background] Step 1: Starting research (recent-post "
                "dedupe + rotating Tavily news + NVIDIA Nemotron topic)...",
                flush=True,
            )
            try:
                recent_posts = await asyncio.to_thread(
                    _fetch_recent_posts, RECENT_POSTS_LIMIT
                )
                print(
                    f"[AutoBlog Background] Step 1/6: Loaded "
                    f"{len(recent_posts)} recent post(s) for dedupe.",
                    flush=True,
                )
                research, results = await _research_and_select_topic(recent_posts)
                print(
                    f"[AutoBlog Background] Step 1/6: Research complete "
                    f"({len(results)} results, topic={research.topic!r}). "
                    f"Waiting {RESEARCH_PAUSE_SECONDS}s...",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - curated fallback keeps the run alive
                topic = FALLBACK_TOPICS[_day_index() % len(FALLBACK_TOPICS)]
                research = Research(topic=topic, keywords=[], snippets=[])
                print(
                    f"[AutoBlog Background] Step 1/6: Research failed "
                    f"({type(exc).__name__}: {exc}); fallback topic. "
                    f"Waiting {RESEARCH_PAUSE_SECONDS}s...",
                    flush=True,
                )
            await asyncio.sleep(RESEARCH_PAUSE_SECONDS)

            # Step 2/6 — Keywords (Nemotron Ultra).
            print(
                "[AutoBlog Background] Step 2: Extracting keywords "
                "(NVIDIA Nemotron Ultra)...",
                flush=True,
            )
            try:
                research.keywords = await _keyword_chunk(research.topic, results)
                print(
                    f"[AutoBlog Background] Step 2/6: Keywords complete "
                    f"({', '.join(research.keywords)}). "
                    f"Waiting {RESEARCH_PAUSE_SECONDS}s...",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[AutoBlog Background] Step 2/6: Keywords failed "
                    f"({type(exc).__name__}: {exc}). Aborting run.",
                    flush=True,
                )
                return None
            await asyncio.sleep(RESEARCH_PAUSE_SECONDS)

            # Step 3/6 — Meta title + description (Groq → Gemini on 429/5xx).
            print(
                "[AutoBlog Background] Step 3: Generating meta title and "
                "description (Groq → Gemini fallback)...",
                flush=True,
            )
            try:
                meta_title, meta_description = await _meta_chunk(research)
                print(
                    f"[AutoBlog Background] Step 3/6: Meta data complete. "
                    f"Waiting {WRITE_PAUSE_SECONDS}s...",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[AutoBlog Background] Step 3/6: Meta failed "
                    f"({type(exc).__name__}: {exc}). Aborting run.",
                    flush=True,
                )
                return None
            await asyncio.sleep(WRITE_PAUSE_SECONDS)

            # Step 4/6 — Article writing (Groq → Gemini on 429/5xx).
            print(
                "[AutoBlog Background] Step 4: Writing article "
                "(Groq → Gemini fallback)...",
                flush=True,
            )
            try:
                article = await _write_article(research)
                print(
                    f"[AutoBlog Background] Step 4/6: Article written. "
                    f"Waiting {WRITE_PAUSE_SECONDS}s...",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[AutoBlog Background] Step 4/6: Writing failed "
                    f"({type(exc).__name__}: {exc}). Aborting run.",
                    flush=True,
                )
                return None
            await asyncio.sleep(WRITE_PAUSE_SECONDS)

            # SEO Gate — lightweight on-page rules (stdlib only, zero RAM).
            # A failing draft gets ONE rewrite with the exact feedback; a
            # second failure discards the run WITHOUT saving anything.
            seo_keyword = research.keywords[0] if research.keywords else research.topic
            seo_report = evaluate_seo_lightweight(
                meta_title, article["content_html"], seo_keyword, meta_description
            )
            if not seo_report["passed"]:
                print(
                    f"[AutoBlog Background] SEO gate score "
                    f"{seo_report['score']}/100 — {seo_report['feedback']} "
                    "One rewrite attempt...",
                    flush=True,
                )
                try:
                    article = await _write_article(
                        research, seo_feedback=seo_report["feedback"]
                    )
                except Exception as exc:  # noqa: BLE001 - discard, never crash
                    print(
                        "[AutoBlog Background] SEO gate rewrite failed "
                        f"({type(exc).__name__}: {exc}). Draft discarded "
                        "(nothing saved).",
                        flush=True,
                    )
                    return None
                seo_report = evaluate_seo_lightweight(
                    meta_title, article["content_html"], seo_keyword, meta_description
                )
            if not seo_report["passed"]:
                print(
                    f"[AutoBlog Background] SEO gate FAILED "
                    f"({seo_report['score']}/100) — {seo_report['feedback']} "
                    "Draft discarded (nothing saved).",
                    flush=True,
                )
                return None
            print(
                f"[AutoBlog Background] SEO gate passed "
                f"({seo_report['score']}/100).",
                flush=True,
            )

            # Step 5/6 — Images (cover first, then 1-3 unique in-article images).
            print(
                "[AutoBlog Background] Step 5: Fetching images "
                "(Pixabay, fallback Unsplash)...",
                flush=True,
            )
            try:
                cover, cover_source = await _cover_image(
                    meta_title, research.keywords
                )
                content_html, content_images = await _images_chunk(
                    article["content_html"],
                    research.keywords,
                    exclude_urls={cover, cover_source},
                )
                print(
                    f"[AutoBlog Background] Step 5/6: Images complete "
                    f"({len(content_images)} in-article + cover).",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[AutoBlog Background] Step 5/6: Images failed "
                    f"({type(exc).__name__}: {exc}). Aborting run.",
                    flush=True,
                )
                return None

            # Step 6/6 — Upload (Supabase).
            print(
                "[AutoBlog Background] Step 6: Saving post to Supabase...",
                flush=True,
            )
            try:
                payload = {
                    "slug_base": slugify(meta_title),
                    "title": meta_title,
                    "content": content_html,
                    "excerpt": meta_description,
                    "cover_image_url": cover,
                    "content_images": content_images,
                    "tags": article["tags"],
                    "keywords": research.keywords,
                }
                post = await asyncio.to_thread(_insert_post, payload)
                clear_read_caches()  # new post -> drop the public read caches
                print(
                    f"[AutoBlog Background] Step 6/6: Saved post "
                    f"id={post.get('id')} slug={post.get('slug')}",
                    flush=True,
                )
                print("[AutoBlog Background]  TASK COMPLETED SUCCESSFULLY!", flush=True)
                return {
                    "post": post,
                    "topic": research.topic,
                    "keywords": research.keywords,
                }
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[AutoBlog Background] Step 6/6: Upload failed "
                    f"({type(exc).__name__}: {exc}). Aborting run.",
                    flush=True,
                )
                return None
        finally:
            _RUN_LOCK.release()
    except Exception as e:
        # Global catch — a background task raises INTO the void, so this is
        # the only place the exact failure can surface in the nohup log.
        print(f"[AutoBlog Background] 💥 CRITICAL TASK ERROR: {e}", flush=True)
        import traceback

        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        return None


async def run_pipeline() -> dict[str, Any] | None:
    """Backwards-compatible alias — runs the staggered pipeline."""
    return await run_staggered_blog_pipeline()


def run_scheduled_job() -> None:
    """APScheduler entry point (runs on a worker thread — owns its own loop)."""
    try:
        result = asyncio.run(run_staggered_blog_pipeline())
        if result:
            print(f"[AutoBlog] scheduled run OK -> /blog/{result['post'].get('slug')}")
        else:
            print("[AutoBlog] scheduled run exited gracefully (see error above)")
    except Exception as exc:  # noqa: BLE001 - never kill the scheduler thread
        print(f"[AutoBlog] scheduled run failed: {type(exc).__name__}: {exc}")
