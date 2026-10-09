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
                 (PIXABAY_API_KEY). Pollinations.ai URL is the
                 per-image fallback when Pixabay has no hits.   [step 5]
  - Supabase   : save the finished post.                       [step 6]

The pipeline is deliberately split into chunks with pauses between them so
no provider sees a burst of back-to-back calls (avoids 429s):

  1. Research      — Tavily search in the "AI & Technology" niche, then
                     Nemotron Ultra picks ONE trending topic.     [2s]
  2. Keywords      — Nemotron Ultra distills the top 3-5 SEO
                     keywords for the topic (heuristic fallback). [2s]
  3. Meta          — Groq writes the catchy meta title + meta
                     description (plain fallbacks; Gemini retry on
                     429/5xx).                                    [3s]
  4. Write         — Groq writes a ~1000-word, SEO-optimised article
                     as strict JSON (title / excerpt / content_html /
                     tags); same Gemini retry on 429/5xx.         [3s]
  5. Images        — Pixabay cover + up to three in-article photos,
                     fetched ONE BY ONE with a 3s pause between each
                     (Pollinations URL fallback per image). The URLs
                     are embedded as <figure> tags and stored in
                     blog_posts.content_images.
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
import html
import os
import re
import sys
import threading
import uuid
import urllib.parse
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
from app.config import get_settings

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

# How many in-article images to embed (one per H2 section, first N sections).
MAX_CONTENT_IMAGES = 3

# Deliberate pauses between chunks — keep every provider under its rate limit.
RESEARCH_PAUSE_SECONDS = 2  # after topic (step 1) and keywords (step 2)
WRITE_PAUSE_SECONDS = 3  # after meta (step 3) and article (step 4)
IMAGE_PAUSE_SECONDS = 3  # between in-article image generations

# Single niche for the daily research query.
NICHE = "AI & Technology"

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

_WRITER_SYSTEM = """You are a senior staff writer for ATLAS, a learning OS for students. Your articles get published on our public blog.

Non-negotiable voice rules:
- Write like a sharp friend explaining something interesting: contractions, direct "you", occasional rhetorical questions. Warm, specific, opinionated. A human wrote this.
- BANNED words and phrases (using any of them fails the brief): delve, delve into, moreover, furthermore, in conclusion, additionally, it's worth noting, it is important to note, in today's fast-paced world, ever-evolving landscape, landscape of, tapestry, testament, testament to, unlock, unleash, dive deep, dive into, game-changer, revolutionize, revolutionary, seamless, robust, cutting-edge, elevate, harness the power, in a world where, boasts, myriad, plethora, navigating, embarking, fostering, realm.
- Short paragraphs only: 1-3 sentences each. White space is a feature.
- Use bullet lists (HTML <ul><li>) for tips, steps or facts — at least one list per article.
- Never use em dashes (—) or en dashes (–) anywhere in the title, excerpt or body. Use commas, colons or parentheses instead.
- Structure: 3-5 sections, each opened by an engaging <h2>; <h3> where it helps. Plain HTML fragment — never <html>, <head> or <body>.
- SEO: work the primary keyword into the first 100 words and into one <h2>; use the other keywords naturally, never stuffed.
- Ground claims in the research provided. Link 2-3 of the source URLs inline with <a href="..." rel="noopener">. If unsure about a statistic, phrase it generally — never invent studies or numbers.
- Length: about 1000 words (900-1100).

Return ONLY strict JSON (no markdown fences, no commentary before or after):
{"title": "...", "excerpt": "...", "content_html": "...", "tags": ["...", "..."]}

- title: max 110 chars, specific and engaging, no clickbait lies.
- excerpt: 140-180 chars of plain text containing the primary keyword (used as the meta description).
- content_html: the full article body as an HTML fragment.
- tags: 3-5 lowercase topic tags."""

_TOPIC_SYSTEM = """You are an SEO trend analyst for ATLAS, a learning OS for students.
From the research snippets, pick ONE specific, currently trending, searchable blog topic (not a generic pillar term).

Return ONLY strict JSON (no fences, no commentary):
{"topic": "..."}

- topic: a concrete angle a reader would click, max 120 chars."""

_KEYWORD_SYSTEM = """You are an SEO content strategist for ATLAS, a learning OS for students.
Extract the top 3 to 5 SEO keywords for the given blog topic.

Return ONLY strict JSON (no fences, no commentary):
{"keywords": ["...", "...", "..."]}

- keywords: 3 to 5, each 1-3 words, lowercase, no dupes, primary keyword first."""

_META_SYSTEM = """You are an SEO meta-copy expert. From the topic and keywords provided, write ONE catchy meta title and ONE meta description for the blog post.

Return ONLY strict JSON (no fences, no commentary):
{"meta_title": "...", "meta_description": "..."}

- meta_title: catchy and specific, max 110 chars, may include a number or power word.
- meta_description: 140-180 chars, contains the primary keyword, entices a click. Plain text only.
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


def _tavily_search(query: str) -> list[dict[str, Any]]:
    """Sync Tavily call — executed in a worker thread via asyncio.to_thread."""
    settings = get_settings()
    if not settings.tavily_api_key:
        raise BlogPipelineError("TAVILY_API_KEY is not configured.", status_code=503)
    client = TavilyClient(api_key=settings.tavily_api_key)
    response = client.search(
        query=query,
        topic="general",
        search_depth="basic",
        max_results=MAX_RESULTS,
    )
    results = response.get("results") if isinstance(response, dict) else None
    return results if isinstance(results, list) else []


def _clean_result(item: dict[str, Any]) -> dict[str, str]:
    return {
        "title": str(item.get("title") or "").strip(),
        "url": str(item.get("url") or "").strip(),
        "content": str(item.get("content") or "").strip(),
    }


async def _research_chunk() -> list[dict[str, str]]:
    """Step 1/6 (part A): Tavily trending-topic search (raises on failure)."""
    query = f"{NICHE} {_utc().year} trending"
    raw = await asyncio.to_thread(_tavily_search, query)
    return [
        r for r in (_clean_result(item) for item in raw) if r["title"] or r["content"]
    ]


async def _topic_chunk(results: list[dict[str, str]]) -> str:
    """Step 1/6 (part B): Nemotron Ultra picks the day's trending topic.

    Falls back to a curated daily topic when Tavily returned nothing, and to
    the first headline when Nemotron extraction fails (never raises).
    """
    if not results:
        topic = FALLBACK_TOPICS[_day_index() % len(FALLBACK_TOPICS)]
        print(f"[AutoBlog] research fallback -> topic={topic!r}")
        return topic

    blob = "\n".join(
        f"- {r['title']} ({r['url']}): {r['content'][:500]}" for r in results
    )
    try:
        text = await asyncio.to_thread(
            _nim_chat,
            _TOPIC_SYSTEM,
            f"Today's research:\n{blob}",
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


async def _keyword_chunk(topic: str, results: list[dict[str, str]]) -> list[str]:
    """Step 2/6: Nemotron Ultra extracts the top 3-5 SEO keywords.

    Heuristic padding (_pad_keywords) keeps the list usable when NIM fails.
    """
    blob = (
        "\n".join(
            f"- {r['title']} ({r['url']}): {r['content'][:500]}" for r in results
        )
        or "(no research snippets)"
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
    try:
        text = await _groq_with_gemini_fallback(
            _META_SYSTEM,
            f"Topic: {research.topic}\n"
            f"Keywords: {', '.join(research.keywords)}\n"
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
    return research.topic[:300], fallback_desc


# ────────────────────────────── Stage 2 · Write ───────────────────────────────


async def _write_article(research: Research) -> dict[str, Any]:
    sources = "\n".join(f"- {s}" for s in research.snippets) or "(none)"
    user_prompt = (
        f"Trending topic: {research.topic}\n"
        f"Target keywords (weave all of them in naturally): "
        f"{', '.join(research.keywords)}\n"
        f"Today's research (ground the article in these, link 2-3 inline):\n"
        f"{sources}\n\n"
        "Write the article now — remember: banned words fail the brief, "
        "short paragraphs, bullet points, engaging H2/H3 headings, ~1000 words. "
        "Write in basic, simple, human-like language. Avoid robotic AI words "
        "(delve, moreover, testament). Avoid keyword stuffing. Naturally place "
        "the extracted keywords. 100% unique, no plagiarism."
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


async def fetch_relevant_image(keywords: str) -> str | None:
    """Fetch a relevant photo from Pixabay using article keywords.

    Pixabay auto-serves WebP to supporting browsers, so only the image URL
    is stored (no image data travels through the DB). Returns None on any
    failure (no key, no hits, network error) — the caller falls back to a
    Pollinations URL so the pipeline never dies on images.
    """
    if not PIXABAY_API_KEY:
        return None
    query = (keywords or "").strip()
    if not query:
        return None
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://pixabay.com/api/",
                params={
                    "key": PIXABAY_API_KEY,
                    "q": query,
                    "image_type": "photo",
                    "orientation": "horizontal",
                    "safesearch": "true",
                    "min_width": 1280,
                    "per_page": 1,
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
        hits = data.get("hits") if isinstance(data, dict) else None
        if hits:
            # largeImageURL — Pixabay auto-serves WebP to supporting browsers.
            url = str(hits[0].get("largeImageURL") or "").strip()
            if url:
                print(f"[AutoBlog] Pixabay image OK for {query!r}", flush=True)
                return url
        print(f"[AutoBlog] Pixabay no hits for {query!r} — using fallback.", flush=True)
    except Exception as exc:  # noqa: BLE001 - fallback keeps the run alive
        print(
            f"[AutoBlog] Pixabay fetch failed ({type(exc).__name__}: {exc}) "
            "— using fallback.",
            flush=True,
        )
    return None


def _pollinations_cover_url(title: str, keywords: list[str]) -> str:
    """Pollinations.ai FALLBACK cover (keyless) when Pixabay has no hit.

    The prompt lives inside the URL, so the image is generated lazily the
    first time the blog <img> is requested. Nothing is downloaded/uploaded.
    """
    keyword = keywords[0] if keywords else title.split(":")[0]
    prompt = f"{title[:140]}, {keyword}, cinematic, high quality, 8k resolution"
    encoded = urllib.parse.quote(prompt, safe="")
    return (
        f"https://image.pollinations.ai/prompt/{encoded}"
        "?width=1200&height=630&nologo=true"
    )


_H2_RE = re.compile(r"(<h2[^>]*>.*?</h2>)", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


def _pollinations_content_url(section_title: str, keyword: str) -> str:
    """Pollinations.ai FALLBACK in-article image when Pixabay has no hit."""
    prompt = f"{section_title[:120]}, {keyword}, cinematic, high quality"
    encoded = urllib.parse.quote(prompt, safe="")
    return (
        f"https://image.pollinations.ai/prompt/{encoded}"
        "?width=800&height=450&nologo=true"
    )


async def _cover_image_url(title: str, keywords: list[str]) -> str:
    """Pixabay first (up to 3 keywords in the query); Pollinations fallback."""
    query = " ".join(keywords[:3]) if keywords else title.split(":")[0]
    url = await fetch_relevant_image(query)
    return url or _pollinations_cover_url(title, keywords)


async def _content_image_url(section_title: str, keyword: str) -> str:
    """Pixabay first for the section keyword; Pollinations fallback."""
    url = await fetch_relevant_image(keyword) if keyword else None
    return url or _pollinations_content_url(section_title, keyword)


def _figure_html(url: str, caption: str) -> str:
    alt = html.escape(caption, quote=True)
    text = html.escape(caption)
    return (
        f'<figure><img src="{url}" alt="{alt}" loading="lazy" '
        f'width="800" height="450" />'
        f"<figcaption>{text}</figcaption></figure>"
    )


async def _images_chunk(
    content_html: str, keywords: list[str]
) -> tuple[str, list[dict[str, str]]]:
    """Step 5/6: Pixabay images ONE BY ONE (pause between each).

    One figure at the end of each of the first MAX_CONTENT_IMAGES <h2>
    sections; a single image after the first paragraph when the article
    has no <h2> at all. Every image falls back to a Pollinations URL when
    Pixabay returns nothing.
    """
    parts = _H2_RE.split(content_html)
    if len(parts) < 3:
        caption = _TAG_RE.sub(" ", parts[0])[:120].strip() or "Article illustration"
        keyword = keywords[0] if keywords else "technology"
        url = await _content_image_url(caption, keyword)
        await asyncio.sleep(IMAGE_PAUSE_SECONDS)
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
        if section < MAX_CONTENT_IMAGES and keywords:
            caption = _TAG_RE.sub(" ", h2).strip()[:120] or f"Section {section + 1}"
            url = await _content_image_url(
                caption, keywords[section % len(keywords)]
            )
            if images:
                await asyncio.sleep(IMAGE_PAUSE_SECONDS)
            print(
                f"[AutoBlog] content image {len(images) + 1}/{MAX_CONTENT_IMAGES} "
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

            # Step 1/6 — Research: Tavily search + Nemotron Ultra topic pick.
            results: list[dict[str, str]] = []
            print(
                "[AutoBlog Background] Step 1: Starting research (Tavily + "
                "NVIDIA Nemotron Ultra topic)...",
                flush=True,
            )
            try:
                results = await _research_chunk()
                topic = await _topic_chunk(results)
                research = Research(
                    topic=topic,
                    keywords=[],
                    snippets=[
                        f"{r['title']} ({r['url']}): {r['content'][:500]}"
                        for r in results
                        if r["content"] or r["title"]
                    ],
                )
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

            # Step 5/6 — Images (Pixabay → Pollinations fallback, one by one).
            print(
                "[AutoBlog Background] Step 5: Fetching images "
                "(Pixabay, fallback Pollinations)...",
                flush=True,
            )
            try:
                content_html, content_images = await _images_chunk(
                    article["content_html"], research.keywords
                )
                cover = await _cover_image_url(meta_title, research.keywords)
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
