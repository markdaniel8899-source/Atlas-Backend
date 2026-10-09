"""Auto AI Blog Writer — research → write → cover URL → save.

Runs on demand via POST /api/blog/generate and daily at 09:00 UTC through
APScheduler (wired up in app.main).

Stages:
  1. Research      — Tavily Search in the "AI & Technology" niche; Gemini
                     distills one trending topic + the top 3-5 SEO keywords
                     from the results. Tavily failure falls back to a
                     curated niche topic (never crashes the run).
  2. Write         — Gemini 1.5 Flash writes a ~1000-word, SEO-optimised
                     article as strict JSON (title / excerpt / content_html /
                     tags). The system prompt bans robotic AI phrasing and
                     enforces short paragraphs + bullet points + H2/H3.
  3. Cover image   — Pollinations.ai: free, keyless, no hotlinking limits.
                     The URL itself carries the prompt, so nothing is
                     downloaded or uploaded — the blog <img> renders it.
  3b. Content images — up to three more Pollinations images (800x450), one
                     appended to the end of the first three <h2> sections as
                     a <figure><img><figcaption>; the URLs are also stored in
                     blog_posts.content_images (jsonb).
  4. Save          — supabase-py insert into public.blog_posts with the
                     service-role key (RLS-bypassing writer).

Error handling: the whole pipeline is wrapped in try/except — any Tavily,
Gemini, or database failure is logged as [AutoBlog] and the run exits
gracefully (returns None) without ever crashing the server.

Async note: clients are created per run on purpose — the route awaits on
uvicorn's event loop while the scheduler job runs asyncio.run() on its own
thread loop; no client object may be shared across loops.
"""

from __future__ import annotations

import asyncio
import html
import re
import threading
import uuid
import urllib.parse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import google.generativeai as genai
from supabase import Client, create_client
from tavily import TavilyClient

from app import ai_service
from app.config import get_settings

MAX_RESULTS = 5  # "top 3-5 keywords" → cap at 5

# How many in-article images to embed (one per H2 section, first N sections).
MAX_CONTENT_IMAGES = 3

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

_OUTLINE_SYSTEM = """You are an SEO content strategist. From the research snippets, pick ONE specific, currently trending, searchable blog topic (not a generic pillar term) and extract the top 3 to 5 SEO keywords for it.

Return ONLY strict JSON (no fences, no commentary):
{"topic": "...", "keywords": ["...", "...", "..."]}

- topic: a concrete angle a reader would click, max 120 chars.
- keywords: 3 to 5, each 1-3 words, lowercase, no dupes."""


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


# ───────────────────────────── Gemini helpers ────────────────────────────────


def _gemini_model(system_instruction: str) -> genai.GenerativeModel:
    settings = get_settings()
    if not settings.google_api_key:
        raise BlogPipelineError("GOOGLE_API_KEY is not configured.", status_code=503)
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


async def _outline(results: list[dict[str, str]]) -> dict[str, Any]:
    """Gemini picks the day's topic + top 3-5 keywords from the Tavily results."""
    blob = "\n".join(
        f"- {r['title']} ({r['url']}): {r['content'][:500]}" for r in results
    )
    try:
        model = _gemini_model(_OUTLINE_SYSTEM)
        text = await _generate(
            model,
            f"Today's research:\n{blob}",
            temperature=0.3,
            max_tokens=400,
        )
        return ai_service.extract_json(text, prefer="keywords")
    except Exception as exc:  # noqa: BLE001 - any failure falls back to heuristics
        print(f"[AutoBlog] outline extraction failed: {type(exc).__name__}: {exc}")
        return {}


async def _research() -> Research:
    query = f"{NICHE} {_utc().year} trending"

    raw: list[dict[str, Any]] = []
    try:
        raw = await asyncio.to_thread(_tavily_search, query)
    except Exception as exc:  # noqa: BLE001 - fallback topic covers this
        print(f"[AutoBlog] Tavily research failed — fallback topic: {exc}")

    results = [r for r in (_clean_result(item) for item in raw) if r["title"] or r["content"]]

    if not results:
        # Tavily down → curated fallback topic for the day (keeps the run alive).
        topic = FALLBACK_TOPICS[_day_index() % len(FALLBACK_TOPICS)]
        keywords = _pad_keywords([], topic)
        print(f"[AutoBlog] research fallback -> topic={topic!r}")
        return Research(topic=topic, keywords=keywords, snippets=[])

    outline = await _outline(results)
    topic = str(outline.get("topic") or "").strip()
    if not topic:
        # Gemini failed → first result's headline is a decent specific topic.
        topic = results[0]["title"].split("|")[0].split(" - ")[0].strip()[:120]
    raw_keywords = outline.get("keywords")
    keywords = _pad_keywords(
        [str(k) for k in raw_keywords] if isinstance(raw_keywords, list) else [],
        topic,
    )
    snippets = [
        f"{r['title']} ({r['url']}): {r['content'][:500]}"
        for r in results
        if r["content"] or r["title"]
    ]
    return Research(topic=topic, keywords=keywords, snippets=snippets)


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
        "short paragraphs, bullet points, engaging H2/H3 headings, ~1000 words."
    )

    model = _gemini_model(_WRITER_SYSTEM)
    text = await _generate(model, user_prompt, temperature=0.8, max_tokens=4096)

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


# ───────────────────────── Stage 3 · Cover image (Pollinations) ──────────────


def _cover_image_url(title: str, keywords: list[str]) -> str:
    """Pollinations.ai — free, NO API key, no hotlinking restrictions.

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


def _content_image_url(section_title: str, keyword: str) -> str:
    """Pollinations URL for an in-article image (800x450, same lazy pattern)."""
    prompt = f"{section_title[:120]}, {keyword}, cinematic, high quality"
    encoded = urllib.parse.quote(prompt, safe="")
    return (
        f"https://image.pollinations.ai/prompt/{encoded}"
        "?width=800&height=450&nologo=true"
    )


def _figure_html(url: str, caption: str) -> str:
    alt = html.escape(caption, quote=True)
    text = html.escape(caption)
    return (
        f'<figure><img src="{url}" alt="{alt}" loading="lazy" '
        f'width="800" height="450" />'
        f"<figcaption>{text}</figcaption></figure>"
    )


def _embed_content_images(
    content_html: str, keywords: list[str]
) -> tuple[str, list[dict[str, str]]]:
    """Append one <figure> image to the end of the first few <h2> sections.

    Returns the updated HTML plus the metadata stored in
    blog_posts.content_images. Articles with no <h2> at all get a single
    image right after the first paragraph instead.
    """
    parts = _H2_RE.split(content_html)
    if len(parts) < 3:
        caption = _TAG_RE.sub(" ", parts[0])[:120].strip() or "Article illustration"
        keyword = keywords[0] if keywords else "technology"
        url = _content_image_url(caption, keyword)
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
            url = _content_image_url(caption, keywords[section % len(keywords)])
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


# ──────────────────────────────── Orchestrator ────────────────────────────────


async def run_pipeline() -> dict[str, Any] | None:
    """Run all four stages once.

    Returns the saved row plus research metadata, or None when any stage
    failed — the error is logged ([AutoBlog] …) and the run exits
    gracefully without ever raising out to the caller (except a 409 when a
    run is already in progress).
    """
    if not _RUN_LOCK.acquire(blocking=False):
        raise BlogPipelineError(
            "A blog generation run is already in progress.", status_code=409
        )
    try:
        try:
            started = _utc().isoformat(timespec="seconds")
            print(f"[AutoBlog] pipeline start {started}")

            research = await _research()
            print(f"[AutoBlog] topic={research.topic!r} keywords={research.keywords}")

            article = await _write_article(research)
            content_html, content_images = _embed_content_images(
                article["content_html"], research.keywords
            )
            print(f"[AutoBlog] embedded {len(content_images)} content image(s)")
            cover = _cover_image_url(article["title"], research.keywords)
            print(f"[AutoBlog] cover image: {cover[:110]}")

            payload = {
                "slug_base": slugify(article["title"]),
                "title": article["title"],
                "content": content_html,
                "excerpt": article["excerpt"],
                "cover_image_url": cover,
                "content_images": content_images,
                "tags": article["tags"],
                "keywords": research.keywords,
            }
            post = await asyncio.to_thread(_insert_post, payload)

            print(f"[AutoBlog] saved post id={post.get('id')} slug={post.get('slug')}")
            return {"post": post, "topic": research.topic, "keywords": research.keywords}
        except Exception as exc:  # noqa: BLE001 - log + graceful exit, never crash
            print(f"[AutoBlog] pipeline aborted: {type(exc).__name__}: {exc}")
            return None
    finally:
        _RUN_LOCK.release()


def run_scheduled_job() -> None:
    """APScheduler entry point (runs on a worker thread — owns its own loop)."""
    try:
        result = asyncio.run(run_pipeline())
        if result:
            print(f"[AutoBlog] scheduled run OK -> /blog/{result['post'].get('slug')}")
        else:
            print("[AutoBlog] scheduled run exited gracefully (see error above)")
    except Exception as exc:  # noqa: BLE001 - never kill the scheduler thread
        print(f"[AutoBlog] scheduled run failed: {type(exc).__name__}: {exc}")
