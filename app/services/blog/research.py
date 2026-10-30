from __future__ import annotations

import asyncio
import html
import logging
import re
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from tavily import AsyncTavilyClient

from app.config import get_settings

from .types import NumericFact, ResearchBundle, ResearchSource, SourceTier

logger = logging.getLogger(__name__)


class ResearchError(RuntimeError):
    pass


TIER_1_HOSTS = {
    "arxiv.org",
    "ieee.org",
    "nature.com",
    "science.org",
    "pubmed.ncbi.nlm.nih.gov",
    "who.int",
    "oecd.org",
    "nber.org",
}

TIER_2_HOSTS = {
    "bbc.com",
    "bbc.co.uk",
    "reuters.com",
    "apnews.com",
    "economist.com",
    "ft.com",
    "wsj.com",
    "nytimes.com",
    "washingtonpost.com",
    "hbr.org",
    "technologyreview.com",
    "wired.com",
    "arstechnica.com",
    "theverge.com",
    "techcrunch.com",
    "docs.python.org",
    "developer.mozilla.org",
    "docs.github.com",
}

BLACKLISTED_HOSTS = {
    "pinterest.com",
    "quora.com",
    "answers.com",
    "ehow.com",
    "slideshare.net",
    "coursehero.com",
}

TRACKING_PARAMS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
}

PERCENT_RE = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s?%"
)
CURRENCY_RE = re.compile(
    r"(?<![\w.])([$€£])\s?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)"
)
UNIT_RE = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s?"
    r"(million|billion|thousand|hours?|minutes?|days?|weeks?|"
    r"months?|years?|GB|TB|MB|ms)\b",
    flags=re.IGNORECASE,
)
BARE_NUMBER_RE = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})*(?:\.\d+)?)(?![\w.])"
)
STAT_HINT_RE = re.compile(
    r"\b(study|survey|participants|score|scores|increase|decrease|"
    r"found|reported|rate|percent|percentage|average|median)\b",
    flags=re.IGNORECASE,
)
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")


def canonicalize_url(url: str) -> str | None:
    raw = (url or "").strip()
    if not raw:
        return None
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError:
        return None
    if parsed.scheme.lower() not in ("http", "https"):
        return None
    host = (parsed.hostname or "").lower().strip(".")
    if not host:
        return None
    if host.startswith("www."):
        host = host[4:]
    netloc = host if port in (None, 80, 443) else f"{host}:{port}"
    query = urlencode(
        [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.lower().startswith("utm_")
            and key.lower() not in TRACKING_PARAMS
        ]
    )
    path = parsed.path.rstrip("/") or ""
    return urlunsplit(("https", netloc, path, query, ""))


def domain_from_url(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower().strip(".")
    return host[4:] if host.startswith("www.") else host


def source_tier(domain: str) -> SourceTier:
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return SourceTier.TIER_1
    if _host_matches(domain, TIER_1_HOSTS):
        return SourceTier.TIER_1
    if _host_matches(domain, TIER_2_HOSTS):
        return SourceTier.TIER_2
    return SourceTier.TIER_3


def _host_matches(domain: str, hosts: set[str]) -> bool:
    return any(domain == host or domain.endswith(f".{host}") for host in hosts)


def _is_blacklisted(domain: str) -> bool:
    return _host_matches(domain, BLACKLISTED_HOSTS)


def _strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


class ResearchService:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: Any | None = None,
        now: datetime | None = None,
        max_results: int = 8,
        max_per_domain: int = 2,
        max_sources: int = 12,
        retry_delay: float = 0.5,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.api_key = (
            api_key
            if api_key is not None
            else get_settings().tavily_api_key
        )
        self._client = client
        self._now = now
        self.max_results = max(1, max_results)
        self.max_per_domain = max(1, max_per_domain)
        self.max_sources = max(3, max_sources)
        self.retry_delay = max(0.0, retry_delay)
        self._sleep = sleep or asyncio.sleep

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise ResearchError(
                "TAVILY_API_KEY is not configured on the server."
            )
        self._client = AsyncTavilyClient(api_key=self.api_key)
        return self._client

    def build_queries(
        self, topic: str, primary_keyword: str | None = None
    ) -> list[str]:
        now = self._now or datetime.now(timezone.utc)
        keyword = (primary_keyword or topic).strip()
        return [
            f'"{topic.strip()}" {now.year}',
            f'"{keyword}" statistics OR report {now.year}',
            f'"{topic.strip()}" study OR expert analysis',
        ]

    async def research(
        self,
        topic: str,
        primary_keyword: str | None = None,
        *,
        queries: Sequence[str] | None = None,
    ) -> ResearchBundle:
        clean_topic = (topic or "").strip()
        if not clean_topic:
            raise ResearchError("Research topic cannot be empty.")
        keyword = (primary_keyword or clean_topic).strip()
        query_list = list(queries or self.build_queries(clean_topic, keyword))
        self._get_client()
        raw_results: list[dict[str, Any]] = []

        for query in query_list:
            payload = await self._search(query)
            raw_results.extend(payload)

        if not raw_results:
            raise ResearchError("Tavily returned no usable results.")

        sources = self._collect_sources(raw_results)
        if len(sources) < 3:
            raise ResearchError(
                f"Only {len(sources)} usable sources found; need at least 3."
            )
        if not any(source.tier <= SourceTier.TIER_2 for source in sources):
            raise ResearchError(
                "Research needs at least one authoritative tier 1/2 source."
            )

        facts: list[NumericFact] = []
        for source in sources:
            facts.extend(_extract_numeric_facts(source))
        facts = _dedupe_facts(facts)[:40]

        return ResearchBundle(
            topic=clean_topic,
            primary_keyword=keyword,
            queries=query_list,
            sources=sources,
            numeric_facts=facts,
            accessed_at=self._now or datetime.now(timezone.utc),
        )

    async def _search(self, query: str) -> list[dict[str, Any]]:
        client = self._get_client()
        last_error: Exception | None = None
        for attempt in range(1, 3):
            try:
                payload = await client.search(
                    query,
                    search_depth="advanced",
                    max_results=self.max_results,
                    include_answer=False,
                    include_raw_content=False,
                    topic="general",
                    timeout=30,
                )
                results = payload.get("results", []) if payload else []
                return [item for item in results if isinstance(item, dict)]
            except Exception as exc:
                last_error = exc
                if attempt == 1 and self.retry_delay:
                    await self._sleep(self.retry_delay)
        logger.warning(
            "Tavily query failed after retry: %s (%s)",
            query,
            type(last_error).__name__,
        )
        return []

    def _collect_sources(
        self, raw_results: Sequence[dict[str, Any]]
    ) -> list[ResearchSource]:
        seen_urls: set[str] = set()
        domain_counts: dict[str, int] = {}
        sources: list[ResearchSource] = []

        for item in raw_results:
            url = canonicalize_url(str(item.get("url") or ""))
            if not url or url in seen_urls:
                continue
            domain = domain_from_url(url)
            if not domain or _is_blacklisted(domain):
                continue
            if domain_counts.get(domain, 0) >= self.max_per_domain:
                continue
            snippet = _strip_html(
                str(item.get("content") or item.get("raw_content") or "")
            )
            if not snippet:
                continue
            relevance = item.get("score")
            sources.append(
                ResearchSource(
                    title=(
                        str(item.get("title") or domain).strip()[:300]
                    ),
                    url=url,
                    domain=domain,
                    snippet=snippet[:1_500],
                    tier=source_tier(domain),
                    relevance=(
                        float(relevance)
                        if isinstance(relevance, (int, float))
                        else None
                    ),
                    published_at=(
                        str(
                            item.get("published_date")
                            or item.get("published_at")
                        )
                        if item.get("published_date")
                        or item.get("published_at")
                        else None
                    ),
                )
            )
            seen_urls.add(url)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        sources.sort(
            key=lambda source: (
                int(source.tier),
                -(source.relevance or 0.0),
            )
        )
        return sources[: self.max_sources]


def _extract_numeric_facts(source: ResearchSource) -> list[NumericFact]:
    facts: list[NumericFact] = []
    seen: set[tuple[float, str, str]] = set()

    def add(value: float, unit: str, sentence: str) -> None:
        context = sentence.strip()[:300]
        key = (round(value, 6), unit, context)
        if not context or key in seen:
            return
        seen.add(key)
        facts.append(
            NumericFact(
                value=value,
                unit=unit,
                context=context,
                source_url=source.url,
            )
        )

    for sentence in SENTENCE_RE.findall(source.snippet):
        covered: list[tuple[int, int]] = []
        for match in PERCENT_RE.finditer(sentence):
            add(_number(match.group(1)), "%", sentence)
            covered.append(match.span())
        for match in CURRENCY_RE.finditer(sentence):
            add(_number(match.group(2)), match.group(1), sentence)
            covered.append(match.span())
        for match in UNIT_RE.finditer(sentence):
            add(_number(match.group(1)), match.group(2).lower(), sentence)
            covered.append(match.span())

        if STAT_HINT_RE.search(sentence):
            for match in BARE_NUMBER_RE.finditer(sentence):
                if any(
                    match.start() < end and match.end() > start
                    for start, end in covered
                ):
                    continue
                value = _number(match.group(1))
                if 1900 <= value <= 2100 and float(value).is_integer():
                    continue
                add(value, "", sentence)

    return facts


def _number(raw: str) -> float:
    return float(raw.replace(",", ""))


def _dedupe_facts(facts: Sequence[NumericFact]) -> list[NumericFact]:
    result: list[NumericFact] = []
    seen: set[tuple[float, str, str, str]] = set()
    for fact in facts:
        key = (
            round(fact.value, 6),
            fact.unit,
            fact.context,
            fact.source_url,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(fact)
    return result
