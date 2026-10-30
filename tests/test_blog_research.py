from __future__ import annotations

import unittest
from datetime import datetime, timezone

from app.services.blog.research import (
    ResearchError,
    ResearchService,
    canonicalize_url,
    source_tier,
)
from app.services.blog.types import SourceTier


def result(
    url: str,
    content: str,
    *,
    title: str = "Source",
    score: float = 0.9,
) -> dict[str, object]:
    return {
        "url": url,
        "title": title,
        "content": content,
        "score": score,
    }


class FakeTavily:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def search(
        self, query: str, **kwargs: object
    ) -> dict[str, object]:
        self.calls.append((query, kwargs))
        if not self.responses:
            raise AssertionError("Unexpected Tavily request.")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return {"results": item}


async def no_sleep(_: float) -> None:
    return None


def build_service(
    results: list[object] | None = None,
) -> tuple[ResearchService, FakeTavily]:
    fake = FakeTavily(list(results or []))
    service = ResearchService(
        api_key="test-key",
        client=fake,
        now=datetime(2026, 10, 28, 12, tzinfo=timezone.utc),
        retry_delay=0,
        sleep=no_sleep,
    )
    return service, fake


class ResearchHelperTests(unittest.TestCase):
    def test_canonicalize_url(self) -> None:
        self.assertEqual(
            canonicalize_url(
                "https://www.Example.com/a/b/?utm_source=x&utm_medium=y"
                "&b=2#section"
            ),
            "https://example.com/a/b?b=2",
        )
        self.assertIsNone(canonicalize_url("https://example.com:bad/path"))
        self.assertIsNone(canonicalize_url("ftp://example.com/file"))
        self.assertIsNone(canonicalize_url(""))

    def test_source_tiers(self) -> None:
        self.assertEqual(source_tier("nist.gov"), SourceTier.TIER_1)
        self.assertEqual(source_tier("mit.edu"), SourceTier.TIER_1)
        self.assertEqual(source_tier("reuters.com"), SourceTier.TIER_2)
        self.assertEqual(source_tier("example.com"), SourceTier.TIER_3)


class ResearchServiceTests(unittest.IsolatedAsyncioTestCase):
    def sources(self) -> list[dict[str, object]]:
        return [
            result(
                "https://www.nist.gov/report?utm_source=newsletter",
                "A 2026 study found 72% of students improved. "
                "The pilot cost $1,200 and saved 3 hours per week.",
                title="NIST report",
                score=0.95,
            ),
            result(
                "https://nist.gov/report",
                "Duplicate URL.",
                title="Duplicate",
            ),
            result(
                "https://example.edu/a",
                "A survey reported a 40% improvement.",
                title="University A",
                score=0.8,
            ),
            result(
                "https://example.edu/b",
                "Another study found 55% retention.",
                title="University B",
                score=0.7,
            ),
            result(
                "https://example.edu/c",
                "Third result from the same university.",
                title="University C",
                score=0.6,
            ),
            result(
                "https://medium.com/article",
                "A general article about studying.",
                title="Medium article",
                score=0.5,
            ),
            result(
                "https://www.quora.com/question",
                "Blacklisted result.",
                title="Quora",
            ),
        ]

    async def test_research_collects_filtered_sources(self) -> None:
        service, fake = build_service([self.sources()] * 3)
        bundle = await service.research(
            "AI note-taking", "AI note-taking app"
        )

        self.assertEqual(len(bundle.sources), 4)
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(
            sum(source.domain == "example.edu" for source in bundle.sources),
            2,
        )
        self.assertTrue(
            all(source.domain != "quora.com" for source in bundle.sources)
        )
        self.assertIn("advanced", fake.calls[0][1]["search_depth"])

        percent = [
            fact
            for fact in bundle.numeric_facts
            if fact.value == 72 and fact.unit == "%"
        ]
        self.assertEqual(len(percent), 1)
        self.assertTrue(
            any(
                fact.value == 1200 and fact.unit == "$"
                for fact in bundle.numeric_facts
            )
        )
        self.assertTrue(
            any(
                fact.value == 3 and fact.unit == "hours"
                for fact in bundle.numeric_facts
            )
        )

    async def test_requires_three_sources(self) -> None:
        service, _ = build_service(
            [
                [
                    result("https://example.edu/a", "One.", score=0.9),
                    result("https://reuters.com/b", "Two.", score=0.8),
                ]
            ]
            * 3
        )
        with self.assertRaises(ResearchError):
            await service.research("Topic")

    async def test_requires_authoritative_source(self) -> None:
        service, _ = build_service(
            [
                [
                    result("https://medium.com/a", "One.", score=0.9),
                    result("https://dev.to/b", "Two.", score=0.8),
                    result("https://example.com/c", "Three.", score=0.7),
                ]
            ]
            * 3
        )
        with self.assertRaises(ResearchError) as raised:
            await service.research("Topic")
        self.assertIn("authoritative", str(raised.exception))

    async def test_query_retry(self) -> None:
        service, fake = build_service(
            [
                RuntimeError("temporary failure"),
                [
                    result("https://nist.gov/a", "One.", score=0.9),
                    result("https://reuters.com/b", "Two.", score=0.8),
                    result("https://example.edu/c", "Three.", score=0.7),
                ],
                [],
                [],
            ]
        )
        bundle = await service.research("Topic")
        self.assertEqual(len(bundle.sources), 3)
        self.assertEqual(len(fake.calls), 4)

    async def test_missing_api_key_fails_closed(self) -> None:
        service = ResearchService(api_key="", client=None)
        with self.assertRaises(ResearchError) as raised:
            await service.research("Topic")
        self.assertIn("TAVILY_API_KEY", str(raised.exception))

    async def test_empty_topic_fails(self) -> None:
        service, _ = build_service()
        with self.assertRaises(ResearchError):
            await service.research("  ")


if __name__ == "__main__":
    unittest.main()
