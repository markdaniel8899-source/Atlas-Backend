from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from app.services.blog.topics import (
    category_counts,
    find_duplicate,
    jaccard,
    load_topic_seeds,
    rank_topics,
    score_candidate,
    seed_candidates,
    select_topic,
    topic_slug,
    validate_topic_candidates,
)
from app.services.blog.types import (
    Category,
    TopicCandidate,
    TopicHistoryEntry,
    TopicSource,
)

NOW = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)


def candidate(
    topic: str = "AI note-taking for students",
    *,
    category: Category = Category.AI,
    keyword: str = "AI note-taking app",
    trend: float = 80,
    seo: float = 70,
    product: float = 60,
) -> TopicCandidate:
    return TopicCandidate(
        topic=topic,
        category=category,
        primary_keyword=keyword,
        trend_score=trend,
        seo_score=seo,
        product_fit=product,
    )


def history_entry(
    topic: str,
    *,
    category: Category = Category.AI,
    keyword: str | None = None,
    days_ago: int = 10,
) -> TopicHistoryEntry:
    return TopicHistoryEntry(
        topic=topic,
        category=category,
        primary_keyword=keyword,
        used_at=NOW - timedelta(days=days_ago),
    )


class TopicSeedTests(unittest.TestCase):
    def test_load_seeds(self) -> None:
        seeds = load_topic_seeds()
        self.assertEqual(len(seeds), 48)
        counts = {category: 0 for category in Category}
        for seed in seeds:
            counts[seed.category] += 1
        self.assertTrue(all(count == 8 for count in counts.values()))
        self.assertEqual(len({seed.id for seed in seeds}), 48)

    def test_seed_candidates(self) -> None:
        candidates = seed_candidates()
        self.assertEqual(len(candidates), 48)
        self.assertTrue(
            all(candidate.source is TopicSource.SEED for candidate in candidates)
        )
        self.assertTrue(
            all(2 <= len(item.primary_keyword.split()) <= 5 for item in candidates)
        )


class TopicScoringTests(unittest.TestCase):
    def test_weighted_score(self) -> None:
        scored = score_candidate(
            candidate(trend=100, seo=80, product=60),
            {category: 0 for category in Category},
        )
        self.assertEqual(scored.diversity_score, 100)
        self.assertEqual(scored.total_score, 85)

    def test_category_cap_avoids_overused_category(self) -> None:
        history = [
            history_entry(f"AI topic {index}", days_ago=index + 1)
            for index in range(7)
        ]
        ai = candidate(
            "AI study workflow",
            category=Category.AI,
            keyword="AI study workflow",
            trend=100,
            seo=100,
            product=100,
        )
        software = candidate(
            "Student dashboard software",
            category=Category.SOFTWARE,
            keyword="student dashboard software",
            trend=20,
            seo=20,
            product=20,
        )
        selected = select_topic([ai, software], history, now=NOW)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.category, Category.SOFTWARE)

    def test_banned_topic_is_rejected(self) -> None:
        item = candidate(
            "Political debate about AI regulation",
            keyword="AI regulation politics",
            trend=100,
            seo=100,
            product=100,
        )
        self.assertIsNone(select_topic([item], now=NOW))
        self.assertEqual(rank_topics([item], now=NOW), [])

    def test_empty_candidates(self) -> None:
        self.assertIsNone(select_topic([], now=NOW))


class TopicDeduplicationTests(unittest.TestCase):
    def test_exact_topic_within_90_days(self) -> None:
        item = candidate("AI note-taking for students")
        history = [history_entry("AI note-taking for students")]
        self.assertIn("exact topic", find_duplicate(item, history, now=NOW) or "")

    def test_near_duplicate_topic(self) -> None:
        item = candidate("AI note-taking apps for students")
        history = [
            history_entry("AI note-taking apps for busy students")
        ]
        self.assertGreaterEqual(
            jaccard(item.topic, history[0].topic), 0.8
        )
        self.assertIn(
            "near-duplicate", find_duplicate(item, history, now=NOW) or ""
        )

    def test_primary_keyword_within_60_days(self) -> None:
        item = candidate(
            "A different topic",
            category=Category.SOFTWARE,
            keyword="student productivity software",
        )
        history = [
            history_entry(
                "Earlier productivity article",
                category=Category.SOFTWARE,
                keyword="student productivity software",
            )
        ]
        self.assertIn(
            "keyword", find_duplicate(item, history, now=NOW) or ""
        )

    def test_old_history_is_allowed(self) -> None:
        item = candidate("AI note-taking for students")
        history = [
            history_entry(
                "AI note-taking for students",
                keyword="AI note-taking app",
                days_ago=100,
            )
        ]
        self.assertIsNone(find_duplicate(item, history, now=NOW))

    def test_duplicate_candidates_are_collapsed(self) -> None:
        first = candidate("AI study planning")
        second = candidate("AI study planning")
        ranked = rank_topics([first, second], now=NOW)
        self.assertEqual(len(ranked), 1)


class TopicValidationTests(unittest.TestCase):
    def test_valid_candidates(self) -> None:
        parsed = validate_topic_candidates(
            {
                "candidates": [
                    {
                        "topic": "AI note-taking for students",
                        "category": "ai",
                        "primary_keyword": "AI note-taking app",
                    },
                    {
                        "topic": "Student laptop buying guide",
                        "category": "hardware",
                        "primary_keyword": "student laptop guide",
                    },
                ]
            }
        )
        self.assertEqual(len(parsed), 2)
        self.assertTrue(
            all(item.source is TopicSource.LLM for item in parsed)
        )

    def test_duplicate_topic_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_topic_candidates(
                {
                    "candidates": [
                        {
                            "topic": "AI note-taking for students",
                            "category": "ai",
                            "primary_keyword": "AI note-taking app",
                        },
                        {
                            "topic": "AI note-taking for students",
                            "category": "software",
                            "primary_keyword": "AI notes software",
                        },
                    ]
                }
            )

    def test_primary_keyword_length_is_enforced(self) -> None:
        with self.assertRaises(ValueError):
            validate_topic_candidates(
                {
                    "candidates": [
                        {
                            "topic": "A long topic",
                            "category": "ai",
                            "primary_keyword": "one two three four five six",
                        }
                    ]
                }
            )

    def test_missing_candidates_raise(self) -> None:
        with self.assertRaises(ValueError):
            validate_topic_candidates({"candidates": []})


if __name__ == "__main__":
    unittest.main()
