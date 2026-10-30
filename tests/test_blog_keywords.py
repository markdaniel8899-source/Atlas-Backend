from __future__ import annotations

import unittest

from app.services.blog.keywords import (
    KeywordValidationError,
    validate_keyword_plan,
)
from app.services.blog.types import SearchIntent


def valid_plan(**overrides: object) -> dict[str, object]:
    plan: dict[str, object] = {
        "primary": "AI note-taking app",
        "secondary": [
            "AI study notes",
            "best note-taking app",
            "student note app",
        ],
        "long_tail": [
            "best AI note-taking app for students",
            "how to use AI for study notes",
            "AI note-taking app for exam preparation",
            "student note-taking app comparison",
        ],
        "lsi": [
            "retrieval practice",
            "active recall",
            "study workflow",
            "note organization",
            "spaced repetition",
        ],
        "search_intent": "informational",
        "title_candidates": [
            "The Best AI Note-Taking Apps for Students",
            "How AI Notes Improve Retrieval Practice",
        ],
    }
    plan.update(overrides)
    return plan


class KeywordValidationTests(unittest.TestCase):
    def test_valid_plan(self) -> None:
        plan = validate_keyword_plan(valid_plan())
        self.assertEqual(plan.primary, "AI note-taking app")
        self.assertEqual(len(plan.secondary), 3)
        self.assertEqual(len(plan.long_tail), 4)
        self.assertEqual(len(plan.lsi), 5)
        self.assertEqual(plan.search_intent, SearchIntent.INFORMATIONAL)

    def test_intent_is_normalized(self) -> None:
        plan = validate_keyword_plan(
            valid_plan(search_intent=" Commercial ")
        )
        self.assertEqual(plan.search_intent, SearchIntent.COMMERCIAL)

    def test_invalid_intent(self) -> None:
        with self.assertRaises(KeywordValidationError):
            validate_keyword_plan(valid_plan(search_intent="viral"))

    def test_primary_keyword_length(self) -> None:
        for value in ("note", "one two three four five six"):
            with self.subTest(value=value):
                with self.assertRaises(KeywordValidationError):
                    validate_keyword_plan(valid_plan(primary=value))

    def test_required_counts(self) -> None:
        cases = {
            "secondary": ["only one"],
            "long_tail": ["one", "two", "three"],
            "lsi": ["one", "two", "three", "four"],
            "title_candidates": ["Only one title"],
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                with self.assertRaises(KeywordValidationError):
                    validate_keyword_plan(valid_plan(**{field: value}))

    def test_duplicate_items_are_rejected(self) -> None:
        with self.assertRaises(KeywordValidationError):
            validate_keyword_plan(
                valid_plan(
                    secondary=[
                        "AI study notes",
                        "ai study notes",
                        "student note app",
                    ]
                )
            )

    def test_missing_primary(self) -> None:
        with self.assertRaises(KeywordValidationError):
            validate_keyword_plan(valid_plan(primary=""))

    def test_empty_list_item(self) -> None:
        with self.assertRaises(KeywordValidationError):
            validate_keyword_plan(
                valid_plan(
                    secondary=["AI study notes", "   ", "student note app"]
                )
            )

    def test_non_object_plan(self) -> None:
        with self.assertRaises(KeywordValidationError):
            validate_keyword_plan(["not", "an", "object"])


if __name__ == "__main__":
    unittest.main()
