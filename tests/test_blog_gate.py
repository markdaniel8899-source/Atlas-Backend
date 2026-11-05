from __future__ import annotations

import unittest
from datetime import datetime, timezone

from app.services.blog.gate import run_gate
from app.services.blog.types import (
    KeywordPlan,
    ResearchBundle,
    VisualProduction,
    WriterOutput,
)


class GateSeverityTests(unittest.TestCase):
    def test_soft_checks_become_warnings(self) -> None:
        content = (
            "<p>AI note-taking app helps students. It is 50% faster "
            "\u2014 and better.</p>"
            "<h2>AI note-taking app guide</h2>"
            "<p>Use the app for notes and review.</p>"
        )
        output = WriterOutput(
            title="AI note-taking app guide",
            slug="ai-note-taking-app-guide",
            excerpt="A" * 120,
            content=content,
        )
        keywords = KeywordPlan(primary="AI note-taking app")
        research = ResearchBundle(
            topic="AI note-taking app",
            primary_keyword="AI note-taking app",
            accessed_at=datetime(2026, 11, 5, tzinfo=timezone.utc),
        )
        production = VisualProduction(
            content=content,
            cover_image_url="https://example.com/cover.png",
            cover_alt="Cover",
            cover_caption="Cover",
            cover_width=1200,
            cover_height=630,
        )
        report = run_gate(
            output,
            keywords,
            research,
            production,
            min_score=80,
        )
        hard_ids = {failure.id for failure in report.hard_failures}
        warning_ids = {warning.id for warning in report.warnings}
        self.assertNotIn("H24", hard_ids)
        self.assertNotIn("H15", hard_ids)
        self.assertIn("H24", warning_ids)
        self.assertIn("H15", warning_ids)


if __name__ == "__main__":
    unittest.main()
