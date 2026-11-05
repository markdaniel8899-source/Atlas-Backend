from __future__ import annotations

import unittest
from datetime import datetime, timezone

from app.services.blog.types import (
    ContentVisualPlan,
    CoverPlan,
    ResearchBundle,
    ResearchSource,
    VisualDataset,
    VisualPlan,
    VisualType,
)
from app.services.blog.visuals import (
    _expand_visual_coverage,
    _is_time_series,
    validate_visual_plan,
    VisualPlanError,
)

SOURCE_URL = "https://example.com/report"


def research() -> ResearchBundle:
    return ResearchBundle(
        topic="Low code platforms for students",
        primary_keyword="low code platforms for students",
        accessed_at=datetime(2026, 11, 5, tzinfo=timezone.utc),
        sources=[
            ResearchSource(
                title="Example report",
                url=SOURCE_URL,
                domain="example.com",
            )
        ],
    )


def long_content(sections: int = 6, words_per_section: int = 160) -> str:
    parts = ["<p>Intro paragraph with a few words for the article.</p>"]
    for index in range(sections):
        body = " ".join(["study"] * words_per_section)
        parts.append(f"<h2>Section {index + 1}</h2><p>{body}</p>")
    parts.append("<h2>Sources</h2><p>Source list.</p>")
    return "".join(parts)


def chart_plan(heading: str) -> ContentVisualPlan:
    return ContentVisualPlan(
        after_heading=heading,
        after_paragraph=1,
        type=VisualType.CHART,
        purpose="comparison",
        alt="Comparison chart",
        caption="Comparison chart",
        data=VisualDataset(
            title="Adoption",
            chart_type="bar",
            labels=["2023", "2024", "2025"],
            values=[10, 20, 30],
            unit="%",
            source_url=SOURCE_URL,
            source_label="Example",
        ),
    )


class VisualCoverageTests(unittest.TestCase):
    def test_expands_text_only_tail_with_photo(self) -> None:
        content = long_content()
        plan = VisualPlan(
            cover=CoverPlan(
                query="student desk",
                alt="Cover",
                caption="Cover",
            ),
            visuals=[chart_plan("Section 1")],
        )
        expanded = _expand_visual_coverage(plan, content, research())
        self.assertGreater(len(expanded.visuals), len(plan.visuals))
        photos = [
            item for item in expanded.visuals if item.type is VisualType.PHOTO
        ]
        self.assertTrue(photos)
        self.assertTrue(photos[0].query)

    def test_short_article_is_left_alone(self) -> None:
        content = (
            "<h2>Section 1</h2><p>Short body text for a short article.</p>"
        )
        plan = VisualPlan(
            cover=CoverPlan(
                query="student desk",
                alt="Cover",
                caption="Cover",
            ),
            visuals=[chart_plan("Section 1")],
        )
        expanded = _expand_visual_coverage(plan, content, research())
        self.assertEqual(len(expanded.visuals), len(plan.visuals))

    def test_time_series_uses_line_chart(self) -> None:
        self.assertTrue(_is_time_series(["2022", "2023", "2024"]))
        self.assertTrue(_is_time_series(["Q1", "Q2", "Q3"]))
        self.assertTrue(_is_time_series(["Jan", "Feb", "Mar"]))
        self.assertFalse(_is_time_series(["Laptop", "Phone", "Tablet"]))
        content = long_content()
        plan = VisualPlan(
            cover=CoverPlan(
                query="student desk",
                alt="Cover",
                caption="Cover",
            ),
            visuals=[chart_plan("Section 1")],
        )
        validate_visual_plan(
            plan, content, research(), required_count=len(plan.visuals)
        )
        self.assertEqual(plan.visuals[0].data.chart_type, "line")

    def test_same_type_visuals_allowed_when_far_apart(self) -> None:
        content = long_content()
        plan = VisualPlan(
            cover=CoverPlan(
                query="student desk",
                alt="Cover",
                caption="Cover",
            ),
            visuals=[chart_plan("Section 1"), chart_plan("Section 5")],
        )
        validate_visual_plan(
            plan, content, research(), required_count=len(plan.visuals)
        )

    def test_same_type_visuals_rejected_when_close(self) -> None:
        content = long_content(sections=2, words_per_section=40)
        plan = VisualPlan(
            cover=CoverPlan(
                query="student desk",
                alt="Cover",
                caption="Cover",
            ),
            visuals=[
                ContentVisualPlan(
                    after_heading="Section 1",
                    after_paragraph=1,
                    type=VisualType.PHOTO,
                    purpose="illustration",
                    alt="Student studying",
                    caption="Student studying",
                    query="student studying",
                ),
                ContentVisualPlan(
                    after_heading="Section 2",
                    after_paragraph=1,
                    type=VisualType.PHOTO,
                    purpose="illustration",
                    alt="Student working",
                    caption="Student working",
                    query="coding screen",
                ),
            ],
        )
        with self.assertRaises(VisualPlanError):
            validate_visual_plan(
                plan, content, research(), required_count=len(plan.visuals)
            )


if __name__ == "__main__":
    unittest.main()
