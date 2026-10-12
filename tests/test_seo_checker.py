import unittest

from app.plugins.seo_checker import evaluate_seo_lightweight

KEYWORD = "ai learning"


def _content(*, words=350, heading="<h2>Why It Matters</h2>", lead=True) -> str:
    lead_html = (
        "<p>This guide explains ai learning in simple, practical terms "
        "for every student.</p>"
        if lead
        else "<p>Generic opening paragraph without the target phrase.</p>"
    )
    body = " ".join(f"word{i}" for i in range(words))
    return f"{heading}\n{lead_html}\n<p>{body}</p>"


def _meta(length=140) -> str:
    return ("Discover how ai learning transforms study habits. " * 8)[:length].ljust(
        length, "."
    )


class TestEvaluateSeoLightweight(unittest.TestCase):
    def test_all_rules_pass(self):
        report = evaluate_seo_lightweight(
            title="Best ai learning Tools for 2026",
            content=_content(),
            keyword=KEYWORD,
            meta_description=_meta(),
        )
        self.assertEqual(report["score"], 100)
        self.assertTrue(report["passed"])
        self.assertEqual(report["feedback"], "All SEO checks passed.")

    def test_single_rule_failure_still_passes_at_80(self):
        report = evaluate_seo_lightweight(
            title="Best ai learning Tools for 2026",
            content=_content(),
            keyword=KEYWORD,
            meta_description="too short",
        )
        self.assertEqual(report["score"], 80)
        self.assertTrue(report["passed"])
        self.assertIn("meta description", report["feedback"])

    def test_word_count_rule_is_strictly_over_300(self):
        # Control the total exactly: empty heading (0 words) + lead (8 words)
        # + body -> 300 words total fails, 301 passes.
        lead = "<p>ai learning makes study simpler for every student.</p>"

        def body(n: int) -> str:
            return " ".join(f"w{i}" for i in range(n))

        exact = evaluate_seo_lightweight(
            title="ai learning",
            content=f"<h2></h2>{lead}<p>{body(292)}</p>",
            keyword=KEYWORD,
            meta_description=_meta(),
        )
        self.assertEqual(exact["score"], 80)  # 300 words is NOT > 300
        self.assertTrue(exact["passed"])
        self.assertIn("words", exact["feedback"])

        one_more = evaluate_seo_lightweight(
            title="ai learning",
            content=f"<h2></h2>{lead}<p>{body(293)}</p>",
            keyword=KEYWORD,
            meta_description=_meta(),
        )
        self.assertEqual(one_more["score"], 100)
        self.assertTrue(one_more["passed"])

    def test_missing_heading_and_short_content_fails(self):
        report = evaluate_seo_lightweight(
            title="ai learning",
            content=_content(words=100, heading=""),
            keyword=KEYWORD,
            meta_description=_meta(),
        )
        self.assertEqual(report["score"], 60)
        self.assertFalse(report["passed"])
        self.assertTrue(report["feedback"].startswith("Failed:"))

    def test_h3_heading_satisfies_structure_rule(self):
        report = evaluate_seo_lightweight(
            title="ai learning",
            content=_content(heading='<h3 id="x">Deep dive</h3>'),
            keyword=KEYWORD,
            meta_description=_meta(),
        )
        self.assertEqual(report["score"], 100)

    def test_keyword_missing_from_title_and_opening(self):
        report = evaluate_seo_lightweight(
            title="A completely different headline",
            content=_content(lead=False),
            keyword=KEYWORD,
            meta_description=_meta(),
        )
        self.assertEqual(report["score"], 60)  # both keyword checks failed
        self.assertFalse(report["passed"])
        self.assertIn("keyword 'ai learning'", report["feedback"])

    def test_keyword_only_missing_from_title_keeps_80(self):
        report = evaluate_seo_lightweight(
            title="A completely different headline",
            content=_content(),
            keyword=KEYWORD,
            meta_description=_meta(),
        )
        self.assertEqual(report["score"], 80)
        self.assertTrue(report["passed"])
        self.assertIn("missing from the meta title", report["feedback"])

    def test_empty_keyword_counts_as_two_failed_checks(self):
        report = evaluate_seo_lightweight(
            title="Anything",
            content=_content(),
            keyword="",
            meta_description=_meta(),
        )
        self.assertEqual(report["score"], 60)
        self.assertFalse(report["passed"])
        self.assertIn("no target keyword", report["feedback"])

    def test_empty_inputs_do_not_crash(self):
        report = evaluate_seo_lightweight("", "", "", "")
        self.assertEqual(report["score"], 0)
        self.assertFalse(report["passed"])
        self.assertIsInstance(report["feedback"], str)
