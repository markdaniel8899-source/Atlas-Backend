"""Tests for blog_content_gate evaluation."""

from __future__ import annotations

import unittest

from app.services.blog_content_gate import (
    evaluate_article,
    _plain_text,
    _word_count,
    _first_n_words,
    _extract_headings,
    _has_heading_echo,
    _fix_heading_echo,
    _check_banned_words,
    _fix_dashes,
    _keyword_word_boundary,
    _count_faq_questions,
    _normalize_text,
    GateResult,
)


class TestPlainText(unittest.TestCase):
    def test_strips_tags(self):
        self.assertEqual(_plain_text("<p>Hello <strong>world</strong></p>"), "Hello world")

    def test_collapses_whitespace(self):
        self.assertEqual(_plain_text("Hello   world"), "Hello world")


class TestWordCount(unittest.TestCase):
    def test_counts_words(self):
        self.assertEqual(_word_count("<p>Hello world</p>"), 2)
        self.assertEqual(_word_count("<p>One two three</p>"), 3)


class TestFirstNWords(unittest.TestCase):
    def test_returns_first_n(self):
        text = "one two three four five"
        self.assertEqual(_first_n_words(text, 3), "one two three")


class TestExtractHeadings(unittest.TestCase):
    def test_extracts_h2_and_h3(self):
        html = "<h2>Title</h2><p>Body</p><h3>Sub</h3><p>More</p>"
        headings = _extract_headings(html)
        self.assertEqual(len(headings), 2)
        self.assertEqual(headings[0], ("h2", "Title"))
        self.assertEqual(headings[1], ("h3", "Sub"))


class TestNormalizeText(unittest.TestCase):
    def test_normalizes(self):
        self.assertEqual(_normalize_text("Hello, World!"), "hello world")
        self.assertEqual(_normalize_text("AI's Best"), "ai s best")


class TestHeadingEcho(unittest.TestCase):
    def test_detects_echo(self):
        html = "<h2>Best Tools</h2><p>Best Tools are essential for work.</p>"
        echoes = _has_heading_echo(html)
        self.assertEqual(echoes, ["Best Tools"])

    def test_no_echo(self):
        html = "<h2>Best Tools</h2><p>These tools are essential for work.</p>"
        echoes = _has_heading_echo(html)
        self.assertEqual(echoes, [])


class TestFixHeadingEcho(unittest.TestCase):
    def test_removes_echo(self):
        html = "<h2>Best Tools</h2><p>Best Tools help you work better.</p>"
        fixed, fixed_list = _fix_heading_echo(html)
        self.assertIn("Best Tools", fixed_list)
        self.assertNotIn("Best Tools help you work better", fixed)
        self.assertIn("help you work better", fixed)

    def test_removes_empty_paragraph(self):
        html = "<h2>Best Tools</h2><p>Best Tools</p>"
        fixed, fixed_list = _fix_heading_echo(html)
        self.assertIn("Best Tools", fixed_list)
        self.assertNotIn("<p></p>", fixed)
        self.assertNotIn("<p>Best Tools</p>", fixed)


class TestBannedWords(unittest.TestCase):
    def test_finds_banned(self):
        text = "In today's fast-paced world, we must delve into the details."
        banned = _check_banned_words(text)
        self.assertIn("in today's fast-paced world", banned)
        self.assertIn("delve into", banned)

    def test_no_false_positive(self):
        text = "Today's world is fast paced."  # not exact phrase
        banned = _check_banned_words(text)
        self.assertNotIn("in today's fast-paced world", banned)


class TestFixDashes(unittest.TestCase):
    def test_fixes_em_dash(self):
        self.assertEqual(_fix_dashes("Hello—world"), "Hello, world")

    def test_fixes_en_dash(self):
        self.assertEqual(_fix_dashes("Hello–world"), "Hello, world")


class TestKeywordWordBoundary(unittest.TestCase):
    def test_matches_whole_word(self):
        self.assertTrue(_keyword_word_boundary("ai", "This is about ai and ml"))
        self.assertTrue(_keyword_word_boundary("machine learning", "About machine learning today"))

    def test_no_partial_match(self):
        self.assertFalse(_keyword_word_boundary("ai", "This is said plainly"))
        self.assertFalse(_keyword_word_boundary("ai", "Email me"))


class TestCountFaqQuestions(unittest.TestCase):
    def test_counts_faq(self):
        html = "<h2>FAQ</h2><h3>Q1?</h3><p>A1</p><h3>Q2?</h3><p>A2</p><h3>Q3?</h3><p>A3</p>"
        self.assertEqual(_count_faq_questions(html), 3)

    def test_ignores_non_faq_h3(self):
        html = "<h2>Section</h2><h3>Sub</h3><p>Text</p><h2>FAQ</h2><h3>Q?</h3><p>A</p>"
        self.assertEqual(_count_faq_questions(html), 1)


class TestEvaluateArticle(unittest.TestCase):
    def _long_content(self):
        """Generate content with ~1100 words (within 900-1200 range)."""
        base = (
            "<p>Direct answer to the question with the test keyword included naturally in the opening paragraph. "
            "This article provides comprehensive coverage of the test topic with detailed explanations and examples. "
            "The test keyword appears multiple times throughout the content to satisfy the keyword density requirements. "
            "We will explore various aspects of the topic in detail, providing examples, definitions, and comparisons "
            "as needed to reach the target word count of 900 to 1200 words.</p>"
        )
        # Add multiple sections to build word count - aim for ~1000-1100 words
        sections = []
        for i in range(6):
            sections.append(
                f"<h2>Section {i+1}: Test Keyword Analysis</h2>"
                f"<p>This section discusses the test keyword in depth with multiple examples and detailed explanations. "
                f"The test keyword appears naturally in context. We provide specific examples, step-by-step processes, "
                f"and comparisons to illustrate the key points. The content is designed to be informative and comprehensive "
                f"while maintaining natural keyword usage throughout the discussion. "
                f"Additional details about the test keyword help expand the explanation naturally. "
                f"More content about the test keyword ensures we meet the minimum word count requirement naturally. "
                f"Further discussion of the test keyword continues to build the article's depth and breadth. "
                f"Even more content about the test keyword adds value and substance to the article.</p>"
                f"<ul><li>First point about test keyword with detailed explanation</li>"
                f"<li>Second point with examples and practical applications</li>"
                f"<li>Third point with comparison to alternative approaches</li></ul>"
            )
        return base + "".join(sections) + (
            "<h2>FAQ</h2>"
            "<h3>What is the test keyword?</h3><p>Direct clear answer about the test keyword with forty to sixty words of explanation here that explains the concept thoroughly.</p>"
            "<h3>How does the test keyword work?</h3><p>Direct clear answer explaining how the test keyword works in practice with sufficient detail to be helpful and informative.</p>"
            "<h3>Why is the test keyword important?</h3><p>Direct clear answer explaining the importance of the test keyword with relevant context and practical examples.</p>"
            "<h3>When should you use the test keyword?</h3><p>Direct clear answer about when to use the test keyword with practical guidance and specific scenarios.</p>"
        )

    def make_article(self, content=None, title="Test Title with test keyword", excerpt="Meta description with test keyword included and it is long enough to pass the check"):
        content = content or self._long_content()
        return {
            "title": title,
            "excerpt": excerpt,
            "content_html": content,
            "tags": ["test"],
            "visuals": [],
            "_json_meta": {"repaired": False},
        }

    def make_research(self, topic="Test Topic", keywords=None):
        from app.services.auto_blog import Research
        return Research(
            topic=topic,
            keywords=keywords or ["test keyword"],
            snippets=["Test snippet with 100 words and numbers 100%"],
            source_urls=["https://example.com"],
        )

    def make_profile(self, article_type="explainer"):
        base = {
            "article_type": article_type,
            "cover_photo": True,
            "max_content_photos": 1,
            "required_visuals": (),
            "max_visuals": 1,
            "downgraded": False,
        }
        return base

    def test_passes_minimal_valid(self):
        article = self.make_article()
        research = self.make_research()
        profile = self.make_profile("explainer")

        result = evaluate_article(article, "Test Title with test keyword", "Meta description with test keyword here and it is long enough to pass the check and meet the minimum length requirement of 120 chars", research, profile, False)
        self.assertTrue(result.passed, f"Failed: {result.reasons}")

    def test_fails_word_count_too_short(self):
        content = "<p>Short.</p><h2>FAQ</h2><h3>Q?</h3><p>A</p><h3>Q2?</h3><p>A2</p><h3>Q3?</h3><p>A3</p>"
        article = self.make_article(content)
        research = self.make_research(keywords=["test"])
        profile = self.make_profile("explainer")

        result = evaluate_article(article, "Test Title test", "Meta description with test here and it is long enough to pass the check", research, profile, False)
        self.assertFalse(result.passed)
        self.assertTrue(any("words" in r for r in result.reasons))

    def test_fails_word_count_too_long(self):
        content = "<p>" + "word " * 1300 + "</p><h2>FAQ</h2><h3>Q?</h3><p>A</p><h3>Q2?</h3><p>A2</p><h3>Q3?</h3><p>A3</p>"
        article = self.make_article(content)
        research = self.make_research(keywords=["test"])
        profile = self.make_profile("explainer")

        result = evaluate_article(article, "Test Title test", "Meta description with test here and it is long enough to pass the check", research, profile, False)
        self.assertFalse(result.passed)
        self.assertTrue(any("words" in r for r in result.reasons))

    def test_fails_faq_count(self):
        content = "<p>Direct answer with test keyword.</p><h2>Section</h2><p>Content.</p>"
        article = self.make_article(content)
        research = self.make_research(keywords=["test"])
        profile = self.make_profile("explainer")

        result = evaluate_article(article, "Test Title test", "Meta description with test here and it is long enough to pass the check", research, profile, False)
        self.assertFalse(result.passed)
        self.assertTrue(any("FAQ" in r for r in result.reasons))

    def test_fails_keyword_in_title(self):
        content = "<p>Direct answer with test keyword.</p><h2>Section</h2><p>Content.</p><h2>FAQ</h2><h3>Q?</h3><p>A</p><h3>Q2?</h3><p>A2</p><h3>Q3?</h3><p>A3</p>"
        article = self.make_article(content)
        research = self.make_research(keywords=["missing"])
        profile = self.make_profile("explainer")

        result = evaluate_article(article, "Test Title without it", "Meta description with missing here and it is long enough to pass the check", research, profile, False)
        self.assertFalse(result.passed)
        self.assertTrue(any("meta title" in r.lower() for r in result.reasons))

    def test_fails_keyword_in_first_100(self):
        # Keyword "missingkw" does not appear in first 100 words
        content = (
            "<p>Direct answer without the target term.</p>"
            "<h2>Section</h2><p>Content with other words but not the missing keyword.</p>"
            "<h2>FAQ</h2><h3>Q?</h3><p>A</p><h3>Q2?</h3><p>A2</p><h3>Q3?</h3><p>A3</p>"
        )
        article = self.make_article(content)
        research = self.make_research(keywords=["missingkw"])
        profile = self.make_profile("explainer")

        result = evaluate_article(article, "Test Title missingkw", "Meta description with missingkw here and it is long enough to pass the check", research, profile, False)
        self.assertFalse(result.passed)
        self.assertTrue(any("first 100" in r for r in result.reasons))

    def test_fails_keyword_in_h2(self):
        content = "<p>Direct answer with test keyword.</p><h2>Section without it</h2><p>Content.</p><h2>FAQ</h2><h3>Q?</h3><p>A</p><h3>Q2?</h3><p>A2</p><h3>Q3?</h3><p>A3</p>"
        article = self.make_article(content)
        research = self.make_research(keywords=["keyword"])
        profile = self.make_profile("explainer")

        result = evaluate_article(article, "Test Title keyword", "Meta description with keyword here and it is long enough to pass the check", research, profile, False)
        self.assertFalse(result.passed)
        self.assertTrue(any("<h2>" in r for r in result.reasons))

    def test_fails_meta_description_length(self):
        content = "<p>Direct answer with test.</p><h2>Section</h2><p>Content.</p><h2>FAQ</h2><h3>Q?</h3><p>A</p><h3>Q2?</h3><p>A2</p><h3>Q3?</h3><p>A3</p>"
        article = self.make_article(content)
        research = self.make_research(keywords=["test"])
        profile = self.make_profile("explainer")

        result = evaluate_article(article, "Test Title test", "Short", research, profile, False)
        self.assertFalse(result.passed)
        self.assertTrue(any("meta description" in r.lower() for r in result.reasons))

    def test_fails_heading_echo(self):
        # Heading "Section" echoed in first paragraph after it
        content = (
            "<p>Direct answer with test keyword.</p>"
            "<h2>Section</h2><p>Section starts with same words.</p>"
            "<h2>FAQ</h2><h3>Q?</h3><p>A</p><h3>Q2?</h3><p>A2</p><h3>Q3?</h3><p>A3</p>"
        )
        article = self.make_article(content)
        research = self.make_research(keywords=["test"])
        profile = self.make_profile("explainer")

        result = evaluate_article(article, "Test Title test", "Meta description with test here and it is long enough to pass the check", research, profile, False)
        # Should auto-fix the echo and pass
        self.assertTrue(result.passed or result.fixed_html is not None, f"Failed: {result.reasons}, fixed_html={result.fixed_html}")

    def test_fails_banned_word(self):
        content = "<p>In today's fast-paced world, we test.</p><h2>Section</h2><p>Content.</p><h2>FAQ</h2><h3>Q?</h3><p>A</p><h3>Q2?</h3><p>A2</p><h3>Q3?</h3><p>A3</p>"
        article = self.make_article(content)
        research = self.make_research(keywords=["test"])
        profile = self.make_profile("explainer")

        result = evaluate_article(article, "Test Title test", "Meta description with test here and it is long enough to pass the check", research, profile, False)
        self.assertFalse(result.passed)
        self.assertTrue(any("banned" in r.lower() for r in result.reasons))

    def test_truncated_json_fails(self):
        content = "<p>Direct answer with test.</p><h2>Section</h2><p>Content.</p><h2>FAQ</h2><h3>Q?</h3><p>A</p><h3>Q2?</h3><p>A2</p><h3>Q3?</h3><p>A3</p>"
        article = self.make_article(content)
        research = self.make_research(keywords=["test"])
        profile = self.make_profile("explainer")

        result = evaluate_article(article, "Test Title test", "Meta description with test here and it is long enough to pass the check", research, profile, True)
        self.assertFalse(result.passed)
        self.assertTrue(any("truncat" in r.lower() for r in result.reasons))

    def test_meta_only_failure_flag(self):
        # Content is valid but meta description too short - only meta should fail
        content = self._long_content()
        article = self.make_article(content)
        research = self.make_research(keywords=["test keyword"])
        profile = self.make_profile("explainer")

        # Meta description too short - only meta fails
        result = evaluate_article(article, "Test Title test keyword", "Short", research, profile, False)
        self.assertTrue(result.meta_only_failure, f"Expected meta_only_failure=True, got {result.meta_only_failure}, reasons={result.reasons}")

    def make_research(self, topic="Test Topic", keywords=None):
        from app.services.auto_blog import Research
        return Research(
            topic=topic,
            keywords=keywords or ["test keyword"],
            snippets=["Test snippet with 100 words and numbers 100%"],
            source_urls=["https://example.com"],
        )

    def make_profile(self, article_type="explainer"):
        base = {
            "article_type": article_type,
            "cover_photo": True,
            "max_content_photos": 1,
            "required_visuals": (),
            "max_visuals": 1,
            "downgraded": False,
        }
        return base


if __name__ == "__main__":
    unittest.main()