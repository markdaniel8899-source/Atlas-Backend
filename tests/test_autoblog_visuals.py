"""Tests for blog_visuals rendering and validation."""

from __future__ import annotations

import unittest

from app.services.blog_visuals import (
    render_table,
    render_chart,
    render_visual,
    render_visual_html_fallback,
    replace_visual_markers,
    _render_svg_bar_chart,
    _render_svg_line_chart,
)


class TestRenderTable(unittest.TestCase):
    def test_basic_table(self):
        v = {
            "kind": "table",
            "title": "Comparison",
            "caption": "Features compared",
            "columns": ["Feature", "Model A", "Model B"],
            "rows": [
                ["Speed", "100ms", "80ms"],
                ["Accuracy", "95%", "97%"],
            ],
            "source_url": "https://example.com",
            "intro_sentence": "The table below compares the two models.",
            "takeaway": "Model B is faster and more accurate.",
        }
        html = render_table(v)
        self.assertIn('<figure class="blog-visual">', html)
        self.assertIn('<table class="blog-table">', html)
        self.assertIn("<th>Feature</th>", html)
        self.assertIn("<th>Model A</th>", html)
        self.assertIn("<td>100ms</td>", html)
        self.assertIn('<figcaption>Features compared</figcaption>', html)
        self.assertIn("Source:", html)

    def test_table_highlights_best_for_benchmark(self):
        v = {
            "kind": "table",
            "title": "Benchmarks",
            "caption": "Speed comparison",
            "columns": ["Model", "Score"],
            "rows": [
                ["Model A", "100"],
                ["Model B", "120"],
                ["Model C", "90"],
            ],
            "source_url": "https://example.com",
            "intro_sentence": "Scores compared.",
            "takeaway": "Model B wins.",
        }
        html = render_table(v, highlight_best=True)
        self.assertIn('class="blog-best"', html)
        # Model B (120) should be highlighted
        self.assertIn('>120<', html)  # value present
        # Check that the best value cell has the class
        # The row with 120 should have blog-best class
        self.assertTrue('class="blog-best"' in html or 'blog-best' in html)

    def test_table_no_highlight_when_not_benchmark(self):
        v = {
            "kind": "table",
            "title": "Comparison",
            "caption": "Features",
            "columns": ["Feature", "A", "B"],
            "rows": [
                ["Speed", "100", "80"],
            ],
            "source_url": "https://example.com",
            "intro_sentence": "Comparison.",
            "takeaway": "A is faster.",
        }
        html = render_table(v, highlight_best=False)
        self.assertNotIn("blog-best", html)

    def test_table_empty_columns_returns_empty(self):
        v = {"kind": "table", "columns": [], "rows": []}
        self.assertEqual(render_table(v), "")

    def test_table_empty_rows_returns_empty(self):
        v = {"kind": "table", "columns": ["A"], "rows": []}
        self.assertEqual(render_table(v), "")


class TestRenderChart(unittest.TestCase):
    def test_bar_chart_renders_svg(self):
        v = {
            "kind": "bar",
            "title": "Growth Chart",
            "caption": "Revenue growth",
            "unit": "$M",
            "labels": ["Q1", "Q2", "Q3"],
            "values": [10, 15, 20],
            "source_url": "https://example.com/data",
            "intro_sentence": "The chart shows growth.",
            "takeaway": "Steady growth.",
        }
        html = render_chart(v)
        self.assertIn('<figure class="blog-visual">', html)
        self.assertIn('<svg', html)
        self.assertIn('viewBox="0 0 640 320"', html)
        self.assertIn('role="img"', html)
        self.assertIn("aria-label", html)
        self.assertIn("Growth Chart", html)
        self.assertIn("Revenue growth", html)
        self.assertIn("$M", html)
        self.assertIn("Q1", html)
        self.assertIn("10", html)
        self.assertIn("Source:", html)

    def test_line_chart_renders_svg(self):
        v = {
            "kind": "line",
            "title": "Latency Trend",
            "caption": "P99 latency",
            "unit": "ms",
            "labels": ["Jan", "Feb", "Mar"],
            "values": [100, 95, 90],
            "source_url": "https://example.com",
            "intro_sentence": "Latency improved.",
            "takeaway": "Getting faster.",
        }
        html = render_chart(v)
        self.assertIn('<svg', html)
        self.assertIn("Latency Trend", html)
        self.assertIn("ms", html)

    def test_chart_empty_returns_empty(self):
        v = {"kind": "bar", "labels": [], "values": []}
        self.assertEqual(render_chart(v), "")

    def test_chart_mismatched_labels_values_returns_empty(self):
        v = {"kind": "bar", "labels": ["A", "B"], "values": [1]}
        self.assertEqual(render_chart(v), "")

    def test_chart_non_numeric_values_returns_empty(self):
        v = {"kind": "bar", "labels": ["A", "B"], "values": ["x", "y"]}
        self.assertEqual(render_chart(v), "")

    def test_chart_unknown_kind_returns_empty(self):
        v = {"kind": "pie", "labels": ["A"], "values": [1]}
        self.assertEqual(render_chart(v), "")


class TestRenderVisualDispatcher(unittest.TestCase):
    def test_dispatch_table(self):
        v = {"kind": "table", "columns": ["A"], "rows": [["1"]]}
        html = render_visual(v)
        self.assertIn("<table", html)

    def test_dispatch_bar(self):
        v = {"kind": "bar", "labels": ["A"], "values": [1]}
        html = render_visual(v)
        self.assertIn("<svg", html)

    def test_dispatch_line(self):
        v = {"kind": "line", "labels": ["A"], "values": [1]}
        html = render_visual(v)
        self.assertIn("<svg", html)

    def test_dispatch_unknown_returns_empty(self):
        v = {"kind": "pie", "labels": ["A"], "values": [1]}
        self.assertEqual(render_visual(v), "")


class TestReplaceVisualMarkers(unittest.TestCase):
    def test_replaces_single_marker(self):
        content = "Intro <!--VISUAL:0--> after"
        visuals = [{"kind": "table", "columns": ["A"], "rows": [["1"]], "source_url": "x"}]
        out, dropped = replace_visual_markers(content, visuals)
        self.assertNotIn("<!--VISUAL:0-->", out)
        self.assertIn("<table", out)
        self.assertEqual(dropped, [])

    def test_missing_marker_drops_visual(self):
        content = "No marker here"
        visuals = [{"kind": "table", "columns": ["A"], "rows": [["1"]], "source_url": "x"}]
        out, dropped = replace_visual_markers(content, visuals)
        self.assertEqual(out, "No marker here")
        self.assertEqual(len(dropped), 1)
        self.assertIn("missing marker", dropped[0])

    def test_multiple_markers_replaced_in_order(self):
        content = "Start <!--VISUAL:0--> mid <!--VISUAL:1--> end"
        visuals = [
            {"kind": "table", "columns": ["A"], "rows": [["1"]], "source_url": "x"},
            {"kind": "bar", "labels": ["X"], "values": [1], "unit": "ms", "source_url": "y"},
        ]
        out, dropped = replace_visual_markers(content, visuals)
        self.assertNotIn("<!--VISUAL:0-->", out)
        self.assertNotIn("<!--VISUAL:1-->", out)
        self.assertIn("<table", out)
        self.assertIn("<svg", out)
        self.assertEqual(dropped, [])

    def test_leftover_markers_removed(self):
        content = "Start <!--VISUAL:0--> mid <!--VISUAL:2--> end"
        visuals = [{"kind": "table", "columns": ["A"], "rows": [["1"]], "source_url": "x"}]
        out, dropped = replace_visual_markers(content, visuals)
        self.assertNotIn("<!--VISUAL:0-->", out)
        self.assertNotIn("<!--VISUAL:2-->", out)

    def test_intro_sentence_inserted_when_missing(self):
        # Preceding paragraph doesn't mention table/chart
        content = "<p>Just some intro text.</p><!--VISUAL:0-->"
        visuals = [{
            "kind": "table",
            "title": "T",
            "columns": ["A"],
            "rows": [["1"]],
            "source_url": "x",
            "intro_sentence": "The table below shows the data.",
            "takeaway": "Key insight.",
        }]
        out, dropped = replace_visual_markers(content, visuals)
        # Should insert intro sentence
        self.assertIn("The table below shows the data", out)

    def test_intro_not_inserted_when_preceding_mentions_table(self):
        content = "<p>The table below shows data.</p><!--VISUAL:0-->"
        visuals = [{
            "kind": "table",
            "columns": ["A"],
            "rows": [["1"]],
            "source_url": "x",
        }]
        out, dropped = replace_visual_markers(content, visuals)
        # Should not duplicate intro
        self.assertEqual(out.count("The table below shows data"), 1)

    def test_takeaway_rendered(self):
        content = "<p>Intro</p><!--VISUAL:0-->"
        visuals = [{
            "kind": "table",
            "columns": ["A"],
            "rows": [["1"]],
            "source_url": "x",
            "takeaway": "This is the key takeaway.",
        }]
        out, dropped = replace_visual_markers(content, visuals)
        self.assertIn("This is the key takeaway", out)
        self.assertIn('class="blog-takeaway"', out)

    def test_highlight_best_for_benchmark_table(self):
        content = "<p>Intro</p><!--VISUAL:0-->"
        visuals = [{
            "kind": "table",
            "title": "Benchmarks",
            "columns": ["Model", "Score"],
            "rows": [["A", "100"], ["B", "120"]],
            "source_url": "x",
        }]
        out, dropped = replace_visual_markers(content, visuals, highlight_best=True)
        self.assertIn('class="blog-best"', out)


class TestRenderVisualHtmlFallback(unittest.TestCase):
    def test_bar_chart_fallback_renders_table(self):
        v = {
            "kind": "bar",
            "title": "Chart",
            "labels": ["A", "B"],
            "values": [10, 20],
            "unit": "units",
        }
        html = render_visual_html_fallback(v)
        self.assertIn("<table", html)
        self.assertIn("blog-chart-table", html)
        self.assertIn("A", html)
        self.assertIn("20", html)
        self.assertIn("units", html)

    def test_non_bar_returns_empty(self):
        v = {"kind": "line", "labels": ["A"], "values": [1]}
        self.assertEqual(render_visual_html_fallback(v), "")


class TestSvgRendering(unittest.TestCase):
    def test_bar_chart_svg_has_aria_label(self):
        v = {
            "kind": "bar",
            "title": "Test Chart",
            "labels": ["A", "B", "C"],
            "values": [1, 2, 3],
            "unit": "units",
        }
        svg = _render_svg_bar_chart(["A", "B", "C"], [1, 2, 3], "units", "Test Chart")
        self.assertIn('role="img"', svg)
        self.assertIn("aria-label", svg)
        self.assertIn("Test Chart", svg)

    def test_line_chart_svg_has_aria_label(self):
        v = {
            "kind": "line",
            "title": "Line Chart",
            "labels": ["A", "B"],
            "values": [1, 2],
            "unit": "ms",
        }
        svg = _render_svg_line_chart(["A", "B"], [1, 2], "ms", "Line Chart")
        self.assertIn('role="img"', svg)
        self.assertIn("aria-label", svg)
        self.assertIn("Line Chart", svg)

    def test_bar_chart_svg_has_value_labels(self):
        svg = _render_svg_bar_chart(["A", "B"], [10, 20], "units", "Chart")
        self.assertIn("10 units", svg)
        self.assertIn("20 units", svg)

    def test_bar_chart_svg_handles_negative_values(self):
        svg = _render_svg_bar_chart(["A", "B"], [-5, 10], "units", "Chart")
        self.assertIn("-5", svg)
        self.assertIn("10", svg)


if __name__ == "__main__":
    unittest.main()