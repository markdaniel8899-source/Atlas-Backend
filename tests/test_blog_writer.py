from __future__ import annotations

import unittest

from app.services.blog.writer import WriterParseError, parse_writer_output


def article(content: str) -> str:
    return (
        "<TITLE>Test article title</TITLE>\n"
        "<SLUG>test-article-title</SLUG>\n"
        "<EXCERPT>"
        "A practical guide for students who want to understand the topic and "
        "apply it step by step in their own study workflow."
        "</EXCERPT>\n"
        f"<CONTENT>{content}</CONTENT>"
    )


class WriterParseTests(unittest.TestCase):
    def test_repairs_bare_table_headers_and_rows(self) -> None:
        output = parse_writer_output(
            article(
                "<h2>Comparison</h2>"
                "<table>"
                "<tr><th>Metric</th><th>Value</th></tr>"
                "<tr><td>Speed</td><td>Fast</td></tr>"
                "</table>"
            )
        )
        self.assertIn('<th scope="col">Metric</th>', output.content)
        self.assertIn("<thead>", output.content)
        self.assertIn("<tbody>", output.content)
        self.assertIn("</tbody></table>", output.content)

    def test_preserves_existing_header_scope(self) -> None:
        output = parse_writer_output(
            article(
                "<h2>Comparison</h2>"
                '<table><thead><tr><th scope="row">Metric</th></tr></thead>'
                "<tbody><tr><td>Speed</td></tr></tbody></table>"
            )
        )
        self.assertIn('<th scope="row">Metric</th>', output.content)
        self.assertNotIn('scope="col" scope="row"', output.content)

    def test_rejects_output_without_content(self) -> None:
        with self.assertRaises(WriterParseError):
            parse_writer_output(
                "<TITLE>Test article title</TITLE>"
                "<SLUG>test-article-title</SLUG>"
                "<EXCERPT>Short excerpt.</EXCERPT>"
            )


if __name__ == "__main__":
    unittest.main()
