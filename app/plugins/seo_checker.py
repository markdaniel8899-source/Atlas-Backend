"""Zero-RAM SEO checker - Google's core on-page rules with stdlib only.

No spaCy / nltk / redis / any NLP package: just `re` + string operations,
so evaluation costs microseconds and zero extra RAM on the 1GB box.

Five atomic checks, 20 points each (score 0-100):

  1. Word count   - content has MORE than 300 words.
  2a. Keyword     - the target keyword appears in the meta title.
  2b. Keyword     - the target keyword appears in the first 100 words.
  3. Structure    - content has at least one <h2> or <h3> heading.
  4. Meta         - meta description is 120-160 characters long.

score >= 80 -> passed (i.e. at most one check may fail).
"""

from __future__ import annotations

import re

_RULE_POINTS = 20
_PASS_SCORE = 80
_MIN_WORDS = 300
_META_MIN = 120
_META_MAX = 160

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_HEADING_RE = re.compile(r"<h[23][\s>]", re.IGNORECASE)


def _plain_text(content: str) -> str:
    """Strip HTML tags and collapse whitespace - words only, no markup."""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", content)).strip()


def evaluate_seo_lightweight(
    title: str,
    content: str,
    keyword: str,
    meta_description: str,
) -> dict:
    """Score a draft against four lightweight Google on-page rules.

    Returns {"score": 0-100, "passed": bool, "feedback": "..."} where
    `feedback` is a single human-readable sentence (also suitable as a
    rewrite instruction for the AI writer).
    """
    title = str(title or "")
    content = str(content or "")
    keyword = str(keyword or "").strip()
    meta = str(meta_description or "")

    plain = _plain_text(content)
    words = plain.split()
    first_100 = " ".join(words[:100]).lower()

    score = 0
    issues: list[str] = []

    # Rule 1 - word count > 300.
    if len(words) > _MIN_WORDS:
        score += _RULE_POINTS
    else:
        issues.append(f"content has only {len(words)} words (needs > {_MIN_WORDS})")

    # Rules 2a + 2b - keyword in the title and in the first 100 words
    # (graded separately so the score still reaches 0-100 in 20-pt steps).
    if not keyword:
        issues.append("no target keyword provided")
    else:
        if keyword.lower() not in title.lower():
            issues.append(f"keyword '{keyword}' is missing from the meta title")
        else:
            score += _RULE_POINTS
        if keyword.lower() not in first_100:
            issues.append(
                f"keyword '{keyword}' is missing from the first 100 words"
            )
        else:
            score += _RULE_POINTS

    # Rule 3 - at least one <h2>/<h3> heading.
    if _HEADING_RE.search(content):
        score += _RULE_POINTS
    else:
        issues.append("content has no <h2> or <h3> heading")

    # Rule 4 - meta description 120-160 chars.
    if _META_MIN <= len(meta) <= _META_MAX:
        score += _RULE_POINTS
    else:
        issues.append(
            f"meta description is {len(meta)} chars (needs {_META_MIN}-{_META_MAX})"
        )

    passed = score >= _PASS_SCORE
    if not issues:
        feedback = "All SEO checks passed."
    elif passed:
        feedback = "Passed with minor issues: " + "; ".join(issues) + "."
    else:
        feedback = "Failed: " + "; ".join(issues) + "."

    return {"score": score, "passed": passed, "feedback": feedback}
