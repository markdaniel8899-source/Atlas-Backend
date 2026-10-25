"""Fact-checking module for ATLAS articles."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Optional

from app.autoblog.llm.clients import (
    BlogPipelineError,
    extract_json,
    groq_research_with_nemotron_fallback,
)
from app.autoblog.research import FactSheet
from app.autoblog.writer.dynamic import _NUMBER_RE
from app.autoblog.sources import AI_MODEL_RE


@dataclass
class ClaimCheck:
    claim: str
    kind: Literal["number", "entity", "date", "quote", "other"]
    status: Literal["supported", "unsupported", "contradicted"]
    evidence: Optional[str]
    sentence_index: int


@dataclass
class FactCheckResult:
    checks: list
    fixed_html: str
    removed_count: int
    rewritten_count: int
    passed: bool


_NUMBER_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(%|percent|billion|million|trillion|thousand|ms|GB|TB|MB|tokens?|users?|USD|\$)",
    re.IGNORECASE,
)


def _normalize_number(value: str, unit: str) -> tuple[float, str]:
    """Normalize number to standard format."""
    value = value.replace(",", "").strip()
    try:
        num = float(value)
    except ValueError:
        return 0.0, unit

    unit = unit.lower().strip()
    if unit in ("percent", "%"):
        return num, "%"
    elif unit in ("billion", "b"):
        return num * 1_000_000_000, "B"
    elif unit in ("million", "m"):
        return num * 1_000_000, "M"
    elif unit in ("trillion", "t"):
        return num * 1_000_000_000_000, "T"
    elif unit in ("thousand", "k"):
        return num * 1000, "K"
    elif unit in ("usd", "$"):
        return num, "$"
    return num, unit


def _extract_article_numbers(text: str) -> list[dict]:
    """Extract numbers with context from article text."""
    numbers = []
    for match in _NUMBER_RE.finditer(text):
        value, unit = match.group(1), match.group(2)
        start = max(match.start() - 100, 0)
        end = min(match.end() + 100, len(text))
        context = text[start:end].strip()
        norm_val, norm_unit = _normalize_number(value, unit)
        numbers.append({
            "raw_value": match.group(0),
            "value": value,
            "unit": unit,
            "normalized_value": norm_val,
            "normalized_unit": norm_unit,
            "context": context,
            "position": match.start(),
        })
    return numbers


def _extract_article_entities(text: str) -> list[dict]:
    """Extract model/company names from article text."""
    entities = []
    for model in AI_MODEL_RE.findall(text):
        model = " ".join(model.split())
        entities.append({
            "name": model,
            "type": "model",
            "context": model,
        })
    # Deduplicate
    seen = set()
    unique = []
    for e in entities:
        key = e["name"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


def _find_sentence_index(text: str, position: int) -> int:
    """Find which sentence index a position falls in."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pos = 0
    for i, sent in enumerate(sentences):
        pos += len(sent) + 1
        if pos > position:
            return i
    return len(sentences) - 1


def _match_fact_sheet(number: dict, fact_sheet: FactSheet) -> tuple[bool, Optional[str]]:
    """Check if a number is supported by fact_sheet."""
    norm_val = number["normalized_value"]
    norm_unit = number["normalized_unit"]

    for item in fact_sheet.numbers:
        fs_val = str(item.get("value", "")).replace(",", "").strip()
        fs_unit = item.get("unit", "").strip().lower()
        try:
            fs_norm_val = float(fs_val)
            fs_norm_unit = fs_unit
        except (ValueError, TypeError):
            continue

        if fs_norm_unit == number["normalized_unit"] and abs(fs_norm_val - number["normalized_value"]) < 0.01 * max(abs(number["normalized_value"]), 1):
            return True, item.get("source_url")

    return False, None


async def fact_check_article(draft: Any, fact_sheet: FactSheet) -> dict:
    """
    Fact-check an article against the fact sheet.

    Returns FactCheckResult as dict.
    """
    text = draft.content_html if hasattr(draft, 'content_html') else draft.get('content_html', '')

    # Strip HTML for text analysis
    clean_text = re.sub(r"<[^>]+>", " ", text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    # Extract claims from article
    article_numbers = _extract_article_numbers(text)
    article_entities = _extract_article_entities(text)

    checks = []
    sentence_indices = {}

    # Check numbers
    for num in article_numbers:
        sentence_idx = _find_sentence_index(text, num["position"])
        supported, evidence = _match_fact_sheet(num, fact_sheet)
        status = "supported" if supported else "unsupported"
        evidence_text = evidence or num.get("context", "")[:200]

        checks.append({
            "claim": num["raw_value"],
            "kind": "number",
            "status": status,
            "evidence": evidence_text if evidence else None,
            "sentence_index": sentence_idx,
        })
        sentence_indices.setdefault(sentence_idx, []).append(len(checks) - 1)

    # Check entities
    for entity in article_entities:
        # Find position in text
        pos = text.find(entity["name"])
        if pos >= 0:
            sentence_idx = _find_sentence_index(text, pos)
            # Check if entity exists in fact_sheet
            supported = any(entity["name"].lower() in e.get("name", "").lower()
                            for e in fact_sheet.entities)
            status = "supported" if supported else "unsupported"
            checks.append({
                "claim": entity["name"],
                "kind": "entity",
                "status": status,
                "evidence": None if supported else "Entity not found in fact sheet",
                "sentence_index": sentence_idx,
            })
            sentence_indices.setdefault(sentence_idx, []).append(len(checks) - 1)

    # LLM pass for complex claims (contradicted numbers)
    unsupported_indices = [i for i, c in enumerate(checks) if c["status"] == "unsupported"]
    contradicted_indices = [i for i, c in enumerate(checks) if c["status"] == "contradicted"]

    # For now, skip LLM pass and use deterministic only
    # (Can add LLM pass later if needed)

    # Build fixed HTML
    fixed_html = ""
    # For now, just return the checks

    passed = all(c["status"] == "supported" for c in checks)

    return {
        "checks": checks,
        "fixed_html": "",  # Will be filled by rewrite logic
        "removed_count": 0,
        "rewritten_count": 0,
        "passed": passed,
    }


async def fact_check_article_async(draft: Any, fact_sheet: FactSheet) -> dict:
    """Async wrapper for fact_check_article."""
    return await fact_check_article(draft, fact_sheet)


__all__ = ["fact_check_article_async", "fact_check_article"]