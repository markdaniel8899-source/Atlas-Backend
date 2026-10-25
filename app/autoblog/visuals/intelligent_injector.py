"""Visual Intelligence Module - Smart visual generation and injection for ATLAS."""

from __future__ import annotations

import html as html_lib
import json
import re
import urllib.parse
from typing import Any

from app.autoblog.config.settings import get_autoblog_settings
from app.autoblog.images import fetch_content_image
from app.autoblog.images.dynamic import ImageResult, inject_content_images
from app.autoblog.models import KeywordPlan, TopicTrend
from app.autoblog.research import FactSheet
from app.autoblog.visuals.labels import clean_label


class VisualIntelligence:
    """Smart visual generation and injection for ATLAS articles."""

    def __init__(self, keywords: KeywordPlan, fact_sheet: FactSheet):
        self.keywords = keywords
        self.fact_sheet = fact_sheet
        self.used_image_urls: set[str] = set()
        self.visual_count = 0
        self.max_visuals = 3

    async def analyze_content_needs(
        self, text: str, key_data_points: list[str]
    ) -> list[dict[str, Any]]:
        """
        Analyze content to determine visual needs.
        Returns list of visual requirements with exact placement.
        """
        needs = []
        h2_positions = [(m.start(), m.end()) for m in re.finditer(r"<h2[^>]*>", text, re.IGNORECASE)]

        # Analyze key_data_points for structured data
        has_structured_data = len(self.fact_sheet.numbers) >= 2
        has_comparison_data = any(
            "%" in str(item.get("value", "")) or "growth" in item.get("context", "").lower()
            for item in self.fact_sheet.numbers
        )

        # Determine visual needs based on content type
        if has_structured_data and len(self.fact_sheet.numbers) >= 3:
            # Data-driven article: table + chart
            needs.append({
                "type": "table",
                "after_h2": 1,
                "data": self.fact_sheet.numbers,
                "priority": 1,
            })
            needs.append({
                "type": "chart",
                "after_h2": 2 if len(h2_positions) >= 2 else 1,
                "data": self.fact_sheet.numbers,
                "priority": 2,
            })
        elif has_structured_data:
            # Some data but not enough for chart - just table
            needs.append({
                "type": "table",
                "after_h2": 1,
                "data": self.fact_sheet.numbers,
                "priority": 1,
            })

        # Image for theory/concept articles (after H2 3)
        if len(h2_positions) >= 3:
            needs.append({
                "type": "image",
                "after_h2": 3,
                "query": self.keywords.primary_keyword,
                "priority": 3,
            })

        # Sort by priority and limit
        needs.sort(key=lambda x: x["priority"])
        return needs[:self.max_visuals]

    def _build_table(self, numbers: list[dict]) -> tuple[str, list[dict]]:
        """Build proper HTML table from fact_sheet numbers."""
        rows = []
        for item in numbers[:6]:
            metric = item.get("context", "").strip()
            value = f"{item.get('value', '')} {item.get('unit', '')}".strip()
            source = item.get("source_url", "")
            if metric and value:
                rows.append({"metric": metric, "value": value, "source": source})

        if not rows:
            return "", []

        table_html = '<table class="data-table"><thead><tr><th>Metric</th><th>Value</th><th>Source</th></tr></thead><tbody>'
        for row in rows:
            table_html += f"<tr><td>{html_lib.escape(row['metric'])}</td><td>{html_lib.escape(row['value'])}</td><td>"
            if row["source"]:
                table_html += f"<a href='{html_lib.escape(row['source'], quote=True)}' target='_blank' rel='noopener'>Source</a>"
            table_html += "</td></tr>"
        table_html += "</tbody></table>"
        return table_html, rows

    def _build_chart_url(self, numbers: list[dict]) -> str:
        """Build QuickChart.io URL with clean, truncated labels using clean_label."""
        labels = []
        values = []
        for item in numbers[:6]:
            metric = item.get("context", "").strip().lower()
            try:
                val = float(str(item.get("value", "")).replace(",", "").replace("%", "").split()[0])
                if val is not None:
                    # Use clean_label for automatic cleaning and truncation
                    label = clean_label(item.get("context", ""), max_len=15)
                    labels.append(f'"{html_lib.escape(label)}"')
                    values.append(str(val))
            except (ValueError, IndexError):
                continue

        if not labels or not values:
            return ""

        chart_config = {
            "type": "bar",
            "data": {
                "labels": [l.strip('"') for l in labels],
                "datasets": [{
                    "label": "Value",
                    "data": [float(v) for v in values],
                    "backgroundColor": "rgba(59, 130, 246, 0.8)"
                }]
            },
            "options": {
                "plugins": {"legend": {"display": False}},
                "scales": {"y": {"beginAtZero": True, "max": max([float(v) for v in values]) * 1.2}}
            }
        }
        return f"https://quickchart.io/chart?c={urllib.parse.quote(json.dumps(chart_config))}"

    async def generate_dynamic_visual(self, req: dict[str, Any]) -> dict[str, Any]:
        """Generate a single visual based on requirement."""
        vtype = req["type"]

        if vtype == "table":
            table_html, _ = self._build_table(req.get("data", self.fact_sheet.numbers))
            return {
                "type": "table",
                "html": table_html,
                "after_h2": req["after_h2"],
            }

        elif vtype == "chart":
            chart_url = self._build_chart_url(req.get("data", self.fact_sheet.numbers))
            if chart_url:
                return {
                    "type": "chart",
                    "html": f'<img src="{chart_url}" alt="Data Chart" class="data-visualization" loading="lazy" style="max-width:100%;height:auto;" />',
                    "after_h2": req["after_h2"],
                }
            return {"type": "chart", "html": "", "after_h2": req["after_h2"]}

        elif vtype == "image":
            # Fetch UNIQUE image from Pixabay
            query = req.get("query", self.keywords.primary_keyword)
            try:
                images = await fetch_content_image(
                    req.get("query", "AI technology"),
                    self.keywords.primary_keyword,
                    count=3,  # Fetch more to find unused
                )
                for img in images:
                    if img.url not in self.used_image_urls:
                        self.used_image_urls.add(img.url)
                        return {
                            "type": "image",
                            "html": f'<img src="{img.url}" alt="{html_lib.escape(img.caption)}" class="content-image" loading="lazy" style="max-width:100%;height:auto;" />',
                            "after_h2": req["after_h2"],
                        }
            except Exception:
                pass
            return {"type": "image", "html": "", "after_h2": req["after_h2"]}

        return {"type": vtype, "html": "", "after_h2": req["after_h2"]}

    def inject_visuals_intelligently(self, text: str, visuals: list[dict]) -> str:
        """Inject visuals at exact H2 positions."""
        if not visuals:
            return text

        # Find all H2 positions
        h2_matches = list(re.finditer(r"</h2>", text, re.IGNORECASE))
        if not h2_matches:
            return text

        # Build insertion map: after_h2_index -> list of html blocks
        insertions: dict[int, list[str]] = {}
        for v in visuals:
            idx = v.get("after_h2", 1) - 1  # Convert to 0-index
            if 0 <= idx < len(h2_matches):
                insertions.setdefault(idx, []).append(v["html"])

        # Insert from end to start to preserve positions
        result = text
        for idx in sorted(insertions.keys(), reverse=True):
            pos = h2_matches[idx].end()
            visual_block = "\n".join(insertions[idx])
            result = result[:pos] + f"\n{' '.join(insertions[idx])}\n" + result[pos:]

        return result


async def analyze_content_needs(text: str, key_data_points: list[str], fact_sheet: FactSheet, keywords: KeywordPlan) -> list[dict]:
    """Standalone function for pipeline integration."""
    vi = VisualIntelligence(keywords, fact_sheet)
    return await vi.analyze_content_needs(text, key_data_points)


async def generate_dynamic_visual(req: dict, fact_sheet: FactSheet, keywords: KeywordPlan) -> dict:
    """Standalone function for pipeline integration."""
    vi = VisualIntelligence(keywords, fact_sheet)
    return await vi.generate_dynamic_visual(req)


def inject_visuals_intelligently(text: str, visuals: list[dict], fact_sheet: FactSheet, keywords: KeywordPlan) -> str:
    """Standalone function for pipeline integration."""
    vi = VisualIntelligence(keywords, fact_sheet)
    return vi.inject_visuals_intelligently(text, visuals)