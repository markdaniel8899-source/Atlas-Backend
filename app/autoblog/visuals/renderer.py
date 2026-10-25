"""Visual Renderer - Deterministic rendering and validation of visual specs."""

from __future__ import annotations

import html as html_lib
import json
import re
import urllib.parse
from typing import Any, Optional

from app.autoblog.models import FactSheet, KeywordPlan
from app.autoblog.visuals.labels import clean_label


def _fuzzy_match_h2(heading: str, h2_texts: list[str], threshold: float = 0.85) -> Optional[int]:
    """Find best matching H2 index using fuzzy matching."""
    from difflib import SequenceMatcher

    best_idx = None
    best_score = 0.0
    for i, h2 in enumerate(h2_texts):
        score = SequenceMatcher(None, heading.lower(), h2.lower()).ratio()
        if score > best_score and score >= threshold:
            best_score = score
            best_idx = i
    return best_idx


def _build_table(spec: dict) -> str:
    """Render table spec to proper HTML with semantic structure."""
    caption = spec.get("caption", "")
    columns = spec.get("columns", [])
    rows = spec.get("rows", [])
    source_url = spec.get("source_url")

    html = '<figure class="ab-figure">'
    html += '<div class="ab-table-wrap"><table class="ab-table">'
    if caption:
        html += f"<caption>{html_lib.escape(caption)}</caption>"
    html += "<thead><tr>"
    for col in columns:
        html += f"<th>{html_lib.escape(col)}</th>"
    html += "</tr></thead><tbody>"
    for row in rows:
        html += "<tr>"
        for cell in row:
            html += f"<td>{html_lib.escape(str(cell))}</td>"
        html += "</tr>"
    html += "</tbody></table></div>"
    if source_url:
        html += f'<figcaption>Source: <a href="{html_lib.escape(source_url, quote=True)}" target="_blank" rel="noopener">data source</a></figcaption>'
    html += "</figure>"
    return html


def _build_chart_url(numbers: list[dict]) -> str:
    """Build QuickChart.io URL with clean, truncated labels."""
    labels = []
    values = []
    label_map = {
        "total donations by elon musk": "Musk Donations",
        "approximate amount musk spent": "Trump Support",
        "amount openai is aiming to raise": "OpenAI Raise",
        "pre-money valuation targeted": "Pre-Money Val",
        "valuation mentioned in earlier report": "Earlier Val",
    }

    for item in numbers[:6]:
        metric = item.get("context", "").strip().lower()
        try:
            val = float(str(item.get("value", "")).replace(",", "").replace("%", "").split()[0])
            if val is not None:
                label = metric
                for k, v in label_map.items():
                    if k in metric:
                        label = v
                        break
                if len(label) > 15:
                    label = label[:12] + "..."
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


def render_table(spec: dict) -> str:
    """Render table spec to proper HTML."""
    caption = spec.get("caption", "")
    columns = spec.get("columns", [])
    rows = spec.get("rows", [])
    source_url = spec.get("source_url")

    html = '<figure class="ab-figure">'
    html += '<div class="ab-table-wrap"><table class="ab-table">'
    if caption:
        html += f"<caption>{html_lib.escape(caption)}</caption>"
    html += "<thead><tr>"
    for col in columns:
        html += f"<th>{html_lib.escape(col)}</th>"
    html += "</tr></thead><tbody>"
    for row in rows:
        html += "<tr>"
        for cell in row:
            html += f"<td>{html_lib.escape(str(cell))}</td>"
        html += "</tr>"
    html += "</tbody></table></div>"
    if source_url:
        html += f'<figcaption>Source: <a href="{html_lib.escape(source_url, quote=True)}" target="_blank" rel="noopener">data source</a></figcaption>'
    html += "</figure>"
    return html


def render_chart(spec: dict) -> str:
    """Render chart spec to QuickChart.io image."""
    chart_type = spec.get("chart_type", "bar")
    title = spec.get("title", "Data Chart")
    unit = spec.get("unit", "")
    x_label = spec.get("x_label")
    y_label = spec.get("y_label")
    points = spec.get("points", [])
    source_url = spec.get("source_url")
    why = spec.get("why", "")

    if len(points) < 2:
        return ""

    labels = [p.get("label", "") for p in points]
    values = [p.get("value", 0) for p in points]

    chart_config = {
        "type": "bar" if chart_type == "hbar" else chart_type,
        "data": {
            "labels": labels,
            "datasets": [{
                "label": unit,
                "data": values,
                "backgroundColor": "rgba(59, 130, 246, 0.8)"
            }]
        },
        "options": {
            "plugins": {
                "legend": {"display": False},
                "title": {
                    "display": True,
                    "text": title,
                    "font": {"size": 16}
                },
                "tooltip": {
                    "callbacks": {
                        "label": f"function(context) {{ return context.label + ': ' + context.parsed.y + ' {unit}'; }}"
                    }
                }
            },
            "scales": {
                "y": {
                    "beginAtZero": True,
                    "title": {"display": True, "text": y_label or unit},
                    "grid": {"color": "rgba(0,0,0,0.05)"}
                },
                "x": {
                    "title": {"display": True, "text": x_label or ""},
                    "grid": {"display": False}
                }
            }
        }
    }

    if chart_type == "hbar":
            chart_config["options"]["indexAxis"] = "y"

    chart_url = f"https://quickchart.io/chart?c={urllib.parse.quote(json.dumps(chart_config))}"

    html = f'<figure class="ab-figure">'
    html += f'<img src="{chart_url}" alt="{html_lib.escape(title)}" class="data-visualization" loading="lazy" style="max-width:100%;height:auto;" />'
    if why:
        html += f'<figcaption>{html_lib.escape(why)}</figcaption>'
    html += "</figure>"
    return html


def render_image(spec: dict, image_result: Any) -> str:
    """Render image spec with Pixabay image."""
    caption = spec.get("caption")
    alt = spec.get("alt", "")
    url = image_result.url if hasattr(image_result, 'url') else image_result.get('url', '')
    caption_text = spec.get("caption") or (image_result.caption if hasattr(image_result, 'caption') else "")

    html = f'<figure class="ab-figure">'
    html += f'<img src="{html_lib.escape(url)}" alt="{html_lib.escape(alt)}" class="content-image" loading="lazy" style="max-width:100%;height:auto;" />'
    if caption_text:
        html += f'<figcaption>{html_lib.escape(caption_text)}</figcaption>'
    html += "</figure>"
    return html


def render_stat_cards(spec: dict) -> str:
    """Render stat cards."""
    cards = spec.get("cards", [])
    if len(cards) < 2:
        return ""

    html = '<div class="ab-stats">'
    for card in cards[:4]:
        value = card.get("value", "")
        label = card.get("label", "")
        html += f'<div><div class="ab-stat-value">{html_lib.escape(value)}</div><div class="ab-stat-label">{html_lib.escape(label)}</div></div>'
    html += "</div>"
    return html


def validate_and_render(spec: dict, article_text: str, fact_sheet: Any) -> tuple[bool, str]:
    """Validate spec and render if valid. Returns (is_valid, html)."""
    vtype = spec.get("type")
    after_heading = spec.get("after_section_heading", "")

    if not after_heading:
        return False, ""

    if vtype == "table":
        columns = spec.get("columns", [])
        rows = spec.get("rows", [])
        if len(columns) < 2 or len(rows) < 2:
            return False, ""
        html = render_table(spec)
        return True, html

    elif vtype == "chart":
        points = spec.get("points", [])
        if len(points) < 3:
            return False, ""
        html = render_chart(spec)
        return True, html if html else ""

    elif vtype == "image":
        return True, ""

    elif vtype == "stat_cards":
        cards = spec.get("cards", [])
        if len(cards) < 2:
            return False, ""
        html = render_stat_cards(spec)
        return True, html

    return False, ""


def inject_visuals(text: str, specs: list[dict], fact_sheet: Any) -> str:
    """Inject validated visuals into article text at correct H2 positions."""
    if not specs:
        return text

    h2_matches = list(re.finditer(r"<h2[^>]*>(.*?)</h2>", text, re.IGNORECASE | re.DOTALL))
    h2_texts = [re.sub(r"<[^>]+>", "", m.group(1)).strip() for m in h2_matches]
    h2_positions = [m.end() for m in re.finditer(r"</h2>", text, re.IGNORECASE)]

    if not h2_matches:
        return text

    visuals_by_position: dict[int, list[str]] = {}

    for spec in specs:
        after_heading = spec.get("after_section_heading", "")
        idx = _fuzzy_match_h2(after_heading, h2_texts)
        if idx is None:
            continue

        is_valid, html = validate_and_render(spec, "", None)
        if not is_valid:
            continue

        visuals_by_position.setdefault(idx, []).append(spec.get("html", "") or render_spec(spec))

    result = text
    for idx in sorted(visuals_by_position.keys(), reverse=True):
        if idx < len(h2_positions):
            pos = h2_positions[idx]
            visual_block = "\n".join(visuals_by_position[idx])
            result = result[:pos] + f"\n{' '.join(visuals_by_position[idx])}\n" + result[pos:]

    return result


def render_spec(spec: dict) -> str:
    """Render a spec to HTML."""
    vtype = spec.get("type")
    if vtype == "table":
        return render_table(spec)
    elif vtype == "chart":
        return render_chart(spec)
    elif vtype == "image":
        return ""
    elif vtype == "stat_cards":
        return render_stat_cards(spec)
    return ""


import json
import re
from typing import Any, Optional
from difflib import SequenceMatcher

__all__ = ["render_table", "render_chart", "render_image", "render_stat_cards", "validate_and_render", "inject_visuals"]