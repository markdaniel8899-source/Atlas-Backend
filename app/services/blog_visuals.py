"""Render structured visuals (tables, charts) for AutoBlog articles.

No external dependencies — pure stdlib HTML/SVG generation.
"""

from __future__ import annotations

import html
import re
from typing import Any

# Dark theme colors
AXIS_COLOR = "#94a3b8"
SERIES_COLOR = "#38bdf8"
GRID_COLOR = "#1e293b"
BG_COLOR = "#0f172a"
TEXT_COLOR = "#e2e8f0"
HIGHLIGHT_COLOR = "#fbbf24"


def _escape(text: str) -> str:
    return html.escape(str(text), quote=True)


def _is_number(val: Any) -> bool:
    if isinstance(val, (int, float)):
        return True
    if isinstance(val, str):
        s = val.strip().replace(",", "").replace("%", "")
        try:
            float(s)
            return True
        except ValueError:
            return False
    return False


def _to_number(val: Any) -> float | None:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = val.strip().replace(",", "").replace("%", "")
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _canonical_number(val: Any) -> float | None:
    """Convert to a canonical numeric value for grounding checks."""
    return _to_number(val)


def render_table(v: dict[str, Any], highlight_best: bool = False) -> str:
    """Render a table visual as HTML figure with table."""
    title = _escape(v.get("title", "Table"))
    caption = _escape(v.get("caption", ""))
    columns = [str(c) for c in v.get("columns", [])]
    rows = v.get("rows", [])

    if not columns or not rows:
        return ""

    # Detect numeric columns for potential highlighting
    numeric_cols: list[bool] = []
    col_values: list[list[float]] = [[] for _ in columns]

    for row in rows:
        for i, cell in enumerate(row):
            if i < len(columns):
                num = _to_number(cell)
                if num is not None:
                    col_values[i].append(num)

    for i in range(len(columns)):
        numeric_cols.append(len(col_values[i]) >= 2 and len(set(col_values[i])) > 1)

    # Determine best value per numeric column (higher is better by default)
    best_per_col: dict[int, float] = {}
    if highlight_best:
        for i, is_num in enumerate(numeric_cols):
            if is_num and col_values[i]:
                best_per_col[i] = max(col_values[i])

    # Build HTML
    parts = [
        '<figure class="blog-visual">',
        f'<table class="blog-table">',
        "<thead><tr>",
    ]
    for col in columns:
        parts.append(f"<th>{_escape(col)}</th>")
    parts.append("</tr></thead><tbody>")

    for row in rows:
        parts.append("<tr>")
        for i, cell in enumerate(row):
            cell_escaped = _escape(cell)
            td_class = ""
            if i in best_per_col:
                cell_num = _to_number(cell)
                if cell_num is not None and cell_num == best_per_col[i]:
                    td_class = ' class="blog-best"'
            parts.append(f"<td{td_class}>{cell_escaped}</td>")
        parts.append("</tr>")

    parts.extend(
        [
            "</tbody></table>",
            f'<figcaption>{caption}</figcaption>',
            f'<div class="blog-visual-source">Source: {_escape(v.get("source_url", ""))}</div>',
        ]
    )

    # Add takeaway if provided
    takeaway = v.get("takeaway", "").strip()
    if takeaway:
        parts.append(f'<p class="blog-takeaway">{_escape(takeaway)}</p>')

    parts.append("</figure>")
    return "".join(parts)


def _render_svg_bar_chart(
    labels: list[str],
    values: list[float],
    unit: str,
    title: str,
    width: int = 640,
    height: int = 320,
) -> str:
    """Generate inline SVG for a horizontal bar chart."""
    if not labels or not values or len(labels) != len(values):
        return ""

    max_val = max(values) if values else 1
    min_val = min(values) if values else 0
    # Include zero for baseline
    scale_min = min(0, min_val)
    scale_max = max(max_val, 0)
    value_range = scale_max - scale_min
    if value_range == 0:
        value_range = 1

    padding_left = 160
    padding_right = 40
    padding_top = 30
    padding_bottom = 40
    chart_width = width - padding_left - padding_right
    chart_height = height - padding_top - padding_bottom

    # Build bars
    bars = []
    for i, (label, val) in enumerate(zip(labels, values)):
        y = padding_top + i * (chart_height / len(labels)) + (chart_height / len(labels)) * 0.1
        bar_height = (chart_height / len(labels)) * 0.8

        # Normalize value to 0-1 within range
        norm = (val - scale_min) / value_range
        bar_width = norm * chart_width

        x_start = padding_left
        if val < 0:
            x_start = padding_left + (-scale_min / value_range) * chart_width
            bar_width = abs(val - scale_min) / value_range * chart_width

        bar_color = SERIES_COLOR
        if val < 0:
            bar_color = "#f87171"

        bars.append(
            f'<rect x="{x_start:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" fill="{bar_color}" rx="2"/>'
        )

        # Value label
        label_x = x_start + bar_width + 5 if val >= 0 else x_start - 5
        bars.append(
            f'<text x="{label_x:.1f}" y="{y + bar_height / 2 + 4:.1f}" '
            f'font-size="11" fill="{TEXT_COLOR}" text-anchor="start" dominant-baseline="middle">'
            f'{_escape(str(val))} {_escape(unit)}</text>'
        )

    # Grid lines
    grid_lines = []
    for g in range(5):
        x = padding_left + (g / 4) * chart_width
        grid_lines.append(
            f'<line x1="{x:.1f}" y1="{padding_top:.1f}" '
            f'x2="{x:.1f}" y2="{padding_top + chart_height:.1f}" '
            f'stroke="{GRID_COLOR}" stroke-width="0.5" stroke-dasharray="2,2"/>'
        )

    # Y-axis labels
    y_labels = []
    for i, label in enumerate(labels):
        y = padding_top + i * (chart_height / len(labels)) + (chart_height / len(labels)) / 2
        y_labels.append(
            f'<text x="{padding_left - 10:.1f}" y="{y:.1f}" '
            f'font-size="11" fill="{TEXT_COLOR}" text-anchor="end" dominant-baseline="middle">'
            f'{_escape(label)}</text>'
        )

    # X-axis
    x_axis = (
        f'<line x1="{padding_left}" y1="{padding_top + chart_height}" '
        f'x2="{width - padding_right}" y2="{padding_top + chart_height}" '
        f'stroke="{AXIS_COLOR}" stroke-width="1"/>'
    )

    # Title
    title_el = (
        f'<text x="{width / 2}" y="20" font-size="14" fill="{TEXT_COLOR}" '
        f'text-anchor="middle" font-weight="600">{_escape(title)}</text>'
    )

    return f'''<svg viewBox="0 0 {width} {height}" role="img" 
        aria-label="{_escape(f"Bar chart: {title}. Values: {', '.join(f'{l}: {v} {unit}' for l, v in zip(labels, values))}")}"
        style="background:{BG_COLOR}; border-radius:8px; max-width:100%; height:auto;">
    {title_el}
    {"".join(grid_lines)}
    {"".join(bars)}
    {"".join(y_labels)}
    {x_axis}
</svg>'''


def _render_svg_line_chart(
    labels: list[str],
    values: list[float],
    unit: str,
    title: str,
    width: int = 640,
    height: int = 320,
) -> str:
    """Generate inline SVG for a line chart."""
    if not labels or not values or len(labels) != len(values):
        return ""

    max_val = max(values) if values else 1
    min_val = min(values) if values else 0
    padding = (max_val - min_val) * 0.1 if max_val != min_val else 1
    scale_min = min_val - padding
    scale_max = max_val + padding
    value_range = scale_max - scale_min
    if value_range == 0:
        value_range = 1

    padding_left = 70
    padding_right = 30
    padding_top = 40
    padding_bottom = 50
    chart_width = width - padding_left - padding_right
    chart_height = height - padding_top - padding_bottom

    points = []
    for i, val in enumerate(values):
        x = padding_left + (i / (len(values) - 1)) * chart_width if len(values) > 1 else padding_left + chart_width / 2
        y = padding_top + chart_height - ((val - scale_min) / value_range) * chart_height
        points.append(f"{x:.1f},{y:.1f}")

    # Grid lines (horizontal)
    grid_lines = []
    for g in range(5):
        y = padding_top + (g / 4) * chart_height
        grid_lines.append(
            f'<line x1="{padding_left}" y1="{y:.1f}" '
            f'x2="{width - padding_right}" y2="{y:.1f}" '
            f'stroke="{GRID_COLOR}" stroke-width="0.5" stroke-dasharray="2,2"/>'
        )

    # Y-axis value labels
    y_labels = []
    for g in range(5):
        y = padding_top + (g / 4) * chart_height
        val = scale_max - (g / 4) * value_range
        y_labels.append(
            f'<text x="{padding_left - 10}" y="{y:.1f}" '
            f'font-size="11" fill="{TEXT_COLOR}" text-anchor="end" dominant-baseline="middle">'
            f'{_escape(f"{val:.1f}")}</text>'
        )

    # X-axis labels
    x_labels = []
    for i, label in enumerate(labels):
        x = padding_left + (i / (len(labels) - 1)) * chart_width if len(labels) > 1 else padding_left + chart_width / 2
        x_labels.append(
            f'<text x="{x:.1f}" y="{padding_top + chart_height + 20:.1f}" '
            f'font-size="11" fill="{TEXT_COLOR}" text-anchor="middle" dominant-baseline="middle">'
            f'{_escape(label)}</text>'
        )

    # Axis lines
    x_axis = (
        f'<line x1="{padding_left}" y1="{padding_top + chart_height}" '
        f'x2="{width - padding_right}" y2="{padding_top + chart_height}" '
        f'stroke="{AXIS_COLOR}" stroke-width="1"/>'
    )
    y_axis = (
        f'<line x1="{padding_left}" y1="{padding_top}" '
        f'x2="{padding_left}" y2="{padding_top + chart_height}" '
        f'stroke="{AXIS_COLOR}" stroke-width="1"/>'
    )

    # Line path
    line_path = (
        f'<polyline points="{" ".join(points)}" '
        f'fill="none" stroke="{SERIES_COLOR}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>'
    )

    # Points with value labels
    point_elements = []
    for i, (x_str, y_str) in enumerate(
        [(p.split(",")[0], p.split(",")[1]) for p in points]
    ):
        point_elements.append(
            f'<circle cx="{x_str}" cy="{y_str}" r="4" fill="{SERIES_COLOR}"/>'
        )
        point_elements.append(
            f'<text x="{x_str}" y="{float(y_str) - 10}" font-size="11" fill="{TEXT_COLOR}" '
            f'text-anchor="middle">{_escape(str(values[i]))} {_escape(unit)}</text>'
        )

    title_el = (
        f'<text x="{width / 2}" y="20" font-size="14" fill="{TEXT_COLOR}" '
        f'text-anchor="middle" font-weight="600">{_escape(title)}</text>'
    )

    return f'''<svg viewBox="0 0 {width} {height}" role="img"
        aria-label="{_escape(f"Line chart: {title}. Values: {', '.join(f'{l}: {v} {unit}' for l, v in zip(labels, values))}")}"
        style="background:{BG_COLOR}; border-radius:8px; max-width:100%; height:auto;">
    {title_el}
    {"".join(grid_lines)}
    {y_axis}
    {x_axis}
    {"".join(y_labels)}
    {"".join(x_labels)}
    {line_path}
    {"".join(point_elements)}
</svg>'''


def render_chart(v: dict[str, Any]) -> str:
    """Render a chart visual (bar or line) as inline SVG."""
    kind = v.get("kind", "bar")
    title = _escape(v.get("title", "Chart"))
    caption = _escape(v.get("caption", ""))
    unit = _escape(v.get("unit", ""))
    labels = [str(l) for l in v.get("labels", [])]
    values = v.get("values", [])

    if not labels or not values or len(labels) != len(values):
        return ""

    num_values = [_to_number(v) for v in values]
    if any(v is None for v in num_values):
        return ""

    if kind == "bar":
        svg = _render_svg_bar_chart(labels, num_values, unit, v.get("title", "Chart"))
    elif kind == "line":
        svg = _render_svg_line_chart(labels, num_values, unit, v.get("title", "Chart"))
    else:
        return ""

    if not svg:
        return ""

    source_domain = ""
    src_url = v.get("source_url", "")
    if src_url:
        try:
            source_domain = re.sub(r"^https?://(?:www\.)?([^/]+).*", r"\1", src_url)
        except Exception:
            source_domain = src_url

    # Build figure with optional takeaway
    parts = [
        f'''<figure class="blog-visual">
{svg}
<figcaption>{caption} <span class="blog-visual-unit">{unit}</span>
    <span class="blog-visual-source">Source: {_escape(source_domain)}</span>
</figcaption>'''
    ]

    # Add takeaway if provided
    takeaway = v.get("takeaway", "").strip()
    if takeaway:
        parts.append(f'<p class="blog-takeaway">{_escape(takeaway)}</p>')

    parts.append("</figure>")
    return "".join(parts)


def render_visual(v: dict[str, Any], highlight_best: bool = False) -> str:
    """Dispatch to appropriate renderer based on visual kind."""
    kind = v.get("kind", "")
    if kind == "table":
        return render_table(v, highlight_best=highlight_best)
    elif kind in ("bar", "line"):
        return render_chart(v)
    return ""


def render_visual_html_fallback(v: dict[str, Any]) -> str:
    """HTML-only fallback for environments where SVG/style is stripped.

    Renders bar chart as a table with inline width bars (if style allowed)
    or plain table if style is stripped.
    """
    kind = v.get("kind", "")
    title = _escape(v.get("title", "Chart"))
    caption = _escape(v.get("caption", ""))
    unit = _escape(v.get("unit", ""))
    labels = [str(l) for l in v.get("labels", [])]
    values = v.get("values", [])

    if kind != "bar" or not labels or not values or len(labels) != len(values):
        return ""

    num_values = [_to_number(v) for v in values]
    if any(v is None for v in num_values):
        return ""

    max_val = max(num_values)
    if max_val == 0:
        max_val = 1

    parts = [
        '<figure class="blog-visual">',
        f'<table class="blog-chart-table">',
        "<thead><tr><th>Label</th><th>Value</th><th>Visual</th></tr></thead>",
        "<tbody>",
    ]

    for label, val in zip(labels, num_values):
        pct = (val / max_val) * 100 if max_val > 0 else 0
        bar_html = f'<div style="width:{pct:.1f}%; height:100%; background:{SERIES_COLOR}; border-radius:2px;"></div>'
        parts.append(
            f"<tr><td>{_escape(label)}</td><td>{_escape(str(val))} {unit}</td>"
            f'<td><div class="blog-bar-cell">{bar_html}</div></td></tr>'
        )

    parts.append("</tbody></table>")
    parts.append(f'<figcaption>{caption}</figcaption>')
    parts.append("</figure>")
    return "".join(parts)


def replace_visual_markers(
    content: str,
    visuals: list[dict[str, Any]],
    highlight_best: bool = False,
) -> tuple[str, list[str]]:
    """Replace <!--VISUAL:n--> markers with rendered visuals.

    Returns (updated_content, dropped_reasons) where dropped_reasons
    contains reasons for any visuals that couldn't be placed.
    """
    dropped = []
    used = [False] * len(visuals)

    for i, v in enumerate(visuals):
        marker = f"<!--VISUAL:{i}-->"
        idx = content.find(marker)
        if idx == -1:
            dropped.append(f"Visual {i} ({v.get('kind')}): missing marker")
            continue

        highlight = highlight_best and v.get("kind") == "table"
        rendered = render_visual(v, highlight_best=highlight)
        if not rendered:
            dropped.append(f"Visual {i} ({v.get('kind')}): render failed")
            continue

        # Check if preceding paragraph introduces the visual
        # Look backwards for a paragraph
        before = content[:idx]
        # Find last </p> before marker
        p_end = before.rfind("</p>")
        if p_end != -1:
            # Get text after last </p> (should be just whitespace/newlines)
            after_p = before[p_end + 4 :]
            if after_p.strip() == "":
                # Preceding paragraph exists, check if it introduces
                p_start = before.rfind("<p", 0, p_end)
                if p_start != -1:
                    p_content = before[p_start:p_end]
                    intro_text = re.sub(r"<[^>]+>", " ", p_content).strip()
                    intro_lower = intro_text.lower()
                    intro_keywords = ("table", "chart", "below", "compares", "shows", "illustrates", "depicts")
                    if not any(kw in intro_lower for kw in intro_keywords):
                        # Insert intro sentence if provided
                        intro_sentence = v.get("intro_sentence", "").strip()
                        if intro_sentence:
                            intro_html = f"<p>{_escape(intro_sentence)}</p>"
                            content = content[:idx] + intro_html + content[idx:]
                            idx += len(intro_html)

        # Replace marker with rendered visual
        content = content[:idx] + rendered + content[idx + len(marker) :]
        used[i] = True

    # Remove any leftover markers (including those beyond visual count)
    # Find all VISUAL markers in content and remove them
    marker_pattern = re.compile(r"<!--VISUAL:\d+-->")
    content = marker_pattern.sub("", content)

    return content, dropped


__all__ = [
    "render_table",
    "render_chart",
    "render_visual",
    "render_visual_html_fallback",
    "replace_visual_markers",
]