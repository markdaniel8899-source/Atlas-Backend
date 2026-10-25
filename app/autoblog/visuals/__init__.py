"""Render structured visuals (tables, charts) for AutoBlog v2 - fact-sheet grounded."""

from __future__ import annotations

import html
import re
from typing import Any

# ─── Theme ────────────────────────────────────────────────────────────────────

AXIS_COLOR = "#94a3b8"
SERIES_COLOR = "#38bdf8"
GRID_COLOR = "#1e293b"
BG_COLOR = "#0f172a"
TEXT_COLOR = "#e2e8f0"
HIGHLIGHT_COLOR = "#fbbf24"


def _escape(text: str) -> str:
    return html.escape(str(text), quote=True)


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


# ─── Table ────────────────────────────────────────────────────────────────────

def render_table(vis: dict[str, Any]) -> str:
    """Render a table visual as HTML figure. All data must come from fact sheet."""
    title = _escape(vis.get("title", "Table"))
    caption = _escape(vis.get("caption", ""))
    columns = [str(c) for c in vis.get("columns", [])]
    rows = vis.get("rows", [])
    source_url = _escape(vis.get("source_url", ""))
    intro = _escape(vis.get("intro_sentence", ""))
    takeaway = _escape(vis.get("takeaway", ""))

    if not columns or not rows:
        return ""

    # Detect numeric columns for highlighting
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

    best_per_col: dict[int, float] = {}
    for i, is_num in enumerate(numeric_cols):
        if is_num and col_values[i]:
            best_per_col[i] = max(col_values[i])

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

    parts.extend([
        "</tbody></table>",
        f'<figcaption>{caption}</figcaption>',
    ])
    if source_url:
        parts.append(
            f'<div class="blog-visual-source">Source: '
            f'<a href="{source_url}" rel="noopener" target="_blank">{source_url}</a></div>'
        )

    if intro:
        parts.append(f'<p class="blog-visual-intro">{intro}</p>')
    if takeaway:
        parts.append(f'<p class="blog-takeaway">{takeaway}</p>')

    parts.append("</figure>")
    return "".join(parts)


# ─── Bar Chart ────────────────────────────────────────────────────────────────

def _render_svg_bar_chart(
    labels: list[str],
    values: list[float],
    unit: str,
    title: str,
    width: int = 640,
    height: int = 320,
) -> str:
    if not labels or not values or len(labels) != len(values):
        return ""

    max_val = max(values) if values else 1
    min_val = min(values) if values else 0
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

    bars = []
    for i, (label, val) in enumerate(zip(labels, values)):
        y = padding_top + i * (chart_height / len(labels)) + (chart_height / len(labels)) * 0.1
        bar_height = (chart_height / len(labels)) * 0.8

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

        label_x = x_start + bar_width + 5 if val >= 0 else x_start - 5
        bars.append(
            f'<text x="{label_x:.1f}" y="{y + bar_height / 2 + 4:.1f}" '
            f'font-size="11" fill="{TEXT_COLOR}" text-anchor="start" dominant-baseline="middle">'
            f'{_escape(str(val))} {_escape(unit)}</text>'
        )

    grid_lines = []
    for g in range(5):
        x = padding_left + (g / 4) * chart_width
        grid_lines.append(
            f'<line x1="{x:.1f}" y1="{padding_top:.1f}" '
            f'x2="{x:.1f}" y2="{padding_top + chart_height:.1f}" '
            f'stroke="{GRID_COLOR}" stroke-width="0.5" stroke-dasharray="2,2"/>'
        )

    y_labels = []
    for i, label in enumerate(labels):
        y = padding_top + i * (chart_height / len(labels)) + (chart_height / len(labels)) / 2
        y_labels.append(
            f'<text x="{padding_left - 10:.1f}" y="{y:.1f}" '
            f'font-size="11" fill="{TEXT_COLOR}" text-anchor="end" dominant-baseline="middle">'
            f'{_escape(label)}</text>'
        )

    x_axis = (
        f'<line x1="{padding_left}" y1="{padding_top + chart_height}" '
        f'x2="{width - padding_right}" y2="{padding_top + chart_height}" '
        f'stroke="{AXIS_COLOR}" stroke-width="1"/>'
    )

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


# ─── Line Chart ───────────────────────────────────────────────────────────────

def _render_svg_line_chart(
    labels: list[str],
    values: list[float],
    unit: str,
    title: str,
    width: int = 640,
    height: int = 320,
) -> str:
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

    grid_lines = []
    for g in range(5):
        y = padding_top + (g / 4) * chart_height
        grid_lines.append(
            f'<line x1="{padding_left}" y1="{y:.1f}" '
            f'x2="{width - padding_right}" y2="{y:.1f}" '
            f'stroke="{GRID_COLOR}" stroke-width="0.5" stroke-dasharray="2,2"/>'
        )

    y_labels = []
    for g in range(5):
        y = padding_top + (g / 4) * chart_height
        val = scale_max - (g / 4) * value_range
        y_labels.append(
            f'<text x="{padding_left - 10}" y="{y:.1f}" '
            f'font-size="11" fill="{TEXT_COLOR}" text-anchor="end" dominant-baseline="middle">'
            f'{_escape(f"{val:.1f}")}</text>'
        )

    x_labels = []
    for i, label in enumerate(labels):
        x = padding_left + (i / (len(labels) - 1)) * chart_width if len(labels) > 1 else padding_left + chart_width / 2
        x_labels.append(
            f'<text x="{x:.1f}" y="{padding_top + chart_height + 20:.1f}" '
            f'font-size="11" fill="{TEXT_COLOR}" text-anchor="middle" dominant-baseline="middle">'
            f'{_escape(label)}</text>'
        )

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

    polyline = f'<polyline points="{" ".join(points)}" fill="none" stroke="{SERIES_COLOR}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
    dots = []
    for i, (x_str, y_str) in enumerate((p.split(",") for p in points)):
        dots.append(
            f'<circle cx="{x_str}" cy="{y_str}" r="3" fill="{SERIES_COLOR}"/>'
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
    {"".join(y_labels)}
    {"".join(x_labels)}
    {x_axis}
    {y_axis}
    {polyline}
    {"".join(dots)}
</svg>'''


# ─── Dispatch ─────────────────────────────────────────────────────────────────

def render_visual(vis: dict[str, Any]) -> str:
    """Dispatch to renderer. Returns empty string if data invalid."""
    kind = vis.get("kind")
    if kind == "table":
        return render_table(vis)
    elif kind == "bar":
        labels = vis.get("labels", [])
        values = [_to_number(v) for v in vis.get("values", [])]
        values = [v for v in values if v is not None]
        if len(labels) != len(values):
            return ""
        return _render_svg_bar_chart(labels, values, vis.get("unit", ""), vis.get("title", ""))
    elif kind == "line":
        labels = vis.get("labels", [])
        values = [_to_number(v) for v in vis.get("values", [])]
        values = [v for v in values if v is not None]
        if len(labels) != len(values):
            return ""
        return _render_svg_line_chart(labels, values, vis.get("unit", ""), vis.get("title", ""))
    return ""


def replace_visual_markers(html_text: str, visuals: list[dict[str, Any]]) -> str:
    """Replace <!--VISUAL:n--> markers with rendered visuals."""
    for i, visual in enumerate(visuals):
        rendered = render_visual(visual)
        if rendered:
            marker = f"<!--VISUAL:{i}-->"
            html_text = html_text.replace(marker, rendered, 1)
    return html_text


# ─── Exports ──────────────────────────────────────────────────────────────────

__all__ = [
    "render_visual",
    "replace_visual_markers",
    "render_table",
]