from __future__ import annotations

import html
import math

from .types import VisualDataset

ACCENT = "#cf9eff"
SECONDARY = "#9db4ff"
BACKGROUND = "#030305"
TEXT = "#ffffff"
MUTED = "#a8a8b3"
WIDTH = 800
HEIGHT = 450
PLOT_LEFT = 74
PLOT_RIGHT = 32
PLOT_TOP = 78
PLOT_BOTTOM = 72


class ChartError(ValueError):
    pass


def render_chart(
    data: VisualDataset,
    *,
    alt: str,
    caption: str,
) -> str:
    if not data.title.strip():
        raise ChartError("Chart data needs a title.")
    if not 2 <= len(data.labels) <= 8:
        raise ChartError("Charts need 2-8 labels.")
    if len(data.labels) != len(data.values):
        raise ChartError("Chart labels and values must have equal length.")
    if not all(math.isfinite(value) for value in data.values):
        raise ChartError("Chart values must be finite numbers.")
    if not data.source_url:
        raise ChartError("Chart data needs a source URL.")
    chart_type = (data.chart_type or "bar").lower()
    if chart_type not in ("bar", "line"):
        raise ChartError("Only bar and line charts are supported.")

    if chart_type == "line":
        plot = _line_plot(data)
    else:
        plot = _bar_plot(data)

    source = data.source_label or data.source_url
    return (
        "<figure>"
        f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        f'aria-label="{_attr(alt)}" xmlns="http://www.w3.org/2000/svg">'
        f"<title>{_text(data.title)}</title>"
        f"<desc>{_text(alt or data.title)}</desc>"
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BACKGROUND}"/>'
        f'<text x="{PLOT_LEFT}" y="42" fill="{TEXT}" font-size="23" '
        'font-family="Arial, sans-serif" font-weight="700">'
        f"{_text(data.title)}</text>"
        + plot
        + f'<text x="{PLOT_LEFT}" y="{HEIGHT - 24}" fill="{MUTED}" '
        'font-size="12" font-family="Arial, sans-serif">'
        f"Source: {_text(source)}</text>"
        "</svg>"
        f"<figcaption>{_text(caption)}</figcaption>"
        "</figure>"
    )


def _bar_plot(data: VisualDataset) -> str:
    plot_width = WIDTH - PLOT_LEFT - PLOT_RIGHT
    plot_height = HEIGHT - PLOT_TOP - PLOT_BOTTOM
    baseline = PLOT_TOP + plot_height
    maximum = max(max(data.values), 0.0) or 1.0
    slot = plot_width / len(data.values)
    bar_width = min(72.0, slot * 0.52)
    parts = [
        f'<line x1="{PLOT_LEFT}" y1="{baseline}" x2="{WIDTH - PLOT_RIGHT}" '
        f'y2="{baseline}" stroke="{MUTED}" stroke-width="1"/>'
    ]
    for index, (label, value) in enumerate(zip(data.labels, data.values)):
        height = max(0.0, value / maximum * plot_height)
        x = PLOT_LEFT + index * slot + (slot - bar_width) / 2
        y = baseline - height
        fill = ACCENT if index % 2 == 0 else SECONDARY
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
            f'height="{height:.2f}" rx="6" fill="{fill}"/>'
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{max(y - 10, PLOT_TOP + 16):.2f}" '
            f'fill="{TEXT}" font-size="13" text-anchor="middle" '
            'font-family="Arial, sans-serif">'
            f"{_text(_value(value, data.unit))}</text>"
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{baseline + 24}" fill="{MUTED}" '
            f'font-size="12" text-anchor="middle" font-family="Arial, sans-serif">'
            f"{_text(_short(label, 18))}</text>"
        )
    return "".join(parts)


def _line_plot(data: VisualDataset) -> str:
    plot_width = WIDTH - PLOT_LEFT - PLOT_RIGHT
    plot_height = HEIGHT - PLOT_TOP - PLOT_BOTTOM
    baseline = PLOT_TOP + plot_height
    maximum = max(max(data.values), 0.0) or 1.0
    step = plot_width / max(1, len(data.values) - 1)
    points = []
    for index, value in enumerate(data.values):
        x = PLOT_LEFT + index * step
        y = baseline - value / maximum * plot_height
        points.append((x, y))
    point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    parts = [
        f'<line x1="{PLOT_LEFT}" y1="{baseline}" x2="{WIDTH - PLOT_RIGHT}" '
        f'y2="{baseline}" stroke="{MUTED}" stroke-width="1"/>',
        f'<polyline points="{point_text}" fill="none" stroke="{ACCENT}" '
        'stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>',
    ]
    for index, (point, label, value) in enumerate(
        zip(points, data.labels, data.values)
    ):
        x, y = point
        fill = ACCENT if index % 2 == 0 else SECONDARY
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{fill}"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{max(y - 14, PLOT_TOP + 10):.2f}" '
            f'fill="{TEXT}" font-size="13" text-anchor="middle" '
            'font-family="Arial, sans-serif">'
            f"{_text(_value(value, data.unit))}</text>"
        )
        parts.append(
            f'<text x="{x:.2f}" y="{baseline + 24}" fill="{MUTED}" '
            f'font-size="12" text-anchor="middle" font-family="Arial, sans-serif">'
            f"{_text(_short(label, 18))}</text>"
        )
    return "".join(parts)


def _value(value: float, unit: str) -> str:
    number = f"{value:g}"
    if unit == "%":
        return f"{number}%"
    if unit in ("$", "€", "£"):
        return f"{unit}{number}"
    return f"{number} {unit}".strip()


def _short(value: str, limit: int) -> str:
    text = (value or "").strip()
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _text(value: str) -> str:
    return html.escape(value or "", quote=False)


def _attr(value: str) -> str:
    return html.escape(value or "", quote=True)
