from __future__ import annotations

import io
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ACCENT = (207, 158, 255)
SECONDARY = (157, 180, 255)
TEXT = (255, 255, 255)
MUTED = (176, 176, 188)
BACKGROUND = (3, 3, 5)


def generate_branded_cover(title: str, category: str) -> bytes:
    return _generate(title, category, (1200, 630), title_size=64)


def generate_branded_content(title: str, category: str) -> bytes:
    return _generate(title, category, (1200, 675), title_size=58)


def _generate(
    title: str,
    category: str,
    size: tuple[int, int],
    *,
    title_size: int,
) -> bytes:
    width, height = size
    image = Image.new("RGB", size, BACKGROUND)
    draw = ImageDraw.Draw(image)
    _draw_gradient(draw, width, height)
    _draw_accents(draw, width, height)

    title_font = _font(title_size, bold=True)
    label_font = _font(20, bold=True)
    mark_font = _font(24, bold=True)

    draw.text((72, 58), "ATLAS", font=mark_font, fill=TEXT)
    draw.line((72, 98, 168, 98), fill=ACCENT, width=4)
    label = (category or "learning").upper()
    label_width = draw.textlength(label, font=label_font)
    draw.text(
        (width - 72 - label_width, 62),
        label,
        font=label_font,
        fill=SECONDARY,
    )

    max_width = width - 144
    lines = _wrap(title or "ATLAS learning guide", draw, title_font, max_width)
    line_height = int(title_size * 1.22)
    block_height = line_height * len(lines)
    start_y = max(170, int((height - block_height) / 2) - 10)
    for index, line in enumerate(lines):
        draw.text(
            (72, start_y + index * line_height),
            line,
            font=title_font,
            fill=TEXT,
        )

    draw.text(
        (72, height - 66),
        "atlas.app",
        font=_font(18, bold=False),
        fill=MUTED,
    )
    draw.line(
        (width - 240, height - 58, width - 72, height - 58),
        fill=ACCENT,
        width=3,
    )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _draw_gradient(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
) -> None:
    for y in range(height):
        ratio = y / max(1, height - 1)
        red = int(BACKGROUND[0] + (20 - BACKGROUND[0]) * ratio)
        green = int(BACKGROUND[1] + (10 - BACKGROUND[1]) * ratio)
        blue = int(BACKGROUND[2] + (38 - BACKGROUND[2]) * ratio)
        draw.line((0, y, width, y), fill=(red, green, blue))


def _draw_accents(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
) -> None:
    draw.ellipse(
        (width - 360, -170, width + 120, 310),
        fill=(36, 20, 58),
    )
    draw.ellipse(
        (width - 260, -70, width + 30, 220),
        fill=(48, 28, 78),
    )
    draw.polygon(
        [
            (width - 90, height),
            (width - 320, height),
            (width - 170, height - 190),
        ],
        fill=(26, 18, 44),
    )
    draw.line(
        (0, height - 4, width, height - 4),
        fill=ACCENT,
        width=4,
    )


def _wrap(
    text: str,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and draw.textlength(candidate, font=font) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    if not lines:
        lines = ["ATLAS learning guide"]
    if len(lines) > 4:
        lines = lines[:4]
        lines[-1] = f"{lines[-1].rstrip(' .')}…"
    return lines


def _font(size: int, *, bold: bool) -> ImageFont.ImageFont:
    candidates = []
    env_path = os.getenv("BLOG_BRAND_FONT", "").strip()
    if env_path:
        candidates.append(Path(env_path))
    root = Path(__file__).resolve().parents[2]
    bundled = root / "data" / "fonts"
    candidates.extend(
        [
            bundled / ("Inter-Bold.ttf" if bold else "Inter-Regular.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
            Path(
                "/usr/share/fonts/truetype/dejavu/"
                + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
            ),
        ]
    )
    for path in candidates:
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default()
