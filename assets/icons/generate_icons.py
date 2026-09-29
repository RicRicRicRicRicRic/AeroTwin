"""Generate the AeroTwin AI application icons (Windows .ico + PNG).

electron-builder requires real icon files in ``assets/icons/`` before it can
package installers, so this script draws the AeroTwin mark deterministically:
a dark slate app tile with a white building silhouette and a cyan crack
polyline running through it (post-disaster inspection motif).

Usage (from the repository root)::

    python assets/icons/generate_icons.py

Outputs:
* ``assets/icons/icon.png`` — 512x512 RGBA (Linux / macOS / documentation)
* ``assets/icons/icon.ico`` — multi-size Windows icon (16...256 px)

Requires Pillow (``pip install pillow``); no network access is used.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ICON_DIR: Path = Path(__file__).resolve().parent
CANVAS: int = 1024
BACKGROUND_TOP = (11, 18, 32, 255)
BACKGROUND_BOTTOM = (22, 35, 58, 255)
STRUCTURE = (236, 242, 250, 255)
STRUCTURE_SHADOW = (148, 163, 184, 255)
ACCENT = (34, 211, 238, 255)
ICO_SIZES: tuple[int, ...] = (16, 24, 32, 48, 64, 128, 256)
PNG_SIZE: int = 512


def _buildings(draw: ImageDraw.ImageDraw, scale: float) -> None:
    """Draw the inspected-building silhouette (three slabs of differing height)."""
    base_y = 0.80 * CANVAS
    specs = (
        (0.22, 0.34, 0.30, STRUCTURE_SHADOW),  # x, width, height, colour
        (0.38, 0.24, 0.46, STRUCTURE),
        (0.56, 0.22, 0.36, STRUCTURE),
    )
    for x_frac, width_frac, height_frac, colour in specs:
        left = x_frac * CANVAS
        right = (x_frac + width_frac) * CANVAS
        top = base_y - height_frac * CANVAS
        draw.rectangle((left, top, right, base_y), fill=colour)
        # Window rows: punched-out darker slats.
        row_height = max(8.0, 0.018 * CANVAS * scale)
        gap = max(14.0, 0.034 * CANVAS * scale)
        y = top + gap
        while y + row_height < base_y - gap:
            draw.rectangle((left + gap, y, right - gap, y + row_height), fill=(15, 23, 42, 255))
            y += row_height + gap


def _crack(draw: ImageDraw.ImageDraw, scale: float) -> None:
    """Cyan crack polyline crossing the buildings (the detection target)."""
    width = max(6, int(0.022 * CANVAS * scale))
    points = [
        (0.14 * CANVAS, 0.30 * CANVAS),
        (0.32 * CANVAS, 0.44 * CANVAS),
        (0.26 * CANVAS, 0.58 * CANVAS),
        (0.42 * CANVAS, 0.70 * CANVAS),
        (0.36 * CANVAS, 0.86 * CANVAS),
        (0.52 * CANVAS, 0.96 * CANVAS),
    ]
    draw.line(points, fill=ACCENT, width=width, joint="curve")


def _rounded_mask() -> Image.Image:
    mask = Image.new("L", (CANVAS, CANVAS), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, CANVAS - 1, CANVAS - 1), radius=int(0.22 * CANVAS), fill=255
    )
    return mask


def render_master() -> Image.Image:
    """Render the 1024 px master icon (background gradient + motif)."""
    background = Image.new("RGBA", (CANVAS, CANVAS), BACKGROUND_TOP)
    gradient = ImageDraw.Draw(background)
    for y in range(CANVAS):
        ratio = y / (CANVAS - 1)
        gradient.line(
            [(0, y), (CANVAS, y)],
            fill=tuple(
                round(BACKGROUND_TOP[i] + (BACKGROUND_BOTTOM[i] - BACKGROUND_TOP[i]) * ratio)
                for i in range(4)
            ),
        )

    motif = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(motif)
    _buildings(draw, scale=1.0)
    _crack(draw, scale=1.0)

    icon = Image.alpha_composite(background, motif)
    icon.putalpha(_rounded_mask())
    return icon


def main() -> None:
    master = render_master()
    ICON_DIR.mkdir(parents=True, exist_ok=True)

    png_path = ICON_DIR / "icon.png"
    master.resize((PNG_SIZE, PNG_SIZE), Image.LANCZOS).save(png_path, format="PNG")

    ico_path = ICON_DIR / "icon.ico"
    master.save(ico_path, format="ICO", sizes=[(size, size) for size in ICO_SIZES])

    print(f"[icons] wrote {png_path} ({PNG_SIZE}x{PNG_SIZE})")
    print(f"[icons] wrote {ico_path} (sizes: {', '.join(str(s) for s in ICO_SIZES)})")


if __name__ == "__main__":
    main()
