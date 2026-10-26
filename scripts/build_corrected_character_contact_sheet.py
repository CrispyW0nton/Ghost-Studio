"""Build a labeled contact sheet for the corrected enclave FBX cast."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def _font(size: int):
    for path in (Path("C:/Windows/Fonts/segoeui.ttf"), Path("C:/Windows/Fonts/arial.ttf")):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def main() -> int:
    args = _args()
    rows = json.loads(args.receipt.read_text(encoding="utf-8"))["characters"]
    columns = 5
    cell_w, cell_h = 320, 430
    canvas = Image.new(
        "RGB",
        (columns * cell_w, math.ceil(len(rows) / columns) * cell_h),
        (18, 18, 22),
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _font(19)
    detail_font = _font(13)
    for index, row in enumerate(rows):
        utc = str(row["utc_resref"])
        action = Path(row["animations"][0]["asset"])
        filename = f"normalized_actions__{_safe(action.stem)}__animated__front.png"
        source = args.images / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        image = Image.open(source).convert("RGB")
        image.thumbnail((300, 360), Image.Resampling.LANCZOS)
        x = (index % columns) * cell_w
        y = (index // columns) * cell_h
        draw.text((x + 10, y + 8), utc, fill=(245, 220, 170), font=title_font)
        draw.text(
            (x + 10, y + 32),
            f"{action.stem} | {len(row['mesh_names'])} mesh sections",
            fill=(160, 220, 190),
            font=detail_font,
        )
        canvas.paste(image, (x + (cell_w - image.width) // 2, y + 58))
        draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline=(70, 70, 82))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
