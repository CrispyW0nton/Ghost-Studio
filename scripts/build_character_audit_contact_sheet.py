"""Build side-by-side GLB/FBX visual-audit sheets for enclave characters."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb-receipt", type=Path, required=True)
    parser.add_argument("--fbx-receipt", type=Path, required=True)
    parser.add_argument("--glb-images", type=Path, required=True)
    parser.add_argument("--fbx-images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state", choices=("animated", "rest"), default="animated")
    return parser.parse_args()


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def _font(size: int):
    candidates = (
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def main() -> int:
    args = _args()
    glb = json.loads(args.glb_receipt.read_text(encoding="utf-8"))["exports"]
    fbx = json.loads(args.fbx_receipt.read_text(encoding="utf-8"))["characters"]
    fbx_by_name = {str(row["utc_resref"]).casefold(): row for row in fbx}
    rows = [row for row in glb if str(row["utc_resref"]).casefold() in fbx_by_name]
    cell_w, cell_h = 640, 440
    columns = 4
    font = _font(22)
    small = _font(16)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for view in ("front", "left"):
        canvas = Image.new(
            "RGB",
            (columns * cell_w, math.ceil(len(rows) / columns) * cell_h),
            (20, 20, 24),
        )
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(rows):
            name = str(row["utc_resref"])
            character = fbx_by_name[name.casefold()]
            if args.state == "animated":
                action = Path(character["animations"][0]["asset"])
                glb_name = f"glb_source__SK_{_safe(name)}_BAS__animated__{view}.png"
                fbx_name = (
                    f"normalized_actions__{_safe(action.stem)}__animated__{view}.png"
                )
            else:
                glb_name = f"glb_source__SK_{_safe(name)}_BAS__rest__{view}.png"
                fbx_name = (
                    f"normalized__SK_{_safe(name)}_BAS_Unreal__rest__{view}.png"
                )
            images = []
            for path in (args.glb_images / glb_name, args.fbx_images / fbx_name):
                image = Image.open(path).convert("RGB")
                image.thumbnail((310, 388), Image.Resampling.LANCZOS)
                images.append(image)
            column = index % columns
            line = index // columns
            x, y = column * cell_w, line * cell_h
            draw.text((x + 8, y + 6), name, font=font, fill=(245, 220, 170))
            draw.text((x + 8, y + 32), "GLB source", font=small, fill=(150, 210, 255))
            draw.text((x + 328, y + 32), "FBX normalized", font=small, fill=(160, 245, 185))
            for offset, image in zip((8, 328), images):
                canvas.paste(image, (x + offset, y + 52))
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline=(70, 70, 80))
        output = args.output_dir / (
            f"enclave_character_glb_vs_fbx_{args.state}_{view}_v2.png"
        )
        canvas.save(output)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
