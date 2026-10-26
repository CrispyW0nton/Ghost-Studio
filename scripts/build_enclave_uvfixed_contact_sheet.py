"""Build a rest/walk visual acceptance sheet for the UV-fixed enclave cast."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def font(size: int):
    for path in (Path("C:/Windows/Fonts/segoeui.ttf"), Path("C:/Windows/Fonts/arial.ttf")):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--rest", type=Path, required=True)
    parser.add_argument("--walk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads-output", type=Path)
    parser.add_argument("--normalized-rest", type=Path)
    parser.add_argument("--head-compare-output", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.receipt.read_text(encoding="utf-8"))["characters"]
    columns = 3
    cell_w, cell_h = 720, 470
    canvas = Image.new("RGB", (columns * cell_w, math.ceil(len(rows) / columns) * cell_h), (18, 18, 22))
    draw = ImageDraw.Draw(canvas)
    title = font(22)
    detail = font(15)
    for index, row in enumerate(rows):
        utc = str(row["utc_resref"])
        stem = f"glb_source__SK_{safe(utc)}_BAS"
        sources = (
            args.rest / f"{stem}__rest__front.png",
            args.walk / f"{stem}__animated__front.png",
        )
        images = []
        for source in sources:
            if not source.is_file():
                raise FileNotFoundError(source)
            image = Image.open(source).convert("RGB")
            image.thumbnail((340, 390), Image.Resampling.LANCZOS)
            images.append(image)
        x = (index % columns) * cell_w
        y = (index // columns) * cell_h
        draw.text((x + 12, y + 8), utc, fill=(245, 220, 170), font=title)
        draw.text((x + 12, y + 38), "REST / WALK", fill=(160, 220, 190), font=detail)
        for offset, image in zip((8, 370), images):
            canvas.paste(image, (x + offset + (340 - image.width) // 2, y + 65))
        draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline=(70, 70, 82))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output)
    if args.heads_output:
        head_columns = 4
        head_cell_w, head_cell_h = 360, 360
        head_canvas = Image.new(
            "RGB",
            (
                head_columns * head_cell_w,
                math.ceil(len(rows) / head_columns) * head_cell_h,
            ),
            (18, 18, 22),
        )
        head_draw = ImageDraw.Draw(head_canvas)
        for index, row in enumerate(rows):
            utc = str(row["utc_resref"])
            stem = f"glb_source__SK_{safe(utc)}_BAS"
            source = args.rest / f"{stem}__rest__front.png"
            image = Image.open(source).convert("RGB")
            # The deterministic front gate is 640x800.  This crop keeps the
            # complete head, neck seam, and both shoulders so attachment
            # centering can be judged instead of hidden by a face-only crop.
            head = image.crop((130, 0, 510, 300))
            head.thumbnail((340, 290), Image.Resampling.LANCZOS)
            x = (index % head_columns) * head_cell_w
            y = (index // head_columns) * head_cell_h
            head_draw.text((x + 12, y + 8), utc, fill=(245, 220, 170), font=title)
            head_canvas.paste(
                head,
                (x + (head_cell_w - head.width) // 2, y + 48),
            )
            head_draw.rectangle(
                (x, y, x + head_cell_w - 1, y + head_cell_h - 1),
                outline=(70, 70, 82),
            )
        args.heads_output.parent.mkdir(parents=True, exist_ok=True)
        head_canvas.save(args.heads_output)
        print(args.heads_output)
    if args.head_compare_output:
        if args.normalized_rest is None:
            parser.error("--head-compare-output requires --normalized-rest")
        compare_columns = 3
        compare_cell_w, compare_cell_h = 720, 360
        compare_canvas = Image.new(
            "RGB",
            (
                compare_columns * compare_cell_w,
                math.ceil(len(rows) / compare_columns) * compare_cell_h,
            ),
            (18, 18, 22),
        )
        compare_draw = ImageDraw.Draw(compare_canvas)
        for index, row in enumerate(rows):
            utc = str(row["utc_resref"])
            glb_stem = f"glb_source__SK_{safe(utc)}_BAS"
            fbx_stem = f"normalized__SK_{safe(utc)}_BAS_Unreal"
            sources = (
                args.rest / f"{glb_stem}__rest__front.png",
                args.normalized_rest / f"{fbx_stem}__rest__front.png",
            )
            heads = []
            for source in sources:
                if not source.is_file():
                    raise FileNotFoundError(source)
                image = Image.open(source).convert("RGB")
                head = image.crop((130, 0, 510, 300))
                head.thumbnail((340, 290), Image.Resampling.LANCZOS)
                heads.append(head)
            x = (index % compare_columns) * compare_cell_w
            y = (index // compare_columns) * compare_cell_h
            compare_draw.text((x + 12, y + 8), utc, fill=(245, 220, 170), font=title)
            compare_draw.text(
                (x + 12, y + 38),
                "SOURCE GLB / NORMALIZED FBX",
                fill=(160, 220, 190),
                font=detail,
            )
            for offset, head in zip((8, 370), heads):
                compare_canvas.paste(head, (x + offset, y + 64))
            compare_draw.rectangle(
                (x, y, x + compare_cell_w - 1, y + compare_cell_h - 1),
                outline=(70, 70, 82),
            )
        args.head_compare_output.parent.mkdir(parents=True, exist_ok=True)
        compare_canvas.save(args.head_compare_output)
        print(args.head_compare_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
