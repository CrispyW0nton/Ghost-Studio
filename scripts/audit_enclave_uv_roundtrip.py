"""Compare UV loop signatures before and after the enclave GLB -> FBX normalization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def _args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--fbx", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def _reset() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _capture() -> dict:
    result = {}
    for obj in sorted((item for item in bpy.context.scene.objects if item.type == "MESH"), key=lambda item: item.name.casefold()):
        layer = obj.data.uv_layers.active
        if layer is None:
            result[obj.name] = {"loops": 0, "materials": [slot.name for slot in obj.data.materials]}
            continue
        values = [(float(item.uv.x), float(item.uv.y)) for item in layer.data]
        result[obj.name] = {
            "loops": len(values),
            "sum_u": round(sum(u for u, _ in values), 6),
            "sum_v": round(sum(v for _, v in values), 6),
            "sum_u2": round(sum(u * u for u, _ in values), 6),
            "sum_v2": round(sum(v * v for _, v in values), 6),
            "min_u": round(min((u for u, _ in values), default=0.0), 6),
            "max_u": round(max((u for u, _ in values), default=0.0), 6),
            "min_v": round(min((v for _, v in values), default=0.0), 6),
            "max_v": round(max((v for _, v in values), default=0.0), 6),
            "materials": [slot.name for slot in obj.data.materials],
        }
    return result


def main() -> int:
    args = _args()
    _reset()
    bpy.ops.import_scene.gltf(filepath=str(args.glb), import_pack_images=True)
    glb = _capture()
    _reset()
    bpy.ops.import_scene.fbx(filepath=str(args.fbx), use_anim=False)
    fbx = _capture()
    rows = []
    for name in sorted(set(glb) | set(fbx), key=str.casefold):
        before = glb.get(name, {})
        after = fbx.get(name, {})
        loops = int(before.get("loops", 0))
        excluded_helper = bool(
            before
            and not after
            and not list(before.get("materials", []) or [])
        )
        preserved = excluded_helper or bool(
            loops
            and loops == int(after.get("loops", -1))
            and abs(float(before.get("sum_u", 0.0)) - float(after.get("sum_u", 0.0))) < 0.001
            and abs(float(before.get("sum_v", 0.0)) - float(after.get("sum_v", 0.0))) < 0.001
        )
        v_flipped = bool(
            loops
            and loops == int(after.get("loops", -1))
            and abs((loops - float(before.get("sum_v", 0.0))) - float(after.get("sum_v", 0.0))) < 0.001
        )
        rows.append({
            "mesh": name,
            "uv_preserved": preserved,
            "uv_v_flipped": v_flipped,
            "excluded_unmaterialed_source_helper": excluded_helper,
            "glb": before,
            "fbx": after,
        })
    payload = {
        "schema": "ghoststudio.enclave-uv-roundtrip.v1",
        "all_uv_preserved": all(row["uv_preserved"] for row in rows),
        "any_uv_v_flipped": any(row["uv_v_flipped"] for row in rows),
        "meshes": rows,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": str(args.receipt), "all_uv_preserved": payload["all_uv_preserved"], "any_uv_v_flipped": payload["any_uv_v_flipped"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
