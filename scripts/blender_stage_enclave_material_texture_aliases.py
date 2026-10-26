"""Stage FBX-material-name aliases for normalized enclave base-color images."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import bpy


def _args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--normalization-receipt",
        type=Path,
        required=True,
        help="Normalization receipt used to remove only stale generated aliases.",
    )
    return parser.parse_args(argv)


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "material"


def _reset() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    # Mesh datablocks retain references to their materials after the objects
    # are removed. Purge meshes first or materials from the previous character
    # contaminate the next character's alias set.
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def _upstream_image_node(socket, visited=None):
    visited = set() if visited is None else visited
    if socket is None or not socket.is_linked:
        return None
    for link in socket.links:
        node = link.from_node
        if node.as_pointer() in visited:
            continue
        visited.add(node.as_pointer())
        if node.type == "TEX_IMAGE" and getattr(node, "image", None) is not None:
            return node
        for input_socket in node.inputs:
            image_node = _upstream_image_node(input_socket, visited)
            if image_node is not None:
                return image_node
    return None


def _base_color_image(material):
    if not material.use_nodes or material.node_tree is None:
        return None
    for node in material.node_tree.nodes:
        if node.type != "BSDF_PRINCIPLED":
            continue
        socket = node.inputs.get("Base Color")
        image_node = _upstream_image_node(socket)
        if image_node is not None:
            return image_node.image
    return None


def main() -> int:
    args = _args()
    normalized = json.loads(args.normalization_receipt.read_text(encoding="utf-8"))
    authored_textures = {
        str(row["utc_resref"]).casefold(): {
            Path(item).name.casefold() for item in row.get("texture_files", [])
        }
        for row in normalized["characters"]
    }
    records = []
    for base_fbx in sorted(args.export_root.glob("*/normalized/*.fbx")):
        _reset()
        bpy.ops.import_scene.fbx(
            filepath=str(base_fbx), automatic_bone_orientation=False
        )
        texture_root = base_fbx.parent.parent / "textures"
        texture_root.mkdir(parents=True, exist_ok=True)
        character_key = base_fbx.parent.parent.name.casefold()
        allowed = authored_textures.get(character_key)
        if allowed is None:
            raise RuntimeError(
                f"{base_fbx.parent.parent.name} missing from normalization receipt"
            )
        removed_stale_aliases = []
        for item in texture_root.glob("*.png"):
            if item.name.casefold() in allowed:
                continue
            item.unlink()
            removed_stale_aliases.append(str(item))
        aliases = []
        for material in sorted(
            bpy.data.materials, key=lambda item: item.name.casefold()
        ):
            image = _base_color_image(material)
            if image is None:
                continue
            material_key = re.sub(
                r"[._]\d{3}$", "", material.name, flags=re.IGNORECASE
            )
            destination = texture_root / f"{_safe(material_key)}.png"
            if not destination.exists():
                image.filepath_raw = str(destination)
                image.file_format = "PNG"
                image.save()
            aliases.append(
                {
                    "material": material.name,
                    "image": image.name,
                    "alias": str(destination),
                }
            )
        records.append(
            {
                "character": base_fbx.parent.parent.name,
                "base_fbx": str(base_fbx),
                "removed_stale_aliases": removed_stale_aliases,
                "aliases": aliases,
            }
        )
        print(f"[{len(records)}] {base_fbx.parent.parent.name}", flush=True)
    payload = {
        "schema": "ghoststudio.enclave-material-texture-aliases.v1",
        "character_count": len(records),
        "characters": records,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": str(args.receipt), "characters": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
