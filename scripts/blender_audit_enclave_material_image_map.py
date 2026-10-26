"""Record authored FBX material-slot to KOTOR base-color bitmap mappings."""

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
    parser.add_argument("--normalization-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def _reset() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.armatures,
        bpy.data.actions,
    ):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def _key(value: str) -> str:
    value = re.sub(r"[._]\d{3}$", "", str(value), flags=re.IGNORECASE)
    value = re.sub(r"_BaseColor$", "", value, flags=re.IGNORECASE)
    return value.casefold()


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
    records = []
    for row in normalized["characters"]:
        _reset()
        source = Path(row["source_glb"])
        bpy.ops.import_scene.gltf(filepath=str(source), import_pack_images=True)
        mapping = {}
        details = []
        for material in sorted(
            bpy.data.materials, key=lambda item: item.name.casefold()
        ):
            image = _base_color_image(material)
            if image is None:
                continue
            material_key = _key(material.name)
            image_key = _key(Path(image.name).stem)
            previous = mapping.get(material_key)
            if previous is not None and previous != image_key:
                raise RuntimeError(
                    f"{row['utc_resref']} material {material_key} maps to both "
                    f"{previous} and {image_key}"
                )
            mapping[material_key] = image_key
            details.append(
                {
                    "material": material.name,
                    "material_key": material_key,
                    "image": image.name,
                    "image_key": image_key,
                }
            )
        records.append(
            {
                "utc_resref": row["utc_resref"],
                "source_glb": str(source),
                "mapping": mapping,
                "details": details,
            }
        )
        print(f"[{len(records)}] {row['utc_resref']}", flush=True)
    payload = {
        "schema": "ghoststudio.enclave-material-image-map.v1",
        "character_count": len(records),
        "characters": records,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": str(args.receipt), "characters": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
