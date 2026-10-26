"""Bounded native snapshot IO and explicit static diffuse conversion recipes.

No source editing, inference service, silent material substitution or collision
guessing. Policies are creator data, independent from source and executable code.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import tempfile

from PIL import Image
from src.core.modules.unreal_import_state import revision
from src.io.unreal_level_bundle import MAX_JSON_BYTES, _json_object, read_bundle_file
from src.math.unreal_native_surface import linear_rgb_to_srgb8, normalize_native_section


def load_native_snapshot(path: str | Path) -> dict:
    path = Path(path).resolve()
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Native snapshot manifest exceeds size limit.")
    raw = path.read_bytes()
    data = _json_object(raw)
    if data.get("schema") != "ghoststudio.unreal-snapshot/v1" or data.get("exporter_version") != "0.1.0":
        raise ValueError("Unsupported native snapshot/exporter version.")
    if data.get("engine_version") != "5.8.2-56702186+++UE5+Release-5.8":
        raise ValueError("Unqualified native engine version.")
    if data.get("dirty_packages") or data.get("source_preserved") is not True:
        raise ValueError("Snapshot lacks saved, preserved source evidence.")
    if len(data.get("components", [])) > 10000:
        raise ValueError("Snapshot exceeds component budget.")
    # Validate every referenced sidecar before conversion, including excluded
    # geometry; a stale or partial export is never an accepted snapshot.
    read_bundle_file(path.parent, data["source_map"])
    data["resolved_meshes"] = {name: _json_object(read_bundle_file(path.parent, ref)) for name, ref in data["meshes"].items()}
    data["texture_bytes"] = {name: read_bundle_file(path.parent, ref) for name, ref in data["textures"].items()}
    data["snapshot_sha256"] = hashlib.sha256(raw).hexdigest()
    return data


def material_revision(snapshot: dict, material_id: str) -> str:
    """Invalidate approved bakes on material/texture dependency changes.

Conservatively includes the material dependency set, excluding map and mesh
packages so a prop transform/mesh edit does not require rebaking all textures.
"""
    geometry = {key.split(".")[0] for key in snapshot["meshes"]} | {snapshot["map_package"]}
    return revision({"material": snapshot["materials"][material_id],
                     "dependencies": [d for d in snapshot["dependencies"] if d["package"] not in geometry]})


def _diffuse(snapshot: dict, key: str, recipe: dict, policy_root: Path):
    material = snapshot["materials"][key]
    if recipe.get("recipe") == "approved-texture-v1":
        if recipe.get("source_revision") != material_revision(snapshot, key):
            raise ValueError(f"Approved material bake is stale: {key}")
        if not recipe.get("conversion_note"):
            raise ValueError(f"Reviewed material conversion needs its loss/intent note: {key}")
        content = read_bundle_file(policy_root, recipe["diffuse_file"])
        with Image.open(BytesIO(content)) as check:
            if check.format != "TGA" or max(check.size) > 4096:
                raise ValueError(f"Approved material must be a bounded TGA: {key}")
            check.load()
        return content, recipe["blend_mode"], recipe.get("uv_scale", [1, 1]), bool(recipe.get("unlit")), recipe["conversion_note"]
    if recipe.get("recipe", "simple-diffuse-v1") != "simple-diffuse-v1":
        raise ValueError(f"Unsupported material recipe: {key}")
    graph = material.get("base_color")
    if material.get("base_color_channel", "") not in {"", "RGB"}:
        raise ValueError(f"Diffuse output channel needs an explicit bake: {key}")
    if not graph or material.get("world_position_offset") or material.get("emissive"):
        raise ValueError(f"Material needs an explicit native bake/reviewed conversion: {key}")
    scale = [1.0, 1.0]
    texture_key = None
    if graph["class"] == "MaterialExpressionTextureSample":
        texture_key = graph["texture"]
        with Image.open(BytesIO(snapshot["texture_bytes"][texture_key])) as source:
            if source.format != "TGA" or max(source.size) > 4096:
                raise ValueError(f"Texture needs explicit resize/bake: {key}")
            image = source.convert("RGBA")
        inputs = graph.get("inputs", [])
        coord = inputs[0] if inputs else None
        if any(inputs[1:]):
            raise ValueError(f"Texture sample uses unsupported derivative/mip inputs: {key}")
        if coord:
            if coord["class"] != "MaterialExpressionTextureCoordinate" or coord.get("coordinate_index") != 0:
                raise ValueError(f"Texture needs a UV bake: {key}")
            scale = [coord["u_tiling"], coord["v_tiling"]]
    elif graph["class"] == "MaterialExpressionConstant3Vector":
        image = Image.new("RGBA", (4, 4), linear_rgb_to_srgb8(graph["constant"][:3]) + (255,))
    else:
        raise ValueError(f"Unsupported diffuse graph; bake it explicitly: {key}")
    mode = material["blend_mode"]
    if "BLEND_OPAQUE" in mode:
        blend = "opaque"
        image.putalpha(255)
    elif "BLEND_MASKED" in mode:
        mask = material.get("opacity_mask") or {}
        if mask != graph or material.get("opacity_mask_channel") != "A" or not texture_key:
            raise ValueError(f"Masked material needs a native alpha bake: {key}")
        cutoff = float(material["mask_cutoff"])
        if not 0 <= cutoff <= 1:
            raise ValueError(f"Invalid native alpha cutoff: {key}")
        image.putalpha(image.getchannel("A").point(lambda value: 255 if value / 255 >= cutoff else 0))
        blend = "masked"
    else:
        raise ValueError(f"Blend mode requires a reviewed conversion: {key}")
    output = BytesIO()
    image.save(output, format="TGA")
    return output.getvalue(), blend, scale, False, "Explicit diffuse-only recipe; Unreal lighting, normals/specular and runtime shading are not translated."


def convert_native_snapshot(snapshot_path: str | Path, policy_path: str | Path, output_dir: str | Path) -> Path:
    """Create a new v1 conversion bundle from an explicit immutable policy."""
    snapshot_path, policy_path, target = Path(snapshot_path).resolve(), Path(policy_path).resolve(), Path(output_dir).resolve()
    snapshot = load_native_snapshot(snapshot_path)
    if policy_path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Conversion policy exceeds size limit.")
    policy_bytes = policy_path.read_bytes()
    policy = _json_object(policy_bytes)
    if policy.get("schema") != "ghoststudio.unreal-conversion-policy/v1" or policy.get("accept_diffuse_only") is not True:
        raise ValueError("Choose and accept an explicit diffuse-only conversion policy.")
    if policy.get("snapshot_sha256") != snapshot["snapshot_sha256"]:
        raise ValueError("Conversion policy preview is stale; inspect this source revision again.")
    if snapshot.get("world_partition") or snapshot.get("streaming_levels"):
        raise ValueError("World Partition/streaming conversion is not qualified.")
    excluded = policy.get("excluded_components", {})
    if any(not isinstance(reason, str) or not reason.strip() for reason in excluded.values()):
        raise ValueError("Every exclusion requires an explicit reason.")
    ids = {c["id"] for c in snapshot["components"]}
    if set(excluded) - ids:
        raise ValueError("Exclusion refers to an unknown component.")
    for diagnostic in snapshot["diagnostics"]:
        if diagnostic["severity"] == "unsupported" and diagnostic["subject"] not in excluded:
            raise ValueError(f"{diagnostic['subject']}: {diagnostic['message']} {diagnostic['repair']}")
    for component in snapshot["components"]:
        if component["disposition"] == "static_geometry" and not component.get("visible", True) and component["id"] not in excluded and component["id"] not in policy.get("include_hidden_components", []):
            raise ValueError(f"Choose whether to include hidden component: {component['label']}")
    for root in (snapshot_path.parent, policy_path.parent):
        if target == root or target.is_relative_to(root) or root.is_relative_to(target):
            raise ValueError("Conversion output must be separate from source and policy data.")
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ghost-convert-", dir=target.parent))
    try:
        def put(data, extension):
            digest = hashlib.sha256(data).hexdigest()
            path = staging / f"{digest}.{extension}"
            if not path.exists():
                path.write_bytes(data)
            return {"path": path.name, "sha256": digest}

        def put_json(data):
            return put((json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode(), "json")

        material_rows, scales, accounting, actors = [], {}, [], []
        included = [c for c in snapshot["components"] if c["disposition"] == "static_geometry" and c["id"] not in excluded]
        used_materials = {key for component in included for key in component["materials"]}
        for key in sorted(used_materials):
            data, blend, scale, unlit, note = _diffuse(snapshot, key, policy.get("materials", {}).get(key, {}), policy_path.parent)
            scales[key] = scale
            material_rows.append({"id": key, "conversion": "baked-diffuse-v1", "blend_mode": blend,
                "diffuse_file": put(data, "tga"), "two_sided": snapshot["materials"][key]["two_sided"],
                "unlit": unlit, "conversion_note": note})
        collision = policy.get("collision", {})
        for component in snapshot["components"]:
            identity = component["id"]
            if identity in excluded or component["disposition"] == "editor_only":
                accounting.append({"id": identity, "disposition": "excluded", "reason": excluded.get(identity, component.get("reason"))})
                continue
            if component["disposition"] != "static_geometry":
                raise ValueError(f"Choose an explicit exclusion/repair for {component['label']} ({identity}).")
            sections = snapshot["resolved_meshes"][component["mesh"]]["sections"]
            surfaces = []
            for section in sections:
                slot = section["material_slot"]
                if type(slot) is not int or not 0 <= slot < len(component["materials"]):
                    raise ValueError(f"Unresolved section material slot: {identity}")
                key = component["materials"][slot]
                surfaces.append(normalize_native_section(section, key, scales[key]))
            mesh = {"surfaces": surfaces}
            role = "static-dressing"
            if collision.get("component_id") == identity:
                if "walkmesh_file" in collision:
                    mesh["walkmesh"] = _json_object(read_bundle_file(policy_path.parent, collision["walkmesh_file"]))
                elif collision.get("recipe") == "explicit-render-section-v1":
                    index = collision.get("section")
                    if type(index) is not int or not 0 <= index < len(surfaces):
                        raise ValueError("Select a valid collision section.")
                    surface = surfaces[index]
                    mesh["walkmesh"] = {"intent": "authored-walkmesh", "vertices": surface["vertices"],
                        "triangles": surface["triangles"], "surface_ids": [collision["surface_id"]] * len(surface["triangles"])}
                else:
                    raise ValueError("Collision requires an explicit authored sidecar or selected render section.")
                role = "static-room"
            actors.append({"id": identity, "class": "StaticMeshComponent", "role": role,
                "asset_path": component["mesh"], "world_matrix": component["world_matrix"], "mesh_file": put_json(mesh),
                "metadata": {"label": component["label"], "source_actor_guid": component["actor_guid"],
                             "source_actor": component["actor_path"], "backdrop": identity in policy.get("backdrop_components", [])}})
            accounting.append({"id": identity, "disposition": "converted"})
        bundle = {"schema": "ghoststudio.unreal-level/v1", "coordinates": "ue-centimeters-z-up-left-handed",
            "matrix_convention": "row-major-column-vector-local-to-world", "uv_origin": "bottom-left",
            "triangle_winding": "ccw-relative-to-normals", "module": policy["module"], "materials": material_rows, "actors": actors,
            "source": {"engine_version": snapshot["engine_version"], "exporter_version": snapshot["exporter_version"],
                "map_package": snapshot["map_package"], "map_file": put(read_bundle_file(snapshot_path.parent, snapshot["source_map"]), "umap"),
                "scope": "loaded-static-level", "unresolved_packages": [], "world_partition": False, "streaming_levels": [],
                "component_count": len(actors), "native_accounting": accounting, "snapshot_sha256": snapshot["snapshot_sha256"],
                "policy_sha256": hashlib.sha256(policy_bytes).hexdigest()}}
        manifest = staging / "level.json"
        manifest.write_text(json.dumps(bundle, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        from src.io.unreal_level_bundle import load_unreal_level_bundle
        from src.core.modules.unreal_level_import import build_unreal_level_project
        build_unreal_level_project(load_unreal_level_bundle(manifest))
        if policy_path.read_bytes() != policy_bytes or hashlib.sha256(snapshot_path.read_bytes()).hexdigest() != snapshot["snapshot_sha256"]:
            raise ValueError("Source snapshot or conversion policy changed during conversion.")
        if os.name != "nt":
            raise RuntimeError("Atomic directory publication is qualified on Windows only.")
        os.rename(staging, target)
        return target / "level.json"
    finally:
        if staging.exists():
            shutil.rmtree(staging)
