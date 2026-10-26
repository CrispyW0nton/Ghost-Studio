"""Read-only, hash-checked interchange IO for Unreal -> Map Studio.

This is deliberately not a .umap decoder. Unreal must export the loaded scene.
String inspection of a loose package is a dependency hint, never geometry proof.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "ghoststudio.unreal-level/v1"
MAX_JSON_BYTES = 64 * 1024 * 1024


def _json_object(data: bytes) -> dict[str, Any]:
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("Interchange JSON exceeds the 64 MiB limit; split the source export.")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Non-finite JSON value: {value}")

    result = json.loads(data, object_pairs_hook=unique, parse_constant=invalid)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object.")
    return result


def read_bundle_file(root: Path, reference: dict[str, Any]) -> bytes:
    """Reject escapes, stale sidecars and implicit absolute filesystem access."""
    name = reference.get("path", "")
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("Bundle paths must be relative POSIX paths.")
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(p in ("..", ".") for p in name.split("/")):
        raise ValueError(f"Unsafe bundle path: {name}")
    root = root.resolve(strict=True)
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"Bundle path escapes its root: {name}")
    expected = reference.get("sha256", "")
    if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise ValueError(f"Missing SHA256 for {name}")
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"Bundle file exceeds size limit: {name}")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Stale or altered bundle file: {name}")
    return data


def load_unreal_level_bundle(manifest_path: str | Path) -> dict[str, Any]:
    """Resolve explicitly referenced geometry/texture files into an isolated value."""
    path = Path(manifest_path).resolve(strict=True)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Manifest exceeds size limit.")
    manifest_bytes = path.read_bytes()
    result = _json_object(manifest_bytes)
    if result.get("schema") != SCHEMA:
        raise ValueError("Unsupported Unreal level bundle schema.")
    # Source .umap is retained in the bundle: hash verification does not require
    # access to (or mutation of) the creator's original Unreal project.
    read_bundle_file(path.parent, result["source"]["map_file"])
    actors = result.get("actors")
    materials = result.get("materials")
    if not isinstance(actors, list) or not isinstance(materials, list):
        raise ValueError("Bundle requires actor and material lists.")
    if len(actors) > 10000 or len(materials) > 10000:
        raise ValueError("Bundle exceeds the v1 object budget.")
    cache: dict[tuple[str, str], dict] = {}
    for actor in actors:
        if "mesh" in actor:
            raise ValueError("Geometry must use a hashed mesh_file sidecar, not inline mesh data.")
        if "mesh_file" not in actor:
            continue
        ref = actor["mesh_file"]
        key = (ref["path"], ref["sha256"])
        if key not in cache:
            cache[key] = _json_object(read_bundle_file(path.parent, ref))
        actor["mesh"] = cache[key]
    for material in materials:
        ref = material.get("diffuse_file")
        if not ref or not str(ref.get("path", "")).lower().endswith(".tga"):
            raise ValueError("v1 requires a baked TGA for every material.")
        data = read_bundle_file(path.parent, ref)
        # Full decoding/format validation stays in the existing texture pipeline.
        from io import BytesIO
        from PIL import Image
        with Image.open(BytesIO(data)) as image:
            if image.format != "TGA" or image.width > 4096 or image.height > 4096:
                raise ValueError("Texture must be a TGA no larger than 4096 per side.")
            image.load()
        material["diffuse_bytes"] = data
    result["bundle_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    return result


def inspect_loose_umap(path: str | Path) -> dict[str, Any]:
    """Bounded hints for a useful 'select owning project' diagnostic."""
    path = Path(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Loose-map hint inspection is limited to 64 MiB; use Unreal.")
    data = path.read_bytes()
    if data[:4] != b"\xc1\x83\x2a\x9e":
        raise ValueError("Not a supported little-endian Unreal package signature.")
    strings = [s.decode("ascii") for s in re.findall(rb"[ -~]{6,}", data)]
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "engine_branches": sorted({s for s in strings if s.startswith("++UE") and len(s) < 100}),
        "package_hints": sorted({s for s in strings if re.fullmatch(r"/(Game|Engine)/[A-Za-z0-9_/]+", s)}),
        "native_dependencies_verified": False,
        "convertible": False,
        "next_step": "Select the owning .uproject and export the loaded level with its dependency closure.",
    }
