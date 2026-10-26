"""Renderer-neutral KOTOR-to-PBR material translation for export pipelines.

KOTOR material data is diffuse/TXI based rather than physically based.  This
module therefore preserves the authored base color and normal data, translates
Phong shininess to a conservative roughness value, keeps dielectric surfaces
non-metallic unless an explicit PBR override exists, and synthesizes only the
neutral maps required by modern engine shaders.

Owner: ``GhostRigger.Core.IO``.  This module is package-local because the
current FBX/glTF exporters are owned by the Core.IO embedded Python payload;
there is no matching canonical root ``src/core/export`` implementation yet.
"""

from __future__ import annotations

import base64
import json
import math
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping

PBR_SCHEMA = "ghostrigger.pbr_material_package.v1"
_INVALID_TEXTURE_NAMES = {"", "black", "null", "none", "****"}


@dataclass(frozen=True)
class PbrMaterialPayload:
    """One translated material plus encoded PNG texture payloads."""

    name: str
    source_diffuse: str
    source_normal: str
    target: str
    shader: Mapping[str, Any]
    factors: Mapping[str, Any]
    alpha: Mapping[str, Any]
    images: Mapping[str, bytes]
    channels: Mapping[str, Mapping[str, Any]]
    conversion: Mapping[str, Any]

    def data_uri(self, role: str) -> str | None:
        data = self.images.get(role)
        if not data:
            return None
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:image/png;base64,{encoded}"


@dataclass(frozen=True)
class PbrExportResult:
    """Files and material records emitted next to an engine export."""

    manifest_path: Path
    texture_dir: Path
    target: str
    shader_name: str
    materials: tuple[Mapping[str, Any], ...]

    def material_for_binding(
        self,
        mesh_name: str,
        slot: int,
    ) -> Mapping[str, Any] | None:
        mesh_key = str(mesh_name or "").casefold()
        return next(
            (
                material
                for material in self.materials
                if str(material.get("mesh") or "").casefold() == mesh_key
                and int(material.get("slot", -1)) == int(slot)
            ),
            None,
        )

    def summary(self) -> dict[str, Any]:
        shader_variants = sorted({
            str(material.get("shader") or "")
            for material in self.materials
            if str(material.get("shader") or "")
        })
        blend_operations = sorted({
            str(dict(material.get("alpha") or {}).get("blend_operation") or "")
            for material in self.materials
            if str(dict(material.get("alpha") or {}).get("blend_operation") or "")
        })
        return {
            "manifest": self.manifest_path.name,
            "target": self.target,
            "shader": self.shader_name,
            "material_count": len(self.materials),
            "texture_directory": self.texture_dir.relative_to(
                self.manifest_path.parent
            ).as_posix(),
            "shader_variants": shader_variants,
            "blend_operations": blend_operations,
        }


def normalize_pbr_target(value: str | None) -> str:
    key = str(value or "gltf").strip().lower().replace("-", "_")
    aliases = {
        "glb": "gltf",
        "gl_tf": "gltf",
        "ue": "unreal",
        "ue5": "unreal",
        "unreal_engine": "unreal",
        "unity_engine": "unity",
    }
    key = aliases.get(key, key)
    if key not in {"gltf", "unity", "unreal"}:
        raise ValueError(f"Unsupported PBR export target: {value}")
    return key


def shader_contract(
    target: str,
    *,
    blend_operation: str = "OPAQUE",
) -> dict[str, Any]:
    """Return the target's native metallic-roughness shader contract."""
    target = normalize_pbr_target(target)
    blend_operation = str(blend_operation or "OPAQUE").strip().upper()
    if target == "unity":
        contract = {
            "name": "Unity Lit",
            "workflow": "metallic_smoothness",
            "compatible_shaders": [
                "Universal Render Pipeline/Lit",
                "HDRP/Lit",
                "Standard",
            ],
            "bindings": {
                "base_color": ["_BaseMap", "_BaseColorMap", "_MainTex"],
                "normal": ["_BumpMap", "_NormalMap"],
                "metallic_smoothness": ["_MetallicGlossMap", "_MaskMap"],
                "ambient_occlusion": ["_OcclusionMap", "_MaskMap.G"],
            },
        }
        if blend_operation == "ADDITIVE":
            contract.update({
                "name": "Unity Additive Lit",
                "compatible_shaders": [
                    "Universal Render Pipeline/Particles/Lit",
                    "HDRP/Lit (Transparent/Additive)",
                    "Custom Shader Graph (Additive)",
                ],
                "surface_type": "Transparent",
                "blend_mode": "Additive",
                "z_write": False,
            })
        return contract
    if target == "unreal":
        contract = {
            "name": "Unreal Default Lit",
            "workflow": "metallic_roughness",
            "shading_model": "Default Lit",
            "bindings": {
                "base_color": "Base Color",
                "normal": "Normal",
                "orm.r": "Ambient Occlusion",
                "orm.g": "Roughness",
                "orm.b": "Metallic",
            },
        }
        if blend_operation == "ADDITIVE":
            contract.update({
                "name": "Unreal Additive Material",
                "blend_mode": "Additive",
                "shading_model": "Unlit",
                "bindings": {
                    "base_color": "Emissive Color",
                    "base_color.a": "Opacity",
                },
            })
        return contract
    contract = {
        "name": "glTF 2.0 PBR Metallic-Roughness",
        "workflow": "metallic_roughness",
        "bindings": {
            "base_color": "pbrMetallicRoughness.baseColorTexture",
            "orm.g": "pbrMetallicRoughness.metallicRoughnessTexture.roughness",
            "orm.b": "pbrMetallicRoughness.metallicRoughnessTexture.metallic",
            "orm.r": "occlusionTexture",
            "normal": "normalTexture",
        },
    }
    if blend_operation == "ADDITIVE":
        contract.update({
            "blend_mode": "BLEND",
            "requested_blend_operation": "ADDITIVE",
            "fallback": "glTF core has no additive blend mode; use ghostriggerPbr.alpha metadata.",
        })
    return contract


def _texture_name(node: Any) -> str:
    value = (
        getattr(node, "texture_clean", "")
        or getattr(node, "texture", "")
        or ""
    )
    value = str(value).strip()
    return "" if value.casefold() in _INVALID_TEXTURE_NAMES else value


def _normal_name(node: Any) -> str:
    value = (
        getattr(node, "bump_map", "")
        or getattr(node, "txi_bumpmaptexture", "")
        or ""
    )
    value = str(value).strip()
    return "" if value.casefold() in _INVALID_TEXTURE_NAMES else value


def _safe_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clamp01(value: Any, default: float) -> float:
    return max(0.0, min(1.0, _safe_float(value, default)))


def _material_factors(node: Any) -> dict[str, Any]:
    explicit_metallic = getattr(node, "pbr_metallic_factor", None)
    explicit_roughness = getattr(node, "pbr_roughness_factor", None)
    metallic = _clamp01(explicit_metallic, 0.0) if explicit_metallic is not None else 0.0
    if explicit_roughness is not None:
        roughness = _clamp01(explicit_roughness, 0.8)
        roughness_source = "explicit_pbr_override"
    else:
        shininess = max(0.0, _safe_float(getattr(node, "shininess", 1.125), 1.125))
        # Common Blinn-Phong exponent to perceptual roughness approximation.
        roughness = max(0.18, min(1.0, math.sqrt(2.0 / (shininess + 2.0))))
        roughness_source = "converted_from_blinn_phong_shininess"
    return {
        "metallic": metallic,
        "roughness": roughness,
        "ambient_occlusion": 1.0,
        "metallic_source": (
            "explicit_pbr_override" if explicit_metallic is not None else "neutral_dielectric_default"
        ),
        "roughness_source": roughness_source,
    }


def _alpha_contract(node: Any) -> dict[str, Any]:
    alpha = _clamp01(getattr(node, "alpha", 1.0), 1.0)
    blending = int(_safe_float(getattr(node, "txi_blending", 0), 0.0))
    alpha_cutoff = _clamp01(getattr(node, "txi_alpha_test", 0.5), 0.5)
    if blending == 2:
        mode = "MASK"
    elif alpha < 0.999 or blending:
        mode = "BLEND"
    else:
        mode = "OPAQUE"
    blend_operation = (
        "ADDITIVE"
        if blending == 1
        else "ALPHA" if mode == "BLEND" else mode
    )
    return {
        "mode": mode,
        "blend_operation": blend_operation,
        "factor": alpha,
        "cutoff": alpha_cutoff if mode == "MASK" else None,
        "source_txi_blending": blending,
    }


def pbr_material_identity(node: Any, *, target: str) -> tuple[Any, ...]:
    """Return the complete semantic identity used to reuse encoded PBR maps."""
    target = normalize_pbr_target(target)
    factors = _material_factors(node)
    alpha = _alpha_contract(node)
    diffuse_name = _texture_name(node)
    material_name = diffuse_name or str(getattr(node, "name", "material") or "material")
    return (
        target,
        material_name.casefold(),
        diffuse_name.casefold(),
        _normal_name(node).casefold(),
        float(factors["metallic"]),
        float(factors["roughness"]),
        str(factors["metallic_source"]),
        str(factors["roughness_source"]),
        str(alpha["mode"]),
        str(alpha["blend_operation"]),
        float(alpha["factor"]),
        alpha["cutoff"],
        int(alpha["source_txi_blending"]),
    )


def _cache_image(texture_cache: Any, name: str):
    if texture_cache is None or not name:
        return None
    try:
        image = texture_cache.get(name)
    except Exception:
        return None
    if image is None:
        return None
    copier = getattr(image, "copy", None)
    return copier() if callable(copier) else image


def _flip_vertical(image):
    try:
        from PIL import Image

        transpose_enum = getattr(Image, "Transpose", Image)
        return image.transpose(transpose_enum.FLIP_TOP_BOTTOM)
    except Exception:
        return image


def _flip_normal_green(image):
    rgba = image.convert("RGBA")
    red, green, blue, alpha = rgba.split()
    try:
        from PIL import ImageOps

        green = ImageOps.invert(green)
    except Exception:
        green = green.point(lambda value: 255 - int(value))
    from PIL import Image

    return Image.merge("RGBA", (red, green, blue, alpha))


def _png_bytes(image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def build_pbr_material_payload(
    node: Any,
    texture_cache: Any,
    *,
    target: str,
    flip_vertical: bool = True,
) -> PbrMaterialPayload:
    """Translate one KOTOR material node into target-ready PBR textures."""
    from PIL import Image

    target = normalize_pbr_target(target)
    diffuse_name = _texture_name(node)
    normal_name = _normal_name(node)
    factors = _material_factors(node)
    alpha = _alpha_contract(node)

    diffuse = _cache_image(texture_cache, diffuse_name)
    if diffuse is not None:
        diffuse = diffuse.convert("RGBA")
        if flip_vertical:
            diffuse = _flip_vertical(diffuse)

    normal = _cache_image(texture_cache, normal_name)
    if normal is not None:
        normal = normal.convert("RGBA")
        if flip_vertical:
            normal = _flip_vertical(normal)

    size = (
        tuple(diffuse.size)
        if diffuse is not None
        else tuple(normal.size) if normal is not None else (1, 1)
    )
    normal_source = "authored_kotor_bump_map" if normal is not None else "synthesized_flat_normal"
    if normal is None:
        normal = Image.new("RGBA", size, (128, 128, 255, 255))

    # Odyssey/KOTOR is Direct3D-oriented. glTF and Unity authoring convention
    # use Y+ normals, while Unreal consumes DirectX Y- normals.
    target_normal_y = "DirectX Y-" if target == "unreal" else "OpenGL Y+"
    normal_green_flipped = target != "unreal" and normal_source == "authored_kotor_bump_map"
    if normal_green_flipped:
        normal = _flip_normal_green(normal)

    roughness_byte = int(round(float(factors["roughness"]) * 255.0))
    metallic_byte = int(round(float(factors["metallic"]) * 255.0))
    orm = Image.new(
        "RGBA",
        size,
        (255, roughness_byte, metallic_byte, 255),
    )
    images: dict[str, bytes] = {
        "normal": _png_bytes(normal),
        "orm": _png_bytes(orm),
    }
    if diffuse is not None:
        images["base_color"] = _png_bytes(diffuse)
    if target == "unity":
        smoothness_byte = 255 - roughness_byte
        # The same map is valid as a Standard/URP metallic-gloss map and as an
        # HDRP mask map: R=metallic, G=AO, B=detail mask, A=smoothness.
        metallic_smoothness = Image.new(
            "RGBA",
            size,
            (metallic_byte, 255, 0, smoothness_byte),
        )
        ao = Image.new("L", size, 255)
        images["metallic_smoothness"] = _png_bytes(metallic_smoothness)
        images["ambient_occlusion"] = _png_bytes(ao)

    channels: dict[str, dict[str, Any]] = {
        "normal": {
            "color_space": "linear",
            "compression": "normal_map",
            "normal_y": target_normal_y,
            "source": normal_source,
        },
        "orm": {
            "color_space": "linear",
            "packing": {
                "r": "ambient_occlusion",
                "g": "roughness",
                "b": "metallic",
            },
            "source": "neutral_ao_plus_converted_material_factors",
        },
    }
    if diffuse is not None:
        channels["base_color"] = {
            "color_space": "sRGB",
            "alpha": "preserved_straight",
            "source": diffuse_name,
        }
    if target == "unity":
        channels["metallic_smoothness"] = {
            "color_space": "linear",
            "packing": {
                "r": "metallic",
                "g": "ambient_occlusion",
                "b": "detail_mask",
                "a": "smoothness",
            },
        }
        channels["ambient_occlusion"] = {
            "color_space": "linear",
            "source": "neutral_white",
        }

    material_name = diffuse_name or str(getattr(node, "name", "material") or "material")
    return PbrMaterialPayload(
        name=material_name,
        source_diffuse=diffuse_name,
        source_normal=normal_name,
        target=target,
        shader=shader_contract(
            target,
            blend_operation=str(alpha["blend_operation"]),
        ),
        factors=factors,
        alpha=alpha,
        images=images,
        channels=channels,
        conversion={
            "source_model": "KOTOR diffuse/TXI Blinn-Phong",
            "target_workflow": "metallic_roughness",
            "source_normal_y": "DirectX Y-",
            "normal_green_flipped": normal_green_flipped,
            "authored_ao_available": False,
            "authored_metallic_available": getattr(node, "pbr_metallic_factor", None) is not None,
        },
    )


def _iter_material_nodes(model: Any) -> Iterable[tuple[str, int, Any]]:
    try:
        mesh_nodes = list(model.mesh_nodes())
    except Exception:
        try:
            mesh_nodes = [
                node
                for node in model.all_nodes()
                if bool(getattr(node, "is_mesh", False) or getattr(node, "is_skin", False))
            ]
        except Exception:
            mesh_nodes = []
    for mesh in mesh_nodes:
        material_nodes = list(getattr(mesh, "_gr_fbx_material_slots", None) or [mesh])
        for slot, material_node in enumerate(material_nodes):
            yield str(getattr(mesh, "name", "") or ""), slot, material_node


def _safe_stem(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value or "material")).strip("._")
    return value or "material"


def export_pbr_material_package(
    model: Any,
    output_dir: str | Path,
    texture_cache: Any,
    *,
    target: str,
    asset_stem: str,
) -> PbrExportResult:
    """Write target-specific PBR textures and a shader/material manifest."""
    target = normalize_pbr_target(target)
    root = Path(output_dir)
    texture_dir = root / "textures" / _safe_stem(asset_stem) / target
    texture_dir.mkdir(parents=True, exist_ok=True)
    shader = shader_contract(target)
    material_records: list[dict[str, Any]] = []
    payload_cache: dict[tuple[Any, ...], PbrMaterialPayload] = {}
    artifact_cache: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    stem_owners: dict[str, tuple[Any, ...]] = {}

    role_suffixes = {
        "base_color": "BaseColor",
        "normal": "Normal",
        "orm": "ORM",
        "metallic_smoothness": "MetallicSmoothness",
        "ambient_occlusion": "AO",
    }
    for mesh_name, slot, node in _iter_material_nodes(model):
        identity = pbr_material_identity(node, target=target)
        payload = payload_cache.get(identity)
        if payload is None:
            payload = build_pbr_material_payload(node, texture_cache, target=target)
            payload_cache[identity] = payload

        channel_records = artifact_cache.get(identity)
        if channel_records is None:
            file_stem = _safe_stem(payload.name)
            owner = stem_owners.get(file_stem.casefold())
            if owner is not None and owner != identity:
                binding_stem = _safe_stem(mesh_name) or "mesh"
                file_stem = f"{file_stem}_{binding_stem}_slot{slot}"
                suffix = 2
                while (
                    file_stem.casefold() in stem_owners
                    and stem_owners[file_stem.casefold()] != identity
                ):
                    file_stem = f"{file_stem}_{suffix}"
                    suffix += 1
            stem_owners[file_stem.casefold()] = identity

            channel_records = {}
            for role, data in payload.images.items():
                role_suffix = role_suffixes[role]
                path = texture_dir / f"{file_stem}_{role_suffix}.png"
                try:
                    existing = path.read_bytes()
                except FileNotFoundError:
                    existing = None
                if existing != data:
                    path.write_bytes(data)
                record = dict(payload.channels[role])
                record["path"] = path.relative_to(root).as_posix()
                channel_records[role] = record
            artifact_cache[identity] = channel_records

        material_records.append(
            {
                "name": payload.name,
                "mesh": mesh_name,
                "slot": slot,
                "shader": str(payload.shader["name"]),
                "shader_contract": dict(payload.shader),
                "source_diffuse": payload.source_diffuse,
                "source_normal": payload.source_normal,
                "factors": dict(payload.factors),
                "alpha": dict(payload.alpha),
                "channels": {
                    role: dict(record)
                    for role, record in channel_records.items()
                },
                "conversion": dict(payload.conversion),
            }
        )

    manifest_path = root / f"{_safe_stem(asset_stem)}.pbr.json"
    manifest = {
        "schema": PBR_SCHEMA,
        "target": target,
        "asset": str(asset_stem),
        "shader": shader,
        "shader_variants": sorted({
            str(material.get("shader") or "")
            for material in material_records
            if str(material.get("shader") or "")
        }),
        "materials": material_records,
        "color_space_contract": {
            "color_textures": "sRGB",
            "data_textures": "linear",
        },
        "notes": [
            "KOTOR has no authored metallic or AO channels; neutral values are synthesized explicitly.",
            "Roughness is converted from Blinn-Phong shininess unless an explicit PBR override is present.",
            "Original source texture bytes remain owned by the KOTOR resource pipeline and are not mutated.",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return PbrExportResult(
        manifest_path=manifest_path,
        texture_dir=texture_dir,
        target=target,
        shader_name=str(shader["name"]),
        materials=tuple(material_records),
    )


__all__ = [
    "PBR_SCHEMA",
    "PbrExportResult",
    "PbrMaterialPayload",
    "build_pbr_material_payload",
    "export_pbr_material_package",
    "normalize_pbr_target",
    "pbr_material_identity",
    "shader_contract",
]
