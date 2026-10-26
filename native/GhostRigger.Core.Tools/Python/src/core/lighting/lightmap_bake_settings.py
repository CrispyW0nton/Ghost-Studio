"""Validated settings for generated lightmap bakes."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .kotor_engine_lighting import normalize_falloff_family


SUPPORTED_LIGHTMAP_RESOLUTIONS = (64, 128, 256, 512, 768, 1024, 2048)
SUPPORTED_LIGHTMAP_PREVIEW_RESOLUTIONS = (64, 128, 256, 512)
SUPPORTED_LIGHTMAP_FORMATS = ("png", "tga", "jpg", "jpeg")

# "ghost": Ghost Studio's artist bake ((1 - d/r)^2 falloff x 0.45, ambient floor, Reinhard tone map, gamma 2.2).
# It suits authored maps but is not a model of the engine.  "kotor_engine": the engine's light term in stored-byte
# units (src.core.lighting.kotor_engine_lighting): 1 / (1 + 4.5 d^2 / r^2), no tone map, no baked diffuse, linear
# encode.  Use it to validate lights fitted from vanilla lightmaps (the game draws D * (v0 + LM)).
LIGHTING_MODEL_GHOST = "ghost"
LIGHTING_MODEL_KOTOR_ENGINE = "kotor_engine"
LIGHTING_MODELS = (LIGHTING_MODEL_GHOST, LIGHTING_MODEL_KOTOR_ENGINE)
_LIGHTING_MODEL_ALIASES = {
    "ghost": LIGHTING_MODEL_GHOST,
    "artist": LIGHTING_MODEL_GHOST,
    "legacy": LIGHTING_MODEL_GHOST,
    "kotor_engine": LIGHTING_MODEL_KOTOR_ENGINE,
    "kotor": LIGHTING_MODEL_KOTOR_ENGINE,
    "engine": LIGHTING_MODEL_KOTOR_ENGINE,
    "kotor_linear": LIGHTING_MODEL_KOTOR_ENGINE,
}


def uses_kotor_engine_model(settings: object) -> bool:
    """True when ``settings`` (any settings-like object) selects the KOTOR engine light model."""

    key = str(getattr(settings, "lighting_model", LIGHTING_MODEL_GHOST) or LIGHTING_MODEL_GHOST).strip().lower()
    return _LIGHTING_MODEL_ALIASES.get(key) == LIGHTING_MODEL_KOTOR_ENGINE


@dataclass
class LightmapBakeSettings:
    resolution: int = 1024
    # UI labels are artist-facing and one-indexed: UV1, UV2, UV3.
    # Internally the baker stays zero-indexed: UV1 -> 0, UV2 -> 1, UV3 -> 2.
    selected_uv_channel: int = 1
    output_format: str = "png"
    output_directory: str = ""
    filename_prefix: str = ""
    bake_selected_only: bool = False
    bake_visible_only: bool = True
    include_disabled_lights: bool = False
    include_aurora_lights: bool = True
    include_generated_rig_lights: bool = True
    include_dynamic_lights: bool = True
    include_ambient: bool = False
    include_diffuse: bool = True
    include_normal_maps: bool = True
    include_specular: bool = False
    include_environment: bool = False
    use_gpu_acceleration: bool = True
    use_shadows: bool = False
    use_ambient_occlusion: bool = False
    use_direct_lighting: bool = True
    use_indirect_approximation: bool = False
    samples_per_texel: int = 1
    ambient_strength: float = 1.0
    diffuse_strength: float = 1.0
    indirect_strength: float = 1.0
    shadow_strength: float = 1.0
    normal_bias: float = 0.002
    light_falloff_multiplier: float = 1.0
    use_aurora_lights: bool = True
    use_original_lightmap_as_reference: bool = False
    preview_resolution: int = 128
    bake_resolution: int = 0
    shadow_samples: int = 1
    padding_pixels: int = 8
    dilation_passes: int = 8
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    exposure: float = 1.0
    gamma: float = 2.2
    clamp_output: bool = True
    overwrite_existing: bool = False
    generate_manifest: bool = True
    preview_after_bake: bool = True
    quality_preset: str = "standard"
    # Light model (see LIGHTING_MODELS).  The kotor_* fields and encode_gamma apply to "kotor_engine" only.
    lighting_model: str = LIGHTING_MODEL_GHOST
    # Falloff family: "engine" 1/(1 + 4.5 u^2), "ghost" (1 - u)^2, "linear" 1 - u, with u = d / r.
    kotor_falloff: str = "engine"
    # Cut the falloff at kotor_cutoff_scale * r: 1.0 is the engine's candidate rule d/r <= 1; 0 disables the cut
    # (the vertex program itself has none); the 003EBO interior pilot fitted best at 2.0.
    kotor_cutoff_scale: float = 1.0
    # The engine's vertex-program path ignores the MDL multiplier (PartLight +0x8C); True applies colour x multiplier
    # (Ghost light intensity) as the fixed-function scene pass would.
    kotor_apply_multiplier: bool = False
    # Stored value = clamp(E * exposure) ** (1 / encode_gamma); 1.0 keeps the lightmap linear in stored bytes.
    encode_gamma: float = 1.0

    warnings: list[str] = field(default_factory=list, repr=False)

    def normalized(self) -> "LightmapBakeSettings":
        """Return a corrected copy. Invalid user input becomes safe defaults."""
        data = asdict(self)
        data.pop("warnings", None)
        clean = LightmapBakeSettings(**data)
        clean.warnings = []

        if clean.resolution not in SUPPORTED_LIGHTMAP_RESOLUTIONS:
            clean.warnings.append(f"Unsupported resolution {clean.resolution}; using 1024.")
            clean.resolution = 1024
        if clean.bake_resolution <= 0:
            clean.bake_resolution = clean.resolution
        elif clean.bake_resolution not in SUPPORTED_LIGHTMAP_RESOLUTIONS:
            clean.bake_resolution = clean.resolution
        else:
            clean.resolution = clean.bake_resolution

        if clean.preview_resolution not in SUPPORTED_LIGHTMAP_PREVIEW_RESOLUTIONS:
            clean.warnings.append(f"Unsupported preview resolution {clean.preview_resolution}; using 128.")
            clean.preview_resolution = 128

        fmt = str(clean.output_format or "png").lower().lstrip(".")
        if fmt not in SUPPORTED_LIGHTMAP_FORMATS:
            clean.warnings.append(f"Unsupported output format {clean.output_format!r}; using png.")
            fmt = "png"
        clean.output_format = fmt

        clean.samples_per_texel = max(1, _int(clean.samples_per_texel, 1))
        clean.selected_uv_channel = max(0, _int(clean.selected_uv_channel, 1))
        clean.shadow_samples = max(1, _int(clean.shadow_samples, 1))
        clean.padding_pixels = max(0, _int(clean.padding_pixels, 8))
        clean.dilation_passes = max(0, _int(clean.dilation_passes, clean.padding_pixels))
        clean.exposure = _positive_float(clean.exposure, 1.0, "exposure", clean.warnings)
        clean.gamma = _positive_float(clean.gamma, 2.2, "gamma", clean.warnings)
        clean.ambient_strength = _non_negative_float(clean.ambient_strength, 1.0, "ambient strength", clean.warnings)
        clean.diffuse_strength = _non_negative_float(clean.diffuse_strength, 1.0, "diffuse strength", clean.warnings)
        clean.indirect_strength = _non_negative_float(clean.indirect_strength, 1.0, "indirect strength", clean.warnings)
        clean.shadow_strength = min(1.0, _non_negative_float(clean.shadow_strength, 1.0, "shadow strength", clean.warnings))
        clean.normal_bias = _non_negative_float(clean.normal_bias, 0.002, "normal bias", clean.warnings)
        clean.light_falloff_multiplier = _positive_float(
            clean.light_falloff_multiplier,
            1.0,
            "light falloff multiplier",
            clean.warnings,
        )
        clean.include_aurora_lights = bool(clean.include_aurora_lights and clean.use_aurora_lights)
        clean.output_directory = str(clean.output_directory or "").strip()

        model_key = str(clean.lighting_model or LIGHTING_MODEL_GHOST).strip().lower()
        if model_key not in _LIGHTING_MODEL_ALIASES:
            clean.warnings.append(f"Unknown lighting model {clean.lighting_model!r}; using {LIGHTING_MODEL_GHOST!r}.")
        clean.lighting_model = _LIGHTING_MODEL_ALIASES.get(model_key, LIGHTING_MODEL_GHOST)
        try:
            clean.kotor_falloff = normalize_falloff_family(clean.kotor_falloff)
        except ValueError:
            clean.warnings.append(f"Unknown KOTOR falloff {clean.kotor_falloff!r}; using 'engine'.")
            clean.kotor_falloff = "engine"
        clean.kotor_cutoff_scale = _non_negative_float(clean.kotor_cutoff_scale, 1.0, "KOTOR cutoff scale", clean.warnings)
        clean.kotor_apply_multiplier = bool(clean.kotor_apply_multiplier)
        clean.encode_gamma = _positive_float(clean.encode_gamma, 1.0, "encode gamma", clean.warnings)
        return clean

    def validate(self) -> list[str]:
        return self.normalized().warnings

    def to_summary(self) -> dict[str, Any]:
        data = asdict(self.normalized())
        data.pop("warnings", None)
        return data

    @classmethod
    def for_quality(cls, preset: str, **overrides: Any) -> "LightmapBakeSettings":
        key = str(preset or "standard").strip().lower()
        values: dict[str, Any]
        if key == "draft":
            values = {
                "resolution": 512,
                "bake_resolution": 512,
                "preview_resolution": 128,
                "samples_per_texel": 1,
                "shadow_samples": 1,
                "padding_pixels": 4,
                "dilation_passes": 4,
                "quality_preset": "draft",
            }
        elif key == "high":
            values = {
                "resolution": 2048,
                "bake_resolution": 2048,
                "preview_resolution": 256,
                "samples_per_texel": 4,
                "shadow_samples": 4,
                "padding_pixels": 12,
                "dilation_passes": 12,
                "quality_preset": "high",
            }
        elif key == "custom":
            values = {"quality_preset": "custom"}
        else:
            values = {
                "resolution": 1024,
                "bake_resolution": 1024,
                "preview_resolution": 128,
                "samples_per_texel": 2,
                "shadow_samples": 2,
                "padding_pixels": 8,
                "dilation_passes": 8,
                "quality_preset": "standard",
            }
        values.update(overrides)
        return cls(**values).normalized()

    @classmethod
    def for_kotor_engine(cls, **overrides: Any) -> "LightmapBakeSettings":
        """Settings for a KOTOR forward-model bake: the engine light term, linear, in stored-byte units.

        Diffuse is not baked (the game multiplies the diffuse texture at draw time), there is no synthetic ambient or
        indirect term, and the solve runs on the CPU in float64 so results do not depend on the GPU.
        """

        values: dict[str, Any] = {
            "lighting_model": LIGHTING_MODEL_KOTOR_ENGINE,
            "include_diffuse": False,
            "include_ambient": False,
            "use_indirect_approximation": False,
            "use_gpu_acceleration": False,
            "exposure": 1.0,
            "encode_gamma": 1.0,
            "kotor_falloff": "engine",
            "kotor_cutoff_scale": 1.0,
            "kotor_apply_multiplier": False,
            "quality_preset": "custom",
        }
        values.update(overrides)
        return cls(**values).normalized()

    def output_dir_path(self) -> Path:
        return Path(self.output_directory or "exports/lightmaps")


def _int(value: object, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _positive_float(value: object, fallback: float, label: str, warnings: list[str]) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        warnings.append(f"Invalid {label}; using {fallback}.")
        return fallback
    if not math.isfinite(parsed) or parsed <= 0.0:
        warnings.append(f"{label} must be a finite number greater than zero; using {fallback}.")
        return fallback
    return parsed


def _non_negative_float(value: object, fallback: float, label: str, warnings: list[str]) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        warnings.append(f"Invalid {label}; using {fallback}.")
        return fallback
    if not math.isfinite(parsed) or parsed < 0.0:
        warnings.append(f"{label} must be a finite non-negative number; using {fallback}.")
        return fallback
    return parsed
