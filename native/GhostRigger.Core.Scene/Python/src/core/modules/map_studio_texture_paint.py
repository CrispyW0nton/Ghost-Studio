"""Headless, tile-dirty texture painting for Map Studio.

The painter owns deterministic stroke resampling and RGBA compositing only.
It deliberately has no Qt, renderer, project-file, or KOTOR archive imports:
GUI code supplies UV hits, rendering code uploads the returned dirty tiles,
and IO code persists a flattened TGA when a stroke commits.

KOTOR uses the authored mesh's diffuse UV channel for these pixels.  Lightmap
UVs remain independent and must never be modified by this service.
"""

from __future__ import annotations

import io
import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


_RESREF_RE = re.compile(r"^[a-z0-9_]{1,16}$")
_SRGB_TO_LINEAR = tuple(
    (value / 255.0) / 12.92
    if (value / 255.0) <= 0.04045
    else (((value / 255.0) + 0.055) / 1.055) ** 2.4
    for value in range(256)
)
_SRGB_TO_LINEAR_ARRAY = np.asarray(_SRGB_TO_LINEAR, dtype=np.float32)


def _linear_to_srgb_byte(value: float) -> int:
    linear = max(0.0, min(1.0, float(value)))
    encoded = linear * 12.92 if linear <= 0.0031308 else (1.055 * (linear ** (1.0 / 2.4))) - 0.055
    return max(0, min(255, int(round(encoded * 255.0))))


def _linear_to_srgb_array(values: np.ndarray) -> np.ndarray:
    """Encode a linear RGB array with the scalar painter's rounding contract."""

    linear = np.clip(values, 0.0, 1.0)
    encoded = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        (1.055 * np.power(linear, 1.0 / 2.4)) - 0.055,
    )
    return np.clip(np.rint(encoded * 255.0), 0.0, 255.0).astype(np.uint8)


@dataclass(frozen=True)
class TexturePaintBrush:
    """Visible brush controls shared by mouse and tablet input."""

    radius_px: float = 24.0
    opacity: float = 1.0
    flow: float = 1.0
    hardness: float = 0.75
    spacing: float = 0.2
    rotation_degrees: float = 0.0
    jitter: float = 0.0
    pressure_size: bool = True
    pressure_flow: bool = True
    color: tuple[int, int, int, int] = (255, 255, 255, 255)
    stamp_size: tuple[int, int] = (0, 0)
    stamp_rgba: bytes = b""
    stamp_name: str = ""

    def normalised(self) -> "TexturePaintBrush":
        rgba = tuple(max(0, min(255, int(value))) for value in tuple(self.color)[:4])
        if len(rgba) < 4:
            rgba = (*rgba, *(255 for _ in range(4 - len(rgba))))
        stamp_width = max(0, int(tuple(self.stamp_size or (0, 0))[0])) if tuple(self.stamp_size or ()) else 0
        stamp_height = max(0, int(tuple(self.stamp_size or (0, 0, 0))[1])) if len(tuple(self.stamp_size or ())) > 1 else 0
        stamp_payload = bytes(self.stamp_rgba or b"")
        if stamp_width <= 0 or stamp_height <= 0 or len(stamp_payload) != stamp_width * stamp_height * 4:
            stamp_width, stamp_height, stamp_payload = 0, 0, b""
        return TexturePaintBrush(
            radius_px=max(0.5, min(4096.0, float(self.radius_px))),
            opacity=max(0.0, min(1.0, float(self.opacity))),
            flow=max(0.0, min(1.0, float(self.flow))),
            hardness=max(0.0, min(1.0, float(self.hardness))),
            spacing=max(0.01, min(4.0, float(self.spacing))),
            rotation_degrees=float(self.rotation_degrees) % 360.0,
            jitter=max(0.0, min(1.0, float(self.jitter))),
            pressure_size=bool(self.pressure_size),
            pressure_flow=bool(self.pressure_flow),
            color=rgba,
            stamp_size=(stamp_width, stamp_height),
            stamp_rgba=stamp_payload,
            stamp_name=str(self.stamp_name or ""),
        )


@dataclass(frozen=True)
class TexturePaintSample:
    """One deterministic pointer sample in diffuse UV space."""

    uv: tuple[float, float]
    pressure: float = 1.0


@dataclass(frozen=True)
class TexturePaintTile:
    """One renderer-ready top-left RGBA tile."""

    tile_x: int
    tile_y: int
    x: int
    y: int
    width: int
    height: int
    rgba: bytes


@dataclass(frozen=True)
class TexturePaintStrokeResult:
    changed: bool = False
    dirty_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
    dirty_tiles: tuple[tuple[int, int], ...] = ()
    stamp_count: int = 0
    pixels_changed: int = 0


@dataclass(frozen=True)
class _TilePatch:
    tile_x: int
    tile_y: int
    before: bytes
    after: bytes


@dataclass(frozen=True)
class _StrokeCommand:
    patches: tuple[_TilePatch, ...]
    result: TexturePaintStrokeResult


@dataclass
class _ActiveStroke:
    brush: TexturePaintBrush
    before_tiles: dict[tuple[int, int], bytes] = field(default_factory=dict)
    dirty_tiles: set[tuple[int, int]] = field(default_factory=set)
    pending_tiles: set[tuple[int, int]] = field(default_factory=set)
    coverage: np.ndarray | None = None
    dirty_mask: np.ndarray | None = None
    dirty_min_x: int | None = None
    dirty_min_y: int | None = None
    dirty_max_x: int | None = None
    dirty_max_y: int | None = None
    dirty_pixel_count: int = 0
    last_pixel: tuple[float, float] | None = None
    last_pressure: float = 1.0
    stamp_count: int = 0


class TexturePaintSession:
    """Mutable RGBA canvas with dirty-tile feedback and stroke undo/redo."""

    def __init__(
        self,
        width: int,
        height: int,
        rgba: bytes | bytearray,
        *,
        tile_size: int = 64,
        max_history: int = 64,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.tile_size = max(8, int(tile_size))
        self.max_history = max(1, int(max_history))
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Texture dimensions must be positive.")
        if len(rgba) != self.width * self.height * 4:
            raise ValueError("RGBA payload size does not match texture dimensions.")
        self._pixels = bytearray(rgba)
        self._active: _ActiveStroke | None = None
        self._undo: list[_StrokeCommand] = []
        self._redo: list[_StrokeCommand] = []

    @property
    def can_undo(self) -> bool:
        return bool(self._undo) and self._active is None

    @property
    def can_redo(self) -> bool:
        return bool(self._redo) and self._active is None

    @property
    def stroke_active(self) -> bool:
        return self._active is not None

    def rgba_bytes(self) -> bytes:
        return bytes(self._pixels)

    def begin_stroke(self, brush: TexturePaintBrush) -> None:
        if self._active is not None:
            raise RuntimeError("A texture-paint stroke is already active.")
        self._active = _ActiveStroke(
            brush.normalised(),
            coverage=np.zeros((self.height, self.width), dtype=np.float32),
            dirty_mask=np.zeros((self.height, self.width), dtype=np.bool_),
        )

    def break_stroke(self) -> bool:
        """Break interpolation while keeping the current drag one transaction.

        Leaving a valid face, crossing to another mesh surface, or re-entering
        after a picker miss must not draw a long UV-space bridge between hits.
        """

        active = self._active
        if active is None:
            return False
        active.last_pixel = None
        active.last_pressure = 1.0
        return True

    def append_sample(self, sample: TexturePaintSample | tuple[float, float], pressure: float = 1.0) -> int:
        active = self._active
        if active is None:
            raise RuntimeError("begin_stroke() must be called before appending samples.")
        if isinstance(sample, TexturePaintSample):
            uv = sample.uv
            pressure_value = sample.pressure
        else:
            uv = sample
            pressure_value = pressure
        if len(tuple(uv)) < 2:
            raise ValueError("Texture-paint samples require a two-component UV coordinate.")
        # KOTOR diffuse UVs use a bottom-left V origin; the canvas is top-left.
        pixel = ((float(uv[0]) % 1.0) * self.width, (1.0 - (float(uv[1]) % 1.0)) * self.height)
        pixel = (pixel[0] % self.width, pixel[1] % self.height)
        pressure_value = max(0.0, min(1.0, float(pressure_value)))
        stamps = 0
        if active.last_pixel is None:
            stamps += self._stamp(pixel[0], pixel[1], pressure_value)
        else:
            ax, ay = active.last_pixel
            # Select the shortest wrapped route so strokes cross UV seams smoothly.
            dx = pixel[0] - ax
            dy = pixel[1] - ay
            if abs(dx) > self.width * 0.5:
                dx -= math.copysign(self.width, dx)
            if abs(dy) > self.height * 0.5:
                dy -= math.copysign(self.height, dy)
            distance = math.hypot(dx, dy)
            step = max(1.0, active.brush.radius_px * 2.0 * active.brush.spacing)
            count = max(1, int(math.ceil(distance / step)))
            for index in range(1, count + 1):
                t = index / count
                x = (ax + dx * t) % self.width
                y = (ay + dy * t) % self.height
                p = active.last_pressure + ((pressure_value - active.last_pressure) * t)
                stamps += self._stamp(x, y, p)
        active.last_pixel = pixel
        active.last_pressure = pressure_value
        return stamps

    def end_stroke(self) -> TexturePaintStrokeResult:
        active = self._active
        if active is None:
            raise RuntimeError("No texture-paint stroke is active.")
        self._active = None
        if not active.dirty_tiles:
            return TexturePaintStrokeResult(stamp_count=active.stamp_count)
        patches = tuple(
            _TilePatch(tile_x, tile_y, active.before_tiles[(tile_x, tile_y)], self._read_tile(tile_x, tile_y))
            for tile_x, tile_y in sorted(active.dirty_tiles)
        )
        if None in (active.dirty_min_x, active.dirty_min_y, active.dirty_max_x, active.dirty_max_y):
            return TexturePaintStrokeResult(stamp_count=active.stamp_count)
        result = TexturePaintStrokeResult(
            changed=True,
            dirty_rect=(
                int(active.dirty_min_x),
                int(active.dirty_min_y),
                (int(active.dirty_max_x) - int(active.dirty_min_x)) + 1,
                (int(active.dirty_max_y) - int(active.dirty_min_y)) + 1,
            ),
            dirty_tiles=tuple(sorted(active.dirty_tiles)),
            stamp_count=active.stamp_count,
            pixels_changed=int(active.dirty_pixel_count),
        )
        self._undo.append(_StrokeCommand(patches, result))
        if len(self._undo) > self.max_history:
            del self._undo[0]
        self._redo.clear()
        return result

    def cancel_stroke(self) -> bool:
        active = self._active
        if active is None:
            return False
        for (tile_x, tile_y), rgba in active.before_tiles.items():
            self._write_tile(tile_x, tile_y, rgba)
        self._active = None
        return True

    def undo(self) -> TexturePaintStrokeResult | None:
        if not self.can_undo:
            return None
        command = self._undo.pop()
        for patch in command.patches:
            self._write_tile(patch.tile_x, patch.tile_y, patch.before)
        self._redo.append(command)
        return command.result

    def redo(self) -> TexturePaintStrokeResult | None:
        if not self.can_redo:
            return None
        command = self._redo.pop()
        for patch in command.patches:
            self._write_tile(patch.tile_x, patch.tile_y, patch.after)
        self._undo.append(command)
        return command.result

    def dirty_tile_payloads(self, dirty_tiles: Iterable[tuple[int, int]]) -> tuple[TexturePaintTile, ...]:
        payloads: list[TexturePaintTile] = []
        for tile_x, tile_y in sorted(set(tuple(item) for item in dirty_tiles)):
            x, y, width, height = self._tile_bounds(int(tile_x), int(tile_y))
            payloads.append(TexturePaintTile(int(tile_x), int(tile_y), x, y, width, height, self._read_tile(int(tile_x), int(tile_y))))
        return tuple(payloads)

    def pending_tile_payloads(self, *, clear: bool = True) -> tuple[TexturePaintTile, ...]:
        """Return only tiles changed since the previous live renderer upload."""

        active = self._active
        if active is None or not active.pending_tiles:
            return ()
        tiles = tuple(sorted(active.pending_tiles))
        payloads = self.dirty_tile_payloads(tiles)
        if clear:
            active.pending_tiles.difference_update(tiles)
        return payloads

    def active_dirty_tiles(self) -> tuple[tuple[int, int], ...]:
        """Return every tile touched by the current transaction."""

        active = self._active
        return tuple(sorted(active.dirty_tiles)) if active is not None else ()

    def _stamp(self, center_x: float, center_y: float, pressure: float) -> int:
        active = self._active
        if active is None:
            return 0
        brush = active.brush
        radius = brush.radius_px * (max(0.05, pressure) if brush.pressure_size else 1.0)
        flow_scale = brush.flow * (pressure if brush.pressure_flow else 1.0)
        if radius <= 0.0 or brush.opacity <= 0.0 or flow_scale <= 0.0:
            return 0
        if brush.jitter > 0.0 and active.stamp_count > 0:
            # Deterministic low-discrepancy scatter: the same pointer samples
            # always produce the same stroke and therefore the same undo bytes.
            phase = (active.stamp_count * 0.7548776662466927) % 1.0
            radial = (active.stamp_count * 0.5698402909980532) % 1.0
            offset = brush.jitter * radius * radial
            center_x = (center_x + (math.cos(phase * math.tau) * offset)) % self.width
            center_y = (center_y + (math.sin(phase * math.tau) * offset)) % self.height
        active.stamp_count += 1

        def axis_samples(center: float, extent: float, size: int) -> tuple[np.ndarray, np.ndarray]:
            minimum = math.floor(center - extent)
            maximum = math.ceil(center + extent)
            if (maximum - minimum) + 1 <= size:
                raw = np.arange(minimum, maximum + 1, dtype=np.int64)
                return np.mod(raw, size), (raw.astype(np.float32) + 0.5) - float(center)
            wrapped = np.arange(size, dtype=np.int64)
            delta = np.mod((wrapped.astype(np.float32) + 0.5) - float(center) + (size * 0.5), size) - (size * 0.5)
            return wrapped, delta

        x_indices, x_delta = axis_samples(center_x, radius, self.width)
        y_indices, y_delta = axis_samples(center_y, radius, self.height)
        dx, dy = np.meshgrid(x_delta, y_delta)
        px, py = np.meshgrid(x_indices, y_indices)
        distance = np.hypot(dx, dy)
        inside = distance <= radius
        if not np.any(inside):
            return 0
        hard_radius = radius * brush.hardness
        feather = max(1.0e-6, radius - hard_radius)
        falloff = np.where(
            distance <= hard_radius,
            1.0,
            np.clip(1.0 - ((distance - hard_radius) / feather), 0.0, 1.0),
        )[inside].astype(np.float32, copy=False)
        xs = px[inside].astype(np.int64, copy=False)
        ys = py[inside].astype(np.int64, copy=False)
        local_x = (dx[inside] / radius).astype(np.float32, copy=False)
        local_y = (dy[inside] / radius).astype(np.float32, copy=False)
        source_rgba = self._brush_source_rgba_array(brush, local_x, local_y)
        deposit = np.clip(
            flow_scale * falloff * (source_rgba[:, 3].astype(np.float32) / 255.0),
            0.0,
            1.0,
        )
        coverage = active.coverage
        if coverage is None:
            return 0
        previous_coverage = coverage[ys, xs]
        wanted_coverage = previous_coverage + ((1.0 - previous_coverage) * deposit)
        next_coverage = np.minimum(float(brush.opacity), wanted_coverage)
        denominator = np.maximum(1.0e-7, 1.0 - previous_coverage)
        source_alpha = np.clip((next_coverage - previous_coverage) / denominator, 0.0, 1.0)
        eligible = source_alpha > 1.0e-6
        if not np.any(eligible):
            return 0
        xs = xs[eligible]
        ys = ys[eligible]
        source_alpha = source_alpha[eligible]
        source_rgba = source_rgba[eligible]
        coverage[ys, xs] = next_coverage[eligible]

        pixels = np.frombuffer(self._pixels, dtype=np.uint8).reshape((self.height, self.width, 4))
        before = pixels[ys, xs].copy()
        inverse = 1.0 - source_alpha
        source_linear = _SRGB_TO_LINEAR_ARRAY[source_rgba[:, :3]]
        before_linear = _SRGB_TO_LINEAR_ARRAY[before[:, :3]]
        after = np.empty_like(before)
        after[:, :3] = _linear_to_srgb_array(
            (source_linear * source_alpha[:, None]) + (before_linear * inverse[:, None])
        )
        after[:, 3] = np.clip(
            np.rint((255.0 * source_alpha) + (before[:, 3].astype(np.float32) * inverse)),
            0.0,
            255.0,
        ).astype(np.uint8)
        changed_mask = np.any(after != before, axis=1)
        if not np.any(changed_mask):
            return 0
        changed_x = xs[changed_mask]
        changed_y = ys[changed_mask]
        changed_after = after[changed_mask]

        tile_pairs = np.unique(
            np.column_stack((changed_x // self.tile_size, changed_y // self.tile_size)),
            axis=0,
        )
        for tile_x, tile_y in tile_pairs:
            tile = (int(tile_x), int(tile_y))
            if tile not in active.before_tiles:
                active.before_tiles[tile] = self._read_tile(*tile)
            active.dirty_tiles.add(tile)
            active.pending_tiles.add(tile)

        dirty_mask = active.dirty_mask
        if dirty_mask is not None:
            newly_dirty = ~dirty_mask[changed_y, changed_x]
            active.dirty_pixel_count += int(np.count_nonzero(newly_dirty))
            dirty_mask[changed_y, changed_x] = True
        active.dirty_min_x = int(changed_x.min()) if active.dirty_min_x is None else min(active.dirty_min_x, int(changed_x.min()))
        active.dirty_min_y = int(changed_y.min()) if active.dirty_min_y is None else min(active.dirty_min_y, int(changed_y.min()))
        active.dirty_max_x = int(changed_x.max()) if active.dirty_max_x is None else max(active.dirty_max_x, int(changed_x.max()))
        active.dirty_max_y = int(changed_y.max()) if active.dirty_max_y is None else max(active.dirty_max_y, int(changed_y.max()))
        pixels[changed_y, changed_x] = changed_after
        return int(changed_x.size)

    @staticmethod
    def _brush_source_rgba_array(
        brush: TexturePaintBrush,
        local_x: np.ndarray,
        local_y: np.ndarray,
    ) -> np.ndarray:
        """Return vectorized source colors for one brush dab."""

        width, height = brush.stamp_size
        count = int(local_x.size)
        if width <= 0 or height <= 0 or len(brush.stamp_rgba) != width * height * 4:
            return np.broadcast_to(np.asarray(brush.color, dtype=np.uint8), (count, 4)).copy()
        radians = math.radians(float(brush.rotation_degrees))
        cosine = math.cos(radians)
        sine = math.sin(radians)
        rotated_x = (cosine * local_x) + (sine * local_y)
        rotated_y = (-sine * local_x) + (cosine * local_y)
        u = np.clip((rotated_x + 1.0) * 0.5, 0.0, 1.0)
        v = np.clip((rotated_y + 1.0) * 0.5, 0.0, 1.0)
        x = np.minimum(width - 1, np.rint(u * (width - 1)).astype(np.int64))
        y = np.minimum(height - 1, np.rint(v * (height - 1)).astype(np.int64))
        stamp = np.frombuffer(brush.stamp_rgba, dtype=np.uint8).reshape((height, width, 4))
        source = stamp[y, x].astype(np.float32)
        tint = np.asarray(brush.color, dtype=np.float32) / 255.0
        return np.clip(np.rint(source * tint), 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _brush_source_rgba(brush: TexturePaintBrush, local_x: float, local_y: float) -> tuple[int, int, int, int]:
        width, height = brush.stamp_size
        if width <= 0 or height <= 0 or len(brush.stamp_rgba) != width * height * 4:
            return brush.color
        # Stamp sources are top-left RGBA. Nearest sampling stays deterministic;
        # the brush falloff supplies the soft edge while avoiding per-frame resizes.
        u = max(0.0, min(1.0, (float(local_x) + 1.0) * 0.5))
        v = max(0.0, min(1.0, (float(local_y) + 1.0) * 0.5))
        x = min(width - 1, int(round(u * (width - 1))))
        y = min(height - 1, int(round(v * (height - 1))))
        offset = ((y * width) + x) * 4
        source = tuple(int(brush.stamp_rgba[offset + channel]) for channel in range(4))
        return (
            int(round(source[0] * (brush.color[0] / 255.0))),
            int(round(source[1] * (brush.color[1] / 255.0))),
            int(round(source[2] * (brush.color[2] / 255.0))),
            int(round(source[3] * (brush.color[3] / 255.0))),
        )

    def _tile_bounds(self, tile_x: int, tile_y: int) -> tuple[int, int, int, int]:
        x = tile_x * self.tile_size
        y = tile_y * self.tile_size
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            raise ValueError("Texture tile is outside the canvas.")
        return (x, y, min(self.tile_size, self.width - x), min(self.tile_size, self.height - y))

    def _read_tile(self, tile_x: int, tile_y: int) -> bytes:
        x, y, width, height = self._tile_bounds(tile_x, tile_y)
        result = bytearray(width * height * 4)
        for row in range(height):
            source = (((y + row) * self.width) + x) * 4
            target = row * width * 4
            result[target : target + width * 4] = self._pixels[source : source + width * 4]
        return bytes(result)

    def _write_tile(self, tile_x: int, tile_y: int, rgba: bytes) -> None:
        x, y, width, height = self._tile_bounds(tile_x, tile_y)
        if len(rgba) != width * height * 4:
            raise ValueError("Texture tile payload has the wrong size.")
        for row in range(height):
            target = (((y + row) * self.width) + x) * 4
            source = row * width * 4
            self._pixels[target : target + width * 4] = rgba[source : source + width * 4]


def validate_kotor_texture_resref(value: str) -> str:
    """Return a normalized extension-free KOTOR resref or raise ValueError."""

    clean = Path(str(value or "").strip()).stem.lower()
    if not _RESREF_RE.fullmatch(clean):
        raise ValueError("Texture resrefs must be 1-16 lowercase ASCII letters, digits, or underscores.")
    return clean


def suggest_kotor_texture_resref(value: str, existing: Iterable[str] = ()) -> str:
    """Create a unique, safe <=16-character resref for an imported texture."""

    stem = Path(str(value or "texture")).stem.lower()
    stem = re.sub(r"[^a-z0-9_]+", "_", stem).strip("_") or "texture"
    stem = stem[:16]
    used = {str(item or "").lower() for item in existing}
    if stem not in used:
        return stem
    for index in range(2, 10000):
        suffix = f"_{index}"
        candidate = f"{stem[: 16 - len(suffix)]}{suffix}"
        if candidate not in used:
            return candidate
    raise ValueError("Could not allocate a unique KOTOR texture resref.")


def encode_tga_rgba(width: int, height: int, rgba: bytes | bytearray) -> bytes:
    """Encode an uncompressed 32-bit, bottom-origin TGA for KOTOR export.

    ``rgba`` is top-down (as Pillow decodes). The Odyssey engine ignores the TGA
    descriptor's top-left-origin bit and always reads rows bottom-up, so rows are
    written bottom-up with the bit clear (descriptor 0x08): the game then draws
    exactly what Pillow and the Map Studio viewport show.
    """

    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0 or width > 65535 or height > 65535:
        raise ValueError("TGA dimensions must be in the range 1..65535.")
    if len(rgba) != width * height * 4:
        raise ValueError("RGBA payload size does not match TGA dimensions.")
    header = struct.pack("<BBBHHBHHHHBB", 0, 0, 2, 0, 0, 0, 0, 0, width, height, 32, 0x08)
    pixels = np.frombuffer(rgba, dtype=np.uint8).reshape((height, width, 4))
    bgra = np.ascontiguousarray(pixels[::-1, :, (2, 1, 0, 3)]).tobytes()
    return header + bgra


def decode_image_rgba(data: bytes) -> tuple[int, int, bytes]:
    """Decode TGA/TPC/PNG/JPEG/DDS through Pillow/PyKotor-compatible PIL."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - desktop payload includes Pillow
        raise RuntimeError("Pillow is required to import texture images.") from exc
    with Image.open(io.BytesIO(bytes(data))) as image:
        rgba = image.convert("RGBA")
        width, height = rgba.size
        if width <= 0 or height <= 0 or width > 8192 or height > 8192:
            raise ValueError("Imported texture dimensions must be between 1 and 8192 pixels.")
        return int(width), int(height), rgba.tobytes()


__all__ = [
    "TexturePaintBrush",
    "TexturePaintSample",
    "TexturePaintSession",
    "TexturePaintStrokeResult",
    "TexturePaintTile",
    "decode_image_rgba",
    "encode_tga_rgba",
    "suggest_kotor_texture_resref",
    "validate_kotor_texture_resref",
]
