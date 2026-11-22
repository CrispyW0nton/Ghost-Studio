"""Transactional sidecar state for Map Studio texture documents.

KMAP stores texture references, never pixel blobs.  This journal keeps the
corresponding TGA/TPC/TXI before/after bytes only in editor memory so normal
Map Studio command history can undo file-backed paint operations and Discard
can restore the last successfully saved/opened project state.
"""

from __future__ import annotations

import os
import hashlib
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .kmap_model import KMapProject


@dataclass(frozen=True)
class MapStudioTextureSidecarSnapshot:
    """In-memory file state captured before one project texture operation."""

    project_key: tuple[str, str]
    states: tuple[tuple[str, bytes | None], ...] = ()


@dataclass(frozen=True)
class MapStudioTextureSidecarSpan:
    """One bounded changed byte range inside a same-size sidecar."""

    offset: int
    before: bytes
    after: bytes


@dataclass(frozen=True)
class MapStudioTextureSidecarPatch:
    """One file delta; full payloads are reserved for create/delete/resize."""

    project_key: tuple[str, str]
    path: str
    before_exists: bool
    after_exists: bool
    before_size: int = 0
    after_size: int = 0
    before_sha256: str = ""
    after_sha256: str = ""
    spans: tuple[MapStudioTextureSidecarSpan, ...] = ()
    before_payload: bytes | None = None
    after_payload: bytes | None = None

    @property
    def stored_byte_count(self) -> int:
        return sum(len(span.before) + len(span.after) for span in self.spans) + len(
            self.before_payload or b""
        ) + len(self.after_payload or b"")


def _project_key(project: KMapProject) -> tuple[str, str]:
    project_id = str(getattr(project, "project_id", "") or "")
    path_text = str(getattr(project, "path", "") or "").strip()
    resolved = str(Path(path_text).resolve()) if path_text else ""
    return project_id, resolved


def _resolve_project_path(project: KMapProject, value: str | Path) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path.resolve()
    project_path = str(getattr(project, "path", "") or "").strip()
    if not project_path:
        raise ValueError("Save the KMAP before tracking project texture sidecars.")
    return (Path(project_path).resolve().parent / path).resolve()


def managed_project_texture_sidecars(project: KMapProject, *, texture_id: str = "") -> tuple[Path, ...]:
    """Return every image/TXI path referenced by the current KMAP project."""

    paths: set[Path] = set()
    for texture in tuple(getattr(project, "textures", ()) or ()):
        if texture_id and str(getattr(texture, "texture_id", "") or "") != str(texture_id):
            continue
        image_value = str(getattr(texture, "path", "") or "").strip()
        if image_value:
            paths.add(_resolve_project_path(project, image_value))
        metadata = dict(getattr(texture, "metadata", {}) or {})
        txi_value = str(metadata.get("txi_path") or "").strip()
        if txi_value:
            paths.add(_resolve_project_path(project, txi_value))
    return tuple(sorted(paths, key=lambda value: str(value).lower()))


def _read_state(path: Path) -> bytes | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise IsADirectoryError(f"Project texture sidecar path is not a file: {path}")
    return path.read_bytes()


def _state_sha256(payload: bytes | None) -> str:
    return hashlib.sha256(payload).hexdigest() if payload is not None else ""


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            temporary_path = handle.name
            handle.write(bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = ""
    finally:
        if temporary_path:
            try:
                Path(temporary_path).unlink(missing_ok=True)
            except OSError:
                pass


def _write_state(path: Path, payload: bytes | None) -> None:
    if payload is None:
        path.unlink(missing_ok=True)
        return
    _atomic_write(path, payload)


def _normalise_ranges(ranges: Iterable[tuple[int, int]], *, size: int) -> tuple[tuple[int, int], ...]:
    clipped = sorted(
        (max(0, int(start)), min(int(size), int(end)))
        for start, end in tuple(ranges or ())
        if int(end) > int(start) and int(end) > 0 and int(start) < int(size)
    )
    merged: list[tuple[int, int]] = []
    for start, end in clipped:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def tga_dirty_tile_byte_ranges(
    *,
    width: int,
    height: int,
    tile_size: int,
    dirty_tiles: Iterable[tuple[int, int]],
    tga_bytes: bytes | bytearray | memoryview,
) -> tuple[tuple[int, int], ...]:
    """Map dirty tiles to byte rows in GhostRigger's exact authored TGA."""

    image_width = int(width)
    image_height = int(height)
    size = int(tile_size)
    if image_width <= 0 or image_height <= 0 or image_width > 65535 or image_height > 65535:
        raise ValueError("GhostRigger TGA dimensions must be in the range 1..65535.")
    if size <= 0:
        raise ValueError("Texture-paint tile size must be positive.")
    payload = bytes(tga_bytes)
    # Map Studio writes bottom-origin TGAs (descriptor 0x08), because the engine
    # ignores the top-left bit. Sidecars written before that used 0x28 (top-down).
    descriptor = payload[17] if len(payload) >= 18 else -1
    expected_header = struct.pack(
        "<BBBHHBHHHHBB", 0, 0, 2, 0, 0, 0, 0, 0, image_width, image_height, 32,
        descriptor if descriptor in (0x08, 0x28) else 0x08,
    )
    expected_size = 18 + (image_width * image_height * 4)
    if len(payload) != expected_size or payload[:18] != expected_header:
        raise ValueError(
            "Dirty-tile byte mapping requires GhostRigger's uncompressed 32-bit "
            "TGA with no image ID or color map."
        )
    bottom_up = descriptor == 0x08
    ranges: list[tuple[int, int]] = []
    for tile_x, tile_y in sorted(set((int(x), int(y)) for x, y in tuple(dirty_tiles or ()))):
        x = tile_x * size
        y = tile_y * size
        if x < 0 or y < 0 or x >= image_width or y >= image_height:
            continue
        row_width = min(size, image_width - x) * 4
        row_count = min(size, image_height - y)
        for row in range(row_count):
            file_row = (image_height - 1 - (y + row)) if bottom_up else (y + row)
            start = 18 + (((file_row * image_width) + x) * 4)
            ranges.append((start, start + row_width))
    return _normalise_ranges(ranges, size=expected_size)


def _changed_block_ranges(before: bytes, after: bytes) -> tuple[tuple[int, int], ...]:
    block_size = 64 * 1024
    return tuple(
        (start, min(len(before), start + block_size))
        for start in range(0, len(before), block_size)
        if before[start : start + block_size] != after[start : start + block_size]
    )


def _patch_from_states(
    project_key: tuple[str, str],
    path: str,
    before: bytes | None,
    after: bytes | None,
    *,
    ranges: Iterable[tuple[int, int]] = (),
) -> MapStudioTextureSidecarPatch | None:
    if before == after:
        return None
    if before is None or after is None or len(before) != len(after):
        return MapStudioTextureSidecarPatch(
            project_key=project_key,
            path=path,
            before_exists=before is not None,
            after_exists=after is not None,
            before_size=len(before or b""),
            after_size=len(after or b""),
            before_sha256=_state_sha256(before),
            after_sha256=_state_sha256(after),
            before_payload=before,
            after_payload=after,
        )
    supplied_ranges = tuple(ranges or ())
    selected = _normalise_ranges(supplied_ranges, size=len(before))
    if not selected:
        # Generic same-size operations use bounded 64 KiB spans.  Paint passes
        # exact dirty-tile rows, so sparse 4K strokes never retain a full TGA.
        selected = _changed_block_ranges(before, after)
    spans = tuple(
        MapStudioTextureSidecarSpan(start, before[start:end], after[start:end])
        for start, end in selected
        if before[start:end] != after[start:end]
    )
    if supplied_ranges:
        reconstructed = bytearray(before)
        for span in spans:
            reconstructed[span.offset : span.offset + len(span.after)] = span.after
        if bytes(reconstructed) != after:
            selected = _changed_block_ranges(before, after)
            spans = tuple(
                MapStudioTextureSidecarSpan(start, before[start:end], after[start:end])
                for start, end in selected
            )
    if not spans:
        return None
    return MapStudioTextureSidecarPatch(
        project_key=project_key,
        path=path,
        before_exists=True,
        after_exists=True,
        before_size=len(before),
        after_size=len(after),
        before_sha256=_state_sha256(before),
        after_sha256=_state_sha256(after),
        spans=spans,
    )


def _apply_patch_state(path: Path, patch: MapStudioTextureSidecarPatch, *, use_after: bool) -> None:
    current = _read_state(path)
    source_exists = patch.before_exists if use_after else patch.after_exists
    source_size = patch.before_size if use_after else patch.after_size
    source_sha256 = patch.before_sha256 if use_after else patch.after_sha256
    if (current is not None) != source_exists:
        raise RuntimeError(f"Texture sidecar changed outside Map Studio: {path}")
    if current is not None:
        if len(current) != source_size or _state_sha256(current) != source_sha256:
            raise RuntimeError(f"Texture sidecar changed outside Map Studio: {path}")
        for span in patch.spans:
            expected = span.before if use_after else span.after
            start = int(span.offset)
            if current[start : start + len(expected)] != expected:
                raise RuntimeError(f"Texture sidecar span changed outside Map Studio: {path}")
    target_exists = patch.after_exists if use_after else patch.before_exists
    full_payload = patch.after_payload if use_after else patch.before_payload
    target_size = patch.after_size if use_after else patch.before_size
    if not target_exists:
        _write_state(path, None)
        return
    if full_payload is not None:
        _atomic_write(path, full_payload)
        return
    if current is None:
        raise FileNotFoundError(f"Cannot apply texture delta because the sidecar is missing: {path}")
    if len(current) != target_size:
        raise ValueError(f"Cannot apply texture delta because {path.name} changed size outside Map Studio.")
    updated = bytearray(current)
    for span in patch.spans:
        payload = span.after if use_after else span.before
        updated[int(span.offset) : int(span.offset) + len(payload)] = payload
    _atomic_write(path, bytes(updated))


class MapStudioTextureSidecarJournal:
    """Saved baseline plus undoable external-file patches for one KMAP."""

    def __init__(self) -> None:
        self._active_key: tuple[str, str] | None = None
        self._baseline: dict[str, bytes | None] = {}
        self._tracked_paths: set[str] = set()

    @property
    def has_baseline(self) -> bool:
        return self._active_key is not None

    def clear(self) -> None:
        self._active_key = None
        self._baseline.clear()
        self._tracked_paths.clear()

    def _ensure_project(self, project: KMapProject) -> tuple[str, str]:
        key = _project_key(project)
        if self._active_key is None:
            self._active_key = key
        elif self._active_key != key:
            raise RuntimeError("Texture sidecar journal belongs to a different Map Studio project.")
        return key

    def promote(
        self,
        project: KMapProject,
        *,
        abandon_previous: bool = False,
    ) -> MapStudioTextureSidecarSnapshot:
        """Start a new lazy baseline epoch after successful Save/Open."""

        key = _project_key(project)
        if (
            self._active_key is not None
            and self._active_key != key
            and (self._baseline or self._tracked_paths)
            and not abandon_previous
        ):
            raise RuntimeError(
                "Restore or explicitly abandon the previous Map Studio texture sidecar epoch before switching projects."
            )
        self._active_key = key
        self._baseline.clear()
        self._tracked_paths.clear()
        return MapStudioTextureSidecarSnapshot(self._active_key)

    def capture(
        self,
        project: KMapProject,
        *,
        paths: Iterable[str | Path],
    ) -> MapStudioTextureSidecarSnapshot:
        """Capture only sidecars involved in the next import/paint operation."""

        key = self._ensure_project(project)
        values = {
            str(_resolve_project_path(project, value))
            for value in tuple(paths or ())
            if str(value or "").strip()
        }
        states: list[tuple[str, bytes | None]] = []
        for value in sorted(values, key=str.lower):
            state = _read_state(Path(value))
            states.append((value, state))
            self._tracked_paths.add(value)
            # Lazy one-copy baseline: the first unsaved mutation remembers the
            # last Save/Open bytes; later strokes reuse that same checkpoint.
            if value not in self._baseline:
                self._baseline[value] = state
        return MapStudioTextureSidecarSnapshot(key, tuple(states))

    def finish(
        self,
        project: KMapProject,
        before: MapStudioTextureSidecarSnapshot,
        *,
        paths: Iterable[str | Path] = (),
        created_paths: Iterable[str | Path] = (),
        ranges_by_path: dict[str, Iterable[tuple[int, int]]] | None = None,
    ) -> tuple[MapStudioTextureSidecarPatch, ...]:
        """Return changed file patches after one successful sidecar operation."""

        key = self._ensure_project(project)
        if key != before.project_key:
            raise RuntimeError("Texture sidecar transaction crossed Map Studio projects.")
        before_states = dict(before.states)
        values = set(before_states)
        values.update(
            str(_resolve_project_path(project, value))
            for value in tuple(paths or ())
            if str(value or "").strip()
        )
        created_values = {
            str(_resolve_project_path(project, value))
            for value in tuple(created_paths or ())
            if str(value or "").strip()
        }
        values.update(created_values)
        uncaptured = values.difference(before_states)
        if uncaptured:
            raise RuntimeError(
                "Texture sidecar finish paths must be captured before the operation: "
                + ", ".join(sorted(uncaptured, key=str.lower))
            )
        range_map = {
            str(_resolve_project_path(project, path_key)): value
            for path_key, value in dict(ranges_by_path or {}).items()
        }
        patches: list[MapStudioTextureSidecarPatch] = []
        for value in sorted(values, key=str.lower):
            old = before_states[value]
            new = _read_state(Path(value))
            if value in created_values and old is not None:
                raise RuntimeError(f"Declared-created texture sidecar already existed before the operation: {value}")
            if old is None and new is not None and value not in created_values:
                raise RuntimeError(f"New texture sidecar was not declared in created_paths: {value}")
            self._tracked_paths.add(value)
            if value not in self._baseline:
                # New project-managed file created by this operation.
                self._baseline[value] = old
            patch = _patch_from_states(
                key,
                value,
                old,
                new,
                ranges=range_map.get(value, ()),
            )
            if patch is not None:
                patches.append(patch)
        return tuple(patches)

    def apply(
        self,
        project: KMapProject,
        patches: Iterable[MapStudioTextureSidecarPatch],
        *,
        use_after: bool,
    ) -> int:
        """Atomically replace each patched file, rolling back on write failure."""

        rows = tuple(patches or ())
        if not rows:
            return 0
        key = self._ensure_project(project)
        if any(tuple(row.project_key) != key for row in rows):
            raise RuntimeError("Texture sidecar patch belongs to a different Map Studio project epoch.")
        paths = tuple(str(Path(row.path).resolve()) for row in rows)
        if len(set(paths)) != len(paths):
            raise RuntimeError("A texture sidecar transaction contains duplicate file patches.")
        rollback = {path: _read_state(Path(path)) for path in paths}
        # Validate every source before mutating any file, so an external edit
        # blocks the whole transaction instead of causing a partial undo/redo.
        for row, value in zip(rows, paths):
            current = rollback[value]
            source_exists = row.before_exists if use_after else row.after_exists
            source_size = row.before_size if use_after else row.after_size
            source_sha256 = row.before_sha256 if use_after else row.after_sha256
            if (current is not None) != source_exists or (
                current is not None
                and (len(current) != source_size or _state_sha256(current) != source_sha256)
            ):
                raise RuntimeError(f"Texture sidecar changed outside Map Studio: {value}")
        for value in paths:
            self._tracked_paths.add(value)
            if value not in self._baseline:
                self._baseline[value] = rollback[value]
        applied: list[str] = []
        try:
            for row, value in zip(rows, paths):
                path = Path(value)
                _apply_patch_state(path, row, use_after=use_after)
                applied.append(value)
        except Exception:
            for value in reversed(applied):
                try:
                    _write_state(Path(value), rollback[value])
                except Exception:
                    pass
            raise
        return len(rows)

    def restore_baseline(self, project: KMapProject) -> int:
        """Restore the last Save/Open bytes and remove unsaved managed files."""

        if self._active_key is None:
            return 0
        if _project_key(project) != self._active_key:
            raise RuntimeError("Cannot restore texture sidecars for a different Map Studio project epoch.")
        paths = set(self._baseline) | set(self._tracked_paths)
        patches: list[MapStudioTextureSidecarPatch] = []
        for value in sorted(paths, key=str.lower):
            current = _read_state(Path(value))
            baseline = self._baseline.get(value)
            patch = _patch_from_states(self._active_key, value, current, baseline)
            if patch is not None:
                patches.append(patch)
        return self.apply(project, patches, use_after=True)


__all__ = [
    "MapStudioTextureSidecarJournal",
    "MapStudioTextureSidecarPatch",
    "MapStudioTextureSidecarSnapshot",
    "MapStudioTextureSidecarSpan",
    "managed_project_texture_sidecars",
    "tga_dirty_tile_byte_ranges",
]
