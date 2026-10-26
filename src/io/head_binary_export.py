"""Binary MDL/MDX export and raw readback verification for modular heads."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from typing import Any, Iterable

from src.core.characters.head_donor_snapshot import (
    HeadDonorContractDiff,
    HeadDonorSnapshot,
    compare_head_donor_contract,
)
from src.core.game.kotor_loader import load_model_from_bytes
from src.core.mdl.mdl_writer import MDLBinaryWriter


_BASE = 12
_K1_MODEL_POINTERS = (4273776, 4216096)
_K2_MODEL_POINTERS = (4285200, 4216320)
_RESREF_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _f32(value: Any) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _canonical_sha256(payload: Any) -> str:
    return _sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _node_name_at(
    mdl_bytes: bytes,
    node_relative_offset: int,
) -> str:
    node_absolute = _BASE + int(node_relative_offset)
    if node_absolute < _BASE or node_absolute + 6 > len(mdl_bytes):
        raise ValueError("Node pointer is outside the MDL")
    name_index = struct.unpack_from(
        "<H",
        mdl_bytes,
        node_absolute + 4,
    )[0]
    name_table_relative = struct.unpack_from(
        "<I",
        mdl_bytes,
        _BASE + 80 + 104,
    )[0]
    name_count = struct.unpack_from(
        "<I",
        mdl_bytes,
        _BASE + 80 + 108,
    )[0]
    if name_index >= name_count:
        raise ValueError("Node name-table index is outside the table")
    pointer_absolute = (
        _BASE + name_table_relative + int(name_index) * 4
    )
    if pointer_absolute + 4 > len(mdl_bytes):
        raise ValueError("Name-table pointer is outside the MDL")
    name_relative = struct.unpack_from(
        "<I",
        mdl_bytes,
        pointer_absolute,
    )[0]
    name_start = _BASE + name_relative
    if name_start < _BASE or name_start >= len(mdl_bytes):
        raise ValueError("Node name pointer is outside the MDL")
    name_end = mdl_bytes.find(b"\0", name_start)
    if name_end < 0:
        raise ValueError("Node name is not null terminated")
    return mdl_bytes[name_start:name_end].decode("ascii")


def _safe_resref(value: str) -> str:
    resref = str(value or "").strip()
    if not resref or len(resref) > 16:
        raise ValueError("KOTOR head output ResRef must contain 1-16 characters")
    if any(char not in _RESREF_CHARS for char in resref):
        raise ValueError(
            "KOTOR head output ResRef may use only letters, digits, and underscore"
        )
    return resref


def _model_nodes(model: Any) -> list[Any]:
    return list(model.all_nodes())


def _payload_ordinals(
    snapshot: HeadDonorSnapshot,
) -> tuple[int, ...]:
    ordinals = list(snapshot.mutable_payload_node_ordinals)
    for raw in list(
        dict(snapshot.compatibility or {}).get(
            "component_payload_node_ordinals",
            (),
        )
        or ()
    ):
        ordinal = int(raw)
        if ordinal not in ordinals:
            ordinals.append(ordinal)
    if not ordinals:
        raise ValueError("Head export requires at least one mutable payload node")
    return tuple(sorted(ordinals))


def _payload_node(
    model: Any,
    snapshot: HeadDonorSnapshot,
) -> Any:
    ordinals = tuple(snapshot.mutable_payload_node_ordinals)
    if len(ordinals) != 1:
        raise ValueError("Head export requires one primary mutable donor payload node")
    nodes = _model_nodes(model)
    ordinal = ordinals[0]
    if ordinal < 0 or ordinal >= len(nodes):
        raise ValueError("Mutable donor payload ordinal is unavailable")
    return nodes[ordinal]


def _skin_rows(node: Any) -> list[list[list[float | int]]]:
    rows: list[list[list[float | int]]] = []
    for row in list(getattr(node, "skin_data", ()) or ()):
        weights = []
        influences = getattr(row, "influences", None)
        if influences is None:
            influences = getattr(row, "weights", ())
        for weight in list(influences or ()):
            value = _f32(getattr(weight, "weight", 0.0))
            if not math.isfinite(value) or value <= 1.0e-8:
                continue
            weights.append(
                [
                    int(getattr(weight, "bone_index", 0)),
                    value,
                ]
            )
        rows.append(weights)
    return rows


def _payload_facts(
    model: Any,
    snapshot: HeadDonorSnapshot,
) -> dict[str, Any]:
    nodes = _model_nodes(model)
    ordinals = _payload_ordinals(snapshot)
    if any(ordinal < 0 or ordinal >= len(nodes) for ordinal in ordinals):
        raise ValueError("A mutable component payload ordinal is unavailable")
    return {
        "nodes": [
            _node_payload_facts(nodes[ordinal], ordinal)
            for ordinal in ordinals
        ],
    }


def _node_payload_facts(node: Any, ordinal: int) -> dict[str, Any]:
    faces = [
        [int(value) for value in face]
        for face in list(getattr(node, "faces", ()) or ())
    ]
    face_mats = [
        int(value)
        for value in list(getattr(node, "face_mats", ()) or ())
    ]
    if not face_mats and faces:
        face_mats = [0] * len(faces)
    face_uvs = [
        [int(value) for value in face]
        for face in list(getattr(node, "face_uvs", ()) or ())
    ]
    if (
        not face_uvs
        and faces
        and list(getattr(node, "uvs", ()) or ())
    ):
        face_uvs = [list(face) for face in faces]
    texture = str(getattr(node, "texture", "") or "").casefold()
    texture_names = [
        str(value).casefold()
        for value in list(getattr(node, "texture_names", ()) or ())
    ]
    if not texture_names and texture:
        texture_names = [texture]
    return {
        "ordinal": int(ordinal),
        "node_name": str(getattr(node, "name", "") or ""),
        "flags": int(getattr(node, "flags", 0) or 0),
        "render": bool(getattr(node, "render", False)),
        "vertices": [
            [_f32(value) for value in vertex]
            for vertex in list(getattr(node, "vertices", ()) or ())
        ],
        "normals": [
            [_f32(value) for value in normal]
            for normal in list(getattr(node, "normals", ()) or ())
        ],
        "uvs": [
            [_f32(value) for value in uv]
            for uv in list(getattr(node, "uvs", ()) or ())
        ],
        "uvs_lm": [
            [_f32(value) for value in uv]
            for uv in list(getattr(node, "uvs_lm", ()) or ())
        ],
        "uvs_2": [
            [_f32(value) for value in uv]
            for uv in list(getattr(node, "uvs_2", ()) or ())
        ],
        "uvs_3": [
            [_f32(value) for value in uv]
            for uv in list(getattr(node, "uvs_3", ()) or ())
        ],
        "faces": faces,
        "face_mats": face_mats,
        "face_uvs": face_uvs,
        "texture": texture,
        "lightmap": str(getattr(node, "lightmap", "") or "").casefold(),
        "texture_names": texture_names,
        "tex_count": int(getattr(node, "tex_count", 1) or 1),
        "alpha": _f32(getattr(node, "alpha", 1.0)),
        "transparency_hint": int(
            getattr(node, "transparency_hint", 0) or 0
        ),
        # The installed PyKotor reader still seeks the native constraint
        # array with an extra +12 bytes.  Do not use its decoded values as a
        # readback oracle; native constraints/rest positions are verified by
        # the raw dangly contract path.
        "dangly_displacement": _f32(
            getattr(node, "dangly_displacement", 0.5)
        ),
        "dangly_tightness": _f32(
            getattr(node, "dangly_tightness", 0.5)
        ),
        "dangly_period": _f32(
            getattr(node, "dangly_period", 1.0)
        ),
        "bone_map": [
            str(value)
            for value in list(getattr(node, "bone_map", ()) or ())
        ],
        "bone_node_indices": [
            int(value)
            for value in list(
                getattr(node, "bone_node_indices", ()) or ()
            )
        ],
        "skin_rows": _skin_rows(node),
    }


@dataclass(frozen=True, slots=True)
class HeadBinaryInspection:
    """Raw/reloaded facts for one generated MDL/MDX candidate."""

    accepted: bool
    game: str
    output_resref: str
    mdl_size: int
    mdx_size: int
    mdl_sha256: str
    mdx_sha256: str
    model_function_pointers: tuple[int, int]
    geometry_root_offset: int
    attachment_root_offset: int
    geometry_root_name: str
    attachment_root_name: str
    raw_geometry_node_count: int
    raw_model_bounds: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        float,
    ]
    candidate_payload_sha256: str
    reloaded_payload_sha256: str
    donor_contract_diff: HeadDonorContractDiff
    reloaded_model: Any = field(repr=False, compare=False)
    blocking_issues: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    inspection_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.inspection_sha256:
            payload = self.to_dict()
            payload["inspection_sha256"] = ""
            object.__setattr__(
                self,
                "inspection_sha256",
                _canonical_sha256(payload),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "game": self.game,
            "output_resref": self.output_resref,
            "mdl_size": self.mdl_size,
            "mdx_size": self.mdx_size,
            "mdl_sha256": self.mdl_sha256,
            "mdx_sha256": self.mdx_sha256,
            "model_function_pointers": list(
                self.model_function_pointers
            ),
            "geometry_root_offset": self.geometry_root_offset,
            "attachment_root_offset": self.attachment_root_offset,
            "geometry_root_name": self.geometry_root_name,
            "attachment_root_name": self.attachment_root_name,
            "raw_geometry_node_count": self.raw_geometry_node_count,
            "raw_model_bounds": [
                list(self.raw_model_bounds[0]),
                list(self.raw_model_bounds[1]),
                self.raw_model_bounds[2],
            ],
            "candidate_payload_sha256": self.candidate_payload_sha256,
            "reloaded_payload_sha256": self.reloaded_payload_sha256,
            "donor_contract_diff": self.donor_contract_diff.to_dict(),
            "blocking_issues": list(self.blocking_issues),
            "warnings": list(self.warnings),
            "inspection_sha256": self.inspection_sha256,
        }


@dataclass(frozen=True, slots=True)
class HeadBinaryBuild:
    """Verified in-memory export bytes and their inspection."""

    mdl_bytes: bytes = field(repr=False)
    mdx_bytes: bytes = field(repr=False)
    output_model: Any = field(repr=False, compare=False)
    inspection: HeadBinaryInspection


@dataclass(frozen=True, slots=True)
class HeadBinaryExportResult:
    """Files written only after binary inspection succeeds."""

    mdl_path: str
    mdx_path: str
    manifest_path: str
    inspection: HeadBinaryInspection


def build_verified_head_binary(
    model: Any,
    *,
    donor_snapshot: HeadDonorSnapshot,
    game: str,
    output_resref: str,
    allowed_local_animation_names: Iterable[str] = (),
) -> HeadBinaryBuild:
    """Write to memory, inspect raw headers, reload, and compare contracts."""

    normalized_game = str(game or "").upper()
    if normalized_game not in {"K1", "K2"}:
        raise ValueError("Head binary export game must be K1 or K2")
    resref = _safe_resref(output_resref)
    output_model = deepcopy(model)
    output_model.name = resref
    root = getattr(output_model, "root_node", None)
    if root is None:
        raise ValueError("Head binary export requires a geometry root")
    root.name = resref

    blocking: list[str] = []
    warnings: list[str] = []
    mdl_bytes, mdx_bytes = MDLBinaryWriter().write(output_model)
    if len(mdl_bytes) < _BASE + 196:
        blocking.append("Generated MDL is smaller than the model header.")
    declared_mdl = struct.unpack_from("<I", mdl_bytes, 4)[0]
    declared_mdx = struct.unpack_from("<I", mdl_bytes, 8)[0]
    if declared_mdl != len(mdl_bytes) - _BASE:
        blocking.append("MDL file-header size does not match generated bytes.")
    if declared_mdx != len(mdx_bytes):
        blocking.append("MDX file-header size does not match generated bytes.")

    function_pointers = struct.unpack_from("<II", mdl_bytes, _BASE)
    expected_pointers = (
        _K2_MODEL_POINTERS
        if normalized_game == "K2"
        else _K1_MODEL_POINTERS
    )
    if tuple(function_pointers) != expected_pointers:
        blocking.append(
            "Model function-pointer family does not match the target game."
        )
    geometry_root = struct.unpack_from(
        "<I",
        mdl_bytes,
        _BASE + 40,
    )[0]
    attachment_root = struct.unpack_from(
        "<I",
        mdl_bytes,
        _BASE + 80 + 88,
    )[0]
    if not geometry_root or not attachment_root:
        blocking.append(
            "Geometry-root and attachment-root pointers must both be nonzero."
        )
    if geometry_root == attachment_root:
        blocking.append(
            "Geometry-root and attachment-root pointers must remain distinct."
        )
    geometry_root_name = ""
    attachment_root_name = ""
    try:
        geometry_root_name = _node_name_at(mdl_bytes, geometry_root)
        attachment_root_name = _node_name_at(mdl_bytes, attachment_root)
    except Exception as exc:
        blocking.append(f"Raw root/name-table verification failed: {exc}")
    if geometry_root_name.casefold() != resref.casefold():
        blocking.append("Raw geometry root does not use the output ResRef.")
    if attachment_root_name.casefold() != (
        donor_snapshot.attachment_target_name.casefold()
    ):
        blocking.append(
            "Raw attachment root no longer names the donor attachment target."
        )
    raw_node_count = struct.unpack_from(
        "<I",
        mdl_bytes,
        _BASE + 44,
    )[0]
    if raw_node_count != donor_snapshot.inherited_node_declaration:
        blocking.append(
            "Raw inherited geometry-node declaration changed from the donor."
        )
    bounds_offset = _BASE + 80 + 24
    raw_bounds = (
        tuple(struct.unpack_from("<fff", mdl_bytes, bounds_offset)),
        tuple(struct.unpack_from("<fff", mdl_bytes, bounds_offset + 12)),
        struct.unpack_from("<f", mdl_bytes, bounds_offset + 24)[0],
    )
    expected_bounds = (
        tuple(_f32(value) for value in donor_snapshot.retail_bb_min),
        tuple(_f32(value) for value in donor_snapshot.retail_bb_max),
        _f32(donor_snapshot.retail_radius),
    )
    if raw_bounds != expected_bounds:
        blocking.append("Raw retail model bounds changed during export.")

    mdx_buffer_offset = struct.unpack_from(
        "<I",
        mdl_bytes,
        _BASE + 80 + 92,
    )[0]
    raw_mdx_size = struct.unpack_from(
        "<I",
        mdl_bytes,
        _BASE + 80 + 96,
    )[0]
    if mdx_buffer_offset > len(mdx_bytes):
        blocking.append("Raw MDX buffer offset is outside the companion file.")
    if raw_mdx_size > len(mdx_bytes) - mdx_buffer_offset:
        blocking.append("Raw MDX buffer size is outside the companion file.")

    allowed_animation_names = tuple(
        str(value or "").strip()
        for value in allowed_local_animation_names
        if str(value or "").strip()
    )
    reloaded = load_model_from_bytes(mdl_bytes, mdx_bytes)
    if reloaded is None:
        raise ValueError("Generated MDL/MDX could not be reloaded")
    donor_diff = compare_head_donor_contract(
        donor_snapshot,
        reloaded,
        output_resref=resref,
        allowed_local_animation_names=allowed_animation_names,
    )
    if donor_diff.blocking:
        blocking.append(
            "Reloaded MDL/MDX has blocking donor-contract differences."
        )
    candidate_facts = _payload_facts(output_model, donor_snapshot)
    reloaded_facts = _payload_facts(reloaded, donor_snapshot)
    candidate_hash = _canonical_sha256(candidate_facts)
    reloaded_hash = _canonical_sha256(reloaded_facts)
    if candidate_hash != reloaded_hash:
        blocking.append(
            "Reloaded rendered geometry, UV, material, or skin payload changed."
        )
    expected_animation_names = list(donor_snapshot.local_animation_names)
    expected_animation_keys = {
        value.casefold() for value in expected_animation_names
    }
    for value in allowed_animation_names:
        if value.casefold() not in expected_animation_keys:
            expected_animation_names.append(value)
            expected_animation_keys.add(value.casefold())
    if tuple(
        str(getattr(animation, "name", "") or "")
        for animation in list(getattr(reloaded, "animations", ()) or ())
    ) != tuple(expected_animation_names):
        blocking.append(
            "Reloaded local animation inventory differs from the donor plus "
            "the explicitly allowed overrides."
        )

    inspection = HeadBinaryInspection(
        accepted=not blocking,
        game=normalized_game,
        output_resref=resref,
        mdl_size=len(mdl_bytes),
        mdx_size=len(mdx_bytes),
        mdl_sha256=_sha256(mdl_bytes),
        mdx_sha256=_sha256(mdx_bytes),
        model_function_pointers=tuple(function_pointers),
        geometry_root_offset=geometry_root,
        attachment_root_offset=attachment_root,
        geometry_root_name=geometry_root_name,
        attachment_root_name=attachment_root_name,
        raw_geometry_node_count=raw_node_count,
        raw_model_bounds=raw_bounds,
        candidate_payload_sha256=candidate_hash,
        reloaded_payload_sha256=reloaded_hash,
        donor_contract_diff=donor_diff,
        reloaded_model=reloaded,
        blocking_issues=tuple(blocking),
        warnings=tuple(warnings),
    )
    return HeadBinaryBuild(
        mdl_bytes=mdl_bytes,
        mdx_bytes=mdx_bytes,
        output_model=output_model,
        inspection=inspection,
    )


def write_verified_head_binary(
    build: HeadBinaryBuild,
    *,
    output_dir: str | Path,
    overwrite: bool = False,
    manifest_metadata: dict[str, Any] | None = None,
) -> HeadBinaryExportResult:
    """Atomically publish a previously verified MDL/MDX and manifest."""

    if not build.inspection.accepted:
        raise ValueError("Blocked head binary build cannot be written")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stem = build.inspection.output_resref
    mdl_path = destination / f"{stem}.mdl"
    mdx_path = destination / f"{stem}.mdx"
    manifest_path = destination / f"{stem}.head-export.json"
    targets = (mdl_path, mdx_path, manifest_path)
    if not overwrite and any(path.exists() for path in targets):
        raise FileExistsError(
            "Head export target already exists; explicit overwrite is required"
        )
    manifest = {
        "schema": "ghostrigger.head_binary_export",
        "version": 1,
        "inspection": build.inspection.to_dict(),
        "metadata": dict(manifest_metadata or {}),
    }
    payloads = {
        mdl_path: build.mdl_bytes,
        mdx_path: build.mdx_bytes,
        manifest_path: (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    }
    temporary: list[Path] = []
    try:
        for path, data in payloads.items():
            temp = path.with_name(f".{path.name}.tmp")
            with temp.open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.append(temp)
        for path in targets:
            os.replace(path.with_name(f".{path.name}.tmp"), path)
        temporary.clear()
    finally:
        for path in temporary:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    return HeadBinaryExportResult(
        mdl_path=str(mdl_path),
        mdx_path=str(mdx_path),
        manifest_path=str(manifest_path),
        inspection=build.inspection,
    )


__all__ = [
    "HeadBinaryBuild",
    "HeadBinaryExportResult",
    "HeadBinaryInspection",
    "build_verified_head_binary",
    "write_verified_head_binary",
]
