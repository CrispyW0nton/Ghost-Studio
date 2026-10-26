"""
kotor_loader.py  —  Direct PyKotor loader for GhostRigger.

Binary MDL/MDX loads use ``read_mdl_safe`` (``mdl_reader_wrapper``), which
routes through GhostRigger's owned binary reader for K2 layout fixes without
mutating PyKotor global state.  ASCII MDL and TPC handling still call PyKotor
directly.

Public surface
--------------
  load_model_from_bytes(mdl, mdx, game)  → KotorModel
  load_model_from_file(path, mdx, game)  → KotorModel
  load_tpc_as_pil(data)                  → PIL.Image | None
  patch_tpc_header(data)                 → bytes          (fix data_sz=0)

PyKotor bone-weight contract (MDLSkin):
  bonemap[local_idx]                      = MDLNode.node_id  (float32)
  vertex_bones[vi].vertex_indices[j]      = local_idx  (-1.0 = unused)
  vertex_bones[vi].vertex_weights[j]      = blend weight
  → BoneWeight.bone_index = local_idx  (direct index into KotorModel bone_map)
"""

import math
import struct
import logging
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# ── PyKotor imports (installed via pip install pykotor) ──────────────────────
from ..mdl.mdl_reader_wrapper import read_mdl_safe as pk_read_mdl
from pykotor.resource.formats.mdl.mdl_data import (
    MDLNodeType,
    MDLNodeFlags,
)
from pykotor.resource.formats.mdl.mdl_types import MDLControllerType
from pykotor.resource.formats.tpc.tpc_auto import read_tpc as pk_read_tpc
from pykotor.resource.formats.tpc.tpc_data import TPCTextureFormat

# ── PIL (optional, only needed for TPC → PIL) ─────────────────────────────────
try:
    from PIL import Image as _PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# ── GhostRigger model types ───────────────────────────────────────────────────
from ..geometry.model_data import (
    Animation, AnimEvent, BoneWeight, GameVersion,
    KotorModel, ModelNode, NodeFlags, ResolvedAnimationSlot,
    SupermodelChain, SupermodelChainEntry, VertexSkinData,
)
from .import_normalisation import apply_known_skin_bone_map_normalisations

# ── MDLNodeType → NodeFlags ───────────────────────────────────────────────────
# These flag values must match what the old pykotor_bridge.py produced, since
# the rest of the codebase (viewport.py, model_data.py) was built against them.
#
# Key differences vs a naive mapping:
#   DUMMY  → NodeFlags.HEADER (0x01) for ALL dummy nodes (root AND children).
#            Old bridge always assigned HEADER to dummies; giving child dummies
#            flags=0 broke is_dummy checks throughout the pipeline.
#   SKIN   → NodeFlags.HEADER|MESH|SKIN (0x61), matching the KotOR binary MDL
#            format (confirmed via c_bantha raw binary scan).  Skin nodes
#            contain vertex data like trimesh nodes and must have MESH set so
#            that LBS and texture-pipeline code can locate them via is_mesh.
#            is_dummy stays False because HEADER|MESH|SKIN (0x61) ≠ 0x01.
#            Viewport code uses 'if n.is_skin: skip' paths separately.
#   TRIMESH→ NodeFlags.HEADER|MESH (0x21), matching the raw binary MDL flags
#            used by stock KOTOR models.  Reloaded models must preserve this
#            header bit so a write-after-read does not emit bare 0x20 helpers.
_TYPE_FLAGS: Dict[int, int] = {
    int(MDLNodeType.DUMMY):      int(NodeFlags.HEADER),
    int(MDLNodeType.TRIMESH):    int(NodeFlags.HEADER) | int(NodeFlags.MESH),
    int(MDLNodeType.DANGLYMESH): int(NodeFlags.HEADER) | int(NodeFlags.MESH) | int(NodeFlags.DANGLY),
    int(MDLNodeType.LIGHT):      int(NodeFlags.HEADER) | int(NodeFlags.LIGHT),
    int(MDLNodeType.EMITTER):    int(NodeFlags.HEADER) | int(NodeFlags.EMITTER),
    int(MDLNodeType.REFERENCE):  int(NodeFlags.HEADER) | int(NodeFlags.REFERENCE),
    int(MDLNodeType.AABB):       int(NodeFlags.HEADER) | int(NodeFlags.AABB),
    int(MDLNodeType.SKIN):       int(NodeFlags.HEADER) | int(NodeFlags.MESH) | int(NodeFlags.SKIN),
    int(MDLNodeType.SABER):      int(NodeFlags.HEADER) | int(NodeFlags.SABER) | int(NodeFlags.MESH),
    int(MDLNodeType.CAMERA):     int(NodeFlags.HEADER),
    int(MDLNodeType.PATCH):      int(NodeFlags.HEADER) | int(NodeFlags.MESH),
    int(MDLNodeType.BINARY):     int(NodeFlags.HEADER),
}

# ── Controller type ints (from MDLControllerType enum) ───────────────────────
_CT_POS    = int(MDLControllerType.POSITION)     # 8
_CT_ORI    = int(MDLControllerType.ORIENTATION)  # 20
_CT_SCALE  = int(MDLControllerType.SCALE)        # 36
_CT_COLOR  = int(MDLControllerType.COLOR)        # 76
_CT_RADIUS = int(MDLControllerType.RADIUS)       # 88
_CT_ALPHA  = int(MDLControllerType.ALPHA)        # 132
_CT_MULT   = int(MDLControllerType.MULTIPLIER)   # 140
_CT_ILLUM  = int(MDLControllerType.SELFILLUMCOLOR)  # 100


# =============================================================================
#  Public API
# =============================================================================

def load_model_from_bytes(
    mdl_bytes: bytes,
    mdx_bytes: bytes = b'',
    game_version: Optional[GameVersion] = None,
) -> KotorModel:
    """Parse a KotOR MDL+MDX from raw bytes using PyKotor.

    Reads every field PyKotor exposes — vertices, normals, tangents,
    primary + secondary UVs, per-face UV indices, all mesh faces,
    bone-map, per-vertex skin weights, bind-pose controllers, and
    every animation with its keyframe data.

    Returns None if data is invalid or PyKotor fails to parse it.
    """
    if not mdl_bytes or len(mdl_bytes) < 12:
        log.error("load_model_from_bytes: MDL data too small (%d bytes)", len(mdl_bytes))
        return None

    # Detect game version from function pointer in header (bytes 12-15) before
    # passing to PyKotor — PyKotor doesn't expose the detected game on the MDL object.
    detected_version = game_version or _detect_version_from_bytes(mdl_bytes)

    # Read raw classification byte (BASE+80 = offset 92 in MDL binary).
    # Map it using GhostRigger's scheme.  PyKotor uses a different enum — so
    # we read the byte ourselves and patch unknown values to 4 (CHARACTER=4)
    # before passing to PyKotor to avoid MDLClassification ValueError.
    _KNOWN_PK_CLS = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}
    _CLS_MAP_BYTES: Dict[int, str] = {
        0: 'effect', 1: 'effects', 2: 'tile', 4: 'character',
        8: 'door', 16: 'lightsaber', 32: 'placeable', 64: 'flyer',
    }
    raw_cls_byte: Optional[int] = None
    patched_bytes = mdl_bytes
    if len(mdl_bytes) >= 93:
        raw_cls_byte = struct.unpack_from('B', mdl_bytes, 92)[0]
        if raw_cls_byte not in _KNOWN_PK_CLS:
            log.debug("load_model_from_bytes: unknown cls byte=%d → defaulting to character(4)",
                      raw_cls_byte)
            buf = bytearray(mdl_bytes)
            buf[92] = 4  # patch to CHARACTER
            patched_bytes = bytes(buf)

    try:
        pk_mdl = pk_read_mdl(patched_bytes, source_ext=mdx_bytes if mdx_bytes else None)
        model  = _mdl_to_kotormodel(pk_mdl, detected_version)
        _apply_raw_supernode_numbers(model, mdl_bytes)
        _apply_raw_super_root_link(model, mdl_bytes)
        _apply_raw_mesh_header_counts(model, mdl_bytes, mdx_bytes)
        # Override classification from raw byte if known
        if raw_cls_byte is not None:
            model.classification = _CLS_MAP_BYTES.get(raw_cls_byte, 'character')
            model.model_type     = raw_cls_byte

        log.debug("load_model_from_bytes: '%s'  nodes=%d  anims=%d",
                  model.name, len(model.all_nodes()), len(model.animations))
        return model
    except Exception as exc:
        log.error("load_model_from_bytes: parse failed — %s", exc, exc_info=True)
        return None


def load_model_from_file(
    mdl_path: str,
    mdx_path: str = '',
    game_version: Optional[GameVersion] = None,
) -> KotorModel:
    """Parse a KotOR MDL file (+ optional MDX) using PyKotor.

    Returns None on missing file or parse failure.
    """
    p = Path(mdl_path)
    if not p.exists():
        log.error("load_model_from_file: not found: %s", mdl_path)
        return None

    mdx_p: Optional[Path] = None
    if mdx_path:
        mp = Path(mdx_path)
        if mp.exists():
            mdx_p = mp
    else:
        guess = p.with_suffix('.mdx')
        if guess.exists():
            mdx_p = guess

    # Detect game version from function pointer before passing to PyKotor
    mdl_bytes_raw = p.read_bytes()
    mdx_bytes_raw = mdx_p.read_bytes() if mdx_p is not None else b''
    detected_version = game_version or _detect_version_from_bytes(mdl_bytes_raw)

    _CLS_MAP_FILE: Dict[int, str] = {
        0: 'effect', 1: 'effects', 2: 'tile', 4: 'character',
        8: 'door', 16: 'lightsaber', 32: 'placeable', 64: 'flyer',
    }
    raw_cls_byte_f: Optional[int] = None
    if len(mdl_bytes_raw) >= 93:
        raw_cls_byte_f = struct.unpack_from('B', mdl_bytes_raw, 92)[0]

    try:
        pk_mdl = pk_read_mdl(p, source_ext=mdx_p)
        model  = _mdl_to_kotormodel(pk_mdl, detected_version)
        _apply_raw_supernode_numbers(model, mdl_bytes_raw)
        _apply_raw_super_root_link(model, mdl_bytes_raw)
        _apply_raw_mesh_header_counts(model, mdl_bytes_raw, mdx_bytes_raw)
        # Override classification from raw byte for accuracy
        if raw_cls_byte_f is not None:
            model.classification = _CLS_MAP_FILE.get(raw_cls_byte_f, 'character')
            model.model_type     = raw_cls_byte_f
        model.mdl_path = str(p)
        model.mdx_path = str(mdx_p) if mdx_p else ''

        log.debug("load_model_from_file: '%s'  nodes=%d  anims=%d",
                  model.name, len(model.all_nodes()), len(model.animations))
        return model
    except Exception as exc:
        log.error("load_model_from_file: '%s' — %s", mdl_path, exc, exc_info=True)
        return None


def _apply_raw_supernode_numbers(
    model: Optional[KotorModel],
    mdl_bytes: bytes,
) -> None:
    """Recover node-header +2 identities that PyKotor does not expose.

    The field is a sparse supermodel-node identity, not a DFS array index.
    Character animation matching and modular-head attachment compare it
    directly. PyKotor preserves the +4 name-table index as ``node_id`` but
    drops +2, so retain the latter in ``ModelNode.number`` for round-tripping.
    """

    if model is None or len(mdl_bytes) < 92:
        return
    base = 12
    try:
        root_rel = struct.unpack_from("<I", mdl_bytes, base + 40)[0]
        geometry_node_count = struct.unpack_from(
            "<I", mdl_bytes, base + 44
        )[0]
    except struct.error:
        return
    # Preserve the raw field rather than replacing it with the number of
    # locally traversed records. Supermodel-derived characters use a
    # cumulative inheritance-chain span here (PFHA04: 564 declared / 38
    # local), which retail needs when it follows a modular neck_g link.
    model.geometry_node_count = int(geometry_node_count)
    if root_rel <= 0:
        return

    number_by_name_index: Dict[int, int] = {}
    queue = [int(root_rel)]
    seen: set[int] = set()
    while queue:
        node_rel = queue.pop(0)
        if node_rel in seen:
            continue
        seen.add(node_rel)
        node_abs = base + node_rel
        if node_abs < base or node_abs + 80 > len(mdl_bytes):
            continue
        try:
            supernode_number = struct.unpack_from("<H", mdl_bytes, node_abs + 2)[0]
            name_index = struct.unpack_from("<H", mdl_bytes, node_abs + 4)[0]
            child_array_rel = struct.unpack_from("<I", mdl_bytes, node_abs + 44)[0]
            child_count = struct.unpack_from("<I", mdl_bytes, node_abs + 48)[0]
        except struct.error:
            continue
        number_by_name_index[int(name_index)] = int(supernode_number)
        child_array_abs = base + int(child_array_rel)
        if (
            child_array_rel <= 0
            or child_count <= 0
            or child_count > 0x10000
            or child_array_abs + int(child_count) * 4 > len(mdl_bytes)
        ):
            continue
        for child_index in range(int(child_count)):
            child_rel = struct.unpack_from(
                "<I", mdl_bytes, child_array_abs + child_index * 4
            )[0]
            if child_rel > 0:
                queue.append(int(child_rel))

    if not number_by_name_index:
        return
    for node in model.all_nodes():
        name_index = int(getattr(node, "index", -1) or 0)
        if name_index in number_by_name_index:
            node.number = number_by_name_index[name_index]
    for animation in list(getattr(model, "animations", []) or []):
        for node in list(getattr(animation, "nodes", []) or []):
            name_index = int(getattr(node, "index", -1) or 0)
            if name_index in number_by_name_index:
                node.number = number_by_name_index[name_index]
    setattr(model, "_gr_native_supernode_numbers", dict(number_by_name_index))
    # A complete, collision-free binary identity map is safe to round-trip
    # through the full writer.  Without this opt-in the writer intentionally
    # falls back to dense DFS numbering, which breaks stock modular heads such
    # as PFHA04 even though their sparse +2 identities were read correctly.
    geometry_nodes = list(model.all_nodes())
    geometry_numbers = [
        int(getattr(node, "number", -1))
        for node in geometry_nodes
    ]
    model.preserve_native_supernode_numbers = bool(
        geometry_nodes
        and len(number_by_name_index) == len(geometry_nodes)
        and all(0 <= value <= 0xFFFF for value in geometry_numbers)
        and len(set(geometry_numbers)) == len(geometry_numbers)
    )


def _apply_raw_super_root_link(
    model: Optional[KotorModel],
    mdl_bytes: bytes,
) -> None:
    """Preserve a non-root model-header attachment link.

    The geometry header and model header normally point at the same root node.
    Stock modular player heads deliberately differ: the geometry root remains
    the AuroraBase, while ``offset_to_super_root`` points at ``neck_g``. This is
    the binary contract historically restored by VarsityPuppet's HeadFixer.
    PyKotor does not expose the second pointer, so recover its target by raw
    node offset for lossless Ghost Studio read/write behavior.
    """

    if model is None or len(mdl_bytes) < 12 + 196:
        return
    base = 12
    try:
        geometry_root_rel = int(
            struct.unpack_from("<I", mdl_bytes, base + 40)[0]
        )
        super_root_rel = int(
            struct.unpack_from("<I", mdl_bytes, base + 80 + 88)[0]
        )
    except struct.error:
        return
    if (
        geometry_root_rel <= 0
        or super_root_rel <= 0
        or super_root_rel == geometry_root_rel
    ):
        model.super_root_node_name = ""
        return

    target_abs = base + super_root_rel
    if target_abs < base or target_abs + 80 > len(mdl_bytes):
        log.warning(
            "MDL offset_to_super_root 0x%X lies outside the geometry data",
            super_root_rel,
        )
        return
    try:
        name_index = int(struct.unpack_from("<H", mdl_bytes, target_abs + 4)[0])
        name_table_rel = int(
            struct.unpack_from("<I", mdl_bytes, base + 80 + 104)[0]
        )
        name_count = int(
            struct.unpack_from("<I", mdl_bytes, base + 80 + 108)[0]
        )
        if not 0 <= name_index < name_count:
            return
        name_offset_rel = int(
            struct.unpack_from(
                "<I",
                mdl_bytes,
                base + name_table_rel + name_index * 4,
            )[0]
        )
        name_start = base + name_offset_rel
        name_end = mdl_bytes.find(b"\0", name_start)
        if name_start < base or name_end < name_start:
            return
        target_name = mdl_bytes[name_start:name_end].decode(
            "ascii", errors="replace"
        )
    except (struct.error, ValueError):
        return

    matches = [
        node
        for node in model.all_nodes()
        if str(getattr(node, "name", "") or "").casefold()
        == target_name.casefold()
    ]
    if len(matches) != 1:
        log.warning(
            "MDL offset_to_super_root target %r matched %d converted nodes",
            target_name,
            len(matches),
        )
        return
    model.super_root_node_name = str(matches[0].name)
    setattr(
        model,
        "_gr_raw_super_root_link",
        {
            "geometry_root_offset": geometry_root_rel,
            "super_root_offset": super_root_rel,
            "node": str(matches[0].name),
        },
    )


def _apply_raw_mesh_header_counts(
    model: Optional[KotorModel],
    mdl_bytes: bytes,
    mdx_bytes: bytes = b'',
) -> None:
    """Preserve raw mesh header fields lost by PyKotor's object model."""

    if model is None or not mdl_bytes:
        return
    try:
        from ..mdl.ghostrigger_mdl_reader import GhostRiggerMDLBinaryReader

        reader = GhostRiggerMDLBinaryReader(
            mdl_bytes,
            0,
            len(mdl_bytes),
            mdx_bytes or b'',
            0,
            len(mdx_bytes or b''),
        )
        reader.load()
        raw_mesh_headers: Dict[int, Dict[str, object]] = {}
        for _offset, bin_node in sorted((getattr(reader, "_gr_bin_nodes", None) or {}).items()):
            header = getattr(bin_node, "header", None)
            trimesh = getattr(bin_node, "trimesh", None)
            if header is None or trimesh is None:
                continue
            try:
                node_id = int(getattr(header, "node_id"))
            except Exception:
                continue
            raw_mesh_headers.setdefault(
                node_id,
                {
                    "tex_count": max(0, int(getattr(trimesh, "texture_count", 0) or 0)),
                    "has_lightmap": bool(getattr(trimesh, "has_lightmap", False)),
                    "rotate_texture": bool(getattr(trimesh, "rotate_texture", False)),
                    "background_geometry": bool(getattr(trimesh, "background", False)),
                    "has_shadow": bool(getattr(trimesh, "has_shadow", False)),
                    "beaming": bool(getattr(trimesh, "beaming", False)),
                    "render": bool(getattr(trimesh, "render", True)),
                    "transparency_hint": int(getattr(trimesh, "transparency_hint", 0) or 0),
                    "diffuse": (
                        float(getattr(getattr(trimesh, "diffuse", None), "x", 1.0)),
                        float(getattr(getattr(trimesh, "diffuse", None), "y", 1.0)),
                        float(getattr(getattr(trimesh, "diffuse", None), "z", 1.0)),
                    ),
                    "ambient": (
                        float(getattr(getattr(trimesh, "ambient", None), "x", 1.0)),
                        float(getattr(getattr(trimesh, "ambient", None), "y", 1.0)),
                        float(getattr(getattr(trimesh, "ambient", None), "z", 1.0)),
                    ),
                    "dirt_enabled": bool(getattr(trimesh, "dirt_enabled", False)),
                    "dirt_texture": int(getattr(trimesh, "dirt_texture", 0) or 0),
                    "dirt_coord_space": int(getattr(trimesh, "dirt_worldspace", 0) or 0),
                    "hide_in_holograms": bool(getattr(trimesh, "hologram_donotdraw", False)),
                    "mesh_indices_counts": [
                        int(value) for value in (getattr(trimesh, "indices_counts", []) or [])
                    ],
                    "mesh_inverted_counters": [
                        int(value) for value in (getattr(trimesh, "inverted_counters", []) or [])
                    ],
                },
            )
        if not raw_mesh_headers:
            return
        for node in model.all_nodes():
            try:
                node_id = int(getattr(node, "index"))
            except Exception:
                continue
            raw_header = raw_mesh_headers.get(node_id)
            if not raw_header:
                continue
            tex_count = int(raw_header["tex_count"])
            node.tex_count = tex_count
            setattr(node, "_gr_raw_tex_count", tex_count)
            for attr in (
                "has_lightmap",
                "rotate_texture",
                "background_geometry",
                "has_shadow",
                "beaming",
                "render",
                "transparency_hint",
                "diffuse",
                "ambient",
                "dirt_enabled",
                "dirt_texture",
                "dirt_coord_space",
                "hide_in_holograms",
                "mesh_indices_counts",
                "mesh_inverted_counters",
            ):
                setattr(node, attr, raw_header[attr])
            setattr(node, "_gr_raw_mesh_header", dict(raw_header))
    except Exception as exc:
        log.debug("raw mesh header preservation failed: %s", exc, exc_info=True)


def _game_name(game: Optional[object], model: Optional[KotorModel] = None) -> str:
    """Return the KOTOR game name expected by ResourceManager-style loaders."""

    if game is not None:
        raw = getattr(game, "value", game)
    elif model is not None:
        raw = getattr(getattr(model, "game_version", None), "value", getattr(model, "game_version", "K1"))
    else:
        raw = "K1"
    text = str(raw or "K1").upper()
    if text in {"2", "K2", "TSL"}:
        return "K2"
    return "K1"


def _is_null_supermodel(resref: Optional[str]) -> bool:
    return not resref or str(resref).strip().lower() in {"", "null", "none"}


def _configure_supermodel_resource_manager(resource_manager: object | None) -> None:
    if resource_manager is None:
        return
    from ..animation.animation_engine import SuperModelResolver

    SuperModelResolver.configure(resource_manager)


def load_supermodel_chain(
    model: KotorModel,
    *,
    game: Optional[object] = None,
    resource_manager: object | None = None,
) -> SupermodelChain:
    """Resolve model.supermodel metadata in chain order.

    This is a read-only diagnostic/API helper for retargeting and export
    planning. It uses the same :class:`SuperModelResolver` cache and resource
    manager as viewport animation playback, so local animation slot decisions
    are made against the same inheritance chain users will preview.
    """

    from ..animation.animation_engine import SuperModelResolver

    _configure_supermodel_resource_manager(resource_manager)
    chain = SupermodelChain(root_model_name=str(getattr(model, "name", "") or ""))
    game_name = _game_name(game, model)
    visited = {str(getattr(model, "name", "") or "").lower()}
    super_ref = str(getattr(model, "supermodel", "") or "")

    while not _is_null_supermodel(super_ref):
        key = super_ref.lower()
        if key in visited:
            log.warning("load_supermodel_chain: cycle detected at %r", super_ref)
            break
        visited.add(key)

        super_model = SuperModelResolver.load_supermodel(super_ref, game_name)
        if super_model is None:
            chain.entries.append(
                SupermodelChainEntry(resref=super_ref, loaded=False)
            )
            break

        chain.entries.append(
            SupermodelChainEntry(
                resref=super_ref,
                model_name=str(getattr(super_model, "name", "") or super_ref),
                supermodel=str(getattr(super_model, "supermodel", "") or "NULL"),
                anim_scale=float(getattr(super_model, "anim_scale", 1.0) or 1.0),
                loaded=True,
            )
        )
        super_ref = str(getattr(super_model, "supermodel", "") or "")

    return chain


def get_valid_animation_slots(
    model: KotorModel,
    *,
    game: Optional[object] = None,
    resource_manager: object | None = None,
) -> List[str]:
    """Return local and inherited animation slot names, local overrides first."""

    from ..animation.animation_engine import SuperModelResolver

    _configure_supermodel_resource_manager(resource_manager)
    return [
        name
        for name, _source, _scale in SuperModelResolver.list_all_animations(
            model,
            _game_name(game, model),
        )
    ]


def resolve_animation_slot(
    model: KotorModel,
    slot_name: str,
    *,
    game: Optional[object] = None,
    resource_manager: object | None = None,
    require_valid: bool = False,
) -> ResolvedAnimationSlot:
    """Resolve an animation slot through local-first supermodel inheritance.

    A local animation block with the same name as an inherited supermodel slot
    is reported as ``inherited=False`` and is the selected override. When
    ``require_valid`` is true, an unresolved slot raises ``ValueError`` so CLI
    export flows can stop before producing an in-game-invalid patch.
    """

    from ..animation.animation_engine import SuperModelResolver

    _configure_supermodel_resource_manager(resource_manager)
    wanted = str(slot_name or "").strip()
    if not wanted:
        raise ValueError("Animation slot name cannot be empty")

    game_name = _game_name(game, model)
    animation, scale = SuperModelResolver.resolve_animation(model, wanted, game_name)
    source = ""
    inherited = False

    if animation is not None:
        wanted_key = wanted.lower()
        local_names = {anim.name.lower() for anim in getattr(model, "animations", [])}
        inherited = wanted_key not in local_names
        for name, source_name, _entry_scale in SuperModelResolver.list_all_animations(model, game_name):
            if name.lower() == wanted_key:
                source = source_name
                break
        source = source or (getattr(model, "name", "") if not inherited else "")
    elif require_valid:
        raise ValueError(f"Animation slot '{wanted}' is not available on {getattr(model, 'name', 'model')}")

    return ResolvedAnimationSlot(
        slot_name=getattr(animation, "name", wanted) if animation is not None else wanted,
        animation=animation,
        source_model_name=str(source or ""),
        inherited=inherited,
        cumulative_scale=float(scale or 1.0),
        transtime=float(getattr(animation, "transition_time", 0.25) if animation is not None else 0.25),
        anim_root=str(getattr(animation, "anim_root", "") if animation is not None else ""),
        events=list(getattr(animation, "events", []) if animation is not None else []),
    )


def patch_tpc_header(data: bytes) -> bytes:
    """Patch TPC header data_sz=0 for stock KotOR DXT1/DXT5 files.

    Stock KotOR DXT TPC files store data_sz=0.  PyKotor needs a valid value
    to find the TXI trailer.  We compute it and patch in-place (copy only).
    Uncompressed RGB/RGBA files also use data_sz=0, so only patch when the
    payload is too small to contain uncompressed texels.
    """
    if len(data) < 128:
        return data
    if struct.unpack_from('<I', data, 0)[0] != 0:
        return data          # already valid

    width  = struct.unpack_from('<H', data, 8)[0]
    height = struct.unpack_from('<H', data, 10)[0]
    enc    = struct.unpack_from('B',  data, 12)[0]
    mips   = struct.unpack_from('B',  data, 13)[0]
    if enc not in (2, 4) or width == 0 or height == 0:
        return data          # uncompressed or invalid — leave alone

    bpp = 3 if enc == 2 else 4
    total_uncompressed = 0
    uw, uh = width, height
    for _ in range(max(1, mips)):
        total_uncompressed += max(1, uw) * max(1, uh) * bpp
        uw = max(1, uw >> 1)
        uh = max(1, uh >> 1)
    if len(data) >= 128 + total_uncompressed:
        return data          # enc=2/4 with full texel payload is RGB/RGBA, not DXT

    total = 0
    w, h  = width, height
    for _ in range(max(1, mips)):
        bw     = max(1, (w + 3) // 4)
        bh     = max(1, (h + 3) // 4)
        bsz    = 8 if enc == 2 else 16   # DXT1=8 B/block, DXT5=16 B/block
        total += bw * bh * bsz
        w = max(1, w >> 1)
        h = max(1, h >> 1)

    buf = bytearray(data)
    struct.pack_into('<I', buf, 0, total)
    return bytes(buf)


def load_tpc_as_pil(data: bytes):
    """Decode KotOR TPC/TGA/DDS bytes to a PIL RGBA Image using PyKotor.

    Uses pykotor.resource.formats.tpc directly — the same code that the
    PyKotor OpenGL renderer (pykotor.gl.shader.texture.Texture.from_tpc)
    uses to read every KotOR texture format:
      • TPC binary (DXT1 / DXT3 / DXT5 / RGB / RGBA / Grey / BGRA)
      • TGA (uncompressed and RLE, colour-mapped and true-colour)
      • DDS (with DXT compression)

    PyKotor's ``read_tpc()`` auto-detects the format, ``TPC.convert()``
    decompresses DXT using pykotor's pure-Python DXT1/3/5 decoders, and
    ``TPCMipmap.to_pil_image()`` wraps the result in a PIL Image.

    The returned image is **always bottom-up** (OpenGL convention: row 0 =
    bottom of the texture).  DXT-compressed textures are flipped from
    PyKotor's top-down output to bottom-up.  This matches the contract of
    viewport.py's _load_tpc_bytes and resource_manager._decode_texture.
    The GPU renderer's ``_upload()`` does NO flip on upload, and the
    vertex shader's ``1.0 - in_uv.y`` converts KotOR's D3D UV convention
    (V=0 at top) to GL convention (V=0 at bottom).
    Phase D11 fix: previously returned top-down for all formats, causing
    upside-down textures when used by paths that skip viewport.py's
    TextureCache (e.g. ResourceManager → gpu_renderer).

    Cross-references
    ----------------
    • pykotor.resource.formats.tpc.tpc_auto.read_tpc   — format detection
    • pykotor.resource.formats.tpc.io_tpc.TPCBinaryReader.load — TPC parse
    • pykotor.resource.formats.tpc.io_tga.TPCTGAReader.load   — TGA parse
    • pykotor.resource.formats.tpc.io_dds.TPCDDSReader.load   — DDS parse
    • pykotor.resource.formats.tpc.tpc_data.TPC.convert       — DXT decode
    • pykotor.resource.formats.tpc.tpc_data.TPCMipmap.to_pil_image
    • pykotor.gl.shader.texture.Texture.from_tpc              — GL upload

    Returns None if PIL is unavailable or data is invalid/corrupt.
    Attaches ._txi_str, ._tpc_raw, ._txi_alpha_test to the returned image.
    """
    if not _HAS_PIL or not data or len(data) < 128:
        return None

    data = patch_tpc_header(data)
    try:
        # ── PyKotor source: read_tpc auto-detects TPC / TGA / DDS ────────
        tpc = pk_read_tpc(data)

        # ── FIX-VFLIP-D11: Detect DXT before conversion ─────────────────
        # PyKotor's to_pil_image returns DXT as top-down, uncompressed as
        # bottom-up.  We must flip DXT to bottom-up for consistency with
        # viewport.py's _load_tpc_bytes contract.
        _orig_fmt = tpc.format()
        _is_dxt = _orig_fmt in (
            TPCTextureFormat.DXT1, TPCTextureFormat.DXT3, TPCTextureFormat.DXT5
        ) if all(hasattr(TPCTextureFormat, x) for x in ('DXT1', 'DXT3', 'DXT5')) else False
        if not _is_dxt and len(data) > 12:
            _enc_byte = data[12]
            _dsz = struct.unpack_from('<I', data, 0)[0] if len(data) >= 4 else 0
            if _enc_byte in (2, 4, 10, 12, 13, 14):
                _w = struct.unpack_from('<H', data, 8)[0] if len(data) >= 10 else 0
                _h = struct.unpack_from('<H', data, 10)[0] if len(data) >= 12 else 0
                _pdlen = len(data) - 128
                _ucmin = {1: _w*_h, 2: _w*_h*3, 4: _w*_h*4}.get(_enc_byte, _w*_h*4)
                if _dsz != 0 or (_w > 0 and _h > 0 and _pdlen < _ucmin):
                    _is_dxt = True

        # ── PyKotor source: TPC.convert decompresses DXT to RGBA ─────────
        tpc.convert(TPCTextureFormat.RGBA)

        # ── PyKotor source: TPCMipmap.to_pil_image ──────────────────────
        img = tpc.get(0, 0).to_pil_image()
        if img is None:
            return None
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        # FIX-VFLIP-D11: Flip DXT textures from top-down to bottom-up.
        # Uncompressed textures are already bottom-up from PyKotor.
        if _is_dxt:
            img = img.transpose(_PILImage.FLIP_TOP_BOTTOM)

        # ── Attach TXI metadata ──────────────────────────────────────────
        txi = ''
        try:
            txi = (tpc.txi or '').strip() if isinstance(getattr(tpc, 'txi', None), str) else ''
        except Exception:
            pass
        img._txi_str  = txi          # type: ignore[attr-defined]
        img._tpc_raw  = data         # type: ignore[attr-defined]
        try:
            at = struct.unpack_from('<f', data, 4)[0]
            img._txi_alpha_test = at if 0.0 < at <= 1.0 else None  # type: ignore[attr-defined]
        except Exception:
            img._txi_alpha_test = None  # type: ignore[attr-defined]

        return img
    except Exception as exc:
        log.debug("load_tpc_as_pil: %s", exc)
        return None


# =============================================================================
#  PyKotor MDL → KotorModel conversion
# =============================================================================

def _mdl_to_kotormodel(pk_mdl, game_version: Optional[GameVersion]) -> KotorModel:
    model = KotorModel()
    model.name        = pk_mdl.name or 'unnamed'
    model.supermodel  = str(getattr(pk_mdl, 'supermodel', 'NULL') or 'NULL')
    model.anim_scale  = float(getattr(pk_mdl, 'animation_scale', 1.0) or 1.0)
    model.game_version = game_version or _detect_version(pk_mdl)
    model.disable_fog = bool(getattr(pk_mdl, 'fog', False))

    # Classification: read raw byte from binary header (bytes BASE+80 = model_type)
    # and map using GhostRigger's ModelClassification table (differs from PyKotor's enum).
    # PyKotor classification enum values differ: e.g. PyKotor PLACEABLE=16 vs ours LIGHTSABER=16.
    _CLS_MAP: Dict[int, str] = {
        0: 'effect', 1: 'effects', 2: 'tile', 4: 'character',
        8: 'door', 16: 'lightsaber', 32: 'placeable', 64: 'flyer',
    }
    try:
        # mdl_bytes may or may not be available — read from pk_mdl.classification enum
        # The enum .value gives the raw integer from the binary.
        raw_cls = int(getattr(pk_mdl, 'classification', None) or 0)
        model.classification = _CLS_MAP.get(raw_cls, str(raw_cls))
        model.model_type = raw_cls
    except Exception:
        model.classification = 'character'
        model.model_type = 4

    declared_bb_min = model.bb_min
    declared_bb_max = model.bb_max
    declared_radius = model.radius
    try:
        model.bb_min = (float(pk_mdl.bmin.x), float(pk_mdl.bmin.y), float(pk_mdl.bmin.z))
        model.bb_max = (float(pk_mdl.bmax.x), float(pk_mdl.bmax.y), float(pk_mdl.bmax.z))
        model.radius = float(getattr(pk_mdl, 'radius', 0.0) or 0.0)
        declared_bb_min = model.bb_min
        declared_bb_max = model.bb_max
        declared_radius = model.radius
    except Exception:
        pass

    # node_id → pk_node  (needed for skin bone-map resolution)
    id_to_pknode: Dict[int, object] = {}
    try:
        for n in pk_mdl.all_nodes():
            id_to_pknode[int(n.node_id)] = n
    except Exception:
        pass

    # Convert geometry hierarchy
    if pk_mdl.root is not None:
        model.root_node = _convert_node(pk_mdl.root, None, id_to_pknode)

    # Convert animations
    for pk_anim in (pk_mdl.anims or []):
        anim = _convert_anim(pk_anim)
        if anim is not None:
            model.animations.append(anim)

    # D20-M: Assign per-node vertex_space at load time.
    # This is the SINGLE source of truth for "does this node need
    # world_transform applied?" — no centroid heuristics allowed.
    try:
        from src.core.geometry.vertex_space import compute_vertex_space
        for nd in model.all_nodes():
            nd.vertex_space = int(compute_vertex_space(nd, model))
    except Exception as _vs_exc:
        log.debug("vertex_space assignment failed: %s", _vs_exc)

    # Keep the binary model-header envelope source-preserving.  Retail
    # character heads commonly declare deliberately broad culling bounds that
    # differ from their tight rendered-geometry bounds.  The viewport still
    # needs the latter, so compute and expose it through the established
    # ``_gr_render_bounds`` side channel, then restore the serialized values
    # for round-trip writing.
    model.compute_bounds()
    setattr(model, "_gr_render_bounds", (model.bb_min, model.bb_max))
    setattr(model, "_gr_render_radius", model.radius)
    setattr(model, "_gr_bounds_prepared", True)
    model.bb_min = declared_bb_min
    model.bb_max = declared_bb_max
    model.radius = declared_radius
    _fill_missing_normals(model)
    _apply_bind_pose(model)
    return model


def _detect_version_from_bytes(mdl_bytes: bytes) -> GameVersion:
    """Detect K1 vs K2 from the geometry function pointer in the MDL header.

    The MDL binary starts with 12 bytes of file header, then the geometry
    header.  Bytes 12-15 (BASE+0) contain fp1 — a function pointer that
    differs by game and platform, and is the most reliable K1/K2 discriminator.

    K2 PC fp1 values: 4285200 (0x414250), 4284816 (0x414110)
    K2 Xbox fp1:      4285872
    K1 PC fp1 values: 4273776, 4273392
    K1 Xbox fp1:      4254992
    """
    _K2_FP1 = {4285200, 4284816, 4285872}
    if len(mdl_bytes) >= 16:
        try:
            fp1 = struct.unpack_from('<I', mdl_bytes, 12)[0]
            if fp1 in _K2_FP1:
                return GameVersion.K2
        except Exception:
            pass
    return GameVersion.K1


# Xbox function pointer values (K1 and K2)
_XBOX_FP1 = {4254992, 4285872}

def _is_xbox_from_bytes(mdl_bytes: bytes) -> bool:
    """Return True if the MDL was built for Xbox (different function pointers)."""
    if len(mdl_bytes) >= 16:
        try:
            fp1 = struct.unpack_from('<I', mdl_bytes, 12)[0]
            return fp1 in _XBOX_FP1
        except Exception:
            pass
    return False


def _detect_version(pk_mdl) -> GameVersion:
    """Detect game version from PyKotor MDL object (fallback for file-less callers)."""
    try:
        geo = getattr(pk_mdl, 'geometry_type', None)
        if geo is not None and int(geo) in (4, 5):
            return GameVersion.K2
    except Exception:
        pass
    return GameVersion.K1


# ── Node conversion ───────────────────────────────────────────────────────────

def _convert_node(pk_node, parent: Optional[ModelNode],
                  id_to_pknode: Dict) -> ModelNode:
    gr = ModelNode()
    gr.name   = pk_node.name or 'node'
    gr.index  = int(pk_node.node_id)
    gr.parent = parent

    # Position + Rotation straight from PyKotor
    try:
        p = pk_node.position
        gr.position = (float(p.x), float(p.y), float(p.z))
    except Exception:
        gr.position = (0.0, 0.0, 0.0)

    try:
        # Quaternion convention (locked in for the entire pipeline)
        # -----------------------------------------------------------
        # Binary MDL on disk stores node orientation as W,X,Y,Z (W first) —
        # see PyKotor ``_NodeHeader.read`` and xoreos
        # ``ModelNode_KotOR::load`` which reads ``rad2deg(acos(W) * 2)``
        # followed by X, Y, Z as an axis/angle pair.  PyKotor already
        # deserialises that into its ``Vector4(x,y,z,w)`` accessor, so by
        # the time we touch ``pk_node.orientation`` it is internally
        # XYZW-ordered.  We keep it in XYZW form throughout the GhostRigger
        # pipeline: ``_quat_mul``, ``_quat_rotate``, ``_quat_normalize_bind``
        # in model_data.py, and every callsite in viewport.py / writers.
        # Tests: ``test_quaternion_convention_consistency`` in
        # ``test_geometry_phase_g2.py``.
        o = pk_node.orientation
        gr.rotation = (float(o.x), float(o.y), float(o.z), float(o.w))
    except Exception:
        gr.rotation = (0.0, 0.0, 0.0, 1.0)

    # Flags: use _TYPE_FLAGS for ALL node types including DUMMY.
    # _TYPE_FLAGS maps DUMMY → NodeFlags.HEADER (0x01) for both root AND child
    # dummy nodes.  The old code gave child dummies flags=0, breaking is_dummy
    # checks throughout the pipeline (auto_rigger, retarget_engine, main_window,
    # viewport skeleton rendering).  The old pykotor_bridge._NODETYPE_TO_FLAGS
    # always assigned NodeFlags.HEADER to ALL dummy nodes — this matches that.
    ntype = int(getattr(pk_node, 'node_type', int(MDLNodeType.DUMMY)))
    gr.flags = _TYPE_FLAGS.get(ntype, 0)

    # Controllers (bind-pose keyframes)
    _read_controllers(pk_node, gr)

    # Geometry — MDLMesh for trimesh/skin; MDLSkin for bone/weight data
    mesh_obj = getattr(pk_node, 'mesh', None)
    if mesh_obj is not None:
        _read_mesh(mesh_obj, gr)

    if ntype == int(MDLNodeType.DANGLYMESH):
        # Dangly parameters live on mesh_obj.dangly (MDLDangly sub-object)
        if mesh_obj is not None:
            _read_dangly(mesh_obj, gr)

    if ntype == int(MDLNodeType.SKIN):
        skin_obj = getattr(pk_node, 'skin', None)
        if skin_obj is not None:
            _read_skin_textures(skin_obj, gr)       # texture overrides from skin
            _read_skin_weights(skin_obj, gr, id_to_pknode)

    if ntype == int(MDLNodeType.LIGHT):
        _read_light(pk_node, gr)

    if ntype == int(MDLNodeType.EMITTER):
        _read_emitter(pk_node, gr)

    if ntype == int(MDLNodeType.REFERENCE):
        reference = getattr(pk_node, 'reference', None)
        if reference is not None:
            gr.reference_model = str(getattr(reference, 'model', '') or '')
            gr.reference_reattachable = bool(getattr(reference, 'reattachable', False))

    # Recurse children
    for child_pk in (pk_node.children or []):
        child_gr = _convert_node(child_pk, gr, id_to_pknode)
        gr.children.append(child_gr)

    return gr


def _read_mesh(mesh, gr: ModelNode) -> None:
    """Read all geometry data directly from a PyKotor MDLMesh object."""

    # ── Textures (lowercase for consistent lookup) ──────────────────────────────
    tex1 = str(getattr(mesh, 'texture_1', '') or '').strip().lower()
    tex2 = str(getattr(mesh, 'texture_2', '') or '').strip().lower()
    if tex1:
        gr.texture       = tex1
        gr.texture_names = [tex1]
        gr.tex_count     = 1
    if tex2:
        gr.lightmap = tex2
        if tex2 not in gr.texture_names:
            gr.texture_names.append(tex2)
        gr.tex_count = len(gr.texture_names)

    # ── Material colours ──────────────────────────────────────────────────────
    diff = getattr(mesh, 'diffuse', None)
    if diff is not None:
        try:
            gr.diffuse = (float(diff.r), float(diff.g), float(diff.b))
        except Exception:
            pass
    amb = getattr(mesh, 'ambient', None)
    if amb is not None:
        try:
            gr.ambient = (float(amb.r), float(amb.g), float(amb.b))
        except Exception:
            pass

    # ── Render flags ──────────────────────────────────────────────────────────
    gr.render              = bool(getattr(mesh, 'render',             True))
    gr.has_shadow          = bool(getattr(mesh, 'shadow',             False))
    gr.beaming             = bool(getattr(mesh, 'beaming',            False))
    gr.background_geometry = bool(getattr(mesh, 'background_geometry',False))
    gr.transparency_hint   = int( getattr(mesh, 'transparency_hint',  0))
    gr.rotate_texture      = bool(getattr(mesh, 'rotate_texture',     False))
    gr.has_lightmap        = bool(getattr(mesh, 'has_lightmap',       False))

    # ── K2-specific mesh-header fields (Phase G2) ─────────────────────────────
    # These extra bytes exist only in KotOR 2's trimesh sub-header.  PyKotor
    # exposes them on the mesh object regardless of game, defaulting to the
    # "no-op" values on K1 models, so reading them unconditionally is safe.
    #
    #   dirt_enabled / dirt_texture / dirt_coordinate_space:
    #       K2 dirt-overlay decal support (used on armour/creature weathering).
    #   hologram_donotdraw / hide_in_hologram:
    #       K2 hologram render-pass flag.  PyKotor exposes both the modern
    #       name (``hologram_donotdraw``) and the legacy alias
    #       (``hide_in_hologram``); we OR them so either source wins.
    #
    # We preserve these verbatim so round-trip writing through mdl_writer can
    # emit a byte-identical K2 mesh header.  viewport.py will grow an opt-in
    # respecter for ``hide_in_holograms`` in a follow-up; for now the field is
    # populated purely for fidelity.
    gr.dirt_enabled      = bool(getattr(mesh, 'dirt_enabled', False))
    gr.dirt_texture      = int( getattr(mesh, 'dirt_texture', 0) or 0)
    gr.dirt_coord_space  = int( getattr(mesh, 'dirt_coordinate_space', 0) or 0)
    gr.hide_in_holograms = bool(
        getattr(mesh, 'hologram_donotdraw', False)
        or getattr(mesh, 'hide_in_hologram', False)
    )

    # ── UV animation ─────────────────────────────────────────────────────────
    gr.animate_uv      = bool( getattr(mesh, 'animate_uv',       False))
    gr.uv_dir_x        = float(getattr(mesh, 'uv_direction_x',   0.0) or 0.0)
    gr.uv_dir_y        = float(getattr(mesh, 'uv_direction_y',   0.0) or 0.0)
    gr.uv_jitter       = float(getattr(mesh, 'uv_jitter',        0.0) or 0.0)
    gr.uv_jitter_speed = float(getattr(mesh, 'uv_jitter_speed',  0.0) or 0.0)

    # ── Safe attribute helper for mesh data lists ────────────────────────────
    # PyKotor returns:
    #   • list[VectorN]  — good, use directly
    #   • bool (True/False) — tangent_space is a FLAG, not a data list
    #   • None            — attribute absent
    #   • str             — default-arg sentinel ('NO_ATTR' etc.)
    #   • method/callable — some attrs are methods; skip those too
    # We accept ONLY plain lists (or list-like containers that are NOT bool/str).
    def _safe_vec3_list(attr_val):
        """Return attr_val as a list of 3-vectors, or [] if not a valid list."""
        if attr_val is None or isinstance(attr_val, (bool, str, bytes)):
            return []
        if callable(attr_val):
            return []
        return list(attr_val) if hasattr(attr_val, '__iter__') else []

    def _safe_vec2_list(attr_val):
        """Return attr_val as a list of 2-vectors, or [] if not a valid list."""
        if attr_val is None or isinstance(attr_val, (bool, str, bytes)):
            return []
        if callable(attr_val):
            return []
        return list(attr_val) if hasattr(attr_val, '__iter__') else []

    # ── UV/Vertex sanity threshold ─────────────────────────────────────────────
    # KotOR module geometry uses legitimately large tiled UVs (e.g. N_sithpraet
    # pelvis U=[-13,+13]).  However, PyKotor can return astronomically large or
    # non-finite values when the MDX binary contains a corrupt or misaligned
    # data_offset (observed in generated test files: U≈1e28, V≈1e-38, etc.).
    # These corrupt values come from reading the wrong byte region of the MDX file.
    #
    # Strategy:
    #   • Position values that are NaN, ±Inf, or |x| > _GEOM_MAX are corrupt.
    #   • UV magnitudes are not classified here; tiled UVs pass through unchanged.
    #   • Vertex positions with |x|>10000 are also corrupt (KotOR worlds fit in ~500 units).
    #
    # If MORE than 50% of vertices are corrupt, the whole mesh has a bad MDX offset;
    # we clear all UVs so the renderer shows the mesh as flat-shaded rather than
    # invisibly broken.  This is preferable to the user seeing garbage geometry.
    _GEOM_MAX = 1e6   # anything beyond ±1,000,000 is definitely corrupt

    def _safe_float(x: float, fallback: float = 0.0) -> float:
        """Return x if finite and within range, else fallback."""
        if not math.isfinite(x) or abs(x) > _GEOM_MAX:
            return fallback
        return x

    def _safe_uv(x: float, y: float) -> tuple:
        """Return UV pair; replace only non-finite components with 0.5."""
        xu = x if math.isfinite(x) else 0.5
        yu = y if math.isfinite(y) else 0.5
        return (xu, yu)

    # ── Vertex positions ─────────────────────────────────────────────────────
    # PyKotor: MDLMesh.vertex_positions → list[Vector3]
    vp = _safe_vec3_list(mesh.vertex_positions)
    raw_verts = []
    corrupt_vert_count = 0
    for v in vp:
        vx, vy, vz = float(v.x), float(v.y), float(v.z)
        if not (math.isfinite(vx) and math.isfinite(vy) and math.isfinite(vz)) or \
                max(abs(vx), abs(vy), abs(vz)) > _GEOM_MAX:
            corrupt_vert_count += 1
            raw_verts.append((_safe_float(vx), _safe_float(vy), _safe_float(vz)))
        else:
            raw_verts.append((vx, vy, vz))
    gr.vertices = raw_verts

    # Detect mesh with corrupt MDX data: extreme or non-finite values (misaligned MDX).
    # We set _mesh_is_mdx_corrupt to suppress UV loading for very-corrupt meshes
    # (>50% corrupt) where UVs are also garbage.  Partially corrupt meshes still load UVs.
    _has_any_corrupt = corrupt_vert_count > 0
    _mesh_is_mdx_corrupt = (len(raw_verts) > 0 and
                            corrupt_vert_count > len(raw_verts) // 2)
    if _has_any_corrupt:
        log.debug("_read_mesh '%s': %d/%d verts have suspect MDX positions (clamped)",
                  gr.name, corrupt_vert_count, len(raw_verts))
    if _mesh_is_mdx_corrupt:
        # For severely-corrupt meshes (>50% bad), normals are also garbage — clear them.
        # Faces are loaded below (they reference MDL vertex indices, not MDX).
        # UVs are skipped for >50% corrupt meshes (they're from MDX and also garbage).
        gr.normals = []

    # ── Vertex normals ────────────────────────────────────────────────────────
    # PyKotor: MDLMesh.vertex_normals → list[Vector3]
    # Skip for severely-corrupt meshes (>50% bad verts — normals are also garbage).
    # For partially-corrupt meshes (<50% bad), normals are loaded and used.
    if not _mesh_is_mdx_corrupt:
        vn = _safe_vec3_list(mesh.vertex_normals)
        gr.normals = [(_safe_float(float(n.x)), _safe_float(float(n.y)), _safe_float(float(n.z), 1.0))
                      for n in vn]

    # ── Tangents (MDX bump-map stream) ───────────────────────────────────────
    # NOTE: MDLMesh.tangent_space is a BOOLEAN FLAG in PyKotor (True = has
    # tangent data), NOT the tangent vector list.  Do NOT iterate it.
    # PyKotor does not expose pre-computed tangent vectors in this version;
    # tangents are computed on-demand in the exporter from normals+UVs.
    # Preserve the source MDX channel contract separately so the binary writer
    # can keep known no-tangent models (for example c_drexlf) compact while
    # rebuilding the full B/T/N rows used by bump-mapped models such as
    # c_rancor.
    gr.mdx_tangent_space = bool(getattr(mesh, 'tangent_space', False))
    gr.tangents = []  # filled by exporter when needed

    # ── Primary UVs ──────────────────────────────────────────────────────────
    # PyKotor: MDLMesh.vertex_uv1  (= .vertex_uv property alias)
    # Use explicit None checks so that an empty-list attribute ([]) doesn't
    # fall through to a method-object lookup (which is not iterable).
    # If >50% of verts are corrupt, skip UV loading — UVs are likely garbage too.
    if not _mesh_is_mdx_corrupt:
        _uv1_attr = getattr(mesh, 'vertex_uv1', None)
        if _uv1_attr is None or isinstance(_uv1_attr, bool):
            _uv1_raw = getattr(mesh, 'vertex_uv', None)
            # If the fallback is callable (a method), call it; otherwise use as-is
            if callable(_uv1_raw):
                uv1 = _uv1_raw()
            else:
                uv1 = _safe_vec2_list(_uv1_raw)
        else:
            uv1 = _uv1_attr
        # Keep finite UVs exactly as authored; large values are valid tiling data.
        gr.uvs = [_safe_uv(float(u.x), float(u.y)) for u in (uv1 or [])]

        # ── Secondary / lightmap UVs ─────────────────────────────────────────────
        # PyKotor: MDLMesh.vertex_uv2
        _uv2_attr = getattr(mesh, 'vertex_uv2', None)
        uv2 = _safe_vec2_list(_uv2_attr)
        gr.uvs_2  = [_safe_uv(float(u.x), float(u.y)) for u in uv2]
        gr.uvs_lm = list(gr.uvs_2)   # same data — lightmap consumer alias

    # ── Faces + per-face UV indices + face materials ──────────────────────────
    # PyKotor: MDLFace fields — v1/v2/v3 (vertex idx), t1/t2/t3 (UV idx, -1=same as vertex)
    # MDLFace.material: lower 5 bits = material slot (& 0x1F strips upper garbage bits)
    # Always load faces — they reference MDL vertex indices which are correct regardless
    # of MDX corruption.
    gr.faces     = []
    gr.face_uvs  = []
    gr.face_mats = []
    _face_iter = mesh.faces or []
    for f in _face_iter:
        v1, v2, v3 = int(f.v1), int(f.v2), int(f.v3)
        gr.faces.append((v1, v2, v3))
        t1 = int(f.t1) if getattr(f, 't1', None) is not None and f.t1 >= 0 else v1
        t2 = int(f.t2) if getattr(f, 't2', None) is not None and f.t2 >= 0 else v2
        t3 = int(f.t3) if getattr(f, 't3', None) is not None and f.t3 >= 0 else v3
        gr.face_uvs.append((t1, t2, t3))
        # material: strip high bits — only lower 5 bits are valid (& 0x1F).
        # The KotOR binary face-material field stores surface/smoothing-group
        # zone indices.  These can exceed the number of actual texture slots
        # (tex_count).  We clamp to [0, tex_count-1] so that face_mats are
        # always valid indices into texture_names, preserving the invariant
        # face_mats[i] in [0, tex_count-1] (and len(texture_names)==tex_count).
        mat_raw = int(getattr(f, 'material', 0) or 0) & 0x1F
        mat = min(mat_raw, max(0, gr.tex_count - 1))
        gr.face_mats.append(mat)

    # ── Bounding box ──────────────────────────────────────────────────────────
    try:
        gr.bb_min = (float(mesh.bb_min.x), float(mesh.bb_min.y), float(mesh.bb_min.z))
        gr.bb_max = (float(mesh.bb_max.x), float(mesh.bb_max.y), float(mesh.bb_max.z))
        # PyKotor exposes the complete trimesh culling envelope, but the
        # historical conversion retained only min/max. Modular-head export
        # freezes the donor's per-node radius too, so preserve the raw mesh
        # values explicitly for the writer and structural reload verifier.
        gr.mesh_bb_min = gr.bb_min
        gr.mesh_bb_max = gr.bb_max
        gr.radius = float(getattr(mesh, "radius", 0.0) or 0.0)
        gr.mesh_radius = gr.radius
        average = getattr(mesh, "average", None)
        if average is not None:
            gr.mesh_average_point = (
                float(average.x),
                float(average.y),
                float(average.z),
            )
    except Exception:
        pass

    # ── FIX-LMROLE-V2 (Phase D10): Infer lightmap role from texture_2 + UV2 ──
    # KotOR module/area binary MDL meshes sometimes have has_lightmap=False even
    # when texture_2 IS a genuine lightmap and vertex_uv2 contains valid lightmap
    # UVs.  This causes the renderer to miss lightmap compositing.
    #
    # The correct KotOR texture model (per xoreos, KotOR.js, KotorBlender):
    #   - texture_1 = diffuse (UV0)
    #   - texture_2 = lightmap (UV1) when present
    #   - face_mats is a WALK-MESH SURFACE INDICATOR, NOT a texture selector
    #
    # Heuristic: if ALL of these hold, promote has_lightmap to True:
    #   1. has_lightmap is currently False
    #   2. tex_count == 2 (exactly one diffuse + one secondary)
    #   3. uvs_lm has real data (same count as uvs, i.e. came from vertex_uv2)
    #
    # NOTE: face_mats is NOT checked — it is a walk-mesh surface indicator and
    # must never be used for texture-routing decisions (Phase D10 fix).
    #
    # Cross-ref: KotOR.js checks texture_2 + UV2 presence; Kotor.NET treats
    # module tex2 as lightmap; xoreos uses has_lightmap flag only.
    if (not gr.has_lightmap
            and gr.tex_count == 2
            and len(gr.uvs_lm) > 0
            and len(gr.uvs_lm) == len(gr.uvs)):
        gr.has_lightmap = True
        log.debug("FIX-LMROLE-V2: inferred has_lightmap=True for node '%s' "
                   "(tex2='%s', %d LM UVs)",
                   gr.name, gr.lightmap, len(gr.uvs_lm))


def _read_skin_textures(skin, gr: ModelNode) -> None:
    """Override texture/material from MDLSkin (for SKIN nodes)."""
    tex1 = str(getattr(skin, 'texture_1', '') or '').strip()
    tex2 = str(getattr(skin, 'texture_2', '') or '').strip()
    if tex1:
        gr.texture       = tex1
        gr.texture_names = [tex1]
        gr.tex_count     = 1
    if tex2:
        gr.lightmap = tex2
        if tex2 not in gr.texture_names:
            gr.texture_names.append(tex2)
        gr.tex_count = len(gr.texture_names)
    diff = getattr(skin, 'diffuse', None)
    if diff is not None:
        try:
            gr.diffuse = (float(diff.r), float(diff.g), float(diff.b))
        except Exception:
            pass
    amb = getattr(skin, 'ambient', None)
    if amb is not None:
        try:
            gr.ambient = (float(amb.r), float(amb.g), float(amb.b))
        except Exception:
            pass
    gr.render    = bool(getattr(skin, 'render',  True))
    gr.has_shadow = bool(getattr(skin, 'shadow', True))


def _read_skin_weights(skin, gr: ModelNode, id_to_pknode: Dict) -> None:
    """Read bone-map and per-vertex weights directly from PyKotor MDLSkin.

    PyKotor MDLSkin exposes two related arrays:
      skin.bone_indices  — compact uint16[16] header array of active node_ids.
                           Vertex MDX data stores float32 indices into THIS array
                           (offset_to_mdx_bones in the skin header).
      skin.bonemap       — separate float32 array at offset_to_bonemap, length =
                           bonemap_count (up to total node count, e.g. 46).
                           This is a DIFFERENT structure from bone_indices.

    MDLBoneVertex.vertex_indices[j] indexes into bone_indices (compact uint16[16]),
    NOT into bonemap.  This is confirmed by PyKotor io_mdl.py: bone_indices is read
    as uint16[16] from the skin header (self.bones), while bonemap is read separately
    as float32 values from offset_to_bonemap.  They are independent structures.

    Fallback strategy:
      1. Use bone_indices (uint16[16]) — correct for real MDL files.
      2. If bone_indices produces an all-empty map (all entries -1/0xFFFF/invalid),
         fall back to bonemap — handles synthetic/mock skins and edge cases.
    """
    # Step 1: bone_map — compact slot → bone node name
    # Use skin.bone_indices (compact list of active node_ids) so that
    # vertex_indices[j] maps correctly to the right bone name.
    gr.bone_map = []
    raw_bone_indices = list(getattr(skin, 'bone_indices', None) or [])
    for nid_raw in raw_bone_indices:
        try:
            nid = int(nid_raw)
        except (TypeError, ValueError):
            gr.bone_map.append('')
            continue
        if nid < 0 or nid == 0xFFFF:
            gr.bone_map.append('')
            continue
        pk_n = id_to_pknode.get(nid)
        gr.bone_map.append(pk_n.name if pk_n else '')

    # Cross-validate the palette against the node-indexed ``bonemap`` array.
    # ``bone_indices`` is a FIXED uint16[16] header block: unused tail slots
    # carry padding that is sometimes 0 (which aliases the model root node)
    # and sometimes uninitialised garbage.  Vanilla files (e.g. K2 c_drexlf
    # tailGeo: bones = [9,8,7,6,5,4,11,12,13,10, 0,0,0,0,0,0]) would otherwise
    # gain phantom palette entries pointing at the root.  A slot ``s`` is a
    # real palette entry iff bonemap[node_id] == s — that is exactly how the
    # engine associates nodes with palette slots.  Only applies when a
    # bonemap is present; mid-array blanks are preserved (indices matter),
    # trailing blanks are trimmed.
    _raw_bonemap_check = list(getattr(skin, 'bonemap', None) or [])
    if _raw_bonemap_check and any(name for name in gr.bone_map):
        _validated = []
        for _slot, nid_raw in enumerate(raw_bone_indices[:len(gr.bone_map)]):
            try:
                _nid = int(nid_raw)
            except (TypeError, ValueError):
                _validated.append('')
                continue
            try:
                _confirmed = (
                    0 <= _nid < len(_raw_bonemap_check)
                    and int(_raw_bonemap_check[_nid]) == _slot
                )
            except (TypeError, ValueError, OverflowError):
                _confirmed = False
            _validated.append(gr.bone_map[_slot] if _confirmed else '')
        if any(name for name in _validated):
            while _validated and not _validated[-1]:
                _validated.pop()
            gr.bone_map = _validated

    # Fallback: if bone_indices produced no valid entries (all were
    # -1/0xFFFF/invalid or the array was empty), derive the palette from the
    # NODE-indexed bonemap by inversion (bonemap[node_id] = palette_slot).
    # This handles:
    #   - Synthetic mock skins (unit tests with MockMDLSkin(bone_indices=all_invalid))
    #   - Models where bone_indices is genuinely absent
    # Note: We check whether ANY non-empty name was produced, not just array length.
    # Legacy tolerance: some synthetic/mock skins store palette-length arrays
    # of node ids (bonemap[slot] = node_id).  When inversion yields nothing
    # (no entry maps back to a plausible slot), fall back to that legacy read.
    _has_valid_bi = any(name for name in gr.bone_map)
    if not _has_valid_bi:
        _raw_bm = list(getattr(skin, 'bonemap', None) or [])
        slot_to_node: Dict[int, int] = {}
        for _node_id, _slot_raw in enumerate(_raw_bm):
            try:
                _slot = int(_slot_raw)
            except (TypeError, ValueError):
                continue
            if 0 <= _slot < len(_raw_bm) and _slot not in slot_to_node:
                slot_to_node[_slot] = _node_id
        gr.bone_map = []
        if slot_to_node and max(slot_to_node) < max(len(_raw_bm), 1):
            for _slot in range(max(slot_to_node) + 1):
                _nid = slot_to_node.get(_slot, -1)
                pk_n = id_to_pknode.get(_nid) if _nid >= 0 else None
                gr.bone_map.append(pk_n.name if pk_n else '')
        if not any(name for name in gr.bone_map):
            gr.bone_map = []
            for raw in _raw_bm:
                try:
                    nid = int(raw)
                except (TypeError, ValueError):
                    gr.bone_map.append('')
                    continue
                if nid < 0 or nid == 0xFFFF:
                    gr.bone_map.append('')
                    continue
                pk_n = id_to_pknode.get(nid)
                gr.bone_map.append(pk_n.name if pk_n else '')

    apply_known_skin_bone_map_normalisations(gr, skin, id_to_pknode)
    n_bones = len(gr.bone_map)

    # Some shipped creature skins use vertex palette indices beyond the fixed
    # 16-entry ``bone_indices`` header array while the node-indexed ``bonemap``
    # table still assigns those overflow palette slots to valid nodes.
    # Example: c_brith/Brith_mesh has vertices weighted to local index 16;
    # ``bone_indices`` has slots 0..15, and some node's bonemap entry == 16.
    # Dropping those influences leaves zero-weight vertices and frozen
    # triangles during animation.  Recover the overflow slot by scanning the
    # NODE-indexed bonemap for the node whose entry equals the slot number
    # (bonemap[node_id] = palette_slot — verified against vanilla K2
    # c_drexlf raw bytes, T2526).
    try:
        max_vertex_idx = -1
        for bv in (getattr(skin, 'vertex_bones', None) or []):
            for j in range(4):
                try:
                    raw_idx = float(bv.vertex_indices[j])
                except (TypeError, ValueError, IndexError):
                    continue
                if math.isfinite(raw_idx):
                    max_vertex_idx = max(max_vertex_idx, int(raw_idx))

        raw_bonemap = list(getattr(skin, 'bonemap', None) or [])
        if max_vertex_idx >= n_bones and raw_bonemap:
            slot_to_node = {}
            for node_id, slot_raw in enumerate(raw_bonemap):
                try:
                    slot_val = int(slot_raw)
                except (TypeError, ValueError):
                    continue
                if slot_val >= 0 and slot_val not in slot_to_node:
                    slot_to_node[slot_val] = node_id
            for slot in range(n_bones, max_vertex_idx + 1):
                name = ''
                nid = slot_to_node.get(slot, -1)
                if nid >= 0:
                    pk_n = id_to_pknode.get(nid)
                    name = pk_n.name if pk_n else ''
                gr.bone_map.append(name)
            n_bones = len(gr.bone_map)
            log.debug(
                "_read_skin_weights '%s': extended bone_map to %d slot(s) "
                "using bonemap overflow (max vertex index=%d)",
                gr.name, n_bones, max_vertex_idx,
            )
    except Exception as _e:
        log.debug("_read_skin_weights '%s': bonemap overflow extension skipped: %s", gr.name, _e)

    # Preserve the compact palette's exact DFS node ids alongside its display
    # names. Odyssey permits duplicate node names (K2 PFBCM repeats part of
    # the left-hand chain); names alone cannot round-trip those palettes or
    # select the matching qBone/tBone row. The fixed bones[16] block and the
    # node-indexed bonemap together provide the unambiguous slot -> node map.
    _slot_to_node_id: Dict[int, int] = {}
    for _node_id, _slot_raw in enumerate(_raw_bonemap_check):
        try:
            _slot = int(_slot_raw)
        except (TypeError, ValueError):
            continue
        if _slot >= 0 and _slot not in _slot_to_node_id:
            _slot_to_node_id[_slot] = _node_id
    gr.bone_node_indices = []
    for _slot in range(len(gr.bone_map)):
        _node_id = -1
        if _slot < len(raw_bone_indices):
            try:
                _candidate_id = int(raw_bone_indices[_slot])
            except (TypeError, ValueError):
                _candidate_id = -1
            try:
                _candidate_confirmed = (
                    0 <= _candidate_id < len(_raw_bonemap_check)
                    and int(_raw_bonemap_check[_candidate_id]) == _slot
                )
            except (TypeError, ValueError, OverflowError):
                _candidate_confirmed = False
            if _candidate_confirmed:
                _node_id = _candidate_id
        if _node_id < 0:
            _node_id = int(_slot_to_node_id.get(_slot, -1))
        gr.bone_node_indices.append(_node_id)

    # v7.1 FIX-QBONETBONE (Finding 2.5 — reone mdlmdxreader.cpp cross-ref):
    # Read qBone (quaternion) and tBone (translation) arrays from PyKotor skin object.
    # These provide per-bone bind-pose transforms that serve as fallback matrices
    # when world_transform() fails during FBX export.
    # PyKotor stores: skin.qbones = list[Vector4], skin.tbones = list[Vector3]
    gr.qbone_list = []
    gr.tbone_list = []
    try:
        _qbones_raw = getattr(skin, 'qbones', None) or []
        _tbones_raw = getattr(skin, 'tbones', None) or []
        for _qb in _qbones_raw:
            try:
                _qx = float(getattr(_qb, 'x', _qb[0]) if hasattr(_qb, 'x') else _qb[0])
                _qy = float(getattr(_qb, 'y', _qb[1]) if hasattr(_qb, 'y') else _qb[1])
                _qz = float(getattr(_qb, 'z', _qb[2]) if hasattr(_qb, 'z') else _qb[2])
                _qw = float(getattr(_qb, 'w', _qb[3]) if hasattr(_qb, 'w') else _qb[3])
                gr.qbone_list.append((_qx, _qy, _qz, _qw))
            except (TypeError, IndexError, ValueError):
                gr.qbone_list.append((0.0, 0.0, 0.0, 1.0))
        for _tb in _tbones_raw:
            try:
                _tx = float(getattr(_tb, 'x', _tb[0]) if hasattr(_tb, 'x') else _tb[0])
                _ty = float(getattr(_tb, 'y', _tb[1]) if hasattr(_tb, 'y') else _tb[1])
                _tz = float(getattr(_tb, 'z', _tb[2]) if hasattr(_tb, 'z') else _tb[2])
                gr.tbone_list.append((_tx, _ty, _tz))
            except (TypeError, IndexError, ValueError):
                gr.tbone_list.append((0.0, 0.0, 0.0))
        if gr.qbone_list:
            log.debug("_read_skin_weights '%s': read %d qBone + %d tBone bind-pose entries",
                      gr.name, len(gr.qbone_list), len(gr.tbone_list))
    except Exception as _e:
        log.debug("_read_skin_weights '%s': qBone/tBone read skipped: %s", gr.name, _e)

    # Step 2: per-vertex skin data
    #
    # Phase G2: we track out-of-range indices so a malformed MDL surfaces as
    # a single summary warning rather than being silently swallowed.  Silent
    # skipping is safe (the vertex just loses one influence and its other
    # weights renormalise), but without any log line the user has no clue
    # why a skin mesh looks wrong.  We log at most one WARNING line per
    # skin node, regardless of how many vertices are affected.
    gr.skin_data = []
    _oob_count = 0
    for bv in (getattr(skin, 'vertex_bones', None) or []):
        vsd = VertexSkinData()
        for j in range(4):
            try:
                _raw_idx = float(bv.vertex_indices[j])
                if not math.isfinite(_raw_idx):
                    continue
                local_idx = int(_raw_idx)
            except (TypeError, ValueError, IndexError):
                continue
            w         = float(bv.vertex_weights[j])
            if local_idx < 0 or local_idx >= n_bones:
                # Out-of-range bone index: record and skip this influence.
                # Per-vertex weight renormalisation (below) redistributes the
                # dropped weight across the surviving influences.
                if local_idx >= 0:
                    _oob_count += 1
                continue
            if w <= 1e-6 or not math.isfinite(w):
                continue
            vsd.influences.append(BoneWeight(bone_index=local_idx, weight=w))

        # Normalise
        if vsd.influences:
            total = sum(bw.weight for bw in vsd.influences)
            if total > 1e-5 and abs(total - 1.0) > 1e-4:
                inv = 1.0 / total
                for bw in vsd.influences:
                    bw.weight *= inv

        gr.skin_data.append(vsd)

    if _oob_count:
        log.warning(
            "Skin node '%s': %d vertex bone-index(es) exceeded bone_map "
            "size %d — influences dropped and weights renormalised",
            gr.name, _oob_count, n_bones,
        )

    log.debug("_read_skin_weights '%s': %d bones, %d verts",
              gr.name, n_bones, len(gr.skin_data))


def _read_dangly(mesh, gr: ModelNode) -> None:
    """Read dangly mesh parameters from a PyKotor MDLDangly object.

    In PyKotor, MDLDangly extends MDLMesh — so the dangly-specific
    attributes (displacement, tightness, period, constraints) live
    DIRECTLY on the mesh object passed in (no sub-attribute needed).

    Constraint weights:
      PyKotor stores the float weight (0-255 game range) in
      MDLConstraint.type as: int(weight_0_to_255 * 1_000_000).
      We decode that back to a normalised 0-1 value.
    """
    # mesh IS the MDLDangly object; dangly attrs are on it directly
    try:
        gr.dangly_displacement = float(getattr(mesh, 'displacement', 0.5) or 0.5)
    except Exception:
        pass
    try:
        gr.dangly_tightness = float(getattr(mesh, 'tightness', 0.1) or 0.1)
    except Exception:
        pass
    try:
        gr.dangly_period = float(getattr(mesh, 'period', 1.0) or 1.0)
    except Exception:
        pass

    # Constraint weights (MDLConstraint.type = int(float_0_255 * 1_000_000))
    try:
        raw_constraints = list(getattr(mesh, 'constraints', []) or [])
        parsed = []
        for c in raw_constraints:
            if hasattr(c, 'type'):
                # PyKotor encoding: int(weight_0_255 * 1e6)
                val_0_255 = float(c.type) / 1_000_000.0
                parsed.append(max(0.0, min(1.0, val_0_255 / 255.0)))
            elif hasattr(c, 'weight'):
                v = float(c.weight)
                parsed.append(max(0.0, min(1.0, v / 255.0)) if v > 1.0 else v)
            else:
                try:
                    v = float(c)
                    parsed.append(max(0.0, min(1.0, v / 255.0)) if v > 1.0 else v)
                except Exception:
                    parsed.append(0.0)
        gr.dangly_constraints = parsed
    except Exception as exc:
        log.debug("_read_dangly '%s': constraints failed — %s", gr.name, exc)


# ── Controllers ───────────────────────────────────────────────────────────────

# Controller type ID → name mapping (subset; extended mapping in MDLBinaryParser._parse_controllers)
def _read_light(pk_node, gr: ModelNode) -> None:
    """Read binary Aurora light-header flags exposed by PyKotor."""
    light = getattr(pk_node, 'light', None)
    if light is None:
        return
    gr.light_ambient_only = bool(getattr(light, 'ambient_only', 0))
    gr.light_dynamic = int(getattr(light, 'dynamic_type', 0) or 0)
    gr.light_shadow = bool(getattr(light, 'shadow', 0))
    gr.light_flare = bool(getattr(light, 'flare', 0))
    gr.light_fading = bool(getattr(light, 'fading_light', 0))
    gr.light_flare_radius = float(getattr(light, 'flare_radius', 0.0) or 0.0)
    gr.light_priority = int(getattr(light, 'light_priority', 0) or 0)
    gr.light_affect_dynamic = bool(getattr(light, 'affect_dynamic', 0))
    gr.light_flare_sizes = [float(value) for value in (getattr(light, 'flare_sizes', None) or [])]
    gr.light_flare_positions = [float(value) for value in (getattr(light, 'flare_positions', None) or [])]
    gr.light_flare_color_shifts = [
        tuple(float(component) for component in value[:3])
        for value in (getattr(light, 'flare_color_shifts', None) or [])
    ]
    gr.light_flare_textures = [str(value) for value in (getattr(light, 'flare_textures', None) or [])]


def _read_emitter(pk_node, gr: ModelNode) -> None:
    """Preserve the complete fixed emitter sub-header used by the writer."""

    emitter = getattr(pk_node, 'emitter', None)
    if emitter is None:
        return
    binary = getattr(emitter, '_gr_binary_emitter', None)
    unknown1 = int(binary.get('unknown1', 0) or 0) if isinstance(binary, dict) else 0
    gr.emitter_params = {
        'deadspace': float(getattr(emitter, 'dead_space', 0.0) or 0.0),
        'blastradius': float(getattr(emitter, 'blast_radius', 0.0) or 0.0),
        'blastlength': float(getattr(emitter, 'blast_length', 0.0) or 0.0),
        'numbranches': int(getattr(emitter, 'branch_count', 0) or 0),
        'controlptsmoothing': int(getattr(emitter, 'control_point_smoothing', 0) or 0),
        'xgrid': int(getattr(emitter, 'x_grid', 0) or 0),
        'ygrid': int(getattr(emitter, 'y_grid', 0) or 0),
        'spawntype': int(getattr(emitter, 'spawn_type', 0) or 0),
        'update': str(getattr(emitter, 'update', '') or ''),
        'emitter_render': str(getattr(emitter, 'render', '') or ''),
        'blend': str(getattr(emitter, 'blend', '') or ''),
        'texture': str(getattr(emitter, 'texture', '') or ''),
        'chunkname': str(getattr(emitter, 'chunk_name', '') or ''),
        'twosidedtex': int(getattr(emitter, 'two_sided_texture', 0) or 0),
        'loop': int(getattr(emitter, 'loop', 0) or 0),
        'renderorder': int(getattr(emitter, 'render_order', 0) or 0),
        'frameblending': int(getattr(emitter, 'frame_blender', 0) or 0),
        'depth_texture_name': str(getattr(emitter, 'depth_texture', '') or ''),
        'unknown1': unknown1 & 0xFF,
        'flags': int(getattr(emitter, 'flags', 0) or 0),
    }


_CT_NAMES: Dict[int, str] = {
    8:   'position',
    20:  'orientation',
    36:  'scale',
    76:  'color',
    88:  'radius',
    96:  'shadow_radius',
    100: 'selfillum_color',
    128: 'alpha',
    132: 'alpha',
    140: 'multiplier',
}
# Canonical column counts per controller type
_CT_COLS: Dict[int, int] = {
    8: 3, 20: 4, 36: 1, 76: 3, 88: 1, 96: 1, 100: 3, 128: 1, 132: 1, 140: 1,
}


def _read_controllers(pk_node, gr: ModelNode) -> None:
    """Copy controller keyframe data directly from PyKotor MDLController list.

    Each controller dict has keys: type, name, times, values, columns. Binary
    round-trip metadata is retained under ``binary_*`` keys; Bezier channels
    keep their expanded value/in/out-tangent rows in ``binary_bezier_rows``.
    """
    def _with_binary_metadata(ctrl, payload: Dict) -> Dict:
        raw = getattr(ctrl, "_gr_binary_controller", None)
        if isinstance(raw, dict):
            if "unknown0" in raw:
                payload["binary_unknown0"] = int(raw.get("unknown0") or 0)
            if "column_count" in raw:
                payload["binary_column_count"] = int(raw.get("column_count") or payload.get("columns", 1))
            unknown1 = raw.get("unknown1")
            if isinstance(unknown1, (list, tuple)):
                payload["binary_unknown1"] = [int(v) & 0xFF for v in list(unknown1)[:3]]
            words = raw.get("compressed_quaternion_words")
            if isinstance(words, (list, tuple)):
                payload["binary_compressed_quaternion_words"] = [int(v) & 0xFFFFFFFF for v in words]
            binary_columns = int(raw.get("column_count") or 0)
            if binary_columns & 0x10:
                bezier_rows = raw.get("bezier_rows")
                controller_rows = list(getattr(ctrl, "rows", None) or ())
                if (
                    not isinstance(bezier_rows, (list, tuple))
                    or len(bezier_rows) < len(controller_rows)
                ):
                    bezier_rows = [
                        list(getattr(row, "data", ()) or ())
                        for row in controller_rows
                    ]
                payload["is_bezier"] = True
                payload["binary_bezier_rows"] = [
                    [float(value) for value in tuple(row or ())]
                    for row in bezier_rows
                ]
        elif bool(getattr(ctrl, "is_bezier", False)):
            # ASCII or third-party controller objects may expose Bezier rows
            # without GhostRigger's raw-entry metadata.  Keep their expanded
            # value/in/out-tangent rows in the same domain-model field; the
            # writer will derive the 0x10 flag from the logical column count.
            payload["is_bezier"] = True
            payload["binary_bezier_rows"] = [
                [float(value) for value in tuple(getattr(row, "data", ()) or ())]
                for row in (getattr(ctrl, "rows", None) or ())
            ]
        return payload

    for ctrl in (getattr(pk_node, 'controllers', None) or []):
        ct    = int(ctrl.controller_type)
        rows  = ctrl.rows
        if not rows:
            continue
        first = rows[0].data
        times  = [float(r.time) for r in rows]
        values = [[float(v) for v in r.data] for r in rows]
        name = _CT_NAMES.get(ct, f'ctrl_{ct}')
        raw = getattr(ctrl, "_gr_binary_controller", None)
        raw_columns = int(raw.get("column_count", 0) or 0) if isinstance(raw, dict) else 0
        cols = (raw_columns & 0x0F) or _CT_COLS.get(ct, len(first) if first else 1)

        # Emitter IDs overlap mesh/light IDs but use different widths and
        # meanings.  Preserve their raw rows instead of applying the generic
        # mesh controller table (for example emitter type 100 is one float,
        # while mesh type 100 is a three-float self-illumination colour).
        if gr.is_emitter:
            gr.controllers.append(_with_binary_metadata(ctrl, {
                'type': ct,
                'name': name,
                'columns': cols,
                'times': times,
                'values': [v[:cols] for v in values],
            }))
            continue

        if ct == _CT_POS and len(first) >= 3:
            gr.controllers.append(_with_binary_metadata(ctrl, {'type': _CT_POS,   'name': name, 'columns': 3,
                                   'times': times, 'values': [v[:3] for v in values]}))
        elif ct == _CT_ORI and len(first) >= 4:
            gr.controllers.append(_with_binary_metadata(ctrl, {'type': _CT_ORI,   'name': name, 'columns': 4,
                                   'times': times, 'values': [v[:4] for v in values]}))
        elif ct == _CT_SCALE:
            gr.controllers.append(_with_binary_metadata(ctrl, {'type': _CT_SCALE, 'name': name, 'columns': 1,
                                   'times': times, 'values': [[v[0]] for v in values]}))
        elif ct == _CT_COLOR and len(first) >= 3:
            gr.controllers.append(_with_binary_metadata(ctrl, {'type': _CT_COLOR, 'name': name, 'columns': 3,
                                   'times': times, 'values': [v[:3] for v in values]}))
        elif ct == _CT_RADIUS:
            gr.controllers.append(_with_binary_metadata(ctrl, {'type': _CT_RADIUS, 'name': name, 'columns': 1,
                                   'times': times, 'values': [[v[0]] for v in values]}))
        elif ct == _CT_ALPHA:
            gr.controllers.append(_with_binary_metadata(ctrl, {'type': _CT_ALPHA, 'name': name, 'columns': 1,
                                   'times': times, 'values': [[v[0]] for v in values]}))
        elif ct == _CT_MULT:
            gr.controllers.append(_with_binary_metadata(ctrl, {'type': _CT_MULT, 'name': name, 'columns': 1,
                                   'times': times, 'values': [[v[0]] for v in values]}))
        elif ct == _CT_ILLUM and len(first) >= 3:
            gr.controllers.append(_with_binary_metadata(ctrl, {'type': _CT_ILLUM, 'name': name, 'columns': 3,
                                   'times': times, 'values': [v[:3] for v in values]}))
        else:
            # Preserve all other controller types with metadata
            gr.controllers.append(_with_binary_metadata(ctrl, {'type': ct, 'name': name, 'columns': cols,
                                   'times': times, 'values': values}))


# ── Animations ────────────────────────────────────────────────────────────────

def _convert_anim(pk_anim) -> Optional[Animation]:
    try:
        anim = Animation()
        anim.name            = pk_anim.name or 'default'
        anim.length          = float(getattr(pk_anim, 'length', None) or
                                     getattr(pk_anim, 'anim_length', 0) or 0.0)
        transition_time = getattr(pk_anim, 'transition_time', None)
        if transition_time is None:
            transition_time = getattr(pk_anim, 'transition_length', None)
        anim.transition_time = float(0.25 if transition_time is None else transition_time)
        anim.anim_root = str(getattr(pk_anim, 'root_model', '') or '')

        for evt in (getattr(pk_anim, 'events', None) or []):
            t = float(getattr(evt, 'activation_time', None) or 0.0)
            n = str(getattr(evt, 'name', '') or '')
            anim.events.append(AnimEvent(time=t, name=n))

        # Animation nodes -- PyKotor exposes the donor animation tree through
        # child links even though parent links are not populated on those nodes.
        # Preserve that sparse tree so binary exports can round-trip creature
        # animations without manufacturing full-geometry animation branches.
        pk_nodes = list(_walk_nodes(pk_anim))
        converted_by_id: Dict[int, ModelNode] = {}
        for pk_anode in pk_nodes:
            an = ModelNode()
            an.name  = pk_anode.name or 'node'
            an.index = int(pk_anode.node_id)
            _read_controllers(pk_anode, an)
            anim.nodes.append(an)
            converted_by_id[id(pk_anode)] = an

        for pk_anode in pk_nodes:
            parent = converted_by_id.get(id(pk_anode))
            if parent is None:
                continue
            for pk_child in getattr(pk_anode, 'children', []) or []:
                child = converted_by_id.get(id(pk_child))
                if child is None or child is parent:
                    continue
                child.parent = parent
                if child not in parent.children:
                    parent.children.append(child)

        return anim
    except Exception as exc:
        log.error("_convert_anim '%s': %s", getattr(pk_anim, 'name', '?'), exc, exc_info=True)
        return None


def _walk_nodes(pk_obj):
    """Yield all nodes via pk_obj.all_nodes(), or DFS from .root as fallback."""
    try:
        yield from pk_obj.all_nodes()
        return
    except Exception:
        pass
    root = getattr(pk_obj, 'root', None)
    if root is None:
        return
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        for c in reversed(list(getattr(n, 'children', []) or [])):
            stack.append(c)


# ── Post-load fixups ──────────────────────────────────────────────────────────

def _fill_missing_normals(model: KotorModel) -> None:
    """Compute flat normals for any mesh node that PyKotor didn't provide them for."""
    if model.root_node is None:
        return
    stack = [model.root_node]
    while stack:
        node = stack.pop()
        for c in reversed(node.children):
            stack.append(c)
        if not node.is_mesh or not node.vertices:
            continue
        if node.normals and len(node.normals) == len(node.vertices):
            continue
        verts = node.vertices
        acc   = [[0.0, 0.0, 0.0] for _ in verts]
        cnt   = [0] * len(verts)
        for face in (node.faces or []):
            if len(face) < 3:
                continue
            a, b, c = face[0], face[1], face[2]
            if max(a, b, c) >= len(verts):
                continue
            ax, ay, az = verts[a]
            bx, by, bz = verts[b]
            cx, cy, cz = verts[c]
            ux, uy, uz = bx - ax, by - ay, bz - az
            vx, vy, vz = cx - ax, cy - ay, cz - az
            nx = uy*vz - uz*vy
            ny = uz*vx - ux*vz
            nz = ux*vy - uy*vx
            ln = math.sqrt(nx*nx + ny*ny + nz*nz) or 1.0
            nx /= ln; ny /= ln; nz /= ln
            for vi in (a, b, c):
                acc[vi][0] += nx; acc[vi][1] += ny; acc[vi][2] += nz
                cnt[vi] += 1
        result = []
        for i in range(len(verts)):
            if cnt[i]:
                nx, ny, nz = acc[i][0]/cnt[i], acc[i][1]/cnt[i], acc[i][2]/cnt[i]
                ln = math.sqrt(nx*nx + ny*ny + nz*nz) or 1.0
                result.append((nx/ln, ny/ln, nz/ln))
            else:
                result.append((0.0, 0.0, 1.0))
        node.normals = result


def _apply_bind_pose(model: KotorModel) -> None:
    """Push static bind-pose controllers into node fields.

    Controller types applied:
      8   (position)       → node.position    (only when position is zero)
      20  (orientation)    → node.rotation    (only when rotation is identity)
      100 (selfillum_color)→ node.selfillum
      128 (alpha fallback) → node.alpha       (only when alpha is still default 1.0)
      132 (alpha)          → node.alpha
    """
    _ZERO_POS = (0.0, 0.0, 0.0)
    _IDENT_ROT = (0.0, 0.0, 0.0, 1.0)

    if model.root_node is None:
        return
    stack = [model.root_node]
    while stack:
        node = stack.pop()
        for c in reversed(node.children):
            stack.append(c)
        for ctrl in node.controllers:
            ct   = ctrl.get('type')
            vals = ctrl.get('values', [])
            if not vals:
                continue
            v0 = vals[0]
            if ct == _CT_POS and len(v0) >= 3:
                # ctype == 8: position — only override when zero
                if node.position == _ZERO_POS:
                    node.position = tuple(v0[:3])
            elif ct == _CT_ORI and len(v0) >= 4:
                # ctype == 20: orientation — the controller IS the authoritative
                # bind-pose source in binary MDL; the header rotation is often left
                # at identity/zero as a placeholder (verified against old mdl_parser.py
                # _apply_bind_pose_controllers which always applied the controller value).
                # Normalize and apply as long as the quaternion has non-zero magnitude.
                import math as _math
                cx, cy, cz, cw = float(v0[0]), float(v0[1]), float(v0[2]), float(v0[3])
                mag = _math.sqrt(cx*cx + cy*cy + cz*cz + cw*cw)
                if mag > 1e-9:
                    node.rotation = (cx/mag, cy/mag, cz/mag, cw/mag)
            elif ct == 100 and len(v0) >= 3:
                # ctype == 100: selfillum_color (CTRL_MESH_SELFILLUMCOLOR)
                node.selfillum = tuple(v0[:3])
            elif ct == _CT_COLOR and len(v0) >= 3 and node.is_light:
                # Keep the sign: the engine lights with negative (darkening) colours as authored
                # (RE02: 0x0050A050 flags rgb.r < 0; the vertex program uses PartLight +0x4C as is),
                # and 395 vanilla light nodes in 8 K2 modules are negative (vanilla_relight census).
                # Viewport light upload still clamps for display; the lightmap bake keeps the sign.
                node.light_color = tuple(float(x) for x in v0[:3])
            elif ct == _CT_RADIUS and len(v0) >= 1 and node.is_light:
                node.light_radius = max(0.0, float(v0[0]))
            elif ct == _CT_MULT and len(v0) >= 1 and node.is_light:
                node.light_multiplier = max(0.0, float(v0[0]))
            elif ct == 132 and len(v0) >= 1:
                # ctype == 132: alpha (CTRL_MESH_ALPHA)
                node.alpha = float(v0[0])
            elif ct == 128 and len(v0) >= 1:
                # ctype == 128: alpha fallback (xoreos CTRL_ALPHA) — only default
                if node.alpha == 1.0:
                    node.alpha = float(v0[0])


# =============================================================================
#  reparent_head_nodes — xoreos parity helper (opt-in, not in default path)
# =============================================================================
#
# xoreos' Model_KotOR::reparentHeadNodes() walks the model tree and moves any
# node named exactly ``head`` or ``tongue`` so it becomes a direct child of
# the model root.  The rationale in xoreos is explicitly a perf optimisation
# for the animation system's per-frame node lookup — NOT a geometric fix —
# and the implementation explicitly preserves each moved node's world-space
# transform by recomputing its local transform against the new parent.  See
# ``xoreos/src/graphics/aurora/model_kotor.cpp`` (``reparentHeadNodes``) and
# ``modelnode.cpp`` (``reparentTo``).
#
# GhostRigger's ``ModelNode.world_transform()`` already accumulates the full
# parent chain, so world positions are identical regardless of tree depth.
# We therefore do NOT call this from the default load pipeline; calling it
# would change the tree shape (and thus diffs of ``model_inspector.py``
# output) without changing rendering.  Callers that want 1:1 xoreos tree
# topology for e.g. animation-system comparisons can invoke this explicitly
# after loading:
#
#     model = load_model_from_file(mdl)
#     reparent_head_nodes(model)   # explicit xoreos-parity flattening
#
# The function is idempotent — a second call finds the target nodes already
# parented to root and returns ``0``.

def _iter_nodes_dfs(root: ModelNode):
    """Yield every node in the subtree rooted at ``root`` (DFS, cycle-safe)."""
    stack: List[ModelNode] = [root]
    visited: set = set()
    while stack:
        n = stack.pop()
        if n is None:
            continue
        nid = id(n)
        if nid in visited:
            continue
        visited.add(nid)
        yield n
        for c in reversed(n.children):
            stack.append(c)


def _find_node_exact(root: ModelNode, target_name: str) -> Optional[ModelNode]:
    """Return the first node whose name matches ``target_name`` (case-insens.).

    Uses an *exact* lower-case match — we deliberately do not substring match,
    because in KotOR body MDLs ``head_hook`` / ``headTip`` / ``tongue_hook``
    are attachment/dummy nodes that are meant to stay where they are in the
    tree.  Only the geometry-bearing ``head`` / ``tongue`` nodes should be
    flattened.  This mirrors xoreos' ``Model::getNode("head")`` exact-match
    behaviour.
    """
    needle = target_name.lower()
    for node in _iter_nodes_dfs(root):
        if (node.name or '').lower() == needle:
            return node
    return None


def reparent_head_nodes(model: KotorModel) -> int:
    """Flatten ``head`` / ``tongue`` under the model root, preserving world xform.

    xoreos parity helper.  **Not** wired into ``load_model_from_file`` /
    ``load_model_from_bytes`` — this changes the node tree topology, which in
    turn changes diagnostic output (``model_inspector.py``) even though the
    rendered image is unchanged.  Only call this when you need strict xoreos
    tree parity (animation-system cross-validation, reverse-engineering
    diffs, etc.).

    Algorithm per target node::

        1. Remember its world transform   (wp, wr)  := node.world_transform()
        2. Detach from old parent         parent.children.remove(node)
        3. Attach to root                 root.children.append(node);
                                          node.parent = root
        4. Rewrite its local transform so (wp', wr') == (wp, wr).
           Since the new parent is the root (world at identity, rotation
           identity in our convention), the new local transform *is* the
           preserved world transform — no per-parent de-rotation needed.

    Parameters
    ----------
    model : KotorModel
        Model to mutate in-place.  ``None`` / missing root are safe no-ops.

    Returns
    -------
    int
        Number of nodes reparented.  ``0`` when neither ``head`` nor
        ``tongue`` exist, or when both are already direct children of root.
    """
    if model is None or model.root_node is None:
        return 0

    root = model.root_node
    reparented = 0

    # xoreos reparents exactly these two nodes.  We keep the same set for
    # parity; if more need flattening later (e.g. 'gogglesmid'), add them
    # here after confirming against xoreos source.
    for target_name in ('head', 'tongue'):
        node = _find_node_exact(root, target_name)
        if node is None:
            continue
        if node is root:
            # Pathological: a model whose root is literally named "head".
            # Nothing to do — already at the "top".
            continue
        if node.parent is root:
            # Already a direct child of root — idempotent skip.
            continue

        # Capture the world transform BEFORE mutating the tree.  After we
        # move the node its parent chain changes, so a second call would
        # give a different answer.
        try:
            world_pos, world_rot = node.world_transform()
        except Exception as exc:  # pragma: no cover — defensive only
            log.warning(
                "reparent_head_nodes: skipping %r — world_transform failed: %s",
                target_name, exc,
            )
            continue

        # Detach from the old parent.  ``remove`` is O(N) in children but
        # skeleton fan-out is small (<64 children per node in practice).
        old_parent = node.parent
        if old_parent is not None:
            try:
                old_parent.children.remove(node)
            except ValueError:
                # Child list was mutated mid-flight (shouldn't happen with
                # a single-threaded loader) — recover by filtering by id.
                old_parent.children = [
                    c for c in old_parent.children if c is not node
                ]

        # Re-parent under root and rewrite the local transform.  Our root
        # always sits at the world origin with identity rotation (model-local
        # == world), so the new local is exactly the preserved world.
        node.parent = root
        root.children.append(node)
        node.position = (float(world_pos[0]),
                         float(world_pos[1]),
                         float(world_pos[2]))
        node.rotation = (float(world_rot[0]),
                         float(world_rot[1]),
                         float(world_rot[2]),
                         float(world_rot[3]))

        log.debug(
            "reparent_head_nodes: moved '%s' from parent '%s' to root; "
            "world pos preserved at (%+.4f, %+.4f, %+.4f)",
            target_name,
            getattr(old_parent, 'name', '?'),
            world_pos[0], world_pos[1], world_pos[2],
        )
        reparented += 1

    if reparented:
        log.info("reparent_head_nodes: reparented %d node(s) to root",
                 reparented)
    return reparented


# =============================================================================
#  build_donor_skin_data_from_model — anatomical-partition donor assembly
# =============================================================================
#
# Ownership note (PR C.1 / T2508): the anatomical partitioner
# (``src.math.anatomical_partition``) consumes a frame-consistent
# ``DonorSkinData`` but does not build one — building it from a loaded model is a
# resource/extraction concern, so it lives here next to ``_read_skin_weights``
# (which already reads the same qBone/tBone skin arrays).  Core.Math stays
# model-agnostic; this function only *reads* duck-typed model/node attributes.
#
# Frame correctness (the whole reason PR C.1 exists): a KotOR creature ships as
# several skin nodes with distinct local transforms (Drexl: offsets up to ~2u,
# ``tailGeo`` also carries a rotation).  Concatenating ``node.vertices`` raw
# mixes node-local vertices with world-space bone pivots.  We therefore transform
# every node's vertices by the node's full parent-chain WORLD transform before
# accumulation, so vertices and ``bone_positions`` share one world frame.  Bone
# pivots already come from ``node.bone_world_position()`` (world) and are left as
# they are.

def _quat_rotate_xyzw(q, v):
    """Rotate ``v`` by quaternion ``q=[x,y,z,w]`` (matches model_data._quat_rotate)."""
    qx, qy, qz, qw = q
    vx, vy, vz = v
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def build_donor_skin_data_from_model(model):
    """Assemble a frame-consistent ``DonorSkinData`` from a loaded KotOR model.

    Concatenates every skin node's geometry into a single WORLD-frame donor:

    - each node's vertices are transformed by the node's full parent-chain world
      transform (translation AND rotation) before accumulation;
    - ``bone_positions`` come from ``node.bone_world_position()`` (already world);
    - per-vertex bone indices are remapped from each node's local ``bone_map`` to
      a shared global bone table.

    Returns a ``DonorSkinData`` with ``frame="world_space_v1"``.  Raises
    ``ValueError`` if the model exposes no usable skin nodes.
    """
    import numpy as np

    # Canonical (merged-``src``-namespace) imports; lazy so module import stays
    # cheap and so the Rendering/Math dependencies are only touched at call time
    # (avoids any load-order cycle between the loader and the renderer helper).
    from src.math.anatomical_partition import DonorSkinData
    from src.core.rendering.skeleton_render_data import extract_skinning_arrays
    try:
        from src.core.geometry.model_data import _quat_rotate as _qrot
    except Exception:  # pragma: no cover - defensive across import styles
        _qrot = _quat_rotate_xyzw

    def _verts_to_world(node, verts):
        # PR C.1a: use the canonical ModelNode.world_transform() bind-world
        # placement — the same accessor OBJExporter._node_bind_world_verts uses.
        # Unlike a raw parent-chain walk, world_transform() collapses parent 180°
        # bind-flips (the c_brith / Wardroid fix), so this is correct for
        # non-Drexl donors too (byte-identical to the old walk on Drexl).
        v = np.asarray(verts, dtype=np.float64)
        wp, wo = node.world_transform()
        wp = np.asarray(wp, dtype=np.float64)
        if (wo[0] ** 2 + wo[1] ** 2 + wo[2] ** 2) ** 0.5 < 0.001:
            return v + wp  # identity rotation → translation only
        return np.array([_qrot(tuple(wo), tuple(p)) for p in v], dtype=np.float64) + wp

    nodes = list(model.all_nodes())
    lookup = {str(n.name).lower(): n for n in nodes}
    skin_nodes = [
        n for n in nodes if bool(getattr(n, "is_skin", False)) and getattr(n, "vertices", None)
    ]
    if not skin_nodes:
        raise ValueError(
            "build_donor_skin_data_from_model: model has no skin nodes; "
            "cannot assemble an anatomical-partition donor."
        )

    all_v, all_f, all_bi, all_bw = [], [], [], []
    global_names: List[str] = []
    global_index: Dict[str, int] = {}
    vert_offset = 0
    for node in skin_nodes:
        v_local = np.asarray(node.vertices, dtype=np.float64)
        n_local = len(v_local)
        skin = extract_skinning_arrays(node, n_local)
        if skin.bone_indices is None or skin.bone_weights is None:
            continue  # skin node without usable weights — skip (offset unchanged)

        v_world = _verts_to_world(node, v_local)
        faces = np.asarray(node.faces, dtype=np.int64)
        bi_local = np.asarray(skin.bone_indices, dtype=np.int64)
        bw = np.asarray(skin.bone_weights, dtype=np.float64)

        bone_map = list(getattr(node, "bone_map", []) or [])
        local_to_global = []
        for name in bone_map:
            key = str(name).lower()
            if key not in global_index:
                global_index[key] = len(global_names)
                global_names.append(str(name))
            local_to_global.append(global_index[key])
        l2g = np.asarray(local_to_global, dtype=np.int64)

        valid = (bi_local >= 0) & (bi_local < len(bone_map))
        bi_global = np.where(
            valid, l2g[np.clip(bi_local, 0, max(len(bone_map) - 1, 0))], -1
        )

        all_v.append(v_world)
        all_f.append(faces + vert_offset)
        all_bi.append(bi_global)
        all_bw.append(bw)
        vert_offset += n_local

    if not all_v:
        raise ValueError(
            "build_donor_skin_data_from_model: skin nodes present but none had "
            "usable skin weights."
        )

    vertices = np.vstack(all_v)
    faces = np.vstack(all_f)
    bone_indices = np.vstack(all_bi)
    bone_weights = np.vstack(all_bw)

    bone_positions = np.zeros((len(global_names), 3), dtype=np.float64)
    for i, name in enumerate(global_names):
        nd = lookup.get(str(name).lower())
        if nd is not None:
            try:
                bone_positions[i] = np.asarray(nd.bone_world_position()[:3], dtype=np.float64)
            except Exception:
                pass

    return DonorSkinData(
        vertices=vertices,
        faces=faces,
        bone_indices=bone_indices,
        bone_weights=bone_weights,
        bone_names=global_names,
        bone_positions=bone_positions,
        frame="world_space_v1",
    )
