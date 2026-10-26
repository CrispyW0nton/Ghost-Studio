"""
MDL Binary Writer  –  writes KotOR 1/2 .mdl + .mdx files
==========================================================
Phase 7.1 implementation.

References:
  • PyKotor/resource/formats/mdl/io_mdl.py   (4,783 lines, NickHugi/PyKotor)
  • Kotor.NET/Formats/KotorMDL/MDLBinaryWriter.cs  (575 lines)
  • KotorBlender/io_scene_kotor/format/mdl/reader.py  (seedhartha)
  • xoreos/src/aurora/model_nwn.cpp
  • GhostRigger mdl_parser.py  (verified offsets)

Binary layout overview
----------------------
[0 .. 11]      File header   (12 bytes):  unused(4) mdl_size(4) mdx_size(4)
[12 ..]        BASE = 12

Geometry header  (80 bytes, at BASE+0):
  +0  funcptr1   uint32
  +4  funcptr2   uint32
  +8  name       char[32]
  +40 root_off   uint32    (relative to BASE)
  +44 node_count uint32
  +48 unknown0   uint32
  +52 ref_count  uint32
  +56..75        runtime/ref padding
  +76 geo_type   uint8    (MaxTree subtype: 2=model, 5=animation)
  +77..79        padding / unknown bytes (3 bytes)

Model fields  (116 bytes, at BASE+80; together with geometry = _ModelHeader 0xC4):
  +0   model_type       uint8
  +1   subclassification uint8
  +2   padding0          uint8
  +3   fog               uint8
  +4   child_model_count uint32
  +8   anim_array_off    uint32   (relative to BASE)
  +12  anim_count        uint32
  +16  anim_count2       uint32
  +20  parent_model_ptr  uint32
  +24  bb_min            3×float
  +36  bb_max            3×float
  +48  radius            float
  +52  anim_scale        float
  +56  supermodel        char[32]
  +88  offset_to_super_root uint32
  +92  mdx_data_buffer_offset uint32
  +96  mdx_size          uint32
  +100 mdx_offset        uint32
  +104 name_offsets_off  uint32
  +108 name_offsets_count uint32
  +112 name_offsets_count2 uint32

  Total _ModelHeader = 196 bytes → name offset table usually starts at BASE+196

  → Name offset table starts at BASE + names_off
  → Each entry: uint32 offset (relative to BASE) pointing to null-term string

Node header  (80 bytes per node):
  +0   node_type    uint16
  +2   index_num    uint16
  +4   node_num     uint16
  +6   pad          uint16
  +8   root_off     uint32   (relative to BASE)
  +12  parent_off   uint32   (relative to BASE)
  +16  position     3×float
  +28  rotation     4×float  (file layout is w,x,y,z; our ModelNode uses x,y,z,w internally —
                               verified against PyKotor _NodeHeader.read, KotorBlender
                               reader.py, and KotOR.js OdysseyModelNodeHeader)
  +44  child_arr_off uint32  (relative to BASE)
  +48  child_cnt    uint32
  +52  child_cnt2   uint32
  +56  ctrl_arr_off uint32   (relative to BASE)
  +60  ctrl_cnt     uint32
  +64  ctrl_cnt2    uint32
  +68  ctrl_data_off uint32  (relative to BASE)
  +72  ctrl_data_cnt uint32
  +76  ctrl_data_cnt2 uint32
  Total: 80 bytes

Mesh header  (332 bytes for K1, 340 bytes for K2 [+8 dirt fields]):
  (see _write_mesh_header for full layout)

Skin header  (100 bytes after mesh header):
  (see _write_skin_header)

Dangly header  (28 bytes after mesh header):
  (see _write_dangly_header)

Emitter header  (224 bytes after base node header):
  (see _write_emitter_header)

MDX companion buffer:
  Per-vertex stride (determined by _mdx_stride):
    Vertex XYZ  3×float32 (always)
    Normal      3×float32 (if has normals)
    UV0         2×float32 (if has UVs)
    LM UV       2×float32 (if has uvs_lm)
    Tangent basis 9×float32 (bitangent, tangent, tangent-space normal)
    Skin weights 4×float32 + 4×float32 bone_refs (if SKIN)
"""

from __future__ import annotations

import struct
import math
import logging
import copy
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..geometry.model_data import (
    Animation, AnimEvent, BoneWeight, GameVersion, KotorModel,
    ModelNode, NodeFlags, VertexSkinData,
)

log = logging.getLogger(__name__)

# ─────────────────────────────  Constants  ──────────────────────────────────
#
# PC function pointer pairs — verified against KotorBlender types.py
# (`io_scene_kotor/format/mdl/types.py` in OpenKotOR/kotorblender) and
# PyKotor's ``io_mdl._TrimeshHeader`` K1/K2 function pointer constants.
#
# There are FOUR distinct FP-pair families in a KotOR binary MDL:
#
#   1. MODEL_FN_PTR  — the TOP-LEVEL geometry header (at BASE + 0).  This is the
#                      one written into the model's outer 80-byte geometry
#                      header.  These values differ between K1/K2 PC vs XBOX.
#   2. ANIM_FN_PTR   — the geometry header embedded in every Animation block.
#                      Completely distinct values from MODEL_FN_PTR.
#   3. MESH_FN_PTR   — the first 8 bytes of every TrimeshHeader (type=MESH).
#                      PyKotor calls these ``K1_FUNCTION_POINTER0/1``.
#   4. SKIN_FN_PTR / DANGLY_FN_PTR — replace MESH_FN_PTR when the node carries
#                      the SKIN or DANGLY flag respectively.
#
# Prior revisions of this writer mixed families (used MODEL_FN_PTR in place of
# MESH_FN_PTR, and a geometry fp2 in place of ANIM_FN_PTR_2), which produces
# files that crash the engine and confuse PyKotor's reader.

# ── Model (top-level geometry header) ────────────────────────────────────────
_K1_MODEL_FP1 = 4273776   # 0x0041BCC0  (MODEL_FN_PTR_1_K1_PC)
_K1_MODEL_FP2 = 4216096   # 0x0040A120  (MODEL_FN_PTR_2_K1_PC)
_K2_MODEL_FP1 = 4285200   # 0x0041E850  (MODEL_FN_PTR_1_K2_PC)
_K2_MODEL_FP2 = 4216320   # 0x0040A200  (MODEL_FN_PTR_2_K2_PC)

# Backwards-compat aliases (legacy name used elsewhere in the module).
_K1_GEOM_FP1 = _K1_MODEL_FP1
_K1_GEOM_FP2 = _K1_MODEL_FP2
_K2_GEOM_FP1 = _K2_MODEL_FP1
_K2_GEOM_FP2 = _K2_MODEL_FP2

# ── Animation geometry header ────────────────────────────────────────────────
# Used for the geometry header embedded inside each animation block.
# Source: KotorBlender types.py ANIM_FN_PTR_{1,2}_K{1,2}_PC.
_K1_ANIM_FP1 = 4273392   # 0x0041BB30  (ANIM_FN_PTR_1_K1_PC)
_K1_ANIM_FP2 = 4451552   # 0x00440F20  (ANIM_FN_PTR_2_K1_PC)  ← NOT 4216096
_K2_ANIM_FP1 = 4284816   # 0x0041E6D0  (ANIM_FN_PTR_1_K2_PC)
_K2_ANIM_FP2 = 4522928   # 0x00451C70  (ANIM_FN_PTR_2_K2_PC)  ← NOT 4216320

# ── Mesh / Skin / Dangly node-type function pointers ─────────────────────────
# These populate the first 8 bytes of every TrimeshHeader subheader.
# Source: PyKotor ``io_mdl._TrimeshHeader.K*_FUNCTION_POINTER0/1`` and
# KotorBlender types.py MESH/SKIN/DANGLY FN_PTR constants.
_K1_MESH_FP1   = 4216656  # 0x00405750
_K1_MESH_FP2   = 4216672  # 0x00405760
_K2_MESH_FP1   = 4216880  # 0x00405830
_K2_MESH_FP2   = 4216896  # 0x00405840

_K1_SKIN_FP1   = 4216592  # 0x00405710
_K1_SKIN_FP2   = 4216608  # 0x00405720
_K2_SKIN_FP1   = 4216816  # 0x004057F0
_K2_SKIN_FP2   = 4216832  # 0x00405800

# NOTE: KotorBlender/PyKotor list the DANGLY pair as (FP1=0x00405740,
# FP2=0x00405730) — fp2 is lower than fp1 on purpose, this is NOT a swap.
_K1_DANGLY_FP1 = 4216640  # 0x00405740
_K1_DANGLY_FP2 = 4216624  # 0x00405730
_K2_DANGLY_FP1 = 4216864  # 0x00405820
_K2_DANGLY_FP2 = 4216848  # 0x00405810


def _mesh_fp_pair(is_k2: bool, flags: int) -> Tuple[int, int]:
    """Return the (fp1, fp2) pair that belongs at the start of a trimesh
    subheader, selected by node flags (SKIN / DANGLY specialisation)."""
    if flags & NodeFlags.SKIN:
        return (_K2_SKIN_FP1, _K2_SKIN_FP2) if is_k2 else (_K1_SKIN_FP1, _K1_SKIN_FP2)
    if flags & NodeFlags.DANGLY:
        return (_K2_DANGLY_FP1, _K2_DANGLY_FP2) if is_k2 else (_K1_DANGLY_FP1, _K1_DANGLY_FP2)
    return (_K2_MESH_FP1, _K2_MESH_FP2) if is_k2 else (_K1_MESH_FP1, _K1_MESH_FP2)


def _node_flags_for_write(flags: int) -> int:
    """Return engine-style node flags with the base header bit preserved."""

    out = int(flags or 0)
    if out == 0:
        out = int(NodeFlags.HEADER)
    else:
        out |= int(NodeFlags.HEADER)
    if out & (int(NodeFlags.SKIN) | int(NodeFlags.DANGLY) | int(NodeFlags.SABER)):
        out |= int(NodeFlags.MESH)
    # AABB walkmesh nodes embed a full trimesh header on disk (vanilla walk
    # nodes are 0x0221); the loader strips the MESH bit in memory, and
    # writing 0x0201 without a mesh header produces a malformed node.
    if out & int(NodeFlags.AABB):
        out |= int(NodeFlags.MESH)
    return out


_BASE = 12          # all MDL offsets are relative to byte 12
_MODEL_FIELDS_ABS = _BASE + 80
_MODEL_ANIM_ARRAY_OFF_ABS = _MODEL_FIELDS_ABS + 8
_MODEL_ANIM_COUNT_ABS = _MODEL_FIELDS_ABS + 12
_MODEL_ANIM_COUNT2_ABS = _MODEL_FIELDS_ABS + 16
_MODEL_NAME_OFFSETS_OFF_ABS = _MODEL_FIELDS_ABS + 104
_MODEL_NAME_COUNT_ABS = _MODEL_FIELDS_ABS + 108

# Controller type IDs (verified against KotorBlender types.py)
CTRL_POSITION    = 8
CTRL_ORIENTATION = 20
CTRL_SCALE       = 36
CTRL_ALPHA       = 132    # KotorBlender CTRL_MESH_ALPHA
CTRL_SELFILLUM   = 100    # CTRL_MESH_SELFILLUMCOLOR (r,g,b)
CTRL_COLOR       = 76

# ─────────────────────────────  Helper writers  ──────────────────────────────

def _wu32(val: int) -> bytes:
    return struct.pack('<I', val & 0xFFFFFFFF)

def _wi32(val: int) -> bytes:
    return struct.pack('<i', val)

def _wu16(val: int) -> bytes:
    return struct.pack('<H', val & 0xFFFF)

def _sanitize_float(val: float) -> float:
    """Sanitize a float for KOTOR MDL/MDX serialization.

    - NaN/Inf → 0.0 (engine crashes on non-finite values)
    - -0.0 → +0.0 (negative zero crashes the Linux engine per
      KotorModdingKnowledgeBase/docs/engine/known-issues.md)
    """
    try:
        f = float(val)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f):
        return 0.0
    # IEEE 754: -0.0 == 0.0 is True, but copysign distinguishes them.
    if f == 0.0 and math.copysign(1.0, f) < 0.0:
        return 0.0
    return f


def _wf32(val: float) -> bytes:
    return struct.pack('<f', _sanitize_float(val))

def _wstr(s: str, n: int) -> bytes:
    """Encode string into n-byte null-padded ASCII field."""
    b = s.encode('ascii', errors='replace')[:n]
    return b.ljust(n, b'\x00')

def _align4(buf: BytesIO) -> None:
    """Pad BytesIO buffer to 4-byte alignment."""
    pos = buf.tell()
    pad = (4 - (pos % 4)) % 4
    if pad:
        buf.write(b'\x00' * pad)

# ─────────────────────────────  Offset tracking  ─────────────────────────────

@dataclass
class _NodeInfo:
    """Per-node write metadata collected during the first pass."""
    node:              ModelNode
    node_off:          int = 0    # absolute offset of node header in MDL buffer
    child_arr_off:     int = 0    # absolute offset of child pointer array
    ctrl_arr_off:      int = 0    # absolute offset of controller entry array
    ctrl_data_off:     int = 0    # absolute offset of controller float data
    ctrl_data_cnt:     int = 0
    name_index:        int = 0
    parent_off:        int = 0    # absolute offset of parent node (0 if root)
    root_off:          int = 0    # absolute offset of root node
    mesh_faces_off:    int = 0
    mesh_verts_off:    int = 0    # fallback vertex array offset (MDL)
    mdx_data_off:      int = 0    # byte offset into MDX buffer for this node
    mdx_stride:        int = 0    # bytes per vertex in MDX
    skin_bm_off:       int = 0    # bone_map float array offset (in MDL)
    skin_bm_cnt:       int = 0
    dangly_cst_off:    int = 0    # constraint float array offset (in MDL)
    dangly_cst_cnt:    int = 0


# ─────────────────────────────  Main writer  ─────────────────────────────────

class MDLBinaryWriter:
    """
    Serialise a KotorModel to binary MDL + MDX bytes.

    Usage::

        mdl_bytes, mdx_bytes = MDLBinaryWriter().write(model)
        # or to files:
        MDLBinaryWriter().write_files(model, '/path/to/model.mdl')
    """

    def write_files(self, model: KotorModel, mdl_path: str) -> None:
        """Write MDL and MDX files to disk (MDX path derived from mdl_path)."""
        mdl_bytes, mdx_bytes = self.write(model)
        p = Path(mdl_path)
        p.write_bytes(mdl_bytes)
        mdx_path = p.with_suffix('.mdx')
        mdx_path.write_bytes(mdx_bytes)
        log.debug(f"Wrote {len(mdl_bytes)} B MDL + {len(mdx_bytes)} B MDX → {mdl_path}")

    def write_animation_override_files(
        self,
        model: KotorModel,
        source_mdl_path: str | Path,
        source_mdx_path: str | Path | None,
        output_mdl_path: str | Path,
        animation: Animation,
        *,
        replace_existing: bool = True,
    ) -> None:
        """Write an animation-only override while preserving source mesh bytes.

        Retarget exports should not rebuild PMBAM's skin payload. This path
        keeps the source MDL geometry/name/node/mesh bytes and the source MDX
        buffer byte-for-byte, then appends a fresh local animation table and the
        supplied animation block. Existing local animations remain referenced
        unless ``replace_existing`` excludes one with the same name.
        """

        mdl_bytes, mdx_bytes = self.inject_animation_override_bytes(
            model,
            Path(source_mdl_path).read_bytes(),
            Path(source_mdx_path).read_bytes() if source_mdx_path and Path(source_mdx_path).exists() else b"",
            animation,
            replace_existing=replace_existing,
        )
        output_mdl = Path(output_mdl_path)
        output_mdl.write_bytes(mdl_bytes)
        output_mdl.with_suffix(".mdx").write_bytes(mdx_bytes)

    def inject_animation_override_bytes(
        self,
        model: KotorModel,
        source_mdl_bytes: bytes,
        source_mdx_bytes: bytes,
        animation: Animation,
        *,
        replace_existing: bool = True,
    ) -> Tuple[bytes, bytes]:
        """Return MDL/MDX bytes for a surgical animation-block injection."""

        if len(source_mdl_bytes) < _BASE + 196:
            raise ValueError("Source MDL is too small to contain a full model header.")

        self._prepare_animation_only_state(model, source_mdl_bytes)

        old_offsets = self._read_animation_offsets(source_mdl_bytes)
        slot_name = str(getattr(animation, "name", "") or "").lower()
        kept_offsets: List[int] = []
        for offset in old_offsets:
            old_name = self._read_animation_name(source_mdl_bytes, offset).lower()
            if replace_existing and old_name == slot_name:
                continue
            kept_offsets.append(offset)

        out = BytesIO(bytearray(source_mdl_bytes))
        out.seek(0, 2)
        _align4(out)
        anim_table_rel = out.tell() - _BASE
        new_count = len(kept_offsets) + 1
        for offset in kept_offsets:
            out.write(_wu32(offset))
        new_anim_offset_patch = out.tell()
        out.write(_wu32(0))

        new_anim_rel = out.tell() - _BASE
        end = out.tell()
        out.seek(new_anim_offset_patch)
        out.write(_wu32(new_anim_rel))
        out.seek(end)
        self._write_animation(out, animation)

        mdl_bytes = bytearray(out.getvalue())
        struct.pack_into("<I", mdl_bytes, 4, len(mdl_bytes) - _BASE)
        if source_mdx_bytes:
            struct.pack_into("<I", mdl_bytes, 8, len(source_mdx_bytes))
        struct.pack_into("<I", mdl_bytes, _MODEL_ANIM_ARRAY_OFF_ABS, anim_table_rel)
        struct.pack_into("<I", mdl_bytes, _MODEL_ANIM_COUNT_ABS, new_count)
        struct.pack_into("<I", mdl_bytes, _MODEL_ANIM_COUNT2_ABS, new_count)
        return bytes(mdl_bytes), bytes(source_mdx_bytes)

    def _prepare_animation_only_state(self, model: KotorModel, source_mdl_bytes: bytes) -> None:
        """Initialize the subset of writer state needed for animation blocks."""

        self._model = model
        self._is_k2 = (model.game_version == GameVersion.K2)
        self._animation_override_only = True
        self._target_node_number_by_name_index = self._read_geometry_node_numbers(
            source_mdl_bytes
        )
        self._nodes = []

        def _collect(node: ModelNode) -> None:
            self._nodes.append(node)
            for child in getattr(node, "children", []) or []:
                _collect(child)

        if model.root_node:
            _collect(model.root_node)

        self._node_index_by_name = {}
        for idx, node in enumerate(self._nodes):
            key = str(getattr(node, "name", "") or "").lower()
            if key and key not in self._node_index_by_name:
                self._node_index_by_name[key] = idx

        names = self._read_name_table(source_mdl_bytes)
        if not names:
            names = []
            seen = set()
            for node in self._nodes:
                name = str(getattr(node, "name", "") or "unnamed")
                if name not in seen:
                    seen.add(name)
                    names.append(name)

        self._names = []
        self._name_idx = {}
        self._name_idx_ci = {}
        for name in names:
            self._register_name(name)

    @staticmethod
    def _read_animation_offsets(source_mdl_bytes: bytes) -> List[int]:
        count = struct.unpack_from("<I", source_mdl_bytes, _MODEL_ANIM_COUNT_ABS)[0]
        table_rel = struct.unpack_from("<I", source_mdl_bytes, _MODEL_ANIM_ARRAY_OFF_ABS)[0]
        if count <= 0 or table_rel <= 0:
            return []
        table_abs = _BASE + table_rel
        offsets: List[int] = []
        for index in range(count):
            entry_abs = table_abs + index * 4
            if entry_abs + 4 > len(source_mdl_bytes):
                break
            offset = struct.unpack_from("<I", source_mdl_bytes, entry_abs)[0]
            if 0 < _BASE + offset < len(source_mdl_bytes):
                offsets.append(offset)
        return offsets

    @staticmethod
    def _read_animation_name(source_mdl_bytes: bytes, animation_offset: int) -> str:
        start = _BASE + int(animation_offset) + 8
        end = start + 32
        if start < 0 or end > len(source_mdl_bytes):
            return ""
        return source_mdl_bytes[start:end].split(b"\x00", 1)[0].decode("ascii", errors="replace")

    @staticmethod
    def _read_name_table(source_mdl_bytes: bytes) -> List[str]:
        table_rel = struct.unpack_from("<I", source_mdl_bytes, _MODEL_NAME_OFFSETS_OFF_ABS)[0]
        count = struct.unpack_from("<I", source_mdl_bytes, _MODEL_NAME_COUNT_ABS)[0]
        table_abs = _BASE + table_rel
        names: List[str] = []
        if count <= 0 or table_abs <= 0 or table_abs + count * 4 > len(source_mdl_bytes):
            return names
        for index in range(count):
            name_rel = struct.unpack_from("<I", source_mdl_bytes, table_abs + index * 4)[0]
            name_abs = _BASE + name_rel
            if name_abs < 0 or name_abs >= len(source_mdl_bytes):
                continue
            end = source_mdl_bytes.find(b"\x00", name_abs)
            if end < 0:
                continue
            names.append(source_mdl_bytes[name_abs:end].decode("ascii", errors="replace"))
        return names

    @staticmethod
    def _read_geometry_node_numbers(source_mdl_bytes: bytes) -> Dict[int, int]:
        """Return stock geometry name-index -> runtime node-number identities."""

        if len(source_mdl_bytes) < _BASE + 80:
            return {}
        root_rel = struct.unpack_from("<I", source_mdl_bytes, _BASE + 40)[0]
        if root_rel <= 0:
            return {}

        result: Dict[int, int] = {}
        queue = [int(root_rel)]
        seen: set[int] = set()
        while queue:
            node_rel = queue.pop(0)
            if node_rel in seen:
                continue
            seen.add(node_rel)
            node_abs = _BASE + node_rel
            if node_abs < _BASE or node_abs + 80 > len(source_mdl_bytes):
                continue

            node_number = struct.unpack_from("<H", source_mdl_bytes, node_abs + 2)[0]
            name_index = struct.unpack_from("<H", source_mdl_bytes, node_abs + 4)[0]
            result[name_index] = node_number

            child_array_rel = struct.unpack_from("<I", source_mdl_bytes, node_abs + 44)[0]
            child_count = struct.unpack_from("<I", source_mdl_bytes, node_abs + 48)[0]
            child_array_abs = _BASE + child_array_rel
            if (
                child_array_rel <= 0
                or child_count <= 0
                or child_count > 0x10000
                or child_array_abs + child_count * 4 > len(source_mdl_bytes)
            ):
                continue
            for index in range(child_count):
                child_rel = struct.unpack_from(
                    "<I",
                    source_mdl_bytes,
                    child_array_abs + index * 4,
                )[0]
                if child_rel > 0:
                    queue.append(int(child_rel))
        return result

    def write(self, model: KotorModel) -> Tuple[bytes, bytes]:
        """
        Returns (mdl_bytes, mdx_bytes).

        Algorithm (mirrors Kotor.NET MDLBinaryWriter):
          1. Flatten node tree → ordered name list
          2. Write MDX vertex buffer (all mesh nodes concatenated)
          3. Write MDL:
             a. 12-byte file header  (sizes patched at end)
             b. Geometry header (80 B)
             c. Model header    (88 B)
             d. Name block      (48 B + offset table + strings)
             e. Animation offset array + animation blocks
             f. For each node (DFS): node header, child ptr array,
                ctrl array, ctrl data, type-specific data
          4. Patch mdl_size / mdx_size in file header
        """
        self._model = model
        self._is_k2 = (model.game_version == GameVersion.K2)
        self._animation_override_only = False
        self._child_arr_data_locs: Dict[int, int] = {}  # id(node) → abs pos of child ptr array data

        # ── 1. Collect nodes (DFS pre-order) ────────────────────────────────
        self._nodes: List[ModelNode] = []
        self._node_off: Dict[int, int] = {}   # id(node) → MDL offset
        self._node_info: Dict[int, _NodeInfo] = {}  # id(node) → _NodeInfo

        def _collect(n: ModelNode):
            self._nodes.append(n)
            for c in n.children:
                _collect(c)

        if model.root_node:
            _collect(model.root_node)

        self._node_index_by_name: Dict[str, int] = {}
        self._node_index_by_id: Dict[int, int] = {}
        self._preserve_native_supernode_numbers = bool(
            getattr(model, "preserve_native_supernode_numbers", False)
        )
        self._supernode_number_by_name: Dict[str, int] = {}
        self._supernode_number_by_id: Dict[int, int] = {}
        used_supernode_numbers: Dict[int, str] = {}
        #: Engine contract (vanilla r00_test: 21 names for 21 nodes, dupes
        #: kept): the name table has ONE ENTRY PER NODE, never deduplicated.
        #: Deduplating left name indices pointing past the table -> the
        #: engine's name lookup read garbage (the 0x4b3a8 strlen crash).
        self._name_idx_by_node: Dict[int, int] = {}
        for idx, nd in enumerate(self._nodes):
            self._node_index_by_id[id(nd)] = idx
            nm = (nd.name or '').lower()
            if nm and nm not in self._node_index_by_name:
                self._node_index_by_name[nm] = idx
            supernode_number = (
                int(getattr(nd, "number", idx))
                if self._preserve_native_supernode_numbers
                else idx
            )
            if not 0 <= supernode_number <= 0xFFFF:
                raise ValueError(
                    f"Geometry node '{getattr(nd, 'name', '')}' has invalid "
                    f"supernode number {supernode_number}."
                )
            if self._preserve_native_supernode_numbers:
                previous = used_supernode_numbers.get(supernode_number)
                if previous is not None:
                    raise ValueError(
                        "Preserved supernode number collision: "
                        f"{previous!r} and {getattr(nd, 'name', '')!r} both use "
                        f"{supernode_number}."
                    )
                used_supernode_numbers[supernode_number] = str(
                    getattr(nd, "name", "") or ""
                )
            self._supernode_number_by_id[id(nd)] = supernode_number
            if nm and nm not in self._supernode_number_by_name:
                self._supernode_number_by_name[nm] = supernode_number

        # This field is not always the number of local geometry records.
        # Supermodel-derived character resources store the cumulative node
        # span of the complete inheritance chain. For example, stock K2
        # PFHA04 contains 38 local nodes but declares 564 after S_Female03.
        # A modular neck_g link makes retail consume that inherited span, so
        # collapsing it to len(self._nodes) can terminate the game at load.
        requested_geometry_node_count = int(
            getattr(model, "geometry_node_count", 0) or 0
        )
        if (
            requested_geometry_node_count > 0
            and requested_geometry_node_count < len(self._nodes)
        ):
            raise ValueError(
                "Geometry node span cannot be smaller than the local node "
                f"tree: {requested_geometry_node_count} < {len(self._nodes)}."
            )
        self._geometry_node_count = (
            requested_geometry_node_count
            if requested_geometry_node_count > 0
            else len(self._nodes)
        )

        # ── 2. Build name list (deduplicated, preserving DFS order) ─────────
        self._names: List[str] = []
        self._name_idx: Dict[str, int] = {}
        self._name_idx_ci: Dict[str, int] = {}
        for idx, nd in enumerate(self._nodes):
            # One name entry per node (duplicates preserved) - see contract
            # note above.
            self._name_idx_by_node[id(nd)] = len(self._names)
            nm = str(nd.name or 'unnamed')
            self._name_idx.setdefault(nm, len(self._names))
            self._name_idx_ci.setdefault(nm.lower(), len(self._names))
            self._names.append(nm)
        # Also add animation node names (deduplicated against node names)
        for anim in model.animations:
            for an in anim.nodes:
                self._register_name(an.name)

        # ── 3. Build MDX buffer ──────────────────────────────────────────────
        self._mdx_buf = BytesIO()
        self._node_mdx: Dict[int, Tuple[int, int]] = {}  # id(node) → (offset, stride)
        for nd in self._nodes:
            # Use effective on-disk flags: AABB walk nodes are mesh-bearing
            # even when the in-memory flags lack the MESH bit.
            if _node_flags_for_write(int(nd.flags or 0)) & NodeFlags.MESH and nd.vertices:
                self._write_mdx_node(nd)

        mdx_bytes = self._mdx_buf.getvalue()

        # ── 4. Write MDL ─────────────────────────────────────────────────────
        buf = BytesIO()

        # 4a. File header placeholder (patched later)
        buf.write(b'\x00' * 12)

        # 4b. Geometry header (80 bytes)
        fp1 = _K2_GEOM_FP1 if self._is_k2 else _K1_GEOM_FP1
        fp2 = _K2_GEOM_FP2 if self._is_k2 else _K1_GEOM_FP2
        buf.write(_wu32(fp1))
        buf.write(_wu32(fp2))
        buf.write(_wstr(model.name or 'unnamed', 32))
        self._root_off_patch = buf.tell()  # patch root_off later
        buf.write(_wu32(0))                # root_node_off (patched)
        buf.write(_wu32(self._geometry_node_count)) # inherited geometry span
        # MaxTree subtype byte lives at geometry header +0x4C.  The K1 engine's
        # AsModel() masks this byte with 0x7F and requires value 2 before
        # InputBinary::Read writes model fields through the returned pointer.
        buf.write(b'\x00' * (76 - (buf.tell() - _BASE)))
        buf.write(b'\x02')                 # geometry_type = 2 (model)
        buf.write(b'\x00' * (80 - (buf.tell() - _BASE)))

        # 4c. Model fields (116 bytes at BASE+80).
        # PyKotor's _ModelHeader.SIZE is 0xC4, which includes the 80-byte
        # geometry header plus these 116 bytes. K1 InputBinary::Reset reads the
        # late fields directly, so a compact 88-byte model header leaves
        # offset_to_super_root / mdx_size / name offsets in the wrong slots.
        assert buf.tell() == _BASE + 80, f"Model fields at wrong offset: {buf.tell()}"
        buf.write(struct.pack('B', model.model_type & 0xFF))
        buf.write(struct.pack('B', getattr(model, 'subclassification', 0) & 0xFF))
        buf.write(struct.pack('B', getattr(model, 'padding0', getattr(model, 'unknown_byte', 0)) & 0xFF))
        buf.write(struct.pack('B', 1 if model.disable_fog else 0))
        buf.write(_wu32(0))               # child_model_count
        self._anim_arr_off_patch = buf.tell()   # anim_array_off (patched)
        buf.write(_wu32(0))
        buf.write(_wu32(len(model.animations)))  # anim_count
        buf.write(_wu32(len(model.animations)))  # anim_count2
        buf.write(_wu32(0))               # parent_model_pointer
        # Bounding box
        bb_min = model.bb_min or (0.0, 0.0, 0.0)
        bb_max = model.bb_max or (0.0, 0.0, 0.0)
        buf.write(struct.pack('<fff', *bb_min))
        buf.write(struct.pack('<fff', *bb_max))
        buf.write(_wf32(getattr(model, 'radius', 0.0)))
        buf.write(_wf32(model.anim_scale or 1.0))
        buf.write(_wstr(model.supermodel or 'NULL', 32))
        self._super_root_off_patch = buf.tell()
        buf.write(_wu32(0))               # offset_to_super_root (patched)
        buf.write(_wu32(0))               # mdx_data_buffer_offset
        buf.write(_wu32(len(mdx_bytes)))  # mdx_size
        buf.write(_wu32(0))               # mdx_offset
        self._names_arr_off_patch = buf.tell()   # name_offsets_off (patched)
        buf.write(_wu32(0))
        buf.write(_wu32(len(self._names)))
        buf.write(_wu32(len(self._names)))
        assert buf.tell() == _BASE + 196, f"After _ModelHeader: {buf.tell()}"

        # Write name offset table + strings
        names_table_off = buf.tell() - _BASE  # relative to BASE
        # First pass: write placeholder offsets
        names_table_start = buf.tell()
        for _ in self._names:
            buf.write(_wu32(0))  # patched below
        # Write strings
        name_str_offsets: List[int] = []
        for nm in self._names:
            name_str_offsets.append(buf.tell() - _BASE)
            buf.write(nm.encode('ascii', errors='replace') + b'\x00')
        _align4(buf)
        # Patch names_off and offset table
        end = buf.tell()
        buf.seek(self._names_arr_off_patch)
        buf.write(_wu32(names_table_off))
        buf.seek(names_table_start)
        for off in name_str_offsets:
            buf.write(_wu32(off))
        buf.seek(end)

        # ── 4e. Animation offset array + blocks ─────────────────────────────
        #
        # Vanilla K2 and MDLOps place the animation table/block region before
        # the static geometry node tree.  The game loader mutates child arrays
        # into runtime pointers while walking the binary, and Drexl proved that
        # writing static nodes before the animation block can crash KOTOR2 even
        # when the logical hierarchy reloads in tools.
        if model.animations:
            anim_arr_rel = buf.tell() - _BASE
            end = buf.tell()
            buf.seek(self._anim_arr_off_patch)
            buf.write(_wu32(anim_arr_rel))
            buf.seek(end)

            anim_off_table_start = buf.tell()
            for _ in model.animations:
                buf.write(_wu32(0))

            anim_offsets: List[int] = []
            for anim in model.animations:
                anim_offsets.append(buf.tell() - _BASE)
                self._write_animation(buf, anim)

            end = buf.tell()
            buf.seek(anim_off_table_start)
            for off in anim_offsets:
                buf.write(_wu32(off))
            buf.seek(end)

        # ── 4f. Write node tree ──────────────────────────────────────────────
        root_off = self._write_node_tree(buf, model.root_node)
        super_root_off = root_off
        super_root_name = str(
            getattr(model, "super_root_node_name", "") or ""
        ).strip()
        if super_root_name:
            matching_super_roots = [
                node
                for node in self._nodes
                if str(getattr(node, "name", "") or "").casefold()
                == super_root_name.casefold()
            ]
            if len(matching_super_roots) != 1:
                raise ValueError(
                    "Model super-root link must resolve to exactly one geometry "
                    f"node; {super_root_name!r} matched "
                    f"{len(matching_super_roots)} nodes."
                )
            super_root_off = self._node_off.get(
                id(matching_super_roots[0]), 0
            )
            if super_root_off <= 0:
                raise ValueError(
                    f"Model super-root link {super_root_name!r} was not written."
                )
        # Patch root_off in geometry header
        end = buf.tell()
        buf.seek(self._root_off_patch)
        buf.write(_wu32(root_off))
        buf.seek(self._super_root_off_patch)
        buf.write(_wu32(super_root_off))
        buf.seek(end)

        if not model.animations:
            end = buf.tell()
            buf.seek(self._anim_arr_off_patch)
            buf.write(_wu32(root_off))
            buf.seek(end)

        # ── 4h. Patch file header ────────────────────────────────────────────
        mdl_buf = buf.getvalue()
        mdl_size = len(mdl_buf) - _BASE
        mdx_size = len(mdx_bytes)
        mdl_buf = (b'\x00' * 4
                   + struct.pack('<I', mdl_size)
                   + struct.pack('<I', mdx_size)
                   + mdl_buf[12:])

        log.debug(f"MDLBinaryWriter: {model.name!r} → MDL {len(mdl_buf)} B, "
                  f"MDX {mdx_size} B, {len(self._nodes)} nodes, "
                  f"declared span {self._geometry_node_count}, "
                  f"{len(model.animations)} anims")

        return mdl_buf, mdx_bytes

    def build(self, model: KotorModel) -> Tuple[bytes, bytes]:
        """Alias for write() — returns (mdl_bytes, mdx_bytes).

        Many tests call writer.build(model); this is identical to write(model).
        """
        return self.write(model)

    def _register_name(self, name: str) -> int:
        """Register an MDL name table entry without changing its original case."""

        nm = str(name or 'unnamed')
        if nm not in self._name_idx:
            self._name_idx[nm] = len(self._names)
            self._name_idx_ci.setdefault(nm.lower(), self._name_idx[nm])
            self._names.append(nm)
        return self._name_idx[nm]

    def _name_index_for(self, name: str) -> int:
        """Return the name-table index, preferring byte/case-exact node names."""

        nm = str(name or 'unnamed')
        if nm in self._name_idx:
            return self._name_idx[nm]
        return self._name_idx_ci.get(nm.lower(), 0)

    # ─────────────────────────────  MDX  ────────────────────────────────────

    def _mdx_stride_for(self, node: ModelNode) -> Tuple[int, Dict[str, int]]:
        """
        Compute MDX vertex stride and channel offsets for a mesh node.

        Channel layout (matches KotorBlender types.py MDX_FLAG_* values):
          offset 0        : XYZ       3×float32  12 bytes  (always)
          offset 12       : Normal    3×float32  12 bytes  (if normals)
          offset 24/12    : UV0       2×float32   8 bytes  (if uvs)
          ...
          Skin weights come after geometry channels.

        Returns (stride_bytes, channel_offsets_dict).
        channel_offsets_dict keys: 'v', 'n', 'uv', 'lm', 'tan', 'sw', 'br'
        Values are byte offsets within one vertex stride (0xFFFFFFFF = absent).
        """
        ABSENT = 0xFFFFFFFF
        offsets: Dict[str, int] = {
            'v':  0,        # XYZ always at 0
            'n':  ABSENT,
            'uv': ABSENT,
            'lm': ABSENT,
            'tan': ABSENT,  # 3×vec3: bitangent, tangent, tangent-space normal
            'sw': ABSENT,   # skin weights (4×float)
            'br': ABSENT,   # bone refs (4×float)
        }
        off = 12  # after XYZ

        if node.normals:
            offsets['n'] = off
            off += 12

        if node.uvs:
            offsets['uv'] = off
            off += 8

        if node.uvs_lm:
            offsets['lm'] = off
            off += 8

        if self._mesh_uses_tangent_space(node):
            offsets['tan'] = off
            off += 36  # bitangent(3) + tangent(3) + tangent-space normal(3)

        if node.flags & NodeFlags.SKIN and node.skin_data:
            offsets['sw'] = off
            off += 16   # 4×float weights
            offsets['br'] = off
            off += 16   # 4×float bone refs

        # Align stride to 4-byte boundary (KotOR always does this)
        stride = (off + 3) & ~3
        return stride, offsets

    @staticmethod
    def _mesh_uses_tangent_space(node: ModelNode) -> bool:
        """Return whether this PC MDX row must carry KOTOR tangent space.

        Binary-loaded models may carry ``mdx_tangent_space`` so a round trip
        preserves the source vertex format exactly.  Newly generated rendered
        skins default to tangent space: K2's stock Rancor uses the 0x80 channel
        on every rendered skin (and its texture TXI selects a bump map), while
        omitting the 36-byte block leaves the material and vertex contracts out
        of sync.  Non-rendered probes and explicitly non-tangent source models
        keep their original compact rows.
        """

        vertex_count = len(getattr(node, 'vertices', ()) or ())
        if vertex_count <= 0:
            return False
        if len(getattr(node, 'normals', ()) or ()) < vertex_count:
            return False

        source_value = getattr(node, 'mdx_tangent_space', None)
        if source_value is not None:
            return bool(source_value)

        if len(getattr(node, 'uvs', ()) or ()) < vertex_count:
            return False
        return bool(
            getattr(node, 'render', False)
            and (int(getattr(node, 'flags', 0) or 0) & int(NodeFlags.SKIN))
        )

    @staticmethod
    def _normalise_vec3(values, fallback=(1.0, 0.0, 0.0)) -> Tuple[float, float, float]:
        try:
            x, y, z = (_sanitize_float(values[0]), _sanitize_float(values[1]), _sanitize_float(values[2]))
        except (IndexError, TypeError):
            return fallback
        length = math.sqrt(x * x + y * y + z * z)
        if length <= 1.0e-12:
            return fallback
        return (x / length, y / length, z / length)

    @staticmethod
    def _cross_vec3(a, b) -> Tuple[float, float, float]:
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    def _mdx_tangent_basis_for(
        self,
        node: ModelNode,
    ) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]:
        """Build KOTOR PC tangent rows in on-disk B, T, N order.

        Odyssey expects a left-handed tangent basis: ``dot(cross(N,T),B)``
        must be negative (the MDLOps exporter enforces the same contract).
        Face-space tangent and bitangent vectors are accumulated per vertex,
        normalized, and repaired after accumulation so degenerate or mirrored
        UVs cannot produce a non-finite MDX row.
        """

        vertices = list(getattr(node, 'vertices', ()) or ())
        normals = list(getattr(node, 'normals', ()) or ())
        uvs = list(getattr(node, 'uvs', ()) or ())
        vertex_count = len(vertices)
        tangent_sums = [[0.0, 0.0, 0.0] for _ in range(vertex_count)]
        bitangent_sums = [[0.0, 0.0, 0.0] for _ in range(vertex_count)]

        for face in list(getattr(node, 'faces', ()) or ()):
            try:
                i0, i1, i2 = (int(face[0]), int(face[1]), int(face[2]))
                if min(i0, i1, i2) < 0 or max(i0, i1, i2) >= vertex_count:
                    continue
                p0, p1, p2 = vertices[i0], vertices[i1], vertices[i2]
                uv0, uv1, uv2 = uvs[i0], uvs[i1], uvs[i2]
                e1 = tuple(_sanitize_float(p1[j]) - _sanitize_float(p0[j]) for j in range(3))
                e2 = tuple(_sanitize_float(p2[j]) - _sanitize_float(p0[j]) for j in range(3))
                du1 = _sanitize_float(uv1[0]) - _sanitize_float(uv0[0])
                dv1 = _sanitize_float(uv1[1]) - _sanitize_float(uv0[1])
                du2 = _sanitize_float(uv2[0]) - _sanitize_float(uv0[0])
                dv2 = _sanitize_float(uv2[1]) - _sanitize_float(uv0[1])
            except (IndexError, TypeError, ValueError):
                continue

            determinant = du1 * dv2 - dv1 * du2
            face_normal_raw = self._cross_vec3(e1, e2)
            face_normal = self._normalise_vec3(face_normal_raw, (0.0, 0.0, 1.0))
            if abs(determinant) <= 1.0e-12:
                tangent = (1.0, 0.0, 0.0)
                bitangent = self._normalise_vec3(
                    self._cross_vec3(tangent, face_normal),
                    (0.0, 1.0, 0.0),
                )
            else:
                inv_det = 1.0 / determinant
                tangent = self._normalise_vec3(tuple(
                    (e1[j] * dv2 - e2[j] * dv1) * inv_det
                    for j in range(3)
                ))
                bitangent = self._normalise_vec3(tuple(
                    (e2[j] * du1 - e1[j] * du2) * inv_det
                    for j in range(3)
                ))

                # KOTOR/MDLOps tangent space is left-handed.
                cross_nt = self._cross_vec3(face_normal, tangent)
                if sum(cross_nt[j] * bitangent[j] for j in range(3)) > 0.0:
                    tangent = tuple(-value for value in tangent)

                # MDLOps also flips both axes for mirrored UV triangles.
                if determinant < 0.0:
                    tangent = tuple(-value for value in tangent)
                    bitangent = tuple(-value for value in bitangent)

            face_weight = math.sqrt(sum(value * value for value in face_normal_raw)) or 1.0
            for vertex_index in (i0, i1, i2):
                for component in range(3):
                    tangent_sums[vertex_index][component] += tangent[component] * face_weight
                    bitangent_sums[vertex_index][component] += bitangent[component] * face_weight

        result = []
        for index in range(vertex_count):
            normal = self._normalise_vec3(normals[index], (0.0, 0.0, 1.0))
            tangent = self._normalise_vec3(tangent_sums[index])
            bitangent = self._normalise_vec3(bitangent_sums[index])

            # Repair unused/degenerate vertices with an axis orthogonal to N.
            if tangent_sums[index] == [0.0, 0.0, 0.0]:
                axis = (1.0, 0.0, 0.0) if abs(normal[0]) < 0.9 else (0.0, 1.0, 0.0)
                projection = sum(axis[j] * normal[j] for j in range(3))
                tangent = self._normalise_vec3(tuple(
                    axis[j] - projection * normal[j] for j in range(3)
                ))
            if bitangent_sums[index] == [0.0, 0.0, 0.0]:
                bitangent = self._normalise_vec3(
                    self._cross_vec3(tangent, normal),
                    (0.0, 1.0, 0.0),
                )

            cross_nt = self._cross_vec3(normal, tangent)
            if sum(cross_nt[j] * bitangent[j] for j in range(3)) >= 0.0:
                tangent = tuple(-value for value in tangent)
            result.append((bitangent, tangent, normal))
        return result

    def _write_mdx_node(self, node: ModelNode) -> None:
        """Write one mesh node's vertex data into the MDX buffer."""
        stride, offsets = self._mdx_stride_for(node)
        mdx_start = self._mdx_buf.tell()
        ABSENT = 0xFFFFFFFF
        tangent_basis = (
            self._mdx_tangent_basis_for(node)
            if offsets['tan'] != ABSENT
            else []
        )

        vert_cnt = len(node.vertices)
        for i in range(vert_cnt):
            row = bytearray(stride)

            # XYZ
            vx, vy, vz = node.vertices[i]
            struct.pack_into('<fff', row, offsets['v'],
                             _sanitize_float(vx), _sanitize_float(vy), _sanitize_float(vz))

            # Normal
            if offsets['n'] != ABSENT and i < len(node.normals):
                nx, ny, nz = node.normals[i]
                struct.pack_into('<fff', row, offsets['n'],
                                 _sanitize_float(nx), _sanitize_float(ny), _sanitize_float(nz))

            # UV0
            if offsets['uv'] != ABSENT and i < len(node.uvs):
                u, v = node.uvs[i]
                u, v = self._uv_pair_for_mdx(node, u, v)
                struct.pack_into('<ff', row, offsets['uv'], _sanitize_float(u), _sanitize_float(v))

            # Lightmap UV
            if offsets['lm'] != ABSENT and i < len(node.uvs_lm):
                u, v = node.uvs_lm[i]
                u, v = self._uv_pair_for_mdx(node, u, v)
                struct.pack_into('<ff', row, offsets['lm'], _sanitize_float(u), _sanitize_float(v))

            # KOTOR PC tangent space is 9 floats in B, T, N order.
            if offsets['tan'] != ABSENT and i < len(tangent_basis):
                bitangent, tangent, tangent_normal = tangent_basis[i]
                struct.pack_into(
                    '<fffffffff',
                    row,
                    offsets['tan'],
                    *bitangent,
                    *tangent,
                    *tangent_normal,
                )

            # Skin weights + bone refs
            if (offsets['sw'] != ABSENT and offsets['br'] != ABSENT
                    and i < len(node.skin_data)):
                sd = node.skin_data[i]
                wts = [0.0, 0.0, 0.0, 0.0]
                brs = [-1.0, -1.0, -1.0, -1.0]
                for k, bw in enumerate(sd.influences[:4]):
                    wts[k] = float(bw.weight)
                    brs[k] = float(bw.bone_index)
                struct.pack_into('<ffff', row, offsets['sw'], *wts)
                struct.pack_into('<ffff', row, offsets['br'], *brs)

            self._mdx_buf.write(bytes(row))

        # Trailing pad vertex (num_verts + 1): the engine reads one row past
        # the declared vertex count. KotorBlender's proven writer emits it as
        # a sentinel — position (1e7,1e7,1e7), every other channel zero, skin
        # weight[0]=1. Omitting it makes the engine read past our buffer on the
        # last vertex (out-of-bounds read on module geometry load).
        pad = bytearray(stride)
        struct.pack_into('<fff', pad, offsets['v'], 1.0e7, 1.0e7, 1.0e7)
        if offsets['sw'] != ABSENT:
            struct.pack_into('<ffff', pad, offsets['sw'], 1.0, 0.0, 0.0, 0.0)
        self._mdx_buf.write(bytes(pad))

        self._node_mdx[id(node)] = (mdx_start, stride)

    @staticmethod
    def _uv_pair_for_mdx(node: ModelNode, u: float, v: float) -> Tuple[float, float]:
        """Return the UV pair in KOTOR MDX storage orientation.

        Head Builder carries an explicit ``_gr_mdx_uv_transform`` so its
        serialized choice is independent from the editor renderer's preview
        transform.  Older import paths retain the historical ``uv_v_flip``
        fallback unchanged.

        DCC/renderer-imported nodes can carry top-left style UV rows marked with
        ``uv_v_flip=False`` because the preview renderer flips V at upload time.
        The game consumes the MDX bytes directly, so the writer must perform
        that conversion here.  Native KOTOR reloads and ASCII MDL imports are
        already in game orientation and preserve their rows unchanged.
        """

        out_u = float(u)
        out_v = float(v)
        explicit = str(
            getattr(node, "_gr_mdx_uv_transform", "") or ""
        ).strip().lower()
        if explicit not in {"", "identity", "flip_v"}:
            raise ValueError(
                "Unsupported explicit MDX UV transform: "
                f"{explicit!r}"
            )
        if explicit == "flip_v" or (
            not explicit and getattr(node, "uv_v_flip", True) is False
        ):
            out_v = 1.0 - out_v
        return out_u, out_v

    # ─────────────────────────────  Node tree  ──────────────────────────────

    def _write_node_tree(self, buf: BytesIO,
                         root: Optional[ModelNode]) -> int:
        """
        Write all geometry nodes DFS.
        Returns offset of root node (relative to BASE), or 0 if no root.

        Two-pass strategy:
          Pass 1: Write every node header + type-specific data + ctrl arrays.
                  Record each node's absolute offset.
                  Leave child pointer arrays as placeholder zeros.
          Pass 2: Patch child pointer arrays now that all node offsets are known.
                  Also patch root_off and parent_off fields.
        """
        if root is None:
            return 0

        # Collect DFS order
        ordered: List[ModelNode] = []
        def _dfs(n: ModelNode):
            ordered.append(n)
            for c in n.children:
                _dfs(c)
        _dfs(root)

        # Pass 1: write all node headers + type-specific data, record offsets
        node_abs_off: Dict[int, int] = {}           # id(node) → absolute offset
        child_arr_patches: Dict[int, int] = {}       # id(node) → abs position of child_arr_off field
        root_off_patches: Dict[int, int] = {}        # id(node) → abs position of root_off field
        parent_off_patches: Dict[int, int] = {}      # id(node) → abs position of parent_off field

        for nd in ordered:
            _align4(buf)
            node_abs_off[id(nd)] = buf.tell()
            self._write_one_node_pass1(
                buf, nd,
                child_arr_patches, root_off_patches, parent_off_patches)

        # Retain resource-relative node offsets for model-header links that are
        # intentionally not the geometry root. Stock modular heads use this to
        # point ``offset_to_super_root`` at ``neck_g`` (the community
        # HeadFixer contract) while the geometry header continues to point at
        # the real AuroraBase root.
        self._node_off = {
            node_id: absolute_offset - _BASE
            for node_id, absolute_offset in node_abs_off.items()
        }

        # Pass 2: patch root_off, parent_off, and child pointer arrays
        root_abs = node_abs_off[id(ordered[0])]
        cur_end = buf.tell()

        for nd in ordered:
            nd_abs = node_abs_off[id(nd)]
            parent_abs = node_abs_off.get(id(nd.parent), 0)

            # Node-header +8 ("off_root") stays 0. Every vanilla model (K1/K2
            # rooms, placeables, creatures) writes 0 here -- it is a runtime
            # pointer the engine fills during load. Writing root_abs-BASE made
            # the engine treat *(node+8) as the geometry object, fail its
            # type-2/5 cast (a node is neither), and NULL-deref at
            # swkotor2+0x44b3a8 during the "<roomname>a" node-name lookup.
            # (The geometry header's own root_off at +40 still points to root.)
            if id(nd) in root_off_patches:
                buf.seek(root_off_patches[id(nd)])
                buf.write(_wu32(0))

            # Patch parent_off
            if id(nd) in parent_off_patches:
                buf.seek(parent_off_patches[id(nd)])
                buf.write(_wu32(parent_abs - _BASE if parent_abs else 0))

        buf.seek(cur_end)

        # Patch child pointer arrays (written after all nodes, or at end of each node body)
        # We already wrote them inline in pass1; now overwrite with real values.
        # The child ptr arrays were written right after the node body; their locations
        # were stored in child_arr_patches as the position of the child_arr_off field.
        # We need to seek to each child array and write correct child offsets.
        # The arrays themselves were written as placeholder zeros inline.
        # We need the actual arrays' locations — they were written as part of the node body.
        # Retrieve from our per-node child_arr_info stored during pass1.
        for nd in ordered:
            if not nd.children:
                continue
            arr_loc = self._child_arr_data_locs.get(id(nd))
            if arr_loc is None:
                continue
            buf.seek(arr_loc)
            for child in nd.children:
                c_abs = node_abs_off.get(id(child), 0)
                buf.write(_wu32(c_abs - _BASE if c_abs else 0))

        buf.seek(cur_end)
        return root_abs - _BASE

    def _write_one_node_pass1(self, buf: BytesIO, node: ModelNode,
                               child_arr_patches: Dict[int, int],
                               root_off_patches: Dict[int, int],
                               parent_off_patches: Dict[int, int]) -> None:
        """
        Write a single node: header + type-specific data + child ptr array +
        controller arrays.  Forward references (root_off, parent_off, child
        ptr values) are written as 0 and patched in pass 2.
        """
        node_start = buf.tell()

        name_idx = self._name_idx_by_node.get(id(node), self._name_index_for(node.name))
        node_flags = _node_flags_for_write(int(getattr(node, "flags", 0) or 0))

        # ── Node header  (80 bytes) ──────────────────────────────────────────
        # Header +2 is the sparse supermodel-node identity. Native character
        # rigs deliberately use values far beyond their DFS node count; retail
        # compares this field directly when applying animation tracks and when
        # resolving modular-head hooks. Generated generic models retain the
        # historical dense fallback, while native-rig rebuilds opt into exact
        # preservation through ``preserve_native_supernode_numbers``.
        node_num = self._supernode_number_by_id.get(
            id(node), self._node_index_by_id.get(id(node), name_idx)
        )
        buf.write(_wu16(node_flags))
        buf.write(_wu16(node_num))   # sparse supermodel-node identity
        buf.write(_wu16(name_idx))   # name index
        buf.write(_wu16(0))          # pad

        root_off_patches[id(node)] = buf.tell()
        buf.write(_wu32(0))          # root_off (patched in pass 2)
        parent_off_patches[id(node)] = buf.tell()
        buf.write(_wu32(0))          # parent_off (patched in pass 2)

        px, py, pz = node.position if node.position else (0.0, 0.0, 0.0)
        buf.write(struct.pack('<fff', px, py, pz))

        # Rotation: internal tuple is (x, y, z, w); on-disk layout is W-first.
        # Verified against three independent references, all of which read the
        # quaternion in W → X → Y → Z order after the node position field:
        #   • PyKotor  ``_NodeHeader.read`` (io_mdl.py): reads ``orientation.w``,
        #     ``orientation.x``, ``orientation.y``, ``orientation.z`` in turn.
        #   • KotorBlender ``reader.py``   (format/mdl/reader.py)
        #   • KotOR.js     ``OdysseyModelNodeHeader.ts``
        # So the writer must emit W first.  Do NOT swap.
        rx, ry, rz, rw = (node.rotation if node.rotation
                          else (0.0, 0.0, 0.0, 1.0))
        buf.write(struct.pack('<ffff', rw, rx, ry, rz))

        # Child array placeholder
        child_arr_patch = buf.tell()
        child_arr_patches[id(node)] = child_arr_patch
        buf.write(_wu32(0))                           # child_arr_off (patched)
        buf.write(_wu32(len(node.children)))          # child_cnt
        buf.write(_wu32(len(node.children)))          # child_cnt2

        # Controller array placeholder
        ctrl_arr_patch = buf.tell()
        buf.write(_wu32(0))                           # ctrl_arr_off (patched)
        ctrl_count = len(node.controllers)
        buf.write(_wu32(ctrl_count))
        buf.write(_wu32(ctrl_count))

        # Controller data placeholder
        ctrl_data_patch = buf.tell()
        buf.write(_wu32(0))                           # ctrl_data_off (patched)
        ctrl_data_cnt = self._count_ctrl_data(node)
        buf.write(_wu32(ctrl_data_cnt))
        buf.write(_wu32(ctrl_data_cnt))

        assert buf.tell() == node_start + 80, (
            f"Node header wrong size: {buf.tell() - node_start}")

        # ── Type-specific data ───────────────────────────────────────────────
        if node_flags & NodeFlags.EMITTER:
            self._write_emitter_header(buf, node)

        if node_flags & NodeFlags.REFERENCE:
            self._write_reference_header(buf, node)

        if node_flags & NodeFlags.MESH:
            self._write_mesh_header(buf, node)

        # LIGHT sub-header follows the base node header (mesh/skin are mutually
        # exclusive with LIGHT in practice, so for a pure light node this lands
        # at node_start+80 exactly where the reader expects it).
        if node_flags & NodeFlags.LIGHT:
            self._write_light_header(buf, node)

        # ── Child pointer array (data) ───────────────────────────────────────
        # Write placeholder zeros; pass 2 will overwrite with real offsets.
        _align4(buf)
        child_arr_data_off = buf.tell() - _BASE
        if not hasattr(self, '_child_arr_data_locs'):
            self._child_arr_data_locs: Dict[int, int] = {}
        self._child_arr_data_locs[id(node)] = buf.tell()
        for _ in node.children:
            buf.write(_wu32(0))   # placeholder; patched in pass 2

        # Patch child_arr_off in node header
        end = buf.tell()
        buf.seek(child_arr_patch)
        buf.write(_wu32(child_arr_data_off))
        buf.seek(end)

        # ── Controller array + data ──────────────────────────────────────────
        self._write_controllers(buf, node, ctrl_arr_patch, ctrl_data_patch)

    def _count_ctrl_data(self, node: ModelNode) -> int:
        """Count total float entries in controller data pool."""
        total = 0
        for ctrl in node.controllers:
            rows = len(ctrl.get('times', []))
            if self._uses_compressed_orientation_payload(ctrl, rows):
                total += rows
                total += rows
                continue
            bezier_payload = self._bezier_controller_payload(ctrl, rows)
            if bezier_payload is not None:
                _column_flag, values_per_row, _values = bezier_payload
                total += rows
                total += rows * values_per_row
                continue
            cols = self._linear_controller_column_count(ctrl)
            total += rows          # time keys
            total += rows * cols   # value data
        return total

    @staticmethod
    def _linear_controller_column_count(ctrl: Dict) -> int:
        """Return the exact on-disk width for a non-Bezier controller.

        Controller type IDs are node-kind dependent.  In particular, emitter
        controller 100 is a one-float channel even though mesh controller 100
        is a three-float self-illumination colour.  A binary read therefore
        carries the authoritative column byte in ``binary_column_count``;
        replacing it with the generic type map corrupts the following rows.
        """

        raw = int(ctrl.get('binary_column_count', ctrl.get('columns', 1)) or 1)
        # Bit 0x10 belongs to Bezier payloads, handled before this helper.
        return max(1, raw & 0x0F)

    @staticmethod
    def _uses_compressed_orientation_payload(ctrl: Dict, rows: int) -> bool:
        if int(ctrl.get('type', 0) or 0) != CTRL_ORIENTATION:
            return False
        if int(ctrl.get('binary_column_count', ctrl.get('columns', 1)) or 1) != 2:
            return False
        words = ctrl.get('binary_compressed_quaternion_words') or []
        return len(words) >= rows > 0

    @staticmethod
    def _bezier_controller_payload(
        ctrl: Dict,
        rows: int,
    ) -> Optional[Tuple[int, int, List[List[float]]]]:
        """Return the lossless Aurora Bezier layout for one controller.

        The controller-entry column byte uses bit ``0x10`` as the Bezier
        flag.  Its low nibble is the number of logical components; every
        logical component stores three float32 values per row (value,
        incoming tangent, outgoing tangent).  K1 ``m14aa_01f`` therefore
        stores its three-component ``Dummy01`` position controller as
        ``0x13`` with nine floats per key.

        ``values`` remains the decoded logical channel used by viewport
        animation evaluation.  ``binary_bezier_rows`` is the independent raw
        payload retained by the reader/model conversion for exact writing.
        Silently substituting the three decoded values would flatten the path,
        so incomplete tangent metadata is an export error.
        """

        raw_column_count = ctrl.get('binary_column_count')
        column_flag = int(raw_column_count or 0) & 0xFF
        is_bezier = bool(ctrl.get('is_bezier', False)) or bool(column_flag & 0x10)
        if not is_bezier:
            return None

        base_columns = column_flag & 0x0F
        if base_columns <= 0:
            base_columns = int(ctrl.get('columns', 1) or 1)
            column_flag = 0x10 | (base_columns & 0x0F)
        values_per_row = base_columns * 3
        payload_rows = ctrl.get('binary_bezier_rows')
        if not isinstance(payload_rows, (list, tuple)):
            payload_rows = ctrl.get('values') or []
        if len(payload_rows) < rows:
            raise ValueError(
                f"Bezier controller type {ctrl.get('type', 0)!r} has {len(payload_rows)} "
                f"payload row(s) for {rows} time key(s)."
            )

        values: List[List[float]] = []
        for index in range(rows):
            row = list(payload_rows[index] or [])
            if len(row) < values_per_row:
                raise ValueError(
                    f"Bezier controller type {ctrl.get('type', 0)!r} row {index} has "
                    f"{len(row)} value(s); column flag 0x{column_flag:02x} requires "
                    f"{values_per_row}."
                )
            values.append([float(value) for value in row[:values_per_row]])
        return column_flag, values_per_row, values

    def _write_controllers(self, buf: BytesIO, node: ModelNode,
                           arr_patch: int, data_patch: int) -> None:
        """Write controller entry array and controller data pool."""
        if not node.controllers:
            return

        # Build controller entries
        # Each entry = 16 bytes:
        #   type(4) unknown(2) row_count(2) time_off(2) data_off(2) columns(1) pad(1) pad2(2)
        # Source: mdl_parser._parse_controllers

        _align4(buf)
        ctrl_arr_off = buf.tell() - _BASE
        # Controller entry stubs (patched with offsets below)
        entries_start = buf.tell()
        for ctrl in node.controllers:
            buf.write(b'\x00' * 16)

        _align4(buf)
        ctrl_data_off = buf.tell() - _BASE
        data_start = buf.tell()

        # Write time keys + values for all controllers
        ctrl_meta: List[Tuple[int, int]] = []   # (time_pool_off, data_pool_off)
        running = 0

        for ctrl in node.controllers:
            times  = ctrl.get('times', [])
            values = ctrl.get('values', [])
            cols   = self._linear_controller_column_count(ctrl)
            rows   = len(times)
            compressed_orientation = self._uses_compressed_orientation_payload(ctrl, rows)
            bezier_payload = self._bezier_controller_payload(ctrl, rows)

            time_off = running
            for t in times:
                buf.write(_wf32(float(t)))
                running += 1

            val_off = running
            if compressed_orientation:
                words = list(ctrl.get('binary_compressed_quaternion_words') or [])
                for index in range(rows):
                    buf.write(_wu32(int(words[index]) & 0xFFFFFFFF))
                    running += 1
            elif bezier_payload is not None:
                _column_flag, values_per_row, bezier_rows = bezier_payload
                for row in bezier_rows:
                    for value in row:
                        buf.write(_wf32(value))
                    running += values_per_row
            else:
                for row in values:
                    for col in range(cols):
                        v = row[col] if col < len(row) else 0.0
                        buf.write(_wf32(float(v)))
                    running += cols

            ctrl_meta.append((time_off, val_off))

        end = buf.tell()

        # Patch controller entries
        buf.seek(entries_start)
        for (ctrl, (t_off, v_off)) in zip(node.controllers, ctrl_meta):
            ctype = ctrl.get('type', 0)
            rows  = len(ctrl.get('times', []))
            compressed_orientation = self._uses_compressed_orientation_payload(ctrl, rows)
            bezier_payload = self._bezier_controller_payload(ctrl, rows)
            cols  = (
                int(ctrl.get('binary_column_count', 2) or 2)
                if compressed_orientation
                else (
                    int(bezier_payload[0])
                    if bezier_payload is not None
                    else int(
                        ctrl.get(
                            'binary_column_count',
                            self._linear_controller_column_count(ctrl),
                        )
                        or 1
                    )
                )
            )
            # Vanilla controller keys carry 0xFFFF (-1) in this slot; only
            # synthetic controllers left it 0.  Every stock K1/K2 model uses
            # the 0xFFFF sentinel, so default to it when a source value is
            # absent (parsed models keep their own binary_unknown0).
            _unk0 = ctrl.get('binary_unknown0', ctrl.get('unknown0', 0xFFFF))
            unknown0 = 0xFFFF if _unk0 is None else (int(_unk0) & 0xFFFF)
            unknown1 = bytes(
                (int(v) & 0xFF)
                for v in list(ctrl.get('binary_unknown1', []))[:3]
            ).ljust(3, b'\x00')
            buf.write(_wu32(ctype))
            buf.write(_wu16(unknown0))
            buf.write(_wu16(rows))         # row_count
            buf.write(_wu16(t_off & 0xFFFF))
            buf.write(_wu16(v_off & 0xFFFF))
            # Preserve the full byte: bit 0x10 is Aurora's Bezier flag.
            # Masking to the low nibble changed vanilla 0x13 into linear 3.
            buf.write(struct.pack('B', cols & 0xFF))
            buf.write(unknown1)

        buf.seek(entries_start - (entries_start - (buf.tell())))  # reset
        buf.seek(end)

        # Patch ctrl_arr_off and ctrl_data_off in node header
        buf.seek(arr_patch)
        buf.write(_wu32(ctrl_arr_off))
        buf.seek(data_patch)
        buf.write(_wu32(ctrl_data_off))
        buf.seek(end)

    # ─────────────────────────────  Light header  ───────────────────────────

    def _write_light_header(self, buf: BytesIO, node: ModelNode) -> None:
        """Write the Odyssey LIGHT node sub-header (92 bytes).

        Layout mirrors PyKotor ``io_mdl._LightHeader`` (the reader the in-house
        loader delegates to): five ``(offset, count, count2)`` array descriptors
        followed by ``flare_radius`` (f32) and seven uint32 params.  We emit the
        no-flare / no-lens-flare case — all fifteen array descriptor words are
        zero — which covers every baked GUI-scene light (mainmenu, charrec_light
        et al. carry no flare arrays).

        Light COLOUR / RADIUS / MULTIPLIER are NOT in this header; the engine
        stores them as controllers (types 76 / 88 / 140) which ``_write_controllers``
        emits from ``node.controllers``.

        Field defaults match measured vanilla K1 GUI lights (priority 1,
        affect_dynamic 0, flare_radius 0.0).  ``light_priority``,
        ``light_affect_dynamic`` and ``light_flare_radius`` are read via
        ``getattr`` so a future ModelNode/reader that carries them round-trips
        without touching this writer.
        """
        node_start_ok = buf.tell()
        # 5 array descriptors × (offset, count, count2) uint32 — empty (no flares)
        buf.write(b"\x00" * (15 * 4))
        buf.write(_wf32(float(getattr(node, "light_flare_radius", 0.0) or 0.0)))
        buf.write(_wu32(int(getattr(node, "light_priority", 1) or 0) & 0xFFFFFFFF))
        buf.write(_wu32(1 if node.light_ambient_only else 0))
        buf.write(_wu32(int(getattr(node, "light_dynamic", 0) or 0) & 0xFFFFFFFF))
        buf.write(_wu32(1 if getattr(node, "light_affect_dynamic", False) else 0))
        buf.write(_wu32(1 if node.light_shadow else 0))
        buf.write(_wu32(1 if node.light_flare else 0))
        buf.write(_wu32(1 if node.light_fading else 0))
        assert buf.tell() - node_start_ok == 92, (
            f"Light header wrong size: {buf.tell() - node_start_ok} (expected 92)")

    # ─────────────────────────────  Mesh header  ────────────────────────────

    def _write_mesh_header(self, buf: BytesIO, node: ModelNode) -> None:
        """
        Write the mesh header matching the exact parser layout
        (verified against mdl_parser._parse_mesh field-by-field).

        Exact layout:
          +0    funcptr1/2        (8 B)
          +8    faces off/cnt/cnt2 (12 B)
          +20   bb_min            (12 B)
          +32   bb_max            (12 B)
          +44   radius            (4 B)
          +48   avg_position      (12 B)
          +60   diffuse           (12 B)
          +72   ambient           (12 B)
          +84   transparency_hint (4 B)
          +88   tex_name          (32 B)
          +120  lm_name           (32 B)
          +152  unknown0          (24 B)   opaque bytes preserved from read;
                                           NOT two 12-byte bitmap names
          +176  vic array         (12 B)   off/cnt/cnt2 (zeros)
          +188  vo array          (12 B)   off/cnt/cnt2 (zeros)
          +200  inv array         (12 B)   off/cnt/cnt2 (zeros)
          +212  {-1,-1,0}         (12 B)   unknown
          +224  saber vals        (8 B)    unknown
          +232  animate_uv        (4 B)    UV animation flag
          +236  uv_dir_x          (4 B)
          +240  uv_dir_y          (4 B)
          +244  uv_jitter         (4 B)
          +248  uv_jitter_speed   (4 B)
          +252  mdx_data_size     (4 B)
          +256  mdx_data_bitmap   (4 B)
          +260  MDX offsets[11]   (44 B)   11×uint32
          +304  vert_cnt          (2 B)    uint16
          +306  tex_cnt           (2 B)    uint16
          +308  has_lightmap      (1 B)
          +309  rotate_texture    (1 B)
          +310  background_geometry (1 B)
          +311  has_shadow        (1 B)
          +312  beaming           (1 B)
          +313  render            (1 B)
          [K2 only]:
          +314  dirt_enabled      (1 B)
          +315  padding           (1 B)
          +316  dirt_texture      (2 B)
          +318  dirt_coord_space  (2 B)
          +320  hide_in_holograms (1 B)
          +321  padding           (1 B)
          +314/322  2 pad bytes
          +316/324  total_area    (4 B)
          +320/328  unknown       (4 B)
          +324/332  mdx_data_off  (4 B)  ← byte offset into MDX buffer
          +328/336  verts_off     (4 B)  ← MDL vertex fallback array

          Followed by:
            face array (face_cnt × 32 B)
            MDL vertex array (vert_cnt × 12 B)
          Then skin/dangly headers if applicable.
        """
        # Mesh-subheader FPs differ by node flag family (MESH vs SKIN vs DANGLY).
        # Using the TOP-LEVEL geometry fp pair here (the old behaviour) produces
        # files that the KotOR engine refuses to load because the runtime vtable
        # dispatch in ``CMDLMesh`` / ``CMDLSkinMesh`` matches on these FPs.
        fp1, fp2 = _mesh_fp_pair(self._is_k2, int(node.flags))

        vert_cnt  = len(node.vertices)
        face_cnt  = len(node.faces)
        raw_tex_name = str(node.texture or '')
        raw_lm_name = str(node.lightmap or '')
        tex_name = 'NULL' if raw_tex_name.strip().lower() == 'null' else raw_tex_name.lower()
        lm_name = 'NULL' if raw_lm_name.strip().lower() == 'null' else raw_lm_name.lower()
        # NOTE: the 24 bytes at mesh-header offset +152 are NOT two 12-byte
        # bitmap name slots.  PyKotor (and KotorBlender) parse them as an
        # opaque ``unknown0`` block (``reader.read_bytes(24)``).  Writing
        # nul-terminated text there can be mis-interpreted by downstream
        # readers and validators.  We preserve any bytes we captured at load
        # time via ``node.mesh_unknown0``; otherwise we emit 24 zero bytes.
        mesh_unknown0: bytes = bytes(getattr(node, 'mesh_unknown0', b'') or b'')
        if len(mesh_unknown0) < 24:
            mesh_unknown0 = mesh_unknown0.ljust(24, b'\x00')
        else:
            mesh_unknown0 = mesh_unknown0[:24]

        vertices = list(getattr(node, 'vertices', ()) or ())
        bb_min  = getattr(node, 'mesh_bb_min', None) or getattr(node, 'bb_min', None) or (0.0, 0.0, 0.0)
        bb_max  = getattr(node, 'mesh_bb_max', None) or getattr(node, 'bb_max', None) or (0.0, 0.0, 0.0)
        avg_pt  = getattr(node, 'mesh_average_point', None) or (0.0, 0.0, 0.0)
        mesh_radius = getattr(node, 'mesh_radius', None)
        if mesh_radius is None:
            mesh_radius = getattr(node, 'radius', 0.0)
        if vertices and (bb_min == (0.0, 0.0, 0.0) and bb_max == (0.0, 0.0, 0.0)):
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            zs = [v[2] for v in vertices]
            bb_min = (min(xs), min(ys), min(zs))
            bb_max = (max(xs), max(ys), max(zs))
        if vertices and not mesh_radius:
            cx = (bb_min[0] + bb_max[0]) * 0.5
            cy = (bb_min[1] + bb_max[1]) * 0.5
            cz = (bb_min[2] + bb_max[2]) * 0.5
            mesh_radius = max(
                math.sqrt((v[0] - cx) ** 2 + (v[1] - cy) ** 2 + (v[2] - cz) ** 2)
                for v in vertices
            )
        if vertices and avg_pt == (0.0, 0.0, 0.0):
            inv_count = 1.0 / float(len(vertices))
            avg_pt = (
                sum(v[0] for v in vertices) * inv_count,
                sum(v[1] for v in vertices) * inv_count,
                sum(v[2] for v in vertices) * inv_count,
            )
        diffuse = node.diffuse or (1.0, 1.0, 1.0)
        ambient = node.ambient or (0.0, 0.0, 0.0)
        raw_tex_count = getattr(node, 'tex_count', None)
        if raw_tex_count is None:
            tex_cnt = 0 if tex_name in ('', 'NULL') else 1
        else:
            tex_cnt = max(0, int(raw_tex_count))

        # MDX channel layout
        if node.vertices:
            mdx_off, mdx_stride = self._node_mdx.get(id(node), (0, 0))
        else:
            mdx_off, mdx_stride = 0, 0

        _, ch_offsets = self._mdx_stride_for(node)
        ABSENT = 0xFFFFFFFF

        # 11-slot MDX channel offset table (slot order from parser):
        #   [0] xyz  [1] normal  [2] vert_color  [3] uv0  [4] lm_uv
        #   [5] uv2  [6] uv3  [7..10] tangent spaces
        tvert_offs = [
            ch_offsets['v'],        # [0] xyz
            ch_offsets['n'],        # [1] normal
            ABSENT,                 # [2] vertex color
            ch_offsets['uv'],       # [3] UV0
            ch_offsets['lm'],       # [4] LM UV
            ABSENT,                 # [5] UV2
            ABSENT,                 # [6] UV3
            ch_offsets['tan'],       # [7] tangent space 1 (B, T, N)
            ABSENT, ABSENT, ABSENT,  # [8-10] tangent spaces 2-4
        ]

        # MDX bitmap
        mdx_bitmap = 0x0001  # XYZ always present
        if ch_offsets['n'] != ABSENT:
            mdx_bitmap |= 0x0020
        if ch_offsets['uv'] != ABSENT:
            mdx_bitmap |= 0x0002
        if ch_offsets['lm'] != ABSENT:
            mdx_bitmap |= 0x0004
        if ch_offsets['tan'] != ABSENT:
            mdx_bitmap |= 0x0080

        # ── Write header fields in exact parser order ────────────────────────
        mesh_hdr_start = buf.tell()

        # +0  fp1/fp2
        buf.write(_wu32(fp1))
        buf.write(_wu32(fp2))

        # +8  faces descriptor (off / cnt / cnt2)
        faces_off_patch = buf.tell()
        buf.write(_wu32(0))             # faces_off (patched later)
        buf.write(_wu32(face_cnt))
        buf.write(_wu32(face_cnt))

        # +20 bounding box + radius + avg_pos
        buf.write(struct.pack('<fff', *bb_min))
        buf.write(struct.pack('<fff', *bb_max))
        buf.write(_wf32(mesh_radius))
        buf.write(struct.pack('<fff', *avg_pt))

        # +60 diffuse + ambient + transparency_hint
        buf.write(struct.pack('<fff', *diffuse))
        buf.write(struct.pack('<fff', *ambient))
        buf.write(_wu32(getattr(node, 'transparency_hint', 0)))

        # +88  tex_name (32 B) + lm_name (32 B)
        buf.write(_wstr(tex_name, 32))
        buf.write(_wstr(lm_name, 32))

        # +152: 24-byte ``unknown0`` block.  Previously written as two 12-byte
        # ASCII strings ("bm3_name" / "bm4_name") — that layout does not match
        # any known reference reader (PyKotor reads opaque bytes here, so do
        # KotorBlender and Kotor.NET).  Emit the raw bytes captured at load
        # time so round-trip tests produce identical files.
        buf.write(mesh_unknown0)

        # +176 render-batch arrays (3 x 12 B):
        #   indices_counts  : uint32 count of triangle-list indices per batch
        #   indices_offsets : uint32 MDL offset to each batch's uint16 indices
        #   inverted_counters: opaque per-batch uint32 consumed by the engine
        #
        # These are not optional for Odyssey's in-game loader.  A zeroed
        # descriptor is tolerated by some offline tools, but K2's model-load
        # bookkeeping can dereference the missing batch pointer for renderable
        # creature meshes.  We synthesize one batch covering every face unless
        # a loaded model carried compatible batch counts.
        indices_counts_desc_patch = buf.tell()
        buf.write(_wu32(0))
        buf.write(_wu32(0))
        buf.write(_wu32(0))
        indices_offsets_desc_patch = buf.tell()
        buf.write(_wu32(0))
        buf.write(_wu32(0))
        buf.write(_wu32(0))
        counters_desc_patch = buf.tell()
        buf.write(_wu32(0))
        buf.write(_wu32(0))
        buf.write(_wu32(0))
        buf.write(b'\xff\xff\xff\xff' * 2 + b'\x00' * 4)  # {-1,-1,0}
        # Engine contract (stock r00_test.mdl +224, KotorBlender writer):
        # first saber-unknown byte is 3, not 0.
        buf.write(struct.pack('B', 3) + b'\x00' * 7)

        # +232 UV animation fields (20 B); vanilla default uv_dir is (1, 0).
        buf.write(_wu32(1 if getattr(node, 'animate_uv', False) else 0))
        buf.write(_wf32(getattr(node, 'uv_dir_x', 0.0)))  # T2548: no forced 1.0
        buf.write(_wf32(getattr(node, 'uv_dir_y', 0.0)))
        buf.write(_wf32(getattr(node, 'uv_jitter', 0.0)))
        buf.write(_wf32(getattr(node, 'uv_jitter_speed', 0.0)))

        # +252 mdx_data_size + mdx_data_bitmap
        buf.write(_wu32(mdx_stride))
        buf.write(_wu32(mdx_bitmap))

        # +260 11 MDX channel offsets (44 B)
        for off in tvert_offs:
            buf.write(_wu32(off))

        # +304 vert_cnt + tex_cnt
        buf.write(_wu16(vert_cnt))
        buf.write(_wu16(tex_cnt))

        # +308 6 flag bytes
        buf.write(struct.pack('B', 1 if node.has_lightmap else 0))
        buf.write(struct.pack('B', 1 if node.rotate_texture else 0))
        buf.write(struct.pack('B', 1 if getattr(node, 'background_geometry', False) else 0))
        buf.write(struct.pack('B', 1 if node.has_shadow else 0))
        buf.write(struct.pack('B', 1 if node.beaming else 0))
        buf.write(struct.pack('B', 1 if node.render else 0))

        # K2-specific fields (8 B)
        if self._is_k2:
            # Vanilla K2 meshes carry dirt_texture=1, dirt_coord_space=1
            # even when dirt is disabled (stock r00_test +316).
            buf.write(struct.pack('B', 1 if getattr(node, 'dirt_enabled', False) else 0))
            buf.write(b'\x00')
            buf.write(_wu16((getattr(node, 'dirt_texture', 0) or 1)))
            buf.write(_wu16((getattr(node, 'dirt_coord_space', 0) or 1)))
            buf.write(struct.pack('B', 1 if getattr(node, 'hide_in_holograms', False) else 0))
            buf.write(b'\x00')

        # 2 pad + 4 total_area + 4 unknown; vanilla stores the real summed
        # face area (stock r00_test: 1978.1), not zero.
        total_area = float(getattr(node, 'mesh_total_area', 0.0) or 0.0)
        if not total_area and node.vertices and node.faces:
            for fa, fb, fc in node.faces:
                try:
                    ax, ay, az = node.vertices[fa]
                    bx, by, bz = node.vertices[fb]
                    cx, cy, cz = node.vertices[fc]
                except (IndexError, ValueError):
                    continue
                ux, uy, uz = bx - ax, by - ay, bz - az
                vx, vy, vz = cx - ax, cy - ay, cz - az
                nx = (uy * vz) - (uz * vy)
                ny = (uz * vx) - (ux * vz)
                nz = (ux * vy) - (uy * vx)
                total_area += 0.5 * ((nx * nx + ny * ny + nz * nz) ** 0.5)
        buf.write(b'\x00' * 2)
        buf.write(_wf32(total_area))
        buf.write(b'\x00' * 4)

        # mdx_data_off + verts_off
        buf.write(_wu32(mdx_off))           # MDX byte offset

        verts_off_patch = buf.tell()
        buf.write(_wu32(0))                 # MDL vertex array (patched below)

        # ── Skin header (if SKIN) — must be at o = right after fixed mesh header ──
        if node.flags & NodeFlags.SKIN:
            self._write_skin_header(buf, node)

        # ── Dangly header (if DANGLY) — must be at o = right after fixed mesh header ──
        if node.flags & NodeFlags.DANGLY:
            self._write_dangly_header(buf, node)

        # ── AABB: 4-byte ``offset_to_aabb`` field + minimal tree ─────────────
        # PyKotor ``_Node.read`` reads a single int32 immediately after the
        # trimesh/skin/dangly headers when ``type_id & AABB`` is set.  Without
        # this field walkmesh nodes fail to parse (reader gets misaligned and
        # subsequent data / children pointers go to garbage offsets).
        if node.flags & NodeFlags.AABB:
            self._write_aabb_section(buf, node)

        # ── Face array — placed after skin/dangly headers, offset patched above ──
        _align4(buf)
        faces_off_abs = buf.tell()
        flat_face_indices: list[int] = []
        # Engine contract (stock r00_test faces): real plane equations and
        # adjacency (0xFFFF = no neighbor). Zeros here crashed swkotor2.
        edge_map: dict[tuple[int, int], list[int]] = {}
        for i, (a, b, c) in enumerate(node.faces):
            for e0, e1 in ((a, b), (b, c), (c, a)):
                edge_map.setdefault((min(e0, e1), max(e0, e1)), []).append(i)
        for i, (v1, v2, v3) in enumerate(node.faces):
            mat = node.face_mats[i] if i < len(node.face_mats) else 0
            try:
                ax, ay, az = node.vertices[v1]
                bx, by, bz = node.vertices[v2]
                cx, cy, cz = node.vertices[v3]
                ux, uy, uz = bx - ax, by - ay, bz - az
                wx, wy, wz = cx - ax, cy - ay, cz - az
                nx = (uy * wz) - (uz * wy)
                ny = (uz * wx) - (ux * wz)
                nz = (ux * wy) - (uy * wx)
                ln = (nx * nx + ny * ny + nz * nz) ** 0.5
                if ln <= 1.0e-12:
                    # Imported retail WOKs can intentionally retain
                    # zero-area legacy faces.  The embedded room-MDL AABB
                    # still needs a finite unit plane for Odyssey's face
                    # table, so preserve the face indices and use the
                    # horizontal plane through its first vertex.
                    nx, ny, nz = 0.0, 0.0, 1.0
                else:
                    nx, ny, nz = nx / ln, ny / ln, nz / ln
                # Odyssey stores the plane in implicit form n·p + d = 0.
                # True K2 CHITIN rooms (001ebo1/r00_test) therefore carry
                # d = -dot(n, point).  The previous positive sign made every
                # generated embedded AABB disagree with its external WOK.
                dist = -((nx * ax) + (ny * ay) + (nz * az))
            except (IndexError, ValueError):
                nx, ny, nz, dist = 0.0, 0.0, 1.0, 0.0
            adjacency = []
            for e0, e1 in ((v1, v2), (v2, v3), (v3, v1)):
                others = [f for f in edge_map.get((min(e0, e1), max(e0, e1)), []) if f != i]
                adjacency.append(others[0] if others else 0xFFFF)
            # 32-byte face entry:
            #   normal(12) plane_dist(4) material(4) adj_faces(6) vertices(6)
            buf.write(struct.pack('<fff', nx, ny, nz))
            buf.write(_wf32(dist))
            buf.write(_wu32(mat))
            buf.write(struct.pack('<HHH', *adjacency))
            buf.write(struct.pack('<HHH', v1, v2, v3))
            flat_face_indices.extend((int(v1), int(v2), int(v3)))

        # Patch faces_off
        end = buf.tell()
        buf.seek(faces_off_patch)
        buf.write(_wu32(faces_off_abs - _BASE))
        buf.seek(end)

        render_counts = self._mesh_render_batch_counts(node, len(flat_face_indices))
        if render_counts:
            _align4(buf)
            counts_off_abs = buf.tell()
            for count in render_counts:
                buf.write(_wu32(count))
            end = buf.tell()
            buf.seek(indices_counts_desc_patch)
            buf.write(_wu32(counts_off_abs - _BASE))
            buf.write(_wu32(len(render_counts)))
            buf.write(_wu32(len(render_counts)))
            buf.seek(end)

        # ── MDL vertex array (fallback XYZ only) ────────────────────────────
        _align4(buf)
        verts_off_abs = buf.tell()
        for vx, vy, vz in node.vertices:
            buf.write(struct.pack('<fff', vx, vy, vz))

        # Patch verts_off
        end = buf.tell()
        buf.seek(verts_off_patch)
        buf.write(_wu32(verts_off_abs - _BASE))
        buf.seek(end)

        if render_counts:
            counters = self._mesh_render_batch_counters(node, len(render_counts))

            _align4(buf)
            offsets_off_abs = buf.tell()
            offset_value_patches: list[int] = []
            for _ in render_counts:
                offset_value_patches.append(buf.tell())
                buf.write(_wu32(0))
            end = buf.tell()
            buf.seek(indices_offsets_desc_patch)
            buf.write(_wu32(offsets_off_abs - _BASE))
            buf.write(_wu32(len(render_counts)))
            buf.write(_wu32(len(render_counts)))
            buf.seek(end)

            _align4(buf)
            counters_off_abs = buf.tell()
            for counter in counters:
                buf.write(_wu32(counter))
            end = buf.tell()
            buf.seek(counters_desc_patch)
            buf.write(_wu32(counters_off_abs - _BASE))
            buf.write(_wu32(len(counters)))
            buf.write(_wu32(len(counters)))
            buf.seek(end)

            cursor = 0
            batch_offsets: list[int] = []
            for count in render_counts:
                batch_offsets.append(buf.tell() - _BASE)
                for index in flat_face_indices[cursor:cursor + count]:
                    buf.write(_wu16(index))
                cursor += count

            end = buf.tell()
            for patch_pos, batch_off in zip(offset_value_patches, batch_offsets):
                buf.seek(patch_pos)
                buf.write(_wu32(batch_off))
            buf.seek(end)

    def _mesh_render_batch_counts(self, node: ModelNode, flat_index_count: int) -> list[int]:
        if flat_index_count <= 0:
            return []
        raw_counts = list(getattr(node, 'mesh_indices_counts', []) or [])
        counts: list[int] = []
        for value in raw_counts:
            try:
                count = int(value)
            except Exception:
                continue
            if count > 0:
                counts.append(count)
        if counts and sum(counts) == flat_index_count:
            return counts
        return [flat_index_count]

    @staticmethod
    def _inverted_mesh_counter(seq: int) -> int:
        """KOTOR's per-mesh 'inverted counter' (KotorBlender get_inverted_counter).

        seq is the 1-based mesh ordinal in DFS write order. For seq<100 this is
        99-seq (verified against vanilla 001ebo1: mesh#1=98 ... #20=79). The
        engine uses it to size/index a runtime mesh array; our old value of 0
        collided and overran (swkotor2+0x4920e array-grow crash).
        """

        quo = seq // 100
        mod = seq % 100
        return int(pow(2, quo) * 100 - seq + (100 * quo if mod else 0) + (0 if quo else -1)) & 0xFFFFFFFF

    def _mesh_render_batch_counters(self, node: ModelNode, batch_count: int) -> list[int]:
        raw_counters = (
            list(getattr(node, 'mesh_inverted_counters', []) or [])
            or list(getattr(node, 'mesh_counters', []) or [])
        )
        if raw_counters:
            counters: list[int] = []
            for value in raw_counters[:batch_count]:
                try:
                    counters.append(int(value) & 0xFFFFFFFF)
                except Exception:
                    counters.append(0)
            while len(counters) < batch_count:
                counters.append(0)
            return counters
        # No source counters: synthesize the sequential per-mesh value.
        self._mesh_seq = getattr(self, "_mesh_seq", 0) + 1
        return [self._inverted_mesh_counter(self._mesh_seq)] * max(1, batch_count)

    def _write_skin_header(self, buf: BytesIO, node: ModelNode) -> None:
        """
        Write the 100-byte skin sub-header (``_SkinmeshHeader`` in PyKotor
        parlance) that immediately follows the mesh header for SKIN nodes,
        plus the qbone / tbone / bonemap data arrays pointed at by it.

        Layout — verified field-for-field against PyKotor
        ``io_mdl._SkinmeshHeader.read`` / ``.write`` (100 bytes on the nose):

          +0   unknown_weights       int32
          +4   unknown3              int32
          +8   unknown4              int32
          +12  offset_to_mdx_weights uint32  (MDX weight-channel byte offset)
          +16  offset_to_mdx_bones   uint32  (MDX bone-ref-channel byte offset)
          +20  offset_to_bonemap     uint32  (rel BASE)
          +24  bonemap_count         uint32
          +28  offset_to_qbones      uint32
          +32  qbones_count          uint32
          +36  qbones_count2         uint32
          +40  offset_to_tbones      uint32
          +44  tbones_count          uint32
          +48  tbones_count2         uint32
          +52  offset_to_unknown0    uint32
          +56  unknown0_count        uint32
          +60  unknown0_count2       uint32
          +64  bones[16]             uint16[16]  (32 B — NOT 17!)
          +96  unknown1              uint32      (4 B pad — NOT 2!)
          = 100 B

        Previous revisions wrote ``bone_parts[17]`` + 2 pad bytes.  That summed
        to 100 B by coincidence but desynchronised PyKotor's reader on the
        ``bones`` tuple length and trailing padding.
        """
        _, ch_offsets = self._mdx_stride_for(node)

        sw_off  = ch_offsets.get('sw', 0xFFFFFFFF)
        sbr_off = ch_offsets.get('br', 0xFFFFFFFF)

        # ── On-disk array semantics (verified against vanilla K2 c_drexlf raw
        # bytes, reone mdlmdxreader.cpp, and MDLOps; see CHANGES.md T2526) ──
        #
        #   bonemap  : NODE-indexed.  bonemap[node_id] = palette_slot (float32)
        #              or -1.0 for nodes not in this skin's palette.
        #              Length = total node count of the model.
        #   qbones   : NODE-indexed Vector4 bind quaternion per node id.
        #   tbones   : NODE-indexed Vector3 bind translation per node id.
        #   bones[16]: palette-indexed uint16.  bones[slot] = node_id.
        #              (MDX bone-ref channel floats index into THIS palette.)
        #
        # Previous revisions wrote bonemap/qbones/tbones palette-indexed
        # (bonemap[slot] = node_id, palette-length arrays).  The game engine
        # builds its bone palette by scanning bonemap per node id, so a
        # palette-indexed array made it bind the wrong bones — MDLedit
        # reported "The bone numbers do not match up", the game crashed on
        # load, and MDLOps' weight repair produced garbage deformation.
        #
        # In-memory ``node.bone_map`` is the compact palette (bone names, MDX
        # refs index into it).  ``node.bone_map_floats`` / ``qbone_list`` /
        # ``tbone_list`` are authoritative NODE-indexed copies captured at
        # binary load; freshly-built exports carry palette-length arrays
        # (skeleton_builder / split remap), which we expand to node-indexed
        # form here.
        bone_map_names = list(node.bone_map or [])
        recorded_node_ids = list(
            getattr(node, 'bone_node_indices', []) or []
        )
        palette_node_ids = []
        for slot, bone_name in enumerate(bone_map_names):
            try:
                recorded_id = int(recorded_node_ids[slot])
            except (IndexError, TypeError, ValueError):
                recorded_id = -1
            if (
                0 <= recorded_id < len(self._nodes)
                and str(getattr(self._nodes[recorded_id], 'name', '') or '').lower()
                    == str(bone_name or '').lower()
            ):
                palette_node_ids.append(recorded_id)
            else:
                palette_node_ids.append(self._skin_bone_node_index(bone_name))
        n_nodes          = len(self._nodes)

        def _is_node_indexed(arr: list) -> bool:
            # Node-indexed pass-through only when the array spans the whole
            # node table AND that length is distinguishable from the palette.
            return bool(arr) and len(arr) == n_nodes and n_nodes != len(bone_map_names)

        raw_bm_floats = list(getattr(node, 'bone_map_floats', []) or [])
        if _is_node_indexed(raw_bm_floats):
            bonemap_floats = [float(v) for v in raw_bm_floats]
        else:
            bonemap_floats = [-1.0] * n_nodes
            for slot, nid in enumerate(palette_node_ids):
                if 0 <= nid < n_nodes:
                    bonemap_floats[nid] = float(slot)
        bm_cnt = len(bonemap_floats)

        def _expand_bind(entries: list, filler: tuple) -> list:
            """Return a node-indexed bind array (qbones/tbones)."""
            if _is_node_indexed(entries):
                return [tuple(e) for e in entries]
            out = [filler] * n_nodes
            for slot, nid in enumerate(palette_node_ids):
                if 0 <= nid < n_nodes and slot < len(entries):
                    out[nid] = tuple(entries[slot])
            return out

        qbones_in = list(getattr(node, 'qbone_list', []) or [])
        tbones_in = list(getattr(node, 'tbone_list', []) or [])
        # Vanilla files fill non-palette slots with (-1,0,0,0) / (0,0,0).
        qbones = _expand_bind(qbones_in, (-1.0, 0.0, 0.0, 0.0)) if qbones_in else []
        tbones = _expand_bind(tbones_in, (0.0, 0.0, 0.0)) if tbones_in else []

        skin_hdr_start = buf.tell()

        # +0  unknown_weights / unknown3 / unknown4 (3×int32)
        buf.write(b'\x00' * 12)
        # +12 MDX channel offsets (weights / bone refs)
        buf.write(_wu32(sw_off))
        buf.write(_wu32(sbr_off))
        # +20 bonemap desc (patched below)
        bm_off_patch = buf.tell()
        buf.write(_wu32(0))
        buf.write(_wu32(bm_cnt))
        # +28 qbones desc (patched below)
        qb_off_patch = buf.tell()
        buf.write(_wu32(0))
        buf.write(_wu32(len(qbones)))
        buf.write(_wu32(len(qbones)))
        # +40 tbones desc (patched below)
        tb_off_patch = buf.tell()
        buf.write(_wu32(0))
        buf.write(_wu32(len(tbones)))
        buf.write(_wu32(len(tbones)))
        # +52 offset_to_unknown0 / counts.  Vanilla K2 skin nodes carry this
        # as a node-indexed float32 block alongside bonemap/qbones/tbones; the
        # values are unused, but external validators compare the family of
        # node-indexed skin arrays.
        unknown0_count = bm_cnt if bm_cnt > 0 else 0
        unknown0_off_patch = buf.tell()
        buf.write(_wu32(0))
        buf.write(_wu32(unknown0_count))
        buf.write(_wu32(unknown0_count))
        # +64 bones[16] uint16 — partial-skin "head bone" references.
        # Unused slots are zero-filled to match vanilla binary padding and to
        # avoid confusing tools that compare all 16 entries byte-for-byte.
        for i in range(16):
            bone_idx = palette_node_ids[i] if i < len(palette_node_ids) else -1
            if bone_idx >= 0:
                buf.write(_wu16(bone_idx))
            else:
                buf.write(_wu16(0))
        # +96 unknown1 uint32 pad
        buf.write(_wu32(0))

        assert buf.tell() - skin_hdr_start == 100, (
            f"Skin sub-header wrong size: {buf.tell() - skin_hdr_start}")

        # ── Data arrays pointed-to by the header above ────────────────────
        # Bonemap float32 array.  Count = bm_cnt (may be 0 for skin nodes
        # whose weights use direct bone indices instead of a remap table).
        _align4(buf)
        if bm_cnt > 0:
            bm_abs = buf.tell()
            for v in bonemap_floats[:bm_cnt]:
                buf.write(_wf32(float(v)))
            # Pad the array up to bm_cnt entries (if provided list is short).
            pad_entries = bm_cnt - len(bonemap_floats)
            for _ in range(max(0, pad_entries)):
                buf.write(_wf32(-1.0))
            end = buf.tell()
            buf.seek(bm_off_patch)
            buf.write(_wu32(bm_abs - _BASE))
            buf.seek(end)

        # qbones: Vector4 each (16 bytes)
        if qbones:
            _align4(buf)
            qb_abs = buf.tell()
            for qx, qy, qz, qw in qbones:
                buf.write(struct.pack('<ffff', qx, qy, qz, qw))
            end = buf.tell()
            buf.seek(qb_off_patch)
            buf.write(_wu32(qb_abs - _BASE))
            buf.seek(end)

        # tbones: Vector3 each (12 bytes)
        if tbones:
            _align4(buf)
            tb_abs = buf.tell()
            for tx, ty, tz in tbones:
                buf.write(struct.pack('<fff', tx, ty, tz))
            end = buf.tell()
            buf.seek(tb_off_patch)
            buf.write(_wu32(tb_abs - _BASE))
            buf.seek(end)

        # unknown0: node-indexed float32 array, observed as zero-filled in
        # vanilla K2 skins.  GhostRigger does not consume the values, but
        # emitting the block keeps the skin-header array counts vanilla-shaped.
        if unknown0_count:
            _align4(buf)
            unknown0_abs = buf.tell()
            for _ in range(unknown0_count):
                buf.write(_wf32(0.0))
            end = buf.tell()
            buf.seek(unknown0_off_patch)
            buf.write(_wu32(unknown0_abs - _BASE))
            buf.seek(end)

    def _skin_bone_node_index(self, name: str) -> int:
        """Return the node id this writer will emit for a skin palette bone."""
        key = str(name or '').lower()
        if not key:
            return -1
        return int(getattr(self, '_node_index_by_name', {}).get(key, -1))

    def _write_dangly_header(self, buf: BytesIO, node: ModelNode) -> None:
        """
        Write the 28-byte dangly header that follows the mesh header.
        Layout (verified against stock K2 PFHA04 and reone):
          +0  constraints_off uint32
          +4  constraints_cnt uint32
          +8  constraints_cnt2 uint32
          +12 displacement float
          +16 tightness float
          +20 period float
          +24 rest_vertices_off uint32

        Both pointers use the normal MDL ``absolute file offset - 12``
        convention.  Retail dangly meshes store one constraint float and one
        local-space rest-position vec3 per rendered vertex.  The second array
        is not an optional runtime pointer: Odyssey uses it as the simulation
        baseline.
        """
        constraints = list(node.dangly_constraints or [])
        vertices = list(node.vertices or [])
        if len(constraints) != len(vertices):
            raise ValueError(
                f"Dangly node {node.name!r} needs one constraint per vertex: "
                f"{len(constraints)} constraints for {len(vertices)} vertices"
            )
        # Convert back to 0-255 range for binary format
        cst_denorm = [max(0.0, min(255.0, c * 255.0)) for c in constraints]
        cst_cnt = len(cst_denorm)

        cst_off_patch = buf.tell()
        buf.write(_wu32(0))             # constraints_off (patched)
        buf.write(_wu32(cst_cnt))
        buf.write(_wu32(cst_cnt))
        buf.write(_wf32(node.dangly_displacement or 0.0))
        buf.write(_wf32(node.dangly_tightness or 0.25))
        buf.write(_wf32(node.dangly_period or 1.0))
        rest_off_patch = buf.tell()
        buf.write(_wu32(0))             # local-space rest vertices (patched)

        if not cst_cnt:
            return

        # Constraint float array.
        _align4(buf)
        cst_abs = buf.tell()
        for constraint in cst_denorm:
            buf.write(_wf32(constraint))

        # Retail PFHA04 stores the exact mesh-local vertex positions again as
        # the dangly simulation rest pose.
        _align4(buf)
        rest_abs = buf.tell()
        for x, y, z in vertices:
            buf.write(struct.pack(
                "<fff",
                _sanitize_float(x),
                _sanitize_float(y),
                _sanitize_float(z),
            ))

        end = buf.tell()
        buf.seek(cst_off_patch)
        buf.write(_wu32(cst_abs - _BASE))
        buf.seek(rest_off_patch)
        buf.write(_wu32(rest_abs - _BASE))
        buf.seek(end)

    @staticmethod
    def _build_aabb_tree(vertices, faces) -> list:
        """Build a KOTOR AABB walkmesh tree (ported from KotorBlender aabb.py).

        Returns a flat list of nodes in DFS order; each node is
        ``[min_xyz(3), max_xyz(3), left_idx, right_idx, face_idx, plane]``
        where child indices reference positions in this list (-1 for leaves)
        and ``plane`` is 0 for leaves or 1/2/3 for split axis X/Y/Z.
        """

        def bbox(face_list):
            lo = [1.0e9, 1.0e9, 1.0e9]
            hi = [-1.0e9, -1.0e9, -1.0e9]
            cx = cy = cz = 0.0
            for _idx, verts, centroid in face_list:
                for v in verts:
                    for a in range(3):
                        if v[a] < lo[a]:
                            lo[a] = v[a]
                        if v[a] > hi[a]:
                            hi[a] = v[a]
                cx += centroid[0]
                cy += centroid[1]
                cz += centroid[2]
            n = len(face_list)
            return lo, hi, (cx / n, cy / n, cz / n)

        def emit(tree, face_list, depth=0):
            lo, hi, center = bbox(face_list)
            if len(face_list) == 1 or depth > 128:
                tree.append([lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], -1, -1, face_list[0][0], 0])
                return
            axis = max(range(3), key=lambda a: hi[a] - lo[a])
            left, right = [], []
            for i in range(4):
                left, right = [], []
                for face in face_list:
                    (left if face[2][axis] < center[axis] else right).append(face)
                if left and right:
                    break
                if i == 3:
                    src = left if left else right
                    dst = right if left else left
                    for _ in range(len(src) // 2):
                        dst.append(src.pop())
                axis = (axis + 1) % 3
            node_index = len(tree)
            tree.append([lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], 0, 0, -1, 1 + axis])
            tree[node_index][6] = len(tree)
            emit(tree, left, depth + 1)
            tree[node_index][7] = len(tree)
            emit(tree, right, depth + 1)

        face_list = []
        for i, face in enumerate(faces):
            try:
                v0, v1, v2 = (vertices[int(face[0])], vertices[int(face[1])], vertices[int(face[2])])
            except (IndexError, ValueError):
                continue
            centroid = ((v0[0] + v1[0] + v2[0]) / 3.0, (v0[1] + v1[1] + v2[1]) / 3.0, (v0[2] + v1[2] + v2[2]) / 3.0)
            face_list.append((i, (v0, v1, v2), centroid))
        tree: list = []
        if face_list:
            emit(tree, face_list)
        return tree

    # KOTOR AABB significant-plane bit flags (KotorBlender mdl_types.py).
    _AABB_PLANE = {0: 0x00, 1: 0x01, 2: 0x02, 3: 0x04}

    def _write_aabb_section(self, buf: BytesIO, node: ModelNode) -> None:
        """Write the AABB tree pointer + a real recursive walkmesh tree.

        On-disk layout matches KotorBlender's proven exporter (loads in-game):
          +0  offset_to_aabb  int32  (BASE-relative pointer to entry 0)
          then N × 40-byte entries:
            bounding_box_min   3×float (12)
            bounding_box_max   3×float (12)
            offset_to_left     uint32  (4)  BASE-relative, 0 for leaves
            offset_to_right    uint32  (4)
            leaf_face_index    int32   (4)  -1 for internal nodes
            most_significant_plane uint32 (4)  0 leaf / 0x01,0x02,0x04 X,Y,Z

        The tree is built from the node's own vertices+faces (the walkmesh
        geometry). A room MDL without this node crashed the engine's area
        loader (type-guarded NULL deref at swkotor2+0x4b3a8).
        """
        tree = self._build_aabb_tree(node.vertices, node.faces)
        if not tree:
            # No faces: emit a single degenerate leaf so the pointer is valid.
            bb_min = getattr(node, 'mesh_bb_min', None) or (0.0, 0.0, 0.0)
            bb_max = getattr(node, 'mesh_bb_max', None) or (0.0, 0.0, 0.0)
            tree = [[bb_min[0], bb_min[1], bb_min[2], bb_max[0], bb_max[1], bb_max[2], -1, -1, -1, 0]]

        tree_off_rel = (buf.tell() + 4) - _BASE
        buf.write(_wi32(tree_off_rel))
        entry_base = buf.tell() - _BASE  # BASE-relative offset of entry 0
        for entry in tree:
            left_idx, right_idx, face_idx, plane = entry[6], entry[7], entry[8], entry[9]
            off_left = 0 if face_idx != -1 else entry_base + left_idx * 40
            off_right = 0 if face_idx != -1 else entry_base + right_idx * 40
            buf.write(struct.pack('<ffffff', *entry[:6]))
            buf.write(_wu32(off_left))
            buf.write(_wu32(off_right))
            buf.write(_wi32(face_idx))
            buf.write(_wu32(self._AABB_PLANE.get(plane, 0)))

    def _write_emitter_header(self, buf: BytesIO, node: ModelNode) -> None:
        """
        Write the 224-byte emitter header (verified against mdl_parser._parse_emitter).
        """
        ep = node.emitter_params
        buf.write(_wf32(ep.get('deadspace', 0.0)))
        buf.write(_wf32(ep.get('blastradius', 0.0)))
        buf.write(_wf32(ep.get('blastlength', 0.0)))
        buf.write(_wu32(ep.get('numbranches', 0)))
        # MDLOps/PyKotor's 224-byte emitter header stores this as uint32,
        # despite older prose documentation describing it as a float.
        buf.write(_wu32(ep.get('controlptsmoothing', 0)))
        buf.write(_wu32(ep.get('xgrid', 0)))
        buf.write(_wu32(ep.get('ygrid', 0)))
        buf.write(_wu32(ep.get('spawntype', 0)))
        buf.write(_wstr(ep.get('update', ''), 32))
        buf.write(_wstr(ep.get('emitter_render', ''), 32))
        buf.write(_wstr(ep.get('blend', ''), 32))
        buf.write(_wstr(ep.get('texture', ''), 32))
        buf.write(_wstr(ep.get('chunkname', ''), 16))
        buf.write(_wu32(ep.get('twosidedtex', 0)))
        buf.write(_wu32(ep.get('loop', 0)))
        buf.write(_wu16(ep.get('renderorder', 0)))
        buf.write(struct.pack('B', ep.get('frameblending', 0)))
        buf.write(_wstr(ep.get('depth_texture_name', ''), 32))
        buf.write(struct.pack('B', int(ep.get('unknown1', 0)) & 0xFF))
        # Parsed rooms supply the authoritative packed field.  Only synthesize
        # flags from friendly booleans for a newly authored emitter.
        flags = int(ep.get('flags', ep.get('emitter_flags', 0)) or 0)
        _flag_map = [
            ('p2p', 0x0001), ('p2p_sel', 0x0002), ('affected_wind', 0x0004),
            ('tinted', 0x0008), ('bounce', 0x0010), ('random', 0x0020),
            ('inherit', 0x0040), ('inheritvel', 0x0080), ('inherit_local', 0x0100),
            ('splat', 0x0200), ('inherit_part', 0x0400), ('depth_texture', 0x0800),
        ]
        if 'flags' not in ep and 'emitter_flags' not in ep:
            for fname, fbit in _flag_map:
                if ep.get(fname, 0):
                    flags |= fbit
        buf.write(_wu32(flags))

    def _write_reference_header(self, buf: BytesIO, node: ModelNode) -> None:
        """Write the 36-byte reference node header."""
        ep = node.emitter_params
        ref_model = str(getattr(node, 'reference_model', '') or ep.get('ref_model', '') or '')
        reattach = 1 if bool(
            getattr(node, 'reference_reattachable', False)
            or ep.get('reattachable', False)
        ) else 0
        buf.write(_wstr(ref_model, 32))
        buf.write(_wu32(reattach))

    # ─────────────────────────────  Animations  ─────────────────────────────

    def _write_animation(self, buf: BytesIO, anim: Animation) -> None:
        """
        Write one animation block.
        Layout mirrors the model geometry + model headers:
          Geometry header (80 B) — uses anim fp1/fp2
          Animation model header (52 B)
          Event array
          Animation node tree
        """
        has_source_tree = self._animation_has_source_hierarchy(anim)
        anim_nodes = self._animation_nodes_with_hierarchy(anim, self._nodes)
        self._validate_animation_export_tree(
            anim,
            anim_nodes,
            self._nodes,
            allow_source_subset=has_source_tree,
        )
        anim_root_name = anim.anim_root or (anim_nodes[0].name if anim_nodes else '')
        # The animation header declares the serialized animation-tree span,
        # not the target geometry's node count.  A target-native sparse tree
        # intentionally omits unrelated branches; declaring the full geometry
        # span makes UpdateAnimFootprint walk past the actual child traversal.
        node_count = len(anim_nodes)

        fp1 = _K2_ANIM_FP1 if self._is_k2 else _K1_ANIM_FP1
        fp2 = _K2_ANIM_FP2 if self._is_k2 else _K1_ANIM_FP2

        geo_start = buf.tell()

        # Geometry header (80 bytes)
        buf.write(_wu32(fp1))
        buf.write(_wu32(fp2))
        buf.write(_wstr(anim.name or 'default', 32))
        anim_root_off_patch = buf.tell()
        buf.write(_wu32(0))              # root_node_off (patched)
        buf.write(_wu32(node_count))
        # Animation geometry headers use the same MaxTree subtype byte at
        # +0x4C; stock animation blocks use value 5.
        buf.write(b'\x00' * (76 - (buf.tell() - geo_start)))
        buf.write(b'\x05')               # geometry_type = 5 (animation)
        buf.write(b'\x00' * (80 - (buf.tell() - geo_start)))

        # Animation model header (starts at geo_start+80)
        assert buf.tell() == geo_start + 80
        buf.write(_wf32(anim.length or 0.0))
        transition_time = getattr(anim, 'transition_time', None)
        buf.write(_wf32(0.25 if transition_time is None else transition_time))
        buf.write(_wstr(anim_root_name, 32))

        events_off_patch = buf.tell()
        buf.write(_wu32(0))              # events_off (patched)
        buf.write(_wu32(len(anim.events)))
        buf.write(_wu32(len(anim.events)))
        # Runtime model-data base slot at animation +0x84. K2's 0x512800
        # unconditionally writes mdlBase here before relocating the root.
        # Omitting this word placed a zero-event root at +0x84, so the engine
        # overwrote its flags, skipped node relocation, and crashed while
        # dereferencing the still-raw child_arr_off at swkotor2+0x4962c.
        buf.write(_wu32(0))

        # Events array
        _align4(buf)
        if anim.events:
            events_off_abs = buf.tell()
            for ev in anim.events:
                buf.write(_wf32(ev.time))
                buf.write(_wstr(ev.name or '', 32))
            # Patch events_off
            end = buf.tell()
            buf.seek(events_off_patch)
            buf.write(_wu32(events_off_abs - _BASE))
            buf.seek(end)

        # Animation node tree
        if anim_nodes:
            anim_node_abs: Dict[int, int] = {}
            root_off = self._write_anim_node_tree(
                buf,
                anim_nodes[0],
                anim_node_abs,
                animation_geometry_off=geo_start - _BASE,
            )

            # Patch root_off in geo header
            end = buf.tell()
            buf.seek(anim_root_off_patch)
            buf.write(_wu32(root_off))
            buf.seek(end)

    def _animation_nodes_with_hierarchy(self, anim: Animation, all_nodes: List[ModelNode]) -> List[ModelNode]:
        """Return animation nodes as a full target geometry-hierarchy tree.

        Retarget preview can use sparse/flat animation-node lists because
        playback resolves controllers by name. The KotOR engine walks the
        binary animation tree itself in routines such as UpdateAnimFootprint,
        so every serialized parent edge must follow the target Aurora tree.
        A carried, target-native sparse tree is preserved. A carried donor tree
        is rebuilt on the target ancestor closure so required shoulders, neck
        segments, and helpers are restored without bloating every clip with
        unrelated mesh siblings. A truly flat override still expands to the
        complete target tree because it carries no hierarchy evidence at all.
        """
        source_nodes = list(getattr(anim, 'nodes', None) or [])
        if self._animation_has_source_hierarchy(anim):
            if self._animation_source_hierarchy_matches_target(anim, all_nodes):
                return self._clone_source_animation_tree(source_nodes)
            return self._rebuild_source_animation_tree_on_target(
                source_nodes,
                all_nodes,
            )

        if not all_nodes:
            return []

        order_by_name = {
            str(getattr(node, 'name', '') or '').lower(): index
            for index, node in enumerate(all_nodes or [])
            if getattr(node, 'name', '')
        }

        def clone_stub_node(node: ModelNode) -> ModelNode:
            cloned = node.clone_shallow() if hasattr(node, 'clone_shallow') else copy.copy(node)
            cloned.children = []
            cloned.parent = None
            cloned.controllers = []
            return cloned

        controller_by_name: Dict[str, list] = {}
        for node in source_nodes:
            key = str(getattr(node, 'name', '') or '').lower()
            if key:
                controller_by_name[key] = list(getattr(node, 'controllers', []) or [])

        local_by_source_id: Dict[int, ModelNode] = {}
        fallback_orientation_times = self._animation_export_key_times(anim)
        for geom in all_nodes:
            clone = clone_stub_node(geom)
            key = str(getattr(geom, 'name', '') or '').lower()
            clone.controllers = list(controller_by_name.get(key, []))
            self._ensure_export_orientation_controller(clone, fallback_orientation_times)
            local_by_source_id[id(geom)] = clone

        roots: List[ModelNode] = []
        for geom in all_nodes:
            node = local_by_source_id[id(geom)]
            parent = getattr(geom, 'parent', None)
            parent_node = local_by_source_id.get(id(parent)) if parent is not None else None
            if parent_node is not None and parent_node is not node:
                node.parent = parent_node
                parent_node.children.append(node)
            else:
                roots.append(node)

        root_node = min(
            roots,
            key=lambda node: order_by_name.get(
                str(getattr(node, 'name', '') or '').lower(),
                1_000_000,
            ),
        ) if roots else None

        ordered: List[ModelNode] = []
        seen = set()

        def visit(node: ModelNode) -> None:
            nid = id(node)
            if nid in seen:
                return
            seen.add(nid)
            ordered.append(node)
            node.children.sort(
                key=lambda child: order_by_name.get(
                    str(getattr(child, 'name', '') or '').lower(),
                    1_000_000,
                )
            )
            for child in node.children:
                visit(child)

        if root_node is not None:
            visit(root_node)
        for node in sorted(
            local_by_source_id.values(),
            key=lambda item: order_by_name.get(
                str(getattr(item, 'name', '') or '').lower(),
                1_000_000,
            ),
        ):
            visit(node)
        return ordered

    @staticmethod
    def _animation_has_source_hierarchy(anim: Animation) -> bool:
        """Return True when the animation already carries a donor node tree."""

        source_nodes = list(getattr(anim, 'nodes', None) or [])
        if not source_nodes:
            return False
        source_ids = {id(node) for node in source_nodes}
        for node in source_nodes:
            for child in getattr(node, 'children', []) or []:
                if id(child) in source_ids:
                    return True
        return False

    @classmethod
    def _animation_source_hierarchy_matches_target(
        cls,
        anim: Animation,
        target_nodes: List[ModelNode],
    ) -> bool:
        """Return True only when a carried animation tree is target-native.

        Retargeted clips often arrive with valid node names but with the donor
        rig's parent edges.  GhostRigger playback resolves tracks by name and
        can hide that mismatch; Odyssey walks the serialized animation tree
        and requires every carried edge to agree with the model hierarchy.
        Preserve a sparse source tree only when its direct parent edges are
        already target-correct.  Otherwise the normal export path overlays its
        controllers onto the complete target hierarchy, restoring required
        shoulder, neck, helper, and mesh placeholders.

        Real binary models can contain duplicate node names, so node IDs are
        preferred whenever they are unique.  Name matching is only the
        fallback for synthetic/editor trees whose IDs were not populated.
        """

        source_nodes = list(getattr(anim, 'nodes', None) or [])
        if not cls._animation_has_source_hierarchy(anim) or not target_nodes:
            return False

        identity_mode = cls._animation_node_identity_mode(
            source_nodes,
            target_nodes,
        )
        target_by_identity = {
            cls._animation_node_identity_key(node, identity_mode): node
            for node in target_nodes
        }
        source_ids = {id(node) for node in source_nodes}
        for source in source_nodes:
            source_identity = cls._animation_node_identity_key(
                source,
                identity_mode,
            )
            target = target_by_identity.get(source_identity)
            if target is None:
                return False

            source_parent = getattr(source, 'parent', None)
            source_parent_identity = (
                cls._animation_node_identity_key(source_parent, identity_mode)
                if source_parent is not None and id(source_parent) in source_ids
                else None
            )
            target_parent = getattr(target, 'parent', None)
            target_parent_identity = (
                cls._animation_node_identity_key(target_parent, identity_mode)
                if target_parent is not None else None
            )
            if source_parent_identity != target_parent_identity:
                return False
        return True

    @staticmethod
    def _animation_node_identity_mode(
        source_nodes: List[ModelNode],
        target_nodes: List[ModelNode],
    ) -> str:
        """Choose an unambiguous animation-to-geometry identity contract."""

        source_indexes = [int(getattr(node, 'index', 0) or 0) for node in source_nodes]
        target_indexes = [int(getattr(node, 'index', 0) or 0) for node in target_nodes]
        if (
            len(source_indexes) == len(set(source_indexes))
            and len(target_indexes) == len(set(target_indexes))
            and set(source_indexes).issubset(set(target_indexes))
        ):
            return 'index'

        def name(node: ModelNode) -> str:
            return str(getattr(node, 'name', '') or '').strip().lower()

        source_names = [name(node) for node in source_nodes]
        target_names = [name(node) for node in target_nodes]
        if (
            all(source_names)
            and all(target_names)
            and len(source_names) == len(set(source_names))
            and len(target_names) == len(set(target_names))
            and set(source_names).issubset(set(target_names))
        ):
            return 'name'

        raise ValueError(
            "Animation nodes cannot be matched safely to the target hierarchy: "
            "node IDs are not unique and node names are duplicated or missing."
        )

    @staticmethod
    def _animation_node_identity_key(node: ModelNode, mode: str):
        if mode == 'index':
            return int(getattr(node, 'index', 0) or 0)
        return str(getattr(node, 'name', '') or '').strip().lower()

    def _rebuild_source_animation_tree_on_target(
        self,
        source_nodes: List[ModelNode],
        target_nodes: List[ModelNode],
    ) -> List[ModelNode]:
        """Overlay source controllers on the target node ancestor closure."""

        identity_mode = self._animation_node_identity_mode(
            source_nodes,
            target_nodes,
        )
        source_by_identity = {
            self._animation_node_identity_key(node, identity_mode): node
            for node in source_nodes
        }
        target_by_identity = {
            self._animation_node_identity_key(node, identity_mode): node
            for node in target_nodes
        }
        missing = sorted(set(source_by_identity).difference(target_by_identity))
        if missing:
            raise ValueError(
                "Animation tree references nodes missing from the target model: "
                f"{missing!r}."
            )

        required_ids: set[int] = set()
        for identity in source_by_identity:
            target = target_by_identity[identity]
            seen: set[int] = set()
            while target is not None and id(target) not in seen:
                seen.add(id(target))
                required_ids.add(id(target))
                target = getattr(target, 'parent', None)

        ordered_targets = [
            node for node in target_nodes
            if id(node) in required_ids
        ]
        clones: Dict[int, ModelNode] = {}
        for target in ordered_targets:
            clone = target.clone_shallow() if hasattr(target, 'clone_shallow') else copy.copy(target)
            clone.children = []
            clone.parent = None
            source = source_by_identity.get(
                self._animation_node_identity_key(target, identity_mode)
            )
            clone.controllers = copy.deepcopy(
                list(getattr(source, 'controllers', []) or [])
                if source is not None else []
            )
            clones[id(target)] = clone

        for target in ordered_targets:
            clone = clones[id(target)]
            parent = getattr(target, 'parent', None)
            parent_clone = clones.get(id(parent)) if parent is not None else None
            if parent_clone is not None:
                clone.parent = parent_clone
                parent_clone.children.append(clone)

        return [clones[id(target)] for target in ordered_targets]

    def _clone_source_animation_tree(self, source_nodes: List[ModelNode]) -> List[ModelNode]:
        """Clone an existing animation tree without expanding it to geometry nodes."""

        source_ids = {id(node) for node in source_nodes}
        clones: Dict[int, ModelNode] = {}
        order_by_source = {id(node): index for index, node in enumerate(source_nodes)}

        def clone_node(node: ModelNode) -> ModelNode:
            cloned = node.clone_shallow() if hasattr(node, 'clone_shallow') else copy.copy(node)
            cloned.children = []
            cloned.parent = None
            cloned.controllers = copy.deepcopy(list(getattr(node, 'controllers', []) or []))
            clones[id(node)] = cloned
            return cloned

        for node in source_nodes:
            clone_node(node)

        child_ids: set[int] = set()
        for node in source_nodes:
            cloned_parent = clones[id(node)]
            children = [
                child
                for child in getattr(node, 'children', []) or []
                if id(child) in source_ids
            ]
            children.sort(key=lambda child: order_by_source.get(id(child), 1_000_000))
            for child in children:
                child_ids.add(id(child))
                cloned_child = clones[id(child)]
                cloned_child.parent = cloned_parent
                cloned_parent.children.append(cloned_child)

        roots = [node for node in source_nodes if id(node) not in child_ids]
        if not roots:
            roots = source_nodes[:1]

        ordered: List[ModelNode] = []
        seen: set[int] = set()

        def visit(node: ModelNode) -> None:
            nid = id(node)
            if nid in seen:
                return
            seen.add(nid)
            cloned = clones[nid]
            ordered.append(cloned)
            for child in getattr(node, 'children', []) or []:
                if id(child) in source_ids:
                    visit(child)

        for root in roots:
            visit(root)
        for node in source_nodes:
            visit(node)

        return ordered

    @staticmethod
    def _animation_export_key_times(anim: Animation) -> List[float]:
        times: List[float] = []
        for node in getattr(anim, 'nodes', []) or []:
            for ctrl in getattr(node, 'controllers', []) or []:
                for raw_time in ctrl.get('times', []) or []:
                    try:
                        value = float(raw_time)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(value) and value >= 0.0:
                        times.append(value)
        if times:
            return sorted(set(round(value, 7) for value in times))

        length = float(getattr(anim, 'length', 0.0) or 0.0)
        if math.isfinite(length) and length > 0.0:
            return [0.0, round(length, 7)]
        return [0.0]

    @staticmethod
    def _normalized_xyzw(values: Tuple[float, float, float, float]) -> List[float]:
        raw = list(values or (0.0, 0.0, 0.0, 1.0))[:4]
        while len(raw) < 4:
            raw.append(1.0 if len(raw) == 3 else 0.0)
        mag_sq = sum(float(value) * float(value) for value in raw)
        if mag_sq <= 1e-12 or not math.isfinite(mag_sq):
            return [0.0, 0.0, 0.0, 1.0]
        mag = math.sqrt(mag_sq)
        return [float(value) / mag for value in raw]

    @classmethod
    def _ensure_export_orientation_controller(cls, node: ModelNode, times: List[float]) -> None:
        for ctrl in getattr(node, 'controllers', []) or []:
            if int(ctrl.get('type', 0) or 0) == CTRL_ORIENTATION or str(ctrl.get('name', '')).lower() == 'orientation':
                return
        quat = cls._normalized_xyzw(getattr(node, 'rotation', (0.0, 0.0, 0.0, 1.0)))
        node.controllers.append(
            {
                'type': CTRL_ORIENTATION,
                'name': 'orientation',
                'columns': 4,
                'times': list(times or [0.0]),
                'values': [list(quat) for _ in (times or [0.0])],
            }
        )

    @classmethod
    def _validate_animation_export_tree(
        cls,
        anim: Animation,
        anim_nodes: List[ModelNode],
        target_nodes: List[ModelNode],
        *,
        allow_source_subset: bool = False,
    ) -> None:
        """Fail early if the binary animation tree no longer mirrors target nodes."""

        if not target_nodes:
            return
        expected_names = [str(getattr(node, 'name', '') or '') for node in target_nodes]
        actual_names = [str(getattr(node, 'name', '') or '') for node in anim_nodes]
        if allow_source_subset:
            identity_mode = cls._animation_node_identity_mode(
                anim_nodes,
                target_nodes,
            )
            expected_keys = {
                cls._animation_node_identity_key(node, identity_mode)
                for node in target_nodes
            }
            actual_keys = [
                cls._animation_node_identity_key(node, identity_mode)
                for node in anim_nodes
            ]
            missing_from_target = [
                key for key in actual_keys if key not in expected_keys
            ]
            if missing_from_target:
                raise ValueError(
                    f"Animation '{getattr(anim, 'name', '')}' export tree references nodes not present "
                    f"in the target model hierarchy: {missing_from_target!r}."
                )

            actual_by_identity = {
                cls._animation_node_identity_key(node, identity_mode): node
                for node in anim_nodes
            }
            target_by_identity = {
                cls._animation_node_identity_key(node, identity_mode): node
                for node in target_nodes
            }
            actual_object_ids = {id(node) for node in anim_nodes}
            for actual in anim_nodes:
                if actual.parent is actual or any(child is actual for child in getattr(actual, 'children', []) or []):
                    raise ValueError(
                        f"Animation '{getattr(anim, 'name', '')}' export tree has a self-reference at "
                        f"node '{getattr(actual, 'name', '')}'."
                    )
                for child in getattr(actual, 'children', []) or []:
                    if id(child) not in actual_object_ids:
                        raise ValueError(
                            f"Animation '{getattr(anim, 'name', '')}' export tree references missing child "
                            f"node '{getattr(child, 'name', '')}'."
                        )
                actual_identity = cls._animation_node_identity_key(
                    actual,
                    identity_mode,
                )
                target = target_by_identity.get(actual_identity)
                if target is None:
                    continue
                actual_parent = getattr(actual, 'parent', None)
                target_parent = getattr(target, 'parent', None)
                actual_parent_identity = (
                    cls._animation_node_identity_key(actual_parent, identity_mode)
                    if actual_parent is not None else None
                )
                target_parent_identity = (
                    cls._animation_node_identity_key(target_parent, identity_mode)
                    if target_parent is not None else None
                )
                if actual_parent_identity != target_parent_identity:
                    actual_name = str(getattr(actual, 'name', '') or '')
                    actual_parent_name = (
                        str(getattr(actual_parent, 'name', '') or '')
                        if actual_parent is not None else ''
                    )
                    target_parent_name = (
                        str(getattr(target_parent, 'name', '') or '')
                        if target_parent is not None else ''
                    )
                    raise ValueError(
                        f"Animation '{getattr(anim, 'name', '')}' export tree has donor parent "
                        f"'{actual_parent_name}' for node '{actual_name}'; target parent is "
                        f"'{target_parent_name}'."
                    )
            return

        if actual_names != expected_names:
            raise ValueError(
                f"Animation '{getattr(anim, 'name', '')}' export tree has {len(anim_nodes)} nodes "
                f"but target model hierarchy has {len(target_nodes)} nodes; binary Aurora "
                "animations must preserve the full target hierarchy, including unkeyed placeholders "
                "and original node-name casing."
            )

        actual_by_name = {
            str(getattr(node, 'name', '') or ''): node
            for node in anim_nodes
            if getattr(node, 'name', '')
        }
        for target, actual in zip(target_nodes, anim_nodes):
            expected_parent = (
                str(getattr(target.parent, 'name', '') or '')
                if getattr(target, 'parent', None) is not None
                else None
            )
            actual_parent = (
                str(getattr(actual.parent, 'name', '') or '')
                if getattr(actual, 'parent', None) is not None
                else None
            )
            if actual_parent != expected_parent:
                raise ValueError(
                    f"Animation '{getattr(anim, 'name', '')}' export tree parent mismatch for "
                    f"node '{getattr(actual, 'name', '')}': expected {expected_parent!r}, got {actual_parent!r}."
                )

            expected_children = [
                str(getattr(child, 'name', '') or '')
                for child in getattr(target, 'children', []) or []
            ]
            actual_children = [
                str(getattr(child, 'name', '') or '')
                for child in getattr(actual, 'children', []) or []
            ]
            if actual_children != expected_children:
                raise ValueError(
                    f"Animation '{getattr(anim, 'name', '')}' export tree child mismatch for "
                    f"node '{getattr(actual, 'name', '')}': expected {expected_children!r}, got {actual_children!r}."
                )
            if actual.parent is actual or any(child is actual for child in getattr(actual, 'children', []) or []):
                raise ValueError(
                    f"Animation '{getattr(anim, 'name', '')}' export tree has a self-reference at "
                    f"node '{getattr(actual, 'name', '')}'."
                )
            for child_name in actual_children:
                if child_name not in actual_by_name:
                    raise ValueError(
                        f"Animation '{getattr(anim, 'name', '')}' export tree references missing child "
                        f"node '{child_name}'."
                    )

    def _write_anim_node_tree(
        self,
        buf: BytesIO,
        root: ModelNode,
        anim_node_abs: Dict[int, int],
        *,
        animation_geometry_off: int,
    ) -> int:
        """Write animation nodes in vanilla depth-first layout.

        Stock Aurora animation blocks place a node's controller arrays after
        its child subtrees, not immediately after the node header.  The engine's
        ResetMdlNode/UpdateAnimFootprint path mutates the raw child offset
        arrays into runtime pointer arrays while walking this layout, so the
        emitted order here intentionally mirrors MDLOps/vanilla blocks:

            node header -> child pointer array -> child subtrees -> controllers
        """
        root_off_patches: Dict[int, int] = {}
        parent_off_patches: Dict[int, int] = {}
        anim_child_data_locs: Dict[int, int] = {}
        anim_controller_patches: Dict[int, Tuple[int, int]] = {}
        ordered: List[ModelNode] = []
        seen: set[int] = set()

        def _write_depth_first(node: ModelNode) -> None:
            nid = id(node)
            if nid in seen:
                return
            seen.add(nid)
            ordered.append(node)
            _align4(buf)
            anim_node_abs[nid] = buf.tell()
            self._write_one_anim_node_header_and_child_array(
                buf,
                node,
                root_off_patches,
                parent_off_patches,
                anim_child_data_locs,
                anim_controller_patches,
            )
            for child in getattr(node, "children", []) or []:
                _write_depth_first(child)
            ctrl_patches = anim_controller_patches.get(nid)
            if ctrl_patches is not None:
                self._write_controllers(buf, node, ctrl_patches[0], ctrl_patches[1])

        _write_depth_first(root)

        # Patch root_off, parent_off, and child ptr arrays after every child
        # node has a concrete file offset.
        root_abs = anim_node_abs[id(root)]
        cur_end = buf.tell()

        for nd in ordered:
            parent_abs = anim_node_abs.get(id(nd.parent), 0)
            # Static-model nodes store 0 at +8, but animation nodes do not:
            # every vanilla K1/K2 animation node points back to its owning
            # animation geometry block. K2's 0x512930 relocation dispatch uses
            # this owner link; writing 0 left child_arr_off raw (for c_rancor's
            # cpause1, 0x2CAC) and the following 0x449450 size walk dereferenced
            # that small offset as a pointer at swkotor2+0x4962c.
            if id(nd) in root_off_patches:
                buf.seek(root_off_patches[id(nd)])
                buf.write(_wu32(animation_geometry_off))
            if id(nd) in parent_off_patches:
                buf.seek(parent_off_patches[id(nd)])
                buf.write(_wu32(parent_abs - _BASE if parent_abs else 0))

        buf.seek(cur_end)

        for nd in ordered:
            if not nd.children:
                continue
            arr_loc = anim_child_data_locs.get(id(nd))
            if arr_loc is None:
                continue
            buf.seek(arr_loc)
            for child in nd.children:
                c_abs = anim_node_abs.get(id(child), 0)
                buf.write(_wu32(c_abs - _BASE if c_abs else 0))

        buf.seek(cur_end)
        return root_abs - _BASE

    def _write_one_anim_node_header_and_child_array(
        self,
        buf: BytesIO,
        node: ModelNode,
        root_off_patches: Dict[int, int],
        parent_off_patches: Dict[int, int],
        anim_child_data_locs: Dict[int, int],
        anim_controller_patches: Dict[int, Tuple[int, int]],
    ) -> None:
        """Write one animation node header and its child pointer array.

        Real KotOR animation nodes are always DUMMY (type 1) regardless of what
        the geometry node carries.  Writing MESH/SKIN flags here would cause
        PyKotor (and the original NWN engine) to expect a full TrimeshHeader
        after the 80-byte base, which we do not emit for animation nodes.
        """
        node_start = buf.tell()
        name_idx = self._name_index_for(node.name)
        if getattr(self, '_animation_override_only', False):
            source_name_idx = int(getattr(node, 'index', name_idx))
            if (
                0 <= source_name_idx < len(getattr(self, '_names', []))
                and str(self._names[source_name_idx]).lower()
                == str(getattr(node, 'name', '') or '').lower()
            ):
                # The parsed node's ``index`` is the stock name-table index.
                # Preserve it so duplicate geometry names remain unambiguous.
                name_idx = source_name_idx

        # Animation nodes must always be DUMMY (0x0001) — see NWN binary spec
        # and PyKotor io_mdl.py.  Mesh flags belong only in geometry nodes.
        anim_flags = 1  # NodeType.DUMMY
        node_key = str(getattr(node, 'name', '') or '').lower()
        if getattr(self, '_animation_override_only', False):
            # Surgical injection preserves the stock geometry bytes, including
            # their non-DFS node numbers (PFBNM root children include 97/100/1).
            # PyKotor retains the +4 name-table index, so recover the matching
            # raw +2 node number from the source geometry tree.
            node_index = int(
                getattr(self, '_target_node_number_by_name_index', {}).get(
                    name_idx,
                    getattr(node, 'index', name_idx),
                )
            )
        elif getattr(self, '_preserve_native_supernode_numbers', False):
            # Native-rig rebuilds retain the donor's sparse +2 identity. Match
            # local animation tracks to that exact geometry identity by name.
            node_index = int(
                getattr(self, '_supernode_number_by_name', {}).get(
                    node_key,
                    getattr(node, 'number', getattr(node, 'index', name_idx)),
                )
            )
        else:
            # Generic full rebuilds keep the historical dense fallback.
            node_index = int(
                getattr(self, '_node_index_by_name', {}).get(
                    node_key,
                    getattr(node, 'index', name_idx),
                )
            )
        if not 0 <= node_index <= 0xFFFF:
            raise ValueError(
                f"Animation node '{getattr(node, 'name', '')}' has an invalid geometry node index: "
                f"{node_index}."
            )
        buf.write(_wu16(anim_flags))
        # +2 is the target geometry supernode identity, while +4 is the model
        # name-table index. Ghost preview resolves controllers by name, but
        # retail Aurora compares the sparse +2 identity directly.
        buf.write(_wu16(node_index))
        buf.write(_wu16(name_idx))
        buf.write(_wu16(0))

        root_off_patches[id(node)] = buf.tell()
        buf.write(_wu32(0))          # root_off (patched pass 2)
        parent_off_patches[id(node)] = buf.tell()
        buf.write(_wu32(0))          # parent_off (patched pass 2)

        px, py, pz = node.position or (0.0, 0.0, 0.0)
        buf.write(struct.pack('<fff', px, py, pz))
        rx, ry, rz, rw = node.rotation or (0.0, 0.0, 0.0, 1.0)
        buf.write(struct.pack('<ffff', rw, rx, ry, rz))

        # Children
        child_arr_patch = buf.tell()
        buf.write(_wu32(0))
        buf.write(_wu32(len(node.children)))
        buf.write(_wu32(len(node.children)))

        ctrl_arr_patch = buf.tell()
        buf.write(_wu32(0))
        ctrl_count = len(node.controllers)
        buf.write(_wu32(ctrl_count))
        buf.write(_wu32(ctrl_count))

        ctrl_data_patch = buf.tell()
        buf.write(_wu32(0))
        ctrl_data_cnt = self._count_ctrl_data(node)
        buf.write(_wu32(ctrl_data_cnt))
        buf.write(_wu32(ctrl_data_cnt))
        anim_controller_patches[id(node)] = (ctrl_arr_patch, ctrl_data_patch)

        assert buf.tell() == node_start + 80

        if node.children:
            # Child pointer array (placeholder zeros; patched in pass 2)
            _align4(buf)
            child_arr_data_off = buf.tell() - _BASE
            anim_child_data_locs[id(node)] = buf.tell()
            for _ in node.children:
                buf.write(_wu32(0))
            end = buf.tell()
            buf.seek(child_arr_patch)
            buf.write(_wu32(child_arr_data_off))
            buf.seek(end)
