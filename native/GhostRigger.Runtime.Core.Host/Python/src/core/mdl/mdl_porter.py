"""
KotOR Cross-Game Porter
=======================
Direct binary K1 ↔ K2 model conversion WITHOUT going through ASCII.

  porter = CrossGamePorter()
  k2_model = porter.port(k1_model, target_game='K2')
  MDLBinaryWriter().write(k2_model, 'output.mdl', 'output.mdx')

Also handles:
  • fp1/fp2 magic number patching
  • K2 mesh header 8-byte extra field insertion/removal
  • Supermodel name remapping (S_Female02 → S_Female03, etc.)
  • Texture name remapping via user-supplied lookup table
  • Game version tag update on all nodes

MDLBinaryWriter
==============
Writes a KotorModel back to binary .mdl + .mdx files directly,
bypassing MDLOps entirely.  Supports both K1 and K2 output.

This is the most complex part because we must rebuild all the
offset arrays from scratch.  The strategy used here is:
  1. Build the MDX buffer (per-vertex stride data: positions, normals, UVs)
  2. Walk the node tree and build each node's binary block
  3. Assemble the full MDL with correct offset fixups
"""

import struct
import math
import logging
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  KotOR magic FP constants
# ─────────────────────────────────────────────────────────────────────────────

FP1_K1 = 4273776   # geometry header fp1 for K1 binary
FP2_K1 = 4216096   # geometry header fp2 for K1 binary
FP1_K2 = 4285200   # geometry header fp1 for K2 binary
FP2_K2 = 4216320   # geometry header fp2 for K2 binary

# ─────────────────────────────────────────────────────────────────────────────
#  Supermodel name remapping tables
# ─────────────────────────────────────────────────────────────────────────────

# K1 → K2 supermodel remap
SUPERMODEL_K1_TO_K2 = {
    "s_female02":   "s_female03",
    "s_male02":     "s_male03",
    "s_female01":   "s_female01",   # same in both (old aliens)
    "s_male01":     "s_male01",
    "s_creature01": "s_creature01",
    "null":         "null",
    "":             "",
}

# K2 → K1 supermodel remap (reverse)
SUPERMODEL_K2_TO_K1 = {v: k for k, v in SUPERMODEL_K1_TO_K2.items()
                        if k != v}
SUPERMODEL_K2_TO_K1.update({
    "s_female03": "s_female02",
    "s_male03":   "s_male02",
})

# ─────────────────────────────────────────────────────────────────────────────
#  Default texture remapping tables (K1 ↔ K2)
# ─────────────────────────────────────────────────────────────────────────────

# Common textures that exist in both games under different names
# These are the most frequently needed remaps for porting characters
TEXTURE_REMAP_K1_TO_K2 = {
    # Player character textures
    "pmhc01":   "pmhc01",   # mostly same
    "pfhc01":   "pfhc01",
    # Common NPC shared textures
    "n_genguard001":  "n_genguard001",
    # Sith robes
    "p_sithrobe01":   "n_sithrobe01",
}

TEXTURE_REMAP_K2_TO_K1 = {v: k for k, v in TEXTURE_REMAP_K1_TO_K2.items()
                            if k != v}


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-Game Porter
# ─────────────────────────────────────────────────────────────────────────────

class CrossGamePorter:
    """
    Port a KotorModel from K1→K2 or K2→K1 entirely in-memory.

    Usage::

        from src.core.mdl.mdl_porter import CrossGamePorter
        porter = CrossGamePorter(texture_map={'old_tex': 'new_tex'})
        ported = porter.port(model, target_game='K2')

    What changes between K1 and K2:
      • fp1/fp2 in geometry header  (8 bytes at MDL offset 12/16)
      • 8-byte extra block in each MESH node header (+324/+328 in K2)
      • Supermodel name (S_Female02↔S_Female03, S_Male02↔S_Male03)
      • game_version attribute on KotorModel and each ModelNode
      • Optionally texture names (via lookup table)
    """

    def __init__(self,
                 texture_map:     Optional[Dict[str,str]] = None,
                 remap_supermodel: bool = True,
                 remap_textures: bool = True):
        self.texture_map      = texture_map or {}
        self.remap_supermodel = remap_supermodel
        self.remap_textures = remap_textures

    # ── Public API ──────────────────────────────────────────────────────────

    def port(self, model, target_game: str):
        """
        Return a deep-copied KotorModel ported to target_game ('K1' or 'K2').
        The original is not modified.
        """
        import copy
        from ..geometry.model_data import GameVersion

        if target_game.upper() not in ('K1', 'K2'):
            raise ValueError('target_game must be K1 or K2')
        target_v = GameVersion.K2 if target_game.upper() == 'K2' else GameVersion.K1
        if model.game_version == target_v:
            log.info(f"CrossGamePorter: model already {target_game}, cloning as-is")
            return copy.deepcopy(model)

        ported = copy.deepcopy(model)
        ported.game_version = target_v

        direction = "K1→K2" if target_v.name == 'K2' else "K2→K1"
        log.info(f"CrossGamePorter: {direction}  model={model.name!r}")

        # Remap supermodel
        if self.remap_supermodel:
            sm = (ported.supermodel or "").lower()
            table = SUPERMODEL_K1_TO_K2 if target_v.name == 'K2' else SUPERMODEL_K2_TO_K1
            if sm in table:
                new_sm = table[sm]
                log.info(f"  supermodel: {sm!r} → {new_sm!r}")
                ported.supermodel = new_sm

        # Walk all nodes and remap textures + update version
        self._walk_nodes(ported.root_node, target_v)

        # Remap textures in animations (some animations reference texture controllers)
        for anim in ported.animations:
            anim.game_version = target_v
            for node in (getattr(anim, "nodes", None) or []):
                self._walk_nodes(node, target_v)

        return ported

    def build_texture_report(self, model) -> List[Tuple[str,str,str]]:
        """
        Analyse all texture references in a model and report what needs remapping.
        Returns list of (node_name, old_tex, suggested_new_tex).
        """
        report = []
        all_nodes = list(self._iter_nodes(model.root_node))
        from ..geometry.model_data import GameVersion
        target_game = 'K2' if model.game_version == GameVersion.K1 else 'K1'
        table = TEXTURE_REMAP_K1_TO_K2 if target_game == 'K2' else TEXTURE_REMAP_K2_TO_K1
        for node in all_nodes:
            tex = getattr(node, 'texture', '') or ''
            if tex:
                new_tex = self.texture_map.get(tex.lower()) or table.get(tex.lower()) or tex
                report.append((node.name, tex, new_tex))
        return report

    # ── Internal ─────────────────────────────────────────────────────────────

    def _iter_nodes(self, node):
        if node is None:
            return
        yield node
        for child in (node.children or []):
            yield from self._iter_nodes(child)

    def _walk_nodes(self, node, target_v):
        if node is None:
            return
        node.game_version = target_v

        # Remap texture names
        for attr in (('texture', 'lightmap') if self.remap_textures else ()):
            tex = getattr(node, attr, '') or ''
            if tex:
                # User-supplied map takes priority, then built-in table
                table = TEXTURE_REMAP_K1_TO_K2 if target_v.name == 'K2' else TEXTURE_REMAP_K2_TO_K1
                new_tex = (self.texture_map.get(tex.lower())
                           or table.get(tex.lower())
                           or tex)
                if new_tex != tex:
                    log.debug(f"  {node.name}.{attr}: {tex!r} → {new_tex!r}")
                setattr(node, attr, new_tex)

        for child in (node.children or []):
            self._walk_nodes(child, target_v)


# ─────────────────────────────────────────────────────────────────────────────
#  Binary MDL Writer
# ─────────────────────────────────────────────────────────────────────────────

class MDLBinaryWriter:
    """
    Write a KotorModel to binary .mdl + .mdx files.

    This is a FULL re-serialisation of the Aurora binary model format.
    Strategy:
      • We build two byte buffers: mdl_buf and mdx_buf
      • The MDL is split into:
          - 12-byte file header
          - Geometry header (80 bytes at offset BASE=12)
          - Model header (88 bytes)
          - Names array header + string table
          - Node blocks (variable size)
          - Animation blocks (variable size)
      • MDX holds only raw per-vertex stride data (positions, normals, UVs)
      • After building the node tree we go back and fix up all offset fields
        (stored as 32-bit offsets relative to BASE=12)

    Limitations (v1 – safe subset):
      • TRIMESH and SKIN nodes serialised with full vertex/face data
      • DANGLY, EMITTER, LIGHT, REFERENCE: node header only (no extra data)
      • Animations: full controller track serialisation
      • Tested against K1 and K2 binaries; round-trip parse→write→parse passes
    """

    BASE = 12   # MDL geometry header starts at byte 12

    def write(self, model, mdl_path: str, mdx_path: str = ""):
        """
        Write model to mdl_path (and optionally mdx_path).
        If mdx_path is empty, derive it from mdl_path (same name, .mdx extension).
        """
        from ..geometry.model_data import GameVersion
        if not mdx_path:
            mdx_path = str(Path(mdl_path).with_suffix('.mdx'))

        is_k2 = (model.game_version == GameVersion.K2)
        log.info(f"MDLBinaryWriter: writing {'K2' if is_k2 else 'K1'} "
                 f"model {model.name!r} → {mdl_path}")

        mdl_data, mdx_data = self.build(model)

        Path(mdl_path).write_bytes(mdl_data)
        if mdx_data:
            Path(mdx_path).write_bytes(mdx_data)
        log.info(f"  wrote {len(mdl_data)} MDL bytes, {len(mdx_data)} MDX bytes")

    def build(self, model) -> tuple:
        """
        Public convenience wrapper: build binary MDL+MDX bytes without writing to disk.
        Returns (mdl_bytes, mdx_bytes).
        """
        # Use the maintained Ghost Studio writer. The legacy private
        # _build below omits extension payloads and mesh vtable pairs.
        from .mdl_writer import MDLBinaryWriter as MaintainedBinaryWriter
        return MaintainedBinaryWriter().build(model)

    # ── Build ──────────────────────────────────────────────────────────────

    def _build(self, model, is_k2: bool) -> Tuple[bytes, bytes]:
        # K1HEADS01: the legacy layout below omits SKIN/DANGLY subheaders.
        # Use the canonical Ghost Studio writer, which preserves those payloads.
        from .mdl_writer import MDLBinaryWriter as CanonicalMDLBinaryWriter
        return CanonicalMDLBinaryWriter().write(model)

        from ..geometry.model_data import NodeFlags, GameVersion

        fp1 = FP1_K2 if is_k2 else FP1_K1
        fp2 = FP2_K2 if is_k2 else FP2_K1

        # ── 1. Collect all nodes in depth-first order ─────────────────────
        all_nodes = []
        def _collect(n):
            if n is None: return
            all_nodes.append(n)
            for c in (n.children or []):
                _collect(c)
        _collect(model.root_node)

        # ── 2. Build name table ───────────────────────────────────────────
        # ── 2. Build name table ───────────────────────────────────────────
        # names array: one entry per node, stored as null-terminated strings
        # Plan layout (all offsets relative to BASE):
        #   BASE+0:    geo header (80)
        #   BASE+80:   model fields (116) completing _ModelHeader.SIZE=0xC4
        #   BASE+196:  name ptr array (4 * n_nodes)
        #   BASE+196+4n: name strings (null-terminated, padded to 4-byte alignment)
        name_strs = [n.name for n in all_nodes]
        ptr_array_rel    = 196   # relative to BASE
        string_base_rel  = ptr_array_rel + 4 * len(all_nodes)

        # Build name string data
        name_data = bytearray()
        name_cum_offsets = []   # cumulative byte offsets within name_data
        for nm in name_strs:
            name_cum_offsets.append(len(name_data))
            name_data += nm.encode('ascii', 'replace') + b'\x00'
        while len(name_data) % 4:
            name_data += b'\x00'

        # Name ptr array: each entry = offset relative to BASE pointing at the string
        name_ptr_section = bytearray()
        for cum in name_cum_offsets:
            name_ptr_section += struct.pack('<I', string_base_rel + cum)

        # ── 3. Compute node section layout ────────────────────────────────
        #
        # TRUE KotOR binary layout (verified against KotorBlender writer.py):
        #
        #   For each node (in DFS order):
        #     [80-byte node header]
        #     [optional light sub-header, IMMEDIATELY after header]
        #     [optional emitter sub-header, IMMEDIATELY after light]
        #     [optional mesh sub-header, IMMEDIATELY after emitter/header]
        #
        #   Child pointer arrays → stored AFTER all node blocks, at absolute offsets
        #   Controller key arrays → stored AFTER child ptr arrays
        #   Controller data arrays → stored AFTER controller key arrays
        #   Face data → stored within mesh sub-header region (inline after mesh hdr)
        #
        # The node header's children_arr.offset points into the child ptr section.
        # The node header's ctrl_arr.offset points into the controller key section.
        # The node header's ctrl_data_arr.offset points into the controller data section.
        #
        # This means the mesh sub-header starts at node_off + 80 (not at node_off+80+child_ptrs+ctrl).

        MESH_HDR_SIZE = 340 if is_k2 else 332

        # Pre-serialize controllers for each node
        node_ctrl_entries:    List[List[bytes]]  = []
        node_ctrl_data_floats: List[List[float]] = []
        for node in all_nodes:
            ce, cf = self._serialize_controllers(node)
            node_ctrl_entries.append(ce)
            node_ctrl_data_floats.append(cf)

        # Pre-build MDX and face data for mesh nodes
        mdx = bytearray()
        node_mdx_offsets: Dict[int, int] = {}   # id(node) → offset in MDX
        node_face_blocks: Dict[int, bytes] = {}  # id(node) → face data bytes

        # First pass: compute node block sizes (= 80 + sub-header sizes)
        # This tells us where each node starts so we can compute face offsets.
        node_block_sizes: Dict[int, int] = {}
        for node in all_nodes:
            flags = node.flags or 0
            sz = 80  # base header always
            if flags & 0x0020:  # MESH
                sz += MESH_HDR_SIZE
            node_block_sizes[id(node)] = sz

        # Layout:
        #   node_section_start_rel: right after names
        node_section_start_rel = string_base_rel + len(name_data)
        node_offsets: Dict[int, int] = {}
        current_off = node_section_start_rel
        for node in all_nodes:
            node_offsets[id(node)] = current_off
            current_off += node_block_sizes[id(node)]
        node_section_end_rel = current_off

        # Face data section: for each mesh node, face block follows immediately
        # in the flat MDL. We place it after all node blocks in DFS order.
        # The mesh sub-header's faces_off field points here.
        face_section_start_rel = node_section_end_rel
        node_face_off_rel: Dict[int, int] = {}  # id(node) → abs offset of face data
        current_face_off = face_section_start_rel
        face_data_total = bytearray()
        for node in all_nodes:
            flags = node.flags or 0
            if not (flags & 0x0020):
                continue
            node_face_off_rel[id(node)] = current_face_off
            face_block = self._build_face_block(node)
            face_data_total += face_block
            current_face_off += len(face_block)
        face_section_end_rel = current_face_off

        # MDL vertex array section: for each mesh node, store XYZ fallback vertices
        # after the face data.  PyKotor uses vertices_offset as a fallback when
        # mdx_data_offset == 0 (the first mesh node always has MDX offset 0).
        vert_section_start_rel = face_section_end_rel
        node_vert_off_rel: Dict[int, int] = {}   # id(node) → rel offset of vert array
        current_vert_off = vert_section_start_rel
        vert_data_total = bytearray()
        for node in all_nodes:
            flags = node.flags or 0
            if not (flags & 0x0020):
                continue
            verts = node.vertices or []
            if verts:
                node_vert_off_rel[id(node)] = current_vert_off
                for vx, vy, vz in verts:
                    vert_data_total += struct.pack('<fff', vx, vy, vz)
                current_vert_off += 12 * len(verts)
        vert_section_end_rel = current_vert_off

        # Child ptr arrays: one per node, after vertex data
        child_ptr_section_start_rel = vert_section_end_rel
        node_child_arr_off: Dict[int, int] = {}
        current_off = child_ptr_section_start_rel
        for node in all_nodes:
            node_child_arr_off[id(node)] = current_off
            current_off += 4 * len(node.children or [])
        child_ptr_section_end_rel = current_off

        # Controller key arrays: after child ptrs
        ctrl_key_section_start_rel = child_ptr_section_end_rel
        node_ctrl_arr_off: Dict[int, int] = {}
        current_off = ctrl_key_section_start_rel
        for i, node in enumerate(all_nodes):
            node_ctrl_arr_off[id(node)] = current_off
            current_off += 16 * len(node_ctrl_entries[i])
        ctrl_key_section_end_rel = current_off

        # Controller data arrays: after ctrl key arrays
        ctrl_data_section_start_rel = ctrl_key_section_end_rel
        node_ctrl_data_off: Dict[int, int] = {}
        current_off = ctrl_data_section_start_rel
        for i, node in enumerate(all_nodes):
            node_ctrl_data_off[id(node)] = current_off
            current_off += 4 * len(node_ctrl_data_floats[i])
        ctrl_data_section_end_rel = current_off

        # Build MDX data for mesh nodes (needed to know mdx_data_off_in_mdx)
        for node in all_nodes:
            flags = node.flags or 0
            if not (flags & 0x0020):
                continue
            node_mdx_offsets[id(node)] = len(mdx)
            mdx_seg = self._build_mdx_segment(node)
            mdx += mdx_seg

        # ── 4. Build node blocks ──────────────────────────────────────────
        root_node_off = node_offsets.get(id(model.root_node), 0) if model.root_node else 0
        node_blocks: List[bytes] = []
        for i, node in enumerate(all_nodes):
            flags       = node.flags or 1
            node_off    = node_offsets[id(node)]
            node_idx    = i
            child_cnt   = len(node.children or [])
            ctrl_cnt    = len(node_ctrl_entries[i])
            ctrl_d_cnt  = len(node_ctrl_data_floats[i])
            parent_off  = (node_offsets.get(id(node.parent), 0)
                           if node.parent is not None else 0)

            # 80-byte node header
            hdr = bytearray(80)
            struct.pack_into('<H',    hdr,  0, int(flags) & 0xFFFF)
            struct.pack_into('<H',    hdr,  2, node_idx & 0xFFFF)    # node_number
            struct.pack_into('<H',    hdr,  4, node_idx & 0xFFFF)    # name_index
            struct.pack_into('<H',    hdr,  6, 0)                    # padding
            struct.pack_into('<I',    hdr,  8, root_node_off)        # off_root
            struct.pack_into('<I',    hdr, 12, parent_off)           # off_parent
            pos = node.position or (0.0, 0.0, 0.0)
            struct.pack_into('<fff',  hdr, 16, *pos)
            rot = node.rotation or (0.0, 0.0, 0.0, 1.0)
            x, y, z, w = rot
            struct.pack_into('<ffff', hdr, 28, w, x, y, z)          # binary: w,x,y,z
            # children array descriptor: (off, cnt, cnt2)
            struct.pack_into('<I',    hdr, 44, node_child_arr_off[id(node)])
            struct.pack_into('<I',    hdr, 48, child_cnt)
            struct.pack_into('<I',    hdr, 52, child_cnt)
            # controller key array descriptor
            struct.pack_into('<I',    hdr, 56, node_ctrl_arr_off[id(node)])
            struct.pack_into('<I',    hdr, 60, ctrl_cnt)
            struct.pack_into('<I',    hdr, 64, ctrl_cnt)
            # controller data array descriptor
            struct.pack_into('<I',    hdr, 68, node_ctrl_data_off[id(node)])
            struct.pack_into('<I',    hdr, 72, ctrl_d_cnt)
            struct.pack_into('<I',    hdr, 76, ctrl_d_cnt)

            blk = bytearray(hdr)

            # Mesh sub-header (IMMEDIATELY after 80-byte header, per KotOR spec)
            if flags & 0x0020:
                face_off     = node_face_off_rel.get(id(node), 0)
                mdx_data_off = node_mdx_offsets.get(id(node), 0)
                vert_off_rel = node_vert_off_rel.get(id(node), 0)
                mesh_hdr = self._build_mesh_header(node, is_k2,
                                                   face_off_rel=face_off,
                                                   mdx_data_off=mdx_data_off,
                                                   mdl_verts_off=vert_off_rel)
                blk += mesh_hdr

            node_blocks.append(bytes(blk))

        # ── 5. Build child pointer arrays ────────────────────────────────
        child_ptr_data = bytearray()
        for node in all_nodes:
            for child in (node.children or []):
                child_off = node_offsets.get(id(child), 0)
                child_ptr_data += struct.pack('<I', child_off)

        # ── 6. Build controller key + data arrays ────────────────────────
        ctrl_key_data = bytearray()
        for i, node in enumerate(all_nodes):
            for entry in node_ctrl_entries[i]:
                ctrl_key_data += entry

        ctrl_float_data = bytearray()
        for i, node in enumerate(all_nodes):
            for fv in node_ctrl_data_floats[i]:
                ctrl_float_data += struct.pack('<f', fv)

        # ── 7. Animation section ──────────────────────────────────────────
        anim_section_start_rel = ctrl_data_section_end_rel
        anim_blocks, anim_offsets = self._build_animations(
            model, all_nodes, node_offsets, is_k2, anim_section_start_rel)
        anim_data = b''.join(anim_blocks)

        anim_ptr_start_rel = anim_section_start_rel + len(anim_data)
        anim_ptr_data = bytearray()
        for ao in anim_offsets:
            anim_ptr_data += struct.pack('<I', ao)

        # ── 8. Assemble MDL ───────────────────────────────────────────────
        # Geometry header (80 bytes)
        geo_hdr = bytearray(80)
        struct.pack_into('<I', geo_hdr, 0, fp1)
        struct.pack_into('<I', geo_hdr, 4, fp2)
        nm_enc = model.name.encode('ascii','replace')[:32].ljust(32, b'\x00')
        geo_hdr[8:40] = nm_enc
        struct.pack_into('<I', geo_hdr, 40, root_node_off)
        struct.pack_into('<I', geo_hdr, 44, len(all_nodes))
        geo_hdr[76] = 2   # geometry type = 2 (model) at MaxTree +0x4C

        # Model fields (116 bytes after the 80-byte geometry header).  Together
        # with geo_hdr this is PyKotor's / the engine's 0xC4 _ModelHeader.
        mod_hdr = bytearray(116)
        mod_hdr[0] = model.model_type or 4
        mod_hdr[1] = getattr(model, 'subclassification', 0) & 0xFF
        mod_hdr[2] = getattr(model, 'padding0', getattr(model, 'unknown_byte', 0)) & 0xFF
        mod_hdr[3] = 1 if model.disable_fog else 0
        struct.pack_into('<I', mod_hdr,  8, anim_ptr_start_rel)
        struct.pack_into('<I', mod_hdr, 12, len(model.animations))
        struct.pack_into('<I', mod_hdr, 16, len(model.animations))
        bb_min = model.bb_min or (0,0,0)
        bb_max = model.bb_max or (0,0,0)
        struct.pack_into('<fff', mod_hdr, 24, *bb_min)
        struct.pack_into('<fff', mod_hdr, 36, *bb_max)
        struct.pack_into('<f',   mod_hdr, 48, model.radius or 1.0)
        struct.pack_into('<f',   mod_hdr, 52, model.anim_scale or 1.0)
        sm = (model.supermodel or "").encode('ascii','replace')[:32].ljust(32, b'\x00')
        mod_hdr[56:88] = sm
        struct.pack_into('<I', mod_hdr,  88, root_node_off)     # offset_to_super_root
        struct.pack_into('<I', mod_hdr,  92, 0)                 # mdx_data_buffer_offset
        struct.pack_into('<I', mod_hdr,  96, len(mdx))          # mdx_size
        struct.pack_into('<I', mod_hdr, 100, 0)                 # mdx_offset
        struct.pack_into('<I', mod_hdr, 104, ptr_array_rel)     # offset_to_name_offsets
        struct.pack_into('<I', mod_hdr, 108, len(all_nodes))    # name_offsets_count
        struct.pack_into('<I', mod_hdr, 112, len(all_nodes))    # name_offsets_count2

        # Assemble everything
        mdl_body = bytearray()
        mdl_body += geo_hdr
        mdl_body += mod_hdr
        mdl_body += name_ptr_section
        mdl_body += name_data
        for blk in node_blocks:
            mdl_body += blk
        mdl_body += face_data_total
        mdl_body += vert_data_total
        mdl_body += child_ptr_data
        mdl_body += ctrl_key_data
        mdl_body += ctrl_float_data
        mdl_body += anim_data
        mdl_body += anim_ptr_data

        mdl_size = self.BASE + len(mdl_body)
        file_hdr = struct.pack('<III', 0, mdl_size - self.BASE, len(mdx))

        return bytes(file_hdr) + bytes(mdl_body), bytes(mdx)

    def _serialize_controllers(self, node) -> Tuple[List[bytes], List[float]]:
        """
        Serialize a node's controller data to binary format.

        Returns (ctrl_entries, ctrl_data_floats) where:
          ctrl_entries   — list of 16-byte binary controller entry blocks
          ctrl_data_floats — list of float32 values (controller data array)

        If the node has raw controllers (from parse), use those directly.
        Otherwise generate minimal bind-pose controllers for position,
        orientation, alpha, and selfillum.

        Controller binary layout (16 bytes each):
          +0  ctrl_type  (uint32)
          +4  unknown    (uint16)  = 0
          +6  row_count  (uint16)  = number of keyframe rows
          +8  timekey_off(uint16)  = float index of first time key in data array
          +10 data_off   (uint16)  = float index of first value in data array
          +12 columns    (uint8)   = number of float values per keyframe
          +13 padding    (3 bytes) = 0,0,0
        """
        entries: List[bytes] = []
        floats:  List[float] = []

        raw_ctrls = getattr(node, 'controllers', None)
        if raw_ctrls:
            # Serialize raw controllers (from parsed binary MDL)
            # Build a shared float data array, then emit entries pointing into it.
            for ctrl in raw_ctrls:
                ctype   = ctrl.get('type',    0)
                times   = ctrl.get('times',   [])
                values  = ctrl.get('values',  [])
                columns = ctrl.get('columns', 1)
                if not times or not values:
                    continue
                row_count = min(len(times), len(values), 65535)
                # Time keys come first in the data array
                t_off = len(floats)
                for t in times[:row_count]:
                    floats.append(float(t))
                # Value data follows time keys
                d_off = len(floats)
                for row in values[:row_count]:
                    # row is a list of floats (may have more cols than 'columns')
                    for i in range(columns):
                        floats.append(float(row[i]) if i < len(row) else 0.0)

                entry = bytearray(16)
                struct.pack_into('<I', entry, 0,  ctype)
                struct.pack_into('<H', entry, 4,  0)
                struct.pack_into('<H', entry, 6,  row_count)
                struct.pack_into('<H', entry, 8,  t_off)
                struct.pack_into('<H', entry, 10, d_off)
                entry[12] = columns & 0xFF
                entries.append(bytes(entry))
            return entries, floats

        # No raw controllers: return empty.
        # Position and orientation are embedded directly in the 80-byte node header,
        # so synthetic bind-pose controllers are not needed.  Emitting synthetic
        # controllers causes PyKotor's _get_node_order traversal (which reads the
        # controller-count field as the child-count) to incorrectly walk into
        # controller data, crashing with out-of-bounds seeks.
        return entries, floats

    def _build_mdx_segment(self, node) -> bytes:
        """Build the per-vertex MDX data for a mesh node. Returns raw bytes."""
        verts   = node.vertices or []
        normals = node.normals  or []
        uvs     = node.uvs      or []
        uvs_lm  = node.uvs_lm  or []
        uvs_2   = getattr(node, 'uvs_2', None) or []
        uvs_3   = getattr(node, 'uvs_3', None) or []
        n_verts = len(verts)

        has_normals = len(normals) == n_verts and n_verts > 0
        has_uvs     = len(uvs)     == n_verts and n_verts > 0
        has_lm      = len(uvs_lm)  == n_verts and n_verts > 0
        has_uv2     = len(uvs_2)   == n_verts and n_verts > 0
        has_uv3     = len(uvs_3)   == n_verts and n_verts > 0

        stride    = 12   # positions always present
        mdx_n_off  = 0xFFFFFFFF
        mdx_t1_off = 0xFFFFFFFF
        mdx_lm_off = 0xFFFFFFFF
        mdx_t2_off = 0xFFFFFFFF
        mdx_t3_off = 0xFFFFFFFF

        if has_normals: mdx_n_off  = stride; stride += 12
        if has_uvs:     mdx_t1_off = stride; stride += 8
        if has_lm:      mdx_lm_off = stride; stride += 8
        if has_uv2:     mdx_t2_off = stride; stride += 8
        if has_uv3:     mdx_t3_off = stride; stride += 8

        seg = bytearray()
        for i in range(n_verts):
            row = bytearray(stride)
            struct.pack_into('<fff', row, 0, *verts[i])
            if has_normals and i < len(normals):
                struct.pack_into('<fff', row, mdx_n_off, *normals[i])
            if has_uvs and i < len(uvs):
                struct.pack_into('<ff',  row, mdx_t1_off, *uvs[i])
            if has_lm and i < len(uvs_lm):
                struct.pack_into('<ff',  row, mdx_lm_off, *uvs_lm[i])
            if has_uv2 and i < len(uvs_2):
                struct.pack_into('<ff',  row, mdx_t2_off, *uvs_2[i])
            if has_uv3 and i < len(uvs_3):
                struct.pack_into('<ff',  row, mdx_t3_off, *uvs_3[i])
            seg += row
        return bytes(seg)

    def _build_face_block(self, node) -> bytes:
        """Build the face data block for a mesh node. Returns raw bytes."""
        verts     = node.vertices or []
        faces     = node.faces    or []
        face_mats = node.face_mats or ([0] * len(faces))
        face_block = bytearray()
        for fi, face_tri in enumerate(faces):
            fb = bytearray(32)
            v1, v2, v3 = face_tri[0], face_tri[1], face_tri[2]
            mat = face_mats[fi] if fi < len(face_mats) else 0

            nx, ny, nz = 0.0, 0.0, 1.0
            d = 0.0
            if verts and v1 < len(verts) and v2 < len(verts) and v3 < len(verts):
                ax, ay, az = verts[v1]
                bx, by, bz = verts[v2]
                cx, cy, cz = verts[v3]
                ux, uy, uz = bx-ax, by-ay, bz-az
                vx, vy, vz = cx-ax, cy-ay, cz-az
                cx2 = uy*vz - uz*vy
                cy2 = uz*vx - ux*vz
                cz2 = ux*vy - uy*vx
                mag = math.sqrt(cx2*cx2 + cy2*cy2 + cz2*cz2)
                if mag > 1e-9:
                    nx, ny, nz = cx2/mag, cy2/mag, cz2/mag
                    d = nx*ax + ny*ay + nz*az

            struct.pack_into('<fff', fb,  0, nx, ny, nz)
            struct.pack_into('<f',   fb, 12, d)
            struct.pack_into('<I',   fb, 16, mat)
            struct.pack_into('<HHH', fb, 20, 0xFFFF, 0xFFFF, 0xFFFF)
            struct.pack_into('<HHH', fb, 26, v1, v2, v3)
            face_block += fb
        return bytes(face_block)

    def _build_mesh_header(self, node, is_k2: bool,
                           face_off_rel: int = 0,
                           mdx_data_off: int = 0,
                           mdl_verts_off: int = 0) -> bytes:
        """
        Build the fixed-size mesh sub-header (332 or 340 bytes for K2).

        This is written IMMEDIATELY after the 80-byte node header per the
        real KotOR binary format (verified against KotorBlender writer.py).

        MDX stride and channel offsets are computed from the node's data.
        face_off_rel: absolute offset (from BASE) to the face data.
        mdx_data_off: absolute offset into the MDX file for this mesh's vertex data.
        """
        MESH_HDR_SIZE = 340 if is_k2 else 332
        mh = bytearray(MESH_HDR_SIZE)

        verts   = node.vertices or []
        normals = node.normals  or []
        uvs     = node.uvs      or []
        uvs_lm  = node.uvs_lm  or []
        uvs_2   = getattr(node, 'uvs_2', None) or []
        uvs_3   = getattr(node, 'uvs_3', None) or []
        n_verts = len(verts)
        n_faces = len(node.faces or [])

        has_normals = len(normals) == n_verts and n_verts > 0
        has_uvs     = len(uvs)     == n_verts and n_verts > 0
        has_lm      = len(uvs_lm)  == n_verts and n_verts > 0
        has_uv2     = len(uvs_2)   == n_verts and n_verts > 0
        has_uv3     = len(uvs_3)   == n_verts and n_verts > 0

        stride      = 12
        mdx_v_off   = 0
        mdx_n_off   = 0xFFFFFFFF
        mdx_vc_off  = 0xFFFFFFFF
        mdx_t1_off  = 0xFFFFFFFF
        mdx_lm_off  = 0xFFFFFFFF
        mdx_t2_off  = 0xFFFFFFFF
        mdx_t3_off  = 0xFFFFFFFF
        mdx_tan1_off = 0xFFFFFFFF
        mdx_tan2_off = 0xFFFFFFFF
        mdx_tan3_off = 0xFFFFFFFF
        mdx_tan4_off = 0xFFFFFFFF

        if has_normals: mdx_n_off  = stride; stride += 12
        if has_uvs:     mdx_t1_off = stride; stride += 8
        if has_lm:      mdx_lm_off = stride; stride += 8
        if has_uv2:     mdx_t2_off = stride; stride += 8
        if has_uv3:     mdx_t3_off = stride; stride += 8

        # fp1/fp2 at +0/+4 — zero (unused in written models)
        # Faces array at +8
        struct.pack_into('<I', mh,  8, face_off_rel)
        struct.pack_into('<I', mh, 12, n_faces)
        struct.pack_into('<I', mh, 16, n_faces)

        # Bounding box
        bb_min = node.bb_min or (0,0,0)
        bb_max = node.bb_max or (0,0,0)
        struct.pack_into('<fff', mh, 20, *bb_min)
        struct.pack_into('<fff', mh, 32, *bb_max)

        # Colors
        diff = node.diffuse or (1,1,1)
        amb  = node.ambient or (1,1,1)
        transp = getattr(node, 'transparency_hint', 1)
        struct.pack_into('<fff', mh, 60, *diff)
        struct.pack_into('<fff', mh, 72, *amb)
        struct.pack_into('<I',   mh, 84, transp)

        # Texture names
        tex = (node.texture  or "").encode('ascii','replace')[:32].ljust(32, b'\x00')
        lm  = (node.lightmap or "").encode('ascii','replace')[:32].ljust(32, b'\x00')
        mh[88:120]  = tex
        mh[120:152] = lm
        _tnames = getattr(node, 'texture_names', []) or []
        bm3 = (_tnames[2] if len(_tnames)>2 else "").encode('ascii','replace')[:12].ljust(12, b'\x00')
        bm4 = (_tnames[3] if len(_tnames)>3 else "").encode('ascii','replace')[:12].ljust(12, b'\x00')
        mh[152:164] = bm3
        mh[164:176] = bm4

        # indices arrays, unknown12, saber unknowns — leave as zero
        # UV animation fields at +232..+251
        animate_uv = 1 if getattr(node, 'animate_uv', False) else 0
        struct.pack_into('<I', mh, 232, animate_uv)
        struct.pack_into('<f', mh, 236, float(getattr(node, 'uv_dir_x', 0.0) or 0.0))
        struct.pack_into('<f', mh, 240, float(getattr(node, 'uv_dir_y', 0.0) or 0.0))
        struct.pack_into('<f', mh, 244, float(getattr(node, 'uv_jitter', 0.0) or 0.0))
        struct.pack_into('<f', mh, 248, float(getattr(node, 'uv_jitter_speed', 0.0) or 0.0))

        # MDX data size + bitmap at +252/+256
        struct.pack_into('<I', mh, 252, stride)
        mdx_bitmap = 0
        if mdx_v_off   != 0xFFFFFFFF: mdx_bitmap |= 0x0001
        if mdx_n_off   != 0xFFFFFFFF: mdx_bitmap |= 0x0020
        if mdx_vc_off  != 0xFFFFFFFF: mdx_bitmap |= 0x0040
        if mdx_t1_off  != 0xFFFFFFFF: mdx_bitmap |= 0x0002
        if mdx_lm_off  != 0xFFFFFFFF: mdx_bitmap |= 0x0004
        if mdx_t2_off  != 0xFFFFFFFF: mdx_bitmap |= 0x0008
        if mdx_t3_off  != 0xFFFFFFFF: mdx_bitmap |= 0x0010
        if mdx_tan1_off != 0xFFFFFFFF: mdx_bitmap |= 0x0080
        struct.pack_into('<I', mh, 256, mdx_bitmap)

        # 11 MDX channel offsets at +260
        struct.pack_into('<I', mh, 260, mdx_v_off)
        struct.pack_into('<I', mh, 264, mdx_n_off)
        struct.pack_into('<I', mh, 268, mdx_vc_off)
        struct.pack_into('<I', mh, 272, mdx_t1_off)
        struct.pack_into('<I', mh, 276, mdx_lm_off)
        struct.pack_into('<I', mh, 280, mdx_t2_off)
        struct.pack_into('<I', mh, 284, mdx_t3_off)
        struct.pack_into('<I', mh, 288, mdx_tan1_off)
        struct.pack_into('<I', mh, 292, mdx_tan2_off)
        struct.pack_into('<I', mh, 296, mdx_tan3_off)
        struct.pack_into('<I', mh, 300, mdx_tan4_off)

        # Vert count + tex count at +304/+306
        tex_count = max(1, getattr(node, 'tex_count', 1) or 1)
        if has_lm  and tex_count < 2: tex_count = 2
        if has_uv2 and tex_count < 3: tex_count = 3
        if has_uv3 and tex_count < 4: tex_count = 4
        struct.pack_into('<H', mh, 304, n_verts)
        struct.pack_into('<H', mh, 306, tex_count)

        # Flags at +308..+313
        mh[308] = 1 if (node.has_lightmap and lm.rstrip(b'\x00')) else 0
        mh[311] = 1 if node.has_shadow else 0
        mh[313] = 1 if node.render else 0

        # K2 extra bytes at +314..+321 (dirt/hologram) — zero for generated models
        # Then 2 padding + total_area(0) + 4 unknown at +314/+316/+320/+(322/+324)
        # For K1:  +314(2pad) +316(area) +320(unk4) → mdx_data_off at +324
        # For K2:  +314(8 dirt) +322(2pad) +324(area) +328(unk4) → mdx_data_off at +332

        # MDX data offset at +324 (K1) or +332 (K2)
        mdx_field = 332 if is_k2 else 324
        struct.pack_into('<I', mh, mdx_field, mdx_data_off)

        # MDL fallback vertex array offset at +328 (K1) or +336 (K2)
        # PyKotor uses this as a fallback when mdx_data_offset == 0.
        verts_field = mdx_field + 4
        struct.pack_into('<I', mh, verts_field, mdl_verts_off)

        return bytes(mh)
        """Build the mesh sub-header + MDX vertex data."""
        verts    = node.vertices or []
        normals  = node.normals  or []
        uvs      = node.uvs      or []
        uvs_lm   = node.uvs_lm  or []
        faces    = node.faces    or []
        face_mats = node.face_mats or ([0] * len(faces))

        n_verts = len(verts)
        n_faces = len(faces)

        # ── Build MDX stride ──────────────────────────────────────────────────
        # KotorBlender MDX bitmap flags (verified from types.py + deadlystream.com):
        #   0x0001 = Vertex XYZ present         (slot 0, 12 bytes)
        #   0x0020 = Vertex Normals present      (slot 1, 12 bytes)
        #   0x0040 = Vertex Colors (RGBA)        (slot 2,  4 bytes) - rare
        #   0x0002 = Texture0 UV (tverts)        (slot 3,  8 bytes)
        #   0x0004 = Texture1/lightmap UV        (slot 4,  8 bytes)
        #   0x0008 = Texture2 UV                 (slot 5,  8 bytes)
        #   0x0010 = Texture3 UV                 (slot 6,  8 bytes)
        #   0x0080 = Tangent-space Tex0          (slot 7, 36 bytes)
        #   0x0100 = Tangent-space Tex1          (slot 8, 36 bytes) - rarely used
        #   0x0200 = Tangent-space Tex2          (slot 9, 36 bytes) - rarely used
        #   0x0400 = Tangent-space Tex3          (slot10, 36 bytes) - rarely used
        has_normals = len(normals) == n_verts
        has_uvs     = len(uvs)     == n_verts
        has_lm      = len(uvs_lm)  == n_verts
        uvs_2 = getattr(node, 'uvs_2', None) or []
        uvs_3 = getattr(node, 'uvs_3', None) or []
        has_uv2 = len(uvs_2) == n_verts and n_verts > 0
        has_uv3 = len(uvs_3) == n_verts and n_verts > 0

        stride = 12  # positions always
        mdx_v_off   = 0
        mdx_n_off   = 0xFFFFFFFF
        mdx_vc_off  = 0xFFFFFFFF
        mdx_t1_off  = 0xFFFFFFFF
        mdx_lm_off  = 0xFFFFFFFF
        mdx_t2_off  = 0xFFFFFFFF
        mdx_t3_off  = 0xFFFFFFFF
        mdx_tan1_off = 0xFFFFFFFF
        mdx_tan2_off = 0xFFFFFFFF
        mdx_tan3_off = 0xFFFFFFFF
        mdx_tan4_off = 0xFFFFFFFF

        if has_normals:
            mdx_n_off = stride; stride += 12
        if has_uvs:
            mdx_t1_off = stride; stride += 8
        if has_lm:
            mdx_lm_off = stride; stride += 8
        if has_uv2:
            mdx_t2_off = stride; stride += 8
        if has_uv3:
            mdx_t3_off = stride; stride += 8

        mdx_seg = bytearray()
        for i in range(n_verts):
            row = bytearray(stride)
            struct.pack_into('<fff', row, 0, *verts[i])
            if has_normals and i < len(normals):
                struct.pack_into('<fff', row, mdx_n_off, *normals[i])
            if has_uvs and i < len(uvs):
                struct.pack_into('<ff',  row, mdx_t1_off, *uvs[i])
            if has_lm and i < len(uvs_lm):
                struct.pack_into('<ff',  row, mdx_lm_off, *uvs_lm[i])
            if has_uv2 and i < len(uvs_2):
                struct.pack_into('<ff',  row, mdx_t2_off, *uvs_2[i])
            if has_uv3 and i < len(uvs_3):
                struct.pack_into('<ff',  row, mdx_t3_off, *uvs_3[i])
            mdx_seg += row

        # ── Build face block ──────────────────────────────────────────────
        # Each face = 32 bytes:
        #   +0   face_normal (3×float32 = 12 bytes)  — plane normal unit vector
        #   +12  plane_dist  (float32   =  4 bytes)  — signed plane distance from origin
        #   +16  mat_idx     (uint32    =  4 bytes)  — material/texture slot index
        #   +20  adj_faces   (3×uint16  =  6 bytes)  — adjacent face indices (0xFFFF=none)
        #   +26  vert_idx    (3×uint16  =  6 bytes)  — vertex indices
        #   Total: 32 bytes per face
        #
        # Face normals and plane distances are used by the engine for:
        #   - Back-face culling
        #   - Shadow volume extrusion
        #   - Collision detection (walkmesh faces)
        # Previously all normals were zero — now computed from vertex positions.
        face_block = bytearray()
        for fi, face_tri in enumerate(faces):
            fb = bytearray(32)
            v1, v2, v3 = face_tri[0], face_tri[1], face_tri[2]
            mat = face_mats[fi] if fi < len(face_mats) else 0

            # Compute face normal + plane distance from vertex positions
            # Normal = normalize((B-A) × (C-A))
            nx, ny, nz = 0.0, 0.0, 1.0  # default up-facing normal
            d = 0.0
            if verts and v1 < len(verts) and v2 < len(verts) and v3 < len(verts):
                ax, ay, az = verts[v1]
                bx, by, bz = verts[v2]
                cx, cy, cz = verts[v3]
                ux, uy, uz = bx - ax, by - ay, bz - az
                vx, vy, vz = cx - ax, cy - ay, cz - az
                cx2 = uy * vz - uz * vy
                cy2 = uz * vx - ux * vz
                cz2 = ux * vy - uy * vx
                mag = math.sqrt(cx2*cx2 + cy2*cy2 + cz2*cz2)
                if mag > 1e-9:
                    nx, ny, nz = cx2 / mag, cy2 / mag, cz2 / mag
                    # Odyssey plane equation is n·p + d = 0.  Vanilla room
                    # faces therefore store the negated point projection.
                    d = -(nx * ax + ny * ay + nz * az)

            struct.pack_into('<fff', fb,  0, nx, ny, nz)   # face normal
            struct.pack_into('<f',   fb, 12, d)             # plane distance
            struct.pack_into('<I',   fb, 16, mat)
            struct.pack_into('<HHH', fb, 20, 0xFFFF, 0xFFFF, 0xFFFF)  # adj faces
            struct.pack_into('<HHH', fb, 26, v1, v2, v3)
            face_block += fb

        # ── Mesh header (fixed 332 or 340 bytes for K2) ───────────────────
        MESH_HDR_SIZE = 340 if is_k2 else 332
        mh = bytearray(MESH_HDR_SIZE)

        # fp1/fp2 at +0/+4 (mesh node function pointers — just write zeros)
        # faces array — offset is absolute from BASE
        struct.pack_into('<I', mh, 8,  face_off_rel)
        struct.pack_into('<I', mh, 12, n_faces)
        struct.pack_into('<I', mh, 16, n_faces)

        # bounding box from node
        bb_min = node.bb_min or (0,0,0)
        bb_max = node.bb_max or (0,0,0)
        struct.pack_into('<fff', mh, 20, *bb_min)
        struct.pack_into('<fff', mh, 32, *bb_max)

        # texture name (32 bytes) and lightmap (32 bytes)
        tex = (node.texture or "").encode('ascii','replace')[:32].ljust(32, b'\x00')
        lm  = (node.lightmap or "").encode('ascii','replace')[:32].ljust(32, b'\x00')
        mh[88:120]  = tex
        mh[120:152] = lm

        # bitmap3 / bitmap4 (12 bytes each at +152/+164) — tertiary/quaternary texture names.
        # Rare; used in area/tile models. Mirror from node.texture_names if available.
        _tnames = getattr(node, 'texture_names', []) or []
        bm3 = (_tnames[2] if len(_tnames) > 2 else "").encode('ascii','replace')[:12].ljust(12, b'\x00')
        bm4 = (_tnames[3] if len(_tnames) > 3 else "").encode('ascii','replace')[:12].ljust(12, b'\x00')
        mh[152:164] = bm3
        mh[164:176] = bm4

        # Vert count + tex count at +304/+306
        tex_count = max(1, getattr(node, 'tex_count', 1) or 1)
        # If lightmap present, at least 2 texture slots
        if has_lm and tex_count < 2:
            tex_count = 2
        if has_uv2 and tex_count < 3:
            tex_count = 3
        if has_uv3 and tex_count < 4:
            tex_count = 4
        struct.pack_into('<H', mh, 304, n_verts)
        struct.pack_into('<H', mh, 306, tex_count)
        mh[308] = 1 if (node.has_lightmap  and lm.rstrip(b'\x00')) else 0
        mh[311] = 1 if node.has_shadow     else 0
        mh[313] = 1 if node.render         else 0

        # MDX data size / bitmap
        # Correct MDX bitmap flag values (verified vs KotorBlender types.py + deadlystream.com):
        #   0x0001 = Vertex XYZ present       (slot 0)
        #   0x0020 = Vertex Normals present   (slot 1)
        #   0x0040 = Vertex Colors (RGBA)     (slot 2) - rare
        #   0x0002 = Texture0 UV (tverts)     (slot 3)
        #   0x0004 = Texture1/lightmap UV     (slot 4)
        #   0x0008 = Texture2 UV              (slot 5)
        #   0x0010 = Texture3 UV              (slot 6)
        #   0x0080 = Tangent-space Tex0       (slot 7, 36 bytes per vertex)
        #   0x0100 = Tangent-space Tex1       (slot 8)
        #   0x0200 = Tangent-space Tex2       (slot 9)
        #   0x0400 = Tangent-space Tex3       (slot10)
        struct.pack_into('<I', mh, 252, stride)
        mdx_bitmap = 0
        if mdx_v_off   != 0xFFFFFFFF: mdx_bitmap |= 0x0001   # Vertex XYZ
        if mdx_n_off   != 0xFFFFFFFF: mdx_bitmap |= 0x0020   # Normals
        if mdx_vc_off  != 0xFFFFFFFF: mdx_bitmap |= 0x0040   # Vertex colors
        if mdx_t1_off  != 0xFFFFFFFF: mdx_bitmap |= 0x0002   # Texture0 UV
        if mdx_lm_off  != 0xFFFFFFFF: mdx_bitmap |= 0x0004   # Lightmap/Texture1 UV
        if mdx_t2_off  != 0xFFFFFFFF: mdx_bitmap |= 0x0008   # Texture2 UV
        if mdx_t3_off  != 0xFFFFFFFF: mdx_bitmap |= 0x0010   # Texture3 UV
        if mdx_tan1_off != 0xFFFFFFFF: mdx_bitmap |= 0x0080  # Tangent Tex0
        if mdx_tan2_off != 0xFFFFFFFF: mdx_bitmap |= 0x0100  # Tangent Tex1
        if mdx_tan3_off != 0xFFFFFFFF: mdx_bitmap |= 0x0200  # Tangent Tex2
        if mdx_tan4_off != 0xFFFFFFFF: mdx_bitmap |= 0x0400  # Tangent Tex3
        struct.pack_into('<I', mh, 256, mdx_bitmap)

        # MDX channel offsets (11 × 4 bytes at +260)
        # Slot mapping (KotorBlender confirmed):
        #   +260 slot 0: vertex XYZ
        #   +264 slot 1: normals
        #   +268 slot 2: vertex colors
        #   +272 slot 3: UV set 1 / Texture0
        #   +276 slot 4: UV set 2 / lightmap / Texture1
        #   +280 slot 5: UV set 3 / Texture2
        #   +284 slot 6: UV set 4 / Texture3
        #   +288 slot 7: Tangent-space Tex0
        #   +292 slot 8: Tangent-space Tex1
        #   +296 slot 9: Tangent-space Tex2
        #   +300 slot10: Tangent-space Tex3
        struct.pack_into('<I', mh, 260, mdx_v_off)
        struct.pack_into('<I', mh, 264, mdx_n_off)
        struct.pack_into('<I', mh, 268, mdx_vc_off)
        struct.pack_into('<I', mh, 272, mdx_t1_off)
        struct.pack_into('<I', mh, 276, mdx_lm_off)
        struct.pack_into('<I', mh, 280, mdx_t2_off)
        struct.pack_into('<I', mh, 284, mdx_t3_off)
        struct.pack_into('<I', mh, 288, mdx_tan1_off)
        struct.pack_into('<I', mh, 292, mdx_tan2_off)
        struct.pack_into('<I', mh, 296, mdx_tan3_off)
        struct.pack_into('<I', mh, 300, mdx_tan4_off)

        # MDX data offset at +324 (K1) or +332 (K2) — written directly (no placeholder)
        mdx_data_field = 332 if is_k2 else 324
        struct.pack_into('<I', mh, mdx_data_field, mdx_data_off)

        return bytes(mh)


    def _build_animations(self, model, all_nodes, node_offsets, is_k2,
                           anim_section_start_rel):
        """Build binary blocks for all animations. Returns (blocks, offsets)."""
        blocks  = []
        offsets = []
        current = anim_section_start_rel
        for anim in model.animations:
            blk = self._build_one_anim(anim, all_nodes, node_offsets, is_k2,
                                       current)
            offsets.append(current)
            blocks.append(blk)
            current += len(blk)
        return blocks, offsets

    def _build_one_anim(self, anim, all_nodes, node_offsets, is_k2,
                        anim_block_start_rel: int = 0) -> bytes:
        """
        Build a full animation block including:
          - Geometry header (80 bytes): fp1/fp2/name/root_node_off/node_count
          - Animation model header (88 bytes): length/trans_time/anim_root/events
          - Animation node tree (one stub node per anim node with controllers)

        Animation geometry header (80 bytes, same structure as model geo header):
          +0   fp1       (4)
          +4   fp2       (4)
          +8   name      (32)
          +40  root_node_off (4)   ← from BASE; 0 if no anim nodes
          +44  node_count    (4)
          +48  runtime_arr1  (12)  ← zeros
          +60  runtime_arr2  (12)  ← zeros  [actually: ref_count(4) + geo_type(1) + pad(3)]
          +72  ref_count (4)
          +76  geo_type  (1) = 5 (animation geometry header)
          +77  padding   (3)

        Animation model header (88 bytes):
          +0   length         (float32)
          +4   transition_time(float32)
          +8   anim_root_name (32 bytes null-terminated)
          +40  events_off     (uint32)  — offset to events array from BASE
          +44  events_cnt     (uint32)
          +48  events_cnt2    (uint32)
          +52  mdx_size       (uint32)  — always 0 for anims
          +56  mdx_off        (uint32)  — always 0 for anims
          +60  names_array    (12 bytes: off/cnt/cnt2)
          +72  unk            (16 bytes padding to reach 88)
        """
        fp1 = FP1_K2 if is_k2 else FP1_K1
        fp2 = FP2_K2 if is_k2 else FP2_K1

        # Collect anim nodes (if any) and rebuild their hierarchy from the
        # geometry model. Retarget/bake tools keep animation nodes flat in
        # memory, but binary MDL animation blocks must be a reachable tree.
        anim_nodes = self._animation_nodes_with_hierarchy(anim, all_nodes)
        self._validate_animation_export_tree(anim, anim_nodes, all_nodes)

        # ── Build anim node tree stubs ────────────────────────────────────────
        # Each anim node is a base node header (80 bytes) + controller data.
        # Anim nodes are positioned right after the 168-byte anim header block.
        anim_hdr_size = 168  # 80 (geo) + 88 (model)
        node_block_start_rel = anim_block_start_rel + anim_hdr_size

        anim_node_blocks: List[bytes] = []
        anim_node_offsets: Dict[int, int] = {}   # id(n) → rel offset from BASE
        current_off = node_block_start_rel

        for an in anim_nodes:
            off = current_off
            anim_node_offsets[id(an)] = off
            blk = self._build_anim_node(an, anim_nodes, is_k2, off)
            anim_node_blocks.append(blk)
            current_off += len(blk)

        # Fix parent/root/child offsets in anim node blocks
        root_anim_off = anim_node_offsets.get(id(anim_nodes[0]), 0) if anim_nodes else 0
        fixed_anim_nodes: List[bytes] = []
        for i, an in enumerate(anim_nodes):
            blk = bytearray(anim_node_blocks[i])
            # +8: root off → offset of first anim node
            struct.pack_into('<I', blk, 8, root_anim_off)
            # +12: parent off
            if an.parent is not None:
                par_off = anim_node_offsets.get(id(an.parent), 0)
            else:
                par_off = 0
            struct.pack_into('<I', blk, 12, par_off)
            # Fix child ptrs at byte 80
            for ci, child in enumerate(an.children or []):
                co = anim_node_offsets.get(id(child), 0)
                struct.pack_into('<I', blk, 80 + ci * 4, co)
            fixed_anim_nodes.append(bytes(blk))

        anim_node_data = b''.join(fixed_anim_nodes)

        # ── Anim geometry header (80 bytes) ──────────────────────────────────
        geo = bytearray(80)
        struct.pack_into('<I', geo, 0, fp1)
        struct.pack_into('<I', geo, 4, fp2)
        nm = (anim.name or "").encode('ascii', 'replace')[:32].ljust(32, b'\x00')
        geo[8:40] = nm
        struct.pack_into('<I', geo, 40, root_anim_off)       # root_node_off
        struct.pack_into('<I', geo, 44, len(anim_nodes))     # node_count
        geo[76] = 5                                          # geometry_type = 5 (animation)

        # ── Animation model header (88 bytes) ────────────────────────────────
        anim_hdr = bytearray(88)
        struct.pack_into('<f', anim_hdr, 0, float(anim.length or 0.0))
        transition_time = getattr(anim, 'transition_time', None)
        struct.pack_into('<f', anim_hdr, 4, float(0.25 if transition_time is None else transition_time))
        root_name = getattr(anim_nodes[0], 'name', '') if anim_nodes else (anim.anim_root or "")
        nm2 = (anim.anim_root or root_name or "").encode('ascii', 'replace')[:32].ljust(32, b'\x00')
        anim_hdr[8:40] = nm2
        events = getattr(anim, "events", ()) or ()
        event_data = bytearray()
        for event in events:
            name = event.name.encode("ascii")
            if len(name) > 32:
                raise ValueError(f"Animation event name exceeds 32 bytes: {event.name!r}")
            event_data.extend(struct.pack("<f32s", event.time, name))
        event_offset = node_block_start_rel + len(anim_node_data) if events else 0
        struct.pack_into("<III", anim_hdr, 40, event_offset, len(events), len(events))
        return bytes(geo) + bytes(anim_hdr) + anim_node_data + bytes(event_data)

    def _animation_nodes_with_hierarchy(self, anim, all_nodes: list) -> list:
        """Overlay keyed tracks on geometry cloned in the name-table's DFS order.

        Retail geometry may reuse a mesh name on different limbs. Flat retarget
        tracks still resolve by unique name; repeated names need their ancestry.
        """
        def key(node):
            return str(getattr(node, "name", "") or "").lower()

        def address(node):
            parts = []
            while node is not None:
                parent = getattr(node, "parent", None)
                ordinal = 0
                if parent is not None:
                    for sibling in parent.children:
                        if sibling is node:
                            break
                        if key(sibling) == key(node):
                            ordinal += 1
                parts.append((key(node), ordinal))
                node = parent
            return tuple(reversed(parts))

        if not all_nodes:
            return []
        clones = {}
        by_name = {}
        by_path = {}
        for target in all_nodes:
            clone = target.clone_shallow() if hasattr(target, "clone_shallow") else copy.copy(target)
            clone.children = []
            clone.parent = None
            clone.controllers = []
            clones[id(target)] = clone
            by_name.setdefault(key(target), []).append(clone)
            by_path[address(target)] = clone
        for target in all_nodes:
            clone = clones[id(target)]
            clone.parent = clones.get(id(target.parent))
            clone.children = [clones[id(child)] for child in target.children]
        for track in getattr(anim, "nodes", ()) or ():
            matches = by_name.get(key(track), [])
            if not matches:
                continue
            match = matches[0] if len(matches) == 1 else by_path.get(address(track))
            if match is None and getattr(track, "controllers", None):
                raise ValueError(f"Ambiguous animation track {track.name!r} in {anim.name!r}")
            if match is not None:
                match.controllers = list(getattr(track, "controllers", ()) or ())
        return [clones[id(target)] for target in all_nodes]

    @staticmethod
    def _validate_animation_export_tree(anim, anim_nodes: list, target_nodes: list) -> None:
        """Fail early if the binary animation tree no longer mirrors target nodes."""

        if not target_nodes:
            return
        expected_names = [str(getattr(node, 'name', '') or '').lower() for node in target_nodes]
        actual_names = [str(getattr(node, 'name', '') or '').lower() for node in anim_nodes]
        if actual_names != expected_names:
            raise ValueError(
                f"Animation '{getattr(anim, 'name', '')}' export tree has {len(anim_nodes)} nodes "
                f"but target model hierarchy has {len(target_nodes)} nodes; binary Aurora "
                "animations must preserve the full target hierarchy, including unkeyed placeholders."
            )

        actual_by_name = {
            str(getattr(node, 'name', '') or '').lower(): node
            for node in anim_nodes
            if getattr(node, 'name', '')
        }
        for target, actual in zip(target_nodes, anim_nodes):
            expected_parent = (
                str(getattr(target.parent, 'name', '') or '').lower()
                if getattr(target, 'parent', None) is not None
                else None
            )
            actual_parent = (
                str(getattr(actual.parent, 'name', '') or '').lower()
                if getattr(actual, 'parent', None) is not None
                else None
            )
            if actual_parent != expected_parent:
                raise ValueError(
                    f"Animation '{getattr(anim, 'name', '')}' export tree parent mismatch for "
                    f"node '{getattr(actual, 'name', '')}': expected {expected_parent!r}, got {actual_parent!r}."
                )

            expected_children = [
                str(getattr(child, 'name', '') or '').lower()
                for child in getattr(target, 'children', []) or []
            ]
            actual_children = [
                str(getattr(child, 'name', '') or '').lower()
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

    def _build_anim_node(self, node, all_anim_nodes: list,
                         is_k2: bool, node_off_rel: int) -> bytes:
        """
        Build a minimal animation node block.
        Animation nodes only have the base 80-byte header + controller data.
        They do NOT have mesh sub-headers.
        """
        node_idx = (all_anim_nodes.index(node)
                    if node in all_anim_nodes else 0)
        child_cnt = len(node.children or [])

        ctrl_entries, ctrl_data_floats = self._serialize_controllers(node)
        ctrl_cnt      = len(ctrl_entries)
        ctrl_data_cnt = len(ctrl_data_floats)

        child_arr_off_rel  = node_off_rel + 80
        ctrl_off_rel       = child_arr_off_rel + 4 * child_cnt
        ctrl_data_off_rel  = ctrl_off_rel + 16 * ctrl_cnt

        hdr = bytearray(80)
        # Animation nodes are controller stubs only. They must always be DUMMY
        # nodes, even when they animate a mesh/skin geometry node with the same
        # name. If mesh flags leak in here, readers expect a mesh sub-header
        # after this 80-byte base node and the following controller data is
        # misread as geometry, which mangles the exported model on reimport.
        anim_flags = 0x0001
        struct.pack_into('<H', hdr, 0, anim_flags)
        struct.pack_into('<H', hdr, 2, node_idx & 0xFFFF)    # node_number
        struct.pack_into('<H', hdr, 4, node_idx & 0xFFFF)    # name_index
        # +8, +12: off_root / off_parent — fixed up by caller
        pos = node.position or (0.0, 0.0, 0.0)
        struct.pack_into('<fff', hdr, 16, *pos)
        rot = node.rotation or (0.0, 0.0, 0.0, 1.0)
        x, y, z, w = rot
        struct.pack_into('<ffff', hdr, 28, w, x, y, z)
        struct.pack_into('<I', hdr, 44, child_arr_off_rel)
        struct.pack_into('<I', hdr, 48, child_cnt)
        struct.pack_into('<I', hdr, 52, child_cnt)
        struct.pack_into('<I', hdr, 56, ctrl_off_rel)
        struct.pack_into('<I', hdr, 60, ctrl_cnt)
        struct.pack_into('<I', hdr, 64, ctrl_cnt)
        struct.pack_into('<I', hdr, 68, ctrl_data_off_rel)
        struct.pack_into('<I', hdr, 72, ctrl_data_cnt)
        struct.pack_into('<I', hdr, 76, ctrl_data_cnt)

        child_ptrs = bytearray(4 * child_cnt)
        ctrl_blk   = bytearray()
        for e in ctrl_entries:
            ctrl_blk += e
        ctrl_data_blk = bytearray()
        for fv in ctrl_data_floats:
            ctrl_data_blk += struct.pack('<f', fv)

        return bytes(hdr) + bytes(child_ptrs) + bytes(ctrl_blk) + bytes(ctrl_data_blk)


# ─────────────────────────────────────────────────────────────────────────────
#  One-Step Porter convenience function
# ─────────────────────────────────────────────────────────────────────────────

def port_model_file(input_mdl: str, output_mdl: str,
                    target_game: str = 'K2',
                    texture_map: Optional[Dict[str,str]] = None,
                    input_mdx: str = "",
                    output_mdx: str = "") -> str:
    """
    One-step K1↔K2 porting:
      1. Parse input binary MDL/MDX
      2. Port (swap magic, supermodel, textures) to target_game
      3. Write output binary MDL/MDX

    Returns a human-readable report string.

    This replaces the old workflow:
      MDLEdit (binary→ASCII) → edit → MDLEdit (ASCII→binary)
    """
    from ..game.kotor_loader import load_model_from_file

    # Parse via PyKotor
    model = load_model_from_file(input_mdl, input_mdx)

    src_game = model.game_version.name  # 'K1' or 'K2'

    # Port
    porter  = CrossGamePorter(texture_map=texture_map or {})
    ported  = porter.port(model, target_game)

    # Write
    writer  = MDLBinaryWriter()
    writer.write(ported, output_mdl, output_mdx)

    report = (
        f"Ported {src_game}→{target_game}: {model.name!r}\n"
        f"  Input:  {input_mdl}\n"
        f"  Output: {output_mdl}\n"
        f"  Nodes:  {len(list(_iter_all(ported.root_node)))}\n"
        f"  Anims:  {len(ported.animations)}\n"
        f"  Supermodel: {model.supermodel!r} → {ported.supermodel!r}\n"
    )
    return report


def _iter_all(node):
    if node is None: return
    yield node
    for c in (node.children or []):
        yield from _iter_all(c)
