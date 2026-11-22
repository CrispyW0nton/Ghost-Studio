"""GhostRigger-owned binary MDL reader.

This module keeps GhostRigger's K2 binary fixes behind ``read_mdl_safe``
without mutating PyKotor's global module state:

* K2 trimesh headers use the KotOR.js/KotorBlender dirt+hologram tail layout.
* ``mdx_data_offset == 0`` is treated as a valid MDX base offset.

The implementation deliberately reuses PyKotor's conversion flow where possible
and only owns the binary parsing points GhostRigger must correct.
"""

from __future__ import annotations

import math
from types import FunctionType
from typing import Any, Dict

from pykotor.common.misc import Game
from pykotor.common.stream import BinaryReader
from pykotor.resource.formats.mdl import io_mdl as _iom


K1_TRIMESH_SIZE = 332
K2_TRIMESH_SIZE = 340

_ACTIVE_READERS: Dict[int, "GhostRiggerMDLBinaryReader"] = {}


def _logical_reader_size(reader: BinaryReader) -> int:
    """Return the largest logical MDL offset readable from this BinaryReader."""
    return max(0, reader.size() - int(getattr(reader, "_offset", 0) or 0))


def _array_count_within_reader(offset: int, count: int, stride: int, reader: BinaryReader) -> int:
    if count <= 0 or offset in (0, 0xFFFFFFFF) or offset < 0 or stride <= 0:
        return 0
    logical_size = _logical_reader_size(reader)
    if offset >= logical_size:
        return 0
    return min(count, (logical_size - offset) // stride)


class GhostRiggerTrimeshHeader(_iom._TrimeshHeader):  # type: ignore[attr-defined]
    """Trimesh header reader with GhostRigger's corrected K2 tail layout."""

    def read(self, reader: BinaryReader, game: Game) -> "GhostRiggerTrimeshHeader":
        start_pos = reader.position()
        self.function_pointer0 = reader.read_uint32()
        self.function_pointer1 = reader.read_uint32()
        self.offset_to_faces = reader.read_uint32()
        faces_count_raw = reader.read_uint32()
        if faces_count_raw > 0x7FFFFFFF:
            faces_count_raw = 0x7FFFFFFF
        self.faces_count = faces_count_raw
        faces_count2_raw = reader.read_uint32()
        if faces_count2_raw > 0x7FFFFFFF:
            faces_count2_raw = 0x7FFFFFFF
        self.faces_count2 = faces_count2_raw
        self.bounding_box_min = reader.read_vector3()
        self.bounding_box_max = reader.read_vector3()
        self.radius = reader.read_single()
        self.average = reader.read_vector3()
        self.diffuse = reader.read_vector3()
        self.ambient = reader.read_vector3()
        self.transparency_hint = reader.read_uint32()
        self.texture1 = reader.read_terminated_string("\0", 32)
        self.texture2 = reader.read_terminated_string("\0", 32)
        self.unknown0 = reader.read_bytes(24)
        self.offset_to_indices_counts = reader.read_uint32()
        indices_counts_count_raw = reader.read_uint32()
        if indices_counts_count_raw > 0x7FFFFFFF:
            indices_counts_count_raw = 0x7FFFFFFF
        self.indices_counts_count = indices_counts_count_raw
        indices_counts_count2_raw = reader.read_uint32()
        if indices_counts_count2_raw > 0x7FFFFFFF:
            indices_counts_count2_raw = 0x7FFFFFFF
        self.indices_counts_count2 = indices_counts_count2_raw
        self.offset_to_indices_offset = reader.read_uint32()
        indices_offsets_count_raw = reader.read_uint32()
        if indices_offsets_count_raw > 0x7FFFFFFF:
            indices_offsets_count_raw = 0x7FFFFFFF
        self.indices_offsets_count = indices_offsets_count_raw
        indices_offsets_count2_raw = reader.read_uint32()
        if indices_offsets_count2_raw > 0x7FFFFFFF:
            indices_offsets_count2_raw = 0x7FFFFFFF
        self.indices_offsets_count2 = indices_offsets_count2_raw
        self.offset_to_counters = reader.read_uint32()
        counters_count_raw = reader.read_uint32()
        if counters_count_raw > 0x7FFFFFFF:
            counters_count_raw = 0x7FFFFFFF
        self.counters_count = counters_count_raw
        counters_count2_raw = reader.read_uint32()
        if counters_count2_raw > 0x7FFFFFFF:
            counters_count2_raw = 0x7FFFFFFF
        self.counters_count2 = counters_count2_raw
        self.unknown1 = reader.read_bytes(12)
        self.saber_unknowns = reader.read_bytes(8)
        self.unknown2 = reader.read_int32()
        self.uv_direction = reader.read_vector2()
        self.uv_jitter = reader.read_single()
        self.uv_speed = reader.read_single()

        def _read_i32_as_u32() -> int:
            value = reader.read_int32()
            return 0xFFFFFFFF if value < 0 else value

        self.mdx_data_size = _read_i32_as_u32()
        self.mdx_data_bitmap = _read_i32_as_u32()
        self.mdx_vertex_offset = _read_i32_as_u32()
        self.mdx_normal_offset = _read_i32_as_u32()
        self.mdx_color_offset = _read_i32_as_u32()
        self.mdx_texture1_offset = _read_i32_as_u32()
        self.mdx_texture2_offset = _read_i32_as_u32()
        self.mdx_uv3_offset = _read_i32_as_u32()
        self.mdx_uv4_offset = _read_i32_as_u32()
        self.mdx_tangent_offset = _read_i32_as_u32()
        self.mdx_unknown_offset = _read_i32_as_u32()
        self.mdx_unknown2_offset = _read_i32_as_u32()
        self.mdx_unknown3_offset = _read_i32_as_u32()
        self.vertex_count = reader.read_uint16()
        self.texture_count = reader.read_uint16()
        self.has_lightmap = reader.read_uint8()
        self.rotate_texture = reader.read_uint8()
        self.background = reader.read_uint8()
        self.has_shadow = reader.read_uint8()
        self.beaming = reader.read_uint8()
        self.render = reader.read_uint8()
        if game == Game.K2:
            self.dirt_enabled = reader.read_uint8() != 0
            reader.read_uint8()
            self.dirt_texture = reader.read_int16()
            self.dirt_worldspace = reader.read_int16()
            self.hologram_donotdraw = reader.read_uint8() == 1
            reader.read_uint8()
            self.tail_short = 0
            self.k2_tail_long1 = 0
            self.k2_tail_long2 = 0
            reader.read_bytes(2)
        else:
            self.tail_short = reader.read_uint16()
        self.total_area = reader.read_single()
        self.tail_long0 = reader.read_uint32()
        self.mdx_data_offset = reader.read_uint32()
        self.vertices_offset = reader.read_uint32()
        expected = K1_TRIMESH_SIZE if game == Game.K1 else K2_TRIMESH_SIZE
        reader.seek(start_pos + expected)
        return self

    def read_extra(self, reader: BinaryReader):
        """Read variable-size trimesh arrays using logical MDL bounds."""

        counts = _array_count_within_reader(
            self.offset_to_indices_counts,
            self.indices_counts_count,
            4,
            reader,
        )
        if counts:
            reader.seek(self.offset_to_indices_counts)
            self.indices_counts = [reader.read_uint32() for _ in range(counts)]
            self.indices_counts_count = counts
            self.indices_counts_count2 = min(self.indices_counts_count2, counts)
        else:
            self.indices_counts = []
            self.indices_counts_count = 0
            self.indices_counts_count2 = 0

        offsets = _array_count_within_reader(
            self.offset_to_indices_offset,
            self.indices_offsets_count,
            4,
            reader,
        )
        if offsets:
            reader.seek(self.offset_to_indices_offset)
            self.indices_offsets = [reader.read_uint32() for _ in range(offsets)]
            self.indices_offsets_count = offsets
            self.indices_offsets_count2 = min(self.indices_offsets_count2, offsets)
        else:
            self.indices_offsets = []
            self.indices_offsets_count = 0
            self.indices_offsets_count2 = 0

        counters = _array_count_within_reader(
            self.offset_to_counters,
            self.counters_count,
            4,
            reader,
        )
        if counters:
            reader.seek(self.offset_to_counters)
            self.inverted_counters = [reader.read_uint32() for _ in range(counters)]
            self.counters_count = counters
            self.counters_count2 = min(self.counters_count2, counters)
        else:
            self.inverted_counters = []
            self.counters_count = 0
            self.counters_count2 = 0

        faces = _array_count_within_reader(self.offset_to_faces, self.faces_count, _iom._Face.SIZE, reader)
        if faces:
            reader.seek(self.offset_to_faces)
            self.faces = [_iom._Face().read(reader) for _ in range(faces)]  # type: ignore[attr-defined]
            self.faces_count = faces
            self.faces_count2 = min(self.faces_count2, faces)
            self._sanitize_faces()
        else:
            self.faces = []
            self.faces_count = 0
            self.faces_count2 = 0

        if self.faces:
            max_vertex_index = 0
            for face in self.faces:
                max_vertex_index = max(max_vertex_index, face.vertex1, face.vertex2, face.vertex3)
            self.vertex_count = max(self.vertex_count, max_vertex_index + 1)

        vertices = _array_count_within_reader(self.vertices_offset, self.vertex_count, 12, reader)
        if vertices:
            reader.seek(self.vertices_offset)
            self.vertices = [reader.read_vector3() for _ in range(vertices)]
            self.vertex_count = vertices
        else:
            self.vertices = []
            if self.vertices_offset not in (0, 0xFFFFFFFF):
                self.vertex_count = 0

    def _sanitize_faces(self) -> None:
        for face in self.faces:
            if not math.isfinite(face.plane_coefficient):
                face.plane_coefficient = 0.0
            normal = face.normal
            if not (
                math.isfinite(normal.x)
                and math.isfinite(normal.y)
                and math.isfinite(normal.z)
            ):
                face.normal = _iom.Vector3.from_null()


class GhostRiggerNode(_iom._Node):  # type: ignore[attr-defined]
    """PyKotor binary node with GhostRigger's fixed trimesh header reader."""

    def read(self, reader: BinaryReader, game: Game) -> "GhostRiggerNode":
        start_pos = reader.position()
        self.header = _iom._NodeHeader().read(reader)  # type: ignore[attr-defined]

        if self.header.type_id & _iom.MDLNodeFlags.MESH:
            self.trimesh = GhostRiggerTrimeshHeader().read(reader, game)

        if self.header.type_id & _iom.MDLNodeFlags.SKIN:
            self.skin = _iom._SkinmeshHeader().read(reader)  # type: ignore[attr-defined]

        if self.header.type_id & _iom.MDLNodeFlags.LIGHT:
            self.light = _iom._LightHeader().read(reader)  # type: ignore[attr-defined]
            self._sanitize_light_header(reader)

        if self.header.type_id & _iom.MDLNodeFlags.EMITTER:
            self.emitter = _iom._EmitterHeader().read(reader)  # type: ignore[attr-defined]

        if self.header.type_id & _iom.MDLNodeFlags.REFERENCE:
            self.reference = _iom._ReferenceHeader().read(reader)  # type: ignore[attr-defined]

        if self.header.type_id & _iom.MDLNodeFlags.DANGLY and self.trimesh is not None:
            self.dangly = _iom._DanglymeshHeader().read(reader)  # type: ignore[attr-defined]

        if self.header.type_id & _iom.MDLNodeFlags.AABB and self.trimesh is not None:
            aabb_offset_raw = reader.read_int32()
            self.trimesh.offset_to_aabb = aabb_offset_raw if aabb_offset_raw > 0 else 0

        if self.trimesh is not None:
            self.trimesh.read_extra(reader)
        if self.skin is not None:
            self.skin.read_extra(reader)

        if self.header.children_count == 0:
            self.children_offsets = []
            self._record_for_active_reader(reader, start_pos)
            return self

        child_loc = self.header.offset_to_children
        if (
            child_loc in (0, 0xFFFFFFFF)
            or child_loc >= reader.size()
            or self.header.children_count > 0x7FFFFFFF
            or (self.header.children_count * 4) + child_loc > reader.size()
        ):
            self.header.children_count = 0
            self.header.children_count2 = 0
            self.children_offsets = []
            self._record_for_active_reader(reader, start_pos)
            return self

        try:
            reader.seek(child_loc)
            child_offsets = [reader.read_uint32() for _ in range(self.header.children_count)]
            logical_size = _logical_reader_size(reader)
            self.children_offsets = [
                child_offset
                for child_offset in child_offsets
                if child_offset not in (0, 0xFFFFFFFF)
                and child_offset >= 0
                and child_offset + _iom._NodeHeader.SIZE <= logical_size
            ]
            self.header.children_count = len(self.children_offsets)
            self.header.children_count2 = min(self.header.children_count2, self.header.children_count)
        except Exception:
            self.header.children_count = 0
            self.header.children_count2 = 0
            self.children_offsets = []

        self._record_for_active_reader(reader, start_pos)
        return self

    def _record_for_active_reader(self, reader: BinaryReader, offset: int) -> None:
        owner = _ACTIVE_READERS.get(id(reader))
        if owner is not None:
            owner._gr_bin_nodes[offset] = self
            if self.dangly is not None:
                # PyKotor's delegate seeks to offset + 12 even though this
                # reader is already based at the 12-byte MDL prefix. Let the
                # GhostRigger reader own this array and retain the raw header.
                self._gr_constraint_offset = self.dangly.offset_to_contraints
                self.dangly.offset_to_contraints = 0

    def _sanitize_light_header(self, reader: BinaryReader) -> None:
        light = self.light
        if light is None:
            return

        def _trim(offset_attr: str, count_attr: str, count2_attr: str, stride: int) -> None:
            count = _array_count_within_reader(getattr(light, offset_attr), getattr(light, count_attr), stride, reader)
            setattr(light, count_attr, count)
            setattr(light, count2_attr, min(getattr(light, count2_attr), count))

        _trim("offset_to_flare_sizes", "flare_sizes_count", "flare_sizes_count2", 4)
        _trim("offset_to_flare_positions", "flare_positions_count", "flare_positions_count2", 4)
        _trim("offset_to_flare_colors", "flare_colors_count", "flare_colors_count2", 12)
        _trim("offset_to_flare_textures", "flare_textures_count", "flare_textures_count2", 4)
        _trim("offset_to_unknown0", "unknown0_count", "unknown0_count2", 4)


_LOAD_NODE_GLOBALS: Dict[str, Any] = dict(_iom.MDLBinaryReader._load_node.__globals__)
_LOAD_NODE_GLOBALS["_Node"] = GhostRiggerNode
_GHOSTRIGGER_LOAD_NODE = FunctionType(
    _iom.MDLBinaryReader._load_node.__code__,
    _LOAD_NODE_GLOBALS,
    name="_load_node",
    argdefs=_iom.MDLBinaryReader._load_node.__defaults__,
    closure=_iom.MDLBinaryReader._load_node.__closure__,
)


class GhostRiggerMDLBinaryReader(_iom.MDLBinaryReader):
    """MDLBinaryReader variant that owns GhostRigger's K2 binary fixes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gr_bin_nodes: Dict[int, GhostRiggerNode] = {}
        self._gr_node_order_seen: set[int] = set()

    def _load_names(self, model_header) -> None:
        # Surgical appends can place strings apart from the pointer table.
        reader = self._reader
        limit = _logical_reader_size(reader)
        offset = model_header.offset_to_name_offsets
        count = model_header.name_offsets_count
        if count > 65535 or offset < 0 or offset + count * 4 > limit:
            raise ValueError("Invalid MDL name pointer array")
        reader.seek(offset)
        pointers = [reader.read_uint32() for _ in range(count)]
        self._names = []
        for pointer in pointers:
            if pointer >= limit:
                raise ValueError("Invalid MDL name pointer")
            reader.seek(pointer)
            self._names.append(reader.read_terminated_string("\0", min(256, limit - pointer)))

    def _get_node_order(self, startnode: int) -> None:
        """Collect node order using the reader's logical MDL offset space.

        PyKotor's current implementation adds the binary MDL 12-byte prefix
        again even though ``MDLBinaryReader`` has already installed that prefix
        as the ``BinaryReader`` base offset.  Some stock supermodels, including
        K1 ``S_Male02``, then read controller metadata as child offsets and walk
        out of bounds before GhostRigger's safer node reader can recover.
        """
        if startnode in (0, 0xFFFFFFFF) or startnode in self._gr_node_order_seen:
            return
        if startnode < 0 or startnode + _iom._NodeHeader.SIZE > self._reader.size():
            return

        self._gr_node_order_seen.add(startnode)

        try:
            self._reader.seek(startnode + 4)
            name_index: int = self._reader.read_uint16()
            self._order2nameindex.append(name_index)

            self._reader.seek(startnode + 44)
            child_array_offset: int = self._reader.read_uint32()
            child_array_length: int = self._reader.read_uint32()
        except OSError:
            return

        if child_array_length <= 0 or child_array_offset in (0, 0xFFFFFFFF):
            return
        if child_array_length > 0x7FFFFFFF:
            return
        if child_array_offset < 0:
            return
        if child_array_offset + (child_array_length * 4) > self._reader.size():
            return

        try:
            self._reader.seek(child_array_offset)
            child_offsets = [
                self._reader.read_uint32()
                for _ in range(child_array_length)
            ]
        except OSError:
            return

        for child_offset in child_offsets:
            self._get_node_order(child_offset)

    def _load_node(self, offset: int, parent):
        previous = _ACTIVE_READERS.get(id(self._reader))
        _ACTIVE_READERS[id(self._reader)] = self
        try:
            node = _GHOSTRIGGER_LOAD_NODE(self, offset, parent)
        finally:
            if previous is None:
                _ACTIVE_READERS.pop(id(self._reader), None)
            else:
                _ACTIVE_READERS[id(self._reader)] = previous

        bin_node = self._gr_bin_nodes.get(offset)
        if bin_node is not None:
            self._preserve_raw_node_flags(node, bin_node)
            self._preserve_raw_node_payload(node, bin_node)
            self._restore_dangly_constraints(node, bin_node)
            self._fill_mdx_offset_zero_vertices(node, bin_node)
            # BinaryReader already carries the 12-byte MDL base. PyKotor adds it a
            # second time for dangly constraints, shifting them by three vertices.
            # Re-read them through the same bounded logical-offset reader.
            if bin_node.dangly is not None and node.dangly is not None:
                from pykotor.resource.formats.mdl.mdl_data import MDLConstraint

                header = bin_node.dangly
                count = int(header.constraints_count)
                offset = int(header.offset_to_contraints)
                if count and _array_count_within_reader(offset, count, 4, self._reader) != count:
                    raise ValueError("Dangly constraint array outside MDL")
                saved = self._reader.position()
                try:
                    self._reader.seek(offset)
                    constraints = []
                    for _ in range(count):
                        value = self._reader.read_single()
                        if not math.isfinite(value) or not 0.0 <= value <= 255.0:
                            raise ValueError("Invalid dangly constraint weight")
                        constraint = MDLConstraint()
                        constraint.type = int(value * 1000000)
                        constraints.append(constraint)
                    node.dangly.constraints = constraints
                finally:
                    self._reader.seek(saved)
        return node

    def _restore_dangly_constraints(self, node, bin_node: GhostRiggerNode) -> None:
        header = bin_node.dangly
        if header is None:
            return
        header.offset_to_contraints = bin_node._gr_constraint_offset
        count = header.constraints_count
        if not count or node.dangly is None:
            return
        reader = self._reader
        offset = header.offset_to_contraints
        if _array_count_within_reader(offset, count, 4, reader) != count:
            raise ValueError("Invalid MDL dangly constraint array")
        saved_position = reader.position()
        try:
            reader.seek(offset)
            weights = tuple(reader.read_single() for _ in range(count))
        finally:
            reader.seek(saved_position)
        from pykotor.resource.formats.mdl.mdl_data import MDLConstraint

        constraints = []
        for weight in weights:
            if not math.isfinite(weight):
                raise ValueError("Non-finite MDL dangly constraint weight")
            constraint = MDLConstraint()
            constraint.type = int(weight * 1_000_000)
            constraints.append(constraint)
        node.dangly.constraints = constraints
        node._gr_dangly_weights = weights

    def _load_controller(self, offset: int, data_offset: int):
        """Load a controller and preserve its original binary entry metadata."""

        raw_metadata = self._read_raw_controller_metadata(offset, data_offset)
        controller = super()._load_controller(offset, data_offset)
        if raw_metadata:
            setattr(controller, "_gr_binary_controller", raw_metadata)
        return controller

    def _read_raw_controller_metadata(self, offset: int, data_offset: int) -> Dict[str, Any]:
        saved_pos = self._reader.position()
        try:
            if offset in (0, 0xFFFFFFFF) or offset < 0:
                return {}
            if offset + _iom._Controller.SIZE > self._reader.size():
                return {}
            self._reader.seek(offset)
            raw = _iom._Controller().read(self._reader)  # type: ignore[attr-defined]
            metadata: Dict[str, Any] = {
                # Absolute file offset of the first value, for in-place controller edits.
                "binary_file_value_offset": int(getattr(self._reader, "_offset", 0)) + int(data_offset) + int(raw.data_offset) * 4,
                "type": int(raw.type_id),
                "unknown0": int(raw.unknown0),
                "row_count": int(raw.row_count),
                "key_offset": int(raw.key_offset),
                "data_offset": int(raw.data_offset),
                "column_count": int(raw.column_count),
                "unknown1": list(bytes(raw.unknown1 or b"\x00\x00\x00")[:3].ljust(3, b"\x00")),
            }
            column_count = int(raw.column_count)
            if int(raw.type_id) == int(_iom.MDLControllerType.ORIENTATION) and column_count == 2:
                words = []
                value_pos = int(data_offset) + int(raw.data_offset) * 4
                if 0 <= value_pos < self._reader.size():
                    self._reader.seek(value_pos)
                    for _ in range(int(raw.row_count)):
                        if self._reader.position() + 4 > self._reader.size():
                            break
                        words.append(int(self._reader.read_uint32()))
                metadata["compressed_quaternion_words"] = words
            elif column_count & 0x10:
                # Aurora stores a Bezier controller row as three floats for
                # every logical component: value, incoming tangent, outgoing
                # tangent.  PyKotor exposes those expanded rows, but the
                # GhostRigger domain conversion historically kept only the
                # first logical values.  Capture the raw float32 rows here so
                # a load -> model -> write round trip can preserve both the
                # 0x10 flag and every tangent without depending on a decoded
                # interpolation representation.
                base_columns = column_count & 0x0F
                values_per_row = base_columns * 3
                rows: list[list[float]] = []
                value_pos = int(data_offset) + int(raw.data_offset) * 4
                bytes_per_row = values_per_row * 4
                if values_per_row > 0 and 0 <= value_pos < self._reader.size():
                    self._reader.seek(value_pos)
                    for _ in range(int(raw.row_count)):
                        if self._reader.position() + bytes_per_row > self._reader.size():
                            break
                        rows.append([
                            float(self._reader.read_single())
                            for _ in range(values_per_row)
                        ])
                metadata["bezier_rows"] = rows
            return metadata
        except Exception:
            return {}
        finally:
            try:
                self._reader.seek(saved_pos)
            except Exception:
                pass

    def _preserve_raw_node_flags(self, node, bin_node: GhostRiggerNode) -> None:
        """Restore semantic node types that PyKotor collapses during conversion."""

        if int(bin_node.header.type_id) & int(_iom.MDLNodeFlags.SABER):
            node.node_type = _iom.MDLNodeType.SABER

    @staticmethod
    def _preserve_raw_node_payload(node, bin_node: GhostRiggerNode) -> None:
        """Retain binary-only sub-header bytes omitted by PyKotor's model API."""

        emitter = getattr(node, "emitter", None)
        raw_emitter = getattr(bin_node, "emitter", None)
        if emitter is not None and raw_emitter is not None:
            setattr(
                emitter,
                "_gr_binary_emitter",
                {"unknown1": int(getattr(raw_emitter, "unknown1", 0) or 0) & 0xFF},
            )

    def _fill_mdx_offset_zero_vertices(self, node, bin_node: GhostRiggerNode) -> None:
        trimesh = bin_node.trimesh
        mesh = getattr(node, "mesh", None)
        if trimesh is None or mesh is None or self._reader_ext is None:
            return
        if trimesh.mdx_data_offset != 0 or trimesh.mdx_data_size <= 0 or trimesh.vertex_count <= 0:
            return
        if not bool(trimesh.mdx_data_bitmap & _iom._MDXDataFlags.VERTEX):
            return
        if self._reader_ext.size() <= 0:
            return

        vertex_offset = 0 if trimesh.mdx_vertex_offset == 0xFFFFFFFF else trimesh.mdx_vertex_offset
        positions = []
        saved_pos = self._reader_ext.position()
        try:
            for index in range(trimesh.vertex_count):
                seek_pos = index * trimesh.mdx_data_size + vertex_offset
                if seek_pos + 12 > self._reader_ext.size():
                    return
                self._reader_ext.seek(seek_pos)
                positions.append(
                    _iom.Vector3(
                        self._reader_ext.read_single(),
                        self._reader_ext.read_single(),
                        self._reader_ext.read_single(),
                    )
                )
        finally:
            self._reader_ext.seek(saved_pos)

        mesh.vertex_positions = positions
