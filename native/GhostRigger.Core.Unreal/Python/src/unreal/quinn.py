"""SKM_Quinn_Simple target skeleton support."""

from __future__ import annotations

import array
from dataclasses import dataclass
import math
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET
import zlib

from src.core.geometry.model_data import (
    BoneWeight,
    KotorModel,
    ModelClassification,
    ModelNode,
    NodeFlags,
    VertexSkinData,
)


def _resolve_skels_dir() -> Path:
    """Locate the bundled SKM_Quinn_Simple assets.

    After the native split the skeleton files ship in the host runtime payload
    (native/GhostRigger.Native.Core.Host/RuntimePayload/src/unreal/skels), not
    beside this package copy, so a module-relative ``skels`` directory is empty
    in source checkouts and the Quinn target silently had no bones.
    """

    local = Path(__file__).resolve().parent / "skels"
    candidates = [local]
    for ancestor in Path(__file__).resolve().parents:
        if (ancestor / "native" / "GhostRigger.Native.Core.Host").is_dir():
            candidates.append(
                ancestor / "native" / "GhostRigger.Native.Core.Host" / "RuntimePayload" / "src" / "unreal" / "skels"
            )
            break
    for candidate in candidates:
        if (candidate / "SKM_Quinn_Simple_BoneMap.xml").is_file():
            return candidate
    return local


SKELS_DIR = _resolve_skels_dir()
QUINN_BONE_MAP = SKELS_DIR / "SKM_Quinn_Simple_BoneMap.xml"
QUINN_FBX = SKELS_DIR / "SKM_Quinn_Simple.FBX"


@dataclass(frozen=True)
class UnrealBone:
    index: int
    name: str
    side: str = ""
    group: str = ""
    role: str = ""


@dataclass(frozen=True)
class UnrealSkeletonAsset:
    name: str
    source: str
    bone_map_path: Path
    fbx_path: Path
    texture_paths: tuple[Path, ...]
    bones: tuple[UnrealBone, ...]

    @property
    def bone_count(self) -> int:
        return len(self.bones)


@dataclass
class _FbxNode:
    name: str
    props: list
    children: list["_FbxNode"]

    def child(self, name: str) -> "_FbxNode | None":
        for item in self.children:
            if item.name == name:
                return item
        return None

    def children_named(self, name: str) -> list["_FbxNode"]:
        return [item for item in self.children if item.name == name]


def _fbx_clean_name(value: object) -> str:
    return str(value or "").split("\x00", 1)[0]


def _fbx_child_value(node: _FbxNode, name: str, default=None):
    child = node.child(name)
    if child is None or not child.props:
        return default
    return child.props[0]


def _fbx_property70(node: _FbxNode, name: str) -> list:
    props = node.child("Properties70")
    if props is None:
        return []
    for prop in props.children_named("P"):
        if prop.props and prop.props[0] == name:
            return list(prop.props[4:])
    return []


def _quat_mul_xyzw(a, b) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _axis_angle_quat(axis: str, degrees: float) -> tuple[float, float, float, float]:
    half = math.radians(float(degrees)) * 0.5
    s = math.sin(half)
    c = math.cos(half)
    if axis == "x":
        return (s, 0.0, 0.0, c)
    if axis == "y":
        return (0.0, s, 0.0, c)
    return (0.0, 0.0, s, c)


def _euler_xyz_to_quat(values: list) -> tuple[float, float, float, float]:
    if len(values) < 3:
        return (0.0, 0.0, 0.0, 1.0)
    qx = _axis_angle_quat("x", float(values[0]))
    qy = _axis_angle_quat("y", float(values[1]))
    qz = _axis_angle_quat("z", float(values[2]))
    q = _quat_mul_xyzw(_quat_mul_xyzw(qx, qy), qz)
    length = math.sqrt(sum(part * part for part in q))
    if length <= 1e-8:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(float(part / length) for part in q)


def _yaw_180_point(point) -> tuple[float, float, float]:
    return (-float(point[0]), -float(point[1]), float(point[2]))


def _fbx_model_lookup(objects: _FbxNode) -> tuple[dict[int, _FbxNode], dict[str, int]]:
    by_id: dict[int, _FbxNode] = {}
    id_by_name: dict[str, int] = {}
    for item in objects.children:
        if item.name != "Model" or not item.props:
            continue
        object_id = int(item.props[0])
        by_id[object_id] = item
        if len(item.props) > 1:
            id_by_name[_fbx_clean_name(item.props[1]).lower()] = object_id
    return by_id, id_by_name


def _fbx_parent_map(connections: _FbxNode, model_ids: set[int] | None = None) -> dict[int, int]:
    parents: dict[int, int] = {}
    for conn in connections.children_named("C"):
        props = conn.props
        if len(props) >= 3 and str(props[0]) == "OO":
            child_id = int(props[1])
            parent_id = int(props[2])
            if model_ids is not None and (child_id not in model_ids or parent_id not in model_ids):
                continue
            parents[child_id] = parent_id
    return parents


def _fbx_children_map(connections: _FbxNode) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for conn in connections.children_named("C"):
        props = conn.props
        if len(props) >= 3 and str(props[0]) == "OO":
            child_id = int(props[1])
            parent_id = int(props[2])
            children.setdefault(parent_id, []).append(child_id)
    return children


def _infer_unreal_bone_side(name: str) -> str:
    lower = name.lower()
    if lower.endswith("_l") or lower.endswith("_left"):
        return "left"
    if lower.endswith("_r") or lower.endswith("_right"):
        return "right"
    return "center"


def _infer_unreal_bone_group(name: str) -> str:
    lower = name.lower()
    if lower.startswith("ik_") or lower in {"interaction", "center_of_mass"}:
        return "control"
    if any(part in lower for part in ("thumb", "index", "middle", "ring", "pinky", "metacarpal")):
        return "hand"
    if any(part in lower for part in ("clavicle", "upperarm", "lowerarm", "hand")):
        return "arm"
    if any(part in lower for part in ("thigh", "calf", "foot", "ball")):
        return "leg"
    if any(part in lower for part in ("pelvis", "spine", "neck", "head", "root")):
        return "body"
    return "skeleton"


def _infer_unreal_bone_role(name: str, fbx_kind: str) -> str:
    lower = name.lower()
    if lower.startswith("ik_"):
        return "ik_control"
    if lower in {"interaction", "center_of_mass"}:
        return "control"
    if "twist" in lower:
        return "twist"
    return "root" if fbx_kind.lower() == "root" else "deform"


def _fbx_skeleton_bones(path: str | Path) -> tuple[UnrealBone, ...]:
    roots = _read_binary_fbx(path)
    objects = next((node for node in roots if node.name == "Objects"), None)
    if objects is None:
        return ()
    bones: list[UnrealBone] = []
    for item in objects.children:
        if item.name != "Model" or len(item.props) < 3:
            continue
        fbx_kind = str(item.props[2] or "")
        if fbx_kind not in {"Root", "LimbNode"}:
            continue
        bone_name = _fbx_clean_name(item.props[1])
        if not bone_name:
            continue
        bones.append(
            UnrealBone(
                index=len(bones),
                name=bone_name,
                side=_infer_unreal_bone_side(bone_name),
                group=_infer_unreal_bone_group(bone_name),
                role=_infer_unreal_bone_role(bone_name, fbx_kind),
            )
        )
    return tuple(bones)


def _build_fbx_skeleton_model(
    asset: UnrealSkeletonAsset,
    objects: _FbxNode,
    connections: _FbxNode,
    *,
    yaw_180: bool,
) -> KotorModel:
    """Build Quinn target bones from FBX hierarchy and bind-pose positions."""
    model_nodes, id_by_name = _fbx_model_lookup(objects)
    parents = _fbx_parent_map(connections, set(model_nodes))
    bind_positions: dict[int, tuple[float, float, float]] = {}
    for pose in objects.children_named("Pose"):
        if len(pose.props) >= 3 and str(pose.props[2]) != "BindPose":
            continue
        for pose_node in pose.children_named("PoseNode"):
            node_ref = pose_node.child("Node")
            matrix_node = pose_node.child("Matrix")
            if node_ref is None or matrix_node is None or not node_ref.props or not matrix_node.props:
                continue
            matrix = matrix_node.props[0]
            if isinstance(matrix, list) and len(matrix) >= 15:
                try:
                    position = (
                        float(matrix[12]) * 0.01,
                        float(matrix[13]) * 0.01,
                        float(matrix[14]) * 0.01,
                    )
                    bind_positions[int(node_ref.props[0])] = _yaw_180_point(position) if yaw_180 else position
                except Exception:
                    pass

    model = KotorModel()
    model.name = asset.name
    model.supermodel = "NULL"
    model.classification = "character"
    model.model_type = int(ModelClassification.CHARACTER)
    root = ModelNode(name=asset.name, flags=int(NodeFlags.HEADER))
    model.root_node = root

    nodes_by_id: dict[int, ModelNode] = {}
    for bone in asset.bones:
        object_id = id_by_name.get(bone.name.lower())
        fbx_model = model_nodes.get(object_id) if object_id is not None else None
        pos_values = _fbx_property70(fbx_model, "Lcl Translation") if fbx_model else []
        rot_values = _fbx_property70(fbx_model, "Lcl Rotation") if fbx_model else []
        node = ModelNode(name=bone.name, flags=int(NodeFlags.HEADER), index=bone.index)
        if len(pos_values) >= 3:
            position = (
                float(pos_values[0]) * 0.01,
                float(pos_values[1]) * 0.01,
                float(pos_values[2]) * 0.01,
            )
            node.position = _yaw_180_point(position) if yaw_180 else position
        node.rotation = (0.0, 0.0, 0.0, 1.0) if object_id in bind_positions else _euler_xyz_to_quat(rot_values)
        if object_id is not None:
            nodes_by_id[object_id] = node

    for object_id, node in nodes_by_id.items():
        parent_id = parents.get(object_id)
        parent = nodes_by_id.get(parent_id)
        if parent is None:
            parent = root
        if object_id in bind_positions:
            world = bind_positions[object_id]
            if parent_id in bind_positions:
                node.position = (
                    world[0] - bind_positions[parent_id][0],
                    world[1] - bind_positions[parent_id][1],
                    world[2] - bind_positions[parent_id][2],
                )
            else:
                node.position = world
        node.parent = parent
        parent.children.append(node)

    for bone in asset.bones:
        if model.find_node(bone.name) is None:
            node = ModelNode(name=bone.name, flags=int(NodeFlags.HEADER), index=bone.index)
            node._missing_fbx_transform = True
            node._hide_skeleton_overlay = True
            node.parent = root
            root.children.append(node)

    return model


def _read_fbx_property(data: bytes, pos: int) -> tuple[object, int]:
    code = chr(data[pos])
    pos += 1
    scalar_types = {
        "Y": ("<h", 2),
        "C": ("<?", 1),
        "I": ("<i", 4),
        "F": ("<f", 4),
        "D": ("<d", 8),
        "L": ("<q", 8),
    }
    if code in scalar_types:
        fmt, size = scalar_types[code]
        return struct.unpack_from(fmt, data, pos)[0], pos + size
    if code in {"S", "R", "A"}:
        length = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        raw = data[pos:pos + length]
        pos += length
        if code == "S":
            return raw.decode("utf-8", "replace"), pos
        return raw, pos
    if code in {"f", "d", "i", "I", "l", "b"}:
        count, encoding, byte_count = struct.unpack_from("<III", data, pos)
        pos += 12
        raw = data[pos:pos + byte_count]
        pos += byte_count
        if encoding:
            raw = zlib.decompress(raw)
        typecode = {
            "f": "f",
            "d": "d",
            "i": "i",
            "I": "I",
            "l": "q",
            "b": "b",
        }[code]
        values = array.array(typecode)
        values.frombytes(raw)
        if sys.byteorder != "little":
            values.byteswap()
        if len(values) != count:
            values = values[:count]
        return values.tolist(), pos
    raise ValueError(f"Unsupported FBX property type {code!r}")


def _read_fbx_node(data: bytes, pos: int, limit: int, wide_offsets: bool) -> tuple[_FbxNode | None, int]:
    header_size = 25 if wide_offsets else 13
    if pos + header_size > limit:
        return None, pos
    if wide_offsets:
        end_offset, prop_count, _prop_len = struct.unpack_from("<QQQ", data, pos)
        pos += 24
    else:
        end_offset, prop_count, _prop_len = struct.unpack_from("<III", data, pos)
        pos += 12
    name_len = data[pos]
    pos += 1
    name = data[pos:pos + name_len].decode("utf-8", "replace")
    pos += name_len
    if end_offset == 0 and prop_count == 0 and _prop_len == 0 and name_len == 0:
        return None, pos

    props = []
    for _ in range(prop_count):
        value, pos = _read_fbx_property(data, pos)
        props.append(value)

    children: list[_FbxNode] = []
    while pos < end_offset - header_size:
        child, pos = _read_fbx_node(data, pos, end_offset, wide_offsets)
        if child is None:
            break
        children.append(child)
    return _FbxNode(name=name, props=props, children=children), int(end_offset)


def _read_binary_fbx(path: str | Path) -> list[_FbxNode]:
    data = Path(path).read_bytes()
    if not data.startswith(b"Kaydara FBX Binary  \x00\x1a\x00"):
        raise ValueError("Only binary FBX files are supported for viewport import")
    version = struct.unpack_from("<I", data, 23)[0]
    wide_offsets = version >= 7500
    pos = 27
    roots: list[_FbxNode] = []
    header_size = 25 if wide_offsets else 13
    while pos < len(data) - header_size:
        node, pos = _read_fbx_node(data, pos, len(data), wide_offsets)
        if node is None:
            break
        roots.append(node)
    return roots


def _fbx_object_texture_names(objects: _FbxNode, connections: _FbxNode) -> list[str]:
    material_ids: list[int] = []
    texture_files: dict[int, str] = {}
    diffuse_texture_for_material: dict[int, int] = {}

    for item in objects.children:
        if not item.props:
            continue
        object_id = int(item.props[0])
        if item.name == "Material":
            material_ids.append(object_id)
        elif item.name == "Texture":
            rel = _fbx_child_value(item, "RelativeFilename", "") or _fbx_child_value(item, "FileName", "")
            if rel:
                texture_files[object_id] = Path(str(rel)).stem

    for conn in connections.children_named("C"):
        props = conn.props
        if len(props) < 3:
            continue
        kind = str(props[0])
        child_id = int(props[1])
        parent_id = int(props[2])
        relation = str(props[3]) if len(props) > 3 else ""
        if kind == "OP" and relation == "DiffuseColor":
            diffuse_texture_for_material[parent_id] = child_id

    names: list[str] = []
    for mat_id in material_ids:
        texture_id = diffuse_texture_for_material.get(mat_id)
        names.append(texture_files.get(texture_id, f"material_{len(names)}"))
    return names or ["NULL"]


def _fbx_geometry_to_mesh_node(
    geometry: _FbxNode,
    texture_names: list[str],
    scale: float,
    flip_v: bool,
    yaw_180: bool,
) -> ModelNode:
    raw_vertices = _fbx_child_value(geometry, "Vertices", []) or []
    raw_indices = _fbx_child_value(geometry, "PolygonVertexIndex", []) or []
    if not raw_vertices or not raw_indices:
        raise ValueError("FBX geometry does not contain mesh vertex/index arrays")

    vertices = []
    for i in range(0, len(raw_vertices) - 2, 3):
        vertex = (
            float(raw_vertices[i]) * scale,
            float(raw_vertices[i + 1]) * scale,
            float(raw_vertices[i + 2]) * scale,
        )
        vertices.append(_yaw_180_point(vertex) if yaw_180 else vertex)

    normals: list[tuple[float, float, float]] = []
    normal_layer = geometry.child("LayerElementNormal")
    if normal_layer is not None:
        raw_normals = _fbx_child_value(normal_layer, "Normals", []) or []
        if len(raw_normals) // 3 == len(vertices):
            normals = []
            for i in range(0, len(raw_normals) - 2, 3):
                normal = (
                    float(raw_normals[i]),
                    float(raw_normals[i + 1]),
                    float(raw_normals[i + 2]),
                )
                normals.append(_yaw_180_point(normal) if yaw_180 else normal)

    uvs: list[tuple[float, float]] = []
    uv_layer = geometry.child("LayerElementUV")
    if uv_layer is not None:
        raw_uvs = _fbx_child_value(uv_layer, "UV", []) or []
        if len(raw_uvs) // 2 == len(vertices):
            for i in range(0, len(raw_uvs) - 1, 2):
                u = float(raw_uvs[i])
                v = float(raw_uvs[i + 1])
                uvs.append((u, 1.0 - v if flip_v else v))

    polygon_mats: list[int] = []
    material_layer = geometry.child("LayerElementMaterial")
    if material_layer is not None:
        polygon_mats = [int(value) for value in (_fbx_child_value(material_layer, "Materials", []) or [])]

    faces: list[tuple[int, int, int]] = []
    face_mats: list[int] = []
    polygon: list[int] = []
    polygon_index = 0
    for raw_index in raw_indices:
        raw_index = int(raw_index)
        end_polygon = raw_index < 0
        vertex_index = -raw_index - 1 if end_polygon else raw_index
        polygon.append(vertex_index)
        if not end_polygon:
            continue
        if len(polygon) >= 3:
            mat_index = polygon_mats[polygon_index] if polygon_index < len(polygon_mats) else 0
            mat_index = max(0, min(mat_index, len(texture_names) - 1))
            for i in range(1, len(polygon) - 1):
                faces.append((polygon[0], polygon[i], polygon[i + 1]))
                face_mats.append(mat_index)
        polygon = []
        polygon_index += 1

    name = _fbx_clean_name(geometry.props[1] if len(geometry.props) > 1 else "SKM_Quinn_Simple")
    mesh = ModelNode(name=f"{name}_mesh", flags=int(NodeFlags.MESH))
    mesh.vertices = vertices
    mesh.normals = normals
    mesh.uvs = uvs
    mesh.faces = faces
    mesh.face_mats = face_mats
    mesh.texture_names = texture_names
    mesh.tex_count = len(texture_names)
    mesh.texture = texture_names[0] if texture_names else "NULL"
    mesh.render = True
    mesh.vertex_space = 1
    mesh._imported = True
    mesh.compute_bounds()
    return mesh


def _fbx_apply_skinning(
    mesh: ModelNode,
    geometry: _FbxNode,
    objects: _FbxNode,
    connections: _FbxNode,
) -> None:
    """Populate ``mesh.bone_map`` and per-control-point skin weights from FBX clusters."""
    if not geometry.props:
        return
    object_by_id = {
        int(item.props[0]): item
        for item in objects.children
        if item.props
    }
    children_by_parent = _fbx_children_map(connections)
    model_name_by_id = {
        int(item.props[0]): _fbx_clean_name(item.props[1])
        for item in objects.children
        if item.name == "Model" and len(item.props) > 1
    }
    geometry_id = int(geometry.props[0])
    skin_ids = [
        child_id
        for child_id in children_by_parent.get(geometry_id, [])
        if (
            object_by_id.get(child_id) is not None
            and object_by_id[child_id].name == "Deformer"
            and len(object_by_id[child_id].props) >= 3
            and str(object_by_id[child_id].props[2]).lower() == "skin"
        )
    ]
    if not skin_ids:
        return

    bone_map: list[str] = []
    influences_by_vertex: list[list[tuple[int, float]]] = [[] for _ in mesh.vertices]
    for skin_id in skin_ids:
        for cluster_id in children_by_parent.get(skin_id, []):
            cluster = object_by_id.get(cluster_id)
            if (
                cluster is None
                or cluster.name != "Deformer"
                or len(cluster.props) < 3
                or str(cluster.props[2]).lower() != "cluster"
            ):
                continue
            bone_name = ""
            for child_id in children_by_parent.get(cluster_id, []):
                if child_id in model_name_by_id:
                    bone_name = model_name_by_id[child_id]
                    break
            if not bone_name:
                continue
            indexes = [int(value) for value in (_fbx_child_value(cluster, "Indexes", []) or [])]
            weights = [float(value) for value in (_fbx_child_value(cluster, "Weights", []) or [])]
            if not indexes or not weights:
                continue
            local_bone_index = len(bone_map)
            bone_map.append(bone_name)
            for vertex_index, weight in zip(indexes, weights):
                if 0 <= vertex_index < len(influences_by_vertex) and weight > 0.0:
                    influences_by_vertex[vertex_index].append((local_bone_index, weight))

    if not bone_map:
        return

    skin_data: list[VertexSkinData] = []
    for vertex_influences in influences_by_vertex:
        top = sorted(vertex_influences, key=lambda item: item[1], reverse=True)[:4]
        data = VertexSkinData(
            influences=[BoneWeight(bone_index=bone_index, weight=weight) for bone_index, weight in top]
        )
        data.normalize()
        skin_data.append(data)

    mesh.flags = int(NodeFlags.MESH | NodeFlags.SKIN)
    mesh.bone_map = bone_map
    mesh.skin_data = skin_data


def load_unreal_bone_map(path: str | Path) -> tuple[str, str, tuple[UnrealBone, ...]]:
    """Read a flat Unreal skeleton bone-map XML file."""
    xml_path = Path(path)
    root = ET.parse(xml_path).getroot()
    name = str(root.attrib.get("name") or xml_path.stem)
    source = str(root.attrib.get("source") or "")
    bones: list[UnrealBone] = []
    for elem in root.findall("./Bones/Bone"):
        try:
            index = int(elem.attrib.get("index", len(bones)))
        except ValueError:
            index = len(bones)
        bone_name = str(elem.attrib.get("name") or "").strip()
        if not bone_name:
            continue
        bones.append(
            UnrealBone(
                index=index,
                name=bone_name,
                side=str(elem.attrib.get("side") or ""),
                group=str(elem.attrib.get("group") or ""),
                role=str(elem.attrib.get("role") or ""),
            )
        )
    bones.sort(key=lambda bone: bone.index)
    return name, source, tuple(bones)


def load_quinn_skeleton_asset(base_dir: str | Path | None = None) -> UnrealSkeletonAsset:
    """Load the bundled SKM_Quinn_Simple target metadata."""
    skel_dir = Path(base_dir) if base_dir is not None else SKELS_DIR
    bone_map = skel_dir / QUINN_BONE_MAP.name
    fbx = skel_dir / QUINN_FBX.name
    fbx_bones = _fbx_skeleton_bones(fbx) if fbx.exists() else ()
    if fbx_bones:
        name, source, bones = "SKM_Quinn_Simple", "SKM_Quinn_Simple.FBX", fbx_bones
    elif bone_map.exists():
        name, source, bones = load_unreal_bone_map(bone_map)
    else:
        name, source, bones = "SKM_Quinn_Simple", "Missing bone map", ()
    textures = tuple(sorted(skel_dir.glob("MI_Quinn_*.png")))
    return UnrealSkeletonAsset(
        name=name,
        source=source,
        bone_map_path=bone_map,
        fbx_path=fbx,
        texture_paths=textures,
        bones=bones,
    )


def unreal_skeleton_model(asset: UnrealSkeletonAsset) -> KotorModel:
    """Represent the Unreal target skeleton as a lightweight model."""
    if asset.fbx_path.exists():
        try:
            roots = _read_binary_fbx(asset.fbx_path)
            objects = next((node for node in roots if node.name == "Objects"), None)
            connections = next((node for node in roots if node.name == "Connections"), None)
            if objects is not None and connections is not None:
                return _build_fbx_skeleton_model(asset, objects, connections, yaw_180=True)
        except Exception:
            pass
    model = KotorModel()
    model.name = asset.name
    model.supermodel = "NULL"
    model.classification = "character"
    model.model_type = int(ModelClassification.CHARACTER)
    root = ModelNode(name=asset.name, flags=int(NodeFlags.HEADER))
    model.root_node = root
    for bone in asset.bones:
        node = ModelNode(name=bone.name, flags=int(NodeFlags.HEADER), index=bone.index)
        node.parent = root
        root.children.append(node)
    return model


def load_quinn_fbx_model(
    asset: UnrealSkeletonAsset | None = None,
    fbx_path: str | Path | None = None,
    *,
    scale: float = 0.01,
    flip_v: bool = True,
    yaw_180: bool = True,
) -> KotorModel:
    """Load SKM_Quinn_Simple's binary FBX mesh into a viewport-ready model.

    FBX UVs arrive in DCC/Unreal convention, while GhostRigger's software and
    GPU viewport samplers address PIL images from the top-left.  ``flip_v``
    keeps Quinn's base-color textures aligned in the workbench.

    Quinn's FBX forward axis is opposite GhostRigger's front-view convention, so
    ``yaw_180`` rotates imported vertices and bind positions around Z.
    """
    target_asset = asset or load_quinn_skeleton_asset()
    path = Path(fbx_path) if fbx_path is not None else target_asset.fbx_path
    roots = _read_binary_fbx(path)
    objects = next((node for node in roots if node.name == "Objects"), None)
    connections = next((node for node in roots if node.name == "Connections"), None)
    if objects is None or connections is None:
        raise ValueError("FBX is missing Objects or Connections sections")

    texture_names = _fbx_object_texture_names(objects, connections)
    geometry_nodes = [
        node
        for node in objects.children
        if node.name == "Geometry" and len(node.props) >= 3 and str(node.props[2]).lower() == "mesh"
    ]
    if not geometry_nodes:
        raise ValueError("FBX does not contain mesh geometry")

    model = _build_fbx_skeleton_model(target_asset, objects, connections, yaw_180=yaw_180)
    model.name = _fbx_clean_name(geometry_nodes[0].props[1] if len(geometry_nodes[0].props) > 1 else target_asset.name)
    model.mdl_path = str(path)
    root = model.root_node
    if root is None:
        root = ModelNode(name=model.name, flags=int(NodeFlags.HEADER))
        model.root_node = root

    for geometry in geometry_nodes:
        mesh = _fbx_geometry_to_mesh_node(geometry, texture_names, scale, flip_v, yaw_180)
        _fbx_apply_skinning(mesh, geometry, objects, connections)
        mesh.parent = root
        root.children.append(mesh)

    model.compute_bounds()
    model.compute_all_tangents()
    return model
