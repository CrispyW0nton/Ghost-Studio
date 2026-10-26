"""FBX export through Autodesk Python FBX SDK."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Callable

from src.core.geometry.model_data import KotorModel, ModelNode, NodeFlags
from src.core.scene.kmax_scene import KMaxScene
from src.io.export_control import check_export_cancelled
from src.math.transform_math import rotate_vector

from .fbx_scene_adapter import (
    FbxExportSummary,
    euler_degrees_to_quat_xyz,
    gr_material_to_fbx_material,
    gr_mesh_to_fbx_mesh,
    iter_renderable_mesh_nodes,
    quat_xyzw_to_euler_degrees,
    sdk_class,
)
from .fbx_sdk_loader import configure_fbx_sdk_paths, get_fbx_modules, get_fbx_sdk_status

log = logging.getLogger(__name__)


class FbxExportError(RuntimeError):
    """Raised when FBX export fails."""


class FbxSdkUnavailableError(RuntimeError):
    """Raised when Autodesk FBX SDK bindings are unavailable."""


DEFAULT_EXPORT_OPTIONS: dict[str, Any] = {
    "export_selection_only": False,
    "export_ascii": False,
    "embed_media": False,
    "apply_unit_conversion": True,
    "bake_transforms": False,
    "triangulate": True,
    "merge_selection": False,
}


def export_fbx(scene_or_selection: Any, path: str, options: dict[str, Any] | None = None) -> bool:
    """Export a GhostRigger model/scene/selection to FBX using Autodesk SDK."""
    opts = {**DEFAULT_EXPORT_OPTIONS, **(options or {})}
    target = _validate_output_path(path)
    configure_fbx_sdk_paths(opts.get("fbx_sdk"), refresh=True)
    modules = get_fbx_modules()
    if modules.fbx is None:
        raise FbxSdkUnavailableError(get_fbx_sdk_status())
    fbx = modules.fbx
    manager = None
    exporter = None
    try:
        manager, scene = _create_sdk_scene(fbx, modules.FbxCommon, target.stem or "GhostRiggerScene")
        _configure_scene(fbx, manager, scene, opts)
        summary = FbxExportSummary()
        _add_payload_to_scene(fbx, manager, scene, scene_or_selection, opts, summary)
        exporter_cls = sdk_class(fbx, "FbxExporter", "KFbxExporter")
        exporter = exporter_cls.Create(manager, "")
        file_format = _choose_writer_format(manager, opts)
        if not exporter.Initialize(str(target), file_format, manager.GetIOSettings()):
            raise FbxExportError(_sdk_error(exporter, f"Could not initialize FBX exporter for {target}"))
        if not exporter.Export(scene):
            raise FbxExportError(_sdk_error(exporter, f"Could not export FBX scene to {target}"))
        log.info("FBX export summary for %s: %s", target.name, summary.log_line())
        return True
    except FbxSdkUnavailableError:
        raise
    except FbxExportError:
        raise
    except Exception as exc:
        raise FbxExportError(f"FBX export failed for {target}: {exc}") from exc
    finally:
        if exporter is not None:
            _destroy(exporter)
        if manager is not None:
            _destroy(manager)


def _validate_output_path(path: str) -> Path:
    target = Path(path)
    if target.suffix.lower() != ".fbx":
        target = target.with_suffix(".fbx")
    if target.parent:
        target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _create_sdk_scene(fbx: Any, common: Any, name: str):
    if common is not None and hasattr(common, "InitializeSdkObjects"):
        manager, scene = common.InitializeSdkObjects()
        scene.SetName(name)
        return manager, scene
    manager_cls = sdk_class(fbx, "FbxManager", "KFbxSdkManager")
    scene_cls = sdk_class(fbx, "FbxScene", "KFbxScene")
    ios_cls = sdk_class(fbx, "FbxIOSettings", "KFbxIOSettings")
    manager = manager_cls.Create()
    if ios_cls is not None and hasattr(fbx, "IOSROOT"):
        manager.SetIOSettings(ios_cls.Create(manager, fbx.IOSROOT))
    return manager, scene_cls.Create(manager, name)


def _configure_scene(fbx: Any, manager: Any, scene: Any, options: dict[str, Any]) -> None:
    info_cls = sdk_class(fbx, "FbxDocumentInfo", "KFbxDocumentInfo")
    if info_cls is not None:
        info = info_cls.Create(manager, "GhostRiggerExportInfo")
        info.mTitle = scene.GetName()
        info.mSubject = "GhostRigger FBX export"
        scene.SetSceneInfo(info)
    try:
        axis_cls = sdk_class(fbx, "FbxAxisSystem", "KFbxAxisSystem")
        axis = axis_cls(axis_cls.eZAxis, axis_cls.eParityOdd, axis_cls.eRightHanded)
        scene.GetGlobalSettings().SetAxisSystem(axis)
    except Exception:
        log.debug("Could not set FBX axis system", exc_info=True)
    try:
        units_cls = sdk_class(fbx, "FbxSystemUnit", "KFbxSystemUnit")
        scene.GetGlobalSettings().SetSystemUnit(units_cls.cm)
    except Exception:
        log.debug("Could not set FBX system unit", exc_info=True)
    _set_io_bool(fbx, manager, "EXP_FBX_MATERIAL", True)
    _set_io_bool(fbx, manager, "EXP_FBX_TEXTURE", True)
    _set_io_bool(fbx, manager, "EXP_FBX_EMBEDDED", bool(options.get("embed_media")))


def _add_payload_to_scene(fbx: Any, manager: Any, scene: Any, payload: Any, options: dict[str, Any], summary: FbxExportSummary) -> None:
    root = scene.GetRootNode()
    for model, prefix in _models_from_payload(payload, options):
        parent = _new_node(fbx, manager, prefix or getattr(model, "name", "Model"))
        root.AddChild(parent)
        for node in iter_renderable_mesh_nodes(model):
            fbx_node = _create_mesh_node(fbx, manager, node, options, summary)
            parent.AddChild(fbx_node)
        if getattr(model, "animations", None):
            warning = "Phase 2 TODO: animation export is not implemented in Autodesk SDK bridge."
            summary.warnings.append(warning)
            log.warning(warning)


def _models_from_payload(payload: Any, options: dict[str, Any]) -> list[tuple[KotorModel, str]]:
    if isinstance(payload, KotorModel):
        return [(payload, payload.name)]
    if isinstance(payload, KMaxScene):
        objects = list(payload.objects)
        if options.get("export_selection_only"):
            objects = [obj for obj in objects if getattr(obj, "selected", False)]
        if options.get("merge_selection"):
            name = "selection" if options.get("export_selection_only") else (payload.name or "scene")
            return [(merge_selected_scene_objects(objects, name=name), name)]
        result = []
        for obj in objects:
            model = (getattr(obj, "metadata", {}) or {}).get("_runtime_model")
            if isinstance(model, KotorModel):
                transformed = _copy_model_with_instance_transform(model, obj)
                result.append((transformed, getattr(obj, "name", model.name)))
        if not result:
            raise FbxExportError("Scene export has no runtime mesh objects to export.")
        return result
    if isinstance(payload, (list, tuple)):
        if options.get("merge_selection"):
            return [(merge_selected_scene_objects(payload, name="selection"), "selection")]
        result = []
        for item in payload:
            if isinstance(item, KotorModel):
                result.append((item, item.name))
            else:
                model = (getattr(item, "metadata", {}) or {}).get("_runtime_model")
                if isinstance(model, KotorModel):
                    result.append((_copy_model_with_instance_transform(model, item), getattr(item, "name", model.name)))
        if result:
            return result
    raise FbxExportError("FBX export expects a KotorModel, KMaxScene, or selected scene objects with runtime models.")


def merge_selected_scene_objects(
    scene_objects: Any,
    *,
    name: str = "selection",
    is_cancelled=None,
    node_filter: Callable[[ModelNode], bool] | None = None,
) -> KotorModel:
    """Bake selected static scene objects into one FBX-ready mesh.

    Vertices are expanded per polygon corner so meshes with independent
    texture-vertex indices retain their UV seams after becoming one FBX mesh.
    An optional node filter can replace the FBX renderability filter for
    callers that need another exporter's mesh-selection semantics.
    """
    items = list(scene_objects or [])
    merged = ModelNode(
        name=str(name or "selection"),
        flags=int(NodeFlags.HEADER | NodeFlags.MESH),
        vertex_space=1,
    )
    material_slots: list[ModelNode] = []
    material_indices: dict[tuple[Any, ...], int] = {}
    first_model: KotorModel | None = None

    for item in items:
        check_export_cancelled(is_cancelled)
        if isinstance(item, KotorModel):
            model = item
            transform = None
        else:
            model = (getattr(item, "metadata", {}) or {}).get("_runtime_model")
            transform = getattr(item, "transform", None)
        if not isinstance(model, KotorModel):
            continue
        if first_model is None:
            first_model = model

        scene_position = _vec3(getattr(transform, "position", None), (0.0, 0.0, 0.0))
        scene_rotation = euler_degrees_to_quat_xyz(
            _vec3(getattr(transform, "rotation", None), (0.0, 0.0, 0.0))
        )
        scene_scale = _vec3(getattr(transform, "scale", None), (1.0, 1.0, 1.0))

        nodes = (
            model.mesh_nodes()
            if node_filter is not None
            else iter_renderable_mesh_nodes(model)
        )
        for node in nodes:
            check_export_cancelled(is_cancelled)
            if node_filter is not None and not node_filter(node):
                continue
            if not getattr(node, "faces", None):
                continue
            if int(getattr(node, "vertex_space", 0) or 0) == 2:
                continue
            node_slots = _material_slots_for_node(node)
            node_material_indices = []
            for slot in node_slots:
                key = _material_key(slot)
                material_index = material_indices.get(key)
                if material_index is None:
                    material_index = len(material_slots)
                    material_indices[key] = material_index
                    material_slots.append(slot)
                node_material_indices.append(material_index)
            world_position, world_rotation = node.world_transform()
            vertices_are_world = bool(
                int(getattr(node, "vertex_space", 0) or 0) == 1
                or getattr(node, "_gr_vertices_in_kotor_world", False)
            )

            for face_index, face in enumerate(node.faces):
                if face_index % 256 == 0:
                    check_export_cancelled(is_cancelled)
                if len(face) != 3 or any(int(index) < 0 or int(index) >= len(node.vertices) for index in face):
                    continue
                output_face = []
                for corner, raw_vertex_index in enumerate(face):
                    vertex_index = int(raw_vertex_index)
                    point = tuple(float(v) for v in node.vertices[vertex_index][:3])
                    normal = (
                        tuple(float(v) for v in node.normals[vertex_index][:3])
                        if vertex_index < len(node.normals)
                        else (0.0, 0.0, 1.0)
                    )
                    if not vertices_are_world:
                        point = _add3(rotate_vector(world_rotation, point), world_position)
                        normal = rotate_vector(world_rotation, normal)
                    point = _apply_scene_point(point, scene_position, scene_rotation, scene_scale)
                    normal = _apply_scene_normal(normal, scene_rotation, scene_scale)

                    uv_index = vertex_index
                    if face_index < len(node.face_uvs) and corner < len(node.face_uvs[face_index]):
                        candidate = int(node.face_uvs[face_index][corner])
                        if 0 <= candidate < len(node.uvs):
                            uv_index = candidate
                    uv = node.uvs[uv_index] if 0 <= uv_index < len(node.uvs) else (0.0, 0.0)
                    uv_lm = node.uvs_lm[vertex_index] if vertex_index < len(node.uvs_lm) else (0.0, 0.0)

                    output_face.append(len(merged.vertices))
                    merged.vertices.append(point)
                    merged.normals.append(normal)
                    merged.uvs.append(tuple(float(v) for v in uv[:2]))
                    merged.uvs_lm.append(tuple(float(v) for v in uv_lm[:2]))

                merged.faces.append(tuple(output_face))
                source_material = (
                    int(node.face_mats[face_index])
                    if _uses_face_material_slots(node) and face_index < len(node.face_mats)
                    else 0
                )
                source_material = max(0, min(source_material, len(node_slots) - 1))
                merged.face_mats.append(node_material_indices[source_material])

    check_export_cancelled(is_cancelled)
    if first_model is None or not merged.faces:
        raise FbxExportError("Selected scene objects contain no runtime mesh geometry to merge.")

    merged._gr_fbx_material_slots = material_slots
    merged.texture_names = [slot.texture for slot in material_slots]
    merged.tex_count = len(material_slots)
    merged.texture = merged.texture_names[0] if merged.texture_names else ""
    result = KotorModel(
        name=str(name or "selection"),
        supermodel="NULL",
        classification=first_model.classification,
        game_version=first_model.game_version,
        model_type=first_model.model_type,
        root_node=merged,
    )
    result.compute_bounds()
    return result


def _material_slots_for_node(node: ModelNode) -> list[ModelNode]:
    names = [str(getattr(node, "texture", "") or "")]
    if _uses_face_material_slots(node):
        names = [str(value or "") for value in (getattr(node, "texture_names", None) or names)]
    slots = []
    for index, texture_name in enumerate(names):
        slot = copy.copy(node)
        slot.name = f"{node.name}_material_{index + 1}"
        slot.texture = texture_name
        slot.texture_names = [texture_name] if texture_name else []
        slot.tex_count = 1
        slots.append(slot)
    return slots


def _uses_face_material_slots(node: ModelNode) -> bool:
    return bool(
        getattr(node, "imported_ascii", False)
        and not getattr(node, "has_lightmap", False)
        and len(getattr(node, "texture_names", None) or []) > 1
        and getattr(node, "face_mats", None)
    )


def _material_key(node: ModelNode) -> tuple[Any, ...]:
    return (
        str(getattr(node, "texture_clean", "") or getattr(node, "texture", "") or "").lower(),
        tuple(float(value) for value in getattr(node, "diffuse", (0.8, 0.8, 0.8))[:3]),
        tuple(float(value) for value in getattr(node, "ambient", (0.2, 0.2, 0.2))[:3]),
        tuple(float(value) for value in getattr(node, "specular", (0.0, 0.0, 0.0))[:3]),
        float(getattr(node, "shininess", 0.0)),
        float(getattr(node, "alpha", 1.0)),
    )


def _vec3(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    try:
        values = tuple(value)
        return (float(values[0]), float(values[1]), float(values[2]))
    except (TypeError, ValueError, IndexError):
        return default


def _add3(a: Any, b: Any) -> tuple[float, float, float]:
    return (float(a[0]) + float(b[0]), float(a[1]) + float(b[1]), float(a[2]) + float(b[2]))


def _apply_scene_point(
    point: Any,
    position: tuple[float, float, float],
    rotation: tuple[float, float, float, float],
    scale: tuple[float, float, float],
) -> tuple[float, float, float]:
    scaled = tuple(float(point[index]) * scale[index] for index in range(3))
    return _add3(rotate_vector(rotation, scaled), position)


def _apply_scene_normal(
    normal: Any,
    rotation: tuple[float, float, float, float],
    scale: tuple[float, float, float],
) -> tuple[float, float, float]:
    inverse_scaled = tuple(
        float(normal[index]) / scale[index] if abs(scale[index]) > 1e-12 else 0.0
        for index in range(3)
    )
    rotated = rotate_vector(rotation, inverse_scaled)
    length = sum(float(value) * float(value) for value in rotated) ** 0.5
    if length <= 1e-12:
        return (0.0, 0.0, 1.0)
    return tuple(float(value) / length for value in rotated)


def _copy_model_with_instance_transform(model: KotorModel, instance: Any) -> KotorModel:
    clone = copy.deepcopy(model)
    transform = getattr(instance, "transform", None)
    if clone.root_node is not None and transform is not None:
        clone.root_node.position = tuple(float(v) for v in getattr(transform, "position", (0.0, 0.0, 0.0))[:3])
        # KMAX stores Euler degrees, while KotorModel nodes use quaternions. Keep
        # scale as Phase 2 unless bake_transforms becomes necessary for KMAX.
    return clone


def _create_mesh_node(fbx: Any, manager: Any, node: ModelNode, options: dict[str, Any], summary: FbxExportSummary) -> Any:
    fbx_node = _new_node(fbx, manager, node.name)
    fbx_mesh = gr_mesh_to_fbx_mesh(fbx, manager, node, triangulate=bool(options.get("triangulate", True)))
    fbx_node.SetNodeAttribute(fbx_mesh)
    _set_node_transform(fbx, fbx_node, node)
    material_slots = list(getattr(node, "_gr_fbx_material_slots", None) or [node])
    for material_node in material_slots:
        material = gr_material_to_fbx_material(fbx, manager, material_node)
        if material is not None:
            fbx_node.AddMaterial(material)
            summary.materials += 1
            if getattr(material_node, "texture", ""):
                summary.textures += 1
    summary.meshes += 1
    summary.vertices += len(node.vertices)
    summary.faces += len(node.faces)
    return fbx_node


def _new_node(fbx: Any, manager: Any, name: str) -> Any:
    node_cls = sdk_class(fbx, "FbxNode", "KFbxNode")
    return node_cls.Create(manager, str(name or "Node")[:64])


def _set_node_transform(fbx: Any, fbx_node: Any, node: ModelNode) -> None:
    double3_cls = sdk_class(fbx, "FbxDouble3", "KFbxDouble3")
    try:
        fbx_node.LclTranslation.Set(double3_cls(*tuple(float(v) for v in node.position[:3])))
    except Exception:
        pass
    try:
        fbx_node.LclRotation.Set(double3_cls(*quat_xyzw_to_euler_degrees(node.rotation)))
    except Exception:
        pass


def _choose_writer_format(manager: Any, options: dict[str, Any]) -> int:
    if not options.get("export_ascii"):
        return -1
    try:
        registry = manager.GetIOPluginRegistry()
        count = registry.GetWriterFormatCount()
        for index in range(count):
            desc = str(registry.GetWriterFormatDescription(index)).lower()
            if "fbx ascii" in desc:
                return index
    except Exception:
        pass
    return -1


def _set_io_bool(fbx: Any, manager: Any, prop_name: str, value: bool) -> None:
    try:
        ios = manager.GetIOSettings()
        prop = getattr(fbx, prop_name, prop_name)
        ios.SetBoolProp(prop, bool(value))
    except Exception:
        pass


def _sdk_error(sdk_object: Any, fallback: str) -> str:
    try:
        status = sdk_object.GetStatus()
        error = status.GetErrorString()
        return f"{fallback}: {error}" if error else fallback
    except Exception:
        return fallback


def _destroy(obj: Any) -> None:
    destroy = getattr(obj, "Destroy", None)
    if callable(destroy):
        destroy()
