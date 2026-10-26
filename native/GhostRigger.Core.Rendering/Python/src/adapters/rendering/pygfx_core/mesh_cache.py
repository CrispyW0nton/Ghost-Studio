"""Retained pygfx mesh cache keyed by renderer-neutral revisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PygfxMeshRecord:
    mesh_id: int
    source: Any
    geometry_key: tuple[Any, ...]
    material_key: tuple[Any, ...]
    mesh: Any
    geometry: Any
    material: Any
    edge_positions: Any
    gfx: Any
    scene: Any
    base_color: tuple[float, float, float, float]
    source_revision: tuple[Any, ...]
    material_revision: tuple[Any, ...]
    diffuse_map: Any = None
    lightmap_map: Any = None
    double_sided: bool = False
    alpha_mode: str = "OPAQUE"
    alpha_cutoff: float = 0.5
    blend_mode: str = "ALPHA"
    sprite_alpha_source: int = 0
    sprite_glow: float = 0.0
    unlit: bool = False
    is_skinned: bool = False
    skinning_cpu_fallback: bool = False
    skeleton: Any = None
    edge_mesh: Any = None
    edge_material: Any = None
    sprite_proxy_mesh: Any = None
    sprite_proxy_material: Any = None
    world_matrix_key: tuple[float, ...] = ()
    view_style_key: tuple[Any, ...] = ()
    selected: bool = False
    hovered: bool = False
    transform_dirty: bool = False
    geometry_dirty: bool = False
    material_dirty: bool = False
    edge_geometry_dirty: bool = False


class PygfxMeshCache:
    """Owns retained gfx Geometry/Material/Mesh objects for mesh DTOs."""

    def __init__(self) -> None:
        self.records: dict[int, PygfxMeshRecord] = {}
        self.texture_maps: dict[tuple[Any, ...], Any] = {}
        self.geometry_updates_this_frame = 0
        self.dynamic_geometry_updates_this_frame = 0
        self.material_updates_this_frame = 0

    def begin_frame(self) -> None:
        self.geometry_updates_this_frame = 0
        self.dynamic_geometry_updates_this_frame = 0
        self.material_updates_this_frame = 0

    def clear(self) -> None:
        self.records.clear()
        self.texture_maps.clear()
        self.begin_frame()

    def update_texture_regions(self, texture_name: str, image, regions) -> bool:
        """Patch resident pygfx texture pixels without rebuilding scene DTOs."""
        clean = str(texture_name or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
        clean = clean.rsplit(".", 1)[0].lower()
        if not clean or image is None:
            return False
        requested = tuple(regions or ())
        matched = False
        incoming_by_alpha_source: dict[int, Any] = {}
        for key, texture_map in tuple(self.texture_maps.items()):
            texture_id = str(key[0] if isinstance(key, tuple) and key else key).lower()
            if texture_id != clean and not texture_id.startswith(clean + ":"):
                continue
            sprite_alpha_source = int(key[4] or 0) if isinstance(key, tuple) and len(key) > 4 else 0
            incoming = incoming_by_alpha_source.get(sprite_alpha_source)
            if incoming is None:
                incoming = self._texture_pixels(image, sprite_alpha_source=sprite_alpha_source)
                if incoming is None:
                    continue
                incoming_by_alpha_source[sprite_alpha_source] = incoming
            texture = getattr(texture_map, "texture", None)
            target = getattr(texture, "data", None)
            if texture is None or target is None:
                continue
            try:
                target_array = np.asarray(target)
            except Exception:
                continue
            if target_array.shape != incoming.shape or not bool(target_array.flags.writeable):
                continue
            update_range = getattr(texture, "update_range", None)
            update_full = getattr(texture, "update_full", None)
            if not callable(update_range) and not callable(update_full):
                continue
            matched = True
            for region in requested:
                try:
                    x, y, width, height = (int(value) for value in region)
                except (TypeError, ValueError):
                    continue
                x0 = max(0, x)
                y0 = max(0, y)
                x1 = min(int(incoming.shape[1]), x + width)
                y1 = min(int(incoming.shape[0]), y + height)
                if width <= 0 or height <= 0 or x1 <= x0 or y1 <= y0:
                    continue
                np.copyto(
                    target_array[y0:y1, x0:x1, :4],
                    incoming[y0:y1, x0:x1, :4],
                )
                if callable(update_range):
                    update_range((x0, y0, 0), (x1 - x0, y1 - y0, 1))
                else:
                    update_full()
        return matched

    def invalidate_texture(self, texture_name: str, image=None) -> bool:
        """Evict one retained texture map and dirty only its mesh materials."""
        clean = str(texture_name or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
        clean = clean.rsplit(".", 1)[0].lower()
        if not clean:
            return False
        image_suffix = f":{id(image)}" if image is not None else ""

        matching_keys = []
        for key in tuple(self.texture_maps):
            texture_id = str(key[0] if isinstance(key, tuple) and key else key).lower()
            matches_name = texture_id == clean or texture_id.startswith(clean + ":")
            matches_image = not image_suffix or texture_id == clean + image_suffix
            if matches_name and matches_image:
                matching_keys.append(key)
        if not matching_keys and image_suffix:
            matching_keys = [
                key
                for key in tuple(self.texture_maps)
                if str(key[0] if isinstance(key, tuple) and key else key).lower() == clean
                or str(key[0] if isinstance(key, tuple) and key else key).lower().startswith(clean + ":")
            ]
        removed_map_ids = {
            id(self.texture_maps.pop(key))
            for key in matching_keys
            if key in self.texture_maps
        }
        if not removed_map_ids:
            return False
        for record in self.records.values():
            if id(record.diffuse_map) in removed_map_ids or id(record.lightmap_map) in removed_map_ids:
                record.material_dirty = True
                record.view_style_key = ()
        return True

    @staticmethod
    def _view_style_key(
        *,
        show_solid: bool = True,
        show_wireframe: bool = False,
        show_texture: bool = True,
        show_diffuse: bool = True,
        show_lightmap: bool = True,
        render_mode: str = "realistic",
        cull_faces: bool = False,
        xray: bool = False,
        show_mesh_hover: bool = True,
        wire_color: tuple[float, float, float, float] = (0.18, 0.62, 0.95, 1.0),
        hover_color: tuple[float, float, float, float] = (0.0, 215 / 255.0, 181 / 255.0, 0.45),
        selection_color: tuple[float, float, float, float] = (1.0, 210 / 255.0, 63 / 255.0, 1.0),
    ) -> tuple[Any, ...]:
        return (
            bool(show_solid),
            bool(show_wireframe),
            bool(show_texture),
            bool(show_diffuse),
            bool(show_lightmap),
            str(render_mode or "realistic").strip().lower(),
            bool(cull_faces),
            bool(xray),
            bool(show_mesh_hover),
            tuple(round(float(c), 6) for c in wire_color[:4]),
            tuple(round(float(c), 6) for c in hover_color[:4]),
            tuple(round(float(c), 6) for c in selection_color[:4]),
        )

    def apply_view_style(
        self,
        *,
        show_solid: bool = True,
        show_wireframe: bool = False,
        show_texture: bool = True,
        show_diffuse: bool = True,
        show_lightmap: bool = True,
        render_mode: str = "realistic",
        cull_faces: bool = False,
        xray: bool = False,
        show_mesh_hover: bool = True,
        wire_color: tuple[float, float, float, float] = (0.18, 0.62, 0.95, 1.0),
        hover_color: tuple[float, float, float, float] = (0.0, 215 / 255.0, 181 / 255.0, 0.45),
        selection_color: tuple[float, float, float, float] = (1.0, 210 / 255.0, 63 / 255.0, 1.0),
    ) -> None:
        style_key = self._view_style_key(
            show_solid=show_solid,
            show_wireframe=show_wireframe,
            show_texture=show_texture,
            show_diffuse=show_diffuse,
            show_lightmap=show_lightmap,
            render_mode=render_mode,
            cull_faces=cull_faces,
            xray=xray,
            show_mesh_hover=show_mesh_hover,
            wire_color=wire_color,
            hover_color=hover_color,
            selection_color=selection_color,
        )
        for record in self.records.values():
            if record.view_style_key == style_key:
                continue
            record.view_style_key = style_key
            self._apply_material_style(record, style_key)
            self.material_updates_this_frame += 1

    def remove_missing(self, live_mesh_ids: set[int], scene) -> None:
        for mesh_id in list(self.records):
            if mesh_id in live_mesh_ids:
                continue
            record = self.records.pop(mesh_id)
            try:
                scene.remove(record.mesh)
            except Exception:
                pass
            if record.edge_mesh is not None:
                try:
                    scene.remove(record.edge_mesh)
                except Exception:
                    pass

    def get_or_create(
        self,
        mesh_data,
        gfx,
        scene,
        *,
        selected: bool = False,
        hovered: bool = False,
        force_geometry_update: bool = False,
    ) -> PygfxMeshRecord:
        mesh_id = int(mesh_data.mesh_id)
        source_revision = tuple(mesh_data.source_revision or ())
        material_revision = tuple(getattr(mesh_data.material, "source_revision", ()) or ())
        geometry_key = (mesh_id, source_revision)
        material_key = (
            getattr(mesh_data.material, "material_id", ""),
            material_revision,
            bool(selected),
        )
        record = self.records.get(mesh_id)
        if record is None:
            geometry = self._build_geometry(mesh_data, gfx)
            diffuse_map = self._texture_map(mesh_data, gfx, channel=0, lightmap=False)
            lightmap_map = self._texture_map(mesh_data, gfx, channel=1, lightmap=True)
            material = self._build_material(
                mesh_data,
                gfx,
                selected=selected,
                diffuse_map=diffuse_map,
                lightmap_map=lightmap_map,
            )
            skeleton = self._build_skeleton(mesh_data, gfx)
            if skeleton is not None and hasattr(gfx, "SkinnedMesh"):
                mesh = gfx.SkinnedMesh(geometry, material)
                try:
                    mesh.bind(skeleton, bind_matrix=np.eye(4, dtype=np.float32))
                except Exception:
                    skeleton = None
                    mesh = gfx.Mesh(geometry, material)
            else:
                mesh = gfx.Mesh(geometry, material)
            try:
                mesh.name = str(getattr(mesh_data.source, "name", "") or mesh_id)
            except Exception:
                pass
            scene.add(mesh)
            self.geometry_updates_this_frame += 1
            self.material_updates_this_frame += 1
            record = PygfxMeshRecord(
                mesh_id=mesh_id,
                source=getattr(mesh_data, "source", None),
                geometry_key=geometry_key,
                material_key=material_key,
                mesh=mesh,
                geometry=geometry,
                material=material,
                edge_positions=None,
                gfx=gfx,
                scene=scene,
                base_color=self._base_color(mesh_data),
                source_revision=source_revision,
                material_revision=material_revision,
                diffuse_map=diffuse_map,
                lightmap_map=lightmap_map,
                double_sided=bool(getattr(mesh_data.material, "double_sided", False)),
                alpha_mode=str(getattr(mesh_data.material, "alpha_mode", "OPAQUE") or "OPAQUE").upper(),
                alpha_cutoff=float(getattr(mesh_data.material, "alpha_cutoff", 0.5) or 0.5),
                blend_mode=str(getattr(mesh_data.material, "blend_mode", "ALPHA") or "ALPHA").upper(),
                sprite_alpha_source=int(getattr(mesh_data.material, "sprite_alpha_source", 0) or 0),
                sprite_glow=float(getattr(mesh_data.material, "sprite_glow", 0.0) or 0.0),
                unlit=bool(getattr(mesh_data.material, "unlit", False)),
                is_skinned=bool(getattr(mesh_data, "is_skinned", False)),
                skinning_cpu_fallback=bool(getattr(mesh_data, "skinning_cpu_fallback", False)),
                skeleton=skeleton,
                selected=bool(selected),
                hovered=bool(hovered),
                edge_geometry_dirty=True,
            )
            self.records[mesh_id] = record
            return record

        record.source = getattr(mesh_data, "source", record.source)
        dynamic_only = self._can_update_dynamic_skin_buffers(record, source_revision, mesh_data)
        if force_geometry_update and record.geometry_key == geometry_key:
            if not self._update_geometry_buffers(record, mesh_data, dynamic_only=dynamic_only):
                record.geometry_dirty = True
        if record.geometry_dirty or record.geometry_key != geometry_key:
            updated_in_place = self._update_geometry_buffers(record, mesh_data, dynamic_only=dynamic_only)
            if not updated_in_place:
                record.geometry = self._build_geometry(mesh_data, gfx)
                record.mesh.geometry = record.geometry
                if record.edge_mesh is not None:
                    record.edge_positions = self._build_edge_positions(mesh_data)
                    record.edge_mesh.geometry = self._build_edge_geometry(record, gfx)
                self.geometry_updates_this_frame += 1
            else:
                if record.edge_mesh is not None:
                    record.edge_positions = self._build_edge_positions(mesh_data)
                    self._update_edge_geometry(record, gfx)
            record.geometry_key = geometry_key
            record.source_revision = source_revision
            record.is_skinned = bool(getattr(mesh_data, "is_skinned", False))
            record.skinning_cpu_fallback = bool(getattr(mesh_data, "skinning_cpu_fallback", False))
            record.edge_geometry_dirty = record.edge_mesh is not None
            record.geometry_dirty = False
        if record.material_dirty or record.material_key != material_key:
            record.diffuse_map = self._texture_map(mesh_data, gfx, channel=0, lightmap=False)
            record.lightmap_map = self._texture_map(mesh_data, gfx, channel=1, lightmap=True)
            record.material = self._build_material(
                mesh_data,
                gfx,
                selected=selected,
                diffuse_map=record.diffuse_map,
                lightmap_map=record.lightmap_map,
            )
            record.mesh.material = record.material
            record.material_key = material_key
            record.material_revision = material_revision
            record.base_color = self._base_color(mesh_data)
            record.double_sided = bool(getattr(mesh_data.material, "double_sided", False))
            record.alpha_mode = str(getattr(mesh_data.material, "alpha_mode", "OPAQUE") or "OPAQUE").upper()
            record.alpha_cutoff = float(getattr(mesh_data.material, "alpha_cutoff", 0.5) or 0.5)
            record.blend_mode = str(getattr(mesh_data.material, "blend_mode", "ALPHA") or "ALPHA").upper()
            record.sprite_alpha_source = int(getattr(mesh_data.material, "sprite_alpha_source", 0) or 0)
            record.sprite_glow = float(getattr(mesh_data.material, "sprite_glow", 0.0) or 0.0)
            record.unlit = bool(getattr(mesh_data.material, "unlit", False))
            record.is_skinned = bool(getattr(mesh_data, "is_skinned", False))
            record.skinning_cpu_fallback = bool(getattr(mesh_data, "skinning_cpu_fallback", False))
            record.selected = bool(selected)
            record.view_style_key = ()
            record.material_dirty = False
            self.material_updates_this_frame += 1
        elif record.selected != bool(selected):
            self._apply_selection_state(record, bool(selected), bool(hovered))
        elif record.hovered != bool(hovered):
            record.hovered = bool(hovered)
            record.view_style_key = ()
        return record

    @staticmethod
    def _can_update_dynamic_skin_buffers(record: PygfxMeshRecord, source_revision: tuple[Any, ...], mesh_data) -> bool:
        if not bool(getattr(mesh_data, "skinning_cpu_fallback", False)):
            return False
        previous = tuple(getattr(record, "source_revision", ()) or ())
        current = tuple(source_revision or ())
        if len(previous) < 2 or len(current) < 2:
            return False
        return previous[:-1] == current[:-1]

    def mark_transform_dirty(self, node=None) -> None:
        source_id = id(node) if node is not None else None
        for record in self.records.values():
            if source_id is None or id(record.source) == source_id:
                record.transform_dirty = True
                record.world_matrix_key = ()

    def mark_geometry_dirty(self, node) -> None:
        record = self.records.get(id(node))
        if record is not None:
            record.geometry_dirty = True
            record.transform_dirty = True

    def update_selection(self, gfx, selected_source_ids: set[int], hovered_source_id: int | None = None) -> None:
        for record in self.records.values():
            selected = id(record.source) in selected_source_ids
            hovered = hovered_source_id is not None and id(record.source) == hovered_source_id
            if record.selected != selected or record.hovered != hovered:
                self._apply_selection_state(record, selected, hovered)

    def update_visibility(self) -> None:
        for record in self.records.values():
            visible = self._source_visible(record.source)
            try:
                record.mesh.visible = bool(visible)
            except Exception:
                pass
            if record.edge_mesh is not None:
                try:
                    record.edge_mesh.visible = bool(visible and record.view_style_key and record.view_style_key[1])
                except Exception:
                    pass
            if record.sprite_proxy_mesh is not None:
                try:
                    record.sprite_proxy_mesh.visible = bool(visible and self._use_sprite_proxy(record))
                except Exception:
                    pass

    def _build_geometry(self, mesh_data, gfx):
        kwargs = {"positions": mesh_data.positions}
        indices = self._mesh_indices(mesh_data)
        if indices is not None:
            kwargs["indices"] = indices.reshape((-1, 3))
        if mesh_data.normals is not None:
            kwargs["normals"] = mesh_data.normals
        if mesh_data.uvs0 is not None:
            kwargs["texcoords"] = self._pygfx_uvs(mesh_data.uvs0)
        if mesh_data.uvs1 is not None:
            kwargs["texcoords1"] = self._pygfx_uvs(mesh_data.uvs1)
        if bool(getattr(mesh_data, "is_skinned", False)):
            bone_indices = getattr(mesh_data, "bone_indices", None)
            bone_weights = getattr(mesh_data, "bone_weights", None)
            if bone_indices is not None and bone_weights is not None:
                try:
                    kwargs["skin_indices"] = np.asarray(bone_indices, dtype=np.uint32)[:, :4]
                    kwargs["skin_weights"] = np.asarray(bone_weights, dtype=np.float32)[:, :4]
                except Exception:
                    pass
        return gfx.Geometry(**kwargs)

    def _build_skeleton(self, mesh_data, gfx):
        if not bool(getattr(mesh_data, "is_skinned", False)):
            return None
        bone_indices = getattr(mesh_data, "bone_indices", None)
        if bone_indices is None or not hasattr(gfx, "Skeleton") or not hasattr(gfx, "Bone"):
            return None
        try:
            arr = np.asarray(bone_indices, dtype=np.uint32)
            if arr.size == 0:
                return None
            count = max(1, min(128, int(arr.max()) + 1))
            bones = [gfx.Bone(f"bone_{index}") for index in range(count)]
            inverses = [np.eye(4, dtype=np.float32) for _ in range(count)]
            return gfx.Skeleton(bones, bone_inverses=inverses)
        except Exception:
            return None

    def update_skin_palette(self, record: PygfxMeshRecord, anim_pose, model=None, anim_base_pose=None) -> None:
        is_bas_attachment = bool(getattr(getattr(record, "source", None), "_gr_bas_attachment_layer", False))
        if record is None or record.skeleton is None or (anim_pose is None and not is_bas_attachment):
            return
        try:
            from src.core.animation.gpu_skinning import MatrixPaletteUploader, MAX_BONES
            from src.core.rendering.skeleton_render_data import (
                _cached_matrix_palette_uploader,
                bas_attachment_root_local_skin_palette,
            )
            from src.core.rendering.mesh_render_data import (
                animation_pose_applies_to_node,
                animation_pose_for_node,
                bas_attachment_palette_model_for_node,
                runtime_source_model_for_node,
            )

            source_model = runtime_source_model_for_node(record.source) or model
            if bool(getattr(record.source, "_gr_bas_attachment_layer", False)):
                source_model = bas_attachment_palette_model_for_node(record.source) or source_model
            if anim_pose is not None and not animation_pose_applies_to_node(record.source, anim_pose):
                if not bool(getattr(record.source, "_gr_bas_attachment_layer", False)):
                    return
                anim_pose = None
            node_anim_base_pose = animation_pose_for_node(record.source, anim_base_pose) if anim_base_pose is not None else None
            uploader = _cached_matrix_palette_uploader(source_model, MAX_BONES, MatrixPaletteUploader)
            uploader.compute_skin_node_palette(
                record.source,
                anim_pose,
                anim_base_pose=node_anim_base_pose,
            )
            palette = uploader.as_numpy_array()
            palette = bas_attachment_root_local_skin_palette(record.source, palette, anim_pose)
            buffer = getattr(record.skeleton, "bone_matrices_buffer", None)
            data = getattr(buffer, "data", None)
            if palette is None or data is None or len(data) == 0:
                return
            count = min(len(data), len(palette))
            # pygfx SkinnedMesh calls skeleton.update() during its object
            # update. GhostRigger computes the final KotOR palette itself, so
            # make this skeleton direct-buffer driven to avoid pygfx replacing
            # the palette with identity Bone transforms every draw.
            if not bool(getattr(record.skeleton, "_gr_direct_palette_update", False)):
                try:
                    record.skeleton.update = lambda: None
                    record.skeleton._gr_direct_palette_update = True
                except Exception:
                    pass
            try:
                record.skeleton.update()
            except Exception:
                pass
            data[:count]["bone_matrices"] = np.asarray(palette[:count], dtype=np.float32).transpose((0, 2, 1))
            update_range = getattr(buffer, "update_range", None)
            if callable(update_range):
                update_range(0, count)
            else:
                update_full = getattr(buffer, "update_full", None)
                if callable(update_full):
                    update_full()
        except Exception:
            return

    def _build_edge_positions(self, mesh_data):
        positions = np.asarray(getattr(mesh_data, "positions", None), dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] < 3 or len(positions) < 2:
            return None
        indices = getattr(mesh_data, "indices", None)
        geometric = self._uses_sprite_wire_hull(getattr(mesh_data, "material", None))
        edge_indices = self._build_edge_indices(indices, len(positions), positions=positions, geometric=geometric)
        if edge_indices is None or len(edge_indices) < 2:
            return None
        try:
            return np.ascontiguousarray(positions[np.asarray(edge_indices, dtype=np.uint32), :3], dtype=np.float32)
        except Exception:
            return None

    def _update_geometry_buffers(self, record: PygfxMeshRecord, mesh_data, *, dynamic_only: bool = False) -> bool:
        geometry = record.geometry
        checks = [
            ("positions", getattr(mesh_data, "positions", None)),
            ("normals", getattr(mesh_data, "normals", None)),
        ]
        if not dynamic_only:
            checks.extend(
                (
                    ("texcoords", self._pygfx_uvs(getattr(mesh_data, "uvs0", None))),
                    ("texcoords1", self._pygfx_uvs(getattr(mesh_data, "uvs1", None))),
                )
            )
        for attr, values in checks:
            if values is None:
                continue
            buffer = getattr(geometry, attr, None)
            data = getattr(buffer, "data", None)
            if data is None:
                return False
            arr = np.asarray(values, dtype=np.asarray(data).dtype)
            if tuple(arr.shape) != tuple(np.asarray(data).shape):
                return False
        index_values = None if dynamic_only else self._mesh_indices(mesh_data)
        index_buffer = getattr(geometry, "indices", None)
        if index_values is not None and index_buffer is not None and getattr(index_buffer, "data", None) is not None:
            arr = np.asarray(index_values, dtype=np.asarray(index_buffer.data).dtype).reshape(np.asarray(index_buffer.data).shape)
            if tuple(arr.shape) != tuple(np.asarray(index_buffer.data).shape):
                return False
        try:
            for attr, values in checks:
                if values is None:
                    continue
                buffer = getattr(geometry, attr, None)
                data = getattr(buffer, "data", None)
                if data is None:
                    continue
                arr = np.asarray(values, dtype=np.asarray(data).dtype).reshape(np.asarray(data).shape)
                data[...] = arr
                update_range = getattr(buffer, "update_range", None)
                if callable(update_range):
                    update_range(0, len(data))
            if index_values is not None and index_buffer is not None and getattr(index_buffer, "data", None) is not None:
                data = index_buffer.data
                arr = np.asarray(index_values, dtype=np.asarray(data).dtype).reshape(np.asarray(data).shape)
                data[...] = arr
                update_range = getattr(index_buffer, "update_range", None)
                if callable(update_range):
                    update_range(0, len(data))
            self.dynamic_geometry_updates_this_frame += 1
            return True
        except Exception:
            return False

    @staticmethod
    def _mesh_indices(mesh_data):
        indices = getattr(mesh_data, "indices", None)
        if indices is not None:
            try:
                return np.asarray(indices, dtype=np.uint32).reshape((-1, 3))
            except Exception:
                return None
        try:
            positions = np.asarray(getattr(mesh_data, "positions", None), dtype=np.float32)
        except Exception:
            return None
        if positions.ndim != 2 or positions.shape[1] < 3:
            return None
        vertex_count = int(len(positions))
        if vertex_count < 3 or vertex_count % 3 != 0:
            return None
        return np.arange(vertex_count, dtype=np.uint32).reshape((-1, 3))

    def _base_color(self, mesh_data) -> tuple[float, float, float, float]:
        base = tuple(float(c) for c in (mesh_data.material_color or (0.72, 0.74, 0.76, 1.0))[:4])
        if len(base) < 4:
            base = (*base[:3], 1.0)
        return base

    @staticmethod
    def _pygfx_uvs(values):
        if values is None:
            return None
        try:
            arr = np.asarray(values, dtype=np.float32)
        except Exception:
            return None
        if arr.ndim != 2 or arr.shape[1] < 2:
            return None
        out = np.array(arr[:, :2], dtype=np.float32, copy=True)
        # WGPU/D3D parity: renderer-neutral mesh data preserves KotOR's D3D
        # convention where V=0 is the texture top. pygfx samples its texture
        # arrays directly, so flip V here just like the WGPU shader does.
        out[:, 1] = 1.0 - out[:, 1]
        return out

    def _build_material(self, mesh_data, gfx, *, selected: bool = False, diffuse_map=None, lightmap_map=None):
        base = self._base_color(mesh_data)
        color = self._selection_color(base) if selected else base
        material = gfx.MeshPhongMaterial(color=color)
        if diffuse_map is not None:
            self._set_material_attr(material, "map", diffuse_map)
        if lightmap_map is not None:
            self._set_material_attr(material, "light_map", lightmap_map)
            self._set_material_attr(material, "light_map_intensity", 0.55)
        self._apply_material_flags(
            material,
            alpha_mode=str(getattr(mesh_data.material, "alpha_mode", "OPAQUE") or "OPAQUE").upper(),
            alpha_cutoff=float(getattr(mesh_data.material, "alpha_cutoff", 0.5) or 0.5),
            unlit=bool(getattr(mesh_data.material, "unlit", False)),
            color=color,
        )
        try:
            material.side = "both"
        except Exception:
            pass
        return material

    def _texture_map(self, mesh_data, gfx, *, channel: int, lightmap: bool = False):
        texture_attr = "lightmap_texture_data" if lightmap else "diffuse_texture_data"
        texture_data = getattr(getattr(mesh_data, "material", None), texture_attr, None)
        source = getattr(texture_data, "source", None)
        if source is None:
            return None
        texture_id = str(getattr(texture_data, "texture_id", "") or getattr(texture_data, "name", "") or id(source))
        sprite_alpha_source = 0 if lightmap else int(
            getattr(getattr(mesh_data, "material", None), "sprite_alpha_source", 0) or 0
        )
        revision = tuple(getattr(texture_data, "source_revision", ()) or ())
        key = (texture_id, revision, int(channel), bool(lightmap), int(sprite_alpha_source))
        cached = self.texture_maps.get(key)
        if cached is not None:
            return cached
        try:
            pixels = self._texture_pixels(
                source,
                sprite_alpha_source=sprite_alpha_source,
            )
            if pixels is None:
                return None
            texture = gfx.Texture(
                pixels,
                dim=2,
                colorspace="physical" if lightmap else "srgb",
                generate_mipmaps=not lightmap,
            )
            texture_map = gfx.TextureMap(
                texture,
                uv_channel=int(channel),
                wrap="clamp" if lightmap else "repeat",
                filter="linear",
                mipmap_filter="nearest" if lightmap else "linear",
            )
            self.texture_maps[key] = texture_map
            return texture_map
        except Exception:
            return None

    @staticmethod
    def _texture_pixels(source, *, sprite_alpha_source: int = 0):
        try:
            image = source.convert("RGBA") if hasattr(source, "convert") else source
            # Own writable storage so Texture.update_range can stream paint
            # into the retained pygfx resource in place.
            pixels = np.array(image, dtype=np.uint8, copy=True, order="C")
        except Exception:
            return None
        if pixels.ndim != 3 or pixels.shape[2] < 4:
            return None
        # TextureCache normalizes PIL rows to bottom-up for the shader-driven
        # paths. PyGFX uploads NumPy rows directly, so restore top-down image
        # order here; _pygfx_uvs supplies the one KotOR/D3D V conversion.
        pixels = np.ascontiguousarray(pixels[::-1, :, :4])
        if sprite_alpha_source:
            rgb = pixels[:, :, :3].astype(np.float32) / 255.0
            peak = np.max(rgb, axis=2)
            keyed = np.clip((peak - 0.08) / (0.40 - 0.08), 0.0, 1.0)
            keyed = keyed * keyed * (3.0 - 2.0 * keyed)
            existing_alpha = pixels[:, :, 3].astype(np.float32) / 255.0
            alpha = np.where(existing_alpha < 0.999, np.minimum(existing_alpha, keyed), keyed)
            pixels[:, :, 3] = np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8)
        return pixels

    @staticmethod
    def _selection_color(base: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return (
            min(1.0, base[0] * 0.6 + 0.35),
            min(1.0, base[1] * 0.6 + 0.32),
            min(1.0, base[2] * 0.6 + 0.08),
            base[3],
        )

    def _apply_selection_state(self, record: PygfxMeshRecord, selected: bool, hovered: bool | None = None) -> None:
        record.selected = bool(selected)
        if hovered is not None:
            record.hovered = bool(hovered)
        color = self._selection_color(record.base_color) if selected else record.base_color
        record.view_style_key = record.view_style_key or self._view_style_key()
        self._set_material_color(record.material, color)
        self._apply_material_style(record, record.view_style_key)
        self.material_updates_this_frame += 1

    def _apply_material_style(self, record: PygfxMeshRecord, style_key: tuple[Any, ...]) -> None:
        (
            show_solid,
            show_wireframe,
            show_texture,
            show_diffuse,
            show_lightmap,
            render_mode,
            cull_faces,
            xray,
            show_mesh_hover,
            wire_color,
            hover_color,
            selection_color,
        ) = style_key
        render_mode = str(render_mode or "realistic").strip().lower()
        self._ensure_material_class(record, render_mode)
        color = self._selection_color(record.base_color) if record.selected else record.base_color
        map_enabled = bool(show_texture) and bool(show_diffuse)
        if not map_enabled:
            luminance = max(0.18, min(0.82, color[0] * 0.3 + color[1] * 0.45 + color[2] * 0.25))
            color = (luminance, luminance, luminance, color[3])
        if render_mode == "flat":
            color = (min(1.0, color[0] * 1.08), min(1.0, color[1] * 1.08), min(1.0, color[2] * 1.08), color[3])
        elif render_mode == "shaded":
            color = (color[0] * 0.78, color[1] * 0.78, color[2] * 0.78, color[3])
        if bool(xray):
            color = (color[0], color[1], color[2], min(color[3], 0.38))
        self._set_material_color(record.material, color)
        self._set_material_attr(record.material, "opacity", color[3])
        self._apply_material_flags(
            record.material,
            alpha_mode=record.alpha_mode,
            alpha_cutoff=record.alpha_cutoff,
            unlit=record.unlit,
            color=color,
        )
        self._set_material_attr(record.material, "map", record.diffuse_map if map_enabled else None)
        lightmap_enabled = bool(show_texture) and bool(show_lightmap)
        self._set_material_attr(record.material, "light_map", record.lightmap_map if lightmap_enabled else None)
        self._set_material_attr(record.material, "light_map_intensity", 0.55 if lightmap_enabled else 0.0)
        try:
            record.material.side = "front" if bool(cull_faces) and not bool(record.double_sided) else "both"
        except Exception:
            pass
        self._set_material_attr(record.material, "wireframe", bool(show_wireframe) and not bool(show_solid))
        self._set_material_attr(record.material, "flat_shading", render_mode in {"flat", "shaded"})
        use_sprite_proxy = self._use_sprite_proxy(record) and bool(show_solid)
        self._set_sprite_proxy_visible(record, use_sprite_proxy, map_enabled)
        self._set_material_attr(record.mesh, "visible", self._source_visible(record.source) and (bool(show_solid) or bool(show_wireframe)))
        if bool(show_solid):
            self._set_material_attr(record.mesh, "visible", self._source_visible(record.source) and not use_sprite_proxy)
        else:
            self._set_material_attr(record.mesh, "visible", False)
        edge_visible = bool(show_wireframe) or bool(record.selected) or (bool(show_mesh_hover) and bool(record.hovered))
        edge_color = wire_color
        if record.selected:
            edge_color = selection_color
        elif record.hovered and bool(show_mesh_hover):
            edge_color = hover_color
        self._set_edge_overlay_visible(record, edge_visible, edge_color)

    def _use_sprite_proxy(self, record: PygfxMeshRecord) -> bool:
        return bool(
            int(getattr(record, "sprite_alpha_source", 0) or 0) > 0
            and str(getattr(record, "blend_mode", "") or "").upper() in {"ADDITIVE", "LIGHTEN"}
        )

    def _set_sprite_proxy_visible(self, record: PygfxMeshRecord, visible: bool, map_enabled: bool) -> None:
        if not visible or not map_enabled:
            if record.sprite_proxy_mesh is not None:
                self._set_material_attr(record.sprite_proxy_mesh, "visible", False)
            return
        if record.sprite_proxy_mesh is None:
            positions = self._sprite_proxy_positions(record)
            if positions is None:
                return
            try:
                geometry = record.gfx.Geometry(positions=positions)
                color = self._sprite_proxy_color(record)
                material = record.gfx.LineSegmentMaterial(
                    color=color,
                    thickness=7.5 + max(0.0, min(4.0, float(record.sprite_glow))) * 1.5,
                    thickness_space="screen",
                )
                proxy = record.gfx.Line(geometry, material, render_order=2400, name=f"{getattr(record.mesh, 'name', record.mesh_id)}-sprite")
                self._copy_local_transform(record.mesh, proxy)
                record.scene.add(proxy)
                record.sprite_proxy_mesh = proxy
                record.sprite_proxy_material = material
            except Exception:
                return
        self._copy_local_transform(record.mesh, record.sprite_proxy_mesh)
        self._set_material_attr(record.sprite_proxy_mesh, "visible", self._source_visible(record.source))
        if record.sprite_proxy_material is not None:
            self._set_material_color(record.sprite_proxy_material, self._sprite_proxy_color(record))

    def _sprite_proxy_positions(self, record: PygfxMeshRecord):
        try:
            points = np.asarray(record.geometry.positions.data, dtype=np.float32)
        except Exception:
            return None
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 3:
            return None
        mins = np.min(points[:, :3], axis=0)
        maxs = np.max(points[:, :3], axis=0)
        axis = int(np.argmax(maxs - mins))
        center = (mins + maxs) * 0.5
        a = center.copy()
        b = center.copy()
        a[axis] = mins[axis]
        b[axis] = maxs[axis]
        return np.asarray([a, b], dtype=np.float32)

    @staticmethod
    def _sprite_proxy_color(record: PygfxMeshRecord) -> tuple[float, float, float, float]:
        try:
            pixels = np.asarray(record.diffuse_map.texture.data, dtype=np.uint8)
            rgb = pixels[:, :, :3].astype(np.float32) / 255.0
            alpha = pixels[:, :, 3].astype(np.float32) / 255.0
            score = np.max(rgb, axis=2) * np.maximum(alpha, 0.05)
            index = np.unravel_index(int(np.argmax(score)), score.shape)
            color = rgb[index]
            peak = max(float(color[0]), float(color[1]), float(color[2]), 1e-5)
            color = np.clip(color / peak, 0.0, 1.0)
            return (float(color[0]), float(color[1]), float(color[2]), 0.95)
        except Exception:
            return (0.45, 0.85, 1.0, 0.95)

    def _ensure_material_class(self, record: PygfxMeshRecord, render_mode: str) -> None:
        target_name = "MeshBasicMaterial" if render_mode == "flat" else "MeshPhongMaterial"
        if type(record.material).__name__ == target_name:
            return
        cls = getattr(record.gfx, target_name, None)
        if cls is None:
            return
        try:
            record.material = cls(color=record.base_color)
            record.mesh.material = record.material
            self._set_material_attr(record.material, "map", record.diffuse_map)
            self._set_material_attr(record.material, "light_map", record.lightmap_map)
            self.material_updates_this_frame += 1
        except Exception:
            pass

    def _apply_material_flags(
        self,
        material,
        *,
        alpha_mode: str,
        alpha_cutoff: float,
        unlit: bool,
        color: tuple[float, float, float, float],
    ) -> None:
        mode = str(alpha_mode or "OPAQUE").upper()
        if mode in {"MASK", "CUTOUT"}:
            self._set_material_attr(material, "alpha_mode", "auto")
            self._set_material_attr(material, "alpha_test", float(alpha_cutoff or 0.5))
            self._set_material_attr(material, "depth_write", True)
        elif mode == "BLEND" or color[3] < 0.999:
            self._set_material_attr(material, "alpha_mode", "blend")
            self._set_material_attr(material, "depth_write", False)
        else:
            self._set_material_attr(material, "alpha_mode", "solid")
            self._set_material_attr(material, "depth_write", True)
        if bool(unlit):
            self._set_material_attr(material, "emissive", color[:3])

    def _set_edge_overlay_visible(self, record: PygfxMeshRecord, visible: bool, color) -> None:
        if not visible:
            if record.edge_mesh is not None:
                self._set_material_attr(record.edge_mesh, "visible", False)
            return
        if self._use_skinned_wire_overlay(record):
            self._set_skinned_edge_overlay_visible(record, color)
            return
        if record.edge_positions is None or len(record.edge_positions) < 2:
            record.edge_positions = self._edge_positions_from_geometry(record)
            if record.edge_positions is None or len(record.edge_positions) < 2:
                if record.edge_mesh is not None:
                    self._set_material_attr(record.edge_mesh, "visible", False)
                return
        if record.edge_mesh is None:
            try:
                material = record.gfx.LineSegmentMaterial(color=color, thickness=1.25, thickness_space="screen")
                edge_mesh = record.gfx.Line(self._build_edge_geometry(record, record.gfx), material, render_order=2500, name=f"{getattr(record.mesh, 'name', record.mesh_id)}-wire")
                edge_mesh.name = f"{getattr(record.mesh, 'name', record.mesh_id)}-wire"
                self._copy_local_transform(record.mesh, edge_mesh)
                record.scene.add(edge_mesh)
                record.edge_mesh = edge_mesh
                record.edge_material = material
                record.edge_geometry_dirty = False
            except Exception:
                return
        elif record.edge_geometry_dirty:
            self._update_edge_geometry(record, record.gfx)
            record.edge_geometry_dirty = False
        self._copy_local_transform(record.mesh, record.edge_mesh)
        self._set_material_color(record.edge_material, color)
        self._set_material_attr(record.edge_material, "thickness", 1.6 if record.selected or record.hovered else 1.15)
        self._set_material_attr(record.edge_mesh, "visible", self._source_visible(record.source))

    @staticmethod
    def _use_skinned_wire_overlay(record: PygfxMeshRecord) -> bool:
        return bool(
            getattr(record, "is_skinned", False)
            and getattr(record, "skeleton", None) is not None
            and hasattr(getattr(record, "gfx", None), "SkinnedMesh")
        )

    def _set_skinned_edge_overlay_visible(self, record: PygfxMeshRecord, color) -> None:
        if record.edge_mesh is not None and not bool(getattr(record.edge_mesh, "_gr_pygfx_skinned_wire_overlay", False)):
            try:
                record.scene.remove(record.edge_mesh)
            except Exception:
                pass
            record.edge_mesh = None
            record.edge_material = None
            record.edge_positions = None
        if record.edge_mesh is None:
            try:
                material = record.gfx.MeshBasicMaterial(color=color)
                self._set_material_attr(material, "wireframe", True)
                self._set_material_attr(material, "opacity", float(color[3]) if len(color) > 3 else 1.0)
                self._set_material_attr(material, "alpha_mode", "blend" if len(color) > 3 and float(color[3]) < 0.999 else "solid")
                edge_mesh = record.gfx.SkinnedMesh(record.geometry, material)
                edge_mesh.name = f"{getattr(record.mesh, 'name', record.mesh_id)}-wire"
                edge_mesh._gr_pygfx_skinned_wire_overlay = True
                try:
                    edge_mesh.bind(record.skeleton, bind_matrix=np.eye(4, dtype=np.float32))
                except Exception:
                    pass
                self._copy_local_transform(record.mesh, edge_mesh)
                record.scene.add(edge_mesh)
                record.edge_mesh = edge_mesh
                record.edge_material = material
                record.edge_geometry_dirty = False
            except Exception:
                return
        self._copy_local_transform(record.mesh, record.edge_mesh)
        self._set_material_color(record.edge_material, color)
        self._set_material_attr(record.edge_material, "wireframe", True)
        self._set_material_attr(record.edge_mesh, "visible", self._source_visible(record.source))

    def _build_edge_geometry(self, record: PygfxMeshRecord, gfx):
        return gfx.Geometry(positions=np.asarray(record.edge_positions, dtype=np.float32))

    def _edge_positions_from_geometry(self, record: PygfxMeshRecord):
        try:
            positions = np.asarray(getattr(getattr(record.geometry, "positions", None), "data", None), dtype=np.float32)
            indices_buffer = getattr(record.geometry, "indices", None)
            indices = getattr(indices_buffer, "data", None)
            if indices is not None:
                indices = np.asarray(indices, dtype=np.uint32).reshape(-1)
            edge_indices = self._build_edge_indices(indices, len(positions), positions=positions)
            if edge_indices is None or len(edge_indices) < 2:
                return None
            return np.ascontiguousarray(positions[np.asarray(edge_indices, dtype=np.uint32), :3], dtype=np.float32)
        except Exception:
            return None

    def _update_edge_geometry(self, record: PygfxMeshRecord, gfx) -> None:
        if record.edge_mesh is None or record.edge_positions is None:
            return
        geometry = getattr(record.edge_mesh, "geometry", None)
        buffer = getattr(geometry, "positions", None)
        data = getattr(buffer, "data", None)
        if data is None or tuple(np.asarray(data).shape) != tuple(np.asarray(record.edge_positions).shape):
            try:
                record.edge_mesh.geometry = self._build_edge_geometry(record, gfx)
            except Exception:
                pass
            return
        try:
            data[...] = np.asarray(record.edge_positions, dtype=np.asarray(data).dtype).reshape(np.asarray(data).shape)
            update_range = getattr(buffer, "update_range", None)
            if callable(update_range):
                update_range(0, len(data))
        except Exception:
            pass

    @staticmethod
    def _copy_local_transform(source, target) -> None:
        try:
            target.local.matrix = getattr(source.local, "matrix", None)
        except Exception:
            pass
        try:
            target.local.position = getattr(source.local, "position", None)
        except Exception:
            pass

    @staticmethod
    def _uses_sprite_wire_hull(material) -> bool:
        if material is None:
            return False
        blend_mode = str(getattr(material, "blend_mode", "") or "").upper()
        sprite_alpha = int(getattr(material, "sprite_alpha_source", 0) or 0)
        return blend_mode in {"ADDITIVE", "LIGHTEN"} and sprite_alpha > 0

    def _build_edge_indices(self, indices, vertex_count: int, *, positions=None, geometric: bool = False):
        if vertex_count <= 1:
            return None
        if indices is None:
            tri_indices = np.arange(vertex_count, dtype=np.uint32)
        else:
            tri_indices = np.asarray(indices, dtype=np.uint32).reshape(-1)
        if len(tri_indices) < 3:
            return None
        usable = len(tri_indices) - (len(tri_indices) % 3)
        if geometric and positions is not None:
            hull = self._build_geometric_boundary_edges(tri_indices[:usable], positions, vertex_count)
            if hull is not None and len(hull):
                return hull
        edge_set: set[tuple[int, int]] = set()
        out: list[int] = []
        for i in range(0, usable, 3):
            tri = [int(tri_indices[i]), int(tri_indices[i + 1]), int(tri_indices[i + 2])]
            if any(v < 0 or v >= vertex_count for v in tri):
                continue
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                key = (a, b) if a <= b else (b, a)
                if key in edge_set:
                    continue
                edge_set.add(key)
                out.extend((a, b))
        if not out:
            return None
        return np.ascontiguousarray(out, dtype=np.uint32)

    @staticmethod
    def _build_geometric_boundary_edges(tri_indices, positions, vertex_count: int):
        pos = np.asarray(positions, dtype=np.float32)
        tri = np.asarray(tri_indices, dtype=np.uint32).reshape(-1)
        if pos.ndim != 2 or pos.shape[1] < 3 or len(tri) < 3:
            return None

        def key(vertex_index: int) -> tuple[int, int, int]:
            vertex = pos[int(vertex_index), :3]
            return tuple(int(round(float(component) * 100000.0)) for component in vertex)

        edge_counts: dict[tuple[tuple[int, int, int], tuple[int, int, int]], int] = {}
        edge_representatives: dict[tuple[tuple[int, int, int], tuple[int, int, int]], tuple[int, int]] = {}
        for i in range(0, (len(tri) // 3) * 3, 3):
            face = [int(tri[i]), int(tri[i + 1]), int(tri[i + 2])]
            if any(v < 0 or v >= vertex_count for v in face):
                continue
            for left, right in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                a = key(left)
                b = key(right)
                if a == b:
                    continue
                edge_key = (a, b) if a <= b else (b, a)
                edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1
                edge_representatives.setdefault(edge_key, (left, right))

        out: list[int] = []
        for edge_key, count in edge_counts.items():
            if count != 1:
                continue
            left, right = edge_representatives[edge_key]
            out.extend((left, right))
        if not out:
            return None
        return np.ascontiguousarray(out, dtype=np.uint32)

    @staticmethod
    def _source_visible(source) -> bool:
        return not bool(getattr(source, "_gr_hidden", False)) and getattr(source, "render", True) is not False

    @staticmethod
    def _set_material_color(material, color: tuple[float, float, float, float]) -> None:
        try:
            material.color = color
        except Exception:
            pass

    @staticmethod
    def _set_material_attr(target, name: str, value) -> None:
        try:
            if hasattr(target, name):
                setattr(target, name, value)
        except Exception:
            pass

    def diagnostics(self) -> dict[str, int]:
        return {
            "mesh_cache_size": len(self.records),
            "texture_cache_size": len(self.texture_maps),
            "geometry_updates_this_frame": int(self.geometry_updates_this_frame),
            "dynamic_geometry_updates_this_frame": int(self.dynamic_geometry_updates_this_frame),
            "material_updates_this_frame": int(self.material_updates_this_frame),
            "transform_dirty_count": sum(1 for record in self.records.values() if record.transform_dirty),
        }
