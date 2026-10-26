from __future__ import annotations

import inspect
import os
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import src.adapters.rendering.pygfx_core.renderer as pygfx_renderer_module
import src.adapters.rendering.pygfx_core.scene_bridge as pygfx_scene_bridge_module
import src.adapters.rendering.moderngl_resources as moderngl_resources_module
import src.core.animation.gpu_skinning as gpu_skinning_module
import src.core.rendering.mesh_render_data as mesh_render_data_module
import src.core.rendering.skeleton_render_data as skeleton_render_data_module
from src.adapters.rendering.native_core.renderer import NativeViewportRenderer
from src.adapters.rendering.moderngl_resources import _GlTexCache
from src.adapters.rendering.pygfx_core.scene_bridge import PygfxSceneBridge
from src.adapters.rendering.pygfx_core.mesh_cache import PygfxMeshCache
from src.adapters.rendering.pygfx_core.renderer import PygfxViewportRenderer
from src.adapters.rendering.wgpu_core.resources import WgpuResourceCache
from src.adapters.rendering.wgpu_core.renderer import WgpuRenderer
from src.adapters.rendering.renderer_factory import (
    FallbackViewportRenderer,
    fallback_order,
    renderer_capabilities_snapshot,
)
from src.core.graphics.tex_atlas import TexArrayCache
from src.core.rendering.renderer_performance import ViewportFrameGovernor
from src.core.rendering.renderer_backend import RendererBackend, normalize_renderer_backend, renderer_backend_label
from src.core.rendering.renderer_settings import RendererSettings
from src.core.rendering.frame_core.renderer_textures import RendererTextureMixin
from src.core.rendering.frame_core.texture_cache import TextureCache
from src.gui.viewports.viewport_core.widgets.resource_cache import ViewportResourceCacheMixin
from src.gui.viewports.viewport_core.widgets.picking_hover import ViewportPickingHoverMixin
from src.gui.viewports.viewport_core.widgets.selection_mesh import ViewportSelectionMeshMixin
from src.gui.viewports.viewport_core.widgets.state_helpers import ViewportStateMixin


def test_frame_renderer_texture_residency_query_never_resolves_missing_texture() -> None:
    calls: list[str] = []

    class Cache:
        _cache = {"resident": object()}

        def get(self, name):
            calls.append(name)
            raise AssertionError("residency query must not resolve/decode a texture")

    owner = SimpleNamespace(textures={"local": object()}, tex_cache=Cache())

    assert RendererTextureMixin.is_texture_resident(owner, "LOCAL") is True
    assert RendererTextureMixin.is_texture_resident(owner, "resident") is True
    assert RendererTextureMixin.is_texture_resident(owner, "missing") is False
    assert RendererTextureMixin.is_texture_resident(owner, "NULL") is False
    assert calls == []


class _FakeGeometry:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        if "positions" in kwargs:
            self.positions = _FakeBuffer(kwargs["positions"])
        if "normals" in kwargs:
            self.normals = _FakeBuffer(kwargs["normals"])
        if "texcoords" in kwargs:
            self.texcoords = _FakeBuffer(kwargs["texcoords"])
        if "texcoords1" in kwargs:
            self.texcoords1 = _FakeBuffer(kwargs["texcoords1"])
        if "indices" in kwargs:
            self.indices = _FakeBuffer(kwargs["indices"])


class _FakeBuffer:
    def __init__(self, data):
        self.data = np.asarray(data).copy()
        self.update_calls = []

    def update_range(self, offset=0, size=None):
        self.update_calls.append((offset, size))


class _FakeMaterial:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.side = ""
        self.color = kwargs.get("color")
        self.opacity = 1.0
        self.wireframe = False
        self.flat_shading = False
        self.map = None
        self.light_map = None
        self.light_map_intensity = 0.0
        self.alpha_mode = "auto"
        self.alpha_test = 0.0
        self.thickness = kwargs.get("thickness", 1.0)
        self.depth_write = True


class MeshPhongMaterial(_FakeMaterial):
    pass


class MeshBasicMaterial(_FakeMaterial):
    pass


class _FakeLocal:
    def __init__(self):
        self.matrix = None
        self.position = None


class _FakeMesh:
    def __init__(self, geometry, material, **kwargs):
        self.geometry = geometry
        self.material = material
        self.local = _FakeLocal()
        self.name = str(kwargs.get("name", ""))
        self.visible = True


class _FakeSkinnedMesh(_FakeMesh):
    def bind(self, skeleton, bind_matrix=None):
        self.skeleton = skeleton
        self.bind_matrix = bind_matrix


class _FakeGfx:
    Geometry = _FakeGeometry
    MeshPhongMaterial = MeshPhongMaterial
    MeshBasicMaterial = MeshBasicMaterial
    Mesh = _FakeMesh
    SkinnedMesh = _FakeSkinnedMesh
    LineSegmentMaterial = _FakeMaterial
    PointsMaterial = _FakeMaterial
    Line = _FakeMesh
    Points = _FakeMesh

    class Bone:
        def __init__(self, name=""):
            self.name = name

    class Skeleton:
        def __init__(self, bones, bone_inverses=None):
            self.bones = bones
            self.bone_inverses = bone_inverses
            dtype = [("bone_matrices", np.float32, (4, 4))]
            self.bone_matrices_buffer = _FakeBuffer(np.zeros((max(1, len(bones)),), dtype=dtype))

        def update(self):
            return None

    class Background:
        def __init__(self, color):
            self.color = color

        @classmethod
        def from_color(cls, color):
            return cls(color)

    class Texture:
        def __init__(self, data, **kwargs):
            self.data = data
            self.kwargs = kwargs
            self.update_ranges = []

        def update_range(self, offset, size):
            self.update_ranges.append((tuple(offset), tuple(size)))

    class TextureMap:
        def __init__(self, texture, **kwargs):
            self.texture = texture
            self.kwargs = kwargs

    class AmbientLight:
        def __init__(self, color, intensity=1.0):
            self.color = color
            self.intensity = intensity

    class DirectionalLight:
        def __init__(self, color, intensity=1.0):
            self.color = color
            self.intensity = intensity
            self.local = _FakeLocal()

        def look_at(self, target):
            self.target = target

    class PointLight(DirectionalLight):
        pass


class _FakeScene:
    def __init__(self):
        self.children = []

    def add(self, obj):
        self.children.append(obj)

    def remove(self, obj):
        self.children.remove(obj)


class _ViewportPygfxPickProbe(ViewportPickingHoverMixin, ViewportStateMixin, ViewportSelectionMeshMixin):
    def _is_selected_model_root(self, node) -> bool:
        return False


def _mesh_data(*, mesh_id=7, source_revision=(3, 1, 0), material_revision=(1, 0, 0, 0)):
    diffuse_texture_data = None
    lightmap_texture_data = None
    if material_revision == ("texture",):
        diffuse = Image.new("RGBA", (2, 2), (180, 90, 40, 255))
        diffuse_texture_data = SimpleNamespace(
            texture_id="unit_diffuse",
            name="unit_diffuse",
            source=diffuse,
            source_revision=(id(diffuse), 2, 2),
        )
    if material_revision == ("lightmap",):
        diffuse = Image.new("RGBA", (2, 2), (180, 90, 40, 255))
        lightmap = Image.new("RGBA", (2, 2), (90, 120, 160, 255))
        diffuse_texture_data = SimpleNamespace(
            texture_id="unit_diffuse",
            name="unit_diffuse",
            source=diffuse,
            source_revision=(id(diffuse), 2, 2),
        )
        lightmap_texture_data = SimpleNamespace(
            texture_id="unit_lightmap",
            name="unit_lightmap",
            source=lightmap,
            source_revision=(id(lightmap), 2, 2),
        )
    material = SimpleNamespace(
        material_id="mat-a",
        source_revision=material_revision,
        double_sided=False,
        diffuse_texture_data=diffuse_texture_data,
        lightmap_texture_data=lightmap_texture_data,
        alpha_mode="OPAQUE",
        alpha_cutoff=0.5,
        unlit=False,
    )
    return SimpleNamespace(
        mesh_id=mesh_id,
        source=SimpleNamespace(name="mesh-a"),
        positions=np.asarray([(0, 0, 0), (1, 0, 0), (0, 1, 0)], dtype=np.float32),
        normals=None,
        uvs0=None,
        uvs1=None,
        indices=np.asarray([0, 1, 2], dtype=np.uint32),
        material=material,
        material_color=(0.5, 0.6, 0.7, 1.0),
        world_matrix=np.eye(4, dtype=np.float32),
        source_revision=source_revision,
    )


def _skinned_mesh_data(*, mesh_id=17, source_revision=(4, 2, 1, 1)):
    data = _mesh_data(mesh_id=mesh_id, source_revision=source_revision)
    data.is_skinned = True
    data.bone_indices = np.zeros((3, 4), dtype=np.uint32)
    data.bone_weights = np.zeros((3, 4), dtype=np.float32)
    data.bone_weights[:, 0] = 1.0
    data.skinning_cpu_fallback = False
    return data


def test_pygfx_backend_id_alias_and_label() -> None:
    assert RendererBackend.PYGFX_WGPU.value == "pygfx_wgpu"
    assert normalize_renderer_backend("pygfx") is RendererBackend.PYGFX_WGPU
    assert normalize_renderer_backend("pygfx/wgpu") is RendererBackend.PYGFX_WGPU
    assert renderer_backend_label(RendererBackend.PYGFX_WGPU) == "pygfx (WGPU)"


def test_legacy_native_aliases_migrate_to_direct3d_wgpu_mode() -> None:
    assert RendererBackend.NATIVE_D3D12.value == "native_d3d12"
    assert normalize_renderer_backend("native") is RendererBackend.WGPU_D3D12
    assert normalize_renderer_backend("native/d3d12") is RendererBackend.WGPU_D3D12
    assert normalize_renderer_backend("ghostrigger_native") is RendererBackend.WGPU_D3D12
    assert renderer_backend_label(RendererBackend.WGPU_D3D12) == "Direct3D (WGPU)"


def test_pygfx_capability_snapshot_includes_optional_backend() -> None:
    by_id = {caps.backend_id: caps for caps in renderer_capabilities_snapshot()}

    assert RendererBackend.PYGFX_WGPU.value in by_id
    assert by_id[RendererBackend.PYGFX_WGPU.value].name == "pygfx (WGPU)"
    assert by_id[RendererBackend.PYGFX_WGPU.value].api == "pygfx/WGPU"


def test_capability_snapshot_excludes_native_placeholder_backend() -> None:
    by_id = {caps.backend_id: caps for caps in renderer_capabilities_snapshot()}

    assert RendererBackend.NATIVE_D3D12.value not in by_id


def test_pygfx_fallback_order_prefers_pygfx_then_direct3d_wgpu() -> None:
    order = fallback_order(RendererSettings(backend=RendererBackend.PYGFX_WGPU, allow_fallback=True))

    assert order == [
        RendererBackend.PYGFX_WGPU,
        RendererBackend.WGPU_D3D12,
        RendererBackend.MODERNGL_GL330,
        RendererBackend.NULL_DIAGNOSTIC,
    ]


def test_legacy_native_fallback_order_uses_direct3d_wgpu() -> None:
    order = fallback_order(RendererSettings(backend=RendererBackend.NATIVE_D3D12, allow_fallback=True))

    assert order == [
        RendererBackend.WGPU_D3D12,
        RendererBackend.MODERNGL_GL330,
        RendererBackend.NULL_DIAGNOSTIC,
    ]


def test_native_renderer_reports_unavailable_when_runtime_missing(monkeypatch) -> None:
    def fake_load():
        raise OSError("native runtime missing for test")

    monkeypatch.setattr("src.adapters.rendering.native_core.renderer.NativeRuntimeBinding.load", fake_load)

    renderer = NativeViewportRenderer()
    caps = renderer.get_capabilities()

    assert renderer.is_available() is False
    assert caps.backend_id == RendererBackend.NATIVE_D3D12.value
    assert caps.available is False
    assert caps.api == "Native/D3D12"
    assert "native runtime missing for test" in caps.reason


def test_native_renderer_owns_runtime_scene_handle_when_binding_available() -> None:
    class FakeBinding:
        path = "fake-runtime.dll"

        def __init__(self):
            self.destroyed = []
            self.scene_destroyed = []
            self.cleared = []
            self.meshes = []
            self.mesh_buffer_updates = []
            self.mesh_vertex_range_updates = []
            self.mesh_index_range_updates = []
            self.mesh_skinning_updates = []
            self.mesh_skin_palette_bindings = []
            self.mesh_transform_updates = []
            self.material_updates = []
            self.material_state_updates = []
            self.removed_meshes = []
            self.textures = []
            self.texture_data_updates = []
            self.texture_region_updates = []
            self.removed_textures = []
            self.palettes = []
            self.palette_updates = []
            self.palette_matrix_updates = []
            self.palette_matrix_range_updates = []
            self.removed_palettes = []
            self.frames = []
            self.picks = []
            self.bounds_queries = []
            self.draw_lists = []
            self.command_records = []
            self.resource_residency_queries = []
            self.resource_upload_plan_queries = []
            self.device_resource_allocations = []
            self._device_handles = {}
            self._device_generations = {}
            self._device_uploaded_generations = {}
            self._device_states = {}
            self._next_device_handle = 1
            self.gpu_skinning_dispatch_queries = []
            self.cpu_skinning_fallback_batch_queries = []
            self.cpu_skinning_fallback_execute_calls = []
            self.cpu_skinned_positions = {}
            self.cpu_skinned_position_readbacks = []
            self.animation_samples = []
            self.animation_palette_samples = []
            self.cpu_skinning_calls = []

        def create(self):
            return 11

        def scene_create(self, handle):
            assert handle == 11
            return 22

        def capabilities(self):
            return {
                "backend_id": "native_d3d12",
                "name": "GhostRigger Native Runtime",
                "available": True,
                "api": "Native/D3D12",
                "diagnostic_only": True,
            }

        def version(self):
            return "fake"

        def diagnostics(self, handle):
            assert handle == 11
            return {"phase": "N2 retained scene contract"}

        def scene_diagnostics(self, handle, scene):
            assert (handle, scene) == (11, 22)
            return {"scene_id": 1, "clear_count": len(self.cleared)}

        def scene_clear(self, handle, scene):
            self.cleared.append((handle, scene))
            return True

        def scene_add_mesh(
            self,
            handle,
            scene,
            *,
            vertex_count,
            index_count,
            material_slot=0,
            flags=0,
            bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(0.0, 0.0, 0.0),
        ):
            assert (handle, scene) == (11, 22)
            mesh_id = len(self.meshes) + 100
            self.meshes.append(
                {
                    "mesh_id": mesh_id,
                    "vertex_count": vertex_count,
                    "index_count": index_count,
                    "material_slot": material_slot,
                    "flags": flags,
                    "bounds_min": bounds_min,
                    "bounds_max": bounds_max,
                }
            )
            return mesh_id

        def scene_remove_mesh(self, handle, scene, mesh_id):
            assert (handle, scene) == (11, 22)
            self.removed_meshes.append(mesh_id)
            return True

        def scene_update_mesh_buffers(self, handle, scene, mesh_id, *, positions, indices=None, flags=0):
            assert (handle, scene) == (11, 22)
            position_values = np.asarray(positions, dtype=np.float32).reshape(-1)
            index_values = np.asarray(indices if indices is not None else [], dtype=np.uint32).reshape(-1)
            self.mesh_buffer_updates.append(
                {
                    "mesh_id": mesh_id,
                    "vertex_count": int(position_values.size // 3),
                    "index_count": int(index_values.size),
                    "position_checksum": float(position_values.sum()),
                    "index_checksum": int(index_values.sum()),
                    "flags": flags,
                }
            )
            return True

        def scene_update_mesh_vertex_range(self, handle, scene, mesh_id, *, start_vertex, positions, flags=0):
            assert (handle, scene) == (11, 22)
            position_values = np.asarray(positions, dtype=np.float32)
            self.mesh_vertex_range_updates.append(
                {
                    "mesh_id": mesh_id,
                    "start_vertex": start_vertex,
                    "vertex_count": int(position_values.reshape(-1, 3).shape[0]),
                    "position_checksum": float(position_values.sum()),
                    "flags": flags,
                }
            )
            return True

        def scene_update_mesh_index_range(self, handle, scene, mesh_id, *, start_index, indices, flags=0):
            assert (handle, scene) == (11, 22)
            index_values = np.asarray(indices, dtype=np.uint32).reshape(-1)
            self.mesh_index_range_updates.append(
                {
                    "mesh_id": mesh_id,
                    "start_index": start_index,
                    "index_count": int(index_values.size),
                    "index_checksum": int(index_values.sum()),
                    "flags": flags,
                }
            )
            return True

        def scene_update_mesh_material(
            self,
            handle,
            scene,
            mesh_id,
            *,
            material_slot=0,
            flags=0,
            diffuse_texture_id=0,
            lightmap_texture_id=0,
            base_color=(1.0, 1.0, 1.0, 1.0),
        ):
            assert (handle, scene) == (11, 22)
            self.material_updates.append(
                {
                    "mesh_id": mesh_id,
                    "material_slot": material_slot,
                    "flags": flags,
                    "diffuse_texture_id": diffuse_texture_id,
                    "lightmap_texture_id": lightmap_texture_id,
                    "base_color": base_color,
                }
            )
            return True

        def scene_update_mesh_material_state(
            self,
            handle,
            scene,
            mesh_id,
            *,
            flags=0,
            base_color=(1.0, 1.0, 1.0, 1.0),
        ):
            assert (handle, scene) == (11, 22)
            self.material_state_updates.append(
                {
                    "mesh_id": mesh_id,
                    "flags": flags,
                    "base_color": base_color,
                }
            )
            return True

        def scene_update_mesh_skinning(self, handle, scene, mesh_id, *, bone_indices, bone_weights, flags=0):
            assert (handle, scene) == (11, 22)
            indices = np.asarray(bone_indices, dtype=np.uint32)
            weights = np.asarray(bone_weights, dtype=np.float32)
            self.mesh_skinning_updates.append(
                {
                    "mesh_id": mesh_id,
                    "vertex_count": int(indices.shape[0]),
                    "influences_per_vertex": int(indices.shape[1]),
                    "bone_index_checksum": int(indices.sum()),
                    "bone_weight_checksum": float(weights.sum()),
                    "flags": flags,
                }
            )
            return True

        def scene_bind_mesh_skin_palette(self, handle, scene, mesh_id, *, palette_id, flags=0):
            assert (handle, scene) == (11, 22)
            self.mesh_skin_palette_bindings.append(
                {
                    "mesh_id": mesh_id,
                    "palette_id": palette_id,
                    "flags": flags,
                }
            )
            return True

        def scene_update_mesh_transform(self, handle, scene, mesh_id, *, world_matrix=None, flags=0):
            assert (handle, scene) == (11, 22)
            matrix = np.asarray(world_matrix if world_matrix is not None else np.eye(4), dtype=np.float32)
            self.mesh_transform_updates.append(
                {
                    "mesh_id": mesh_id,
                    "checksum": float(matrix.reshape(-1).sum()),
                    "flags": flags,
                }
            )
            return True

        def scene_add_texture(
            self,
            handle,
            scene,
            *,
            width,
            height,
            byte_size,
            format_id=0,
            flags=0,
        ):
            assert (handle, scene) == (11, 22)
            texture_id = len(self.textures) + 200
            self.textures.append(
                {
                    "texture_id": texture_id,
                    "width": width,
                    "height": height,
                    "byte_size": byte_size,
                    "format_id": format_id,
                    "flags": flags,
                }
            )
            return texture_id

        def scene_remove_texture(self, handle, scene, texture_id):
            assert (handle, scene) == (11, 22)
            self.removed_textures.append(texture_id)
            return True

        def scene_update_texture_data(self, handle, scene, texture_id, *, data, row_pitch=0, flags=0):
            assert (handle, scene) == (11, 22)
            payload = bytes(data)
            self.texture_data_updates.append(
                {
                    "texture_id": texture_id,
                    "byte_count": len(payload),
                    "row_pitch": row_pitch,
                    "checksum": sum(payload),
                    "flags": flags,
                }
            )
            return True

        def scene_update_texture_region(
            self,
            handle,
            scene,
            texture_id,
            *,
            x,
            y,
            width,
            height,
            data,
            row_pitch=0,
            flags=0,
        ):
            assert (handle, scene) == (11, 22)
            payload = bytes(data)
            self.texture_region_updates.append(
                {
                    "texture_id": texture_id,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "byte_count": len(payload),
                    "row_pitch": row_pitch,
                    "checksum": sum(payload),
                    "flags": flags,
                }
            )
            return True

        def scene_add_skin_palette(self, handle, scene, *, bone_count, flags=0):
            assert (handle, scene) == (11, 22)
            palette_id = len(self.palettes) + 300
            self.palettes.append(
                {
                    "palette_id": palette_id,
                    "bone_count": bone_count,
                    "flags": flags,
                }
            )
            return palette_id

        def scene_update_skin_palette(self, handle, scene, palette_id, *, bone_count, flags=0):
            assert (handle, scene) == (11, 22)
            self.palette_updates.append(
                {
                    "palette_id": palette_id,
                    "bone_count": bone_count,
                    "flags": flags,
                }
            )
            return True

        def scene_update_skin_palette_matrices(self, handle, scene, palette_id, *, matrices, flags=0):
            assert (handle, scene) == (11, 22)
            values = np.asarray(matrices, dtype=np.float32).reshape(-1)
            self.palette_matrix_updates.append(
                {
                    "palette_id": palette_id,
                    "matrix_count": int(values.size // 16),
                    "checksum": float(values.sum()),
                    "flags": flags,
                }
            )
            return True

        def scene_update_skin_palette_matrix_range(
            self,
            handle,
            scene,
            palette_id,
            *,
            start_matrix,
            matrices,
            flags=0,
        ):
            assert (handle, scene) == (11, 22)
            values = np.asarray(matrices, dtype=np.float32).reshape(-1)
            self.palette_matrix_range_updates.append(
                {
                    "palette_id": palette_id,
                    "start_matrix": start_matrix,
                    "matrix_count": int(values.size // 16),
                    "checksum": float(values.sum()),
                    "flags": flags,
                }
            )
            return True

        def scene_remove_skin_palette(self, handle, scene, palette_id):
            assert (handle, scene) == (11, 22)
            self.removed_palettes.append(palette_id)
            return True

        def scene_update_animation_sample(
            self,
            handle,
            scene,
            *,
            clip_hash,
            time_seconds,
            duration_seconds=0.0,
            pose_matrices=None,
            flags=0,
        ):
            assert (handle, scene) == (11, 22)
            values = np.asarray(pose_matrices if pose_matrices is not None else [], dtype=np.float32).reshape(-1)
            self.animation_samples.append(
                {
                    "clip_hash": clip_hash,
                    "time_seconds": time_seconds,
                    "duration_seconds": duration_seconds,
                    "pose_matrix_count": int(values.size // 16),
                    "checksum": float(values.sum()),
                    "flags": flags,
                }
            )
            return True

        def cpu_skin_vertices(
            self,
            handle,
            *,
            positions,
            bone_indices,
            bone_weights,
            bone_matrices,
            normals=None,
            flags=0,
        ):
            assert handle == 11
            position_values = np.asarray(positions, dtype=np.float32).reshape(-1)
            normal_values = np.asarray(normals if normals is not None else [], dtype=np.float32).reshape(-1)
            index_values = np.asarray(bone_indices, dtype=np.uint32)
            weight_values = np.asarray(bone_weights, dtype=np.float32)
            matrix_values = np.asarray(bone_matrices, dtype=np.float32).reshape(-1)
            self.cpu_skinning_calls.append(
                {
                    "vertex_count": int(position_values.size // 3),
                    "influences_per_vertex": int(index_values.shape[1]),
                    "matrix_count": int(matrix_values.size // 16),
                    "flags": flags,
                }
            )
            return {
                "available": True,
                "positions": [1.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                "normals": normal_values.tolist(),
                "skinned_vertex_count": int(position_values.size // 3),
                "influence_count": int(weight_values.size),
                "position_checksum": 4.0,
                "normal_checksum": float(normal_values.sum()),
                "flags": flags,
            }

        def sample_animation_palette(
            self,
            handle,
            *,
            previous_matrices,
            next_matrices,
            interpolation_t,
            flags=0,
        ):
            assert handle == 11
            previous_values = np.asarray(previous_matrices, dtype=np.float32)
            next_values = np.asarray(next_matrices, dtype=np.float32)
            output = previous_values + (next_values - previous_values) * float(interpolation_t)
            self.animation_palette_samples.append(
                {
                    "matrix_count": int(output.reshape(-1).size // 16),
                    "interpolation_t": float(interpolation_t),
                    "checksum": float(output.sum()),
                    "flags": flags,
                }
            )
            return {
                "available": True,
                "matrices": output.reshape(-1).tolist(),
                "matrix_count": int(output.reshape(-1).size // 16),
                "interpolation_t": float(interpolation_t),
                "output_checksum": float(output.sum()),
                "flags": flags,
            }

        def scene_render_frame(
            self,
            handle,
            scene,
            *,
            viewport_width,
            viewport_height,
            device_pixel_ratio=1.0,
            time_seconds=0.0,
            flags=0,
            dirty_mesh_count=0,
            dirty_texture_count=0,
            dirty_skin_palette_count=0,
        ):
            assert (handle, scene) == (11, 22)
            frame = {
                "available": True,
                "frame_index": len(self.frames) + 1,
                "visible_mesh_count": len(self.meshes) - len(self.removed_meshes),
                "draw_call_count": len(self.meshes) - len(self.removed_meshes),
                "triangle_count": 1,
                "texture_count": len(self.textures) - len(self.removed_textures),
                "skin_palette_count": len(self.palettes) - len(self.removed_palettes),
                "viewport_width": viewport_width,
                "viewport_height": viewport_height,
                "device_pixel_ratio": device_pixel_ratio,
                "time_seconds": time_seconds,
                "flags": flags,
                "dirty_resource_count": dirty_mesh_count + dirty_texture_count + dirty_skin_palette_count,
                "cpu_frame_ms": 0.01,
            }
            self.frames.append(frame)
            return frame

        def scene_pick_bounds(self, handle, scene, *, origin, direction, flags=0):
            assert (handle, scene) == (11, 22)
            result = {
                "available": True,
                "hit": bool(self.meshes and direction[2] > 0.0),
                "mesh_id": self.meshes[0]["mesh_id"] if self.meshes and direction[2] > 0.0 else 0,
                "candidate_count": len(self.meshes) - len(self.removed_meshes),
                "distance": 10.0,
                "world_position": (origin[0], origin[1], origin[2] + 10.0),
                "bounds_min": (0.0, 0.0, 0.0),
                "bounds_max": (1.0, 1.0, 0.0),
                "flags": flags,
            }
            self.picks.append(result)
            return result

        def scene_query_bounds(self, handle, scene, *, bounds_min, bounds_max, flags=0):
            assert (handle, scene) == (11, 22)
            result = {
                "available": True,
                "candidate_count": len(self.meshes) - len(self.removed_meshes),
                "visible_count": 1 if self.meshes else 0,
                "first_visible_mesh_id": self.meshes[0]["mesh_id"] if self.meshes else 0,
                "visible_bounds_min": (0.0, 0.0, 0.0),
                "visible_bounds_max": (1.0, 1.0, 0.0),
                "bounds_valid": bool(self.meshes),
                "flags": flags,
            }
            self.bounds_queries.append(
                {
                    **result,
                    "query_bounds_min": tuple(bounds_min),
                    "query_bounds_max": tuple(bounds_max),
                }
            )
            return result

        def scene_assemble_draw_list(
            self,
            handle,
            scene,
            *,
            bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(0.0, 0.0, 0.0),
            flags=0,
            max_draw_count=0,
            max_item_count=None,
        ):
            assert (handle, scene) == (11, 22)
            result = {
                "available": True,
                "candidate_count": len(self.meshes) - len(self.removed_meshes),
                "draw_count": 1 if self.meshes else 0,
                "batch_count": 1 if self.meshes else 0,
                "triangle_count": 1 if self.meshes else 0,
                "first_mesh_id": self.meshes[0]["mesh_id"] if self.meshes else 0,
                "material_texture_binding_count": 0,
                "draw_bounds_min": (0.0, 0.0, 0.0),
                "draw_bounds_max": (1.0, 1.0, 0.0),
                "bounds_valid": bool(self.meshes),
                "flags": flags,
                "mesh_ids": [self.meshes[0]["mesh_id"]] if self.meshes and max_draw_count else [],
                "draw_items": [
                    {
                        "mesh_id": self.meshes[0]["mesh_id"],
                        "index_count": self.meshes[0]["index_count"],
                        "diffuse_texture_id": 0,
                        "lightmap_texture_id": 0,
                        "material_slot": self.meshes[0]["material_slot"],
                        "material_flags": 7,
                        "mesh_flags": self.meshes[0]["flags"],
                    }
                ]
                if self.meshes and max_draw_count
                else [],
                "draw_batches": [
                    {
                        "start_draw": 0,
                        "draw_count": 1,
                        "material_flags": 7,
                        "material_slot": self.meshes[0]["material_slot"],
                        "diffuse_texture_id": 0,
                        "lightmap_texture_id": 0,
                    }
                ]
                if self.meshes and max_draw_count
                else [],
            }
            self.draw_lists.append(
                {
                    **result,
                    "bounds_min": tuple(bounds_min),
                    "bounds_max": tuple(bounds_max),
                    "max_draw_count": max_draw_count,
                    "max_item_count": max_item_count,
                }
            )
            return result

        def scene_record_commands(
            self,
            handle,
            scene,
            *,
            bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(0.0, 0.0, 0.0),
            flags=0,
            max_draw_count=0,
            max_item_count=None,
        ):
            assert (handle, scene) == (11, 22)
            result = {
                "available": True,
                "candidate_count": len(self.meshes) - len(self.removed_meshes),
                "draw_count": 1 if self.meshes else 0,
                "batch_count": 1 if self.meshes else 0,
                "command_count": 2 if self.meshes else 0,
                "state_change_count": 1 if self.meshes else 0,
                "texture_bind_count": 0,
                "triangle_count": 1 if self.meshes else 0,
                "flags": flags,
            }
            self.command_records.append(
                {
                    **result,
                    "bounds_min": tuple(bounds_min),
                    "bounds_max": tuple(bounds_max),
                    "max_draw_count": max_draw_count,
                    "max_item_count": max_item_count,
                }
            )
            return result

        def scene_get_resource_residency(
            self,
            handle,
            scene,
            *,
            bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(0.0, 0.0, 0.0),
            flags=0,
            max_draw_count=0,
        ):
            assert (handle, scene) == (11, 22)
            result = {
                "available": True,
                "candidate_count": len(self.meshes) - len(self.removed_meshes),
                "draw_count": 1 if self.meshes else 0,
                "resident_mesh_count": 1 if self.meshes else 0,
                "missing_mesh_buffer_count": 0,
                "texture_reference_count": 0,
                "resident_texture_count": 0,
                "missing_texture_count": 0,
                "skin_palette_reference_count": 1 if self.mesh_skin_palette_bindings else 0,
                "resident_skin_palette_count": 1 if self.mesh_skin_palette_bindings else 0,
                "missing_skin_palette_count": 0,
                "vertex_buffer_bytes": 36 if self.meshes else 0,
                "index_buffer_bytes": 12 if self.meshes else 0,
                "texture_bytes": 0,
                "skin_palette_bytes": 128 if self.mesh_skin_palette_bindings else 0,
                "ready": bool(self.meshes),
                "flags": flags,
            }
            self.resource_residency_queries.append(
                {
                    **result,
                    "bounds_min": tuple(bounds_min),
                    "bounds_max": tuple(bounds_max),
                    "max_draw_count": max_draw_count,
                }
            )
            return result

        def scene_get_resource_upload_plan(self, handle, scene, *, flags=0, max_item_count=0):
            assert (handle, scene) == (11, 22)
            items = []
            has_active_mesh = bool(self.meshes) and 100 not in self.removed_meshes
            has_active_texture = bool(self.textures) and 200 not in self.removed_textures
            if has_active_mesh:
                items.append(
                    {
                        "resource_id": 100,
                        "vertex_buffer_bytes": 36,
                        "index_buffer_bytes": 12,
                        "texture_bytes": 0,
                        "skin_palette_bytes": 0,
                        "generation": 4,
                        "resource_type": 1,
                        "status": 1,
                    }
                )
            if has_active_texture:
                texture_byte_count = (
                    self.texture_data_updates[-1]["byte_count"]
                    if self.texture_data_updates
                    else self.textures[-1]["byte_size"]
                )
                items.append(
                    {
                        "resource_id": 200,
                        "vertex_buffer_bytes": 0,
                        "index_buffer_bytes": 0,
                        "texture_bytes": texture_byte_count,
                        "skin_palette_bytes": 0,
                        "generation": max(1, len(self.texture_data_updates) + len(self.texture_region_updates)),
                        "resource_type": 2,
                        "status": 1,
                    }
                )
            if self.mesh_skin_palette_bindings:
                items.append(
                    {
                        "resource_id": 300,
                        "vertex_buffer_bytes": 0,
                        "index_buffer_bytes": 0,
                        "texture_bytes": 0,
                        "skin_palette_bytes": 128,
                        "generation": 1,
                        "resource_type": 3,
                        "status": 1,
                    }
                )
            emitted_items = items[: max(0, int(max_item_count))]
            result = {
                "available": True,
                "mesh_upload_count": 1 if has_active_mesh else 0,
                "texture_upload_count": 1 if has_active_texture else 0,
                "skin_palette_upload_count": 1 if self.mesh_skin_palette_bindings else 0,
                "vertex_buffer_bytes": 36 if has_active_mesh else 0,
                "index_buffer_bytes": 12 if has_active_mesh else 0,
                "texture_bytes": sum(item["texture_bytes"] for item in items),
                "skin_palette_bytes": 128 if self.mesh_skin_palette_bindings else 0,
                "emitted_item_count": len(emitted_items),
                "items": emitted_items,
                "ready": bool(items),
                "flags": flags,
            }
            self.resource_upload_plan_queries.append({**result, "max_item_count": max_item_count})
            return result

        def scene_allocate_device_resources(self, handle, scene, *, flags=0, max_item_count=0):
            assert (handle, scene) == (11, 22)
            upload_plan = self.scene_get_resource_upload_plan(
                handle,
                scene,
                flags=flags,
                max_item_count=max_item_count,
            )
            allocated_handle_count = 0
            reused_resource_count = 0
            items = []

            def handle_for(key):
                nonlocal allocated_handle_count
                if key not in self._device_handles:
                    self._device_handles[key] = self._next_device_handle
                    self._next_device_handle += 1
                    allocated_handle_count += 1
                return self._device_handles[key]

            for upload_item in upload_plan["items"]:
                resource_type = upload_item["resource_type"]
                resource_id = upload_item["resource_id"]
                generation = upload_item["generation"]
                generation_key = (resource_type, resource_id)
                reused = self._device_generations.get(generation_key) == generation
                if reused:
                    reused_resource_count += 1
                self._device_generations[generation_key] = generation
                item = {
                    "resource_id": resource_id,
                    "vertex_buffer_handle": 0,
                    "index_buffer_handle": 0,
                    "texture_handle": 0,
                    "skin_palette_buffer_handle": 0,
                    "generation": generation,
                    "byte_count": (
                        upload_item["vertex_buffer_bytes"]
                        + upload_item["index_buffer_bytes"]
                        + upload_item["texture_bytes"]
                        + upload_item["skin_palette_bytes"]
                    ),
                    "resource_type": resource_type,
                    "status": 5 if reused else 1,
                }
                if resource_type == 1:
                    item["vertex_buffer_handle"] = handle_for(("mesh-v", resource_id))
                    item["index_buffer_handle"] = handle_for(("mesh-i", resource_id))
                elif resource_type == 2:
                    item["texture_handle"] = handle_for(("texture", resource_id))
                elif resource_type == 3:
                    item["skin_palette_buffer_handle"] = handle_for(("palette", resource_id))
                items.append(item)

            result = {
                "available": True,
                "mesh_resource_count": upload_plan["mesh_upload_count"],
                "texture_resource_count": upload_plan["texture_upload_count"],
                "skin_palette_resource_count": upload_plan["skin_palette_upload_count"],
                "allocated_handle_count": allocated_handle_count,
                "reused_resource_count": reused_resource_count,
                "missing_resource_count": 0,
                "vertex_buffer_bytes": upload_plan["vertex_buffer_bytes"],
                "index_buffer_bytes": upload_plan["index_buffer_bytes"],
                "texture_bytes": upload_plan["texture_bytes"],
                "skin_palette_bytes": upload_plan["skin_palette_bytes"],
                "emitted_item_count": len(items),
                "items": items,
                "ready": bool(items),
                "flags": flags,
            }
            self.device_resource_allocations.append({**result, "max_item_count": max_item_count})
            return result

        def scene_commit_device_resource_uploads(self, handle, scene, *, flags=0, max_item_count=0):
            assert (handle, scene) == (11, 22)
            allocation = self.scene_allocate_device_resources(
                handle,
                scene,
                flags=flags,
                max_item_count=max_item_count,
            )
            committed = 0
            skipped = 0
            items = []
            for allocation_item in allocation["items"]:
                generation_key = (allocation_item["resource_type"], allocation_item["resource_id"])
                generation = allocation_item["generation"]
                if self._device_uploaded_generations.get(generation_key) == generation:
                    skipped += 1
                    status = 4
                else:
                    committed += 1
                    status = 1
                    self._device_uploaded_generations[generation_key] = generation
                    self._device_states[generation_key] = 1
                items.append(
                    {
                        "resource_id": allocation_item["resource_id"],
                        "generation": generation,
                        "byte_count": allocation_item["byte_count"],
                        "resource_type": allocation_item["resource_type"],
                        "status": status,
                    }
                )
            return {
                "available": True,
                "committed_resource_count": committed,
                "skipped_resource_count": skipped,
                "missing_resource_count": 0,
                "vertex_buffer_bytes": allocation["vertex_buffer_bytes"],
                "index_buffer_bytes": allocation["index_buffer_bytes"],
                "texture_bytes": allocation["texture_bytes"],
                "skin_palette_bytes": allocation["skin_palette_bytes"],
                "emitted_item_count": len(items),
                "items": items,
                "ready": True,
                "flags": flags,
            }

        def scene_transition_device_resources(self, handle, scene, *, flags=0, max_item_count=0):
            assert (handle, scene) == (11, 22)
            commit = self.scene_commit_device_resource_uploads(
                handle,
                scene,
                flags=flags,
                max_item_count=max_item_count,
            )
            transition_count = 0
            already_ready_count = 0
            items = []
            for commit_item in commit["items"]:
                generation_key = (commit_item["resource_type"], commit_item["resource_id"])
                target_state = 2 if commit_item["resource_type"] == 1 else 4
                before_state = self._device_states.get(generation_key, 0)
                if before_state == target_state:
                    already_ready_count += 1
                    status = 4
                else:
                    transition_count += 1
                    status = 1
                    self._device_states[generation_key] = target_state
                items.append(
                    {
                        "resource_id": commit_item["resource_id"],
                        "generation": commit_item["generation"],
                        "byte_count": commit_item["byte_count"],
                        "resource_type": commit_item["resource_type"],
                        "before_state": before_state,
                        "after_state": target_state,
                        "status": status,
                    }
                )
            return {
                "available": True,
                "transition_count": transition_count,
                "already_ready_count": already_ready_count,
                "missing_upload_count": 0,
                "vertex_buffer_bytes": commit["vertex_buffer_bytes"],
                "index_buffer_bytes": commit["index_buffer_bytes"],
                "texture_bytes": commit["texture_bytes"],
                "skin_palette_bytes": commit["skin_palette_bytes"],
                "emitted_item_count": len(items),
                "items": items,
                "ready": True,
                "flags": flags,
            }

        def scene_get_gpu_skinning_dispatch(
            self,
            handle,
            scene,
            *,
            bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(0.0, 0.0, 0.0),
            flags=0,
            max_draw_count=0,
            max_item_count=None,
        ):
            assert (handle, scene) == (11, 22)
            ready = bool(self.mesh_skinning_updates and self.mesh_skin_palette_bindings)
            mesh_id = self.mesh_skinning_updates[-1]["mesh_id"] if self.mesh_skinning_updates else 0
            palette_id = (
                self.mesh_skin_palette_bindings[-1]["palette_id"]
                if self.mesh_skin_palette_bindings
                else 0
            )
            items = []
            item_capacity = max_item_count if max_item_count is not None else max_draw_count
            if item_capacity and (self.mesh_skinning_updates or self.mesh_skin_palette_bindings):
                status = 1 if ready else 2
                if not self.mesh_skin_palette_bindings:
                    status |= 4
                if not self.mesh_skinning_updates:
                    status |= 8
                items.append(
                    {
                        "mesh_id": mesh_id,
                        "skin_palette_id": palette_id,
                        "skinned_vertex_count": 3 if self.mesh_skinning_updates else 0,
                        "influence_count": 12 if self.mesh_skinning_updates else 0,
                        "palette_matrix_count": 2 if self.mesh_skin_palette_bindings else 0,
                        "palette_buffer_bytes": 128 if self.mesh_skin_palette_bindings else 0,
                        "status": status,
                        "flags": 7,
                    }
                )
            result = {
                "available": True,
                "candidate_count": len(self.meshes) - len(self.removed_meshes),
                "skinned_mesh_count": 1 if self.mesh_skinning_updates else 0,
                "gpu_ready_mesh_count": 1 if ready else 0,
                "cpu_fallback_mesh_count": 0 if ready else 1,
                "missing_palette_count": 0 if self.mesh_skin_palette_bindings else 1,
                "missing_influence_count": 0 if self.mesh_skinning_updates else 1,
                "skinned_vertex_count": 3 if self.mesh_skinning_updates else 0,
                "influence_count": 12 if self.mesh_skinning_updates else 0,
                "palette_matrix_count": 2 if self.mesh_skin_palette_bindings else 0,
                "palette_buffer_bytes": 128 if self.mesh_skin_palette_bindings else 0,
                "emitted_item_count": len(items),
                "items": items,
                "ready": ready,
                "flags": flags,
            }
            self.gpu_skinning_dispatch_queries.append(
                {
                    **result,
                    "bounds_min": tuple(bounds_min),
                    "bounds_max": tuple(bounds_max),
                    "max_draw_count": max_draw_count,
                    "max_item_count": max_item_count,
                }
            )
            return result

        def scene_get_cpu_skinning_fallback_batch(
            self,
            handle,
            scene,
            *,
            bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(0.0, 0.0, 0.0),
            flags=0,
            max_draw_count=0,
            max_item_count=None,
        ):
            assert (handle, scene) == (11, 22)
            ready = bool(self.mesh_skinning_updates and self.mesh_skin_palette_bindings)
            force_cpu_fallback = bool(flags & 2)
            fallback = force_cpu_fallback or not ready
            item_capacity = max_item_count if max_item_count is not None else max_draw_count
            items = []
            if item_capacity and fallback:
                status = 2
                if ready:
                    status |= 1
                if not self.mesh_skin_palette_bindings:
                    status |= 4
                if not self.mesh_skinning_updates:
                    status |= 8
                items.append(
                    {
                        "mesh_id": self.mesh_skinning_updates[-1]["mesh_id"] if self.mesh_skinning_updates else 0,
                        "skin_palette_id": self.mesh_skin_palette_bindings[-1]["palette_id"]
                        if self.mesh_skin_palette_bindings
                        else 0,
                        "skinned_vertex_count": 3 if self.mesh_skinning_updates else 0,
                        "influence_count": 12 if self.mesh_skinning_updates else 0,
                        "palette_matrix_count": 2 if self.mesh_skin_palette_bindings else 0,
                        "output_position_offset_bytes": 0,
                        "output_position_bytes": 36 if self.mesh_skinning_updates else 0,
                        "output_normal_offset_bytes": 0,
                        "output_normal_bytes": 36 if self.mesh_skinning_updates else 0,
                        "status": status,
                        "flags": 7,
                    }
                )
            result = {
                "available": True,
                "candidate_count": len(self.meshes) - len(self.removed_meshes),
                "skinned_mesh_count": 1 if self.mesh_skinning_updates else 0,
                "fallback_mesh_count": 1 if fallback else 0,
                "gpu_ready_mesh_count": 1 if ready else 0,
                "missing_palette_count": 0 if self.mesh_skin_palette_bindings else 1,
                "missing_influence_count": 0 if self.mesh_skinning_updates else 1,
                "skinned_vertex_count": 3 if self.mesh_skinning_updates else 0,
                "influence_count": 12 if self.mesh_skinning_updates else 0,
                "palette_matrix_count": 2 if fallback and self.mesh_skin_palette_bindings else 0,
                "output_position_bytes": 36 if fallback and self.mesh_skinning_updates else 0,
                "output_normal_bytes": 36 if fallback and self.mesh_skinning_updates else 0,
                "emitted_item_count": len(items),
                "items": items,
                "ready": fallback and ready,
                "flags": flags,
            }
            self.cpu_skinning_fallback_batch_queries.append(
                {
                    **result,
                    "bounds_min": tuple(bounds_min),
                    "bounds_max": tuple(bounds_max),
                    "max_draw_count": max_draw_count,
                    "max_item_count": max_item_count,
                }
            )
            return result

        def scene_execute_cpu_skinning_fallback(
            self,
            handle,
            scene,
            *,
            bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(0.0, 0.0, 0.0),
            flags=0,
            max_draw_count=0,
        ):
            assert (handle, scene) == (11, 22)
            ready = bool(self.mesh_skinning_updates and self.mesh_skin_palette_bindings)
            result = {
                "available": True,
                "candidate_count": len(self.meshes) - len(self.removed_meshes),
                "executed_mesh_count": 1 if ready else 0,
                "skipped_mesh_count": 0 if ready else 1,
                "skinned_vertex_count": 3 if ready else 0,
                "influence_count": 3 if ready else 0,
                "output_position_bytes": 36 if ready else 0,
                "position_checksum": 2.0 if ready else 0.0,
                "ready": ready,
                "flags": flags,
            }
            if ready:
                self.cpu_skinned_positions[self.mesh_skinning_updates[-1]["mesh_id"]] = [
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            self.cpu_skinning_fallback_execute_calls.append(
                {
                    **result,
                    "bounds_min": tuple(bounds_min),
                    "bounds_max": tuple(bounds_max),
                    "max_draw_count": max_draw_count,
                }
            )
            return result

        def scene_read_cpu_skinned_positions(
            self,
            handle,
            scene,
            mesh_id,
            *,
            start_vertex=0,
            vertex_count=0,
            flags=0,
        ):
            assert (handle, scene) == (11, 22)
            values = self.cpu_skinned_positions.get(mesh_id, [])
            available = len(values) // 3
            start = max(0, int(start_vertex))
            count = max(0, int(vertex_count))
            copied = values[start * 3 : (start + count) * 3] if start < available else []
            result = {
                "available": True,
                "positions": copied,
                "available_vertex_count": available,
                "copied_vertex_count": len(copied) // 3,
                "position_checksum": float(sum(copied)),
                "flags": flags,
            }
            self.cpu_skinned_position_readbacks.append(
                {
                    **result,
                    "mesh_id": mesh_id,
                    "start_vertex": start_vertex,
                    "vertex_count": vertex_count,
                }
            )
            return result

        def scene_destroy(self, handle, scene):
            self.scene_destroyed.append((handle, scene))

        def destroy(self, handle):
            self.destroyed.append(handle)

    binding = FakeBinding()
    renderer = NativeViewportRenderer(binding=binding)

    assert renderer.is_available() is True
    assert renderer.get_diagnostics()["scene"]["scene_id"] == 1
    mesh = _skinned_mesh_data(mesh_id=81)
    mesh.material.material_slot = 4
    mesh.material.double_sided = True
    mesh.material.unlit = True
    native_mesh_id = renderer.upload_mesh(mesh)
    assert native_mesh_id == 100
    assert renderer.upload_mesh(mesh) == 100
    assert binding.meshes == [
        {
            "mesh_id": 100,
            "vertex_count": 3,
            "index_count": 3,
            "material_slot": 4,
            "flags": 7,
            "bounds_min": (0.0, 0.0, 0.0),
            "bounds_max": (1.0, 1.0, 0.0),
        }
    ]
    assert binding.mesh_buffer_updates == [
        {
            "mesh_id": 100,
            "vertex_count": 3,
            "index_count": 3,
            "position_checksum": 2.0,
            "index_checksum": 3,
            "flags": 7,
        }
    ]
    assert renderer.update_mesh_vertex_range(
        mesh.mesh_id,
        start_vertex=1,
        positions=np.asarray([[4.0, 0.0, 0.0]], dtype=np.float32),
        flags=13,
    ) is True
    assert binding.mesh_vertex_range_updates == [
        {
            "mesh_id": 100,
            "start_vertex": 1,
            "vertex_count": 1,
            "position_checksum": 4.0,
            "flags": 13,
        }
    ]
    assert renderer.update_mesh_index_range(
        mesh.mesh_id,
        start_index=2,
        indices=np.asarray([9], dtype=np.uint32),
        flags=17,
    ) is True
    assert binding.mesh_index_range_updates == [
        {
            "mesh_id": 100,
            "start_index": 2,
            "index_count": 1,
            "index_checksum": 9,
            "flags": 17,
        }
    ]
    assert binding.mesh_skinning_updates == [
        {
            "mesh_id": 100,
            "vertex_count": 3,
            "influences_per_vertex": 4,
            "bone_index_checksum": 0,
            "bone_weight_checksum": 3.0,
            "flags": 7,
        }
    ]
    assert binding.mesh_transform_updates == [
        {
            "mesh_id": 100,
            "checksum": 4.0,
            "flags": 7,
        }
    ]
    assert binding.material_updates == [
        {
            "mesh_id": 100,
            "material_slot": 4,
            "flags": 7,
            "diffuse_texture_id": 0,
            "lightmap_texture_id": 0,
            "base_color": (0.5, 0.6, 0.7, 1.0),
        }
    ]
    assert renderer.update_mesh_material_state(
        mesh.mesh_id,
        flags=19,
        base_color=(1.0, 0.5, 0.25, 1.0),
    ) is True
    assert binding.material_state_updates == [
        {
            "mesh_id": 100,
            "flags": 19,
            "base_color": (1.0, 0.5, 0.25, 1.0),
        }
    ]
    renderer.show_wireframe = True
    renderer.show_texture = True
    renderer.show_grid = True
    image = renderer.render(None, None, 320, 200, device_pixel_ratio=2.0, time_seconds=12.5)
    assert image.size == (320, 200)
    assert binding.frames[-1]["viewport_width"] == 320
    assert binding.frames[-1]["viewport_height"] == 200
    assert binding.frames[-1]["dirty_resource_count"] == 4
    assert binding.frames[-1]["flags"] == 15
    assert renderer.get_diagnostics()["native_frame"]["frame_index"] == 1
    renderer.render(None, None, 320, 200)
    assert binding.frames[-1]["dirty_resource_count"] == 0
    hit = renderer.pick(
        SimpleNamespace(
            x=8,
            y=9,
            ray_origin=(0.25, 0.25, -10.0),
            ray_direction=(0.0, 0.0, 1.0),
        )
    )
    assert hit.hit is True
    assert hit.kind == "mesh_bounds"
    assert hit.raw_id == 100
    assert hit.mesh_id == mesh.mesh_id
    assert hit.world_position == (0.25, 0.25, 0.0)
    assert hit.diagnostic["candidate_count"] == 1
    visible = renderer.query_bounds((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0), flags=23)
    assert visible["available"] is True
    assert visible["visible_count"] == 1
    assert visible["first_visible_mesh_id"] == 100
    assert visible["bounds_valid"] is True
    assert visible["flags"] == 23
    assert binding.bounds_queries[-1]["query_bounds_min"] == (-1.0, -1.0, -1.0)
    assert binding.bounds_queries[-1]["query_bounds_max"] == (2.0, 2.0, 2.0)
    draw_list = renderer.assemble_draw_list(
        bounds_min=(-1.0, -1.0, -1.0),
        bounds_max=(2.0, 2.0, 2.0),
        flags=25,
        max_draw_count=4,
    )
    assert draw_list["available"] is True
    assert draw_list["draw_count"] == 1
    assert draw_list["batch_count"] == 1
    assert draw_list["triangle_count"] == 1
    assert draw_list["first_mesh_id"] == 100
    assert draw_list["mesh_ids"] == [100]
    assert draw_list["draw_items"] == [
        {
            "mesh_id": 100,
            "index_count": 3,
            "diffuse_texture_id": 0,
            "lightmap_texture_id": 0,
            "material_slot": 4,
            "material_flags": 7,
            "mesh_flags": 7,
        }
    ]
    assert draw_list["draw_batches"] == [
        {
            "start_draw": 0,
            "draw_count": 1,
            "material_flags": 7,
            "material_slot": 4,
            "diffuse_texture_id": 0,
            "lightmap_texture_id": 0,
        }
    ]
    assert draw_list["bounds_valid"] is True
    assert draw_list["flags"] == 25
    assert binding.draw_lists[-1]["bounds_min"] == (-1.0, -1.0, -1.0)
    assert binding.draw_lists[-1]["bounds_max"] == (2.0, 2.0, 2.0)
    assert binding.draw_lists[-1]["max_draw_count"] == 4
    command_stats = renderer.record_commands(
        bounds_min=(-1.0, -1.0, -1.0),
        bounds_max=(2.0, 2.0, 2.0),
        flags=27,
        max_draw_count=4,
    )
    assert command_stats["available"] is True
    assert command_stats["draw_count"] == 1
    assert command_stats["batch_count"] == 1
    assert command_stats["state_change_count"] == 1
    assert command_stats["command_count"] == 2
    assert command_stats["texture_bind_count"] == 0
    assert command_stats["triangle_count"] == 1
    assert command_stats["flags"] == 27
    assert binding.command_records[-1]["bounds_min"] == (-1.0, -1.0, -1.0)
    assert binding.command_records[-1]["bounds_max"] == (2.0, 2.0, 2.0)
    assert binding.command_records[-1]["max_draw_count"] == 4
    native_palette_id = renderer.upload_skin_palette("skin-a", bone_count=64, flags=5)
    assert native_palette_id == 300
    assert renderer.bind_mesh_skin_palette(mesh.mesh_id, "skin-a", flags=31) is True
    assert binding.mesh_skin_palette_bindings == [
        {
            "mesh_id": 100,
            "palette_id": 300,
            "flags": 31,
        }
    ]
    residency = renderer.get_resource_residency(
        bounds_min=(-1.0, -1.0, -1.0),
        bounds_max=(2.0, 2.0, 2.0),
        flags=29,
        max_draw_count=4,
    )
    assert residency["available"] is True
    assert residency["ready"] is True
    assert residency["resident_mesh_count"] == 1
    assert residency["missing_mesh_buffer_count"] == 0
    assert residency["skin_palette_reference_count"] == 1
    assert residency["resident_skin_palette_count"] == 1
    assert residency["missing_skin_palette_count"] == 0
    assert residency["vertex_buffer_bytes"] == 36
    assert residency["index_buffer_bytes"] == 12
    assert residency["skin_palette_bytes"] == 128
    assert residency["flags"] == 29
    assert binding.resource_residency_queries[-1]["bounds_min"] == (-1.0, -1.0, -1.0)
    assert binding.resource_residency_queries[-1]["bounds_max"] == (2.0, 2.0, 2.0)
    assert binding.resource_residency_queries[-1]["max_draw_count"] == 4
    upload_plan = renderer.get_resource_upload_plan(flags=30, max_item_count=4)
    assert upload_plan["available"] is True
    assert upload_plan["ready"] is True
    assert upload_plan["mesh_upload_count"] == 1
    assert upload_plan["texture_upload_count"] == 0
    assert upload_plan["skin_palette_upload_count"] == 1
    assert upload_plan["vertex_buffer_bytes"] == 36
    assert upload_plan["index_buffer_bytes"] == 12
    assert upload_plan["skin_palette_bytes"] == 128
    assert upload_plan["emitted_item_count"] == 2
    assert upload_plan["items"] == [
        {
            "resource_id": 100,
            "vertex_buffer_bytes": 36,
            "index_buffer_bytes": 12,
            "texture_bytes": 0,
            "skin_palette_bytes": 0,
            "generation": 4,
            "resource_type": 1,
            "status": 1,
        },
        {
            "resource_id": 300,
            "vertex_buffer_bytes": 0,
            "index_buffer_bytes": 0,
            "texture_bytes": 0,
            "skin_palette_bytes": 128,
            "generation": 1,
            "resource_type": 3,
            "status": 1,
        },
    ]
    assert upload_plan["flags"] == 30
    assert upload_plan["backend_id"] == RendererBackend.NATIVE_D3D12.value
    assert upload_plan["method"] == "Native resource upload plan"
    assert binding.resource_upload_plan_queries[-1]["max_item_count"] == 4
    allocation = renderer.allocate_device_resources(flags=32, max_item_count=4)
    assert allocation["available"] is True
    assert allocation["ready"] is True
    assert allocation["mesh_resource_count"] == 1
    assert allocation["texture_resource_count"] == 0
    assert allocation["skin_palette_resource_count"] == 1
    assert allocation["allocated_handle_count"] == 3
    assert allocation["reused_resource_count"] == 0
    assert allocation["missing_resource_count"] == 0
    assert allocation["vertex_buffer_bytes"] == 36
    assert allocation["index_buffer_bytes"] == 12
    assert allocation["skin_palette_bytes"] == 128
    assert allocation["emitted_item_count"] == 2
    assert allocation["items"] == [
        {
            "resource_id": 100,
            "vertex_buffer_handle": 1,
            "index_buffer_handle": 2,
            "texture_handle": 0,
            "skin_palette_buffer_handle": 0,
            "generation": 4,
            "byte_count": 48,
            "resource_type": 1,
            "status": 1,
        },
        {
            "resource_id": 300,
            "vertex_buffer_handle": 0,
            "index_buffer_handle": 0,
            "texture_handle": 0,
            "skin_palette_buffer_handle": 3,
            "generation": 1,
            "byte_count": 128,
            "resource_type": 3,
            "status": 1,
        },
    ]
    assert allocation["flags"] == 32
    assert allocation["backend_id"] == RendererBackend.NATIVE_D3D12.value
    assert allocation["method"] == "Native device resource allocation"
    commit = renderer.commit_device_resource_uploads(flags=34, max_item_count=4)
    assert commit["available"] is True
    assert commit["ready"] is True
    assert commit["committed_resource_count"] == 2
    assert commit["skipped_resource_count"] == 0
    assert commit["missing_resource_count"] == 0
    assert commit["vertex_buffer_bytes"] == 36
    assert commit["index_buffer_bytes"] == 12
    assert commit["skin_palette_bytes"] == 128
    assert commit["emitted_item_count"] == 2
    assert commit["items"] == [
        {
            "resource_id": 100,
            "generation": 4,
            "byte_count": 48,
            "resource_type": 1,
            "status": 1,
        },
        {
            "resource_id": 300,
            "generation": 1,
            "byte_count": 128,
            "resource_type": 3,
            "status": 1,
        },
    ]
    assert commit["flags"] == 34
    assert commit["backend_id"] == RendererBackend.NATIVE_D3D12.value
    assert commit["method"] == "Native device resource upload commit"
    commit_again = renderer.commit_device_resource_uploads(flags=36, max_item_count=4)
    assert commit_again["committed_resource_count"] == 0
    assert commit_again["skipped_resource_count"] == 2
    assert [item["status"] for item in commit_again["items"]] == [4, 4]
    transition = renderer.transition_device_resources(flags=38, max_item_count=4)
    assert transition["available"] is True
    assert transition["ready"] is True
    assert transition["transition_count"] == 2
    assert transition["already_ready_count"] == 0
    assert transition["missing_upload_count"] == 0
    assert transition["vertex_buffer_bytes"] == 36
    assert transition["index_buffer_bytes"] == 12
    assert transition["skin_palette_bytes"] == 128
    assert transition["emitted_item_count"] == 2
    assert transition["items"] == [
        {
            "resource_id": 100,
            "generation": 4,
            "byte_count": 48,
            "resource_type": 1,
            "before_state": 1,
            "after_state": 2,
            "status": 1,
        },
        {
            "resource_id": 300,
            "generation": 1,
            "byte_count": 128,
            "resource_type": 3,
            "before_state": 1,
            "after_state": 4,
            "status": 1,
        },
    ]
    assert transition["flags"] == 38
    assert transition["backend_id"] == RendererBackend.NATIVE_D3D12.value
    assert transition["method"] == "Native device resource transition"
    transition_again = renderer.transition_device_resources(flags=40, max_item_count=4)
    assert transition_again["transition_count"] == 0
    assert transition_again["already_ready_count"] == 2
    assert [item["status"] for item in transition_again["items"]] == [4, 4]
    dispatch = renderer.get_gpu_skinning_dispatch(
        bounds_min=(-1.0, -1.0, -1.0),
        bounds_max=(2.0, 2.0, 2.0),
        flags=33,
        max_draw_count=4,
    )
    assert dispatch["available"] is True
    assert dispatch["ready"] is True
    assert dispatch["skinned_mesh_count"] == 1
    assert dispatch["gpu_ready_mesh_count"] == 1
    assert dispatch["cpu_fallback_mesh_count"] == 0
    assert dispatch["skinned_vertex_count"] == 3
    assert dispatch["influence_count"] == 12
    assert dispatch["palette_matrix_count"] == 2
    assert dispatch["palette_buffer_bytes"] == 128
    assert dispatch["emitted_item_count"] == 1
    assert dispatch["items"] == [
        {
            "mesh_id": 100,
            "skin_palette_id": 300,
            "skinned_vertex_count": 3,
            "influence_count": 12,
            "palette_matrix_count": 2,
            "palette_buffer_bytes": 128,
            "status": 1,
            "flags": 7,
        }
    ]
    assert dispatch["flags"] == 33
    assert binding.gpu_skinning_dispatch_queries[-1]["bounds_min"] == (-1.0, -1.0, -1.0)
    assert binding.gpu_skinning_dispatch_queries[-1]["bounds_max"] == (2.0, 2.0, 2.0)
    assert binding.gpu_skinning_dispatch_queries[-1]["max_draw_count"] == 4
    fallback = renderer.get_cpu_skinning_fallback_batch(
        bounds_min=(-1.0, -1.0, -1.0),
        bounds_max=(2.0, 2.0, 2.0),
        flags=35,
        max_draw_count=4,
    )
    assert fallback["available"] is True
    assert fallback["ready"] is True
    assert fallback["fallback_mesh_count"] == 1
    assert fallback["gpu_ready_mesh_count"] == 1
    assert fallback["missing_palette_count"] == 0
    assert fallback["missing_influence_count"] == 0
    assert fallback["skinned_vertex_count"] == 3
    assert fallback["influence_count"] == 12
    assert fallback["palette_matrix_count"] == 2
    assert fallback["output_position_bytes"] == 36
    assert fallback["output_normal_bytes"] == 36
    assert fallback["emitted_item_count"] == 1
    assert fallback["items"] == [
        {
            "mesh_id": 100,
            "skin_palette_id": 300,
            "skinned_vertex_count": 3,
            "influence_count": 12,
            "palette_matrix_count": 2,
            "output_position_offset_bytes": 0,
            "output_position_bytes": 36,
            "output_normal_offset_bytes": 0,
            "output_normal_bytes": 36,
            "status": 3,
            "flags": 7,
        }
    ]
    assert fallback["flags"] == 35
    assert binding.cpu_skinning_fallback_batch_queries[-1]["bounds_min"] == (-1.0, -1.0, -1.0)
    assert binding.cpu_skinning_fallback_batch_queries[-1]["bounds_max"] == (2.0, 2.0, 2.0)
    assert binding.cpu_skinning_fallback_batch_queries[-1]["max_draw_count"] == 4
    executed = renderer.execute_cpu_skinning_fallback(
        bounds_min=(-1.0, -1.0, -1.0),
        bounds_max=(2.0, 2.0, 2.0),
        flags=37,
        max_draw_count=4,
    )
    assert executed["available"] is True
    assert executed["ready"] is True
    assert executed["executed_mesh_count"] == 1
    assert executed["skipped_mesh_count"] == 0
    assert executed["skinned_vertex_count"] == 3
    assert executed["influence_count"] == 3
    assert executed["output_position_bytes"] == 36
    assert executed["position_checksum"] == 2.0
    assert executed["flags"] == 37
    assert binding.cpu_skinning_fallback_execute_calls[-1]["bounds_min"] == (-1.0, -1.0, -1.0)
    assert binding.cpu_skinning_fallback_execute_calls[-1]["bounds_max"] == (2.0, 2.0, 2.0)
    assert binding.cpu_skinning_fallback_execute_calls[-1]["max_draw_count"] == 4
    readback = renderer.read_cpu_skinned_positions(mesh.mesh_id, vertex_count=3, flags=39)
    assert readback["available"] is True
    assert readback["available_vertex_count"] == 3
    assert readback["copied_vertex_count"] == 3
    assert readback["positions"] == [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert readback["position_checksum"] == 2.0
    assert readback["flags"] == 39
    assert binding.cpu_skinned_position_readbacks[-1]["mesh_id"] == 100
    assert binding.cpu_skinned_position_readbacks[-1]["vertex_count"] == 3
    miss = renderer.pick(
        SimpleNamespace(
            x=8,
            y=9,
            ray_origin=(0.25, 0.25, -10.0),
            ray_direction=(0.0, 0.0, -1.0),
        )
    )
    assert miss.hit is False
    assert miss.diagnostic["result"] == "miss"
    renderer.release_resource(mesh.mesh_id)
    assert binding.removed_meshes == [100]
    texture = SimpleNamespace(
        texture_id="diffuse-a",
        source=Image.new("RGBA", (2, 3), (255, 0, 0, 128)),
        has_alpha=True,
        is_lightmap=True,
        format_id=9,
    )
    native_texture_id = renderer.upload_texture(texture)
    assert native_texture_id == 200
    assert renderer.upload_texture(texture) == 200
    assert binding.textures == [
        {
            "texture_id": 200,
            "width": 2,
            "height": 3,
            "byte_size": 24,
            "format_id": 9,
            "flags": 3,
        }
    ]
    assert binding.texture_data_updates == [
        {
            "texture_id": 200,
            "byte_count": 24,
            "row_pitch": 8,
            "checksum": 2298,
            "flags": 3,
        }
    ]
    assert renderer.update_texture_region(
        texture.texture_id,
        x=1,
        y=0,
        width=1,
        height=1,
        data=bytes([0, 0, 0, 0]),
        row_pitch=4,
        flags=5,
    ) is True
    assert binding.texture_region_updates == [
        {
            "texture_id": 200,
            "x": 1,
            "y": 0,
            "width": 1,
            "height": 1,
            "byte_count": 4,
            "row_pitch": 4,
            "checksum": 0,
            "flags": 5,
        }
    ]
    texture_upload_plan = renderer.get_resource_upload_plan(flags=6, max_item_count=2)
    assert texture_upload_plan["available"] is True
    assert texture_upload_plan["ready"] is True
    assert texture_upload_plan["mesh_upload_count"] == 0
    assert texture_upload_plan["texture_upload_count"] == 1
    assert texture_upload_plan["skin_palette_upload_count"] == 1
    assert texture_upload_plan["texture_bytes"] == 24
    assert texture_upload_plan["skin_palette_bytes"] == 128
    assert texture_upload_plan["emitted_item_count"] == 2
    assert texture_upload_plan["items"] == [
        {
            "resource_id": 200,
            "vertex_buffer_bytes": 0,
            "index_buffer_bytes": 0,
            "texture_bytes": 24,
            "skin_palette_bytes": 0,
            "generation": 2,
            "resource_type": 2,
            "status": 1,
        },
        {
            "resource_id": 300,
            "vertex_buffer_bytes": 0,
            "index_buffer_bytes": 0,
            "texture_bytes": 0,
            "skin_palette_bytes": 128,
            "generation": 1,
            "resource_type": 3,
            "status": 1,
        }
    ]
    assert texture_upload_plan["flags"] == 6
    texture_allocation = renderer.allocate_device_resources(flags=8, max_item_count=3)
    assert texture_allocation["available"] is True
    assert texture_allocation["ready"] is True
    assert texture_allocation["mesh_resource_count"] == 0
    assert texture_allocation["texture_resource_count"] == 1
    assert texture_allocation["skin_palette_resource_count"] == 1
    assert texture_allocation["allocated_handle_count"] == 1
    assert texture_allocation["reused_resource_count"] == 1
    assert texture_allocation["missing_resource_count"] == 0
    assert texture_allocation["texture_bytes"] == 24
    assert texture_allocation["skin_palette_bytes"] == 128
    assert texture_allocation["emitted_item_count"] == 2
    assert texture_allocation["items"] == [
        {
            "resource_id": 200,
            "vertex_buffer_handle": 0,
            "index_buffer_handle": 0,
            "texture_handle": 4,
            "skin_palette_buffer_handle": 0,
            "generation": 2,
            "byte_count": 24,
            "resource_type": 2,
            "status": 1,
        },
        {
            "resource_id": 300,
            "vertex_buffer_handle": 0,
            "index_buffer_handle": 0,
            "texture_handle": 0,
            "skin_palette_buffer_handle": 3,
            "generation": 1,
            "byte_count": 128,
            "resource_type": 3,
            "status": 5,
        },
    ]
    assert texture_allocation["flags"] == 8
    texture_commit = renderer.commit_device_resource_uploads(flags=10, max_item_count=3)
    assert texture_commit["available"] is True
    assert texture_commit["ready"] is True
    assert texture_commit["committed_resource_count"] == 1
    assert texture_commit["skipped_resource_count"] == 1
    assert texture_commit["missing_resource_count"] == 0
    assert texture_commit["texture_bytes"] == 24
    assert texture_commit["skin_palette_bytes"] == 128
    assert texture_commit["emitted_item_count"] == 2
    assert texture_commit["items"] == [
        {
            "resource_id": 200,
            "generation": 2,
            "byte_count": 24,
            "resource_type": 2,
            "status": 1,
        },
        {
            "resource_id": 300,
            "generation": 1,
            "byte_count": 128,
            "resource_type": 3,
            "status": 4,
        },
    ]
    assert texture_commit["flags"] == 10
    texture_transition = renderer.transition_device_resources(flags=12, max_item_count=3)
    assert texture_transition["available"] is True
    assert texture_transition["ready"] is True
    assert texture_transition["transition_count"] == 1
    assert texture_transition["already_ready_count"] == 1
    assert texture_transition["missing_upload_count"] == 0
    assert texture_transition["texture_bytes"] == 24
    assert texture_transition["skin_palette_bytes"] == 128
    assert texture_transition["emitted_item_count"] == 2
    assert texture_transition["items"] == [
        {
            "resource_id": 200,
            "generation": 2,
            "byte_count": 24,
            "resource_type": 2,
            "before_state": 1,
            "after_state": 4,
            "status": 1,
        },
        {
            "resource_id": 300,
            "generation": 1,
            "byte_count": 128,
            "resource_type": 3,
            "before_state": 4,
            "after_state": 4,
            "status": 4,
        },
    ]
    assert texture_transition["flags"] == 12
    renderer.render(None, None, 320, 200)
    assert binding.frames[-1]["dirty_resource_count"] == 5
    renderer.release_resource(texture.texture_id)
    assert binding.removed_textures == [200]
    native_palette_id = renderer.upload_skin_palette("skin-a", bone_count=64, flags=5)
    assert native_palette_id == 300
    assert renderer.upload_skin_palette("skin-a", bone_count=64, flags=5) == 300
    assert binding.palettes == [
        {
            "palette_id": 300,
            "bone_count": 64,
            "flags": 5,
        }
    ]
    palette_matrices = np.stack([np.eye(4, dtype=np.float32), np.ones((4, 4), dtype=np.float32)])
    assert renderer.update_skin_palette("skin-a", bone_count=72, flags=7, matrices=palette_matrices) is True
    assert binding.palette_updates == [
        {
            "palette_id": 300,
            "bone_count": 72,
            "flags": 7,
        }
    ]
    assert binding.palette_matrix_updates == [
        {
            "palette_id": 300,
            "matrix_count": 2,
            "checksum": 20.0,
            "flags": 7,
        }
    ]
    range_matrix = np.eye(4, dtype=np.float32) * 2.0
    assert renderer.update_skin_palette_matrix_range(
        "skin-a",
        start_matrix=1,
        matrices=range_matrix.reshape(1, 4, 4),
        flags=9,
    ) is True
    assert binding.palette_matrix_range_updates == [
        {
            "palette_id": 300,
            "start_matrix": 1,
            "matrix_count": 1,
            "checksum": 8.0,
            "flags": 9,
        }
    ]
    assert renderer.update_animation_sample(
        "walk",
        time_seconds=0.25,
        duration_seconds=1.0,
        pose_matrices=palette_matrices,
        looped=True,
        flags=4,
    ) is True
    assert binding.animation_samples == [
        {
            "clip_hash": binding.animation_samples[0]["clip_hash"],
            "time_seconds": 0.25,
            "duration_seconds": 1.0,
            "pose_matrix_count": 2,
            "checksum": 20.0,
            "flags": 5,
        }
    ]
    assert binding.animation_samples[0]["clip_hash"] != 0
    sampled_palette = renderer.sample_animation_palette(
        previous_matrices=np.stack([np.eye(4, dtype=np.float32), np.zeros((4, 4), dtype=np.float32)]),
        next_matrices=np.stack([np.ones((4, 4), dtype=np.float32), np.ones((4, 4), dtype=np.float32) * 2.0]),
        interpolation_t=0.25,
        flags=6,
    )
    assert sampled_palette["available"] is True
    assert sampled_palette["matrix_count"] == 2
    assert sampled_palette["output_checksum"] == pytest.approx(15.0)
    assert binding.animation_palette_samples == [
        {
            "matrix_count": 2,
            "interpolation_t": 0.25,
            "checksum": pytest.approx(15.0),
            "flags": 6,
        }
    ]
    skinned = renderer.cpu_skin_vertices(
        positions=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        normals=np.asarray([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        bone_indices=np.asarray([[0], [1]], dtype=np.uint32),
        bone_weights=np.asarray([[1.0], [1.0]], dtype=np.float32),
        bone_matrices=palette_matrices,
        flags=11,
    )
    assert skinned["available"] is True
    assert skinned["positions"] == [1.0, 0.0, 0.0, 3.0, 0.0, 0.0]
    assert skinned["position_checksum"] == 4.0
    assert binding.cpu_skinning_calls == [
        {
            "vertex_count": 2,
            "influences_per_vertex": 1,
            "matrix_count": 2,
            "flags": 11,
        }
    ]
    renderer.release_resource("skin-a")
    assert binding.removed_palettes == [300]
    renderer.clear_caches()
    assert binding.cleared == [(11, 22)]
    renderer.shutdown()
    assert binding.scene_destroyed == [(11, 22)]
    assert binding.destroyed == [11]


def test_pygfx_renderer_reports_unavailable_when_optional_import_missing(monkeypatch) -> None:
    PygfxViewportRenderer._probe_cache = None

    def fake_find_spec(name):
        if name == "pygfx":
            return None
        return object()

    monkeypatch.setattr(pygfx_renderer_module.importlib_util, "find_spec", fake_find_spec)

    caps = PygfxViewportRenderer.probe_availability()

    assert caps.available is False
    assert "pygfx" in caps.reason
    PygfxViewportRenderer._probe_cache = None


def test_pygfx_availability_probe_does_not_import_gpu_runtime(monkeypatch) -> None:
    PygfxViewportRenderer._probe_cache = None
    imported = []

    def fake_import(name):
        imported.append(name)
        raise AssertionError("probe should not import optional GPU modules")

    monkeypatch.setattr(pygfx_renderer_module.importlib, "import_module", fake_import)
    monkeypatch.setattr(pygfx_renderer_module.importlib_util, "find_spec", lambda _name: object())

    caps = PygfxViewportRenderer.probe_availability()

    assert caps.available is True
    assert imported == []
    PygfxViewportRenderer._probe_cache = None


def test_pygfx_mesh_cache_retains_mesh_and_updates_same_shape_revision_in_place() -> None:
    cache = PygfxMeshCache()
    scene = _FakeScene()
    data = _mesh_data()

    first = cache.get_or_create(data, _FakeGfx, scene)
    cache.begin_frame()
    second = cache.get_or_create(data, _FakeGfx, scene)

    assert second is first
    assert second.geometry is first.geometry
    assert cache.geometry_updates_this_frame == 0
    assert cache.material_updates_this_frame == 0

    cache.begin_frame()
    third = cache.get_or_create(_mesh_data(source_revision=(3, 1, 1)), _FakeGfx, scene)

    assert third is first
    assert cache.geometry_updates_this_frame == 0
    assert cache.dynamic_geometry_updates_this_frame == 1
    assert cache.material_updates_this_frame == 0


def test_pygfx_viewport_bounds_and_gizmo_use_rendered_mesh_cache_records() -> None:
    source = SimpleNamespace(
        name="translated-render-mesh",
        vertices=[(-100.0, -100.0, 0.0), (-99.0, -100.0, 0.0), (-100.0, -99.0, 0.0)],
        faces=[(0, 1, 2)],
    )
    data = _mesh_data(mesh_id=44)
    data.source = source
    data.positions = np.asarray([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)], dtype=np.float32)
    data.indices = np.asarray([0, 1, 2], dtype=np.uint32)
    cache = PygfxMeshCache()
    scene = _FakeScene()
    record = cache.get_or_create(data, _FakeGfx, scene, selected=False)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = (5.0, 7.0, 0.0)
    record.mesh.local.matrix = matrix

    probe = _ViewportPygfxPickProbe()
    probe._gpu_renderer = SimpleNamespace(
        backend_id="pygfx_wgpu",
        scene_bridge=SimpleNamespace(mesh_cache=cache),
    )
    probe._renderer = SimpleNamespace(
        _proj_batch=lambda points, _w, _h: [tuple(point) for point in points],
        _get_world_verts_for_node=lambda _node: [
            (-100.0, -100.0, 0.0),
            (-99.0, -100.0, 0.0),
            (-100.0, -99.0, 0.0),
        ],
    )
    probe.model = SimpleNamespace(root_node=None)

    bounds = probe._projected_mesh_bounds(source, 100, 100)

    assert bounds is not None
    assert bounds[:4] == pytest.approx((1.0, 3.0, 11.0, 13.0))
    assert probe._gizmo_world_position(source) == pytest.approx((6.0, 8.0, 0.0))


def test_pygfx_mesh_cache_flips_d3d_uvs_like_wgpu_shader() -> None:
    cache = PygfxMeshCache()
    scene = _FakeScene()
    data = _mesh_data()
    data.uvs0 = np.asarray([(0.0, 0.0), (0.5, 0.25), (1.0, 1.0)], dtype=np.float32)
    data.uvs1 = np.asarray([(0.0, 0.1), (0.5, 0.6), (1.0, 0.9)], dtype=np.float32)

    record = cache.get_or_create(data, _FakeGfx, scene)

    assert np.allclose(record.geometry.texcoords.data[:, 0], data.uvs0[:, 0])
    assert np.allclose(record.geometry.texcoords.data[:, 1], 1.0 - data.uvs0[:, 1])
    assert np.allclose(record.geometry.texcoords1.data[:, 0], data.uvs1[:, 0])
    assert np.allclose(record.geometry.texcoords1.data[:, 1], 1.0 - data.uvs1[:, 1])


def test_pygfx_mesh_cache_adds_indices_for_expanded_vbo_triangle_lists() -> None:
    cache = PygfxMeshCache()
    scene = _FakeScene()
    data = _mesh_data()
    data.indices = None
    data.positions = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ],
        dtype=np.float32,
    )

    record = cache.get_or_create(data, _FakeGfx, scene)

    assert record.geometry.indices.data.tolist() == [[0, 1, 2], [3, 4, 5]]


def test_pygfx_mesh_cache_updates_only_dynamic_skin_buffers_for_animation_frames() -> None:
    cache = PygfxMeshCache()
    scene = _FakeScene()
    data = _mesh_data(source_revision=(3, 1, 0, 22, 1, 0))
    data.normals = np.asarray([(0.0, 0.0, 1.0)] * 3, dtype=np.float32)
    data.uvs0 = np.asarray([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], dtype=np.float32)
    data.skinning_cpu_fallback = True
    record = cache.get_or_create(data, _FakeGfx, scene)

    for buffer_name in ("positions", "normals", "texcoords", "indices"):
        getattr(record.geometry, buffer_name).update_calls.clear()
    next_data = _mesh_data(source_revision=(3, 1, 0, 22, 1, 16))
    next_data.positions = data.positions + np.asarray((0.1, 0.0, 0.0), dtype=np.float32)
    next_data.normals = data.normals
    next_data.uvs0 = data.uvs0
    next_data.skinning_cpu_fallback = True

    cache.get_or_create(next_data, _FakeGfx, scene, force_geometry_update=True)

    assert record.geometry.positions.update_calls
    assert record.geometry.normals.update_calls
    assert record.geometry.texcoords.update_calls == []
    assert record.geometry.indices.update_calls == []


def test_pygfx_mesh_cache_rebuilds_geometry_when_topology_shape_changes() -> None:
    cache = PygfxMeshCache()
    scene = _FakeScene()
    first = cache.get_or_create(_mesh_data(), _FakeGfx, scene)
    changed = _mesh_data(source_revision=(4, 2, 0))
    changed.positions = np.asarray([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], dtype=np.float32)
    changed.indices = np.asarray([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    cache.begin_frame()

    updated = cache.get_or_create(changed, _FakeGfx, scene)

    assert updated is first
    assert cache.geometry_updates_this_frame == 1
    assert cache.dynamic_geometry_updates_this_frame == 0


def test_pygfx_mesh_cache_updates_same_shape_animation_buffers_in_place() -> None:
    cache = PygfxMeshCache()
    scene = _FakeScene()
    first = _mesh_data(source_revision=(3, 1, 0))
    record = cache.get_or_create(first, _FakeGfx, scene)
    cache.begin_frame()
    moved = _mesh_data(source_revision=(3, 1, 99))
    moved.positions = np.asarray([(0, 0, 1), (1, 0, 1), (0, 1, 1)], dtype=np.float32)

    updated = cache.get_or_create(moved, _FakeGfx, scene)

    assert updated is record
    assert updated.geometry is record.geometry
    assert cache.geometry_updates_this_frame == 0
    assert cache.dynamic_geometry_updates_this_frame == 1
    assert np.allclose(record.geometry.positions.data, moved.positions)
    assert record.geometry.positions.update_calls


def test_pygfx_mesh_cache_force_updates_animation_buffers_even_when_revision_key_is_stable() -> None:
    cache = PygfxMeshCache()
    scene = _FakeScene()
    record = cache.get_or_create(_mesh_data(source_revision=(3, 1, 0)), _FakeGfx, scene)
    cache.begin_frame()
    moved = _mesh_data(source_revision=(3, 1, 0))
    moved.positions = np.asarray([(0, 0, 2), (1, 0, 2), (0, 1, 2)], dtype=np.float32)

    cache.get_or_create(moved, _FakeGfx, scene, force_geometry_update=True)

    assert cache.geometry_updates_this_frame == 0
    assert cache.dynamic_geometry_updates_this_frame == 1
    assert np.allclose(record.geometry.positions.data, moved.positions)


def test_viewport_frame_governor_tracks_pygfx_style_and_animation_dirty_flags() -> None:
    governor = ViewportFrameGovernor()

    governor.request_redraw("style switch", style=True)
    assert governor.dirty_flags["style"] is True
    assert governor.dirty_flags["scene"] is False
    governor.mark_clean_after_render("style switch")

    governor.request_redraw("animation pose", animation=True)
    assert governor.dirty_flags["animation"] is True
    assert governor.dirty_flags["scene"] is False

    governor.mark_clean_after_render("animation pose")
    governor.request_redraw("paint pixels", texture=True)
    assert governor.dirty_flags["texture"] is True
    assert governor.dirty_flags["scene"] is False


def test_pygfx_mesh_cache_updates_material_for_selection_without_geometry_rebuild() -> None:
    cache = PygfxMeshCache()
    scene = _FakeScene()
    data = _mesh_data()
    cache.get_or_create(data, _FakeGfx, scene, selected=False)

    cache.begin_frame()
    cache.get_or_create(data, _FakeGfx, scene, selected=True)

    assert cache.geometry_updates_this_frame == 0
    assert cache.material_updates_this_frame == 1


def test_pygfx_texture_upload_restores_top_down_rows_from_texture_cache() -> None:
    source = Image.fromarray(
        np.asarray(
            [
                [[0, 0, 255, 255]],
                [[255, 0, 0, 255]],
            ],
            dtype=np.uint8,
        ),
        mode="RGBA",
    )

    pixels = PygfxMeshCache._texture_pixels(source)

    assert pixels[0, 0].tolist() == [255, 0, 0, 255]
    assert pixels[1, 0].tolist() == [0, 0, 255, 255]


def test_pygfx_scene_bridge_updates_dirty_transform_without_geometry_rebuild() -> None:
    cache = PygfxMeshCache()
    scene = _FakeScene()
    data = _mesh_data()
    record = cache.get_or_create(data, _FakeGfx, scene)
    record.source.world_position = lambda: (2.0, 3.0, 4.0)
    cache.begin_frame()
    cache.mark_transform_dirty(record.source)

    bridge = PygfxSceneBridge(_FakeGfx, scene, cache)
    bridge.update_dirty_transforms()

    assert cache.geometry_updates_this_frame == 0
    assert cache.material_updates_this_frame == 0
    assert record.transform_dirty is False
    assert record.mesh.local.matrix[0, 3] == pytest.approx(2.0)


def test_pygfx_scene_bridge_uses_identity_model_matrix_for_skinned_vbo_data(monkeypatch) -> None:
    data = _mesh_data()
    data.is_skinned = True
    data.world_matrix = np.eye(4, dtype=np.float32)
    data.world_matrix[0, 3] = 42.0

    def fake_iter_mesh_render_data(model, **kwargs):
        return [data]

    monkeypatch.setattr(pygfx_scene_bridge_module, "iter_mesh_render_data", fake_iter_mesh_render_data)
    bridge = PygfxSceneBridge(_FakeGfx, _FakeScene(), PygfxMeshCache())

    bridge.update_scene(SimpleNamespace(name="model"))

    record = next(iter(bridge.mesh_cache.records.values()))
    assert np.allclose(record.mesh.local.matrix, np.eye(4, dtype=np.float32))


def test_pygfx_scene_bridge_uses_wgpu_vbo_builder_for_uv_seams(monkeypatch) -> None:
    seen = {}

    def fake_iter_mesh_render_data(model, **kwargs):
        seen["vbo_builder"] = kwargs.get("vbo_builder")
        return []

    monkeypatch.setattr(pygfx_scene_bridge_module, "iter_mesh_render_data", fake_iter_mesh_render_data)
    bridge = PygfxSceneBridge(_FakeGfx, _FakeScene(), PygfxMeshCache())

    bridge.update_scene(SimpleNamespace(name="model"))

    assert callable(seen["vbo_builder"])
    assert seen["vbo_builder"].__name__ == "_build_vbo_data"


def test_skinned_animation_vbo_normals_are_cached_between_frames(monkeypatch) -> None:
    node = SimpleNamespace(
        name="skin",
        is_skin=True,
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        normals=[(0.0, 0.0, 1.0)] * 3,
        faces=[(0, 1, 2)],
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        uvs_lm=[],
        bone_map=["root"],
        skin_data=[(0, 1.0)],
        render=True,
        _gr_revision=3,
        world_transform=lambda: ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )
    model = SimpleNamespace(all_nodes=lambda: [node])
    vbo = np.zeros((3, 22), dtype=np.float32)
    vbo[:, 0:3] = np.asarray(node.vertices, dtype=np.float32)
    vbo[:, 3:6] = np.asarray(node.normals, dtype=np.float32)
    vbo[:, 6:8] = np.asarray(node.uvs, dtype=np.float32)
    vbo[:, 14] = 0.0
    vbo[:, 18] = 1.0
    calls = {"smooth": 0, "vbo": 0}

    def fake_smooth(positions, normals, indices):
        calls["smooth"] += 1
        return normals

    def fake_vbo_builder(*_args, **_kwargs):
        calls["vbo"] += 1
        return vbo, np.asarray([0, 1, 2], dtype=np.uint32)

    monkeypatch.setattr(mesh_render_data_module, "smooth_render_normals", fake_smooth)

    first = list(
        mesh_render_data_module.iter_mesh_render_data(
            model,
            anim_pose=SimpleNamespace(time=0.0),
            allow_cpu_skinning=False,
            vbo_builder=fake_vbo_builder,
        )
    )
    second = list(
        mesh_render_data_module.iter_mesh_render_data(
            model,
            anim_pose=SimpleNamespace(time=0.1),
            allow_cpu_skinning=False,
            vbo_builder=fake_vbo_builder,
        )
    )

    assert len(first) == 1
    assert len(second) == 1
    assert calls == {"smooth": 1, "vbo": 1}
    assert second[0].source_revision[-1] == 1


def test_rigid_animation_vbo_rows_are_cached_between_frames(monkeypatch) -> None:
    node = SimpleNamespace(
        name="rigid",
        is_skin=False,
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        normals=[(0.0, 0.0, 1.0)] * 3,
        faces=[(0, 1, 2)],
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        uvs_lm=[],
        render=True,
        _gr_revision=9,
        world_transform=lambda: ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )
    model = SimpleNamespace(all_nodes=lambda: [node])
    vbo = np.zeros((3, 10), dtype=np.float32)
    vbo[:, 0:3] = np.asarray(node.vertices, dtype=np.float32)
    vbo[:, 3:6] = np.asarray(node.normals, dtype=np.float32)
    vbo[:, 6:8] = np.asarray(node.uvs, dtype=np.float32)
    calls = {"smooth": 0, "vbo": 0}

    def fake_smooth(positions, normals, indices):
        calls["smooth"] += 1
        return normals

    def fake_vbo_builder(*_args, **_kwargs):
        calls["vbo"] += 1
        return vbo, np.asarray([0, 1, 2], dtype=np.uint32)

    monkeypatch.setattr(mesh_render_data_module, "smooth_render_normals", fake_smooth)

    first = list(
        mesh_render_data_module.iter_mesh_render_data(
            model,
            anim_pose=SimpleNamespace(time=0.0),
            vbo_builder=fake_vbo_builder,
        )
    )
    second = list(
        mesh_render_data_module.iter_mesh_render_data(
            model,
            anim_pose=SimpleNamespace(time=0.1),
            vbo_builder=fake_vbo_builder,
        )
    )

    assert len(first) == 1
    assert len(second) == 1
    assert calls == {"smooth": 1, "vbo": 1}


def test_cpu_skinning_reuses_model_inverse_bind_uploader(monkeypatch) -> None:
    calls = {"build": 0, "palette": 0}

    class FakeUploader:
        def __init__(self, max_bones):
            self.max_bones = max_bones

        def build_inverse_bind_pose(self, model):
            calls["build"] += 1
            return len(model.all_nodes())

        def compute_skin_node_palette(self, _node, _anim_pose, anim_base_pose=None):
            calls["palette"] += 1

        def as_numpy_array(self):
            return np.asarray([np.eye(4, dtype=np.float32)], dtype=np.float32)

    monkeypatch.setattr(gpu_skinning_module, "MatrixPaletteUploader", FakeUploader)
    node = SimpleNamespace(
        name="skin",
        parent=None,
        position=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        _gr_revision=0,
    )
    model = SimpleNamespace(name="model", supermodel="", all_nodes=lambda: [node])
    skinning = SimpleNamespace(
        is_skinned=True,
        bone_indices=np.zeros((3, 4), dtype=np.uint16),
        bone_weights=np.asarray([[1.0, 0.0, 0.0, 0.0]] * 3, dtype=np.float32),
    )
    positions = np.asarray([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], dtype=np.float32)
    normals = np.asarray([(0.0, 0.0, 1.0)] * 3, dtype=np.float32)

    skeleton_render_data_module.cpu_skin_vbo_arrays(node, positions, normals, skinning, SimpleNamespace(nodes={}), model=model)
    skeleton_render_data_module.cpu_skin_vbo_arrays(node, positions, normals, skinning, SimpleNamespace(nodes={}), model=model)

    assert calls == {"build": 1, "palette": 2}


def test_bas_attachment_skin_meshes_switch_bind_buffers_for_body_animation(monkeypatch) -> None:
    root = SimpleNamespace(
        name="head_root",
        parent=None,
        position=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        children=[],
        _gr_bas_attachment_layer=True,
        _gr_bas_attachment_root=True,
    )
    skin = SimpleNamespace(
        name="Head",
        parent=root,
        children=[],
        is_skin=True,
        position=(0.0, 0.0, 2.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        vertices=[(0.0, 0.0, -2.0), (1.0, 0.0, -2.0), (0.0, 1.0, -2.0)],
        normals=[(0.0, 0.0, 1.0)] * 3,
        faces=[(0, 1, 2)],
        uvs=[],
        uvs_lm=[],
        bone_map=["head_root"],
        skin_data=[object(), object(), object()],
        qbone_list=[(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
        tbone_list=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        render=True,
        _gr_bas_attachment_layer=True,
        _gr_bas_attachment_root_ref=root,
        _gr_revision=1,
    )
    root.children = [skin]
    vbo = np.zeros((3, 22), dtype=np.float32)
    vbo[:, 3:6] = np.asarray(skin.normals, dtype=np.float32)
    vbo[:, 14] = 0.0
    vbo[:, 18] = 1.0
    model = SimpleNamespace(all_nodes=lambda: [root, skin])
    calls = {}

    def fake_vbo_builder(_node, world_pos, _world_orient, **kwargs):
        calls["apply_skin_node_transform_for_bind"] = kwargs.get("apply_skin_node_transform_for_bind")
        out = vbo.copy()
        out[:, 0:3] = np.asarray(skin.vertices, dtype=np.float32)
        if kwargs.get("apply_skin_node_transform_for_bind"):
            out[:, 0:3] += np.asarray(world_pos, dtype=np.float32)
        return out, np.asarray([0, 1, 2], dtype=np.uint32)

    rest_rows = list(
        mesh_render_data_module.iter_mesh_render_data(
            model,
            anim_pose=None,
            allow_cpu_skinning=False,
            vbo_builder=fake_vbo_builder,
        )
    )

    assert len(rest_rows) == 1
    assert calls["apply_skin_node_transform_for_bind"] is True
    assert tuple(np.round(rest_rows[0].positions[:, 2], 4)) == (0.0, 0.0, 0.0)

    rows = list(
        mesh_render_data_module.iter_mesh_render_data(
            model,
            anim_pose=SimpleNamespace(time=0.25, nodes={}),
            allow_cpu_skinning=False,
            vbo_builder=fake_vbo_builder,
        )
    )

    assert len(rows) == 1
    assert rows[0].is_skinned is False
    assert rows[0].skinning_cpu_fallback is True
    assert calls["apply_skin_node_transform_for_bind"] is False
    assert tuple(np.round(rows[0].positions[:, 2], 4)) == (-2.0, -2.0, -2.0)
    assert rows[0].bone_indices is None
    assert rows[0].bone_weights is None
    palette_model = mesh_render_data_module.bas_attachment_palette_model_for_node(skin)
    assert palette_model is not None
    assert getattr(palette_model, "_gr_bas_attachment_palette_model") is True
    assert [node.name for node in palette_model.all_nodes()] == ["head_root", "Head"]


def test_pygfx_skin_palette_uses_bas_attachment_local_model(monkeypatch) -> None:
    calls = {"build_nodes": []}

    class FakeUploader:
        def __init__(self, max_bones):
            self.max_bones = max_bones

        def build_inverse_bind_pose(self, model):
            nodes = list(model.all_nodes())
            calls["build_nodes"].append([getattr(node, "name", "") for node in nodes])
            return len(nodes)

        def compute_skin_node_palette(self, _node, _anim_pose, anim_base_pose=None):
            pass

        def as_numpy_array(self):
            matrix = np.eye(4, dtype=np.float32)
            matrix[2, 3] = 2.0
            return np.asarray([matrix], dtype=np.float32)

    monkeypatch.setattr(gpu_skinning_module, "MatrixPaletteUploader", FakeUploader)

    root = SimpleNamespace(
        name="head_root",
        parent=None,
        children=[],
        position=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        _gr_bas_attachment_layer=True,
        _gr_bas_attachment_root=True,
    )
    skin = SimpleNamespace(
        name="Head",
        parent=root,
        children=[],
        is_skin=True,
        _gr_bas_attachment_layer=True,
        _gr_bas_attachment_root_ref=root,
    )
    root.children = [skin]
    skeleton_buffer = SimpleNamespace(
        data=np.zeros(1, dtype=[("bone_matrices", np.float32, (4, 4))]),
        update_range=lambda *_args, **_kwargs: None,
    )
    record = SimpleNamespace(source=skin, skeleton=SimpleNamespace(bone_matrices_buffer=skeleton_buffer))

    PygfxMeshCache().update_skin_palette(
        record,
        SimpleNamespace(time=0.1, nodes={}),
        model=SimpleNamespace(name="body", all_nodes=lambda: []),
    )

    assert calls["build_nodes"] == [["head_root", "Head"]]
    assert skeleton_buffer.data[0]["bone_matrices"][3, 2] == pytest.approx(2.0)
    assert skeleton_buffer.data[0]["bone_matrices"][2, 3] == pytest.approx(0.0)


def test_pygfx_skin_palette_uses_runtime_actor_source_model(monkeypatch) -> None:
    calls = {"build_models": []}

    class FakeUploader:
        def __init__(self, max_bones):
            self.max_bones = max_bones

        def build_inverse_bind_pose(self, model):
            calls["build_models"].append(model)
            return len(list(model.all_nodes()))

        def compute_skin_node_palette(self, _node, _anim_pose, anim_base_pose=None):
            pass

        def as_numpy_array(self):
            return np.asarray([np.eye(4, dtype=np.float32)], dtype=np.float32)

    monkeypatch.setattr(gpu_skinning_module, "MatrixPaletteUploader", FakeUploader)

    actor_model = SimpleNamespace(name="PMBAM")
    wrapper = SimpleNamespace(
        name="map_studio_pie_player",
        parent=None,
        _gr_runtime_source_model_ref=actor_model,
    )
    skin = SimpleNamespace(
        name="Torso",
        parent=wrapper,
        children=[],
        is_skin=True,
        _gr_runtime_source_model_id=id(actor_model),
    )
    actor_model.all_nodes = lambda: [skin]
    map_model = SimpleNamespace(name="207TEL", all_nodes=lambda: [wrapper, skin])
    skeleton_buffer = SimpleNamespace(
        data=np.zeros(1, dtype=[("bone_matrices", np.float32, (4, 4))]),
        update_range=lambda *_args, **_kwargs: None,
    )
    record = SimpleNamespace(source=skin, skeleton=SimpleNamespace(bone_matrices_buffer=skeleton_buffer))

    PygfxMeshCache().update_skin_palette(
        record,
        SimpleNamespace(time=0.1, nodes={}),
        model=map_model,
    )

    assert calls["build_models"] == [actor_model]
    assert mesh_render_data_module.runtime_source_model_for_node(skin) is actor_model
    assert skeleton_render_data_module._skinning_palette_model_for_node(skin, map_model) is actor_model


def test_bas_attachment_skin_palette_is_attachment_root_local() -> None:
    root = SimpleNamespace(
        name="head_root",
        parent=None,
        children=[],
        position=(0.0, 0.0, 10.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        _gr_bas_attachment_layer=True,
        _gr_bas_attachment_root=True,
        _gr_bas_attachment_slot="head",
    )
    skin = SimpleNamespace(
        name="Head",
        parent=root,
        children=[],
        is_skin=True,
        _gr_bas_attachment_layer=True,
        _gr_bas_attachment_root_ref=root,
    )
    root.children = [skin]
    palette = np.asarray([np.eye(4, dtype=np.float32)], dtype=np.float32)
    palette[0, 2, 3] = 12.0

    adjusted = skeleton_render_data_module.bas_attachment_root_local_skin_palette(skin, palette, None)

    assert adjusted[0, 2, 3] == pytest.approx(2.0)


def test_bas_attachment_nonskin_face_parts_use_only_matching_head_local_pose() -> None:
    socket = SimpleNamespace(
        name="headhook",
        parent=None,
        children=[],
        position=(10.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )
    root = SimpleNamespace(
        name="p_carthbbh",
        parent=socket,
        children=[],
        position=(0.0, 0.0, 1.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        _gr_bas_attachment_layer=True,
        _gr_bas_attachment_root=True,
        _gr_bas_socket_name="headhook",
        _gr_bas_attachment_source_model_id=123,
    )
    eye = SimpleNamespace(
        name="eye",
        parent=root,
        children=[],
        position=(0.0, 0.0, 2.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        _gr_bas_attachment_layer=True,
        _gr_bas_attachment_root_ref=root,
        _gr_bas_attachment_source_model_id=123,
    )
    socket.children = [root]
    root.children = [eye]

    bind_pose = SimpleNamespace(
        time=0.0,
        nodes={"eye": SimpleNamespace(name="eye", position=(1.0, 0.0, 2.0), rotation=(0.0, 0.0, 0.0, 1.0))},
        _gr_animation_source_model_id=999,
    )
    head_pose = SimpleNamespace(
        time=0.0,
        nodes={"eye": SimpleNamespace(name="eye", position=(1.0, 0.0, 2.0), rotation=(0.0, 0.0, 0.0, 1.0))},
        _gr_animation_source_model_id=123,
    )

    mismatch = mesh_render_data_module.mesh_model_matrix_for_node(eye, anim_pose=bind_pose)
    matched = mesh_render_data_module.mesh_model_matrix_for_node(eye, anim_pose=head_pose)

    assert tuple(np.round(mismatch[:3, 3], 4)) == (10.0, 0.0, 3.0)
    assert tuple(np.round(matched[:3, 3], 4)) == (11.0, 0.0, 3.0)


def test_bas_head_attachment_face_parts_accept_inherited_body_pose_as_local_transform() -> None:
    socket = SimpleNamespace(
        name="headhook",
        parent=None,
        children=[],
        position=(10.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )
    root = SimpleNamespace(
        name="pmha01",
        parent=socket,
        children=[],
        position=(0.0, 0.0, 1.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        _gr_bas_attachment_layer=True,
        _gr_bas_attachment_root=True,
        _gr_bas_attachment_slot="head",
        _gr_bas_socket_name="headhook",
        _gr_bas_attachment_source_model_id=123,
    )
    eye = SimpleNamespace(
        name="eye",
        parent=root,
        children=[],
        position=(0.0, 0.0, 2.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        _gr_bas_attachment_layer=True,
        _gr_bas_attachment_root_ref=root,
        _gr_bas_attachment_source_model_id=123,
    )
    socket.children = [root]
    root.children = [eye]
    inherited_body_pose = SimpleNamespace(
        time=0.0,
        nodes={"eye": SimpleNamespace(name="eye", position=(1.0, 0.0, 2.0), rotation=(0.0, 0.0, 0.0, 1.0))},
        _gr_animation_source_model_id=999,
    )

    matrix = mesh_render_data_module.mesh_model_matrix_for_node(eye, anim_pose=inherited_body_pose)

    assert tuple(np.round(matrix[:3, 3], 4)) == (11.0, 0.0, 3.0)


def test_bas_head_attachment_inherited_pose_includes_head_parent_chain() -> None:
    socket = SimpleNamespace(
        name="headhook",
        parent=None,
        children=[],
        position=(10.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )
    root = SimpleNamespace(
        name="pmha01",
        parent=socket,
        children=[],
        position=(0.0, 0.0, 1.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        _gr_bas_attachment_layer=True,
        _gr_bas_attachment_root=True,
        _gr_bas_attachment_slot="head",
        _gr_bas_socket_name="headhook",
        _gr_bas_attachment_source_model_id=123,
    )
    hturn = SimpleNamespace(
        name="Hturn_g",
        parent=root,
        children=[],
        position=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        _gr_bas_attachment_layer=True,
        _gr_bas_attachment_root_ref=root,
        _gr_bas_attachment_source_model_id=123,
    )
    head = SimpleNamespace(
        name="head_g",
        parent=hturn,
        children=[],
        position=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        _gr_bas_attachment_layer=True,
        _gr_bas_attachment_root_ref=root,
        _gr_bas_attachment_source_model_id=123,
    )
    eye = SimpleNamespace(
        name="eyera",
        parent=head,
        children=[],
        position=(0.0, 0.0, 2.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        _gr_bas_attachment_layer=True,
        _gr_bas_attachment_root_ref=root,
        _gr_bas_attachment_source_model_id=123,
    )
    socket.children = [root]
    root.children = [hturn]
    hturn.children = [head]
    head.children = [eye]
    inherited_body_pose = SimpleNamespace(
        time=0.0,
        nodes={
            "hturn_g": SimpleNamespace(name="Hturn_g", position=(0.0, 1.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0)),
            "head_g": SimpleNamespace(name="head_g", position=(0.0, 2.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0)),
            "eyera": SimpleNamespace(name="eyera", position=(0.0, 0.0, 3.0), rotation=(0.0, 0.0, 0.0, 1.0)),
        },
        _gr_animation_source_model_id=999,
    )

    matrix = mesh_render_data_module.mesh_model_matrix_for_node(eye, anim_pose=inherited_body_pose)

    assert tuple(np.round(matrix[:3, 3], 4)) == (10.0, 3.0, 4.0)


def test_carth_pmha01_bas_head_stays_socket_local_for_pause2() -> None:
    k1_path = os.environ.get("K1_PATH", "")
    if not k1_path or not os.path.isdir(k1_path):
        pytest.skip("K1 install not available")

    from src.adapters.rendering.moderngl_resources import _build_vbo_data
    from src.core.animation.animation_engine import AnimationEngine, SuperModelResolver
    from src.core.assets.resource_manager import ResourceManager
    from src.systems.bas.preview_composer import build_bas_preview_model

    manager = ResourceManager()
    assert manager.set_k1_dir(k1_path)
    SuperModelResolver.configure(manager)
    body = manager.load_model("P_CarthBB", "K1")
    head = manager.load_model("pmha01", "K1")
    assert body is not None
    assert head is not None
    preview = build_bas_preview_model(
        body_model=body,
        attachment_models={"head": head},
        attachment_transforms={"head": {"position": [0, 0, 0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]}},
    )
    engine = AnimationEngine(body)
    assert engine.play("pause2", loop=True, blend=False)
    engine.seek(9.82)
    pose = engine.evaluate(9.82)
    pose._gr_animation_source_model_id = id(body)
    pose._gr_animation_source_model_name = body.name
    pose._gr_animation_name = "pause2"

    def head_bounds(anim_pose):
        for mesh_data in mesh_render_data_module.iter_mesh_render_data(
            preview,
            anim_pose=anim_pose,
            allow_cpu_skinning=False,
            vbo_builder=_build_vbo_data,
        ):
            if getattr(mesh_data.source, "name", "") != "head" or not getattr(mesh_data.source, "_gr_bas_attachment_layer", False):
                continue
            positions = np.asarray(mesh_data.positions, dtype=np.float32)
            world_matrix = np.asarray(mesh_data.world_matrix, dtype=np.float32).reshape(4, 4)
            world = (world_matrix @ np.column_stack([positions, np.ones(len(positions), dtype=np.float32)]).T).T[:, :3]
            return positions, world, mesh_data
        raise AssertionError("pmha01 head skin mesh not found")

    rest_vbo, rest_world, rest_mesh = head_bounds(None)
    anim_vbo, anim_world, anim_mesh = head_bounds(pose)

    assert float(rest_world[:, 2].min()) == pytest.approx(1.50, abs=0.08)
    assert float(rest_world[:, 2].max()) < 1.90
    assert float(anim_world[:, 2].min()) == pytest.approx(1.50, abs=0.08)
    assert float(anim_world[:, 2].max()) < 1.90
    assert rest_mesh.is_skinned is False
    assert anim_mesh.is_skinned is False
    assert anim_mesh.skinning_cpu_fallback is True
    assert float(anim_vbo[:, 2].max()) < 0.35
    assert float(np.linalg.norm(anim_world.max(axis=0) - anim_world.min(axis=0))) < 0.85


def test_carth_pmha01_bas_head_uses_standalone_head_local_tracks_for_pause2() -> None:
    k1_path = os.environ.get("K1_PATH", "")
    if not k1_path or not os.path.isdir(k1_path):
        pytest.skip("K1 install not available")

    from src.adapters.rendering.moderngl_resources import _build_vbo_data
    from src.core.animation.animation_engine import AnimationEngine, SuperModelResolver
    from src.core.assets.resource_manager import ResourceManager
    from src.systems.bas.preview_composer import build_bas_preview_model

    manager = ResourceManager()
    assert manager.set_k1_dir(k1_path)
    SuperModelResolver.configure(manager)
    body = manager.load_model("P_CarthBB", "K1")
    head = manager.load_model("pmha01", "K1")
    assert body is not None
    assert head is not None
    preview = build_bas_preview_model(body_model=body, attachment_models={"head": head})

    body_engine = AnimationEngine(body)
    head_engine = AnimationEngine(head)
    assert body_engine.play("pause2", loop=True, blend=False)
    assert head_engine.play("pause2", loop=True, blend=False)
    body_pose = body_engine.evaluate(9.82)
    body_pose._gr_animation_source_model_id = id(body)
    body_pose._gr_animation_source_model_name = body.name
    body_pose._gr_animation_name = "pause2"
    head_pose = head_engine.evaluate(9.82)
    head_pose._gr_animation_source_model_id = id(head)
    head_pose._gr_animation_source_model_name = head.name
    head_pose._gr_animation_name = "pause2"

    head_nodes = {str(n.name).lower(): n for n in head.all_nodes()}
    bas_nodes = {
        str(n.name).lower(): n
        for n in preview.all_nodes()
        if getattr(n, "_gr_bas_attachment_layer", False)
    }
    standalone_root = head_nodes["pmha01"]
    bas_root = bas_nodes["pmha01"]
    standalone_anchor = head_nodes["necklwr_g"]
    bas_anchor = bas_nodes["necklwr_g"]
    standalone_root_inv = np.linalg.inv(mesh_render_data_module.mesh_model_matrix_for_node(standalone_root, anim_pose=head_pose))
    bas_root_inv = np.linalg.inv(mesh_render_data_module.mesh_model_matrix_for_node(bas_root, anim_pose=body_pose))
    standalone_anchor_inv = np.linalg.inv(
        mesh_render_data_module.mesh_model_matrix_for_node(standalone_anchor, anim_pose=head_pose)
    )
    bas_anchor_inv = np.linalg.inv(mesh_render_data_module.mesh_model_matrix_for_node(bas_anchor, anim_pose=body_pose))

    for name in ("rootdummy", "torso_g", "torsoupr_g"):
        standalone_local = standalone_root_inv @ mesh_render_data_module.mesh_model_matrix_for_node(
            head_nodes[name], anim_pose=head_pose
        )
        bas_local = bas_root_inv @ mesh_render_data_module.mesh_model_matrix_for_node(bas_nodes[name], anim_pose=body_pose)
        assert not np.allclose(bas_local[:3, 3], standalone_local[:3, 3], atol=1.0e-4), name

    for name in ("hturn_g", "head_g", "talkdummy", "eyera", "eyela", "f_jaw_g"):
        standalone_local = standalone_anchor_inv @ mesh_render_data_module.mesh_model_matrix_for_node(
            head_nodes[name], anim_pose=head_pose
        )
        bas_local = bas_anchor_inv @ mesh_render_data_module.mesh_model_matrix_for_node(bas_nodes[name], anim_pose=body_pose)
        assert np.allclose(bas_local[:3, 3], standalone_local[:3, 3], atol=1.0e-4), name

    def head_skin_root_local(model, pose, *, bas_layer: bool):
        for mesh_data in mesh_render_data_module.iter_mesh_render_data(
            model,
            anim_pose=pose,
            allow_cpu_skinning=True,
            vbo_builder=_build_vbo_data,
        ):
            if getattr(mesh_data.source, "name", "") != "head":
                continue
            if bool(getattr(mesh_data.source, "_gr_bas_attachment_layer", False)) != bool(bas_layer):
                continue
            positions = np.asarray(mesh_data.positions, dtype=np.float32)
            world_matrix = np.asarray(mesh_data.world_matrix, dtype=np.float32).reshape(4, 4)
            world = (world_matrix @ np.column_stack([positions, np.ones(len(positions), dtype=np.float32)]).T).T[:, :3]
            root_inv = bas_anchor_inv if bas_layer else standalone_anchor_inv
            return (root_inv @ np.column_stack([world, np.ones(len(world), dtype=np.float32)]).T).T[:, :3]
        raise AssertionError("head skin mesh not found")

    standalone_head = head_skin_root_local(head, head_pose, bas_layer=False)
    attached_head = head_skin_root_local(preview, body_pose, bas_layer=True)

    assert standalone_head.shape == attached_head.shape
    placement_delta = attached_head - standalone_head
    deformation_residual = placement_delta - placement_delta.mean(axis=0, keepdims=True)
    assert float(np.max(np.abs(deformation_residual))) < 1.0e-4


def test_pygfx_optional_scene_camera_light_cube_smoke() -> None:
    gfx = pytest.importorskip("pygfx")

    scene = gfx.Scene()
    camera = gfx.PerspectiveCamera(45.0, 1.0)
    light = gfx.AmbientLight((0.08, 0.08, 0.08), 1.0)
    geometry = gfx.Geometry(
        positions=np.asarray(
            [
                (-0.5, -0.5, 0.0),
                (0.5, -0.5, 0.0),
                (0.0, 0.5, 0.0),
            ],
            dtype=np.float32,
        ),
        indices=np.asarray([[0, 1, 2]], dtype=np.uint32),
    )
    cube = gfx.Mesh(geometry, gfx.MeshPhongMaterial(color=(0.7, 0.8, 0.9, 1.0)))
    scene.add(light, cube)

    assert camera is not None
    assert cube in scene.children
    assert light in scene.children


def test_existing_wgpu_auto_request_uses_direct3d_wgpu_backend() -> None:
    renderer = WgpuRenderer(RendererBackend.WGPU_AUTO, settings=RendererSettings())

    assert renderer.backend_id == RendererBackend.WGPU_D3D12.value


def test_pygfx_diagnostics_request_native_surface_passthrough() -> None:
    renderer = PygfxViewportRenderer(settings=RendererSettings())

    assert renderer.get_diagnostics()["native_surface_passthrough"] is True
    assert renderer.get_diagnostics()["supports_gizmo_drawing"] is False


def test_pygfx_keeps_legacy_overlay_gizmo_and_cpu_picking_contract() -> None:
    caps = PygfxViewportRenderer.probe_availability()
    renderer = PygfxViewportRenderer(settings=RendererSettings())

    assert caps.supports_textures is True
    assert caps.supports_gizmo_drawing is False
    assert caps.supports_gizmo_interaction is True
    assert caps.supports_cpu_ray_picking is True
    assert caps.supports_gpu_id_picking is False
    assert renderer.use_native_gizmo_overlay is False
    assert renderer.use_native_skeleton_overlay is True
    assert renderer.use_native_light_helper_overlay is False


def test_pygfx_camera_uses_z_up_reference() -> None:
    renderer = PygfxViewportRenderer(settings=RendererSettings())
    gfx = pytest.importorskip("pygfx")
    renderer.camera = gfx.PerspectiveCamera(45.0, 1.0)

    renderer._configure_camera_up()

    assert tuple(float(v) for v in renderer.camera.local.reference_up[:3]) == pytest.approx((0.0, 0.0, 1.0))


def test_pygfx_grid_is_lightweight_z_up_line_grid() -> None:
    renderer = PygfxViewportRenderer(settings=RendererSettings())
    gfx = pytest.importorskip("pygfx")

    grid = renderer._create_z_up_grid(gfx, size=100.0, divisions=40)

    positions = np.asarray(grid.geometry.positions.data)
    assert positions.shape[0] == (40 + 1) * 4
    assert np.allclose(positions[:, 2], 0.0)
    assert type(grid.material).__name__ == "LineSegmentMaterial"


def test_viewport_pipeline_keeps_tool_overlay_for_pygfx_live_surface() -> None:
    from src.gui.viewports.viewport_core.widgets.rendering_pipeline import ViewportRenderingPipelineMixin

    source = inspect.getsource(ViewportRenderingPipelineMixin._render_frame)

    assert "_gpu_renderer_requires_native_surface_passthrough" in source
    assert "_draw_live_surface_tool_overlay" in source
    assert "_update_live_surface_diagnostics()" in source


def test_viewport_pipeline_skips_duplicate_cpu_skeleton_overlay_when_native_supported() -> None:
    from src.gui.viewports.viewport_core.widgets.rendering_pipeline import ViewportRenderingPipelineMixin

    overlay_source = inspect.getsource(ViewportRenderingPipelineMixin._draw_gpu_viewport_overlays)
    live_source = inspect.getsource(ViewportRenderingPipelineMixin._draw_live_surface_tool_overlay)
    dirty_source = inspect.getsource(ViewportRenderingPipelineMixin._can_skip_live_overlay_rebuild)
    capability_source = inspect.getsource(ViewportRenderingPipelineMixin._gpu_renderer_supports_native_skeleton_overlay)

    assert "native_skeleton = self._gpu_renderer_supports_native_skeleton_overlay()" in overlay_source
    assert "self._renderer.show_bones and not native_skeleton" in overlay_source
    assert "_can_skip_animation_cpu_overlay()" in live_source
    assert "_can_skip_animation_cpu_overlay()" in dirty_source
    assert "skeleton_overlay_supported" in capability_source


def test_pygfx_animation_uses_native_skinning_and_retained_overlay_contract() -> None:
    scene_source = inspect.getsource(PygfxSceneBridge.update_scene)
    renderer_source = inspect.getsource(PygfxViewportRenderer.render)
    cache_source = inspect.getsource(PygfxMeshCache._build_geometry)

    assert "allow_cpu_skinning=False" in scene_source
    assert "update_skin_palette" in scene_source
    assert "can_update_animation_only" in renderer_source
    assert "update_animation(scene" in renderer_source
    assert "update_skeleton_overlay" in renderer_source
    assert "skin_indices" in cache_source
    assert "skin_weights" in cache_source


def test_pygfx_live_surface_skips_legacy_hover_outline_to_avoid_duplicate_edges() -> None:
    from src.gui.viewports.viewport_core.widgets.drag_interactions import ViewportDragInteractionsMixin

    source = inspect.getsource(ViewportDragInteractionsMixin._draw_hovered_mesh_outline)

    assert "pygfx_wgpu" in source
    assert "canvas.is_live_surface()" in source


def test_pygfx_live_surface_uses_native_helper_overlay_not_screen_markers() -> None:
    from src.gui.viewports.viewport_core.widgets.overlay_layers import ViewportOverlayLayersMixin
    from src.gui.viewports.viewport_core.widgets.rendering_pipeline import ViewportRenderingPipelineMixin

    overlay_source = inspect.getsource(ViewportOverlayLayersMixin._draw_wgpu_helper_markers)
    pipeline_source = inspect.getsource(ViewportRenderingPipelineMixin._render_gpu_frame)

    assert "pygfx_wgpu" in overlay_source
    assert "canvas.is_live_surface()" in overlay_source
    assert "_build_pygfx_helper_render_data()" in pipeline_source
    assert "helper_render_data=helper_render_data" in pipeline_source


def test_pygfx_scene_bridge_builds_native_overlay_lines() -> None:
    from src.core.gizmo.gizmo_draw_data import GizmoDrawCommand, GizmoRenderData

    bridge = PygfxSceneBridge(_FakeGfx, _FakeScene(), PygfxMeshCache())
    gizmo = GizmoRenderData(
        commands=(
            GizmoDrawCommand(
                kind="line",
                points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                colour=(1.0, 0.0, 0.0, 1.0),
                thickness=3.0,
            ),
            GizmoDrawCommand(
                kind="polyline",
                points=((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0)),
                colour=(0.0, 1.0, 0.0, 1.0),
                thickness=2.0,
            ),
        )
    )
    skeleton = SimpleNamespace(
        show_links=True,
        show_dots=True,
        bones=(
            SimpleNamespace(
                visible=True,
                selected=True,
                head_position=(0.0, 0.0, 0.0),
                tail_position=(0.0, 0.0, 1.0),
            ),
        ),
    )

    bridge.update_overlays(gizmo_render_data=gizmo, skeleton_render_data=skeleton)

    assert bridge.gizmo_overlay_segments == 3
    assert bridge.skeleton_overlay_segments == 1
    assert len(bridge._overlay_objects) >= 2


def test_pygfx_scene_bridge_builds_native_helper_points() -> None:
    bridge = PygfxSceneBridge(_FakeGfx, _FakeScene(), PygfxMeshCache())
    helpers = SimpleNamespace(
        helpers=(
            SimpleNamespace(position=(1.0, 2.0, 3.0), selected=False, hovered=False, visible=True),
            SimpleNamespace(position=(2.0, 3.0, 4.0), selected=False, hovered=True, visible=True),
            SimpleNamespace(position=(3.0, 4.0, 5.0), selected=True, hovered=False, visible=True),
        )
    )

    bridge.update_overlays(helper_render_data=helpers)

    point_names = {obj.name for obj in bridge._overlay_objects}
    assert {"pygfx-helper", "pygfx-helper-hover", "pygfx-helper-selected"} <= point_names


def test_pygfx_scene_bridge_installs_default_lighting_when_scene_lights_missing() -> None:
    scene = _FakeScene()
    bridge = PygfxSceneBridge(_FakeGfx, scene, PygfxMeshCache())

    bridge.update_lighting(None)

    assert bridge._ambient_light is not None
    assert bridge._ambient_light.intensity >= 0.55
    assert bridge._default_directional_light is not None
    assert bridge._default_directional_light in scene.children


def test_pygfx_mesh_cache_applies_toolbar_view_style_without_geometry_rebuild() -> None:
    cache = PygfxMeshCache()
    scene = _FakeScene()
    data = _mesh_data()
    record = cache.get_or_create(data, _FakeGfx, scene, selected=False)
    cache.begin_frame()

    cache.apply_view_style(
        show_solid=True,
        show_wireframe=True,
        show_texture=False,
        render_mode="flat",
        xray=True,
    )

    assert cache.geometry_updates_this_frame == 0
    assert type(record.material).__name__ == "MeshBasicMaterial"
    assert record.mesh.visible is True
    assert record.edge_mesh.visible is True
    assert record.material.wireframe is False
    assert record.material.opacity == pytest.approx(0.38)


def test_pygfx_mesh_cache_uses_deduped_line_edges_not_material_wireframe() -> None:
    cache = PygfxMeshCache()
    scene = _FakeScene()
    data = _mesh_data()
    data.positions = np.asarray([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)], dtype=np.float32)
    data.indices = np.asarray([0, 1, 2, 2, 1, 3], dtype=np.uint32)
    record = cache.get_or_create(data, _FakeGfx, scene, selected=False)

    cache.apply_view_style(show_solid=False, show_wireframe=True, wire_color=(0.18, 0.62, 0.95, 1.0))

    assert record.edge_mesh is not None
    assert record.edge_mesh.geometry is not record.geometry
    assert record.edge_mesh.geometry.positions.data.shape == (10, 3)
    assert record.edge_material.color == pytest.approx((0.18, 0.62, 0.95, 1.0))


def test_pygfx_skinned_wire_overlay_uses_skinned_mesh_to_follow_palette() -> None:
    cache = PygfxMeshCache()
    scene = _FakeScene()
    data = _skinned_mesh_data()
    record = cache.get_or_create(data, _FakeGfx, scene, selected=False)

    cache.apply_view_style(show_solid=True, show_wireframe=True, wire_color=(0.18, 0.62, 0.95, 1.0))

    assert record.edge_mesh is not None
    assert isinstance(record.edge_mesh, _FakeSkinnedMesh)
    assert record.edge_mesh.geometry is record.geometry
    assert record.edge_mesh.skeleton is record.skeleton
    assert record.edge_material.wireframe is True
    assert record.edge_material.color == pytest.approx((0.18, 0.62, 0.95, 1.0))


def test_pygfx_animation_only_path_rebuilds_cpu_fallback_attachment_geometry() -> None:
    bridge = PygfxSceneBridge(_FakeGfx, _FakeScene(), PygfxMeshCache())
    data = _mesh_data(source_revision=(8, 1, 0, 250))
    data.skinning_cpu_fallback = True
    bridge.mesh_cache.get_or_create(data, _FakeGfx, bridge.scene, selected=False)

    assert bridge.can_update_animation_only() is False


def test_pygfx_edge_overlay_copies_primary_mesh_transform_when_created_after_sync() -> None:
    cache = PygfxMeshCache()
    scene = _FakeScene()
    record = cache.get_or_create(_mesh_data(), _FakeGfx, scene, selected=False)
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 3] = 12.0
    matrix[1, 3] = -5.0
    matrix[2, 3] = 2.5
    record.mesh.local.matrix = matrix

    cache.apply_view_style(show_solid=True, show_wireframe=True)

    assert record.edge_mesh is not None
    assert np.allclose(record.edge_mesh.local.matrix, matrix)


def test_pygfx_scene_bridge_clear_removes_retained_edge_meshes() -> None:
    cache = PygfxMeshCache()
    scene = _FakeScene()
    bridge = PygfxSceneBridge(_FakeGfx, scene, cache)
    record = cache.get_or_create(_mesh_data(), _FakeGfx, scene, selected=False)
    cache.apply_view_style(show_solid=False, show_wireframe=True)
    edge_mesh = record.edge_mesh

    assert edge_mesh in scene.children

    bridge.clear()

    assert record.mesh not in scene.children
    assert edge_mesh not in scene.children
    assert cache.records == {}


def test_pygfx_hover_and_wire_edges_use_distinct_wgpu_palette_colors() -> None:
    cache = PygfxMeshCache()
    scene = _FakeScene()
    data = _mesh_data()
    record = cache.get_or_create(data, _FakeGfx, scene, selected=False, hovered=True)

    cache.apply_view_style(
        show_solid=True,
        show_wireframe=False,
        show_mesh_hover=True,
        wire_color=(0.18, 0.62, 0.95, 1.0),
        hover_color=(0.0, 215 / 255.0, 181 / 255.0, 0.45),
    )

    assert record.edge_mesh.visible is True
    assert record.edge_material.color == pytest.approx((0.0, 215 / 255.0, 181 / 255.0, 0.45))

    cache.update_selection(_FakeGfx, set(), None)
    cache.apply_view_style(
        show_solid=False,
        show_wireframe=True,
        wire_color=(0.18, 0.62, 0.95, 1.0),
        hover_color=(0.0, 215 / 255.0, 181 / 255.0, 0.45),
    )

    assert record.edge_mesh.visible is True
    assert record.edge_material.color == pytest.approx((0.18, 0.62, 0.95, 1.0))


def test_pygfx_mesh_cache_maps_realistic_shaded_flat_and_mesh_modes() -> None:
    cache = PygfxMeshCache()
    scene = _FakeScene()
    record = cache.get_or_create(_mesh_data(material_revision=("texture",)), _FakeGfx, scene, selected=False)

    cache.begin_frame()
    cache.apply_view_style(show_solid=True, show_wireframe=False, show_texture=True, render_mode="realistic")
    assert type(record.material).__name__ == "MeshPhongMaterial"
    assert record.material.map is record.diffuse_map
    assert record.mesh.visible is True
    assert record.edge_mesh is None or record.edge_mesh.visible is False

    cache.apply_view_style(show_solid=True, show_wireframe=False, show_texture=True, render_mode="shaded")
    assert type(record.material).__name__ == "MeshPhongMaterial"
    assert record.material.map is record.diffuse_map
    assert record.material.flat_shading is True

    cache.apply_view_style(show_solid=True, show_wireframe=False, show_texture=True, render_mode="flat")
    assert type(record.material).__name__ == "MeshBasicMaterial"
    assert record.material.map is record.diffuse_map

    cache.apply_view_style(show_solid=False, show_wireframe=True, show_texture=True, render_mode="realistic")
    assert record.mesh.visible is False
    assert record.edge_mesh is not None
    assert record.edge_mesh.visible is True

    cache.apply_view_style(show_solid=True, show_wireframe=True, show_texture=True, render_mode="realistic")
    assert record.mesh.visible is True
    assert record.edge_mesh.visible is True


def test_pygfx_mesh_cache_uses_retained_texture_map_without_geometry_rebuild() -> None:
    cache = PygfxMeshCache()
    scene = _FakeScene()
    data = _mesh_data(material_revision=("texture",))
    record = cache.get_or_create(data, _FakeGfx, scene, selected=False)

    assert record.diffuse_map is not None
    assert record.material.map is record.diffuse_map
    assert cache.diagnostics()["texture_cache_size"] == 1

    cache.begin_frame()
    cache.apply_view_style(show_texture=False)
    assert cache.geometry_updates_this_frame == 0
    assert record.material.map is None

    cache.apply_view_style(show_texture=True)
    assert cache.geometry_updates_this_frame == 0
    assert record.material.map is record.diffuse_map

    cache.apply_view_style(show_texture=True, show_diffuse=False)
    assert cache.geometry_updates_this_frame == 0
    assert record.material.map is None


def test_pygfx_mesh_cache_bakes_sprite_luminance_matte_into_texture_alpha() -> None:
    cache = PygfxMeshCache()
    scene = _FakeScene()
    data = _mesh_data(material_revision=("texture",))
    data.material.alpha_mode = "BLEND"
    data.material.blend_mode = "LIGHTEN"
    data.material.sprite_alpha_source = 1
    data.material.sprite_glow = 1.6
    source = Image.new("RGBA", (2, 1))
    source.putdata([(0, 0, 0, 255), (200, 20, 20, 255)])
    data.material.diffuse_texture_data = SimpleNamespace(
        texture_id="sprite_diffuse",
        name="sprite_diffuse",
        source=source,
        source_revision=(id(source), 2, 1),
    )

    record = cache.get_or_create(data, _FakeGfx, scene, selected=False)
    pixels = record.material.map.texture.data

    assert int(pixels[0, 0, 3]) == 0
    assert int(pixels[0, 1, 3]) > 200
    assert record.material.alpha_mode == "blend"
    assert record.material.depth_write is False

    cache.apply_view_style(show_solid=True, show_wireframe=False, show_texture=True, render_mode="realistic")
    assert record.sprite_proxy_mesh is not None
    assert record.sprite_proxy_mesh.visible is True
    assert record.mesh.visible is False


def test_pygfx_mesh_cache_uses_retained_lightmap_uv1_channel() -> None:
    cache = PygfxMeshCache()
    scene = _FakeScene()
    data = _mesh_data(material_revision=("lightmap",))
    data.uvs0 = np.asarray([(0.0, 0.0), (0.5, 0.25), (1.0, 1.0)], dtype=np.float32)
    data.uvs1 = np.asarray([(0.0, 0.1), (0.5, 0.6), (1.0, 0.9)], dtype=np.float32)
    record = cache.get_or_create(data, _FakeGfx, scene, selected=False)

    assert record.diffuse_map is not None
    assert record.lightmap_map is not None
    assert record.lightmap_map.kwargs["uv_channel"] == 1

    cache.begin_frame()
    cache.apply_view_style(show_texture=True, show_lightmap=True)
    assert record.material.light_map is record.lightmap_map
    assert record.material.light_map_intensity == pytest.approx(0.55)

    cache.apply_view_style(show_texture=True, show_lightmap=False)
    assert record.material.light_map is None


def test_pygfx_mesh_cache_applies_explicit_culling_without_changing_default() -> None:
    cache = PygfxMeshCache()
    scene = _FakeScene()
    record = cache.get_or_create(_mesh_data(material_revision=("texture",)), _FakeGfx, scene, selected=False)

    cache.apply_view_style(cull_faces=False)
    assert record.material.side == "both"

    cache.apply_view_style(cull_faces=True)
    assert record.material.side == "front"

    data = _mesh_data(mesh_id=8, material_revision=("texture",))
    data.material.double_sided = True
    double_sided = cache.get_or_create(data, _FakeGfx, scene, selected=False)
    cache.apply_view_style(cull_faces=True)
    assert double_sided.material.side == "both"


def test_pygfx_grid_toggle_updates_native_helper_visibility() -> None:
    renderer = PygfxViewportRenderer(settings=RendererSettings())
    renderer._grid_helper = SimpleNamespace(visible=True)
    renderer.show_grid = False

    renderer._apply_view_state()

    assert renderer._grid_helper.visible is False


def test_pygfx_renderer_applies_theme_tokens_to_scene_helpers() -> None:
    class Theme:
        def color(self, key, fallback=""):
            return {
                "viewport.background": "#112233",
                "viewport.gridMinor": "#445566",
            }.get(key, fallback)

    renderer = PygfxViewportRenderer(settings=RendererSettings())
    renderer._gfx = _FakeGfx
    renderer.scene = _FakeScene()
    renderer._install_empty_scene_helpers(_FakeGfx)

    renderer.set_theme_colors(Theme())

    assert renderer.viewport_background == pytest.approx((0x11 / 255.0, 0x22 / 255.0, 0x33 / 255.0))
    assert renderer.grid_minor_color == pytest.approx((0x44 / 255.0, 0x55 / 255.0, 0x66 / 255.0))
    assert renderer._background.color == pytest.approx((*renderer.viewport_background, 1.0))
    assert renderer._grid_helper.material.color == pytest.approx((*renderer.grid_minor_color, 1.0))


def test_renderer_surface_host_keeps_live_surface_visible() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets
    from src.gui.viewports.viewport_host import RendererSurfaceHost

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    host = RendererSurfaceHost()
    surface = QtWidgets.QWidget()

    host.resize(320, 200)
    host.set_renderer_surface(surface, backend_id=RendererBackend.PYGFX_WGPU.value, live_surface=True)
    host.show()
    app.processEvents()

    assert host.current_surface() is surface
    assert host.is_live_surface() is True
    assert surface.isVisible() is True
    assert host.overlay_layer_active() is False


class _TextureUpdateSoftwareRenderer(RendererTextureMixin):
    def __init__(self) -> None:
        self.textures = {}
        self.tex_cache = TextureCache()
        self._tex_arr_cache = TexArrayCache(max_entries=8)


class _TextureUpdateFakeGlTexture:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self.writes: list[dict] = []
        self.mipmap_levels: list[int] = []
        self.release_count = 0
        self.filter = (
            moderngl_resources_module.moderngl.LINEAR_MIPMAP_LINEAR,
            moderngl_resources_module.moderngl.LINEAR,
        )

    def write(self, data, *, viewport, alignment) -> None:
        self.writes.append(
            {
                "data": bytes(data),
                "viewport": tuple(viewport),
                "alignment": int(alignment),
            }
        )

    def build_mipmaps(self, *, max_level) -> None:
        self.mipmap_levels.append(int(max_level))

    def release(self) -> None:
        self.release_count += 1


def test_texture_cache_patches_clipped_region_without_flipping_or_replacing_image() -> None:
    cache = TextureCache()
    original = Image.new("RGBA", (4, 4), (10, 20, 30, 255))
    cached, initial_regions = cache.update_image_regions("paint_live", original)
    assert cached is original
    assert initial_regions == ((0, 0, 4, 4),)

    assert cache.get_mip1(cached) is not None
    assert id(cached) in cache._mip_bias_cache

    incoming = Image.new("RGBA", (4, 4), (200, 5, 6, 255))
    updated, regions = cache.update_image_regions(
        "paint_live.tga",
        incoming,
        [(-1, 0, 3, 1), (99, 99, 4, 4)],
    )

    assert updated is cached
    assert regions == ((0, 0, 2, 1),)
    # Cached KOTOR images are bottom-up: y=0 remains y=0 through the API.
    assert updated.getpixel((0, 0)) == (200, 5, 6, 255)
    assert updated.getpixel((1, 0)) == (200, 5, 6, 255)
    assert updated.getpixel((0, 3)) == (10, 20, 30, 255)
    assert id(updated) not in cache._mip_bias_cache


def test_software_texture_update_evicts_only_edited_numpy_array() -> None:
    renderer = _TextureUpdateSoftwareRenderer()
    painted = Image.new("RGBA", (8, 8), (1, 2, 3, 255))
    unrelated = Image.new("RGBA", (8, 8), (4, 5, 6, 255))
    renderer.update_texture_regions("painted", painted)
    painted_array_before = renderer._tex_arr_cache.get(painted)
    unrelated_array_before = renderer._tex_arr_cache.get(unrelated)

    painted.putpixel((2, 0), (90, 80, 70, 255))
    updated, regions = renderer.update_texture_regions("painted", painted, [(2, 0, 1, 1)])

    assert updated is painted
    assert regions == ((2, 0, 1, 1),)
    assert renderer._tex_arr_cache.get(painted) is not painted_array_before
    assert renderer._tex_arr_cache.get(unrelated) is unrelated_array_before


def test_moderngl_region_upload_writes_only_dirty_bytes_and_never_clears_cache() -> None:
    image = Image.new("RGBA", (512, 512), (15, 25, 35, 255))
    image.paste((200, 100, 50, 255), (10, 20, 14, 24))
    cache = _GlTexCache(SimpleNamespace())
    texture = _TextureUpdateFakeGlTexture(image.size)
    cache._cache[id(image)] = (weakref.ref(image), texture)
    cache.clear = lambda: (_ for _ in ()).throw(AssertionError("full cache clear invoked"))

    assert cache.update_regions(image, [(10, 20, 4, 4)]) is True

    assert texture.writes == [
        {
            "data": bytes((200, 100, 50, 255)) * 16,
            "viewport": (10, 20, 4, 4),
            "alignment": 1,
        }
    ]
    assert texture.mipmap_levels == [6]
    assert texture.release_count == 0
    assert cache.region_update_count == 1
    assert cache.region_update_bytes == 4 * 4 * 4
    assert cache.region_update_bytes < 512 * 512 * 4


def test_moderngl_live_paint_defers_mip_rebuild_until_stroke_finalize() -> None:
    image = Image.new("RGBA", (512, 512), (15, 25, 35, 255))
    cache = _GlTexCache(SimpleNamespace())
    texture = _TextureUpdateFakeGlTexture(image.size)
    cache._cache[id(image)] = (weakref.ref(image), texture)

    assert cache.update_regions(image, [(10, 20, 4, 4)], build_mipmaps=False) is True
    assert texture.writes
    assert texture.mipmap_levels == []
    assert texture.filter == (
        moderngl_resources_module.moderngl.LINEAR,
        moderngl_resources_module.moderngl.LINEAR,
    )
    assert cache.update_regions(image, (), build_mipmaps=True) is True
    assert texture.mipmap_levels == [6]
    assert texture.filter == (
        moderngl_resources_module.moderngl.LINEAR_MIPMAP_LINEAR,
        moderngl_resources_module.moderngl.LINEAR,
    )


def test_moderngl_reports_live_texture_streaming_capability() -> None:
    from src.adapters.rendering.moderngl_renderer import ModernGLRenderer

    assert ModernGLRenderer().get_capabilities().supports_texture_streaming is True


def test_renderer_factory_delegates_region_update_without_cache_clear() -> None:
    class Backend:
        def __init__(self) -> None:
            self.updated = []
            self.clear_count = 0

        def update_texture_regions(self, texture_name, image, regions, *, finalize=True) -> bool:
            self.updated.append((texture_name, image, tuple(regions), bool(finalize)))
            return True

        def clear_caches(self) -> None:
            self.clear_count += 1

    backend = Backend()
    renderer = FallbackViewportRenderer()
    object.__setattr__(renderer, "_active", backend)
    image = Image.new("RGBA", (4, 4))

    assert renderer.update_texture_regions("live", image, ((0, 0, 1, 1),)) is True
    assert backend.updated == [("live", image, ((0, 0, 1, 1),), True)]
    assert backend.clear_count == 0


def test_renderer_factory_does_not_hide_live_texture_backend_typeerror() -> None:
    class Backend:
        def update_texture_regions(self, texture_name, image, regions, *, finalize=True) -> bool:
            raise TypeError("backend texture defect")

    renderer = FallbackViewportRenderer()
    object.__setattr__(renderer, "_active", Backend())

    with pytest.raises(TypeError, match="backend texture defect"):
        renderer.update_texture_regions("live", Image.new("RGBA", (2, 2)), (), finalize=False)


def test_wgpu_texture_update_fallback_invalidates_only_named_texture_and_materials() -> None:
    cache = WgpuResourceCache(SimpleNamespace())
    painted_image = Image.new("RGBA", (4, 4))
    painted = SimpleNamespace(byte_size=64, fallback=False, lightmap=False)
    other = SimpleNamespace(byte_size=64, fallback=False, lightmap=False)
    cache.textures = {
        f"painted:{id(painted_image)}": painted,
        "other:22": other,
    }
    cache.materials = {
        "uses-painted": SimpleNamespace(diffuse_texture_resource=painted, lightmap_texture_resource=None),
        "uses-other": SimpleNamespace(diffuse_texture_resource=other, lightmap_texture_resource=None),
    }

    assert cache.invalidate_texture_source("painted", image=painted_image) is True
    assert cache.textures == {"other:22": other}
    assert tuple(cache.materials) == ("uses-other",)


def test_pygfx_texture_update_fallback_dirties_only_records_using_named_texture() -> None:
    cache = PygfxMeshCache()
    painted_image = Image.new("RGBA", (4, 4))
    painted_map = SimpleNamespace()
    other_map = SimpleNamespace()
    cache.texture_maps = {
        (f"painted:{id(painted_image)}", (), 0, False, 0): painted_map,
        ("other:22", (), 0, False, 0): other_map,
    }
    painted_record = SimpleNamespace(
        diffuse_map=painted_map,
        lightmap_map=None,
        material_dirty=False,
        view_style_key=(1,),
    )
    other_record = SimpleNamespace(
        diffuse_map=other_map,
        lightmap_map=None,
        material_dirty=False,
        view_style_key=(1,),
    )
    cache.records = {1: painted_record, 2: other_record}

    assert cache.invalidate_texture("painted", image=painted_image) is True
    assert tuple(cache.texture_maps) == (("other:22", (), 0, False, 0),)
    assert painted_record.material_dirty is True
    assert painted_record.view_style_key == ()
    assert other_record.material_dirty is False


def test_pygfx_live_texture_update_patches_resident_pixels_without_scene_sync() -> None:
    cache = PygfxMeshCache()
    scene = _FakeScene()
    data = _mesh_data(material_revision=("texture",))
    record = cache.get_or_create(data, _FakeGfx, scene, selected=False)
    painted_image = data.material.diffuse_texture_data.source
    painted_image.putpixel((1, 0), (4, 90, 180, 255))

    assert cache.update_texture_regions("unit_diffuse", painted_image, ((1, 0, 1, 1),)) is True
    assert tuple(record.diffuse_map.texture.data[0, 1]) == (4, 90, 180, 255)
    assert record.diffuse_map.texture.update_ranges == [((1, 0, 0), (1, 1, 1))]

    renderer = PygfxViewportRenderer(settings=RendererSettings())
    model = SimpleNamespace()
    renderer._active_model_id = id(model)
    renderer.mesh_cache.records[1] = SimpleNamespace()
    assert renderer._needs_full_scene_sync(id(model), {"texture": True}) is False


def test_qt_viewport_texture_update_requests_redraw_without_scene_invalidation() -> None:
    class Software:
        def update_texture_regions(self, name, image, regions):
            return image, tuple(regions)

    class Gpu:
        def __init__(self) -> None:
            self.clear_count = 0
            self.calls = []

        def update_texture_regions(self, name, image, regions, *, finalize=True) -> bool:
            self.calls.append((name, image, tuple(regions), bool(finalize)))
            return True

        def clear_caches(self) -> None:
            self.clear_count += 1

    class Viewport(ViewportResourceCacheMixin):
        def __init__(self) -> None:
            self._renderer = Software()
            self._gpu_renderer = Gpu()
            self.redraws = []

        def _request_render(self, **kwargs) -> None:
            self.redraws.append(kwargs)

    viewport = Viewport()
    image = Image.new("RGBA", (8, 8))

    cached, regions = viewport.update_texture_regions("painted", image, [(1, 2, 3, 4)])

    assert cached is image
    assert regions == ((1, 2, 3, 4),)
    assert viewport._gpu_renderer.clear_count == 0
    assert viewport.redraws == [
        {
            "fast": True,
            "reason": "texture pixels changed: painted",
            "texture": True,
        }
    ]


def test_qt_viewport_does_not_hide_live_texture_backend_typeerror() -> None:
    class Software:
        def update_texture_regions(self, name, image, regions):
            return image, tuple(regions or ())

    class Gpu:
        def update_texture_regions(self, name, image, regions, *, finalize=True) -> bool:
            raise TypeError("gpu texture defect")

    viewport = SimpleNamespace(
        _renderer=Software(),
        _gpu_renderer=Gpu(),
        _request_render=lambda **_kwargs: None,
    )

    with pytest.raises(TypeError, match="gpu texture defect"):
        ViewportResourceCacheMixin.update_texture_regions(
            viewport,
            "painted",
            Image.new("RGBA", (2, 2)),
            ((0, 0, 1, 1),),
            finalize=False,
        )
