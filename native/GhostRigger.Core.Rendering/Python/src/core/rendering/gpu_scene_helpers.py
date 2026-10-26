"""Backend-owned scene helpers shared by GPU render adapters."""

from __future__ import annotations

import logging
import math

import numpy as np

from src.core.geometry.model_data import KOTOR_BASE_SKELETONS
from src.core.rendering.mesh_render_data import _node_has_renderable_rigid_surface
from src.math.gpu_math import _quat_rotate_batch

log = logging.getLogger(__name__)


def _compute_model_bounds(model) -> dict:
    """
    D20-M: Walk all renderable mesh nodes and return the world-space AABB,
    using the per-node vertex_space contract (same as _build_vbo_data).

    Returns dict with keys: min_x, max_x, min_y, max_y, min_z, max_z,
    center_x/y/z, extent_x/y/z, max_extent, radius.
    Falls back to the model's stored bb_min/bb_max if no vertex data is found.

    No centroid heuristics, no _WORLDSPACE_THRESHOLD, no accessory detection.
    """
    _bounds_model_cls = (str(getattr(model, "classification", "character") or "character")).lower()
    _bounds_model_type_raw = getattr(model, "model_type", None)
    _bounds_model_type = int(_bounds_model_type_raw) if _bounds_model_type_raw is not None else 4
    _bounds_is_module = _bounds_model_cls in ("effect", "tile", "other") or _bounds_model_type in (0, 2)

    all_pts = []

    _all_nodes_fn = getattr(model, "all_nodes", None)
    nodes = list(_all_nodes_fn()) if _all_nodes_fn else getattr(model, "nodes", [])

    for node in nodes:
        if getattr(node, "render", True) is False:
            continue
        verts = getattr(node, "vertices", None) or []
        if not verts:
            continue
        uvs = getattr(node, "uvs", []) or []
        has_uvs = bool(uvs)

        if not has_uvs and not _bounds_is_module:
            continue
        if (
            node.name.lower().endswith(("_g", "_g0", "_dum"))
            and not _node_has_renderable_rigid_surface(node)
        ):
            continue
        try:
            v_arr = np.array(verts, dtype=np.float64)
        except Exception:
            continue

        _node_vs = (
            1
            if bool(getattr(node, "_gr_vertices_in_kotor_world", False))
            else getattr(node, "vertex_space", 0)
        )

        if _node_vs == 0:
            try:
                wt = node.world_transform()
                wp = np.array(wt[0], dtype=np.float64)
                wo = np.array(wt[1], dtype=np.float64)
                qlen = np.linalg.norm(wo)
                if qlen > 1e-9:
                    wo = wo / qlen
                is_id_rot = (
                    abs(wo[3]) > 0.9999
                    and abs(wo[0]) < 1e-6
                    and abs(wo[1]) < 1e-6
                    and abs(wo[2]) < 1e-6
                )
                if not is_id_rot:
                    v_arr = _quat_rotate_batch(wo, v_arr)
                v_arr = v_arr + wp
            except Exception:
                pass
        elif _node_vs == 1:
            pass
        elif _node_vs == 2:
            continue

        all_pts.append(v_arr)

    if all_pts:
        all_v = np.vstack(all_pts)
        mn_x, mn_y, mn_z = float(all_v[:, 0].min()), float(all_v[:, 1].min()), float(all_v[:, 2].min())
        mx_x, mx_y, mx_z = float(all_v[:, 0].max()), float(all_v[:, 1].max()), float(all_v[:, 2].max())
    else:
        bb_min = getattr(model, "bb_min", (-1, -1, -1)) or (-1, -1, -1)
        bb_max = getattr(model, "bb_max", (1, 1, 1)) or (1, 1, 1)
        mn_x, mn_y, mn_z = float(bb_min[0]), float(bb_min[1]), float(bb_min[2])
        mx_x, mx_y, mx_z = float(bb_max[0]), float(bb_max[1]), float(bb_max[2])

    cx = (mn_x + mx_x) * 0.5
    cy = (mn_y + mx_y) * 0.5
    cz = (mn_z + mx_z) * 0.5
    ext_x = mx_x - mn_x
    ext_y = mx_y - mn_y
    ext_z = mx_z - mn_z
    max_ext = max(ext_x, ext_y, ext_z, 0.01)
    radius = math.sqrt(ext_x**2 + ext_y**2 + ext_z**2) * 0.5

    return {
        "min_x": mn_x,
        "max_x": mx_x,
        "min_y": mn_y,
        "max_y": mx_y,
        "min_z": mn_z,
        "max_z": mx_z,
        "center_x": cx,
        "center_y": cy,
        "center_z": cz,
        "extent_x": ext_x,
        "extent_y": ext_y,
        "extent_z": ext_z,
        "max_extent": max_ext,
        "radius": radius,
    }


def _apply_txi_from_textures_to_model(model, textures: dict) -> None:
    """Apply TXI metadata extracted from texture sources to model nodes."""
    if not textures:
        return
    from src.core.graphics.tpc import _extract_txi_from_tpc
    from src.core.graphics.txi import _apply_txi_to_node

    _txi_cache: dict = {}
    _at_cache: dict = {}
    for tex_name, tex_obj in textures.items():
        _img_at = getattr(tex_obj, "_txi_alpha_test", None)
        if _img_at is not None:
            try:
                _at_v = float(_img_at)
                if 0.0 < _at_v <= 1.0:
                    _at_cache[tex_name] = _at_v
            except (TypeError, ValueError):
                pass

        if tex_name in _txi_cache:
            continue
        txi_str = getattr(tex_obj, "_txi_str", None)
        if txi_str:
            _txi_cache[tex_name] = txi_str
            continue
        raw = getattr(tex_obj, "_tpc_raw", None)
        if raw:
            try:
                txi_str = _extract_txi_from_tpc(raw)
                if txi_str:
                    _txi_cache[tex_name] = txi_str
                if tex_name not in _at_cache and len(raw) >= 8:
                    import struct as _st

                    _at_v = _st.unpack_from("<f", raw, 4)[0]
                    if 0.0 < _at_v <= 1.0:
                        _at_cache[tex_name] = _at_v
            except Exception:
                pass

    try:
        all_nodes_fn = getattr(model, "all_nodes", None)
        nodes = list(all_nodes_fn()) if all_nodes_fn else getattr(model, "nodes", [])
        for node in nodes:
            if not getattr(node, "is_mesh", False):
                continue
            _all_tex_names = set()
            tex_name = str(getattr(node, "texture", "") or "").strip().lower()
            if tex_name and tex_name not in ("null", "", "none"):
                _all_tex_names.add(tex_name)
            for _tn in getattr(node, "texture_names", []):
                _tn_clean = str(_tn or "").strip().lower()
                if _tn_clean and _tn_clean not in ("null", "", "none"):
                    _all_tex_names.add(_tn_clean)

            if not _all_tex_names:
                continue

            _primary_txi = _txi_cache.get(tex_name, "") if tex_name else ""
            _primary_at = _at_cache.get(tex_name, float(getattr(node, "txi_alpha_test", 0.5)))
            if _primary_txi or _primary_at != 0.5:
                try:
                    _apply_txi_to_node(node, _primary_txi, _primary_at)
                except Exception:
                    pass

            for _sec_name in _all_tex_names:
                if _sec_name == tex_name:
                    continue
                _sec_txi = _txi_cache.get(_sec_name, "")
                if _sec_txi:
                    try:
                        _apply_txi_to_node(node, _sec_txi, float(getattr(node, "txi_alpha_test", 0.5)))
                    except Exception:
                        pass
    except Exception as e:
        log.debug("_apply_txi_from_textures_to_model error: %s", e)


class _CompositeModel:
    """Lightweight wrapper that merges two KotorModel objects for combined rendering."""

    def __init__(self, head_model, body_model):
        self._head = head_model
        self._body = body_model

        def _bb(m, attr, default):
            v = getattr(m, attr, None)
            return v if v is not None else default

        h_min = _bb(head_model, "bb_min", (-1, -1, -1))
        h_max = _bb(head_model, "bb_max", (1, 1, 1))
        b_min = _bb(body_model, "bb_min", (-1, -1, -1))
        b_max = _bb(body_model, "bb_max", (1, 1, 1))
        self.bb_min = (min(h_min[0], b_min[0]), min(h_min[1], b_min[1]), min(h_min[2], b_min[2]))
        self.bb_max = (max(h_max[0], b_max[0]), max(h_max[1], b_max[1]), max(h_max[2], b_max[2]))
        dx = self.bb_max[0] - self.bb_min[0]
        dy = self.bb_max[1] - self.bb_min[1]
        dz = self.bb_max[2] - self.bb_min[2]
        self.radius = math.sqrt(dx * dx + dy * dy + dz * dz) * 0.5

        self._nonskin_head_offset = (0.0, 0.0, 0.0)
        try:
            _ANCHOR_BONES = ("head_g", "Hturn_g", "neck_g", "necklwr_g", "headhook")

            _body_bones: dict = {}
            _body_all = list(body_model.all_nodes()) if hasattr(body_model, "all_nodes") else list(getattr(body_model, "nodes", []))
            for _bn in _body_all:
                try:
                    _bwp, _ = _bn.world_transform()
                    _body_bones[_bn.name.lower()] = _bwp
                except Exception:
                    pass

            _head_bones: dict = {}
            _head_all = list(head_model.all_nodes()) if hasattr(head_model, "all_nodes") else list(getattr(head_model, "nodes", []))
            for _hn in _head_all:
                try:
                    _hwp, _ = _hn.world_transform()
                    _head_bones[_hn.name.lower()] = _hwp
                except Exception:
                    pass

            for _anchor in _ANCHOR_BONES:
                _bwp = _body_bones.get(_anchor)
                _hwp = _head_bones.get(_anchor)
                if _bwp is not None and _hwp is not None:
                    self._nonskin_head_offset = (
                        float(_bwp[0]) - float(_hwp[0]),
                        float(_bwp[1]) - float(_hwp[1]),
                        float(_bwp[2]) - float(_hwp[2]),
                    )
                    log.debug(
                        "_CompositeModel: anchor bone %r body=%s head=%s offset=%s",
                        _anchor,
                        tuple(round(x, 3) for x in _bwp),
                        tuple(round(x, 3) for x in _hwp),
                        tuple(round(x, 3) for x in self._nonskin_head_offset),
                    )
                    break
        except Exception as _e:
            log.debug("_CompositeModel: could not compute non-skin offset: %s", _e)

    def __getattr__(self, name):
        return getattr(self._head, name)

    def all_nodes(self):
        """Return head nodes first, then body nodes."""
        _h_fn = getattr(self._head, "all_nodes", None)
        head_nodes = list(_h_fn()) if _h_fn else list(getattr(self._head, "nodes", []))

        _b_fn = getattr(self._body, "all_nodes", None)
        body_nodes = list(_b_fn()) if _b_fn else list(getattr(self._body, "nodes", []))

        for _bn in body_nodes:
            try:
                _bn._model_ref = self._body
            except (AttributeError, TypeError):
                pass

        _off = self._nonskin_head_offset
        for _hn in head_nodes:
            try:
                _hn._model_ref = self._head
            except (AttributeError, TypeError):
                pass
            try:
                _hn._composite_nonskin_offset = _off
            except (AttributeError, TypeError):
                pass

        return head_nodes + body_nodes

    @property
    def nodes(self):
        return self.all_nodes()


_BASE_SKELETONS: frozenset = KOTOR_BASE_SKELETONS

__all__ = tuple(name for name in globals() if not name.startswith("__"))
