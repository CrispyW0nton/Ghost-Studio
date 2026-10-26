"""
GhostRigger MCP Debug Skinning Bridge — Observability-First Runtime Inspection.

Provides MCP tool commands for inspecting runtime skinning/animation data
in the GhostRigger engine.  Designed to expose internal state *before*
fixing animations — enabling data-driven diagnosis instead of guesswork.

Architecture (Khononov, "Balancing Coupling in Software Design"):
  • Each handler depends on the headless runtime session (_DebugSession)
  • The session wraps KotorLoader, AnimationEngine, GpuRenderer, and
    MatrixPaletteUploader — all accessed through stable internal APIs
  • Tool outputs are pure JSON data contracts — no GUI state leaks

Commands implemented (23 total):
  Lifecycle:      launch_app, close_app, get_runtime_status
  Game Library:   set_game_library_path, verify_game_library
  Model:          load_model_by_resref, get_loaded_asset_info
  Animation:      list_animations, set_animation, set_animation_time,
                  set_bind_pose
  Camera/Render:  set_camera_preset, capture_viewport,
                  capture_named_validation_set
  Skinning:       get_skinning_state, get_renderer_state,
                  get_bone_hierarchy, get_bone_map_for_selected_mesh,
                  get_palette_remap_table, get_bind_pose_matrices,
                  get_animated_pose_matrices, get_uploaded_skinning_palette,
                  sample_vertex_influences
  Comparison:     compare_cpu_gpu_skinning
  Export:         export_debug_bundle
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import struct
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── Ensure project root on path ──────────────────────────────────────────────
_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from kotormcp.utils import json_content

import re as _re

_MAX_BIF_RESOURCE = 256 * 1024 * 1024
_MIN_CAPTURE = 16
_MAX_CAPTURE = 4096
_SAFE_SET_NAME = _re.compile(r"[A-Za-z0-9_-]{1,64}")
_UNSAFE_FILE_CHARS = _re.compile(r"[^A-Za-z0-9_-]")
_SAFE_RESREF = _re.compile(r"[A-Za-z0-9_-]{1,16}")
_MAX_ANIMATION_NAME = 64
_MAX_PATH_TEXT = 1024


def _bytes_identity(data: Optional[bytes]) -> Optional[Dict[str, Any]]:
    """SHA-256 and size, the same shape as ghostrigger_source_identity."""
    if data is None:
        return None
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


# ─────────────────────────────────────────────────────────────────────────────
#  Headless Debug Session — Singleton
# ─────────────────────────────────────────────────────────────────────────────

class _DebugSession:
    """Manages a headless GhostRigger runtime for debug inspection.

    Holds:
      - game library path + verified state
      - loaded model + metadata
      - animation engine + current pose
      - GpuRenderer + MatrixPaletteUploader
      - last rendered image
    """

    def __init__(self):
        self.game_dir: Optional[str] = None
        self.game_verified: bool = False
        self.model = None
        self.model_resref: str = ""
        self.model_path: str = ""
        self.anim_engine = None
        self.renderer = None
        self.skin_uploader = None
        self.current_anim_name: Optional[str] = None
        self.current_anim_time: float = 0.0
        self.current_pose = None
        # FIX-SKIN-ANIM-D2: Store the animation's first-frame (t=0) pose
        # as the bind reference for GPU skinning palette computation.
        self.anim_base_pose = None
        self.last_image = None
        self.camera_preset: str = "front"
        self.camera_azimuth: float = 90.0
        self.camera_elevation: float = 15.0
        self.started: bool = False
        self._start_time: float = 0.0
        # BIF extraction cache
        self._key_data: Optional[dict] = None
        # FIX-TEXLOAD-D5: ResourceManager for texture loading pipeline.
        # Previously textures={} was passed to the renderer, causing all
        # models to render without textures (flat white/grey surfaces).
        self._resource_manager = None
        self._model_textures: dict = {}  # cached: name→PIL Image
        # K1 or K2, detected from the executable in the game directory.
        self.game_tag: str = "K1"
        # Session identity (new UUID per launch) and the loaded model's exact
        # source bytes; state_revision() binds both to the observable state.
        self.session_id: str = ""
        self.model_source: Optional[dict] = None

    def launch(self):
        # Relaunching replaces the session instead of leaking a second
        # GpuRenderer beside the first.
        if self.started or self.renderer is not None:
            self.close()
        self.started = True
        self._start_time = time.time()
        self.session_id = str(uuid.uuid4())
        # Lazy-init renderer
        try:
            from src.adapters.rendering.moderngl_legacy_bridge import GpuRenderer
            self.renderer = GpuRenderer()
        except Exception as e:
            log.warning(f"DebugSession: GpuRenderer init failed: {e}")
            self.renderer = None

    def close(self):
        if self.renderer:
            try:
                self.renderer.release()
            except Exception:
                pass
        self.__init__()

    @property
    def uptime_s(self) -> float:
        if not self.started:
            return 0.0
        return time.time() - self._start_time

    def state_revision(self) -> str:
        """SHA-256 over everything that decides a capture: session, install,
        model source bytes, animation, time and camera. Identical state gives
        the identical revision (a validation set that restores its state keeps
        it); Kaigen compares it before every mutation and capture."""
        state = {
            "session_id": self.session_id,
            "started": self.started,
            "game_dir": self.game_dir,
            "game_tag": self.game_tag,
            "model_resref": self.model_resref if self.model is not None else "",
            "model_source": self.model_source if self.model is not None else None,
            "animation": self.current_anim_name,
            "time": repr(float(self.current_anim_time)),
            "camera": [self.camera_preset, self.camera_azimuth, self.camera_elevation],
        }
        return hashlib.sha256(
            json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def status(self) -> dict:
        return {
            "started": self.started,
            "session_id": self.session_id or None,
            "revision": self.state_revision() if self.started else None,
            "uptime_s": round(self.uptime_s, 1),
            "game_dir": self.game_dir,
            "game_tag": self.game_tag if self.game_dir else None,
            "game_verified": self.game_verified,
            "model_loaded": self.model is not None,
            "model_resref": self.model_resref,
            "model_source": self.model_source if self.model is not None else None,
            "current_animation": self.current_anim_name,
            "current_time": self.current_anim_time,
            "camera_preset": self.camera_preset,
            "renderer_available": self.renderer is not None,
            "skin_uploader_active": self.skin_uploader is not None,
        }

    # ── Game library ─────────────────────────────────────────────────────────

    def detect_game_tag(self) -> str:
        """Return 'K2' when the directory holds swkotor2.exe, else 'K1'."""
        if self.game_dir and os.path.isfile(os.path.join(self.game_dir, "swkotor2.exe")):
            return "K2"
        return "K1"

    def set_game_path(self, path: str) -> dict:
        self.game_dir = path
        self.game_verified = False
        self._key_data = None
        self._model_textures = {}  # clear texture cache on path change
        self._resource_manager = None
        self.game_tag = self.detect_game_tag()
        # FIX-TEXLOAD-D5: Initialize ResourceManager for texture pipeline.
        # TSL installs are indexed as K2 (previously always set_k1_dir).
        try:
            from src.core.assets.resource_manager import ResourceManager
            rm = ResourceManager()
            ok = rm.set_k2_dir(path) if self.game_tag == "K2" else rm.set_k1_dir(path)
            if ok:
                self._resource_manager = rm
                log.info(f"DebugSession: ResourceManager initialized for {path}")
            else:
                log.warning(f"DebugSession: ResourceManager failed to index {path}")
        except Exception as e:
            log.warning(f"DebugSession: ResourceManager init error: {e}")
        # Auto-verify
        return self.verify_game()

    def verify_game(self) -> dict:
        if not self.game_dir:
            return {"ok": False, "error": "No game path set"}
        key_path = os.path.join(self.game_dir, "chitin.key")
        models_bif = os.path.join(self.game_dir, "data", "models.bif")
        issues = []
        if not os.path.exists(key_path):
            issues.append(f"chitin.key missing at {key_path}")
        if not os.path.exists(models_bif):
            issues.append(f"data/models.bif missing at {models_bif}")
        self.game_verified = len(issues) == 0
        return {
            "ok": self.game_verified,
            "game_dir": self.game_dir,
            "chitin_key": os.path.exists(key_path),
            "models_bif": os.path.exists(models_bif),
            "issues": issues,
        }

    # ── BIF extraction ───────────────────────────────────────────────────────

    def _parse_key(self):
        """Parse chitin.key once and cache."""
        if self._key_data is not None:
            return self._key_data
        key_path = os.path.join(self.game_dir, "chitin.key")
        with open(key_path, "rb") as f:
            f.read(8)
            bif_count = struct.unpack("<I", f.read(4))[0]
            key_count = struct.unpack("<I", f.read(4))[0]
            off_file_table = struct.unpack("<I", f.read(4))[0]
            off_key_table = struct.unpack("<I", f.read(4))[0]
            bif_files = []
            for i in range(bif_count):
                f.seek(off_file_table + i * 12)
                f.read(4)
                name_offset = struct.unpack("<I", f.read(4))[0]
                name_size = struct.unpack("<H", f.read(2))[0]
                f.read(2)
                pos = f.tell()
                f.seek(name_offset)
                bif_name = (
                    f.read(name_size)
                    .rstrip(b"\x00")
                    .decode("ascii", errors="replace")
                    .replace("\\", "/")
                )
                bif_files.append(bif_name)
                f.seek(pos)
            entries = []
            for i in range(key_count):
                f.seek(off_key_table + i * 22)
                resref = (
                    f.read(16)
                    .rstrip(b"\x00")
                    .decode("ascii", errors="replace")
                    .lower()
                )
                res_type = struct.unpack("<H", f.read(2))[0]
                res_id = struct.unpack("<I", f.read(4))[0]
                entries.append((resref, res_type, res_id))
        self._key_data = {"bif_files": bif_files, "entries": entries}
        return self._key_data

    def extract_model(self, resref: str) -> Tuple[Optional[bytes], Optional[bytes]]:
        """Extract MDL+MDX bytes for a resref from game BIFs."""
        if not self.game_verified:
            return None, None
        kd = self._parse_key()
        target = resref.lower()
        results = {}  # ext → bytes
        for rr, rtype, rid in kd["entries"]:
            if rr == target and rtype in (2002, 3008):
                bif_idx = (rid >> 20) & 0xFFF
                res_idx = rid & 0xFFFFF
                if bif_idx < len(kd["bif_files"]):
                    bif_path = os.path.join(self.game_dir, kd["bif_files"][bif_idx])
                    if os.path.exists(bif_path):
                        bif_size = os.path.getsize(bif_path)
                        with open(bif_path, "rb") as bf:
                            bf.read(8)
                            bf.read(4)
                            bf.read(4)
                            var_table_offset = struct.unpack("<I", bf.read(4))[0]
                            if var_table_offset + (res_idx + 1) * 16 > bif_size:
                                raise ValueError(f"BIF resource table out of range: {bif_path}")
                            bf.seek(var_table_offset + res_idx * 16)
                            bf.read(4)
                            data_offset = struct.unpack("<I", bf.read(4))[0]
                            data_size = struct.unpack("<I", bf.read(4))[0]
                            bf.read(4)
                            if data_size > _MAX_BIF_RESOURCE or data_offset + data_size > bif_size:
                                raise ValueError(f"BIF resource size/offset out of range: {bif_path}")
                            bf.seek(data_offset)
                            raw = bf.read(data_size)
                            ext = "mdl" if rtype == 2002 else "mdx"
                            results[ext] = raw
        return results.get("mdl"), results.get("mdx")

    # ── Model loading ────────────────────────────────────────────────────────

    def resolve_model(self, resref: str) -> Tuple[str, bytes, bytes]:
        """Resolve MDL+MDX through the locator the export and model tools use
        (its filesystem candidates, then the installation: Override, modules,
        chitin), so the session loads exactly the bytes that
        ghostrigger_source_identity reports. The BIF-only extract_model
        ignored Override and could load a different model."""
        if not isinstance(resref, str) or not _SAFE_RESREF.fullmatch(resref):
            raise ValueError("resref must be 1-16 letters, digits, '_' or '-'")
        if not self.game_dir or not self.game_verified:
            raise ValueError("Game library is not set or not verified")
        from kotormcp.tools.ghostrigger import _get_services  # noqa: PLC0415

        path_label, mdl_bytes, mdx_bytes = _get_services().locator.locate(
            resref.lower(), self.game_tag.lower(), self.game_dir
        )
        if not mdl_bytes:
            raise FileNotFoundError(f"MDL not found: {resref}")
        if len(mdl_bytes) > _MAX_BIF_RESOURCE or len(mdx_bytes or b"") > _MAX_BIF_RESOURCE:
            raise ValueError("Model resource exceeds bound")
        return str(path_label), mdl_bytes, mdx_bytes or b""

    def load_model(self, resref: str) -> dict:
        """Load a model by resref from the game library."""
        try:
            path_label, mdl_bytes, mdx_bytes = self.resolve_model(resref)
        except (ValueError, FileNotFoundError) as e:
            return {"ok": False, "error": str(e)[:500]}

        try:
            from src.core.game.kotor_loader import load_model_from_bytes
            model = load_model_from_bytes(mdl_bytes, mdx_bytes)
        except Exception as e:
            return {"ok": False, "error": f"Parse failed: {type(e).__name__}: {str(e)[:300]}"}

        self.model = model
        self.model_resref = resref.lower()
        self.model_path = path_label
        self.model_source = {
            "resref": resref.lower(),
            "path": path_label,
            "mdl": _bytes_identity(mdl_bytes),
            "mdx": _bytes_identity(mdx_bytes),
        }
        # The previous model's animation bind reference must not reach the
        # renderer with the new model (it was kept across loads).
        self.anim_base_pose = None
        self.current_anim_name = None
        self.current_anim_time = 0.0
        self.current_pose = None
        # FIX-TEXLOAD-D5: Clear texture cache so new model's textures are loaded
        self._model_textures = {}

        # Init animation engine
        try:
            from src.core.animation.animation_engine import AnimationEngine
            self.anim_engine = AnimationEngine(model)
        except Exception as e:
            log.warning(f"AnimationEngine init failed: {e}")
            self.anim_engine = None

        # Init skin uploader
        try:
            from src.core.animation.gpu_skinning import MatrixPaletteUploader
            self.skin_uploader = MatrixPaletteUploader()
            n_bones = self.skin_uploader.build_inverse_bind_pose(model)
        except Exception as e:
            log.warning(f"MatrixPaletteUploader init failed: {e}")
            self.skin_uploader = None
            n_bones = 0

        # Collect stats
        nodes = list(model.all_nodes()) if hasattr(model, "all_nodes") else []
        mesh_nodes = [n for n in nodes if getattr(n, "is_mesh", False)]
        skin_nodes = [n for n in nodes if getattr(n, "is_skin", False)]
        anims = _get_anim_names(model)

        # Skeleton depth
        def _depth(nd, d=0):
            p = getattr(nd, "parent", None)
            if p is None:
                return d
            return _depth(p, d + 1)

        max_depth = max((_depth(n) for n in nodes), default=0)

        return {
            "ok": True,
            "resref": self.model_resref,
            "node_count": len(nodes),
            "mesh_count": len(mesh_nodes),
            "skin_count": len(skin_nodes),
            "bone_count": n_bones,
            "skeleton_depth": max_depth,
            "animations": anims,
            "supermodel": getattr(model, "supermodel_name", ""),
            "classification": str(getattr(model, "classification", "")),
            "source": self.model_source,
        }

    # ── Asset info ───────────────────────────────────────────────────────────

    def get_asset_info(self) -> dict:
        if self.model is None:
            return {"ok": False, "error": "No model loaded"}
        nodes = list(self.model.all_nodes()) if hasattr(self.model, "all_nodes") else []
        mesh_nodes = [n for n in nodes if getattr(n, "is_mesh", False)]
        skin_nodes = [n for n in nodes if getattr(n, "is_skin", False)]
        anims = _get_anim_names(self.model)

        skin_info = []
        for sn in skin_nodes:
            bmap = getattr(sn, "bone_map", [])
            sd = getattr(sn, "skin_data", [])
            skin_info.append({
                "name": getattr(sn, "name", "?"),
                "bone_map_size": len(bmap),
                "bone_map": bmap[:20],  # first 20
                "vertex_count": len(getattr(sn, "vertices", getattr(sn, "verts", []))),
                "skin_data_count": len(sd),
            })

        return {
            "ok": True,
            "resref": self.model_resref,
            "node_count": len(nodes),
            "mesh_count": len(mesh_nodes),
            "skin_count": len(skin_nodes),
            "skin_nodes": skin_info,
            "animations": anims,
            "current_animation": self.current_anim_name,
            "current_time": self.current_anim_time,
        }

    # ── Animation control ────────────────────────────────────────────────────

    def list_animations(self) -> list:
        if self.model is None:
            return []
        raw_anims = getattr(self.model, "animations", []) or []
        result = []
        # Handle both list[Animation] and dict[str, Animation]
        if isinstance(raw_anims, dict):
            items = raw_anims.items()
        else:
            items = [(getattr(a, "name", f"anim_{i}"), a) for i, a in enumerate(raw_anims)]
        for name, anim in items:
            na = getattr(anim, "node_anims", {})
            na_count = len(na) if isinstance(na, dict) else len(na) if hasattr(na, '__len__') else 0
            result.append({
                "name": name,
                "length": getattr(anim, "length", 0.0),
                "transition_time": getattr(anim, "transition_time", 0.0),
                "node_count": na_count,
            })
        return result

    def set_animation(self, anim_name: str) -> dict:
        if self.anim_engine is None:
            return {"ok": False, "error": "No animation engine"}
        if (
            not isinstance(anim_name, str)
            or len(anim_name) > _MAX_ANIMATION_NAME
            or anim_name not in _get_anim_names(self.model)
        ):
            return {"ok": False, "error": "Animation is not defined on the loaded model"}
        try:
            self.anim_engine.play(anim_name, loop=False, blend=False)
            self.current_anim_name = anim_name
            self.current_anim_time = 0.0
            self.current_pose = self.anim_engine.evaluate(0.0)
            # FIX-SKIN-ANIM-D2: Capture the first-frame pose as bind reference.
            # This matches the xoreos approach where the base transform is built
            # from the animation's initial frame, not the static hierarchy.
            self.anim_base_pose = self.anim_engine.evaluate(0.0)
            # Also update the skin uploader if available
            if self.skin_uploader is not None:
                self.skin_uploader.set_bind_pose_from_anim(self.anim_base_pose)
            return {"ok": True, "animation": anim_name, "time": 0.0}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def set_animation_time(self, t: float) -> dict:
        if self.anim_engine is None:
            return {"ok": False, "error": "No animation engine"}
        if self.current_anim_name is None:
            return {"ok": False, "error": "No animation active"}
        anim = _get_anim_dict(self.model).get(self.current_anim_name)
        length = float(getattr(anim, "length", 0.0) or 0.0)
        if (
            isinstance(t, bool)
            or not isinstance(t, (int, float))
            or not math.isfinite(t)
            or t < 0.0
            or t > length + 1e-6
        ):
            return {"ok": False, "error": f"time must be a finite number between 0 and {length:g}"}
        t = min(float(t), length)
        try:
            self.anim_engine.seek(t)
            self.current_anim_time = t
            self.current_pose = self.anim_engine.evaluate(t)
            return {
                "ok": True,
                "animation": self.current_anim_name,
                "time": t,
                "pose_nodes": len(self.current_pose.nodes) if self.current_pose else 0,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def set_bind_pose(self) -> dict:
        self.current_anim_name = None
        self.current_anim_time = 0.0
        self.current_pose = None
        self.anim_base_pose = None  # FIX-SKIN-ANIM-D2
        if self.anim_engine:
            self.anim_engine.stop()
        # FIX-SKIN-ANIM-D2: Clear animation bind reference
        if self.skin_uploader is not None:
            self.skin_uploader._inv_bind_anim = None
        return {"ok": True, "mode": "bind_pose"}

    # ── Camera ───────────────────────────────────────────────────────────────

    # ArcBallCamera places the eye at azimuth 90 on +Y, its canonical front
    # (KOTOR models face +Y). The old table used 0 for "front", so "front"
    # captures showed the model's side and "side" showed its front.
    CAMERA_PRESETS = {
        "front": (90.0, 15.0),
        "diagonal": (45.0, 25.0),
        "side": (0.0, 15.0),
        "top": (90.0, 80.0),
        "back": (270.0, 15.0),
    }

    def set_camera(self, preset: str) -> dict:
        if isinstance(preset, str) and preset in self.CAMERA_PRESETS:
            self.camera_azimuth, self.camera_elevation = self.CAMERA_PRESETS[preset]
            self.camera_preset = preset
            return {"ok": True, "preset": preset,
                    "azimuth": self.camera_azimuth, "elevation": self.camera_elevation}
        return {"ok": False, "error": "preset must be one of " + ", ".join(self.CAMERA_PRESETS)}

    # ── Rendering ────────────────────────────────────────────────────────────

    def capture_viewport(self, width: int = 512, height: int = 512,
                         output_path: Optional[str] = None,
                         exclusive: bool = False) -> dict:
        if self.model is None:
            return {"ok": False, "error": "No model loaded"}
        if self.renderer is None:
            return {"ok": False, "error": "No renderer available"}
        if not (_MIN_CAPTURE <= int(width) <= _MAX_CAPTURE and _MIN_CAPTURE <= int(height) <= _MAX_CAPTURE):
            return {"ok": False, "error": f"width/height must be between {_MIN_CAPTURE} and {_MAX_CAPTURE}"}
        if output_path and exclusive and os.path.exists(output_path):
            return {"ok": False, "error": "Refusing to overwrite existing capture"}

        try:
            from src.core.camera.arcball_camera import ArcBallCamera
            camera = ArcBallCamera()
            camera.azimuth = self.camera_azimuth
            camera.elevation = self.camera_elevation
            # MDL headers can carry oversized placeholder bounds, which left
            # the model a speck in the frame. Frame the renderable world-space
            # geometry, as ghostrigger_render_model does.
            try:
                from src.core.rendering.gpu_scene_helpers import _compute_model_bounds

                rb = _compute_model_bounds(self.model)
                bb_min = tuple(float(rb[f"min_{a}"]) for a in "xyz")
                bb_max = tuple(float(rb[f"max_{a}"]) for a in "xyz")
                if not all(math.isfinite(v) for v in bb_min + bb_max):
                    raise ValueError("non-finite bounds")
            except Exception:
                bb_min = getattr(self.model, "bb_min", None)
                bb_max = getattr(self.model, "bb_max", None)
            if bb_min is not None and bb_max is not None:
                camera.frame_bounds(bb_min, bb_max)
            else:
                camera.distance = 3.5
                camera.target = [0.0, 0.0, 0.9]

            # FIX-TEXLOAD-D5: Load textures from ResourceManager.
            # Previously textures={} was passed, causing untextured renders.
            # Now we resolve all model textures through the full KotOR
            # resource chain: Override > module ERFs > TexturePacks > BIF.
            if not self._model_textures and self._resource_manager is not None:
                try:
                    from src.core.assets.resource_manager import resolve_model_textures
                    self._model_textures = resolve_model_textures(
                        self.model, self._resource_manager, game=self.game_tag)
                    log.info(f"DebugSession: loaded {len(self._model_textures)} textures")
                except Exception as tex_err:
                    log.warning(f"DebugSession: texture load error: {tex_err}")

            img = self.renderer.render(
                self.model, camera, width, height,
                textures=self._model_textures,
                anim_pose=self.current_pose,
                anim_time=self.current_anim_time,
                anim_base_pose=self.anim_base_pose,  # FIX-SKIN-ANIM-D2
            )

            if img is None:
                return {"ok": False, "error": "Render returned None"}

            self.last_image = img

            if output_path:
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                if exclusive:
                    with open(output_path, "xb") as handle:
                        img.save(handle, format="PNG")
                else:
                    img.save(output_path)

            return {
                "ok": True,
                "width": img.width,
                "height": img.height,
                "output_path": output_path,
                "backend": self.renderer.perf.get("backend", "unknown"),
                "frame_ms": self.renderer.perf.get("last_frame_ms", 0),
                "textures": len(self._model_textures),
                "game_tag": self.game_tag,
            }
        except Exception as e:
            # No traceback: it disclosed local paths to MCP clients.
            return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:500]}"}

    def capture_validation_set(self, set_name: str,
                               output_dir: Optional[str] = None,
                               exclusive: bool = False) -> dict:
        """Capture a standard validation set: bind-pose + animated views."""
        if self.model is None:
            return {"ok": False, "error": "No model loaded"}
        if not _SAFE_SET_NAME.fullmatch(str(set_name or "")):
            return {"ok": False, "error": "set_name must be 1-64 letters, digits, '_' or '-'"}

        if output_dir is None:
            output_dir = os.path.join(_PROJECT_ROOT, "debug_bundles", self.model_resref)
        os.makedirs(output_dir, exist_ok=True)

        captures = []
        # Bind-pose captures
        saved_anim = self.current_anim_name
        saved_time = self.current_anim_time
        saved_camera = self.camera_preset

        self.set_bind_pose()
        for preset in ["front", "diagonal"]:
            self.set_camera(preset)
            fname = f"{set_name}_bindpose_{preset}.png"
            fpath = os.path.join(output_dir, fname)
            r = self.capture_viewport(output_path=fpath, exclusive=exclusive)
            captures.append({"file": fname, "mode": "bind_pose",
                             "camera": preset, "ok": r.get("ok", False)})

        # Animated captures at 30% and 70% of each animation
        anims = _get_anim_names(self.model)
        anim_dict = _get_anim_dict(self.model)
        test_anims = [a for a in anims if a.lower() in ("cwalk", "crun", "cpause1")]
        if not test_anims and anims:
            test_anims = anims[:1]

        for anim_name in test_anims:
            anim = anim_dict.get(anim_name)
            if anim is None:
                continue
            length = getattr(anim, "length", 1.0)
            self.set_animation(anim_name)
            for pct in (0.3, 0.7):
                t = length * pct
                self.set_animation_time(t)
                for preset in ["front", "diagonal"]:
                    self.set_camera(preset)
                    safe_anim = _UNSAFE_FILE_CHARS.sub("_", str(anim_name))[:64]
                    fname = f"{set_name}_{safe_anim}_t{int(pct*100)}_{preset}.png"
                    fpath = os.path.join(output_dir, fname)
                    r = self.capture_viewport(output_path=fpath, exclusive=exclusive)
                    captures.append({
                        "file": fname,
                        "mode": "animated",
                        "animation": anim_name,
                        "time": round(t, 3),
                        "camera": preset,
                        "ok": r.get("ok", False),
                    })

        # Restore the caller's pose and camera (the camera used to be left on
        # the last preset, silently changing the session under the caller).
        if saved_anim:
            self.set_animation(saved_anim)
            self.set_animation_time(saved_time)
        else:
            self.set_bind_pose()
        if saved_camera in self.CAMERA_PRESETS:
            self.set_camera(saved_camera)

        return {
            "ok": True,
            "set_name": set_name,
            "output_dir": output_dir,
            "captures": captures,
            "total": len(captures),
            "passed": sum(1 for c in captures if c["ok"]),
        }

    # ── Skinning state queries ───────────────────────────────────────────────

    def get_skinning_state(self) -> dict:
        if self.model is None:
            return {"ok": False, "error": "No model loaded"}
        nodes = list(self.model.all_nodes()) if hasattr(self.model, "all_nodes") else []
        skin_nodes = [n for n in nodes if getattr(n, "is_skin", False)]
        return {
            "ok": True,
            "resref": self.model_resref,
            "skin_node_count": len(skin_nodes),
            "skin_nodes": [getattr(sn, "name", "?") for sn in skin_nodes],
            "uploader_active": self.skin_uploader is not None,
            "bone_count": self.skin_uploader.bone_count if self.skin_uploader else 0,
            "palette_size": len(self.skin_uploader.palette) if self.skin_uploader else 0,
            "current_animation": self.current_anim_name,
            "current_time": self.current_anim_time,
            "pose_active": self.current_pose is not None,
        }

    def get_renderer_state(self) -> dict:
        if self.renderer is None:
            return {"ok": False, "error": "No renderer"}
        return {
            "ok": True,
            "gpu_available": getattr(self.renderer, "_gpu_available", False),
            "force_cpu": getattr(self.renderer, "force_cpu", False),
            "perf": dict(getattr(self.renderer, "perf", {})),
            "mesh_cache_size": len(getattr(self.renderer, "_mesh_cache", {})),
            "skin_uploader_active": getattr(self.renderer, "_skin_uploader", None) is not None,
            "skin_model_id": getattr(self.renderer, "_skin_model_id", 0),
        }

    def get_bone_hierarchy(self) -> dict:
        if self.model is None:
            return {"ok": False, "error": "No model loaded"}
        nodes = list(self.model.all_nodes()) if hasattr(self.model, "all_nodes") else []

        def _build_tree(node, depth=0):
            children = getattr(node, "children", [])
            return {
                "name": getattr(node, "name", "?"),
                "depth": depth,
                "is_mesh": getattr(node, "is_mesh", False),
                "is_skin": getattr(node, "is_skin", False),
                "position": list(getattr(node, "position", (0, 0, 0)) or (0, 0, 0)),
                "rotation": list(getattr(node, "rotation", (0, 0, 0, 1)) or (0, 0, 0, 1)),
                "children": [_build_tree(c, depth + 1) for c in children],
            }

        root = getattr(self.model, "root_node", None)
        if root is None:
            return {"ok": False, "error": "No root node"}

        return {
            "ok": True,
            "resref": self.model_resref,
            "total_nodes": len(nodes),
            "hierarchy": _build_tree(root),
        }

    def get_bone_map_for_mesh(self, mesh_name: str) -> dict:
        if self.model is None:
            return {"ok": False, "error": "No model loaded"}
        nodes = list(self.model.all_nodes()) if hasattr(self.model, "all_nodes") else []
        target = mesh_name.lower()
        for node in nodes:
            if getattr(node, "name", "").lower() == target:
                bmap = getattr(node, "bone_map", [])
                # Build remap table
                remap = {}
                if self.skin_uploader and bmap:
                    for i, bname in enumerate(bmap):
                        if bname:
                            pidx = self.skin_uploader.bone_index(bname)
                            remap[i] = {"bone_name": bname, "palette_index": pidx}
                        else:
                            remap[i] = {"bone_name": "", "palette_index": 0}

                return {
                    "ok": True,
                    "mesh_name": mesh_name,
                    "is_skin": getattr(node, "is_skin", False),
                    "bone_map": bmap,
                    "bone_map_size": len(bmap),
                    "remap_table": remap,
                }
        return {"ok": False, "error": f"Mesh '{mesh_name}' not found"}

    def get_palette_remap_table(self) -> dict:
        if self.model is None or self.skin_uploader is None:
            return {"ok": False, "error": "No model/uploader"}
        nodes = list(self.model.all_nodes()) if hasattr(self.model, "all_nodes") else []
        skin_nodes = [n for n in nodes if getattr(n, "is_skin", False)]

        tables = {}
        for sn in skin_nodes:
            sname = getattr(sn, "name", "?")
            bmap = getattr(sn, "bone_map", [])
            remap = {}
            for i, bname in enumerate(bmap):
                if bname:
                    pidx = self.skin_uploader.bone_index(bname)
                    remap[str(i)] = {
                        "local_idx": i,
                        "bone_name": bname,
                        "palette_idx": pidx,
                        "valid": pidx >= 0,
                    }
            tables[sname] = remap

        return {
            "ok": True,
            "resref": self.model_resref,
            "skin_node_count": len(skin_nodes),
            "remap_tables": tables,
        }

    def get_bind_pose_matrices(self, bone_names: Optional[List[str]] = None) -> dict:
        if self.skin_uploader is None:
            return {"ok": False, "error": "No skin uploader"}
        # Compute bind-pose palette (all identity)
        self.skin_uploader.compute_palette(None)
        palette = self.skin_uploader.palette

        entries = {}
        for bm in palette:
            if bone_names and bm.bone_name not in [b.lower() for b in bone_names]:
                continue
            # Check if identity
            ident = _mat4_to_flat_col_ident()
            is_ident = all(abs(a - b) < 1e-6 for a, b in zip(bm.flat_col, ident))
            entries[bm.bone_name] = {
                "index": bm.bone_index,
                "matrix": [round(v, 6) for v in bm.flat_col],
                "is_identity": is_ident,
            }

        return {
            "ok": True,
            "mode": "bind_pose",
            "bone_count": len(palette),
            "all_identity": all(e["is_identity"] for e in entries.values()),
            "matrices": entries,
        }

    def get_animated_pose_matrices(self, bone_names: Optional[List[str]] = None) -> dict:
        if self.skin_uploader is None:
            return {"ok": False, "error": "No skin uploader"}
        self.skin_uploader.compute_palette(self.current_pose)
        palette = self.skin_uploader.palette

        entries = {}
        for bm in palette:
            if bone_names and bm.bone_name not in [b.lower() for b in bone_names]:
                continue
            ident = _mat4_to_flat_col_ident()
            is_ident = all(abs(a - b) < 1e-6 for a, b in zip(bm.flat_col, ident))
            # Extract translation from column-major
            tx = bm.flat_col[12]
            ty = bm.flat_col[13]
            tz = bm.flat_col[14]
            entries[bm.bone_name] = {
                "index": bm.bone_index,
                "matrix": [round(v, 6) for v in bm.flat_col],
                "is_identity": is_ident,
                "translation": [round(tx, 4), round(ty, 4), round(tz, 4)],
            }

        non_ident = sum(1 for e in entries.values() if not e["is_identity"])
        return {
            "ok": True,
            "mode": "animated",
            "animation": self.current_anim_name,
            "time": self.current_anim_time,
            "bone_count": len(palette),
            "non_identity_count": non_ident,
            "matrices": entries,
        }

    def get_uploaded_palette(self) -> dict:
        """Get the palette that would be uploaded to GPU."""
        if self.skin_uploader is None:
            return {"ok": False, "error": "No skin uploader"}
        self.skin_uploader.compute_palette(self.current_pose)
        palette = self.skin_uploader.palette
        flat_bytes = self.skin_uploader.as_flat_bytes()
        return {
            "ok": True,
            "bone_count": len(palette),
            "byte_size": len(flat_bytes),
            "format": "std430 mat4 array, column-major, float32",
            "first_5_bones": [
                {
                    "index": bm.bone_index,
                    "name": bm.bone_name,
                    "matrix_col_major": [round(v, 4) for v in bm.flat_col],
                }
                for bm in palette[:5]
            ],
        }

    def sample_vertex_influences(self, mesh_name: str,
                                  vertex_indices: Optional[List[int]] = None,
                                  max_samples: int = 10) -> dict:
        if self.model is None:
            return {"ok": False, "error": "No model loaded"}
        nodes = list(self.model.all_nodes()) if hasattr(self.model, "all_nodes") else []
        target = mesh_name.lower()
        for node in nodes:
            if getattr(node, "name", "").lower() == target:
                sd = getattr(node, "skin_data", [])
                bmap = getattr(node, "bone_map", [])
                verts = getattr(node, "vertices", getattr(node, "verts", []))
                if not sd:
                    return {"ok": False, "error": f"No skin_data on '{mesh_name}'"}

                if vertex_indices is None:
                    # Sample evenly
                    step = max(1, len(sd) // max_samples)
                    vertex_indices = list(range(0, len(sd), step))[:max_samples]

                samples = []
                for vi in vertex_indices:
                    if vi >= len(sd):
                        continue
                    entry = sd[vi]
                    infl = getattr(entry, "influences", [])
                    influences = []
                    for bw in infl:
                        local_idx = int(getattr(bw, "bone_index", 0))
                        weight = float(getattr(bw, "weight", 0.0))
                        bone_name = bmap[local_idx] if local_idx < len(bmap) else "?"
                        palette_idx = -1
                        if self.skin_uploader:
                            palette_idx = self.skin_uploader.bone_index(bone_name)
                        influences.append({
                            "local_bone_idx": local_idx,
                            "bone_name": bone_name,
                            "palette_idx": palette_idx,
                            "weight": round(weight, 6),
                        })
                    pos = list(verts[vi]) if vi < len(verts) else [0, 0, 0]
                    samples.append({
                        "vertex_index": vi,
                        "position": [round(p, 4) for p in pos],
                        "influences": influences,
                        "weight_sum": round(sum(inf["weight"] for inf in influences), 6),
                    })

                return {
                    "ok": True,
                    "mesh_name": mesh_name,
                    "total_vertices": len(verts),
                    "samples": samples,
                }
        return {"ok": False, "error": f"Mesh '{mesh_name}' not found"}

    # ── CPU vs GPU comparison ────────────────────────────────────────────────

    def compare_cpu_gpu_skinning(self, mesh_name: str,
                                  vertex_indices: Optional[List[int]] = None,
                                  max_verts: int = 30) -> dict:
        """Apply CPU LBS transform and compare with GPU palette data."""
        if self.model is None or self.skin_uploader is None:
            return {"ok": False, "error": "No model/uploader"}

        nodes = list(self.model.all_nodes()) if hasattr(self.model, "all_nodes") else []
        target = mesh_name.lower()
        node = None
        for n in nodes:
            if getattr(n, "name", "").lower() == target:
                node = n
                break
        if node is None:
            return {"ok": False, "error": f"Mesh '{mesh_name}' not found"}

        sd = getattr(node, "skin_data", [])
        bmap = getattr(node, "bone_map", [])
        verts = getattr(node, "vertices", getattr(node, "verts", []))

        if not sd or not verts:
            return {"ok": False, "error": "No skin_data/vertices"}

        # Compute palette
        self.skin_uploader.compute_palette(self.current_pose)
        palette = self.skin_uploader.palette

        if vertex_indices is None:
            step = max(1, len(sd) // max_verts)
            vertex_indices = list(range(0, len(sd), step))[:max_verts]

        comparisons = []
        max_diff = 0.0
        for vi in vertex_indices:
            if vi >= len(sd) or vi >= len(verts):
                continue
            pos = list(verts[vi])
            entry = sd[vi]
            infl = getattr(entry, "influences", [])

            # CPU LBS
            cpu_pos = [0.0, 0.0, 0.0]
            for bw in infl:
                local_idx = int(getattr(bw, "bone_index", 0))
                weight = float(getattr(bw, "weight", 0.0))
                bone_name = bmap[local_idx] if local_idx < len(bmap) else ""
                pidx = self.skin_uploader.bone_index(bone_name)
                if pidx < 0 or pidx >= len(palette):
                    pidx = 0
                m = palette[pidx].flat_col  # column-major
                # Transform: M × [x,y,z,1]
                vx, vy, vz = float(pos[0]), float(pos[1]), float(pos[2])
                # Column-major: m[0..3]=col0, m[4..7]=col1, m[8..11]=col2, m[12..15]=col3
                rx = m[0]*vx + m[4]*vy + m[8]*vz + m[12]
                ry = m[1]*vx + m[5]*vy + m[9]*vz + m[13]
                rz = m[2]*vx + m[6]*vy + m[10]*vz + m[14]
                cpu_pos[0] += weight * rx
                cpu_pos[1] += weight * ry
                cpu_pos[2] += weight * rz

            diff = math.sqrt(sum((a - b)**2 for a, b in zip(pos, cpu_pos)))
            max_diff = max(max_diff, diff)
            comparisons.append({
                "vertex_index": vi,
                "bind_pos": [round(p, 4) for p in pos],
                "cpu_skinned_pos": [round(p, 4) for p in cpu_pos],
                "diff": round(diff, 6),
            })

        return {
            "ok": True,
            "mesh_name": mesh_name,
            "animation": self.current_anim_name,
            "time": self.current_anim_time,
            "vertex_count": len(comparisons),
            "max_diff": round(max_diff, 6),
            "parity_pass": max_diff < 0.01,
            "comparisons": comparisons,
        }

    # ── Export debug bundle ───────────────────────────────────────────────────

    def export_debug_bundle(self, output_dir: Optional[str] = None,
                            exclusive: bool = False) -> dict:
        """Export a comprehensive debug bundle for the currently loaded asset.

        With ``exclusive`` the output directory must be new or empty and every
        file is created exclusively (no overwrite).
        """
        if self.model is None:
            return {"ok": False, "error": "No model loaded"}

        if output_dir is None:
            output_dir = os.path.join(
                _PROJECT_ROOT, "debug_bundles", self.model_resref
            )
        if exclusive and os.path.isdir(output_dir) and os.listdir(output_dir):
            return {"ok": False, "error": "Refusing to write into a non-empty bundle directory"}
        os.makedirs(output_dir, exist_ok=True)

        bundle = {
            "resref": self.model_resref,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "asset_info": self.get_asset_info(),
            "skinning_state": self.get_skinning_state(),
            "bone_hierarchy": self.get_bone_hierarchy(),
            "palette_remap": self.get_palette_remap_table(),
            "bind_pose_matrices": self.get_bind_pose_matrices(),
        }

        # Animated pose matrices if animation is active
        if self.current_anim_name:
            bundle["animated_pose_matrices"] = self.get_animated_pose_matrices()

        # Sample vertex influences for each skin node
        nodes = list(self.model.all_nodes()) if hasattr(self.model, "all_nodes") else []
        skin_nodes = [n for n in nodes if getattr(n, "is_skin", False)]
        bundle["vertex_samples"] = {}
        for sn in skin_nodes:
            sname = getattr(sn, "name", "?")
            bundle["vertex_samples"][sname] = self.sample_vertex_influences(sname)

        # CPU vs GPU comparison for first skin node
        if skin_nodes:
            first_skin = getattr(skin_nodes[0], "name", "?")
            bundle["cpu_gpu_comparison"] = self.compare_cpu_gpu_skinning(first_skin)

        # Validation captures
        set_name = _UNSAFE_FILE_CHARS.sub("_", self.model_resref)[:64] or "model"
        captures = self.capture_validation_set(set_name, output_dir, exclusive=exclusive)
        bundle["validation_captures"] = captures

        # Save bundle JSON
        bundle_path = os.path.join(output_dir, "debug_bundle.json")
        with open(bundle_path, "x" if exclusive else "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2, default=str)

        return {
            "ok": True,
            "output_dir": output_dir,
            "bundle_path": bundle_path,
            "sections": list(bundle.keys()),
            "capture_count": captures.get("total", 0),
            "captures_passed": captures.get("passed", 0),
            "game_tag": self.game_tag,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_session: Optional[_DebugSession] = None


def _get_session() -> _DebugSession:
    global _session
    if _session is None:
        _session = _DebugSession()
    return _session


def _reset_session():
    """Reset the debug session (for testing)."""
    global _session
    if _session is not None:
        _session.close()
    _session = None


def session_source_identity(resref: str, game_path: str) -> Dict[str, Any]:
    """Read-only: the source identity load_model would bind for ``resref``.

    Uses a throwaway probe (no renderer, no ResourceManager) so the
    process-wide session is untouched. Same shape as ``model_source``.
    """
    if (
        not isinstance(game_path, str)
        or len(game_path) > _MAX_PATH_TEXT
        or not os.path.isabs(game_path)
        or not os.path.isdir(game_path)
    ):
        raise ValueError("game_path must be an absolute game directory")
    probe = _DebugSession()
    probe.game_dir = game_path
    probe.game_tag = probe.detect_game_tag()
    if not probe.verify_game().get("ok"):
        raise ValueError("Game library is missing chitin.key or data/models.bif")
    path_label, mdl_bytes, mdx_bytes = probe.resolve_model(resref)
    return {
        "model": {
            "resref": resref.lower(),
            "path": path_label,
            "mdl": _bytes_identity(mdl_bytes),
            "mdx": _bytes_identity(mdx_bytes),
        },
        "game_tag": probe.game_tag,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────────────────────────────────────

def _mat4_to_flat_col_ident() -> list:
    """Return a flat column-major identity matrix."""
    return [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]


def _get_anim_names(model) -> list:
    """Get animation names from a model, handling both list and dict formats."""
    raw = getattr(model, "animations", []) or []
    if isinstance(raw, dict):
        return list(raw.keys())
    return [getattr(a, "name", f"anim_{i}") for i, a in enumerate(raw)]


def _get_anim_dict(model) -> dict:
    """Get animations as a dict {name: anim}, handling both list and dict formats."""
    raw = getattr(model, "animations", []) or []
    if isinstance(raw, dict):
        return raw
    return {getattr(a, "name", f"anim_{i}"): a for i, a in enumerate(raw)}


# ─────────────────────────────────────────────────────────────────────────────
#  Tool definitions
# ─────────────────────────────────────────────────────────────────────────────

def get_tools() -> List[Dict[str, Any]]:
    """Return debug-skinning MCP tool definitions (23 commands)."""
    return [
        {
            "name": "ghostrigger_debug_launch_app",
            "description": (
                "Launch the in-process headless GhostRigger debug session (renderer only; "
                "not the GhostRigger Debug app). Relaunching replaces the current session. "
                "Returns a new session_id and the state revision."
            ),
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "ghostrigger_debug_close_app",
            "description": "Close the debug session, releasing all resources.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "ghostrigger_debug_get_runtime_status",
            "description": "Get the current runtime status of the debug session.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "ghostrigger_debug_set_game_library_path",
            "description": (
                "Set the KotOR game directory for the debug session (read only). "
                "Indexed as K2 when it holds swkotor2.exe, otherwise K1."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "maxLength": 1024,
                        "description": "Absolute path to the KotOR game directory (containing chitin.key and data/models.bif)",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "ghostrigger_debug_verify_game_library",
            "description": "Verify that the game library path is valid and contains required files.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "ghostrigger_debug_load_model",
            "description": (
                "Load a KotOR model by resref, resolved through the same locator as the "
                "model/export tools (Override, modules, chitin). Returns the MDL/MDX SHA-256."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "resref": {
                        "type": "string",
                        "pattern": "^[A-Za-z0-9_-]{1,16}$",
                        "description": "Model resource name (e.g. c_kraytdragon, n_commf)",
                    },
                },
                "required": ["resref"],
                "additionalProperties": False,
            },
        },
        {
            "name": "ghostrigger_debug_get_loaded_asset_info",
            "description": "Get detailed information about the currently loaded asset (nodes, skins, animations).",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "ghostrigger_debug_list_animations",
            "description": "List all animations available on the currently loaded model.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "ghostrigger_debug_set_animation",
            "description": "Set the current animation (one of the loaded model's animations) and start at time 0.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "animation_name": {"type": "string", "maxLength": 64, "description": "Animation name (e.g. cwalk, crun)"},
                },
                "required": ["animation_name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "ghostrigger_debug_set_animation_time",
            "description": "Seek the current animation to a time between 0 and its length (seconds).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "time": {"type": "number", "minimum": 0, "description": "Time in seconds"},
                },
                "required": ["time"],
                "additionalProperties": False,
            },
        },
        {
            "name": "ghostrigger_debug_set_bind_pose",
            "description": "Reset to bind pose (no animation active).",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "ghostrigger_debug_set_camera_preset",
            "description": "Set camera to a named preset (front, diagonal, side, top, back).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "preset": {
                        "type": "string",
                        "enum": ["front", "diagonal", "side", "top", "back"],
                        "description": "Camera preset name",
                    },
                },
                "required": ["preset"],
                "additionalProperties": False,
            },
        },
        {
            "name": "ghostrigger_debug_capture_viewport",
            "description": "Render the current model/pose/camera and optionally save a PNG screenshot.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "width": {"type": "integer", "minimum": 16, "maximum": 4096, "default": 512},
                    "height": {"type": "integer", "minimum": 16, "maximum": 4096, "default": 512},
                    "output_path": {"type": "string", "maxLength": 1024, "description": "Absolute .png path"},
                    "overwrite": {
                        "type": "boolean",
                        "default": True,
                        "description": "False creates the PNG exclusively (refuses an existing file)",
                    },
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "ghostrigger_debug_capture_validation_set",
            "description": (
                "Capture a validation set (bind pose, then up to three animations at 30%/70%, "
                "front + diagonal; at most 14 PNGs). Restores the session's pose and camera."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "set_name": {
                        "type": "string",
                        "pattern": "^[A-Za-z0-9_-]{1,64}$",
                        "description": "Name for this validation set (file prefix)",
                    },
                    "output_dir": {"type": "string", "maxLength": 1024, "description": "Absolute directory"},
                    "overwrite": {
                        "type": "boolean",
                        "default": True,
                        "description": "False requires a new or empty output_dir and creates every PNG exclusively",
                    },
                },
                "required": ["set_name", "output_dir"],
                "additionalProperties": False,
            },
        },
        {
            "name": "ghostrigger_debug_get_skinning_state",
            "description": "Get current skinning state: skin nodes, bone count, palette, animation status.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "ghostrigger_debug_get_renderer_state",
            "description": "Get current renderer state: GPU availability, performance counters, cache sizes.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "ghostrigger_debug_get_bone_hierarchy",
            "description": "Get the full bone/node hierarchy tree of the loaded model.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "ghostrigger_debug_get_bone_map",
            "description": "Get the bone_map for a specific skin mesh node, including palette remap.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mesh_name": {"type": "string", "description": "Name of the skin mesh node"},
                },
                "required": ["mesh_name"],
            },
        },
        {
            "name": "ghostrigger_debug_get_palette_remap_table",
            "description": "Get the palette remap tables for all skin nodes (local_idx → palette_idx mapping).",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "ghostrigger_debug_get_bind_pose_matrices",
            "description": "Get the bind-pose skinning matrices (should all be identity).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "bone_names": {"type": "array", "items": {"type": "string"},
                                   "description": "Optional filter: only these bones"},
                },
            },
        },
        {
            "name": "ghostrigger_debug_get_animated_pose_matrices",
            "description": "Get the animated-pose skinning matrices for the current animation/time.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "bone_names": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        {
            "name": "ghostrigger_debug_get_uploaded_palette",
            "description": "Get the palette data that would be uploaded to the GPU SSBO.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "ghostrigger_debug_sample_vertex_influences",
            "description": "Sample vertex bone influences (weights, bone IDs, names) for a skin mesh.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mesh_name": {"type": "string"},
                    "vertex_indices": {"type": "array", "items": {"type": "integer"}},
                    "max_samples": {"type": "integer", "default": 10},
                },
                "required": ["mesh_name"],
            },
        },
        {
            "name": "ghostrigger_debug_compare_cpu_gpu_skinning",
            "description": "Run CPU LBS transform and compare with GPU palette to verify parity.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mesh_name": {"type": "string"},
                    "vertex_indices": {"type": "array", "items": {"type": "integer"}},
                    "max_verts": {"type": "integer", "default": 30},
                },
                "required": ["mesh_name"],
            },
        },
        {
            "name": "ghostrigger_debug_export_debug_bundle",
            "description": "Export a comprehensive debug bundle (JSON + screenshots) for the loaded asset.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_dir": {"type": "string"},
                    "overwrite": {
                        "type": "boolean",
                        "default": True,
                        "description": "False requires a new or empty output_dir and creates every file exclusively",
                    },
                },
            },
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  Handlers
# ─────────────────────────────────────────────────────────────────────────────

def _reply(result: Dict[str, Any], max_chars: Optional[int] = None) -> Dict[str, Any]:
    """JSON reply; ``ok: false`` payloads are flagged as MCP errors."""
    raw = json_content(result, max_chars=max_chars) if max_chars else json_content(result)
    return {**raw, "isError": True} if result.get("ok") is False else raw


def _session_reply(s: _DebugSession, result: Dict[str, Any], max_chars: Optional[int] = None) -> Dict[str, Any]:
    """Successful session mutations report the session id and new revision."""
    if result.get("ok") is not False:
        result = {
            **result,
            "session_id": s.session_id or None,
            "revision": s.state_revision() if s.started else None,
        }
    return _reply(result, max_chars)


def _absolute_path(value: Any, suffix: str = "") -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= _MAX_PATH_TEXT
        and os.path.isabs(value)
        and (not suffix or value.lower().endswith(suffix))
    )


async def handle_launch_app(arguments: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_session()
    s.launch()
    return _session_reply(s, {"status": "ok", "message": "Debug session launched"})


async def handle_close_app(arguments: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_session()
    s.close()
    return json_content({"status": "ok", "message": "Debug session closed"})


async def handle_get_runtime_status(arguments: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_session()
    return json_content(s.status())


async def handle_set_game_library_path(arguments: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_session()
    path = arguments.get("path")
    if not _absolute_path(path) or not os.path.isdir(path):
        return _reply({"ok": False, "error": "path must be an absolute game directory"})
    if not s.started:
        s.launch()
    result = s.set_game_path(os.path.normpath(path))
    return _session_reply(s, {**result, "game_tag": s.game_tag})


async def handle_verify_game_library(arguments: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_session()
    return json_content(s.verify_game())


async def handle_load_model(arguments: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_session()
    if not s.started:
        s.launch()
    result = s.load_model(arguments.get("resref", ""))
    return _session_reply(s, result, max_chars=120_000)


async def handle_get_loaded_asset_info(arguments: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_session()
    return json_content(s.get_asset_info())


async def handle_list_animations(arguments: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_session()
    anims = s.list_animations()
    return json_content({"animations": anims, "count": len(anims)})


async def handle_set_animation(arguments: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_session()
    return _session_reply(s, s.set_animation(arguments.get("animation_name", "")))


async def handle_set_animation_time(arguments: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_session()
    # No float() coercion: NaN, infinities, strings and booleans are refused.
    return _session_reply(s, s.set_animation_time(arguments.get("time")))


async def handle_set_bind_pose(arguments: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_session()
    return _session_reply(s, s.set_bind_pose())


async def handle_set_camera_preset(arguments: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_session()
    return _session_reply(s, s.set_camera(arguments.get("preset", "front")))


async def handle_capture_viewport(arguments: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_session()
    w, h = arguments.get("width", 512), arguments.get("height", 512)
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (w, h)):
        return _reply({"ok": False, "error": "width/height must be integers"})
    out = arguments.get("output_path")
    if out is not None and not _absolute_path(out, ".png"):
        return _reply({"ok": False, "error": "output_path must be an absolute .png path"})
    exclusive = arguments.get("overwrite", True) is False
    return _session_reply(s, s.capture_viewport(w, h, out, exclusive=exclusive))


async def handle_capture_validation_set(arguments: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_session()
    name = arguments.get("set_name", "validation")
    out_dir = arguments.get("output_dir")
    # The default used to be inside the Ghost checkout.
    if not _absolute_path(out_dir):
        return _reply({"ok": False, "error": "output_dir must be an absolute directory path"})
    exclusive = arguments.get("overwrite", True) is False
    if exclusive and os.path.isdir(out_dir) and os.listdir(out_dir):
        return _reply({"ok": False, "error": "Refusing to write into a non-empty capture directory"})
    try:
        result = s.capture_validation_set(name, out_dir, exclusive=exclusive)
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
    return _session_reply(s, result, max_chars=120_000)


async def handle_get_skinning_state(arguments: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_session()
    return json_content(s.get_skinning_state())


async def handle_get_renderer_state(arguments: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_session()
    return json_content(s.get_renderer_state())


async def handle_get_bone_hierarchy(arguments: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_session()
    return json_content(s.get_bone_hierarchy())


async def handle_get_bone_map(arguments: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_session()
    mesh = arguments.get("mesh_name", "")
    return json_content(s.get_bone_map_for_mesh(mesh))


async def handle_get_palette_remap_table(arguments: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_session()
    return json_content(s.get_palette_remap_table())


async def handle_get_bind_pose_matrices(arguments: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_session()
    bone_names = arguments.get("bone_names")
    return json_content(s.get_bind_pose_matrices(bone_names))


async def handle_get_animated_pose_matrices(arguments: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_session()
    bone_names = arguments.get("bone_names")
    return json_content(s.get_animated_pose_matrices(bone_names))


async def handle_get_uploaded_palette(arguments: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_session()
    return json_content(s.get_uploaded_palette())


async def handle_sample_vertex_influences(arguments: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_session()
    mesh = arguments.get("mesh_name", "")
    vi = arguments.get("vertex_indices")
    mx = int(arguments.get("max_samples", 10))
    return json_content(s.sample_vertex_influences(mesh, vi, mx))


async def handle_compare_cpu_gpu(arguments: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_session()
    mesh = arguments.get("mesh_name", "")
    vi = arguments.get("vertex_indices")
    mx = int(arguments.get("max_verts", 30))
    return json_content(s.compare_cpu_gpu_skinning(mesh, vi, mx))


async def handle_export_debug_bundle(arguments: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_session()
    out_dir = arguments.get("output_dir")
    try:
        result = s.export_debug_bundle(out_dir, exclusive=arguments.get("overwrite", True) is False)
    except FileExistsError:
        result = {"ok": False, "error": "Refusing to overwrite existing bundle file"}
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
    raw = json_content(result, max_chars=120_000)
    return {**raw, "isError": True} if not result.get("ok") else raw
