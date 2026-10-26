"""
KotOR Animation Engine
======================
Handles animation playback, interpolation, export (BVH / FBX-ASCII / JSON),
and import (JSON / BVH) for KotOR MDL models.

Animation data pipeline:
  Binary MDL  →  load_model_from_bytes (PyKotor)  →  KotorModel.animations  →  AnimationEngine
                                                              ↕
                                               viewport pose evaluation
"""

import math
import json
import logging
import os
import struct
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from ..geometry.model_data import (
    Animation, AnimEvent, GameVersion, ModelNode, KotorModel,
    _quat_mul, _quat_rotate, _quat_normalize_bind
)
try:
    from ..scene.node_identity import classify_scene_model, classify_scene_node
except Exception:  # pragma: no cover - animation must import before scene package assembly
    def classify_scene_model(model, resource_ref=None):  # type: ignore[no-redef]
        class _Identity:
            asset_kind = "model"
            animation_kind = "skeletal" if bool(getattr(model, "animations", None)) else "static"
            skeleton_kind = "none"
            joint_names = frozenset()
            animated_node_names = frozenset()
            dummy_node_names = frozenset()
        return _Identity()

    def classify_scene_node(node, identity=None) -> str:  # type: ignore[no-redef]
        try:
            if bool(getattr(node, "is_skin", False)):
                return "skin_mesh"
            if bool(getattr(node, "is_mesh", False)):
                return "mesh"
            if bool(getattr(node, "is_dummy", False)):
                return "dummy"
        except Exception:
            pass
        return str(getattr(node, "type_label", "") or "node")

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
#  Super-model chain resolver
# ─────────────────────────────────────────────────────────────────
#
# KotOR character / creature / placeable models rarely carry their own
# animation blocks.  Each model stores a ``supermodel`` resref (e.g.
# "S_Female02") naming a parent model that contains the shared animation
# library.  Supermodels can themselves have supermodels, forming a chain:
#
#     c_bantha → S_Female02 → S_Female03 → NULL
#     p_head  → P_BastilaBB → S_Female02 → S_Female03 → NULL
#
# Animation lookup is bottom-to-top (first match wins): own animations,
# then supermodel, then supermodel's supermodel, and so on until NULL.
#
# References:
#   • xoreos              src/graphics/aurora/model.cpp
#                         Model::getAnimation() / Model::getAnimationScale()
#   • KotorBlender        io_scene_kotor/scene/model.py (animscale rules)
#   • Deadly Stream wiki  "Animations are always pulled from the lowest
#                         (i.e. first) instance in the chain."
#
# Scale semantics (xoreos + KotorBlender agree):
#   * The model's OWN animations are played with scale factor 1.0 — no
#     animscale multiplication, even if ``model.anim_scale != 1.0``.
#   * Animations inherited from a supermodel are scaled by the PRODUCT
#     of ``anim_scale`` values along the chain from the requesting
#     model down to (not including) the owning model.
#   * Scaling applies to POSITION-delta keyframes only.  Orientation and
#     other channels are not multiplied.


class SuperModelResolver:
    """Resolves super-model animation chains for KotOR models.

    Keeps a process-wide cache of loaded super-models to avoid duplicate
    file I/O (mirroring xoreos' ``ModelCache``).  A single
    ``ResourceManager`` is configured once — the resolver uses its
    ``load_model(resref, game)`` API to fetch supermodels on demand.
    """

    # Class-level cache: lower(resref) → KotorModel | None
    # ``None`` is stored for known-missing resrefs so repeated lookups are
    # still O(1).  Tests can invoke ``clear_cache()`` between runs.
    _cache: Dict[Tuple[str, str], Optional[KotorModel]] = {}

    # Configured by ``configure()``.  When None, the resolver still works
    # for in-memory / pre-loaded models but cannot load chains from disk.
    _resource_manager: Any = None
    _resource_manager_revision = 0

    _NULL_REFS = frozenset({'', 'null', 'none'})

    @classmethod
    def configure(cls, resource_manager: Any) -> None:
        """Install the ResourceManager used to load supermodel MDL/MDX.

        Safe to call repeatedly. Cached chains remain valid only while both
        the manager object and its published resource revision are unchanged.
        """

        try:
            revision = max(0, int(getattr(resource_manager, "revision", 0) or 0))
        except (TypeError, ValueError):
            revision = 0
        if resource_manager is not cls._resource_manager or revision != cls._resource_manager_revision:
            cls.clear_cache()
        cls._resource_manager = resource_manager
        cls._resource_manager_revision = revision

    @classmethod
    def clear_cache(cls) -> None:
        """Empty the resolver cache (call from tests / between game installs)."""
        cls._cache.clear()

    @classmethod
    def prime_cache(cls, resref: str, model: Optional[KotorModel], game: Optional[str] = None) -> None:
        """Inject a pre-loaded model into the cache (used by tests)."""
        cls._cache[(cls._cache_game_key(game), resref.lower())] = model

    @classmethod
    def _cache_game_key(cls, game: Optional[str]) -> str:
        text = str(game or "").strip().upper()
        if text in {"K1", "K2"}:
            return text
        if text in {"1", "GAMEVERSION.K1"}:
            return "K1"
        if text in {"2", "GAMEVERSION.K2"}:
            return "K2"
        if "KOTOR2" in text or "KOTOR 2" in text or "KOTOR II" in text:
            return "K2"
        return "K1"

    @classmethod
    def _is_null_ref(cls, resref: Optional[str]) -> bool:
        if not resref:
            return True
        return resref.strip().lower() in cls._NULL_REFS

    @classmethod
    def load_supermodel(
        cls,
        resref: str,
        game: Optional[str] = None,
    ) -> Optional[KotorModel]:
        """Load a supermodel by resref, honouring the cache.

        ``game`` is passed verbatim to ``ResourceManager.load_model`` when
        set; defaults to 'K1' to match the existing RM default.

        Returns ``None`` when the resref is NULL-like, when no resource
        manager is configured, or when the model cannot be located.
        """
        if cls._is_null_ref(resref):
            return None
        game_tag = cls._cache_game_key(game)
        key = (game_tag, resref.lower())
        if key in cls._cache:
            return cls._cache[key]

        if cls._resource_manager is None:
            log.debug(
                "SuperModelResolver: no resource_manager, cannot load %r",
                resref,
            )
            cls._cache[key] = None
            return None

        try:
            # Animation inheritance is part of the target game's model
            # contract.  Prefer the strict loader when the configured resource
            # provider exposes it so a K2 preview cannot inherit a K1-only
            # supermodel.  Lightweight/legacy providers retain compatibility.
            loader = getattr(cls._resource_manager, "load_model_strict", None)
            if not callable(loader):
                loader = cls._resource_manager.load_model
            super_model = loader(resref, game_tag)
        except Exception as exc:  # pragma: no cover - defensive
            log.debug(
                "SuperModelResolver: load_model(%r) raised %s",
                resref, exc,
            )
            super_model = None

        cls._cache[key] = super_model
        # Proactively pre-load the rest of the chain so later lookups are hot.
        next_super_ref = str(getattr(super_model, "supermodel", "") or "")
        if super_model is not None and not cls._is_null_ref(next_super_ref):
            cls.load_supermodel(next_super_ref, game_tag)
        return super_model

    @classmethod
    def resolve_animation(
        cls,
        model: KotorModel,
        anim_name: str,
        game: Optional[str] = None,
    ) -> Tuple[Optional[Animation], float]:
        """Walk the super-model chain to find an animation.

        Returns ``(animation, cumulative_anim_scale)``.  When the
        animation is defined on ``model`` itself, ``cumulative_anim_scale``
        is always ``1.0`` (own animations are never scaled).

        When the animation is inherited, the returned scale is the
        product of ``anim_scale`` values from ``model`` down through each
        intermediate supermodel, NOT including the owning model's own
        ``anim_scale`` (the owner animates at its natural rate; the
        chain scale only accounts for the import-space conversion of
        every step that delegated the lookup).

        On lookup failure returns ``(None, 1.0)``.
        """
        nl = anim_name.lower()

        # 1. Local-model fast path (own animations are never scaled).
        for a in model.animations:
            if a.name.lower() == nl:
                return (a, 1.0)

        # 2. Walk up the supermodel chain.  ``cumulative`` accumulates the
        #    anim_scale values of every step we had to traverse.
        cumulative: float = model.anim_scale if model.anim_scale else 1.0
        super_ref   = model.supermodel
        visited     = {model.name.lower()}

        while not cls._is_null_ref(super_ref):
            key = super_ref.lower()
            if key in visited:
                log.warning(
                    "SuperModelResolver: cycle detected at %r (chain=%s)",
                    super_ref, sorted(visited),
                )
                return (None, 1.0)
            visited.add(key)

            super_model = cls.load_supermodel(super_ref, game)
            if super_model is None:
                break

            for a in super_model.animations:
                if a.name.lower() == nl:
                    return (a, cumulative)

            # Multiply the next step's scale into the cumulative factor.
            step_scale = super_model.anim_scale if super_model.anim_scale else 1.0
            cumulative *= step_scale
            super_ref = super_model.supermodel

        return (None, 1.0)

    @classmethod
    def list_all_animations(
        cls,
        model: KotorModel,
        game: Optional[str] = None,
    ) -> List[Tuple[str, str, float]]:
        """Return ``[(anim_name, source_model_name, cumulative_scale), …]``
        for every animation available through the full super-model chain.

        If a name appears on both ``model`` and a supermodel, the model's
        own definition wins (scale = 1.0, source = model.name).  The
        returned list is sorted alphabetically by name for stable UI
        ordering.
        """
        entries: Dict[str, Tuple[str, str, float]] = {}

        for a in model.animations:
            entries[a.name.lower()] = (a.name, model.name, 1.0)

        cumulative: float = model.anim_scale if model.anim_scale else 1.0
        super_ref   = model.supermodel
        visited     = {model.name.lower()}

        while not cls._is_null_ref(super_ref):
            key = super_ref.lower()
            if key in visited:
                break
            visited.add(key)

            super_model = cls.load_supermodel(super_ref, game)
            if super_model is None:
                break

            for a in super_model.animations:
                nl = a.name.lower()
                if nl not in entries:
                    entries[nl] = (a.name, super_model.name, cumulative)

            step_scale = super_model.anim_scale if super_model.anim_scale else 1.0
            cumulative *= step_scale
            super_ref = super_model.supermodel

        return sorted(entries.values(), key=lambda tup: tup[0].lower())

    @classmethod
    def animation_source_type(
        cls,
        model: KotorModel,
        anim_name: str,
        source_model_name: str,
        game: Optional[str] = None,
    ) -> str:
        """Classify an effective animation source as local, inherited, or override.

        ``override`` means the effective source wins over a same-named clip
        later in the supermodel chain. Bastila's local head animations are a
        common example: the clips live on ``P_BastilaH`` but shadow shared
        supermodel clips of the same names.
        """
        name_key = str(anim_name or "").lower()
        source_key = str(source_model_name or "").lower()
        own_key = str(getattr(model, "name", "") or "").lower()
        if not name_key or not source_key:
            return "inherited"

        chain: list[KotorModel] = []
        visited: set[str] = set()
        current: Optional[KotorModel] = model
        while current is not None:
            current_key = str(getattr(current, "name", "") or "").lower()
            if not current_key or current_key in visited:
                break
            visited.add(current_key)
            chain.append(current)
            super_ref = getattr(current, "supermodel", "")
            current = cls.load_supermodel(super_ref, game)

        source_index = -1
        for index, chain_model in enumerate(chain):
            if str(getattr(chain_model, "name", "") or "").lower() == source_key:
                source_index = index
                break

        shadows_ancestor = False
        if source_index >= 0:
            for chain_model in chain[source_index + 1:]:
                for anim in getattr(chain_model, "animations", []) or []:
                    if str(getattr(anim, "name", "") or "").lower() == name_key:
                        shadows_ancestor = True
                        break
                if shadows_ancestor:
                    break

        if source_key == own_key:
            return "override" if shadows_ancestor else "local"
        return "override" if shadows_ancestor else "inherited"


# ─────────────────────────────────────────────────────────────────
#  Interpolation helpers
# ─────────────────────────────────────────────────────────────────

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp3(a: Tuple, b: Tuple, t: float) -> Tuple:
    return (
        _lerp(a[0], b[0], t),
        _lerp(a[1], b[1], t),
        _lerp(a[2], b[2], t),
    )


def _slerp(q1: List[float], q2: List[float], t: float) -> List[float]:
    """Spherical linear interpolation between two quaternions [x,y,z,w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    dot = x1*x2 + y1*y2 + z1*z2 + w1*w2
    # Ensure shortest path
    if dot < 0:
        x2, y2, z2, w2 = -x2, -y2, -z2, -w2
        dot = -dot
    if dot > 0.9995:
        # Linear interpolation for nearly identical quats
        rx = x1 + t*(x2-x1)
        ry = y1 + t*(y2-y1)
        rz = z1 + t*(z2-z1)
        rw = w1 + t*(w2-w1)
    else:
        theta_0 = math.acos(dot)
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        sin_theta_0 = math.sin(theta_0)
        s1 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s2 = sin_theta / sin_theta_0
        rx = s1*x1 + s2*x2
        ry = s1*y1 + s2*y2
        rz = s1*z1 + s2*z2
        rw = s1*w1 + s2*w2
    # Normalize
    mag = math.sqrt(rx*rx + ry*ry + rz*rz + rw*rw)
    if mag > 1e-9:
        rx /= mag; ry /= mag; rz /= mag; rw /= mag
    return [rx, ry, rz, rw]


def _is_finite_vec(v) -> bool:
    """Return True if all elements of v are finite (not NaN/Inf)."""
    return all(math.isfinite(x) for x in v)


def _ensure_quat_sign_consistency(values: List[List[float]]) -> List[List[float]]:
    """
    Ensure that consecutive quaternion keyframes are in the same hemisphere
    (dot product >= 0) so that SLERP always takes the shortest arc.

    Background: KotOR stores some quaternion sequences where the exporter
    flips the sign of individual keyframes (q and -q represent the same
    rotation but SLERP between them traces the long way around).  The
    _slerp() function's inner "if dot < 0: negate q2" guard handles
    two-keyframe cases, but for multi-keyframe sequences it only looks at
    adjacent pairs; if the base keyframe is already wrong the guard has no
    earlier reference to flip against.

    This pre-pass walks the keyframe sequence once (O(n)) and ensures each
    keyframe is in the same hemisphere as the PREVIOUS one.  The result is
    equivalent to picking the canonical sign for each quaternion based on
    the running reference direction — consistent with how xoreos normalises
    orientation tracks in animation.cpp.

    Only applied to quaternion channels (len(values[0]) == 4).
    Returns a new list; the original is not mutated.
    """
    if not values or len(values[0]) != 4:
        return values

    out: List[List[float]] = [list(values[0])]
    px, py, pz, pw = values[0]

    for v in values[1:]:
        x, y, z, w = v
        # dot with previous (sign-consistent) keyframe
        if (px*x + py*y + pz*z + pw*w) < 0.0:
            # Flip to the nearer hemisphere
            x, y, z, w = -x, -y, -z, -w
        out.append([x, y, z, w])
        px, py, pz, pw = x, y, z, w

    return out


_SANITIZED_CHANNEL_KEY = "_sanitized_channel_cache"


def _sanitize_channel(times: List[float], values: List[List[float]]):
    """One-time channel cleanup so per-frame sampling can skip validation.

    Drops keyframes with non-finite components (the documented NaN-skip
    semantics of ``_interp_channel``) and sign-normalizes quaternion
    sequences once, instead of rebuilding the sequence on every sample.
    """

    if not times or not values:
        return (), ()
    clean_times: List[float] = []
    clean_values: List[List[float]] = []
    for index in range(min(len(times), len(values))):
        value = values[index]
        if _is_finite_vec(value):
            clean_times.append(times[index])
            clean_values.append(list(value))
    if clean_values and len(clean_values[0]) == 4:
        clean_values = _ensure_quat_sign_consistency(clean_values)
    return clean_times, clean_values


def _channel_samples(ctrl: dict):
    """Cached sanitized (times, values) for a controller channel dict."""

    cached = ctrl.get(_SANITIZED_CHANNEL_KEY)
    if cached is None:
        cached = _sanitize_channel(ctrl.get("times") or [], ctrl.get("values") or [])
        ctrl[_SANITIZED_CHANNEL_KEY] = cached
    return cached


def _interp_channel_sanitized(times, values, t: float) -> Optional[List[float]]:
    """Lean sampler over a pre-sanitized channel (finite, sign-consistent)."""

    if not times:
        return None
    if t <= times[0]:
        return list(values[0])
    if t >= times[-1]:
        return list(values[-1])
    lo, hi = 0, len(times) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if times[mid] <= t:
            lo = mid
        else:
            hi = mid
    tf = (t - times[lo]) / max(1e-9, times[hi] - times[lo])
    v0, v1 = values[lo], values[hi]
    if len(v0) == 4:
        return _slerp(v0, v1, tf)
    if len(v0) == 3:
        return list(_lerp3(v0, v1, tf))
    return [_lerp(v0[k], v1[k], tf) for k in range(min(len(v0), len(v1)))]


def _interp_channel(times: List[float], values: List[List[float]],
                    t: float) -> Optional[List[float]]:
    """
    Interpolate a controller channel at time t.
    Returns the interpolated value list, or None if no keys.

    NaN/Inf safety: skips keyframe pairs that contain non-finite values
    so that corrupt data in bezier-spline tail regions does not propagate.

    Quaternion sign-consistency: before interpolating a quaternion channel
    (4-component values) the keyframe sequence is pre-normalised so that
    every consecutive pair has a non-negative dot product.  This ensures
    SLERP always takes the shortest arc even when the exporter stored
    antipodal keyframes (q and -q are the same rotation but SLERP between
    them would spin 360° without this fix).
    See _ensure_quat_sign_consistency() for full rationale.
    """
    if not times or not values:
        return None

    # Pre-normalise quaternion channels for sign consistency
    # (only for 4-component channels; 3-component position channels unchanged)
    if values and len(values[0]) == 4:
        values = _ensure_quat_sign_consistency(values)

    # Find last valid keyframe at or before t
    # Walk forward/backward to skip NaN-contaminated entries
    def safe_val(idx):
        v = values[idx]
        return v if _is_finite_vec(v) else None

    if t <= times[0]:
        v = safe_val(0)
        return list(v) if v else None
    if t >= times[-1]:
        # Walk backward to find last finite keyframe
        for i in range(len(times) - 1, -1, -1):
            v = safe_val(i)
            if v is not None:
                return list(v)
        return None

    # Binary search for bracketing keys
    lo, hi = 0, len(times) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if times[mid] <= t:
            lo = mid
        else:
            hi = mid

    # Expand lo/hi to valid (finite) values
    lo2 = lo
    while lo2 >= 0 and safe_val(lo2) is None:
        lo2 -= 1
    hi2 = hi
    while hi2 < len(times) and safe_val(hi2) is None:
        hi2 += 1

    if lo2 < 0 and hi2 >= len(times):
        return None
    if lo2 < 0:
        return list(safe_val(hi2))
    if hi2 >= len(times):
        return list(safe_val(lo2))

    tf = (t - times[lo2]) / max(1e-9, times[hi2] - times[lo2])
    v0, v1 = values[lo2], values[hi2]
    if len(v0) == 4:   # quaternion – slerp
        return _slerp(v0, v1, tf)
    elif len(v0) == 3:
        return list(_lerp3(v0, v1, tf))
    else:
        return [_lerp(v0[k], v1[k], tf) for k in range(min(len(v0), len(v1)))]


# ─────────────────────────────────────────────────────────────────
#  Pose snapshot
# ─────────────────────────────────────────────────────────────────

@dataclass
class NodePose:
    """Animated pose for a single node at a given time."""
    name:     str
    position: Tuple[float, float, float]        = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # xyzw
    scale:    float = 1.0
    # Per-node material animation (CTRL_MESH_ALPHA=132, CTRL_MESH_SELFILLUMCOLOR=100)
    alpha:      Optional[float]                     = None  # None = use bind-pose value
    selfillum:  Optional[Tuple[float,float,float]]  = None  # None = use bind-pose value


@dataclass
class AnimPose:
    """Complete model pose at a given time step."""
    time:   float = 0.0
    nodes:  Dict[str, NodePose] = field(default_factory=dict)


@dataclass(frozen=True)
class AuroraTransform:
    """Parent-local or world-space Aurora transform using GhostRigger XYZW quats."""

    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


@dataclass
class EvaluatedAuroraPose:
    """Deterministic headless pose evaluation for animation-controller gates."""

    time: float
    local_transforms_by_node: Dict[str, AuroraTransform] = field(default_factory=dict)
    world_transforms_by_node: Dict[str, AuroraTransform] = field(default_factory=dict)


def _normalize_quat_xyzw(quat: Tuple[float, float, float, float] | List[float]) -> Tuple[float, float, float, float]:
    x, y, z, w = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    mag_sq = x * x + y * y + z * z + w * w
    if mag_sq <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    mag = math.sqrt(mag_sq)
    return (x / mag, y / mag, z / mag, w / mag)


def _controller_matches(ctrl: Dict[str, Any], controller_type: int, controller_name: str) -> bool:
    raw_type = ctrl.get("type")
    if raw_type == controller_type:
        return True
    return str(ctrl.get("name", "")).lower() == controller_name


def _find_controller(
    controllers: List[Dict[str, Any]],
    controller_type: int,
    controller_name: str,
) -> Optional[Dict[str, Any]]:
    """Return the first matching controller channel, if present."""

    return next(
        (
            controller
            for controller in controllers
            if _controller_matches(controller, controller_type, controller_name)
        ),
        None,
    )


_C_KATH_ROOT_BIND_POSITION = (0.00186379, -0.37697601, 0.81954002)
_ABSOLUTE_LOCAL_POSITION_CACHE_KEY = "_gr_absolute_local_position"


def _position_controller_is_absolute_local(
    model: KotorModel,
    node_name: str,
    bind_position: Tuple[float, float, float] | List[float],
    controller: Optional[Dict[str, Any]],
) -> bool:
    """Identify the verified retail K1 ``c_kathroot`` position convention.

    Every local animation in the retail K1 ``c_kath`` model authors the
    ``c_kathroot`` POSITION rows as absolute node-local coordinates.  The row
    shapes are not uniform: combat clips commonly finish on the bind row while
    locomotion clips (``cwalk``, ``crun``, and the pause/turn family) keep a
    different authored Y/Z baseline for the whole clip.  Keying this decision
    to the closing row therefore fixed only part of the animation set and left
    locomotion with the bind position added a second time.

    Keep the exception constrained to the exact game/model/node/bind contract;
    other Aurora POSITION controllers retain the ordinary bind-plus-delta
    semantics.
    """

    if controller is None:
        return False
    cached = controller.get(_ABSOLUTE_LOCAL_POSITION_CACHE_KEY)
    if isinstance(cached, bool):
        return cached

    model_name = str(getattr(model, "name", "") or "").strip().casefold()
    contains_c_kath_root = model_name == "c_kath"
    if not contains_c_kath_root:
        try:
            contains_c_kath_root = any(
                str(getattr(node, "name", "") or "").strip().casefold() == "c_kath"
                for node in model.all_nodes()
            )
        except Exception:
            contains_c_kath_root = False

    result = (
        getattr(model, "game_version", None) == GameVersion.K1
        and contains_c_kath_root
        and str(node_name or "").strip().casefold() == "c_kathroot"
        and len(bind_position) >= 3
        and all(
            math.isclose(
                float(bind_position[index]),
                _C_KATH_ROOT_BIND_POSITION[index],
                abs_tol=1e-6,
            )
            for index in range(3)
        )
    )
    controller[_ABSOLUTE_LOCAL_POSITION_CACHE_KEY] = result
    return result


def _compose_position_controller_value(
    bind_position: Tuple[float, float, float] | List[float],
    controller_value: Tuple[float, float, float] | List[float],
    *,
    animation_scale: float = 1.0,
    absolute_local: bool = False,
    absolute_reference: Optional[Tuple[float, float, float] | List[float]] = None,
) -> Tuple[float, float, float]:
    """Compose one authored POSITION row with its bind transform."""

    value = (
        float(controller_value[0]),
        float(controller_value[1]),
        float(controller_value[2]),
    )
    if absolute_local:
        if absolute_reference is not None and len(absolute_reference) >= 3:
            # Character preview keeps the actor at its authored scene origin
            # while preserving the clip's root-motion delta.  Retail c_kath
            # locomotion clips use a different absolute root baseline from the
            # static MDL; applying that baseline directly makes the creature
            # jump toward the camera as soon as playback starts.
            return (
                float(bind_position[0]) + value[0] - float(absolute_reference[0]),
                float(bind_position[1]) + value[1] - float(absolute_reference[1]),
                float(bind_position[2]) + value[2] - float(absolute_reference[2]),
            )
        return value

    scale = float(animation_scale)
    return (
        float(bind_position[0]) + value[0] * scale,
        float(bind_position[1]) + value[1] * scale,
        float(bind_position[2]) + value[2] * scale,
    )


_SORTED_SAMPLING_TIMES_KEY = "_gr_sorted_sampling_times"


def mark_controller_times_sorted_for_sampling(controller: Dict[str, Any]) -> bool:
    """Mark one immutable-in-time-order list channel for zero-copy sampling.

    The marker stores the exact list object, so replacing ``times``
    automatically invalidates it. Callers retaining the list must keep its
    order unchanged; controller values may still be replaced in place.
    """

    raw_times = controller.get("times", [])
    raw_values = controller.get("values", [])
    safe = (
        isinstance(raw_times, list)
        and isinstance(raw_values, list)
        and bool(raw_times)
        and len(raw_times) == len(raw_values)
        and all(type(value) is float and math.isfinite(value) for value in raw_times)
        and all(isinstance(value, list) for value in raw_values)
        and all(raw_times[index - 1] <= raw_times[index] for index in range(1, len(raw_times)))
    )
    if safe:
        controller[_SORTED_SAMPLING_TIMES_KEY] = raw_times
        return True
    controller.pop(_SORTED_SAMPLING_TIMES_KEY, None)
    return False


def _sample_controller_absolute(
    controllers: List[Dict[str, Any]],
    controller_type: int,
    controller_name: str,
    time_seconds: float,
    *,
    clamp: bool,
) -> Optional[List[float]]:
    """Sample a controller channel as an absolute parent-local value."""

    for ctrl in controllers:
        if not _controller_matches(ctrl, controller_type, controller_name):
            continue
        raw_times = ctrl.get("times", [])
        raw_values = ctrl.get("values", [])
        if not raw_times or not raw_values:
            return None

        # Dense validation clips can contain thousands of keys and are sampled
        # thousands of times. Explicitly marked float/list channels have
        # already paid the order-validation cost once and can be sampled
        # without repeatedly copying and sorting every row. All unmarked
        # inputs keep the exact historical normalization/stable-sort path.
        direct_channel = ctrl.get(_SORTED_SAMPLING_TIMES_KEY) is raw_times
        if direct_channel:
            times = raw_times
            values = raw_values
        else:
            times = [float(value) for value in raw_times]
            values = [list(value) for value in raw_values]
            rows = sorted(zip(times, values), key=lambda item: item[0])
            times = [row[0] for row in rows]
            values = [row[1] for row in rows]
        sample_time = time_seconds
        if clamp:
            sample_time = max(times[0], min(times[-1], sample_time))
        return _interp_channel(times, values, sample_time)
    return None


def _compose_transform(parent: Optional[AuroraTransform], local: AuroraTransform) -> AuroraTransform:
    """Compose parent and local transforms with parent-first FK semantics."""

    local_rot = _normalize_quat_xyzw(local.rotation)
    if parent is None:
        return AuroraTransform(position=local.position, rotation=local_rot)

    parent_rot = _normalize_quat_xyzw(parent.rotation)
    rotated_pos = _quat_rotate(parent_rot, local.position)
    world_pos = (
        parent.position[0] + rotated_pos[0],
        parent.position[1] + rotated_pos[1],
        parent.position[2] + rotated_pos[2],
    )
    world_rot = _normalize_quat_xyzw(_quat_mul(parent_rot, local_rot))
    return AuroraTransform(position=world_pos, rotation=world_rot)


def evaluate_aurora_animation_pose(
    model: KotorModel,
    animation_block: Animation,
    time_seconds: float,
    *,
    clamp: bool = True,
) -> EvaluatedAuroraPose:
    """Evaluate one Aurora animation block with engine-verified key semantics.

    This helper is intentionally independent from viewport playback. It exists
    as a deterministic validation oracle for export/retargeting gates:
    POSITION keys are DELTA offsets ADDED to the rest-local position (matching
    AnimationEngine._eval_node, xoreos ``arePositionFramesRelative()==true``
    and KotorBlender ``p1 = restloc + animscale*val``; T2558 — this oracle
    used to REPLACE positions, which offset every position-keyed bone by its
    rest position and made audit poses disagree with the viewport), ORIENTATION
    keys replace the rest-local rotation, unkeyed components stay at rest,
    and parent transforms propagate by FK.
    """

    anim_nodes = {
        str(node.name or "").lower(): node
        for node in getattr(animation_block, "nodes", [])
        if str(node.name or "").strip()
    }
    pose = EvaluatedAuroraPose(time=float(time_seconds))

    for node in model.all_nodes():
        local_position = tuple(float(value) for value in node.position)
        local_rotation = _normalize_quat_xyzw(list(node.rotation))
        anim_node = anim_nodes.get(str(node.name or "").lower())

        if anim_node is not None:
            position_controller = _find_controller(
                anim_node.controllers,
                AnimationEngine.CTRL_POSITION,
                "position",
            )
            sampled_position = _sample_controller_absolute(
                [position_controller] if position_controller is not None else [],
                AnimationEngine.CTRL_POSITION,
                "position",
                float(time_seconds),
                clamp=clamp,
            )
            if sampled_position is not None and len(sampled_position) >= 3:
                local_position = _compose_position_controller_value(
                    node.position,
                    sampled_position,
                    absolute_local=_position_controller_is_absolute_local(
                        model,
                        node.name,
                        node.position,
                        position_controller,
                    ),
                )

            sampled_rotation = _sample_controller_absolute(
                anim_node.controllers,
                AnimationEngine.CTRL_ORIENTATION,
                "orientation",
                float(time_seconds),
                clamp=clamp,
            )
            if sampled_rotation is not None and len(sampled_rotation) >= 4:
                local_rotation = _normalize_quat_xyzw(sampled_rotation[:4])

        local = AuroraTransform(position=local_position, rotation=local_rotation)
        parent_world = None
        if node.parent is not None:
            parent_world = pose.world_transforms_by_node.get(node.parent.name)
        world = _compose_transform(parent_world, local)
        pose.local_transforms_by_node[node.name] = local
        pose.world_transforms_by_node[node.name] = world

    return pose


# ─────────────────────────────────────────────────────────────────
#  Animation Engine
# ─────────────────────────────────────────────────────────────────

class AnimationEngine:
    """
    Manages animation playback for a KotorModel.

    Usage:
        engine = AnimationEngine(model)
        engine.play('cwalk')
        pose   = engine.evaluate(0.5)   # pose at t=0.5s
        engine.stop()
    """

    # Controller type IDs  (verified against KotorBlender types.py and xoreos)
    CTRL_POSITION       = 8
    CTRL_ORIENTATION    = 20
    CTRL_SCALE          = 36
    CTRL_SELFILLUMCOLOR = 100   # CTRL_MESH_SELFILLUMCOLOR: r,g,b (3 floats)
    CTRL_ALPHA_XOREOS   = 128   # xoreos CTRL_ALPHA = 128 (some tools use this)
    CTRL_ALPHA          = 132   # CTRL_MESH_ALPHA = 132 (KotorBlender convention)

    def __init__(self, model: KotorModel):
        self.model    = model
        try:
            from .skinning_profiles import resolve_skinning_profile
            self.skinning_profile = resolve_skinning_profile(model)
        except Exception as exc:
            log.debug("Skinning profile resolution failed for %s: %s", getattr(model, "name", ""), exc)
            self.skinning_profile = None
        self._current_anim: Optional[Animation] = None
        self._loop    = True
        self._playing = False
        self._time    = 0.0
        # Build lookup: name → base-pose node
        self._base_nodes: Dict[str, ModelNode] = {}
        self._scene_identity = classify_scene_model(model)
        self._node_kinds_by_name: Dict[str, str] = {}
        self._node_scene_keys_by_name: Dict[str, str] = {}
        if model.root_node:
            for n in model.all_nodes():
                name_key = n.name.lower()
                # BAS preview models retain complete attachment DAGs beneath
                # body sockets.  Those DAGs intentionally repeat ordinary
                # Odyssey names such as ``rootdummy`` and ``torso_g``.  A
                # single body animation pose must use the primary body node's
                # bind transform for its rest-local position delta; allowing a
                # later head attachment node to overwrite it collapses the
                # body by roughly the body's rootdummy height.  Head-local
                # animation remains independent through the attachment-source
                # AnimationEngine in mesh_render_data.
                existing = self._base_nodes.get(name_key)
                if existing is not None:
                    existing_is_attachment = bool(
                        getattr(existing, "_gr_bas_attachment_layer", False)
                    )
                    incoming_is_attachment = bool(
                        getattr(n, "_gr_bas_attachment_layer", False)
                    )
                    # Preserve the first native DAG node for duplicate Odyssey
                    # names. K2 PFBCM contains a second ``lhand_g`` nested
                    # beneath the real wrist joint; last-wins lookup selects
                    # that helper, turns its parent into a name-based self
                    # cycle, and stretches the hand away from the sleeve.
                    # A primary body node may still replace an attachment node
                    # in a BAS composite, but attachments and later native
                    # duplicates never replace an established body joint.
                    if not existing_is_attachment or incoming_is_attachment:
                        continue
                self._base_nodes[name_key] = n
                self._node_kinds_by_name[name_key] = str(
                    getattr(n, "_gr_scene_node_kind", "") or classify_scene_node(n, self._scene_identity)
                )
                self._node_scene_keys_by_name[name_key] = str(
                    getattr(n, "_gr_scene_node_key", "") or name_key
                )
        # Cross-fade transition blending state
        self._blend_from_pose: Optional[AnimPose] = None
        self._blend_t:         float = 0.0   # blend fraction [0.0 → 1.0]
        self._blend_elapsed:   float = 0.0   # wall-clock elapsed since blend start (s)
        self._blend_duration:  float = 0.0   # total blend duration in seconds
        # Phase-synchronized cross-fade support (Gregory §12.6.3)
        # When sync_phase=True is passed to play(), we start the new clip at the
        # same NORMALIZED phase as the previous clip, preventing foot-slip during
        # transitions between locomotion cycles of different durations.
        self._blend_sync_phase: bool  = False  # whether to use phase-synced blend
        self._blend_from_anim:  Optional[Animation] = None   # previous clip for phase eval
        self._blend_from_time:  float = 0.0   # previous clip's local time at blend start
        # Fired-event tracking so each event fires exactly once per loop
        self._fired_events: set = set()  # indices into current_anim.events
        # Super-model chain scaling (Phase 5).
        # Cumulative anim_scale product for the currently-playing animation.
        # 1.0 when the animation is defined on this model; > 0 otherwise.
        # Updated by ``_find_anim`` whenever a new animation is resolved.
        # Multiplied into POSITION-delta keyframe values so inherited
        # animations from a supermodel are scaled correctly
        # (matches KotorBlender ``p1 = restloc + animscale * val`` and
        # xoreos Model::getAnimationScale).
        self._current_anim_scale: float = 1.0

    # ── Playback control ────────────────────────────────────────────────────

    def node_kind(self, node_name: str) -> str:
        """Return the detected scene/animation role for an authored node name."""
        return self._node_kinds_by_name.get(str(node_name or "").lower(), "node")

    def is_joint_node(self, node_name: str) -> bool:
        """Return True when *node_name* is detected as a skeleton joint."""
        return self.node_kind(node_name) == "joint"

    def is_animated_node(self, node_name: str) -> bool:
        """Return True for authored nodes that receive animation controllers."""
        key = str(node_name or "").lower()
        if key in getattr(self._scene_identity, "animated_node_names", frozenset()):
            return True
        return self.node_kind(node_name) in {"animated_node", "animated_dummy"}

    def is_dummy_node(self, node_name: str) -> bool:
        """Return True for dummy helper nodes, including animated dummies."""
        return self.node_kind(node_name) in {"dummy", "animated_dummy"}

    def node_scene_key(self, node_name: str) -> str:
        """Return the import-qualified node key when the model belongs to a KMAX scene."""
        name_key = str(node_name or "").lower()
        return self._node_scene_keys_by_name.get(name_key, name_key)

    @property
    def scene_identity(self):
        return self._scene_identity

    @property
    def asset_kind(self) -> str:
        return str(getattr(self._scene_identity, "asset_kind", "model"))

    @property
    def animation_kind(self) -> str:
        return str(getattr(self._scene_identity, "animation_kind", "static"))

    @property
    def skeleton_kind(self) -> str:
        return str(getattr(self._scene_identity, "skeleton_kind", "none"))

    @property
    def joint_names(self) -> frozenset[str]:
        return frozenset(getattr(self._scene_identity, "joint_names", frozenset()))

    @property
    def animated_node_names(self) -> frozenset[str]:
        return frozenset(getattr(self._scene_identity, "animated_node_names", frozenset()))

    @property
    def dummy_node_names(self) -> frozenset[str]:
        return frozenset(getattr(self._scene_identity, "dummy_node_names", frozenset()))

    @property
    def is_skeletal_animation(self) -> bool:
        return self.animation_kind.startswith("skeletal")

    @property
    def is_rigid_animation(self) -> bool:
        return self.animation_kind == "rigid"

    @property
    def current_animation(self) -> Optional[Animation]:
        return self._current_anim

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def current_time(self) -> float:
        return self._time

    def play(self, anim_name: str, loop: bool = True, blend: bool = True,
             sync_phase: bool = False):
        """Start playing the named animation.

        If *blend* is True and an animation is already active, captures a
        pose snapshot and cross-fades from it to the new animation over the
        new animation's ``transition_time`` seconds.

        Parameters
        ----------
        anim_name   : name of the animation to play (case-insensitive)
        loop        : whether to loop the animation
        blend       : if True, cross-fade from the current pose
        sync_phase  : if True, start the new clip at the same NORMALIZED phase
                      (u = t/T) as the current clip.  This prevents foot-slip
                      when blending between locomotion cycles of different
                      durations (e.g. walk→run).  Has no effect if blend=False.

                      Reference: Gregory §12.6.3 — phase-synchronized cross-fade.
                      Normalized time u ∈ [0,1]: u_new = u_old means both clips
                      start at the same relative point in their cycle, keeping
                      foot contacts aligned throughout the blend.
        """
        anim = self._find_anim(anim_name)
        if anim is None:
            log.warning(f"Animation '{anim_name}' not found in model '{self.model.name}'")
            return False
        # Cross-fade: capture current pose before switching
        if blend and self._playing and self._current_anim is not None:
            self._blend_from_pose = self.evaluate()
            self._blend_t         = 0.0
            self._blend_duration  = max(0.0, anim.transition_time)
            # Phase-synchronization: record previous clip's state for phase-aware eval
            self._blend_sync_phase = sync_phase
            self._blend_from_anim  = self._current_anim
            self._blend_from_time  = self._time
        else:
            self._blend_from_pose  = None
            self._blend_t          = 0.0
            self._blend_elapsed    = 0.0
            self._blend_duration   = 0.0
            self._blend_sync_phase = False
            self._blend_from_anim  = None
            self._blend_from_time  = 0.0
        self._current_anim = anim
        self._loop         = loop
        self._playing      = True
        self._fired_events = set()

        # Phase-sync: start new clip at the same normalized time as the old one
        if sync_phase and self._blend_from_anim is not None:
            old_length = max(0.001, self._blend_from_anim.length)
            old_phase  = self._blend_from_time / old_length          # u ∈ [0,1]
            new_length = max(0.001, anim.length)
            self._time = old_phase * new_length                       # map phase to new T
            log.debug(
                f"Phase-sync: '{self._blend_from_anim.name}'(t={self._blend_from_time:.3f}/"
                f"{old_length:.3f}) → '{anim.name}' starts at t={self._time:.3f}/{new_length:.3f}"
            )
        else:
            self._time = 0.0
        return True

    def stop(self):
        self._playing = False

    def pause(self):
        self._playing = not self._playing

    def seek(self, t: float):
        if self._current_anim:
            length = max(0.001, self._current_anim.length)
            self._time = t % length if self._loop else min(t, length)
            # Reset fired-events so events before the new position don't re-fire
            self._fired_events = set()
            # Cancel any in-progress blend when manually seeking
            self._blend_from_pose = None
            self._blend_t         = 0.0
            self._blend_elapsed   = 0.0
            self._blend_duration  = 0.0

    def advance(self, dt: float) -> bool:
        """Advance animation time by *dt* seconds.

        Returns ``True`` if playback is **still active** after the advance,
        ``False`` when the animation has ended (non-looping) or was not
        playing at all.
        """
        if not self._playing or not self._current_anim:
            return False
        length = max(0.001, self._current_anim.length)
        self._time += dt

        # ── Fire time-based events (once per loop cycle) ─────────────────────
        for ei, ev in enumerate(self._current_anim.events):
            if ev.time <= self._time and ei not in self._fired_events:
                self._fired_events.add(ei)
                log.debug(f"AnimEvent '{ev.name}' fired @ {ev.time:.3f}s")

        # ── Advance cross-fade blend ──────────────────────────────────────────
        # _blend_elapsed tracks real time since the blend started, independent
        # of the new clip's absolute time (_time).  This is essential when
        # phase-sync starts the new clip mid-way through (e.g. _time=1.125 after
        # a phase-sync walk→run transition): using _time/duration would
        # immediately finish the blend since _time >> duration.
        # Reference: Gregory §12.6.3 — blend fraction β = elapsed/duration.
        if self._blend_duration > 0.0:
            self._blend_elapsed += dt
            self._blend_t = min(1.0, self._blend_elapsed / self._blend_duration)
            if self._blend_t >= 1.0:
                self._blend_from_pose = None  # blend finished
                self._blend_t         = 0.0
                self._blend_elapsed   = 0.0
                self._blend_duration  = 0.0

        # ── Loop / stop ───────────────────────────────────────────────────────
        if self._time >= length:
            if self._loop:
                self._time %= length
                self._fired_events = set()  # reset so events fire again
            else:
                self._time    = length
                self._playing = False
        return self._playing

    # ── Pose evaluation ─────────────────────────────────────────────────────

    def evaluate(self, t: Optional[float] = None) -> AnimPose:
        """
        Evaluate the animation pose at time *t* (defaults to current time).

        When a cross-fade blend is active, the returned pose is smoothly
        interpolated (lerp position / slerp rotation) from the captured
        ``_blend_from_pose`` towards the new animation pose using the
        current ``_blend_t`` fraction [0 = old, 1 = new].

        When ``sync_phase=True`` was passed to :meth:`play`, the "from" pose
        is re-evaluated every frame at the same normalized phase as the new
        clip, preventing foot-slip during locomotion-cycle transitions.

        Reference: Gregory §12.6.3 — phase-synchronized cross-fade.

        Returns an :class:`AnimPose` with all animated node transforms.
        """
        if t is None:
            t = self._time
        anim = self._current_anim
        if anim is None:
            return AnimPose(time=t)

        pose = AnimPose(time=t)
        # The emitter particle pass needs the active Animation block to sample
        # emitter channels (birthrate/alpha gates such as the Star Map "on").
        # Viewport models frequently carry an empty ``animations`` list — the
        # Animation Browser resolves clips through source/supermodel chains —
        # so the resolved block rides along on the evaluated pose.
        pose._gr_animation = anim
        for anim_node in anim.nodes:
            np_ = self._eval_node(anim_node, t)
            if np_:
                pose.nodes[np_.name.lower()] = np_

        # ── Cross-fade blend ──────────────────────────────────────────────────
        if self._blend_from_pose is not None and 0.0 < self._blend_t < 1.0:
            alpha = self._blend_t  # 0 = fully old pose, 1 = fully new pose

            # Phase-synchronized "from" pose: re-evaluate the old clip every frame
            # at a time that corresponds to the same normalized phase as the new clip.
            # This prevents foot contacts from sliding during the blend transition.
            # Reference: Gregory §12.6.3 — u_old = u_new (normalized time matching).
            if (self._blend_sync_phase and
                    self._blend_from_anim is not None and
                    self._blend_from_anim.nodes):
                old_anim = self._blend_from_anim
                old_length = max(0.001, old_anim.length)
                new_length = max(0.001, anim.length)
                u_new = (t % new_length) / new_length              # [0, 1]
                old_t = u_new * old_length                          # same phase in old clip
                # Build a live "from" pose by evaluating old clip at old_t
                live_from: Dict[str, NodePose] = {}
                for anim_node in old_anim.nodes:
                    np_ = self._eval_node(anim_node, old_t)
                    if np_:
                        live_from[np_.name.lower()] = np_
                from_nodes = live_from
            else:
                from_nodes = self._blend_from_pose.nodes

            for name, new_np in list(pose.nodes.items()):
                old_np = from_nodes.get(name)
                if old_np is None:
                    continue
                # Lerp position
                op = old_np.position; npos = new_np.position
                blended_pos = (
                    op[0] + (npos[0] - op[0]) * alpha,
                    op[1] + (npos[1] - op[1]) * alpha,
                    op[2] + (npos[2] - op[2]) * alpha,
                )
                # Slerp rotation (shortest-path)
                blended_rot = tuple(
                    _slerp(list(old_np.rotation), list(new_np.rotation), alpha)
                )
                # Lerp scale
                blended_scale = old_np.scale + (new_np.scale - old_np.scale) * alpha
                # Lerp alpha (None means "use bind-pose value")
                if old_np.alpha is not None and new_np.alpha is not None:
                    blended_alpha: Optional[float] = old_np.alpha + (new_np.alpha - old_np.alpha) * alpha
                elif new_np.alpha is not None:
                    blended_alpha = new_np.alpha
                else:
                    blended_alpha = old_np.alpha
                # Lerp selfillum
                if old_np.selfillum is not None and new_np.selfillum is not None:
                    osi = old_np.selfillum; nsi = new_np.selfillum
                    blended_si: Optional[Tuple[float,float,float]] = (
                        osi[0] + (nsi[0]-osi[0])*alpha,
                        osi[1] + (nsi[1]-osi[1])*alpha,
                        osi[2] + (nsi[2]-osi[2])*alpha,
                    )
                elif new_np.selfillum is not None:
                    blended_si = new_np.selfillum
                else:
                    blended_si = old_np.selfillum
                pose.nodes[name] = NodePose(
                    name      = new_np.name,
                    position  = blended_pos,
                    rotation  = blended_rot,
                    scale     = blended_scale,
                    alpha     = blended_alpha,
                    selfillum = blended_si,
                )

        return pose

    def _eval_node(self, anim_node: ModelNode, t: float) -> Optional[NodePose]:
        """Evaluate a single animation node at time t.

        KotOR animation coordinate conventions
        ──────────────────────────────────────
        POSITION controller (type 8):
            Keyframe values are DELTA OFFSETS in parent-local space, added to the
            node's bind-pose local position.  They are NOT absolute positions.
            Formula:  animated_local_pos = bind_local_pos + keyframe_delta

            Verified against both xoreos (model_kotor.cpp, arePositionFramesRelative()
            always returns true) and KotorBlender (animnode.py,
            convert_mdl_position_to_bl_location: p1 = restloc + animscale * val).

            Evidence:
              c_bantha BTHips:  bind=(0,-1.096,1.619), k0=(0,0,0)
                → animated=(0,-1.096,1.619)  [bone stays at bind = T-pose] ✓
              c_bantha BTSpine1: bind=(0,-0.436,1.557), k0=(0.055,-0.004,-0.003)
                → animated=(0.055,-0.44,1.555)  [small walking nudge] ✓
              c_bmspecdiff RootDummy: bind=(0,0.137,0.962), k0=(-0.028,-0.009,0.962)
                → animated=(-0.028,0.128,1.924)  [creature rises to active stance] ✓
              c_kinrath RootDummy: bind=(-0.002,-0.138,0.6), k0=(-0.044,0,-0.076)
                → animated=(-0.046,-0.138,0.524)  [step forward] ✓

            NOTE: Treating these as absolute would collapse all bones toward the
            origin (the 'exploded skeleton' bug).

        ORIENTATION controller (type 20):
            Keyframe values are ABSOLUTE quaternions in parent-local space.
            They REPLACE the bind-pose local rotation entirely.
            Evidence: BTHips bind=[0,0,0,1] (identity), anim=[0.001,0.061,-0.07,0.996]
            (small rotation from T-pose) — clearly an absolute rotation, not a
            multiplication of the bind rotation by the keyframe.
        """
        # Start with base-pose values
        base = self._base_nodes.get(anim_node.name.lower())
        pos  = list(base.position) if base else [0.0, 0.0, 0.0]
        rot  = list(base.rotation) if base else [0.0, 0.0, 0.0, 1.0]
        scale = 1.0

        # Normalize base rotation to unit length only.
        # NOTE: We do NOT canonicalize to positive-w here.  xoreos stores some
        # bind-pose quaternions with negative w (e.g. r_shlder: w=-0.2266), and
        # some animation keyframes also have negative w.  Flipping the sign of
        # the BASE pose quaternion so w>0 would make it inconsistent with the
        # keyframe signs, causing _slerp to compute a longer-path interpolation
        # through the antipodal instead of the geodesic. The _slerp function
        # already handles shortest-path via "if dot < 0: negate q2; dot=-dot".
        # Verified against xoreos animation.cpp interpolateOrientation().
        if base:
            r2 = rot[0]*rot[0] + rot[1]*rot[1] + rot[2]*rot[2] + rot[3]*rot[3]
            if r2 > 1e-9 and abs(r2 - 1.0) > 1e-4:
                rs = math.sqrt(r2)
                rot = [rot[0]/rs, rot[1]/rs, rot[2]/rs, rot[3]/rs]

        # Material animation accumulators (None = no keyframe found)
        node_alpha_anim: Optional[float]                    = None
        node_si_anim:    Optional[Tuple[float,float,float]] = None

        for ctrl in anim_node.controllers:
            ctype = ctrl['type']
            # Hot path: sample the cached pre-sanitized channel. Sanitizing
            # (finite filter + quaternion sign consistency) used to run per
            # sample per frame and dominated evaluate() cost on PIE modules.
            channel_times, channel_values = _channel_samples(ctrl)
            val = _interp_channel_sanitized(channel_times, channel_values, t)
            if val is None:
                continue

            if ctype == self.CTRL_POSITION and len(val) >= 3:
                # KotOR position keyframes are DELTA OFFSETS added to bind-pose
                # local position (NOT absolute positions).  Verified against xoreos
                # (arePositionFramesRelative()=true) and KotorBlender (animnode.py).
                #
                # Super-model chain scaling (Phase 5)
                # ----------------------------------
                # ``self._current_anim_scale`` is 1.0 for animations defined
                # on this model and therefore a true no-op multiply on the
                # common path.  It is > 0 only when the current animation
                # was inherited from a supermodel — in that case we apply the
                # cumulative anim_scale product, matching KotorBlender's
                #     p1 = restloc + animscale * val
                # (scene/animnode.py) and xoreos' Model::getAnimationScale()
                # for chain-inherited animations.  Xoreos *does not* scale
                # a model's OWN animations, which is exactly why the local
                # branch of _find_anim sets _current_anim_scale = 1.0.
                if all(math.isfinite(v) for v in val[:3]):
                    absolute_local = _position_controller_is_absolute_local(
                        self.model,
                        anim_node.name,
                        pos,
                        ctrl,
                    )
                    absolute_reference = None
                    if absolute_local:
                        controller_values = ctrl.get("values", []) or []
                        if controller_values and len(controller_values[0]) >= 3:
                            absolute_reference = controller_values[0]
                    pos = list(
                        _compose_position_controller_value(
                            pos,
                            val,
                            animation_scale=self._current_anim_scale,
                            absolute_local=absolute_local,
                            absolute_reference=absolute_reference,
                        )
                    )
            elif ctype == self.CTRL_ORIENTATION and len(val) >= 4:
                # KotOR orientation keyframes are ABSOLUTE quaternions (replace bind rot).
                # Validate and normalize animated rotation.
                # NOTE: Do NOT canonicalize to positive-w. xoreos stores negative-w
                # quaternions in some animation data; the _slerp function handles
                # shortest-path via "if dot < 0: negate q2". Forcing positive-w here
                # would cause incorrect long-path interpolations between consecutive
                # keyframes that straddle the w=0 boundary.
                rv = val[:4]
                if all(math.isfinite(v) for v in rv):
                    r2 = rv[0]*rv[0] + rv[1]*rv[1] + rv[2]*rv[2] + rv[3]*rv[3]
                    if r2 > 1e-9:
                        rs = math.sqrt(r2)
                        rot = [rv[0]/rs, rv[1]/rs, rv[2]/rs, rv[3]/rs]
                    else:
                        rot = [0.0, 0.0, 0.0, 1.0]  # fallback identity
            elif ctype == self.CTRL_SCALE and len(val) >= 1:
                sv = val[0]
                if math.isfinite(sv):
                    scale = max(0.0, sv)
            elif ctype == self.CTRL_ALPHA and len(val) >= 1:
                av = val[0]
                if math.isfinite(av):
                    node_alpha_anim = max(0.0, min(1.0, av))
                else:
                    node_alpha_anim = None
            elif ctype == self.CTRL_ALPHA_XOREOS and len(val) >= 1:
                # xoreos uses CTRL_ALPHA = 128; handle alongside KotorBlender's 132
                av = val[0]
                if math.isfinite(av) and node_alpha_anim is None:
                    node_alpha_anim = max(0.0, min(1.0, av))
            elif ctype == self.CTRL_SELFILLUMCOLOR and len(val) >= 3:
                sv0, sv1, sv2 = val[0], val[1], val[2]
                if math.isfinite(sv0) and math.isfinite(sv1) and math.isfinite(sv2):
                    node_si_anim = (max(0.0,sv0), max(0.0,sv1), max(0.0,sv2))
                else:
                    node_si_anim = None

        # Canonicalize output rotation to positive-w before returning the pose.
        # This prevents 360° flips when the pose is used as the slerp start-point
        # in the viewport's _node_world_transform chain.  The canonicalization
        # happens at the OUTPUT stage (after all controllers are applied) so it
        # does NOT affect the intermediate slerp consistency between keyframes.
        # The _slerp function handles shortest-path via "if dot < 0: negate q2"
        # independently; this canonicalization ensures the base-pose start-point
        # for the NEXT slerp evaluation has a predictable positive-w form.
        # Reference: test_v37_bug_fixes.py TestViewportWCanonicalization.
        if rot[3] < 0.0:
            rot = [-rot[0], -rot[1], -rot[2], -rot[3]]

        return NodePose(
            name      = anim_node.name,
            position  = tuple(pos),
            rotation  = tuple(rot),
            scale     = scale,
            alpha     = node_alpha_anim,
            selfillum = node_si_anim,
        )

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _find_anim(self, name: str) -> Optional[Animation]:
        """Locate an animation by name, walking the super-model chain if needed.

        Lookup order (matches xoreos + KotorBlender):
          1. This model's own animations (scale = 1.0).
          2. Fuzzy alias fall-back for 'usecomp' variants (own animations only).
          3. SuperModelResolver chain walk (scale = cumulative anim_scale
             product of every step traversed).

        On success ``self._current_anim_scale`` is updated so that
        ``_eval_node`` can scale POSITION-delta keyframes appropriately.
        """
        nl = name.lower()
        # 1. Local animations — own animations never receive anim_scale (xoreos).
        for a in self.model.animations:
            if a.name.lower() == nl:
                self._current_anim_scale = 1.0
                return a
        # 2. 'usecomp' alias fall-back (local only).  This must stay ABOVE
        #    the chain walk so a supermodel with a literal 'usecomp' does
        #    not shadow a local 'use_comp' on the character model.
        if nl in ('usecomp', 'use_comp', 'use comp'):
            for a in self.model.animations:
                al = a.name.lower()
                if al in ('usecomp', 'use_comp', 'use comp'):
                    self._current_anim_scale = 1.0
                    return a
        # 3. Super-model chain resolution.
        game = getattr(self.model, 'game_version', None)
        try:
            game_tag = game.value if hasattr(game, 'value') else (
                str(game) if game else None
            )
        except Exception:
            game_tag = None
        anim, scale = SuperModelResolver.resolve_animation(
            self.model, name, game_tag,
        )
        if anim is not None:
            self._current_anim_scale = scale
            return anim
        # Not found anywhere — leave scale at 1.0 to avoid poisoning
        # subsequent lookups with stale state.
        self._current_anim_scale = 1.0
        return None

    # ── Usecomp / composite animation helpers ───────────────────────────────

    def is_usecomp_model(self) -> bool:
        """Return True if this model has a 'usecomp' animation (composite geometry model).

        KotOR composite models (e.g. N_CaloNord) use a single 'usecomp' animation
        that drives skeletal pose for all composite body-part meshes simultaneously.
        The model itself acts as the skeleton root; body parts are attached via hooks.
        """
        return any(a.name.lower() in ('usecomp', 'use_comp', 'use comp')
                   for a in self.model.animations)

    def get_usecomp_bone_names(self) -> List[str]:
        """Return sorted list of bone names animated by the 'usecomp' animation."""
        anim = self._find_anim('usecomp')
        if anim is None:
            return []
        return sorted(n.name for n in anim.nodes)

    def merge_usecomp_from(self, parent_engine: 'AnimationEngine') -> int:
        """Merge the 'usecomp' animation from a *parent_engine* into this engine's model.

        Used when a body-part (head, arm segment, etc.) needs to inherit the composite
        skeleton pose from the master composite model.  Copies only nodes whose names
        exist in this engine's model so we don't inject irrelevant bones.

        Returns the number of nodes merged.
        """
        src_anim = parent_engine._find_anim('usecomp')
        if src_anim is None:
            return 0

        local_names = {n.name.lower() for n in self.model.all_nodes()}
        filtered_nodes = [n for n in src_anim.nodes
                          if n.name.lower() in local_names]
        if not filtered_nodes:
            return 0

        import copy
        new_anim = copy.deepcopy(src_anim)
        new_anim.nodes = filtered_nodes

        # Replace if already present
        for i, a in enumerate(self.model.animations):
            if a.name.lower() in ('usecomp', 'use_comp', 'use comp'):
                self.model.animations[i] = new_anim
                log.info("merge_usecomp_from: replaced usecomp in '%s' (%d nodes)",
                         self.model.name, len(filtered_nodes))
                return len(filtered_nodes)

        self.model.animations.append(new_anim)
        log.info("merge_usecomp_from: added usecomp to '%s' (%d nodes)",
                 self.model.name, len(filtered_nodes))
        return len(filtered_nodes)

    def build_bone_remap(self, target_model: KotorModel,
                         fuzzy: bool = True) -> Dict[str, str]:
        """Build a bone-name remap from this engine's model to *target_model*.

        Convenience wrapper around the module-level :func:`build_bone_remap`.

        Returns:
            ``dict[src_bone_name, tgt_bone_name]``
        """
        return build_bone_remap(self.model, target_model, fuzzy=fuzzy)

    def retarget_usecomp(self,
                         target_model: KotorModel,
                         bone_remap: Optional[Dict[str, str]] = None,
                         *,
                         inject: bool = True) -> Optional['Animation']:
        """Retarget this model's 'usecomp' animation onto *target_model*'s skeleton.

        Finds the 'usecomp' animation in this engine's model, builds (or uses the
        supplied) *bone_remap*, retargets the animation, and optionally injects the
        result directly into *target_model*.

        Args:
            target_model: The model to receive the retargeted animation.
            bone_remap:   Pre-built remap dict; if None, :meth:`build_bone_remap`
                          is called automatically.
            inject:       If True (default), append/replace the animation in
                          *target_model.animations*.

        Returns:
            The retargeted :class:`Animation`, or None if no usecomp found.
        """
        src_anim = self._find_anim('usecomp')
        if src_anim is None:
            log.debug("retarget_usecomp: no usecomp animation in '%s'", self.model.name)
            return None

        if bone_remap is None:
            bone_remap = build_bone_remap(self.model, target_model)

        out_anim = retarget_usecomp(src_anim, target_model, bone_remap)

        if inject and out_anim.nodes:
            tgt_eng = AnimationEngine(target_model)
            # Replace if already present, else append
            for i, a in enumerate(target_model.animations):
                if a.name.lower() in ('usecomp', 'use_comp', 'use comp'):
                    target_model.animations[i] = out_anim
                    log.info("retarget_usecomp: replaced usecomp in '%s' (%d nodes)",
                             target_model.name, len(out_anim.nodes))
                    return out_anim
            target_model.animations.append(out_anim)
            log.info("retarget_usecomp: injected usecomp into '%s' (%d nodes)",
                     target_model.name, len(out_anim.nodes))

        return out_anim

    def list_animations(self) -> List[Dict[str, Any]]:
        """Return list of animation info dicts for this model's OWN animations.

        For the full set including inherited super-model animations use
        :meth:`list_all_animations`.
        """
        result = []
        for a in self.model.animations:
            total_keys = sum(
                len(c['times'])
                for n in a.nodes
                for c in n.controllers
            )
            result.append({
                'name':       a.name,
                'length':     a.length,
                'trans_time': a.transition_time,
                'node_count': len(a.nodes),
                'key_count':  total_keys,
                'event_count':len(a.events),
                'anim_root':  a.anim_root,
                'source':     self.model.name,
                'source_type': 'local',
                'source_scope': 'local',
                'overrides_inherited': False,
                'inherited':  False,
                'anim_scale': 1.0,
            })
        return result

    def list_all_animations(self) -> List[Dict[str, Any]]:
        """Return every animation reachable through the super-model chain.

        Equivalent to :meth:`list_animations` but additionally walks
        ``model.supermodel`` (recursively) via :class:`SuperModelResolver`
        and includes inherited animations.  Own-model animations always
        override same-named supermodel ones (first match wins, bottom-up).

        Each entry carries extra bookkeeping fields on top of the shape
        returned by :meth:`list_animations`:

        ``source``      name of the model that actually stores the clip.
        ``source_type`` one of ``local``, ``inherited``, or ``override``.
        ``inherited``   True when ``source`` differs from ``self.model.name``.
        ``anim_scale``  cumulative anim_scale for POSITION-delta playback;
                        always 1.0 for own animations.
        """
        game = getattr(self.model, 'game_version', None)
        try:
            game_tag = game.value if hasattr(game, 'value') else (
                str(game) if game else None
            )
        except Exception:
            game_tag = None

        entries = SuperModelResolver.list_all_animations(self.model, game_tag)
        own_name_lower = self.model.name.lower()
        result: List[Dict[str, Any]] = []
        for anim_name, source, scale in entries:
            anim, _resolved_scale = SuperModelResolver.resolve_animation(
                self.model, anim_name, game_tag,
            )
            if anim is None:
                continue
            total_keys = sum(
                len(c['times'])
                for n in anim.nodes
                for c in n.controllers
            )
            source_type = SuperModelResolver.animation_source_type(
                self.model,
                anim.name,
                source,
                game_tag,
            )
            inherited = source.lower() != own_name_lower
            result.append({
                'name':        anim.name,
                'length':      anim.length,
                'trans_time':  anim.transition_time,
                'node_count':  len(anim.nodes),
                'key_count':   total_keys,
                'event_count': len(anim.events),
                'anim_root':   anim.anim_root,
                'source':      source,
                'source_type':  source_type,
                'source_scope': 'inherited' if inherited else 'local',
                'overrides_inherited': source_type == 'override',
                'inherited':   inherited,
                'anim_scale':  scale,
            })
        return result

    def get_animation_fps_estimate(self, anim: Animation) -> float:
        """
        Estimate the original bake FPS from keyframe density.

        Strategy:
          1. Compute raw_fps = max_keys_in_any_channel / anim.length.
          2. Snap to the nearest standard KotOR FPS tier: 15, 24, 25, 30, 60.
             KotOR NWN exporters bake at these rates almost exclusively.
          3. Fall back to 30 if the animation has no keys or zero length.

        This prevents the ``~29 fps`` / ``~31 fps`` drift caused by integer
        rounding of float division and gives cleaner values to the FPS selector
        in the animations panel.
        """
        if not anim.nodes or anim.length <= 0:
            return 30.0
        max_keys = 0
        for n in anim.nodes:
            for c in n.controllers:
                if c['times']:
                    max_keys = max(max_keys, len(c['times']))
        if max_keys <= 1:
            return 30.0
        raw = max_keys / anim.length
        # Snap to nearest standard tier
        _TIERS = (15.0, 24.0, 25.0, 30.0, 60.0)
        best   = min(_TIERS, key=lambda t: abs(t - raw))
        # Only snap if within 20 % of the tier; otherwise return rounded raw value
        if abs(best - raw) / best <= 0.20:
            return best
        return float(round(raw))

    def get_recommended_playback_fps(self, anim: Animation) -> int:
        """
        Return the recommended integer UI FPS setting for this animation.
        Clamps the estimated FPS into the combobox values [15, 24, 25, 30, 60].
        """
        fps = self.get_animation_fps_estimate(anim)
        _VALID = [15, 24, 25, 30, 60]
        # Pick the closest valid tier
        return min(_VALID, key=lambda v: abs(v - fps))

    # ── Export ───────────────────────────────────────────────────────────────

    def export_animation_json(self, anim_name: str, output_path: str) -> bool:
        """Export a single animation to JSON format."""
        anim = self._find_anim(anim_name)
        if anim is None:
            log.error(f"Animation '{anim_name}' not found")
            return False

        data = {
            'format':      'kotor_animation_v1',
            'model_name':  self.model.name,
            'name':        anim.name,        # convenience alias
            'anim_name':   anim.name,
            'length':      anim.length,
            'trans_time':  anim.transition_time,
            'anim_root':   anim.anim_root,
            'fps_estimate': self.get_animation_fps_estimate(anim),
            'events': [
                {'time': ev.time, 'name': ev.name}
                for ev in anim.events
            ],
            'nodes': []
        }

        for n in anim.nodes:
            ndata = {
                'name':        n.name,
                'base_position': list(n.position),
                'base_rotation': list(n.rotation),
                'controllers': []
            }
            for ctrl in n.controllers:
                ndata['controllers'].append({
                    'type':    ctrl['type'],
                    'name':    ctrl['name'],
                    'columns': ctrl['columns'],
                    'times':   ctrl['times'],
                    'values':  [list(v) for v in ctrl['values']],
                })
            data['nodes'].append(ndata)

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        log.info(f"Exported animation '{anim_name}' → {output_path}")
        return True

    def export_animation_bvh(self, anim_name: str, output_path: str) -> bool:
        """
        Export animation to BVH (BioVision Hierarchy) format.
        BVH is widely supported by Blender, Maya, MotionBuilder, etc.
        """
        anim = self._find_anim(anim_name)
        if anim is None:
            log.error(f"Animation '{anim_name}' not found")
            return False

        fps = max(15.0, min(120.0, self.get_animation_fps_estimate(anim)))
        frame_count = max(1, int(anim.length * fps))
        frame_time  = 1.0 / fps

        # Build bone list in hierarchy order from model
        bones = []
        if self.model.root_node:
            def _collect_bones(node, depth=0):
                bones.append((node, depth))
                for c in node.children:
                    _collect_bones(c, depth + 1)
            _collect_bones(self.model.root_node)

        if not bones:
            log.error("No bones found in model")
            return False

        lines = []

        # HIERARCHY section
        lines.append("HIERARCHY")

        # Build anim_node set BEFORE hierarchy so _bvh_bone can detect animated leaves
        # BUG-03 FIX: a leaf node that has animation keys is a real JOINT and must
        # receive CHANNELS in the hierarchy section (and frame data in MOTION section).
        # Only promote it to a proper JOINT; a synthetic end-site child is appended so
        # the BVH is still syntactically valid (every non-leaf JOINT must close with a
        # child "End Site" or another JOINT).
        anim_nodes_set: set = {n.name.lower() for n in anim.nodes}

        def _bvh_bone(node, depth):
            indent = "  " * depth
            is_root = (node.parent is None)
            is_true_leaf = (not node.children)
            # BUG-03: an animated leaf is treated as a JOINT, not an End Site
            is_animated_leaf = is_true_leaf and node.name.lower() in anim_nodes_set

            if is_root:
                lines.append(f"{indent}ROOT {node.name}")
            elif is_true_leaf and not is_animated_leaf:
                # Pure End Site — no channels, no frame data
                lines.append(f"{indent}End Site")
                lines.append(f"{indent}{{")
                px, py, pz = node.position
                lines.append(f"{indent}  OFFSET {px:.6f} {py:.6f} {pz:.6f}")
                lines.append(f"{indent}}}")
                return
            else:
                lines.append(f"{indent}JOINT {node.name}")

            lines.append(f"{indent}{{")
            px, py, pz = node.position
            lines.append(f"{indent}  OFFSET {px:.6f} {py:.6f} {pz:.6f}")

            if is_root:
                lines.append(f"{indent}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
            else:
                lines.append(f"{indent}  CHANNELS 3 Zrotation Xrotation Yrotation")

            for child in node.children:
                _bvh_bone(child, depth + 1)

            # BUG-03: animated leaf — append synthetic End Site so BVH is valid
            if is_animated_leaf:
                lines.append(f"{indent}  End Site")
                lines.append(f"{indent}  {{")
                lines.append(f"{indent}    OFFSET 0.000000 0.000000 0.000000")
                lines.append(f"{indent}  }}")

            lines.append(f"{indent}}}")

        _bvh_bone(self.model.root_node, 0)

        # MOTION section
        lines.append("MOTION")
        lines.append(f"Frames: {frame_count}")
        lines.append(f"Frame Time: {frame_time:.6f}")

        # Build anim_node lookup (already defined anim_nodes_set above for hierarchy)
        anim_nodes: Dict[str, ModelNode] = {n.name.lower(): n for n in anim.nodes}

        def _quat_to_euler_zxy(q: List[float]) -> Tuple[float, float, float]:
            """Convert quaternion to ZXY Euler angles in degrees (BVH convention)."""
            x, y, z, w = q
            # Normalize
            mag = math.sqrt(x*x + y*y + z*z + w*w)
            if mag > 1e-9:
                x /= mag; y /= mag; z /= mag; w /= mag

            # ZXY order: R = Rz * Rx * Ry
            sinX = 2*(w*x + y*z)
            cosX = 1 - 2*(x*x + y*y)
            rx = math.atan2(sinX, cosX)

            sinY = 2*(w*y - z*x)
            sinY = max(-1.0, min(1.0, sinY))
            ry = math.asin(sinY)

            sinZ = 2*(w*z + x*y)
            cosZ = 1 - 2*(y*y + z*z)
            rz = math.atan2(sinZ, cosZ)

            return (
                math.degrees(rz),
                math.degrees(rx),
                math.degrees(ry),
            )

        for fi in range(frame_count):
            t = fi * frame_time
            frame_vals = []

            def _collect_frame(node, is_root=False):
                anim_n = anim_nodes.get(node.name.lower())
                base   = self._base_nodes.get(node.name.lower())

                # Get position
                pos = list(base.position) if base else [0.0, 0.0, 0.0]
                rot = list(base.rotation) if base else [0.0, 0.0, 0.0, 1.0]

                if anim_n:
                    position_controller = _find_controller(
                        anim_n.controllers,
                        AnimationEngine.CTRL_POSITION,
                        "position",
                    )
                    ctrl_pos_t, ctrl_pos_v = _get_ctrl(anim_n, AnimationEngine.CTRL_POSITION)
                    ctrl_rot_t, ctrl_rot_v = _get_ctrl(anim_n, AnimationEngine.CTRL_ORIENTATION)
                    ev_pos = _interp_channel(ctrl_pos_t, ctrl_pos_v, t)
                    ev_rot = _interp_channel(ctrl_rot_t, ctrl_rot_v, t)
                    if ev_pos and len(ev_pos) >= 3 and all(math.isfinite(v) for v in ev_pos[:3]):
                        pos = list(
                            _compose_position_controller_value(
                                pos,
                                ev_pos,
                                absolute_local=_position_controller_is_absolute_local(
                                    self.model,
                                    node.name,
                                    pos,
                                    position_controller,
                                ),
                            )
                        )
                    if ev_rot and len(ev_rot) >= 4:
                        rot = ev_rot[:4]

                ez, ex, ey = _quat_to_euler_zxy(rot)

                if is_root:
                    frame_vals.extend([pos[0], pos[1], pos[2], ez, ex, ey])
                else:
                    frame_vals.extend([ez, ex, ey])

                for child in node.children:
                    # BUG-03 FIX: only skip true End Site nodes (no children AND not animated).
                    # Animated leaf joints were incorrectly skipped, losing their keyframe data.
                    if not child.children and child.name.lower() not in anim_nodes_set:
                        continue   # skip pure end sites (no channels in hierarchy)
                    _collect_frame(child)

            _collect_frame(self.model.root_node, is_root=True)
            lines.append(" ".join(f"{v:.6f}" for v in frame_vals))

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='ascii') as f:
            f.write('\n'.join(lines) + '\n')
        log.info(f"Exported BVH '{anim_name}' ({frame_count} frames @ {fps:.0f}fps) → {output_path}")
        return True

    def export_all_animations(self, output_dir: str,
                               fmt: str = 'json') -> List[str]:
        """Export all animations to the given directory. Returns list of output paths."""
        os.makedirs(output_dir, exist_ok=True)
        exported = []
        for anim in self.model.animations:
            safe_name = anim.name.replace('/', '_').replace('\\', '_')
            if fmt == 'bvh':
                path = os.path.join(output_dir, f"{safe_name}.bvh")
                ok = self.export_animation_bvh(anim.name, path)
            else:
                path = os.path.join(output_dir, f"{safe_name}.json")
                ok = self.export_animation_json(anim.name, path)
            if ok:
                exported.append(path)
        return exported

    # ── Import ───────────────────────────────────────────────────────────────

    def import_animation_json(self, json_path: str) -> Optional[Animation]:
        """Import an animation from a JSON file (previously exported)."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if data.get('format') not in ('kotor_animation_v1', None):
                log.warning(f"Unknown animation format: {data.get('format')}")

            anim = Animation(
                name            = data.get('anim_name', os.path.basename(json_path)),
                length          = float(data.get('length', 1.0)),
                transition_time = float(data.get('trans_time', 0.25)),
                anim_root       = data.get('anim_root', ''),
            )

            for ev_data in data.get('events', []):
                anim.events.append(AnimEvent(
                    time = float(ev_data.get('time', 0.0)),
                    name = ev_data.get('name', ''),
                ))

            for n_data in data.get('nodes', []):
                from ..geometry.model_data import ModelNode, NodeFlags
                node = ModelNode(
                    name     = n_data.get('name', 'node'),
                    position = tuple(n_data.get('base_position', [0,0,0])),
                    rotation = tuple(n_data.get('base_rotation', [0,0,0,1])),
                )
                for c_data in n_data.get('controllers', []):
                    node.controllers.append({
                        'type':    int(c_data.get('type', 0)),
                        'name':    c_data.get('name', ''),
                        'columns': int(c_data.get('columns', 1)),
                        'times':   [float(t) for t in c_data.get('times', [])],
                        'values':  [list(v) for v in c_data.get('values', [])],
                    })
                anim.nodes.append(node)

            log.info(f"Imported animation '{anim.name}' from {json_path}")
            return anim

        except Exception as e:
            log.error(f"Failed to import animation from {json_path}: {e}")
            return None

    def get_pose(self, t: Optional[float] = None) -> AnimPose:
        """Alias for evaluate() – returns an AnimPose at time t (or current time).
        Provided for convenience so callers can use either name."""
        return self.evaluate(t)

    def get_fired_events(self) -> List[str]:
        """Return names of events that have fired so far in the current loop cycle."""
        if not self._current_anim:
            return []
        return [
            self._current_anim.events[i].name
            for i in sorted(self._fired_events)
            if i < len(self._current_anim.events)
        ]

    def is_blending(self) -> bool:
        """Return True if a cross-fade blend is currently active."""
        return self._blend_from_pose is not None and self._blend_t < 1.0

    def blend_fraction(self) -> float:
        """Return the current blend fraction [0.0 = start of blend, 1.0 = complete]."""
        return self._blend_t

    def add_animation(self, anim: Animation):
        """Add an animation to the model's animation list."""
        # Replace if same name exists
        for i, a in enumerate(self.model.animations):
            if a.name.lower() == anim.name.lower():
                self.model.animations[i] = anim
                log.info(f"Replaced animation '{anim.name}'")
                return
        self.model.animations.append(anim)
        log.info(f"Added animation '{anim.name}' to model '{self.model.name}'")

    def remove_animation(self, anim_name: str) -> bool:
        """Remove an animation by name."""
        for i, a in enumerate(self.model.animations):
            if a.name.lower() == anim_name.lower():
                del self.model.animations[i]
                if self._current_anim and self._current_anim.name.lower() == anim_name.lower():
                    self._current_anim = None
                    self._playing = False
                return True
        return False


# ─────────────────────────────────────────────────────────────────
#  Module-level usecomp helper
# ─────────────────────────────────────────────────────────────────

def merge_usecomp_animations(child_model: KotorModel,
                              parent_model: KotorModel) -> int:
    """Merge the 'usecomp' animation from *parent_model* into *child_model*.

    Convenience wrapper around :meth:`AnimationEngine.merge_usecomp_from`.
    Useful when assembling creature head/body parts that inherit composite
    bone poses from a master skeleton model (e.g. N_CaloNord body parts).

    Returns the number of animation nodes merged (0 = nothing to merge).
    """
    parent_eng = AnimationEngine(parent_model)
    child_eng  = AnimationEngine(child_model)
    return child_eng.merge_usecomp_from(parent_eng)


# ─────────────────────────────────────────────────────────────────
#  Bone-remap table helpers (usecomp / composite model retargeting)
# ─────────────────────────────────────────────────────────────────

def build_bone_remap(
    source_model: KotorModel,
    target_model: KotorModel,
    *,
    fuzzy: bool = True,
) -> Dict[str, str]:
    """Build a bone-name remap dictionary from *source_model* to *target_model*.

    The remap maps ``source_bone_name → target_bone_name`` for every bone that
    can be matched between the two skeletons.

    Matching strategy (in priority order):
      1. **Exact** match (case-insensitive).
      2. **Prefix strip** — KotOR composite models often prefix part bones with
         the model name (e.g. ``NordHead_jaw`` → ``jaw``).  We try stripping the
         source model name prefix.
      3. **Suffix normalisation** — compare the last *N* characters when full
         name differs only by a numeric suffix (e.g. ``spine01`` ↔ ``spine1``).
      4. **Common KotOR bone aliases** (CaloNord ↔ generic humanoid skeleton).

    Args:
        source_model: The model whose bone names are the keys.
        target_model: The model whose bone names are the values.
        fuzzy: If False, only exact (case-insensitive) matches are used.

    Returns:
        ``dict[source_bone_name, target_bone_name]`` – may be empty.
    """
    _KOTOR_ALIASES: Dict[str, str] = {
        # NWN / KotOR common composite skeleton aliases
        'rhand':     'rhand',
        'lhand':     'lhand',
        'rforearm':  'rforearm',
        'lforearm':  'lforearm',
        'ruparm':    'ruparm',
        'luparm':    'luparm',
        'rshldr':    'rshldr',
        'lshldr':    'lshldr',
        'neck':      'neck',
        'head':      'head',
        'chest':     'chest',
        'spine':     'spine',
        'rthigh':    'rthigh',
        'lthigh':    'lthigh',
        'rshin':     'rshin',
        'lshin':     'lshin',
        'rfoot':     'rfoot',
        'lfoot':     'lfoot',
        'pelvis':    'pelvis',
        # NordHead / CaloNord specific aliases
        'nordhead_jaw':        'jaw',
        'nordhead_neck':       'neck',
        'nordhead_lbrow':      'lbrow',
        'nordhead_rbrow':      'rbrow',
        'nordhead_lorbital':   'lorbital',
        'nordhead_rorbital':   'rorbital',
    }

    src_names: Dict[str, str] = {}
    for node in source_model.all_nodes():
        if node.name:
            src_names[node.name.lower()] = node.name

    tgt_names: Dict[str, str] = {}
    for node in target_model.all_nodes():
        if node.name:
            tgt_names[node.name.lower()] = node.name

    remap: Dict[str, str] = {}
    src_prefix = (source_model.name or '').lower().rstrip('_ ')

    for src_lo, src_orig in src_names.items():
        # 1. Exact match
        if src_lo in tgt_names:
            remap[src_orig] = tgt_names[src_lo]
            continue

        if not fuzzy:
            continue

        # 2. Strip source-model prefix
        stripped = src_lo
        if src_prefix and src_lo.startswith(src_prefix):
            stripped = src_lo[len(src_prefix):].lstrip('_')
        if stripped and stripped in tgt_names:
            remap[src_orig] = tgt_names[stripped]
            continue

        # 3. Known alias table
        alias = _KOTOR_ALIASES.get(src_lo) or _KOTOR_ALIASES.get(stripped)
        if alias and alias.lower() in tgt_names:
            remap[src_orig] = tgt_names[alias.lower()]
            continue

        # 4. Suffix-normalised match: drop trailing digits and compare
        src_base = src_lo.rstrip('0123456789')
        if src_base:
            candidates = [tl for tl in tgt_names if tl.rstrip('0123456789') == src_base]
            if len(candidates) == 1:
                remap[src_orig] = tgt_names[candidates[0]]

    return remap


def retarget_usecomp(
    source_anim: 'Animation',
    target_model: KotorModel,
    bone_remap: Optional[Dict[str, str]] = None,
    *,
    copy: bool = True,
) -> 'Animation':
    """Retarget a *usecomp* (or any) animation onto *target_model*'s skeleton.

    Filters and renames animation nodes to match bones present in *target_model*,
    optionally using a pre-built *bone_remap* dictionary.  If *bone_remap* is
    None, only exact-name matching is performed.

    Args:
        source_anim:  The animation to retarget (e.g. the 'usecomp' animation
                      from a master composite model).
        target_model: The model whose skeleton will receive the retargeted anim.
        bone_remap:   Optional ``{src_name → tgt_name}`` dict (from
                      :func:`build_bone_remap`).  When provided, nodes are
                      renamed as well as filtered.
        copy:         If True (default), deep-copy the animation before modifying;
                      if False, modify in place (faster, but destructive).

    Returns:
        The retargeted :class:`Animation` (a deep copy when *copy* is True).
    """
    import copy as _copy

    out_anim = _copy.deepcopy(source_anim) if copy else source_anim

    tgt_names_lo: set = {n.name.lower() for n in target_model.all_nodes() if n.name}

    kept_nodes = []
    for node in out_anim.nodes:
        # Apply remap rename first
        new_name = (bone_remap or {}).get(node.name, node.name)
        node.name = new_name
        # Keep only nodes whose (possibly renamed) bone exists in target
        if new_name.lower() in tgt_names_lo:
            kept_nodes.append(node)

    out_anim.nodes = kept_nodes
    log.debug(
        "retarget_usecomp: '%s' → '%s': kept %d/%d nodes",
        source_anim.name,
        target_model.name,
        len(kept_nodes),
        len(source_anim.nodes),
    )
    return out_anim


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────

def _get_ctrl(node: ModelNode, ctrl_type: int) -> Tuple[List, List]:
    """Get (times, values) for a controller type. Returns empty lists if not found."""
    for c in node.controllers:
        if c['type'] == ctrl_type:
            return c['times'], c['values']
    return [], []


# ─────────────────────────────────────────────────────────────────
#  Dangly Mesh Verlet Cloth Simulator
# ─────────────────────────────────────────────────────────────────

class DanglySimulator:
    """
    Self-contained Verlet integration cloth simulator for KotOR dangly mesh nodes.

    KotOR dangly nodes model cloth/chain geometry with per-vertex physics constraints:
      - dangly_displacement: maximum allowed displacement from rest position (in units)
      - dangly_tightness:    spring stiffness [0,1] — 1 = rigid, 0 = floppy
      - dangly_period:       oscillation period (seconds) — lower = faster oscillation
      - dangly_constraints:  per-vertex constraint value [0,1]
                             1.0 = fully pinned (fixed vertex, cannot move)
                             0.0 = fully free (obeys physics)

    Implementation uses position-based Verlet integration with spring-mass system.
    This is the method described in:
      - Millington, *Game Physics Engine Development* §13 (spring-mass cloth)
      - Lengyel, *Mathematics for 3D Game Programming* §15.2
      - Game Engine Architecture §12.7 (cloth/dangly simulation)

    Usage:
        sim = DanglySimulator(node)
        sim.reset()           # initialize to bind pose
        # each frame:
        positions = sim.step(dt, wind_dir=(0.1, 0.0, 0.0), gravity_scale=0.5)
        # positions is List[Tuple[float,float,float]] — world-space vertex positions
    """

    # Pin threshold: vertices with constraint >= this value are treated as fully pinned
    PIN_THRESHOLD: float = 0.95

    def __init__(self, node: ModelNode):
        self.node = node
        nv = len(node.vertices)

        # Current and previous positions (Verlet integration state)
        self._pos:      List[List[float]] = [[v[0], v[1], v[2]] for v in node.vertices]
        self._prev_pos: List[List[float]] = [[v[0], v[1], v[2]] for v in node.vertices]

        # Constraint values clamped to [0, 1]
        constraints = node.dangly_constraints
        self._constraints: List[float] = [
            max(0.0, min(1.0, constraints[i] if i < len(constraints) else 0.0))
            for i in range(nv)
        ]

        # Physics parameters from node
        self._displacement = max(0.001, node.dangly_displacement)
        self._tightness    = max(0.0, min(1.0, node.dangly_tightness))
        self._period       = max(0.01, node.dangly_period)

        # Build spring edges from face adjacency
        self._edges: List[Tuple[int, int, float]] = self._build_edges()

    def _build_edges(self) -> List[Tuple[int, int, float]]:
        """Build spring edges from mesh faces with rest lengths."""
        verts = self.node.vertices
        edges: set = set()
        for face in self.node.faces:
            i0, i1, i2 = face
            for a, b in ((i0, i1), (i1, i2), (i0, i2)):
                if a != b:
                    edges.add((min(a, b), max(a, b)))
        result = []
        for a, b in edges:
            if a < len(verts) and b < len(verts):
                va, vb = verts[a], verts[b]
                dx = vb[0] - va[0]; dy = vb[1] - va[1]; dz = vb[2] - va[2]
                rest = math.sqrt(dx*dx + dy*dy + dz*dz)
                if rest > 1e-6:
                    result.append((a, b, rest))
        return result

    def reset(self) -> None:
        """Reset simulation to bind pose (stops all motion)."""
        verts = self.node.vertices
        self._pos      = [[v[0], v[1], v[2]] for v in verts]
        self._prev_pos = [[v[0], v[1], v[2]] for v in verts]

    def step(self,
             dt: float,
             wind_dir: Tuple[float, float, float] = (0.0, 0.0, 0.0),
             gravity_scale: float = 1.0,
             wind_strength: float = 0.3) -> List[Tuple[float, float, float]]:
        """
        Advance simulation by dt seconds and return updated vertex positions.

        Parameters
        ----------
        dt              : time step in seconds (typically 1/60 or 1/30)
        wind_dir        : normalized wind direction vector
        gravity_scale   : gravity multiplier (default 1.0 = standard gravity)
        wind_strength   : wind force magnitude

        Returns
        -------
        List of (x, y, z) tuples — one per vertex, in model/world space.

        NOTE: For pinned vertices (constraint >= PIN_THRESHOLD), the original
        bind-pose position is always returned unchanged.
        """
        if dt <= 0.0:
            return [(p[0], p[1], p[2]) for p in self._pos]

        # Clamp dt to avoid instability on lag spikes
        dt = min(dt, 0.05)

        GRAVITY = (0.0, 0.0, -9.8 * gravity_scale)
        nv = len(self._pos)
        bind_verts = self.node.vertices

        # --- Verlet integration step ---
        # Stiffness coefficient from tightness: higher tightness → stronger return force
        # period controls oscillation speed: shorter period → faster spring
        stiffness = self._tightness * (4.0 * math.pi * math.pi / (self._period * self._period))

        new_pos = [[0.0, 0.0, 0.0] for _ in range(nv)]
        for i in range(nv):
            c = self._constraints[i]
            if c >= self.PIN_THRESHOLD:
                # Pinned vertex: always at bind pose
                bv = bind_verts[i]
                new_pos[i] = [bv[0], bv[1], bv[2]]
                continue

            cx, cy, cz = self._pos[i]
            px, py, pz = self._prev_pos[i]
            bx, by, bz = bind_verts[i]

            # Verlet: new_pos = 2*pos - prev_pos + acceleration * dt^2
            # Acceleration = gravity + wind + spring-return-to-bind
            spring_ax = stiffness * (bx - cx) * (1.0 - c)
            spring_ay = stiffness * (by - cy) * (1.0 - c)
            spring_az = stiffness * (bz - cz) * (1.0 - c)

            wdx, wdy, wdz = wind_dir
            wind_ax = wdx * wind_strength * (1.0 - c)
            wind_ay = wdy * wind_strength * (1.0 - c)
            wind_az = wdz * wind_strength * (1.0 - c)

            ax = GRAVITY[0] + spring_ax + wind_ax
            ay = GRAVITY[1] + spring_ay + wind_ay
            az = GRAVITY[2] + spring_az + wind_az

            dt2 = dt * dt
            nx = 2*cx - px + ax * dt2
            ny = 2*cy - py + ay * dt2
            nz = 2*cz - pz + az * dt2

            # Clamp displacement to dangly_displacement limit
            ddx = nx - bx; ddy = ny - by; ddz = nz - bz
            dist = math.sqrt(ddx*ddx + ddy*ddy + ddz*ddz)
            if dist > self._displacement:
                scale = self._displacement / dist
                nx = bx + ddx * scale
                ny = by + ddy * scale
                nz = bz + ddz * scale

            new_pos[i] = [nx, ny, nz]

        # --- Spring constraint relaxation (1 iteration) ---
        for a, b, rest in self._edges:
            ca = self._constraints[a]
            cb = self._constraints[b]
            if ca >= self.PIN_THRESHOLD and cb >= self.PIN_THRESHOLD:
                continue
            pax, pay, paz = new_pos[a]
            pbx, pby, pbz = new_pos[b]
            dx = pbx - pax; dy = pby - pay; dz = pbz - paz
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist < 1e-6:
                continue
            diff = (dist - rest) / (dist * 2.0)
            # Distribute correction weighted by which end is more free
            wa = (1.0 - ca) if ca < self.PIN_THRESHOLD else 0.0
            wb = (1.0 - cb) if cb < self.PIN_THRESHOLD else 0.0
            total = wa + wb
            if total < 1e-6:
                continue
            corr_a = wa / total
            corr_b = wb / total
            if ca < self.PIN_THRESHOLD:
                new_pos[a][0] += dx * diff * corr_a
                new_pos[a][1] += dy * diff * corr_a
                new_pos[a][2] += dz * diff * corr_a
            if cb < self.PIN_THRESHOLD:
                new_pos[b][0] -= dx * diff * corr_b
                new_pos[b][1] -= dy * diff * corr_b
                new_pos[b][2] -= dz * diff * corr_b

        # Commit new state
        self._prev_pos = self._pos
        self._pos = new_pos

        # Pinned vertices always snap back to bind pose
        for i in range(nv):
            if self._constraints[i] >= self.PIN_THRESHOLD:
                bv = bind_verts[i]
                self._pos[i] = [bv[0], bv[1], bv[2]]
                self._prev_pos[i] = [bv[0], bv[1], bv[2]]

        # Post-constraint displacement clamp: re-enforce maximum displacement after
        # spring relaxation, since relaxation can push vertices slightly beyond the limit.
        # Applied only to free (non-pinned) vertices.
        for i in range(nv):
            if self._constraints[i] >= self.PIN_THRESHOLD:
                continue
            bv = bind_verts[i]
            px_, py_, pz_ = self._pos[i]
            ddx_ = px_ - bv[0]; ddy_ = py_ - bv[1]; ddz_ = pz_ - bv[2]
            dist_ = math.sqrt(ddx_*ddx_ + ddy_*ddy_ + ddz_*ddz_)
            if dist_ > self._displacement:
                scale_ = self._displacement / dist_
                self._pos[i][0] = bv[0] + ddx_ * scale_
                self._pos[i][1] = bv[1] + ddy_ * scale_
                self._pos[i][2] = bv[2] + ddz_ * scale_

        return [(p[0], p[1], p[2]) for p in self._pos]

    @property
    def num_free_vertices(self) -> int:
        """Return count of physics-driven (non-pinned) vertices."""
        return sum(1 for c in self._constraints if c < self.PIN_THRESHOLD)

    @property
    def num_pinned_vertices(self) -> int:
        """Return count of pinned (fixed) vertices."""
        return sum(1 for c in self._constraints if c >= self.PIN_THRESHOLD)


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 8 — Animation State Machine
#  References: Gregory §12.12; Dunsky §7–8
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnimTransition:
    """A directed edge in the animation state graph.

    Attributes
    ----------
    target_state : str
        Name of the destination ``AnimState``.
    condition : callable | None
        If provided, a zero-arg callable returning ``bool``.  The transition
        fires only when the condition returns ``True``.  If ``None`` the
        transition is unconditional (triggers when explicitly requested via
        :meth:`AnimStateMachine.request_transition`).
    blend : bool
        Whether to cross-fade into the target state (default ``True``).
    sync_phase : bool
        Whether to use phase-synchronized cross-fade (Gregory §12.6.3).
        Useful for walk→run locomotion transitions.
    priority : int
        Higher-priority transitions are evaluated first when multiple
        transitions share the same source state.
    """
    target_state : str
    condition     : Optional[Any] = None     # callable() → bool | None
    blend         : bool          = True
    sync_phase    : bool          = False
    priority      : int           = 0


@dataclass
class AnimState:
    """A single node in the animation state graph.

    Attributes
    ----------
    name : str
        Unique state name (must match a key in ``AnimStateMachine._states``).
    anim_name : str
        Animation clip to play (passed to :meth:`AnimationEngine.play`).
    loop : bool
        Whether the clip should loop.
    speed : float
        Time-scale multiplier (1.0 = normal speed).
    on_enter : callable | None
        Optional callback ``(state_name: str) → None`` fired on entry.
    on_exit : callable | None
        Optional callback ``(state_name: str) → None`` fired on exit.
    transitions : list[AnimTransition]
        Outgoing transitions evaluated in descending priority order.
    """
    name        : str
    anim_name   : str
    loop        : bool                        = True
    speed       : float                       = 1.0
    on_enter    : Optional[Any]               = None   # callable(state_name) | None
    on_exit     : Optional[Any]               = None   # callable(state_name) | None
    transitions : List[AnimTransition]        = field(default_factory=list)


class AnimStateMachine:
    """Hierarchical animation state machine for KotOR character controllers.

    Implements the architecture described in Gregory §12.12 and Dunsky §7–8:
    a directed graph of ``AnimState`` nodes connected by ``AnimTransition``
    edges.  Each state plays one animation clip via an ``AnimationEngine``.

    Usage
    -----
    ::

        engine = AnimationEngine(model)
        sm     = AnimStateMachine(engine)

        sm.add_state(AnimState('idle',  'cpause1', loop=True))
        sm.add_state(AnimState('walk',  'cwalk',   loop=True))
        sm.add_state(AnimState('run',   'crun',    loop=True))
        sm.add_state(AnimState('attack', 'g1_a1',  loop=False))

        sm.add_transition('idle', AnimTransition('walk', priority=10))
        sm.add_transition('walk', AnimTransition('run',  sync_phase=True, priority=10))
        sm.add_transition('walk', AnimTransition('idle', priority=5))
        sm.add_transition('any',  AnimTransition('attack', priority=100))  # global

        sm.set_initial('idle')
        sm.start()

        # Game loop:
        sm.advance(dt)
        pose = engine.evaluate()

    Features
    --------
    * Any-state global transitions (registered under key ``'any'``).
    * Phase-synchronized cross-fade for locomotion blends (§12.6.3).
    * Per-state playback speed scaling (time-warp).
    * ``on_enter`` / ``on_exit`` callbacks for game-logic hooks.
    * ``request_transition(target)`` for unconditional imperative jumps.
    * ``current_state_name`` / ``previous_state_name`` read-only properties.
    * ``history()`` — ordered list of states visited since ``start()``.
    """

    def __init__(self, engine: AnimationEngine):
        self._engine   : AnimationEngine = engine
        self._states   : Dict[str, AnimState] = {}
        self._initial  : Optional[str] = None
        self._current  : Optional[AnimState] = None
        self._previous : Optional[str] = None
        self._history  : List[str] = []
        self._running  : bool = False
        # Pending transition requested via request_transition()
        self._pending_transition : Optional[str] = None

    # ── State registration ────────────────────────────────────────────────────

    def add_state(self, state: AnimState) -> 'AnimStateMachine':
        """Register an ``AnimState`` and return self for chaining."""
        if state.name in self._states:
            log.warning(f"AnimStateMachine: overwriting state '{state.name}'")
        self._states[state.name] = state
        return self

    def remove_state(self, name: str) -> bool:
        """Remove a state by name.  Returns ``True`` if it existed."""
        if name in self._states:
            del self._states[name]
            return True
        return False

    def add_transition(self, from_state: str, transition: AnimTransition) -> 'AnimStateMachine':
        """Add an outgoing transition from *from_state* (use ``'any'`` for global).

        Transitions are stored inside the source ``AnimState``.  The special
        key ``'any'`` is stored in a virtual ``AnimState`` named ``'any'``
        and evaluated before per-state transitions every tick.
        """
        if from_state == 'any':
            if 'any' not in self._states:
                self._states['any'] = AnimState(name='any', anim_name='')
            self._states['any'].transitions.append(transition)
        elif from_state in self._states:
            self._states[from_state].transitions.append(transition)
        else:
            log.warning(f"AnimStateMachine.add_transition: unknown source state '{from_state}'")
        return self

    def set_initial(self, name: str) -> 'AnimStateMachine':
        """Set the initial state (must be added first)."""
        if name not in self._states:
            log.warning(f"AnimStateMachine.set_initial: unknown state '{name}'")
        self._initial = name
        return self

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Enter the initial state and begin playback.

        Returns
        -------
        bool
            ``True`` if started successfully.
        """
        if not self._initial or self._initial not in self._states:
            log.error("AnimStateMachine.start: no valid initial state")
            return False
        self._running = True
        self._history.clear()
        self._enter_state(self._initial, blend=False)
        return True

    def stop(self):
        """Stop the state machine (does not stop the underlying engine)."""
        self._running = False

    def reset(self):
        """Stop and reset; call :meth:`start` to restart."""
        self.stop()
        self._current  = None
        self._previous = None
        self._history.clear()
        self._pending_transition = None

    # ── Imperative transition ──────────────────────────────────────────────────

    def request_transition(self, target_state: str,
                            blend: bool = True,
                            sync_phase: bool = False) -> bool:
        """Request an immediate transition to *target_state*.

        The transition is deferred to the next :meth:`advance` tick so that
        the current state's ``on_exit`` callback fires cleanly.

        Parameters
        ----------
        target_state : str
            Name of the destination state.
        blend : bool
            Cross-fade on entry.
        sync_phase : bool
            Phase-synced cross-fade (Gregory §12.6.3).

        Returns
        -------
        bool
            ``True`` if the target state exists.
        """
        if target_state not in self._states:
            log.warning(f"AnimStateMachine.request_transition: unknown state '{target_state}'")
            return False
        self._pending_transition = target_state
        # Store blend/sync options on the pending transition
        self._pending_blend      = blend
        self._pending_sync_phase = sync_phase
        return True

    # ── Advance ───────────────────────────────────────────────────────────────

    def advance(self, dt: float) -> Optional[str]:
        """Advance the state machine by *dt* seconds.

        Algorithm (per tick):
        1.  Process any pending ``request_transition``.
        2.  Evaluate global ('any') transitions in priority order.
        3.  Evaluate per-state transitions in priority order.
        4.  Advance the engine by ``dt * current_state.speed``.
        5.  If the current clip ended (non-loop) automatically evaluate
            exit transitions.

        Parameters
        ----------
        dt : float
            Wall-clock delta time in seconds.

        Returns
        -------
        str | None
            Name of a state we just entered this tick, or ``None``.
        """
        if not self._running or self._current is None:
            return None

        entered : Optional[str] = None

        # ── 1. Pending imperative transition ─────────────────────────────────
        if self._pending_transition is not None:
            target = self._pending_transition
            blend  = getattr(self, '_pending_blend', True)
            sync   = getattr(self, '_pending_sync_phase', False)
            self._pending_transition = None
            self._enter_state(target, blend=blend, sync_phase=sync)
            entered = target

        if self._current is None:
            return entered

        # ── 2. Global any-state transitions ──────────────────────────────────
        any_state = self._states.get('any')
        if any_state and entered is None:
            fired = self._eval_transitions(any_state)
            if fired:
                entered = fired

        # ── 3. Per-state transitions ─────────────────────────────────────────
        if entered is None:
            fired = self._eval_transitions(self._current)
            if fired:
                entered = fired

        # ── 4. Advance engine ─────────────────────────────────────────────────
        speed = self._current.speed if self._current else 1.0
        still_playing = self._engine.advance(dt * speed)

        # ── 5. Auto-exit for non-looping clips ─────────────────────────────
        if not still_playing and self._current and not self._current.loop:
            fired = self._eval_transitions(self._current)
            if not fired:
                # No transition defined — stay on the last frame
                pass
            else:
                entered = fired

        return entered

    # ── Read-only accessors ───────────────────────────────────────────────────

    @property
    def current_state_name(self) -> Optional[str]:
        """Name of the currently active state, or ``None``."""
        return self._current.name if self._current else None

    @property
    def previous_state_name(self) -> Optional[str]:
        """Name of the previously active state, or ``None``."""
        return self._previous

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def state_names(self) -> List[str]:
        """Sorted list of all registered state names (excluding 'any')."""
        return sorted(n for n in self._states if n != 'any')

    def history(self) -> List[str]:
        """Return a copy of the state-visit history (oldest → newest)."""
        return list(self._history)

    def get_state(self, name: str) -> Optional[AnimState]:
        """Return the ``AnimState`` for *name*, or ``None``."""
        return self._states.get(name)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _enter_state(self, name: str, blend: bool = True,
                     sync_phase: bool = False):
        """Internal: fire on_exit, switch state, fire on_enter, start engine."""
        target = self._states.get(name)
        if target is None:
            log.warning(f"AnimStateMachine: cannot enter unknown state '{name}'")
            return

        # on_exit for current state
        if self._current is not None:
            try:
                if callable(self._current.on_exit):
                    self._current.on_exit(self._current.name)
            except Exception as e:
                log.warning(f"AnimStateMachine on_exit error: {e}")
            self._previous = self._current.name

        self._current = target
        self._history.append(name)

        # on_enter
        try:
            if callable(target.on_enter):
                target.on_enter(target.name)
        except Exception as e:
            log.warning(f"AnimStateMachine on_enter error: {e}")

        # Start animation engine
        if target.anim_name:
            self._engine.play(
                target.anim_name,
                loop=target.loop,
                blend=blend,
                sync_phase=sync_phase,
            )
        log.debug(f"AnimStateMachine → '{name}' (anim='{target.anim_name}')")

    def _eval_transitions(self, state: AnimState) -> Optional[str]:
        """Evaluate outgoing transitions (highest priority first).

        Returns the name of the entered state, or ``None``.
        """
        transitions = sorted(state.transitions, key=lambda t: -t.priority)
        for tr in transitions:
            if tr.target_state == (self._current.name if self._current else ''):
                # Don't re-enter the same state via condition (use explicit request)
                continue
            if tr.condition is None:
                continue   # unconditional: only fires via request_transition
            try:
                if callable(tr.condition) and tr.condition():
                    self._enter_state(tr.target_state,
                                      blend=tr.blend,
                                      sync_phase=tr.sync_phase)
                    return tr.target_state
            except Exception as e:
                log.warning(f"AnimStateMachine transition condition error: {e}")
        return None
