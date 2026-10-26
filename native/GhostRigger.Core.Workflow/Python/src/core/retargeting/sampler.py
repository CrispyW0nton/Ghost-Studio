"""Fixed-rate sampling for GhostRigger retargeting.

This module turns KotOR's variable-key animation controller data into a dense
per-frame TRS clip.  It deliberately delegates pose evaluation to
``AnimationEngine`` and palette evaluation to ``MatrixPaletteUploader`` so the
retargeter does not grow a second, subtly different animation/skinning math
path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.animation.animation_engine import AnimationEngine, SuperModelResolver
from src.core.animation.gpu_skinning import MatrixPaletteUploader
from src.core.game.kotor_loader import load_model_from_file
from src.core.geometry.model_data import Animation, GameVersion, KotorModel, ModelNode


def _default_corpus_root() -> Path:
    """Locate the Day 4 stock fixture corpus at the repository root.

    A fixed ``parents[3]`` resolves to native/GhostRigger.Core.Workflow/Python
    after the native split, where the fixtures do not exist.
    """

    from .skeleton_renamer import _resolve_repo_root  # noqa: PLC0415

    return _resolve_repo_root() / "tests" / "fixtures" / "kotor_stock"


DEFAULT_CORPUS_ROOT = _default_corpus_root()


@dataclass
class SampledClip:
    """Dense, fixed-rate local-space TRS animation data.

    Rotations are stored in WXYZ order for retargeting code and MCP payloads.
    GhostRigger's in-memory model data uses XYZW, so conversion happens at the
    sampler boundary.
    """

    clip_name: str
    fps: float
    frame_count: int
    bone_names: List[str]
    positions: np.ndarray
    rotations: np.ndarray
    scales: np.ndarray
    source_model: str = ""
    source_chain: List[str] = field(default_factory=list)
    resolved_clip_source: str = ""
    duration_s: float = 0.0
    bake_math_audit_id: str = "G5"
    palette_frames: int = 0


@dataclass(frozen=True)
class BoneControllers:
    """Controller ownership for one node in an effective clip lookup."""

    source_model: str
    clip_name: str
    node_name: str
    controllers: tuple
    anim_scale: float = 1.0


class StockCorpusResourceManager:
    """Minimal ``load_model`` adapter backed by ``tests/fixtures/kotor_stock``."""

    def __init__(self, corpus_root: str | Path | None = None):
        self.corpus_root = Path(corpus_root or DEFAULT_CORPUS_ROOT)

    def load_model(self, resref: str, game: str = "K1") -> Optional[KotorModel]:
        try:
            return load_fixture_model(resref, game=game, corpus_root=self.corpus_root)
        except FileNotFoundError:
            return None


def _game_dir(game: str) -> str:
    return "k2" if str(game or "").lower() in {"2", "k2", "tsl"} else "k1"


def _game_version(game: str) -> GameVersion:
    return GameVersion.K2 if _game_dir(game) == "k2" else GameVersion.K1


def _find_fixture_file(root: Path, game: str, resref: str, ext: str) -> Path:
    """Find a fixture file case-insensitively."""

    game_root = root / _game_dir(game)
    candidate = game_root / f"{resref}.{ext}"
    if candidate.exists():
        return candidate
    wanted = f"{resref}.{ext}".lower()
    if game_root.exists():
        for path in game_root.iterdir():
            if path.name.lower() == wanted:
                return path
    raise FileNotFoundError(f"{resref}.{ext} not found under {game_root}")


def load_fixture_model(
    resref: str,
    *,
    game: str = "k1",
    corpus_root: str | Path | None = None,
) -> KotorModel:
    """Load an MDL/MDX pair from GhostRigger's isolated stock corpus."""

    root = Path(corpus_root or DEFAULT_CORPUS_ROOT)
    mdl = _find_fixture_file(root, game, resref, "mdl")
    mdx = _find_fixture_file(root, game, resref, "mdx")
    return load_model_from_file(str(mdl), str(mdx), _game_version(game))


def _load_model_for_sampling(
    model_resref: str,
    *,
    game: str,
    corpus_root: str | Path | None,
) -> KotorModel:
    path = Path(model_resref)
    if path.exists():
        mdx = path.with_suffix(".mdx")
        return load_model_from_file(str(path), str(mdx) if mdx.exists() else "", _game_version(game))
    return load_fixture_model(model_resref, game=game, corpus_root=corpus_root)


def _load_chain_models(
    model_resref: str,
    *,
    game: str = "k1",
    corpus_root: str | Path | None = None,
) -> list[KotorModel]:
    """Load model plus supermodel chain from the isolated corpus."""

    manager = StockCorpusResourceManager(corpus_root)
    SuperModelResolver.configure(manager)
    SuperModelResolver.clear_cache()

    model = _load_model_for_sampling(model_resref, game=game, corpus_root=corpus_root)
    chain = [model]
    visited = {str(model.name or model_resref).lower()}
    super_ref = str(getattr(model, "supermodel", "") or "")
    while super_ref and super_ref.strip().lower() not in SuperModelResolver._NULL_REFS:
        key = super_ref.lower()
        if key in visited:
            break
        visited.add(key)
        super_model = manager.load_model(super_ref, game)
        if super_model is None:
            break
        chain.append(super_model)
        super_ref = str(getattr(super_model, "supermodel", "") or "")
    return chain


def resolve_supermodel_chain(
    model_resref: str,
    game: str,
    *,
    corpus_root: str | Path | None = None,
) -> list[str]:
    """Return the model -> supermodel chain using the stock corpus."""

    return [str(model.name or "") for model in _load_chain_models(
        model_resref,
        game=game,
        corpus_root=corpus_root,
    )]


def _find_animation(model: KotorModel, clip_name: str) -> Optional[Animation]:
    target = clip_name.lower()
    for anim in getattr(model, "animations", []) or []:
        if str(anim.name or "").lower() == target:
            return anim
    return None


def build_effective_controller_table(
    chain: list[str],
    clip_name: str,
    *,
    game: str = "k1",
    corpus_root: str | Path | None = None,
) -> dict[str, BoneControllers]:
    """Return first-wins controller ownership by node name.

    The table is diagnostic-friendly: it records which model in the chain first
    supplied controller data for each node in ``clip_name``.  Sampling itself is
    still delegated to ``AnimationEngine`` so playback behavior stays aligned
    with GhostRigger's viewport.
    """

    table: dict[str, BoneControllers] = {}
    cumulative_scale = 1.0
    for index, resref in enumerate(chain):
        try:
            model = load_fixture_model(resref, game=game, corpus_root=corpus_root)
        except FileNotFoundError:
            cumulative_scale *= 1.0
            continue
        anim = _find_animation(model, clip_name)
        if anim is not None:
            for node in anim.nodes:
                node_key = str(getattr(node, "name", "") or "").lower()
                if not node_key or node_key in table:
                    continue
                table[node_key] = BoneControllers(
                    source_model=str(model.name or resref),
                    clip_name=str(anim.name or clip_name),
                    node_name=str(getattr(node, "name", "") or ""),
                    controllers=tuple(getattr(node, "controllers", []) or []),
                    anim_scale=1.0 if index == 0 else cumulative_scale,
                )
        step_scale = float(getattr(model, "anim_scale", 1.0) or 1.0)
        cumulative_scale *= step_scale
    return table


def _retarget_bone_nodes(model: KotorModel) -> list[ModelNode]:
    """Return the DFS node set sampled as the target skeleton."""

    return [node for node in model.all_nodes() if not bool(getattr(node, "is_skin", False))]


def _xyzw_to_wxyz(rotation: Any) -> tuple[float, float, float, float]:
    x, y, z, w = (list(rotation or (0.0, 0.0, 0.0, 1.0)) + [0.0, 0.0, 0.0, 1.0])[:4]
    return (float(w), float(x), float(y), float(z))


def _node_pose_values(model_node: ModelNode, pose_node: Any | None) -> tuple[tuple, tuple, tuple]:
    if pose_node is None:
        pos = getattr(model_node, "position", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0)
        rot = getattr(model_node, "rotation", (0.0, 0.0, 0.0, 1.0)) or (0.0, 0.0, 0.0, 1.0)
        scale = 1.0
    else:
        pos = getattr(pose_node, "position", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0)
        rot = getattr(pose_node, "rotation", (0.0, 0.0, 0.0, 1.0)) or (0.0, 0.0, 0.0, 1.0)
        scale = float(getattr(pose_node, "scale", 1.0) or 1.0)
    return (
        tuple(float(v) for v in pos[:3]),
        _xyzw_to_wxyz(rot),
        (scale, scale, scale),
    )


def sample_clip_to_fixed_rate(
    source_model: str,
    clip_name: str,
    fps: float = 60.0,
    *,
    game: str = "k1",
    corpus_root: str | Path | None = None,
) -> SampledClip:
    """Sample a KotOR clip to fixed-rate local-space TRS arrays.

    The returned sample is sourced from the isolated corpus by default.  Palette
    evaluation calls the G5-backed ``MatrixPaletteUploader`` once per frame to
    keep the bake path tied to GhostRigger's audited skinning formula.
    """

    if fps <= 0:
        raise ValueError("fps must be positive")

    manager = StockCorpusResourceManager(corpus_root)
    SuperModelResolver.configure(manager)
    SuperModelResolver.clear_cache()

    model = _load_model_for_sampling(source_model, game=game, corpus_root=corpus_root)
    SuperModelResolver.prime_cache(str(model.name or source_model), model)
    engine = AnimationEngine(model)
    if not engine.play(clip_name, loop=False, blend=False):
        raise ValueError(f"Animation {clip_name!r} not found for {source_model!r}")
    anim = engine.current_animation
    if anim is None:
        raise ValueError(f"Animation {clip_name!r} did not resolve for {source_model!r}")

    duration = max(0.0, float(getattr(anim, "length", 0.0) or 0.0))
    frame_count = max(1, int(round(duration * fps)) + 1)
    nodes = _retarget_bone_nodes(model)
    bone_names = [str(node.name or "") for node in nodes]
    bone_count = len(nodes)

    positions = np.zeros((frame_count, bone_count, 3), dtype=np.float32)
    rotations = np.zeros((frame_count, bone_count, 4), dtype=np.float32)
    scales = np.ones((frame_count, bone_count, 3), dtype=np.float32)

    uploader = MatrixPaletteUploader()
    uploader.build_inverse_bind_pose(model)
    base_pose = engine.evaluate(0.0)
    palette_frames = 0

    for frame_index in range(frame_count):
        t = min(duration, frame_index / fps)
        pose = engine.evaluate(t)
        uploader.compute_palette(pose, anim_base_pose=base_pose)
        palette_frames += 1
        pose_nodes = {str(name).lower(): value for name, value in pose.nodes.items()}
        for bone_index, node in enumerate(nodes):
            pos, rot_wxyz, scale_vec = _node_pose_values(
                node,
                pose_nodes.get(str(node.name or "").lower()),
            )
            positions[frame_index, bone_index, :] = pos
            rotations[frame_index, bone_index, :] = rot_wxyz
            scales[frame_index, bone_index, :] = scale_vec

    chain = resolve_supermodel_chain(source_model, game, corpus_root=corpus_root)
    resolved_source = ""
    table = build_effective_controller_table(chain, clip_name, game=game, corpus_root=corpus_root)
    if table:
        resolved_source = next(iter(table.values())).source_model

    return SampledClip(
        clip_name=str(anim.name or clip_name),
        fps=float(fps),
        frame_count=frame_count,
        bone_names=bone_names,
        positions=positions,
        rotations=rotations,
        scales=scales,
        source_model=str(model.name or source_model),
        source_chain=chain,
        resolved_clip_source=resolved_source,
        duration_s=duration,
        bake_math_audit_id="G5",
        palette_frames=palette_frames,
    )
