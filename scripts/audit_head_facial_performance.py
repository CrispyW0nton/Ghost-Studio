"""Audit and render a custom head's real KOTOR dialogue facial performance.

The command is model-agnostic.  Xaria is the first production fixture, but no
application rule or default path is tied to her.  The report measures skinned
mouth deformation through all 16 talk slots, audits every supplied LIP
timeline, and optionally renders a labeled range-of-motion contact sheet.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _configure_imports() -> None:
    for path in (ROOT, ROOT / "src"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    from scripts.mcp.start_kotormcp_stdio import _python_roots

    for path in reversed(_python_roots(ROOT)):
        text = str(path)
        if path.exists() and text not in sys.path:
            sys.path.insert(0, text)


_configure_imports()


def _facial_frame_bounds(model: Any) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Return a face-only camera envelope derived from actual skin weights."""

    import numpy as np

    for node in model.all_nodes() if hasattr(model, "all_nodes") else ():
        vertices = list(getattr(node, "vertices", ()) or ())
        rows = list(getattr(node, "skin_data", ()) or ())
        palette = [
            str(value or "").casefold()
            for value in list(getattr(node, "bone_map", ()) or ())
        ]
        facial_slots = {
            index
            for index, name in enumerate(palette)
            if name.startswith("f_")
        }
        if not vertices or len(rows) != len(vertices) or not facial_slots:
            continue
        selected_indices = [
            index
            for index, row in enumerate(rows)
            if sum(
                float(getattr(influence, "weight", 0.0) or 0.0)
                for influence in list(getattr(row, "influences", ()) or ())
                if int(getattr(influence, "bone_index", -1)) in facial_slots
            )
            >= 0.08
        ]
        if len(selected_indices) < 20:
            continue
        from src.core.characters.head_facial_transplant import (
            _node_vertices_in_world,
        )

        world_vertices = _node_vertices_in_world(node)
        if len(world_vertices) != len(vertices):
            continue
        selected = [world_vertices[index] for index in selected_indices]
        points = np.asarray(selected, dtype=np.float64)
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        center = (minimum + maximum) * 0.5
        extent = np.maximum(maximum - minimum, 1.0e-4)
        # Keep forehead/chin context while preventing long hair and horns from
        # shrinking the mouth to a few pixels.
        extent[0] *= 1.35
        extent[2] *= 1.35
        extent[1] = max(extent[1], extent[0] * 0.6)
        return (
            tuple(float(value) for value in center - extent * 0.5),
            tuple(float(value) for value in center + extent * 0.5),
        )
    model.compute_bounds()
    return tuple(model.bb_min), tuple(model.bb_max)


def _load_head(mdl_path: Path, mdx_path: Path, game_dir: Path) -> Any:
    from src.core.animation.animation_engine import SuperModelResolver
    from src.core.assets.resource_manager import ResourceManager
    from src.core.game.kotor_loader import load_model_from_file
    from src.core.geometry.model_data import GameVersion

    manager = ResourceManager()
    if not manager.set_k2_dir(str(game_dir)):
        raise RuntimeError(f"Could not configure the K2 installation: {game_dir}")
    SuperModelResolver.clear_cache()
    SuperModelResolver.configure(manager)
    return load_model_from_file(
        str(mdl_path),
        str(mdx_path),
        GameVersion.K2,
    )


def _audit_lips(lip_paths: list[Path]) -> list[dict[str, Any]]:
    from src.core.characters.facial_rig_qa import audit_lip_timeline
    from src.core.special.lip_reader import LIPFile

    reports = []
    for path in sorted(lip_paths):
        lip = LIPFile.from_file(str(path))
        reports.append(audit_lip_timeline(lip, name=path.stem).to_dict())
    return reports


def _probe_media_duration(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return max(0.0, float(completed.stdout.strip()))


def _audio_lip_sync(
    *,
    lip_duration: float,
    audio_duration: float,
) -> dict[str, Any]:
    from src.core.characters.facial_rig_qa import audit_audio_lip_sync

    return audit_audio_lip_sync(
        lip_duration=lip_duration,
        audio_duration=audio_duration,
    ).to_dict()


def _load_texture_aliases(
    model: Any,
    texture_path: Path | None,
) -> dict[str, Any]:
    """Expose one authored texture under every material name used by the head."""

    if texture_path is None:
        return {}
    from PIL import Image

    image = Image.open(texture_path).convert("RGBA")
    aliases = {texture_path.stem.casefold()}
    for node in model.all_nodes() if hasattr(model, "all_nodes") else ():
        texture = str(getattr(node, "texture", "") or "").strip()
        if texture and texture.casefold() != "null":
            aliases.add(texture.casefold())
        for name in list(getattr(node, "texture_names", ()) or ()):
            clean = str(name or "").strip()
            if clean and clean.casefold() != "null":
                aliases.add(clean.casefold())
    return {alias: image for alias in aliases}


def _render_contact_sheet(
    model: Any,
    shape_times: tuple[float, ...],
    output_path: Path,
    texture_path: Path | None,
) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    from src.adapters.rendering.moderngl_legacy_bridge import GpuRenderer
    from src.core.animation.animation_engine import AnimationEngine
    from src.core.animation.facial_performance import KOTOR_VISEME_NAMES
    from src.core.camera.arcball_camera import ArcBallCamera

    engine = AnimationEngine(model)
    if not engine.play("talk", loop=False, blend=False):
        raise RuntimeError("Head has no local or inherited talk animation")
    camera = ArcBallCamera()
    camera.frame_bounds(*_facial_frame_bounds(model), reset_view=True)
    camera.elevation = 4.0
    camera.distance *= 0.92
    textures = _load_texture_aliases(model, texture_path)

    tile_size = 384
    label_height = 32
    rendered: list[Image.Image] = []
    renderer = GpuRenderer()
    renderer.show_grid = False
    renderer.cull_faces = False
    renderer.lighting_mode = "fullbright"
    try:
        if not renderer._ensure_context():
            raise RuntimeError("ModernGL standalone renderer is unavailable")
        neutral = engine.evaluate(shape_times[0])
        for index, time_value in enumerate(shape_times):
            pose = engine.evaluate(time_value)
            for target in (neutral, pose):
                setattr(target, "_gr_animation_source_model_id", id(model))
                setattr(
                    target,
                    "_gr_animation_source_model_name",
                    str(getattr(model, "name", "") or "head"),
                )
                setattr(target, "_gr_animation_name", "talk")
            image = renderer.render(
                model,
                camera,
                tile_size,
                tile_size,
                textures=textures,
                anim_pose=pose,
                anim_time=time_value,
                anim_base_pose=neutral,
            )
            if image is None:
                raise RuntimeError(f"Renderer returned no image for shape {index}")
            tile = Image.new(
                "RGB",
                (tile_size, tile_size + label_height),
                (18, 18, 18),
            )
            tile.paste(image.convert("RGB"), (0, 0))
            draw = ImageDraw.Draw(tile)
            draw.text(
                (10, tile_size + 8),
                f"{index:02d}  {KOTOR_VISEME_NAMES[index]}",
                fill=(235, 235, 235),
            )
            rendered.append(tile)
    finally:
        renderer.release()

    sheet = Image.new(
        "RGB",
        (tile_size * 4, (tile_size + label_height) * 4),
        (12, 12, 12),
    )
    for index, tile in enumerate(rendered):
        sheet.paste(
            tile,
            ((index % 4) * tile_size, (index // 4) * (tile_size + label_height)),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return {
        "path": str(output_path.resolve()),
        "width": sheet.width,
        "height": sheet.height,
        "shape_count": len(rendered),
    }


def _render_dialogue_video(
    model: Any,
    lip_path: Path,
    audio_path: Path,
    output_path: Path,
    texture_path: Path | None,
    *,
    fps: float,
    size: int,
) -> dict[str, Any]:
    from PIL import Image

    from src.adapters.rendering.moderngl_legacy_bridge import GpuRenderer
    from src.core.camera.arcball_camera import ArcBallCamera
    from src.core.characters.character_builder import LIPPlayback
    from src.core.special.lip_reader import LIPFile

    playback = LIPPlayback()
    lip = LIPFile.from_file(str(lip_path))
    if not playback.load_lip(lip) or not playback.load_talk_animation(model):
        raise RuntimeError("Could not bind the LIP timeline to the head talk animation")
    rate = max(1.0, float(fps))
    dimension = max(128, int(size))
    frame_count = max(1, int(round(playback.duration * rate)))
    camera = ArcBallCamera()
    camera.frame_bounds(*_facial_frame_bounds(model), reset_view=True)
    camera.elevation = 4.0
    camera.distance *= 0.92
    textures = _load_texture_aliases(model, texture_path)
    neutral = playback.animation_pose_for_viseme(0)
    if neutral is None:
        raise RuntimeError("Could not evaluate Xaria's neutral talk pose")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_sync = _audio_lip_sync(
        lip_duration=playback.duration,
        audio_duration=_probe_media_duration(audio_path),
    )
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{dimension}x{dimension}",
        "-r",
        f"{rate:.6f}",
        "-i",
        "-",
        "-i",
        str(audio_path),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-af",
        "apad",
        "-t",
        f"{playback.duration:.6f}",
        str(output_path),
    ]
    encoder = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    renderer = GpuRenderer()
    renderer.show_grid = False
    renderer.cull_faces = False
    renderer.lighting_mode = "fullbright"
    try:
        if not renderer._ensure_context():
            raise RuntimeError("ModernGL standalone renderer is unavailable")
        if encoder.stdin is None:
            raise RuntimeError("FFmpeg video pipe did not open")
        for frame_index in range(frame_count):
            time_value = frame_index / rate
            pose = playback.animation_pose_at_time(time_value)
            image = renderer.render(
                model,
                camera,
                dimension,
                dimension,
                textures=textures,
                anim_pose=pose,
                anim_time=time_value,
                anim_base_pose=neutral,
            )
            if image is None:
                raise RuntimeError(
                    f"Renderer returned no image at dialogue frame {frame_index}"
                )
            encoder.stdin.write(image.convert("RGB").tobytes())
        encoder.stdin.close()
        stderr = encoder.stderr.read() if encoder.stderr is not None else b""
        return_code = encoder.wait()
        if return_code:
            raise RuntimeError(
                "FFmpeg could not encode the dialogue proof: "
                + stderr.decode("utf-8", errors="replace")
            )
    finally:
        renderer.release()
        if encoder.poll() is None:
            encoder.kill()
    return {
        "path": str(output_path.resolve()),
        "lip": str(lip_path.resolve()),
        "audio": str(audio_path.resolve()),
        "fps": rate,
        "frame_count": frame_count,
        "duration": playback.duration,
        "width": dimension,
        "height": dimension,
        "audio_sync": audio_sync,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from src.core.characters.facial_rig_qa import audit_head_facial_range

    model = _load_head(args.mdl, args.mdx, args.game_dir)
    facial = audit_head_facial_range(model)
    lip_paths = (
        list(args.lip_dir.glob("*.lip"))
        if args.lip_dir is not None
        else list(args.lip)
    )
    report: dict[str, Any] = {
        "schema": "ghostrigger.head-facial-performance-audit.v1",
        "model": {
            "name": str(getattr(model, "name", "") or args.mdl.stem),
            "mdl": str(args.mdl.resolve()),
            "mdx": str(args.mdx.resolve()),
            "supermodel": str(getattr(model, "supermodel", "") or ""),
        },
        "facial_range": facial.to_dict(),
        "lip_timelines": _audit_lips(lip_paths),
    }
    report["ok"] = bool(
        facial.ok
        and report["lip_timelines"]
        and all(item["ok"] for item in report["lip_timelines"])
    )
    if args.contact_sheet is not None:
        report["contact_sheet"] = _render_contact_sheet(
            model,
            tuple(facial.shape_times),
            args.contact_sheet,
            args.texture,
        )
    if args.preview_video is not None:
        if args.preview_lip is None or args.preview_audio is None:
            raise ValueError(
                "--preview-video requires --preview-lip and --preview-audio"
            )
        report["dialogue_preview"] = _render_dialogue_video(
            model,
            args.preview_lip,
            args.preview_audio,
            args.preview_video,
            args.texture,
            fps=args.preview_fps,
            size=args.preview_size,
        )
        report["ok"] = bool(
            report["ok"]
            and report["dialogue_preview"]["audio_sync"]["ok"]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    from src.resources.game_detector import detect_kotor_dirs

    _k1_dir, detected_k2_dir = detect_kotor_dirs(prefer_config=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdl", type=Path, required=True)
    parser.add_argument("--mdx", type=Path, required=True)
    parser.add_argument(
        "--game-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "K2_PATH",
                detected_k2_dir
                or r"h:\steam\steamapps\common\Knights of the Old Republic II",
            )
        ),
    )
    parser.add_argument("--lip-dir", type=Path)
    parser.add_argument("--lip", type=Path, action="append", default=[])
    parser.add_argument("--texture", type=Path)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--preview-lip", type=Path)
    parser.add_argument("--preview-audio", type=Path)
    parser.add_argument("--preview-video", type=Path)
    parser.add_argument("--preview-fps", type=float, default=24.0)
    parser.add_argument("--preview-size", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = run(args)
    print(json.dumps({"ok": report["ok"], "output": str(args.output.resolve())}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
