"""Focused installed-game room FBX/OBJ export proof (no source writes)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from src.core.assets.resource_manager import ResourceManager
    from src.core.level.kmap_model import KMapProject, RoomInstance, LevelTransform
    from src.core.level.kmap_serializer import KMapSerializer
    from src.core.level.level_export_bridge import LevelExportBridge

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k2-dir", type=Path, help="KOTOR II installation; defaults to local settings.json")
    args = parser.parse_args()
    settings_path = ROOT / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.is_file() else {}
    game_dir = args.k2_dir or settings.get("k2_dir")
    if not game_dir:
        parser.error("Supply --k2-dir or configure k2_dir in settings.json")
    manager = ResourceManager()
    if not manager.set_k2_dir(str(game_dir)):
        raise RuntimeError("Configured K2 installation could not be indexed")
    project = KMapProject(name="level_export_proof", game="K2", source_game="K2", target_game="K2", rooms=[
        RoomInstance(name="Ebon Hawk", model_resref="001ebo1"),
        RoomInstance(name="Translated Ebon Hawk", model_resref="001ebo1", transform=LevelTransform(position=(100, 0, 0))),
    ])
    args.output.mkdir(parents=True, exist_ok=True)
    KMapSerializer().save(project, args.output / "level_export_proof.kmap")
    for extension in ("fbx", "obj"):
        result = LevelExportBridge().export_scene(project, args.output / f"level_export_proof.{extension}", resource_manager=manager)
        print(json.dumps(result.__dict__, default=str, indent=2))
        if not result.ok:
            raise RuntimeError(result.message)


if __name__ == "__main__":
    main()
