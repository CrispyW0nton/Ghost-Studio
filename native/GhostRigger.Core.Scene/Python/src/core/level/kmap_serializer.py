"""JSON serializer for GhostRigger KMAP files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .kmap_model import (
    KMAP_FILE_TYPE,
    KMAP_FILE_VERSION,
    BlueprintEntry,
    KMapProject,
    MaterialReference,
    ModuleInstance,
    RoomInstance,
    TextureReference,
    WalkmeshReference,
    new_kmap_project,
    utc_now_iso,
)


class KMapSerializer:
    """Load/save versioned, human-readable `.kmap` JSON files."""

    @classmethod
    def load(cls, path: str | Path) -> KMapProject:
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid KMAP JSON: {exc}") from exc
        if data.get("unreal_source"):
            from src.io.unreal_kmap_storage import hydrate_unreal_document
            data = hydrate_unreal_document(data, source)
        project = cls.from_dict(data)
        project.path = str(source)
        project.dirty = False
        return project

    @classmethod
    def save(cls, project: KMapProject, path: str | Path | None = None) -> None:
        target = Path(path or project.path)
        if not target:
            raise ValueError("No KMAP output path was provided.")
        target.parent.mkdir(parents=True, exist_ok=True)
        modified_at = utc_now_iso()
        if project.extra_sections.get("unreal_source"):
            from src.io.unreal_kmap_storage import persist_unreal_document
            document = cls.to_dict(project)
            document["project"]["modified_at"] = modified_at
            persist_unreal_document(document, target, project.path)
            project.modified_at, project.path, project.dirty = modified_at, str(target), False
            return
        project.modified_at = modified_at
        target.write_text(json.dumps(cls.to_dict(project), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        project.path = str(target)
        project.dirty = False

    @classmethod
    def validate_schema(cls, data: Any) -> list[str]:
        errors: list[str] = []
        if not isinstance(data, dict):
            return ["KMAP root must be a JSON object."]
        if data.get("file_type") != KMAP_FILE_TYPE:
            errors.append(f"file_type must be {KMAP_FILE_TYPE}.")
        version = data.get("file_version")
        if not isinstance(version, int):
            errors.append("file_version must be an integer.")
        elif version > KMAP_FILE_VERSION:
            errors.append(f"KMAP version {version} is newer than supported version {KMAP_FILE_VERSION}.")
        project = data.get("project")
        if not isinstance(project, dict):
            errors.append("project must be an object.")
        else:
            for key in ("id", "name", "game", "created_at", "modified_at"):
                if key not in project:
                    errors.append(f"project.{key} is required.")
        for key in ("modules", "rooms", "walkmeshes", "blueprints", "textures", "materials"):
            if key in data and not isinstance(data.get(key), list):
                errors.append(f"{key} must be a list.")
        return errors

    @classmethod
    def migrate(cls, data: dict[str, Any]) -> dict[str, Any]:
        version = int(data.get("file_version") or 1)
        if version == KMAP_FILE_VERSION:
            return data
        if version < 1:
            data["file_version"] = 1
        return data

    @classmethod
    def from_dict(cls, data: Any) -> KMapProject:
        errors = cls.validate_schema(data)
        if errors:
            raise ValueError("Invalid KMAP schema: " + "; ".join(errors))
        data = cls.migrate(dict(data))
        project_data = dict(data.get("project") or {})
        units = dict(data.get("units") or {})
        project = new_kmap_project(
            name=str(project_data.get("name") or "new_level"),
            game=str(project_data.get("game") or "K1"),
            author=str(project_data.get("author") or ""),
        )
        project.project_id = str(project_data.get("id") or project.project_id)
        project.description = str(project_data.get("description") or "")
        project.source_game = str(project_data.get("source_game") or project.game).upper()
        project.target_game = str(project_data.get("target_game") or project.game).upper()
        project.created_at = str(project_data.get("created_at") or project.created_at)
        project.modified_at = str(project_data.get("modified_at") or project.modified_at)
        project.source_directory = str(project_data.get("source_directory") or "")
        project.output_directory = str(project_data.get("output_directory") or "")
        project.metadata = dict(project_data.get("metadata") or data.get("metadata") or {})
        project.system_unit = str(units.get("system_unit") or "cm")
        project.display_unit = str(units.get("display_unit") or "cm")
        project.modules = [ModuleInstance.from_dict(item) for item in data.get("modules", []) or []]
        project.rooms = [RoomInstance.from_dict(item) for item in data.get("rooms", []) or []]
        project.objects = [dict(item) for item in data.get("objects", []) or [] if isinstance(item, dict)]
        project.lights = [dict(item) for item in data.get("lights", []) or [] if isinstance(item, dict)]
        project.cameras = [dict(item) for item in data.get("cameras", []) or [] if isinstance(item, dict)]
        project.sequences = [dict(item) for item in data.get("sequences", []) or [] if isinstance(item, dict)]
        project.materials = [MaterialReference.from_dict(item) for item in data.get("materials", []) or []]
        project.textures = [TextureReference.from_dict(item) for item in data.get("textures", []) or []]
        project.walkmeshes = [WalkmeshReference.from_dict(item) for item in data.get("walkmeshes", []) or []]
        project.blueprints = [BlueprintEntry.from_dict(item) for item in data.get("blueprints", []) or []]
        project.exports = dict(data.get("exports") or {})
        known = {
            "file_type",
            "file_version",
            "project",
            "units",
            "modules",
            "rooms",
            "objects",
            "lights",
            "cameras",
            "sequences",
            "materials",
            "textures",
            "walkmeshes",
            "blueprints",
            "exports",
            "metadata",
        }
        project.extra_sections = {key: value for key, value in data.items() if key not in known}
        project.dirty = False
        return project

    @classmethod
    def to_dict(cls, project: KMapProject) -> dict[str, Any]:
        data: dict[str, Any] = {
            "file_type": KMAP_FILE_TYPE,
            "file_version": KMAP_FILE_VERSION,
            "project": {
                "id": project.project_id,
                "name": project.name,
                "description": project.description,
                "game": project.game,
                "source_game": project.source_game,
                "target_game": project.target_game,
                "created_at": project.created_at,
                "modified_at": project.modified_at,
                "author": project.author,
                "source_directory": project.source_directory,
                "output_directory": project.output_directory,
                "metadata": dict(project.metadata),
            },
            "units": {
                "system_unit": project.system_unit,
                "display_unit": project.display_unit,
            },
            "modules": [module.to_dict() for module in project.modules],
            "rooms": [room.to_dict() for room in project.rooms],
            "objects": list(project.objects),
            "lights": list(project.lights),
            "cameras": list(project.cameras),
            "sequences": list(project.sequences),
            "materials": [material.to_dict() for material in project.materials],
            "textures": [texture.to_dict() for texture in project.textures],
            "walkmeshes": [walkmesh.to_dict() for walkmesh in project.walkmeshes],
            "blueprints": [blueprint.to_dict() for blueprint in project.blueprints],
            "exports": dict(project.exports),
            "metadata": dict(project.metadata),
        }
        data.update(project.extra_sections)
        return data
