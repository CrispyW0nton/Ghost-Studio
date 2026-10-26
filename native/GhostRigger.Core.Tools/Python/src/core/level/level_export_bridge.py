"""Compatibility contracts for the Core.IO KMAP FBX exporter."""
from __future__ import annotations

from dataclasses import dataclass, field
from .kmap_validator import KMapValidator, KMapValidationIssue


@dataclass(frozen=True)
class LevelExportOptions:
    selected_only: bool = False
    visible_only: bool = True
    include_textures: bool = True
    include_lightmaps: bool = True
    include_walkmesh: bool = True
    include_lights: bool = True
    include_cameras: bool = True
    bake_transforms: bool = True
    copy_textures: bool = True
    generate_sidecar_manifest: bool = True
    dry_run: bool = False
    overwrite: bool = True


@dataclass
class LevelExportResult:
    ok: bool = False
    code: str = "not_exported"
    message: str = ""
    fbx_path: str = ""
    output_path: str = ""
    manifest_path: str = ""
    warnings: list[str] = field(default_factory=list)
    issues: list[KMapValidationIssue] = field(default_factory=list)


class LevelExportBridge:
    def __init__(self, validator=None):
        self.validator = validator or KMapValidator()

    def export_fbx(self, project, output_path, options=None, **context):
        return self.export_scene(project, output_path, options, **context)

    def export_obj(self, project, output_path, options=None, **context):
        return self.export_scene(project, output_path, options, **context)

    def export_scene(self, project, output_path, options=None, **context):
        from src.io.level_scene_export import export_level_scene
        options = options or LevelExportOptions()
        issues = self.validator.validate(project)
        if options.dry_run:
            return LevelExportResult(ok=True, code="dry_run", message="Scene export dry run: no files written.", issues=issues)
        # Game-readiness diagnostics remain visible. Geometry export has its
        # own resource/transform gates; gameplay/pathfinding is not required.
        return export_level_scene(project, output_path, options, issues=issues, **context)
