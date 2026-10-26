"""KOTOR narrative resource inspection, packaging, and safe test staging.

Archive writes use PyKotor and are accepted only after byte-for-byte resource
readback.  Override deployment is deliberately split into two operations:
``stage_override`` writes an ordinary build folder, while ``install_override``
(and its compare-and-swap undo, ``restore_override_file``) are the only APIs in
this module that may write beneath a game installation's Override folder.
Existing Override resources are never replaced unless the caller explicitly
selects the backup policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import uuid4

from .project import NarrativeProject, NarrativeProjectService


OVERRIDE_STAGE_FILE_TYPE = "GhostStudioOverrideStage"
PACKAGE_MANIFEST_FILE_TYPE = "GhostStudioNarrativePackage"
_RESREF_PATTERN = re.compile(r"^[a-z0-9_]{1,16}$")
_ARCHIVE_RESOURCE_TYPES = {"erf", "mod", "rim", "sav"}
_GLOBAL_INSTALL_RESOURCE_TYPES = {"tlk"}
_BLUEPRINT_KIND = {
    "utc": "creature",
    "uti": "item",
    "utp": "placeable",
    "utd": "door",
    "ute": "encounter",
    "utm": "store",
    "uts": "sound",
    "utt": "trigger",
    "utw": "waypoint",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(bytes(data)).hexdigest()


def _resource_type(extension: object):
    from pykotor.resource.type import ResourceType

    value = str(extension or "").strip().lower().lstrip(".")
    resource_type = ResourceType.from_extension(value)
    if int(resource_type.type_id) < 0 or not value:
        raise ValueError(f"Unknown KOTOR resource type: {value or '<empty>'}")
    return resource_type


def _resref(value: object) -> str:
    text = str(value or "").strip().lower()
    if not _RESREF_PATTERN.fullmatch(text):
        raise ValueError("Resource identifiers must use 1-16 lowercase letters, numbers, or underscores.")
    return text


def _game_key(value: object) -> str:
    game = str(value or "K2").strip().upper()
    if game in {"K1", "1", "KOTOR", "KOTOR1"}:
        return "K1"
    if game in {"K2", "2", "TSL", "KOTOR2"}:
        return "K2"
    raise ValueError("Target game must be K1 or K2.")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(bytes(data))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(path, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))


@dataclass(frozen=True)
class NarrativePackageIssue:
    severity: str
    code: str
    message: str
    resource: str = ""

    @property
    def blocking(self) -> bool:
        return self.severity.lower() in {"blocking", "error"}


@dataclass(frozen=True)
class PackageResource:
    resref: str
    restype: str
    data: bytes
    source_asset_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "resref", _resref(self.resref))
        object.__setattr__(self, "restype", _resource_type(self.restype).extension.lower())
        object.__setattr__(self, "data", bytes(self.data or b""))
        object.__setattr__(self, "source_asset_id", str(self.source_asset_id or ""))

    @property
    def filename(self) -> str:
        return f"{self.resref}.{self.restype}"

    @property
    def sha256(self) -> str:
        return _sha256(self.data)

    @property
    def identity(self) -> tuple[str, str]:
        return self.resref, self.restype

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        resref: str | None = None,
        restype: str | None = None,
        source_asset_id: str = "",
    ) -> "PackageResource":
        requested = Path(path)
        if not requested.is_file() or requested.is_symlink():
            raise FileNotFoundError(f"Package resource is missing or unsafe: {requested}")
        source = requested.resolve()
        return cls(
            resref or source.stem,
            restype or source.suffix,
            source.read_bytes(),
            source_asset_id,
        )


@dataclass(frozen=True)
class NarrativePackageInspection:
    archive_type: str
    resources: tuple[PackageResource, ...]
    sha256: str
    byte_count: int


@dataclass(frozen=True)
class NarrativePackageResult:
    output_path: str
    archive_type: str
    resources: tuple[PackageResource, ...]
    issues: tuple[NarrativePackageIssue, ...] = ()
    committed: bool = False
    manifest_path: str = ""

    @property
    def ok(self) -> bool:
        return self.committed and not any(row.blocking for row in self.issues)


@dataclass(frozen=True)
class OverrideStageResult:
    stage_path: str
    game: str
    resources: tuple[PackageResource, ...]
    issues: tuple[NarrativePackageIssue, ...] = ()
    committed: bool = False
    manifest_path: str = ""

    @property
    def ok(self) -> bool:
        return self.committed and not any(row.blocking for row in self.issues)


@dataclass(frozen=True)
class OverrideInstallResult:
    game_root: str
    game: str
    installed: tuple[str, ...] = ()
    skipped_identical: tuple[str, ...] = ()
    backup_path: str = ""
    receipt_path: str = ""
    issues: tuple[NarrativePackageIssue, ...] = ()
    committed: bool = False

    @property
    def ok(self) -> bool:
        return self.committed and not any(row.blocking for row in self.issues)


@dataclass(frozen=True)
class OverrideRestoreResult:
    """Receipt for undoing one Override install (prior bytes back, or removal)."""

    game_root: str
    filename: str
    action: str = ""
    restored_sha256: str = ""
    pre_restore_sha256: str = ""
    backup_path: str = ""
    receipt_path: str = ""
    issues: tuple[NarrativePackageIssue, ...] = ()
    committed: bool = False

    @property
    def ok(self) -> bool:
        return self.committed and not any(row.blocking for row in self.issues)


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _external_backup_root(backup_root: str | Path, game_root: Path) -> Path:
    """An existing, non-linked folder outside (and not containing) the game root."""

    requested = Path(backup_root)
    if not requested.is_absolute():
        raise ValueError("backup_root must be an absolute path.")
    external = requested.resolve(strict=True)
    if not external.is_dir() or requested.is_symlink() or external.is_symlink():
        raise ValueError("backup_root must be an existing safe directory.")
    if external == game_root or game_root in external.parents or external in game_root.parents:
        raise ValueError("backup_root and the game root must be separate directory trees.")
    return external


@dataclass(frozen=True)
class GlobalTlkInstallResult:
    """Receipt for an explicit game-root ``dialog.tlk`` install or restore."""

    game_root: str
    game: str
    dialog_path: str = ""
    backup_path: str = ""
    receipt_path: str = ""
    installed_sha256: str = ""
    restored: bool = False
    issues: tuple[NarrativePackageIssue, ...] = ()
    committed: bool = False

    @property
    def ok(self) -> bool:
        return self.committed and not any(row.blocking for row in self.issues)


def _project_resources(project: NarrativeProject, *, include_source: bool) -> tuple[PackageResource, ...]:
    project_issues = NarrativeProjectService.validate_project(project)
    blocking = tuple(row for row in project_issues if row.blocking)
    if blocking:
        raise ValueError("Project is not package-ready: " + "; ".join(row.message for row in blocking))
    root = Path(project.root_path).resolve()
    resources: list[PackageResource] = []
    for asset in project.assets:
        if asset.role == "source" and not include_source:
            continue
        source = (root / Path(asset.path)).resolve()
        try:
            source.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Project asset escapes its root: {asset.path}") from error
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Project asset is missing or unsafe: {asset.path}")
        resources.append(PackageResource(asset.resref, asset.restype, source.read_bytes(), asset.asset_id))
    return tuple(resources)


def _validated_resources(resources: Iterable[PackageResource]) -> tuple[PackageResource, ...]:
    rows = tuple(row if isinstance(row, PackageResource) else PackageResource(**row) for row in resources)
    identities: dict[tuple[str, str], bytes] = {}
    unique: list[PackageResource] = []
    for row in rows:
        prior = identities.get(row.identity)
        if prior is not None and prior != row.data:
            raise ValueError(f"Conflicting bytes were supplied for {row.filename}.")
        if prior is None:
            identities[row.identity] = row.data
            unique.append(row)
    if not unique:
        raise ValueError("At least one resource is required.")
    return tuple(sorted(unique, key=lambda row: (row.resref, row.restype)))


def _archive_bytes(resources: Sequence[PackageResource], archive_type: str) -> bytes:
    from pykotor.resource.formats.erf import ERF, ERFType, bytes_erf
    from pykotor.resource.type import ResourceType

    kind = str(archive_type or "").strip().upper()
    if kind not in {"ERF", "MOD", "SAV"}:
        raise ValueError("Narrative archives must be ERF, MOD, or SAV.")
    # Odyssey save games use the same physical ``MOD `` container signature as
    # modules.  PyKotor models that distinction with ``is_save`` and the output
    # ResourceType, rather than a third ERFType enum value.
    archive = ERF(ERFType.MOD if kind in {"MOD", "SAV"} else ERFType.ERF, is_save=kind == "SAV")
    for row in resources:
        archive.set_data(row.resref, _resource_type(row.restype), row.data)
    output_type = {
        "ERF": ResourceType.ERF,
        "MOD": ResourceType.MOD,
        "SAV": ResourceType.SAV,
    }[kind]
    return bytes(bytes_erf(archive, output_type))


def inspect_narrative_archive(source: str | Path | bytes | bytearray) -> NarrativePackageInspection:
    from pykotor.resource.formats.erf import read_erf

    source_path = Path(source) if isinstance(source, (str, Path)) else None
    data = source_path.read_bytes() if source_path is not None else bytes(source)
    archive = read_erf(data)
    rows = tuple(
        sorted(
            (
                PackageResource(str(resource.resref), resource.restype.extension, bytes(resource.data))
                for resource in archive
            ),
            key=lambda row: (row.resref, row.restype),
        )
    )
    # A .sav intentionally has a physical ``MOD `` signature.  Preserve the
    # logical container type when a path supplies that otherwise-lost context.
    logical_type = "SAV" if source_path is not None and source_path.suffix.casefold() == ".sav" else str(archive.erf_type.name)
    return NarrativePackageInspection(
        logical_type,
        rows,
        _sha256(data),
        len(data),
    )


def _readback_matches(payload: bytes, expected: Sequence[PackageResource], archive_type: str) -> bool:
    inspection = inspect_narrative_archive(payload)
    expected_map = {row.identity: row.data for row in expected}
    actual_map = {row.identity: row.data for row in inspection.resources}
    expected_type = archive_type.upper()
    physical_type = "MOD" if expected_type == "SAV" else expected_type
    return inspection.archive_type == physical_type and actual_map == expected_map


def _promote_package_files(
    output: Path,
    archive_payload: bytes,
    manifest_path: Path,
    manifest_payload: dict[str, Any],
    *,
    resources: Sequence[PackageResource],
    archive_type: str,
) -> None:
    """Promote archive + audit manifest together and roll both back on failure."""

    output.parent.mkdir(parents=True, exist_ok=True)
    archive_stage = output.with_name(f".{output.name}.stage-{uuid4().hex}")
    manifest_stage = manifest_path.with_name(f".{manifest_path.name}.stage-{uuid4().hex}")
    archive_backup = output.with_name(f".{output.name}.backup-{uuid4().hex}")
    manifest_backup = manifest_path.with_name(f".{manifest_path.name}.backup-{uuid4().hex}")
    archive_had_prior = output.exists()
    manifest_had_prior = manifest_path.exists()
    archive_promoted = False
    manifest_promoted = False
    try:
        archive_stage.write_bytes(archive_payload)
        manifest_stage.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
        if archive_had_prior:
            os.replace(output, archive_backup)
        if manifest_had_prior:
            os.replace(manifest_path, manifest_backup)
        os.replace(archive_stage, output)
        archive_promoted = True
        os.replace(manifest_stage, manifest_path)
        manifest_promoted = True
        written = output.read_bytes()
        if written != archive_payload or not _readback_matches(written, resources, archive_type):
            raise ValueError("Written archive failed exact resource readback.")
        written_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            written_manifest.get("file_type") != PACKAGE_MANIFEST_FILE_TYPE
            or written_manifest.get("archive_sha256") != _sha256(written)
        ):
            raise ValueError("Written archive manifest failed readback.")
    except Exception:
        if manifest_promoted:
            manifest_path.unlink(missing_ok=True)
        if archive_promoted:
            output.unlink(missing_ok=True)
        if archive_had_prior and archive_backup.exists():
            os.replace(archive_backup, output)
        if manifest_had_prior and manifest_backup.exists():
            os.replace(manifest_backup, manifest_path)
        raise
    finally:
        archive_stage.unlink(missing_ok=True)
        manifest_stage.unlink(missing_ok=True)
        if output.exists():
            archive_backup.unlink(missing_ok=True)
        if manifest_path.exists():
            manifest_backup.unlink(missing_ok=True)


def _summary_value(value: object) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"byte_count": len(value)}
    if hasattr(value, "get") and value.__class__.__name__ == "ResRef":
        try:
            return str(value.get())
        except Exception:
            return str(value)
    if value.__class__.__name__ == "LocalizedString":
        substrings = dict(getattr(value, "_substrings_internal", {}) or {})
        return {
            "stringref": int(getattr(value, "stringref", -1)),
            "embedded_string_count": len(substrings),
        }
    if isinstance(value, (list, tuple)):
        return {"item_count": len(value)}
    if hasattr(value, "struct_id"):
        return {"struct_id": int(getattr(value, "struct_id", -1)), "field_count": len(value)}
    return str(value)


def inspect_gff_resource(data: bytes, *, restype: str) -> dict[str, Any]:
    """Return compact typed metadata without rewriting the GFF resource."""

    from pykotor.resource.formats.gff import read_gff

    resource_type = _resource_type(restype)
    if str(getattr(resource_type, "contents", "")) != "gff":
        raise ValueError(f"{resource_type.extension} is not a GFF-backed KOTOR resource type.")
    gff = read_gff(bytes(data))
    struct_count = 0
    field_count = 0
    list_count = 0

    def walk(struct: object) -> None:
        nonlocal struct_count, field_count, list_count
        struct_count += 1
        for _label, field_type, value in struct:
            field_count += 1
            kind = str(getattr(field_type, "name", field_type))
            if kind == "Struct":
                walk(value)
            elif kind == "List":
                list_count += 1
                for child in value:
                    walk(child)

    walk(gff.root)
    top_level: list[dict[str, Any]] = []
    semantic: dict[str, Any] = {}
    semantic_labels = {
        "Tag",
        "TemplateResRef",
        "Conversation",
        "Appearance_Type",
        "FirstName",
        "LastName",
        "LocalizedName",
        "Name",
        "Description",
    }
    for label, field_type, value in gff.root:
        type_name = str(getattr(field_type, "name", field_type))
        top_level.append({"label": str(label), "type": type_name})
        if str(label) in semantic_labels or str(label).lower().startswith("script"):
            semantic[str(label)] = _summary_value(value)
    extension = resource_type.extension.lower()
    return {
        "restype": extension,
        "content": str(getattr(gff.content, "name", gff.content)),
        "is_blueprint": extension in _BLUEPRINT_KIND,
        "blueprint_kind": _BLUEPRINT_KIND.get(extension, ""),
        "root_struct_id": int(getattr(gff.root, "struct_id", -1)),
        "struct_count": struct_count,
        "field_count": field_count,
        "list_count": list_count,
        "top_level_fields": top_level,
        "semantic_fields": semantic,
        "sha256": _sha256(data),
        "byte_count": len(data),
    }


def _owned_override_stage(path: Path) -> bool:
    manifest = path / "override-stage.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload.get("file_type") == OVERRIDE_STAGE_FILE_TYPE and int(payload.get("schema_version", 0)) == 1


def _promote_stage(staging: Path, target: Path, *, replace_owned: bool) -> None:
    backup = target.with_name(f".{target.name}.backup-{uuid4().hex}")
    previous_moved = False
    try:
        if target.exists():
            entries = tuple(target.iterdir()) if target.is_dir() else ()
            if entries and (not replace_owned or not _owned_override_stage(target)):
                raise FileExistsError("Refusing to replace a nonempty folder not owned by GhostStudio Override staging.")
            if entries:
                os.replace(target, backup)
                previous_moved = True
            else:
                target.rmdir()
        os.replace(staging, target)
    except Exception:
        if previous_moved and not target.exists():
            os.replace(backup, target)
        raise
    finally:
        if backup.exists() and target.exists():
            shutil.rmtree(backup, ignore_errors=True)


def _load_override_stage(stage_path: str | Path) -> tuple[str, tuple[PackageResource, ...]]:
    stage = Path(stage_path).resolve()
    manifest_path = stage / "override-stage.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("file_type") != OVERRIDE_STAGE_FILE_TYPE or int(payload.get("schema_version", 0)) != 1:
        raise ValueError("This is not a supported GhostStudio Override stage.")
    override = (stage / "Override").resolve()
    if not override.is_dir() or override.is_symlink():
        raise ValueError("Override stage resource folder is missing or unsafe.")
    resources: list[PackageResource] = []
    for item in payload.get("resources", ()):
        resref = _resref(item.get("resref", ""))
        restype = _resource_type(item.get("restype", "")).extension.lower()
        filename = str(item.get("filename", ""))
        if filename.casefold() != f"{resref}.{restype}".casefold() or Path(filename).name != filename:
            raise ValueError("Override stage contains an unsafe or mismatched filename.")
        source = (override / filename).resolve()
        try:
            source.relative_to(override)
        except ValueError as error:
            raise ValueError("Override stage resource escapes its folder.") from error
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Override stage resource is missing or unsafe: {filename}")
        data = source.read_bytes()
        if _sha256(data) != str(item.get("sha256", "")) or len(data) != int(item.get("byte_count", -1)):
            raise ValueError(f"Override stage resource fingerprint failed: {filename}")
        resources.append(PackageResource(resref, restype, data, str(item.get("source_asset_id", ""))))
    return _game_key(payload.get("game", "K2")), _validated_resources(resources)


class NarrativePackagingService:
    """Build verified archives and explicit, rollback-capable test installs."""

    @staticmethod
    def project_resources(project: NarrativeProject, *, include_source: bool = False) -> tuple[PackageResource, ...]:
        return _validated_resources(_project_resources(project, include_source=include_source))

    @staticmethod
    def resources_from_paths(paths: Iterable[str | Path]) -> tuple[PackageResource, ...]:
        """Create typed packer rows for drag/drop or file-picker workflows."""

        return _validated_resources(PackageResource.from_path(path) for path in paths)

    @staticmethod
    def build_archive(
        project_or_resources: NarrativeProject | Sequence[PackageResource],
        output_path: str | Path,
        *,
        archive_type: str | None = None,
        include_source: bool = False,
        overwrite: bool = False,
    ) -> NarrativePackageResult:
        output = Path(output_path).resolve()
        kind = str(archive_type or output.suffix.lstrip(".")).strip().upper()
        resources = (
            NarrativePackagingService.project_resources(project_or_resources, include_source=include_source)
            if isinstance(project_or_resources, NarrativeProject)
            else _validated_resources(project_or_resources)
        )
        global_only = tuple(row for row in resources if row.restype in _GLOBAL_INSTALL_RESOURCE_TYPES)
        if global_only:
            issue = NarrativePackageIssue(
                "blocking",
                "narrative_package.global_resource_blocked",
                "Game-global resources cannot be embedded in a narrative ERF/MOD/SAV. "
                "Use the dedicated backed-up game-root installer instead: "
                + ", ".join(row.filename for row in global_only),
            )
            return NarrativePackageResult(str(output), kind, resources, (issue,))
        if output.suffix and output.suffix.lower() != f".{kind.lower()}":
            issue = NarrativePackageIssue(
                "blocking",
                "narrative_package.extension_mismatch",
                f"{kind} archives must use the .{kind.lower()} extension.",
            )
            return NarrativePackageResult(str(output), kind, resources, (issue,))
        manifest_path = output.with_suffix(output.suffix + ".ghoststudio.json")
        if (output.exists() or manifest_path.exists()) and not overwrite:
            issue = NarrativePackageIssue(
                "blocking",
                "narrative_package.output_exists",
                f"Package output or its GhostStudio manifest already exists: {output}",
            )
            return NarrativePackageResult(str(output), kind, resources, (issue,))
        try:
            payload = _archive_bytes(resources, kind)
            if not _readback_matches(payload, resources, kind):
                raise ValueError("PyKotor archive readback did not reproduce every resource exactly.")
            manifest = {
                "file_type": PACKAGE_MANIFEST_FILE_TYPE,
                "schema_version": 1,
                "created_at": _utc_now(),
                "archive_type": kind,
                "archive_filename": output.name,
                "archive_sha256": _sha256(payload),
                "archive_byte_count": len(payload),
                "engine_proof": "not_recorded",
                "resources": [
                    {
                        "resref": row.resref,
                        "restype": row.restype,
                        "sha256": row.sha256,
                        "byte_count": len(row.data),
                        "source_asset_id": row.source_asset_id,
                    }
                    for row in resources
                ],
            }
            _promote_package_files(
                output,
                payload,
                manifest_path,
                manifest,
                resources=resources,
                archive_type=kind,
            )
        except Exception as error:
            issue = NarrativePackageIssue("blocking", "narrative_package.build_failed", str(error))
            return NarrativePackageResult(str(output), kind, resources, (issue,))
        return NarrativePackageResult(str(output), kind, resources, (), True, str(manifest_path))

    @staticmethod
    def stage_override(
        project_or_resources: NarrativeProject | Sequence[PackageResource],
        output_dir: str | Path,
        *,
        game: str | None = None,
        include_source: bool = False,
        replace_owned: bool = False,
    ) -> OverrideStageResult:
        target = Path(output_dir).resolve()
        project_game = _game_key(
            project_or_resources.game if isinstance(project_or_resources, NarrativeProject) else (game or "K2")
        )
        resources = (
            NarrativePackagingService.project_resources(project_or_resources, include_source=include_source)
            if isinstance(project_or_resources, NarrativeProject)
            else _validated_resources(project_or_resources)
        )
        blocked = tuple(row for row in resources if row.restype in _ARCHIVE_RESOURCE_TYPES)
        if blocked:
            issue = NarrativePackageIssue(
                "blocking",
                "override_stage.archive_resource_blocked",
                "Module/archive files belong in the game's Modules folder, not Override: "
                + ", ".join(row.filename for row in blocked),
            )
            return OverrideStageResult(str(target), project_game, resources, (issue,))
        global_only = tuple(row for row in resources if row.restype in _GLOBAL_INSTALL_RESOURCE_TYPES)
        if global_only:
            issue = NarrativePackageIssue(
                "blocking",
                "override_stage.global_resource_blocked",
                "Game-global resources do not belong in Override. Use the dedicated backed-up "
                "game-root installer instead: "
                + ", ".join(row.filename for row in global_only),
            )
            return OverrideStageResult(str(target), project_game, resources, (issue,))
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f".{target.name}.stage-{uuid4().hex}")
        try:
            override = staging / "Override"
            override.mkdir(parents=True)
            for row in resources:
                (override / row.filename).write_bytes(row.data)
            manifest = {
                "file_type": OVERRIDE_STAGE_FILE_TYPE,
                "schema_version": 1,
                "created_at": _utc_now(),
                "game": str(project_game).upper(),
                "engine_proof": "not_recorded",
                "resources": [
                    {
                        "resref": row.resref,
                        "restype": row.restype,
                        "filename": row.filename,
                        "sha256": row.sha256,
                        "byte_count": len(row.data),
                        "source_asset_id": row.source_asset_id,
                    }
                    for row in resources
                ],
            }
            _atomic_json(staging / "override-stage.json", manifest)
            _load_override_stage(staging)
            _promote_stage(staging, target, replace_owned=replace_owned)
        except Exception as error:
            shutil.rmtree(staging, ignore_errors=True)
            issue = NarrativePackageIssue("blocking", "override_stage.failed", str(error))
            return OverrideStageResult(str(target), project_game, resources, (issue,))
        return OverrideStageResult(
            str(target), project_game, resources, (), True, str(target / "override-stage.json")
        )

    @staticmethod
    def inspect_override_stage(stage_path: str | Path) -> OverrideStageResult:
        try:
            game, resources = _load_override_stage(stage_path)
        except Exception as error:
            issue = NarrativePackageIssue("blocking", "override_stage.invalid", str(error))
            return OverrideStageResult(str(Path(stage_path).resolve()), "", (), (issue,))
        path = Path(stage_path).resolve()
        return OverrideStageResult(str(path), game, resources, (), True, str(path / "override-stage.json"))

    @staticmethod
    def install_override(
        stage_path: str | Path,
        game_root: str | Path,
        *,
        on_conflict: str = "block",
        backup_root: str | Path | None = None,
        expected_existing: dict[str, str] | None = None,
    ) -> OverrideInstallResult:
        """Explicitly install a verified stage, backing up conflicts on request.

        ``backup_root`` (optional) moves the install transaction, conflict
        backups and receipt out of the game root into an existing folder of a
        separate tree; the game root then only gains the Override files.
        ``expected_existing`` (optional) maps filenames to ``"absent"`` or the
        SHA-256 the current Override file must have (compare-and-swap against a
        caller's preview); any mismatch refuses the install before writing.
        """

        root = Path(game_root).resolve()
        policy = str(on_conflict or "block").strip().lower()
        if policy not in {"block", "backup"}:
            raise ValueError("Override conflict policy must be 'block' or 'backup'.")
        external: Path | None = None
        try:
            game, resources = _load_override_stage(stage_path)
            if not root.is_dir() or root.is_symlink():
                raise ValueError(f"Game root is missing or unsafe: {root}")
            if backup_root is not None:
                external = _external_backup_root(backup_root, root)
            expected = {
                str(name).casefold(): str(value or "").strip().lower()
                for name, value in dict(expected_existing or {}).items()
            }
            if any(value != "absent" and not _SHA256_PATTERN.fullmatch(value) for value in expected.values()):
                raise ValueError("expected_existing values must be 'absent' or a SHA-256 hex digest.")
            override = root / "Override"
            if override.exists() and (not override.is_dir() or override.is_symlink()):
                raise ValueError("Game Override path is not a safe directory.")
            conflicts: list[PackageResource] = []
            identical: list[str] = []
            prior_sha256: dict[str, str] = {}
            mismatched: list[str] = []
            for row in resources:
                destination = override / row.filename
                want = expected.get(row.filename.casefold())
                if destination.exists():
                    if not destination.is_file() or destination.is_symlink():
                        raise ValueError(f"Override destination is not a safe file: {destination}")
                    current = destination.read_bytes()
                    prior_sha256[row.filename] = _sha256(current)
                    if want is not None and want != prior_sha256[row.filename]:
                        mismatched.append(row.filename)
                    elif current == row.data:
                        identical.append(row.filename)
                    else:
                        conflicts.append(row)
                elif want not in (None, "absent"):
                    mismatched.append(row.filename)
            if mismatched:
                issue = NarrativePackageIssue(
                    "blocking",
                    "override_install.expected_state_mismatch",
                    "Override files changed since the caller's preview: " + ", ".join(mismatched),
                )
                return OverrideInstallResult(str(root), game, (), (), issues=(issue,))
            if conflicts and policy == "block":
                issue = NarrativePackageIssue(
                    "blocking",
                    "override_install.conflict",
                    "Existing Override files differ; choose the explicit backup policy to replace them: "
                    + ", ".join(row.filename for row in conflicts),
                )
                return OverrideInstallResult(str(root), game, (), tuple(identical), issues=(issue,))
        except Exception as error:
            issue = NarrativePackageIssue("blocking", "override_install.preflight_failed", str(error))
            return OverrideInstallResult(str(root), "", issues=(issue,))

        install_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:10]
        if external is None:
            transaction = root / f".ghoststudio-override-install-{install_id}"
            backup = root / "GhostStudioBackups" / install_id
        else:
            backup = external / install_id
            transaction = backup / ".transaction"
        installed: list[str] = []
        prior_names = {row.filename.casefold() for row in conflicts}
        try:
            incoming = transaction / "incoming"
            incoming.mkdir(parents=True)
            for row in resources:
                if row.filename in identical:
                    continue
                staged = incoming / row.filename
                staged.write_bytes(row.data)
                if _sha256(staged.read_bytes()) != row.sha256:
                    raise ValueError(f"Install transaction copy failed verification: {row.filename}")
            backup_override = backup / "Override"
            backup_override.mkdir(parents=True)
            for row in conflicts:
                source = override / row.filename
                saved = backup_override / row.filename
                shutil.copyfile(source, saved)
                if saved.read_bytes() != source.read_bytes():
                    raise ValueError(f"Could not verify backup for {row.filename}")
                if _sha256(saved.read_bytes()) != prior_sha256[row.filename]:
                    raise ValueError(f"Override destination changed during the install: {row.filename}")
            override.mkdir(parents=True, exist_ok=True)
            for row in resources:
                if row.filename in identical:
                    continue
                if row.filename not in prior_sha256 and (override / row.filename).exists():
                    raise ValueError(f"Override destination appeared during the install: {row.filename}")
                _atomic_write(override / row.filename, (incoming / row.filename).read_bytes())
                installed.append(row.filename)
            receipt = {
                "file_type": "GhostStudioOverrideInstallReceipt",
                "schema_version": 1,
                "install_id": install_id,
                "installed_at": _utc_now(),
                "game": game,
                "game_root": str(root),
                "stage_path": str(Path(stage_path).resolve()),
                "conflict_policy": policy,
                "backup_location": "game_root" if external is None else "external",
                "installed": installed,
                "installed_sha256": {row.filename: row.sha256 for row in resources if row.filename in installed},
                "skipped_identical": identical,
                "backed_up": [row.filename for row in conflicts],
                "backed_up_sha256": {row.filename: prior_sha256[row.filename] for row in conflicts},
            }
            receipt_path = backup / "install-receipt.json"
            _atomic_json(receipt_path, receipt)
        except Exception as error:
            rollback_errors: list[str] = []
            for filename in reversed(installed):
                destination = override / filename
                prior = backup / "Override" / filename
                try:
                    if filename.casefold() in prior_names and prior.is_file():
                        _atomic_write(destination, prior.read_bytes())
                    else:
                        destination.unlink(missing_ok=True)
                except Exception as rollback_error:
                    rollback_errors.append(f"{filename}: {rollback_error}")
            message = f"Override install failed and rollback was attempted: {error}"
            if rollback_errors:
                message += ". Manual recovery is required from the backup folder: " + "; ".join(rollback_errors)
            shutil.rmtree(transaction, ignore_errors=True)
            issue = NarrativePackageIssue("blocking", "override_install.failed", message)
            return OverrideInstallResult(
                str(root), game, tuple(installed), tuple(identical), str(backup), issues=(issue,)
            )
        finally:
            shutil.rmtree(transaction, ignore_errors=True)

        return OverrideInstallResult(
            str(root),
            game,
            tuple(installed),
            tuple(identical),
            str(backup),
            str(backup / "install-receipt.json"),
            (),
            True,
        )

    @staticmethod
    def restore_override_file(
        game_root: str | Path,
        resref: str,
        restype: str,
        *,
        expected_current_sha256: str,
        backup_root: str | Path,
        prior_data: bytes | bytearray | memoryview | None = None,
    ) -> OverrideRestoreResult:
        """Undo one Override install with a compare-and-swap guard.

        The current Override file must still hold exactly the installed bytes
        (``expected_current_sha256``); otherwise nothing is changed. Its bytes
        are first saved (exclusively, verified) under ``backup_root``, a folder
        outside the game root. ``prior_data`` is then written back atomically,
        or, when it is ``None`` (the install created the file), the file is
        removed. Any failure puts the installed bytes back.
        """

        root = Path(game_root).resolve()
        filename = ""
        try:
            resource_type = _resource_type(restype)
            filename = f"{_resref(resref)}.{resource_type.extension.lower()}"
            if resource_type.extension.lower() in _ARCHIVE_RESOURCE_TYPES | _GLOBAL_INSTALL_RESOURCE_TYPES:
                raise ValueError("Archive and game-global resources are not Override files.")
            expected = str(expected_current_sha256 or "").strip().lower()
            if not _SHA256_PATTERN.fullmatch(expected):
                raise ValueError("expected_current_sha256 must be a SHA-256 hex digest.")
            if not root.is_dir() or root.is_symlink():
                raise ValueError(f"Game root is missing or unsafe: {root}")
            external = _external_backup_root(backup_root, root)
            override = root / "Override"
            if not override.is_dir() or override.is_symlink():
                raise ValueError("Game Override folder is missing or unsafe.")
            destination = override / filename
            if not destination.is_file() or destination.is_symlink():
                raise ValueError(f"Override file to restore is missing or unsafe: {filename}")
            current = destination.read_bytes()
            if _sha256(current) != expected:
                raise ValueError(f"Override file changed since it was installed; restore refused: {filename}")
            prior = None if prior_data is None else bytes(prior_data)
        except Exception as error:
            issue = NarrativePackageIssue("blocking", "override_restore.preflight_failed", str(error))
            return OverrideRestoreResult(str(root), filename, issues=(issue,))

        restore_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:10]
        backup = external / f"restore-{restore_id}"
        saved = backup / "Override" / filename
        receipt_path = backup / "restore-receipt.json"
        action = "removed" if prior is None else "restored"
        try:
            saved.parent.mkdir(parents=True, exist_ok=False)
            with saved.open("xb") as handle:
                handle.write(current)
            if saved.read_bytes() != current:
                raise ValueError("Could not verify the pre-restore backup.")
            # Re-check just before mutating; a concurrent change is never clobbered.
            if _sha256(destination.read_bytes()) != expected:
                raise ValueError("Override file changed during the restore.")
            try:
                if prior is None:
                    destination.unlink()
                    if destination.exists():
                        raise ValueError("Override file is still present after removal.")
                else:
                    _atomic_write(destination, prior)
                    if destination.read_bytes() != prior:
                        raise ValueError("Restored Override file failed exact readback.")
                _atomic_json(
                    receipt_path,
                    {
                        "file_type": "GhostStudioOverrideRestoreReceipt",
                        "schema_version": 1,
                        "restore_id": restore_id,
                        "restored_at": _utc_now(),
                        "game_root": str(root),
                        "filename": filename,
                        "action": action,
                        "pre_restore_sha256": expected,
                        "restored_sha256": "" if prior is None else _sha256(prior),
                        "pre_restore_backup": f"Override/{filename}",
                        "engine_proof": "not_recorded",
                    },
                )
            except Exception:
                _atomic_write(destination, current)
                raise
        except Exception as error:
            issue = NarrativePackageIssue(
                "blocking",
                "override_restore.failed",
                f"Override restore failed before changing the file or was rolled back: {error}",
            )
            return OverrideRestoreResult(
                str(root), filename, "", "", expected, str(backup), issues=(issue,)
            )
        return OverrideRestoreResult(
            str(root),
            filename,
            action,
            "" if prior is None else _sha256(prior),
            expected,
            str(backup),
            str(receipt_path),
            (),
            True,
        )

    @staticmethod
    def install_global_tlk(
        tlk_data: bytes | bytearray | memoryview,
        game_root: str | Path,
        *,
        game: str,
    ) -> GlobalTlkInstallResult:
        """Install a validated ``dialog.tlk`` with an exact, permanent backup.

        TLK is game-global and cannot be delivered through Override or a MOD.
        This deliberately separate operation requires an existing game-root
        TLK, validates the replacement before touching disk, and records enough
        information for :meth:`restore_global_tlk` to restore the prior bytes.
        """

        root = Path(game_root).resolve()
        target_game = _game_key(game)
        target = root / "dialog.tlk"
        payload = bytes(tlk_data)
        try:
            from .data_authoring import TalkTableDocument

            if not root.is_dir() or root.is_symlink():
                raise ValueError(f"Game root is missing or unsafe: {root}")
            if not target.is_file() or target.is_symlink():
                raise ValueError("The selected game root does not contain a safe dialog.tlk file.")
            document = TalkTableDocument.load(payload)
            blocking = tuple(row.message for row in document.validate() if row.blocking)
            if blocking:
                raise ValueError("Replacement dialog.tlk is invalid: " + "; ".join(blocking))
        except Exception as error:
            issue = NarrativePackageIssue("blocking", "global_tlk.preflight_failed", str(error))
            return GlobalTlkInstallResult(str(root), target_game, str(target), issues=(issue,))

        install_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:10]
        backup = root / "GhostStudioBackups" / "TalkTable" / install_id
        original_path = backup / "dialog.tlk"
        receipt_path = backup / "install-receipt.json"
        original = target.read_bytes()
        original_sha = _sha256(original)
        replacement_sha = _sha256(payload)
        try:
            backup.mkdir(parents=True, exist_ok=False)
            original_path.write_bytes(original)
            if original_path.read_bytes() != original:
                raise ValueError("Could not verify the original dialog.tlk backup.")
            _atomic_write(target, payload)
            installed = target.read_bytes()
            if installed != payload:
                raise ValueError("Installed dialog.tlk failed exact readback.")
            _atomic_json(
                receipt_path,
                {
                    "file_type": "GhostStudioGlobalTlkInstallReceipt",
                    "schema_version": 1,
                    "install_id": install_id,
                    "installed_at": _utc_now(),
                    "game": target_game,
                    "game_root": str(root),
                    "dialog_path": str(target),
                    "original_sha256": original_sha,
                    "installed_sha256": replacement_sha,
                    "backup_filename": "dialog.tlk",
                    "engine_proof": "not_recorded",
                },
            )
        except Exception as error:
            rollback_error = ""
            try:
                if original_path.is_file():
                    _atomic_write(target, original_path.read_bytes())
            except Exception as restore_error:
                rollback_error = f" Manual recovery is required from {original_path}: {restore_error}"
            issue = NarrativePackageIssue(
                "blocking",
                "global_tlk.install_failed",
                f"Global TLK install failed and rollback was attempted: {error}.{rollback_error}",
            )
            return GlobalTlkInstallResult(
                str(root), target_game, str(target), str(original_path), issues=(issue,)
            )
        return GlobalTlkInstallResult(
            str(root),
            target_game,
            str(target),
            str(original_path),
            str(receipt_path),
            replacement_sha,
            False,
            (),
            True,
        )

    @staticmethod
    def restore_global_tlk(
        receipt_or_folder: str | Path,
        game_root: str | Path,
    ) -> GlobalTlkInstallResult:
        """Restore a prior TLK install receipt without discarding current bytes."""

        root = Path(game_root).resolve()
        receipt_path = Path(receipt_or_folder).resolve()
        if receipt_path.is_dir():
            receipt_path = receipt_path / "install-receipt.json"
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("file_type") != "GhostStudioGlobalTlkInstallReceipt":
                raise ValueError("The selected file is not a GhostStudio global TLK receipt.")
            if Path(str(receipt.get("game_root", ""))).resolve() != root:
                raise ValueError("The TLK receipt belongs to a different game root.")
            target = root / "dialog.tlk"
            backup = receipt_path.parent / str(receipt.get("backup_filename") or "dialog.tlk")
            if not target.is_file() or target.is_symlink() or not backup.is_file() or backup.is_symlink():
                raise ValueError("The live or backed-up dialog.tlk is missing or unsafe.")
            original = backup.read_bytes()
            if _sha256(original) != str(receipt.get("original_sha256") or ""):
                raise ValueError("The backed-up dialog.tlk fingerprint no longer matches its receipt.")
            from .data_authoring import TalkTableDocument

            TalkTableDocument.load(original)
            current = target.read_bytes()
            pre_restore = receipt_path.parent / "pre-restore-dialog.tlk"
            if pre_restore.exists():
                raise FileExistsError("This TLK receipt has already created a pre-restore backup.")
            pre_restore.write_bytes(current)
            if pre_restore.read_bytes() != current:
                raise ValueError("Could not verify the pre-restore dialog.tlk backup.")
            try:
                _atomic_write(target, original)
                if target.read_bytes() != original:
                    raise ValueError("Restored dialog.tlk failed exact readback.")
            except Exception:
                _atomic_write(target, current)
                raise
            restore_receipt = receipt_path.parent / "restore-receipt.json"
            _atomic_json(
                restore_receipt,
                {
                    "file_type": "GhostStudioGlobalTlkRestoreReceipt",
                    "schema_version": 1,
                    "restored_at": _utc_now(),
                    "game": receipt.get("game"),
                    "game_root": str(root),
                    "restored_sha256": _sha256(original),
                    "pre_restore_sha256": _sha256(current),
                },
            )
            return GlobalTlkInstallResult(
                str(root),
                str(receipt.get("game") or ""),
                str(target),
                str(pre_restore),
                str(restore_receipt),
                _sha256(original),
                True,
                (),
                True,
            )
        except Exception as error:
            issue = NarrativePackageIssue("blocking", "global_tlk.restore_failed", str(error))
            return GlobalTlkInstallResult(str(root), "", issues=(issue,))


__all__ = [
    "GlobalTlkInstallResult",
    "NarrativePackageInspection",
    "NarrativePackageIssue",
    "NarrativePackageResult",
    "NarrativePackagingService",
    "OverrideInstallResult",
    "OverrideRestoreResult",
    "OverrideStageResult",
    "PackageResource",
    "inspect_gff_resource",
    "inspect_narrative_archive",
]
