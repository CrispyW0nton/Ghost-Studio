"""Merge-safe gameplay packaging and reversible installation for foreign rigs.

The package format mirrors KOTOR Patch Manager's patch-owned ``additional``
resources, but does not pretend that the current Patch Manager launcher installs
those supplemental files.  Ghost Studio resolves the live ``appearance.2da``
row at staging time, patches the UTC token, previews every Override target, and
records byte-for-byte backups for restoration.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import struct
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from src.core.project.custom_rigged_character_project import (
    CustomRiggedCharacterProject,
    MaterialAssignment,
    sha256_file,
)
from src.core.secondary_motion.profile_io import serialize_secondary_motion_profile
from src.core.secondary_motion.workflow import compile_secondary_motion_profile
from .custom_rigged_character_behavior_service import (
    spawn_test_script_resref,
    spawn_test_script_source,
)


PACKAGE_SCHEMA = "ghostrigger.custom_rigged_character_package.v1"
INSTALL_PLAN_SCHEMA = "ghostrigger.custom_rigged_character_install_plan.v1"
INSTALL_SESSION_SCHEMA = "ghostrigger.custom_rigged_character_install_session.v1"
APPEARANCE_PATCH_SCHEMA = "ghostrigger.merge_safe_2da_row.v1"
SOUNDSET_PATCH_SCHEMA = "ghostrigger.merge_safe_creature_soundset.v1"
ANIMATION_REGISTRY_SCHEMA = "kotor_patch_manager.custom_animation_registry.v1"
APPEARANCE_ROW_TOKEN = "2DAMEMORY_GHOSTRIGGER_CUSTOM_APPEARANCE"
SOUNDSET_ROW_TOKEN = "2DAMEMORY_GHOSTRIGGER_CUSTOM_SOUNDSET"

_RUNTIME_SUFFIXES = {".mdl", ".mdx", ".tga", ".tpc", ".txi", ".utc", ".ncs", ".wav", ".ssf"}
_KNOWN_K2_STEAM_ASPYR = "306f3cf9c45b8d9a086afe10964a3512fc202477d7f8398511b297550990ae51"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary = stream.name
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)


def _safe_name(value: str, *, fallback: str = "custom_character") -> str:
    result = "_".join(
        part for part in "".join(char.lower() if char.isalnum() else " " for char in value).split()
        if part
    )
    return result or fallback


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _twoda_to_binary_v2b(table: Any) -> bytes:
    output = bytearray(b"2DA V2.b\n")
    for column in table.columns:
        output += str(column).encode("ascii") + b"\t"
    output += b"\0"
    output += struct.pack("<I", len(table))
    for index in range(len(table)):
        output += str(index).encode("ascii") + b"\t"
    data = bytearray()
    dedup: dict[str, int] = {}
    offsets: list[int] = []
    for row in range(len(table)):
        for column in table.columns:
            value = str(table.get(row, column) or "")
            offset = dedup.get(value)
            if offset is None:
                offset = len(data)
                dedup[value] = offset
                data += value.encode("cp1252", "replace") + b"\0"
            offsets.append(offset)
    if len(data) >= 65536:
        raise ValueError("The merged 2DA exceeds the V2.b data-block limit.")
    for offset in offsets:
        output += struct.pack("<H", offset)
    output += struct.pack("<H", len(data))
    output += data
    return bytes(output)


def _source_texture(project: CustomRiggedCharacterProject, assignment: MaterialAssignment) -> Path:
    if assignment.source_texture:
        return project.resolve_path(assignment.source_texture)
    folder = project.resolve_path(project.texture_folder) if project.texture_folder else Path()
    if not folder.is_dir():
        return Path()
    stems = [assignment.texture_resref, assignment.material_name]
    extensions = (".tga", ".tpc", ".png", ".dds", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")
    by_name = {path.name.casefold(): path for path in folder.iterdir() if path.is_file()}
    for stem in stems:
        clean = Path(str(stem).replace("\\", "/")).stem
        for extension in extensions:
            candidate = by_name.get(f"{clean}{extension}".casefold())
            if candidate is not None:
                return candidate
    return Path()


def _build_utc_template(
    project: CustomRiggedCharacterProject,
    source: bytes | None = None,
    hook_overrides: Mapping[str, str] | None = None,
) -> bytes:
    from src.formats.gff_reader import read_gff
    from src.formats.gff_types import GffFieldType, GffFile, GffStruct, LocString, ResRef
    from src.formats.gff_writer import write_gff

    if source:
        utc = read_gff(source)
    else:
        utc = GffFile(file_type="UTC ", file_version="V3.2", root=GffStruct())
    root = utc.root

    def set_field(label: str, field_type: Any, value: Any) -> None:
        if label in root.fields:
            current = root.fields[label]
            if current.type == GffFieldType.RESREF:
                current.value = ResRef(str(value))
            elif current.type == GffFieldType.CEXOLOCSTRING:
                if hasattr(current.value, "set_text"):
                    current.value.set_text(str(value))
                else:
                    current.value = LocString(-1, {0: str(value)})
            else:
                try:
                    current.value = type(current.value)(value)
                except Exception:
                    current.value = value
        else:
            root.set(label, field_type, value)

    utc_settings = dict(project.utc_settings)
    gameplay = dict(project.gameplay_settings)
    behavior_profile = dict(project.behavior_profile or {})
    inherit_template_stats = bool(source) and bool(
        behavior_profile.get("inherit_template_combat_stats", True)
    )
    utc_resref = str(utc_settings.get("resref") or project.resource_name)[:16].lower()
    set_field("Tag", GffFieldType.CEXOSTRING, str(utc_settings.get("tag") or utc_resref))
    set_field("TemplateResRef", GffFieldType.RESREF, ResRef(utc_resref))
    set_field(
        "FirstName",
        GffFieldType.CEXOLOCSTRING,
        str(utc_settings.get("display_name") or project.creature_name or project.resource_name),
    )
    set_field("Appearance_Type", GffFieldType.UINT16, 0)
    level = max(1, min(255, int(utc_settings.get("level", 5))))
    if not inherit_template_stats:
        set_field("FactionID", GffFieldType.UINT16, int(utc_settings.get("faction_id", 5)))
        hit_points = max(1, min(32767, int(utc_settings.get("hit_points", 45))))
        for label, field_type, value in (
            ("Gender", GffFieldType.BYTE, int(utc_settings.get("gender", 2))),
            ("Race", GffFieldType.BYTE, int(utc_settings.get("race", 6))),
            ("Class1", GffFieldType.BYTE, int(utc_settings.get("class_id", 0))),
            ("Level", GffFieldType.BYTE, level),
            ("HitPoints", GffFieldType.INT16, hit_points),
            ("CurrentHitPoints", GffFieldType.INT16, hit_points),
            ("MaxHitPoints", GffFieldType.INT16, hit_points),
            ("Str", GffFieldType.BYTE, int(utc_settings.get("strength", 14))),
            ("Dex", GffFieldType.BYTE, int(utc_settings.get("dexterity", 10))),
            ("Con", GffFieldType.BYTE, int(utc_settings.get("constitution", 12))),
            ("Int", GffFieldType.BYTE, int(utc_settings.get("intelligence", 6))),
            ("Wis", GffFieldType.BYTE, int(utc_settings.get("wisdom", 8))),
            ("Cha", GffFieldType.BYTE, int(utc_settings.get("charisma", 6))),
            ("SoundSetFile", GffFieldType.UINT16, int(utc_settings.get("soundset", 0))),
        ):
            set_field(label, field_type, value)
        class_struct = GffStruct(type_id=2)
        class_struct.set("Class", GffFieldType.BYTE, int(utc_settings.get("class_id", 0)))
        class_struct.set("ClassLevel", GffFieldType.BYTE, level)
        root.set("ClassList", GffFieldType.LIST, [class_struct])

    # A fresh UTC gets conservative defaults.  A copied installed template keeps
    # its complete event surface unless the project explicitly overrides a hook.
    if not source:
        script_defaults = {
            "ScriptSpawn": "k_def_spawn01",
            "ScriptDeath": "k_def_death01",
            "ScriptDamaged": "k_def_damage01",
            "ScriptAttacked": "k_def_combat01",
        }
        if str(gameplay.get("behavior_preset") or "").casefold() == "custom_scripts":
            script_defaults = {}
        for label, default in script_defaults.items():
            set_field(label, GffFieldType.RESREF, ResRef(str(utc_settings.get(label) or default)))
    for label, row_value in sorted(dict(behavior_profile.get("script_hooks") or {}).items()):
        row = dict(row_value or {})
        if str(row.get("mode") or "inherit").casefold() not in {"existing", "custom"}:
            continue
        script_resref = str(row.get("resref") or "").strip().lower()
        if script_resref:
            set_field(label, GffFieldType.RESREF, ResRef(script_resref))
    for label, script_resref in sorted(dict(hook_overrides or {}).items()):
        clean = str(script_resref or "").strip().lower()
        if clean:
            set_field(label, GffFieldType.RESREF, ResRef(clean))
    result = write_gff(utc)
    check = read_gff(result)
    if str(check.file_type).strip().upper() != "UTC":
        raise ValueError("Generated UTC template did not reload as a UTC resource.")
    return result


def _patch_utc_runtime_rows(
    value: bytes,
    appearance_row: int,
    soundset_row: int | None = None,
) -> bytes:
    from src.formats.gff_reader import read_gff
    from src.formats.gff_types import GffFieldType
    from src.formats.gff_writer import write_gff

    utc = read_gff(value)
    if "Appearance_Type" in utc.root.fields:
        field = utc.root.fields["Appearance_Type"]
        field.value = type(field.value)(appearance_row)
    else:
        utc.root.set("Appearance_Type", GffFieldType.UINT16, int(appearance_row))
    if soundset_row is not None:
        if "SoundSetFile" in utc.root.fields:
            field = utc.root.fields["SoundSetFile"]
            field.value = type(field.value)(soundset_row)
        else:
            utc.root.set("SoundSetFile", GffFieldType.UINT16, int(soundset_row))
    result = write_gff(utc)
    check = read_gff(result)
    if int(check.root.get("Appearance_Type", -1)) != int(appearance_row):
        raise ValueError("UTC appearance token did not survive GFF reload.")
    if soundset_row is not None and int(check.root.get("SoundSetFile", -1)) != int(soundset_row):
        raise ValueError("UTC soundset token did not survive GFF reload.")
    return result


@dataclass
class CustomRiggedPackageResult:
    ok: bool
    package_directory: str = ""
    report_path: str = ""
    install_plan_path: str = ""
    files: dict[str, str] = field(default_factory=dict)
    hashes: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "package_directory": self.package_directory,
            "report_path": self.report_path,
            "install_plan_path": self.install_plan_path,
            "files": dict(sorted(self.files.items())),
            "hashes": dict(sorted(self.hashes.items())),
            "warnings": list(self.warnings),
            "error": self.error,
        }


@dataclass
class InstallPreview:
    ok: bool
    preview_id: str = ""
    game_directory: str = ""
    executable_path: str = ""
    executable_sha256: str = ""
    appearance_row: int = -1
    soundset_row: int = -1
    files: list[dict[str, Any]] = field(default_factory=list)
    candidate_directory: str = ""
    requires_patch_manager: bool = False
    messages: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "preview_id": self.preview_id,
            "game_directory": self.game_directory,
            "executable_path": self.executable_path,
            "executable_sha256": self.executable_sha256,
            "appearance_row": self.appearance_row,
            "soundset_row": self.soundset_row,
            "files": list(self.files),
            "candidate_directory": self.candidate_directory,
            "requires_patch_manager": self.requires_patch_manager,
            "messages": list(self.messages),
            "error": self.error,
        }


@dataclass
class InstallResult:
    ok: bool
    session_manifest: str = ""
    installed_files: list[str] = field(default_factory=list)
    restored_files: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    error: str = ""


class CustomRiggedCharacterPackagingService:
    """Build a portable package and stage it through a reversible transaction."""

    def __init__(self, *, process_is_running: Callable[[str], bool] | None = None) -> None:
        # Production uses the fail-closed operating-system process check. Tests
        # can inject an isolated probe without weakening that boundary.
        self._process_is_running = process_is_running or self._game_running

    def build_package(
        self,
        project: CustomRiggedCharacterProject,
        model_outputs: Mapping[str, str],
        destination: str | Path,
        *,
        utc_template_bytes: bytes | None = None,
        behavior_resources: Iterable[tuple[str, str, bytes]] = (),
        utc_hook_overrides: Mapping[str, str] | None = None,
        behavior_report: Mapping[str, Any] | None = None,
        allow_overwrite: bool = False,
    ) -> CustomRiggedPackageResult:
        resref = str(project.resource_name or "").strip().lower()
        if not resref:
            return CustomRiggedPackageResult(ok=False, error="A KOTOR resource name is required.")
        package_root = Path(destination)
        if package_root.exists() and any(package_root.iterdir()) and not allow_overwrite:
            ownership_error = self._prepare_owned_package_rebuild(project, package_root, resref)
            if ownership_error:
                return CustomRiggedPackageResult(ok=False, error=ownership_error)
        additional = package_root / "additional"
        additional.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        warnings: list[str] = []
        source_hashes_before = self._source_hashes(project)
        try:
            for kind in ("mdl", "mdx"):
                source_text = str(model_outputs.get(kind) or "")
                source = Path(source_text)
                if not source.is_file():
                    raise FileNotFoundError(f"Built {kind.upper()} not found: {source}")
                target = additional / f"{resref}.{kind}"
                _write_atomic(target, source.read_bytes())
                written[target.relative_to(package_root).as_posix()] = target

            identity = dict(project.secondary_motion.get("model_identity") or {})
            identity.update({
                "model_resrefs": [resref],
                "mdl_sha256": sha256_file(additional / f"{resref}.mdl"),
                "mdx_sha256": sha256_file(additional / f"{resref}.mdx"),
            })
            project.secondary_motion["model_identity"] = identity
            compiled_secondary_motion = compile_secondary_motion_profile(
                project.secondary_motion,
                target_kind="custom_character",
            )
            project.secondary_motion = compiled_secondary_motion
            profile_id = str(compiled_secondary_motion.get("profile_id") or "secondary_motion")
            profile_path = additional / "secondary-motion" / f"{_safe_name(profile_id)}.json"
            _write_atomic(profile_path, serialize_secondary_motion_profile(compiled_secondary_motion))
            profile_relative = profile_path.relative_to(package_root).as_posix()
            written[profile_relative] = profile_path
            requires_cloth_runtime = any(
                str(region.get("backend") or "") == "runtime_pbd"
                for region in compiled_secondary_motion.get("regions") or ()
            )

            report_source = Path(str(model_outputs.get("report") or ""))
            if report_source.is_file():
                target = additional / f"{resref}.model-build-report.json"
                _write_atomic(target, report_source.read_bytes())
                written[target.relative_to(package_root).as_posix()] = target

            conversions = []
            for assignment in project.material_assignments:
                record, paths = self._convert_texture(project, assignment, additional)
                conversions.append(record)
                for path in paths:
                    written[path.relative_to(package_root).as_posix()] = path

            utc_resref = str(project.utc_settings.get("resref") or resref)[:16].lower()
            utc_template = _build_utc_template(project, utc_template_bytes, utc_hook_overrides)
            utc_path = additional / f"{utc_resref}.utc.template"
            _write_atomic(utc_path, utc_template)
            written[utc_path.relative_to(package_root).as_posix()] = utc_path

            for script_resref, script_type, script_data in behavior_resources:
                script_resref = str(script_resref or "").strip().lower()
                script_type = str(script_type or "").strip().lower().lstrip(".")
                if not script_resref or len(script_resref) > 16 or not all(
                    value.isalnum() or value == "_" for value in script_resref
                ):
                    raise ValueError(f"Behavior script has an unsafe KOTOR resource name: {script_resref!r}")
                if script_type not in {"nss", "ncs", "wav"}:
                    raise ValueError(f"Unsupported behavior resource type: {script_type!r}")
                target = additional / f"{script_resref}.{script_type}"
                payload = bytes(script_data or b"")
                if not payload:
                    raise ValueError(f"Behavior resource is empty: {target.name}")
                if target.exists() and target.read_bytes() != payload:
                    raise ValueError(f"Behavior resource output collides with different data: {target.name}")
                _write_atomic(target, payload)
                written[target.relative_to(package_root).as_posix()] = target

            appearance_patch = self._appearance_patch(project)
            appearance_path = additional / "appearance.2da.patch.json"
            _write_atomic(appearance_path, _stable_json(appearance_patch))
            written[appearance_path.relative_to(package_root).as_posix()] = appearance_path

            soundset_patch: dict[str, Any] | None = None
            soundset_path: Path | None = None
            if project.creature_sound_cues:
                soundset_patch = self._soundset_patch(project, behavior_report or {})
                soundset_path = additional / "soundset.2da.patch.json"
                _write_atomic(soundset_path, _stable_json(soundset_patch))
                written[soundset_path.relative_to(package_root).as_posix()] = soundset_path

            registrations = self._animation_registry(project)
            registry_path = additional / f"{resref}.animation-registry.json"
            _write_atomic(registry_path, _stable_json(registrations))
            written[registry_path.relative_to(package_root).as_posix()] = registry_path

            if bool(project.gameplay_settings.get("generate_spawn_script")):
                spawn_resref = spawn_test_script_resref(project)
                spawn_path = additional / f"{spawn_resref}.nss"
                spawn_source = spawn_test_script_source(project).encode("utf-8")
                if spawn_path.exists() and spawn_path.read_bytes() != spawn_source:
                    raise ValueError(f"Test-spawn source collides with different data: {spawn_path.name}")
                _write_atomic(spawn_path, spawn_source)
                written[spawn_path.relative_to(package_root).as_posix()] = spawn_path
                spawn_ncs = additional / f"{spawn_resref}.ncs"
                if not spawn_ncs.is_file():
                    warnings.append(
                        "The optional spawn source is not installed because no compiled NCS was supplied."
                    )

            runtime_files = [
                path.name for path in additional.iterdir()
                if path.is_file() and path.suffix.casefold() in _RUNTIME_SUFFIXES
            ]
            runtime_files.append(f"{utc_resref}.utc")
            runtime_files = sorted(set(runtime_files), key=str.casefold)
            requires_patch = bool(registrations["registrations"])
            supported_hashes = [
                str(value).lower()
                for value in project.metadata.get("supported_executable_hashes", [])
                if str(value).strip()
            ]
            if requires_patch and project.target_game == "K2" and not supported_hashes:
                supported_hashes = [_KNOWN_K2_STEAM_ASPYR]
            install_plan = {
                "schema": INSTALL_PLAN_SCHEMA,
                "package_schema": PACKAGE_SCHEMA,
                "package_id": f"custom-creature-{_safe_name(resref)}",
                "project_id": project.project_id,
                "target_game": project.target_game,
                "model_resref": resref,
                "utc_resref": utc_resref,
                "appearance_patch": appearance_path.name,
                "appearance_row_token": APPEARANCE_ROW_TOKEN,
                "soundset_patch": soundset_path.name if soundset_path is not None else "",
                "soundset_row_token": SOUNDSET_ROW_TOKEN if soundset_path is not None else "",
                "soundset_resref": str(soundset_patch["resref"]) if soundset_patch else "",
                "requires_dialog_tlk_patch": bool(soundset_patch),
                "utc_template": utc_path.name,
                "runtime_files": runtime_files,
                "requires_custom_animation_patch": requires_patch,
                "required_patch_id": "custom-animation-core" if requires_patch else "",
                "secondary_motion_profile": profile_relative,
                "secondary_motion_runtime_dependency": "cloth-dynamics-k2" if requires_cloth_runtime else "",
                "secondary_motion_write_enabled": bool(
                    dict(compiled_secondary_motion.get("runtime") or {}).get("write_enabled")
                ),
                "supported_executable_hashes": sorted(set(supported_hashes)),
                "temporary_module_placement": {
                    "enabled": bool(project.gameplay_settings.get("prepare_module_placement", False)),
                    "module_resref": str(
                        project.gameplay_settings.get("test_module_resref") or "plcaa"
                    ).strip().lower(),
                    "placement_tag": f"gs_{utc_resref}"[:16],
                    "replace_requested_placement": bool(
                        project.gameplay_settings.get("replace_test_placement", False)
                    ),
                    "position": [
                        float(value)
                        for value in dict(
                            project.gameplay_settings.get("test_placement") or {}
                        ).get("position", [26.0, 30.0, 0.0])
                    ],
                    "bearing": float(
                        dict(project.gameplay_settings.get("test_placement") or {}).get(
                            "bearing", 3.1415927
                        )
                    ),
                },
                "source_paths_in_package": False,
            }
            install_plan_path = additional / "install-plan.json"
            _write_atomic(install_plan_path, _stable_json(install_plan))
            written[install_plan_path.relative_to(package_root).as_posix()] = install_plan_path

            manifest = self._patch_manager_manifest(
                project,
                requires_patch,
                supported_hashes,
                requires_cloth_runtime=requires_cloth_runtime,
            )
            manifest_path = package_root / "manifest.toml"
            _write_atomic(manifest_path, manifest.encode("utf-8"))
            written[manifest_path.relative_to(package_root).as_posix()] = manifest_path
            readme_path = package_root / "README.md"
            _write_atomic(readme_path, self._readme(project, install_plan).encode("utf-8"))
            written[readme_path.relative_to(package_root).as_posix()] = readme_path

            source_hashes_after = self._source_hashes(project)
            if source_hashes_before != source_hashes_after:
                raise RuntimeError("A source asset changed while the package was being generated.")
            hashes = {name: sha256_file(path) for name, path in sorted(written.items())}
            package_report = {
                "schema": PACKAGE_SCHEMA,
                "project_schema_version": project.schema_version,
                "project_id": project.project_id,
                "target_game": project.target_game,
                "model_resref": resref,
                "source_hashes": source_hashes_after,
                "output_hashes": hashes,
                "texture_conversions": conversions,
                "appearance_patch": appearance_patch,
                "soundset_patch": soundset_patch,
                "utc": {
                    "resref": utc_resref,
                    "appearance_row": APPEARANCE_ROW_TOKEN,
                    "soundset_row": SOUNDSET_ROW_TOKEN if soundset_patch else None,
                },
                "behavior": dict(behavior_report or {}),
                "animation_registry": registrations,
                "secondary_motion": compiled_secondary_motion,
                "install_plan": install_plan,
                "warnings": warnings,
                "generated_at": _utc_now(),
                "absolute_developer_paths_in_package": False,
            }
            report_path = package_root / f"{resref}.package-report.json"
            _write_atomic(report_path, _stable_json(package_report))
            written[report_path.relative_to(package_root).as_posix()] = report_path
            hashes[report_path.relative_to(package_root).as_posix()] = sha256_file(report_path)
        except Exception as exc:
            return CustomRiggedPackageResult(ok=False, package_directory=str(package_root), error=str(exc))
        return CustomRiggedPackageResult(
            ok=True,
            package_directory=str(package_root),
            report_path=str(report_path),
            install_plan_path=str(install_plan_path),
            files={name: str(path) for name, path in sorted(written.items())},
            hashes=hashes,
            warnings=warnings,
        )

    def _prepare_owned_package_rebuild(
        self,
        project: CustomRiggedCharacterProject,
        package_root: Path,
        resref: str,
    ) -> str:
        """Verify and remove only files recorded by this project's prior report."""

        report_path = package_root / f"{resref}.package-report.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            return "Package rebuild stopped because the existing package ownership report is unreadable."
        if (
            str(report.get("schema") or "") != PACKAGE_SCHEMA
            or str(report.get("project_id") or "") != project.project_id
            or str(report.get("model_resref") or "").casefold() != resref.casefold()
        ):
            return "Package rebuild stopped because the existing package belongs to a different project."
        expected = dict(report.get("output_hashes") or {})
        root = package_root.resolve()
        owned: list[Path] = []
        for relative, wanted_value in expected.items():
            target = (root / str(relative)).resolve()
            if root != target.parent and root not in target.parents:
                return "Package rebuild stopped because the prior report contains an unsafe path."
            wanted = str(wanted_value or "").lower()
            if not target.is_file() or not wanted or sha256_file(target) != wanted:
                return (
                    f"Package rebuild stopped because {relative} changed after the prior build. "
                    "Choose a new output folder so the changed file is preserved."
                )
            owned.append(target)
        for target in owned:
            target.unlink()
        report_path.unlink(missing_ok=True)
        return ""

    def _source_hashes(self, project: CustomRiggedCharacterProject) -> dict[str, str]:
        rows: dict[str, str] = {}
        assets = [project.primary_fbx, *project.external_animation_assets, *project.texture_assets]
        for index, asset in enumerate(assets):
            if not asset.path:
                continue
            path = project.resolve_path(asset.path)
            if path.is_file():
                rows[f"{asset.role}:{index}:{path.name}"] = sha256_file(path)
        for index, assignment in enumerate(project.material_assignments):
            path = _source_texture(project, assignment)
            if path.is_file():
                rows[f"texture_assignment:{index}:{path.name}"] = sha256_file(path)
        for index, cue in enumerate(project.creature_sound_cues):
            if not cue.source_path:
                continue
            path = project.resolve_path(cue.source_path)
            if path.is_file():
                rows[f"creature_sound:{cue.cue}:{index}:{path.name}"] = sha256_file(path)
        return dict(sorted(rows.items()))

    def _convert_texture(
        self,
        project: CustomRiggedCharacterProject,
        assignment: MaterialAssignment,
        destination: Path,
    ) -> tuple[dict[str, Any], list[Path]]:
        source = _source_texture(project, assignment)
        if not source.is_file():
            raise FileNotFoundError(
                f"Texture for material '{assignment.material_name}' was not found. Choose a source texture before building."
            )
        before_hash = sha256_file(source)
        if assignment.source_sha256 and assignment.source_sha256.lower() != before_hash:
            raise ValueError(f"Source texture changed since import: {source.name}")
        assignment.source_sha256 = before_hash
        resref = str(assignment.texture_resref or source.stem).strip().lower()
        if not resref or len(resref) > 16:
            raise ValueError(f"Texture resource name is not KOTOR-safe: {resref!r}")
        output_format = str(assignment.output_format or "TGA").upper()
        authored_txi = str(assignment.txi or "").strip()
        if authored_txi:
            txi_text = authored_txi
            txi_source = "authored"
        elif str(assignment.alpha_mode or "").casefold() == "cutout":
            txi_text = "blending punchthrough\nalphatest 0.5"
            txi_source = "generated_cutout"
        else:
            txi_text = ""
            txi_source = "none"
        outputs: list[Path] = []
        width = height = 0
        has_alpha = False
        vertical_flip_applied = False
        if source.suffix.casefold() == ".tpc" and output_format == "TPC":
            output = destination / f"{resref}.tpc"
            _write_atomic(output, source.read_bytes())
            outputs.append(output)
        else:
            from PIL import Image, ImageOps

            with Image.open(source) as opened:
                opened.load()
                width, height = opened.size
                has_alpha = "A" in opened.getbands() and opened.getchannel("A").getextrema()[0] < 255
                image = opened.convert("RGBA" if has_alpha or assignment.alpha_mode != "opaque" else "RGB")
                if assignment.flip_vertical_for_kotor:
                    image = ImageOps.flip(image)
                    vertical_flip_applied = True
                buffer = BytesIO()
                image.save(buffer, format="TGA", compression=None)
                tga_bytes = buffer.getvalue()
            if output_format == "TGA":
                output = destination / f"{resref}.tga"
                _write_atomic(output, tga_bytes)
                outputs.append(output)
            elif output_format == "TPC":
                from src.converters.mesh_converter import tga_to_tpc

                temporary_tga = destination / f".{resref}.conversion.tga"
                output = destination / f"{resref}.tpc"
                _write_atomic(temporary_tga, tga_bytes)
                try:
                    if not tga_to_tpc(str(temporary_tga), str(output), txi_text, mipmaps=True):
                        raise ValueError(f"Could not convert {source.name} to TPC.")
                finally:
                    temporary_tga.unlink(missing_ok=True)
                outputs.append(output)
            else:
                raise ValueError(f"Unsupported KOTOR texture output format: {output_format}")
        if output_format == "TGA" and txi_text:
            txi_path = destination / f"{resref}.txi"
            _write_atomic(txi_path, (txi_text.rstrip() + "\n").encode("ascii", "strict"))
            outputs.append(txi_path)
        if sha256_file(source) != before_hash:
            raise RuntimeError(f"Source texture changed during conversion: {source}")
        return ({
            "material": assignment.material_name,
            "source_sha256": before_hash,
            "texture_resref": resref,
            "output_format": output_format,
            "output_files": [path.name for path in outputs],
            "dimensions": [width, height],
            "has_alpha": has_alpha,
            "wrap_mode": assignment.wrap_mode,
            "vertical_flip_applied": vertical_flip_applied,
            "txi_preserved": bool(authored_txi),
            "txi_generated": txi_source == "generated_cutout",
            "txi_source": txi_source,
        }, outputs)

    def _appearance_patch(self, project: CustomRiggedCharacterProject) -> dict[str, Any]:
        settings = dict(project.appearance_settings)
        resref = project.resource_name.lower()
        updates = {
            "label": str(settings.get("label") or f"Creature_{project.creature_name or resref}"),
            "race": resref,
            "walkdist": str(settings.get("walkdist", settings.get("walk_distance", "2.0"))),
            "rundist": str(settings.get("rundist", settings.get("run_distance", "4.0"))),
            "perspace": str(settings.get("perspace", settings.get("personal_space", "1.0"))),
            "creperspace": str(settings.get("creperspace", settings.get("collision_space", "1.0"))),
            "cameraspace": str(settings.get("cameraspace", "1.0")),
            "targetheight": str(settings.get("targetheight", "1.0")),
            "hitdist": str(settings.get("hitdist", "1.0")),
            "prefatckdist": str(settings.get("prefatckdist", "1.0")),
        }
        for key, value in dict(settings.get("column_overrides") or {}).items():
            updates[str(key)] = str(value)
        return {
            "schema": APPEARANCE_PATCH_SCHEMA,
            "table": "appearance.2da",
            "operation": "upsert_row",
            "match": {"column": "race", "value": resref, "case_insensitive": True},
            "donor": {"row": int(settings.get("donor_row", 88))},
            "updates": dict(sorted(updates.items())),
            "result_row_token": APPEARANCE_ROW_TOKEN,
            "hardcoded_result_row": False,
        }

    def _soundset_patch(
        self,
        project: CustomRiggedCharacterProject,
        behavior_report: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Describe a native SSF/TLK/2DA soundset without assigning live rows yet."""

        sound_report = dict(behavior_report.get("creature_sounds") or {})
        cue_rows = [dict(value) for value in sound_report.get("cues") or ()]
        by_cue = {str(value.get("cue") or "").casefold(): value for value in cue_rows}
        expected = {str(value.cue or "").casefold() for value in project.creature_sound_cues}
        if not expected or set(by_cue) != expected:
            raise ValueError(
                "Creature sound metadata is incomplete. Run the behavior preflight again before packaging."
            )
        resref = _safe_name(project.resource_name)[:16]
        if not resref or len(resref) > 16:
            raise ValueError("The creature soundset needs a KOTOR-safe resource name.")
        cues: list[dict[str, Any]] = []
        seen_audio: set[str] = set()
        for cue_name in sorted(expected):
            row = by_cue[cue_name]
            audio_resref = str(row.get("resref") or "").strip().lower()
            slots = [str(value or "").strip().upper() for value in row.get("ssf_slots") or ()]
            if not audio_resref or len(audio_resref) > 16 or not slots:
                raise ValueError(f"Creature sound cue '{cue_name}' has incomplete native soundset metadata.")
            if audio_resref in seen_audio:
                raise ValueError(f"Creature sound resource '{audio_resref}' is assigned more than once.")
            seen_audio.add(audio_resref)
            cues.append({
                "cue": cue_name,
                "label": str(row.get("label") or cue_name),
                "resref": audio_resref,
                "ssf_slots": slots,
                "duration_seconds": float(row.get("duration_seconds") or 0.0),
                "sha256": str(row.get("sha256") or "").lower(),
            })
        return {
            "schema": SOUNDSET_PATCH_SCHEMA,
            "table": "soundset.2da",
            "operation": "upsert_native_creature_soundset",
            "match": {"column": "resref", "value": resref, "case_insensitive": True},
            "resref": resref,
            "label": f"GhostStudio_{project.creature_name or resref}",
            "gender": "0",
            "type": "",
            "cues": cues,
            "result_row_token": SOUNDSET_ROW_TOKEN,
            "hardcoded_result_row": False,
            "dialog_tlk_policy": "append_or_reuse_owned_voiceover_entries_with_backup",
            "preserves_utc_event_hooks": True,
        }

    def _animation_registry(self, project: CustomRiggedCharacterProject) -> dict[str, Any]:
        records: dict[tuple[str, int], dict[str, Any]] = {}
        for registration in project.custom_animation_registrations:
            if registration.animation_id is None:
                continue
            key = (registration.name.casefold(), int(registration.animation_id))
            records[key] = {
                "name": registration.name,
                "animation_id": int(registration.animation_id),
                "source_clip": registration.source_clip,
                "namespace": registration.namespace,
                "model_resref": project.resource_name,
                "additive": True,
                "replaces_vanilla": False,
            }
        for mapping in project.animation_mappings:
            if mapping.assignment != "custom_runtime_animation" or not mapping.confirmed or mapping.runtime_id is None:
                continue
            key = (mapping.exported_name.casefold(), int(mapping.runtime_id))
            records.setdefault(key, {
                "name": mapping.exported_name,
                "animation_id": int(mapping.runtime_id),
                "source_clip": mapping.source_name,
                "namespace": project.resource_name,
                "model_resref": project.resource_name,
                "additive": True,
                "replaces_vanilla": False,
            })
        return {
            "schema": ANIMATION_REGISTRY_SCHEMA,
            "target_game": project.target_game,
            "registrations": [records[key] for key in sorted(records)],
            "vanilla_slots_replaced": [],
        }

    def _patch_manager_manifest(
        self,
        project: CustomRiggedCharacterProject,
        requires_patch: bool,
        supported_hashes: Iterable[str],
        *,
        requires_cloth_runtime: bool = False,
    ) -> str:
        patch_id = f"custom-creature-{_safe_name(project.resource_name)}"
        requirements = []
        if requires_patch:
            requirements.append("custom-animation-core")
        if requires_cloth_runtime:
            requirements.append("cloth-dynamics-k2")
        requires_toml = ", ".join(f'"{value}"' for value in requirements)
        lines = [
            "[patch]",
            f'id = "{patch_id}"',
            f'name = "{(project.creature_name or project.resource_name).replace(chr(34), chr(39))} Custom Creature"',
            'version = "1.0.0"',
            'author = "Ghost Studio user"',
            'description = "Merge-safe custom creature resource package generated by Ghost Studio."',
            f"requires = [{requires_toml}]",
            "conflicts = []",
        ]
        hashes = list(supported_hashes)
        if hashes:
            lines.extend(["", "[patch.supported_versions]"])
            for index, value in enumerate(hashes):
                lines.append(f'generated_target_{index + 1} = "{str(value).upper()}"')
        return "\n".join(lines) + "\n"

    def _readme(self, project: CustomRiggedCharacterProject, plan: Mapping[str, Any]) -> str:
        return (
            f"# {project.creature_name or project.resource_name}\n\n"
            "This package was generated by Ghost Studio's Custom Rigged Character Builder. "
            "The source FBX and source textures are not included or modified.\n\n"
            "Use the builder's **Install and test** page to preview and stage these resources. "
            "It resolves the live appearance row, patches the UTC token, verifies the target, "
            "backs up replaced files, and provides a restore action. Do not copy appearance.2da by hand.\n\n"
            f"Model: `{project.resource_name}.mdl/.mdx`\n\n"
            f"Custom Animation Patch required: {'yes' if plan['requires_custom_animation_patch'] else 'no for mapped vanilla locomotion'}\n"
        )

    def _build_temporary_module_candidate(
        self,
        plan: Mapping[str, Any],
        game_root: Path,
        candidate_root: Path,
    ) -> tuple[Path, dict[str, Any]] | None:
        """Build and verify one merge-safe K2 PLCaa creature placement."""

        placement = dict(plan.get("temporary_module_placement") or {})
        if not bool(placement.get("enabled")):
            return None
        if str(plan.get("target_game") or "").upper() != "K2":
            raise ValueError("The no-console DevRoom placement is currently available for KOTOR II only.")

        module_resref = str(placement.get("module_resref") or "plcaa").strip().lower()
        utc_resref = str(plan.get("utc_resref") or "").strip().lower()
        placement_tag = str(placement.get("placement_tag") or f"gs_{utc_resref}").strip().lower()[:16]
        for label, value in (
            ("test module", module_resref),
            ("creature UTC", utc_resref),
            ("temporary placement tag", placement_tag),
        ):
            if not value or len(value) > 16 or not all(char.isalnum() or char == "_" for char in value):
                raise ValueError(f"The {label} has an unsafe KOTOR resource name: {value!r}")
        position = [float(value) for value in placement.get("position", [26.0, 30.0, 0.0])]
        if len(position) != 3:
            raise ValueError("The temporary test placement needs exactly three position values.")
        bearing = float(placement.get("bearing", 3.1415927))
        replace_requested_placement = bool(placement.get("replace_requested_placement", False))

        module_path = game_root / "Modules" / f"{module_resref}.mod"
        if not module_path.is_file():
            raise FileNotFoundError(
                f"The KOTOR II test module was not found: {module_path}. "
                "Turn off temporary DevRoom placement to install without it."
            )

        from src.core.assets.resource_manager import _ErfIndex
        from src.core.modules import module_save_pipeline as module_pipeline
        from src.formats.gff_reader import read_gff
        from src.formats.gff_writer import write_gff

        def field_value(structure: Any, name: str, default: Any = None) -> Any:
            field = structure.fields.get(name)
            return default if field is None else field.value

        def set_placement_field(structure: Any, name: str, value: Any, *, optional: bool = False) -> None:
            field = structure.fields.get(name)
            if field is None:
                if optional:
                    return
                raise ValueError(f"The PLCaa placement template is missing its {name} field.")
            try:
                field.value = type(field.value)(value)
            except Exception:
                field.value = value

        def normalize(value: Any) -> Any:
            if hasattr(value, "fields"):
                return {
                    name: normalize(field.value)
                    for name, field in sorted(value.fields.items())
                }
            if isinstance(value, list):
                return [normalize(item) for item in value]
            if isinstance(value, (bytes, bytearray, memoryview)):
                return bytes(value).hex()
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return str(value)

        source_index = _ErfIndex(str(module_path))
        resource_keys = list(source_index._index.keys())
        if not resource_keys:
            raise ValueError(f"The test module could not be indexed: {module_path}")
        source_payloads: dict[str, bytes] = {}
        for key in resource_keys:
            resref, separator, restype_text = key.rpartition(":")
            if not separator or not restype_text.isdigit():
                raise ValueError(f"The test module contains an invalid resource key: {key}")
            payload = source_index.read(resref, int(restype_text))
            if payload is None:
                raise ValueError(f"Could not read test-module resource: {key}")
            source_payloads[key] = payload

        git_key = f"{module_resref}:2023"
        source_git = source_payloads.get(git_key)
        if not source_git:
            raise ValueError(f"The test module is missing its {module_resref}.git resource.")
        git = read_gff(source_git)
        creature_field = git.root.fields.get("Creature List")
        if creature_field is None or not isinstance(creature_field.value, list):
            raise ValueError("The test module has no editable creature placement list.")
        creatures = creature_field.value

        def is_temporary_fixture(creature: Any) -> bool:
            return str(field_value(creature, "Tag", "")).casefold() == placement_tag.casefold()

        existing_fixtures = [creature for creature in creatures if is_temporary_fixture(creature)]
        requested_position = list(position)
        requested_occupants = []
        if replace_requested_placement:
            for creature in creatures:
                if is_temporary_fixture(creature):
                    continue
                try:
                    delta_x = float(field_value(creature, "XPosition", 0.0)) - requested_position[0]
                    delta_y = float(field_value(creature, "YPosition", 0.0)) - requested_position[1]
                except (TypeError, ValueError):
                    continue
                if delta_x * delta_x + delta_y * delta_y <= 0.01:
                    requested_occupants.append(creature)
            if len(requested_occupants) > 1:
                raise ValueError(
                    "More than one creature occupies the requested DevRoom test spot; "
                    "Ghost Studio will not choose one to replace automatically."
                )
        replaced_requested = requested_occupants[0] if requested_occupants else None
        occupied_positions: list[tuple[float, float]] = []
        for creature in creatures:
            if is_temporary_fixture(creature) or creature is replaced_requested:
                continue
            try:
                occupied_positions.append((
                    float(field_value(creature, "XPosition", 0.0)),
                    float(field_value(creature, "YPosition", 0.0)),
                ))
            except (TypeError, ValueError):
                continue

        def placement_is_open(candidate: list[float]) -> bool:
            return all(
                (candidate[0] - other_x) ** 2 + (candidate[1] - other_y) ** 2 >= 2.25
                for other_x, other_y in occupied_positions
            )

        collision_adjusted = not placement_is_open(position)
        if collision_adjusted:
            for x_offset in (-2.0, 2.0, -4.0, 4.0, -6.0, 6.0, -8.0, 8.0):
                candidate = [requested_position[0] + x_offset, requested_position[1], requested_position[2]]
                if placement_is_open(candidate):
                    position = candidate
                    break
            else:
                raise ValueError(
                    "Ghost Studio could not find an open nearby DevRoom test spot without moving another creature."
                )
        original_other_creatures = [
            normalize(creature)
            for creature in creatures
            if not is_temporary_fixture(creature) and creature is not replaced_requested
        ]
        original_non_creature_fields = {
            name: normalize(field.value)
            for name, field in sorted(git.root.fields.items())
            if name != "Creature List"
        }
        creatures[:] = [
            creature
            for creature in creatures
            if not is_temporary_fixture(creature) and creature is not replaced_requested
        ]
        if replaced_requested is not None:
            fixture = copy.deepcopy(replaced_requested)
        elif existing_fixtures:
            fixture = copy.deepcopy(existing_fixtures[0])
        elif creatures:
            fixture = copy.deepcopy(creatures[0])
        else:
            raise ValueError("The test module has no existing creature placement to use as a safe template.")
        for name, value, optional in (
            ("TemplateResRef", utc_resref, False),
            ("Tag", placement_tag, False),
            ("XPosition", position[0], False),
            ("YPosition", position[1], False),
            ("ZPosition", position[2], False),
            ("XOrientation", bearing, False),
            # Stock and previously staged creature placements may encode their
            # facing with Bearing/XOrientation only. Do not reject or mutate an
            # unrelated module merely because the clone omits YOrientation.
            ("YOrientation", 0.0, True),
            ("Bearing", bearing, True),
        ):
            set_placement_field(fixture, name, value, optional=optional)
        creatures.append(fixture)
        new_git = write_gff(git)

        id_to_extension = {value: key for key, value in module_pipeline.RESTYPE_IDS.items()}
        entries = []
        for key in resource_keys:
            resref, _, restype_text = key.rpartition(":")
            restype = int(restype_text)
            extension = id_to_extension.get(restype)
            if extension is None:
                raise ValueError(f"Unsupported test-module resource type {restype} for {resref}.")
            entries.append(module_pipeline.ModuleArchiveEntry(
                resref=resref,
                restype=extension,
                data=new_git if key == git_key else source_payloads[key],
                archive_role=module_pipeline._archive_role(extension),
                source=str(module_path),
                changed=(key == git_key),
                serializer="custom_character_temporary_placement",
                warning="",
            ))
        candidate_bytes = module_pipeline.build_erf_v1_archive(entries, archive_type="MOD")
        candidate_path = candidate_root / ".module" / f"{module_resref}.mod"
        _write_atomic(candidate_path, candidate_bytes)

        output_index = _ErfIndex(str(candidate_path))
        if set(output_index._index) != set(resource_keys):
            raise RuntimeError("The test module resource set changed while placing the custom creature.")
        for key in resource_keys:
            if key == git_key:
                continue
            resref, _, restype_text = key.rpartition(":")
            if output_index.read(resref, int(restype_text)) != source_payloads[key]:
                raise RuntimeError(f"A non-placement test-module resource changed: {key}")
        verified_git_bytes = output_index.read(module_resref, 2023)
        if not verified_git_bytes:
            raise RuntimeError("The rebuilt test module lost its GIT resource.")
        verified_git = read_gff(verified_git_bytes)
        verified_creatures = verified_git.root.fields["Creature List"].value
        fixtures = [creature for creature in verified_creatures if is_temporary_fixture(creature)]
        if len(fixtures) != 1:
            raise RuntimeError(f"Expected one temporary custom-creature placement, found {len(fixtures)}.")
        if str(field_value(fixtures[0], "TemplateResRef", "")).casefold() != utc_resref.casefold():
            raise RuntimeError("The temporary creature placement does not reference the generated UTC.")
        verified_other_creatures = [
            normalize(creature) for creature in verified_creatures if not is_temporary_fixture(creature)
        ]
        if verified_other_creatures != original_other_creatures:
            raise RuntimeError("An unrelated test-module creature placement changed.")
        verified_non_creature_fields = {
            name: normalize(field.value)
            for name, field in sorted(verified_git.root.fields.items())
            if name != "Creature List"
        }
        if verified_non_creature_fields != original_non_creature_fields:
            raise RuntimeError("An unrelated test-module GIT field changed.")

        report = {
            "schema": "ghostrigger.custom_character_temporary_module_placement.v1",
            "module_resref": module_resref,
            "source_path": str(module_path),
            "source_sha256": sha256_file(module_path),
            "candidate_sha256": sha256_file(candidate_path),
            "resource_count": len(resource_keys),
            "byte_preserved_non_git_resources": len(resource_keys) - 1,
            "preserved_other_creature_placements": len(original_other_creatures),
            "replaced_prior_temporary_placements": len(existing_fixtures),
            "replace_requested_placement": replace_requested_placement,
            "replaced_requested_placement": (
                {
                    "template_resref": str(field_value(replaced_requested, "TemplateResRef", "")),
                    "tag": str(field_value(replaced_requested, "Tag", "")),
                    "position": requested_position,
                }
                if replaced_requested is not None
                else None
            ),
            "placement_tag": placement_tag,
            "utc_resref": utc_resref,
            "requested_position": requested_position,
            "position": position,
            "bearing": bearing,
            "collision_adjusted": collision_adjusted,
            "verified_fixture_count": len(fixtures),
        }
        report_path = candidate_root / ".evidence" / f"{module_resref}.module-placement-report.json"
        _write_atomic(report_path, _stable_json(report))
        return candidate_path, report

    def preview_install(self, package_directory: str | Path, game_directory: str | Path) -> InstallPreview:
        package_root = Path(package_directory).resolve()
        game_root = Path(game_directory).resolve()
        try:
            plan = json.loads((package_root / "additional" / "install-plan.json").read_text(encoding="utf-8"))
            if plan.get("schema") != INSTALL_PLAN_SCHEMA:
                raise ValueError("Package install plan schema is not supported.")
            exe_name = "swkotor2.exe" if plan.get("target_game") == "K2" else "swkotor.exe"
            exe_path = game_root / exe_name
            if not exe_path.is_file():
                raise FileNotFoundError(f"Expected game executable was not found: {exe_path}")
            exe_hash = sha256_file(exe_path)
            supported = {str(value).lower() for value in plan.get("supported_executable_hashes") or ()}
            requires_patch = bool(plan.get("requires_custom_animation_patch"))
            if requires_patch and exe_hash.lower() not in supported:
                raise ValueError(
                    "This executable fingerprint is not supported by the required Custom Animation Patch. "
                    "The fingerprint check was not bypassed."
                )
            if requires_patch:
                config = game_root / "patch_config.toml"
                if not config.is_file() or str(plan.get("required_patch_id")) not in config.read_text(encoding="utf-8", errors="ignore"):
                    raise ValueError(
                        "Install the verified Custom Animation Patch with KOTOR Patch Manager before staging additive actions."
                    )
            package_report_path = package_root / f"{plan['model_resref']}.package-report.json"
            package_report_hash = sha256_file(package_report_path)
            candidate_root = package_root / ".install-preview" / _sha256_bytes(
                f"{game_root}|{exe_hash}|{package_report_hash}".encode("utf-8")
            )[:16]
            candidate_root.mkdir(parents=True, exist_ok=True)
            appearance_blob, appearance_row = self._merge_live_appearance(package_root, game_root)
            _write_atomic(candidate_root / "appearance.2da", appearance_blob)
            soundset_row = -1
            soundset_resref = ""
            dialog_candidate: Path | None = None
            soundset_report: dict[str, Any] | None = None
            if str(plan.get("soundset_patch") or ""):
                (
                    soundset_blob,
                    soundset_row,
                    soundset_resref,
                    ssf_blob,
                    dialog_blob,
                    soundset_report,
                ) = self._merge_live_soundset(package_root, game_root, str(plan["soundset_patch"]))
                _write_atomic(candidate_root / "soundset.2da", soundset_blob)
                _write_atomic(candidate_root / f"{soundset_resref}.ssf", ssf_blob)
                dialog_candidate = candidate_root / ".global" / "dialog.tlk"
                _write_atomic(dialog_candidate, dialog_blob)
            utc_template = package_root / "additional" / str(plan["utc_template"])
            utc_name = f"{plan['utc_resref']}.utc"
            _write_atomic(
                candidate_root / utc_name,
                _patch_utc_runtime_rows(
                    utc_template.read_bytes(),
                    appearance_row,
                    soundset_row if soundset_row >= 0 else None,
                ),
            )
            for name in plan.get("runtime_files") or ():
                name = Path(str(name)).name
                if name.casefold() in {
                    "appearance.2da",
                    "soundset.2da",
                    utc_name.casefold(),
                    f"{soundset_resref}.ssf".casefold(),
                }:
                    continue
                source = package_root / "additional" / name
                if source.is_file() and source.suffix.casefold() in _RUNTIME_SUFFIXES:
                    _write_atomic(candidate_root / name, source.read_bytes())
            module_candidate = self._build_temporary_module_candidate(plan, game_root, candidate_root)
            override = game_root / "Override"
            files = []
            for candidate in sorted(candidate_root.iterdir(), key=lambda value: value.name.casefold()):
                if not candidate.is_file():
                    continue
                target = override / candidate.name
                status = "new"
                current_hash = ""
                if target.is_file():
                    current_hash = sha256_file(target)
                    status = "unchanged" if current_hash == sha256_file(candidate) else "replace_with_backup"
                files.append({
                    "name": candidate.name,
                    "target": str(target),
                    "status": status,
                    "action": "write",
                    "candidate_relative": candidate.name,
                    "current_sha256": current_hash,
                    "candidate_sha256": sha256_file(candidate),
                    "size": candidate.stat().st_size,
                })
            if dialog_candidate is not None:
                dialog_target = game_root / "dialog.tlk"
                if not dialog_target.is_file():
                    raise FileNotFoundError(f"Expected KOTOR talk table was not found: {dialog_target}")
                current_hash = sha256_file(dialog_target)
                candidate_hash = sha256_file(dialog_candidate)
                files.append({
                    "name": "game-root/dialog.tlk",
                    "target": str(dialog_target),
                    "status": "unchanged" if current_hash == candidate_hash else "replace_with_backup",
                    "action": "write",
                    "candidate_relative": dialog_candidate.relative_to(candidate_root).as_posix(),
                    "current_sha256": current_hash,
                    "candidate_sha256": candidate_hash,
                    "size": dialog_candidate.stat().st_size,
                    "soundset": soundset_report,
                })
            if module_candidate is not None:
                module_path, module_report = module_candidate
                module_target = game_root / "Modules" / module_path.name
                module_current_hash = sha256_file(module_target)
                module_candidate_hash = sha256_file(module_path)
                files.append({
                    "name": f"Modules/{module_path.name}",
                    "target": str(module_target),
                    "status": (
                        "unchanged"
                        if module_current_hash == module_candidate_hash
                        else "replace_with_backup"
                    ),
                    "action": "write",
                    "candidate_relative": module_path.relative_to(candidate_root).as_posix(),
                    "current_sha256": module_current_hash,
                    "candidate_sha256": module_candidate_hash,
                    "size": module_path.stat().st_size,
                    "module_placement": module_report,
                })
                currentgame_target = game_root / "currentgame" / module_path.name
                if currentgame_target.is_file():
                    files.append({
                        "name": f"currentgame/{module_path.name}",
                        "target": str(currentgame_target),
                        "status": "remove_with_backup",
                        "action": "remove",
                        "candidate_relative": "",
                        "current_sha256": sha256_file(currentgame_target),
                        "candidate_sha256": "",
                        "size": 0,
                    })
            files.sort(key=lambda item: str(item["name"]).casefold())
            identity_payload = {
                "game": str(game_root), "exe": exe_hash, "row": appearance_row,
                "soundset_row": soundset_row,
                "files": files, "package_report": package_report_hash,
            }
            preview_id = _sha256_bytes(_stable_json(identity_payload))
            return InstallPreview(
                ok=True,
                preview_id=preview_id,
                game_directory=str(game_root),
                executable_path=str(exe_path),
                executable_sha256=exe_hash,
                appearance_row=appearance_row,
                soundset_row=soundset_row,
                files=files,
                candidate_directory=str(candidate_root),
                requires_patch_manager=requires_patch,
                messages=[
                    "No files have been installed. Confirm this exact preview to continue.",
                    *(
                        [
                            "The custom creature uses KOTOR's native SSF soundset. Preview appends or reuses only "
                            "its owned dialog.tlk voice rows, merges one soundset.2da row, and includes both in Backup and Restore."
                        ]
                        if dialog_candidate is not None
                        else []
                    ),
                    *(
                        [
                            "PLCaa DevRoom will receive one temporary custom-creature placement; "
                            "all unrelated module resources and placements were verified unchanged."
                        ]
                        if module_candidate is not None
                        else []
                    ),
                    *(
                        [
                            "The requested DevRoom spot was occupied, so the temporary creature was "
                            f"moved to the nearest open test spot at "
                            f"({module_candidate[1]['position'][0]:.1f}, {module_candidate[1]['position'][1]:.1f})."
                        ]
                        if module_candidate is not None
                        and bool(module_candidate[1].get("collision_adjusted"))
                        else []
                    ),
                    *(
                        [
                            "The confirmed replace-at-test-spot option will replace "
                            f"{module_candidate[1]['replaced_requested_placement']['template_resref'] or 'the existing creature'} "
                            "at the requested coordinates; all other placements were verified unchanged."
                        ]
                        if module_candidate is not None
                        and module_candidate[1].get("replaced_requested_placement")
                        else []
                    ),
                ],
            )
        except Exception as exc:
            return InstallPreview(ok=False, game_directory=str(game_root), error=str(exc))

    def _merge_live_appearance(self, package_root: Path, game_root: Path) -> tuple[bytes, int]:
        from src.core.templates.twoda import TwoDA

        patch = json.loads((package_root / "additional" / "appearance.2da.patch.json").read_text(encoding="utf-8"))
        override = game_root / "Override" / "appearance.2da"
        if override.is_file():
            blob = override.read_bytes()
        else:
            from pykotor.extract.installation import Installation
            from pykotor.resource.type import ResourceType

            resource = Installation(game_root).resource("appearance", ResourceType.TwoDA)
            if resource is None:
                raise FileNotFoundError("Could not load the game's appearance.2da.")
            blob = bytes(resource.data)
        table = TwoDA.from_bytes(blob)
        match = patch["match"]
        column = str(match["column"])
        value = str(match["value"]).casefold()
        matches = [index for index in range(len(table)) if str(table.get(index, column) or "").casefold() == value]
        if len(matches) > 1:
            raise ValueError(f"Live appearance.2da contains duplicate rows for {value}: {matches}")
        if matches:
            row = matches[0]
        else:
            donor = int(patch["donor"]["row"])
            if donor < 0 or donor >= len(table):
                raise ValueError(f"The selected donor appearance row {donor} does not exist in the live table.")
            row = len(table)
            table._rows.append(list(table._rows[donor]))
            table._labels.append(str(row))
        for update_column, update_value in dict(patch["updates"]).items():
            if update_column in table.columns:
                table._rows[row][table.col_index(update_column)] = str(update_value)
        result = _twoda_to_binary_v2b(table)
        check = TwoDA.from_bytes(result)
        if str(check.get(row, column) or "").casefold() != value:
            raise ValueError("Merged appearance row did not survive the 2DA round trip.")
        return result, row

    def _merge_live_soundset(
        self,
        package_root: Path,
        game_root: Path,
        patch_name: str,
    ) -> tuple[bytes, int, str, bytes, bytes, dict[str, Any]]:
        """Merge one SSF row and its owned TLK voice entries without touching UTC hooks."""

        from src.core.scripting.data_authoring import SoundSetDocument, TalkTableDocument
        from src.core.templates.twoda import TwoDA

        patch_path = package_root / "additional" / Path(patch_name).name
        patch = json.loads(patch_path.read_text(encoding="utf-8"))
        if patch.get("schema") != SOUNDSET_PATCH_SCHEMA:
            raise ValueError("Creature soundset patch schema is not supported.")
        soundset_resref = str(patch.get("resref") or "").strip().lower()
        if not soundset_resref or len(soundset_resref) > 16:
            raise ValueError("Creature soundset has an unsafe KOTOR resource name.")

        override = game_root / "Override" / "soundset.2da"
        if override.is_file():
            table_blob = override.read_bytes()
        else:
            from pykotor.extract.installation import Installation
            from pykotor.resource.type import ResourceType

            resource = Installation(game_root).resource("soundset", ResourceType.TwoDA)
            if resource is None:
                raise FileNotFoundError("Could not load the game's soundset.2da.")
            table_blob = bytes(resource.data)
        table = TwoDA.from_bytes(table_blob)
        if "resref" not in table.columns:
            raise ValueError("The live soundset.2da has no resref column.")
        matches = [
            index
            for index in range(len(table))
            if str(table.get(index, "resref") or "").casefold() == soundset_resref.casefold()
        ]
        if len(matches) > 1:
            raise ValueError(f"Live soundset.2da contains duplicate rows for {soundset_resref}: {matches}")
        if matches:
            soundset_row = matches[0]
        else:
            soundset_row = len(table)
            table._rows.append(["" for _column in table.columns])
            table._labels.append(str(soundset_row))
        updates = {
            "label": str(patch.get("label") or f"GhostStudio_{soundset_resref}"),
            "resref": soundset_resref,
            "strref": "",
            "gender": str(patch.get("gender") or "0"),
            "type": str(patch.get("type") or ""),
        }
        for column, value in updates.items():
            if column in table.columns:
                table._rows[soundset_row][table.col_index(column)] = value
        soundset_blob = _twoda_to_binary_v2b(table)
        table_check = TwoDA.from_bytes(soundset_blob)
        if str(table_check.get(soundset_row, "resref") or "").casefold() != soundset_resref:
            raise ValueError("Merged creature soundset row did not survive the 2DA round trip.")

        dialog_path = game_root / "dialog.tlk"
        if not dialog_path.is_file():
            raise FileNotFoundError(f"Expected KOTOR talk table was not found: {dialog_path}")
        dialog = TalkTableDocument.load(dialog_path)
        soundset = SoundSetDocument()
        valid_slots = set(SoundSetDocument.slot_names())
        occupied_slots: set[str] = set()
        tlk_rows: list[dict[str, Any]] = []
        for cue in [dict(value) for value in patch.get("cues") or ()]:
            cue_name = str(cue.get("cue") or "").strip().lower()
            audio_resref = str(cue.get("resref") or "").strip().lower()
            duration = max(0.0, float(cue.get("duration_seconds") or 0.0))
            wanted_text = f"DO NOT TRANSLATE - Ghost Studio {soundset_resref} {cue_name}"
            wav_path = package_root / "additional" / f"{audio_resref}.wav"
            if not wav_path.is_file():
                raise FileNotFoundError(f"Creature sound WAV is missing from the package: {wav_path.name}")
            expected_hash = str(cue.get("sha256") or "").lower()
            if expected_hash and sha256_file(wav_path) != expected_hash:
                raise ValueError(f"Creature sound WAV changed after build: {wav_path.name}")
            matching_entries = [
                entry
                for entry in dialog.entries
                if str(entry.voiceover or "").casefold() == audio_resref.casefold()
            ]
            if len(matching_entries) > 1:
                raise ValueError(f"dialog.tlk contains duplicate voiceover rows for {audio_resref}.")
            if matching_entries:
                current = matching_entries[0]
                if current.text != wanted_text:
                    raise ValueError(
                        f"dialog.tlk already uses voiceover {audio_resref} for unrelated text; choose another sound resource name."
                    )
                strref = int(current.strref)
                dialog.update_entry(strref, sound_length=duration)
                action = "reuse_owned_entry"
            else:
                strref = dialog.add_entry(
                    wanted_text,
                    voiceover=audio_resref,
                    sound_length=duration,
                )
                action = "append_owned_entry"
            slots = [str(value or "").strip().upper() for value in cue.get("ssf_slots") or ()]
            if not slots:
                raise ValueError(f"Creature sound cue '{cue_name}' has no native SSF slot.")
            for slot in slots:
                if slot not in valid_slots:
                    raise ValueError(f"Creature sound cue '{cue_name}' uses unknown SSF slot '{slot}'.")
                if slot in occupied_slots:
                    raise ValueError(f"Native SSF slot '{slot}' is assigned more than once.")
                occupied_slots.add(slot)
                soundset.set_slot(slot, strref)
            tlk_rows.append({
                "cue": cue_name,
                "resref": audio_resref,
                "strref": strref,
                "slots": slots,
                "action": action,
                "duration_seconds": duration,
            })

        dialog_blob = dialog.to_bytes()
        dialog_check = TalkTableDocument.load(dialog_blob)
        for row in tlk_rows:
            entry = dialog_check.entry(int(row["strref"]))
            if str(entry.voiceover).casefold() != str(row["resref"]).casefold():
                raise ValueError("Generated dialog.tlk voiceover mapping did not survive readback.")
        ssf_blob = soundset.to_bytes()
        ssf_check = SoundSetDocument.load(ssf_blob)
        for row in tlk_rows:
            for slot in row["slots"]:
                if ssf_check.get_slot(slot) != int(row["strref"]):
                    raise ValueError("Generated SSF mapping did not survive readback.")
        report = {
            "schema": SOUNDSET_PATCH_SCHEMA,
            "soundset_row": soundset_row,
            "soundset_resref": soundset_resref,
            "dialog_tlk_entries": tlk_rows,
            "direct_utc_event_hooks_preserved": True,
            "dialog_tlk_source_sha256": sha256_file(dialog_path),
            "dialog_tlk_candidate_sha256": _sha256_bytes(dialog_blob),
            "ssf_sha256": _sha256_bytes(ssf_blob),
        }
        return soundset_blob, soundset_row, soundset_resref, ssf_blob, dialog_blob, report

    def install(self, preview: InstallPreview, *, confirmed_preview_id: str) -> InstallResult:
        if not preview.ok or not preview.preview_id or confirmed_preview_id != preview.preview_id:
            return InstallResult(ok=False, error="The exact install preview was not confirmed.")
        game_root = Path(preview.game_directory)
        exe_path = Path(preview.executable_path)
        if not exe_path.is_file() or sha256_file(exe_path) != preview.executable_sha256:
            return InstallResult(ok=False, error="The game executable changed after preview; preview again.")
        if self._process_is_running(exe_path.name):
            return InstallResult(ok=False, error=f"Close {exe_path.name} before installation.")
        candidate_root = Path(preview.candidate_directory)
        for item in preview.files:
            action = str(item.get("action") or "write")
            if action == "write":
                candidate = candidate_root / str(item.get("candidate_relative") or item["name"])
                if not candidate.is_file() or sha256_file(candidate) != item["candidate_sha256"]:
                    return InstallResult(ok=False, error=f"Install candidate changed after preview: {item['name']}")
            elif action != "remove":
                return InstallResult(ok=False, error=f"Unsupported install action: {action}")
            target = Path(item["target"])
            current = sha256_file(target) if target.is_file() else ""
            if current != item["current_sha256"]:
                return InstallResult(ok=False, error=f"Target file changed after preview: {item['name']}")
        session = candidate_root.parent / "install-sessions" / datetime.now().strftime("%Y%m%dT%H%M%S")
        before = session / "before"
        before.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        installed: list[str] = []
        try:
            for item in preview.files:
                target = Path(item["target"])
                backup = before / item["name"]
                existed = target.is_file()
                if existed:
                    _write_atomic(backup, target.read_bytes())
                records.append({
                    "name": item["name"], "target": str(target), "existed": existed,
                    "action": str(item.get("action") or "write"),
                    "before_sha256": item["current_sha256"], "backup": str(backup) if existed else "",
                    "installed_sha256": item["candidate_sha256"],
                })
            for item in preview.files:
                target = Path(item["target"])
                action = str(item.get("action") or "write")
                if action == "remove":
                    target.unlink(missing_ok=True)
                    if target.exists():
                        raise RuntimeError(f"Could not clear the cached test module: {item['name']}")
                else:
                    candidate = candidate_root / str(item.get("candidate_relative") or item["name"])
                    _write_atomic(target, candidate.read_bytes())
                    if sha256_file(target) != item["candidate_sha256"]:
                        raise RuntimeError(f"Installed hash mismatch: {item['name']}")
                installed.append(str(target))
            manifest = {
                "schema": INSTALL_SESSION_SCHEMA,
                "installed_at": _utc_now(),
                "game_directory": str(game_root),
                "executable_path": str(exe_path),
                "executable_sha256": preview.executable_sha256,
                "appearance_row": preview.appearance_row,
                "soundset_row": preview.soundset_row,
                "files": records,
            }
            manifest_path = session / "install-session.json"
            _write_atomic(manifest_path, _stable_json(manifest))
            return InstallResult(ok=True, session_manifest=str(manifest_path), installed_files=installed)
        except Exception as exc:
            for record in reversed(records):
                target = Path(record["target"])
                if record["existed"] and Path(record["backup"]).is_file():
                    _write_atomic(target, Path(record["backup"]).read_bytes())
                elif not record["existed"] and target.is_file():
                    target.unlink()
            return InstallResult(ok=False, installed_files=installed, error=f"Install rolled back: {exc}")

    def restore(self, session_manifest: str | Path) -> InstallResult:
        manifest_path = Path(session_manifest)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema") != INSTALL_SESSION_SCHEMA:
                raise ValueError("Install session schema is not supported.")
            exe_path = Path(manifest["executable_path"])
            if self._process_is_running(exe_path.name):
                raise RuntimeError(f"Close {exe_path.name} before restoring files.")
            restored: list[str] = []
            for record in manifest["files"]:
                target = Path(record["target"])
                current_hash = sha256_file(target) if target.is_file() else ""
                if current_hash != record["installed_sha256"]:
                    raise RuntimeError(
                        f"'{target.name}' changed after this install. Restore stopped instead of overwriting newer work."
                    )
            for record in reversed(manifest["files"]):
                target = Path(record["target"])
                if record["existed"]:
                    backup = Path(record["backup"])
                    if not backup.is_file() or sha256_file(backup) != record["before_sha256"]:
                        raise RuntimeError(f"Backup is missing or damaged: {backup}")
                    _write_atomic(target, backup.read_bytes())
                elif target.is_file():
                    target.unlink()
                restored.append(str(target))
            restore_report = manifest_path.with_name("restore-report.json")
            _write_atomic(restore_report, _stable_json({
                "schema": INSTALL_SESSION_SCHEMA, "restored_at": _utc_now(), "files": restored,
            }))
            return InstallResult(ok=True, restored_files=restored, session_manifest=str(manifest_path))
        except Exception as exc:
            return InstallResult(ok=False, session_manifest=str(manifest_path), error=str(exc))

    @staticmethod
    def _game_running(executable_name: str) -> bool:
        try:
            import psutil

            return any(
                str(process.info.get("name") or "").casefold() == executable_name.casefold()
                for process in psutil.process_iter(["name"])
            )
        except ImportError:
            pass
        except Exception:
            # Fall through to the OS-owned process snapshot. Embedded builds do
            # not necessarily include psutil, and a damaged optional dependency
            # must not weaken the install boundary.
            pass

        if os.name == "nt":
            try:
                import ctypes
                from ctypes import wintypes

                create_snapshot = ctypes.windll.kernel32.CreateToolhelp32Snapshot
                process_first = ctypes.windll.kernel32.Process32FirstW
                process_next = ctypes.windll.kernel32.Process32NextW
                close_handle = ctypes.windll.kernel32.CloseHandle

                class PROCESSENTRY32W(ctypes.Structure):
                    _fields_ = [
                        ("dwSize", wintypes.DWORD),
                        ("cntUsage", wintypes.DWORD),
                        ("th32ProcessID", wintypes.DWORD),
                        ("th32DefaultHeapID", ctypes.c_size_t),
                        ("th32ModuleID", wintypes.DWORD),
                        ("cntThreads", wintypes.DWORD),
                        ("th32ParentProcessID", wintypes.DWORD),
                        ("pcPriClassBase", ctypes.c_long),
                        ("dwFlags", wintypes.DWORD),
                        ("szExeFile", wintypes.WCHAR * 260),
                    ]

                create_snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
                create_snapshot.restype = wintypes.HANDLE
                process_first.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
                process_first.restype = wintypes.BOOL
                process_next.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
                process_next.restype = wintypes.BOOL
                close_handle.argtypes = [wintypes.HANDLE]
                close_handle.restype = wintypes.BOOL
                snapshot = create_snapshot(0x00000002, 0)
                if snapshot == ctypes.c_void_p(-1).value:
                    raise OSError("CreateToolhelp32Snapshot failed")
                try:
                    entry = PROCESSENTRY32W()
                    entry.dwSize = ctypes.sizeof(entry)
                    if not process_first(snapshot, ctypes.byref(entry)):
                        raise OSError("Process32FirstW failed")
                    wanted = executable_name.casefold()
                    while True:
                        if str(entry.szExeFile).casefold() == wanted:
                            return True
                        if not process_next(snapshot, ctypes.byref(entry)):
                            return False
                finally:
                    close_handle(snapshot)
            except Exception as exc:
                # Installation and restore are destructive boundaries. If
                # process inspection is unavailable, never guess that the game
                # is closed.
                raise RuntimeError("Ghost Studio could not verify that the game is closed.") from exc

        raise RuntimeError("Ghost Studio could not verify that the game is closed.")


__all__ = [
    "ANIMATION_REGISTRY_SCHEMA",
    "APPEARANCE_PATCH_SCHEMA",
    "APPEARANCE_ROW_TOKEN",
    "CustomRiggedCharacterPackagingService",
    "CustomRiggedPackageResult",
    "INSTALL_PLAN_SCHEMA",
    "INSTALL_SESSION_SCHEMA",
    "InstallPreview",
    "InstallResult",
    "PACKAGE_SCHEMA",
    "SOUNDSET_PATCH_SCHEMA",
    "SOUNDSET_ROW_TOKEN",
]
