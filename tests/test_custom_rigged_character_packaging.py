from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps

from src.core.characters.custom_rigged_character_packaging_service import (
    APPEARANCE_ROW_TOKEN,
    CustomRiggedCharacterPackagingService,
)
from src.core.characters.creature_package_builder import CreatureSpec, _generate_utc
from src.core.project.custom_rigged_character_project import (
    CustomRiggedCharacterProject,
    MaterialAssignment,
    SourceAsset,
)
from src.core.templates.twoda import TwoDA
from src.formats.gff_reader import read_gff
from src.formats.gff_types import GffFieldType, GffFile, GffStruct, ResRef
from src.formats.gff_writer import write_gff


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_blank_creature_utc_uses_kotor_faction_word(tmp_path: Path) -> None:
    utc_path = _generate_utc(
        tmp_path,
        CreatureSpec(resref="c_hostile", display_name="Hostile Creature", faction_id=1),
        1,
    )

    utc = read_gff(utc_path.read_bytes())
    assert utc.root.fields["FactionID"].type is GffFieldType.UINT16
    assert int(utc.root.get("FactionID", -1)) == 1


def _appearance_bytes() -> bytes:
    return (
        "2DA V2.0\n\n"
        "          label             race              walkdist          rundist           perspace          creperspace       cameraspace       targetheight      hitdist           prefatckdist\n"
        "0         Creature_Donor    c_donor           1.0               2.0               0.5               0.5               0.5               0.5               0.5               0.5\n"
    ).encode("ascii")


def _plcaa_module_bytes() -> bytes:
    from src.core.modules import module_save_pipeline as module_pipeline

    creature = GffStruct(type_id=4)
    creature.set("TemplateResRef", GffFieldType.RESREF, ResRef("old_creature"))
    creature.set("Tag", GffFieldType.CEXOSTRING, "old_creature")
    creature.set("XPosition", GffFieldType.FLOAT, 26.0)
    creature.set("YPosition", GffFieldType.FLOAT, 30.0)
    creature.set("ZPosition", GffFieldType.FLOAT, 0.0)
    creature.set("XOrientation", GffFieldType.FLOAT, 0.0)
    creature.set("Bearing", GffFieldType.FLOAT, 0.0)
    root = GffStruct()
    root.set("Creature List", GffFieldType.LIST, [creature])
    root.set("LegacyField", GffFieldType.CEXOSTRING, "preserve-this")
    git = write_gff(GffFile(file_type="GIT ", file_version="V3.2", root=root))
    return module_pipeline.build_erf_v1_archive(
        [
            module_pipeline.ModuleArchiveEntry("plcaa", "git", git),
            module_pipeline.ModuleArchiveEntry("keep_script", "nss", b"void main() {}\n"),
        ],
        archive_type="MOD",
    )


def _project(tmp_path: Path) -> tuple[CustomRiggedCharacterProject, Path, Path]:
    source = tmp_path / "foreign.fbx"
    source.write_bytes(b"read-only-fbx-evidence")
    texture = tmp_path / "foreign_diffuse.png"
    image = Image.new("RGBA", (2, 2))
    image.putdata((
        (255, 0, 0, 255), (0, 255, 0, 255),
        (0, 0, 255, 128), (255, 255, 0, 0),
    ))
    image.save(texture)
    project = CustomRiggedCharacterProject(
        creature_name="Foreign Creature",
        resource_name="c_foreign",
        primary_fbx=SourceAsset(str(source), _sha(source), "primary_fbx", True),
        output_project_folder=str(tmp_path),
        build_destination=str(tmp_path / "build"),
        material_assignments=[MaterialAssignment(
            material_name="body",
            texture_resref="foreign_tex",
            source_texture=str(texture),
            source_sha256=_sha(texture),
            output_format="TGA",
            wrap_mode="repeat",
            alpha_mode="blend",
            txi="blending punchthrough\n",
        )],
        appearance_settings={"donor_row": 0, "label": "Creature_Foreign"},
        utc_settings={"resref": "c_foreign", "faction_id": 5, "hit_points": 60},
    )
    return project, source, texture


def test_package_is_portable_merge_safe_and_preserves_sources(tmp_path: Path) -> None:
    project, source, texture = _project(tmp_path)
    model_root = tmp_path / "model"
    model_root.mkdir()
    mdl = model_root / "c_foreign.mdl"
    mdx = model_root / "c_foreign.mdx"
    report = model_root / "c_foreign.build-report.json"
    mdl.write_bytes(b"mdl-output")
    mdx.write_bytes(b"mdx-output")
    report.write_text('{"ok": true}\n', encoding="utf-8")
    before = {source: _sha(source), texture: _sha(texture)}

    result = CustomRiggedCharacterPackagingService(process_is_running=lambda _name: False).build_package(
        project,
        {"mdl": str(mdl), "mdx": str(mdx), "report": str(report)},
        tmp_path / "package",
    )

    assert result.ok, result.error
    assert {path: _sha(path) for path in before} == before
    package = Path(result.package_directory)
    patch = json.loads((package / "additional" / "appearance.2da.patch.json").read_text(encoding="utf-8"))
    assert patch["operation"] == "upsert_row"
    assert patch["result_row_token"] == APPEARANCE_ROW_TOKEN
    assert patch["hardcoded_result_row"] is False
    assert (package / "additional" / "c_foreign.utc.template").is_file()
    assert (package / "additional" / "foreign_tex.tga").is_file()
    assert (package / "additional" / "foreign_tex.txi").is_file()
    with Image.open(texture) as source_image, Image.open(package / "additional" / "foreign_tex.tga") as output_image:
        assert output_image.convert("RGBA").tobytes() == ImageOps.flip(source_image.convert("RGBA")).tobytes()
    plan = json.loads((package / "additional" / "install-plan.json").read_text(encoding="utf-8"))
    assert plan["source_paths_in_package"] is False
    assert plan["requires_custom_animation_patch"] is False
    profile_paths = list((package / "additional" / "secondary-motion").glob("*.json"))
    assert len(profile_paths) == 1
    profile = json.loads(profile_paths[0].read_text(encoding="utf-8"))
    assert profile["schema"] == "ghostrigger.secondary_motion_profile.v1"
    assert profile["model_identity"]["target_kind"] == "custom_character"
    assert profile["runtime"]["write_enabled"] is False
    assert plan["secondary_motion_profile"] == profile_paths[0].relative_to(package).as_posix()
    assert plan["secondary_motion_write_enabled"] is False
    package_report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    assert package_report["absolute_developer_paths_in_package"] is False
    assert package_report["secondary_motion"]["profile_id"] == profile["profile_id"]
    assert package_report["secondary_motion"]["runtime"]["state"] == "capture_only"
    assert str(tmp_path).lower() not in Path(result.report_path).read_text(encoding="utf-8").lower()

    rebuilt = CustomRiggedCharacterPackagingService(process_is_running=lambda _name: False).build_package(
        project,
        {"mdl": str(mdl), "mdx": str(mdx), "report": str(report)},
        package,
    )
    assert rebuilt.ok, rebuilt.error
    (package / "additional" / "c_foreign.mdl").write_bytes(b"user-modified")
    refused = CustomRiggedCharacterPackagingService(process_is_running=lambda _name: False).build_package(
        project,
        {"mdl": str(mdl), "mdx": str(mdx), "report": str(report)},
        package,
    )
    assert not refused.ok
    assert "changed after the prior build" in refused.error


def test_texture_orientation_can_be_disabled_and_cutout_txi_is_generated(tmp_path: Path) -> None:
    project, _source, texture = _project(tmp_path)
    assignment = project.material_assignments[0]
    assignment.flip_vertical_for_kotor = False
    assignment.alpha_mode = "cutout"
    assignment.txi = ""
    destination = tmp_path / "converted"
    destination.mkdir()

    report, outputs = CustomRiggedCharacterPackagingService(
        process_is_running=lambda _name: False
    )._convert_texture(project, assignment, destination)

    with Image.open(texture) as source_image, Image.open(destination / "foreign_tex.tga") as output_image:
        assert output_image.convert("RGBA").tobytes() == source_image.convert("RGBA").tobytes()
    assert (destination / "foreign_tex.txi").read_text(encoding="ascii") == (
        "blending punchthrough\nalphatest 0.5\n"
    )
    assert {path.name for path in outputs} == {"foreign_tex.tga", "foreign_tex.txi"}
    assert report["vertical_flip_applied"] is False
    assert report["txi_source"] == "generated_cutout"


def test_preview_install_merge_backup_and_restore_are_reversible(tmp_path: Path) -> None:
    project, source, texture = _project(tmp_path)
    model_root = tmp_path / "model"
    model_root.mkdir()
    mdl = model_root / "c_foreign.mdl"
    mdx = model_root / "c_foreign.mdx"
    mdl.write_bytes(b"mdl-output")
    mdx.write_bytes(b"mdx-output")
    service = CustomRiggedCharacterPackagingService(process_is_running=lambda _name: False)
    package = service.build_package(
        project, {"mdl": str(mdl), "mdx": str(mdx)}, tmp_path / "package"
    )
    assert package.ok, package.error

    game = tmp_path / "KOTOR2"
    override = game / "Override"
    override.mkdir(parents=True)
    (game / "swkotor2.exe").write_bytes(b"fake-test-executable")
    appearance = override / "appearance.2da"
    appearance.write_bytes(_appearance_bytes())
    unrelated = override / "unrelated_mod.tga"
    unrelated.write_bytes(b"keep-me")
    before_appearance = appearance.read_bytes()
    before_unrelated = unrelated.read_bytes()

    preview = service.preview_install(package.package_directory, game)
    assert preview.ok, preview.error
    assert preview.appearance_row == 1
    assert any(row["name"] == "appearance.2da" for row in preview.files)
    assert all(row["name"] != unrelated.name for row in preview.files)
    assert appearance.read_bytes() == before_appearance

    installed = service.install(preview, confirmed_preview_id=preview.preview_id)
    assert installed.ok, installed.error
    table = TwoDA.from_bytes(appearance.read_bytes())
    assert table.get(1, "race") == "c_foreign"
    utc = read_gff((override / "c_foreign.utc").read_bytes())
    assert int(utc.root.get("Appearance_Type", -1)) == 1
    assert unrelated.read_bytes() == before_unrelated
    assert source.read_bytes() == b"read-only-fbx-evidence"

    restored = service.restore(installed.session_manifest)
    assert restored.ok, restored.error
    assert appearance.read_bytes() == before_appearance
    assert not (override / "c_foreign.mdl").exists()
    assert not (override / "c_foreign.mdx").exists()
    assert not (override / "c_foreign.utc").exists()
    assert unrelated.read_bytes() == before_unrelated


def test_install_refuses_unconfirmed_or_changed_preview(tmp_path: Path) -> None:
    project, _source, _texture = _project(tmp_path)
    model_root = tmp_path / "model"
    model_root.mkdir()
    mdl = model_root / "c_foreign.mdl"
    mdx = model_root / "c_foreign.mdx"
    mdl.write_bytes(b"mdl-output")
    mdx.write_bytes(b"mdx-output")
    service = CustomRiggedCharacterPackagingService(process_is_running=lambda _name: False)
    package = service.build_package(project, {"mdl": str(mdl), "mdx": str(mdx)}, tmp_path / "package")
    assert package.ok
    game = tmp_path / "KOTOR2"
    (game / "Override").mkdir(parents=True)
    (game / "swkotor2.exe").write_bytes(b"fake-test-executable")
    (game / "Override" / "appearance.2da").write_bytes(_appearance_bytes())
    preview = service.preview_install(package.package_directory, game)
    assert preview.ok
    assert not service.install(preview, confirmed_preview_id="wrong").ok
    (game / "Override" / "appearance.2da").write_bytes(_appearance_bytes() + b"\n")
    changed = service.install(preview, confirmed_preview_id=preview.preview_id)
    assert not changed.ok
    assert "changed after preview" in changed.error


def test_preview_is_read_only_while_game_runs_but_install_stays_blocked(tmp_path: Path) -> None:
    project, _source, _texture = _project(tmp_path)
    model_root = tmp_path / "model"
    model_root.mkdir()
    mdl = model_root / "c_foreign.mdl"
    mdx = model_root / "c_foreign.mdx"
    mdl.write_bytes(b"mdl-output")
    mdx.write_bytes(b"mdx-output")

    build_service = CustomRiggedCharacterPackagingService(process_is_running=lambda _name: False)
    package = build_service.build_package(
        project,
        {"mdl": str(mdl), "mdx": str(mdx)},
        tmp_path / "package",
    )
    assert package.ok, package.error

    game = tmp_path / "KOTOR2"
    override = game / "Override"
    override.mkdir(parents=True)
    (game / "swkotor2.exe").write_bytes(b"fake-test-executable")
    appearance = override / "appearance.2da"
    appearance.write_bytes(_appearance_bytes())
    before = appearance.read_bytes()

    running_service = CustomRiggedCharacterPackagingService(process_is_running=lambda _name: True)
    preview = running_service.preview_install(package.package_directory, game)
    assert preview.ok, preview.error
    assert appearance.read_bytes() == before

    installed = running_service.install(preview, confirmed_preview_id=preview.preview_id)
    assert not installed.ok
    assert "Close swkotor2.exe before installation" in installed.error
    assert appearance.read_bytes() == before
    assert not (override / "c_foreign.utc").exists()


def test_temporary_devroom_placement_is_merge_safe_and_restores_module_cache(tmp_path: Path) -> None:
    from src.core.assets.resource_manager import _ErfIndex

    project, _source, _texture = _project(tmp_path)
    project.gameplay_settings.update({
        "prepare_module_placement": True,
        "test_module_resref": "plcaa",
        "test_placement": {"position": [26.0, 30.0, 0.0], "bearing": 3.1415927},
    })
    model_root = tmp_path / "model"
    model_root.mkdir()
    mdl = model_root / "c_foreign.mdl"
    mdx = model_root / "c_foreign.mdx"
    mdl.write_bytes(b"mdl-output")
    mdx.write_bytes(b"mdx-output")
    service = CustomRiggedCharacterPackagingService(process_is_running=lambda _name: False)
    package = service.build_package(project, {"mdl": str(mdl), "mdx": str(mdx)}, tmp_path / "package")
    assert package.ok, package.error
    plan = json.loads(Path(package.install_plan_path).read_text(encoding="utf-8"))
    assert plan["temporary_module_placement"]["enabled"] is True
    assert plan["temporary_module_placement"]["module_resref"] == "plcaa"
    assert plan["temporary_module_placement"]["placement_tag"] == "gs_c_foreign"

    game = tmp_path / "KOTOR2"
    override = game / "Override"
    modules = game / "Modules"
    currentgame = game / "currentgame"
    override.mkdir(parents=True)
    modules.mkdir()
    currentgame.mkdir()
    (game / "swkotor2.exe").write_bytes(b"fake-test-executable")
    (override / "appearance.2da").write_bytes(_appearance_bytes())
    module = modules / "plcaa.mod"
    module.write_bytes(_plcaa_module_bytes())
    cache = currentgame / "plcaa.mod"
    cache.write_bytes(b"cached-before-install")
    before_module = module.read_bytes()
    before_cache = cache.read_bytes()

    preview = service.preview_install(package.package_directory, game)
    assert preview.ok, preview.error
    module_row = next(row for row in preview.files if row["name"] == "Modules/plcaa.mod")
    cache_row = next(row for row in preview.files if row["name"] == "currentgame/plcaa.mod")
    assert module_row["action"] == "write"
    assert module_row["status"] == "replace_with_backup"
    assert module_row["module_placement"]["byte_preserved_non_git_resources"] == 1
    assert module_row["module_placement"]["preserved_other_creature_placements"] == 1
    assert module_row["module_placement"]["collision_adjusted"] is True
    assert module_row["module_placement"]["requested_position"] == [26.0, 30.0, 0.0]
    assert module_row["module_placement"]["position"] == [24.0, 30.0, 0.0]
    assert cache_row["action"] == "remove"
    assert module.read_bytes() == before_module
    assert cache.read_bytes() == before_cache

    installed = service.install(preview, confirmed_preview_id=preview.preview_id)
    assert installed.ok, installed.error
    assert not cache.exists()
    index = _ErfIndex(str(module))
    assert index.read("keep_script", 2009) == b"void main() {}\n"
    git = read_gff(index.read("plcaa", 2023))
    creatures = git.root.fields["Creature List"].value
    assert len(creatures) == 2
    assert str(creatures[0].get("TemplateResRef")) == "old_creature"
    fixture = next(creature for creature in creatures if str(creature.get("Tag")) == "gs_c_foreign")
    assert str(fixture.get("TemplateResRef")) == "c_foreign"
    assert float(fixture.get("XPosition")) == 24.0
    assert git.root.get("LegacyField") == "preserve-this"

    restored = service.restore(installed.session_manifest)
    assert restored.ok, restored.error
    assert module.read_bytes() == before_module
    assert cache.read_bytes() == before_cache

    project.gameplay_settings["replace_test_placement"] = True
    replacement_package = service.build_package(
        project,
        {"mdl": str(mdl), "mdx": str(mdx)},
        tmp_path / "package-replace",
    )
    assert replacement_package.ok, replacement_package.error
    replacement_plan = json.loads(Path(replacement_package.install_plan_path).read_text(encoding="utf-8"))
    assert replacement_plan["temporary_module_placement"]["replace_requested_placement"] is True

    replacement_preview = service.preview_install(replacement_package.package_directory, game)
    assert replacement_preview.ok, replacement_preview.error
    replacement_row = next(
        row for row in replacement_preview.files if row["name"] == "Modules/plcaa.mod"
    )
    replacement_report = replacement_row["module_placement"]
    assert replacement_report["collision_adjusted"] is False
    assert replacement_report["position"] == [26.0, 30.0, 0.0]
    assert replacement_report["preserved_other_creature_placements"] == 0
    assert replacement_report["replaced_requested_placement"]["template_resref"] == "old_creature"

    replacement_install = service.install(
        replacement_preview,
        confirmed_preview_id=replacement_preview.preview_id,
    )
    assert replacement_install.ok, replacement_install.error
    replacement_index = _ErfIndex(str(module))
    replacement_git = read_gff(replacement_index.read("plcaa", 2023))
    replacement_creatures = replacement_git.root.fields["Creature List"].value
    assert len(replacement_creatures) == 1
    assert str(replacement_creatures[0].get("TemplateResRef")) == "c_foreign"
    assert str(replacement_creatures[0].get("Tag")) == "gs_c_foreign"
    assert float(replacement_creatures[0].get("XPosition")) == 26.0

    replacement_restore = service.restore(replacement_install.session_manifest)
    assert replacement_restore.ok, replacement_restore.error
    assert module.read_bytes() == before_module
    assert cache.read_bytes() == before_cache
