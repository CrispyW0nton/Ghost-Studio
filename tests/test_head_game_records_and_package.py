"""Focused game-record, package, install, and restore contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from pykotor.resource.formats.twoda import bytes_2da, read_2da
from pykotor.resource.formats.twoda.twoda_data import TwoDA

from src.io.head_builder_package import (
    HEAD_INSTALL_SESSION_SCHEMA,
    HeadPackageInstaller,
    build_head_package,
)
from src.io.head_game_records import (
    HeadGameRecordError,
    HeadGameRecordPatch,
    merge_head_game_records,
)
from src.io.head_texture_asset import (
    build_head_texture_output_policy,
    inspect_head_texture,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _table(headers: list[str], rows: list[dict[str, str]]) -> bytes:
    table = TwoDA(headers)
    for index, row in enumerate(rows):
        table.add_row(str(index), dict(row))
    return bytes(bytes_2da(table))


def _heads() -> bytes:
    return _table(
        [
            "head",
            "alttexture",
            "headtexvvve",
            "headtexvve",
            "headtexve",
            "headtexe",
            "headtexg",
            "headtexvg",
        ],
        [
            {
                "head": "PFHA04",
                "alttexture": "PFHA04",
                "headtexvvve": "PFHA04D2",
                "headtexvve": "PFHA04D1",
                "headtexve": "PFHA04D1",
                "headtexe": "PFHA04D1",
            },
            {"head": "P_OTHER", "alttexture": "P_OTHER"},
        ],
    )


def _heads_k1() -> bytes:
    return _table(
        [
            "head",
            "headtexvvve",
            "headtexvve",
            "headtexve",
            "headtexe",
            "headtexg",
            "headtexvg",
        ],
        [
            {
                "head": "PMHA04",
                "headtexvvve": "PMHA04D",
                "headtexvve": "PMHA04D3",
                "headtexve": "PMHA04D2",
                "headtexe": "PMHA04D1",
            }
        ],
    )


def _appearance() -> bytes:
    return _table(
        [
            "label",
            "normalhead",
            "modela",
            "texa",
            "portrait",
            "race",
        ],
        [
            {
                "label": "P_FEM_A_SML_04",
                "normalhead": "0",
                "modela": "PFBAM",
                "texa": "PFBAL",
                "portrait": "",
                "race": "",
            },
            {
                "label": "Other",
                "normalhead": "1",
                "modela": "P_OTHER",
                "texa": "P_OTHER",
                "portrait": "po_other",
                "race": "P_OTHER",
            },
        ],
    )


def _portraits() -> bytes:
    return _table(
        [
            "baseresref",
            "sex",
            "appearancenumber",
            "appearance_s",
            "appearance_l",
            "forpc",
            "baseresrefe",
            "baseresrefve",
            "baseresrefvve",
            "baseresrefvvve",
        ],
        [
            {
                "baseresref": "po_pfha04",
                "sex": "1",
                "appearancenumber": "0",
                "appearance_s": "0",
                "appearance_l": "0",
                "forpc": "1",
                "baseresrefe": "po_pfha04d1",
                "baseresrefve": "po_pfha04d2",
                "baseresrefvve": "po_pfha04d3",
                "baseresrefvvve": "po_pfha04d",
            }
        ],
    )


def _patch(*, portrait: bool = False) -> HeadGameRecordPatch:
    return HeadGameRecordPatch(
        game="K2",
        output_head_resref="P_CDH01",
        texture_resref="P_CDH01",
        donor_head_resref="PFHA04",
        body_resref="PFBAM",
        appearance_donor_label="P_FEM_A_SML_04",
        appearance_label="GhostStudio_P_CDH01",
        portrait_resref="po_cdh01" if portrait else "",
        portrait_donor_resref="po_pfha04" if portrait else "",
    )


def _binary_export(tmp_path: Path):
    mdl = tmp_path / "built" / "P_CDH01.mdl"
    mdx = tmp_path / "built" / "P_CDH01.mdx"
    mdl.parent.mkdir()
    mdl.write_bytes(b"MDL verified fixture")
    mdx.write_bytes(b"MDX verified fixture")
    inspection = SimpleNamespace(
        mdl_sha256=_sha(mdl.read_bytes()),
        mdx_sha256=_sha(mdx.read_bytes()),
        to_dict=lambda: {
            "mdl_sha256": _sha(mdl.read_bytes()),
            "mdx_sha256": _sha(mdx.read_bytes()),
        },
    )
    return SimpleNamespace(
        mdl_path=str(mdl),
        mdx_path=str(mdx),
        inspection=inspection,
    )


def _texture(tmp_path: Path):
    path = tmp_path / "source" / "checker.tga"
    path.parent.mkdir()
    Image.new("RGBA", (4, 4), (20, 100, 180, 255)).save(path)
    asset = inspect_head_texture(path)
    policy = build_head_texture_output_policy(
        asset,
        output_resref="P_CDH01",
        output_format="TGA",
        txi_delivery="sidecar",
        clamp_s=True,
        clamp_t=True,
    )
    return asset, policy


def _build(tmp_path: Path, *, portrait: bool = False):
    asset, policy = _texture(tmp_path)
    result = build_head_package(
        project_id="project-head-test",
        display_name="Test Custom Head",
        binary_export=_binary_export(tmp_path),
        texture_asset=asset,
        texture_policy=policy,
        game_record_patch=_patch(portrait=portrait),
        reference_heads_bytes=_heads(),
        reference_appearance_bytes=_appearance(),
        reference_portraits_bytes=(
            _portraits() if portrait else None
        ),
        destination=tmp_path / "package",
    )
    assert result.ok, result.error
    return result


def _game_tree(tmp_path: Path) -> Path:
    root = tmp_path / "game"
    override = root / "Override"
    override.mkdir(parents=True)
    (root / "swkotor2.exe").write_bytes(b"unaltered game exe")
    (override / "heads.2da").write_bytes(_heads())
    (override / "appearance.2da").write_bytes(_appearance())
    return root


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_dynamic_merge_is_idempotent_and_never_overwrites_existing_rows() -> None:
    heads = _heads()
    appearance = _appearance()
    result = merge_head_game_records(
        _patch(),
        heads_bytes=heads,
        appearance_bytes=appearance,
    )
    assert result.accepted
    assert result.heads_row == 2
    assert result.appearance_row == 2
    assert result.report["rows"]["heads"]["action"] == "append"
    assert result.report["rows"]["appearance"]["action"] == "clone_append"
    assert result.report["no_existing_row_modified"] is True

    heads_check = read_2da(result.heads_bytes)
    appearance_check = read_2da(result.appearance_bytes)
    assert heads_check.get_cell(0, "head") == "PFHA04"
    assert heads_check.get_cell(1, "head") == "P_OTHER"
    assert heads_check.get_cell(2, "head") == "P_CDH01"
    assert appearance_check.get_cell(0, "texa") == "PFBAL"
    assert appearance_check.get_cell(1, "label") == "Other"
    assert appearance_check.get_cell(2, "normalhead") == "2"

    rebuilt = merge_head_game_records(
        _patch(),
        heads_bytes=result.heads_bytes,
        appearance_bytes=result.appearance_bytes,
    )
    assert rebuilt.report["rows"]["heads"]["action"] == "reuse_exact"
    assert rebuilt.report["rows"]["appearance"]["action"] == "reuse_exact"
    assert rebuilt.heads_bytes == result.heads_bytes
    assert rebuilt.appearance_bytes == result.appearance_bytes

    conflicting_heads = read_2da(result.heads_bytes)
    conflicting_heads.set_cell(result.heads_row, "alttexture", "OTHER_MOD")
    with pytest.raises(HeadGameRecordError, match="refusing to overwrite"):
        merge_head_game_records(
            _patch(),
            heads_bytes=bytes(bytes_2da(conflicting_heads)),
            appearance_bytes=result.appearance_bytes,
        )


def test_optional_portrait_links_dynamic_appearance_row() -> None:
    result = merge_head_game_records(
        _patch(portrait=True),
        heads_bytes=_heads(),
        appearance_bytes=_appearance(),
        portraits_bytes=_portraits(),
    )
    assert result.accepted
    assert result.portraits_row == 1
    portraits = read_2da(result.portraits_bytes)
    assert portraits.get_cell(1, "baseresref") == "po_cdh01"
    assert portraits.get_cell(1, "appearancenumber") == "2"
    assert portraits.get_cell(1, "appearance_s") == "2"
    assert portraits.get_cell(1, "appearance_l") == "2"


def test_optional_portrait_alignment_columns_are_explicitly_overridden() -> None:
    base = _patch(portrait=True)
    patch = replace(
        base,
        portrait_texture_columns={
            "baseresrefe": "po_cdh01d1",
            "baseresrefve": "po_cdh01d2",
            "baseresrefvve": "po_cdh01d3",
            "baseresrefvvve": "po_cdh01d",
        },
    )
    result = merge_head_game_records(
        patch,
        heads_bytes=_heads(),
        appearance_bytes=_appearance(),
        portraits_bytes=_portraits(),
    )
    portraits = read_2da(result.portraits_bytes)
    assert portraits.get_cell(result.portraits_row, "baseresrefe") == "po_cdh01d1"
    assert portraits.get_cell(result.portraits_row, "baseresrefve") == "po_cdh01d2"
    assert portraits.get_cell(result.portraits_row, "baseresrefvve") == "po_cdh01d3"
    assert portraits.get_cell(result.portraits_row, "baseresrefvvve") == "po_cdh01d"


def test_k1_portrait_merge_adds_requested_light_side_columns() -> None:
    patch = HeadGameRecordPatch(
        game="K1",
        output_head_resref="P_KNEOH",
        texture_resref="P_KNEOH",
        donor_head_resref="PMHA04",
        body_resref="PMBAM",
        appearance_donor_label="P_MAL_A_MED_04",
        appearance_label="GhostStudio_P_KNEOH",
        portrait_resref="po_kneo",
        portrait_donor_resref="po_pfha04",
        portrait_texture_columns={
            "baseresrefg": "po_kneoL",
            "baseresrefvg": "po_kneoL",
        },
    )
    appearance = _table(
        ["label", "normalhead", "modela", "portrait"],
        [{
            "label": "P_MAL_A_MED_04",
            "normalhead": "0",
            "modela": "PMBAM",
            "portrait": "",
        }],
    )

    result = merge_head_game_records(
        patch,
        heads_bytes=_heads_k1(),
        appearance_bytes=appearance,
        portraits_bytes=_portraits(),
    )

    portraits = read_2da(result.portraits_bytes)
    assert "baseresrefg" in portraits.get_headers()
    assert "baseresrefvg" in portraits.get_headers()
    assert portraits.get_cell(result.portraits_row, "baseresrefg") == "po_kneoL"
    assert portraits.get_cell(result.portraits_row, "baseresrefvg") == "po_kneoL"

    upgraded = merge_head_game_records(
        patch,
        heads_bytes=result.heads_bytes,
        appearance_bytes=result.appearance_bytes,
        portraits_bytes=result.portraits_bytes,
    )
    assert upgraded.report["rows"]["portraits"]["action"] == "reuse_exact"
    assert upgraded.portraits_bytes == result.portraits_bytes


def test_k1_existing_owned_portrait_can_add_only_new_light_columns() -> None:
    first = HeadGameRecordPatch(
        game="K1",
        output_head_resref="P_KNEOH",
        texture_resref="P_KNEOH",
        donor_head_resref="PMHA04",
        body_resref="PMBAM",
        appearance_donor_label="P_MAL_A_MED_04",
        appearance_label="GhostStudio_P_KNEOH",
        portrait_resref="po_kneo",
        portrait_donor_resref="po_pfha04",
    )
    appearance = _table(
        ["label", "normalhead", "modela", "portrait"],
        [{
            "label": "P_MAL_A_MED_04",
            "normalhead": "0",
            "modela": "PMBAM",
            "portrait": "",
        }],
    )
    installed = merge_head_game_records(
        first,
        heads_bytes=_heads_k1(),
        appearance_bytes=appearance,
        portraits_bytes=_portraits(),
    )
    upgrade = replace(
        first,
        portrait_texture_columns={
            "baseresrefg": "po_kneoL",
            "baseresrefvg": "po_kneoL",
        },
    )

    result = merge_head_game_records(
        upgrade,
        heads_bytes=installed.heads_bytes,
        appearance_bytes=installed.appearance_bytes,
        portraits_bytes=installed.portraits_bytes,
    )

    assert result.report["rows"]["portraits"]["action"] == "upgrade_additive_columns"
    portraits = read_2da(result.portraits_bytes)
    assert portraits.get_cell(result.portraits_row, "baseresrefg") == "po_kneoL"
    assert portraits.get_cell(result.portraits_row, "baseresrefvg") == "po_kneoL"


def test_k1_heads_without_alttexture_can_append_a_custom_head() -> None:
    patch = HeadGameRecordPatch(
        game="K1",
        output_head_resref="P_KNEOH",
        texture_resref="P_KNEOH",
        donor_head_resref="PMHA04",
        body_resref="PMBAM",
        appearance_donor_label="P_MAL_A_MED_04",
        appearance_label="GhostStudio_P_KNEOH",
        head_texture_columns={
            "headtexe": "P_KNEOH",
            "headtexve": "P_KNEOH",
            "headtexvve": "P_KNEOH",
            "headtexvvve": "P_KNEOH",
        },
    )
    appearance = _table(
        ["label", "normalhead", "modela", "texa"],
        [
            {
                "label": "P_MAL_A_MED_04",
                "normalhead": "0",
                "modela": "PMBAM",
                "texa": "PMBAL",
            }
        ],
    )

    result = merge_head_game_records(
        patch,
        heads_bytes=_heads_k1(),
        appearance_bytes=appearance,
    )

    assert result.accepted
    heads = read_2da(result.heads_bytes)
    assert "alttexture" not in heads.get_headers()
    assert heads.get_cell(result.heads_row, "head") == "P_KNEOH"
    assert heads.get_cell(result.heads_row, "headtexe") == "P_KNEOH"


def test_missing_alttexture_is_rejected_when_the_game_contract_requires_it() -> None:
    appearance = _table(
        ["label", "normalhead", "modela", "texa"],
        [
            {
                "label": "P_MAL_A_MED_04",
                "normalhead": "0",
                "modela": "PMBAM",
                "texa": "PMBAL",
            }
        ],
    )
    k1_patch = HeadGameRecordPatch(
        game="K1",
        output_head_resref="P_KNEOH",
        texture_resref="P_OTHER",
        donor_head_resref="PMHA04",
        body_resref="PMBAM",
        appearance_donor_label="P_MAL_A_MED_04",
        appearance_label="GhostStudio_P_KNEOH",
    )
    with pytest.raises(HeadGameRecordError, match="ResRefs to match"):
        merge_head_game_records(
            k1_patch,
            heads_bytes=_heads_k1(),
            appearance_bytes=appearance,
        )

    k2_patch = replace(_patch(), game="K2")
    with pytest.raises(HeadGameRecordError, match="requires the alttexture"):
        merge_head_game_records(
            k2_patch,
            heads_bytes=_heads_k1(),
            appearance_bytes=_appearance(),
        )


def test_package_contains_verified_runtime_merge_metadata_and_tslpatcher(
    tmp_path: Path,
) -> None:
    result = _build(tmp_path)
    root = Path(result.package_directory)
    assert (root / "additional" / "P_CDH01.mdl").read_bytes() == (
        b"MDL verified fixture"
    )
    assert (root / "additional" / "P_CDH01.mdx").read_bytes() == (
        b"MDX verified fixture"
    )
    assert (root / "additional" / "P_CDH01.tga").is_file()
    assert (root / "additional" / "P_CDH01.txi").is_file()
    assert (root / "additional" / "head-game-records.patch.json").is_file()
    changes = (root / "tslpatchdata" / "changes.ini").read_text(
        encoding="utf-8"
    )
    assert "[2DAList]" in changes
    assert "Table0=heads.2da" in changes
    assert "Table1=appearance.2da" in changes
    assert "ExclusiveColumn=head" in changes
    assert "ExclusiveColumn=label" in changes
    assert "normalhead=2DAMEMORY0" in changes
    assert "2DAMEMORY1=RowIndex" in changes
    report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    assert report["source_files_modified"] is False
    assert report["install_plan"]["executable_edits"] is False
    assert report["install_plan"]["cache_actions"] == []
    for relative, wanted in result.hashes.items():
        assert _sha((root / relative).read_bytes()) == wanted


def test_preview_install_running_guard_backups_and_restore_are_exact(
    tmp_path: Path,
) -> None:
    package = _build(tmp_path)
    game = _game_tree(tmp_path)
    override = game / "Override"
    old_mdl = b"older unrelated test version"
    (override / "P_CDH01.mdl").write_bytes(old_mdl)
    before = _tree_hashes(game)

    running = {"swkotor2.exe": False, "launcher.exe": False}
    installer = HeadPackageInstaller(
        process_is_running=lambda name: running.get(name.casefold(), False)
    )
    preview = installer.preview(package.package_directory, game)
    assert preview.ok, preview.error
    assert _tree_hashes(game) == before
    assert preview.heads_row == 2
    assert preview.appearance_row == 2
    assert {
        value["status"] for value in preview.files
    } >= {"new", "replace_with_backup"}
    assert all(
        Path(value["target"]).parent == override.resolve()
        for value in preview.files
    )

    wrong = installer.install(
        preview,
        confirmed_preview_id="not-the-preview",
    )
    assert not wrong.ok
    assert _tree_hashes(game) == before

    running["swkotor2.exe"] = True
    blocked = installer.install(
        preview,
        confirmed_preview_id=preview.preview_id,
    )
    assert not blocked.ok
    assert "Close swkotor2.exe" in blocked.error
    assert _tree_hashes(game) == before

    running["swkotor2.exe"] = False
    installed = installer.install(
        preview,
        confirmed_preview_id=preview.preview_id,
    )
    assert installed.ok, installed.error
    assert (override / "P_CDH01.mdl").read_bytes() == (
        b"MDL verified fixture"
    )
    session = json.loads(
        Path(installed.session_manifest).read_text(encoding="utf-8")
    )
    assert session["schema"] == HEAD_INSTALL_SESSION_SCHEMA
    assert session["status"] == "installed"
    assert session["cache_actions"] == []
    assert all(
        not Path(record["target"]).name.casefold().endswith(".exe")
        for record in session["files"]
    )
    for record in session["files"]:
        if record["existed"]:
            backup = Path(record["backup"])
            assert backup.is_file()
            assert _sha(backup.read_bytes()) == record["backup_sha256"]

    restored = installer.restore(installed.session_manifest)
    assert restored.ok, restored.error
    assert _tree_hashes(game) == before
    assert (override / "P_CDH01.mdl").read_bytes() == old_mdl
    restored_session = json.loads(
        Path(installed.session_manifest).read_text(encoding="utf-8")
    )
    assert restored_session["status"] == "restored"


def test_preview_and_restore_refuse_drift(
    tmp_path: Path,
) -> None:
    package = _build(tmp_path)
    game = _game_tree(tmp_path)
    installer = HeadPackageInstaller(
        process_is_running=lambda _name: False
    )
    preview = installer.preview(package.package_directory, game)
    assert preview.ok
    target = Path(preview.files[0]["target"])
    original_target = target.read_bytes() if target.is_file() else None
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"changed after preview")
    blocked = installer.install(
        preview,
        confirmed_preview_id=preview.preview_id,
    )
    assert not blocked.ok
    assert "Target changed after preview" in blocked.error
    if original_target is None:
        target.unlink()
    else:
        target.write_bytes(original_target)

    preview = installer.preview(package.package_directory, game)
    assert preview.ok
    installed = installer.install(
        preview,
        confirmed_preview_id=preview.preview_id,
    )
    assert installed.ok
    installed_target = Path(preview.files[-1]["target"])
    installed_target.write_bytes(b"newer modder work")
    restore = installer.restore(installed.session_manifest)
    assert not restore.ok
    assert "changed after this test install" in restore.error
    assert installed_target.read_bytes() == b"newer modder work"
