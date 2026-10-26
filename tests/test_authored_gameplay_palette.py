from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def _install_native_payload_paths() -> None:
    repo = Path(__file__).resolve().parents[1]
    for rel in (
        "native/GhostRigger.Core.Project/Python",
        "native/GhostRigger.Core.IO/Python",
        "native/GhostRigger.Core.Workflow/Python",
        "native/GhostRigger.Core.Scene/Python",
        "native/GhostRigger.Core.Scene/Python",
        "native/GhostRigger.Core.Resources/Python",
        "native/GhostRigger.Core.Scene/Python",
        "native/GhostRigger.Core.Scene/Python",
        "native/GhostRigger.Core.Math/Python",
        "native/GhostRigger.Core.Math/Python",
        "native/GhostRigger.Core.Math/Python",
        "native/GhostRigger.Core.Rendering/Python",
        ".",
    ):
        path = str((repo / rel).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


def test_t2656_palette_maps_template_resource_types_to_gameplay_kinds() -> None:
    _install_native_payload_paths()

    from src.core.modules.authored_gameplay_palette import authored_gameplay_palette_from_library_rows

    rows = [
        {"game": "K1", "resref": "tat_guard", "restype": "utc", "category": "Templates"},
        {"game": "K1", "resref": "plc_footlker", "restype": "utp", "category": "Templates"},
        {"game": "K1", "resref": "door_t01", "restype": "utd", "category": "Templates"},
        {"game": "K1", "resref": "shop_test", "restype": "utm", "category": "Templates"},
    ]

    entries = authored_gameplay_palette_from_library_rows(rows, game="K1")
    by_resref = {entry.template_resref: entry for entry in entries}

    assert by_resref["tat_guard"].kind == "creature"
    assert by_resref["plc_footlker"].kind == "placeable"
    assert by_resref["door_t01"].kind == "door"
    assert by_resref["shop_test"].kind == "store"
    assert all(entry.confidence == "template" for entry in entries)
    assert all(not entry.warning for entry in entries)


def test_t2656_palette_only_allows_safe_model_category_fallbacks_with_warnings() -> None:
    _install_native_payload_paths()

    from src.core.modules.authored_gameplay_palette import authored_gameplay_palette_from_library_rows

    rows = [
        {"game": "K1", "resref": "c_rancor", "category": "Creatures", "source": "swkotor"},
        {"game": "K1", "resref": "plc_bench", "category": "Placeables", "source": "swkotor"},
        {"game": "K2", "resref": "dor_metal01", "category": "Doors", "source": "swkotor2"},
        {"game": "K1", "resref": "w_blstrpstl_001", "category": "Weapons", "source": "swkotor"},
    ]

    entries = authored_gameplay_palette_from_library_rows(rows)
    by_resref = {entry.template_resref: entry for entry in entries}

    assert by_resref["c_rancor"].kind == "creature"
    # Placeable and door geometry names are not runtime template authority.
    # Those rows enter the placement palette only after library discovery has
    # attached a true UTP/UTD resref.
    assert "plc_bench" not in by_resref
    assert "dor_metal01" not in by_resref
    assert "w_blstrpstl_001" not in by_resref
    assert "Verify this resref has a matching creature template" in by_resref["c_rancor"].warning


def test_t2656_palette_filters_by_game_kind_and_query() -> None:
    _install_native_payload_paths()

    from src.core.modules.authored_gameplay_palette import authored_gameplay_palette_from_library_rows

    rows = [
        {"game": "K1", "resref": "tat_guard", "restype": "utc"},
        {"game": "K2", "resref": "tel_guard", "restype": "utc"},
        {"game": "K1", "resref": "plc_bench", "restype": "utp"},
    ]

    entries = authored_gameplay_palette_from_library_rows(rows, game="K1", kind="creature", query="guard")

    assert [entry.template_resref for entry in entries] == ["tat_guard"]


def test_placeable_palette_does_not_silently_truncate_large_k2_template_library() -> None:
    _install_native_payload_paths()

    from src.core.modules.authored_gameplay_palette import authored_gameplay_palette_from_library_rows

    rows = [
        {"game": "K2", "resref": f"plc_{index:04d}", "restype": "utp", "category": "Placeables"}
        for index in range(882)
    ]

    entries = authored_gameplay_palette_from_library_rows(rows, game="K2", kind="placeable")

    assert len(entries) == 882
    assert entries[0].template_resref == "plc_0000"
    assert entries[-1].template_resref == "plc_0881"


def test_t2656_animated_doors_share_placeables_family_without_losing_utd_kind() -> None:
    _install_native_payload_paths()

    from src.core.modules.authored_gameplay_palette import authored_gameplay_palette_from_library_rows

    rows = [
        {"game": "K2", "resref": "plc_bench", "restype": "utp", "category": "Placeables"},
        {"game": "K2", "resref": "dor_metal01", "restype": "utd", "category": "Doors"},
    ]

    entries = authored_gameplay_palette_from_library_rows(rows, game="K2", kind="placeable")
    by_resref = {entry.template_resref: entry for entry in entries}

    assert set(by_resref) == {"plc_bench", "dor_metal01"}
    assert by_resref["plc_bench"].kind == "placeable"
    assert by_resref["dor_metal01"].kind == "door"
    assert by_resref["dor_metal01"].authoring_family == "placeable"
    assert by_resref["dor_metal01"].category == "Placeables / Animated Doors"
    assert by_resref["dor_metal01"].metadata["restype"] == "utd"


def test_t2656_map_studio_palette_preserves_door_kind_when_chosen_from_placeables() -> None:
    repo = Path(__file__).resolve().parents[1]
    for relative in (
        "native/GhostRigger.Core.GUI.Display/Python/src/gui/panels/module_editor/placement_tab.py",
        "native/GhostRigger.Core.GUI.Display/Python/src/gui/panels/module_editor/builder_tab.py",
    ):
        source = (repo / relative).read_text(encoding="utf-8")
        assert 'entry_family = ' in source
        assert 'entry_kind != kind and entry_family != kind' in source
    placement_source = (
        repo
        / "native/GhostRigger.Core.GUI.Display/Python/src/gui/panels/module_editor/placement_tab.py"
    ).read_text(encoding="utf-8")
    assert "self.kind_combo.findData(entry_kind)" in placement_source
    assert "exported as UTD + GIT Door List" in placement_source


def test_t2656_module_editor_builder_exposes_searchable_gameplay_palette() -> None:
    repo = Path(__file__).resolve().parents[1]
    builder_source = (
        repo
        / "native"
        / "GhostRigger.Core.GUI.Display"
        / "Python"
        / "src"
        / "gui"
        / "panels"
        / "module_editor"
        / "builder_tab.py"
    ).read_text(encoding="utf-8")
    controller_source = (
        repo
        / "native"
        / "GhostRigger.Core.Scene"
        / "Python"
        / "src"
        / "core"
        / "modules"
        / "module_editor_controller.py"
    ).read_text(encoding="utf-8")
    window_source = (
        repo
        / "native"
        / "GhostRigger.Core.Tools"
        / "Python"
        / "src"
        / "gui"
        / "windows"
        / "module_editor_window.py"
    ).read_text(encoding="utf-8")

    assert "mapStudioGameplayPaletteSearchLineEdit" in builder_source
    assert "mapStudioGameplayPaletteComboBox" in builder_source
    assert "mapStudioGameplayPaletteResultLabel" in builder_source
    assert "mapStudioUseGameplayPaletteButton" in builder_source
    assert "self._gameplay_palette_page_limit = 192" in builder_source
    assert "matches[: self._gameplay_palette_page_limit]" in builder_source
    assert "set_gameplay_palette_entries" in builder_source
    assert "authored_gameplay_palette_entries" in controller_source
    assert "authored_gameplay_palette_from_library_rows" in controller_source
    assert "palette = self.controller.authored_gameplay_palette_entries(self._library_rows)" in window_source
    assert "self.builder_tab.set_gameplay_palette_entries(palette)" in window_source


def test_placeable_utp_export_preserves_base_unknown_fields_and_reads_back(tmp_path: Path) -> None:
    _install_native_payload_paths()
    from hashlib import sha256

    from pykotor.common.misc import Game, ResRef
    from pykotor.resource.formats.gff import bytes_gff, read_gff
    from pykotor.resource.formats.twoda import TwoDA, bytes_2da
    from pykotor.resource.generics.utp import UTP, dismantle_utp

    from src.core.placeables.placeable_utp_io import export_placeable_utp
    from src.core.project.placeable_asset import (
        PlaceableAppearanceMappingEvidence,
        PlaceableAsset,
        PlaceableBaseTemplateEvidence,
        PlaceableGameplay,
        PlaceableResourceRefs,
    )
    from src.core.project.resource_address import ResourceAddress

    base_ref = ResourceAddress(scheme="game_resource", game="k2", resref="plc_bench", restype="UTP", layer="base")
    base_utp = UTP()
    base_utp.resref = ResRef("plc_bench")
    base_utp.tag = "plc_bench"
    base_utp.appearance_id = 1
    base_gff = dismantle_utp(base_utp, Game.K2)
    base_gff.root.set_string("GRUnknownField", "keep_me")
    base_bytes = bytes_gff(base_gff)

    table = TwoDA(["modelname"])
    table.add_row("0", {"modelname": "plc_unused"})
    table.add_row("1", {"modelname": "gr_terminal"})
    appearance_bytes = bytes_2da(table)
    mdl = ResourceAddress(scheme="project_resource", game="k2", resref="gr_terminal", restype="MDL", path="gr_terminal.mdl")
    asset = PlaceableAsset(
        game="K2",
        template_resref="gr_terminal",
        tag="gr_terminal",
        display_name="Ghost Terminal",
        category="terminal",
        visual_source="custom",
        appearance_id=1,
        gameplay=PlaceableGameplay(
            useable=True,
            has_inventory=True,
            inventory_items=["g_i_datapad01"],
            lockable=True,
            locked=True,
            unlock_dc=18,
            maximum_hp=25,
            current_hp=20,
            conversation_resref="gr_terminal_dlg",
        ),
        scripts={"on_used": "gr_terminal_use"},
        resources=PlaceableResourceRefs(
            mdl=mdl,
            mdx=ResourceAddress(scheme="project_resource", game="k2", resref="gr_terminal", restype="MDX", path="gr_terminal.mdx"),
            pwk=ResourceAddress(scheme="project_resource", game="k2", resref="gr_terminal", restype="PWK", path="gr_terminal.pwk"),
            textures=[ResourceAddress(scheme="project_resource", game="k2", resref="gr_terminal01", restype="TPC", path="gr_terminal01.tpc")],
        ),
        base_template=base_ref,
        base_evidence=PlaceableBaseTemplateEvidence(
            template=base_ref,
            sha256=sha256(base_bytes).hexdigest(),
            field_count=len(base_gff.root),
            source="templates.bif",
        ),
        appearance_evidence=PlaceableAppearanceMappingEvidence(
            game="K2",
            appearance_id=1,
            model_resref="gr_terminal",
            source="placeables.2da",
            source_sha256=sha256(appearance_bytes).hexdigest(),
            verified=True,
        ),
    )

    result = export_placeable_utp(
        asset,
        base_utp_bytes=base_bytes,
        appearance_2da_bytes=appearance_bytes,
        output_path=tmp_path / "gr_terminal.utp",
    )
    output_gff = read_gff(result.utp_bytes)

    assert result.readback.template_resref == "gr_terminal"
    assert result.readback.appearance_id == 1
    assert result.readback.unlock_dc == 18
    assert result.readback.maximum_hp == 25
    assert result.readback.scripts["on_used"] == "gr_terminal_use"
    assert result.structurally_grounded is True
    assert result.engine_ready is False
    assert result.readiness_status == "structurally_grounded_unproven"
    assert "GRUnknownField" in result.preserved_unknown_labels
    assert output_gff.root.get_string("GRUnknownField") == "keep_me"
    assert {"LocName", "HP", "OpenLockDC"} <= set(output_gff.root.keys())
    assert {"LocalizedName", "MaxHP", "LockDC"}.isdisjoint(output_gff.root.keys())
    assert list(tmp_path.glob(".*.tmp")) == []


def test_placeable_bundle_flags_dependencies_and_rejects_resource_collisions() -> None:
    _install_native_payload_paths()

    from src.core.placeables.placeable_utp_io import (
        PlaceableResourceCollisionError,
        build_placeable_resource_bundle,
        export_placeable_utp,
    )
    from src.core.project.placeable_asset import PlaceableAsset, PlaceableGameplay, PlaceableResourceRefs
    from src.core.project.resource_address import ResourceAddress

    mdl = ResourceAddress(scheme="project_resource", game="k2", resref="gr_crate", restype="MDL", path="a/gr_crate.mdl")
    asset = PlaceableAsset(
        game="K2",
        template_resref="gr_crate",
        tag="gr_crate",
        category="container",
        visual_source="custom",
        appearance_id=4,
        gameplay=PlaceableGameplay(
            has_inventory=True,
            inventory_items=["g_i_parts01"],
            conversation_resref="gr_crate_dlg",
        ),
        scripts={"on_used": "gr_crate_use"},
        resources=PlaceableResourceRefs(
            mdl=mdl,
            mdx=ResourceAddress(scheme="project_resource", game="k2", resref="gr_crate", restype="MDX", path="a/gr_crate.mdx"),
        ),
    )
    exported = export_placeable_utp(asset)
    payloads = {"MDL": b"mdl", "MDX": b"mdx"}
    bundle = build_placeable_resource_bundle(asset, exported, resource_reader=lambda address: payloads[address.restype])

    missing = {issue.resource_key for issue in bundle.issues if issue.code == "missing_placeable_dependency"}
    assert missing == {("gr_crate_use", "NCS"), ("gr_crate_dlg", "DLG"), ("g_i_parts01", "UTI")}
    assert bundle.engine_ready is False
    assert ("gr_crate", ".UTP", exported.utp_bytes) in bundle.output_resources

    dependencies = (
        ("gr_crate_use", ".NCS", b"ncs"),
        ("gr_crate_dlg", ".DLG", b"dlg"),
        ("g_i_parts01", ".UTI", b"uti"),
    )
    complete = build_placeable_resource_bundle(
        asset,
        exported,
        resource_reader=lambda address: payloads[address.restype],
        existing_resources=dependencies,
    )
    assert complete.has_blocking is False

    with pytest.raises(PlaceableResourceCollisionError, match="gr_crate.mdl"):
        build_placeable_resource_bundle(
            asset,
            exported,
            resource_reader=lambda address: payloads[address.restype],
            existing_resources=(("gr_crate", ".MDL", b"different"),),
        )


def test_placeable_workflow_facade_returns_library_rows_and_export_resources(tmp_path: Path) -> None:
    _install_native_payload_paths()

    from src.core.project.placeable_asset import PlaceableAsset, save_placeable_asset
    from src.core.workflow.placeable_builder_service import (
        placeable_library_rows,
        referenced_placeable_resource_report,
        referenced_placeable_resources,
    )

    asset = PlaceableAsset(
        game="K1",
        template_resref="gr_statue",
        tag="gr_statue",
        display_name="Ghost Statue",
        category="decor",
        appearance_id=9,
    )
    save_placeable_asset(asset, tmp_path / "gr_statue.ghostplaceable.json")

    rows = placeable_library_rows(tmp_path, game="K1")
    report = referenced_placeable_resource_report(tmp_path, ("gr_statue",), game="K1")
    resources = referenced_placeable_resources(tmp_path, ("gr_statue",), game="K1")

    assert rows[0]["resref"] == "gr_statue"
    assert rows[0]["restype"] == "utp"
    assert report.selected_template_resrefs == ("gr_statue",)
    assert report.has_blocking is False
    assert report.engine_ready is False
    assert resources == report.resources
    assert resources[0][0:2] == ("gr_statue", ".UTP")
    assert len(report.secondary_motion_profiles) == 1
    profile_id, profile_bytes = report.secondary_motion_profiles[0]
    assert profile_id == asset.secondary_motion["profile_id"]
    profile = json.loads(profile_bytes)
    assert profile["model_identity"]["target_kind"] == "placeable"
    assert profile["runtime"]["state"] == "capture_only"
    assert profile["runtime"]["write_enabled"] is False


def test_placeable_workflow_accepts_dependency_that_resolves_from_target_game(tmp_path: Path) -> None:
    _install_native_payload_paths()

    from src.core.project.placeable_asset import PlaceableAsset, save_placeable_asset
    from src.core.project.resource_address import ResourceAddress
    from src.core.resources.game_resource_provider import GameResourceRecord, InMemoryGameResourceProvider
    from src.core.workflow.placeable_builder_service import referenced_placeable_resource_report

    asset = PlaceableAsset(
        game="K2",
        template_resref="gr_terminal",
        tag="gr_terminal",
        category="terminal",
        appearance_id=4,
        scripts={"on_used": "k_existing_use"},
    )
    save_placeable_asset(asset, tmp_path / "gr_terminal.ghostplaceable.json")
    script_address = ResourceAddress(
        scheme="game_resource",
        game="K2",
        layer="base",
        resref="k_existing_use",
        restype="NCS",
    )
    provider = InMemoryGameResourceProvider(
        ((GameResourceRecord(address=script_address, source="chitin:scripts.bif", priority=40), b"compiled-script"),)
    )

    report = referenced_placeable_resource_report(
        tmp_path,
        ("gr_terminal",),
        game="K2",
        provider=provider,
    )

    assert report.has_blocking is False
    resolved = [issue for issue in report.issues if issue.code == "placeable_dependency_external_resolved"]
    assert len(resolved) == 1
    assert resolved[0].resource_key == ("k_existing_use", "NCS")
    assert any(resource[0:2] == ("gr_terminal", ".UTP") for resource in report.resources)


def test_placeable_tool_service_clones_stock_without_losing_known_or_unknown_utp_fields(tmp_path: Path) -> None:
    _install_native_payload_paths()
    from pykotor.common.misc import Game, ResRef
    from pykotor.resource.formats.gff import bytes_gff, read_gff
    from pykotor.resource.formats.twoda import TwoDA, bytes_2da
    from pykotor.resource.generics.utp import UTP, dismantle_utp
    from src.core.project.resource_address import ResourceAddress
    from src.core.resources.game_resource_provider import GameResourceRecord, InMemoryGameResourceProvider
    from src.core.tools.placeable_builder_tool_service import PlaceableBuilderToolService

    stock = UTP()
    stock.resref = ResRef("plc_terminal")
    stock.tag = "plc_terminal"
    stock.appearance_id = 1
    stock.useable = True
    stock.auto_remove_key = True
    stock.trap_detectable = True
    stock.trap_detect_dc = 17
    stock.trap_disarmable = True
    stock.trap_disarm_dc = 19
    stock.trap_flag = 3
    stock.trap_one_shot = True
    stock.plot = True
    stock.min1_hp = True
    stock.not_blastable = True
    stock.party_interact = True
    stock_gff = dismantle_utp(stock, Game.K2)
    stock_gff.root.set_string("GRUnknownStock", "preserve")
    stock_bytes = bytes_gff(stock_gff)
    table = TwoDA(["modelname"])
    table.add_row("0", {"modelname": "plc_unused"})
    table.add_row("1", {"modelname": "plc_terminal"})
    twoda_bytes = bytes_2da(table)

    def record(resref: str, restype: str) -> GameResourceRecord:
        return GameResourceRecord(
            address=ResourceAddress(
                scheme="game_resource",
                game="K2",
                layer="base",
                resref=resref,
                restype=restype,
            ),
            source="chitin:templates.bif" if restype == "UTP" else "chitin:2da.bif",
            priority=40,
        )

    provider = InMemoryGameResourceProvider(
        ((record("plc_terminal", "UTP"), stock_bytes), (record("placeables", "2DA"), twoda_bytes))
    )
    service = PlaceableBuilderToolService(tmp_path, provider=provider)
    row = next(row for row in service.rows(game="K2") if row["resref"] == "plc_terminal")
    loaded = service.load_row(row)

    assert loaded.gameplay.auto_remove_key is True
    assert loaded.gameplay.trap_detect_dc == 17
    assert loaded.gameplay.trap_disarm_dc == 19
    assert loaded.gameplay.trap_flag == 3
    assert loaded.gameplay.trap_one_shot is True
    assert loaded.gameplay.plot is True
    assert loaded.gameplay.min1_hp is True
    assert loaded.gameplay.not_blastable is True
    assert loaded.gameplay.party_interact is True
    assert loaded.base_evidence is not None
    assert loaded.appearance_evidence is not None and loaded.appearance_evidence.verified is True

    cloned = service.clone_asset(loaded)
    assert cloned.asset_id != loaded.asset_id
    assert cloned.template_resref != loaded.template_resref
    assert len(cloned.template_resref) <= 16
    saved = service.save(cloned)
    assert saved.ok is True, saved.messages
    assert Path(saved.sidecar_path).is_file()
    assert Path(saved.utp_path).is_file()
    assert saved.utp_result is not None and saved.utp_result.structurally_grounded is True
    output = read_gff(Path(saved.utp_path).read_bytes())
    assert output.root.get_string("GRUnknownStock") == "preserve"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_placeable_tool_service_new_and_invalid_documents_keep_engine_proof_honest(tmp_path: Path) -> None:
    _install_native_payload_paths()
    from src.core.tools.placeable_builder_tool_service import PlaceableBuilderToolService

    service = PlaceableBuilderToolService(tmp_path)
    first = service.new_asset(game="K1")
    second = service.new_asset(game="K1")
    assert first.asset_id != second.asset_id
    result = service.save(first)
    assert result.ok is False
    assert result.engine_ready is False
    assert any("Template resref" in message for message in result.messages)
    assert not list(tmp_path.glob("*.utp"))
