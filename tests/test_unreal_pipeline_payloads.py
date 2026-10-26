"""Focused registration checks; unrelated in-progress payloads are not repaired."""
import hashlib
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAYLOADS = [
    ("IO", "io/unreal_editor_connection.py"),
    ("IO", "io/unreal_pipeline_store.py"),
    ("Unreal", "adapters/kotor_retail_pipeline.py"),
    ("Workflow", "core/workflow/unreal_pipeline_jobs.py"),
    ("GUI.Display", "gui/dialogs/unreal_conversion_dialog.py"),
    ("GUI.Display", "gui/panels/unreal_pipeline_panel.py"),
    ("Scene", "core/modules/unreal_level_import.py"),
    ("Scene", "core/modules/unreal_import_state.py"),
    ("Scene", "core/modules/unreal_authored_reimport.py"),
    ("Scene", "core/modules/unreal_sector_projection.py"),
    ("IO", "io/unreal_project_assets.py"),
    ("IO", "io/unreal_native_snapshot.py"),
    ("IO", "io/unreal_kmap_storage.py"),
    ("Math", "math/unreal_native_surface.py"),
    ("Workflow", "core/workflow/unreal_map_import.py"),
    ("GUI.Display", "gui/dialogs/unreal_level_import_dialog.py"),
]


@pytest.mark.parametrize("owner,path", PAYLOADS)
def test_unreal_owner_payload_matches_canonical_source_and_all_build_lists(owner, path):
    project = f"GhostRigger.Core.{owner}"
    directory = ROOT / "native" / project
    packaged = f"Python/src/{path}"
    payload = directory / packaged
    data = (ROOT / "src" / path).read_bytes()
    assert payload.read_bytes() == data
    manifest = json.loads((directory / "GhostRiggerPythonPayload.json").read_text())
    row = next(row for row in manifest["files"] if row["packaged_path"] == packaged)
    assert row["sha256"] == hashlib.sha256(data).hexdigest()
    assert f'{row["resource_name"]} RCDATA "{packaged}"' in (directory / "GhostRiggerPythonPayload.rc").read_text()
    for suffix in (".vcxproj", ".vcxproj.filters"):
        assert packaged.replace("/", "\\") in (directory / (project + suffix)).read_text()
    entry = next(entry for entry in json.loads((ROOT / "native/GhostRigger.PythonPayloadManifest.json").read_text()) if entry["project"] == project)
    assert f"src/{path}".replace("/", "\\") in entry["files"]


def test_canonical_qt_import_resolves_without_constructing_widgets():
    from src.gui.qt_lib.dialogs.unreal_level_import_dialog import import_unreal_level
    assert callable(import_unreal_level)


def test_pipeline_qt_facades_import_without_constructing_widgets():
    from src.gui.qt_lib.dialogs.unreal_conversion_dialog import UnrealConversionDialog
    from src.gui.qt_lib.panels.unreal_pipeline_panel import UnrealPipelinePanel
    assert UnrealConversionDialog and UnrealPipelinePanel
