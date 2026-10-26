from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "native"
    / "GhostRigger.Core.Automation"
    / "Python"
    / "src"
    / "kotormcp"
    / "tools"
    / "ghostrigger.py"
)
GLTF_SOURCE = (
    ROOT
    / "native"
    / "GhostRigger.Core.IO"
    / "Python"
    / "src"
    / "converters"
    / "mesh_converter.py"
)


def test_unreal_export_schema_exposes_bas_head_and_texture_overrides():
    source = SOURCE.read_text(encoding="utf-8")
    assert '"head_resref"' in source
    assert '"body_texture"' in source
    assert '"head_texture"' in source
    assert "Ghost Studio BAS attaches it" in source


def test_unreal_export_exposes_rigged_glb_interchange_path():
    source = SOURCE.read_text(encoding="utf-8")
    assert '"enum": ["fbx", "glb"]' in source
    assert "def _export_glb_for_unreal(" in source
    assert '"unreal_glb_intermediate"' in source
    assert 'transfer_format == "glb"' in source


def test_glb_hierarchy_tracks_duplicate_named_nodes_by_identity():
    source = GLTF_SOURCE.read_text(encoding="utf-8")
    assert "gltf_model_node_idx: Dict[int, int]" in source
    assert "gltf_model_node_idx[id(mn)] = idx_" in source
    assert "child_gltf_idx = gltf_model_node_idx[id(mn)]" in source


def test_unreal_export_routes_body_and_head_through_bas_composition():
    source = SOURCE.read_text(encoding="utf-8")
    assert "def _compose_bas_character(" in source
    assert "build_bas_preview_model(" in source
    assert 'attachment_models={"head": head_model}' in source
    assert "prepare_bas_composed_export_model(" in source
    assert '"head_resref": head_resref' in source


def test_unreal_export_preserves_head_animation_tracks_as_supplemental():
    source = SOURCE.read_text(encoding="utf-8")
    assert "supplemental_models: Tuple[Any, ...] = ()" in source
    assert "supplemental_models=tuple(supplemental_models or ())" in source
    assert "supplemental_models=(head_model,)" in source


def test_unreal_export_applies_explicit_utc_uti_texture_variation():
    source = SOURCE.read_text(encoding="utf-8")
    assert "def _apply_primary_skin_texture(" in source
    assert 'texture_overrides["head"]["sections_changed"]' in source
    assert '"texture_overrides": texture_overrides' in source
