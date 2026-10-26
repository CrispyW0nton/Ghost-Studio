from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NORMALIZER = ROOT / "scripts" / "blender_batch_normalize_enclave_characters.py"
MATERIAL_MAP = ROOT / "scripts" / "blender_audit_enclave_material_image_map.py"
ALIASES = ROOT / "scripts" / "blender_stage_enclave_material_texture_aliases.py"


def test_character_export_materializes_images_before_fbx_serialization():
    source = NORMALIZER.read_text(encoding="utf-8")
    materialize = source.index("materialized_texture_files = _materialize_base_color_images(")
    export = source.index("_export_base(base_path)")
    assert materialize < export
    assert "_material_family(material.name)" in source


def test_material_family_tools_follow_mix_nodes_to_the_source_image():
    for script in (NORMALIZER, MATERIAL_MAP, ALIASES):
        source = script.read_text(encoding="utf-8")
        assert "def _upstream_image_node(" in source
        assert "for input_socket in node.inputs:" in source
        assert "_upstream_image_node(input_socket, visited)" in source
