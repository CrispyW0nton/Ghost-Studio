from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "blender_sanitize_unreal_animation_fbx.py"


def test_sanitizer_exports_only_the_armature_and_not_bas_helpers() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert 'object_types={"ARMATURE"}' in source
    assert 'armature.select_set(True)' in source
    assert 'bpy.ops.object.select_all(action="DESELECT")' in source
    assert 'object_types={"ARMATURE", "MESH", "EMPTY"}' not in source
    assert '"excluded_object_count"' in source
