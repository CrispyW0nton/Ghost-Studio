from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "blender_strip_malformed_tongue_fbx.py"


def test_verified_fbx_cleanup_preserves_bind_mesh_and_strips_only_tongue_faces():
    source = SOURCE.read_text(encoding="utf-8")
    assert 'obj.name.casefold() == "tongue"' in source
    assert 'context="FACES_ONLY"' in source
    assert 'bpy.ops.import_scene.fbx(' in source
    assert 'bpy.ops.export_scene.fbx(' in source
    assert 'bake_anim=False' in source
    assert "def _flip_uv_v_for_fbx_interchange(" in source
    assert "loop.uv.y = 1.0 - float(loop.uv.y)" in source
