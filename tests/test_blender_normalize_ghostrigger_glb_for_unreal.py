from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "blender_normalize_ghostrigger_glb_for_unreal.py"


def test_blender_normalizer_imports_glb_and_exports_binary_fbx():
    source = SOURCE.read_text(encoding="utf-8")
    assert "bpy.ops.import_scene.gltf" in source
    assert "bpy.ops.export_scene.fbx" in source
    assert 'object_types={"ARMATURE", "MESH", "EMPTY"}' in source
    assert "embed_textures=True" in source


def test_blender_normalizer_exports_all_animation_actions():
    source = SOURCE.read_text(encoding="utf-8")
    assert "bake_anim_use_all_actions=selected_action is None" in source
    assert "bake_anim_use_nla_strips=False" in source
    assert '"actions": actions' in source


def test_blender_normalizer_can_export_one_named_action_for_unreal():
    source = SOURCE.read_text(encoding="utf-8")
    assert 'parser.add_argument("--action", default="")' in source
    assert "armature.animation_data.action = selected_action" in source
    assert "bpy.context.scene.render.fps = 30" in source
    assert "bpy.context.scene.frame_end = int(round(end))" in source
    assert '"selected_action": selected_action.name if selected_action else ""' in source


def test_blender_normalizer_rejects_missing_character_contracts():
    source = SOURCE.read_text(encoding="utf-8")
    assert "did not produce an armature" in source
    assert "did not produce any meshes" in source
    assert "did not produce any animation actions" in source
