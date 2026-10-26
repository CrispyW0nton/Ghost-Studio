from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "maya_normalize_ghostrigger_fbx_for_unreal.py"


def test_normalizer_uses_autodesk_fbx_and_requires_real_skinning():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'cmds.loadPlugin("fbxmaya"' in source
    assert 'cmds.ls(type="joint"' in source
    assert 'cmds.ls(type="skinCluster"' in source
    assert "if not joints or not meshes or not skins" in source


def test_normalizer_bakes_animation_and_exports_binary_fbx():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "FBXExportInAscii -v false" in source
    assert "FBXExportBakeComplexAnimation -v true" in source
    assert "FBXExportQuaternion -v resample" in source
    assert "FBXExportAxisConversionMethod convertAnimation" in source


def test_normalizer_writes_an_auditable_unreal_handoff_report():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"ghoststudio.maya-unreal-fbx-normalization.v1"' in source
    assert '"animation_curve_count"' in source
    assert '"frame_range"' in source
