import asyncio
from hashlib import sha256
from io import BytesIO
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from src.core.export.unity_export_bridge import (
    build_output_paths,
    export_model_for_unity,
    inspect_fbx_skin_objects,
    summarize_model,
)
from src.core.export.unity_import_validator import build_unity_import_manifest
from src.core.geometry.model_data import Animation, BoneWeight, KotorModel, ModelNode, NodeFlags, VertexSkinData
from src.converters.mesh_converter import FBXExporter
from kotormcp.tools import get_all_tools, handle_tool
from kotormcp.tools import ghostrigger


class _Node:
    def __init__(self, name, is_mesh=False, vertices=None, faces=None):
        self.name = name
        self.is_mesh = is_mesh
        self.vertices = vertices or []
        self.faces = faces or []


class _Anim:
    def __init__(self, name):
        self.name = name


class _Model:
    name = "n_darthmalak"
    supermodel = "NULL"
    model_type = 4
    animations = [_Anim("pause1"), _Anim("tlknorm")]

    def all_nodes(self):
        return [
            _Node("n_darthmalak"),
            _Node("headhook"),
            _Node("torso", is_mesh=True, vertices=[1, 2, 3], faces=[1]),
        ]


def test_build_output_paths_stays_inside_unity_assets():
    unity_project = Path("C:/Unity/Kotor-Unity")

    asset_path, metadata_path = build_output_paths(
        unity_project,
        "Assets/KotorImported/MainMenu/Models/Malak",
        "n_darthmalak",
        "fbx",
    )

    assert asset_path == unity_project / "Assets/KotorImported/MainMenu/Models/Malak/n_darthmalak.fbx"
    assert metadata_path == unity_project / "Assets/KotorImported/MainMenu/Models/Malak/n_darthmalak.ghostrigger.json"


def test_summarize_model_records_counts_and_unity_asset_path():
    unity_project = Path("C:/Unity/Kotor-Unity")
    asset_path = unity_project / "Assets/KotorImported/Malak/n_darthmalak.fbx"

    metadata = summarize_model(_Model(), "K1", "n_darthmalak", asset_path, unity_project)

    assert metadata["source"]["game"] == "K1"
    assert metadata["source"]["resref"] == "n_darthmalak"
    assert metadata["counts"]["nodes"] == 3
    assert metadata["counts"]["mesh_nodes"] == 1
    assert metadata["counts"]["vertices"] == 3
    assert metadata["counts"]["faces"] == 1
    assert metadata["counts"]["animations"] == 2
    assert metadata["animations"] == ["pause1", "tlknorm"]
    assert metadata["unity"]["asset_path"] == "Assets/KotorImported/Malak/n_darthmalak.fbx"
    assert metadata["unity"]["compatibility_profile"] == "unity"
    assert metadata["unity"]["recommended_import"]["scale_factor"] == 1.0
    assert metadata["unity"]["recommended_import"]["bake_axis_conversion"] is True


def test_mcp_tool_manifest_exposes_unity_export_action():
    names = {tool["name"] for tool in get_all_tools()}

    assert "ghostrigger_export_model_for_unity" in names
    assert "ghostrigger_validate_unity_import" in names


def test_mcp_unity_export_action_writes_asset_and_metadata(monkeypatch):
    out_root = Path(".pytest_tmp_unity_bridge") / "UnityProject"

    class _Locator:
        def locate(self, resref, game, game_path):
            assert resref == "n_darthmalak"
            assert game == "k1"
            return "installation:n_darthmalak.mdl", b"mdl", b"mdx"

    class _Parser:
        def parse(self, mdl_bytes, mdx_bytes, path_label):
            assert mdl_bytes == b"mdl"
            assert mdx_bytes == b"mdx"
            return _Model()

    def _fake_exporter(model, out_path, export_rigging):
        assert export_rigging is True
        out_path.write_text("fbx", encoding="utf-8")
        return True

    services = SimpleNamespace(locator=_Locator(), parser=_Parser(), analyzer=None, registry=None)
    monkeypatch.setattr(ghostrigger, "_get_services", lambda: services)
    monkeypatch.setattr(ghostrigger, "_export_fbx_for_unity", _fake_exporter)

    try:
        result = asyncio.run(handle_tool(
            "ghostrigger_export_model_for_unity",
            {
                "game": "k1",
                "resref": "n_darthmalak",
                "unity_project": str(out_root),
                "asset_subdir": "Assets/KotorImported/Test",
            },
        ))
        payload = json.loads(result["text"])

        assert payload["status"] == "ok"
        assert payload["unity_asset_path"] == "Assets/KotorImported/Test/n_darthmalak.fbx"
        assert Path(payload["asset"]).exists()
        assert Path(payload["metadata"]).exists()
        sidecar = json.loads(Path(payload["metadata"]).read_text(encoding="utf-8"))
        assert sidecar["source"]["path"] == "installation:n_darthmalak.mdl"
        assert sidecar["counts"]["animations"] == 2
        assert sidecar["fbx"]["checked"] is True
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


def test_mcp_unity_export_resolves_supermodel_for_bind_pose(monkeypatch):
    out_root = Path(".pytest_tmp_unity_bridge") / "UnitySupermodelProject"
    exported = {}

    model = _Model()
    model.supermodel = "S_Male02"
    base_skeleton = _Model()
    base_skeleton.name = "S_Male02"

    class _Locator:
        def locate(self, resref, game, game_path):
            return f"installation:{resref}.mdl", resref.encode(), b"mdx"

    class _Parser:
        def parse(self, mdl_bytes, mdx_bytes, path_label):
            return base_skeleton if mdl_bytes == b"S_Male02" else model

    def _fake_exporter(candidate, out_path, export_rigging):
        exported["base"] = getattr(candidate, "_gr_fbx_base_skeleton_model", None)
        out_path.write_text("fbx", encoding="utf-8")
        return True

    services = SimpleNamespace(locator=_Locator(), parser=_Parser(), analyzer=None, registry=None)
    monkeypatch.setattr(ghostrigger, "_get_services", lambda: services)
    monkeypatch.setattr(ghostrigger, "_export_fbx_for_unity", _fake_exporter)

    try:
        result = asyncio.run(handle_tool(
            "ghostrigger_export_model_for_unity",
            {
                "game": "k1",
                "resref": "pmbam",
                "unity_project": str(out_root),
                "asset_subdir": "Assets/KotorImported/Test",
            },
        ))
        payload = json.loads(result["text"])
        assert payload["status"] == "ok"
        assert exported["base"] is base_skeleton
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


def test_unity_export_sidecar_records_fbx_skin_diagnostics():
    out_root = Path(".pytest_tmp_unity_bridge") / "UnityProject"

    def _fake_exporter(model, out_path, export_rigging):
        out_path.write_text(
            '\n'.join([
                'Model: 10, "Model::pelvis_g", "LimbNode" {',
                'NodeAttribute: 11, "NodeAttribute::pelvis_g", "LimbNode" {',
                'Deformer: 1, "Deformer::torso_Skin", "Skin" {',
                'Deformer: 2, "SubDeformer::root", "Cluster" {',
                'Transform: *16 {',
                'TransformLink: *16 {',
                'Pose: 3, "Pose::BIND_POSES", "BindPose" {',
                'PoseNode:  {',
                'Texture: 4, "Texture::torso_diff", "" {',
                'P: "WrapModeU","enum","","",0',
                'P: "WrapModeV","enum","","",0',
                'AnimationStack: 5, "AnimStack::pause1", "" {',
                'AnimationLayer: 6, "AnimLayer::pause1", "" {',
                'AnimationCurve: 7, "AnimCurve::pause1", "" {',
            ]),
            encoding="utf-8",
        )
        return True

    try:
        result = export_model_for_unity(
            _Model(),
            game="K1",
            resref="n_darthmalak",
            unity_project=out_root,
            asset_subdir="Assets/KotorImported/Test",
            extension="fbx",
            export_rigging=True,
            exporter=_fake_exporter,
        )
        sidecar = json.loads(Path(result["metadata"]).read_text(encoding="utf-8"))

        fbx = sidecar["fbx"]
        assert fbx["checked"] is True
        assert fbx["skin_deformers"] == 1
        assert fbx["clusters"] == 1
        assert fbx["legacy_subdeformer_clusters"] == 0
        assert fbx["skeleton_node_attributes"] == 1
        assert fbx["limb_node_models"] == 1
        assert fbx["bind_poses"] == 1
        assert fbx["pose_nodes"] == 1
        assert fbx["texture_wrap_u"] == 1
        assert fbx["texture_wrap_v"] == 1
        assert fbx["animation_stacks"] == 1
        assert fbx["animation_layers"] == 1
        assert fbx["animation_curves"] == 1
        assert fbx["skin_contract_ok"] is True
        assert fbx["texture_contract_ok"] is True
        assert fbx["duplicate_object_ids"] == {}
        assert fbx["ok"] is True
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


def test_unity_export_can_use_custom_output_name():
    out_root = Path(".pytest_tmp_unity_bridge") / "UnityProject"

    def _fake_exporter(model, out_path, export_rigging):
        out_path.write_text("fbx", encoding="utf-8")
        return True

    try:
        result = export_model_for_unity(
            _Model(),
            game="K1",
            resref="n_darthmalak",
            asset_name="N_DarthMalak_GhostRiggerFresh",
            unity_project=out_root,
            asset_subdir="Assets/KotorImported/Test",
            extension="fbx",
            export_rigging=True,
            exporter=_fake_exporter,
        )

        assert result["unity_asset_path"] == (
            "Assets/KotorImported/Test/N_DarthMalak_GhostRiggerFresh.fbx"
        )
        sidecar = json.loads(Path(result["metadata"]).read_text(encoding="utf-8"))
        assert sidecar["source"]["resref"] == "n_darthmalak"
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


def test_fbx_skin_diagnostics_flags_legacy_subdeformer_clusters(tmp_path):
    path = tmp_path / "bad.fbx"
    path.write_text(
        '\n'.join([
            'SubDeformer: 42, "SubDeformer::root", "Cluster" {',
            'SubDeformer: 42, "SubDeformer::root", "Cluster" {',
        ]),
        encoding="utf-8",
    )

    diagnostics = inspect_fbx_skin_objects(path)

    assert diagnostics["ok"] is False
    assert diagnostics["clusters"] == 0
    assert diagnostics["legacy_subdeformer_clusters"] == 2


def test_fbx_skin_diagnostics_counts_unity_compatible_cluster_records(tmp_path):
    path = tmp_path / "good.fbx"
    path.write_text(
        '\n'.join([
            'Model: 10, "Model::pelvis_g", "LimbNode" {',
            'NodeAttribute: 11, "NodeAttribute::pelvis_g", "LimbNode" {',
            'Deformer: 41, "Deformer::torso_Skin", "Skin" {',
            'Deformer: 42, "SubDeformer::root", "Cluster" {',
            'Transform: *16 {',
            'TransformLink: *16 {',
            'Pose: 43, "Pose::BIND_POSES", "BindPose" {',
            'PoseNode:  {',
        ]),
        encoding="utf-8",
    )

    diagnostics = inspect_fbx_skin_objects(path)

    assert diagnostics["ok"] is True
    assert diagnostics["skin_deformers"] == 1
    assert diagnostics["clusters"] == 1
    assert diagnostics["legacy_subdeformer_clusters"] == 0


def test_unity_import_manifest_reports_missing_skin_warning():
    transfer = {
        "source": {
            "game": "K1",
            "resref": "n_darthmalak",
            "character_mode": "creature",
        },
        "unity": {
            "asset_path": "Assets/KotorImported/Test/n_darthmalak.fbx",
        },
        "counts": {
            "animations": 2,
        },
        "animations": ["pause1", "tlknorm"],
    }
    unity_summary = {
        "asset_path": "Assets/KotorImported/Test/n_darthmalak.fbx",
        "clips": [{"name": "pause1", "length": 1.0}],
        "renderers": [{"type": "MeshRenderer", "materialCount": 1}],
    }

    manifest = build_unity_import_manifest(transfer, unity_summary)

    assert manifest["status"] == "warning"
    assert manifest["ok"] is True
    assert manifest["counts"]["mesh_renderers"] == 1
    assert manifest["counts"]["skinned_mesh_renderers"] == 0
    assert "tlknorm" in manifest["missing_clips"]
    assert {item["code"] for item in manifest["warnings"]} >= {
        "missing_skinned_renderer",
        "missing_animation_clips",
    }


def test_unity_import_manifest_errors_on_bad_fbx_skin_diagnostics():
    transfer = {
        "source": {"game": "K1", "resref": "n_darthmalak", "character_mode": "creature"},
        "unity": {"asset_path": "Assets/KotorImported/Test/n_darthmalak.fbx"},
        "counts": {"animations": 1},
        "animations": ["pause1"],
        "fbx": {
            "checked": True,
            "ok": False,
            "duplicate_object_ids": {"42": 2},
        },
    }
    unity_summary = {
        "asset_path": "Assets/KotorImported/Test/n_darthmalak.fbx",
        "clips": [{"name": "pause1"}],
        "renderers": [{"type": "SkinnedMeshRenderer", "materialCount": 1, "boneCount": 1, "bindposeCount": 1}],
    }

    manifest = build_unity_import_manifest(transfer, unity_summary)

    assert manifest["status"] == "error"
    assert manifest["ok"] is False
    assert {item["code"] for item in manifest["errors"]} == {"fbx_skin_object_error"}


def test_mcp_unity_import_validator_writes_manifest():
    out_root = Path(".pytest_tmp_unity_bridge") / "validator"
    sidecar = out_root / "n_darthmalak.ghostrigger.json"
    manifest_path = out_root / "validation.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({
        "source": {"game": "K1", "resref": "n_darthmalak", "character_mode": "creature"},
        "unity": {"asset_path": "Assets/KotorImported/Test/n_darthmalak.fbx"},
        "counts": {"animations": 1},
        "animations": ["pause1"],
    }), encoding="utf-8")

    try:
        result = asyncio.run(handle_tool(
            "ghostrigger_validate_unity_import",
            {
                "transfer_metadata_path": str(sidecar),
                "unity_summary": {
                    "asset_path": "Assets/KotorImported/Test/n_darthmalak.fbx",
                    "clips": [{"name": "pause1", "length": 1.0}],
                    "renderers": [{
                        "type": "SkinnedMeshRenderer",
                        "materialCount": 2,
                        "boneCount": 14,
                        "bindposeCount": 14,
                    }],
                },
                "output_path": str(manifest_path),
            },
        ))
        payload = json.loads(result["text"])

        assert payload["status"] == "ok"
        assert payload["counts"]["skinned_mesh_renderers"] == 1
        assert manifest_path.exists()
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


def test_fbx_export_merges_duplicate_bone_map_clusters(tmp_path):
    root = ModelNode(name="root", flags=int(NodeFlags.HEADER))
    skin = ModelNode(
        name="torso",
        flags=int(NodeFlags.MESH | NodeFlags.SKIN),
        parent=root,
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        normals=[(0.0, 0.0, 1.0)] * 3,
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        faces=[(0, 1, 2)],
        texture="torso",
        bone_map=["root", "root"],
        skin_data=[
            VertexSkinData([BoneWeight(0, 0.25), BoneWeight(1, 0.75)]),
            VertexSkinData([BoneWeight(1, 1.0)]),
            VertexSkinData([BoneWeight(0, 1.0)]),
        ],
    )
    root.children.append(skin)
    model = KotorModel(name="duplicate_cluster_case", root_node=root)
    out_path = tmp_path / "duplicate_cluster_case.fbx"

    assert FBXExporter()._export_fbx_ascii(model, str(out_path))
    text = out_path.read_text(encoding="utf-8")

    assert text.count('"SubDeformer::root", "Cluster"') == 1
    assert "Indexes: *3" in text
    assert "\ta: 0,1,2" in text


def test_fbx_export_uses_skeleton_alias_when_mesh_node_is_bone(tmp_path):
    root = ModelNode(name="root", flags=int(NodeFlags.HEADER))
    renderable_bone = ModelNode(
        name="RArm",
        flags=int(NodeFlags.MESH),
        parent=root,
        vertices=[(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0)],
        faces=[(0, 1, 2)],
        texture="arm",
    )
    skin = ModelNode(
        name="Legs",
        flags=int(NodeFlags.MESH | NodeFlags.SKIN),
        parent=root,
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
        texture="legs",
        bone_map=["RArm"],
        skin_data=[
            VertexSkinData([BoneWeight(0, 1.0)]),
            VertexSkinData([BoneWeight(0, 1.0)]),
            VertexSkinData([BoneWeight(0, 1.0)]),
        ],
    )
    root.children.extend([renderable_bone, skin])
    model = KotorModel(name="mesh_bone_alias_case", root_node=root)
    out_path = tmp_path / "mesh_bone_alias_case.fbx"

    assert FBXExporter()._export_fbx_ascii(model, str(out_path))
    text = out_path.read_text(encoding="utf-8")

    mesh_model_id = re.search(r'Model: (\d+), "Model::RArm", "Mesh"', text).group(1)
    alias_model_id = re.search(r'Model: (\d+), "Model::RArm_bone", "LimbNode"', text).group(1)
    cluster_id = re.search(r'Deformer: (\d+), "SubDeformer::RArm", "Cluster"', text).group(1)
    assert f'C: "OO",{alias_model_id},{cluster_id}' in text
    assert f'C: "OO",{mesh_model_id},{cluster_id}' not in text
    assert '"NodeAttribute::RArm_bone", "LimbNode"' in text


def test_fbx_export_animates_skeleton_alias_when_mesh_node_is_bone(tmp_path):
    root = ModelNode(name="root", flags=int(NodeFlags.HEADER))
    renderable_bone = ModelNode(
        name="RArm",
        flags=int(NodeFlags.MESH),
        parent=root,
        vertices=[(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0)],
        faces=[(0, 1, 2)],
        texture="arm",
    )
    skin = ModelNode(
        name="Legs",
        flags=int(NodeFlags.MESH | NodeFlags.SKIN),
        parent=root,
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
        texture="legs",
        bone_map=["RArm"],
        skin_data=[
            VertexSkinData([BoneWeight(0, 1.0)]),
            VertexSkinData([BoneWeight(0, 1.0)]),
            VertexSkinData([BoneWeight(0, 1.0)]),
        ],
    )
    root.children.extend([renderable_bone, skin])
    anim_node = ModelNode(
        name="RArm",
        controllers=[
            {
                "type": 20,
                "times": [0.0, 1.0],
                "values": [(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.7071068, 0.7071068)],
            }
        ],
    )
    model = KotorModel(
        name="mesh_bone_animation_alias_case",
        root_node=root,
        animations=[Animation(name="turn", length=1.0, anim_root="root", nodes=[anim_node])],
    )
    out_path = tmp_path / "mesh_bone_animation_alias_case.fbx"

    assert FBXExporter()._export_fbx_ascii(model, str(out_path))
    text = out_path.read_text(encoding="utf-8")

    mesh_model_id = re.search(r'Model: (\d+), "Model::RArm", "Mesh"', text).group(1)
    alias_model_id = re.search(r'Model: (\d+), "Model::RArm_bone", "LimbNode"', text).group(1)
    assert re.search(rf'C: "OP",\d+,{alias_model_id},"Lcl Rotation"', text)
    assert not re.search(rf'C: "OP",\d+,{mesh_model_id},"Lcl Rotation"', text)


def test_fbx_export_aliases_animated_rigid_mesh_without_skin_reference(tmp_path):
    root = ModelNode(name="root", flags=int(NodeFlags.HEADER))
    eye = ModelNode(
        name="Rancor_eyeL",
        flags=int(NodeFlags.MESH),
        parent=root,
        position=(0.1, 0.2, 0.3),
        vertices=[(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0)],
        uvs=[(0.25, 0.10), (0.75, 0.20), (0.50, 0.90)],
        faces=[(0, 1, 2)],
        texture="eye",
    )
    root.children.append(eye)
    anim_node = ModelNode(
        name="Rancor_eyeL",
        controllers=[
            {
                "type": 8,
                "times": [0.0, 1.0],
                "values": [(0.0, 0.0, 0.0), (0.0, 0.0, 0.1)],
            }
        ],
    )
    model = KotorModel(
        name="animated_rigid_mesh_alias_case",
        root_node=root,
        animations=[Animation(name="blink", length=1.0, anim_root="root", nodes=[anim_node])],
    )
    out_path = tmp_path / "animated_rigid_mesh_alias_case.fbx"

    assert FBXExporter()._export_fbx_ascii(model, str(out_path))
    text = out_path.read_text(encoding="utf-8")

    mesh_model_id = re.search(r'Model: (\d+), "Model::Rancor_eyeL", "Mesh"', text).group(1)
    alias_model_id = re.search(r'Model: (\d+), "Model::Rancor_eyeL_bone", "LimbNode"', text).group(1)
    mesh_block = re.search(
        r'Model: \d+, "Model::Rancor_eyeL", "Mesh" \{(.*?)\n\t\}',
        text,
        re.S,
    ).group(1)

    assert f'C: "OO",{mesh_model_id},{alias_model_id}' in text
    assert re.search(rf'C: "OP",\d+,{alias_model_id},"Lcl Translation"', text)
    assert not re.search(rf'C: "OP",\d+,{mesh_model_id},"Lcl Translation"', text)
    assert 'P: "Lcl Translation","Lcl Translation","","A",0.000000,0.000000,0.000000' in mesh_block
    assert "a: 0.250000,0.900000,0.750000,0.800000,0.500000,0.100000" in text


def test_fbx_export_rotation_curves_are_absolute_local_values(tmp_path):
    root = ModelNode(name="root", flags=int(NodeFlags.HEADER))
    bone = ModelNode(
        name="jaw",
        flags=0,
        parent=root,
        rotation=(0.0, 0.0, 0.3826834, 0.9238795),
    )
    root.children.append(bone)
    anim_node = ModelNode(
        name="jaw",
        controllers=[
            {
                "type": 20,
                "times": [0.0, 1.0],
                "values": [
                    (0.0, 0.0, 0.3826834, 0.9238795),
                    (0.0, 0.0, 0.3826834, 0.9238795),
                ],
            }
        ],
    )
    model = KotorModel(
        name="absolute_rotation_curve_case",
        root_node=root,
        animations=[Animation(name="hold", length=1.0, anim_root="root", nodes=[anim_node])],
    )
    out_path = tmp_path / "absolute_rotation_curve_case.fbx"

    assert FBXExporter()._export_fbx_ascii(model, str(out_path))
    text = out_path.read_text(encoding="utf-8")

    bone_model_id = re.search(r'Model: (\d+), "Model::jaw", "LimbNode"', text).group(1)

    assert re.search(rf'C: "OP",\d+,{bone_model_id},"Lcl Rotation"', text)
    assert "a: 44.999998,44.999998" in text or "a: 45.000000,45.000000" in text


def test_fbx_export_root_skin_influence_is_limb_node(tmp_path):
    root = ModelNode(name="c_rancorS", flags=int(NodeFlags.HEADER))
    skin = ModelNode(
        name="Legs",
        flags=int(NodeFlags.MESH | NodeFlags.SKIN),
        parent=root,
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
        texture="legs",
        bone_map=["c_rancorS"],
        skin_data=[
            VertexSkinData([BoneWeight(0, 1.0)]),
            VertexSkinData([BoneWeight(0, 1.0)]),
            VertexSkinData([BoneWeight(0, 1.0)]),
        ],
    )
    root.children.append(skin)
    model = KotorModel(name="root_influence_case", root_node=root)
    out_path = tmp_path / "root_influence_case.fbx"

    assert FBXExporter()._export_fbx_ascii(model, str(out_path))
    text = out_path.read_text(encoding="utf-8")

    root_model_id = re.search(r'Model: (\d+), "Model::c_rancorS", "LimbNode"', text).group(1)
    cluster_id = re.search(r'Deformer: (\d+), "SubDeformer::c_rancorS", "Cluster"', text).group(1)
    assert f'C: "OO",{root_model_id},{cluster_id}' in text
    assert '"NodeAttribute::c_rancorS", "LimbNode"' in text


def test_fbx_export_keeps_every_cluster_ancestor_in_the_skeleton_chain(tmp_path):
    root = ModelNode(name="creature_root", flags=int(NodeFlags.HEADER))
    cutscene = ModelNode(
        name="cutscenedummy",
        flags=int(NodeFlags.HEADER),
        parent=root,
    )
    rig_root = ModelNode(
        name="rootdummy",
        flags=int(NodeFlags.HEADER),
        parent=cutscene,
    )
    wing = ModelNode(
        name="wing_bone",
        flags=int(NodeFlags.MESH),
        parent=rig_root,
        render=False,
    )
    skin = ModelNode(
        name="creature_skin",
        flags=int(NodeFlags.MESH | NodeFlags.SKIN),
        parent=root,
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
        texture="creature",
        bone_map=["wing_bone"],
        skin_data=[
            VertexSkinData([BoneWeight(0, 1.0)]),
            VertexSkinData([BoneWeight(0, 1.0)]),
            VertexSkinData([BoneWeight(0, 1.0)]),
        ],
    )
    root.children.extend([skin, cutscene])
    cutscene.children.append(rig_root)
    rig_root.children.append(wing)
    model = KotorModel(name="cluster_ancestor_case", root_node=root)
    out_path = tmp_path / "cluster_ancestor_case.fbx"

    assert FBXExporter()._export_fbx_ascii(
        model,
        str(out_path),
        compatibility_profile="unreal",
    )
    text = out_path.read_text(encoding="utf-8")

    for node_name in ("creature_root", "cutscenedummy", "rootdummy", "wing_bone"):
        assert f'"Model::{node_name}", "LimbNode"' in text

    assert text.index("Connections:  {") < text.index("Takes:  {")


def test_rigging_sidecars_publish_coordinate_and_animation_semantics(tmp_path):
    import json
    from converters.mesh_converter import _export_rigging_data

    root = ModelNode(name="creature_root", flags=int(NodeFlags.HEADER))
    model = KotorModel(name="semantic_creature", root_node=root)
    model.anim_scale = 1.75
    model.animations = [
        SimpleNamespace(
            name="idle",
            length=1.0,
            transition_time=0.25,
            anim_root="creature_root",
            events=[],
            nodes=[SimpleNamespace(name="creature_root", controllers=[])],
        )
    ]

    assert _export_rigging_data(model, tmp_path) == 2
    rig_dir = tmp_path / "rigging"
    skeleton = json.loads((rig_dir / "semantic_creature.skeleton.json").read_text())
    animation = json.loads((rig_dir / "semantic_creature.idle.anim.json").read_text())

    assert skeleton["schema"] == "ghostrigger.kotor.rigging.v2"
    assert skeleton["animation_scale"] == 1.75
    assert skeleton["coordinate_system"]["linear_unit"] == "meter"
    assert animation["schema"] == "ghostrigger.kotor.animation.v2"
    assert animation["animation_scale"] == 1.75
    assert animation["position_controller_semantics"] == "bind_plus_scaled_delta"
    assert animation["rotation_controller_semantics"] == "absolute_local_quaternion_xyzw"


def test_fbx_export_scales_tiny_finger_bone_display_size(tmp_path):
    root = ModelNode(name="root", flags=int(NodeFlags.HEADER))
    hand = ModelNode(name="Ran_handR", flags=int(NodeFlags.HEADER), parent=root, position=(1.0, 0.0, 0.0))
    finger_1 = ModelNode(
        name="Ran_Index_01_R",
        flags=int(NodeFlags.HEADER),
        parent=hand,
        position=(0.09, -0.46, 0.02),
    )
    finger_2 = ModelNode(
        name="Ran_Index_02_R",
        flags=int(NodeFlags.HEADER),
        parent=finger_1,
        position=(0.0, -0.25, 0.0),
    )
    skin = ModelNode(
        name="RArm",
        flags=int(NodeFlags.MESH | NodeFlags.SKIN),
        parent=root,
        vertices=[(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0)],
        faces=[(0, 1, 2)],
        texture="arm",
        bone_map=["Ran_Index_01_R"],
        skin_data=[
            VertexSkinData([BoneWeight(0, 1.0)]),
            VertexSkinData([BoneWeight(0, 1.0)]),
            VertexSkinData([BoneWeight(0, 1.0)]),
        ],
    )
    root.children.extend([hand, skin])
    hand.children.append(finger_1)
    finger_1.children.append(finger_2)
    model = KotorModel(name="finger_display_size_case", root_node=root)
    out_path = tmp_path / "finger_display_size_case.fbx"

    assert FBXExporter()._export_fbx_ascii(model, str(out_path))
    text = out_path.read_text(encoding="utf-8")

    finger_attr = re.search(
        r'NodeAttribute: \d+, "NodeAttribute::Ran_Index_01_R", "LimbNode" \{(.*?)\n\t\}',
        text,
        re.S,
    ).group(1)
    size_match = re.search(r'P: "Size", "double", "Number", "",([0-9.]+)', finger_attr)
    limb_match = re.search(r'P: "LimbLength", "double", "Number", "",([0-9.]+)', finger_attr)

    assert size_match is not None
    assert limb_match is not None
    assert float(size_match.group(1)) < 0.35
    assert float(limb_match.group(1)) == float(size_match.group(1))


def test_fbx_export_parents_rigid_mesh_bone_under_alias_at_identity(tmp_path):
    root = ModelNode(name="root", flags=int(NodeFlags.HEADER))
    jaw = ModelNode(
        name="Ran_Jaw",
        flags=int(NodeFlags.MESH),
        parent=root,
        position=(1.0, 2.0, 3.0),
        vertices=[(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0)],
        faces=[(0, 1, 2)],
        texture="jaw",
    )
    skin = ModelNode(
        name="torso",
        flags=int(NodeFlags.MESH | NodeFlags.SKIN),
        parent=root,
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
        texture="torso",
        bone_map=["Ran_Jaw"],
        skin_data=[
            VertexSkinData([BoneWeight(0, 1.0)]),
            VertexSkinData([BoneWeight(0, 1.0)]),
            VertexSkinData([BoneWeight(0, 1.0)]),
        ],
    )
    root.children.extend([jaw, skin])
    model = KotorModel(name="rigid_mesh_bone_parent_case", root_node=root)
    out_path = tmp_path / "rigid_mesh_bone_parent_case.fbx"

    assert FBXExporter()._export_fbx_ascii(model, str(out_path))
    text = out_path.read_text(encoding="utf-8")

    mesh_model_id = re.search(r'Model: (\d+), "Model::Ran_Jaw", "Mesh"', text).group(1)
    alias_model_id = re.search(r'Model: (\d+), "Model::Ran_Jaw_bone", "LimbNode"', text).group(1)
    mesh_block = re.search(
        r'Model: \d+, "Model::Ran_Jaw", "Mesh" \{(.*?)\n\t\}',
        text,
        re.S,
    ).group(1)
    alias_block = re.search(
        r'Model: \d+, "Model::Ran_Jaw_bone", "LimbNode" \{(.*?)\n\t\}',
        text,
        re.S,
    ).group(1)

    assert f'C: "OO",{mesh_model_id},{alias_model_id}' in text
    assert 'P: "Lcl Translation","Lcl Translation","","A",0.000000,0.000000,0.000000' in mesh_block
    assert 'P: "Lcl Rotation","Lcl Rotation","","A",0.0000,0.0000,0.0000' in mesh_block
    assert 'P: "Lcl Translation","Lcl Translation","","A",1.000000,2.000000,3.000000' in alias_block


def test_fbx_export_references_texture_sidecar_paths(tmp_path):
    root = ModelNode(name="root", flags=int(NodeFlags.HEADER))
    mesh = ModelNode(
        name="torso",
        flags=int(NodeFlags.MESH),
        parent=root,
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
        texture="c_rancor01",
    )
    root.children.append(mesh)
    model = KotorModel(name="texture_sidecar_case", root_node=root)
    out_path = tmp_path / "texture_sidecar_case.fbx"

    assert FBXExporter()._export_fbx_ascii(
        model,
        str(out_path),
        texture_paths={"c_rancor01": "textures/c_rancor01.png"},
    )
    text = out_path.read_text(encoding="utf-8")

    assert 'Filename: "textures/c_rancor01.png"' in text
    assert 'RelativeFilename: "textures/c_rancor01.png"' in text


def test_fbx_export_writes_texture_sidecar_files(tmp_path):
    class _FakeImage:
        mode = "RGBA"

        def save(self, path):
            Path(path).write_bytes(b"fake-png")

    class _FakeTextureCache:
        def get(self, name):
            return _FakeImage() if name == "c_rancor01" else None

    root = ModelNode(name="root", flags=int(NodeFlags.HEADER))
    mesh = ModelNode(
        name="torso",
        flags=int(NodeFlags.MESH),
        parent=root,
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
        texture="c_rancor01",
    )
    root.children.append(mesh)
    model = KotorModel(name="texture_write_case", root_node=root)

    paths = FBXExporter._export_fbx_textures_to_dir(
        model,
        tmp_path / "textures",
        _FakeTextureCache(),
        tmp_path,
    )

    assert paths == {"c_rancor01": "textures/c_rancor01.png"}
    assert (tmp_path / "textures" / "c_rancor01.png").read_bytes() == b"fake-png"


def test_fbx_export_aliases_mesh_bones_case_insensitively(tmp_path):
    root = ModelNode(name="root", flags=int(NodeFlags.HEADER))
    eye = ModelNode(
        name="Rancor_eyeL",
        flags=int(NodeFlags.MESH),
        parent=root,
        position=(0.1, 0.2, 0.3),
        vertices=[(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0)],
        faces=[(0, 1, 2)],
        texture="eye",
    )
    skin = ModelNode(
        name="torso",
        flags=int(NodeFlags.MESH | NodeFlags.SKIN),
        parent=root,
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
        texture="torso",
        bone_map=["Rancor_EyeL"],
        skin_data=[
            VertexSkinData([BoneWeight(0, 1.0)]),
            VertexSkinData([BoneWeight(0, 1.0)]),
            VertexSkinData([BoneWeight(0, 1.0)]),
        ],
    )
    root.children.extend([eye, skin])
    anim_node = ModelNode(
        name="Rancor_eyeL",
        controllers=[
            {
                "type": 8,
                "times": [0.0, 1.0],
                "values": [(0.0, 0.0, 0.0), (0.0, 0.0, 0.1)],
            }
        ],
    )
    model = KotorModel(
        name="case_insensitive_mesh_bone_alias_case",
        root_node=root,
        animations=[Animation(name="blink", length=1.0, anim_root="root", nodes=[anim_node])],
    )
    out_path = tmp_path / "case_insensitive_mesh_bone_alias_case.fbx"

    assert FBXExporter()._export_fbx_ascii(model, str(out_path))
    text = out_path.read_text(encoding="utf-8")

    mesh_model_id = re.search(r'Model: (\d+), "Model::Rancor_eyeL", "Mesh"', text).group(1)
    alias_model_id = re.search(r'Model: (\d+), "Model::Rancor_eyeL_bone", "LimbNode"', text).group(1)
    cluster_id = re.search(r'Deformer: (\d+), "SubDeformer::Rancor_EyeL", "Cluster"', text).group(1)
    assert f'C: "OO",{alias_model_id},{cluster_id}' in text
    assert f'C: "OO",{mesh_model_id},{cluster_id}' not in text
    assert re.search(rf'C: "OP",\d+,{alias_model_id},"Lcl Translation"', text)
    assert not re.search(rf'C: "OP",\d+,{mesh_model_id},"Lcl Translation"', text)


def test_fbx_export_writes_kotor_unreal_sidecar_manifest(tmp_path):
    root = ModelNode(name="root", flags=int(NodeFlags.HEADER))
    head_hook = ModelNode(name="head_g", flags=int(NodeFlags.HEADER), parent=root)
    skin = ModelNode(
        name="torso",
        flags=int(NodeFlags.MESH | NodeFlags.SKIN),
        parent=root,
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        normals=[(0.0, 0.0, 1.0)] * 3,
        tangents=[(1.0, 0.0, 0.0)] * 3,
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        uvs_lm=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        faces=[(0, 1, 2)],
        texture="torso_diff",
        lightmap="torso_lm",
        bump_map="torso_n",
        texture_names=["torso_diff", "torso_lm"],
        bone_map=["root", "head_g"],
        qbone_list=[(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)],
        tbone_list=[(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        skin_data=[
            VertexSkinData([BoneWeight(0, 1.0)]),
            VertexSkinData([BoneWeight(0, 0.5), BoneWeight(1, 0.5)]),
            VertexSkinData([BoneWeight(1, 1.0)]),
        ],
    )
    root.children.extend([head_hook, skin])
    model = KotorModel(name="manifest_case", root_node=root, supermodel="S_MALE02")
    out_path = tmp_path / "manifest_case.fbx"

    assert FBXExporter().export(model, str(out_path), export_rigging=False)
    manifest_path = out_path.with_suffix(".ghostrigger.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema"] == "ghostrigger.kotor_fbx_manifest.v1"
    assert manifest["source"]["supermodel"] == "S_MALE02"
    assert manifest["coordinate_system"]["fbx_axis"] == "Z-up, -Y forward, +X right"
    assert manifest["counts"]["exported_mesh_nodes"] == 1
    assert manifest["hooks"] == [{"name": "head_g", "role": "head_attachment_geometry", "parent": "root"}]
    assert manifest["materials"][0]["diffuse"] == "torso_diff"
    assert manifest["materials"][0]["lightmap"] == "torso_lm"
    assert manifest["materials"][0]["bump_map"] == "torso_n"
    assert manifest["meshes"][0]["skin"]["qbone_count"] == 2
    assert manifest["meshes"][0]["skin"]["tbone_count"] == 2
    assert manifest["skeleton"]["skin"]["missing_bones"] == []
    assert manifest["validation"]["ok"] is True
    assert manifest["fbx"]["checked"] is True
    assert manifest["fbx"]["exporter_backend"] == "builtin_ascii"
    assert manifest["unreal"]["recommended_import"]["skeletal_mesh"] is True


def test_unity_compatibility_profile_declares_meters_and_continuous_linear_animation(tmp_path):
    root = ModelNode(name="root", flags=int(NodeFlags.HEADER))
    bone = ModelNode(name="jaw", flags=int(NodeFlags.HEADER), parent=root)
    skin = ModelNode(
        name="face",
        flags=int(NodeFlags.MESH | NodeFlags.SKIN),
        parent=root,
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        normals=[(0.0, 0.0, 1.0)] * 3,
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        faces=[(0, 1, 2)],
        texture="face_diff",
        bone_map=["jaw"],
        skin_data=[VertexSkinData([BoneWeight(0, 1.0)]) for _ in range(3)],
    )
    root.children.extend([bone, skin])

    def _z_quat(degrees):
        half = math.radians(degrees) * 0.5
        return (0.0, 0.0, math.sin(half), math.cos(half))

    anim_node = ModelNode(
        name="jaw",
        controllers=[{
            "type": 20,
            "times": [0.0, 1.0],
            "values": [_z_quat(179.0), _z_quat(-179.0)],
        }],
    )
    model = KotorModel(
        name="unity_profile_case",
        root_node=root,
        animations=[Animation(name="talk", length=1.0, anim_root="root", nodes=[anim_node])],
    )
    out_path = tmp_path / "unity_profile_case.fbx"

    assert FBXExporter().export(
        model,
        str(out_path),
        export_rigging=False,
        compatibility_profile="unity",
    )
    text = out_path.read_text(encoding="utf-8")
    manifest = json.loads(out_path.with_suffix(".ghostrigger.json").read_text(encoding="utf-8"))

    assert 'P: "UnitScaleFactor", "double", "Number", "",100' in text
    assert "KeyAttrFlags: *1" in text
    assert "\n\t\t\ta: 4\n" in text
    assert "179.000000,181.000000" in text
    assert 'AnimationStack:' in text and '"AnimStack::talk"' in text
    assert '"AnimStack::|talk"' not in text
    assert manifest["fbx"]["compatibility_profile"] == "unity"
    assert manifest["unity"]["recommended_import"]["convert_units"] is True
    assert "1 meter" in manifest["coordinate_system"]["unit_scale"]


def test_unreal_compatibility_profile_embeds_engine_curves_and_preserves_global_cluster_bind(tmp_path):
    root = ModelNode(name="root", flags=int(NodeFlags.HEADER))
    bone = ModelNode(
        name="jaw",
        flags=int(NodeFlags.HEADER),
        parent=root,
        position=(0.2, 0.3, 0.4),
    )
    skin = ModelNode(
        name="face",
        flags=int(NodeFlags.MESH | NodeFlags.SKIN),
        parent=root,
        position=(1.0, 2.0, 3.0),
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        normals=[(0.0, 0.0, 1.0)] * 3,
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        faces=[(0, 1, 2)],
        texture="face_diff",
        bone_map=["jaw"],
        skin_data=[VertexSkinData([BoneWeight(0, 1.0)]) for _ in range(3)],
    )
    root.children.extend([bone, skin])
    anim_node = ModelNode(
        name="jaw",
        controllers=[{
            "type": 8,
            "times": [0.0, 1.0],
            "values": [(0.0, 0.0, 0.0), (0.0, 0.1, 0.0)],
        }],
    )
    model = KotorModel(
        name="unreal_profile_case",
        root_node=root,
        animations=[Animation(name="talk", length=1.0, anim_root="root", nodes=[anim_node])],
    )
    model._gr_fbx_animation_selection = {
        "selected": ["talk"],
        "requested": ["talk", "missing_clip"],
        "embedded": ["talk"],
        "missing": ["missing_clip"],
        "sources": {"talk": "s_male02"},
        "scales": {"talk": 1.25},
    }
    out_path = tmp_path / "unreal_profile_case.fbx"

    assert FBXExporter().export(
        model,
        str(out_path),
        export_rigging=False,
        compatibility_profile="ue5",
    )
    text = out_path.read_text(encoding="utf-8")
    manifest = json.loads(out_path.with_suffix(".ghostrigger.json").read_text(encoding="utf-8"))

    assert "; Compatibility profile: unreal" in text
    assert 'P: "UnitScaleFactor", "double", "Number", "",100' in text
    assert "KeyAttrFlags: *1" in text
    assert "\n\t\t\ta: 4\n" in text
    assert '"AnimStack::|talk"' in text
    cluster = re.search(
        r'Deformer: \d+, "SubDeformer::jaw", "Cluster" \{.*?'
        r'Transform: \*16 \{\s*a: ([^\r\n]+).*?'
        r'TransformLink: \*16 \{\s*a: ([^\r\n]+)',
        text,
        re.S,
    )
    assert cluster is not None
    transform = [float(value) for value in cluster.group(1).split(",")]
    transform_link = [float(value) for value in cluster.group(2).split(",")]
    assert transform[12:15] == [1.0, 2.0, 3.0]
    assert transform_link[12:15] == [0.2, 0.3, 0.4]

    assert manifest["fbx"]["compatibility_profile"] == "unreal"
    assert manifest["coordinate_system"]["unit_scale"].startswith("1 KOTOR unit declared as 1 meter")
    selection = manifest["animation_selection"]
    assert selection["selected"] == ["talk"]
    assert selection["requested"] == ["talk", "missing_clip"]
    assert selection["embedded"] == ["talk"]
    assert selection["missing"] == ["missing_clip"]
    assert selection["sources"] == {"talk": "s_male02"}
    assert selection["scales"] == {"talk": 1.25}
    handoff = manifest["unreal"]
    assert handoff["recommended_import"]["sample_rate_fps"] == 30
    assert handoff["recommended_import"]["animation_interpolation"] == "Linear"
    assert handoff["recommended_import"]["preserve_native_kotor_skeleton"] is True
    assert handoff["recommended_import"]["automatic_quinn_retarget"] is False
    assert handoff["recommended_import"]["import_textures"] is True
    assert handoff["recommended_import"]["import_meshes_in_bone_hierarchy"] is True
    assert any("does not silently retarget" in note for note in handoff["notes"])


def test_unity_cluster_bind_uses_emitted_hierarchy_not_legacy_collapsed_world_transform(tmp_path):
    root = ModelNode(name="root", flags=int(NodeFlags.HEADER))
    # The generic viewport-oriented world_transform intentionally collapses a
    # near-pure X 180-degree parent. FBX emits the rotation, so its bind pose
    # must compose the exact emitted hierarchy instead.
    parent_rotation = (
        0.9993784703983151,
        0.0021312797225932945,
        0.034913866595109005,
        0.004376353555986939,
    )
    parent = ModelNode(
        name="finger_base",
        flags=int(NodeFlags.HEADER),
        parent=root,
        position=(0.001277, 0.004095, -0.083374),
        rotation=parent_rotation,
    )
    tip = ModelNode(
        name="finger_tip",
        flags=int(NodeFlags.HEADER),
        parent=parent,
        position=(-0.007361, 0.0, 0.060654),
    )
    skin = ModelNode(
        name="arm",
        flags=int(NodeFlags.MESH | NodeFlags.SKIN),
        parent=root,
        position=(0.0, 0.79, 0.0),
        vertices=[(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0)],
        normals=[(0.0, 0.0, 1.0)] * 3,
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        faces=[(0, 1, 2)],
        texture="arm_diff",
        bone_map=["finger_tip"],
        skin_data=[VertexSkinData([BoneWeight(0, 1.0)]) for _ in range(3)],
    )
    root.children.extend([parent, skin])
    parent.children.append(tip)
    model = KotorModel(name="unity_bind_hierarchy_case", root_node=root)
    out_path = tmp_path / "unity_bind_hierarchy_case.fbx"

    assert FBXExporter()._export_fbx_ascii(
        model,
        str(out_path),
        compatibility_profile="unity",
    )
    text = out_path.read_text(encoding="utf-8")
    cluster = re.search(
        r'Deformer: \d+, "SubDeformer::finger_tip", "Cluster" \{.*?'
        r'Transform: \*16 \{\s*a: ([^\r\n]+).*?'
        r'TransformLink: \*16 \{\s*a: ([^\r\n]+)',
        text,
        re.S,
    )
    assert cluster is not None
    values = [float(value) for value in cluster.group(1).split(",")]
    link_values = [float(value) for value in cluster.group(2).split(",")]

    def _rotate(quat, vector):
        qx, qy, qz, qw = quat
        vx, vy, vz = vector
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)
        return (
            vx + qw * tx + qy * tz - qz * ty,
            vy + qw * ty + qz * tx - qx * tz,
            vz + qw * tz + qx * ty - qy * tx,
        )

    tip_offset = _rotate(parent_rotation, tip.position)
    bone_world = tuple(parent.position[index] + tip_offset[index] for index in range(3))
    mesh_world = skin.position
    delta = tuple(mesh_world[index] - bone_world[index] for index in range(3))
    inverse_rotation = (
        -parent_rotation[0],
        -parent_rotation[1],
        -parent_rotation[2],
        parent_rotation[3],
    )
    expected_translation = _rotate(inverse_rotation, delta)

    rotated_axes = [
        _rotate(parent_rotation, axis)
        for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    ]
    bone_rotation = tuple(
        tuple(rotated_axes[column][row] for column in range(3))
        for row in range(3)
    )
    expected_raw_rotation = [
        bone_rotation[row][column]
        for row in range(3)
        for column in range(3)
    ]
    actual_raw_rotation = values[0:3] + values[4:7] + values[8:11]
    actual_link_rotation = link_values[0:3] + link_values[4:7] + link_values[8:11]

    for actual, expected in zip(actual_raw_rotation, expected_raw_rotation):
        assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=2e-6)
    for actual, expected in zip(values[12:15], expected_translation):
        assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=2e-6)
    for actual, expected in zip(actual_link_rotation, expected_raw_rotation):
        assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=2e-6)
    for actual, expected in zip(link_values[12:15], bone_world):
        assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=2e-6)


def test_supermodel_synthetic_bones_include_required_ancestor_chain(tmp_path):
    base_root = ModelNode(
        name="s_male02",
        flags=int(NodeFlags.HEADER),
        position=(0.0, 0.1, 0.0),
    )
    neck = ModelNode(
        name="neck_g",
        flags=int(NodeFlags.HEADER),
        parent=base_root,
        position=(0.0, 1.0, 0.0),
    )
    jaw = ModelNode(
        name="f_jaw_g",
        flags=int(NodeFlags.HEADER),
        parent=neck,
        position=(0.0, 0.2, 0.0),
    )
    base_root.children.append(neck)
    neck.children.append(jaw)
    base = KotorModel(name="s_male02", root_node=base_root)

    root = ModelNode(name="head_root", flags=int(NodeFlags.HEADER))
    skin = ModelNode(
        name="head_skin",
        flags=int(NodeFlags.MESH | NodeFlags.SKIN),
        parent=root,
        vertices=[(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0)],
        normals=[(0.0, 0.0, 1.0)] * 3,
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        faces=[(0, 1, 2)],
        texture="head_diff",
        bone_map=["f_jaw_g"],
        skin_data=[VertexSkinData([BoneWeight(0, 1.0)]) for _ in range(3)],
    )
    root.children.append(skin)
    head = KotorModel(name="standalone_head", root_node=root)
    output = tmp_path / "standalone_head.fbx"

    assert FBXExporter()._export_fbx_ascii(
        head,
        str(output),
        base_skeleton_model=base,
        compatibility_profile="unity",
    )
    text = output.read_text(encoding="utf-8")

    ids = {
        name: int(re.search(rf'Model: (\d+), "Model::{name}"', text).group(1))
        for name in ("head_root", "s_male02", "neck_g", "f_jaw_g")
    }
    assert f'C: "OO",{ids["f_jaw_g"]},{ids["neck_g"]}' in text
    assert f'C: "OO",{ids["neck_g"]},{ids["s_male02"]}' in text
    assert f'C: "OO",{ids["s_male02"]},{ids["head_root"]}' in text
    jaw_block = text.split('"Model::f_jaw_g", "LimbNode" {', 1)[1].split("\n\t}", 1)[0]
    assert 'Lcl Translation","Lcl Translation","","A",0.000000,0.200000,0.000000' in jaw_block


def test_3ds_max_cluster_uses_autodesk_global_bind_matrices(tmp_path):
    root = ModelNode(name="root", flags=int(NodeFlags.HEADER))
    bone = ModelNode(
        name="bone",
        flags=int(NodeFlags.HEADER),
        parent=root,
        position=(0.2, 0.3, 0.4),
    )
    skin = ModelNode(
        name="skin",
        flags=int(NodeFlags.MESH | NodeFlags.SKIN),
        parent=root,
        position=(1.0, 2.0, 3.0),
        vertices=[(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0)],
        normals=[(0.0, 0.0, 1.0)] * 3,
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        faces=[(0, 1, 2)],
        texture="skin_diff",
        bone_map=["bone"],
        skin_data=[VertexSkinData([BoneWeight(0, 1.0)]) for _ in range(3)],
    )
    root.children.extend([bone, skin])
    model = KotorModel(name="max_bind_contract", root_node=root)
    output = tmp_path / "max_bind_contract.fbx"

    assert FBXExporter()._export_fbx_ascii(
        model,
        str(output),
        compatibility_profile="3ds_max",
    )
    text = output.read_text(encoding="utf-8")
    cluster = re.search(
        r'Deformer: \d+, "SubDeformer::bone", "Cluster" \{.*?'
        r'Transform: \*16 \{\s*a: ([^\r\n]+).*?'
        r'TransformLink: \*16 \{\s*a: ([^\r\n]+)',
        text,
        re.S,
    )
    assert cluster is not None
    transform = [float(value) for value in cluster.group(1).split(",")]
    transform_link = [float(value) for value in cluster.group(2).split(",")]
    assert transform[12:15] == [1.0, 2.0, 3.0]
    assert transform_link[12:15] == [0.2, 0.3, 0.4]


def test_fbx_export_dialog_filter_maps_compatibility_profiles():
    from src.gui.windows.application_core.shared.model_io import _fbx_compatibility_profile_from_filter

    assert _fbx_compatibility_profile_from_filter("Unity-Compatible FBX (*.fbx)") == "unity"
    assert _fbx_compatibility_profile_from_filter("Unreal Engine-Compatible FBX (*.fbx)") == "unreal"
    assert _fbx_compatibility_profile_from_filter("3ds Max-Compatible FBX (*.fbx)") == "3ds_max"
    assert _fbx_compatibility_profile_from_filter("Standard FBX (*.fbx)") == "standard"


def test_model_io_resolves_base_skeleton_through_main_window_resource_manager():
    from src.gui.windows.application_core.shared.model_io import ModelIoMixin

    base_skeleton = object()
    calls = []

    class _Manager:
        def load_model(self, resref, game):
            calls.append((resref, game))
            return base_skeleton

    class _Harness(ModelIoMixin):
        _current_game = "K1"

        def _get_resource_manager(self):
            return _Manager()

    model = SimpleNamespace(supermodel="S_Male02")
    assert _Harness()._fbx_base_skeleton_for_export(model) is base_skeleton
    assert calls == [("S_Male02", "K1")]


def test_model_io_reuses_animation_browser_supermodel_cache_before_fbx_selector():
    from src.core.animation.animation_engine import SuperModelResolver
    from src.gui.windows.application_core.shared.model_io import ModelIoMixin

    base_skeleton = SimpleNamespace(name="S_Male02", supermodel="NULL")
    calls = []

    class _Manager:
        revision = 1

        def load_model_strict(self, resref, game):
            calls.append((resref, game))
            return base_skeleton

    manager = _Manager()

    class _Harness(ModelIoMixin):
        _current_game = "K1"

        def _get_resource_manager(self):
            return manager

    SuperModelResolver.clear_cache()
    model = SimpleNamespace(supermodel="S_Male02")
    harness = _Harness()
    assert harness._fbx_base_skeleton_for_export(model) is base_skeleton
    assert harness._fbx_base_skeleton_for_export(model) is base_skeleton
    assert calls == [("S_Male02", "K1")]


def test_model_io_worker_passes_unity_profile_textures_and_base_skeleton(monkeypatch, tmp_path):
    from src.converters.mesh_converter import FBXExporter
    from src.gui.windows.application_core.shared.model_io import _work_export_fbx

    captured = {}
    base_skeleton = object()
    texture_cache = object()
    out_path = tmp_path / "main_window_unity.fbx"

    def _fake_export(self, model, path, **kwargs):
        captured.update(kwargs)
        Path(path).write_text("fbx", encoding="utf-8")
        return True

    monkeypatch.setattr(FBXExporter, "export", _fake_export)
    result = _work_export_fbx(
        _Model(),
        str(out_path),
        tex_cache=texture_cache,
        base_skeleton_model=base_skeleton,
        compatibility_profile="unity",
    )

    assert result == str(out_path)
    assert captured["compatibility_profile"] == "unity"
    assert captured["base_skeleton_model"] is base_skeleton
    assert captured["tex_cache"] is texture_cache
    assert captured["export_rigging"] is True


def test_main_fbx_route_passes_unreal_selection_and_resource_context_to_worker(
    monkeypatch,
    tmp_path,
):
    from src.gui.windows.application_core.shared import model_io

    model = _Model()
    manager = object()
    texture_cache = object()
    base_skeleton = object()
    head_animation_source = object()
    captured = {}
    output = tmp_path / "main_unreal.fbx"

    class _Harness(model_io.ModelIoMixin):
        def _require_model(self, _title):
            return model

        def _get_tex_cache_for_export(self):
            return texture_cache

        def _fbx_base_skeleton_for_export(self, candidate):
            assert candidate is model
            return base_skeleton

        def _fbx_supplemental_animation_models(self, candidate):
            assert candidate is model
            return (head_animation_source,)

        def _choose_fbx_animation_sets(self, candidate, profile, **kwargs):
            captured["chooser"] = (candidate, profile, kwargs)
            return ("walk", "tlknorm")

        def _fbx_resource_context_for_export(self, candidate):
            assert candidate is model
            return manager, "K2"

        def _run_io_async(self, *args, **kwargs):
            captured["worker_args"] = args
            captured["worker_kwargs"] = kwargs

        def _log(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(
        model_io.QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (
            str(output),
            "Unreal Engine-Compatible FBX (*.fbx)",
        ),
    )

    _Harness()._export_fbx()

    assert captured["worker_args"][1] is model_io._work_export_fbx
    assert captured["worker_args"][2] is model
    assert captured["worker_args"][3] == str(output)
    kwargs = captured["worker_kwargs"]
    assert kwargs["compatibility_profile"] == "unreal"
    assert kwargs["selected_animation_names"] == ("walk", "tlknorm")
    assert isinstance(kwargs["selected_animation_names"], tuple)
    assert kwargs["animation_resource_manager"] is manager
    assert kwargs["animation_game"] == "K2"
    assert kwargs["supplemental_animation_models"] == (head_animation_source,)
    assert kwargs["base_skeleton_model"] is base_skeleton
    assert kwargs["tex_cache"] is texture_cache


def test_model_io_animation_selector_uses_shared_dialog_contract_and_returns_tuple(
    monkeypatch,
):
    from src.core.animation import fbx_animation_selection
    from src.gui.qt_lib.dialogs import qt_fbx_animation_selection_dialog
    from src.gui.windows.application_core.shared import model_io

    manager = object()
    rows = (SimpleNamespace(name="walk"), SimpleNamespace(name="tlknorm"))
    captured = {}

    monkeypatch.setattr(
        fbx_animation_selection,
        "list_fbx_animation_sets",
        lambda *args, **kwargs: rows,
    )

    class _Dialog:
        def __init__(
            self,
            dialog_rows,
            parent=None,
            *,
            profile="standard",
            initial_selected_names=None,
            current_animation_name="",
        ):
            captured.update(
                rows=dialog_rows,
                parent=parent,
                profile=profile,
                initial_selected_names=initial_selected_names,
                current_animation_name=current_animation_name,
            )

        def exec(self):
            return model_io.QtWidgets.QDialog.Accepted

        def selected_animation_names(self):
            return ["walk"]

    monkeypatch.setattr(
        qt_fbx_animation_selection_dialog,
        "QtFbxAnimationSelectionDialog",
        _Dialog,
    )

    class _AnimationPanel:
        def selected_animation(self):
            return "walk"

    class _Harness(model_io.ModelIoMixin):
        animations_panel = _AnimationPanel()

        def _fbx_resource_context_for_export(self, _model):
            return manager, "K1"

    harness = _Harness()
    model = SimpleNamespace(name="body", animations=[_Anim("pause1")])
    result = harness._choose_fbx_animation_sets(
        model,
        "unreal",
        base_skeleton_model="base",
        supplemental_models=("head",),
    )

    assert result == ("walk",)
    assert isinstance(result, tuple)
    assert captured["rows"] is rows
    assert captured["parent"] is harness
    assert captured["profile"] == "unreal"
    assert captured["initial_selected_names"] == ("pause1", "walk")
    assert captured["current_animation_name"] == "walk"


def test_character_workflows_pass_unity_profile_to_fbx_exporter(monkeypatch, tmp_path):
    from src.core.characters import headless_body_workflow
    from src.core.workflow import composite_workflow

    calls = []

    class _Fbx:
        def export(self, model, path, **kwargs):
            calls.append((model, path, kwargs))
            Path(path).write_text("fbx", encoding="utf-8")
            return True

    class _Unused:
        pass

    monkeypatch.setattr(
        headless_body_workflow,
        "_import_mesh_exporters",
        lambda: (_Fbx, _Unused, _Unused),
    )
    monkeypatch.setattr(
        composite_workflow,
        "_import_mesh_exporters",
        lambda: (_Fbx, _Unused, _Unused),
    )
    body = _Model()
    base_skeleton = object()
    texture_cache = object()
    body._gr_fbx_base_skeleton_model = base_skeleton

    body_result = headless_body_workflow._export_single_format(
        None,
        body,
        "fbx",
        "FBX",
        str(tmp_path),
        "body",
        "unity",
        texture_cache,
    )
    composite_result = composite_workflow._export_composite_single_format(
        body,
        "fbx",
        "FBX",
        str(tmp_path),
        "composite",
        "unity",
        base_skeleton,
        texture_cache,
    )

    assert body_result.ok is True
    assert composite_result.ok is True
    assert [call[2]["compatibility_profile"] for call in calls] == ["unity", "unity"]
    assert all(call[2]["base_skeleton_model"] is base_skeleton for call in calls)
    assert all(call[2]["tex_cache"] is texture_cache for call in calls)


def test_character_workflows_prepare_exact_selected_animation_sets(monkeypatch, tmp_path):
    from src.core.animation import fbx_animation_selection
    from src.core.characters import headless_body_workflow
    from src.core.workflow import composite_workflow

    exporter_calls = []
    prepare_calls = []
    manager = object()
    base_skeleton = object()
    head_source = object()
    body = _Model()
    body._gr_fbx_base_skeleton_model = base_skeleton
    prepared_headless = SimpleNamespace(name="prepared_headless")
    prepared_composite = SimpleNamespace(name="prepared_composite")

    def _prepare(model, selected_names, **kwargs):
        prepare_calls.append((model, selected_names, kwargs))
        return prepared_headless if len(prepare_calls) == 1 else prepared_composite

    class _Fbx:
        def export(self, model, path, **kwargs):
            exporter_calls.append((model, path, kwargs))
            Path(path).write_text("fbx", encoding="utf-8")
            return True

    class _Unused:
        pass

    monkeypatch.setattr(
        fbx_animation_selection,
        "prepare_fbx_animation_export_model",
        _prepare,
    )
    monkeypatch.setattr(
        headless_body_workflow,
        "_import_mesh_exporters",
        lambda: (_Fbx, _Unused, _Unused),
    )
    monkeypatch.setattr(
        composite_workflow,
        "_import_mesh_exporters",
        lambda: (_Fbx, _Unused, _Unused),
    )

    headless_result = headless_body_workflow._export_single_format(
        SimpleNamespace(game_version="K2"),
        body,
        "fbx",
        "FBX",
        str(tmp_path),
        "selected_body",
        fbx_compatibility_profile="unreal",
        fbx_animation_names=["walk", "tlknorm"],
        animation_resource_manager=manager,
    )
    composite_result = composite_workflow._export_composite_single_format(
        body,
        "fbx",
        "FBX",
        str(tmp_path),
        "mesh_only_composite",
        fbx_compatibility_profile="unity",
        base_skeleton_model=base_skeleton,
        fbx_animation_names=(),
        animation_resource_manager=manager,
        animation_game="K1",
        supplemental_animation_models=(head_source,),
    )

    assert headless_result.ok is True
    assert composite_result.ok is True
    assert prepare_calls[0][0] is body
    assert prepare_calls[0][1] == ("walk", "tlknorm")
    assert isinstance(prepare_calls[0][1], tuple)
    assert prepare_calls[0][2]["game"] == "K2"
    assert prepare_calls[0][2]["resource_manager"] is manager
    assert prepare_calls[0][2]["base_skeleton_model"] is base_skeleton
    assert prepare_calls[1][0] is body
    assert prepare_calls[1][1] == ()
    assert isinstance(prepare_calls[1][1], tuple)
    assert prepare_calls[1][2]["game"] == "K1"
    assert prepare_calls[1][2]["resource_manager"] is manager
    assert prepare_calls[1][2]["base_skeleton_model"] is base_skeleton
    assert prepare_calls[1][2]["supplemental_models"] == (head_source,)
    assert exporter_calls[0][0] is prepared_headless
    assert exporter_calls[0][2]["compatibility_profile"] == "unreal"
    assert exporter_calls[1][0] is prepared_composite
    assert exporter_calls[1][2]["compatibility_profile"] == "unity"


def test_character_export_dialog_exposes_unity_compatibility_mode():
    gui_path = ROOT / "native/GhostRigger.Core.GUI.Display/Python/src/gui/dialogs/qt_export_dialog.py"
    tools_path = ROOT / "native/GhostRigger.Core.Tools/Python/src/gui/dialogs/qt_export_dialog.py"
    source = gui_path.read_text(encoding="utf-8")

    assert '"unity", "Unity-Compatible FBX"' in source
    assert "def fbx_compatibility_profile" in source
    assert gui_path.read_bytes() == tools_path.read_bytes()


def test_character_export_dialog_exposes_unreal_profile_and_animation_selector_contract():
    export_gui = ROOT / "native/GhostRigger.Core.GUI.Display/Python/src/gui/dialogs/qt_export_dialog.py"
    export_tools = ROOT / "native/GhostRigger.Core.Tools/Python/src/gui/dialogs/qt_export_dialog.py"
    selector_gui = ROOT / "native/GhostRigger.Core.GUI.Display/Python/src/gui/dialogs/qt_fbx_animation_selection_dialog.py"
    selector_tools = ROOT / "native/GhostRigger.Core.Tools/Python/src/gui/dialogs/qt_fbx_animation_selection_dialog.py"
    qt_lib = ROOT / "native/GhostRigger.Core.GUI.Display/Python/src/gui/qt_lib.py"

    export_source = export_gui.read_text(encoding="utf-8")
    selector_source = selector_gui.read_text(encoding="utf-8")

    assert '("unreal", "Unreal Engine-Compatible FBX")' in export_source
    assert 'setObjectName("fbxProfileCombo")' in export_source
    assert "class QtFbxAnimationSelectionDialog" in selector_source
    assert 'setHeaderLabels(("Animation", "Source", "Scope", "Duration"))' in selector_source
    assert 'setObjectName("fbxAnimationSetList")' in selector_source
    assert 'setObjectName("fbxAnimationSelectCurrentButton")' in selector_source
    assert 'setObjectName("fbxAnimationSelectLocalButton")' in selector_source
    assert 'setObjectName("fbxAnimationSelectAllButton")' in selector_source
    assert 'setObjectName("fbxAnimationSelectNoneButton")' in selector_source
    assert "mesh and rig only" in selector_source
    assert '"qt_fbx_animation_selection_dialog"' in qt_lib.read_text(encoding="utf-8")
    assert export_gui.read_bytes() == export_tools.read_bytes()
    assert selector_gui.read_bytes() == selector_tools.read_bytes()


def test_fbx_animation_select_all_checks_only_rows_visible_under_filter():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from src.gui.qt_lib.dialogs.qt_fbx_animation_selection_dialog import (
        QtFbxAnimationSelectionDialog,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = QtFbxAnimationSelectionDialog(
        (
            {"name": "walk", "source_model_name": "S_Male02", "scope": "inherited"},
            {"name": "run", "source_model_name": "S_Male02", "scope": "inherited"},
            {"name": "blink", "source_model_name": "P_CarthH", "scope": "supplemental"},
        ),
        profile="unity",
        initial_selected_names=("blink",),
    )

    dialog._search_edit.setText("s_male02")
    app.processEvents()
    dialog._select_all()

    assert dialog.selected_animation_names() == ("walk", "run")
    assert dialog._tree.topLevelItem(0).isHidden() is False
    assert dialog._tree.topLevelItem(1).isHidden() is False
    assert dialog._tree.topLevelItem(2).isHidden() is True
    dialog.close()


def test_character_builder_routes_selected_fbx_animation_sets_in_both_payloads():
    gui_path = ROOT / "native/GhostRigger.Core.GUI.Display/Python/src/gui/panels/qt_character_builder_panel.py"
    tools_path = ROOT / "native/GhostRigger.Core.Tools/Python/src/gui/panels/qt_character_builder_panel.py"
    gui_source = gui_path.read_text(encoding="utf-8")

    required_contracts = (
        "fbx_animation_names = None",
        "QtFbxAnimationSelectionDialog(",
        "profile=fbx_compatibility_profile",
        "fbx_animation_names=fbx_animation_names",
        "animation_resource_manager=animation_resource_manager",
        '"fbx_animation_names": (',
    )
    for contract in required_contracts:
        assert contract in gui_source
    assert gui_path.read_bytes() == tools_path.read_bytes()


def test_mcp_engine_export_schemas_expose_explicit_animation_selection():
    tools = {tool["name"]: tool for tool in get_all_tools()}

    unity_properties = tools["ghostrigger_export_model_for_unity"]["inputSchema"]["properties"]
    unreal = tools["ghostrigger_export_model_for_unreal"]
    unreal_properties = unreal["inputSchema"]["properties"]

    assert unity_properties["animation_names"]["type"] == "array"
    assert unity_properties["animation_names"]["items"] == {"type": "string"}
    assert unreal_properties["animation_names"]["type"] == "array"
    assert unreal_properties["export_textures"]["type"] == "boolean"
    assert unreal_properties["export_textures"]["default"] is True
    assert {"output_path", "export_dir"} <= set(unreal_properties)


def test_installation_texture_cache_decodes_tga_and_reports_resolution():
    from PIL import Image

    from kotormcp.adapters import InstallationTextureCache
    from kotormcp.ports import ResourceEntry

    stream = BytesIO()
    Image.new("RGB", (2, 1), (32, 64, 96)).save(stream, format="TGA")
    resource = ResourceEntry(
        resref="v_ehawk01",
        restype="TGA",
        extension="tga",
        size=len(stream.getvalue()),
        source="test",
        data=stream.getvalue(),
    )
    requested = []
    installation = SimpleNamespace(
        get_texture_resource=lambda name: requested.append(name) or resource
    )

    cache = InstallationTextureCache(installation)
    image = cache.get("V_EHawk01.tga")

    assert image is not None
    assert image.mode == "RGBA"
    assert image.size == (2, 1)
    assert requested == ["V_EHawk01"]
    assert cache.summary()["counts"] == {
        "requested": 1,
        "decoded": 1,
        "missing": 0,
        "errors": 0,
    }


def test_installation_model_locator_returns_mdx_companion_bytes():
    from kotormcp.adapters import InstallationModelLocator

    requests = []
    resources = {
        "mdl": SimpleNamespace(data=b"mdl-bytes"),
        "mdx": SimpleNamespace(data=b"mdx-vertex-stream"),
    }
    installation = SimpleNamespace(
        get_resource=lambda resref, restype: (
            requests.append((resref, restype)) or resources.get(restype)
        )
    )
    registry = SimpleNamespace(
        resolve=lambda game: "K1",
        load=lambda game, explicit_path: installation,
    )

    label, mdl, mdx = InstallationModelLocator(registry).locate("c_brith", "k1")

    assert label == "installation:c_brith.mdl"
    assert mdl == b"mdl-bytes"
    assert mdx == b"mdx-vertex-stream"
    assert requests == [("c_brith", "mdl"), ("c_brith", "mdx")]


def test_fbx_export_blocks_skin_mesh_without_vertex_weight_rows(tmp_path):
    model = KotorModel(name="broken_creature")
    root = ModelNode(name="broken_creature", flags=int(NodeFlags.HEADER))
    skin = ModelNode(
        name="body_skin",
        flags=int(NodeFlags.HEADER | NodeFlags.MESH | NodeFlags.SKIN),
        parent=root,
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
    )
    skin.bone_map = ["broken_creature"]
    skin.skin_data = []
    root.children = [skin]
    model.root_node = root
    output = tmp_path / "broken_creature.fbx"

    assert FBXExporter().export(
        model,
        str(output),
        compatibility_profile="unreal",
    ) is False
    assert output.exists() is False


def test_fbx_manifest_marks_missing_skin_rows_as_invalid():
    from src.core.export.kotor_fbx_manifest import _skin_summary, _validation_summary

    skin = ModelNode(
        name="body_skin",
        flags=int(NodeFlags.HEADER | NodeFlags.MESH | NodeFlags.SKIN),
        vertices=[(0.0, 0.0, 0.0)],
    )
    skin.bone_map = ["root"]
    skin.skin_data = []
    summary = _skin_summary([skin], {"root"})
    validation = _validation_summary({"skeleton": {"skin": summary}, "fbx": {}})

    assert summary["skin_meshes"][0]["vertex_count"] == 1
    assert summary["skin_meshes"][0]["vertex_rows"] == 0
    assert validation["ok"] is False
    assert {row["code"] for row in validation["errors"]} == {
        "skin_vertex_row_mismatch",
    }


def test_mcp_unreal_export_materializes_only_selected_animation_sets(monkeypatch):
    out_root = Path(".pytest_tmp_unity_bridge") / "UnrealSelectedAnimations"
    exported = {}

    class _Locator:
        def locate(self, resref, game, game_path):
            return f"installation:{resref}.mdl", b"mdl", b"mdx"

    class _Parser:
        def parse(self, mdl_bytes, mdx_bytes, path_label):
            model = _Model()
            model.anim_scale = 1.0
            return model

    texture_cache = SimpleNamespace(
        summary=lambda: {
            "requested": ["LEH_hull"],
            "decoded": ["LEH_hull"],
            "missing": [],
            "errors": {},
            "counts": {"requested": 1, "decoded": 1, "missing": 0, "errors": 0},
        }
    )

    def _fake_exporter(model, out_path, export_rigging, *, tex_cache=None):
        exported["animations"] = [animation.name for animation in model.animations]
        exported["texture_cache"] = tex_cache
        out_path.write_text("fbx", encoding="utf-8")
        texture_dir = out_path.parent / "textures"
        texture_dir.mkdir(parents=True, exist_ok=True)
        (texture_dir / "LEH_hull.png").write_bytes(b"png")
        out_path.with_suffix(".pbr.json").write_text(
            json.dumps({
                "materials": [{
                    "channels": {
                        "base_color": {"path": "textures/LEH_hull.png"},
                    },
                }],
            }),
            encoding="utf-8",
        )
        return True

    services = SimpleNamespace(
        locator=_Locator(),
        parser=_Parser(),
        analyzer=None,
        registry=None,
        texture_cache_factory=lambda game, game_path: texture_cache,
    )
    monkeypatch.setattr(ghostrigger, "_get_services", lambda: services)
    monkeypatch.setattr(ghostrigger, "_export_fbx_for_unreal", _fake_exporter)

    try:
        result = asyncio.run(handle_tool(
            "ghostrigger_export_model_for_unreal",
            {
                "game": "k1",
                "resref": "n_darthmalak",
                "export_dir": str(out_root),
                "animation_names": ["tlknorm"],
            },
        ))
        payload = json.loads(result["text"])

        assert payload["status"] == "ok"
        assert payload["compatibility_profile"] == "unreal"
        assert payload["animations"] == ["tlknorm"]
        assert payload["animation_selection"]["requested"] == ["tlknorm"]
        assert payload["animation_selection"]["embedded"] == ["tlknorm"]
        assert exported["animations"] == ["tlknorm"]
        assert exported["texture_cache"] is texture_cache
        assert payload["counts"]["texture_sidecars"] == 1
        assert payload["textures"]["sidecars"][0]["name"] == "LEH_hull"
        assert payload["textures"]["sidecars"][0]["sha256"] == sha256(b"png").hexdigest()
        assert payload["textures"]["resolution"]["counts"]["decoded"] == 1
        assert Path(payload["asset"]).exists()
    finally:
        shutil.rmtree(out_root, ignore_errors=True)
