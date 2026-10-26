"""Focused PBR material generation coverage for engine and glTF exports."""

from __future__ import annotations

import asyncio
import base64
import json
import struct
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest


class _TextureCache:
    def __init__(self, images):
        self._images = {str(name).casefold(): image for name, image in images.items()}

    def get(self, name):
        image = self._images.get(str(name).casefold())
        return image.copy() if image is not None else None

    def summary(self):
        return {
            "counts": {
                "requested": len(self._images),
                "decoded": len(self._images),
                "missing": 0,
                "errors": 0,
            }
        }


def _material_model(
    *,
    bump_map: str = "",
    alpha: float = 1.0,
    blending: int = 0,
    alpha_cutoff: float = 0.5,
):
    from src.core.geometry.model_data import KotorModel, ModelNode, NodeFlags

    root = ModelNode(name="root", flags=int(NodeFlags.HEADER))
    mesh = ModelNode(
        name="probe_mesh",
        flags=int(NodeFlags.MESH),
        parent=root,
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        normals=[(0.0, 0.0, 1.0)] * 3,
        tangents=[(1.0, 0.0, 0.0)] * 3,
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        faces=[(0, 1, 2)],
        texture="probe_diffuse",
        bump_map=bump_map,
        shininess=8.0,
        alpha=alpha,
        txi_blending=blending,
        txi_alpha_test=alpha_cutoff,
    )
    root.children.append(mesh)
    return KotorModel(name="pbr_probe", root_node=root), mesh


def _shared_texture_model():
    from src.core.geometry.model_data import KotorModel, ModelNode, NodeFlags

    root = ModelNode(name="root", flags=int(NodeFlags.HEADER))
    meshes = []
    for index, (shininess, clamp_s) in enumerate(((2.0, False), (128.0, True))):
        mesh = ModelNode(
            name=f"probe_mesh_{index}",
            flags=int(NodeFlags.MESH),
            parent=root,
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            normals=[(0.0, 0.0, 1.0)] * 3,
            uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
            faces=[(0, 1, 2)],
            texture="probe_diffuse",
            shininess=shininess,
            txi_clamp_s=clamp_s,
        )
        root.children.append(mesh)
        meshes.append(mesh)
    return KotorModel(name="shared_texture_probe", root_node=root), meshes


def _texture_cache():
    image_module = pytest.importorskip("PIL.Image")
    albedo = image_module.new("RGBA", (2, 2))
    albedo.putdata(
        [
            (255, 0, 0, 255),
            (255, 0, 0, 128),
            (0, 0, 255, 255),
            (0, 0, 255, 128),
        ]
    )
    normal = image_module.new("RGBA", (2, 2), (128, 64, 255, 255))
    return _TextureCache({"probe_diffuse": albedo, "probe_normal": normal})


def _read_glb_json(path: Path) -> dict:
    payload = path.read_bytes()
    magic, version, total_length = struct.unpack_from("<III", payload, 0)
    assert (magic, version, total_length) == (0x46546C67, 2, len(payload))
    json_length, json_type = struct.unpack_from("<II", payload, 12)
    assert json_type == 0x4E4F534A
    return json.loads(payload[20 : 20 + json_length].decode("utf-8"))


def _read_data_uri_pixel(uri: str, xy=(0, 0)):
    from PIL import Image

    encoded = uri.split(",", 1)[1]
    with Image.open(BytesIO(base64.b64decode(encoded))) as image:
        return image.convert("RGBA").getpixel(xy)


def test_pbr_package_writes_unreal_channel_maps_and_shader_contract(tmp_path: Path) -> None:
    from src.core.export.pbr_materials import export_pbr_material_package

    model, _mesh = _material_model(bump_map="probe_normal")
    result = export_pbr_material_package(
        model,
        tmp_path,
        _texture_cache(),
        target="unreal",
        asset_stem="pbr_probe",
    )

    assert result.manifest_path == tmp_path / "pbr_probe.pbr.json"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    material = manifest["materials"][0]
    assert manifest["schema"] == "ghostrigger.pbr_material_package.v1"
    assert manifest["shader"]["name"] == "Unreal Default Lit"
    assert material["channels"]["base_color"]["color_space"] == "sRGB"
    assert material["channels"]["normal"]["normal_y"] == "DirectX Y-"
    assert material["channels"]["orm"]["packing"] == {
        "r": "ambient_occlusion",
        "g": "roughness",
        "b": "metallic",
    }

    from PIL import Image

    base_color = Image.open(tmp_path / material["channels"]["base_color"]["path"])
    orm = Image.open(tmp_path / material["channels"]["orm"]["path"])
    # KOTOR's decoded row order is converted once at the export boundary.
    assert base_color.getpixel((0, 0))[:3] == (0, 0, 255)
    assert orm.getpixel((0, 0))[0] == 255
    assert orm.getpixel((0, 0))[2] == 0


def test_pbr_package_writes_unity_lit_maps_with_linear_data_channels(tmp_path: Path) -> None:
    from src.core.export.pbr_materials import export_pbr_material_package

    model, _mesh = _material_model(bump_map="probe_normal")
    result = export_pbr_material_package(
        model,
        tmp_path,
        _texture_cache(),
        target="unity",
        asset_stem="pbr_probe",
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    material = manifest["materials"][0]
    channels = material["channels"]
    assert manifest["shader"]["name"] == "Unity Lit"
    assert manifest["shader"]["compatible_shaders"] == [
        "Universal Render Pipeline/Lit",
        "HDRP/Lit",
        "Standard",
    ]
    assert channels["normal"]["normal_y"] == "OpenGL Y+"
    assert channels["metallic_smoothness"]["packing"] == {
        "r": "metallic",
        "g": "ambient_occlusion",
        "b": "detail_mask",
        "a": "smoothness",
    }
    assert channels["ambient_occlusion"]["color_space"] == "linear"

    from PIL import Image

    normal = Image.open(tmp_path / channels["normal"]["path"])
    metallic_smoothness = Image.open(
        tmp_path / channels["metallic_smoothness"]["path"]
    )
    assert normal.getpixel((0, 0))[1] == 191
    assert metallic_smoothness.getpixel((0, 0))[0] == 0
    assert metallic_smoothness.getpixel((0, 0))[1] == 255


def test_pbr_package_keeps_distinct_material_factors_for_shared_textures(
    tmp_path: Path,
) -> None:
    from src.core.export.pbr_materials import export_pbr_material_package

    model, _meshes = _shared_texture_model()
    result = export_pbr_material_package(
        model,
        tmp_path,
        _texture_cache(),
        target="unreal",
        asset_stem="shared_texture_probe",
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    first, second = manifest["materials"]
    assert first["factors"]["roughness"] != second["factors"]["roughness"]
    assert first["channels"]["orm"]["path"] != second["channels"]["orm"]["path"]

    from PIL import Image

    first_orm = Image.open(tmp_path / first["channels"]["orm"]["path"])
    second_orm = Image.open(tmp_path / second["channels"]["orm"]["path"])
    assert first_orm.getpixel((0, 0))[1] != second_orm.getpixel((0, 0))[1]


def test_sequential_asset_exports_do_not_mutate_prior_pbr_maps(tmp_path: Path) -> None:
    from src.core.export.pbr_materials import export_pbr_material_package

    first_model, first_mesh = _material_model()
    first_mesh.shininess = 2.0
    first = export_pbr_material_package(
        first_model,
        tmp_path,
        _texture_cache(),
        target="unreal",
        asset_stem="first_asset",
    )
    first_manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    first_paths = [
        tmp_path / channel["path"]
        for channel in first_manifest["materials"][0]["channels"].values()
    ]
    first_hashes = {path: sha256(path.read_bytes()).hexdigest() for path in first_paths}

    second_model, second_mesh = _material_model()
    second_mesh.shininess = 128.0
    second = export_pbr_material_package(
        second_model,
        tmp_path,
        _texture_cache(),
        target="unreal",
        asset_stem="second_asset",
    )
    second_manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    second_paths = {
        tmp_path / channel["path"]
        for channel in second_manifest["materials"][0]["channels"].values()
    }

    assert not second_paths.intersection(first_paths)
    assert {
        path: sha256(path.read_bytes()).hexdigest()
        for path in first_paths
    } == first_hashes


@pytest.mark.parametrize(
    ("target", "shader_name", "blend_mode"),
    [
        ("unity", "Unity Additive Lit", "Additive"),
        ("unreal", "Unreal Additive Material", "Additive"),
    ],
)
def test_additive_txi_materials_use_additive_engine_shader_contracts(
    target: str,
    shader_name: str,
    blend_mode: str,
    tmp_path: Path,
) -> None:
    from src.core.export.pbr_materials import export_pbr_material_package

    model, _mesh = _material_model(alpha=0.6, blending=1)
    result = export_pbr_material_package(
        model,
        tmp_path,
        _texture_cache(),
        target=target,
        asset_stem=f"additive_{target}",
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    material = manifest["materials"][0]
    assert material["alpha"] == {
        "blend_operation": "ADDITIVE",
        "cutoff": None,
        "factor": 0.6,
        "mode": "BLEND",
        "source_txi_blending": 1,
    }
    assert material["shader"] == shader_name
    assert material["shader_contract"]["blend_mode"] == blend_mode


def test_manual_glb_embeds_complete_metallic_roughness_material(tmp_path: Path) -> None:
    from src.converters.mesh_converter import GLTFExporter

    model, _mesh = _material_model(bump_map="probe_normal")
    output = tmp_path / "pbr_probe.glb"

    assert GLTFExporter()._export_manual(
        model,
        str(output),
        True,
        tex_cache=_texture_cache(),
    )

    gltf = _read_glb_json(output)
    material = gltf["materials"][0]
    pbr = material["pbrMetallicRoughness"]
    assert "baseColorTexture" in pbr
    assert "metallicRoughnessTexture" in pbr
    assert "normalTexture" in material
    assert "occlusionTexture" in material
    assert material["extras"]["ghostriggerPbr"]["shader"] == (
        "glTF 2.0 PBR Metallic-Roughness"
    )
    assert {image["name"] for image in gltf["images"]} == {
        "probe_diffuse_BaseColor",
        "probe_diffuse_Normal",
        "probe_diffuse_ORM",
    }


def test_pygltflib_glb_embeds_the_same_complete_pbr_contract(tmp_path: Path) -> None:
    pygltflib = pytest.importorskip("pygltflib")
    from src.converters.mesh_converter import GLTFExporter

    model, _mesh = _material_model(bump_map="probe_normal")
    output = tmp_path / "pbr_probe_structured.glb"

    assert GLTFExporter()._export_pygltflib(
        model,
        str(output),
        True,
        tex_cache=_texture_cache(),
    )

    gltf = pygltflib.GLTF2().load(str(output))
    material = gltf.materials[0]
    assert material.pbrMetallicRoughness.baseColorTexture is not None
    assert material.pbrMetallicRoughness.metallicRoughnessTexture is not None
    assert material.normalTexture is not None
    assert material.occlusionTexture is not None
    assert material.extras["ghostriggerPbr"]["shader"] == (
        "glTF 2.0 PBR Metallic-Roughness"
    )


def test_pygltflib_preserves_complete_authored_bas_skin_contract(
    tmp_path: Path,
) -> None:
    """The glTF stage retains the first-skin bind frame used by DCC importers."""
    pygltflib = pytest.importorskip("pygltflib")
    from src.converters.mesh_converter import GLTFExporter
    from src.core.geometry import model_data as md

    root = md.ModelNode(name="root", flags=int(md.NodeFlags.HEADER))
    joint = md.ModelNode(
        name="f_jaw_g",
        flags=int(md.NodeFlags.HEADER),
        parent=root,
        position=(0.0, 0.0, 1.0),
    )
    root.children.append(joint)
    for name, height in (("torso", 2.0), ("tongue", 3.0)):
        skin = md.ModelNode(
            name=name,
            flags=int(md.NodeFlags.HEADER | md.NodeFlags.MESH | md.NodeFlags.SKIN),
            parent=root,
            position=(0.0, 0.0, height),
            vertices=[(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0)],
            normals=[(0.0, 0.0, 1.0)] * 3,
            uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
            faces=[(0, 1, 2)],
            bone_map=["f_jaw_g"],
            skin_data=[md.VertexSkinData([md.BoneWeight(0, 1.0)]) for _ in range(3)],
        )
        skin.bone_node_indices = [1]
        root.children.append(skin)

    output = tmp_path / "authored_bas_skin_contract.glb"
    assert GLTFExporter()._export_pygltflib(
        md.KotorModel(name="authored_bas_skin_contract", root_node=root),
        str(output),
        True,
    )

    gltf = pygltflib.GLTF2().load_binary(str(output))
    mesh_nodes = {
        node.name: node for node in gltf.nodes if node.name in {"torso", "tongue"}
    }
    assert set(mesh_nodes) == {"torso", "tongue"}
    assert mesh_nodes["torso"].translation == pytest.approx([0.0, 0.0, 2.0])
    assert mesh_nodes["tongue"].translation == pytest.approx([0.0, 0.0, 3.0])
    assert mesh_nodes["tongue"].skin is not None

    position_mins = {}
    for name, node in mesh_nodes.items():
        primitive = gltf.meshes[node.mesh].primitives[0]
        position_mins[name] = gltf.accessors[primitive.attributes.POSITION].min

    assert position_mins["torso"] == pytest.approx([0.0, 0.0, 0.0])
    assert position_mins["tongue"] == pytest.approx([0.0, 0.0, 0.0])

    assert mesh_nodes["tongue"].matrix is None


def test_pygltflib_preserves_authored_bas_head_bind_origin(tmp_path: Path) -> None:
    """BAS heads stay mesh-local so DCC armature consolidation cannot stretch them."""
    pygltflib = pytest.importorskip("pygltflib")
    from src.converters.mesh_converter import GLTFExporter
    from src.core.geometry import model_data as md

    root = md.ModelNode(name="root", flags=int(md.NodeFlags.HEADER))
    joint = md.ModelNode(
        name="head_joint",
        flags=int(md.NodeFlags.HEADER),
        parent=root,
        position=(0.0, 0.0, 1.0),
    )
    head = md.ModelNode(
        name="head",
        flags=int(md.NodeFlags.HEADER | md.NodeFlags.MESH | md.NodeFlags.SKIN),
        parent=root,
        position=(0.0, 0.0, 2.0),
        vertices=[(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0)],
        normals=[(0.0, 0.0, 1.0)] * 3,
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        faces=[(0, 1, 2)],
        bone_map=["head_joint"],
        skin_data=[md.VertexSkinData([md.BoneWeight(0, 1.0)]) for _ in range(3)],
    )
    head.bone_node_indices = [1]
    root.children.extend([joint, head])

    output = tmp_path / "authored_head_bind.glb"
    assert GLTFExporter()._export_pygltflib(
        md.KotorModel(name="authored_head_bind", root_node=root),
        str(output),
        True,
    )

    gltf = pygltflib.GLTF2().load_binary(str(output))
    head_node = next(node for node in gltf.nodes if node.name == "head")
    primitive = gltf.meshes[head_node.mesh].primitives[0]
    assert head_node.translation == pytest.approx([0.0, 0.0, 2.0])
    assert gltf.accessors[primitive.attributes.POSITION].min == pytest.approx(
        [0.0, 0.0, 0.0]
    )


@pytest.mark.parametrize("backend", ["_export_manual", "_export_pygltflib"])
def test_glb_preserves_authored_uvs_and_character_atlas_pixels(
    backend: str,
    tmp_path: Path,
) -> None:
    if backend == "_export_pygltflib":
        pytest.importorskip("pygltflib")
    from src.converters.mesh_converter import GLTFExporter

    model, _mesh = _material_model()
    output = tmp_path / f"{backend}_atlas_orientation.glb"

    assert getattr(GLTFExporter(), backend)(
        model,
        str(output),
        True,
        tex_cache=_texture_cache(),
    )

    document = _read_glb_json(output)
    base_color_index = document["materials"][0]["pbrMetallicRoughness"][
        "baseColorTexture"
    ]["index"]
    image_index = document["textures"][base_color_index]["source"]
    uri = document["images"][image_index]["uri"]
    # The fixture's decoded top row is red and bottom row is blue. Both the
    # authored UV rows and atlas pixels stay untouched; glTF importers own the
    # texture-origin conversion at interchange time.
    assert _read_data_uri_pixel(uri, (0, 0)) == (255, 0, 0, 255)
    assert _read_data_uri_pixel(uri, (0, 1)) == (0, 0, 255, 255)

    payload = output.read_bytes()
    json_length = struct.unpack_from("<I", payload, 12)[0]
    bin_header = 20 + json_length
    bin_length, bin_type = struct.unpack_from("<II", payload, bin_header)
    assert bin_type == 0x004E4942
    binary = payload[bin_header + 8 : bin_header + 8 + bin_length]
    primitive = document["meshes"][0]["primitives"][0]
    accessor = document["accessors"][primitive["attributes"]["TEXCOORD_0"]]
    view = document["bufferViews"][accessor["bufferView"]]
    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    stride = int(view.get("byteStride", 8))
    exported_uvs = [
        struct.unpack_from("<ff", binary, start + index * stride)
        for index in range(int(accessor["count"]))
    ]
    assert exported_uvs == pytest.approx([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])


@pytest.mark.parametrize("backend", ["_export_manual", "_export_pygltflib"])
@pytest.mark.parametrize(
    ("alpha", "blending", "expected_mode", "expected_blend", "expected_cutoff"),
    [
        (1.0, 0, "OPAQUE", "OPAQUE", None),
        (0.4, 1, "BLEND", "ADDITIVE", None),
        (0.7, 2, "MASK", "MASK", 0.375),
    ],
)
def test_glb_backends_preserve_kotor_alpha_contract(
    backend: str,
    alpha: float,
    blending: int,
    expected_mode: str,
    expected_blend: str,
    expected_cutoff: float | None,
    tmp_path: Path,
) -> None:
    if backend == "_export_pygltflib":
        pytest.importorskip("pygltflib")
    from src.converters.mesh_converter import GLTFExporter

    model, _mesh = _material_model(
        alpha=alpha,
        blending=blending,
        alpha_cutoff=0.375,
    )
    output = tmp_path / f"{backend}_{expected_mode}.glb"

    assert getattr(GLTFExporter(), backend)(
        model,
        str(output),
        True,
        tex_cache=_texture_cache(),
    )

    material = _read_glb_json(output)["materials"][0]
    assert material["alphaMode"] == expected_mode
    assert material["extras"]["ghostriggerPbr"]["alpha"]["blend_operation"] == (
        expected_blend
    )
    assert material["pbrMetallicRoughness"]["baseColorFactor"][3] == alpha
    if expected_cutoff is None:
        assert "alphaCutoff" not in material
    else:
        assert material["alphaCutoff"] == expected_cutoff


def test_glb_backends_preserve_sampler_state_for_shared_images(tmp_path: Path) -> None:
    pygltflib = pytest.importorskip("pygltflib")
    from src.converters.mesh_converter import GLTFExporter

    model, meshes = _shared_texture_model()
    meshes[1].shininess = meshes[0].shininess
    manual_output = tmp_path / "shared_manual.glb"
    structured_output = tmp_path / "shared_structured.glb"

    exporter = GLTFExporter()
    assert exporter._export_manual(
        model,
        str(manual_output),
        True,
        tex_cache=_texture_cache(),
    )
    assert exporter._export_pygltflib(
        model,
        str(structured_output),
        True,
        tex_cache=_texture_cache(),
    )

    manual_pairs = {
        (sampler["wrapS"], sampler["wrapT"])
        for sampler in _read_glb_json(manual_output)["samplers"]
    }
    structured = pygltflib.GLTF2().load(str(structured_output))
    structured_pairs = {(sampler.wrapS, sampler.wrapT) for sampler in structured.samplers}
    assert manual_pairs == structured_pairs == {(10497, 10497), (33071, 10497)}


def test_manual_glb_keeps_distinct_orm_for_shared_texture_materials(
    tmp_path: Path,
) -> None:
    from src.converters.mesh_converter import GLTFExporter

    model, meshes = _shared_texture_model()
    meshes[1].txi_clamp_s = meshes[0].txi_clamp_s
    output = tmp_path / "shared_distinct_orm.glb"

    assert GLTFExporter()._export_manual(
        model,
        str(output),
        True,
        tex_cache=_texture_cache(),
    )

    document = _read_glb_json(output)
    texture_indices = [
        material["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"]
        for material in document["materials"]
    ]
    assert texture_indices[0] != texture_indices[1]
    orm_green = []
    for texture_index in texture_indices:
        image_index = document["textures"][texture_index]["source"]
        orm_green.append(_read_data_uri_pixel(document["images"][image_index]["uri"])[1])
    assert orm_green[0] != orm_green[1]


def test_pygltflib_glb_reuses_identical_pbr_images_and_samplers(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pygltflib")
    from src.converters.mesh_converter import GLTFExporter

    model, meshes = _shared_texture_model()
    meshes[1].shininess = meshes[0].shininess
    meshes[1].txi_clamp_s = meshes[0].txi_clamp_s
    output = tmp_path / "shared_identical_structured.glb"

    assert GLTFExporter()._export_pygltflib(
        model,
        str(output),
        True,
        tex_cache=_texture_cache(),
    )

    document = _read_glb_json(output)
    assert len(document["images"]) == 3
    assert len(document["samplers"]) == 3
    assert len(document["textures"]) == 3


def test_unreal_fbx_writes_and_links_pbr_material_sidecars(tmp_path: Path) -> None:
    from src.converters.mesh_converter import FBXExporter

    model, _mesh = _material_model(bump_map="probe_normal")
    output = tmp_path / "pbr_probe.fbx"

    assert FBXExporter().export(
        model,
        str(output),
        tex_cache=_texture_cache(),
        export_rigging=False,
        compatibility_profile="unreal",
    )

    text = output.read_text(encoding="utf-8")
    assert 'P: "GhostRiggerPBRShader","KString","","","Unreal Default Lit"' in text
    assert '"NormalMap"' in text
    assert '"OcclusionRoughnessMetallic"' in text
    texture_dir = tmp_path / "textures" / "pbr_probe" / "unreal"
    assert (texture_dir / "probe_diffuse_BaseColor.png").is_file()
    assert (texture_dir / "probe_diffuse_Normal.png").is_file()
    assert (texture_dir / "probe_diffuse_ORM.png").is_file()

    pbr_manifest = json.loads((tmp_path / "pbr_probe.pbr.json").read_text(encoding="utf-8"))
    fbx_manifest = json.loads((tmp_path / "pbr_probe.ghostrigger.json").read_text(encoding="utf-8"))
    assert pbr_manifest["target"] == "unreal"
    assert fbx_manifest["pbr"]["manifest"] == "pbr_probe.pbr.json"
    assert fbx_manifest["pbr"]["shader"] == "Unreal Default Lit"


def test_unreal_fbx_marks_additive_material_shader_and_blend_operation(
    tmp_path: Path,
) -> None:
    from src.converters.mesh_converter import FBXExporter

    model, _mesh = _material_model(alpha=0.6, blending=1)
    output = tmp_path / "additive_probe.fbx"

    assert FBXExporter().export(
        model,
        str(output),
        tex_cache=_texture_cache(),
        export_rigging=False,
        compatibility_profile="unreal",
    )

    text = output.read_text(encoding="utf-8")
    assert 'P: "GhostRiggerPBRShader","KString","","","Unreal Additive Material"' in text
    assert 'P: "GhostRiggerBlendOperation","KString","","","ADDITIVE"' in text


def test_unreal_fbx_keeps_shared_texture_material_bindings_distinct(
    tmp_path: Path,
) -> None:
    from src.converters.mesh_converter import FBXExporter

    model, _meshes = _shared_texture_model()
    output = tmp_path / "shared_texture_probe.fbx"

    assert FBXExporter().export(
        model,
        str(output),
        tex_cache=_texture_cache(),
        export_rigging=False,
        compatibility_profile="unreal",
    )

    manifest = json.loads(output.with_suffix(".pbr.json").read_text(encoding="utf-8"))
    first, second = manifest["materials"]
    text = output.read_text(encoding="utf-8")
    for material in (first, second):
        assert material["channels"]["orm"]["path"] in text
        roughness = float(material["factors"]["roughness"])
        assert f'P: "GhostRiggerRoughness","double","Number","",{roughness:.6f}' in text
    assert text.count(',"OcclusionRoughnessMetallic"') == 2
    assert text.count(',"DiffuseColor"') == 2
    assert 'P: "WrapModeU","enum","","",0' in text
    assert 'P: "WrapModeU","enum","","",1' in text


def test_unity_automation_enables_texture_backed_pbr_export_by_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from kotormcp.tools import get_all_tools, ghostrigger, handle_tool

    model, _mesh = _material_model()
    texture_cache = _texture_cache()
    captured = {}

    class _Locator:
        def locate(self, resref, game, game_path):
            return f"installation:{resref}.mdl", b"mdl", b"mdx"

    class _Parser:
        def parse(self, mdl_bytes, mdx_bytes, path_label):
            return model

    real_exporter = ghostrigger._export_fbx_for_unity

    def _capturing_exporter(candidate, out_path, export_rigging, *, tex_cache=None):
        captured["cache"] = tex_cache
        return real_exporter(
            candidate,
            out_path,
            export_rigging,
            tex_cache=tex_cache,
        )

    services = SimpleNamespace(
        locator=_Locator(),
        parser=_Parser(),
        analyzer=None,
        registry=None,
        texture_cache_factory=lambda game, game_path: texture_cache,
    )
    monkeypatch.setattr(ghostrigger, "_get_services", lambda: services)
    monkeypatch.setattr(ghostrigger, "_export_fbx_for_unity", _capturing_exporter)

    stale_dir = tmp_path / "Assets" / "GhostRigger" / "textures"
    stale_dir.mkdir(parents=True)
    (stale_dir / "stale.png").write_bytes(b"not part of this export")

    result = asyncio.run(
        handle_tool(
            "ghostrigger_export_model_for_unity",
            {
                "game": "k1",
                "resref": "pbr_probe",
                "unity_project": str(tmp_path),
                "asset_subdir": "Assets/GhostRigger",
            },
        )
    )
    payload = json.loads(result["text"])
    properties = {
        tool["name"]: tool for tool in get_all_tools()
    }["ghostrigger_export_model_for_unity"]["inputSchema"]["properties"]

    assert payload["status"] == "ok"
    assert captured["cache"] is texture_cache
    assert properties["export_textures"]["type"] == "boolean"
    assert properties["export_textures"]["default"] is True
    asset_path = Path(payload["asset"])
    assert 'P: "GhostRiggerPBRShader","KString","","","Unity Lit"' in asset_path.read_text(
        encoding="utf-8"
    )
    pbr_manifest = json.loads(asset_path.with_suffix(".pbr.json").read_text(encoding="utf-8"))
    transfer_manifest = json.loads(Path(payload["metadata"]).read_text(encoding="utf-8"))
    assert pbr_manifest["shader"]["name"] == "Unity Lit"
    assert transfer_manifest["pbr"]["manifest"] == "pbr_probe.pbr.json"
    sidecar_names = {row["name"] for row in payload["textures"]["sidecars"]}
    assert "stale" not in sidecar_names
    assert sidecar_names == {
        "probe_diffuse_AO",
        "probe_diffuse_BaseColor",
        "probe_diffuse_MetallicSmoothness",
        "probe_diffuse_Normal",
        "probe_diffuse_ORM",
    }


def test_unreal_automation_reports_only_current_pbr_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from kotormcp.tools import ghostrigger, handle_tool

    model, _mesh = _material_model(bump_map="probe_normal")
    texture_cache = _texture_cache()

    class _Locator:
        def locate(self, resref, game, game_path):
            return f"installation:{resref}.mdl", b"mdl", b"mdx"

    class _Parser:
        def parse(self, mdl_bytes, mdx_bytes, path_label):
            return model

    services = SimpleNamespace(
        locator=_Locator(),
        parser=_Parser(),
        analyzer=None,
        registry=None,
        texture_cache_factory=lambda game, game_path: texture_cache,
    )
    monkeypatch.setattr(ghostrigger, "_get_services", lambda: services)

    stale_texture = tmp_path / "textures" / "stale.png"
    stale_texture.parent.mkdir(parents=True)
    stale_texture.write_bytes(b"not part of this export")

    fbx_result = asyncio.run(
        handle_tool(
            "ghostrigger_export_model_for_unreal",
            {
                "game": "k1",
                "resref": "pbr_probe",
                "export_dir": str(tmp_path),
                "output_name": "unreal_pbr",
            },
        )
    )
    fbx_payload = json.loads(fbx_result["text"])
    assert fbx_payload["status"] == "ok"
    assert fbx_payload["format"] == "fbx"
    assert {row["name"] for row in fbx_payload["textures"]["sidecars"]} == {
        "probe_diffuse_BaseColor",
        "probe_diffuse_Normal",
        "probe_diffuse_ORM",
    }
    assert fbx_payload["textures"]["embedded"] == {}

    stale_manifest = {
        "materials": [{"channels": {"base_color": {"path": "textures/stale.png"}}}]
    }
    (tmp_path / "unreal_pbr_glb.pbr.json").write_text(
        json.dumps(stale_manifest),
        encoding="utf-8",
    )
    glb_result = asyncio.run(
        handle_tool(
            "ghostrigger_export_model_for_unreal",
            {
                "game": "k1",
                "resref": "pbr_probe",
                "export_dir": str(tmp_path),
                "output_name": "unreal_pbr_glb",
                "format": "glb",
            },
        )
    )
    glb_payload = json.loads(glb_result["text"])
    embedded = glb_payload["textures"]["embedded"]
    assert glb_payload["status"] == "ok"
    assert glb_payload["textures"]["sidecars"] == []
    assert embedded["material_count"] == 1
    assert embedded["image_count"] == 3
    assert len(embedded["sha256"]) == 64
    assert embedded["materials"][0] == {
        "name": "probe_diffuse",
        "shader": "glTF 2.0 PBR Metallic-Roughness",
        "alpha_mode": "OPAQUE",
        "blend_operation": "OPAQUE",
        "base_color_texture": True,
        "metallic_roughness_texture": True,
        "normal_texture": True,
        "occlusion_texture": True,
    }
    assert glb_payload["counts"]["embedded_pbr_materials"] == 1

    (tmp_path / "no_textures.pbr.json").write_text(
        json.dumps(stale_manifest),
        encoding="utf-8",
    )
    no_texture_result = asyncio.run(
        handle_tool(
            "ghostrigger_export_model_for_unreal",
            {
                "game": "k1",
                "resref": "pbr_probe",
                "export_dir": str(tmp_path),
                "output_name": "no_textures",
                "export_textures": False,
            },
        )
    )
    no_texture_payload = json.loads(no_texture_result["text"])
    assert no_texture_payload["status"] == "ok"
    assert no_texture_payload["textures"] == {
        "requested": False,
        "sidecars": [],
        "embedded": {},
        "resolution": {},
    }
