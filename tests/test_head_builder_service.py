"""Focused application-service tests for project and donor orchestration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from pykotor.resource.formats.twoda import bytes_2da
from pykotor.resource.formats.twoda.twoda_data import TwoDA

from src.core.characters.head_builder_project import (
    EvidenceOutcome,
    HeadBuilderStep,
    ResourceOrigin,
    StepStatus,
)
from src.core.characters.head_builder_service import (
    HeadBuilderArtChangedError,
    HeadBuilderArtRejectedError,
    HeadBuilderDonorChangedError,
    HeadBuilderDonorRejectedError,
    HeadBuilderService,
    HeadBuilderServiceError,
    valid_head_output_resref,
)
from src.core.characters.head_facial_transplant import (
    ROLE_NODE_NAMES,
    _apply_rigid_payload,
)
from src.core.characters.head_art_anatomy import HeadArtAnatomyRole
import src.core.characters.head_facial_transplant as head_facial_transplant
from src.core.geometry.model_data import (
    Animation,
    BoneWeight,
    GameVersion,
    KotorModel,
    ModelNode,
    NodeFlags,
    VertexSkinData,
)
from src.core.project.head_builder_repository import (
    FileHeadBuilderProjectRepository,
)
from src.core.project.resource_address import ResourceAddress
from src.core.resources.game_resource_provider import (
    GameResourceRecord,
    InMemoryGameResourceProvider,
)
from src.core.resources.head_donor_catalog import HeadDonorCatalog
from src.core.resources.head_game_install import HeadGameInstallVerification
from src.math.head_alignment import HeadAlignmentAnchor, transform_point


def _attach(parent: ModelNode, child: ModelNode) -> None:
    parent.children.append(child)
    child.parent = parent


def test_facial_transplant_accepts_stock_k1_teeth_node_names() -> None:
    upper = {
        value.casefold()
        for value in ROLE_NODE_NAMES[HeadArtAnatomyRole.UPPER_TEETH]
    }
    lower = {
        value.casefold()
        for value in ROLE_NODE_NAMES[HeadArtAnatomyRole.LOWER_TEETH]
    }

    assert "teethua" in upper
    assert "teethla" in lower


def test_rigid_facial_payload_preserves_secondary_skin_bind_contract() -> None:
    tongue = ModelNode(
        name="tongue",
        flags=int(NodeFlags.HEADER | NodeFlags.MESH | NodeFlags.SKIN),
        bone_map=["f_tonguetip_g", "f_jaw_g"],
        qbone_list=[(0.0, 0.0, 0.0, 1.0)] * 3,
        tbone_list=[(0.0, 0.0, 0.0)] * 3,
        skin_data=[VertexSkinData([BoneWeight(0, 1.0)])],
    )
    tongue.bone_node_indices = [16, 13]
    tongue.bone_map_floats = [-1.0, 0.0, 1.0]
    saved_contract = (
        list(tongue.bone_map),
        list(tongue.bone_node_indices),
        list(tongue.bone_map_floats),
        list(tongue.qbone_list),
        list(tongue.tbone_list),
    )

    _apply_rigid_payload(
        tongue,
        vertices=[(-0.01, 0.0, 0.0), (0.01, 0.0, 0.0), (0.0, 0.02, 0.0)],
        faces=[(0, 1, 2)],
        normals=[(0.0, 0.0, 1.0)] * 3,
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)],
        texture_resref="P_CUSTOMH",
        skin_bone_name="f_jaw_g",
    )

    assert saved_contract == (
        tongue.bone_map,
        tongue.bone_node_indices,
        tongue.bone_map_floats,
        tongue.qbone_list,
        tongue.tbone_list,
    )
    assert len(tongue.skin_data) == 3
    assert all(
        row.influences == [BoneWeight(1, 1.0)]
        for row in tongue.skin_data
    )


def test_facial_skin_transfer_uses_world_space_and_writes_node_local_space() -> None:
    donor = ModelNode(
        name="head",
        position=(0.0, 0.0, 1.7),
        vertices=[
            (-0.1, 0.0, -1.7),
            (0.1, 0.0, -1.7),
            (0.0, 0.0, -1.5),
        ],
    )
    target_world = [
        (-0.08, 0.01, 0.01),
        (0.08, 0.01, 0.01),
        (0.0, 0.01, 0.18),
    ]
    target_normals = [(0.0, -1.0, 0.0)] * 3

    donor_world, target_local, local_normals = (
        head_facial_transplant._skin_transfer_and_payload_channels(
            donor_skin=donor,
            payload_skin=deepcopy(donor),
            target_world_vertices=target_world,
            target_world_normals=target_normals,
        )
    )

    for actual, expected in zip(
        donor_world,
        [(-0.1, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.0, 0.2)],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        target_local,
        [(-0.08, 0.01, -1.69), (0.08, 0.01, -1.69), (0.0, 0.01, -1.52)],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(local_normals, target_normals, strict=True):
        assert actual == pytest.approx(expected)


def _head_model() -> KotorModel:
    root = ModelNode(name="PFHA04", index=0, number=0)
    neck = ModelNode(name="neck_g", index=1, number=30)
    head_bone = ModelNode(name="head_g", index=2, number=32)
    jaw = ModelNode(name="f_jaw_g", index=3, number=37)
    skin = ModelNode(
        name="head",
        flags=int(NodeFlags.HEADER | NodeFlags.MESH | NodeFlags.SKIN),
        index=4,
        number=355,
        vertices=[(-0.1, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.0, 0.2)],
        normals=[(0.0, -1.0, 0.0)] * 3,
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)],
        faces=[(0, 1, 2)],
        bone_map=["head_g", "neck_g", "f_jaw_g"],
        skin_data=[
            VertexSkinData([BoneWeight(0, 1.0)]),
            VertexSkinData([BoneWeight(0, 0.5), BoneWeight(1, 0.5)]),
            VertexSkinData([BoneWeight(2, 1.0)]),
        ],
        bb_min=(-0.1, 0.0, 0.0),
        bb_max=(0.1, 0.0, 0.2),
        radius=0.15,
    )
    skin.bone_node_indices = [2, 1, 3]
    for child in (neck, head_bone, jaw, skin):
        _attach(root, child)
    skin.qbone_list = [(0.0, 0.0, 0.0, 1.0)] * 5
    skin.tbone_list = [(0.0, 0.0, 0.0)] * 5
    model = KotorModel(
        name="PFHA04",
        supermodel="S_Female03",
        game_version=GameVersion.K2,
        model_type=4,
        root_node=root,
        bb_min=(-5.0, -5.0, -1.0),
        bb_max=(5.0, 5.0, 10.0),
        radius=7.0,
        super_root_node_name="neck_g",
        geometry_node_count=564,
        preserve_native_supernode_numbers=True,
    )
    model._gr_render_bounds = ((-0.1, 0.0, 0.0), (0.1, 0.0, 0.2))
    model._gr_render_radius = 0.15
    return model


def _component_head_model(
    resref: str,
    *,
    hair_count: int = 2,
) -> KotorModel:
    model = _head_model()
    model.name = resref
    assert model.root_node is not None
    model.root_node.name = resref
    face = model.find_node("head")
    head_bone = model.find_node("head_g")
    assert face is not None and head_bone is not None
    face.texture = f"{resref}_FACE"

    def mesh(
        name: str,
        texture: str,
        index: int,
        *,
        dangly: bool = False,
    ) -> ModelNode:
        flags = int(NodeFlags.HEADER | NodeFlags.MESH)
        if dangly:
            flags |= int(NodeFlags.DANGLY)
        return ModelNode(
            name=name,
            flags=flags,
            index=index,
            number=400 + index,
            vertices=[
                (index * 0.01, 0.0, 0.0),
                (index * 0.01 + 0.1, 0.0, 0.0),
                (index * 0.01, 0.1, 0.0),
            ],
            normals=[(0.0, 0.0, 1.0)] * 3,
            uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
            faces=[(0, 1, 2)],
            texture=texture,
            dangly_constraints=[1.0, 0.5, 0.0] if dangly else [],
        )

    rows = [
        mesh("eyeLA", f"{resref}_EYE", 5),
        mesh("eyeRA", f"{resref}_EYE", 6),
        mesh("eyeLlid", f"{resref}_LID", 7),
        mesh("eyeRlid", f"{resref}_LID", 8),
        mesh("teethU", f"{resref}_MOUTH", 9),
        mesh("teethL", f"{resref}_MOUTH", 10),
    ]
    rows.extend(
        mesh(
            f"hair0{index + 1}",
            f"{resref}_HAIR",
            11 + index,
            dangly=True,
        )
        for index in range(hair_count)
    )
    for row in rows:
        _attach(head_bone, row)
    node_count = len(model.all_nodes())
    face.qbone_list = [(0.0, 0.0, 0.0, 1.0)] * node_count
    face.tbone_list = [(0.0, 0.0, 0.0)] * node_count
    return model


def _invalid_model() -> KotorModel:
    return KotorModel(
        name="PLC_BENCH",
        supermodel="NULL",
        game_version=GameVersion.K2,
        model_type=32,
        root_node=ModelNode(name="PLC_BENCH"),
        geometry_node_count=1,
    )


def _body_model() -> KotorModel:
    root = ModelNode(name="PFBAM")
    torso = ModelNode(name="torso_g")
    hook = ModelNode(name="headhook", position=(0.0, 0.0, 1.5))
    _attach(root, torso)
    _attach(torso, hook)
    return KotorModel(
        name="PFBAM",
        supermodel="S_Female03",
        game_version=GameVersion.K2,
        root_node=root,
    )


def _female03_model() -> KotorModel:
    return KotorModel(
        name="S_Female03",
        supermodel="S_Female02",
        game_version=GameVersion.K2,
        root_node=ModelNode(name="S_Female03"),
        animations=[
            Animation(
                name="tlknorm",
                length=1.0,
                nodes=[ModelNode(name="f_jaw_g")],
            ),
            Animation(
                name="listen",
                length=1.2,
                nodes=[ModelNode(name="f_jaw_g")],
            ),
            Animation(
                name="walk",
                length=2.0,
                nodes=[ModelNode(name="torso_g")],
            ),
        ],
    )


def _female02_model() -> KotorModel:
    return KotorModel(
        name="S_Female02",
        supermodel="NULL",
        game_version=GameVersion.K2,
        root_node=ModelNode(name="S_Female02"),
        animations=[
            Animation(
                name="talk",
                length=0.8,
                nodes=[ModelNode(name="f_jaw_g")],
            )
        ],
    )


def _record(resref: str, restype: str) -> GameResourceRecord:
    return GameResourceRecord(
        address=ResourceAddress(
            scheme="game_resource",
            game="k2",
            resref=resref,
            restype=restype,
            layer="base",
            path=r"H:\K2\data\models.bif",
        ),
        source="chitin:models.bif",
        source_path=r"H:\K2\data\models.bif",
        priority=40,
    )


def _provider(
    *,
    mdl_bytes: bytes = b"head-mdl",
    include_invalid: bool = False,
) -> InMemoryGameResourceProvider:
    rows = [
        (_record("PFHA04", "MDL"), mdl_bytes),
        (_record("PFHA04", "MDX"), b"head-mdx"),
        (_record("PFBAM", "MDL"), b"body-mdl"),
        (_record("PFBAM", "MDX"), b"body-mdx"),
        (_record("S_Female03", "MDL"), b"female03-mdl"),
        (_record("S_Female03", "MDX"), b"female03-mdx"),
        (_record("S_Female02", "MDL"), b"female02-mdl"),
        (_record("S_Female02", "MDX"), b"female02-mdx"),
    ]
    if include_invalid:
        rows.extend(
            [
                (_record("PLC_BENCH", "MDL"), b"invalid-mdl"),
                (_record("PLC_BENCH", "MDX"), b"invalid-mdx"),
            ]
        )
    return InMemoryGameResourceProvider(rows)


def _loader(mdl: bytes, _mdx: bytes, _game: str) -> KotorModel:
    if mdl == b"invalid-mdl":
        return _invalid_model()
    if mdl == b"body-mdl":
        return deepcopy(_body_model())
    if mdl == b"female03-mdl":
        return deepcopy(_female03_model())
    if mdl == b"female02-mdl":
        return deepcopy(_female02_model())
    return deepcopy(_head_model())


def _service(
    provider: InMemoryGameResourceProvider,
    *,
    install_verifier=None,
) -> HeadBuilderService:
    return HeadBuilderService(
        repository=FileHeadBuilderProjectRepository(),
        donor_catalog=HeadDonorCatalog(provider),
        model_loader=_loader,
        install_verifier=install_verifier,
    )


def _component_service() -> HeadBuilderService:
    provider = InMemoryGameResourceProvider(
        [
            (_record("PFHA04", "MDL"), b"component-carrier-mdl"),
            (_record("PFHA04", "MDX"), b"component-carrier-mdx"),
            (_record("PFHA01", "MDL"), b"component-source-mdl"),
            (_record("PFHA01", "MDX"), b"component-source-mdx"),
        ]
    )

    def loader(mdl: bytes, _mdx: bytes, _game: str) -> KotorModel:
        if mdl == b"component-source-mdl":
            return deepcopy(_component_head_model("PFHA01", hair_count=1))
        return deepcopy(_component_head_model("PFHA04", hair_count=2))

    return HeadBuilderService(
        repository=FileHeadBuilderProjectRepository(),
        donor_catalog=HeadDonorCatalog(provider),
        model_loader=loader,
    )


def _write_head_obj(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "o custom_head",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "vt 0 0",
                "vt 1 0",
                "vt 0 1",
                "f 1/1 2/2 3/3",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def _write_donor_surface_head_obj(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "o donor_surface_head",
                "v -0.1 0 0",
                "v 0.1 0 0",
                "v 0 0 0.2",
                "vt 0 0",
                "vt 1 0",
                "vt 0.5 1",
                "f 1/1 2/2 3/3",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def _write_head_texture(path: Path) -> None:
    from PIL import Image

    image = Image.new("RGBA", (4, 4), (48, 96, 160, 255))
    image.putpixel((0, 0), (255, 255, 255, 64))
    image.save(path, format="TGA")


def _write_head_test_game(root: Path) -> None:
    override = root / "Override"
    override.mkdir(parents=True)
    (root / "swkotor2.exe").write_bytes(b"test game executable")
    heads = TwoDA(
        [
            "head",
            "alttexture",
            "headtexvvve",
            "headtexvve",
            "headtexve",
            "headtexe",
            "headtexg",
            "headtexvg",
        ]
    )
    heads.add_row(
        "0",
        {
            "head": "PFHA04",
            "alttexture": "PFHA04",
        },
    )
    appearance = TwoDA(
        ["label", "normalhead", "modela", "texa", "portrait"]
    )
    appearance.add_row(
        "0",
        {
            "label": "P_FEM_A_SML_04",
            "normalhead": "0",
            "modela": "PFBAM",
            "texa": "PFBAL",
            "portrait": "",
        },
    )
    (override / "heads.2da").write_bytes(bytes(bytes_2da(heads)))
    (override / "appearance.2da").write_bytes(
        bytes(bytes_2da(appearance))
    )


def _test_game_verification(root: Path) -> HeadGameInstallVerification:
    return HeadGameInstallVerification(
        game="K2",
        install_dir=str(root),
        executable_path=str(root / "swkotor2.exe"),
        executable_size=(root / "swkotor2.exe").stat().st_size,
        executable_sha256="a" * 64,
        chitin_key_path=str(root / "chitin.key"),
        chitin_key_size=1,
        chitin_key_sha256="b" * 64,
        chitin_signature="KEY V1  ",
        resource_probe_resref="PFHA04",
        resource_probe_mdl_sha256="c" * 64,
        resource_probe_mdx_sha256="d" * 64,
        resource_probe_source="chitin:models.bif",
        resource_probe_readable=True,
    )


def test_select_donor_records_provenance_snapshot_evidence_and_progress(
    tmp_path: Path,
) -> None:
    service = _service(_provider(include_invalid=True))
    project = service.new_project(
        display_name="Custom Hero",
        game="K2",
        path=tmp_path / "hero.ghosthead.json",
    )
    service.configure_game(
        game="K2",
        resource_view="stock_only",
        game_install_dir=r"H:\K2",
        output_project_dir=str(tmp_path / "output"),
        output_head_resref="P_CUSTOMH",
        character_context={"body": "PFBAM"},
    )

    rows = service.search_donors("pfha")
    advanced_rows = service.search_donors(
        "plc_bench",
        include_nonstandard=True,
    )
    selection = service.select_donor(rows[0].resref)

    assert selection.accepted is True
    assert [row.resref for row in advanced_rows] == ["PLC_BENCH"]
    assert service.selected_model is selection.model
    assert project.resources["native_donor_mdl"].origin is ResourceOrigin.CHITIN_BIF
    assert project.resources["native_donor_mdl"].stock is True
    assert project.resources["native_donor_mdx"].sha256
    assert project.donor_contract["snapshot"]["resref"] == "PFHA04"
    assert project.donor_contract["eligibility"]["eligible"] is True
    evidence = project.validation_results[-1]
    assert evidence.outcome is EvidenceOutcome.PASS
    assert evidence.hashes["structural_sha256"]
    progress = project.workflow_steps[HeadBuilderStep.SELECT_NATIVE_DONOR]
    assert progress.status is StepStatus.COMPLETE
    assert progress.evidence_ids == [evidence.evidence_id]
    assert project.current_step is HeadBuilderStep.ALIGN_NECK_AND_HOOK
    assert service.dirty is True


def test_vanilla_component_recipe_saves_and_rehydrates_without_custom_mesh(
    tmp_path: Path,
) -> None:
    project_path = tmp_path / "stock-mix.ghosthead.json"
    first = _component_service()
    project = first.new_project(game="K2", path=project_path)
    first.select_donor("PFHA04")

    result = first.configure_vanilla_component_recipe(
        face_resref="PFHA01",
        eyes_resref="PFHA01",
        eyelashes_resref="PFHA01",
        hair_resref="PFHA01",
        recipe_name="PFHA01 parts on PFHA04",
    )
    first.save_project()

    assert result.report.accepted
    assert first.candidate_model is result.model
    assert project.import_art == {}
    assert project.appearance_customization["mode"] == "vanilla_components"
    assert project.appearance_customization["selections"]["eyes"] == "PFHA01"
    assert project.workflow_steps[
        HeadBuilderStep.IMPORT_CUSTOM_ART
    ].status is StepStatus.COMPLETE
    assert project.workflow_steps[
        HeadBuilderStep.ALIGN_NECK_AND_HOOK
    ].status is StepStatus.COMPLETE
    assert project.workflow_steps[
        HeadBuilderStep.REPLACE_GEOMETRY_AND_SKIN
    ].status is StepStatus.COMPLETE
    assert project.current_step is HeadBuilderStep.UV_TEXTURES_AND_MATERIALS
    assert result.model.find_node("head").texture == "PFHA01_FACE"
    assert result.model.find_node("eyeLA").texture == "PFHA01_EYE"
    assert result.model.find_node("hair02").vertices == []

    second = _component_service()
    reopened = second.open_project(project_path)
    second.rehydrate_selected_donor()
    rebuilt = second.rehydrate_vanilla_component_recipe()

    assert reopened.appearance_customization["recipe_name"] == (
        "PFHA01 parts on PFHA04"
    )
    assert rebuilt.report.component_payload_sha256 == (
        result.report.component_payload_sha256
    )
    assert second.candidate_model is rebuilt.model


def test_custom_art_alignment_saves_reopens_and_rehydrates_without_blobs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "custom_head.obj"
    project_path = tmp_path / "aligned.ghosthead.json"
    _write_head_obj(source)
    first = _service(_provider())
    project = first.new_project(game="K2", path=project_path)
    imported = first.import_custom_art(source)
    first.select_donor("PFHA04")

    result = first.align_custom_art(
        [
            HeadAlignmentAnchor("neck_center", (0, 0, 0), (5, 2, 3)),
            HeadAlignmentAnchor("neck_left", (1, 0, 0), (5, 3, 3)),
            HeadAlignmentAnchor("neck_front", (0, 1, 0), (4, 2, 3)),
        ],
        headhook_to_body=(
            (1, 0, 0, 5),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        ),
        body_resref="PFBAM",
        headhook_node_path="PFBAM/torso/headhook",
    )
    first.save_project()

    assert imported.accepted
    assert result.rms_error == pytest.approx(0.0, abs=1.0e-10)
    assert transform_point(
        result.imported_to_headhook,
        (0, 0, 0),
    ) == pytest.approx((0, 2, 3))
    assert project.workflow_steps[
        HeadBuilderStep.IMPORT_CUSTOM_ART
    ].status is StepStatus.COMPLETE
    assert project.workflow_steps[
        HeadBuilderStep.ALIGN_NECK_AND_HOOK
    ].status is StepStatus.COMPLETE
    assert project.current_step is HeadBuilderStep.REPLACE_GEOMETRY_AND_SKIN
    assert project.resources["custom_head_art"].origin is ResourceOrigin.IMPORTED_FILE
    saved_text = project_path.read_text(encoding="utf-8")
    assert '"vertices"' not in saved_text
    assert '"faces"' not in saved_text
    assert '"source_path": "./custom_head.obj"' in saved_text

    second = _service(_provider())
    reopened = second.open_project(project_path)
    rehydrated_art = second.rehydrate_custom_art()
    second.rehydrate_selected_donor()

    assert rehydrated_art.document.structural_sha256 == (
        imported.document.structural_sha256
    )
    assert reopened.alignment["result"]["transform_sha256"] == (
        result.transform_sha256
    )
    assert second.dirty is False


def test_import_records_texture_provenance_and_rejects_unsupported_cleanup(
    tmp_path: Path,
) -> None:
    source = tmp_path / "custom_head.obj"
    texture = tmp_path / "custom_head_source.png"
    _write_head_obj(source)
    texture.write_bytes(b"source texture bytes")
    service = _service(_provider())
    project = service.new_project(game="K2")

    imported = service.import_custom_art(
        source,
        source_texture_paths=(texture,),
        cleanup_policy={
            "normal_policy": "recalculate_missing",
            "triangulate": True,
            "weld_exact_duplicates": False,
        },
    )

    assert imported.accepted
    settings = project.import_art["settings"]
    assert settings["normal_policy"] == "recalculate_missing"
    assert settings["source_texture_paths"] == [str(texture.resolve())]
    provenance = project.resources["custom_head_source_texture_1"]
    assert provenance.source_path == str(texture.resolve())
    assert provenance.sha256
    assert provenance.metadata["role"] == "source_texture"

    with pytest.raises(HeadBuilderServiceError, match="welding is not available"):
        service.import_custom_art(
            source,
            cleanup_policy={"weld_exact_duplicates": True},
        )


def test_project_snapshot_restore_is_json_safe_and_resets_runtime(
    tmp_path: Path,
) -> None:
    service = _service(_provider())
    service.new_project(display_name="Snapshot Baseline", game="K2")
    service.configure_game(
        game="K2",
        resource_view="stock_only",
        output_project_dir=str(tmp_path / "first"),
        output_head_resref="P_SNAP01",
    )
    snapshot = service.snapshot_project()
    service.select_donor("PFHA04")

    restored = service.restore_project_snapshot(snapshot)

    assert restored.display_name == "Snapshot Baseline"
    assert restored.output_head_resref == "P_SNAP01"
    assert restored.donor_contract == {}
    assert service.selected_model is None
    assert service.dirty


def test_transplant_weight_edit_saves_reopens_and_preserves_donor_contract(
    tmp_path: Path,
) -> None:
    source = tmp_path / "donor_surface_head.obj"
    project_path = tmp_path / "transplanted.ghosthead.json"
    _write_donor_surface_head_obj(source)
    first = _service(_provider())
    project = first.new_project(game="K2", path=project_path)
    imported = first.import_custom_art(source)
    selected = first.select_donor("PFHA04")
    part = imported.document.parts[0]
    first.align_custom_art(
        [
            HeadAlignmentAnchor(
                f"surface_{index}",
                point,
                point,
            )
            for index, point in enumerate(part.vertices)
        ],
        headhook_to_body=(
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        ),
        body_resref="PFBAM",
        headhook_node_path="PFBAM/torso/headhook",
    )
    vertex_ids = [
        f"{part.part_id}:v:{index}"
        for index in range(len(part.vertices))
    ]
    transplanted = first.transplant_geometry_and_skin(
        part_modes={part.part_id: "surface_transfer"},
        neck_vertex_ids=vertex_ids,
        maximum_surface_distance=1.0e-6,
        allow_distance_fallback=False,
        minimum_neck_weight=0.05,
    )
    edited = first.edit_skin_weights(
        vertex_ids[0],
        {"head_g": 0.9, "neck_g": 0.1},
    )

    with pytest.raises(HeadBuilderServiceError, match="below"):
        first.edit_skin_weights(
            vertex_ids[0],
            {"head_g": 1.0},
        )

    diff = first.compare_donor_contract()
    first.save_project()

    assert transplanted.report.accepted
    assert edited.report.manual_edit_count == 1
    assert edited.report.payload_sha256 != transplanted.report.payload_sha256
    assert not diff.blocking
    assert selected.snapshot.structural_sha256 == (
        project.donor_contract["snapshot"]["structural_sha256"]
    )
    assert project.workflow_steps[
        HeadBuilderStep.REPLACE_GEOMETRY_AND_SKIN
    ].status is StepStatus.COMPLETE
    assert project.current_step is HeadBuilderStep.UV_TEXTURES_AND_MATERIALS
    saved_text = project_path.read_text(encoding="utf-8")
    assert '"vertices"' not in saved_text
    assert '"faces"' not in saved_text
    assert '"manual_edits"' in saved_text

    second = _service(_provider())
    reopened = second.open_project(project_path)
    second.rehydrate_custom_art()
    second.rehydrate_selected_donor()
    second.rehydrate_alignment()
    rebuilt = second.rehydrate_transplant()
    rebuilt_diff = second.compare_donor_contract()

    assert rebuilt.report.geometry_sha256 == edited.report.geometry_sha256
    assert rebuilt.report.final_weight_rows_sha256 == (
        edited.report.final_weight_rows_sha256
    )
    assert rebuilt.report.payload_sha256 == edited.report.payload_sha256
    assert rebuilt.report.manual_edit_count == 1
    assert not rebuilt_diff.blocking
    assert reopened.current_step is HeadBuilderStep.UV_TEXTURES_AND_MATERIALS
    assert second.dirty is False


def test_reset_all_skin_weight_edits_restores_transfer_baseline(
    tmp_path: Path,
) -> None:
    source = tmp_path / "reset_all_head.obj"
    _write_donor_surface_head_obj(source)
    service = _service(_provider())
    project = service.new_project(game="K2")
    imported = service.import_custom_art(source)
    service.select_donor("PFHA04")
    part = imported.document.parts[0]
    service.align_custom_art(
        [
            HeadAlignmentAnchor(f"surface_{index}", point, point)
            for index, point in enumerate(part.vertices)
        ],
        headhook_to_body=(
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        ),
        body_resref="PFBAM",
        headhook_node_path="PFBAM/torso/headhook",
    )
    vertex_ids = [
        f"{part.part_id}:v:{index}"
        for index in range(len(part.vertices))
    ]
    baseline = service.transplant_geometry_and_skin(
        part_modes={part.part_id: "surface_transfer"},
        neck_vertex_ids=vertex_ids,
        maximum_surface_distance=1.0e-6,
        allow_distance_fallback=False,
        minimum_neck_weight=0.05,
    )
    service.edit_skin_weights(
        vertex_ids[0],
        {"head_g": 0.9, "neck_g": 0.1},
    )

    reset = service.reset_all_skin_weight_edits()

    assert reset.rows == baseline.rows
    assert reset.report.manual_edit_count == 0
    assert project.skin_transfer["manual_edits"] == {}
    assert not service.compare_donor_contract().blocking


def test_uv_texture_material_saves_reopens_with_matching_orientation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "donor_surface_head.obj"
    texture = tmp_path / "custom_head_diffuse.tga"
    project_path = tmp_path / "textured.ghosthead.json"
    game_root = tmp_path / "game"
    _write_head_test_game(game_root)
    verification = _test_game_verification(game_root)
    _write_donor_surface_head_obj(source)
    _write_head_texture(texture)
    first = _service(
        _provider(),
        install_verifier=lambda _game, _path: verification,
    )
    project = first.new_project(game="K2", path=project_path)
    first.configure_game(
        game="K2",
        resource_view="stock_only",
        game_install_dir=str(game_root),
        output_project_dir=str(tmp_path / "project-output"),
        output_head_resref="P_CUSTOMH",
    )
    first.verify_game_install()
    imported = first.import_custom_art(source, flip_v=False)
    first.select_donor("PFHA04")
    part = imported.document.parts[0]
    first.align_custom_art(
        [
            HeadAlignmentAnchor(
                f"surface_{index}",
                point,
                point,
            )
            for index, point in enumerate(part.vertices)
        ],
        headhook_to_body=(
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        ),
        body_resref="PFBAM",
        headhook_node_path="PFBAM/torso/headhook",
    )
    vertex_ids = [
        f"{part.part_id}:v:{index}"
        for index in range(len(part.vertices))
    ]
    first.transplant_geometry_and_skin(
        part_modes={part.part_id: "surface_transfer"},
        neck_vertex_ids=vertex_ids,
        maximum_surface_distance=1.0e-6,
        allow_distance_fallback=False,
    )
    textured = first.configure_uv_texture_materials(
        texture,
        output_texture_resref="P_CDH01",
        output_format="TGA",
        serialized_uv_transform="identity",
        preview_uv_transform="identity",
        txi_delivery="sidecar",
        alpha_mode="punchthrough",
        clamp_s=True,
        clamp_t=True,
    )
    preview = first.preview_attachment_and_animations(
        body_resref="PFBAM"
    )
    preflight = first.run_binary_preflight()
    if preflight.unacknowledged_warning_ids:
        preflight = first.acknowledge_preflight_warnings(
            preflight.unacknowledged_warning_ids
        )
    binary_export = first.export_verified_binary(
        output_dir=tmp_path / "export"
    )
    package = first.build_game_records_package(
        appearance_donor_label="P_FEM_A_SML_04",
        package_directory=tmp_path / "package",
    )
    install_preview = first.prepare_test_install()
    install = first.install_prepared_test(
        confirmed_preview_id=install_preview.preview_id
    )
    restore = first.restore_previous_test()
    diff = first.compare_donor_contract()
    first.save_project()

    assert textured.report.accepted
    assert textured.report.preview_matches_serialized
    assert textured.report.packaged_files == (
        "P_CDH01.tga",
        "P_CDH01.txi",
    )
    assert first.candidate_model.all_nodes()[4].texture == "P_CDH01"
    assert preview.report.accepted
    assert preview.report.preview_head_parent_name == "headhook"
    assert preview.report.source_head_local_animation_names == ()
    assert preview.report.preview_head_local_animation_names == ()
    assert set(preview.report.selected_animation_names) == {
        "tlknorm",
        "talk",
        "listen",
        "walk",
    }
    assert {"tlknorm", "talk", "listen"}.issubset(
        set(preview.report.facial_animation_names)
    )
    assert preflight.export_allowed
    assert Path(binary_export.mdl_path).is_file()
    assert Path(binary_export.mdx_path).is_file()
    assert Path(binary_export.manifest_path).is_file()
    assert package.ok
    assert package.reference_merge is not None
    assert package.reference_merge.heads_row == 1
    assert package.reference_merge.appearance_row == 1
    assert install_preview.ok
    assert install.ok
    assert restore.ok
    assert not diff.blocking
    assert project.workflow_steps[
        HeadBuilderStep.UV_TEXTURES_AND_MATERIALS
    ].status is StepStatus.COMPLETE
    assert project.current_step is HeadBuilderStep.SAFE_RETAIL_TEST
    assert project.workflow_steps[
        HeadBuilderStep.GAME_RECORDS_AND_PACKAGE
    ].status is StepStatus.COMPLETE
    assert project.workflow_steps[
        HeadBuilderStep.SAFE_RETAIL_TEST
    ].status is StepStatus.IN_PROGRESS
    assert project.workflow_steps[
        HeadBuilderStep.ATTACHMENT_AND_ANIMATION_PREVIEW
    ].status is StepStatus.COMPLETE
    assert project.workflow_steps[
        HeadBuilderStep.OPTIONAL_HAIR_PHYSICS
    ].status is StepStatus.COMPLETE
    assert project.physics["status"] == "not_requested"
    saved_text = project_path.read_text(encoding="utf-8")
    assert '"rgba"' not in saved_text
    assert '"vertices"' not in saved_text
    assert '"source_path": "./custom_head_diffuse.tga"' in saved_text

    second = _service(
        _provider(),
        install_verifier=lambda _game, _path: verification,
    )
    reopened = second.open_project(project_path)
    second.rehydrate_custom_art()
    second.rehydrate_selected_donor()
    second.rehydrate_alignment()
    second.rehydrate_transplant()
    rebuilt = second.rehydrate_uv_texture_materials()
    rebuilt_preview = second.rehydrate_attachment_preview()
    reopen_snapshot = deepcopy(reopened.to_dict())
    restored_preflight = second.rehydrate_binary_preflight()

    assert restored_preflight.report_sha256 == preflight.report_sha256
    assert reopened.to_dict() == reopen_snapshot
    assert reopened.current_step is HeadBuilderStep.SAFE_RETAIL_TEST
    assert second.dirty is False

    reopened_install_preview = second.prepare_test_install()
    rebuilt_preflight = second.run_binary_preflight()

    assert rebuilt.report.material_payload_sha256 == (
        textured.report.material_payload_sha256
    )
    assert rebuilt.report.serialized_uv_sha256 == (
        textured.report.serialized_uv_sha256
    )
    assert second.candidate_model.all_nodes()[4].texture == "P_CDH01"
    assert rebuilt_preview.report.contract_sha256 == (
        preview.report.contract_sha256
    )
    assert rebuilt_preflight.report_sha256 == preflight.report_sha256
    assert reopened_install_preview.ok
    assert reopened_install_preview.heads_row == 1
    assert reopened_install_preview.appearance_row == 1
    assert (
        reopened.current_step
        is HeadBuilderStep.GAME_RECORDS_AND_PACKAGE
    )
    second.save_project()
    assert second.dirty is False


def test_custom_art_drift_blocks_rehydrate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "drift.obj"
    project_path = tmp_path / "drift.ghosthead.json"
    _write_head_obj(source)
    first = _service(_provider())
    first.new_project(game="K2", path=project_path)
    first.import_custom_art(source)
    first.save_project()
    source.write_text(
        source.read_text(encoding="utf-8") + "# externally changed\n",
        encoding="utf-8",
    )

    reopened = _service(_provider())
    reopened.open_project(project_path)
    with pytest.raises(HeadBuilderArtChangedError, match="bytes"):
        reopened.rehydrate_custom_art()


def test_rejected_art_does_not_replace_accepted_import(
    tmp_path: Path,
) -> None:
    accepted = tmp_path / "accepted.obj"
    rejected = tmp_path / "rejected.obj"
    _write_head_obj(accepted)
    rejected.write_text(
        "\n".join(
            (
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 0 -1 0",
                "v 0 0 1",
                "f 1 2 3",
                "f 2 1 4",
                "f 1 2 5",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    service = _service(_provider())
    project = service.new_project(game="K2")
    selected = service.import_custom_art(accepted)
    accepted_contract = deepcopy(project.import_art)

    with pytest.raises(HeadBuilderArtRejectedError) as raised:
        service.import_custom_art(rejected)

    assert not raised.value.report.accepted
    assert project.import_art == accepted_contract
    assert service.imported_art is selected.document
    assert project.workflow_steps[
        HeadBuilderStep.IMPORT_CUSTOM_ART
    ].status is StepStatus.COMPLETE


def test_alignment_above_tolerance_stays_in_progress(
    tmp_path: Path,
) -> None:
    source = tmp_path / "loose.obj"
    _write_head_obj(source)
    service = _service(_provider())
    project = service.new_project(game="K2")
    service.import_custom_art(source)
    service.select_donor("PFHA04")

    result = service.align_custom_art(
        [
            HeadAlignmentAnchor("a", (0, 0, 0), (0, 0, 0)),
            HeadAlignmentAnchor("b", (1, 0, 0), (1, 0, 0)),
            HeadAlignmentAnchor("c", (0, 1, 0), (0, 2, 0)),
        ],
        headhook_to_body=(
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        ),
        body_resref="PFBAM",
        headhook_node_path="headhook",
        maximum_rms_error=0.001,
    )

    assert result.rms_error > 0.001
    assert project.alignment["within_tolerance"] is False
    assert project.workflow_steps[
        HeadBuilderStep.ALIGN_NECK_AND_HOOK
    ].status is StepStatus.IN_PROGRESS
    assert project.validation_results[-1].outcome is EvidenceOutcome.WARNING


def test_alignment_requires_exact_headhook_context(
    tmp_path: Path,
) -> None:
    source = tmp_path / "context.obj"
    _write_head_obj(source)
    service = _service(_provider())
    service.new_project(game="K2")
    service.import_custom_art(source)
    service.select_donor("PFHA04")

    with pytest.raises(HeadBuilderServiceError, match="exact native headhook"):
        service.align_custom_art(
            [HeadAlignmentAnchor("neck", (0, 0, 0), (0, 0, 0))],
            headhook_to_body=(
                (1, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1),
            ),
            body_resref="PFBAM",
            headhook_node_path="head_hook",
        )


def test_saved_project_reopens_rehydrates_same_donor_and_self_compares(
    tmp_path: Path,
) -> None:
    provider = _provider()
    path = tmp_path / "roundtrip.ghosthead.json"
    first = _service(provider)
    first.new_project(game="K2", path=path)
    first.configure_game(game="K2", resource_view="stock_only")
    selected = first.select_donor("PFHA04")
    first.save_project()

    second = _service(provider)
    reopened = second.open_project(path)
    rehydrated = second.rehydrate_selected_donor()
    diff = second.compare_donor_contract()

    assert reopened.project_id == first.project.project_id
    assert rehydrated.snapshot.structural_sha256 == (
        selected.snapshot.structural_sha256
    )
    assert second.selected_model is rehydrated.model
    assert diff.structurally_compatible is True
    assert diff.allowed_payload_changes == ()
    assert second.dirty is False


def test_rehydrate_blocks_when_installed_donor_bytes_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "drift.ghosthead.json"
    first = _service(_provider(mdl_bytes=b"head-mdl"))
    first.new_project(game="K2", path=path)
    first.configure_game(game="K2", resource_view="stock_only")
    first.select_donor("PFHA04")
    first.save_project()

    changed = _service(_provider(mdl_bytes=b"changed-head-mdl"))
    changed.open_project(path)

    with pytest.raises(HeadBuilderDonorChangedError, match="no longer match"):
        changed.rehydrate_selected_donor()


def test_invalid_resource_is_rejected_without_replacing_accepted_contract(
    tmp_path: Path,
) -> None:
    service = _service(_provider(include_invalid=True))
    project = service.new_project(
        game="K2",
        path=tmp_path / "reject.ghosthead.json",
    )
    service.configure_game(game="K2", resource_view="stock_only")
    service.select_donor("PFHA04")
    accepted_contract = deepcopy(project.donor_contract)

    with pytest.raises(HeadBuilderDonorRejectedError) as raised:
        service.select_donor("PLC_BENCH")

    assert raised.value.report.eligible is False
    assert project.donor_contract == accepted_contract
    progress = project.workflow_steps[HeadBuilderStep.SELECT_NATIVE_DONOR]
    assert progress.status is StepStatus.COMPLETE
    assert progress.evidence_ids != [project.validation_results[-1].evidence_id]
    rejected = project.validation_results[-1]
    assert rejected.outcome is EvidenceOutcome.FAIL
    assert "head.donor.model_type" in {
        issue["check_id"]
        for issue in rejected.metadata["eligibility"]["issues"]
    }


def test_compare_uses_explicit_project_output_resref() -> None:
    service = _service(_provider())
    project = service.new_project(game="K2")
    service.configure_game(
        game="K2",
        resource_view="stock_only",
        output_head_resref="P_CUSTOMH",
    )
    service.select_donor("PFHA04")
    output = deepcopy(service.selected_model)
    output.name = "P_CUSTOMH"
    output.root_node.name = "P_CUSTOMH"

    diff = service.compare_donor_contract(output)

    assert project.output_head_resref == "P_CUSTOMH"
    assert diff.structurally_compatible is True
    assert "nodes[0].name" in {
        row.path for row in diff.allowed_payload_changes
    }


def test_install_verification_records_evidence_and_completes_configured_step(
    tmp_path: Path,
) -> None:
    verification = HeadGameInstallVerification(
        game="K2",
        install_dir=r"H:\K2",
        executable_path=r"H:\K2\swkotor2.exe",
        executable_size=10,
        executable_sha256="a" * 64,
        chitin_key_path=r"H:\K2\chitin.key",
        chitin_key_size=20,
        chitin_key_sha256="b" * 64,
        chitin_signature="KEY V1  ",
        resource_probe_resref="PFHA04",
        resource_probe_mdl_sha256="c" * 64,
        resource_probe_mdx_sha256="d" * 64,
        resource_probe_source="chitin:models.bif",
        resource_probe_readable=True,
    )
    service = _service(
        _provider(),
        install_verifier=lambda _game, _path: verification,
    )
    project = service.new_project(game="K2")
    service.configure_game(
        game="K2",
        resource_view="stock_only",
        game_install_dir=r"H:\K2",
        output_project_dir=str(tmp_path / "head-project"),
        output_head_resref="P_CUSTOMH",
    )

    result = service.verify_game_install()

    assert result is verification
    assert project.extensions["game_install_verification"]["verified"] is True
    progress = project.workflow_steps[HeadBuilderStep.PROJECT_GAME]
    assert progress.status is StepStatus.COMPLETE
    evidence = next(
        row
        for row in project.validation_results
        if row.check_id == "head.game_install.read_only"
    )
    assert evidence.outcome is EvidenceOutcome.PASS
    assert evidence.hashes["executable_sha256"] == "A" * 64
    assert progress.evidence_ids == [evidence.evidence_id]


def test_retail_pass_requires_complete_user_confirmed_observer_evidence(
    tmp_path: Path,
) -> None:
    service = _service(_provider())
    project = service.new_project(game="K2")
    project.package_state = {
        "install_session": {
            "ok": True,
            "session_manifest": str(tmp_path / "install-session.json"),
        }
    }
    capture = tmp_path / "retail-observer.txt"
    capture.write_text("idle walk combat dialogue save warp", encoding="utf-8")
    complete = {
        "idle": True,
        "movement": True,
        "combat": True,
        "dialogue": True,
        "save_load": True,
        "warp": True,
        "attachment": True,
        "texture": True,
    }

    with pytest.raises(HeadBuilderServiceError, match="incomplete"):
        service.confirm_retail_test_pass(
            observer_session="observer-1",
            checklist={**complete, "combat": False},
            artifact_paths=(capture,),
            confirmed_by_user=True,
        )
    with pytest.raises(HeadBuilderServiceError, match="explicit"):
        service.confirm_retail_test_pass(
            observer_session="observer-1",
            checklist=complete,
            artifact_paths=(capture,),
            confirmed_by_user=False,
        )

    evidence = service.confirm_retail_test_pass(
        observer_session="observer-1",
        checklist=complete,
        artifact_paths=(capture,),
        confirmed_by_user=True,
    )

    assert evidence.level.value == "retail_observed"
    assert evidence.outcome is EvidenceOutcome.PASS
    assert evidence.confirmed_by_user is True
    assert evidence.observer_session == "observer-1"
    assert project.retail_test["passed"] is True
    assert project.workflow_steps[
        HeadBuilderStep.SAFE_RETAIL_TEST
    ].status is StepStatus.COMPLETE


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("P_CUSTOMH", True),
        ("head_12345678901", True),
        ("", False),
        ("more_than_sixteen_", False),
        ("has space", False),
        ("head-name", False),
        ("héád", False),
    ],
)
def test_output_head_resref_uses_odyssey_field_contract(
    value: str,
    expected: bool,
) -> None:
    assert valid_head_output_resref(value) is expected
