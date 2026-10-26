"""BAS composed export: a head grafted onto a headless body exports as one model.

The Body Attachment System composes the body and every attached layer (head,
mask, weapons, belt) into a single KotorModel whose hook nodes carry the
layers as real child subtrees.  The main-viewport "Export Composed Model…"
button writes that one model straight through the MDL/OBJ/FBX workers, so a
headless body plus any game head leaves the tool as a single merged file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _configure_native_python_roots() -> None:
    from scripts.mcp.start_kotormcp_stdio import _python_roots

    for item in reversed(_python_roots(ROOT)):
        value = str(item)
        if value not in sys.path:
            sys.path.insert(0, value)


def _body_with_headhook():
    from src.core.geometry import model_data as md

    root = md.ModelNode(name="pmbam", flags=int(md.NodeFlags.HEADER))
    body = md.ModelNode(
        name="pmbam_body",
        flags=int(md.NodeFlags.HEADER) | int(md.NodeFlags.MESH),
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
        texture="pmbam01",
    )
    body.parent = root
    headhook = md.ModelNode(name="headhook", flags=int(md.NodeFlags.HEADER))
    headhook.position = (0.0, 0.0, 1.8)
    headhook.parent = root
    lhand = md.ModelNode(name="lhand", flags=int(md.NodeFlags.HEADER), parent=root)
    rhand = md.ModelNode(name="rhand", flags=int(md.NodeFlags.HEADER), parent=root)
    mask_hook = md.ModelNode(name="MaskHook", flags=int(md.NodeFlags.HEADER), parent=root)
    goggle_hook = md.ModelNode(name="GoggleHook", flags=int(md.NodeFlags.HEADER), parent=root)
    pelvis = md.ModelNode(name="pelvis_g", flags=int(md.NodeFlags.HEADER), parent=root)
    root.children.extend([body, headhook, lhand, rhand, mask_hook, goggle_hook, pelvis])
    return md.KotorModel(name="pmbam", root_node=root)


def _head_model():
    from src.core.geometry import model_data as md

    root = md.ModelNode(name="pmhc01", flags=int(md.NodeFlags.HEADER))
    head = md.ModelNode(
        name="pmhc01_head",
        flags=int(md.NodeFlags.HEADER) | int(md.NodeFlags.MESH),
        vertices=[(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.0)],
        faces=[(0, 1, 2), (1, 3, 2)],
        texture="pmhc01",
    )
    head.parent = root
    root.children.append(head)
    return md.KotorModel(name="pmhc01", root_node=head and root)


def _attachment_model(name: str):
    from src.core.geometry import model_data as md

    root = md.ModelNode(name=f"{name}_root", flags=int(md.NodeFlags.HEADER))
    mesh = md.ModelNode(
        name=f"{name}_mesh",
        flags=int(md.NodeFlags.HEADER) | int(md.NodeFlags.MESH),
        parent=root,
        vertices=[(0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.0, 0.25, 0.0)],
        normals=[(0.0, 0.0, 1.0)] * 3,
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        faces=[(0, 1, 2)],
        texture=f"{name}_diff",
    )
    root.children.append(mesh)
    return md.KotorModel(name=name, root_node=root)


def _node_names(model):
    return {str(getattr(node, "name", "") or "").lower() for node in model.all_nodes()}


def test_head_is_grafted_into_the_composed_model_tree() -> None:
    _configure_native_python_roots()
    from src.systems.bas.preview_composer import build_bas_preview_model

    composed = build_bas_preview_model(
        body_model=_body_with_headhook(),
        attachment_models={"head": _head_model()},
        name="pmbam_bas",
    )
    names = _node_names(composed)
    assert "headhook" in names
    assert "pmhc01_head" in names, "head mesh must be present in the merged tree"
    # The head subtree hangs under the headhook socket, not loose at the root.
    headhook = next(n for n in composed.all_nodes() if str(n.name).lower() == "headhook")
    child_names = {str(getattr(c, "name", "") or "").lower() for c in getattr(headhook, "children", [])}
    assert "pmhc01" in child_names or "pmhc01_head" in child_names


def test_canonical_preview_can_keep_nonvisual_inventory_without_failing() -> None:
    _configure_native_python_roots()
    from src.systems.bas.preview_composer import build_bas_preview_model

    from src.core.geometry import model_data as md

    body_root = md.ModelNode(name="turret", flags=int(md.NodeFlags.HEADER))
    body = md.KotorModel(name="turret", root_node=body_root)
    weapon_root = md.ModelNode(name="blaster", flags=int(md.NodeFlags.HEADER))
    weapon = md.KotorModel(name="w_blstrpstl_001", root_node=weapon_root)
    weapon._gr_source_resref = "w_blstrpstl_001"

    preview = build_bas_preview_model(
        body_model=body,
        attachment_models={"right_weapon": weapon},
        skip_unsupported=True,
    )

    assert preview.name == "turret"
    assert preview._gr_bas_skipped_attachments == (
        {
            "slot": "right_weapon",
            "socket": "rhand",
            "resref": "w_blstrpstl_001",
        },
    )


def test_composed_export_preserves_head_skin_for_shared_facial_animation_frame() -> None:
    _configure_native_python_roots()
    from src.core.geometry import model_data as md
    from src.systems.bas.preview_composer import (
        build_bas_preview_model,
        prepare_bas_composed_export_model,
    )

    body = _body_with_headhook()
    head_root = md.ModelNode(name="n_test_head", flags=int(md.NodeFlags.HEADER))
    head_shell = md.ModelNode(
        name="head",
        flags=int(md.NodeFlags.HEADER) | int(md.NodeFlags.MESH) | int(md.NodeFlags.SKIN),
        parent=head_root,
        position=(0.0, 0.0, 2.0),
        vertices=[(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0)],
        faces=[(0, 1, 2)],
    )
    head_root.children.append(head_shell)
    preview = build_bas_preview_model(
        body_model=body,
        attachment_models={"head": md.KotorModel(name="n_test_head", root_node=head_root)},
    )

    exported, report = prepare_bas_composed_export_model(preview, require_unique_body_names=True)
    exported_shell = next(
        node for node in exported.all_nodes()
        if str(getattr(node, "name", "")).casefold() == "head"
        and bool(getattr(node, "vertices", None))
    )

    assert exported_shell.position == pytest.approx((0.0, 0.0, 2.0))
    assert exported_shell.is_skin is True
    assert report["rigid_head_geometry"] == []
    assert report["animated_head_geometry"] == ["head"]


def test_character_builder_bas_reads_the_canonical_scene_slot() -> None:
    """BAS must not lose a body to a duplicate legacy PartSlot enum."""

    _configure_native_python_roots()
    from src.core.geometry import model_data as md
    from src.gui.qt_lib.panels.qt_character_builder_panel import (
        QtCharacterBuilderWindow,
    )

    scene = md.CharacterScene(game_version="K2")
    body = _body_with_headhook()
    scene.assign(
        md.PartSlot.HEADLESS_BODY,
        body,
        resref="pfbc09",
        game_version="K2",
    )
    harness = type(
        "CharacterBuilderBasHarness",
        (),
        {"scene": scene, "_bas_preview_body": None},
    )()

    assert QtCharacterBuilderWindow._cb_bas_body_model(harness) is body


def test_composed_model_exports_as_one_mdl_with_the_head() -> None:
    _configure_native_python_roots()
    from src.core.mdl.mdl_parser import MDLBinaryParser
    from src.core.mdl.mdl_writer import MDLBinaryWriter
    from src.systems.bas.preview_composer import build_bas_preview_model

    composed = build_bas_preview_model(
        body_model=_body_with_headhook(),
        attachment_models={"head": _head_model()},
        name="pmbam_bas",
    )
    mdl_bytes, mdx_bytes = MDLBinaryWriter().write(composed)
    assert mdl_bytes and mdx_bytes
    reloaded = MDLBinaryParser(mdl_bytes, mdx_bytes).parse()
    names = _node_names(reloaded)
    assert "pmbam_body" in names
    assert "pmhc01_head" in names, "the grafted head must survive the single-MDL roundtrip"


def test_composed_model_exports_as_one_obj_with_the_head(tmp_path) -> None:
    _configure_native_python_roots()
    from src.converters.mesh_converter import OBJExporter
    from src.systems.bas.preview_composer import build_bas_preview_model

    composed = build_bas_preview_model(
        body_model=_body_with_headhook(),
        attachment_models={"head": _head_model()},
        name="pmbam_bas",
    )
    obj_path = tmp_path / "pmbam_bas.obj"
    OBJExporter().export(composed, str(obj_path), export_rigging=False)
    text = obj_path.read_text(encoding="utf-8", errors="ignore").lower()
    # The head geometry (4 verts) must appear alongside the body (3 verts):
    # a single OBJ carries both, so at least 7 vertex rows are present.
    assert text.count("\nv ") + text.startswith("v ") >= 7
    assert "pmhc01_head" in text or "pmhc01" in text


def test_obj_bakes_rotated_attachment_hierarchy_with_fbx_bind_math(tmp_path) -> None:
    _configure_native_python_roots()
    from src.converters.mesh_converter import OBJExporter
    from src.core.geometry.model_data import KotorModel, ModelNode, NodeFlags, _quat_rotate

    root = ModelNode(name="root", flags=int(NodeFlags.HEADER))
    parent_rotation = (
        0.9993784703983151,
        0.0021312797225932945,
        0.034913866595109005,
        0.004376353555986939,
    )
    socket = ModelNode(
        name="rhand",
        flags=int(NodeFlags.HEADER),
        parent=root,
        position=(0.3, 0.4, 0.5),
        rotation=parent_rotation,
    )
    mesh = ModelNode(
        name="weapon_mesh",
        flags=int(NodeFlags.HEADER) | int(NodeFlags.MESH),
        parent=socket,
        position=(0.01, 0.02, 0.03),
        vertices=[(0.05, 0.06, 0.07), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0)],
        normals=[(0.0, 0.0, 1.0)] * 3,
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        faces=[(0, 1, 2)],
        texture="weapon_diff",
    )
    root.children.append(socket)
    socket.children.append(mesh)
    model = KotorModel(name="rotated_weapon", root_node=root)
    path = tmp_path / "rotated_weapon.obj"

    OBJExporter().export(model, str(path), export_rigging=False)
    first_vertex = next(
        tuple(float(value) for value in line.split()[1:4])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("v ")
    )
    rotated = _quat_rotate(
        parent_rotation,
        tuple(mesh.position[index] + mesh.vertices[0][index] for index in range(3)),
    )
    expected = tuple(socket.position[index] + rotated[index] for index in range(3))
    assert first_vertex == pytest.approx(expected, abs=1e-6)


def test_export_button_and_handler_are_wired() -> None:
    _configure_native_python_roots()
    panel = (ROOT / "native/GhostRigger.Core.GUI.Display/Python/src/gui/panels/qt_body_attachment_panel.py").read_text(encoding="utf-8")
    assert "exportComposedRequested" in panel
    assert "Export Composed Model" in panel
    layout = (ROOT / "native/GhostRigger.Core.GUI.Display/Python/src/gui/windows/application_core/shared/main_layout.py").read_text(encoding="utf-8")
    assert "exportComposedRequested.connect(self._handle_bas_export_composed_requested)" in layout
    workflow = (ROOT / "native/GhostRigger.Core.GUI.Display/Python/src/gui/windows/application_core/shared/bas_workflow.py").read_text(encoding="utf-8")
    assert "_handle_bas_export_composed_requested" in workflow
    assert "preview = build_bas_preview_model(" in workflow
    # Tools mirrors must match byte-for-byte.
    for rel in (
        "gui/panels/qt_body_attachment_panel.py",
        "gui/windows/application_core/shared/main_layout.py",
        "gui/windows/application_core/shared/bas_workflow.py",
    ):
        gui = (ROOT / "native/GhostRigger.Core.GUI.Display/Python/src" / rel).read_bytes()
        tools = (ROOT / "native/GhostRigger.Core.Tools/Python/src" / rel).read_bytes()
        assert gui == tools, f"mirror drift: {rel}"


def test_composed_export_normalizes_duplicate_attachment_nodes_and_bone_maps() -> None:
    _configure_native_python_roots()
    from src.core.geometry import model_data as md
    from src.systems.bas.preview_composer import (
        build_bas_preview_model,
        prepare_bas_composed_export_model,
    )

    weapon_root = md.ModelNode(name="weapon_root", flags=int(md.NodeFlags.HEADER))
    weapon_skin = md.ModelNode(
        name="weapon_mesh",
        flags=int(md.NodeFlags.HEADER) | int(md.NodeFlags.MESH) | int(md.NodeFlags.SKIN),
        parent=weapon_root,
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
        bone_map=["weapon_root"],
        skin_data=[md.VertexSkinData([md.BoneWeight(0, 1.0)]) for _ in range(3)],
    )
    weapon_root.children.append(weapon_skin)
    weapon = md.KotorModel(name="weapon", root_node=weapon_root)
    preview = build_bas_preview_model(
        body_model=_body_with_headhook(),
        attachment_models={"left_weapon": weapon, "right_weapon": weapon},
        name="dual_weapon_build",
    )

    prepared, report = prepare_bas_composed_export_model(preview)
    names = [str(node.name).lower() for node in prepared.all_nodes()]
    assert report["unique_names"] is True
    assert report["renamed_count"] == 2
    assert len(names) == len(set(names))
    right_skin = next(
        node for node in prepared.all_nodes()
        if str(getattr(getattr(node, "_gr_bas_attachment_root_ref", None), "_gr_bas_attachment_slot", "")) == "right_weapon"
        and bool(getattr(node, "is_skin", False))
    )
    assert right_skin.bone_map == ["weapon_root__right_weapon"]


def test_composed_head_export_replaces_hidden_body_face_and_remaps_facial_tracks(
    tmp_path,
) -> None:
    _configure_native_python_roots()
    from src.converters.mesh_converter import FBXExporter, OBJExporter
    from src.core.animation.fbx_animation_selection import (
        prepare_fbx_animation_export_model,
    )
    from src.core.geometry import model_data as md
    from src.systems.bas.preview_composer import (
        build_bas_preview_model,
        prepare_bas_composed_export_model,
    )

    body = _body_with_headhook()
    body_head = md.ModelNode(name="head_g", flags=int(md.NodeFlags.HEADER), parent=body.root_node)
    body_jaw = md.ModelNode(name="f_jaw_g", flags=int(md.NodeFlags.HEADER), parent=body_head)
    body_eye = md.ModelNode(
        name="eyeLA",
        flags=int(md.NodeFlags.HEADER) | int(md.NodeFlags.MESH),
        parent=body_head,
        render=False,
        vertices=[(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0)],
        normals=[(0.0, 0.0, 1.0)] * 3,
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        faces=[(0, 1, 2)],
        texture="hidden_body_eye",
    )
    body.root_node.children.append(body_head)
    body_head.children.extend([body_jaw, body_eye])
    body.animations = [
        md.Animation(
            name="talk",
            length=1.0,
            nodes=[md.ModelNode(name="headhook")],
        )
    ]

    head_root = md.ModelNode(name="head_source", flags=int(md.NodeFlags.HEADER))
    head_bone = md.ModelNode(name="head_g", flags=int(md.NodeFlags.HEADER), parent=head_root)
    head_jaw = md.ModelNode(name="f_jaw_g", flags=int(md.NodeFlags.HEADER), parent=head_bone)
    head_eye = md.ModelNode(
        name="eyeLA",
        flags=int(md.NodeFlags.HEADER) | int(md.NodeFlags.MESH),
        parent=head_bone,
        # Several stock KotOR heads mark valid eye/teeth meshes render=0.  The
        # exporter intentionally recognizes those facial meshes by their data
        # and name, so they must still replace the body's hidden duplicate.
        render=False,
        vertices=[(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0)],
        normals=[(0.0, 0.0, 1.0)] * 3,
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        faces=[(0, 1, 2)],
        texture="visible_head_eye",
    )
    head_root.children.append(head_bone)
    head_bone.children.extend([head_jaw, head_eye])
    head = md.KotorModel(name="head_source", root_node=head_root)
    head.animations = [
        md.Animation(
            name="talk",
            length=1.0,
            nodes=[
                md.ModelNode(name="head_g"),
                md.ModelNode(name="f_jaw_g"),
                md.ModelNode(name="eyeLA"),
            ],
        )
    ]

    preview = build_bas_preview_model(
        body_model=body,
        attachment_models={"head": head},
    )
    composed, report = prepare_bas_composed_export_model(
        preview,
        require_unique_body_names=True,
    )
    prepared = prepare_fbx_animation_export_model(
        composed,
        ("talk",),
        game="K1",
        supplemental_models=(head,),
    )

    assert report["suppressed_body_geometry"] == ["eyeLA"]
    head_layer = next(row for row in report["attachment_layers"] if row["slot"] == "head")
    assert head_layer["source_model"] == "head_source"
    assert head_layer["renamed_nodes"]["f_jaw_g"] == "f_jaw_g__head"
    assert head_layer["renamed_nodes"]["eyela"] == "eyeLA__head"
    track_names = [node.name for node in prepared.animations[0].nodes]
    assert track_names == ["headhook", "f_jaw_g__head", "eyeLA__head"]
    assert "head_g__head" not in track_names

    hidden_eye = next(
        node for node in prepared.all_nodes()
        if node.name == "eyeLA" and not bool(getattr(node, "_gr_bas_attachment_layer", False))
    )
    attached_eye = next(node for node in prepared.all_nodes() if node.name == "eyeLA__head")
    assert OBJExporter._is_renderable(hidden_eye) is False
    assert OBJExporter._is_renderable(attached_eye) is True

    output = tmp_path / "facial_replacement.fbx"
    assert FBXExporter()._export_fbx_ascii(
        prepared,
        str(output),
        compatibility_profile="unity",
    )
    text = output.read_text(encoding="utf-8")
    assert '"Geometry::eyeLA__head", "Mesh"' in text
    assert '"Geometry::eyeLA", "Mesh"' not in text


def test_all_attachable_bas_slots_export_in_one_unity_fbx(tmp_path) -> None:
    _configure_native_python_roots()
    from src.converters.mesh_converter import FBXExporter
    from src.systems.bas.preview_composer import (
        build_bas_preview_model,
        prepare_bas_composed_export_model,
    )

    attachments = {
        "head": _head_model(),
        "mask": _attachment_model("mask"),
        "goggles": _attachment_model("goggles"),
        "left_weapon": _attachment_model("left_weapon"),
        "belt": _attachment_model("belt"),
        "right_weapon": _attachment_model("right_weapon"),
    }
    preview = build_bas_preview_model(
        body_model=_body_with_headhook(),
        attachment_models=attachments,
        attachment_transforms={
            "mask": {
                "position": [0.1, 0.2, 0.3],
                "rotation": [0.0, 0.0, 0.70710678, 0.70710678],
            },
        },
        name="all_slots_build",
    )
    prepared, report = prepare_bas_composed_export_model(preview)
    names = _node_names(prepared)
    for expected in (
        "pmhc01_head",
        "mask_mesh",
        "goggles_mesh",
        "left_weapon_mesh",
        "belt_mesh",
        "right_weapon_mesh",
    ):
        assert expected in names
    assert report["unique_names"] is True

    out_path = tmp_path / "all_slots_build.fbx"
    assert FBXExporter().export(
        prepared,
        str(out_path),
        export_rigging=False,
        compatibility_profile="unity",
    )
    text = out_path.read_text(encoding="utf-8")
    assert text.count('Geometry: ') == 7
    assert "; Compatibility profile: unity" in text
    mask_block = text.split('"Model::mask_root",', 1)[1].split("\n\t}", 1)[0]
    assert 'Lcl Translation","Lcl Translation","","A",0.100000,0.200000,0.300000' in mask_block
    assert 'Lcl Rotation","Lcl Rotation","","A",0.0000,0.0000,90.0000' in mask_block


def test_bas_unity_filter_routes_profile_and_base_skeleton_to_fbx_worker(tmp_path, monkeypatch) -> None:
    _configure_native_python_roots()
    from src.gui.windows.application_core.shared import bas_workflow

    preview = bas_workflow.copy.deepcopy(_body_with_headhook())
    captured = {}
    texture_cache = object()
    base_skeleton = object()

    class _Panel:
        def set_status(self, message):
            captured["status"] = str(message)

    class _Harness(bas_workflow.BasWorkflowMixin):
        body_attachment_panel = _Panel()
        _bas_body_model = preview
        _current_model = preview
        _bas_preview_model = preview
        _bas_active_build_name = "unity_composed"
        _bas_attachment_resrefs = {"head": "pmhc01"}
        _bas_attachments = {"head": _head_model()}

        def _rebuild_bas_preview(self):
            return "BAS preview updated."

        def _get_tex_cache_for_export(self):
            return texture_cache

        def _fbx_base_skeleton_for_export(self, model):
            captured["base_model_input"] = model
            return base_skeleton

        def _run_io_async(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        def _log(self, *_args, **_kwargs):
            pass

    dialog_output = tmp_path / "unity_composed.mdl"
    expected_output = tmp_path / "unity_composed.fbx"
    monkeypatch.setattr(
        bas_workflow.QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(dialog_output), "Unity-Compatible FBX (*.fbx)"),
    )

    _Harness()._handle_bas_export_composed_requested()

    assert captured["args"][1].__name__ == "_work_export_fbx"
    assert captured["args"][3] == str(expected_output)
    assert captured["kwargs"]["compatibility_profile"] == "unity"
    assert captured["kwargs"]["base_skeleton_model"] is base_skeleton
    assert captured["kwargs"]["tex_cache"] is texture_cache
    assert captured["base_model_input"] is captured["args"][2]


def test_bas_unreal_filter_routes_empty_animation_selection_and_resource_context(
    tmp_path,
    monkeypatch,
) -> None:
    _configure_native_python_roots()
    from src.gui.windows.application_core.shared import bas_workflow

    preview = bas_workflow.copy.deepcopy(_body_with_headhook())
    head = _head_model()
    captured = {}
    texture_cache = object()
    base_skeleton = object()
    resource_manager = object()

    class _Panel:
        def set_status(self, message):
            captured["status"] = str(message)

    class _Harness(bas_workflow.BasWorkflowMixin):
        body_attachment_panel = _Panel()
        _bas_body_model = preview
        _current_model = preview
        _bas_preview_model = preview
        _bas_active_build_name = "unreal_mesh_only"
        _bas_attachment_resrefs = {"head": "pmhc01"}
        _bas_attachments = {"head": head}

        def _rebuild_bas_preview(self):
            return "BAS preview updated."

        def _get_tex_cache_for_export(self):
            return texture_cache

        def _fbx_base_skeleton_for_export(self, model):
            captured["base_model_input"] = model
            return base_skeleton

        def _choose_fbx_animation_sets(self, model, profile, **kwargs):
            captured["chooser"] = (model, profile, kwargs)
            # Empty is an explicit mesh-and-rig-only choice, not cancellation.
            return ()

        def _fbx_resource_context_for_export(self, model):
            captured["context_model"] = model
            return resource_manager, "K2"

        def _run_io_async(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        def _log(self, *_args, **_kwargs):
            pass

    dialog_output = tmp_path / "unreal_mesh_only.mdl"
    expected_output = tmp_path / "unreal_mesh_only.fbx"
    monkeypatch.setattr(
        bas_workflow.QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (
            str(dialog_output),
            "Unreal Engine-Compatible FBX (*.fbx)",
        ),
    )

    _Harness()._handle_bas_export_composed_requested()

    assert captured["args"][1].__name__ == "_work_export_fbx"
    assert captured["args"][3] == str(expected_output)
    assert captured["kwargs"]["compatibility_profile"] == "unreal"
    assert captured["kwargs"]["selected_animation_names"] == ()
    assert isinstance(captured["kwargs"]["selected_animation_names"], tuple)
    assert captured["kwargs"]["animation_resource_manager"] is resource_manager
    assert captured["kwargs"]["animation_game"] == "K2"
    assert captured["kwargs"]["base_skeleton_model"] is base_skeleton
    assert captured["kwargs"]["tex_cache"] is texture_cache
    assert captured["kwargs"]["supplemental_animation_models"] == (head,)
    chooser_model, chooser_profile, chooser_kwargs = captured["chooser"]
    assert chooser_model is captured["args"][2]
    assert chooser_profile == "unreal"
    assert chooser_kwargs["base_skeleton_model"] is base_skeleton
    assert chooser_kwargs["supplemental_models"] == (head,)
    assert captured["context_model"] is captured["args"][2]


def test_bas_body_slot_strict_loads_selected_game_and_preserves_attachments() -> None:
    _configure_native_python_roots()
    from src.gui.windows.application_core.shared import bas_workflow

    previous_body = _body_with_headhook()
    replacement = _body_with_headhook()
    replacement.name = "bodyk2"
    head = _head_model()
    calls: list[tuple[str, str]] = []

    class _Manager:
        def load_model_strict(self, resref, game):
            calls.append((resref, game))
            return replacement

    class _Panel:
        status = ""

        def selected_model_resref(self):
            return "bodyk2"

        def selected_model_game(self):
            return "K2"

        def set_mode(self, mode):
            self.mode = mode

        def set_body_model(self, model, **kwargs):
            self.body = model
            self.body_kwargs = kwargs

        def set_status(self, message):
            self.status = str(message)

    class _Harness(bas_workflow.BasWorkflowMixin):
        body_attachment_panel = _Panel()
        _bas_body_model = previous_body
        _current_model = previous_body
        _current_game = "K1"
        _bas_attachments = {"head": head}
        _bas_attachment_resrefs = {"head": "pmhc01"}
        _bas_attachment_transforms = {"head": {}}
        _bas_preview_model = None
        _bas_active_build_name = "old_build"

        def _get_resource_manager(self):
            return _Manager()

        def _infer_game_from_model(self, _model):
            return "K1"

        def _rebuild_bas_preview(self):
            self.rebuilt_with = self._bas_body_model
            return "BAS preview updated: pmhc01."

    harness = _Harness()
    harness._handle_bas_attach_requested("body", "bodyk2")

    assert calls == [("bodyk2", "K2")]
    assert harness._bas_body_model is replacement
    assert harness._current_model is replacement
    assert harness._current_game == "K2"
    assert harness._bas_attachments == {"head": head}
    assert harness.rebuilt_with is replacement
    assert harness.body_attachment_panel.body_kwargs == {"resref": "bodyk2", "game": "K2"}
    assert "existing" not in harness.body_attachment_panel.status.lower()


def test_bas_body_slot_failed_load_keeps_the_previous_body() -> None:
    _configure_native_python_roots()
    from src.gui.windows.application_core.shared import bas_workflow

    previous_body = _body_with_headhook()

    class _Manager:
        def load_model_strict(self, _resref, _game):
            return None

    class _Panel:
        status = ""

        def selected_model_resref(self):
            return "missingbody"

        def selected_model_game(self):
            return "K2"

        def set_status(self, message):
            self.status = str(message)

    class _Harness(bas_workflow.BasWorkflowMixin):
        body_attachment_panel = _Panel()
        _bas_body_model = previous_body
        _current_model = previous_body
        _current_game = "K1"

        def _get_resource_manager(self):
            return _Manager()

        def _infer_game_from_model(self, _model):
            return "K1"

    harness = _Harness()
    harness._handle_bas_attach_requested("body", "missingbody")

    assert harness._bas_body_model is previous_body
    assert harness._current_model is previous_body
    assert harness._current_game == "K1"
    assert "could not load k2:missingbody" in harness.body_attachment_panel.status.lower()


def test_bas_recipe_preserves_cross_game_attachment_provenance() -> None:
    _configure_native_python_roots()
    from src.systems.bas.model_recipe import build_bas_model_recipe

    body = _body_with_headhook()
    body._gr_source_resref = "pmbam"
    body._gr_source_game = "K1"
    head = _head_model()
    head._gr_source_resref = "pmhc01"
    head._gr_source_game = "K2"

    recipe = build_bas_model_recipe(
        body_model=body,
        attachment_models={"head": head},
        attachment_resrefs={"head": "pmhc01"},
        game="K1",
    )
    layers = {row["slot"]: row for row in recipe["layers"]}
    assert layers["body"]["game"] == "K1"
    assert layers["head"]["game"] == "K2"


def test_native_body_duplicate_names_are_preserved_and_dcc_export_is_blocked() -> None:
    _configure_native_python_roots()
    from src.core.geometry import model_data as md
    from src.systems.bas.preview_composer import prepare_bas_composed_export_model

    body = _body_with_headhook()
    first = md.ModelNode(name="native_duplicate", flags=int(md.NodeFlags.HEADER), parent=body.root_node)
    second = md.ModelNode(name="native_duplicate", flags=int(md.NodeFlags.HEADER), parent=body.root_node)
    body.root_node.children.extend([first, second])

    native_copy, report = prepare_bas_composed_export_model(body)
    assert [node.name for node in native_copy.all_nodes()].count("native_duplicate") == 2
    assert report["duplicate_body_names"] == ["native_duplicate"]
    with pytest.raises(ValueError, match="unique body node names"):
        prepare_bas_composed_export_model(body, require_unique_body_names=True)


def test_ambiguous_duplicate_names_inside_one_attachment_block_export() -> None:
    _configure_native_python_roots()
    from src.core.geometry import model_data as md
    from src.systems.bas.preview_composer import (
        build_bas_preview_model,
        prepare_bas_composed_export_model,
    )

    root = md.ModelNode(name="attachment_root", flags=int(md.NodeFlags.HEADER))
    first = md.ModelNode(name="bone", flags=int(md.NodeFlags.HEADER), parent=root)
    second = md.ModelNode(name="bone", flags=int(md.NodeFlags.HEADER), parent=root)
    root.children.extend([first, second])
    attachment = md.KotorModel(name="ambiguous", root_node=root)
    preview = build_bas_preview_model(
        body_model=_body_with_headhook(),
        attachment_models={"head": attachment},
    )

    with pytest.raises(ValueError, match="duplicate node names"):
        prepare_bas_composed_export_model(preview)


def test_unbaked_attachment_scale_blocks_dcc_export_instead_of_exporting_wrong_size() -> None:
    _configure_native_python_roots()
    from src.systems.bas.preview_composer import (
        build_bas_preview_model,
        prepare_bas_composed_export_model,
    )

    preview = build_bas_preview_model(
        body_model=_body_with_headhook(),
        attachment_models={"head": _head_model()},
        attachment_transforms={"head": {"scale": [1.25, 1.25, 1.25]}},
    )
    with pytest.raises(ValueError, match="unbaked BAS layer scale"):
        prepare_bas_composed_export_model(preview)
