"""Shared ExportJob transaction helper tests."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.export.export_job import (
    ExportJobContext,
    ExportJobRequest,
    ExportJobStatus,
    ExportOutputSpec,
    promote_staged_files,
    run_export_job,
)
from src.core.ports import FileWriterPort
from src.core.validation.validation_bus import (
    ValidationBus,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
    ValidationSubsystem,
)


def _blocking_report(message: str = "blocked") -> ValidationReport:
    return ValidationReport(
        source="test.export",
        issues=[
            ValidationIssue(
                severity=ValidationSeverity.BLOCKING,
                subsystem=ValidationSubsystem.EXPORT,
                code="test.blocking",
                message=message,
            )
        ],
    )


def _request(output: Path, **kwargs) -> ExportJobRequest:
    return ExportJobRequest(
        job_id=kwargs.pop("job_id", "job1"),
        kind=kwargs.pop("kind", "test_export"),
        outputs=kwargs.pop("outputs", [ExportOutputSpec(output, "txt")]),
        **kwargs,
    )


def test_successful_export_writes_staging_then_promotes(tmp_path: Path) -> None:
    final = tmp_path / "out" / "test.txt"
    seen_staged: list[Path] = []

    def writer(context: ExportJobContext) -> None:
        staged = context.staged_path_for(final)
        assert not final.exists()
        assert isinstance(context, FileWriterPort)
        context.write_text(final, "hello", encoding="utf-8")
        seen_staged.append(staged)

    result = run_export_job(_request(final), writer=writer)

    assert result.succeeded is True
    assert result.status == ExportJobStatus.SUCCEEDED
    assert final.read_text(encoding="utf-8") == "hello"
    assert seen_staged and not seen_staged[0].parent.exists()


def test_dynamic_staged_file_promotion_restores_existing_set_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.core.export import export_job

    staging = tmp_path / "staging"
    target = tmp_path / "target"
    staging.mkdir()
    target.mkdir()
    (staging / "module.obj").write_text("new obj", encoding="utf-8")
    (staging / "module.mtl").write_text("new mtl", encoding="utf-8")
    (target / "module.obj").write_text("old obj", encoding="utf-8")
    (target / "module.mtl").write_text("old mtl", encoding="utf-8")
    real_replace = export_job.os.replace

    def fail_obj_promotion(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path == staging / "module.obj" and destination_path == target / "module.obj":
            raise PermissionError("geometry file is locked")
        return real_replace(source, destination)

    monkeypatch.setattr(export_job.os, "replace", fail_obj_promotion)

    with pytest.raises(OSError, match="geometry file is locked"):
        promote_staged_files(staging, target)

    assert (target / "module.obj").read_text(encoding="utf-8") == "old obj"
    assert (target / "module.mtl").read_text(encoding="utf-8") == "old mtl"


def test_export_context_file_writer_port_writes_bytes_to_staging(tmp_path: Path) -> None:
    final = tmp_path / "out" / "payload.bin"
    seen_staged: list[Path] = []

    def writer(context: ExportJobContext) -> None:
        assert isinstance(context, FileWriterPort)
        staged = context.staged_path_for(final)
        context.write_bytes(final, b"payload")
        seen_staged.append(staged)
        assert staged.exists()
        assert final.exists() is False

    result = run_export_job(_request(final), writer=writer)

    assert result.succeeded is True
    assert final.read_bytes() == b"payload"
    assert seen_staged and not seen_staged[0].parent.exists()


def test_preflight_blocking_report_prevents_writer_call(tmp_path: Path) -> None:
    final = tmp_path / "blocked.txt"
    called = False
    request = _request(final, preflight_report=_blocking_report())

    def writer(_context: ExportJobContext) -> None:
        nonlocal called
        called = True

    result = run_export_job(request, writer=writer)

    assert result.status == ExportJobStatus.PREFLIGHT_FAILED
    assert called is False
    assert not final.exists()


def test_overwrite_false_blocks_existing_output_before_writer(tmp_path: Path) -> None:
    final = tmp_path / "exists.txt"
    final.write_text("original", encoding="utf-8")
    called = False

    def writer(_context: ExportJobContext) -> None:
        nonlocal called
        called = True

    result = run_export_job(_request(final, overwrite=False), writer=writer)

    assert result.status == ExportJobStatus.PREFLIGHT_FAILED
    assert called is False
    assert final.read_text(encoding="utf-8") == "original"
    assert any(issue.code == "export.output.exists" for issue in result.validation_report.issues)


def test_verifier_failure_prevents_promotion(tmp_path: Path) -> None:
    final = tmp_path / "verified.txt"

    def writer(context: ExportJobContext) -> None:
        context.staged_path_for(final).write_text("staged", encoding="utf-8")

    result = run_export_job(
        _request(final),
        writer=writer,
        verifier=lambda _context: _blocking_report("verification failed"),
    )

    assert result.status == ExportJobStatus.FAILED
    assert not final.exists()
    assert any("verification failed" in issue.message for issue in result.validation_report.issues)


def test_writer_exception_cleans_staging(tmp_path: Path) -> None:
    final = tmp_path / "boom.txt"
    staged_parent: list[Path] = []

    def writer(context: ExportJobContext) -> None:
        staged = context.staged_path_for(final)
        staged.write_text("partial", encoding="utf-8")
        staged_parent.append(staged.parent)
        raise RuntimeError("boom")

    result = run_export_job(_request(final), writer=writer)

    assert result.status == ExportJobStatus.FAILED
    assert not final.exists()
    assert staged_parent and not staged_parent[0].exists()
    assert any("boom" in issue.message for issue in result.validation_report.issues)


def test_duplicate_final_output_paths_fail_preflight(tmp_path: Path) -> None:
    final = tmp_path / "dup.txt"
    called = False

    def writer(_context: ExportJobContext) -> None:
        nonlocal called
        called = True

    request = _request(
        final,
        outputs=[
            ExportOutputSpec(final, "txt"),
            ExportOutputSpec(final, "txt-copy"),
        ],
    )
    result = run_export_job(request, writer=writer)

    assert result.status == ExportJobStatus.PREFLIGHT_FAILED
    assert called is False
    assert any(issue.code == "export.output.duplicate" for issue in result.validation_report.issues)


def test_validation_bus_receives_reports(tmp_path: Path) -> None:
    final = tmp_path / "bus.txt"
    bus = ValidationBus()
    request = _request(
        final,
        preflight_report=_blocking_report("bus failure"),
        validation_bus_source="export.job.test",
    )

    run_export_job(request, writer=lambda _context: None, validation_bus=bus)

    snapshot = bus.snapshot()
    assert any(issue.source == "export.job.test" for issue in snapshot.issues)
    assert any("bus failure" in issue.message for issue in snapshot.issues)


def test_non_serializable_metadata_fails_preflight(tmp_path: Path) -> None:
    final = tmp_path / "bad.txt"
    called = False

    def writer(_context: ExportJobContext) -> None:
        nonlocal called
        called = True

    result = run_export_job(_request(final, metadata={"bad": b"bytes"}), writer=writer)

    assert result.status == ExportJobStatus.PREFLIGHT_FAILED
    assert called is False
    assert any("non-JSON-serializable" in issue.message for issue in result.validation_report.issues)


def test_multi_directory_outputs_stage_and_promote_together(tmp_path: Path) -> None:
    out_a = tmp_path / "a" / "one.txt"
    out_b = tmp_path / "b" / "two.txt"
    seen_staging_dirs: list[Path] = []

    def writer(context: ExportJobContext) -> None:
        context.write_text(out_a, "one", encoding="utf-8")
        context.write_text(out_b, "two", encoding="utf-8")
        staged_a = context.staged_path_for(out_a)
        staged_b = context.staged_path_for(out_b)
        assert staged_a.exists()
        assert staged_b.exists()
        assert staged_a.parent != staged_b.parent
        assert not out_a.exists()
        assert not out_b.exists()
        seen_staging_dirs.extend([staged_a.parent, staged_b.parent])

    result = run_export_job(
        _request(
            out_a,
            outputs=[
                ExportOutputSpec(out_a, "txt"),
                ExportOutputSpec(out_b, "txt"),
            ],
        ),
        writer=writer,
    )

    assert result.status == ExportJobStatus.SUCCEEDED
    assert out_a.read_text(encoding="utf-8") == "one"
    assert out_b.read_text(encoding="utf-8") == "two"
    assert Path(result.staged_paths[str(out_a)]).name == "one.txt"
    assert Path(result.staged_paths[str(out_b)]).name == "two.txt"
    assert seen_staging_dirs and all(not path.exists() for path in seen_staging_dirs)


def test_multi_directory_staging_does_not_repeat_long_absolute_paths(tmp_path: Path) -> None:
    output_root = tmp_path / ("user_selected_map_studio_package_" + ("x" * 64))
    module = output_root / "install" / "Modules" / "grstyles.mod"
    room = output_root / "source" / "resources" / "grstylesr001.mdl"
    seen_staged: list[Path] = []

    def writer(context: ExportJobContext) -> None:
        seen_staged.extend((context.staged_path_for(module), context.staged_path_for(room)))
        context.write_bytes(module, b"mod")
        context.write_bytes(room, b"mdl")

    result = run_export_job(
        _request(
            module,
            job_id="map_studio.custom_module_package.grstyles.with_a_long_descriptive_job_name",
            outputs=(
                ExportOutputSpec(module, "module_package"),
                ExportOutputSpec(room, "loose_resource"),
            ),
        ),
        writer=writer,
    )

    assert result.succeeded is True
    assert module.read_bytes() == b"mod"
    assert room.read_bytes() == b"mdl"
    assert seen_staged
    assert all(str(output_root.resolve()).lower() not in path.parent.name.lower() for path in seen_staged)


def test_manifest_writer_output_is_promoted(tmp_path: Path) -> None:
    final = tmp_path / "payload.txt"
    manifest = tmp_path / "payload.manifest.json"

    def writer(context: ExportJobContext) -> None:
        context.staged_path_for(final).write_text("payload", encoding="utf-8")

    def manifest_writer(context: ExportJobContext, result) -> Path:
        staged_manifest = context.staged_path_for(manifest)
        context.write_text(
            manifest,
            json.dumps({"job_id": result.job_id, "kind": result.kind}),
            encoding="utf-8",
        )
        return staged_manifest

    result = run_export_job(
        _request(
            final,
            outputs=[
                ExportOutputSpec(final, "txt"),
                ExportOutputSpec(manifest, "manifest"),
            ],
        ),
        writer=writer,
        manifest_writer=manifest_writer,
    )

    assert result.succeeded is True
    assert result.manifest_path == manifest
    assert manifest.exists()
    assert json.loads(manifest.read_text(encoding="utf-8"))["job_id"] == "job1"


def _selected_export_triangle_model(name: str, texture: str):
    from src.core.geometry.model_data import KotorModel, ModelNode, NodeFlags

    mesh = ModelNode(
        name=f"{name}_mesh",
        flags=int(NodeFlags.HEADER | NodeFlags.MESH),
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        normals=[(math.sqrt(0.5), math.sqrt(0.5), 0.0)] * 3,
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        uvs_lm=[(0.1, 0.2), (0.8, 0.2), (0.1, 0.9)],
        faces=[(0, 1, 2)],
        face_mats=[0],
        texture=texture,
        texture_names=[texture],
        tex_count=1,
    )
    return KotorModel(name=name, classification="tile", root_node=mesh)


def test_selected_fbx_merge_bakes_transforms_and_preserves_material_uv_channels() -> None:
    from src.io.fbx.fbx_exporter import merge_selected_scene_objects

    first = _selected_export_triangle_model("first", "stone")
    first.root_node.position = (1.0, 0.0, 0.0)
    second = _selected_export_triangle_model("second", "sky")
    objects = [
        SimpleNamespace(
            name="First Room",
            metadata={"_runtime_model": first},
            transform=SimpleNamespace(
                position=(10.0, 0.0, 0.0),
                rotation=(0.0, 0.0, 0.0),
                scale=(2.0, 1.0, 1.0),
            ),
        ),
        SimpleNamespace(
            name="Skybox",
            metadata={"_runtime_model": second},
            transform=SimpleNamespace(
                position=(0.0, 5.0, 0.0),
                rotation=(0.0, 0.0, 90.0),
                scale=(1.0, 2.0, 1.0),
            ),
        ),
    ]

    merged = merge_selected_scene_objects(objects, name="module_selection")
    mesh = merged.root_node

    assert mesh is not None
    assert merged.name == "module_selection"
    assert len(merged.mesh_nodes()) == 1
    assert len(mesh.faces) == 2
    assert len(mesh.vertices) == 6
    expected_vertices = [
        (12.0, 0.0, 0.0),
        (14.0, 0.0, 0.0),
        (12.0, 1.0, 0.0),
        (0.0, 5.0, 0.0),
        (0.0, 6.0, 0.0),
        (-2.0, 5.0, 0.0),
    ]
    for actual, expected in zip(mesh.vertices, expected_vertices):
        assert actual == pytest.approx(expected)
    assert mesh.normals[0] == pytest.approx((math.sqrt(0.2), math.sqrt(0.8), 0.0))
    assert mesh.normals[3] == pytest.approx((-math.sqrt(0.2), math.sqrt(0.8), 0.0))
    assert mesh.uvs == first.root_node.uvs + second.root_node.uvs
    assert mesh.uvs_lm == first.root_node.uvs_lm + second.root_node.uvs_lm
    assert mesh.face_mats == [0, 1]
    assert [slot.texture for slot in mesh._gr_fbx_material_slots] == ["stone", "sky"]


def test_builtin_ascii_fbx_exports_selected_objects_as_one_multimaterial_mesh(
    tmp_path: Path,
) -> None:
    from src.converters.mesh_converter import FBXExporter
    from src.io.fbx.fbx_exporter import merge_selected_scene_objects

    selected = [
        SimpleNamespace(
            name=f"Room {index + 1}",
            metadata={
                "_runtime_model": _selected_export_triangle_model(
                    f"room_{index}",
                    texture,
                )
            },
            transform=SimpleNamespace(
                position=(float(index) * 2.0, 0.0, 0.0),
                rotation=(0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
            ),
        )
        for index, texture in enumerate(("stone", "sky"))
    ]
    merged = merge_selected_scene_objects(selected)
    output = tmp_path / "selection.fbx"

    assert FBXExporter().export(
        merged,
        str(output),
        export_rigging=False,
        export_manifest=False,
        force_ascii=True,
    )

    text = output.read_text(encoding="utf-8")
    assert text.count("\tGeometry:") == 1
    assert text.count("\tMaterial:") == 2
    assert 'MappingInformationType: "ByPolygon"' in text
    assert "Materials: *2" in text
    assert "\t\t\t\ta: 0,1" in text


def test_export_selected_module_routes_marquee_selection_to_one_mesh_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.gui.windows.application_core.shared import model_io

    room = SimpleNamespace(
        id="room-id",
        name="m14aa_01c",
        metadata={"module_group": {"module_root": "m14aa_01"}},
    )
    sky = SimpleNamespace(
        id="sky-id",
        name="m14aa_01h",
        metadata={"module_group": {"module_root": "m14aa_01"}},
    )
    captured: dict[str, object] = {}
    output = tmp_path / "module_selection.fbx"

    class _Harness(model_io.ModelIoMixin):
        scene_manager = SimpleNamespace(
            active_scene=SimpleNamespace(objects=[room, sky]),
            get_selected_objects=lambda: [sky],
        )
        viewport = SimpleNamespace(
            _selected_viewport_nodes=[
                SimpleNamespace(_gr_scene_object_id="room-id"),
                SimpleNamespace(_gr_scene_object_id="sky-id"),
            ]
        )
        settings_data = {"fbx_sdk": {}}

        def _get_tex_cache_for_export(self):
            return "texture-cache"

        def _run_io_async(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        def _log(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(
        model_io.QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(output), "Standard FBX (*.fbx)"),
    )

    _Harness()._export_selected_fbx()

    args = captured["args"]
    kwargs = captured["kwargs"]
    assert args[1] is model_io._work_export_selected_fbx
    assert args[2] == [room, sky]
    assert args[3] == str(output)
    assert kwargs["tex_cache"] == "texture-cache"


def test_export_selected_single_static_room_routes_to_one_mesh_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.gui.windows.application_core.shared import model_io

    runtime_model = SimpleNamespace(name="m14aa_01g")
    room = SimpleNamespace(
        id="room-id",
        name="m14aa_01g",
        metadata={"_runtime_model": runtime_model},
    )
    captured: dict[str, object] = {}
    output = tmp_path / "m14aa_01g.fbx"

    class _Harness(model_io.ModelIoMixin):
        scene_manager = SimpleNamespace(
            active_scene=SimpleNamespace(objects=[room]),
            get_selected_objects=lambda: [room],
        )
        viewport = SimpleNamespace(
            _selected_viewport_nodes=[
                SimpleNamespace(_gr_scene_object_id="room-id"),
            ]
        )
        settings_data = {"fbx_sdk": {}}

        def _get_tex_cache_for_export(self):
            return "texture-cache"

        def _runtime_model_for_scene_object(self, _item):
            return runtime_model

        def _fbx_base_skeleton_for_export(self, _model):
            return None

        def _fbx_supplemental_animation_models(self, _model):
            return ()

        def _choose_fbx_animation_sets(self, *_args, **_kwargs):
            return ()

        def _fbx_resource_context_for_export(self, _model):
            return None, None

        def _run_io_async(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        def _log(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(
        model_io.QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(output), "Standard FBX (*.fbx)"),
    )

    _Harness()._export_selected_fbx()

    args = captured["args"]
    kwargs = captured["kwargs"]
    assert args[1] is model_io._work_export_selected_fbx
    assert args[2] == [room]
    assert args[3] == str(output)
    assert kwargs["tex_cache"] == "texture-cache"


def test_full_module_export_resolves_every_object_in_selected_module_group() -> None:
    from src.gui.windows.application_core.shared import model_io

    module_objects = [
        SimpleNamespace(
            id=f"room-{suffix}",
            name=f"m14aa_01{suffix}",
            metadata={"module_group": {"module_root": "m14aa_01"}},
        )
        for suffix in "abcdefghi"
    ]
    unrelated = SimpleNamespace(
        id="other-room",
        name="m13aa_01a",
        metadata={"module_group": {"module_root": "m13aa_01"}},
    )
    scene_manager = SimpleNamespace(
        active_scene=SimpleNamespace(objects=[*module_objects, unrelated]),
        get_selected_objects=lambda: [module_objects[6]],
    )
    viewport = SimpleNamespace(
        _selected_viewport_nodes=[
            SimpleNamespace(_gr_scene_object_id=module_objects[6].id),
        ]
    )

    resolved = model_io._module_group_scene_objects_for_export(
        scene_manager,
        viewport,
    )

    assert resolved == module_objects


def test_export_full_module_routes_all_group_objects_to_one_mesh_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.gui.windows.application_core.shared import model_io

    module_objects = [
        SimpleNamespace(
            id=f"room-{suffix}",
            name=f"m14aa_01{suffix}",
            metadata={
                "module_group": {"module_root": "m14aa_01"},
                "_runtime_model": SimpleNamespace(name=f"m14aa_01{suffix}"),
            },
        )
        for suffix in "abcdefghi"
    ]
    output = tmp_path / "m14aa_01_full.fbx"
    captured: dict[str, object] = {}

    class _Harness(model_io.ModelIoMixin):
        scene_manager = SimpleNamespace(
            active_scene=SimpleNamespace(objects=module_objects),
            get_selected_objects=lambda: [module_objects[6]],
        )
        viewport = SimpleNamespace(
            _selected_viewport_nodes=[
                SimpleNamespace(_gr_scene_object_id=module_objects[6].id),
            ]
        )

        def _get_tex_cache_for_export(self):
            return "texture-cache"

        def _run_io_async(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        def _log(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(
        model_io.QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(output), "Standard FBX (*.fbx)"),
    )

    _Harness()._export_full_module_fbx()

    args = captured["args"]
    kwargs = captured["kwargs"]
    assert args[1] is model_io._work_export_selected_fbx
    assert args[2] == module_objects
    assert args[3] == str(output)
    assert kwargs["tex_cache"] == "texture-cache"


def test_export_full_module_obj_routes_all_group_objects_to_one_mesh_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.gui.windows.application_core.shared import model_io
    from src.io.full_module_obj_export import export_full_module_obj

    module_objects = [
        SimpleNamespace(
            id=f"room-{suffix}",
            name=f"m12aa_01{suffix}",
            metadata={
                "module_group": {"module_root": "m12aa_01"},
                "_runtime_model": SimpleNamespace(name=f"m12aa_01{suffix}"),
            },
        )
        for suffix in "abcdefghijklmnopq"
    ]
    output = tmp_path / "m12aa_01_full.obj"
    captured: dict[str, object] = {}

    class _Harness(model_io.ModelIoMixin):
        scene_manager = SimpleNamespace(
            active_scene=SimpleNamespace(objects=module_objects),
            get_selected_objects=lambda: [module_objects[0]],
        )
        viewport = SimpleNamespace(
            _selected_viewport_nodes=[
                SimpleNamespace(_gr_scene_object_id=module_objects[0].id),
            ]
        )

        def _get_tex_cache_for_export(self):
            return "texture-cache"

        def _run_io_async(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        def _log(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(
        model_io.QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(output), "Wavefront OBJ (*.obj)"),
    )

    _Harness()._export_full_module_obj()

    args = captured["args"]
    kwargs = captured["kwargs"]
    assert args[1] is export_full_module_obj
    assert args[2] == module_objects
    assert args[3] == str(output)
    assert kwargs["tex_cache"] == "texture-cache"


def test_full_module_obj_worker_merges_group_and_publishes_sidecars(
    tmp_path: Path,
) -> None:
    from src.io.full_module_obj_export import export_full_module_obj

    scene_objects = [
        SimpleNamespace(
            name=f"m12aa_01{suffix}",
            metadata={
                "_runtime_model": _selected_export_triangle_model(
                    f"room_{suffix}",
                    f"texture_{suffix}",
                ),
            },
            transform=SimpleNamespace(
                position=(float(index) * 2.0, 0.0, 0.0),
                rotation=(0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
            ),
        )
        for index, suffix in enumerate("abcdefghijklmnopq")
    ]

    class _Image:
        mode = "RGB"

        def transpose(self, _operation):
            return self

        def save(self, path):
            Path(path).write_bytes(b"tga")

    output = tmp_path / "m12aa_01_full.obj"
    output_path, sidecars = export_full_module_obj(
        scene_objects,
        str(output),
        tex_cache=SimpleNamespace(get=lambda _name: _Image()),
    )

    obj_text = output.read_text(encoding="utf-8")
    mtl_text = output.with_suffix(".mtl").read_text(encoding="utf-8")
    assert output_path == str(output)
    assert obj_text.count("\no ") == 1
    assert "usemtl texture_a" in obj_text
    assert "usemtl texture_q" in obj_text
    assert "newmtl texture_a" in mtl_text
    assert "newmtl texture_q" in mtl_text
    assert sidecars.requested == 17
    assert sidecars.saved == 17
    assert set(sidecars.saved_files) == {
        f"texture_{suffix}.tga" for suffix in "abcdefghijklmnopq"
    }


def test_export_maya_room_module_obj_routes_all_group_objects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.gui.windows.application_core.shared import model_io
    from src.io.full_module_obj_export import export_maya_room_module_obj

    module_objects = [
        SimpleNamespace(
            id=f"room-{suffix}",
            name=f"m12aa_01{suffix}",
            metadata={
                "module_group": {"module_root": "m12aa_01"},
                "_runtime_model": SimpleNamespace(name=f"m12aa_01{suffix}"),
            },
        )
        for suffix in "abcdefghijklmnopq"
    ]
    output = tmp_path / "m12aa_01_maya_rooms.obj"
    captured: dict[str, object] = {}

    class _Harness(model_io.ModelIoMixin):
        scene_manager = SimpleNamespace(
            active_scene=SimpleNamespace(objects=module_objects),
            get_selected_objects=lambda: [module_objects[0]],
        )
        viewport = SimpleNamespace(
            _selected_viewport_nodes=[
                SimpleNamespace(_gr_scene_object_id=module_objects[0].id),
            ]
        )

        def _get_tex_cache_for_export(self):
            return "texture-cache"

        def _run_io_async(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        def _log(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(
        model_io.QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(output), "Wavefront OBJ (*.obj)"),
    )

    _Harness()._export_maya_room_module_obj()

    args = captured["args"]
    kwargs = captured["kwargs"]
    assert args[1] is export_maya_room_module_obj
    assert args[2] == module_objects
    assert args[3] == str(output)
    assert kwargs["tex_cache"] == "texture-cache"


def test_maya_room_module_obj_worker_preserves_seventeen_room_objects(
    tmp_path: Path,
) -> None:
    from src.io.full_module_obj_export import export_maya_room_module_obj

    scene_objects = [
        SimpleNamespace(
            name=f"m12aa_01{suffix}",
            metadata={
                "_runtime_model": _selected_export_triangle_model(
                    f"mesh_{suffix}",
                    f"texture_{suffix}",
                ),
            },
            transform=SimpleNamespace(
                position=(float(index) * 2.0, 0.0, 0.0),
                rotation=(0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
            ),
        )
        for index, suffix in enumerate("abcdefghijklmnopq")
    ]

    class _Image:
        mode = "RGB"

        def transpose(self, _operation):
            return self

        def save(self, path):
            Path(path).write_bytes(b"tga")

    output = tmp_path / "m12aa_01_maya_rooms.obj"
    output_path, sidecars = export_maya_room_module_obj(
        scene_objects,
        str(output),
        tex_cache=SimpleNamespace(get=lambda _name: _Image()),
    )

    obj_text = output.read_text(encoding="utf-8")
    assert output_path == str(output)
    assert obj_text.count("\no ") == 17
    assert obj_text.count("\ng ") == 17
    for suffix in "abcdefghijklmnopq":
        assert f"\no m12aa_01{suffix}\n" in obj_text
        assert f"\ng m12aa_01{suffix}\n" in obj_text
    assert sidecars.requested == 17
    assert sidecars.saved == 17


def test_full_module_obj_rejects_group_with_unloaded_room(tmp_path: Path) -> None:
    from src.io.full_module_obj_export import export_full_module_obj

    loaded_room = SimpleNamespace(
        name="m12aa_01a",
        metadata={
            "_runtime_model": _selected_export_triangle_model("loaded", "stone"),
        },
        transform=SimpleNamespace(
            position=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
        ),
    )
    unloaded_room = SimpleNamespace(
        name="m12aa_01b",
        metadata={"_runtime_model": None},
    )
    output = tmp_path / "m12aa_01_full.obj"

    with pytest.raises(ValueError, match=r"m12aa_01b"):
        export_full_module_obj([loaded_room, unloaded_room], str(output))

    assert not output.exists()
    assert not output.with_suffix(".mtl").exists()
    assert not list(tmp_path.glob(".m12aa_01_full-export-*"))


def test_full_module_obj_cancellation_does_not_publish_staged_files(
    tmp_path: Path,
) -> None:
    import threading

    from src.io.export_control import ExportCancelledError
    from src.io.full_module_obj_export import export_full_module_obj

    cancelled = threading.Event()
    scene_objects = [
        SimpleNamespace(
            name="m12aa_01a",
            metadata={
                "_runtime_model": _selected_export_triangle_model(
                    "courtyard",
                    "lda_grass07",
                ),
            },
            transform=SimpleNamespace(
                position=(0.0, 0.0, 0.0),
                rotation=(0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
            ),
        )
    ]

    class _Image:
        mode = "RGB"

        def transpose(self, _operation):
            return self

        def save(self, path):
            Path(path).write_bytes(b"partial staged texture")

    def _get_texture(_name):
        cancelled.set()
        return _Image()

    output = tmp_path / "cancelled.obj"
    with pytest.raises(ExportCancelledError):
        export_full_module_obj(
            scene_objects,
            str(output),
            tex_cache=SimpleNamespace(get=_get_texture),
            is_cancelled=cancelled.is_set,
        )

    assert not output.exists()
    assert not output.with_suffix(".mtl").exists()
    assert not (tmp_path / "lda_grass07.tga").exists()
    assert not list(tmp_path.glob(".cancelled-export-*"))


def _merge_filter_triangle_node(
    name: str,
    *,
    texture: str = "shared_texture",
    render: bool = True,
    shininess: float = 0.0,
    faces: list[tuple[int, int, int]] | None = None,
):
    from src.core.geometry.model_data import ModelNode, NodeFlags

    return ModelNode(
        name=name,
        flags=int(NodeFlags.HEADER | NodeFlags.MESH),
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        normals=[(0.0, 0.0, 1.0)] * 3,
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        faces=[(0, 1, 2)] if faces is None else faces,
        face_mats=[0],
        texture=texture,
        texture_names=[texture],
        tex_count=1,
        render=render,
        shininess=shininess,
    )


def _merge_filter_model(*nodes):
    from src.core.geometry.model_data import KotorModel, ModelNode, NodeFlags

    root = ModelNode(
        name="root",
        flags=int(NodeFlags.HEADER),
        children=list(nodes),
    )
    for node in nodes:
        node.parent = root
    return KotorModel(name="fixture", classification="tile", root_node=root)


def test_node_filter_can_include_render_false_obj_geometry_and_skips_no_face_nodes() -> None:
    from src.converters.mesh_converter import OBJExporter
    from src.io.fbx.fbx_exporter import FbxExportError, merge_selected_scene_objects

    facial_mesh = _merge_filter_triangle_node("eyeRA", texture="eye_texture", render=False)
    no_face_mesh = _merge_filter_triangle_node(
        "eye_helper",
        texture="eye_texture",
        render=False,
        faces=[],
    )
    model = _merge_filter_model(facial_mesh, no_face_mesh)

    with pytest.raises(FbxExportError):
        merge_selected_scene_objects([model])

    merged = merge_selected_scene_objects(
        [model],
        node_filter=OBJExporter._is_renderable,
    )

    assert len(merged.root_node.faces) == 1
    assert [slot.texture for slot in merged.root_node._gr_fbx_material_slots] == [
        "eye_texture"
    ]


def test_material_key_keeps_materials_with_distinct_shininess_separate() -> None:
    from src.io.fbx.fbx_exporter import merge_selected_scene_objects

    dull = _merge_filter_triangle_node("dull", shininess=2.0)
    glossy = _merge_filter_triangle_node("glossy", shininess=128.0)

    merged = merge_selected_scene_objects([_merge_filter_model(dull, glossy)])

    assert merged.root_node.face_mats == [0, 1]
    assert [
        slot.shininess for slot in merged.root_node._gr_fbx_material_slots
    ] == [2.0, 128.0]


def test_clean_module_obj_filter_removes_skybox_and_oversized_background() -> None:
    from src.core.geometry.model_data import KotorModel, ModelNode, NodeFlags
    from src.io.clean_module_obj_export import prepare_clean_module_obj_model

    def _model(name: str, texture: str, vertices) -> KotorModel:
        mesh = ModelNode(
            name=f"{name}_mesh",
            flags=int(NodeFlags.HEADER | NodeFlags.MESH),
            vertices=list(vertices),
            normals=[(0.0, 0.0, 1.0)] * 3,
            uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
            uvs_lm=[(0.1, 0.1), (0.9, 0.1), (0.1, 0.9)],
            faces=[(0, 1, 2)],
            face_mats=[0],
            texture=texture,
            texture_names=[texture],
            tex_count=1,
        )
        return KotorModel(name=name, classification="tile", root_node=mesh)

    models = (
        _model(
            "courtyard",
            "lda_grass07",
            ((0.0, 0.0, 1.0), (10.0, 0.0, 1.0), (0.0, 10.0, 1.0)),
        ),
        _model(
            "sky",
            "lda_sky0001",
            ((0.0, 0.0, 20.0), (10.0, 0.0, 20.0), (0.0, 10.0, 20.0)),
        ),
        _model(
            "background",
            "lda_grass07",
            ((-1000.0, -1000.0, -5.0), (1000.0, -1000.0, -5.0), (-1000.0, 1000.0, -5.0)),
        ),
    )
    scene_objects = [
        SimpleNamespace(
            name=model.name,
            metadata={"_runtime_model": model},
            transform=SimpleNamespace(
                position=(0.0, 0.0, 0.0),
                rotation=(0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
            ),
        )
        for model in models
    ]

    cleaned, summary = prepare_clean_module_obj_model(
        scene_objects,
        name="m14aa_clean",
    )

    mesh = cleaned.root_node
    assert mesh is not None
    assert mesh.faces == [(0, 1, 2)]
    assert mesh.vertices == list(models[0].root_node.vertices)
    assert mesh.uvs == list(models[0].root_node.uvs)
    assert mesh.uvs_lm == list(models[0].root_node.uvs_lm)
    assert [slot.texture for slot in mesh._gr_fbx_material_slots] == ["lda_grass07"]
    assert mesh.face_mats == [0]
    assert summary.source_faces == 3
    assert summary.exported_faces == 1
    assert summary.removed_skybox_faces == 1
    assert summary.removed_background_faces == 1


def test_clean_module_obj_rejects_group_with_unloaded_room() -> None:
    from src.io.clean_module_obj_export import prepare_clean_module_obj_model

    loaded_room = SimpleNamespace(
        name="m14aa_01a",
        metadata={
            "_runtime_model": _selected_export_triangle_model("m14aa_01a", "stone")
        },
        transform=SimpleNamespace(
            position=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
        ),
    )
    unloaded_room = SimpleNamespace(
        name="m14aa_01b",
        metadata={"_runtime_model": None},
    )

    with pytest.raises(ValueError, match=r"m14aa_01b"):
        prepare_clean_module_obj_model(
            [loaded_room, unloaded_room],
            name="m14aa_clean",
        )


def test_obj_exporter_preserves_merged_face_material_slots(tmp_path: Path) -> None:
    from src.converters.mesh_converter import OBJExporter
    from src.io.fbx.fbx_exporter import merge_selected_scene_objects

    scene_objects = [
        SimpleNamespace(
            name=texture,
            metadata={
                "_runtime_model": _selected_export_triangle_model(
                    texture,
                    texture,
                )
            },
            transform=SimpleNamespace(
                position=(float(index) * 2.0, 0.0, 0.0),
                rotation=(0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
            ),
        )
        for index, texture in enumerate(("lda_grass07", "lda_stone02"))
    ]
    merged = merge_selected_scene_objects(scene_objects, name="clean_module")
    output = tmp_path / "clean_module.obj"

    OBJExporter().export(merged, str(output), export_rigging=False)

    obj_text = output.read_text(encoding="utf-8")
    mtl_text = output.with_suffix(".mtl").read_text(encoding="utf-8")
    assert "usemtl lda_grass07" in obj_text
    assert "usemtl lda_stone02" in obj_text
    assert obj_text.count("usemtl ") == 2
    assert "newmtl lda_grass07" in mtl_text
    assert "newmtl lda_stone02" in mtl_text
    assert "map_Kd lda_grass07.tga" in mtl_text
    assert "map_Kd lda_stone02.tga" in mtl_text


def test_obj_exporter_keeps_distinct_materials_that_share_one_texture(
    tmp_path: Path,
) -> None:
    from src.converters.mesh_converter import OBJExporter
    from src.io.fbx.fbx_exporter import merge_selected_scene_objects

    opaque = _selected_export_triangle_model("opaque", "shared_stone")
    opaque.root_node.diffuse = (0.8, 0.7, 0.6)
    opaque.root_node.alpha = 1.0
    translucent = _selected_export_triangle_model("translucent", "shared_stone")
    translucent.root_node.diffuse = (0.2, 0.3, 0.4)
    translucent.root_node.alpha = 0.5
    scene_objects = [
        SimpleNamespace(
            metadata={"_runtime_model": model},
            transform=SimpleNamespace(
                position=(float(index) * 2.0, 0.0, 0.0),
                rotation=(0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
            ),
        )
        for index, model in enumerate((opaque, translucent))
    ]
    output = tmp_path / "shared_texture.obj"

    OBJExporter().export(
        merge_selected_scene_objects(scene_objects, name="shared_texture"),
        str(output),
        export_rigging=False,
    )

    obj_text = output.read_text(encoding="utf-8")
    mtl_text = output.with_suffix(".mtl").read_text(encoding="utf-8")
    assert "usemtl shared_stone\n" in obj_text
    assert "usemtl shared_stone_2\n" in obj_text
    assert "newmtl shared_stone\n" in mtl_text
    assert "newmtl shared_stone_2\n" in mtl_text
    assert mtl_text.count("map_Kd shared_stone.tga") == 2
    assert "d  1.0000" in mtl_text
    assert "d  0.5000" in mtl_text


def test_obj_exporter_reports_cache_misses_and_saved_multi_slot_textures(
    tmp_path: Path,
) -> None:
    from src.converters.mesh_converter import OBJExporter
    from src.io.fbx.fbx_exporter import merge_selected_scene_objects

    scene_objects = [
        SimpleNamespace(
            metadata={
                "_runtime_model": _selected_export_triangle_model(texture, texture)
            },
            transform=SimpleNamespace(
                position=(float(index) * 2.0, 0.0, 0.0),
                rotation=(0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
            ),
        )
        for index, texture in enumerate(("lda_grass07", "lda_stone02"))
    ]
    merged = merge_selected_scene_objects(scene_objects, name="sidecars")

    missing_result = OBJExporter().export(
        merged,
        str(tmp_path / "missing.obj"),
        tex_cache=SimpleNamespace(get=lambda _name: None),
        export_rigging=False,
    )
    assert missing_result.texture_sidecars.requested == 2
    assert missing_result.texture_sidecars.saved == 0
    assert missing_result.texture_sidecars.missing_names == (
        "lda_grass07",
        "lda_stone02",
    )

    class _Image:
        mode = "RGB"

        def transpose(self, _operation):
            return self

        def save(self, path):
            Path(path).write_bytes(b"tga")

    saved_result = OBJExporter().export(
        merged,
        str(tmp_path / "saved.obj"),
        tex_cache=SimpleNamespace(get=lambda _name: _Image()),
        export_rigging=False,
    )
    assert saved_result.texture_sidecars.requested == 2
    assert saved_result.texture_sidecars.saved == 2
    assert saved_result.texture_sidecars.missing_names == ()
    assert set(saved_result.texture_sidecars.saved_files) == {
        "lda_grass07.tga",
        "lda_stone02.tga",
    }


def test_clean_module_obj_cancellation_does_not_publish_staged_files(
    tmp_path: Path,
) -> None:
    import threading

    from src.gui.windows.application_core.shared.model_io import (
        _work_export_clean_module_obj,
    )
    from src.io.export_control import ExportCancelledError

    cancelled = threading.Event()
    model = _selected_export_triangle_model("courtyard", "lda_grass07")
    scene_objects = [
        SimpleNamespace(
            name="courtyard",
            metadata={"_runtime_model": model},
            transform=SimpleNamespace(
                position=(0.0, 0.0, 0.0),
                rotation=(0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
            ),
        )
    ]

    class _Image:
        mode = "RGB"

        def transpose(self, _operation):
            return self

        def save(self, path):
            Path(path).write_bytes(b"partial staged texture")

    def _get_texture(_name):
        cancelled.set()
        return _Image()

    output = tmp_path / "cancelled.obj"
    with pytest.raises(ExportCancelledError):
        _work_export_clean_module_obj(
            scene_objects,
            str(output),
            tex_cache=SimpleNamespace(get=_get_texture),
            is_cancelled=cancelled.is_set,
        )

    assert not output.exists()
    assert not output.with_suffix(".mtl").exists()
    assert not (tmp_path / "lda_grass07.tga").exists()
    assert not list(tmp_path.glob(".cancelled-export-*"))


def test_background_io_worker_cancel_is_immediately_thread_visible() -> None:
    from src.gui.windows.application_core.application_core_lib.shared.workers import (
        BackgroundIOWorker,
    )

    worker = BackgroundIOWorker(lambda: None)
    assert worker.is_cancelled() is False

    worker.request_cancel()

    assert worker.is_cancelled() is True


def test_clean_module_obj_worker_publishes_sidecars_and_summary(
    tmp_path: Path,
) -> None:
    from src.gui.windows.application_core.shared.model_io import (
        _work_export_clean_module_obj,
    )

    model = _selected_export_triangle_model("courtyard", "lda_grass07")
    scene_objects = [
        SimpleNamespace(
            name="courtyard",
            metadata={"_runtime_model": model},
            transform=SimpleNamespace(
                position=(0.0, 0.0, 0.0),
                rotation=(0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
            ),
        )
    ]

    class _Image:
        mode = "RGB"

        def transpose(self, _operation):
            return self

        def save(self, path):
            Path(path).write_bytes(b"tga")

    output = tmp_path / "published.obj"
    output_path, summary = _work_export_clean_module_obj(
        scene_objects,
        str(output),
        tex_cache=SimpleNamespace(get=lambda _name: _Image()),
    )

    assert output_path == str(output)
    assert output.exists()
    assert output.with_suffix(".mtl").exists()
    assert (tmp_path / "lda_grass07.tga").exists()
    assert summary.texture_sidecars.requested == 1
    assert summary.texture_sidecars.saved_files == ("lda_grass07.tga",)
    assert summary.texture_sidecars.unavailable == 0


def test_export_clean_module_obj_routes_group_to_clean_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.gui.windows.application_core.shared import model_io

    module_objects = [
        SimpleNamespace(
            id=f"room-{suffix}",
            name=f"m14aa_01{suffix}",
            metadata={
                "module_group": {"module_root": "m14aa_01"},
                "_runtime_model": SimpleNamespace(name=f"m14aa_01{suffix}"),
            },
        )
        for suffix in "abcdefghi"
    ]
    output = tmp_path / "m14aa_01_clean.obj"
    captured: dict[str, object] = {}
    dialogs: list[tuple[str, str]] = []

    class _Harness(model_io.ModelIoMixin):
        scene_manager = SimpleNamespace(
            active_scene=SimpleNamespace(objects=module_objects),
            get_selected_objects=lambda: [module_objects[2]],
        )
        viewport = SimpleNamespace(
            _selected_viewport_nodes=[
                SimpleNamespace(_gr_scene_object_id=module_objects[2].id),
            ]
        )

        def _get_tex_cache_for_export(self):
            return "texture-cache"

        def _run_io_async(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        def _log(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(
        model_io.QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(output), "Wavefront OBJ (*.obj)"),
    )
    monkeypatch.setattr(
        model_io.QtWidgets.QMessageBox,
        "warning",
        lambda _parent, title, message: dialogs.append((title, message)),
    )

    _Harness()._export_clean_module_obj()

    args = captured["args"]
    kwargs = captured["kwargs"]
    assert args[1] is model_io._work_export_clean_module_obj
    assert args[2] == module_objects
    assert args[3] == str(output)
    assert kwargs["tex_cache"] == "texture-cache"

    from src.io.clean_module_obj_export import CleanModuleObjSummary
    from src.io.export_control import TextureSidecarResult

    kwargs["on_complete"](
        (
            str(output),
            CleanModuleObjSummary(
                scene_objects=9,
                source_faces=100,
                exported_faces=90,
                removed_skybox_faces=8,
                removed_background_faces=2,
                materials=2,
                texture_sidecars=TextureSidecarResult(
                    requested_names=("missing_texture",),
                    missing_names=("missing_texture",),
                ),
            ),
        ),
        cancelled=False,
    )
    assert dialogs
    assert "Texture files written: 0 of 1" in dialogs[0][1]
    assert "missing_texture" in dialogs[0][1]
