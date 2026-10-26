"""Export every loaded module room as one staged Wavefront OBJ."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from src.converters.mesh_converter import OBJExporter
from src.core.export.export_job import promote_staged_files
from src.core.geometry.model_data import KotorModel, ModelNode, NodeFlags
from src.io.clean_module_obj_export import validate_module_scene_objects
from src.io.export_control import check_export_cancelled
from src.io.fbx.fbx_exporter import merge_selected_scene_objects


def _add_maya_object_groups(obj_path: Path, *, is_cancelled=None) -> None:
    """Mirror OBJ object records as groups because Maya splits on ``g``."""

    grouped_path = obj_path.with_name(f".{obj_path.name}.maya-groups")
    try:
        with obj_path.open("r", encoding="utf-8") as source, grouped_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as destination:
            for line_number, line in enumerate(source, start=1):
                destination.write(line)
                if line.startswith("o "):
                    destination.write(f"g {line[2:].strip()}\n")
                if line_number % 4096 == 0:
                    check_export_cancelled(is_cancelled)
        check_export_cancelled(is_cancelled)
        os.replace(grouped_path, obj_path)
    finally:
        grouped_path.unlink(missing_ok=True)


def _prepare_one_mesh_model(
    scene_objects: list[Any],
    *,
    name: str,
    is_cancelled=None,
) -> KotorModel:
    return merge_selected_scene_objects(
        scene_objects,
        name=name,
        is_cancelled=is_cancelled,
        node_filter=OBJExporter._is_renderable,
    )


def _prepare_maya_room_model(
    scene_objects: list[Any],
    *,
    name: str,
    is_cancelled=None,
) -> KotorModel:
    """Build one positioned mesh node per room for Maya object selection."""

    root = ModelNode(name=name, flags=int(NodeFlags.HEADER))
    first_room_model: KotorModel | None = None
    used_names: set[str] = set()
    for index, item in enumerate(scene_objects):
        check_export_cancelled(is_cancelled)
        runtime_model = (
            item
            if isinstance(item, KotorModel)
            else (getattr(item, "metadata", {}) or {}).get("_runtime_model")
        )
        if first_room_model is None:
            first_room_model = runtime_model

        base_name = str(
            getattr(item, "name", "")
            or getattr(runtime_model, "name", "")
            or f"room_{index + 1}"
        ).strip()
        room_name = base_name
        suffix = 2
        while room_name.casefold() in used_names:
            room_name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(room_name.casefold())

        room_model = merge_selected_scene_objects(
            [item],
            name=room_name,
            is_cancelled=is_cancelled,
            node_filter=OBJExporter._is_renderable,
        )
        room_mesh = room_model.root_node
        if room_mesh is None:
            continue
        room_mesh.name = room_name
        room_mesh.parent = root
        root.children.append(room_mesh)

    check_export_cancelled(is_cancelled)
    if first_room_model is None or not root.children:
        raise ValueError("The loaded module contains no exportable room geometry.")
    result = KotorModel(
        name=name,
        supermodel="NULL",
        classification=first_room_model.classification,
        game_version=first_room_model.game_version,
        model_type=first_room_model.model_type,
        root_node=root,
    )
    result.compute_bounds()
    return result


def _export_staged_module_obj(
    scene_objects: Any,
    path: str | Path,
    *,
    model_builder: Callable[..., KotorModel],
    combine_message: str,
    add_maya_groups: bool = False,
    tex_cache=None,
    progress_callback=None,
    is_cancelled=None,
):
    validated_objects = validate_module_scene_objects(
        scene_objects,
        is_cancelled=is_cancelled,
    )
    check_export_cancelled(is_cancelled)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{target.stem}-export-",
            dir=str(target.parent),
        )
    )
    try:
        if progress_callback:
            progress_callback(combine_message, 20)
        export_model = model_builder(
            validated_objects,
            name=target.stem or "selection",
            is_cancelled=is_cancelled,
        )
        check_export_cancelled(is_cancelled)
        if progress_callback:
            progress_callback("Writing OBJ geometry and materials…", 50)
        export_result = OBJExporter().export(
            export_model,
            str(stage_dir / target.name),
            tex_cache=tex_cache,
            export_rigging=False,
            is_cancelled=is_cancelled,
        )
        if add_maya_groups:
            _add_maya_object_groups(
                Path(export_result.obj_path),
                is_cancelled=is_cancelled,
            )
        check_export_cancelled(is_cancelled)
        if progress_callback:
            progress_callback("Publishing full-module OBJ files…", 95)
        promote_staged_files(stage_dir, target.parent)
        if progress_callback:
            progress_callback("Full-module OBJ export complete", 100)
        return str(target), export_result.texture_sidecars
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


def export_full_module_obj(
    scene_objects: Any,
    path: str | Path,
    *,
    tex_cache=None,
    progress_callback=None,
    is_cancelled=None,
):
    """Write all loaded module rooms as one OBJ and its texture sidecars."""

    return _export_staged_module_obj(
        scene_objects,
        path,
        model_builder=_prepare_one_mesh_model,
        combine_message="Combining selected scene objects into one mesh…",
        tex_cache=tex_cache,
        progress_callback=progress_callback,
        is_cancelled=is_cancelled,
    )


def export_maya_room_module_obj(
    scene_objects: Any,
    path: str | Path,
    *,
    tex_cache=None,
    progress_callback=None,
    is_cancelled=None,
):
    """Write a Maya-friendly OBJ with one positioned object per module room."""

    return _export_staged_module_obj(
        scene_objects,
        path,
        model_builder=_prepare_maya_room_model,
        combine_message="Preparing one positioned Maya object per room…",
        add_maya_groups=True,
        tex_cache=tex_cache,
        progress_callback=progress_callback,
        is_cancelled=is_cancelled,
    )


__all__ = ["export_full_module_obj", "export_maya_room_module_obj"]
