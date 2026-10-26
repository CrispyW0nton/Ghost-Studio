"""Model import, export, FBX SDK, MDLOps, and module-file command helpers."""

from __future__ import annotations

import copy
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError as exc:  # pragma: no cover - import gate for Qt runtime
    raise RuntimeError("PySide6 is required for the Qt shell") from exc

from src.gui.qt_lib.dialogs.qt_settings_dialog import save_settings
from src.gui.dialogs.error_report import report_from_exception, show_error_report, show_exception
from src.gui.windows.application_core.application_core_lib.shared.workers import BackgroundIOWorker

log = logging.getLogger(__name__)

FBX_EXPORT_FILTERS = (
    "Standard FBX (*.fbx);;"
    "Unity-Compatible FBX (*.fbx);;"
    "Unreal Engine-Compatible FBX (*.fbx);;"
    "3ds Max-Compatible FBX (*.fbx);;"
    "All files (*.*)"
)


def _fbx_compatibility_profile_from_filter(selected_filter: str) -> str:
    label = str(selected_filter or "").strip().lower()
    if "unity" in label:
        return "unity"
    if "unreal" in label:
        return "unreal"
    if "3ds max" in label or "3dsmax" in label:
        return "3ds_max"
    return "standard"


def _selected_scene_objects_for_export(scene_manager, viewport) -> list:
    """Resolve click or marquee viewport selection to stable scene objects."""
    scene = getattr(scene_manager, "active_scene", None)
    objects = list(getattr(scene, "objects", None) or [])
    objects_by_id = {str(getattr(obj, "id", "") or ""): obj for obj in objects}
    selected = []
    seen = set()
    for node in list(getattr(viewport, "_selected_viewport_nodes", None) or []):
        object_id = str(getattr(node, "_gr_scene_object_id", "") or "")
        if not object_id or object_id in seen or object_id not in objects_by_id:
            continue
        seen.add(object_id)
        selected.append(objects_by_id[object_id])
    if selected:
        return selected
    getter = getattr(scene_manager, "get_selected_objects", None)
    return list(getter() or []) if callable(getter) else []


def _module_group_scene_objects_for_export(scene_manager, viewport) -> list:
    """Resolve the complete loaded module group containing the selected room."""
    selected = _selected_scene_objects_for_export(scene_manager, viewport)
    anchor = next(
        (
            item
            for item in selected
            if isinstance(
                (getattr(item, "metadata", {}) or {}).get("module_group"),
                dict,
            )
        ),
        None,
    )
    if anchor is None:
        return []
    anchor_group = (getattr(anchor, "metadata", {}) or {}).get("module_group") or {}
    module_root = str(anchor_group.get("module_root") or "").strip().lower()
    if not module_root:
        return []
    scene = getattr(scene_manager, "active_scene", None)
    return [
        item
        for item in list(getattr(scene, "objects", None) or [])
        if str(
            (
                (getattr(item, "metadata", {}) or {}).get("module_group")
                or {}
            ).get("module_root")
            or ""
        ).strip().lower()
        == module_root
    ]


# ---------------------------------------------------------------------------
# Plain "work" functions that perform the actual blocking I/O.
#
# These intentionally take only primitive / picklable-ish arguments (paths,
# model objects, resolved config dicts) and never touch ``self`` or any Qt
# object, so they are safe to run on a background ``QThread`` via
# :class:`BackgroundIOWorker`. Each may declare the optional ``progress_callback``
# and ``is_cancelled`` keyword hooks; the worker injects them automatically.
# ---------------------------------------------------------------------------


def _work_import_obj(path: str, *, game_version, progress_callback=None, is_cancelled=None):
    from src.converters.mesh_converter import OBJImporter

    if progress_callback:
        progress_callback("Reading OBJ file\u2026", 20)
    model = OBJImporter().import_file(path, game_version=game_version)
    if progress_callback:
        progress_callback("Finalizing OBJ mesh\u2026", 90)
    return model


def _work_import_gltf(path: str, *, game_version, progress_callback=None, is_cancelled=None):
    from src.converters.mesh_converter import GLTFImporter

    if progress_callback:
        progress_callback("Reading GLB/GLTF file\u2026", 20)
    model = GLTFImporter().import_file(path, game_version=game_version)
    if model is None:
        raise RuntimeError("GLTF import failed. Install pygltflib or trimesh.")
    if progress_callback:
        progress_callback("Finalizing GLB/GLTF mesh\u2026", 90)
    return model


def _work_import_fbx_sdk(path: str, *, game_version, fbx_sdk_settings, progress_callback=None, is_cancelled=None):
    from src.io.fbx.fbx_importer import import_fbx

    if progress_callback:
        progress_callback("Importing FBX via Autodesk SDK\u2026", 30)
    return import_fbx(path, {"game_version": game_version, "fbx_sdk": fbx_sdk_settings})


def _work_import_fbx_blender(path: str, *, game_version, progress_callback=None, is_cancelled=None):
    from src.converters.mesh_converter import FBXImporter

    if progress_callback:
        progress_callback("Importing FBX via Blender bridge\u2026", 30)
    return FBXImporter().import_file(path, game_version=game_version)


def _work_export_obj(model, path: str, *, tex_cache, progress_callback=None, is_cancelled=None):
    from src.converters.mesh_converter import OBJExporter, _export_rigging_data
    from pathlib import Path as _Path

    if progress_callback:
        progress_callback("Writing OBJ geometry…", 15)
    # Phase 1: geometry only (fast)
    OBJExporter().export(model, path, tex_cache=tex_cache, export_rigging=False)

    if progress_callback:
        progress_callback("Exporting skeleton and skin weights…", 50)
    # Phase 2: rigging data (potentially slow — animations)
    out_dir = _Path(path).parent
    try:
        rig_count = _export_rigging_data(model, out_dir)
    except Exception:
        rig_count = 0

    if progress_callback:
        progress_callback("OBJ export complete", 100)
    return path


def _work_export_gltf(model, path: str, *, tex_cache, progress_callback=None, is_cancelled=None):
    from src.converters.mesh_converter import GLTFExporter

    binary = path.lower().endswith(".glb")
    if progress_callback:
        progress_callback("Writing GLB/GLTF file\u2026", 40)
    ok = GLTFExporter().export(model, path, binary=binary, tex_cache=tex_cache, export_rigging=True)
    if not ok:
        raise RuntimeError("GLTF export failed. Install pygltflib or check the log.")
    if progress_callback:
        progress_callback("GLB/GLTF export complete", 100)
    return path


def _work_export_mdl_binary(model, path: str, *, game_version, progress_callback=None, is_cancelled=None):
    from src.core.mdl.mdl_writer import MDLBinaryWriter
    from src.core.geometry.model_data import GameVersion

    mdl = copy.deepcopy(model)
    mdl.game_version = GameVersion.K2 if game_version == "K2" else GameVersion.K1
    mdx_path = str(Path(path).with_suffix(".mdx"))
    if progress_callback:
        progress_callback("Writing binary MDL/MDX\u2026", 50)
    mdl_bytes, mdx_bytes = MDLBinaryWriter().write(mdl)
    Path(path).write_bytes(mdl_bytes)
    Path(mdx_path).write_bytes(mdx_bytes)
    if progress_callback:
        progress_callback("Binary MDL export complete", 100)
    return path, mdx_path


def _work_export_fbx(
    model,
    path: str,
    *,
    tex_cache=None,
    base_skeleton_model=None,
    compatibility_profile="standard",
    selected_animation_names=None,
    animation_resource_manager=None,
    animation_game="",
    supplemental_animation_models=(),
    progress_callback=None,
    is_cancelled=None,
):
    from src.converters.mesh_converter import FBXExporter

    try:
        has_bas_layers = any(
            bool(getattr(node, "_gr_bas_attachment_layer", False))
            for node in model.all_nodes()
        )
    except Exception:
        has_bas_layers = False
    if has_bas_layers:
        from src.systems.bas.preview_composer import prepare_bas_composed_export_model

        model, _report = prepare_bas_composed_export_model(
            model,
            require_unique_body_names=True,
        )

    # Animation inheritance is a workflow concern, not an FBX writer concern.
    # Materialize the exact selected local/head/supermodel takes before the IO
    # layer serializes ``model.animations``.  ``None`` preserves the legacy
    # local-list behavior; an empty tuple deliberately exports mesh + rig only.
    if selected_animation_names is not None:
        if progress_callback:
            progress_callback("Resolving selected animation sets\u2026", 20)
        from src.core.animation.fbx_animation_selection import (
            prepare_fbx_animation_export_model,
        )

        model = prepare_fbx_animation_export_model(
            model,
            tuple(selected_animation_names),
            game=animation_game,
            resource_manager=animation_resource_manager,
            base_skeleton_model=base_skeleton_model,
            supplemental_models=tuple(supplemental_animation_models or ()),
            require_all=True,
        )

    if progress_callback:
        progress_callback("Writing FBX geometry and rigging\u2026", 40)
    ok = FBXExporter().export(
        model,
        path,
        tex_cache=tex_cache,
        export_rigging=True,
        base_skeleton_model=base_skeleton_model,
        compatibility_profile=compatibility_profile,
    )
    if not ok:
        raise RuntimeError("FBX export failed. Check the export log for details.")
    if progress_callback:
        progress_callback("FBX export complete", 100)
    return path


def _work_export_selected_fbx(
    scene_objects,
    path: str,
    *,
    tex_cache=None,
    progress_callback=None,
    is_cancelled=None,
):
    from src.converters.mesh_converter import FBXExporter
    from src.io.fbx.fbx_exporter import merge_selected_scene_objects

    if progress_callback:
        progress_callback("Combining selected scene objects into one mesh…", 20)
    merged = merge_selected_scene_objects(scene_objects, name=Path(path).stem or "selection")
    if progress_callback:
        progress_callback("Writing one-mesh FBX geometry and materials…", 50)
    ok = FBXExporter().export(
        merged,
        path,
        tex_cache=tex_cache,
        export_rigging=False,
        compatibility_profile="standard",
        force_ascii=True,
    )
    if not ok:
        raise RuntimeError("Selected FBX export failed. Check the export log for details.")
    if progress_callback:
        progress_callback("Selected FBX export complete", 100)
    return path


def _work_export_clean_module_obj(
    scene_objects,
    path: str,
    *,
    tex_cache=None,
    progress_callback=None,
    is_cancelled=None,
):
    from src.converters.mesh_converter import OBJExporter
    from src.core.export.export_job import promote_staged_files
    from src.io.export_control import check_export_cancelled
    from src.io.clean_module_obj_export import prepare_clean_module_obj_model

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{target.stem}-export-",
            dir=str(target.parent),
        )
    )
    staged_obj = stage_dir / target.name
    try:
        check_export_cancelled(is_cancelled)
        if progress_callback:
            progress_callback("Combining every room in the loaded module…", 15)
        merged, summary = prepare_clean_module_obj_model(
            scene_objects,
            name=target.stem or "module_clean",
            is_cancelled=is_cancelled,
        )
        if progress_callback:
            progress_callback(
                "Removing skybox and oversized background geometry…",
                40,
            )
        export_result = OBJExporter().export(
            merged,
            str(staged_obj),
            tex_cache=tex_cache,
            export_rigging=False,
            is_cancelled=is_cancelled,
        )
        summary = replace(summary, texture_sidecars=export_result.texture_sidecars)
        check_export_cancelled(is_cancelled)
        if progress_callback:
            progress_callback("Publishing clean map files…", 95)
        promote_staged_files(stage_dir, target.parent)
        if progress_callback:
            progress_callback("Clean module OBJ export complete", 100)
        return path, summary
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


def _work_run_mdlops(cmd, cwd, *, progress_callback=None, is_cancelled=None):
    if progress_callback:
        progress_callback("Running MDLOps\u2026", 30)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(cwd))


class _IoGuiCallbackBridge(QtCore.QObject):
    """Receive background I/O signals on the GUI thread."""

    def __init__(
        self,
        owner,
        description: str,
        worker: BackgroundIOWorker,
        thread: QtCore.QThread,
        progress_dialog: QtWidgets.QProgressDialog,
        on_complete,
        on_error,
        error_category: str,
    ):
        super().__init__(owner)
        self._owner = owner
        self._description = description
        self._worker = worker
        self._thread = thread
        self._progress_dialog = progress_dialog
        self._on_complete = on_complete
        self._on_error = on_error
        self._error_category = error_category
        self._done = False

    def _cleanup(self) -> None:
        if self._done:
            return
        self._done = True
        try:
            self._progress_dialog.reset()
            self._progress_dialog.close()
            self._progress_dialog.deleteLater()
        except RuntimeError:
            pass
        try:
            self._worker.deleteLater()
        except RuntimeError:
            pass
        try:
            self._thread.quit()
        except RuntimeError:
            pass
        self._owner._io_worker = None
        self._owner._io_thread = None
        self._owner._io_progress_dialog = None
        self._owner._io_callback_bridge = None
        self.deleteLater()

    @QtCore.Slot(str, int)
    def on_progress(self, message, percent):
        try:
            self._progress_dialog.setLabelText(message or self._description)
            self._progress_dialog.setValue(max(0, min(100, int(percent))))
        except RuntimeError:
            pass

    @QtCore.Slot(object)
    def on_finished(self, result):
        cancelled = self._worker.is_cancelled()
        self._cleanup()
        try:
            if self._on_complete is not None:
                self._on_complete(result, cancelled=cancelled)
        except Exception as exc:  # noqa: BLE001 - post-processing must not crash UI
            log.error("Post-processing error after %s", self._description, exc_info=True)
            show_exception(self._owner, "io_error", exc, context=f"Finishing {self._description}")

    @QtCore.Slot(str, str, object)
    def on_error(self, message, tb, exc):
        self._cleanup()
        self._owner._log(f"{self._description} failed:\n{tb}", "error")
        handled = False
        if self._on_error is not None:
            try:
                handled = bool(self._on_error(exc))
            except Exception:  # noqa: BLE001 - error handler must not crash UI
                log.error("on_error handler failed", exc_info=True)
        if not handled:
            show_exception(self._owner, self._error_category, exc, context=self._description)

    @QtCore.Slot()
    def on_canceled(self):
        try:
            # The worker thread is occupied by the synchronous export. Setting
            # its threading.Event directly makes cancellation visible now,
            # rather than queuing the request behind the work being cancelled.
            self._worker.request_cancel()
        except RuntimeError:
            pass


class ModelIoMixin:
    """Model import, export, FBX SDK, MDLOps, and module-file command helpers."""

    def _fbx_resource_context_for_export(self, model):
        """Return the headless resource manager and strict K1/K2 game tag."""
        manager = None
        get_manager = getattr(self, "_get_resource_manager", None)
        if callable(get_manager):
            try:
                manager = get_manager()
            except Exception:
                log.debug("Could not obtain the resource manager for FBX export", exc_info=True)
        if manager is None:
            manager = getattr(self, "_resource_manager", None) or getattr(self, "resource_manager", None)
        game = str(
            getattr(self, "_current_game", "")
            or (self._infer_game_from_model(model) if hasattr(self, "_infer_game_from_model") else "")
            or "K1"
        ).upper()
        return manager, game

    def _fbx_base_skeleton_for_export(self, model):
        """Resolve the model's supermodel for complete FBX bind matrices."""
        supermodel = str(getattr(model, "supermodel", "") or "").strip()
        if not supermodel or supermodel.lower() in {"null", "none", "****"}:
            return None
        manager, game = self._fbx_resource_context_for_export(model)
        if manager is None:
            return None
        try:
            # The Animation Browser commonly resolved this same supermodel
            # while populating its inherited catalog. Reuse that strict-game
            # cache instead of reparsing a large model immediately before the
            # FBX selector opens. A cold lookup still uses the manager's strict
            # loader through SuperModelResolver.
            from src.core.animation.animation_engine import SuperModelResolver

            SuperModelResolver.configure(manager)
            return SuperModelResolver.load_supermodel(supermodel, game)
        except Exception:
            log.warning("Could not resolve FBX base skeleton %s (%s)", supermodel, game, exc_info=True)
            return None

    def _fbx_supplemental_animation_models(self, model):
        """Return attached source models whose facial/accessory tracks are usable."""
        result = []
        seen = set()
        try:
            nodes = model.all_nodes()
        except Exception:
            nodes = []
        for node in nodes:
            source = getattr(node, "_gr_bas_attachment_source_model_ref", None)
            if source is None or id(source) in seen:
                continue
            seen.add(id(source))
            result.append(source)
        return tuple(result)

    def _choose_fbx_animation_sets(
        self,
        model,
        compatibility_profile: str,
        *,
        base_skeleton_model=None,
        supplemental_models=(),
    ):
        """Show the shared take selector; return tuple, or ``None`` on cancel."""
        manager, game = self._fbx_resource_context_for_export(model)
        try:
            from src.core.animation.fbx_animation_selection import list_fbx_animation_sets
            from src.gui.qt_lib.dialogs.qt_fbx_animation_selection_dialog import (
                QtFbxAnimationSelectionDialog,
            )

            rows = list_fbx_animation_sets(
                model,
                game=game,
                resource_manager=manager,
                base_skeleton_model=base_skeleton_model,
                supplemental_models=tuple(supplemental_models or ()),
            )
        except Exception as exc:
            log.exception("Could not enumerate FBX animation sets")
            QtWidgets.QMessageBox.warning(
                self,
                "Export FBX",
                f"Animation sets could not be enumerated:\n{exc}",
            )
            return None

        initial = [
            str(getattr(anim, "name", "") or "")
            for anim in list(getattr(model, "animations", []) or [])
            if str(getattr(anim, "name", "") or "")
        ]
        current_name = ""
        panel = getattr(self, "animations_panel", None)
        selected_getter = getattr(panel, "selected_animation", None)
        if callable(selected_getter):
            try:
                current_name = str(selected_getter() or "")
            except Exception:
                current_name = ""
        if current_name and current_name.lower() not in {name.lower() for name in initial}:
            initial.append(current_name)

        dialog = QtFbxAnimationSelectionDialog(
            rows,
            self,
            profile=compatibility_profile,
            initial_selected_names=tuple(initial),
            current_animation_name=current_name,
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return None
        return tuple(dialog.selected_animation_names())

    # ------------------------------------------------------------------
    # Async I/O routing helpers
    # ------------------------------------------------------------------

    def _io_worker_is_running(self) -> bool:
        thread = getattr(self, "_io_thread", None)
        if thread is None:
            return False
        try:
            return thread.isRunning()
        except RuntimeError:
            self._io_thread = None
            self._io_worker = None
            return False

    def _run_io_async(
        self,
        description: str,
        fn,
        *args,
        on_complete=None,
        on_error=None,
        error_category: str = "io_error",
        **kwargs,
    ):
        """Run a blocking callable on a background ``QThread``.

        Shows a window-modal :class:`QProgressDialog` (with a Cancel button)
        while the work runs off the GUI thread. ``on_complete(result,
        cancelled=False)`` and ``on_error(exc) -> bool`` callbacks run on the GUI
        thread. If ``on_error`` returns True the failure is considered handled
        and no generic :class:`ErrorReport` dialog is shown.

        Returns the :class:`BackgroundIOWorker`, or ``None`` if another
        background I/O job is already running.
        """

        if self._io_worker_is_running():
            self._log("Another background operation is already running.", "warning")
            return None

        worker = BackgroundIOWorker(fn, args=args, kwargs=kwargs)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        self._io_worker = worker
        self._io_thread = thread

        progress_dialog = QtWidgets.QProgressDialog(description, "Cancel", 0, 100, self)
        progress_dialog.setWindowTitle(description)
        progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setAutoClose(True)
        progress_dialog.setAutoReset(False)
        progress_dialog.setValue(0)
        progress_dialog.setMinimumWidth(360)
        self._io_progress_dialog = progress_dialog
        bridge = _IoGuiCallbackBridge(
            self,
            description,
            worker,
            thread,
            progress_dialog,
            on_complete,
            on_error,
            error_category,
        )
        self._io_callback_bridge = bridge

        thread.started.connect(worker.run)
        thread.finished.connect(thread.deleteLater)
        worker.progress.connect(bridge.on_progress, QtCore.Qt.QueuedConnection)
        worker.error.connect(bridge.on_error, QtCore.Qt.QueuedConnection)
        worker.finished.connect(bridge.on_finished, QtCore.Qt.QueuedConnection)
        progress_dialog.canceled.connect(bridge.on_canceled)
        thread.start()
        return worker

    def _import_obj(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import OBJ",
            str(Path(self.settings_data.get("last_import") or self.app_root)),
            "OBJ files (*.obj);;All files (*.*)",
        )
        if path:
            self._import_obj_from_path(path)
    def _import_obj_from_path(self, path: str):
        game_version = self._game_version()

        def _on_complete(model, cancelled=False):
            if cancelled or model is None:
                return
            self._texture_dir = str(Path(path).parent)
            self._set_model_internal(model, path)
            self._log(f"Imported OBJ: {Path(path).name}", "success")

        self._run_io_async(
            f"Importing OBJ \u2014 {Path(path).name}",
            _work_import_obj,
            path,
            game_version=game_version,
            on_complete=_on_complete,
            error_category="import_error",
        )

    def _import_fbx(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import FBX",
            str(Path(self.settings_data.get("last_import") or self.app_root)),
            "FBX files (*.fbx);;All files (*.*)",
        )
        if not path:
            return
        backend = self._choose_fbx_import_backend(path)
        if backend is None:
            return

        def _on_complete(model, cancelled=False):
            if cancelled or model is None:
                return
            self._texture_dir = str(Path(path).parent)
            self._set_model_internal(model, path)
            summary = getattr(model, "fbx_import_summary", None)
            suffix = f" ({summary.log_line()})" if summary is not None else ""
            self._log(f"Imported FBX: {Path(path).name}{suffix}", "success")

        def _on_error(exc):
            try:
                from src.io.fbx.fbx_importer import FbxSdkUnavailableError
            except Exception:
                return False
            if isinstance(exc, FbxSdkUnavailableError):
                self._show_missing_fbx_sdk_dialog(str(exc))
                return True
            return False

        self._import_fbx_model(
            path,
            backend=backend,
            on_complete=_on_complete,
            on_error=_on_error,
        )

    def _auto_detect_fbx_import_backend(self, path: str) -> tuple[str | None, str]:
        """Choose the best FBX import backend before import begins."""

        try:
            self._configure_fbx_sdk_paths(refresh=True)
            from src.io.fbx.fbx_sdk_loader import is_fbx_sdk_available

            if is_fbx_sdk_available():
                return "autodesk_sdk", "Autodesk FBX SDK is configured and available for SDK-backed scene import."
        except Exception as exc:
            sdk_reason = f"Autodesk FBX SDK is not available: {exc}"
        else:
            sdk_reason = "Autodesk FBX SDK is not configured for this Python runtime."

        try:
            from src.core.retargeting.fbx_exporter import find_blender_executable

            blender_exe = find_blender_executable()
            return "blender", f"{sdk_reason} Blender FBX is available at {blender_exe}."
        except Exception as exc:
            return None, f"{sdk_reason} Blender FBX is also unavailable: {exc}"
    def _choose_fbx_import_backend(self, path: str) -> str | None:
        detected_backend, reason = self._auto_detect_fbx_import_backend(path)
        labels = {
            "autodesk_sdk": "Autodesk FBX SDK",
            "blender": "Blender FBX",
        }
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Import FBX")
        box.setText("Choose an FBX import backend.")
        box.setInformativeText(
            "Auto-detection checks configured Autodesk FBX SDK first, then the Blender bridge.\n\n"
            f"Detected: {labels.get(detected_backend, 'No usable backend')}\n"
            f"Reason: {reason}\n\n"
            "Autodesk FBX SDK and Blender FBX are separate import backends; GhostRigger will not silently switch after import starts."
        )
        auto_btn = None
        if detected_backend is not None:
            auto_btn = box.addButton(f"Use Auto: {labels[detected_backend]}", QtWidgets.QMessageBox.AcceptRole)
        sdk_btn = box.addButton("Autodesk FBX SDK", QtWidgets.QMessageBox.AcceptRole)
        blender_btn = box.addButton("Blender FBX", QtWidgets.QMessageBox.ActionRole)
        box.addButton(QtWidgets.QMessageBox.Cancel)
        if auto_btn is not None:
            box.setDefaultButton(auto_btn)
        box.exec()
        clicked = box.clickedButton()
        if auto_btn is not None and clicked is auto_btn:
            self._log(f"Auto-detected FBX import backend: {labels[detected_backend]} ({reason})", "info")
            return detected_backend
        if clicked is sdk_btn:
            return "autodesk_sdk"
        if clicked is blender_btn:
            return "blender"
        return None
    def _import_fbx_model(self, path: str, *, backend: str, on_complete=None, on_error=None):
        """Import FBX through the explicitly selected backend.

        When ``on_complete`` is provided the import runs asynchronously on a
        background thread via :meth:`_run_io_async`; ``on_complete(model,
        cancelled=False)`` is invoked on the GUI thread. When ``on_complete`` is
        ``None`` (the synchronous fallback) the model is returned directly with a
        wait cursor shown, preserving backward compatibility.
        """

        fbx_settings = self.settings_data.get("fbx_sdk") or {}
        game_version = self._game_version()

        if backend == "autodesk_sdk":
            if not self._ensure_fbx_sdk_available_for_action("Import FBX"):
                if on_complete is not None:
                    on_complete(None, cancelled=False)
                return None
            self._configure_fbx_sdk_paths(refresh=True)
            work = _work_import_fbx_sdk
            work_kwargs = {"game_version": game_version, "fbx_sdk_settings": fbx_settings}
        elif backend == "blender":
            work = _work_import_fbx_blender
            work_kwargs = {"game_version": game_version}
        else:
            raise ValueError(f"Unknown FBX import backend: {backend}")

        if on_complete is None:
            # Synchronous fallback with a wait cursor.
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                try:
                    return work(path, **work_kwargs)
                except Exception as exc:
                    if on_error is not None and on_error(exc):
                        return None
                    if backend == "autodesk_sdk":
                        try:
                            from src.io.fbx.fbx_importer import FbxSdkUnavailableError
                        except Exception:
                            FbxSdkUnavailableError = ()  # type: ignore
                        if isinstance(exc, FbxSdkUnavailableError):
                            self._show_missing_fbx_sdk_dialog(str(exc))
                            return None
                    raise
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()

        # Asynchronous path.
        self._run_io_async(
            f"Importing FBX \u2014 {Path(path).name}",
            work,
            path,
            on_complete=on_complete,
            on_error=on_error,
            error_category="import_error",
            **work_kwargs,
        )
        return None
    def _import_gltf(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import GLB / GLTF",
            str(Path(self.settings_data.get("last_import") or self.app_root)),
            "GLB/GLTF files (*.glb *.gltf);;All files (*.*)",
        )
        if not path:
            return
        game_version = self._game_version()

        def _on_complete(model, cancelled=False):
            if cancelled or model is None:
                return
            self._texture_dir = str(Path(path).parent)
            self._set_model_internal(model, path)
            self._log(f"Imported GLB/GLTF: {Path(path).name}", "success")

        self._run_io_async(
            f"Importing GLB/GLTF \u2014 {Path(path).name}",
            _work_import_gltf,
            path,
            game_version=game_version,
            on_complete=_on_complete,
            error_category="import_error",
        )
    def _save_ascii_mdl(self):
        model = self._require_model("Save ASCII MDL")
        if model is None:
            return
        chosen_gv = self._pick_export_game_version()
        if not chosen_gv:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save ASCII MDL",
            f"{getattr(model, 'name', 'model')}.mdl",
            "MDL files (*.mdl);;All files (*.*)",
        )
        if not path:
            return
        try:
            from src.core.mdl.mdl_parser import MDLAsciiWriter
            from src.core.geometry.model_data import GameVersion

            mdl = copy.deepcopy(model)
            mdl.game_version = GameVersion.K2 if chosen_gv == "K2" else GameVersion.K1
            MDLAsciiWriter().write(mdl, path)
            self._log(f"Saved ASCII MDL ({chosen_gv}) -> {Path(path).name}", "success")
        except Exception as exc:
            self._log(f"Save error: {exc}", "error")
            show_exception(self, "save_error", exc, context="Saving ASCII MDL")
    def _export_mdl_binary(self):
        model = self._require_model("Export Binary MDL")
        if model is None:
            return
        chosen_gv = self._pick_export_game_version()
        if not chosen_gv:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Binary MDL",
            f"{getattr(model, 'name', 'model')}.mdl",
            "MDL files (*.mdl);;All files (*.*)",
        )
        if not path:
            return

        def _on_complete(result, cancelled=False):
            if cancelled or result is None:
                return
            _path, mdx_path = result
            self._log(
                f"Exported binary MDL ({chosen_gv}) -> {Path(_path).name} (+ {Path(mdx_path).name})",
                "success",
            )

        self._run_io_async(
            f"Exporting binary MDL \u2014 {Path(path).name}",
            _work_export_mdl_binary,
            model,
            path,
            game_version=chosen_gv,
            on_complete=_on_complete,
            error_category="export_error",
        )
    def _export_obj(self):
        model = self._require_model("Export OBJ")
        if model is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export OBJ",
            f"{getattr(model, 'name', 'model')}.obj",
            "OBJ files (*.obj);;All files (*.*)",
        )
        if not path:
            return
        tex_cache = self._get_tex_cache_for_export()

        def _on_complete(result, cancelled=False):
            if cancelled or result is None:
                return
            self._log(f"Exported OBJ -> {Path(result).name}", "success")

        self._run_io_async(
            f"Exporting OBJ \u2014 {Path(path).name}",
            _work_export_obj,
            model,
            path,
            tex_cache=tex_cache,
            on_complete=_on_complete,
            error_category="export_error",
        )
    def _export_fbx(self):
        model = self._require_model("Export FBX")
        if model is None:
            return
        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export FBX",
            f"{getattr(model, 'name', 'model')}.fbx",
            FBX_EXPORT_FILTERS,
        )
        if not path:
            return
        tex_cache = self._get_tex_cache_for_export()
        compatibility_profile = _fbx_compatibility_profile_from_filter(selected_filter)
        base_skeleton_model = self._fbx_base_skeleton_for_export(model)
        supplemental_models = self._fbx_supplemental_animation_models(model)
        selected_animation_names = self._choose_fbx_animation_sets(
            model,
            compatibility_profile,
            base_skeleton_model=base_skeleton_model,
            supplemental_models=supplemental_models,
        )
        if selected_animation_names is None:
            self._log("FBX export cancelled before animation selection.", "info")
            return
        animation_resource_manager, animation_game = self._fbx_resource_context_for_export(model)

        def _on_complete(result, cancelled=False):
            if cancelled or result is None:
                return
            self._log(f"Exported FBX -> {Path(result).name}", "success")

        def _on_error(exc):
            try:
                from src.io.fbx.fbx_exporter import FbxSdkUnavailableError
            except Exception:
                return False
            if isinstance(exc, FbxSdkUnavailableError):
                self._show_missing_fbx_sdk_dialog(str(exc))
                return True
            return False

        self._run_io_async(
            f"Exporting FBX \u2014 {Path(path).name}",
            _work_export_fbx,
            model,
            path,
            tex_cache=tex_cache,
            base_skeleton_model=base_skeleton_model,
            compatibility_profile=compatibility_profile,
            selected_animation_names=selected_animation_names,
            animation_resource_manager=animation_resource_manager,
            animation_game=animation_game,
            supplemental_animation_models=supplemental_models,
            on_complete=_on_complete,
            on_error=_on_error,
            error_category="export_error",
        )
    def _export_selected_fbx(self):
        selected = _selected_scene_objects_for_export(
            self.scene_manager,
            getattr(self, "viewport", None),
        )
        if not selected:
            QtWidgets.QMessageBox.information(self, "Export Selected FBX", "Select a scene object first.")
            return
        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Selected FBX",
            f"{selected[0].name if len(selected) == 1 else 'selection'}.fbx",
            FBX_EXPORT_FILTERS,
        )
        if not path:
            return
        compatibility_profile = _fbx_compatibility_profile_from_filter(selected_filter)
        if len(selected) != 1 and compatibility_profile != "standard":
            QtWidgets.QMessageBox.information(
                self,
                "Export Selected FBX",
                "Unity-, Unreal-, and 3ds Max-compatible FBX export requires one runtime model. "
                "For a body with attached layers, use Body Attachment System > "
                "Export Composed Model… so the rig can be normalized as one asset.",
            )
            return
        if compatibility_profile == "standard":
            def _on_complete(result, cancelled=False):
                if cancelled or result is None:
                    return
                self._log(f"Exported selected objects as one FBX mesh -> {Path(path).name}", "success")

            self._run_io_async(
                f"Combining and exporting selection — {Path(path).name}",
                _work_export_selected_fbx,
                selected,
                path,
                tex_cache=self._get_tex_cache_for_export(),
                on_complete=_on_complete,
                error_category="export_error",
            )
            return
        if len(selected) == 1:
            model_getter = getattr(self, "_runtime_model_for_scene_object", None)
            model = model_getter(selected[0]) if callable(model_getter) else (getattr(selected[0], "metadata", {}) or {}).get("_runtime_model")
            if model is not None:
                tex_cache = self._get_tex_cache_for_export()
                base_skeleton_model = self._fbx_base_skeleton_for_export(model)
                supplemental_models = self._fbx_supplemental_animation_models(model)
                selected_animation_names = self._choose_fbx_animation_sets(
                    model,
                    compatibility_profile,
                    base_skeleton_model=base_skeleton_model,
                    supplemental_models=supplemental_models,
                )
                if selected_animation_names is None:
                    self._log("Selected FBX export cancelled before animation selection.", "info")
                    return
                animation_resource_manager, animation_game = self._fbx_resource_context_for_export(model)

                def _on_complete(result, cancelled=False):
                    if cancelled or result is None:
                        return
                    self._log(f"Exported selected FBX -> {Path(result).name}", "success")

                self._run_io_async(
                    f"Exporting selected FBX \u2014 {Path(path).name}",
                    _work_export_fbx,
                    model,
                    path,
                    tex_cache=tex_cache,
                    base_skeleton_model=base_skeleton_model,
                    compatibility_profile=compatibility_profile,
                    selected_animation_names=selected_animation_names,
                    animation_resource_manager=animation_resource_manager,
                    animation_game=animation_game,
                    supplemental_animation_models=supplemental_models,
                    on_complete=_on_complete,
                    error_category="export_error",
                )
                return
            if compatibility_profile != "standard":
                QtWidgets.QMessageBox.information(
                    self,
                    "Export Selected FBX",
                    "The selected scene object has no runtime model to send through "
                    "the Unity, Unreal Engine, or 3ds Max compatibility exporter.",
                )
                return
        try:
            from src.io.fbx.fbx_exporter import FbxSdkUnavailableError, export_fbx

            export_fbx(selected, path, {"export_selection_only": True, "fbx_sdk": self.settings_data.get("fbx_sdk")})
            self._log(f"Exported selected FBX -> {Path(path).name}", "success")
        except FbxSdkUnavailableError as exc:
            self._show_missing_fbx_sdk_dialog(str(exc))
        except Exception as exc:
            self._log(f"Selected FBX export error: {exc}", "error")
            show_exception(self, "export_error", exc, context="Exporting selected mesh to FBX")

    def _export_full_module_fbx(self):
        module_objects = _module_group_scene_objects_for_export(
            self.scene_manager,
            getattr(self, "viewport", None),
        )
        if not module_objects:
            QtWidgets.QMessageBox.information(
                self,
                "Export Full Module as One FBX",
                "Select any room in a loaded module layout first.",
            )
            return
        module_group = (
            (getattr(module_objects[0], "metadata", {}) or {}).get("module_group")
            or {}
        )
        module_root = str(module_group.get("module_root") or "module").strip()
        path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Full Module as One FBX",
            f"{module_root}_full.fbx",
            "Standard FBX (*.fbx);;All files (*.*)",
        )
        if not path:
            return

        def _on_complete(result, cancelled=False):
            if cancelled or result is None:
                return
            self._log(
                f"Exported full module as one FBX mesh "
                f"({len(module_objects)} scene objects) -> {Path(result).name}",
                "success",
            )

        self._run_io_async(
            f"Combining and exporting full module — {Path(path).name}",
            _work_export_selected_fbx,
            module_objects,
            path,
            tex_cache=self._get_tex_cache_for_export(),
            on_complete=_on_complete,
            error_category="export_error",
        )

    def _export_module_obj_variant(
        self,
        *,
        worker,
        title: str,
        filename_suffix: str,
        progress_text: str,
        success_text: str,
    ):
        module_objects = _module_group_scene_objects_for_export(
            self.scene_manager,
            getattr(self, "viewport", None),
        )
        if not module_objects:
            QtWidgets.QMessageBox.information(
                self,
                title,
                "Select any room in a loaded module layout first.",
            )
            return
        module_group = (
            (getattr(module_objects[0], "metadata", {}) or {}).get("module_group")
            or {}
        )
        module_root = str(module_group.get("module_root") or "module").strip()
        path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            title,
            f"{module_root}_{filename_suffix}.obj",
            "Wavefront OBJ (*.obj);;All files (*.*)",
        )
        if not path:
            return

        def _on_complete(result, cancelled=False):
            if cancelled or result is None:
                return
            output_path, texture_sidecars = result
            self._log(
                f"{success_text} "
                f"({len(module_objects)} scene objects, "
                f"{texture_sidecars.saved}/{texture_sidecars.requested} textures written) -> "
                f"{Path(output_path).name}",
                "success",
            )

        self._run_io_async(
            f"{progress_text} — {Path(path).name}",
            worker,
            module_objects,
            path,
            tex_cache=self._get_tex_cache_for_export(),
            on_complete=_on_complete,
            error_category="export_error",
        )

    def _export_full_module_obj(self):
        from src.io.full_module_obj_export import export_full_module_obj

        self._export_module_obj_variant(
            worker=export_full_module_obj,
            title="Export Full Module as One OBJ",
            filename_suffix="full",
            progress_text="Combining and exporting full module OBJ",
            success_text="Exported full module as one OBJ mesh",
        )

    def _export_maya_room_module_obj(self):
        from src.io.full_module_obj_export import export_maya_room_module_obj

        self._export_module_obj_variant(
            worker=export_maya_room_module_obj,
            title="Export Maya-Friendly Module (Room Objects)",
            filename_suffix="maya_rooms",
            progress_text="Preparing Maya-friendly room-object OBJ",
            success_text="Exported Maya-friendly module OBJ with one object per room",
        )

    def _export_clean_module_obj(self):
        module_objects = _module_group_scene_objects_for_export(
            self.scene_manager,
            getattr(self, "viewport", None),
        )
        if not module_objects:
            QtWidgets.QMessageBox.information(
                self,
                "Export Clean Full Map as OBJ",
                "Select any room in a loaded module layout first.\n\n"
                "Ghost Studio will automatically include every room in that map.",
            )
            return
        module_group = (
            (getattr(module_objects[0], "metadata", {}) or {}).get("module_group")
            or {}
        )
        module_root = str(module_group.get("module_root") or "module").strip()
        path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Clean Full Map as OBJ",
            f"{module_root}_map_no_skybox.obj",
            "Wavefront OBJ (*.obj);;All files (*.*)",
        )
        if not path:
            return

        def _on_complete(result, cancelled=False):
            if cancelled or result is None:
                return
            output_path, summary = result
            removed_faces = (
                summary.removed_skybox_faces
                + summary.removed_background_faces
            )
            texture_sidecars = summary.texture_sidecars
            self._log(
                f"Exported clean full-map OBJ ({summary.scene_objects} rooms, "
                f"{summary.exported_faces:,} faces, {removed_faces:,} backdrop "
                f"faces removed, {texture_sidecars.saved}/"
                f"{texture_sidecars.requested} textures written) -> "
                f"{Path(output_path).name}",
                "success",
            )
            texture_status = (
                f"Texture files written: {texture_sidecars.saved} of "
                f"{texture_sidecars.requested}"
            )
            unavailable = texture_sidecars.unavailable
            if unavailable:
                missing_names = (
                    *texture_sidecars.missing_names,
                    *texture_sidecars.failed_names,
                )
                preview = ", ".join(missing_names[:6])
                if len(missing_names) > 6:
                    preview += f", and {len(missing_names) - 6} more"
                texture_status += (
                    f"\nWarning: {unavailable} texture(s) were unavailable"
                    + (f" ({preview})" if preview else "")
                    + ". The OBJ and MTL are complete, but those materials "
                    "will need their texture files supplied separately."
                )
            message = (
                f"Exported the complete map as one OBJ.\n\n"
                f"Rooms included: {summary.scene_objects}\n"
                f"Map faces: {summary.exported_faces:,}\n"
                f"Skybox faces removed: {summary.removed_skybox_faces:,}\n"
                f"Large background faces removed: "
                f"{summary.removed_background_faces:,}\n"
                f"Materials: {summary.materials}\n"
                f"{texture_status}\n\n"
                f"OBJ and MTL files were written at:\n"
                f"{Path(output_path).parent}"
            )
            dialog = (
                QtWidgets.QMessageBox.warning
                if unavailable
                else QtWidgets.QMessageBox.information
            )
            dialog(
                self,
                "Clean Map Export Complete",
                message,
            )

        self._run_io_async(
            f"Preparing clean full-map OBJ — {Path(path).name}",
            _work_export_clean_module_obj,
            module_objects,
            path,
            tex_cache=self._get_tex_cache_for_export(),
            on_complete=_on_complete,
            error_category="export_error",
        )

    def _show_missing_fbx_sdk_dialog(self, details: str = "") -> None:
        message = (
            "Autodesk FBX Python SDK is not installed or not available to this Python environment. "
            "FBX import/export is disabled until the SDK is installed."
        )
        if details:
            message = f"{message}\n\n{details}"
        self._log("FBX SDK unavailable; import/export disabled.", "warning")
        QtWidgets.QMessageBox.warning(self, "Autodesk FBX SDK Missing", message)
    def _ensure_fbx_sdk_available_for_action(self, action: str) -> bool:
        self._configure_fbx_sdk_paths()
        try:
            from src.io.fbx.fbx_sdk_loader import is_fbx_sdk_available

            if is_fbx_sdk_available():
                return True
        except Exception:
            pass
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle(action)
        box.setText("FBX support requires Autodesk FBX Python SDK.")
        box.setInformativeText(
            "GhostRigger does not bundle Autodesk SDK files. You must download and install the SDK separately from Autodesk, then configure the SDK path.\n\nOpen setup assistant now?"
        )
        setup_btn = box.addButton("Open Setup Assistant", QtWidgets.QMessageBox.AcceptRole)
        download_btn = box.addButton("Open Autodesk Download Page", QtWidgets.QMessageBox.ActionRole)
        box.addButton(QtWidgets.QMessageBox.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is setup_btn:
            self._open_fbx_sdk_setup()
        elif clicked is download_btn:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl("https://aps.autodesk.com/developer/overview/fbx-sdk"))
            self._log("Opened Autodesk FBX SDK download page.", "info")
        return False
    def _show_fbx_sdk_status(self) -> None:
        self._configure_fbx_sdk_paths(refresh=True)
        try:
            from src.io.fbx.fbx_diagnostics import build_fbx_diagnostic_report

            report = build_fbx_diagnostic_report(self.settings_data.get("fbx_sdk"))
        except Exception as exc:
            report = f"FBX SDK diagnostic failed: {exc}"
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("FBX SDK Status")
        dialog.resize(720, 480)
        layout = QtWidgets.QVBoxLayout(dialog)
        text = QtWidgets.QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(report)
        layout.addWidget(text, 1)
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button, 0, QtCore.Qt.AlignRight)
        dialog.exec()
    def _open_fbx_sdk_setup(self) -> None:
        try:
            from src.gui.qt_lib.dialogs.fbx_sdk_setup_dialog import FbxSdkSetupDialog

            dialog = FbxSdkSetupDialog(self.settings_data, self)
            dialog.configurationSaved.connect(self._save_fbx_sdk_settings)
            dialog.exec()
        except Exception as exc:
            self._log(f"FBX SDK setup error: {exc}", "error")
            show_exception(self, "fbx_sdk_error", exc, context="FBX SDK setup")
    def _save_fbx_sdk_settings(self, fbx_settings: dict) -> None:
        self.settings_data["fbx_sdk"] = dict(fbx_settings or {})
        self._configure_fbx_sdk_paths(refresh=True)
        try:
            save_settings(self.settings_path, self.settings_data)
            self._log("FBX SDK path configured.", "success" if fbx_settings.get("last_verified_ok") else "warning")
        except Exception as exc:
            self._log(f"FBX SDK settings save failed: {exc}", "error")
    def _configure_fbx_sdk_paths(self, *, refresh: bool = False) -> None:
        try:
            from src.io.fbx.fbx_sdk_loader import configure_fbx_sdk_paths

            configure_fbx_sdk_paths(self.settings_data.get("fbx_sdk") or {}, refresh=refresh)
        except Exception as exc:
            log.debug("FBX SDK path configuration failed: %s", exc, exc_info=True)
    def _export_gltf(self):
        model = self._require_model("Export GLB/GLTF")
        if model is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export GLB / GLTF",
            f"{getattr(model, 'name', 'model')}.glb",
            "GLB binary (*.glb);;GLTF JSON (*.gltf);;All files (*.*)",
        )
        if not path:
            return
        tex_cache = self._get_tex_cache_for_export()
        binary = path.lower().endswith(".glb")

        def _on_complete(result, cancelled=False):
            if cancelled or result is None:
                return
            self._log(f"Exported {'GLB' if binary else 'GLTF'} -> {Path(result).name}", "success")

        self._run_io_async(
            f"Exporting {'GLB' if binary else 'GLTF'} \u2014 {Path(path).name}",
            _work_export_gltf,
            model,
            path,
            tex_cache=tex_cache,
            on_complete=_on_complete,
            error_category="export_error",
        )
    def _export_humanoid_template(self):
        chosen_gv = self._pick_export_game_version()
        if not chosen_gv:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Universal Humanoid Template",
            f"gr_humanoid_template_{chosen_gv.lower()}.mdl",
            "MDL files (*.mdl);;All files (*.*)",
        )
        if not path:
            return
        try:
            from src.core.templates.template_builder import build_humanoid_template, save_template_manifest
            from src.core.mdl.mdl_writer import MDLBinaryWriter

            model = build_humanoid_template(game_version=chosen_gv, name=Path(path).stem)
            mdx_path = str(Path(path).with_suffix(".mdx"))
            mdl_bytes, mdx_bytes = MDLBinaryWriter().write(model)
            Path(path).write_bytes(mdl_bytes)
            Path(mdx_path).write_bytes(mdx_bytes)
            manifest_path = save_template_manifest(model, str(Path(path).parent))
            self._log(
                f"Exported Humanoid Template ({chosen_gv}) -> {Path(path).name} (+ {Path(mdx_path).name})",
                "success",
            )
            self._log(f"Manifest -> {Path(manifest_path).name}", "info")
            if QtWidgets.QMessageBox.question(
                self,
                "Template Exported",
                "Load the exported template into the viewer now?",
            ) == QtWidgets.QMessageBox.Yes:
                self._set_model_internal(model, path)
        except Exception as exc:
            self._log(f"Template export error: {exc}", "error")
            show_exception(self, "export_error", exc, context="Exporting humanoid template")
    def _find_mdlops(self) -> str:
        configured = str(self.settings_data.get("mdlops_path") or "")
        guesses = [
            configured,
            str(self.app_root / "mdlops.pl"),
            str(self.app_root / "tools" / "mdlops.pl"),
        ]
        for candidate in guesses:
            if candidate and Path(candidate).exists():
                return candidate
        return ""
    def _set_mdlops(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Locate mdlops.pl or mdlops.exe",
            str(self.app_root),
            "MDLOps (*.pl *.exe *.py);;All files (*.*)",
        )
        if not path:
            return
        self.settings_data["mdlops_path"] = path
        try:
            save_settings(self.settings_path, self.settings_data)
        except Exception as exc:
            self._log(f"Could not save MDLOps setting: {exc}", "warning")
        self._log(f"MDLOps set: {path}", "success")
    def _mdlops_command(self, mdlops: str, game_flag: str, mode_flag: str, path: str) -> list[str]:
        if mdlops.lower().endswith(".pl"):
            return ["perl", mdlops, game_flag, mode_flag, path]
        return [mdlops, game_flag, mode_flag, path]
    def _compile_mdlops(self):
        model = self._require_model("Compile ASCII MDL to Binary")
        if model is None:
            return
        work_dir = Path(self.settings_data.get("work_dir") or self.app_root)
        work_dir.mkdir(parents=True, exist_ok=True)
        ascii_path = work_dir / f"{getattr(model, 'name', 'model')}.mdl"
        try:
            from src.core.mdl.mdl_parser import MDLAsciiWriter

            MDLAsciiWriter().write(model, str(ascii_path))
        except Exception as exc:
            self._log(f"Could not write ASCII MDL: {exc}", "error")
            return

        mdlops = self._find_mdlops()
        if not mdlops:
            QtWidgets.QMessageBox.information(
                self,
                "MDLOps",
                "MDLOps was not found. Set the path via MDLOps > Set MDLOps Path.\n\n"
                f"ASCII MDL has been saved to:\n{ascii_path}",
            )
            return
        game_name = getattr(getattr(model, "game_version", ""), "name", str(getattr(model, "game_version", "K1")))
        game_flag = "-k2" if game_name.upper() == "K2" else "-k1"
        cmd = self._mdlops_command(mdlops, game_flag, "-c", str(ascii_path))
        self._run_mdlops(cmd, work_dir)
    def _decompile_mdlops(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select binary MDL to decompile",
            str(Path(self._model_path).parent if self._model_path else self.app_root),
            "MDL files (*.mdl);;All files (*.*)",
        )
        if not path:
            return
        mdlops = self._find_mdlops()
        if not mdlops:
            QtWidgets.QMessageBox.information(
                self,
                "MDLOps",
                "Set the MDLOps path first with MDLOps > Set MDLOps Path.",
            )
            return
        cmd = self._mdlops_command(mdlops, "-k1", "-d", path)
        self._run_mdlops(cmd, Path(path).parent)
    def _run_mdlops(self, cmd: list[str], cwd: Path):
        self._log(f"Running MDLOps: {' '.join(cmd)}")

        def _on_complete(result, cancelled=False):
            if cancelled or result is None:
                return
            if result.stdout:
                self._log(result.stdout.strip())
            if result.stderr:
                self._log(result.stderr.strip(), "warning")
            if result.returncode == 0:
                self._log("MDLOps operation complete.", "success")
            else:
                self._log(f"MDLOps exited with code {result.returncode}", "warning")

        def _on_error(exc):
            # FileNotFoundError / TimeoutExpired are expected, user-fixable
            # failures; log them and suppress the generic error dialog.
            if isinstance(exc, FileNotFoundError):
                self._log(
                    "'perl' was not found. Install Perl or use the Windows MDLOps exe.",
                    "error",
                )
                return True
            if isinstance(exc, subprocess.TimeoutExpired):
                self._log("MDLOps timed out.", "error")
                return True
            return False

        self._run_io_async(
            "Running MDLOps",
            _work_run_mdlops,
            cmd,
            cwd,
            on_complete=_on_complete,
            on_error=_on_error,
            error_category="mdlops_error",
        )
    def _port_current_model(self):
        model = self._require_model("Port Current Model")
        if model is None:
            return
        current = getattr(getattr(model, "game_version", ""), "name", str(getattr(model, "game_version", "K1"))).upper()
        target = "K2" if current == "K1" else "K1"
        if QtWidgets.QMessageBox.question(
            self,
            "Port Current Model",
            f"Port '{getattr(model, 'name', 'model')}' to {target} and load the ported copy?",
        ) != QtWidgets.QMessageBox.Yes:
            return
        try:
            from src.core.mdl.mdl_porter import CrossGamePorter

            ported = CrossGamePorter().port(model, target)
            self._set_model_internal(ported, self._model_path)
            self._log(f"Ported current model to {target}.", "success")
        except Exception as exc:
            self._log(f"Port error: {exc}", "error")
            show_exception(self, "port_error", exc, context="Porting current model")
    def _generate_module_files(self):
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select output directory for module files",
            str(self.app_root),
        )
        if not out_dir:
            return
        module_name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Generate Module Files",
            "Module resref:",
            text="mymodule",
        )
        if not ok or not module_name.strip():
            return
        mod = module_name.strip().lower()
        room = f"{mod}_r01"
        files = {
            f"{mod}.lyt": f"filedependancy {mod}\n\nbeginlayout\n  room 0 {room} 0.0 0.0 0.0\nendlayout\n",
            f"{mod}.vis": f"{room}\n",
            f"{mod}.are.txt": f"# Starter ARE template for {mod}\n# Import into a KotOR GFF tool and save as {mod}.are\n",
            f"{mod}.git.txt": f"# Starter GIT template for {mod}\n# Add creatures, doors, placeables, sounds, and triggers here.\n",
            f"{mod}.ifo.txt": f"# Starter IFO template for {mod}\nMod_ID = \"{mod}\"\nMod_Entry_Area = \"{room}\"\n",
            "README_module_starter.txt": (
                f"Starter module files for {mod}.\n"
                "Convert the .txt GFF templates into ARE/GIT/IFO with your preferred GFF tool.\n"
            ),
        }
        try:
            output = Path(out_dir)
            for name, text in files.items():
                (output / name).write_text(text, encoding="utf-8")
            self._last_module_output_dir = str(output)
            self._log(f"Generated starter module files for {mod} in {output}", "success")
        except Exception as exc:
            self._log(f"Module generation error: {exc}", "error")
            show_exception(self, "module_generation_error", exc, context="Generating starter module files")
    def _handle_module_action(self, action: str):
        if action in {"Generate Module Files", "Validate Module", "Open Output"}:
            if action == "Generate Module Files":
                self._generate_module_files()
            elif action == "Open Output":
                path = getattr(self, "_last_module_output_dir", "")
                if path:
                    QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))
                else:
                    self._log("Generate module files first, then Open Output.", "warning")
            else:
                self._log(f"{action} needs a generated/open module workspace first.", "warning")
            return
        if action in {"Port K1 to K2", "Port K2 to K1"}:
            self._port_current_model()
            return
        if action == "Open Blueprint":
            self._open_blueprint_editor_window()
            self.blueprint_panel.open_blueprint()
            return
        if action == "Save Blueprint":
            self._open_blueprint_editor_window()
            self.blueprint_panel.save_blueprint()
            return
        if action == "Send to GModular":
            self._ipc_notify_saved()
            return
        self._log(f"{action} is waiting for deeper Qt module-editor migration.", "warning")
