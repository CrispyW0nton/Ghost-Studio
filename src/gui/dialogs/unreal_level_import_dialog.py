"""Map Studio presentation for native static imports and explicit conflicts."""
from __future__ import annotations
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile

from PySide6 import QtCore, QtWidgets

_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="unreal-import")


def _run(parent, label, operation):
    """Keep events responsive. Cancellation discards results after safe IO ends."""
    progress = QtWidgets.QProgressDialog(label, "Cancel", 0, 0, parent)
    progress.setObjectName("mapStudioUnrealImportProgress")
    progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
    progress.setMinimumDuration(0)
    progress.show()
    future = _WORKER.submit(operation)
    loop, timer = QtCore.QEventLoop(parent), QtCore.QTimer(parent)
    timer.setInterval(50)
    timer.timeout.connect(lambda: loop.quit() if future.done() else None)
    timer.start()
    loop.exec()
    timer.stop()
    cancelled = progress.wasCanceled()
    progress.close()
    progress.deleteLater()
    timer.deleteLater()
    loop.deleteLater()
    if cancelled:
        return None
    return future.result()


def _review(parent, preview):
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Review source update")
    layout = QtWidgets.QVBoxLayout(dialog)
    label = QtWidgets.QLabel(f"{len(preview.component_labels)} source objects will be applied as one undoable edit.", dialog)
    label.setWordWrap(True)
    layout.addWidget(label)
    details = QtWidgets.QPlainTextEdit(dialog)
    details.setReadOnly(True)
    details.setPlainText("\n".join(preview.component_labels + preview.warnings))
    layout.addWidget(details)
    choices = {}
    if preview.conflicts:
        text = QtWidgets.QLabel("Both Unreal and Ghost changed these values. Choose which edit to keep.", dialog)
        text.setWordWrap(True)
        layout.addWidget(text)
        table = QtWidgets.QTableWidget(len(preview.conflicts), 2, dialog)
        table.setHorizontalHeaderLabels(["Changed value", "Resolution"])
        table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for row, conflict in enumerate(preview.conflicts):
            item = QtWidgets.QTableWidgetItem(conflict.replace("/", " → "))
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 0, item)
            choice = QtWidgets.QComboBox(table)
            choice.addItem("Choose…", None)
            choice.addItem("Keep Ghost edit", "keep_local")
            choice.addItem("Accept Unreal update", "accept_source")
            table.setCellWidget(row, 1, choice)
            choices[conflict] = choice
        layout.addWidget(table)
    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel, dialog)
    apply = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    apply.setText("Apply source update")
    def ready():
        apply.setEnabled(all(c.currentData() for c in choices.values()))
    for choice in choices.values():
        choice.currentIndexChanged.connect(ready)
    ready()
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    return {key: c.currentData() for key,c in choices.items()} if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted else None


def import_unreal_level(window, path=None, *, review_settings=False, policy_path=None):
    from src.io.unreal_native_snapshot import load_native_snapshot, convert_native_snapshot
    from src.io.unreal_editor_connection import capture_conversion_profile, restore_conversion_profile
    from src.core.workflow.unreal_map_import import preview_unreal_import, apply_unreal_import
    from src.gui.qt_lib.dialogs.unreal_conversion_dialog import UnrealConversionDialog
    if not window.project.path:
        window.save_kmap_as()
        if not window.project.path:
            return False
    if not path:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(window, "Import / Update Unreal Level", "", "Unreal snapshot or conversion bundle (*.json)")
    if not path:
        return False
    try:
        with tempfile.TemporaryDirectory(prefix="ghost-unreal-import-") as temporary:
            source = Path(path)
            from src.io.unreal_level_bundle import MAX_JSON_BYTES, _json_object
            if source.stat().st_size > MAX_JSON_BYTES:
                raise ValueError("Import manifest exceeds the size limit.")
            kind = _json_object(source.read_bytes()).get("schema")
            binding = None
            if kind == "ghoststudio.unreal-snapshot/v1":
                snapshot = _run(window, "Verifying native source and dependencies…", lambda: load_native_snapshot(source))
                if snapshot is None:
                    return False
                root = Path(window.project.path).parent
                saved = window.project.extra_sections.get("unreal_source", {}).get("conversion_profile")
                if policy_path:
                    policy = policy_path
                elif saved and not review_settings:
                    policy = restore_conversion_profile(root, saved, snapshot, Path(temporary) / "policy")
                else:
                    defaults = None
                    if saved:
                        from src.io.unreal_project_assets import UnrealProjectAssets
                        defaults = UnrealProjectAssets(root).read_json(saved)["policy"]
                    dialog = UnrealConversionDialog(snapshot, defaults, window)
                    while True:
                        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                            return False
                        try:
                            policy = dialog.write_policy(Path(temporary) / "policy")
                            break
                        except Exception as error:
                            dialog.error.setText(str(error))
                converted = _run(window, "Converting static geometry and materials…", lambda: convert_native_snapshot(source, policy, Path(temporary) / "bundle"))
                if converted is None:
                    return False
                binding = capture_conversion_profile(root, policy, snapshot)
                source = converted
            captured = copy.deepcopy(window.project)
            preview = _run(window, "Comparing Unreal changes and Ghost edits…", lambda: preview_unreal_import(captured, source))
            if preview is None:
                return False
            decisions = _review(window, preview)
            if decisions is None:
                return False
            updated = _run(window, "Validating the source update…", lambda: apply_unreal_import(captured, preview, decisions))
            if updated is None:
                return False
            if binding:
                updated.extra_sections["unreal_source"]["conversion_profile"] = binding
            window.controller.commit_unreal_import(preview, updated)
            window._refresh_all("Unreal source imported. Review spawn and walkmesh before building.")
            return True
    except Exception as error:
        QtWidgets.QMessageBox.warning(window, "Unreal import needs attention", str(error))
        return False
