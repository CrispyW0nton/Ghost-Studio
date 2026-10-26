"""Responsive scene export presentation for the Level Editor."""
from __future__ import annotations

from copy import deepcopy
from PySide6 import QtCore, QtWidgets
from src.core.level.level_export_bridge import LevelExportBridge, LevelExportResult


class _ExportProgress(QtWidgets.QProgressDialog):
    def __init__(self, *args):
        super().__init__(*args)
        self.result = None

    @QtCore.Slot(object)
    def complete(self, result):
        self.result = result
        self.accept()

    def reject(self):
        # ExportJob promotion cannot be cancelled halfway through writing.
        if self.result is not None:
            super().reject()


class _LevelExportWorker(QtCore.QObject):
    finished = QtCore.Signal(object)

    def __init__(self, project, path, options, context):
        super().__init__()
        self.project, self.path, self.options, self.context = project, path, options, context

    @QtCore.Slot()
    def run(self):
        try:
            result = LevelExportBridge().export_scene(self.project, self.path, self.options, **self.context)
        except Exception as exc:
            result = LevelExportResult(code="export_failed", message=str(exc))
        self.finished.emit(result)


def run_level_scene_export(parent, project, path, options, **context):
    """Snapshot on the UI thread; assemble and write on a dedicated worker."""
    snapshot = deepcopy(project)
    progress = _ExportProgress("Exporting level geometry and textures…", "", 0, 0, parent)
    progress.setWindowTitle("Export Whole Level")
    progress.setCancelButton(None)
    progress.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
    progress.setWindowFlag(QtCore.Qt.WindowType.WindowCloseButtonHint, False)
    thread = QtCore.QThread()
    worker = _LevelExportWorker(snapshot, path, options, context)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(progress.complete, QtCore.Qt.ConnectionType.QueuedConnection)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    QtCore.QTimer.singleShot(0, thread.start)
    progress.exec()
    thread.quit()
    thread.wait()
    result = progress.result or LevelExportResult(code="export_failed", message="Export interrupted")
    progress.deleteLater()
    thread.deleteLater()
    return result
