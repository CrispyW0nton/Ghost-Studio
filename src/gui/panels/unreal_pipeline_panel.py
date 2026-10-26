"""Map Studio's creator-facing Unreal import, build and playtest surface."""
from __future__ import annotations
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from PySide6 import QtCore, QtGui, QtWidgets


class UnrealPipelinePanel(QtWidgets.QWidget):
    def __init__(self, window):
        super().__init__(window)
        self.editor = window
        self.setObjectName("mapStudioUnrealPipeline")
        self.setProperty("ghostLayoutId", "map_studio.unreal_pipeline")
        self.jobs = None
        self.project_directory = None
        self.future = None
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="unreal-playtest")
        self.layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Unreal → KOTOR II", self)
        self.layout.addWidget(title)
        intro = QtWidgets.QLabel("Send a saved level from Unreal, review its conversion, then build and playtest your editable Ghost map.", self)
        intro.setWordWrap(True)
        self.layout.addWidget(intro)
        self.connect_button = self._button("Connect an Unreal project…", self._connect_editor)
        source_group = QtWidgets.QGroupBox("1 · Source and conversion", self)
        source_layout = QtWidgets.QVBoxLayout(source_group)
        self.source = QtWidgets.QLineEdit(self)
        self.source.setPlaceholderText("Native snapshot or converted level.json")
        self.source.setAccessibleName("Unreal source snapshot")
        source_layout.addWidget(self.source)
        browse = QtWidgets.QPushButton("Browse source…", self)
        browse.clicked.connect(self._browse_source)
        source_layout.addWidget(browse)
        self.review_settings = QtWidgets.QCheckBox("Review conversion settings again", self)
        source_layout.addWidget(self.review_settings)
        self.import_button = QtWidgets.QPushButton("Import / update source", self)
        self.import_button.clicked.connect(self._import)
        source_layout.addWidget(self.import_button)
        policy = QtWidgets.QPushButton("Import with saved conversion settings…", self)
        policy.clicked.connect(self._import_policy)
        source_layout.addWidget(policy)
        self.layout.addWidget(source_group)
        build_group = QtWidgets.QGroupBox("2 · Build and validate", self)
        build_layout = QtWidgets.QVBoxLayout(build_group)
        self.build_button = QtWidgets.QPushButton("Build current map", self)
        self.build_button.clicked.connect(self._build)
        build_layout.addWidget(self.build_button)
        self.history = QtWidgets.QComboBox(self)
        self.history.setAccessibleName("Build history")
        self.history.currentIndexChanged.connect(self._show_job)
        build_layout.addWidget(self.history)
        self.progress = QtWidgets.QProgressBar(self)
        self.progress.setRange(0, 0)
        self.progress.hide()
        build_layout.addWidget(self.progress)
        self.status = QtWidgets.QLabel("Import a source level to begin.", self)
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        build_layout.addWidget(self.status)
        self.cancel_button = QtWidgets.QPushButton("Cancel build", self)
        self.cancel_button.clicked.connect(self._cancel)
        build_layout.addWidget(self.cancel_button)
        output = QtWidgets.QPushButton("Open build and recovery folder", self)
        output.clicked.connect(self._open_output)
        build_layout.addWidget(output)
        self.layout.addWidget(build_group)
        play_group = QtWidgets.QGroupBox("3 · Install and playtest", self)
        play_layout = QtWidgets.QVBoxLayout(play_group)
        self.profile_fields = {}
        for key, label, hint in (("game_root", "KOTOR II folder", "Folder containing swkotor2.exe"),
                ("launcher", "Patched launcher", "KOTOR II shortcut (.lnk) or KPatchLauncher.exe"),
                ("debugger_script", "Debugger (optional)", "kotor_debugger.py"),
                ("debugger_python", "Debugger Python", "Python executable; required only with debugger")):
            row = QtWidgets.QHBoxLayout()
            field = QtWidgets.QLineEdit(self)
            field.setPlaceholderText(hint)
            field.setAccessibleName(label)
            browse = QtWidgets.QPushButton("Browse…", self)
            browse.clicked.connect(lambda checked=False, key=key: self._browse_profile(key))
            play_layout.addWidget(QtWidgets.QLabel(label, self))
            row.addWidget(field, 1)
            row.addWidget(browse)
            play_layout.addLayout(row)
            self.profile_fields[key] = field
        note = QtWidgets.QLabel("Close the game before installing or restoring files. Previous resources are backed up. Launch uses your configured patched shortcut; load a clean save and use the warp command below.", self)
        note.setWordWrap(True)
        play_layout.addWidget(note)
        self.install_button = QtWidgets.QPushButton("Install build and launch KOTOR II", self)
        self.install_button.clicked.connect(self._install)
        play_layout.addWidget(self.install_button)
        self.rollback_button = QtWidgets.QPushButton("Restore files from this build's backup", self)
        self.rollback_button.clicked.connect(self._rollback)
        play_layout.addWidget(self.rollback_button)
        self.warp = QtWidgets.QLineEdit(self)
        self.warp.setReadOnly(True)
        self.warp.setAccessibleName("Warp command")
        play_layout.addWidget(self.warp)
        copy_button = QtWidgets.QPushButton("Copy warp command", self)
        copy_button.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(self.warp.text()))
        play_layout.addWidget(copy_button)
        self.checks = {}
        for key, label in (("loaded", "This build loaded in game"), ("spawn", "Spawn and facing are correct"),
                           ("traversal", "Walkmesh traversal is correct"), ("save_reload", "Save and reload succeeded")):
            check = QtWidgets.QCheckBox(label, self)
            check.toggled.connect(self._acceptance_dirty)
            play_layout.addWidget(check)
            self.checks[key] = check
        self.notes = QtWidgets.QLineEdit(self)
        self.notes.setPlaceholderText("Playtest notes")
        self.notes.textEdited.connect(self._acceptance_dirty)
        play_layout.addWidget(self.notes)
        self.record_button = QtWidgets.QPushButton("Record observed playtest results", self)
        self.record_button.clicked.connect(self._record)
        play_layout.addWidget(self.record_button)
        self.layout.addWidget(play_group)
        self.layout.addStretch()
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self._acceptance_edited = False
        self._shown_job = None
        self._tick()

    def _button(self, text, callback):
        button = QtWidgets.QPushButton(text, self)
        button.clicked.connect(callback)
        self.layout.addWidget(button)
        return button

    def apply_ghost_layout(self, layout):
        spacing = layout.spacing_value("panel_spacing", 8)
        self.layout.setSpacing(spacing)
        self.layout.setContentsMargins(spacing, spacing, spacing, spacing)

    def _error(self, error):
        QtWidgets.QMessageBox.warning(self, "Unreal pipeline needs attention", str(error))

    def _context(self):
        path = self.editor.project.path
        if not path:
            return False
        root = Path(path).resolve().parent
        if self.project_directory != root:
            from src.core.workflow.unreal_pipeline_jobs import UnrealPipelineJobs
            from src.io.unreal_pipeline_store import read_receipt
            self.project_directory = root
            self.jobs = UnrealPipelineJobs(root)
            profile_path = root / ".ghoststudio/pipeline-profile.json"
            profile = read_receipt(profile_path) if profile_path.exists() else {}
            for key, field in self.profile_fields.items():
                field.setText(profile.get(key, ""))
            if not self.source.text():
                self.source.setText(profile.get("source", ""))
            self._history()
        return True

    def _save_profile(self):
        from src.io.unreal_pipeline_store import write_receipt
        profile = {key: field.text().strip() for key, field in self.profile_fields.items()}
        profile["source"] = self.source.text().strip()
        write_receipt(self.project_directory / ".ghoststudio/pipeline-profile.json", profile)
        return profile

    def set_source(self, path):
        self.source.setText(str(path))
        self.status.setText("Source received. Import it to review objects, materials, walkmesh and spawn.")

    def _browse_source(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose Unreal source", self.source.text(), "Unreal snapshot / bundle (*.json)")
        if path:
            self.set_source(path)

    def _import_policy(self):
        policy, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load conversion settings", "", "Conversion policy (*.json)")
        if policy:
            self._import(policy_path=policy)

    def _import(self, checked=False, *, policy_path=None):
        from src.gui.qt_lib.dialogs.unreal_level_import_dialog import import_unreal_level
        if not self.source.text().strip():
            self._browse_source()
        if not self.source.text().strip():
            return
        if import_unreal_level(self.editor, self.source.text().strip(), review_settings=self.review_settings.isChecked(), policy_path=policy_path):
            self._context()
            self._save_profile()
            self.status.setText("Source applied. Review the viewport, then build the current map.")

    def _build(self):
        try:
            if not self._context():
                raise ValueError("Save the KMAP and import a source level first.")
            resources = self.editor.controller.authored_project_extra_resources()
            job_id = self.jobs.submit(copy.deepcopy(self.editor.project), resources)
            self._history(job_id)
        except Exception as error:
            self._error(error)

    def _history(self, selected=None):
        selected = selected or self.history.currentData()
        self.history.blockSignals(True)
        self.history.clear()
        for receipt in self.jobs.history():
            self.history.addItem(f"{receipt.get('module', 'Map')} · {receipt['job_id'][:8]}", receipt["job_id"])
        index = self.history.findData(selected)
        self.history.setCurrentIndex(max(0, index))
        self.history.blockSignals(False)
        self._show_job()

    def _show_job(self):
        job_id = self.history.currentData()
        if not self.jobs or not job_id:
            return
        receipt = self.jobs.status(job_id)
        self.status.setText(receipt["state"].replace("_", " ").capitalize() + "\n" + receipt["message"] +
                            ("\n" + "\n".join(receipt.get("warnings", [])) if receipt.get("warnings") else ""))
        self.warp.setText(receipt.get("warp_command", ""))
        if self._shown_job != job_id:
            self._shown_job = job_id
            for key, check in self.checks.items():
                check.setChecked(receipt.get("acceptance", {}).get(key, False))
            self.notes.setText(receipt.get("acceptance_notes", ""))
            self._acceptance_edited = False
        busy = receipt["state"] in {"queued", "building", "validating", "installing"}
        self.progress.setVisible(busy or self.future is not None)
        self.cancel_button.setEnabled(receipt["state"] in {"queued", "building", "validating"})
        self.install_button.setEnabled(not self.future and receipt["state"] in {"ready", "waiting_for_game_close", "installed", "launch_requested"})
        self.rollback_button.setEnabled(not self.future and receipt.get("installation", {}).get("state") in {"installed", "installing", "recovery_required"})
        self.record_button.setEnabled(not self.future and receipt["state"] in {"installed", "launch_requested"})

    def _tick(self):
        try:
            self._context()
            if self.future and self.future.done():
                future, self.future = self.future, None
                try:
                    future.result()
                except Exception as error:
                    self._error(error)
            imported = bool(self.editor.project.extra_sections.get("unreal_source"))
            self.build_button.setEnabled(imported and not self.future)
            if self.history.currentData():
                self._show_job()
            else:
                for button in (self.cancel_button, self.install_button, self.rollback_button, self.record_button):
                    button.setEnabled(False)
        except Exception as error:
            self.status.setText(str(error))

    def _cancel(self):
        try:
            self.jobs.cancel(self.history.currentData())
        except Exception as error:
            self._error(error)

    def _browse_profile(self, key):
        if key == "game_root":
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "KOTOR II installation")
        else:
            filter = "Python (*.py)" if key == "debugger_script" else "Application / shortcut (*.exe *.lnk)"
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose " + key.replace("_", " "), "", filter)
        if path:
            self.profile_fields[key].setText(path)

    def _install(self):
        try:
            from src.core.workflow.unreal_map_import import project_revision
            profile = self._save_profile()
            revision = project_revision(self.editor.project)
            self.future = self.worker.submit(self.jobs.install_and_play, self.history.currentData(), profile, revision)
            self._show_job()
        except Exception as error:
            self._error(error)

    def _rollback(self):
        self.future = self.worker.submit(self.jobs.rollback, self.history.currentData())
        self._show_job()

    def _open_output(self):
        if self.jobs and self.history.currentData():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.jobs.root / self.history.currentData())))

    def _acceptance_dirty(self):
        self._acceptance_edited = True

    def _record(self):
        try:
            self.jobs.record_acceptance(self.history.currentData(), {k:c.isChecked() for k,c in self.checks.items()}, self.notes.text())
            self._acceptance_edited = False
            self.editor.statusBar().showMessage("Playtest observations saved for this exact build.", 7000)
        except Exception as error:
            self._error(error)

    def _connect_editor(self):
        project, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Connect an Unreal project", "", "Unreal project (*.uproject)")
        if not project:
            return
        try:
            from src.io.unreal_editor_connection import connect_editor
            executable = Path(sys.executable)
            if executable.name.lower() != "ghoststudio.exe":
                path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose GhostStudio.exe", "", "Ghost Studio (GhostStudio.exe)")
                if not path:
                    return
                executable = Path(path)
            candidates = [executable.parent / "UnrealPlugin/GhostStudioExporter"]
            candidates.extend(parent / "native/GhostRigger.Core.Unreal/EditorPlugin/GhostStudioExporter" for parent in executable.parents)
            plugin = next((p for p in candidates if (p / "GhostStudioExporter.uplugin").is_file()), None)
            if plugin is None:
                raise ValueError("Editor plugin is missing from this Ghost Studio installation.")
            result = connect_editor(project, plugin, executable)
            self.status.setText(result["message"] + "\nSetup backup: " + result["backup"])
        except Exception as error:
            self._error(error)
