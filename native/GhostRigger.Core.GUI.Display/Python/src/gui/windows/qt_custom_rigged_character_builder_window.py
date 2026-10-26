"""Guided Qt workbench for converting a foreign rig into an Odyssey creature."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from src.core.project.custom_rigged_character_project import (
    AnimationMapping,
    CreatureSoundCue,
    CustomAnimationRegistration,
    CustomRiggedCharacterProject,
    MaterialAssignment,
    SourceAsset,
    load_custom_rigged_character_project,
    save_custom_rigged_character_project,
    sha256_file,
)
from src.core.rendering.renderer_settings import RendererSettings
from src.core.rendering.viewport_navigation import DEFAULT_VIEWPORT_NAVIGATION_PROFILE
from src.gui.qt_lib.viewports.qt_viewport import QtMainViewportWidget
from src.gui.panels.qt_secondary_motion_editor import QtSecondaryMotionEditor
from src.core.characters.custom_rigged_character_build_service import (
    allocate_animation_id,
    namespaced_animation_name,
    suggest_semantic_mapping,
)
from src.core.characters.custom_rigged_character_behavior_service import (
    CREATURE_SOUND_CUE_DEFINITIONS,
    UTC_SCRIPT_HOOK_LABELS,
    creature_sound_resref,
)
from src.resources.kotor_utc_template_catalog import UTC_SCRIPT_HOOK_FIELDS


WORKFLOW_PAGES = (
    ("source_assets", "Project and source assets"),
    ("rig_inspection", "Rig inspection"),
    ("scale_ground", "Scale, facing, pivot, and ground contact"),
    ("animation_library", "Animation library"),
    ("animation_preparation", "Animation preparation"),
    ("materials_uvs", "Materials, textures, and UVs"),
    ("secondary_motion", "Secondary Motion — Cloth & Hair"),
    ("gameplay", "KOTOR gameplay integration"),
    ("validate_build", "Validation and build"),
    ("install_test", "Install and test"),
)

RUNTIME_TEST_CHECKLIST = (
    "Load a save outside the test module, then enter or warp into it so KOTOR creates a fresh creature instance.",
    "Creature is visible.",
    "Creature is at the correct height.",
    "Textures are correctly wrapped.",
    "Idle plays.",
    "Walk plays while moving.",
    "Run plays while moving quickly.",
    "Turning does not distort the skeleton.",
    "No vertices remain behind or explode.",
    "The creature notices a hostile target and starts combat.",
    "A hostile creature has a red health bar; KOTOR II keeps its name text cyan by design.",
    "Attack and damage reactions run without script errors.",
    "Round-end and death behavior match the chosen template.",
    "The generated UTC can be spawned again after restarting the game.",
    "Reloading the module still works.",
    "Custom actions trigger when requested.",
)


class _PathRow(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(
        self,
        label: str,
        *,
        folder: bool = False,
        file_filter: str = "All files (*)",
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.folder = folder
        self.file_filter = file_filter
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        caption = QtWidgets.QLabel(label)
        caption.setMinimumWidth(145)
        self.edit = QtWidgets.QLineEdit()
        self.edit.setClearButtonEnabled(True)
        self.status = QtWidgets.QLabel("Not selected")
        self.status.setMinimumWidth(95)
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self.edit.textChanged.connect(self._refresh_status)
        layout.addWidget(caption)
        layout.addWidget(self.edit, 1)
        layout.addWidget(browse)
        layout.addWidget(self.status)

    def _browse(self) -> None:
        if self.folder:
            value = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose folder", self.edit.text())
        else:
            value, _selected = QtWidgets.QFileDialog.getOpenFileName(
                self, "Choose source file", self.edit.text(), self.file_filter
            )
        if value:
            self.edit.setText(value)

    def _refresh_status(self) -> None:
        text = self.edit.text().strip()
        if not text:
            self.status.setText("Not selected")
        else:
            path = Path(text)
            exists = path.is_dir() if self.folder else path.is_file()
            self.status.setText("Ready" if exists else "Not found")
        self.changed.emit()

    def path(self) -> str:
        return self.edit.text().strip()

    def set_path(self, value: str) -> None:
        self.edit.setText(str(value or ""))


class _Page(QtWidgets.QWidget):
    def __init__(self, title: str, help_text: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        heading = QtWidgets.QLabel(title)
        font = heading.font()
        font.setPointSize(font.pointSize() + 4)
        font.setBold(True)
        heading.setFont(font)
        self.layout.addWidget(heading)
        help_label = QtWidgets.QLabel(help_text)
        help_label.setWordWrap(True)
        self.layout.addWidget(help_label)


class _HumanScaleWidget(QtWidgets.QWidget):
    """Small theme-aware 1.8-unit reference silhouette for the ground page."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.creature_height = 0.0
        self.setMinimumWidth(125)
        self.setToolTip("Human reference: approximately 1.8 KOTOR model units")

    def set_creature_height(self, value: float) -> None:
        self.creature_height = max(0.0, float(value or 0.0))
        self.update()

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        color = self.palette().color(QtGui.QPalette.Text)
        muted = self.palette().color(QtGui.QPalette.Mid)
        painter.setPen(QtGui.QPen(muted, 1))
        floor = self.height() - 28
        painter.drawLine(10, floor, self.width() - 10, floor)
        usable = max(80, floor - 24)
        scale = usable / max(2.2, self.creature_height, 1.8)
        human_h = 1.8 * scale
        x = self.width() * 0.35
        top = floor - human_h
        painter.setPen(QtGui.QPen(color, 2))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(QtCore.QPointF(x, top + 8), 7, 7)
        painter.drawLine(QtCore.QPointF(x, top + 16), QtCore.QPointF(x, floor - 32))
        painter.drawLine(QtCore.QPointF(x, top + 28), QtCore.QPointF(x - 12, top + 55))
        painter.drawLine(QtCore.QPointF(x, top + 28), QtCore.QPointF(x + 12, top + 55))
        painter.drawLine(QtCore.QPointF(x, floor - 32), QtCore.QPointF(x - 10, floor))
        painter.drawLine(QtCore.QPointF(x, floor - 32), QtCore.QPointF(x + 10, floor))
        painter.drawText(8, self.height() - 8, "Human 1.8")
        if self.creature_height > 0:
            creature_x = self.width() * 0.72
            creature_top = floor - self.creature_height * scale
            painter.setPen(QtGui.QPen(self.palette().color(QtGui.QPalette.Highlight), 3))
            painter.drawLine(QtCore.QPointF(creature_x, creature_top), QtCore.QPointF(creature_x, floor))
            painter.drawText(int(creature_x - 28), max(12, int(creature_top - 4)), f"{self.creature_height:.2f}")


class QtCustomRiggedCharacterBuilderWindow(QtWidgets.QMainWindow):
    """Independent nine-step authoring window with beginner-first language."""

    nativeBuilderRequested = QtCore.Signal()
    projectChanged = QtCore.Signal(object)
    importRequested = QtCore.Signal(object)
    validateRequested = QtCore.Signal(object)
    buildRequested = QtCore.Signal(object)
    animationPreviewRequested = QtCore.Signal(str)
    previewInstallRequested = QtCore.Signal(object)
    installRequested = QtCore.Signal(object)
    restoreRequested = QtCore.Signal(object)
    launchPatchManagerRequested = QtCore.Signal(object)
    openBuildFolderRequested = QtCore.Signal(str)
    behaviorCatalogRequested = QtCore.Signal(object)
    behaviorTemplateRequested = QtCore.Signal(str)
    behaviorStarterRequested = QtCore.Signal(str)
    behaviorHookApplyRequested = QtCore.Signal(object)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Ghost Studio — Custom Rigged Character Builder")
        self.setObjectName("customRiggedCharacterBuilderWindow")
        self.setProperty("ghostLayoutId", "customRiggedCharacterBuilder")
        self.setMinimumSize(1180, 760)
        self.setAcceptDrops(True)
        self.project = CustomRiggedCharacterProject()
        self.project_path: Path | None = None
        self._loading = False
        self._preview_model: object | None = None
        self._placement_snapshot: object | None = None
        self._material_snapshots: list[object] = []
        self._animation_inventory: list[dict[str, Any]] = []
        self._install_preview_id = ""
        self._settings = QtCore.QSettings("GhostStudio", "CustomRiggedCharacterBuilder")
        self._pages: dict[str, QtWidgets.QWidget] = {}
        self._autosave = QtCore.QTimer(self)
        self._autosave.setSingleShot(True)
        self._autosave.setInterval(1200)
        self._autosave.timeout.connect(self._save_automatically)
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()
        self._load_project_into_form()

    def _build_toolbar(self) -> None:
        toolbar = self.addToolBar("Custom character project")
        toolbar.setObjectName("customCharacterToolbar")
        toolbar.addWidget(QtWidgets.QLabel("Character type"))
        self.character_type = QtWidgets.QComboBox()
        self.character_type.addItem("Custom Rigged Character", "custom_rigged_character")
        self.character_type.addItem("Native KOTOR Character", "native_kotor_character")
        self.character_type.currentIndexChanged.connect(self._character_type_changed)
        toolbar.addWidget(self.character_type)
        toolbar.addSeparator()
        new_action = toolbar.addAction("New")
        open_action = toolbar.addAction("Open…")
        save_action = toolbar.addAction("Save")
        save_as_action = toolbar.addAction("Save As…")
        new_action.triggered.connect(self.new_project)
        open_action.triggered.connect(self.open_project)
        save_action.triggered.connect(self.save_project)
        save_as_action.triggered.connect(lambda: self.save_project(save_as=True))

    def _build_central(self) -> None:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setObjectName("customCharacterMainSplitter")
        splitter.setProperty("ghostLayoutId", "customRiggedCharacter.mainSplitter")
        self.step_list = QtWidgets.QListWidget()
        self.step_list.setObjectName("customCharacterWorkflowSteps")
        self.step_list.setMinimumWidth(325)
        for index, (key, label) in enumerate(WORKFLOW_PAGES, start=1):
            item = QtWidgets.QListWidgetItem(f"{index}. {label}")
            item.setData(QtCore.Qt.UserRole, key)
            self.step_list.addItem(item)
        self.stack = QtWidgets.QStackedWidget()
        builders: dict[str, Callable[[], QtWidgets.QWidget]] = {
            "source_assets": self._source_assets_page,
            "rig_inspection": self._rig_inspection_page,
            "scale_ground": self._scale_ground_page,
            "animation_library": self._animation_library_page,
            "animation_preparation": self._animation_preparation_page,
            "materials_uvs": self._materials_page,
            "secondary_motion": self._secondary_motion_page,
            "gameplay": self._gameplay_page,
            "validate_build": self._validation_build_page,
            "install_test": self._install_test_page,
        }
        for key, _label in WORKFLOW_PAGES:
            page = builders[key]()
            page.setProperty("ghostLayoutId", f"customRiggedCharacter.{key}")
            self._pages[key] = page
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            scroll.setWidget(page)
            self.stack.addWidget(scroll)
        self.step_list.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.step_list.setCurrentRow(0)
        splitter.addWidget(self.step_list)
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([335, 835])
        self.setCentralWidget(splitter)

    def _build_statusbar(self) -> None:
        self.project_status = QtWidgets.QLabel("New project — source files stay read-only")
        self.statusBar().addWidget(self.project_status, 1)
        self.validation_status = QtWidgets.QLabel("Not validated")
        self.statusBar().addPermanentWidget(self.validation_status)

    def _source_assets_page(self) -> QtWidgets.QWidget:
        page = _Page(
            "Project and source assets",
            "Choose the creature, its source files, and where Ghost Studio may create converted copies. Your FBX and textures are never rewritten.",
        )
        self.game_directory = _PathRow("KOTOR game folder", folder=True)
        self.game_directory.changed.connect(self._form_changed)
        page.layout.addWidget(self.game_directory)
        form = QtWidgets.QFormLayout()
        self.creature_name = QtWidgets.QLineEdit()
        self.resource_name = QtWidgets.QLineEdit()
        self.resource_name.setMaxLength(16)
        self.resource_name.setPlaceholderText("1–16 letters, numbers, or underscores")
        self.target_game = QtWidgets.QComboBox()
        self.target_game.addItem("KOTOR II", "K2")
        self.target_game.addItem("KOTOR I", "K1")
        self.skeleton_root = QtWidgets.QComboBox()
        self.skeleton_root.setPlaceholderText("Detected after import")
        self.skeleton_root.setEnabled(False)
        form.addRow("Creature name", self.creature_name)
        form.addRow("KOTOR resource name", self.resource_name)
        form.addRow("Target game", self.target_game)
        form.addRow("Deform hierarchy", self.skeleton_root)
        page.layout.addLayout(form)
        self.source_fbx = _PathRow("Source FBX", file_filter="FBX files (*.fbx)")
        self.animation_folder = _PathRow("Animation FBX folder", folder=True)
        self.texture_folder = _PathRow("Texture folder", folder=True)
        self.output_folder = _PathRow("Output project folder", folder=True)
        for row in (self.source_fbx, self.animation_folder, self.texture_folder, self.output_folder):
            row.changed.connect(self._form_changed)
            page.layout.addWidget(row)
        external_group = QtWidgets.QGroupBox("Optional external animation FBX files")
        external_layout = QtWidgets.QVBoxLayout(external_group)
        self.external_animation_list = QtWidgets.QListWidget()
        self.external_animation_list.setAlternatingRowColors(True)
        external_layout.addWidget(self.external_animation_list)
        external_actions = QtWidgets.QHBoxLayout()
        add_external = QtWidgets.QPushButton("Add animation FBX…")
        remove_external = QtWidgets.QPushButton("Remove selected")
        add_external.clicked.connect(self._add_external_animation_files)
        remove_external.clicked.connect(self._remove_external_animation_files)
        external_actions.addWidget(add_external)
        external_actions.addWidget(remove_external)
        external_actions.addStretch(1)
        external_layout.addLayout(external_actions)
        page.layout.addWidget(external_group)
        buttons = QtWidgets.QHBoxLayout()
        self.import_button = QtWidgets.QPushButton("Import and inspect")
        self.import_button.clicked.connect(self._request_import)
        buttons.addStretch(1)
        buttons.addWidget(self.import_button)
        page.layout.addLayout(buttons)
        summary_group = QtWidgets.QGroupBox("Import summary")
        summary_layout = QtWidgets.QFormLayout(summary_group)
        self.import_summary_labels: dict[str, QtWidgets.QLabel] = {}
        for key, label in (
            ("mesh_count", "Meshes"),
            ("root_name", "Skeleton root"),
            ("bone_count", "Bones"),
            ("skinned_vertex_count", "Skinned vertices"),
            ("animation_count", "Animations"),
            ("texture_count", "Textures"),
            ("attention_count", "Items needing attention"),
        ):
            value = QtWidgets.QLabel("—")
            self.import_summary_labels[key] = value
            summary_layout.addRow(label, value)
        page.layout.addWidget(summary_group)
        page.layout.addStretch(1)
        for widget in (self.creature_name, self.resource_name):
            widget.textChanged.connect(self._form_changed)
        self.target_game.currentIndexChanged.connect(self._form_changed)
        self.skeleton_root.currentIndexChanged.connect(self._skeleton_root_changed)
        return page

    def _preview_shell(self, help_text: str) -> tuple[QtWidgets.QGroupBox, QtWidgets.QLabel]:
        group = QtWidgets.QGroupBox("Preview")
        layout = QtWidgets.QVBoxLayout(group)
        preview = QtWidgets.QLabel(help_text)
        preview.setAlignment(QtCore.Qt.AlignCenter)
        preview.setMinimumHeight(290)
        preview.setFrameShape(QtWidgets.QFrame.StyledPanel)
        preview.setWordWrap(True)
        layout.addWidget(preview, 1)
        controls = QtWidgets.QHBoxLayout()
        view_buttons: dict[str, QtWidgets.QPushButton] = {}
        for label in ("Perspective", "Front", "Side", "Back", "Frame selected", "Wireframe"):
            button = QtWidgets.QPushButton(label)
            button.setCheckable(label == "Wireframe")
            controls.addWidget(button)
            view_buttons[label] = button
        controls.addStretch(1)
        layout.addLayout(controls)
        setattr(group, "_view_buttons", view_buttons)
        return group, preview

    @staticmethod
    def _wire_view_controls(group: QtWidgets.QGroupBox, viewport: QtMainViewportWidget) -> None:
        buttons = dict(getattr(group, "_view_buttons", {}) or {})
        connections = {
            "Perspective": viewport.set_view_perspective,
            "Front": viewport.set_view_front,
            "Side": viewport.set_view_right,
            "Back": viewport.set_view_back,
            "Frame selected": viewport.reset_camera,
        }
        for label, callback in connections.items():
            if label in buttons:
                buttons[label].clicked.connect(lambda _checked=False, fn=callback: fn())
        if "Wireframe" in buttons:
            buttons["Wireframe"].toggled.connect(viewport.toggle_wireframe)

    def _rig_inspection_page(self) -> QtWidgets.QWidget:
        page = _Page(
            "Rig inspection",
            "Inspect the deform skeleton Ghost Studio will export. Errors stop the build; warnings explain risks; information records useful details.",
        )
        preview, self.rig_preview = self._preview_shell(
            "Import a creature to see its mesh, skeleton, bind pose, floor grid, and world axes."
        )
        preview.layout().removeWidget(self.rig_preview)
        self.rig_preview.deleteLater()
        self.rig_viewport = QtMainViewportWidget(self, map_studio_authoring_chrome=False)
        self.rig_viewport.setObjectName("customRiggedCharacterRigViewport")
        self.rig_viewport.set_renderer_settings(
            RendererSettings.from_settings(getattr(self.parent(), "settings_data", {}) or {})
        )
        self.rig_viewport.set_navigation_profile(DEFAULT_VIEWPORT_NAVIGATION_PROFILE)
        preview.layout().insertWidget(0, self.rig_viewport, 1)
        self._wire_view_controls(preview, self.rig_viewport)
        toggles = QtWidgets.QHBoxLayout()
        self.rig_toggles: dict[str, QtWidgets.QCheckBox] = {}
        for label in ("Textures", "Skeleton overlay", "Bind pose", "Floor grid", "Skin weights"):
            box = QtWidgets.QCheckBox(label)
            box.setChecked(label in {"Textures", "Floor grid"})
            toggles.addWidget(box)
            self.rig_toggles[label] = box
        self.rig_toggles["Textures"].toggled.connect(self.rig_viewport.toggle_texture)
        self.rig_toggles["Skeleton overlay"].toggled.connect(self.rig_viewport.toggle_bones)
        self.rig_toggles["Floor grid"].toggled.connect(self.rig_viewport.toggle_grid)
        self.rig_toggles["Skin weights"].toggled.connect(self.rig_viewport.toggle_weight_heatmap)
        self.rig_toggles["Bind pose"].toggled.connect(
            lambda checked: self.rig_viewport.clear_animation_pose() if checked else None
        )
        preview.layout().addLayout(toggles)
        page.layout.addWidget(preview)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.hierarchy_tree = QtWidgets.QTreeWidget()
        self.hierarchy_tree.setHeaderLabels(["Bone / node", "Export"])
        self.rig_issues = QtWidgets.QTreeWidget()
        self.rig_issues.setHeaderLabels(["Level", "What needs attention", "Can Ghost Studio fix it?"])
        split.addWidget(self.hierarchy_tree)
        split.addWidget(self.rig_issues)
        page.layout.addWidget(split, 1)
        repair = QtWidgets.QGroupBox("Safe automatic repairs")
        repair_layout = QtWidgets.QGridLayout(repair)
        labels = (
            "Bake constraints and IK",
            "Normalize transforms",
            "Remove unused control bones",
            "Use one selected deform hierarchy",
            "Limit and normalize skin influences",
            "Repair clear bind-matrix omissions",
            "Reorient to KOTOR axes",
        )
        self.repair_controls: dict[str, QtWidgets.QCheckBox] = {}
        repair_keys = (
            "bake_constraints", "normalize_transforms", "remove_unused_controls",
            "select_one_hierarchy", "limit_influences", "repair_bind_matrices", "reorient_axes",
        )
        for index, (key, label) in enumerate(zip(repair_keys, labels)):
            box = QtWidgets.QCheckBox(label)
            box.toggled.connect(self._repair_setting_changed)
            repair_layout.addWidget(box, index // 2, index % 2)
            self.repair_controls[key] = box
        page.layout.addWidget(repair)
        return page

    def _scale_ground_page(self) -> QtWidgets.QWidget:
        page = _Page(
            "Scale, facing, pivot, and ground contact",
            "Place the creature at a believable size and make its feet, claws, hooves, or chosen contact points meet KOTOR's floor.",
        )
        preview, self.ground_preview = self._preview_shell(
            "The exported root pivot, contact points, ground plane, and a human-size comparison silhouette appear here."
        )
        preview.layout().removeWidget(self.ground_preview)
        self.ground_preview.deleteLater()
        self.ground_viewport = QtMainViewportWidget(self, map_studio_authoring_chrome=False)
        self.ground_viewport.setObjectName("customRiggedCharacterGroundViewport")
        self.ground_viewport.set_renderer_settings(
            RendererSettings.from_settings(getattr(self.parent(), "settings_data", {}) or {})
        )
        self.ground_viewport.set_navigation_profile(DEFAULT_VIEWPORT_NAVIGATION_PROFILE)
        self._wire_view_controls(preview, self.ground_viewport)
        visual_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        visual_split.addWidget(self.ground_viewport)
        self.human_scale = _HumanScaleWidget()
        visual_split.addWidget(self.human_scale)
        visual_split.setStretchFactor(0, 1)
        visual_split.setStretchFactor(1, 0)
        preview.layout().insertWidget(0, visual_split, 1)
        page.layout.addWidget(preview)
        form = QtWidgets.QFormLayout()
        self.height_display = QtWidgets.QLabel("Not measured")
        self.runtime_height_display = QtWidgets.QLabel("Not detected")
        self.runtime_height_display.setWordWrap(True)
        self.global_scale = QtWidgets.QDoubleSpinBox()
        self.global_scale.setRange(0.0001, 10000.0)
        self.global_scale.setDecimals(5)
        self.global_scale.setValue(1.0)
        self.ground_offset = QtWidgets.QDoubleSpinBox()
        self.ground_offset.setRange(-10000.0, 10000.0)
        self.ground_offset.setDecimals(5)
        self.facing_preset = QtWidgets.QComboBox()
        for label, degrees in (
            ("Detect automatically", None), ("+Y forward", 0.0), ("-Y forward", 180.0),
            ("+X forward", -90.0), ("-X forward", 90.0),
        ):
            self.facing_preset.addItem(label, degrees)
        self.pivot_x = QtWidgets.QDoubleSpinBox()
        self.pivot_y = QtWidgets.QDoubleSpinBox()
        self.pivot_z = QtWidgets.QDoubleSpinBox()
        for spin in (self.pivot_x, self.pivot_y, self.pivot_z):
            spin.setRange(-10000.0, 10000.0)
            spin.setDecimals(5)
        pivot_row = QtWidgets.QWidget()
        pivot_layout = QtWidgets.QHBoxLayout(pivot_row)
        pivot_layout.setContentsMargins(0, 0, 0, 0)
        for label, spin in (("X", self.pivot_x), ("Y", self.pivot_y), ("Z", self.pivot_z)):
            pivot_layout.addWidget(QtWidgets.QLabel(label))
            pivot_layout.addWidget(spin)
        self.contact_nodes = QtWidgets.QLineEdit()
        self.contact_nodes.setPlaceholderText("Example: foot_l, foot_r, claw_front")
        form.addRow("Height in KOTOR model units", self.height_display)
        form.addRow("Automatic KOTOR height correction", self.runtime_height_display)
        form.addRow("Global scale", self.global_scale)
        form.addRow("Manual vertical offset", self.ground_offset)
        form.addRow("Forward direction", self.facing_preset)
        form.addRow("Exported root pivot offset", pivot_row)
        form.addRow("Ground contact bones or points", self.contact_nodes)
        page.layout.addLayout(form)
        actions = QtWidgets.QHBoxLayout()
        detect_contacts = QtWidgets.QPushButton("Detect lowest contacts")
        place_ground = QtWidgets.QPushButton("Place contacts on ground")
        preview_pivot = QtWidgets.QPushButton("Preview exported root pivot")
        detect_contacts.clicked.connect(self._detect_lowest_contacts)
        place_ground.clicked.connect(self._place_contacts_on_ground)
        preview_pivot.clicked.connect(lambda: self.ground_viewport.toggle_dummy_helpers(True))
        for button in (detect_contacts, place_ground, preview_pivot):
            actions.addWidget(button)
        actions.addStretch(1)
        page.layout.addLayout(actions)
        self.ground_warning = QtWidgets.QLabel("No ground analysis yet.")
        self.ground_warning.setWordWrap(True)
        page.layout.addWidget(self.ground_warning)
        self.global_scale.valueChanged.connect(self._form_changed)
        self.ground_offset.valueChanged.connect(self._form_changed)
        self.facing_preset.currentIndexChanged.connect(self._form_changed)
        for spin in (self.pivot_x, self.pivot_y, self.pivot_z):
            spin.valueChanged.connect(self._form_changed)
        self.contact_nodes.textChanged.connect(self._form_changed)
        return page

    def _animation_library_page(self) -> QtWidgets.QWidget:
        page = _Page(
            "Animation library",
            "Preview each action and decide whether KOTOR already knows when to request it, the Custom Animation Patch must register it, or it should stay unassigned.",
        )
        explanation = QtWidgets.QLabel(
            "Vanilla behavior alias: use a name such as cpause1, cwalk, or crun that KOTOR already requests.\n"
            "Custom runtime animation: register an additive namespaced action and ID; gameplay must still request it.\n"
            "Unassigned: keep the source action in this project without exporting it."
        )
        explanation.setWordWrap(True)
        page.layout.addWidget(explanation)
        self.animation_table = QtWidgets.QTableWidget(0, 11)
        self.animation_table.setHorizontalHeaderLabels((
            "Source action", "Duration", "Frames / rate", "Loop", "Root motion",
            "Animated bones", "Preview", "KOTOR behavior", "Exported name", "Custom ID", "Status",
        ))
        self.animation_table.horizontalHeader().setStretchLastSection(True)
        self.animation_table.currentCellChanged.connect(self._animation_row_selected)
        page.layout.addWidget(self.animation_table, 1)
        actions = QtWidgets.QHBoxLayout()
        suggest = QtWidgets.QPushButton("Suggest mappings")
        add_external = QtWidgets.QPushButton("Add external animation FBX…")
        suggest.clicked.connect(self._confirm_suggested_mappings)
        add_external.clicked.connect(self._add_external_animation_files)
        actions.addWidget(suggest)
        actions.addWidget(add_external)
        actions.addStretch(1)
        page.layout.addLayout(actions)
        return page

    def _animation_preparation_page(self) -> QtWidgets.QWidget:
        page = _Page(
            "Animation preparation",
            "Trim, loop, and retime the selected action, then compare the source preview with the converted KOTOR preview.",
        )
        previews = QtWidgets.QHBoxLayout()
        self.animation_preview_labels: dict[str, QtWidgets.QLabel] = {}
        self.animation_preview_viewports: dict[str, QtMainViewportWidget] = {}
        for key, title in (("before", "Before conversion"), ("after", "After conversion")):
            group = QtWidgets.QGroupBox(title)
            layout = QtWidgets.QVBoxLayout(group)
            view = QtMainViewportWidget(self, map_studio_authoring_chrome=False)
            view.setObjectName(f"customRiggedCharacterAnimation{key.title()}Viewport")
            view.setMinimumHeight(220)
            view.set_renderer_settings(
                RendererSettings.from_settings(getattr(self.parent(), "settings_data", {}) or {})
            )
            view.set_navigation_profile(DEFAULT_VIEWPORT_NAVIGATION_PROFILE)
            status = QtWidgets.QLabel("Choose an animation")
            status.setAlignment(QtCore.Qt.AlignCenter)
            status.setWordWrap(True)
            layout.addWidget(view, 1)
            layout.addWidget(status)
            previews.addWidget(group)
            self.animation_preview_viewports[key] = view
            self.animation_preview_labels[key] = status
        page.layout.addLayout(previews)
        form = QtWidgets.QFormLayout()
        self.loop_trim = QtWidgets.QCheckBox("Loop this action")
        self.playback_speed = QtWidgets.QDoubleSpinBox()
        self.playback_speed.setRange(0.01, 10.0)
        self.playback_speed.setValue(1.0)
        self.trim_start = QtWidgets.QDoubleSpinBox()
        self.trim_end = QtWidgets.QDoubleSpinBox()
        for spin in (self.trim_start, self.trim_end):
            spin.setRange(0.0, 3600.0)
            spin.setDecimals(4)
        self.retime_duration = QtWidgets.QDoubleSpinBox()
        self.retime_duration.setRange(0.0, 3600.0)
        self.retime_duration.setDecimals(4)
        self.retime_duration.setSpecialValueText("Keep source duration")
        self.root_motion = QtWidgets.QComboBox()
        self.root_motion.addItems(("Convert to in-place", "Keep source root motion", "Extract for analysis only"))
        self.bake_rate = QtWidgets.QComboBox()
        self.bake_rate.addItems(("30 samples/second", "60 samples/second", "Source rate"))
        form.addRow("Loop trim", self.loop_trim)
        form.addRow("Playback speed", self.playback_speed)
        form.addRow("Start trim (seconds)", self.trim_start)
        form.addRow("End trim (seconds; 0 = full clip)", self.trim_end)
        form.addRow("Retime duration (0 = use speed)", self.retime_duration)
        form.addRow("Root motion", self.root_motion)
        form.addRow("Bake sampling rate", self.bake_rate)
        page.layout.addLayout(form)
        advanced = QtWidgets.QGroupBox("Advanced")
        advanced.setCheckable(True)
        advanced.setChecked(False)
        advanced_layout = QtWidgets.QFormLayout(advanced)
        self.transition_time = QtWidgets.QDoubleSpinBox()
        self.transition_time.setRange(0.0, 10.0)
        self.transition_time.setDecimals(3)
        self.transition_time.setValue(0.25)
        self.retarget_button = QtWidgets.QPushButton("Review mapping…")
        advanced_layout.addRow("Transition time", self.transition_time)
        advanced_layout.addRow("Explicit retarget mapping", self.retarget_button)
        page.layout.addWidget(advanced)
        self.animation_diagnostics = QtWidgets.QLabel(
            "Continuity, loop seams, root jumps, target nodes, scale keys, and external-rig bind pose have not been checked."
        )
        self.animation_diagnostics.setWordWrap(True)
        page.layout.addWidget(self.animation_diagnostics)
        for widget_signal in (
            self.loop_trim.toggled,
            self.playback_speed.valueChanged,
            self.trim_start.valueChanged,
            self.trim_end.valueChanged,
            self.retime_duration.valueChanged,
            self.root_motion.currentIndexChanged,
            self.bake_rate.currentIndexChanged,
            self.transition_time.valueChanged,
        ):
            widget_signal.connect(self._animation_preparation_changed)
        return page

    def _materials_page(self) -> QtWidgets.QWidget:
        page = _Page(
            "Materials, textures, and UVs",
            "Assign KOTOR-safe texture names and compare the original material, KOTOR approximation, and a UV checker. Converted copies go only to the build folder.",
        )
        self.material_table = QtWidgets.QTableWidget(0, 8)
        self.material_table.setHorizontalHeaderLabels((
            "Material", "Source texture", "KOTOR texture", "UV range", "Wrap", "Alpha", "Output", "Status"
        ))
        self.material_table.horizontalHeader().setStretchLastSection(True)
        self.material_table.currentCellChanged.connect(self._material_row_selected)
        self.material_table.cellDoubleClicked.connect(self._material_cell_double_clicked)
        self.material_table.itemChanged.connect(self._material_table_item_changed)
        page.layout.addWidget(self.material_table, 1)
        previews = QtWidgets.QHBoxLayout()
        self.material_preview_labels: dict[str, QtWidgets.QLabel] = {}
        for key, title in (("source", "Source material preview"), ("kotor", "KOTOR approximation"), ("uv", "UV checker preview")):
            group = QtWidgets.QGroupBox(title)
            layout = QtWidgets.QVBoxLayout(group)
            view = QtWidgets.QLabel("Choose a material")
            view.setAlignment(QtCore.Qt.AlignCenter)
            view.setMinimumHeight(180)
            view.setFrameShape(QtWidgets.QFrame.StyledPanel)
            layout.addWidget(view)
            previews.addWidget(group)
            self.material_preview_labels[key] = view
        page.layout.addLayout(previews)
        output = QtWidgets.QGroupBox("Generated texture copies")
        output_layout = QtWidgets.QVBoxLayout(output)
        format_row = QtWidgets.QHBoxLayout()
        self.output_tga = QtWidgets.QRadioButton("TGA + TXI")
        self.output_tpc = QtWidgets.QRadioButton("TPC")
        self.output_tga.setChecked(True)
        self.preserve_repeat = QtWidgets.QCheckBox("Preserve repeat wrapping")
        self.preserve_repeat.setChecked(True)
        self.orient_for_kotor = QtWidgets.QCheckBox("Orient imported image for KOTOR (recommended)")
        self.orient_for_kotor.setChecked(True)
        self.orient_for_kotor.setToolTip(
            "Matches imported image rows to Odyssey MDX texture coordinates. "
            "Turn this off only when the image was already authored in KOTOR orientation."
        )
        format_row.addWidget(self.output_tga)
        format_row.addWidget(self.output_tpc)
        format_row.addWidget(self.preserve_repeat)
        format_row.addStretch(1)
        output_layout.addLayout(format_row)
        output_layout.addWidget(self.orient_for_kotor)
        page.layout.addWidget(output)
        cutout_help = QtWidgets.QLabel(
            "Choose Cutout / punch-through in the Alpha column for hair, leaves, or other hard-edged transparency. "
            "Ghost Studio adds the standard KOTOR settings automatically."
        )
        cutout_help.setWordWrap(True)
        page.layout.addWidget(cutout_help)
        advanced = QtWidgets.QGroupBox("Advanced TXI settings")
        advanced.setCheckable(True)
        advanced.setChecked(False)
        advanced_layout = QtWidgets.QVBoxLayout(advanced)
        self.txi_editor = QtWidgets.QPlainTextEdit()
        self.txi_editor.setPlaceholderText("Example: blending punchthrough")
        self.txi_editor.textChanged.connect(self._material_controls_changed)
        advanced_layout.addWidget(self.txi_editor)
        self.output_tga.toggled.connect(self._material_controls_changed)
        self.output_tpc.toggled.connect(self._material_controls_changed)
        self.preserve_repeat.toggled.connect(self._material_controls_changed)
        self.orient_for_kotor.toggled.connect(self._material_controls_changed)
        page.layout.addWidget(advanced)
        return page

    def _secondary_motion_page(self) -> QtWidgets.QWidget:
        self.secondary_motion_editor = QtSecondaryMotionEditor(
            self,
            target_kind="custom_character",
            target_game=self.project.target_game,
        )
        self.secondary_motion_editor.set_profile(self.project.secondary_motion)
        self.secondary_motion_editor.profileChanged.connect(self._secondary_motion_changed)
        return self.secondary_motion_editor

    def _secondary_motion_changed(self, profile: Mapping[str, Any]) -> None:
        self.project.secondary_motion = dict(profile)
        self._form_changed()

    def _gameplay_page(self) -> QtWidgets.QWidget:
        page = _Page(
            "Behavior and KOTOR gameplay integration",
            "Start from a creature that already works in your installed game. Ghost Studio copies its UTC in memory, keeps unknown combat data, and lets you change only the parts you intend.",
        )

        templates = QtWidgets.QGroupBox("1. Choose a proven creature template")
        templates.setObjectName("customCharacterBehaviorTemplates")
        template_layout = QtWidgets.QVBoxLayout(templates)
        template_help = QtWidgets.QLabel(
            "This searchable list is built read-only from every installed UTC blueprint. "
            "The standard Zakkeg is recommended for Borhek because its AI scripts are globally available; "
            "encounter-specific Zakkeg scripts are marked as module-bound."
        )
        template_help.setWordWrap(True)
        template_layout.addWidget(template_help)
        template_actions = QtWidgets.QHBoxLayout()
        self.behavior_template_search = QtWidgets.QLineEdit()
        self.behavior_template_search.setPlaceholderText("Search a character name or UTC resource name…")
        self.behavior_template_search.setClearButtonEnabled(True)
        self.refresh_behavior_templates = QtWidgets.QPushButton("Read installed character templates")
        self.use_zakkeg_template = QtWidgets.QPushButton("Use standard Zakkeg for Borhek")
        self.use_zakkeg_template.setToolTip("Selects c_zakkeg01.utc, the global K2 Zakkeg creature template.")
        template_actions.addWidget(self.behavior_template_search, 1)
        template_actions.addWidget(self.refresh_behavior_templates)
        template_actions.addWidget(self.use_zakkeg_template)
        template_layout.addLayout(template_actions)
        self.behavior_template_table = QtWidgets.QTreeWidget()
        self.behavior_template_table.setObjectName("customCharacterBehaviorTemplateTable")
        self.behavior_template_table.setHeaderLabels(("Character", "UTC resource", "Location", "Portability"))
        self.behavior_template_table.setRootIsDecorated(False)
        self.behavior_template_table.setAlternatingRowColors(True)
        self.behavior_template_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.behavior_template_table.setMinimumHeight(235)
        self.behavior_template_table.header().setStretchLastSection(False)
        self.behavior_template_table.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        # Installed mods may report long archive or Override locations.  Keep the
        # beginner-facing table inside the workbench and expose full values as
        # tooltips instead of allowing one path to widen the entire page.
        self.behavior_template_table.header().setSectionResizeMode(1, QtWidgets.QHeaderView.Interactive)
        self.behavior_template_table.header().setSectionResizeMode(2, QtWidgets.QHeaderView.Interactive)
        self.behavior_template_table.header().setSectionResizeMode(3, QtWidgets.QHeaderView.Interactive)
        self.behavior_template_table.setColumnWidth(1, 145)
        self.behavior_template_table.setColumnWidth(2, 165)
        self.behavior_template_table.setColumnWidth(3, 155)
        template_layout.addWidget(self.behavior_template_table)
        self.behavior_template_details = QtWidgets.QLabel(
            "Choose the KOTOR game folder on Install and test, then read its character templates."
        )
        self.behavior_template_details.setWordWrap(True)
        self.behavior_template_details.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        template_layout.addWidget(self.behavior_template_details)
        use_template_row = QtWidgets.QHBoxLayout()
        self.use_selected_behavior_template = QtWidgets.QPushButton("Use selected template")
        self.use_selected_behavior_template.setEnabled(False)
        self.behavior_catalog_status = QtWidgets.QLabel("Template catalog not loaded")
        self.behavior_catalog_status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.behavior_catalog_status.setWordWrap(True)
        self.behavior_catalog_status.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Preferred,
        )
        use_template_row.addWidget(self.use_selected_behavior_template)
        use_template_row.addWidget(self.behavior_catalog_status, 1)
        template_layout.addLayout(use_template_row)
        page.layout.addWidget(templates)

        inherited = QtWidgets.QGroupBox("2. Keep or tune the template's combat setup")
        inherited_layout = QtWidgets.QVBoxLayout(inherited)
        self.inherit_template_stats = QtWidgets.QCheckBox(
            "Keep the template's combat stats, class, feats, skills, equipment, perception, and sound set"
        )
        self.inherit_template_stats.setChecked(True)
        self.inherit_template_stats.setToolTip(
            "Recommended. Identity and the new appearance are changed, while the selected UTC's combat fields stay intact."
        )
        inherited_layout.addWidget(self.inherit_template_stats)
        form = QtWidgets.QFormLayout()
        self.appearance_name = QtWidgets.QLineEdit()
        self.display_name = QtWidgets.QLineEdit()
        self.behavior_preset = QtWidgets.QComboBox()
        self.behavior_preset.addItems((
            "Installed character template",
            "Passive creature", "Hostile melee creature", "Ranged creature",
            "Stationary creature", "Flying / hovering creature", "Custom scripts",
        ))
        self.faction = QtWidgets.QComboBox()
        self.faction.addItems(("Neutral", "Hostile", "Friendly", "Custom"))
        self.movement_rate = QtWidgets.QComboBox()
        self.movement_rate.addItems(("Default", "Slow", "Normal", "Fast"))
        self.soundset = QtWidgets.QLineEdit()
        self.collision_size = QtWidgets.QDoubleSpinBox()
        self.collision_size.setRange(0.0, 100.0)
        self.collision_size.setValue(1.0)
        self.perception_range = QtWidgets.QDoubleSpinBox()
        self.perception_range.setRange(0.0, 100.0)
        self.perception_range.setValue(10.0)
        self.level = QtWidgets.QSpinBox()
        self.level.setRange(1, 255)
        self.level.setValue(5)
        self.hit_points = QtWidgets.QSpinBox()
        self.hit_points.setRange(1, 32767)
        self.hit_points.setValue(45)
        form.addRow("Appearance name", self.appearance_name)
        form.addRow("Creature display name", self.display_name)
        form.addRow("Base behavior", self.behavior_preset)
        form.addRow("Faction", self.faction)
        form.addRow("Movement rate", self.movement_rate)
        form.addRow("Template soundset row", self.soundset)
        form.addRow("Personal space / collision size", self.collision_size)
        form.addRow("Perception range", self.perception_range)
        form.addRow("Creature level", self.level)
        form.addRow("Hit points", self.hit_points)
        inherited_layout.addLayout(form)
        page.layout.addWidget(inherited)

        sounds = QtWidgets.QGroupBox("3. Add creature sounds")
        sounds.setObjectName("customCharacterCreatureSounds")
        sounds_layout = QtWidgets.QVBoxLayout(sounds)
        sounds_help = QtWidgets.QLabel(
            "Choose ordinary mono 16-bit PCM WAV files. Ghost Studio builds a native KOTOR SSF soundset, "
            "adds one merge-safe soundset.2da row, and appends only the required voice rows to dialog.tlk. "
            "The exact global changes are previewed, backed up, and restorable; the creature's direct AI event scripts are not wrapped or replaced."
        )
        sounds_help.setWordWrap(True)
        sounds_layout.addWidget(sounds_help)
        sound_actions = QtWidgets.QHBoxLayout()
        self.choose_creature_sound_folder = QtWidgets.QPushButton("Choose a sound folder and match cues…")
        self.choose_creature_sound_file = QtWidgets.QPushButton("Choose WAV for selected cue…")
        self.clear_creature_sounds = QtWidgets.QPushButton("Clear creature sounds")
        sound_actions.addWidget(self.choose_creature_sound_folder)
        sound_actions.addWidget(self.choose_creature_sound_file)
        sound_actions.addWidget(self.clear_creature_sounds)
        sound_actions.addStretch(1)
        sounds_layout.addLayout(sound_actions)
        self.creature_sound_table = QtWidgets.QTreeWidget()
        self.creature_sound_table.setObjectName("customCharacterCreatureSoundTable")
        self.creature_sound_table.setHeaderLabels(("Native sound cue", "Selected WAV", "KOTOR resource", "Status"))
        self.creature_sound_table.setRootIsDecorated(False)
        self.creature_sound_table.setAlternatingRowColors(True)
        self.creature_sound_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.creature_sound_table.setMinimumHeight(210)
        self.creature_sound_table.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.creature_sound_table.header().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.creature_sound_table.header().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.creature_sound_table.header().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        sounds_layout.addWidget(self.creature_sound_table)
        self.creature_sound_status = QtWidgets.QLabel("No custom creature sounds selected; the installed template soundset remains available.")
        self.creature_sound_status.setWordWrap(True)
        self.creature_sound_status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        sounds_layout.addWidget(self.creature_sound_status)
        page.layout.addWidget(sounds)

        hooks = QtWidgets.QGroupBox("4. Review the UTC behavior hooks")
        hooks_layout = QtWidgets.QVBoxLayout(hooks)
        hook_help = QtWidgets.QLabel(
            "KOTOR calls these scripts when the creature sees someone, is attacked, takes damage, finishes a combat round, spawns, dies, or becomes blocked. "
            "Template scripts remain unchanged unless you explicitly replace one."
        )
        hook_help.setWordWrap(True)
        hooks_layout.addWidget(hook_help)
        self.behavior_hook_table = QtWidgets.QTreeWidget()
        self.behavior_hook_table.setObjectName("customCharacterBehaviorHookTable")
        self.behavior_hook_table.setHeaderLabels(("When this happens", "Template script", "Borhek will use", "Status"))
        self.behavior_hook_table.setRootIsDecorated(False)
        self.behavior_hook_table.setAlternatingRowColors(True)
        self.behavior_hook_table.setMinimumHeight(265)
        self.behavior_hook_table.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.behavior_hook_table.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.behavior_hook_table.header().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.behavior_hook_table.header().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        hooks_layout.addWidget(self.behavior_hook_table)
        page.layout.addWidget(hooks)

        advanced = QtWidgets.QGroupBox("Advanced: write or assign a behavior script")
        advanced.setObjectName("customCharacterAdvancedBehaviorCode")
        advanced.setCheckable(True)
        advanced.setChecked(False)
        advanced_layout = QtWidgets.QVBoxLayout(advanced)
        advanced_help = QtWidgets.QLabel(
            "Optional. Pick one event, keep its template script, point to an existing NCS resource, or write auditable NWScript. "
            "Custom source must compile and parse back before Build can package it."
        )
        advanced_help.setWordWrap(True)
        advanced_layout.addWidget(advanced_help)
        advanced_form = QtWidgets.QFormLayout()
        self.behavior_hook_combo = QtWidgets.QComboBox()
        for hook in UTC_SCRIPT_HOOK_FIELDS:
            self.behavior_hook_combo.addItem(UTC_SCRIPT_HOOK_LABELS.get(hook, hook), hook)
        self.behavior_hook_mode = QtWidgets.QComboBox()
        self.behavior_hook_mode.addItem("Keep the template script", "inherit")
        self.behavior_hook_mode.addItem("Use an existing KOTOR script", "existing")
        self.behavior_hook_mode.addItem("Write a custom NWScript", "custom")
        self.behavior_script_resref = QtWidgets.QLineEdit()
        self.behavior_script_resref.setMaxLength(16)
        self.behavior_script_resref.setPlaceholderText("Example: bor_attacked")
        advanced_form.addRow("UTC event", self.behavior_hook_combo)
        advanced_form.addRow("What to use", self.behavior_hook_mode)
        advanced_form.addRow("Script resource name", self.behavior_script_resref)
        advanced_layout.addLayout(advanced_form)
        starter_row = QtWidgets.QHBoxLayout()
        self.load_behavior_starter = QtWidgets.QPushButton("Create safe starter code")
        self.inherited_script_label = QtWidgets.QLabel("Template script: —")
        self.inherited_script_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        starter_row.addWidget(self.load_behavior_starter)
        starter_row.addWidget(self.inherited_script_label, 1)
        advanced_layout.addLayout(starter_row)
        self.behavior_script_source = QtWidgets.QPlainTextEdit()
        self.behavior_script_source.setObjectName("customCharacterBehaviorSourceEditor")
        self.behavior_script_source.setPlaceholderText("void main()\n{\n    // Add KOTOR actions here.\n}")
        self.behavior_script_source.setMinimumHeight(210)
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.behavior_script_source.setFont(font)
        advanced_layout.addWidget(self.behavior_script_source)
        compile_row = QtWidgets.QHBoxLayout()
        self.apply_behavior_hook = QtWidgets.QPushButton("Compile, check, and use this hook")
        self.behavior_compile_status = QtWidgets.QLabel("No explicit change")
        self.behavior_compile_status.setWordWrap(True)
        self.behavior_compile_status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        compile_row.addWidget(self.apply_behavior_hook)
        compile_row.addWidget(self.behavior_compile_status, 1)
        advanced_layout.addLayout(compile_row)
        self.behavior_compile_diagnostics = QtWidgets.QListWidget()
        self.behavior_compile_diagnostics.setMaximumHeight(125)
        advanced_layout.addWidget(self.behavior_compile_diagnostics)
        page.layout.addWidget(advanced)

        optional = QtWidgets.QGroupBox("Optional test setup")
        optional_layout = QtWidgets.QVBoxLayout(optional)
        self.generate_utc = QtWidgets.QCheckBox("Generate a UTC creature blueprint")
        self.generate_utc.setChecked(True)
        self.generate_spawn = QtWidgets.QCheckBox("Generate an optional advanced spawn script")
        self.prepare_module_placement = QtWidgets.QCheckBox(
            "Place one temporary test creature in PLCaa DevRoom (no console needed)"
        )
        self.replace_test_placement = QtWidgets.QCheckBox(
            "Replace the creature already at the requested test spot"
        )
        self.replace_test_placement.setToolTip(
            "Use this when an earlier prototype is already standing at the same test coordinates. "
            "The exact creature being replaced is shown in the install preview before anything changes."
        )
        self.module_placement_help = QtWidgets.QLabel(
            "Ghost Studio safely merges one test placement into KOTOR II's PLCaa DevRoom, "
            "keeps every other module resource and placement, backs up the live module, "
            "and includes it in Restore."
        )
        self.module_placement_help.setWordWrap(True)
        optional_layout.addWidget(self.generate_utc)
        optional_layout.addWidget(self.generate_spawn)
        optional_layout.addWidget(self.prepare_module_placement)
        optional_layout.addWidget(self.replace_test_placement)
        optional_layout.addWidget(self.module_placement_help)
        page.layout.addWidget(optional)
        self.behavior_warnings = QtWidgets.QListWidget()
        self.behavior_warnings.addItem("No attack animation is assigned yet.")
        self.behavior_warnings.addItem("No death animation is assigned yet.")
        page.layout.addWidget(self.behavior_warnings)
        for widget in (self.appearance_name, self.display_name, self.soundset):
            widget.textChanged.connect(self._form_changed)
        self.behavior_preset.currentIndexChanged.connect(self._form_changed)
        self.faction.currentIndexChanged.connect(self._form_changed)
        self.movement_rate.currentIndexChanged.connect(self._form_changed)
        for spin in (self.collision_size, self.perception_range, self.level, self.hit_points):
            spin.valueChanged.connect(self._form_changed)
        for box in (
            self.generate_utc,
            self.generate_spawn,
            self.prepare_module_placement,
            self.replace_test_placement,
        ):
            box.toggled.connect(self._form_changed)
        self.prepare_module_placement.toggled.connect(self.replace_test_placement.setEnabled)
        self.inherit_template_stats.toggled.connect(self._behavior_inheritance_changed)
        self.behavior_template_search.textChanged.connect(self._filter_behavior_templates)
        self.behavior_template_table.itemSelectionChanged.connect(self._behavior_template_selection_changed)
        self.refresh_behavior_templates.clicked.connect(self._request_behavior_catalog)
        self.use_selected_behavior_template.clicked.connect(self._request_selected_behavior_template)
        self.use_zakkeg_template.clicked.connect(self._request_zakkeg_behavior_template)
        self.behavior_hook_combo.currentIndexChanged.connect(self._load_behavior_hook_editor)
        self.behavior_hook_mode.currentIndexChanged.connect(self._behavior_hook_mode_changed)
        self.load_behavior_starter.clicked.connect(
            lambda: self.behaviorStarterRequested.emit(str(self.behavior_hook_combo.currentData() or ""))
        )
        self.apply_behavior_hook.clicked.connect(self._request_behavior_hook_apply)
        self.choose_creature_sound_folder.clicked.connect(self._choose_creature_sound_folder)
        self.choose_creature_sound_file.clicked.connect(self._choose_selected_creature_sound)
        self.clear_creature_sounds.clicked.connect(self._clear_creature_sounds)
        self.creature_sound_table.itemDoubleClicked.connect(
            lambda _item, _column: self._choose_selected_creature_sound()
        )
        self._refresh_creature_sound_table()
        self._refresh_behavior_hook_table()
        self._behavior_hook_mode_changed()
        self._behavior_inheritance_changed()
        return page

    def _refresh_creature_sound_table(self) -> None:
        if not hasattr(self, "creature_sound_table"):
            return
        selected = str(
            self.creature_sound_table.currentItem().data(0, QtCore.Qt.UserRole)
            if self.creature_sound_table.currentItem() is not None else ""
        )
        by_cue = {str(value.cue).casefold(): value for value in self.project.creature_sound_cues}
        self.creature_sound_table.clear()
        selected_item: QtWidgets.QTreeWidgetItem | None = None
        ready = 0
        missing = 0
        for cue, definition in CREATURE_SOUND_CUE_DEFINITIONS.items():
            value = by_cue.get(cue)
            path = self.project.resolve_path(value.source_path) if value and value.source_path else Path()
            exists = bool(value and value.source_path and path.is_file())
            if exists:
                ready += 1
            elif value and value.source_path:
                missing += 1
            item = QtWidgets.QTreeWidgetItem((
                str(definition["label"]),
                path.name if value and value.source_path else "—",
                str(value.output_resref or "Build assigns it") if value else "—",
                "Ready" if exists else ("File not found" if value and value.source_path else "Optional"),
            ))
            item.setData(0, QtCore.Qt.UserRole, cue)
            if value and value.source_path:
                item.setToolTip(1, str(path))
            self.creature_sound_table.addTopLevelItem(item)
            if cue == selected:
                selected_item = item
        if selected_item is not None:
            self.creature_sound_table.setCurrentItem(selected_item)
        if ready:
            detail = f"{ready} creature sound cue(s) ready"
            if missing:
                detail += f"; {missing} source file(s) need to be chosen again"
            self.creature_sound_status.setText(
                detail + ". Build will preserve every selected UTC script and route these WAVs through a native SSF soundset."
            )
        else:
            self.creature_sound_status.setText(
                "No custom creature sounds selected; the installed template soundset remains available."
            )

    def _choose_creature_sound_folder(self) -> None:
        start = str(self._settings.value("last_creature_sound_folder", "") or "")
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose creature sound folder", start)
        if not folder:
            return
        self._sync_project_from_form()
        wav_files = sorted(
            (path for path in Path(folder).rglob("*") if path.is_file() and path.suffix.casefold() == ".wav"),
            key=lambda path: path.name.casefold(),
        )
        matched: list[CreatureSoundCue] = []
        used: set[Path] = set()
        for cue, definition in CREATURE_SOUND_CUE_DEFINITIONS.items():
            chosen = next(
                (
                    path for path in wav_files
                    if path not in used
                    and any(pattern in path.stem.casefold() for pattern in definition["patterns"])
                ),
                None,
            )
            if chosen is None:
                continue
            used.add(chosen)
            try:
                output_resref = creature_sound_resref(self.project, cue)
                digest = sha256_file(chosen)
            except (OSError, ValueError) as exc:
                QtWidgets.QMessageBox.warning(self, "Could not use creature sound", str(exc))
                return
            matched.append(CreatureSoundCue(cue, str(chosen), digest, output_resref))
        if not matched:
            QtWidgets.QMessageBox.information(
                self,
                "No recognizable creature sounds",
                "No WAV filename matched roar, attack, get-hit or hurt, defense, hit-wall, breathing or idle, and death cues. "
                "Select a row and use Choose WAV for selected cue to assign files manually.",
            )
            return
        self.project.creature_sound_cues = matched
        self._settings.setValue("last_creature_sound_folder", folder)
        self._refresh_creature_sound_table()
        self._form_changed()

    def _choose_selected_creature_sound(self) -> None:
        item = self.creature_sound_table.currentItem()
        if item is None:
            QtWidgets.QMessageBox.information(self, "Choose a sound cue", "Select one creature sound row first.")
            return
        cue = str(item.data(0, QtCore.Qt.UserRole) or "")
        existing = next((value for value in self.project.creature_sound_cues if value.cue == cue), None)
        start = str(self.project.resolve_path(existing.source_path)) if existing and existing.source_path else str(
            self._settings.value("last_creature_sound_folder", "") or ""
        )
        source, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            f"Choose {CREATURE_SOUND_CUE_DEFINITIONS[cue]['label']} WAV",
            start,
            "PCM WAV audio (*.wav);;All files (*)",
        )
        if not source:
            return
        self._sync_project_from_form()
        path = Path(source)
        try:
            value = CreatureSoundCue(cue, str(path), sha256_file(path), creature_sound_resref(self.project, cue))
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Could not use creature sound", str(exc))
            return
        self.project.creature_sound_cues = [
            row for row in self.project.creature_sound_cues if row.cue != cue
        ] + [value]
        self._settings.setValue("last_creature_sound_folder", str(path.parent))
        self._refresh_creature_sound_table()
        self._form_changed()

    def _clear_creature_sounds(self) -> None:
        if not self.project.creature_sound_cues:
            return
        self.project.creature_sound_cues = []
        self._refresh_creature_sound_table()
        self._form_changed()

    def _behavior_inheritance_changed(self, *_args: Any) -> None:
        if not hasattr(self, "inherit_template_stats"):
            return
        editable = not self.inherit_template_stats.isChecked()
        for widget in (
            self.faction,
            self.movement_rate,
            self.soundset,
            self.perception_range,
            self.level,
            self.hit_points,
        ):
            widget.setEnabled(editable)
        if not self._loading:
            self._form_changed()

    def _request_behavior_catalog(self) -> None:
        self.refresh_behavior_templates.setEnabled(False)
        self.behavior_catalog_status.setText("Reading installed UTC templates in the background…")
        self.behaviorCatalogRequested.emit(self._sync_project_from_form())

    def set_behavior_catalog_busy(self, busy: bool, message: str = "") -> None:
        self.refresh_behavior_templates.setEnabled(not busy)
        if message:
            self.behavior_catalog_status.setText(message)

    def set_behavior_template_catalog(
        self,
        rows: list[Mapping[str, Any]],
        *,
        report_path: str = "",
    ) -> None:
        self.behavior_template_table.clear()
        preferred = str(self.project.behavior_profile.get("template_resref") or "").casefold()
        preferred_item: QtWidgets.QTreeWidgetItem | None = None
        for row_value in rows:
            row = dict(row_value or {})
            resref = str(row.get("resref") or "").strip().lower()
            module_hooks = list(row.get("module_only_script_hooks") or ())
            global_template = bool(row.get("global_template"))
            portability = "Global / reusable" if global_template and not module_hooks else (
                "Module-bound scripts" if module_hooks else "Module instance"
            )
            item = QtWidgets.QTreeWidgetItem((
                str(row.get("display_name") or resref),
                resref,
                str(row.get("source") or "installation"),
                portability,
            ))
            item.setData(0, QtCore.Qt.UserRole, row)
            item.setToolTip(2, str(row.get("source") or "installation"))
            item.setToolTip(3, (
                "Some assigned scripts exist only inside a specific module. Review before reuse."
                if module_hooks else "This template does not depend on module-only script hooks."
            ))
            self.behavior_template_table.addTopLevelItem(item)
            if resref.casefold() == preferred:
                preferred_item = item
        self.refresh_behavior_templates.setEnabled(True)
        self.behavior_catalog_status.setText(f"{len(rows):,} installed character template(s) ready")
        self.behavior_catalog_status.setToolTip(
            f"Machine-readable catalog: {report_path}" if report_path else ""
        )
        if preferred_item is not None:
            self.behavior_template_table.setCurrentItem(preferred_item)
            self.behavior_template_table.scrollToItem(preferred_item)
        self._filter_behavior_templates(self.behavior_template_search.text())

    def _filter_behavior_templates(self, text: str) -> None:
        if not hasattr(self, "behavior_template_table"):
            return
        query = str(text or "").strip().casefold()
        for index in range(self.behavior_template_table.topLevelItemCount()):
            item = self.behavior_template_table.topLevelItem(index)
            haystack = " ".join(item.text(column) for column in range(4)).casefold()
            item.setHidden(bool(query and query not in haystack))

    def _behavior_template_selection_changed(self) -> None:
        items = self.behavior_template_table.selectedItems()
        self.use_selected_behavior_template.setEnabled(bool(items))
        if not items:
            return
        row = dict(items[0].data(0, QtCore.Qt.UserRole) or {})
        classes = ", ".join(
            f"class {value.get('class_id')} level {value.get('level')}"
            for value in row.get("classes") or ()
        ) or "no class rows"
        hooks = dict(row.get("script_hooks") or {})
        active_hooks = sum(1 for value in hooks.values() if str(value or "").strip())
        warning = ""
        module_hooks = list(row.get("module_only_script_hooks") or ())
        if module_hooks:
            labels = ", ".join(UTC_SCRIPT_HOOK_LABELS.get(value, value) for value in module_hooks)
            warning = f" Warning: module-only scripts are assigned to {labels}."
        self.behavior_template_details.setText(
            f"{row.get('display_name') or row.get('resref')} ({row.get('resref')}.utc) • "
            f"faction {row.get('faction_id')} • {classes} • max HP {row.get('max_hit_points')} • "
            f"challenge {row.get('challenge_rating')} • {active_hooks} active script hook(s).{warning}"
        )

    def _request_selected_behavior_template(self) -> None:
        items = self.behavior_template_table.selectedItems()
        if items:
            self.behaviorTemplateRequested.emit(str(items[0].text(1) or "").strip().lower())

    def _request_zakkeg_behavior_template(self) -> None:
        for index in range(self.behavior_template_table.topLevelItemCount()):
            item = self.behavior_template_table.topLevelItem(index)
            if item.text(1).casefold() == "c_zakkeg01":
                item.setHidden(False)
                self.behavior_template_table.setCurrentItem(item)
                self.behavior_template_table.scrollToItem(item)
                self.behaviorTemplateRequested.emit("c_zakkeg01")
                return
        QtWidgets.QMessageBox.information(
            self,
            "Read templates first",
            "Choose the KOTOR II game folder, then click Read installed character templates. "
            "Ghost Studio will locate c_zakkeg01.utc without guessing a path.",
        )

    def set_behavior_template_summary(self, row_value: Mapping[str, Any]) -> None:
        row = dict(row_value or {})
        self._loading = True
        try:
            self.inherit_template_stats.setChecked(
                bool(self.project.behavior_profile.get("inherit_template_combat_stats", True))
            )
            behavior_index = self.behavior_preset.findText("Installed character template")
            self.behavior_preset.setCurrentIndex(max(0, behavior_index))
            faction_label = {1: "Hostile", 2: "Friendly", 5: "Neutral"}.get(
                int(row.get("faction_id", -1)), "Custom"
            )
            faction_index = self.faction.findText(faction_label)
            self.faction.setCurrentIndex(max(0, faction_index))
            self.soundset.setText(str(max(0, int(row.get("soundset", 0) or 0))))
            self.perception_range.setValue(float(max(0, int(row.get("perception_range", 0) or 0))))
            classes = list(row.get("classes") or ())
            if classes:
                self.level.setValue(max(1, sum(int(value.get("level", 0) or 0) for value in classes)))
            self.hit_points.setValue(max(1, int(row.get("max_hit_points", 1) or 1)))
            self.behavior_template_details.setText(
                f"Using {row.get('display_name') or row.get('resref')} ({row.get('resref')}.utc). "
                "The source resource stays read-only; Borhek receives a project-owned copy during Build."
            )
            self._refresh_behavior_hook_table()
            self._load_behavior_hook_editor()
            self._behavior_inheritance_changed()
        finally:
            self._loading = False
        self._form_changed()

    def _refresh_behavior_hook_table(self) -> None:
        if not hasattr(self, "behavior_hook_table"):
            return
        self.behavior_hook_table.clear()
        profile = dict(self.project.behavior_profile or {})
        template = dict(profile.get("template_snapshot") or {})
        inherited = dict(template.get("script_hooks") or {})
        overrides = dict(profile.get("script_hooks") or {})
        for hook in UTC_SCRIPT_HOOK_FIELDS:
            template_script = str(inherited.get(hook) or "")
            explicit = dict(overrides.get(hook) or {})
            mode = str(explicit.get("mode") or "inherit")
            result_script = template_script if mode == "inherit" else str(explicit.get("resref") or "")
            status = {
                "inherit": "Template",
                "existing": "Existing resource",
                "custom": "Custom source; compiled on Build",
            }.get(mode, mode)
            item = QtWidgets.QTreeWidgetItem((
                UTC_SCRIPT_HOOK_LABELS.get(hook, hook),
                template_script or "—",
                result_script or "—",
                status,
            ))
            item.setData(0, QtCore.Qt.UserRole, hook)
            self.behavior_hook_table.addTopLevelItem(item)

    def _load_behavior_hook_editor(self, *_args: Any) -> None:
        if not hasattr(self, "behavior_hook_combo"):
            return
        hook = str(self.behavior_hook_combo.currentData() or "")
        profile = dict(self.project.behavior_profile or {})
        template = dict(profile.get("template_snapshot") or {})
        inherited = str(dict(template.get("script_hooks") or {}).get(hook) or "")
        override = dict(dict(profile.get("script_hooks") or {}).get(hook) or {})
        mode = str(override.get("mode") or "inherit")
        self.behavior_hook_mode.blockSignals(True)
        self.behavior_script_resref.blockSignals(True)
        self.behavior_script_source.blockSignals(True)
        try:
            index = self.behavior_hook_mode.findData(mode)
            self.behavior_hook_mode.setCurrentIndex(max(0, index))
            self.behavior_script_resref.setText(str(override.get("resref") or ""))
            self.behavior_script_source.setPlainText(str(override.get("source") or ""))
            self.inherited_script_label.setText(f"Template script: {inherited or 'none'}")
            self.behavior_compile_status.setText(
                "Using the template unchanged" if mode == "inherit" else (
                    "Uses an existing NCS resource" if mode == "existing" else "Custom source saved; Build recompiles it"
                )
            )
        finally:
            self.behavior_hook_mode.blockSignals(False)
            self.behavior_script_resref.blockSignals(False)
            self.behavior_script_source.blockSignals(False)
        self._behavior_hook_mode_changed()

    def _behavior_hook_mode_changed(self, *_args: Any) -> None:
        if not hasattr(self, "behavior_hook_mode"):
            return
        mode = str(self.behavior_hook_mode.currentData() or "inherit")
        self.behavior_script_resref.setEnabled(mode in {"existing", "custom"})
        self.behavior_script_source.setEnabled(mode == "custom")
        self.load_behavior_starter.setEnabled(mode == "custom")
        self.apply_behavior_hook.setText(
            "Keep template script" if mode == "inherit" else (
                "Use existing script" if mode == "existing" else "Compile, check, and use this hook"
            )
        )

    def set_behavior_starter_source(self, source: str, suggested_resref: str = "") -> None:
        self.behavior_hook_mode.setCurrentIndex(self.behavior_hook_mode.findData("custom"))
        if suggested_resref and not self.behavior_script_resref.text().strip():
            self.behavior_script_resref.setText(suggested_resref)
        self.behavior_script_source.setPlainText(str(source or ""))
        self.behavior_compile_status.setText("Starter loaded. Review it, then compile and use this hook.")

    def _request_behavior_hook_apply(self) -> None:
        self.behavior_compile_diagnostics.clear()
        self.behavior_compile_status.setText("Checking behavior hook…")
        self.behaviorHookApplyRequested.emit({
            "hook": str(self.behavior_hook_combo.currentData() or ""),
            "mode": str(self.behavior_hook_mode.currentData() or "inherit"),
            "resref": self.behavior_script_resref.text().strip().lower(),
            "source": self.behavior_script_source.toPlainText(),
        })

    def set_behavior_hook_result(self, payload: Mapping[str, Any]) -> None:
        result = dict(payload or {})
        self.behavior_compile_diagnostics.clear()
        for diagnostic in result.get("diagnostics") or ():
            row = dict(diagnostic or {})
            self.behavior_compile_diagnostics.addItem(
                f"{str(row.get('severity') or '').title()}: {row.get('message') or ''}"
            )
        if result.get("ok"):
            self.behavior_compile_status.setText(str(result.get("message") or "Behavior hook is ready."))
            self._refresh_behavior_hook_table()
            self._load_behavior_hook_editor()
            self._autosave.start()
        else:
            self.behavior_compile_status.setText(str(result.get("error") or "This hook is not ready."))

    def _validation_build_page(self) -> QtWidgets.QWidget:
        page = _Page(
            "Validation and build",
            "Run the complete preflight before Build. Every error explains what is wrong, why KOTOR cannot accept it, whether it can be fixed, and what the fix changes.",
        )
        summary = QtWidgets.QGroupBox("Build summary")
        summary_layout = QtWidgets.QFormLayout(summary)
        self.build_summary: dict[str, QtWidgets.QLabel] = {}
        for key, label in (
            ("resref", "Model resource name"),
            ("nodes", "Skeleton nodes"),
            ("meshes", "Meshes"),
            ("mappings", "Animation mappings"),
            ("registrations", "Custom registrations"),
            ("textures", "Texture outputs"),
            ("appearance", "Appearance patch"),
            ("utc", "UTC output"),
            ("destination", "Package destination"),
        ):
            value = QtWidgets.QLabel("—")
            value.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            self.build_summary[key] = value
            summary_layout.addRow(label, value)
        page.layout.addWidget(summary)
        self.validation_tree = QtWidgets.QTreeWidget()
        self.validation_tree.setHeaderLabels(("Level", "What is wrong", "Why it matters", "Available fix"))
        page.layout.addWidget(self.validation_tree, 1)
        actions = QtWidgets.QHBoxLayout()
        validate = QtWidgets.QPushButton("Run complete preflight")
        self.build_button = QtWidgets.QPushButton("Build KOTOR package")
        self.build_button.setEnabled(False)
        open_folder = QtWidgets.QPushButton("Open Build Folder")
        validate.clicked.connect(lambda: self.validateRequested.emit(self._sync_project_from_form()))
        self.build_button.clicked.connect(lambda: self.buildRequested.emit(self._sync_project_from_form()))
        open_folder.clicked.connect(lambda: self.openBuildFolderRequested.emit(self.project.build_destination))
        actions.addWidget(validate)
        actions.addWidget(self.build_button)
        actions.addWidget(open_folder)
        actions.addStretch(1)
        page.layout.addLayout(actions)
        self.report_path_label = QtWidgets.QLabel("No persistent build report yet.")
        self.report_path_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        page.layout.addWidget(self.report_path_label)
        return page

    def _install_test_page(self) -> QtWidgets.QWidget:
        page = _Page(
            "Install and test",
            "Review every file first. Direct installation requires KOTOR to be closed, creates backups, and provides an uninstall or restore route through KOTOR Patch Manager.",
        )
        modes = QtWidgets.QGroupBox("Testing option")
        modes_layout = QtWidgets.QVBoxLayout(modes)
        for index, label in enumerate((
            "Build only", "Install to Override", "Create test UTC",
            "Add a temporary test spawn", "Launch through KOTOR Patch Manager", "Open test checklist",
        )):
            button = QtWidgets.QRadioButton(label)
            button.setChecked(index == 0)
            modes_layout.addWidget(button)
        page.layout.addWidget(modes)
        self.install_file_list = QtWidgets.QTreeWidget()
        self.install_file_list.setHeaderLabels(("File", "Destination", "Existing file", "Backup"))
        page.layout.addWidget(self.install_file_list, 1)
        install_actions = QtWidgets.QHBoxLayout()
        preview = QtWidgets.QPushButton("Preview exact install")
        self.install_button = QtWidgets.QPushButton("Install with backup")
        self.restore_button = QtWidgets.QPushButton("Restore previous files")
        launch = QtWidgets.QPushButton("Launch through Patch Manager")
        self.install_button.setEnabled(False)
        preview.clicked.connect(lambda: self.previewInstallRequested.emit(self._sync_project_from_form()))
        self.install_button.clicked.connect(lambda: self.installRequested.emit(self._sync_project_from_form()))
        self.restore_button.clicked.connect(lambda: self.restoreRequested.emit(self._sync_project_from_form()))
        launch.clicked.connect(lambda: self.launchPatchManagerRequested.emit(self._sync_project_from_form()))
        install_actions.addWidget(preview)
        install_actions.addWidget(self.install_button)
        install_actions.addWidget(self.restore_button)
        install_actions.addWidget(launch)
        install_actions.addStretch(1)
        page.layout.addLayout(install_actions)
        self.install_status = QtWidgets.QLabel("Build a package, choose the game folder, then preview the exact install.")
        self.install_status.setWordWrap(True)
        page.layout.addWidget(self.install_status)
        checklist = QtWidgets.QGroupBox("In-game test checklist")
        checklist_layout = QtWidgets.QVBoxLayout(checklist)
        for item in RUNTIME_TEST_CHECKLIST:
            checklist_layout.addWidget(QtWidgets.QCheckBox(item))
        page.layout.addWidget(checklist)
        return page

    def _add_external_animation_files(self) -> None:
        start = self.animation_folder.path() or str(self._settings.value("last_animation_folder", ""))
        values, _selected = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add external animation FBX files", start, "FBX files (*.fbx)"
        )
        existing = {
            str(self.external_animation_list.item(index).data(QtCore.Qt.UserRole) or "").casefold()
            for index in range(self.external_animation_list.count())
        }
        for value in values:
            if value.casefold() in existing:
                continue
            item = QtWidgets.QListWidgetItem(Path(value).name)
            item.setData(QtCore.Qt.UserRole, value)
            item.setToolTip(value)
            self.external_animation_list.addItem(item)
            existing.add(value.casefold())
        if values:
            self.animation_folder.set_path(str(Path(values[0]).parent))
            self._form_changed()

    def _remove_external_animation_files(self) -> None:
        for item in self.external_animation_list.selectedItems():
            self.external_animation_list.takeItem(self.external_animation_list.row(item))
        self._form_changed()

    def set_skeleton_root_choices(
        self,
        choices: list[str],
        *,
        selected: str = "",
        selection_required: bool = False,
    ) -> None:
        self.skeleton_root.blockSignals(True)
        self.skeleton_root.clear()
        for value in choices:
            self.skeleton_root.addItem(value, value)
        index = -1
        for candidate in range(self.skeleton_root.count()):
            value = str(self.skeleton_root.itemData(candidate) or "")
            if selected and (
                selected.casefold() == value.casefold()
                or value.casefold().endswith(f" :: {selected}".casefold())
            ):
                index = candidate
                break
        if index < 0 and len(choices) == 1 and not selection_required:
            index = 0
        self.skeleton_root.setCurrentIndex(index)
        self.skeleton_root.setEnabled(bool(choices))
        self.skeleton_root.setToolTip(
            "Choose one root and import again; Ghost Studio will not merge multiple rigs silently."
            if selection_required else "The deform hierarchy preserved in the Odyssey model."
        )
        self.skeleton_root.blockSignals(False)

    def _skeleton_root_changed(self, _index: int) -> None:
        if self._loading:
            return
        value = str(self.skeleton_root.currentData() or "")
        if value:
            self.project.selected_skeleton_root = value
            self.project_status.setText("Hierarchy selected — import again to inspect only this deform root")
        self._form_changed()

    def _repair_setting_changed(self, _checked: bool) -> None:
        if self._loading:
            return
        if self.repair_controls.get("select_one_hierarchy") and self.repair_controls["select_one_hierarchy"].isChecked():
            self.skeleton_root.setFocus()
        if self.repair_controls.get("reorient_axes") and self.repair_controls["reorient_axes"].isChecked():
            self.facing_preset.setCurrentIndex(max(0, self.facing_preset.findData(0.0)))
        self._form_changed()

    def set_placement_analysis(self, snapshot: object) -> None:
        self._placement_snapshot = snapshot
        dimensions = tuple(getattr(snapshot, "dimensions", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0))
        height = float(dimensions[2]) if len(dimensions) >= 3 else 0.0
        self.height_display.setText(f"{height:.4g}" if height else "Not measured")
        self.human_scale.set_creature_height(height)
        lowest = getattr(snapshot, "lowest_contact_height", None)
        root = getattr(snapshot, "root_height", None)
        runtime_height = float(getattr(snapshot, "runtime_height_offset", 0.0) or 0.0)
        runtime_source = str(getattr(snapshot, "runtime_height_source", "") or "")
        if runtime_height > 1.0e-6:
            self.runtime_height_display.setText(
                f"{runtime_height:.6g} from {runtime_source or 'the source root joint'} — applied automatically"
            )
        else:
            self.runtime_height_display.setText("No separate runtime correction is needed for this rig")
        if lowest is None:
            self.ground_warning.setText("No finite mesh contact height could be measured.")
        else:
            final = float(lowest) + float(self.ground_offset.value()) + float(self.pivot_z.value())
            self.ground_warning.setText(
                f"Lowest mesh contact: {float(lowest):.5g}; current exported contact: {final:.5g}. "
                + ("Use Place contacts on ground." if abs(final) > 1.0e-3 else "Contact is on the floor.")
                + (f" Root height: {float(root):.5g}." if root is not None else "")
                + (
                    f" KOTOR runtime correction: {runtime_height:.5g} from {runtime_source}."
                    if runtime_height > 1.0e-6 else ""
                )
            )

    def _detect_lowest_contacts(self) -> None:
        snapshot = self._placement_snapshot
        if snapshot is None:
            QtWidgets.QMessageBox.information(self, "Import first", "Import the FBX before detecting contacts.")
            return
        names = [str(getattr(node, "name", "") or "") for node in getattr(snapshot, "nodes", ())]
        suggested = [
            name for name in names
            if any(token in name.casefold() for token in ("foot", "claw", "hoof", "paw", "toe"))
        ]
        if suggested:
            self.contact_nodes.setText(", ".join(suggested))
        self.set_placement_analysis(snapshot)

    def _place_contacts_on_ground(self) -> None:
        snapshot = self._placement_snapshot
        lowest = getattr(snapshot, "lowest_contact_height", None) if snapshot is not None else None
        if lowest is None:
            QtWidgets.QMessageBox.information(self, "No contact measurement", "Import a mesh with finite vertices first.")
            return
        self.ground_offset.setValue(-float(lowest) - float(self.pivot_z.value()))
        self.set_placement_analysis(snapshot)

    def _confirm_suggested_mappings(self) -> None:
        used_aliases = {
            mapping.exported_name.casefold()
            for mapping in self.project.animation_mappings
            if mapping.assignment != "unassigned" and mapping.exported_name
        }
        changed = 0
        for mapping in self.project.animation_mappings:
            if mapping.assignment == "unassigned":
                category, alias = suggest_semantic_mapping(mapping.source_name)
                if alias and alias.casefold() not in used_aliases:
                    mapping.assignment = "vanilla_behavior_alias"
                    mapping.exported_name = alias
                    mapping.loop = category in {"primary_idle", "secondary_idle", "walk", "run", "combat_ready", "dead_pose"}
                    used_aliases.add(alias.casefold())
            if mapping.assignment == "vanilla_behavior_alias" and mapping.exported_name:
                mapping.confirmed = True
                changed += 1
        if changed:
            self.set_animation_inventory(self._animation_inventory)
        self.project_status.setText(f"Confirmed {changed} suggested behavior mapping(s).")
        self._refresh_gameplay_animation_warnings()
        self._form_changed()

    def _animation_row_selected(self, row: int, _column: int, _previous_row: int, _previous_column: int) -> None:
        if row < 0 or self.animation_table.item(row, 0) is None:
            return
        source_name = self.animation_table.item(row, 0).text()
        mapping = next((value for value in self.project.animation_mappings if value.source_name == source_name), None)
        if mapping is None:
            return
        self._loading = True
        try:
            self.loop_trim.setChecked(mapping.loop)
            self.playback_speed.setValue(mapping.playback_speed)
            self.trim_start.setValue(mapping.trim_start)
            self.trim_end.setValue(mapping.trim_end or 0.0)
            self.retime_duration.setValue(mapping.retime_duration or 0.0)
            self.transition_time.setValue(mapping.transition_time)
            root_index = {"in_place": 0, "keep": 1, "analysis_only": 2}.get(mapping.root_motion, 0)
            self.root_motion.setCurrentIndex(root_index)
            bake_index = 1 if mapping.bake_rate >= 59.0 else 0 if mapping.bake_rate > 0 else 2
            self.bake_rate.setCurrentIndex(bake_index)
        finally:
            self._loading = False
        for label in self.animation_preview_labels.values():
            label.setText(f"Preparing {source_name}…")
        self.animationPreviewRequested.emit(source_name)

    def _animation_preparation_changed(self, *_args: Any) -> None:
        if self._loading:
            return
        row = self.animation_table.currentRow()
        if row < 0 or self.animation_table.item(row, 0) is None:
            return
        source_name = self.animation_table.item(row, 0).text()
        mapping = next((value for value in self.project.animation_mappings if value.source_name == source_name), None)
        if mapping is None:
            return
        mapping.loop = self.loop_trim.isChecked()
        mapping.playback_speed = float(self.playback_speed.value())
        mapping.trim_start = float(self.trim_start.value())
        mapping.trim_end = float(self.trim_end.value()) or None
        mapping.retime_duration = float(self.retime_duration.value()) or None
        mapping.root_motion = ("in_place", "keep", "analysis_only")[self.root_motion.currentIndex()]
        mapping.bake_rate = (30.0, 60.0, 0.0)[self.bake_rate.currentIndex()]
        mapping.transition_time = float(self.transition_time.value())
        self.animation_diagnostics.setText(
            "Preparation changed. Preview again to compare the source motion with the converted in-game controller result."
        )
        self._form_changed()

    def set_animation_preview_status(self, before: str, after: str) -> None:
        self.animation_preview_labels["before"].setText(before)
        self.animation_preview_labels["after"].setText(after)

    def set_animation_preview_models(self, before: object, after: object, texture_dir: str = "") -> None:
        self.animation_preview_viewports["before"].load_model(before, texture_dir)
        self.animation_preview_viewports["after"].load_model(after, texture_dir)
        for viewport in self.animation_preview_viewports.values():
            viewport.reset_camera()

    def _material_cell_double_clicked(self, row: int, column: int) -> None:
        if column != 1 or row < 0:
            return
        current = self.material_table.item(row, column).text() if self.material_table.item(row, column) else ""
        value, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose source texture",
            current or self.texture_folder.path(),
            "Textures (*.tga *.tpc *.png *.dds *.bmp *.jpg *.jpeg *.tif *.tiff)",
        )
        if value:
            self.material_table.item(row, column).setText(value)

    def _material_table_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._loading or item.column() not in {1, 2}:
            return
        self._sync_material_row(item.row())

    def _sync_material_row(self, row: int) -> None:
        if row < 0 or self.material_table.item(row, 0) is None:
            return
        name = self.material_table.item(row, 0).text()
        assignment = next((value for value in self.project.material_assignments if value.material_name == name), None)
        if assignment is None:
            assignment = MaterialAssignment(material_name=name)
            self.project.material_assignments.append(assignment)
        assignment.source_texture = self.material_table.item(row, 1).text().strip()
        assignment.texture_resref = self.material_table.item(row, 2).text().strip().lower()
        wrap = self.material_table.cellWidget(row, 4)
        alpha = self.material_table.cellWidget(row, 5)
        output = self.material_table.cellWidget(row, 6)
        if isinstance(wrap, QtWidgets.QComboBox):
            assignment.wrap_mode = str(wrap.currentData() or "repeat")
        if isinstance(alpha, QtWidgets.QComboBox):
            assignment.alpha_mode = str(alpha.currentData() or "opaque")
        if isinstance(output, QtWidgets.QComboBox):
            assignment.output_format = str(output.currentData() or "TGA")
        source = self.project.resolve_path(assignment.source_texture) if assignment.source_texture else Path()
        assignment.source_sha256 = sha256_file(source) if source.is_file() else ""
        self._form_changed()

    def _material_row_selected(self, row: int, _column: int, _previous_row: int, _previous_column: int) -> None:
        if row < 0 or row >= len(self._material_snapshots):
            return
        self._loading = True
        try:
            name = self.material_table.item(row, 0).text()
            assignment = next((value for value in self.project.material_assignments if value.material_name == name), None)
            if assignment:
                self.output_tga.setChecked(assignment.output_format.upper() == "TGA")
                self.output_tpc.setChecked(assignment.output_format.upper() == "TPC")
                self.preserve_repeat.setChecked(assignment.wrap_mode == "repeat")
                self.orient_for_kotor.setChecked(assignment.flip_vertical_for_kotor)
                self.txi_editor.setPlainText(assignment.txi)
        finally:
            self._loading = False
        self._update_material_previews(row)

    def _material_controls_changed(self, *_args: Any) -> None:
        if self._loading:
            return
        row = self.material_table.currentRow()
        if row < 0:
            return
        output = self.material_table.cellWidget(row, 6)
        if isinstance(output, QtWidgets.QComboBox):
            output.setCurrentIndex(output.findData("TPC" if self.output_tpc.isChecked() else "TGA"))
        wrap = self.material_table.cellWidget(row, 4)
        if isinstance(wrap, QtWidgets.QComboBox):
            wrap.setCurrentIndex(wrap.findData("repeat" if self.preserve_repeat.isChecked() else "clamp"))
        name = self.material_table.item(row, 0).text()
        assignment = next((value for value in self.project.material_assignments if value.material_name == name), None)
        if assignment:
            assignment.txi = self.txi_editor.toPlainText()
            assignment.flip_vertical_for_kotor = self.orient_for_kotor.isChecked()
        self._sync_material_row(row)
        self._update_material_previews(row)

    def _update_material_previews(self, row: int) -> None:
        snapshot = self._material_snapshots[row]
        source_text = self.material_table.item(row, 1).text().strip()
        source = self.project.resolve_path(source_text) if source_text else Path()
        pixmap = QtGui.QPixmap()
        if source.is_file() and source.suffix.casefold() != ".tpc":
            try:
                from PIL import Image

                with Image.open(source) as image:
                    rgba = image.convert("RGBA")
                    data = rgba.tobytes("raw", "RGBA")
                    qimage = QtGui.QImage(data, rgba.width, rgba.height, rgba.width * 4, QtGui.QImage.Format_RGBA8888).copy()
                    pixmap = QtGui.QPixmap.fromImage(qimage)
            except Exception:
                pixmap = QtGui.QPixmap()
        if not pixmap.isNull():
            source_scaled = pixmap.scaled(320, 180, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            kotor_pixmap = pixmap.transformed(QtGui.QTransform().scale(1.0, -1.0)) \
                if self.orient_for_kotor.isChecked() else pixmap
            kotor_scaled = kotor_pixmap.scaled(320, 180, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.material_preview_labels["source"].setPixmap(source_scaled)
            self.material_preview_labels["source"].setText("")
            self.material_preview_labels["kotor"].setPixmap(kotor_scaled)
            self.material_preview_labels["kotor"].setText("")
        else:
            self.material_preview_labels["source"].setText("Choose a readable source texture")
            self.material_preview_labels["kotor"].setText("KOTOR copy will appear after conversion")
        checker = QtGui.QPixmap(320, 180)
        checker.fill(self.palette().color(QtGui.QPalette.Base))
        painter = QtGui.QPainter(checker)
        light = self.palette().color(QtGui.QPalette.Light)
        mid = self.palette().color(QtGui.QPalette.Midlight)
        tile = 20
        for y in range(0, checker.height(), tile):
            for x in range(0, checker.width(), tile):
                painter.fillRect(x, y, tile, tile, light if (x // tile + y // tile) % 2 else mid)
        painter.setPen(QtGui.QPen(self.palette().color(QtGui.QPalette.Highlight), 1))
        uvs = list(getattr(snapshot, "uvs", ()) or ())
        for u, v in uvs[:5000]:
            x = int((float(u) % 1.0) * (checker.width() - 1))
            y = int((1.0 - (float(v) % 1.0)) * (checker.height() - 1))
            painter.drawPoint(x, y)
        painter.end()
        self.material_preview_labels["uv"].setPixmap(checker)
        self.material_preview_labels["uv"].setText("")

    def set_install_preview(self, payload: Mapping[str, Any]) -> None:
        self.install_file_list.clear()
        self._install_preview_id = str(payload.get("preview_id") or "")
        for record in payload.get("files") or ():
            status = str(record.get("status") or "")
            action = str(record.get("action") or "write")
            backup_text = "Not needed"
            if status == "replace_with_backup":
                backup_text = "Created before replacement"
            elif action == "remove":
                backup_text = "Created before cache is cleared; restored by Undo"
            item = QtWidgets.QTreeWidgetItem((
                str(record.get("name") or ""),
                str(record.get("target") or ""),
                "No" if status == "new" else "Yes",
                backup_text,
            ))
            self.install_file_list.addTopLevelItem(item)
        self.install_button.setEnabled(bool(payload.get("ok") and self._install_preview_id))
        if payload.get("ok"):
            messages = [str(value) for value in payload.get("messages") or () if str(value).strip()]
            self.install_status.setText(
                "Preview ready. No files have changed. Confirm Install with backup to stage exactly this list."
                + ("\n" + "\n".join(messages[1:]) if len(messages) > 1 else "")
            )
        else:
            self.install_status.setText(str(payload.get("error") or "Install preview failed."))

    def _character_type_changed(self, index: int) -> None:
        if self.character_type.itemData(index) == "native_kotor_character":
            self.character_type.blockSignals(True)
            self.character_type.setCurrentIndex(0)
            self.character_type.blockSignals(False)
            self.nativeBuilderRequested.emit()

    def _form_changed(self, *_args: Any) -> None:
        if self._loading:
            return
        self._install_preview_id = ""
        if hasattr(self, "install_button"):
            self.install_button.setEnabled(False)
        self._sync_project_from_form()
        self._autosave.start()
        self.projectChanged.emit(self.project)

    def _sync_project_from_form(self) -> CustomRiggedCharacterProject:
        if not hasattr(self, "creature_name"):
            return self.project
        self.project.creature_name = self.creature_name.text().strip()
        self.project.resource_name = self.resource_name.text().strip().lower()
        self.project.target_game = str(self.target_game.currentData() or "K2")
        if hasattr(self, "secondary_motion_editor"):
            self.secondary_motion_editor.set_target_game(self.project.target_game)
            self.project.secondary_motion = self.secondary_motion_editor.profile()
        source_path = self.source_fbx.path()
        source_hash = self.project.primary_fbx.sha256
        if source_path != self.project.primary_fbx.path:
            source_hash = ""
        self.project.primary_fbx = SourceAsset(source_path, source_hash, "primary_fbx", True)
        external_paths = [
            str(self.external_animation_list.item(index).data(QtCore.Qt.UserRole) or self.external_animation_list.item(index).text())
            for index in range(self.external_animation_list.count())
        ]
        animation_folder = Path(self.animation_folder.path())
        if animation_folder.is_dir():
            external_paths.extend(str(path) for path in sorted(animation_folder.glob("*.fbx")))
        old_external = {asset.path: asset for asset in self.project.external_animation_assets}
        self.project.external_animation_assets = [
            SourceAsset(path, old_external.get(path, SourceAsset()).sha256, "external_animation", False)
            for path in dict.fromkeys(external_paths)
            if path and path != source_path
        ]
        self.project.texture_folder = self.texture_folder.path()
        self.project.output_project_folder = self.output_folder.path()
        self.project.build_destination = str(Path(self.output_folder.path()) / "build") if self.output_folder.path() else ""
        selected_root = str(self.skeleton_root.currentData() or "")
        if selected_root:
            self.project.selected_skeleton_root = selected_root
        self.project.global_scale = float(self.global_scale.value())
        self.project.ground_offset = float(self.ground_offset.value())
        facing = self.facing_preset.currentData()
        if facing is not None:
            self.project.forward_rotation_degrees = float(facing)
        self.project.pivot_offset = [
            float(self.pivot_x.value()), float(self.pivot_y.value()), float(self.pivot_z.value())
        ]
        self.project.ground_contact_nodes = [
            value.strip() for value in self.contact_nodes.text().split(",") if value.strip()
        ]
        self.project.appearance_settings["label"] = self.appearance_name.text().strip()
        self.project.appearance_settings["personal_space"] = float(self.collision_size.value())
        self.project.appearance_settings["collision_space"] = float(self.collision_size.value())
        self.project.utc_settings["display_name"] = self.display_name.text().strip()
        self.project.utc_settings["resref"] = self.project.resource_name
        self.project.utc_settings["soundset"] = int(self.soundset.text()) if self.soundset.text().strip().isdigit() else 0
        self.project.utc_settings["level"] = int(self.level.value())
        self.project.utc_settings["hit_points"] = int(self.hit_points.value())
        self.project.utc_settings["faction_id"] = {"Neutral": 5, "Hostile": 1, "Friendly": 2}.get(self.faction.currentText(), 5)
        self.project.gameplay_settings["behavior_preset"] = {
            "Installed character template": "installed_utc_template",
            "Passive creature": "passive_creature",
            "Hostile melee creature": "hostile_melee_creature",
            "Ranged creature": "ranged_creature",
            "Stationary creature": "stationary_creature",
            "Flying / hovering creature": "flying_hovering_creature",
            "Custom scripts": "custom_scripts",
        }.get(self.behavior_preset.currentText(), "passive_creature")
        self.project.gameplay_settings["faction"] = self.faction.currentText()
        self.project.gameplay_settings["movement_rate"] = self.movement_rate.currentText()
        self.project.gameplay_settings["perception_range"] = float(self.perception_range.value())
        self.project.gameplay_settings["soundset"] = self.soundset.text().strip()
        self.project.gameplay_settings["generate_utc"] = self.generate_utc.isChecked()
        self.project.gameplay_settings["generate_spawn_script"] = self.generate_spawn.isChecked()
        self.project.gameplay_settings["prepare_module_placement"] = self.prepare_module_placement.isChecked()
        self.project.gameplay_settings["replace_test_placement"] = self.replace_test_placement.isChecked()
        self.project.gameplay_settings.setdefault("test_module_resref", "plcaa")
        self.project.gameplay_settings.setdefault(
            "test_placement",
            {"position": [26.0, 30.0, 0.0], "bearing": 3.1415927},
        )
        self.project.behavior_profile["inherit_template_combat_stats"] = self.inherit_template_stats.isChecked()
        self.project.metadata["game_directory"] = self.game_directory.path()
        self.project.skin_repair_settings.update({
            key: box.isChecked()
            for key, box in self.repair_controls.items()
            if key in self.project.skin_repair_settings
        })
        self.project.custom_animation_registrations = [
            CustomAnimationRegistration(
                name=mapping.exported_name,
                animation_id=mapping.runtime_id,
                source_clip=mapping.source_name,
                namespace=(mapping.exported_name.split("_", 1)[0] if "_" in mapping.exported_name else self.project.resource_name),
            )
            for mapping in self.project.animation_mappings
            if mapping.assignment == "custom_runtime_animation" and mapping.exported_name and mapping.confirmed
        ]
        recent = [
            source_path, self.animation_folder.path(), self.texture_folder.path(),
            self.output_folder.path(), self.game_directory.path(),
            *[
                str(self.project.resolve_path(value.source_path).parent)
                for value in self.project.creature_sound_cues
                if value.source_path
            ],
        ]
        self.project.recent_paths = list(dict.fromkeys(
            [value for value in recent if value] + list(self.project.recent_paths)
        ))[:20]
        for key, value in (
            ("last_source", source_path), ("last_animation_folder", self.animation_folder.path()),
            ("last_texture_folder", self.texture_folder.path()), ("last_output_folder", self.output_folder.path()),
            ("last_game_directory", self.game_directory.path()),
        ):
            if value:
                self._settings.setValue(key, value)
        self._refresh_build_summary()
        return self.project

    def _load_project_into_form(self) -> None:
        self._loading = True
        try:
            def opened_path(value: str, fallback: object = "") -> str:
                chosen = str(value or fallback or "")
                return str(self.project.resolve_path(chosen)) if chosen else ""

            self.creature_name.setText(self.project.creature_name)
            self.resource_name.setText(self.project.resource_name)
            game_index = self.target_game.findData(self.project.target_game)
            self.target_game.setCurrentIndex(max(0, game_index))
            if hasattr(self, "secondary_motion_editor"):
                self.secondary_motion_editor.set_profile(self.project.secondary_motion)
            self.source_fbx.set_path(opened_path(
                self.project.primary_fbx.path, self._settings.value("last_source", "")
            ))
            self.animation_folder.set_path(str(self._settings.value("last_animation_folder", "")))
            self.external_animation_list.clear()
            for asset in self.project.external_animation_assets:
                resolved_asset = opened_path(asset.path)
                item = QtWidgets.QListWidgetItem(Path(resolved_asset).name or resolved_asset)
                item.setToolTip(resolved_asset)
                item.setData(QtCore.Qt.UserRole, resolved_asset)
                self.external_animation_list.addItem(item)
            self.texture_folder.set_path(opened_path(
                self.project.texture_folder, self._settings.value("last_texture_folder", "")
            ))
            self.output_folder.set_path(opened_path(
                self.project.output_project_folder, self._settings.value("last_output_folder", "")
            ))
            self.game_directory.set_path(str(self.project.metadata.get("game_directory") or self._settings.value("last_game_directory", "")))
            self.global_scale.setValue(self.project.global_scale)
            self.ground_offset.setValue(self.project.ground_offset)
            if self.project.runtime_height_offset > 1.0e-6:
                self.runtime_height_display.setText(
                    f"{self.project.runtime_height_offset:.6g} from "
                    f"{self.project.runtime_height_source or 'the source root joint'} — applied automatically"
                )
            else:
                self.runtime_height_display.setText("Import the FBX to detect this automatically")
            self.pivot_x.setValue(float((self.project.pivot_offset + [0.0, 0.0, 0.0])[0]))
            self.pivot_y.setValue(float((self.project.pivot_offset + [0.0, 0.0, 0.0])[1]))
            self.pivot_z.setValue(float((self.project.pivot_offset + [0.0, 0.0, 0.0])[2]))
            facing_index = self.facing_preset.findData(float(self.project.forward_rotation_degrees))
            self.facing_preset.setCurrentIndex(facing_index if facing_index >= 0 else 0)
            self.contact_nodes.setText(", ".join(self.project.ground_contact_nodes))
            self.appearance_name.setText(str(self.project.appearance_settings.get("label") or ""))
            self.display_name.setText(str(self.project.utc_settings.get("display_name") or ""))
            self.soundset.setText(str(self.project.gameplay_settings.get("soundset") or ""))
            self.collision_size.setValue(float(self.project.appearance_settings.get("personal_space", 1.0) or 1.0))
            self.perception_range.setValue(float(self.project.gameplay_settings.get("perception_range", 10.0) or 10.0))
            self.level.setValue(int(self.project.utc_settings.get("level", 5) or 5))
            self.hit_points.setValue(int(self.project.utc_settings.get("hit_points", 45) or 45))
            behavior_labels = {
                "installed_utc_template": "Installed character template",
                "passive_creature": "Passive creature",
                "hostile_melee_creature": "Hostile melee creature",
                "ranged_creature": "Ranged creature",
                "stationary_creature": "Stationary creature",
                "flying_hovering_creature": "Flying / hovering creature",
                "custom_scripts": "Custom scripts",
            }
            behavior_text = behavior_labels.get(
                str(self.project.gameplay_settings.get("behavior_preset") or ""),
                str(self.project.gameplay_settings.get("behavior_preset") or "Passive creature"),
            )
            behavior_index = self.behavior_preset.findText(behavior_text)
            self.behavior_preset.setCurrentIndex(max(0, behavior_index))
            faction_index = self.faction.findText(str(self.project.gameplay_settings.get("faction") or "Neutral"))
            self.faction.setCurrentIndex(max(0, faction_index))
            movement_index = self.movement_rate.findText(str(self.project.gameplay_settings.get("movement_rate") or "Default"))
            self.movement_rate.setCurrentIndex(max(0, movement_index))
            self.generate_utc.setChecked(bool(self.project.gameplay_settings.get("generate_utc", True)))
            self.generate_spawn.setChecked(bool(self.project.gameplay_settings.get("generate_spawn_script", False)))
            self.prepare_module_placement.setChecked(bool(self.project.gameplay_settings.get("prepare_module_placement", False)))
            self.replace_test_placement.setChecked(bool(self.project.gameplay_settings.get("replace_test_placement", False)))
            self.replace_test_placement.setEnabled(self.prepare_module_placement.isChecked())
            self.inherit_template_stats.setChecked(
                bool(self.project.behavior_profile.get("inherit_template_combat_stats", True))
            )
            for key, box in self.repair_controls.items():
                if key in self.project.skin_repair_settings:
                    box.setChecked(bool(self.project.skin_repair_settings[key]))
            self.set_skeleton_root_choices(
                list(self.project.last_import_summary.get("available_skeleton_roots") or ()),
                selected=self.project.selected_skeleton_root,
                selection_required=bool(self.project.last_import_summary.get("skeleton_selection_required")),
            )
            self._apply_import_summary(self.project.last_import_summary)
            self._refresh_creature_sound_table()
            self._refresh_behavior_hook_table()
            self._load_behavior_hook_editor()
            template = dict(self.project.behavior_profile.get("template_snapshot") or {})
            if template:
                self.behavior_template_details.setText(
                    f"Using {template.get('display_name') or template.get('resref')} "
                    f"({template.get('resref')}.utc). Read installed templates to compare or change it."
                )
            self._behavior_inheritance_changed()
            self._refresh_build_summary()
        finally:
            self._loading = False

    def _request_import(self) -> None:
        project = self._sync_project_from_form()
        source = project.resolve_path(project.primary_fbx.path)
        if not source.is_file():
            QtWidgets.QMessageBox.warning(self, "Source FBX not found", "Choose an existing FBX file before importing.")
            return
        try:
            project.primary_fbx.sha256 = sha256_file(source)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Could not read source FBX", str(exc))
            return
        self.project_status.setText("Source recorded read-only; waiting for import results")
        self.importRequested.emit(project)
        self._autosave.start()

    def set_import_summary(self, summary: dict[str, Any]) -> None:
        self.project.last_import_summary = dict(summary or {})
        self._apply_import_summary(summary)
        # Import/autosave may have created a previously missing output folder.
        # Refresh the visible state so users are not left with a stale red flag.
        self.output_folder._refresh_status()
        self._autosave.start()

    def set_preview_model(self, model: object, texture_dir: str = "") -> None:
        """Show the imported candidate in the actual shared viewport widgets."""

        for viewport in (self.rig_viewport, self.ground_viewport):
            viewport.load_model(model, texture_dir)

    def set_hierarchy_rows(self, nodes: list[object]) -> None:
        self.hierarchy_tree.clear()
        items: dict[str, QtWidgets.QTreeWidgetItem] = {}
        pending = list(nodes or [])
        while pending:
            progressed = False
            for node in list(pending):
                name = str(getattr(node, "name", "") or "")
                parent_name = str(getattr(node, "parent", "") or "")
                if parent_name and parent_name not in items:
                    continue
                item = QtWidgets.QTreeWidgetItem((name, "Yes" if getattr(node, "exported", True) else "No"))
                if parent_name:
                    items[parent_name].addChild(item)
                else:
                    self.hierarchy_tree.addTopLevelItem(item)
                items[name] = item
                pending.remove(node)
                progressed = True
            if not progressed:
                for node in pending:
                    item = QtWidgets.QTreeWidgetItem((str(getattr(node, "name", "") or ""), "Review"))
                    self.hierarchy_tree.addTopLevelItem(item)
                break
        self.hierarchy_tree.expandToDepth(1)

    def set_animation_inventory(self, rows: list[dict[str, Any]]) -> None:
        self._animation_inventory = [dict(value) for value in (rows or [])]
        self.animation_table.setRowCount(0)
        mappings = {value.source_name: value for value in self.project.animation_mappings}
        for source in self._animation_inventory:
            name = str(source.get("name") or "")
            mapping = mappings.get(name)
            row = self.animation_table.rowCount()
            self.animation_table.insertRow(row)
            duration = source.get("duration_seconds", source.get("duration"))
            frame_count = source.get("frame_count")
            fps = source.get("fps")
            values = (
                name,
                f"{float(duration):.3f} s" if duration is not None else "—",
                f"{frame_count} @ {float(fps):.3g} fps" if frame_count is not None and fps else str(frame_count or "—"),
                "Yes" if mapping and mapping.loop else "No",
                "Yes" if bool(source.get("root_motion")) else "No",
                str(source.get("animated_bone_count") or "—"),
                "",
            )
            for column, value in enumerate(values):
                self.animation_table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
            preview_button = QtWidgets.QPushButton("Preview")
            preview_button.clicked.connect(lambda _checked=False, key=name: self.animationPreviewRequested.emit(key))
            self.animation_table.setCellWidget(row, 6, preview_button)
            behavior = QtWidgets.QComboBox()
            for label, assignment, alias in (
                ("Unassigned", "unassigned", ""),
                ("Idle (KOTOR already requests this)", "vanilla_behavior_alias", "cpause1"),
                ("Secondary idle", "vanilla_behavior_alias", "cpause2"),
                ("Walk", "vanilla_behavior_alias", "cwalk"),
                ("Run", "vanilla_behavior_alias", "crun"),
                ("Monster attack 1 (Zakkeg)", "vanilla_behavior_alias", "m0a1"),
                ("Monster attack 2 (Zakkeg)", "vanilla_behavior_alias", "m0a2"),
                ("Generic monster attack 1", "vanilla_behavior_alias", "g0a1"),
                ("Generic monster attack 2", "vanilla_behavior_alias", "g0a2"),
                ("Damage reaction", "vanilla_behavior_alias", "cdamages"),
                ("Dodge", "vanilla_behavior_alias", "cdodgeg"),
                ("Combat ready loop", "vanilla_behavior_alias", "creadyr"),
                ("Turn-to-walk ready", "vanilla_behavior_alias", "creadyrtw"),
                ("Injured walk", "vanilla_behavior_alias", "cwalkinj"),
                ("Knockdown", "vanilla_behavior_alias", "ckdbck"),
                ("Knockdown loop", "vanilla_behavior_alias", "ckdbcklp"),
                ("Get up", "vanilla_behavior_alias", "cgustandb"),
                ("Die", "vanilla_behavior_alias", "cdie"),
                ("Dead pose loop", "vanilla_behavior_alias", "cdead"),
                ("Roar / taunt", "vanilla_behavior_alias", "ctaunt"),
                ("Victory", "vanilla_behavior_alias", "cvictory"),
                ("Custom runtime action", "custom_runtime_animation", ""),
            ):
                behavior.addItem(label, (assignment, alias))
            selected = 0
            if mapping:
                for index in range(behavior.count()):
                    assignment, alias = behavior.itemData(index)
                    if assignment == mapping.assignment and (not alias or alias == mapping.exported_name):
                        selected = index
                        break
            behavior.setCurrentIndex(selected)
            exported = QtWidgets.QLineEdit(mapping.exported_name if mapping else "")
            exported.setMaxLength(16)
            runtime_id = QtWidgets.QSpinBox()
            runtime_id.setRange(0, 2147483647)
            if mapping and mapping.runtime_id is not None:
                runtime_id.setValue(mapping.runtime_id)
            runtime_id.setSpecialValueText("Not needed")
            status = QtWidgets.QTableWidgetItem("Confirm mapping" if mapping and not mapping.confirmed else "Ready")
            self.animation_table.setCellWidget(row, 7, behavior)
            self.animation_table.setCellWidget(row, 8, exported)
            self.animation_table.setCellWidget(row, 9, runtime_id)
            self.animation_table.setItem(row, 10, status)
            behavior.currentIndexChanged.connect(
                lambda _index, n=name, b=behavior, e=exported, i=runtime_id, s=status:
                self._animation_mapping_changed(n, b, e, i, s)
            )
            exported.textChanged.connect(
                lambda _text, n=name, b=behavior, e=exported, i=runtime_id, s=status:
                self._animation_mapping_changed(n, b, e, i, s)
            )
            runtime_id.valueChanged.connect(
                lambda _value, n=name, b=behavior, e=exported, i=runtime_id, s=status:
                self._animation_mapping_changed(n, b, e, i, s)
            )
        self._refresh_gameplay_animation_warnings()

    def _animation_mapping_changed(
        self,
        source_name: str,
        behavior: QtWidgets.QComboBox,
        exported: QtWidgets.QLineEdit,
        runtime_id: QtWidgets.QSpinBox,
        status: QtWidgets.QTableWidgetItem,
    ) -> None:
        assignment, alias = behavior.currentData()
        mapping = next((value for value in self.project.animation_mappings if value.source_name == source_name), None)
        if mapping is None:
            mapping = AnimationMapping(source_name=source_name)
            self.project.animation_mappings.append(mapping)
        mapping.assignment = assignment
        if assignment == "custom_runtime_animation" and not exported.text().strip():
            exported.blockSignals(True)
            exported.setText(namespaced_animation_name(self.project.resource_name, source_name))
            exported.blockSignals(False)
        if assignment == "custom_runtime_animation" and runtime_id.value() == 0:
            occupied = [
                value.runtime_id for value in self.project.animation_mappings
                if value is not mapping and value.runtime_id is not None
            ]
            runtime_id.blockSignals(True)
            runtime_id.setValue(allocate_animation_id(occupied))
            runtime_id.blockSignals(False)
        if alias and exported.text() != alias:
            exported.blockSignals(True)
            exported.setText(alias)
            exported.blockSignals(False)
        mapping.exported_name = exported.text().strip()
        mapping.runtime_id = runtime_id.value() if assignment == "custom_runtime_animation" and runtime_id.value() else None
        mapping.confirmed = assignment == "unassigned" or bool(mapping.exported_name)
        runtime_id.setEnabled(assignment == "custom_runtime_animation")
        exported.setReadOnly(bool(alias))
        status.setText("Ready" if mapping.confirmed else "Name or ID required")
        self._refresh_gameplay_animation_warnings()
        self._form_changed()

    def _refresh_gameplay_animation_warnings(self) -> None:
        if not hasattr(self, "behavior_warnings"):
            return
        aliases = {
            mapping.exported_name.casefold()
            for mapping in self.project.animation_mappings
            if mapping.assignment == "vanilla_behavior_alias" and mapping.confirmed
        }
        messages: list[str] = []
        if not aliases.intersection({"m0a1", "m0a2", "g0a1", "g0a2"}):
            messages.append("Choose at least one attack animation before the in-game combat test.")
        if "cdamages" not in aliases:
            messages.append("Choose a damage reaction so hits have visible feedback.")
        if "cdie" not in aliases:
            messages.append("Choose a death animation before the in-game combat test.")
        if not messages:
            messages.append("Combat animation essentials are assigned and ready for validation.")
        self.behavior_warnings.clear()
        self.behavior_warnings.addItems(messages)

    def set_material_inventory(self, rows: list[object]) -> None:
        self._loading = True
        self._material_snapshots = list(rows or [])
        try:
            self.material_table.setRowCount(0)
            assignments = {value.material_name: value for value in self.project.material_assignments}
            for material in self._material_snapshots:
                row = self.material_table.rowCount()
                self.material_table.insertRow(row)
                name = str(getattr(material, "material_name", ""))
                assignment = assignments.get(name)
                if assignment is None:
                    assignment = MaterialAssignment(
                        material_name=name,
                        texture_resref=str(getattr(material, "texture_resref", "")),
                        source_texture=str(getattr(material, "source_texture", "")),
                        output_format="TGA",
                        wrap_mode=str(getattr(material, "wrap_mode", "repeat")),
                        alpha_mode=str(getattr(material, "alpha_mode", "opaque")),
                    )
                    self.project.material_assignments.append(assignment)
                    assignments[name] = assignment
                uvs = list(getattr(material, "uvs", ()) or ())
                uv_range = "—"
                if uvs:
                    us = [value[0] for value in uvs]
                    vs = [value[1] for value in uvs]
                    uv_range = f"U {min(us):.3g}…{max(us):.3g}; V {min(vs):.3g}…{max(vs):.3g}"
                source_value = assignment.source_texture or str(getattr(material, "source_texture", ""))
                for column, value in enumerate((name, source_value, assignment.texture_resref, uv_range)):
                    item = QtWidgets.QTableWidgetItem(str(value))
                    if column in {0, 3}:
                        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                    self.material_table.setItem(row, column, item)
                wrap = QtWidgets.QComboBox()
                wrap.addItem("Repeat / tile", "repeat")
                wrap.addItem("Clamp to edge", "clamp")
                wrap.setCurrentIndex(max(0, wrap.findData(assignment.wrap_mode)))
                alpha = QtWidgets.QComboBox()
                alpha.addItem("Opaque", "opaque")
                alpha.addItem("Cutout / punch-through", "cutout")
                alpha.addItem("Blend", "blend")
                alpha.setCurrentIndex(max(0, alpha.findData(assignment.alpha_mode)))
                output = QtWidgets.QComboBox()
                output.addItem("TGA + optional TXI", "TGA")
                output.addItem("TPC", "TPC")
                output.setCurrentIndex(max(0, output.findData(assignment.output_format.upper())))
                self.material_table.setCellWidget(row, 4, wrap)
                self.material_table.setCellWidget(row, 5, alpha)
                self.material_table.setCellWidget(row, 6, output)
                source_path = self.project.resolve_path(source_value) if source_value else Path()
                status = "Ready" if assignment.texture_resref and source_path.is_file() else "Choose source texture"
                self.material_table.setItem(row, 7, QtWidgets.QTableWidgetItem(status))
                wrap.currentIndexChanged.connect(lambda _index, r=row: self._sync_material_row(r))
                alpha.currentIndexChanged.connect(lambda _index, r=row: self._sync_material_row(r))
                output.currentIndexChanged.connect(lambda _index, r=row: self._sync_material_row(r))
        finally:
            self._loading = False
        if self.material_table.rowCount():
            self.material_table.setCurrentCell(0, 0)

    def _apply_import_summary(self, summary: dict[str, Any]) -> None:
        for key, label in self.import_summary_labels.items():
            label.setText(str((summary or {}).get(key, "—")))

    def set_validation_results(self, issues: list[dict[str, Any]], *, build_ready: bool) -> None:
        self.validation_tree.clear()
        for issue in issues:
            item = QtWidgets.QTreeWidgetItem((
                str(issue.get("severity") or "Information").title(),
                str(issue.get("message") or ""),
                str(issue.get("why") or ""),
                str(issue.get("automatic_fix") or "Manual review"),
            ))
            item.setData(0, QtCore.Qt.UserRole, dict(issue))
            self.validation_tree.addTopLevelItem(item)
        self.build_button.setEnabled(bool(build_ready))
        self.validation_status.setText("Ready to build" if build_ready else "Needs attention")
        self.project.last_validation_result = {"issues": issues, "build_ready": bool(build_ready)}
        self._autosave.start()

    def _refresh_build_summary(self) -> None:
        if not hasattr(self, "build_summary"):
            return
        summary = self.project.last_import_summary
        self.build_summary["resref"].setText(self.project.resource_name or "—")
        self.build_summary["nodes"].setText(str(summary.get("bone_count", "—")))
        self.build_summary["meshes"].setText(str(summary.get("mesh_count", "—")))
        self.build_summary["mappings"].setText(str(len(self.project.animation_mappings)))
        self.build_summary["registrations"].setText(str(len(self.project.custom_animation_registrations)))
        self.build_summary["textures"].setText(str(len(self.project.material_assignments)))
        appearance_label = str(self.project.appearance_settings.get("label") or "").strip()
        self.build_summary["appearance"].setText(
            appearance_label
            or f"{self.project.resource_name or 'Custom creature'} — merge-safe row resolved at install"
        )
        utc_name = str(self.project.utc_settings.get("display_name") or "").strip()
        utc_resref = str(self.project.utc_settings.get("resref") or self.project.resource_name or "custom_creature")
        template_resref = str(self.project.behavior_profile.get("template_resref") or "").strip()
        self.build_summary["utc"].setText(
            utc_name
            or f"{utc_resref}.utc"
            + (f" cloned from {template_resref}.utc" if template_resref else "")
        )
        self.build_summary["destination"].setText(self.project.build_destination or "—")

    def new_project(self) -> None:
        if self.project_path and not self._confirm_discard_or_save():
            return
        self.project = CustomRiggedCharacterProject()
        self.project_path = None
        self._load_project_into_form()
        self.project_status.setText("New project — source files stay read-only")

    def open_project(self) -> None:
        value, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open custom character project", "", "Ghost Studio character (*.ghostcharacter.json);;JSON (*.json)"
        )
        if not value:
            return
        try:
            project = load_custom_rigged_character_project(value)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Could not open project", str(exc))
            return
        self.project = project
        self.project_path = Path(value)
        self._load_project_into_form()
        self.project_status.setText(f"Project: {self.project_path.name}")

    def save_project(self, *, save_as: bool = False) -> bool:
        self._sync_project_from_form()
        target = self.project_path
        if save_as or not target:
            initial = self.project.output_project_folder or ""
            suggested = f"{self.project.resource_name or 'custom_character'}.ghostcharacter.json"
            value, _selected = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save custom character project", str(Path(initial) / suggested),
                "Ghost Studio character (*.ghostcharacter.json)"
            )
            if not value:
                return False
            target = Path(value)
        try:
            allow_replace = False
            if target.exists():
                try:
                    different = load_custom_rigged_character_project(target).project_id != self.project.project_id
                except Exception:
                    different = True
                if different:
                    choice = QtWidgets.QMessageBox.question(
                        self,
                        "Replace a different project?",
                        f"{target.name} belongs to another project. Replace it only if you intentionally want to overwrite that file.",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                        QtWidgets.QMessageBox.No,
                    )
                    if choice != QtWidgets.QMessageBox.Yes:
                        return False
                    allow_replace = True
            save_custom_rigged_character_project(self.project, target, allow_replace_different_project=allow_replace)
        except FileExistsError as exc:
            QtWidgets.QMessageBox.warning(self, "Project not overwritten", str(exc))
            return False
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Could not save project", str(exc))
            return False
        self.project_path = target
        self.project_status.setText(f"Saved: {target.name}")
        return True

    def _save_automatically(self) -> None:
        if self.project_path:
            self.save_project()
            return
        folder = self.project.output_project_folder
        if not folder or not self.project.resource_name:
            return
        target = Path(folder) / f"{self.project.resource_name}.ghostcharacter.json"
        if target.exists():
            try:
                existing = load_custom_rigged_character_project(target)
            except Exception:
                return
            if existing.project_id != self.project.project_id:
                return
        self.project_path = target
        self.save_project()

    def _confirm_discard_or_save(self) -> bool:
        choice = QtWidgets.QMessageBox.question(
            self,
            "Save this project?",
            "Save the current custom character project before starting another one?",
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Save,
        )
        if choice == QtWidgets.QMessageBox.Cancel:
            return False
        if choice == QtWidgets.QMessageBox.Save:
            return self.save_project()
        return True

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if any(url.toLocalFile().lower().endswith((".fbx", ".ghostcharacter.json")) for url in urls):
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # noqa: N802
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        project_files = [path for path in paths if path.name.lower().endswith(".ghostcharacter.json")]
        if project_files:
            try:
                self.project = load_custom_rigged_character_project(project_files[0])
                self.project_path = project_files[0]
                self._load_project_into_form()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Could not open dropped project", str(exc))
            return
        fbx_files = [path for path in paths if path.suffix.lower() == ".fbx"]
        if fbx_files:
            if not self.source_fbx.path():
                self.source_fbx.set_path(str(fbx_files[0]))
                fbx_files = fbx_files[1:]
            self.project.external_animation_assets.extend(
                SourceAsset(str(path), "", "external_animation", False) for path in fbx_files
            )
            self._form_changed()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self._save_automatically()
        event.accept()

    def apply_ghost_theme(self, theme: object) -> None:
        color = getattr(theme, "color", None)
        if callable(color):
            selector = "QMainWindow#customRiggedCharacterBuilderWindow"
            self.setStyleSheet(f"""
                {selector} QTableWidget {{
                    background: {color("table.background")};
                    color: {color("table.text")};
                    gridline-color: {color("table.grid")};
                    alternate-background-color: {color("panel.alternate", color("table.background"))};
                }}
                {selector} QTableWidget::item:selected {{
                    background: {color("selection.background")};
                    color: {color("selection.text")};
                }}
                {selector} QTableWidget QComboBox {{
                    background: {color("input.background")};
                    color: {color("input.text")};
                    border: 1px solid {color("input.border", color("panel.border"))};
                }}
                {selector} QHeaderView::section {{
                    background: {color("table.headerBackground")};
                    color: {color("table.headerText")};
                    border: 1px solid {color("table.grid")};
                }}
            """)
        self.update()

    def apply_ghost_layout(self, _layout: object) -> None:
        pass


__all__ = [
    "RUNTIME_TEST_CHECKLIST",
    "WORKFLOW_PAGES",
    "QtCustomRiggedCharacterBuilderWindow",
]
