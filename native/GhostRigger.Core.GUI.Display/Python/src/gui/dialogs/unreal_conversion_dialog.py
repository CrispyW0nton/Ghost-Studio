"""Native Qt conversion review: named components, explicit collision, materials."""
from __future__ import annotations
import copy
from pathlib import Path
from PySide6 import QtCore, QtWidgets


class UnrealConversionDialog(QtWidgets.QDialog):
    def __init__(self, snapshot, defaults=None, parent=None):
        super().__init__(parent)
        self.snapshot = snapshot
        self.defaults = copy.deepcopy(defaults or {})
        self.result_policy = None
        self.setObjectName("mapStudioUnrealConversionDialog")
        self.setWindowTitle("Review Unreal level")
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Review what will become your KOTOR II map", self)
        title.setWordWrap(True)
        layout.addWidget(title)
        tabs = QtWidgets.QTabWidget(self)
        layout.addWidget(tabs)
        components_page = QtWidgets.QWidget(self)
        components_layout = QtWidgets.QVBoxLayout(components_page)
        self.search = QtWidgets.QLineEdit(self)
        self.search.setPlaceholderText("Filter source objects…")
        self.search.textChanged.connect(self._filter)
        components_layout.addWidget(self.search)
        self.components = QtWidgets.QListWidget(self)
        self.rows = []
        for component in snapshot["components"]:
            if component["disposition"] == "editor_only":
                continue
            text = component["label"]
            if component["disposition"] != "static_geometry":
                text += " (not supported in this static map)"
            elif not component.get("visible", True):
                text += " (hidden in Unreal)"
            item = QtWidgets.QListWidgetItem(text, self.components)
            item.setToolTip(component["id"])
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            included = component["disposition"] == "static_geometry" and component.get("visible", True) and component["id"] not in self.defaults.get("excluded_components", {})
            item.setCheckState(QtCore.Qt.CheckState.Checked if included else QtCore.Qt.CheckState.Unchecked)
            self.rows.append((component, item))
        components_layout.addWidget(self.components)
        tabs.addTab(components_page, "Objects")
        placement_page = QtWidgets.QWidget(self)
        form = QtWidgets.QFormLayout(placement_page)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        module = self.defaults.get("module", {})
        self.module_id = QtWidgets.QLineEdit(module.get("resref", "ue_map"), self)
        self.display_name = QtWidgets.QLineEdit(module.get("name", snapshot["map_package"].rsplit("/", 1)[-1]), self)
        form.addRow("Module ID", self.module_id)
        form.addRow("Map name", self.display_name)
        self.collision = QtWidgets.QComboBox(self)
        self.collision.addItem("Choose a floor object…", None)
        for component, _ in self.rows:
            if component["disposition"] == "static_geometry":
                self.collision.addItem(component["label"], component["id"])
        self.section = QtWidgets.QSpinBox(self)
        self.collision.currentIndexChanged.connect(self._collision_changed)
        form.addRow("Walkable floor", self.collision)
        form.addRow("Floor mesh section", self.section)
        note = QtWidgets.QLabel("Select a connected floor section. Every triangle in it becomes walkable. Complex authored collision can be loaded with saved conversion settings.", self)
        note.setWordWrap(True)
        form.addRow(note)
        self.spawn = []
        values = module.get("entry_position_cm", [0, 0, 0]) + [module.get("entry_yaw_degrees", 0)]
        for label, value in zip(("Spawn X (cm)", "Spawn Y (cm)", "Spawn Z (cm)", "Facing (degrees)"), values):
            field = QtWidgets.QDoubleSpinBox(self)
            field.setRange(-10000000, 10000000)
            field.setDecimals(3)
            field.setValue(value)
            form.addRow(label, field)
            self.spawn.append(field)
        propose = QtWidgets.QPushButton("Place spawn on selected floor", self)
        propose.clicked.connect(self._propose_spawn)
        form.addRow(propose)
        tabs.addTab(placement_page, "Floor & spawn")
        materials_page = QtWidgets.QWidget(self)
        materials_layout = QtWidgets.QVBoxLayout(materials_page)
        material_note = QtWidgets.QLabel("Simple diffuse textures and colors convert directly. For other shaders, choose a baked TGA and its KOTOR blend mode. Unreal lighting and shader effects are not reproduced.", self)
        material_note.setWordWrap(True)
        materials_layout.addWidget(material_note)
        self.materials = QtWidgets.QTableWidget(len(snapshot["materials"]), 3, self)
        self.materials.setHorizontalHeaderLabels(["Material", "Diffuse image", "Blend"])
        self.materials.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.materials.horizontalHeader().setStretchLastSection(True)
        self.material_rows = {}
        for row, key in enumerate(sorted(snapshot["materials"])):
            item = QtWidgets.QTableWidgetItem(key.rsplit("/", 1)[-1].split(".")[0])
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            item.setToolTip(key)
            self.materials.setItem(row, 0, item)
            button = QtWidgets.QPushButton("Automatic / choose TGA…", self)
            button.clicked.connect(lambda checked=False, key=key: self._choose_texture(key))
            self.materials.setCellWidget(row, 1, button)
            blend = QtWidgets.QComboBox(self)
            for label, value in (("Opaque", "opaque"), ("Cutout", "masked"), ("Additive", "additive"), ("Alpha", "alpha")):
                blend.addItem(label, value)
            self.materials.setCellWidget(row, 2, blend)
            self.material_rows[key] = [button, blend, None]
        materials_layout.addWidget(self.materials)
        tabs.addTab(materials_page, "Materials")
        self.accept_loss = QtWidgets.QCheckBox("Accept diffuse conversion and omit the unchecked source objects.", self)
        layout.addWidget(self.accept_loss)
        self.error = QtWidgets.QLabel(self)
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel, self)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Review import")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        collision_id = self.defaults.get("collision", {}).get("component_id")
        if collision_id:
            self.collision.setCurrentIndex(max(0, self.collision.findData(collision_id)))

    def _filter(self, text):
        for component, item in self.rows:
            item.setHidden(text.casefold() not in component["label"].casefold())

    def _component(self):
        return next((c for c, _ in self.rows if c["id"] == self.collision.currentData()), None)

    def _collision_changed(self):
        component = self._component()
        count = len(self.snapshot["resolved_meshes"][component["mesh"]]["sections"]) if component else 1
        self.section.setRange(0, max(0, count - 1))

    def _propose_spawn(self):
        from src.math.unreal_native_surface import propose_section_spawn
        try:
            component = self._component()
            if component is None:
                raise ValueError("Choose a floor object first.")
            section = self.snapshot["resolved_meshes"][component["mesh"]]["sections"][self.section.value()]
            for field, value in zip(self.spawn, propose_section_spawn(section, component["world_matrix"])):
                field.setValue(value)
            self.error.clear()
        except Exception as error:
            self.error.setText(str(error))

    def _choose_texture(self, key):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose a baked diffuse texture", "", "Targa (*.tga)")
        if path:
            self.material_rows[key][2] = path
            self.material_rows[key][0].setText(Path(path).name)

    def _accept(self):
        from src.core.modules.unreal_level_import import RESREF
        if not self.accept_loss.isChecked():
            self.error.setText("Accept the conversion choices before continuing.")
            return
        if not RESREF.fullmatch(self.module_id.text().strip()):
            self.error.setText("Module ID must start with a lowercase letter and use up to 16 lowercase letters, digits or underscores.")
            return
        excluded = {c["id"]: "Creator excluded this object during conversion review." for c, item in self.rows if item.checkState() != QtCore.Qt.CheckState.Checked}
        if not self.collision.currentData() or self.collision.currentData() in excluded:
            self.error.setText("Choose an included floor object for the walkmesh.")
            return
        self.result_policy = {"schema": "ghoststudio.unreal-conversion-policy/v1", "snapshot_sha256": self.snapshot["snapshot_sha256"],
            "accept_diffuse_only": True, "module": {"resref": self.module_id.text().strip(), "name": self.display_name.text(),
                "entry_position_cm": [field.value() for field in self.spawn[:3]], "entry_yaw_degrees": self.spawn[3].value()},
            "collision": {"component_id": self.collision.currentData(), "recipe": "explicit-render-section-v1", "section": self.section.value(), "surface_id": 4},
            "excluded_components": excluded, "materials": {},
            "include_hidden_components": [c["id"] for c, item in self.rows if not c.get("visible", True) and item.checkState() == QtCore.Qt.CheckState.Checked]}
        self.accept()

    def write_policy(self, directory):
        from src.io.unreal_editor_connection import write_reviewed_policy
        choices = {key: (path, blend.currentData()) for key, (_, blend, path) in self.material_rows.items() if path}
        return write_reviewed_policy(directory, self.result_policy, self.snapshot, choices)
