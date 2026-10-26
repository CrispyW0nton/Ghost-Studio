"""Shared beginner editor for cloth, hair, and flexible attachments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Optional

from PySide6 import QtCore, QtWidgets

from src.core.project.secondary_motion_profile import SecondaryMotionProfile, default_secondary_motion_profile
from src.core.secondary_motion.workflow import SECONDARY_MOTION_PRESETS, runtime_capability_descriptor
from src.core.validation.secondary_motion_validator import validate_secondary_motion_profile


_PRESET_LABELS = (
    ("Robe / Skirt", "robe_skirt"),
    ("Sleeve", "sleeve"),
    ("Cape", "cape"),
    ("Ponytail / Braid", "ponytail_braid"),
    ("Hair Lock", "hair_lock"),
    ("Banner", "banner"),
    ("Cable / Hose", "cable_hose"),
    ("Native Dangly", "native_dangly"),
    ("Rigid", "rigid"),
)


class QtSecondaryMotionEditor(QtWidgets.QWidget):
    """One reusable editor contract embedded by all character/placeable tools."""

    profileChanged = QtCore.Signal(object)
    previewRequested = QtCore.Signal(object)
    resetPreviewRequested = QtCore.Signal()
    validateRequested = QtCore.Signal(object)

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        target_kind: str = "character",
        target_game: str = "K2",
    ) -> None:
        super().__init__(parent)
        self.setObjectName("secondaryMotionEditor")
        self._target_kind = str(target_kind or "character")
        self._profile = default_secondary_motion_profile(target_game=target_game)
        self._build_ui()
        self.set_profile(self._profile)

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        heading = QtWidgets.QLabel("Secondary Motion — Cloth & Hair")
        heading.setObjectName("secondaryMotionHeading")
        heading.setStyleSheet("font-weight: 700;")
        root.addWidget(heading)
        help_label = QtWidgets.QLabel(
            "Choose what should move, select a safe preset, attach it to a named bone or anchor, "
            "then preview and validate. Source files and unrelated weights stay unchanged."
        )
        help_label.setWordWrap(True)
        root.addWidget(help_label)

        form = QtWidgets.QFormLayout()
        self.region_edit = QtWidgets.QLineEdit()
        self.region_edit.setObjectName("secondaryMotionRegionLineEdit")
        self.region_edit.setPlaceholderText("Mesh or vertex group")
        form.addRow("1. What should move", self.region_edit)
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.setObjectName("secondaryMotionPresetCombo")
        for label, key in _PRESET_LABELS:
            self.preset_combo.addItem(label, key)
        form.addRow("2. Preset", self.preset_combo)
        self.anchor_edit = QtWidgets.QLineEdit()
        self.anchor_edit.setObjectName("secondaryMotionAnchorLineEdit")
        self.anchor_edit.setPlaceholderText("Named root bone or placeable anchor")
        form.addRow("3. Fixed attachment", self.anchor_edit)
        self.driver_edit = QtWidgets.QLineEdit()
        self.driver_edit.setObjectName("secondaryMotionDriverLineEdit")
        self.driver_edit.setPlaceholderText("Existing driver bone; custom names are allowed")
        form.addRow("4. Driver chain", self.driver_edit)
        root.addLayout(form)

        actions = QtWidgets.QHBoxLayout()
        self.add_region_button = QtWidgets.QPushButton("Add / update region")
        self.add_region_button.setObjectName("secondaryMotionAddRegionButton")
        self.add_region_button.clicked.connect(self._add_region)
        actions.addWidget(self.add_region_button)
        self.preview_button = QtWidgets.QPushButton("Preview")
        self.preview_button.setObjectName("secondaryMotionPreviewButton")
        self.preview_button.clicked.connect(lambda: self.previewRequested.emit(self.profile()))
        actions.addWidget(self.preview_button)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self.resetPreviewRequested.emit)
        actions.addWidget(self.reset_button)
        self.validate_button = QtWidgets.QPushButton("Validate")
        self.validate_button.clicked.connect(self._validate)
        actions.addWidget(self.validate_button)
        root.addLayout(actions)

        self.test_motion_combo = QtWidgets.QComboBox()
        self.test_motion_combo.setObjectName("secondaryMotionTestMotionCombo")
        motions = ["Idle", "Walk", "Run", "Stop", "Turn", "Combat / raised arms", "Dialogue / head turn", "Teleport / reset", "Low FPS", "Repeated animation sampling"]
        if self._target_kind == "placeable":
            motions = ["Swing", "Rotation", "Module reload", "Object destruction / reload"]
        self.test_motion_combo.addItems(motions)
        root.addWidget(self.test_motion_combo)

        capabilities = runtime_capability_descriptor()
        advanced = QtWidgets.QGroupBox("Advanced runtime limits")
        advanced.setCheckable(True)
        advanced.setChecked(False)
        advanced_layout = QtWidgets.QVBoxLayout(advanced)
        limits = QtWidgets.QLabel(
            f"Fixed {capabilities['fixed_step_hz']} Hz preview; at most {capabilities['max_substeps']} catch-up steps, "
            f"{capabilities['max_driver_writes_per_actor']} driver writes, and "
            f"{capabilities['max_colliders_per_region']} colliders per region."
        )
        limits.setWordWrap(True)
        advanced_layout.addWidget(limits)
        root.addWidget(advanced)

        self.runtime_status = QtWidgets.QLabel()
        self.runtime_status.setObjectName("secondaryMotionRuntimeStatus")
        self.runtime_status.setWordWrap(True)
        root.addWidget(self.runtime_status)
        root.addStretch(1)

    def profile(self) -> dict[str, Any]:
        return deepcopy(self._profile)

    def set_profile(self, value: Mapping[str, Any] | None) -> None:
        self._profile = SecondaryMotionProfile.from_dict(
            value or default_secondary_motion_profile()
        ).to_dict()
        self._refresh_status()

    def set_target_game(self, value: str) -> None:
        self._profile["target_game"] = str(value or "K2").upper()
        self._refresh_status()

    def _add_region(self) -> None:
        mesh = self.region_edit.text().strip()
        preset_key = str(self.preset_combo.currentData() or "rigid")
        preset = deepcopy(SECONDARY_MOTION_PRESETS[preset_key])
        if not mesh:
            self.runtime_status.setText("Choose a mesh region or vertex group first.")
            return
        regions = list(self._profile.get("regions") or ())
        region_id = "".join(character if character.isalnum() or character == "_" else "_" for character in mesh.casefold()).strip("_")
        existing = next((index for index, row in enumerate(regions) if str(row.get("region_id")) == region_id), None)
        driver_name = self.driver_edit.text().strip()
        anchor_name = self.anchor_edit.text().strip()
        drivers = []
        if driver_name:
            drivers.append({
                "name": driver_name,
                "parent_name": anchor_name,
                "rest_position": [0.0, 0.0, 0.0],
                "rest_quaternion": [0.0, 0.0, 0.0, 1.0],
                "rest_scale": [1.0, 1.0, 1.0],
            })
        region = {
            "region_id": region_id or f"region_{len(regions) + 1}",
            "label": mesh,
            "semantic_type": preset.pop("semantic_type"),
            "backend": preset.pop("backend"),
            "mesh_membership": [mesh],
            "vertex_group": mesh,
            "drivers": drivers,
            "anchors": {"driver_names": [anchor_name] if anchor_name else [], "skin_weight_evidence": {}},
            "colliders": [],
            "solver": {"preset": preset_key, **preset},
        }
        if existing is None:
            regions.append(region)
        else:
            regions[existing] = region
        self._profile["regions"] = regions
        runtime = dict(self._profile.get("runtime") or {})
        runtime.update({"state": "capture_only", "write_enabled": False})
        self._profile["runtime"] = runtime
        self.profileChanged.emit(self.profile())
        self._refresh_status()

    def _validate(self) -> None:
        report = validate_secondary_motion_profile(self._profile, target_kind=self._target_kind)
        self.validateRequested.emit(self.profile())
        self._refresh_status(report)

    def _refresh_status(self, report=None) -> None:
        report = report or validate_secondary_motion_profile(self._profile, target_kind=self._target_kind)
        runtime = dict(self._profile.get("runtime") or {})
        if report.can_enable_runtime_writes:
            message = "Write-enabled for this exact model after accepted calibration and visible proof."
        elif self._target_kind == "placeable" and any(str(row.get("backend")) == "runtime_pbd" for row in self._profile.get("regions") or ()):
            message = "Needs runtime compatibility proof — placeable PBD remains capture-only."
        else:
            message = "Safe authoring mode — runtime PBD is capture-only until this exact model passes its read-only validation gates."
        if not report.can_export_profile:
            message += f" Fix {sum(issue.severity == 'blocking' for issue in report.issues)} blocking profile issue(s) before export."
        elif runtime.get("write_enabled"):
            message += " A requested write state was reduced to the validated gate result during compilation."
        self.runtime_status.setText(message)


__all__ = ["QtSecondaryMotionEditor"]
