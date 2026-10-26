"""Body Attachment System toolbox."""

from __future__ import annotations

from typing import Any

from PySide6 import QtCore, QtWidgets

from src.gui.qt_lib.assets.qt_theme import heading
from src.systems.bas.attachment_catalog import (
    BasAttachmentCatalog,
    saber_color_variants,
    saber_family,
)
from src.systems.bas.head_resolution import normalize_bas_model_resref


BAS_SLOT_LABELS: dict[str, str] = {
    "head": "HEAD",
    "mask": "MASK",
    "goggles": "GOGGLES",
    "body": "BODY",
    "belt": "BELT",
    "left_hand": "L. HAND",
    "right_hand": "R. HAND",
    "left_weapon": "L. Weapon",
    "right_weapon": "R. Wep",
}

BAS_SLOT_POSITIONS: dict[str, tuple[int, int]] = {
    "head": (1, 1),
    "mask": (1, 0),
    "goggles": (1, 2),
    "left_hand": (2, 0),
    "body": (2, 1),
    "right_hand": (2, 2),
    "left_weapon": (3, 0),
    "belt": (3, 1),
    "right_weapon": (3, 2),
}

BAS_PRESET_MODELS: dict[str, tuple[tuple[str, str], ...]] = {
    "head": (
        ("Player Male Head A 01", "pmha01"),
        ("Player Male Head C 01", "pmhc01"),
        ("Player Female Head A 01", "pfha01"),
        ("Player Female Head C 01", "pfhc01"),
    ),
    "left_weapon": (
        ("Vibroblade", "w_vbroswrd_001"),
        ("Short Sword", "w_vbroshort_001"),
        ("Lightsaber", "w_lghtsbr_001"),
    ),
    "right_weapon": (
        ("Blaster Pistol", "w_blstrpstl_001"),
        ("Blaster Rifle", "w_blstrrfl_001"),
        ("Lightsaber", "w_lghtsbr_001"),
    ),
    "mask": (
        ("Infragoggles", "i_mask_001"),
        ("Motion Goggles", "i_mask_002"),
    ),
    "goggles": (
        ("Infragoggles", "i_mask_001"),
        ("Motion Goggles", "i_mask_002"),
    ),
    "belt": (
        ("Cardio Power System", "i_belt_001"),
        ("Stealth Field Generator", "i_belt_010"),
    ),
}

BAS_MODE_LABELS: dict[str, str] = {
    "headless_body": "Headless Body",
    "full_body": "Full Body",
    "character_preset": "Character Preset",
}

BAS_CATALOG_GAME_ROLE = int(QtCore.Qt.ItemDataRole.UserRole) + 1


class _BasCharacterCatalogSignals(QtCore.QObject):
    ready = QtCore.Signal(object, object)
    failed = QtCore.Signal(str, object)


class _BasCharacterCatalogJob(QtCore.QRunnable):
    """Resolve UTC presets away from the Qt UI thread."""

    def __init__(self, manager: Any, revision: int | None) -> None:
        super().__init__()
        self.manager = manager
        self.revision = revision
        self.signals = _BasCharacterCatalogSignals()

    @QtCore.Slot()
    def run(self) -> None:
        try:
            from src.systems.bas.character_presets import build_bas_character_preset_catalog

            catalog = build_bas_character_preset_catalog(self.manager)
        except Exception as exc:  # pragma: no cover - installation-specific failure
            self.signals.failed.emit(str(exc), self.revision)
            return
        self.signals.ready.emit(catalog, self.revision)


class QtBodyAttachmentPanel(QtWidgets.QWidget):
    """Attach heads, hands, and weapons to the active body preview."""

    attachRequested = QtCore.Signal(str, str)
    clearRequested = QtCore.Signal(str)
    saveBuildRequested = QtCore.Signal()
    exportComposedRequested = QtCore.Signal()
    slotSelected = QtCore.Signal(str)
    modeChanged = QtCore.Signal(str)
    catalogRefreshRequested = QtCore.Signal()
    characterPresetRequested = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._slot_buttons: dict[str, QtWidgets.QToolButton] = {}
        self._slot_models: dict[str, str] = {}
        self._slot_model_games: dict[str, str] = {}
        self._selected_slot = "head"
        self._mode = "headless_body"
        self._composition_mode = "headless_body"
        self._syncing_layer_selection = False
        self._catalog: BasAttachmentCatalog | None = None
        self._catalog_revision: int | None = None
        self._character_catalog: Any = None
        self._character_catalog_revision: int | None = None
        self._character_catalog_loading_revision: int | None = None
        self._character_catalog_job: _BasCharacterCatalogJob | None = None
        self._syncing_color_combo = False
        self._build()
        self.set_selected_slot("head")

    def showEvent(self, event) -> None:  # noqa: N802 - Qt virtual
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self.catalogRefreshRequested.emit)

    def _build(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.addWidget(heading("Body Attachment System"))

        mode_form = QtWidgets.QFormLayout()
        self.mode_combo = QtWidgets.QComboBox()
        for key, label in BAS_MODE_LABELS.items():
            self.mode_combo.addItem(label, key)
        self.mode_combo.currentIndexChanged.connect(self._handle_mode_changed)
        mode_form.addRow("Mode", self.mode_combo)
        root.addLayout(mode_form)

        self.character_preset_group = QtWidgets.QGroupBox("Canonical Character")
        preset_layout = QtWidgets.QVBoxLayout(self.character_preset_group)
        preset_form = QtWidgets.QFormLayout()
        self.character_game_combo = QtWidgets.QComboBox(self.character_preset_group)
        self.character_game_combo.addItem("Both games", "")
        self.character_game_combo.addItem("KOTOR 1", "K1")
        self.character_game_combo.addItem("KOTOR 2", "K2")
        self.character_game_combo.currentIndexChanged.connect(self._populate_character_combo)
        preset_form.addRow("Game", self.character_game_combo)
        self.character_combo = QtWidgets.QComboBox(self.character_preset_group)
        self.character_combo.setObjectName("basCharacterPresetComboBox")
        self.character_combo.setEditable(True)
        self.character_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.character_combo.setToolTip(
            "Search every effective UTC character template from both configured games."
        )
        self.character_combo.currentIndexChanged.connect(self._sync_character_preset_details)
        completer = self.character_combo.completer()
        completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        completer.setFilterMode(QtCore.Qt.MatchContains)
        preset_form.addRow("Character", self.character_combo)
        preset_layout.addLayout(preset_form)
        self.character_preset_details = QtWidgets.QLabel(self.character_preset_group)
        self.character_preset_details.setObjectName("basCharacterPresetDetails")
        self.character_preset_details.setWordWrap(True)
        self.character_preset_details.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        preset_layout.addWidget(self.character_preset_details)
        self.apply_character_preset_button = QtWidgets.QPushButton(
            "Load Canonical Character",
            self.character_preset_group,
        )
        self.apply_character_preset_button.setObjectName("basApplyCharacterPresetButton")
        self.apply_character_preset_button.clicked.connect(self._emit_character_preset)
        preset_layout.addWidget(self.apply_character_preset_button)
        root.addWidget(self.character_preset_group)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        for row in range(5):
            for col in range(3):
                slot = next((key for key, pos in BAS_SLOT_POSITIONS.items() if pos == (row, col)), "")
                if slot:
                    button = QtWidgets.QToolButton()
                    button.setText(BAS_SLOT_LABELS[slot])
                    button.setCheckable(True)
                    button.setMinimumSize(88, 34)
                    button.clicked.connect(lambda _checked=False, key=slot: self.set_selected_slot(key))
                    self._slot_buttons[slot] = button
                    grid.addWidget(button, row, col)
                else:
                    filler = QtWidgets.QLabel("")
                    filler.setMinimumSize(88, 34)
                    grid.addWidget(filler, row, col)
        root.addLayout(grid)

        form = QtWidgets.QFormLayout()
        self.slot_label = QtWidgets.QLabel("")
        form.addRow("Slot", self.slot_label)
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.currentIndexChanged.connect(self._sync_color_combo)
        form.addRow("Model", self.model_combo)
        self.color_combo = QtWidgets.QComboBox()
        self.color_combo.setObjectName("basLightsaberColorComboBox")
        self.color_combo.setToolTip(
            "Blade colors for the selected lightsaber family from both installed games. "
            "Changing the color re-attaches the matching model variation."
        )
        self.color_combo.setEnabled(False)
        self.color_combo.activated.connect(self._handle_color_selected)
        form.addRow("Color", self.color_combo)
        root.addLayout(form)

        controls = QtWidgets.QHBoxLayout()
        self.attach_button = QtWidgets.QPushButton("Attach")
        self.attach_button.clicked.connect(self._emit_attach)
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.clear_button.clicked.connect(lambda: self.clearRequested.emit(self._selected_slot))
        controls.addWidget(self.attach_button)
        controls.addWidget(self.clear_button)
        root.addLayout(controls)

        self.save_build_button = QtWidgets.QPushButton("Save Build")
        self.save_build_button.clicked.connect(self.saveBuildRequested.emit)
        root.addWidget(self.save_build_button)

        self.export_composed_button = QtWidgets.QPushButton("Export Composed Model…")
        self.export_composed_button.setObjectName("basExportComposedButton")
        self.export_composed_button.setToolTip(
            "Export the body with every attached layer (head, mask, weapons, belt) "
            "combined into one model: binary MDL/MDX for the game, or OBJ/FBX for DCC tools. "
            "The export keeps the composed geometry, hierarchy, skins, materials, and textures; "
            "animation takes are those already baked or owned by the composed body model."
        )
        self.export_composed_button.clicked.connect(self.exportComposedRequested.emit)
        root.addWidget(self.export_composed_button)

        self.layer_tree = QtWidgets.QTreeWidget()
        self.layer_tree.setHeaderLabels(["Layer", "Model", "State"])
        self.layer_tree.setRootIsDecorated(False)
        self.layer_tree.setAlternatingRowColors(True)
        self.layer_tree.setMaximumHeight(150)
        self.layer_tree.itemSelectionChanged.connect(self._handle_layer_selection_changed)
        root.addWidget(self.layer_tree)

        self.status = QtWidgets.QPlainTextEdit()
        self.status.setReadOnly(True)
        self.status.setMaximumHeight(72)
        root.addWidget(self.status)
        root.addStretch(1)
        self._refresh_layers()
        self._sync_character_preset_visibility()

    def selected_slot(self) -> str:
        return self._selected_slot

    def selected_mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        mode = mode if mode in BAS_MODE_LABELS else "headless_body"
        if self._mode == mode:
            return
        self._mode = mode
        if mode in {"headless_body", "full_body"}:
            self._composition_mode = mode
        index = self.mode_combo.findData(mode)
        if index >= 0 and self.mode_combo.currentIndex() != index:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(index)
            self.mode_combo.blockSignals(False)
        if self._composition_mode == "full_body" and self._selected_slot == "head":
            self.set_selected_slot("mask")
        else:
            self.set_selected_slot(self._selected_slot)
        self._sync_character_preset_visibility()
        self._refresh_layers()
        self.modeChanged.emit(mode)
        if mode == "character_preset":
            QtCore.QTimer.singleShot(0, self.catalogRefreshRequested.emit)

    def set_composition_mode(self, mode: str) -> None:
        """Set the selected preset's real body mode without leaving preset UI."""

        self._composition_mode = mode if mode in {"headless_body", "full_body"} else "headless_body"
        if self._composition_mode == "full_body" and self._selected_slot == "head":
            self.set_selected_slot("mask")
        else:
            self.set_selected_slot(self._selected_slot)
        self._refresh_layers()

    def selected_model_resref(self) -> str:
        text = str(self.model_combo.currentText() or "").strip()
        if text and text != str(self.model_combo.itemText(self.model_combo.currentIndex()) or "").strip():
            return normalize_bas_model_resref(text)
        return normalize_bas_model_resref(self.model_combo.currentData() or text or "")

    def selected_model_game(self) -> str:
        """Return the explicit K1/K2 owner of the selected catalog row."""

        game = str(self.model_combo.currentData(BAS_CATALOG_GAME_ROLE) or "").strip().upper()
        return game if game in {"K1", "K2"} else ""

    def set_selected_slot(self, slot: str) -> None:
        slot = slot if slot in BAS_SLOT_LABELS else "head"
        self._selected_slot = slot
        for key, button in self._slot_buttons.items():
            button.setChecked(key == slot)
        self.slot_label.setText(BAS_SLOT_LABELS[slot])
        self._populate_model_combo(slot)
        attachable = slot not in {"left_hand", "right_hand"} and not (
            slot == "head" and self._composition_mode == "full_body"
        )
        self.model_combo.setEnabled(attachable)
        self.attach_button.setEnabled(attachable)
        self.clear_button.setEnabled(slot != "body" and attachable)
        self.attach_button.setText("Use Body" if slot == "body" else "Attach")
        self._sync_color_combo()
        self._sync_selected_layer()
        self.slotSelected.emit(slot)

    def set_body_model(self, model: Any, *, resref: str = "", game: str = "") -> None:
        name = str(
            resref
            or getattr(model, "_gr_source_resref", "")
            or getattr(model, "name", "")
            or "BODY"
        ) if model is not None else "BODY"
        source_game = str(game or getattr(model, "_gr_source_game", "") or "").strip().upper()
        self._slot_models["body"] = name
        self._slot_model_games["body"] = source_game if source_game in {"K1", "K2"} else ""
        self._set_slot_text("body", name)

    def set_slot_model(self, slot: str, model: Any = None, *, resref: str = "") -> None:
        label = str(resref or getattr(model, "name", "") or BAS_SLOT_LABELS.get(slot, slot))
        self._slot_models[slot] = label
        source_game = str(getattr(model, "_gr_source_game", "") or "").strip().upper()
        self._slot_model_games[slot] = source_game if source_game in {"K1", "K2"} else ""
        self._set_slot_text(slot, label)

    def clear_slot_model(self, slot: str) -> None:
        self._slot_models.pop(slot, None)
        self._slot_model_games.pop(slot, None)
        self._set_slot_text(slot, BAS_SLOT_LABELS.get(slot, slot))

    def set_status(self, message: str) -> None:
        self.status.setPlainText(str(message or ""))

    def set_attachment_catalog(
        self,
        catalog: BasAttachmentCatalog | None,
        *,
        revision: int | None = None,
    ) -> None:
        """Adopt the game-derived item catalog and refresh the active slot."""

        self._catalog = catalog if catalog is not None and not getattr(catalog, "empty", True) else None
        self._catalog_revision = int(revision) if revision is not None else None
        self._populate_model_combo(self._selected_slot)
        self._sync_color_combo()

    def attachment_catalog(self) -> BasAttachmentCatalog | None:
        return self._catalog

    def attachment_catalog_revision(self) -> int | None:
        return self._catalog_revision

    def character_preset_catalog(self):
        return self._character_catalog

    def character_preset_catalog_revision(self) -> int | None:
        return self._character_catalog_revision

    def load_character_preset_catalog(self, manager: Any, *, revision: int | None = None) -> None:
        """Start one background UTC/UTI preset scan for the active resource revision."""

        normalized_revision = int(revision) if revision is not None else None
        if self._character_catalog is not None and (
            normalized_revision is None or self._character_catalog_revision == normalized_revision
        ):
            return
        if self._character_catalog_job is not None and self._character_catalog_loading_revision == normalized_revision:
            return
        self._character_catalog_loading_revision = normalized_revision
        self.character_combo.setEnabled(False)
        self.apply_character_preset_button.setEnabled(False)
        self.character_preset_details.setText("Reading canonical UTC characters and equipment…")
        job = _BasCharacterCatalogJob(manager, normalized_revision)
        job.signals.ready.connect(self._handle_character_catalog_ready)
        job.signals.failed.connect(self._handle_character_catalog_failed)
        self._character_catalog_job = job
        QtCore.QThreadPool.globalInstance().start(job)

    @QtCore.Slot(object, object)
    def _handle_character_catalog_ready(self, catalog: Any, revision: Any) -> None:
        if self._character_catalog_loading_revision is not None and revision != self._character_catalog_loading_revision:
            return
        self._character_catalog_job = None
        self._character_catalog_loading_revision = None
        self._character_catalog = catalog if catalog is not None and not getattr(catalog, "empty", True) else None
        self._character_catalog_revision = int(revision) if revision is not None else None
        self._populate_character_combo()
        if self._character_catalog is None:
            errors = tuple(getattr(catalog, "errors", ()) or ())
            self.character_preset_details.setText(
                errors[0] if errors else "No installed UTC character presets were found."
            )

    @QtCore.Slot(str, object)
    def _handle_character_catalog_failed(self, message: str, revision: Any) -> None:
        if self._character_catalog_loading_revision is not None and revision != self._character_catalog_loading_revision:
            return
        self._character_catalog_job = None
        self._character_catalog_loading_revision = None
        self.character_combo.setEnabled(False)
        self.apply_character_preset_button.setEnabled(False)
        self.character_preset_details.setText(f"Character preset scan failed: {message}")

    def _populate_character_combo(self, *_args) -> None:
        current = self.selected_character_preset()
        current_key = str(getattr(current, "key", "") or "")
        game = str(self.character_game_combo.currentData() or "").upper()
        self.character_combo.blockSignals(True)
        self.character_combo.clear()
        presets = self._character_catalog.for_game(game) if self._character_catalog is not None else ()
        selected_index = -1
        for preset in presets:
            self.character_combo.addItem(preset.label, preset)
            if preset.key == current_key:
                selected_index = self.character_combo.count() - 1
        if selected_index >= 0:
            self.character_combo.setCurrentIndex(selected_index)
        self.character_combo.blockSignals(False)
        self.character_combo.setEnabled(bool(presets))
        self._sync_character_preset_details()

    def selected_character_preset(self):
        index = self.character_combo.currentIndex()
        if index < 0 or self.character_combo.currentText().strip() != self.character_combo.itemText(index).strip():
            return None
        value = self.character_combo.currentData()
        return value if value is not None and hasattr(value, "utc_resref") else None

    def _sync_character_preset_details(self, *_args) -> None:
        preset = self.selected_character_preset()
        if preset is None:
            if self._character_catalog is not None:
                self.character_preset_details.setText("Choose a character from the installed game templates.")
            self.apply_character_preset_button.setEnabled(False)
            return
        details = str(getattr(preset, "summary", "") or "")
        warnings = tuple(getattr(preset, "warnings", ()) or ())
        if warnings:
            details += f"\nWarnings: {'; '.join(warnings[:3])}"
        self.character_preset_details.setText(details)
        self.apply_character_preset_button.setEnabled(bool(getattr(preset, "loadable", False)))

    def _emit_character_preset(self) -> None:
        preset = self.selected_character_preset()
        if preset is not None and bool(getattr(preset, "loadable", False)):
            self.characterPresetRequested.emit(preset)

    def _sync_character_preset_visibility(self) -> None:
        self.character_preset_group.setVisible(self._mode == "character_preset")

    def _populate_model_combo(self, slot: str) -> None:
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        entries = self._catalog.entries(slot) if self._catalog is not None else ()
        if entries:
            for entry in entries:
                games = f" [{entry.games_label}]" if entry.games_label else ""
                self.model_combo.addItem(f"{entry.label}{games} - {entry.resref}", entry.resref)
                self.model_combo.setItemData(
                    self.model_combo.count() - 1,
                    str(getattr(entry, "game", "") or "").upper(),
                    BAS_CATALOG_GAME_ROLE,
                )
        else:
            for label, value in BAS_PRESET_MODELS.get(slot, ()):
                self.model_combo.addItem(f"{label} - {value}", value)
        if self.model_combo.count() == 0:
            self.model_combo.addItem("", "")
        selected_resref = normalize_bas_model_resref(self._slot_models.get(slot, ""))
        selected_game = str(self._slot_model_games.get(slot, "") or "").upper()
        if selected_resref:
            for index in range(self.model_combo.count()):
                if normalize_bas_model_resref(self.model_combo.itemData(index)) != selected_resref:
                    continue
                item_game = str(self.model_combo.itemData(index, BAS_CATALOG_GAME_ROLE) or "").upper()
                if selected_game and item_game and item_game != selected_game:
                    continue
                self.model_combo.setCurrentIndex(index)
                break
        self.model_combo.blockSignals(False)
        self._sync_color_combo()

    def _sync_color_combo(self) -> None:
        """Mirror the selected model's saber family into the Color choices."""

        combo = getattr(self, "color_combo", None)
        if combo is None or self._syncing_color_combo:
            return
        self._syncing_color_combo = True
        try:
            resref = self.selected_model_resref()
            family = saber_family(resref)
            combo.blockSignals(True)
            combo.clear()
            if not family:
                combo.setEnabled(False)
                return
            current_index = 0
            for index, variant in enumerate(saber_color_variants(resref, self._catalog, self._selected_slot)):
                games = f" [{variant.games_label}]" if variant.games_label else ""
                combo.addItem(f"{variant.color}{games}", variant.resref)
                if variant.resref == resref:
                    current_index = index
            combo.setCurrentIndex(current_index)
            combo.setEnabled(combo.count() > 0 and self.model_combo.isEnabled())
        finally:
            combo.blockSignals(False)
            self._syncing_color_combo = False

    def _handle_color_selected(self, index: int) -> None:
        """Swap the model to the chosen blade color, re-attaching if live."""

        variant = str(self.color_combo.itemData(index) or "").strip()
        if not variant:
            return
        previous = self.selected_model_resref()
        match = self.model_combo.findData(variant)
        self.model_combo.blockSignals(True)
        if match >= 0:
            self.model_combo.setCurrentIndex(match)
        else:
            self.model_combo.setEditText(variant)
        self.model_combo.blockSignals(False)
        slot = self._selected_slot
        attached = str(self._slot_models.get(slot, "") or "").strip()
        if attached and saber_family(attached) and saber_family(attached) == saber_family(previous or variant):
            self.attachRequested.emit(slot, variant)

    def _emit_attach(self) -> None:
        if self._selected_slot in {"left_hand", "right_hand"} or (
            self._selected_slot == "head" and self._composition_mode == "full_body"
        ):
            return
        self.attachRequested.emit(self._selected_slot, self.selected_model_resref())

    def _set_slot_text(self, slot: str, text: str) -> None:
        button = self._slot_buttons.get(slot)
        if button is None:
            return
        base = BAS_SLOT_LABELS.get(slot, slot)
        value = str(text or "").strip()
        button.setText(base if not value or value == base else f"{base}\n{value}")
        self._refresh_layers()

    def _refresh_layers(self) -> None:
        tree = getattr(self, "layer_tree", None)
        if tree is None:
            return
        order = ("body", "head", "mask", "goggles", "left_hand", "right_hand", "left_weapon", "belt", "right_weapon")
        tree.blockSignals(True)
        tree.clear()
        for slot in order:
            model = str(self._slot_models.get(slot, "") or "").strip()
            if slot == "body":
                model = model or "BODY"
                state = "Base"
            elif slot in {"left_hand", "right_hand"} or (
                slot == "head" and self._composition_mode == "full_body"
            ):
                state = "Socket"
            else:
                state = "Attached" if model else "Empty"
            item = QtWidgets.QTreeWidgetItem([BAS_SLOT_LABELS.get(slot, slot), model or "", state])
            item.setData(0, QtCore.Qt.UserRole, slot)
            if slot == "body":
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            tree.addTopLevelItem(item)
        tree.resizeColumnToContents(0)
        tree.resizeColumnToContents(1)
        tree.blockSignals(False)
        self._sync_selected_layer()

    def _sync_selected_layer(self) -> None:
        tree = getattr(self, "layer_tree", None)
        if tree is None:
            return
        self._syncing_layer_selection = True
        try:
            for index in range(tree.topLevelItemCount()):
                item = tree.topLevelItem(index)
                if str(item.data(0, QtCore.Qt.UserRole) or "") == self._selected_slot:
                    tree.setCurrentItem(item)
                    break
        finally:
            self._syncing_layer_selection = False

    def _handle_layer_selection_changed(self) -> None:
        if self._syncing_layer_selection:
            return
        item = self.layer_tree.currentItem()
        slot = str(item.data(0, QtCore.Qt.UserRole) or "") if item is not None else ""
        if slot:
            self.set_selected_slot(slot)

    def _handle_mode_changed(self) -> None:
        self.set_mode(str(self.mode_combo.currentData() or "headless_body"))

    def layer_rows(self) -> list[tuple[str, str, str]]:
        rows: list[tuple[str, str, str]] = []
        for index in range(self.layer_tree.topLevelItemCount()):
            item = self.layer_tree.topLevelItem(index)
            rows.append((item.text(0), item.text(1), item.text(2)))
        return rows
